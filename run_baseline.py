import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from argparse import Namespace
from eth_env import ETHTradingEnv

# ---------------------------------
# 1. 全局常量与配置
# ---------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 买入/卖出 动作
BUY   = 0.5
SELL  = -0.5
FULL_BUY  = 1
FULL_SELL = -1

# 技术指标中用到的 SMA 周期
SMA_PERIODS = [5, 10, 15, 20, 30]

# 回测时间段配置：验证集、测试集
DATES = ['2023-02-01', '2023-08-01', '2024-02-01']
VAL_START, VAL_END   = DATES[0], DATES[1]
TEST_START, TEST_END = DATES[1], DATES[2]


# ---------------------------------
# 2. 数据预处理与特征工程封装
# ---------------------------------
class FeatureEngineer:
    """
    对原始 ETH 日线数据，计算各项技术指标并将结果加回 DataFrame。
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        # 确保 date 列为 datetime 类型
        self.df['date'] = pd.to_datetime(self.df['snapped_at'])
        # 按日期排序（如数据未排序）
        self.df.sort_values('date', inplace=True)
        self.df.reset_index(drop=True, inplace=True)

    def add_sma_std(self, periods):
        """
        计算 SMA 与 STD 指标
        """
        for p in periods:
            self.df[f'SMA_{p}'] = self.df['open'].rolling(window=p).mean()
            self.df[f'STD_{p}'] = self.df['open'].rolling(window=p).std()

    def add_macd(self):
        """
        计算 EMA12, EMA26, MACD 以及 Signal Line
        """
        self.df['EMA_12'] = self.df['open'].ewm(span=12, adjust=False).mean()
        self.df['EMA_26'] = self.df['open'].ewm(span=26, adjust=False).mean()
        self.df['MACD']   = self.df['EMA_12'] - self.df['EMA_26']
        self.df['Signal_Line'] = self.df['MACD'].ewm(span=9, adjust=False).mean()

    def run_all(self):
        """
        一次性执行所有需要的特征计算
        """
        self.add_sma_std(SMA_PERIODS)
        self.add_macd()
        return self.df


# ---------------------------------
# 3. LSTM 网络封装与信号预测
# ---------------------------------
class LSTMModel(nn.Module):
    """
    简单的单层 LSTM + 全连接网络，用于未来价格预测
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim  = hidden_dim
        self.num_layers  = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        # 初始化 LSTM 的隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        # 取最后一个时刻的输出
        out = self.fc(out[:, -1, :])
        return out


class LSTMModelWrapper:
    """
    负责：
    1) 从给定 DataFrame 和时间窗口中，裁剪训练数据，做归一化和 DataLoader;
    2) 训练 LSTM 模型，保存训练好的模型；
    3) 对最后一个 look_back 序列进行预测，返回“涨/跌”信号。
    """
    def __init__(self, look_back=5, hidden_dim=100, num_layers=2, lr=1e-3, epochs=100, batch_size=64):
        self.look_back   = look_back
        self.hidden_dim  = hidden_dim
        self.num_layers  = num_layers
        self.lr          = lr
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.scaler      = MinMaxScaler(feature_range=(0, 1))
        self.model       = None  # 训练完成后保存模型对象

    def _create_dataset(self, arr: np.ndarray):
        """
        从一个 Nx1 的数组里，生成 (N - look_back, look_back, feature_dim) 的输入 X
        和 (N - look_back, 1) 的输出 Y。
        """
        X, Y = [], []
        for i in range(len(arr) - self.look_back):
            X.append(arr[i:(i + self.look_back), :])       # shape: (look_back, feature_dim)
            Y.append(arr[i + self.look_back, 0])           # 取开盘价作为回归目标
        X = np.array(X)  # (样本数, look_back, feature_dim)
        Y = np.array(Y)  # (样本数,)
        # 转成 tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)
        Y_tensor = torch.tensor(Y, dtype=torch.float32).view(-1, 1)
        return TensorDataset(X_tensor, Y_tensor)

    def train(self, df: pd.DataFrame, start_date: str, end_date: str):
        """
        1) 从 df 中裁剪 start_date 到 end_date 的连续数据；
        2) 提取 open 列，归一化；
        3) 构造 DataLoader,训练模型；
        4) 返回训练好的模型
        """
        # 1. 裁剪时间区间 & 提取 open 列
        data_slice = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
        prices     = data_slice['open'].values.reshape(-1, 1)  # N×1
        # 2. 归一化
        scaled = self.scaler.fit_transform(prices)             # N×1
        # 3. 构造 Dataset & DataLoader
        dataset = self._create_dataset(scaled)                 
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        # 4. 初始化模型、损失函数、优化器
        self.model = LSTMModel(input_dim=1,
                               hidden_dim=self.hidden_dim,
                               num_layers=self.num_layers,
                               output_dim=1).to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # 5. 训练循环
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for X_batch, Y_batch in loader:
                X_batch = X_batch.to(DEVICE)
                Y_batch = Y_batch.to(DEVICE)
                optimizer.zero_grad()
                out = self.model(X_batch)
                loss = criterion(out, Y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss = loss.item()
            if epoch % 10 == 0:
                print(f'[LSTM] Epoch {epoch}, Loss: {epoch_loss:.6f}')

        return self.model

    def predict_signal(self, df: pd.DataFrame, start_date: str, end_date: str):
        """
        用训练好的 self.model,预测 end_date 这一天的“下一日开盘价”，
        并与当前开盘价做比较，输出 'Buy' / 'Sell' / 'Hold'。
        """
        # 1. 同样裁剪数据、提取 open 列并归一化
        data_slice = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
        prices     = data_slice['open'].values.reshape(-1, 1)
        scaled     = self.scaler.transform(prices)  # 直接用训练时的 scaler 做 transform
        # 2. 取最后 look_back 条作为输入序列
        last_seq = scaled[-self.look_back:]  # shape: (look_back, 1)
        X_input  = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, look_back, 1)
        # 3. 前向预测
        self.model.eval()
        with torch.no_grad():
            pred_scaled = self.model(X_input).cpu().numpy()  # 形状 (1,1)
        # 4. 反缩放
        pred_price = self.scaler.inverse_transform(pred_scaled)[0][0]
        current_price = prices[-1][0]
        # 5. 信号判断
        if pred_price > current_price:
            return 'Buy'
        elif pred_price < current_price:
            return 'Sell'
        else:
            return 'Hold'


# ---------------------------------
# 4. 策略基类与各策略实现
# ---------------------------------
class BaseStrategy:
    """
    策略接口基类，只需实现 decide_action 方法：
    输入：
        - current_row: DataFrame 中当前行，包含 open、各种 SMA、MACD 等列
        - env_state: 来自回测环境的当前状态字典，包含 'open'、'cash'、'eth_held'、'net_worth' 等
    输出：
        - action: BUY / SELL / 0 / FULL_BUY / FULL_SELL 等数值
    """
    def decide_action(self, current_row: pd.Series, env_state: dict):
        raise NotImplementedError("所有策略都必须实现 decide_action 方法")


class SMAStrategy(BaseStrategy):
    def __init__(self, period):
        self.period = period
        self.sma_col = f'SMA_{period}'

    def decide_action(self, current_row: pd.Series, env_state: dict):
        price = env_state['open']
        cash  = env_state['cash']
        held  = env_state['eth_held']

        # 当 price > SMA 时发出“买入”信号，反之发出“卖出”
        if price > current_row[self.sma_col] and cash > 0:
            return BUY
        elif price < current_row[self.sma_col] and held > 0:
            return SELL
        else:
            return 0


class SLMAStrategy(BaseStrategy):
    def __init__(self, short: str, long: str):
        """
        short, long 均为 DataFrame 中两列的列名，例如 'SMA_5', 'SMA_20'。
        """
        self.short_col = short
        self.long_col  = long

    def decide_action(self, current_row: pd.Series, env_state: dict):
        price = env_state['open']
        held  = env_state['eth_held']

        # 如果短期 SMA 在长期 SMA 之上，就买；反之就卖
        if current_row[self.short_col] > current_row[self.long_col]:
            return BUY
        elif current_row[self.short_col] < current_row[self.long_col] and held > 0:
            return SELL
        else:
            return 0


class MACDStrategy(BaseStrategy):
    def decide_action(self, current_row: pd.Series, env_state: dict):
        price = env_state['open']
        cash  = env_state['cash']
        held  = env_state['eth_held']

        # 当 MACD 下穿 Signal Line，认为是买点；上穿则卖点
        if current_row['MACD'] < current_row['Signal_Line'] and cash > 0:
            return BUY
        elif current_row['MACD'] > current_row['Signal_Line'] and held > 0:
            return SELL
        else:
            return 0


class BollingerBandsStrategy(BaseStrategy):
    def __init__(self, period: int, multiplier: float):
        self.period     = period
        self.multiplier = multiplier
        self.sma_col    = f'SMA_{period}'
        self.std_col    = f'STD_{period}'

    def decide_action(self, current_row: pd.Series, env_state: dict):
        price = env_state['open']
        cash  = env_state['cash']
        held  = env_state['eth_held']

        sma = current_row[self.sma_col]
        sd  = current_row[self.std_col]
        upper = sma + self.multiplier * sd
        lower = sma - self.multiplier * sd

        if price < lower and cash > 0:
            return BUY
        elif price > upper and held > 0:
            return SELL
        else:
            return 0


class BuyAndHoldStrategy(BaseStrategy):
    def decide_action(self, current_row: pd.Series, env_state: dict):
        # 如果还没买过，就买满；否则一直持有
        cash = env_state['cash']
        if cash > 0:
            return FULL_BUY
        else:
            return 0


class OptimalStrategy(BaseStrategy):
    def __init__(self, full_df: pd.DataFrame):
        """
        需要整张 df 来参考“下一天的开盘价”，所以在初始化时把整个 DataFrame 传进来。
        """
        self.df = full_df.reset_index(drop=True)

    def decide_action(self, current_row: pd.Series, env_state: dict):
        idx   = current_row.name  # row 的索引
        if idx + 1 >= len(self.df):
            return 0
        next_open = self.df.loc[idx + 1, 'open']
        price     = env_state['open']

        if price < next_open:
            return FULL_BUY
        elif price > next_open:
            return FULL_SELL
        else:
            return 0


class LSTMStrategy(BaseStrategy):
    def __init__(self, wrapper: LSTMModelWrapper):
        self.wrapper = wrapper
        # 假设 wrapper 已经在初始化时被训练过

    def decide_action(self, current_row: pd.Series, env_state: dict):
        # wrapper.predict_signal 需要完整的 df 和时间窗口才能做出“Buy/Sell/Hold”
        # 由于回测是按行做遍历的，“current_row” 包含 date；我们可以每次都传入训练时指定的 start/end
        sig = self.wrapper.predict_signal(
            df=env_state['full_df'], 
            start_date=env_state['start_date'], 
            end_date=current_row['date'].strftime('%Y-%m-%d')
        )
        cash = env_state['cash']
        held = env_state['eth_held']
        if sig == 'Buy' and cash > 0:
            return BUY
        elif sig == 'Sell' and held > 0:
            return SELL
        else:
            return 0


# ---------------------------------
# 5. 回测框架：BacktestEngine
# ---------------------------------
class BacktestEngine:
    """
    统一的回测框架：遍历某个时间区间内的所有天，按策略决策动作，
    将动作输入 env.step()，并实时计算 IRR、夏普率等指标。
    """
    def __init__(self, df: pd.DataFrame, strategy: BaseStrategy, start_date: str, end_date: str):
        # 1. DataFrame 与时间区间
        self.df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy().reset_index(drop=True)
        self.strategy    = strategy
        # 2. 初始化交易环境
        self.env         = ETHTradingEnv(Namespace(starting_date=start_date, ending_date=end_date, dataset='eth'))
        # 3. 记录回测起始净值
        init_state = self.env.reset()
        # 兼容 env.reset() 返回 tuple 或 dict
        if isinstance(init_state, tuple):
            state = init_state[0]
        else:
            state = init_state
        self.start_worth = state['net_worth']
        self.previous_worth = self.start_worth
        # 4. 给 LSTMStrategy 用
        self.full_df     = df.reset_index(drop=True)
        self.start_date  = start_date
        self.end_date    = end_date

        # 用于计算每天收益率的列表
        self.daily_irrs  = []

    def run(self):
        # 正确初始化 state，兼容 env.reset() 返回 tuple 或 dict
        init_state = self.env.reset()
        if isinstance(init_state, tuple):
            state = init_state[0]
        else:
            state = init_state
        for idx, row in self.df.iterrows():
            date = row['date']
            # 从 env 获取最新状态
            # 假设 env._state 里包含 'open', 'cash', 'eth_held', 'net_worth'
            open_price = state['open']
            cash       = state['cash']
            held       = state['eth_held']
            networth   = state['net_worth']

            # 记录这一天的收益率
            daily_irr = (networth / self.previous_worth) - 1
            self.daily_irrs.append(daily_irr * 100)
            self.previous_worth = networth

            # 准备给策略的 env_state 字典
            env_state = {
                'open': open_price,
                'cash': cash,
                'eth_held': held,
                'net_worth': networth,
                # 给 LSTMStrategy 用
                'full_df': self.full_df,
                'start_date': self.start_date,
                'end_date': date.strftime('%Y-%m-%d')
            }

            # 询问策略今天该执行什么动作
            action = self.strategy.decide_action(current_row=row, env_state=env_state)

            # 送到环境中执行一步
            state, reward, done, info = self.env.step(action)
            if done:
                break

        # 回测结束后，计算总 IRR 与夏普率
        final_networth = state['net_worth']
        total_irr = (final_networth / self.start_worth) - 1
        irrs_arr  = np.array(self.daily_irrs)
        irr_mean  = irrs_arr.mean()
        irr_std   = irrs_arr.std(ddof=0)  # 样本方差
        risk_free = 0
        sharpe    = (irr_mean - risk_free) / (irr_std + 1e-8)  # 避免除零
        """
        注意：这里的夏普率计算是基于日收益率的，假设风险无风险利率为 0
            - irr_mean 是日收益率的平均值
            - irr_std 是日收益率的标准差
            - risk_free 是无风险利率，这里假设为 0
        """

        return {
            'total_irr': total_irr,
            'sharpe_ratio': sharpe
        }


# ---------------------------------
# 6. 主函数示例（执行各种策略回测）
# ---------------------------------
def main():
    # 1. 读数据并做特征工程
    df_raw = pd.read_csv('data/eth_daily.csv')
    fe     = FeatureEngineer(df_raw)
    df_all = fe.run_all()

    # 2. 打印每个时间段的统计信息（可选）
    for i in range(len(DATES) - 1):
        s_d, e_d = DATES[i], DATES[i + 1]
        tmp = df_all[(df_all['date'] >= s_d) & (df_all['date'] <= e_d)]
        print(f'[{s_d} ~ {e_d}] 共有 {len(tmp)} 条数据，首尾开盘价 & 最大 & 最小：', 
              tmp.iloc[0]['open'], tmp['open'].max(), tmp['open'].min(), tmp.iloc[-1]['open'])
    print()

    # 3. 准备 LSTM 模型（如果要跑 LSTM 策略，需要先训练）
    lstm_wrapper = LSTMModelWrapper(
        look_back=5, hidden_dim=100, num_layers=2, 
        lr=1e-3, epochs=100, batch_size=64
    )
    print('开始训练 LSTM 模型（训练区间：验证集）...')
    lstm_wrapper.train(df_all, VAL_START, VAL_END)
    print('LSTM 模型训练完毕。\n')

    # 4. 全量策略回测
    results = {}

    # 4.1 Optimal 策略
    strat_opt = OptimalStrategy(full_df=df_all)
    engine_opt = BacktestEngine(df=df_all, strategy=strat_opt, start_date=TEST_START, end_date=TEST_END)
    res_opt = engine_opt.run()
    print('[Optimal] 总 IRR:{:.2f}%,Sharpe:{:.2f}'.format(res_opt['total_irr']*100, res_opt['sharpe_ratio']))
    results['optimal'] = res_opt

    # 4.2 Buy&Hold 策略
    strat_bh = BuyAndHoldStrategy()
    engine_bh = BacktestEngine(df=df_all, strategy=strat_bh, start_date=TEST_START, end_date=TEST_END)
    res_bh = engine_bh.run()
    print('[Buy&Hold] 总 IRR:{:.2f}%,Sharpe:{:.2f}'.format(res_bh['total_irr']*100, res_bh['sharpe_ratio']))
    results['buy_and_hold'] = res_bh

    # 4.3 SMA 策略（遍历不同周期）
    for p in SMA_PERIODS:
        strat_sma = SMAStrategy(period=p)
        engine_sma = BacktestEngine(df=df_all, strategy=strat_sma, start_date=VAL_START, end_date=VAL_END)
        res_sma = engine_sma.run()
        print(f'[SMA {p}] 验证集 总 IRR:{res_sma["total_irr"]*100:.2f}%,Sharpe:{res_sma["sharpe_ratio"]:.6f}')
        # 测试集
        engine_sma_t = BacktestEngine(df=df_all, strategy=strat_sma, start_date=TEST_START, end_date=TEST_END)
        res_sma_t = engine_sma_t.run()
        print(f'[SMA {p}] 测试集   总 IRR:{res_sma_t["total_irr"]*100:.2f}%,Sharpe:{res_sma_t["sharpe_ratio"]:.6f}')
        results[f'sma_{p}'] = (res_sma, res_sma_t)

    # 4.4 SLMA 策略（遍历所有短/长均线组合）
    for i in range(len(SMA_PERIODS)-1):
        for j in range(i+1, len(SMA_PERIODS)):
            short_col = f'SMA_{SMA_PERIODS[i]}'
            long_col  = f'SMA_{SMA_PERIODS[j]}'
            strat_slma = SLMAStrategy(short=short_col, long=long_col)
            engine_slma = BacktestEngine(df=df_all, strategy=strat_slma, start_date=VAL_START, end_date=VAL_END)
            res_slma = engine_slma.run()
            print(f'[SLMA {short_col}/{long_col}] 验证集 IRR:{res_slma["total_irr"]*100:.2f}%,Sharpe:{res_slma["sharpe_ratio"]:.6f}')
            engine_slma_t = BacktestEngine(df=df_all, strategy=strat_slma, start_date=TEST_START, end_date=TEST_END)
            res_slma_t = engine_slma_t.run()
            print(f'[SLMA {short_col}/{long_col}] 测试集   IRR:{res_slma_t["total_irr"]*100:.2f}%,Sharpe:{res_slma_t["sharpe_ratio"]:.6f}')
            results[f'slma_{short_col}_{long_col}'] = (res_slma, res_slma_t)

    # 4.5 MACD 策略
    strat_macd = MACDStrategy()
    engine_macd = BacktestEngine(df=df_all, strategy=strat_macd, start_date=TEST_START, end_date=TEST_END)
    res_macd = engine_macd.run()
    print(f'[MACD] 测试集 IRR:{res_macd["total_irr"]*100:.2f}%,Sharpe:{res_macd["sharpe_ratio"]:.6f}')
    results['macd'] = res_macd

    # 4.6 Bollinger Bands 策略（period=20，multiplier=2）
    strat_bb = BollingerBandsStrategy(period=20, multiplier=2)
    engine_bb = BacktestEngine(df=df_all, strategy=strat_bb, start_date=TEST_START, end_date=TEST_END)
    res_bb = engine_bb.run()
    print(f'[BollingerBands] 测试集 IRR{res_bb["total_irr"]*100:.2f}%,Sharpe:{res_bb["sharpe_ratio"]:.6f}')
    results['bollinger_bands'] = res_bb

    # 4.7 LSTM 策略
    strat_lstm = LSTMStrategy(wrapper=lstm_wrapper)
    engine_lstm = BacktestEngine(df=df_all, strategy=strat_lstm, start_date=TEST_START, end_date=TEST_END)
    res_lstm = engine_lstm.run()
    print(f'[LSTM] 测试集 IRR:{res_lstm["total_irr"]*100:.2f}%,Sharpe:{res_lstm["sharpe_ratio"]:.6f}')
    results['lstm'] = res_lstm

    return results


if __name__ == '__main__':
    all_results = main()
    # 可以把 all_results 保存到文件，或者进⾏可视化分析