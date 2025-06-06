# preprocess table and run baseline

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from eth_env import ETHTradingEnv
from argparse import Namespace

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

device = 'cuda:7'
BUY, SELL = 0.5, -0.5
# BUY, SELL = 1, -1
FULL_BUY, FULL_SELL = 1, -1
strategies = ['SMA', 'MACD']
# strategies = ['SMA', 'MACD', 'SLMA', 'BollingerBands', 'buy_and_hold', 'optimal', 'LSTM', 'Multimodal']
sma_periods = [5, 10, 15, 20, 30]
# dates = ['2022-02-01','2023-02-01', '2024-02-01']
dates = ['2023-02-01','2023-08-01', '2024-02-01']
# dates = ['2023-12-01','2024-01-01', '2024-02-01']
VAL_START, VAL_END = dates[-3], dates[-2]
TEST_START, TEST_END = dates[-2], dates[-1]
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


df = pd.read_csv('data/eth_daily.csv')
df['date'] = pd.to_datetime(df['snapped_at'])

# SMA
for period in sma_periods:
    df[f'SMA_{period}'] = df['open'].rolling(window=period).mean()
    df[f'STD_{period}'] = df['open'].rolling(window=period).std()

# MACD and Signal Line
df['EMA_12'] = df['open'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['open'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

# dataset stats
for mi in range(len(dates)-1):
    starting_date = dates[mi]
    ending_date = dates[mi+1]
    y, m, _ = starting_date.split('-')
    df_m = df[(df['date'] >= starting_date) & (df['date'] <= ending_date)]
    print(f'{starting_date} to {ending_date} length:', len(df_m))
    stat = [df_m.iloc[0]['open'], df_m['open'].max(), df_m['open'].min(), df_m.iloc[-1]['open']]
    print('open, max, min, close:', [f'{s:.2f}' for s in stat])
    # df_m.to_csv(f'data/eth_f'{y}{m}'.csv', index=False)
print()

# # create dataset code for lstm
# def create_dataset(dataset, look_back=1):
#     X, Y = [], []
#     for i in range(len(dataset)-look_back):
#         a = dataset[i:(i+look_back), 0]
#         X.append(a)
#         Y.append(dataset[i + look_back, 0])
#     return np.array(X), np.array(Y)

def create_dataset(dataset, look_back=5, extra_features=None):
    # 支持多指标融合
    X, Y = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), :]
        if extra_features is not None:
            # 拼接Agent等结构化特征
            a = np.concatenate([a, extra_features[i:(i+look_back)]], axis=1)
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(Y), dtype=torch.float32).view(-1, 1)



class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(x.device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

# ==== 新增：多模型结构 ====
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, output_dim=1):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)
    def forward(self, x):
        # x: (batch, seq, input_dim)
        x = self.input_linear(x)
        x = x.permute(1, 0, 2)  # (seq, batch, d_model)
        x = self.transformer(x)
        x = x[-1]  # 取最后一个时间步
        return self.fc(x)

class ALSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attn = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attn(out), dim=1)
        context = (out * attn_weights).sum(dim=1)
        return self.fc(context)

# # LSTM strategy function
# def lstm_strategy(df, start_date, end_date, look_back=1):
#     # Filter the data
#     data = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
#     data = data['open'].values.reshape(-1, 1)
    
#     # Scale the data
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     data_scaled = scaler.fit_transform(data)
    
#     # Create the dataset
#     X, Y = create_dataset(data_scaled, look_back)
#     # dataset = TensorDataset(X, Y)
    
    
#     # Reshape X for sklearn compatibility
#     X = X.reshape(X.shape[0], look_back)
    
#     # Split the data into training and test sets
#     train_size = int(len(X) * 0.67)
#     trainX, trainY = X[:train_size], Y[:train_size]
    
#     # Define and train the linear regression model
#     model = LinearRegression()
#     model.fit(trainX, trainY)
    
#     # Make predictions
#     last_train_batch = trainX[-1:].reshape(1, look_back)
#     next_day_prediction = model.predict(last_train_batch)
#     next_day_prediction = scaler.inverse_transform(next_day_prediction.reshape(-1, 1))
#     current_price = scaler.inverse_transform(trainY[-1].reshape(-1, 1))
    
#     # Decide action based on prediction, buy, sell or hold
#     if next_day_prediction > current_price:
#         action = 'Buy'
#     elif next_day_prediction < current_price:
#         action = 'Sell'
#     else:
#         action = 0
    
#     return action

 
def lstm_strategy(df, start_date, end_date, look_back=5, model_type='LSTM', tpe_config=None):
    # 多指标融合
    data = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    # 选取多指标
    features = ['open', 'volume', 'SMA_5', 'MACD']  # 可扩展
    data_values = data[features].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data_values)
    # Agent结构化特征
    extra_features = get_extra_features(df, start_date, end_date)
    X, Y = create_dataset(data_scaled, look_back, extra_features=extra_features)
    dataset = TensorDataset(X, Y)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = get_model(model_type, input_dim=X.shape[2])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_model(model, train_loader, criterion, optimizer, num_epochs=100, tpe_config=tpe_config)
    last_sequence = data_scaled[-look_back:]
    last_extra = extra_features[-look_back:]
    last_input = np.concatenate([last_sequence, last_extra], axis=1)
    last_input = torch.tensor(last_input, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        model.eval()
        next_day_prediction = model(last_input)
        next_day_prediction = next_day_prediction.cpu()
    next_day_prediction = scaler.inverse_transform(
        np.concatenate([next_day_prediction.numpy(), np.zeros((1, len(features)-1))], axis=1)
    )[0, 0]
    current_price = scaler.inverse_transform(
        np.concatenate([[Y[-1].item()], np.zeros(len(features)-1)]).reshape(1, -1)
    )[0, 0]
    # 反思Agent反馈
    feedback = reflection_agent_feedback(next_day_prediction, current_price)
    action = 'Hold'
    if next_day_prediction > current_price + feedback:
        action = 'Buy'
    elif next_day_prediction < current_price + feedback:
        action = 'Sell'
    else:
        action = 0
    return action

# 1st strategy: Simple MA 
# when the asset's open price is below the its SMA, and the volume is above the its SMA it's a buying signal, and vice versa for selling.

# 2nd strategy: MACD
# MACD = 12-day EMA - 26-day EMA
# Signal Line = 9-day EMA of MACD
# When MACD crosses above the signal line, it's a buying signal, and vice versa for selling.

# 3rd strategy: short and long strategy (SLMA) - If the short period SMA is below the long period SMA, it means that the trend is going down, so it's a sell signal, it's also known as the death cross.
# Otherwise, the trend is shiftting up, and it's a buy signal, it's also called the golden cross.
    
# 4th strategy: Bollinger Bands




def run_strategy(strategy, sargs):
    env = ETHTradingEnv(Namespace(starting_date=sargs['starting_date'], ending_date=sargs['ending_date'], dataset='eth'))
    df_tmp = df[(df['date'] >= sargs['starting_date']) & (df['date'] <= sargs['ending_date'])]
    df_tmp.reset_index(drop=True, inplace=True)
    state, reward, done, info = env.reset()  # only use env to act and track profit

    starting_net_worth = state['net_worth']
    irrs = []
    previous_signal = None  # Track the previous day signal
    previous_net_worth = starting_net_worth
    # Iterate through each row in the DataFrame to simulate trading

    for index, row in df_tmp.iterrows():
        open_price = state['open']
        cash = state['cash']
        eth_held = state['eth_held']
        net_worth = state['net_worth']
        date = row['date']
        y, m, d = date.year, date.month, date.day
        irrs.append((net_worth / previous_net_worth) - 1)
        previous_net_worth = net_worth
        if done:
            break

        if strategy == 'SMA':
            period = sargs['period']
            sma_column = f'SMA_{period}'
            current_signal = 'hold'
            if open_price > row[sma_column]:  # golden cross?
                # current_signal = 'sell'
                current_signal = 'buy'
            elif open_price < row[sma_column]:  # death cross?
                # current_signal = 'buy'
                current_signal = 'sell'
                
            action = 0
            # if current_signal != previous_signal:
            if True:
                if current_signal == 'buy' and cash > 0:
                    action = BUY
                elif current_signal == 'sell' and eth_held > 0:
                    action = SELL
            previous_signal = current_signal
                
        elif strategy == 'SLMA':
            short = sargs['short']
            long = sargs['long']
            current_signal = 'hold'
            if row[short] > row[long]:  # golden cross?
                current_signal = 'buy'
            elif row[short] < row[long]:  # death cross?
                current_signal = 'sell'

            action = 0
            # if current_signal != previous_signal:
            if True:
                if current_signal == 'buy':
                    action = BUY
                elif current_signal == 'sell' and eth_held > 0:
                    action = SELL
            previous_signal = current_signal

        elif strategy == 'MACD':
            current_signal = 'hold'
            if row['MACD'] < row['Signal_Line']:
                current_signal = 'buy'
            elif row['MACD'] > row['Signal_Line']:
                current_signal = 'sell'

            action = 0
            # if current_signal != previous_signal:
            if True:
                if current_signal == 'buy' and cash > 0:
                    action = BUY
                elif current_signal == 'sell' and eth_held > 0:
                    action = SELL
            previous_signal = current_signal

        elif strategy == 'BollingerBands':
            period = sargs['period']  # e.g., 20 for a 20-day SMA
            multiplier = sargs['multiplier']  # Commonly set to 2
            sma = row[f'SMA_{period}']
            sd = row[f'STD_{period}']
            
            upper_band = sma + (sd * multiplier)
            lower_band = sma - (sd * multiplier)

            current_signal = 'hold'
            if open_price < lower_band:
                current_signal = 'buy'
            elif open_price > upper_band:
                current_signal = 'sell'

            action = 0
            # if current_signal != previous_signal:
            if True:
                if current_signal == 'buy' and cash > 0:
                    action = BUY
                elif current_signal == 'sell' and eth_held > 0:
                    action = SELL
            previous_signal = current_signal

        elif strategy == 'buy_and_hold':
            action = 0
            if cash > 0:
                action = FULL_BUY
        
        # here to add LSTM/Transformer/ALSTM等多模型策略
        elif strategy in ['LSTM', 'Transformer', 'ALSTM']:
            # 支持多模型选择与训练管道增强
            action = lstm_strategy(
                df, sargs['starting_date'], sargs['ending_date'],
                look_back=5,
                model_type=strategy,
                tpe_config={'zero_bias': True, 'time_weight': True}
            )
            if action == 'Buy' and cash > 0:
                action = BUY
            elif action == 'Sell' and eth_held > 0:
                action = SELL
            else:
                action = 0

        elif strategy == 'optimal':
            next_open = df_tmp.iloc[index+1]['open']
            if open_price < next_open:
                action = FULL_BUY
            elif open_price > next_open:
                action = FULL_SELL
            else:
                action = 0

        else:
            raise ValueError('Invalid strategy')

        state, reward, done, info = env.step(action)


    net_worth = state['net_worth']
    total_irr = (net_worth / starting_net_worth) - 1
    irrs = np.array(irrs) * 100
    irr_mean = np.mean(irrs)
    irr_std = np.std(irrs)
    risk_free_rate = 0  # same as sociodojo
    result = {
        'total_irr': total_irr,
        'sharp_ratio': (irr_mean - risk_free_rate) / irr_std,
    }
    result_str = f'Total IRR: {total_irr*100:.2f} %, Sharp Ratio: {result["sharp_ratio"]:.2f}'
    print(result_str)
    

# strategy = 'LSTM'
# print(strategy)
# run_strategy(strategy, {'starting_date': TEST_START, 'ending_date': TEST_END})


strategy = 'optimal'
print(strategy)
run_strategy(strategy, {'starting_date': TEST_START, 'ending_date': TEST_END})


strategy = 'buy_and_hold'
print(strategy)
run_strategy(strategy, {'starting_date': TEST_START, 'ending_date': TEST_END})


strategy = 'SMA'
for period in sma_periods:
    sargs = {'period': period, 'starting_date': VAL_START, 'ending_date': VAL_END}
    print(f'{strategy}, Period: {period}')
    run_strategy(strategy, sargs)

period = 15
print(f'{strategy}, Period: {period}')
sargs = {'period': period, 'starting_date': TEST_START, 'ending_date': TEST_END}
run_strategy(strategy, sargs)


strategy = 'SLMA'
for i in range(len(sma_periods)-1):
    for j in range(i+1, len(sma_periods)):
        short = f'SMA_{sma_periods[i]}'
        long = f'SMA_{sma_periods[j]}'
        sargs = {'short': short, 'long': long, 'starting_date': VAL_START, 'ending_date': VAL_END}
        print(f'{strategy}, Short: {short}, Long: {long}')
        run_strategy(strategy, sargs)

short, long = 'SMA_15', 'SMA_30'
sargs = {'short': short, 'long': long, 'starting_date': TEST_START, 'ending_date': TEST_END}
print(f'{strategy}, Short: {short}, Long: {long}')
run_strategy(strategy, sargs)


strategy = 'MACD'
sargs = {'starting_date': TEST_START, 'ending_date': TEST_END}
print(f'{strategy}')
run_strategy(strategy, sargs)


strategy = 'BollingerBands'
period = 20
multiplier = 2
sargs = {'period': period, 'multiplier': multiplier, 'starting_date': TEST_START, 'ending_date': TEST_END}
print(f'{strategy}, Period: {period}, Multiplier: {multiplier}')
run_strategy(strategy, sargs)

# ==== 预留：细粒度实时预测与可视化接口 ====
def realtime_predict_and_visualize(df, model, interval='1min', look_back=5, features=None, scaler=None):
    """
    实时预测与动态可视化
    df: 包含历史数据的DataFrame，需有'date'列
    model: 已训练好的时序模型
    interval: 预测步长（如'1min'）
    look_back: 滑动窗口长度
    features: 用于模型输入的特征名列表
    scaler: 训练时用的scaler
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    if features is None:
        features = ['open', 'volume', 'SMA_5', 'MACD']
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(df[features].values)

    # 按interval重采样（如需分钟级，需有分钟数据）
    df_sorted = df.sort_values('date').reset_index(drop=True)
    # 若数据为日线，interval可忽略
    data_values = df_sorted[features].values
    data_scaled = scaler.transform(data_values)

    pred_dates = []
    pred_prices = []
    true_prices = []

    for i in range(look_back, len(df_sorted)):
        input_seq = data_scaled[i-look_back:i]
        input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(model.fc.weight.device)
        with torch.no_grad():
            model.eval()
            pred = model(input_tensor).cpu().numpy()
        # 反归一化，仅取open
        pred_full = np.concatenate([pred, np.zeros((1, len(features)-1))], axis=1)
        pred_price = scaler.inverse_transform(pred_full)[0, 0]
        true_price = df_sorted.iloc[i]['open']
        pred_dates.append(df_sorted.iloc[i]['date'])
        pred_prices.append(pred_price)
        true_prices.append(true_price)

    plt.figure(figsize=(12, 6))
    plt.plot(pred_dates, true_prices, label='真实价格', color='blue')
    plt.plot(pred_dates, pred_prices, label='实时预测', color='orange')
    plt.fill_between(pred_dates, np.array(pred_prices)*0.98, np.array(pred_prices)*1.02, color='orange', alpha=0.2, label='预测置信区间')
    plt.xlabel('时间')
    plt.ylabel('价格')
    plt.title('实时预测价格与实际价格对比')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()


# ====== 示例：训练模型并可视化预测 ======
if __name__ == '__main__':
    # 选择测试区间
    test_df = df[(df['date'] >= TEST_START) & (df['date'] <= TEST_END)].reset_index(drop=True)
    # 修正特征，移除volume
    features = ['open', 'SMA_5', 'MACD']
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(test_df[features].values)
    look_back = 5

    # 构造数据集
    data_values = test_df[features].values
    data_scaled = scaler.transform(data_values)
    X, Y = create_dataset(data_scaled, look_back)
    dataset = TensorDataset(X, Y)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 训练LSTM模型（可换成Transformer/ALSTM）
    model = LSTMModel(input_dim=X.shape[2], hidden_dim=100, num_layers=2, output_dim=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(30):  # 训练轮数可调
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # 可视化预测
    realtime_predict_and_visualize(test_df, model, look_back=look_back, features=features, scaler=scaler)
