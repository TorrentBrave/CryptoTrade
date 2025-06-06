from utils import get_chat

class RAGENAgent:
    def __init__(self, base_prompt, model, seed, memory=None):
        self.base_prompt = base_prompt
        self.model = model
        self.seed = seed
        self.memory = memory or []
        self.policy_prompt = base_prompt

    def act(self, state, history, reward=None):
        # 组装 RAGEN 风格的 prompt，包含历史、奖励、反思等
        prompt = self._build_prompt(state, history, reward)
        action = get_chat(prompt, self.model, self.seed)
        return action

    def _build_prompt(self, state, history, reward):
        # 参考 RAGEN 论文，拼接历史、奖励、反思、当前状态
        prompt = self.policy_prompt
        if history:
            prompt += "\n\nHistory:\n"
            for h in history[-5:]:
                prompt += f"{h}\n"
        if reward is not None:
            prompt += f"\nLast reward: {reward}\n"
        prompt += f"\nCurrent state: {state}\n"
        prompt += "\nWhat is your next action? (output as a float in [-1,1])"
        return prompt

    def update_policy(self, feedback):
        # RAGEN: 用 reward/反思/环境反馈进化 policy prompt
        self.policy_prompt += f"\n[Self-Reflection/Policy Update]: {feedback}\n"