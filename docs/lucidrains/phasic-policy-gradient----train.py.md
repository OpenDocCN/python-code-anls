# `.\lucidrains\phasic-policy-gradient\train.py`

```py
# 导入必要的库
import os
import fire
from collections import deque, namedtuple

from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.distributions import Categorical
import torch.nn.functional as F

import gym

# 定义常量
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义命名元组
Memory = namedtuple('Memory', ['state', 'action', 'action_log_prob', 'reward', 'done', 'value'])
AuxMemory = namedtuple('Memory', ['state', 'target_value', 'old_values'])

# 定义数据集类
class ExperienceDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, ind):
        return tuple(map(lambda t: t[ind], self.data))

# 创建混洗数据加载器
def create_shuffled_dataloader(data, batch_size):
    ds = ExperienceDataset(data)
    return DataLoader(ds, batch_size = batch_size, shuffle = True)

# 辅助函数

# 检查值是否存在
def exists(val):
    return val is not None

# 归一化函数
def normalize(t, eps = 1e-5):
    return (t - t.mean()) / (t.std() + eps)

# 更新网络参数
def update_network_(loss, optimizer):
    optimizer.zero_grad()
    loss.mean().backward()
    optimizer.step()

# 初始化网络参数
def init_(m):
    if isinstance(m, nn.Linear):
        gain = torch.nn.init.calculate_gain('tanh')
        torch.nn.init.orthogonal_(m.weight, gain)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

# 定义 Actor 神经网络类
class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, num_actions),
            nn.Softmax(dim=-1)
        )

        self.value_head = nn.Linear(hidden_dim, 1)
        self.apply(init_)

    def forward(self, x):
        hidden = self.net(x)
        return self.action_head(hidden), self.value_head(hidden)

# 定义 Critic 神经网络类
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(init_)

    def forward(self, x):
        return self.net(x)

# 定义 PPG 代理类
class PPG:
    def __init__(
        self,
        state_dim,
        num_actions,
        actor_hidden_dim,
        critic_hidden_dim,
        epochs,
        epochs_aux,
        minibatch_size,
        lr,
        betas,
        lam,
        gamma,
        beta_s,
        eps_clip,
        value_clip
    ):
        self.actor = Actor(state_dim, actor_hidden_dim, num_actions).to(device)
        self.critic = Critic(state_dim, critic_hidden_dim).to(device)
        self.opt_actor = Adam(self.actor.parameters(), lr=lr, betas=betas)
        self.opt_critic = Adam(self.critic.parameters(), lr=lr, betas=betas)

        self.minibatch_size = minibatch_size

        self.epochs = epochs
        self.epochs_aux = epochs_aux

        self.lam = lam
        self.gamma = gamma
        self.beta_s = beta_s

        self.eps_clip = eps_clip
        self.value_clip = value_clip

    # 保存模型参数
    def save(self):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, f'./ppg.pt')
    # 加载模型参数
    def load(self):
        # 检查是否存在模型参数文件
        if not os.path.exists('./ppg.pt'):
            return

        # 从文件中加载模型参数
        data = torch.load(f'./ppg.pt')
        # 更新 actor 模型参数
        self.actor.load_state_dict(data['actor'])
        # 更新 critic 模型参数
        self.critic.load_state_dict(data['critic'])

    # 学习函数，用于训练模型
    def learn(self, memories, aux_memories, next_state):
        # 从记忆中提取并准备训练数据
        states = []
        actions = []
        old_log_probs = []
        rewards = []
        masks = []
        values = []

        for mem in memories:
            states.append(mem.state)
            actions.append(torch.tensor(mem.action))
            old_log_probs.append(mem.action_log_prob)
            rewards.append(mem.reward)
            masks.append(1 - float(mem.done))
            values.append(mem.value)

        # 计算广义优势估计值
        next_state = torch.from_numpy(next_state).to(device)
        next_value = self.critic(next_state).detach()
        values = values + [next_value]

        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * masks[i] - values[i]
            gae = delta + self.gamma * self.lam * masks[i] * gae
            returns.insert(0, gae + values[i])

        # 将值转换为 torch 张量
        to_torch_tensor = lambda t: torch.stack(t).to(device).detach()

        states = to_torch_tensor(states)
        actions = to_torch_tensor(actions)
        old_values = to_torch_tensor(values[:-1])
        old_log_probs = to_torch_tensor(old_log_probs)

        rewards = torch.tensor(returns).float().to(device)

        # 将状态和目标值存储到辅助内存缓冲区以供后续训练使用
        aux_memory = AuxMemory(states, rewards, old_values)
        aux_memories.append(aux_memory)

        # 为策略阶段训练准备数据加载器
        dl = create_shuffled_dataloader([states, actions, old_log_probs, rewards, old_values], self.minibatch_size)

        # 策略阶段训练，类似于原始的 PPO
        for _ in range(self.epochs):
            for states, actions, old_log_probs, rewards, old_values in dl:
                action_probs, _ = self.actor(states)
                values = self.critic(states)
                dist = Categorical(action_probs)
                action_log_probs = dist.log_prob(actions)
                entropy = dist.entropy()

                # 计算剪切的替代目标，经典的 PPO 损失
                ratios = (action_log_probs - old_log_probs).exp()
                advantages = normalize(rewards - old_values.detach())
                surr1 = ratios * advantages
                surr2 = ratios.clamp(1 - self.eps_clip, 1 + self.eps_clip) * advantages
                policy_loss = - torch.min(surr1, surr2) - self.beta_s * entropy

                # 更新策略网络
                update_network_(policy_loss, self.opt_actor)

                # 计算值损失并更新值网络，与策略网络分开
                value_loss = clipped_value_loss(values, rewards, old_values, self.value_clip)

                update_network_(value_loss, self.opt_critic)
    # 定义一个辅助学习函数，用于训练辅助记忆
    def learn_aux(self, aux_memories):
        # 将状态和目标值合并成一个张量
        states = []
        rewards = []
        old_values = []
        for state, reward, old_value in aux_memories:
            states.append(state)
            rewards.append(reward)
            old_values.append(old_value)

        # 将状态、奖励和旧值连接成一个张量
        states = torch.cat(states)
        rewards = torch.cat(rewards)
        old_values = torch.cat(old_values)

        # 获取用于最小化 kl 散度和剪切的旧动作预测值
        old_action_probs, _ = self.actor(states)
        old_action_probs.detach_()

        # 为辅助阶段训练准备数据加载器
        dl = create_shuffled_dataloader([states, old_action_probs, rewards, old_values], self.minibatch_size)

        # 提出的辅助阶段训练
        # 在将值蒸馏到策略网络的同时，确保策略网络不改变动作预测值（kl 散度损失）
        for epoch in range(self.epochs_aux):
            for states, old_action_probs, rewards, old_values in tqdm(dl, desc=f'auxiliary epoch {epoch}'):
                action_probs, policy_values = self.actor(states)
                action_logprobs = action_probs.log()

                # 策略网络损失由 kl 散度损失和辅助损失组成
                aux_loss = clipped_value_loss(policy_values, rewards, old_values, self.value_clip)
                loss_kl = F.kl_div(action_logprobs, old_action_probs, reduction='batchmean')
                policy_loss = aux_loss + loss_kl

                # 更新策略网络
                update_network_(policy_loss, self.opt_actor)

                # 论文指出在辅助阶段额外训练值网络非常重要
                values = self.critic(states)
                value_loss = clipped_value_loss(values, rewards, old_values, self.value_clip)

                # 更新值网络
                update_network_(value_loss, self.opt_critic)
# 主函数
def main(
    env_name = 'LunarLander-v2',  # 环境名称，默认为'LunarLander-v2'
    num_episodes = 50000,  # 总的训练轮数，默认为50000
    max_timesteps = 500,  # 每轮最大时间步数，默认为500
    actor_hidden_dim = 32,  # Actor神经网络隐藏层维度，默认为32
    critic_hidden_dim = 256,  # Critic神经网络隐藏层维度，默认为256
    minibatch_size = 64,  # 每次训练的样本批量大小，默认为64
    lr = 0.0005,  # 学习率，默认为0.0005
    betas = (0.9, 0.999),  # Adam优化器的beta参数，默认为(0.9, 0.999)
    lam = 0.95,  # GAE的lambda参数，默认为0.95
    gamma = 0.99,  # 折扣因子，默认为0.99
    eps_clip = 0.2,  # PPO算法的epsilon clip参数，默认为0.2
    value_clip = 0.4,  # Critic的值函数clip参数，默认为0.4
    beta_s = .01,  # 熵损失的权重参数，默认为0.01
    update_timesteps = 5000,  # 更新模型的时间步数间隔，默认为5000
    num_policy_updates_per_aux = 32,  # 辅助网络更新次数，默认为32
    epochs = 1,  # 主网络训练轮数，默认为1
    epochs_aux = 6,  # 辅助网络训练轮数，默认为6
    seed = None,  # 随机种子，默认为None
    render = False,  # 是否渲染环境，默认为False
    render_every_eps = 250,  # 每隔多少轮渲染一次，默认为250
    save_every = 1000,  # 每隔多少轮保存模型，默认为1000
    load = False,  # 是否加载已有模型，默认为False
    monitor = False  # 是否监视环境，默认为False
):
    env = gym.make(env_name)  # 创建环境

    if monitor:
        env = gym.wrappers.Monitor(env, './tmp/', force=True)  # 监视环境

    state_dim = env.observation_space.shape[0]  # 状态空间维度
    num_actions = env.action_space.n  # 动作空间维度

    memories = deque([])  # 存储经验的队列
    aux_memories = deque([])  # 存储辅助经验的队列

    agent = PPG(  # 创建PPO算法的代理
        state_dim,
        num_actions,
        actor_hidden_dim,
        critic_hidden_dim,
        epochs,
        epochs_aux,
        minibatch_size,
        lr,
        betas,
        lam,
        gamma,
        beta_s,
        eps_clip,
        value_clip
    )

    if load:
        agent.load()  # 加载模型

    if exists(seed):  # 如果存在随机种子
        torch.manual_seed(seed)  # 设置PyTorch随机种子
        np.random.seed(seed)  # 设置NumPy随机种子

    time = 0  # 时间步数
    updated = False  # 是否更新模型
    num_policy_updates = 0  # 策略更新次数

    for eps in tqdm(range(num_episodes), desc='episodes'):  # 遍历训练轮数
        render_eps = render and eps % render_every_eps == 0  # 是否渲染当前轮次
        state = env.reset()  # 重置环境状态
        for timestep in range(max_timesteps):  # 遍历每个时间步
            time += 1  # 时间步数加1

            if updated and render_eps:  # 如果已更新并需要渲染
                env.render()  # 渲染环境

            state = torch.from_numpy(state).to(device)  # 转换状态为PyTorch张量
            action_probs, _ = agent.actor(state)  # 获取动作概率
            value = agent.critic(state)  # 获取值函数

            dist = Categorical(action_probs)  # 创建分类分布
            action = dist.sample()  # 采样动作
            action_log_prob = dist.log_prob(action)  # 计算动作对数概率
            action = action.item()  # 转换动作为标量

            next_state, reward, done, _ = env.step(action)  # 执行动作

            memory = Memory(state, action, action_log_prob, reward, done, value)  # 创建经验
            memories.append(memory)  # 将经验添加到队列

            state = next_state  # 更新状态

            if time % update_timesteps == 0:  # 如果达到更新时间步
                agent.learn(memories, aux_memories, next_state)  # 更新主网络
                num_policy_updates += 1  # 策略更新次数加1
                memories.clear()  # 清空经验队列

                if num_policy_updates % num_policy_updates_per_aux == 0:  # 达到辅助网络更新次数
                    agent.learn_aux(aux_memories)  # 更新辅助网络
                    aux_memories.clear()  # 清空辅助经验队列

                updated = True  # 设置为已更新

            if done:  # 如果环境结束
                if render_eps:  # 如果需要渲染
                    updated = False  # 设置为未更新
                break  # 跳出循环

        if render_eps:  # 如果需要渲染
            env.close()  # 关闭环境

        if eps % save_every == 0:  # 每隔一定轮次保存模型
            agent.save()  # 保存模型

if __name__ == '__main__':
    fire.Fire(main)  # 使用Fire库执行主函数
```