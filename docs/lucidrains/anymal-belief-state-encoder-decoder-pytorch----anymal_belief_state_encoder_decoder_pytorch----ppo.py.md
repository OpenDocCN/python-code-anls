# `.\lucidrains\anymal-belief-state-encoder-decoder-pytorch\anymal_belief_state_encoder_decoder_pytorch\ppo.py`

```py
# 导入必要的库
from collections import namedtuple, deque
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from anymal_belief_state_encoder_decoder_pytorch import Anymal
from anymal_belief_state_encoder_decoder_pytorch.networks import unfreeze_all_layers_
from einops import rearrange

# 定义一个命名元组Memory，用于存储经验数据
Memory = namedtuple('Memory', ['state', 'action', 'action_log_prob', 'reward', 'done', 'value'])

# 定义一个数据集类ExperienceDataset，用于存储经验数据
class ExperienceDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, ind):
        return tuple(map(lambda t: t[ind], self.data))

# 创建一个混洗数据加载器函数
def create_shuffled_dataloader(data, batch_size):
    ds = ExperienceDataset(data)
    return DataLoader(ds, batch_size = batch_size, shuffle = True)

# 定义一个归一化函数，用于对张量进行归一化处理
def normalize(t, eps = 1e-5):
    return (t - t.mean()) / (t.std() + eps)

# 定义一个裁剪值损失函数，用于计算值函数的损失
def clipped_value_loss(values, rewards, old_values, clip):
    value_clipped = old_values + (values - old_values).clamp(-clip, clip)
    value_loss_1 = (value_clipped.flatten() - rewards) ** 2
    value_loss_2 = (values.flatten() - rewards) ** 2
    return torch.mean(torch.max(value_loss_1, value_loss_2))

# 定义一个模拟环境类MockEnv，用于模拟环境状态和动作
class MockEnv(object):
    def __init__(
        self,
        proprio_dim,
        extero_dim,
        privileged_dim,
        num_legs = 4
    ):
        self.proprio_dim = proprio_dim
        self.extero_dim = extero_dim
        self.privileged_dim = privileged_dim
        self.num_legs = num_legs

    def rand_state(self):
        return (
            torch.randn((self.proprio_dim,)),
            torch.randn((self.num_legs, self.extero_dim,)),
            torch.randn((self.privileged_dim,))
        )

    def reset(self):
        return self.rand_state()

    def step(self, action):
        reward = torch.randn((1,))
        done = torch.tensor([False])
        return self.rand_state(), reward, done, None

# 定义一个PPO类，用于执行PPO算法
class PPO(nn.Module):
    def __init__(
        self,
        *,
        env,
        anymal,
        epochs = 2,
        lr = 5e-4,
        betas = (0.9, 0.999),
        eps_clip = 0.2,
        beta_s = 0.005,
        value_clip = 0.4,
        max_timesteps = 10000,
        update_timesteps = 5000,
        lam = 0.95,
        gamma = 0.99,
        minibatch_size = 8300
    ):
        super().__init__()
        assert isinstance(anymal, Anymal)
        self.env = env
        self.anymal = anymal

        self.minibatch_size = minibatch_size
        self.optimizer = Adam(anymal.teacher.parameters(), lr = lr, betas = betas)
        self.epochs = epochs

        self.max_timesteps = max_timesteps
        self.update_timesteps = update_timesteps

        self.beta_s = beta_s
        self.eps_clip = eps_clip
        self.value_clip = value_clip

        self.lam = lam
        self.gamma = gamma

        # 在论文中，他们说传递给teacher的观察值是通过运行均值进行归一化的

        self.running_proprio, self.running_extero = anymal.get_observation_running_stats()

    def learn_from_memories(
        self,
        memories,
        next_states
    ):
        device = next(self.parameters()).device

        # 从内存中检索和准备数据进行训练
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

        states = tuple(zip(*states))

        # 计算广义优势估计

        next_states = map(lambda t: t.to(device), next_states)
        next_states = map(lambda t: rearrange(t, '... -> 1 ...'), next_states)

        _, next_value = self.anymal.forward_teacher(*next_states, return_value_head = True)
        next_value = next_value.detach()

        values = values + [next_value]

        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * masks[i] - values[i]
            gae = delta + self.gamma * self.lam * masks[i] * gae
            returns.insert(0, gae + values[i])

        # 将值转换为torch张量

        to_torch_tensor = lambda t: torch.stack(t).to(device).detach()

        states = map(to_torch_tensor, states)
        actions = to_torch_tensor(actions)
        old_log_probs = to_torch_tensor(old_log_probs)

        old_values = to_torch_tensor(values[:-1])
        old_values = rearrange(old_values, '... 1 -> ...')

        rewards = torch.tensor(returns).float().to(device)

        # 为策略阶段训练准备数据加载器

        dl = create_shuffled_dataloader([*states, actions, old_log_probs, rewards, old_values], self.minibatch_size)

        # 策略阶段训练，类似于原始的PPO

        for _ in range(self.epochs):
            for proprio, extero, privileged, actions, old_log_probs, rewards, old_values in dl:

                dist, values = self.anymal.forward_teacher(
                    proprio, extero, privileged,
                    return_value_head = True,
                    return_action_categorical_dist = True
                )

                action_log_probs = dist.log_prob(actions)

                entropy = dist.entropy()
                ratios = (action_log_probs - old_log_probs).exp()
                advantages = normalize(rewards - old_values.detach())
                surr1 = ratios * advantages
                surr2 = ratios.clamp(1 - self.eps_clip, 1 + self.eps_clip) * advantages

                policy_loss = - torch.min(surr1, surr2) - self.beta_s * entropy

                value_loss = clipped_value_loss(values, rewards, old_values, self.value_clip)

                (policy_loss.mean() + value_loss.mean()).backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

    # 执行一个episode的学习
    # 定义一个前向传播函数，用于执行模型的前向传播操作
    def forward(self):
        # 获取模型参数中的设备信息
        device = next(self.parameters()).device
        # 解冻所有层的参数
        unfreeze_all_layers_(self.anymal)

        # 初始化时间步数和状态信息
        time = 0
        states = self.env.reset() # 状态假设为（本体感知，外部感知，特权信息）
        memories = deque([])

        # 清空本体感知和外部感知的运行均值
        self.running_proprio.clear()
        self.running_extero.clear()

        # 循环执行最大时间步数次
        for timestep in range(self.max_timesteps):
            time += 1

            # 将状态信息转移到指定设备上
            states = list(map(lambda t: t.to(device), states))
            proprio, extero, privileged = states

            # 更新用于教师的观测运行均值
            self.running_proprio.push(proprio)
            self.running_extero.push(extero)

            # 对教师的观测状态进行归一化处理（本体感知和外部感知）
            states = (
                self.running_proprio.norm(proprio),
                self.running_extero.norm(extero),
                privileged
            )

            # 将状态信息重新排列为适合模型输入的形式
            anymal_states = list(map(lambda t: rearrange(t, '... -> 1 ...'), states))

            # 执行模型的前向传播操作，获取动作分布和值
            dist, values = self.anymal.forward_teacher(
                *anymal_states,
                return_value_head = True,
                return_action_categorical_dist = True
            )

            # 从动作分布中采样动作
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
            action = action.item()

            # 执行动作，获取下一个状态、奖励、是否结束标志和额外信息
            next_states, reward, done, _ = self.env.step(action)

            # 创建记忆对象，存储状态、动作、动作对数概率、奖励、是否结束标志和值
            memory = Memory(states, action, action_log_prob, reward, done, values)
            memories.append(memory)

            # 更新状态信息为下一个状态
            states = next_states

            # 每隔一定时间步数执行一次经验回放和学习
            if time % self.update_timesteps == 0:
                self.learn_from_memories(memories, next_states)
                memories.clear()

            # 如果环境结束，则跳出循环
            if done:
                break

        # 打印训练完成一���的信息
        print('trained for 1 episode')
```