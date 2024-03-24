# `.\lucidrains\PaLM-rlhf-pytorch\palm_rlhf_pytorch\ppo.py`

```
import math
from pathlib import Path
import copy
from tqdm import tqdm
from functools import partial
from collections import deque, namedtuple
from random import randrange

from beartype import beartype
from beartype.typing import List, Optional, Callable, Deque

import torch
from torch import nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

from palm_rlhf_pytorch.palm import PaLM
from palm_rlhf_pytorch.reward import RewardModel
from palm_rlhf_pytorch.optimizer import get_optimizer
from palm_rlhf_pytorch.utils import masked_mean, eval_decorator

from accelerate import Accelerator

# actor critic - PaLM with lora

PPOActionCriticReturn = namedtuple('PPOActionCriticReturn', [
    'actions',
    'sequence',
    'mask',
    'prompt_mask',
    'action_logits',
    'values'
])

@beartype
class ActorCritic(nn.Module):
    def __init__(
        self,
        palm: PaLM,
        critic_palm: Optional[PaLM] = None,
        pooled_values = False,
        actor_lora = True,
        critic_lora = True,
        actor_lora_r = 8,
        critic_lora_r = 8,
        actor_lora_scope = 'actor',
        critic_lora_scope = 'critic',
        actor_dropout = 0.,
        critic_dropout = 0.
    ):
        super().__init__()
        self.actor_palm = palm

        self.critic_palm = critic_palm

        if not exists(self.critic_palm):
            self.critic_palm = copy.deepcopy(palm)

        self.actor_palm.set_dropout(actor_dropout)
        self.critic_palm.set_dropout(critic_dropout)

        self.actor_lora = actor_lora
        self.critic_lora = critic_lora

        self.actor_lora_scope = actor_lora_scope if actor_lora else None
        self.critic_lora_scope = critic_lora_scope if critic_lora else None

        if self.actor_lora:
            self.actor_palm.add_finetune_params(actor_lora_scope, lora_r = actor_lora_r)

        if self.critic_lora:
            self.critic_palm.add_finetune_params(critic_lora_scope, lora_r = critic_lora_r)

        self.pooled_values = pooled_values
        self.value_head = nn.Sequential(
            nn.Linear(palm.dim, 1),
            Rearrange('... 1 -> ...')
        )

        nn.init.zeros_(self.value_head[0].bias)
        nn.init.orthogonal_(self.value_head[0].weight, gain = math.sqrt(2))

    def actor_parameters(self):
        # 返回 actor 参数，如果不使用 lora，则返回 actor_palm 的参数
        if not self.actor_lora:
            return self.actor_palm.parameters()

        return [
            *self.actor_palm.finetune_parameters(self.actor_lora_scope)
        ]

    def critic_parameters(self):
        # 返回 critic 参数，如果不使用 lora，则返回 critic_palm 和 value_head 的参数
        if not self.actor_lora:
            return [*self.critic_palm.parameters(), *self.value_head.parameters()]

        return [
            *self.critic_palm.finetune_parameters(self.critic_lora_scope),
            *self.value_head.parameters()
        ]

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        state,
        max_seq_len,
        eos_token = None,
        return_values = False,
        **kwargs
    # 生成动作序列，根据当前状态和最大序列长度
    actions = self.actor_palm.generate(
        max_seq_len,
        prompt = state,       
        eos_token = eos_token,     
        finetune_scope = self.actor_lora_scope,
        use_tqdm = True,
        **kwargs
    )

    # 将当前状态和生成的动作序列拼接在一起
    sequence = torch.cat((state, actions), dim = -1)
    action_len = actions.shape[-1]
    state_len = state.shape[-1]

    # 创建用于标记当前状态的掩码
    prompt_mask = torch.arange(sequence.shape[-1], device = state.device) < state_len
    prompt_mask = repeat(prompt_mask, 'n -> b n', b = sequence.shape[0])

    # 创建用于标记动作的掩码
    action_mask = ~prompt_mask

    mask = None
    # 如果存在结束标记，创建用于标记结束标记的掩码
    if exists(eos_token):
        mask = ((sequence == eos_token).cumsum(dim = -1) == 0)
        mask = F.pad(mask, (1, -1), value = True) # include eos token
        action_mask &= mask

    # 获取动作的logits和值
    action_logits, value = self.forward(
        sequence,
        mask = action_mask,
        return_values = return_values
    )        

    # 返回动作和值的对象
    return PPOActionCriticReturn(
        actions,
        sequence,
        mask,
        prompt_mask,
        action_logits,
        value
    )

def forward(
    self,
    x,
    mask = None,
    return_values = True
):
    # 获取动作的logits
    action_logits = self.actor_palm(
        x,
        finetune_scope = self.actor_lora_scope
    )

    # 如果不需要返回值，直接返回动作logits
    if not return_values:
        return action_logits, None

    # 获取评论者的嵌入
    critic_embeds = self.critic_palm(
        x,
        return_only_embedding = True,
        finetune_scope = self.critic_lora_scope
    )

    # 如果使用池化值，计算平均值
    if self.pooled_values:
        critic_embeds = shift(critic_embeds, shift = 1, dim = -2)
        critic_embeds = masked_mean(critic_embeds, mask, dim = 1)

    # 获取值
    values = self.value_head(critic_embeds)

    # 返回动作logits和值
    return action_logits, values
# 定义一个命名元组 Memory，包含了序列、提示掩码、掩码、动作概率、动作对数概率、奖励和价值
Memory = namedtuple('Memory', [
    'sequence',
    'prompt_mask',
    'mask',
    'action_prob',
    'action_log_prob',
    'reward',
    'value'
])

# ExperienceDataset 类，继承自 Dataset 类，用于处理经验数据集
class ExperienceDataset(Dataset):
    def __init__(
        self,
        data: List[torch.Tensor],  # 接受一个包含 torch.Tensor 的列表作为数据
        device = None  # 设备参数，默认为 None
    ):
        super().__init__()
        self.data = data  # 存储数据
        self.device = device  # 存储设备信息

    def __len__(self):
        return self.data[0].shape[0]  # 返回数据的第一个维度大小

    def __getitem__(self, ind):
        return tuple(map(lambda t: t[ind].to(self.device), self.data))  # 返回指定索引的数据，并将其移动到指定设备上

# 创建数据加载器函数，接受数据、批量大小、是否打乱数据、设备等参数
def create_dataloader(data, batch_size, shuffle = True, device = None, **kwargs):
    ds = ExperienceDataset(data, device = device)  # 创建 ExperienceDataset 实例
    return DataLoader(ds, batch_size = batch_size, shuffle = shuffle, **kwargs)  # 返回 DataLoader 实例

# 辅助函数

# 判断值是否存在
def exists(val):
    return val is not None

# 返回默认值
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# 对张量进行归一化处理
def masked_normalize(t, eps = 1e-5, mask = None, dim = None):
    dim = default(dim, tuple(range(t.ndim)))  # 获取维度信息
    kwargs = dict(dim = dim, keepdim = True)

    mean = masked_mean(t, mask = mask, **kwargs)  # 计算均值
    mean_centered = t - mean  # 中心化
    var = masked_mean(mean_centered ** 2, mask = mask, **kwargs)  # 计算方差

    return mean_centered * var.clamp(min = eps).rsqrt()  # 返回归一化后的结果

# 对序列进行固定填充
def pad_sequence_fixed(sequences, *args, **kwargs):
    first_el = sequences[0]  # 获取第一个元素
    has_no_dimension = first_el.ndim == 0  # 判断是否没有维度

    # 如果没有维度，添加一个维度
    if has_no_dimension:
        sequences = tuple(map(lambda t: t[None], sequences))

    out = pad_sequence(sequences, *args, **kwargs)  # 使用 pad_sequence 进行填充

    if has_no_dimension:
        out = rearrange(out, '... 1 -> ...')  # 重新排列维度

    return out

# 计算张量的对数
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

# 计算对数概率
def log_prob(prob, indices):
    assert prob.shape[:2] == indices.shape, f'preceding shapes of prob {prob.shape[:2]} and indices {indices.shape} must match'
    return log(prob.gather(-1, indices[..., None])).squeeze(-1)

# 对张量进行移位
def shift(t, value = 0, shift = 1, dim = -1):
    zeros = (0, 0) * (-dim - 1)
    return F.pad(t, (*zeros, shift, -shift), value = value)

# 计算掩码熵
def masked_entropy(prob, dim = -1, mask = None):
    entropies = (prob * log(prob)).sum(dim = -1)
    return masked_mean(entropies, mask = mask).mean()

# 计算掩码 KL 散度
def masked_kl_div(prob1, prob2, mask = None, reduce_batch = False):
    """
    need to account for variable sequence lengths, therefore not using the built-in functional version
    """
    kl_divs = (prob1 * (log(prob1) - log(prob2))).sum(dim = -1)
    loss = masked_mean(kl_divs, mask)

    if reduce_batch:
        return loss.mean()

    return loss

# 计算截断值损失
def clipped_value_loss(values, rewards, old_values, clip):
    value_clipped = old_values + (values - old_values).clamp(-clip, clip)
    value_loss_1 = (value_clipped.flatten() - rewards) ** 2
    value_loss_2 = (values.flatten() - rewards) ** 2
    return torch.mean(torch.max(value_loss_1, value_loss_2))

# RLHFTrainer 类，继承自 nn.Module
class RLHFTrainer(nn.Module):
    # 初始化函数，设置模型的各种参数和超参数
    def __init__(
        self,
        *,
        prompts: Optional[List[str]] = None,  # 提示语列表
        prompts_path: Optional[str] = None,  # 提示语文件路径
        prompt_token_ids: Optional[torch.Tensor] = None,  # 提示语的token ids
        tokenizer: Callable = None,  # 分词器
        palm: PaLM,  # 主模型
        reward_model: RewardModel,  # 奖励模型
        critic_palm: Optional[PaLM] = None,  # 评论者模型
        actor_critic: Optional[ActorCritic] = None,  # 演员评论者模型
        actor_lr = 1e-4,  # 演员学习率
        critic_lr = 1e-4,  # 评论者学习率
        actor_wd = 0.,  # 演员权重衰减
        critic_wd = 0.,  # 评论者权重衰减
        actor_adam_eps = 1e-7,  # 演员Adam优化器epsilon
        critic_adam_eps = 1e-7,  # 评论者Adam优化器epsilon
        actor_lora = True,  # 演员是否使用LoRA
        critic_lora = True,  # 评论者是否使用LoRA
        actor_lora_r = 8,  # 演员LoRA半径
        critic_lora_r = 8,  # 评论者LoRA半径
        critic_pooled_values = True,  # 评论者是否使用池化值
        actor_dropout = 0.,  # 演员Dropout
        critic_dropout = 0.,  # 评论者Dropout
        betas = (0.9, 0.999),  # Adam优化器betas
        max_norm = None,  # 梯度裁剪最大范数
        eps_clip = 0.2,  # PPO算法epsilon裁剪
        value_clip = 0.4,  # 值函数裁剪
        beta_s = .01,  # beta_s参数
        pad_value = 0.,  # token填充值
        minibatch_size = 16,  # 小批量大小
        epochs = 1,  # 训练轮数
        kl_div_loss_weight = 0.1,  # KL散度损失权重
        accelerate_kwargs: dict = {},  # 加速器参数
        use_lion = False  # 是否使用LION
    ):
        # 调用父类初始化函数
        super().__init__()

        # 初始化加速器
        self.accelerate = Accelerator(**accelerate_kwargs)

        # 处理提示语到token ids的转换
        assert (exists(prompts) + exists(prompts_path) + exists(prompt_token_ids)) == 1

        if exists(prompts_path):
            path = Path(prompts_path)
            prompts = path.read_text().split('\n')

        if exists(prompts):
            assert len(prompts) > 0, 'no prompts'
            assert exists(tokenizer), 'tokenizer must be passed in if raw text prompts are given'
            prompt_token_ids = tokenizer(prompts)

        self.pad_value = pad_value  # token填充值
        self.num_prompts = prompt_token_ids.shape[0]  # 提示语数量
        self.register_buffer('prompt_token_ids', prompt_token_ids)  # 注册提示语token ids

        # 初始化模型
        self.palm = palm

        if not exists(actor_critic):
            actor_critic = ActorCritic(
                palm = palm,
                critic_palm = critic_palm,
                actor_lora = actor_lora,
                critic_lora = critic_lora,
                actor_lora_r = actor_lora_r,
                critic_lora_r = critic_lora_r,
                pooled_values = critic_pooled_values,
                actor_dropout = actor_dropout,
                critic_dropout = critic_dropout
            ).to(palm.device)

        self.actor_critic = actor_critic  # 演员评论者模型

        self.reward_model = reward_model.eval()  # 奖励模型

        # 训练超参数
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.max_norm = max_norm
        self.kl_div_loss_weight = kl_div_loss_weight

        # 优化器
        self.actor_optim = get_optimizer(actor_critic.actor_parameters(), lr = actor_lr, wd = actor_wd, betas = betas, eps = actor_adam_eps, use_lion = use_lion)
        self.critic_optim = get_optimizer(actor_critic.critic_parameters(), lr = critic_lr, wd = critic_wd, betas = betas, eps = critic_adam_eps, use_lion = use_lion)

        # PPO算法超参数
        self.eps_clip = eps_clip
        self.value_clip = value_clip
        self.beta_s = beta_s

        # 准备加速器
        (
            self.actor_critic,
            self.reward_model,
            self.actor_optim,
            self.critic_optim
        ) = self.accelerate.prepare(
            self.actor_critic,
            self.reward_model,
            self.actor_optim,
            self.critic_optim
        )

    # 打印函数
    def print(self, msg):
        return self.accelerate.print(msg)

    # 保存模型参数
    def save(self, filepath = './checkpoint.pt'):
        torch.save(self.actor_critic.state_dict(), filepath)

    # 加载模型参数
    def load(self, filepath = './checkpoint.pt'):
        state_dict = torch.load(filepath)
        self.actor_critic.load_state_dict(state_dict)

    # 设备属性
    @property
    def device(self):
        return self.accelerate.device

    # 禁用梯度计算
    @torch.no_grad()
    # 定义一个生成器函数，用于生成文本序列
    def generate(
        self,
        max_seq_len,
        *args,
        prompt,
        num_samples = 4,  # 每个提示生成4个样本，选择具有最高奖励的一个
        **kwargs
    ):
        # 断言只有一个提示允许在同一时间
        assert prompt.ndim == 1, 'only one prompt allowed at a time for now'
        # 复制提示以匹配生成的样本数量
        prompt = repeat(prompt, 'n -> b n', b = num_samples)

        # 获取未加速的 actor_critic 模型
        actor_critic = self.accelerate.unwrap_model(self.actor_critic)
        # 获取未加速的 reward_model 模型
        reward_model = self.accelerate.unwrap_model(self.reward_model)

        # 设置 actor_critic 模型为评估模式
        actor_critic.eval()

        # 生成动作、序列、掩码、提示掩码、动作概率等信息
        (
            actions,
            sequences,
            mask,
            prompt_mask,
            action_logits,
            _
        ) = actor_critic.generate(
            prompt,
            *args,
            max_seq_len = max_seq_len,
            return_values = False,
            **kwargs
        )

        # 使用奖励模型计算奖励
        rewards = reward_model(
            sequences,
            prompt_mask = prompt_mask,
            mask = mask,
            sample = True
        )

        # 选择具有最高奖励的序列索引
        best_sequence_index = rewards.topk(1, dim = -1).indices

        # 获取最佳序列
        best_sequence = sequences[best_sequence_index]
        # 重新排列最佳序列的维度
        best_sequence = rearrange(best_sequence, '1 ... -> ...')

        # 返回最佳序列
        return best_sequence

    # 定义一个学习函数，用于学习记忆
    def learn(
        self,
        memories: Deque[Memory]
    
    # 定义一个训练函数，用于训练模型
    def train(
        self,
        num_episodes = 50000,
        max_timesteps = 500,
        update_timesteps = 5000,
        max_batch_size = 16,
        max_seq_len = 2048,
        eos_token = None,
        temperature = 1.
        ):
        # 获取当前环境设备
        device = self.device

        # 初始化时间步长和记忆队列
        time = 0
        memories = deque([])

        # 循环执行一定数量的 episodes
        for eps in tqdm(range(num_episodes), desc='episodes'):
            # 在每个 episode 中执行一定数量的时间步长
            for timestep in range(max_timesteps):
                time += 1

                # 选择一组随机状态（提示）并获取动作（从 palm 中采样的序列以及动作概率）
                # 使用奖励模型计算奖励并存储

                # 随机选择一个提示的索引
                rand_prompt_index = randrange(0, self.num_prompts)

                # 获取状态（提示）的 token ID
                state = self.prompt_token_ids[rand_prompt_index]

                # 去除状态中的填充
                state_mask = state != self.pad_value
                state = state[state_mask]

                # 生成预测序列
                (
                    actions,
                    sequence,
                    mask,
                    prompt_mask,
                    action_logits,
                    value
                ) = self.actor_critic.generate(
                    rearrange(state, 'n -> 1 n'),
                    max_seq_len=max_seq_len,
                    eos_token=eos_token,
                    temperature=temperature,
                    return_values=True
                )
                action_logits = shift(action_logits, shift=1, dim=-2)  # 需要沿着序列维度移动 1，因为动作从最后一个提示（状态）标记开始

                action_prob = action_logits.softmax(dim=-1)

                action_len = actions.shape[-1]
                action_log_prob = log_prob(action_prob, sequence)
                action_log_prob = action_log_prob[:, -action_len:]

                actions = rearrange(actions, '1 ... -> ...')

                # 使用经过监督训练的奖励模型获取奖励
                sequence = torch.cat((state, actions), dim=0)

                prompt_length = len(state)
                prompt_mask = torch.arange(sequence.shape[-1], device=device) < prompt_length

                sequence = rearrange(sequence, 'n -> 1 n')
                prompt_mask = rearrange(prompt_mask, 'n -> 1 n')
                mask = default(mask, lambda: torch.ones(sequence.shape, dtype=torch.bool, device=device))

                reward = self.reward_model(
                    sequence,
                    prompt_mask=prompt_mask,
                    mask=mask,
                    sample=True
                )

                detach_to_cpu_ = lambda t: rearrange(t.detach().cpu(), '1 ... -> ...')

                # 存储用于学习的记忆
                memories.append(Memory(*map(detach_to_cpu_, (
                    sequence,
                    prompt_mask,
                    mask,
                    action_prob,
                    action_log_prob,
                    reward,
                    value
                )))

                # 从存储的记忆中学习
                if time % update_timesteps == 0:
                    self.learn(memories)
                    memories.clear()

        print('rlhf training complete')
```