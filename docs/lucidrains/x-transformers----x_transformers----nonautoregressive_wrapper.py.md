# `.\lucidrains\x-transformers\x_transformers\nonautoregressive_wrapper.py`

```py
import math
from random import random
from contextlib import nullcontext
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange, repeat, pack, unpack

from x_transformers.x_transformers import TransformerWrapper
from typing import Optional

# 定义一个命名元组 Losses，包含 loss、generator_loss 和 critic_loss 三个字段
Losses = namedtuple('Losses', ['loss', 'generator_loss', 'critic_loss'])

# 辅助函数

# 判断值是否存在
def exists(val):
    return val is not None

# 如果值存在则返回该值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 采样辅助函数

# 从 logits 中选择 top-k 的概率值
def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = logits.topk(k, dim = -1)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(2, ind, val)
    return probs

# 计算对数
def log(t, eps = 1e-10):
    return torch.log(t + eps)

# 生成 Gumbel 噪声
def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

# 从 Gumbel 噪声中采样
def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

# 概率辅助函数

# 根据概率值进行采样
def sample_prob(prob):
    return random() < prob

# 抛硬币，返回 True 或 False
def coin_flip():
    return sample_prob(0.5)

# 张量辅助函数

# 根据掩码和概率值获取子集掩码
def get_mask_subset_prob(mask, prob, min_mask = 0):
    batch, seq, device = *mask.shape, mask.device
    num_to_mask = (mask.sum(dim = -1, keepdim = True) * prob).clamp(min = min_mask)
    logits = torch.rand((batch, seq), device = device)
    logits = logits.masked_fill(~mask, -1)

    randperm = logits.argsort(dim = -1).argsort(dim = -1).float()

    num_padding = (~mask).sum(dim = -1, keepdim = True)
    randperm -= num_padding

    subset_mask = randperm < num_to_mask
    subset_mask.masked_fill_(~mask, False)
    return subset_mask

# 调度函数

# 线性调度函数
def linear_schedule(t):
    return 1 - t

# 余弦调度函数
def cosine_schedule(t):
    """ https://arxiv.org/abs/2202.04200 """
    return torch.cos(t * math.pi / 2)

# 自标记评论者类
# 受 Nijkamp 等人启发 - https://aclanthology.org/2021.naacl-main.409/

class SelfCritic(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

        dim = net.attn_layers.dim
        self.to_logits = nn.Linear(dim, 1)

    def forward(self, x):
        embed = self.net(x, return_embeddings = True)
        return self.to_logits(embed)

# 非自回归包装器类
# 参考 https://arxiv.org/abs/1904.09324 和 https://arxiv.org/abs/2202.04200

class NonAutoregressiveWrapper(nn.Module):
    def __init__(
        self,
        net,
        *,
        mask_id,
        steps = 18,
        self_cond = False,
        self_cond_train_prob = 0.75,
        no_replace_prob = 0.15,          # 有多少百分比的标记将保持不变，原始 MLM 论文中进行了这样的操作
        random_token_prob = 0.1,         # 有多少百分比的标记将被替换为随机标记，原始 MLM 论文中进行了这样的操作
        schedule = 'linear',
        can_mask_prev_unmasked = False,  # 当取消掩码时，是否可以重新掩码先前未掩码的标记
        token_critic: Optional[TransformerWrapper] = None,
        self_token_critic = False,
        critic_loss_weight = 1.
        # 调用父类的构造函数
        super().__init__()
        # 断言确保 self_token_critic 为假且 token_critic 不存在
        assert not (self_token_critic and exists(token_critic))

        # 设置网络模型
        self.net = net

        # 获取嵌入维度和词汇表大小
        dim = net.emb_dim
        self.dim = dim
        self.num_tokens = net.num_tokens

        # 设置掩码的标识符
        self.mask_id = mask_id

        # 设置不替换概率和随机替换概率
        self.no_replace_prob = no_replace_prob
        self.random_token_prob = random_token_prob

        # 获取最大序列长度和步数
        self.max_seq_len = net.max_seq_len
        self.steps = steps

        # 根据调度方式设置调度函数
        if callable(schedule):
            self.schedule_fn = schedule
        if schedule == 'linear':
            self.schedule_fn = linear_schedule
        elif schedule == 'cosine':
            self.schedule_fn = cosine_schedule
        else:
            raise ValueError(f'invalid schedule {schedule}')

        # 设置是否可以掩盖之前未掩盖的标记
        self.can_mask_prev_unmasked = can_mask_prev_unmasked

        # 设置自条件
        self.self_cond = self_cond

        # 如果存在自条件，则初始化空嵌入和线性层
        if self_cond:
            self.null_embed = nn.Parameter(torch.randn(dim))
            self.to_self_cond = nn.Linear(dim, dim, bias=False) if self_cond else None
            self.self_cond_train_prob = self_cond_train_prob

        # 设置标记评论者
        self.token_critic = token_critic

        # 如果存在 self_token_critic，则初始化 SelfCritic 类
        if self_token_critic:
            self.token_critic = SelfCritic(net)

        # 设置评论者损失权重
        self.critic_loss_weight = critic_loss_weight

    # 生成函数，不进行梯度计算
    @torch.no_grad()
    def generate(
        self,
        batch_size=None,
        start_temperature=1.,
        filter_thres=0.7,
        noise_level_scale=1.,
        **kwargs
    ):
        # 检查是否存在 batch_size 变量，如果不存在则设置为默认值 1
        sample_one = not exists(batch_size)
        batch_size = default(batch_size, 1)

        # 获取神经网络参数的设备信息
        device = next(self.net.parameters()).device

        # 保存当前模型的训练状态，并将模型设置为评估模式
        was_training = self.training
        self.eval()

        # 在0到1之间生成self.steps + 1个时间点
        times = torch.linspace(0., 1., self.steps + 1)

        # 初始化序列和掩码，将序列初始值设为mask_id，掩码初始值设为True
        shape = (batch_size, self.max_seq_len)
        seq = torch.full(shape, self.mask_id, device=device)
        mask = torch.full(shape, True, device=device)

        # 计算所有掩码的数量
        all_mask_num_tokens = (self.schedule_fn(times[1:]) * self.max_seq_len).long()

        # 判断是否有自我条件
        has_self_cond = self.self_cond
        last_embed = self.null_embed if has_self_cond else None

        # 逐步解除掩码
        for mask_num_tokens, steps_until_x0 in zip(all_mask_num_tokens.tolist(), reversed(range(self.steps))):

            # 如果有自我条件，则计算自我条件
            self_cond = self.to_self_cond(last_embed) if has_self_cond else None

            # 获取神经网络的输出logits和embeds
            logits, embeds = self.net(
                seq,
                sum_embeds=self_cond,
                return_logits_and_embeddings=True,
                **kwargs
            )

            # 如果有自我条件，则更新last_embed
            if has_self_cond:
                last_embed = embeds

            # 如果存在filter_thres，则对logits进行top_k筛选
            if exists(filter_thres):
                logits = top_k(logits, filter_thres)

            # 计算温度和概率
            annealing_scale = steps_until_x0 / self.steps
            temperature = start_temperature * annealing_scale
            probs = (logits / max(temperature, 1e-3)).softmax(dim=-1)

            # 从logits中采样得到sampled_ids
            sampled_ids = gumbel_sample(logits, temperature=max(temperature, 1e-3))

            # 根据掩码mask更新序列seq
            seq = torch.where(mask, sampled_ids, seq)

            # 如果存在token_critic，则计算scores
            if exists(self.token_critic):
                scores = self.token_critic(seq)
                scores = rearrange(scores, 'b n 1 -> b n')
                scores = scores + noise_level_scale * gumbel_noise(scores) * annealing_scale
            else:
                scores = 1 - logits.softmax(dim=-1)
                scores = scores.gather(2, rearrange(sampled_ids, 'b n -> b n 1'))
                scores = rearrange(scores, 'b n 1 -> b n')

            # 如果mask_num_tokens为0，则跳过
            if mask_num_tokens == 0:
                pass

            # 如果不允许掩盖之前未掩盖的标记，则将scores中的非掩码位置设为最小值
            if not self.can_mask_prev_unmasked:
                scores = scores.masked_fill(~mask, -torch.finfo(scores.dtype).max)

            # 根据scores中的topk值更新掩码mask
            mask_indices = scores.topk(mask_num_tokens, dim=-1).indices
            mask = torch.zeros_like(scores, dtype=torch.bool).scatter(1, mask_indices, True)
            seq = seq.masked_fill(mask, self.mask_id)

        # 恢复模型的训练状态
        self.train(was_training)

        # 如果sample_one为True，则将seq重新排列
        if sample_one:
            seq = rearrange(seq, '1 n -> n')

        # 返回生成的序列seq
        return seq

    # 定义前向传播函数
    def forward(
        self,
        x,
        only_train_generator=False,
        only_train_critic=False,
        generator_sample_temperature=None,
        **kwargs
    ):
        # 获取输入张量 x 的形状、维度大小 n 和设备信息
        b, n, device = *x.shape, x.device
        # 断言维度大小 n 等于 self.max_seq_len
        assert n == self.max_seq_len

        # 复制原始序列 x
        orig_seq = x.clone()

        # 生成随机数 rand_times，范围在 [0, 1] 之间
        rand_times = torch.empty(b, device = device).uniform_(0, 1)
        # 生成随机排列的索引 batched_randperm
        batched_randperm = torch.rand((b, n), device = device).argsort(dim = -1).float()

        # 根据随机数 rand_times 调用 schedule_fn 函数生成随机概率 rand_probs
        rand_probs = self.schedule_fn(rand_times)
        # 计算每个样本应mask的 token 数量
        num_tokens_mask = (rand_probs * n).clamp(min = 1.)
        # 生成 mask，用于随机 mask token
        mask = batched_randperm < rearrange(num_tokens_mask, 'b -> b 1')

        # 为了确保所有 token 都产生嵌入，而不仅仅是 [mask] 输入中的 token，与经典的 BERT MLM 论文中所做的不同
        # 可能需要为了 self-conditioning（对嵌入的自我调节）良好运作
        replace_mask_id_mask = mask.clone()
        frac_seq_left = 1.

        # 如果 self.no_replace_prob 大于 0 且 coin_flip() 为真
        if self.no_replace_prob > 0. and coin_flip():
            frac_seq_left -= self.no_replace_prob

            # 获取 no_replace_prob_mask，用于不替换 token
            no_replace_prob_mask = get_mask_subset_prob(mask, self.no_replace_prob)
            replace_mask_id_mask &= ~no_replace_prob_mask

        # 如果 self.random_token_prob 大于 0 且 coin_flip() 为真
        if self.random_token_prob > 0. and coin_flip():
            # 获取 random_token_prob_mask，用于随机替换 token
            random_token_prob_mask = get_mask_subset_prob(replace_mask_id_mask, self.random_token_prob * frac_seq_left)
            # 生成随机 token
            random_tokens = torch.randint(0, self.num_tokens, (b, n), device = device)

            # 根据 random_token_prob_mask 替换 token
            x = torch.where(random_token_prob_mask, random_tokens, x)
            replace_mask_id_mask &= ~random_token_prob_mask

        # 根据 replace_mask_id_mask 进行 mask 操作，用 self.mask_id 替换 token
        masked = torch.where(replace_mask_id_mask, self.mask_id, x)

        # self conditioning

        # 如果 self.self_cond 为真
        if self.self_cond:
            self_cond = self.null_embed

            # 如果以 self_cond_train_prob 的概率进行采样
            if sample_prob(self.self_cond_train_prob):
                with torch.no_grad():
                    # 通过网络获取 self_cond
                    self_cond = self.net(masked, return_embeddings = True, **kwargs).detach()

            # 更新 kwargs，添加 sum_embeds 信息
            kwargs.update(sum_embeds = self.to_self_cond(self_cond))

        # logits

        # 根据 only_train_critic 决定 context
        context = torch.no_grad if only_train_critic else nullcontext

        with context():
            # 获取 logits
            logits = self.net(masked, **kwargs)

        # 交叉熵损失
        loss = F.cross_entropy(
            logits[mask],
            orig_seq[mask]
        )

        # 如果不存在 token_critic 或者只训练生成器
        if not exists(self.token_critic) or only_train_generator:
            return Losses(loss, loss, None)

        # 采样生成的 token
        sampled_ids = gumbel_sample(logits, temperature = default(generator_sample_temperature, random()))
        generated = torch.where(mask, sampled_ids, orig_seq)

        # 获取 critic_logits 和 critic_labels
        critic_logits = self.token_critic(generated)
        critic_labels = (sampled_ids != orig_seq).float()

        # critic 损失
        critic_loss = F.binary_cross_entropy_with_logits(
            rearrange(critic_logits, '... 1 -> ...'),
            critic_labels
        )

        # 根据研究人员想要训练的内容确定要返回的损失
        if only_train_critic:
            total_loss = critic_loss
            loss = None
        else:
            total_loss = loss + critic_loss * self.critic_loss_weight

        return Losses(total_loss, loss,  critic_loss)
```