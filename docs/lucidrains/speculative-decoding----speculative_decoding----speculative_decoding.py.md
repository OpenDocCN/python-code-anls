# `.\lucidrains\speculative-decoding\speculative_decoding\speculative_decoding.py`

```py
import math
# 导入数学库

import torch
# 导入 PyTorch 库
from torch.nn import Module, ModuleList
# 从 PyTorch 中导入 Module 和 ModuleList
from torch import nn, einsum, Tensor
# 从 PyTorch 中导入 nn、einsum 和 Tensor
import torch.nn.functional as F
# 从 PyTorch 中导入 nn.functional，并简称为 F

from rotary_embedding_torch import RotaryEmbedding
# 导入自定义的 RotaryEmbedding 模块
from beartype import beartype
# 导入 beartype 模块，用于类型检查

from collections import namedtuple
# 导入 namedtuple 模块

from einops import rearrange
# 导入 einops 中的 rearrange 函数

# constants

Cache = namedtuple('Cache', ['cached_kvs', 'embeds'])
# 定义一个命名元组 Cache，包含 cached_kvs 和 embeds 两个字段

# helper functions

def exists(val):
    return val is not None
# 定义函数 exists，用于判断值是否存在

def default(val, d):
    return val if exists(val) else d
# 定义函数 default，用于返回值或默认值

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))
# 定义函数 log，用于计算对数并进行截断处理

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))
# 定义函数 gumbel_noise，生成 Gumbel 噪声

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)
# 定义函数 gumbel_sample，用于根据温度参数进行采样

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(-1, ind, val)
    return probs
# 定义函数 top_k，用于获取前 k 个最大值并进行处理

# rotary embeddings

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    # 定义 RotaryEmbedding 类，用于生成旋转嵌入

    def forward(self, seq_len):
        t = torch.arange(seq_len, device = self.inv_freq.device).type_as(self.inv_freq)
        freqs = einsum('i, j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)
        return freqs
    # 前向传播函数，生成旋转嵌入

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)
# 定义函数 rotate_half，用于旋转张量的一半

def apply_rotary_pos_emb(pos, t):
    seq_len = t.shape[-2]
    pos = pos[-seq_len:, :]
    return t * pos.cos() + rotate_half(t) * pos.sin()
# 定义函数 apply_rotary_pos_emb，应用旋转位置嵌入到张量中

# different decoding strategies

@torch.no_grad()
def base_decoding(
    net: Module,
    prompt: Tensor,
    seq_len: int,
    temperature = 1.,
    filter_thres = 0.9,
):
    prompt_seq_len, out = prompt.shape[-1], prompt.clone()
    sample_num_times = max(0, seq_len - prompt_seq_len)

    cache = None

    for _ in range(sample_num_times):
        logits, cache = net(out, cache = cache, return_cache = True)
        logits = logits[:, -1]

        logits = top_k(logits, thres = filter_thres)
        sample = gumbel_sample(logits, temperature = temperature, dim = -1)

        out = torch.cat((out, sample[..., None]), dim = -1)

    return out[..., prompt_seq_len:]
# 定义函数 base_decoding，基础解码策略

# speculative decoding functions

def safe_div(num, den, eps = 1e-10):
    return num / max(den, eps)
# 定义函数 safe_div，安全除法

def find_first_true_index(bool_tensor, dim = -1):
    return (bool_tensor.cumsum(dim = dim) == 0).sum(dim = dim)
# 定义函数 find_first_true_index，查找第一个为真的索引

@torch.no_grad()
def speculative_decoding(
    net: Module,
    small_net: Module,
    prompt: Tensor,
    seq_len: int,
    gamma: int = 5,
    temperature = 1.,
    filter_thres = 0.9,
    lenience = 1.,
    pad_id = 0
):
    """
    eq. algorithm 1 in paper https://arxiv.org/abs/2211.17192
    """
    # 假设性解码函数，参考论文中的算法1

    batch, prompt_seq_len, out, device = *prompt.shape, prompt.clone(), prompt.device
    sample_num_times = max(0, seq_len - prompt_seq_len)

    cache = None
    small_cache = None

    num_steps = 0
    total_accepted = 0

    batch_range = torch.arange(batch, device = device, dtype = torch.long)[..., None]
    seq_lens = torch.full((batch,), prompt_seq_len, device = device, dtype = torch.long)

    # now left align

    num_pad_left = out.shape[-1] - seq_lens
    max_pad_left = num_pad_left.amax()
    out = F.pad(out, (0, max_pad_left), value = pad_id)

    seq_len_range = torch.arange(seq_len, device = device, dtype = torch.long)
    out = out[batch_range, seq_len_range + num_pad_left[..., None]]

    return out[..., prompt_seq_len:], total_accepted / num_steps
# 定义函数 speculative_decoding，假设性解码函数

@torch.no_grad()
def speculative_decoding_with_same_model(
    net: Module,
    prompt: Tensor,
    seq_len: int,
    gamma: int = 5,
    temperature = 1.,
    filter_thres = 0.9,
    lenience = 1.,
    pad_id = 0
):
    """
    eq. algorithm 1 in paper https://arxiv.org/abs/2211.17192
    """
    # 假设性解码函数，参考论文中的算法1
    # 将 prompt 的形状解包为 batch, prompt_seq_len, out, device
    batch, prompt_seq_len, out, device = *prompt.shape, prompt.clone(), prompt.device
    # 计算需要采样的次数
    sample_num_times = max(0, seq_len - prompt_seq_len)

    # 初始化缓存变量
    cache = None
    small_cache = None

    # 初始化步数和接受总数
    num_steps = 0
    total_accepted = 0

    # 创建 batch_range 和 seq_lens 张量
    batch_range = torch.arange(batch, device=device, dtype=torch.long)[..., None]
    seq_lens = torch.full((batch,), prompt_seq_len, device=device, dtype=torch.long)

    # 对输出进行左对齐填充
    num_pad_left = out.shape[-1] - seq_lens
    max_pad_left = num_pad_left.amax()
    out = F.pad(out, (0, max_pad_left), value=pad_id)

    # 选择左对齐后的输出
    seq_len_range = torch.arange(seq_len, device=device, dtype=torch.long)
    out = out[batch_range, seq_len_range + num_pad_left[..., None]]

    # 返回处理后的输出和接受率
    return out[..., prompt_seq_len:], total_accepted / num_steps
# 定义一个模块，用于对输入进行 RMS 归一化处理
class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

# 定义一个模块，实现自注意力机制
class CausalAttention(Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        dim_inner = dim_head * heads

        self.norm = RMSNorm(dim)

        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(
        self,
        x,
        cache = None,
        context_mask = None,
        rotary_emb = None
    ):
        h, device = self.heads, x.device

        x = self.norm(x)

        q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv = 3, h = h)

        if exists(cache):
            ck, cv = cache.unbind(dim = 1)
            k = torch.cat((ck, k), dim = -2)
            v = torch.cat((cv, v), dim = -2)

        cached_kv = torch.stack((k, v), dim = 1)

        if exists(rotary_emb):
            q = apply_rotary_pos_emb(rotary_emb, q)
            k = apply_rotary_pos_emb(rotary_emb, k)

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        i, j = sim.shape[-2:]
        causal_mask = torch.ones((i, j), device = device, dtype = torch.bool).triu(j - i + 1)

        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        if exists(context_mask):
            context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~context_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out, cached_kv

# 定义一个前馈神经网络模块
def FeedForward(dim, mult = 4):
    dim_inner = dim * mult
    return nn.Sequential(
        RMSNorm(dim),
        nn.Linear(dim, dim_inner),
        nn.GELU(),
        nn.Linear(dim_inner, dim)
    )

# 主要的解码器类
class Decoder(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        heads = 8,
        dim_head = 64,
        ff_mult = 4,
        ignore_index = -1,
        early_exit_layer = None,
        early_exit_extra_transformer_blocks = 0,
        detach_early_exit_hiddens = False
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.layers = ModuleList([])

        self.rotary_emb = RotaryEmbedding(dim = dim_head)

        # 创建多个解码器层，每个层包含自注意力和前馈神经网络模块
        for _ in range(depth):
            self.layers.append(ModuleList([
                CausalAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        # 输出层，将解码器输出映射到标记空间
        self.to_logits = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, num_tokens, bias = False)
        )

        self.detach_early_exit_hiddens = detach_early_exit_hiddens
        self.early_exit_layer = early_exit_layer
        self.to_early_exit_logits = None
        self.early_exit_transformer_blocks = ModuleList([])

        # 如果存在提前退出层，则创建额外的解码器层
        if exists(early_exit_layer):
            for _ in range(early_exit_extra_transformer_blocks):
                self.early_exit_transformer_blocks.append(ModuleList([
                    CausalAttention(dim = dim, dim_head = dim_head, heads = heads, rotary_emb = rotary_emb),
                    FeedForward(dim = dim, mult = ff_mult)
                ]))

            # 提前退出层的输出层
            self.to_early_exit_logits = nn.Sequential(
                RMSNorm(dim),
                nn.Linear(dim, num_tokens, bias = False)
            )

        self.ignore_index = ignore_index
    # 定义一个方法用于前向传播
    def forward(
        self,
        x,
        return_loss = False,  # 是否返回损失，默认为False
        return_cache = False,  # 是否返回缓存，默认为False
        seq_start_pos = None,  # 序列起始位置，默认为None
        cache = None,  # 缓存，默认为None
        early_exit_cache = None,  # 提前退出缓存，默认为None
        return_early_exit_only = False,  # 是否仅返回提前退出，默认为False
        start_from_early_exit_hiddens = False  # 是否从提前退出隐藏状态开始，默认为False
```