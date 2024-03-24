# `.\lucidrains\pause-transformer\pause_transformer\pause_transformer.py`

```
import torch
import torch.nn.functional as F
from torch import nn, Tensor, einsum
from torch.nn import Module, ModuleList, Sequential

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

# functions

# 检查变量是否存在的函数
def exists(v):
    return v is not None

# tensor functions

# 计算张量的对数，避免出现负无穷
def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

# 计算张量的熵
def entropy(t, dim = -1):
    prob = t.softmax(dim = dim)
    return (prob * log(prob)).sum(dim = dim)

# norm

# RMS 归一化
class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

# cheap relative positions
# from Peng Bo's RWKV

# 移动 token 的模块
class ShiftTokens(Module):
    def forward(self, x):
        x, x_shift = x.chunk(2, dim = -1)
        x_shift = F.pad(x_shift, (0, 0, 1, -1), value = 0.)
        return torch.cat((x, x_shift), dim = -1)

# feedforward

# 前馈神经网络
def FeedForward(dim, mult = 4):
    dim_inner = int(dim * mult)
    return Sequential(
        ShiftTokens(),
        RMSNorm(dim),
        nn.Linear(dim, dim_inner),
        nn.GELU(),
        nn.Linear(dim_inner, dim)
    )

# CausalAttention

# 因果注意力机制
class CausalAttention(Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.scale = dim ** -0.5
        dim_inner = dim_head * heads

        self.norm = RMSNorm(dim)

        self.to_qkv = Sequential(
            nn.Linear(dim, dim_inner * 3, bias = False),
            Rearrange('b n (qkv h d) -> qkv b h n d', qkv = 3, h = heads)
        )

        self.to_out = Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(dim_inner, dim, bias = False)
        )

    def forward(self, x):
        x = self.norm(x)

        q, k, v = self.to_qkv(x)

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        i, j = sim.shape[-2:]
        causal_mask = torch.ones((i, j), device = x.device).triu(j - i + 1)

        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        return self.to_out(out), torch.stack((k, v))

# integrate previous pause / thinking information

# 整合之前的暂停/思考信息
class IntegratePreviousThought(Module):
    def __init__(self, dim):
        super().__init__()
        self.net =  Sequential(
            RMSNorm(dim),
            Rearrange('b n p d -> b n (p d)'),
            nn.Linear(dim * 2, dim)
        )

    def forward(
        self,
        x,
        pause_tokens,
        pause_lengths = None
    ):
        if not exists(pause_lengths):
            p = pause_tokens[:, :, -1]
        else:
            batch, seq_len = x.shape[:2]
            batch_arange = torch.arange(batch, device = x.device)[:, None, None]
            seq_arange = torch.arange(seq_len, device = x.device)[:, None]
            pause_lengths = pause_lengths[:, :, None]

            p = pause_tokens[batch_arange, seq_arange, pause_lengths]
            p = rearrange(p, '... 1 d -> ... d')

        p = F.pad(p, (0, 0, 1, -1), value = 0.)

        x = torch.stack((x, p), dim = -2)
        out = self.net(x)
        return out

# class

# 暂停 Transformer
class PauseTransformer(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        max_pause_length = 2,
        dim_head = 64,
        heads = 8,
        ff_mult = 4
    ):
        # 调用父类的构造函数
        super().__init__()

        # 创建一个嵌入层，用于将输入的 token 映射为指定维度的向量
        self.token_emb = nn.Embedding(num_tokens, dim)

        # 设置最大暂停长度
        self.max_pause_length = max_pause_length

        # 创建一个可学习的参数，表示暂停的 token
        self.pause_tokens = nn.Parameter(torch.randn(max_pause_length, dim))

        # 创建一个用于整合前一个暂停的模块
        self.integrate_prev_pause = IntegratePreviousThought(dim)

        # 创建一个空的模块列表，用于存储多个层
        self.layers = ModuleList([])

        # 根据指定的深度循环创建多个层
        for _ in range(depth):
            # 每个层包含一个自注意力机制和一个前馈神经网络
            self.layers.append(ModuleList([
                CausalAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        # 创建一个用于输出 logits 的序列模块
        self.to_logits = Sequential(
            RMSNorm(dim),
            nn.Linear(dim, num_tokens, bias = False)
        )

    def forward(
        self,
        x,
        return_loss = False,
        return_logit_entropy = False,
        arrest_pausing = False,
        no_prev_pause_integration = False,
        pause_lengths = None,
        rand_uniform_pausing = False        # this would do random pausing uniform from [0, max_pause_length]
    ):
        """
        einstein notation:
        b - batch
        n - main sequence length
        p - thinking sequence length (pause)
        d - feature dimension
        """

        # 如果需要返回损失，则提取输入序列和标签序列
        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        # 如果不需要阻止暂停
        if not arrest_pausing:
            # 如果需要随机暂停且暂停长度未指定，则随机生成暂停长度
            if rand_uniform_pausing and not exists(pause_lengths):
                pause_lengths = torch.randint(0, self.max_pause_length, x.shape)

        # 获取输入张量的批量大小和序列长度
        batch, seq_len = x.shape

        # 将输入 token 映射为向量
        x = self.token_emb(x)

        # 重复暂停 token，以便与输入张量形状匹配
        p = repeat(self.pause_tokens, 'p d -> b n p d', b = batch, n = seq_len)

        # 如果暂停长度已指定
        if exists(pause_lengths):
            max_pause = int(pause_lengths.amax().item())
            p = p[:, :, :(max_pause + 1)]

            # 如果最大暂停长度为 0，则阻止暂停
            arrest_pausing = max_pause == 0

        # 遍历每个层的自注意力机制和前馈神经网络
        for attn, ff in self.layers:
            attn_out, cached_kvs = attn(x)
            x = x + attn_out
            x = ff(x) + x

            # 如果阻止暂停，则跳过暂停处理
            if arrest_pausing:
                continue

            # 处理思考 token

            x, ps = pack([x, p], 'b n * d')
            x = rearrange(x, '... p d -> (...) p d')

            attn_out, _ = attn(x)

            x = x + attn_out
            x = ff(x) + x

            x = rearrange(x, '(b n) p d -> b n p d', b = batch)
            x, p = unpack(x, ps, 'b n * d')

            # 在训练过程中，允许每个 token 独立思考，不受前一个 token 思考的影响
            if no_prev_pause_integration:
                continue

            # 整合前一个暂停的最后一个 token
            x = x + self.integrate_prev_pause(x, p, pause_lengths)

        # 如果不阻止暂停，则重新打包输入张量和暂停张量
        if not arrest_pausing:
            x, _ = pack([x, p], 'b n * d')

        # 计算 logits
        logits = self.to_logits(x)

        # 如果需要返回 logits 的熵
        if return_logit_entropy:
            return entropy(logits)

        # 如果不需要返回损失，则返回 logits
        if not return_loss:
            return logits

        # 如果阻止暂停，则重新排列 logits 的形状
        if arrest_pausing:
            logits = rearrange(logits, 'b n d -> b d n')
        else:
            labels = repeat(labels, 'b n -> (b p) n', p = self.max_pause_length + 1)
            logits = rearrange(logits, 'b n p d -> (b p) d n')

        # 计算交叉熵损失
        loss = F.cross_entropy(logits, labels)
        return loss
```