# `.\lucidrains\hourglass-transformer-pytorch\hourglass_transformer_pytorch\hourglass_transformer_pytorch.py`

```py
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块、einsum 函数
from torch import nn, einsum
# 从 torch.nn.functional 库中导入 F 模块
import torch.nn.functional as F
# 从 einops 库中导入 rearrange、reduce、repeat 函数

from einops import rearrange, reduce, repeat

# helpers

# 判断变量是否存在的函数
def exists(val):
    return val is not None

# 如果变量存在则返回该变量，否则返回默认值的函数
def default(val, d):
    return val if exists(val) else d

# 将张量填充到指定的倍数的函数
def pad_to_multiple(tensor, multiple, dim = -1, value = 0):
    seq_len = tensor.shape[dim]
    m = seq_len / multiple
    if m.is_integer():
        return tensor
    remainder = math.ceil(m) * multiple - seq_len
    pad_offset = (0,) * (-1 - dim) * 2
    return F.pad(tensor, (*pad_offset, 0, remainder), value = value)

# 将输入值转换为元组的函数
def cast_tuple(val, depth = 1):
    return val if isinstance(val, tuple) else ((val,) * depth)

# factory

# 获取 hourglass transformer 的工厂函数
def get_hourglass_transformer(
    dim,
    *,
    depth,
    shorten_factor,
    attn_resampling,
    updown_sample_type,
    **kwargs
):
    assert isinstance(depth, int) or (isinstance(depth, tuple)  and len(depth) == 3), 'depth must be either an integer or a tuple of 3, indicating (pre_transformer_depth, <nested-hour-glass-config>, post_transformer_depth)'
    assert not (isinstance(depth, int) and shorten_factor), 'there does not need to be a shortening factor when only a single transformer block is indicated (depth of one integer value)'

    if isinstance(depth, int):
        return Transformer(dim = dim, depth = depth, **kwargs)

    return HourglassTransformer(dim = dim, depth = depth, shorten_factor = shorten_factor, attn_resampling = attn_resampling, updown_sample_type = updown_sample_type, **kwargs)

# up and down sample classes

# 下采样类
class NaiveDownsample(nn.Module):
    def __init__(self, shorten_factor):
        super().__init__()
        self.shorten_factor = shorten_factor

    def forward(self, x):
        return reduce(x, 'b (n s) d -> b n d', 'mean', s = self.shorten_factor)

# 上采样类
class NaiveUpsample(nn.Module):
    def __init__(self, shorten_factor):
        super().__init__()
        self.shorten_factor = shorten_factor

    def forward(self, x):
        return repeat(x, 'b n d -> b (n s) d', s = self.shorten_factor)

# 线性下采样类
class LinearDownsample(nn.Module):
    def __init__(self, dim, shorten_factor):
        super().__init__()
        self.proj = nn.Linear(dim * shorten_factor, dim)
        self.shorten_factor = shorten_factor

    def forward(self, x):
        x = rearrange(x, 'b (n s) d -> b n (s d)', s = self.shorten_factor)
        return self.proj(x)

# 线性上采样类
class LinearUpsample(nn.Module):
    def __init__(self, dim, shorten_factor):
        super().__init__()
        self.proj = nn.Linear(dim, dim * shorten_factor)
        self.shorten_factor = shorten_factor

    def forward(self, x):
        x = self.proj(x)
        return rearrange(x, 'b n (s d) -> b (n s) d', s = self.shorten_factor)

# classes

# 预归一化残差类
class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) + x

# 注意力机制类
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        causal = False
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)
    # 定义一个前向传播函数，接受输入 x，上下文 context 和掩码 mask
    def forward(self, x, context = None, mask = None):
        # 获取头数和设备信息
        h, device = self.heads, x.device
        # 如果没有指定上下文，则使用输入 x 作为键值对输入
        kv_input = default(context, x)

        # 将输入 x 分别转换为查询 q，键 k 和值 v
        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1)
        # 将查询 q，键 k 和值 v 重排维度，以适应多头注意力机制
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # 对查询 q 进行缩放
        q = q * self.scale

        # 计算查询和键之间的相似度
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        # 定义掩码值为负无穷
        mask_value = -torch.finfo(sim.dtype).max

        # 如果存在掩码，则将相似度矩阵进行掩码处理
        if exists(mask):
            mask = rearrange(mask, 'b j -> b () () j')
            sim = sim.masked_fill(~mask, mask_value)

        # 如果启用因果性，生成一个上三角掩码矩阵
        if self.causal:
            i, j = sim.shape[-2:]
            mask = torch.ones(i, j, device = device, dtype = torch.bool).triu_(j - i + 1)
            mask = rearrange(mask, 'i j -> () () i j')
            sim = sim.masked_fill(mask, mask_value)

        # 对相似度矩阵进行 softmax 操作
        attn = sim.softmax(dim = -1)
        # 对注意力矩阵进行 dropout 操作
        attn = self.dropout(attn)

        # 根据注意力矩阵计算输出
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # 重排输出维度，以适应后续处理
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        # 返回输出结果
        return self.to_out(out)
def FeedForward(dim, mult = 4, dropout = 0.):
    # 返回一个包含线性层、GELU激活函数、Dropout层和另一个线性层的序列模块
    return nn.Sequential(
        nn.Linear(dim, dim * mult),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )

# transformer classes

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        causal = False,
        heads = 8,
        dim_head = 64,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        norm_out = False
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            # 为每个深度创建一个包含注意力和前馈网络的预层归一化残差模块
            self.layers.append(nn.ModuleList([
                PreNormResidual(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout, causal = causal)),
                PreNormResidual(dim, FeedForward(dim, mult = ff_mult, dropout = ff_dropout))
            ]))

        # 如果需要输出归一化，则使用LayerNorm，否则使用Identity
        self.norm = nn.LayerNorm(dim) if norm_out else nn.Identity()

    def forward(self, x, context = None, mask = None):
        for attn, ff in self.layers:
            # 依次对每个层进行前向传播：注意力层 -> 前馈网络
            x = attn(x, context = context, mask = mask)
            x = ff(x)

        return self.norm(x)

class HourglassTransformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        shorten_factor = 2,
        attn_resampling = True,
        updown_sample_type = 'naive',
        heads = 8,
        dim_head = 64,
        causal = False,
        norm_out = False
    ):
        super().__init__()
        assert len(depth) == 3, 'depth should be a tuple of length 3'
        assert updown_sample_type in {'naive', 'linear'}, 'downsample / upsample type must be either naive (average pool and repeat) or linear (linear projection and reshape)'

        pre_layers_depth, valley_depth, post_layers_depth = depth

        if isinstance(shorten_factor, (tuple, list)):
            shorten_factor, *rest_shorten_factor = shorten_factor
        elif isinstance(valley_depth, int):
            shorten_factor, rest_shorten_factor = shorten_factor, None
        else:
            shorten_factor, rest_shorten_factor = shorten_factor, shorten_factor

        transformer_kwargs = dict(
            dim = dim,
            heads = heads,
            dim_head = dim_head
        )

        self.causal = causal
        self.shorten_factor = shorten_factor

        if updown_sample_type == 'naive':
            # 使用NaiveDownsample和NaiveUpsample进行下采样和上采样
            self.downsample = NaiveDownsample(shorten_factor)
            self.upsample   = NaiveUpsample(shorten_factor)
        elif updown_sample_type == 'linear':
            # 使用LinearDownsample和LinearUpsample进行下采样和上采样
            self.downsample = LinearDownsample(dim, shorten_factor)
            self.upsample   = LinearUpsample(dim, shorten_factor)
        else:
            raise ValueError(f'unknown updown_sample_type keyword value - must be either naive or linear for now')

        # 获取中间层的Transformer
        self.valley_transformer = get_hourglass_transformer(
            shorten_factor = rest_shorten_factor,
            depth = valley_depth,
            attn_resampling = attn_resampling,
            updown_sample_type = updown_sample_type,
            causal = causal,
            **transformer_kwargs
        )

        # 如果需要注意力重采样，则创建前后的Transformer
        self.attn_resampling_pre_valley = Transformer(depth = 1, **transformer_kwargs) if attn_resampling else None
        self.attn_resampling_post_valley = Transformer(depth = 1, **transformer_kwargs) if attn_resampling else None

        # 创建前向Transformer和后向Transformer
        self.pre_transformer = Transformer(depth = pre_layers_depth, causal = causal, **transformer_kwargs)
        self.post_transformer = Transformer(depth = post_layers_depth, causal = causal, **transformer_kwargs)
        # 如果需要输出归一化，则使用LayerNorm，否则使用Identity
        self.norm_out = nn.LayerNorm(dim) if norm_out else nn.Identity()
    def forward(self, x, mask = None):
        # 定义变量含义：b 为 batch 大小，n 为序列长度，d 为特征维度，s 为缩短因子

        s, b, n = self.shorten_factor, *x.shape[:2]

        # hourglass 的上半部分，前置 transformer 层

        x = self.pre_transformer(x, mask = mask)

        # 填充到缩短因子的倍数，为池化做准备

        x = pad_to_multiple(x, s, dim = -2)

        if exists(mask):
            padded_mask = pad_to_multiple(mask, s, dim = -1, value = False)

        # 保存残差，并用于“注意力重采样”在下采样和上采样时

        x_residual = x.clone()

        # 如果是自回归的，进行移位操作，移位量为缩短因子减一

        if self.causal:
            shift = s - 1
            x = F.pad(x, (0, 0, shift, -shift), value = 0.)

            if exists(mask):
                padded_mask = F.pad(padded_mask, (shift, -shift), value = False)

        # 简单的平均池化

        downsampled = self.downsample(x)

        if exists(mask):
            downsampled_mask = reduce(padded_mask, 'b (n s) -> b n', 'sum', s = s) > 0
        else:
            downsampled_mask = None

        # 前谷“注意力重采样” - 每个桶中的池化令牌与预池化的令牌进行关注

        if exists(self.attn_resampling_pre_valley):
            if exists(mask):
                attn_resampling_mask = rearrange(padded_mask, 'b (n s) -> (b n) s', s = s)
            else:
                attn_resampling_mask = None

            downsampled = self.attn_resampling_pre_valley(
                rearrange(downsampled, 'b n d -> (b n) () d'),
                rearrange(x, 'b (n s) d -> (b n) s d', s = s),
                mask = attn_resampling_mask
            )

            downsampled = rearrange(downsampled, '(b n) () d -> b n d', b = b)

        # “谷” - 可能是一个常规 transformer 或另一个 hourglass

        x = self.valley_transformer(downsampled, mask = downsampled_mask)

        valley_out = x.clone()

        # 简单的重复上采样

        x = self.upsample(x)

        # 加上残差

        x = x + x_residual

        # 后谷“注意力重采样”

        if exists(self.attn_resampling_post_valley):
            x = self.attn_resampling_post_valley(
                rearrange(x, 'b (n s) d -> (b n) s d', s = s),
                rearrange(valley_out, 'b n d -> (b n) () d')
            )

            x = rearrange(x, '(b n) s d -> b (n s) d', b = b)

        # 将序列恢复到原始长度，如果为了池化而填充

        x = x[:, :n]

        # 后置 transformer 层

        x = self.post_transformer(x, mask = mask)
        return self.norm_out(x)
# 主要类定义

class HourglassTransformerLM(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,  # 标记的数量
        dim,  # 向量维度
        max_seq_len,  # 最大序列长度
        depth,  # 深度
        shorten_factor = None,  # 缩短因子，默认为None
        heads = 8,  # 头数，默认为8
        dim_head = 64,  # 头的维度，默认为64
        attn_resampling = True,  # 注意力重采样，默认为True
        updown_sample_type = 'naive',  # 上下采样类型，默认为'naive'
        causal = True  # 因果关系，默认为True
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        # 标记嵌入层
        self.token_emb = nn.Embedding(num_tokens, dim)
        # 位置嵌入层
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        # 获取 HourglassTransformer 模型
        self.transformer = get_hourglass_transformer(
            dim = dim,
            depth = depth,
            shorten_factor = shorten_factor,
            attn_resampling = attn_resampling,
            updown_sample_type = updown_sample_type,
            dim_head = dim_head,
            heads = heads,
            causal = causal,
            norm_out = True
        )

        # 线性层，用于输出logits
        self.to_logits = nn.Linear(dim, num_tokens)

    def forward(self, x, mask = None):
        device = x.device
        x = self.token_emb(x)
        pos_emb = self.pos_emb(torch.arange(x.shape[-2], device = device))
        x = x + rearrange(pos_emb, 'n d -> () n d')

        # 使用 Transformer 处理输入数据
        x = self.transformer(x, mask = mask)
        return self.to_logits(x)
```