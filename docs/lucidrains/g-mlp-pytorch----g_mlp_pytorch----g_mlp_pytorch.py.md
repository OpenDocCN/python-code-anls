# `.\lucidrains\g-mlp-pytorch\g_mlp_pytorch\g_mlp_pytorch.py`

```
# 从 random 模块中导入 randrange 函数
# 从 torch 模块中导入相关函数和类
# 从 einops 模块中导入 rearrange, repeat 函数以及 Rearrange, Reduce 类
from random import randrange
import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

# functions

# 判断值是否存在的函数
def exists(val):
    return val is not None

# 将输入值转换为元组的函数
def pair(val):
    return (val, val) if not isinstance(val, tuple) else val

# 对层进行 dropout 处理的函数
def dropout_layers(layers, prob_survival):
    if prob_survival == 1:
        return layers

    num_layers = len(layers)
    to_drop = torch.zeros(num_layers).uniform_(0., 1.) > prob_survival

    # 确保至少有一层保留
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers

# 对张量进行平移的函数
def shift(t, amount, mask = None):
    if amount == 0:
        return t
    return F.pad(t, (0, 0, amount, -amount), value = 0.)

# helper classes

# 残差连接的类
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

# 对输入进行预平移的类
class PreShiftTokens(nn.Module):
    def __init__(self, shifts, fn):
        super().__init__()
        self.fn = fn
        self.shifts = tuple(shifts)

    def forward(self, x, **kwargs):
        if self.shifts == (0,):
            return self.fn(x, **kwargs)

        shifts = self.shifts
        segments = len(shifts)
        feats_per_shift = x.shape[-1] // segments
        splitted = x.split(feats_per_shift, dim = -1)
        segments_to_shift, rest = splitted[:segments], splitted[segments:]
        segments_to_shift = list(map(lambda args: shift(*args), zip(segments_to_shift, shifts)))
        x = torch.cat((*segments_to_shift, *rest), dim = -1)
        return self.fn(x, **kwargs)

# 对输入进行预归一化的类
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

# 注意力机制类
class Attention(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inner, causal = False):
        super().__init__()
        self.scale = dim_inner ** -0.5
        self.causal = causal

        self.to_qkv = nn.Linear(dim_in, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim_out)

    def forward(self, x):
        device = x.device
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if self.causal:
            mask = torch.ones(sim.shape[-2:], device = device).triu(1).bool()
            sim.masked_fill_(mask[None, ...], -torch.finfo(q.dtype).max)

        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        return self.to_out(out)

# 空间门控单元类
class SpatialGatingUnit(nn.Module):
    def __init__(
        self,
        dim,
        dim_seq,
        causal = False,
        act = nn.Identity(),
        heads = 1,
        init_eps = 1e-3,
        circulant_matrix = False
    ):
        super().__init__()
        dim_out = dim // 2
        self.heads = heads
        self.causal = causal
        self.norm = nn.LayerNorm(dim_out)

        self.act = act

        # 参数

        if circulant_matrix:
            self.circulant_pos_x = nn.Parameter(torch.ones(heads, dim_seq))
            self.circulant_pos_y = nn.Parameter(torch.ones(heads, dim_seq))

        self.circulant_matrix = circulant_matrix
        shape = (heads, dim_seq,) if circulant_matrix else (heads, dim_seq, dim_seq)
        weight = torch.zeros(shape)

        self.weight = nn.Parameter(weight)
        init_eps /= dim_seq
        nn.init.uniform_(self.weight, -init_eps, init_eps)

        self.bias = nn.Parameter(torch.ones(heads, dim_seq))
    # 定义前向传播函数，接受输入 x 和门控信息 gate_res
    def forward(self, x, gate_res = None):
        # 获取输入 x 的设备信息、特征维度 n 和注意力头数 h
        device, n, h = x.device, x.shape[1], self.heads

        # 将输入 x 切分为结果 res 和门控信息 gate
        res, gate = x.chunk(2, dim = -1)
        # 对门控信息 gate 进行归一化处理
        gate = self.norm(gate)

        # 获取权重和偏置参数
        weight, bias = self.weight, self.bias

        # 如果使用循环矩阵
        if self.circulant_matrix:
            # 构建循环矩阵

            # 获取权重参数的最后一个维度大小
            dim_seq = weight.shape[-1]
            # 在权重参数的最后一个维度上进行填充
            weight = F.pad(weight, (0, dim_seq), value = 0)
            weight = repeat(weight, '... n -> ... (r n)', r = dim_seq)
            weight = weight[:, :-dim_seq].reshape(h, dim_seq, 2 * dim_seq - 1)
            weight = weight[:, :, (dim_seq - 1):]

            # 赋予循环矩阵绝对位置感知

            pos_x, pos_y = self.circulant_pos_x, self.circulant_pos_y
            weight = weight * rearrange(pos_x, 'h i -> h i ()') * rearrange(pos_y, 'h j -> h () j')

        # 如果是因果关系
        if self.causal:
            # 裁剪权重和偏置参数
            weight, bias = weight[:, :n, :n], bias[:, :n]
            # 创建掩码，使得只能看到当前位置及之前的信息
            mask = torch.ones(weight.shape[-2:], device = device).triu_(1).bool()
            mask = rearrange(mask, 'i j -> () i j')
            weight = weight.masked_fill(mask, 0.)

        # 重排门控信息 gate 的维度
        gate = rearrange(gate, 'b n (h d) -> b h n d', h = h)

        # 执行矩阵乘法操作
        gate = einsum('b h n d, h m n -> b h m d', gate, weight)
        # 加上偏置参数
        gate = gate + rearrange(bias, 'h n -> () h n ()')

        # 重排门控信息 gate 的维度
        gate = rearrange(gate, 'b h n d -> b n (h d)')

        # 如果存在门控信息 gate_res，则将其加到 gate 上
        if exists(gate_res):
            gate = gate + gate_res

        # 对 gate 执行激活函数，并乘以结果 res
        return self.act(gate) * res
# 定义 gMLPBlock 类，继承自 nn.Module 类
class gMLPBlock(nn.Module):
    # 初始化函数
    def __init__(
        self,
        *,
        dim,  # 输入维度
        dim_ff,  # Feed-Forward 层维度
        seq_len,  # 序列长度
        heads = 1,  # 多头注意力机制中的头数
        attn_dim = None,  # 注意力机制的维度
        causal = False,  # 是否使用因果关系
        act = nn.Identity(),  # 激活函数，默认为恒等映射
        circulant_matrix = False  # 是否使用循环矩阵
    ):
        super().__init__()
        # 输入投影层，包含线性变换和 GELU 激活函数
        self.proj_in = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.GELU()
        )

        # 如果存在注意力机制的维度，则创建注意力对象
        self.attn = Attention(dim, dim_ff // 2, attn_dim, causal) if exists(attn_dim) else None

        # 空间门控单元
        self.sgu = SpatialGatingUnit(dim_ff, seq_len, causal, act, heads, circulant_matrix = circulant_matrix)
        # 输出投影层
        self.proj_out = nn.Linear(dim_ff // 2, dim)

    # 前向传播函数
    def forward(self, x):
        # 如果存在注意力对象，则进行注意力计算
        gate_res = self.attn(x) if exists(self.attn) else None
        x = self.proj_in(x)  # 输入投影
        x = self.sgu(x, gate_res = gate_res)  # 空间门控单元
        x = self.proj_out(x)  # 输出投影
        return x

# 主要类

# 定义 gMLP 类，继承自 nn.Module 类
class gMLP(nn.Module):
    # 初始化函数
    def __init__(
        self,
        *,
        num_tokens = None,  # 标记数量
        dim,  # 输入维度
        depth,  # 深度
        seq_len,  # 序列长度
        heads = 1,  # 多头注意力机制中的头数
        ff_mult = 4,  # Feed-Forward 层维度倍数
        attn_dim = None,  # 注意力机制的维度
        prob_survival = 1.,  # 生存概率
        causal = False,  # 是否使用因果关系
        circulant_matrix = False,  # 是否使用循环矩阵
        shift_tokens = 0,  # 标记偏移
        act = nn.Identity()  # 激活函数，默认为恒等映射
    ):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by number of heads'

        dim_ff = dim * ff_mult
        self.seq_len = seq_len
        self.prob_survival = prob_survival

        # Embedding 层
        self.to_embed = nn.Embedding(num_tokens, dim) if exists(num_tokens) else nn.Identity()

        token_shifts = tuple(range(0 if causal else -shift_tokens, shift_tokens + 1))
        # 层列表
        self.layers = nn.ModuleList([Residual(PreNorm(dim, PreShiftTokens(token_shifts, gMLPBlock(dim = dim, heads = heads, dim_ff = dim_ff, seq_len = seq_len, attn_dim = attn_dim, causal = causal, act = act, circulant_matrix = circulant_matrix))) for i in range(depth)])

        # 输出层
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        ) if exists(num_tokens) else nn.Identity()

    # 前向传播函数
    def forward(self, x):
        x = self.to_embed(x)  # Embedding
        layers = self.layers if not self.training else dropout_layers(self.layers, self.prob_survival)
        out = nn.Sequential(*layers)(x)  # 层序列
        return self.to_logits(out)  # 输出层

# 定义 gMLPVision 类，继承自 nn.Module 类
class gMLPVision(nn.Module):
    # 初始化函数
    def __init__(
        self,
        *,
        image_size,  # 图像尺寸
        patch_size,  # 补丁尺寸
        num_classes,  # 类别数量
        dim,  # 输入维度
        depth,  # 深度
        heads = 1,  # 多头注意力机制中的头数
        ff_mult = 4,  # Feed-Forward 层维度倍数
        channels = 3,  # 通道数
        attn_dim = None,  # 注意力机制的维度
        prob_survival = 1.  # 生存概率
    ):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by number of heads'

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert (image_height % patch_height) == 0 and (image_width % patch_width) == 0, 'image height and width must be divisible by patch size'
        num_patches = (image_height // patch_height) * (image_width // patch_width)

        dim_ff = dim * ff_mult

        # 补丁嵌入层
        self.to_patch_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1 = patch_height, p2 = patch_width),
            nn.Linear(channels * patch_height * patch_width, dim)
        )

        self.prob_survival = prob_survival

        # 层列表
        self.layers = nn.ModuleList([Residual(PreNorm(dim, gMLPBlock(dim = dim, heads = heads, dim_ff = dim_ff, seq_len = num_patches, attn_dim = attn_dim))) for i in range(depth)])

        # 输出层
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            Reduce('b n d -> b d', 'mean'),
            nn.Linear(dim, num_classes)
        )

    # 前向传播函数
    def forward(self, x):
        x = self.to_patch_embed(x)  # 补丁嵌入
        layers = self.layers if not self.training else dropout_layers(self.layers, self.prob_survival)
        x = nn.Sequential(*layers)(x)  # 层序列
        return self.to_logits(x)  # 输出层
```