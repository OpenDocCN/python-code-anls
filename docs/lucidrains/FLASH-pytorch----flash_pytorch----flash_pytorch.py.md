# `.\lucidrains\FLASH-pytorch\flash_pytorch\flash_pytorch.py`

```py
# 导入数学库和 PyTorch 库
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum

# 导入 einops 库中的 rearrange 函数和 rotary_embedding_torch 库中的 RotaryEmbedding 类
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding

# 辅助函数

# 判断值是否存在
def exists(val):
    return val is not None

# 如果值存在则返回该值，否则返回默认值
def default(val, d):
    return val if exists(val) else d

# 将数字 n 填充到最接近的 mult 的倍数
def padding_to_multiple_of(n, mult):
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder

# scalenorm

# 缩放归一化层
class ScaleNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        return x / norm.clamp(min = self.eps) * self.g

# absolute positional encodings

# 缩放的正弦嵌入层
class ScaledSinuEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1,))
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        n, device = x.shape[1], x.device
        t = torch.arange(n, device = device).type_as(self.inv_freq)
        sinu = einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((sinu.sin(), sinu.cos()), dim = -1)
        return emb * self.scale

# T5 relative positional bias

# T5 相对位置偏置层
class T5RelativePositionBias(nn.Module):
    def __init__(
        self,
        scale,
        causal = False,
        num_buckets = 32,
        max_distance = 128
    ):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, 1)

    @staticmethod
    def _relative_position_bucket(
        relative_position,
        causal = True,
        num_buckets = 32,
        max_distance = 128
    ):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, x):
        i, j, device = *x.shape[-2:], x.device
        q_pos = torch.arange(i, dtype = torch.long, device = device)
        k_pos = torch.arange(j, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, causal = self.causal, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j 1 -> i j')
        return bias * self.scale

# class

# 偏移缩放层
class OffsetScale(nn.Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std = 0.02)

    def forward(self, x):
        out = einsum('... d, h d -> ... h d', x, self.gamma) + self.beta
        return out.unbind(dim = -2)

# activation functions

# ReLU 平方激活函数
class ReLUSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2

# 拉普拉斯注意力函数
class LaplacianAttnFn(nn.Module):
    """ https://arxiv.org/abs/2209.10655 claims this is more stable than Relu squared """
    # 定义一个前向传播函数，接受输入 x
    def forward(self, x):
        # 计算均值 mu
        mu = math.sqrt(0.5)
        # 计算标准差 std
        std = math.sqrt((4 * math.pi) ** -1)
        # 使用误差函数计算激活函数的输出值，并返回
        return (1 + torch.special.erf((x - mu) / (std * math.sqrt(2)))) * 0.5
# 定义了一个名为GAU的类，表示门控注意力单元
class GAU(nn.Module):
    def __init__(
        self,
        *,
        dim,  # 输入维度
        query_key_dim = 128,  # 查询和键的维度，默认为128
        expansion_factor = 2.,  # 扩展因子，默认为2
        add_residual = True,  # 是否添加残差连接，默认为True
        causal = False,  # 是否使用因果注意力，默认为False
        dropout = 0.,  # dropout概率，默认为0
        laplace_attn_fn = False,  # 是否使用拉普拉斯注意力函数，默认为False
        rel_pos_bias = False,  # 是否使用相对位置偏置，默认为False
        norm_klass = nn.LayerNorm  # 规范化层，默认为nn.LayerNorm
    ):
        super().__init__()
        hidden_dim = int(expansion_factor * dim)

        self.norm = norm_klass(dim)  # 初始化规范化层
        self.causal = causal  # 初始化因果注意力标志
        self.dropout = nn.Dropout(dropout)  # 初始化dropout层

        self.attn_fn = ReLUSquared() if not laplace_attn_fn else LaplacianAttnFn()  # 初始化注意力函数

        self.rel_pos_bias = T5RelativePositionBias(scale = dim ** 0.5, causal = causal)  # 初始化相对位置偏置

        self.to_hidden = nn.Sequential(  # 隐藏层映射
            nn.Linear(dim, hidden_dim * 2),  # 线性变换
            nn.SiLU()  # 激活函数
        )

        self.to_qk = nn.Sequential(  # 查询和键映射
            nn.Linear(dim, query_key_dim),  # 线性变换
            nn.SiLU()  # 激活函数
        )

        self.offsetscale = OffsetScale(query_key_dim, heads = 2)  # 初始化偏移和缩放层

        self.to_out = nn.Sequential(  # 输出映射
            nn.Linear(hidden_dim, dim),  # 线性变换
            nn.Dropout(dropout)  # dropout层
        )

        self.add_residual = add_residual  # 是否添加残差连接

    def forward(
        self,
        x,  # 输入张量
        rel_pos_bias = None,  # 相对位置偏置，默认为None
        mask = None  # 掩码，默认为None
    ):
        seq_len, device = x.shape[-2], x.device  # 获取序列长度和设备信息

        normed_x = self.norm(x)  # 规范化输入张量
        v, gate = self.to_hidden(normed_x).chunk(2, dim = -1)  # 隐藏层映射并分割为两部分

        qk = self.to_qk(normed_x)  # 查询和键映射
        q, k = self.offsetscale(qk)  # 偏移和缩放

        sim = einsum('b i d, b j d -> b i j', q, k)  # 计算相似度

        if exists(self.rel_pos_bias):  # 如果存在相对位置偏置
            sim = sim + self.rel_pos_bias(sim)  # 加上相对位置偏置

        if exists(rel_pos_bias):  # 如果存在传入的相对位置偏置
            sim = sim + rel_pos_bias  # 加上传入的相对位置偏置

        attn = self.attn_fn(sim / seq_len)  # 计算注意力权重
        attn = self.dropout(attn)  # dropout

        if exists(mask):  # 如果存在掩码
            mask = rearrange(mask, 'b j -> b 1 j')  # 重排掩码形状
            attn = attn.masked_fill(~mask, 0.)  # 根据掩码填充注意力权重

        if self.causal:  # 如果是因果注意力
            causal_mask = torch.ones((seq_len, seq_len), dtype = torch.bool, device = device).triu(1)  # 创建因果掩码
            attn = attn.masked_fill(causal_mask, 0.)  # 根据因果掩码填充注意力权重

        out = einsum('b i j, b j d -> b i d', attn, v)  # 计算输出
        out = out * gate  # 门控

        out = self.to_out(out)  # 输出映射

        if self.add_residual:  # 如果添加残差连接
            out = out + x  # 添加残差连接

        return out  # 返回输出

# 定义了一个名为FLASH的类，表示快���自注意力流水线
class FLASH(nn.Module):
    def __init__(
        self,
        *,
        dim,  # 输入维度
        group_size = 256,  # 组大小，默认为256
        query_key_dim = 128,  # 查询和键的维度，默认为128
        expansion_factor = 2.,  # 扩展因子，默认为2
        causal = False,  # 是否使用因果注意力，默认为False
        dropout = 0.,  # dropout概率，默认为0
        rotary_pos_emb = None,  # 旋转位置嵌入，默认为None
        norm_klass = nn.LayerNorm,  # 规范化层，默认为nn.LayerNorm
        shift_tokens = False,  # 是否移动令牌，默认为False
        laplace_attn_fn = False,  # 是否使用拉普拉斯注意力函数，默认为False
        reduce_group_non_causal_attn = True  # 是否在非因果线性注意力中减少组，默认为True
    ):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.group_size = group_size  # 组大小
        self.causal = causal  # 因果注意力标志
        self.shift_tokens = shift_tokens  # 移动令牌标志

        self.attn_fn = ReLUSquared() if not laplace_attn_fn else LaplacianAttnFn()  # 初始化注意力函数

        # 位置嵌入
        self.rotary_pos_emb = rotary_pos_emb
        self.rel_pos_bias = T5RelativePositionBias(query_key_dim ** 0.5, causal = causal)  # 初始化相对位置偏置

        # 规范化层
        self.norm = norm_klass(dim)
        self.dropout = nn.Dropout(dropout)

        # 是否在非因果线性注意力中减少组
        self.reduce_group_non_causal_attn = reduce_group_non_causal_attn

        # 投影
        self.to_hidden = nn.Sequential(  # 隐藏层映射
            nn.Linear(dim, hidden_dim * 2),  # 线性变换
            nn.SiLU()  # 激活函数
        )

        self.to_qk = nn.Sequential(  # 查询和键映射
            nn.Linear(dim, query_key_dim),  # 线性变换
            nn.SiLU()  # 激活函数
        )

        self.qk_offset_scale = OffsetScale(query_key_dim, heads = 4)  # 偏移和缩放
        self.to_out = nn.Linear(hidden_dim, dim)  # 输出映射

    def forward(
        self,
        x,  # 输入张量
        *,
        mask = None  # 掩码，默认为None
    ):
        """
        b - batch
        n - sequence length (within groups)
        g - group dimension
        d - feature dimension (keys)
        e - feature dimension (values)
        i - sequence dimension (source)
        j - sequence dimension (target)
        """

        # 获取输入张量的形状信息
        b, n, device, g = x.shape[0], x.shape[-2], x.device, self.group_size

        # 对输入进行预处理
        normed_x = self.norm(x)

        # 执行令牌移位操作
        if self.shift_tokens:
            x_shift, x_pass = normed_x.chunk(2, dim = -1)
            x_shift = F.pad(x_shift, (0, 0, 1, -1), value = 0.)
            normed_x = torch.cat((x_shift, x_pass), dim = -1)

        # 初始投影
        v, gate = self.to_hidden(normed_x).chunk(2, dim = -1)
        qk = self.to_qk(normed_x)

        # 偏移和缩放
        quad_q, lin_q, quad_k, lin_k = self.qk_offset_scale(qk)

        # 屏蔽线性注意力键
        if exists(mask):
            lin_mask = rearrange(mask, '... -> ... 1')
            lin_k = lin_k.masked_fill(~lin_mask, 0.)

        # 旋转查询和键
        if exists(self.rotary_pos_emb):
            quad_q, lin_q, quad_k, lin_k = map(self.rotary_pos_emb.rotate_queries_or_keys, (quad_q, lin_q, quad_k, lin_k))

        # 对组进行填充
        padding = padding_to_multiple_of(n, g)

        if padding > 0:
            quad_q, quad_k, lin_q, lin_k, v = map(lambda t: F.pad(t, (0, 0, 0, padding), value = 0.), (quad_q, quad_k, lin_q, lin_k, v))

            mask = default(mask, torch.ones((b, n), device = device, dtype = torch.bool))
            mask = F.pad(mask, (0, padding), value = False)

        # 沿着序列对组进行分组
        quad_q, quad_k, lin_q, lin_k, v = map(lambda t: rearrange(t, 'b (n g) d -> b n g d', g = self.group_size), (quad_q, quad_k, lin_q, lin_k, v))

        if exists(mask):
            mask = rearrange(mask, 'b (g j) -> b g 1 j', j = g)

        # 计算二次注意力输出
        sim = einsum('... i d, ... j d -> ... i j', quad_q, quad_k) / g
        sim = sim + self.rel_pos_bias(sim)
        attn = self.attn_fn(sim)
        attn = self.dropout(attn)

        if exists(mask):
            attn = attn.masked_fill(~mask, 0.)

        if self.causal:
            causal_mask = torch.ones((g, g), dtype = torch.bool, device = device).triu(1)
            attn = attn.masked_fill(causal_mask, 0.)

        quad_out = einsum('... i j, ... j d -> ... i d', attn, v)

        # 计算线性注意力输出
        if self.causal:
            lin_kv = einsum('b g n d, b g n e -> b g d e', lin_k, v) / g

            # 沿着组维度进行排他性累加
            lin_kv = lin_kv.cumsum(dim = 1)
            lin_kv = F.pad(lin_kv, (0, 0, 0, 0, 1, -1), value = 0.)

            lin_out = einsum('b g d e, b g n d -> b g n e', lin_kv, lin_q)
        else:
            context_einsum_eq = 'b d e' if self.reduce_group_non_causal_attn else 'b g d e'
            lin_kv = einsum(f'b g n d, b g n e -> {context_einsum_eq}', lin_k, v) / n
            lin_out = einsum(f'b g n d, {context_einsum_eq} -> b g n e', lin_q, lin_kv)

        # 将组折叠回完整序列，并去除填充
        quad_attn_out, lin_attn_out = map(lambda t: rearrange(t, 'b g n d -> b (g n) d')[:, :n], (quad_out, lin_out))

        # 门控
        out = gate * (quad_attn_out + lin_attn_out)

        # 投影输出并添加残差连接
        return self.to_out(out) + x
# FLASH Transformer 类定义
class FLASHTransformer(nn.Module):
    # 初始化函数
    def __init__(
        self,
        *,
        dim,  # 特征维度
        num_tokens,  # token 的数量
        depth,  # 层数
        group_size = 256,  # 分组大小，默认为 256
        query_key_dim = 128,  # 查询键的维度，默认为 128
        expansion_factor = 2.,  # 扩展因子，默认为 2.0
        causal = False,  # 是否是因果的，默认为 False
        attn_dropout = 0.,  # 注意力机制的 dropout，默认为 0
        norm_type = 'scalenorm',  # 归一化类型，默认为 scalenorm
        shift_tokens = True,  # 是否移动 token，默认为 True
        laplace_attn_fn = False,  # 是否使用拉普拉斯注意力函数，默认为 False
        reduce_group_non_causal_attn = True  # 是否减少非因果注意力，默认为 True
    ):
        super().__init__()
        # 断言，确保 norm_type 是 scalenorm 或 layernorm
        assert norm_type in ('scalenorm', 'layernorm'), 'norm_type must be one of scalenorm or layernorm'

        # 根据 norm_type 选择不同的归一化类
        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm

        # 创建 token 的嵌入层
        self.token_emb = nn.Embedding(num_tokens, dim)
        # 创建绝对位置嵌入层
        self.abs_pos_emb = ScaledSinuEmbedding(dim)
        # 设置分组大小
        self.group_size = group_size

        # 创建旋转位置嵌入层
        rotary_pos_emb = RotaryEmbedding(dim = min(32, query_key_dim))
        # 最大旋转嵌入维度为 32，部分旋转嵌入，来自 Wang 等人 - GPT-J

        # 创建多层 FLASH 模块
        self.layers = nn.ModuleList([FLASH(dim = dim, group_size = group_size, query_key_dim = query_key_dim, expansion_factor = expansion_factor, causal = causal, dropout = attn_dropout, rotary_pos_emb = rotary_pos_emb, norm_klass = norm_klass, shift_tokens = shift_tokens, reduce_group_non_causal_attn = reduce_group_non_causal_attn, laplace_attn_fn = laplace_attn_fn) for _ in range(depth)])

        # 创建输出层
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),  # 归一化层
            nn.Linear(dim, num_tokens)  # 线性层，将特征维度映射到 token 数量
        )

    # 前向传播函数
    def forward(
        self,
        x,  # 输入张量
        *,
        mask = None  # 掩码，默认为 None
    ):
        x = self.token_emb(x)  # 对输入张量进行 token 嵌入
        x = self.abs_pos_emb(x) + x  # 添加绝对位置嵌入

        # 遍历每个 FLASH 模块
        for flash in self.layers:
            x = flash(x, mask = mask)  # 调用 FLASH 模块的前向传播

        return self.to_logits(x)  # 返回输出结果
```