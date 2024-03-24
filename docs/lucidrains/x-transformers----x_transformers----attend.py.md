# `.\lucidrains\x-transformers\x_transformers\attend.py`

```
# 导入所需模块和库
from functools import partial
from typing import Optional, Tuple
import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from collections import namedtuple
from functools import wraps
from packaging import version
from dataclasses import dataclass
from einops import rearrange, repeat

# 定义数据类 Intermediates，包含了一些可选的 Tensor 类型字段
@dataclass
class Intermediates:
    qk_similarities: Optional[Tensor] = None
    pre_softmax_attn: Optional[Tensor] = None
    post_softmax_attn: Optional[Tensor] = None
    cached_kv: Optional[Tuple[Tensor, Tensor]] = None

    # 将字段转换为元组
    def to_tuple(self):
        return (self.qk_similarities, self.pre_softmax_attn, self.post_softmax_attn)

# 定义一些辅助函数

# 判断值是否存在
def exists(val):
    return val is not None

# 返回默认值
def default(val, d):
    return val if exists(val) else d

# 过滤掉不存在的值
def compact(arr):
    return [*filter(exists, arr]

# 保证函数只调用一次
def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

# 打印函数，只打印一次
print_once = once(print)

# 创建因果掩码的函数
def create_causal_mask(i, j, device):
    return torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)

# 为 ONNX CPU 创建因果掩码的函数
def onnx_create_causal_mask(i, j, device):
    r = torch.arange(i, device=device)
    causal_mask = rearrange(r, 'i -> i 1') < rearrange(r, 'j -> 1 j')
    causal_mask = F.pad(causal_mask, (j - i, 0), value=False)
    return causal_mask

# 主类 Attend
class Attend(nn.Module):
    def __init__(
        self,
        *,
        dropout=0.,
        causal=False,
        heads=None,
        talking_heads=False,
        sparse_topk=None,
        scale=None,
        qk_norm=False,
        flash=False,
        add_zero_kv=False,
        onnxable=False,
        sdp_kwargs: dict = dict(
            enable_flash=True,
            enable_math=True,
            enable_mem_efficient=True
        )
    ):
        super().__init__()
        self.scale = scale

        self.causal = causal
        self.create_causal_mask = onnx_create_causal_mask if onnxable else create_causal_mask

        self.attn_fn = partial(F.softmax, dtype=torch.float32) if not qk_norm else F.softmax

        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        # talking heads
        assert not (flash and talking_heads), 'talking heads not compatible with flash attention'
        self.talking_heads = talking_heads
        if talking_heads:
            self.pre_softmax_talking_heads = nn.Conv2d(heads, heads, 1, bias=False)
            self.post_softmax_talking_heads = nn.Conv2d(heads, heads, 1, bias=False)

        # sparse topk
        assert not (flash and sparse_topk), 'sparse topk not compatible with flash attention'
        self.sparse_topk = sparse_topk

        # 添加一个由零组成的键/值令牌，以帮助控制异常值
        self.add_zero_kv = add_zero_kv

        # flash attention
        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        self.sdp_kwargs = sdp_kwargs

    def flash_attn(
        self,
        q, k, v,
        mask=None,
        attn_bias=None
        ):
            # 解包输入张量的形状和其他属性
            batch, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

            # 推荐使用 Tri Dao 的多查询单键值注意力
            # 将键值对的形状从 torch.Size([1, 512, 64]) 扩展为 torch.Size([1, 8, 512, 64])

            if k.ndim == 3:
                k = repeat(k, 'b ... -> b h ...', h = q.shape[1])

            if v.ndim == 3:
                v = repeat(v, 'b ... -> b h ...', h = q.shape[1])

            # 处理缩放 - 默认情况下按 dim_head ** -0.5 缩放，但如果使用余弦相似度注意力，则需要注意

            if exists(self.scale):
                default_scale = q.shape[-1] ** -0.5
                q = q * (self.scale / default_scale)

            # 检查是否存在掩码并扩展为兼容的形状
            # 掩码是 B L，因此必须扩展为 B H N L

            causal = self.causal

            # 在 kv 缓存中只有一个令牌的情况下（q_len == 1），只需关闭因果掩码
            # 在推测解码中，这可能会增加到 5-6，因此那里将需要右对齐的因果掩码

            if q_len == 1 and causal:
                causal = False

            # 扩展键填充掩码

            if exists(mask):
                assert mask.ndim == 4
                mask = mask.expand(batch, heads, q_len, k_len)

            # 处理 kv 缓存 - 这应该可以在更新的 flash attention 2 中绕过

            if k_len > q_len and causal:
                causal_mask = self.create_causal_mask(q_len, k_len, device = device)
                if not exists(mask):
                    mask = ~causal_mask
                else:
                    mask = mask & ~causal_mask
                causal = False

            # 手动处理因果掩码，如果给定了另一个掩码

            row_is_entirely_masked = None

            if exists(mask) and causal:
                causal_mask = self.create_causal_mask(q_len, k_len, device = device)
                mask = mask & ~causal_mask

                # 防止整行被掩盖

                row_is_entirely_masked = ~mask.any(dim = -1)
                mask[..., 0] = mask[..., 0] | row_is_entirely_masked

                causal = False

            # 处理 alibi 位置偏差
            # 从布尔值转换为浮点数

            if exists(attn_bias):
                attn_bias = rearrange(attn_bias, 'h i j -> 1 h i j').expand(batch, heads, -1, -1)

                # 如果给定了掩码，掩码已经包含了上述逻辑中的因果掩��
                # 否则，如果没有给定掩码但仍然是因果的，将 alibi 位置偏差掩盖为一个很大的负数

                mask_value = -torch.finfo(q.dtype).max

                if exists(mask):
                    attn_bias = attn_bias.masked_fill(~mask, mask_value // 2)
                elif causal:
                    causal_mask = self.create_causal_mask(q_len, k_len, device = device)
                    attn_bias = attn_bias.masked_fill(causal_mask, mask_value // 2)
                    causal = False

                # scaled_dot_product_attention 将 attn_mask 作为布尔值或加性偏差处理
                # 这里将其作为加性偏差

                mask = attn_bias

            # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale

            with torch.backends.cuda.sdp_kernel(**self.sdp_kwargs):
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask = mask,
                    dropout_p = self.dropout if self.training else 0., 
                    is_causal = causal
                )

            # 对于整行被完全掩盖的情况，应将该行令牌的输出置零

            if exists(row_is_entirely_masked):
                out = out.masked_fill(row_is_entirely_masked[..., None], 0.)

            return out, Intermediates()

        def forward(
            self,
            q, k, v,
            mask = None,
            attn_bias = None,
            prev_attn = None
    ):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        # 获取输入张量的形状信息
        n, heads, kv_heads, device = q.shape[-2], q.shape[1], k.shape[1], q.device

        # 设置缩放因子，默认为特征维度的倒数平方根
        scale = default(self.scale, q.shape[-1] ** -0.5)

        # 是否启用因果注意力
        causal = self.causal

        # 处理缓存的键值对解码
        if n == 1 and causal:
            causal = False

        # 处理分组的多查询注意力
        if kv_heads == 1:
            k, v = map(lambda t: rearrange(t, 'b 1 n d -> b n d'), (k, v))
        elif kv_heads < heads:
            k, v = map(lambda t: repeat(t, 'b kvh n d -> b (r kvh) n d', r = heads // kv_heads), (k, v))

        # 处理零键值对，允许网络关注空内容
        if self.add_zero_kv:
            k, v = map(lambda t: F.pad(t, (0, 0, 1, 0), value = 0.), (k, v))

            if exists(mask):
                mask = F.pad(mask, (1, 0), value = True)

            if exists(attn_bias):
                attn_bias = F.pad(attn_bias, (1, 0), value = 0.)

        # 如果启用了flash attention，则返回flash attention结果
        if self.flash:
            assert not exists(prev_attn), 'residual attention not compatible with flash attention'
            return self.flash_attn(q, k, v, mask = mask, attn_bias = attn_bias)

        # 根据键值对的维度选择相应的乘法运算
        kv_einsum_eq = 'b j d' if k.ndim == 3 else 'b h j d'

        # 计算点积
        dots = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', q, k) * scale

        # 如果存在先前的注意力，加上先前的注意力值
        if exists(prev_attn):
            dots = dots + prev_attn

        # 复制点积结果
        qk_similarities = dots.clone()

        # 如果启用了talking heads，对点积结果进行预处理
        if self.talking_heads:
            dots = self.pre_softmax_talking_heads(dots)

        # 如果存在注意力偏置，加上注意力偏置
        if exists(attn_bias):
            dots = dots + attn_bias

        # 获取点积结果的形状信息
        i, j, dtype = *dots.shape[-2:], dots.dtype

        # 设置掩码值为负无穷
        mask_value = -torch.finfo(dots.dtype).max

        # 如果存在稀疏topk参数且小于j，则只保留topk个值
        if exists(self.sparse_topk) and self.sparse_topk < j:
            top_values, _ = dots.topk(self.sparse_topk, dim = -1)
            sparse_topk_mask = dots < top_values[..., -1:]
            mask = (mask & sparse_topk_mask) if exists(mask) else sparse_topk_mask

        # 如果存在掩码，根据掩码值进行填充
        if exists(mask):
            dots = dots.masked_fill(~mask, mask_value)

        # 如果启用了因果注意力，根据因果掩码进行填充
        if causal:
            causal_mask = self.create_causal_mask(i, j, device = device)
            dots = dots.masked_fill(causal_mask, mask_value)

        # 复制点积结果作为预softmax的注意力值
        pre_softmax_attn = dots.clone()

        # 计算softmax得到注意力权重
        attn = self.attn_fn(dots, dim = -1)
        attn = attn.type(dtype)

        # 复制softmax后的注意力权重
        post_softmax_attn = attn.clone()

        # 对注意力权重进行dropout
        attn = self.attn_dropout(attn)

        # 如果启用了talking heads，对注意力权重进行后处理
        if self.talking_heads:
            attn = self.post_softmax_talking_heads(attn)

        # 计算输出结果
        out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)

        # 保存中间结果
        intermediates = Intermediates(
            qk_similarities = qk_similarities,
            pre_softmax_attn = pre_softmax_attn,
            post_softmax_attn = post_softmax_attn
        )

        # 返回输出结果和中间结果
        return out, intermediates
```