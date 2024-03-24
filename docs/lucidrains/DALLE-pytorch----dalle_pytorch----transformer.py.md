# `.\lucidrains\DALLE-pytorch\dalle_pytorch\transformer.py`

```py
# 导入必要的库
from collections import deque
from collections.abc import Iterable
from functools import partial
from itertools import islice, cycle

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange

# 导入自定义模块
from dalle_pytorch.reversible import ReversibleSequence, SequentialSequence
from dalle_pytorch.attention import Attention, SparseAttention, SparseConvCausalAttention, SparseAxialCausalAttention

# 导入旋转嵌入模块
from rotary_embedding_torch import RotaryEmbedding, broadcat

# 辅助函数

# 判断变量是否存在
def exists(val):
    return val is not None

# 返回默认值
def default(val, d):
    return val if exists(val) else d

# 将变量转换为元组
def cast_tuple(val, depth = 1):
    return val if isinstance(val, Iterable) else (val,) * depth

# 类

# 最大值分割类
class DivideMax(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        maxes = x.amax(dim = self.dim, keepdim = True).detach()
        return x / maxes

# 非缓存类
class NonCached(nn.Module):
    """
    A wrapper for layers that don't support the inference cache themselves.
    Reconstructs the full sequence before the layer and
    cuts the suffix of the outputs after the layer.
    """

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *, cache = None, cache_key = None, **kwargs):
        n = x.shape[-2]
        if exists(cache):
            if cache_key in cache:
                x = torch.cat([cache[cache_key], x], dim=-2)
            cache[cache_key] = x

        out = self.fn(x, **kwargs)

        return out[:, -n:]

# 缓存类
class CachedAs(nn.Module):
    """
    A wrapper that defines a key for the inference cache.
    """

    def __init__(self, cache_key, fn):
        super().__init__()
        self.cache_key = cache_key
        self.fn = fn

    def forward(self, x, *, cache=None, **kwargs):
        return self.fn(x, cache=cache, cache_key=self.cache_key, **kwargs)

# 层缩放类
class LayerScale(nn.Module):
    def __init__(self, dim, depth, fn):
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

# 层归一化类

class PreNorm(nn.Module):
    def __init__(self, dim, fn, sandwich = False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim) if sandwich else nn.Identity()
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        x = self.fn(x, **kwargs)
        return self.norm_out(x)

# 前馈类

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, dropout = 0., mult = 4.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, cache=None, cache_key=None):
        return self.net(x)

# 标记移位类

class PreShiftToken(nn.Module):
    def __init__(self, fn, image_size, seq_len):
        super().__init__()
        self.fn = fn
        self.image_size = image_size
        self.seq_len = seq_len
        self.img_seq_len = image_size ** 2
        self.text_len = seq_len - self.img_seq_len + 1
    # 定义前向传播函数，接受输入 x，缓存 cache，缓存键 cache_key，以及其他关键字参数 kwargs
    def forward(self, x, cache=None, cache_key=None, **kwargs):
        # 获取序列长度、图像大小、文本长度
        seq_len, image_size, text_len = self.seq_len, self.image_size, self.text_len

        # 如果缓存存在且缓存键存在于缓存中
        if exists(cache) and cache_key in cache:
            # 从缓存中获取偏移量
            offset = cache['offset']
            # 断言偏移量大于等于文本长度，不支持文本的缓存推断
            assert offset >= text_len, "cached inference for text is not supported"
            # 从缓存中获取队列 q
            q = cache[cache_key]
            # 断言 q 是双端队列且长度为图像大小
            assert isinstance(q, deque) and len(q) == image_size

            # 将输入 x 按照最后一个维度分割成四部分
            x_top, x_left, *x_pass = x[:, -1].chunk(4, dim=-1)

            # 将 x_top 和 x_left 添加到队列 q 中
            q.append((x_top, x_left))
            # 弹出队列 q 中的第一个元素，并更新 x_top 和 x_left
            x_top = q.popleft()[0]
            x_left = q[-2][1]
            # 如果偏移量减去文本长度对图像大小取模等于 0，则将 x_left 置零
            if (offset - text_len) % image_size == 0:
                x_left = torch.zeros_like(x_left)

            # 将 x_top、x_left 和其他部分拼接在一起
            x = torch.cat((x_top, x_left, *x_pass), dim=-1)
            # 调用 self.fn 函数，传入 x[:, None] 作为输入，同时传入缓存和其他关键字参数
            return self.fn(x[:, None], cache=cache, **kwargs)

        # 获取输入 x 的形状中的第二个维度大小
        n = x.shape[1]
        # 计算需要填充的数量
        padding = seq_len - n + 1

        # 如果序列长度小于文本长度，则没有图像令牌需要移动
        if n < text_len:
            return self.fn(x, **kwargs)

        # 获取文本和图像令牌
        x_text, x_img = x[:, :text_len], x[:, text_len:]
        # 对图像令牌进行填充
        x_img = F.pad(x_img, (0, 0, 0, padding))
        # 重新排列图像令牌的形状
        x_img = rearrange(x_img, 'b (h w) d -> b h w d', h=image_size)

        # 对文本令牌进行左移 1 位
        x_text_shift, x_text_pass = x_text.chunk(2, dim=-1)
        x_text_shift = F.pad(x_text_shift, (0, 0, 1, -1))
        x_text = torch.cat((x_text_shift, x_text_pass), dim=-1)

        # 对图像令���进行从上和从左的移动
        x_img_shift_top, x_img_shift_left, *x_img_pass = x_img.chunk(4, dim=-1)
        x_img_shift_left = F.pad(x_img_shift_left, (0, 0, 1, -1))
        x_img_shift_top = F.pad(x_img_shift_top, (0, 0, 0, 0, 1, -1))
        x_img = torch.cat((x_img_shift_top, x_img_shift_left, *x_img_pass), dim=-1)

        # 将文本和图像序列合并在一起
        x_img = rearrange(x_img, 'b h w d -> b (h w) d')
        x_img = x_img[:, :-padding]
        x = torch.cat((x_text, x_img), dim=1)

        # 如果缓存存在
        if exists(cache):
            # 创建虚拟的顶部和左侧令牌
            dummy_top, dummy_left, *_ = x[:, -1].chunk(4, dim=-1)
            dummy_top, dummy_left = torch.zeros_like(dummy_top), torch.zeros_like(dummy_left)

            # 创建双端队列 q
            q = deque()
            x_img = x_img[:, -image_size:]
            # 将虚拟令牌添加到队列 q 中，直到队列大小为图像大小
            for _ in range(image_size - x_img.shape[1]):
                q.append((dummy_top, dummy_left))
            # 将图像令牌添加到队列 q 中
            for i in range(x_img.shape[1]):
                q.append(x_img[:, i].chunk(4, dim=-1)[:2])
            # 将队列 q 存入缓存中
            cache[cache_key] = q

        # 调用 self.fn 函数，传入 x 作为输入，同时传入缓存和其他关键字参数
        return self.fn(x, cache=cache, **kwargs)
# 主要的Transformer类
class Transformer(nn.Module):
    # 初始化函数
    def __init__(
        self,
        *,
        dim,
        depth,
        seq_len,
        reversible = False,
        causal = True,
        heads = 8,
        dim_head = 64,
        ff_mult = 4,
        attn_dropout = 0.,
        ff_dropout = 0.,
        attn_types = None,
        image_fmap_size = None,
        sparse_attn = False,
        stable = False,
        sandwich_norm = False,
        shift_tokens = False,
        rotary_emb = True,
        shared_attn_ids = None,
        shared_ff_ids = None,
        optimize_for_inference = False,  # 使用缓存友好的掩码注意力代替稀疏注意力
    # 前向传播函数
    def forward(self, x, **kwargs):
        return self.layers(x, rotary_pos_emb = self.pos_emb, **kwargs)

    # 获取注意力掩码函数
    def _get_attention_mask(self, attn_type):
        # 计算图像序列长度
        img_seq_len = self.image_fmap_size ** 2
        # 计算文本长度
        text_len = self.seq_len + 1 - img_seq_len

        # 创建静态掩码
        static_mask = torch.zeros(self.seq_len, self.seq_len, dtype=torch.bool)
        static_mask[:, :text_len] = True
        # 根据不同的注意力类型生成不同的静态掩码
        if attn_type == 'axial_row':
            for row in range(self.image_fmap_size):
                begin = text_len + row * self.image_fmap_size
                end = text_len + (row + 1) * self.image_fmap_size
                static_mask[begin:end, begin:end] = True
        elif attn_type == 'axial_col':
            for col in range(self.image_fmap_size):
                begin = text_len + col
                static_mask[begin::self.image_fmap_size, begin::self.image_fmap_size] = True
        else:
            raise ValueError(f'attention type "{attn_type}" can\'t be simulated with a static mask')
        return static_mask
```