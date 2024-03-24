# `.\lucidrains\halonet-pytorch\halonet_pytorch\halonet_pytorch.py`

```py
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat

# 导入所需的库

# 相对位置编码

def to(x):
    return {'device': x.device, 'dtype': x.dtype}

# 返回包含设备和数据类型信息的字典

def pair(x):
    return (x, x) if not isinstance(x, tuple) else x

# 如果输入不是元组，则返回包含两个相同元素的元组，否则返回原元组

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim = dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

# 在指定维度上扩展张量的大小

def rel_to_abs(x):
    b, l, m = x.shape
    r = (m + 1) // 2

    col_pad = torch.zeros((b, l, 1), **to(x))
    x = torch.cat((x, col_pad), dim = 2)
    flat_x = rearrange(x, 'b l c -> b (l c)')
    flat_pad = torch.zeros((b, m - l), **to(x))
    flat_x_padded = torch.cat((flat_x, flat_pad), dim = 1)
    final_x = flat_x_padded.reshape(b, l + 1, m)
    final_x = final_x[:, :l, -r:]
    return final_x

# 将相对位置编码转换为绝对位置编码

def relative_logits_1d(q, rel_k):
    b, h, w, _ = q.shape
    r = (rel_k.shape[0] + 1) // 2

    logits = einsum('b x y d, r d -> b x y r', q, rel_k)
    logits = rearrange(logits, 'b x y r -> (b x) y r')
    logits = rel_to_abs(logits)

    logits = logits.reshape(b, h, w, r)
    logits = expand_dim(logits, dim = 2, k = r)
    return logits

# 计算相对位置的一维逻辑值

class RelPosEmb(nn.Module):
    def __init__(
        self,
        block_size,
        rel_size,
        dim_head
    ):
        super().__init__()
        height = width = rel_size
        scale = dim_head ** -0.5

        self.block_size = block_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        block = self.block_size

        q = rearrange(q, 'b (x y) c -> b x y c', x = block)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b x i y j-> b (x y) (i j)')

        q = rearrange(q, 'b x y d -> b y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b x i y j -> b (y x) (j i)')
        return rel_logits_w + rel_logits_h

# 相对位置编码类

# classes

class HaloAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        block_size,
        halo_size,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        assert halo_size > 0, 'halo size must be greater than 0'

        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.block_size = block_size
        self.halo_size = halo_size

        inner_dim = dim_head * heads

        self.rel_pos_emb = RelPosEmb(
            block_size = block_size,
            rel_size = block_size + (halo_size * 2),
            dim_head = dim_head
        )

        self.to_q  = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

# HaloAttention 类，实现了自注意力机制
    # 定义前向传播函数，接受输入张量 x
    def forward(self, x):
        # 解包输入张量 x 的形状信息，包括批大小 b，通道数 c，高度 h，宽度 w，块大小 block，边界大小 halo，头数 heads，设备信息 device
        b, c, h, w, block, halo, heads, device = *x.shape, self.block_size, self.halo_size, self.heads, x.device
        # 断言高度和宽度能够被块大小整除，确保 fmap 的维度必须是块大小的整数倍
        assert h % block == 0 and w % block == 0, 'fmap dimensions must be divisible by the block size'
        # 断言通道数等于指定的维度
        assert c == self.dim, f'channels for input ({c}) does not equal to the correct dimension ({self.dim})'

        # 获取块的邻域，并为推导键值准备一个带有边界的版本（带有填充的块）

        # 重排输入张量 x，将其形状变为 '(b c (h p1) (w p2) -> (b h w) (p1 p2) c'，其中 p1 和 p2 为块大小
        q_inp = rearrange(x, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1 = block, p2 = block)

        # 使用 F.unfold 函数对 x 进行展开，设置卷积核大小为 block + halo * 2，步长为 block，填充为 halo
        kv_inp = F.unfold(x, kernel_size = block + halo * 2, stride = block, padding = halo)
        # 重排展开后的张量 kv_inp，将其形状变为 '(b (c j) i -> (b i) j c'，其中 j 为块大小
        kv_inp = rearrange(kv_inp, 'b (c j) i -> (b i) j c', c = c)

        # 推导查询、键、值

        # 将 q_inp 输入到 self.to_q 函数中得到查询 q
        q = self.to_q(q_inp)
        # 将 kv_inp 输入到 self.to_kv 函数中得到键 k 和值 v，并按最后一个维度分割成两部分
        k, v = self.to_kv(kv_inp).chunk(2, dim = -1)

        # 分割头部

        # 对查询 q、键 k、值 v 进行重排，将其形状变为 '(b n (h d) -> (b h) n d'，其中 h 为头部数
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = heads), (q, k, v))

        # 缩放

        q *= self.scale

        # 注意力计算

        sim = einsum('b i d, b j d -> b i j', q, k)

        # 添加相对位置偏置

        sim += self.rel_pos_emb(q)

        # 掩码填充（在论文中，他们声称不需要掩码，但是对于填充怎么处理？）

        # 创建全为 1 的掩码张量 mask，形状为 (1, 1, h, w)，设备为 device
        mask = torch.ones(1, 1, h, w, device = device)
        # 使用 F.unfold 函数对 mask 进行展开，设置卷积核大小为 block + (halo * 2)，步长为 block，填充为 halo
        mask = F.unfold(mask, kernel_size = block + (halo * 2), stride = block, padding = halo)
        # 重复 mask 张量，形状变为 '(() j i -> (b i h) () j'，其中 b 为批大小，h 为头部数
        mask = repeat(mask, '() j i -> (b i h) () j', b = b, h = heads)
        # 将 mask 转换为布尔类型张量
        mask = mask.bool()

        # 计算最大负值
        max_neg_value = -torch.finfo(sim.dtype).max
        # 使用 mask 对 sim 进行掩码填充，将 mask 为 True 的位置���充为最大负值
        sim.masked_fill_(mask, max_neg_value)

        # 注意力计算

        attn = sim.softmax(dim = -1)

        # 聚合

        out = einsum('b i j, b j d -> b i d', attn, v)

        # 合并和组合头部

        out = rearrange(out, '(b h) n d -> b n (h d)', h = heads)
        out = self.to_out(out)

        # 将块合并回原始特征图

        out = rearrange(out, '(b h w) (p1 p2) c -> b c (h p1) (w p2)', b = b, h = (h // block), w = (w // block), p1 = block, p2 = block)
        return out
```