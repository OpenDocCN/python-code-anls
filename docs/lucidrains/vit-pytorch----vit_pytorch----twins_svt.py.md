# `.\lucidrains\vit-pytorch\vit_pytorch\twins_svt.py`

```py
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# 辅助方法

# 根据条件将字典分组
def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

# 根据前缀分组并移除前缀
def group_by_key_prefix_and_remove_prefix(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(lambda x: x.startswith(prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

# 类

# 残差连接
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# 层归一化
class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

# 前馈神经网络
class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            LayerNorm(dim),
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# 图像块嵌入
class PatchEmbedding(nn.Module):
    def __init__(self, *, dim, dim_out, patch_size):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.patch_size = patch_size

        self.proj = nn.Sequential(
            LayerNorm(patch_size ** 2 * dim),
            nn.Conv2d(patch_size ** 2 * dim, dim_out, 1),
            LayerNorm(dim_out)
        )

    def forward(self, fmap):
        p = self.patch_size
        fmap = rearrange(fmap, 'b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = p, p2 = p)
        return self.proj(fmap)

# 像素级注意力
class PEG(nn.Module):
    def __init__(self, dim, kernel_size = 3):
        super().__init__()
        self.proj = Residual(nn.Conv2d(dim, dim, kernel_size = kernel_size, padding = kernel_size // 2, groups = dim, stride = 1))

    def forward(self, x):
        return self.proj(x)

# 局部注意力
class LocalAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., patch_size = 7):
        super().__init__()
        inner_dim = dim_head *  heads
        self.patch_size = patch_size
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = LayerNorm(dim)
        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias = False)
        self.to_kv = nn.Conv2d(dim, inner_dim * 2, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, fmap):
        fmap = self.norm(fmap)

        shape, p = fmap.shape, self.patch_size
        b, n, x, y, h = *shape, self.heads
        x, y = map(lambda t: t // p, (x, y))

        fmap = rearrange(fmap, 'b c (x p1) (y p2) -> (b x y) c p1 p2', p1 = p, p2 = p)

        q, k, v = (self.to_q(fmap), *self.to_kv(fmap).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h d) p1 p2 -> (b h) (p1 p2) d', h = h), (q, k, v))

        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = dots.softmax(dim = - 1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b x y h) (p1 p2) d -> b (h d) (x p1) (y p2)', h = h, x = x, y = y, p1 = p, p2 = p)
        return self.to_out(out)

class GlobalAttention(nn.Module):
    # 初始化函数，设置注意力机制的参数
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., k = 7):
        # 调用父类的初始化函数
        super().__init__()
        # 计算内部维度
        inner_dim = dim_head *  heads
        # 设置头数和缩放因子
        self.heads = heads
        self.scale = dim_head ** -0.5

        # 归一化层
        self.norm = LayerNorm(dim)

        # 转换查询向量
        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias = False)
        # 转换键值对
        self.to_kv = nn.Conv2d(dim, inner_dim * 2, k, stride = k, bias = False)

        # 丢弃部分数据
        self.dropout = nn.Dropout(dropout)

        # 输出层
        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    # 前向传播函数
    def forward(self, x):
        # 对输入数据进行归一化
        x = self.norm(x)

        # 获取输入数据的形状
        shape = x.shape
        b, n, _, y, h = *shape, self.heads
        # 分别计算查询、键、值
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1))

        # 重排查询、键、值的维度
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h = h), (q, k, v))

        # 计算点积
        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale

        # 计算注意力分布
        attn = dots.softmax(dim = -1)
        attn = self.dropout(attn)

        # 计算输出
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, y = y)
        return self.to_out(out)
class Transformer(nn.Module):
    # 定义 Transformer 类，继承自 nn.Module
    def __init__(self, dim, depth, heads = 8, dim_head = 64, mlp_mult = 4, local_patch_size = 7, global_k = 7, dropout = 0., has_local = True):
        # 初始化函数，接受多个参数
        super().__init__()
        # 调用父类的初始化函数
        self.layers = nn.ModuleList([])
        # 初始化 layers 为一个空的 ModuleList
        for _ in range(depth):
            # 循环 depth 次
            self.layers.append(nn.ModuleList([
                # 向 layers 中添加一个 ModuleList
                Residual(LocalAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout, patch_size = local_patch_size)) if has_local else nn.Identity(),
                # 添加 LocalAttention 或者 Identity 到 ModuleList
                Residual(FeedForward(dim, mlp_mult, dropout = dropout)) if has_local else nn.Identity(),
                # 添加 FeedForward 或者 Identity 到 ModuleList
                Residual(GlobalAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout, k = global_k)),
                # 添加 GlobalAttention 到 ModuleList
                Residual(FeedForward(dim, mlp_mult, dropout = dropout))
                # 添加 FeedForward 到 ModuleList
            ]))
        # 循环结束后，layers 中包含 depth 个 ModuleList
    def forward(self, x):
        # 定义 forward 函数，接受输入 x
        for local_attn, ff1, global_attn, ff2 in self.layers:
            # 遍历 layers 中的每个 ModuleList
            x = local_attn(x)
            # 对 x 应用 local_attn
            x = ff1(x)
            # 对 x 应用 ff1
            x = global_attn(x)
            # 对 x 应用 global_attn
            x = ff2(x)
            # 对 x 应用 ff2
        return x
        # 返回处理后的 x

class TwinsSVT(nn.Module):
    # 定义 TwinsSVT 类，继承自 nn.Module
    def __init__(
        self,
        *,
        num_classes,
        s1_emb_dim = 64,
        s1_patch_size = 4,
        s1_local_patch_size = 7,
        s1_global_k = 7,
        s1_depth = 1,
        s2_emb_dim = 128,
        s2_patch_size = 2,
        s2_local_patch_size = 7,
        s2_global_k = 7,
        s2_depth = 1,
        s3_emb_dim = 256,
        s3_patch_size = 2,
        s3_local_patch_size = 7,
        s3_global_k = 7,
        s3_depth = 5,
        s4_emb_dim = 512,
        s4_patch_size = 2,
        s4_local_patch_size = 7,
        s4_global_k = 7,
        s4_depth = 4,
        peg_kernel_size = 3,
        dropout = 0.
    ):
        # 初始化函数，接受多个参数
        super().__init__()
        # 调用父类的初始化函数
        kwargs = dict(locals())
        # 将参数保存为字典

        dim = 3
        # 初始化维度为 3
        layers = []
        # 初始化 layers 为空列表

        for prefix in ('s1', 's2', 's3', 's4'):
            # 遍历前缀列表
            config, kwargs = group_by_key_prefix_and_remove_prefix(f'{prefix}_', kwargs)
            # 从参数字典中提取以当前前缀开头的参数
            is_last = prefix == 's4'
            # 判断是否是最后一个前缀

            dim_next = config['emb_dim']
            # 获取下一个维度

            layers.append(nn.Sequential(
                # 向 layers 中添加一个 Sequential 模块
                PatchEmbedding(dim = dim, dim_out = dim_next, patch_size = config['patch_size']),
                # 添加 PatchEmbedding 到 Sequential
                Transformer(dim = dim_next, depth = 1, local_patch_size = config['local_patch_size'], global_k = config['global_k'], dropout = dropout, has_local = not is_last),
                # 添加 Transformer 到 Sequential
                PEG(dim = dim_next, kernel_size = peg_kernel_size),
                # 添加 PEG 到 Sequential
                Transformer(dim = dim_next, depth = config['depth'],  local_patch_size = config['local_patch_size'], global_k = config['global_k'], dropout = dropout, has_local = not is_last)
                # 添加 Transformer 到 Sequential
            ))

            dim = dim_next
            # 更新维度为下一个维度

        self.layers = nn.Sequential(
            # 将 layers 中的模块组合成一个 Sequential
            *layers,
            # 展开 layers 中的模块
            nn.AdaptiveAvgPool2d(1),
            # 添加 AdaptiveAvgPool2d 到 Sequential
            Rearrange('... () () -> ...'),
            # 添加 Rearrange 到 Sequential
            nn.Linear(dim, num_classes)
            # 添加 Linear 到 Sequential
        )

    def forward(self, x):
        # 定义 forward 函数，接受输入 x
        return self.layers(x)
        # 返回处理后的 x
```