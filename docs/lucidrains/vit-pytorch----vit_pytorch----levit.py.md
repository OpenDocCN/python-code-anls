# `.\lucidrains\vit-pytorch\vit_pytorch\levit.py`

```py
# 从 math 模块中导入 ceil 函数
from math import ceil

# 导入 torch 模块及相关子模块
import torch
from torch import nn, einsum
import torch.nn.functional as F

# 导入 einops 模块中的 rearrange 和 repeat 函数，以及 torch 子模块中的 Rearrange 类
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# 辅助函数

# 判断变量是否存在的函数
def exists(val):
    return val is not None

# 返回默认值的函数
def default(val, d):
    return val if exists(val) else d

# 将输入值转换为元组的函数
def cast_tuple(val, l = 3):
    val = val if isinstance(val, tuple) else (val,)
    return (*val, *((val[-1],) * max(l - len(val), 0))

# 返回固定值的函数
def always(val):
    return lambda *args, **kwargs: val

# 类

# 前馈神经网络类
class FeedForward(nn.Module):
    def __init__(self, dim, mult, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# 注意力机制类
class Attention(nn.Module):
    def __init__(self, dim, fmap_size, heads = 8, dim_key = 32, dim_value = 64, dropout = 0., dim_out = None, downsample = False):
        super().__init__()
        inner_dim_key = dim_key *  heads
        inner_dim_value = dim_value *  heads
        dim_out = default(dim_out, dim)

        self.heads = heads
        self.scale = dim_key ** -0.5

        self.to_q = nn.Sequential(nn.Conv2d(dim, inner_dim_key, 1, stride = (2 if downsample else 1), bias = False), nn.BatchNorm2d(inner_dim_key))
        self.to_k = nn.Sequential(nn.Conv2d(dim, inner_dim_key, 1, bias = False), nn.BatchNorm2d(inner_dim_key))
        self.to_v = nn.Sequential(nn.Conv2d(dim, inner_dim_value, 1, bias = False), nn.BatchNorm2d(inner_dim_value))

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        out_batch_norm = nn.BatchNorm2d(dim_out)
        nn.init.zeros_(out_batch_norm.weight)

        self.to_out = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(inner_dim_value, dim_out, 1),
            out_batch_norm,
            nn.Dropout(dropout)
        )

        # 位置偏置

        self.pos_bias = nn.Embedding(fmap_size * fmap_size, heads)

        q_range = torch.arange(0, fmap_size, step = (2 if downsample else 1))
        k_range = torch.arange(fmap_size)

        q_pos = torch.stack(torch.meshgrid(q_range, q_range, indexing = 'ij'), dim = -1)
        k_pos = torch.stack(torch.meshgrid(k_range, k_range, indexing = 'ij'), dim = -1)

        q_pos, k_pos = map(lambda t: rearrange(t, 'i j c -> (i j) c'), (q_pos, k_pos))
        rel_pos = (q_pos[:, None, ...] - k_pos[None, :, ...]).abs()

        x_rel, y_rel = rel_pos.unbind(dim = -1)
        pos_indices = (x_rel * fmap_size) + y_rel

        self.register_buffer('pos_indices', pos_indices)

    def apply_pos_bias(self, fmap):
        bias = self.pos_bias(self.pos_indices)
        bias = rearrange(bias, 'i j h -> () h i j')
        return fmap + (bias / self.scale)

    def forward(self, x):
        b, n, *_, h = *x.shape, self.heads

        q = self.to_q(x)
        y = q.shape[2]

        qkv = (q, self.to_k(x), self.to_v(x))
        q, k, v = map(lambda t: rearrange(t, 'b (h d) ... -> b h (...) d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        dots = self.apply_pos_bias(dots)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', h = h, y = y)
        return self.to_out(out)

class Transformer(nn.Module):
    # 初始化函数，设置模型参数和结构
    def __init__(self, dim, fmap_size, depth, heads, dim_key, dim_value, mlp_mult = 2, dropout = 0., dim_out = None, downsample = False):
        # 调用父类的初始化函数
        super().__init__()
        # 如果未指定输出维度，则默认为输入维度
        dim_out = default(dim_out, dim)
        # 初始化一个空的模块列表用于存储每个层
        self.layers = nn.ModuleList([])
        # 判断是否使用注意力机制的残差连接
        self.attn_residual = (not downsample) and dim == dim_out

        # 根据深度循环创建每个层
        for _ in range(depth):
            # 每个层包含一个注意力机制和一个前馈神经网络
            self.layers.append(nn.ModuleList([
                Attention(dim, fmap_size = fmap_size, heads = heads, dim_key = dim_key, dim_value = dim_value, dropout = dropout, downsample = downsample, dim_out = dim_out),
                FeedForward(dim_out, mlp_mult, dropout = dropout)
            ]))
    
    # 前向传播函数，处理输入数据
    def forward(self, x):
        # 遍历每个层
        for attn, ff in self.layers:
            # 如果使用注意力机制的残差连接，则保存输入数据
            attn_res = (x if self.attn_residual else 0)
            # 经过注意力机制处理后，加上残差连接
            x = attn(x) + attn_res
            # 经过前馈神经网络处理后，加上残差连接
            x = ff(x) + x
        # 返回处理后的数据
        return x
# 定义 LeViT 类，继承自 nn.Module
class LeViT(nn.Module):
    # 初始化函数，接收多个参数
    def __init__(
        self,
        *,
        image_size,  # 图像大小
        num_classes,  # 类别数量
        dim,  # 维度
        depth,  # 深度
        heads,  # 头数
        mlp_mult,  # MLP 倍数
        stages = 3,  # 阶段数，默认为 3
        dim_key = 32,  # 键维度，默认为 32
        dim_value = 64,  # 值维度，默认为 64
        dropout = 0.,  # Dropout，默认为 0
        num_distill_classes = None  # 蒸馏类别数量，默认为 None
    ):
        # 调用父类的初始化函数
        super().__init__()

        # 将 dim、depth、heads 转换为元组
        dims = cast_tuple(dim, stages)
        depths = cast_tuple(depth, stages)
        layer_heads = cast_tuple(heads, stages)

        # 断言确保 dimensions、depths、heads 必须是小于指定阶段数的元组
        assert all(map(lambda t: len(t) == stages, (dims, depths, layer_heads))), 'dimensions, depths, and heads must be a tuple that is less than the designated number of stages'

        # 定义卷积嵌入层
        self.conv_embedding = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride = 2, padding = 1),
            nn.Conv2d(32, 64, 3, stride = 2, padding = 1),
            nn.Conv2d(64, 128, 3, stride = 2, padding = 1),
            nn.Conv2d(128, dims[0], 3, stride = 2, padding = 1)
        )

        # 计算特征图大小
        fmap_size = image_size // (2 ** 4)
        layers = []

        # 遍历阶段，构建 Transformer 层
        for ind, dim, depth, heads in zip(range(stages), dims, depths, layer_heads):
            is_last = ind == (stages - 1)
            layers.append(Transformer(dim, fmap_size, depth, heads, dim_key, dim_value, mlp_mult, dropout))

            if not is_last:
                next_dim = dims[ind + 1]
                layers.append(Transformer(dim, fmap_size, 1, heads * 2, dim_key, dim_value, dim_out = next_dim, downsample = True))
                fmap_size = ceil(fmap_size / 2)

        # 构建骨干网络
        self.backbone = nn.Sequential(*layers)

        # 定义池化层
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('... () () -> ...')
        )

        # 定义蒸馏头部
        self.distill_head = nn.Linear(dim, num_distill_classes) if exists(num_distill_classes) else always(None)
        # 定义 MLP 头部
        self.mlp_head = nn.Linear(dim, num_classes)

    # 前向传播函数
    def forward(self, img):
        # 图像经过卷积嵌入层
        x = self.conv_embedding(img)

        # 特征图经过骨干网络
        x = self.backbone(x)        

        # 特征图经过池化层
        x = self.pool(x)

        # 输出结果经过 MLP 头部
        out = self.mlp_head(x)
        # 蒸馏结果经过蒸馏头部
        distill = self.distill_head(x)

        # 如果存在蒸馏结果，则返回输出结果和蒸馏结果
        if exists(distill):
            return out, distill

        # 否则只返回输出结果
        return out
```