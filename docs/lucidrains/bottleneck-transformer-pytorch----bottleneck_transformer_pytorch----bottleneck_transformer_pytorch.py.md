# `.\lucidrains\bottleneck-transformer-pytorch\bottleneck_transformer_pytorch\bottleneck_transformer_pytorch.py`

```
# 导入 math 和 torch 模块
import math
import torch
# 从 torch 模块中导入 nn 和 einsum 函数
from torch import nn, einsum
# 从 einops 模块中导入 rearrange 函数

# 从 tensorflow 代码翻译而来
# https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2

# 位置嵌入的辅助函数

# 如果 x 不是元组，则返回 (x, x)，否则返回 x
def pair(x):
    return (x, x) if not isinstance(x, tuple) else x

# 在指定维度 dim 上扩展张量 t 的维度为 k
def expand_dim(t, dim, k):
    t = t.unsqueeze(dim = dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

# 将相对位置编码转换为绝对位置编码
def rel_to_abs(x):
    b, h, l, _, device, dtype = *x.shape, x.device, x.dtype
    dd = {'device': device, 'dtype': dtype}
    col_pad = torch.zeros((b, h, l, 1), **dd)
    x = torch.cat((x, col_pad), dim = 3)
    flat_x = rearrange(x, 'b h l c -> b h (l c)')
    flat_pad = torch.zeros((b, h, l - 1), **dd)
    flat_x_padded = torch.cat((flat_x, flat_pad), dim = 2)
    final_x = flat_x_padded.reshape(b, h, l + 1, 2 * l - 1)
    final_x = final_x[:, :, :l, (l-1):]
    return final_x

# 计算相对位置编码的一维相对注意力权重
def relative_logits_1d(q, rel_k):
    b, heads, h, w, dim = q.shape
    logits = einsum('b h x y d, r d -> b h x y r', q, rel_k)
    logits = rearrange(logits, 'b h x y r -> b (h x) y r')
    logits = rel_to_abs(logits)
    logits = logits.reshape(b, heads, h, w, w)
    logits = expand_dim(logits, dim = 3, k = h)
    return logits

# 位置嵌入

# 绝对位置嵌入类
class AbsPosEmb(nn.Module):
    def __init__(
        self,
        fmap_size,
        dim_head
    ):
        super().__init__()
        height, width = pair(fmap_size)
        scale = dim_head ** -0.5
        self.height = nn.Parameter(torch.randn(height, dim_head) * scale)
        self.width = nn.Parameter(torch.randn(width, dim_head) * scale)

    def forward(self, q):
        emb = rearrange(self.height, 'h d -> h () d') + rearrange(self.width, 'w d -> () w d')
        emb = rearrange(emb, ' h w d -> (h w) d')
        logits = einsum('b h i d, j d -> b h i j', q, emb)
        return logits

# 相对位置嵌入类
class RelPosEmb(nn.Module):
    def __init__(
        self,
        fmap_size,
        dim_head
    ):
        super().__init__()
        height, width = pair(fmap_size)
        scale = dim_head ** -0.5
        self.fmap_size = fmap_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        h, w = self.fmap_size

        q = rearrange(q, 'b h (x y) d -> b h x y d', x = h, y = w)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b h x i y j-> b h (x y) (i j)')

        q = rearrange(q, 'b h x y d -> b h y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b h x i y j -> b h (y x) (j i)')
        return rel_logits_w + rel_logits_h

# 注意力机制类
class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        fmap_size,
        heads = 4,
        dim_head = 128,
        rel_pos_emb = False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias = False)

        rel_pos_class = AbsPosEmb if not rel_pos_emb else RelPosEmb
        self.pos_emb = rel_pos_class(fmap_size, dim_head)

    def forward(self, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        q, k, v = self.to_qkv(fmap).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h = heads), (q, k, v))

        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim = sim + self.pos_emb(q)

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return out

class BottleBlock(nn.Module):
    # 初始化函数，设置网络的参数和结构
    def __init__(
        self,
        *,
        dim,
        fmap_size,
        dim_out,
        proj_factor,
        downsample,
        heads = 4,
        dim_head = 128,
        rel_pos_emb = False,
        activation = nn.ReLU()
    ):
        # 调用父类的初始化函数
        super().__init__()

        # shortcut

        # 如果输入维度不等于输出维度或者需要下采样
        if dim != dim_out or downsample:
            # 根据是否下采样设置卷积核大小、步长和填充
            kernel_size, stride, padding = (3, 2, 1) if downsample else (1, 1, 0)

            # 创建 shortcut 层，包括卷积、批归一化和激活函数
            self.shortcut = nn.Sequential(
                nn.Conv2d(dim, dim_out, kernel_size, stride = stride, padding = padding, bias = False),
                nn.BatchNorm2d(dim_out),
                activation
            )
        else:
            # 如果不需要下采样，则使用恒等映射
            self.shortcut = nn.Identity()

        # contraction and expansion

        # 计算注意力机制输入维度和输出维度
        attn_dim_in = dim_out // proj_factor
        attn_dim_out = heads * dim_head

        # 创建网络结构，包括卷积、批归一化、激活函数、注意力机制、平均池化、卷积和批归一化
        self.net = nn.Sequential(
            nn.Conv2d(dim, attn_dim_in, 1, bias = False),
            nn.BatchNorm2d(attn_dim_in),
            activation,
            Attention(
                dim = attn_dim_in,
                fmap_size = fmap_size,
                heads = heads,
                dim_head = dim_head,
                rel_pos_emb = rel_pos_emb
            ),
            nn.AvgPool2d((2, 2)) if downsample else nn.Identity(),
            nn.BatchNorm2d(attn_dim_out),
            activation,
            nn.Conv2d(attn_dim_out, dim_out, 1, bias = False),
            nn.BatchNorm2d(dim_out)
        )

        # 初始化最后一个批归一化层的权重为零
        nn.init.zeros_(self.net[-1].weight)

        # final activation

        # 设置最终的激活函数
        self.activation = activation

    # 前向传播函数
    def forward(self, x):
        # 计算 shortcut
        shortcut = self.shortcut(x)
        # 经过网络结构
        x = self.net(x)
        # 将 shortcut 加到输出上
        x = x + shortcut
        # 返回激活后的输出
        return self.activation(x)
# 定义一个名为BottleStack的类，继承自nn.Module
class BottleStack(nn.Module):
    # 初始化函数，接受一系列参数
    def __init__(
        self,
        *,
        dim,  # 特征维度
        fmap_size,  # 特征图大小
        dim_out = 2048,  # 输出维度，默认为2048
        proj_factor = 4,  # 投影因子，默认为4
        num_layers = 3,  # 层数，默认为3
        heads = 4,  # 多头注意力机制中的头数，默认为4
        dim_head = 128,  # 多头注意力机制中每个头的维度，默认为128
        downsample = True,  # 是否下采样，默认为True
        rel_pos_emb = False,  # 是否使用相对位置编码，默认为False
        activation = nn.ReLU()  # 激活函数，默认为ReLU
    ):
        super().__init__()  # 调用父类的初始化函数

        fmap_size = pair(fmap_size)  # 将特征图大小转换为元组形式

        self.dim = dim  # 初始化特征维度
        self.fmap_size = fmap_size  # 初始化特征图大小

        layers = []  # 初始化一个空列表用于存放层

        # 循环创建num_layers个BottleBlock层
        for i in range(num_layers):
            is_first = i == 0  # 判断是否是第一层
            dim = (dim if is_first else dim_out)  # 如果是第一层，则维度为dim，否则为dim_out
            layer_downsample = is_first and downsample  # 如果是第一层且需要下采样，则为True

            fmap_divisor = (2 if downsample and not is_first else 1)  # 计算特征图大小的除数
            layer_fmap_size = tuple(map(lambda t: t // fmap_divisor, fmap_size))  # 计算当前层的特征图大小

            # 创建一个BottleBlock层，并添加到layers列表中
            layers.append(BottleBlock(
                dim = dim,
                fmap_size = layer_fmap_size,
                dim_out = dim_out,
                proj_factor = proj_factor,
                heads = heads,
                dim_head = dim_head,
                downsample = layer_downsample,
                rel_pos_emb = rel_pos_emb,
                activation = activation
            ))

        # 将所有层组合成一个神经网络
        self.net = nn.Sequential(*layers)

    # 前向传播函数，接受输入x，返回网络输出
    def forward(self, x):
        _, c, h, w = x.shape  # 获取输入x的形状信息
        assert c == self.dim, f'channels of feature map {c} must match channels given at init {self.dim}'  # 断言通道数与初始化时给定的特征维度相匹配
        assert h == self.fmap_size[0] and w == self.fmap_size[1], f'height and width ({h} {w}) of feature map must match the fmap_size given at init {self.fmap_size}'  # 断言特征图的高度和宽度与初始化时给定的特征图大小相匹配
        return self.net(x)  # 返回网络的输出
```