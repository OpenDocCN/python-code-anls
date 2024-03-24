# `.\lucidrains\segformer-pytorch\segformer_pytorch\segformer_pytorch.py`

```
# 从 math 模块中导入 sqrt 函数
from math import sqrt
# 从 functools 模块中导入 partial 函数
from functools import partial
# 导入 torch 模块
import torch
# 从 torch 模块中导入 nn、einsum 函数
from torch import nn, einsum
# 从 torch.nn 模块中导入 functional 模块
import torch.nn.functional as F

# 从 einops 模块中导入 rearrange、reduce 函数
from einops import rearrange, reduce
# 从 einops.layers.torch 模块中导入 Rearrange 类

# 定义一个函数，用于判断变量是否存在
def exists(val):
    return val is not None

# 定义一个函数，用于将变量转换为元组
def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth

# 定义一个类 DsConv2d，继承自 nn.Module 类
class DsConv2d(nn.Module):
    # 初始化方法
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride = 1, bias = True):
        super().__init__()
        # 定义一个神经网络序列
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    # 前向传播方法
    def forward(self, x):
        return self.net(x)

# 定义一个类 LayerNorm，继承自 nn.Module 类
class LayerNorm(nn.Module):
    # 初始化方法
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    # 前向传播方法
    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

# 定义一个类 PreNorm，继承自 nn.Module 类
class PreNorm(nn.Module):
    # 初始化方法
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    # 前向传播方法
    def forward(self, x):
        return self.fn(self.norm(x))

# 定义一个类 EfficientSelfAttention，继承自 nn.Module 类
class EfficientSelfAttention(nn.Module):
    # 初始化方法
    def __init__(
        self,
        *,
        dim,
        heads,
        reduction_ratio
    ):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads

        self.to_q = nn.Conv2d(dim, dim, 1, bias = False)
        self.to_kv = nn.Conv2d(dim, dim * 2, reduction_ratio, stride = reduction_ratio, bias = False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias = False)

    # 前向传播方法
    def forward(self, x):
        h, w = x.shape[-2:]
        heads = self.heads

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = heads), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) c -> b (h c) x y', h = heads, x = h, y = w)
        return self.to_out(out)

# 定义一个类 MixFeedForward，继承自 nn.Module 类
class MixFeedForward(nn.Module):
    # 初始化方法
    def __init__(
        self,
        *,
        dim,
        expansion_factor
    ):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            DsConv2d(hidden_dim, hidden_dim, 3, padding = 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )

    # 前向传播方法
    def forward(self, x):
        return self.net(x)

# 定义一个类 MiT，继承自 nn.Module 类
class MiT(nn.Module):
    # 初始化方法
    def __init__(
        self,
        *,
        channels,
        dims,
        heads,
        ff_expansion,
        reduction_ratio,
        num_layers
    # 定义一个继承自 nn.Module 的类
    ):
        # 超类初始化
        super().__init__()
        # 定义每个阶段的卷积核大小、步长和填充
        stage_kernel_stride_pad = ((7, 4, 3), (3, 2, 1), (3, 2, 1), (3, 2, 1))

        # 将通道数和维度组合成一个元组
        dims = (channels, *dims)
        # 将维度两两配对
        dim_pairs = list(zip(dims[:-1], dims[1:]))

        # 初始化阶段列表
        self.stages = nn.ModuleList([])

        # 遍历每个阶段的参数
        for (dim_in, dim_out), (kernel, stride, padding), num_layers, ff_expansion, heads, reduction_ratio in zip(dim_pairs, stage_kernel_stride_pad, num_layers, ff_expansion, heads, reduction_ratio):
            # 创建获取重叠补丁的对象
            get_overlap_patches = nn.Unfold(kernel, stride = stride, padding = padding)
            # 创建重叠补丁嵌入的卷积层
            overlap_patch_embed = nn.Conv2d(dim_in * kernel ** 2, dim_out, 1)

            # 初始化层列表
            layers = nn.ModuleList([])

            # 根据层数循环创建自注意力和前馈网络层
            for _ in range(num_layers):
                layers.append(nn.ModuleList([
                    PreNorm(dim_out, EfficientSelfAttention(dim = dim_out, heads = heads, reduction_ratio = reduction_ratio)),
                    PreNorm(dim_out, MixFeedForward(dim = dim_out, expansion_factor = ff_expansion)),
                ]))

            # 将当前阶段的组件添加到阶段列表中
            self.stages.append(nn.ModuleList([
                get_overlap_patches,
                overlap_patch_embed,
                layers
            ]))

    # 前向传播函数
    def forward(
        self,
        x,
        return_layer_outputs = False
    ):
        # 获取输入张量的高度和宽度
        h, w = x.shape[-2:]

        # 初始化存储每个阶段输出的列表
        layer_outputs = []
        # 遍历每个阶段
        for (get_overlap_patches, overlap_embed, layers) in self.stages:
            # 对输入张量进行重叠补丁提取
            x = get_overlap_patches(x)

            # 计算补丁数量和比例
            num_patches = x.shape[-1]
            ratio = int(sqrt((h * w) / num_patches))
            x = rearrange(x, 'b c (h w) -> b c h w', h = h // ratio)

            # 对补丁进行嵌入
            x = overlap_embed(x)
            # 遍历当前阶段的每一层
            for (attn, ff) in layers:
                x = attn(x) + x
                x = ff(x) + x

            # 将当前阶段的��出添加到列表中
            layer_outputs.append(x)

        # 根据是否返回每个阶段的输出，选择返回值
        ret = x if not return_layer_outputs else layer_outputs
        return ret
class Segformer(nn.Module):
    # 定义 Segformer 类，继承自 nn.Module
    def __init__(
        self,
        *,
        dims = (32, 64, 160, 256),
        heads = (1, 2, 5, 8),
        ff_expansion = (8, 8, 4, 4),
        reduction_ratio = (8, 4, 2, 1),
        num_layers = 2,
        channels = 3,
        decoder_dim = 256,
        num_classes = 4
    ):
        # 初始化函数，接受一系列参数
        super().__init__()
        # 调用父类的初始化函数

        # 将参数转换为长度为4的元组
        dims, heads, ff_expansion, reduction_ratio, num_layers = map(partial(cast_tuple, depth = 4), (dims, heads, ff_expansion, reduction_ratio, num_layers))
        # 断言参数长度为4
        assert all([*map(lambda t: len(t) == 4, (dims, heads, ff_expansion, reduction_ratio, num_layers))]), 'only four stages are allowed, all keyword arguments must be either a single value or a tuple of 4 values'

        # 创建 MiT 模型
        self.mit = MiT(
            channels = channels,
            dims = dims,
            heads = heads,
            ff_expansion = ff_expansion,
            reduction_ratio = reduction_ratio,
            num_layers = num_layers
        )

        # 创建转换到融合层的模块列表
        self.to_fused = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, decoder_dim, 1),
            nn.Upsample(scale_factor = 2 ** i)
        ) for i, dim in enumerate(dims)])

        # 创建转换到分割层的模块
        self.to_segmentation = nn.Sequential(
            nn.Conv2d(4 * decoder_dim, decoder_dim, 1),
            nn.Conv2d(decoder_dim, num_classes, 1),
        )

    def forward(self, x):
        # 前向传播函数
        layer_outputs = self.mit(x, return_layer_outputs = True)

        # 对每个输出应用转换到融合层的模块
        fused = [to_fused(output) for output, to_fused in zip(layer_outputs, self.to_fused)]
        # 在通道维度上拼接融合后的输出
        fused = torch.cat(fused, dim = 1)
        # 返回分割层的输出
        return self.to_segmentation(fused)
```