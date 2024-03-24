# `.\lucidrains\axial-positional-embedding\axial_positional_embedding\axial_positional_embedding.py`

```py
# 导入 torch 库
import torch
# 从 torch 库中导入 nn 模块
from torch import nn
# 从 operator 模块中导入 mul 函数
from operator import mul
# 从 functools 模块中导入 reduce 函数
from functools import reduce

# 定义 AxialPositionalEmbedding 类，继承自 nn.Module
class AxialPositionalEmbedding(nn.Module):
    # 初始化函数，接受维度 dim、轴形状 axial_shape 和轴维度 axial_dims
    def __init__(self, dim, axial_shape, axial_dims = None):
        super().__init__()

        # 初始化对象的属性
        self.dim = dim
        self.shape = axial_shape
        self.max_seq_len = reduce(mul, axial_shape, 1)

        # 判断是否需要对轴维度进行求和
        self.summed = axial_dims is None
        axial_dims = ((dim,) * len(axial_shape)) if self.summed else axial_dims

        # 断言轴形状和轴维度的长度相等
        assert len(self.shape) == len(axial_dims), 'number of axial dimensions must equal the number of dimensions in the shape'
        # 断言轴维度的总和等于目标维度
        assert self.summed or not self.summed and sum(axial_dims) == dim, f'axial dimensions must sum up to the target dimension {dim}'

        # 初始化权重列表
        self.weights = ParameterList(self, 'weights', len(axial_shape))

        # 遍历轴形状和轴维度，创建轴位置嵌入
        for ind, (shape, axial_dim) in enumerate(zip(self.shape, axial_dims)):
            ax_shape = [1] * len(self.shape)
            ax_shape[ind] = shape
            ax_shape = (1, *ax_shape, axial_dim)
            ax_emb = nn.Parameter(torch.zeros(ax_shape).normal_(0, 1))
            self.weights.append(ax_emb)

    # 前向传播函数
    def forward(self, x):
        b, t, e = x.shape
        # 断言序列长度小于等于最大序列长度
        assert (t <= self.max_seq_len), f'Sequence length ({t}) must be less than the maximum sequence length allowed ({self.max_seq_len})'
        embs = []

        # 遍历权重列表，扩展维度并拼接轴位置嵌入
        for ax_emb in self.weights.to_list():
            axial_dim = ax_emb.shape[-1]
            expand_shape = (b, *self.shape, axial_dim)
            emb = ax_emb.expand(expand_shape).reshape(b, self.max_seq_len, axial_dim)
            embs.append(emb)

        # 求和或拼接轴位置嵌入
        pos_emb = sum(embs) if self.summed else torch.cat(embs, dim=-1)
        return pos_emb[:, :t].to(x)

# 一个模拟参数列表对象，直到下面的问题得到解决
# https://github.com/pytorch/pytorch/issues/36035
class ParameterList(object):
    def __init__(self, kls, prefix, length):
        self.ind = 0
        self.kls = kls
        self.prefix = prefix
        self.length = length

    def _keyname(self, prefix, ind):
        return f'{prefix}_{ind}'

    def append(self, x):
        setattr(self.kls, self._keyname(self.prefix, self.ind), x)
        self.ind += 1

    def to_list(self):
        return [getattr(self.kls, self._keyname(self.prefix, i)) for i in range(self.length)]

# 为图像定义 AxialPositionalEmbedding 类

class AxialPositionalEmbeddingImage(nn.Module):
    def __init__(self, dim, axial_shape, axial_dims = None):
        super().__init__()
        # 断言轴形状必须有 2 个维度，适用于图像
        assert len(axial_shape) == 2, 'Axial shape must have 2 dimensions for images'
        # 创建 AxialPositionalEmbedding 对象
        self.pos_emb = AxialPositionalEmbedding(dim, axial_shape, axial_dims)

    # 前向传播函数
    def forward(self, img):
        b, c, h, w = img.shape
        img = img.permute(0, 2, 3, 1).reshape(b, h * w, c)
        pos_emb = self.pos_emb(img)
        return pos_emb.reshape(b, h, w, c).permute(0, 3, 1, 2)
```