# `.\lucidrains\res-mlp-pytorch\res_mlp_pytorch\res_mlp_pytorch.py`

```py
import torch
from torch import nn, einsum
from einops.layers.torch import Rearrange, Reduce

# 导入必要的库

# 定义一个函数，如果输入不是元组，则返回一个包含相同值的元组
def pair(val):
    return (val, val) if not isinstance(val, tuple) else val

# 定义一个仿射变换类
class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, 1, dim))
        self.b = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        return x * self.g + self.b

# 定义一个预仿射后层缩放类
class PreAffinePostLayerScale(nn.Module): # https://arxiv.org/abs/2103.17239
    def __init__(self, dim, depth, fn):
        super().__init__()
        # 根据深度选择初始化值
        if depth <= 18:
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.affine = Affine(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.affine(x)) * self.scale + x

# 定义一个ResMLP模型
def ResMLP(*, image_size, patch_size, dim, depth, num_classes, expansion_factor = 4):
    image_height, image_width = pair(image_size)
    assert (image_height % patch_size) == 0 and (image_width % patch_size) == 0, 'image height and width must be divisible by patch size'
    num_patches = (image_height // patch_size) * (image_width // patch_size)
    wrapper = lambda i, fn: PreAffinePostLayerScale(dim, i + 1, fn)

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        nn.Linear((patch_size ** 2) * 3, dim),
        *[nn.Sequential(
            wrapper(i, nn.Conv1d(num_patches, num_patches, 1)),
            wrapper(i, nn.Sequential(
                nn.Linear(dim, dim * expansion_factor),
                nn.GELU(),
                nn.Linear(dim * expansion_factor, dim)
            ))
        ) for i in range(depth)],
        Affine(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )

# 返回一个包含ResMLP模型结构的序列
```