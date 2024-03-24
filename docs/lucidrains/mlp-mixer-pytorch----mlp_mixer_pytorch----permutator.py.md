# `.\lucidrains\mlp-mixer-pytorch\mlp_mixer_pytorch\permutator.py`

```py
# 导入需要的模块
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce

# 定义一个带有 LayerNorm 的残差块
class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

# 定义一个并行求和的模块
class ParallelSum(nn.Module):
    def __init__(self, *fns):
        super().__init__()
        self.fns = nn.ModuleList(fns)

    def forward(self, x):
        return sum(map(lambda fn: fn(x), self.fns))

# 定义 Permutator 模块，用于生成一个序列模型
def Permutator(*, image_size, patch_size, dim, depth, num_classes, segments, expansion_factor = 4, dropout = 0.):
    # 检查图像大小是否能被分块大小整除
    assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
    # 检查维度是否能被分段数整除
    assert (dim % segments) == 0, 'dimension must be divisible by the number of segments'
    height = width = image_size // patch_size
    s = segments

    return nn.Sequential(
        # 重排输入数据的维度
        Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        # 线性变换
        nn.Linear((patch_size ** 2) * 3, dim),
        # 创建深度为 depth 的模块序列
        *[nn.Sequential(
            # 带有残差连接的预层归一化
            PreNormResidual(dim, nn.Sequential(
                # 并行求和模块
                ParallelSum(
                    nn.Sequential(
                        # 重排数据维度
                        Rearrange('b h w (c s) -> b w c (h s)', s = s),
                        nn.Linear(height * s, height * s),
                        Rearrange('b w c (h s) -> b h w (c s)', s = s),
                    ),
                    nn.Sequential(
                        # 重排数据维度
                        Rearrange('b h w (c s) -> b h c (w s)', s = s),
                        nn.Linear(width * s, width * s),
                        Rearrange('b h c (w s) -> b h w (c s)', s = s),
                    ),
                    nn.Linear(dim, dim)
                ),
                nn.Linear(dim, dim)
            )),
            # 带有残差连接的预层归一化
            PreNormResidual(dim, nn.Sequential(
                nn.Linear(dim, dim * expansion_factor),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * expansion_factor, dim),
                nn.Dropout(dropout)
            ))
        ) for _ in range(depth)],
        # 层归一化
        nn.LayerNorm(dim),
        # 对数据进行降维
        Reduce('b h w c -> b c', 'mean'),
        # 线性变换
        nn.Linear(dim, num_classes)
    )
```