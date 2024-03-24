# `.\lucidrains\mlp-mixer-pytorch\mlp_mixer_pytorch\mlp_mixer_pytorch.py`

```
# 导入需要的模块
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce

# 定义一个 lambda 函数，用于确保输入是元组类型
pair = lambda x: x if isinstance(x, tuple) else (x, x)

# 定义一个预标准化残差块
class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

# 定义一个前馈神经网络层
def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )

# 定义一个MLP-Mixer模型
def MLPMixer(*, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size) * (image_w // patch_size)
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        # 重排输入数据，将图像分成多个 patch，并将通道维度放在最后
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        # 将每个 patch 的像素值映射到指定维度
        nn.Linear((patch_size ** 2) * channels, dim),
        # 创建多个深度为 depth 的块
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
        ) for _ in range(depth)],
        # 对输出进行标准化
        nn.LayerNorm(dim),
        # 对每个 patch 的特征进行平均池化
        Reduce('b n c -> b c', 'mean'),
        # 将特征映射到类别数量的维度
        nn.Linear(dim, num_classes)
    )
```