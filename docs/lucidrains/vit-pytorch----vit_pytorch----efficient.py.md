# `.\lucidrains\vit-pytorch\vit_pytorch\efficient.py`

```py
import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# 定义一个函数，用于确保输入是一个元组
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# 定义一个名为 ViT 的类，继承自 nn.Module
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, transformer, pool = 'cls', channels = 3):
        super().__init__()
        image_size_h, image_size_w = pair(image_size)
        # 检查图像尺寸是否能被 patch 大小整除
        assert image_size_h % patch_size == 0 and image_size_w % patch_size == 0, 'image dimensions must be divisible by the patch size'
        # 检查池化类型是否为 'cls' 或 'mean'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_patches = (image_size_h // patch_size) * (image_size_w // patch_size)
        patch_dim = channels * patch_size ** 2

        # 定义将图像切片转换为嵌入向量的序列
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        # 初始化位置编码和类别标记
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = transformer

        self.pool = pool
        self.to_latent = nn.Identity()

        # 定义 MLP 头部
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    # 前向传播函数
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # 重复类别标记以匹配批次大小
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.transformer(x)

        # 根据池化类型选择不同的池化方式
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
```