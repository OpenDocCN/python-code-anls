# `.\lucidrains\cross-transformers-pytorch\cross_transformers_pytorch\cross_transformers_pytorch.py`

```py
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange

class CrossTransformer(nn.Module):
    def __init__(
        self,
        dim = 512,
        dim_key = 128,
        dim_value = 128
    ):
        # 初始化 CrossTransformer 类
        super().__init__()
        # 设置缩放因子为维度关键字的负平方根
        self.scale = dim_key ** -0.5
        # 将输入转换为查询和键的卷积层
        self.to_qk = nn.Conv2d(dim, dim_key, 1, bias = False)
        # 将输入转换为值的卷积层
        self.to_v = nn.Conv2d(dim, dim_value, 1, bias = False)

    def forward(self, model, img_query, img_supports):
        """
        dimensions names:
        
        b - batch
        k - num classes
        n - num images in a support class
        c - channels
        h, i - height
        w, j - width
        """

        # 获取支持集图像的形状
        b, k, *_ = img_supports.shape

        # 对查询图像进行模型处理
        query_repr = model(img_query)
        *_, h, w = query_repr.shape

        # 重排支持集图像的维度
        img_supports = rearrange(img_supports, 'b k n c h w -> (b k n) c h w', b = b)
        # 对支持集图像进行模型处理
        supports_repr = model(img_supports)

        # 将查询图像转换为查询和值
        query_q, query_v = self.to_qk(query_repr), self.to_v(query_repr)

        # 将支持集图像转换为键和值
        supports_k, supports_v = self.to_qk(supports_repr), self.to_v(supports_repr)
        # 重排支持集图像的维度
        supports_k, supports_v = map(lambda t: rearrange(t, '(b k n) c h w -> b k n c h w', b = b, k = k), (supports_k, supports_v))

        # 计算查询图像和支持集图像之间的相似度
        sim = einsum('b c h w, b k n c i j -> b k h w n i j', query_q, supports_k) * self.scale
        sim = rearrange(sim, 'b k h w n i j -> b k h w (n i j)')

        # 对相似度进行 softmax 操作
        attn = sim.softmax(dim = -1)
        attn = rearrange(attn, 'b k h w (n i j) -> b k h w n i j', i = h, j = w)

        # 计算输出
        out = einsum('b k h w n i j, b k n c i j -> b k c h w', attn, supports_v)

        # 重排输出的维度
        out = rearrange(out, 'b k c h w -> b k (c h w)')
        query_v = rearrange(query_v, 'b c h w -> b () (c h w)')

        # 计算欧氏距离
        euclidean_dist = ((query_v - out) ** 2).sum(dim = -1) / (h * w)
        return -euclidean_dist
```