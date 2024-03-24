# `.\lucidrains\vector-quantize-pytorch\vector_quantize_pytorch\random_projection_quantizer.py`

```py
import torch
from torch import nn, einsum
import torch.nn.functional as F
from vector_quantize_pytorch.vector_quantize_pytorch import VectorQuantize

from einops import rearrange, repeat, pack, unpack

def exists(val):
    return val is not None

class RandomProjectionQuantizer(nn.Module):
    """ https://arxiv.org/abs/2202.01855 """

    def __init__(
        self,
        *,
        dim,
        codebook_size,
        codebook_dim,
        num_codebooks = 1,
        norm = True,
        **kwargs
    ):
        super().__init__()
        self.num_codebooks = num_codebooks

        # 初始化随机投影矩阵，形状为(num_codebooks, dim, codebook_dim)
        rand_projs = torch.empty(num_codebooks, dim, codebook_dim)
        nn.init.xavier_normal_(rand_projs)

        # 将随机投影矩阵注册为模型的缓冲区
        self.register_buffer('rand_projs', rand_projs)

        # 根据输入参数决定是否进行归一化
        self.norm = nn.LayerNorm(dim, elementwise_affine = False) if norm else nn.Identity()

        # 创建向量量化层
        self.vq = VectorQuantize(
            dim = codebook_dim * num_codebooks,
            heads = num_codebooks,
            codebook_size = codebook_size,
            use_cosine_sim = True,
            separate_codebook_per_head = True,
            **kwargs
        )

    def forward(
        self,
        x,
        indices = None
    ):
        return_loss = exists(indices)

        # 对输入数据进行归一化
        x = self.norm(x)

        # 进行随机投影
        x = einsum('b n d, h d e -> b n h e', x, self.rand_projs)
        x, ps = pack([x], 'b n *')

        # 将向量量化层设置为评估模式
        self.vq.eval()
        # 使用向量量化层处理输入数据
        out = self.vq(x, indices = indices)

        if return_loss:
            _, ce_loss = out
            return ce_loss

        _, indices, _ = out
        return indices
```