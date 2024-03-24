# `.\lucidrains\invariant-point-attention\tests\invariance.py`

```py
# 导入所需的库
import torch
from torch import nn
from einops import repeat
from invariant_point_attention import InvariantPointAttention, IPABlock
from invariant_point_attention.utils import rot

# 测试不变性点注意力机制的函数
def test_ipa_invariance():
    # 创建不变性点注意力机制对象
    attn = InvariantPointAttention(
        dim = 64,
        heads = 8,
        scalar_key_dim = 16,
        scalar_value_dim = 16,
        point_key_dim = 4,
        point_value_dim = 4
    )

    # 创建随机输入序列、成对表示、掩码
    seq           = torch.randn(1, 256, 64)
    pairwise_repr = torch.randn(1, 256, 256, 64)
    mask          = torch.ones(1, 256).bool()

    # 创建随机旋转和平移
    rotations     = repeat(rot(*torch.randn(3)), 'r1 r2 -> b n r1 r2', b = 1, n = 256)
    translations  = torch.randn(1, 256, 3)

    # 随机旋转，用于测试不变性
    random_rotation = rot(*torch.randn(3))

    # 获取不变性点注意力机制的输出
    attn_out = attn(
        seq,
        pairwise_repr = pairwise_repr,
        rotations = rotations,
        translations = translations,
        mask = mask
    )

    # 获取旋转后的不变性点注意力机制的输出
    rotated_attn_out = attn(
        seq,
        pairwise_repr = pairwise_repr,
        rotations = rotations @ random_rotation,
        translations = translations @ random_rotation,
        mask = mask
    )

    # 输出必须是不变的
    diff = (attn_out - rotated_attn_out).max()
    assert diff <= 1e-6, 'must be invariant to global rotation'

# 测试不变性点注意力机制块的函数
def test_ipa_block_invariance():
    # 创建不变性点注意力机制块对象
    attn = IPABlock(
        dim = 64,
        heads = 8,
        scalar_key_dim = 16,
        scalar_value_dim = 16,
        point_key_dim = 4,
        point_value_dim = 4
    )

    # 创建随机输入序列、成对表示、掩码
    seq           = torch.randn(1, 256, 64)
    pairwise_repr = torch.randn(1, 256, 256, 64)
    mask          = torch.ones(1, 256).bool()

    # 创建随机旋转和平移
    rotations     = repeat(rot(*torch.randn(3)), 'r1 r2 -> b n r1 r2', b = 1, n = 256)
    translations  = torch.randn(1, 256, 3)

    # 随机旋转，用于测试不变性
    random_rotation = rot(*torch.randn(3))

    # 获取不变性点注意力机制块的输出
    attn_out = attn(
        seq,
        pairwise_repr = pairwise_repr,
        rotations = rotations,
        translations = translations,
        mask = mask
    )

    # 获取旋转后的不变性点注意力机制块的输出
    rotated_attn_out = attn(
        seq,
        pairwise_repr = pairwise_repr,
        rotations = rotations @ random_rotation,
        translations = translations @ random_rotation,
        mask = mask
    )

    # 输出必须是不变的
    diff = (attn_out - rotated_attn_out).max()
    assert diff <= 1e-6, 'must be invariant to global rotation'
```