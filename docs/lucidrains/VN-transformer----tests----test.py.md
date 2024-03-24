# `.\lucidrains\VN-transformer\tests\test.py`

```
# 导入 pytest 库
import pytest

# 导入 torch 库
import torch
# 从 VN_transformer 模块中导入 VNTransformer、VNInvariant、VNAttention 类和 rot 函数
from VN_transformer.VN_transformer import VNTransformer, VNInvariant, VNAttention
from VN_transformer.rotations import rot

# 设置默认的 torch 数据类型为 float64
torch.set_default_dtype(torch.float64)

# 测试不变层
def test_vn_invariant():
    # 创建一个 VNInvariant 层对象，输入维度为 64
    layer = VNInvariant(64)

    # 生成一个形状为 (1, 32, 64, 3) 的随机张量
    coors = torch.randn(1, 32, 64, 3)

    # 生成一个随机旋转矩阵 R
    R = rot(*torch.randn(3))
    # 对输入张量和经过旋转的输入张量进行 VNInvariant 层的计算
    out1 = layer(coors)
    out2 = layer(coors @ R)

    # 检查经过不变层计算的两个输出张量是否在给定的容差范围内相等
    assert torch.allclose(out1, out2, atol = 1e-6)

# 测试等变性
@pytest.mark.parametrize('l2_dist_attn', [True, False])
def test_equivariance(l2_dist_attn):

    # 创建一个 VNTransformer 模型对象，设置相关参数
    model = VNTransformer(
        dim = 64,
        depth = 2,
        dim_head = 64,
        heads = 8,
        l2_dist_attn = l2_dist_attn
    )

    # 生成一个形状为 (1, 32, 3) 的随机张量
    coors = torch.randn(1, 32, 3)
    # 创建一个形状为 (1, 32) 的全为 True 的布尔张量
    mask  = torch.ones(1, 32).bool()

    # 生成一个随机旋转矩阵 R
    R   = rot(*torch.randn(3))
    # 对输入张量和经过旋转的输入张量进行 VNTransformer 模型的计算
    out1 = model(coors @ R, mask = mask)
    out2 = model(coors, mask = mask) @ R

    # 检查经过模型计算的两个输出张量是否在给定的容差范围内相等
    assert torch.allclose(out1, out2, atol = 1e-6), 'is not equivariant'

# 测试 VN Perceiver 注意力等变性
@pytest.mark.parametrize('l2_dist_attn', [True, False])
def test_perceiver_vn_attention_equivariance(l2_dist_attn):

    # 创建一个 VNAttention 模型对象，设置相关参数
    model = VNAttention(
        dim = 64,
        dim_head = 64,
        heads = 8,
        num_latents = 2,
        l2_dist_attn = l2_dist_attn
    )

    # 生成一个形状为 (1, 32, 64, 3) 的随机张量
    coors = torch.randn(1, 32, 64, 3)
    # 创建一个形状为 (1, 32) 的全为 True 的布尔张量
    mask  = torch.ones(1, 32).bool()

    # 生成一个随机旋转矩阵 R
    R   = rot(*torch.randn(3))
    # 对输入张量和经过旋转的输入张量进行 VNAttention 模型的计算
    out1 = model(coors @ R, mask = mask)
    out2 = model(coors, mask = mask) @ R

    # ��查输出张量的形状是否符合预期
    assert out1.shape[1] == 2
    # 检查经过模型计算的两个输出张量是否在给定的容差范围内相等
    assert torch.allclose(out1, out2, atol = 1e-6), 'is not equivariant'

# 测试 SO(3) 早期融合等变性
@pytest.mark.parametrize('l2_dist_attn', [True, False])
def test_equivariance_with_early_fusion(l2_dist_attn):

    # 创建一个 VNTransformer 模型对象，设置相关参数
    model = VNTransformer(
        dim = 64,
        depth = 2,
        dim_head = 64,
        heads = 8,
        dim_feat = 64,
        l2_dist_attn = l2_dist_attn
    )

    # 生成一个形状为 (1, 32, 64) 的随机张量
    feats = torch.randn(1, 32, 64)
    # 生成一个形状为 (1, 32, 3) 的随机张量
    coors = torch.randn(1, 32, 3)
    # 创建一个形状为 (1, 32) 的全为 True 的布尔张量
    mask  = torch.ones(1, 32).bool()

    # 生成一个随机旋转矩阵 R
    R   = rot(*torch.randn(3))
    # 对输入张量和特征张量进行 VNTransformer 模型的计算
    out1, _ = model(coors @ R, feats = feats, mask = mask, return_concatted_coors_and_feats = False)

    out2, _ = model(coors, feats = feats, mask = mask, return_concatted_coors_and_feats = False)
    out2 = out2 @ R

    # 检查经过模型计算的两个输出张量是否在给定的容差范围内相等
    assert torch.allclose(out1, out2, atol = 1e-6), 'is not equivariant'

# 测试 SE(3) 早期融合等变性
@pytest.mark.parametrize('l2_dist_attn', [True, False])
def test_se3_equivariance_with_early_fusion(l2_dist_attn):

    # 创建一个 VNTransformer 模型对象，设置相关参数
    model = VNTransformer(
        dim = 64,
        depth = 2,
        dim_head = 64,
        heads = 8,
        dim_feat = 64,
        translation_equivariance = True,
        l2_dist_attn = l2_dist_attn
    )

    # 生成一个形状为 (1, 32, 64) 的随机张量
    feats = torch.randn(1, 32, 64)
    # 生成一个形状为 (1, 32, 3) 的随机张量
    coors = torch.randn(1, 32, 3)
    # 创建一个形状为 (1, 32) 的全为 True 的布尔张量
    mask  = torch.ones(1, 32).bool()

    # 生成一个随机平移向量 T 和旋转矩阵 R
    T   = torch.randn(3)
    R   = rot(*torch.randn(3))
    # 对输入张量和特征张量进行 VNTransformer 模型的计算
    out1, _ = model((coors + T) @ R, feats = feats, mask = mask, return_concatted_coors_and_feats = False)

    out2, _ = model(coors, feats = feats, mask = mask, return_concatted_coors_and_feats = False)
    out2 = (out2 + T) @ R

    # 检查经过模型计算的两个输出张量是否在给定的容差范围内相等
    assert torch.allclose(out1, out2, atol = 1e-6), 'is not equivariant'
```