# `.\lucidrains\equiformer-pytorch\tests\test_equivariance.py`

```
# 导入 pytest 库
import pytest

# 导入 torch 库
import torch
# 导入 Equiformer 类
from equiformer_pytorch.equiformer_pytorch import Equiformer
# 导入 rot 函数
from equiformer_pytorch.irr_repr import rot

# 导入 utils 模块中的函数
from equiformer_pytorch.utils import (
    torch_default_dtype,
    cast_tuple,
    to_order,
    exists
)

# 测试输出形状

# 使用参数化装饰器定义测试函数
@pytest.mark.parametrize('dim', [32])
def test_transformer(dim):
    # 创建 Equiformer 模型对象
    model = Equiformer(
        dim = dim,
        depth = 2,
        num_degrees = 3,
        init_out_zero = False
    )

    # 生成随机输入特征、坐标和掩码
    feats = torch.randn(1, 32, dim)
    coors = torch.randn(1, 32, 3)
    mask  = torch.ones(1, 32).bool()

    # 调用模型进行前向传播
    type0, _ = model(feats, coors, mask)
    # 断言输出形状是否符合预期
    assert type0.shape == (1, 32, dim), 'output must be of the right shape'

# 测试等变性

# 使用参数化装饰器定义测试函数
@pytest.mark.parametrize('dim', [32, (4, 8, 16)])
@pytest.mark.parametrize('dim_in', [32, (32, 32)])
@pytest.mark.parametrize('l2_dist_attention', [True, False])
@pytest.mark.parametrize('reversible', [True, False])
def test_equivariance(
    dim,
    dim_in,
    l2_dist_attention,
    reversible
):
    # 将 dim_in 转换为元组
    dim_in = cast_tuple(dim_in)

    # 创建 Equiformer 模型对象
    model = Equiformer(
        dim = dim,
        dim_in = dim_in,
        input_degrees = len(dim_in),
        depth = 2,
        l2_dist_attention = l2_dist_attention,
        reversible = reversible,
        num_degrees = 3,
        reduce_dim_out = True,
        init_out_zero = False
    )

    # 生成不同度数的随机输入特征
    feats = {deg: torch.randn(1, 32, dim, to_order(deg)) for deg, dim in enumerate(dim_in)}
    type0, type1 = feats[0], feats.get(1, None)

    # 生成随机输入坐标和掩码
    coors = torch.randn(1, 32, 3)
    mask  = torch.ones(1, 32).bool()

    # 生成随机旋转矩阵 R
    R   = rot(*torch.randn(3))

    # 创建可能旋转后的特征字典
    maybe_rotated_feats = {0: type0}

    # 如果存在第二个特征，则将其旋转后加入字典
    if exists(type1):
        maybe_rotated_feats[1] = type1 @ R

    # 调用模型进行前向传播
    _, out1 = model(maybe_rotated_feats, coors @ R, mask)
    out2 = model(feats, coors, mask)[1] @ R

    # 断言两次前向传播结果是否等价
    assert torch.allclose(out1, out2, atol = 1e-4), 'is not equivariant'
```