# `.\lucidrains\equiformer-pytorch\tests\test_edges.py`

```
# 导入 pytest 库
import pytest

# 导入 torch 库
import torch
# 从 equiformer_pytorch 包中导入 Equiformer 类
from equiformer_pytorch.equiformer_pytorch import Equiformer
# 从 equiformer_pytorch 包中导入 rot 函数
from equiformer_pytorch.irr_repr import rot
# 从 equiformer_pytorch 包中导入 torch_default_dtype 函数
from equiformer_pytorch.utils import torch_default_dtype

# 测试边的等变性

# 使用 pytest.mark.parametrize 装饰器，参数化测试函数
@pytest.mark.parametrize('l2_dist_attention', [True, False])
@pytest.mark.parametrize('reversible', [True, False])
def test_edges_equivariance(
    l2_dist_attention,
    reversible
):
    # 创建 Equiformer 模型对象
    model = Equiformer(
        num_tokens = 28,
        dim = 64,
        num_edge_tokens = 4,
        edge_dim = 16,
        depth = 2,
        input_degrees = 1,
        num_degrees = 3,
        l2_dist_attention = l2_dist_attention,
        reversible = reversible,
        init_out_zero = False,
        reduce_dim_out = True
    )

    # 生成随机原子索引
    atoms = torch.randint(0, 28, (2, 32))
    # 生成随机键索引
    bonds = torch.randint(0, 4, (2, 32, 32))
    # 生成随机坐标
    coors = torch.randn(2, 32, 3)
    # 创建掩码
    mask  = torch.ones(2, 32).bool()

    # 生成随机旋转矩阵
    R   = rot(*torch.randn(3))
    # 使用模型处理数据，得到输出
    _, out1 = model(atoms, coors @ R, mask, edges = bonds)
    # 使用模型处理数据，得到输出，并进行旋转
    out2 = model(atoms, coors, mask, edges = bonds)[1] @ R

    # 断言输出是否等变
    assert torch.allclose(out1, out2, atol = 1e-4), 'is not equivariant'

# 测试邻接矩阵的等变性

# 使用 pytest.mark.parametrize 装饰器，参数化测试函数
@pytest.mark.parametrize('l2_dist_attention', [True, False])
@pytest.mark.parametrize('reversible', [True, False])
def test_adj_mat_equivariance(
    l2_dist_attention,
    reversible
):
    # 创建 Equiformer 模型对象
    model = Equiformer(
        dim = 32,
        heads = 8,
        depth = 1,
        dim_head = 64,
        num_degrees = 2,
        valid_radius = 10,
        l2_dist_attention = l2_dist_attention,
        reversible = reversible,
        attend_sparse_neighbors = True,
        num_neighbors = 0,
        num_adj_degrees_embed = 2,
        max_sparse_neighbors = 8,
        init_out_zero = False,
        reduce_dim_out = True
    )

    # 生成随机特征
    feats = torch.randn(1, 128, 32)
    # 生成随机坐标
    coors = torch.randn(1, 128, 3)
    # 创建掩码
    mask  = torch.ones(1, 128).bool()

    # 创建邻接矩阵
    i = torch.arange(128)
    adj_mat = (i[:, None] <= (i[None, :] + 1)) & (i[:, None] >= (i[None, :] - 1))

    # 生成随机旋转矩阵
    R   = rot(*torch.randn(3))
    # 使用模型处理数据，得到输出
    _, out1 = model(feats, coors @ R, mask, adj_mat = adj_mat)
    # 使用模型处理数据，得到输出，并进行旋转
    out2 = model(feats, coors, mask, adj_mat = adj_mat)[1] @ R

    # 断言输出是否等变
    assert torch.allclose(out1, out2, atol = 1e-4), 'is not equivariant'
```