# `.\lucidrains\En-transformer\tests\test_equivariance.py`

```
# 导入 torch 库
import torch
# 从 en_transformer.utils 模块中导入 rot 函数
from en_transformer.utils import rot
# 从 en_transformer 模块中导入 EnTransformer 类
from en_transformer import EnTransformer

# 设置默认张量数据类型为 float64
torch.set_default_dtype(torch.float64)

# 测试函数，用于测试 README 中的示例
def test_readme():
    # 创建 EnTransformer 模型对象，设置参数
    model = EnTransformer(
        dim = 512,
        depth = 1,
        dim_head = 64,
        heads = 8,
        edge_dim = 4,
        neighbors = 6
    )

    # 生成随机输入特征、坐标和边
    feats = torch.randn(1, 32, 512)
    coors = torch.randn(1, 32, 3)
    edges = torch.randn(1, 32, 1024, 4)

    # 创建掩码张量
    mask = torch.ones(1, 32).bool()

    # 调用模型进行前向传播
    feats, coors = model(feats, coors, edges, mask = mask)
    # 断言测试结果为真
    assert True, 'it runs'

# 测试函数，用于测试等变性
def test_equivariance():
    # 创建 EnTransformer 模型对象，设置参数
    model = EnTransformer(
        dim = 512,
        depth = 1,
        edge_dim = 4,
        rel_pos_emb = True
    )

    # 生成随机旋转矩阵 R 和平移向量 T
    R = rot(*torch.rand(3))
    T = torch.randn(1, 1, 3)

    # 生成随机输入特征、坐标和边
    feats = torch.randn(1, 16, 512)
    coors = torch.randn(1, 16, 3)
    edges = torch.randn(1, 16, 16, 4)

    # 调用模型进行前向传播
    feats1, coors1 = model(feats, coors @ R + T, edges)
    feats2, coors2 = model(feats, coors, edges)

    # 断言特征等变
    assert torch.allclose(feats1, feats2, atol = 1e-6), 'type 0 features are invariant'
    # 断言坐标等变
    assert torch.allclose(coors1, (coors2 @ R + T), atol = 1e-6), 'type 1 features are equivariant'

# 其他测试函数的注释与上述两个测试函数类似，不再重复注释
# 请根据上述示例注释完成以下测试函数

def test_equivariance_with_cross_product():
    model = EnTransformer(
        dim = 512,
        depth = 1,
        edge_dim = 4,
        rel_pos_emb = True,
        use_cross_product = True
    )

    R = rot(*torch.rand(3))
    T = torch.randn(1, 1, 3)

    feats = torch.randn(1, 16, 512)
    coors = torch.randn(1, 16, 3)
    edges = torch.randn(1, 16, 16, 4)

    feats1, coors1 = model(feats, coors @ R + T, edges)
    feats2, coors2 = model(feats, coors, edges)

    assert torch.allclose(feats1, feats2, atol = 1e-6), 'type 0 features are invariant'
    assert torch.allclose(coors1, (coors2 @ R + T), atol = 1e-6), 'type 1 features are equivariant'

def test_equivariance_with_nearest_neighbors():
    model = EnTransformer(
        dim = 512,
        depth = 1,
        edge_dim = 4,
        neighbors = 5
    )

    R = rot(*torch.rand(3))
    T = torch.randn(1, 1, 3)

    feats = torch.randn(1, 16, 512)
    coors = torch.randn(1, 16, 3)
    edges = torch.randn(1, 16, 16, 4)

    feats1, coors1 = model(feats, coors @ R + T, edges)
    feats2, coors2 = model(feats, coors, edges)

    assert torch.allclose(feats1, feats2, atol = 1e-6), 'type 0 features are invariant'
    assert torch.allclose(coors1, (coors2 @ R + T), atol = 1e-6), 'type 1 features are equivariant'

def test_equivariance_with_sparse_neighbors():
    model = EnTransformer(
        dim = 512,
        depth = 1,
        heads = 4,
        dim_head = 32,
        neighbors = 0,
        only_sparse_neighbors = True
    )

    R = rot(*torch.rand(3))
    T = torch.randn(1, 1, 3)

    feats = torch.randn(1, 16, 512)
    coors = torch.randn(1, 16, 3)

    i = torch.arange(feats.shape[1])
    adj_mat = (i[:, None] <= (i[None, :] + 1)) & (i[:, None] >= (i[None, :] - 1))

    feats1, coors1 = model(feats, coors @ R + T, adj_mat = adj_mat)
    feats2, coors2 = model(feats, coors, adj_mat = adj_mat)

    assert torch.allclose(feats1, feats2, atol = 1e-6), 'type 0 features are invariant'
    assert torch.allclose(coors1, (coors2 @ R + T), atol = 1e-6), 'type 1 features are equivariant'

def test_depth():
    model = EnTransformer(
        dim = 8,
        depth = 12,
        edge_dim = 4,
        neighbors = 16
    )

    feats = torch.randn(1, 128, 8)
    coors = torch.randn(1, 128, 3)
    edges = torch.randn(1, 128, 128, 4)

    feats, coors = model(feats, coors, edges)

    assert not torch.any(torch.isnan(feats)), 'no NaN in features'
    assert not torch.any(torch.isnan(coors)), 'no NaN in coordinates'
```