# `.\lucidrains\se3-transformer-pytorch\tests\test_equivariance.py`

```py
import torch
from se3_transformer_pytorch.se3_transformer_pytorch import SE3Transformer
from se3_transformer_pytorch.irr_repr import rot
from se3_transformer_pytorch.utils import torch_default_dtype, fourier_encode

# 测试普通 SE3Transformer 模型
def test_transformer():
    model = SE3Transformer(
        dim = 64,
        depth = 1,
        num_degrees = 2,
        num_neighbors = 4,
        valid_radius = 10
    )

    feats = torch.randn(1, 32, 64)
    coors = torch.randn(1, 32, 3)
    mask  = torch.ones(1, 32).bool()

    out = model(feats, coors, mask, return_type = 0)
    assert out.shape == (1, 32, 64), 'output must be of the right shape'

# 测试有因果性的 SE3Transformer 模型
def test_causal_se3_transformer():
    model = SE3Transformer(
        dim = 64,
        depth = 1,
        num_degrees = 2,
        num_neighbors = 4,
        valid_radius = 10,
        causal = True
    )

    feats = torch.randn(1, 32, 64)
    coors = torch.randn(1, 32, 3)
    mask  = torch.ones(1, 32).bool()

    out = model(feats, coors, mask, return_type = 0)
    assert out.shape == (1, 32, 64), 'output must be of the right shape'

# 测试带全局节点的 SE3Transformer 模型
def test_se3_transformer_with_global_nodes():
    model = SE3Transformer(
        dim = 64,
        depth = 1,
        num_degrees = 2,
        num_neighbors = 4,
        valid_radius = 10,
        global_feats_dim = 16
    )

    feats = torch.randn(1, 32, 64)
    coors = torch.randn(1, 32, 3)
    mask  = torch.ones(1, 32).bool()

    global_feats = torch.randn(1, 2, 16)

    out = model(feats, coors, mask, return_type = 0, global_feats = global_feats)
    assert out.shape == (1, 32, 64), 'output must be of the right shape'

# 测试带单头键值对的 SE3Transformer 模型
def test_one_headed_key_values_se3_transformer_with_global_nodes():
    model = SE3Transformer(
        dim = 64,
        depth = 1,
        num_degrees = 2,
        num_neighbors = 4,
        valid_radius = 10,
        global_feats_dim = 16,
        one_headed_key_values = True
    )

    feats = torch.randn(1, 32, 64)
    coors = torch.randn(1, 32, 3)
    mask  = torch.ones(1, 32).bool()

    global_feats = torch.randn(1, 2, 16)

    out = model(feats, coors, mask, return_type = 0, global_feats = global_feats)
    assert out.shape == (1, 32, 64), 'output must be of the right shape'

# 测试带边的 SE3Transformer 模型
def test_transformer_with_edges():
    model = SE3Transformer(
        dim = 64,
        depth = 1,
        num_degrees = 2,
        num_neighbors = 4,
        edge_dim = 4,
        num_edge_tokens = 4
    )

    feats = torch.randn(1, 32, 64)
    edges = torch.randint(0, 4, (1, 32))
    coors = torch.randn(1, 32, 3)
    mask  = torch.ones(1, 32).bool()

    out = model(feats, coors, mask, edges = edges, return_type = 0)
    assert out.shape == (1, 32, 64), 'output must be of the right shape'

# 测试带连续边的 SE3Transformer 模型
def test_transformer_with_continuous_edges():
    model = SE3Transformer(
        dim = 64,
        depth = 1,
        attend_self = True,
        num_degrees = 2,
        output_degrees = 2,
        edge_dim = 34
    )

    feats = torch.randn(1, 32, 64)
    coors = torch.randn(1, 32, 3)
    mask  = torch.ones(1, 32).bool()

    pairwise_continuous_values = torch.randint(0, 4, (1, 32, 32, 2))

    edges = fourier_encode(
        pairwise_continuous_values,
        num_encodings = 8,
        include_self = True
    )

    out = model(feats, coors, mask, edges = edges, return_type = 1)
    assert True

# 测试不同输入维度的 SE3Transformer 模型
def test_different_input_dimensions_for_types():
    model = SE3Transformer(
        dim_in = (4, 2),
        dim = 4,
        depth = 1,
        input_degrees = 2,
        num_degrees = 2,
        output_degrees = 2,
        reduce_dim_out = True
    )

    atom_feats  = torch.randn(2, 32, 4, 1)
    coors_feats = torch.randn(2, 32, 2, 3)

    features = {'0': atom_feats, '1': coors_feats}
    coors = torch.randn(2, 32, 3)
    mask  = torch.ones(2, 32).bool()

    refined_coors = coors + model(features, coors, mask, return_type = 1)
    assert True

# 测试等变性
def test_equivariance():
    # 创建一个 SE3Transformer 模型对象，设置参数：维度为64，深度为1，自我关注为True，邻居数量为4，角度数量为2，输出角度数量为2，距离进行傅立叶编码为True
    model = SE3Transformer(
        dim = 64,
        depth = 1,
        attend_self = True,
        num_neighbors = 4,
        num_degrees = 2,
        output_degrees = 2,
        fourier_encode_dist = True
    )

    # 生成一个大小为(1, 32, 64)的随机张量作为特征
    feats = torch.randn(1, 32, 64)
    # 生成一个大小为(1, 32, 3)的随机张量作为坐标
    coors = torch.randn(1, 32, 3)
    # 生成一个大小为(1, 32)的全为True的布尔张量作为掩码
    mask  = torch.ones(1, 32).bool()

    # 生成一个旋转矩阵 R，旋转角度为(15, 0, 45)
    R   = rot(15, 0, 45)
    # 使用模型对特征、经过旋转后的坐标、掩码进行前向传播，返回类型为1
    out1 = model(feats, coors @ R, mask, return_type = 1)
    # 使用模型对特征、原始坐标、掩码进行前向传播，返回类型为1，然后再乘以旋转矩阵 R
    out2 = model(feats, coors, mask, return_type = 1) @ R

    # 计算两个输出之间的最大差异
    diff = (out1 - out2).max()
    # 断言差异小于1e-4，如果不成立则抛出异常 'is not equivariant'
    assert diff < 1e-4, 'is not equivariant'
# 测试具有 EGNN 骨干的等变性
def test_equivariance_with_egnn_backbone():
    # 创建 SE3Transformer 模型
    model = SE3Transformer(
        dim = 64,
        depth = 1,
        attend_self = True,
        num_neighbors = 4,
        num_degrees = 2,
        output_degrees = 2,
        fourier_encode_dist = True,
        use_egnn = True
    )

    # 生成随机特征、坐标和掩码
    feats = torch.randn(1, 32, 64)
    coors = torch.randn(1, 32, 3)
    mask  = torch.ones(1, 32).bool()

    # 生成旋转矩阵
    R   = rot(15, 0, 45)
    # 使用旋转后的坐标进行模型推理
    out1 = model(feats, coors @ R, mask, return_type = 1)
    # 使用旋转后的特征进行模型推理，然后再旋转输出
    out2 = model(feats, coors, mask, return_type = 1) @ R

    # 计算输出之间的差异
    diff = (out1 - out2).max()
    # 断言输出的差异小于给定阈值
    assert diff < 1e-4, 'is not equivariant'

# 测试旋转
def test_rotary():
    # 创建 SE3Transformer 模型
    model = SE3Transformer(
        dim = 64,
        depth = 1,
        attend_self = True,
        num_neighbors = 4,
        num_degrees = 2,
        output_degrees = 2,
        fourier_encode_dist = True,
        rotary_position = True,
        rotary_rel_dist = True
    )

    # 生成随机特征、坐标和掩码
    feats = torch.randn(1, 32, 64)
    coors = torch.randn(1, 32, 3)
    mask  = torch.ones(1, 32).bool()

    # 生成旋转矩阵
    R   = rot(15, 0, 45)
    # 使用旋转后的坐标进行模型推理
    out1 = model(feats, coors @ R, mask, return_type = 1)
    # 使用旋转后的特征进行模型推理，然后再旋转输出
    out2 = model(feats, coors, mask, return_type = 1) @ R

    # 计算输出之间的差异
    diff = (out1 - out2).max()
    # 断言输出的差异小于给定阈值
    assert diff < 1e-4, 'is not equivariant'

# 测试等变性线性投影键
def test_equivariance_linear_proj_keys():
    # 创建 SE3Transformer 模型
    model = SE3Transformer(
        dim = 64,
        depth = 1,
        attend_self = True,
        num_neighbors = 4,
        num_degrees = 2,
        output_degrees = 2,
        fourier_encode_dist = True,
        linear_proj_keys = True
    )

    # 生成随机特征、坐标和掩码
    feats = torch.randn(1, 32, 64)
    coors = torch.randn(1, 32, 3)
    mask  = torch.ones(1, 32).bool()

    # 生成旋转矩阵
    R   = rot(15, 0, 45)
    # 使用旋转后的坐标进行模型推理
    out1 = model(feats, coors @ R, mask, return_type = 1)
    # 使用旋转后的特征进行模型推理，然后再旋转输出
    out2 = model(feats, coors, mask, return_type = 1) @ R

    # 计算输出之间的差异
    diff = (out1 - out2).max()
    # 断言输出的差异小于给定阈值
    assert diff < 1e-4, 'is not equivariant'

# 测试仅稀疏邻居的等变性
@torch_default_dtype(torch.float64)
def test_equivariance_only_sparse_neighbors():
    # 创建 SE3Transformer 模型
    model = SE3Transformer(
        dim = 64,
        depth = 1,
        attend_self = True,
        num_degrees = 2,
        output_degrees = 2,
        num_neighbors = 0,
        attend_sparse_neighbors = True,
        num_adj_degrees = 2,
        adj_dim = 4
    )

    # 生成随机特征、坐标和掩码
    feats = torch.randn(1, 32, 64)
    coors = torch.randn(1, 32, 3)
    mask  = torch.ones(1, 32).bool()

    # 生成邻接矩阵
    seq = torch.arange(32)
    adj_mat = (seq[:, None] >= (seq[None, :] - 1)) & (seq[:, None] <= (seq[None, :] + 1))

    # 生成旋转矩阵
    R   = rot(15, 0, 45)
    # 使用旋转后的坐标和邻接矩阵进行模型推理
    out1 = model(feats, coors @ R, mask, adj_mat = adj_mat, return_type = 1)
    # 使用旋转后的特征和邻接矩阵进行模型推理，然后再旋转输出
    out2 = model(feats, coors, mask, adj_mat = adj_mat, return_type = 1) @ R

    # 计算输出之间的差异
    diff = (out1 - out2).max()
    # 断言输出的差异小于给定阈值
    assert diff < 1e-4, 'is not equivariant'

# 测试具有可逆网络的等变性
def test_equivariance_with_reversible_network():
    # 创建 SE3Transformer 模型
    model = SE3Transformer(
        dim = 64,
        depth = 1,
        attend_self = True,
        num_neighbors = 4,
        num_degrees = 2,
        output_degrees = 2,
        reversible = True
    )

    # 生成随机特征、坐标和掩码
    feats = torch.randn(1, 32, 64)
    coors = torch.randn(1, 32, 3)
    mask  = torch.ones(1, 32).bool()

    # 生成旋转矩阵
    R   = rot(15, 0, 45)
    # 使用旋转后的坐标进行模型推理
    out1 = model(feats, coors @ R, mask, return_type = 1)
    # 使用旋转后的特征进行模型推理，然后再旋转输出
    out2 = model(feats, coors, mask, return_type = 1) @ R

    # 计算输出之间的差异
    diff = (out1 - out2).max()
    # 断言输出的差异小于给定阈值
    assert diff < 1e-4, 'is not equivariant'

# 测试具有类型一输入的等变性
def test_equivariance_with_type_one_input():
    # 创建 SE3Transformer 模型
    model = SE3Transformer(
        dim = 64,
        depth = 1,
        attend_self = True,
        num_neighbors = 4,
        num_degrees = 2,
        input_degrees = 2,
        output_degrees = 2
    )

    # 生成随机原子特征和预测坐标
    atom_features = torch.randn(1, 32, 64, 1)
    pred_coors = torch.randn(1, 32, 64, 3)

    # 生成随机坐标和掩码
    coors = torch.randn(1, 32, 3)
    mask  = torch.ones(1, 32).bool()

    # 生成旋转矩阵
    R   = rot(15, 0, 45)
    # 使用旋转后的坐标和预测坐标进行模型推理
    out1 = model({'0': atom_features, '1': pred_coors @ R}, coors @ R, mask, return_type = 1)
    # 使用旋转后的原子特征和预测坐标进行模型推理，然后再旋转输出
    out2 = model({'0': atom_features, '1': pred_coors}, coors, mask, return_type = 1) @ R

    # 计算输出之间的差异
    diff = (out1 - out2).max()
    # 断言输出的差异小于给定阈值
    assert diff < 1e-4, 'is not equivariant'
```