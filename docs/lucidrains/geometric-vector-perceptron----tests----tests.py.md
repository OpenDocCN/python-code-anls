# `.\lucidrains\geometric-vector-perceptron\tests\tests.py`

```
# 导入 torch 库
import torch
# 从 geometric_vector_perceptron 库中导入 GVP, GVPDropout, GVPLayerNorm, GVP_MPNN
from geometric_vector_perceptron import GVP, GVPDropout, GVPLayerNorm, GVP_MPNN

# 定义容差值
TOL = 1e-2

# 生成随机旋转矩阵
def random_rotation():
    q, r = torch.qr(torch.randn(3, 3))
    return q

# 计算向量之间的差值矩阵
def diff_matrix(vectors):
    b, _, d = vectors.shape
    diff = vectors[..., None, :] - vectors[:, None, ...]
    return diff.reshape(b, -1, d)

# 测试等变性
def test_equivariance():
    R = random_rotation()

    # 创建 GVP 模型
    model = GVP(
        dim_vectors_in = 1024,
        dim_feats_in = 512,
        dim_vectors_out = 256,
        dim_feats_out = 512
    )

    feats = torch.randn(1, 512)
    vectors = torch.randn(1, 32, 3)

    feats_out, vectors_out = model( (feats, diff_matrix(vectors)) )
    feats_out_r, vectors_out_r = model( (feats, diff_matrix(vectors @ R)) )

    err = ((vectors_out @ R) - vectors_out_r).max()
    assert err < TOL, 'equivariance must be respected'

# 测试所有层类型
def test_all_layer_types():
    R = random_rotation()

    # 创建 GVP 模型
    model = GVP(
        dim_vectors_in = 1024,
        dim_feats_in = 512,
        dim_vectors_out = 256,
        dim_feats_out = 512
    )
    dropout = GVPDropout(0.2)
    layer_norm = GVPLayerNorm(512)

    feats = torch.randn(1, 512)
    message = torch.randn(1, 512)
    vectors = torch.randn(1, 32, 3)

    # GVP 层
    feats_out, vectors_out = model( (feats, diff_matrix(vectors)) )
    assert list(feats_out.shape) == [1, 512] and list(vectors_out.shape) == [1, 256, 3]

    # GVP Dropout
    feats_out, vectors_out = dropout(feats_out, vectors_out)
    assert list(feats_out.shape) == [1, 512] and list(vectors_out.shape) == [1, 256, 3]

    # GVP Layer Norm
    feats_out, vectors_out = layer_norm(feats_out, vectors_out)
    assert list(feats_out.shape) == [1, 512] and list(vectors_out.shape) == [1, 256, 3]

# 测试 MPNN
def test_mpnn():
    # 输入数据
    x = torch.randn(5, 32)
    edge_idx = torch.tensor([[0,2,3,4,1], [1,1,3,3,4]]).long()
    edge_attr = torch.randn(5, 16)
    # 节点 (8 个标量和 8 个向量) || 边 (4 个标量和 3 个向量)
    dropout = 0.1
    # 定义层
    gvp_mpnn = GVP_MPNN(feats_x_in = 8,
                        vectors_x_in = 8,
                        feats_x_out = 8,
                        vectors_x_out = 8, 
                        feats_edge_in = 4,
                        vectors_edge_in = 4,
                        feats_edge_out = 4,
                        vectors_edge_out = 4,
                        dropout=0.1 )
    x_out = gvp_mpnn(x, edge_idx, edge_attr)

    assert x.shape == x_out.shape, "Input and output shapes don't match"

# 主函数入口
if __name__ == "__main__":
    # 执行等变性测试
    test_equivariance()
    # 执行所有层类型测试
    test_all_layer_types()
    # 执行 MPNN 测试
    test_mpnn()
```