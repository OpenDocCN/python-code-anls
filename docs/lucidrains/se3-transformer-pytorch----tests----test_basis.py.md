# `.\lucidrains\se3-transformer-pytorch\tests\test_basis.py`

```
# 导入 torch 库
import torch
# 从 se3_transformer_pytorch.basis 模块中导入 get_basis, get_R_tensor, basis_transformation_Q_J 函数
from se3_transformer_pytorch.basis import get_basis, get_R_tensor, basis_transformation_Q_J
# 从 se3_transformer_pytorch.irr_repr 模块中导入 irr_repr 函数

# 定义测试函数 test_basis
def test_basis():
    # 设置最大阶数为 3
    max_degree = 3
    # 生成一个形状为 (2, 1024, 3) 的随机张量
    x = torch.randn(2, 1024, 3)
    # 调用 get_basis 函数获取基函数
    basis = get_basis(x, max_degree)
    # 断言基函数字典的长度是否为 (max_degree + 1) 的平方
    assert len(basis.keys()) == (max_degree + 1) ** 2, 'correct number of basis kernels'

# 定义测试函数 test_basis_transformation_Q_J
def test_basis_transformation_Q_J():
    # 生成一个形状为 (4, 3) 的随机角度张量
    rand_angles = torch.rand(4, 3)
    # 设置 J, order_out, order_in 的值为 1
    J, order_out, order_in = 1, 1, 1
    # 调用 basis_transformation_Q_J 函数获取变换矩阵 Q_J，并转换为浮点型
    Q_J = basis_transformation_Q_J(J, order_in, order_out).float()
    # 断言对于随机角度中的每个角度 (a, b, c)，基函数变换矩阵和不可约表示矩阵的乘积是否与 Q_J 和不可约表示函数的乘积相近
    assert all(torch.allclose(get_R_tensor(order_out, order_in, a, b, c) @ Q_J, Q_J @ irr_repr(J, a, b, c)) for a, b, c in rand_angles)
```