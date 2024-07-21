# `.\pytorch\test\distributions\test_utils.py`

```py
# Owner(s): ["module: distributions"]

# 导入 pytest 库，用于测试框架
import pytest

# 导入 PyTorch 库
import torch

# 从 torch.distributions.utils 中导入矩阵下三角部分与向量互相转换的函数
from torch.distributions.utils import tril_matrix_to_vec, vec_to_tril_matrix

# 从 torch.testing._internal.common_utils 中导入运行测试的辅助函数
from torch.testing._internal.common_utils import run_tests

# 使用 pytest 的 parametrize 装饰器指定不同的矩阵形状作为参数
@pytest.mark.parametrize(
    "shape",
    [
        (2, 2),
        (3, 3),
        (2, 4, 4),
        (2, 2, 4, 4),
    ],
)
def test_tril_matrix_to_vec(shape):
    # 生成指定形状的随机矩阵
    mat = torch.randn(shape)
    # 获取矩阵的最后一个维度大小
    n = mat.shape[-1]
    # 遍历可能的对角线偏移值
    for diag in range(-n, n):
        # 获取矩阵的下三角部分（包含对角线）
        actual = mat.tril(diag)
        # 将下三角部分的矩阵转换为向量表示
        vec = tril_matrix_to_vec(actual, diag)
        # 将向量转换回下三角矩阵表示
        tril_mat = vec_to_tril_matrix(vec, diag)
        # 断言转换后的矩阵与原始下三角部分矩阵相似
        assert torch.allclose(tril_mat, actual)

# 如果脚本作为主程序运行，则执行测试
if __name__ == "__main__":
    run_tests()
```