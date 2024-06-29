# `.\numpy\numpy\matrixlib\tests\test_matrix_linalg.py`

```
# 导入 NumPy 库并起一个别名 np
import numpy as np

# 从 NumPy 的线性代数测试模块中导入所需的测试用例和辅助函数
from numpy.linalg.tests.test_linalg import (
    LinalgCase, apply_tag, TestQR as _TestQR, LinalgTestCase,
    _TestNorm2D, _TestNormDoubleBase, _TestNormSingleBase, _TestNormInt64Base,
    SolveCases, InvCases, EigvalsCases, EigCases, SVDCases, CondCases,
    PinvCases, DetCases, LstsqCases)

# 定义一个空列表用于存放测试用例
CASES = []

# 添加 "square" 标签的测试用例到 CASES 列表中
CASES += apply_tag('square', [
    # 测试用例：0x0 矩阵，空的双精度浮点型矩阵视图
    LinalgCase("0x0_matrix",
               np.empty((0, 0), dtype=np.double).view(np.matrix),
               np.empty((0, 1), dtype=np.double).view(np.matrix),
               tags={'size-0'}),
    # 测试用例：仅有矩阵 b
    LinalgCase("matrix_b_only",
               np.array([[1., 2.], [3., 4.]]),
               np.matrix([2., 1.]).T),
    # 测试用例：同时包含矩阵 a 和 b
    LinalgCase("matrix_a_and_b",
               np.matrix([[1., 2.], [3., 4.]]),
               np.matrix([2., 1.]).T),
])

# 添加 "hermitian" 标签的测试用例到 CASES 列表中
CASES += apply_tag('hermitian', [
    # 测试用例：对称矩阵，包含矩阵 a 和 b
    LinalgCase("hmatrix_a_and_b",
               np.matrix([[1., 2.], [2., 1.]]),
               None),
])

# 定义一个测试用例类 MatrixTestCase，继承自 LinalgTestCase，用于统一管理所有测试用例
class MatrixTestCase(LinalgTestCase):
    TEST_CASES = CASES

# 下面是各种具体的测试类，每个类都继承自 MatrixTestCase 并实现特定的测试功能

# 求解矩阵的测试类
class TestSolveMatrix(SolveCases, MatrixTestCase):
    pass

# 求逆矩阵的测试类
class TestInvMatrix(InvCases, MatrixTestCase):
    pass

# 求特征值的测试类
class TestEigvalsMatrix(EigvalsCases, MatrixTestCase):
    pass

# 求特征向量的测试类
class TestEigMatrix(EigCases, MatrixTestCase):
    pass

# 奇异值分解的测试类
class TestSVDMatrix(SVDCases, MatrixTestCase):
    pass

# 矩阵条件数的测试类
class TestCondMatrix(CondCases, MatrixTestCase):
    pass

# 伪逆矩阵的测试类
class TestPinvMatrix(PinvCases, MatrixTestCase):
    pass

# 行列式的测试类
class TestDetMatrix(DetCases, MatrixTestCase):
    pass

# 最小二乘解的测试类
class TestLstsqMatrix(LstsqCases, MatrixTestCase):
    pass

# 二维范数的测试类，使用 np.matrix 作为数组类型
class _TestNorm2DMatrix(_TestNorm2D):
    array = np.matrix

# 双精度范数的测试类，继承自 _TestNorm2DMatrix 和 _TestNormDoubleBase
class TestNormDoubleMatrix(_TestNorm2DMatrix, _TestNormDoubleBase):
    pass

# 单精度范数的测试类，继承自 _TestNorm2DMatrix 和 _TestNormSingleBase
class TestNormSingleMatrix(_TestNorm2DMatrix, _TestNormSingleBase):
    pass

# Int64 范数的测试类，继承自 _TestNorm2DMatrix 和 _TestNormInt64Base
class TestNormInt64Matrix(_TestNorm2DMatrix, _TestNormInt64Base):
    pass

# QR 分解的测试类，使用 np.matrix 作为数组类型
class TestQRMatrix(_TestQR):
    array = np.matrix
```