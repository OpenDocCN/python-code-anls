# `.\numpy\numpy\matrixlib\tests\test_numeric.py`

```py
import numpy as np                              # 导入NumPy库，用于科学计算
from numpy.testing import assert_equal          # 从NumPy的测试模块导入断言函数assert_equal，用于比较测试结果

class TestDot:
    def test_matscalar(self):
        b1 = np.matrix(np.ones((3, 3), dtype=complex))  # 创建一个3x3的复数矩阵，元素值为1
        assert_equal(b1*1.0, b1)                        # 断言：矩阵b1乘以1.0应该等于原始矩阵b1本身

def test_diagonal():
    b1 = np.matrix([[1,2],[3,4]])               # 创建一个2x2的NumPy矩阵b1
    diag_b1 = np.matrix([[1, 4]])              # 创建一个1x2的矩阵diag_b1，表示b1的对角线元素
    array_b1 = np.array([1, 4])                 # 创建一个NumPy数组array_b1，表示b1的对角线元素

    assert_equal(b1.diagonal(), diag_b1)        # 断言：b1对象的对角线元素应该与diag_b1矩阵相等
    assert_equal(np.diagonal(b1), array_b1)     # 断言：b1矩阵的对角线元素应该与array_b1数组相等
    assert_equal(np.diag(b1), array_b1)         # 断言：使用np.diag()函数提取b1矩阵的对角线元素，应该与array_b1数组相等
```