# `D:\src\scipysrc\scipy\scipy\linalg\tests\test_matmul_toeplitz.py`

```
"""Test functions for linalg.matmul_toeplitz function
"""

# 导入必要的库
import numpy as np
from scipy.linalg import toeplitz, matmul_toeplitz

# 导入断言函数
from pytest import raises as assert_raises
from numpy.testing import assert_allclose

# 定义测试类 TestMatmulToeplitz
class TestMatmulToeplitz:

    # 设置每个测试方法的前置条件
    def setup_method(self):
        self.rng = np.random.RandomState(42)  # 初始化随机数生成器
        self.tolerance = 1.5e-13  # 设置误差容限

    # 测试实数情况
    def test_real(self):
        cases = []

        n = 1
        c = self.rng.normal(size=n)  # 生成长度为 n 的正态分布随机数
        r = self.rng.normal(size=n)  # 生成长度为 n 的正态分布随机数
        x = self.rng.normal(size=(n, 1))  # 生成形状为 (n, 1) 的正态分布随机数
        cases.append((x, c, r, False))  # 将测试数据添加到 cases 列表中

        n = 2
        c = self.rng.normal(size=n)  # 生成长度为 n 的正态分布随机数
        r = self.rng.normal(size=n)  # 生成长度为 n 的正态分布随机数
        x = self.rng.normal(size=(n, 1))  # 生成形状为 (n, 1) 的正态分布随机数
        cases.append((x, c, r, False))  # 将测试数据添加到 cases 列表中

        n = 101
        c = self.rng.normal(size=n)  # 生成长度为 n 的正态分布随机数
        r = self.rng.normal(size=n)  # 生成长度为 n 的正态分布随机数
        x = self.rng.normal(size=(n, 1))  # 生成形状为 (n, 1) 的正态分布随机数
        cases.append((x, c, r, True))  # 将测试数据添加到 cases 列表中

        n = 1000
        c = self.rng.normal(size=n)  # 生成长度为 n 的正态分布随机数
        r = self.rng.normal(size=n)  # 生成长度为 n 的正态分布随机数
        x = self.rng.normal(size=(n, 1))  # 生成形状为 (n, 1) 的正态分布随机数
        cases.append((x, c, r, False))  # 将测试数据添加到 cases 列表中

        n = 100
        c = self.rng.normal(size=n)  # 生成长度为 n 的正态分布随机数
        r = self.rng.normal(size=n)  # 生成长度为 n 的正态分布随机数
        x = self.rng.normal(size=(n, self.rng.randint(1, 10)))  # 生成不定形状的正态分布随机数
        cases.append((x, c, r, False))  # 将测试数据添加到 cases 列表中

        n = 100
        c = self.rng.normal(size=(n, 1))  # 生成形状为 (n, 1) 的正态分布随机数
        r = self.rng.normal(size=(n, 1))  # 生成形状为 (n, 1) 的正态分布随机数
        x = self.rng.normal(size=(n, self.rng.randint(1, 10)))  # 生成不定形状的正态分布随机数
        cases.append((x, c, r, True))  # 将测试数据添加到 cases 列表中

        n = 100
        c = self.rng.normal(size=(n, 1))  # 生成形状为 (n, 1) 的正态分布随机数
        r = None
        x = self.rng.normal(size=(n, self.rng.randint(1, 10)))  # 生成不定形状的正态分布随机数
        cases.append((x, c, r, True, -1))  # 将测试数据添加到 cases 列表中

        n = 100
        c = self.rng.normal(size=(n, 1))  # 生成形状为 (n, 1) 的正态分布随机数
        r = None
        x = self.rng.normal(size=n)  # 生成长度为 n 的正态分布随机数
        cases.append((x, c, r, False))  # 将测试数据添加到 cases 列表中

        n = 101
        c = self.rng.normal(size=n)  # 生成长度为 n 的正态分布随机数
        r = self.rng.normal(size=n - 27)  # 生成长度为 n-27 的正态分布随机数
        x = self.rng.normal(size=(n - 27, 1))  # 生成形状为 (n-27, 1) 的正态分布随机数
        cases.append((x, c, r, True))  # 将测试数据添加到 cases 列表中

        n = 100
        c = self.rng.normal(size=n)  # 生成长度为 n 的正态分布随机数
        r = self.rng.normal(size=n // 4)  # 生成长度为 n//4 的正态分布随机数
        x = self.rng.normal(size=(n // 4, self.rng.randint(1, 10)))  # 生成不定形状的正态分布随机数
        cases.append((x, c, r, True))  # 将测试数据添加到 cases 列表中

        # 对每个测试数据执行测试
        [self.do(*i) for i in cases]

    # 测试复数情况
    def test_complex(self):
        n = 127
        c = self.rng.normal(size=(n, 1)) + self.rng.normal(size=(n, 1)) * 1j  # 生成复数形式的随机数
        r = self.rng.normal(size=(n, 1)) + self.rng.normal(size=(n, 1)) * 1j  # 生成复数形式的随机数
        x = self.rng.normal(size=(n, 3)) + self.rng.normal(size=(n, 3)) * 1j  # 生成复数形式的随机数
        self.do(x, c, r, False)  # 执行测试

        n = 100
        c = self.rng.normal(size=(n, 1)) + self.rng.normal(size=(n, 1)) * 1j  # 生成复数形式的随机数
        r = self.rng.normal(size=(n // 2, 1)) + self.rng.normal(size=(n // 2, 1)) * 1j  # 生成复数形式的随机数
        x = self.rng.normal(size=(n // 2, 3)) + self.rng.normal(size=(n // 2, 3)) * 1j  # 生成复数形式的随机数
        self.do(x, c, r, False)  # 执行测试

    # 测试空输入情况
    def test_empty(self):
        c = []  # 空列表
        r = []  # 空列表
        x = []  # 空列表
        self.do(x, c, r, False)  # 执行测试

        x = np.empty((0, 0))  # 创建一个空的 numpy 数组
        self.do(x, c, r, False)  # 执行测试
    # 定义测试异常情况的方法
    def test_exceptions(self):
        # 设定数组大小为100，生成正态分布随机数填充数组c
        n = 100
        c = self.rng.normal(size=n)
        # 生成正态分布随机数填充数组r，数组长度为2n
        r = self.rng.normal(size=2*n)
        # 生成正态分布随机数填充数组x
        x = self.rng.normal(size=n)
        # 断言调用matmul_toeplitz函数时抛出ValueError异常
        assert_raises(ValueError, matmul_toeplitz, (c, r), x, True)

        # 设定数组大小为100，生成正态分布随机数填充数组c
        n = 100
        c = self.rng.normal(size=n)
        # 生成正态分布随机数填充数组r，数组长度为n
        r = self.rng.normal(size=n)
        # 生成正态分布随机数填充数组x，数组长度为n-1
        x = self.rng.normal(size=n-1)
        # 断言调用matmul_toeplitz函数时抛出ValueError异常
        assert_raises(ValueError, matmul_toeplitz, (c, r), x, True)

        # 设定数组大小为100，生成正态分布随机数填充数组c
        n = 100
        c = self.rng.normal(size=n)
        # 生成正态分布随机数填充数组r，数组长度为n//2
        r = self.rng.normal(size=n//2)
        # 生成正态分布随机数填充数组x，数组长度为n//2-1
        x = self.rng.normal(size=n//2-1)
        # 断言调用matmul_toeplitz函数时抛出ValueError异常
        assert_raises(ValueError, matmul_toeplitz, (c, r), x, True)

    # 对于Toeplitz矩阵，do()方法验证matmul_toeplitz()函数与@运算符的等价性
    def do(self, x, c, r=None, check_finite=False, workers=None):
        # 如果r为None，调用matmul_toeplitz函数计算结果
        if r is None:
            actual = matmul_toeplitz(c, x, check_finite, workers)
        else:
            # 否则，调用matmul_toeplitz函数计算结果，传入参数为(c, r)
            actual = matmul_toeplitz((c, r), x, check_finite)
        # 计算期望结果，使用Toeplitz矩阵的乘法运算符@
        desired = toeplitz(c, r) @ x
        # 断言actual与desired的近似程度，在给定的相对误差和绝对误差下
        assert_allclose(actual, desired,
                        rtol=self.tolerance, atol=self.tolerance)
```