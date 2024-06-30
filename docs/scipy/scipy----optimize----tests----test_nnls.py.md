# `D:\src\scipysrc\scipy\scipy\optimize\tests\test_nnls.py`

```
# 导入 NumPy 库，并将其别名为 np
import numpy as np
# 从 NumPy 的测试模块中导入 assert_allclose 函数，用于比较数组是否几乎相等
from numpy.testing import assert_allclose
# 从 pytest 库中导入 raises 函数，并将其别名为 assert_raises，用于检查是否引发异常
from pytest import raises as assert_raises
# 从 SciPy 库中导入 nnls 函数，用于执行非负最小二乘法
from scipy.optimize import nnls


# 定义测试类 TestNNLS
class TestNNLS:
    # 在每个测试方法执行前运行的设置方法
    def setup_method(self):
        # 创建一个随机数生成器 rng，并将其赋给实例变量 self.rng
        self.rng = np.random.default_rng(1685225766635251)

    # 测试用例：测试 nnls 函数的基本功能
    def test_nnls(self):
        # 创建一个 5x5 的二维数组 a，其元素为 0 到 24
        a = np.arange(25.0).reshape(-1, 5)
        # 创建一个长度为 5 的一维数组 x，其元素为 0 到 4
        x = np.arange(5.0)
        # 创建一维数组 y，通过矩阵乘法计算 a @ x 的结果
        y = a @ x
        # 调用 nnls 函数计算非负最小二乘解 x 和残差 res
        x, res = nnls(a, y)
        # 断言残差 res 小于 1e-7
        assert res < 1e-7
        # 断言通过矩阵乘法计算得到的 a @ x 与 y 的差的二范数小于 1e-7
        assert np.linalg.norm((a @ x) - y) < 1e-7

    # 测试用例：测试 nnls 函数对于高身瘦矩阵的处理能力
    def test_nnls_tall(self):
        # 创建一个大小为 50x10 的二维数组 a，其元素为 -10 到 10 之间的均匀分布随机数
        a = self.rng.uniform(low=-10, high=10, size=[50, 10])
        # 创建一个大小为 10 的一维数组 x，其元素为 -2 到 2 之间的均匀分布随机数的绝对值
        x = np.abs(self.rng.uniform(low=-2, high=2, size=[10]))
        # 将 x 中索引为偶数的元素置为 0
        x[::2] = 0
        # 创建一维数组 b，通过矩阵乘法计算 a @ x 的结果
        b = a @ x
        # 调用 nnls 函数计算非负最小二乘解 xact 和残差 rnorm
        xact, rnorm = nnls(a, b, atol=500*np.linalg.norm(a, 1)*np.spacing(1.))
        # 使用 assert_allclose 函数断言 xact 和 x 在绝对误差小于等于 1e-10 的情况下几乎相等
        assert_allclose(xact, x, rtol=0., atol=1e-10)
        # 断言残差 rnorm 小于 1e-12
        assert rnorm < 1e-12

    # 测试用例：测试 nnls 函数对于宽矩阵的处理能力
    def test_nnls_wide(self):
        # 创建一个大小为 100x120 的二维数组 a，其元素为 -10 到 10 之间的均匀分布随机数
        a = self.rng.uniform(low=-10, high=10, size=[100, 120])
        # 创建一个大小为 120 的一维数组 x，其元素为 -2 到 2 之间的均匀分布随机数的绝对值
        x = np.abs(self.rng.uniform(low=-2, high=2, size=[120]))
        # 将 x 中索引为偶数的元素置为 0
        x[::2] = 0
        # 创建一维数组 b，通过矩阵乘法计算 a @ x 的结果
        b = a @ x
        # 调用 nnls 函数计算非负最小二乘解 xact 和残差 rnorm
        xact, rnorm = nnls(a, b, atol=500*np.linalg.norm(a, 1)*np.spacing(1.))
        # 使用 assert_allclose 函数断言 xact 和 x 在绝对误差小于等于 1e-10 的情况下几乎相等
        assert_allclose(xact, x, rtol=0., atol=1e-10)
        # 断言残差 rnorm 小于 1e-12
        assert rnorm < 1e-12

    # 测试用例：测试 nnls 函数的 maxiter 参数能否正确停止迭代
    def test_maxiter(self):
        # 创建大小为 (5, 10) 的二维数组 a 和大小为 5 的一维数组 b，其元素为均匀分布随机数
        a = self.rng.uniform(size=(5, 10))
        b = self.rng.uniform(size=5)
        # 使用 assert_raises 上下文管理器断言调用 nnls 函数时会引发 RuntimeError 异常
        with assert_raises(RuntimeError):
            nnls(a, b, maxiter=1)
```