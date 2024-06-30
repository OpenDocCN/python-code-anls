# `D:\src\scipysrc\scipy\scipy\optimize\tests\test_isotonic_regression.py`

```
# 导入必要的库和模块
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest

# 从 scipy.optimize._pava_pybind 模块导入 pava 函数
from scipy.optimize._pava_pybind import pava
# 从 scipy.optimize 模块导入 isotonic_regression 函数
from scipy.optimize import isotonic_regression

# 定义测试类 TestIsotonicRegression
class TestIsotonicRegression:
    # 使用 pytest.mark.parametrize 标记的参数化测试函数，测试参数异常情况
    @pytest.mark.parametrize(
        ("y", "w", "msg"),
        [
            ([[0, 1]], None,
             "array has incorrect number of dimensions: 2; expected 1"),
            ([0, 1], [[1, 2]],
             "Input arrays y and w must have one dimension of equal length"),
            ([0, 1], [1],
             "Input arrays y and w must have one dimension of equal length"),
            (1, [1, 2],
             "Input arrays y and w must have one dimension of equal length"),
            ([1, 2], 1,
             "Input arrays y and w must have one dimension of equal length"),
            ([0, 1], [0, 1],
             "Weights w must be strictly positive"),
        ]
    )
    # 测试函数，验证是否能正确抛出预期的 ValueError 异常
    def test_raise_error(self, y, w, msg):
        with pytest.raises(ValueError, match=msg):
            isotonic_regression(y=y, weights=w)

    # 测试简单的 PAVA 算法示例
    def test_simple_pava(self):
        # 设置输入数据 y 和权重 w
        # y 是浮点数数组
        y = np.array([8, 4, 8, 2, 2, 0, 8], dtype=np.float64)
        w = np.ones_like(y)
        # 设置输出结果数组 r
        r = np.full(shape=y.shape[0] + 1, fill_value=-1, dtype=np.intp)
        # 调用 pava 函数进行 PAVA 算法计算
        pava(y, w, r)
        # 验证输出 y 是否符合预期
        assert_allclose(y, [4, 4, 4, 4, 4, 4, 8])
        # 验证输出 w 的前两个元素是否符合预期
        assert_allclose(w, [6, 1, 1, 1, 1, 1, 1])
        # 验证输出 r 的前三个元素是否符合预期
        assert_allclose(r, [0, 6, 7, -1, -1, -1, -1, -1])

    # 使用 pytest.mark.parametrize 标记的参数化测试函数，测试不同数据类型和权重类型的情况
    @pytest.mark.parametrize("y_dtype", [np.float64, np.float32, np.int64, np.int32])
    @pytest.mark.parametrize("w_dtype", [np.float64, np.float32, np.int64, np.int32])
    @pytest.mark.parametrize("w", [None, "ones"])
    def test_simple_isotonic_regression(self, w, w_dtype, y_dtype):
        # 设置输入数据 y 和权重 w
        y = np.array([8, 4, 8, 2, 2, 0, 8], dtype=y_dtype)
        if w is not None:
            w = np.ones_like(y, dtype=w_dtype)
        # 调用 isotonic_regression 函数进行保序回归计算
        res = isotonic_regression(y, weights=w)
        # 验证返回结果的数据类型是否符合预期
        assert res.x.dtype == np.float64
        assert res.weights.dtype == np.float64
        # 验证返回结果的 x 是否符合预期
        assert_allclose(res.x, [4, 4, 4, 4, 4, 4, 8])
        # 验证返回结果的 weights 是否符合预期
        assert_allclose(res.weights, [6, 1])
        # 验证返回结果的 blocks 是否符合预期
        assert_allclose(res.blocks, [0, 6, 7])
        # 验证输入数据 y 是否未被改写
        assert_equal(y, np.array([8, 4, 8, 2, 2, 0, 8], dtype=np.float64))

    # 使用 pytest.mark.parametrize 标记的参数化测试函数，测试保序回归中的 linspace 函数情况
    @pytest.mark.parametrize("increasing", [True, False])
    def test_linspace(self, increasing):
        # 设置 linspace 的参数
        n = 10
        y = np.linspace(0, 1, n) if increasing else np.linspace(1, 0, n)
        # 调用 isotonic_regression 函数进行保序回归计算
        res = isotonic_regression(y, increasing=increasing)
        # 验证返回结果的 x 是否符合预期
        assert_allclose(res.x, y)
        # 验证返回结果的 blocks 是否符合预期
        assert_allclose(res.blocks, np.arange(n + 1))
    # 定义测试函数，用于验证 isotonic_regression 函数在给定权重下的输出结果是否正确
    def test_weights(self):
        # 创建输入数据和权重数组
        w = np.array([1, 2, 5, 0.5, 0.5, 0.5, 1, 3])
        y = np.array([3, 2, 1, 10, 9, 8, 20, 10])
        # 调用 isotonic_regression 函数进行回归计算
        res = isotonic_regression(y, weights=w)
        # 断言确保回归结果的 x 值接近预期值
        assert_allclose(res.x, [12/8, 12/8, 12/8, 9, 9, 9, 50/4, 50/4])
        # 断言确保回归结果的权重值接近预期值
        assert_allclose(res.weights, [8, 1.5, 4])
        # 断言确保回归结果的分块情况接近预期值
        assert_allclose(res.blocks, [0, 3, 6, 8])

        # 在第二组测试中，权重类似于重复的观测，重复第三个元素 5 次
        w2 = np.array([1, 2, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 1, 3])
        y2 = np.array([3, 2, 1, 1, 1, 1, 1, 10, 9, 8, 20, 10])
        # 调用 isotonic_regression 函数进行回归计算
        res2 = isotonic_regression(y2, weights=w2)
        # 断言确保回归结果的前七个 x 值的差接近零
        assert_allclose(np.diff(res2.x[0:7]), 0)
        # 断言确保回归结果的部分 x 值接近第一组测试的结果
        assert_allclose(res2.x[4:], res.x)
        # 断言确保回归结果的权重值接近第一组测试的结果
        assert_allclose(res2.weights, res.weights)
        # 断言确保回归结果的分块情况相对于第一组测试的结果减去4
        assert_allclose(res2.blocks[1:] - 4, res.blocks[1:])
    # 定义测试函数，用于测试 isotonic_regression 函数对非单调序列的回归效果
    def test_against_R_monotone(self):
        # 设置输入序列 y
        y = [0, 6, 8, 3, 5, 2, 1, 7, 9, 4]
        # 调用 isotonic_regression 函数得到结果
        res = isotonic_regression(y)
        # 对比 R 语言中的期望输出结果 x_R
        x_R = [
            0, 4.1666667, 4.1666667, 4.1666667, 4.1666667, 4.1666667,
            4.1666667, 6.6666667, 6.6666667, 6.6666667,
        ]
        # 断言返回的 x 值与预期的 x_R 值接近
        assert_allclose(res.x, x_R)
        # 断言返回的 blocks（分段位置）与预期的 [0, 1, 7, 10] 相等

        assert_equal(res.blocks, [0, 1, 7, 10])

        # 设置另一个输入序列 y
        n = 100
        y = np.linspace(0, 1, num=n, endpoint=False)
        y = 5 * y + np.sin(10 * y)
        # 调用 isotonic_regression 函数得到结果
        res = isotonic_regression(y)
        # 对比 R 语言中的期望输出结果 x_R
        x_R = [
            0.00000000, 0.14983342, 0.29866933, 0.44552021, 0.58941834, 0.72942554,
            0.86464247, 0.99421769, 1.11735609, 1.23332691, 1.34147098, 1.44120736,
            1.53203909, 1.57081100, 1.57081100, 1.57081100, 1.57081100, 1.57081100,
            1.57081100, 1.57081100, 1.57081100, 1.57081100, 1.57081100, 1.57081100,
            1.57081100, 1.57081100, 1.57081100, 1.57081100, 1.57081100, 1.57081100,
            1.57081100, 1.57081100, 1.57081100, 1.57081100, 1.57081100, 1.57081100,
            1.57081100, 1.57081100, 1.57081100, 1.57081100, 1.57081100, 1.57081100,
            1.57081100, 1.57081100, 1.57081100, 1.57081100, 1.57081100, 1.57081100,
            1.57081100, 1.57081100, 1.57081100, 1.62418532, 1.71654534, 1.81773256,
            1.92723551, 2.04445967, 2.16873336, 2.29931446, 2.43539782, 2.57612334,
            2.72058450, 2.86783750, 3.01691060, 3.16681390, 3.31654920, 3.46511999,
            3.61154136, 3.75484992, 3.89411335, 4.02843976, 4.15698660, 4.27896904,
            4.39366786, 4.50043662, 4.59870810, 4.68799998, 4.76791967, 4.83816823,
            4.86564130, 4.86564130, 4.86564130, 4.86564130, 4.86564130, 4.86564130,
            4.86564130, 4.86564130, 4.86564130, 4.86564130, 4.86564130, 4.86564130,
            4.86564130, 4.86564130, 4.86564130, 4.86564130, 4.86564130, 4.86564130,
            4.86564130, 4.86564130, 4.86564130, 4.86564130,
        ]
        # 断言返回的 x 值与预期的 x_R 值接近
        assert_allclose(res.x, x_R)

        # 测试返回的 x 序列是否严格递增
        assert np.all(np.diff(res.x) >= 0)

        # 测试平衡属性：输入序列 y 和输出序列 x 的和是否接近
        assert_allclose(np.sum(res.x), np.sum(y))

        # 对逆序列进行测试
        res_inv = isotonic_regression(-y, increasing=False)
        # 断言逆序列的 x 值与原序列的 x 值相反
        assert_allclose(-res_inv.x, res.x)
        # 断言逆序列的 blocks（分段位置）与原序列相同
        assert_equal(res_inv.blocks, res.blocks)
    # 定义一个测试方法，用于测试非连续数组的情况
    def test_non_contiguous_arrays(self):
        # 创建一个从0到9的浮点数数组，并选取步长为3的元素
        x = np.arange(10, dtype=float)[::3]
        # 创建一个长度为10的浮点数数组，所有元素初始化为1，并选取步长为3的元素
        w = np.ones(10, dtype=float)[::3]
        
        # 断言数组x不是C连续存储的
        assert not x.flags.c_contiguous
        # 断言数组x不是Fortran连续存储的
        assert not x.flags.f_contiguous
        # 断言数组w不是C连续存储的
        assert not w.flags.c_contiguous
        # 断言数组w不是Fortran连续存储的
        assert not w.flags.f_contiguous

        # 调用isotonic_regression函数，对数组x进行保序回归，传入权重数组w
        res = isotonic_regression(x, weights=w)
        
        # 断言结果res中的x数组所有元素都是有限数
        assert np.all(np.isfinite(res.x))
        # 断言结果res中的weights数组所有元素都是有限数
        assert np.all(np.isfinite(res.weights))
        # 断言结果res中的blocks数组所有元素都是有限数
        assert np.all(np.isfinite(res.blocks))
```