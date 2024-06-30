# `D:\src\scipysrc\scipy\scipy\special\tests\test_spfun_stats.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
from numpy.testing import (assert_array_equal,
        assert_array_almost_equal_nulp, assert_almost_equal)  # 导入 NumPy 测试工具函数
from pytest import raises as assert_raises  # 导入 pytest 的 raises 断言函数

from scipy.special import gammaln, multigammaln  # 导入 SciPy 库中的 gamma 函数和多重 gamma 函数


class TestMultiGammaLn:

    def test1(self):
        # A test of the identity
        #     Gamma_1(a) = Gamma(a)
        np.random.seed(1234)  # 设置随机种子以确保可重复性
        a = np.abs(np.random.randn())  # 生成随机数并取绝对值，作为测试参数 a
        assert_array_equal(multigammaln(a, 1), gammaln(a))  # 断言多重 gamma 函数的计算结果与 gamma 函数的计算结果相等

    def test2(self):
        # A test of the identity
        #     Gamma_2(a) = sqrt(pi) * Gamma(a) * Gamma(a - 0.5)
        a = np.array([2.5, 10.0])  # 定义测试参数 a 为一个 NumPy 数组
        result = multigammaln(a, 2)  # 计算多重 gamma 函数的结果
        expected = np.log(np.sqrt(np.pi)) + gammaln(a) + gammaln(a - 0.5)  # 计算预期结果
        assert_almost_equal(result, expected)  # 断言多重 gamma 函数的计算结果与预期结果相近

    def test_bararg(self):
        assert_raises(ValueError, multigammaln, 0.5, 1.2)  # 断言调用多重 gamma 函数时会引发 ValueError 异常


def _check_multigammaln_array_result(a, d):
    # Test that the shape of the array returned by multigammaln
    # matches the input shape, and that all the values match
    # the value computed when multigammaln is called with a scalar.
    result = multigammaln(a, d)  # 调用多重 gamma 函数计算结果
    assert_array_equal(a.shape, result.shape)  # 断言结果数组的形状与输入数组的形状相同
    a1 = a.ravel()  # 将输入数组展平
    result1 = result.ravel()  # 将结果数组展平
    for i in range(a.size):
        assert_array_almost_equal_nulp(result1[i], multigammaln(a1[i], d))  # 断言展平后的每个元素与对应调用标量参数时的计算结果相近


def test_multigammaln_array_arg():
    # Check that the array returned by multigammaln has the correct
    # shape and contains the correct values.  The cases have arrays
    # with several different shapes.
    # The cases include a regression test for ticket #1849
    # (a = np.array([2.0]), an array with a single element).
    np.random.seed(1234)  # 设置随机种子以确保可重复性

    cases = [
        # a, d
        (np.abs(np.random.randn(3, 2)) + 5, 5),  # 第一种情况，a 是形状为 (3, 2) 的随机数数组，d 是标量 5
        (np.abs(np.random.randn(1, 2)) + 5, 5),  # 第二种情况，a 是形状为 (1, 2) 的随机数数组，d 是标量 5
        (np.arange(10.0, 18.0).reshape(2, 2, 2), 3),  # 第三种情况，a 是形状为 (2, 2, 2) 的 NumPy 数组，d 是标量 3
        (np.array([2.0]), 3),  # 第四种情况，a 是包含单个元素 2.0 的数组，d 是标量 3
        (np.float64(2.0), 3),  # 第五种情况，a 是标量 2.0，d 是标量 3
    ]

    for a, d in cases:
        _check_multigammaln_array_result(a, d)  # 调用 _check_multigammaln_array_result 函数进行检查
```