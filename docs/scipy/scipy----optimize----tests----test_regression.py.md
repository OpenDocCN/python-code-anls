# `D:\src\scipysrc\scipy\scipy\optimize\tests\test_regression.py`

```
"""Regression tests for optimize.

"""
import numpy as np  # 导入NumPy库，用于数值计算
from numpy.testing import assert_almost_equal  # 导入NumPy测试模块中的近似相等断言
from pytest import raises as assert_raises  # 导入pytest库中的raises别名为assert_raises，用于异常断言

import scipy.optimize  # 导入SciPy优化模块


class TestRegression:

    def test_newton_x0_is_0(self):
        # Regression test for gh-1601
        tgt = 1  # 目标值为1
        res = scipy.optimize.newton(lambda x: x - 1, 0)  # 使用牛顿法求解函数x-1=0在初始点0附近的根
        assert_almost_equal(res, tgt)  # 断言结果res近似等于目标值tgt

    def test_newton_integers(self):
        # Regression test for gh-1741
        root = scipy.optimize.newton(lambda x: x**2 - 1, x0=2,  # 使用牛顿法求解函数x^2-1=0的根，起始点为2
                                    fprime=lambda x: 2*x)  # 提供函数导数2x
        assert_almost_equal(root, 1.0)  # 断言根root近似等于1.0

    def test_lmdif_errmsg(self):
        # This shouldn't cause a crash on Python 3
        class SomeError(Exception):  # 定义自定义异常类SomeError
            pass

        counter = [0]  # 定义计数器列表，初始值为0

        def func(x):
            counter[0] += 1  # 每调用一次func函数，计数器加1
            if counter[0] < 3:  # 如果计数器小于3
                return x**2 - np.array([9, 10, 11])  # 返回x的平方与数组[9, 10, 11]的差组成的NumPy数组
            else:
                raise SomeError()  # 抛出自定义异常SomeError

        assert_raises(SomeError,  # 使用assert_raises断言捕获SomeError异常
                      scipy.optimize.leastsq,  # 调用SciPy的leastsq函数
                      func, [1, 2, 3])  # func作为函数参数，[1, 2, 3]作为初始值列表
```