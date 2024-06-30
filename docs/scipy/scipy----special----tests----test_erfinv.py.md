# `D:\src\scipysrc\scipy\scipy\special\tests\test_erfinv.py`

```
import numpy as np  # 导入NumPy库，通常用于数值计算
from numpy.testing import assert_allclose, assert_equal  # 导入NumPy测试模块中的断言函数
import pytest  # 导入pytest，用于编写和运行测试用例

import scipy.special as sc  # 导入SciPy库中的special模块，用于数学特殊函数的计算


class TestInverseErrorFunction:
    def test_compliment(self):
        # 测试 erfcinv(1 - x) == erfinv(x)
        x = np.linspace(-1, 1, 101)  # 在[-1, 1]范围内生成101个均匀分布的点作为输入
        assert_allclose(sc.erfcinv(1 - x), sc.erfinv(x), rtol=0, atol=1e-15)  # 断言erfcinv(1 - x)与erfinv(x)非常接近

    def test_literal_values(self):
        # 预期值是通过mpmath计算得到的:
        #
        #   import mpmath
        #   mpmath.mp.dps = 200
        #   for y in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        #       x = mpmath.erfinv(y)
        #       print(x)
        #
        y = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])  # 创建NumPy数组，包含0到0.9的十个数值
        actual = sc.erfinv(y)  # 计算erfinv(y)的实际值
        expected = [
            0.0,
            0.08885599049425769,
            0.1791434546212917,
            0.2724627147267543,
            0.37080715859355795,
            0.4769362762044699,
            0.5951160814499948,
            0.7328690779592167,
            0.9061938024368233,
            1.1630871536766743,
        ]
        assert_allclose(actual, expected, rtol=0, atol=1e-15)  # 断言实际值与预期值非常接近

    @pytest.mark.parametrize(
        'f, x, y',
        [
            (sc.erfinv, -1, -np.inf),
            (sc.erfinv, 0, 0),
            (sc.erfinv, 1, np.inf),
            (sc.erfinv, -100, np.nan),
            (sc.erfinv, 100, np.nan),
            (sc.erfcinv, 0, np.inf),
            (sc.erfcinv, 1, -0.0),
            (sc.erfcinv, 2, -np.inf),
            (sc.erfcinv, -100, np.nan),
            (sc.erfcinv, 100, np.nan),
        ],
        ids=[
            'erfinv下界',
            'erfinv中点',
            'erfinv上界',
            'erfinv低于下界',
            'erfinv高于上界',
            'erfcinv下界',
            'erfcinv中点',
            'erfcinv上界',
            'erfcinv低于下界',
            'erfcinv高于上界',
        ]
    )
    def test_domain_bounds(self, f, x, y):
        assert_equal(f(x), y)  # 断言f(x)的返回值等于预期值y
    def test_erfinv_asympt(self):
        # 对 erfinv(x) 在小 x 值处的精度回归测试
        # 预期值通过 mpmath 预先计算得出:
        # >>> mpmath.mp.dps = 100
        # >>> expected = [float(mpmath.erfinv(t)) for t in x]
        
        # 定义输入数组 x，包含多个小值
        x = np.array([1e-20, 1e-15, 1e-14, 1e-10, 1e-8, 0.9e-7, 1.1e-7, 1e-6])
        
        # 预期输出的数组，使用 mpmath 预先计算的值
        expected = np.array([8.86226925452758e-21,
                             8.862269254527581e-16,
                             8.86226925452758e-15,
                             8.862269254527581e-11,
                             8.86226925452758e-09,
                             7.97604232907484e-08,
                             9.74849617998037e-08,
                             8.8622692545299e-07])
        
        # 使用 assert_allclose 检查 erfinv(x) 的输出是否与预期值接近
        assert_allclose(sc.erfinv(x), expected,
                        rtol=1e-15)

        # 同时测试 erfinv(x) 与 erf(erfinv(x)) 的一致性
        assert_allclose(sc.erf(sc.erfinv(x)),
                        x,
                        rtol=5e-15)
```