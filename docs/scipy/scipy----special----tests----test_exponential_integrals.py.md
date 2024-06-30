# `D:\src\scipysrc\scipy\scipy\special\tests\test_exponential_integrals.py`

```
import pytest  # 导入 pytest 模块

import numpy as np  # 导入 numpy 库并重命名为 np
from numpy.testing import assert_allclose  # 导入 numpy.testing 模块中的 assert_allclose 函数
import scipy.special as sc  # 导入 scipy.special 库并重命名为 sc


class TestExp1:
    # 定义测试类 TestExp1

    def test_branch_cut(self):
        # 定义测试方法 test_branch_cut，在该方法中测试 branch cut 相关功能
        assert np.isnan(sc.exp1(-1))  # 断言 sc.exp1(-1) 是否为 NaN
        assert sc.exp1(complex(-1, 0)).imag == (
            -sc.exp1(complex(-1, -0.0)).imag
        )  # 断言复数形式下的 exp1(-1) 的虚部是否满足对称性质

        assert_allclose(
            sc.exp1(complex(-1, 0)),
            sc.exp1(-1 + 1e-20j),
            atol=0,
            rtol=1e-15
        )  # 使用 assert_allclose 函数检查复数形式下 exp1(-1) 与 -1 + 1e-20j 的近似程度
        assert_allclose(
            sc.exp1(complex(-1, -0.0)),
            sc.exp1(-1 - 1e-20j),
            atol=0,
            rtol=1e-15
        )  # 使用 assert_allclose 函数检查复数形式下 exp1(-1) 与 -1 - 1e-20j 的近似程度

    def test_834(self):
        # 定义测试方法 test_834，用于回归测试 #834
        a = sc.exp1(-complex(19.9999990))  # 计算 -19.9999990 的指数积分
        b = sc.exp1(-complex(19.9999991))  # 计算 -19.9999991 的指数积分
        assert_allclose(a.imag, b.imag, atol=0, rtol=1e-15)  # 使用 assert_allclose 函数检查两个结果的虚部近似性


class TestScaledExp1:
    # 定义测试类 TestScaledExp1

    @pytest.mark.parametrize('x, expected', [(0, 0), (np.inf, 1)])
    def test_limits(self, x, expected):
        # 定义测试方法 test_limits，使用 pytest.mark.parametrize 对输入进行参数化测试
        y = sc._ufuncs._scaled_exp1(x)  # 调用 _scaled_exp1 方法计算
        assert y == expected  # 断言计算结果是否等于期望值

    # 下面的注释提供了期望值的计算方法，使用 mpmath 库进行精确计算，然后列出了一些特定输入值的期望输出
    @pytest.mark.parametrize('x, expected',
                             [(1e-25, 5.698741165994961e-24),
                              (0.1, 0.20146425447084518),
                              (0.9995, 0.5962509885831002),
                              (1.0, 0.5963473623231941),
                              (1.0005, 0.5964436833238044),
                              (2.5, 0.7588145912149602),
                              (10.0, 0.9156333393978808),
                              (100.0, 0.9901942286733019),
                              (500.0, 0.9980079523802055),
                              (1000.0, 0.9990019940238807),
                              (1249.5, 0.9992009578306811),
                              (1250.0, 0.9992012769377913),
                              (1250.25, 0.9992014363957858),
                              (2000.0, 0.9995004992514963),
                              (1e4, 0.9999000199940024),
                              (1e10, 0.9999999999),
                              (1e15, 0.999999999999999),
                              ])
    def test_scaled_exp1(self, x, expected):
        # 定义参数化测试方法 test_scaled_exp1，测试 _scaled_exp1 方法的精确性
        y = sc._ufuncs._scaled_exp1(x)  # 调用 _scaled_exp1 方法计算
        assert_allclose(y, expected, rtol=2e-15)  # 使用 assert_allclose 函数检查计算结果与期望值的近似度


class TestExpi:
    # 定义测试类 TestExpi

    @pytest.mark.parametrize('result', [
        sc.expi(complex(-1, 0)),
        sc.expi(complex(-1, -0.0)),
        sc.expi(-1)
    ])
    def test_branch_cut(self, result):
        # 定义测试方法 test_branch_cut，测试 expi 方法在 branch cut 上的表现
        desired = -0.21938393439552027368  # 使用 Mpmath 计算得到的期望值
        assert_allclose(result, desired, atol=0, rtol=1e-14)  # 使用 assert_allclose 函数检查计算结果与期望值的近似度
    # 定义一个测试方法，用于检验在接近分支切割时的行为
    def test_near_branch_cut(self):
        # 从复平面上方接近 -1 的指数积分值
        lim_from_above = sc.expi(-1 + 1e-20j)
        # 从复平面下方接近 -1 的指数积分值
        lim_from_below = sc.expi(-1 - 1e-20j)
        # 断言实部在两个接近值之间的近似相等性
        assert_allclose(
            lim_from_above.real,
            lim_from_below.real,
            atol=0,
            rtol=1e-15
        )
        # 断言虚部在两个接近值之间的近似相反性
        assert_allclose(
            lim_from_above.imag,
            -lim_from_below.imag,
            atol=0,
            rtol=1e-15
        )

    # 定义一个测试方法，用于检验在正实轴上的连续性
    def test_continuity_on_positive_real_axis(self):
        # 断言复数 1+0j 和 1-0j 对应的指数积分值在数值上的近似相等性
        assert_allclose(
            sc.expi(complex(1, 0)),
            sc.expi(complex(1, -0.0)),
            atol=0,
            rtol=1e-15
        )
# 定义一个测试类 TestExpn
class TestExpn:

    # 定义测试方法 test_out_of_domain，用于测试 sc.expn 函数在非法参数输入时的行为
    def test_out_of_domain(self):
        # 使用 assert 语句检查以下条件是否为真：
        # 列表中所有元素都应为 NaN（Not a Number），即无效的数值
        assert all(np.isnan([sc.expn(-1, 1.0), sc.expn(1, -1.0)]))
```