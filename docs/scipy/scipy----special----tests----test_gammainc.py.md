# `D:\src\scipysrc\scipy\scipy\special\tests\test_gammainc.py`

```
import pytest  # 导入 pytest 库

import numpy as np  # 导入 NumPy 库，并使用 np 别名
from numpy.testing import assert_allclose, assert_array_equal  # 从 NumPy 测试模块中导入函数

import scipy.special as sc  # 导入 SciPy 的特殊函数模块，并使用 sc 别名
from scipy.special._testutils import FuncData  # 从 SciPy 特殊函数的测试工具模块中导入类

INVALID_POINTS = [  # 定义一个包含无效参数组合的列表
    (1, -1),
    (0, 0),
    (-1, 1),
    (np.nan, 1),
    (1, np.nan)
]

class TestGammainc:  # 定义测试类 TestGammainc

    @pytest.mark.parametrize('a, x', INVALID_POINTS)  # 使用 pytest 的参数化装饰器，指定无效参数对
    def test_domain(self, a, x):  # 定义测试方法 test_domain
        assert np.isnan(sc.gammainc(a, x))  # 断言调用 gammainc 函数后返回 NaN

    def test_a_eq_0_x_gt_0(self):  # 定义测试方法 test_a_eq_0_x_gt_0
        assert sc.gammainc(0, 1) == 1  # 断言调用 gammainc 函数返回 1，针对 a=0, x>0 的情况

    @pytest.mark.parametrize('a, x, desired', [  # 使用 pytest 的参数化装饰器，指定参数和期望值
        (np.inf, 1, 0),
        (np.inf, 0, 0),
        (np.inf, np.inf, np.nan),
        (1, np.inf, 1)
    ])
    def test_infinite_arguments(self, a, x, desired):  # 定义测试方法 test_infinite_arguments
        result = sc.gammainc(a, x)  # 调用 gammainc 函数计算结果
        if np.isnan(desired):  # 如果期望值为 NaN
            assert np.isnan(result)  # 断言结果也为 NaN
        else:
            assert result == desired  # 否则断言结果与期望值相等

    def test_infinite_limits(self):  # 定义测试方法 test_infinite_limits
        # 测试大参数趋近于无穷时，是否收敛到硬编码的极限值
        assert_allclose(
            sc.gammainc(1000, 100),  # 调用 gammainc 函数
            sc.gammainc(np.inf, 100),  # 调用 gammainc 函数，参数 a 为无穷
            atol=1e-200,  # 设置绝对误差容忍度，因为函数趋近于 0
            rtol=0
        )
        assert sc.gammainc(100, 1000) == sc.gammainc(100, np.inf)  # 断言两个函数调用结果相等

    def test_x_zero(self):  # 定义测试方法 test_x_zero
        a = np.arange(1, 10)  # 创建一个 NumPy 数组
        assert_array_equal(sc.gammainc(a, 0), 0)  # 断言调用 gammainc 函数结果与 0 的数组相等

    def test_limit_check(self):  # 定义测试方法 test_limit_check
        result = sc.gammainc(1e-10, 1)  # 调用 gammainc 函数计算结果
        limit = sc.gammainc(0, 1)  # 调用 gammainc 函数，a=0
        assert np.isclose(result, limit)  # 断言结果与极限值非常接近

    def gammainc_line(self, x):  # 定义函数 gammainc_line，计算特定曲线的伽马函数值
        # 在 a = x 的情况下，使用简化的渐近展开式
        c = np.array([-1/3, -1/540, 25/6048, 101/155520,
                      -3184811/3695155200, -2745493/8151736420])  # 定义系数数组
        res = 0  # 初始化结果变量
        xfac = 1  # 初始化 x 的阶乘变量
        for ck in c:  # 遍历系数数组
            res -= ck*xfac  # 更新结果
            xfac /= x  # 更新 x 的阶乘
        res /= np.sqrt(2*np.pi*x)  # 对结果进行调整
        res += 0.5  # 加上常数项
        return res  # 返回计算结果

    def test_line(self):  # 定义测试方法 test_line
        x = np.logspace(np.log10(25), 300, 500)  # 生成对数间隔的数组
        a = x  # 将 x 赋给 a
        dataset = np.vstack((a, x, self.gammainc_line(x))).T  # 垂直堆叠数组以创建数据集
        FuncData(sc.gammainc, dataset, (0, 1), 2, rtol=1e-11).check()  # 使用 FuncData 类进行检查

    def test_roundtrip(self):  # 定义测试方法 test_roundtrip
        a = np.logspace(-5, 10, 100)  # 生成对数间隔的数组
        x = np.logspace(-5, 10, 100)  # 生成对数间隔的数组

        y = sc.gammaincinv(a, sc.gammainc(a, x))  # 调用逆函数计算 y
        assert_allclose(x, y, rtol=1e-10)  # 断言 x 和 y 的值非常接近


class TestGammaincc:  # 定义测试类 TestGammaincc

    @pytest.mark.parametrize('a, x', INVALID_POINTS)  # 使用 pytest 的参数化装饰器，指定无效参数对
    def test_domain(self, a, x):  # 定义测试方法 test_domain
        assert np.isnan(sc.gammaincc(a, x))  # 断言调用 gammaincc 函数后返回 NaN

    def test_a_eq_0_x_gt_0(self):  # 定义测试方法 test_a_eq_0_x_gt_0
        assert sc.gammaincc(0, 1) == 0  # 断言调用 gammaincc 函数返回 0，针对 a=0, x>0 的情况

    @pytest.mark.parametrize('a, x, desired', [  # 使用 pytest 的参数化装饰器，指定参数和期望值
        (np.inf, 1, 1),
        (np.inf, 0, 1),
        (np.inf, np.inf, np.nan),
        (1, np.inf, 0)
    ])
    def test_infinite_arguments(self, a, x, desired):  # 定义测试方法 test_infinite_arguments
        result = sc.gammaincc(a, x)  # 调用 gammaincc 函数计算结果
        if np.isnan(desired):  # 如果期望值为 NaN
            assert np.isnan(result)  # 断言结果也为 NaN
        else:
            assert result == desired  # 否则断言结果与期望值相等
    # 测试大参数收敛到无穷远处的硬编码极限
    def test_infinite_limits(self):
        # 检查大参数时函数收敛到无穷远处的情况
        assert sc.gammaincc(1000, 100) == sc.gammaincc(np.inf, 100)
        # 检查函数在接近无穷时的收敛性，使用 `atol` 因为函数收敛到 0
        assert_allclose(
            sc.gammaincc(100, 1000),
            sc.gammaincc(100, np.inf),
            atol=1e-200,
            rtol=0
        )

    # 检查极限情况
    def test_limit_check(self):
        result = sc.gammaincc(1e-10,1)
        limit = sc.gammaincc(0,1)
        # 断言结果与已知极限值接近
        assert np.isclose(result, limit)

    # 检查 x 接近零时的情况
    def test_x_zero(self):
        a = np.arange(1, 10)
        # 断言对于 x 接近零时函数值等于 1
        assert_array_equal(sc.gammaincc(a, 0), 1)

    # 测试逆向计算
    def test_roundtrip(self):
        a = np.logspace(-5, 10, 100)
        x = np.logspace(-5, 10, 100)

        # 进行逆向计算，验证逆运算的精度
        y = sc.gammainccinv(a, sc.gammaincc(a, x))
        assert_allclose(x, y, rtol=1e-14)
```