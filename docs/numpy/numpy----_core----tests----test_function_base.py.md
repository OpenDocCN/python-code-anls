# `.\numpy\numpy\_core\tests\test_function_base.py`

```py
import sys  # 导入sys模块，用于系统相关的操作

import pytest  # 导入pytest模块，用于编写和运行测试用例

import numpy as np  # 导入NumPy库，用于科学计算
from numpy import (
    logspace, linspace, geomspace, dtype, array, arange, isnan,  # 导入NumPy的各种函数和类
    ndarray, sqrt, nextafter, stack, errstate
    )
from numpy._core import sctypes  # 导入NumPy内部的_core模块中的sctypes函数
from numpy._core.function_base import add_newdoc  # 导入NumPy内部的function_base模块中的add_newdoc函数
from numpy.testing import (  # 导入NumPy的测试模块中的各种断言函数
    assert_, assert_equal, assert_raises, assert_array_equal, assert_allclose,
    IS_PYPY
    )


class PhysicalQuantity(float):
    def __new__(cls, value):  # 物理量类，继承自float，用于带单位的数值计算
        return float.__new__(cls, value)

    def __add__(self, x):  # 重载加法运算符，确保操作数是PhysicalQuantity类型
        assert_(isinstance(x, PhysicalQuantity))
        return PhysicalQuantity(float(x) + float(self))

    __radd__ = __add__  # 右加法的实现与左加法相同

    def __sub__(self, x):  # 重载减法运算符，确保操作数是PhysicalQuantity类型
        assert_(isinstance(x, PhysicalQuantity))
        return PhysicalQuantity(float(self) - float(x))

    def __rsub__(self, x):  # 右减法的实现与左减法相反
        assert_(isinstance(x, PhysicalQuantity))
        return PhysicalQuantity(float(x) - float(self))

    def __mul__(self, x):  # 重载乘法运算符，确保操作数是PhysicalQuantity类型
        return PhysicalQuantity(float(x) * float(self))

    __rmul__ = __mul__  # 右乘法的实现与左乘法相同

    def __div__(self, x):  # 重载除法运算符，确保操作数是PhysicalQuantity类型
        return PhysicalQuantity(float(self) / float(x))

    def __rdiv__(self, x):  # 右除法的实现与左除法相反
        return PhysicalQuantity(float(x) / float(self))


class PhysicalQuantity2(ndarray):  # 物理量2类，继承自NumPy的ndarray类
    __array_priority__ = 10  # 设定数组优先级为10，用于运算符重载的优先级设置


class TestLogspace:

    def test_basic(self):  # 测试logspace基本功能
        y = logspace(0, 6)  # 生成指数等间隔的数组
        assert_(len(y) == 50)  # 断言生成数组长度为50
        y = logspace(0, 6, num=100)  # 指定生成100个点的指数等间隔数组
        assert_(y[-1] == 10 ** 6)  # 断言最后一个元素为10^6
        y = logspace(0, 6, endpoint=False)  # 不包含结束点的指数等间隔数组
        assert_(y[-1] < 10 ** 6)  # 断言最后一个元素小于10^6
        y = logspace(0, 6, num=7)  # 指定生成7个点的指数等间隔数组
        assert_array_equal(y, [1, 10, 100, 1e3, 1e4, 1e5, 1e6])  # 断言生成数组与预期数组相等

    def test_start_stop_array(self):  # 测试接受数组作为起始和结束点的logspace功能
        start = array([0., 1.])  # 起始点数组
        stop = array([6., 7.])  # 结束点数组
        t1 = logspace(start, stop, 6)  # 生成指数等间隔数组，起始和结束点分别为数组元素
        t2 = stack([logspace(_start, _stop, 6)
                    for _start, _stop in zip(start, stop)], axis=1)  # 使用zip函数生成多维数组
        assert_equal(t1, t2)  # 断言两个数组相等
        t3 = logspace(start, stop[0], 6)  # 使用数组和标量生成指数等间隔数组
        t4 = stack([logspace(_start, stop[0], 6)
                    for _start in start], axis=1)  # 使用列表推导式生成多维数组
        assert_equal(t3, t4)  # 断言两个数组相等
        t5 = logspace(start, stop, 6, axis=-1)  # 指定轴向生成指数等间隔数组
        assert_equal(t5, t2.T)  # 断言生成数组与转置后的多维数组相等

    @pytest.mark.parametrize("axis", [0, 1, -1])
    def test_base_array(self, axis: int):  # 测试接受数组作为基数的logspace功能
        start = 1  # 起始点
        stop = 2  # 结束点
        num = 6  # 生成点数
        base = array([1, 2])  # 基数数组
        t1 = logspace(start, stop, num=num, base=base, axis=axis)  # 生成指定基数的指数等间隔数组
        t2 = stack(
            [logspace(start, stop, num=num, base=_base) for _base in base],  # 使用列表推导式生成多维数组
            axis=(axis + 1) % t1.ndim,  # 指定轴向堆叠数组
        )
        assert_equal(t1, t2)  # 断言两个数组相等

    @pytest.mark.parametrize("axis", [0, 1, -1])
    # 测试以数组中的元素作为 stop 参数时的 logspace 函数
    def test_stop_base_array(self, axis: int):
        # 设定起始值
        start = 1
        # 用数组指定 stop 值
        stop = array([2, 3])
        # 指定数量
        num = 6
        # 用数组指定 base 值
        base = array([1, 2])
        # 调用 logspace 函数，生成第一个数组 t1
        t1 = logspace(start, stop, num=num, base=base, axis=axis)
        # 通过列表推导式生成第二个数组 t2，其中用到了不同的 stop 和 base
        t2 = stack(
            [logspace(start, _stop, num=num, base=_base)
             for _stop, _base in zip(stop, base)],
            axis=(axis + 1) % t1.ndim,
        )
        # 断言 t1 和 t2 应该相等
        assert_equal(t1, t2)

    # 测试不同 dtype 参数对 logspace 函数的影响
    def test_dtype(self):
        # 测试 dtype='float32'
        y = logspace(0, 6, dtype='float32')
        assert_equal(y.dtype, dtype('float32'))
        # 测试 dtype='float64'
        y = logspace(0, 6, dtype='float64')
        assert_equal(y.dtype, dtype('float64'))
        # 测试 dtype='int32'
        y = logspace(0, 6, dtype='int32')
        assert_equal(y.dtype, dtype('int32'))

    # 测试物理量对象 PhysicalQuantity 作为参数时的 logspace 函数
    def test_physical_quantities(self):
        # 创建两个物理量对象
        a = PhysicalQuantity(1.0)
        b = PhysicalQuantity(5.0)
        # 断言 logspace(a, b) 应该等同于 logspace(1.0, 5.0)
        assert_equal(logspace(a, b), logspace(1.0, 5.0))

    # 测试视图为 PhysicalQuantity2 类型的数组作为参数时的 logspace 函数
    def test_subclass(self):
        # 创建两个视图为 PhysicalQuantity2 的数组
        a = array(1).view(PhysicalQuantity2)
        b = array(7).view(PhysicalQuantity2)
        # 调用 logspace 函数，生成第一个数组 ls
        ls = logspace(a, b)
        # 断言 ls 应该是 PhysicalQuantity2 类型的对象
        assert type(ls) is PhysicalQuantity2
        # 断言 ls 应该等同于 logspace(1.0, 7.0)
        assert_equal(ls, logspace(1.0, 7.0))
        # 再次调用 logspace 函数，用 1 作为参数，生成第二个数组 ls
        ls = logspace(a, b, 1)
        # 断言 ls 应该是 PhysicalQuantity2 类型的对象
        assert type(ls) is PhysicalQuantity2
        # 断言 ls 应该等同于 logspace(1.0, 7.0, 1)
        assert_equal(ls, logspace(1.0, 7.0, 1))
# 定义一个测试类 TestGeomspace，用于测试 geomspace 函数的各种情况
class TestGeomspace:

    # 测试 geomspace 函数的基本用法
    def test_basic(self):
        # 默认参数情况下生成从 1 到 1e6 的等比数列，长度为 50
        y = geomspace(1, 1e6)
        assert_(len(y) == 50)

        # 指定生成长度为 100 的等比数列，确保最后一个元素等于 1e6
        y = geomspace(1, 1e6, num=100)
        assert_(y[-1] == 10 ** 6)

        # 不包括终点的等比数列，确保最后一个元素小于 1e6
        y = geomspace(1, 1e6, endpoint=False)
        assert_(y[-1] < 10 ** 6)

        # 指定生成长度为 7 的等比数列，验证生成的数列与预期的一致
        y = geomspace(1, 1e6, num=7)
        assert_array_equal(y, [1, 10, 100, 1e3, 1e4, 1e5, 1e6])

        # 测试非常规情况下的等比数列生成
        y = geomspace(8, 2, num=3)
        assert_allclose(y, [8, 4, 2])  # 确保数列的近似相等性
        assert_array_equal(y.imag, 0)  # 确保数列的虚部全部为零

        # 负数范围内的等比数列生成，验证生成的数列与预期的一致
        y = geomspace(-1, -100, num=3)
        assert_array_equal(y, [-1, -10, -100])
        assert_array_equal(y.imag, 0)

        y = geomspace(-100, -1, num=3)
        assert_array_equal(y, [-100, -10, -1])
        assert_array_equal(y.imag, 0)

    # 测试边界条件，确保生成的数列的起始点和终点与指定的起始点和终点完全一致
    def test_boundaries_match_start_and_stop_exactly(self):
        start = 0.3
        stop = 20.3

        # 测试只生成一个数的情况，确保生成的数等于指定的起始点
        y = geomspace(start, stop, num=1)
        assert_equal(y[0], start)

        # 测试不包括终点的情况下，生成只有一个数，确保生成的数等于指定的起始点
        y = geomspace(start, stop, num=1, endpoint=False)
        assert_equal(y[0], start)

        # 测试生成长度为 3 的数列，确保第一个数等于起始点，最后一个数等于终点
        y = geomspace(start, stop, num=3)
        assert_equal(y[0], start)
        assert_equal(y[-1], stop)

        # 测试不包括终点的情况下，生成长度为 3 的数列，确保第一个数等于起始点
        y = geomspace(start, stop, num=3, endpoint=False)
        assert_equal(y[0], start)

    # 测试在数列内部含有 NaN 值的情况
    def test_nan_interior(self):
        # 忽略无效值错误，生成长度为 4 的数列，确保第一个数和最后一个数正确，中间的数全部为 NaN
        with errstate(invalid='ignore'):
            y = geomspace(-3, 3, num=4)

        assert_equal(y[0], -3.0)
        assert_(isnan(y[1:-1]).all())  # 确保中间的数全部为 NaN
        assert_equal(y[3], 3.0)

        # 同样的情况下，不包括终点的数列，确保生成的数列的第一个数正确，中间的数全部为 NaN
        with errstate(invalid='ignore'):
            y = geomspace(-3, 3, num=4, endpoint=False)

        assert_equal(y[0], -3.0)
        assert_(isnan(y[1:]).all())  # 确保中间的数全部为 NaN
    def test_complex(self):
        # 测试复数情况

        # 纯虚数情况
        y = geomspace(1j, 16j, num=5)
        assert_allclose(y, [1j, 2j, 4j, 8j, 16j])
        assert_array_equal(y.real, 0)

        y = geomspace(-4j, -324j, num=5)
        assert_allclose(y, [-4j, -12j, -36j, -108j, -324j])
        assert_array_equal(y.real, 0)

        y = geomspace(1+1j, 1000+1000j, num=4)
        assert_allclose(y, [1+1j, 10+10j, 100+100j, 1000+1000j])

        y = geomspace(-1+1j, -1000+1000j, num=4)
        assert_allclose(y, [-1+1j, -10+10j, -100+100j, -1000+1000j])

        # 对数螺线情况
        y = geomspace(-1, 1, num=3, dtype=complex)
        assert_allclose(y, [-1, 1j, +1])

        y = geomspace(0+3j, -3+0j, 3)
        assert_allclose(y, [0+3j, -3/sqrt(2)+3j/sqrt(2), -3+0j])
        y = geomspace(0+3j, 3+0j, 3)
        assert_allclose(y, [0+3j, 3/sqrt(2)+3j/sqrt(2), 3+0j])
        y = geomspace(-3+0j, 0-3j, 3)
        assert_allclose(y, [-3+0j, -3/sqrt(2)-3j/sqrt(2), 0-3j])
        y = geomspace(0+3j, -3+0j, 3)
        assert_allclose(y, [0+3j, -3/sqrt(2)+3j/sqrt(2), -3+0j])
        y = geomspace(-2-3j, 5+7j, 7)
        assert_allclose(y, [-2-3j, -0.29058977-4.15771027j,
                            2.08885354-4.34146838j, 4.58345529-3.16355218j,
                            6.41401745-0.55233457j, 6.75707386+3.11795092j,
                            5+7j])

        # 类型转换应防止 -5 变成 NaN
        y = geomspace(3j, -5, 2)
        assert_allclose(y, [3j, -5])
        y = geomspace(-5, 3j, 2)
        assert_allclose(y, [-5, 3j])

    def test_complex_shortest_path(self):
        # 测试最短对数螺线路径，参见 gh-25644
        x = 1.2 + 3.4j
        y = np.exp(1j*(np.pi-.1)) * x
        z = np.geomspace(x, y, 5)
        expected = np.array([1.2 + 3.4j, -1.47384 + 3.2905616j,
                        -3.33577588 + 1.36842949j, -3.36011056 - 1.30753855j,
                        -1.53343861 - 3.26321406j])
        np.testing.assert_array_almost_equal(z, expected)


    def test_dtype(self):
        # 测试不同数据类型

        y = geomspace(1, 1e6, dtype='float32')
        assert_equal(y.dtype, dtype('float32'))
        y = geomspace(1, 1e6, dtype='float64')
        assert_equal(y.dtype, dtype('float64'))
        y = geomspace(1, 1e6, dtype='int32')
        assert_equal(y.dtype, dtype('int32'))

        # 原生数据类型
        y = geomspace(1, 1e6, dtype=float)
        assert_equal(y.dtype, dtype('float64'))
        y = geomspace(1, 1e6, dtype=complex)
        assert_equal(y.dtype, dtype('complex128'))
    def test_start_stop_array_scalar(self):
        # 定义整数类型的数组边界
        lim1 = array([120, 100], dtype="int8")
        lim2 = array([-120, -100], dtype="int8")
        # 定义无符号短整型数组边界
        lim3 = array([1200, 1000], dtype="uint16")
        # 创建等比数列，使用 lim1 数组中的边界值
        t1 = geomspace(lim1[0], lim1[1], 5)
        # 创建等比数列，使用 lim2 数组中的边界值
        t2 = geomspace(lim2[0], lim2[1], 5)
        # 创建等比数列，使用 lim3 数组中的边界值
        t3 = geomspace(lim3[0], lim3[1], 5)
        # 创建等比数列，使用浮点数边界值
        t4 = geomspace(120.0, 100.0, 5)
        # 创建等比数列，使用负浮点数边界值
        t5 = geomspace(-120.0, -100.0, 5)
        # 创建等比数列，使用浮点数边界值
        t6 = geomspace(1200.0, 1000.0, 5)

        # 断言比较 t1 和 t4，相对误差容差为 1e-2
        assert_allclose(t1, t4, rtol=1e-2)
        # 断言比较 t2 和 t5，相对误差容差为 1e-2
        assert_allclose(t2, t5, rtol=1e-2)
        # 断言比较 t3 和 t6，相对误差容差为 1e-5
        assert_allclose(t3, t6, rtol=1e-5)

    def test_start_stop_array(self):
        # 定义包含特殊情况的起始数组
        start = array([1.e0, 32., 1j, -4j, 1+1j, -1])
        # 定义包含特殊情况的结束数组
        stop = array([1.e4, 2., 16j, -324j, 10000+10000j, 1])
        # 创建多个等比数列，每对起始和结束值生成一个列，组合成 t1
        t1 = geomspace(start, stop, 5)
        # 使用 zip 将 start 和 stop 中的每对元素作为参数传递给 geomspace，创建 t2
        t2 = stack([geomspace(_start, _stop, 5)
                    for _start, _stop in zip(start, stop)], axis=1)
        # 断言 t1 等于 t2
        assert_equal(t1, t2)
        # 创建多个等比数列，每个起始值与同一个结束值生成一个列，组合成 t3
        t3 = geomspace(start, stop[0], 5)
        # 使用 zip 将 start 中的每个元素与 stop[0] 作为参数传递给 geomspace，创建 t4
        t4 = stack([geomspace(_start, stop[0], 5)
                    for _start in start], axis=1)
        # 断言 t3 等于 t4
        assert_equal(t3, t4)
        # 创建多个等比数列，每对起始和结束值生成一个列，并沿着轴反转，组合成 t5
        t5 = geomspace(start, stop, 5, axis=-1)
        # 断言 t5 等于 t2 的转置
        assert_equal(t5, t2.T)

    def test_physical_quantities(self):
        # 创建物理量对象 a 和 b
        a = PhysicalQuantity(1.0)
        b = PhysicalQuantity(5.0)
        # 断言 geomspace(a, b) 等于 geomspace(1.0, 5.0)
        assert_equal(geomspace(a, b), geomspace(1.0, 5.0))

    def test_subclass(self):
        # 创建 PhysicalQuantity2 类型的数组 a 和 b
        a = array(1).view(PhysicalQuantity2)
        b = array(7).view(PhysicalQuantity2)
        # 使用 geomspace(a, b) 创建等比数列 gs
        gs = geomspace(a, b)
        # 断言 gs 的类型是 PhysicalQuantity2
        assert type(gs) is PhysicalQuantity2
        # 断言 gs 等于 geomspace(1.0, 7.0)
        assert_equal(gs, geomspace(1.0, 7.0))
        # 使用 geomspace(a, b, 1) 创建等比数列 gs
        gs = geomspace(a, b, 1)
        # 断言 gs 的类型是 PhysicalQuantity2
        assert type(gs) is PhysicalQuantity2
        # 断言 gs 等于 geomspace(1.0, 7.0, 1)

    def test_bounds(self):
        # 断言调用 geomspace(0, 10) 会引发 ValueError 异常
        assert_raises(ValueError, geomspace, 0, 10)
        # 断言调用 geomspace(10, 0) 会引发 ValueError 异常
        assert_raises(ValueError, geomspace, 10, 0)
        # 断言调用 geomspace(0, 0) 会引发 ValueError 异常
        assert_raises(ValueError, geomspace, 0, 0)
    # 定义一个名为 TestLinspace 的测试类
class TestLinspace:

    # 定义测试基本功能的方法
    def test_basic(self):
        # 调用 linspace 函数生成等间隔的数列 y，长度为 50
        y = linspace(0, 10)
        # 断言 y 的长度为 50
        assert_(len(y) == 50)
        # 再次调用 linspace 函数生成等间隔的数列 y，包括终点值 10，数量为 100
        y = linspace(2, 10, num=100)
        # 断言 y 的最后一个值为 10
        assert_(y[-1] == 10)
        # 再次调用 linspace 函数生成等间隔的数列 y，不包括终点值 10
        y = linspace(2, 10, endpoint=False)
        # 断言 y 的最后一个值小于 10
        assert_(y[-1] < 10)
        # 使用 assert_raises 断言 ValueError 是否会被引发
        assert_raises(ValueError, linspace, 0, 10, num=-1)

    # 定义测试边界情况的方法
    def test_corner(self):
        # 调用 linspace 函数生成等间隔的数列 y，包括起点和终点值，数量为 1
        y = list(linspace(0, 1, 1))
        # 断言 y 应为 [0.0]
        assert_(y == [0.0], y)
        # 使用 assert_raises 断言 TypeError 是否会被引发
        assert_raises(TypeError, linspace, 0, 1, num=2.5)

    # 定义测试数据类型的方法
    def test_type(self):
        # 调用 linspace 函数生成等间隔的数列，数量为 0，获取其数据类型
        t1 = linspace(0, 1, 0).dtype
        # 调用 linspace 函数生成等间隔的数列，数量为 1，获取其数据类型
        t2 = linspace(0, 1, 1).dtype
        # 调用 linspace 函数生成等间隔的数列，数量为 2，获取其数据类型
        t3 = linspace(0, 1, 2).dtype
        # 断言 t1 与 t2 相等
        assert_equal(t1, t2)
        # 断言 t2 与 t3 相等
        assert_equal(t2, t3)

    # 定义测试数据类型参数的方法
    def test_dtype(self):
        # 调用 linspace 函数生成等间隔的浮点数数列，数据类型为 'float32'
        y = linspace(0, 6, dtype='float32')
        # 断言 y 的数据类型为 'float32'
        assert_equal(y.dtype, dtype('float32'))
        # 调用 linspace 函数生成等间隔的浮点数数列，数据类型为 'float64'
        y = linspace(0, 6, dtype='float64')
        # 断言 y 的数据类型为 'float64'
        assert_equal(y.dtype, dtype('float64'))
        # 调用 linspace 函数生成等间隔的整数数列，数据类型为 'int32'
        y = linspace(0, 6, dtype='int32')
        # 断言 y 的数据类型为 'int32'
        assert_equal(y.dtype, dtype('int32'))

    # 定义测试使用数组、标量作为起始和结束值的方法
    def test_start_stop_array_scalar(self):
        # 创建包含两个元素的 int8 类型数组 lim1 和 lim2
        lim1 = array([-120, 100], dtype="int8")
        lim2 = array([120, -100], dtype="int8")
        # 创建包含两个元素的 uint16 类型数组 lim3
        lim3 = array([1200, 1000], dtype="uint16")
        # 调用 linspace 函数生成 lim1[0] 到 lim1[1] 之间的等间隔数列，数量为 5
        t1 = linspace(lim1[0], lim1[1], 5)
        # 调用 linspace 函数生成 lim2[0] 到 lim2[1] 之间的等间隔数列，数量为 5
        t2 = linspace(lim2[0], lim2[1], 5)
        # 调用 linspace 函数生成 lim3[0] 到 lim3[1] 之间的等间隔数列，数量为 5
        t3 = linspace(lim3[0], lim3[1], 5)
        # 调用 linspace 函数生成 -120.0 到 100.0 之间的等间隔数列，数量为 5
        t4 = linspace(-120.0, 100.0, 5)
        # 调用 linspace 函数生成 120.0 到 -100.0 之间的等间隔数列，数量为 5
        t5 = linspace(120.0, -100.0, 5)
        # 调用 linspace 函数生成 1200.0 到 1000.0 之间的等间隔数列，数量为 5
        t6 = linspace(1200.0, 1000.0, 5)
        # 断言 t1 与 t4 相等
        assert_equal(t1, t4)
        # 断言 t2 与 t5 相等
        assert_equal(t2, t5)
        # 断言 t3 与 t6 相等
        assert_equal(t3, t6)

    # 定义测试使用数组作为起始和结束值的方法
    def test_start_stop_array(self):
        # 创建包含两个元素的 int8 类型数组 start 和 stop
        start = array([-120, 120], dtype="int8")
        stop = array([100, -100], dtype="int8")
        # 调用 linspace 函数分别以 start 和 stop 中的每个元素作为起始和结束值生成等间隔数列，数量为 5
        t1 = linspace(start, stop, 5)
        # 使用 stack 函数将生成的数列按列堆叠成二维数组 t2
        t2 = stack([linspace(_start, _stop, 5)
                    for _start, _stop in zip(start, stop)], axis=1)
        # 断言 t1 与 t2 相等
        assert_equal(t1, t2)
        # 调用 linspace 函数以 start 中的每个元素作为起始值，stop[0] 作为结束值生成等间隔数列，数量为 5
        t3 = linspace(start, stop[0], 5)
        # 使用 stack 函数将生成的数列按列堆叠成二维数组 t4
        t4 = stack([linspace(_start, stop[0], 5)
                    for _start in start], axis=1)
        # 断言 t3 与 t4 相等
        assert_equal(t3, t4)
        # 调用 linspace 函数以 start 和 stop 中的每个元素作为起始和结束值生成等间隔数列，数量为 5，按行堆叠成二维数组 t5
        t5 = linspace(start, stop, 5, axis=-1)
        # 断言 t5 与 t2 的转置相等
        assert_equal(t5, t2.T)

    # 定义测试生成复数数列的方法
    def test_complex(self):
        # 调用 linspace 函数生成复数数列 lim1，起始值为 1+2j，结束值为 3+4j，数量为 5
        lim1 = linspace(1 + 2j, 3 + 4j, 5)
        # 创建期望的复数数列 t1
        t1 = array([1.0+2.j, 1.5+2.5j,  2.0+3j, 2.5+3.5j, 3.0+4j])
        # 调用 linspace 函数生成复数数列 lim2，起始值为 1j，结束值为 10，数量为 5
        lim2 = linspace(1j, 10, 5)
        # 创建期望的复数数列 t2
        t2 = array([0.0+1.j, 2.5+0.75j, 5.0+0.5j, 7
    def test_array_interface(self):
        # 回归测试，用于检查 https://github.com/numpy/numpy/pull/6659
        # 确保 start/stop 可以是实现了 __array_interface__ 并且可以转换为数值标量的对象

        class Arrayish:
            """
            支持 __array_interface__ 的通用对象，因此理论上可以转换为数值标量，
            但不被认为是数值型，同时也支持浮点数乘法。

            数据应该是一个实现了缓冲区接口的对象，至少包含 4 个字节。
            """

            def __init__(self, data):
                self._data = data

            @property
            def __array_interface__(self):
                return {'shape': (), 'typestr': '<i4', 'data': self._data,
                        'version': 3}

            def __mul__(self, other):
                # 对于这个测试，任何乘法都是一个恒等操作 :)
                return self

        one = Arrayish(array(1, dtype='<i4'))
        five = Arrayish(array(5, dtype='<i4'))

        assert_equal(linspace(one, five), linspace(1, 5))

    def test_denormal_numbers(self):
        # 回归测试，用于检查 gh-5437。在使用 ICC 编译时可能会失败，因为 ICC 会将非规格化数变为零
        for ftype in sctypes['float']:
            stop = nextafter(ftype(0), ftype(1)) * 5  # 一个非规格化数
            assert_(any(linspace(0, stop, 10, endpoint=False, dtype=ftype)))

    def test_equivalent_to_arange(self):
        for j in range(1000):
            assert_equal(linspace(0, j, j+1, dtype=int),
                         arange(j+1, dtype=int))

    def test_retstep(self):
        for num in [0, 1, 2]:
            for ept in [False, True]:
                y = linspace(0, 1, num, endpoint=ept, retstep=True)
                assert isinstance(y, tuple) and len(y) == 2
                if num == 2:
                    y0_expect = [0.0, 1.0] if ept else [0.0, 0.5]
                    assert_array_equal(y[0], y0_expect)
                    assert_equal(y[1], y0_expect[1])
                elif num == 1 and not ept:
                    assert_array_equal(y[0], [0.0])
                    assert_equal(y[1], 1.0)
                else:
                    assert_array_equal(y[0], [0.0][:num])
                    assert isnan(y[1])

    def test_object(self):
        start = array(1, dtype='O')
        stop = array(2, dtype='O')
        y = linspace(start, stop, 3)
        assert_array_equal(y, array([1., 1.5, 2.]))
                    
    def test_round_negative(self):
        y = linspace(-1, 3, num=8, dtype=int)
        t = array([-1, -1, 0, 0, 1, 1, 2, 3], dtype=int)
        assert_array_equal(y, t)
    # 定义一个测试函数，测试在 any_step_zero 为 True 且 _mult_inplace 为 False 的情况下的行为
    def test_any_step_zero_and_not_mult_inplace(self):
        # 创建起始点数组，包含两个浮点数元素 [0.0, 1.0]
        start = array([0.0, 1.0])
        # 创建结束点数组，包含两个浮点数元素 [2.0, 1.0]
        stop = array([2.0, 1.0])
        # 使用 linspace 函数生成介于 start 和 stop 之间的均匀分布的数组，数组长度为 3
        y = linspace(start, stop, 3)
        # 使用 assert_array_equal 函数断言 y 的值等于预期的二维数组 [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
        assert_array_equal(y, array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]))
# 定义一个名为 TestAdd_newdoc 的测试类
class TestAdd_newdoc:

    # 用 pytest.mark.skipif 装饰器标记此测试，如果 Python 是以 -OO 优化模式运行，则跳过
    @pytest.mark.skipif(sys.flags.optimize == 2, reason="Python running -OO")
    # 用 pytest.mark.xfail 装饰器标记此测试，如果是在 PyPy 上运行，则标记为预期失败
    @pytest.mark.xfail(IS_PYPY, reason="PyPy does not modify tp_doc")
    # 定义一个测试方法 test_add_doc
    def test_add_doc(self):
        # 断言：验证 np.add_newdoc 成功地附加了文档字符串
        tgt = "Current flat index into the array."
        assert_equal(np._core.flatiter.index.__doc__[:len(tgt)], tgt)
        # 断言：验证 np._core.ufunc.identity 的文档字符串长度大于 300
        assert_(len(np._core.ufunc.identity.__doc__) > 300)
        # 断言：验证 np.lib._index_tricks_impl.mgrid 的文档字符串长度大于 300
        assert_(len(np.lib._index_tricks_impl.mgrid.__doc__) > 300)

    # 用 pytest.mark.skipif 装饰器标记此测试，如果 Python 是以 -OO 优化模式运行，则跳过
    @pytest.mark.skipif(sys.flags.optimize == 2, reason="Python running -OO")
    # 定义另一个测试方法 test_errors_are_ignored
    def test_errors_are_ignored(self):
        # 获取 np._core.flatiter.index 的当前文档字符串
        prev_doc = np._core.flatiter.index.__doc__
        # 添加一个新文档字符串到 "numpy._core" 模块的 "flatiter" 类的 "index" 方法
        add_newdoc("numpy._core", "flatiter", ("index", "bad docstring"))
        # 断言：验证 np._core.flatiter.index 的文档字符串没有改变
        assert prev_doc == np._core.flatiter.index.__doc__
```