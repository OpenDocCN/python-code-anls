# `.\numpy\numpy\polynomial\tests\test_classes.py`

```
"""Test inter-conversion of different polynomial classes.

This tests the convert and cast methods of all the polynomial classes.

"""
# 导入所需模块和库
import operator as op
from numbers import Number

import pytest  # 导入 pytest 测试框架
import numpy as np  # 导入 NumPy 库
from numpy.polynomial import (  # 导入 NumPy 中的多项式类
    Polynomial, Legendre, Chebyshev, Laguerre, Hermite, HermiteE)
from numpy.testing import (  # 导入 NumPy 测试工具
    assert_almost_equal, assert_raises, assert_equal, assert_,
    )
from numpy.exceptions import RankWarning  # 导入 NumPy 异常处理模块

#
# fixtures
#

# 定义测试用的多项式类和对应的名称
classes = (
    Polynomial, Legendre, Chebyshev, Laguerre,
    Hermite, HermiteE
    )
classids = tuple(cls.__name__ for cls in classes)

@pytest.fixture(params=classes, ids=classids)
def Poly(request):
    return request.param

#
# helper functions
#

# 定义辅助函数 assert_poly_almost_equal，用于比较两个多项式几乎相等
random = np.random.random


def assert_poly_almost_equal(p1, p2, msg=""):
    try:
        # 检查域和窗口是否相同
        assert_(np.all(p1.domain == p2.domain))
        assert_(np.all(p1.window == p2.window))
        # 检查系数是否几乎相等
        assert_almost_equal(p1.coef, p2.coef)
    except AssertionError:
        msg = f"Result: {p1}\nTarget: {p2}"
        raise AssertionError(msg)


#
# Test conversion methods that depend on combinations of two classes.
#

# 设置两个多项式类 Poly1 和 Poly2 用于测试转换
Poly1 = Poly
Poly2 = Poly


def test_conversion(Poly1, Poly2):
    x = np.linspace(0, 1, 10)
    coef = random((3,))

    d1 = Poly1.domain + random((2,))*.25
    w1 = Poly1.window + random((2,))*.25
    p1 = Poly1(coef, domain=d1, window=w1)

    d2 = Poly2.domain + random((2,))*.25
    w2 = Poly2.window + random((2,))*.25
    p2 = p1.convert(kind=Poly2, domain=d2, window=w2)

    # 断言转换后的多项式与预期的域和窗口几乎相等，并且在给定的点上几乎相等
    assert_almost_equal(p2.domain, d2)
    assert_almost_equal(p2.window, w2)
    assert_almost_equal(p2(x), p1(x))


def test_cast(Poly1, Poly2):
    x = np.linspace(0, 1, 10)
    coef = random((3,))

    d1 = Poly1.domain + random((2,))*.25
    w1 = Poly1.window + random((2,))*.25
    p1 = Poly1(coef, domain=d1, window=w1)

    d2 = Poly2.domain + random((2,))*.25
    w2 = Poly2.window + random((2,))*.25
    p2 = Poly2.cast(p1, domain=d2, window=w2)

    # 断言转换后的多项式与预期的域和窗口几乎相等，并且在给定的点上几乎相等
    assert_almost_equal(p2.domain, d2)
    assert_almost_equal(p2.window, w2)
    assert_almost_equal(p2(x), p1(x))


#
# test methods that depend on one class
#


def test_identity(Poly):
    d = Poly.domain + random((2,))*.25
    w = Poly.window + random((2,))*.25
    x = np.linspace(d[0], d[1], 11)
    p = Poly.identity(domain=d, window=w)
    # 断言身份多项式的域和窗口与预期相等，并且在给定的点上几乎相等
    assert_equal(p.domain, d)
    assert_equal(p.window, w)
    assert_almost_equal(p(x), x)


def test_basis(Poly):
    d = Poly.domain + random((2,))*.25
    w = Poly.window + random((2,))*.25
    p = Poly.basis(5, domain=d, window=w)
    # 断言基函数多项式的域和窗口与预期相等，并且系数为预期的基函数系数
    assert_equal(p.domain, d)
    assert_equal(p.window, w)
    assert_equal(p.coef, [0]*5 + [1])


def test_fromroots(Poly):
    # 检查通过给定根来生成多项式，确保多项式的次数、域和窗口正确
    d = Poly.domain + random((2,))*.25
    w = Poly.window + random((2,))*.25
    r = random((5,))
    p1 = Poly.fromroots(r, domain=d, window=w)
    assert_equal(p1.degree(), len(r))
    assert_equal(p1.domain, d)
    # 断言：验证对象 p1 的 window 属性是否与预期值 w 相等
    assert_equal(p1.window, w)
    # 断言：验证对象 p1 在输入 r 处的计算结果是否接近 0
    assert_almost_equal(p1(r), 0)

    # 检查多项式是否为首一多项式（最高次数项系数为 1）
    # 获取当前多项式类的域和窗口设置
    pdom = Polynomial.domain
    pwin = Polynomial.window
    # 将 p1 转换为指定域和窗口的多项式对象 p2
    p2 = Polynomial.cast(p1, domain=pdom, window=pwin)
    # 断言：验证 p2 的最高次数项系数是否接近于 1
    assert_almost_equal(p2.coef[-1], 1)
def test_bad_conditioned_fit(Poly):

    x = [0., 0., 1.]
    y = [1., 2., 3.]

    # 检查是否引发了 RankWarning 警告
    with pytest.warns(RankWarning) as record:
        # 对 Poly 对象进行拟合，预期引发 RankWarning 警告
        Poly.fit(x, y, 2)
    # 断言第一个记录的警告消息是 "The fit may be poorly conditioned"
    assert record[0].message.args[0] == "The fit may be poorly conditioned"


def test_fit(Poly):

    def f(x):
        return x*(x - 1)*(x - 2)
    x = np.linspace(0, 3)
    y = f(x)

    # 检查默认的 domain 和 window 值
    p = Poly.fit(x, y, 3)
    assert_almost_equal(p.domain, [0, 3])
    assert_almost_equal(p(x), y)
    assert_equal(p.degree(), 3)

    # 检查使用给定的 domain 和 window 值
    d = Poly.domain + random((2,))*.25
    w = Poly.window + random((2,))*.25
    p = Poly.fit(x, y, 3, domain=d, window=w)
    assert_almost_equal(p(x), y)
    assert_almost_equal(p.domain, d)
    assert_almost_equal(p.window, w)
    p = Poly.fit(x, y, [0, 1, 2, 3], domain=d, window=w)
    assert_almost_equal(p(x), y)
    assert_almost_equal(p.domain, d)
    assert_almost_equal(p.window, w)

    # 检查使用类的 domain 默认值
    p = Poly.fit(x, y, 3, [])
    assert_equal(p.domain, Poly.domain)
    assert_equal(p.window, Poly.window)
    p = Poly.fit(x, y, [0, 1, 2, 3], [])
    assert_equal(p.domain, Poly.domain)
    assert_equal(p.window, Poly.window)

    # 检查 fit 方法是否接受权重
    w = np.zeros_like(x)
    z = y + random(y.shape)*.25
    w[::2] = 1
    p1 = Poly.fit(x[::2], z[::2], 3)
    p2 = Poly.fit(x, z, 3, w=w)
    p3 = Poly.fit(x, z, [0, 1, 2, 3], w=w)
    assert_almost_equal(p1(x), p2(x))
    assert_almost_equal(p2(x), p3(x))


def test_equal(Poly):
    p1 = Poly([1, 2, 3], domain=[0, 1], window=[2, 3])
    p2 = Poly([1, 1, 1], domain=[0, 1], window=[2, 3])
    p3 = Poly([1, 2, 3], domain=[1, 2], window=[2, 3])
    p4 = Poly([1, 2, 3], domain=[0, 1], window=[1, 2])
    assert_(p1 == p1)
    assert_(not p1 == p2)
    assert_(not p1 == p3)
    assert_(not p1 == p4)


def test_not_equal(Poly):
    p1 = Poly([1, 2, 3], domain=[0, 1], window=[2, 3])
    p2 = Poly([1, 1, 1], domain=[0, 1], window=[2, 3])
    p3 = Poly([1, 2, 3], domain=[1, 2], window=[2, 3])
    p4 = Poly([1, 2, 3], domain=[0, 1], window=[1, 2])
    assert_(not p1 != p1)
    assert_(p1 != p2)
    assert_(p1 != p3)
    assert_(p1 != p4)


def test_add(Poly):
    # 检查加法的交换性，而非数值正确性
    c1 = list(random((4,)) + .5)
    c2 = list(random((3,)) + .5)
    p1 = Poly(c1)
    p2 = Poly(c2)
    p3 = p1 + p2
    assert_poly_almost_equal(p2 + p1, p3)
    assert_poly_almost_equal(p1 + c2, p3)
    assert_poly_almost_equal(c2 + p1, p3)
    assert_poly_almost_equal(p1 + tuple(c2), p3)
    assert_poly_almost_equal(tuple(c2) + p1, p3)
    assert_poly_almost_equal(p1 + np.array(c2), p3)
    assert_poly_almost_equal(np.array(c2) + p1, p3)
    assert_raises(TypeError, op.add, p1, Poly([0], domain=Poly.domain + 1))
    assert_raises(TypeError, op.add, p1, Poly([0], window=Poly.window + 1))
    # 如果 Poly 类型是 Polynomial 类型的话，执行以下语句块
    if Poly is Polynomial:
        # 断言应该会抛出 TypeError 异常，使用 op.add 对象试图将 p1 与 Chebyshev([0]) 相加
        assert_raises(TypeError, op.add, p1, Chebyshev([0]))
    # 如果 Poly 类型不是 Polynomial 类型的话，执行以下语句块
    else:
        # 断言应该会抛出 TypeError 异常，使用 op.add 对象试图将 p1 与 Polynomial([0]) 相加
        assert_raises(TypeError, op.add, p1, Polynomial([0]))
def test_sub(Poly):
    # 定义一个测试函数，用于测试多项式对象的减法操作
    # 这里的测试侧重于检查交换性，而不是数值上的正确性

    # 生成随机系数数组，并创建两个多项式对象
    c1 = list(random((4,)) + .5)
    c2 = list(random((3,)) + .5)
    p1 = Poly(c1)
    p2 = Poly(c2)

    # 计算两个多项式的差，并进行断言验证
    p3 = p1 - p2
    assert_poly_almost_equal(p2 - p1, -p3)  # 断言交换性质
    assert_poly_almost_equal(p1 - c2, p3)
    assert_poly_almost_equal(c2 - p1, -p3)
    assert_poly_almost_equal(p1 - tuple(c2), p3)
    assert_poly_almost_equal(tuple(c2) - p1, -p3)
    assert_poly_almost_equal(p1 - np.array(c2), p3)
    assert_poly_almost_equal(np.array(c2) - p1, -p3)

    # 测试异常情况：类型错误
    assert_raises(TypeError, op.sub, p1, Poly([0], domain=Poly.domain + 1))
    assert_raises(TypeError, op.sub, p1, Poly([0], window=Poly.window + 1))
    if Poly is Polynomial:
        assert_raises(TypeError, op.sub, p1, Chebyshev([0]))
    else:
        assert_raises(TypeError, op.sub, p1, Polynomial([0]))


def test_mul(Poly):
    # 定义一个测试函数，用于测试多项式对象的乘法操作

    # 生成随机系数数组，并创建两个多项式对象
    c1 = list(random((4,)) + .5)
    c2 = list(random((3,)) + .5)
    p1 = Poly(c1)
    p2 = Poly(c2)

    # 计算两个多项式的乘积，并进行断言验证
    p3 = p1 * p2
    assert_poly_almost_equal(p2 * p1, p3)  # 断言交换性质
    assert_poly_almost_equal(p1 * c2, p3)
    assert_poly_almost_equal(c2 * p1, p3)
    assert_poly_almost_equal(p1 * tuple(c2), p3)
    assert_poly_almost_equal(tuple(c2) * p1, p3)
    assert_poly_almost_equal(p1 * np.array(c2), p3)
    assert_poly_almost_equal(np.array(c2) * p1, p3)

    # 测试乘以标量的情况
    assert_poly_almost_equal(p1 * 2, p1 * Poly([2]))
    assert_poly_almost_equal(2 * p1, p1 * Poly([2]))

    # 测试异常情况：类型错误
    assert_raises(TypeError, op.mul, p1, Poly([0], domain=Poly.domain + 1))
    assert_raises(TypeError, op.mul, p1, Poly([0], window=Poly.window + 1))
    if Poly is Polynomial:
        assert_raises(TypeError, op.mul, p1, Chebyshev([0]))
    else:
        assert_raises(TypeError, op.mul, p1, Polynomial([0]))


def test_floordiv(Poly):
    # 定义一个测试函数，用于测试多项式对象的整除操作

    # 生成随机系数数组，并创建三个多项式对象
    c1 = list(random((4,)) + .5)
    c2 = list(random((3,)) + .5)
    c3 = list(random((2,)) + .5)
    p1 = Poly(c1)
    p2 = Poly(c2)
    p3 = Poly(c3)

    # 计算一个复合多项式，并从中提取系数作为数组
    p4 = p1 * p2 + p3
    c4 = list(p4.coef)

    # 断言整除操作的结果
    assert_poly_almost_equal(p4 // p2, p1)
    assert_poly_almost_equal(p4 // c2, p1)
    assert_poly_almost_equal(c4 // p2, p1)
    assert_poly_almost_equal(p4 // tuple(c2), p1)
    assert_poly_almost_equal(tuple(c4) // p2, p1)
    assert_poly_almost_equal(p4 // np.array(c2), p1)
    assert_poly_almost_equal(np.array(c4) // p2, p1)

    # 测试整除以标量的情况
    assert_poly_almost_equal(2 // p2, Poly([0]))
    assert_poly_almost_equal(p2 // 2, 0.5 * p2)

    # 测试异常情况：类型错误
    assert_raises(
        TypeError, op.floordiv, p1, Poly([0], domain=Poly.domain + 1))
    assert_raises(
        TypeError, op.floordiv, p1, Poly([0], window=Poly.window + 1))
    if Poly is Polynomial:
        assert_raises(TypeError, op.floordiv, p1, Chebyshev([0]))
    else:
        assert_raises(TypeError, op.floordiv, p1, Polynomial([0]))


def test_truediv(Poly):
    # 定义一个测试函数，用于测试多项式对象的真除操作

    # 创建一个简单的多项式对象和它的倍数
    p1 = Poly([1, 2, 3])
    p2 = p1 * 5

    # 这里没有具体的测试动作，仅提供了注释说明
    # 真除法仅在分母是数字且不是 Python 布尔值时有效
    # 遍历 numpy 中定义的标量类型
    for stype in np.ScalarType:
        # 如果当前类型不是 Number 的子类，或者是 bool 类型的子类，则跳过当前循环
        if not issubclass(stype, Number) or issubclass(stype, bool):
            continue
        # 使用当前标量类型 stype 创建一个值为 5 的实例 s
        s = stype(5)
        # 断言通过 op.truediv 计算 p2 除以 s 的结果与 p1 几乎相等
        assert_poly_almost_equal(op.truediv(p2, s), p1)
        # 断言调用 op.truediv 计算 s 除以 p2 会引发 TypeError 异常
        assert_raises(TypeError, op.truediv, s, p2)
    
    # 遍历特定的标量类型 int 和 float
    for stype in (int, float):
        # 使用当前类型 stype 创建一个值为 5 的实例 s
        s = stype(5)
        # 断言通过 op.truediv 计算 p2 除以 s 的结果与 p1 几乎相等
        assert_poly_almost_equal(op.truediv(p2, s), p1)
        # 断言调用 op.truediv 计算 s 除以 p2 会引发 TypeError 异常
        assert_raises(TypeError, op.truediv, s, p2)
    
    # 遍历特定的标量类型 complex（复数）
    for stype in [complex]:
        # 使用当前类型 stype 创建一个实部为 5、虚部为 0 的复数实例 s
        s = stype(5, 0)
        # 断言通过 op.truediv 计算 p2 除以 s 的结果与 p1 几乎相等
        assert_poly_almost_equal(op.truediv(p2, s), p1)
        # 断言调用 op.truediv 计算 s 除以 p2 会引发 TypeError 异常
        assert_raises(TypeError, op.truediv, s, p2)
    
    # 遍历非标量类型 tuple、list、dict、bool、numpy array
    for s in [tuple(), list(), dict(), bool(), np.array([1])]:
        # 断言调用 op.truediv 计算 p2 除以当前类型 s 会引发 TypeError 异常
        assert_raises(TypeError, op.truediv, p2, s)
        # 断言调用 op.truediv 计算当前类型 s 除以 p2 会引发 TypeError 异常
        assert_raises(TypeError, op.truediv, s, p2)
    
    # 遍历类列表 classes 中的每一个类类型 ptype
    for ptype in classes:
        # 断言调用 op.truediv 计算 p2 除以 ptype 类型的实例会引发 TypeError 异常
        assert_raises(TypeError, op.truediv, p2, ptype(1))
# 测试多项式类的模运算功能
def test_mod(Poly):
    # 检查交换性，不验证数值正确性
    c1 = list(random((4,)) + .5)   # 创建包含4个随机数的列表c1
    c2 = list(random((3,)) + .5)   # 创建包含3个随机数的列表c2
    c3 = list(random((2,)) + .5)   # 创建包含2个随机数的列表c3
    p1 = Poly(c1)                  # 使用列表c1创建多项式对象p1
    p2 = Poly(c2)                  # 使用列表c2创建多项式对象p2
    p3 = Poly(c3)                  # 使用列表c3创建多项式对象p3
    p4 = p1 * p2 + p3              # 计算多项式运算 p1 * p2 + p3，得到新的多项式对象p4
    c4 = list(p4.coef)             # 获取p4的系数并存储在列表c4中
    # 断言多项式模运算结果与p3几乎相等
    assert_poly_almost_equal(p4 % p2, p3)
    assert_poly_almost_equal(p4 % c2, p3)
    assert_poly_almost_equal(c4 % p2, p3)
    assert_poly_almost_equal(p4 % tuple(c2), p3)
    assert_poly_almost_equal(tuple(c4) % p2, p3)
    assert_poly_almost_equal(p4 % np.array(c2), p3)
    assert_poly_almost_equal(np.array(c4) % p2, p3)
    assert_poly_almost_equal(2 % p2, Poly([2]))
    assert_poly_almost_equal(p2 % 2, Poly([0]))
    # 检查是否引发TypeError异常，如果多项式的域或窗口不匹配
    assert_raises(TypeError, op.mod, p1, Poly([0], domain=Poly.domain + 1))
    assert_raises(TypeError, op.mod, p1, Poly([0], window=Poly.window + 1))
    # 如果Poly是Polynomial类，则继续检查是否引发TypeError异常
    if Poly is Polynomial:
        assert_raises(TypeError, op.mod, p1, Chebyshev([0]))
    else:
        assert_raises(TypeError, op.mod, p1, Polynomial([0]))


# 测试多项式类的整除取余功能
def test_divmod(Poly):
    # 检查交换性，不验证数值正确性
    c1 = list(random((4,)) + .5)   # 创建包含4个随机数的列表c1
    c2 = list(random((3,)) + .5)   # 创建包含3个随机数的列表c2
    c3 = list(random((2,)) + .5)   # 创建包含2个随机数的列表c3
    p1 = Poly(c1)                  # 使用列表c1创建多项式对象p1
    p2 = Poly(c2)                  # 使用列表c2创建多项式对象p2
    p3 = Poly(c3)                  # 使用列表c3创建多项式对象p3
    p4 = p1 * p2 + p3              # 计算多项式运算 p1 * p2 + p3，得到新的多项式对象p4
    c4 = list(p4.coef)             # 获取p4的系数并存储在列表c4中
    # 使用divmod函数计算p4除以p2的商和余数
    quo, rem = divmod(p4, p2)
    assert_poly_almost_equal(quo, p1)  # 断言商几乎等于p1
    assert_poly_almost_equal(rem, p3)  # 断言余数几乎等于p3
    quo, rem = divmod(p4, c2)
    assert_poly_almost_equal(quo, p1)
    assert_poly_almost_equal(rem, p3)
    quo, rem = divmod(c4, p2)
    assert_poly_almost_equal(quo, p1)
    assert_poly_almost_equal(rem, p3)
    quo, rem = divmod(p4, tuple(c2))
    assert_poly_almost_equal(quo, p1)
    assert_poly_almost_equal(rem, p3)
    quo, rem = divmod(tuple(c4), p2)
    assert_poly_almost_equal(quo, p1)
    assert_poly_almost_equal(rem, p3)
    quo, rem = divmod(p4, np.array(c2))
    assert_poly_almost_equal(quo, p1)
    assert_poly_almost_equal(rem, p3)
    quo, rem = divmod(np.array(c4), p2)
    assert_poly_almost_equal(quo, p1)
    assert_poly_almost_equal(rem, p3)
    quo, rem = divmod(p2, 2)
    assert_poly_almost_equal(quo, 0.5*p2)
    assert_poly_almost_equal(rem, Poly([0]))
    quo, rem = divmod(2, p2)
    assert_poly_almost_equal(quo, Poly([0]))
    assert_poly_almost_equal(rem, Poly([2]))
    # 检查是否引发TypeError异常，如果多项式的域或窗口不匹配
    assert_raises(TypeError, divmod, p1, Poly([0], domain=Poly.domain + 1))
    assert_raises(TypeError, divmod, p1, Poly([0], window=Poly.window + 1))
    # 如果Poly是Polynomial类，则继续检查是否引发TypeError异常
    if Poly is Polynomial:
        assert_raises(TypeError, divmod, p1, Chebyshev([0]))
    else:
        assert_raises(TypeError, divmod, p1, Polynomial([0]))


# 测试多项式类的根功能
def test_roots(Poly):
    d = Poly.domain * 1.25 + .25   # 计算新的域范围d
    w = Poly.window                # 获取窗口w
    tgt = np.linspace(d[0], d[1], 5)   # 创建一个包含5个均匀分布数值的数组tgt
    # 使用给定的根、域和窗口创建多项式，然后计算其根并排序
    res = np.sort(Poly.fromroots(tgt, domain=d, window=w).roots())
    assert_almost_equal(res, tgt)   # 断言计算出的根几乎等于tgt
    # 使用默认的域和窗口创建多项式，并计算其根并排序
    res = np.sort(Poly.fromroots(tgt).roots())
    assert_almost_equal(res, tgt)   # 断言计算出的根几乎等于tgt


# 测试多项式类的次数功能
def test_degree(Poly):
    p = Poly.basis(5)   # 创建一个次数为5的多项式对象p
    # 使用断言检查对象 p 的度数是否等于 5
    assert_equal(p.degree(), 5)
def test_copy(Poly):
    # 使用 Poly 类的 basis 方法创建一个多项式 p1
    p1 = Poly.basis(5)
    # 使用 p1 的 copy 方法复制出一个新的多项式 p2
    p2 = p1.copy()
    # 断言 p1 和 p2 的值相等
    assert_(p1 == p2)
    # 断言 p1 和 p2 不是同一个对象
    assert_(p1 is not p2)
    # 断言 p1 和 p2 的 coef 属性不是同一个对象
    assert_(p1.coef is not p2.coef)
    # 断言 p1 和 p2 的 domain 属性不是同一个对象
    assert_(p1.domain is not p2.domain)
    # 断言 p1 和 p2 的 window 属性不是同一个对象
    assert_(p1.window is not p2.window)


def test_integ(Poly):
    P = Polynomial
    # 检查默认情况
    p0 = Poly.cast(P([1*2, 2*3, 3*4]))
    # 对 p0 进行积分，并将结果转换为多项式 p1
    p1 = P.cast(p0.integ())
    # 对 p0 进行带参数的积分，并将结果转换为多项式 p2
    p2 = P.cast(p0.integ(2))
    # 断言 p1 和 P([0, 2, 3, 4]) 几乎相等
    assert_poly_almost_equal(p1, P([0, 2, 3, 4]))
    # 断言 p2 和 P([0, 0, 1, 1, 1]) 几乎相等
    assert_poly_almost_equal(p2, P([0, 0, 1, 1, 1]))
    # 检查带 k 参数的积分
    p0 = Poly.cast(P([1*2, 2*3, 3*4]))
    p1 = P.cast(p0.integ(k=1))
    p2 = P.cast(p0.integ(2, k=[1, 1]))
    assert_poly_almost_equal(p1, P([1, 2, 3, 4]))
    assert_poly_almost_equal(p2, P([1, 1, 1, 1, 1]))
    # 检查带 lbnd 参数的积分
    p0 = Poly.cast(P([1*2, 2*3, 3*4]))
    p1 = P.cast(p0.integ(lbnd=1))
    p2 = P.cast(p0.integ(2, lbnd=1))
    assert_poly_almost_equal(p1, P([-9, 2, 3, 4]))
    assert_poly_almost_equal(p2, P([6, -9, 1, 1, 1]))
    # 检查缩放
    d = 2*Poly.domain
    p0 = Poly.cast(P([1*2, 2*3, 3*4]), domain=d)
    p1 = P.cast(p0.integ())
    p2 = P.cast(p0.integ(2))
    assert_poly_almost_equal(p1, P([0, 2, 3, 4]))
    assert_poly_almost_equal(p2, P([0, 0, 1, 1, 1]))


def test_deriv(Poly):
    # 检查导数是否是积分的逆操作，假设积分在其他地方已经检查过。
    d = Poly.domain + random((2,))*.25
    w = Poly.window + random((2,))*.25
    p1 = Poly([1, 2, 3], domain=d, window=w)
    p2 = p1.integ(2, k=[1, 2])
    p3 = p1.integ(1, k=[1])
    assert_almost_equal(p2.deriv(1).coef, p3.coef)
    assert_almost_equal(p2.deriv(2).coef, p1.coef)
    # 默认的 domain 和 window
    p1 = Poly([1, 2, 3])
    p2 = p1.integ(2, k=[1, 2])
    p3 = p1.integ(1, k=[1])
    assert_almost_equal(p2.deriv(1).coef, p3.coef)
    assert_almost_equal(p2.deriv(2).coef, p1.coef)


def test_linspace(Poly):
    d = Poly.domain + random((2,))*.25
    w = Poly.window + random((2,))*.25
    p = Poly([1, 2, 3], domain=d, window=w)
    # 检查默认 domain
    xtgt = np.linspace(d[0], d[1], 20)
    ytgt = p(xtgt)
    xres, yres = p.linspace(20)
    assert_almost_equal(xres, xtgt)
    assert_almost_equal(yres, ytgt)
    # 检查指定 domain
    xtgt = np.linspace(0, 2, 20)
    ytgt = p(xtgt)
    xres, yres = p.linspace(20, domain=[0, 2])
    assert_almost_equal(xres, xtgt)
    assert_almost_equal(yres, ytgt)


def test_pow(Poly):
    d = Poly.domain + random((2,))*.25
    w = Poly.window + random((2,))*.25
    tgt = Poly([1], domain=d, window=w)
    tst = Poly([1, 2, 3], domain=d, window=w)
    for i in range(5):
        assert_poly_almost_equal(tst**i, tgt)
        tgt = tgt * tst
    # 默认 domain 和 window
    tgt = Poly([1])
    tst = Poly([1, 2, 3])
    for i in range(5):
        assert_poly_almost_equal(tst**i, tgt)
        tgt = tgt * tst
    # 检查无效指数的错误
    assert_raises(ValueError, op.pow, tgt, 1.5)
    assert_raises(ValueError, op.pow, tgt, -1)
# 定义一个测试函数，接受一个多项式类 Poly 作为参数
def test_call(Poly):
    # 将 Polynomial 类赋值给 P
    P = Polynomial
    # 获取 Poly 的域范围
    d = Poly.domain
    # 生成一个包含 11 个元素的 numpy 数组，元素均匀分布在 d[0] 到 d[1] 之间
    x = np.linspace(d[0], d[1], 11)

    # 检查默认情况
    p = Poly.cast(P([1, 2, 3]))  # 将 [1, 2, 3] 转换成 Poly 类型的多项式 p
    tgt = 1 + x*(2 + 3*x)  # 计算目标值，即 1 + x*(2 + 3*x)
    res = p(x)  # 计算多项式 p 在 x 上的取值
    assert_almost_equal(res, tgt)  # 断言 res 与 tgt 几乎相等


# 定义一个测试函数，接受一个多项式类 Poly 作为参数
def test_call_with_list(Poly):
    # 使用列表 [1, 2, 3] 创建 Poly 类型的多项式 p
    p = Poly([1, 2, 3])
    # 给定 x 值列表 [-1, 0, 2]
    x = [-1, 0, 2]
    # 计算多项式 p 在 x 上的取值
    res = p(x)
    # 断言 res 等于 p 在 numpy 数组 x 上的取值
    assert_equal(res, p(np.array(x)))


# 定义一个测试函数，接受一个多项式类 Poly 作为参数
def test_cutdeg(Poly):
    # 使用系数 [1, 2, 3] 创建 Poly 类型的多项式 p
    p = Poly([1, 2, 3])
    # 断言 cutdeg 方法对小数值抛出 ValueError 异常
    assert_raises(ValueError, p.cutdeg, .5)
    # 断言 cutdeg 方法对负数抛出 ValueError 异常
    assert_raises(ValueError, p.cutdeg, -1)
    # 断言 cutdeg 方法在阶数为 3 时返回的多项式长度为 3
    assert_equal(len(p.cutdeg(3)), 3)
    # 断言 cutdeg 方法在阶数为 2 时返回的多项式长度为 3
    assert_equal(len(p.cutdeg(2)), 3)
    # 断言 cutdeg 方法在阶数为 1 时返回的多项式长度为 2
    assert_equal(len(p.cutdeg(1)), 2)
    # 断言 cutdeg 方法在阶数为 0 时返回的多项式长度为 1
    assert_equal(len(p.cutdeg(0)), 1)


# 定义一个测试函数，接受一个多项式类 Poly 作为参数
def test_truncate(Poly):
    # 使用系数 [1, 2, 3] 创建 Poly 类型的多项式 p
    p = Poly([1, 2, 3])
    # 断言 truncate 方法对小数值抛出 ValueError 异常
    assert_raises(ValueError, p.truncate, .5)
    # 断言 truncate 方法对零抛出 ValueError 异常
    assert_raises(ValueError, p.truncate, 0)
    # 断言 truncate 方法在阶数为 4 时返回的多项式长度为 3
    assert_equal(len(p.truncate(4)), 3)
    # 断言 truncate 方法在阶数为 3 时返回的多项式长度为 3
    assert_equal(len(p.truncate(3)), 3)
    # 断言 truncate 方法在阶数为 2 时返回的多项式长度为 2
    assert_equal(len(p.truncate(2)), 2)
    # 断言 truncate 方法在阶数为 1 时返回的多项式长度为 1
    assert_equal(len(p.truncate(1)), 1)


# 定义一个测试函数，接受一个多项式类 Poly 作为参数
def test_trim(Poly):
    # 创建系数列表 [1, 1e-6, 1e-12, 0] 的 Poly 类型多项式 p
    c = [1, 1e-6, 1e-12, 0]
    p = Poly(c)
    # 断言 trim 方法后多项式的系数与预期相符，保留有效数字至第三项
    assert_equal(p.trim().coef, c[:3])
    # 断言 trim 方法保留小于给定阈值 1e-10 的系数，应返回前两项系数
    assert_equal(p.trim(1e-10).coef, c[:2])
    # 断言 trim 方法保留小于给定阈值 1e-5 的系数，应返回第一项系数
    assert_equal(p.trim(1e-5).coef, c[:1])


# 定义一个测试函数，接受一个多项式类 Poly 作为参数
def test_mapparms(Poly):
    # 检查默认情况下的参数映射，应返回标识映射 [0, 1]
    d = Poly.domain
    w = Poly.window
    p = Poly([1], domain=d, window=w)
    assert_almost_equal([0, 1], p.mapparms())
    #
    # 将窗口设置为 2*d + 1
    w = 2*d + 1
    p = Poly([1], domain=d, window=w)
    # 断言参数映射为 [1, 2]
    assert_almost_equal([1, 2], p.mapparms())


# 定义一个测试函数，接受一个多项式类 Poly 作为参数
def test_ufunc_override(Poly):
    # 使用系数 [1, 2, 3] 创建 Poly 类型的多项式 p
    p = Poly([1, 2, 3])
    # 创建一个全为 1 的 numpy 数组 x
    x = np.ones(3)
    # 断言使用 np.add 函数将多项式 p 与 x 相加抛出 TypeError 异常
    assert_raises(TypeError, np.add, p, x)
    # 断言使用 np.add 函数将 x 与多项式 p 相加抛出 TypeError 异常
    assert_raises(TypeError, np.add, x, p)


#
# 仅适用于某些类的测试类方法
#


class TestInterpolate:

    # 定义一个测试函数 f，接受 x 参数并返回 x*(x-1)*(x-2) 的结果
    def f(self, x):
        return x * (x - 1) * (x - 2)

    # 测试函数，断言 Chebyshev.interpolate 方法对于给定的参数抛出 ValueError 异常
    def test_raises(self):
        assert_raises(ValueError, Chebyshev.interpolate, self.f, -1)
        assert_raises(TypeError, Chebyshev.interpolate, self.f, 10.)

    # 测试函数，断言 Chebyshev.interpolate 方法返回的多项式阶数与预期相符
    def test_dimensions(self):
        for deg in range(1, 5):
            assert_(Chebyshev.interpolate(self.f, deg).degree() == deg)

    # 测试函数，断言 Chebyshev.interpolate 方法返回的多项式与原函数 powx 在给定区间上的近似程度
    def test_approximation(self):

        # 定义函数 powx，接受 x 和 p 参数，返回 x 的 p 次幂
        def powx(x, p):
            return x**p

        # 创建一个包含 10 个元素的 numpy 数组 x，均匀分布在 [0, 2] 区间上
        x = np.linspace(0, 2, 10)
        # 遍历不同的多项式阶数 deg
        for deg in range(0, 10):
            # 遍历不同的次幂 t
            for t in range(0, deg + 1):
                # 使用 Chebyshev.interpolate 方法创建多项式 p，近似函数 powx 的结果
                p = Chebyshev.interpolate(powx, deg, domain=[0, 2], args=(t,))
                # 断言多项式 p 在数组 x 上的取值与 powx(x, t) 的近似程度在 11 位小数上相等
```