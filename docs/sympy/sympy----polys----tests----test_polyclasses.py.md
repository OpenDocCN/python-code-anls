# `D:\src\scipysrc\sympy\sympy\polys\tests\test_polyclasses.py`

```
"""Tests for OO layer of several polynomial representations. """

# 导入所需的库函数和模块
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys.domains import ZZ, QQ
from sympy.polys.polyclasses import DMP, DMF, ANP
from sympy.polys.polyerrors import (CoercionFailed, ExactQuotientFailed,
                                    NotInvertible)
from sympy.polys.specialpolys import f_polys
from sympy.testing.pytest import raises, warns_deprecated_sympy

# 从 f_polys 函数返回的多项式列表中，转换为稠密表示，并分别赋值给变量 f_0 到 f_6
f_0, f_1, f_2, f_3, f_4, f_5, f_6 = [ f.to_dense() for f in f_polys() ]

# 定义测试函数 test_DMP___init__()
def test_DMP___init__():
    # 创建 DMP 对象 f，使用指定的系数表示和域
    f = DMP([[ZZ(0)], [], [ZZ(0), ZZ(1), ZZ(2)], [ZZ(3)]], ZZ)
    
    # 断言检查对象属性
    assert f._rep == [[1, 2], [3]]
    assert f.dom == ZZ
    assert f.lev == 1

    # 创建 DMP 对象 f，使用指定的系数表示、指定的域和级别
    f = DMP([[ZZ(1), ZZ(2)], [ZZ(3)]], ZZ, 1)
    
    # 断言检查对象属性
    assert f._rep == [[1, 2], [3]]
    assert f.dom == ZZ
    assert f.lev == 1

    # 使用字典创建 DMP 对象 f，给定指数到系数的映射、级别和域
    f = DMP.from_dict({(1, 1): ZZ(1), (0, 0): ZZ(2)}, 1, ZZ)
    
    # 断言检查对象属性
    assert f._rep == [[1, 0], [2]]
    assert f.dom == ZZ
    assert f.lev == 1


# 定义测试函数 test_DMP_rep_deprecation()
def test_DMP_rep_deprecation():
    # 创建 DMP 对象 f，使用指定的系数表示和域
    f = DMP([1, 2, 3], ZZ)

    # 使用 warns_deprecated_sympy() 上下文管理器捕获警告
    with warns_deprecated_sympy():
        # 断言检查对象的 rep 属性
        assert f.rep == [1, 2, 3]


# 定义测试函数 test_DMP___eq__()
def test_DMP___eq__():
    # 断言比较两个 DMP 对象是否相等，使用相同的系数表示和域
    assert DMP([[ZZ(1), ZZ(2)], [ZZ(3)]], ZZ) == \
        DMP([[ZZ(1), ZZ(2)], [ZZ(3)]], ZZ)

    # 断言比较两个 DMP 对象是否相等，系数使用不同的域表示
    assert DMP([[ZZ(1), ZZ(2)], [ZZ(3)]], ZZ) == \
        DMP([[QQ(1), QQ(2)], [QQ(3)]], QQ)
    assert DMP([[QQ(1), QQ(2)], [QQ(3)]], QQ) == \
        DMP([[ZZ(1), ZZ(2)], [ZZ(3)]], ZZ)

    # 断言比较两个 DMP 对象是否不相等，系数结构不同
    assert DMP([[[ZZ(1)]]], ZZ) != DMP([[ZZ(1)]], ZZ)
    assert DMP([[ZZ(1)]], ZZ) != DMP([[[ZZ(1)]]], ZZ)


# 定义测试函数 test_DMP___bool__()
def test_DMP___bool__():
    # 断言检查 DMP 对象的布尔值，空列表表示 False
    assert bool(DMP([[]], ZZ)) is False
    # 断言检查 DMP 对象的布尔值，非空列表表示 True
    assert bool(DMP([[ZZ(1)]], ZZ)) is True


# 定义测试函数 test_DMP_to_dict()
def test_DMP_to_dict():
    # 创建 DMP 对象 f，使用指定的系数表示和域
    f = DMP([[ZZ(3)], [], [ZZ(2)], [], [ZZ(8)]], ZZ)

    # 断言检查 to_dict 方法的输出，将多项式表示为指数到系数的字典
    assert f.to_dict() == \
        {(4, 0): 3, (2, 0): 2, (0, 0): 8}
    
    # 断言检查 to_sympy_dict 方法的输出，将多项式表示为 SymPy 可识别的字典
    assert f.to_sympy_dict() == \
        {(4, 0): ZZ.to_sympy(3), (2, 0): ZZ.to_sympy(2), (0, 0):
         ZZ.to_sympy(8)}


# 定义测试函数 test_DMP_properties()
def test_DMP_properties():
    # 断言检查 DMP 对象的属性 is_zero，空列表表示 True
    assert DMP([[]], ZZ).is_zero is True
    # 断言检查 DMP 对象的属性 is_zero，非空列表表示 False
    assert DMP([[ZZ(1)]], ZZ).is_zero is False

    # 断言检查 DMP 对象的属性 is_one，表示为 [1] 表示 True
    assert DMP([[ZZ(1)]], ZZ).is_one is True
    # 断言检查 DMP 对象的属性 is_one，非 [1] 表示 False
    assert DMP([[ZZ(2)]], ZZ).is_one is False

    # 断言检查 DMP 对象的属性 is_ground，表示为单一常数表示 True
    assert DMP([[ZZ(1)]], ZZ).is_ground is True
    # 断言检查 DMP 对象的属性 is_ground，包含多项式表示 False
    assert DMP([[ZZ(1)], [ZZ(2)], [ZZ(1)]], ZZ).is_ground is False

    # 断言检查 DMP 对象的属性 is_sqf，表示为平方自由表示 True
    assert DMP([[ZZ(1)], [ZZ(2), ZZ(0)], [ZZ(1), ZZ(0)]], ZZ).is_sqf is True
    # 断言检查 DMP 对象的属性 is_sqf，包含平方自由表示 False
    assert DMP([[ZZ(1)], [ZZ(2), ZZ(0)], [ZZ(1), ZZ(0), ZZ(0)]], ZZ).is_sqf is False

    # 断言检查 DMP 对象的属性 is_monic，首项系数为 1 表示 True
    assert DMP([[ZZ(1), ZZ(2)], [ZZ(3)]], ZZ).is_monic is True
    # 断言检查 DMP 对象的属性 is_monic，首项系数不为 1 表示 False
    assert DMP([[ZZ(2), ZZ(2)], [ZZ(3)]], ZZ).is_monic is False

    # 断言检查 DMP 对象的属性 is_primitive，系数表示是否原始表示 True
    assert DMP([[ZZ(1), ZZ(2)], [ZZ(3)]], ZZ).is_primitive is True
    # 断言检查 DMP 对象的属性 is_primitive，系数表示不是原始表示 False
    assert DMP([[ZZ(2), ZZ(4)], [ZZ(6)]], ZZ).is_primitive is False


# 定义测试函数 test_DMP_arithmetics()
def test_DMP_arithmetics():
    # 创建 DMP 对象 f，使用指定的系数表示和域
    f = DMP([[ZZ(2)], [ZZ(2), ZZ(0)]], ZZ)

    # 断言检查 DMP 对象的乘以常数操作
    assert f.mul_ground(2) == DMP([[ZZ(4)], [ZZ(4), ZZ(0)]], ZZ)
    # 断言检查 DMP 对象的除以常数操作
    assert f.quo_ground(2) == DMP([[ZZ(1)], [ZZ(1), ZZ(0)]], ZZ)

    # 使用 lambda 函数检查精确除法操作引发的异常
    raises(ExactQuotientFailed, lambda: f.exquo_ground(3))

    # 创建两个 DMP 对象 f 和 g
    # 断言 f 的绝对值等于 g
    assert abs(f) == g

    # 断言 g 的负值等于 f
    assert g.neg() == f
    # 断言 g 的负值等于 -f
    assert -g == f

    # 创建一个 DMP 对象 h，表示一个空多项式
    h = DMP([[]], ZZ)

    # 断言 f 加上 g 等于 h
    assert f.add(g) == h
    # 断言 f 加上 g 等于 h（使用运算符重载）
    assert f + g == h
    # 断言 g 加上 f 等于 h（交换律）
    assert g + f == h
    # 断言 f 加上 5 等于 h
    assert f + 5 == h
    # 断言 5 加上 f 等于 h（交换律）
    assert 5 + f == h

    # 创建一个 DMP 对象 h，表示一个包含 -10 的多项式
    h = DMP([[ZZ(-10)]], ZZ)

    # 断言 f 减去 g 等于 h
    assert f.sub(g) == h
    # 断言 f 减去 g 等于 h（使用运算符重载）
    assert f - g == h
    # 断言 g 减去 f 等于 -h
    assert g - f == -h
    # 断言 f 减去 5 等于 h
    assert f - 5 == h
    # 断言 5 减去 f 等于 -h
    assert 5 - f == -h

    # 创建一个 DMP 对象 h，表示一个包含 -25 的多项式
    h = DMP([[ZZ(-25)]], ZZ)

    # 断言 f 乘以 g 等于 h
    assert f.mul(g) == h
    # 断言 f 乘以 g 等于 h（使用运算符重载）
    assert f * g == h
    # 断言 g 乘以 f 等于 h（乘法交换律）
    assert g * f == h
    # 断言 f 乘以 5 等于 h
    assert f * 5 == h
    # 断言 5 乘以 f 等于 h（乘法交换律）
    assert 5 * f == h

    # 创建一个 DMP 对象 h，表示一个包含 25 的多项式
    h = DMP([[ZZ(25)]], ZZ)

    # 断言 f 的平方等于 h
    assert f.sqr() == h
    # 断言 f 的平方等于 h（使用 pow 方法）
    assert f.pow(2) == h
    # 断言 f 的平方等于 h（使用 ** 运算符）
    assert f**2 == h

    # 使用 lambda 函数断言 f 的 'x' 次幂引发 TypeError 异常
    raises(TypeError, lambda: f.pow('x'))

    # 重新定义 f 和 g 的值
    f = DMP([[ZZ(1)], [], [ZZ(1), ZZ(0), ZZ(0)]], ZZ)
    g = DMP([[ZZ(2)], [ZZ(-2), ZZ(0)]], ZZ)

    # 创建商 q 和余数 r 的 DMP 对象
    q = DMP([[ZZ(2)], [ZZ(2), ZZ(0)]], ZZ)
    r = DMP([[ZZ(8), ZZ(0), ZZ(0)]], ZZ)

    # 断言 f 除以 g 的多项式除法等于 (q, r)
    assert f.pdiv(g) == (q, r)
    # 断言 f 除以 g 的多项式商等于 q
    assert f.pquo(g) == q
    # 断言 f 除以 g 的多项式余数等于 r
    assert f.prem(g) == r

    # 使用 lambda 函数断言 f 除以 g 的精确商引发 ExactQuotientFailed 异常
    raises(ExactQuotientFailed, lambda: f.pexquo(g))

    # 重新定义 f 和 g 的值
    f = DMP([[ZZ(1)], [], [ZZ(1), ZZ(0), ZZ(0)]], ZZ)
    g = DMP([[ZZ(1)], [ZZ(-1), ZZ(0)]], ZZ)

    # 创建商 q 和余数 r 的 DMP 对象
    q = DMP([[ZZ(1)], [ZZ(1), ZZ(0)]], ZZ)
    r = DMP([[ZZ(2), ZZ(0), ZZ(0)]], ZZ)

    # 断言 f 除以 g 的整数除法等于 (q, r)
    assert f.div(g) == (q, r)
    # 断言 f 除以 g 的整数商等于 q
    assert f.quo(g) == q
    # 断言 f 除以 g 的整数余数等于 r
    assert f.rem(g) == r

    # 使用 divmod 函数断言 f 除以 g 的商和余数等于 (q, r)
    assert divmod(f, g) == (q, r)
    # 断言 f 除以 g 的整数商等于 q
    assert f // g == q
    # 断言 f 除以 g 的整数余数等于 r
    assert f % g == r

    # 使用 lambda 函数断言 f 除以 g 的精确商引发 ExactQuotientFailed 异常
    raises(ExactQuotientFailed, lambda: f.exquo(g))

    # 重新定义 f 和 g 的值
    f = DMP([ZZ(1), ZZ(0), ZZ(-1)], ZZ)
    g = DMP([ZZ(2), ZZ(-2)], ZZ)

    # 创建商 q 和余数 r 的 DMP 对象
    q = DMP([], ZZ)
    r = f

    # 创建整数商 pq 和整数余数 pr 的 DMP 对象
    pq = DMP([ZZ(2), ZZ(2)], ZZ)
    pr = DMP([], ZZ)

    # 断言 f 除以 g 的整数除法等于 (q, r)
    assert f.div(g) == (q, r)
    # 断言 f 除以 g 的整数商等于 q
    assert f.quo(g) == q
    # 断言 f 除以 g 的整数余数等于 r
    assert f.rem(g) == r

    # 使用 divmod 函数断言 f 除以 g 的商和余数等于 (q, r)
    assert divmod(f, g) == (q, r)
    # 断言 f 除以 g 的整数商等于 q
    assert f // g == q
    # 断言 f 除以 g 的整数余数等于 r
    assert f % g == r

    # 使用 lambda 函数断言 f 除以 g 的精确商引发 ExactQuotientFailed 异常
    raises(ExactQuotientFailed, lambda: f.exquo(g))

    # 断言 f 除以 g 的多项式除法等于 (pq, pr)
    assert f.pdiv(g) == (pq, pr)
    # 断言 f 除以 g 的多项式商等于 pq
    assert f.pquo(g) == pq
    # 断言 f 除以 g 的多项式余数等于 pr
    assert f.prem(g) == pr
    # 断言 f 除以 g 的多项式精确商等于 pq
    assert f.pexquo(g) == pq
# 定义测试函数 test_DMP_functionality
def test_DMP_functionality():
    # 创建多项式对象 f，使用整数环 ZZ
    f = DMP([[ZZ(1)], [ZZ(2), ZZ(0)], [ZZ(1), ZZ(0), ZZ(0)]], ZZ)
    # 创建多项式对象 g，使用整数环 ZZ
    g = DMP([[ZZ(1)], [ZZ(1), ZZ(0)]], ZZ)
    # 创建多项式对象 h，使用整数环 ZZ
    h = DMP([[ZZ(1)]], ZZ)

    # 断言 f 的次数为 2
    assert f.degree() == 2
    # 断言 f 的各项次数列表为 (2, 2)
    assert f.degree_list() == (2, 2)
    # 断言 f 的总次数为 2
    assert f.total_degree() == 2

    # 断言 f 的首项系数为 1
    assert f.LC() == ZZ(1)
    # 断言 f 的尾项系数为 0
    assert f.TC() == ZZ(0)
    # 断言 f 的第一个一次项系数为 2
    assert f.nth(1, 1) == ZZ(2)

    # 测试当输入非整数时，期望引发 TypeError 异常
    raises(TypeError, lambda: f.nth(0, 'x'))

    # 断言 f 的最大范数为 2
    assert f.max_norm() == 2
    # 断言 f 的 L1 范数为 4
    assert f.l1_norm() == 4

    # 创建多项式对象 u，使用整数环 ZZ
    u = DMP([[ZZ(2)], [ZZ(2), ZZ(0)]], ZZ)

    # 断言 f 对 m=1, j=0 的求导结果为 u
    assert f.diff(m=1, j=0) == u
    # 断言 f 对 m=1, j=1 的求导结果为 u
    assert f.diff(m=1, j=1) == u

    # 测试当输入非整数时，期望引发 TypeError 异常
    raises(TypeError, lambda: f.diff(m='x', j=0))

    # 创建多项式对象 u 和 v，使用整数环 ZZ
    u = DMP([ZZ(1), ZZ(2), ZZ(1)], ZZ)
    v = DMP([ZZ(1), ZZ(2), ZZ(1)], ZZ)

    # 断言 f 在 a=1, j=0 处求值结果为 u
    assert f.eval(a=1, j=0) == u
    # 断言 f 在 a=1, j=1 处求值结果为 v
    assert f.eval(a=1, j=1) == v

    # 断言 f 在两次求值后的结果为 4
    assert f.eval(1).eval(1) == ZZ(4)

    # 断言 f 与 g 的因式对应操作结果为 (g, g, h)
    assert f.cofactors(g) == (g, g, h)
    # 断言 f 与 g 的最大公约数为 g
    assert f.gcd(g) == g
    # 断言 f 与 g 的最小公倍数为 f
    assert f.lcm(g) == f

    # 创建多项式对象 u 和 v，使用有理数环 QQ
    u = DMP([[QQ(45), QQ(30), QQ(5)]], QQ)
    v = DMP([[QQ(1), QQ(2, 3), QQ(1, 9)]], QQ)

    # 断言 u 的首项系数归一化结果为 v
    assert u.monic() == v

    # 断言 (4*f) 的内容为 4
    assert (4*f).content() == ZZ(4)
    # 断言 (4*f) 的原始部分为 (4, f)
    assert (4*f).primitive() == (ZZ(4), f)

    # 创建多项式对象 f 和 g，使用有理数环 QQ
    f = DMP([QQ(1,3), QQ(1)], QQ)
    g = DMP([QQ(1,7), QQ(1)], QQ)

    # 断言 f 与 g 的取消化结果为 (取消结果, 包含模式为 True 时的取消结果) 和 (取消结果, 包含模式为 False 时的取消结果)
    assert f.cancel(g) == f.cancel(g, include=True) == (
        DMP([QQ(7), QQ(21)], QQ),
        DMP([QQ(3), QQ(21)], QQ)
    )
    assert f.cancel(g, include=False) == (
        QQ(7),
        QQ(3),
        DMP([QQ(1), QQ(3)], QQ),
        DMP([QQ(1), QQ(7)], QQ)
    )

    # 创建多项式对象 f，使用整数环 ZZ
    f = DMP([[ZZ(1)], [ZZ(2)], [ZZ(3)], [ZZ(4)], [ZZ(5)], [ZZ(6)]], ZZ)

    # 断言 f 对于截断 3 的结果为 DMP([[ZZ(1)], [ZZ(-1)], [], [ZZ(1)], [ZZ(-1)], []], ZZ)
    assert f.trunc(3) == DMP([[ZZ(1)], [ZZ(-1)], [], [ZZ(1)], [ZZ(-1)], []], ZZ)

    # 重新定义多项式对象 f，使用 f_4 变量和整数环 ZZ
    f = DMP(f_4, ZZ)

    # 断言 f 的平方自由部分为 -f
    assert f.sqf_part() == -f
    # 断言 f 的平方自由列表为 (ZZ(-1), [(-f, 1)])
    assert f.sqf_list() == (ZZ(-1), [(-f, 1)])

    # 创建多项式对象 f, g 和 h，使用整数环 ZZ
    f = DMP([[ZZ(-1)], [], [], [ZZ(5)]], ZZ)
    g = DMP([[ZZ(3), ZZ(1)], [], []], ZZ)
    h = DMP([[ZZ(45), ZZ(30), ZZ(5)]], ZZ)

    # 断言 f 对 g 的子结果为 [f, g, h]
    assert f.subresultants(g) == [f, g, h]
    # 断言 f 对 g 的结果为 r
    assert f.resultant(g) == r

    # 创建多项式对象 f，使用整数环 ZZ
    f = DMP([ZZ(1), ZZ(3), ZZ(9), ZZ(-13)], ZZ)

    # 断言 f 的判别式为 -11664
    assert f.discriminant() == -11664

    # 创建多项式对象 f 和 g，使用有理数环 QQ
    f = DMP([QQ(2), QQ(0)], QQ)
    g = DMP([QQ(1), QQ(0), QQ(-16)], QQ)

    # 创建多项式对象 s, t 和 h，使用有理数环 QQ
    s = DMP([QQ(1, 32), QQ(0)], QQ)
    t = DMP([QQ(-1, 16)], QQ)
    h = DMP([QQ(1)], QQ)

    # 断言 f 对 g 的半扩展欧算法结果为 (s, h)
    assert f.half_gcdex(g) == (s, h)
    # 断言 f 对 g 的扩展欧几里得算法结果为 (s, t, h)
    assert f.gcdex(g) == (s, t, h)

    # 断言 f 对 g 的求逆结果为 s
    assert f.invert(g) == s

    # 重新定义多项式对象 f，使用有理数环 QQ
    f = DMP([[QQ(1)], [QQ(2)], [QQ(3)]], QQ)

    # 测试当输入参数与 f 相同时，期望引发 ValueError 异常
    raises(ValueError, lambda: f.half_gcdex(f))
    raises(ValueError, lambda: f.gcdex(f))

    # 测试当输入参数与 f 相同时，期望引发 ValueError 异常
    raises(ValueError, lambda: f.invert(f))

    # 创建多项式对象 f, g 和 h，使用整数环 ZZ
    f = DMP(ZZ.map([1, 0, 20, 0, 150, 0, 500, 0, 625, -2, 0, -10, 9]), ZZ)
    g = DMP([ZZ(1), ZZ(0), ZZ(0),
    # 定义一个列表 J，包含整数 0 到 25（注意：23 缺失）
    J = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 24, 25]
    
    # 断言：使用 DMP 类对 f 进行操作，并指定环 ZZ（整数环）后应该返回 (J, DMP([ZZ(1), ZZ(0)], ZZ))
    assert DMP(f, ZZ).exclude() == (J, DMP([ZZ(1), ZZ(0)], ZZ))
    
    # 断言：使用 DMP 类对二维列表 [[ZZ(1)], [ZZ(1), ZZ(0)]] 进行操作，并指定环 ZZ 后应该返回 ([], DMP([[ZZ(1)], [ZZ(1), ZZ(0)]], ZZ))
    assert DMP([[ZZ(1)], [ZZ(1), ZZ(0)]], ZZ).exclude() ==\
            ([], DMP([[ZZ(1)], [ZZ(1), ZZ(0)]], ZZ))
# 定义测试函数 test_DMF__init__()
def test_DMF__init__():
    # 创建 DMF 对象 f，使用给定的参数进行初始化
    f = DMF(([[0], [], [0, 1, 2], [3]], [[1, 2, 3]]), ZZ)

    # 断言对象 f 的属性值符合预期
    assert f.num == [[1, 2], [3]]
    assert f.den == [[1, 2, 3]]
    assert f.lev == 1
    assert f.dom == ZZ

    # 创建 DMF 对象 f，使用另一组参数进行初始化
    f = DMF(([[1, 2], [3]], [[1, 2, 3]]), ZZ, 1)

    # 断言对象 f 的属性值符合预期
    assert f.num == [[1, 2], [3]]
    assert f.den == [[1, 2, 3]]
    assert f.lev == 1
    assert f.dom == ZZ

    # 创建 DMF 对象 f，使用另一组参数进行初始化
    f = DMF(([[-1], [-2]], [[3], [-4]]), ZZ)

    # 断言对象 f 的属性值符合预期
    assert f.num == [[-1], [-2]]
    assert f.den == [[3], [-4]]
    assert f.lev == 1
    assert f.dom == ZZ

    # 创建 DMF 对象 f，使用另一组参数进行初始化
    f = DMF(([[1], [2]], [[-3], [4]]), ZZ)

    # 断言对象 f 的属性值符合预期
    assert f.num == [[-1], [-2]]
    assert f.den == [[3], [-4]]
    assert f.lev == 1
    assert f.dom == ZZ

    # 创建 DMF 对象 f，使用另一组参数进行初始化
    f = DMF(([[]], [[-3], [4]]), ZZ)

    # 断言对象 f 的属性值符合预期
    assert f.num == [[]]
    assert f.den == [[1]]
    assert f.lev == 1
    assert f.dom == ZZ

    # 创建 DMF 对象 f，使用另一组参数进行初始化
    f = DMF(17, ZZ, 1)

    # 断言对象 f 的属性值符合预期
    assert f.num == [[17]]
    assert f.den == [[1]]
    assert f.lev == 1
    assert f.dom == ZZ

    # 创建 DMF 对象 f，使用另一组参数进行初始化
    f = DMF(([[1], [2]]), ZZ)

    # 断言对象 f 的属性值符合预期
    assert f.num == [[1], [2]]
    assert f.den == [[1]]
    assert f.lev == 1
    assert f.dom == ZZ

    # 创建 DMF 对象 f，使用另一组参数进行初始化
    f = DMF([[0], [], [0, 1, 2], [3]], ZZ)

    # 断言对象 f 的属性值符合预期
    assert f.num == [[1, 2], [3]]
    assert f.den == [[1]]
    assert f.lev == 1
    assert f.dom == ZZ

    # 创建 DMF 对象 f，使用另一组参数进行初始化
    f = DMF({(1, 1): 1, (0, 0): 2}, ZZ, 1)

    # 断言对象 f 的属性值符合预期
    assert f.num == [[1, 0], [2]]
    assert f.den == [[1]]
    assert f.lev == 1
    assert f.dom == ZZ

    # 创建 DMF 对象 f，使用另一组参数进行初始化
    f = DMF(([[QQ(1)], [QQ(2)]], [[-QQ(3)], [QQ(4)]]), QQ)

    # 断言对象 f 的属性值符合预期
    assert f.num == [[-QQ(1)], [-QQ(2)]]
    assert f.den == [[QQ(3)], [-QQ(4)]]
    assert f.lev == 1
    assert f.dom == QQ

    # 创建 DMF 对象 f，使用另一组参数进行初始化
    f = DMF(([[QQ(1, 5)], [QQ(2, 5)]], [[-QQ(3, 7)], [QQ(4, 7)]]), QQ)

    # 断言对象 f 的属性值符合预期
    assert f.num == [[-QQ(7)], [-QQ(14)]]
    assert f.den == [[QQ(15)], [-QQ(20)]]
    assert f.lev == 1
    assert f.dom == QQ

    # 测试 DMF 对象初始化时的异常情况处理
    raises(ValueError, lambda: DMF(([1], [[1]]), ZZ))
    raises(ZeroDivisionError, lambda: DMF(([1], []), ZZ))


# 定义测试函数 test_DMF__bool__()
def test_DMF__bool__():
    # 断言空 DMF 对象的布尔值为 False
    assert bool(DMF([[]], ZZ)) is False
    # 断言非空 DMF 对象的布尔值为 True
    assert bool(DMF([[1]], ZZ)) is True


# 定义测试函数 test_DMF_properties()
def test_DMF_properties():
    # 断言空 DMF 对象的 is_zero 属性为 True，is_one 属性为 False
    assert DMF([[]], ZZ).is_zero is True
    assert DMF([[]], ZZ).is_one is False

    # 断言非空 DMF 对象的 is_zero 属性为 False，is_one 属性为 True
    assert DMF([[1]], ZZ).is_zero is False
    assert DMF([[1]], ZZ).is_one is True

    # 断言包含分子和分母的 DMF 对象的 is_one 属性为 False
    assert DMF(([[1]], [[2]]), ZZ).is_one is False


# 定义测试函数 test_DMF_arithmetics()
def test_DMF_arithmetics():
    # 创建两个 DMF 对象 f 和 g，并进行相关算术运算的断言
    f = DMF([[7], [-9]], ZZ)
    g = DMF([[-7], [9]], ZZ)

    assert f.neg() == -f == g

    f = DMF(([[1]], [[1], []]), ZZ)
    g = DMF(([[1]], [[1, 0]]), ZZ)

    h = DMF(([[1], [1, 0]], [[1, 0], []]), ZZ)

    assert f.add(g) == f + g == h
    assert g.add(f) == g + f == h

    h = DMF(([[-1], [1, 0]], [[1, 0], []]), ZZ)

    assert f.sub(g) == f - g == h

    h = DMF(([[1]], [[1, 0], []]), ZZ)

    assert f.mul(g) == f*g == h
    assert g.mul(f) == g*f == h

    h = DMF(([[1, 0]], [[1], []]), ZZ)

    assert f.quo(g) == f/g == h

    h = DMF(([[1]], [[1], [], [], []]), ZZ)
    # 断言：验证 f 的立方等于 f 的三次方等于 h
    assert f.pow(3) == f**3 == h

    # 创建一个 DMF 对象 h，参数是两个矩阵：([[1]], [[1, 0, 0, 0]])，并指定环为 ZZ
    h = DMF(([[1]], [[1, 0, 0, 0]]), ZZ)

    # 断言：验证 g 的立方等于 g 的三次方等于 h
    assert g.pow(3) == g**3 == h

    # 更新 h 为一个新的 DMF 对象，参数是两个矩阵：([[1, 0]], [[1]])，环为 ZZ
    h = DMF(([[1, 0]], [[1]]), ZZ)

    # 断言：验证 g 的逆的立方等于 g 的逆的三次方等于 h
    assert g.pow(-1) == g**-1 == h
# 定义一个测试函数，用于测试 ANP 类的初始化方法 __init__
def test_ANP___init__():
    # 创建两个列表作为 ANP 类初始化的参数
    rep = [QQ(1), QQ(1)]
    mod = [QQ(1), QQ(0), QQ(1)]

    # 使用初始化参数创建 ANP 对象 f
    f = ANP(rep, mod, QQ)

    # 断言 f 的 to_list 方法返回的列表与预期结果相同
    assert f.to_list() == [QQ(1), QQ(1)]
    # 断言 f 的 mod_to_list 方法返回的列表与预期结果相同
    assert f.mod_to_list() == [QQ(1), QQ(0), QQ(1)]
    # 断言 f 的 dom 属性与预期的 QQ 类型相同
    assert f.dom == QQ

    # 修改初始化参数为字典形式
    rep = {1: QQ(1), 0: QQ(1)}
    mod = {2: QQ(1), 0: QQ(1)}

    # 使用新的初始化参数创建 ANP 对象 f
    f = ANP(rep, mod, QQ)

    # 断言 f 的 to_list 方法返回的列表与预期结果相同
    assert f.to_list() == [QQ(1), QQ(1)]
    # 断言 f 的 mod_to_list 方法返回的列表与预期结果相同
    assert f.mod_to_list() == [QQ(1), QQ(0), QQ(1)]
    # 断言 f 的 dom 属性与预期的 QQ 类型相同
    assert f.dom == QQ

    # 使用部分参数为整数和列表的形式创建 ANP 对象 f
    f = ANP(1, mod, QQ)

    # 断言 f 的 to_list 方法返回的列表与预期结果相同
    assert f.to_list() == [QQ(1)]
    # 断言 f 的 mod_to_list 方法返回的列表与预期结果相同
    assert f.mod_to_list() == [QQ(1), QQ(0), QQ(1)]
    # 断言 f 的 dom 属性与预期的 QQ 类型相同
    assert f.dom == QQ

    # 使用部分参数为列表和模数的形式创建 ANP 对象 f
    f = ANP([1, 0.5], mod, QQ)

    # 断言 f 的 to_list 方法返回的所有元素均为 QQ 类型
    assert all(QQ.of_type(a) for a in f.to_list())

    # 使用不支持的参数类型创建 ANP 对象，预期会抛出 CoercionFailed 异常
    raises(CoercionFailed, lambda: ANP([sqrt(2)], mod, QQ))


# 定义一个测试函数，用于测试 ANP 类的相等性判断方法 __eq__
def test_ANP___eq__():
    # 创建两个 ANP 对象 a 和 b
    a = ANP([QQ(1), QQ(1)], [QQ(1), QQ(0), QQ(1)], QQ)
    b = ANP([QQ(1), QQ(1)], [QQ(1), QQ(0), QQ(2)], QQ)

    # 断言 a 等于自身，应返回 True
    assert (a == a) is True
    # 断言 a 不等于自身，应返回 False
    assert (a != a) is False

    # 断言 a 不等于 b，应返回 False
    assert (a == b) is False
    # 断言 a 不等于 b，应返回 True
    assert (a != b) is True

    # 修改 ANP 对象 b 的部分参数，使其与 a 不相等
    b = ANP([QQ(1), QQ(2)], [QQ(1), QQ(0), QQ(1)], QQ)

    # 断言 a 不等于 b，应返回 False
    assert (a == b) is False
    # 断言 a 不等于 b，应返回 True
    assert (a != b) is True


# 定义一个测试函数，用于测试 ANP 类的布尔值转换方法 __bool__
def test_ANP___bool__():
    # 断言空列表作为 rep 参数时，ANP 对象的布尔值为 False
    assert bool(ANP([], [QQ(1), QQ(0), QQ(1)], QQ)) is False
    # 断言包含 QQ(1) 的列表作为 rep 参数时，ANP 对象的布尔值为 True
    assert bool(ANP([QQ(1)], [QQ(1), QQ(0), QQ(1)], QQ)) is True


# 定义一个测试函数，用于测试 ANP 类的属性方法
def test_ANP_properties():
    # 创建模数列表 mod
    mod = [QQ(1), QQ(0), QQ(1)]

    # 断言 ANP 对象表示的值为 [QQ(0)] 时，is_zero 方法返回 True
    assert ANP([QQ(0)], mod, QQ).is_zero is True
    # 断言 ANP 对象表示的值为 [QQ(1)] 时，is_zero 方法返回 False
    assert ANP([QQ(1)], mod, QQ).is_zero is False

    # 断言 ANP 对象表示的值为 [QQ(1)] 时，is_one 方法返回 True
    assert ANP([QQ(1)], mod, QQ).is_one is True
    # 断言 ANP 对象表示的值为 [QQ(2)] 时，is_one 方法返回 False
    assert ANP([QQ(2)], mod, QQ).is_one is False


# 定义一个测试函数，用于测试 ANP 类的算术运算方法
def test_ANP_arithmetics():
    # 创建模数列表 mod
    mod = [QQ(1), QQ(0), QQ(0), QQ(-2)]

    # 创建两个 ANP 对象 a 和 b
    a = ANP([QQ(2), QQ(-1), QQ(1)], mod, QQ)
    b = ANP([QQ(1), QQ(2)], mod, QQ)

    # 创建用于比较的 ANP 对象 c
    c = ANP([QQ(-2), QQ(1), QQ(-1)], mod, QQ)

    # 断言 a 的相反数等于 -a，应与 c 相等
    assert a.neg() == -a == c

    # 修改 ANP 对象 a 的参数，使其与 c 相等
    c = ANP([QQ(2), QQ(0), QQ(3)], mod, QQ)

    # 断言 a 加 b 等于 a + b，应与 c 相等
    assert a.add(b) == a + b == c
    # 断言 b 加 a 等于 b + a，应与 c 相等
    assert b.add(a) == b + a == c

    # 修改 ANP 对象 a 和 b 的参数，使其与 c 不相等
    c = ANP([QQ(2), QQ(-2), QQ(-1)], mod, QQ)

    # 断言 a 减 b 等于 a - b，应与 c 相等
    assert a.sub(b) == a - b == c

    # 修改 ANP 对象 b 的参数，使其与 c 不相等
    c = ANP([QQ(-2), QQ(2), QQ(1)], mod, QQ)

    # 断言 b 减 a 等于 b - a，应与 c 相等
    assert b.sub(a) == b - a == c

    # 修改 ANP 对象 c 的参数，使其与预期结果不同
    c = ANP([QQ(3), QQ(-1), QQ(6)], mod, QQ)

    # 断言 a 乘以 b 等于 a * b，应与 c 相等
    assert a.mul(b) == a*b == c
    # 断言 b 乘以 a 等于 b * a，应与 c 相等
    assert b.mul(a) == b*a == c

    # 创建 ANP 对象 c 作为预期结果
    c = ANP([QQ(-1, 43), QQ(9, 43), QQ(5, 43)], mod, QQ)

    # 断言 a 的零次幂等于 a ** 0，应与 ANP(1, mod, QQ) 相等
    assert a.pow(0) == a**(0) == ANP(1, mod, QQ)
    # 断言 a 的一次幂等于 a ** 1，应与 a 相等
    assert a.pow(1) == a**(1) == a

    # 断言 a 的负一次幂等于 a ** -1，应与 c
    # 断言：确保将 b 统一化到 a 的类型，并且结果为 QQ 类型
    assert b.unify(a)[0] == QQ
    
    # 断言：确保将 a 统一化到自身的类型，并且结果为 QQ 类型
    assert a.unify(a)[0] == QQ
    
    # 断言：确保将 b 统一化到自身的类型，并且结果为 ZZ 类型
    assert b.unify(b)[0] == ZZ
    
    # 断言：确保将 a 统一化到 b 的 ANP（具有特定属性的新原子）类型，并且结果的最后一个元素为 QQ 类型
    assert a.unify_ANP(b)[-1] == QQ
    
    # 断言：确保将 b 统一化到 a 的 ANP 类型，并且结果的最后一个元素为 QQ 类型
    assert b.unify_ANP(a)[-1] == QQ
    
    # 断言：确保将 a 统一化到自身的 ANP 类型，并且结果的最后一个元素为 QQ 类型
    assert a.unify_ANP(a)[-1] == QQ
    
    # 断言：确保将 b 统一化到自身的 ANP 类型，并且结果的最后一个元素为 ZZ 类型
    assert b.unify_ANP(b)[-1] == ZZ
```