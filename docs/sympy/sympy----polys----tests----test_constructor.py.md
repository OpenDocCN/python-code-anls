# `D:\src\scipysrc\sympy\sympy\polys\tests\test_constructor.py`

```
"""Tests for tools for constructing domains for expressions. """

# 从 sympy.polys.constructor 模块导入 construct_domain 函数
from sympy.polys.constructor import construct_domain
# 从 sympy.polys.domains 中导入整数环 ZZ, 有理数域 QQ, 整数环扩展到复数域的 ZZ_I 和 QQ_I
from sympy.polys.domains import ZZ, QQ, ZZ_I, QQ_I, RR, CC, EX
# 从 sympy.polys.domains.realfield 中导入 RealField 类
from sympy.polys.domains.realfield import RealField
# 从 sympy.polys.domains.complexfield 中导入 ComplexField 类

from sympy.core import (Catalan, GoldenRatio)
from sympy.core.numbers import (E, Float, I, Rational, pi)
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.abc import x, y

# 定义测试函数 test_construct_domain
def test_construct_domain():

    # 断言测试 construct_domain 函数对于 [1, 2, 3] 返回 (ZZ, [ZZ(1), ZZ(2), ZZ(3)])
    assert construct_domain([1, 2, 3]) == (ZZ, [ZZ(1), ZZ(2), ZZ(3)])
    # 断言测试 construct_domain 函数对于 [1, 2, 3] 且 field=True 返回 (QQ, [QQ(1), QQ(2), QQ(3)])
    assert construct_domain([1, 2, 3], field=True) == (QQ, [QQ(1), QQ(2), QQ(3)])

    # 断言测试 construct_domain 函数对于 [S.One, S(2), S(3)] 返回 (ZZ, [ZZ(1), ZZ(2), ZZ(3)])
    assert construct_domain([S.One, S(2), S(3)]) == (ZZ, [ZZ(1), ZZ(2), ZZ(3)])
    # 断言测试 construct_domain 函数对于 [S.One, S(2), S(3)] 且 field=True 返回 (QQ, [QQ(1), QQ(2), QQ(3)])
    assert construct_domain([S.One, S(2), S(3)], field=True) == (QQ, [QQ(1), QQ(2), QQ(3)])

    # 断言测试 construct_domain 函数对于 [S.Half, S(2)] 返回 (QQ, [QQ(1, 2), QQ(2)])
    assert construct_domain([S.Half, S(2)]) == (QQ, [QQ(1, 2), QQ(2)])
    # 测试 construct_domain 函数对于 [3.14, 1, S.Half] 返回的结果类型是 RealField，并且数据部分符合预期
    result = construct_domain([3.14, 1, S.Half])
    assert isinstance(result[0], RealField)
    assert result[1] == [RR(3.14), RR(1.0), RR(0.5)]

    # 测试 construct_domain 函数对于 [3.14, I, S.Half] 返回的结果类型是 ComplexField，并且数据部分符合预期
    result = construct_domain([3.14, I, S.Half])
    assert isinstance(result[0], ComplexField)
    assert result[1] == [CC(3.14), CC(1.0j), CC(0.5)]

    # 断言测试 construct_domain 函数对于 [1.0+I] 返回 (CC, [CC(1.0, 1.0)])
    assert construct_domain([1.0+I]) == (CC, [CC(1.0, 1.0)])
    # 断言测试 construct_domain 函数对于 [2.0+3.0*I] 返回 (CC, [CC(2.0, 3.0)])
    assert construct_domain([2.0+3.0*I]) == (CC, [CC(2.0, 3.0)])

    # 断言测试 construct_domain 函数对于 [1, I] 返回 (ZZ_I, [ZZ_I(1, 0), ZZ_I(0, 1)])
    assert construct_domain([1, I]) == (ZZ_I, [ZZ_I(1, 0), ZZ_I(0, 1)])
    # 断言测试 construct_domain 函数对于 [1, I/2] 返回 (QQ_I, [QQ_I(1, 0), QQ_I(0, S.Half)])
    assert construct_domain([1, I/2]) == (QQ_I, [QQ_I(1, 0), QQ_I(0, S.Half)])

    # 断言测试 construct_domain 函数对于 [3.14, sqrt(2)] 且 extension=None 返回 (EX, [EX(3.14), EX(sqrt(2))])
    assert construct_domain([3.14, sqrt(2)], extension=None) == (EX, [EX(3.14), EX(sqrt(2))])
    # 断言测试 construct_domain 函数对于 [3.14, sqrt(2)] 且 extension=True 返回 (EX, [EX(3.14), EX(sqrt(2))])
    assert construct_domain([3.14, sqrt(2)], extension=True) == (EX, [EX(3.14), EX(sqrt(2))])

    # 断言测试 construct_domain 函数对于 [1, sqrt(2)] 且 extension=None 返回 (EX, [EX(1), EX(sqrt(2))])
    assert construct_domain([1, sqrt(2)], extension=None) == (EX, [EX(1), EX(sqrt(2))])

    # 断言测试 construct_domain 函数对于 [x, sqrt(x)] 返回 (EX, [EX(x), EX(sqrt(x))])
    assert construct_domain([x, sqrt(x)]) == (EX, [EX(x), EX(sqrt(x))])
    # 断言测试 construct_domain 函数对于 [x, sqrt(x), sqrt(y)] 返回 (EX, [EX(x), EX(sqrt(x)), EX(sqrt(y))])
    assert construct_domain([x, sqrt(x), sqrt(y)]) == (EX, [EX(x), EX(sqrt(x)), EX(sqrt(y))])

    # 测试构造 QQ 中关于 sqrt(2) 的代数扩展
    alg = QQ.algebraic_field(sqrt(2))
    assert construct_domain([7, S.Half, sqrt(2)], extension=True) == \
        (alg, [alg.convert(7), alg.convert(S.Half), alg.convert(sqrt(2))])

    # 测试构造 QQ 中关于 sqrt(2) + sqrt(3) 的代数扩展
    alg = QQ.algebraic_field(sqrt(2) + sqrt(3))
    assert construct_domain([7, sqrt(2), sqrt(3)], extension=True) == \
        (alg, [alg.convert(7), alg.convert(sqrt(2)), alg.convert(sqrt(3))])

    # 测试构造 ZZ[x] 中的多项式环
    dom = ZZ[x]
    assert construct_domain([2*x, 3]) == \
        (dom, [dom.convert(2*x), dom.convert(3)])

    # 测试构造 ZZ[x, y] 中的多项式环
    dom = ZZ[x, y]
    assert construct_domain([2*x, 3*y]) == \
        (dom, [dom.convert(2*x), dom.convert(3*y)])

    # 测试构造 QQ[x] 中的多项式环
    dom = QQ[x]
    assert construct_domain([x/2, 3]) == \
        (dom, [dom.convert(x/2), dom.convert(3)])

    # 测试构造 QQ[x, y] 中的多项式环
    dom = QQ[x, y]
    assert construct_domain([x/2, 3*y]) == \
        (dom, [dom.convert(x/2), dom.convert(3*y)])

    # 测试构造 ZZ_I[x] 中的多项式环
    dom = ZZ_I[x]
    # 断言：使用构造函数 construct_domain 对输入参数进行测试
    assert construct_domain([2*x, I]) == \
        (dom, [dom.convert(2*x), dom.convert(I)])

    # 定义一个多项式环，包含整数环和两个变量 x, y
    dom = ZZ_I[x, y]

    # 断言：使用构造函数 construct_domain 对输入参数进行测试
    assert construct_domain([2*x, I*y]) == \
        (dom, [dom.convert(2*x), dom.convert(I*y)])

    # 定义一个有理数域，包含一个变量 x
    dom = QQ_I[x]

    # 断言：使用构造函数 construct_domain 对输入参数进行测试
    assert construct_domain([x/2, I]) == \
        (dom, [dom.convert(x/2), dom.convert(I)])

    # 定义一个有理数域，包含两个变量 x, y
    dom = QQ_I[x, y]

    # 断言：使用构造函数 construct_domain 对输入参数进行测试
    assert construct_domain([x/2, I*y]) == \
        (dom, [dom.convert(x/2), dom.convert(I*y)])

    # 定义一个实数域，包含一个变量 x
    dom = RR[x]

    # 断言：使用构造函数 construct_domain 对输入参数进行测试
    assert construct_domain([x/2, 3.5]) == \
        (dom, [dom.convert(x/2), dom.convert(3.5)])

    # 定义一个实数域，包含两个变量 x, y
    dom = RR[x, y]

    # 断言：使用构造函数 construct_domain 对输入参数进行测试
    assert construct_domain([x/2, 3.5*y]) == \
        (dom, [dom.convert(x/2), dom.convert(3.5*y)])

    # 定义一个复数域，包含一个变量 x
    dom = CC[x]

    # 断言：使用构造函数 construct_domain 对输入参数进行测试
    assert construct_domain([I*x/2, 3.5]) == \
        (dom, [dom.convert(I*x/2), dom.convert(3.5)])

    # 定义一个复数域，包含两个变量 x, y
    dom = CC[x, y]

    # 断言：使用构造函数 construct_domain 对输入参数进行测试
    assert construct_domain([I*x/2, 3.5*y]) == \
        (dom, [dom.convert(I*x/2), dom.convert(3.5*y)])

    # 定义一个复数域，包含一个变量 x
    dom = CC[x]

    # 断言：使用构造函数 construct_domain 对输入参数进行测试
    assert construct_domain([x/2, I*3.5]) == \
        (dom, [dom.convert(x/2), dom.convert(I*3.5)])

    # 定义一个复数域，包含两个变量 x, y
    dom = CC[x, y]

    # 断言：使用构造函数 construct_domain 对输入参数进行测试
    assert construct_domain([x/2, I*3.5*y]) == \
        (dom, [dom.convert(x/2), dom.convert(I*3.5*y)])

    # 定义一个有理数域的分式域，包含一个变量 x
    dom = ZZ.frac_field(x)

    # 断言：使用构造函数 construct_domain 对输入参数进行测试
    assert construct_domain([2/x, 3]) == \
        (dom, [dom.convert(2/x), dom.convert(3)])

    # 定义一个有理数域的分式域，包含两个变量 x, y
    dom = ZZ.frac_field(x, y)

    # 断言：使用构造函数 construct_domain 对输入参数进行测试
    assert construct_domain([2/x, 3*y]) == \
        (dom, [dom.convert(2/x), dom.convert(3*y)])

    # 定义一个实数域的分式域，包含一个变量 x
    dom = RR.frac_field(x)

    # 断言：使用构造函数 construct_domain 对输入参数进行测试
    assert construct_domain([2/x, 3.5]) == \
        (dom, [dom.convert(2/x), dom.convert(3.5)])

    # 定义一个实数域的分式域，包含两个变量 x, y
    dom = RR.frac_field(x, y)

    # 断言：使用构造函数 construct_domain 对输入参数进行测试
    assert construct_domain([2/x, 3.5*y]) == \
        (dom, [dom.convert(2/x), dom.convert(3.5*y)])

    # 定义一个精度为 336 的实数域，包含一个变量 x
    dom = RealField(prec=336)[x]

    # 断言：使用构造函数 construct_domain 对输入参数进行测试
    assert construct_domain([pi.evalf(100)*x]) == \
        (dom, [dom.convert(pi.evalf(100)*x)])

    # 断言：使用构造函数 construct_domain 对整数 2 进行测试
    assert construct_domain(2) == (ZZ, ZZ(2))
    
    # 断言：使用构造函数 construct_domain 对有理数 S(2)/3 进行测试
    assert construct_domain(S(2)/3) == (QQ, QQ(2, 3))
    
    # 断言：使用构造函数 construct_domain 对有理数 Rational(2, 3) 进行测试
    assert construct_domain(Rational(2, 3)) == (QQ, QQ(2, 3))
    
    # 断言：使用构造函数 construct_domain 对空字典进行测试
    assert construct_domain({}) == (ZZ, {})
# 测试复数指数的情况
def test_complex_exponential():
    # 计算复数指数 e^(-i * 2 * pi / 3)，并禁用求值
    w = exp(-I * 2 * pi / 3, evaluate=False)
    # 为 w 创建一个有理数域
    alg = QQ.algebraic_field(w)
    # 断言构建域 [w**2, w, 1] 的结果，扩展域为真
    assert construct_domain([w**2, w, 1], extension=True) == (
        alg,
        [alg.convert(w**2),
         alg.convert(w),
         alg.convert(1)]
    )


# 测试复合选项的情况
def test_composite_option():
    # 断言构建域 {(1,): sin(y)} 的结果，composite=False
    assert construct_domain({(1,): sin(y)}, composite=False) == \
        (EX, {(1,): EX(sin(y))})

    # 断言构建域 {(1,): y} 的结果，composite=False
    assert construct_domain({(1,): y}, composite=False) == \
        (EX, {(1,): EX(y)})

    # 断言构建域 {(1, 1): 1} 的结果，composite=False
    assert construct_domain({(1, 1): 1}, composite=False) == \
        (ZZ, {(1, 1): 1})

    # 断言构建域 {(1, 0): y} 的结果，composite=False
    assert construct_domain({(1, 0): y}, composite=False) == \
        (EX, {(1, 0): EX(y)})


# 测试精度的情况
def test_precision():
    # 创建浮点数 f1 和 f2
    f1 = Float("1.01")
    f2 = Float("1.0000000000000000000001")
    # 遍历不同的测试用例
    for u in [1, 1e-2, 1e-6, 1e-13, 1e-14, 1e-16, 1e-20, 1e-100, 1e-300,
              f1, f2]:
        # 计算 construct_domain([u]) 的结果
        result = construct_domain([u])
        # 获取结果的浮点数表示
        v = float(result[1][0])
        # 断言相对精度小于 1e-14
        assert abs(u - v) / u < 1e-14  # 测试相对精度

    # 断言 construct_domain([f1]) 的结果的精度
    result = construct_domain([f1])
    y = result[1][0]
    assert y - 1 > 1e-50

    # 断言 construct_domain([f2]) 的结果的精度
    result = construct_domain([f2])
    y = result[1][0]
    assert y - 1 > 1e-50


# 测试问题 11538 的情况
def test_issue_11538():
    # 遍历常数 E, pi, Catalan
    for n in [E, pi, Catalan]:
        # 断言 construct_domain(n) 的结果的整数部分是整数环 ZZ[n]
        assert construct_domain(n)[0] == ZZ[n]
        # 断言 construct_domain(x + n) 的结果包含变量 x 和常数 n
        assert construct_domain(x + n)[0] == ZZ[x, n]
    # 断言 construct_domain(GoldenRatio) 的结果是扩展域 EX
    assert construct_domain(GoldenRatio)[0] == EX
    # 断言 construct_domain(x + GoldenRatio) 的结果包含变量 x 和 GoldenRatio
    assert construct_domain(x + GoldenRatio)[0] == EX
```