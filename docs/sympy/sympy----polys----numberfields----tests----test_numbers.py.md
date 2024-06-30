# `D:\src\scipysrc\sympy\sympy\polys\numberfields\tests\test_numbers.py`

```
"""Tests on algebraic numbers. """

# 导入所需模块和类
from sympy.core.containers import Tuple
from sympy.core.numbers import (AlgebraicNumber, I, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys.polytools import Poly
from sympy.polys.numberfields.subfield import to_number_field
from sympy.polys.polyclasses import DMP
from sympy.polys.domains import QQ
from sympy.polys.rootoftools import CRootOf
from sympy.abc import x, y

# 定义测试函数 test_AlgebraicNumber
def test_AlgebraicNumber():
    # 定义最小多项式和代数数的根
    minpoly, root = x**2 - 2, sqrt(2)

    # 创建代数数对象 a
    a = AlgebraicNumber(root, gen=x)

    # 断言各属性值
    assert a.rep == DMP([QQ(1), QQ(0)], QQ)
    assert a.root == root
    assert a.alias is None
    assert a.minpoly == minpoly
    assert a.is_number

    # 断言 is_aliased 属性为 False
    assert a.is_aliased is False

    # 断言 coeffs() 和 native_coeffs() 方法返回的值
    assert a.coeffs() == [S.One, S.Zero]
    assert a.native_coeffs() == [QQ(1), QQ(0)]

    # 创建带别名的代数数对象 a
    a = AlgebraicNumber(root, gen=x, alias='y')

    # 断言各属性值
    assert a.rep == DMP([QQ(1), QQ(0)], QQ)
    assert a.root == root
    assert a.alias == Symbol('y')
    assert a.minpoly == minpoly
    assert a.is_number

    # 断言 is_aliased 属性为 True
    assert a.is_aliased is True

    # 创建带 Symbol 类型别名的代数数对象 a
    a = AlgebraicNumber(root, gen=x, alias=Symbol('y'))

    # 断言各属性值
    assert a.rep == DMP([QQ(1), QQ(0)], QQ)
    assert a.root == root
    assert a.alias == Symbol('y')
    assert a.minpoly == minpoly
    assert a.is_number

    # 断言 is_aliased 属性为 True
    assert a.is_aliased is True

    # 断言不同输入的 AlgebraicNumber 对象的表示
    assert AlgebraicNumber(sqrt(2), []).rep == DMP([], QQ)
    assert AlgebraicNumber(sqrt(2), ()).rep == DMP([], QQ)
    assert AlgebraicNumber(sqrt(2), (0, 0)).rep == DMP([], QQ)

    assert AlgebraicNumber(sqrt(2), [8]).rep == DMP([QQ(8)], QQ)
    assert AlgebraicNumber(sqrt(2), [Rational(8, 3)]).rep == DMP([QQ(8, 3)], QQ)

    assert AlgebraicNumber(sqrt(2), [7, 3]).rep == DMP([QQ(7), QQ(3)], QQ)
    assert AlgebraicNumber(
        sqrt(2), [Rational(7, 9), Rational(3, 2)]).rep == DMP([QQ(7, 9), QQ(3, 2)], QQ)

    assert AlgebraicNumber(sqrt(2), [1, 2, 3]).rep == DMP([QQ(2), QQ(5)], QQ)

    # 创建嵌套 AlgebraicNumber 对象的代数数对象 a
    a = AlgebraicNumber(AlgebraicNumber(root, gen=x), [1, 2])

    # 断言各属性值
    assert a.rep == DMP([QQ(1), QQ(2)], QQ)
    assert a.root == root
    assert a.alias is None
    assert a.minpoly == minpoly
    assert a.is_number

    # 断言 is_aliased 属性为 False
    assert a.is_aliased is False

    # 断言 coeffs() 和 native_coeffs() 方法返回的值
    assert a.coeffs() == [S.One, S(2)]
    assert a.native_coeffs() == [QQ(1), QQ(2)]

    # 创建带有元组输入的代数数对象 a
    a = AlgebraicNumber((minpoly, root), [1, 2])

    # 断言各属性值
    assert a.rep == DMP([QQ(1), QQ(2)], QQ)
    assert a.root == root
    assert a.alias is None
    assert a.minpoly == minpoly
    assert a.is_number

    # 断言 is_aliased 属性为 False
    assert a.is_aliased is False

    # 创建带有多项式输入的代数数对象 a
    a = AlgebraicNumber((Poly(minpoly), root), [1, 2])

    # 断言各属性值
    assert a.rep == DMP([QQ(1), QQ(2)], QQ)
    assert a.root == root
    assert a.alias is None
    assert a.minpoly == minpoly
    assert a.is_number

    # 断言 is_aliased 属性为 False
    assert a.is_aliased is False

    # 断言使用正负根号创建的代数数对象的表示
    assert AlgebraicNumber( sqrt(3)).rep == DMP([ QQ(1), QQ(0)], QQ)
    assert AlgebraicNumber(-sqrt(3)).rep == DMP([ QQ(1), QQ(0)], QQ)

    # 创建仅带根的代数数对象 a
    a = AlgebraicNumber(sqrt(2))
    # 创建一个代数数域对象 `b`，使用平方根 2 初始化
    b = AlgebraicNumber(sqrt(2))
    
    # 断言 `a` 等于 `b`
    assert a == b
    
    # 创建一个代数数域对象 `c`，使用平方根 2 和生成元 `x` 初始化
    c = AlgebraicNumber(sqrt(2), gen=x)
    
    # 断言 `a` 同时等于 `b` 和 `c`
    assert a == b
    assert a == c
    
    # 使用平方根 2 和系数 `[1, 2]` 创建代数数域对象 `a`
    a = AlgebraicNumber(sqrt(2), [1, 2])
    # 使用平方根 2 和系数 `[1, 3]` 创建代数数域对象 `b`
    b = AlgebraicNumber(sqrt(2), [1, 3])
    
    # 断言 `a` 不等于 `b`，且 `a` 不等于 `sqrt(2) + 3`
    assert a != b and a != sqrt(2) + 3
    
    # 断言 `(a == x)` 为假，`(a != x)` 为真
    assert (a == x) is False and (a != x) is True
    
    # 使用平方根 2 和系数 `[1, 0]` 创建代数数域对象 `a`
    a = AlgebraicNumber(sqrt(2), [1, 0])
    # 使用平方根 2、系数 `[1, 0]` 和别名 `y` 创建代数数域对象 `b`
    b = AlgebraicNumber(sqrt(2), [1, 0], alias=y)
    
    # 断言 `a` 转换为多项式后结果等于 `Poly(x, domain='QQ')`
    assert a.as_poly(x) == Poly(x, domain='QQ')
    # 断言 `b` 转换为多项式后结果等于 `Poly(y, domain='QQ')`
    assert b.as_poly() == Poly(y, domain='QQ')
    
    # 断言 `a` 转换为表达式后结果等于 `sqrt(2)`
    assert a.as_expr() == sqrt(2)
    # 断言 `a` 转换为表达式（带变量 `x`）后结果等于 `x`
    assert a.as_expr(x) == x
    # 断言 `b` 转换为表达式后结果等于 `sqrt(2)`
    assert b.as_expr() == sqrt(2)
    # 断言 `b` 转换为表达式（带变量 `x`）后结果等于 `x`
    assert b.as_expr(x) == x
    
    # 使用平方根 2 和系数 `[2, 3]` 创建代数数域对象 `a`
    a = AlgebraicNumber(sqrt(2), [2, 3])
    # 使用平方根 2、系数 `[2, 3]` 和别名 `y` 创建代数数域对象 `b`
    b = AlgebraicNumber(sqrt(2), [2, 3], alias=y)
    
    # 将 `a` 转换为多项式 `p`
    p = a.as_poly()
    
    # 断言 `p` 等于 `Poly(2*p.gen + 3)`
    assert p == Poly(2*p.gen + 3)
    
    # 断言 `a` 转换为多项式后结果（带变量 `x`）等于 `Poly(2*x + 3, domain='QQ')`
    assert a.as_poly(x) == Poly(2*x + 3, domain='QQ')
    # 断言 `b` 转换为多项式后结果等于 `Poly(2*y + 3, domain='QQ')`
    assert b.as_poly() == Poly(2*y + 3, domain='QQ')
    
    # 断言 `a` 转换为表达式结果等于 `2*sqrt(2) + 3`
    assert a.as_expr() == 2*sqrt(2) + 3
    # 断言 `a` 转换为表达式（带变量 `x`）结果等于 `2*x + 3`
    assert a.as_expr(x) == 2*x + 3
    # 断言 `b` 转换为表达式结果等于 `2*sqrt(2) + 3`
    assert b.as_expr() == 2*sqrt(2) + 3
    # 断言 `b` 转换为表达式（带变量 `x`）结果等于 `2*x + 3`
    assert b.as_expr(x) == 2*x + 3
    
    # 使用平方根 2 创建代数数域对象 `a`
    a = AlgebraicNumber(sqrt(2))
    # 使用平方根 2 创建到数域的转换 `b`
    b = to_number_field(sqrt(2))
    # 断言 `a` 和 `b` 的参数都等于 `(sqrt(2), Tuple(1, 0))`
    assert a.args == b.args == (sqrt(2), Tuple(1, 0))
    
    # 使用平方根 2 和系数 `[1, 2, 3]` 创建代数数域对象 `a`
    a = AlgebraicNumber(sqrt(2), [1, 2, 3])
    # 断言 `a` 的参数等于 `(sqrt(2), Tuple(1, 2, 3))`
    assert a.args == (sqrt(2), Tuple(1, 2, 3))
    
    # 使用平方根 2 和系数 `[1, 2]` 以及别名 `"alpha"` 创建代数数域对象 `a`
    a = AlgebraicNumber(sqrt(2), [1, 2], "alpha")
    # 使用 `a` 初始化代数数域对象 `b`
    b = AlgebraicNumber(a)
    # 使用 `a` 和别名 `"gamma"` 初始化代数数域对象 `c`
    c = AlgebraicNumber(a, alias="gamma")
    
    # 断言 `a` 等于 `b`
    assert a == b
    # 断言 `c` 的别名为 `"gamma"`
    assert c.alias.name == "gamma"
    
    # 使用 `sqrt(2) + sqrt(3)` 和系数 `[S(1)/2, 0, S(-9)/2, 0]` 创建代数数域对象 `a`
    a = AlgebraicNumber(sqrt(2) + sqrt(3), [S(1)/2, 0, S(-9)/2, 0])
    # 使用 `a` 和系数 `[1, 0, 0]` 初始化代数数域对象 `b`
    b = AlgebraicNumber(a, [1, 0, 0])
    
    # 断言 `b` 的根与 `a` 的根相同
    assert b.root == a.root
    # 断言 `a` 转换为根的结果等于 `sqrt(2)`
    assert a.to_root() == sqrt(2)
    # 断言 `b` 转换为根的结果等于 `2`
    assert b.to_root() == 2
    
    # 使用整数 `2` 创建代数数域对象 `a`
    a = AlgebraicNumber(2)
    
    # 断言 `a` 是原始元素
    assert a.is_primitive_element is True
def test_to_algebraic_integer():
    # 创建代数数对象，使用根号3作为根，并指定生成元为x，转换为代数整数形式
    a = AlgebraicNumber(sqrt(3), gen=x).to_algebraic_integer()

    # 断言：最小多项式应为x^2 - 3
    assert a.minpoly == x**2 - 3
    # 断言：代数数的根应为sqrt(3)
    assert a.root == sqrt(3)
    # 断言：代数数的表示应为DMP([QQ(1), QQ(0)], QQ)
    assert a.rep == DMP([QQ(1), QQ(0)], QQ)

    # 创建代数数对象，使用2*sqrt(3)作为根，并指定生成元为x，转换为代数整数形式
    a = AlgebraicNumber(2*sqrt(3), gen=x).to_algebraic_integer()
    # 断言：最小多项式应为x^2 - 12
    assert a.minpoly == x**2 - 12
    # 断言：代数数的根应为2*sqrt(3)
    assert a.root == 2*sqrt(3)
    # 断言：代数数的表示应为DMP([QQ(1), QQ(0)], QQ)
    assert a.rep == DMP([QQ(1), QQ(0)], QQ)

    # 创建代数数对象，使用sqrt(3)/2作为根，并指定生成元为x，转换为代数整数形式
    a = AlgebraicNumber(sqrt(3)/2, gen=x).to_algebraic_integer()
    # 断言：最小多项式应为x^2 - 12
    assert a.minpoly == x**2 - 12
    # 断言：代数数的根应为2*sqrt(3)
    assert a.root == 2*sqrt(3)
    # 断言：代数数的表示应为DMP([QQ(1), QQ(0)], QQ)
    assert a.rep == DMP([QQ(1), QQ(0)], QQ)

    # 创建代数数对象，使用sqrt(3)/2作为根，系数为[7/19, 3]，生成元为x，转换为代数整数形式
    a = AlgebraicNumber(sqrt(3)/2, [Rational(7, 19), 3], gen=x).to_algebraic_integer()
    # 断言：最小多项式应为x^2 - 12
    assert a.minpoly == x**2 - 12
    # 断言：代数数的根应为2*sqrt(3)
    assert a.root == 2*sqrt(3)
    # 断言：代数数的表示应为DMP([QQ(7, 19), QQ(3)], QQ)
    assert a.rep == DMP([QQ(7, 19), QQ(3)], QQ)


def test_AlgebraicNumber_to_root():
    # 断言：对于根号2的代数数对象，转换为根应为sqrt(2)
    assert AlgebraicNumber(sqrt(2)).to_root() == sqrt(2)

    # 创建代数数对象，使用x^5 - 1的第4个根作为根，系数为[1, 0, 0]，转换为根应为x^4 + x^3 + x^2 + x + 1的第1个根
    zeta5_squared = AlgebraicNumber(CRootOf(x**5 - 1, 4), coeffs=[1, 0, 0])
    assert zeta5_squared.to_root() == CRootOf(x**4 + x**3 + x**2 + x + 1, 1)

    # 创建代数数对象，使用x^3 - 1的第2个根作为根，系数为[1, 0, 0]，转换为根应为-x/2 - sqrt(3)*I/2
    zeta3_squared = AlgebraicNumber(CRootOf(x**3 - 1, 2), coeffs=[1, 0, 0])
    assert zeta3_squared.to_root() == -S(1)/2 - sqrt(3)*I/2
    # 断言：关闭根式化选项后，转换为根应为x^2 + x + 1的第0个根
    assert zeta3_squared.to_root(radicals=False) == CRootOf(x**2 + x + 1, 0)
```