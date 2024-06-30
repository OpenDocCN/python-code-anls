# `D:\src\scipysrc\sympy\sympy\functions\special\tests\test_singularity_functions.py`

```
# 导入必要的符号和函数模块
from sympy.core.function import (Derivative, diff)
from sympy.core.numbers import (Float, I, nan, oo, pi)
from sympy.core.relational import Eq
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.delta_functions import (DiracDelta, Heaviside)
from sympy.functions.special.singularity_functions import SingularityFunction
from sympy.series.order import O

# 导入额外的符号和函数模块
from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
from sympy.testing.pytest import raises

# 定义符号变量
x, y, a, n = symbols('x y a n')

# 定义测试函数 test_fdiff，测试 SingularityFunction 类的 fdiff 方法
def test_fdiff():
    # 测试 SingularityFunction 对象的 fdiff 方法
    assert SingularityFunction(x, 4, 5).fdiff() == 5*SingularityFunction(x, 4, 4)
    assert SingularityFunction(x, 4, -1).fdiff() == SingularityFunction(x, 4, -2)
    assert SingularityFunction(x, 4, -2).fdiff() == SingularityFunction(x, 4, -3)
    assert SingularityFunction(x, 4, -3).fdiff() == SingularityFunction(x, 4, -4)
    assert SingularityFunction(x, 4, 0).fdiff() == SingularityFunction(x, 4, -1)

    # 测试 SingularityFunction 对象的 diff 方法
    assert SingularityFunction(y, 6, 2).diff(y) == 2*SingularityFunction(y, 6, 1)
    assert SingularityFunction(y, -4, -1).diff(y) == SingularityFunction(y, -4, -2)
    assert SingularityFunction(y, 4, 0).diff(y) == SingularityFunction(y, 4, -1)
    assert SingularityFunction(y, 4, 0).diff(y, 2) == SingularityFunction(y, 4, -2)

    # 使用符号 n 的正数特性，测试 SingularityFunction 对象的 fdiff 方法
    n = Symbol('n', positive=True)
    assert SingularityFunction(x, a, n).fdiff() == n*SingularityFunction(x, a, n - 1)
    assert SingularityFunction(y, a, n).diff(y) == n*SingularityFunction(y, a, n - 1)

    # 测试 diff 函数对包含 SingularityFunction 对象的表达式的处理
    expr_in = 4*SingularityFunction(x, a, n) + 3*SingularityFunction(x, a, -1) + -10*SingularityFunction(x, a, 0)
    expr_out = n*4*SingularityFunction(x, a, n - 1) + 3*SingularityFunction(x, a, -2) - 10*SingularityFunction(x, a, -1)
    assert diff(expr_in, x) == expr_out

    # 测试 diff 函数对 SingularityFunction 对象的 evaluate 参数的处理
    assert SingularityFunction(x, -10, 5).diff(evaluate=False) == (
        Derivative(SingularityFunction(x, -10, 5), x))

    # 测试使用 fdiff 方法时的参数错误处理
    raises(ArgumentIndexError, lambda: SingularityFunction(x, 4, 5).fdiff(2))

# 定义测试函数 test_eval，测试 SingularityFunction 类的 eval 方法
def test_eval():
    # 测试 SingularityFunction 对象的 func 属性
    assert SingularityFunction(x, a, n).func == SingularityFunction
    # 测试 unchanged 函数的使用
    assert unchanged(SingularityFunction, x, 5, n)
    # 测试 SingularityFunction 对象的特定实例
    assert SingularityFunction(5, 3, 2) == 4
    assert SingularityFunction(3, 5, 1) == 0
    assert SingularityFunction(3, 3, 0) == 1
    assert SingularityFunction(3, 3, 1) == 0
    # 测试对零符号值的处理
    assert SingularityFunction(Symbol('z', zero=True), 0, 1) == 0  # like sin(z) == 0
    # 测试无穷大的特定实例
    assert SingularityFunction(4, 4, -1) is oo
    assert SingularityFunction(4, 2, -1) == 0
    assert SingularityFunction(4, 7, -1) == 0
    assert SingularityFunction(5, 6, -2) == 0
    assert SingularityFunction(4, 2, -2) is oo
    assert SingularityFunction(4, 2, -3) == 0
    assert SingularityFunction(8, 8, -3) is oo
    assert SingularityFunction(4, 2, -4) == 0
    assert SingularityFunction(8, 8, -4) is oo
    # 测试 evalf 方法
    assert (SingularityFunction(6.1, 4, 5)).evalf(5) == Float('40.841', '5')
    # 测试 SingularityFunction 函数，验证其对于特定参数的行为是否符合预期
    
    # 断言：验证 SingularityFunction(6.1, pi, 2) 的返回值是否等于 (-pi + 6.1)**2
    assert SingularityFunction(6.1, pi, 2) == (-pi + 6.1)**2
    
    # 断言：验证 SingularityFunction(x, a, nan) 的返回值是否是 nan
    assert SingularityFunction(x, a, nan) is nan
    
    # 断言：验证 SingularityFunction(x, nan, 1) 的返回值是否是 nan
    assert SingularityFunction(x, nan, 1) is nan
    
    # 断言：验证 SingularityFunction(nan, a, n) 的返回值是否是 nan
    assert SingularityFunction(nan, a, n) is nan
    
    # 引发 ValueError 异常，检验 SingularityFunction 函数在给定参数下是否会引发异常
    raises(ValueError, lambda: SingularityFunction(x, a, I))
    
    # 引发 ValueError 异常，检验 SingularityFunction 函数在给定参数下是否会引发异常
    raises(ValueError, lambda: SingularityFunction(2*I, I, n))
    
    # 引发 ValueError 异常，检验 SingularityFunction 函数在给定参数下是否会引发异常
    raises(ValueError, lambda: SingularityFunction(x, a, -5))
#`
# 定义测试函数 test_leading_term，用于测试 SingularityFunction 类的 as_leading_term 方法
def test_leading_term():
    # 定义符号变量 l，并指定其为正数
    l = Symbol('l', positive=True)
    # 断言：SingularityFunction(x, 3, 2).as_leading_term(x) 的返回值应为 0
    assert SingularityFunction(x, 3, 2).as_leading_term(x) == 0
    # 断言：SingularityFunction(x, -2, 1).as_leading_term(x) 的返回值应为 2
    assert SingularityFunction(x, -2, 1).as_leading_term(x) == 2
    # 断言：SingularityFunction(x, 0, 0).as_leading_term(x) 的返回值应为 1
    assert SingularityFunction(x, 0, 0).as_leading_term(x) == 1
    # 断言：SingularityFunction(x, 0, 0).as_leading_term(x, cdir=-1) 的返回值应为 0
    assert SingularityFunction(x, 0, 0).as_leading_term(x, cdir=-1) == 0
    # 断言：SingularityFunction(x, 0, -1).as_leading_term(x) 的返回值应为 0
    assert SingularityFunction(x, 0, -1).as_leading_term(x) == 0
    # 断言：SingularityFunction(x, 0, -2).as_leading_term(x) 的返回值应为 0
    assert SingularityFunction(x, 0, -2).as_leading_term(x) == 0
    # 断言：SingularityFunction(x, 0, -3).as_leading_term(x) 的返回值应为 0
    assert SingularityFunction(x, 0, -3).as_leading_term(x) == 0
    # 断言：SingularityFunction(x, 0, -4).as_leading_term(x) 的返回值应为 0
    assert SingularityFunction(x, 0, -4).as_leading_term(x) == 0
    # 断言：(SingularityFunction(x + l, 0, 1)/2 - SingularityFunction(x + l, l/2, 1) + SingularityFunction(x + l, l, 1)/2).as_leading_term(x) 的返回值应为 -x/2
    assert (SingularityFunction(x + l, 0, 1)/2 \
        - SingularityFunction(x + l, l/2, 1) \
        + SingularityFunction(x + l, l, 1)/2).as_leading_term(x) == -x/2


# 定义测试函数 test_series，用于测试 SingularityFunction 类的 series 方法
def test_series():
    # 定义符号变量 l，并指定其为正数
    l = Symbol('l', positive=True)
    # 断言：SingularityFunction(x, -3, 2).series(x) 的返回值应为 x**2 + 6*x + 9
    assert SingularityFunction(x, -3, 2).series(x) == x**2 + 6*x + 9
    # 断言：SingularityFunction(x, -2, 1).series(x) 的返回值应为 x + 2
    assert SingularityFunction(x, -2, 1).series(x) == x + 2
    # 断言：SingularityFunction(x, 0, 0).series(x) 的返回值应为 1
    assert SingularityFunction(x, 0, 0).series(x) == 1
    # 断言：SingularityFunction(x, 0, 0).series(x, dir='-') 的返回值应为 0
    assert SingularityFunction(x, 0, 0).series(x, dir='-') == 0
    # 断言：SingularityFunction(x, 0, -1).series(x) 的返回值应为 0
    assert SingularityFunction(x, 0, -1).series(x) == 0
    # 断言：SingularityFunction(x, 0, -2).series(x) 的返回值应为 0
    assert SingularityFunction(x, 0, -2).series(x) == 0
    # 断言：SingularityFunction(x, 0, -3).series(x) 的返回值应为 0
    assert SingularityFunction(x, 0, -3).series(x) == 0
    # 断言：SingularityFunction(x, 0, -4).series(x) 的返回值应为 0
    assert SingularityFunction(x, 0, -4).series(x) == 0
    # 断言：(SingularityFunction(x + l, 0, 1)/2 - SingularityFunction(x + l, l/2, 1) + SingularityFunction(x + l, l, 1)/2).nseries(x) 的返回值应为 -x/2 + O(x**6)
    assert (SingularityFunction(x + l, 0, 1)/2 \
        - SingularityFunction(x + l, l/2, 1) \
        + SingularityFunction(x + l, l, 1)/2).nseries(x) == -x/2 + O(x**6)


# 定义测试函数 test_rewrite，用于测试 SingularityFunction 类的 rewrite 方法
def test_rewrite():
    # 断言：SingularityFunction(x, 4, 5).rewrite(Piecewise) 的返回值应为 Piecewise(((x - 4)**5, x - 4 >= 0), (0, True))
    assert SingularityFunction(x, 4, 5).rewrite(Piecewise) == (
        Piecewise(((x - 4)**5, x - 4 >= 0), (0, True)))
    # 断言：SingularityFunction(x, -10, 0).rewrite(Piecewise) 的返回值应为 Piecewise((1, x + 10 >= 0), (0, True))
    assert SingularityFunction(x, -10, 0).rewrite(Piecewise) == (
        Piecewise((1, x + 10 >= 0), (0, True)))
    # 断言：SingularityFunction(x, 2, -1).rewrite(Piecewise) 的返回值应为 Piecewise((oo, Eq(x - 2, 0)), (0, True))
    assert SingularityFunction(x, 2, -1).rewrite(Piecewise) == (
        Piecewise((oo, Eq(x - 2, 0)), (0, True)))
    # 断言：SingularityFunction(x, 0, -2).rewrite(Piecewise) 的返回值应为 Piecewise((oo, Eq(x, 0)), (0, True))
    assert SingularityFunction(x, 0, -2).rewrite(Piecewise) == (
        Piecewise((oo, Eq(x, 0)), (0, True)))

    # 定义符号变量 n，并指定其为非负数
    n = Symbol('n', nonnegative=True)
    # 使用符号变量 a 和 n 调用 rewrite 方法并断言其返回值
    p = SingularityFunction(x, a, n).rewrite(Piecewise)
    assert p == (
        Piecewise(((x - a)**n, x - a >= 0), (0, True)))
    # 对 rewrite 方法返回的结果进行符号替换，并断言结果
    assert p.subs(x, a).subs(n, 0) == 1

    # 定义符号表达式 expr_in 和 expr_out，分别使用 rewrite 方法将其转换为 Heaviside 和 DiracDelta 形式，并进行断言
    expr_in = SingularityFunction(x, 4, 5) + SingularityFunction(x, -3, -1) - SingularityFunction(x, 0, -2)
    expr_out = (x - 4)**5*Heaviside(x - 4, 1) + DiracDelta(x + 3) - DiracDelta(x, 1)
    assert expr_in.rewrite(Heaviside) == expr_out
    assert expr_in.rewrite(DiracDelta) == expr_out
    assert expr_in.rewrite('HeavisideDiracDelta') == expr_out

    # 定义符号表达式 expr_in 和 expr_out，分别使用 rewrite 方法将其转换为 Heaviside 和 DiracDelta 形式，并进行断言
    expr_in = SingularityFunction(x, a, n) + SingularityFunction(x, a, -1) - SingularityFunction(x, a, -2)
    expr_out = (x - a)**n*Heaviside(x - a, 1) + DiracDelta(x - a) + DiracDelta(a - x, 1)
    assert expr_in.rewrite(Heaviside) == expr_out
    assert expr_in.rewrite(DiracDelta) ==```
# 定义一个函数用于测试 SingularityFunction 类的 as_leading_term 方法
def test_leading_term():
    # 创建一个正的符号 l
    l = Symbol('l', positive=True)
    # 断言：SingularityFunction(x, 3, 2) 的主导项在变量 x 上等于 0
    assert SingularityFunction(x, 3, 2).as_leading_term(x) == 0
    # 断言：SingularityFunction(x, -2, 1) 的主导项在变量 x 上等于 2
    assert SingularityFunction(x, -2, 1).as_leading_term(x) == 2
    # 断言：SingularityFunction(x, 0, 0) 的主导项在变量 x 上等于 1
    assert SingularityFunction(x, 0, 0).as_leading_term(x) == 1
    # 断言：SingularityFunction(x, 0, 0) 的主导项在变量 x 上，指定 cdir=-1 时等于 0
    assert SingularityFunction(x, 0, 0).as_leading_term(x, cdir=-1) == 0
    # 断言：SingularityFunction(x, 0, -1) 的主导项在变量 x 上等于 0
    assert SingularityFunction(x, 0, -1).as_leading_term(x) == 0
    # 断言：SingularityFunction(x, 0, -2) 的主导项在变量 x 上等于 0
    assert SingularityFunction(x, 0, -2).as_leading_term(x) == 0
    # 断言：SingularityFunction(x, 0, -3) 的主导项在变量 x 上等于 0
    assert SingularityFunction(x, 0, -3).as_leading_term(x) == 0
    # 断言：SingularityFunction(x, 0, -4) 的主导项在变量 x 上等于 0
    assert SingularityFunction(x, 0, -4).as_leading_term(x) == 0
    # 断言：表达式的主导项，其中包含多项 SingularityFunction 的和与差，求其在变量 x 上的主导项
    assert (SingularityFunction(x + l, 0, 1)/2\
        - SingularityFunction(x + l, l/2, 1)\
        + SingularityFunction(x + l, l, 1)/2).as_leading_term(x) == -x/2


# 定义一个函数用于测试 SingularityFunction 类的 series 方法
def test_series():
    # 创建一个正的符号 l
    l = Symbol('l', positive=True)
    # 断言：SingularityFunction(x, -3, 2) 的级数展开在变量 x 上等于 x^2 + 6*x + 9
    assert SingularityFunction(x, -3, 2).series(x) == x**2 + 6*x + 9
    # 断言：SingularityFunction(x, -2, 1) 的级数展开在变量 x 上等于 x + 2
    assert SingularityFunction(x, -2, 1).series(x) == x + 2
    # 断言：SingularityFunction(x, 0, 0) 的级数展开在变量 x 上等于 1
    assert SingularityFunction(x, 0, 0).series(x) == 1
    # 断言：SingularityFunction(x, 0, 0) 的级数展开在变量 x 上，方向为 '-' 时等于 0
    assert SingularityFunction(x, 0, 0).series(x, dir='-') == 0
    # 断言：SingularityFunction(x, 0, -1) 的级数展开在变量 x 上等于 0
    assert SingularityFunction(x, 0, -1).series(x) == 0
    # 断言：SingularityFunction(x, 0, -2) 的级数展开在变量 x 上等于 0
    assert SingularityFunction(x, 0, -2).series(x) == 0
    # 断言：SingularityFunction(x, 0, -3) 的级数展开在变量 x 上等于 0
    assert SingularityFunction(x, 0, -3).series(x) == 0
    # 断言：SingularityFunction(x, 0, -4) 的级数展开在变量 x 上等于 0
    assert SingularityFunction(x, 0, -4).series(x) == 0
    # 断言：表达式的级数展开，其中包含多项 SingularityFunction 的和与差，求其在变量 x 上的级数展开，并展示到 O(x**6) 阶
    assert (SingularityFunction(x + l, 0, 1)/2\
        - SingularityFunction(x + l, l/2, 1)\
        + SingularityFunction(x + l, l, 1)/2).nseries(x) == -x/2 + O(x**6)


# 定义一个函数用于测试 SingularityFunction 类的 rewrite 方法
def test_rewrite():
    # 断言：将 SingularityFunction(x, 4, 5) 重写为 Piecewise 形式，应为 Piecewise(((x - 4)**5, x - 4 >= 0), (0, True))
    assert SingularityFunction(x, 4, 5).rewrite(Piecewise) == (
        Piecewise(((x - 4)**5, x - 4 >= 0), (0, True)))
    # 断言：将 SingularityFunction(x, -10, 0) 重写为 Piecewise 形式，应为 Piecewise((1, x + 10 >= 0), (0, True))
    assert SingularityFunction(x, -10, 0).rewrite(Piecewise) == (
        Piecewise((1, x + 10 >= 0), (0, True)))
    # 断言：将 SingularityFunction(x, 2, -1) 重写为 Piecewise 形式，应为 Piecewise((oo, Eq(x - 2, 0)), (0, True))
    assert SingularityFunction(x, 2, -1).rewrite(Piecewise) == (
        Piecewise((oo, Eq(x - 2, 0)), (0, True)))
    # 断言：将 SingularityFunction(x, 0, -2) 重写为 Piecewise 形式，应为 Piecewise((oo, Eq(x, 0)), (0, True))
    assert SingularityFunction(x, 0, -2).rewrite(Piecewise) == (
        Piecewise((oo, Eq(x, 0)), (0, True)))

    # 创建一个非负的符号 n
    n = Symbol('n', nonnegative=True)
    # 使用符号 a 和 n 创建一个 Piecewise 形式的表达式，然后断言其结果
    p = SingularityFunction(x, a, n).rewrite(Piecewise)
    assert p == (
        Piecewise(((x - a)**n, x - a >= 0), (0, True)))
    # 断言：将表达式 p 中的 x 替换为 a，并将 n 替换为 0，结果应为 1
    assert p.subs(x, a).subs(n, 0) == 1

    # 创建一个包含多项 SingularityFunction 的表达式 expr_in
    expr_in = SingularityFunction(x, 4, 5) + SingularityFunction(x, -3, -1) - SingularityFunction(x, 0, -2)
    # 创建一个预期的表达式 expr_out，包含相应的 Heaviside 和 DiracDelta 函数
    expr_out = (```
# 定义一个函数用于测试 SingularityFunction 类的 as_leading_term 方法
def test_leading_term():
    # 创建一个正的符号 l
    l = Symbol('l', positive=True)
    # 断言：SingularityFunction(x, 3, 2) 的主导项在变量 x 上等于 0
    assert SingularityFunction(x, 3, 2).as_leading_term(x) == 0
    # 断言：SingularityFunction(x, -2, 1) 的主导项在变量 x 上等于 2
    assert SingularityFunction(x, -2, 1).as_leading_term(x) == 2
    # 断言：SingularityFunction(x, 0, 0) 的主导项在变量 x 上等于 1
    assert SingularityFunction(x, 0, 0).as_leading_term(x) == 1
    # 断言：SingularityFunction(x, 0, 0) 的主导项在变量 x 上，指定 cdir=-1 时等于 0
    assert SingularityFunction(x, 0, 0).as_leading_term(x, cdir=-1) == 0
    # 断言：SingularityFunction(x, 0, -1) 的主导项在变量 x 上等于 0
    assert SingularityFunction(x, 0, -1).as_leading_term(x) == 0
    # 断言：SingularityFunction(x, 0, -2) 的主导项在变量 x 上等于 0
    assert SingularityFunction(x, 0, -2).as_leading_term(x) == 0
    # 断言：SingularityFunction(x, 0, -3) 的主导项在变量 x 上等于 0
    assert SingularityFunction(x, 0, -3).as_leading_term(x) == 0
    # 断言：SingularityFunction(x, 0, -4) 的主导项在变量 x 上等于 0
    assert SingularityFunction(x, 0, -4).as_leading_term(x) == 0
    # 断言：表达式的主导项，其中包含多项 SingularityFunction 的和与差，求其在变量 x 上的主导项
    assert (SingularityFunction(x + l, 0, 1)/2\
        - SingularityFunction(x + l, l/2, 1)\
        + SingularityFunction(x + l, l, 1)/2).as_leading_term(x) == -x/2


# 定义一个函数用于测试 SingularityFunction 类的 series 方法
def test_series():
    # 创建一个正的符号 l
    l = Symbol('l', positive=True)
    # 断言：SingularityFunction(x, -3, 2) 的级数展开在变量 x 上等于 x^2 + 6*x + 9
    assert SingularityFunction(x, -3, 2).series(x) == x**2 + 6*x + 9
    # 断言：SingularityFunction(x, -2, 1) 的级数展开在变量 x 上等于 x + 2
    assert SingularityFunction(x, -2, 1).series(x) == x + 2
    # 断言：SingularityFunction(x, 0, 0) 的级数展开在变量 x 上等于 1
    assert SingularityFunction(x, 0, 0).series(x) == 1
    # 断言：SingularityFunction(x, 0, 0) 的级数展开在变量 x 上，方向为 '-' 时等于 0
    assert SingularityFunction(x, 0, 0).series(x, dir='-') == 0
    # 断言：SingularityFunction(x, 0, -1) 的级数展开在变量 x 上等于 0
    assert SingularityFunction(x, 0, -1).series(x) == 0
    # 断言：SingularityFunction(x, 0, -2) 的级数展开在变量 x 上等于 0
    assert SingularityFunction(x, 0, -2).series(x) == 0
    # 断言：SingularityFunction(x, 0, -3) 的级数展开在变量 x 上等于 0
    assert SingularityFunction(x, 0, -3).series(x) == 0
    # 断言：SingularityFunction(x, 0, -4) 的级数展开在变量 x 上等于 0
    assert SingularityFunction(x, 0, -4).series(x) == 0
    # 断言：表达式的级数展开，其中包含多项 SingularityFunction 的和与差，求其在变量 x 上的级数展开，并展示到 O(x**6) 阶
    assert (SingularityFunction(x + l, 0, 1)/2\
        - SingularityFunction(x + l, l/2, 1)\
        + SingularityFunction(x + l, l, 1)/2).nseries(x) == -x/2 + O(x**6)


# 定义一个函数用于测试 SingularityFunction 类的 rewrite 方法
def test_rewrite():
    # 断言：将 SingularityFunction(x, 4, 5) 重写为 Piecewise 形式，应为 Piecewise(((x - 4)**5, x - 4 >= 0), (0, True))
    assert SingularityFunction(x, 4, 5).rewrite(Piecewise) == (
        Piecewise(((x - 4)**5, x - 4 >= 0), (0, True)))
    # 断言：将 SingularityFunction(x, -10, 0) 重写为 Piecewise 形式，应为 Piecewise((1, x + 10 >= 0), (0, True))
    assert SingularityFunction(x, -10, 0).rewrite(Piecewise) == (
        Piecewise((1, x + 10 >= 0), (0, True)))
    # 断言：将 SingularityFunction(x, 2, -1) 重写为 Piecewise 形式，应为 Piecewise((oo, Eq(x - 2, 0)), (0, True))
    assert SingularityFunction(x, 2, -1).rewrite(Piecewise) == (
        Piecewise((oo, Eq(x - 2, 0)), (0, True)))
    # 断言：将 SingularityFunction(x, 0, -2) 重写为 Piecewise 形式，应为 Piecewise((oo, Eq(x, 0)), (0, True))
    assert SingularityFunction(x, 0, -2).rewrite(Piecewise) == (
        Piecewise((oo, Eq(x, 0)), (0, True)))

    # 创建一个非负的符号 n
    n = Symbol('n', nonnegative=True)
    # 使用符号 a 和 n 创建一个 Piecewise 形式的表达式，然后断言其结果
    p = SingularityFunction(x, a, n).rewrite(Piecewise)
    assert p == (
        Piecewise(((x - a)**n, x - a >= 0), (0, True)))
    # 断言：将表达式 p 中的 x 替换为 a，并将 n 替换为 0，结果应为 1
    assert p.subs(x, a).subs(n, 0) == 1

    # 创建一个包含多项 SingularityFunction 的表达式 expr_in
    expr_in = SingularityFunction(x, 4, 5) + SingularityFunction(x, -3, -1) - SingularityFunction(x, 0, -2)
    # 创建一个预期的表达式 expr_out，包含相应的 Heaviside 和 DiracDelta 函数
    expr_out = (x - 4)**5*Heaviside(x - 4, 1) + DiracDelta(x + 3) - DiracDelta(x, 1)
    # 断言：将表达式 expr_in 重写为 Heaviside 形式，结果应为 expr_out
    assert expr_in.rewrite(Heaviside) == expr_out
    # 断言：将表达式 expr_in 重写为 DiracDelta 形式，结果应为 expr_out
    assert expr_in.rewrite(DiracDelta) == expr_out
    # 断言：将表达式 expr_in 重写为 'HeavisideDiracDelta' 形式，结果应为 expr_out
    assert expr_in.rewrite('HeavisideDiracDelta') == expr_out

    # 创建一个包含多项 SingularityFunction 的表达式 expr_in
    expr_in = SingularityFunction(x, a, n) + SingularityFunction(x, a, -1) - SingularityFunction(x, a, -2)
```