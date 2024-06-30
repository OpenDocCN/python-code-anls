# `D:\src\scipysrc\sympy\sympy\functions\special\tests\test_delta_functions.py`

```
# 从 sympy.core.numbers 模块导入指定的符号常量和数学常数
from sympy.core.numbers import (I, nan, oo, pi)
# 从 sympy.core.relational 模块导入等式和不等式相关的类
from sympy.core.relational import (Eq, Ne)
# 从 sympy.core.singleton 模块导入符号常量 S
from sympy.core.singleton import S
# 从 sympy.core.symbol 模块导入符号变量相关的类和函数
from sympy.core.symbol import (Symbol, symbols)
# 从 sympy.functions.elementary.complexes 模块导入复数相关的函数
from sympy.functions.elementary.complexes import (adjoint, conjugate, sign, transpose)
# 从 sympy.functions.elementary.miscellaneous 模块导入平方根函数 sqrt
from sympy.functions.elementary.miscellaneous import sqrt
# 从 sympy.functions.elementary.piecewise 模块导入分段函数 Piecewise
from sympy.functions.elementary.piecewise import Piecewise
# 从 sympy.functions.special.delta_functions 模块导入 Delta 函数和 Heaviside 函数
from sympy.functions.special.delta_functions import (DiracDelta, Heaviside)
# 从 sympy.functions.special.singularity_functions 模块导入 SingularityFunction 函数
from sympy.functions.special.singularity_functions import SingularityFunction
# 从 sympy.simplify.simplify 模块导入符号简化函数 signsimp
from sympy.simplify.simplify import signsimp

# 从 sympy.testing.pytest 模块导入 raises 函数
from sympy.testing.pytest import raises

# 从 sympy.core.expr 模块导入 unchanged 函数
from sympy.core.expr import unchanged

# 从 sympy.core.function 模块导入 ArgumentIndexError 异常类
from sympy.core.function import ArgumentIndexError

# 定义符号变量 x 和 y
x, y = symbols('x y')
# 定义带特定条件的符号变量 i, j, k
i = symbols('t', nonzero=True)
j = symbols('j', positive=True)
k = symbols('k', negative=True)

# 定义测试函数 test_DiracDelta
def test_DiracDelta():
    # 检查 Delta 函数在不同参数下的值是否符合预期
    assert DiracDelta(1) == 0
    assert DiracDelta(5.1) == 0
    assert DiracDelta(-pi) == 0
    assert DiracDelta(5, 7) == 0
    assert DiracDelta(x, 0) == DiracDelta(x)
    assert DiracDelta(i) == 0
    assert DiracDelta(j) == 0
    assert DiracDelta(k) == 0
    assert DiracDelta(nan) is nan
    assert DiracDelta(0).func is DiracDelta
    assert DiracDelta(x).func is DiracDelta

    # FIXME: 在 x=0 时通常是未定义的
    #         但 limit(Delta(c)*Heaviside(x),x,-oo)
    #         需要被实现。
    # assert 0*DiracDelta(x) == 0

    # 检查 Delta 函数的共轭、伴随、转置性质是否正确
    assert adjoint(DiracDelta(x)) == DiracDelta(x)
    assert adjoint(DiracDelta(x - y)) == DiracDelta(x - y)
    assert conjugate(DiracDelta(x)) == DiracDelta(x)
    assert conjugate(DiracDelta(x - y)) == DiracDelta(x - y)
    assert transpose(DiracDelta(x)) == DiracDelta(x)
    assert transpose(DiracDelta(x - y)) == DiracDelta(x - y)

    # 检查 Delta 函数的导数计算是否正确
    assert DiracDelta(x).diff(x) == DiracDelta(x, 1)
    assert DiracDelta(x, 1).diff(x) == DiracDelta(x, 2)

    # 检查 Delta 函数的简单性质
    assert DiracDelta(x).is_simple(x) is True
    assert DiracDelta(3*x).is_simple(x) is True
    assert DiracDelta(x**2).is_simple(x) is False
    assert DiracDelta(sqrt(x)).is_simple(x) is False
    assert DiracDelta(x).is_simple(y) is False

    # 检查 Delta 函数的展开性质
    assert DiracDelta(x*y).expand(diracdelta=True, wrt=x) == DiracDelta(x)/abs(y)
    assert DiracDelta(x*y).expand(diracdelta=True, wrt=y) == DiracDelta(y)/abs(x)
    assert DiracDelta(x**2*y).expand(diracdelta=True, wrt=x) == DiracDelta(x**2*y)
    assert DiracDelta(y).expand(diracdelta=True, wrt=x) == DiracDelta(y)
    assert DiracDelta((x - 1)*(x - 2)*(x - 3)).expand(diracdelta=True, wrt=x) == (
        DiracDelta(x - 3)/2 + DiracDelta(x - 2) + DiracDelta(x - 1)/2)

    # 检查 Delta 函数的尺度性质
    assert DiracDelta(2*x) != DiracDelta(x)  # scaling property

    # 检查 Delta 函数的偶函数性质
    assert DiracDelta(x) == DiracDelta(-x)  # even function

    # 检查 Delta 函数的奇导数性质
    assert DiracDelta(-x, 2) == DiracDelta(x, 2)
    assert DiracDelta(-x, 1) == -DiracDelta(x, 1)  # odd deriv is odd

    # 检查 Delta 函数在特定极限情况下的性质
    assert DiracDelta(-oo*x) == DiracDelta(oo*x)
    assert DiracDelta(x - y) != DiracDelta(y - x)

    # 使用符号简化函数检查 Delta 函数的差值是否为零
    assert signsimp(DiracDelta(x - y) - DiracDelta(y - x)) == 0
    # 断言：对于 DiracDelta 函数的展开，检查在特定条件下是否成立
    assert DiracDelta(x*y).expand(diracdelta=True, wrt=x) == DiracDelta(x)/abs(y)
    
    # 断言：对于 DiracDelta 函数的展开，检查在特定条件下是否成立
    assert DiracDelta(x*y).expand(diracdelta=True, wrt=y) == DiracDelta(y)/abs(x)
    
    # 断言：对于 DiracDelta 函数的展开，检查在特定条件下是否成立
    assert DiracDelta(x**2*y).expand(diracdelta=True, wrt=x) == DiracDelta(x**2*y)
    
    # 断言：对于 DiracDelta 函数的展开，检查在特定条件下是否成立
    assert DiracDelta(y).expand(diracdelta=True, wrt=x) == DiracDelta(y)
    
    # 断言：对于 DiracDelta 函数的展开，检查在特定条件下是否成立
    assert DiracDelta((x - 1)*(x - 2)*(x - 3)).expand(diracdelta=True) == (
            DiracDelta(x - 3)/2 + DiracDelta(x - 2) + DiracDelta(x - 1)/2)
    
    # 断言：测试 DiracDelta 函数的二阶导数是否引发 ArgumentIndexError 异常
    raises(ArgumentIndexError, lambda: DiracDelta(x).fdiff(2))
    
    # 断言：测试 DiracDelta 函数的构造中是否会引发 ValueError 异常
    raises(ValueError, lambda: DiracDelta(x, -1))
    
    # 断言：测试 DiracDelta 函数构造中使用虚数是否会引发 ValueError 异常
    raises(ValueError, lambda: DiracDelta(I))
    
    # 断言：测试 DiracDelta 函数构造中使用复数是否会引发 ValueError 异常
    raises(ValueError, lambda: DiracDelta(2 + 3*I))
def test_heaviside():
    # 测试 Heaviside 函数在不同输入下的返回值是否符合预期
    assert Heaviside(-5) == 0
    assert Heaviside(1) == 1
    assert Heaviside(0) == S.Half

    # 测试带有变量参数的 Heaviside 函数
    assert Heaviside(0, x) == x
    # 测试带有 nan 参数的 Heaviside 函数
    assert unchanged(Heaviside, x, nan)
    assert Heaviside(0, nan) == nan

    # 创建几个不同参数的 Heaviside 函数实例
    h0  = Heaviside(x, 0)
    h12 = Heaviside(x, S.Half)
    h1  = Heaviside(x, 1)

    # 验证各实例的参数和部分参数是否正确设置
    assert h0.args == h0.pargs == (x, 0)
    assert h1.args == h1.pargs == (x, 1)
    assert h12.args == (x, S.Half)
    assert h12.pargs == (x,) # 默认的 1/2 被省略

    # 验证 Heaviside 函数的伴随、共轭和转置操作
    assert adjoint(Heaviside(x)) == Heaviside(x)
    assert adjoint(Heaviside(x - y)) == Heaviside(x - y)
    assert conjugate(Heaviside(x)) == Heaviside(x)
    assert conjugate(Heaviside(x - y)) == Heaviside(x - y)
    assert transpose(Heaviside(x)) == Heaviside(x)
    assert transpose(Heaviside(x - y)) == Heaviside(x - y)

    # 验证 Heaviside 函数对变量 x 的微分是否为 DiracDelta 函数
    assert Heaviside(x).diff(x) == DiracDelta(x)
    assert Heaviside(x + I).is_Function is True
    assert Heaviside(I*x).is_Function is True

    # 验证 Heaviside 函数在异常情况下的异常处理
    raises(ArgumentIndexError, lambda: Heaviside(x).fdiff(2))
    raises(ValueError, lambda: Heaviside(I))
    raises(ValueError, lambda: Heaviside(2 + 3*I))


def test_rewrite():
    # 定义符号变量 x 和 y，其中 x 是实数，y 是通用符号
    x, y = Symbol('x', real=True), Symbol('y')
    
    # 验证 Heaviside 函数通过 Piecewise 重写的结果是否正确
    assert Heaviside(x).rewrite(Piecewise) == (
        Piecewise((0, x < 0), (Heaviside(0), Eq(x, 0)), (1, True)))
    assert Heaviside(y).rewrite(Piecewise) == (
        Piecewise((0, y < 0), (Heaviside(0), Eq(y, 0)), (1, True)))
    assert Heaviside(x, y).rewrite(Piecewise) == (
        Piecewise((0, x < 0), (y, Eq(x, 0)), (1, True)))
    assert Heaviside(x, 0).rewrite(Piecewise) == (
        Piecewise((0, x <= 0), (1, True)))
    assert Heaviside(x, 1).rewrite(Piecewise) == (
        Piecewise((0, x < 0), (1, True)))
    assert Heaviside(x, nan).rewrite(Piecewise) == (
        Piecewise((0, x < 0), (nan, Eq(x, 0)), (1, True)))

    # 验证 Heaviside 函数通过 sign 函数重写的结果是否正确
    assert Heaviside(x).rewrite(sign) == \
        Heaviside(x, H0=Heaviside(0)).rewrite(sign) == \
        Piecewise(
            (sign(x)/2 + S(1)/2, Eq(Heaviside(0), S(1)/2)),
            (Piecewise(
                (sign(x)/2 + S(1)/2, Ne(x, 0)), (Heaviside(0), True)), True)
        )

    assert Heaviside(y).rewrite(sign) == Heaviside(y)
    assert Heaviside(x, S.Half).rewrite(sign) == (sign(x)+1)/2
    assert Heaviside(x, y).rewrite(sign) == \
        Piecewise(
            (sign(x)/2 + S(1)/2, Eq(y, S(1)/2)),
            (Piecewise(
                (sign(x)/2 + S(1)/2, Ne(x, 0)), (y, True)), True)
        )

    # 验证 DiracDelta 函数通过 Piecewise 重写的结果是否正确
    assert DiracDelta(y).rewrite(Piecewise) == Piecewise((DiracDelta(0), Eq(y, 0)), (0, True))
    assert DiracDelta(y, 1).rewrite(Piecewise) == DiracDelta(y, 1)
    assert DiracDelta(x - 5).rewrite(Piecewise) == (
        Piecewise((DiracDelta(0), Eq(x - 5, 0)), (0, True)))

    # 验证包含 SingularityFunction 的表达式是否重写正确
    assert (x*DiracDelta(x - 10)).rewrite(SingularityFunction) == x*SingularityFunction(x, 10, -1)
    assert 5*x*y*DiracDelta(y, 1).rewrite(SingularityFunction) == 5*x*y*SingularityFunction(y, 0, -2)
    # 断言：使用 DiracDelta(0) 对象调用 rewrite 方法，并期望其结果等于 SingularityFunction(0, 0, -1)
    assert DiracDelta(0).rewrite(SingularityFunction) == SingularityFunction(0, 0, -1)
    
    # 断言：使用 DiracDelta(0, 1) 对象调用 rewrite 方法，并期望其结果等于 SingularityFunction(0, 0, -2)
    assert DiracDelta(0, 1).rewrite(SingularityFunction) == SingularityFunction(0, 0, -2)

    # 断言：使用 Heaviside(x) 对象调用 rewrite 方法，并期望其结果等于 SingularityFunction(x, 0, 0)
    assert Heaviside(x).rewrite(SingularityFunction) == SingularityFunction(x, 0, 0)
    
    # 断言：使用 5*x*y*Heaviside(y + 1) 对象调用 rewrite 方法，并期望其结果等于 5*x*y*SingularityFunction(y, -1, 0)
    assert 5*x*y*Heaviside(y + 1).rewrite(SingularityFunction) == 5*x*y*SingularityFunction(y, -1, 0)
    
    # 断言：使用 ((x - 3)**3*Heaviside(x - 3)) 对象调用 rewrite 方法，并期望其结果等于 (x - 3)**3*SingularityFunction(x, 3, 0)
    assert ((x - 3)**3*Heaviside(x - 3)).rewrite(SingularityFunction) == (x - 3)**3*SingularityFunction(x, 3, 0)
    
    # 断言：使用 Heaviside(0) 对象调用 rewrite 方法，并期望其结果等于 S.Half
    assert Heaviside(0).rewrite(SingularityFunction) == S.Half
```