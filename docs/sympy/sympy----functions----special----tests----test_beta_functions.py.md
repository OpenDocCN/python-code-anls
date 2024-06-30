# `D:\src\scipysrc\sympy\sympy\functions\special\tests\test_beta_functions.py`

```
# 从 sympy 库中导入所需的函数和符号
from sympy.core.function import (diff, expand_func)
from sympy.core.numbers import I, Rational, pi
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.combinatorial.numbers import catalan
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.beta_functions import (beta, betainc, betainc_regularized)
from sympy.functions.special.gamma_functions import gamma, polygamma
from sympy.functions.special.hyper import hyper
from sympy.integrals.integrals import Integral
from sympy.core.function import ArgumentIndexError
from sympy.core.expr import unchanged
from sympy.testing.pytest import raises

# 定义测试函数 test_beta
def test_beta():
    # 定义符号 x, y
    x, y = symbols('x y')
    # 创建一个虚拟符号 t
    t = Dummy('t')

    # 断言 beta 函数在参数 x, y 上不变
    assert unchanged(beta, x, y)
    # 断言 beta 函数在参数 x, x 上不变
    assert unchanged(beta, x, x)

    # 断言 beta(5, -3) 是实数
    assert beta(5, -3).is_real == True
    # 断言 beta(3, y) 的实部为 None
    assert beta(3, y).is_real is None

    # 断言展开 beta(x, y) 的结果等于 gamma(x)*gamma(y)/gamma(x + y)
    assert expand_func(beta(x, y)) == gamma(x)*gamma(y)/gamma(x + y)
    # 断言 beta(x, y) - beta(y, x) 的展开结果为 0，即 beta 函数对称
    assert expand_func(beta(x, y) - beta(y, x)) == 0
    # 断言展开 beta(x, y) 的结果等于展开 beta(x, y + 1) + beta(x + 1, y) 后简化的结果
    assert expand_func(beta(x, y)) == expand_func(beta(x, y + 1) + beta(x + 1, y)).simplify()

    # 断言 beta(x, y) 对 x 的偏导数
    assert diff(beta(x, y), x) == beta(x, y)*(polygamma(0, x) - polygamma(0, x + y))
    # 断言 beta(x, y) 对 y 的偏导数
    assert diff(beta(x, y), y) == beta(x, y)*(polygamma(0, y) - polygamma(0, x + y))

    # 断言 beta(x, y) 的共轭等于 beta(x 的共轭, y 的共轭)
    assert conjugate(beta(x, y)) == beta(conjugate(x), conjugate(y))

    # 断言 beta(x, y) 在第三个参数为 3 时引发 ArgumentIndexError 异常
    raises(ArgumentIndexError, lambda: beta(x, y).fdiff(3))

    # 断言 beta(x, y) 重写为 gamma 函数的形式
    assert beta(x, y).rewrite(gamma) == gamma(x)*gamma(y)/gamma(x + y)
    # 断言 beta(x) 重写为 gamma 函数的形式
    assert beta(x).rewrite(gamma) == gamma(x)**2/gamma(2*x)
    # 断言 beta(x, y) 重写为积分的形式，并与给定的积分式相等
    assert beta(x, y).rewrite(Integral).dummy_eq(Integral(t**(x - 1) * (1 - t)**(y - 1), (t, 0, 1)))
    # 断言 beta 函数的特定参数组合的值
    assert beta(Rational(-19, 10), Rational(-1, 10)) == S.Zero
    assert beta(Rational(-19, 10), Rational(-9, 10)) == \
        800*2**(S(4)/5)*sqrt(pi)*gamma(S.One/10)/(171*gamma(-S(7)/5))
    assert beta(Rational(19, 10), Rational(29, 10)) == 100/(551*catalan(Rational(19, 10)))
    assert beta(1, 0) == S.ComplexInfinity
    assert beta(0, 1) == S.ComplexInfinity
    assert beta(2, 3) == S.One/12
    assert unchanged(beta, x, x + 1)
    assert unchanged(beta, x, 1)
    assert unchanged(beta, 1, y)
    assert beta(x, x + 1).doit() == 1/(x*(x+1)*catalan(x))
    assert beta(1, y).doit() == 1/y
    assert beta(x, 1).doit() == 1/x
    assert beta(Rational(-19, 10), Rational(-1, 10), evaluate=False).doit() == S.Zero
    assert beta(2) == beta(2, 2)
    assert beta(x, evaluate=False) != beta(x, x)
    assert beta(x, evaluate=False).doit() == beta(x, x)


# 定义测试函数 test_betainc
def test_betainc():
    # 定义符号 a, b, x1, x2
    a, b, x1, x2 = symbols('a b x1 x2')

    # 断言 betainc 函数在参数 a, b, x1, x2 上不变
    assert unchanged(betainc, a, b, x1, x2)
    # 断言 betainc 函数在参数 a, b, 0, x1 上不变
    assert unchanged(betainc, a, b, 0, x1)

    # 断言 betainc(1, 2, 0, -5) 是实数
    assert betainc(1, 2, 0, -5).is_real == True
    # 断言 betainc(1, 2, 0, x2) 的实部为 None
    assert betainc(1, 2, 0, x2).is_real is None
    # 断言 betainc 函数的共轭等于其参数的共轭函数
    assert conjugate(betainc(I, 2, 3 - I, 1 + 4*I)) == betainc(-I, 2, 3 + I, 1 - 4*I)

    # 断言 betainc(a, b, 0, 1) 的重写形式与 beta 函数的重写形式相等
    assert betainc(a, b, 0, 1).rewrite(Integral).dummy_eq(beta(a, b).rewrite(Integral))
    # 断言：使用 betainc 函数计算并验证结果是否等于 x2*hyper((1, -1), (2,), x2)，其中 betainc 是不完全贝塔函数
    assert betainc(1, 2, 0, x2).rewrite(hyper) == x2*hyper((1, -1), (2,), x2)
    
    # 断言：使用 betainc 函数计算并验证在参数为 (1, 2, 3, 3) 时的数值结果是否等于 0
    assert betainc(1, 2, 3, 3).evalf() == 0
# 定义测试函数 test_betainc_regularized，用于测试 betainc_regularized 函数的不同情况
def test_betainc_regularized():
    # 使用 symbols 函数定义符号变量 a, b, x1, x2
    a, b, x1, x2 = symbols('a b x1 x2')

    # 断言 betainc_regularized 函数对参数 a, b, x1, x2 不产生变化
    assert unchanged(betainc_regularized, a, b, x1, x2)
    # 断言 betainc_regularized 函数对参数 a, b, 0, x1 不产生变化
    assert unchanged(betainc_regularized, a, b, 0, x1)

    # 断言当参数为 (3, 5, 0, -1) 时，betainc_regularized 返回的结果是实数
    assert betainc_regularized(3, 5, 0, -1).is_real == True
    # 断言当参数为 (3, 5, 0, x2) 时，betainc_regularized 返回的结果是 None
    assert betainc_regularized(3, 5, 0, x2).is_real is None
    # 断言 betainc_regularized 函数的共轭结果等于对应参数的共轭 betainc_regularized 函数的结果
    assert conjugate(betainc_regularized(3*I, 1, 2 + I, 1 + 2*I)) == betainc_regularized(-3*I, 1, 2 - I, 1 - 2*I)

    # 断言当参数为 (a, b, 0, 1) 时，betainc_regularized 使用 Integral 重写后结果为 1
    assert betainc_regularized(a, b, 0, 1).rewrite(Integral) == 1
    # 断言当参数为 (1, 2, x1, x2) 时，betainc_regularized 使用 hyper 函数重写后结果为指定的表达式
    assert betainc_regularized(1, 2, x1, x2).rewrite(hyper) == 2*x2*hyper((1, -1), (2,), x2) - 2*x1*hyper((1, -1), (2,), x1)

    # 断言当参数为 (4, 1, 5, 5) 时，betainc_regularized 计算后的浮点数结果等于 0
    assert betainc_regularized(4, 1, 5, 5).evalf() == 0
```