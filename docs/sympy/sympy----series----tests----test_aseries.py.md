# `D:\src\scipysrc\sympy\sympy\series\tests\test_aseries.py`

```
from sympy.core.function import PoleError  # 导入 PoleError 异常类
from sympy.core.numbers import oo  # 导入无穷大 oo
from sympy.core.singleton import S  # 导入 SymPy 单例 S
from sympy.core.symbol import Symbol  # 导入符号变量 Symbol
from sympy.functions.elementary.exponential import (exp, log)  # 导入指数和对数函数
from sympy.functions.elementary.miscellaneous import sqrt  # 导入平方根函数
from sympy.functions.elementary.trigonometric import (cos, sin)  # 导入余弦和正弦函数
from sympy.series.order import O  # 导入 O 大O符号
from sympy.abc import x  # 导入符号变量 x

from sympy.testing.pytest import raises  # 导入 raises 函数用于测试异常

def test_simple():
    # Gruntz' theses pp. 91 to 96
    # 6.6
    e = sin(1/x + exp(-x)) - sin(1/x)
    # 断言表达式的渐近级数展开结果符合预期
    assert e.aseries(x) == (1/(24*x**4) - 1/(2*x**2) + 1 + O(x**(-6), (x, oo)))*exp(-x)

    e = exp(x) * (exp(1/x + exp(-x)) - exp(1/x))
    # 断言表达式的渐近级数展开结果符合预期，展开到 n=4 阶
    assert e.aseries(x, n=4) == 1/(6*x**3) + 1/(2*x**2) + 1/x + 1 + O(x**(-4), (x, oo))

    e = exp(exp(x) / (1 - 1/x))
    # 断言表达式的渐近级数展开结果符合预期
    assert e.aseries(x) == exp(exp(x) / (1 - 1/x))

    # The implementation of bound in aseries is incorrect currently. This test
    # should be commented out when that is fixed.
    # assert e.aseries(x, bound=3) == exp(exp(x) / x**2)*exp(exp(x) / x)*exp(-exp(x) + exp(x)/(1 - 1/x) - \
    #         exp(x) / x - exp(x) / x**2) * exp(exp(x))

    e = exp(sin(1/x + exp(-exp(x)))) - exp(sin(1/x))
    # 断言表达式的渐近级数展开结果符合预期，展开到 n=4 阶
    assert e.aseries(x, n=4) == (-1/(2*x**3) + 1/x + 1 + O(x**(-4), (x, oo)))*exp(-exp(x))

    e3 = lambda x:exp(exp(exp(x)))
    e = e3(x)/e3(x - 1/e3(x))
    # 断言表达式的渐近级数展开结果符合预期，展开到 n=3 阶
    assert e.aseries(x, n=3) == 1 + exp(x + exp(x))*exp(-exp(exp(x)))\
            + ((-exp(x)/2 - S.Half)*exp(x + exp(x))\
            + exp(2*x + 2*exp(x))/2)*exp(-2*exp(exp(x))) + O(exp(-3*exp(exp(x))), (x, oo))

    e = exp(exp(x)) * (exp(sin(1/x + 1/exp(exp(x)))) - exp(sin(1/x)))
    # 断言表达式的渐近级数展开结果符合预期，展开到 n=4 阶
    assert e.aseries(x, n=4) == -1/(2*x**3) + 1/x + 1 + O(x**(-4), (x, oo))

    n = Symbol('n', integer=True)
    e = (sqrt(n)*log(n)**2*exp(sqrt(log(n))*log(log(n))**2*exp(sqrt(log(log(n)))*log(log(log(n)))**3)))/n
    # 断言表达式的渐近级数展开结果符合预期，展开到 n 阶
    assert e.aseries(n) == \
            exp(exp(sqrt(log(log(n)))*log(log(log(n)))**3)*sqrt(log(n))*log(log(n))**2)*log(n)**2/sqrt(n)


def test_hierarchical():
    e = sin(1/x + exp(-x))
    # 断言表达式的渐近级数展开结果符合预期，hir=True 表示使用层次展开
    assert e.aseries(x, n=3, hir=True) == -exp(-2*x)*sin(1/x)/2 + \
            exp(-x)*cos(1/x) + sin(1/x) + O(exp(-3*x), (x, oo))

    e = sin(x) * cos(exp(-x))
    # 断言表达式的渐近级数展开结果符合预期，hir=True 表示使用层次展开
    assert e.aseries(x, hir=True) == exp(-4*x)*sin(x)/24 - \
            exp(-2*x)*sin(x)/2 + sin(x) + O(exp(-6*x), (x, oo))
    raises(PoleError, lambda: e.aseries(x))
```