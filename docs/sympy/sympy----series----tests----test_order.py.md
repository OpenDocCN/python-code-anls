# `D:\src\scipysrc\sympy\sympy\series\tests\test_order.py`

```
from sympy.core.add import Add  # 导入 Add 类，用于表示 Sympy 表达式的加法操作
from sympy.core.function import (Function, expand)  # 导入 Function 类和 expand 函数
from sympy.core.numbers import (I, Rational, nan, oo, pi)  # 导入常见的数学常数和对象
from sympy.core.singleton import S  # 导入 Sympy 的单例对象
from sympy.core.symbol import (Symbol, symbols)  # 导入 Symbol 类和 symbols 函数
from sympy.functions.elementary.complexes import (conjugate, transpose)  # 导入复数和转置操作相关函数
from sympy.functions.elementary.exponential import (exp, log)  # 导入指数和对数函数
from sympy.functions.elementary.miscellaneous import sqrt  # 导入平方根函数
from sympy.functions.elementary.trigonometric import (cos, sin)  # 导入三角函数
from sympy.integrals.integrals import Integral  # 导入积分函数
from sympy.series.order import O, Order  # 导入阶数展开相关函数
from sympy.core.expr import unchanged  # 导入 unchanged 函数
from sympy.testing.pytest import raises  # 导入 raises 函数用于测试
from sympy.abc import w, x, y, z  # 导入符号 w, x, y, z

def test_caching_bug():
    # 需要作为第一个测试，以确保所有缓存都是干净的
    # 缓存 Order(w) 对象
    O(w)
    # 并且测试它不会引发异常
    O(w**(-1/x/log(3)*log(5)), w)

def test_free_symbols():
    # 检查 Order 对象的自由符号集合是否正确
    assert Order(1).free_symbols == set()
    assert Order(x).free_symbols == {x}
    assert Order(1, x).free_symbols == {x}
    assert Order(x*y).free_symbols == {x, y}
    assert Order(x, x, y).free_symbols == {x, y}

def test_simple_1():
    o = Rational(0)
    # 测试 Order 对象的基本性质
    assert Order(2*x) == Order(x)
    assert Order(x)*3 == Order(x)
    assert -28*Order(x) == Order(x)
    assert Order(Order(x)) == Order(x)
    assert Order(Order(x), y) == Order(Order(x), x, y)
    assert Order(-23) == Order(1)
    assert Order(exp(x)) == Order(1, x)
    assert Order(exp(1/x)).expr == exp(1/x)
    assert Order(x*exp(1/x)).expr == x*exp(1/x)
    assert Order(x**(o/3)).expr == x**(o/3)
    assert Order(x**(o*Rational(5, 3))).expr == x**(o*Rational(5, 3))
    assert Order(x**2 + x + y, x) == O(1, x)
    assert Order(x**2 + x + y, y) == O(1, y)
    raises(ValueError, lambda: Order(exp(x), x, x))
    raises(TypeError, lambda: Order(x, 2 - x))

def test_simple_2():
    # 测试 Order 对象的乘法和除法
    assert Order(2*x)*x == Order(x**2)
    assert Order(2*x)/x == Order(1, x)
    assert Order(2*x)*x*exp(1/x) == Order(x**2*exp(1/x))
    assert (Order(2*x)*x*exp(1/x)/log(x)**3).expr == x**2*exp(1/x)*log(x)**-3

def test_simple_3():
    # 测试 Order 对象的加法
    assert Order(x) + x == Order(x)
    assert Order(x) + 2 == 2 + Order(x)
    assert Order(x) + x**2 == Order(x)
    assert Order(x) + 1/x == 1/x + Order(x)
    assert Order(1/x) + 1/x**2 == 1/x**2 + Order(1/x)
    assert Order(x) + exp(1/x) == Order(x) + exp(1/x)

def test_simple_4():
    # 测试 Order 对象的幂运算
    assert Order(x)**2 == Order(x**2)

def test_simple_5():
    # 测试 Order 对象的加法与其他 Order 对象的组合
    assert Order(x) + Order(x**2) == Order(x)
    assert Order(x) + Order(x**-2) == Order(x**-2)
    assert Order(x) + Order(1/x) == Order(1/x)

def test_simple_6():
    # 测试 Order 对象的减法
    assert Order(x) - Order(x) == Order(x)
    assert Order(x) + Order(1) == Order(1)
    assert Order(x) + Order(x**2) == Order(x)
    assert Order(1/x) + Order(1) == Order(1/x)
    assert Order(x) + Order(exp(1/x)) == Order(exp(1/x))
    assert Order(x**3) + Order(exp(2/x)) == Order(exp(2/x))
    assert Order(x**-3) + Order(exp(2/x)) == Order(exp(2/x))

def test_simple_7():
    # 这里可以继续添加测试，以验证 Order 对象的更多性质
    pass
    # 断言：1 + O(1) 等于 O(1)，即常数时间复杂度加上常数仍然是常数时间复杂度
    assert 1 + O(1) == O(1)
    
    # 断言：2 + O(1) 等于 O(1)，即常数时间复杂度加上常数仍然是常数时间复杂度
    assert 2 + O(1) == O(1)
    
    # 断言：x + O(1) 等于 O(1)，即变量 x 加上常数时间复杂度仍然是常数时间复杂度
    assert x + O(1) == O(1)
    
    # 断言：1/x + O(1) 等于 1/x + O(1)，即 x 的倒数加上常数时间复杂度仍然是 1/x 加上常数时间复杂度
    assert 1/x + O(1) == 1/x + O(1)
# 测试简单的 O 表达式比较
def test_simple_8():
    # 断言 O(sqrt(-x)) 等于 O(sqrt(x))
    assert O(sqrt(-x)) == O(sqrt(x))
    # 断言 O(x**2*sqrt(x)) 等于 O(x**(5/2))
    assert O(x**2*sqrt(x)) == O(x**Rational(5, 2))
    # 断言 O(x**3*sqrt(-(-x)**3)) 等于 O(x**(9/2))
    assert O(x**3*sqrt(-(-x)**3)) == O(x**Rational(9, 2))
    # 断言 O(x**(3/2)*sqrt((-x)**3)) 等于 O(x**3)
    assert O(x**Rational(3, 2)*sqrt((-x)**3)) == O(x**3)
    # 断言 O(x*(-2*x)**(I/2)) 等于 O(x*(-x)**(I/2))
    assert O(x*(-2*x)**(I/2)) == O(x*(-x)**(I/2))


# 测试 Order 类的 as_expr_variables 方法
def test_as_expr_variables():
    # 断言 Order(x).as_expr_variables(None) 返回 (x, ((x, 0),))
    assert Order(x).as_expr_variables(None) == (x, ((x, 0),))
    # 断言 Order(x).as_expr_variables(((x, 0),)) 返回 (x, ((x, 0),))
    assert Order(x).as_expr_variables(((x, 0),)) == (x, ((x, 0),))
    # 断言 Order(y).as_expr_variables(((x, 0),)) 返回 (y, ((x, 0), (y, 0)))
    assert Order(y).as_expr_variables(((x, 0),)) == (y, ((x, 0), (y, 0)))
    # 断言 Order(y).as_expr_variables(((x, 0), (y, 0))) 返回 (y, ((x, 0), (y, 0)))
    assert Order(y).as_expr_variables(((x, 0), (y, 0))) == (y, ((x, 0), (y, 0)))


# 测试 Order 类的 contains 方法的部分情况
def test_contains_0():
    # 断言 Order(1, x).contains(Order(1, x)) 返回 True
    assert Order(1, x).contains(Order(1, x))
    # 断言 Order(1, x).contains(Order(1)) 返回 True
    assert Order(1, x).contains(Order(1))
    # 断言 Order(1).contains(Order(1, x)) 返回 False
    assert Order(1).contains(Order(1, x)) is False


# 测试 Order 类的 contains 方法的更多情况
def test_contains_1():
    # 断言 Order(x).contains(Order(x)) 返回 True
    assert Order(x).contains(Order(x))
    # 断言 Order(x).contains(Order(x**2)) 返回 True
    assert Order(x).contains(Order(x**2))
    # 断言 Order(x**2).contains(Order(x)) 返回 False
    assert not Order(x**2).contains(Order(x))
    # 断言 Order(x).contains(Order(1/x)) 返回 False
    assert not Order(x).contains(Order(1/x))
    # 断言 Order(1/x).contains(Order(exp(1/x))) 返回 False
    assert not Order(1/x).contains(Order(exp(1/x)))
    # 断言 Order(x).contains(Order(exp(1/x))) 返回 False
    assert not Order(x).contains(Order(exp(1/x)))
    # 断言 Order(1/x).contains(Order(x)) 返回 True
    assert Order(1/x).contains(Order(x))
    # 断言 Order(exp(1/x)).contains(Order(x)) 返回 True
    assert Order(exp(1/x)).contains(Order(x))
    # 断言 Order(exp(1/x)).contains(Order(1/x)) 返回 True
    assert Order(exp(1/x)).contains(Order(1/x))
    # 断言 Order(exp(1/x)).contains(Order(exp(1/x))) 返回 True
    assert Order(exp(1/x)).contains(Order(exp(1/x)))
    # 断言 Order(exp(2/x)).contains(Order(exp(1/x))) 返回 True
    assert Order(exp(2/x)).contains(Order(exp(1/x)))
    # 断言 Order(exp(1/x)).contains(Order(exp(2/x))) 返回 False
    assert not Order(exp(1/x)).contains(Order(exp(2/x)))


# 测试 Order 类的 contains 方法的更多情况
def test_contains_2():
    # 断言 Order(x).contains(Order(y)) 返回 None
    assert Order(x).contains(Order(y)) is None
    # 断言 Order(x).contains(Order(y*x)) 返回 True
    assert Order(x).contains(Order(y*x))
    # 断言 Order(y*x).contains(Order(x)) 返回 True
    assert Order(y*x).contains(Order(x))
    # 断言 Order(y).contains(Order(x*y)) 返回 None
    assert Order(y).contains(Order(x*y))
    # 断言 Order(x).contains(Order(y**2*x)) 返回 None
    assert Order(x).contains(Order(y**2*x))


# 测试 Order 类的 contains 方法的更多情况
def test_contains_3():
    # 断言 Order(x*y**2).contains(Order(x**2*y)) 返回 None
    assert Order(x*y**2).contains(Order(x**2*y)) is None
    # 断言 Order(x**2*y).contains(Order(x*y**2)) 返回 None
    assert Order(x**2*y).contains(Order(x*y**2)) is None


# 测试 Order 类的 contains 方法关于 sin 和 cos 的情况
def test_contains_4():
    # 断言 Order(sin(1/x**2)).contains(Order(cos(1/x**2))) 返回 True
    assert Order(sin(1/x**2)).contains(Order(cos(1/x**2))) is True
    # 断言 Order(cos(1/x**2)).contains(Order(sin(1/x**2))) 返回 True
    assert Order(cos(1/x**2)).contains(Order(sin(1/x**2))) is True


# 测试 Order 类的 in 运算符重载
def test_contains():
    # 断言 Order(1, x) not in Order(1) 返回 True
    assert Order(1, x) not in Order(1)
    # 断言 Order(1) in Order(1, x) 返回 True
    assert Order(1) in Order(1, x)
    # 断言 lambda: Order(x*y**2) in Order(x**2*y) 抛出 TypeError 异常
    raises(TypeError, lambda: Order(x*y**2) in Order(x**2*y))


# 测试 Order 类的加法运算
def test_add_1():
    # 断言 Order(x + x) == Order(x) 返回 True
    assert Order(x + x) == Order(x)
    # 断言 Order(3*x - 2*x**2) == Order(x) 返回 True
    assert Order(3*x - 2*x**2) == Order(x)
    # 断言 Order(1 + x) == Order(1, x) 返回 True
    assert Order(1 + x) == Order(1, x)
    # 断言 Order(1 + 1/x) == Order(1/x) 返回 True
    assert Order(1 + 1/x) == Order(1/x)
    # 断言 Order(log(x) + 1/log(x)) == Order((log(x)**2 + 1)/log(x)) 为 TODO
    assert Order(log(x) + 1/log(x)) == Order((log(x)**2 + 1)/log(x))
    # 断言 Order(exp(1/x) + x) == Order(exp(1/x)) 返回 True
    assert Order(exp(1/x) + x) == Order(exp(1/x))
    # 断言 Order(exp(1/x) + 1/x**20) == Order(exp(1/x)) 返回 True
    assert Order(exp(1/x) + 1/x**20) == Order(exp(1/x))


# 测试 Order 类的 ln 参数的加法运算
def test_ln_args():
    # 断言 O(log(x)) + O(log(2*x)) == O(log(x)) 返回 True
    assert O(log(x)) + O(log(2*x)) == O(log(x))
    # 断言 O(log(x)) + O(log(x**3)) == O(log(x)) 返回 True
    assert O(log(x)) + O(log(x**3)) == O(log(x))
    # 断言 O(log(x*y)) + O(log(x) + log(y)) == O(log(x) + log(y), x, y) 返回 True
    assert O(log(x*y)) + O(log(x) + log(y)) == O(log(x) + log(y), x, y)


# 测试 Order 类的多变量情况
def test_multivar_0():
    # 断言 Order(x*y
    # 断言：验证 Order(x*y**2, y) 的 expr 属性是否等于 y**2
    assert Order(x*y**2, y).expr == y**2
    
    # 断言：验证 Order(x*y*z) 的 expr 属性是否等于 x*y*z
    assert Order(x*y*z).expr == x*y*z
    
    # 断言：验证 Order(x/y) 的 expr 属性是否等于 x/y
    assert Order(x/y).expr == x/y
    
    # 断言：验证 Order(x*exp(1/y)) 的 expr 属性是否等于 x*exp(1/y)
    assert Order(x*exp(1/y)).expr == x*exp(1/y)
    
    # 断言：验证 Order(exp(x)*exp(1/y)) 的 expr 属性是否等于 exp(x)*exp(1/y)
    assert Order(exp(x)*exp(1/y)).expr == exp(x)*exp(1/y)
def test_multivar_0a():
    assert Order(exp(1/x)*exp(1/y)).expr == exp(1/x)*exp(1/y)

def test_multivar_1():
    assert Order(x + y).expr == x + y
    assert Order(x + 2*y).expr == x + y
    assert (Order(x + y) + x).expr == (x + y)
    assert (Order(x + y) + x**2) == Order(x + y)
    assert (Order(x + y) + 1/x) == 1/x + Order(x + y)
    assert Order(x**2 + y*x).expr == x**2 + y*x

def test_multivar_2():
    assert Order(x**2*y + y**2*x, x, y).expr == x**2*y + y**2*x

def test_multivar_mul_1():
    assert Order(x + y)*x == Order(x**2 + y*x, x, y)

def test_multivar_3():
    assert (Order(x) + Order(y)).args in [
        (Order(x), Order(y)),
        (Order(y), Order(x))]
    assert Order(x) + Order(y) + Order(x + y) == Order(x + y)
    assert (Order(x**2*y) + Order(y**2*x)).args in [
        (Order(x*y**2), Order(y*x**2)),
        (Order(y*x**2), Order(x*y**2))]
    assert (Order(x**2*y) + Order(y*x)) == Order(x*y)

def test_issue_3468():
    y = Symbol('y', negative=True)
    z = Symbol('z', complex=True)

    # 检查 Order 不会修改符号的假设
    Order(x)
    Order(y)
    Order(z)

    assert x.is_positive is None
    assert y.is_positive is False
    assert z.is_positive is None

def test_leading_order():
    assert (x + 1 + 1/x**5).extract_leading_order(x) == ((1/x**5, O(1/x**5)),)
    assert (1 + 1/x).extract_leading_order(x) == ((1/x, O(1/x)),)
    assert (1 + x).extract_leading_order(x) == ((1, O(1, x)),)
    assert (1 + x**2).extract_leading_order(x) == ((1, O(1, x)),)
    assert (2 + x**2).extract_leading_order(x) == ((2, O(1, x)),)
    assert (x + x**2).extract_leading_order(x) == ((x, O(x)),)

def test_leading_order2():
    assert set((2 + pi + x**2).extract_leading_order(x)) == {(pi, O(1, x)),
            (S(2), O(1, x))}
    assert set((2*x + pi*x + x**2).extract_leading_order(x)) == {(2*x, O(x)),
            (x*pi, O(x))}

def test_order_leadterm():
    assert O(x**2)._eval_as_leading_term(x) == O(x**2)

def test_order_symbols():
    e = x*y*sin(x)*Integral(x, (x, 1, 2))
    assert O(e) == O(x**2*y, x, y)
    assert O(e, x) == O(x**2)

def test_nan():
    assert O(nan) is nan
    assert not O(x).contains(nan)

def test_O1():
    assert O(1, x) * x == O(x)
    assert O(1, y) * x == O(1, y)

def test_getn():
    # 测试套件已经随意测试了其他行
    assert O(x).getn() == 1
    assert O(x/log(x)).getn() == 1
    assert O(x**2/log(x)**2).getn() == 2
    assert O(x*log(x)).getn() == 1
    raises(NotImplementedError, lambda: (O(x) + O(y)).getn())

def test_diff():
    assert O(x**2).diff(x) == O(x)

def test_getO():
    assert (x).getO() is None
    assert (x).removeO() == x
    assert (O(x)).getO() == O(x)
    assert (O(x)).removeO() == 0
    assert (z + O(x) + O(y)).getO() == O(x) + O(y)
    assert (z + O(x) + O(y)).removeO() == z
    raises(NotImplementedError, lambda: (O(x) + O(y)).getn())

def test_leading_term():
    # 从 sympy 库中的 gamma_functions 模块导入 digamma 函数
    from sympy.functions.special.gamma_functions import digamma
    # 使用断言来验证 O(1/digamma(1/x)) 是否等价于 O(1/log(x))
    assert O(1/digamma(1/x)) == O(1/log(x))
# 定义测试函数 test_eval，用于测试 Order 类的各种操作和属性
def test_eval():
    # 断言：替换 Order(x) 中的 x 为 1，应该返回 1
    assert Order(x).subs(Order(x), 1) == 1
    # 断言：替换 Order(x) 中的 x 为 y，应该返回 Order(y)
    assert Order(x).subs(x, y) == Order(y)
    # 断言：替换 Order(x) 中的 y 为 x，应该返回 Order(x)
    assert Order(x).subs(y, x) == Order(x)
    # 断言：替换 Order(x) 中的 x 为 x + y，应该返回 Order(x + y, (x, -y))
    assert Order(x).subs(x, x + y) == Order(x + y, (x, -y))
    # 断言：检查 O(1)**x 是否是一个指数函数
    assert (O(1)**x).is_Pow


# 定义测试函数 test_issue_4279，用于测试与 GitHub 问题编号 4279 相关的功能
def test_issue_4279():
    # 定义符号变量 a, b
    a, b = symbols('a b')
    # 断言：O(a, a, b) + O(1, a, b) 应该等于 O(1, a, b)
    assert O(a, a, b) + O(1, a, b) == O(1, a, b)
    # 断言：O(b, a, b) + O(1, a, b) 应该等于 O(1, a, b)
    assert O(b, a, b) + O(1, a, b) == O(1, a, b)
    # 断言：O(a + b, a, b) + O(1, a, b) 应该等于 O(1, a, b)
    assert O(a + b, a, b) + O(1, a, b) == O(1, a, b)
    # 断言：O(1, a, b) + O(a, a, b) 应该等于 O(1, a, b)
    assert O(1, a, b) + O(a, a, b) == O(1, a, b)
    # 断言：O(1, a, b) + O(b, a, b) 应该等于 O(1, a, b)
    assert O(1, a, b) + O(b, a, b) == O(1, a, b)
    # 断言：O(1, a, b) + O(a + b, a, b) 应该等于 O(1, a, b)
    assert O(1, a, b) + O(a + b, a, b) == O(1, a, b)


# 定义测试函数 test_issue_4855，用于测试与 GitHub 问题编号 4855 相关的功能
def test_issue_4855():
    # 断言：1/O(1) 不等于 O(1)
    assert 1/O(1) != O(1)
    # 断言：1/O(x) 不等于 O(1/x)
    assert 1/O(x) != O(1/x)
    # 断言：1/O(x, (x, oo)) 不等于 O(1/x, (x, oo))
    assert 1/O(x, (x, oo)) != O(1/x, (x, oo))

    # 定义函数 f
    f = Function('f')
    # 断言：1/O(f(x)) 不等于 O(1/x)
    assert 1/O(f(x)) != O(1/x)


# 定义测试函数 test_order_conjugate_transpose，用于测试 Order 类的共轭和转置操作
def test_order_conjugate_transpose():
    # 定义符号变量 x 和 y 的属性
    x = Symbol('x', real=True)
    y = Symbol('y', imaginary=True)
    # 断言：共轭操作对 Order(x) 返回 Order(conjugate(x))
    assert conjugate(Order(x)) == Order(conjugate(x))
    # 断言：共轭操作对 Order(y) 返回 Order(conjugate(y))
    assert conjugate(Order(y)) == Order(conjugate(y))
    # 断言：共轭操作对 Order(x**2) 返回 Order(conjugate(x)**2)
    assert conjugate(Order(x**2)) == Order(conjugate(x)**2)
    # 断言：转置操作对 Order(x) 返回 Order(transpose(x))
    assert transpose(Order(x)) == Order(transpose(x))
    # 断言：转置操作对 Order(y) 返回 Order(transpose(y))
    assert transpose(Order(y)) == Order(transpose(y))
    # 断言：转置操作对 Order(x**2) 返回 Order(transpose(x)**2)
    assert transpose(Order(x**2)) == Order(transpose(x)**2)
    # 断言：转置操作对 Order(y**2) 返回 Order(transpose(y)**2)
    assert transpose(Order(y**2)) == Order(transpose(y)**2)


# 定义测试函数 test_order_noncommutative，用于测试 Order 类的非交换性操作
def test_order_noncommutative():
    # 定义非交换符号 A
    A = Symbol('A', commutative=False)
    # 断言：Order(A + A*x, x) 应该等于 Order(1, x)
    assert Order(A + A*x, x) == Order(1, x)
    # 断言：(A + A*x)*Order(x) 应该等于 Order(x)
    assert (A + A*x)*Order(x) == Order(x)
    # 断言：(A*x)*Order(x) 应该等于 Order(x**2, x)
    assert (A*x)*Order(x) == Order(x**2, x)
    # 断言：expand((1 + Order(x))*A*A*x) 应该等于 A*A*x + Order(x**2, x)
    assert expand((1 + Order(x))*A*A*x) == A*A*x + Order(x**2, x)
    # 断言：expand((A*A + Order(x))*x) 应该等于 A*A*x + Order(x**2, x)
    assert expand((A*A + Order(x))*x) == A*A*x + Order(x**2, x)
    # 断言：expand((A + Order(x))*A*x) 应该等于 A*A*x + Order(x**2, x)
    assert expand((A + Order(x))*A*x) == A*A*x + Order(x**2, x)


# 定义测试函数 test_issue_6753，用于测试与 GitHub 问题编号 6753 相关的功能
def test_issue_6753():
    # 断言：(1 + x**2)**10000*O(x) 应该等于 O(x)
    assert (1 + x**2)**10000*O(x) == O(x)


# 定义测试函数 test_order_at_infinity，用于测试 Order 类在无穷远点的行为
def test_order_at_infinity():
    # 断言：Order(1 + x, (x, oo)) 应该等于 Order(x, (x, oo))
    assert Order(1 + x, (x, oo)) == Order(x, (x, oo))
    # 断言：Order(3*x, (x, oo)) 应该等于 Order(x, (x, oo))
    assert Order(3*x, (x, oo)) == Order(x, (x, oo))
    # 断言：Order(x, (x, oo))*3 应该等于 Order(x, (x, oo))
    assert Order(x, (x, oo))*3 == Order(x, (x, oo))
    # 断言：-28*Order(x, (x, oo)) 应该等于 Order(x, (x, oo))
    assert -28*Order(x, (x, oo)) == Order(x, (x, oo))
    # 断言：Order(Order(x, (x, oo)), (x, oo)) 应该等于 Order(x, (x, oo))
    assert Order(Order(x, (x, oo)), (x, oo)) == Order(x, (x, oo))
    # 断言：Order(Order(x, (x, oo)), (y, oo)) 应该等于 Order(x, (x, oo), (y, oo))
    assert Order(Order(x, (x, oo)), (y, oo)) == Order(x, (x, oo), (y, oo))
    # 断言：Order(3, (x, oo)) 应该等于 Order(1, (x, oo))
    assert Order(3, (x, oo)) == Order(1, (x, oo))
    #
    # 断言：计算 1/x 的渐近展开与 1/x**2 的和等于 1/x**2 的渐近展开，且它们都等于 1/x 的渐近展开
    assert Order(1/x, (x, oo)) + 1/x**2 == 1/x**2 + Order(1/x, (x, oo)) == Order(1/x, (x, oo))
    
    # 断言：计算 x 的渐近展开与 exp(1/x) 的和等于 exp(1/x) 的渐近展开，且它们都等于 x 的渐近展开
    assert Order(x, (x, oo)) + exp(1/x) == exp(1/x) + Order(x, (x, oo))
    
    # 断言：计算 x 的渐近展开的平方等于 x**2 的渐近展开
    assert Order(x, (x, oo))**2 == Order(x**2, (x, oo))
    
    # 断言：计算 x 的渐近展开与 x**2 的渐近展开的和等于 x**2 的渐近展开
    assert Order(x, (x, oo)) + Order(x**2, (x, oo)) == Order(x**2, (x, oo))
    
    # 断言：计算 x 的渐近展开与 x**-2 的渐近展开的和等于 x 的渐近展开
    assert Order(x, (x, oo)) + Order(x**-2, (x, oo)) == Order(x, (x, oo))
    
    # 断言：计算 x 的渐近展开与 1/x 的渐近展开的和等于 x 的渐近展开
    assert Order(x, (x, oo)) + Order(1/x, (x, oo)) == Order(x, (x, oo))
    
    # 断言：计算 x 的渐近展开减去自身的渐近展开等于 x 的渐近展开
    assert Order(x, (x, oo)) - Order(x, (x, oo)) == Order(x, (x, oo))
    
    # 断言：计算 x 的渐近展开与常数 1 的渐近展开的和等于 x 的渐近展开
    assert Order(x, (x, oo)) + Order(1, (x, oo)) == Order(x, (x, oo))
    
    # 断言：计算 x 的渐近展开与 x**2 的渐近展开的和等于 x**2 的渐近展开
    assert Order(x, (x, oo)) + Order(x**2, (x, oo)) == Order(x**2, (x, oo))
    
    # 断言：计算 1/x 的渐近展开与常数 1 的渐近展开的和等于常数 1 的渐近展开
    assert Order(1/x, (x, oo)) + Order(1, (x, oo)) == Order(1, (x, oo))
    
    # 断言：计算 x 的渐近展开与 exp(1/x) 的渐近展开的和等于 x 的渐近展开
    assert Order(x, (x, oo)) + Order(exp(1/x), (x, oo)) == Order(x, (x, oo))
    
    # 断言：计算 x**3 的渐近展开与 exp(2/x) 的渐近展开的和等于 x**3 的渐近展开
    assert Order(x**3, (x, oo)) + Order(exp(2/x), (x, oo)) == Order(x**3, (x, oo))
    
    # 断言：计算 x**-3 的渐近展开与 exp(2/x) 的渐近展开的和等于 exp(2/x) 的渐近展开
    assert Order(x**-3, (x, oo)) + Order(exp(2/x), (x, oo)) == Order(exp(2/x), (x, oo))
    
    # issue 7207
    # 断言：检查 Order(exp(x), (x, oo)).expr 是否等于 Order(2*exp(x), (x, oo)).expr 是否等于 exp(x)
    assert Order(exp(x), (x, oo)).expr == Order(2*exp(x), (x, oo)).expr == exp(x)
    
    # 断言：检查 Order(y**x, (x, oo)).expr 是否等于 Order(2*y**x, (x, oo)).expr 是否等于 exp(x*log(y))
    assert Order(y**x, (x, oo)).expr == Order(2*y**x, (x, oo)).expr == exp(x*log(y))
    
    # issue 19545
    # 断言：检查 Order(1/x - 3/(3*x + 2), (x, oo)).expr 是否等于 x**(-2)
    assert Order(1/x - 3/(3*x + 2), (x, oo)).expr == x**(-2)
# 测试混合顺序在零和无穷处的 Order 对象相加是否为 Add 对象
assert (Order(x, (x, 0)) + Order(x, (x, oo))).is_Add
# 检查 Order 对象相加的交换律
assert Order(x, (x, 0)) + Order(x, (x, oo)) == Order(x, (x, oo)) + Order(x, (x, 0))
# 检查 Order 对象嵌套的简化
assert Order(Order(x, (x, oo))) == Order(x, (x, oo))

# 以下操作目前不支持，引发 NotImplementedError 异常
raises(NotImplementedError, lambda: Order(x, (x, 0))*Order(x, (x, oo)))
raises(NotImplementedError, lambda: Order(x, (x, oo))*Order(x, (x, 0)))
raises(NotImplementedError, lambda: Order(Order(x, (x, oo)), y))
raises(NotImplementedError, lambda: Order(Order(x), (x, oo)))


# 测试某点处的 Order 对象
assert Order(x, (x, 1)) == Order(1, (x, 1))
assert Order(2*x - 2, (x, 1)) == Order(x - 1, (x, 1))
assert Order(-x + 1, (x, 1)) == Order(x - 1, (x, 1))
assert Order(x - 1, (x, 1))**2 == Order((x - 1)**2, (x, 1))
assert Order(x - 2, (x, 2)) - O(x - 2, (x, 2)) == Order(x - 2, (x, 2))


# 测试 Order 对象在替换极限时的行为
# issue 3333
assert (1 + Order(x)).subs(x, 1/x) == 1 + Order(1/x, (x, oo))
assert (1 + Order(x)).limit(x, 0) == 1
# issue 5769
assert ((x + Order(x**2))/x).limit(x, 0) == 1

assert Order(x**2).subs(x, y - 1) == Order((y - 1)**2, (y, 1))
assert Order(10*x**2, (x, 2)).subs(x, y - 1) == Order(1, (y, 3))


# 测试指数函数在级数展开时的 Order 对象
assert exp(x).series(x, 10, 1) == exp(10) + Order(x - 10, (x, 10))


# 测试 Order 对象乘法和幂运算
assert O(1)*O(1) == O(1)
assert O(1)**O(1) == O(1)


# 测试特定函数表达式的 Order 对象
assert O(x*log(x) + sin(x), (x, oo)) == O(x*log(x), (x, oo))


# 测试添加 Order 对象的性能
l = [x**i for i in range(1000)]
l.append(O(x**1001))
assert Add(*l).subs(x,1) == O(1)


# 测试包含 Order 对象的数值和分母
assert (x**(-4) + x**(-3) + x**(-1) + O(x**(-6), (x, oo))).as_numer_denom() == (
    x**4 + x**5 + x**7 + O(x**2, (x, oo)), x**8)
assert (x**3 + O(x**2, (x, oo))).is_Add
assert O(x**2, (x, oo)).contains(x**3) is False
assert O(x, (x, oo)).contains(O(x, (x, 0))) is None
assert O(x, (x, 0)).contains(O(x, (x, oo))) is None
raises(NotImplementedError, lambda: O(x**3).contains(x**w))


# 测试包含 Order 对象的函数
assert O(1/x**2 + 1/x**4, (x, -oo)) == O(1/x**2, (x, -oo))
assert O(1/x**4 + exp(x), (x, -oo)) == O(1/x**4, (x, -oo))
assert O(1/x**4 + exp(-x), (x, -oo)) == O(exp(-x), (x, -oo))
assert O(1/x, (x, oo)).subs(x, -x) == O(-1/x, (x, -oo))


# 测试 Order 对象的不变性
assert unchanged(Order, 0)


# 测试 Order 对象是否包含特定属性
assert O(log(x)).contains(2)


# 测试包含变量指数/幂次的表达式的 Order 对象
assert O(x**x + 2**x, (x, oo)) == O(exp(x*log(x)), (x, oo))
assert O(x**x + x**2, (x, oo)) == O(exp(x*log(x)), (x, oo))
assert O(x**x + 1/x**2, (x, oo)) == O(exp(x*log(x)), (x, oo))
assert O(2**x + 3**x , (x, oo)) == O(exp(x*log(3)), (x, oo))


# 测试特定函数的 Order 对象
assert O(x*sin(x) + 1, (x, oo)) == O(x, (x, oo))
```