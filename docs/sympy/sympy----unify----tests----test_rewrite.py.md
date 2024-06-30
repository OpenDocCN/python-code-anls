# `D:\src\scipysrc\sympy\sympy\unify\tests\test_rewrite.py`

```
# 导入所需的函数和符号类
from sympy.unify.rewrite import rewriterule
from sympy.core.basic import Basic
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.trigonometric import sin
from sympy.abc import x, y
from sympy.strategies.rl import rebuild
from sympy.assumptions import Q

# 定义符号变量 p 和 q
p, q = Symbol('p'), Symbol('q')

# 定义测试函数 test_simple
def test_simple():
    # 创建一个重写规则对象 rl，将 Basic(p, 1) 重写为 Basic(p, 2)，仅考虑变量 p
    rl = rewriterule(Basic(p, S(1)), Basic(p, S(2)), variables=(p,))
    # 断言应用 rl 到 Basic(S(3), S(1)) 后的结果为 [Basic(S(3), S(2))]
    assert list(rl(Basic(S(3), S(1)))) == [Basic(S(3), S(2))]

    # 定义 p1 和 p2，分别为 p 的平方和 p 的立方
    p1 = p**2
    p2 = p**3
    # 创建一个重写规则对象 rl，将 p1 重写为 p2，仅考虑变量 p
    rl = rewriterule(p1, p2, variables=(p,))

    # 定义表达式 expr = x 的平方
    expr = x**2
    # 断言应用 rl 到 expr 后的结果为 [x 的立方]
    assert list(rl(expr)) == [x**3]

# 定义测试函数 test_simple_variables
def test_simple_variables():
    # 创建一个重写规则对象 rl，将 Basic(x, 1) 重写为 Basic(x, 2)，仅考虑变量 x
    rl = rewriterule(Basic(x, S(1)), Basic(x, S(2)), variables=(x,))
    # 断言应用 rl 到 Basic(S(3), S(1)) 后的结果为 [Basic(S(3), S(2))]
    assert list(rl(Basic(S(3), S(1)))) == [Basic(S(3), S(2))]

    # 创建一个重写规则对象 rl，将 x 的平方 重写为 x 的立方，仅考虑变量 x
    rl = rewriterule(x**2, x**3, variables=(x,))
    # 断言应用 rl 到 y 的平方 后的结果为 [y 的立方]
    assert list(rl(y**2)) == [y**3]

# 定义测试函数 test_moderate
def test_moderate():
    # 定义 p1 和 p2，分别为 p 的平方加 q 的立方 和 p 乘 q 的四次方
    p1 = p**2 + q**3
    p2 = (p*q)**4
    # 创建一个重写规则对象 rl，将 p1 重写为 p2，考虑变量 p 和 q
    rl = rewriterule(p1, p2, (p, q))

    # 定义表达式 expr = x 的平方加 y 的立方
    expr = x**2 + y**3
    # 断言应用 rl 到 expr 后的结果为 [(x 乘 y) 的四次方]
    assert list(rl(expr)) == [(x*y)**4]

# 定义测试函数 test_sincos
def test_sincos():
    # 定义 p1 和 p2，分别为 sin(p) 的平方加 sin(p) 的平方 和 1
    p1 = sin(p)**2 + sin(p)**2
    p2 = 1
    # 创建一个重写规则对象 rl，将 p1 重写为 p2，考虑变量 p 和 q
    rl = rewriterule(p1, p2, (p, q))

    # 断言应用 rl 到 sin(x) 的平方加 sin(x) 的平方 后的结果为 [1]
    assert list(rl(sin(x)**2 + sin(x)**2)) == [1]
    # 断言应用 rl 到 sin(y) 的平方加 sin(y) 的平方 后的结果为 [1]
    assert list(rl(sin(y)**2 + sin(y)**2)) == [1]

# 定义测试函数 test_Exprs_ok
def test_Exprs_ok():
    # 创建一个重写规则对象 rl，将 p 加 q 重写为 q 加 p，考虑变量 p 和 q
    rl = rewriterule(p+q, q+p, (p, q))
    # 检查 next(rl(x+y)) 的交换律属性
    next(rl(x+y)).is_commutative
    # 将 next(rl(x+y)) 转换为字符串
    str(next(rl(x+y)))

# 定义测试函数 test_condition_simple
def test_condition_simple():
    # 创建一个重写规则对象 rl，将 x 重写为 x 加 1，只对 x 小于 10 的情况适用
    rl = rewriterule(x, x+1, [x], lambda x: x < 10)
    # 断言应用 rl 到 S(15) 后的结果为空列表
    assert not list(rl(S(15)))
    # 断言应用 rl 到 S(5) 后，重建结果为 6
    assert rebuild(next(rl(S(5)))) == 6

# 定义测试函数 test_condition_multiple
def test_condition_multiple():
    # 创建一个重写规则对象 rl，将 x 加 y 重写为 x 的 y 次方，仅对整数 x 适用
    rl = rewriterule(x + y, x**y, [x,y], lambda x, y: x.is_integer)

    # 定义符号变量 a 和 b，b 为整数
    a = Symbol('a')
    b = Symbol('b', integer=True)
    # 定义表达式 expr = a 加 b
    expr = a + b
    # 断言应用 rl 到 expr 后的结果为 [b 的 a 次方]
    assert list(rl(expr)) == [b**a]

    # 定义符号变量 c 和 d，c 和 d 都是整数
    c = Symbol('c', integer=True)
    d = Symbol('d', integer=True)
    # 断言应用 rl 到 c 加 d 后的结果集合为 {c 的 d 次方, d 的 c 次方}
    assert set(rl(c + d)) == {c**d, d**c}

# 定义测试函数 test_assumptions
def test_assumptions():
    # 创建一个重写规则对象 rl，将 x 加 y 重写为 x 的 y 次方，其中 x 是整数
    rl = rewriterule(x + y, x**y, [x, y], assume=Q.integer(x))

    # 定义符号变量 a 和 b
    a, b = map(Symbol, 'ab')
    # 定义表达式 expr = a 加 b
    expr = a + b
    # 断言应用 rl 到 expr 并假设 b 是整数 后的结果为 [b 的 a 次方]
    assert list(rl(expr, Q.integer(b))) == [b**a]
```