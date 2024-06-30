# `D:\src\scipysrc\sympy\sympy\simplify\tests\test_powsimp.py`

```
# 导入 sympy 库中的各种模块和函数
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import (E, I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import sin
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.simplify.powsimp import (powdenest, powsimp)
from sympy.simplify.simplify import (signsimp, simplify)
from sympy.core.symbol import Str

# 从 sympy.abc 模块中导入变量 x, y, z, a, b
from sympy.abc import x, y, z, a, b

# 定义测试函数 test_powsimp
def test_powsimp():
    # 定义符号变量 x, y, z, n
    x, y, z, n = symbols('x,y,z,n')
    # 定义一个函数 f
    f = Function('f')
    
    # 断言语句，测试 powsimp 函数的作用
    assert powsimp( 4**x * 2**(-x) * 2**(-x) ) == 1
    assert powsimp( (-4)**x * (-2)**(-x) * 2**(-x) ) == 1

    assert powsimp(
        f(4**x * 2**(-x) * 2**(-x)) ) == f(4**x * 2**(-x) * 2**(-x))
    assert powsimp( f(4**x * 2**(-x) * 2**(-x)), deep=True ) == f(1)
    assert exp(x)*exp(y) == exp(x)*exp(y)
    assert powsimp(exp(x)*exp(y)) == exp(x + y)
    assert powsimp(exp(x)*exp(y)*2**x*2**y) == (2*E)**(x + y)
    assert powsimp(exp(x)*exp(y)*2**x*2**y, combine='exp') == \
        exp(x + y)*2**(x + y)
    assert powsimp(exp(x)*exp(y)*exp(2)*sin(x) + sin(y) + 2**x*2**y) == \
        exp(2 + x + y)*sin(x) + sin(y) + 2**(x + y)
    assert powsimp(sin(exp(x)*exp(y))) == sin(exp(x)*exp(y))
    assert powsimp(sin(exp(x)*exp(y)), deep=True) == sin(exp(x + y))
    assert powsimp(x**2*x**y) == x**(2 + y)
    # 此处保持因式分解，因为 deep=True 应该像旧版自动结合指数一样工作
    assert powsimp((1 + E*exp(E))*exp(-E), combine='exp', deep=True) == \
        (1 + exp(1 + E))*exp(-E)
    assert powsimp((1 + E*exp(E))*exp(-E), deep=True) == \
        (1 + exp(1 + E))*exp(-E)
    assert powsimp((1 + E*exp(E))*exp(-E)) == (1 + exp(1 + E))*exp(-E)
    assert powsimp((1 + E*exp(E))*exp(-E), combine='exp') == \
        (1 + exp(1 + E))*exp(-E)
    assert powsimp((1 + E*exp(E))*exp(-E), combine='base') == \
        (1 + E*exp(E))*exp(-E)
    # 定义变量 x, y 为非负数，n 为实数符号变量
    x, y = symbols('x,y', nonnegative=True)
    n = Symbol('n', real=True)
    assert powsimp(y**n * (y/x)**(-n)) == x**n
    assert powsimp(x**(x**(x*y)*y**(x*y))*y**(x**(x*y)*y**(x*y)), deep=True) \
        == (x*y)**(x*y)**(x*y)
    assert powsimp(2**(2**(2*x)*x), deep=False) == 2**(2**(2*x)*x)
    assert powsimp(2**(2**(2*x)*x), deep=True) == 2**(x*4**x)
    assert powsimp(
        exp(-x + exp(-x)*exp(-x*log(x))), deep=False, combine='exp') == \
        exp(-x + exp(-x)*exp(-x*log(x)))
    assert powsimp(
        exp(-x + exp(-x)*exp(-x*log(x))), deep=False, combine='exp') == \
        exp(-x + exp(-x)*exp(-x*log(x)))
    assert powsimp((x + y)/(3*z), deep=False, combine='exp') == (x + y)/(3*z)
    # 验证 powsimp 函数对表达式进行简化后是否符合预期结果
    assert powsimp((x/3 + y/3)/z, deep=True, combine='exp') == (x/3 + y/3)/z
    # 验证 powsimp 函数对指数表达式进行深度简化后是否正确
    assert powsimp(exp(x)/(1 + exp(x)*exp(y)), deep=True) == \
        exp(x)/(1 + exp(x + y))
    # 验证 powsimp 函数对复合指数表达式进行深度简化后是否正确
    assert powsimp(x*y**(z**x*z**y), deep=True) == x*y**(z**(x + y))
    # 验证 powsimp 函数对复合指数表达式进行深度简化后是否正确
    assert powsimp((z**x*z**y)**x, deep=True) == (z**(x + y))**x
    # 验证 powsimp 函数对复合指数表达式进行深度简化后是否正确
    assert powsimp(x*(z**x*z**y)**x, deep=True) == x*(z**(x + y))**x

    # 定义符号 p 为正数
    p = symbols('p', positive=True)
    # 验证 powsimp 函数对复合指数表达式进行简化后是否正确
    assert powsimp((1/x)**log(2)/x) == (1/x)**(1 + log(2))
    # 验证 powsimp 函数对复合指数表达式进行简化后是否正确
    assert powsimp((1/p)**log(2)/p) == p**(-1 - log(2))

    # 只有对正数底数的幂次方才能进行系数简化
    assert powsimp(2**(2*x)) == 4**x
    # 验证 powsimp 函数对幂次方表达式进行简化后是否正确
    assert powsimp((-1)**(2*x)) == (-1)**(2*x)
    # 定义符号 i 为整数
    i = symbols('i', integer=True)
    # 验证 powsimp 函数对幂次方表达式进行简化后是否正确
    assert powsimp((-1)**(2*i)) == 1
    # 验证 powsimp 函数对幂次方表达式进行简化后是否不等于预期结果 (-1)**x
    assert powsimp((-1)**(-x)) != (-1)**x  # 可能是 1/((-1)**x)，但不是
    # 使用 force=True 覆盖默认假设
    assert powsimp((-1)**(2*x), force=True) == 1

    # 有理数指数允许负数项合并
    w, n, m = symbols('w n m', negative=True)
    e = i/a  # 如果 `a` 是未知数，则不是有理数指数
    ex = w**e*n**e*m**e
    # 验证 powsimp 函数对指数表达式进行简化后是否正确
    assert powsimp(ex) == m**(i/a)*n**(i/a)*w**(i/a)
    e = i/3
    ex = w**e*n**e*m**e
    # 验证 powsimp 函数对指数表达式进行简化后是否正确
    assert powsimp(ex) == (-1)**i*(-m*n*w)**(i/3)
    e = (3 + i)/i
    ex = w**e*n**e*m**e
    # 验证 powsimp 函数对指数表达式进行简化后是否正确
    assert powsimp(ex) == (-1)**(3*e)*(-m*n*w)**e

    eq = x**(a*Rational(2, 3))
    # eq != (x**a)**(2/3)（尝试 x = -1 和 a = 3 查看）
    assert powsimp(eq).exp == eq.exp == a*Rational(2, 3)
    # powdenest 反向操作
    assert powsimp(2**(2*x)) == 4**x

    # 验证 powsimp 函数对指数表达式进行简化后是否正确
    assert powsimp(exp(p/2)) == exp(p/2)

    # issue 6368
    eq = Mul(*[sqrt(Dummy(imaginary=True)) for i in range(3)])
    # 验证 powsimp 函数对表达式进行简化后是否正确，并且结果是 Mul 类型
    assert powsimp(eq) == eq and eq.is_Mul

    # 验证 powsimp 函数对表达式进行简化后是否不变
    assert all(powsimp(e) == e for e in (sqrt(x**a), sqrt(x**2)))

    # issue 8836
    # 验证 powsimp 函数对表达式进行简化后是否正确
    assert str( powsimp(exp(I*pi/3)*root(-1,3)) ) == '(-1)**(2/3)'

    # issue 9183
    # 验证 powsimp 函数对表达式进行简化后是否正确
    assert powsimp(-0.1**x) == -0.1**x

    # issue 10095
    # 验证 powsimp 函数对表达式进行简化后是否正确
    assert powsimp((1/(2*E))**oo) == (exp(-1)/2)**oo

    # PR 13131
    eq = sin(2*x)**2*sin(2.0*x)**2
    # 验证 powsimp 函数对表达式进行简化后是否正确
    assert powsimp(eq) == eq

    # issue 14615
    # 验证 powsimp 函数对表达式进行简化后是否正确
    assert powsimp(x**2*y**3*(x*y**2)**Rational(3, 2)
        ) == x*y*(x*y**2)**Rational(5, 2)
```python`
def test_powsimp_negated_base():
    # 测试 powsimp 函数对于负数基数的处理
    assert powsimp((-x + y)/sqrt(x - y)) == -sqrt(x - y)  # 验证 (-x + y)/sqrt(x - y) 的简化结果应为 -sqrt(x - y)
    assert powsimp((-x + y)*(-z + y)/sqrt(x - y)/sqrt(z - y)) == sqrt(x - y)*sqrt(z - y)  # 验证 (-x + y)*(-z + y)/sqrt(x - y)/sqrt(z - y) 的简化结果应为 sqrt(x - y)*sqrt(z - y)
    p = symbols('p', positive=True)  # 定义一个正数符号 p
    reps = {p: 2, a: S.Half}  # 定义符号替换字典，p 替换为 2，a 替换为 1/2
    assert powsimp((-p)**a/p**a).subs(reps) == ((-1)**a).subs(reps)  # 验证 (-p)**a/p**a 的简化结果，替换符号后应为 (-1)**a
    assert powsimp((-p)**a*p**a).subs(reps) == ((-p**2)**a).subs(reps)  # 验证 (-p)**a * p**a 的简化结果，替换符号后应为 (-p**2)**a
    n = symbols('n', negative=True)  # 定义一个负数符号 n
    reps = {p: -2, a: S.Half}  # 定义符号替换字典，p 替换为 -2，a 替换为 1/2
    assert powsimp((-n)**a/n**a).subs(reps) == (-1)**(-a).subs(a, S.Half)  # 验证 (-n)**a/n**a 的简化结果，替换符号后应为 (-1)**(-a)
    assert powsimp((-n)**a*n**a).subs(reps) == ((-n**2)**a).subs(reps)  # 验证 (-n)**a * n**a 的简化结果，替换符号后应为 (-n**2)**a
    # 检查当 x 为 0 时 (-x)**a/x**a 的简化结果是否为 (-1)**a
    eq = (-x)**a/x**a
    assert powsimp(eq) == eq  # 验证等式是否成立

def test_powsimp_nc():
    # 测试 powsimp 函数在非交换运算下的表现
    x, y, z = symbols('x,y,z')  # 定义符号 x, y, z
    A, B, C = symbols('A B C', commutative=False)  # 定义符号 A, B, C，设为非交换符号

    assert powsimp(A**x*A**y, combine='all') == A**(x + y)  # 验证 A**x * A**y 的简化结果，合并所有幂次后应为 A**(x + y)
    assert powsimp(A**x*A**y, combine='base') == A**x*A**y  # 验证 A**x * A**y 的简化结果，基底合并后应为 A**x * A**y
    assert powsimp(A**x*A**y, combine='exp') == A**(x + y)  # 验证 A**x * A**y 的简化结果，指数合并后应为 A**(x + y)

    assert powsimp(A**x*B**x, combine='all') == A**x*B**x  # 验证 A**x * B**x 的简化结果，合并所有幂次后应为 A**x * B**x
    assert powsimp(A**x*B**x, combine='base') == A**x*B**x  # 验证 A**x * B**x 的简化结果，基底合并后应为 A**x * B**x
    assert powsimp(A**x*B**x, combine='exp') == A**x*B**x  # 验证 A**x * B**x 的简化结果，指数合并后应为 A**x * B**x

    assert powsimp(B**x*A**x, combine='all') == B**x*A**x  # 验证 B**x * A**x 的简化结果，合并所有幂次后应为 B**x * A**x
    assert powsimp(B**x*A**x, combine='base') == B**x*A**x  # 验证 B**x * A**x 的简化结果，基底合并后应为 B**x * A**x
    assert powsimp(B**x*A**x, combine='exp') == B**x*A**x  # 验证 B**x * A**x 的简化结果，指数合并后应为 B**x * A**x

    assert powsimp(A**x*A**y*A**z, combine='all') == A**(x + y + z)  # 验证 A**x * A**y * A**z 的简化结果，合并所有幂次后应为 A**(x + y + z)
    assert powsimp(A**x*A**y*A**z, combine='base') == A**x*A**y*A**z  # 验证 A**x * A**y * A**z 的简化结果，基底合并后应为 A**x * A**y * A**z
    assert powsimp(A**x*A**y*A**z, combine='exp') == A**(x + y + z)  # 验证 A**x * A**y * A**z 的简化结果，指数合并后应为 A**(x + y + z)

    assert powsimp(A**x*B**x*C**x, combine='all') == A**x*B**x*C**x  # 验证 A**x * B**x * C**x 的简化结果，合并所有幂次后应为 A**x * B**x * C**x
    assert powsimp(A**x*B**x*C**x, combine='base') == A**x*B**x*C**x  # 验证 A**x * B**x * C**x 的简化结果，基底合并后应为 A**x * B**x * C**x
    assert powsimp(A**x*B**x*C**x, combine='exp') == A**x*B**x*C**x  # 验证 A**x * B**x * C**x 的简化结果，指数合并后应为 A**x * B**x * C**x

    assert powsimp(B**x*A**x*C**x, combine='all') == B**x*A**x*C**x  # 验证 B**x * A**x * C**x 的简化结果，合并所有幂次后应为 B**x * A**x * C**x
    assert powsimp(B**x*A**x*C**x, combine='base') == B**x*A**x*C**x  # 验证 B**x * A**x * C**x 的简化结果，基底合并后应为 B**x * A**x * C**x
    assert powsimp(B**x*A**x*C**x, combine='exp') == B**x*A**x*C**x  # 验证 B**x * A**x * C**x 的简化结果，指数合并后应为 B**x * A**x * C**x

def test_issue_6440():
    # 测试特定问题 6440 的修复
    assert powsimp(16*2**a*8**b) == 2**(a + 3*b + 4)  # 验证 16*2**a*8**b 的简化结果应为 2**(a + 3*b + 4)

def test_powdenest():
    # 测试 powdenest 函数的功能，进行幂次化简
    x, y = symbols('x,y')  # 定义符号 x 和 y
    p, q = symbols('p q', positive=True)  # 定义正数符号 p 和 q
    i, j = symbols('i,j', integer=True)  # 定义整数符号 i 和 j

    assert powdenest(x) == x  # 验证 powdenest 对于常数 x 的处理结果应为 x
    assert powdenest(x + 2*(x**(a*Rational(2, 3)))**(3*x)) == (x +```python
# 定义一个测试函数，用于测试 powsimp 函数处理各种数学表达式的正确性
def test_powsimp_negated_base():
    # 测试当表达式为 (-x + y)/sqrt(x - y) 时，经过 powsimp 处理应得到 -sqrt(x - y)
    assert powsimp((-x + y)/sqrt(x - y)) == -sqrt(x - y)
    # 测试当表达式为 (-x + y)*(-z + y)/sqrt(x - y)/sqrt(z - y) 时，经过 powsimp 处理应得到 sqrt(x - y)*sqrt(z - y)
    assert powsimp((-x + y)*(-z + y)/sqrt(x - y)/sqrt(z - y)) == sqrt(x - y)*sqrt(z - y)
    # 定义 p 为正数，并设定变量替换字典 reps
    p = symbols('p', positive=True)
    reps = {p: 2, a: S.Half}
    # 测试当表达式为 (-p)**a/p**a 经过 powsimp 处理并进行变量替换后应为 ((-1)**a).subs(reps)
    assert powsimp((-p)**a/p**a).subs(reps) == ((-1)**a).subs(reps)
    # 测试当表达式为 (-p)**a*p**a 经过 powsimp 处理并进行变量替换后应为 ((-p**2)**a).subs(reps)
    assert powsimp((-p)**a*p**a).subs(reps) == ((-p**2)**a).subs(reps)
    # 定义 n 为负数，并设定新的变量替换字典 reps
    n = symbols('n', negative=True)
    reps = {p: -2, a: S.Half}
    # 测试当表达式为 (-n)**a/n**a 经过 powsimp 处理并进行变量替换后应为 (-1)**(-a).subs(a, S.Half)
    assert powsimp((-n)**a/n**a).subs(reps) == (-1)**(-a).subs(a, S.Half)
    # 测试当表达式为 (-n)**a*n**a 经过 powsimp 处理并进行变量替换后应为 ((-n**2)**a).subs(reps)
    assert powsimp((-n)**a*n**a).subs(reps) == ((-n**2)**a).subs(reps)
    # 当 x 为 0 时，eq 表达式为 0**a*oo**a，此时应不经过 powsimp 处理仍为 (-1)**a
    eq = (-x)**a/x**a
    assert powsimp(eq) == eq


# 定义测试函数，用于测试 powsimp 函数在非可交换符号的情况下的处理
def test_powsimp_nc():
    x, y, z = symbols('x,y,z')
    A, B, C = symbols('A B C', commutative=False)

    # 测试当表达式为 A**x*A**y 时，经过 powsimp 处理应为 A**(x + y)
    assert powsimp(A**x*A**y, combine='all') == A**(x + y)
    # 测试当表达式为 A**x*A**y 时，只合并基数时经过 powsimp 处理仍为 A**x*A**y
    assert powsimp(A**x*A**y, combine='base') == A**x*A**y
    # 测试当表达式为 A**x```python
# 定义一个测试函数，用于测试 powsimp 函数处理各种数学表达式的正确性
def test_powsimp_negated_base():
    # 测试当表达式为 (-x + y)/sqrt(x - y) 时，经过 powsimp 处理应得到 -sqrt(x - y)
    assert powsimp((-x + y)/sqrt(x - y)) == -sqrt(x - y)
    # 测试当表达式为 (-x + y)*(-z + y)/sqrt(x - y)/sqrt(z - y) 时，经过 powsimp 处理应得到 sqrt(x - y)*sqrt(z - y)
    assert powsimp((-x + y)*(-z + y)/sqrt(x - y)/sqrt(z - y)) == sqrt(x - y)*sqrt(z - y)
    # 定义 p 为正数，并设定变量替换字典 reps
    p = symbols('p', positive=True)
    reps = {p: 2, a: S.Half}
    # 测试当表达式为 (-p)**a/p**a 经过 powsimp 处理并进行变量替换后应为 ((-1)**a).subs(reps)
    assert powsimp((-p)**a/p**a).subs(reps) == ((-1)**a).subs(reps)
    # 测试当表达式为 (-p)**a*p**a 经过 powsimp 处理并进行变量替换后应为 ((-p**2)**a).subs(reps)
    assert powsimp((-p)**a*p**a).subs(reps) == ((-p**2)**a).subs(reps)
    # 定义 n 为负数，并设定新的变量替换字典 reps
    n = symbols('n', negative=True)
    reps = {p: -2, a: S.Half}
    # 测试当表达式为 (-n)**a/n**a 经过 powsimp 处理并进行变量替换后应为 (-1)**(-a).subs(a, S.Half)
    assert powsimp((-n)**a/n**a).subs(reps) == (-1)**(-a).subs(a, S.Half)
    # 测试当表达式为 (-n)**a*n**a 经过 powsimp 处理并进行变量替换后应为 ((-n**2)**a).subs(reps)
    assert powsimp((-n)**a*n**a).subs(reps) == ((-n**2)**a).subs(reps)
    # 当 x 为 0 时，eq 表达式为 0**a*oo**a，此时应不经过 powsimp 处理仍为 (-1)**a
    eq = (-x)**a/x**a
    assert powsimp(eq) == eq


# 定义测试函数，用于测试 powsimp 函数在非可交换符号的情况下的处理
def test_powsimp_nc():
    x, y, z = symbols('x,y,z')
    A, B, C = symbols('A B C', commutative=False)

    # 测试当表达式为 A**x*A**y 时，经过 powsimp 处理应为 A**(x + y)
    assert powsimp(A**x*A**y, combine='all') == A**(x + y)
    # 测试当表达式为 A**x*A**y 时，只合并基数时经过 powsimp 处理仍为 A**x*A**y
    assert powsimp(A**x*A**y, combine='base') == A**x*A**y
    # 测试当表达式为 A**x*A**y 时，只合并指数时经过 powsimp 处理应为 A**(x + y)
    assert powsimp(A**x*A**y, combine='exp') == A**(x + y)

    # 其他类似的测试情况，验证 powsimp 处理对非可交换符号的不同合并方式的处理结果

# 定义一个测试函数，测试 powsimp 函数处理数学表达式中的特定问题
def test_issue_6440():
    # 测试 powsimp 函数处理 16*2**a*8**b 应得到 2**(a + 3*b + 4)
    assert powsimp(16*2**a*8**b) == 2**(a + 3*b + 4)


# 定义一个测试函数，测试 powdenest 函数的不同情况
def test_powdenest():
    x, y = symbols('x,y')
    p, q = symbols('p q', positive=True)
    i, j = symbols('i,j', integer=True)

    # 测试 powdenest 函数处理 x 应不改变
    assert powdenest(x) == x
    # 测试 powdenest 函数处理复杂表达式不应改变
    assert powdenest(x + 2*(x**(a*Rational(2, 3)))**(3*x)) == (x + 2*(x**(a*Rational(2, 3)))**(3*x))
    # 测试 powdenest 函数处理 (exp(a*Rational(2, 3)))**(3*x) 应得到 (exp(a/3))**(6*x)
    assert powdenest((exp(a*Rational(2, 3)))**(3*x))
    # 测试 powdenest 函数处理 (x**(a*Rational(2, 3)))**(3*x) 不应改变
    assert powdenest((x**(a*Rational(2, 3)))**(3*x)) == ((x**(a*Rational(2, 3)))**(3*x))
    # 测试 powdenest 函数处理 exp(3*x*log(2)) 应得到 2**(3*x)
    assert powdenest(exp(3*x*log(2))) == 2**(3*x)
    # 测试 powdenest 函数处理 sqrt(p**2) 应得到 p
    assert powdenest(sqrt(p**2)) == p
    # 其他类似的测试情况，验证 powdenest 函数对不同表达式的处理结果
    assert powdenest(eq) == (p*q**2)**(2*i)
    assert powdenest((x**x)**(i + j))
    assert powdenest(exp(3*y*log(x))) == x**(3*y)
    assert powdenest(exp(y*(log(a) + log(b)))) == (a*b)**y
    assert powdenest(exp(3*(log(a) + log(b)))) == a**3*b**3
    assert powdenest(((x**(2*i))**(3*y))**x) == ((x**(2*i))**(3*y))**x
    assert powdenest(((x**(2*i))**(3*y))**x, force=True) == x**(6*i*x*y)
    # 对表达式进行指数幂简化，确保结果等于 x**(6*i*x*y)

    assert powdenest(((x**(a*Rational(2, 3)))**(3*y/i))**x) == \
        (((x**(a*Rational(2, 3)))**(3*y/i))**x)
    # 对表达式进行指数幂简化，不强制简化，结果应与原表达式相同

    assert powdenest((x**(2*i)*y**(4*i))**z, force=True) == (x*y**2)**(2*i*z)
    # 对表达式进行指数幂简化，确保结果等于 (x*y**2)**(2*i*z)

    assert powdenest((p**(2*i)*q**(4*i))**j) == (p*q**2)**(2*i*j)
    # 对表达式进行指数幂简化，确保结果等于 (p*q**2)**(2*i*j)

    e = ((p**(2*a))**(3*y))**x
    # 创建一个表达式 e

    assert powdenest(e) == e
    # 对表达式 e 进行指数幂简化，结果应与 e 相同

    e = ((x**2*y**4)**a)**(x*y)
    # 创建一个表达式 e

    assert powdenest(e) == e
    # 对表达式 e 进行指数幂简化，结果应与 e 相同

    e = (((x**2*y**4)**a)**(x*y))**3
    # 创建一个表达式 e

    assert powdenest(e) == ((x**2*y**4)**a)**(3*x*y)
    # 对表达式 e 进行指数幂简化，结果应与 ((x**2*y**4)**a)**(3*x*y) 相同

    assert powdenest((((x**2*y**4)**a)**(x*y)), force=True) == \
        (x*y**2)**(2*a*x*y)
    # 对表达式进行指数幂简化，确保结果等于 (x*y**2)**(2*a*x*y)，强制简化

    assert powdenest((((x**2*y**4)**a)**(x*y))**3, force=True) == \
        (x*y**2)**(6*a*x*y)
    # 对表达式进行指数幂简化，确保结果等于 (x*y**2)**(6*a*x*y)，强制简化

    assert powdenest((x**2*y**6)**i) != (x*y**3)**(2*i)
    # 对表达式进行指数幂简化，结果不应等于 (x*y**3)**(2*i)

    x, y = symbols('x,y', positive=True)
    # 声明符号 x 和 y，要求为正数

    assert powdenest((x**2*y**6)**i) == (x*y**3)**(2*i)
    # 对表达式进行指数幂简化，确保结果等于 (x*y**3)**(2*i)

    assert powdenest((x**(i*Rational(2, 3))*y**(i/2))**(2*i)) == (x**Rational(4, 3)*y)**(i**2)
    # 对表达式进行指数幂简化，确保结果等于 (x**Rational(4, 3)*y)**(i**2)

    assert powdenest(sqrt(x**(2*i)*y**(6*i))) == (x*y**3)**i
    # 对表达式进行指数幂简化，确保结果等于 (x*y**3)**i

    assert powdenest(4**x) == 2**(2*x)
    # 对表达式进行指数幂简化，确保结果等于 2**(2*x)

    assert powdenest((4**x)**y) == 2**(2*x*y)
    # 对表达式进行指数幂简化，确保结果等于 2**(2*x*y)

    assert powdenest(4**x*y) == 2**(2*x)*y
    # 对表达式进行指数幂简化，确保结果等于 2**(2*x)*y
def test_powdenest_polar():
    # 定义符号 x, y, z 为极坐标系
    x, y, z = symbols('x y z', polar=True)
    # 定义符号 a, b, c
    a, b, c = symbols('a b c')
    # 测试极坐标系下的幂函数展开
    assert powdenest((x*y*z)**a) == x**a*y**a*z**a
    assert powdenest((x**a*y**b)**c) == x**(a*c)*y**(b*c)
    assert powdenest(((x**a)**b*y**c)**c) == x**(a*b*c)*y**(c**2)


def test_issue_5805():
    # 计算给定表达式的幂函数展开
    arg = ((gamma(x)*hyper((), (), x))*pi)**2
    assert powdenest(arg) == (pi*gamma(x)*hyper((), (), x))**2
    assert arg.is_positive is None


def test_issue_9324_powsimp_on_matrix_symbol():
    # 创建一个矩阵符号 M
    M = MatrixSymbol('M', 10, 10)
    # 对矩阵进行幂函数简化操作，不递归到元素级别
    expr = powsimp(M, deep=True)
    assert expr == M
    # 断言矩阵表达式的第一个元素为字符串 'M'
    assert expr.args[0] == Str('M')


def test_issue_6367():
    # 定义复杂表达式 z
    z = -5*sqrt(2)/(2*sqrt(2*sqrt(29) + 29)) + sqrt(-sqrt(29)/29 + S.Half)
    # 断言对 z 的标准化形式进行幂函数简化后得到 0
    assert Mul(*[powsimp(a) for a in Mul.make_args(z.normal())]) == 0
    assert powsimp(z.normal()) == 0
    assert simplify(z) == 0
    assert powsimp(sqrt(2 + sqrt(3))*sqrt(2 - sqrt(3)) + 1) == 2
    assert powsimp(z) != 0


def test_powsimp_polar():
    # 导入相关函数库
    from sympy.functions.elementary.complexes import polar_lift
    from sympy.functions.elementary.exponential import exp_polar
    # 定义符号 x, y, z
    x, y, z = symbols('x y z')
    # 定义极坐标系下的符号 p, q, r
    p, q, r = symbols('p q r', polar=True)

    # 断言极坐标系下的幂函数简化
    assert (polar_lift(-1))**(2*x) == exp_polar(2*pi*I*x)
    assert powsimp(p**x * q**x) == (p*q)**x
    assert p**x * (1/p)**x == 1
    assert (1/p)**x == p**(-x)

    # 断言指数函数的乘积
    assert exp_polar(x)*exp_polar(y) == exp_polar(x)*exp_polar(y)
    assert powsimp(exp_polar(x)*exp_polar(y)) == exp_polar(x + y)
    assert powsimp(exp_polar(x)*exp_polar(y)*p**x*p**y) == \
        (p*exp_polar(1))**(x + y)
    assert powsimp(exp_polar(x)*exp_polar(y)*p**x*p**y, combine='exp') == \
        exp_polar(x + y)*p**(x + y)
    assert powsimp(
        exp_polar(x)*exp_polar(y)*exp_polar(2)*sin(x) + sin(y) + p**x*p**y) \
        == p**(x + y) + sin(x)*exp_polar(2 + x + y) + sin(y)
    assert powsimp(sin(exp_polar(x)*exp_polar(y))) == \
        sin(exp_polar(x)*exp_polar(y))
    assert powsimp(sin(exp_polar(x)*exp_polar(y)), deep=True) == \
        sin(exp_polar(x + y))


def test_issue_5728():
    # 定义数学表达式
    b = x*sqrt(y)
    a = sqrt(b)
    c = sqrt(sqrt(x)*y)
    # 断言幂函数简化结果
    assert powsimp(a*b) == sqrt(b)**3
    assert powsimp(a*b**2*sqrt(y)) == sqrt(y)*a**5
    assert powsimp(a*x**2*c**3*y) == c**3*a**5
    assert powsimp(a*x*c**3*y**2) == c**7*a
    assert powsimp(x*c**3*y**2) == c**7
    assert powsimp(x*c**3*y) == x*y*c**3
    assert powsimp(sqrt(x)*c**3*y) == c**5
    assert powsimp(sqrt(x)*a**3*sqrt(y)) == sqrt(x)*sqrt(y)*a**3
    assert powsimp(Mul(sqrt(x)*c**3*sqrt(y), y, evaluate=False)) == \
        sqrt(x)*sqrt(y)**3*c**3
    assert powsimp(a**2*a*x**2*y) == a**7

    # 断言符号幂函数的运算结果
    b = x**y*y
    a = b*sqrt(b)
    assert a.is_Mul is True
    assert powsimp(a) == sqrt(b)**3

    # 断言指数函数的结果
    a = x*exp(y*Rational(2, 3))
    assert powsimp(a*sqrt(a)) == sqrt(a)**3
    assert powsimp(a**2*sqrt(a)) == sqrt(a)**5
    assert powsimp(a**2*sqrt(sqrt(a))) == sqrt(sqrt(a))**9
    # 定义四个符号变量 n1, n2, n3, n4，并指定它们为负数
    n1, n2, n3, n4 = symbols('n1 n2 n3 n4', negative=True)
    
    # 使用 powsimp 函数验证平方根的乘积简化后是否等于预期结果
    assert (powsimp(sqrt(n1)*sqrt(n2)*sqrt(n3)) ==
        -I*sqrt(-n1)*sqrt(-n2)*sqrt(-n3))
    
    # 使用 powsimp 函数验证多个根号乘积的简化是否等于预期结果
    assert (powsimp(root(n1, 3)*root(n2, 3)*root(n3, 3)*root(n4, 3)) ==
        -(-1)**Rational(1, 3)*
        (-n1)**Rational(1, 3)*(-n2)**Rational(1, 3)*(-n3)**Rational(1, 3)*(-n4)**Rational(1, 3))
# 定义测试函数，用于验证问题 10195
def test_issue_10195():
    # 创建整数符号变量 a
    a = Symbol('a', integer=True)
    # 创建偶数且非零符号变量 l
    l = Symbol('l', even=True, nonzero=True)
    # 创建奇数符号变量 n
    n = Symbol('n', odd=True)
    # 计算表达式 e_x
    e_x = (-1)**(n/2 - S.Half) - (-1)**(n*Rational(3, 2) - S.Half)
    # 断言简化幂运算后的结果
    assert powsimp((-1)**(l/2)) == I**l
    assert powsimp((-1)**(n/2)) == I**n
    assert powsimp((-1)**(n*Rational(3, 2))) == -I**n
    assert powsimp(e_x) == (-1)**(n/2 - S.Half) + (-1)**(n*Rational(3, 2) +
            S.Half)
    assert powsimp((-1)**(a*Rational(3, 2))) == (-I)**a

# 定义测试函数，用于验证问题 15709
def test_issue_15709():
    # 断言简化幂运算后的结果
    assert powsimp(3**x*Rational(2, 3)) == 2*3**(x-1)
    assert powsimp(2*3**x/3) == 2*3**(x-1)

# 定义测试函数，用于验证问题 11981
def test_issue_11981():
    # 创建不可交换符号变量 x, y
    x, y = symbols('x y', commutative=False)
    # 断言简化幂运算后的结果
    assert powsimp((x*y)**2 * (y*x)**2) == (x*y)**2 * (y*x)**2

# 定义测试函数，用于验证问题 17524
def test_issue_17524():
    # 创建实数符号变量 a
    a = symbols("a", real=True)
    # 创建表达式 e
    e = (-1 - a**2)*sqrt(1 + a**2)
    # 断言符号简化和幂简化后的结果相同
    assert signsimp(powsimp(e)) == signsimp(e) == -(a**2 + 1)**(S(3)/2)

# 定义测试函数，用于验证问题 19627
def test_issue_19627():
    # 断言强制幂展开后的结果
    assert powdenest(sqrt(sin(x)**2), force=True) == sin(x)
    assert powdenest((x**(S.Half/y))**(2*y), force=True) == x
    # 导入模块并创建表达式 e
    from sympy.core.function import expand_power_base
    e = 1 - a
    # 创建复合表达式 expr
    expr = (exp(z/e)*x**(b/e)*y**((1 - b)/e))**e
    # 断言幂展开和强制幂展开后的结果相同
    assert powdenest(expand_power_base(expr, force=True), force=True
        ) == x**b*y**(1 - b)*exp(z)

# 定义测试函数，用于验证问题 22546
def test_issue_22546():
    # 创建正数符号变量 p1, p2
    p1, p2 = symbols('p1, p2', positive=True)
    # 简化幂运算后的参考结果 ref
    ref = powsimp(p1**z/p2**z)
    # 创建符号表达式 e
    e = z + 1
    # 用符号替换并断言结果是幂对象
    ans = ref.subs(z, e)
    assert ans.is_Pow
    # 断言简化幂运算后的结果
    assert powsimp(p1**e/p2**e) == ans
    # 创建整数符号变量 i
    i = symbols('i', integer=True)
    # 简化幂运算后的参考结果 ref
    ref = powsimp(x**i/y**i)
    # 创建符号表达式 e
    e = i + 1
    # 用符号替换并断言结果是幂对象
    ans = ref.subs(i, e)
    assert ans.is_Pow
    # 断言简化幂运算后的结果
    assert powsimp(x**e/y**e) == ans
```