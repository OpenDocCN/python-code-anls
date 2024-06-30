# `D:\src\scipysrc\sympy\sympy\core\tests\test_expand.py`

```
# 导入必要的符号和函数模块
from sympy.core.expr import unchanged
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational as R, pi)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.series.order import O
from sympy.simplify.radsimp import expand_numer
from sympy.core.function import (expand, expand_multinomial,
    expand_power_base, expand_log)

# 导入测试相关的模块和函数
from sympy.testing.pytest import raises
from sympy.core.random import verify_numerically

# 导入符号变量 x, y, z
from sympy.abc import x, y, z


# 定义测试函数 test_expand_no_log，测试 log=False 时的展开效果
def test_expand_no_log():
    assert (
        (1 + log(x**4))**2).expand(log=False) == 1 + 2*log(x**4) + log(x**4)**2
    assert ((1 + log(x**4))*(1 + log(x**3))).expand(
        log=False) == 1 + log(x**4) + log(x**3) + log(x**4)*log(x**3)


# 定义测试函数 test_expand_no_multinomial，测试 multinomial=False 时的展开效果
def test_expand_no_multinomial():
    assert ((1 + x)*(1 + (1 + x)**4)).expand(multinomial=False) == \
        1 + x + (1 + x)**4 + x*(1 + x)**4


# 定义测试函数 test_expand_negative_integer_powers，测试负整数次幂展开的效果
def test_expand_negative_integer_powers():
    expr = (x + y)**(-2)
    assert expr.expand() == 1 / (2*x*y + x**2 + y**2)
    assert expr.expand(multinomial=False) == (x + y)**(-2)
    expr = (x + y)**(-3)
    assert expr.expand() == 1 / (3*x*x*y + 3*x*y*y + x**3 + y**3)
    assert expr.expand(multinomial=False) == (x + y)**(-3)
    expr = (x + y)**(2) * (x + y)**(-4)
    assert expr.expand() == 1 / (2*x*y + x**2 + y**2)
    assert expr.expand(multinomial=False) == (x + y)**(-2)


# 定义测试函数 test_expand_non_commutative，测试非交换符号展开的效果
def test_expand_non_commutative():
    A = Symbol('A', commutative=False)
    B = Symbol('B', commutative=False)
    C = Symbol('C', commutative=False)
    a = Symbol('a')
    b = Symbol('b')
    i = Symbol('i', integer=True)
    n = Symbol('n', negative=True)
    m = Symbol('m', negative=True)
    p = Symbol('p', polar=True)
    np = Symbol('p', polar=False)

    assert (C*(A + B)).expand() == C*A + C*B
    assert (C*(A + B)).expand() != A*C + B*C
    assert ((A + B)**2).expand() == A**2 + A*B + B*A + B**2
    assert ((A + B)**3).expand() == (A**2*B + B**2*A + A*B**2 + B*A**2 +
                                     A**3 + B**3 + A*B*A + B*A*B)
    # issue 6219
    assert ((a*A*B*A**-1)**2).expand() == a**2*A*B**2/A
    # 注意，(a*A*B*A**-1)**2 会自动转换为 a**2*(A*B*A**-1)**2
    assert ((a*A*B*A**-1)**2).expand(deep=False) == a**2*(A*B*A**-1)**2
    assert ((a*A*B*A**-1)**2).expand() == a**2*(A*B**2*A**-1)
    assert ((a*A*B*A**-1)**2).expand(force=True) == a**2*A*B**2*A**(-1)
    assert ((a*A*B)**2).expand() == a**2*A*B*A*B
    assert ((a*A)**2).expand() == a**2*A**2
    assert ((a*A*B)**i).expand() == a**i*(A*B)**i
    assert ((a*A*(B*(A*B/A)**2))**i).expand() == a**i*(A*B*A*B**2/A)**i
    # issue 6558
    assert (A*B*(A*B)**-1).expand() == 1
    assert ((a*A)**i).expand() == a**i*A**i


这样，每一行代码都被适当地注释说明了其作用和功能。
    # 断言：展开表达式并检查是否与预期相等
    assert ((a*A*B*A**-1)**3).expand() == a**3*A*B**3/A
    
    # 断言：展开表达式并检查是否与预期相等
    assert ((a*A*B*A*B/A)**3).expand() == \
        a**3*A*B*(A*B**2)*(A*B**2)*A*B*A**(-1)
    
    # 断言：展开表达式并检查是否与预期相等
    assert ((a*A*B*A*B/A)**-2).expand() == \
        A*B**-1*A**-1*B**-2*A**-1*B**-1*A**-1/a**2
    
    # 断言：展开表达式并检查是否与预期相等
    assert ((a*b*A*B*A**-1)**i).expand() == a**i*b**i*(A*B/A)**i
    
    # 断言：展开表达式并检查是否与预期相等
    assert ((a*(a*b)**i)**i).expand() == a**i*a**(i**2)*b**(i**2)
    
    # 计算表达式 e = (a * (1/a) * A * B)^2，不进行求值
    e = Pow(Mul(a, 1/a, A, B, evaluate=False), S(2), evaluate=False)
    
    # 断言：展开表达式并检查是否与预期相等
    assert e.expand() == A*B*A*B
    
    # 断言：展开表达式并检查是否与预期相等
    assert sqrt(a*(A*b)**i).expand() == sqrt(a*b**i*A**i)
    
    # 断言：展开表达式并检查是否与预期相等
    assert (sqrt(-a)**a).expand() == sqrt(-a)**a
    
    # 断言：展开表达式并检查是否与预期相等
    assert expand((-2*n)**(i/3)) == 2**(i/3)*(-n)**(i/3)
    
    # 断言：展开表达式并检查是否与预期相等
    assert expand((-2*n*m)**(i/a)) == (-2)**(i/a)*(-n)**(i/a)*(-m)**(i/a)
    
    # 断言：展开表达式并检查是否与预期相等
    assert expand((-2*a*p)**b) == 2**b*p**b*(-a)**b
    
    # 断言：展开表达式并检查是否与预期相等
    assert expand((-2*a*np)**b) == 2**b*(-a*np)**b
    
    # 断言：展开表达式并检查是否与预期相等
    assert expand(sqrt(A*B)) == sqrt(A*B)
    
    # 断言：展开表达式并检查是否与预期相等
    assert expand(sqrt(-2*a*b)) == sqrt(2)*sqrt(-a*b)
# 定义函数 test_expand_radicals，用于测试展开根式表达式的功能
def test_expand_radicals():
    # 设置 a 为 (x + y) 的平方根
    a = (x + y)**R(1, 2)

    # 断言 (a**1).expand() 展开后应该等于 a 本身
    assert (a**1).expand() == a
    # 断言 (a**3).expand() 展开后应该等于 x*a + y*a
    assert (a**3).expand() == x*a + y*a
    # 断言 (a**5).expand() 展开后应该等于 x**2*a + 2*x*y*a + y**2*a
    assert (a**5).expand() == x**2*a + 2*x*y*a + y**2*a

    # 断言 (1/a**1).expand() 展开后应该等于 1/a
    assert (1/a**1).expand() == 1/a
    # 断言 (1/a**3).expand() 展开后应该等于 1/(x*a + y*a)
    assert (1/a**3).expand() == 1/(x*a + y*a)
    # 断言 (1/a**5).expand() 展开后应该等于 1/(x**2*a + 2*x*y*a + y**2*a)
    assert (1/a**5).expand() == 1/(x**2*a + 2*x*y*a + y**2*a)

    # 重新设置 a 为 (x + y) 的立方根
    a = (x + y)**R(1, 3)

    # 断言 (a**1).expand() 展开后应该等于 a 本身
    assert (a**1).expand() == a
    # 断言 (a**2).expand() 展开后应该等于 a**2
    assert (a**2).expand() == a**2
    # 断言 (a**4).expand() 展开后应该等于 x*a + y*a
    assert (a**4).expand() == x*a + y*a
    # 断言 (a**5).expand() 展开后应该等于 x*a**2 + y*a**2
    assert (a**5).expand() == x*a**2 + y*a**2
    # 断言 (a**7).expand() 展开后应该等于 x**2*a + 2*x*y*a + y**2*a
    assert (a**7).expand() == x**2*a + 2*x*y*a + y**2*a


# 定义函数 test_expand_modulus，用于测试在模数下展开表达式的功能
def test_expand_modulus():
    # 断言 ((x + y)**11).expand(modulus=11) 展开后应该等于 x**11 + y**11
    assert ((x + y)**11).expand(modulus=11) == x**11 + y**11
    # 断言 ((x + sqrt(2)*y)**11).expand(modulus=11) 展开后应该等于 x**11 + 10*sqrt(2)*y**11
    assert ((x + sqrt(2)*y)**11).expand(modulus=11) == x**11 + 10*sqrt(2)*y**11
    # 断言 (x + y/2).expand(modulus=1) 展开后应该等于 y/2
    assert (x + y/2).expand(modulus=1) == y/2

    # 断言 ((x + y)**11).expand(modulus=0) 会抛出 ValueError 异常
    raises(ValueError, lambda: ((x + y)**11).expand(modulus=0))
    # 断言 ((x + y)**11).expand(modulus=x) 会抛出 ValueError 异常
    raises(ValueError, lambda: ((x + y)**11).expand(modulus=x))


# 定义函数 test_issue_5743，用于测试特定问题的解决方案
def test_issue_5743():
    # 断言 x*sqrt(x + y)*(1 + sqrt(x + y)) 展开后应该等于 x**2 + x*y + x*sqrt(x + y)
    assert (x*sqrt(x + y)*(1 + sqrt(x + y))).expand() == x**2 + x*y + x*sqrt(x + y)
    # 断言 x*sqrt(x + y)*(1 + x*sqrt(x + y)) 展开后应该等于 x**3 + x**2*y + x*sqrt(x + y)
    assert (x*sqrt(x + y)*(1 + x*sqrt(x + y))).expand() == x**3 + x**2*y + x*sqrt(x + y)


# 定义函数 test_expand_frac，用于测试在分数形式下展开表达式的功能
def test_expand_frac():
    # 断言 expand((x + y)*y/x/(x + 1), frac=True) 展开后应该等于 (x*y + y**2)/(x**2 + x)
    assert expand((x + y)*y/x/(x + 1), frac=True) == (x*y + y**2)/(x**2 + x)
    # 断言 expand((x + y)*y/x/(x + 1), numer=True) 展开后应该等于 (x*y + y**2)/(x*(x + 1))
    assert expand((x + y)*y/x/(x + 1), numer=True) == (x*y + y**2)/(x*(x + 1))
    # 断言 expand((x + y)*y/x/(x + 1), denom=True) 展开后应该等于 y*(x + y)/(x**2 + x)
    assert expand((x + y)*y/x/(x + 1), denom=True) == y*(x + y)/(x**2 + x)
    
    # 设置 eq 为 (x + 1)**2/y
    eq = (x + 1)**2/y
    # 断言 expand_numer(eq, multinomial=False) 展开后应该等于 eq 本身
    assert expand_numer(eq, multinomial=False) == eq
    
    # 设置 eq 为 (exp(x*z) - exp(y*z))/exp(z*(x + y))
    eq = (exp(x*z) - exp(y*z))/exp(z*(x + y))
    # 设置 ans 为 exp(-y*z) - exp(-x*z)
    ans = exp(-y*z) - exp(-x*z)
    # 断言 eq.expand(numer=True) 展开后不等于 ans
    assert eq.expand(numer=True) != ans
    # 断言 eq.expand(numer=True, exact=True) 展开后等于 ans
    assert eq.expand(numer=True, exact=True) == ans
    # 断言 expand_numer(eq) 展开后不等于 ans
    assert expand_numer(eq) != ans
    # 断言 expand_numer(eq, exact=True) 展开后等于 ans
    assert expand_numer(eq, exact=True) == ans


# 定义函数 test_issue_6121，用于测试特定问题的解决方案
def test_issue_6121():
    # 设置 eq 为 -I*exp(-3*I*pi/4)/(4*pi**(S(3)/2)*sqrt(x))
    eq = -I*exp(-3*I*pi/4)/(4*pi**(S(3)/2)*sqrt(x))
    # 断言 eq.expand(complex=True) 不会导致无限递归
    assert eq.expand(complex=True)
    
    # 设置 eq 为 -I*exp(-3*I*pi/4)/(4*pi**(R(3, 2))*sqrt(x))
    eq = -I*exp(-3*I*pi/4)/(4*pi**(R(3, 2))*sqrt(x))
    # 断言 eq.expand(complex=True) 不会导致无限递归
    assert eq.expand(complex=True)


# 定义函数 test_expand_power_base，用于测试幂指数基数展开的功能
def test_expand_power_base():
    # 断言 expand_power_base((x*y*z)**4) 展开后应该等于 x**4*y**4*z**4
    assert expand_power_base((x*y*z)**4) == x**4*y**4*z**4
    # 断言 expand_power_base((x*y*z)**x).is_Pow 返回 True
    assert expand_power_base((x*y*z)**x).is_Pow
    # 断言 expand_power_base((x*y*z)**x, force=True) 展开后应该等于 x**x*y**x*z**x
    assert expand_power_base((x*y*z)**x, force=True) == x**x*y**x*z**
    # 使用 assert 语句来检查函数 expand_power_base 的输出是否符合预期
    assert expand_power_base(
        # 计算 (exp(x)*exp(y))**z，然后应用 expand_power_base 函数进行展开
        (exp(x)*exp(y))**z, force=True) == 
        # 期望展开结果为 exp(x)**z * exp(y)**z
        exp(x)**z * exp(y)**z
def test_expand_arit():
    # 创建符号变量a, b, c
    a = Symbol("a")
    b = Symbol("b", positive=True)
    c = Symbol("c")

    # 创建整数对象p
    p = R(5)
    
    # 定义表达式e，展示符号a和b的加法，乘以符号c
    e = (a + b)*c
    
    # 断言表达式e与c*(a + b)相等
    assert e == c*(a + b)
    
    # 断言展开后的表达式e减去a*c减去b*c等于整数对象R(0)
    assert (e.expand() - a*c - b*c) == R(0)
    
    # 重新定义表达式e为(a + b)的平方
    e = (a + b)*(a + b)
    
    # 断言表达式e与(a + b)的平方相等
    assert e == (a + b)**2
    
    # 断言展开后的表达式e等于2*a*b加上a的平方加上b的平方
    assert e.expand() == 2*a*b + a**2 + b**2
    
    # 重新定义表达式e为(a + b)乘以(a + b)的R(2)次方
    e = (a + b)*(a + b)**R(2)
    
    # 断言表达式e与(a + b)的R(3)次方相等
    assert e == (a + b)**3
    
    # 断言展开后的表达式e等于3*b*a的平方加上3*a*b的平方加上a的R(3)次方加上b的R(3)次方
    assert e.expand() == 3*b*a**2 + 3*a*b**2 + a**3 + b**3
    
    # 重新定义表达式e为(a + b)乘以(a + c)乘以(b + c)
    e = (a + b)*(a + c)*(b + c)
    
    # 断言表达式e与(a + c)乘以(a + b)乘以(b + c)相等
    assert e == (a + c)*(a + b)*(b + c)
    
    # 断言展开后的表达式e等于2*a*b*c加上b*a的平方加上c*a的平方加上b*c的平方加上a*c的平方加上c*b的平方加上a*b的平方
    assert e.expand() == 2*a*b*c + b*a**2 + c*a**2 + b*c**2 + a*c**2 + c*b**2 + a*b**2
    
    # 重新定义表达式e为(a + R(1))的p次方
    e = (a + R(1))**p
    
    # 断言表达式e与(1 + a)的5次方相等
    assert e == (1 + a)**5
    
    # 断言展开后的表达式e等于1加上5*a加上10*a的平方加上10*a的R(3)次方加上5*a的R(4)次方加上a的R(5)次方
    assert e.expand() == 1 + 5*a + 10*a**2 + 10*a**3 + 5*a**4 + a**5
    
    # 重新定义表达式e为(a + b + c)乘以(a + c + p)
    e = (a + b + c)*(a + c + p)
    
    # 断言表达式e与(5 + a + c)乘以(a + b + c)相等
    assert e == (5 + a + c)*(a + b + c)
    
    # 断言展开后的表达式e等于5*a加上5*b加上5*c加上2*a*c加上b*c加上a*b加上a的平方加上c的平方
    assert e.expand() == 5*a + 5*b + 5*c + 2*a*c + b*c + a*b + a**2 + c**2
    
    # 创建符号变量x
    x = Symbol("x")
    
    # 创建表达式s，计算exp(x*x)减去1
    s = exp(x*x) - 1
    
    # 计算s在x等于0，展开到6次幂，除以x的平方
    e = s.nseries(x, 0, 6)/x**2
    
    # 断言展开后的表达式e等于1加上x的平方除以2加上O(x的四次方)
    assert e.expand() == 1 + x**2/2 + O(x**4)
    
    # 重新定义表达式e为x乘以(y + z)的(x乘以(y + z))次方，乘以(x + y)
    e = (x*(y + z))**(x*(y + z))*(x + y)
    
    # 断言展开后的表达式e，使用power_exp=False和power_base=False，等于x乘以(x乘以y加上x乘以z)的(x乘以y加上x乘以z)次方加上y乘以(x乘以y加上x乘以z)的(x乘以y加上x乘以z)次方
    assert e.expand(power_exp=False, power_base=False) == x*(x*y + x*z)**(x*y + x*z) + y*(x*y + x*z)**(x*y + x*z)
    
    # 断言展开后的表达式e，使用power_exp=False和power_base=False和deep=False，等于x乘以(x乘以(y + z))的(x乘以(y + z))次方加上y乘以(x乘以(y + z))的(x乘以(y + z))次方
    assert e.expand(power_exp=False, power_base=False, deep=False) == x*(x*(y + z))**(x*(y + z)) + y*(x*(y + z))**(x*(y + z))
    
    # 重新定义表达式e为x乘以(x加上(y + 1)的平方)
    e = x * (x + (y + 1)**2)
    
    # 断言展开后的表达式e，使用deep=False，等于x的平方加上x乘以(y + 1)的平方
    assert e.expand(deep=False) == x**2 + x*(y + 1)**2
    
    # 重新定义表达式e为(x乘以(y + z))的z次方
    e = (x*(y + z))**z
    
    # 断言展开后的表达式e，使用power_base=True和mul=True和deep=True，等于x的z次方乘以(y + z)的z次方或者(x乘以y加上x乘以z)的z次方
    assert e.expand(power_base=True, mul=True, deep=True) in [x**z*(y + z)**z, (x*y + x*z)**z]
    
    # 断言展开后的表达式((2乘以y)的z次方)等于2的z次方乘以y的z次方
    assert ((2*y)**z).expand() == 2**z*y**z
    
    # 创建符号变量p，限定为正数
    p = Symbol('p', positive=True)
    
    # 断言对sqrt(-x)展开后的表达式是幂对象
    assert sqrt(-x).expand().is_Pow
    
    # 断言对sqrt(-x)使用force=True展开后的表达式等于虚数单位乘以sqrt(x)
    assert sqrt(-x).expand(force=True) == I*sqrt(x)
    
    # 断言展开后的表达式((2乘以y乘以p)的z次方)等于2的z次方乘以p的z次方乘以y的z次方
    assert ((2*y*p)**z).expand() == 2**z*p**z*y**z
    
    # 断言展开后的表达式((2乘以y乘以p乘以x)的z次方)等于2的z次方乘以p的z次方乘以(x乘以y)的z次方
    assert ((2*y*p*x)**z).expand() == 2**z*p**z*(x*y)**z
    
    # 断言展开后的表达式((2乘以y乘
    # 创建符号变量 'a'，表示代数中的符号 a
    a = Symbol('a')
    
    # 创建符号变量 'b'，表示代数中的符号 b
    b = Symbol('b')
    
    # 创建代数表达式 p，表示 (a + b) 的平方
    p = (a + b)**2
    
    # 使用代数表达式方法 expand() 展开 p，并与预期的展开式 a**2 + b**2 + 2*a*b 进行断言比较
    assert p.expand() == a**2 + b**2 + 2*a*b
    
    # 重新赋值 p，创建一个复杂的代数表达式，展开后与预期的 9 + 4*a**2 + 12*a 进行断言比较
    p = (1 + 2*(1 + a))**2
    assert p.expand() == 9 + 4*(a**2) + 12*a
    
    # 重新赋值 p，创建代数表达式 2**(a + b)，展开后与预期的 2**a * 2**b 进行断言比较
    p = 2**(a + b)
    assert p.expand() == 2**a * 2**b
    
    # 创建非交换符号变量 'A'，表示在非交换代数环中的符号 A
    A = Symbol('A', commutative=False)
    
    # 创建非交换符号变量 'B'，表示在非交换代数环中的符号 B
    B = Symbol('B', commutative=False)
    
    # 断言 2**(A + B) 在非交换环中的展开结果为其本身
    assert (2**(A + B)).expand() == 2**(A + B)
    
    # 断言 A**(a + b) 在非交换环中的展开结果不等于 A**(a + b)
    assert (A**(a + b)).expand() != A**(a + b)
def test_issues_5919_6830():
    # issue 5919: 计算表达式 z
    n = -1 + 1/x
    z = n/x/(-n)**2 - 1/n/x
    # 断言 z 的展开结果
    assert expand(z) == 1/(x**2 - 2*x + 1) - 1/(x - 2 + 1/x) - 1/(-x + 1)

    # issue 6830: 计算多项式展开
    p = (1 + x)**2
    # 断言多项式展开结果
    assert expand_multinomial((1 + x*p)**2) == (
        x**2*(x**4 + 4*x**3 + 6*x**2 + 4*x + 1) + 2*x*(x**2 + 2*x + 1) + 1)
    # 断言带参数的多项式展开结果
    assert expand_multinomial((1 + (y + x)*p)**2) == (
        2*((x + y)*(x**2 + 2*x + 1)) + (x**2 + 2*x*y + y**2)*
        (x**4 + 4*x**3 + 6*x**2 + 4*x + 1) + 1)
    
    # 定义符号 A，并计算相关多项式展开
    A = Symbol('A', commutative=False)
    p = (1 + A)**2
    assert expand_multinomial((1 + x*p)**2) == (
        x**2*(1 + 4*A + 6*A**2 + 4*A**3 + A**4) + 2*x*(1 + 2*A + A**2) + 1)
    assert expand_multinomial((1 + (y + x)*p)**2) == (
        (x + y)*(1 + 2*A + A**2)*2 + (x**2 + 2*x*y + y**2)*
        (1 + 4*A + 6*A**2 + 4*A**3 + A**4) + 1)
    assert expand_multinomial((1 + (y + x)*p)**3) == (
        (x + y)*(1 + 2*A + A**2)*3 + (x**2 + 2*x*y + y**2)*(1 + 4*A +
        6*A**2 + 4*A**3 + A**4)*3 + (x**3 + 3*x**2*y + 3*x*y**2 + y**3)*(1 + 6*A
        + 15*A**2 + 20*A**3 + 15*A**4 + 6*A**5 + A**6) + 1)
    
    # unevaluate powers
    eq = (Pow((x + 1)*((A + 1)**2), 2, evaluate=False))
    # 断言展开结果，此处基数不是 Add 类型，因此不会进一步展开
    assert expand_multinomial(eq) == \
        (x**2 + 2*x + 1)*(1 + 4*A + 6*A**2 + 4*A**3 + A**4)
    # 在这种情况下，展开基数是 Add 类型，因此会进行展开
    eq = (Pow(((A + 1)**2), 2, evaluate=False))
    assert expand_multinomial(eq) == 1 + 4*A + 6*A**2 + 4*A**3 + A**4

    # coverage: 测试数值验证函数 ok
    def ok(a, b, n):
        e = (a + I*b)**n
        return verify_numerically(e, expand_multinomial(e))

    for a in [2, S.Half]:
        for b in [3, R(1, 3)]:
            for n in range(2, 6):
                assert ok(a, b, n)

    # 断言多项式展开结果，考虑高阶项
    assert expand_multinomial((x + 1 + O(z))**2) == \
        1 + 2*x + x**2 + O(z)
    assert expand_multinomial((x + 1 + O(z))**3) == \
        1 + 3*x + 3*x**2 + x**3 + O(z)

    # 断言对数展开结果
    assert expand(log(t**2) - log(t**2/4) - 2*log(2)) == 0
    assert expand_log(log(7*6)/log(6)) == 1 + log(7)/log(6)
    # 断言阶乘对数展开结果
    b = factorial(10)
    assert expand_log(log(7*b**4)/log(b)
        ) == 4 + log(7)/log(b)


def test_issue_23952():
    # 断言指数展开结果
    assert (x**(y + z)).expand(force=True) == x**y*x**z
    # 定义整数符号并进行断言
    one = Symbol('1', integer=True, prime=True, odd=True, positive=True)
    two = Symbol('2', integer=True, prime=True, even=True)
    e = two - one
    # 对于 b 在 (0, x) 中的每一个值，进行如下断言检查
    for b in (0, x):
        # 对于指数为 e 的幂运算，应保持不变性
        assert unchanged(Pow, b, e)  # power_exp
        # 对于指数为 -e 的幂运算，应保持不变性
        assert unchanged(Pow, b, -e)  # power_exp
        # 对于指数为 y - x 的幂运算，应保持不变性
        assert unchanged(Pow, b, y - x)  # power_exp
        # 对于指数为 3 - x 的幂运算，应保持不变性
        assert unchanged(Pow, b, 3 - x)  # multinomial
        # 对于 b 的 e 次幂展开后，应该仍然是 Pow 对象
        assert (b**e).expand().is_Pow  # power_exp
        # 对于 b 的 -e 次幂展开后，应该仍然是 Pow 对象
        assert (b**-e).expand().is_Pow  # power_exp
        # 对于 b 的 (y - x) 次幂展开后，应该仍然是 Pow 对象
        assert (b**(y - x)).expand().is_Pow  # power_exp
        # 对于 b 的 (3 - x) 次幂展开后，应该仍然是 Pow 对象
        assert (b**(3 - x)).expand().is_Pow  # multinomial
    
    # 定义非负符号变量 nn1, nn2, nn3
    nn1 = Symbol('nn1', nonnegative=True)
    nn2 = Symbol('nn2', nonnegative=True)
    nn3 = Symbol('nn3', nonnegative=True)
    
    # 对以下幂运算进行不变性检查
    assert (x**(nn1 + nn2)).expand() == x**nn1*x**nn2
    assert (x**(-nn1 - nn2)).expand() == x**-nn1*x**-nn2
    assert unchanged(Pow, x, nn1 + nn2 - nn3)
    assert unchanged(Pow, x, 1 + nn2 - nn3)
    assert unchanged(Pow, x, nn1 - nn2)
    assert unchanged(Pow, x, 1 - nn2)
    assert unchanged(Pow, x, -1 + nn2)
```