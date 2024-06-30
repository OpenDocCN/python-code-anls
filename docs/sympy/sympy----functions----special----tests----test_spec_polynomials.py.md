# `D:\src\scipysrc\sympy\sympy\functions\special\tests\test_spec_polynomials.py`

```
# 导入 sympy 库中的具体模块和函数
from sympy.concrete.summations import Sum
from sympy.core.function import (Derivative, diff)
from sympy.core.numbers import (Rational, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.combinatorial.factorials import (RisingFactorial, binomial, factorial)
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper
from sympy.functions.special.polynomials import (assoc_laguerre, assoc_legendre, chebyshevt, chebyshevt_root, chebyshevu, chebyshevu_root, gegenbauer, hermite, hermite_prob, jacobi, jacobi_normalized, laguerre, legendre)
from sympy.polys.orthopolys import laguerre_poly
from sympy.polys.polyroots import roots

# 导入 sympy 核心模块中的表达式和异常处理相关内容
from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
from sympy.testing.pytest import raises

# 创建符号变量 x
x = Symbol('x')

# 定义测试函数 test_jacobi
def test_jacobi():
    # 创建符号变量 n, a, b
    n = Symbol("n")
    a = Symbol("a")
    b = Symbol("b")

    # 断言语句，验证 jacobi 多项式的计算结果
    assert jacobi(0, a, b, x) == 1
    assert jacobi(1, a, b, x) == a/2 - b/2 + x*(a/2 + b/2 + 1)

    assert jacobi(n, a, a, x) == RisingFactorial(
        a + 1, n)*gegenbauer(n, a + S.Half, x)/RisingFactorial(2*a + 1, n)
    assert jacobi(n, a, -a, x) == ((-1)**a*(-x + 1)**(-a/2)*(x + 1)**(a/2)*assoc_legendre(n, a, x)*
                                   factorial(-a + n)*gamma(a + n + 1)/(factorial(a + n)*gamma(n + 1)))
    assert jacobi(n, -b, b, x) == ((-x + 1)**(b/2)*(x + 1)**(-b/2)*assoc_legendre(n, b, x)*
                                   gamma(-b + n + 1)/gamma(n + 1))
    assert jacobi(n, 0, 0, x) == legendre(n, x)
    assert jacobi(n, S.Half, S.Half, x) == RisingFactorial(
        Rational(3, 2), n)*chebyshevu(n, x)/factorial(n + 1)
    assert jacobi(n, Rational(-1, 2), Rational(-1, 2), x) == RisingFactorial(
        S.Half, n)*chebyshevt(n, x)/factorial(n)

    # 验证 jacobi 函数返回的类型是 jacobi 类型
    X = jacobi(n, a, b, x)
    assert isinstance(X, jacobi)

    # 更多的 jacobi 多项式性质验证
    assert jacobi(n, a, b, -x) == (-1)**n*jacobi(n, b, a, x)
    assert jacobi(n, a, b, 0) == 2**(-n)*gamma(a + n + 1)*hyper(
        (-b - n, -n), (a + 1,), -1)/(factorial(n)*gamma(a + 1))
    assert jacobi(n, a, b, 1) == RisingFactorial(a + 1, n)/factorial(n)

    # 断言 jacobi 函数在无穷远点 a,b 依然保持不变
    m = Symbol("m", positive=True)
    assert jacobi(m, a, b, oo) == oo*RisingFactorial(a + b + m + 1, m)
    assert unchanged(jacobi, n, a, b, oo)

    # 断言 jacobi 函数的共轭性质
    assert conjugate(jacobi(m, a, b, x)) == \
        jacobi(m, conjugate(a), conjugate(b), conjugate(x))

    # 使用 Dummy 变量进行 jacobi 函数的导数计算
    _k = Dummy('k')
    assert diff(jacobi(n, a, b, x), n) == Derivative(jacobi(n, a, b, x), n)
    # 断言：对 jacobi 函数关于参数 a 的偏导数是否等于给定的表达式
    assert diff(jacobi(n, a, b, x), a).dummy_eq(Sum((jacobi(n, a, b, x) +
        (2*_k + a + b + 1)*RisingFactorial(_k + b + 1, -_k + n)*jacobi(_k, a,
        b, x)/((-_k + n)*RisingFactorial(_k + a + b + 1, -_k + n)))/(_k + a
        + b + n + 1), (_k, 0, n - 1)))
    
    # 断言：对 jacobi 函数关于参数 b 的偏导数是否等于给定的表达式
    assert diff(jacobi(n, a, b, x), b).dummy_eq(Sum(((-1)**(-_k + n)*(2*_k +
        a + b + 1)*RisingFactorial(_k + a + 1, -_k + n)*jacobi(_k, a, b, x)/
        ((-_k + n)*RisingFactorial(_k + a + b + 1, -_k + n)) + jacobi(n, a,
        b, x))/(_k + a + b + n + 1), (_k, 0, n - 1)))
    
    # 断言：对 jacobi 函数关于参数 x 的偏导数是否等于给定的表达式
    assert diff(jacobi(n, a, b, x), x) == \
        (a/2 + b/2 + n/2 + S.Half)*jacobi(n - 1, a + 1, b + 1, x)
    
    # 断言：jacobi_normalized 函数是否等于给定的归一化表达式
    assert jacobi_normalized(n, a, b, x) == \
           (jacobi(n, a, b, x)/sqrt(2**(a + b + 1)*gamma(a + n + 1)*gamma(b + n + 1)
                                    /((a + b + 2*n + 1)*factorial(n)*gamma(a + b + n + 1))))
    
    # 断言：使用非法参数值调用 jacobi 函数是否引发 ValueError 异常
    raises(ValueError, lambda: jacobi(-2.1, a, b, x))
    
    # 断言：使用符合特定条件的 Dummy 参数值调用 jacobi 函数是否引发 ValueError 异常
    raises(ValueError, lambda: jacobi(Dummy(positive=True, integer=True), 1, 2, oo))
    
    # 断言：jacobi 函数使用 rewrite 方法重写为 Sum 表达式后是否等于给定的表达式
    assert jacobi(n, a, b, x).rewrite(Sum).dummy_eq(Sum((S.Half - x/2)
        **_k*RisingFactorial(-n, _k)*RisingFactorial(_k + a + 1, -_k + n)*
        RisingFactorial(a + b + n + 1, _k)/factorial(_k), (_k, 0, n))/factorial(n))
    
    # 断言：jacobi 函数使用 rewrite 方法重写为 "polynomial" 形式后是否等于给定的表达式
    assert jacobi(n, a, b, x).rewrite("polynomial").dummy_eq(Sum((S.Half - x/2)
        **_k*RisingFactorial(-n, _k)*RisingFactorial(_k + a + 1, -_k + n)*
        RisingFactorial(a + b + n + 1, _k)/factorial(_k), (_k, 0, n))/factorial(n))
    
    # 断言：尝试对 jacobi 函数使用 fdiff 方法指定超出索引范围的偏导数引发 ArgumentIndexError 异常
    raises(ArgumentIndexError, lambda: jacobi(n, a, b, x).fdiff(5))
# 定义一个测试函数 test_gegenbauer
def test_gegenbauer():
    # 定义符号变量 n 和 a
    n = Symbol("n")
    a = Symbol("a")

    # 断言：Gegenbauer 多项式的特定值
    assert gegenbauer(0, a, x) == 1
    assert gegenbauer(1, a, x) == 2*a*x
    assert gegenbauer(2, a, x) == -a + x**2*(2*a**2 + 2*a)
    assert gegenbauer(3, a, x) == \
        x**3*(4*a**3/3 + 4*a**2 + a*Rational(8, 3)) + x*(-2*a**2 - 2*a)

    # 断言：当 n 为负数时，Gegenbauer 多项式为 0
    assert gegenbauer(-1, a, x) == 0

    # 断言：Gegenbauer 多项式与 Legendre 多项式的关系
    assert gegenbauer(n, S.Half, x) == legendre(n, x)

    # 断言：Gegenbauer 多项式与 ChebyshevU 多项式的关系
    assert gegenbauer(n, 1, x) == chebyshevu(n, x)

    # 断言：当 a 为 -1 时，Gegenbauer 多项式为 0
    assert gegenbauer(n, -1, x) == 0

    # 计算 Gegenbauer 多项式并赋值给 X，断言 X 的类型为 gegenbauer
    X = gegenbauer(n, a, x)
    assert isinstance(X, gegenbauer)

    # 断言：Gegenbauer 多项式在 -x 处的性质
    assert gegenbauer(n, a, -x) == (-1)**n*gegenbauer(n, a, x)

    # 断言：Gegenbauer 多项式在 x=0 处的值
    assert gegenbauer(n, a, 0) == 2**n*sqrt(pi) * \
        gamma(a + n/2)/(gamma(a)*gamma(-n/2 + S.Half)*gamma(n + 1))

    # 断言：Gegenbauer 多项式在 x=1 处的值
    assert gegenbauer(n, a, 1) == gamma(2*a + n)/(gamma(2*a)*gamma(n + 1))

    # 断言：当 a 为 Rational(3, 4) 时，Gegenbauer 多项式在 x=-1 处的值为 zoo
    assert gegenbauer(n, Rational(3, 4), -1) is zoo

    # 断言：当 a 为 Rational(1, 4) 时，Gegenbauer 多项式在 x=-1 处的值
    assert gegenbauer(n, Rational(1, 4), -1) == (sqrt(2)*cos(pi*(n + S.One/4))*
                      gamma(n + S.Half)/(sqrt(pi)*gamma(n + 1)))

    # 定义一个正数符号变量 m
    m = Symbol("m", positive=True)

    # 断言：当 a 为无穷大时，Gegenbauer 多项式的性质
    assert gegenbauer(m, a, oo) == oo*RisingFactorial(a, m)

    # 断言：对 Gegenbauer 多项式的变换不改变其本身
    assert unchanged(gegenbauer, n, a, oo)

    # 断言：Gegenbauer 多项式的共轭与参数的共轭关系
    assert conjugate(gegenbauer(n, a, x)) == gegenbauer(n, conjugate(a), conjugate(x))

    # 定义一个虚拟变量 _k
    _k = Dummy('k')

    # 断言：对 Gegenbauer 多项式关于 n 的求导性质
    assert diff(gegenbauer(n, a, x), n) == Derivative(gegenbauer(n, a, x), n)

    # 断言：对 Gegenbauer 多项式关于 a 的求导性质
    assert diff(gegenbauer(n, a, x), a).dummy_eq(Sum((2*(-1)**(-_k + n) + 2)*
        (_k + a)*gegenbauer(_k, a, x)/((-_k + n)*(_k + 2*a + n)) + ((2*_k +
        2)/((_k + 2*a)*(2*_k + 2*a + 1)) + 2/(_k + 2*a + n))*gegenbauer(n, a
        , x), (_k, 0, n - 1)))

    # 断言：对 Gegenbauer 多项式关于 x 的求导性质
    assert diff(gegenbauer(n, a, x), x) == 2*a*gegenbauer(n - 1, a + 1, x)

    # 断言：Gegenbauer 多项式的求和重写
    assert gegenbauer(n, a, x).rewrite(Sum).dummy_eq(
        Sum((-1)**_k*(2*x)**(-2*_k + n)*RisingFactorial(a, -_k + n)
        /(factorial(_k)*factorial(-2*_k + n)), (_k, 0, floor(n/2))))

    # 断言：Gegenbauer 多项式的多项式重写
    assert gegenbauer(n, a, x).rewrite("polynomial").dummy_eq(
        Sum((-1)**_k*(2*x)**(-2*_k + n)*RisingFactorial(a, -_k + n)
        /(factorial(_k)*factorial(-2*_k + n)), (_k, 0, floor(n/2))))

    # 断言：对 Gegenbauer 多项式进行 ArgumentIndexError 异常测试
    raises(ArgumentIndexError, lambda: gegenbauer(n, a, x).fdiff(4))


# 定义一个测试函数 test_legendre
def test_legendre():
    # 断言：Legendre 多项式的特定值
    assert legendre(0, x) == 1
    assert legendre(1, x) == x
    assert legendre(2, x) == ((3*x**2 - 1)/2).expand()
    assert legendre(3, x) == ((5*x**3 - 3*x)/2).expand()
    assert legendre(4, x) == ((35*x**4 - 30*x**2 + 3)/8).expand()
    assert legendre(5, x) == ((63*x**5 - 70*x**3 + 15*x)/8).expand()
    assert legendre(6, x) == ((231*x**6 - 315*x**4 + 105*x**2 - 5)/16).expand()

    # 断言：Legendre 多项式在特定点的值
    assert legendre(10, -1) == 1
    assert legendre(11, -1) == -1
    assert legendre(10, 1) == 1
    assert legendre(11, 1) == 1
    assert legendre(10, 0) != 0
    assert legendre(11, 0) == 0

    # 断言：当 n 为负数时，Legendre 多项式为 1
    assert legendre(-1, x) == 1

    # 定义符号变量 k
    k = Symbol('k')

    # 断言：Legendre 多项式中包含变量 k，并对其进行替换和展开
    assert legendre(5 - k, x).subs(k, 2) == ((5*x**3 - 3*x)/2).expand()
    # 使用 legendre 函数计算 Legendre 多项式的根，验证结果是否符合预期
    assert roots(legendre(4, x), x) == {
        sqrt(Rational(3, 7) - Rational(2, 35)*sqrt(30)): 1,
        -sqrt(Rational(3, 7) - Rational(2, 35)*sqrt(30)): 1,
        sqrt(Rational(3, 7) + Rational(2, 35)*sqrt(30)): 1,
        -sqrt(Rational(3, 7) + Rational(2, 35)*sqrt(30)): 1,
    }

    # 创建一个符号变量 n
    n = Symbol("n")

    # 计算 legendre(n, x)，并检查返回结果的类型是否为 legendre 类
    X = legendre(n, x)
    assert isinstance(X, legendre)
    
    # 验证 legendre 函数在变量 n 和 x 下是否保持不变
    assert unchanged(legendre, n, x)

    # 验证 legendre 函数的特定值
    assert legendre(n, 0) == sqrt(pi)/(gamma(S.Half - n/2)*gamma(n/2 + 1))
    assert legendre(n, 1) == 1
    assert legendre(n, oo) is oo
    assert legendre(-n, x) == legendre(n - 1, x)
    assert legendre(n, -x) == (-1)**n*legendre(n, x)
    assert unchanged(legendre, -n + k, x)

    # 验证 legendre 函数的共轭
    assert conjugate(legendre(n, x)) == legendre(n, conjugate(x))

    # 验证 legendre 函数对变量 x 的导数
    assert diff(legendre(n, x), x) == \
        n*(x*legendre(n, x) - legendre(n - 1, x))/(x**2 - 1)
    # 验证 legendre 函数对变量 n 的导数
    assert diff(legendre(n, x), n) == Derivative(legendre(n, x), n)

    # 使用 Sum 重写 legendre 函数，验证结果是否等价
    _k = Dummy('k')
    assert legendre(n, x).rewrite(Sum).dummy_eq(Sum((-1)**_k*(S.Half -
            x/2)**_k*(x/2 + S.Half)**(-_k + n)*binomial(n, _k)**2, (_k, 0, n)))
    # 使用 "polynomial" 重写 legendre 函数，验证结果是否等价
    assert legendre(n, x).rewrite("polynomial").dummy_eq(Sum((-1)**_k*(S.Half -
            x/2)**_k*(x/2 + S.Half)**(-_k + n)*binomial(n, _k)**2, (_k, 0, n)))
    
    # 验证 legendre 函数在给定参数下是否引发预期的异常
    raises(ArgumentIndexError, lambda: legendre(n, x).fdiff(1))
    raises(ArgumentIndexError, lambda: legendre(n, x).fdiff(3))
# 定义测试函数 test_assoc_legendre，用于测试关联勒让德多项式的性质
def test_assoc_legendre():
    # 将 assoc_legendre 函数赋值给 Plm 变量，简化后续的调用
    Plm = assoc_legendre
    # 定义 Q 为 sqrt(1 - x**2)，用于简化后续的表达式
    Q = sqrt(1 - x**2)

    # 下面开始逐个断言测试不同的勒让德多项式 Plm(n, m, x) 的值是否符合预期

    # 断言 Plm(0, 0, x) 的值为 1
    assert Plm(0, 0, x) == 1
    # 断言 Plm(1, 0, x) 的值为 x
    assert Plm(1, 0, x) == x
    # 断言 Plm(1, 1, x) 的值为 -Q
    assert Plm(1, 1, x) == -Q
    # 断言 Plm(2, 0, x) 的值为 (3*x**2 - 1)/2
    assert Plm(2, 0, x) == (3*x**2 - 1)/2
    # 断言 Plm(2, 1, x) 的值为 -3*x*Q
    assert Plm(2, 1, x) == -3*x*Q
    # 断言 Plm(2, 2, x) 的值为 3*Q**2
    assert Plm(2, 2, x) == 3*Q**2
    # 断言 Plm(3, 0, x) 的值为 (5*x**3 - 3*x)/2
    assert Plm(3, 0, x) == (5*x**3 - 3*x)/2
    # 断言 Plm(3, 1, x) 展开后与给定表达式相等
    assert Plm(3, 1, x).expand() == ((3*(1 - 5*x**2)/2).expand() * Q).expand()
    # 断言 Plm(3, 2, x) 的值为 15*x * Q**2
    assert Plm(3, 2, x) == 15*x * Q**2
    # 断言 Plm(3, 3, x) 的值为 -15 * Q**3
    assert Plm(3, 3, x) == -15 * Q**3

    # 负 m 的情况下的断言
    # 断言 Plm(1, -1, x) 的值等于 Plm(1, 1, x) 的负值除以 2
    assert Plm(1, -1, x) == -Plm(1, 1, x)/2
    # 断言 Plm(2, -2, x) 的值等于 Plm(2, 2, x) 除以 24
    assert Plm(2, -2, x) == Plm(2, 2, x)/24
    # 断言 Plm(2, -1, x) 的值等于 Plm(2, 1, x) 的负值除以 6
    assert Plm(2, -1, x) == -Plm(2, 1, x)/6
    # 断言 Plm(3, -3, x) 的值等于 Plm(3, 3, x) 的负值除以 720
    assert Plm(3, -3, x) == -Plm(3, 3, x)/720
    # 断言 Plm(3, -2, x) 的值等于 Plm(3, 2, x) 除以 120
    assert Plm(3, -2, x) == Plm(3, 2, x)/120
    # 断言 Plm(3, -1, x) 的值等于 Plm(3, 1, x) 的负值除以 12
    assert Plm(3, -1, x) == -Plm(3, 1, x)/12

    # 定义符号变量 n 和 m
    n = Symbol("n")
    m = Symbol("m")
    # 将 Plm(n, m, x) 赋值给 X，并断言 X 的类型为 assoc_legendre
    X = Plm(n, m, x)
    assert isinstance(X, assoc_legendre)

    # 断言 Plm(n, 0, x) 的值等于 legendre(n, x)
    assert Plm(n, 0, x) == legendre(n, x)
    # 断言 Plm(n, m, 0) 的值符合给定的复杂表达式
    assert Plm(n, m, 0) == 2**m*sqrt(pi)/(gamma(-m/2 - n/2 +
                           S.Half)*gamma(-m/2 + n/2 + 1))

    # 断言 Plm(m, n, x) 对 x 的导数等于给定的表达式
    assert diff(Plm(m, n, x), x) == (m*x*assoc_legendre(m, n, x) -
                (m + n)*assoc_legendre(m - 1, n, x))/(x**2 - 1)

    # 定义 _k 为 Dummy('k')，并断言 Plm(m, n, x) 重写为 Sum 后与给定表达式相等
    _k = Dummy('k')
    assert Plm(m, n, x).rewrite(Sum).dummy_eq(
            (1 - x**2)**(n/2)*Sum((-1)**_k*2**(-m)*x**(-2*_k + m - n)*factorial
             (-2*_k + 2*m)/(factorial(_k)*factorial(-_k + m)*factorial(-2*_k + m
              - n)), (_k, 0, floor(m/2 - n/2))))
    # 断言 Plm(m, n, x) 重写为 "polynomial" 后与给定表达式相等
    assert Plm(m, n, x).rewrite("polynomial").dummy_eq(
            (1 - x**2)**(n/2)*Sum((-1)**_k*2**(-m)*x**(-2*_k + m - n)*factorial
             (-2*_k + 2*m)/(factorial(_k)*factorial(-_k + m)*factorial(-2*_k + m
              - n)), (_k, 0, floor(m/2 - n/2))))
    # 断言 assoc_legendre(n, m, x) 的共轭等于 assoc_legendre(n, conjugate(m), conjugate(x))
    assert conjugate(assoc_legendre(n, m, x)) == \
        assoc_legendre(n, conjugate(m), conjugate(x))
    # 断言调用 Plm(0, 1, x) 会引发 ValueError 异常
    raises(ValueError, lambda: Plm(0, 1, x))
    # 断言调用 Plm(-1, 1, x) 会引发 ValueError 异常
    raises(ValueError, lambda: Plm(-1, 1, x))
    # 断言调用 Plm(n, m, x).fdiff(1) 会引发 ArgumentIndexError 异常
    raises(ArgumentIndexError, lambda: Plm(n, m, x).fdiff(1))
    # 断言调用 Plm(n, m, x).fdiff(2) 会引发 ArgumentIndexError 异常
    raises(ArgumentIndexError, lambda: Plm(n, m, x).fdiff(2))
    # 断言调用 Plm(n, m, x).fdiff(4) 会引发 ArgumentIndexError 异常
    raises(ArgumentIndexError, lambda: Plm(n, m, x).fdiff(4))


# 定义测试函数 test_chebyshev，用于测试切比雪夫多项式的性质
def test_chebyshev():
    # 断言 chebyshevt(0, x) 的值为 1
    assert chebyshevt(0, x) == 1
    # 断言 chebyshevt(1, x) 的值为 x
    assert chebyshevt(1, x) == x
    # 断言 chebyshevt(2, x) 的值为 2*x**2 - 1
    assert chebyshevt(2, x) == 2*x**2 - 1
    # 断言 chebyshe
    # 断言：共轭函数应用于 n 次切比雪夫多项式后的结果等于切比雪夫多项式应用于共轭数 x 后的结果的共轭
    assert conjugate(chebyshevt(n, x)) == chebyshevt(n, conjugate(x))

    # 断言：切比雪夫多项式 n 阶关于变量 x 的导数等于 n 乘以切比雪夫函数第二类 (n-1) 阶关于 x 的结果
    assert diff(chebyshevt(n, x), x) == n*chebyshevu(n - 1, x)

    # 定义变量 X 为切比雪夫函数第二类 n 阶关于 x 的结果
    X = chebyshevu(n, x)
    # 断言：X 的类型应为切比雪夫函数第二类
    assert isinstance(X, chebyshevu)

    # 定义符号变量 y
    y = Symbol('y')
    # 断言：切比雪夫函数第二类 n 阶关于 -x 等于 (-1)^n 乘以切比雪夫函数第二类 n 阶关于 x
    assert chebyshevu(n, -x) == (-1)**n*chebyshevu(n, x)
    # 断言：切比雪夫函数第二类 -n 阶关于 x 等于 -1 乘以切比雪夫函数第二类 (n-2) 阶关于 x
    assert chebyshevu(-n, x) == -chebyshevu(n - 2, x)
    # 断言：未改变的切比雪夫函数第二类函数对于 -n + y 与 x 的结果
    assert unchanged(chebyshevu, -n + y, x)

    # 断言：切比雪夫函数第二类 n 阶关于 0 的值等于 cos(pi*n/2)
    assert chebyshevu(n, 0) == cos(pi*n/2)
    # 断言：切比雪夫函数第二类 n 阶关于 1 的值等于 n + 1
    assert chebyshevu(n, 1) == n + 1
    # 断言：切比雪夫函数第二类 n 阶关于 oo 的值为 oo
    assert chebyshevu(n, oo) is oo

    # 断言：共轭函数应用于切比雪夫函数第二类 n 阶的结果等于切比雪夫函数第二类 n 阶应用于共轭数 x 后的结果的共轭
    assert conjugate(chebyshevu(n, x)) == chebyshevu(n, conjugate(x))

    # 断言：切比雪夫函数第二类 n 阶关于 x 的导数
    assert diff(chebyshevu(n, x), x) == \
        (-x*chebyshevu(n, x) + (n + 1)*chebyshevt(n + 1, x))/(x**2 - 1)

    # 定义符号变量 _k
    _k = Dummy('k')
    # 断言：切比雪夫多项式 n 阶关于 x 通过求和重写后的结果等于指定求和表达式
    assert chebyshevt(n, x).rewrite(Sum).dummy_eq(Sum(x**(-2*_k + n)
                    *(x**2 - 1)**_k*binomial(n, 2*_k), (_k, 0, floor(n/2))))
    # 断言：切比雪夫多项式 n 阶关于 x 通过多项式重写后的结果等于指定求和表达式
    assert chebyshevt(n, x).rewrite("polynomial").dummy_eq(Sum(x**(-2*_k + n)
                    *(x**2 - 1)**_k*binomial(n, 2*_k), (_k, 0, floor(n/2))))
    # 断言：切比雪夫函数第二类 n 阶关于 x 通过求和重写后的结果等于指定求和表达式
    assert chebyshevu(n, x).rewrite(Sum).dummy_eq(Sum((-1)**_k*(2*x)
                    **(-2*_k + n)*factorial(-_k + n)/(factorial(_k)*
                       factorial(-2*_k + n)), (_k, 0, floor(n/2))))
    # 断言：切比雪夫函数第二类 n 阶关于 x 通过多项式重写后的结果等于指定求和表达式
    assert chebyshevu(n, x).rewrite("polynomial").dummy_eq(Sum((-1)**_k*(2*x)
                    **(-2*_k + n)*factorial(-_k + n)/(factorial(_k)*
                       factorial(-2*_k + n)), (_k, 0, floor(n/2))))
    # 断言：尝试对切比雪夫多项式 n 阶关于 x 进行超出参数范围的导数操作，应引发参数索引错误
    raises(ArgumentIndexError, lambda: chebyshevt(n, x).fdiff(1))
    # 断言：尝试对切比雪夫多项式 n 阶关于 x 进行超出参数范围的导数操作，应引发参数索引错误
    raises(ArgumentIndexError, lambda: chebyshevt(n, x).fdiff(3))
    # 断言：尝试对切比雪夫函数第二类 n 阶关于 x 进行超出参数范围的导数操作，应引发参数索引错误
    raises(ArgumentIndexError, lambda: chebyshevu(n, x).fdiff(1))
    # 断言：尝试对切比雪夫函数第二类 n 阶关于 x 进行超出参数范围的导数操作，应引发参数索引错误
    raises(ArgumentIndexError, lambda: chebyshevu(n, x).fdiff(3))
# 定义测试 Hermite 多项式的函数
def test_hermite():
    # 断言 Hermite 多项式的特定值
    assert hermite(0, x) == 1
    assert hermite(1, x) == 2*x
    assert hermite(2, x) == 4*x**2 - 2
    assert hermite(3, x) == 8*x**3 - 12*x
    assert hermite(4, x) == 16*x**4 - 48*x**2 + 12
    assert hermite(6, x) == 64*x**6 - 480*x**4 + 720*x**2 - 120

    # 定义符号变量 n
    n = Symbol("n")
    # 断言 hermite 函数对 n 和 x 的不变性
    assert unchanged(hermite, n, x)
    # 断言 hermite 函数的负数版本
    assert hermite(n, -x) == (-1)**n*hermite(n, x)
    # 断言 hermite 函数对 -n 和 x 的不变性
    assert unchanged(hermite, -n, x)

    # 断言 hermite 函数在 x = 0 处的值
    assert hermite(n, 0) == 2**n*sqrt(pi)/gamma(S.Half - n/2)
    # 断言 hermite 函数在 x = oo 处的值
    assert hermite(n, oo) is oo

    # 断言 hermite 函数的共轭
    assert conjugate(hermite(n, x)) == hermite(n, conjugate(x))

    # 定义临时变量 _k
    _k = Dummy('k')
    # 使用 Sum 重写 hermite 函数，并进行等式比较
    assert hermite(n, x).rewrite(Sum).dummy_eq(factorial(n)*Sum((-1)
        **_k*(2*x)**(-2*_k + n)/(factorial(_k)*factorial(-2*_k + n)), (_k,
        0, floor(n/2))))
    # 使用 "polynomial" 方法重写 hermite 函数，并进行等式比较
    assert hermite(n, x).rewrite("polynomial").dummy_eq(factorial(n)*Sum((-1)
        **_k*(2*x)**(-2*_k + n)/(factorial(_k)*factorial(-2*_k + n)), (_k,
        0, floor(n/2))))

    # 断言 hermite 函数对 x 的偏导数
    assert diff(hermite(n, x), x) == 2*n*hermite(n - 1, x)
    # 断言 hermite 函数对 n 的导数
    assert diff(hermite(n, x), n) == Derivative(hermite(n, x), n)
    # 断言尝试在 hermite 函数上进行第三阶导数抛出 ArgumentIndexError 异常
    raises(ArgumentIndexError, lambda: hermite(n, x).fdiff(3))

    # 使用 hermite_prob 重写 hermite 函数，并进行等式比较
    assert hermite(n, x).rewrite(hermite_prob) == \
            sqrt(2)**n * hermite_prob(n, x*sqrt(2))


# 定义测试 Hermite probabilistic 多项式的函数
def test_hermite_prob():
    # 断言 Hermite probabilistic 多项式的特定值
    assert hermite_prob(0, x) == 1
    assert hermite_prob(1, x) == x
    assert hermite_prob(2, x) == x**2 - 1
    assert hermite_prob(3, x) == x**3 - 3*x
    assert hermite_prob(4, x) == x**4 - 6*x**2 + 3
    assert hermite_prob(6, x) == x**6 - 15*x**4 + 45*x**2 - 15

    # 定义符号变量 n
    n = Symbol("n")
    # 断言 hermite_prob 函数对 n 和 x 的不变性
    assert unchanged(hermite_prob, n, x)
    # 断言 hermite_prob 函数的负数版本
    assert hermite_prob(n, -x) == (-1)**n*hermite_prob(n, x)
    # 断言 hermite_prob 函数对 -n 和 x 的不变性
    assert unchanged(hermite_prob, -n, x)

    # 断言 hermite_prob 函数在 x = 0 处的值
    assert hermite_prob(n, 0) == sqrt(pi)/gamma(S.Half - n/2)
    # 断言 hermite_prob 函数在 x = oo 处的值
    assert hermite_prob(n, oo) is oo

    # 断言 hermite_prob 函数的共轭
    assert conjugate(hermite_prob(n, x)) == hermite_prob(n, conjugate(x))

    # 定义临时变量 _k
    _k = Dummy('k')
    # 使用 Sum 重写 hermite_prob 函数，并进行等式比较
    assert hermite_prob(n, x).rewrite(Sum).dummy_eq(factorial(n) *
        Sum((-S.Half)**_k * x**(n-2*_k) / (factorial(_k) * factorial(n-2*_k)),
        (_k, 0, floor(n/2))))
    # 使用 "polynomial" 方法重写 hermite_prob 函数，并进行等式比较
    assert hermite_prob(n, x).rewrite("polynomial").dummy_eq(factorial(n) *
        Sum((-S.Half)**_k * x**(n-2*_k) / (factorial(_k) * factorial(n-2*_k)),
        (_k, 0, floor(n/2))))

    # 断言 hermite_prob 函数对 x 的偏导数
    assert diff(hermite_prob(n, x), x) == n*hermite_prob(n-1, x)
    # 断言 hermite_prob 函数对 n 的导数
    assert diff(hermite_prob(n, x), n) == Derivative(hermite_prob(n, x), n)
    # 断言尝试在 hermite_prob 函数上进行第三阶导数抛出 ArgumentIndexError 异常
    raises(ArgumentIndexError, lambda: hermite_prob(n, x).fdiff(3))


# 定义测试 Laguerre 多项式的函数
def test_laguerre():
    # 定义符号变量 n 和 m
    n = Symbol("n")
    m = Symbol("m", negative=True)

    # 断言 Laguerre 多项式的特定值
    assert laguerre(0, x) == 1
    assert laguerre(1, x) == -x + 1
    assert laguerre(2, x) == x**2/2 - 2*x + 1
    assert laguerre(3, x) == -x**3/6 + 3*x**2/2 - 3*x + 1
    assert laguerre(-2, x) == (x + 1)*exp(x)

    # 定义 X 作为 laguerre 函数的返回值，并断言其类型为 laguerre
    X = laguerre(n, x)
    assert isinstance(X, laguerre)

    # 断言 laguerre 函数在 x = 0 处的值
    assert laguerre(n, 0) == 1
    # 断言：验证拉盖尔多项式在无穷远处的极限值
    assert laguerre(n, oo) == (-1)**n*oo
    # 断言：验证拉盖尔多项式在负无穷处的值
    assert laguerre(n, -oo) is oo
    
    # 断言：验证拉盖尔多项式的共轭与共轭参数的拉盖尔多项式相等
    assert conjugate(laguerre(n, x)) == laguerre(n, conjugate(x))
    
    # 创建一个虚拟符号变量 _k
    _k = Dummy('k')
    
    # 断言：验证拉盖尔多项式重写为求和形式后的等式
    assert laguerre(n, x).rewrite(Sum).dummy_eq(
        Sum(x**_k*RisingFactorial(-n, _k)/factorial(_k)**2, (_k, 0, n)))
    # 断言：验证拉盖尔多项式重写为多项式形式后的等式
    assert laguerre(n, x).rewrite("polynomial").dummy_eq(
        Sum(x**_k*RisingFactorial(-n, _k)/factorial(_k)**2, (_k, 0, n)))
    # 断言：验证指数形式的拉盖尔多项式重写为求和形式后的等式
    assert laguerre(m, x).rewrite(Sum).dummy_eq(
        exp(x)*Sum((-x)**_k*RisingFactorial(m + 1, _k)/factorial(_k)**2,
            (_k, 0, -m - 1)))
    # 断言：验证指数形式的拉盖尔多项式重写为多项式形式后的等式
    assert laguerre(m, x).rewrite("polynomial").dummy_eq(
        exp(x)*Sum((-x)**_k*RisingFactorial(m + 1, _k)/factorial(_k)**2,
            (_k, 0, -m - 1)))
    
    # 断言：验证拉盖尔多项式对变量 x 的一阶导数等于关联拉盖尔多项式
    assert diff(laguerre(n, x), x) == -assoc_laguerre(n - 1, 1, x)
    
    # 创建一个符号变量 k
    k = Symbol('k')
    # 断言：验证负整数阶的拉盖尔多项式与指数形式的关系
    assert laguerre(-n, x) == exp(x)*laguerre(n - 1, -x)
    # 断言：验证特定的负整数阶拉盖尔多项式与指定的关系
    assert laguerre(-3, x) == exp(x)*laguerre(2, -x)
    # 断言：验证未改变的拉盖尔多项式参数
    assert unchanged(laguerre, -n + k, x)
    
    # 引发异常测试：验证负实数阶的拉盖尔多项式引发 ValueError 异常
    raises(ValueError, lambda: laguerre(-2.1, x))
    # 引发异常测试：验证有理数阶的拉盖尔多项式引发 ValueError 异常
    raises(ValueError, lambda: laguerre(Rational(5, 2), x))
    # 引发异常测试：验证尝试对拉盖尔多项式的超出索引数的导数操作引发 ArgumentIndexError 异常
    raises(ArgumentIndexError, lambda: laguerre(n, x).fdiff(1))
    # 引发异常测试：验证尝试对拉盖尔多项式的超出索引数的导数操作引发 ArgumentIndexError 异常
    raises(ArgumentIndexError, lambda: laguerre(n, x).fdiff(3))
def test_assoc_laguerre():
    n = Symbol("n")  # 定义符号变量 n
    m = Symbol("m")  # 定义符号变量 m
    alpha = Symbol("alpha")  # 定义符号变量 alpha

    # generalized Laguerre polynomials:
    # 测试广义拉盖尔多项式的特定值，验证其结果
    assert assoc_laguerre(0, alpha, x) == 1
    assert assoc_laguerre(1, alpha, x) == -x + alpha + 1
    assert assoc_laguerre(2, alpha, x).expand() == \
        (x**2/2 - (alpha + 2)*x + (alpha + 2)*(alpha + 1)/2).expand()
    assert assoc_laguerre(3, alpha, x).expand() == \
        (-x**3/6 + (alpha + 3)*x**2/2 - (alpha + 2)*(alpha + 3)*x/2 +
        (alpha + 1)*(alpha + 2)*(alpha + 3)/6).expand()

    # Test the lowest 10 polynomials with laguerre_poly, to make sure it works:
    # 使用 laguerre_poly 测试前 10 个拉盖尔多项式，确保它们的展开结果一致
    for i in range(10):
        assert assoc_laguerre(i, 0, x).expand() == laguerre_poly(i, x)

    X = assoc_laguerre(n, m, x)
    assert isinstance(X, assoc_laguerre)  # 验证 X 的类型是否为 assoc_laguerre

    assert assoc_laguerre(n, 0, x) == laguerre(n, x)  # 验证特定情况下与普通拉盖尔多项式的关系
    assert assoc_laguerre(n, alpha, 0) == binomial(alpha + n, alpha)  # 验证特定情况下的二项式系数
    p = Symbol("p", positive=True)
    assert assoc_laguerre(p, alpha, oo) == (-1)**p*oo  # 验证极限情况
    assert assoc_laguerre(p, alpha, -oo) is oo  # 验证极限情况

    assert diff(assoc_laguerre(n, alpha, x), x) == \
        -assoc_laguerre(n - 1, alpha + 1, x)  # 求导数并验证结果

    _k = Dummy('k')
    assert diff(assoc_laguerre(n, alpha, x), alpha).dummy_eq(
        Sum(assoc_laguerre(_k, alpha, x)/(-alpha + n), (_k, 0, n - 1)))  # 验证关于 alpha 的偏导数

    assert conjugate(assoc_laguerre(n, alpha, x)) == \
        assoc_laguerre(n, conjugate(alpha), conjugate(x))  # 验证共轭的结果

    assert assoc_laguerre(n, alpha, x).rewrite(Sum).dummy_eq(
            gamma(alpha + n + 1)*Sum(x**_k*RisingFactorial(-n, _k)/
            (factorial(_k)*gamma(_k + alpha + 1)), (_k, 0, n))/factorial(n))  # 使用 Sum 重写并验证结果

    assert assoc_laguerre(n, alpha, x).rewrite("polynomial").dummy_eq(
            gamma(alpha + n + 1)*Sum(x**_k*RisingFactorial(-n, _k)/
            (factorial(_k)*gamma(_k + alpha + 1)), (_k, 0, n))/factorial(n))  # 使用 "polynomial" 重写并验证结果

    raises(ValueError, lambda: assoc_laguerre(-2.1, alpha, x))  # 验证异常情况：负数输入
    raises(ArgumentIndexError, lambda: assoc_laguerre(n, alpha, x).fdiff(1))  # 验证异常情况：非法的 fdiff 参数
    raises(ArgumentIndexError, lambda: assoc_laguerre(n, alpha, x).fdiff(4))  # 验证异常情况：非法的 fdiff 参数
```