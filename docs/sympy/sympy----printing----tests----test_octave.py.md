# `D:\src\scipysrc\sympy\sympy\printing\tests\test_octave.py`

```
# 导入所需的符号和函数模块，包括数学常数和特殊函数
from sympy.core import (S, pi, oo, symbols, Function, Rational, Integer,
                        Tuple, Symbol, EulerGamma, GoldenRatio, Catalan,
                        Lambda, Mul, Pow, Mod, Eq, Ne, Le, Lt, Gt, Ge)
from sympy.codegen.matrix_nodes import MatrixSolve
from sympy.functions import (arg, atan2, bernoulli, beta, ceiling, chebyshevu,
                             chebyshevt, conjugate, DiracDelta, exp, expint,
                             factorial, floor, harmonic, Heaviside, im,
                             laguerre, LambertW, log, Max, Min, Piecewise,
                             polylog, re, RisingFactorial, sign, sinc, sqrt,
                             zeta, binomial, legendre, dirichlet_eta,
                             riemann_xi)
from sympy.functions import (sin, cos, tan, cot, sec, csc, asin, acos, acot,
                             atan, asec, acsc, sinh, cosh, tanh, coth, csch,
                             sech, asinh, acosh, atanh, acoth, asech, acsch)
from sympy.testing.pytest import raises, XFAIL
from sympy.utilities.lambdify import implemented_function
from sympy.matrices import (eye, Matrix, MatrixSymbol, Identity,
                            HadamardProduct, SparseMatrix, HadamardPower)
from sympy.functions.special.bessel import (jn, yn, besselj, bessely, besseli,
                                            besselk, hankel1, hankel2, airyai,
                                            airybi, airyaiprime, airybiprime)
from sympy.functions.special.gamma_functions import (gamma, lowergamma,
                                                     uppergamma, loggamma,
                                                     polygamma)
from sympy.functions.special.error_functions import (Chi, Ci, erf, erfc, erfi,
                                                     erfcinv, erfinv, fresnelc,
                                                     fresnels, li, Shi, Si, Li,
                                                     erf2, Ei)
from sympy.printing.octave import octave_code, octave_code as mcode

# 定义符号变量 x, y, z
x, y, z = symbols('x,y,z')

# 测试函数，测试整数转换成 Octave 代码的函数
def test_Integer():
    assert mcode(Integer(67)) == "67"
    assert mcode(Integer(-1)) == "-1"

# 测试函数，测试有理数转换成 Octave 代码的函数
def test_Rational():
    assert mcode(Rational(3, 7)) == "3/7"
    assert mcode(Rational(18, 9)) == "2"
    assert mcode(Rational(3, -7)) == "-3/7"
    assert mcode(Rational(-3, -7)) == "3/7"
    assert mcode(x + Rational(3, 7)) == "x + 3/7"
    assert mcode(Rational(3, 7)*x) == "3*x/7"

# 测试函数，测试关系运算符转换成 Octave 代码的函数
def test_Relational():
    assert mcode(Eq(x, y)) == "x == y"
    assert mcode(Ne(x, y)) == "x != y"
    assert mcode(Le(x, y)) == "x <= y"
    assert mcode(Lt(x, y)) == "x < y"
    assert mcode(Gt(x, y)) == "x > y"
    assert mcode(Ge(x, y)) == "x >= y"

# 测试函数，测试数学函数转换成 Octave 代码的函数
def test_Function():
    assert mcode(sin(x) ** cos(x)) == "sin(x).^cos(x)"
    assert mcode(sign(x)) == "sign(x)"
    assert mcode(exp(x)) == "exp(x)"
    assert mcode(log(x)) == "log(x)"
    assert mcode(factorial(x)) == "factorial(x)"
    # 断言：验证调用 mcode 函数后返回的字符串是否与预期相符，如果不符则抛出 AssertionError
    
    assert mcode(floor(x)) == "floor(x)"
    
    # 断言：验证调用 mcode 函数后返回的字符串是否与预期相符，如果不符则抛出 AssertionError
    
    assert mcode(atan2(y, x)) == "atan2(y, x)"
    
    # 断言：验证调用 mcode 函数后返回的字符串是否与预期相符，如果不符则抛出 AssertionError
    
    assert mcode(beta(x, y)) == 'beta(x, y)'
    
    # 断言：验证调用 mcode 函数后返回的字符串是否与预期相符，如果不符则抛出 AssertionError
    
    assert mcode(polylog(x, y)) == 'polylog(x, y)'
    
    # 断言：验证调用 mcode 函数后返回的字符串是否与预期相符，如果不符则抛出 AssertionError
    
    assert mcode(harmonic(x)) == 'harmonic(x)'
    
    # 断言：验证调用 mcode 函数后返回的字符串是否与预期相符，如果不符则抛出 AssertionError
    
    assert mcode(bernoulli(x)) == "bernoulli(x)"
    
    # 断言：验证调用 mcode 函数后返回的字符串是否与预期相符，如果不符则抛出 AssertionError
    
    assert mcode(bernoulli(x, y)) == "bernoulli(x, y)"
    
    # 断言：验证调用 mcode 函数后返回的字符串是否与预期相符，如果不符则抛出 AssertionError
    
    assert mcode(legendre(x, y)) == "legendre(x, y)"
def test_Function_change_name():
    # 检查 mcode 函数对 abs(x) 的输出是否正确
    assert mcode(abs(x)) == "abs(x)"
    # 检查 mcode 函数对 ceiling(x) 的输出是否正确
    assert mcode(ceiling(x)) == "ceil(x)"
    # 检查 mcode 函数对 arg(x) 的输出是否正确
    assert mcode(arg(x)) == "angle(x)"
    # 检查 mcode 函数对 im(x) 的输出是否正确
    assert mcode(im(x)) == "imag(x)"
    # 检查 mcode 函数对 re(x) 的输出是否正确
    assert mcode(re(x)) == "real(x)"
    # 检查 mcode 函数对 conjugate(x) 的输出是否正确
    assert mcode(conjugate(x)) == "conj(x)"
    # 检查 mcode 函数对 chebyshevt(y, x) 的输出是否正确
    assert mcode(chebyshevt(y, x)) == "chebyshevT(y, x)"
    # 检查 mcode 函数对 chebyshevu(y, x) 的输出是否正确
    assert mcode(chebyshevu(y, x)) == "chebyshevU(y, x)"
    # 检查 mcode 函数对 laguerre(x, y) 的输出是否正确
    assert mcode(laguerre(x, y)) == "laguerreL(x, y)"
    # 检查 mcode 函数对 Chi(x) 的输出是否正确
    assert mcode(Chi(x)) == "coshint(x)"
    # 检查 mcode 函数对 Shi(x) 的输出是否正确
    assert mcode(Shi(x)) == "sinhint(x)"
    # 检查 mcode 函数对 Ci(x) 的输出是否正确
    assert mcode(Ci(x)) == "cosint(x)"
    # 检查 mcode 函数对 Si(x) 的输出是否正确
    assert mcode(Si(x)) == "sinint(x)"
    # 检查 mcode 函数对 li(x) 的输出是否正确
    assert mcode(li(x)) == "logint(x)"
    # 检查 mcode 函数对 loggamma(x) 的输出是否正确
    assert mcode(loggamma(x)) == "gammaln(x)"
    # 检查 mcode 函数对 polygamma(x, y) 的输出是否正确
    assert mcode(polygamma(x, y)) == "psi(x, y)"
    # 检查 mcode 函数对 RisingFactorial(x, y) 的输出是否正确
    assert mcode(RisingFactorial(x, y)) == "pochhammer(x, y)"
    # 检查 mcode 函数对 DiracDelta(x) 的输出是否正确
    assert mcode(DiracDelta(x)) == "dirac(x)"
    # 检查 mcode 函数对 DiracDelta(x, 3) 的输出是否正确
    assert mcode(DiracDelta(x, 3)) == "dirac(3, x)"
    # 检查 mcode 函数对 Heaviside(x) 的输出是否正确
    assert mcode(Heaviside(x)) == "heaviside(x, 1/2)"
    # 检查 mcode 函数对 Heaviside(x, y) 的输出是否正确
    assert mcode(Heaviside(x, y)) == "heaviside(x, y)"
    # 检查 mcode 函数对 binomial(x, y) 的输出是否正确
    assert mcode(binomial(x, y)) == "bincoeff(x, y)"
    # 检查 mcode 函数对 Mod(x, y) 的输出是否正确
    assert mcode(Mod(x, y)) == "mod(x, y)"


def test_minmax():
    # 检查 mcode 函数对 Max(x, y) + Min(x, y) 的输出是否正确
    assert mcode(Max(x, y) + Min(x, y)) == "max(x, y) + min(x, y)"
    # 检查 mcode 函数对 Max(x, y, z) 的输出是否正确
    assert mcode(Max(x, y, z)) == "max(x, max(y, z))"
    # 检查 mcode 函数对 Min(x, y, z) 的输出是否正确
    assert mcode(Min(x, y, z)) == "min(x, min(y, z))"


def test_Pow():
    # 检查 mcode 函数对 x**3 的输出是否正确
    assert mcode(x**3) == "x.^3"
    # 检查 mcode 函数对 x**(y**3) 的输出是否正确
    assert mcode(x**(y**3)) == "x.^(y.^3)"
    # 检查 mcode 函数对 x**Rational(2, 3) 的输出是否正确
    assert mcode(x**Rational(2, 3)) == 'x.^(2/3)'
    # 创建 g 函数，检查 mcode 函数对复杂表达式的输出是否正确
    g = implemented_function('g', Lambda(x, 2*x))
    assert mcode(1/(g(x)*3.5)**(x - y**x)/(x**2 + y)) == \
        "(3.5*2*x).^(-x + y.^x)./(x.^2 + y)"
    # 检查 mcode 函数对 Mul(-2, x, Pow(Mul(y,y,evaluate=False), -1, evaluate=False)) 的输出是否正确
    # 针对 issue 14160
    assert mcode(Mul(-2, x, Pow(Mul(y,y,evaluate=False), -1, evaluate=False),
                                                evaluate=False)) == '-2*x./(y.*y)'


def test_basic_ops():
    # 检查 mcode 函数对 x*y 的输出是否正确
    assert mcode(x*y) == "x.*y"
    # 检查 mcode 函数对 x + y 的输出是否正确
    assert mcode(x + y) == "x + y"
    # 检查 mcode 函数对 x - y 的输出是否正确
    assert mcode(x - y) == "x - y"
    # 检查 mcode 函数对 -x 的输出是否正确
    assert mcode(-x) == "-x"


def test_1_over_x_and_sqrt():
    # 检查 mcode 函数对 1/x 的输出是否正确
    assert mcode(1/x) == '1./x'
    # 检查 mcode 函数对 x**-1 和 x**-1.0 的输出是否正确
    assert mcode(x**-1) == mcode(x**-1.0) == '1./x'
    # 检查 mcode 函数对 1/sqrt(x) 的输出是否正确
    assert mcode(1/sqrt(x)) == '1./sqrt(x)'
    # 检查 mcode 函数对 x**-S.Half 和 x**-0.5 的输出是否正确
    assert mcode(x**-S.Half) == mcode(x**-0.5) == '1./sqrt(x)'
    # 检查 mcode 函数对 sqrt(x) 的输出是否正确
    assert mcode(sqrt(x)) == 'sqrt(x)'
    # 检查 mcode 函数对 x**S.Half 和 x**0.5 的输出是否正确
    assert mcode(x**S.Half) == mcode(x**0.5) == 'sqrt(x)'
    # 检查 mcode 函数对 1/pi 的输出是否正确
    assert mcode(1/pi) == '1/pi'
    # 检查 mcode 函数对 pi**-1 和 pi**-1.0 的输出是否正确
    assert mcode(pi**-1) == mcode(pi**-1.0) == '1/pi'
    # 检查 mcode 函数对 pi**-0.5 的输出是否正确
    assert mcode(pi**-0.5) == '1/sqrt(pi)'


def test_mix_number_mult_symbols():
    # 检查 mcode 函数对 3*x 的输出是否正确
    assert mcode(3*x) == "3*x"
    # 检查 mcode 函数对 pi*x 的输出是否正确
    assert mcode(pi*x) == "pi*x"
    # 检查 mcode 函数对 3/x 的输出是否正确
    assert mcode(3/x) == "3./x"
    # 检查 mcode 函数对 pi/x 的输出是否正确
    assert mcode(pi/x) == "pi./x"
    # 检查 mcode 函数对 x/3 的输出是否正确
    assert mcode(x/3) == "x/3"
    # 检查 mcode 函数对 x/pi 的输出是否正确
    # 断言，验证 mcode 函数对于给定表达式的输出是否符合预期
    assert mcode(x/y*z) == "x.*z./y"
    
    # 断言，验证 mcode 函数对于给定表达式的输出是否符合预期
    assert mcode(1/x/y) == "1./(x.*y)"
    
    # 断言，验证 mcode 函数对于给定表达式的输出是否符合预期
    assert mcode(2*pi*x/y/z) == "2*pi*x./(y.*z)"
    
    # 断言，验证 mcode 函数对于给定表达式的输出是否符合预期
    assert mcode(3*pi/x) == "3*pi./x"
    
    # 断言，验证 mcode 函数对于给定表达式的输出是否符合预期
    assert mcode(S(3)/5) == "3/5"
    
    # 断言，验证 mcode 函数对于给定表达式的输出是否符合预期
    assert mcode(S(3)/5*x) == "3*x/5"
    
    # 断言，验证 mcode 函数对于给定表达式的输出是否符合预期
    assert mcode(x/y/z) == "x./(y.*z)"
    
    # 断言，验证 mcode 函数对于给定表达式的输出是否符合预期
    assert mcode((x+y)/z) == "(x + y)./z"
    
    # 断言，验证 mcode 函数对于给定表达式的输出是否符合预期
    assert mcode((x+y)/(z+x)) == "(x + y)./(x + z)"
    
    # 断言，验证 mcode 函数对于给定表达式的输出是否符合预期
    assert mcode((x+y)/EulerGamma) == "(x + y)/%s" % EulerGamma.evalf(17)
    
    # 断言，验证 mcode 函数对于给定表达式的输出是否符合预期
    assert mcode(x/3/pi) == "x/(3*pi)"
    
    # 断言，验证 mcode 函数对于给定表达式的输出是否符合预期
    assert mcode(S(3)/5*x*y/pi) == "3*x.*y/(5*pi)"
# 测试混合数值和符号的幂运算
def test_mix_number_pow_symbols():
    # 断言 mcode(pi**3) 的结果为 'pi^3'
    assert mcode(pi**3) == 'pi^3'
    # 断言 mcode(x**2) 的结果为 'x.^2'
    assert mcode(x**2) == 'x.^2'
    # 断言 mcode(x**(pi**3)) 的结果为 'x.^(pi^3)'
    assert mcode(x**(pi**3)) == 'x.^(pi^3)'
    # 断言 mcode(x**y) 的结果为 'x.^y'
    assert mcode(x**y) == 'x.^y'
    # 断言 mcode(x**(y**z)) 的结果为 'x.^(y.^z)'
    assert mcode(x**(y**z)) == 'x.^(y.^z)'
    # 断言 mcode((x**y)**z) 的结果为 '(x.^y).^z'
    assert mcode((x**y)**z) == '(x.^y).^z'


# 测试虚数常量
def test_imag():
    I = S('I')
    # 断言 mcode(I) 的结果为 "1i"
    assert mcode(I) == "1i"
    # 断言 mcode(5*I) 的结果为 "5i"
    assert mcode(5*I) == "5i"
    # 断言 mcode((S(3)/2)*I) 的结果为 "3*1i/2"
    assert mcode((S(3)/2)*I) == "3*1i/2"
    # 断言 mcode(3+4*I) 的结果为 "3 + 4i"
    assert mcode(3+4*I) == "3 + 4i"
    # 断言 mcode(sqrt(3)*I) 的结果为 "sqrt(3)*1i"
    assert mcode(sqrt(3)*I) == "sqrt(3)*1i"


# 测试数学常数
def test_constants():
    # 断言 mcode(pi) 的结果为 "pi"
    assert mcode(pi) == "pi"
    # 断言 mcode(oo) 的结果为 "inf"
    assert mcode(oo) == "inf"
    # 断言 mcode(-oo) 的结果为 "-inf"
    assert mcode(-oo) == "-inf"
    # 断言 mcode(S.NegativeInfinity) 的结果为 "-inf"
    assert mcode(S.NegativeInfinity) == "-inf"
    # 断言 mcode(S.NaN) 的结果为 "NaN"
    assert mcode(S.NaN) == "NaN"
    # 断言 mcode(S.Exp1) 的结果为 "exp(1)"
    assert mcode(S.Exp1) == "exp(1)"
    # 断言 mcode(exp(1)) 的结果为 "exp(1)"
    assert mcode(exp(1)) == "exp(1)"


# 测试其它常数
def test_constants_other():
    # 断言 mcode(2*GoldenRatio) 的结果为 "2*(1+sqrt(5))/2"
    assert mcode(2*GoldenRatio) == "2*(1+sqrt(5))/2"
    # 断言 mcode(2*Catalan) 的结果为 "2*%s" % Catalan.evalf(17)
    assert mcode(2*Catalan) == "2*%s" % Catalan.evalf(17)
    # 断言 mcode(2*EulerGamma) 的结果为 "2*%s" % EulerGamma.evalf(17)
    assert mcode(2*EulerGamma) == "2*%s" % EulerGamma.evalf(17)


# 测试布尔运算
def test_boolean():
    # 断言 mcode(x & y) 的结果为 "x & y"
    assert mcode(x & y) == "x & y"
    # 断言 mcode(x | y) 的结果为 "x | y"
    assert mcode(x | y) == "x | y"
    # 断言 mcode(~x) 的结果为 "~x"
    assert mcode(~x) == "~x"
    # 断言 mcode(x & y & z) 的结果为 "x & y & z"
    assert mcode(x & y & z) == "x & y & z"
    # 断言 mcode(x | y | z) 的结果为 "x | y | z"
    assert mcode(x | y | z) == "x | y | z"
    # 断言 mcode((x & y) | z) 的结果为 "z | x & y"
    assert mcode((x & y) | z) == "z | x & y"
    # 断言 mcode((x | y) & z) 的结果为 "z & (x | y)"
    assert mcode((x | y) & z) == "z & (x | y)"


# 测试 KroneckerDelta 函数
def test_KroneckerDelta():
    from sympy.functions import KroneckerDelta
    # 断言 mcode(KroneckerDelta(x, y)) 的结果为 "double(x == y)"
    assert mcode(KroneckerDelta(x, y)) == "double(x == y)"
    # 断言 mcode(KroneckerDelta(x, y + 1)) 的结果为 "double(x == (y + 1))"
    assert mcode(KroneckerDelta(x, y + 1)) == "double(x == (y + 1))"
    # 断言 mcode(KroneckerDelta(2**x, y)) 的结果为 "double((2.^x) == y)"
    assert mcode(KroneckerDelta(2**x, y)) == "double((2.^x) == y)"


# 测试矩阵
def test_Matrices():
    # 断言 mcode(Matrix(1, 1, [10])) 的结果为 "10"
    assert mcode(Matrix(1, 1, [10])) == "10"
    A = Matrix([[1, sin(x/2), abs(x)],
                [0, 1, pi],
                [0, exp(1), ceiling(x)]]);
    expected = "[1 sin(x/2) abs(x); 0 1 pi; 0 exp(1) ceil(x)]"
    # 断言 mcode(A) 的结果为 expected
    assert mcode(A) == expected
    # 断言 mcode(A[:,0]) 的结果为 "[1; 0; 0]"
    assert mcode(A[:,0]) == "[1; 0; 0]"
    # 断言 mcode(A[0,:]) 的结果为 "[1 sin(x/2) abs(x)]"
    assert mcode(A[0,:]) == "[1 sin(x/2) abs(x)]"
    # 断言 mcode(Matrix(0, 0, [])) 的结果为 '[]'
    assert mcode(Matrix(0, 0, [])) == '[]'
    # 断言 mcode(Matrix(0, 3, [])) 的结果为 'zeros(0, 3)'
    assert mcode(Matrix(0, 3, [])) == 'zeros(0, 3)'
    # 断言 mcode(Matrix([[x, x - y, -y]])) 的结果为 "[x x - y -y]"
    assert mcode(Matrix([[x, x - y, -y]])) == "[x x - y -y]"


# 测试向量和哈达玛积
def test_vector_entries_hadamard():
    A = Matrix([[1, sin(2/x), 3*pi/x/5]])
    # 断言 mcode(A) 的结果为 "[1 sin(2./x) 3*pi./(5*x)]"
    assert mcode(A) == "[1 sin(2./x) 3*pi./(5*x)]"
    # 断言 mcode(A.T) 的结果为 "[1; sin(2./x); 3*pi./(5*x)]"
    assert mcode(A.T) == "[1; sin(2./x); 3*pi./(5*x)]"


# 测试 MatrixSymbol
def test_MatrixSymbol():
    n = Symbol('n', integer=True)
    A = MatrixSymbol('A', n, n)
    B = MatrixSymbol('B', n, n)
    # 断言 mcode(A*B) 的结果为 "A*B"
    assert mcode(A*B) == "A*B"
    # 断言 mcode(B*A) 的结果为 "B*A"
    assert mcode(B*A) == "B*A"
    # 断言：使用函数 mcode 对表达式 2*A*B 进行编码后的结果应为 "2*A*B"
    assert mcode(2*A*B) == "2*A*B"
    
    # 断言：使用函数 mcode 对表达式 B*2*A 进行编码后的结果应为 "2*B*A"
    assert mcode(B*2*A) == "2*B*A"
    
    # 断言：使用函数 mcode 对表达式 A*(B + 3*Identity(n)) 进行编码后的结果应为 "A*(3*eye(n) + B)"
    assert mcode(A*(B + 3*Identity(n))) == "A*(3*eye(n) + B)"
    
    # 断言：使用函数 mcode 对表达式 A**(x**2) 进行编码后的结果应为 "A^(x.^2)"
    assert mcode(A**(x**2)) == "A^(x.^2)"
    
    # 断言：使用函数 mcode 对表达式 A**3 进行编码后的结果应为 "A^3"
    assert mcode(A**3) == "A^3"
    
    # 断言：使用函数 mcode 对表达式 A**S.Half 进行编码后的结果应为 "A^(1/2)"
    assert mcode(A**S.Half) == "A^(1/2)"
def test_MatrixSolve():
    # 定义整数符号变量 n
    n = Symbol('n', integer=True)
    # 定义 n x n 的符号矩阵 A
    A = MatrixSymbol('A', n, n)
    # 定义 n x 1 的符号矩阵 x
    x = MatrixSymbol('x', n, 1)
    # 断言求解线性方程组 A * x = b 后的 Octave/MATLAB 代码
    assert mcode(MatrixSolve(A, x)) == "A \\ x"

def test_special_matrices():
    # 断言生成特定尺寸的单位矩阵的 Octave/MATLAB 代码
    assert mcode(6*Identity(3)) == "6*eye(3)"


def test_containers():
    # 断言转换 Python 列表为 Octave/MATLAB 代码
    assert mcode([1, 2, 3, [4, 5, [6, 7]], 8, [9, 10], 11]) == \
        "{1, 2, 3, {4, 5, {6, 7}}, 8, {9, 10}, 11}"
    # 断言转换 Python 元组为 Octave/MATLAB 代码
    assert mcode((1, 2, (3, 4))) == "{1, 2, {3, 4}}"
    # 断言转换包含单个元素的 Python 列表为 Octave/MATLAB 代码
    assert mcode([1]) == "{1}"
    # 断言转换包含单个元素的 Python 元组为 Octave/MATLAB 代码
    assert mcode((1,)) == "{1}"
    # 断言转换包含多个元素的 Python 元组为 Octave/MATLAB 代码
    assert mcode(Tuple(*[1, 2, 3])) == "{1, 2, 3}"
    # 断言转换包含表达式的 Python 元组为 Octave/MATLAB 代码
    assert mcode((1, x*y, (3, x**2))) == "{1, x.*y, {3, x.^2}}"
    # 断言转换包含不同类型数据的 Python 元组为 Octave/MATLAB 代码
    assert mcode((1, eye(3), Matrix(0, 0, []), [])) == "{1, [1 0 0; 0 1 0; 0 0 1], [], {}}"


def test_octave_noninline():
    # 断言非内联形式的 Octave/MATLAB 代码生成
    source = mcode((x+y)/Catalan, assign_to='me', inline=False)
    expected = (
        "Catalan = %s;\n"
        "me = (x + y)/Catalan;"
    ) % Catalan.evalf(17)
    assert source == expected


def test_octave_piecewise():
    # 断言分段函数的 Octave/MATLAB 代码生成
    expr = Piecewise((x, x < 1), (x**2, True))
    assert mcode(expr) == "((x < 1).*(x) + (~(x < 1)).*(x.^2))"
    assert mcode(expr, assign_to="r") == (
        "r = ((x < 1).*(x) + (~(x < 1)).*(x.^2));")
    assert mcode(expr, assign_to="r", inline=False) == (
        "if (x < 1)\n"
        "  r = x;\n"
        "else\n"
        "  r = x.^2;\n"
        "end")
    expr = Piecewise((x**2, x < 1), (x**3, x < 2), (x**4, x < 3), (x**5, True))
    expected = ("((x < 1).*(x.^2) + (~(x < 1)).*( ...\n"
                "(x < 2).*(x.^3) + (~(x < 2)).*( ...\n"
                "(x < 3).*(x.^4) + (~(x < 3)).*(x.^5))))")
    assert mcode(expr) == expected
    assert mcode(expr, assign_to="r") == "r = " + expected + ";"
    assert mcode(expr, assign_to="r", inline=False) == (
        "if (x < 1)\n"
        "  r = x.^2;\n"
        "elseif (x < 2)\n"
        "  r = x.^3;\n"
        "elseif (x < 3)\n"
        "  r = x.^4;\n"
        "else\n"
        "  r = x.^5;\n"
        "end")
    # 检查不包含默认条件 True 的分段函数的错误处理
    expr = Piecewise((x, x < 1), (x**2, x > 1), (sin(x), x > 0))
    raises(ValueError, lambda: mcode(expr))


def test_octave_piecewise_times_const():
    # 断言分段函数乘以常数的 Octave/MATLAB 代码生成
    pw = Piecewise((x, x < 1), (x**2, True))
    assert mcode(2*pw) == "2*((x < 1).*(x) + (~(x < 1)).*(x.^2))"
    assert mcode(pw/x) == "((x < 1).*(x) + (~(x < 1)).*(x.^2))./x"
    assert mcode(pw/(x*y)) == "((x < 1).*(x) + (~(x < 1)).*(x.^2))./(x.*y)"
    assert mcode(pw/3) == "((x < 1).*(x) + (~(x < 1)).*(x.^2))/3"


def test_octave_matrix_assign_to():
    # 断言分配矩阵到符号变量的 Octave/MATLAB 代码生成
    A = Matrix([[1, 2, 3]])
    assert mcode(A, assign_to='a') == "a = [1 2 3];"
    A = Matrix([[1, 2], [3, 4]])
    assert mcode(A, assign_to='A') == "A = [1 2; 3 4];"


def test_octave_matrix_assign_to_more():
    # 断言分配矩阵到符号矩阵变量的 Octave/MATLAB 代码生成
    A = Matrix([[1, 2, 3]])
    B = MatrixSymbol('B', 1, 3)
    C = MatrixSymbol('C', 2, 3)
    assert mcode(A, assign_to=B) == "B = [1 2 3];"
    # 使用 raises 函数来验证 mcode 函数在特定情况下是否会引发 ValueError 异常，lambda 函数用于传递 mcode 的调用
    raises(ValueError, lambda: mcode(A, assign_to=x))
    # 同上，但这里测试的是另一种情况，assign_to 参数为 C
    raises(ValueError, lambda: mcode(A, assign_to=C))
def test_octave_matrix_1x1():
    # 创建一个 1x1 的矩阵 A，包含单个元素 3
    A = Matrix([[3]])
    # 创建一个符号矩阵 B，大小为 1x1
    B = MatrixSymbol('B', 1, 1)
    # 创建一个符号矩阵 C，大小为 1x2
    C = MatrixSymbol('C', 1, 2)
    # 断言将矩阵 A 转换为 Octave/MATLAB 代码时，赋值给 B 的表达式为 "B = 3;"
    assert mcode(A, assign_to=B) == "B = 3;"
    # FIXME?
    # assert mcode(A, assign_to=x) == "x = 3;"
    # 断言将矩阵 A 转换为 Octave/MATLAB 代码时，赋值给 C 的操作会引发 ValueError 异常
    raises(ValueError, lambda: mcode(A, assign_to=C))


def test_octave_matrix_elements():
    # 创建一个包含符号变量的矩阵 A
    A = Matrix([[x, 2, x*y]])
    # 断言将矩阵 A 的第一行元素转换为 Octave/MATLAB 代码的表达式
    assert mcode(A[0, 0]**2 + A[0, 1] + A[0, 2]) == "x.^2 + x.*y + 2"
    # 创建一个符号矩阵符号化对象 AA，大小为 1x3
    A = MatrixSymbol('AA', 1, 3)
    # 断言将符号矩阵符号化对象 A 转换为 Octave/MATLAB 代码的表达式
    assert mcode(A) == "AA"
    # 断言将矩阵 A 的复杂表达式转换为 Octave/MATLAB 代码的表达式
    assert mcode(A[0, 0]**2 + sin(A[0,1]) + A[0,2]) == \
           "sin(AA(1, 2)) + AA(1, 1).^2 + AA(1, 3)"
    # 断言将矩阵 A 的所有元素求和转换为 Octave/MATLAB 代码的表达式
    assert mcode(sum(A)) == "AA(1, 1) + AA(1, 2) + AA(1, 3)"


def test_octave_boolean():
    # 断言将 Python 的 True 转换为 Octave/MATLAB 的表达式 "true"
    assert mcode(True) == "true"
    # 断言将 SymPy 的 S.true 转换为 Octave/MATLAB 的表达式 "true"
    assert mcode(S.true) == "true"
    # 断言将 Python 的 False 转换为 Octave/MATLAB 的表达式 "false"
    assert mcode(False) == "false"
    # 断言将 SymPy 的 S.false 转换为 Octave/MATLAB 的表达式 "false"
    assert mcode(S.false) == "false"


def test_octave_not_supported():
    # 断言对不支持的 SymPy 表达式抛出 NotImplementedError 异常
    with raises(NotImplementedError):
        mcode(S.ComplexInfinity)
    # 创建一个函数对象 f(x)，然后断言对其求导数的 Octave/MATLAB 转换结果
    assert mcode(f(x).diff(x), strict=False) == (
        "% Not supported in Octave:\n"
        "% Derivative\n"
        "Derivative(f(x), x)"
    )


def test_octave_not_supported_not_on_whitelist():
    # 导入一个不支持的 SymPy 函数并断言抛出 NotImplementedError 异常
    from sympy.functions.special.polynomials import assoc_laguerre
    with raises(NotImplementedError):
        mcode(assoc_laguerre(x, y, z))


def test_octave_expint():
    # 断言对 expint(1, x) 的 Octave/MATLAB 转换结果
    assert mcode(expint(1, x)) == "expint(x)"
    # 断言对 expint(2, x) 的 Octave/MATLAB 转换结果会抛出 NotImplementedError 异常
    with raises(NotImplementedError):
        mcode(expint(2, x))
    # 断言对 expint(y, x) 的 Octave/MATLAB 转换结果，且不严格检查转换
    assert mcode(expint(y, x), strict=False) == (
        "% Not supported in Octave:\n"
        "% expint\n"
        "expint(y, x)"
    )


def test_trick_indent_with_end_else_words():
    # 创建一个分段函数 pw，然后断言其在 Octave/MATLAB 中的转换结果
    pw = Piecewise((t1, x < 0), (t2, x <= 1), (1, True))
    assert mcode(pw, inline=False) == (
        "if (x < 0)\n"
        "  endless\n"
        "elseif (x <= 1)\n"
        "  elsewhere\n"
        "else\n"
        "  1\n"
        "end")


def test_hadamard():
    # 创建符号矩阵 A、B、v、h
    A = MatrixSymbol('A', 3, 3)
    B = MatrixSymbol('B', 3, 3)
    v = MatrixSymbol('v', 3, 1)
    h = MatrixSymbol('h', 1, 3)
    # 创建 Hadamard 乘积 C
    C = HadamardProduct(A, B)
    # 创建符号变量 n
    n = Symbol('n')
    # 断言将 Hadamard 乘积 C 转换为 Octave/MATLAB 代码的表达式
    assert mcode(C) == "A.*B"
    # 断言将 Hadamard 乘积 C 乘以向量 v 转换为 Octave/MATLAB 代码的表达式
    assert mcode(C*v) == "(A.*B)*v"
    # 断言将向量 h 乘以 Hadamard 乘积 C 乘以向量 v 转换为 Octave/MATLAB 代码的表达式
    assert mcode(h*C*v) == "h*(A.*B)*v"
    # 断言将 Hadamard 乘积 C 乘以矩阵 A 转换为 Octave/MATLAB 代码的表达式
    assert mcode(C*A) == "(A.*B)*A"
    # 断言将 Hadamard 乘积 C 乘以标量 x*y 转换为 Octave/MATLAB 代码的表达式
    assert mcode(C*x*y) == "(x.*y)*(A.*B)"

    # 测试 HadamardPower:
    # 断言将矩阵 A 的 HadamardPower(n) 转换为 Octave/MATLAB 代码的表达式
    assert mcode(HadamardPower(A, n)) == "A.**n"
    # 断言将矩阵 A 的 HadamardPower(1+n) 转换为 Octave/MATLAB 代码的表达式
    assert mcode(HadamardPower(A, 1+n)) == "A.**(n + 1)"
    # 断言将矩阵 A*B.T 的 HadamardPower(1+n) 转换为 Octave/MATLAB 代码的表达式
    assert mcode(HadamardPower(A*B.T, 1+n)) == "(A*B.T).**(n + 1)"


def test_sparse():
    # 创建一个稀疏矩阵 M
    M = SparseMatrix(5, 6, {})
    M[2, 2] = 10;
    M[1, 2] = 20;
    M[1, 3] = 22;
    M[0, 3] = 30;
    M[3, 0] = x*y;
    # 断言将稀疏矩阵 M 转换为 Octave/MATLAB 代码的表达式
    assert mcode(M) == (
        "sparse([4 2 3 1 2], [1 3 3 4 4], [x.*y 20 10 30 22], 5, 6)"
    )


def test_sinc():
    # 断言对 sinc(x) 的 Octave/MATLAB 转换
    # 断言测试：确保调用 mcode 函数并传入 sinc(pi*(x + 3)) 的结果等于字符串 'sinc(x + 3)'
    assert mcode(sinc(pi*(x + 3))) == 'sinc(x + 3)'
# 定义测试三角函数的函数
def test_trigfun():
    # 对每个三角函数进行迭代测试
    for f in (sin, cos, tan, cot, sec, csc, asin, acos, acot, atan, asec, acsc,
              sinh, cosh, tanh, coth, csch, sech, asinh, acosh, atanh, acoth,
              asech, acsch):
        # 断言每个函数的 Octave 代码是否正确
        assert octave_code(f(x) == f.__name__ + '(x)')


# 定义测试特殊函数的函数
def test_specfun():
    # 创建符号变量 n
    n = Symbol('n')
    # 对每个特殊函数进行迭代测试
    for f in [besselj, bessely, besseli, besselk]:
        # 断言每个函数的 Octave 代码是否正确
        assert octave_code(f(n, x)) == f.__name__ + '(n, x)'
    # 对一些其他特殊函数进行迭代测试
    for f in (erfc, erfi, erf, erfinv, erfcinv, fresnelc, fresnels, gamma):
        # 断言每个函数的 Octave 代码是否正确
        assert octave_code(f(x)) == f.__name__ + '(x)'
    # 断言汉克尔函数的 Octave 代码是否正确
    assert octave_code(hankel1(n, x)) == 'besselh(n, 1, x)'
    assert octave_code(hankel2(n, x)) == 'besselh(n, 2, x)'
    # 断言艾里函数的 Octave 代码是否正确
    assert octave_code(airyai(x)) == 'airy(0, x)'
    assert octave_code(airyaiprime(x)) == 'airy(1, x)'
    assert octave_code(airybi(x)) == 'airy(2, x)'
    assert octave_code(airybiprime(x)) == 'airy(3, x)'
    # 断言上伽马函数的 Octave 代码是否正确
    assert octave_code(uppergamma(n, x)) == '(gammainc(x, n, \'upper\').*gamma(n))'
    # 断言下伽马函数的 Octave 代码是否正确
    assert octave_code(lowergamma(n, x)) == '(gammainc(x, n).*gamma(n))'
    # 断言幂运算的 Octave 代码是否正确
    assert octave_code(z**lowergamma(n, x)) == 'z.^(gammainc(x, n).*gamma(n))'
    # 断言贝塞尔函数的 Octave 代码是否正确
    assert octave_code(jn(n, x)) == 'sqrt(2)*sqrt(pi)*sqrt(1./x).*besselj(n + 1/2, x)/2'
    assert octave_code(yn(n, x)) == 'sqrt(2)*sqrt(pi)*sqrt(1./x).*bessely(n + 1/2, x)/2'
    # 断言 Lambert W 函数的 Octave 代码是否正确
    assert octave_code(LambertW(x)) == 'lambertw(x)'
    assert octave_code(LambertW(x, n)) == 'lambertw(n, x)'
    # 自动重写的断言
    assert octave_code(Ei(x)) == '(logint(exp(x)))'
    assert octave_code(dirichlet_eta(x)) == '(((x == 1).*(log(2)) + (~(x == 1)).*((1 - 2.^(1 - x)).*zeta(x))))'
    assert octave_code(riemann_xi(x)) == '(pi.^(-x/2).*x.*(x - 1).*gamma(x/2).*zeta(x)/2)'


# 测试 MatrixElement 打印问题的函数
def test_MatrixElement_printing():
    # 测试 issue #11821 的用例
    A = MatrixSymbol("A", 1, 3)
    B = MatrixSymbol("B", 1, 3)
    C = MatrixSymbol("C", 1, 3)

    # 断言 MatrixElement 的 Octave 代码是否正确
    assert mcode(A[0, 0]) == "A(1, 1)"
    assert mcode(3 * A[0, 0]) == "3*A(1, 1)"

    # 断言符号表达式的 Octave 代码是否正确
    F = C[0, 0].subs(C, A - B)
    assert mcode(F) == "(A - B)(1, 1)"


# 测试 zeta 函数打印问题的函数
def test_zeta_printing_issue_14820():
    # 断言 zeta 函数的 Octave 代码是否正确
    assert octave_code(zeta(x)) == 'zeta(x)'
    # 使用 raises 检查未实现的情况
    with raises(NotImplementedError):
        octave_code(zeta(x, y))


# 测试自动重写的函数
def test_automatic_rewrite():
    # 断言 Li 函数的 Octave 代码是否正确
    assert octave_code(Li(x)) == '(logint(x) - logint(2))'
    # 断言 erf2 函数的 Octave 代码是否正确
    assert octave_code(erf2(x, y)) == '(-erf(x) + erf(y))'
```