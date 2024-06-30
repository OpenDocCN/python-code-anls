# `D:\src\scipysrc\sympy\sympy\printing\tests\test_fortran.py`

```
# 导入必要的 SymPy 模块和类
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import (Function, Lambda, diff)
from sympy.core.mod import Mod
from sympy.core import (Catalan, EulerGamma, GoldenRatio)
from sympy.core.numbers import (E, Float, I, Integer, Rational, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import (conjugate, sign)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (atan2, cos, sin)
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import Integral
from sympy.sets.fancysets import Range

# 导入 SymPy 代码生成相关模块和类
from sympy.codegen import For, Assignment, aug_assign
from sympy.codegen.ast import Declaration, Variable, float32, float64, \
        value_const, real, bool_, While, FunctionPrototype, FunctionDefinition, \
        integer, Return, Element
from sympy.core.expr import UnevaluatedExpr
from sympy.core.relational import Relational
from sympy.logic.boolalg import And, Or, Not, Equivalent, Xor
from sympy.matrices import Matrix, MatrixSymbol
from sympy.printing.fortran import fcode, FCodePrinter
from sympy.tensor import IndexedBase, Idx
from sympy.tensor.array.expressions import ArraySymbol, ArrayElement
from sympy.utilities.lambdify import implemented_function
from sympy.testing.pytest import raises

# 定义测试函数 test_UnevaluatedExpr()
def test_UnevaluatedExpr():
    # 定义实数符号 p, q, r
    p, q, r = symbols("p q r", real=True)
    # 创建 UnevaluatedExpr 对象 q_r，表示 q + r 的表达式
    q_r = UnevaluatedExpr(q + r)
    # 创建表达式 expr，表示 exp(p + q_r) 的绝对值
    expr = abs(exp(p+q_r))
    # 断言 fcode 函数处理 expr 后的结果等于 "exp(p + (q + r))"
    assert fcode(expr, source_format="free") == "exp(p + (q + r))"
    # 定义复数符号 x, y, z
    x, y, z = symbols("x y z")
    # 创建 UnevaluatedExpr 对象 y_z，表示 y + z 的表达式
    y_z = UnevaluatedExpr(y + z)
    # 创建表达式 expr2，表示 exp(x+y_z) 的绝对值
    expr2 = abs(exp(x+y_z))
    # 断言 fcode 函数处理 expr2 后的结果（不含人类可读标记）以 "exp(re(x) + re(y + z))" 开头
    assert fcode(expr2, human=False)[2].lstrip() == "exp(re(x) + re(y + z))"
    # 断言 fcode 函数处理 expr2 后的结果（使用自定义函数 re）等于 "exp(realpart(x) + realpart(y + z))"
    assert fcode(expr2, user_functions={"re": "realpart"}).lstrip() == "exp(realpart(x) + realpart(y + z))"

# 定义测试函数 test_printmethod()
def test_printmethod():
    # 定义符号 x
    x = symbols('x')

    # 定义 nint 函数类，继承自 Function 类
    class nint(Function):
        # 重载 _fcode 方法，返回格式化的字符串 "nint(%s)" % printer._print(self.args[0])
        def _fcode(self, printer):
            return "nint(%s)" % printer._print(self.args[0])
    
    # 断言 fcode 函数处理 nint(x) 后的结果等于 "      nint(x)"
    assert fcode(nint(x)) == "      nint(x)"

# 定义测试函数 test_fcode_sign()
def test_fcode_sign():  #issue 12267
    # 定义符号 x, y, z
    x=symbols('x')
    y=symbols('y', integer=True)
    z=symbols('z', complex=True)
    # 断言 fcode 函数处理 sign(x) 后的结果，使用 standard=95 和 source_format='free' 参数
    assert fcode(sign(x), standard=95, source_format='free') == "merge(0d0, dsign(1d0, x), x == 0d0)"
    # 断言 fcode 函数处理 sign(y) 后的结果，使用 standard=95 和 source_format='free' 参数
    assert fcode(sign(y), standard=95, source_format='free') == "merge(0, isign(1, y), y == 0)"
    # 断言 fcode 函数处理 sign(z) 后的结果，使用 standard=95 和 source_format='free' 参数
    assert fcode(sign(z), standard=95, source_format='free') == "merge(cmplx(0d0, 0d0), z/abs(z), abs(z) == 0d0)"
    # 断言 lambda 函数抛出 NotImplementedError 异常，当处理 sign(x) 时
    raises(NotImplementedError, lambda: fcode(sign(x)))

# 定义测试函数 test_fcode_Pow()
def test_fcode_Pow():
    # 定义符号 x, y
    x, y = symbols('x,y')
    # 定义整数符号 n
    n = symbols('n', integer=True)

    # 断言 fcode 函数处理 x**3 后的结果等于 "      x**3"
    assert fcode(x**3) == "      x**3"
    # 断言 fcode 函数处理 x**(y**3) 后的结果等于 "      x**(y**3)"
    assert fcode(x**(y**3)) == "      x**(y**3)"
    # 确保函数 fcode 正确转换给定表达式为 Fortran 代码的格式
    assert fcode(1/(sin(x)*3.5)**(x - y**x)/(x**2 + y)) == \
        "      (3.5d0*sin(x))**(-x + y**x)/(x**2 + y)"
    
    # 确保函数 fcode 正确将 sqrt(x) 转换为 Fortran 代码格式
    assert fcode(sqrt(x)) == '      sqrt(x)'
    
    # 确保函数 fcode 正确将 sqrt(n) 转换为 Fortran 代码格式，使用 dble(n) 转换 n 为双精度
    assert fcode(sqrt(n)) == '      sqrt(dble(n))'
    
    # 确保函数 fcode 正确将 x**0.5 转换为 Fortran 代码格式
    assert fcode(x**0.5) == '      sqrt(x)'
    
    # 确保函数 fcode 正确将 sqrt(x) 转换为 Fortran 代码格式
    assert fcode(sqrt(x)) == '      sqrt(x)'
    
    # 确保函数 fcode 正确将 sqrt(10) 转换为 Fortran 代码格式，将 10 转换为双精度
    assert fcode(sqrt(10)) == '      sqrt(10.0d0)'
    
    # 确保函数 fcode 正确将 x**-1.0 转换为 Fortran 代码格式
    assert fcode(x**-1.0) == '      1d0/x'
    
    # 确保函数 fcode 正确将 x**-2.0 转换为 Fortran 代码格式，使用自定义变量名 'y'，并且输出格式为自由格式
    assert fcode(x**Rational(3, 7)) == '      x**(3.0d0/7.0d0)'
# 定义测试函数 test_fcode_Rational
def test_fcode_Rational():
    # 定义符号变量 x
    x = symbols('x')
    # 断言 Rational(3, 7) 被格式化为 "      3.0d0/7.0d0"
    assert fcode(Rational(3, 7)) == "      3.0d0/7.0d0"
    # 断言 Rational(18, 9) 被格式化为 "      2"
    assert fcode(Rational(18, 9)) == "      2"
    # 断言 Rational(3, -7) 被格式化为 "      -3.0d0/7.0d0"
    assert fcode(Rational(3, -7)) == "      -3.0d0/7.0d0"
    # 断言 Rational(-3, -7) 被格式化为 "      3.0d0/7.0d0"
    assert fcode(Rational(-3, -7)) == "      3.0d0/7.0d0"
    # 断言 x + Rational(3, 7) 被格式化为 "      x + 3.0d0/7.0d0"
    assert fcode(x + Rational(3, 7)) == "      x + 3.0d0/7.0d0"
    # 断言 Rational(3, 7)*x 被格式化为 "      (3.0d0/7.0d0)*x"
    assert fcode(Rational(3, 7)*x) == "      (3.0d0/7.0d0)*x"


# 定义测试函数 test_fcode_Integer
def test_fcode_Integer():
    # 断言 Integer(67) 被格式化为 "      67"
    assert fcode(Integer(67)) == "      67"
    # 断言 Integer(-1) 被格式化为 "      -1"
    assert fcode(Integer(-1)) == "      -1"


# 定义测试函数 test_fcode_Float
def test_fcode_Float():
    # 断言 Float(42.0) 被格式化为 "      42.0000000000000d0"
    assert fcode(Float(42.0)) == "      42.0000000000000d0"
    # 断言 Float(-1e20) 被格式化为 "      -1.00000000000000d+20"
    assert fcode(Float(-1e20)) == "      -1.00000000000000d+20"


# 定义测试函数 test_fcode_functions
def test_fcode_functions():
    # 定义符号变量 x, y
    x, y = symbols('x,y')
    # 断言 sin(x) ** cos(y) 被格式化为 "      sin(x)**cos(y)"
    assert fcode(sin(x) ** cos(y)) == "      sin(x)**cos(y)"
    # 断言 Mod(x, y) 抛出 NotImplementedError
    raises(NotImplementedError, lambda: fcode(Mod(x, y), standard=66))
    # 断言 x % y 抛出 NotImplementedError
    raises(NotImplementedError, lambda: fcode(x % y, standard=66))
    # 断言 Mod(x, y) 抛出 NotImplementedError
    raises(NotImplementedError, lambda: fcode(Mod(x, y), standard=77))
    # 断言 x % y 抛出 NotImplementedError
    raises(NotImplementedError, lambda: fcode(x % y, standard=77))
    # 遍历多个标准版本
    for standard in [90, 95, 2003, 2008]:
        # 断言 Mod(x, y) 被格式化为 "      modulo(x, y)"
        assert fcode(Mod(x, y), standard=standard) == "      modulo(x, y)"
        # 断言 x % y 被格式化为 "      modulo(x, y)"
        assert fcode(x % y, standard=standard) == "      modulo(x, y)"


# 定义测试函数 test_case
def test_case():
    # 创建 FCodePrinter 对象
    ob = FCodePrinter()
    # 定义多个符号变量
    x, x_, x__, y, X, X_, Y = symbols('x,x_,x__,y,X,X_,Y')
    # 断言 exp(x_) + sin(x*y) + cos(X*Y) 被格式化为 '      exp(x_) + sin(x*y) + cos(X__*Y_)'
    assert fcode(exp(x_) + sin(x*y) + cos(X*Y)) == '      exp(x_) + sin(x*y) + cos(X__*Y_)'
    # 断言 exp(x__) + 2*x*Y*X_**Rational(7, 2) 被格式化为 '      2*X_**(7.0d0/2.0d0)*Y*x + exp(x__)'
    assert fcode(exp(x__) + 2*x*Y*X_**Rational(7, 2)) == '      2*X_**(7.0d0/2.0d0)*Y*x + exp(x__)'
    # 断言 exp(x_) + sin(x*y) + cos(X*Y)，关闭名称混淆时被格式化为 '      exp(x_) + sin(x*y) + cos(X*Y)'
    assert fcode(exp(x_) + sin(x*y) + cos(X*Y), name_mangling=False) == '      exp(x_) + sin(x*y) + cos(X*Y)'
    # 断言 x - cos(X)，关闭名称混淆时被格式化为 '      x - cos(X)'
    assert fcode(x - cos(X), name_mangling=False) == '      x - cos(X)'
    # 断言 ob.doprint(X*sin(x) + x_, assign_to='me') 被格式化为 '      me = X*sin(x_) + x__'
    assert ob.doprint(X*sin(x) + x_, assign_to='me') == '      me = X*sin(x_) + x__'
    # 断言 ob.doprint(X*sin(x), assign_to='mu') 被格式化为 '      mu = X*sin(x_)'
    assert ob.doprint(X*sin(x), assign_to='mu') == '      mu = X*sin(x_)'
    # 断言 ob.doprint(x_, assign_to='ad') 被格式化为 '      ad = x__'
    assert ob.doprint(x_, assign_to='ad') == '      ad = x__'
    # 定义整数符号变量 n, m
    n, m = symbols('n,m', integer=True)
    # 定义 IndexedBase 对象 A, x, y
    A = IndexedBase('A')
    x = IndexedBase('x')
    y = IndexedBase('y')
    # 定义索引变量 i, I
    i = Idx('i', m)
    I = Idx('I', n)
    # 断言 fcode(A[i, I]*x[I], assign_to=y[i], source_format='free') 被格式化为如下字符串
    assert fcode(A[i, I]*x[I], assign_to=y[i], source_format='free') == (
                                            "do i = 1, m\n"
                                            "   y(i) = 0\n"
                                            "end do\n"
                                            "do i = 1, m\n"
                                            "   do I_ = 1, n\n"
                                            "      y(i) = A(i, I_)*x(I_) + y(i)\n"
                                            "   end do\n"
                                            "end do" )


# 定义测试函数 test_fcode_functions_with_integers
def test_fcode_functions_with_integers():
    # 定义符号变量 x
    x = symbols('x')
    # 计算 log(10) 的数值，精确到 17 位小数
    log10_17 = log(10).evalf(17)
    # 预计算的 log(10) 的值
    loglog10_17 = '0.8340324452479558d0'
    # 断言 x * log(10) 被格式化为 "      x*%sd0" % log10_17
    assert fcode(x * log(10)) == "      x*%sd0" % log10_17
    # 确认调用 fcode 函数处理 x * log(10) 表达式后的输出是否与预期相符
    assert fcode(x * log(10)) == "      x*%sd0" % log10_17
    # 确认调用 fcode 函数处理 x * log(S(10)) 表达式后的输出是否与预期相符
    assert fcode(x * log(S(10))) == "      x*%sd0" % log10_17
    # 确认调用 fcode 函数处理 log(S(10)) 表达式后的输出是否与预期相符
    assert fcode(log(S(10))) == "      %sd0" % log10_17
    # 确认调用 fcode 函数处理 exp(10) 表达式后的输出是否与预期相符
    assert fcode(exp(10)) == "      %sd0" % exp(10).evalf(17)
    # 确认调用 fcode 函数处理 x * log(log(10)) 表达式后的输出是否与预期相符
    assert fcode(x * log(log(10))) == "      x*%s" % loglog10_17
    # 确认调用 fcode 函数处理 x * log(log(S(10))) 表达式后的输出是否与预期相符
    assert fcode(x * log(log(S(10)))) == "      x*%s" % loglog10_17
# 定义测试函数 test_fcode_NumberSymbol，用于测试 fcode 函数处理数学常数的情况
def test_fcode_NumberSymbol():
    # 设置精度为 17
    prec = 17
    # 创建 FCodePrinter 实例 p
    p = FCodePrinter()
    # 断言输出 Catala 常数的 Fortran 代码
    assert fcode(Catalan) == '      parameter (Catalan = %sd0)\n      Catalan' % Catalan.evalf(prec)
    # 断言输出 EulerGamma 常数的 Fortran 代码
    assert fcode(EulerGamma) == '      parameter (EulerGamma = %sd0)\n      EulerGamma' % EulerGamma.evalf(prec)
    # 断言输出 E 常数的 Fortran 代码
    assert fcode(E) == '      parameter (E = %sd0)\n      E' % E.evalf(prec)
    # 断言输出 GoldenRatio 常数的 Fortran 代码
    assert fcode(GoldenRatio) == '      parameter (GoldenRatio = %sd0)\n      GoldenRatio' % GoldenRatio.evalf(prec)
    # 断言输出 pi 常数的 Fortran 代码
    assert fcode(pi) == '      parameter (pi = %sd0)\n      pi' % pi.evalf(prec)
    # 断言输出带有 5 位精度的 pi 常数的 Fortran 代码
    assert fcode(pi, precision=5) == '      parameter (pi = %sd0)\n      pi' % pi.evalf(5)
    # 断言以非人类可读方式输出 Catala 常数的 Fortran 代码
    assert fcode(Catalan, human=False) == ({(Catalan, p._print(Catalan.evalf(prec)))}, set(), '      Catalan')
    # 断言以非人类可读方式输出 EulerGamma 常数的 Fortran 代码
    assert fcode(EulerGamma, human=False) == ({(EulerGamma, p._print(EulerGamma.evalf(prec)))}, set(), '      EulerGamma')
    # 断言以非人类可读方式输出 E 常数的 Fortran 代码
    assert fcode(E, human=False) == ({(E, p._print(E.evalf(prec)))}, set(), '      E')
    # 断言以非人类可读方式输出 GoldenRatio 常数的 Fortran 代码
    assert fcode(GoldenRatio, human=False) == ({(GoldenRatio, p._print(GoldenRatio.evalf(prec)))}, set(), '      GoldenRatio')
    # 断言以非人类可读方式输出 pi 常数的 Fortran 代码
    assert fcode(pi, human=False) == ({(pi, p._print(pi.evalf(prec)))}, set(), '      pi')
    # 断言以非人类可读方式输出带有 5 位精度的 pi 常数的 Fortran 代码
    assert fcode(pi, precision=5, human=False) == ({(pi, p._print(pi.evalf(5)))}, set(), '      pi')


# 定义测试函数 test_fcode_complex，用于测试 fcode 函数处理复数的情况
def test_fcode_complex():
    # 断言输出虚数单位 I 的 Fortran 代码
    assert fcode(I) == "      cmplx(0,1)"
    # 定义符号 x
    x = symbols('x')
    # 断言输出 4*I 的 Fortran 代码
    assert fcode(4*I) == "      cmplx(0,4)"
    # 断言输出 3 + 4*I 的 Fortran 代码
    assert fcode(3 + 4*I) == "      cmplx(3,4)"
    # 断言输出 3 + 4*I + x 的 Fortran 代码
    assert fcode(3 + 4*I + x) == "      cmplx(3,4) + x"
    # 断言输出 I*x 的 Fortran 代码
    assert fcode(I*x) == "      cmplx(0,1)*x"
    # 断言输出 3 + 4*I - x 的 Fortran 代码
    assert fcode(3 + 4*I - x) == "      cmplx(3,4) - x"
    # 重新定义符号 x，其虚部为 True
    x = symbols('x', imaginary=True)
    # 断言输出 5*x 的 Fortran 代码
    assert fcode(5*x) == "      5*x"
    # 断言输出 I*x 的 Fortran 代码
    assert fcode(I*x) == "      cmplx(0,1)*x"
    # 断言输出 3 + x 的 Fortran 代码
    assert fcode(3 + x) == "      x + 3"


# 定义测试函数 test_implicit，用于测试 fcode 函数处理隐式函数的情况
def test_implicit():
    # 定义符号 x, y
    x, y = symbols('x,y')
    # 断言输出 sin(x) 的 Fortran 代码
    assert fcode(sin(x)) == "      sin(x)"
    # 断言输出 atan2(x, y) 的 Fortran 代码
    assert fcode(atan2(x, y)) == "      atan2(x, y)"
    # 断言输出 conjugate(x) 的 Fortran 代码
    assert fcode(conjugate(x)) == "      conjg(x)"


# 定义测试函数 test_not_fortran，用于测试 fcode 函数在不支持情况下的行为
def test_not_fortran():
    # 定义符号 x
    x = symbols('x')
    # 创建函数 g
    g = Function('g')
    # 使用 raises 断言 gamma(x) 在 Fortran 中不支持
    with raises(NotImplementedError):
        fcode(gamma(x))
    # 断言不严格模式下输出 Integral(sin(x)) 的 Fortran 代码
    assert fcode(Integral(sin(x)), strict=False) == "C     Not supported in Fortran:\nC     Integral\n      Integral(sin(x), x)"
    # 使用 raises 断言 g(x) 在 Fortran 中不支持
    with raises(NotImplementedError):
        fcode(g(x))


# 定义测试函数 test_user_functions，用于测试 fcode 函数处理用户自定义函数的情况
def test_user_functions():
    # 定义符号 x
    x = symbols('x')
    # 断言输出 sin(x) 的 Fortran 代码，其中将 sin 函数映射为 zsin
    assert fcode(sin(x), user_functions={"sin": "zsin"}) == "      zsin(x)"
    # 重新定义符号 x
    x = symbols('x')
    # 断言输出 gamma(x) 的 Fortran 代码，其中将 gamma 函数映射为 mygamma
    assert fcode(gamma(x), user_functions={"gamma": "mygamma"}) == "      mygamma(x)"
    # 创建函数 g
    g = Function('g')
    # 断言输出 g(x) 的 Fortran 代码，其中将 g 函数映射为 great
    assert fcode(g(x), user_functions={"g": "great"}) == "      great(x)"
    # 定义符号 n，并指定为整数
    n = symbols('n', integer=True)
    # 断言输出 factorial(n) 的 Fortran 代码，其中将 factorial 函数映射为 fct
    assert fcode(factorial(n), user_functions={"factorial": "fct"}) == "      fct(n)"


# 定义测试函数 test_inline_function，用于测试 fcode 函数处理内联函数的情况
def test_inline_function():
    # 定义符号 x
    x = symbols('x')
    # 创建函数 g，并定义为 Lambda(x, 2*x)
    g = implemented_function('g', Lambda(x, 2*x))
    # 断言输出 g(x) 的 Fortran 代码
    assert fcode(g(x)) == "      2*x"
    # 使用 SymPy 的 implemented_function 创建一个函数 g(x)，定义为 2*pi/x
    g = implemented_function('g', Lambda(x, 2*pi/x))
    
    # 断言语句，验证 fcode(g(x)) 的输出是否与指定字符串匹配
    assert fcode(g(x)) == (
        "      parameter (pi = %sd0)\n"
        "      2*pi/x"
    ) % pi.evalf(17)
    
    # 创建一个 IndexedBase 对象 A，用于表示数组
    A = IndexedBase('A')
    
    # 创建一个指数 Idx 对象 i，表示一个整数符号，范围在 1 到 n 之间
    i = Idx('i', symbols('n', integer=True))
    
    # 使用 implemented_function 创建一个函数 g(x)，定义为 x*(1 + x)*(2 + x)
    g = implemented_function('g', Lambda(x, x*(1 + x)*(2 + x)))
    
    # 断言语句，验证 fcode(g(A[i]), assign_to=A[i]) 的输出是否与指定字符串匹配
    assert fcode(g(A[i]), assign_to=A[i]) == (
        "      do i = 1, n\n"
        "         A(i) = (A(i) + 1)*(A(i) + 2)*A(i)\n"
        "      end do"
    )
# 定义一个测试函数 `test_assign_to()`
def test_assign_to():
    # 创建符号变量 x
    x = symbols('x')
    # 断言 fcode(sin(x), assign_to="s") 的结果等于 "      s = sin(x)"

# 定义一个测试函数 `test_line_wrapping()`
def test_line_wrapping():
    # 创建符号变量 x, y
    x, y = symbols('x,y')
    # 断言 fcode(((x + y)**10).expand(), assign_to="var") 的结果等于多行字符串：
    # "      var = x**10 + 10*x**9*y + 45*x**8*y**2 + 120*x**7*y**3 + 210*x**6*
    #      @ y**4 + 252*x**5*y**5 + 210*x**4*y**6 + 120*x**3*y**7 + 45*x**2*y
    #      @ **8 + 10*x*y**9 + y**10"
    
    # 创建列表 e，其中每个元素为 x 的幂
    e = [x**i for i in range(11)]
    # 断言 fcode(Add(*e)) 的结果等于 "      x**10 + x**9 + x**8 + x**7 + x**6 + x**5 + x**4 + x**3 + x**2 + x
    #      @ + 1"

# 定义一个测试函数 `test_fcode_precedence()`
def test_fcode_precedence():
    # 创建符号变量 x, y
    x, y = symbols("x y")
    # 断言 fcode(And(x < y, y < x + 1), source_format="free") 的结果等于 "x < y .and. y < x + 1"
    # 断言 fcode(Or(x < y, y < x + 1), source_format="free") 的结果等于 "x < y .or. y < x + 1"
    # 断言 fcode(Xor(x < y, y < x + 1, evaluate=False), source_format="free") 的结果等于 "x < y .neqv. y < x + 1"
    # 断言 fcode(Equivalent(x < y, y < x + 1), source_format="free") 的结果等于 "x < y .eqv. y < x + 1"

# 定义一个测试函数 `test_fcode_Logical()`
def test_fcode_Logical():
    # 创建符号变量 x, y, z
    x, y, z = symbols("x y z")
    
    # 断言 fcode(Not(x), source_format="free") 的结果等于 ".not. x"
    # 断言 fcode(And(x, y), source_format="free") 的结果等于 "x .and. y"
    # 断言 fcode(And(x, Not(y)), source_format="free") 的结果等于 "x .and. .not. y"
    # 断言 fcode(And(Not(x), y), source_format="free") 的结果等于 "y .and. .not. x"
    # 断言 fcode(And(Not(x), Not(y)), source_format="free") 的结果等于 ".not. x .and. .not. y"
    # 断言 fcode(Not(And(x, y), evaluate=False), source_format="free") 的结果等于 ".not. (x .and. y)"
    
    # 断言 fcode(Or(x, y), source_format="free") 的结果等于 "x .or. y"
    # 断言 fcode(Or(x, Not(y)), source_format="free") 的结果等于 "x .or. .not. y"
    # 断言 fcode(Or(Not(x), y), source_format="free") 的结果等于 "y .or. .not. x"
    # 断言 fcode(Or(Not(x), Not(y)), source_format="free") 的结果等于 ".not. x .or. .not. y"
    # 断言 fcode(Not(Or(x, y), evaluate=False), source_format="free") 的结果等于 ".not. (x .or. y)"
    
    # 断言 fcode(And(Or(y, z), x), source_format="free") 的结果等于 "x .and. (y .or. z)"
    # 断言 fcode(And(Or(z, x), y), source_format="free") 的结果等于 "y .and. (x .or. z)"
    # 断言 fcode(And(Or(x, y), z), source_format="free") 的结果等于 "z .and. (x .or. y)"
    # 断言 fcode(Or(And(y, z), x), source_format="free") 的结果等于 "x .or. y .and. z"
    # 断言 fcode(Or(And(z, x), y), source_format="free") 的结果等于 "y .or. x .and. z"
    # 断言 fcode(Or(And(x, y), z), source_format="free") 的结果等于 "z .or. x .and. y"
    
    # 断言 fcode(And(x, y, z), source_format="free") 的结果等于 "x .and. y .and. z"
    # 断言 fcode(And(x, y, Not(z)), source_format="free") 的结果等于 "x .and. y .and. .not. z"
    # 断言 fcode(And(x, Not(y), z), source_format="free") 的结果等于 "x .and. z .and. .not. y"
    # 断言 fcode(And(Not(x), y, z), source_format="free") 的结果等于 "y .and. z .and. .not. x"
    # 断言：验证对于 Not(And(x, y, z), evaluate=False) 的转换是否正确
    assert fcode(Not(And(x, y, z), evaluate=False), source_format="free") == \
        ".not. (x .and. y .and. z)"
    # 断言：验证对于 Or(x, y, z) 的转换是否正确
    assert fcode(Or(x, y, z), source_format="free") == "x .or. y .or. z"
    # 断言：验证对于 Or(x, y, Not(z)) 的转换是否正确
    assert fcode(Or(x, y, Not(z)), source_format="free") == \
        "x .or. y .or. .not. z"
    # 断言：验证对于 Or(x, Not(y), z) 的转换是否正确
    assert fcode(Or(x, Not(y), z), source_format="free") == \
        "x .or. z .or. .not. y"
    # 断言：验证对于 Or(Not(x), y, z) 的转换是否正确
    assert fcode(Or(Not(x), y, z), source_format="free") == \
        "y .or. z .or. .not. x"
    # 断言：验证对于 Not(Or(x, y, z), evaluate=False) 的转换是否正确
    assert fcode(Not(Or(x, y, z), evaluate=False), source_format="free") == \
        ".not. (x .or. y .or. z)"
def test_fcode_Xlogical():
    # 定义符号变量 x, y, z
    x, y, z = symbols("x y z")
    # 测试二元异或操作，evaluate=False 禁用求值
    assert fcode(Xor(x, y, evaluate=False), source_format="free") == \
        "x .neqv. y"
    # 测试带有 Not 的二元异或操作
    assert fcode(Xor(x, Not(y), evaluate=False), source_format="free") == \
        "x .neqv. .not. y"
    # 测试另一种带有 Not 的二元异或操作
    assert fcode(Xor(Not(x), y, evaluate=False), source_format="free") == \
        "y .neqv. .not. x"
    # 测试带有两个 Not 的二元异或操作
    assert fcode(Xor(Not(x), Not(y), evaluate=False),
        source_format="free") == ".not. x .neqv. .not. y"
    # 测试对异或操作的否定
    assert fcode(Not(Xor(x, y, evaluate=False), evaluate=False),
        source_format="free") == ".not. (x .neqv. y)"
    # 测试等价操作
    assert fcode(Equivalent(x, y), source_format="free") == "x .eqv. y"
    # 测试带有 Not 的等价操作
    assert fcode(Equivalent(x, Not(y)), source_format="free") == \
        "x .eqv. .not. y"
    # 测试另一种带有 Not 的等价操作
    assert fcode(Equivalent(Not(x), y), source_format="free") == \
        "y .eqv. .not. x"
    # 测试带有两个 Not 的等价操作
    assert fcode(Equivalent(Not(x), Not(y)), source_format="free") == \
        ".not. x .eqv. .not. y"
    # 测试对等价操作的否定
    assert fcode(Not(Equivalent(x, y), evaluate=False),
        source_format="free") == ".not. (x .eqv. y)"
    # 测试混合 And 和 Equivalent 操作
    assert fcode(Equivalent(And(y, z), x), source_format="free") == \
        "x .eqv. y .and. z"
    assert fcode(Equivalent(And(z, x), y), source_format="free") == \
        "y .eqv. x .and. z"
    assert fcode(Equivalent(And(x, y), z), source_format="free") == \
        "z .eqv. x .and. y"
    assert fcode(And(Equivalent(y, z), x), source_format="free") == \
        "x .and. (y .eqv. z)"
    assert fcode(And(Equivalent(z, x), y), source_format="free") == \
        "y .and. (x .eqv. z)"
    assert fcode(And(Equivalent(x, y), z), source_format="free") == \
        "z .and. (x .eqv. y)"
    # 测试混合 Or 和 Equivalent 操作
    assert fcode(Equivalent(Or(y, z), x), source_format="free") == \
        "x .eqv. y .or. z"
    assert fcode(Equivalent(Or(z, x), y), source_format="free") == \
        "y .eqv. x .or. z"
    assert fcode(Equivalent(Or(x, y), z), source_format="free") == \
        "z .eqv. x .or. y"
    assert fcode(Or(Equivalent(y, z), x), source_format="free") == \
        "x .or. (y .eqv. z)"
    assert fcode(Or(Equivalent(z, x), y), source_format="free") == \
        "y .or. (x .eqv. z)"
    assert fcode(Or(Equivalent(x, y), z), source_format="free") == \
        "z .or. (x .eqv. y)"
    # 测试混合 Xor 和 Equivalent 操作
    assert fcode(Equivalent(Xor(y, z, evaluate=False), x),
        source_format="free") == "x .eqv. (y .neqv. z)"
    assert fcode(Equivalent(Xor(z, x, evaluate=False), y),
        source_format="free") == "y .eqv. (x .neqv. z)"
    assert fcode(Equivalent(Xor(x, y, evaluate=False), z),
        source_format="free") == "z .eqv. (x .neqv. y)"
    assert fcode(Xor(Equivalent(y, z), x, evaluate=False),
        source_format="free") == "x .neqv. (y .eqv. z)"
    assert fcode(Xor(Equivalent(z, x), y, evaluate=False),
        source_format="free") == "y .neqv. (x .eqv. z)"
    # 断言语句：验证函数 fcode 的输出是否符合预期结果，用于测试逻辑表达式与 Fortran 代码的转换
    
    # 测试混合 And/Xor 操作的逻辑表达式
    assert fcode(Xor(Equivalent(x, y), z, evaluate=False),
        source_format="free") == "z .neqv. (x .eqv. y)"
    
    # 测试混合 And/Xor 操作的逻辑表达式
    assert fcode(Xor(And(y, z), x, evaluate=False), source_format="free") == \
        "x .neqv. y .and. z"
    
    # 测试混合 And/Xor 操作的逻辑表达式
    assert fcode(Xor(And(z, x), y, evaluate=False), source_format="free") == \
        "y .neqv. x .and. z"
    
    # 测试混合 And/Xor 操作的逻辑表达式
    assert fcode(Xor(And(x, y), z, evaluate=False), source_format="free") == \
        "z .neqv. x .and. y"
    
    # 测试 And/Xor 操作的逻辑表达式
    assert fcode(And(Xor(y, z, evaluate=False), x), source_format="free") == \
        "x .and. (y .neqv. z)"
    
    # 测试 And/Xor 操作的逻辑表达式
    assert fcode(And(Xor(z, x, evaluate=False), y), source_format="free") == \
        "y .and. (x .neqv. z)"
    
    # 测试 And/Xor 操作的逻辑表达式
    assert fcode(And(Xor(x, y, evaluate=False), z), source_format="free") == \
        "z .and. (x .neqv. y)"
    
    # 测试混合 Or/Xor 操作的逻辑表达式
    assert fcode(Xor(Or(y, z), x, evaluate=False), source_format="free") == \
        "x .neqv. y .or. z"
    
    # 测试混合 Or/Xor 操作的逻辑表达式
    assert fcode(Xor(Or(z, x), y, evaluate=False), source_format="free") == \
        "y .neqv. x .or. z"
    
    # 测试混合 Or/Xor 操作的逻辑表达式
    assert fcode(Xor(Or(x, y), z, evaluate=False), source_format="free") == \
        "z .neqv. x .or. y"
    
    # 测试 Or/Xor 操作的逻辑表达式
    assert fcode(Or(Xor(y, z, evaluate=False), x), source_format="free") == \
        "x .or. (y .neqv. z)"
    
    # 测试 Or/Xor 操作的逻辑表达式
    assert fcode(Or(Xor(z, x, evaluate=False), y), source_format="free") == \
        "y .or. (x .neqv. z)"
    
    # 测试 Or/Xor 操作的逻辑表达式
    assert fcode(Or(Xor(x, y, evaluate=False), z), source_format="free") == \
        "z .or. (x .neqv. y)"
    
    # 测试三元 Xor 操作的逻辑表达式
    assert fcode(Xor(x, y, z, evaluate=False), source_format="free") == \
        "x .neqv. y .neqv. z"
    
    # 测试三元 Xor 操作的逻辑表达式
    assert fcode(Xor(x, y, Not(z), evaluate=False), source_format="free") == \
        "x .neqv. y .neqv. .not. z"
    
    # 测试三元 Xor 操作的逻辑表达式
    assert fcode(Xor(x, Not(y), z, evaluate=False), source_format="free") == \
        "x .neqv. z .neqv. .not. y"
    
    # 测试三元 Xor 操作的逻辑表达式
    assert fcode(Xor(Not(x), y, z, evaluate=False), source_format="free") == \
        "y .neqv. z .neqv. .not. x"
# 定义测试函数 test_fcode_Relational，用于测试关系运算表达式的转换
def test_fcode_Relational():
    # 定义符号变量 x 和 y
    x, y = symbols("x y")
    # 断言将关系表达式 x == y 转换为自由格式的 Fortran 代码 "x == y"
    assert fcode(Relational(x, y, "=="), source_format="free") == "x == y"
    # 断言将关系表达式 x != y 转换为自由格式的 Fortran 代码 "x /= y"
    assert fcode(Relational(x, y, "!="), source_format="free") == "x /= y"
    # 断言将关系表达式 x >= y 转换为自由格式的 Fortran 代码 "x >= y"
    assert fcode(Relational(x, y, ">="), source_format="free") == "x >= y"
    # 断言将关系表达式 x <= y 转换为自由格式的 Fortran 代码 "x <= y"
    assert fcode(Relational(x, y, "<="), source_format="free") == "x <= y"
    # 断言将关系表达式 x > y 转换为自由格式的 Fortran 代码 "x > y"
    assert fcode(Relational(x, y, ">"), source_format="free") == "x > y"
    # 断言将关系表达式 x < y 转换为自由格式的 Fortran 代码 "x < y"


# 定义测试函数 test_fcode_Piecewise，用于测试 Piecewise 函数的转换
def test_fcode_Piecewise():
    # 定义符号变量 x
    x = symbols('x')
    # 创建 Piecewise 表达式 expr
    expr = Piecewise((x, x < 1), (x**2, True))
    # 检查当没有指定标准版本时，是否会抛出 NotImplementedError 异常
    raises(NotImplementedError, lambda: fcode(expr))
    # 将 Piecewise 表达式 expr 转换为标准版本为 95 的 Fortran 代码
    code = fcode(expr, standard=95)
    expected = "      merge(x, x**2, x < 1)"
    assert code == expected
    # 断言将 Piecewise 表达式转换为赋值语句的 Fortran 代码块
    assert fcode(Piecewise((x, x < 1), (x**2, True)), assign_to="var") == (
        "      if (x < 1) then\n"
        "         var = x\n"
        "      else\n"
        "         var = x**2\n"
        "      end if"
    )
    # 定义两个数学表达式 a 和 b
    a = cos(x)/x
    b = sin(x)/x
    # 对 a 和 b 分别进行 10 次求导
    for i in range(10):
        a = diff(a, x)
        b = diff(b, x)
    # 预期的 Fortran 代码块，根据 x 的值不同选择不同的数学表达式
    expected = (
        "      if (x < 0) then\n"
        "         weird_name = -cos(x)/x + 10*sin(x)/x**2 + 90*cos(x)/x**3 - 720*\n"
        "     @ sin(x)/x**4 - 5040*cos(x)/x**5 + 30240*sin(x)/x**6 + 151200*cos(x\n"
        "     @ )/x**7 - 604800*sin(x)/x**8 - 1814400*cos(x)/x**9 + 3628800*sin(x\n"
        "     @ )/x**10 + 3628800*cos(x)/x**11\n"
        "      else\n"
        "         weird_name = -sin(x)/x - 10*cos(x)/x**2 + 90*sin(x)/x**3 + 720*\n"
        "     @ cos(x)/x**4 - 5040*sin(x)/x**5 - 30240*cos(x)/x**6 + 151200*sin(x\n"
        "     @ )/x**7 + 604800*cos(x)/x**8 - 1814400*sin(x)/x**9 - 3628800*cos(x\n"
        "     @ )/x**10 + 3628800*sin(x)/x**11\n"
        "      end if"
    )
    # 将 Piecewise 表达式转换为以 weird_name 为变量名的 Fortran 代码块
    code = fcode(Piecewise((a, x < 0), (b, True)), assign_to="weird_name")
    assert code == expected
    # 将包含三个条件的 Piecewise 表达式转换为标准版本为 95 的 Fortran 代码
    code = fcode(Piecewise((x, x < 1), (x**2, x > 1), (sin(x), True)), standard=95)
    expected = "      merge(x, merge(x**2, sin(x), x > 1), x < 1)"
    assert code == expected
    # 检查 Piecewise 表达式没有默认条件时是否会抛出 ValueError 异常
    expr = Piecewise((x, x < 1), (x**2, x > 1), (sin(x), x > 0))
    raises(ValueError, lambda: fcode(expr))


# 定义测试函数 test_wrap_fortran，用于测试 FCodePrinter 类的初始化
def test_wrap_fortran():
    # 创建 FCodePrinter 类的实例 printer
    printer = FCodePrinter()
    # 给定的多行字符串列表，包含长注释或Fortran语句，需要进行适当的换行处理
    lines = [
        "C     This is a long comment on a single line that must be wrapped properly to produce nice output",
        "      this = is + a + long + and + nasty + fortran + statement + that * must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +  that * must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +   that * must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement + that*must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +   that*must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +    that*must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +     that*must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement + that**must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +  that**must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +   that**must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +    that**must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +     that**must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement(that)/must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran +     statement(that)/must + be + wrapped + properly",
    ]
    
    # 使用 printer 对象的 _wrap_fortran 方法对给定的 Fortran 行进行适当的换行处理
    wrapped_lines = printer._wrap_fortran(lines)
    # 预期的包装过的行列表，每行必须在72个字符以内
    expected_lines = [
        "C     This is a long comment on a single line that must be wrapped",
        "C     properly to produce nice output",
        "      this = is + a + long + and + nasty + fortran + statement + that *",
        "     @ must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +  that *",
        "     @ must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +   that",
        "     @ * must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement + that*",
        "     @ must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +   that*",
        "     @ must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +    that",
        "     @ *must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +",
        "     @ that*must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement + that**",
        "     @ must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +  that**",
        "     @ must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +   that",
        "     @ **must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +    that",
        "     @ **must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +",
        "     @ that**must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement(that)/",
        "     @ must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran +     statement(that)",
        "     @ /must + be + wrapped + properly",
    ]
    # 检查每个包装过的行是否与预期的行相等
    for line in wrapped_lines:
        assert len(line) <= 72
    # 逐一比较包装过的行和预期的行是否完全一致
    for w, e in zip(wrapped_lines, expected_lines):
        assert w == e
    # 检查包装过的行和预期的行的数量是否相同
    assert len(wrapped_lines) == len(expected_lines)
# 定义一个名为 test_wrap_fortran_keep_d0 的测试函数
def test_wrap_fortran_keep_d0():
    # 创建 FCodePrinter 的实例对象
    printer = FCodePrinter()
    # 定义一个长列表，包含多个 Fortran 代码行
    lines = [
        '      this_variable_is_very_long_because_we_try_to_test_line_break=1.0d0',
        '      this_variable_is_very_long_because_we_try_to_test_line_break =1.0d0',
        '      this_variable_is_very_long_because_we_try_to_test_line_break  = 1.0d0',
        '      this_variable_is_very_long_because_we_try_to_test_line_break   = 1.0d0',
        '      this_variable_is_very_long_because_we_try_to_test_line_break    = 1.0d0',
        '      this_variable_is_very_long_because_we_try_to_test_line_break = 10.0d0'
    ]
    # 定义预期输出，包含被包装的 Fortran 代码行
    expected = [
        '      this_variable_is_very_long_because_we_try_to_test_line_break=1.0d0',
        '      this_variable_is_very_long_because_we_try_to_test_line_break =',
        '     @ 1.0d0',
        '      this_variable_is_very_long_because_we_try_to_test_line_break  =',
        '     @ 1.0d0',
        '      this_variable_is_very_long_because_we_try_to_test_line_break   =',
        '     @ 1.0d0',
        '      this_variable_is_very_long_because_we_try_to_test_line_break    =',
        '     @ 1.0d0',
        '      this_variable_is_very_long_because_we_try_to_test_line_break =',
        '     @ 10.0d0'
    ]
    # 断言调用 _wrap_fortran 方法后返回的结果符合预期
    assert printer._wrap_fortran(lines) == expected


# 定义一个名为 test_settings 的测试函数
def test_settings():
    # 使用 lambda 函数和 raises 断言检查调用 fcode 函数时，传入一个无效参数时是否引发 TypeError 异常
    raises(TypeError, lambda: fcode(S(4), method="garbage"))


# 定义一个名为 test_free_form_code_line 的测试函数
def test_free_form_code_line():
    # 创建符号变量 x 和 y
    x, y = symbols('x,y')
    # 使用 source_format='free' 参数调用 fcode 函数，检查生成的 Fortran 代码是否符合预期
    assert fcode(cos(x) + sin(y), source_format='free') == "sin(y) + cos(x)"


# 定义一个名为 test_free_form_continuation_line 的测试函数
def test_free_form_continuation_line():
    # 创建符号变量 x 和 y
    x, y = symbols('x,y')
    # 使用 source_format='free' 参数调用 fcode 函数，检查生成的 Fortran 代码是否符合预期
    result = fcode(((cos(x) + sin(y))**(7)).expand(), source_format='free')
    # 定义预期的 Fortran 代码输出，包含了多行的连续行
    expected = (
        'sin(y)**7 + 7*sin(y)**6*cos(x) + 21*sin(y)**5*cos(x)**2 + 35*sin(y)**4* &\n'
        '      cos(x)**3 + 35*sin(y)**3*cos(x)**4 + 21*sin(y)**2*cos(x)**5 + 7* &\n'
        '      sin(y)*cos(x)**6 + cos(x)**7'
    )
    # 断言生成的 Fortran 代码是否符合预期
    assert result == expected


# 定义一个名为 test_free_form_comment_line 的测试函数
def test_free_form_comment_line():
    # 创建 source_format='free' 的 FCodePrinter 实例对象
    printer = FCodePrinter({'source_format': 'free'})
    # 定义一个包含长注释的列表
    lines = [ "! This is a long comment on a single line that must be wrapped properly to produce nice output"]
    # 定义预期的注释被正确包装成多行输出
    expected = [
        '! This is a long comment on a single line that must be wrapped properly',
        '! to produce nice output'
    ]
    # 断言调用 _wrap_fortran 方法后返回的结果符合预期
    assert printer._wrap_fortran(lines) == expected


# 定义一个名为 test_loops 的测试函数
def test_loops():
    # 创建整数类型的符号变量 n 和 m
    n, m = symbols('n,m', integer=True)
    # 创建 IndexedBase 对象 A、x 和 y
    A = IndexedBase('A')
    x = IndexedBase('x')
    y = IndexedBase('y')
    # 创建索引对象 i 和 j
    i = Idx('i', m)
    j = Idx('j', n)
    # 定义预期的循环结构字符串，包含了嵌套的 do 循环和赋值操作
    expected = (
        'do i = 1, m\n'
        '   y(i) = 0\n'
        'end do\n'
        'do i = 1, m\n'
        '   do j = 1, n\n'
        '      y(i) = %(rhs)s\n'
        '   end do\n'
        'end do'
    )
    # 调用 fcode 函数，生成 Fortran 代码，检查其是否符合预期
    code = fcode(A[i, j]*x[j], assign_to=y[i], source_format='free')
    # 断言语句，验证 code 是否等于 expected 中的某种格式
    assert (code == expected % {'rhs': 'y(i) + A(i, j)*x(j)'} or
            code == expected % {'rhs': 'y(i) + x(j)*A(i, j)'} or
            code == expected % {'rhs': 'x(j)*A(i, j) + y(i)'} or
            code == expected % {'rhs': 'A(i, j)*x(j) + y(i)'})
def test_dummy_loops():
    # 创建符号变量 i 和 m，均为整数类型的虚拟变量
    i, m = symbols('i m', integer=True, cls=Dummy)
    # 创建 IndexedBase 对象 x 和 y
    x = IndexedBase('x')
    y = IndexedBase('y')
    # 创建索引对象 i，范围从 1 到 m
    i = Idx(i, m)

    # 格式化预期输出字符串，用于生成 Fortran 代码中的 do 循环
    expected = (
        'do i_%(icount)i = 1, m_%(mcount)i\n'
        '   y(i_%(icount)i) = x(i_%(icount)i)\n'
        'end do'
    ) % {'icount': i.label.dummy_index, 'mcount': m.dummy_index}
    # 将 x[i] 转换为 Fortran 代码，赋值给 y[i]，输出格式为自由格式
    code = fcode(x[i], assign_to=y[i], source_format='free')
    # 断言生成的代码与预期输出一致
    assert code == expected


def test_fcode_Indexed_without_looking_for_contraction():
    # 定义长度为 5 的 IndexedBase 对象 y、x 和 Dy
    len_y = 5
    y = IndexedBase('y', shape=(len_y,))
    x = IndexedBase('x', shape=(len_y,))
    Dy = IndexedBase('Dy', shape=(len_y-1,))
    # 创建索引对象 i，范围从 0 到 len_y-2
    i = Idx('i', len_y-1)
    # 创建表示差分方程的等式对象 e
    e = Eq(Dy[i], (y[i+1] - y[i]) / (x[i+1] - x[i]))
    # 将 e.rhs 转换为 Fortran 代码，赋值给 e.lhs，不进行合并操作
    code0 = fcode(e.rhs, assign_to=e.lhs, contract=False)
    # 断言生成的代码以特定字符串结尾
    assert code0.endswith('Dy(i) = (y(i + 1) - y(i))/(x(i + 1) - x(i))')


def test_element_like_objects():
    # 定义长度为 5 的 ArraySymbol 对象 y、x 和 Dy
    len_y = 5
    y = ArraySymbol('y', shape=(len_y,))
    x = ArraySymbol('x', shape=(len_y,))
    Dy = ArraySymbol('Dy', shape=(len_y-1,))
    # 创建索引对象 i，范围从 0 到 len_y-2
    i = Idx('i', len_y-1)
    # 创建表示差分方程的等式对象 e
    e = Eq(Dy[i], (y[i+1] - y[i]) / (x[i+1] - x[i]))
    # 将 e.lhs 和 e.rhs 转换为 Fortran 代码，进行赋值操作
    code0 = fcode(Assignment(e.lhs, e.rhs))
    # 断言生成的代码以特定字符串结尾
    assert code0.endswith('Dy(i) = (y(i + 1) - y(i))/(x(i + 1) - x(i))')

    # 定义 ElementExpr 类，继承自 Element 和 Expr
    class ElementExpr(Element, Expr):
        pass

    # 将 e 中的 ArrayElement 替换为 ElementExpr，并重新生成 Fortran 代码
    e = e.subs((a, ElementExpr(a.name, a.indices)) for a in e.atoms(ArrayElement))
    e = Eq(Dy[i], (y[i+1] - y[i]) / (x[i+1] - x[i]))
    code0 = fcode(Assignment(e.lhs, e.rhs))
    # 断言生成的代码以特定字符串结尾
    assert code0.endswith('Dy(i) = (y(i + 1) - y(i))/(x(i + 1) - x(i))')


def test_derived_classes():
    # 定义 MyFancyFCodePrinter 类，继承自 FCodePrinter
    class MyFancyFCodePrinter(FCodePrinter):
        _default_settings = FCodePrinter._default_settings.copy()

    # 创建 MyFancyFCodePrinter 的实例 printer
    printer = MyFancyFCodePrinter()
    # 创建符号变量 x
    x = symbols('x')
    # 使用 MyFancyFCodePrinter 实例 printer 输出 sin(x) 的 Fortran 代码
    assert printer.doprint(sin(x), "bork") == "      bork = sin(x)"


def test_indent():
    # 定义包含多行代码的字符串 codelines
    codelines = (
        'subroutine test(a)\n'
        'integer :: a, i, j\n'
        '\n'
        'do\n'
        'do \n'
        'do j = 1, 5\n'
        'if (a>b) then\n'
        'if(b>0) then\n'
        'a = 3\n'
        'donot_indent_me = 2\n'
        'do_not_indent_me_either = 2\n'
        'ifIam_indented_something_went_wrong = 2\n'
        'if_I_am_indented_something_went_wrong = 2\n'
        'end should not be unindented here\n'
        'end if\n'
        'endif\n'
        'end do\n'
        'end do\n'
        'enddo\n'
        'end subroutine\n'
        '\n'
        'subroutine test2(a)\n'
        'integer :: a\n'
        'do\n'
        'a = a + 1\n'
        'end do \n'
        'end subroutine\n'
    )
    expected = (
        'subroutine test(a)\n'  # 定义一个名为test的子程序，接受参数a
        'integer :: a, i, j\n'  # 声明整数变量a, i, j
        '\n'  # 空行
        'do\n'  # 开始一个无限循环
        '   do \n'  # 开始一个嵌套循环
        '      do j = 1, 5\n'  # 开始一个j从1到5的循环
        '         if (a>b) then\n'  # 如果a大于b则执行下面的代码块
        '            if(b>0) then\n'  # 如果b大于0则执行下面的代码块
        '               a = 3\n'  # 将a赋值为3
        '               donot_indent_me = 2\n'  # 不要缩进我等于2
        '               do_not_indent_me_either = 2\n'  # 不要缩进我等于2
        '               ifIam_indented_something_went_wrong = 2\n'  # 如果我被缩进，肯定出了什么问题等于2
        '               if_I_am_indented_something_went_wrong = 2\n'  # 如果我被缩进，肯定出了什么问题等于2
        '               end should not be unindented here\n'  # 这里的end不应该取消缩进
        '            end if\n'  # 结束内部if语句
        '         endif\n'  # 结束外部if语句
        '      end do\n'  # 结束j的循环
        '   end do\n'  # 结束嵌套的do循环
        'enddo\n'  # 结束无限循环
        'end subroutine\n'  # 结束子程序test
        '\n'  # 空行
        'subroutine test2(a)\n'  # 定义一个名为test2的子程序，接受参数a
        'integer :: a\n'  # 声明整数变量a
        'do\n'  # 开始一个无限循环
        '   a = a + 1\n'  # a自增1
        'end do \n'  # 结束循环
        'end subroutine\n'  # 结束子程序test2
    )
    p = FCodePrinter({'source_format': 'free'})  # 创建一个FCodePrinter对象，指定源代码格式为自由格式
    result = p.indent_code(codelines)  # 对输入的代码进行缩进处理
    assert result == expected  # 断言处理后的代码与期望的结果相同
def test_Matrix_printing():
    x, y, z = symbols('x,y,z')
    # 定义一个包含符号变量 x, y, z 的矩阵
    mat = Matrix([x*y, Piecewise((2 + x, y>0), (y, True)), sin(z)])
    # 定义一个矩阵符号 A，用于存储 mat 转换为 Fortran 代码后的结果
    A = MatrixSymbol('A', 3, 1)
    # 断言转换后的 Fortran 代码与预期的结果相等
    assert fcode(mat, A) == (
        "      A(1, 1) = x*y\n"
        "      if (y > 0) then\n"
        "         A(2, 1) = x + 2\n"
        "      else\n"
        "         A(2, 1) = y\n"
        "      end if\n"
        "      A(3, 1) = sin(z)")
    
    # 测试在表达式中使用矩阵元素
    expr = Piecewise((2*A[2, 0], x > 0), (A[2, 0], True)) + sin(A[1, 0]) + A[0, 0]
    # 断言转换后的 Fortran 代码与预期的结果相等
    assert fcode(expr, standard=95) == (
        "      merge(2*A(3, 1), A(3, 1), x > 0) + sin(A(2, 1)) + A(1, 1)")
    
    # 测试在矩阵中使用矩阵元素
    q = MatrixSymbol('q', 5, 1)
    M = MatrixSymbol('M', 3, 3)
    m = Matrix([[sin(q[1,0]), 0, cos(q[2,0])],
        [q[1,0] + q[2,0], q[3, 0], 5],
        [2*q[4, 0]/q[1,0], sqrt(q[0,0]) + 4, 0]])
    # 断言转换后的 Fortran 代码与预期的结果相等
    assert fcode(m, M) == (
        "      M(1, 1) = sin(q(2, 1))\n"
        "      M(2, 1) = q(2, 1) + q(3, 1)\n"
        "      M(3, 1) = 2*q(5, 1)/q(2, 1)\n"
        "      M(1, 2) = 0\n"
        "      M(2, 2) = q(4, 1)\n"
        "      M(3, 2) = sqrt(q(1, 1)) + 4\n"
        "      M(1, 3) = cos(q(3, 1))\n"
        "      M(2, 3) = 5\n"
        "      M(3, 3) = 0")


def test_fcode_For():
    x, y = symbols('x y')

    # 定义一个 For 循环，迭代 x 在区间 [0, 10) 中每隔 2 的值
    f = For(x, Range(0, 10, 2), [Assignment(y, x * y)])
    # 断言转换后的 Fortran 代码与预期的结果相等
    sol = fcode(f)
    assert sol == ("      do x = 0, 9, 2\n"
                   "         y = x*y\n"
                   "      end do")


def test_fcode_Declaration():
    def check(expr, ref, **kwargs):
        # 断言转换后的 Fortran 代码与预期的结果相等
        assert fcode(expr, standard=95, source_format='free', **kwargs) == ref

    i = symbols('i', integer=True)
    var1 = Variable.deduced(i)
    dcl1 = Declaration(var1)
    # 检查变量声明的 Fortran 代码
    check(dcl1, "integer*4 :: i")

    x, y = symbols('x y')
    var2 = Variable(x, float32, value=42, attrs={value_const})
    dcl2b = Declaration(var2)
    # 检查变量声明的 Fortran 代码
    check(dcl2b, 'real*4, parameter :: x = 42')

    var3 = Variable(y, type=bool_)
    dcl3 = Declaration(var3)
    # 检查变量声明的 Fortran 代码
    check(dcl3, 'logical :: y')

    # 检查类型别名的转换
    check(float32, "real*4")
    check(float64, "real*8")
    check(real, "real*4", type_aliases={real: float32})
    check(real, "real*8", type_aliases={real: float64})


def test_MatrixElement_printing():
    # 测试矩阵元素的打印
    A = MatrixSymbol("A", 1, 3)
    B = MatrixSymbol("B", 1, 3)
    C = MatrixSymbol("C", 1, 3)

    assert(fcode(A[0, 0]) == "      A(1, 1)")
    assert(fcode(3 * A[0, 0]) == "      3*A(1, 1)")

    F = C[0, 0].subs(C, A - B)
    assert(fcode(F) == "      (A - B)(1, 1)")


def test_aug_assign():
    x = symbols('x')
    # 测试增强赋值的转换
    assert fcode(aug_assign(x, '+', 1), source_format='free') == 'x = x + 1'


def test_While():
    x = symbols('x')
    # 测试 While 循环的转换
    assert fcode(While(abs(x) > 1, [aug_assign(x, '-', 1)]), source_format='free') == (
        'do while (abs(x) > 1)\n'
        '   x = x - 1\n'
        'end do'
    )


def test_FunctionPrototype_print():
    x = symbols('x')
    # 测试函数原型的打印
    # 创建一个整数类型的符号变量 n
    n = symbols('n', integer=True)
    
    # 创建一个实数类型的符号变量 vx，其名称为 x
    vx = Variable(x, type=real)
    
    # 创建一个整数类型的符号变量 vn，其名称为 n
    vn = Variable(n, type=integer)
    
    # 创建一个函数原型 fp1，其返回类型为实数类型，函数名为 'power'，参数列表为 [vx, vn]
    fp1 = FunctionPrototype(real, 'power', [vx, vn])
    
    # 当多行代码生成功能正常工作时，应修改为适当的测试方式
    # 参见 https://github.com/sympy/sympy/issues/15824
    # 断言调用 fcode(fp1) 会引发 NotImplementedError 异常
    raises(NotImplementedError, lambda: fcode(fp1))
# 定义一个测试函数，用于测试打印功能的函数定义
def test_FunctionDefinition_print():
    # 创建符号变量 x 和 n
    x = symbols('x')
    n = symbols('n', integer=True)
    # 创建类型为实数的变量 vx 和类型为整数的变量 vn
    vx = Variable(x, type=real)
    vn = Variable(n, type=integer)
    # 定义函数体，包含赋值表达式和返回语句
    body = [Assignment(x, x**n), Return(x)]
    # 创建一个实数类型的函数定义，名称为 'power'，参数为 vx 和 vn，函数体为 body
    fd1 = FunctionDefinition(real, 'power', [vx, vn], body)
    # 一旦多行生成工作正常，应更改为适当的测试方式
    # 参见 https://github.com/sympy/sympy/issues/15824
    # 断言抛出 NotImplementedError 异常，调用 fcode(fd1) 时
    raises(NotImplementedError, lambda: fcode(fd1))
```