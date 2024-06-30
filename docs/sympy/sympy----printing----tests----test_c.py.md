# `D:\src\scipysrc\sympy\sympy\printing\tests\test_c.py`

```
# 导入必要的 SymPy 模块和函数
from sympy.core import (
    S, pi, oo, Symbol, symbols, Rational, Integer, Float, Function, Mod, GoldenRatio, EulerGamma, Catalan,
    Lambda, Dummy, nan, Mul, Pow, UnevaluatedExpr
)
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
from sympy.functions import (
    Abs, acos, acosh, asin, asinh, atan, atanh, atan2, ceiling, cos, cosh, erf,
    erfc, exp, floor, gamma, log, loggamma, Max, Min, Piecewise, sign, sin, sinh,
    sqrt, tan, tanh, fibonacci, lucas
)
from sympy.sets import Range
from sympy.logic import ITE, Implies, Equivalent
from sympy.codegen import For, aug_assign, Assignment
from sympy.testing.pytest import raises, XFAIL
from sympy.printing.codeprinter import PrintMethodNotImplementedError
from sympy.printing.c import C89CodePrinter, C99CodePrinter, get_math_macros
from sympy.codegen.ast import (
    AddAugmentedAssignment, Element, Type, FloatType, Declaration, Pointer, Variable, value_const, pointer_const,
    While, Scope, Print, FunctionPrototype, FunctionDefinition, FunctionCall, Return,
    real, float32, float64, float80, float128, intc, Comment, CodeBlock, stderr, QuotedString
)
from sympy.codegen.cfunctions import expm1, log1p, exp2, log2, fma, log10, Cbrt, hypot, Sqrt
from sympy.codegen.cnodes import restrict
from sympy.utilities.lambdify import implemented_function
from sympy.tensor import IndexedBase, Idx
from sympy.matrices import Matrix, MatrixSymbol, SparseMatrix

# 导入 ccode 函数用于生成 C 语言代码
from sympy.printing.codeprinter import ccode

# 定义符号变量 x, y, z
x, y, z = symbols('x,y,z')

# 测试函数1: 测试继承自 Abs 类的 fabs 类的 _ccode 方法
def test_printmethod():
    class fabs(Abs):
        def _ccode(self, printer):
            return "fabs(%s)" % printer._print(self.args[0])

    # 断言 fabs(x) 的 C 语言代码为 "fabs(x)"
    assert ccode(fabs(x)) == "fabs(x)"

# 测试函数2: 测试开平方函数 sqrt 的 C 语言代码生成
def test_ccode_sqrt():
    # 断言 sqrt(x) 的 C 语言代码为 "sqrt(x)"
    assert ccode(sqrt(x)) == "sqrt(x)"
    # 断言 x**0.5 的 C 语言代码为 "sqrt(x)"
    assert ccode(x**0.5) == "sqrt(x)"
    # 再次断言 sqrt(x) 的 C 语言代码为 "sqrt(x)"，以确认稳定性
    assert ccode(sqrt(x)) == "sqrt(x)"

# 测试函数3: 测试指数函数 Pow 的 C 语言代码生成
def test_ccode_Pow():
    # 断言 x**3 的 C 语言代码为 "pow(x, 3)"
    assert ccode(x**3) == "pow(x, 3)"
    # 断言 x**(y**3) 的 C 语言代码为 "pow(x, pow(y, 3))"
    assert ccode(x**(y**3)) == "pow(x, pow(y, 3))"
    # 定义一个自定义函数 g(x) = 2*x
    g = implemented_function('g', Lambda(x, 2*x))
    # 测试一个复杂表达式的 C 语言代码生成
    assert ccode(1/(g(x)*3.5)**(x - y**x)/(x**2 + y)) == \
        "pow(3.5*2*x, -x + pow(y, x))/(pow(x, 2) + y)"
    # 断言 x**-1.0 的 C 语言代码为 '1.0/x'
    assert ccode(x**-1.0) == '1.0/x'
    # 断言 x**(2/3) 的 C 语言代码为 'pow(x, 2.0/3.0)'
    assert ccode(x**Rational(2, 3)) == 'pow(x, 2.0/3.0)'
    # 使用类型别名 float80，断言 x**(2/3) 的 C 语言代码为 'powl(x, 2.0L/3.0L)'
    assert ccode(x**Rational(2, 3), type_aliases={real: float80}) == 'powl(x, 2.0L/3.0L)'
    # 定义一个条件函数列表，根据条件生成不同的 Pow 函数
    _cond_cfunc = [(lambda base, exp: exp.is_integer, "dpowi"),
                   (lambda base, exp: not exp.is_integer, "pow")]
    # 使用条件函数列表生成 x**3 的 C 语言代码，断言为 'dpowi(x, 3)'
    assert ccode(x**3, user_functions={'Pow': _cond_cfunc}) == 'dpowi(x, 3)'
    # 使用条件函数列表生成 x**0.5 的 C 语言代码，断言为 'pow(x, 0.5)'
    assert ccode(x**0.5, user_functions={'Pow': _cond_cfunc}) == 'pow(x, 0.5)'
    # 使用条件函数列表生成 x**(16/5) 的 C 语言代码，断言为 'pow(x, 16.0/5.0)'
    assert ccode(x**Rational(16, 5), user_functions={'Pow': _cond_cfunc}) == 'pow(x, 16.0/5.0)'
    # 定义另一个条件函数列表，根据不同的 base 生成不同的 Pow 函数
    _cond_cfunc2 = [(lambda base, exp: base == 2, lambda base, exp: 'exp2(%s)' % exp),
                    (lambda base, exp: base != 2, 'pow')]
    # 测试 2**x 的 C 语言代码生成，断言为 'exp2(x)'
    assert ccode(2**x, user_functions={'Pow': _cond_cfunc2}) == 'exp2(x)'
    # 测试 x**2 的 C 语言代码生成，断言为 'pow(x, 2)'
    assert ccode(x**2, user_functions={'Pow': _cond_cfunc2}) == 'pow(x, 2)'
    # 对于问题 14160 的断言，验证表达式 ccode(Mul(-2, x, Pow(Mul(y,y,evaluate=False), -1, evaluate=False),
    # evaluate=False)) 的结果是否等于 '-2*x/(y*y)'
    assert ccode(Mul(-2, x, Pow(Mul(y,y,evaluate=False), -1, evaluate=False),
                                    evaluate=False)) == '-2*x/(y*y)'
def test_ccode_Max():
    # Test for gh-11926
    # 断言调用 ccode 函数输出的字符串与预期的字符串相等
    assert ccode(Max(x,x*x),user_functions={"Max":"my_max", "Pow":"my_pow"}) == 'my_max(x, my_pow(x, 2))'


def test_ccode_Min_performance():
    # Shouldn't take more than a few seconds
    # 创建一个 Min 对象，包含从 a[0] 到 a[49] 的符号变量，用于性能测试
    big_min = Min(*symbols('a[0:50]'))
    # 遍历不同的 C 标准版本，生成对应的 C 代码并进行断言
    for curr_standard in ('c89', 'c99', 'c11'):
        output = ccode(big_min, standard=curr_standard)
        # 断言生成的代码中左括号和右括号的数量相等
        assert output.count('(') == output.count(')')


def test_ccode_constants_mathh():
    # 断言数学常数的 C 代码生成是否正确
    assert ccode(exp(1)) == "M_E"
    assert ccode(pi) == "M_PI"
    assert ccode(oo, standard='c89') == "HUGE_VAL"
    assert ccode(-oo, standard='c89') == "-HUGE_VAL"
    assert ccode(oo) == "INFINITY"
    assert ccode(-oo, standard='c99') == "-INFINITY"
    assert ccode(pi, type_aliases={real: float80}) == "M_PIl"


def test_ccode_constants_other():
    # 断言其他数学常数的 C 代码生成是否正确
    assert ccode(2*GoldenRatio) == "const double GoldenRatio = %s;\n2*GoldenRatio" % GoldenRatio.evalf(17)
    assert ccode(
        2*Catalan) == "const double Catalan = %s;\n2*Catalan" % Catalan.evalf(17)
    assert ccode(2*EulerGamma) == "const double EulerGamma = %s;\n2*EulerGamma" % EulerGamma.evalf(17)


def test_ccode_Rational():
    # 断言有理数的 C 代码生成是否正确
    assert ccode(Rational(3, 7)) == "3.0/7.0"
    assert ccode(Rational(3, 7), type_aliases={real: float80}) == "3.0L/7.0L"
    assert ccode(Rational(18, 9)) == "2"
    assert ccode(Rational(3, -7)) == "-3.0/7.0"
    assert ccode(Rational(3, -7), type_aliases={real: float80}) == "-3.0L/7.0L"
    assert ccode(Rational(-3, -7)) == "3.0/7.0"
    assert ccode(Rational(-3, -7), type_aliases={real: float80}) == "3.0L/7.0L"
    assert ccode(x + Rational(3, 7)) == "x + 3.0/7.0"
    assert ccode(x + Rational(3, 7), type_aliases={real: float80}) == "x + 3.0L/7.0L"
    assert ccode(Rational(3, 7)*x) == "(3.0/7.0)*x"
    assert ccode(Rational(3, 7)*x, type_aliases={real: float80}) == "(3.0L/7.0L)*x"


def test_ccode_Integer():
    # 断言整数的 C 代码生成是否正确
    assert ccode(Integer(67)) == "67"
    assert ccode(Integer(-1)) == "-1"


def test_ccode_functions():
    # 断言函数的 C 代码生成是否正确
    assert ccode(sin(x) ** cos(x)) == "pow(sin(x), cos(x))"


def test_ccode_inline_function():
    x = symbols('x')
    # 创建一个内联函数 g(x) = 2*x，断言生成的 C 代码是否正确
    g = implemented_function('g', Lambda(x, 2*x))
    assert ccode(g(x)) == "2*x"
    # 创建一个内联函数 g(x) = 2*x/Catalan，断言生成的 C 代码是否正确
    g = implemented_function('g', Lambda(x, 2*x/Catalan))
    assert ccode(
        g(x)) == "const double Catalan = %s;\n2*x/Catalan" % Catalan.evalf(17)
    # 创建一个 IndexedBase 对象 A 和一个内联函数 g(A[i]) = A[i] * (A[i] + 1) * (A[i] + 2)，断言生成的 C 代码是否正确
    A = IndexedBase('A')
    i = Idx('i', symbols('n', integer=True))
    g = implemented_function('g', Lambda(x, x*(1 + x)*(2 + x)))
    assert ccode(g(A[i]), assign_to=A[i]) == (
        "for (int i=0; i<n; i++){\n"
        "   A[i] = (A[i] + 1)*(A[i] + 2)*A[i];\n"
        "}"
    )


def test_ccode_exceptions():
    # 断言特定函数（gamma 函数）的 C 代码生成是否正确，并处理未实现的 C89 标准异常
    assert ccode(gamma(x), standard='C99') == "tgamma(x)"
    with raises(PrintMethodNotImplementedError):
        ccode(gamma(x), standard='C89')
    with raises(PrintMethodNotImplementedError):
        ccode(gamma(x), standard='C89', allow_unknown_functions=False)

    # 处理 gamma 函数的 C 代码生成，允许未知函数的异常
    ccode(gamma(x), standard='C89', allow_unknown_functions=True)
def test_ccode_functions2():
    # 检查 ccode 函数对 ceiling(x) 的输出是否等于 "ceil(x)"
    assert ccode(ceiling(x)) == "ceil(x)"
    # 检查 ccode 函数对 Abs(x) 的输出是否等于 "fabs(x)"
    assert ccode(Abs(x)) == "fabs(x)"
    # 检查 ccode 函数对 gamma(x) 的输出是否等于 "tgamma(x)"
    assert ccode(gamma(x)) == "tgamma(x)"
    # 声明两个实数符号 r, s
    r, s = symbols('r,s', real=True)
    # 检查 ccode 函数对 Mod(ceiling(r), ceiling(s)) 的输出是否等于 '((ceil(r) % ceil(s)) + ceil(s)) % ceil(s)'
    assert ccode(Mod(ceiling(r), ceiling(s))) == '((ceil(r) % ceil(s)) + ' \
                                                 'ceil(s)) % ceil(s)'
    # 检查 ccode 函数对 Mod(r, s) 的输出是否等于 "fmod(r, s)"
    assert ccode(Mod(r, s)) == "fmod(r, s)"
    # 声明两个整数符号 p1, p2，并检查 ccode 函数对 Mod(p1, p2) 的输出是否等于 'p1 % p2'
    p1, p2 = symbols('p1 p2', integer=True, positive=True)
    assert ccode(Mod(p1, p2)) == 'p1 % p2'
    # 检查 ccode 函数对 Mod(p1, p2 + 3) 的输出是否等于 'p1 % (p2 + 3)'
    assert ccode(Mod(p1, p2 + 3)) == 'p1 % (p2 + 3)'
    # 检查 ccode 函数对 Mod(-3, -7, evaluate=False) 的输出是否等于 '(-3) % (-7)'
    assert ccode(Mod(-3, -7, evaluate=False)) == '(-3) % (-7)'
    # 检查 ccode 函数对 -Mod(3, 7, evaluate=False) 的输出是否等于 '-(3 % 7)'
    assert ccode(-Mod(3, 7, evaluate=False)) == '-(3 % 7)'
    # 检查 ccode 函数对 r*Mod(p1, p2) 的输出是否等于 'r*(p1 % p2)'
    assert ccode(r*Mod(p1, p2)) == 'r*(p1 % p2)'
    # 检查 ccode 函数对 Mod(p1, p2)**s 的输出是否等于 'pow(p1 % p2, s)'
    assert ccode(Mod(p1, p2)**s) == 'pow(p1 % p2, s)'
    # 声明一个整数符号 n，并检查 ccode 函数对 Mod(-n, p2) 的输出是否等于 '(-n) % p2'
    n = symbols('n', integer=True, negative=True)
    assert ccode(Mod(-n, p2)) == '(-n) % p2'
    # 检查 ccode 函数对 fibonacci(n) 的输出是否等于 '((1.0/5.0)*pow(2, -n)*sqrt(5)*(-pow(1 - sqrt(5), n) + pow(1 + sqrt(5), n)))'
    assert ccode(fibonacci(n)) == '((1.0/5.0)*pow(2, -n)*sqrt(5)*(-pow(1 - sqrt(5), n) + pow(1 + sqrt(5), n)))'
    # 检查 ccode 函数对 lucas(n) 的输出是否等于 '(pow(2, -n)*(pow(1 - sqrt(5), n) + pow(1 + sqrt(5), n)))'
    assert ccode(lucas(n)) == '(pow(2, -n)*(pow(1 - sqrt(5), n) + pow(1 + sqrt(5), n)))'


def test_ccode_user_functions():
    # 声明一个实数符号 x 和一个整数符号 n
    x = symbols('x', integer=False)
    n = symbols('n', integer=True)
    # 自定义函数字典，用于 ccode 函数的用户自定义函数
    custom_functions = {
        "ceiling": "ceil",
        "Abs": [(lambda x: not x.is_integer, "fabs"), (lambda x: x.is_integer, "abs")],
    }
    # 检查 ccode 函数对 ceiling(x) 的输出是否等于 "ceil(x)"，使用自定义函数
    assert ccode(ceiling(x), user_functions=custom_functions) == "ceil(x)"
    # 检查 ccode 函数对 Abs(x) 的输出是否等于 "fabs(x)"，使用自定义函数
    assert ccode(Abs(x), user_functions=custom_functions) == "fabs(x)"
    # 检查 ccode 函数对 Abs(n) 的输出是否等于 "abs(n)"，使用自定义函数
    assert ccode(Abs(n), user_functions=custom_functions) == "abs(n)"

    # 声明符号表达式 expr 和函数 muladd
    expr = Symbol('a')
    muladd = Function('muladd')
    # 循环创建符号表达式，用于 ccode 函数的用户自定义函数
    for i in range(0, 100):
        # 大量的项用作 gh-23839 的回归测试
        expr = muladd(Rational(1, 2), Symbol(f'a{i}'), expr)
    # 检查 ccode 函数对 expr 的输出是否包含 'a99'
    out = ccode(expr, user_functions={'muladd':'muladd'})
    assert 'a99' in out
    # 检查 ccode 函数对 expr 的输出中 'muladd' 的出现次数是否为 100
    assert out.count('muladd') == 100


def test_ccode_boolean():
    # 检查 ccode 函数对 True 的输出是否等于 "true"
    assert ccode(True) == "true"
    # 检查 ccode 函数对 S.true 的输出是否等于 "true"
    assert ccode(S.true) == "true"
    # 检查 ccode 函数对 False 的输出是否等于 "false"
    assert ccode(False) == "false"
    # 检查 ccode 函数对 S.false 的输出是否等于 "false"
    assert ccode(S.false) == "false"
    # 检查 ccode 函数对 x & y 的输出是否等于 "x && y"
    assert ccode(x & y) == "x && y"
    # 检查 ccode 函数对 x | y 的输出是否等于 "x || y"
    assert ccode(x | y) == "x || y"
    # 检查 ccode 函数对 ~x 的输出是否等于 "!x"
    assert ccode(~x) == "!x"
    # 检查 ccode 函数对 x & y & z 的输出是否等于 "x && y && z"
    assert ccode(x & y & z) == "x && y && z"
    # 检查 ccode 函数对 x | y | z 的输出是否等于 "x || y || z"
    assert ccode(x | y | z) == "x || y || z"
    # 检查 ccode 函数对 (x & y) | z 的输出是否等于 "z || x && y"
    assert ccode((x & y) | z) == "z || x && y"
    # 检查 ccode 函数对 (x | y) & z 的输出是否等于 "z && (x || y)"
    assert ccode((x | y) & z) == "z && (x || y)"
    # 检查 ccode 函数对 x ^ y 的输出是否等于 '(x || y) && (!x || !y)'
    assert ccode(x ^ y) == '(x || y) && (!x || !y)'
    # 检查 ccode 函数对 (x ^ y) ^ z 的输出是否等于 '(x || y || z) && (x || !y || !z) && (y || !x || !z) && (z || !x || !y)'
    assert ccode((x ^ y) ^ z) == '(x || y || z) && (x || !y || !z) && (y || !x || !z) && (z || !x || !y)'
    # 检查 ccode 函数对 Implies(x, y) 的输出是否等于 'y || !x'
    assert ccode(Implies(x, y)) == 'y || !x'
    # 检查 ccode 函数对 Equivalent(x, z ^ y, Implies(z, x)) 的输出是否等于 '(x || (y
    # 创建一个 Piecewise 对象 expr，它根据条件分段定义表达式
    expr = Piecewise((x, x < 1), (x**2, True))
    # 断言生成的 C 代码与预期值相符
    assert ccode(expr) == (
            "((x < 1) ? (\n"
            "   x\n"
            ")\n"
            ": (\n"
            "   pow(x, 2)\n"
            "))")
    # 断言生成的赋值到变量 c 的 C 代码与预期值相符
    assert ccode(expr, assign_to="c") == (
            "if (x < 1) {\n"
            "   c = x;\n"
            "}\n"
            "else {\n"
            "   c = pow(x, 2);\n"
            "}")
    # 创建一个包含三个分段的 Piecewise 对象 expr
    expr = Piecewise((x, x < 1), (x + 1, x < 2), (x**2, True))
    # 断言生成的 C 代码与预期值相符
    assert ccode(expr) == (
            "((x < 1) ? (\n"
            "   x\n"
            ")\n"
            ": ((x < 2) ? (\n"
            "   x + 1\n"
            ")\n"
            ": (\n"
            "   pow(x, 2)\n"
            ")))")
    # 断言生成的赋值到变量 c 的 C 代码与预期值相符
    assert ccode(expr, assign_to='c') == (
            "if (x < 1) {\n"
            "   c = x;\n"
            "}\n"
            "else if (x < 2) {\n"
            "   c = x + 1;\n"
            "}\n"
            "else {\n"
            "   c = pow(x, 2);\n"
            "}")
    # 检查 Piecewise 对象 expr 中没有默认条件 True 的情况，预期引发 ValueError 异常
    raises(ValueError, lambda: ccode(expr))
# 测试生成 C 代码的函数 sinc 的使用
def test_ccode_sinc():
    # 导入 sympy 库中的 sinc 函数
    from sympy.functions.elementary.trigonometric import sinc
    # 计算 sinc(x) 的表达式
    expr = sinc(x)
    # 断言生成的 C 代码与预期结果相等
    assert ccode(expr) == (
            "(((x != 0) ? (\n"
            "   sin(x)/x\n"
            ")\n"
            ": (\n"
            "   1\n"
            ")))")

# 测试生成 C 代码的函数 Piecewise_deep 的使用
def test_ccode_Piecewise_deep():
    # 生成 Piecewise 函数的 C 代码
    p = ccode(2*Piecewise((x, x < 1), (x + 1, x < 2), (x**2, True)))
    # 断言生成的 C 代码与预期结果相等
    assert p == (
            "2*((x < 1) ? (\n"
            "   x\n"
            ")\n"
            ": ((x < 2) ? (\n"
            "   x + 1\n"
            ")\n"
            ": (\n"
            "   pow(x, 2)\n"
            ")))")
    # 构建复杂表达式并生成其 C 代码，然后进行断言
    expr = x*y*z + x**2 + y**2 + Piecewise((0, x < 0.5), (1, True)) + cos(z) - 1
    assert ccode(expr) == (
            "pow(x, 2) + x*y*z + pow(y, 2) + ((x < 0.5) ? (\n"
            "   0\n"
            ")\n"
            ": (\n"
            "   1\n"
            ")) + cos(z) - 1")
    # 指定生成的 C 代码赋值给变量 c，并进行断言
    assert ccode(expr, assign_to='c') == (
            "c = pow(x, 2) + x*y*z + pow(y, 2) + ((x < 0.5) ? (\n"
            "   0\n"
            ")\n"
            ": (\n"
            "   1\n"
            ")) + cos(z) - 1;")

# 测试生成 C 代码的 ITE 函数的使用
def test_ccode_ITE():
    # 生成 ITE 表达式的 C 代码
    expr = ITE(x < 1, y, z)
    # 断言生成的 C 代码与预期结果相等
    assert ccode(expr) == (
            "((x < 1) ? (\n"
            "   y\n"
            ")\n"
            ": (\n"
            "   z\n"
            "))")

# 测试生成 C 代码时的设置选项
def test_ccode_settings():
    # 断言设置错误的方法时会抛出 TypeError 异常
    raises(TypeError, lambda: ccode(sin(x), method="garbage"))

# 测试生成 IndexedBase 对象的 C 代码
def test_ccode_Indexed():
    # 定义整数符号
    s, n, m, o = symbols('s n m o', integer=True)
    # 定义索引对象
    i, j, k = Idx('i', n), Idx('j', m), Idx('k', o)

    # 创建 IndexedBase 对象和 C99CodePrinter 对象
    x = IndexedBase('x')[j]
    A = IndexedBase('A')[i, j]
    B = IndexedBase('B')[i, j, k]

    p = C99CodePrinter()

    # 断言打印后的 IndexedBase 对象的 C 代码与预期结果相等
    assert p._print_Indexed(x) == 'x[j]'
    assert p._print_Indexed(A) == 'A[%s]' % (m*i+j)
    assert p._print_Indexed(B) == 'B[%s]' % (i*o*m+j*o+k)

    # 改变形状后再次断言打印后的 IndexedBase 对象的 C 代码与预期结果相等
    A = IndexedBase('A', shape=(5,3))[i, j]
    assert p._print_Indexed(A) == 'A[%s]' % (3*i + j)

    # 使用 F 格式的步长并断言生成的 C 代码与预期结果相等
    A = IndexedBase('A', shape=(5,3), strides='F')[i, j]
    assert ccode(A) == 'A[%s]' % (i + 5*j)

    # 使用偏移量和步长并断言生成的 C 代码与预期结果相等
    A = IndexedBase('A', shape=(29,29), strides=(1, s), offset=o)[i, j]
    assert ccode(A) == 'A[o + s*j + i]'

    # 创建带步长和偏移量的 IndexedBase 对象，并断言生成的 C 代码与预期结果相等
    Abase = IndexedBase('A', strides=(s, m, n), offset=o)
    assert ccode(Abase[i, j, k]) == 'A[m*j + n*k + o + s*i]'
    assert ccode(Abase[2, 3, k]) == 'A[3*m + n*k + o + 2*s]'

# 测试生成 Element 对象的 C 代码
def test_Element():
    # 断言生成的 Element 对象的 C 代码与预期结果相等
    assert ccode(Element('x', 'ij')) == 'x[i][j]'
    assert ccode(Element('x', 'ij', strides='kl', offset='o')) == 'x[i*k + j*l + o]'
    assert ccode(Element('x', (3,))) == 'x[3]'
    assert ccode(Element('x', (3,4,5))) == 'x[3][4][5]'

# 测试生成 IndexedBase 对象的 C 代码，不考虑缩并
def test_ccode_Indexed_without_looking_for_contraction():
    # 定义变量和 IndexedBase 对象
    len_y = 5
    y = IndexedBase('y', shape=(len_y,))
    x = IndexedBase('x', shape=(len_y,))
    Dy = IndexedBase('Dy', shape=(len_y-1,))
    i = Idx('i', len_y-1)
    # 定义等式 e，并生成其右侧表达式的 C 代码，不缩并
    e = Eq(Dy[i], (y[i+1]-y[i])/(x[i+1]-x[i]))
    code0 = ccode(e.rhs, assign_to=e.lhs, contract=False)
    # 断言生成的 C 代码与预期结果相等
    assert code0 == 'Dy[i] = (y[%s] - y[i])/(x[%s] - x[i]);' % (i + 1, i + 1)
def test_ccode_loops_matrix_vector():
    n, m = symbols('n m', integer=True)  # 声明符号变量 n 和 m，均为整数
    A = IndexedBase('A')  # 定义 IndexedBase 对象 A
    x = IndexedBase('x')  # 定义 IndexedBase 对象 x
    y = IndexedBase('y')  # 定义 IndexedBase 对象 y
    i = Idx('i', m)  # 定义索引对象 i，范围为 m
    j = Idx('j', n)  # 定义索引对象 j，范围为 n

    s = (
        'for (int i=0; i<m; i++){\n'  # 循环 m 次，i 从 0 到 m-1
        '   y[i] = 0;\n'  # 将 y[i] 初始化为 0
        '}\n'
        'for (int i=0; i<m; i++){\n'  # 再次循环 m 次，i 从 0 到 m-1
        '   for (int j=0; j<n; j++){\n'  # 内部循环 n 次，j 从 0 到 n-1
        '      y[i] = A[%s]*x[j] + y[i];\n' % (i*n + j) +\  # 计算 y[i] 的新值
        '   }\n'
        '}'  # 结束所有循环
    )
    assert ccode(A[i, j]*x[j], assign_to=y[i]) == s  # 断言生成的 C 代码符合预期


def test_dummy_loops():
    i, m = symbols('i m', integer=True, cls=Dummy)  # 声明虚拟符号变量 i 和 m，均为整数
    x = IndexedBase('x')  # 定义 IndexedBase 对象 x
    y = IndexedBase('y')  # 定义 IndexedBase 对象 y
    i = Idx(i, m)  # 定义索引对象 i，范围为 m

    expected = (
        'for (int i_%(icount)i=0; i_%(icount)i<m_%(mcount)i; i_%(icount)i++){\n'  # 根据虚拟索引生成循环
        '   y[i_%(icount)i] = x[i_%(icount)i];\n'  # 将 x[i] 赋值给 y[i]
        '}'
    ) % {'icount': i.label.dummy_index, 'mcount': m.dummy_index}

    assert ccode(x[i], assign_to=y[i]) == expected  # 断言生成的 C 代码符合预期


def test_ccode_loops_add():
    n, m = symbols('n m', integer=True)  # 声明符号变量 n 和 m，均为整数
    A = IndexedBase('A')  # 定义 IndexedBase 对象 A
    x = IndexedBase('x')  # 定义 IndexedBase 对象 x
    y = IndexedBase('y')  # 定义 IndexedBase 对象 y
    z = IndexedBase('z')  # 定义 IndexedBase 对象 z
    i = Idx('i', m)  # 定义索引对象 i，范围为 m
    j = Idx('j', n)  # 定义索引对象 j，范围为 n

    s = (
        'for (int i=0; i<m; i++){\n'  # 循环 m 次，i 从 0 到 m-1
        '   y[i] = x[i] + z[i];\n'  # 计算 y[i] 的新值
        '}\n'
        'for (int i=0; i<m; i++){\n'  # 再次循环 m 次，i 从 0 到 m-1
        '   for (int j=0; j<n; j++){\n'  # 内部循环 n 次，j 从 0 到 n-1
        '      y[i] = A[%s]*x[j] + y[i];\n' % (i*n + j) +\  # 计算 y[i] 的新值
        '   }\n'
        '}'  # 结束所有循环
    )
    assert ccode(A[i, j]*x[j] + x[i] + z[i], assign_to=y[i]) == s  # 断言生成的 C 代码符合预期


def test_ccode_loops_multiple_contractions():
    n, m, o, p = symbols('n m o p', integer=True)  # 声明符号变量 n, m, o, p，均为整数
    a = IndexedBase('a')  # 定义 IndexedBase 对象 a
    b = IndexedBase('b')  # 定义 IndexedBase 对象 b
    y = IndexedBase('y')  # 定义 IndexedBase 对象 y
    i = Idx('i', m)  # 定义索引对象 i，范围为 m
    j = Idx('j', n)  # 定义索引对象 j，范围为 n
    k = Idx('k', o)  # 定义索引对象 k，范围为 o
    l = Idx('l', p)  # 定义索引对象 l，范围为 p

    s = (
        'for (int i=0; i<m; i++){\n'  # 循环 m 次，i 从 0 到 m-1
        '   y[i] = 0;\n'  # 将 y[i] 初始化为 0
        '}\n'
        'for (int i=0; i<m; i++){\n'  # 再次循环 m 次，i 从 0 到 m-1
        '   for (int j=0; j<n; j++){\n'  # 内部循环 n 次，j 从 0 到 n-1
        '      for (int k=0; k<o; k++){\n'  # 再次循环 o 次，k 从 0 到 o-1
        '         for (int l=0; l<p; l++){\n'  # 最内部循环 p 次，l 从 0 到 p-1
        '            y[i] = a[%s]*b[%s] + y[i];\n' % (i*n*o*p + j*o*p + k*p + l, j*o*p + k*p + l) +\  # 计算 y[i] 的新值
        '         }\n'
        '      }\n'
        '   }\n'
        '}'  # 结束所有循环
    )
    assert ccode(b[j, k, l]*a[i, j, k, l], assign_to=y[i]) == s  # 断言生成的 C 代码符合预期


def test_ccode_loops_addfactor():
    n, m, o, p = symbols('n m o p', integer=True)  # 声明符号变量 n, m, o, p，均为整数
    a = IndexedBase('a')  # 定义 IndexedBase 对象 a
    b = IndexedBase('b')  # 定义 IndexedBase 对象 b
    c = IndexedBase('c')  # 定义 IndexedBase 对象 c
    y = IndexedBase('y')  # 定义 IndexedBase 对象 y
    i = Idx('i', m)  # 定义索引对象 i，范围为 m
    j = Idx('j', n)  # 定义索引对象 j，范围为 n
    k = Idx('k', o)  # 定义索引对象 k，范围为 o
    l = Idx('l', p)  # 定义索引对象 l，范围为 p

    s = (
        'for (int i=0; i<m; i++){\n'  # 循环 m 次，i 从 0 到 m-1
        '   y[i] = 0;\n'  # 将 y[i] 初始化为 0
        '}\n'
        'for (int i=0; i<m; i++){\n'  # 再次循环 m 次，i 从 0 到 m-1
        '   for (
    # 对表达式进行断言检查，验证计算结果是否符合预期
    assert ccode((a[i, j, k, l] + b[i, j, k, l])*c[j, k, l], assign_to=y[i]) == s
# 测试用例：测试多项式 C 语言代码生成的多重循环
def test_ccode_loops_multiple_terms():
    # 定义整数符号变量
    n, m, o, p = symbols('n m o p', integer=True)
    # 创建 IndexedBase 对象表示数组
    a = IndexedBase('a')
    b = IndexedBase('b')
    c = IndexedBase('c')
    y = IndexedBase('y')
    # 创建索引对象
    i = Idx('i', m)
    j = Idx('j', n)
    k = Idx('k', o)

    # 第一个代码片段 s0
    s0 = (
        'for (int i=0; i<m; i++){\n'
        '   y[i] = 0;\n'
        '}\n'
    )

    # 第二个代码片段 s1
    s1 = (
        'for (int i=0; i<m; i++){\n'
        '   for (int j=0; j<n; j++){\n'
        '      for (int k=0; k<o; k++){\n'
        '         y[i] = b[j]*b[k]*c[%s] + y[i];\n' % (i*n*o + j*o + k) +
        '      }\n'
        '   }\n'
        '}\n'
    )

    # 第三个代码片段 s2
    s2 = (
        'for (int i=0; i<m; i++){\n'
        '   for (int k=0; k<o; k++){\n'
        '      y[i] = a[%s]*b[k] + y[i];\n' % (i*o + k) +
        '   }\n'
        '}\n'
    )

    # 第四个代码片段 s3
    s3 = (
        'for (int i=0; i<m; i++){\n'
        '   for (int j=0; j<n; j++){\n'
        '      y[i] = a[%s]*b[j] + y[i];\n' % (i*n + j) +
        '   }\n'
        '}\n'
    )

    # 使用 SymPy 的 ccode 函数生成 C 代码表达式
    c = ccode(b[j]*a[i, j] + b[k]*a[i, k] + b[j]*b[k]*c[i, j, k], assign_to=y[i])

    # 断言生成的 C 代码字符串符合其中一种预期顺序
    assert (c == s0 + s1 + s2 + s3[:-1] or
            c == s0 + s1 + s3 + s2[:-1] or
            c == s0 + s2 + s1 + s3[:-1] or
            c == s0 + s2 + s3 + s1[:-1] or
            c == s0 + s3 + s1 + s2[:-1] or
            c == s0 + s3 + s2 + s1[:-1])


# 测试用例：测试解引用打印
def test_dereference_printing():
    # 创建数学表达式
    expr = x + y + sin(z) + z
    # 断言生成的 C 代码符合预期的解引用格式
    assert ccode(expr, dereference=[z]) == "x + y + (*z) + sin((*z))"


# 测试用例：测试矩阵打印
def test_Matrix_printing():
    # 测试返回 Matrix 对象的 C 代码生成
    mat = Matrix([x*y, Piecewise((2 + x, y>0), (y, True)), sin(z)])
    A = MatrixSymbol('A', 3, 1)
    # 断言生成的 C 代码符合预期的矩阵赋值格式
    assert ccode(mat, A) == (
        "A[0] = x*y;\n"
        "if (y > 0) {\n"
        "   A[1] = x + 2;\n"
        "}\n"
        "else {\n"
        "   A[1] = y;\n"
        "}\n"
        "A[2] = sin(z);")

    # 测试在表达式中使用 MatrixElement 对象的 C 代码生成
    expr = Piecewise((2*A[2, 0], x > 0), (A[2, 0], True)) + sin(A[1, 0]) + A[0, 0]
    # 断言生成的 C 代码符合预期的矩阵元素引用格式
    assert ccode(expr) == (
        "((x > 0) ? (\n"
        "   2*A[2]\n"
        ")\n"
        ": (\n"
        "   A[2]\n"
        ")) + sin(A[1]) + A[0]")

    # 测试在矩阵中使用 MatrixElement 对象的 C 代码生成
    q = MatrixSymbol('q', 5, 1)
    M = MatrixSymbol('M', 3, 3)
    m = Matrix([[sin(q[1,0]), 0, cos(q[2,0])],
        [q[1,0] + q[2,0], q[3, 0], 5],
        [2*q[4, 0]/q[1,0], sqrt(q[0,0]) + 4, 0]])
    # 断言生成的 C 代码符合预期的矩阵元素引用格式
    assert ccode(m, M) == (
        "M[0] = sin(q[1]);\n"
        "M[1] = 0;\n"
        "M[2] = cos(q[2]);\n"
        "M[3] = q[1] + q[2];\n"
        "M[4] = q[3];\n"
        "M[5] = 5;\n"
        "M[6] = 2*q[4]/q[1];\n"
        "M[7] = sqrt(q[0]) + 4;\n"
        "M[8] = 0;")


# 测试用例：测试稀疏矩阵
def test_sparse_matrix():
    # 测试稀疏矩阵生成代码的异常处理
    with raises(PrintMethodNotImplementedError):
        ccode(SparseMatrix([[1, 2, 3]]))

    # 断言 C89CodePrinter 生成的 C 代码包含特定的错误信息
    assert 'Not supported in C' in C89CodePrinter({'strict': False}).doprint(SparseMatrix([[1, 2, 3]]))


# 测试用例：测试保留字用法
def test_ccode_reserved_words():
    # 测试保留字用作符号名称的情况
    x, y = symbols('x, if')
    # 使用 pytest 中的 raises 函数检查是否会抛出 ValueError 异常
    with raises(ValueError):
        # 调用 ccode 函数，传入参数 y**2，同时设置 error_on_reserved=True 和 standard='C99'
        ccode(y**2, error_on_reserved=True, standard='C99')

    # 使用 assert 断言，验证 ccode 函数对 y**2 的输出是否为 'pow(if_, 2)'
    assert ccode(y**2) == 'pow(if_, 2)'

    # 使用 assert 断言，验证 ccode 函数对 x * y**2 的输出是否为 'pow((*if_), 2)*x'，
    # 同时传入 dereference=[y] 参数用于解除 y 的引用
    assert ccode(x * y**2, dereference=[y]) == 'pow((*if_), 2)*x'

    # 使用 assert 断言，验证 ccode 函数对 y**2 的输出是否为 'pow(if_unreserved, 2)'，
    # 同时传入 reserved_word_suffix='_unreserved' 参数用于更改保留字后缀
    assert ccode(y**2, reserved_word_suffix='_unreserved') == 'pow(if_unreserved, 2)'
# 定义测试函数，用于测试生成 C 语言风格代码的函数 ccode
def test_ccode_sign():
    # 定义表达式和参考值，表达式包括函数 sign 的调用和相应的参考 C 语言代码字符串
    expr1, ref1 = sign(x) * y, 'y*(((x) > 0) - ((x) < 0))'
    expr2, ref2 = sign(cos(x)), '(((cos(x)) > 0) - ((cos(x)) < 0))'
    expr3, ref3 = sign(2 * x + x**2) * x + x**2, 'pow(x, 2) + x*(((pow(x, 2) + 2*x) > 0) - ((pow(x, 2) + 2*x) < 0))'
    # 断言生成的 C 语言代码与预期参考代码相等
    assert ccode(expr1) == ref1
    # 断言生成的 C 语言代码与预期参考代码赋值给变量 z 相等
    assert ccode(expr1, 'z') == 'z = %s;' % ref1
    assert ccode(expr2) == ref2
    assert ccode(expr3) == ref3

# 定义测试函数，用于测试赋值表达式生成的 C 语言风格代码
def test_ccode_Assignment():
    assert ccode(Assignment(x, y + z)) == 'x = y + z;'
    assert ccode(aug_assign(x, '+', y + z)) == 'x += y + z;'

# 定义测试函数，用于测试 for 循环生成的 C 语言风格代码
def test_ccode_For():
    # 创建 For 对象 f，包含初始化 x、范围 0 到 10（步长为 2）以及循环体内的乘法赋值操作
    f = For(x, Range(0, 10, 2), [aug_assign(y, '*', x)])
    # 断言生成的 C 语言代码与预期代码块相等
    assert ccode(f) == ("for (x = 0; x < 10; x += 2) {\n"
                        "   y *= x;\n"
                        "}")

# 定义测试函数，用于测试最大值和最小值函数生成的 C 语言风格代码
def test_ccode_Max_Min():
    # 断言生成的 C 语言代码与预期代码（C89 标准）相等
    assert ccode(Max(x, 0), standard='C89') == '((0 > x) ? 0 : x)'
    # 断言生成的 C 语言代码与预期代码（C99 标准）相等
    assert ccode(Max(x, 0), standard='C99') == 'fmax(0, x)'
    # 断言生成的 C 语言代码与预期代码（C89 标准）相等
    assert ccode(Min(x, 0, sqrt(x)), standard='c89') == (
        '((0 < ((x < sqrt(x)) ? x : sqrt(x))) ? 0 : ((x < sqrt(x)) ? x : sqrt(x)))'
    )

# 定义测试函数，用于测试特定 C 语言标准下数学函数生成的 C 语言风格代码
def test_ccode_standard():
    assert ccode(expm1(x), standard='c99') == 'expm1(x)'
    assert ccode(nan, standard='c99') == 'NAN'
    assert ccode(float('nan'), standard='c99') == 'NAN'

# 定义测试函数，用于测试 C89CodePrinter 类
def test_C89CodePrinter():
    # 创建 C89CodePrinter 对象 c89printer
    c89printer = C89CodePrinter()
    # 断言 c89printer 的语言属性为 'C'
    assert c89printer.language == 'C'
    # 断言 c89printer 的标准属性为 'C89'
    assert c89printer.standard == 'C89'
    # 断言保留字 'void' 在 c89printer 的保留字列表中
    assert 'void' in c89printer.reserved_words
    # 断言保留字 'template' 不在 c89printer 的保留字列表中
    assert 'template' not in c89printer.reserved_words

# 定义测试函数，用于测试 C99CodePrinter 类
def test_C99CodePrinter():
    # 使用 C99CodePrinter 类生成 C 语言风格代码并断言与预期结果相等
    assert C99CodePrinter().doprint(expm1(x)) == 'expm1(x)'
    assert C99CodePrinter().doprint(log1p(x)) == 'log1p(x)'
    assert C99CodePrinter().doprint(exp2(x)) == 'exp2(x)'
    assert C99CodePrinter().doprint(log2(x)) == 'log2(x)'
    assert C99CodePrinter().doprint(fma(x, y, -z)) == 'fma(x, y, -z)'
    assert C99CodePrinter().doprint(log10(x)) == 'log10(x)'
    assert C99CodePrinter().doprint(Cbrt(x)) == 'cbrt(x)'  # 注意 Cbrt 因为 cbrt 已被使用。
    assert C99CodePrinter().doprint(hypot(x, y)) == 'hypot(x, y)'
    assert C99CodePrinter().doprint(loggamma(x)) == 'lgamma(x)'
    assert C99CodePrinter().doprint(Max(x, 3, x**2)) == 'fmax(3, fmax(x, pow(x, 2)))'
    assert C99CodePrinter().doprint(Min(x, 3)) == 'fmin(3, x)'
    # 创建 C99CodePrinter 对象 c99printer
    c99printer = C99CodePrinter()
    # 断言 c99printer 的语言属性为 'C'
    assert c99printer.language == 'C'
    # 断言 c99printer 的标准属性为 'C99'
    assert c99printer.standard == 'C99'
    # 断言保留字 'restrict' 在 c99printer 的保留字列表中
    assert 'restrict' in c99printer.reserved_words
    # 断言保留字 'using' 不在 c99printer 的保留字列表中
    assert 'using' not in c99printer.reserved_words

# 定义带有 @XFAIL 装饰器的测试函数，用于测试特定精度下数学函数生成的 C 语言风格代码
@XFAIL
def test_C99CodePrinter__precision_f80():
    # 创建 C99CodePrinter 对象 f80_printer，使用特定的类型别名设置
    f80_printer = C99CodePrinter({"type_aliases": {real: float80}})
    # 断言生成的 C 语言代码与预期结果相等
    assert f80_printer.doprint(sin(x + Float('2.1'))) == 'sinl(x + 2.1L)'

# 定义测试函数，用于测试不同精度下数学函数生成的 C 语言风格代码
def test_C99CodePrinter__precision():
    # 创建整数符号变量 n 和 p
    n = symbols('n', integer=True)
    p = symbols('p', integer=True, positive=True)
    # 创建 C99CodePrinter 对象 f32_printer 和 f64_printer，分别使用特定的类型别名设置
    f32_printer = C99CodePrinter({"type_aliases": {real: float32}})
    f64_printer = C99CodePrinter({"type_aliases": {real: float64}})
    # 创建一个用于打印 C99 代码的打印机对象，其中指定了实数类型的别名为 float80
    f80_printer = C99CodePrinter({"type_aliases": {real: float80}})
    
    # 使用 f32_printer 打印 sin(x+2.1) 的结果，并进行断言检查
    assert f32_printer.doprint(sin(x+2.1)) == 'sinf(x + 2.1F)'
    
    # 使用 f64_printer 打印 sin(x+2.1) 的结果，并进行断言检查
    assert f64_printer.doprint(sin(x+2.1)) == 'sin(x + 2.1000000000000001)'
    
    # 使用 f80_printer 打印 sin(x+Float('2.0')) 的结果，并进行断言检查
    assert f80_printer.doprint(sin(x+Float('2.0'))) == 'sinl(x + 2.0L)'

    # 遍历 [f32_printer, f64_printer, f80_printer] 这三个打印机对象，对每个打印机对象执行一系列检查
    for printer, suffix in zip([f32_printer, f64_printer, f80_printer], ['f', '', 'l']):
        # 定义一个用于检查打印结果的函数 check
        def check(expr, ref):
            # 断言打印出的结果与预期的结果 ref 格式化后相等
            assert printer.doprint(expr) == ref.format(s=suffix, S=suffix.upper())
        
        # 检查 Abs(n) 的打印结果
        check(Abs(n), 'abs(n)')
        
        # 检查 Abs(x + 2.0) 的打印结果
        check(Abs(x + 2.0), 'fabs{s}(x + 2.0{S})')
        
        # 检查 sin(x + 4.0)**cos(x - 2.0) 的打印结果
        check(sin(x + 4.0)**cos(x - 2.0), 'pow{s}(sin{s}(x + 4.0{S}), cos{s}(x - 2.0{S}))')
        
        # 检查 exp(x*8.0) 的打印结果
        check(exp(x*8.0), 'exp{s}(8.0{S}*x)')
        
        # 检查 exp2(x) 的打印结果
        check(exp2(x), 'exp2{s}(x)')
        
        # 检查 expm1(x*4.0) 的打印结果
        check(expm1(x*4.0), 'expm1{s}(4.0{S}*x)')
        
        # 检查 Mod(p, 2) 的打印结果
        check(Mod(p, 2), 'p % 2')
        
        # 检查 Mod(2*p + 3, 3*p + 5, evaluate=False) 的打印结果
        check(Mod(2*p + 3, 3*p + 5, evaluate=False), '(2*p + 3) % (3*p + 5)')
        
        # 检查 Mod(x + 2.0, 3.0) 的打印结果
        check(Mod(x + 2.0, 3.0), 'fmod{s}(1.0{S}*x + 2.0{S}, 3.0{S})')
        
        # 检查 Mod(x, 2.0*x + 3.0) 的打印结果
        check(Mod(x, 2.0*x + 3.0), 'fmod{s}(1.0{S}*x, 2.0{S}*x + 3.0{S})')
        
        # 检查 log(x/2) 的打印结果
        check(log(x/2), 'log{s}((1.0{S}/2.0{S})*x)')
        
        # 检查 log10(3*x/2) 的打印结果
        check(log10(3*x/2), 'log10{s}((3.0{S}/2.0{S})*x)')
        
        # 检查 log2(x*8.0) 的打印结果
        check(log2(x*8.0), 'log2{s}(8.0{S}*x)')
        
        # 检查 log1p(x) 的打印结果
        check(log1p(x), 'log1p{s}(x)')
        
        # 检查 2**x 的打印结果
        check(2**x, 'pow{s}(2, x)')
        
        # 检查 2.0**x 的打印结果
        check(2.0**x, 'pow{s}(2.0{S}, x)')
        
        # 检查 x**3 的打印结果
        check(x**3, 'pow{s}(x, 3)')
        
        # 检查 x**4.0 的打印结果
        check(x**4.0, 'pow{s}(x, 4.0{S})')
        
        # 检查 sqrt(3+x) 的打印结果
        check(sqrt(3+x), 'sqrt{s}(x + 3)')
        
        # 检查 Cbrt(x-2.0) 的打印结果
        check(Cbrt(x-2.0), 'cbrt{s}(x - 2.0{S})')
        
        # 检查 hypot(x, y) 的打印结果
        check(hypot(x, y), 'hypot{s}(x, y)')
        
        # 检查 sin(3.*x + 2.) 的打印结果
        check(sin(3.*x + 2.), 'sin{s}(3.0{S}*x + 2.0{S})')
        
        # 检查 cos(3.*x - 1.) 的打印结果
        check(cos(3.*x - 1.), 'cos{s}(3.0{S}*x - 1.0{S})')
        
        # 检查 tan(4.*y + 2.) 的打印结果
        check(tan(4.*y + 2.), 'tan{s}(4.0{S}*y + 2.0{S})')
        
        # 检查 asin(3.*x + 2.) 的打印结果
        check(asin(3.*x + 2.), 'asin{s}(3.0{S}*x + 2.0{S})')
        
        # 检查 acos(3.*x + 2.) 的打印结果
        check(acos(3.*x + 2.), 'acos{s}(3.0{S}*x + 2.0{S})')
        
        # 检查 atan(3.*x + 2.) 的打印结果
        check(atan(3.*x + 2.), 'atan{s}(3.0{S}*x + 2.0{S})')
        
        # 检查 atan2(3.*x, 2.*y) 的打印结果
        check(atan2(3.*x, 2.*y), 'atan2{s}(3.0{S}*x, 2.0{S}*y)')
        
        # 检查 sinh(3.*x + 2.) 的打印结果
        check(sinh(3.*x + 2.), 'sinh{s}(3.0{S}*x + 2.0{S})')
        
        # 检查 cosh(3.*x - 1.) 的打印结果
        check(cosh(3.*x - 1.), 'cosh{s}(3.0{S}*x - 1.0{S})')
        
        # 检查 tanh(4.0*y + 2.) 的打印结果
        check(tanh(4.0*y + 2.), 'tanh{s}(4.0{S}*y + 2.0{S})')
        
        # 检查 asinh(3.*x + 2.) 的打印结果
        check(asinh(3.*x + 2.), 'asinh{s}(3.0{S}*x + 2.0{S})')
        
        # 检查 acosh(3.*x + 2.) 的打印结果
        check(acosh(3.*x + 2.), 'acosh{s}(3.0{S}*x + 2.0{S})')
        
        # 检查 atanh(3.*x + 2.) 的打印结果
        check(atanh(3.*x + 2.), 'atanh{s}(3.0{S}*x + 2.0{S})')
        
        # 检查 erf(42.*x) 的
# 定义用于测试数学宏的函数
def test_get_math_macros():
    # 获取数学宏的字典
    macros = get_math_macros()
    # 断言表达式exp(1)的数学宏为'M_E'
    assert macros[exp(1)] == 'M_E'
    # 断言表达式1/Sqrt(2)的数学宏为'M_SQRT1_2'
    assert macros[1/Sqrt(2)] == 'M_SQRT1_2'


# 测试声明语句生成函数Declaration的各种情况
def test_ccode_Declaration():
    # 创建整数符号'i'
    i = symbols('i', integer=True)
    # 创建整数变量var1，类型为Type.from_expr(i)
    var1 = Variable(i, type=Type.from_expr(i))
    # 创建var1的声明语句dcl1，并断言其生成的C代码为'int i'
    dcl1 = Declaration(var1)
    assert ccode(dcl1) == 'int i'

    # 创建浮点数变量x，类型为float32，带有属性value_const
    var2 = Variable(x, type=float32, attrs={value_const})
    # 创建var2的声明语句dcl2a，并断言其生成的C代码为'const float x'
    dcl2a = Declaration(var2)
    assert ccode(dcl2a) == 'const float x'
    # 将var2作为带有pi值的声明语句dcl2b，并断言其生成的C代码为'const float x = M_PI'
    dcl2b = var2.as_Declaration(value=pi)
    assert ccode(dcl2b) == 'const float x = M_PI'

    # 创建布尔变量y，类型为Type('bool')
    var3 = Variable(y, type=Type('bool'))
    # 创建var3的声明语句dcl3
    dcl3 = Declaration(var3)
    # 创建C89CodePrinter对象printer
    printer = C89CodePrinter()
    # 断言printer的headers属性中不包含'stdbool.h'
    assert 'stdbool.h' not in printer.headers
    # 断言printer打印dcl3的结果为'bool y'
    assert printer.doprint(dcl3) == 'bool y'
    # 断言printer的headers属性中包含'stdbool.h'
    assert 'stdbool.h' in printer.headers

    # 创建实数符号'u'
    u = symbols('u', real=True)
    # 推导指针ptr4，指向u，带有属性pointer_const和restrict
    ptr4 = Pointer.deduced(u, attrs={pointer_const, restrict})
    # 创建ptr4的声明语句dcl4，并断言其生成的C代码为'double * const restrict u'
    dcl4 = Declaration(ptr4)
    assert ccode(dcl4) == 'double * const restrict u'

    # 创建浮点数变量x，类型为'__float128'，带有属性value_const
    var5 = Variable(x, Type('__float128'), attrs={value_const})
    # 创建var5的声明语句dcl5a，并断言其生成的C代码为'const __float128 x'
    dcl5a = Declaration(var5)
    assert ccode(dcl5a) == 'const __float128 x'
    # 使用pi值创建一个新的变量var5b，并创建其声明语句dcl5b，断言其生成的C代码为'const __float128 x = M_PI'
    var5b = Variable(var5.symbol, var5.type, pi, attrs=var5.attrs)
    dcl5b = Declaration(var5b)
    assert ccode(dcl5b) == 'const __float128 x = M_PI'


# 测试C99CodePrinter定制类型的情况
def test_C99CodePrinter_custom_type():
    # 创建_Float128类型f128，设置其位数、尾数和指数
    f128 = FloatType('_Float128', float128.nbits, float128.nmant, float128.nexp)
    # 创建C99CodePrinter对象p128，使用f128作为类型别名，并设置相关后缀和宏
    p128 = C99CodePrinter({
        "type_aliases": {real: f128},
        "type_literal_suffixes": {f128: 'Q'},
        "type_func_suffixes": {f128: 'f128'},
        "type_math_macro_suffixes": {
            real: 'f128',
            f128: 'f128'
        },
        "type_macros": {
            f128: ('__STDC_WANT_IEC_60559_TYPES_EXT__',)
        }
    })
    # 断言p128打印符号x的结果为'x'
    assert p128.doprint(x) == 'x'
    # 断言p128的headers、libraries、macros属性均为空
    assert not p128.headers
    assert not p128.libraries
    assert not p128.macros
    # 断言p128打印2.0的结果为'2.0Q'，并且headers、libraries属性仍为空，macros包含'__STDC_WANT_IEC_60559_TYPES_EXT__'
    assert p128.doprint(2.0) == '2.0Q'
    assert not p128.headers
    assert not p128.libraries
    assert p128.macros == {'__STDC_WANT_IEC_60559_TYPES_EXT__'}

    # 断言p128打印Rational(1, 2)的结果为'1.0Q/2.0Q'
    assert p128.doprint(Rational(1, 2)) == '1.0Q/2.0Q'
    # 断言p128打印sin(x)的结果为'sinf128(x)'
    assert p128.doprint(sin(x)) == 'sinf128(x)'
    # 断言p128打印cos(2., evaluate=False)的结果为'cosf128(2.0Q)'
    assert p128.doprint(cos(2., evaluate=False)) == 'cosf128(2.0Q)'
    # 断言p128打印x**-1.0的结果为'1.0Q/x'
    assert p128.doprint(x**-1.0) == '1.0Q/x'

    # 创建浮点数变量x，类型为f128，带有属性value_const
    var5 = Variable(x, f128, attrs={value_const})
    # 创建var5的声明语句dcl5a，并断言其生成的C代码为'const _Float128 x'
    dcl5a = Declaration(var5)
    assert ccode(dcl5a) == 'const _Float128 x'
    # 使用pi值创建一个新的变量var5b，并创建其声明语句dcl5b，断言其生成的C代码为'const _Float128 x = M_PIf128'
    var5b = Variable(x, f128, pi, attrs={value_const})
    dcl5b = Declaration(var5b)
    assert p128.doprint(dcl5b) == 'const _Float128 x = M_PIf128'
    # 使用Catalan.evalf(38)值创建一个新的变量var5b，并创建其声明语句dcl5c，断言其生成的C代码为'const _Float128 x = %sQ' % Catalan.evalf(f128.decimal_dig)
    var5b = Variable(x, f128, value=Catalan.evalf(38), attrs={value_const})
    dcl5c = Declaration(var5b)
    assert p128.doprint(dcl5c) == 'const _Float128 x = %sQ' % Catalan.evalf(f128.decimal_dig)


# 测试MatrixElement的打印情况
def test_MatrixElement_printing():
    # 创建MatrixSymbol A、B、C，分别为(1, 3)维度
    A = MatrixSymbol("A", 1, 3)
    B = MatrixSymbol("B", 1, 3)
    C = MatrixSymbol("C", 1, 3)

    # 断言打印A[0, 0]的结果为"A[0]"
    assert(ccode(A[0, 0]) == "A[0]")
    # 断言：验证条件ccode(3 * A[0, 0]) == "3*A[0]"是否为真
    assert(ccode(3 * A[0, 0]) == "3*A[0]")
    
    # 将符号表达式C[0, 0]中的符号C替换为(A - B)，得到新的符号表达式F
    F = C[0, 0].subs(C, A - B)
    # 断言：验证条件ccode(F) == "(A - B)[0]"是否为真
    assert(ccode(F) == "(A - B)[0]")
def test_ccode_math_macros():
    # 检查生成的 C 代码是否正确表达了给定的数学表达式
    assert ccode(z + exp(1)) == 'z + M_E'
    assert ccode(z + log2(exp(1))) == 'z + M_LOG2E'
    assert ccode(z + 1/log(2)) == 'z + M_LOG2E'
    assert ccode(z + log(2)) == 'z + M_LN2'
    assert ccode(z + log(10)) == 'z + M_LN10'
    assert ccode(z + pi) == 'z + M_PI'
    assert ccode(z + pi/2) == 'z + M_PI_2'
    assert ccode(z + pi/4) == 'z + M_PI_4'
    assert ccode(z + 1/pi) == 'z + M_1_PI'
    assert ccode(z + 2/pi) == 'z + M_2_PI'
    assert ccode(z + 2/sqrt(pi)) == 'z + M_2_SQRTPI'
    assert ccode(z + 2/Sqrt(pi)) == 'z + M_2_SQRTPI'
    assert ccode(z + sqrt(2)) == 'z + M_SQRT2'
    assert ccode(z + Sqrt(2)) == 'z + M_SQRT2'
    assert ccode(z + 1/sqrt(2)) == 'z + M_SQRT1_2'
    assert ccode(z + 1/Sqrt(2)) == 'z + M_SQRT1_2'


def test_ccode_Type():
    # 检查是否正确生成了给定类型的 C 代码表示
    assert ccode(Type('float')) == 'float'
    assert ccode(intc) == 'int'


def test_ccode_codegen_ast():
    # 检查 Abstract Syntax Tree (AST) 节点是否能正确转换为 C 代码
    # 注意：C 只允许 /* ... */ 形式的注释，不允许使用 // 形式的注释
    assert ccode(Comment("this is a comment")) == "/* this is a comment */"  # not //
    assert ccode(While(abs(x) > 1, [aug_assign(x, '-', 1)])) == (
        'while (fabs(x) > 1) {\n'
        '   x -= 1;\n'
        '}'
    )
    assert ccode(Scope([AddAugmentedAssignment(x, 1)])) == (
        '{\n'
        '   x += 1;\n'
        '}'
    )
    inp_x = Declaration(Variable(x, type=real))
    assert ccode(FunctionPrototype(real, 'pwer', [inp_x])) == 'double pwer(double x)'
    assert ccode(FunctionDefinition(real, 'pwer', [inp_x], [Assignment(x, x**2)])) == (
        'double pwer(double x){\n'
        '   x = pow(x, 2);\n'
        '}'
    )

    # CodeBlock 的元素被格式化为语句：
    block = CodeBlock(
        x,
        Print([x, y], "%d %d"),
        Print([QuotedString('hello'), y], "%s %d", file=stderr),
        FunctionCall('pwer', [x]),
        Return(x),
    )
    assert ccode(block) == '\n'.join([
        'x;',
        'printf("%d %d", x, y);',
        'fprintf(stderr, "%s %d", "hello", y);',
        'pwer(x);',
        'return x;',
    ])


def test_ccode_UnevaluatedExpr():
    # 检查未评估的表达式是否正确转换为 C 代码
    assert ccode(UnevaluatedExpr(y * x) + z) == "z + x*y"
    assert ccode(UnevaluatedExpr(y + x) + z) == "z + (x + y)"  # gh-21955
    w = symbols('w')
    assert ccode(UnevaluatedExpr(y + x) + UnevaluatedExpr(z + w)) == "(w + z) + (x + y)"

    p, q, r = symbols("p q r", real=True)
    q_r = UnevaluatedExpr(q + r)
    expr = abs(exp(p+q_r))
    assert ccode(expr) == "exp(p + (q + r))"


def test_ccode_array_like_containers():
    # 检查类数组的容器是否正确转换为 C 代码
    assert ccode([2,3,4]) == "{2, 3, 4}"
    assert ccode((2,3,4)) == "{2, 3, 4}"
```