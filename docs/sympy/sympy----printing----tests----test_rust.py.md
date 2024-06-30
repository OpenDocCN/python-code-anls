# `D:\src\scipysrc\sympy\sympy\printing\tests\test_rust.py`

```
# 导入所需的符号和函数模块
from sympy.core import (S, pi, oo, symbols, Rational, Integer,
                        GoldenRatio, EulerGamma, Catalan, Lambda, Dummy,
                        Eq, Ne, Le, Lt, Gt, Ge, Mod)
from sympy.functions import (Piecewise, sin, cos, Abs, exp, ceiling, sqrt,
                             sign, floor)
from sympy.logic import ITE
from sympy.testing.pytest import raises
from sympy.utilities.lambdify import implemented_function
from sympy.tensor import IndexedBase, Idx
from sympy.matrices import MatrixSymbol, SparseMatrix, Matrix

# 导入 Rust 代码生成器
from sympy.printing.rust import rust_code

# 定义符号变量 x, y, z
x, y, z = symbols('x,y,z')

# 测试整数转换为 Rust 代码
def test_Integer():
    assert rust_code(Integer(42)) == "42"
    assert rust_code(Integer(-56)) == "-56"

# 测试关系运算符转换为 Rust 代码
def test_Relational():
    assert rust_code(Eq(x, y)) == "x == y"
    assert rust_code(Ne(x, y)) == "x != y"
    assert rust_code(Le(x, y)) == "x <= y"
    assert rust_code(Lt(x, y)) == "x < y"
    assert rust_code(Gt(x, y)) == "x > y"
    assert rust_code(Ge(x, y)) == "x >= y"

# 测试有理数转换为 Rust 代码
def test_Rational():
    assert rust_code(Rational(3, 7)) == "3_f64/7.0"
    assert rust_code(Rational(18, 9)) == "2"
    assert rust_code(Rational(3, -7)) == "-3_f64/7.0"
    assert rust_code(Rational(-3, -7)) == "3_f64/7.0"
    assert rust_code(x + Rational(3, 7)) == "x + 3_f64/7.0"
    assert rust_code(Rational(3, 7)*x) == "(3_f64/7.0)*x"

# 测试基本运算符转换为 Rust 代码
def test_basic_ops():
    assert rust_code(x + y) == "x + y"
    assert rust_code(x - y) == "x - y"
    assert rust_code(x * y) == "x*y"
    assert rust_code(x / y) == "x/y"
    assert rust_code(-x) == "-x"

# 测试自定义打印方法转换为 Rust 代码
def test_printmethod():
    class fabs(Abs):
        def _rust_code(self, printer):
            return "%s.fabs()" % printer._print(self.args[0])
    assert rust_code(fabs(x)) == "x.fabs()"
    a = MatrixSymbol("a", 1, 3)
    assert rust_code(a[0,0]) == 'a[0]'

# 测试函数转换为 Rust 代码
def test_Functions():
    assert rust_code(sin(x) ** cos(x)) == "x.sin().powf(x.cos())"
    assert rust_code(abs(x)) == "x.abs()"
    assert rust_code(ceiling(x)) == "x.ceil()"
    assert rust_code(floor(x)) == "x.floor()"

    # 自动重写测试
    assert rust_code(Mod(x, 3)) == 'x - 3*((1_f64/3.0)*x).floor()'

# 测试幂运算转换为 Rust 代码
def test_Pow():
    assert rust_code(1/x) == "x.recip()"
    assert rust_code(x**-1) == rust_code(x**-1.0) == "x.recip()"
    assert rust_code(sqrt(x)) == "x.sqrt()"
    assert rust_code(x**S.Half) == rust_code(x**0.5) == "x.sqrt()"

    assert rust_code(1/sqrt(x)) == "x.sqrt().recip()"
    assert rust_code(x**-S.Half) == rust_code(x**-0.5) == "x.sqrt().recip()"

    assert rust_code(1/pi) == "PI.recip()"
    assert rust_code(pi**-1) == rust_code(pi**-1.0) == "PI.recip()"
    assert rust_code(pi**-0.5) == "PI.sqrt().recip()"

    assert rust_code(x**Rational(1, 3)) == "x.cbrt()"
    assert rust_code(2**x) == "x.exp2()"
    assert rust_code(exp(x)) == "x.exp()"
    assert rust_code(x**3) == "x.powi(3)"
    assert rust_code(x**(y**3)) == "x.powf(y.powi(3))"
    assert rust_code(x**Rational(2, 3)) == "x.powf(2_f64/3.0)"
    # 使用 implemented_function 创建一个名为 'g' 的函数，该函数返回输入值的两倍
    g = implemented_function('g', Lambda(x, 2*x))
    
    # 使用 rust_code 函数生成给定表达式的 Rust 代码，并断言生成的代码符合预期输出
    assert rust_code(1/(g(x)*3.5)**(x - y**x)/(x**2 + y)) == \
        "(3.5*2*x).powf(-x + y.powf(x))/(x.powi(2) + y)"
    
    # 定义一个条件化的函数列表 _cond_cfunc，根据指数是否为整数选择不同的 Rust 函数名
    _cond_cfunc = [(lambda base, exp: exp.is_integer, "dpowi", 1),
                   (lambda base, exp: not exp.is_integer, "pow", 1)]
    
    # 使用 rust_code 函数生成 x**3 的 Rust 代码，并断言生成的代码符合预期输出
    assert rust_code(x**3, user_functions={'Pow': _cond_cfunc}) == 'x.dpowi(3)'
    
    # 使用 rust_code 函数生成 x**3.2 的 Rust 代码，并断言生成的代码符合预期输出
    assert rust_code(x**3.2, user_functions={'Pow': _cond_cfunc}) == 'x.pow(3.2)'
# 测试常数的转换为 Rust 代码的函数
def test_constants():
    # 断言将数学常数 pi 转换为 Rust 代码 "PI"
    assert rust_code(pi) == "PI"
    # 断言将 oo (正无穷大) 转换为 Rust 代码 "INFINITY"
    assert rust_code(oo) == "INFINITY"
    # 断言将 SymPy 中定义的 Infinity 转换为 Rust 代码 "INFINITY"
    assert rust_code(S.Infinity) == "INFINITY"
    # 断言将 -oo (负无穷大) 转换为 Rust 代码 "NEG_INFINITY"
    assert rust_code(-oo) == "NEG_INFINITY"
    # 断言将 SymPy 中定义的 NegativeInfinity 转换为 Rust 代码 "NEG_INFINITY"
    assert rust_code(S.NegativeInfinity) == "NEG_INFINITY"
    # 断言将 SymPy 中定义的 NaN 转换为 Rust 代码 "NAN"
    assert rust_code(S.NaN) == "NAN"
    # 断言将 exp(1) (自然对数的底 e) 转换为 Rust 代码 "E"
    assert rust_code(exp(1)) == "E"
    # 断言将 SymPy 中定义的 Exp1 (自然对数的底 e) 转换为 Rust 代码 "E"
    assert rust_code(S.Exp1) == "E"


# 测试其他常数的转换为 Rust 代码的函数
def test_constants_other():
    # 断言将 2*GoldenRatio 转换为 Rust 代码 "const GoldenRatio: f64 = %s;\n2*GoldenRatio" % GoldenRatio.evalf(17)
    assert rust_code(2*GoldenRatio) == "const GoldenRatio: f64 = %s;\n2*GoldenRatio" % GoldenRatio.evalf(17)
    # 断言将 2*Catalan 转换为 Rust 代码 "const Catalan: f64 = %s;\n2*Catalan" % Catalan.evalf(17)
    assert rust_code(2*Catalan) == "const Catalan: f64 = %s;\n2*Catalan" % Catalan.evalf(17)
    # 断言将 2*EulerGamma 转换为 Rust 代码 "const EulerGamma: f64 = %s;\n2*EulerGamma" % EulerGamma.evalf(17)
    assert rust_code(2*EulerGamma) == "const EulerGamma: f64 = %s;\n2*EulerGamma" % EulerGamma.evalf(17)


# 测试布尔值和逻辑运算符的转换为 Rust 代码的函数
def test_boolean():
    # 断言将 True 转换为 Rust 代码 "true"
    assert rust_code(True) == "true"
    # 断言将 SymPy 中定义的 true 转换为 Rust 代码 "true"
    assert rust_code(S.true) == "true"
    # 断言将 False 转换为 Rust 代码 "false"
    assert rust_code(False) == "false"
    # 断言将 SymPy 中定义的 false 转换为 Rust 代码 "false"
    assert rust_code(S.false) == "false"
    # 断言将 x & y 转换为 Rust 代码 "x && y"
    assert rust_code(x & y) == "x && y"
    # 断言将 x | y 转换为 Rust 代码 "x || y"
    assert rust_code(x | y) == "x || y"
    # 断言将 ~x 转换为 Rust 代码 "!x"
    assert rust_code(~x) == "!x"
    # 断言将 x & y & z 转换为 Rust 代码 "x && y && z"
    assert rust_code(x & y & z) == "x && y && z"
    # 断言将 x | y | z 转换为 Rust 代码 "x || y || z"
    assert rust_code(x | y | z) == "x || y || z"
    # 断言将 (x & y) | z 转换为 Rust 代码 "z || x && y"
    assert rust_code((x & y) | z) == "z || x && y"
    # 断言将 (x | y) & z 转换为 Rust 代码 "z && (x || y)"
    assert rust_code((x | y) & z) == "z && (x || y)"


# 测试 Piecewise 条件表达式的转换为 Rust 代码的函数
def test_Piecewise():
    expr = Piecewise((x, x < 1), (x + 2, True))
    # 断言将 Piecewise 条件表达式转换为 Rust 代码
    assert rust_code(expr) == (
            "if (x < 1) {\n"
            "    x\n"
            "} else {\n"
            "    x + 2\n"
            "}")
    # 断言将 Piecewise 条件表达式转换为 Rust 代码，并赋值给变量 r
    assert rust_code(expr, assign_to="r") == (
        "r = if (x < 1) {\n"
        "    x\n"
        "} else {\n"
        "    x + 2\n"
        "};")
    # 断言将 Piecewise 条件表达式转换为内联的 Rust 代码，并赋值给变量 r
    assert rust_code(expr, assign_to="r", inline=True) == (
        "r = if (x < 1) { x } else { x + 2 };")
    # 断言将带有多个条件分支的 Piecewise 条件表达式转换为内联的 Rust 代码
    expr = Piecewise((x, x < 1), (x + 1, x < 5), (x + 2, True))
    assert rust_code(expr, inline=True) == (
        "if (x < 1) { x } else if (x < 5) { x + 1 } else { x + 2 }")
    # 断言将带有多个条件分支的 Piecewise 条件表达式转换为 Rust 代码，并赋值给变量 r
    assert rust_code(expr, assign_to="r", inline=True) == (
        "r = if (x < 1) { x } else if (x < 5) { x + 1 } else { x + 2 };")
    # 断言将带有多个条件分支的 Piecewise 条件表达式转换为 Rust 代码，并赋值给变量 r
    assert rust_code(expr, assign_to="r") == (
        "r = if (x < 1) {\n"
        "    x\n"
        "} else if (x < 5) {\n"
        "    x + 1\n"
        "} else {\n"
        "    x + 2\n"
        "};")
    # 断言检查不带 True 条件的 Piecewise 表达式会引发 ValueError
    expr = Piecewise((x, x < 1), (x**2, x > 1), (sin(x), x > 0))
    raises(ValueError, lambda: rust_code(expr))


# 测试解引用打印的转换为 Rust 代码的函数
def test_dereference_printing():
    expr = x + y + sin(z) + z
    # 断言将表达式中 z 的解引用转换为 Rust 代码 "x + y + (*z) + (*z).sin()"
    assert rust_code(expr, dereference=[z]) == "x + y + (*z) + (*z).sin()"


# 测试符号函数的转换为 Rust 代码的函数
def test_sign():
    expr = sign(x) * y
    # 断言将 sign(x) * y 转换为 Rust 代码 "y*x.signum()"
    assert rust_code(expr) == "y*x.signum()"
    # 使用 rust_code 函数生成 Rust 代码，并断言生成的代码与预期的字符串相等
    assert rust_code(expr, assign_to='r') == "r = y*x.signum();"

    # 定义表达式 expr，包含了对 x + y 的符号函数 sign 和常数 42 的加法
    expr = sign(x + y) + 42
    # 使用 rust_code 函数生成 Rust 代码，并断言生成的代码与预期的字符串相等
    assert rust_code(expr) == "(x + y).signum() + 42"
    # 使用 rust_code 函数生成 Rust 代码，并断言生成的代码与预期的字符串相等，同时指定了赋值给 r
    assert rust_code(expr, assign_to='r') == "r = (x + y).signum() + 42;"

    # 定义表达式 expr，包含了对 x 的余弦函数 cos 的符号函数 sign
    expr = sign(cos(x))
    # 使用 rust_code 函数生成 Rust 代码，并断言生成的代码与预期的字符串相等
    assert rust_code(expr) == "x.cos().signum()"
# 定义函数 test_reserved_words，用于测试保留字处理功能
def test_reserved_words():

    # 使用 symbols 函数创建符号变量 x 和 y，其中 y 是一个保留字 "if"
    x, y = symbols("x if")

    # 创建表达式 sin(y)
    expr = sin(y)
    # 断言生成的 Rust 代码符合预期 "if_.sin()"
    assert rust_code(expr) == "if_.sin()"
    # 断言使用 dereference 参数处理后的 Rust 代码符合预期 "(*if_).sin()"
    assert rust_code(expr, dereference=[y]) == "(*if_).sin()"
    # 断言使用 reserved_word_suffix 参数处理后的 Rust 代码符合预期 "if_unreserved.sin()"
    assert rust_code(expr, reserved_word_suffix='_unreserved') == "if_unreserved.sin()"

    # 使用 raises 函数验证在 error_on_reserved=True 时抛出 ValueError 异常
    with raises(ValueError):
        rust_code(expr, error_on_reserved=True)


# 定义函数 test_ITE，用于测试 ITE 表达式的 Rust 代码生成
def test_ITE():
    # 创建 ITE 表达式 x < 1 ? y : z
    expr = ITE(x < 1, y, z)
    # 断言生成的 Rust 代码符合预期格式
    assert rust_code(expr) == (
            "if (x < 1) {\n"
            "    y\n"
            "} else {\n"
            "    z\n"
            "}")


# 定义函数 test_Indexed，用于测试 IndexedBase 的 Rust 代码生成
def test_Indexed():
    # 创建整数符号变量 n, m, o
    n, m, o = symbols('n m o', integer=True)
    # 创建索引对象 i, j, k
    i, j, k = Idx('i', n), Idx('j', m), Idx('k', o)

    # 创建 IndexedBase 对象 x[j]，生成 Rust 代码 "x[j]"
    x = IndexedBase('x')[j]
    assert rust_code(x) == "x[j]"

    # 创建 IndexedBase 对象 A[i, j]，生成 Rust 代码 "A[m*i + j]"
    A = IndexedBase('A')[i, j]
    assert rust_code(A) == "A[m*i + j]"

    # 创建 IndexedBase 对象 B[i, j, k]，生成 Rust 代码 "B[m*o*i + o*j + k]"
    B = IndexedBase('B')[i, j, k]
    assert rust_code(B) == "B[m*o*i + o*j + k]"


# 定义函数 test_dummy_loops，用于测试虚拟循环的 Rust 代码生成
def test_dummy_loops():
    # 创建整数符号变量 i, m，并设置 i 为虚拟符号
    i, m = symbols('i m', integer=True, cls=Dummy)
    # 创建 IndexedBase 对象 x 和 y
    x = IndexedBase('x')
    y = IndexedBase('y')
    # 创建索引对象 i
    i = Idx(i, m)

    # 断言生成的 Rust 代码符合预期格式
    assert rust_code(x[i], assign_to=y[i]) == (
        "for i in 0..m {\n"
        "    y[i] = x[i];\n"
        "}")


# 定义函数 test_loops，用于测试循环的 Rust 代码生成
def test_loops():
    # 创建整数符号变量 m, n
    m, n = symbols('m n', integer=True)
    # 创建 IndexedBase 对象 A, x, y, z
    A = IndexedBase('A')
    x = IndexedBase('x')
    y = IndexedBase('y')
    z = IndexedBase('z')
    # 创建索引对象 i, j
    i = Idx('i', m)
    j = Idx('j', n)

    # 断言生成的 Rust 代码符合预期格式
    assert rust_code(A[i, j]*x[j], assign_to=y[i]) == (
        "for i in 0..m {\n"
        "    y[i] = 0;\n"
        "}\n"
        "for i in 0..m {\n"
        "    for j in 0..n {\n"
        "        y[i] = A[n*i + j]*x[j] + y[i];\n"
        "    }\n"
        "}")

    # 断言生成的 Rust 代码符合预期格式
    assert rust_code(A[i, j]*x[j] + x[i] + z[i], assign_to=y[i]) == (
        "for i in 0..m {\n"
        "    y[i] = x[i] + z[i];\n"
        "}\n"
        "for i in 0..m {\n"
        "    for j in 0..n {\n"
        "        y[i] = A[n*i + j]*x[j] + y[i];\n"
        "    }\n"
        "}")


# 定义函数 test_loops_multiple_contractions，用于测试多重缩并的循环的 Rust 代码生成
def test_loops_multiple_contractions():
    # 创建整数符号变量 n, m, o, p
    n, m, o, p = symbols('n m o p', integer=True)
    # 创建 IndexedBase 对象 a, b, y
    a = IndexedBase('a')
    b = IndexedBase('b')
    y = IndexedBase('y')
    # 创建索引对象 i, j, k, l
    i = Idx('i', m)
    j = Idx('j', n)
    k = Idx('k', o)
    l = Idx('l', p)

    # 断言生成的 Rust 代码符合预期格式
    assert rust_code(b[j, k, l]*a[i, j, k, l], assign_to=y[i]) == (
        "for i in 0..m {\n"
        "    y[i] = 0;\n"
        "}\n"
        "for i in 0..m {\n"
        "    for j in 0..n {\n"
        "        for k in 0..o {\n"
        "            for l in 0..p {\n"
        "                y[i] = a[%s]*b[%s] + y[i];\n" % (i*n*o*p + j*o*p + k*p + l, j*o*p + k*p + l) +
        "            }\n"
        "        }\n"
        "    }\n"
        "}")


# 定义函数 test_loops_addfactor，用于测试带因子的循环的 Rust 代码生成
def test_loops_addfactor():
    # 创建整数符号变量 m, n, o, p
    m, n, o, p = symbols('m n o p', integer=True)
    # 创建 IndexedBase 对象 a, b, c, y
    a = IndexedBase('a')
    b = IndexedBase('b')
    c = IndexedBase('c')
    y = IndexedBase('y')
    # 创建索引对象 i, j, k, l
    i = Idx('i', m)
    j = Idx('j', n)
    k = Idx('k', o)
    l = Idx('l', p)

    # 生成 Rust 代码并赋值给变量 code
    code = rust_code((a[i, j, k, l] + b[i, j, k, l])*c[j, k, l], assign_to=y[i])
    # 确保代码块的内容与预期的字符串相等
    assert code == (
        "for i in 0..m {\n"  # 循环 i 从 0 到 m
        "    y[i] = 0;\n"    # 将 y[i] 初始化为 0
        "}\n"
        "for i in 0..m {\n"         # 循环 i 从 0 到 m
        "    for j in 0..n {\n"     # 循环 j 从 0 到 n
        "        for k in 0..o {\n" # 循环 k 从 0 到 o
        "            for l in 0..p {\n"                         # 循环 l 从 0 到 p
        "                y[i] = (a[%s] + b[%s])*c[%s] + y[i];\n" % (i*n*o*p + j*o*p + k*p + l, i*n*o*p + j*o*p + k*p + l, j*o*p + k*p + l) +\
        "            }\n"             # 将 y[i] 更新为新值
        "        }\n"                 # 结束 l 循环
        "    }\n"                     # 结束 k 循环
        "}\n"                         # 结束 j 循环
        "}")                          # 结束 i 循环
# 测试函数，用于检查 rust_code 函数是否能正确处理 TypeError 异常
def test_settings():
    raises(TypeError, lambda: rust_code(sin(x), method="garbage"))


# 测试内联函数的处理能力
def test_inline_function():
    # 定义符号变量 x
    x = symbols('x')
    # 创建 g 函数，其定义为 2*x
    g = implemented_function('g', Lambda(x, 2*x))
    # 断言 rust_code(g(x)) 的输出是否为 "2*x"
    assert rust_code(g(x)) == "2*x"

    # 将 g 函数重新定义为 2*x/Catalan，其中 Catalan 是一个常数
    g = implemented_function('g', Lambda(x, 2*x/Catalan))
    # 断言 rust_code(g(x)) 的输出是否为指定字符串
    assert rust_code(g(x)) == (
        "const Catalan: f64 = %s;\n2*x/Catalan" % Catalan.evalf(17))

    # 定义 IndexedBase 对象 A 和索引变量 i
    A = IndexedBase('A')
    i = Idx('i', symbols('n', integer=True))
    # 创建 g 函数，其定义为 x*(1 + x)*(2 + x)
    g = implemented_function('g', Lambda(x, x*(1 + x)*(2 + x)))
    # 断言 rust_code(g(A[i]), assign_to=A[i]) 的输出是否符合预期
    assert rust_code(g(A[i]), assign_to=A[i]) == (
        "for i in 0..n {\n"
        "    A[i] = (A[i] + 1)*(A[i] + 2)*A[i];\n"
        "}")


# 测试用户自定义函数的处理能力
def test_user_functions():
    # 定义符号变量 x 和 n
    x = symbols('x', integer=False)
    n = symbols('n', integer=True)
    # 自定义函数字典，包含 "ceiling" 和 "Abs" 函数的映射关系
    custom_functions = {
        "ceiling": "ceil",
        "Abs": [(lambda x: not x.is_integer, "fabs", 4), (lambda x: x.is_integer, "abs", 4)],
    }
    # 断言 rust_code(ceiling(x), user_functions=custom_functions) 的输出是否为 "x.ceil()"
    assert rust_code(ceiling(x), user_functions=custom_functions) == "x.ceil()"
    # 断言 rust_code(Abs(x), user_functions=custom_functions) 的输出是否为 "fabs(x)"
    assert rust_code(Abs(x), user_functions=custom_functions) == "fabs(x)"
    # 断言 rust_code(Abs(n), user_functions=custom_functions) 的输出是否为 "abs(n)"
    assert rust_code(Abs(n), user_functions=custom_functions) == "abs(n)"


# 测试矩阵的处理能力
def test_matrix():
    # 断言 rust_code(Matrix([1, 2, 3])) 的输出是否为 '[1, 2, 3]'
    assert rust_code(Matrix([1, 2, 3])) == '[1, 2, 3]'
    # 使用 raises 检查 rust_code(Matrix([[1, 2, 3]])) 是否会引发 ValueError 异常
    with raises(ValueError):
        rust_code(Matrix([[1, 2, 3]]))


# 测试稀疏矩阵的处理能力
def test_sparse_matrix():
    # 使用 raises 检查 rust_code(SparseMatrix([[1, 2, 3]])) 是否会引发 NotImplementedError 异常
    # gh-15791 是相关 GitHub 问题的编号
    with raises(NotImplementedError):
        rust_code(SparseMatrix([[1, 2, 3]]))
```