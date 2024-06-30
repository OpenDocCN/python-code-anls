# `D:\src\scipysrc\sympy\sympy\printing\tests\test_rcode.py`

```
# 导入 SymPy 库中的核心模块和函数，包括符号、常数、函数等
from sympy.core import (S, pi, oo, Symbol, symbols, Rational, Integer,
                        GoldenRatio, EulerGamma, Catalan, Lambda, Dummy)
# 导入 SymPy 库中的数学函数，如分段函数、三角函数、指数函数等
from sympy.functions import (Piecewise, sin, cos, Abs, exp, ceiling, sqrt,
                             gamma, sign, Max, Min, factorial, beta)
# 导入 SymPy 库中的关系运算符，如等于、大于等于、小于等于等
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
# 导入 SymPy 库中的集合和范围对象
from sympy.sets import Range
# 导入 SymPy 库中的逻辑表达式对象
from sympy.logic import ITE
# 导入 SymPy 库中的代码生成相关模块，如循环、增量赋值、赋值等
from sympy.codegen import For, aug_assign, Assignment
# 导入 SymPy 测试框架中的异常处理函数
from sympy.testing.pytest import raises
# 导入 SymPy 中用于 R 代码打印的打印器
from sympy.printing.rcode import RCodePrinter
# 导入 SymPy 中用于实现函数的工具函数
from sympy.utilities.lambdify import implemented_function
# 导入 SymPy 中用于张量计算的基本索引和索引对象
from sympy.tensor import IndexedBase, Idx
# 导入 SymPy 中用于矩阵计算的矩阵和矩阵符号对象
from sympy.matrices import Matrix, MatrixSymbol

# 导入 SymPy 中的 rcode 函数，用于将 SymPy 表达式转换为 R 代码字符串
from sympy.printing.rcode import rcode

# 定义符号变量 x, y, z
x, y, z = symbols('x,y,z')

# 定义测试函数 test_printmethod
def test_printmethod():
    # 定义一个继承自 Abs 的新类 fabs，重载了 _rcode 方法
    class fabs(Abs):
        # 实现 _rcode 方法，将其参数打印成字符串 "abs(%s)"
        def _rcode(self, printer):
            return "abs(%s)" % printer._print(self.args[0])

    # 断言调用 rcode 函数对 fabs(x) 的输出结果是 "abs(x)"
    assert rcode(fabs(x)) == "abs(x)"

# 定义测试函数 test_rcode_sqrt
def test_rcode_sqrt():
    # 断言调用 rcode 函数对 sqrt(x) 的输出结果是 "sqrt(x)"
    assert rcode(sqrt(x)) == "sqrt(x)"
    # 断言调用 rcode 函数对 x**0.5 的输出结果是 "sqrt(x)"
    assert rcode(x**0.5) == "sqrt(x)"
    # 断言调用 rcode 函数对 sqrt(x) 的输出结果是 "sqrt(x)"
    assert rcode(sqrt(x)) == "sqrt(x)"

# 定义测试函数 test_rcode_Pow
def test_rcode_Pow():
    # 断言调用 rcode 函数对 x**3 的输出结果是 "x^3"
    assert rcode(x**3) == "x^3"
    # 断言调用 rcode 函数对 x**(y**3) 的输出结果是 "x^(y^3)"
    assert rcode(x**(y**3)) == "x^(y^3)"
    # 定义一个 Lambda 函数 g(x) = 2*x
    g = implemented_function('g', Lambda(x, 2*x))
    # 断言调用 rcode 函数对复杂表达式的输出结果
    assert rcode(1/(g(x)*3.5)**(x - y**x)/(x**2 + y)) == \
        "(3.5*2*x)^(-x + y^x)/(x^2 + y)"
    # 断言调用 rcode 函数对 x**-1.0 的输出结果是 '1.0/x'
    assert rcode(x**-1.0) == '1.0/x'
    # 断言调用 rcode 函数对 x**Rational(2, 3) 的输出结果是 'x^(2.0/3.0)'
    assert rcode(x**Rational(2, 3)) == 'x^(2.0/3.0)'
    # 定义条件函数列表 _cond_cfunc
    _cond_cfunc = [(lambda base, exp: exp.is_integer, "dpowi"),
                   (lambda base, exp: not exp.is_integer, "pow")]
    # 断言调用 rcode 函数对 x**3 的输出结果，使用自定义函数名字典
    assert rcode(x**3, user_functions={'Pow': _cond_cfunc}) == 'dpowi(x, 3)'
    # 断言调用 rcode 函数对 x**3.2 的输出结果，使用自定义函数名字典
    assert rcode(x**3.2, user_functions={'Pow': _cond_cfunc}) == 'pow(x, 3.2)'

# 定义测试函数 test_rcode_Max
def test_rcode_Max():
    # 断言调用 rcode 函数对 Max(x,x*x) 的输出结果，使用自定义函数名字典
    assert rcode(Max(x,x*x),user_functions={"Max":"my_max", "Pow":"my_pow"}) == 'my_max(x, my_pow(x, 2))'

# 定义测试函数 test_rcode_constants_mathh
def test_rcode_constants_mathh():
    # 断言调用 rcode 函数对 exp(1) 的输出结果是 "exp(1)"
    assert rcode(exp(1)) == "exp(1)"
    # 断言调用 rcode 函数对 pi 的输出结果是 "pi"
    assert rcode(pi) == "pi"
    # 断言调用 rcode 函数对 oo 的输出结果是 "Inf"
    assert rcode(oo) == "Inf"
    # 断言调用 rcode 函数对 -oo 的输出结果是 "-Inf"
    assert rcode(-oo) == "-Inf"

# 定义测试函数 test_rcode_constants_other
def test_rcode_constants_other():
    # 断言调用 rcode 函数对 2*GoldenRatio 的输出结果
    assert rcode(2*GoldenRatio) == "GoldenRatio = 1.61803398874989;\n2*GoldenRatio"
    # 断言调用 rcode 函数对 2*Catalan 的输出结果
    assert rcode(
        2*Catalan) == "Catalan = 0.915965594177219;\n2*Catalan"
    # 断言调用 rcode 函数对 2*EulerGamma 的输出结果
    assert rcode(2*EulerGamma) == "EulerGamma = 0.577215664901533;\n2*EulerGamma"

# 定义测试函数 test_rcode_Rational
def test_rcode_Rational():
    # 断言调用 rcode 函数对 Rational(3, 7) 的输出结果是 "3.0/7.0"
    assert rcode(Rational(3, 7)) == "3.0/7.0"
    # 断言调用 rcode 函数对 Rational(18, 9) 的输出结果是 "2"
    assert rcode(Rational(18, 9)) == "2"
    # 断言调用 rcode 函数对 Rational(3, -7) 的输出结果是 "-3.0/7.0"
    assert rcode(Rational(3, -7)) == "-3.0/7.0"
    # 断言调用 rcode 函数对 Rational(-3, -7) 的输出结果是 "3.0/7.0"
    assert rcode(Rational(-3, -7)) == "3.0/7.0"
    # 断言调用 rcode 函数对 x + Rational(3, 7) 的输出结果是 "x + 3.0/7.0"
    assert rcode(x + Rational(3, 7)) == "x + 3.0/7.0"
    # 断言调用 rcode 函数对 Rational(3, 7)*x 的输出结果是 "(3.0/7.0)*x"
    assert rcode(Rational(3, 7)*x) == "(3.0/7.0)*x"

# 定义测试函数 test_rcode_Integer
def test_rcode_Integer():
    # 断言调用 rcode 函数对 Integer(67) 的输出结果是 "67"
    # 创建一个符号变量 x
    x = symbols('x')
    
    # 创建一个函数 g，该函数使用 Lambda 表达式定义为 2*x
    g = implemented_function('g', Lambda(x, 2*x))
    
    # 使用 rcode 函数检查 g(x) 的字符串表示是否为 "2*x"
    assert rcode(g(x)) == "2*x"
    
    # 更新函数 g，使用 Lambda 表达式定义为 2*x/Catalan
    g = implemented_function('g', Lambda(x, 2*x/Catalan))
    
    # 使用 rcode 函数检查 g(x) 的字符串表示是否为 "Catalan = <value>; 2*x/Catalan"，其中 Catalan 是一个数学常数
    assert rcode(g(x)) == "Catalan = %s;\n2*x/Catalan" % Catalan.n()
    
    # 创建一个 IndexedBase 对象 A，用于表示一组索引为 i 的符号
    A = IndexedBase('A')
    
    # 创建一个索引变量 i，其范围为 1 到 n
    i = Idx('i', symbols('n', integer=True))
    
    # 更新函数 g，使用 Lambda 表达式定义为 x*(1 + x)*(2 + x)
    g = implemented_function('g', Lambda(x, x*(1 + x)*(2 + x)))
    
    # 使用 rcode 函数生成 g(A[i]) 的字符串表示，并将结果赋值给 A[i]
    res = rcode(g(A[i]), assign_to=A[i])
    
    # 定义字符串 ref，表示一个 for 循环的代码块，用于计算 A[i] 的新值
    ref = (
        "for (i in 1:n){\n"
        "   A[i] = (A[i] + 1)*(A[i] + 2)*A[i];\n"
        "}"
    )
    
    # 使用 assert 语句检查 res 是否等于 ref，确保生成的代码正确
    assert res == ref
# 测试异常情况下的 rcode 函数调用
def test_rcode_exceptions():
    assert rcode(ceiling(x)) == "ceiling(x)"  # 测试对 ceiling 函数的调用
    assert rcode(Abs(x)) == "abs(x)"  # 测试对 Abs 函数的调用
    assert rcode(gamma(x)) == "gamma(x)"  # 测试对 gamma 函数的调用


# 测试使用自定义函数的 rcode 函数调用
def test_rcode_user_functions():
    x = symbols('x', integer=False)
    n = symbols('n', integer=True)
    custom_functions = {
        "ceiling": "myceil",  # 将 ceiling 函数映射为 myceil
        "Abs": [(lambda x: not x.is_integer, "fabs"), (lambda x: x.is_integer, "abs")],  # 根据 x 是否为整数选择不同的函数映射
    }
    assert rcode(ceiling(x), user_functions=custom_functions) == "myceil(x)"  # 测试使用自定义函数映射的 ceiling 函数调用
    assert rcode(Abs(x), user_functions=custom_functions) == "fabs(x)"  # 测试使用自定义函数映射的 Abs 函数调用
    assert rcode(Abs(n), user_functions=custom_functions) == "abs(n)"  # 测试使用自定义函数映射的 Abs 函数调用


# 测试布尔表达式的 rcode 函数调用
def test_rcode_boolean():
    assert rcode(True) == "True"  # 测试对 True 值的调用
    assert rcode(S.true) == "True"  # 测试对 S.true 的调用
    assert rcode(False) == "False"  # 测试对 False 值的调用
    assert rcode(S.false) == "False"  # 测试对 S.false 的调用
    assert rcode(x & y) == "x & y"  # 测试对位运算 & 的调用
    assert rcode(x | y) == "x | y"  # 测试对位运算 | 的调用
    assert rcode(~x) == "!x"  # 测试对位取反 ~x 的调用
    assert rcode(x & y & z) == "x & y & z"  # 测试对多个位运算 & 的调用
    assert rcode(x | y | z) == "x | y | z"  # 测试对多个位运算 | 的调用
    assert rcode((x & y) | z) == "z | x & y"  # 测试对混合位运算的调用
    assert rcode((x | y) & z) == "z & (x | y)"  # 测试对混合位运算的调用


# 测试关系运算符的 rcode 函数调用
def test_rcode_Relational():
    assert rcode(Eq(x, y)) == "x == y"  # 测试相等关系运算符 ==
    assert rcode(Ne(x, y)) == "x != y"  # 测试不等关系运算符 !=
    assert rcode(Le(x, y)) == "x <= y"  # 测试小于等于关系运算符 <=
    assert rcode(Lt(x, y)) == "x < y"  # 测试小于关系运算符 <
    assert rcode(Gt(x, y)) == "x > y"  # 测试大于关系运算符 >
    assert rcode(Ge(x, y)) == "x >= y"  # 测试大于等于关系运算符 >=


# 测试 Piecewise 函数的 rcode 函数调用
def test_rcode_Piecewise():
    expr = Piecewise((x, x < 1), (x**2, True))  # 创建 Piecewise 表达式
    res = rcode(expr)
    ref = "ifelse(x < 1,x,x^2)"  # 预期的 Piecewise 表达式转换结果
    assert res == ref

    tau = Symbol("tau")
    res = rcode(expr, tau)
    ref = "tau = ifelse(x < 1,x,x^2);"  # 预期的 Piecewise 表达式转换结果，带有赋值语句
    assert res == ref

    expr = 2 * Piecewise((x, x < 1), (x**2, x < 2), (x**3, True))  # 创建带系数的 Piecewise 表达式
    assert rcode(expr) == "2*ifelse(x < 1,x,ifelse(x < 2,x^2,x^3))"  # 预期的带系数的 Piecewise 表达式转换结果
    res = rcode(expr, assign_to='c')
    assert res == "c = 2*ifelse(x < 1,x,ifelse(x < 2,x^2,x^3));"  # 预期带有赋值语句的带系数的 Piecewise 表达式转换结果

    # 检查 Piecewise 函数没有默认条件 True 时的错误情况
    # expr = Piecewise((x, x < 1), (x**2, x > 1), (sin(x), x > 0))
    # raises(ValueError, lambda: rcode(expr))
    expr = 2 * Piecewise((x, x < 1), (x**2, x < 2))
    assert (rcode(expr)) == "2*ifelse(x < 1,x,ifelse(x < 2,x^2,NA))"  # 预期的带缺省值的 Piecewise 表达式转换结果


# 测试 sinc 函数的 rcode 函数调用
def test_rcode_sinc():
    from sympy.functions.elementary.trigonometric import sinc
    expr = sinc(x)
    res = rcode(expr)
    ref = "(ifelse(x != 0,sin(x)/x,1))"  # 预期的 sinc 函数表达式转换结果
    assert res == ref


# 测试深层次的 Piecewise 函数的 rcode 函数调用
def test_rcode_Piecewise_deep():
    p = rcode(2 * Piecewise((x, x < 1), (x + 1, x < 2), (x**2, True)))
    assert p == "2*ifelse(x < 1,x,ifelse(x < 2,x + 1,x^2))"  # 预期的深层次 Piecewise 表达式转换结果
    expr = x * y * z + x**2 + y**2 + Piecewise((0, x < 0.5), (1, True)) + cos(z) - 1
    p = rcode(expr)
    ref = "x^2 + x*y*z + y^2 + ifelse(x < 0.5,0,1) + cos(z) - 1"  # 预期的深层次 Piecewise 表达式转换结果
    assert p == ref

    ref = "c = x^2 + x*y*z + y^2 + ifelse(x < 0.5,0,1) + cos(z) - 1;"  # 预期带有赋值语句的深层次 Piecewise 表达式转换结果
    p = rcode(expr, assign_to='c')
    assert p == ref


# 测试 ITE 表达式的 rcode 函数调用
def test_rcode_ITE():
    expr = ITE(x < 1, y, z)
    p = rcode(expr)
    ref = "ifelse(x < 1,y,z)"  # 预期的 ITE 表达式转换结果
    assert p == ref


# 测试 rcode 函数的设置选项
def test_rcode_settings():
    raises(TypeError, lambda: rcode(sin(x), method="garbage"))  # 测试 rcode 函数设置选项时的异常情况
def test_rcode_Indexed():
    # 定义符号变量 n, m, o
    n, m, o = symbols('n m o', integer=True)
    # 创建索引对象 i, j, k
    i, j, k = Idx('i', n), Idx('j', m), Idx('k', o)
    # 创建 RCodePrinter 对象
    p = RCodePrinter()
    # 初始化空集合 _not_r
    p._not_r = set()

    # 创建 IndexedBase 对象 x，并打印其表达式
    x = IndexedBase('x')[j]
    assert p._print_Indexed(x) == 'x[j]'
    # 创建 IndexedBase 对象 A，并打印其表达式
    A = IndexedBase('A')[i, j]
    assert p._print_Indexed(A) == 'A[i, j]'
    # 创建 IndexedBase 对象 B，并打印其表达式
    B = IndexedBase('B')[i, j, k]
    assert p._print_Indexed(B) == 'B[i, j, k]'

    # 验证 _not_r 是否为空集合
    assert p._not_r == set()

def test_rcode_Indexed_without_looking_for_contraction():
    # 设置长度变量 len_y 为 5
    len_y = 5
    # 创建形状为 (len_y,) 的 IndexedBase 对象 y
    y = IndexedBase('y', shape=(len_y,))
    # 创建形状为 (len_y,) 的 IndexedBase 对象 x
    x = IndexedBase('x', shape=(len_y,))
    # 创建形状为 (len_y-1,) 的 IndexedBase 对象 Dy
    Dy = IndexedBase('Dy', shape=(len_y-1,))
    # 创建索引对象 i，范围为 0 到 len_y-2
    i = Idx('i', len_y-1)
    # 创建方程对象 e，表示 Dy[i] = (y[i+1] - y[i]) / (x[i+1] - x[i])
    e = Eq(Dy[i], (y[i+1] - y[i]) / (x[i+1] - x[i]))
    # 将方程 e 右侧的 R 代码生成为字符串，不进行合并操作
    code0 = rcode(e.rhs, assign_to=e.lhs, contract=False)
    assert code0 == 'Dy[i] = (y[%s] - y[i])/(x[%s] - x[i]);' % (i + 1, i + 1)


def test_rcode_loops_matrix_vector():
    # 定义符号变量 n, m
    n, m = symbols('n m', integer=True)
    # 创建 IndexedBase 对象 A, x, y
    A = IndexedBase('A')
    x = IndexedBase('x')
    y = IndexedBase('y')
    # 创建索引对象 i, j
    i = Idx('i', m)
    j = Idx('j', n)

    # 构造预期的 R 代码字符串 s
    s = (
        'for (i in 1:m){\n'
        '   y[i] = 0;\n'
        '}\n'
        'for (i in 1:m){\n'
        '   for (j in 1:n){\n'
        '      y[i] = A[i, j]*x[j] + y[i];\n'
        '   }\n'
        '}'
    )
    # 将 A[i, j]*x[j] 的 R 代码生成为字符串 c
    c = rcode(A[i, j]*x[j], assign_to=y[i])
    assert c == s


def test_dummy_loops():
    # 可以选择以下任一行代码
    # [Dummy(s, integer=True) for s in 'im']
    # 或 [Dummy(integer=True) for s in 'im']
    # 定义整数符号变量 i, m 作为 Dummy 变量
    i, m = symbols('i m', integer=True, cls=Dummy)
    # 创建 IndexedBase 对象 x, y
    x = IndexedBase('x')
    y = IndexedBase('y')
    # 创建索引对象 i，表示范围为 1 到 m
    i = Idx(i, m)

    # 构造预期的 R 代码字符串 expected
    expected = (
            'for (i_%(icount)i in 1:m_%(mcount)i){\n'
        '   y[i_%(icount)i] = x[i_%(icount)i];\n'
        '}'
    ) % {'icount': i.label.dummy_index, 'mcount': m.dummy_index}
    # 将 x[i] 的 R 代码生成为字符串 code
    code = rcode(x[i], assign_to=y[i])
    assert code == expected


def test_rcode_loops_add():
    # 定义符号变量 n, m
    n, m = symbols('n m', integer=True)
    # 创建 IndexedBase 对象 A, x, y, z
    A = IndexedBase('A')
    x = IndexedBase('x')
    y = IndexedBase('y')
    z = IndexedBase('z')
    # 创建索引对象 i, j
    i = Idx('i', m)
    j = Idx('j', n)

    # 构造预期的 R 代码字符串 s
    s = (
        'for (i in 1:m){\n'
        '   y[i] = x[i] + z[i];\n'
        '}\n'
        'for (i in 1:m){\n'
        '   for (j in 1:n){\n'
        '      y[i] = A[i, j]*x[j] + y[i];\n'
        '   }\n'
        '}'
    )
    # 将 A[i, j]*x[j] + x[i] + z[i] 的 R 代码生成为字符串 c
    c = rcode(A[i, j]*x[j] + x[i] + z[i], assign_to=y[i])
    assert c == s


def test_rcode_loops_multiple_contractions():
    # 定义符号变量 n, m, o, p
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

    # 构造预期的 R 代码字符串 s
    s = (
        'for (i in 1:m){\n'
        '   y[i] = 0;\n'
        '}\n'
        'for (i in 1:m){\n'
        '   for (j in 1:n){\n'
        '      for (k in 1:o){\n'
        '         for (l in 1:p){\n'
        '            y[i] = a[i, j, k, l]*b[j, k, l] + y[i];\n'
        '         }\n'
        '      }\n'
        '   }\n'
        '}'
    )
    # 将 a[i, j, k, l]*b[j, k, l] 的 R 代码生成为字符串 c
    c = rcode(a[i, j, k, l]*b[j, k, l], assign_to=y[i])
    assert c == s
    # 调用函数 rcode，并将表达式 rcode(b[j, k, l]*a[i, j, k, l], assign_to=y[i]) 的结果赋值给变量 c
    c = rcode(b[j, k, l]*a[i, j, k, l], assign_to=y[i])
    # 使用断言确保 c 的值等于 s，如果不等则会触发 AssertionError
    assert c == s
def test_rcode_loops_addfactor():
    # 定义符号变量
    n, m, o, p = symbols('n m o p', integer=True)
    # 定义索引基类
    a = IndexedBase('a')
    b = IndexedBase('b')
    c = IndexedBase('c')
    y = IndexedBase('y')
    # 定义索引变量
    i = Idx('i', m)
    j = Idx('j', n)
    k = Idx('k', o)
    l = Idx('l', p)

    # 构造多重循环的字符串表达式
    s = (
        'for (i in 1:m){\n'
        '   y[i] = 0;\n'
        '}\n'
        'for (i in 1:m){\n'
        '   for (j in 1:n){\n'
        '      for (k in 1:o){\n'
        '         for (l in 1:p){\n'
        '            y[i] = (a[i, j, k, l] + b[i, j, k, l])*c[j, k, l] + y[i];\n'
        '         }\n'
        '      }\n'
        '   }\n'
        '}'
    )
    # 调用 rcode 函数生成 R 代码，并赋值给 c
    c = rcode((a[i, j, k, l] + b[i, j, k, l])*c[j, k, l], assign_to=y[i])
    # 断言生成的 R 代码 c 与预期字符串 s 相等
    assert c == s


def test_rcode_loops_multiple_terms():
    # 定义符号变量
    n, m, o, p = symbols('n m o p', integer=True)
    # 定义索引基类
    a = IndexedBase('a')
    b = IndexedBase('b')
    c = IndexedBase('c')
    y = IndexedBase('y')
    # 定义索引变量
    i = Idx('i', m)
    j = Idx('j', n)
    k = Idx('k', o)

    # 定义多个不同顺序的循环字符串表达式
    s0 = (
        'for (i in 1:m){\n'
        '   y[i] = 0;\n'
        '}\n'
    )
    s1 = (
        'for (i in 1:m){\n'
        '   for (j in 1:n){\n'
        '      for (k in 1:o){\n'
        '         y[i] = b[j]*b[k]*c[i, j, k] + y[i];\n'
        '      }\n'
        '   }\n'
        '}\n'
    )
    s2 = (
        'for (i in 1:m){\n'
        '   for (k in 1:o){\n'
        '      y[i] = a[i, k]*b[k] + y[i];\n'
        '   }\n'
        '}\n'
    )
    s3 = (
        'for (i in 1:m){\n'
        '   for (j in 1:n){\n'
        '      y[i] = a[i, j]*b[j] + y[i];\n'
        '   }\n'
        '}\n'
    )
    # 调用 rcode 函数生成 R 代码，并赋值给 c
    c = rcode(
        b[j]*a[i, j] + b[k]*a[i, k] + b[j]*b[k]*c[i, j, k], assign_to=y[i])

    # 创建参考的预期输出结果字典
    ref={}
    ref[0] = s0 + s1 + s2 + s3[:-1]
    ref[1] = s0 + s1 + s3 + s2[:-1]
    ref[2] = s0 + s2 + s1 + s3[:-1]
    ref[3] = s0 + s2 + s3 + s1[:-1]
    ref[4] = s0 + s3 + s1 + s2[:-1]
    ref[5] = s0 + s3 + s2 + s1[:-1]

    # 断言生成的 R 代码 c 在预期结果字典中
    assert (c == ref[0] or
            c == ref[1] or
            c == ref[2] or
            c == ref[3] or
            c == ref[4] or
            c == ref[5])


def test_dereference_printing():
    # 定义表达式
    expr = x + y + sin(z) + z
    # 断言调用 rcode 函数生成的结果符合预期
    assert rcode(expr, dereference=[z]) == "x + y + (*z) + sin((*z))"


def test_Matrix_printing():
    # 测试生成矩阵
    mat = Matrix([x*y, Piecewise((2 + x, y>0), (y, True)), sin(z)])
    A = MatrixSymbol('A', 3, 1)
    # 调用 rcode 函数生成 R 代码，并赋值给 p
    p = rcode(mat, A)
    # 断言生成的 R 代码 p 与预期结果相等
    assert p == (
        "A[0] = x*y;\n"
        "A[1] = ifelse(y > 0,x + 2,y);\n"
        "A[2] = sin(z);")
    # 测试在表达式中使用矩阵元素
    expr = Piecewise((2*A[2, 0], x > 0), (A[2, 0], True)) + sin(A[1, 0]) + A[0, 0]
    # 调用 rcode 函数生成 R 代码，并赋值给 p
    p = rcode(expr)
    # 断言生成的 R 代码 p 与预期结果相等
    assert p  == ("ifelse(x > 0,2*A[2],A[2]) + sin(A[1]) + A[0]")
    # 测试在矩阵中使用矩阵元素
    q = MatrixSymbol('q', 5, 1)
    M = MatrixSymbol('M', 3, 3)
    m = Matrix([[sin(q[1,0]), 0, cos(q[2,0])],
        [q[1,0] + q[2,0], q[3, 0], 5],
        [2*q[4, 0]/q[1,0], sqrt(q[0,0]) + 4, 0]])
    # 使用 assert 语句检查函数 rcode(m, M) 的返回结果是否符合预期值
    assert rcode(m, M) == (
        # 定义字符串，表示期望的 M 数组的各个元素赋值语句
        "M[0] = sin(q[1]);\n"
        "M[1] = 0;\n"
        "M[2] = cos(q[2]);\n"
        "M[3] = q[1] + q[2];\n"
        "M[4] = q[3];\n"
        "M[5] = 5;\n"
        "M[6] = 2*q[4]/q[1];\n"
        "M[7] = sqrt(q[0]) + 4;\n"
        "M[8] = 0;")
# 定义测试函数 test_rcode_sgn，用于测试 rcode 函数处理符号表达式的情况
def test_rcode_sgn():

    # 创建表达式 expr，其中包含符号函数 sign(x) 乘以变量 y
    expr = sign(x) * y
    # 断言 rcode 函数处理后的结果符合预期
    assert rcode(expr) == 'y*sign(x)'
    
    # 调用 rcode 函数处理 expr，指定赋值字符串为 'z'
    p = rcode(expr, 'z')
    # 断言处理后的结果符合预期赋值形式
    assert p == 'z = y*sign(x);'

    # 调用 rcode 处理复杂表达式 sign(2*x + x**2) * x + x**2
    p = rcode(sign(2 * x + x**2) * x + x**2)
    # 断言处理后的结果符合预期字符串形式
    assert p == "x^2 + x*sign(x^2 + 2*x)"

    # 创建表达式 expr，其中包含符号函数 sign(cos(x))
    expr = sign(cos(x))
    # 调用 rcode 函数处理 expr
    p = rcode(expr)
    # 断言处理后的结果符合预期字符串形式
    assert p == 'sign(cos(x))'

# 定义测试函数 test_rcode_Assignment，用于测试 rcode 处理赋值表达式的情况
def test_rcode_Assignment():
    # 断言 rcode 函数处理赋值表达式 Assignment(x, y + z) 的结果符合预期
    assert rcode(Assignment(x, y + z)) == 'x = y + z;'
    # 断言 rcode 函数处理增量赋值表达式 aug_assign(x, '+', y + z) 的结果符合预期
    assert rcode(aug_assign(x, '+', y + z)) == 'x += y + z;'

# 定义测试函数 test_rcode_For，用于测试 rcode 处理 For 循环表达式的情况
def test_rcode_For():
    # 创建 For 循环对象 f，其中 x 在区间 [0, 10) 中以步长 2 递增，循环体为将 y 与 x 相乘的增量赋值表达式
    f = For(x, Range(0, 10, 2), [aug_assign(y, '*', x)])
    # 调用 rcode 函数处理 For 循环对象 f
    sol = rcode(f)
    # 断言处理后的结果符合预期字符串形式
    assert sol == ("for(x in seq(from=0, to=9, by=2){\n"
                   "   y *= x;\n"
                   "}")

# 定义测试函数 test_MatrixElement_printing，用于测试 rcode 处理矩阵元素的打印情况
def test_MatrixElement_printing():
    # 创建矩阵符号 A, B, C，分别为 1x3 矩阵
    A = MatrixSymbol("A", 1, 3)
    B = MatrixSymbol("B", 1, 3)
    C = MatrixSymbol("C", 1, 3)

    # 断言 rcode 函数处理 A 的第一个元素 A[0, 0] 的结果符合预期
    assert rcode(A[0, 0]) == "A[0]"
    # 断言 rcode 函数处理表达式 3 * A[0, 0] 的结果符合预期
    assert rcode(3 * A[0, 0]) == "3*A[0]"

    # 创建表达式 F，其中 C 替换为 A - B，然后取其第一个元素
    F = C[0, 0].subs(C, A - B)
    # 断言 rcode 函数处理表达式 F 的结果符合预期
    assert rcode(F) == "(A - B)[0]"
```