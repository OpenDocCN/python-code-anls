# `D:\src\scipysrc\sympy\sympy\printing\tests\test_jscode.py`

```
from sympy.core import (pi, oo, symbols, Rational, Integer, GoldenRatio,
                        EulerGamma, Catalan, Lambda, Dummy, S, Eq, Ne, Le,
                        Lt, Gt, Ge, Mod)
# 从 sympy.core 模块导入多个符号和常数

from sympy.functions import (Piecewise, sin, cos, Abs, exp, ceiling, sqrt,
                             sinh, cosh, tanh, asin, acos, acosh, Max, Min)
# 从 sympy.functions 模块导入多个数学函数

from sympy.testing.pytest import raises
# 导入 pytest 的 raises 函数，用于测试时处理异常

from sympy.printing.jscode import JavascriptCodePrinter
# 导入 JavascriptCodePrinter 类，用于将 SymPy 表达式转换为 JavaScript 代码的打印器

from sympy.utilities.lambdify import implemented_function
# 导入 implemented_function 函数，用于创建基于 Lambda 表达式的实现函数

from sympy.tensor import IndexedBase, Idx
# 导入 IndexedBase 和 Idx 类，用于处理张量的索引

from sympy.matrices import Matrix, MatrixSymbol
# 导入 Matrix 和 MatrixSymbol 类，用于处理矩阵和矩阵符号

from sympy.printing.jscode import jscode
# 导入 jscode 函数，用于将 SymPy 表达式转换为对应的 JavaScript 代码

x, y, z = symbols('x,y,z')
# 创建符号变量 x, y, z

def test_printmethod():
    assert jscode(Abs(x)) == "Math.abs(x)"
    # 测试绝对值函数的 JavaScript 代码转换

def test_jscode_sqrt():
    assert jscode(sqrt(x)) == "Math.sqrt(x)"
    assert jscode(x**0.5) == "Math.sqrt(x)"
    assert jscode(x**(S.One/3)) == "Math.cbrt(x)"
    # 测试平方根和立方根函数的 JavaScript 代码转换

def test_jscode_Pow():
    g = implemented_function('g', Lambda(x, 2*x))
    assert jscode(x**3) == "Math.pow(x, 3)"
    assert jscode(x**(y**3)) == "Math.pow(x, Math.pow(y, 3))"
    assert jscode(1/(g(x)*3.5)**(x - y**x)/(x**2 + y)) == \
        "Math.pow(3.5*2*x, -x + Math.pow(y, x))/(Math.pow(x, 2) + y)"
    assert jscode(x**-1.0) == '1/x'
    # 测试幂函数的 JavaScript 代码转换

def test_jscode_constants_mathh():
    assert jscode(exp(1)) == "Math.E"
    assert jscode(pi) == "Math.PI"
    assert jscode(oo) == "Number.POSITIVE_INFINITY"
    assert jscode(-oo) == "Number.NEGATIVE_INFINITY"
    # 测试数学常数的 JavaScript 代码转换

def test_jscode_constants_other():
    assert jscode(
        2*GoldenRatio) == "var GoldenRatio = %s;\n2*GoldenRatio" % GoldenRatio.evalf(17)
    assert jscode(2*Catalan) == "var Catalan = %s;\n2*Catalan" % Catalan.evalf(17)
    assert jscode(
        2*EulerGamma) == "var EulerGamma = %s;\n2*EulerGamma" % EulerGamma.evalf(17)
    # 测试其他常数的 JavaScript 代码转换

def test_jscode_Rational():
    assert jscode(Rational(3, 7)) == "3/7"
    assert jscode(Rational(18, 9)) == "2"
    assert jscode(Rational(3, -7)) == "-3/7"
    assert jscode(Rational(-3, -7)) == "3/7"
    # 测试有理数的 JavaScript 代码转换

def test_Relational():
    assert jscode(Eq(x, y)) == "x == y"
    assert jscode(Ne(x, y)) == "x != y"
    assert jscode(Le(x, y)) == "x <= y"
    assert jscode(Lt(x, y)) == "x < y"
    assert jscode(Gt(x, y)) == "x > y"
    assert jscode(Ge(x, y)) == "x >= y"
    # 测试关系运算的 JavaScript 代码转换

def test_Mod():
    assert jscode(Mod(x, y)) == '((x % y) + y) % y'
    assert jscode(Mod(x, x + y)) == '((x % (x + y)) + (x + y)) % (x + y)'
    p1, p2 = symbols('p1 p2', positive=True)
    assert jscode(Mod(p1, p2)) == 'p1 % p2'
    assert jscode(Mod(p1, p2 + 3)) == 'p1 % (p2 + 3)'
    assert jscode(Mod(-3, -7, evaluate=False)) == '(-3) % (-7)'
    assert jscode(-Mod(p1, p2)) == '-(p1 % p2)'
    assert jscode(x*Mod(p1, p2)) == 'x*(p1 % p2)'
    # 测试取模运算的 JavaScript 代码转换

def test_jscode_Integer():
    assert jscode(Integer(67)) == "67"
    assert jscode(Integer(-1)) == "-1"
    # 测试整数的 JavaScript 代码转换

def test_jscode_functions():
    assert jscode(sin(x) ** cos(x)) == "Math.pow(Math.sin(x), Math.cos(x))"
    assert jscode(sinh(x) * cosh(x)) == "Math.sinh(x)*Math.cosh(x)"
    # 测试函数的 JavaScript 代码转换
    # 断言：验证生成的 JavaScript 代码是否与预期的字符串相等
    assert jscode(Max(x, y) + Min(x, y)) == "Math.max(x, y) + Math.min(x, y)"
    
    # 断言：验证生成的 JavaScript 代码是否与预期的字符串相等
    assert jscode(tanh(x)*acosh(y)) == "Math.tanh(x)*Math.acosh(y)"
    
    # 断言：验证生成的 JavaScript 代码是否与预期的字符串相等
    assert jscode(asin(x)-acos(y)) == "-Math.acos(y) + Math.asin(x)"
def test_jscode_inline_function():
    # 定义符号变量 x
    x = symbols('x')
    # 创建一个 lambda 函数 g(x) = 2*x 的实现
    g = implemented_function('g', Lambda(x, 2*x))
    # 断言将 g(x) 转换为 JavaScript 代码应该是 "2*x"
    assert jscode(g(x)) == "2*x"
    # 更新 g 为 lambda 函数 g(x) = 2*x/Catalan 的实现
    g = implemented_function('g', Lambda(x, 2*x/Catalan))
    # 断言将 g(x) 转换为 JavaScript 代码应该是 "var Catalan = %s;\n2*x/Catalan" % Catalan.evalf(17)
    assert jscode(g(x)) == "var Catalan = %s;\n2*x/Catalan" % Catalan.evalf(17)
    # 创建 IndexedBase 对象 A
    A = IndexedBase('A')
    # 定义整数索引 i
    i = Idx('i', symbols('n', integer=True))
    # 创建 lambda 函数 g(x) = x*(1 + x)*(2 + x) 的实现
    g = implemented_function('g', Lambda(x, x*(1 + x)*(2 + x)))
    # 断言将 g(A[i]) 转换为 JavaScript 代码并将结果赋给 A[i]
    assert jscode(g(A[i]), assign_to=A[i]) == (
        "for (var i=0; i<n; i++){\n"
        "   A[i] = (A[i] + 1)*(A[i] + 2)*A[i];\n"
        "}"
    )


def test_jscode_exceptions():
    # 断言将 ceiling(x) 转换为 JavaScript 代码应该是 "Math.ceil(x)"
    assert jscode(ceiling(x)) == "Math.ceil(x)"
    # 断言将 Abs(x) 转换为 JavaScript 代码应该是 "Math.abs(x)"
    assert jscode(Abs(x)) == "Math.abs(x)"


def test_jscode_boolean():
    # 断言将 x & y 转换为 JavaScript 代码应该是 "x && y"
    assert jscode(x & y) == "x && y"
    # 断言将 x | y 转换为 JavaScript 代码应该是 "x || y"
    assert jscode(x | y) == "x || y"
    # 断言将 ~x 转换为 JavaScript 代码应该是 "!x"
    assert jscode(~x) == "!x"
    # 断言将 x & y & z 转换为 JavaScript 代码应该是 "x && y && z"
    assert jscode(x & y & z) == "x && y && z"
    # 断言将 x | y | z 转换为 JavaScript 代码应该是 "x || y || z"
    assert jscode(x | y | z) == "x || y || z"
    # 断言将 (x & y) | z 转换为 JavaScript 代码应该是 "z || x && y"
    assert jscode((x & y) | z) == "z || x && y"
    # 断言将 (x | y) & z 转换为 JavaScript 代码应该是 "z && (x || y)"
    assert jscode((x | y) & z) == "z && (x || y)"


def test_jscode_Piecewise():
    # 定义 Piecewise 表达式 expr
    expr = Piecewise((x, x < 1), (x**2, True))
    # 断言将 expr 转换为 JavaScript 代码应该是预期的多行字符串 s
    p = jscode(expr)
    s = \
"""\
((x < 1) ? (
   x
)
: (
   Math.pow(x, 2)
))\
"""
    assert p == s
    # 断言将 expr 转换为 JavaScript 代码并将结果赋给变量 c
    assert jscode(expr, assign_to="c") == (
    "if (x < 1) {\n"
    "   c = x;\n"
    "}\n"
    "else {\n"
    "   c = Math.pow(x, 2);\n"
    "}")
    # 检查 Piecewise 中没有 True (默认) 条件时会引发 ValueError
    expr = Piecewise((x, x < 1), (x**2, x > 1), (sin(x), x > 0))
    raises(ValueError, lambda: jscode(expr))


def test_jscode_Piecewise_deep():
    # 断言将 2*Piecewise((x, x < 1), (x**2, True)) 转换为 JavaScript 代码应该是预期的多行字符串 s
    p = jscode(2*Piecewise((x, x < 1), (x**2, True)))
    s = \
"""\
2*((x < 1) ? (
   x
)
: (
   Math.pow(x, 2)
))\
"""
    assert p == s


def test_jscode_settings():
    # 断言当指定 method="garbage" 时，调用 jscode(sin(x)) 会引发 TypeError
    raises(TypeError, lambda: jscode(sin(x), method="garbage"))


def test_jscode_Indexed():
    # 定义整数符号变量 n, m, o
    n, m, o = symbols('n m o', integer=True)
    # 定义整数索引变量 i, j, k
    i, j, k = Idx('i', n), Idx('j', m), Idx('k', o)
    # 创建 JavascriptCodePrinter 对象 p
    p = JavascriptCodePrinter()
    p._not_c = set()

    # 创建 IndexedBase 对象 x 并打印
    x = IndexedBase('x')[j]
    assert p._print_Indexed(x) == 'x[j]'
    # 创建 IndexedBase 对象 A 并打印
    A = IndexedBase('A')[i, j]
    assert p._print_Indexed(A) == 'A[%s]' % (m*i+j)
    # 创建 IndexedBase 对象 B 并打印
    B = IndexedBase('B')[i, j, k]
    assert p._print_Indexed(B) == 'B[%s]' % (i*o*m+j*o+k)

    # 断言 p._not_c 为空集合
    assert p._not_c == set()


def test_jscode_loops_matrix_vector():
    # 定义整数符号变量 n, m
    n, m = symbols('n m', integer=True)
    # 创建 IndexedBase 对象 A, x, y
    A = IndexedBase('A')
    x = IndexedBase('x')
    y = IndexedBase('y')
    # 定义整数索引变量 i, j
    i = Idx('i', m)
    j = Idx('j', n)

    # 期望的 JavaScript 循环代码字符串 s
    s = (
        'for (var i=0; i<m; i++){\n'
        '   y[i] = 0;\n'
        '}\n'
        'for (var i=0; i<m; i++){\n'
        '   for (var j=0; j<n; j++){\n'
        '      y[i] = A[n*i + j]*x[j] + y[i];\n'
        '   }\n'
        '}'
    )
    # 断言将 A[i, j]*x[j] 转换为 JavaScript 代码并将结果赋给 y[i]
    c = jscode(A[i, j]*x[j], assign_to=y[i])
    assert c == s


def test_dummy_loops():
    # 定义符号变量 i, m，并声明为虚拟符号
    i, m = symbols('i m', integer=True, cls=Dummy)
    # 创建 IndexedBase 对象 x, y
    x = IndexedBase('x')
    y = IndexedBase('y')
    # 定义整数索引变量 i
    i = Idx(i, m)
    # 构建预期的 JavaScript 代码字符串，包含一个 for 循环，用来将数组 x 的元素赋值给数组 y
    expected = (
        'for (var i_%(icount)i=0; i_%(icount)i<m_%(mcount)i; i_%(icount)i++){\n'
        '   y[i_%(icount)i] = x[i_%(icount)i];\n'
        '}'
    ) % {'icount': i.label.dummy_index, 'mcount': m.dummy_index}
    # 生成 JavaScript 代码，将 x[i] 的值赋给 y[i]
    code = jscode(x[i], assign_to=y[i])
    # 断言生成的 JavaScript 代码与预期的代码字符串相同
    assert code == expected
# 定义一个测试函数，用于生成 JavaScript 代码，计算多个数学运算
def test_jscode_loops_add():
    # 定义符号变量 n 和 m 为整数
    n, m = symbols('n m', integer=True)
    # 创建 IndexedBase 对象 A, x, y, z 分别表示索引基础
    A = IndexedBase('A')
    x = IndexedBase('x')
    y = IndexedBase('y')
    z = IndexedBase('z')
    # 创建索引对象 i 和 j
    i = Idx('i', m)
    j = Idx('j', n)

    # 定义 JavaScript 代码字符串 s，实现两个嵌套的 for 循环
    s = (
        'for (var i=0; i<m; i++){\n'  # 第一个 for 循环，遍历 i 从 0 到 m-1
        '   y[i] = x[i] + z[i];\n'    # 对 y[i] 赋值为 x[i] + z[i]
        '}\n'                         # 结束第一个 for 循环
        'for (var i=0; i<m; i++){\n'  # 第二个 for 循环，遍历 i 从 0 到 m-1
        '   for (var j=0; j<n; j++){\n'  # 内嵌的 for 循环，遍历 j 从 0 到 n-1
        '      y[i] = A[n*i + j]*x[j] + y[i];\n'  # 对 y[i] 进行累加操作
        '   }\n'                     # 结束内嵌的 for 循环
        '}'                          # 结束第二个 for 循环
    )
    # 调用 jscode 函数生成 JavaScript 代码，并使用 assert 语句检查生成的代码是否符合预期 s
    c = jscode(A[i, j]*x[j] + x[i] + z[i], assign_to=y[i])
    assert c == s


# 定义一个测试函数，用于生成 JavaScript 代码，执行多重收缩
def test_jscode_loops_multiple_contractions():
    # 定义符号变量 n, m, o, p 为整数
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

    # 定义 JavaScript 代码字符串 s，实现四重嵌套的 for 循环
    s = (
        'for (var i=0; i<m; i++){\n'  # 第一个 for 循环，遍历 i 从 0 到 m-1
        '   y[i] = 0;\n'             # 初始化 y[i] 为 0
        '}\n'                         # 结束第一个 for 循环
        'for (var i=0; i<m; i++){\n'  # 第二个 for 循环，遍历 i 从 0 到 m-1
        '   for (var j=0; j<n; j++){\n'  # 第三个嵌套的 for 循环，遍历 j 从 0 到 n-1
        '      for (var k=0; k<o; k++){\n'  # 第四个嵌套的 for 循环，遍历 k 从 0 到 o-1
        '         for (var l=0; l<p; l++){\n'  # 最内层的 for 循环，遍历 l 从 0 到 p-1
        '            y[i] = a[%s]*b[%s] + y[i];\n' % (i*n*o*p + j*o*p + k*p + l, j*o*p + k*p + l) +\
        '         }\n'                 # 对 y[i] 进行累加操作
        '      }\n'                     # 结束第四个嵌套的 for 循环
        '   }\n'                         # 结束第三个嵌套的 for 循环
        '}\n'                         # 结束第二个 for 循环
    )
    # 调用 jscode 函数生成 JavaScript 代码，并使用 assert 语句检查生成的代码是否符合预期 s
    c = jscode(b[j, k, l]*a[i, j, k, l], assign_to=y[i])
    assert c == s


# 定义一个测试函数，用于生成 JavaScript 代码，计算添加因子的运算
def test_jscode_loops_addfactor():
    # 定义符号变量 n, m, o, p 为整数
    n, m, o, p = symbols('n m o p', integer=True)
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

    # 定义 JavaScript 代码字符串 s，实现四重嵌套的 for 循环
    s = (
        'for (var i=0; i<m; i++){\n'  # 第一个 for 循环，遍历 i 从 0 到 m-1
        '   y[i] = 0;\n'             # 初始化 y[i] 为 0
        '}\n'                         # 结束第一个 for 循环
        'for (var i=0; i<m; i++){\n'  # 第二个 for 循环，遍历 i 从 0 到 m-1
        '   for (var j=0; j<n; j++){\n'  # 第三个嵌套的 for 循环，遍历 j 从 0 到 n-1
        '      for (var k=0; k<o; k++){\n'  # 第四个嵌套的 for 循环，遍历 k 从 0 到 o-1
        '         for (var l=0; l<p; l++){\n'  # 最内层的 for 循环，遍历 l 从 0 到 p-1
        '            y[i] = (a[%s] + b[%s])*c[%s] + y[i];\n' % (i*n*o*p + j*o*p + k*p + l, i*n*o*p + j*o*p + k*p + l, j*o*p + k*p + l) +\
        '         }\n'                 # 对 y[i] 进行累加操作
        '      }\n'                     # 结束第四个嵌套的 for 循环
        '   }\n'                         # 结束第三个嵌套的 for 循环
        '}\n'                         # 结束第二个 for 循环
    )
    # 调用 jscode 函数生成 JavaScript 代码，并使用 assert 语句检查生成的代码是否符合预期 s
    c = jscode((a[i, j, k, l] + b[i, j, k, l])*c[j, k, l], assign_to=y[i])
    assert c == s


# 定义一个测试函数，用于生成 JavaScript 代码，执行多个术语的运算
def test_jscode_loops_multiple_terms():
    # 定义符号变量 n, m, o, p 为整数
    n, m, o, p = symbols('n m o p', integer=True)
    # 创建 IndexedBase 对象 a, b, c, y
    a = IndexedBase('a')
    b = IndexedBase('b')
    c = IndexedBase('c')
    y = IndexedBase('y')
    # 创建索引对象 i, j, k
    i = Idx('i', m)
    j = Idx('j', n)
    k = Idx('k', o)

    # 定义 JavaScript 代码字符串 s0，初始化 y[i] 为 0
    s0 = (
        '
    s3 = (
        'for (var i=0; i<m; i++){\n'  # 循环变量 i 从 0 到 m-1
        '   for (var j=0; j<n; j++){\n'  # 循环变量 j 从 0 到 n-1
        '      y[i] = a[%s]*b[j] + y[i];\n' % (i*n + j) +\
        '   }\n'  # 内部循环结束
        '}\n'  # 外部循环结束
    )
    c = jscode(
        b[j]*a[i, j] + b[k]*a[i, k] + b[j]*b[k]*c[i, j, k], assign_to=y[i])  # 构造 JavaScript 代码表达式 c
    assert (c == s0 + s1 + s2 + s3[:-1] or  # 断言 c 与预期表达式之一相等
            c == s0 + s1 + s3 + s2[:-1] or
            c == s0 + s2 + s1 + s3[:-1] or
            c == s0 + s2 + s3 + s1[:-1] or
            c == s0 + s3 + s1 + s2[:-1] or
            c == s0 + s3 + s2 + s1[:-1])
# 定义一个函数来测试矩阵输出
def test_Matrix_printing():
    # 测试返回一个矩阵
    mat = Matrix([x*y, Piecewise((2 + x, y>0), (y, True)), sin(z)])
    # 创建一个 3x1 的矩阵符号对象 A
    A = MatrixSymbol('A', 3, 1)
    # 断言转换后的 JavaScript 代码与预期结果相符
    assert jscode(mat, A) == (
        "A[0] = x*y;\n"
        "if (y > 0) {\n"
        "   A[1] = x + 2;\n"
        "}\n"
        "else {\n"
        "   A[1] = y;\n"
        "}\n"
        "A[2] = Math.sin(z);")
    
    # 测试在表达式中使用矩阵元素
    expr = Piecewise((2*A[2, 0], x > 0), (A[2, 0], True)) + sin(A[1, 0]) + A[0, 0]
    # 断言转换后的 JavaScript 代码与预期结果相符
    assert jscode(expr) == (
        "((x > 0) ? (\n"
        "   2*A[2]\n"
        ")\n"
        ": (\n"
        "   A[2]\n"
        ")) + Math.sin(A[1]) + A[0]")
    
    # 测试在矩阵中使用矩阵元素
    q = MatrixSymbol('q', 5, 1)
    M = MatrixSymbol('M', 3, 3)
    # 创建一个矩阵 m，其中包含矩阵元素的表达式
    m = Matrix([[sin(q[1,0]), 0, cos(q[2,0])],
        [q[1,0] + q[2,0], q[3, 0], 5],
        [2*q[4, 0]/q[1,0], sqrt(q[0,0]) + 4, 0]])
    # 断言转换后的 JavaScript 代码与预期结果相符
    assert jscode(m, M) == (
        "M[0] = Math.sin(q[1]);\n"
        "M[1] = 0;\n"
        "M[2] = Math.cos(q[2]);\n"
        "M[3] = q[1] + q[2];\n"
        "M[4] = q[3];\n"
        "M[5] = 5;\n"
        "M[6] = 2*q[4]/q[1];\n"
        "M[7] = Math.sqrt(q[0]) + 4;\n"
        "M[8] = 0;")


def test_MatrixElement_printing():
    # 测试问题 #11821 的用例
    A = MatrixSymbol("A", 1, 3)
    B = MatrixSymbol("B", 1, 3)
    C = MatrixSymbol("C", 1, 3)

    # 断言转换后的 JavaScript 代码与预期结果相符
    assert(jscode(A[0, 0]) == "A[0]")
    # 断言转换后的 JavaScript 代码与预期结果相符
    assert(jscode(3 * A[0, 0]) == "3*A[0]")

    # 计算 F，并使用符号替换
    F = C[0, 0].subs(C, A - B)
    # 断言转换后的 JavaScript 代码与预期结果相符
    assert(jscode(F) == "(A - B)[0]")
```