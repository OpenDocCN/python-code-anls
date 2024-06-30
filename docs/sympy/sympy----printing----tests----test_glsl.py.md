# `D:\src\scipysrc\sympy\sympy\printing\tests\test_glsl.py`

```
# 导入所需的符号、常数和函数
from sympy.core import (pi, symbols, Rational, Integer, GoldenRatio, EulerGamma,
                        Catalan, Lambda, Dummy, Eq, Ne, Le, Lt, Gt, Ge)
# 导入数学函数
from sympy.functions import Piecewise, sin, cos, Abs, exp, ceiling, sqrt
# 导入测试相关模块
from sympy.testing.pytest import raises, warns_deprecated_sympy
# 导入 GLSL 打印器
from sympy.printing.glsl import GLSLPrinter
# 导入字符串打印器
from sympy.printing.str import StrPrinter
# 导入 lambdify 函数
from sympy.utilities.lambdify import implemented_function
# 导入 IndexedBase 和 Idx 类
from sympy.tensor import IndexedBase, Idx
# 导入矩阵相关类和函数
from sympy.matrices import Matrix, MatrixSymbol
# 导入 Tuple 类
from sympy.core import Tuple
# 导入 GLSL 代码生成函数
from sympy.printing.glsl import glsl_code
# 导入文本包装模块
import textwrap

# 定义符号变量
x, y, z = symbols('x,y,z')

# 测试打印绝对值函数的 GLSL 代码
def test_printmethod():
    assert glsl_code(Abs(x)) == "abs(x)"

# 测试不使用运算符的 GLSL 代码生成
def test_print_without_operators():
    assert glsl_code(x*y, use_operators=False) == 'mul(x, y)'
    assert glsl_code(x**y+z, use_operators=False) == 'add(pow(x, y), z)'
    assert glsl_code(x*(y+z), use_operators=False) == 'mul(x, add(y, z))'
    assert glsl_code(x*(y+z), use_operators=False) == 'mul(x, add(y, z))'
    assert glsl_code(x*(y+z**y**0.5), use_operators=False) == 'mul(x, add(y, pow(z, sqrt(y))))'
    assert glsl_code(-x-y, use_operators=False, zero='zero()') == 'sub(zero(), add(x, y))'
    assert glsl_code(-x-y, use_operators=False) == 'sub(0.0, add(x, y))'

# 测试平方根函数的 GLSL 代码生成
def test_glsl_code_sqrt():
    assert glsl_code(sqrt(x)) == "sqrt(x)"
    assert glsl_code(x**0.5) == "sqrt(x)"
    assert glsl_code(sqrt(x)) == "sqrt(x)"

# 测试幂函数的 GLSL 代码生成
def test_glsl_code_Pow():
    g = implemented_function('g', Lambda(x, 2*x))
    assert glsl_code(x**3) == "pow(x, 3.0)"
    assert glsl_code(x**(y**3)) == "pow(x, pow(y, 3.0))"
    assert glsl_code(1/(g(x)*3.5)**(x - y**x)/(x**2 + y)) == \
        "pow(3.5*2*x, -x + pow(y, x))/(pow(x, 2.0) + y)"
    assert glsl_code(x**-1.0) == '1.0/x'

# 测试关系运算符的 GLSL 代码生成
def test_glsl_code_Relational():
    assert glsl_code(Eq(x, y)) == "x == y"
    assert glsl_code(Ne(x, y)) == "x != y"
    assert glsl_code(Le(x, y)) == "x <= y"
    assert glsl_code(Lt(x, y)) == "x < y"
    assert glsl_code(Gt(x, y)) == "x > y"
    assert glsl_code(Ge(x, y)) == "x >= y"

# 测试数学常数的 GLSL 代码生成
def test_glsl_code_constants_mathh():
    assert glsl_code(exp(1)) == "float E = 2.71828183;\nE"
    assert glsl_code(pi) == "float pi = 3.14159265;\npi"
    # assert glsl_code(oo) == "Number.POSITIVE_INFINITY"
    # assert glsl_code(-oo) == "Number.NEGATIVE_INFINITY"

# 测试其他常数的 GLSL 代码生成
def test_glsl_code_constants_other():
    assert glsl_code(2*GoldenRatio) == "float GoldenRatio = 1.61803399;\n2*GoldenRatio"
    assert glsl_code(2*Catalan) == "float Catalan = 0.915965594;\n2*Catalan"
    assert glsl_code(2*EulerGamma) == "float EulerGamma = 0.577215665;\n2*EulerGamma"

# 测试有理数的 GLSL 代码生成
def test_glsl_code_Rational():
    assert glsl_code(Rational(3, 7)) == "3.0/7.0"
    assert glsl_code(Rational(18, 9)) == "2"
    assert glsl_code(Rational(3, -7)) == "-3.0/7.0"
    assert glsl_code(Rational(-3, -7)) == "3.0/7.0"

# 测试整数的 GLSL 代码生成
def test_glsl_code_Integer():
    assert glsl_code(Integer(67)) == "67"
    assert glsl_code(Integer(-1)) == "-1"
def test_glsl_code_functions():
    # 检查 glsl_code 函数是否能正确转换 sin(x) ** cos(x) 表达式为 GLSL 代码
    assert glsl_code(sin(x) ** cos(x)) == "pow(sin(x), cos(x))"


def test_glsl_code_inline_function():
    # 定义符号变量 x
    x = symbols('x')
    # 创建一个实现函数 g(x) = 2*x 的 Lambda 函数对象
    g = implemented_function('g', Lambda(x, 2*x))
    # 检查 glsl_code 函数是否能正确转换 g(x) 表达式为 GLSL 代码 "2*x"
    assert glsl_code(g(x)) == "2*x"
    # 创建一个实现函数 g(x) = 2*x/Catalan 的 Lambda 函数对象
    g = implemented_function('g', Lambda(x, 2*x/Catalan))
    # 检查 glsl_code 函数是否能正确转换 g(x) 表达式为 GLSL 代码 "float Catalan = 0.915965594;\n2*x/Catalan"
    assert glsl_code(g(x)) == "float Catalan = 0.915965594;\n2*x/Catalan"
    # 创建 IndexedBase 对象 A 和整数索引 i
    A = IndexedBase('A')
    i = Idx('i', symbols('n', integer=True))
    # 创建一个实现函数 g(A[i]) = A[i]*(1 + A[i])*(2 + A[i]) 的 Lambda 函数对象
    g = implemented_function('g', Lambda(x, x*(1 + x)*(2 + x)))
    # 检查 glsl_code 函数是否能正确转换 g(A[i]) 表达式为 GLSL 代码，并将其赋值给 A[i]
    assert glsl_code(g(A[i]), assign_to=A[i]) == (
        "for (int i=0; i<n; i++){\n"
        "   A[i] = (A[i] + 1)*(A[i] + 2)*A[i];\n"
        "}"
    )


def test_glsl_code_exceptions():
    # 检查 glsl_code 函数是否能正确转换 ceiling(x) 表达式为 GLSL 代码 "ceil(x)"
    assert glsl_code(ceiling(x)) == "ceil(x)"
    # 检查 glsl_code 函数是否能正确转换 Abs(x) 表达式为 GLSL 代码 "abs(x)" 
    assert glsl_code(Abs(x)) == "abs(x)"


def test_glsl_code_boolean():
    # 检查 glsl_code 函数是否能正确转换 x & y 表达式为 GLSL 代码 "x && y"
    assert glsl_code(x & y) == "x && y"
    # 检查 glsl_code 函数是否能正确转换 x | y 表达式为 GLSL 代码 "x || y"
    assert glsl_code(x | y) == "x || y"
    # 检查 glsl_code 函数是否能正确转换 ~x 表达式为 GLSL 代码 "!x"
    assert glsl_code(~x) == "!x"
    # 检查 glsl_code 函数是否能正确转换 x & y & z 表达式为 GLSL 代码 "x && y && z"
    assert glsl_code(x & y & z) == "x && y && z"
    # 检查 glsl_code 函数是否能正确转换 x | y | z 表达式为 GLSL 代码 "x || y || z"
    assert glsl_code(x | y | z) == "x || y || z"
    # 检查 glsl_code 函数是否能正确转换 (x & y) | z 表达式为 GLSL 代码 "z || x && y"
    assert glsl_code((x & y) | z) == "z || x && y"
    # 检查 glsl_code 函数是否能正确转换 (x | y) & z 表达式为 GLSL 代码 "z && (x || y)"
    assert glsl_code((x | y) & z) == "z && (x || y)"


def test_glsl_code_Piecewise():
    # 创建 Piecewise 表达式
    expr = Piecewise((x, x < 1), (x**2, True))
    # 检查 glsl_code 函数是否能正确转换 expr 表达式为 GLSL 代码
    p = glsl_code(expr)
    # 预期的 GLSL 代码字符串
    s = \
"""\
((x < 1) ? (
   x
)
: (
   pow(x, 2.0)
))\
"""
    # 检查生成的 GLSL 代码是否与预期相符
    assert p == s
    # 检查 glsl_code 函数是否能正确转换 expr 表达式为 GLSL 代码，并将其赋值给变量 c
    assert glsl_code(expr, assign_to="c") == (
    "if (x < 1) {\n"
    "   c = x;\n"
    "}\n"
    "else {\n"
    "   c = pow(x, 2.0);\n"
    "}")
    # 检查 Piecewise 表达式没有默认条件时，glsl_code 函数是否会引发 ValueError 异常
    expr = Piecewise((x, x < 1), (x**2, x > 1), (sin(x), x > 0))
    raises(ValueError, lambda: glsl_code(expr))


def test_glsl_code_Piecewise_deep():
    # 创建 Piecewise 表达式乘以常数 2
    p = glsl_code(2*Piecewise((x, x < 1), (x**2, True)))
    # 预期的 GLSL 代码字符串
    s = \
"""\
2*((x < 1) ? (
   x
)
: (
   pow(x, 2.0)
))\
"""
    # 检查生成的 GLSL 代码是否与预期相符
    assert p == s


def test_glsl_code_settings():
    # 检查传递无效 method 参数给 glsl_code 函数是否会引发 TypeError 异常
    raises(TypeError, lambda: glsl_code(sin(x), method="garbage"))


def test_glsl_code_Indexed():
    # 定义整数符号变量 n, m, o
    n, m, o = symbols('n m o', integer=True)
    # 创建整数索引 i, j, k
    i, j, k = Idx('i', n), Idx('j', m), Idx('k', o)
    # 创建 GLSLPrinter 对象 p，并初始化属性 _not_c
    p = GLSLPrinter()
    p._not_c = set()

    # 创建 IndexedBase 对象 x，验证打印 x[j] 的 GLSL 代码
    x = IndexedBase('x')[j]
    assert p._print_Indexed(x) == 'x[j]'
    # 创建 IndexedBase 对象 A，验证打印 A[i, j] 的 GLSL 代码
    A = IndexedBase('A')[i, j]
    assert p._print_Indexed(A) == 'A[%s]' % (m*i+j)
    # 创建 IndexedBase 对象 B，验证打印 B[i, j, k] 的 GLSL 代码
    B = IndexedBase('B')[i, j, k]
    assert p._print_Indexed(B) == 'B[%s]' % (i*o*m+j*o+k)

    # 检查属性 _not_c 是否为空集合
    assert p._not_c == set()


def test_glsl_code_list_tuple_Tuple():
    # 检查 glsl_code 函数是否能正确转换列表 [1,2,3,4] 表达式为 GLSL 代码 'vec4(1, 2, 3, 4)'
    assert glsl_code([1,2,3,4]) == 'vec4(1, 2, 3, 4)'
    # 检查 glsl_code 函数是否能正确转换列表 [1,2,3] 表达式为 GLSL 代码 'float[3](1, 2, 3)'
    assert glsl_code([1,2,3],glsl_types=False) == 'float[3](1, 2, 3)'
    # 检查 glsl_code 函数是否能正确转换列表 [1,2,3] 表达式为 GLSL 代码，并与元组 (1,2,3) 的 GLSL 代码比较
    assert glsl_code([1,2,3]) == glsl_code((1,2,3))
    # 检查 glsl_code 函数是否能正确转换列表 [1,2,3] 表达式为 GLSL 代码，并与 Tuple(1,2,
    # 定义一个字符串 `s`，其中包含两个嵌套的循环，用于计算矩阵乘法 `A*x` 的结果并赋值给 `y`
    s = (
        'for (int i=0; i<m; i++){\n'  # 外层循环，遍历 `y` 向量
        '   y[i] = 0.0;\n'            # 初始化 `y[i]` 为 0
        '}\n'
        'for (int i=0; i<m; i++){\n'  # 内层循环，遍历矩阵 `A` 的行
        '   for (int j=0; j<n; j++){\n'  # 内层循环，遍历矩阵 `A` 的列
        '      y[i] = A[n*i + j]*x[j] + y[i];\n'  # 计算 `A[i][j]*x[j]` 并累加到 `y[i]`
        '   }\n'
        '}'  # 内外层循环结束
    )
    
    # 调用 `glsl_code` 函数，生成 GLSL 代码，计算 `A[i, j]*x[j]` 并赋值给 `y[i]`
    c = glsl_code(A[i, j]*x[j], assign_to=y[i])
    # 断言生成的 GLSL 代码 `c` 等于预期的字符串 `s`
    assert c == s
# 定义一个测试函数，生成一个简单的循环代码块，并验证生成的 GLSL 代码是否符合预期
def test_dummy_loops():
    # 定义符号变量 i 和 m，并将它们声明为虚拟符号
    i, m = symbols('i m', integer=True, cls=Dummy)
    # 创建 IndexedBase 对象 x 和 y，用于表示数组 x 和 y
    x = IndexedBase('x')
    y = IndexedBase('y')
    # 将 i 封装成一个索引对象 Idx
    i = Idx(i, m)

    # 构造预期的 GLSL 循环代码，通过字符串格式化填充占位符 %(icount)i 和 %(mcount)i
    expected = (
        'for (int i_%(icount)i=0; i_%(icount)i<m_%(mcount)i; i_%(icount)i++){\n'
        '   y[i_%(icount)i] = x[i_%(icount)i];\n'
        '}'
    ) % {'icount': i.label.dummy_index, 'mcount': m.dummy_index}
    # 生成 GLSL 代码，将 x[i] 赋值给 y[i]
    code = glsl_code(x[i], assign_to=y[i])
    # 验证生成的 GLSL 代码是否等于预期的代码块
    assert code == expected


# 定义一个测试函数，生成包含加法操作的 GLSL 循环代码块，并验证其生成是否正确
def test_glsl_code_loops_add():
    # 定义符号变量 n 和 m，并声明为整数符号
    n, m = symbols('n m', integer=True)
    # 创建 IndexedBase 对象 A、x、y 和 z，用于表示数组 A、x、y 和 z
    A = IndexedBase('A')
    x = IndexedBase('x')
    y = IndexedBase('y')
    z = IndexedBase('z')
    # 将 i 和 j 封装成索引对象 Idx
    i = Idx('i', m)
    j = Idx('j', n)

    # 构造预期的 GLSL 循环代码，包含一个简单的加法操作和嵌套循环
    s = (
        'for (int i=0; i<m; i++){\n'
        '   y[i] = x[i] + z[i];\n'
        '}\n'
        'for (int i=0; i<m; i++){\n'
        '   for (int j=0; j<n; j++){\n'
        '      y[i] = A[n*i + j]*x[j] + y[i];\n'
        '   }\n'
        '}'
    )
    # 生成 GLSL 代码，计算 A[i, j]*x[j] + x[i] + z[i] 并将结果赋值给 y[i]
    c = glsl_code(A[i, j]*x[j] + x[i] + z[i], assign_to=y[i])
    # 验证生成的 GLSL 代码是否等于预期的代码块
    assert c == s


# 定义一个测试函数，生成包含多重收缩操作的 GLSL 循环代码块，并验证其生成是否正确
def test_glsl_code_loops_multiple_contractions():
    # 定义符号变量 n、m、o 和 p，并声明为整数符号
    n, m, o, p = symbols('n m o p', integer=True)
    # 创建 IndexedBase 对象 a、b 和 y，用于表示数组 a、b 和 y
    a = IndexedBase('a')
    b = IndexedBase('b')
    y = IndexedBase('y')
    # 将 i、j、k 和 l 封装成索引对象 Idx
    i = Idx('i', m)
    j = Idx('j', n)
    k = Idx('k', o)
    l = Idx('l', p)

    # 构造预期的 GLSL 循环代码，包含多重嵌套的循环和乘法操作
    s = (
        'for (int i=0; i<m; i++){\n'
        '   y[i] = 0.0;\n'
        '}\n'
        'for (int i=0; i<m; i++){\n'
        '   for (int j=0; j<n; j++){\n'
        '      for (int k=0; k<o; k++){\n'
        '         for (int l=0; l<p; l++){\n'
        '            y[i] = a[%s]*b[%s] + y[i];\n' % (i*n*o*p + j*o*p + k*p + l, j*o*p + k*p + l) +\
        '         }\n'
        '      }\n'
        '   }\n'
        '}'
    )
    # 生成 GLSL 代码，计算 b[j, k, l]*a[i, j, k, l] 并将结果累加到 y[i] 中
    c = glsl_code(b[j, k, l]*a[i, j, k, l], assign_to=y[i])
    # 验证生成的 GLSL 代码是否等于预期的代码块
    assert c == s


# 定义一个测试函数，生成包含多项式操作的 GLSL 循环代码块，并验证其生成是否正确
def test_glsl_code_loops_addfactor():
    # 定义符号变量 n、m、o 和 p，并声明为整数符号
    n, m, o, p = symbols('n m o p', integer=True)
    # 创建 IndexedBase 对象 a、b、c 和 y，用于表示数组 a、b、c 和 y
    a = IndexedBase('a')
    b = IndexedBase('b')
    c = IndexedBase('c')
    y = IndexedBase('y')
    # 将 i、j、k 和 l 封装成索引对象 Idx
    i = Idx('i', m)
    j = Idx('j', n)
    k = Idx('k', o)
    l = Idx('l', p)

    # 构造预期的 GLSL 循环代码，包含多重嵌套的循环、加法和乘法操作
    s = (
        'for (int i=0; i<m; i++){\n'
        '   y[i] = 0.0;\n'
        '}\n'
        'for (int i=0; i<m; i++){\n'
        '   for (int j=0; j<n; j++){\n'
        '      for (int k=0; k<o; k++){\n'
        '         for (int l=0; l<p; l++){\n'
        '            y[i] = (a[%s] + b[%s])*c[%s] + y[i];\n' % (i*n*o*p + j*o*p + k*p + l, i*n*o*p + j*o*p + k*p + l, j*o*p + k*p + l) +\
        '         }\n'
        '      }\n'
        '   }\n'
        '}'
    )
    # 生成 GLSL 代码，计算 (a[i, j, k, l] + b[i, j, k, l])*c[j, k, l] 并将结果累加到 y[i] 中
    c = glsl_code((a[i, j, k, l] + b[i, j, k, l])*c[j, k, l], assign_to=y[i])
    # 验证生成的 GLSL 代码是否等于预期的代码块
    assert c == s


# 定义一个测试函数，生成包含多项式和乘法操作的 GLSL 循环代码块，并验证其生成是否正确
def test_glsl_code_loops_multiple_terms():
    # 定义符号变量 n、m、o 和 p，并声明为整数符号
    n, m, o, p = symbols('n m o p', integer=True)
    # 创建 IndexedBase 对象 a、b、c 和 y，用于表示数组 a、b、c 和 y
    a = IndexedBase('a')
    b = IndexedBase('b')
    c = IndexedBase('c')
    y = IndexedBase('y')
    # 将 i、j 和 k 封装成索引对象 Idx
    i = Idx('i', m)
    j = Idx('j', n)
    k = Idx('k', o)

    # 部分预期的 GLSL 循环代码，初始化 y[i] 为 0.0
    s0 = (
        'for (int
    # 构建字符串 s1，表示三层嵌套的 C 语言风格的循环，用于计算 y[i]
    s1 = (
        'for (int i=0; i<m; i++){\n'
        '   for (int j=0; j<n; j++){\n'
        '      for (int k=0; k<o; k++){\n'
        '         y[i] = b[j]*b[k]*c[%s] + y[i];\n' % (i*n*o + j*o + k) +
        '      }\n'
        '   }\n'
        '}\n'
    )
    
    # 构建字符串 s2，表示两层嵌套的 C 语言风格的循环，用于计算 y[i]
    s2 = (
        'for (int i=0; i<m; i++){\n'
        '   for (int k=0; k<o; k++){\n'
        '      y[i] = a[%s]*b[k] + y[i];\n' % (i*o + k) +
        '   }\n'
        '}\n'
    )
    
    # 构建字符串 s3，表示两层嵌套的 C 语言风格的循环，用于计算 y[i]
    s3 = (
        'for (int i=0; i<m; i++){\n'
        '   for (int j=0; j<n; j++){\n'
        '      y[i] = a[%s]*b[j] + y[i];\n' % (i*n + j) +
        '   }\n'
        '}\n'
    )
    
    # 调用 glsl_code 函数生成 GLSL 代码，并将其赋值给变量 c
    c = glsl_code(
        # 构造 GLSL 代码，涉及 b[j]*a[i, j] + b[k]*a[i, k] + b[j]*b[k]*c[i, j, k]，将结果赋给 y[i]
        b[j]*a[i, j] + b[k]*a[i, k] + b[j]*b[k]*c[i, j, k], assign_to=y[i]
    )
    
    # 使用断言验证生成的 GLSL 代码是否等于预期的几种组合
    assert (c == s0 + s1 + s2 + s3[:-1] or
            c == s0 + s1 + s3 + s2[:-1] or
            c == s0 + s2 + s1 + s3[:-1] or
            c == s0 + s2 + s3 + s1[:-1] or
            c == s0 + s3 + s1 + s2[:-1] or
            c == s0 + s3 + s2 + s1[:-1])
# 定义一个测试函数，用于测试矩阵的打印输出
def test_Matrix_printing():
    # 创建一个包含特定表达式的矩阵
    mat = Matrix([x*y, Piecewise((2 + x, y>0), (y, True)), sin(z)])
    # 创建一个符号矩阵 A，大小为 3x1
    A = MatrixSymbol('A', 3, 1)
    # 断言生成的 GLSL 代码与预期的输出相等
    assert glsl_code(mat, assign_to=A) == (
'''A[0][0] = x*y;
if (y > 0) {
   A[1][0] = x + 2;
}
else {
   A[1][0] = y;
}
A[2][0] = sin(z);''' )
    # 断言生成的 GLSL 代码与预期的输出相等
    assert glsl_code(Matrix([A[0],A[1]]))
    
    # 测试在表达式中使用矩阵元素
    expr = Piecewise((2*A[2, 0], x > 0), (A[2, 0], True)) + sin(A[1, 0]) + A[0, 0]
    # 断言生成的 GLSL 代码与预期的输出相等
    assert glsl_code(expr) == (
'''((x > 0) ? (
   2*A[2][0]
)
: (
   A[2][0]
)) + sin(A[1][0]) + A[0][0]''' )

    # 测试在矩阵中使用矩阵元素
    q = MatrixSymbol('q', 5, 1)
    M = MatrixSymbol('M', 3, 3)
    m = Matrix([[sin(q[1,0]), 0, cos(q[2,0])],
        [q[1,0] + q[2,0], q[3, 0], 5],
        [2*q[4, 0]/q[1,0], sqrt(q[0,0]) + 4, 0]])
    # 断言生成的 GLSL 代码与预期的输出相等
    assert glsl_code(m,M) == (
'''M[0][0] = sin(q[1]);
M[0][1] = 0;
M[0][2] = cos(q[2]);
M[1][0] = q[1] + q[2];
M[1][1] = q[3];
M[1][2] = 5;
M[2][0] = 2*q[4]/q[1];
M[2][1] = sqrt(q[0]) + 4;
M[2][2] = 0;'''
        )

# 定义测试函数，测试大小为 1x7 的矩阵生成 GLSL 代码
def test_Matrices_1x7():
    gl = glsl_code
    A = Matrix([1,2,3,4,5,6,7])
    # 断言生成的 GLSL 代码与预期的输出相等
    assert gl(A) == 'float[7](1, 2, 3, 4, 5, 6, 7)'
    # 断言生成的 GLSL 代码与预期的输出相等
    assert gl(A.transpose()) == 'float[7](1, 2, 3, 4, 5, 6, 7)'

# 定义测试函数，测试大小为 1x7 的矩阵生成 GLSL 代码（整型数组类型）
def test_Matrices_1x7_array_type_int():
    gl = glsl_code
    A = Matrix([1,2,3,4,5,6,7])
    # 断言生成的 GLSL 代码与预期的输出相等
    assert gl(A, array_type='int') == 'int[7](1, 2, 3, 4, 5, 6, 7)'

# 定义测试函数，测试使用自定义类型名称的元组数组生成 GLSL 代码
def test_Tuple_array_type_custom():
    gl = glsl_code
    A = symbols('a b c')
    # 断言生成的 GLSL 代码与预期的输出相等
    assert gl(A, array_type='AbcType', glsl_types=False) == 'AbcType[3](a, b, c)'

# 定义测试函数，测试将 1x7 的矩阵元素分配给嵌套符号的情况
def test_Matrices_1x7_spread_assign_to_symbols():
    gl = glsl_code
    A = Matrix([1,2,3,4,5,6,7])
    assign_to = symbols('x.a x.b x.c x.d x.e x.f x.g')
    # 断言生成的 GLSL 代码与预期的输出相等
    assert gl(A, assign_to=assign_to) == textwrap.dedent('''\
        x.a = 1;
        x.b = 2;
        x.c = 3;
        x.d = 4;
        x.e = 5;
        x.f = 6;
        x.g = 7;'''
    )

# 定义测试函数，测试将元组数组分配给嵌套符号的情况
def test_spread_assign_to_nested_symbols():
    gl = glsl_code
    expr = ((1,2,3), (1,2,3))
    assign_to = (symbols('a b c'), symbols('x y z'))
    # 断言生成的 GLSL 代码与预期的输出相等
    assert gl(expr, assign_to=assign_to) == textwrap.dedent('''\
        a = 1;
        b = 2;
        c = 3;
        x = 1;
        y = 2;
        z = 3;'''
    )

# 定义测试函数，测试将深度嵌套的元组分配给嵌套符号的情况
def test_spread_assign_to_deeply_nested_symbols():
    gl = glsl_code
    a, b, c, x, y, z = symbols('a b c x y z')
    expr = (((1,2),3), ((1,2),3))
    assign_to = (((a, b), c), ((x, y), z))
    # 断言生成的 GLSL 代码与预期的输出相等
    assert gl(expr, assign_to=assign_to) == textwrap.dedent('''\
        a = 1;
        b = 2;
        c = 3;
        x = 1;
        y = 2;
        z = 3;'''
    )

# 定义测试函数，测试将元组矩阵分配给多层嵌套符号的情况
def test_matrix_of_tuples_spread_assign_to_symbols():
    gl = glsl_code
    with warns_deprecated_sympy():
        expr = Matrix([[(1,2),(3,4)],[(5,6),(7,8)]])
    assign_to = (symbols('a b'), symbols('c d'), symbols('e f'), symbols('g h'))
    # 使用 assert 语句来验证调用 gl 函数后的返回结果是否等于预期值，这里使用 textwrap.dedent 来移除代码块的缩进
    assert gl(expr, assign_to) == textwrap.dedent('''\
        a = 1;
        b = 2;
        c = 3;
        d = 4;
        e = 5;
        f = 6;
        g = 7;
        h = 8;'''
    )
# 定义一个测试函数，用于测试在长度不匹配时无法分配值的情况
def test_cannot_assign_to_cause_mismatched_length():
    # 定义一个元组表达式
    expr = (1, 2)
    # 创建符号变量 x y z
    assign_to = symbols('x y z')
    # 断言在使用 glsl_code 函数处理 expr 分配到 assign_to 时会引发 ValueError 异常
    raises(ValueError, lambda: glsl_code(expr, assign_to))

# 定义一个测试函数，测试 4x4 矩阵赋值情况
def test_matrix_4x4_assign():
    # 将 glsl_code 函数赋值给 gl 变量
    gl = glsl_code
    # 定义一个矩阵表达式 A * B + C，其中 A、B、C 均为 4x4 矩阵符号
    expr = MatrixSymbol('A', 4, 4) * MatrixSymbol('B', 4, 4) + MatrixSymbol('C', 4, 4)
    # 定义一个 4x4 矩阵符号 X 作为赋值目标
    assign_to = MatrixSymbol('X', 4, 4)
    # 断言 gl 函数处理 expr 并分配到 assign_to 后的结果等于预期的多行字符串
    assert gl(expr, assign_to=assign_to) == textwrap.dedent('''\
        X[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0] + A[0][2]*B[2][0] + A[0][3]*B[3][0] + C[0][0];
        X[0][1] = A[0][0]*B[0][1] + A[0][1]*B[1][1] + A[0][2]*B[2][1] + A[0][3]*B[3][1] + C[0][1];
        X[0][2] = A[0][0]*B[0][2] + A[0][1]*B[1][2] + A[0][2]*B[2][2] + A[0][3]*B[3][2] + C[0][2];
        X[0][3] = A[0][0]*B[0][3] + A[0][1]*B[1][3] + A[0][2]*B[2][3] + A[0][3]*B[3][3] + C[0][3];
        X[1][0] = A[1][0]*B[0][0] + A[1][1]*B[1][0] + A[1][2]*B[2][0] + A[1][3]*B[3][0] + C[1][0];
        X[1][1] = A[1][0]*B[0][1] + A[1][1]*B[1][1] + A[1][2]*B[2][1] + A[1][3]*B[3][1] + C[1][1];
        X[1][2] = A[1][0]*B[0][2] + A[1][1]*B[1][2] + A[1][2]*B[2][2] + A[1][3]*B[3][2] + C[1][2];
        X[1][3] = A[1][0]*B[0][3] + A[1][1]*B[1][3] + A[1][2]*B[2][3] + A[1][3]*B[3][3] + C[1][3];
        X[2][0] = A[2][0]*B[0][0] + A[2][1]*B[1][0] + A[2][2]*B[2][0] + A[2][3]*B[3][0] + C[2][0];
        X[2][1] = A[2][0]*B[0][1] + A[2][1]*B[1][1] + A[2][2]*B[2][1] + A[2][3]*B[3][1] + C[2][1];
        X[2][2] = A[2][0]*B[0][2] + A[2][1]*B[1][2] + A[2][2]*B[2][2] + A[2][3]*B[3][2] + C[2][2];
        X[2][3] = A[2][0]*B[0][3] + A[2][1]*B[1][3] + A[2][2]*B[2][3] + A[2][3]*B[3][3] + C[2][3];
        X[3][0] = A[3][0]*B[0][0] + A[3][1]*B[1][0] + A[3][2]*B[2][0] + A[3][3]*B[3][0] + C[3][0];
        X[3][1] = A[3][0]*B[0][1] + A[3][1]*B[1][1] + A[3][2]*B[2][1] + A[3][3]*B[3][1] + C[3][1];
        X[3][2] = A[3][0]*B[0][2] + A[3][1]*B[1][2] + A[3][2]*B[2][2] + A[3][3]*B[3][2] + C[3][2];
        X[3][3] = A[3][0]*B[0][3] + A[3][1]*B[1][3] + A[3][2]*B[2][3] + A[3][3]*B[3][3] + C[3][3];'''
    )

# 定义一个测试函数，测试 1xN 向量的情况
def test_1xN_vecs():
    # 将 glsl_code 函数赋值给 gl 变量
    gl = glsl_code
    # 对于每个 1 到 9 的范围内的数值 i
    for i in range(1, 10):
        # 创建一个行向量 A，其长度为 i
        A = Matrix(range(i))
        # 断言 gl 函数处理 A 的转置与 A 自身的结果相同
        assert gl(A.transpose()) == gl(A)
        # 断言 gl 函数处理 A 并进行矩阵转置后的结果与处理 A 自身的结果相同
        assert gl(A, mat_transpose=True) == gl(A)
        # 如果 i 大于 1
        if i > 1:
            # 如果 i 小于等于 4
            if i <= 4:
                # 断言 gl 函数处理 A 的结果等于格式化的字符串 'vec%s(%s)'
                assert gl(A) == 'vec%s(%s)' % (i, ', '.join(str(s) for s in range(i)))
            else:
                # 断言 gl 函数处理 A 的结果等于格式化的字符串 'float[%s](%s)'
                assert gl(A) == 'float[%s](%s)' % (i, ', '.join(str(s) for s in range(i)))

# 定义一个测试函数，测试 MxN 矩阵的情况
def test_MxN_mats():
    # 创建一个字符串，包含待生成的测试函数代码
    generatedAssertions = 'def test_misc_mats():\n'
    # 循环创建矩阵和相关的 GLSL 代码，生成断言语句
    for i in range(1,6):
        for j in range(1,6):
            # 根据循环变量 i 和 j 创建 Matrix 对象 A，A 中的元素为 x + y*j，其中 x 从 0 到 j-1，y 从 0 到 i-1
            A = Matrix([[x + y*j for x in range(j)] for y in range(i)])
            # 生成 A 的 GLSL 代码
            gl = glsl_code(A)
            # 生成 A 的转置后的 GLSL 代码
            glTransposed = glsl_code(A, mat_transpose=True)
            # 将生成的断言语句添加到 generatedAssertions 变量中，验证生成的 GLSL 代码与预期一致
            generatedAssertions+='    mat = '+StrPrinter()._print(A)+'\n\n'
            generatedAssertions+='    gl = \'\'\''+gl+'\'\'\'\n'
            generatedAssertions+='    glTransposed = \'\'\''+glTransposed+'\'\'\'\n\n'
            generatedAssertions+='    assert glsl_code(mat) == gl\n'
            generatedAssertions+='    assert glsl_code(mat,mat_transpose=True) == glTransposed\n'
            # 根据条件生成特定的断言语句，验证生成的 GLSL 代码是否符合预期
            if i == 1 and j == 1:
                assert gl == '0'
            elif i <= 4 and j <= 4 and i > 1 and j > 1:
                assert gl.startswith('mat%s' % j)
                assert glTransposed.startswith('mat%s' % i)
            elif i == 1 and j <= 4:
                assert gl.startswith('vec')
            elif j == 1 and i <= 4:
                assert gl.startswith('vec')
            elif i == 1:
                assert gl.startswith('float[%s](' % (j*i))
                assert glTransposed.startswith('float[%s](' % (j*i))
            elif j == 1:
                assert gl.startswith('float[%s](' % (i*j))
                assert glTransposed.startswith('float[%s](' % (i*j))
            else:
                assert gl.startswith('float[%s](' % (i*j))
                assert glTransposed.startswith('float[%s](' % (i*j))
                # 如果满足特定条件，生成嵌套矩阵的 GLSL 代码，并进行断言验证
                glNested = glsl_code(A, mat_nested=True)
                glNestedTransposed = glsl_code(A, mat_transpose=True, mat_nested=True)
                assert glNested.startswith('float[%s][%s]' % (i,j))
                assert glNestedTransposed.startswith('float[%s][%s]' % (j,i))
                generatedAssertions+='    glNested = \'\'\''+glNested+'\'\'\'\n'
                generatedAssertions+='    glNestedTransposed = \'\'\''+glNestedTransposed+'\'\'\'\n\n'
                generatedAssertions+='    assert glsl_code(mat,mat_nested=True) == glNested\n'
                generatedAssertions+='    assert glsl_code(mat,mat_nested=True,mat_transpose=True) == glNestedTransposed\n\n'
    # 将生成的断言语句写入到文件中（如果 generateAssertions 为 True）
    generateAssertions = False # set this to true to write bake these generated tests to a file
    if generateAssertions:
        gen = open('test_glsl_generated_matrices.py','w')
        gen.write(generatedAssertions)
        gen.close()
# 这些断言是从前一个函数生成的，用于测试不同的矩阵情况
# GLSL（OpenGL着色语言）有复杂的规则，这些测试用例可以方便地检查所有情况

def test_misc_mats():

    # 创建一个包含单个元素的矩阵
    mat = Matrix([[0]])

    # 创建 GLSL 表示，对于单个元素的矩阵，GLSL 代码为 '0'
    gl = '''0'''
    glTransposed = '''0'''

    # 断言：调用 glsl_code 函数并检查结果是否与预期相符
    assert glsl_code(mat) == gl
    assert glsl_code(mat, mat_transpose=True) == glTransposed

    # 创建一个包含两个元素的矩阵
    mat = Matrix([[0, 1]])

    # 创建 GLSL 表示，对于二维向量，GLSL 代码为 'vec2(0, 1)'
    gl = '''vec2(0, 1)'''
    glTransposed = '''vec2(0, 1)'''

    # 断言：调用 glsl_code 函数并检查结果是否与预期相符
    assert glsl_code(mat) == gl
    assert glsl_code(mat, mat_transpose=True) == glTransposed

    # 创建一个包含三个元素的矩阵
    mat = Matrix([[0, 1, 2]])

    # 创建 GLSL 表示，对于三维向量，GLSL 代码为 'vec3(0, 1, 2)'
    gl = '''vec3(0, 1, 2)'''
    glTransposed = '''vec3(0, 1, 2)'''

    # 断言：调用 glsl_code 函数并检查结果是否与预期相符
    assert glsl_code(mat) == gl
    assert glsl_code(mat, mat_transpose=True) == glTransposed

    # 创建一个包含四个元素的矩阵
    mat = Matrix([[0, 1, 2, 3]])

    # 创建 GLSL 表示，对于四维向量，GLSL 代码为 'vec4(0, 1, 2, 3)'
    gl = '''vec4(0, 1, 2, 3)'''
    glTransposed = '''vec4(0, 1, 2, 3)'''

    # 断言：调用 glsl_code 函数并检查结果是否与预期相符
    assert glsl_code(mat) == gl
    assert glsl_code(mat, mat_transpose=True) == glTransposed

    # 创建一个包含五个元素的矩阵
    mat = Matrix([[0, 1, 2, 3, 4]])

    # 创建 GLSL 表示，对于大小为五的一维数组，GLSL 代码为 'float[5](0, 1, 2, 3, 4)'
    gl = '''float[5](0, 1, 2, 3, 4)'''
    glTransposed = '''float[5](0, 1, 2, 3, 4)'''

    # 断言：调用 glsl_code 函数并检查结果是否与预期相符
    assert glsl_code(mat) == gl
    assert glsl_code(mat, mat_transpose=True) == glTransposed

    # 创建一个包含两行一列的矩阵
    mat = Matrix([[0],
                  [1]])

    # 创建 GLSL 表示，对于二维向量，GLSL 代码为 'vec2(0, 1)'
    gl = '''vec2(0, 1)'''
    glTransposed = '''vec2(0, 1)'''

    # 断言：调用 glsl_code 函数并检查结果是否与预期相符
    assert glsl_code(mat) == gl
    assert glsl_code(mat, mat_transpose=True) == glTransposed

    # 创建一个包含两行两列的矩阵
    mat = Matrix([[0, 1],
                  [2, 3]])

    # 创建 GLSL 表示，对于 2x2 矩阵，GLSL 代码为 'mat2(0, 1, 2, 3)'
    gl = '''mat2(0, 1, 2, 3)'''
    glTransposed = '''mat2(0, 2, 1, 3)'''

    # 断言：调用 glsl_code 函数并检查结果是否与预期相符
    assert glsl_code(mat) == gl
    assert glsl_code(mat, mat_transpose=True) == glTransposed

    # 创建一个包含两行三列的矩阵
    mat = Matrix([[0, 1, 2],
                  [3, 4, 5]])

    # 创建 GLSL 表示，对于 3x2 矩阵，GLSL 代码为 'mat3x2(0, 1, 2, 3, 4, 5)'
    gl = '''mat3x2(0, 1, 2, 3, 4, 5)'''
    glTransposed = '''mat2x3(0, 3, 1, 4, 2, 5)'''

    # 断言：调用 glsl_code 函数并检查结果是否与预期相符
    assert glsl_code(mat) == gl
    assert glsl_code(mat, mat_transpose=True) == glTransposed

    # 创建一个包含两行四列的矩阵
    mat = Matrix([[0, 1, 2, 3],
                  [4, 5, 6, 7]])

    # 创建 GLSL 表示，对于 4x2 矩阵，GLSL 代码为 'mat4x2(0, 1, 2, 3, 4, 5, 6, 7)'
    gl = '''mat4x2(0, 1, 2, 3, 4, 5, 6, 7)'''
    glTransposed = '''mat2x4(0, 4, 1, 5, 2, 6, 3, 7)'''

    # 断言：调用 glsl_code 函数并检查结果是否与预期相符
    assert glsl_code(mat) == gl
    assert glsl_code(mat, mat_transpose=True) == glTransposed

    # 创建一个包含两行五列的矩阵
    mat = Matrix([[0, 1, 2, 3, 4],
                  [5, 6, 7, 8, 9]])

    # 创建 GLSL 表示，对于大小为 10 的一维数组，GLSL 代码为 'float[10](0, 1, 2, 3, 4, 5, 6, 7, 8, 9) /* a 2x5 matrix */'
    gl = '''float[10](
   0, 1, 2, 3, 4,
   5, 6, 7, 8, 9
) /* a 2x5 matrix */'''
    glTransposed = '''float[10](
   0, 5,
   1, 6,
   2, 7,
   3, 8,
   4, 9
) /* a 5x2 matrix */'''

    # 断言：调用 glsl_code 函数并检查结果是否与预期相符
    assert glsl_code(mat) == gl
    assert glsl_code(mat, mat_transpose=True) == glTransposed

    # 创建一个包含三行一列的矩阵
    mat = Matrix([[0],
                  [1],
                  [2]])

    # 创建 GLSL 表示，对于三维向量，GLSL 代码为 'vec3(0, 1, 2)'
    gl = '''vec3(0, 1, 2)'''
    glTransposed = '''vec3(0, 1, 2)'''

    # 断言：调用 glsl_code 函数并检查结果是否与预期相符
    assert glsl_code(mat) == gl
    assert glsl_code(mat, mat_transpose=True) == glTransposed

    # 创建一个包含三行两列的矩阵
    mat = Matrix([[0, 1],
                  [2, 3],
                  [4, 5]])

    # 创建 GLSL 表示，对于 2x3 矩阵，GLSL 代码为 'mat2x3(0, 1, 2
    # 定义一个 GLSL 着色器代码中的变换矩阵，这里使用了固定的转置顺序
    glTransposed = '''mat3x2(0, 2, 4, 1, 3, 5)'''

    # 断言：使用默认参数调用 glsl_code 函数，检查返回的 GLSL 代码是否等于 gl 变量的值
    assert glsl_code(mat) == gl

    # 断言：使用 mat_transpose 参数调用 glsl_code 函数，检查返回的 GLSL 代码是否等于 glTransposed 变量的值
    assert glsl_code(mat, mat_transpose=True) == glTransposed

    # 初始化一个 Matrix 对象，并将其赋给 mat 变量
    mat = Matrix([
# 创建一个 Matrix 对象，表示一个3x3的矩阵
mat = Matrix([
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]])

# 生成 GLSL 代码，表示一个3x3的矩阵
gl = '''mat3(0, 1, 2, 3, 4, 5, 6, 7, 8)'''
# 生成 GLSL 代码，表示一个3x3的矩阵的转置
glTransposed = '''mat3(0, 3, 6, 1, 4, 7, 2, 5, 8)'''

# 断言：生成的 GLSL 代码与预期的相符
assert glsl_code(mat) == gl
# 断言：生成的 GLSL 代码与预期的转置相符
assert glsl_code(mat, mat_transpose=True) == glTransposed

# 创建一个新的 Matrix 对象，表示一个3x4的矩阵
mat = Matrix([
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11]])

# 生成 GLSL 代码，表示一个3x4的矩阵
gl = '''mat4x3(0, 1,  2,  3, 4, 5,  6,  7, 8, 9, 10, 11)'''
# 生成 GLSL 代码，表示一个3x4的矩阵的转置
glTransposed = '''mat3x4(0, 4,  8, 1, 5,  9, 2, 6, 10, 3, 7, 11)'''

# 断言：生成的 GLSL 代码与预期的相符
assert glsl_code(mat) == gl
# 断言：生成的 GLSL 代码与预期的转置相符
assert glsl_code(mat, mat_transpose=True) == glTransposed

# 创建一个新的 Matrix 对象，表示一个3x5的矩阵
mat = Matrix([
    [ 0,  1,  2,  3],
    [ 4,  5,  6,  7],
    [ 8,  9, 10, 11]])

# 生成 GLSL 代码，表示一个3x5的矩阵
gl = '''float[15](
   0,  1,  2,  3,  4,
   5,  6,  7,  8,  9,
   10, 11, 12, 13, 14
) /* a 3x5 matrix */'''
# 生成 GLSL 代码，表示一个3x5的矩阵的转置
glTransposed = '''float[15](
   0, 5, 10,
   1, 6, 11,
   2, 7, 12,
   3, 8, 13,
   4, 9, 14
) /* a 5x3 matrix */'''

# 断言：生成的 GLSL 代码与预期的相符
assert glsl_code(mat) == gl
# 断言：生成的 GLSL 代码与预期的转置相符
assert glsl_code(mat, mat_transpose=True) == glTransposed

# 创建一个新的 Matrix 对象，表示一个4x1的矩阵
mat = Matrix([
    [0],
    [1],
    [2],
    [3]])

# 生成 GLSL 代码，表示一个4x1的矩阵
gl = '''vec4(0, 1, 2, 3)'''
# 生成 GLSL 代码，表示一个4x1的矩阵的转置
glTransposed = '''vec4(0, 1, 2, 3)'''

# 断言：生成的 GLSL 代码与预期的相符
assert glsl_code(mat) == gl
# 断言：生成的 GLSL 代码与预期的转置相符
assert glsl_code(mat, mat_transpose=True) == glTransposed

# 创建一个新的 Matrix 对象，表示一个4x2的矩阵
mat = Matrix([
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7]])

# 生成 GLSL 代码，表示一个4x2的矩阵
gl = '''mat2x4(0, 1, 2, 3, 4, 5, 6, 7)'''
# 生成 GLSL 代码，表示一个4x2的矩阵的转置
glTransposed = '''mat4x2(0, 2, 4, 6, 1, 3, 5, 7)'''

# 断言：生成的 GLSL 代码与预期的相符
assert glsl_code(mat) == gl
# 断言：生成的 GLSL 代码与预期的转置相符
assert glsl_code(mat, mat_transpose=True) == glTransposed

# 创建一个新的 Matrix 对象，表示一个4x3的矩阵
mat = Matrix([
    [ 0,  1,  2],
    [ 3,  4,  5],
    [ 6,  7,  8],
    [ 9, 10, 11]])

# 生成 GLSL 代码，表示一个4x3的矩阵
gl = '''mat3x4(0,  1,  2, 3,  4,  5, 6,  7,  8, 9, 10, 11)'''
# 生成 GLSL 代码，表示一个4x3的矩阵的转置
glTransposed = '''mat4x3(0, 3, 6,  9, 1, 4, 7, 10, 2, 5, 8, 11)'''

# 断言：生成的 GLSL 代码与预期的相符
assert glsl_code(mat) == gl
# 断言：生成的 GLSL 代码与预期的转置相符
assert glsl_code(mat, mat_transpose=True) == glTransposed

# 创建一个新的 Matrix 对象，表示一个4x4的矩阵
mat = Matrix([
    [ 0,  1,  2,  3],
    [ 4,  5,  6,  7],
    [ 8,  9, 10, 11],
    [12, 13, 14, 15]])

# 生成 GLSL 代码，表示一个4x4的矩阵
gl = '''mat4( 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15)'''
# 生成 GLSL 代码，表示一个4x4的矩阵的转置
glTransposed = '''mat4(0, 4,  8, 12, 1, 5,  9, 13, 2, 6, 10, 14, 3, 7, 11, 15)'''

# 断言：生成的 GLSL 代码与预期的相符
assert glsl_code(mat) == gl
# 断言：生成的 GLSL 代码与预期的转置相符
assert glsl_code(mat, mat_transpose=True) == glTransposed

# 创建一个新的 Matrix 对象，表示一个4x5的矩阵
mat = Matrix([
    [ 0,  1,  2,  3,  4],
    [ 5,  6,  7,  8,  9],
    [10, 11, 12, 13, 14],
    [15, 16, 17, 18, 19]])

# 生成 GLSL 代码，表示一个4x5的矩阵
gl = '''float[20](
   0,  1,  2,  3,  4,
   5,  6,  7,  8,  9,
   10, 11, 12, 13, 14,
   15, 16, 17, 18, 19
) /* a 4x5 matrix */'''
# 生成 GLSL 代码，表示一个4x5的矩阵
    # 定义一个包含多个行和列的二维数组，每行包含5个浮点数
    glNested = '''float[4][5](
   float[]( 0,  1,  2,  3,  4),  # 第一行数据
   float[]( 5,  6,  7,  8,  9),  # 第二行数据
   float[](10, 11, 12, 13, 14),  # 第三行数据
   float[](15, 16, 17, 18, 19)   # 第四行数据
'''
glNestedTransposed = '''float[5][4](
float[](0, 5, 10, 15),
float[](1, 6, 11, 16),
float[](2, 7, 12, 17),
float[](3, 8, 13, 18),
float[](4, 9, 14, 19)
)'''
'''

'''
assert glsl_code(mat,mat_nested=True) == glNested
assert glsl_code(mat,mat_nested=True,mat_transpose=True) == glNestedTransposed
'''

mat = Matrix([
    [0],
    [1],
    [2],
    [3],
    [4]])

'''
gl = '''float[5](0, 1, 2, 3, 4)'''
glTransposed = '''float[5](0, 1, 2, 3, 4)'''
'''

'''
assert glsl_code(mat) == gl
assert glsl_code(mat,mat_transpose=True) == glTransposed
'''

mat = Matrix([
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9]])

'''
gl = '''float[10](
0, 1,
2, 3,
4, 5,
6, 7,
8, 9
) /* a 5x2 matrix */'''
glTransposed = '''float[10](
0, 2, 4, 6, 8,
1, 3, 5, 7, 9
) /* a 2x5 matrix */'''
'''

'''
assert glsl_code(mat) == gl
assert glsl_code(mat,mat_transpose=True) == glTransposed
glNested = '''float[5][2](
float[](0, 1),
float[](2, 3),
float[](4, 5),
float[](6, 7),
float[](8, 9)
)'''
glNestedTransposed = '''float[2][5](
float[](0, 2, 4, 6, 8),
float[](1, 3, 5, 7, 9)
)'''
'''

'''
assert glsl_code(mat,mat_nested=True) == glNested
assert glsl_code(mat,mat_nested=True,mat_transpose=True) == glNestedTransposed
'''

mat = Matrix([
    [ 0,  1,  2],
    [ 3,  4,  5],
    [ 6,  7,  8],
    [ 9, 10, 11],
    [12, 13, 14]])

'''
gl = '''float[15](
0,  1,  2,
3,  4,  5,
6,  7,  8,
9, 10, 11,
12, 13, 14
) /* a 5x3 matrix */'''
glTransposed = '''float[15](
0, 3, 6,  9, 12,
1, 4, 7, 10, 13,
2, 5, 8, 11, 14
) /* a 3x5 matrix */'''
'''

'''
assert glsl_code(mat) == gl
assert glsl_code(mat,mat_transpose=True) == glTransposed
glNested = '''float[5][3](
float[]( 0,  1,  2),
float[]( 3,  4,  5),
float[]( 6,  7,  8),
float[]( 9, 10, 11),
float[](12, 13, 14)
)'''
glNestedTransposed = '''float[3][5](
float[](0, 3, 6,  9, 12),
float[](1, 4, 7, 10, 13),
float[](2, 5, 8, 11, 14)
)'''
'''

'''
assert glsl_code(mat,mat_nested=True) == glNested
assert glsl_code(mat,mat_nested=True,mat_transpose=True) == glNestedTransposed
'''

mat = Matrix([
    [ 0,  1,  2,  3],
    [ 4,  5,  6,  7],
    [ 8,  9, 10, 11],
    [12, 13, 14, 15],
    [16, 17, 18, 19]])

'''
gl = '''float[20](
0,  1,  2,  3,
4,  5,  6,  7,
8,  9, 10, 11,
12, 13, 14, 15,
16, 17, 18, 19
) /* a 5x4 matrix */'''
glTransposed = '''float[20](
0, 4,  8, 12, 16,
1, 5,  9, 13, 17,
2, 6, 10, 14, 18,
3, 7, 11, 15, 19
) /* a 4x5 matrix */'''
'''

'''
assert glsl_code(mat) == gl
assert glsl_code(mat,mat_transpose=True) == glTransposed
glNested = '''float[5][4](
float[]( 0,  1,  2,  3),
float[]( 4,  5,  6,  7),
float[]( 8,  9, 10, 11),
float[](12, 13, 14, 15),
float[](16, 17, 18, 19)
)'''
glNestedTransposed = '''float[4][5](
float[](0, 4,  8, 12, 16),
float[](1, 5,  9, 13, 17),
float[](2, 6, 10, 14, 18),
float[](3, 7, 11, 15, 19)
)'''
'''

'''
assert glsl_code(mat,mat_nested=True) == glNested
    # 使用断言检查调用函数 glsl_code 的返回值是否等于 glNestedTransposed
    assert glsl_code(mat, mat_nested=True, mat_transpose=True) == glNestedTransposed
    
    # 创建一个 Matrix 对象，使用给定的列表初始化
    mat = Matrix([
# 创建一个包含数字的二维列表，代表一个5x5的矩阵
[
    [ 0,  1,  2,  3,  4],
    [ 5,  6,  7,  8,  9],
    [10, 11, 12, 13, 14],
    [15, 16, 17, 18, 19],
    [20, 21, 22, 23, 24]
]

# 将上述矩阵表示为GLSL代码的字符串形式，注释指出这是一个5x5矩阵
gl = '''float[25](
   0,  1,  2,  3,  4,
   5,  6,  7,  8,  9,
   10, 11, 12, 13, 14,
   15, 16, 17, 18, 19,
   20, 21, 22, 23, 24
) /* a 5x5 matrix */'''

# 将矩阵进行转置后的GLSL代码的字符串形式，注释指出这是一个5x5矩阵的转置
glTransposed = '''float[25](
   0, 5, 10, 15, 20,
   1, 6, 11, 16, 21,
   2, 7, 12, 17, 22,
   3, 8, 13, 18, 23,
   4, 9, 14, 19, 24
) /* a 5x5 matrix */'''

# 使用assert语句验证GLSL代码生成函数生成的GLSL代码与预期的gl相等
assert glsl_code(mat) == gl

# 使用assert语句验证GLSL代码生成函数生成的GLSL代码与预期的glTransposed相等，同时传入mat_transpose=True参数
assert glsl_code(mat, mat_transpose=True) == glTransposed

# 将嵌套列表形式的矩阵表示为GLSL代码的字符串形式，注释指出这是一个5x5嵌套矩阵
glNested = '''float[5][5](
   float[]( 0,  1,  2,  3,  4),
   float[]( 5,  6,  7,  8,  9),
   float[](10, 11, 12, 13, 14),
   float[](15, 16, 17, 18, 19),
   float[](20, 21, 22, 23, 24)
)'''

# 将嵌套列表形式的矩阵进行转置后的GLSL代码的字符串形式，注释指出这是一个5x5嵌套矩阵的转置
glNestedTransposed = '''float[5][5](
   float[](0, 5, 10, 15, 20),
   float[](1, 6, 11, 16, 21),
   float[](2, 7, 12, 17, 22),
   float[](3, 8, 13, 18, 23),
   float[](4, 9, 14, 19, 24)
)'''

# 使用assert语句验证GLSL代码生成函数生成的GLSL代码与预期的glNested相等，同时传入mat_nested=True参数
assert glsl_code(mat, mat_nested=True) == glNested

# 使用assert语句验证GLSL代码生成函数生成的GLSL代码与预期的glNestedTransposed相等，同时传入mat_nested=True和mat_transpose=True参数
assert glsl_code(mat, mat_nested=True, mat_transpose=True) == glNestedTransposed
```