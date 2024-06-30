# `D:\src\scipysrc\sympy\sympy\utilities\tests\test_codegen_julia.py`

```
from io import StringIO  # 导入StringIO类，用于操作内存中的字符串流

from sympy.core import S, symbols, Eq, pi, Catalan, EulerGamma, Function  # 导入SymPy核心模块的相关符号和函数
from sympy.core.relational import Equality  # 导入SymPy关系模块的Equality类
from sympy.functions.elementary.piecewise import Piecewise  # 导入SymPy元素函数模块的Piecewise类
from sympy.matrices import Matrix, MatrixSymbol  # 导入SymPy矩阵模块的Matrix和MatrixSymbol类
from sympy.utilities.codegen import JuliaCodeGen, codegen, make_routine  # 导入SymPy代码生成工具模块的相关函数和类
from sympy.testing.pytest import XFAIL  # 导入SymPy测试模块的XFAIL类
import sympy  # 导入SymPy库

x, y, z = symbols('x,y,z')  # 定义符号变量 x, y, z

# 测试空的Julia代码生成，预期输出为空字符串
def test_empty_jl_code():
    code_gen = JuliaCodeGen()  # 创建Julia代码生成器对象
    output = StringIO()  # 创建内存中的字符串流对象
    code_gen.dump_jl([], output, "file", header=False, empty=False)  # 调用dump_jl方法生成Julia代码
    source = output.getvalue()  # 获取生成的代码字符串
    assert source == ""  # 断言生成的代码为空字符串

# 测试简单的Julia代码生成，生成一个乘法表达式的函数
def test_jl_simple_code():
    name_expr = ("test", (x + y)*z)  # 定义函数名和表达式元组
    result, = codegen(name_expr, "Julia", header=False, empty=False)  # 生成Julia代码
    assert result[0] == "test.jl"  # 断言生成的文件名为"test.jl"
    source = result[1]  # 获取生成的代码字符串
    expected = (
        "function test(x, y, z)\n"  # 预期的Julia函数定义
        "    out1 = z .* (x + y)\n"
        "    return out1\n"
        "end\n"
    )
    assert source == expected  # 断言生成的代码与预期一致

# 测试带有头部信息的简单Julia代码生成
def test_jl_simple_code_with_header():
    name_expr = ("test", (x + y)*z)  # 定义函数名和表达式元组
    result, = codegen(name_expr, "Julia", header=True, empty=False)  # 生成Julia代码包含头部信息
    assert result[0] == "test.jl"  # 断言生成的文件名为"test.jl"
    source = result[1]  # 获取生成的代码字符串
    expected = (
        "#   Code generated with SymPy " + sympy.__version__ + "\n"  # 预期的头部信息
        "#\n"
        "#   See http://www.sympy.org/ for more information.\n"
        "#\n"
        "#   This file is part of 'project'\n"
        "function test(x, y, z)\n"  # 预期的Julia函数定义
        "    out1 = z .* (x + y)\n"
        "    return out1\n"
        "end\n"
    )
    assert source == expected  # 断言生成的代码与预期一致

# 测试带有赋值表达式的简单Julia代码生成
def test_jl_simple_code_nameout():
    expr = Equality(z, (x + y))  # 定义等式表达式
    name_expr = ("test", expr)  # 定义函数名和表达式元组
    result, = codegen(name_expr, "Julia", header=False, empty=False)  # 生成Julia代码
    source = result[1]  # 获取生成的代码字符串
    expected = (
        "function test(x, y)\n"  # 预期的Julia函数定义
        "    z = x + y\n"
        "    return z\n"
        "end\n"
    )
    assert source == expected  # 断言生成的代码与预期一致

# 测试带有数学常数的Julia代码生成
def test_jl_numbersymbol():
    name_expr = ("test", pi**Catalan)  # 定义函数名和表达式元组
    result, = codegen(name_expr, "Julia", header=False, empty=False)  # 生成Julia代码
    source = result[1]  # 获取生成的代码字符串
    expected = (
        "function test()\n"  # 预期的Julia函数定义
        "    out1 = pi ^ catalan\n"
        "    return out1\n"
        "end\n"
    )
    assert source == expected  # 断言生成的代码与预期一致

# 测试带有数学常数且不内联的Julia代码生成（预期失败）
@XFAIL
def test_jl_numbersymbol_no_inline():
    # FIXME: how to pass inline=False to the JuliaCodePrinter?
    name_expr = ("test", [pi**Catalan, EulerGamma])  # 定义函数名和表达式列表
    result, = codegen(name_expr, "Julia", header=False,
                      empty=False, inline=False)  # 生成Julia代码
    source = result[1]  # 获取生成的代码字符串
    expected = (
        "function test()\n"  # 预期的Julia函数定义
        "    Catalan = 0.915965594177219\n"
        "    EulerGamma = 0.5772156649015329\n"
        "    out1 = pi ^ Catalan\n"
        "    out2 = EulerGamma\n"
        "    return out1, out2\n"
        "end\n"
    )
    assert source == expected  # 断言生成的代码与预期一致

# 测试带有特定参数顺序的Julia代码生成
def test_jl_code_argument_order():
    expr = x + y  # 定义表达式
    routine = make_routine("test", expr, argument_sequence=[z, x, y], language="julia")  # 创建特定参数顺序的Julia函数
    code_gen = JuliaCodeGen()  # 创建Julia代码生成器对象
    output = StringIO()  # 创建内存中的字符串流对象
    # 调用code_gen对象的dump_jl方法，生成Julia语言代码并写入output对象，命名为"test"
    code_gen.dump_jl([routine], output, "test", header=False, empty=False)
    # 从output对象中获取生成的Julia代码内容
    source = output.getvalue()
    # 预期的Julia代码内容，包括一个函数定义和返回语句
    expected = (
        "function test(z, x, y)\n"
        "    out1 = x + y\n"
        "    return out1\n"
        "end\n"
    )
    # 断言生成的Julia代码与预期的Julia代码内容相等
    assert source == expected
def test_multiple_results_m():
    # Here the output order is the input order

    # Define expressions
    expr1 = (x + y)*z      # Calculate (x + y) * z
    expr2 = (x - y)*z      # Calculate (x - y) * z

    # Create a tuple of the name and expressions
    name_expr = ("test", [expr1, expr2])

    # Generate code in Julia for the given expressions
    result, = codegen(name_expr, "Julia", header=False, empty=False)

    # Extract the source code from the result
    source = result[1]

    # Expected Julia function definition
    expected = (
        "function test(x, y, z)\n"
        "    out1 = z .* (x + y)\n"
        "    out2 = z .* (x - y)\n"
        "    return out1, out2\n"
        "end\n"
    )

    # Assert that the generated source matches the expected function definition
    assert source == expected


def test_results_named_unordered():
    # Here output order is based on name_expr

    # Define symbols
    A, B, C = symbols('A,B,C')

    # Define expressions with SymPy Equalities
    expr1 = Equality(C, (x + y)*z)
    expr2 = Equality(A, (x - y)*z)
    expr3 = Equality(B, 2*x)

    # Create a tuple of the name and expressions
    name_expr = ("test", [expr1, expr2, expr3])

    # Generate code in Julia for the given expressions
    result, = codegen(name_expr, "Julia", header=False, empty=False)

    # Extract the source code from the result
    source = result[1]

    # Expected Julia function definition
    expected = (
        "function test(x, y, z)\n"
        "    C = z .* (x + y)\n"
        "    A = z .* (x - y)\n"
        "    B = 2 * x\n"
        "    return C, A, B\n"
        "end\n"
    )

    # Assert that the generated source matches the expected function definition
    assert source == expected


def test_results_named_ordered():
    # Define symbols
    A, B, C = symbols('A,B,C')

    # Define expressions with SymPy Equalities
    expr1 = Equality(C, (x + y)*z)
    expr2 = Equality(A, (x - y)*z)
    expr3 = Equality(B, 2*x)

    # Create a tuple of the name and expressions
    name_expr = ("test", [expr1, expr2, expr3])

    # Generate code in Julia for the given expressions, with specified argument sequence
    result = codegen(name_expr, "Julia", header=False, empty=False,
                     argument_sequence=(x, z, y))

    # Assert that the generated file name matches the expected
    assert result[0][0] == "test.jl"

    # Extract the source code from the result
    source = result[0][1]

    # Expected Julia function definition
    expected = (
        "function test(x, z, y)\n"
        "    C = z .* (x + y)\n"
        "    A = z .* (x - y)\n"
        "    B = 2 * x\n"
        "    return C, A, B\n"
        "end\n"
    )

    # Assert that the generated source matches the expected function definition
    assert source == expected


def test_complicated_jl_codegen():
    # Import necessary functions from SymPy
    from sympy.functions.elementary.trigonometric import (cos, sin, tan)

    # Define a complex expression involving trigonometric functions
    name_expr = ("testlong",
            [ ((sin(x) + cos(y) + tan(z))**3).expand(),
            cos(cos(cos(cos(cos(cos(cos(cos(x + y + z))))))))
    ])

    # Generate code in Julia for the given expressions
    result = codegen(name_expr, "Julia", header=False, empty=False)

    # Assert that the generated file name matches the expected
    assert result[0][0] == "testlong.jl"

    # Extract the source code from the result
    source = result[0][1]

    # Expected Julia function definition
    expected = (
        "function testlong(x, y, z)\n"
        "    out1 = sin(x) .^ 3 + 3 * sin(x) .^ 2 .* cos(y) + 3 * sin(x) .^ 2 .* tan(z)"
        " + 3 * sin(x) .* cos(y) .^ 2 + 6 * sin(x) .* cos(y) .* tan(z) + 3 * sin(x) .* tan(z) .^ 2"
        " + cos(y) .^ 3 + 3 * cos(y) .^ 2 .* tan(z) + 3 * cos(y) .* tan(z) .^ 2 + tan(z) .^ 3\n"
        "    out2 = cos(cos(cos(cos(cos(cos(cos(cos(x + y + z))))))))\n"
        "    return out1, out2\n"
        "end\n"
    )

    # Assert that the generated source matches the expected function definition
    assert source == expected


def test_jl_output_arg_mixed_unordered():
    # named outputs are alphabetical, unnamed output appear in the given order

    # Import necessary functions from SymPy
    from sympy.functions.elementary.trigonometric import (cos, sin)

    # Define symbols
    a = symbols("a")

    # Define expressions with SymPy Equalities
    name_expr = ("foo", [cos(2*x), Equality(y, sin(x)), cos(x), Equality(a, sin(2*x))])

    # Generate code in Julia for the given expressions
    result, = codegen(name_expr, "Julia", header=False, empty=False)

    # Assert that the generated file name matches the expected
    assert result[0] == "foo.jl"

    # Extract the source code from the result
    source = result[1];
    expected = (
        'function foo(x)\n'  # 定义一个名为 foo 的函数，参数为 x
        '    out1 = cos(2 * x)\n'  # 计算 cos(2 * x)，并将结果赋给 out1
        '    y = sin(x)\n'  # 计算 sin(x)，并将结果赋给 y
        '    out3 = cos(x)\n'  # 计算 cos(x)，并将结果赋给 out3
        '    a = sin(2 * x)\n'  # 计算 sin(2 * x)，并将结果赋给 a
        '    return out1, y, out3, a\n'  # 返回 out1, y, out3, a 这四个变量的值
        'end\n'  # 函数定义结束
    )
    # 断言 source 变量的值与 expected 变量的值相等
    assert source == expected
# 定义测试函数，用于生成带有 Piecewise 表达式的 Julia 代码
def test_jl_piecewise_():
    # 定义 Piecewise 表达式 pw，包含四个条件分支
    pw = Piecewise((0, x < -1), (x**2, x <= 1), (-x+2, x > 1), (1, True), evaluate=False)
    # 将表达式命名为 "pwtest"，传递给 codegen 函数生成 Julia 代码，不包含头部信息，不允许空函数
    name_expr = ("pwtest", pw)
    # 调用 codegen 函数生成 Julia 代码，返回结果
    result, = codegen(name_expr, "Julia", header=False, empty=False)
    # 获取生成的代码
    source = result[1]
    # 预期的 Julia 代码字符串
    expected = (
        "function pwtest(x)\n"
        "    out1 = ((x < -1) ? (0) :\n"
        "    (x <= 1) ? (x .^ 2) :\n"
        "    (x > 1) ? (2 - x) : (1))\n"
        "    return out1\n"
        "end\n"
    )
    # 断言生成的代码与预期的代码相符合
    assert source == expected


@XFAIL
def test_jl_piecewise_no_inline():
    # FIXME: 如何向 JuliaCodePrinter 传递 inline=False？
    # 定义 Piecewise 表达式 pw，包含四个条件分支
    pw = Piecewise((0, x < -1), (x**2, x <= 1), (-x+2, x > 1), (1, True))
    # 将表达式命名为 "pwtest"，传递给 codegen 函数生成 Julia 代码，不包含头部信息，不允许空函数，禁用内联优化
    name_expr = ("pwtest", pw)
    # 调用 codegen 函数生成 Julia 代码，返回结果
    result, = codegen(name_expr, "Julia", header=False, empty=False,
                      inline=False)
    # 获取生成的代码
    source = result[1]
    # 预期的 Julia 代码字符串，使用 if-elseif 结构表示 Piecewise 表达式
    expected = (
        "function pwtest(x)\n"
        "    if (x < -1)\n"
        "        out1 = 0\n"
        "    elseif (x <= 1)\n"
        "        out1 = x .^ 2\n"
        "    elseif (x > 1)\n"
        "        out1 = -x + 2\n"
        "    else\n"
        "        out1 = 1\n"
        "    end\n"
        "    return out1\n"
        "end\n"
    )
    # 断言生成的代码与预期的代码相符合
    assert source == expected


def test_jl_multifcns_per_file():
    # 定义多个名字和表达式的列表
    name_expr = [("foo", [2*x, 3*y]), ("bar", [y**2, 4*y])]
    # 调用 codegen 函数生成 Julia 代码，不包含头部信息，不允许空函数
    result = codegen(name_expr, "Julia", header=False, empty=False)
    # 断言生成的文件名和内容与预期相符
    assert result[0][0] == "foo.jl"
    source = result[0][1]
    # 预期的 Julia 代码字符串，包含两个函数 foo 和 bar
    expected = (
        "function foo(x, y)\n"
        "    out1 = 2 * x\n"
        "    out2 = 3 * y\n"
        "    return out1, out2\n"
        "end\n"
        "function bar(y)\n"
        "    out1 = y .^ 2\n"
        "    out2 = 4 * y\n"
        "    return out1, out2\n"
        "end\n"
    )
    # 断言生成的代码与预期的代码相符合
    assert source == expected


def test_jl_multifcns_per_file_w_header():
    # 定义多个名字和表达式的列表
    name_expr = [("foo", [2*x, 3*y]), ("bar", [y**2, 4*y])]
    # 调用 codegen 函数生成 Julia 代码，包含头部信息，不允许空函数
    result = codegen(name_expr, "Julia", header=True, empty=False)
    # 断言生成的文件名和内容与预期相符
    assert result[0][0] == "foo.jl"
    source = result[0][1]
    # 预期的 Julia 代码字符串，包含 SymPy 版本信息和头部信息，以及两个函数 foo 和 bar
    expected = (
        "#   Code generated with SymPy " + sympy.__version__ + "\n"
        "#\n"
        "#   See http://www.sympy.org/ for more information.\n"
        "#\n"
        "#   This file is part of 'project'\n"
        "function foo(x, y)\n"
        "    out1 = 2 * x\n"
        "    out2 = 3 * y\n"
        "    return out1, out2\n"
        "end\n"
        "function bar(y)\n"
        "    out1 = y .^ 2\n"
        "    out2 = 4 * y\n"
        "    return out1, out2\n"
        "end\n"
    )
    # 断言生成的代码与预期的代码相符合
    assert source == expected


def test_jl_filename_match_prefix():
    # 定义多个名字和表达式的列表
    name_expr = [("foo", [2*x, 3*y]), ("bar", [y**2, 4*y])]
    # 调用 codegen 函数生成 Julia 代码，文件名前缀为 "baz"，不包含头部信息，不允许空函数
    result, = codegen(name_expr, "Julia", prefix="baz", header=False,
                     empty=False)
    # 断言生成的文件名与预期相符
    assert result[0] == "baz.jl"


def test_jl_matrix_named():
    # 定义一个矩阵表达式 e2
    e2 = Matrix([[x, 2*y, pi*z]])
    # 将矩阵命名为 "myout1"，传递给 codegen 函数生成 Julia 代码，不包含头部信息，不允许空函数
    name_expr = ("test", Equality(MatrixSymbol('myout1', 1, 3), e2))
    # 调用 codegen 函数生成 Julia 代码，返回结果
    result = codegen(name_expr, "Julia", header=False, empty=False)
    # 断言检查第一个元素的第一个条目是否为 "test.jl"
    assert result[0][0] == "test.jl"
    
    # 将结果中第一个元素的第二个条目赋给变量 source
    source = result[0][1]
    
    # 定义预期的字符串形式的函数内容
    expected = (
        "function test(x, y, z)\n"
        "    myout1 = [x 2 * y pi * z]\n"
        "    return myout1\n"
        "end\n"
    )
    
    # 断言变量 source 是否等于预期的函数内容字符串
    assert source == expected
# 定义一个测试函数，生成一个名为 myout1 的 1x3 矩阵符号
def test_jl_matrix_named_matsym():
    myout1 = MatrixSymbol('myout1', 1, 3)
    # 创建一个包含表达式的矩阵 e2
    e2 = Matrix([[x, 2*y, pi*z]])
    # 将名字和表达式组成元组 name_expr
    name_expr = ("test", Equality(myout1, e2, evaluate=False))
    # 使用 codegen 函数生成 Julia 语言的代码，不包含头部和空行
    result, = codegen(name_expr, "Julia", header=False, empty=False)
    # 获取生成的代码字符串
    source = result[1]
    # 预期的 Julia 函数代码
    expected = (
        "function test(x, y, z)\n"
        "    myout1 = [x 2 * y pi * z]\n"
        "    return myout1\n"
        "end\n"
    )
    # 断言生成的代码与预期代码相符
    assert source == expected


# 定义一个测试函数，生成一个自动命名的矩阵表达式
def test_jl_matrix_output_autoname():
    expr = Matrix([[x, x+y, 3]])
    # 将名字和表达式组成元组 name_expr
    name_expr = ("test", expr)
    # 使用 codegen 函数生成 Julia 语言的代码，不包含头部和空行
    result, = codegen(name_expr, "Julia", header=False, empty=False)
    # 获取生成的代码字符串
    source = result[1]
    # 预期的 Julia 函数代码
    expected = (
        "function test(x, y)\n"
        "    out1 = [x x + y 3]\n"
        "    return out1\n"
        "end\n"
    )
    # 断言生成的代码与预期代码相符
    assert source == expected


# 定义一个测试函数，生成多个自动命名的表达式
def test_jl_matrix_output_autoname_2():
    e1 = (x + y)
    e2 = Matrix([[2*x, 2*y, 2*z]])
    e3 = Matrix([[x], [y], [z]])
    e4 = Matrix([[x, y], [z, 16]])
    # 将名字和表达式组成元组 name_expr
    name_expr = ("test", (e1, e2, e3, e4))
    # 使用 codegen 函数生成 Julia 语言的代码，不包含头部和空行
    result, = codegen(name_expr, "Julia", header=False, empty=False)
    # 获取生成的代码字符串
    source = result[1]
    # 预期的 Julia 函数代码
    expected = (
        "function test(x, y, z)\n"
        "    out1 = x + y\n"
        "    out2 = [2 * x 2 * y 2 * z]\n"
        "    out3 = [x, y, z]\n"
        "    out4 = [x  y;\n"
        "    z 16]\n"
        "    return out1, out2, out3, out4\n"
        "end\n"
    )
    # 断言生成的代码与预期代码相符
    assert source == expected


# 定义一个测试函数，生成多个命名的表达式并按顺序输出
def test_jl_results_matrix_named_ordered():
    B, C = symbols('B,C')
    A = MatrixSymbol('A', 1, 3)
    expr1 = Equality(C, (x + y)*z)
    expr2 = Equality(A, Matrix([[1, 2, x]]))
    expr3 = Equality(B, 2*x)
    # 将名字和表达式列表组成元组 name_expr
    name_expr = ("test", [expr1, expr2, expr3])
    # 使用 codegen 函数生成 Julia 语言的代码，不包含头部和空行，指定参数顺序
    result, = codegen(name_expr, "Julia", header=False, empty=False,
                     argument_sequence=(x, z, y))
    # 获取生成的代码字符串
    source = result[1]
    # 预期的 Julia 函数代码
    expected = (
        "function test(x, z, y)\n"
        "    C = z .* (x + y)\n"
        "    A = [1 2 x]\n"
        "    B = 2 * x\n"
        "    return C, A, B\n"
        "end\n"
    )
    # 断言生成的代码与预期代码相符
    assert source == expected


# 定义一个测试函数，生成矩阵符号的切片操作
def test_jl_matrixsymbol_slice():
    A = MatrixSymbol('A', 2, 3)
    B = MatrixSymbol('B', 1, 3)
    C = MatrixSymbol('C', 1, 3)
    D = MatrixSymbol('D', 2, 1)
    # 将名字和切片等式列表组成元组 name_expr
    name_expr = ("test", [Equality(B, A[0, :]),
                          Equality(C, A[1, :]),
                          Equality(D, A[:, 2])])
    # 使用 codegen 函数生成 Julia 语言的代码，不包含头部和空行
    result, = codegen(name_expr, "Julia", header=False, empty=False)
    # 获取生成的代码字符串
    source = result[1]
    # 预期的 Julia 函数代码
    expected = (
        "function test(A)\n"
        "    B = A[1,:]\n"
        "    C = A[2,:]\n"
        "    D = A[:,3]\n"
        "    return B, C, D\n"
        "end\n"
    )
    # 断言生成的代码与预期代码相符
    assert source == expected


# 定义一个测试函数，生成更复杂的矩阵符号切片操作
def test_jl_matrixsymbol_slice2():
    A = MatrixSymbol('A', 3, 4)
    B = MatrixSymbol('B', 2, 2)
    C = MatrixSymbol('C', 2, 2)
    # 将名字和切片等式列表组成元组 name_expr
    name_expr = ("test", [Equality(B, A[0:2, 0:2]),
                          Equality(C, A[0:2, 1:3])])
    # 使用 codegen 函数生成 Julia 语言的代码，不包含头部和空行
    result, = codegen(name_expr, "Julia", header=False, empty=False)
    # 获取生成的代码字符串
    source = result[1]
    # 定义期望的字符串，表示一个函数的源代码片段
    expected = (
        "function test(A)\n"            # 定义函数 'test'，接受参数 A
        "    B = A[1:2,1:2]\n"           # 计算并赋值 B 为 A 的子矩阵，行 1 到 2，列 1 到 2
        "    C = A[1:2,2:3]\n"           # 计算并赋值 C 为 A 的子矩阵，行 1 到 2，列 2 到 3
        "    return B, C\n"             # 返回 B 和 C
        "end\n"                         # 函数结束
    )
    # 使用断言检查实际的源代码是否与期望的一致
    assert source == expected
# 定义测试函数，演示如何在 Julia 中进行矩阵符号切片
def test_jl_matrixsymbol_slice3():
    # 创建符号矩阵 A，大小为 8x7
    A = MatrixSymbol('A', 8, 7)
    # 创建符号矩阵 B，大小为 2x2
    B = MatrixSymbol('B', 2, 2)
    # 创建符号矩阵 C，大小为 4x2
    C = MatrixSymbol('C', 4, 2)
    # 定义一个名称和表达式的元组
    name_expr = ("test", [Equality(B, A[6:, 1::3]), Equality(C, A[::2, ::3])])
    # 调用 codegen 函数生成 Julia 代码，不包含头部和空行
    result, = codegen(name_expr, "Julia", header=False, empty=False)
    # 获取生成的 Julia 代码
    source = result[1]
    # 期望的 Julia 函数字符串
    expected = (
        "function test(A)\n"
        "    B = A[7:end,2:3:end]\n"
        "    C = A[1:2:end,1:3:end]\n"
        "    return B, C\n"
        "end\n"
    )
    # 断言生成的 Julia 代码与期望的一致
    assert source == expected


def test_jl_matrixsymbol_slice_autoname():
    # 创建符号矩阵 A，大小为 2x3
    A = MatrixSymbol('A', 2, 3)
    # 创建符号矩阵 B，大小为 1x3
    B = MatrixSymbol('B', 1, 3)
    # 定义一个名称和表达式的元组
    name_expr = ("test", [Equality(B, A[0,:]), A[1,:], A[:,0], A[:,1]])
    # 调用 codegen 函数生成 Julia 代码，不包含头部和空行
    result, = codegen(name_expr, "Julia", header=False, empty=False)
    # 获取生成的 Julia 代码
    source = result[1]
    # 期望的 Julia 函数字符串
    expected = (
        "function test(A)\n"
        "    B = A[1,:]\n"
        "    out2 = A[2,:]\n"
        "    out3 = A[:,1]\n"
        "    out4 = A[:,2]\n"
        "    return B, out2, out3, out4\n"
        "end\n"
    )
    # 断言生成的 Julia 代码与期望的一致
    assert source == expected


def test_jl_loops():
    # 导入必要的符号和张量模块
    from sympy.tensor import IndexedBase, Idx
    from sympy.core.symbol import symbols
    # 定义整数符号变量 n 和 m
    n, m = symbols('n m', integer=True)
    # 创建 IndexedBase 对象 A、x 和 y
    A = IndexedBase('A')
    x = IndexedBase('x')
    y = IndexedBase('y')
    # 创建索引变量 i 和 j
    i = Idx('i', m)
    j = Idx('j', n)
    # 调用 codegen 函数生成 Julia 代码，不包含头部和空行
    result, = codegen(('mat_vec_mult', Eq(y[i], A[i, j]*x[j])), "Julia", header=False, empty=False)
    # 获取生成的 Julia 代码
    source = result[1]
    # 期望的 Julia 函数字符串，使用 %(rhs)s 进行动态替换
    expected = (
        'function mat_vec_mult(y, A, m, n, x)\n'
        '    for i = 1:m\n'
        '        y[i] = 0\n'
        '    end\n'
        '    for i = 1:m\n'
        '        for j = 1:n\n'
        '            y[i] = %(rhs)s + y[i]\n'
        '        end\n'
        '    end\n'
        '    return y\n'
        'end\n'
    )
    # 断言生成的 Julia 代码与期望的一致，替换 %(rhs)s
    assert (source == expected % {'rhs': 'A[%s,%s] .* x[j]' % (i, j)} or
            source == expected % {'rhs': 'x[j] .* A[%s,%s]' % (i, j)})


def test_jl_tensor_loops_multiple_contractions():
    # 导入必要的符号和张量模块
    from sympy.tensor import IndexedBase, Idx
    from sympy.core.symbol import symbols
    # 定义整数符号变量 n、m、o 和 p
    n, m, o, p = symbols('n m o p', integer=True)
    # 创建 IndexedBase 对象 A、B 和 y
    A = IndexedBase('A')
    B = IndexedBase('B')
    y = IndexedBase('y')
    # 创建索引变量 i、j、k 和 l
    i = Idx('i', m)
    j = Idx('j', n)
    k = Idx('k', o)
    l = Idx('l', p)
    # 调用 codegen 函数生成 Julia 代码，不包含头部和空行
    result, = codegen(('tensorthing', Eq(y[i], B[j, k, l]*A[i, j, k, l])), "Julia", header=False, empty=False)
    # 获取生成的 Julia 代码
    source = result[1]
    # 定义一个期望的字符串，表示一个特定的函数实现
    expected = (
        'function tensorthing(y, A, B, m, n, o, p)\n'  # 定义一个名为 tensorthing 的函数，接受参数 y, A, B, m, n, o, p
        '    for i = 1:m\n'                           # 开始一个循环，遍历 i 从 1 到 m
        '        y[i] = 0\n'                          # 将 y[i] 初始化为 0
        '    end\n'                                   # 循环结束
        '    for i = 1:m\n'                           # 开始一个循环，遍历 i 从 1 到 m
        '        for j = 1:n\n'                       # 在第一个循环中再嵌套一个循环，遍历 j 从 1 到 n
        '            for k = 1:o\n'                   # 在第二个循环中再嵌套一个循环，遍历 k 从 1 到 o
        '                for l = 1:p\n'               # 在第三个循环中再嵌套一个循环，遍历 l 从 1 到 p
        '                    y[i] = A[i,j,k,l] .* B[j,k,l] + y[i]\n'  # 计算并更新 y[i] 的值
        '                end\n'                       # 第四个循环结束
        '            end\n'                           # 第三个循环结束
        '        end\n'                               # 第二个循环结束
        '    end\n'                                   # 第一个循环结束
        '    return y\n'                              # 返回最终的 y 数组
        'end\n'                                       # 函数定义结束
    )
    
    # 使用断言检查变量 source 是否与期望的字符串 expected 相等
    assert source == expected
# 定义一个测试函数，测试生成Julia语言的代码，处理形如x=x^2的等式
def test_jl_InOutArgument():
    expr = Equality(x, x**2)
    name_expr = ("mysqr", expr)
    # 调用codegen函数生成Julia代码，不包括头部信息，不允许为空
    result, = codegen(name_expr, "Julia", header=False, empty=False)
    # 获取生成的代码
    source = result[1]
    # 预期的生成代码
    expected = (
        "function mysqr(x)\n"
        "    x = x .^ 2\n"
        "    return x\n"
        "end\n"
    )
    # 断言生成的代码与预期代码相同
    assert source == expected


# 定义测试函数，测试生成Julia语言的代码，处理形如x=x^2+y的等式，可以指定参数顺序
def test_jl_InOutArgument_order():
    # 定义表达式
    expr = Equality(x, x**2 + y)
    name_expr = ("test", expr)
    # 调用codegen函数生成Julia代码，不包括头部信息，不允许为空，并指定参数顺序为(x, y)
    result, = codegen(name_expr, "Julia", header=False,
                      empty=False, argument_sequence=(x,y))
    # 获取生成的代码
    source = result[1]
    # 预期的生成代码
    expected = (
        "function test(x, y)\n"
        "    x = x .^ 2 + y\n"
        "    return x\n"
        "end\n"
    )
    # 断言生成的代码与预期代码相同
    assert source == expected
    # 确保参数顺序为(x, y)，而非(y, x)
    expr = Equality(x, x**2 + y)
    name_expr = ("test", expr)
    # 再次调用codegen函数生成Julia代码，不包括头部信息，不允许为空
    result, = codegen(name_expr, "Julia", header=False, empty=False)
    # 获取生成的代码
    source = result[1]
    # 预期的生成代码
    expected = (
        "function test(x, y)\n"
        "    x = x .^ 2 + y\n"
        "    return x\n"
        "end\n"
    )
    # 断言生成的代码与预期代码相同
    assert source == expected


# 定义测试函数，测试生成Julia语言的代码，处理不支持的情况
def test_jl_not_supported():
    # 创建函数f
    f = Function('f')
    # 定义名称表达式
    name_expr = ("test", [f(x).diff(x), S.ComplexInfinity])
    # 调用codegen函数生成Julia代码，不包括头部信息，不允许为空
    result, = codegen(name_expr, "Julia", header=False, empty=False)
    # 获取生成的代码
    source = result[1]
    # 预期的生成代码
    expected = (
        "function test(x)\n"
        "    # unsupported: Derivative(f(x), x)\n"
        "    # unsupported: zoo\n"
        "    out1 = Derivative(f(x), x)\n"
        "    out2 = zoo\n"
        "    return out1, out2\n"
        "end\n"
    )
    # 断言生成的代码与预期代码相同
    assert source == expected


# 定义测试函数，测试生成Julia语言的代码，处理全局变量
def test_global_vars_octave():
    # 定义符号变量
    x, y, z, t = symbols("x y z t")
    # 调用codegen函数生成Julia代码，不包括头部信息，不允许为空，指定全局变量y
    result = codegen(('f', x*y), "Julia", header=False, empty=False,
                     global_vars=(y,))
    # 获取生成的代码
    source = result[0][1]
    # 预期的生成代码
    expected = (
        "function f(x)\n"
        "    out1 = x .* y\n"
        "    return out1\n"
        "end\n"
    )
    # 断言生成的代码与预期代码相同
    assert source == expected

    # 调用codegen函数生成Julia代码，不包括头部信息，不允许为空，指定参数顺序为(x, y)，全局变量为(z, t)
    result = codegen(('f', x*y+z), "Julia", header=False, empty=False,
                     argument_sequence=(x, y), global_vars=(z, t))
    # 获取生成的代码
    source = result[0][1]
    # 预期的生成代码
    expected = (
        "function f(x, y)\n"
        "    out1 = x .* y + z\n"
        "    return out1\n"
        "end\n"
    )
    # 断言生成的代码与预期代码相同
    assert source == expected
```