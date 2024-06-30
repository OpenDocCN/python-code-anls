# `D:\src\scipysrc\sympy\sympy\utilities\tests\test_codegen_rust.py`

```
# 导入所需的模块和函数
from io import StringIO
# 导入 SymPy 的核心符号和函数
from sympy.core import S, symbols, pi, Catalan, EulerGamma, Function
# 导入 SymPy 的关系运算模块中的相等关系
from sympy.core.relational import Equality
# 导入 SymPy 的基础函数模块中的分段函数
from sympy.functions.elementary.piecewise import Piecewise
# 导入 SymPy 的代码生成工具中的 Rust 代码生成器和相关函数
from sympy.utilities.codegen import RustCodeGen, codegen, make_routine
# 导入 SymPy 的测试模块中的 XFAIL 标记
from sympy.testing.pytest import XFAIL
# 导入 sympy 根命名空间
import sympy

# 定义符号变量 x, y, z
x, y, z = symbols('x,y,z')

# 测试空 Rust 代码生成
def test_empty_rust_code():
    # 创建 RustCodeGen 对象
    code_gen = RustCodeGen()
    # 创建内存文件对象
    output = StringIO()
    # 调用 RustCodeGen.dump_rs 生成空的 Rust 代码
    code_gen.dump_rs([], output, "file", header=False, empty=False)
    # 获取生成的代码字符串
    source = output.getvalue()
    # 断言生成的代码为空字符串
    assert source == ""

# 测试简单的 Rust 代码生成
def test_simple_rust_code():
    # 定义测试用的名称和表达式
    name_expr = ("test", (x + y)*z)
    # 调用 codegen 函数生成 Rust 代码，不包含头部信息
    result, = codegen(name_expr, "Rust", header=False, empty=False)
    # 断言生成的代码文件名为 test.rs
    assert result[0] == "test.rs"
    # 获取生成的 Rust 代码字符串
    source = result[1]
    # 期待的 Rust 代码格式
    expected = (
        "fn test(x: f64, y: f64, z: f64) -> f64 {\n"
        "    let out1 = z*(x + y);\n"
        "    out1\n"
        "}\n"
    )
    # 断言生成的代码与期待的格式一致
    assert source == expected

# 测试包含头部信息的简单 Rust 代码生成
def test_simple_code_with_header():
    # 定义测试用的名称和表达式
    name_expr = ("test", (x + y)*z)
    # 调用 codegen 函数生成 Rust 代码，包含头部信息
    result, = codegen(name_expr, "Rust", header=True, empty=False)
    # 断言生成的代码文件名为 test.rs
    assert result[0] == "test.rs"
    # 获取生成的 Rust 代码字符串
    source = result[1]
    # 构建版本信息字符串
    version_str = "Code generated with SymPy %s" % sympy.__version__
    # 居中格式化版本信息行
    version_line = version_str.center(76).rstrip()
    # 期待的 Rust 代码格式，包含头部信息和版本信息
    expected = (
        "/*\n"
        " *%(version_line)s\n"
        " *\n"
        " *              See http://www.sympy.org/ for more information.\n"
        " *\n"
        " *                       This file is part of 'project'\n"
        " */\n"
        "fn test(x: f64, y: f64, z: f64) -> f64 {\n"
        "    let out1 = z*(x + y);\n"
        "    out1\n"
        "}\n"
    ) % {'version_line': version_line}
    # 断言生成的代码与期待的格式一致
    assert source == expected

# 测试简单的 Rust 代码生成，输出包含名称赋值
def test_simple_code_nameout():
    # 定义测试用的表达式
    expr = Equality(z, (x + y))
    # 构建名称和表达式的元组
    name_expr = ("test", expr)
    # 调用 codegen 函数生成 Rust 代码，不包含头部信息
    result, = codegen(name_expr, "Rust", header=False, empty=False)
    # 获取生成的 Rust 代码字符串
    source = result[1]
    # 期待的 Rust 代码格式，包含名称赋值
    expected = (
        "fn test(x: f64, y: f64) -> f64 {\n"
        "    let z = x + y;\n"
        "    z\n"
        "}\n"
    )
    # 断言生成的代码与期待的格式一致
    assert source == expected

# 测试使用数学符号的 Rust 代码生成
def test_numbersymbol():
    # 定义测试用的名称和表达式
    name_expr = ("test", pi**Catalan)
    # 调用 codegen 函数生成 Rust 代码，不包含头部信息
    result, = codegen(name_expr, "Rust", header=False, empty=False)
    # 获取生成的 Rust 代码字符串
    source = result[1]
    # 获取 Catalan 常数的数值表示
    catalan_value = Catalan.evalf(17)
    # 期待的 Rust 代码格式，包含常数定义和数学运算
    expected = (
        "fn test() -> f64 {\n"
        "    const Catalan: f64 = %s;\n"
        "    let out1 = PI.powf(Catalan);\n"
        "    out1\n"
        "}\n"
    ) % catalan_value
    # 断言生成的代码与期待的格式一致
    assert source == expected

# 测试失败的 Rust 代码生成，包含内联函数调用
@XFAIL
def test_numbersymbol_inline():
    # FIXME: 如何将内联传递给 RustCodePrinter？
    # 定义测试用的名称和表达式列表
    name_expr = ("test", [pi**Catalan, EulerGamma])
    # 调用 codegen 函数生成 Rust 代码，不包含头部信息，内联函数调用
    result, = codegen(name_expr, "Rust", header=False, empty=False, inline=True)
    # 获取生成的 Rust 代码字符串
    source = result[1]
    # 获取 Catalan 和 EulerGamma 常数的数值表示
    catalan_value = Catalan.evalf(17)
    euler_gamma_value = EulerGamma.evalf(17)
    # 期待的 Rust 代码格式，包含常数定义和数学运算，多返回值
    expected = (
        "fn test() -> (f64, f64) {\n"
        "    const Catalan: f64 = %s;\n"
        "    const EulerGamma: f64 = %s;\n"
        "    let out1 = PI.powf(Catalan);\n"
        "    let out2 = EulerGamma);\n"
        "    (out1, out2)\n"
        "}\n"
    ) % (catalan_value, euler_gamma_value)
    # 断言生成的代码与期待的格式一致
    assert source == expected
    # 使用字符串格式化操作符将Catalan常数和EulerGamma常数格式化到字符串中
    ) % (Catalan.evalf(17), EulerGamma.evalf(17))
    # 断言源字符串等于期望字符串，用于测试结果是否符合预期
    assert source == expected
def test_argument_order():
    # 创建表达式 x + y
    expr = x + y
    # 使用 make_routine 函数生成一个名为 "test" 的例程，接受参数顺序为 [z, x, y]，语言为 "rust"
    routine = make_routine("test", expr, argument_sequence=[z, x, y], language="rust")
    # 创建 RustCodeGen 对象
    code_gen = RustCodeGen()
    # 创建一个字符串输出流
    output = StringIO()
    # 将例程 dump 到输出流中，生成名为 "test" 的 Rust 代码，不包含头部和空行
    code_gen.dump_rs([routine], output, "test", header=False, empty=False)
    # 从输出流中获取生成的 Rust 代码
    source = output.getvalue()
    # 预期的 Rust 代码字符串
    expected = (
        "fn test(z: f64, x: f64, y: f64) -> f64 {\n"
        "    let out1 = x + y;\n"
        "    out1\n"
        "}\n"
    )
    # 断言生成的代码与预期的代码相等
    assert source == expected


def test_multiple_results_rust():
    # 这里输出顺序与输入顺序相同
    expr1 = (x + y)*z
    expr2 = (x - y)*z
    # 创建一个名为 "test" 的例程，包含两个表达式 expr1 和 expr2
    name_expr = ("test", [expr1, expr2])
    # 调用 codegen 函数生成 Rust 代码，不包含头部和空行
    result, = codegen(name_expr, "Rust", header=False, empty=False)
    # 获取生成的 Rust 代码
    source = result[1]
    # 预期的 Rust 代码字符串
    expected = (
        "fn test(x: f64, y: f64, z: f64) -> (f64, f64) {\n"
        "    let out1 = z*(x + y);\n"
        "    let out2 = z*(x - y);\n"
        "    (out1, out2)\n"
        "}\n"
    )
    # 断言生成的代码与预期的代码相等
    assert source == expected


def test_results_named_unordered():
    # 这里输出顺序基于 name_expr 中的顺序
    A, B, C = symbols('A,B,C')
    expr1 = Equality(C, (x + y)*z)
    expr2 = Equality(A, (x - y)*z)
    expr3 = Equality(B, 2*x)
    # 创建一个名为 "test" 的例程，包含三个表达式 expr1, expr2 和 expr3
    name_expr = ("test", [expr1, expr2, expr3])
    # 调用 codegen 函数生成 Rust 代码，不包含头部和空行
    result, = codegen(name_expr, "Rust", header=False, empty=False)
    # 获取生成的 Rust 代码
    source = result[1]
    # 预期的 Rust 代码字符串
    expected = (
        "fn test(x: f64, y: f64, z: f64) -> (f64, f64, f64) {\n"
        "    let C = z*(x + y);\n"
        "    let A = z*(x - y);\n"
        "    let B = 2*x;\n"
        "    (C, A, B)\n"
        "}\n"
    )
    # 断言生成的代码与预期的代码相等
    assert source == expected


def test_results_named_ordered():
    A, B, C = symbols('A,B,C')
    expr1 = Equality(C, (x + y)*z)
    expr2 = Equality(A, (x - y)*z)
    expr3 = Equality(B, 2*x)
    # 创建一个名为 "test" 的例程，包含三个表达式 expr1, expr2 和 expr3
    name_expr = ("test", [expr1, expr2, expr3])
    # 调用 codegen 函数生成 Rust 代码，不包含头部和空行，参数顺序为 (x, z, y)
    result = codegen(name_expr, "Rust", header=False, empty=False,
                     argument_sequence=(x, z, y))
    # 断言生成的第一个文件名为 "test.rs"
    assert result[0][0] == "test.rs"
    # 获取生成的 Rust 代码
    source = result[0][1]
    # 预期的 Rust 代码字符串
    expected = (
        "fn test(x: f64, z: f64, y: f64) -> (f64, f64, f64) {\n"
        "    let C = z*(x + y);\n"
        "    let A = z*(x - y);\n"
        "    let B = 2*x;\n"
        "    (C, A, B)\n"
        "}\n"
    )
    # 断言生成的代码与预期的代码相等
    assert source == expected


def test_complicated_rs_codegen():
    from sympy.functions.elementary.trigonometric import (cos, sin, tan)
    # 创建一个名为 "testlong" 的例程，包含复杂的表达式
    name_expr = ("testlong",
            [ ((sin(x) + cos(y) + tan(z))**3).expand(),
            cos(cos(cos(cos(cos(cos(cos(cos(x + y + z))))))))
    ])
    # 调用 codegen 函数生成 Rust 代码，不包含头部和空行
    result = codegen(name_expr, "Rust", header=False, empty=False)
    # 断言生成的第一个文件名为 "testlong.rs"
    assert result[0][0] == "testlong.rs"
    # 获取生成的 Rust 代码
    source = result[0][1]
    # 没有预期的 Rust 代码字符串进行比较，因为这里没有提供预期结果的示例
    expected = (
        "fn testlong(x: f64, y: f64, z: f64) -> (f64, f64) {\n"
        "    let out1 = x.sin().powi(3) + 3*x.sin().powi(2)*y.cos()"
        " + 3*x.sin().powi(2)*z.tan() + 3*x.sin()*y.cos().powi(2)"
        " + 6*x.sin()*y.cos()*z.tan() + 3*x.sin()*z.tan().powi(2)"
        " + y.cos().powi(3) + 3*y.cos().powi(2)*z.tan()"
        " + 3*y.cos()*z.tan().powi(2) + z.tan().powi(3);\n"
        "    let out2 = (x + y + z).cos().cos().cos().cos()"
        ".cos().cos().cos().cos();\n"
        "    (out1, out2)\n"
        "}\n"
    )
    # 定义一个字符串变量 expected，其中包含一个 Rust 函数的字符串表示
    # 函数 testlong 接受三个参数 x, y, z，并返回一个包含两个 f64 类型值的元组
    # out1 表示函数中的第一个结果，通过数学运算得出
    # out2 表示函数中的第二个结果，通过复合余弦函数的运算得出
    # 将 out1 和 out2 包装在元组中，作为函数 testlong 的返回值
    assert source == expected
    # 断言变量 source 的值等于预期值 expected，用于测试代码是否生成了正确的 Rust 函数字符串
# 测试混合无序输出参数的功能
def test_output_arg_mixed_unordered():
    # 导入必要的函数
    from sympy.functions.elementary.trigonometric import (cos, sin)
    # 创建符号变量a
    a = symbols("a")
    # 定义包含名称和表达式的元组
    name_expr = ("foo", [cos(2*x), Equality(y, sin(x)), cos(x), Equality(a, sin(2*x))])
    # 调用codegen函数生成Rust代码，禁用头部信息和空行
    result, = codegen(name_expr, "Rust", header=False, empty=False)
    # 断言生成的文件名
    assert result[0] == "foo.rs"
    # 获取生成的源代码
    source = result[1];
    # 期望生成的Rust代码
    expected = (
        "fn foo(x: f64) -> (f64, f64, f64, f64) {\n"
        "    let out1 = (2*x).cos();\n"
        "    let y = x.sin();\n"
        "    let out3 = x.cos();\n"
        "    let a = (2*x).sin();\n"
        "    (out1, y, out3, a)\n"
        "}\n"
    )
    # 断言生成的源代码与期望代码一致
    assert source == expected


# 测试Piecewise函数生成代码的功能
def test_piecewise_():
    # 创建Piecewise对象pw
    pw = Piecewise((0, x < -1), (x**2, x <= 1), (-x+2, x > 1), (1, True), evaluate=False)
    # 定义包含名称和Piecewise对象的元组
    name_expr = ("pwtest", pw)
    # 调用codegen函数生成Rust代码，禁用头部信息和空行
    result, = codegen(name_expr, "Rust", header=False, empty=False)
    # 获取生成的源代码
    source = result[1]
    # 期望生成的Rust代码
    expected = (
        "fn pwtest(x: f64) -> f64 {\n"
        "    let out1 = if (x < -1) {\n"
        "        0\n"
        "    } else if (x <= 1) {\n"
        "        x.powi(2)\n"
        "    } else if (x > 1) {\n"
        "        2 - x\n"
        "    } else {\n"
        "        1\n"
        "    };\n"
        "    out1\n"
        "}\n"
    )
    # 断言生成的源代码与期望代码一致
    assert source == expected


# 标记为XFAIL，测试内联Piecewise函数生成代码的功能
@XFAIL
def test_piecewise_inline():
    # FIXME: how to pass inline to the RustCodePrinter?
    # 创建Piecewise对象pw
    pw = Piecewise((0, x < -1), (x**2, x <= 1), (-x+2, x > 1), (1, True))
    # 定义包含名称和Piecewise对象的元组
    name_expr = ("pwtest", pw)
    # 调用codegen函数生成Rust代码，禁用头部信息和空行，启用内联
    result, = codegen(name_expr, "Rust", header=False, empty=False,
                      inline=True)
    # 获取生成的源代码
    source = result[1]
    # 期望生成的Rust代码
    expected = (
        "fn pwtest(x: f64) -> f64 {\n"
        "    let out1 = if (x < -1) { 0 } else if (x <= 1) { x.powi(2) }"
        " else if (x > 1) { -x + 2 } else { 1 };\n"
        "    out1\n"
        "}\n"
    )
    # 断言生成的源代码与期望代码一致
    assert source == expected


# 测试生成多个函数在同一个文件中的功能
def test_multifcns_per_file():
    # 定义包含名称和表达式列表的列表
    name_expr = [ ("foo", [2*x, 3*y]), ("bar", [y**2, 4*y]) ]
    # 调用codegen函数生成Rust代码，禁用头部信息和空行
    result = codegen(name_expr, "Rust", header=False, empty=False)
    # 断言生成的第一个文件名
    assert result[0][0] == "foo.rs"
    # 获取生成的第一个文件的源代码
    source = result[0][1];
    # 期望生成的Rust代码
    expected = (
        "fn foo(x: f64, y: f64) -> (f64, f64) {\n"
        "    let out1 = 2*x;\n"
        "    let out2 = 3*y;\n"
        "    (out1, out2)\n"
        "}\n"
        "fn bar(y: f64) -> (f64, f64) {\n"
        "    let out1 = y.powi(2);\n"
        "    let out2 = 4*y;\n"
        "    (out1, out2)\n"
        "}\n"
    )
    # 断言生成的源代码与期望代码一致
    assert source == expected


# 测试生成多个函数在同一个文件中的功能（包含头部信息）
def test_multifcns_per_file_w_header():
    # 定义包含名称和表达式列表的列表
    name_expr = [ ("foo", [2*x, 3*y]), ("bar", [y**2, 4*y]) ]
    # 调用codegen函数生成Rust代码，启用头部信息和禁用空行
    result = codegen(name_expr, "Rust", header=True, empty=False)
    # 断言生成的第一个文件名
    assert result[0][0] == "foo.rs"
    # 获取生成的第一个文件的源代码
    source = result[0][1];
    # 根据SymPy版本创建版本字符串
    version_str = "Code generated with SymPy %s" % sympy.__version__
    # 居中对齐版本字符串
    version_line = version_str.center(76).rstrip()
    expected = (
        "/*\n"  # 创建一个多行字符串，表示包含版本信息的注释块
        " *%(version_line)s\n"  # 插入版本信息行
        " *\n"  # 空行
        " *              See http://www.sympy.org/ for more information.\n"  # 提供一个网址链接
        " *\n"  # 空行
        " *                       This file is part of 'project'\n"  # 指明文件属于项目的一部分
        " */\n"  # 注释块结束符号
        "fn foo(x: f64, y: f64) -> (f64, f64) {\n"  # 定义一个函数 foo，接受两个 f64 类型参数，返回两个 f64 类型值
        "    let out1 = 2*x;\n"  # 计算 2*x，将结果赋给 out1
        "    let out2 = 3*y;\n"  # 计算 3*y，将结果赋给 out2
        "    (out1, out2)\n"  # 返回元组 (out1, out2)
        "}\n"  # 函数定义结束
        "fn bar(y: f64) -> (f64, f64) {\n"  # 定义一个函数 bar，接受一个 f64 类型参数，返回两个 f64 类型值
        "    let out1 = y.powi(2);\n"  # 计算 y 的平方，将结果赋给 out1
        "    let out2 = 4*y;\n"  # 计算 4*y，将结果赋给 out2
        "    (out1, out2)\n"  # 返回元组 (out1, out2)
        "}\n"  # 函数定义结束
    ) % {'version_line': version_line}  # 使用提供的 version_line 字符串格式化 expected 变量中的版本行

    assert source == expected  # 断言 source 变量的值与 expected 变量的值相等
def test_filename_match_prefix():
    # 定义一个名称-表达式列表
    name_expr = [ ("foo", [2*x, 3*y]), ("bar", [y**2, 4*y]) ]
    # 调用 codegen 函数生成 Rust 语言的代码，设置前缀为 "baz"，不包含头部信息，不允许为空
    result, = codegen(name_expr, "Rust", prefix="baz", header=False,
                     empty=False)
    # 断言生成的第一个结果文件名为 "baz.rs"
    assert result[0] == "baz.rs"


def test_InOutArgument():
    # 创建一个表达式，表示 x 等于 x 的平方
    expr = Equality(x, x**2)
    # 定义名称-表达式元组
    name_expr = ("mysqr", expr)
    # 调用 codegen 函数生成 Rust 语言的代码，不包含头部信息，不允许为空
    result, = codegen(name_expr, "Rust", header=False, empty=False)
    # 获取生成的代码的第二部分（源代码）
    source = result[1]
    # 定义预期的生成代码字符串
    expected = (
        "fn mysqr(x: f64) -> f64 {\n"
        "    let x = x.powi(2);\n"
        "    x\n"
        "}\n"
    )
    # 断言生成的源代码与预期相符
    assert source == expected


def test_InOutArgument_order():
    # 创建一个表达式，表示 x 等于 x 的平方加 y
    expr = Equality(x, x**2 + y)
    # 定义名称-表达式元组
    name_expr = ("test", expr)
    # 调用 codegen 函数生成 Rust 语言的代码，不包含头部信息，不允许为空，指定参数顺序为 (x, y)
    result, = codegen(name_expr, "Rust", header=False,
                      empty=False, argument_sequence=(x,y))
    # 获取生成的代码的第二部分（源代码）
    source = result[1]
    # 定义预期的生成代码字符串
    expected = (
        "fn test(x: f64, y: f64) -> f64 {\n"
        "    let x = x.powi(2) + y;\n"
        "    x\n"
        "}\n"
    )
    # 断言生成的源代码与预期相符
    assert source == expected
    # 再次调用 codegen 函数生成 Rust 语言的代码，不包含头部信息，不允许为空
    expr = Equality(x, x**2 + y)
    name_expr = ("test", expr)
    result, = codegen(name_expr, "Rust", header=False, empty=False)
    # 获取生成的代码的第二部分（源代码）
    source = result[1]
    # 定义预期的生成代码字符串
    expected = (
        "fn test(x: f64, y: f64) -> f64 {\n"
        "    let x = x.powi(2) + y;\n"
        "    x\n"
        "}\n"
    )
    # 断言生成的源代码与预期相符
    assert source == expected


def test_not_supported():
    # 创建一个函数 f(x)
    f = Function('f')
    # 定义名称-表达式元组，其中包含不支持的表达式和值
    name_expr = ("test", [f(x).diff(x), S.ComplexInfinity])
    # 调用 codegen 函数生成 Rust 语言的代码，不包含头部信息，不允许为空
    result, = codegen(name_expr, "Rust", header=False, empty=False)
    # 获取生成的代码的第二部分（源代码）
    source = result[1]
    # 定义预期的生成代码字符串
    expected = (
        "fn test(x: f64) -> (f64, f64) {\n"
        "    // unsupported: Derivative(f(x), x)\n"
        "    // unsupported: zoo\n"
        "    let out1 = Derivative(f(x), x);\n"
        "    let out2 = zoo;\n"
        "    (out1, out2)\n"
        "}\n"
    )
    # 断言生成的源代码与预期相符
    assert source == expected


def test_global_vars_rust():
    # 定义符号变量 x, y, z, t
    x, y, z, t = symbols("x y z t")
    # 调用 codegen 函数生成 Rust 语言的代码，不包含头部信息，不允许为空，指定全局变量 y
    result = codegen(('f', x*y), "Rust", header=False, empty=False,
                     global_vars=(y,))
    # 获取生成的第一个结果的第二部分（源代码）
    source = result[0][1]
    # 定义预期的生成代码字符串
    expected = (
        "fn f(x: f64) -> f64 {\n"
        "    let out1 = x*y;\n"
        "    out1\n"
        "}\n"
    )
    # 断言生成的源代码与预期相符
    assert source == expected

    # 调用 codegen 函数生成 Rust 语言的代码，不包含头部信息，不允许为空，指定参数顺序为 (x, y)，全局变量为 z, t
    result = codegen(('f', x*y+z), "Rust", header=False, empty=False,
                     argument_sequence=(x, y), global_vars=(z, t))
    # 获取生成的第一个结果的第二部分（源代码）
    source = result[0][1]
    # 定义预期的生成代码字符串
    expected = (
        "fn f(x: f64, y: f64) -> f64 {\n"
        "    let out1 = x*y + z;\n"
        "    out1\n"
        "}\n"
    )
    # 断言生成的源代码与预期相符
    assert source == expected
```