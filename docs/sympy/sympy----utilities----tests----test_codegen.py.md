# `D:\src\scipysrc\sympy\sympy\utilities\tests\test_codegen.py`

```
# 从 io 模块导入 StringIO 类，用于在内存中操作字符串数据
from io import StringIO

# 从 sympy.core 模块导入符号相关的类和函数
from sympy.core import symbols, Eq, pi, Catalan, Lambda, Dummy

# 从 sympy.core.relational 模块导入 Equality 类
from sympy.core.relational import Equality

# 从 sympy.core.symbol 模块导入 Symbol 类
from sympy.core.symbol import Symbol

# 从 sympy.functions.special.error_functions 模块导入 erf 函数
from sympy.functions.special.error_functions import erf

# 从 sympy.integrals.integrals 模块导入 Integral 类
from sympy.integrals.integrals import Integral

# 从 sympy.matrices 模块导入 Matrix 和 MatrixSymbol 类
from sympy.matrices import Matrix, MatrixSymbol

# 从 sympy.utilities.codegen 模块导入多个类和函数
from sympy.utilities.codegen import (
    codegen, make_routine, CCodeGen, C89CodeGen, C99CodeGen, InputArgument,
    CodeGenError, FCodeGen, CodeGenArgumentListError, OutputArgument,
    InOutArgument)

# 从 sympy.testing.pytest 模块导入 raises 函数
from sympy.testing.pytest import raises

# 从 sympy.utilities.lambdify 模块导入 implemented_function 函数
from sympy.utilities.lambdify import implemented_function


def get_string(dump_fn, routines, prefix="file", header=False, empty=False):
    """Wrapper for dump_fn. dump_fn writes its results to a stream object and
       this wrapper returns the contents of that stream as a string. This
       auxiliary function is used by many tests below.

       The header and the empty lines are not generated to facilitate the
       testing of the output.
    """
    # 创建一个 StringIO 对象，用于捕获 dump_fn 函数的输出
    output = StringIO()
    # 调用 dump_fn 函数，将其结果写入 output 对象中
    dump_fn(routines, output, prefix, header, empty)
    # 获取 output 对象中的内容作为字符串
    source = output.getvalue()
    # 关闭 output 对象
    output.close()
    # 返回捕获的字符串内容
    return source


def test_Routine_argument_order():
    # 定义符号变量 a, x, y, z
    a, x, y, z = symbols('a x y z')
    # 定义表达式 (x + y) * z
    expr = (x + y)*z
    # 测试 make_routine 函数，验证参数顺序错误是否会引发异常
    raises(CodeGenArgumentListError, lambda: make_routine("test", expr,
           argument_sequence=[z, x]))
    raises(CodeGenArgumentListError, lambda: make_routine("test", Eq(a,
           expr), argument_sequence=[z, x, y]))
    # 使用 make_routine 函数创建 Routine 对象 r，验证其参数顺序和类型
    r = make_routine('test', Eq(a, expr), argument_sequence=[z, x, a, y])
    assert [ arg.name for arg in r.arguments ] == [z, x, a, y]
    assert [ type(arg) for arg in r.arguments ] == [
        InputArgument, InputArgument, OutputArgument, InputArgument  ]
    # 使用 make_routine 函数创建 Routine 对象 r，验证其参数顺序和类型
    r = make_routine('test', Eq(z, expr), argument_sequence=[z, x, y])
    assert [ type(arg) for arg in r.arguments ] == [
        InOutArgument, InputArgument, InputArgument ]

    # 从 sympy.tensor 模块导入 IndexedBase 和 Idx 类
    from sympy.tensor import IndexedBase, Idx
    # 创建 IndexedBase 对象 A, B
    A, B = map(IndexedBase, ['A', 'B'])
    # 定义整数符号 m
    m = symbols('m', integer=True)
    # 创建 Idx 对象 i，表示索引 i
    i = Idx('i', m)
    # 使用 make_routine 函数创建 Routine 对象 r，验证其参数顺序和类型
    r = make_routine('test', Eq(A[i], B[i]), argument_sequence=[B, A, m])
    assert [ arg.name for arg in r.arguments ] == [B.label, A.label, m]

    # 定义积分表达式
    expr = Integral(x*y*z, (x, 1, 2), (y, 1, 3))
    # 使用 make_routine 函数创建 Routine 对象 r，验证其参数顺序和类型
    r = make_routine('test', Eq(a, expr), argument_sequence=[z, x, a, y])
    assert [ arg.name for arg in r.arguments ] == [z, x, a, y]


def test_empty_c_code():
    # 创建 C89CodeGen 对象
    code_gen = C89CodeGen()
    # 调用 get_string 函数，获取 C 代码的字符串表示
    source = get_string(code_gen.dump_c, [])
    # 断言生成的代码字符串与预期的头文件引用相符
    assert source == "#include \"file.h\"\n#include <math.h>\n"


def test_empty_c_code_with_comment():
    # 创建 C89CodeGen 对象
    code_gen = C89CodeGen()
    # 调用 get_string 函数，获取包含注释的 C 代码的字符串表示
    source = get_string(code_gen.dump_c, [], header=True)
    # 断言生成的代码字符串的前82个字符与预期的注释行开头相符
    assert source[:82] == (
        "/******************************************************************************\n *"
    )
          #   "                    Code generated with SymPy 0.7.2-git                    "
    # 确保从第159个字符开始的字符串与以下多行字符串完全匹配，用于检查文件头部注释是否存在和正确
    assert source[158:] == (
            "*\n"
            " *                                                                            *\n"
            " *              See http://www.sympy.org/ for more information.               *\n"
            " *                                                                            *\n"
            " *                       This file is part of 'project'                       *\n"
            " ******************************************************************************/\n"
            "#include \"file.h\"\n"
            "#include <math.h>\n"
            )
# 测试生成一个空的 C 头文件
def test_empty_c_header():
    # 创建 C99 代码生成器的实例
    code_gen = C99CodeGen()
    # 调用 get_string 函数，生成一个空字符串
    source = get_string(code_gen.dump_h, [])
    # 断言生成的字符串是否符合预期的空头文件内容
    assert source == "#ifndef PROJECT__FILE__H\n#define PROJECT__FILE__H\n#endif\n"


# 测试生成一个简单的 C 代码
def test_simple_c_code():
    # 创建符号变量 x, y, z
    x, y, z = symbols('x,y,z')
    # 创建表达式 (x + y)*z
    expr = (x + y)*z
    # 制作一个名为 "test" 的例程，处理表达式 expr
    routine = make_routine("test", expr)
    # 创建 C89 代码生成器的实例
    code_gen = C89CodeGen()
    # 调用 get_string 函数，生成生成的 C 代码字符串
    source = get_string(code_gen.dump_c, [routine])
    # 预期的生成的 C 代码字符串
    expected = (
        "#include \"file.h\"\n"
        "#include <math.h>\n"
        "double test(double x, double y, double z) {\n"
        "   double test_result;\n"
        "   test_result = z*(x + y);\n"
        "   return test_result;\n"
        "}\n"
    )
    # 断言生成的字符串与预期的字符串是否相等
    assert source == expected


# 测试生成包含保留字的 C 代码
def test_c_code_reserved_words():
    # 创建包含保留字 if, typedef, while 的符号变量 x, y, z
    x, y, z = symbols('if, typedef, while')
    # 创建表达式 (x + y) * z
    expr = (x + y) * z
    # 制作一个名为 "test" 的例程，处理表达式 expr
    routine = make_routine("test", expr)
    # 创建 C99 代码生成器的实例
    code_gen = C99CodeGen()
    # 调用 get_string 函数，生成生成的 C 代码字符串
    source = get_string(code_gen.dump_c, [routine])
    # 预期的生成的 C 代码字符串
    expected = (
        "#include \"file.h\"\n"
        "#include <math.h>\n"
        "double test(double if_, double typedef_, double while_) {\n"
        "   double test_result;\n"
        "   test_result = while_*(if_ + typedef_);\n"
        "   return test_result;\n"
        "}\n"
    )
    # 断言生成的字符串与预期的字符串是否相等
    assert source == expected


# 测试生成包含数学常数的 C 代码
def test_numbersymbol_c_code():
    # 制作一个名为 "test" 的例程，处理数学常数 pi**Catalan
    routine = make_routine("test", pi**Catalan)
    # 创建 C89 代码生成器的实例
    code_gen = C89CodeGen()
    # 调用 get_string 函数，生成生成的 C 代码字符串
    source = get_string(code_gen.dump_c, [routine])
    # 使用符号 C 的值，生成预期的 C 代码字符串
    expected = (
        "#include \"file.h\"\n"
        "#include <math.h>\n"
        "double test() {\n"
        "   double test_result;\n"
        "   double const Catalan = %s;\n"
        "   test_result = pow(M_PI, Catalan);\n"
        "   return test_result;\n"
        "}\n"
    ) % Catalan.evalf(17)
    # 断言生成的字符串与预期的字符串是否相等
    assert source == expected


# 测试生成 C 代码时参数顺序的影响
def test_c_code_argument_order():
    # 创建符号变量 x, y, z
    x, y, z = symbols('x,y,z')
    # 创建表达式 x + y
    expr = x + y
    # 制作一个名为 "test" 的例程，处理表达式 expr，指定参数顺序为 [z, x, y]
    routine = make_routine("test", expr, argument_sequence=[z, x, y])
    # 创建 C89 代码生成器的实例
    code_gen = C89CodeGen()
    # 调用 get_string 函数，生成生成的 C 代码字符串
    source = get_string(code_gen.dump_c, [routine])
    # 预期的生成的 C 代码字符串
    expected = (
        "#include \"file.h\"\n"
        "#include <math.h>\n"
        "double test(double z, double x, double y) {\n"
        "   double test_result;\n"
        "   test_result = x + y;\n"
        "   return test_result;\n"
        "}\n"
    )
    # 断言生成的字符串与预期的字符串是否相等
    assert source == expected


# 测试生成简单的 C 头文件
def test_simple_c_header():
    # 创建符号变量 x, y, z
    x, y, z = symbols('x,y,z')
    # 创建表达式 (x + y)*z
    expr = (x + y)*z
    # 制作一个名为 "test" 的例程，处理表达式 expr
    routine = make_routine("test", expr)
    # 创建 C89 代码生成器的实例
    code_gen = C89CodeGen()
    # 调用 get_string 函数，生成生成的 C 头文件字符串
    source = get_string(code_gen.dump_h, [routine])
    # 预期的生成的 C 头文件字符串
    expected = (
        "#ifndef PROJECT__FILE__H\n"
        "#define PROJECT__FILE__H\n"
        "double test(double x, double y, double z);\n"
        "#endif\n"
    )
    # 断言生成的字符串与预期的字符串是否相等
    assert source == expected


# 测试生成简单的 C 代码
def test_simple_c_codegen():
    # 创建符号变量 x, y, z
    x, y, z = symbols('x,y,z')
    # 创建表达式 (x + y)*z
    expr = (x + y)*z
    # 定义期望的输出结果，一个包含两个元组的列表
    expected = [
        ("file.c",  # 第一个元组，文件名为 "file.c"
        "#include \"file.h\"\n"  # 文件内容包括引用 "file.h" 的头文件
        "#include <math.h>\n"    # 包含数学函数库的头文件
        "double test(double x, double y, double z) {\n"  # 定义一个 C 函数 test
        "   double test_result;\n"  # 声明一个 double 类型的变量 test_result
        "   test_result = z*(x + y);\n"  # 计算 test_result 的值
        "   return test_result;\n"  # 返回计算结果
        "}\n"),  # 函数定义结束
        ("file.h",  # 第二个元组，文件名为 "file.h"
        "#ifndef PROJECT__FILE__H\n"  # 预处理指令，防止头文件重复包含
        "#define PROJECT__FILE__H\n"  # 定义头文件标志符
        "double test(double x, double y, double z);\n"  # 函数声明
        "#endif\n")  # 结束头文件声明部分
    ]
    # 调用函数 codegen 生成 C 代码，传入函数名 "test" 和表达式 expr，指定语言为 "C"，文件名前缀为 "file"，不包含头文件，不允许生成空文件
    result = codegen(("test", expr), "C", "file", header=False, empty=False)
    # 使用断言验证生成的代码与期望的输出结果是否一致
    assert result == expected
# 测试函数，用于验证多个结果的情况
def test_multiple_results_c():
    # 符号变量声明
    x, y, z = symbols('x,y,z')
    # 创建两个表达式
    expr1 = (x + y)*z
    expr2 = (x - y)*z
    # 调用函数生成例程，传入名称和表达式列表
    routine = make_routine(
        "test",
        [expr1, expr2]
    )
    # 创建 C99 代码生成器对象
    code_gen = C99CodeGen()
    # 使用 lambda 函数测试抛出 CodeGenError 异常
    raises(CodeGenError, lambda: get_string(code_gen.dump_h, [routine]))


# 测试函数，用于验证没有结果的情况
def test_no_results_c():
    # 使用 lambda 函数测试抛出 ValueError 异常
    raises(ValueError, lambda: make_routine("test", []))


# 测试函数，用于验证 ANSI C 数学函数代码生成
def test_ansi_math1_codegen():
    # 导入需要的数学函数
    # 未包含: log10
    from sympy.functions.elementary.complexes import Abs
    from sympy.functions.elementary.exponential import log
    from sympy.functions.elementary.hyperbolic import (cosh, sinh, tanh)
    from sympy.functions.elementary.integers import (ceiling, floor)
    from sympy.functions.elementary.miscellaneous import sqrt
    from sympy.functions.elementary.trigonometric import (acos, asin, atan, cos, sin, tan)
    # 声明符号变量 x
    x = symbols('x')
    # 定义名称和表达式的列表
    name_expr = [
        ("test_fabs", Abs(x)),
        ("test_acos", acos(x)),
        ("test_asin", asin(x)),
        ("test_atan", atan(x)),
        ("test_ceil", ceiling(x)),
        ("test_cos", cos(x)),
        ("test_cosh", cosh(x)),
        ("test_floor", floor(x)),
        ("test_log", log(x)),
        ("test_ln", log(x)),
        ("test_sin", sin(x)),
        ("test_sinh", sinh(x)),
        ("test_sqrt", sqrt(x)),
        ("test_tan", tan(x)),
        ("test_tanh", tanh(x)),
    ]
    # 调用 codegen 函数生成 C89 格式的代码文件，不生成头文件，文件不为空
    result = codegen(name_expr, "C89", "file", header=False, empty=False)
    # 断言检查生成的文件名是否为 "file.c"
    assert result[0][0] == "file.c"
    # 断言：验证 result 列表中第一个元素的第二个元素是否等于长字符串
    assert result[0][1] == (
        '#include "file.h"\n#include <math.h>\n'
        'double test_fabs(double x) {\n   double test_fabs_result;\n   test_fabs_result = fabs(x);\n   return test_fabs_result;\n}\n'
        'double test_acos(double x) {\n   double test_acos_result;\n   test_acos_result = acos(x);\n   return test_acos_result;\n}\n'
        'double test_asin(double x) {\n   double test_asin_result;\n   test_asin_result = asin(x);\n   return test_asin_result;\n}\n'
        'double test_atan(double x) {\n   double test_atan_result;\n   test_atan_result = atan(x);\n   return test_atan_result;\n}\n'
        'double test_ceil(double x) {\n   double test_ceil_result;\n   test_ceil_result = ceil(x);\n   return test_ceil_result;\n}\n'
        'double test_cos(double x) {\n   double test_cos_result;\n   test_cos_result = cos(x);\n   return test_cos_result;\n}\n'
        'double test_cosh(double x) {\n   double test_cosh_result;\n   test_cosh_result = cosh(x);\n   return test_cosh_result;\n}\n'
        'double test_floor(double x) {\n   double test_floor_result;\n   test_floor_result = floor(x);\n   return test_floor_result;\n}\n'
        'double test_log(double x) {\n   double test_log_result;\n   test_log_result = log(x);\n   return test_log_result;\n}\n'
        'double test_ln(double x) {\n   double test_ln_result;\n   test_ln_result = log(x);\n   return test_ln_result;\n}\n'
        'double test_sin(double x) {\n   double test_sin_result;\n   test_sin_result = sin(x);\n   return test_sin_result;\n}\n'
        'double test_sinh(double x) {\n   double test_sinh_result;\n   test_sinh_result = sinh(x);\n   return test_sinh_result;\n}\n'
        'double test_sqrt(double x) {\n   double test_sqrt_result;\n   test_sqrt_result = sqrt(x);\n   return test_sqrt_result;\n}\n'
        'double test_tan(double x) {\n   double test_tan_result;\n   test_tan_result = tan(x);\n   return test_tan_result;\n}\n'
        'double test_tanh(double x) {\n   double test_tanh_result;\n   test_tanh_result = tanh(x);\n   return test_tanh_result;\n}\n'
    )
    
    # 断言：验证 result 列表中第二个元素的第一个元素是否为 "file.h"
    assert result[1][0] == "file.h"
    
    # 断言：验证 result 列表中第二个元素的第二个元素是否等于长字符串
    assert result[1][1] == (
        '#ifndef PROJECT__FILE__H\n#define PROJECT__FILE__H\n'
        'double test_fabs(double x);\ndouble test_acos(double x);\n'
        'double test_asin(double x);\ndouble test_atan(double x);\n'
        'double test_ceil(double x);\ndouble test_cos(double x);\n'
        'double test_cosh(double x);\ndouble test_floor(double x);\n'
        'double test_log(double x);\ndouble test_ln(double x);\n'
        'double test_sin(double x);\ndouble test_sinh(double x);\n'
        'double test_sqrt(double x);\ndouble test_tan(double x);\n'
        'double test_tanh(double x);\n#endif\n'
    )
# 定义一个测试函数，用于生成特定数学函数的 C 代码
def test_complicated_codegen():
    # 导入三角函数相关模块中的 cos, sin, tan 函数
    from sympy.functions.elementary.trigonometric import (cos, sin, tan)
    # 定义符号变量 x, y, z
    x, y, z = symbols('x,y,z')
    # 定义表达式列表，每个元素是一个元组，包含测试函数名和对应的表达式
    name_expr = [
        ("test1", ((sin(x) + cos(y) + tan(z))**7).expand()),  # test1 函数的表达式
        ("test2", cos(cos(cos(cos(cos(cos(cos(cos(x + y + z))))))))),  # test2 函数的表达式
    ]
    # 调用 codegen 函数生成 C89 格式的代码，写入文件，不生成空函数，不生成头文件
    result = codegen(name_expr, "C89", "file", header=False, empty=False)
    # 断言结果列表的第一个元素是文件名为 "file.c"
    assert result[0][0] == "file.c"
    # 第一个断言：验证 result 列表中第一个元素的第二个元素是否等于一个长字符串
    assert result[0][1] == (
        '#include "file.h"\n#include <math.h>\n'
        'double test1(double x, double y, double z) {\n'
        '   double test1_result;\n'
        '   test1_result = '
        'pow(sin(x), 7) + '
        '7*pow(sin(x), 6)*cos(y) + '
        '7*pow(sin(x), 6)*tan(z) + '
        '21*pow(sin(x), 5)*pow(cos(y), 2) + '
        '42*pow(sin(x), 5)*cos(y)*tan(z) + '
        '21*pow(sin(x), 5)*pow(tan(z), 2) + '
        '35*pow(sin(x), 4)*pow(cos(y), 3) + '
        '105*pow(sin(x), 4)*pow(cos(y), 2)*tan(z) + '
        '105*pow(sin(x), 4)*cos(y)*pow(tan(z), 2) + '
        '35*pow(sin(x), 4)*pow(tan(z), 3) + '
        '35*pow(sin(x), 3)*pow(cos(y), 4) + '
        '140*pow(sin(x), 3)*pow(cos(y), 3)*tan(z) + '
        '210*pow(sin(x), 3)*pow(cos(y), 2)*pow(tan(z), 2) + '
        '140*pow(sin(x), 3)*cos(y)*pow(tan(z), 3) + '
        '35*pow(sin(x), 3)*pow(tan(z), 4) + '
        '21*pow(sin(x), 2)*pow(cos(y), 5) + '
        '105*pow(sin(x), 2)*pow(cos(y), 4)*tan(z) + '
        '210*pow(sin(x), 2)*pow(cos(y), 3)*pow(tan(z), 2) + '
        '210*pow(sin(x), 2)*pow(cos(y), 2)*pow(tan(z), 3) + '
        '105*pow(sin(x), 2)*cos(y)*pow(tan(z), 4) + '
        '21*pow(sin(x), 2)*pow(tan(z), 5) + '
        '7*sin(x)*pow(cos(y), 6) + '
        '42*sin(x)*pow(cos(y), 5)*tan(z) + '
        '105*sin(x)*pow(cos(y), 4)*pow(tan(z), 2) + '
        '140*sin(x)*pow(cos(y), 3)*pow(tan(z), 3) + '
        '105*sin(x)*pow(cos(y), 2)*pow(tan(z), 4) + '
        '42*sin(x)*cos(y)*pow(tan(z), 5) + '
        '7*sin(x)*pow(tan(z), 6) + '
        'pow(cos(y), 7) + '
        '7*pow(cos(y), 6)*tan(z) + '
        '21*pow(cos(y), 5)*pow(tan(z), 2) + '
        '35*pow(cos(y), 4)*pow(tan(z), 3) + '
        '35*pow(cos(y), 3)*pow(tan(z), 4) + '
        '21*pow(cos(y), 2)*pow(tan(z), 5) + '
        '7*cos(y)*pow(tan(z), 6) + '
        'pow(tan(z), 7);\n'
        '   return test1_result;\n'
        '}\n'
        'double test2(double x, double y, double z) {\n'
        '   double test2_result;\n'
        '   test2_result = cos(cos(cos(cos(cos(cos(cos(cos(x + y + z))))))));\n'
        '   return test2_result;\n'
        '}\n'
    )
    
    # 第二个断言：验证 result 列表中第二个元素的第一个元素是否等于 "file.h"
    assert result[1][0] == "file.h"
    
    # 第三个断言：验证 result 列表中第二个元素的第二个元素是否等于一个长字符串
    assert result[1][1] == (
        '#ifndef PROJECT__FILE__H\n'
        '#define PROJECT__FILE__H\n'
        'double test1(double x, double y, double z);\n'
        'double test2(double x, double y, double z);\n'
        '#endif\n'
    )
# 导入必要的符号和张量操作库
from sympy.tensor import IndexedBase, Idx
from sympy.core.symbol import symbols

# 定义符号变量 n, m 为整数
n, m = symbols('n m', integer=True)

# 创建 IndexedBase 对象 A, x, y 分别表示张量 A, x, y
A = IndexedBase('A')
x = IndexedBase('x')
y = IndexedBase('y')

# 创建索引变量 i, j 分别表示索引 i 和 j
i = Idx('i', m)
j = Idx('j', n)

# 调用 codegen 函数生成 C 语言代码，其中 'matrix_vector' 是函数名，Eq(y[i], A[i, j]*x[j]) 是函数表达式
(f1, code), (f2, interface) = codegen(
    ('matrix_vector', Eq(y[i], A[i, j]*x[j])), "C99", "file", header=False, empty=False)

# 断言生成的文件名为 'file.c'
assert f1 == 'file.c'

# 预期生成的 C 代码模板
expected = (
    '#include "file.h"\n'
    '#include <math.h>\n'
    'void matrix_vector(double *A, int m, int n, double *x, double *y) {\n'
    '   for (int i=0; i<m; i++){\n'
    '      y[i] = 0;\n'
    '   }\n'
    '   for (int i=0; i<m; i++){\n'
    '      for (int j=0; j<n; j++){\n'
    '         y[i] = %(rhs)s + y[i];\n'
    '      }\n'
    '   }\n'
    '}\n'
)

# 断言生成的代码满足预期的格式，使用 %(rhs)s 插入具体的右手边表达式
assert (code == expected % {'rhs': 'A[%s]*x[j]' % (i*n + j)} or
        code == expected % {'rhs': 'A[%s]*x[j]' % (j + i*n)} or
        code == expected % {'rhs': 'x[j]*A[%s]' % (i*n + j)} or
        code == expected % {'rhs': 'x[j]*A[%s]' % (j + i*n)})

# 断言生成的头文件名为 'file.h'
assert f2 == 'file.h'

# 验证接口代码符合预期格式
assert interface == (
    '#ifndef PROJECT__FILE__H\n'
    '#define PROJECT__FILE__H\n'
    'void matrix_vector(double *A, int m, int n, double *x, double *y);\n'
    '#endif\n'
)
    # 定义一个字符串变量 `expected`，包含C语言风格的函数实现代码模板
    expected = (
        '#include "file.h"\n'
        '#include <math.h>\n'
        'void matrix_vector(double *A, int m, int n, int o, int p, double *x, double *y) {\n'
        '   for (int i=o; i<%(upperi)s; i++){\n'
        '      y[i] = 0;\n'
        '   }\n'
        '   for (int i=o; i<%(upperi)s; i++){\n'
        '      for (int j=0; j<n; j++){\n'
        '         y[i] = %(rhs)s + y[i];\n'
        '      }\n'
        '   }\n'
        '}\n'
    ) % {'upperi': m - 4, 'rhs': '%(rhs)s'}

    # 使用预期输出字符串 `expected` 和占位符 `'rhs'` 的各种可能性，进行断言检查
    assert (code == expected % {'rhs': 'A[%s]*x[j]' % (i*p + j)} or
            code == expected % {'rhs': 'A[%s]*x[j]' % (j + i*p)} or
            code == expected % {'rhs': 'x[j]*A[%s]' % (i*p + j)} or
            code == expected % {'rhs': 'x[j]*A[%s]' % (j + i*p)})
    
    # 断言 `f2` 等于字符串 'file.h'
    assert f2 == 'file.h'
    
    # 断言 `interface` 等于预期输出的接口声明字符串
    assert interface == (
        '#ifndef PROJECT__FILE__H\n'
        '#define PROJECT__FILE__H\n'
        'void matrix_vector(double *A, int m, int n, int o, int p, double *x, double *y);\n'
        '#endif\n'
    )
# 引入符号操作库中的相关模块和函数
def test_output_arg_c():
    from sympy.core.relational import Equality  # 导入Equality符号关系
    from sympy.functions.elementary.trigonometric import (cos, sin)  # 导入cos和sin函数
    x, y, z = symbols("x,y,z")  # 定义符号变量x, y, z
    r = make_routine("foo", [Equality(y, sin(x)), cos(x)])  # 创建名为foo的例程，其中包含y=sin(x)和cos(x)的等式
    c = C89CodeGen()  # 创建C89代码生成器对象
    result = c.write([r], "test", header=False, empty=False)  # 生成名为test的C代码，不包含头文件，不允许为空
    assert result[0][0] == "test.c"  # 断言生成的第一个文件名为test.c
    expected = (
        '#include "test.h"\n'  # 引入test.h头文件
        '#include <math.h>\n'  # 引入math.h标准数学库
        'double foo(double x, double *y) {\n'  # 定义返回类型为double的foo函数，参数为x和y的指针
        '   (*y) = sin(x);\n'  # 计算sin(x)并将结果赋给y指向的变量
        '   double foo_result;\n'  # 声明变量foo_result，类型为double
        '   foo_result = cos(x);\n'  # 计算cos(x)并将结果赋给foo_result
        '   return foo_result;\n'  # 返回foo_result变量的值
        '}\n'  # 函数定义结束
    )
    assert result[0][1] == expected  # 断言生成的第一个文件内容符合预期


def test_output_arg_c_reserved_words():
    from sympy.core.relational import Equality  # 导入Equality符号关系
    from sympy.functions.elementary.trigonometric import (cos, sin)  # 导入cos和sin函数
    x, y, z = symbols("if, while, z")  # 定义符号变量if_, while_, z
    r = make_routine("foo", [Equality(y, sin(x)), cos(x)])  # 创建名为foo的例程，其中包含y=sin(if_)和cos(if_)的等式
    c = C89CodeGen()  # 创建C89代码生成器对象
    result = c.write([r], "test", header=False, empty=False)  # 生成名为test的C代码，不包含头文件，不允许为空
    assert result[0][0] == "test.c"  # 断言生成的第一个文件名为test.c
    expected = (
        '#include "test.h"\n'  # 引入test.h头文件
        '#include <math.h>\n'  # 引入math.h标准数学库
        'double foo(double if_, double *while_) {\n'  # 定义返回类型为double的foo函数，参数为if_和while_的指针
        '   (*while_) = sin(if_);\n'  # 计算sin(if_)并将结果赋给while_指向的变量
        '   double foo_result;\n'  # 声明变量foo_result，类型为double
        '   foo_result = cos(if_);\n'  # 计算cos(if_)并将结果赋给foo_result
        '   return foo_result;\n'  # 返回foo_result变量的值
        '}\n'  # 函数定义结束
    )
    assert result[0][1] == expected  # 断言生成的第一个文件内容符合预期


def test_multidim_c_argument_cse():
    A_sym = MatrixSymbol('A', 3, 3)  # 定义3x3矩阵符号A
    b_sym = MatrixSymbol('b', 3, 1)  # 定义3x1矩阵符号b
    A = Matrix(A_sym)  # 创建矩阵A的对象
    b = Matrix(b_sym)  # 创建矩阵b的对象
    c = A*b  # 计算矩阵乘法A*b
    cgen = CCodeGen(project="test", cse=True)  # 创建启用公共子表达式消除的C代码生成器对象
    r = cgen.routine("c", c)  # 生成名称为c的例程
    r.arguments[-1].result_var = "out"  # 将最后一个参数的结果变量设置为out
    r.arguments[-1]._name = "out"  # 将最后一个参数的名称设置为out
    code = get_string(cgen.dump_c, [r], prefix="test")  # 获取以test为前缀的C代码字符串
    expected = (
        '#include "test.h"\n'  # 引入test.h头文件
        "#include <math.h>\n"  # 引入math.h标准数学库
        "void c(double *A, double *b, double *out) {\n"  # 定义返回类型为void的c函数，参数为A、b和out的指针
        "   out[0] = A[0]*b[0] + A[1]*b[1] + A[2]*b[2];\n"  # 计算A*b的第一个元素并将结果存储在out数组的第一个位置
        "   out[1] = A[3]*b[0] + A[4]*b[1] + A[5]*b[2];\n"  # 计算A*b的第二个元素并将结果存储在out数组的第二个位置
        "   out[2] = A[6]*b[0] + A[7]*b[1] + A[8]*b[2];\n"  # 计算A*b的第三个元素并将结果存储在out数组的第三个位置
        "}\n"  # 函数定义结束
    )
    assert code == expected  # 断言生成的C代码字符串与预期的一致


def test_ccode_results_named_ordered():
    x, y, z = symbols('x,y,z')  # 定义符号变量x, y, z
    B, C = symbols('B,C')  # 定义符号变量B, C
    A = MatrixSymbol('A', 1, 3)  # 定义1x3矩阵符号A
    expr1 = Equality(A, Matrix([[1, 2, x]]))  # 定义A=[1, 2, x]的等式
    expr2 = Equality(C, (x + y)*z)  # 定义C=(x + y)*z的等式
    expr3 = Equality(B, 2*x)  # 定义B=2*x的等式
    name_expr = ("test", [expr1, expr2, expr3])  # 定义名称为test的表达式列表
    expected = (
        '#include "test.h"\n'  # 引入test.h头文件
        '#include <math.h>\n'  # 引入math.h标准数学库
        'void test(double x, double *C, double z, double y, double *A, double *B) {\n'  # 定义返回类型为void的test函数，参数包括x, C, z, y, A, B
        '   (*C) = z*(x + y);\n'  # 计算(z*(x + y))并将结果存储在*C指向的位置
        '   A[0] = 1;\n'  # 将1赋值给A数组的第一个位置
        '   A[1] = 2;\n'  # 将2赋值给A数组的第二个位置
        '   A[2] = x;\n'  # 将x赋值给A数组的第三个位置
        '   (*B) = 2*x;\n'  # 将(2*x)赋值给*B指向的位置
        '}\n'  # 函数定义结束
    )

    result = codegen(name_expr, "c", "test", header=False, empty=False,
                     argument_sequence=(x, C, z, y, A, B))  # 生成名称为test的C代码字符串
    source = result[0][1]  # 获取生成的C代码
    # 创建一个名为 C 的 1x3 矩阵符号
    C = MatrixSymbol('C', 1, 3)
    # 创建一个名为 D 的 5x1 矩阵符号
    D = MatrixSymbol('D', 5, 1)
    # 定义一个包含名称和表达式的元组，用于代码生成
    name_expr = ("test", [Equality(B, A[0, :]),
                          Equality(C, A[1, :]),
                          Equality(D, A[:, 2])])
    # 调用 codegen 函数生成 C99 格式的代码，不包含头文件，不允许空函数
    result = codegen(name_expr, "c99", "test", header=False, empty=False)
    # 获取生成的代码字符串
    source = result[0][1]
    # 预期生成的代码字符串
    expected = (
        '#include "test.h"\n'
        '#include <math.h>\n'
        'void test(double *A, double *B, double *C, double *D) {\n'
        '   B[0] = A[0];\n'
        '   B[1] = A[1];\n'
        '   B[2] = A[2];\n'
        '   C[0] = A[3];\n'
        '   C[1] = A[4];\n'
        '   C[2] = A[5];\n'
        '   D[0] = A[2];\n'
        '   D[1] = A[5];\n'
        '   D[2] = A[8];\n'
        '   D[3] = A[11];\n'
        '   D[4] = A[14];\n'
        '}\n'
    )
    # 断言生成的代码与预期的代码字符串相同
    assert source == expected
def test_ccode_cse():
    # 定义符号变量 a, b, c, d
    a, b, c, d = symbols('a b c d')
    # 定义一个 3x1 的矩阵符号变量 e
    e = MatrixSymbol('e', 3, 1)
    # 定义名字表达式，包括一个名称和一个等式表达式
    name_expr = ("test", [Equality(e, Matrix([[a*b], [a*b + c*d], [a*b*c*d]]))])
    # 创建一个 C 代码生成器对象，开启公共子表达式消除 (CSE)
    generator = CCodeGen(cse=True)
    # 生成代码，不包含头文件，不保留空函数
    result = codegen(name_expr, code_gen=generator, header=False, empty=False)
    # 获取生成的代码字符串
    source = result[0][1]
    # 期望的代码字符串
    expected = (
        '#include "test.h"\n'
        '#include <math.h>\n'
        'void test(double a, double b, double c, double d, double *e) {\n'
        '   const double x0 = a*b;\n'
        '   const double x1 = c*d;\n'
        '   e[0] = x0;\n'
        '   e[1] = x0 + x1;\n'
        '   e[2] = x0*x1;\n'
        '}\n'
    )
    # 断言生成的代码与期望的代码一致
    assert source == expected

def test_ccode_unused_array_arg():
    # 定义一个 2x1 的矩阵符号变量 x
    x = MatrixSymbol('x', 2, 1)
    # 名字表达式只包括一个名称和一个浮点数
    name_expr = ("test", 1.0)
    # 创建一个默认的 C 代码生成器对象
    generator = CCodeGen()
    # 生成代码，不包含头文件，不保留空函数，指定参数顺序为 (x,)
    result = codegen(name_expr, code_gen=generator, header=False, empty=False, argument_sequence=(x,))
    # 获取生成的代码字符串
    source = result[0][1]
    # 期望的代码字符串
    expected = (
        '#include "test.h"\n'
        '#include <math.h>\n'
        'double test(double *x) {\n'
        '   double test_result;\n'
        '   test_result = 1.0;\n'
        '   return test_result;\n'
        '}\n'
    )
    # 断言生成的代码与期望的代码一致
    assert source == expected

def test_ccode_unused_array_arg_func():
    # 定义 3x1 的两个矩阵符号变量 X 和 Y，以及一个整数符号变量 z
    X = MatrixSymbol('X', 3, 1)
    Y = MatrixSymbol('Y', 3, 1)
    z = symbols('z', integer=True)
    # 定义名字表达式，包括一个名称和一个符号表达式
    name_expr = ('testBug', X[0] + X[1])
    # 生成代码，指定语言为 C，不包含头文件，不保留空函数，指定参数顺序为 (X, Y, z)
    result = codegen(name_expr, language='C', header=False, empty=False, argument_sequence=(X, Y, z))
    # 获取生成的代码字符串
    source = result[0][1]
    # 期望的代码字符串
    expected = (
        '#include "testBug.h"\n'
        '#include <math.h>\n'
        'double testBug(double *X, double *Y, int z) {\n'
        '   double testBug_result;\n'
        '   testBug_result = X[0] + X[1];\n'
        '   return testBug_result;\n'
        '}\n'
    )
    # 断言生成的代码与期望的代码一致
    assert source == expected

def test_empty_f_code():
    # 创建一个 Fortran 95 代码生成器对象
    code_gen = FCodeGen()
    # 获取生成的空代码字符串
    source = get_string(code_gen.dump_f95, [])
    # 断言生成的代码为空字符串
    assert source == ""

def test_empty_f_code_with_header():
    # 创建一个 Fortran 95 代码生成器对象
    code_gen = FCodeGen()
    # 获取生成的带有头部的 Fortran 95 代码字符串
    source = get_string(code_gen.dump_f95, [], header=True)
    # 断言生成的代码前部分与期望的头部字符串一致
    assert source[:82] == (
        "!******************************************************************************\n!*"
    )
    # 断言生成的代码后部分与期望的尾部字符串一致
    assert source[158:] == (
        "*\n"
        "!*                                                                            *\n"
        "!*              See http://www.sympy.org/ for more information.               *\n"
        "!*                                                                            *\n"
        "!*                       This file is part of 'project'                       *\n"
        "!******************************************************************************\n"
    )
    # 创建 FCodeGen 的实例对象
    code_gen = FCodeGen()
    # 调用 code_gen 对象的 dump_h 方法并传入空列表作为参数，获取返回的字符串
    source = get_string(code_gen.dump_h, [])
    # 使用断言确保 source 变量的值为空字符串
    assert source == ""
def test_simple_f_code():
    # 定义符号变量 x, y, z
    x, y, z = symbols('x,y,z')
    # 构建表达式 (x + y)*z
    expr = (x + y)*z
    # 使用 make_routine 函数创建名为 "test" 的例程
    routine = make_routine("test", expr)
    # 创建 FCodeGen 对象
    code_gen = FCodeGen()
    # 生成 Fortran 95 代码字符串
    source = get_string(code_gen.dump_f95, [routine])
    # 期望的 Fortran 代码字符串
    expected = (
        "REAL*8 function test(x, y, z)\n"
        "implicit none\n"
        "REAL*8, intent(in) :: x\n"
        "REAL*8, intent(in) :: y\n"
        "REAL*8, intent(in) :: z\n"
        "test = z*(x + y)\n"
        "end function\n"
    )
    # 断言生成的代码与期望的代码一致
    assert source == expected


def test_numbersymbol_f_code():
    # 使用 make_routine 函数创建名为 "test" 的例程，表达式为 pi**Catalan
    routine = make_routine("test", pi**Catalan)
    # 创建 FCodeGen 对象
    code_gen = FCodeGen()
    # 生成 Fortran 95 代码字符串
    source = get_string(code_gen.dump_f95, [routine])
    # 格式化填充期望的 Fortran 代码字符串
    expected = (
        "REAL*8 function test()\n"
        "implicit none\n"
        "REAL*8, parameter :: Catalan = %sd0\n"
        "REAL*8, parameter :: pi = %sd0\n"
        "test = pi**Catalan\n"
        "end function\n"
    ) % (Catalan.evalf(17), pi.evalf(17))
    # 断言生成的代码与期望的代码一致
    assert source == expected


def test_erf_f_code():
    # 定义符号变量 x
    x = symbols('x')
    # 使用 make_routine 函数创建名为 "test" 的例程，表达式为 erf(x) - erf(-2 * x)
    routine = make_routine("test", erf(x) - erf(-2 * x))
    # 创建 FCodeGen 对象
    code_gen = FCodeGen()
    # 生成 Fortran 95 代码字符串
    source = get_string(code_gen.dump_f95, [routine])
    # 期望的 Fortran 代码字符串
    expected = (
        "REAL*8 function test(x)\n"
        "implicit none\n"
        "REAL*8, intent(in) :: x\n"
        "test = erf(x) + erf(2.0d0*x)\n"
        "end function\n"
    )
    # 断言生成的代码与期望的代码一致
    assert source == expected, source


def test_f_code_argument_order():
    # 定义符号变量 x, y, z
    x, y, z = symbols('x,y,z')
    # 定义表达式 x + y
    expr = x + y
    # 使用 make_routine 函数创建名为 "test" 的例程，使用指定的参数顺序 [z, x, y]
    routine = make_routine("test", expr, argument_sequence=[z, x, y])
    # 创建 FCodeGen 对象
    code_gen = FCodeGen()
    # 生成 Fortran 95 代码字符串
    source = get_string(code_gen.dump_f95, [routine])
    # 期望的 Fortran 代码字符串
    expected = (
        "REAL*8 function test(z, x, y)\n"
        "implicit none\n"
        "REAL*8, intent(in) :: z\n"
        "REAL*8, intent(in) :: x\n"
        "REAL*8, intent(in) :: y\n"
        "test = x + y\n"
        "end function\n"
    )
    # 断言生成的代码与期望的代码一致
    assert source == expected


def test_simple_f_header():
    # 定义符号变量 x, y, z
    x, y, z = symbols('x,y,z')
    # 定义表达式 (x + y)*z
    expr = (x + y)*z
    # 使用 make_routine 函数创建名为 "test" 的例程
    routine = make_routine("test", expr)
    # 创建 FCodeGen 对象
    code_gen = FCodeGen()
    # 生成 Fortran 95 头文件代码字符串
    source = get_string(code_gen.dump_h, [routine])
    # 期望的 Fortran 95 头文件代码字符串
    expected = (
        "interface\n"
        "REAL*8 function test(x, y, z)\n"
        "implicit none\n"
        "REAL*8, intent(in) :: x\n"
        "REAL*8, intent(in) :: y\n"
        "REAL*8, intent(in) :: z\n"
        "end function\n"
        "end interface\n"
    )
    # 断言生成的代码与期望的代码一致
    assert source == expected


def test_simple_f_codegen():
    # 定义符号变量 x, y, z
    x, y, z = symbols('x,y,z')
    # 定义表达式 (x + y)*z
    expr = (x + y)*z
    # 调用 codegen 函数生成代码，F95 语言，输出到文件，不生成空函数头
    result = codegen(
        ("test", expr), "F95", "file", header=False, empty=False)
    # 预期结果列表，包含了两个元组，每个元组分别表示一个文件名和其对应的内容
    expected = [
        ("file.f90",
        "REAL*8 function test(x, y, z)\n"
        "implicit none\n"
        "REAL*8, intent(in) :: x\n"
        "REAL*8, intent(in) :: y\n"
        "REAL*8, intent(in) :: z\n"
        "test = z*(x + y)\n"
        "end function\n"),
        ("file.h",
        "interface\n"
        "REAL*8 function test(x, y, z)\n"
        "implicit none\n"
        "REAL*8, intent(in) :: x\n"
        "REAL*8, intent(in) :: y\n"
        "REAL*8, intent(in) :: z\n"
        "end function\n"
        "end interface\n")
    ]
    # 断言结果与预期的输出相等
    assert result == expected
# 定义一个测试函数，测试多个表达式的代码生成是否引发 CodeGenError 异常
def test_multiple_results_f():
    # 定义符号变量 x, y, z
    x, y, z = symbols('x,y,z')
    # 创建两个表达式
    expr1 = (x + y)*z
    expr2 = (x - y)*z
    # 调用 make_routine 函数创建一个名为 "test" 的例程，包含上述两个表达式
    routine = make_routine(
        "test",
        [expr1, expr2]
    )
    # 创建一个 FCodeGen 实例
    code_gen = FCodeGen()
    # 断言调用 get_string 方法使用 dump_h 函数时会引发 CodeGenError 异常
    raises(CodeGenError, lambda: get_string(code_gen.dump_h, [routine]))


# 定义一个测试函数，测试当没有表达式输入时是否引发 ValueError 异常
def test_no_results_f():
    # 断言调用 make_routine 函数创建一个名为 "test" 的例程，不包含任何表达式时会引发 ValueError 异常
    raises(ValueError, lambda: make_routine("test", []))


# 定义一个测试函数，测试特定数学函数代码生成是否按预期工作
def test_intrinsic_math_codegen():
    # 导入数学函数模块，不包括 log10
    from sympy.functions.elementary.complexes import Abs
    from sympy.functions.elementary.exponential import log
    from sympy.functions.elementary.hyperbolic import (cosh, sinh, tanh)
    from sympy.functions.elementary.miscellaneous import sqrt
    from sympy.functions.elementary.trigonometric import (acos, asin, atan, cos, sin, tan)
    # 定义符号变量 x
    x = symbols('x')
    # 定义一个包含名称和表达式的列表
    name_expr = [
        ("test_abs", Abs(x)),
        ("test_acos", acos(x)),
        ("test_asin", asin(x)),
        ("test_atan", atan(x)),
        ("test_cos", cos(x)),
        ("test_cosh", cosh(x)),
        ("test_log", log(x)),
        ("test_ln", log(x)),  # ln 函数也是 log 函数的别名
        ("test_sin", sin(x)),
        ("test_sinh", sinh(x)),
        ("test_sqrt", sqrt(x)),
        ("test_tan", tan(x)),
        ("test_tanh", tanh(x)),
    ]
    # 调用 codegen 函数生成指定类型（"F95"）、输出到文件（"file"）的代码，不包含头部信息，不为空文件
    result = codegen(name_expr, "F95", "file", header=False, empty=False)
    # 断言结果列表中的第一个元素的第一个值为 "file.f90"
    assert result[0][0] == "file.f90"
    # 预期输出的字符串，包含多个 Fortran 函数的定义
    expected = (
        'REAL*8 function test_abs(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'test_abs = abs(x)\n'
        'end function\n'
        'REAL*8 function test_acos(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'test_acos = acos(x)\n'
        'end function\n'
        'REAL*8 function test_asin(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'test_asin = asin(x)\n'
        'end function\n'
        'REAL*8 function test_atan(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'test_atan = atan(x)\n'
        'end function\n'
        'REAL*8 function test_cos(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'test_cos = cos(x)\n'
        'end function\n'
        'REAL*8 function test_cosh(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'test_cosh = cosh(x)\n'
        'end function\n'
        'REAL*8 function test_log(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'test_log = log(x)\n'
        'end function\n'
        'REAL*8 function test_ln(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'test_ln = log(x)\n'
        'end function\n'
        'REAL*8 function test_sin(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'test_sin = sin(x)\n'
        'end function\n'
        'REAL*8 function test_sinh(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'test_sinh = sinh(x)\n'
        'end function\n'
        'REAL*8 function test_sqrt(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'test_sqrt = sqrt(x)\n'
        'end function\n'
        'REAL*8 function test_tan(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'test_tan = tan(x)\n'
        'end function\n'
        'REAL*8 function test_tanh(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'test_tanh = tanh(x)\n'
        'end function\n'
    )
    
    # 断言检查结果的第一个元素的第二个元素是否等于预期的字符串
    assert result[0][1] == expected
    
    # 断言检查结果的第二个元素的第一个元素是否等于字符串 "file.h"
    assert result[1][0] == "file.h"
    expected = (
        'interface\n'  # 定义接口部分的起始标志
        'REAL*8 function test_abs(x)\n'  # 定义函数 test_abs 的接口，接受一个 REAL*8 类型的参数 x
        'implicit none\n'  # 定义该函数的显式类型声明为 implicit none，即禁止隐式类型转换
        'REAL*8, intent(in) :: x\n'  # 声明函数 test_abs 的参数 x 为输入参数，类型为 REAL*8
        'end function\n'  # 结束函数 test_abs 的定义
        'end interface\n'  # 结束接口部分的定义
        'interface\n'  # 定义接口部分的起始标志（下同）
        'REAL*8 function test_acos(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'end function\n'
        'end interface\n'
        'interface\n'
        'REAL*8 function test_asin(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'end function\n'
        'end interface\n'
        'interface\n'
        'REAL*8 function test_atan(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'end function\n'
        'end interface\n'
        'interface\n'
        'REAL*8 function test_cos(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'end function\n'
        'end interface\n'
        'interface\n'
        'REAL*8 function test_cosh(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'end function\n'
        'end interface\n'
        'interface\n'
        'REAL*8 function test_log(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'end function\n'
        'end interface\n'
        'interface\n'
        'REAL*8 function test_ln(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'end function\n'
        'end interface\n'
        'interface\n'
        'REAL*8 function test_sin(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'end function\n'
        'end interface\n'
        'interface\n'
        'REAL*8 function test_sinh(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'end function\n'
        'end interface\n'
        'interface\n'
        'REAL*8 function test_sqrt(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'end function\n'
        'end interface\n'
        'interface\n'
        'REAL*8 function test_tan(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'end function\n'
        'end interface\n'
        'interface\n'
        'REAL*8 function test_tanh(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'end function\n'
        'end interface\n'
    )
    assert result[1][1] == expected  # 断言：检查 result 列表中的第二个元素的第二个元素是否等于 expected 字符串
def test_intrinsic_math2_codegen():
    # 导入所需的 atan2 函数
    from sympy.functions.elementary.trigonometric import atan2
    # 定义符号变量 x, y
    x, y = symbols('x,y')
    # 定义名称和表达式列表
    name_expr = [
        ("test_atan2", atan2(x, y)),  # 名称为 test_atan2 的 atan2 函数调用
        ("test_pow", x**y),           # 名称为 test_pow 的幂运算表达式
    ]
    # 生成代码，使用 F95 格式，生成到文件，不包含头部信息，不允许为空
    result = codegen(name_expr, "F95", "file", header=False, empty=False)
    # 断言生成的第一个文件名应为 "file.f90"
    assert result[0][0] == "file.f90"
    # 预期生成的代码内容
    expected = (
        'REAL*8 function test_atan2(x, y)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'REAL*8, intent(in) :: y\n'
        'test_atan2 = atan2(x, y)\n'
        'end function\n'
        'REAL*8 function test_pow(x, y)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'REAL*8, intent(in) :: y\n'
        'test_pow = x**y\n'
        'end function\n'
    )
    # 断言生成的第一个文件内容符合预期
    assert result[0][1] == expected

    # 断言生成的第二个文件名应为 "file.h"
    assert result[1][0] == "file.h"
    # 预期生成的代码内容
    expected = (
        'interface\n'
        'REAL*8 function test_atan2(x, y)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'REAL*8, intent(in) :: y\n'
        'end function\n'
        'end interface\n'
        'interface\n'
        'REAL*8 function test_pow(x, y)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'REAL*8, intent(in) :: y\n'
        'end function\n'
        'end interface\n'
    )
    # 断言生成的第二个文件内容符合预期
    assert result[1][1] == expected


def test_complicated_codegen_f95():
    # 导入所需的三角函数：cos, sin, tan
    from sympy.functions.elementary.trigonometric import (cos, sin, tan)
    # 定义符号变量 x, y, z
    x, y, z = symbols('x,y,z')
    # 定义名称和表达式列表
    name_expr = [
        ("test1", ((sin(x) + cos(y) + tan(z))**7).expand()),  # 名称为 test1 的复杂表达式
        ("test2", cos(cos(cos(cos(cos(cos(cos(cos(x + y + z))))))))),  # 名称为 test2 的嵌套余弦函数调用
    ]
    # 生成代码，使用 F95 格式，生成到文件，不包含头部信息，不允许为空
    result = codegen(name_expr, "F95", "file", header=False, empty=False)
    # 断言生成的第一个文件名应为 "file.f90"
    assert result[0][0] == "file.f90"
    # 定义一个长字符串，包含了两个 Fortran 函数的原型和实现，以及相关的隐式声明
    expected = (
        'REAL*8 function test1(x, y, z)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'REAL*8, intent(in) :: y\n'
        'REAL*8, intent(in) :: z\n'
        'test1 = sin(x)**7 + 7*sin(x)**6*cos(y) + 7*sin(x)**6*tan(z) + 21*sin(x) &\n'
        '      **5*cos(y)**2 + 42*sin(x)**5*cos(y)*tan(z) + 21*sin(x)**5*tan(z) &\n'
        '      **2 + 35*sin(x)**4*cos(y)**3 + 105*sin(x)**4*cos(y)**2*tan(z) + &\n'
        '      105*sin(x)**4*cos(y)*tan(z)**2 + 35*sin(x)**4*tan(z)**3 + 35*sin( &\n'
        '      x)**3*cos(y)**4 + 140*sin(x)**3*cos(y)**3*tan(z) + 210*sin(x)**3* &\n'
        '      cos(y)**2*tan(z)**2 + 140*sin(x)**3*cos(y)*tan(z)**3 + 35*sin(x) &\n'
        '      **3*tan(z)**4 + 21*sin(x)**2*cos(y)**5 + 105*sin(x)**2*cos(y)**4* &\n'
        '      tan(z) + 210*sin(x)**2*cos(y)**3*tan(z)**2 + 210*sin(x)**2*cos(y) &\n'
        '      **2*tan(z)**3 + 105*sin(x)**2*cos(y)*tan(z)**4 + 21*sin(x)**2*tan &\n'
        '      (z)**5 + 7*sin(x)*cos(y)**6 + 42*sin(x)*cos(y)**5*tan(z) + 105* &\n'
        '      sin(x)*cos(y)**4*tan(z)**2 + 140*sin(x)*cos(y)**3*tan(z)**3 + 105 &\n'
        '      *sin(x)*cos(y)**2*tan(z)**4 + 42*sin(x)*cos(y)*tan(z)**5 + 7*sin( &\n'
        '      x)*tan(z)**6 + cos(y)**7 + 7*cos(y)**6*tan(z) + 21*cos(y)**5*tan( &\n'
        '      z)**2 + 35*cos(y)**4*tan(z)**3 + 35*cos(y)**3*tan(z)**4 + 21*cos( &\n'
        '      y)**2*tan(z)**5 + 7*cos(y)*tan(z)**6 + tan(z)**7\n'
        'end function\n'
        'REAL*8 function test2(x, y, z)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'REAL*8, intent(in) :: y\n'
        'REAL*8, intent(in) :: z\n'
        'test2 = cos(cos(cos(cos(cos(cos(cos(cos(x + y + z))))))))\n'
        'end function\n'
    )
    
    # 断言结果列表中的第一个元素的第二个元素等于预期的字符串
    assert result[0][1] == expected
    
    # 断言结果列表中的第二个元素的第一个元素等于 "file.h"
    assert result[1][0] == "file.h"
    
    # 定义另一个长字符串，包含了两个 Fortran 函数的接口声明
    expected = (
        'interface\n'
        'REAL*8 function test1(x, y, z)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'REAL*8, intent(in) :: y\n'
        'REAL*8, intent(in) :: z\n'
        'end function\n'
        'end interface\n'
        'interface\n'
        'REAL*8 function test2(x, y, z)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'REAL*8, intent(in) :: y\n'
        'REAL*8, intent(in) :: z\n'
        'end function\n'
        'end interface\n'
    )
    
    # 断言结果列表中的第二个元素的第二个元素等于预期的字符串
    assert result[1][1] == expected
# 导入必要的符号和张量操作模块
from sympy.tensor import IndexedBase, Idx
from sympy.core.symbol import symbols

# 声明整数符号变量 n, m
n, m = symbols('n,m', integer=True)
# 创建 IndexedBase 对象 A, x, y 分别代表索引数组 A, x, y
A, x, y = map(IndexedBase, 'Axy')
# 创建索引对象 i, j 分别表示索引变量 i, j
i = Idx('i', m)
j = Idx('j', n)

# 使用 codegen 函数生成两个代码文件
(f1, code), (f2, interface) = codegen(
    ('matrix_vector', Eq(y[i], A[i, j]*x[j])), "F95", "file", header=False, empty=False)

# 断言生成的文件名为 'file.f90'
assert f1 == 'file.f90'
# 期望的 Fortran 95 代码格式
expected = (
    'subroutine matrix_vector(A, m, n, x, y)\n'
    'implicit none\n'
    'INTEGER*4, intent(in) :: m\n'
    'INTEGER*4, intent(in) :: n\n'
    'REAL*8, intent(in), dimension(1:m, 1:n) :: A\n'
    'REAL*8, intent(in), dimension(1:n) :: x\n'
    'REAL*8, intent(out), dimension(1:m) :: y\n'
    'INTEGER*4 :: i\n'
    'INTEGER*4 :: j\n'
    'do i = 1, m\n'
    '   y(i) = 0\n'
    'end do\n'
    'do i = 1, m\n'
    '   do j = 1, n\n'
    '      y(i) = %(rhs)s + y(i)\n'
    '   end do\n'
    'end do\n'
    'end subroutine\n'
)

# 断言生成的代码与期望的代码一致，替换 %(rhs)s 部分为 'A(i, j)*x(j)' 或 'x(j)*A(i, j)'
assert code == expected % {'rhs': 'A(i, j)*x(j)'} or\
    code == expected % {'rhs': 'x(j)*A(i, j)'}
# 断言生成的接口文件名为 'file.h'
assert f2 == 'file.h'
# 断言生成的接口代码与期望的代码一致
assert interface == (
    'interface\n'
    'subroutine matrix_vector(A, m, n, x, y)\n'
    'implicit none\n'
    'INTEGER*4, intent(in) :: m\n'
    'INTEGER*4, intent(in) :: n\n'
    'REAL*8, intent(in), dimension(1:m, 1:n) :: A\n'
    'REAL*8, intent(in), dimension(1:n) :: x\n'
    'REAL*8, intent(out), dimension(1:m) :: y\n'
    'end subroutine\n'
    'end interface\n'
)
    # 定义预期的 Fortran 子程序代码模板，包含参数和循环结构
    expected = (
        'subroutine matrix_vector(A, m, n, x, y)\n'
        'implicit none\n'
        'INTEGER*4, intent(in) :: m\n'
        'INTEGER*4, intent(in) :: n\n'
        'REAL*8, intent(in), dimension(1:m, 1:n) :: A\n'
        'REAL*8, intent(in), dimension(1:n) :: x\n'
        'REAL*8, intent(inout), dimension(1:m) :: y\n'
        'INTEGER*4 :: i\n'
        'INTEGER*4 :: j\n'
        'do i = 1, m\n'
        '   do j = 1, n\n'
        '      y(i) = %(rhs)s + y(i)\n'  # 将 rhs 表达式加到 y(i) 中
        '   end do\n'
        'end do\n'
        'end subroutine\n'
    )

    # 断言检查生成的代码是否符合预期，允许两种不同的 rhs 表达式格式
    assert (code == expected % {'rhs': 'A(i, j)*x(j)'} or
            code == expected % {'rhs': 'x(j)*A(i, j)'})
    
    # 断言检查 f2 变量是否为 'file.h'
    assert f2 == 'file.h'
    
    # 定义预期的 Fortran 接口代码，用于声明 matrix_vector 子程序的接口
    assert interface == (
        'interface\n'
        'subroutine matrix_vector(A, m, n, x, y)\n'
        'implicit none\n'
        'INTEGER*4, intent(in) :: m\n'
        'INTEGER*4, intent(in) :: n\n'
        'REAL*8, intent(in), dimension(1:m, 1:n) :: A\n'
        'REAL*8, intent(in), dimension(1:n) :: x\n'
        'REAL*8, intent(inout), dimension(1:m) :: y\n'
        'end subroutine\n'
        'end interface\n'
    )
def test_partial_loops_f():
    # 检查循环边界由 Idx 确定，数组步长由 IndexedBase 对象的形状确定。
    from sympy.tensor import IndexedBase, Idx  # 导入 IndexedBase 和 Idx 类
    from sympy.core.symbol import symbols  # 导入 symbols 函数
    n, m, o, p = symbols('n m o p', integer=True)  # 定义符号变量 n, m, o, p，并声明为整数类型
    A = IndexedBase('A', shape=(m, p))  # 创建 IndexedBase 对象 A，指定形状为 (m, p)
    x = IndexedBase('x')  # 创建 IndexedBase 对象 x
    y = IndexedBase('y')  # 创建 IndexedBase 对象 y
    i = Idx('i', (o, m - 5))  # 创建 Idx 对象 i，指定范围为 (o, m - 5)，注意范围是包含的
    j = Idx('j', n)  # 创建 Idx 对象 j，维度 n 对应的范围是 (0, n - 1)

    (f1, code), (f2, interface) = codegen(
        ('matrix_vector', Eq(y[i], A[i, j]*x[j])), "F95", "file", header=False, empty=False)

    expected = (
        'subroutine matrix_vector(A, m, n, o, p, x, y)\n'
        'implicit none\n'
        'INTEGER*4, intent(in) :: m\n'
        'INTEGER*4, intent(in) :: n\n'
        'INTEGER*4, intent(in) :: o\n'
        'INTEGER*4, intent(in) :: p\n'
        'REAL*8, intent(in), dimension(1:m, 1:p) :: A\n'
        'REAL*8, intent(in), dimension(1:n) :: x\n'
        'REAL*8, intent(out), dimension(1:%(iup-ilow)s) :: y\n'
        'INTEGER*4 :: i\n'
        'INTEGER*4 :: j\n'
        'do i = %(ilow)s, %(iup)s\n'
        '   y(i) = 0\n'
        'end do\n'
        'do i = %(ilow)s, %(iup)s\n'
        '   do j = 1, n\n'
        '      y(i) = %(rhs)s + y(i)\n'
        '   end do\n'
        'end do\n'
        'end subroutine\n'
    ) % {
        'rhs': '%(rhs)s',
        'iup': str(m - 4),  # i 的上界
        'ilow': str(1 + o),  # i 的下界
        'iup-ilow': str(m - 4 - o)  # i 的上界减去下界
    }

    assert code == expected % {'rhs': 'A(i, j)*x(j)'} or\
        code == expected % {'rhs': 'x(j)*A(i, j)'}


def test_output_arg_f():
    # 导入必要的函数和类
    from sympy.core.relational import Equality
    from sympy.functions.elementary.trigonometric import (cos, sin)
    x, y, z = symbols("x,y,z")  # 定义符号变量 x, y, z
    r = make_routine("foo", [Equality(y, sin(x)), cos(x)])  # 创建名为 "foo" 的例程
    c = FCodeGen()  # 创建 Fortran 代码生成器实例
    result = c.write([r], "test", header=False, empty=False)  # 生成代码并返回结果
    assert result[0][0] == "test.f90"  # 检查生成的文件名
    assert result[0][1] == (
        'REAL*8 function foo(x, y)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'REAL*8, intent(out) :: y\n'
        'y = sin(x)\n'
        'foo = cos(x)\n'
        'end function\n'
    )


def test_inline_function():
    # 导入必要的函数和类
    from sympy.tensor import IndexedBase, Idx
    from sympy.core.symbol import symbols
    n, m = symbols('n m', integer=True)  # 定义符号变量 n, m，并声明为整数类型
    A, x, y = map(IndexedBase, 'Axy')  # 创建 IndexedBase 对象 A, x, y
    i = Idx('i', m)  # 创建 Idx 对象 i，范围为 (1, m)

    p = FCodeGen()  # 创建 Fortran 代码生成器实例
    func = implemented_function('func', Lambda(n, n*(n + 1)))  # 创建实现函数对象 func
    routine = make_routine('test_inline', Eq(y[i], func(x[i])))  # 创建名为 'test_inline' 的例程
    code = get_string(p.dump_f95, [routine])  # 获取生成的 Fortran 代码字符串
    expected = (
        'subroutine test_inline(m, x, y)\n'
        'implicit none\n'
        'INTEGER*4, intent(in) :: m\n'
        'REAL*8, intent(in), dimension(1:m) :: x\n'
        'REAL*8, intent(out), dimension(1:m) :: y\n'
        'INTEGER*4 :: i\n'
        'do i = 1, m\n'
        '   y(i) = %s*%s\n'  # y(i) 被赋值为 func(x(i))
        'end do\n'
        'end subroutine\n'
    )
    args = ('x(i)', '(x(i) + 1)')
    # 定义一个元组 args，包含两个字符串元素：'x(i)' 和 '(x(i) + 1)'
    assert code == expected % args or\
        code == expected % args[::-1]
    # 执行断言：检查 code 是否等于 expected 格式化后的结果，格式化参数分别是 args 和 args 的反转。
def test_f_code_call_signature_wrap():
    # 定义一个测试函数，用于测试函数签名包装功能
    # Issue #7934

    # 创建一个包含20个符号变量的符号列表 'x'，每个符号变量名为 'x0' 到 'x19'
    x = symbols('x:20')

    # 初始化一个表达式 expr 为 0
    expr = 0

    # 遍历符号列表 x 中的每个符号变量 sym
    for sym in x:
        # 将每个符号变量加到表达式 expr 上
        expr += sym

    # 使用表达式 expr 创建一个名为 'test' 的例行程序
    routine = make_routine("test", expr)

    # 创建一个 Fortran 代码生成器对象
    code_gen = FCodeGen()

    # 生成一个字符串，表示 Fortran 95 代码，使用例行程序 routine
    source = get_string(code_gen.dump_f95, [routine])

    # 预期的 Fortran 95 代码，用于比较生成的代码
    expected = """\
REAL*8 function test(x0, x1, x10, x11, x12, x13, x14, x15, x16, x17, x18, &
      x19, x2, x3, x4, x5, x6, x7, x8, x9)
implicit none
REAL*8, intent(in) :: x0
REAL*8, intent(in) :: x1
REAL*8, intent(in) :: x10
REAL*8, intent(in) :: x11
REAL*8, intent(in) :: x12
REAL*8, intent(in) :: x13
REAL*8, intent(in) :: x14
REAL*8, intent(in) :: x15
REAL*8, intent(in) :: x16
REAL*8, intent(in) :: x17
REAL*8, intent(in) :: x18
REAL*8, intent(in) :: x19
REAL*8, intent(in) :: x2
REAL*8, intent(in) :: x3
REAL*8, intent(in) :: x4
REAL*8, intent(in) :: x5
REAL*8, intent(in) :: x6
REAL*8, intent(in) :: x7
REAL*8, intent(in) :: x8
REAL*8, intent(in) :: x9
test = x0 + x1 + x10 + x11 + x12 + x13 + x14 + x15 + x16 + x17 + x18 + &
      x19 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9
end function
"""

    # 断言生成的代码与预期的代码相同
    assert source == expected


def test_check_case():
    # 定义一个测试函数，用于检查符号大小写异常处理

    # 创建两个符号变量 x 和 X
    x, X = symbols('x,X')

    # 使用 codegen 函数预期引发 CodeGenError 异常
    raises(CodeGenError, lambda: codegen(('test', x*X), 'f95', 'prefix'))


def test_check_case_false_positive():
    # 定义一个测试函数，用于检查符号大小写异常处理的误报情况

    # 对 SymPy 对象进行比较，检测是否触发了大小写异常
    # 这里的代码只用于检测与大小写检查相关的误报情况
    x1 = symbols('x')
    x2 = symbols('x', my_assumption=True)
    try:
        # 使用 codegen 函数预期引发 CodeGenError 异常
        codegen(('test', x1*x2), 'f95', 'prefix')
    except CodeGenError as e:
        if e.args[0].startswith("Fortran ignores case."):
            raise AssertionError("This exception should not be raised!")


def test_c_fortran_omit_routine_name():
    # 定义一个测试函数，用于检查 C 和 Fortran 代码生成，省略例行程序名称时的情况

    # 创建两个符号变量 x 和 y
    x, y = symbols("x,y")

    # 定义一个名称-表达式对列表 name_expr
    name_expr = [("foo", 2*x)]

    # 使用 codegen 函数生成 Fortran 95 代码，不包含头部和空行
    result = codegen(name_expr, "F95", header=False, empty=False)

    # 期望的结果，与上面生成的代码进行比较
    expresult = codegen(name_expr, "F95", "foo", header=False, empty=False)

    # 断言生成的结果与期望的结果相同
    assert result[0][1] == expresult[0][1]

    # 重新定义 name_expr
    name_expr = ("foo", x*y)

    # 使用 codegen 函数生成 Fortran 95 代码，不包含头部和空行
    result = codegen(name_expr, "F95", header=False, empty=False)

    # 期望的结果，与上面生成的代码进行比较
    expresult = codegen(name_expr, "F95", "foo", header=False, empty=False)

    # 断言生成的结果与期望的结果相同
    assert result[0][1] == expresult[0][1]

    # 重新定义 name_expr
    name_expr = ("foo", Matrix([[x, y], [x+y, x-y]]))

    # 使用 codegen 函数生成 C89 代码，不包含头部和空行
    result = codegen(name_expr, "C89", header=False, empty=False)

    # 期望的结果，与上面生成的代码进行比较
    expresult = codegen(name_expr, "C89", "foo", header=False, empty=False)

    # 断言生成的结果与期望的结果相同
    assert result[0][1] == expresult[0][1]


def test_fcode_matrix_output():
    # 定义一个测试函数，用于测试生成矩阵输出的 Fortran 代码

    # 创建三个符号变量 x, y, z
    x, y, z = symbols('x,y,z')

    # 定义表达式 e1 和矩阵表达式 e2
    e1 = x + y
    e2 = Matrix([[x, y], [z, 16]])

    # 定义名称-表达式对 name_expr
    name_expr = ("test", (e1, e2))

    # 使用 codegen 函数生成 Fortran 95 代码，包含头部，不包含空行
    result = codegen(name_expr, "f95", "test", header=False, empty=False)

    # 获取生成的 Fortran 代码
    source = result[0][1]
    expected = (
        "REAL*8 function test(x, y, z, out_%(hash)s)\n"  # 定义一个包含格式化字符串的函数签名模板
        "implicit none\n"  # 声明无隐式类型变量
        "REAL*8, intent(in) :: x\n"  # 定义输入参数 x，实数类型
        "REAL*8, intent(in) :: y\n"  # 定义输入参数 y，实数类型
        "REAL*8, intent(in) :: z\n"  # 定义输入参数 z，实数类型
        "REAL*8, intent(out), dimension(1:2, 1:2) :: out_%(hash)s\n"  # 定义输出参数 out_加上哈希值，是一个 2x2 的实数数组
        "out_%(hash)s(1, 1) = x\n"  # 设置输出数组第一个元素为 x
        "out_%(hash)s(2, 1) = z\n"  # 设置输出数组第二个元素为 z
        "out_%(hash)s(1, 2) = y\n"  # 设置输出数组第三个元素为 y
        "out_%(hash)s(2, 2) = 16\n"  # 设置输出数组第四个元素为 16
        "test = x + y\n"  # 函数返回值为 x + y 的和
        "end function\n"  # 函数定义结束
    )
    # 在源代码中查找第 6 行的内容
    a = source.splitlines()[5]
    # 使用下划线分割字符串 a
    b = a.split('_')
    # 取出分割后的第二部分作为哈希值
    out = b[1]
    # 将期望的函数定义模板中的 %(hash)s 替换为实际的哈希值
    expected = expected % {'hash': out}
    # 断言源代码与期望的函数定义模板相等
    assert source == expected
# 定义一个测试函数，测试生成的 Fortran 95 代码是否符合预期命名和顺序
def test_fcode_results_named_ordered():
    # 定义符号变量 x, y, z
    x, y, z = symbols('x,y,z')
    # 定义符号变量 B, C
    B, C = symbols('B,C')
    # 定义一个 1x3 维度的矩阵符号 A
    A = MatrixSymbol('A', 1, 3)
    # 构建表达式 expr1 表示 A = [1, 2, x]
    expr1 = Equality(A, Matrix([[1, 2, x]]))
    # 构建表达式 expr2 表示 C = (x + y) * z
    expr2 = Equality(C, (x + y) * z)
    # 构建表达式 expr3 表示 B = 2*x
    expr3 = Equality(B, 2*x)
    # 将所有表达式放入元组 name_expr
    name_expr = ("test", [expr1, expr2, expr3])
    # 调用 codegen 函数生成 Fortran 95 代码，并指定相关参数
    result = codegen(name_expr, "f95", "test", header=False, empty=False,
                     argument_sequence=(x, z, y, C, A, B))
    # 获取生成的代码字符串
    source = result[0][1]
    # 预期的 Fortran 95 代码字符串
    expected = (
        "subroutine test(x, z, y, C, A, B)\n"
        "implicit none\n"
        "REAL*8, intent(in) :: x\n"
        "REAL*8, intent(in) :: z\n"
        "REAL*8, intent(in) :: y\n"
        "REAL*8, intent(out) :: C\n"
        "REAL*8, intent(out) :: B\n"
        "REAL*8, intent(out), dimension(1:1, 1:3) :: A\n"
        "C = z*(x + y)\n"
        "A(1, 1) = 1\n"
        "A(1, 2) = 2\n"
        "A(1, 3) = x\n"
        "B = 2*x\n"
        "end subroutine\n"
    )
    # 断言生成的代码与预期的代码相同
    assert source == expected


# 定义一个测试函数，测试矩阵符号的切片操作生成的 Fortran 95 代码是否符合预期
def test_fcode_matrixsymbol_slice():
    # 定义一个 2x3 维度的矩阵符号 A
    A = MatrixSymbol('A', 2, 3)
    # 定义 1x3 维度的矩阵符号 B, C
    B = MatrixSymbol('B', 1, 3)
    C = MatrixSymbol('C', 1, 3)
    # 定义 2x1 维度的矩阵符号 D
    D = MatrixSymbol('D', 2, 1)
    # 将矩阵符号和切片表达式放入元组 name_expr
    name_expr = ("test", [Equality(B, A[0, :]),
                          Equality(C, A[1, :]),
                          Equality(D, A[:, 2])])
    # 调用 codegen 函数生成 Fortran 95 代码，并指定相关参数
    result = codegen(name_expr, "f95", "test", header=False, empty=False)
    # 获取生成的代码字符串
    source = result[0][1]
    # 预期的 Fortran 95 代码字符串
    expected = (
        "subroutine test(A, B, C, D)\n"
        "implicit none\n"
        "REAL*8, intent(in), dimension(1:2, 1:3) :: A\n"
        "REAL*8, intent(out), dimension(1:1, 1:3) :: B\n"
        "REAL*8, intent(out), dimension(1:1, 1:3) :: C\n"
        "REAL*8, intent(out), dimension(1:2, 1:1) :: D\n"
        "B(1, 1) = A(1, 1)\n"
        "B(1, 2) = A(1, 2)\n"
        "B(1, 3) = A(1, 3)\n"
        "C(1, 1) = A(2, 1)\n"
        "C(1, 2) = A(2, 2)\n"
        "C(1, 3) = A(2, 3)\n"
        "D(1, 1) = A(1, 3)\n"
        "D(2, 1) = A(2, 3)\n"
        "end subroutine\n"
    )
    # 断言生成的代码与预期的代码相同
    assert source == expected


# 定义一个测试函数，测试矩阵符号的切片操作（自动命名）生成的 Fortran 95 代码是否符合预期
def test_fcode_matrixsymbol_slice_autoname():
    # 定义一个 2x3 维度的矩阵符号 A
    A = MatrixSymbol('A', 2, 3)
    # 将矩阵符号和切片表达式放入元组 name_expr
    name_expr = ("test", A[:, 1])
    # 调用 codegen 函数生成 Fortran 95 代码，并指定相关参数
    result = codegen(name_expr, "f95", "test", header=False, empty=False)
    # 获取生成的代码字符串
    source = result[0][1]
    # 预期的 Fortran 95 代码字符串
    expected = (
        "subroutine test(A, out_%(hash)s)\n"
        "implicit none\n"
        "REAL*8, intent(in), dimension(1:2, 1:3) :: A\n"
        "REAL*8, intent(out), dimension(1:2, 1:1) :: out_%(hash)s\n"
        "out_%(hash)s(1, 1) = A(1, 2)\n"
        "out_%(hash)s(2, 1) = A(2, 2)\n"
        "end subroutine\n"
    )
    # 获取生成代码中的特殊哈希值
    a = source.splitlines()[3]
    b = a.split('_')
    out = b[1]
    # 替换预期的代码中的哈希值部分
    expected = expected % {'hash': out}
    # 断言生成的代码与预期的代码相同
    assert source == expected


# 定义一个测试函数，测试全局变量在生成的 Fortran 95 代码中的使用情况
def test_global_vars():
    # 定义符号变量 x, y, z, t
    x, y, z, t = symbols("x y z t")
    # 调用 codegen 函数生成与全局变量 y 相关的 Fortran 95 代码，并指定相关参数
    result = codegen(('f', x*y), "F95", header=False, empty=False,
                     global_vars=(y,))
    # 获取生成的代码字符串
    source = result[0][1]
    # 定义预期的源代码字符串，描述一个 Fortran 函数 f(x)，返回 x*y 的结果
    expected = (
        "REAL*8 function f(x)\n"
        "implicit none\n"
        "REAL*8, intent(in) :: x\n"
        "f = x*y\n"
        "end function\n"
        )
    # 断言生成的源代码与预期的源代码相同
    assert source == expected

    # 定义预期的源代码字符串，描述一个 C 函数 f(x, y)，返回 x*y + z 的结果
    expected = (
        '#include "f.h"\n'
        '#include <math.h>\n'
        'double f(double x, double y) {\n'
        '   double f_result;\n'
        '   f_result = x*y + z;\n'
        '   return f_result;\n'
        '}\n'
    )
    # 调用代码生成器生成 C 语言代码，将函数 f(x*y+z) 生成为字符串，不包含头文件，不为空函数
    result = codegen(('f', x*y+z), "C", header=False, empty=False,
                     global_vars=(z, t))
    # 获取生成的源代码字符串
    source = result[0][1]
    # 断言生成的源代码与预期的源代码相同
    assert source == expected
def test_custom_codegen():
    # 导入所需的模块和函数
    from sympy.printing.c import C99CodePrinter
    from sympy.functions.elementary.exponential import exp

    # 创建 C99 代码打印机对象，设置自定义的函数映射
    printer = C99CodePrinter(settings={'user_functions': {'exp': 'fastexp'}})

    # 创建符号变量 x 和 y
    x, y = symbols('x y')
    # 创建表达式 exp(x + y)
    expr = exp(x + y)

    # 创建 C99CodeGen 代码生成器对象，指定预处理语句为 '#include "fastexp.h"'
    gen = C99CodeGen(printer=printer,
                     preprocessor_statements=['#include "fastexp.h"'])

    # 期望的输出字符串
    expected = (
        '#include "expr.h"\n'
        '#include "fastexp.h"\n'
        'double expr(double x, double y) {\n'
        '   double expr_result;\n'
        '   expr_result = fastexp(x + y);\n'
        '   return expr_result;\n'
        '}\n'
    )

    # 生成代码并比较结果
    result = codegen(('expr', expr), header=False, empty=False, code_gen=gen)
    source = result[0][1]
    assert source == expected

    # 使用 math.h 和外部头文件的预处理语句
    gen = C99CodeGen(printer=printer)
    gen.preprocessor_statements.append('#include "fastexp.h"')

    # 更新期望的输出字符串
    expected = (
        '#include "expr.h"\n'
        '#include <math.h>\n'
        '#include "fastexp.h"\n'
        'double expr(double x, double y) {\n'
        '   double expr_result;\n'
        '   expr_result = fastexp(x + y);\n'
        '   return expr_result;\n'
        '}\n'
    )

    # 生成代码并比较结果
    result = codegen(('expr', expr), header=False, empty=False, code_gen=gen)
    source = result[0][1]
    assert source == expected

def test_c_with_printer():
    # 解决问题 13586
    from sympy.printing.c import C99CodePrinter

    # 定义自定义打印机类，用于处理 Pow 表达式
    class CustomPrinter(C99CodePrinter):
        def _print_Pow(self, expr):
            return "fastpow({}, {})".format(self._print(expr.base),
                                            self._print(expr.exp))

    # 创建符号变量 x
    x = symbols('x')
    # 创建表达式 x**3
    expr = x**3

    # 期望的输出结果
    expected = [
        ("file.c",
        "#include \"file.h\"\n"
        "#include <math.h>\n"
        "double test(double x) {\n"
        "   double test_result;\n"
        "   test_result = fastpow(x, 3);\n"
        "   return test_result;\n"
        "}\n"),
        ("file.h",
        "#ifndef PROJECT__FILE__H\n"
        "#define PROJECT__FILE__H\n"
        "double test(double x);\n"
        "#endif\n")
    ]

    # 使用自定义打印机生成代码，并比较结果
    result = codegen(("test", expr), "C", "file", header=False, empty=False, printer=CustomPrinter())
    assert result == expected

def test_fcode_complex():
    # 导入 sympy 中的 codegen 函数并允许复数计算
    import sympy.utilities.codegen
    sympy.utilities.codegen.COMPLEX_ALLOWED = True

    # 创建符号变量 x 和 y
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)

    # 生成 Fortran 95 代码，期望的输出结果
    result = codegen(('test', x + y), 'f95', 'test', header=False, empty=False)
    expected = (
        "REAL*8 function test(x, y)\n"
        "implicit none\n"
        "REAL*8, intent(in) :: x\n"
        "REAL*8, intent(in) :: y\n"
        "test = x + y\n"
        "end function\n"
    )
    # 比较生成的源代码和期望的输出结果
    source = result[0][1]
    assert source == expected

    # 创建只有 y 是实数的符号变量 x
    x = Symbol('x')
    y = Symbol('y', real=True)

    # 生成 Fortran 95 代码，期望的输出结果
    result = codegen(('test', x + y), 'f95', 'test', header=False, empty=False)
    source = result[0][1]
    expected = (
        "COMPLEX*16 function test(x, y)\n"  # 定义预期的字符串内容，表示一个 Fortran 函数签名
        "implicit none\n"  # Fortran 中的隐式声明，声明所有变量需要显式声明
        "COMPLEX*16, intent(in) :: x\n"  # 定义一个复数类型的输入参数 x
        "REAL*8, intent(in) :: y\n"  # 定义一个双精度实数类型的输入参数 y
        "test = x + y\n"  # 函数主体，将输入参数 x 和 y 相加并返回
        "end function\n"  # 函数结束
        )
    assert source==expected  # 使用断言检查变量 source 是否等于预期的字符串 expected

    sympy.utilities.codegen.COMPLEX_ALLOWED = False  # 设置 sympy 库中的 COMPLEX_ALLOWED 属性为 False，禁止复数类型的计算
```