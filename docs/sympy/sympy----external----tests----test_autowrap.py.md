# `D:\src\scipysrc\sympy\sympy\external\tests\test_autowrap.py`

```
import sympy
import tempfile
import os
from sympy.core.mod import Mod  # 导入 Mod 类
from sympy.core.relational import Eq  # 导入 Eq 类
from sympy.core.symbol import symbols  # 导入 symbols 函数
from sympy.external import import_module  # 导入 import_module 函数
from sympy.tensor import IndexedBase, Idx  # 导入 IndexedBase 和 Idx 类
from sympy.utilities.autowrap import autowrap, ufuncify, CodeWrapError  # 导入 autowrap, ufuncify 和 CodeWrapError 类
from sympy.testing.pytest import skip  # 导入 skip 函数

numpy = import_module('numpy', min_module_version='1.6.1')  # 尝试导入 numpy 模块，指定最低版本要求
Cython = import_module('Cython', min_module_version='0.15.1')  # 尝试导入 Cython 模块，指定最低版本要求
f2py = import_module('numpy.f2py', import_kwargs={'fromlist': ['f2py']})  # 尝试导入 numpy.f2py 模块

f2pyworks = False  # 初始化 f2pyworks 变量为 False
if f2py:
    try:
        autowrap(symbols('x'), 'f95', 'f2py')  # 尝试调用 autowrap 函数
    except (CodeWrapError, ImportError, OSError):
        f2pyworks = False  # 若出现 CodeWrapError, ImportError, OSError 异常，则将 f2pyworks 设置为 False
    else:
        f2pyworks = True  # 若 autowrap 调用成功，则将 f2pyworks 设置为 True

a, b, c = symbols('a b c')  # 创建符号变量 a, b, c
n, m, d = symbols('n m d', integer=True)  # 创建整数类型的符号变量 n, m, d
A, B, C = symbols('A B C', cls=IndexedBase)  # 创建 IndexedBase 类型的符号变量 A, B, C
i = Idx('i', m)  # 创建索引 i，范围为 m
j = Idx('j', n)  # 创建索引 j，范围为 n
k = Idx('k', d)  # 创建索引 k，范围为 d

def has_module(module):
    """
    如果模块存在，则返回 True，否则运行 skip() 函数。

    module 应为字符串类型。
    """
    modnames = {'numpy': numpy, 'Cython': Cython, 'f2py': f2py}  # 创建模块名称到模块对象的映射字典

    if modnames[module]:  # 检查给定模块在 modnames 字典中是否存在
        if module == 'f2py' and not f2pyworks:
            skip("Couldn't run f2py.")  # 若模块为 'f2py' 且 f2pyworks 为 False，则跳过测试
        return True  # 如果模块存在，则返回 True
    skip("Couldn't import %s." % module)  # 如果模块不存在，则跳过测试并显示无法导入该模块的消息

#
# test runners used by several language-backend combinations
#

def runtest_autowrap_twice(language, backend):
    """
    测试 autowrap 函数是否正确工作，验证 autowrap 更新模块名称。

    language: 编程语言名称
    backend: 后端名称
    """
    f = autowrap((((a + b)/c)**5).expand(), language, backend)  # 使用 autowrap 将表达式 (((a + b)/c)**5).expand() 编译为指定语言和后端的函数 f
    g = autowrap((((a + b)/c)**4).expand(), language, backend)  # 使用 autowrap 将表达式 (((a + b)/c)**4).expand() 编译为指定语言和后端的函数 g

    assert f(1, -2, 1) == -1.0  # 断言函数 f 在给定参数下的返回值为 -1.0
    assert g(1, -2, 1) == 1.0  # 断言函数 g 在给定参数下的返回值为 1.0


def runtest_autowrap_trace(language, backend):
    """
    测试 autowrap 函数是否正确工作，验证 autowrap 是否能够计算矩阵的迹。

    language: 编程语言名称
    backend: 后端名称
    """
    has_module('numpy')  # 检查 numpy 模块是否可用
    trace = autowrap(A[i, i], language, backend)  # 使用 autowrap 将 A[i, i] 编译为指定语言和后端的函数 trace
    assert trace(numpy.eye(100)) == 100  # 断言函数 trace 在给定参数 numpy.eye(100) 下的返回值为 100


def runtest_autowrap_matrix_vector(language, backend):
    """
    测试 autowrap 函数是否正确工作，验证 autowrap 是否能够生成矩阵向量乘法函数。

    language: 编程语言名称
    backend: 后端名称
    """
    has_module('numpy')  # 检查 numpy 模块是否可用
    x, y = symbols('x y', cls=IndexedBase)  # 创建 IndexedBase 类型的符号变量 x, y
    expr = Eq(y[i], A[i, j]*x[j])  # 创建矩阵向量乘法的表达式
    mv = autowrap(expr, language, backend)  # 使用 autowrap 将表达式 expr 编译为指定语言和后端的函数 mv

    M = numpy.random.rand(10, 20)  # 创建一个随机的 10x20 数组 M
    x = numpy.random.rand(20)  # 创建一个随机的长度为 20 的数组 x
    y = numpy.dot(M, x)  # 计算 numpy.dot(M, x) 的结果，作为参考值
    assert numpy.sum(numpy.abs(y - mv(M, x))) < 1e-13  # 断言函数 mv 在给定参数 M, x 下的返回值与 numpy.dot(M, x) 的结果误差小于 1e-13


def runtest_autowrap_matrix_matrix(language, backend):
    """
    测试 autowrap 函数是否正确工作，验证 autowrap 是否能够生成矩阵乘法函数。

    language: 编程语言名称
    backend: 后端名称
    """
    has_module('numpy')  # 检查 numpy 模块是否可用
    expr = Eq(C[i, j], A[i, k]*B[k, j])  # 创建矩阵乘法的表达式
    matmat = autowrap(expr, language, backend)  # 使用 autowrap 将表达式 expr 编译为指定语言和后端的函数 matmat

    M1 = numpy.random.rand(10, 20)  # 创建一个随机的 10x20 数组 M1
    M2 = numpy.random.rand(20, 15)  # 创建一个随机的 20x15 数组 M2
    M3 = numpy.dot(M1, M2)  # 计算 numpy.dot(M1, M2) 的结果，作为参考值
    assert numpy.sum(numpy.abs(M3 - matmat(M1, M2))) < 1e-13  # 断言函数 matmat 在给定参数 M1, M2 下的返回值与 numpy.dot(M1, M2) 的结果误差小于 1e-13


def runtest_ufuncify(language, backend):
    """
    测试 ufuncify 函数是否正确工作，验证 ufuncify 是否能够生成通用函数。

    language: 编程语言名称
    backend: 后端名称
    """
    has_module('numpy')  # 检查 numpy 模块是否可用
    a, b, c = symbols('a b c')  # 创建符号变量 a, b, c
    fabc = ufuncify([a, b, c], a*b + c, backend=backend)  # 使用 ufuncify 将表达式 a*b + c 编译为指定后端的通用函数 fabc
    facb = ufuncify([a, c, b], a*b + c, backend=
    # 创建一个包含等间距数值的一维数组，范围从-2到2，共50个元素
    grid = numpy.linspace(-2, 2, 50)
    # 创建一个包含等间距数值的一维数组，范围从-5到4，共50个元素
    b = numpy.linspace(-5, 4, 50)
    # 创建一个包含等间距数值的一维数组，范围从-1到1，共50个元素
    c = numpy.linspace(-1, 1, 50)
    # 根据 grid、b 和 c 计算期望值，即 grid*b + c，并将结果保存在 expected 中
    expected = grid * b + c
    # 使用 numpy.testing.assert_allclose 函数检查 fabc 函数的输出是否与 expected 数组非常接近
    numpy.testing.assert_allclose(fabc(grid, b, c), expected)
    # 使用 numpy.testing.assert_allclose 函数检查 facb 函数的输出是否与 expected 数组非常接近
    numpy.testing.assert_allclose(facb(grid, c, b), expected)
# 定义一个测试函数，用于测试 issue 10274 的问题
def runtest_issue_10274(language, backend):
    # 构造一个数学表达式 (a - b + c)**(13)
    expr = (a - b + c)**(13)
    # 创建一个临时目录用于存放中间文件
    tmp = tempfile.mkdtemp()
    # 调用 autowrap 函数生成代码，并返回生成的函数对象 f
    f = autowrap(expr, language, backend, tempdir=tmp,
                 helpers=('helper', a - b + c, (a, b, c)))
    # 断言生成的函数 f(1, 1, 1) 的结果为 1
    assert f(1, 1, 1) == 1

    # 遍历临时目录中的文件
    for file in os.listdir(tmp):
        # 如果文件名不以 "wrapped_code_" 开头或者不以 ".c" 结尾，则跳过
        if not (file.startswith("wrapped_code_") and file.endswith(".c")):
            continue

        # 打开文件，并读取所有行内容到 lines 列表中
        with open(tmp + '/' + file) as fil:
            lines = fil.readlines()
            # 断言文件的第一行是指定的注释行
            assert lines[0] == "/******************************************************************************\n"
            # 断言文件的第二行包含 SymPy 版本信息
            assert "Code generated with SymPy " + sympy.__version__ in lines[1]
            # 断言文件的余下部分与给定的模板内容完全一致
            assert lines[2:] == [
                " *                                                                            *\n",
                " *              See http://www.sympy.org/ for more information.               *\n",
                " *                                                                            *\n",
                " *                      This file is part of 'autowrap'                       *\n",
                " ******************************************************************************/\n",
                # 包含指定的头文件引用
                "#include " + '"' + file[:-1]+ 'h"' + "\n",
                # 包含标准数学库的引用
                "#include <math.h>\n",
                "\n",
                # 定义 helper 函数
                "double helper(double a, double b, double c) {\n",
                "\n",
                # 定义 helper 函数的局部变量
                "   double helper_result;\n",
                # 计算 helper 函数的结果并返回
                "   helper_result = a - b + c;\n",
                "   return helper_result;\n",
                "\n",
                "}\n",
                "\n",
                # 定义 autofunc 函数
                "double autofunc(double a, double b, double c) {\n",
                "\n",
                # 定义 autofunc 函数的局部变量
                "   double autofunc_result;\n",
                # 调用 helper 函数计算并返回 autofunc 函数的结果
                "   autofunc_result = pow(helper(a, b, c), 13);\n",
                "   return autofunc_result;\n",
                "\n",
                "}\n",
            ]


# 定义一个测试函数，用于测试 issue 15337 的问题
def runtest_issue_15337(language, backend):
    # 检查是否存在 numpy 模块
    has_module('numpy')
    # 创建符号变量 a, b, c, d, e
    a, b, c, d, e = symbols('a, b, c, d, e')
    # 构造数学表达式 (a - b + c - d + e)**13
    expr = (a - b + c - d + e)**13
    # 计算表达式的期望结果
    exp_res = (1. - 2. + 3. - 4. + 5.)**13

    # 使用 autowrap 函数生成代码并返回函数对象 f1，helper 是单个元素的元组
    f = autowrap(expr, language, backend, args=(a, b, c, d, e),
                 helpers=('f1', a - b + c, (a, b, c)))
    # 断言 f(1, 2, 3, 4, 5) 的结果与预期结果 exp_res 接近
    numpy.testing.assert_allclose(f(1, 2, 3, 4, 5), exp_res)

    # 使用 autowrap 函数生成代码并返回函数对象 f2，helper 是多个元素的元组
    f = autowrap(expr, language, backend, args=(a, b, c, d, e),
                 helpers=(('f1', a - b, (a, b)), ('f2', c - d, (c, d))))
    # 断言 f(1, 2, 3, 4, 5) 的结果与预期结果 exp_res 接近
    numpy.testing.assert_allclose(f(1, 2, 3, 4, 5), exp_res)
    # 检查是否安装了 'f2py' 模块
    has_module('f2py')
    
    # 创建符号变量 x 和 y
    x, y = symbols('x, y')
    
    # 构造数学表达式 Mod(x, 3.0) - Mod(y, -2.0)
    expr = Mod(x, 3.0) - Mod(y, -2.0)
    
    # 使用 autowrap 函数将表达式 expr 编译为 Fortran 95 语言的函数 f
    f = autowrap(expr, args=[x, y], language='F95')
    
    # 替换表达式中的符号变量 x 和 y，并计算表达式的数值结果
    exp_res = float(expr.xreplace({x: 3.5, y: 2.7}).evalf())
    
    # 断言自动包装后的函数 f 在参数 (3.5, 2.7) 处的计算结果与数值表达式的结果非常接近
    assert abs(f(3.5, 2.7) - exp_res) < 1e-14
    
    # 重新定义符号变量 x 和 y 为整数类型
    x, y = symbols('x, y', integer=True)
    
    # 重新构造数学表达式 Mod(x, 3) - Mod(y, -2)
    expr = Mod(x, 3) - Mod(y, -2)
    
    # 使用 autowrap 函数将整数类型表达式 expr 编译为 Fortran 95 语言的函数 f
    f = autowrap(expr, args=[x, y], language='F95')
    
    # 断言自动包装后的函数 f 在参数 (3, 2) 处的计算结果等于数值表达式的结果
    assert f(3, 2) == expr.xreplace({x: 3, y: 2})
# 测试不同语言后端组合的单元测试

# f2py 后端的测试函数

def test_wrap_twice_f95_f2py():
    # 检查是否安装了 'f2py' 模块
    has_module('f2py')
    # 运行测试，自动包装两次，使用 Fortran 95 和 f2py 后端
    runtest_autowrap_twice('f95', 'f2py')


def test_autowrap_trace_f95_f2py():
    # 检查是否安装了 'f2py' 模块
    has_module('f2py')
    # 运行测试，自动包装跟踪，使用 Fortran 95 和 f2py 后端
    runtest_autowrap_trace('f95', 'f2py')


def test_autowrap_matrix_vector_f95_f2py():
    # 检查是否安装了 'f2py' 模块
    has_module('f2py')
    # 运行测试，自动包装矩阵向量乘法，使用 Fortran 95 和 f2py 后端
    runtest_autowrap_matrix_vector('f95', 'f2py')


def test_autowrap_matrix_matrix_f95_f2py():
    # 检查是否安装了 'f2py' 模块
    has_module('f2py')
    # 运行测试，自动包装矩阵乘法，使用 Fortran 95 和 f2py 后端
    runtest_autowrap_matrix_matrix('f95', 'f2py')


def test_ufuncify_f95_f2py():
    # 检查是否安装了 'f2py' 模块
    has_module('f2py')
    # 运行测试，生成通用函数，使用 Fortran 95 和 f2py 后端
    runtest_ufuncify('f95', 'f2py')


def test_issue_15337_f95_f2py():
    # 检查是否安装了 'f2py' 模块
    has_module('f2py')
    # 运行测试，检查问题 15337，使用 Fortran 95 和 f2py 后端
    runtest_issue_15337('f95', 'f2py')


# Cython 后端的测试函数

def test_wrap_twice_c_cython():
    # 检查是否安装了 'Cython' 模块
    has_module('Cython')
    # 运行测试，自动包装两次，使用 C 和 Cython 后端
    runtest_autowrap_twice('C', 'cython')


def test_autowrap_trace_C_Cython():
    # 检查是否安装了 'Cython' 模块
    has_module('Cython')
    # 运行测试，自动包装跟踪，使用 C99 和 Cython 后端
    runtest_autowrap_trace('C99', 'cython')


def test_autowrap_matrix_vector_C_cython():
    # 检查是否安装了 'Cython' 模块
    has_module('Cython')
    # 运行测试，自动包装矩阵向量乘法，使用 C99 和 Cython 后端
    runtest_autowrap_matrix_vector('C99', 'cython')


def test_autowrap_matrix_matrix_C_cython():
    # 检查是否安装了 'Cython' 模块
    has_module('Cython')
    # 运行测试，自动包装矩阵乘法，使用 C99 和 Cython 后端
    runtest_autowrap_matrix_matrix('C99', 'cython')


def test_ufuncify_C_Cython():
    # 检查是否安装了 'Cython' 模块
    has_module('Cython')
    # 运行测试，生成通用函数，使用 C99 和 Cython 后端
    runtest_ufuncify('C99', 'cython')


def test_issue_10274_C_cython():
    # 检查是否安装了 'Cython' 模块
    has_module('Cython')
    # 运行测试，检查问题 10274，使用 C89 和 Cython 后端
    runtest_issue_10274('C89', 'cython')


def test_issue_15337_C_cython():
    # 检查是否安装了 'Cython' 模块
    has_module('Cython')
    # 运行测试，检查问题 15337，使用 C89 和 Cython 后端
    runtest_issue_15337('C89', 'cython')


def test_autowrap_custom_printer():
    # 检查是否安装了 'Cython' 模块
    has_module('Cython')

    # 导入所需的模块和类
    from sympy.core.numbers import pi
    from sympy.utilities.codegen import C99CodeGen
    from sympy.printing.c import C99CodePrinter

    # 定义一个自定义的 C99CodePrinter 类
    class PiPrinter(C99CodePrinter):
        def _print_Pi(self, expr):
            return "S_PI"

    # 创建一个 PiPrinter 实例
    printer = PiPrinter()
    # 创建一个 C99CodeGen 实例，传入自定义的 printer
    gen = C99CodeGen(printer=printer)
    # 添加预处理语句到代码生成器中
    gen.preprocessor_statements.append('#include "shortpi.h"')

    # 定义一个表达式
    expr = pi * a

    # 期望生成的 C 代码
    expected = (
        '#include "%s"\n'
        '#include <math.h>\n'
        '#include "shortpi.h"\n'
        '\n'
        'double autofunc(double a) {\n'
        '\n'
        '   double autofunc_result;\n'
        '   autofunc_result = S_PI*a;\n'
        '   return autofunc_result;\n'
        '\n'
        '}\n'
    )

    # 创建一个临时目录用于生成的代码
    tmpdir = tempfile.mkdtemp()
    # 写入一个简单的头文件，供生成的代码使用
    with open(os.path.join(tmpdir, 'shortpi.h'), 'w') as f:
        f.write('#define S_PI 3.14')

    # 使用 autowrap 函数生成代码，并指定后端为 'cython'，代码生成器为 gen
    func = autowrap(expr, backend='cython', tempdir=tmpdir, code_gen=gen)

    # 断言生成的函数计算结果与预期值相符
    assert func(4.2) == 3.14 * 4.2

    # 检查生成的代码是否正确
    for filename in os.listdir(tmpdir):
        if filename.startswith('wrapped_code') and filename.endswith('.c'):
            with open(os.path.join(tmpdir, filename)) as f:
                lines = f.readlines()
                # 替换期望的字符串格式，将文件名扩展名替换为 '.h'
                expected = expected % filename.replace('.c', '.h')
                # 断言生成的代码内容与预期的内容相符
                assert ''.join(lines[7:]) == expected


# NumPy 后端的测试函数

def test_ufuncify_numpy():
    # 检查是否安装了 'numpy' 模块
    has_module('numpy')
    # 检查当前环境是否安装了 Cython 模块
    # 如果 Cython 可用，则意味着存在有效的 C 编译器，这是必需的。
    has_module('Cython')
    
    # 运行名为 'runtest_ufuncify' 的测试函数，传入参数 'C99' 和 'numpy'
    runtest_ufuncify('C99', 'numpy')
```