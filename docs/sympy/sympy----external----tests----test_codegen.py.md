# `D:\src\scipysrc\sympy\sympy\external\tests\test_codegen.py`

```
# This tests the compilation and execution of the source code generated with
# utilities.codegen. The compilation takes place in a temporary directory that
# is removed after the test. By default the test directory is always removed,
# but this behavior can be changed by setting the environment variable
# SYMPY_TEST_CLEAN_TEMP to:
#   export SYMPY_TEST_CLEAN_TEMP=always   : the default behavior.
#   export SYMPY_TEST_CLEAN_TEMP=success  : only remove the directories of working tests.
#   export SYMPY_TEST_CLEAN_TEMP=never    : never remove the directories with the test code.
# When a directory is not removed, the necessary information is printed on
# screen to find the files that belong to the (failed) tests. If a test does
# not fail, py.test captures all the output and you will not see the directories
# corresponding to the successful tests. Use the --nocapture option to see all
# the output.

# All tests below have a counterpart in utilities/test/test_codegen.py. In the
# latter file, the resulting code is compared with predefined strings, without
# compilation or execution.

# All the generated Fortran code should conform with the Fortran 95 standard,
# and all the generated C code should be ANSI C, which facilitates the
# incorporation in various projects. The tests below assume that the binary cc
# is somewhere in the path and that it can compile ANSI C code.

from sympy.abc import x, y, z  # 导入 SymPy 中的变量 x, y, z
from sympy.external import import_module  # 导入 import_module 函数从外部导入模块
from sympy.testing.pytest import skip  # 导入 skip 函数，用于测试跳过
from sympy.utilities.codegen import codegen, make_routine, get_code_generator  # 导入代码生成相关的函数和类
import sys  # 导入 sys 模块，用于访问系统相关的参数和函数
import os  # 导入 os 模块，用于访问操作系统功能
import tempfile  # 导入 tempfile 模块，用于创建临时文件和目录
import subprocess  # 导入 subprocess 模块，用于执行外部命令

pyodide_js = import_module('pyodide_js')  # 使用 import_module 导入 'pyodide_js' 模块

# templates for the main program that will test the generated code.

main_template = {}  # 创建空字典 main_template

main_template['F95'] = """
program main
  include "codegen.h"
  integer :: result;
  result = 0

  %(statements)s

  call exit(result)
end program
"""

main_template['C89'] = """
#include "codegen.h"
#include <stdio.h>
#include <math.h>

int main() {
  int result = 0;

  %(statements)s

  return result;
}
"""

main_template['C99'] = main_template['C89']  # C99 和 C89 使用相同的模板

# templates for the numerical tests

numerical_test_template = {}  # 创建空字典 numerical_test_template

numerical_test_template['C89'] = """
  if (fabs(%(call)s)>%(threshold)s) {
    printf("Numerical validation failed: %(call)s=%%e threshold=%(threshold)s\\n", %(call)s);
    result = -1;
  }
"""

numerical_test_template['C99'] = numerical_test_template['C89']  # C99 和 C89 使用相同的模板

numerical_test_template['F95'] = """
  if (abs(%(call)s)>%(threshold)s) then
    write(6,"('Numerical validation failed:')")
    write(6,"('%(call)s=',e15.5,'threshold=',e15.5)") %(call)s, %(threshold)s
    result = -1;
  end if
"""

# command sequences for supported compilers

compile_commands = {}  # 创建空字典 compile_commands

compile_commands['cc'] = [
    "cc -c codegen.c -o codegen.o",
    "cc -c main.c -o main.o",
    "cc main.o codegen.o -lm -o test.exe"
]

compile_commands['gfortran'] = [
    "gfortran -c codegen.f90 -o codegen.o",
    # 编译Fortran源文件main.f90为目标文件main.o，使用gfortran编译器，并禁用自动换行限制
    "gfortran -ffree-line-length-none -c main.f90 -o main.o",
    
    # 使用gfortran链接main.o和codegen.o目标文件，生成可执行文件test.exe
    "gfortran main.o codegen.o -o test.exe"
# 将编译命令存储在 'g95' 编译器的列表中
compile_commands['g95'] = [
    "g95 -c codegen.f90 -o codegen.o",
    "g95 -ffree-line-length-huge -c main.f90 -o main.o",
    "g95 main.o codegen.o -o test.exe"
]

# 将编译命令存储在 'ifort' 编译器的列表中
compile_commands['ifort'] = [
    "ifort -c codegen.f90 -o codegen.o",
    "ifort -c main.f90 -o main.o",
    "ifort main.o codegen.o -o test.exe"
]

# 存储语言与编译器组合的元组列表
combinations_lang_compiler = [
    ('C89', 'cc'),
    ('C99', 'cc'),
    ('F95', 'ifort'),
    ('F95', 'gfortran'),
    ('F95', 'g95')
]


def try_run(commands):
    """运行一系列命令，只有当所有命令都成功运行时才返回 True。"""
    if pyodide_js:
        return False
    with open(os.devnull, 'w') as null:
        for command in commands:
            # 调用子进程执行命令，将标准输出和错误输出重定向到空设备
            retcode = subprocess.call(command, stdout=null, shell=True,
                    stderr=subprocess.STDOUT)
            if retcode != 0:
                return False
    return True


def run_test(label, routines, numerical_tests, language, commands, friendly=True):
    """用于代码生成测试的驱动程序。

       此驱动程序假设在 PATH 中存在编译器 ifort，且 ifort 至少是一个 Fortran 90 编译器。
       生成的代码将写入临时目录，与一个验证生成代码的主程序一起。当编译和验证运行正确时，测试通过。
    """

    # 在操作文件系统之前检查输入参数
    language = language.upper()
    assert language in main_template
    assert language in numerical_test_template

    # 检查环境变量是否合理
    clean = os.getenv('SYMPY_TEST_CLEAN_TEMP', 'always').lower()
    if clean not in ('always', 'success', 'never'):
        raise ValueError("SYMPY_TEST_CLEAN_TEMP must be one of the following: 'always', 'success' or 'never'.")

    # 执行所有编译、运行和验证测试代码的工作
    # 1) 准备临时工作目录，并切换到该目录
    work = tempfile.mkdtemp("_sympy_%s_test" % language, "%s_" % label)
    oldwork = os.getcwd()
    os.chdir(work)

    # 2) 写入生成的代码
    if friendly:
        # 将 routines 解释为 name_expr 列表，并调用友好的代码生成函数
        codegen(routines, language, "codegen", to_files=True)
    else:
        # 获取特定语言的代码生成器，然后写入生成的代码
        code_gen = get_code_generator(language, "codegen")
        code_gen.write(routines, "codegen", to_files=True)

    # 3) 写入一个简单的主程序，链接到生成的代码，并包括数值测试
    test_strings = []
    # 遍历 numerical_tests 列表中的每个测试元组，分别获取函数名、参数、期望值和阈值
    for fn_name, args, expected, threshold in numerical_tests:
        # 构造调用字符串，格式为 "<函数名>(<参数1>,<参数2>,...)-(<期望值>)"
        call_string = "%s(%s)-(%s)" % (
            fn_name, ",".join(str(arg) for arg in args), expected)
        
        # 如果编程语言为 "F95"，调用专门的函数转换双精度常数和阈值为 Fortran 95 格式
        if language == "F95":
            call_string = fortranize_double_constants(call_string)
            threshold = fortranize_double_constants(str(threshold))
        
        # 根据语言选择的模板，生成测试字符串并添加到 test_strings 列表中
        test_strings.append(numerical_test_template[language] % {
            "call": call_string,
            "threshold": threshold,
        })

    # 根据编程语言选择生成相应的文件名
    if language == "F95":
        f_name = "main.f90"
    elif language.startswith("C"):
        f_name = "main.c"
    else:
        # 如果语言未知，则抛出未实现错误，提示需要补充相关文件名扩展名
        raise NotImplementedError(
            "FIXME: filename extension unknown for language: %s" % language)

    # 打开选定的文件名，将主模板内容写入文件
    with open(f_name, "w") as f:
        f.write(
            main_template[language] % {'statements': "".join(test_strings)})

    # 4) 编译和链接
    compiled = try_run(commands)

    # 5) 如果成功编译，则运行
    if compiled:
        executed = try_run(["./test.exe"])
    else:
        executed = False

    # 6) 清理临时文件和目录
    if clean == 'always' or (clean == 'success' and compiled and executed):
        # 定义安全删除文件的函数
        def safe_remove(filename):
            if os.path.isfile(filename):
                os.remove(filename)
        
        # 依次删除指定的临时文件
        safe_remove("codegen.f90")
        safe_remove("codegen.c")
        safe_remove("codegen.h")
        safe_remove("codegen.o")
        safe_remove("main.f90")
        safe_remove("main.c")
        safe_remove("main.o")
        safe_remove("test.exe")
        # 切换回原始工作目录并删除临时工作目录
        os.chdir(oldwork)
        os.rmdir(work)
    else:
        # 如果不满足清理条件，则打印警告信息，并切换回原始工作目录
        print("TEST NOT REMOVED: %s" % work, file=sys.stderr)
        os.chdir(oldwork)

    # 7) 最后进行断言验证
    # 断言编译成功，否则输出失败信息和相关编译命令
    assert compiled, "failed to compile %s code with:\n%s" % (
        language, "\n".join(commands))
    # 断言执行成功，否则输出失败信息和相关执行命令
    assert executed, "failed to execute %s code from:\n%s" % (
        language, "\n".join(commands))
# 定义函数，将代码字符串中的浮点数替换为双精度浮点数
def fortranize_double_constants(code_string):
    """
    Replaces every literal float with literal doubles
    """
    import re
    # 匹配科学计数法表示的浮点数，如1.23e-10
    pattern_exp = re.compile(r'\d+(\.)?\d*[eE]-?\d+')
    # 匹配普通浮点数，但不包括以d结尾的（已经是双精度浮点数）
    pattern_float = re.compile(r'\d+\.\d*(?!\d*d)')

    # 将科学计数法的浮点数中的'e'替换为'd'
    def subs_exp(matchobj):
        return re.sub('[eE]', 'd', matchobj.group(0))

    # 在普通浮点数末尾添加'd0'表示双精度浮点数
    def subs_float(matchobj):
        return "%sd0" % matchobj.group(0)

    # 依次应用替换规则到代码字符串中
    code_string = pattern_exp.sub(subs_exp, code_string)
    code_string = pattern_float.sub(subs_float, code_string)

    return code_string


def is_feasible(language, commands):
    # 创建一个测试例程
    routine = make_routine("test", x)
    # 定义数值测试用例
    numerical_tests = [
        ("test", ( 1.0,), 1.0, 1e-15),
        ("test", (-1.0,), -1.0, 1e-15),
    ]
    try:
        # 运行测试，如果通过，则返回True
        run_test("is_feasible", [routine], numerical_tests, language, commands,
                 friendly=False)
        return True
    except AssertionError:
        # 如果测试失败，返回False
        return False

# 存储有效的语言和编译器命令组合的列表
valid_lang_commands = []
# 存储无效的语言和编译器命令组合的列表
invalid_lang_compilers = []

# 遍历语言和编译器组合列表
for lang, compiler in combinations_lang_compiler:
    # 获取当前编译器对应的命令集合
    commands = compile_commands[compiler]
    # 判断当前语言和编译器是否可行
    if is_feasible(lang, commands):
        # 如果可行，添加到有效语言和命令组合列表中
        valid_lang_commands.append((lang, commands))
    else:
        # 如果不可行，添加到无效语言和编译器组合列表中
        invalid_lang_compilers.append((lang, compiler))

# 测试所有语言和编译器组合，仅报告被跳过的情况

# 测试特定组合（C89, cc）是否被跳过
def test_C89_cc():
    if ("C89", 'cc') in invalid_lang_compilers:
        skip("`cc' command didn't work as expected (C89)")

# 测试特定组合（C99, cc）是否被跳过
def test_C99_cc():
    if ("C99", 'cc') in invalid_lang_compilers:
        skip("`cc' command didn't work as expected (C99)")

# 测试特定组合（F95, ifort）是否被跳过
def test_F95_ifort():
    if ("F95", 'ifort') in invalid_lang_compilers:
        skip("`ifort' command didn't work as expected")

# 测试特定组合（F95, gfortran）是否被跳过
def test_F95_gfortran():
    if ("F95", 'gfortran') in invalid_lang_compilers:
        skip("`gfortran' command didn't work as expected")

# 测试特定组合（F95, g95）是否被跳过
def test_F95_g95():
    if ("F95", 'g95') in invalid_lang_compilers:
        skip("`g95' command didn't work as expected")

# 实际的测试开始

# 测试基本代码生成
def test_basic_codegen():
    numerical_tests = [
        ("test", (1.0, 6.0, 3.0), 21.0, 1e-15),
        ("test", (-1.0, 2.0, -2.5), -2.5, 1e-15),
    ]
    name_expr = [("test", (x + y)*z)]
    # 对于每个有效的语言和命令组合，运行测试
    for lang, commands in valid_lang_commands:
        run_test("basic_codegen", name_expr, numerical_tests, lang, commands)

# 测试内置数学函数1的代码生成
def test_intrinsic_math1_codegen():
    # 导入所需的数学函数模块
    from sympy.core.evalf import N
    from sympy.functions import ln
    from sympy.functions.elementary.exponential import log
    from sympy.functions.elementary.hyperbolic import (cosh, sinh, tanh)
    from sympy.functions.elementary.integers import (ceiling, floor)
    from sympy.functions.elementary.miscellaneous import sqrt
    from sympy.functions.elementary.trigonometric import (acos, asin, atan, cos, sin, tan)
    # 定义一个包含测试名称和数学表达式的列表
    name_expr = [
        ("test_fabs", abs(x)),       # 测试绝对值函数 abs(x)
        ("test_acos", acos(x)),     # 测试反余弦函数 acos(x)
        ("test_asin", asin(x)),     # 测试反正弦函数 asin(x)
        ("test_atan", atan(x)),     # 测试反正切函数 atan(x)
        ("test_cos", cos(x)),       # 测试余弦函数 cos(x)
        ("test_cosh", cosh(x)),     # 测试双曲余弦函数 cosh(x)
        ("test_log", log(x)),       # 测试自然对数函数 log(x)
        ("test_ln", ln(x)),         # 测试自然对数函数 ln(x)
        ("test_sin", sin(x)),       # 测试正弦函数 sin(x)
        ("test_sinh", sinh(x)),     # 测试双曲正弦函数 sinh(x)
        ("test_sqrt", sqrt(x)),     # 测试平方根函数 sqrt(x)
        ("test_tan", tan(x)),       # 测试正切函数 tan(x)
        ("test_tanh", tanh(x)),     # 测试双曲正切函数 tanh(x)
    ]
    
    # 创建一个空列表来存储数值测试结果
    numerical_tests = []
    
    # 对每个测试名称和表达式进行迭代
    for name, expr in name_expr:
        # 对于每个预定义的 x 值（0.2, 0.5, 0.8）
        for xval in 0.2, 0.5, 0.8:
            # 使用 sympy 的 subs 方法计算表达式在给定 x 值下的数值结果
            expected = N(expr.subs(x, xval))
            # 将测试名称、参数元组、预期值和容差添加到数值测试列表中
            numerical_tests.append((name, (xval,), expected, 1e-14))
    
    # 对每个语言和其命令的组合进行迭代
    for lang, commands in valid_lang_commands:
        # 如果语言以 "C" 开头，则添加额外的 C 相关测试
        if lang.startswith("C"):
            name_expr_C = [("test_floor", floor(x)), ("test_ceil", ceiling(x))]
        else:
            name_expr_C = []
        
        # 运行测试，使用测试名称表达式的总和（name_expr + name_expr_C），数值测试，语言和命令
        run_test("intrinsic_math1", name_expr + name_expr_C,
                 numerical_tests, lang, commands)
# 定义一个函数用于测试内置数学函数的代码生成
def test_instrinsic_math2_codegen():
    # 导入数值计算函数 N
    from sympy.core.evalf import N
    # 导入三角函数中的 atan2 函数
    from sympy.functions.elementary.trigonometric import atan2
    # 定义一个包含测试名称和表达式的列表
    name_expr = [
        ("test_atan2", atan2(x, y)),  # 计算 atan2(x, y) 的测试
        ("test_pow", x**y),           # 计算 x**y 的测试
    ]
    numerical_tests = []  # 初始化数值测试列表
    # 对每个测试名称和表达式进行迭代
    for name, expr in name_expr:
        # 对每个预定义的 (xval, yval) 组合进行迭代
        for xval, yval in (0.2, 1.3), (0.5, -0.2), (0.8, 0.8):
            # 计算表达式在指定的 (xval, yval) 上的数值结果
            expected = N(expr.subs(x, xval).subs(y, yval))
            # 添加测试结果到数值测试列表，预期误差为 1e-14
            numerical_tests.append((name, (xval, yval), expected, 1e-14))
    # 对于每种语言和其命令的有效组合，运行测试函数
    for lang, commands in valid_lang_commands:
        run_test("intrinsic_math2", name_expr, numerical_tests, lang, commands)


# 定义一个函数用于测试复杂代码生成
def test_complicated_codegen():
    # 导入数值计算函数 N
    from sympy.core.evalf import N
    # 导入三角函数中的 cos, sin, tan 函数
    from sympy.functions.elementary.trigonometric import (cos, sin, tan)
    # 定义一个包含测试名称和表达式的列表
    name_expr = [
        ("test1", ((sin(x) + cos(y) + tan(z))**7).expand()),  # 计算复杂表达式的测试1
        ("test2", cos(cos(cos(cos(cos(cos(cos(cos(x + y + z))))))))),  # 计算复杂表达式的测试2
    ]
    numerical_tests = []  # 初始化数值测试列表
    # 对每个测试名称和表达式进行迭代
    for name, expr in name_expr:
        # 对每个预定义的 (xval, yval, zval) 组合进行迭代
        for xval, yval, zval in (0.2, 1.3, -0.3), (0.5, -0.2, 0.0), (0.8, 2.1, 0.8):
            # 计算表达式在指定的 (xval, yval, zval) 上的数值结果
            expected = N(expr.subs(x, xval).subs(y, yval).subs(z, zval))
            # 添加测试结果到数值测试列表，预期误差为 1e-12
            numerical_tests.append((name, (xval, yval, zval), expected, 1e-12))
    # 对于每种语言和其命令的有效组合，运行测试函数
    for lang, commands in valid_lang_commands:
        run_test("complicated_codegen", name_expr, numerical_tests, lang, commands)
```