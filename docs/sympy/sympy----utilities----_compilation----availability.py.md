# `D:\src\scipysrc\sympy\sympy\utilities\_compilation\availability.py`

```
# 导入标准库 os 和相关模块中的 compile_run_strings 函数
import os
from .compilation import compile_run_strings
from .util import CompilerNotFoundError

# 检查系统是否支持 Fortran 编译
def has_fortran():
    # 如果 has_fortran 函数没有属性 'result'，则进行如下操作
    if not hasattr(has_fortran, 'result'):
        try:
            # 尝试编译并运行包含 Fortran 代码的字符串
            (stdout, stderr), info = compile_run_strings(
                [('main.f90', (
                    'program foo\n'
                    'print *, "hello world"\n'
                    'end program'
                ))], clean=True
            )
        except CompilerNotFoundError:
            # 如果找不到编译器，设置 has_fortran.result 为 False
            has_fortran.result = False
            # 如果环境变量 SYMPY_STRICT_COMPILER_CHECKS 的值为 '1'，则抛出异常
            if os.environ.get('SYMPY_STRICT_COMPILER_CHECKS', '0') == '1':
                raise
        else:
            # 如果编译和运行成功，并且输出中包含 'hello world' 字符串
            if info['exit_status'] != os.EX_OK or 'hello world' not in stdout:
                # 如果编译运行失败或输出不包含 'hello world'，根据环境变量 SYMPY_STRICT_COMPILER_CHECKS 决定是否抛出异常
                if os.environ.get('SYMPY_STRICT_COMPILER_CHECKS', '0') == '1':
                    raise ValueError("Failed to compile test program:\n%s\n%s\n" % (stdout, stderr))
                # 设置 has_fortran.result 为 False
                has_fortran.result = False
            else:
                # 编译和运行成功且输出包含 'hello world'，设置 has_fortran.result 为 True
                has_fortran.result = True
    # 返回 has_fortran.result
    return has_fortran.result


# 检查系统是否支持 C 编译
def has_c():
    # 如果 has_c 函数没有属性 'result'，则进行如下操作
    if not hasattr(has_c, 'result'):
        try:
            # 尝试编译并运行包含 C 代码的字符串
            (stdout, stderr), info = compile_run_strings(
                [('main.c', (
                    '#include <stdio.h>\n'
                    'int main(){\n'
                    'printf("hello world\\n");\n'
                    'return 0;\n'
                    '}'
                ))], clean=True
            )
        except CompilerNotFoundError:
            # 如果找不到编译器，设置 has_c.result 为 False
            has_c.result = False
            # 如果环境变量 SYMPY_STRICT_COMPILER_CHECKS 的值为 '1'，则抛出异常
            if os.environ.get('SYMPY_STRICT_COMPILER_CHECKS', '0') == '1':
                raise
        else:
            # 如果编译和运行成功，并且输出中包含 'hello world' 字符串
            if info['exit_status'] != os.EX_OK or 'hello world' not in stdout:
                # 如果编译运行失败或输出不包含 'hello world'，根据环境变量 SYMPY_STRICT_COMPILER_CHECKS 决定是否抛出异常
                if os.environ.get('SYMPY_STRICT_COMPILER_CHECKS', '0') == '1':
                    raise ValueError("Failed to compile test program:\n%s\n%s\n" % (stdout, stderr))
                # 设置 has_c.result 为 False
                has_c.result = False
            else:
                # 编译和运行成功且输出包含 'hello world'，设置 has_c.result 为 True
                has_c.result = True
    # 返回 has_c.result
    return has_c.result


# 检查系统是否支持 C++ 编译
def has_cxx():
    # 如果 has_cxx 函数没有属性 'result'，则进行如下操作
    if not hasattr(has_cxx, 'result'):
        try:
            # 尝试编译并运行包含 C++ 代码的字符串
            (stdout, stderr), info = compile_run_strings(
                [('main.cxx', (
                    '#include <iostream>\n'
                    'int main(){\n'
                    'std::cout << "hello world" << std::endl;\n'
                    '}'
                ))], clean=True
            )
        except CompilerNotFoundError:
            # 如果找不到编译器，设置 has_cxx.result 为 False
            has_cxx.result = False
            # 如果环境变量 SYMPY_STRICT_COMPILER_CHECKS 的值为 '1'，则抛出异常
            if os.environ.get('SYMPY_STRICT_COMPILER_CHECKS', '0') == '1':
                raise
        else:
            # 如果编译和运行成功，并且输出中包含 'hello world' 字符串
            if info['exit_status'] != os.EX_OK or 'hello world' not in stdout:
                # 如果编译运行失败或输出不包含 'hello world'，根据环境变量 SYMPY_STRICT_COMPILER_CHECKS 决定是否抛出异常
                if os.environ.get('SYMPY_STRICT_COMPILER_CHECKS', '0') == '1':
                    raise ValueError("Failed to compile test program:\n%s\n%s\n" % (stdout, stderr))
                # 设置 has_cxx.result 为 False
                has_cxx.result = False
            else:
                # 编译和运行成功且输出包含 'hello world'，设置 has_cxx.result 为 True
                has_cxx.result = True
    # 返回 has_cxx.result
    return has_cxx.result
```