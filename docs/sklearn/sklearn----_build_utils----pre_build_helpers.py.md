# `D:\src\scipysrc\scikit-learn\sklearn\_build_utils\pre_build_helpers.py`

```
# 导入必要的库：用于处理文件路径、执行系统命令、创建临时文件等
import glob
import os
import subprocess
import sys
import tempfile
import textwrap

# 定义一个函数，用于编译和运行给定的 C 代码
def compile_test_program(code, extra_preargs=None, extra_postargs=None):
    # 从 setuptools 库中导入编译器相关的功能
    from setuptools.command.build_ext import customize_compiler, new_compiler

    # 创建一个新的编译器对象
    ccompiler = new_compiler()
    # 自定义编译器配置
    customize_compiler(ccompiler)

    # 获取当前工作目录的绝对路径
    start_dir = os.path.abspath(".")

    # 使用临时目录来进行编译和运行测试
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            # 切换到临时目录
            os.chdir(tmp_dir)

            # 写入测试程序到文件中
            with open("test_program.c", "w") as f:
                f.write(code)

            # 创建一个名为 "objects" 的子目录，用于存放编译生成的目标文件
            os.mkdir("objects")

            # 编译测试程序
            ccompiler.compile(
                ["test_program.c"], output_dir="objects", extra_postargs=extra_postargs
            )

            # 获取所有生成的目标文件列表
            objects = glob.glob(os.path.join("objects", "*" + ccompiler.obj_extension))

            # 链接生成可执行文件
            ccompiler.link_executable(
                objects,
                "test_program",
                extra_preargs=extra_preargs,
                extra_postargs=extra_postargs,
            )

            # 检查环境变量中是否包含 PYTHON_CROSSENV
            if "PYTHON_CROSSENV" not in os.environ:
                # 如果不是交叉编译环境，则运行测试程序，并获取输出结果
                output = subprocess.check_output("./test_program")
                output = output.decode(sys.stdout.encoding or "utf-8").splitlines()
            else:
                # 如果是交叉编译环境，则返回空的输出列表，因为无法运行测试程序
                output = []
        except Exception:
            raise
        finally:
            # 无论是否发生异常，始终回到最初的工作目录
            os.chdir(start_dir)

    # 返回测试程序的输出结果
    return output


# 定义一个函数，用于执行基本的 C 代码编译和链接检查
def basic_check_build():
    # 如果运行环境为 PYODIDE，则直接返回，因为下面的检查在 PYODIDE 中无法工作
    if "PYODIDE" in os.environ:
        return

    # 定义一个简单的 C 代码块，用于检查基本的编译和链接功能
    code = textwrap.dedent(
        """\
        #include <stdio.h>
        int main(void) {
        return 0;
        }
        """
    )
    # 调用编译测试程序的函数，传入定义的 C 代码
    compile_test_program(code)
```