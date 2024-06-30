# `D:\src\scipysrc\sympy\sympy\testing\tests\test_module_imports.py`

```
"""
检查 SymPy 是否不包含间接导入。

间接导入是指从一个模块导入一个符号，而该模块本身又从其他地方导入该符号。这种情况使得诊断模块间依赖和导入顺序问题变得更加困难，因此强烈不推荐。

（从最终用户代码进行的间接导入是可以接受的，事实上是最佳实践。）

实现说明：强制 Python 实际卸载已经导入的子模块是一个复杂且部分未记录的过程。为了避免这些问题，实际的诊断代码位于 bin/diagnose_imports 中，作为一个单独的、干净的 Python 进程运行。
"""

# 导入子进程管理模块
import subprocess
# 导入 sys 模块
import sys
# 从 os.path 模块导入几个函数
from os.path import abspath, dirname, join, normpath
# 导入 inspect 模块
import inspect
# 从 sympy.testing.pytest 模块导入 XFAIL 标记
from sympy.testing.pytest import XFAIL

# 使用 XFAIL 标记装饰的测试函数
@XFAIL
def test_module_imports_are_direct():
    # 获取当前文件的绝对路径
    my_filename = abspath(inspect.getfile(inspect.currentframe()))
    # 获取当前文件所在目录的路径
    my_dirname = dirname(my_filename)
    # 构建诊断导入问题的脚本文件路径
    diagnose_imports_filename = join(my_dirname, 'diagnose_imports.py')
    # 规范化路径格式
    diagnose_imports_filename = normpath(diagnose_imports_filename)

    # 启动一个子进程来运行诊断导入问题的脚本
    process = subprocess.Popen(
        [
            sys.executable,  # 使用当前 Python 解释器
            normpath(diagnose_imports_filename),  # 诊断导入问题的脚本路径
            '--problems',  # 指定输出问题
            '--by-importer'  # 按导入者分类输出
        ],
        stdout=subprocess.PIPE,  # 标准输出重定向到管道
        stderr=subprocess.STDOUT,  # 标准错误重定向到标准输出
        bufsize=-1  # 使用默认的缓冲区大小
    )
    # 读取子进程的输出
    output, _ = process.communicate()
    # 断言输出为空，否则输出导入问题的详细信息
    assert output == '', "There are import problems:\n" + output.decode()
```