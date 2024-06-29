# `.\numpy\numpy\compat\py3k.py`

```
"""
Python 3.X compatibility tools.

While this file was originally intended for Python 2 -> 3 transition,
it is now used to create a compatibility layer between different
minor versions of Python 3.

While the active version of numpy may not support a given version of python, we
allow downstream libraries to continue to use these shims for forward
compatibility with numpy while they transition their code to newer versions of
Python.
"""
__all__ = ['bytes', 'asbytes', 'isfileobj', 'getexception', 'strchar',
           'unicode', 'asunicode', 'asbytes_nested', 'asunicode_nested',
           'asstr', 'open_latin1', 'long', 'basestring', 'sixu',
           'integer_types', 'is_pathlib_path', 'npy_load_module', 'Path',
           'pickle', 'contextlib_nullcontext', 'os_fspath', 'os_PathLike']

import sys                              # 导入 sys 模块，用于访问系统相关功能
import os                               # 导入 os 模块，用于访问操作系统功能
from pathlib import Path                # 导入 Path 类，用于处理路径操作
import io                               # 导入 io 模块，用于处理文件流操作

try:
    import pickle5 as pickle            # 尝试导入 pickle5 库，若不成功则导入标准的 pickle 库
except ImportError:
    import pickle                       # 如果导入 pickle5 失败，导入标准的 pickle 库

long = int                              # 定义 long 为 int 类型，用于兼容 Python 3 中移除的 long 类型
integer_types = (int,)                  # 定义 integer_types 为包含 int 的元组，用于兼容 Python 2/3 整数类型的差异
basestring = str                        # 定义 basestring 为 str 类型，用于兼容 Python 2/3 中字符串类型的差异
unicode = str                           # 定义 unicode 为 str 类型，用于兼容 Python 2/3 中的字符串类型
bytes = bytes                           # 定义 bytes 为 bytes 类型，用于兼容 Python 2/3 中的字节类型

def asunicode(s):
    if isinstance(s, bytes):
        return s.decode('latin1')       # 如果 s 是 bytes 类型，则解码为 str 类型，使用 Latin-1 编码
    return str(s)                       # 否则直接转换为 str 类型

def asbytes(s):
    if isinstance(s, bytes):
        return s                        # 如果 s 已经是 bytes 类型，则直接返回
    return str(s).encode('latin1')      # 否则将 s 转换为 str 类型后再编码为 bytes 类型，使用 Latin-1 编码

def asstr(s):
    if isinstance(s, bytes):
        return s.decode('latin1')       # 如果 s 是 bytes 类型，则解码为 str 类型，使用 Latin-1 编码
    return str(s)                       # 否则直接转换为 str 类型

def isfileobj(f):
    if not isinstance(f, (io.FileIO, io.BufferedReader, io.BufferedWriter)):
        return False                    # 如果 f 不是文件对象类型，则返回 False
    try:
        f.fileno()                      # 尝试获取文件描述符，可能会抛出 OSError 异常
        return True                     # 如果成功获取文件描述符，则返回 True
    except OSError:
        return False                    # 获取文件描述符失败，返回 False

def open_latin1(filename, mode='r'):
    return open(filename, mode=mode, encoding='iso-8859-1')  # 使用 Latin-1 编码打开指定文件

def sixu(s):
    return s                            # 返回参数 s，用于兼容性，不进行额外处理

strchar = 'U'                           # 定义 strchar 为 'U'

def getexception():
    return sys.exc_info()[1]            # 返回当前异常信息的第一个元素，即异常对象

def asbytes_nested(x):
    if hasattr(x, '__iter__') and not isinstance(x, (bytes, unicode)):
        return [asbytes_nested(y) for y in x]  # 如果 x 是可迭代对象且不是 bytes 或 unicode 类型，则递归处理每个元素
    else:
        return asbytes(x)               # 否则将 x 转换为 bytes 类型并返回

def asunicode_nested(x):
    if hasattr(x, '__iter__') and not isinstance(x, (bytes, unicode)):
        return [asunicode_nested(y) for y in x]  # 如果 x 是可迭代对象且不是 bytes 或 unicode 类型，则递归处理每个元素
    else:
        return asunicode(x)             # 否则将 x 转换为 unicode 类型并返回

def is_pathlib_path(obj):
    """
    Check whether obj is a `pathlib.Path` object.

    Prefer using ``isinstance(obj, os.PathLike)`` instead of this function.
    """
    return isinstance(obj, Path)        # 检查 obj 是否为 pathlib.Path 对象

# from Python 3.7
class contextlib_nullcontext:
    """Context manager that does no additional processing.

    Used as a stand-in for a normal context manager, when a particular
    block of code is only sometimes used with a normal context manager:

    cm = optional_cm if condition else nullcontext()
    with cm:
        # Perform operation, using optional_cm if condition is True

    .. note::
        Prefer using `contextlib.nullcontext` instead of this context manager.
    """

    def __init__(self, enter_result=None):
        self.enter_result = enter_result  # 初始化上下文管理器，保存进入结果
    # 定义上下文管理器的进入方法，当使用 with 语句时执行
    def __enter__(self):
        # 返回上下文管理器的进入结果，通常是为了与 as 关键字后的变量进行绑定
        return self.enter_result

    # 定义上下文管理器的退出方法，当退出 with 语句块时执行
    def __exit__(self, *excinfo):
        # 占位符方法体，不做任何实际操作，即使在异常发生时也不处理异常
        pass
# 加载一个模块。使用 ``load_module`` 方法，该方法将在 Python 3.12 中被弃用。
# 另外，可以使用 ``exec_module`` 方法，该方法在 numpy.distutils.misc_util.exec_mod_from_location 中定义。

# .. versionadded:: 1.11.2
# 版本添加说明，从版本 1.11.2 开始可用。

# Parameters
# ----------
# name : str
#     完整的模块名称。
# fn : str
#     模块文件的路径。
# info : tuple, optional
#     仅用于向后兼容 Python 2.*。

# Returns
# -------
# mod : module
#     加载并返回的模块对象。

def npy_load_module(name, fn, info=None):
    # 显式延迟导入以避免在启动时导入 importlib 的开销
    from importlib.machinery import SourceFileLoader
    return SourceFileLoader(name, fn).load_module()


# 将 os.fspath 函数赋值给变量 os_fspath，以便更方便地引用该函数
os_fspath = os.fspath
# 将 os.PathLike 类型赋值给变量 os_PathLike，以便更方便地引用该类型
os_PathLike = os.PathLike
```