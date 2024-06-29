# `.\numpy\numpy\distutils\_shell_utils.py`

```
"""
Helper functions for interacting with the shell, and consuming shell-style
parameters provided in config files.
"""
# 导入操作系统相关的模块
import os
# 导入处理 shell 样式参数的模块
import shlex
# 导入子进程操作的模块
import subprocess

# 定义公开的类方法列表
__all__ = ['WindowsParser', 'PosixParser', 'NativeParser']

# 命令行解析器类，用于拆分和连接命令行参数
class CommandLineParser:
    """
    An object that knows how to split and join command-line arguments.

    It must be true that ``argv == split(join(argv))`` for all ``argv``.
    The reverse neednt be true - `join(split(cmd))` may result in the addition
    or removal of unnecessary escaping.
    """
    @staticmethod
    def join(argv):
        """ Join a list of arguments into a command line string """
        raise NotImplementedError

    @staticmethod
    def split(cmd):
        """ Split a command line string into a list of arguments """
        raise NotImplementedError

# Windows 解析器类，用于处理在 Windows 上使用 subprocess.call("string") 时的解析行为
class WindowsParser:
    """
    The parsing behavior used by `subprocess.call("string")` on Windows, which
    matches the Microsoft C/C++ runtime.

    Note that this is _not_ the behavior of cmd.
    """
    @staticmethod
    def join(argv):
        # 注意：list2cmdline 是特定于 Windows 语法的
        return subprocess.list2cmdline(argv)

    @staticmethod
    def split(cmd):
        import ctypes  # guarded import for systems without ctypes
        try:
            ctypes.windll
        except AttributeError:
            raise NotImplementedError

        # Windows 对可执行文件有特殊的解析规则（不需要引号），我们不关心这一点 - 插入一个虚拟元素
        if not cmd:
            return []
        cmd = 'dummy ' + cmd

        CommandLineToArgvW = ctypes.windll.shell32.CommandLineToArgvW
        CommandLineToArgvW.restype = ctypes.POINTER(ctypes.c_wchar_p)
        CommandLineToArgvW.argtypes = (ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int))

        nargs = ctypes.c_int()
        lpargs = CommandLineToArgvW(cmd, ctypes.byref(nargs))
        args = [lpargs[i] for i in range(nargs.value)]
        assert not ctypes.windll.kernel32.LocalFree(lpargs)

        # 去掉我们插入的元素
        assert args[0] == "dummy"
        return args[1:]

# Posix 解析器类，用于处理在 Posix 上使用 subprocess.call("string", shell=True) 时的解析行为
class PosixParser:
    """
    The parsing behavior used by `subprocess.call("string", shell=True)` on Posix.
    """
    @staticmethod
    def join(argv):
        return ' '.join(shlex.quote(arg) for arg in argv)

    @staticmethod
    def split(cmd):
        return shlex.split(cmd, posix=True)

# 根据当前操作系统的类型选择合适的解析器类
if os.name == 'nt':
    NativeParser = WindowsParser
elif os.name == 'posix':
    NativeParser = PosixParser
```