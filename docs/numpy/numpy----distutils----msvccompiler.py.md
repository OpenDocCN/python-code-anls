# `.\numpy\numpy\distutils\msvccompiler.py`

```py
import os  # 导入操作系统模块
from distutils.msvccompiler import MSVCCompiler as _MSVCCompiler  # 导入 MSVC 编译器类
from .system_info import platform_bits  # 导入平台位数信息


def _merge(old, new):
    """Concatenate two environment paths avoiding repeats.

    Here `old` is the environment string before the base class initialize
    function is called and `new` is the string after the call. The new string
    will be a fixed string if it is not obtained from the current environment,
    or the same as the old string if obtained from the same environment. The aim
    here is not to append the new string if it is already contained in the old
    string so as to limit the growth of the environment string.

    Parameters
    ----------
    old : string
        Previous environment string.
    new : string
        New environment string.

    Returns
    -------
    ret : string
        Updated environment string.

    """
    if new in old:
        return old
    if not old:
        return new

    # Neither new nor old is empty. Give old priority.
    return ';'.join([old, new])


class MSVCCompiler(_MSVCCompiler):
    def __init__(self, verbose=0, dry_run=0, force=0):
        _MSVCCompiler.__init__(self, verbose, dry_run, force)

    def initialize(self):
        # The 'lib' and 'include' variables may be overwritten
        # by MSVCCompiler.initialize, so save them for later merge.
        environ_lib = os.getenv('lib', '')  # 获取环境变量 'lib'，默认为空字符串
        environ_include = os.getenv('include', '')  # 获取环境变量 'include'，默认为空字符串
        _MSVCCompiler.initialize(self)  # 调用父类的初始化方法

        # Merge current and previous values of 'lib' and 'include'
        os.environ['lib'] = _merge(environ_lib, os.environ['lib'])  # 合并并更新 'lib' 环境变量
        os.environ['include'] = _merge(environ_include, os.environ['include'])  # 合并并更新 'include' 环境变量

        # msvc9 building for 32 bits requires SSE2 to work around a
        # compiler bug.
        if platform_bits == 32:  # 如果平台位数是 32 位
            self.compile_options += ['/arch:SSE2']  # 添加 SSE2 选项到编译选项
            self.compile_options_debug += ['/arch:SSE2']  # 添加 SSE2 选项到调试编译选项


def lib_opts_if_msvc(build_cmd):
    """ Add flags if we are using MSVC compiler

    We can't see `build_cmd` in our scope, because we have not initialized
    the distutils build command, so use this deferred calculation to run
    when we are building the library.
    """
    if build_cmd.compiler.compiler_type != 'msvc':  # 如果编译器类型不是 MSVC
        return []  # 返回空列表

    # Explicitly disable whole-program optimization.
    flags = ['/GL-']  # 设置禁用整体程序优化的标志

    # Disable voltbl section for vc142 to allow link using mingw-w64; see:
    # https://github.com/matthew-brett/dll_investigation/issues/1#issuecomment-1100468171
    if build_cmd.compiler_opt.cc_test_flags(['-d2VolatileMetadata-']):  # 如果编译器支持 '-d2VolatileMetadata-' 标志
        flags.append('-d2VolatileMetadata-')  # 添加此标志到编译选项

    return flags  # 返回标志列表
```