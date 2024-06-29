# `.\numpy\numpy\distutils\msvc9compiler.py`

```
import os  # 导入操作系统相关的模块
from distutils.msvc9compiler import MSVCCompiler as _MSVCCompiler  # 导入MSVC编译器相关模块

from .system_info import platform_bits  # 从当前包导入平台位数信息


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
    if not old:
        return new
    if new in old:
        return old

    # Neither new nor old is empty. Give old priority.
    return ';'.join([old, new])


class MSVCCompiler(_MSVCCompiler):
    def __init__(self, verbose=0, dry_run=0, force=0):
        _MSVCCompiler.__init__(self, verbose, dry_run, force)

    def initialize(self, plat_name=None):
        # The 'lib' and 'include' variables may be overwritten
        # by MSVCCompiler.initialize, so save them for later merge.
        environ_lib = os.getenv('lib')  # 获取环境变量 'lib' 的值
        environ_include = os.getenv('include')  # 获取环境变量 'include' 的值
        _MSVCCompiler.initialize(self, plat_name)  # 调用父类的初始化方法

        # Merge current and previous values of 'lib' and 'include'
        os.environ['lib'] = _merge(environ_lib, os.environ['lib'])  # 合并并更新环境变量 'lib'
        os.environ['include'] = _merge(environ_include, os.environ['include'])  # 合并并更新环境变量 'include'

        # msvc9 building for 32 bits requires SSE2 to work around a
        # compiler bug.
        if platform_bits == 32:  # 如果平台位数为32位
            self.compile_options += ['/arch:SSE2']  # 添加编译选项 '/arch:SSE2'
            self.compile_options_debug += ['/arch:SSE2']  # 添加调试编译选项 '/arch:SSE2'

    def manifest_setup_ldargs(self, output_filename, build_temp, ld_args):
        ld_args.append('/MANIFEST')  # 向链接参数列表中添加 '/MANIFEST'
        _MSVCCompiler.manifest_setup_ldargs(self, output_filename,
                                            build_temp, ld_args)  # 调用父类的方法设置链接参数
```