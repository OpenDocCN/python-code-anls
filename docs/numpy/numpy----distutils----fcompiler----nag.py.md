# `.\numpy\numpy\distutils\fcompiler\nag.py`

```py
# 导入 sys 模块，用于获取系统相关信息
import sys
# 导入 re 模块，用于正则表达式操作
import re
# 从 numpy.distutils.fcompiler 导入 FCompiler 类
from numpy.distutils.fcompiler import FCompiler

# 定义一个列表，包含两个编译器类名
compilers = ['NAGFCompiler', 'NAGFORCompiler']

# 定义一个名为 BaseNAGFCompiler 的类，继承自 FCompiler 类
class BaseNAGFCompiler(FCompiler):
    # 定义版本号匹配的正则表达式模式
    version_pattern = r'NAG.* Release (?P<version>[^(\s]*)'

    # 方法：匹配版本号
    def version_match(self, version_string):
        # 在版本字符串中搜索匹配版本号的模式
        m = re.search(self.version_pattern, version_string)
        # 如果匹配成功，返回版本号
        if m:
            return m.group('version')
        else:
            return None

    # 方法：获取链接器选项
    def get_flags_linker_so(self):
        return ["-Wl,-shared"]

    # 方法：获取优化选项
    def get_flags_opt(self):
        return ['-O4']

    # 方法：获取架构相关选项
    def get_flags_arch(self):
        return []

# 定义一个名为 NAGFCompiler 的类，继承自 BaseNAGFCompiler 类
class NAGFCompiler(BaseNAGFCompiler):
    # 类属性：编译器类型为 'nag'
    compiler_type = 'nag'
    # 类属性：描述为 'NAGWare Fortran 95 Compiler'
    description = 'NAGWare Fortran 95 Compiler'

    # 类属性：定义不同命令的执行路径
    executables = {
        'version_cmd'  : ["<F90>", "-V"],
        'compiler_f77' : ["f95", "-fixed"],
        'compiler_fix' : ["f95", "-fixed"],
        'compiler_f90' : ["f95"],
        'linker_so'    : ["<F90>"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"]
        }

    # 方法：重写获取链接器选项的方法
    def get_flags_linker_so(self):
        # 如果系统平台是 'darwin'（即 macOS）
        if sys.platform == 'darwin':
            # 返回特定于 macOS 的链接选项
            return ['-unsharedf95', '-Wl,-bundle,-flat_namespace,-undefined,suppress']
        # 否则调用父类的方法获取链接选项
        return BaseNAGFCompiler.get_flags_linker_so(self)

    # 方法：重写获取架构相关选项的方法
    def get_flags_arch(self):
        # 获取当前编译器的版本号
        version = self.get_version()
        # 如果版本号存在且小于 '5.1'
        if version and version < '5.1':
            # 返回特定的目标架构选项
            return ['-target=native']
        else:
            # 否则调用父类的方法获取架构选项
            return BaseNAGFCompiler.get_flags_arch(self)

    # 方法：获取调试选项
    def get_flags_debug(self):
        # 返回调试相关的选项
        return ['-g', '-gline', '-g90', '-nan', '-C']

# 定义一个名为 NAGFORCompiler 的类，继承自 BaseNAGFCompiler 类
class NAGFORCompiler(BaseNAGFCompiler):
    # 类属性：编译器类型为 'nagfor'
    compiler_type = 'nagfor'
    # 类属性：描述为 'NAG Fortran Compiler'
    description = 'NAG Fortran Compiler'

    # 类属性：定义不同命令的执行路径
    executables = {
        'version_cmd'  : ["nagfor", "-V"],
        'compiler_f77' : ["nagfor", "-fixed"],
        'compiler_fix' : ["nagfor", "-fixed"],
        'compiler_f90' : ["nagfor"],
        'linker_so'    : ["nagfor"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"]
        }

    # 方法：重写获取链接器选项的方法
    def get_flags_linker_so(self):
        # 如果系统平台是 'darwin'（即 macOS）
        if sys.platform == 'darwin':
            # 返回特定于 macOS 的链接选项
            return ['-unsharedrts',
                    '-Wl,-bundle,-flat_namespace,-undefined,suppress']
        # 否则调用父类的方法获取链接选项
        return BaseNAGFCompiler.get_flags_linker_so(self)

    # 方法：获取调试选项
    def get_flags_debug(self):
        # 获取当前编译器的版本号
        version = self.get_version()
        # 如果版本号存在且大于 '6.1'
        if version and version > '6.1':
            # 返回特定的调试选项
            return ['-g', '-u', '-nan', '-C=all', '-thread_safe',
                    '-kind=unique', '-Warn=allocation', '-Warn=subnormal']
        else:
            # 否则返回通用的调试选项
            return ['-g', '-nan', '-C=all', '-u', '-thread_safe']

# 如果当前脚本被直接运行
if __name__ == '__main__':
    # 从 distutils 模块中导入 log 对象
    from distutils import log
    # 设置日志的详细程度为 2
    log.set_verbosity(2)
    # 从 numpy.distutils 中导入 customized_fcompiler 函数
    from numpy.distutils import customized_fcompiler
    # 使用 customized_fcompiler 函数创建一个特定编译器的实例，这里是 'nagfor'
    compiler = customized_fcompiler(compiler='nagfor')
    # 打印编译器的版本信息
    print(compiler.get_version())
    # 打印编译器的调试选项
    print(compiler.get_flags_debug())
```