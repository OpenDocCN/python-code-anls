# `.\numpy\numpy\distutils\fcompiler\intel.py`

```py
# 导入 sys 模块，用于系统相关操作
import sys

# 从 numpy.distutils.ccompiler 中导入 simple_version_match 函数
from numpy.distutils.ccompiler import simple_version_match
# 从 numpy.distutils.fcompiler 中导入 FCompiler 类和 dummy_fortran_file 函数
from numpy.distutils.fcompiler import FCompiler, dummy_fortran_file

# 定义一组 Intel 编译器的名称列表
compilers = ['IntelFCompiler', 'IntelVisualFCompiler',
             'IntelItaniumFCompiler', 'IntelItaniumVisualFCompiler',
             'IntelEM64VisualFCompiler', 'IntelEM64TFCompiler']


def intel_version_match(type):
    # 匹配版本字符串中重要部分的函数
    return simple_version_match(start=r'Intel.*?Fortran.*?(?:%s).*?Version' % (type,))


class BaseIntelFCompiler(FCompiler):
    def update_executables(self):
        # 创建一个虚拟的 Fortran 文件并更新执行命令字典
        f = dummy_fortran_file()
        self.executables['version_cmd'] = ['<F77>', '-FI', '-V', '-c',
                                           f + '.f', '-o', f + '.o']

    def runtime_library_dir_option(self, dir):
        # 运行时库目录选项函数，生成链接器选项字符串
        # 注意：这里可以使用 -Xlinker，如果支持的话
        assert "," not in dir

        return '-Wl,-rpath=%s' % dir


class IntelFCompiler(BaseIntelFCompiler):
    # IntelFCompiler 类，继承自 BaseIntelFCompiler 类

    compiler_type = 'intel'
    compiler_aliases = ('ifort',)
    description = 'Intel Fortran Compiler for 32-bit apps'
    version_match = intel_version_match('32-bit|IA-32')

    possible_executables = ['ifort', 'ifc']

    executables = {
        'version_cmd'  : None,          # 由 update_executables 设置
        'compiler_f77' : [None, "-72", "-w90", "-w95"],
        'compiler_f90' : [None],
        'compiler_fix' : [None, "-FI"],
        'linker_so'    : ["<F90>", "-shared"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"]
        }

    pic_flags = ['-fPIC']
    module_dir_switch = '-module '  # 不要移除末尾的空格！
    module_include_switch = '-I'

    def get_flags_free(self):
        # 返回自由格式 Fortran 的编译选项列表
        return ['-FR']

    def get_flags(self):
        # 返回默认的编译选项列表
        return ['-fPIC']

    def get_flags_opt(self):  # Scipy test failures with -O2
        # 获取优化编译选项列表
        v = self.get_version()
        mpopt = 'openmp' if v and v < '15' else 'qopenmp'
        return ['-fp-model', 'strict', '-O1',
                '-assume', 'minus0', '-{}'.format(mpopt)]

    def get_flags_arch(self):
        # 返回架构相关的编译选项列表
        return []

    def get_flags_linker_so(self):
        # 获取用于链接共享对象的链接器选项列表
        opt = FCompiler.get_flags_linker_so(self)
        v = self.get_version()
        if v and v >= '8.0':
            opt.append('-nofor_main')
        if sys.platform == 'darwin':
            # 如果是 macOS 系统，则使用 -dynamiclib
            try:
                idx = opt.index('-shared')
                opt.remove('-shared')
            except ValueError:
                idx = 0
            opt[idx:idx] = ['-dynamiclib', '-Wl,-undefined,dynamic_lookup']
        return opt


class IntelItaniumFCompiler(IntelFCompiler):
    # IntelItaniumFCompiler 类，继承自 IntelFCompiler 类

    compiler_type = 'intele'
    compiler_aliases = ()
    description = 'Intel Fortran Compiler for Itanium apps'

    version_match = intel_version_match('Itanium|IA-64')

    possible_executables = ['ifort', 'efort', 'efc']
    # 定义一个包含多个可执行文件配置的字典
    executables = {
        'version_cmd'  : None,            # 版本命令的配置项，暂未指定
        'compiler_f77' : [None, "-FI", "-w90", "-w95"],  # Fortran 77 编译器的配置项列表
        'compiler_fix' : [None, "-FI"],   # 修复编译器的配置项列表
        'compiler_f90' : [None],          # Fortran 90 编译器的配置项列表，只包含一个元素
        'linker_so'    : ['<F90>', "-shared"],  # 共享库链接器的配置项列表
        'archiver'     : ["ar", "-cr"],   # 归档工具的配置项列表
        'ranlib'       : ["ranlib"]       # ranlib 工具的配置项列表
    }
# 定义一个继承自 IntelFCompiler 的类，用于 Intel EM64T 架构的编译器
class IntelEM64TFCompiler(IntelFCompiler):
    # 编译器类型为 'intelem'
    compiler_type = 'intelem'
    # 没有编译器的别名
    compiler_aliases = ()
    # 描述为“Intel Fortran Compiler for 64-bit apps”
    description = 'Intel Fortran Compiler for 64-bit apps'

    # 匹配版本信息的正则表达式，用于识别基于 EM64T 架构的 Intel 编译器
    version_match = intel_version_match('EM64T-based|Intel\\(R\\) 64|64|IA-64|64-bit')

    # 可能的可执行文件列表
    possible_executables = ['ifort', 'efort', 'efc']

    # 定义可执行文件的命令字典
    executables = {
        'version_cmd'  : None,  # 版本命令为空
        'compiler_f77' : [None, "-FI"],  # Fortran 77 编译器命令
        'compiler_fix' : [None, "-FI"],  # 修复格式 Fortran 编译器命令
        'compiler_f90' : [None],  # Fortran 90 编译器命令
        'linker_so'    : ['<F90>', "-shared"],  # 共享库链接器命令
        'archiver'     : ["ar", "-cr"],  # 静态库打包命令
        'ranlib'       : ["ranlib"]  # ranlib 命令
        }

# Is there no difference in the version string between the above compilers
# and the Visual compilers?


# 定义一个继承自 BaseIntelFCompiler 的类，用于 Intel Visual Fortran 编译器
class IntelVisualFCompiler(BaseIntelFCompiler):
    # 编译器类型为 'intelv'
    compiler_type = 'intelv'
    # 描述为“Intel Visual Fortran Compiler for 32-bit apps”
    description = 'Intel Visual Fortran Compiler for 32-bit apps'
    # 使用正则表达式匹配版本信息，用于识别 32 位 Intel Visual Fortran 编译器
    version_match = intel_version_match('32-bit|IA-32')

    # 更新可执行文件的方法
    def update_executables(self):
        # 创建一个虚拟的 Fortran 文件名
        f = dummy_fortran_file()
        # 设置版本命令的值为 Fortran 编译的命令列表
        self.executables['version_cmd'] = ['<F77>', '/FI', '/c',
                                           f + '.f', '/o', f + '.o']

    # ar_exe 表示 lib.exe 可执行文件名
    ar_exe = 'lib.exe'
    # 可能的可执行文件列表
    possible_executables = ['ifort', 'ifl']

    # 定义可执行文件的命令字典
    executables = {
        'version_cmd'  : None,  # 版本命令为空
        'compiler_f77' : [None],  # Fortran 77 编译器命令
        'compiler_fix' : [None],  # 修复格式 Fortran 编译器命令
        'compiler_f90' : [None],  # Fortran 90 编译器命令
        'linker_so'    : [None],  # 共享库链接器命令为空
        'archiver'     : [ar_exe, "/verbose", "/OUT:"],  # 静态库打包命令
        'ranlib'       : None  # ranlib 命令为空
        }

    # 编译开关为 '/c '
    compile_switch = '/c '
    # 对象开关为 '/Fo'，后面没有空格！
    object_switch = '/Fo'     # No space after /Fo!
    # 库开关为 '/OUT:'，后面没有空格！
    library_switch = '/OUT:'  # No space after /OUT:!
    # 模块目录开关为 '/module:'，后面没有空格！
    module_dir_switch = '/module:'  # No space after /module:
    # 模块包含开关为 '/I'
    module_include_switch = '/I'

    # 返回优化标志列表的方法
    def get_flags(self):
        opt = ['/nologo', '/MD', '/nbs', '/names:lowercase', 
               '/assume:underscore', '/fpp']
        return opt

    # 返回自由格式标志列表的方法
    def get_flags_free(self):
        return []

    # 返回调试标志列表的方法
    def get_flags_debug(self):
        return ['/4Yb', '/d2']

    # 返回优化标志列表的方法
    def get_flags_opt(self):
        return ['/O1', '/assume:minus0']  # Scipy test failures with /O2

    # 返回体系结构标志列表的方法
    def get_flags_arch(self):
        return ["/arch:IA32", "/QaxSSE3"]

    # 运行时库目录选项的方法
    def runtime_library_dir_option(self, dir):
        raise NotImplementedError


# 定义一个继承自 IntelVisualFCompiler 的类，用于 Intel Itanium 架构的 Visual Fortran 编译器
class IntelItaniumVisualFCompiler(IntelVisualFCompiler):
    # 编译器类型为 'intelev'
    compiler_type = 'intelev'
    # 描述为“Intel Visual Fortran Compiler for Itanium apps”

    description = 'Intel Visual Fortran Compiler for Itanium apps'

    # 使用正则表达式匹配版本信息，用于识别 Itanium 架构的 Intel Visual Fortran 编译器
    version_match = intel_version_match('Itanium')

    # 可能的可执行文件列表，这里是一个猜测
    possible_executables = ['efl']  # XXX this is a wild guess
    # ar_exe 和父类的相同
    ar_exe = IntelVisualFCompiler.ar_exe

    # 定义可执行文件的命令字典
    executables = {
        'version_cmd'  : None,  # 版本命令为空
        'compiler_f77' : [None, "-FI", "-w90", "-w95"],  # Fortran 77 编译器命令
        'compiler_fix' : [None, "-FI", "-4L72", "-w"],   # 修复格式 Fortran 编译器命令
        'compiler_f90' : [None],  # Fortran 90 编译器命令
        'linker_so'    : ['<F90>', "-shared"],  # 共享库链接器命令
        'archiver'     : [ar_exe, "/verbose", "/OUT:"],  # 静态库打包命令
        'ranlib'       : None  # ranlib 命令为空
        }


# 定义一个继承自 IntelVisualFCompiler 的类，用于 Intel EM64 架构的 Visual Fortran 编译器
class IntelEM64VisualFCompiler(IntelVisualFCompiler):
    # 编译器类型为 'intelvem'
    compiler_type = 'intelvem'
    # 定义变量 description，描述为“Intel Visual Fortran Compiler for 64-bit apps”
    description = 'Intel Visual Fortran Compiler for 64-bit apps'

    # 定义函数 version_match，用于匹配简单的版本信息，起始匹配字符串为'Intel(R).*?64,'
    version_match = simple_version_match(start=r'Intel\(R\).*?64,')

    # 定义方法 get_flags_arch，返回一个空列表，用于获取特定的编译标志（此处为占位符）
    def get_flags_arch(self):
        return []
# 如果当前脚本作为主程序执行
if __name__ == '__main__':
    # 从 distutils 模块导入 log 函数
    from distutils import log
    # 设置日志的详细程度为 2（即详细输出）
    log.set_verbosity(2)
    # 从 numpy.distutils 模块导入 customized_fcompiler 函数
    from numpy.distutils import customized_fcompiler
    # 打印使用 intel 编译器的 customized_fcompiler 函数的版本信息
    print(customized_fcompiler(compiler='intel').get_version())
```