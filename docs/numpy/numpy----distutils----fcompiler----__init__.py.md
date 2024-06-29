# `.\numpy\numpy\distutils\fcompiler\__init__.py`

```py
"""
Contains FCompiler, an abstract base class that defines the interface
for the numpy.distutils Fortran compiler abstraction model.

Terminology:

To be consistent, where the term 'executable' is used, it means the single
file, like 'gcc', that is executed, and should be a string. In contrast,
'command' means the entire command line, like ['gcc', '-c', 'file.c'], and
should be a list.

But note that FCompiler.executables is actually a dictionary of commands.

"""

__all__ = ['FCompiler', 'new_fcompiler', 'show_fcompilers',
           'dummy_fortran_file']

import os                         # 导入操作系统相关的功能
import sys                        # 导入系统相关的功能
import re                         # 导入正则表达式模块
from pathlib import Path          # 导入路径处理模块 Path

from distutils.sysconfig import get_python_lib   # 导入获取 Python 库路径的函数
from distutils.fancy_getopt import FancyGetopt   # 导入自定义命令行选项处理模块
from distutils.errors import DistutilsModuleError, \
     DistutilsExecError, CompileError, LinkError, DistutilsPlatformError   # 导入 distutils 的各种错误类
from distutils.util import split_quoted, strtobool   # 导入辅助函数 split_quoted 和 strtobool

from numpy.distutils.ccompiler import CCompiler, gen_lib_options   # 导入 CCompiler 类和 gen_lib_options 函数
from numpy.distutils import log   # 导入 numpy.distutils 中的日志模块
from numpy.distutils.misc_util import is_string, all_strings, is_sequence, \
    make_temp_file, get_shared_lib_extension   # 导入一些辅助函数和工具函数
from numpy.distutils.exec_command import find_executable   # 导入查找可执行文件的函数
from numpy.distutils import _shell_utils   # 导入私有模块 _shell_utils

from .environment import EnvironmentConfig   # 导入环境配置相关的模块

__metaclass__ = type   # 使用 Python 2 风格的类定义元类

FORTRAN_COMMON_FIXED_EXTENSIONS = ['.for', '.ftn', '.f77', '.f']   # 常见的 Fortran 文件扩展名列表

class CompilerNotFound(Exception):
    pass

def flaglist(s):
    if is_string(s):   # 如果参数是字符串
        return split_quoted(s)   # 调用 split_quoted 函数分割字符串返回列表
    else:
        return s   # 否则直接返回参数本身

def str2bool(s):
    if is_string(s):   # 如果参数是字符串
        return strtobool(s)   # 调用 strtobool 函数转换为布尔值并返回
    return bool(s)   # 否则直接转换为布尔值并返回

def is_sequence_of_strings(seq):
    return is_sequence(seq) and all_strings(seq)   # 判断是否为字符串序列的函数

class FCompiler(CCompiler):
    """Abstract base class to define the interface that must be implemented
    by real Fortran compiler classes.

    Methods that subclasses may redefine:

        update_executables(), find_executables(), get_version()
        get_flags(), get_flags_opt(), get_flags_arch(), get_flags_debug()
        get_flags_f77(), get_flags_opt_f77(), get_flags_arch_f77(),
        get_flags_debug_f77(), get_flags_f90(), get_flags_opt_f90(),
        get_flags_arch_f90(), get_flags_debug_f90(),
        get_flags_fix(), get_flags_linker_so()

    DON'T call these methods (except get_version) after
    constructing a compiler instance or inside any other method.
    All methods, except update_executables() and find_executables(),
    may call the get_version() method.

    After constructing a compiler instance, always call customize(dist=None)
    method that finalizes compiler construction and makes the following
    attributes available:
      compiler_f77
      compiler_f90
      compiler_fix
      linker_so
      archiver
      ranlib
      libraries
      library_dirs
    """
    
    # These are the environment variables and distutils keys used.
    # Each configuration description is
    # (<hook name>, <environment variable>, <key in distutils.cfg>, <convert>, <append>)
    # The hook names are handled by the self._environment_hook method.
    # - names starting with 'self.' call methods in this class
    # - names starting with 'exe.' return the key in the executables dict
    # - names like 'flags.YYY' return self.get_flag_YYY()
    # convert is either None or a function to convert a string to the
    # appropriate type used.
    
    # 定义distutils_vars变量，用于存储与环境配置相关的选项
    distutils_vars = EnvironmentConfig(
        distutils_section='config_fc',
        noopt = (None, None, 'noopt', str2bool, False),  # 无优化选项，使用str2bool函数将字符串转换为布尔值
        noarch = (None, None, 'noarch', str2bool, False),  # 非架构相关选项，使用str2bool函数将字符串转换为布尔值
        debug = (None, None, 'debug', str2bool, False),  # 调试选项，使用str2bool函数将字符串转换为布尔值
        verbose = (None, None, 'verbose', str2bool, False),  # 详细模式选项，使用str2bool函数将字符串转换为布尔值
    )
    
    # 定义command_vars变量，用于存储编译器和链接器等命令相关的选项
    command_vars = EnvironmentConfig(
        distutils_section='config_fc',
        compiler_f77 = ('exe.compiler_f77', 'F77', 'f77exec', None, False),  # Fortran 77编译器选项
        compiler_f90 = ('exe.compiler_f90', 'F90', 'f90exec', None, False),  # Fortran 90编译器选项
        compiler_fix = ('exe.compiler_fix', 'F90', 'f90exec', None, False),  # 修复编译器选项
        version_cmd = ('exe.version_cmd', None, None, None, False),  # 版本命令选项
        linker_so = ('exe.linker_so', 'LDSHARED', 'ldshared', None, False),  # 共享目标文件链接器选项
        linker_exe = ('exe.linker_exe', 'LD', 'ld', None, False),  # 可执行文件链接器选项
        archiver = (None, 'AR', 'ar', None, False),  # 静态库归档工具选项
        ranlib = (None, 'RANLIB', 'ranlib', None, False),  # ranlib命令选项
    )
    
    # 定义flag_vars变量，用于存储编译和链接标志相关的选项
    flag_vars = EnvironmentConfig(
        distutils_section='config_fc',
        f77 = ('flags.f77', 'F77FLAGS', 'f77flags', flaglist, True),  # Fortran 77编译标志选项，使用flaglist处理标志列表
        f90 = ('flags.f90', 'F90FLAGS', 'f90flags', flaglist, True),  # Fortran 90编译标志选项，使用flaglist处理标志列表
        free = ('flags.free', 'FREEFLAGS', 'freeflags', flaglist, True),  # 自由格式编译标志选项，使用flaglist处理标志列表
        fix = ('flags.fix', None, None, flaglist, False),  # 修复格式编译标志选项，使用flaglist处理标志列表
        opt = ('flags.opt', 'FOPT', 'opt', flaglist, True),  # 优化编译标志选项，使用flaglist处理标志列表
        opt_f77 = ('flags.opt_f77', None, None, flaglist, False),  # Fortran 77优化编译标志选项，使用flaglist处理标志列表
        opt_f90 = ('flags.opt_f90', None, None, flaglist, False),  # Fortran 90优化编译标志选项，使用flaglist处理标志列表
        arch = ('flags.arch', 'FARCH', 'arch', flaglist, False),  # 架构编译标志选项，使用flaglist处理标志列表
        arch_f77 = ('flags.arch_f77', None, None, flaglist, False),  # Fortran 77架构编译标志选项，使用flaglist处理标志列表
        arch_f90 = ('flags.arch_f90', None, None, flaglist, False),  # Fortran 90架构编译标志选项，使用flaglist处理标志列表
        debug = ('flags.debug', 'FDEBUG', 'fdebug', flaglist, True),  # 调试编译标志选项，使用flaglist处理标志列表
        debug_f77 = ('flags.debug_f77', None, None, flaglist, False),  # Fortran 77调试编译标志选项，使用flaglist处理标志列表
        debug_f90 = ('flags.debug_f90', None, None, flaglist, False),  # Fortran 90调试编译标志选项，使用flaglist处理标志列表
        flags = ('self.get_flags', 'FFLAGS', 'fflags', flaglist, True),  # 获取所有标志选项的方法，使用flaglist处理标志列表
        linker_so = ('flags.linker_so', 'LDFLAGS', 'ldflags', flaglist, True),  # 共享目标文件链接器标志选项，使用flaglist处理标志列表
        linker_exe = ('flags.linker_exe', 'LDFLAGS', 'ldflags', flaglist, True),  # 可执行文件链接器标志选项，使用flaglist处理标志列表
        ar = ('flags.ar', 'ARFLAGS', 'arflags', flaglist, True),  # 静态库归档工具标志选项，使用flaglist处理标志列表
    )
    
    # 定义language_map变量，用于存储文件扩展名与编程语言的映射关系
    language_map = {'.f': 'f77',
                    '.for': 'f77',
                    '.F': 'f77',    # 需要预处理器的Fortran 77文件
                    '.ftn': 'f77',
                    '.f77': 'f77',
                    '.f90': 'f90',
                    '.F90': 'f90',  # 需要预处理器的Fortran 90文件
                    '.f95': 'f90',
                    }
    
    # 定义language_order变量，用于存储编译优先顺序的列表
    language_order = ['f90', 'f77']
    
    # 这些变量将由子类设置
    # 初始化变量，声明编译器类型、编译器别名为空，版本模式为空
    compiler_type = None
    compiler_aliases = ()
    version_pattern = None

    # 初始化空列表和字典，包含预定义的执行文件命令列表
    possible_executables = []
    executables = {
        'version_cmd': ["f77", "-v"],     # 版本命令，调用 f77 编译器获取版本信息
        'compiler_f77': ["f77"],          # f77 编译器命令
        'compiler_f90': ["f90"],          # f90 编译器命令
        'compiler_fix': ["f90", "-fixed"],# f90 编译器以固定格式编译
        'linker_so': ["f90", "-shared"],  # f90 编译器链接共享库
        'linker_exe': ["f90"],            # f90 编译器链接可执行文件
        'archiver': ["ar", "-cr"],        # 静态库打包命令 ar -cr
        'ranlib': None,                   # ranlib 命令未定义
        }

    # 如果编译器不支持编译 Fortran 90，则建议使用另一个编译器类型
    suggested_f90_compiler = None

    compile_switch = "-c"                 # 编译开关
    object_switch = "-o "                 # 对象文件生成开关，结尾空格影响字符串拼接
                                         # 如果缺少，则通过字符串连接添加到对象文件名前

    library_switch = "-o "                # 库文件生成开关，同上
                                         # 用于指定模块文件的创建和搜索位置的开关
    module_dir_switch = None              # 模块目录开关，未定义

    module_include_switch = '-I'          # 模块包含路径开关，用于指定模块文件的搜索位置

    pic_flags = []                        # 用于创建位置无关代码的标志

    src_extensions = ['.for', '.ftn', '.f77', '.f', '.f90', '.f95', '.F', '.F90', '.FOR']  # 源文件扩展名列表
    obj_extension = ".o"                  # 对象文件扩展名

    shared_lib_extension = get_shared_lib_extension()  # 共享库文件扩展名，通过函数获取
    static_lib_extension = ".a"           # 静态库文件扩展名，或者 .lib
    static_lib_format = "lib%s%s"         # 静态库格式

    shared_lib_format = "%s%s"            # 共享库格式
    exe_extension = ""                    # 可执行文件扩展名为空

    _exe_cache = {}                       # 可执行文件缓存

    _executable_keys = ['version_cmd', 'compiler_f77', 'compiler_f90',
                        'compiler_fix', 'linker_so', 'linker_exe', 'archiver',
                        'ranlib']

    # new_fcompiler 调用时会被设置，在 command/{build_ext.py, build_clib.py, config.py} 文件中
    c_compiler = None                     # C 编译器未定义

    # extra_{f77,f90}_compile_args 由 build_ext.build_extension 方法设置
    extra_f77_compile_args = []           # 额外的 f77 编译参数
    extra_f90_compile_args = []           # 额外的 f90 编译参数

    def __init__(self, *args, **kw):
        # 调用父类 CCompiler 的初始化方法，并克隆环境钩子以配置 Distutils 变量
        CCompiler.__init__(self, *args, **kw)
        self.distutils_vars = self.distutils_vars.clone(self._environment_hook)
        self.command_vars = self.command_vars.clone(self._environment_hook)
        self.flag_vars = self.flag_vars.clone(self._environment_hook)
        self.executables = self.executables.copy()

        # 将预定义的执行文件键复制到 executables 字典中，如果不存在则设为 None
        for e in self._executable_keys:
            if e not in self.executables:
                self.executables[e] = None

        # 用于跟踪 customize() 方法是否已调用
        self._is_customised = False
    # 实现对象的浅复制操作
    def __copy__(self):
        # 创建一个新的对象，类型与当前对象相同
        obj = self.__new__(self.__class__)
        # 将当前对象的属性复制到新对象中
        obj.__dict__.update(self.__dict__)
        # 使用环境钩子克隆 distutils_vars，command_vars，flag_vars 对象
        obj.distutils_vars = obj.distutils_vars.clone(obj._environment_hook)
        obj.command_vars = obj.command_vars.clone(obj._environment_hook)
        obj.flag_vars = obj.flag_vars.clone(obj._environment_hook)
        # 复制 executables 字典
        obj.executables = obj.executables.copy()
        return obj

    # 返回对象的浅复制
    def copy(self):
        return self.__copy__()

    # 使用属性来访问 CCompiler 使用的属性。直接从 self.executables 字典中设置这些属性可能会出错，
    # 因此每次都从中获取它们。
    def _command_property(key):
        def fget(self):
            # 断言对象已经定制化
            assert self._is_customised
            # 返回指定 key 对应的可执行文件路径
            return self.executables[key]
        return property(fget=fget)
    # 定义各个命令属性
    version_cmd = _command_property('version_cmd')
    compiler_f77 = _command_property('compiler_f77')
    compiler_f90 = _command_property('compiler_f90')
    compiler_fix = _command_property('compiler_fix')
    linker_so = _command_property('linker_so')
    linker_exe = _command_property('linker_exe')
    archiver = _command_property('archiver')
    ranlib = _command_property('ranlib')

    # 使术语一致化。
    def set_executable(self, key, value):
        self.set_command(key, value)

    # 设置多个命令的执行路径。
    def set_commands(self, **kw):
        for k, v in kw.items():
            self.set_command(k, v)

    # 设置单个命令的执行路径。
    def set_command(self, key, value):
        # 如果 key 不在可执行文件关键字列表中，抛出 ValueError 异常
        if not key in self._executable_keys:
            raise ValueError(
                "unknown executable '%s' for class %s" %
                (key, self.__class__.__name__))
        # 如果 value 是字符串类型，将其拆分为列表
        if is_string(value):
            value = split_quoted(value)
        # 断言 value 为 None 或者是字符串列表类型
        assert value is None or is_sequence_of_strings(value[1:]), (key, value)
        # 设置可执行文件字典中指定 key 的值为 value
        self.executables[key] = value

    ######################################################################
    ## Methods that subclasses may redefine. But don't call these methods!
    ## They are private to FCompiler class and may return unexpected
    ## results if used elsewhere. So, you have been warned..

    # 更新可执行文件字典。子类可以重新定义此方法。
    def update_executables(self):
        """Called at the beginning of customisation. Subclasses should
        override this if they need to set up the executables dictionary.

        Note that self.find_executables() is run afterwards, so the
        self.executables dictionary values can contain <F77> or <F90> as
        the command, which will be replaced by the found F77 or F90
        compiler.
        """
        pass

    # 获取通用编译器标志列表。
    def get_flags(self):
        """List of flags common to all compiler types."""
        return [] + self.pic_flags

    # 获取特定于 Fortran 77 的编译器标志列表。
    def _get_command_flags(self, key):
        # 获取指定 key 对应的命令
        cmd = self.executables.get(key, None)
        if cmd is None:
            return []
        # 返回命令的标志列表（去掉第一个元素）
        return cmd[1:]

    # 获取特定于 Fortran 77 的编译器标志列表。
    def get_flags_f77(self):
        """List of Fortran 77 specific flags."""
        return self._get_command_flags('compiler_f77')
    # 返回 Fortran 90 特定编译器标志列表
    def get_flags_f90(self):
        """List of Fortran 90 specific flags."""
        return self._get_command_flags('compiler_f90')

    # 返回 Fortran 90 自由格式特定编译器标志列表（空列表）
    def get_flags_free(self):
        """List of Fortran 90 free format specific flags."""
        return []

    # 返回 Fortran 90 定格式特定编译器标志列表
    def get_flags_fix(self):
        """List of Fortran 90 fixed format specific flags."""
        return self._get_command_flags('compiler_fix')

    # 返回用于构建共享库的链接器标志列表
    def get_flags_linker_so(self):
        """List of linker flags to build a shared library."""
        return self._get_command_flags('linker_so')

    # 返回用于构建可执行文件的链接器标志列表
    def get_flags_linker_exe(self):
        """List of linker flags to build an executable."""
        return self._get_command_flags('linker_exe')

    # 返回归档器标志列表
    def get_flags_ar(self):
        """List of archiver flags. """
        return self._get_command_flags('archiver')

    # 返回体系结构独立的编译器标志列表（空列表）
    def get_flags_opt(self):
        """List of architecture independent compiler flags."""
        return []

    # 返回体系结构相关的编译器标志列表（空列表）
    def get_flags_arch(self):
        """List of architecture dependent compiler flags."""
        return []

    # 返回用于带调试信息编译的编译器标志列表（空列表）
    def get_flags_debug(self):
        """List of compiler flags to compile with debugging information."""
        return []

    # 设置 Fortran 77 和 Fortran 90 版本的体系结构无关编译器标志相同
    get_flags_opt_f77 = get_flags_opt_f90 = get_flags_opt

    # 设置 Fortran 77 和 Fortran 90 版本的体系结构相关编译器标志相同
    get_flags_arch_f77 = get_flags_arch_f90 = get_flags_arch

    # 设置 Fortran 77 和 Fortran 90 版本的调试编译器标志相同
    get_flags_debug_f77 = get_flags_debug_f90 = get_flags_debug

    # 返回编译器库列表
    def get_libraries(self):
        """List of compiler libraries."""
        return self.libraries[:]

    # 返回编译器库目录列表
    def get_library_dirs(self):
        """List of compiler library directories."""
        return self.library_dirs[:]

    # 获取编译器版本信息，如果未找到则引发 CompilerNotFound 异常
    def get_version(self, force=False, ok_status=[0]):
        assert self._is_customised
        version = CCompiler.get_version(self, force=force, ok_status=ok_status)
        if version is None:
            raise CompilerNotFound()
        return version

    ############################################################

    ## Public methods:

    # 打印编译器实例的属性列表
    def dump_properties(self):
        """Print out the attributes of a compiler instance."""
        props = []
        # 收集要打印的属性名
        for key in list(self.executables.keys()) + \
                ['version', 'libraries', 'library_dirs',
                 'object_switch', 'compile_switch']:
            if hasattr(self, key):
                v = getattr(self, key)
                props.append((key, None, '= '+repr(v)))
        props.sort()

        # 使用 FancyGetopt 格式化输出属性信息
        pretty_printer = FancyGetopt(props)
        for l in pretty_printer.generate_help("%s instance properties:" \
                                              % (self.__class__.__name__)):
            if l[:4]=='  --':
                l = '  ' + l[4:]
            print(l)

    ###################
    def _compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
        """Compile 'src' to produce 'obj'."""
        # 初始化一个空字典来存储源文件的特定编译选项
        src_flags = {}
        
        # 检查源文件的后缀名是否在FORTRAN_COMMON_FIXED_EXTENSIONS列表中，并且不含有f90头部信息
        if Path(src).suffix.lower() in FORTRAN_COMMON_FIXED_EXTENSIONS \
           and not has_f90_header(src):
            # 设置为Fortran 77风格
            flavor = ':f77'
            # 使用Fortran 77编译器
            compiler = self.compiler_f77
            # 获取Fortran 77源文件的编译选项
            src_flags = get_f77flags(src)
            # 获取额外的Fortran 77编译选项，如果没有则为空列表
            extra_compile_args = self.extra_f77_compile_args or []
        
        # 如果源文件是自由格式的Fortran 90
        elif is_free_format(src):
            # 设置为Fortran 90风格
            flavor = ':f90'
            # 使用Fortran 90编译器，如果没有则引发异常
            compiler = self.compiler_f90
            if compiler is None:
                raise DistutilsExecError('f90 not supported by %s needed for %s'\
                      % (self.__class__.__name__, src))
            # 获取额外的Fortran 90编译选项，如果没有则为空列表
            extra_compile_args = self.extra_f90_compile_args or []
        
        # 默认情况下，使用固定格式Fortran编译器
        else:
            # 设置为Fortran固定格式风格
            flavor = ':fix'
            # 使用Fortran固定格式编译器，如果没有则引发异常
            compiler = self.compiler_fix
            if compiler is None:
                raise DistutilsExecError('f90 (fixed) not supported by %s needed for %s'\
                      % (self.__class__.__name__, src))
            # 获取额外的Fortran固定格式编译选项，如果没有则为空列表
            extra_compile_args = self.extra_f90_compile_args or []
        
        # 根据self.object_switch的值选择正确的目标文件参数列表
        if self.object_switch[-1]==' ':
            o_args = [self.object_switch.strip(), obj]
        else:
            o_args = [self.object_switch.strip()+obj]
        
        # 确保self.compile_switch不为空，并设置源文件参数列表
        assert self.compile_switch.strip()
        s_args = [self.compile_switch, src]
        
        # 如果存在额外的编译选项，则记录日志
        if extra_compile_args:
            log.info('extra %s options: %r' \
                     % (flavor[1:], ' '.join(extra_compile_args)))
        
        # 获取源文件特定编译选项
        extra_flags = src_flags.get(self.compiler_type, [])
        # 如果存在特定编译选项，则记录日志
        if extra_flags:
            log.info('using compile options from source: %r' \
                     % ' '.join(extra_flags))
        
        # 构建编译命令
        command = compiler + cc_args + extra_flags + s_args + o_args \
                  + extra_postargs + extra_compile_args
        
        # 设置显示信息，包含编译器和源文件名称
        display = '%s: %s' % (os.path.basename(compiler[0]) + flavor,
                              src)
        try:
            # 执行编译命令
            self.spawn(command, display=display)
        except DistutilsExecError as e:
            # 如果发生执行错误，则捕获并重新抛出CompileError异常
            msg = str(e)
            raise CompileError(msg) from None
    # 返回模块选项列表，包括模块目录开关和模块构建目录
    def module_options(self, module_dirs, module_build_dir):
        options = []
        # 检查是否存在模块目录开关
        if self.module_dir_switch is not None:
            # 如果开关的最后一个字符是空格，则扩展选项列表以包括修剪后的开关和模块构建目录
            if self.module_dir_switch[-1]==' ':
                options.extend([self.module_dir_switch.strip(), module_build_dir])
            else:
                # 否则将开关和模块构建目录组合成一个选项并添加到列表中
                options.append(self.module_dir_switch.strip()+module_build_dir)
        else:
            # 如果没有模块目录开关，打印警告信息并指出忽略了模块构建目录选项
            print('XXX: module_build_dir=%r option ignored' % (module_build_dir))
            print('XXX: Fix module_dir_switch for ', self.__class__.__name__)
        # 检查是否存在模块包含目录开关
        if self.module_include_switch is not None:
            # 对于每个模块构建目录和每个模块目录，构建包含目录选项并添加到列表中
            for d in [module_build_dir]+module_dirs:
                options.append('%s%s' % (self.module_include_switch, d))
        else:
            # 如果没有模块包含目录开关，打印警告信息并指出忽略了模块目录选项
            print('XXX: module_dirs=%r option ignored' % (module_dirs))
            print('XXX: Fix module_include_switch for ', self.__class__.__name__)
        # 返回最终的选项列表
        return options

    # 返回链接库选项，以"-l"开头
    def library_option(self, lib):
        return "-l" + lib
    
    # 返回链接库目录选项，以"-L"开头
    def library_dir_option(self, dir):
        return "-L" + dir
    # 将传入的 objects 和 output_dir 参数规范化处理，确保它们符合要求
    objects, output_dir = self._fix_object_args(objects, output_dir)
    
    # 将传入的 libraries、library_dirs 和 runtime_library_dirs 参数规范化处理，确保它们符合要求
    libraries, library_dirs, runtime_library_dirs = \
        self._fix_lib_args(libraries, library_dirs, runtime_library_dirs)
    
    # 根据规范化后的参数生成库选项
    lib_opts = gen_lib_options(self, library_dirs, runtime_library_dirs,
                               libraries)
    
    # 如果 output_dir 是字符串类型，则将 output_filename 放置在 output_dir 中
    if is_string(output_dir):
        output_filename = os.path.join(output_dir, output_filename)
    # 如果 output_dir 不为 None 且不是字符串类型，则抛出类型错误异常
    elif output_dir is not None:
        raise TypeError("'output_dir' must be a string or None")
    
    # 如果需要进行链接操作，则执行以下步骤
    if self._need_link(objects, output_filename):
        # 根据 self.library_switch 的最后一个字符判断需要执行的操作
        if self.library_switch[-1]==' ':
            o_args = [self.library_switch.strip(), output_filename]
        else:
            o_args = [self.library_switch.strip()+output_filename]
        
        # 如果 self.objects 是字符串类型，则将其与 objects 合并
        if is_string(self.objects):
            ld_args = objects + [self.objects]
        else:
            ld_args = objects + self.objects
        
        # 将 ld_args 与 lib_opts 和 o_args 合并
        ld_args = ld_args + lib_opts + o_args
        
        # 如果 debug 标志为真，则在 ld_args 的开头加上 '-g'
        if debug:
            ld_args[:0] = ['-g']
        
        # 如果 extra_preargs 存在，则将其加入到 ld_args 的开头
        if extra_preargs:
            ld_args[:0] = extra_preargs
        
        # 如果 extra_postargs 存在，则将其添加到 ld_args 的末尾
        if extra_postargs:
            ld_args.extend(extra_postargs)
        
        # 确保输出文件的目录存在，如果不存在则创建
        self.mkpath(os.path.dirname(output_filename))
        
        # 根据 target_desc 类型选择相应的链接器
        if target_desc == CCompiler.EXECUTABLE:
            linker = self.linker_exe[:]
        else:
            linker = self.linker_so[:]
        
        # 组合链接器和 ld_args 形成完整的命令
        command = linker + ld_args
        
        # 尝试执行链接命令
        try:
            self.spawn(command)
        except DistutilsExecError as e:
            # 如果发生 DistutilsExecError 异常，则将其转换为 LinkError 异常并抛出
            msg = str(e)
            raise LinkError(msg) from None
    else:
        # 如果不需要进行链接操作，则记录调试信息，表明跳过此文件（因为已经是最新的）
        log.debug("skipping %s (up-to-date)", output_filename)
    
# 根据给定的 hook_name，执行与环境相关的钩子操作，返回相应的结果
def _environment_hook(self, name, hook_name):
    if hook_name is None:
        return None
    if is_string(hook_name):
        # 如果 hook_name 以 'self.' 开头，则从当前对象中获取对应方法并执行
        if hook_name.startswith('self.'):
            hook_name = hook_name[5:]
            hook = getattr(self, hook_name)
            return hook()
        # 如果 hook_name 以 'exe.' 开头，则从 executables 属性中获取对应的变量值
        elif hook_name.startswith('exe.'):
            hook_name = hook_name[4:]
            var = self.executables[hook_name]
            if var:
                return var[0]
            else:
                return None
        # 如果 hook_name 以 'flags.' 开头，则调用相应的 get_flags_ 方法获取标志
        elif hook_name.startswith('flags.'):
            hook_name = hook_name[6:]
            hook = getattr(self, 'get_flags_' + hook_name)
            return hook()
    else:
        # 如果 hook_name 不是字符串类型，则直接返回它
        return hook_name()

# 检查给定的 C 编译器是否能够链接当前编译器生成的对象
def can_ccompiler_link(self, ccompiler):
    """
    检查给定的 C 编译器是否能够链接当前编译器生成的对象。
    """
    return True
    # 定义一个方法，用于包装无法与默认链接器兼容的对象文件，使其兼容
    def wrap_unlinkable_objects(self, objects, output_dir, extra_dll_dir):
        """
        Convert a set of object files that are not compatible with the default
        linker, to a file that is compatible.
    
        Parameters
        ----------
        objects : list
            List of object files to include.
        output_dir : str
            Output directory to place generated object files.
        extra_dll_dir : str
            Output directory to place extra DLL files that need to be
            included on Windows.
    
        Returns
        -------
        converted_objects : list of str
             List of converted object files.
             Note that the number of output files is not necessarily
             the same as inputs.
    
        """
        # 抛出一个未实现错误，表示该方法需要在子类中实现
        raise NotImplementedError()
    
    ## class FCompiler
# 默认的编译器配置，根据不同的操作系统平台和名称映射到对应的编译器列表
_default_compilers = (
    # 对应 win32 平台的编译器列表
    ('win32', ('gnu', 'intelv', 'absoft', 'compaqv', 'intelev', 'gnu95', 'g95',
               'intelvem', 'intelem', 'flang')),
    # 对应 cygwin.* 平台的编译器列表
    ('cygwin.*', ('gnu', 'intelv', 'absoft', 'compaqv', 'intelev', 'gnu95', 'g95')),
    # 对应 linux.* 平台的编译器列表
    ('linux.*', ('arm', 'gnu95', 'intel', 'lahey', 'pg', 'nv', 'absoft', 'nag',
                 'vast', 'compaq', 'intele', 'intelem', 'gnu', 'g95', 
                 'pathf95', 'nagfor', 'fujitsu')),
    # 对应 darwin.* 平台的编译器列表
    ('darwin.*', ('gnu95', 'nag', 'nagfor', 'absoft', 'ibm', 'intel', 'gnu',
                 'g95', 'pg')),
    # 对应 sunos.* 平台的编译器列表
    ('sunos.*', ('sun', 'gnu', 'gnu95', 'g95')),
    # 对应 irix.* 平台的编译器列表
    ('irix.*', ('mips', 'gnu', 'gnu95',)),
    # 对应 aix.* 平台的编译器列表
    ('aix.*', ('ibm', 'gnu', 'gnu95',)),
    # 对应 posix 平台的编译器列表
    ('posix', ('gnu', 'gnu95',)),
    # 对应 nt 平台的编译器列表
    ('nt', ('gnu', 'gnu95',)),
    # 对应 mac 平台的编译器列表
    ('mac', ('gnu95', 'gnu', 'pg')),
)

# 默认情况下，未指定任何编译器类
fcompiler_class = None
# 默认情况下，未指定任何编译器别名
fcompiler_aliases = None

def load_all_fcompiler_classes():
    """缓存所有在 numpy.distutils.fcompiler 包中找到的 FCompiler 类。

    使用 glob 模块获取当前文件夹下所有的 Python 模块文件，并尝试导入这些模块，
    将其中定义的编译器类及其别名缓存起来。

    """
    from glob import glob
    global fcompiler_class, fcompiler_aliases
    # 如果已经缓存了编译器类信息，则直接返回，避免重复加载
    if fcompiler_class is not None:
        return
    # 获取当前文件所在文件夹下所有的 Python 模块文件路径
    pys = os.path.join(os.path.dirname(__file__), '*.py')
    # 初始化编译器类字典和别名字典
    fcompiler_class = {}
    fcompiler_aliases = {}
    # 遍历所有匹配的模块文件路径
    for fname in glob(pys):
        # 提取模块文件名（不包含扩展名）
        module_name, ext = os.path.splitext(os.path.basename(fname))
        # 构建完整的模块名称
        module_name = 'numpy.distutils.fcompiler.' + module_name
        # 动态导入模块
        __import__(module_name)
        # 获取导入的模块对象
        module = sys.modules[module_name]
        # 检查模块是否定义了编译器列表
        if hasattr(module, 'compilers'):
            # 遍历模块中定义的编译器类名
            for cname in module.compilers:
                # 获取编译器类对象
                klass = getattr(module, cname)
                # 构建编译器描述信息
                desc = (klass.compiler_type, klass, klass.description)
                # 将编译器类型及其类对象和描述信息添加到编译器类字典中
                fcompiler_class[klass.compiler_type] = desc
                # 遍历编译器类定义的别名列表
                for alias in klass.compiler_aliases:
                    # 检查别名是否已经存在于别名字典中，避免重复定义
                    if alias in fcompiler_aliases:
                        raise ValueError("alias %r defined for both %s and %s"
                                         % (alias, klass.__name__,
                                            fcompiler_aliases[alias][1].__name__))
                    # 将别名及其描述信息添加到别名字典中
                    fcompiler_aliases[alias] = desc

def _find_existing_fcompiler(compiler_types,
                             osname=None, platform=None,
                             requiref90=False,
                             c_compiler=None):
    """根据指定的条件查找现有的编译器。

    使用 numpy.distutils.core 模块中的 get_distribution 函数获取当前发行版信息，
    以确定编译器的查找范围。

    Args:
        compiler_types: 待查找的编译器类型列表。
        osname: 操作系统名称。
        platform: 平台名称。
        requiref90: 是否需要支持 Fortran 90 编译。
        c_compiler: C 编译器类型。

    """
    from numpy.distutils.core import get_distribution
    # 获取当前的发行版信息
    dist = get_distribution(always=True)
    # 遍历编译器类型列表，逐个尝试不同的编译器类型
    for compiler_type in compiler_types:
        # 初始化变量 v 为 None，用于存储编译器版本信息
        v = None
        try:
            # 创建新的编译器对象 c，传入平台和当前编译器类型
            c = new_fcompiler(plat=platform, compiler=compiler_type,
                              c_compiler=c_compiler)
            # 根据传入的配置信息对编译器对象 c 进行定制化设置
            c.customize(dist)
            # 获取当前编译器对象 c 的版本信息
            v = c.get_version()
            
            # 如果需要支持 Fortran 90，并且当前编译器不支持 Fortran 90
            if requiref90 and c.compiler_f90 is None:
                # 重置 v 为 None
                v = None
                # 获取推荐的 Fortran 90 编译器
                new_compiler = c.suggested_f90_compiler
                if new_compiler:
                    # 输出警告信息，尝试推荐的 Fortran 90 编译器
                    log.warn('Trying %r compiler as suggested by %r '
                             'compiler for f90 support.' % (compiler_type,
                                                            new_compiler))
                    # 使用推荐的 Fortran 90 编译器创建新的编译器对象 c
                    c = new_fcompiler(plat=platform, compiler=new_compiler,
                                      c_compiler=c_compiler)
                    # 对新的编译器对象 c 进行定制化设置
                    c.customize(dist)
                    # 获取新编译器对象的版本信息
                    v = c.get_version()
                    # 如果获取到版本信息，则更新编译器类型为新的编译器类型
                    if v is not None:
                        compiler_type = new_compiler
                        
            # 如果需要支持 Fortran 90 且当前编译器不支持，抛出异常
            if requiref90 and c.compiler_f90 is None:
                raise ValueError('%s does not support compiling f90 codes, '
                                 'skipping.' % (c.__class__.__name__))
                                 
        # 捕获 Distutils 模块错误异常
        except DistutilsModuleError:
            log.debug("_find_existing_fcompiler: compiler_type='%s' raised DistutilsModuleError", compiler_type)
            
        # 捕获编译器未找到异常
        except CompilerNotFound:
            log.debug("_find_existing_fcompiler: compiler_type='%s' not found", compiler_type)
            
        # 如果获取到版本信息 v，则返回当前编译器类型
        if v is not None:
            return compiler_type
    
    # 如果未找到适用的编译器类型，则返回 None
    return None
# 确定适用于特定操作系统和平台的可用 Fortran 编译器类型列表
def available_fcompilers_for_platform(osname=None, platform=None):
    if osname is None:
        osname = os.name
    if platform is None:
        platform = sys.platform
    matching_compiler_types = []
    for pattern, compiler_type in _default_compilers:
        # 如果操作系统或平台名称与给定模式匹配，则将对应的编译器类型添加到列表中
        if re.match(pattern, platform) or re.match(pattern, osname):
            for ct in compiler_type:
                if ct not in matching_compiler_types:
                    matching_compiler_types.append(ct)
    if not matching_compiler_types:
        matching_compiler_types.append('gnu')  # 如果没有匹配的编译器类型，则默认使用 'gnu'
    return matching_compiler_types

# 获取适合特定平台的默认 Fortran 编译器
def get_default_fcompiler(osname=None, platform=None, requiref90=False,
                          c_compiler=None):
    """Determine the default Fortran compiler to use for the given
    platform."""
    matching_compiler_types = available_fcompilers_for_platform(osname,
                                                                platform)
    # 记录匹配的编译器类型到日志
    log.info("get_default_fcompiler: matching types: '%s'",
             matching_compiler_types)
    # 根据匹配的编译器类型查找现有的 Fortran 编译器
    compiler_type =  _find_existing_fcompiler(matching_compiler_types,
                                              osname=osname,
                                              platform=platform,
                                              requiref90=requiref90,
                                              c_compiler=c_compiler)
    return compiler_type

# 用于避免每次重新检查 Fortran 编译器的标志集合
failed_fcompilers = set()

# 为给定的平台和编译器组合生成某个 FCompiler 子类的实例
def new_fcompiler(plat=None,
                  compiler=None,
                  verbose=0,
                  dry_run=0,
                  force=0,
                  requiref90=False,
                  c_compiler=None):
    """Generate an instance of some FCompiler subclass for the supplied
    platform/compiler combination.
    """
    global failed_fcompilers
    fcompiler_key = (plat, compiler)
    # 如果已经知道给定平台和编译器组合的编译失败，则返回 None
    if fcompiler_key in failed_fcompilers:
        return None

    # 加载所有的 Fortran 编译器类
    load_all_fcompiler_classes()
    if plat is None:
        plat = os.name
    if compiler is None:
        # 获取默认的 Fortran 编译器
        compiler = get_default_fcompiler(plat, requiref90=requiref90,
                                         c_compiler=c_compiler)
    # 根据编译器名称查找对应的模块名、类和详细描述
    if compiler in fcompiler_class:
        module_name, klass, long_description = fcompiler_class[compiler]
    elif compiler in fcompiler_aliases:
        module_name, klass, long_description = fcompiler_aliases[compiler]
    else:
        # 如果未知如何在特定平台上编译 Fortran 代码，则记录警告并添加到失败编译器集合中
        msg = "don't know how to compile Fortran code on platform '%s'" % plat
        if compiler is not None:
            msg = msg + " with '%s' compiler." % compiler
            msg = msg + " Supported compilers are: %s)" \
                  % (','.join(fcompiler_class.keys()))
        log.warn(msg)
        failed_fcompilers.add(fcompiler_key)
        return None

    # 创建编译器实例并返回
    compiler = klass(verbose=verbose, dry_run=dry_run, force=force)
    compiler.c_compiler = c_compiler
    return compiler

# 显示 Fortran 编译器信息
def show_fcompilers(dist=None):
    """Print list of available compilers (used by the "--help-fcompiler"
    option to "config_fc").
    """

    # 如果没有传入 dist 参数，则从 distutils.dist 模块导入 Distribution 类
    if dist is None:
        from distutils.dist import Distribution
        # 从 numpy.distutils.command.config_compiler 模块导入 config_fc 函数
        from numpy.distutils.command.config_compiler import config_fc
        # 创建 Distribution 对象
        dist = Distribution()
        # 设置 Distribution 对象的脚本名称为当前脚本的基本文件名
        dist.script_name = os.path.basename(sys.argv[0])
        # 设置 Distribution 对象的脚本参数，包括 'config_fc' 和后续的命令行参数
        dist.script_args = ['config_fc'] + sys.argv[1:]
        # 尝试从脚本参数中移除 '--help-fcompiler'
        try:
            dist.script_args.remove('--help-fcompiler')
        except ValueError:
            pass
        # 将 config_fc 函数设置为 Distribution 对象的 cmdclass 中的 'config_fc' 命令类
        dist.cmdclass['config_fc'] = config_fc
        # 解析配置文件
        dist.parse_config_files()
        # 解析命令行参数
        dist.parse_command_line()

    # 初始化三个空列表来存储编译器信息
    compilers = []
    compilers_na = []
    compilers_ni = []

    # 如果没有指定 fcompiler_class，则加载所有的编译器类
    if not fcompiler_class:
        load_all_fcompiler_classes()

    # 获取当前平台上可用的编译器列表
    platform_compilers = available_fcompilers_for_platform()

    # 遍历平台上可用的每个编译器
    for compiler in platform_compilers:
        v = None
        # 设置日志的详细程度为 -2
        log.set_verbosity(-2)
        try:
            # 创建一个新的编译器实例，并根据 dist 对象的详细程度设置是否显示详细信息
            c = new_fcompiler(compiler=compiler, verbose=dist.verbose)
            # 根据 dist 对象自定义编译器
            c.customize(dist)
            # 获取编译器的版本信息
            v = c.get_version()
        except (DistutilsModuleError, CompilerNotFound) as e:
            # 如果出现异常，记录未找到编译器的调试信息
            log.debug("show_fcompilers: %s not found" % (compiler,))
            log.debug(repr(e))

        # 根据获取到的版本信息将编译器信息添加到不同的列表中
        if v is None:
            compilers_na.append(("fcompiler="+compiler, None,
                              fcompiler_class[compiler][2]))
        else:
            # 打印编译器的属性信息
            c.dump_properties()
            compilers.append(("fcompiler="+compiler, None,
                              fcompiler_class[compiler][2] + ' (%s)' % v))

    # 找出当前平台上未安装的编译器，并添加到对应的列表中
    compilers_ni = list(set(fcompiler_class.keys()) - set(platform_compilers))
    compilers_ni = [("fcompiler="+fc, None, fcompiler_class[fc][2])
                    for fc in compilers_ni]

    # 对三个列表进行排序
    compilers.sort()
    compilers_na.sort()
    compilers_ni.sort()

    # 使用 FancyGetopt 类创建一个美观的打印对象，并打印不同列表中的编译器信息
    pretty_printer = FancyGetopt(compilers)
    pretty_printer.print_help("Fortran compilers found:")

    pretty_printer = FancyGetopt(compilers_na)
    pretty_printer.print_help("Compilers available for this "
                              "platform, but not found:")

    # 如果有不适用于当前平台的编译器，则打印这些编译器信息
    if compilers_ni:
        pretty_printer = FancyGetopt(compilers_ni)
        pretty_printer.print_help("Compilers not available on this platform:")

    # 打印一条消息，指示如何查看编译器详细信息
    print("For compiler details, run 'config_fc --verbose' setup command.")
# 创建一个临时的 Fortran 文件，写入简单的 subroutine 定义，并返回文件名
def dummy_fortran_file():
    fo, name = make_temp_file(suffix='.f')
    fo.write("      subroutine dummy()\n      end\n")
    fo.close()
    return name[:-2]

# 正则表达式，用于检测文件头部是否包含 Fortran 标记
_has_f_header = re.compile(r'-\*-\s*fortran\s*-\*-', re.I).search

# 正则表达式，用于检测文件头部是否包含 f90 标记
_has_f90_header = re.compile(r'-\*-\s*f90\s*-\*-', re.I).search

# 正则表达式，用于检测文件头部是否包含 fix 标记
_has_fix_header = re.compile(r'-\*-\s*fix\s*-\*-', re.I).search

# 正则表达式，用于检测文件是否采用自由格式的 Fortran
_free_f90_start = re.compile(r'[^c*!]\s*[^\s\d\t]', re.I).match

def is_free_format(file):
    """Check if file is in free format Fortran."""
    # 默认假设为 fixed 格式，除非检测到自由格式的迹象。
    result = 0
    with open(file, encoding='latin1') as f:
        line = f.readline()
        n = 10000  # 扫描非注释行的最大行数
        if _has_f_header(line) or _has_fix_header(line):
            n = 0
        elif _has_f90_header(line):
            n = 0
            result = 1
        while n > 0 and line:
            line = line.rstrip()
            if line and line[0] != '!':
                n -= 1
                # 检测自由格式 Fortran 的特征
                if (line[0] != '\t' and _free_f90_start(line[:5])) or line[-1:] == '&':
                    result = 1
                    break
            line = f.readline()
    return result

def has_f90_header(src):
    """Check if source file has an f90 header."""
    with open(src, encoding='latin1') as f:
        line = f.readline()
    return _has_f90_header(line) or _has_fix_header(line)

# 正则表达式，用于解析 Fortran 77 代码中的编译器标志
_f77flags_re = re.compile(r'(c|)f77flags\s*\(\s*(?P<fcname>\w+)\s*\)\s*=\s*(?P<fflags>.*)', re.I)

def get_f77flags(src):
    """
    Search the first 20 lines of fortran 77 code for line pattern
      `CF77FLAGS(<fcompiler type>)=<f77 flags>`
    Return a dictionary {<fcompiler type>:<f77 flags>}.
    """
    flags = {}
    with open(src, encoding='latin1') as f:
        i = 0
        for line in f:
            i += 1
            if i > 20:
                break
            m = _f77flags_re.match(line)
            if not m:
                continue
            fcname = m.group('fcname').strip()
            fflags = m.group('fflags').strip()
            flags[fcname] = split_quoted(fflags)
    return flags

# TODO: implement get_f90flags and use it in _compile similarly to get_f77flags

if __name__ == '__main__':
    show_fcompilers()
```