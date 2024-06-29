# `.\numpy\numpy\distutils\command\build_ext.py`

```
"""
# 导入所需要的模块
import os  # 导入操作系统模块
import subprocess  # 导入子进程管理模块
from glob import glob  # 从 glob 模块中导入 glob 函数

from distutils.dep_util import newer_group  # 从 distutils.dep_util 模块中导入 newer_group 函数
from distutils.command.build_ext import build_ext as old_build_ext  # 从 distutils.command.build_ext 模块中导入 build_ext 类
from distutils.errors import DistutilsFileError, DistutilsSetupError, DistutilsError  # 从 distutils.errors 模块中导入错误类
from distutils.file_util import copy_file  # 从 distutils.file_util 模块中导入 copy_file 函数

from numpy.distutils import log  # 从 numpy.distutils 模块中导入 log 模块
from numpy.distutils.exec_command import filepath_from_subprocess_output  # 从 numpy.distutils.exec_command 模块中导入 filepath_from_subprocess_output 函数
from numpy.distutils.system_info import combine_paths  # 从 numpy.distutils.system_info 模块中导入 combine_paths 函数
from numpy.distutils.misc_util import (  # 从 numpy.distutils.misc_util 模块中导入多个函数
    filter_sources, get_ext_source_files, get_numpy_include_dirs,
    has_cxx_sources, has_f_sources, is_sequence
)
from numpy.distutils.command.config_compiler import show_fortran_compilers  # 从 numpy.distutils.command.config_compiler 模块中导入 show_fortran_compilers 函数
from numpy.distutils.ccompiler_opt import new_ccompiler_opt, CCompilerOpt  # 从 numpy.distutils.ccompiler_opt 模块中导入 new_ccompiler_opt 函数和 CCompilerOpt 类

# 自定义 build_ext 类，继承自 old_build_ext 类
class build_ext (old_build_ext):

    # 描述信息
    description = "build C/C++/F extensions (compile/link to build directory)"

    # 用户选项
    user_options = old_build_ext.user_options + [  # 继承父类的用户选项
        ('fcompiler=', None,  # 指定 Fortran 编译器类型
         "specify the Fortran compiler type"),
        ('parallel=', 'j',  # 指定并行作业数
         "number of parallel jobs"),
        ('warn-error', None,  # 将所有警告转换为错误
         "turn all warnings into errors (-Werror)"),
        ('cpu-baseline=', None,  # 指定启用的基线 CPU 优化列表
         "specify a list of enabled baseline CPU optimizations"),
        ('cpu-dispatch=', None,  # 指定调度 CPU 优化列表
         "specify a list of dispatched CPU optimizations"),
        ('disable-optimization', None,  # 禁用 CPU 优化代码（调度，simd，fast...）
         "disable CPU optimized code(dispatch,simd,fast...)"),
        ('simd-test=', None,  # 指定要针对 NumPy SIMD 接口测试的 CPU 优化列表
         "specify a list of CPU optimizations to be tested against NumPy SIMD interface"),
    ]

    # 帮助选项
    help_options = old_build_ext.help_options + [  # 继承父类的帮助选项
        ('help-fcompiler', None, "list available Fortran compilers",  # 列出可用的 Fortran 编译器
         show_fortran_compilers),
    ]

    # 布尔选项
    boolean_options = old_build_ext.boolean_options + ['warn-error', 'disable-optimization']  # 继承父类的布尔选项

    # 初始化选项
    def initialize_options(self):
        old_build_ext.initialize_options(self)  # 调用父类的初始化选项方法
        self.fcompiler = None  # Fortran 编译器类型
        self.parallel = None  # 并行作业数
        self.warn_error = None  # 所有警告转换为错误
        self.cpu_baseline = None  # 启用的基线 CPU 优化列表
        self.cpu_dispatch = None  # 调度 CPU 优化列表
        self.disable_optimization = None  # 禁用 CPU 优化代码
        self.simd_test = None  # 针对 NumPy SIMD 接口测试的 CPU 优化列表
    # 确定选项的最终值
    def finalize_options(self):
        # 如果使用了并行选项，则将其转换为整数类型，否则抛出数值错误异常
        if self.parallel:
            try:
                self.parallel = int(self.parallel)
            except ValueError as e:
                raise ValueError("--parallel/-j argument must be an integer") from e

        # 确保 self.include_dirs 和 self.distribution.include_dirs 引用的是同一个列表对象
        # finalize_options 将修改 self.include_dirs，但实际构建过程中使用的是 self.distribution.include_dirs
        # 在没有指定路径的情况下，self.include_dirs 为 None
        # include 路径将按顺序传递给编译器: numpy 路径，--include-dirs 路径，Python include 路径
        if isinstance(self.include_dirs, str):
            self.include_dirs = self.include_dirs.split(os.pathsep)
        incl_dirs = self.include_dirs or []
        if self.distribution.include_dirs is None:
            self.distribution.include_dirs = []
        self.include_dirs = self.distribution.include_dirs
        self.include_dirs.extend(incl_dirs)

        # 调用旧的 build_ext.finalize_options 方法
        old_build_ext.finalize_options(self)
        # 设置未定义的选项
        self.set_undefined_options('build',
                                        ('parallel', 'parallel'),
                                        ('warn_error', 'warn_error'),
                                        ('cpu_baseline', 'cpu_baseline'),
                                        ('cpu_dispatch', 'cpu_dispatch'),
                                        ('disable_optimization', 'disable_optimization'),
                                        ('simd_test', 'simd_test')
                                  )
        # 更新 CCompilerOpt.conf_target_groups["simd_test"] 为 self.simd_test
        CCompilerOpt.conf_target_groups["simd_test"] = self.simd_test

    # 处理 swig 源文件，这里不做任何操作，swig 源文件已经在 build_src 命令中处理过了
    def swig_sources(self, sources, extensions=None):
        # 并不做任何操作。Swig 源文件已经在 build_src 命令中处理过了。
        return sources

    # 向 mingwex_sym 添加一个虚拟符号
    def _add_dummy_mingwex_sym(self, c_sources):
        # 获取 "build_src" 命令的 build_src 属性
        build_src = self.get_finalized_command("build_src").build_src
        # 获取 "build_clib" 命令的 build_clib 属性
        build_clib = self.get_finalized_command("build_clib").build_clib
        # 编译 gfortran_vs2003_hack.c 文件，并输出到 build_temp 目录
        objects = self.compiler.compile([os.path.join(build_src, "gfortran_vs2003_hack.c")],
                                        output_dir=self.build_temp)
        # 创建名为 "_gfortran_workaround" 的静态库，输出到 build_clib 目录，如果启用了 debug 模式则输出调试信息
        self.compiler.create_static_lib(
            objects, "_gfortran_workaround", output_dir=build_clib, debug=self.debug)
    # 处理不可链接的目标文件
    def _process_unlinkable_fobjects(self, objects, libraries,
                                     fcompiler, library_dirs,
                                     unlinkable_fobjects):
        # 将参数转换为列表
        libraries = list(libraries)
        objects = list(objects)
        unlinkable_fobjects = list(unlinkable_fobjects)

        # 将可能的假静态库扩展为对象文件；
        # 确保迭代列表的副本，因为遇到“假”库时会被删除
        for lib in libraries[:]:
            for libdir in library_dirs:
                fake_lib = os.path.join(libdir, lib + '.fobjects')
                if os.path.isfile(fake_lib):
                    # 替换假静态库
                    libraries.remove(lib)
                    with open(fake_lib) as f:
                        # 将假静态库的内容添加到不可链接的对象列表中
                        unlinkable_fobjects.extend(f.read().splitlines())

                    # 扩展C对象
                    c_lib = os.path.join(libdir, lib + '.cobjects')
                    with open(c_lib) as f:
                        # 将C对象的内容添加到对象列表中
                        objects.extend(f.read().splitlines())

        # 用链接的对象包装不可链接的对象
        if unlinkable_fobjects:
            # 将不可链接的对象转换为绝对路径
            fobjects = [os.path.abspath(obj) for obj in unlinkable_fobjects]
            # 使用 fcompiler 的方法包装不可链接的对象
            wrapped = fcompiler.wrap_unlinkable_objects(
                    fobjects, output_dir=self.build_temp,
                    extra_dll_dir=self.extra_dll_dir)
            # 将结果追加到对象列表中
            objects.extend(wrapped)

        # 返回更新后的对象列表和库列表
        return objects, libraries
    # 检查是否有指定的编译器和库文件目录，将 g77 编译的静态库文件转换为 MSVC 可用格式
    def _libs_with_msvc_and_fortran(self, fcompiler, c_libraries, c_library_dirs):
        # 如果没有指定编译器，则返回
        if fcompiler is None:
            return
    
        # 遍历传入的 C 库文件列表
        for libname in c_libraries:
            # 若库文件名以 'msvc' 开头，则跳过
            if libname.startswith('msvc'):
                continue
            fileexists = False
            # 遍历传入的 C 库文件目录
            for libdir in c_library_dirs or []:
                # 拼接库文件路径
                libfile = os.path.join(libdir, '%s.lib' % (libname))
                # 如果文件存在，标记为存在，跳出循环
                if os.path.isfile(libfile):
                    fileexists = True
                    break
            if fileexists:
                continue
            # 将 g77 编译的静态库文件转换为 MSVC 可用格式
            fileexists = False
            for libdir in c_library_dirs:
                libfile = os.path.join(libdir, 'lib%s.a' % (libname))
                if os.path.isfile(libfile):
                    # 将 libname.a 文件复制为 name.lib，以便 MSVC 链接器可以找到它
                    libfile2 = os.path.join(self.build_temp, libname + '.lib')
                    copy_file(libfile, libfile2)
                    if self.build_temp not in c_library_dirs:
                        c_library_dirs.append(self.build_temp)
                    fileexists = True
                    break
            if fileexists:
                continue
            # 若找不到库文件，记录警告信息
            log.warn('could not find library %r in directories %s' % (libname, c_library_dirs))
    
        # 使用 MSVC 编译器时，始终使用系统链接器
        f_lib_dirs = []
        for dir in fcompiler.library_dirs:
            # 编译在 Cygwin 环境，但使用普通 Windows Python 时，纠正路径
            if dir.startswith('/usr/lib'):
                try:
                    dir = subprocess.check_output(['cygpath', '-w', dir])
                except (OSError, subprocess.CalledProcessError):
                    pass
                else:
                    dir = filepath_from_subprocess_output(dir)
            f_lib_dirs.append(dir)
        c_library_dirs.extend(f_lib_dirs)
    
        # 将 g77 编译的静态库文件转换为 MSVC 可用格式
        for lib in fcompiler.libraries:
            if not lib.startswith('msvc'):
                c_libraries.append(lib)
                p = combine_paths(f_lib_dirs, 'lib' + lib + '.a')
                if p:
                    dst_name = os.path.join(self.build_temp, lib + '.lib')
                    if not os.path.isfile(dst_name):
                        copy_file(p[0], dst_name)
                    if self.build_temp not in c_library_dirs:
                        c_library_dirs.append(self.build_temp)
    
    # 获取源文件列表
    def get_source_files(self):
        self.check_extensions_list(self.extensions)
        filenames = []
        for ext in self.extensions:
            filenames.extend(get_ext_source_files(ext))
        return filenames
    # 获取模块的输出文件列表
    def get_outputs(self):
        # 检查扩展列表中是否包含有效扩展
        self.check_extensions_list(self.extensions)
    
        # 初始化输出文件列表
        outputs = []
        
        # 遍历每个扩展
        for ext in self.extensions:
            # 如果没有源文件，则跳过
            if not ext.sources:
                continue
            # 获取扩展的完整名称
            fullname = self.get_ext_fullname(ext.name)
            # 将输出文件的完整路径添加到输出列表中
            outputs.append(os.path.join(self.build_lib,
                                        self.get_ext_filename(fullname)))
        
        # 返回输出文件列表
        return outputs
```