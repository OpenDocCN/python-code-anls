# `.\numpy\numpy\distutils\fcompiler\pg.py`

```py
# 导入 sys 模块
import sys

# 从 numpy.distutils.fcompiler 模块导入 FCompiler 类
from numpy.distutils.fcompiler import FCompiler
# 从 sys 模块导入 platform 函数
from sys import platform
# 从 os.path 模块导入 join, dirname, normpath 函数
from os.path import join, dirname, normpath

# 定义 PGroupFCompiler 类，继承自 FCompiler 类
class PGroupFCompiler(FCompiler):

    # 编译器类型
    compiler_type = 'pg'
    # 描述信息
    description = 'Portland Group Fortran Compiler'
    # 版本匹配模式
    version_pattern = r'\s*pg(f77|f90|hpf|fortran) (?P<version>[\d.-]+).*'

    # 根据平台不同选择不同的执行文件配置
    if platform == 'darwin':
        executables = {
            'version_cmd': ["<F77>", "-V"],  # 版本命令
            'compiler_f77': ["pgfortran", "-dynamiclib"],  # Fortran 77 编译器
            'compiler_fix': ["pgfortran", "-Mfixed", "-dynamiclib"],  # 修正格式 Fortran 编译器
            'compiler_f90': ["pgfortran", "-dynamiclib"],  # Fortran 90 编译器
            'linker_so': ["libtool"],  # 共享库链接器
            'archiver': ["ar", "-cr"],  # 静态库打包工具
            'ranlib': ["ranlib"]  # 静态库索引生成工具
        }
        pic_flags = ['']  # 位置无关代码标志
    else:
        executables = {
            'version_cmd': ["<F77>", "-V"],  # 版本命令
            'compiler_f77': ["pgfortran"],  # Fortran 77 编译器
            'compiler_fix': ["pgfortran", "-Mfixed"],  # 修正格式 Fortran 编译器
            'compiler_f90': ["pgfortran"],  # Fortran 90 编译器
            'linker_so': ["<F90>"],  # 共享库链接器
            'archiver': ["ar", "-cr"],  # 静态库打包工具
            'ranlib': ["ranlib"]  # 静态库索引生成工具
        }
        pic_flags = ['-fpic']  # 位置无关代码标志

    module_dir_switch = '-module '  # 模块目录开关
    module_include_switch = '-I'  # 模块包含开关

    # 获取编译器标志
    def get_flags(self):
        opt = ['-Minform=inform', '-Mnosecond_underscore']
        return self.pic_flags + opt  # 返回位置无关代码标志和优化选项

    # 获取优化标志
    def get_flags_opt(self):
        return ['-fast']  # 返回快速优化标志

    # 获取调试标志
    def get_flags_debug(self):
        return ['-g']  # 返回调试标志

    # 根据平台不同获取链接共享库标志
    if platform == 'darwin':
        def get_flags_linker_so(self):
            return ["-dynamic", '-undefined', 'dynamic_lookup']  # 返回动态链接标志
    else:
        def get_flags_linker_so(self):
            return ["-shared", '-fpic']  # 返回共享库和位置无关代码标志

    # 运行时库目录选项
    def runtime_library_dir_option(self, dir):
        return '-R%s' % dir  # 返回运行时库目录选项


# 导入 functools 模块
import functools

# 定义 PGroupFlangCompiler 类，继承自 FCompiler 类
class PGroupFlangCompiler(FCompiler):
    # 编译器类型
    compiler_type = 'flang'
    # 描述信息
    description = 'Portland Group Fortran LLVM Compiler'
    # 版本匹配模式
    version_pattern = r'\s*(flang|clang) version (?P<version>[\d.-]+).*'

    ar_exe = 'lib.exe'  # 静态库打包工具
    possible_executables = ['flang']  # 可能的可执行文件列表

    executables = {
        'version_cmd': ["<F77>", "--version"],  # 版本命令
        'compiler_f77': ["flang"],  # Fortran 77 编译器
        'compiler_fix': ["flang"],  # 修正格式 Fortran 编译器
        'compiler_f90': ["flang"],  # Fortran 90 编译器
        'linker_so': [None],  # 共享库链接器（空值表示无效）
        'archiver': [ar_exe, "/verbose", "/OUT:"],  # 静态库打包工具
        'ranlib': None  # 静态库索引生成工具（空值表示无效）
    }

    library_switch = '/OUT:'  # 库开关
    module_dir_switch = '-module '  # 模块目录开关

    # 获取库列表
    def get_libraries(self):
        opt = FCompiler.get_libraries(self)
        opt.extend(['flang', 'flangrti', 'ompstub'])
        return opt  # 返回扩展后的库列表

    # 获取库目录列表
    @functools.lru_cache(maxsize=128)
    def get_library_dirs(self):
        """List of compiler library directories."""
        opt = FCompiler.get_library_dirs(self)
        flang_dir = dirname(self.executables['compiler_f77'][0])
        opt.append(normpath(join(flang_dir, '..', 'lib')))
        return opt  # 返回库目录列表

    # 获取编译器标志
    def get_flags(self):
        return []  # 返回空列表（无特定编译器标志）
    # 返回一个空列表，表示没有额外的编译选项
    def get_flags_free(self):
        return []
    
    # 返回一个包含编译调试信息的选项列表，这里只包含了 '-g'，表示生成调试信息
    def get_flags_debug(self):
        return ['-g']
    
    # 返回一个包含编译优化选项的列表，这里只包含了 '-O3'，表示进行高级优化
    def get_flags_opt(self):
        return ['-O3']
    
    # 返回一个空列表，表示没有特定的架构相关的编译选项
    def get_flags_arch(self):
        return []
    
    # 抛出一个未实现的异常，表示该方法在子类中需要被实现
    def runtime_library_dir_option(self, dir):
        raise NotImplementedError
if __name__ == '__main__':
    # 检查是否当前脚本作为主程序运行
    from distutils import log
    # 从 distutils 模块导入 log 功能
    log.set_verbosity(2)
    # 设置日志的详细程度为 2 (详细输出)
    from numpy.distutils import customized_fcompiler
    # 从 numpy.distutils 模块导入 customized_fcompiler 函数
    if 'flang' in sys.argv:
        # 如果 'flang' 存在于命令行参数中
        print(customized_fcompiler(compiler='flang').get_version())
        # 使用 flang 编译器获取其版本并打印
    else:
        # 如果 'flang' 不存在于命令行参数中
        print(customized_fcompiler(compiler='pg').get_version())
        # 使用 pg 编译器获取其版本并打印
```