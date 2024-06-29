# `.\numpy\numpy\distutils\fcompiler\lahey.py`

```
# 导入标准库 os
import os

# 从 numpy.distutils.fcompiler 中导入 FCompiler 类
from numpy.distutils.fcompiler import FCompiler

# 定义一个名为 compilers 的列表，包含字符串 'LaheyFCompiler'
compilers = ['LaheyFCompiler']

# 定义 LaheyFCompiler 类，继承自 FCompiler 类
class LaheyFCompiler(FCompiler):

    # 定义编译器类型为 'lahey'
    compiler_type = 'lahey'

    # 编译器的描述信息
    description = 'Lahey/Fujitsu Fortran 95 Compiler'

    # 版本号的正则表达式模式
    version_pattern =  r'Lahey/Fujitsu Fortran 95 Compiler Release (?P<version>[^\s*]*)'

    # 定义一些可执行命令的字典
    executables = {
        'version_cmd'  : ["<F90>", "--version"],   # 版本命令
        'compiler_f77' : ["lf95", "--fix"],        # Fortran 77 编译器命令
        'compiler_fix' : ["lf95", "--fix"],        # 修正命令
        'compiler_f90' : ["lf95"],                 # Fortran 90 编译器命令
        'linker_so'    : ["lf95", "-shared"],      # 共享库链接命令
        'archiver'     : ["ar", "-cr"],            # 静态库归档命令
        'ranlib'       : ["ranlib"]                # Ranlib 命令
        }

    # 模块目录开关，暂未实现，待修复
    module_dir_switch = None  #XXX Fix me

    # 模块包含开关，暂未实现，待修复
    module_include_switch = None #XXX Fix me

    # 获取优化标志的方法，返回包含 '-O' 的列表
    def get_flags_opt(self):
        return ['-O']

    # 获取调试标志的方法，返回包含 '-g', '--chk', '--chkglobal' 的列表
    def get_flags_debug(self):
        return ['-g', '--chk', '--chkglobal']

    # 获取库目录的方法
    def get_library_dirs(self):
        opt = []
        d = os.environ.get('LAHEY')
        if d:
            opt.append(os.path.join(d, 'lib'))
        return opt

    # 获取库列表的方法
    def get_libraries(self):
        opt = []
        opt.extend(['fj9f6', 'fj9i6', 'fj9ipp', 'fj9e6'])
        return opt

# 如果当前模块是主程序
if __name__ == '__main__':

    # 从 distutils 中导入 log
    from distutils import log

    # 设置日志详细级别为 2
    log.set_verbosity(2)

    # 从 numpy.distutils 中导入 customized_fcompiler
    from numpy.distutils import customized_fcompiler

    # 打印使用 compiler='lahey' 参数调用 customized_fcompiler 的版本信息
    print(customized_fcompiler(compiler='lahey').get_version())
```