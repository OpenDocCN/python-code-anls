# `.\numpy\numpy\distutils\fcompiler\g95.py`

```py
# 导入FCompiler类
from numpy.distutils.fcompiler import FCompiler

# 定义G95FCompiler类，继承自FCompiler类
compilers = ['G95FCompiler']

class G95FCompiler(FCompiler):
    # 设置编译器类型为'g95'
    compiler_type = 'g95'
    # 描述为'G95 Fortran Compiler'
    description = 'G95 Fortran Compiler'

#    version_pattern = r'G95 \((GCC (?P<gccversion>[\d.]+)|.*?) \(g95!\) (?P<version>.*)\).*'
    # 定义版本信息的正则表达式模式
    # $ g95 --version
    # G95 (GCC 4.0.3 (g95!) May 22 2006)
    version_pattern = r'G95 \((GCC (?P<gccversion>[\d.]+)|.*?) \(g95 (?P<version>.*)!\) (?P<date>.*)\).*'
    # $ g95 --version
    # G95 (GCC 4.0.3 (g95 0.90!) Aug 22 2006)

    executables = {
        'version_cmd'  : ["<F90>", "--version"],
        'compiler_f77' : ["g95", "-ffixed-form"],
        'compiler_fix' : ["g95", "-ffixed-form"],
        'compiler_f90' : ["g95"],
        'linker_so'    : ["<F90>", "-shared"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"]
        }
    # 设置标志为'-fpic'
    pic_flags = ['-fpic']
    # 设置模块目录开关为'-fmod='
    module_dir_switch = '-fmod='
    # 设置模块包含开关为'-I'
    module_include_switch = '-I'

    # 定义获取标志的方法
    def get_flags(self):
        return ['-fno-second-underscore']
    # 定义获取优化标志的方法
    def get_flags_opt(self):
        return ['-O']
    # 定义获取调试标志的方法
    def get_flags_debug(self):
        return ['-g']

# 如果是直接运行该脚本
if __name__ == '__main__':
    # 导入log和customized_fcompiler函数
    from distutils import log
    from numpy.distutils import customized_fcompiler
    # 设置日志输出详细程度为2
    log.set_verbosity(2)
    # 打印自定义Fortran编译器的版本信息
    print(customized_fcompiler('g95').get_version())
```