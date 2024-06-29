# `.\numpy\numpy\distutils\fcompiler\nv.py`

```py
# 导入 FCompiler 类，用于自定义编译器
from numpy.distutils.fcompiler import FCompiler

# 定义一个列表，包含唯一的编译器类名 'NVHPCFCompiler'
compilers = ['NVHPCFCompiler']

# 创建 NVHPCFCompiler 类，继承自 FCompiler 类，用于 NVIDIA HPC SDK Fortran 编译器
class NVHPCFCompiler(FCompiler):
    """ NVIDIA High Performance Computing (HPC) SDK Fortran Compiler
   
    https://developer.nvidia.com/hpc-sdk
   
    自 2020 年 8 月起，NVIDIA HPC SDK 包含了以前被称为 Portland Group 编译器的编译器，
    https://www.pgroup.com/index.htm.
    参见 `numpy.distutils.fcompiler.pg`。
    """

    # 编译器类型设定为 'nv'
    compiler_type = 'nv'
    # 描述为 'NVIDIA HPC SDK'
    description = 'NVIDIA HPC SDK'
    # 版本模式匹配正则表达式，匹配 'nvfortran' 或者 'xxx (aka nvfortran)' 格式的版本信息
    version_pattern = r'\s*(nvfortran|.+ \(aka nvfortran\)) (?P<version>[\d.-]+).*'

    # 执行命令字典，包含各种编译和链接命令的配置
    executables = {
        'version_cmd': ["<F90>", "-V"],        # 获取版本信息的命令
        'compiler_f77': ["nvfortran"],         # Fortran 77 编译器命令
        'compiler_fix': ["nvfortran", "-Mfixed"],  # 使用固定格式的 Fortran 90 编译器命令
        'compiler_f90': ["nvfortran"],         # Fortran 90 编译器命令
        'linker_so': ["<F90>"],                 # 共享库链接器命令
        'archiver': ["ar", "-cr"],              # 静态库打包命令
        'ranlib': ["ranlib"]                    # ranlib 命令
    }
    
    # 位置无关代码编译选项列表
    pic_flags = ['-fpic']

    # 模块目录开关，用于指定模块输出目录
    module_dir_switch = '-module '
    # 模块包含目录开关，用于指定模块的搜索路径
    module_include_switch = '-I'

    # 获取编译器标志的方法
    def get_flags(self):
        # 优化选项，设置了一些编译选项
        opt = ['-Minform=inform', '-Mnosecond_underscore']
        return self.pic_flags + opt

    # 获取优化标志的方法
    def get_flags_opt(self):
        return ['-fast']

    # 获取调试标志的方法
    def get_flags_debug(self):
        return ['-g']

    # 获取链接共享库标志的方法
    def get_flags_linker_so(self):
        return ["-shared", '-fpic']

    # 运行时库目录选项，返回一个设置了目录的选项字符串
    def runtime_library_dir_option(self, dir):
        return '-R%s' % dir

# 如果脚本作为主程序执行，则执行以下代码块
if __name__ == '__main__':
    # 导入 distutils 中的 log 模块
    from distutils import log
    # 设置日志的详细程度为 2
    log.set_verbosity(2)
    # 导入 numpy.distutils 中的 customized_fcompiler 函数
    from numpy.distutils import customized_fcompiler
    # 输出定制化编译器的版本信息
    print(customized_fcompiler(compiler='nv').get_version())
```