# `.\numpy\numpy\distutils\fcompiler\vast.py`

```py
import os  # 导入标准库 os

from numpy.distutils.fcompiler.gnu import GnuFCompiler  # 从 numpy 的 distutils 子模块中导入 GnuFCompiler 类

compilers = ['VastFCompiler']  # 定义一个列表 compilers，包含字符串 'VastFCompiler'

class VastFCompiler(GnuFCompiler):
    compiler_type = 'vast'  # 设置编译器类型为 'vast'
    compiler_aliases = ()  # 定义编译器别名为空元组
    description = 'Pacific-Sierra Research Fortran 90 Compiler'  # 设置编译器描述信息
    version_pattern = (r'\s*Pacific-Sierra Research vf90 '
                       r'(Personal|Professional)\s+(?P<version>[^\s]*)')  # 定义版本号的正则表达式模式

    # VAST f90 不支持 -o 与 -c 一起使用。因此，对象文件先创建在当前目录，然后移动到构建目录
    object_switch = ' && function _mvfile { mv -v `basename $1` $1 ; } && _mvfile '  # 定义用于移动文件的 shell 命令

    executables = {  # 定义编译器和链接器的可执行命令字典
        'version_cmd'  : ["vf90", "-v"],  # 获取版本信息的命令
        'compiler_f77' : ["g77"],  # Fortran 77 编译器命令
        'compiler_fix' : ["f90", "-Wv,-ya"],  # 修复 Fortran 编译器命令
        'compiler_f90' : ["f90"],  # Fortran 90 编译器命令
        'linker_so'    : ["<F90>"],  # 链接器命令
        'archiver'     : ["ar", "-cr"],  # 归档工具命令
        'ranlib'       : ["ranlib"]  # ranlib 命令
        }

    module_dir_switch = None  # 设置模块目录的开关为 None
    module_include_switch = None  # 设置模块包含的开关为 None

    def find_executables(self):
        pass  # 空方法，用于查找可执行文件

    def get_version_cmd(self):
        f90 = self.compiler_f90[0]  # 获取 Fortran 90 编译器路径
        d, b = os.path.split(f90)  # 分离路径和文件名
        vf90 = os.path.join(d, 'v'+b)  # 构建带有 'v' 前缀的路径
        return vf90  # 返回 vf90 的路径

    def get_flags_arch(self):
        vast_version = self.get_version()  # 获取 VAST 编译器的版本
        gnu = GnuFCompiler()  # 创建一个 GnuFCompiler 的实例
        gnu.customize(None)  # 自定义 GnuFCompiler 实例
        self.version = gnu.get_version()  # 设置当前实例的版本为 GnuFCompiler 的版本
        opt = GnuFCompiler.get_flags_arch(self)  # 调用 GnuFCompiler 的 get_flags_arch 方法获取编译选项
        self.version = vast_version  # 恢复原始的 VAST 编译器版本
        return opt  # 返回编译选项

if __name__ == '__main__':
    from distutils import log  # 从 distutils 模块导入 log 对象
    log.set_verbosity(2)  # 设置日志输出详细级别为 2
    from numpy.distutils import customized_fcompiler  # 从 numpy 的 distutils 子模块导入 customized_fcompiler 函数
    print(customized_fcompiler(compiler='vast').get_version())  # 打印使用 'vast' 编译器的版本信息
```