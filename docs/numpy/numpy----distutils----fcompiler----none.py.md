# `.\numpy\numpy\distutils\fcompiler\none.py`

```py
# 导入 numpy.distutils.fcompiler 模块中的 FCompiler 类
from numpy.distutils.fcompiler import FCompiler
# 导入 numpy.distutils 中的 customized_fcompiler 函数
from numpy.distutils import customized_fcompiler

# 定义一个列表，包含字符串 'NoneFCompiler'
compilers = ['NoneFCompiler']

# 定义一个名为 NoneFCompiler 的类，继承自 FCompiler 类
class NoneFCompiler(FCompiler):

    # 设置编译器类型为 'none'
    compiler_type = 'none'
    # 设置编译器描述为 'Fake Fortran compiler'
    description = 'Fake Fortran compiler'

    # 定义一个字典，包含多个执行文件路径，均设为 None
    executables = {'compiler_f77': None,
                   'compiler_f90': None,
                   'compiler_fix': None,
                   'linker_so': None,
                   'linker_exe': None,
                   'archiver': None,
                   'ranlib': None,
                   'version_cmd': None,
                   }

    # 定义一个方法，用于查找执行文件，但实际上没有实现其功能
    def find_executables(self):
        pass

# 如果作为脚本运行
if __name__ == '__main__':
    # 从 distutils 模块中导入 log 对象
    from distutils import log
    # 设置日志详细程度为 2
    log.set_verbosity(2)
    # 输出 'none' 编译器的版本信息
    print(customized_fcompiler(compiler='none').get_version())
```