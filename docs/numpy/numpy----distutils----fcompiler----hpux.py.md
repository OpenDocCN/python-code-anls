# `.\numpy\numpy\distutils\fcompiler\hpux.py`

```
# 从 numpy.distutils.fcompiler 模块导入 FCompiler 类
from numpy.distutils.fcompiler import FCompiler

# 定义一个列表，包含单个字符串 'HPUXFCompiler'
compilers = ['HPUXFCompiler']

# 定义 HPUXFCompiler 类，继承自 FCompiler 类
class HPUXFCompiler(FCompiler):

    # 定义类变量 compiler_type，表示编译器类型为 'hpux'
    compiler_type = 'hpux'
    
    # 定义类变量 description，描述为 'HP Fortran 90 Compiler'
    description = 'HP Fortran 90 Compiler'
    
    # 定义类变量 version_pattern，使用正则表达式匹配 HP F90 编译器版本
    version_pattern =  r'HP F90 (?P<version>[^\s*,]*)'

    # 定义 executables 字典，包含各种命令及其参数
    executables = {
        'version_cmd'  : ["f90", "+version"],  # 版本查询命令
        'compiler_f77' : ["f90"],              # Fortran 77 编译器命令
        'compiler_fix' : ["f90"],              # 固定格式 Fortran 编译器命令
        'compiler_f90' : ["f90"],              # Fortran 90 编译器命令
        'linker_so'    : ["ld", "-b"],         # 动态链接器命令
        'archiver'     : ["ar", "-cr"],        # 静态库打包命令
        'ranlib'       : ["ranlib"]            # ranlib 命令
        }

    # 定义 module_dir_switch 和 module_include_switch 变量，暂未实现（XXX: fix me）
    module_dir_switch = None #XXX: fix me
    module_include_switch = None #XXX: fix me
    
    # 定义 pic_flags 列表，包含 '+Z' 标志
    pic_flags = ['+Z']
    
    # 定义 get_flags 方法，返回 pic_flags 列表和额外的编译标志 ['+ppu', '+DD64']
    def get_flags(self):
        return self.pic_flags + ['+ppu', '+DD64']
    
    # 定义 get_flags_opt 方法，返回优化编译标志 ['-O3']
    def get_flags_opt(self):
        return ['-O3']
    
    # 定义 get_libraries 方法，返回需要链接的库列表，包含 'm' 库
    def get_libraries(self):
        return ['m']
    
    # 定义 get_library_dirs 方法，返回库文件搜索路径列表 ['/usr/lib/hpux64']
    def get_library_dirs(self):
        opt = ['/usr/lib/hpux64']
        return opt
    
    # 定义 get_version 方法，获取编译器版本信息，处理状态码为 256 的情况
    def get_version(self, force=0, ok_status=[256, 0, 1]):
        # XXX status==256 可能表示 'unrecognized option' 或 'no input file' 的错误，version_cmd 需要进一步工作。
        return FCompiler.get_version(self, force, ok_status)

# 如果脚本被直接执行
if __name__ == '__main__':
    # 从 distutils 模块中导入 log
    from distutils import log
    
    # 设置日志输出详细程度为 10
    log.set_verbosity(10)
    
    # 从 numpy.distutils 模块中导入 customized_fcompiler 函数
    from numpy.distutils import customized_fcompiler
    
    # 输出使用 'hpux' 编译器的版本信息
    print(customized_fcompiler(compiler='hpux').get_version())
```