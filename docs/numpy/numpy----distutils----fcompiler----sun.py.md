# `.\numpy\numpy\distutils\fcompiler\sun.py`

```py
# 从 numpy.distutils.ccompiler 模块中导入 simple_version_match 函数
from numpy.distutils.ccompiler import simple_version_match
# 从 numpy.distutils.fcompiler 模块中导入 FCompiler 类
from numpy.distutils.fcompiler import FCompiler

# 定义一个列表 compilers 包含字符串 'SunFCompiler'
compilers = ['SunFCompiler']

# 定义一个名为 SunFCompiler 的类，继承自 FCompiler 类
class SunFCompiler(FCompiler):

    # 类变量，指定编译器类型为 'sun'
    compiler_type = 'sun'
    # 描述信息，表示这是 Sun 或 Forte Fortran 95 编译器
    description = 'Sun or Forte Fortran 95 Compiler'
    
    # 版本匹配函数，使用 simple_version_match 函数匹配特定的版本字符串
    version_match = simple_version_match(
                      start=r'f9[05]: (Sun|Forte|WorkShop).*Fortran 95')

    # 可执行文件的配置字典，包含不同的命令及其参数
    executables = {
        'version_cmd'  : ["<F90>", "-V"],  # 获取版本信息的命令
        'compiler_f77' : ["f90"],  # Fortran 77 编译器命令
        'compiler_fix' : ["f90", "-fixed"],  # 指定了 -fixed 标志的 Fortran 90 编译器命令
        'compiler_f90' : ["f90"],  # Fortran 90 编译器命令
        'linker_so'    : ["<F90>", "-Bdynamic", "-G"],  # 动态链接器命令
        'archiver'     : ["ar", "-cr"],  # 归档命令
        'ranlib'       : ["ranlib"]  # ranlib 命令
        }
    
    # 模块目录开关的配置项
    module_dir_switch = '-moddir='
    # 模块包含开关的配置项
    module_include_switch = '-M'
    # PIC（Position Independent Code）标志的配置选项
    pic_flags = ['-xcode=pic32']

    # 获取 Fortran 77 编译器标志的方法
    def get_flags_f77(self):
        ret = ["-ftrap=%none"]  # 返回一个包含 '-ftrap=%none' 的列表
        # 如果版本号大于或等于 '7'，则添加 '-f77' 到返回列表中
        if (self.get_version() or '') >= '7':
            ret.append("-f77")
        else:
            ret.append("-fixed")  # 否则添加 '-fixed' 到返回列表中
        return ret
    
    # 获取优化标志的方法
    def get_opt(self):
        return ['-fast', '-dalign']  # 返回包含优化标志 '-fast' 和 '-dalign' 的列表
    
    # 获取架构标志的方法
    def get_arch(self):
        return ['-xtarget=generic']  # 返回一个包含 '-xtarget=generic' 的列表
    
    # 获取库文件标志的方法
    def get_libraries(self):
        opt = []
        opt.extend(['fsu', 'sunmath', 'mvec'])  # 将 'fsu', 'sunmath', 'mvec' 添加到 opt 列表中
        return opt
    
    # 返回运行时库目录选项的方法
    def runtime_library_dir_option(self, dir):
        return '-R%s' % dir  # 返回形如 '-R<dir>' 的字符串

# 如果当前模块是主程序入口
if __name__ == '__main__':
    # 从 distutils 模块中导入 log 对象
    from distutils import log
    log.set_verbosity(2)  # 设置 log 的详细程度为 2
    # 从 numpy.distutils 模块中导入 customized_fcompiler 函数
    from numpy.distutils import customized_fcompiler
    # 打印使用 'sun' 编译器的版本信息
    print(customized_fcompiler(compiler='sun').get_version())
```