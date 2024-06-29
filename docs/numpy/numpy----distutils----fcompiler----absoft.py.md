# `.\numpy\numpy\distutils\fcompiler\absoft.py`

```
# 导入操作系统模块
import os

# 从 numpy.distutils.cpuinfo 模块导入 cpu 函数
# 从 numpy.distutils.fcompiler 模块导入 FCompiler 类和 dummy_fortran_file 函数
# 从 numpy.distutils.misc_util 模块导入 cyg2win32 函数
from numpy.distutils.cpuinfo import cpu
from numpy.distutils.fcompiler import FCompiler, dummy_fortran_file
from numpy.distutils.misc_util import cyg2win32

# 定义可用的编译器列表，其中只包含 AbsoftFCompiler
compilers = ['AbsoftFCompiler']

# 定义 AbsoftFCompiler 类，继承自 FCompiler
class AbsoftFCompiler(FCompiler):

    # 定义编译器类型为 'absoft'
    compiler_type = 'absoft'
    # 定义编译器描述
    description = 'Absoft Corp Fortran Compiler'
    # 定义版本模式匹配规则
    version_pattern = r'(f90:.*?(Absoft Pro FORTRAN Version|FORTRAN 77 Compiler|Absoft Fortran Compiler Version|Copyright Absoft Corporation.*?Version))'+\
                       r' (?P<version>[^\s*,]*)(.*?Absoft Corp|)'

    # 定义可执行文件字典
    executables = {
        'version_cmd'  : None,          # set by update_executables
        'compiler_f77' : ["f77"],
        'compiler_fix' : ["f90"],
        'compiler_f90' : ["f90"],
        'linker_so'    : ["<F90>"],
        'archiver'     : ["ar", "-cr"],
        'ranlib'       : ["ranlib"]
        }

    # 如果操作系统为 Windows
    if os.name=='nt':
        # 定义库开关为 '/out:'
        library_switch = '/out:'      #No space after /out:!

    # 定义模块目录开关为空
    module_dir_switch = None
    # 定义模块包含开关为 '-p'
    module_include_switch = '-p'
    
    # 定义更新可执行程序的方法
    def update_executables(self):
        # 调用 cyg2win32 函数，并传入虚拟的 Fortran 文件名，更新版本命令为 ['<F90>', '-V', '-c', f+'.f', '-o', f+'.o']

    # 定义获取链接器选项的方法
    def get_flags_linker_so(self):
        # 如果操作系统为 Windows
        if os.name=='nt':
            # 定义选项为 ['/dll']
        # 如果版本大于等于 9.0
        elif self.get_version() >= '9.0':
            # 定义选项为 ['-shared']
        else:
            # 定义选项为 ["-K", "shared"]
        # 返回选项

    # 定义库目录选项的方法
    def library_dir_option(self, dir):
        # 如果操作系统为 Windows
        if os.name=='nt':
            # 返回 ['-link', '/PATH:%s' % (dir)]
        # 返回 "-L" + dir

    # 定义库选项的方法
    def library_option(self, lib):
        # 如果操作系统为 Windows
        if os.name=='nt':
            # 返回 '%s.lib' % (lib)
        # 返回 "-l" + lib
    def get_library_dirs(self):
        # 调用父类的方法获取库目录列表
        opt = FCompiler.get_library_dirs(self)
        # 获取环境变量ABSOFT的值
        d = os.environ.get('ABSOFT')
        # 如果存在ABSOFT环境变量
        if d:
            # 获取版本号并判断是否大于等于10.0
            if self.get_version() >= '10.0':
                # 使用共享库，静态库未使用-fPIC编译
                prefix = 'sh'
            else:
                prefix = ''
            # 判断CPU是否为64位
            if cpu.is_64bit():
                suffix = '64'
            else:
                suffix = ''
            # 将目录添加到库目录列表
            opt.append(os.path.join(d, '%slib%s' % (prefix, suffix)))
        return opt

    def get_libraries(self):
        # 调用父类的方法获取库列表
        opt = FCompiler.get_libraries(self)
        # 获取版本号并根据版本号添加不同的库
        if self.get_version() >= '11.0':
            opt.extend(['af90math', 'afio', 'af77math', 'amisc'])
        elif self.get_version() >= '10.0':
            opt.extend(['af90math', 'afio', 'af77math', 'U77'])
        elif self.get_version() >= '8.0':
            opt.extend(['f90math', 'fio', 'f77math', 'U77'])
        else:
            opt.extend(['fio', 'f90math', 'fmath', 'U77'])
        # 如果操作系统为Windows，添加COMDLG32库
        if os.name =='nt':
            opt.append('COMDLG32')
        return opt

    def get_flags(self):
        # 调用父类的方法获取编译选项列表
        opt = FCompiler.get_flags(self)
        # 如果操作系统不是Windows，添加'-s'选项，并根据版本号添加不同的选项
        if os.name != 'nt':
            opt.extend(['-s'])
            if self.get_version():
                if self.get_version()>='8.2':
                    opt.append('-fpic')
        return opt

    def get_flags_f77(self):
        # 调用父类的方法获取Fortran 77编译选项列表
        opt = FCompiler.get_flags_f77(self)
        # 添加特定的编译选项
        opt.extend(['-N22', '-N90', '-N110'])
        v = self.get_version()
        # 如果操作系统为Windows，并且版本号大于等于8.0，添加特定的编译选项
        if os.name == 'nt':
            if v and v>='8.0':
                opt.extend(['-f', '-N15'])
        else:
            opt.append('-f')
            # 根据版本号添加不同的编译选项
            if v:
                if v<='4.6':
                    opt.append('-B108')
                else:
                    opt.append('-N15')
        return opt

    def get_flags_f90(self):
        # 调用父类的方法获取Fortran 90编译选项列表
        opt = FCompiler.get_flags_f90(self)
        # 添加特定的编译选项
        opt.extend(["-YCFRL=1", "-YCOM_NAMES=LCS", "-YCOM_PFX", "-YEXT_PFX",
                    "-YCOM_SFX=_", "-YEXT_SFX=_", "-YEXT_NAMES=LCS"])
        # 如果有版本号，根据版本号添加不同的编译选项
        if self.get_version():
            if self.get_version()>'4.6':
                opt.extend(["-YDEALLOC=ALL"])
        return opt

    def get_flags_fix(self):
        # 调用父类的方法获取固定格式编译选项列表
        opt = FCompiler.get_flags_fix(self)
        # 添加特定的编译选项
        opt.extend(["-YCFRL=1", "-YCOM_NAMES=LCS", "-YCOM_PFX", "-YEXT_PFX",
                    "-YCOM_SFX=_", "-YEXT_SFX=_", "-YEXT_NAMES=LCS"])
        opt.extend(["-f", "fixed"])
        return opt

    def get_flags_opt(self):
        # 返回优化编译选项列表
        opt = ['-O']
        return opt
# 如果当前模块作为脚本直接执行
if __name__ == '__main__':
    # 从 distutils 模块导入 log
    from distutils import log
    # 设置日志记录级别为 2
    log.set_verbosity(2)
    # 从 numpy.distutils 模块导入 customized_fcompiler
    from numpy.distutils import customized_fcompiler
    # 打印使用 absoft 编译器的版本信息
    print(customized_fcompiler(compiler='absoft').get_version())
```