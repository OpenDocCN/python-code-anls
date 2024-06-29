# `.\numpy\numpy\distutils\armccompiler.py`

```
from distutils.unixccompiler import UnixCCompiler                              

class ArmCCompiler(UnixCCompiler):
    """
    Arm compiler subclass inheriting from UnixCCompiler.
    """

    # 设置编译器类型为 'arm'
    compiler_type = 'arm'
    # 设置 C 编译器可执行文件为 'armclang'
    cc_exe = 'armclang'
    # 设置 C++ 编译器可执行文件为 'armclang++'
    cxx_exe = 'armclang++'

    def __init__(self, verbose=0, dry_run=0, force=0):
        # 调用父类 UnixCCompiler 的初始化方法
        UnixCCompiler.__init__(self, verbose, dry_run, force)
        # 将 C 编译器可执行文件保存到 cc_compiler 变量中
        cc_compiler = self.cc_exe
        # 将 C++ 编译器可执行文件保存到 cxx_compiler 变量中
        cxx_compiler = self.cxx_exe
        # 设置编译器的各种参数及选项，以及链接器的选项
        self.set_executables(
            compiler=cc_compiler + ' -O3 -fPIC',        # 设置编译器命令及优化级别和位置无关代码
            compiler_so=cc_compiler + ' -O3 -fPIC',     # 设置用于编译源文件的编译器命令及优化级别和位置无关代码
            compiler_cxx=cxx_compiler + ' -O3 -fPIC',   # 设置用于编译 C++ 源文件的编译器命令及优化级别和位置无关代码
            linker_exe=cc_compiler + ' -lamath',        # 设置用于链接可执行文件的链接器命令及链接数学库
            linker_so=cc_compiler + ' -lamath -shared'  # 设置用于链接共享库的链接器命令及链接数学库和共享标志
        )
```