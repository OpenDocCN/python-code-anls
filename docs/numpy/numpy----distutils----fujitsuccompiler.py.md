# `.\numpy\numpy\distutils\fujitsuccompiler.py`

```py
# 导入 UnixCCompiler 类，用于扩展 FujitsuCCompiler 类
from distutils.unixccompiler import UnixCCompiler

class FujitsuCCompiler(UnixCCompiler):
    """
    Fujitsu compiler.
    继承自 UnixCCompiler 类，用于支持 Fujitsu 编译器的编译功能。
    """

    # 设置编译器类型为 'fujitsu'
    compiler_type = 'fujitsu'
    
    # 指定 C 编译器执行文件名为 'fcc'
    cc_exe = 'fcc'
    
    # 指定 C++ 编译器执行文件名为 'FCC'
    cxx_exe = 'FCC'

    # 初始化方法，接收 verbose、dry_run 和 force 三个参数
    def __init__(self, verbose=0, dry_run=0, force=0):
        # 调用父类 UnixCCompiler 的初始化方法
        UnixCCompiler.__init__(self, verbose, dry_run, force)
        
        # 将 C 编译器执行文件名 'fcc' 赋值给 cc_compiler 变量
        cc_compiler = self.cc_exe
        
        # 将 C++ 编译器执行文件名 'FCC' 赋值给 cxx_compiler 变量
        
        # 设置编译器的可执行文件路径及编译选项
        self.set_executables(
            # 设置 C 编译器的可执行文件路径及编译选项，包括优化级别为 3、禁用 clang 扩展、生成位置独立代码
            compiler=cc_compiler +
            ' -O3 -Nclang -fPIC',
            # 设置用于编译单个 C 源文件的编译器选项与路径，与上一行设置相同
            compiler_so=cc_compiler +
            ' -O3 -Nclang -fPIC',
            # 设置 C++ 编译器的可执行文件路径及编译选项，与上一行设置相同
            compiler_cxx=cxx_compiler +
            ' -O3 -Nclang -fPIC',
            # 设置链接器的可执行文件路径及链接选项，链接库包括 f90、f90f、fjsrcinfo、elf，并生成共享库
            linker_exe=cc_compiler +
            ' -lfj90i -lfj90f -lfjsrcinfo -lelf -shared',
            # 设置用于链接共享库的链接器选项与路径，与上一行设置相同
            linker_so=cc_compiler +
            ' -lfj90i -lfj90f -lfjsrcinfo -lelf -shared'
        )
```