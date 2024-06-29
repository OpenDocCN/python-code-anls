# `.\numpy\numpy\distutils\pathccompiler.py`

```py
# 导入 UnixCCompiler 类，用于扩展 PathScaleCCompiler
from distutils.unixccompiler import UnixCCompiler

# 定义 PathScaleCCompiler 类，继承自 UnixCCompiler 类
class PathScaleCCompiler(UnixCCompiler):

    """
    PathScale compiler compatible with an gcc built Python.
    """
    
    # 设置编译器类型为 'pathcc'
    compiler_type = 'pathcc'
    # 设置 C 编译器的可执行文件名为 'pathcc'
    cc_exe = 'pathcc'
    # 设置 C++ 编译器的可执行文件名为 'pathCC'
    cxx_exe = 'pathCC'

    # 初始化方法，接受 verbose、dry_run 和 force 三个参数
    def __init__ (self, verbose=0, dry_run=0, force=0):
        # 调用父类 UnixCCompiler 的初始化方法
        UnixCCompiler.__init__ (self, verbose, dry_run, force)
        # 将 cc_exe 和 cxx_exe 分别赋值给 cc_compiler 和 cxx_compiler
        cc_compiler = self.cc_exe
        cxx_compiler = self.cxx_exe
        # 设置编译器的可执行文件名，包括编译器、编译器命令、C++ 编译器、链接器和共享库链接器
        self.set_executables(compiler=cc_compiler,
                             compiler_so=cc_compiler,
                             compiler_cxx=cxx_compiler,
                             linker_exe=cc_compiler,
                             linker_so=cc_compiler + ' -shared')
```