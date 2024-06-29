# `.\numpy\numpy\distutils\intelccompiler.py`

```py
# 导入平台模块，用于获取操作系统信息
import platform

# 导入 UnixCCompiler 类，用于处理 Unix-like 系统上的编译器相关操作
from distutils.unixccompiler import UnixCCompiler

# 导入 find_executable 函数，用于查找可执行文件路径
from numpy.distutils.exec_command import find_executable

# 导入 simple_version_match 函数，用于简单的版本匹配功能
from numpy.distutils.ccompiler import simple_version_match

# 如果当前操作系统是 Windows，导入 MSVCCompiler 类
if platform.system() == 'Windows':
    from numpy.distutils.msvc9compiler import MSVCCompiler


class IntelCCompiler(UnixCCompiler):
    """A modified Intel compiler compatible with a GCC-built Python."""
    # 编译器类型为 Intel
    compiler_type = 'intel'
    # 编译器可执行文件为 icc
    cc_exe = 'icc'
    # 编译参数为 fPIC
    cc_args = 'fPIC'

    def __init__(self, verbose=0, dry_run=0, force=0):
        # 调用父类 UnixCCompiler 的初始化方法
        UnixCCompiler.__init__(self, verbose, dry_run, force)

        # 获取编译器版本信息
        v = self.get_version()
        # 根据版本选择 OpenMP 选项
        mpopt = 'openmp' if v and v < '15' else 'qopenmp'
        # 设置编译器可执行文件及参数
        self.cc_exe = ('icc -fPIC -fp-model strict -O3 '
                       '-fomit-frame-pointer -{}').format(mpopt)
        compiler = self.cc_exe

        # 根据操作系统选择共享库标志
        if platform.system() == 'Darwin':
            shared_flag = '-Wl,-undefined,dynamic_lookup'
        else:
            shared_flag = '-shared'
        # 设置编译器及链接器的可执行文件
        self.set_executables(compiler=compiler,
                             compiler_so=compiler,
                             compiler_cxx=compiler,
                             archiver='xiar' + ' cru',
                             linker_exe=compiler + ' -shared-intel',
                             linker_so=compiler + ' ' + shared_flag +
                             ' -shared-intel')


class IntelItaniumCCompiler(IntelCCompiler):
    # 编译器类型为 Itanium Intel
    compiler_type = 'intele'

    # 在 Itanium 平台上，Intel 编译器曾被称为 ecc，现在也可以是 icc，因此搜索这两个可执行文件
    for cc_exe in map(find_executable, ['icc', 'ecc']):
        if cc_exe:
            break


class IntelEM64TCCompiler(UnixCCompiler):
    """
    A modified Intel x86_64 compiler compatible with a 64bit GCC-built Python.
    """
    # 编译器类型为 Intel x86_64
    compiler_type = 'intelem'
    # 编译器可执行文件为 icc -m64
    cc_exe = 'icc -m64'
    # 编译参数为 -fPIC
    cc_args = '-fPIC'

    def __init__(self, verbose=0, dry_run=0, force=0):
        # 调用父类 UnixCCompiler 的初始化方法
        UnixCCompiler.__init__(self, verbose, dry_run, force)

        # 获取编译器版本信息
        v = self.get_version()
        # 根据版本选择 OpenMP 选项
        mpopt = 'openmp' if v and v < '15' else 'qopenmp'
        # 设置编译器可执行文件及参数
        self.cc_exe = ('icc -std=c99 -m64 -fPIC -fp-model strict -O3 '
                       '-fomit-frame-pointer -{}').format(mpopt)
        compiler = self.cc_exe

        # 根据操作系统选择共享库标志
        if platform.system() == 'Darwin':
            shared_flag = '-Wl,-undefined,dynamic_lookup'
        else:
            shared_flag = '-shared'
        # 设置编译器及链接器的可执行文件
        self.set_executables(compiler=compiler,
                             compiler_so=compiler,
                             compiler_cxx=compiler,
                             archiver='xiar' + ' cru',
                             linker_exe=compiler + ' -shared-intel',
                             linker_so=compiler + ' ' + shared_flag +
                             ' -shared-intel')

# 如果当前操作系统是 Windows，则执行以下代码段
if platform.system() == 'Windows':
    # 继承自 MSVCCompiler 类，代表一个修改过的 Intel C 编译器，与 MSVC 构建的 Python 兼容
    class IntelCCompilerW(MSVCCompiler):
        """
        A modified Intel compiler compatible with an MSVC-built Python.
        """
        
        # 编译器类型标识为 'intelw'
        compiler_type = 'intelw'
        
        # C++ 编译器为 'icl'
        compiler_cxx = 'icl'

        # 构造函数，初始化对象
        def __init__(self, verbose=0, dry_run=0, force=0):
            # 调用父类 MSVCCompiler 的构造函数进行初始化
            MSVCCompiler.__init__(self, verbose, dry_run, force)
            
            # 使用正则表达式匹配 Intel 编译器的版本信息
            version_match = simple_version_match(start=r'Intel\(R\).*?32,')
            self.__version = version_match

        # 初始化函数，用于设置编译器相关的路径和选项
        def initialize(self, plat_name=None):
            # 调用父类 MSVCCompiler 的初始化函数
            MSVCCompiler.initialize(self, plat_name)
            
            # 设置 C 编译器的可执行文件路径
            self.cc = self.find_exe('icl.exe')
            
            # 设置库文件的可执行文件路径
            self.lib = self.find_exe('xilib')
            
            # 设置链接器的可执行文件路径
            self.linker = self.find_exe('xilink')
            
            # 设置编译选项，优化等级为 O3，使用标准为 c99，禁用 logo 和警告
            self.compile_options = ['/nologo', '/O3', '/MD', '/W3',
                                    '/Qstd=c99']
            
            # 调试模式下的编译选项，包括禁用 logo、启用调试符号、启用调试宏等
            self.compile_options_debug = ['/nologo', '/Od', '/MDd', '/W3',
                                          '/Qstd=c99', '/Z7', '/D_DEBUG']

    # 继承自 IntelCCompilerW 类，代表一个修改过的 Intel x86_64 编译器，与 64 位 MSVC 构建的 Python 兼容
    class IntelEM64TCCompilerW(IntelCCompilerW):
        """
        A modified Intel x86_64 compiler compatible with
        a 64bit MSVC-built Python.
        """
        
        # 编译器类型标识为 'intelemw'
        compiler_type = 'intelemw'

        # 构造函数，初始化对象
        def __init__(self, verbose=0, dry_run=0, force=0):
            # 调用父类 IntelCCompilerW 的构造函数进行初始化
            MSVCCompiler.__init__(self, verbose, dry_run, force)
            
            # 使用正则表达式匹配 Intel 编译器的版本信息
            version_match = simple_version_match(start=r'Intel\(R\).*?64,')
            self.__version = version_match
```