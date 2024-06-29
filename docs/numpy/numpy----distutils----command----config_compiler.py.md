# `.\numpy\numpy\distutils\command\config_compiler.py`

```
# 从distutils.core模块中导入Command类
from distutils.core import Command
# 从numpy.distutils模块中导入log函数
from numpy.distutils import log

#XXX: Linker flags

# 展示Fortran编译器
def show_fortran_compilers(_cache=None):
    # 使用缓存来防止无限递归
    if _cache:
        return
    # 如果缓存为None，则初始化为空列表
    elif _cache is None:
        _cache = []
    # 将1添加到缓存列表中
    _cache.append(1)
    # 从numpy.distutils.fcompiler模块中导入show_fcompilers函数
    from numpy.distutils.fcompiler import show_fcompilers
    # 从distutils.core模块中导入_setup_distribution属性
    import distutils.core
    dist = distutils.core._setup_distribution
    # 展示Fortran编译器
    show_fcompilers(dist)

# 继承自Command类的config_fc类
class config_fc(Command):
    """ Distutils command to hold user specified options
    to Fortran compilers.

    config_fc command is used by the FCompiler.customize() method.
    """

    # 描述信息
    description = "specify Fortran 77/Fortran 90 compiler information"

    # 用户选项列表
    user_options = [
        ('fcompiler=', None, "specify Fortran compiler type"),
        ('f77exec=', None, "specify F77 compiler command"),
        ('f90exec=', None, "specify F90 compiler command"),
        ('f77flags=', None, "specify F77 compiler flags"),
        ('f90flags=', None, "specify F90 compiler flags"),
        ('opt=', None, "specify optimization flags"),
        ('arch=', None, "specify architecture specific optimization flags"),
        ('debug', 'g', "compile with debugging information"),
        ('noopt', None, "compile without optimization"),
        ('noarch', None, "compile without arch-dependent optimization"),
        ]

    # 帮助选项列表
    help_options = [
        ('help-fcompiler', None, "list available Fortran compilers",
         show_fortran_compilers),
        ]

    # 布尔选项列表
    boolean_options = ['debug', 'noopt', 'noarch']

    # 初始化选项
    def initialize_options(self):
        self.fcompiler = None
        self.f77exec = None
        self.f90exec = None
        self.f77flags = None
        self.f90flags = None
        self.opt = None
        self.arch = None
        self.debug = None
        self.noopt = None
        self.noarch = None

    # 最终选项
    def finalize_options(self):
        log.info('unifing config_fc, config, build_clib, build_ext, build commands --fcompiler options')
        build_clib = self.get_finalized_command('build_clib')
        build_ext = self.get_finalized_command('build_ext')
        config = self.get_finalized_command('config')
        build = self.get_finalized_command('build')
        cmd_list = [self, config, build_clib, build_ext, build]
        for a in ['fcompiler']:
            l = []
            for c in cmd_list:
                v = getattr(c, a)
                if v is not None:
                    if not isinstance(v, str): v = v.compiler_type
                    if v not in l: l.append(v)
            if not l: v1 = None
            else: v1 = l[0]
            if len(l)>1:
                log.warn('  commands have different --%s options: %s'\
                         ', using first in list as default' % (a, l))
            if v1:
                for c in cmd_list:
                    if getattr(c, a) is None: setattr(c, a, v1)

    # 运行方法
    def run(self):
        # 什么也不做
        return

# 继承自Command类的config_cc类
class config_cc(Command):
    # 用于保存用户指定的 C/C++ 编译器选项的 Distutils 命令
    description = "specify C/C++ compiler information"
    
    # 用户选项列表
    user_options = [
        ('compiler=', None, "specify C/C++ compiler type"),
        ]
    
    # 初始化选项
    def initialize_options(self):
        self.compiler = None
    
    # 最终选项
    def finalize_options(self):
        log.info('unifing config_cc, config, build_clib, build_ext, build commands --compiler options')
        build_clib = self.get_finalized_command('build_clib')
        build_ext = self.get_finalized_command('build_ext')
        config = self.get_finalized_command('config')
        build = self.get_finalized_command('build')
        cmd_list = [self, config, build_clib, build_ext, build]
        for a in ['compiler']:
            l = []
            for c in cmd_list:
                v = getattr(c, a)
                if v is not None:
                    if not isinstance(v, str): v = v.compiler_type
                    if v not in l: l.append(v)
            if not l: v1 = None
            else: v1 = l[0]
            if len(l)>1:
                log.warn('  commands have different --%s options: %s'\
                         ', using first in list as default' % (a, l))
            if v1:
                for c in cmd_list:
                    if getattr(c, a) is None: setattr(c, a, v1)
        return
    
    # 运行命令
    def run(self):
        # 什么也不做
        return
```