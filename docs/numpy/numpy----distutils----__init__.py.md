# `.\numpy\numpy\distutils\__init__.py`

```py
"""
An enhanced distutils, providing support for Fortran compilers, for BLAS,
LAPACK and other common libraries for numerical computing, and more.

Public submodules are::

    misc_util
    system_info
    cpu_info
    log
    exec_command

For details, please see the *Packaging* and *NumPy Distutils User Guide*
sections of the NumPy Reference Guide.

For configuring the preference for and location of libraries like BLAS and
LAPACK, and for setting include paths and similar build options, please see
``site.cfg.example`` in the root of the NumPy repository or sdist.

"""

import warnings

# 必须尽快导入本地的 ccompiler 以便使自定义的 CCompiler.spawn 生效。
from . import ccompiler
from . import unixccompiler

# 导入 numpy 包配置
from .npy_pkg_config import *

# 发出警告：numpy.distutils 自 NumPy 1.23.0 起已被弃用，因为 distutils 本身也已被弃用。
# 对于 Python >= 3.12 将移除此模块，建议在这些 Python 版本中使用 setuptools < 60.0。
warnings.warn("\n\n"
    "  `numpy.distutils` is deprecated since NumPy 1.23.0, as a result\n"
    "  of the deprecation of `distutils` itself. It will be removed for\n"
    "  Python >= 3.12. For older Python versions it will remain present.\n"
    "  It is recommended to use `setuptools < 60.0` for those Python versions.\n"
    "  For more details, see:\n"
    "    https://numpy.org/devdocs/reference/distutils_status_migration.html \n\n",
    DeprecationWarning, stacklevel=2
)
# 删除警告模块的引用，清理命名空间
del warnings

# 如果 numpy 安装了，添加 distutils.test()
try:
    # 导入 numpy 的 __config__ 模块
    from . import __config__
    # 通常如果上述导入成功，numpy 已经安装，但中断的就地构建也可能会留下 __config__.py。
    # 在这种情况下，下一个导入可能仍会失败，因此保持在 try 块内。
    from numpy._pytesttester import PytestTester
    # 创建 PytestTester 对象，测试当前模块
    test = PytestTester(__name__)
    # 删除 PytestTester 的引用，清理命名空间
    del PytestTester
except ImportError:
    # 如果导入失败，什么也不做
    pass


def customized_fcompiler(plat=None, compiler=None):
    # 导入 numpy.distutils.fcompiler 中的 new_fcompiler 函数
    from numpy.distutils.fcompiler import new_fcompiler
    # 创建新的 Fortran 编译器对象
    c = new_fcompiler(plat=plat, compiler=compiler)
    # 自定义编译器配置
    c.customize()
    return c

def customized_ccompiler(plat=None, compiler=None, verbose=1):
    # 使用本地的 ccompiler 模块创建新的 C 编译器对象
    c = ccompiler.new_compiler(plat=plat, compiler=compiler, verbose=verbose)
    # 自定义编译器配置，传入空字符串作为参数
    c.customize('')
    return c
```