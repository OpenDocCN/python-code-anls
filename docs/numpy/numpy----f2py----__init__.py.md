# `.\numpy\numpy\f2py\__init__.py`

```
#!/usr/bin/env python3
"""Fortran to Python Interface Generator.

Copyright 1999 -- 2011 Pearu Peterson all rights reserved.
Copyright 2011 -- present NumPy Developers.
Permission to use, modify, and distribute this software is given under the terms
of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
"""

__all__ = ['run_main', 'get_include']

import sys                     # 导入 sys 模块，用于系统相关操作
import subprocess              # 导入 subprocess 模块，用于调用外部程序
import os                      # 导入 os 模块，提供操作系统相关功能
import warnings                # 导入 warnings 模块，用于警告处理

from numpy.exceptions import VisibleDeprecationWarning  # 导入 VisibleDeprecationWarning 异常
from . import f2py2e           # 从当前包中导入 f2py2e 模块
from . import diagnose         # 从当前包中导入 diagnose 模块

run_main = f2py2e.run_main    # 将 f2py2e 模块中的 run_main 函数赋值给 run_main 变量
main = f2py2e.main            # 将 f2py2e 模块中的 main 函数赋值给 main 变量


def get_include():
    """
    Return the directory that contains the ``fortranobject.c`` and ``.h`` files.

    .. note::

        This function is not needed when building an extension with
        `numpy.distutils` directly from ``.f`` and/or ``.pyf`` files
        in one go.

    Python extension modules built with f2py-generated code need to use
    ``fortranobject.c`` as a source file, and include the ``fortranobject.h``
    header. This function can be used to obtain the directory containing
    both of these files.

    Returns
    -------
    include_path : str
        Absolute path to the directory containing ``fortranobject.c`` and
        ``fortranobject.h``.

    Notes
    -----
    .. versionadded:: 1.21.1

    Unless the build system you are using has specific support for f2py,
    building a Python extension using a ``.pyf`` signature file is a two-step
    process. For a module ``mymod``:

    * Step 1: run ``python -m numpy.f2py mymod.pyf --quiet``. This
      generates ``mymodmodule.c`` and (if needed)
      ``mymod-f2pywrappers.f`` files next to ``mymod.pyf``.
    * Step 2: build your Python extension module. This requires the
      following source files:

      * ``mymodmodule.c``
      * ``mymod-f2pywrappers.f`` (if it was generated in Step 1)
      * ``fortranobject.c``

    See Also
    --------
    numpy.get_include : function that returns the numpy include directory

    """
    return os.path.join(os.path.dirname(__file__), 'src')  # 返回包含 fortranobject.c 和 .h 文件的目录路径


def __getattr__(attr):
    """
    Handle dynamic attribute access for the module.

    Parameters
    ----------
    attr : str
        The name of the attribute being accessed dynamically.

    Returns
    -------
    object
        The value of the dynamically accessed attribute.

    Raises
    ------
    AttributeError
        If the requested attribute does not exist.

    Notes
    -----
    This function specifically handles access to the 'test' attribute,
    importing PytestTester if needed.

    """
    if attr == "test":
        from numpy._pytesttester import PytestTester
        test = PytestTester(__name__)   # 使用 PytestTester 创建测试对象
        return test
    else:
        raise AttributeError("module {!r} has no attribute "
                              "{!r}".format(__name__, attr))  # 抛出 AttributeError 异常


def __dir__():
    """
    Return the list of attributes available in the module.

    Returns
    -------
    list
        List of attribute names in the module.

    """
    return list(globals().keys() | {"test"})  # 返回当前模块中的全局变量名列表，加上额外的 "test" 属性
```