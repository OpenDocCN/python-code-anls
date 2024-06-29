# `.\numpy\numpy\__init__.py`

```py
"""
NumPy
=====

Provides
  1. An array object of arbitrary homogeneous items
  2. Fast mathematical operations over arrays
  3. Linear Algebra, Fourier Transforms, Random Number Generation

How to use the documentation
----------------------------
Documentation is available in two forms: docstrings provided
with the code, and a loose standing reference guide, available from
`the NumPy homepage <https://numpy.org>`_.

We recommend exploring the docstrings using
`IPython <https://ipython.org>`_, an advanced Python shell with
TAB-completion and introspection capabilities.  See below for further
instructions.

The docstring examples assume that `numpy` has been imported as ``np``::

  >>> import numpy as np

Code snippets are indicated by three greater-than signs::

  >>> x = 42
  >>> x = x + 1

Use the built-in ``help`` function to view a function's docstring::

  >>> help(np.sort)
  ... # doctest: +SKIP

For some objects, ``np.info(obj)`` may provide additional help.  This is
particularly true if you see the line "Help on ufunc object:" at the top
of the help() page.  Ufuncs are implemented in C, not Python, for speed.
The native Python help() does not know how to view their help, but our
np.info() function does.

Available subpackages
---------------------
lib
    Basic functions used by several sub-packages.
random
    Core Random Tools
linalg
    Core Linear Algebra Tools
fft
    Core FFT routines
polynomial
    Polynomial tools
testing
    NumPy testing tools
distutils
    Enhancements to distutils with support for
    Fortran compilers support and more (for Python <= 3.11)

Utilities
---------
test
    Run numpy unittests
show_config
    Show numpy build configuration
__version__
    NumPy version string

Viewing documentation using IPython
-----------------------------------

Start IPython and import `numpy` usually under the alias ``np``: `import
numpy as np`.  Then, directly past or use the ``%cpaste`` magic to paste
examples into the shell.  To see which functions are available in `numpy`,
type ``np.<TAB>`` (where ``<TAB>`` refers to the TAB key), or use
``np.*cos*?<ENTER>`` (where ``<ENTER>`` refers to the ENTER key) to narrow
down the list.  To view the docstring for a function, use
``np.cos?<ENTER>`` (to view the docstring) and ``np.cos??<ENTER>`` (to view
the source code).

Copies vs. in-place operation
-----------------------------
Most of the functions in `numpy` return a copy of the array argument
(e.g., `np.sort`).  In-place versions of these functions are often
available as array methods, i.e. ``x = np.array([1,2,3]); x.sort()``.
Exceptions to this rule are documented.

"""
import os  # 导入操作系统相关的功能模块
import sys  # 导入系统相关的功能模块
import warnings  # 导入警告处理相关的功能模块

from ._globals import _NoValue, _CopyMode  # 从模块中导入指定的全局变量和复制模式
from ._expired_attrs_2_0 import __expired_attributes__  # 从模块中导入过期属性（版本2.0）

# 如果存储了带有 git 哈希的版本，则使用该版本
from . import version  # 从当前目录中导入版本模块
from .version import __version__  # 从版本模块中导入版本号

# We first need to detect if we're being called as part of the numpy setup
# 检查是否定义了 __NUMPY_SETUP__ 变量，如果未定义则置为 False
try:
    __NUMPY_SETUP__
except NameError:
    __NUMPY_SETUP__ = False

# 如果 __NUMPY_SETUP__ 为 True，则向标准错误流输出一条消息
if __NUMPY_SETUP__:
    sys.stderr.write('Running from numpy source directory.\n')
else:
    # 允许分发商在导入 numpy._core 之前运行自定义初始化代码
    from . import _distributor_init

    try:
        # 尝试导入 numpy.__config__.show，并命名为 show_config
        from numpy.__config__ import show as show_config
    except ImportError as e:
        # 如果导入失败，抛出 ImportError 异常，并显示错误消息
        msg = """Error importing numpy: you should not try to import numpy from
        its source directory; please exit the numpy source tree, and relaunch
        your python interpreter from there."""
        raise ImportError(msg) from e

    # 导入 numpy._core 模块
    from . import _core

    # 注意：以下别名是否应移除仍在讨论中
    # 为一些特定的数据类型创建全局变量别名，如果 _core 模块中找不到对应的属性，则跳过
    for ta in ["float96", "float128", "complex192", "complex256"]:
        try:
            globals()[ta] = getattr(_core, ta)
        except AttributeError:
            pass
    del ta  # 删除循环结束后的 ta 变量

    # 导入 numpy.lib 模块及其别名
    from . import lib
    from .lib import scimath as emath
    # 导入 numpy.lib._histograms_impl 模块中的特定函数
    from .lib._histograms_impl import (
        histogram, histogram_bin_edges, histogramdd
    )
    # 导入 numpy.lib._nanfunctions_impl 模块中的特定函数
    from .lib._nanfunctions_impl import (
        nanargmax, nanargmin, nancumprod, nancumsum, nanmax, nanmean, 
        nanmedian, nanmin, nanpercentile, nanprod, nanquantile, nanstd,
        nansum, nanvar
    )
    # 导入 numpy.lib._function_base_impl 模块中的特定函数
    from .lib._function_base_impl import (
        select, piecewise, trim_zeros, copy, iterable, percentile, diff, 
        gradient, angle, unwrap, sort_complex, flip, rot90, extract, place,
        vectorize, asarray_chkfinite, average, bincount, digitize, cov,
        corrcoef, median, sinc, hamming, hanning, bartlett, blackman,
        kaiser, trapezoid, trapz, i0, meshgrid, delete, insert, append,
        interp, quantile
    )
    # 导入 numpy.lib._twodim_base_impl 模块中的特定函数
    from .lib._twodim_base_impl import (
        diag, diagflat, eye, fliplr, flipud, tri, triu, tril, vander, 
        histogram2d, mask_indices, tril_indices, tril_indices_from, 
        triu_indices, triu_indices_from
    )
    # 导入 numpy.lib._shape_base_impl 模块中的特定函数
    from .lib._shape_base_impl import (
        apply_over_axes, apply_along_axis, array_split, column_stack, dsplit,
        dstack, expand_dims, hsplit, kron, put_along_axis, row_stack, split,
        take_along_axis, tile, vsplit
    )
    # 导入 numpy.lib._type_check_impl 模块中的特定函数
    from .lib._type_check_impl import (
        iscomplexobj, isrealobj, imag, iscomplex, isreal, nan_to_num, real, 
        real_if_close, typename, mintypecode, common_type
    )
    # 导入 numpy.lib._arraysetops_impl 模块中的特定函数
    from .lib._arraysetops_impl import (
        ediff1d, in1d, intersect1d, isin, setdiff1d, setxor1d, union1d,
        unique, unique_all, unique_counts, unique_inverse, unique_values
    )
    # 导入 numpy.lib._ufunclike_impl 模块中的特定函数
    from .lib._ufunclike_impl import fix, isneginf, isposinf
    # 导入 numpy.lib._arraypad_impl 模块中的特定函数
    from .lib._arraypad_impl import pad
    # 导入 numpy.lib._utils_impl 模块中的特定函数
    from .lib._utils_impl import (
        show_runtime, get_include, info
    )
    # 导入 numpy.lib._stride_tricks_impl 模块中的特定函数
    from .lib._stride_tricks_impl import (
        broadcast_arrays, broadcast_shapes, broadcast_to
    )
    # 从内部库中导入多项式相关的函数和类
    from .lib._polynomial_impl import (
        poly, polyint, polyder, polyadd, polysub, polymul, polydiv, polyval,
        polyfit, poly1d, roots
    )
    
    # 从内部库中导入数组输入输出相关的函数
    from .lib._npyio_impl import (
        savetxt, loadtxt, genfromtxt, load, save, savez, packbits,
        savez_compressed, unpackbits, fromregex
    )
    
    # 从内部库中导入索引技巧相关的函数和类
    from .lib._index_tricks_impl import (
        diag_indices_from, diag_indices, fill_diagonal, ndindex, ndenumerate,
        ix_, c_, r_, s_, ogrid, mgrid, unravel_index, ravel_multi_index, 
        index_exp
    )
    
    # 从当前目录中导入 matrixlib 模块，并使用别名 _mat
    from . import matrixlib as _mat
    
    # 从 matrixlib 模块中导入特定的函数和类
    from .matrixlib import (
        asmatrix, bmat, matrix
    )
    
    # 定义一个集合，包含了 NumPy 的公共子模块名称，这些子模块可以通过 __getattr__ 懒加载访问
    # 注意，distutils（已弃用）和 array_api（实验性标签）不在此列表中，因为 `from numpy import *` 
    # 不能引发任何警告，这样做太过分了。
    __numpy_submodules__ = {
        "linalg", "fft", "dtypes", "random", "polynomial", "ma", 
        "exceptions", "lib", "ctypeslib", "testing", "typing",
        "f2py", "test", "rec", "char", "core", "strings",
    }
    
    # 构建用于以前属性的警告消息
    _msg = (
        "module 'numpy' has no attribute '{n}'.\n"
        "`np.{n}` was a deprecated alias for the builtin `{n}`. "
        "To avoid this error in existing code, use `{n}` by itself. "
        "Doing this will not modify any behavior and is safe. {extended_msg}\n"
        "The aliases was originally deprecated in NumPy 1.20; for more "
        "details and guidance see the original release note at:\n"
        "    https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations")
    
    _specific_msg = (
        "If you specifically wanted the numpy scalar type, use `np.{}` here.")
    
    _int_extended_msg = (
        "When replacing `np.{}`, you may wish to use e.g. `np.int64` "
        "or `np.int32` to specify the precision. If you wish to review "
        "your current use, check the release note link for "
        "additional information.")
    
    # 定义一个列表，包含有关特定类型信息的元组，每个元组包括类型名称和扩展消息
    _type_info = [
        ("object", ""),  # NumPy 标量仅存在于名称上。
        ("float", _specific_msg.format("float64")),
        ("complex", _specific_msg.format("complex128")),
        ("str", _specific_msg.format("str_")),
        ("int", _int_extended_msg.format("int"))
    ]
    
    # 创建一个字典 __former_attrs__，包含了针对每种类型的警告消息，使用列表推导式生成
    __former_attrs__ = {
         n: _msg.format(n=n, extended_msg=extended_msg)
         for n, extended_msg in _type_info
    }
    
    # 定义一个集合 __future_scalars__，包含了将来可能定义的标量类型名称，如 str、bytes、object
    __future_scalars__ = {"str", "bytes", "object"}
    
    # 设置变量 __array_api_version__，表示当前代码所使用的数组 API 版本
    __array_api_version__ = "2022.12"
    
    # 导入 _array_api_info 模块中的 __array_namespace_info__ 变量
    from ._array_api_info import __array_namespace_info__
    # 初始化 numpy 核心模块的限制
    _core.getlimits._register_known_types()
    
    # 定义 __all__ 列表，包括 numpy 的子模块、核心模块、矩阵模块、直方图实现、NaN 函数实现、
    # 基本函数实现、二维基础实现、形状基础实现、类型检查实现、数组集合操作实现、ufunc 类似实现、
    # 数组填充实现、工具实现、步幅技巧实现、多项式实现、输入输出实现、索引技巧实现以及一些特定的符号。
    __all__ = list(
        __numpy_submodules__ |
        set(_core.__all__) |
        set(_mat.__all__) |
        set(lib._histograms_impl.__all__) |
        set(lib._nanfunctions_impl.__all__) |
        set(lib._function_base_impl.__all__) |
        set(lib._twodim_base_impl.__all__) |
        set(lib._shape_base_impl.__all__) |
        set(lib._type_check_impl.__all__) |
        set(lib._arraysetops_impl.__all__) |
        set(lib._ufunclike_impl.__all__) |
        set(lib._arraypad_impl.__all__) |
        set(lib._utils_impl.__all__) |
        set(lib._stride_tricks_impl.__all__) |
        set(lib._polynomial_impl.__all__) |
        set(lib._npyio_impl.__all__) |
        set(lib._index_tricks_impl.__all__) |
        {"emath", "show_config", "__version__", "__array_namespace_info__"}
    )
    
    # 过滤掉 Cython 无害警告
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed")
    
    # 定义 __dir__ 函数，返回公共符号列表，包括全局变量和 numpy 子模块
    def __dir__():
        public_symbols = (
            globals().keys() | __numpy_submodules__
        )
        # 排除特定符号，如 matrixlib、matlib、tests、conftest、version、compat、distutils、array_api
        public_symbols -= {
            "matrixlib", "matlib", "tests", "conftest", "version", 
            "compat", "distutils", "array_api"
        }
        return list(public_symbols)
    
    # 导入 Pytest 测试工具
    from numpy._pytesttester import PytestTester
    # 针对当前模块进行 Pytest 测试
    test = PytestTester(__name__)
    # 删除 PytestTester 变量，确保不会被误用
    del PytestTester
    
    # 定义 _sanity_check 函数，用于快速检查常见环境错误
    def _sanity_check():
        """
        快速检查常见环境错误，例如错误的 BLAS ABI 版本等。
        参考 https://github.com/numpy/numpy/issues/8577 和其他类似问题报告。
        """
        try:
            x = ones(2, dtype=float32)
            if not abs(x.dot(x) - float32(2.0)) < 1e-5:
                raise AssertionError()
        except AssertionError:
            # 抛出运行时错误，指示可能的安装问题
            msg = ("当前 Numpy 安装（{!r}）未能通过简单的健全性检查。"
                   "可能的原因包括错误的 BLAS 库链接，或混合使用多种包管理器（pip、conda、apt 等）。"
                   "查阅关闭的 numpy 问题报告寻找类似的问题。")
            raise RuntimeError(msg.format(__file__)) from None
    
    # 执行健全性检查
    _sanity_check()
    # 删除 _sanity_check 函数，确保不会被误用
    del _sanity_check
    def _mac_os_check():
        """
        Quick Sanity check for Mac OS look for accelerate build bugs.
        Testing numpy polyfit calls init_dgelsd(LAPACK)
        """
        try:
            # 创建一个包含数值的数组
            c = array([3., 2., 1.])
            # 生成一个等差数列作为 x 值
            x = linspace(0, 2, 5)
            # 使用多项式系数 c 和 x 值计算 y 值
            y = polyval(c, x)
            # 使用 polyfit 函数拟合多项式，计算拟合结果及协方差矩阵
            _ = polyfit(x, y, 2, cov=True)
        except ValueError:
            pass

    if sys.platform == "darwin":
        from . import exceptions
        # 捕获所有警告信息
        with warnings.catch_warnings(record=True) as w:
            # 执行 Mac OS 系统检查
            _mac_os_check()
            # 如果捕获到警告信息
            if len(w) > 0:
                for _wn in w:
                    if _wn.category is exceptions.RankWarning:
                        # 获取警告信息的类别和内容
                        error_message = f"{_wn.category.__name__}: {str(_wn.message)}"
                        # 构建运行时错误信息
                        msg = (
                            "Polyfit sanity test emitted a warning, most likely due "
                            "to using a buggy Accelerate backend."
                            "\nIf you compiled yourself, more information is available at:"
                            "\nhttps://numpy.org/devdocs/building/index.html"
                            "\nOtherwise report this to the vendor "
                            "that provided NumPy.\n\n{}\n".format(error_message))
                        # 抛出运行时错误
                        raise RuntimeError(msg)
                del _wn
            del w
    del _mac_os_check

    def hugepage_setup():
        """
        We usually use madvise hugepages support, but on some old kernels it
        is slow and thus better avoided. Specifically kernel version 4.6 
        had a bug fix which probably fixed this:
        https://github.com/torvalds/linux/commit/7cf91a98e607c2f935dbcc177d70011e95b8faff
        """
        # 从环境变量中获取是否启用 NUMPY_MADVISE_HUGEPAGE
        use_hugepage = os.environ.get("NUMPY_MADVISE_HUGEPAGE", None)
        # 如果当前系统为 Linux 且未设置 NUMPY_MADVISE_HUGEPAGE
        if sys.platform == "linux" and use_hugepage is None:
            try:
                # 默认启用 hugepage
                use_hugepage = 1
                # 解析内核版本号
                kernel_version = os.uname().release.split(".")[:2]
                kernel_version = tuple(int(v) for v in kernel_version)
                # 如果内核版本低于 4.6，则禁用 hugepage
                if kernel_version < (4, 6):
                    use_hugepage = 0
            except ValueError:
                use_hugepage = 0
        elif use_hugepage is None:
            # 如果不是 Linux，仍然默认启用 hugepage
            use_hugepage = 1
        else:
            # 将环境变量设置转换为整数
            use_hugepage = int(use_hugepage)
        return use_hugepage

    # 注意：此行代码目前仅对 Linux 有效
    _core.multiarray._set_madvise_hugepage(hugepage_setup())
    # 从当前命名空间中删除 hugepage_setup 变量
    del hugepage_setup

    # 如果 NumPy 在子解释器中重新加载或导入，则发出警告
    # 这个操作在 Python 层面完成，因为 C 模块可能不会重新加载，
    # 并且这样组织更加清晰。
    _core.multiarray._multiarray_umath._reload_guard()

    # TODO: 现在环境变量已经是“weak”，可以完全删除它
    # 设置 NumPy 的推广状态，使用环境变量 NPY_PROMOTION_STATE 的值，默认为 "weak"
    _core._set_promotion_state(
        os.environ.get("NPY_PROMOTION_STATE", "weak"))

    # 告诉 PyInstaller 哪里可以找到 hook-numpy.py 文件
    def _pyinstaller_hooks_dir():
        from pathlib import Path
        # 返回包含 _pyinstaller 目录的绝对路径作为字符串列表
        return [str(Path(__file__).with_name("_pyinstaller").resolve())]
# 删除已导入的用于内部使用的标准库模块和警告模块中的符号
del os, sys, warnings
```