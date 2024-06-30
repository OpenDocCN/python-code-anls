# `D:\src\scipysrc\scipy\scipy\linalg\blas.py`

```
"""
Low-level BLAS functions (:mod:`scipy.linalg.blas`)
===================================================

This module contains low-level functions from the BLAS library.

.. versionadded:: 0.12.0

.. note::

   The common ``overwrite_<>`` option in many routines, allows the
   input arrays to be overwritten to avoid extra memory allocation.
   However this requires the array to satisfy two conditions
   which are memory order and the data type to match exactly the
   order and the type expected by the routine.

   As an example, if you pass a double precision float array to any
   ``S....`` routine which expects single precision arguments, f2py
   will create an intermediate array to match the argument types and
   overwriting will be performed on that intermediate array.

   Similarly, if a C-contiguous array is passed, f2py will pass a
   FORTRAN-contiguous array internally. Please make sure that these
   details are satisfied. More information can be found in the f2py
   documentation.

.. warning::

   These functions do little to no error checking.
   It is possible to cause crashes by mis-using them,
   so prefer using the higher-level routines in `scipy.linalg`.

Finding functions
-----------------

.. autosummary::
   :toctree: generated/

   get_blas_funcs
   find_best_blas_type

BLAS Level 1 functions
----------------------

.. autosummary::
   :toctree: generated/

   caxpy
   ccopy
   cdotc
   cdotu
   crotg
   cscal
   csrot
   csscal
   cswap
   dasum
   daxpy
   dcopy
   ddot
   dnrm2
   drot
   drotg
   drotm
   drotmg
   dscal
   dswap
   dzasum
   dznrm2
   icamax
   idamax
   isamax
   izamax
   sasum
   saxpy
   scasum
   scnrm2
   scopy
   sdot
   snrm2
   srot
   srotg
   srotm
   srotmg
   sscal
   sswap
   zaxpy
   zcopy
   zdotc
   zdotu
   zdrot
   zdscal
   zrotg
   zscal
   zswap

BLAS Level 2 functions
----------------------

.. autosummary::
   :toctree: generated/

   sgbmv
   sgemv
   sger
   ssbmv
   sspr
   sspr2
   ssymv
   ssyr
   ssyr2
   stbmv
   stpsv
   strmv
   strsv
   dgbmv
   dgemv
   dger
   dsbmv
   dspr
   dspr2
   dsymv
   dsyr
   dsyr2
   dtbmv
   dtpsv
   dtrmv
   dtrsv
   cgbmv
   cgemv
   cgerc
   cgeru
   chbmv
   chemv
   cher
   cher2
   chpmv
   chpr
   chpr2
   ctbmv
   ctbsv
   ctpmv
   ctpsv
   ctrmv
   ctrsv
   csyr
   zgbmv
   zgemv
   zgerc
   zgeru
   zhbmv
   zhemv
   zher
   zher2
   zhpmv
   zhpr
   zhpr2
   ztbmv
   ztbsv
   ztpmv
   ztrmv
   ztrsv
   zsyr

BLAS Level 3 functions
----------------------

.. autosummary::
   :toctree: generated/

   sgemm
   ssymm
   ssyr2k
   ssyrk
   strmm
   strsm
   dgemm
   dsymm
   dsyr2k
   dsyrk
   dtrmm
   dtrsm
   cgemm
   chemm
   cher2k
   cherk
   csymm
   csyr2k
   csyrk
   ctrmm
   ctrsm
   zgemm
   zhemm
   zher2k
   zherk
   zsymm
   zsyr2k
   zsyrk
   ztrmm
   ztrsm

"""
#
# Author: Pearu Peterson, March 2002
#         refactoring by Fabian Pedregosa, March 2010
#

# 导出模块中的公共函数列表
__all__ = ['get_blas_funcs', 'find_best_blas_type']
import numpy as np  # 导入 NumPy 库，用于科学计算
import functools  # 导入 functools 模块，用于高阶函数（函数式编程）

from scipy.linalg import _fblas  # 从 SciPy 的 linalg 子模块中导入 _fblas
try:
    from scipy.linalg import _cblas  # 尝试从 SciPy 的 linalg 子模块中导入 _cblas
except ImportError:
    _cblas = None  # 如果导入失败，将 _cblas 设为 None

try:
    from scipy.linalg import _fblas_64  # 尝试从 SciPy 的 linalg 子模块中导入 _fblas_64
    HAS_ILP64 = True  # 如果导入成功，表示支持 ILP64 数据类型
except ImportError:
    HAS_ILP64 = False  # 如果导入失败，表示不支持 ILP64 数据类型
    _fblas_64 = None  # 将 _fblas_64 设为 None

# Expose all functions (only fblas --- cblas is an implementation detail)
empty_module = None
from scipy.linalg._fblas import *  # noqa: E402, F403
del empty_module

# all numeric dtypes '?bBhHiIlLqQefdgFDGO' that are safe to be converted to

# single precision float   : '?bBhH!!!!!!ef!!!!!!'
# double precision float   : '?bBhHiIlLqQefdg!!!!'
# single precision complex : '?bBhH!!!!!!ef!!F!!!'
# double precision complex : '?bBhHiIlLqQefdgFDG!'

_type_score = {x: 1 for x in '?bBhHef'}  # 设置基本类型分数为1
_type_score.update({x: 2 for x in 'iIlLqQd'})  # 更新整数类型分数为2

# Handle float128(g) and complex256(G) separately in case non-Windows systems.
# On Windows, the values will be rewritten to the same key with the same value.
_type_score.update({'F': 3, 'D': 4, 'g': 2, 'G': 4})  # 处理特定的大精度浮点和复数类型

# Final mapping to the actual prefixes and dtypes
_type_conv = {1: ('s', np.dtype('float32')),  # 映射得分1到单精度浮点类型
              2: ('d', np.dtype('float64')),  # 映射得分2到双精度浮点类型
              3: ('c', np.dtype('complex64')),  # 映射得分3到单精度复数类型
              4: ('z', np.dtype('complex128'))}  # 映射得分4到双精度复数类型

# some convenience alias for complex functions
_blas_alias = {'cnrm2': 'scnrm2', 'znrm2': 'dznrm2',  # 复数函数的简便别名映射
               'cdot': 'cdotc', 'zdot': 'zdotc',
               'cger': 'cgerc', 'zger': 'zgerc',
               'sdotc': 'sdot', 'sdotu': 'sdot',
               'ddotc': 'ddot', 'ddotu': 'ddot'}


def find_best_blas_type(arrays=(), dtype=None):
    """Find best-matching BLAS/LAPACK type.

    Arrays are used to determine the optimal prefix of BLAS routines.

    Parameters
    ----------
    arrays : sequence of ndarrays, optional
        Arrays can be given to determine optimal prefix of BLAS
        routines. If not given, double-precision routines will be
        used, otherwise the most generic type in arrays will be used.
    dtype : str or dtype, optional
        Data-type specifier. Not used if `arrays` is non-empty.

    Returns
    -------
    prefix : str
        BLAS/LAPACK prefix character.
    dtype : dtype
        Inferred Numpy data type.
    prefer_fortran : bool
        Whether to prefer Fortran order routines over C order.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.linalg.blas as bla
    >>> rng = np.random.default_rng()
    >>> a = rng.random((10,15))
    >>> b = np.asfortranarray(a)  # Change the memory layout order
    >>> bla.find_best_blas_type((a,))
    ('d', dtype('float64'), False)
    >>> bla.find_best_blas_type((a*1j,))
    ('z', dtype('complex128'), False)
    >>> bla.find_best_blas_type((b,))
    ('d', dtype('float64'), True)

    """
    dtype = np.dtype(dtype)  # 转换 dtype 参数为 NumPy 的 dtype 对象
    max_score = _type_score.get(dtype.char, 5)  # 获取指定 dtype 对应的最大分数，默认为5
    prefer_fortran = False  # 初始化是否优先使用 Fortran 顺序的标志为 False
    # 如果arrays非空，则执行以下逻辑
    if arrays:
        # 如果arrays中只有一个元素，使用快速路径
        if len(arrays) == 1:
            # 获取arrays[0]的dtype对应的分数，如果找不到则默认为5
            max_score = _type_score.get(arrays[0].dtype.char, 5)
            # 检查arrays[0]是否以Fortran顺序存储
            prefer_fortran = arrays[0].flags['FORTRAN']
        else:
            # 遍历arrays中每个元素，获取其dtype对应的分数，如果找不到则默认为5
            scores = [_type_score.get(x.dtype.char, 5) for x in arrays]
            # 获取scores中的最大值
            max_score = max(scores)
            # 获取最大值在scores中的索引
            ind_max_score = scores.index(max_score)
            # 如果最大分数为3且同时包含2，则安全地将float64和complex64混合提升为complex128，即前缀'z'
            if max_score == 3 and (2 in scores):
                max_score = 4

            # 检查arrays[ind_max_score]是否以Fortran顺序存储
            if arrays[ind_max_score].flags['FORTRAN']:
                # 优先选择以Fortran顺序存储的数组作为主导数组
                prefer_fortran = True

    # 根据max_score从_type_conv获取对应的LAPACK前缀和dtype，如果找不到则使用'd'和double precision float64作为默认值
    prefix, dtype = _type_conv.get(max_score, ('d', np.dtype('float64')))

    # 返回计算得到的前缀、dtype和是否优先选择Fortran顺序存储的布尔值
    return prefix, dtype, prefer_fortran
# 返回可用的 BLAS/LAPACK 函数。

# 根据输入的参数选择最佳的 BLAS 类型。
# 如果 names 是字符串，则转换成单元素元组，设置 unpack 为 True。
# 转换 dtype 成为 numpy 的数据类型对象。
# 设置 module1 和 module2 分别为 (cmodule, cmodule_name) 和 (fmodule, fmodule_name)。

# 如果 prefer_fortran 为 True，则交换 module1 和 module2 的值。

# 遍历 names 中的每个 name：
#   构造 func_name 为 prefix + name。
#   根据 alias 字典，如果有替换则使用替换后的 func_name。
#   尝试从 module1[0] 中获取对应的 func 对象，如果获取不到则尝试从 module2[0] 中获取。
#   如果仍然获取不到，抛出 ValueError 异常，指示找不到对应的函数。

#   将 func 的 module_name 设置为 module1[1]。
#   设置 func 的 typecode 为 prefix。
#   设置 func 的 dtype 为 dtype。
#   如果 ilp64 为 False，则设置 func 的 int_dtype 为 np.intc。
#   否则设置 func 的 int_dtype 为 np.int64。
#   设置 func 的 prefix 为 prefix（向后兼容）。
#   将 func 添加到 funcs 列表中。

# 如果 unpack 为 True，则返回 funcs 列表中的第一个函数对象。
# 否则返回 funcs 列表。
    if isinstance(ilp64, str):
        # 检查是否 ilp64 是字符串类型
        if ilp64 == 'preferred':
            # 如果 ilp64 是 'preferred'，则将其设置为 HAS_ILP64 的值
            ilp64 = HAS_ILP64
        else:
            # 如果 ilp64 不是有效的字符串值，则抛出数值错误异常
            raise ValueError("Invalid value for 'ilp64'")

    # 如果 ilp64 为 False，则使用 32 位 BLAS 函数
    if not ilp64:
        return _get_funcs(names, arrays, dtype,
                          "BLAS", _fblas, _cblas, "fblas", "cblas",
                          _blas_alias, ilp64=False)
    else:
        # 如果 ilp64 为 True，并且没有 ILP64 支持，则引发运行时错误
        if not HAS_ILP64:
            raise RuntimeError("BLAS ILP64 routine requested, but Scipy "
                               "compiled only with 32-bit BLAS")
        # 否则使用 64 位 BLAS 函数
        return _get_funcs(names, arrays, dtype,
                          "BLAS", _fblas_64, None, "fblas_64", None,
                          _blas_alias, ilp64=True)
```