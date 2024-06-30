# `D:\src\scipysrc\scipy\scipy\linalg\lapack.py`

```
"""
Low-level LAPACK functions (:mod:`scipy.linalg.lapack`)
=======================================================

This module contains low-level functions from the LAPACK library.

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

   get_lapack_funcs

All functions
-------------

"""
#
# Author: Pearu Peterson, March 2002
#

import numpy as np                    # 导入 NumPy 库，用于数值计算
from .blas import _get_funcs, _memoize_get_funcs  # 从当前包中导入 _get_funcs 和 _memoize_get_funcs 函数
from scipy.linalg import _flapack      # 从 SciPy 的 linalg 模块导入 _flapack
from re import compile as regex_compile  # 导入 re 模块中的 compile 函数并重命名为 regex_compile

try:
    from scipy.linalg import _clapack  # 尝试从 SciPy 的 linalg 模块导入 _clapack
except ImportError:
    _clapack = None                    # 若导入失败，则将 _clapack 设为 None

try:
    from scipy.linalg import _flapack_64  # 尝试从 SciPy 的 linalg 模块导入 _flapack_64
    HAS_ILP64 = True                   # 若导入成功，则设置 HAS_ILP64 为 True
except ImportError:
    HAS_ILP64 = False                   # 若导入失败，则设置 HAS_ILP64 为 False
    _flapack_64 = None                  # 并将 _flapack_64 设为 None


# Expose all functions (only flapack --- clapack is an implementation detail)
empty_module = None                     # 初始化空变量 empty_module 为 None
from scipy.linalg._flapack import *     # 从 SciPy 的 linalg._flapack 模块导入所有内容（函数和变量）
del empty_module                        # 删除空变量 empty_module

__all__ = ['get_lapack_funcs']          # 将 get_lapack_funcs 函数加入到模块的 __all__ 列表中

# some convenience alias for complex functions
_lapack_alias = {                       # 定义一个字典 _lapack_alias，用于存储一些复杂函数的简便别名
    'corghr': 'cunghr', 'zorghr': 'zunghr',
    'corghr_lwork': 'cunghr_lwork', 'zorghr_lwork': 'zunghr_lwork',
    'corgqr': 'cungqr', 'zorgqr': 'zungqr',
    'cormqr': 'cunmqr', 'zormqr': 'zunmqr',
    'corgrq': 'cungrq', 'zorgrq': 'zungrq',
}

# Place guards against docstring rendering issues with special characters
p1 = regex_compile(r'with bounds (?P<b>.*?)( and (?P<s>.*?) storage){0,1}\n')  # 编译正则表达式 p1，用于匹配特定格式的文本
p2 = regex_compile(r'Default: (?P<d>.*?)\n')  # 编译正则表达式 p2，用于匹配特定格式的文本


def backtickrepl(m):
    if m.group('s'):
        return ('with bounds ``{}`` with ``{}`` storage\n'
                ''.format(m.group('b'), m.group('s')))  # 如果匹配到第二个分组，则返回特定格式的文本
    else:
        return 'with bounds ``{}``\n'.format(m.group('b'))  # 否则返回另一种格式的文本


for routine in [ssyevr, dsyevr, cheevr, zheevr,
                ssyevx, dsyevx, cheevx, zheevx,
                ssygvd, dsygvd, chegvd, zhegvd]:
    if routine.__doc__:
        routine.__doc__ = p1.sub(backtickrepl, routine.__doc__)  # 对于每个函数文档字符串，应用正则表达式替换
        routine.__doc__ = p2.sub('Default ``\\1``\n', routine.__doc__)  # 对于每个函数文档字符串，应用正则表达式替换
    else:
        # 如果以上条件都不满足，则执行以下代码块
        continue
# 删除变量 regex_compile, p1, p2, backtickrepl
del regex_compile, p1, p2, backtickrepl

# 使用装饰器 @_memoize_get_funcs 对 get_lapack_funcs 函数进行装饰，用于缓存结果
@_memoize_get_funcs
# 定义函数 get_lapack_funcs，返回指定 LAPACK 函数对象
def get_lapack_funcs(names, arrays=(), dtype=None, ilp64=False):
    """Return available LAPACK function objects from names.

    Arrays are used to determine the optimal prefix of LAPACK routines.

    Parameters
    ----------
    names : str or sequence of str
        Name(s) of LAPACK functions without type prefix.

    arrays : sequence of ndarrays, optional
        Arrays can be given to determine optimal prefix of LAPACK
        routines. If not given, double-precision routines will be
        used, otherwise the most generic type in arrays will be used.

    dtype : str or dtype, optional
        Data-type specifier. Not used if `arrays` is non-empty.

    ilp64 : {True, False, 'preferred'}, optional
        Whether to return ILP64 routine variant.
        Choosing 'preferred' returns ILP64 routine if available, and
        otherwise the 32-bit routine. Default: False

    Returns
    -------
    funcs : list
        List containing the found function(s).

    Notes
    -----
    This routine automatically chooses between Fortran/C
    interfaces. Fortran code is used whenever possible for arrays with
    column major order. In all other cases, C code is preferred.

    In LAPACK, the naming convention is that all functions start with a
    type prefix, which depends on the type of the principal
    matrix. These can be one of {'s', 'd', 'c', 'z'} for the NumPy
    types {float32, float64, complex64, complex128} respectively, and
    are stored in attribute ``typecode`` of the returned functions.

    Examples
    --------
    Suppose we would like to use '?lange' routine which computes the selected
    norm of an array. We pass our array in order to get the correct 'lange'
    flavor.

    >>> import numpy as np
    >>> import scipy.linalg as LA
    >>> rng = np.random.default_rng()

    >>> a = rng.random((3,2))
    >>> x_lange = LA.get_lapack_funcs('lange', (a,))
    >>> x_lange.typecode
    'd'
    >>> x_lange = LA.get_lapack_funcs('lange',(a*1j,))
    >>> x_lange.typecode
    'z'

    Several LAPACK routines work best when its internal WORK array has
    the optimal size (big enough for fast computation and small enough to
    avoid waste of memory). This size is determined also by a dedicated query
    to the function which is often wrapped as a standalone function and
    commonly denoted as ``###_lwork``. Below is an example for ``?sysv``

    >>> a = rng.random((1000, 1000))
    >>> b = rng.random((1000, 1)) * 1j
    >>> # We pick up zsysv and zsysv_lwork due to b array
    ... xsysv, xlwork = LA.get_lapack_funcs(('sysv', 'sysv_lwork'), (a, b))
    >>> opt_lwork, _ = xlwork(a.shape[0])  # returns a complex for 'z' prefix
    >>> udut, ipiv, x, info = xsysv(a, b, lwork=int(opt_lwork.real))

    """
    # 处理 ilp64 参数，若其为字符串 'preferred'，则根据 HAS_ILP64 变量的值决定是否使用 ILP64
    if isinstance(ilp64, str):
        if ilp64 == 'preferred':
            ilp64 = HAS_ILP64
        else:
            raise ValueError("Invalid value for 'ilp64'")
    # 如果 ilp64 不为真，则执行以下代码块
    if not ilp64:
        # 调用 _get_funcs 函数，获取与 LAPACK 相关的函数
        return _get_funcs(names, arrays, dtype,
                          "LAPACK", _flapack, _clapack,
                          "flapack", "clapack", _lapack_alias,
                          ilp64=False)
    # 如果 ilp64 为真，则执行以下代码块
    else:
        # 如果没有 ILP64 支持，则抛出 RuntimeError 异常
        if not HAS_ILP64:
            raise RuntimeError("LAPACK ILP64 routine requested, but Scipy "
                               "compiled only with 32-bit BLAS")
        # 调用 _get_funcs 函数，获取与 LAPACK ILP64 相关的函数
        return _get_funcs(names, arrays, dtype,
                          "LAPACK", _flapack_64, None,
                          "flapack_64", None, _lapack_alias,
                          ilp64=True)
# 初始化一个变量，表示 numpy 中 np.int32 的最大值
_int32_max = np.iinfo(np.int32).max
# 初始化一个变量，表示 numpy 中 np.int64 的最大值
_int64_max = np.iinfo(np.int64).max

# 定义一个函数，用于计算 LAPACK 返回的浮点型工作数组大小并转换为整数
def _compute_lwork(routine, *args, **kwargs):
    """
    Round floating-point lwork returned by lapack to integer.

    Several LAPACK routines compute optimal values for LWORK, which
    they return in a floating-point variable. However, for large
    values of LWORK, single-precision floating point is not sufficient
    to hold the exact value --- some LAPACK versions (<= 3.5.0 at
    least) truncate the returned integer to single precision and in
    some cases this can be smaller than the required value.

    Examples
    --------
    >>> from scipy.linalg import lapack
    >>> n = 5000
    >>> s_r, s_lw = lapack.get_lapack_funcs(('sysvx', 'sysvx_lwork'))
    >>> lwork = lapack._compute_lwork(s_lw, n)
    >>> lwork
    32000

    """
    # 获取 routine 的数据类型
    dtype = getattr(routine, 'dtype', None)
    # 获取 routine 的整数数据类型
    int_dtype = getattr(routine, 'int_dtype', None)
    # 调用 LAPACK 的函数计算工作数组大小
    ret = routine(*args, **kwargs)
    # 如果返回的最后一个值不为 0，则抛出 ValueError 异常
    if ret[-1] != 0:
        raise ValueError("Internal work array size computation failed: "
                         "%d" % (ret[-1],))

    # 如果返回值的长度为 2，则返回第一个值转换为整数后的结果
    if len(ret) == 2:
        return _check_work_float(ret[0].real, dtype, int_dtype)
    else:
        # 否则返回每个返回值转换为整数后的结果的元组
        return tuple(_check_work_float(x.real, dtype, int_dtype)
                     for x in ret[:-1])


def _check_work_float(value, dtype, int_dtype):
    """
    Convert LAPACK-returned work array size float to integer,
    carefully for single-precision types.
    """
    # 如果数据类型是 np.float32 或 np.complex64
    if dtype == np.float32 or dtype == np.complex64:
        # 单精度浮点数的情况下，取下一个浮点数以避免 LAPACK 代码中的可能截断
        value = np.nextafter(value, np.inf, dtype=np.float32)

    # 将 value 转换为整数
    value = int(value)
    # 如果整数数据类型的字节大小为 4
    if int_dtype.itemsize == 4:
        # 如果 value 小于 0 或者大于 _int32_max，则抛出异常
        if value < 0 or value > _int32_max:
            raise ValueError("Too large work array required -- computation "
                             "cannot be performed with standard 32-bit"
                             " LAPACK.")
    elif int_dtype.itemsize == 8:
        # 如果整数数据类型的字节大小为 8
        # 如果 value 小于 0 或者大于 _int64_max，则抛出异常
        if value < 0 or value > _int64_max:
            raise ValueError("Too large work array required -- computation"
                             " cannot be performed with standard 64-bit"
                             " LAPACK.")
    # 返回转换后的整数 value
    return value
```