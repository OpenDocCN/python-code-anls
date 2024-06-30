# `D:\src\scipysrc\scipy\scipy\sparse\_sputils.py`

```
""" Utility functions for sparse matrix module
"""

import sys  # 导入系统模块
from typing import Any, Literal, Optional, Union  # 导入类型提示相关的模块
import operator  # 导入运算符模块
import numpy as np  # 导入NumPy库，用np作为别名
from math import prod  # 从math库中导入prod函数（计算乘积）
import scipy.sparse as sp  # 导入SciPy稀疏矩阵模块，并用sp作为别名
from scipy._lib._util import np_long, np_ulong  # 从SciPy的内部工具模块导入np_long和np_ulong

__all__ = ['upcast', 'getdtype', 'getdata', 'isscalarlike', 'isintlike',
           'isshape', 'issequence', 'isdense', 'ismatrix', 'get_sum_dtype']

supported_dtypes = [np.bool_, np.byte, np.ubyte, np.short, np.ushort, np.intc,
                    np.uintc, np_long, np_ulong, np.longlong, np.ulonglong,
                    np.float32, np.float64, np.longdouble, 
                    np.complex64, np.complex128, np.clongdouble]

_upcast_memo = {}  # 空字典，用于缓存类型转换结果


def upcast(*args):
    """Returns the nearest supported sparse dtype for the
    combination of one or more types.

    upcast(t0, t1, ..., tn) -> T  where T is a supported dtype

    Examples
    --------
    >>> from scipy.sparse._sputils import upcast
    >>> upcast('int32')
    <type 'numpy.int32'>
    >>> upcast('bool')
    <type 'numpy.bool_'>
    >>> upcast('int32','float32')
    <type 'numpy.float64'>
    >>> upcast('bool',complex,float)
    <type 'numpy.complex128'>

    """
    t = _upcast_memo.get(hash(args))  # 从缓存中获取已经计算过的类型转换结果
    if t is not None:
        return t

    upcast = np.result_type(*args)  # 计算给定类型组合的最接近的类型

    for t in supported_dtypes:
        if np.can_cast(upcast, t):  # 检查是否可以将计算出的类型转换为支持的类型
            _upcast_memo[hash(args)] = t  # 将结果加入缓存
            return t

    raise TypeError(f'no supported conversion for types: {args!r}')  # 如果无法转换，则抛出类型错误


def upcast_char(*args):
    """Same as `upcast` but taking dtype.char as input (faster)."""
    t = _upcast_memo.get(args)  # 检查是否已经缓存了结果
    if t is not None:
        return t
    t = upcast(*map(np.dtype, args))  # 调用upcast函数计算类型
    _upcast_memo[args] = t  # 将结果加入缓存
    return t


def upcast_scalar(dtype, scalar):
    """Determine data type for binary operation between an array of
    type `dtype` and a scalar.
    """
    return (np.array([0], dtype=dtype) * scalar).dtype  # 计算数组和标量之间二进制操作的数据类型


def downcast_intp_index(arr):
    """
    Down-cast index array to np.intp dtype if it is of a larger dtype.

    Raise an error if the array contains a value that is too large for
    intp.
    """
    if arr.dtype.itemsize > np.dtype(np.intp).itemsize:  # 如果索引数组的数据类型比np.intp大
        if arr.size == 0:  # 如果数组为空，则将其转换为np.intp类型
            return arr.astype(np.intp)
        maxval = arr.max()  # 获取数组中的最大值
        minval = arr.min()  # 获取数组中的最小值
        if maxval > np.iinfo(np.intp).max or minval < np.iinfo(np.intp).min:  # 如果最大值或最小值超出了np.intp的范围
            raise ValueError("Cannot deal with arrays with indices larger "
                             "than the machine maximum address size "
                             "(e.g. 64-bit indices on 32-bit machine).")
        return arr.astype(np.intp)  # 将数组转换为np.intp类型
    return arr  # 如果不需要转换，则返回原始数组


def to_native(A):
    """
    Ensure that the data type of the NumPy array `A` has native byte order.

    `A` must be a NumPy array.  If the data type of `A` does not have native
    byte order, a copy of `A` with a native byte order is returned. Otherwise
    `A` is returned.
    """
    dt = A.dtype  # 获取数组A的数据类型
    # 如果数据类型 dt 是本地字节顺序，则直接返回数组 A，避免不必要地创建输入数组的视图。
    if dt.isnative:
        return A
    # 如果数据类型 dt 不是本地字节顺序，则调用 np.asarray() 将数组 A 转换为本地字节顺序的数组。
    return np.asarray(A, dtype=dt.newbyteorder('native'))
# 从输入的数据类型或默认值中获取有效的数据类型
def getdtype(dtype, a=None, default=None):
    """Function used to simplify argument processing. If 'dtype' is not
    specified (is None), returns a.dtype; otherwise returns a np.dtype
    object created from the specified dtype argument. If 'dtype' and 'a'
    are both None, construct a data type out of the 'default' parameter.
    Furthermore, 'dtype' must be in 'allowed' set.
    """
    # 如果 dtype 为 None，则尝试获取参数 a 的数据类型
    if dtype is None:
        try:
            newdtype = a.dtype
        except AttributeError as e:
            # 如果无法获取 a 的数据类型，并且提供了默认值，则使用默认值创建数据类型
            if default is not None:
                newdtype = np.dtype(default)
            else:
                # 否则抛出类型错误异常
                raise TypeError("could not interpret data type") from e
    else:
        # 如果 dtype 不为 None，则直接使用指定的 dtype 创建数据类型
        newdtype = np.dtype(dtype)

    # 检查新的数据类型是否在支持的数据类型集合中
    if newdtype not in supported_dtypes:
        supported_dtypes_fmt = ", ".join(t.__name__ for t in supported_dtypes)
        # 如果不在支持的数据类型中，则抛出值错误异常
        raise ValueError(f"scipy.sparse does not support dtype {newdtype.name}. "
                         f"The only supported types are: {supported_dtypes_fmt}.")
    
    return newdtype


# 从给定对象生成一个 ndarray，并在结果为对象数组时生成警告
def getdata(obj, dtype=None, copy=False) -> np.ndarray:
    """
    This is a wrapper of `np.array(obj, dtype=dtype, copy=copy)`
    that will generate a warning if the result is an object array.
    """
    # 使用 np.array() 函数根据指定的参数生成 ndarray
    data = np.array(obj, dtype=dtype, copy=copy)
    # 调用 getdtype 检查数据类型是否合法，此处仅用于验证，不需要返回值
    getdtype(data.dtype)
    return data


# 根据输入数组的整数类型，确定一个适当的索引数据类型，以容纳数组中的数据
def get_index_dtype(arrays=(), maxval=None, check_contents=False):
    """
    Based on input (integer) arrays `a`, determine a suitable index data
    type that can hold the data in the arrays.

    Parameters
    ----------
    arrays : tuple of array_like
        Input arrays whose types/contents to check
    maxval : float, optional
        Maximum value needed
    check_contents : bool, optional
        Whether to check the values in the arrays and not just their types.
        Default: False (check only the types)

    Returns
    -------
    dtype : dtype
        Suitable index data type (int32 or int64)

    """

    # 定义 np.int32 的最小值和最大值
    int32min = np.int32(np.iinfo(np.int32).min)
    int32max = np.int32(np.iinfo(np.int32).max)

    # 根据 np.intc().itemsize 来决定使用 np.int32 还是 np.int64，由于与 pythran 的不良交互，不直接使用 intc
    dtype = np.int32 if np.intc().itemsize == 4 else np.int64
    
    # 如果指定了 maxval，则根据其值来确定使用的数据类型
    if maxval is not None:
        maxval = np.int64(maxval)
        if maxval > int32max:
            dtype = np.int64

    # 如果 arrays 是 np.ndarray 类型，则转换为元组处理
    if isinstance(arrays, np.ndarray):
        arrays = (arrays,)
    # 遍历输入的数组列表
    for arr in arrays:
        # 将每个数组转换为 NumPy 数组
        arr = np.asarray(arr)
        # 检查当前数组的数据类型是否可以转换为 np.int32
        if not np.can_cast(arr.dtype, np.int32):
            # 如果需要检查数组内容
            if check_contents:
                # 如果数组大小为0，则无需更大的数据类型
                if arr.size == 0:
                    # 跳过当前数组，选择下一个数组
                    continue
                # 如果数组的数据类型是整数类型
                elif np.issubdtype(arr.dtype, np.integer):
                    # 计算数组中的最大值和最小值
                    maxval = arr.max()
                    minval = arr.min()
                    # 如果数组的取值范围在int32的表示范围内
                    if minval >= int32min and maxval <= int32max:
                        # 无需更大的数据类型，跳过当前数组
                        continue

            # 如果以上条件都不满足，则选择 np.int64 作为数据类型
            dtype = np.int64
            # 结束循环，已找到满足条件的数据类型
            break

    # 返回最终确定的数据类型
    return dtype
# 模仿 numpy 的 np.sum 的类型转换，根据给定的 dtype 返回一个合适的数据类型
def get_sum_dtype(dtype: np.dtype) -> np.dtype:
    if dtype.kind == 'u' and np.can_cast(dtype, np.uint):
        return np.uint
    if np.can_cast(dtype, np.int_):
        return np.int_
    return dtype

# 检查 x 是否是标量、数组标量或者零维数组
def isscalarlike(x) -> bool:
    return np.isscalar(x) or (isdense(x) and x.ndim == 0)

# 检查 x 是否可以作为稀疏矩阵的索引，返回布尔值
def isintlike(x) -> bool:
    if np.ndim(x) != 0:
        return False
    try:
        operator.index(x)  # 尝试将 x 转换为整数索引
    except (TypeError, ValueError):
        try:
            loose_int = bool(int(x) == x)  # 尝试强制转换 x 到整数并检查是否相等
        except (TypeError, ValueError):
            return False
        if loose_int:
            msg = "Inexact indices into sparse matrices are not allowed"
            raise ValueError(msg)  # 如果 x 是近似整数，抛出值错误异常
        return loose_int
    return True

# 检查 x 是否是一个有效的维度元组，根据参数检查是否为非负数且是否允许 1 维形状
def isshape(x, nonneg=False, *, allow_1d=False) -> bool:
    ndim = len(x)
    if ndim != 2 and not (allow_1d and ndim == 1):
        return False
    for d in x:
        if not isintlike(d):
            return False
        if nonneg and d < 0:
            return False
    return True

# 检查 t 是否为序列（list、tuple 或 numpy 数组），且其中元素应为标量
def issequence(t) -> bool:
    return ((isinstance(t, (list, tuple)) and
            (len(t) == 0 or np.isscalar(t[0]))) or
            (isinstance(t, np.ndarray) and (t.ndim == 1)))

# 检查 t 是否为矩阵（包含序列，且序列中每个元素为标量），或者为二维 numpy 数组
def ismatrix(t) -> bool:
    return ((isinstance(t, (list, tuple)) and
             len(t) > 0 and issequence(t[0])) or
            (isinstance(t, np.ndarray) and t.ndim == 2))

# 检查 x 是否为密集数组（numpy.ndarray 类型）
def isdense(x) -> bool:
    return isinstance(x, np.ndarray)

# 验证轴参数的类型和取值范围，类似于 NumPy 的实现，不允许传入元组类型的轴
def validateaxis(axis) -> None:
    if axis is None:
        return
    axis_type = type(axis)
    if axis_type == tuple:
        raise TypeError("Tuples are not accepted for the 'axis' parameter. "
                        "Please pass in one of the following: "
                        "{-2, -1, 0, 1, None}.")
    if not np.issubdtype(np.dtype(axis_type), np.integer):
        raise TypeError(f"axis must be an integer, not {axis_type.__name__}")
    if not (-2 <= axis <= 1):
        raise ValueError("axis out of range")

# 模仿 numpy.matrix 处理 shape 参数的方式，验证形状参数是否合法，返回一个整数元组
def check_shape(args, current_shape=None, *, allow_1d=False) -> tuple[int, ...]:
    Parameters
    ----------
    args : array_like
        Data structures providing information about the shape of the sparse array.
    current_shape : tuple, optional
        The current shape of the sparse array or matrix.
        If None (default), the current shape will be inferred from args.
    allow_1d : bool, optional
        If True, then 1-D or 2-D arrays are accepted.
        If False (default), then only 2-D arrays are accepted and an error is
        raised otherwise.

    Returns
    -------
    new_shape: tuple
        The new shape after validation.
    """
    # 如果参数 args 的长度为 0，则抛出类型错误异常
    if len(args) == 0:
        raise TypeError("function missing 1 required positional argument: "
                        "'shape'")
    
    # 如果参数 args 的长度为 1，则尝试将其转换为迭代器，若无法转换则认为其为一个整数，构造成包含一个元素的元组 new_shape
    if len(args) == 1:
        try:
            shape_iter = iter(args[0])
        except TypeError:
            new_shape = (operator.index(args[0]), )
        else:
            new_shape = tuple(operator.index(arg) for arg in shape_iter)
    else:
        # 将每个参数转换为整数，并构造成元组 new_shape
        new_shape = tuple(operator.index(arg) for arg in args)

    # 如果当前形状 current_shape 为 None，则根据 allow_1d 参数检查 new_shape 的长度，并确保其元素都为正整数
    if current_shape is None:
        if allow_1d:
            if len(new_shape) not in (1, 2):
                raise ValueError('shape must be a 1- or 2-tuple of positive '
                                 'integers')
        elif len(new_shape) != 2:
            raise ValueError('shape must be a 2-tuple of positive integers')
        if any(d < 0 for d in new_shape):
            raise ValueError("'shape' elements cannot be negative")
    else:
        # 只有在需要时才检查当前大小
        current_size = prod(current_shape)

        # 检查是否存在负数
        negative_indexes = [i for i, x in enumerate(new_shape) if x < 0]
        if not negative_indexes:
            new_size = prod(new_shape)
            if new_size != current_size:
                raise ValueError('cannot reshape array of size {} into shape {}'
                                 .format(current_size, new_shape))
        elif len(negative_indexes) == 1:
            # 若存在一个负数，则根据该负数进行处理
            skip = negative_indexes[0]
            specified = prod(new_shape[:skip] + new_shape[skip+1:])
            unspecified, remainder = divmod(current_size, specified)
            if remainder != 0:
                err_shape = tuple('newshape' if x < 0 else x for x in new_shape)
                raise ValueError('cannot reshape array of size {} into shape {}'
                                 ''.format(current_size, err_shape))
            new_shape = new_shape[:skip] + (unspecified,) + new_shape[skip+1:]
        else:
            # 若存在多个负数，则抛出异常
            raise ValueError('can only specify one unknown dimension')

    # 最终检查 new_shape 的长度是否为 2（或者若允许 1D，则长度是否为 1），否则抛出异常
    if len(new_shape) != 2 and not (allow_1d and len(new_shape) == 1):
        raise ValueError('matrix shape must be two-dimensional')

    # 返回经过验证的 new_shape
    return new_shape
def check_reshape_kwargs(kwargs):
    """Unpack keyword arguments for reshape function.

    This is useful because keyword arguments after star arguments are not
    allowed in Python 2, but star keyword arguments are. This function unpacks
    'order' and 'copy' from the star keyword arguments (with defaults) and
    throws an error for any remaining.
    """

    # 从 star keyword arguments 中解析 'order'，默认为 'C'
    order = kwargs.pop('order', 'C')
    # 从 star keyword arguments 中解析 'copy'，默认为 False
    copy = kwargs.pop('copy', False)
    if kwargs:  # 如果还有未使用的 kwargs
        # 抛出类型错误，指明哪些关键字参数是不被期望的
        raise TypeError('reshape() got unexpected keywords arguments: {}'
                        .format(', '.join(kwargs.keys())))
    return order, copy


def is_pydata_spmatrix(m) -> bool:
    """
    Check whether object is pydata/sparse matrix, avoiding importing the module.
    """
    # 获取 'sparse' 模块中的 'SparseArray' 类
    base_cls = getattr(sys.modules.get('sparse'), 'SparseArray', None)
    # 返回是否找到 'SparseArray' 类且输入对象 m 是该类的实例
    return base_cls is not None and isinstance(m, base_cls)


def convert_pydata_sparse_to_scipy(
    arg: Any,
    target_format: Optional[Literal["csc", "csr"]] = None,
    accept_fv: Any = None,
) -> Union[Any, "sp.spmatrix"]:
    """
    Convert a pydata/sparse array to scipy sparse matrix,
    pass through anything else.
    """
    if is_pydata_spmatrix(arg):
        # 如果输入对象是 pydata/sparse 稀疏矩阵
        try:
            # 尝试将其转换为 scipy 稀疏矩阵，使用 accept_fv 参数（若提供）
            arg = arg.to_scipy_sparse(accept_fv=accept_fv)
        except TypeError:
            # 如果不支持 accept_fv 参数，则尝试无参数转换
            arg = arg.to_scipy_sparse()
        if target_format is not None:
            # 如果指定了目标格式，则转换为相应格式
            arg = arg.asformat(target_format)
        elif arg.format not in ("csc", "csr"):
            # 否则，如果格式不是 'csc' 或 'csr'，转换为 'csc' 格式
            arg = arg.tocsc()
    return arg


###############################################################################
# Wrappers for NumPy types that are deprecated

# Numpy versions of these functions raise deprecation warnings, the
# ones below do not.

def matrix(*args, **kwargs):
    # 返回通过 np.array() 创建的矩阵视图
    return np.array(*args, **kwargs).view(np.matrix)


def asmatrix(data, dtype=None):
    if isinstance(data, np.matrix) and (dtype is None or data.dtype == dtype):
        # 如果输入数据已经是 np.matrix 类型且数据类型匹配，则返回原始数据
        return data
    # 否则，将输入数据转换为 np.matrix 类型
    return np.asarray(data, dtype=dtype).view(np.matrix)

###############################################################################


def _todata(s) -> np.ndarray:
    """Access nonzero values, possibly after summing duplicates.

    Parameters
    ----------
    s : sparse array
        Input sparse array.

    Returns
    -------
    data: ndarray
      Nonzero values of the array, with shape (s.nnz,)

    """
    if isinstance(s, sp._data._data_matrix):
        # 如果输入是 pydata/sparse 的 _data_matrix 类型，则获取去重后的数据
        return s._deduped_data()

    if isinstance(s, sp.dok_array):
        # 如果输入是 pydata/sparse 的 dok_array 类型，则获取数据值数组
        return np.fromiter(s.values(), dtype=s.dtype, count=s.nnz)

    if isinstance(s, sp.lil_array):
        # 如果输入是 pydata/sparse 的 lil_array 类型，则展开数据到数组中
        data = np.empty(s.nnz, dtype=s.dtype)
        sp._csparsetools.lil_flatten_to_array(s.data, data)
        return data

    # 否则，将输入转换为 coo 格式并获取去重后的数据
    return s.tocoo()._deduped_data()
```