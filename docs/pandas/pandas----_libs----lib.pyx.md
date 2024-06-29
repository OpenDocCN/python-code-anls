# `D:\src\scipysrc\pandas\pandas\_libs\lib.pyx`

```
# 导入abc模块中的collections类
from collections import abc
# 导入Decimal类
from decimal import Decimal
# 导入Enum类
from enum import Enum
# 导入getsizeof函数
from sys import getsizeof
# 导入Literal和_GenericAlias类型
from typing import (
    Literal,
    _GenericAlias,
)

# 导入Cython库中的部分模块和函数
cimport cython
from cpython.datetime cimport (
    PyDate_Check,
    PyDateTime_Check,
    PyDelta_Check,
    PyTime_Check,
    date,
    datetime,
    import_datetime,
    time,
    timedelta,
)
from cpython.iterator cimport PyIter_Check
from cpython.number cimport PyNumber_Check
from cpython.object cimport (
    Py_EQ,
    PyObject,
    PyObject_RichCompareBool,
)
from cpython.ref cimport Py_INCREF
from cpython.sequence cimport PySequence_Check
from cpython.tuple cimport (
    PyTuple_New,
    PyTuple_SET_ITEM,
)
from cython cimport (
    Py_ssize_t,
    floating,
)

# 导入pandas._config模块中的using_pyarrow_string_dtype函数
from pandas._config import using_pyarrow_string_dtype

# 导入pandas._libs.missing模块中的check_na_tuples_nonequal函数
from pandas._libs.missing import check_na_tuples_nonequal

# 调用import_datetime函数
import_datetime()

# 导入numpy库，并将其别名为np
import numpy as np

# 导入cnp模块中的numpy部分内容
cimport numpy as cnp
from numpy cimport (
    NPY_OBJECT,
    PyArray_Check,
    PyArray_GETITEM,
    PyArray_ITER_DATA,
    PyArray_ITER_NEXT,
    PyArray_IterNew,
    PyArray_SETITEM,
    complex128_t,
    flatiter,
    float64_t,
    int32_t,
    int64_t,
    intp_t,
    ndarray,
    uint8_t,
    uint64_t,
)

# 调用cnp.import_array()函数
cnp.import_array()

# 导入pandas._libs.interval模块中的Interval类
from pandas._libs.interval import Interval

# 从pandas/parser/pd_parser.h文件中外部导入floatify和PandasParser_IMPORT函数
cdef extern from "pandas/parser/pd_parser.h":
    int floatify(object, float64_t *result, int *maybe_int) except -1
    void PandasParser_IMPORT()

# 调用PandasParser_IMPORT函数
PandasParser_IMPORT

# 导入pandas._libs.util模块中的部分内容
from pandas._libs cimport util
from pandas._libs.util cimport (
    INT64_MAX,
    INT64_MIN,
    UINT64_MAX,
    is_nan,
)

# 从pandas._libs.tslibs中导入特定模块和类
from pandas._libs.tslibs import (
    OutOfBoundsDatetime,
    OutOfBoundsTimedelta,
)
from pandas._libs.tslibs.period import Period

# 导入pandas._libs.missing模块中的部分函数和常量
from pandas._libs.missing cimport (
    C_NA,
    checknull,
    is_matching_na,
    is_null_datetime64,
    is_null_timedelta64,
)

# 从pandas._libs.tslibs.conversion中导入convert_to_tsobject函数
from pandas._libs.tslibs.conversion cimport convert_to_tsobject

# 从pandas._libs.tslibs.nattype中导入部分内容
from pandas._libs.tslibs.nattype cimport (
    NPY_NAT,
    c_NaT as NaT,
    checknull_with_nat,
)

# 从pandas._libs.tslibs.offsets中导入is_offset_object函数
from pandas._libs.tslibs.offsets cimport is_offset_object

# 从pandas._libs.tslibs.period中导入is_period_object函数
from pandas._libs.tslibs.period cimport is_period_object

# 从pandas._libs.tslibs.timedeltas中导入convert_to_timedelta64函数
from pandas._libs.tslibs.timedeltas cimport convert_to_timedelta64

# 从pandas._libs.tslibs.timezones中导入tz_compare函数
from pandas._libs.tslibs.timezones cimport tz_compare

# 定义常量，将C语言中的整型常量转换为Python对象
cdef:
    object oINT64_MAX = <int64_t>INT64_MAX
    object oINT64_MIN = <int64_t>INT64_MIN
    object oUINT64_MAX = <uint64_t>UINT64_MAX

    # 将np.nan赋值给NaN变量
    float64_t NaN = <float64_t>np.nan

# 定义Python可见的整型常量
i8max = <int64_t>INT64_MAX
u8max = <uint64_t>UINT64_MAX

# 尝试导入pyarrow库，如果失败则将PYARROW_INSTALLED设置为False
cdef bint PYARROW_INSTALLED = False
try:
    import pyarrow as pa
    PYARROW_INSTALLED = True
except ImportError:
    pa = None

# 定义一个函数，使用Cython装饰器指定wraparound和boundscheck参数为False
@cython.wraparound(False)
@cython.boundscheck(False)
def memory_usage_of_objects(arr: object[:]) -> int64_t:
    """
    Return the memory usage of an object array in bytes.

    Does not include the actual bytes of the pointers
    """
    # 定义变量 i、n 为 Python 对象的长度以及初始大小为 0 的 int64_t 类型的 size
    cdef:
        Py_ssize_t i
        Py_ssize_t n
        int64_t size = 0

    # 获取数组 arr 的长度
    n = len(arr)
    
    # 遍历数组 arr 中的每个元素，累加每个元素占用的内存大小到 size 变量中
    for i in range(n):
        size += getsizeof(arr[i])
    
    # 返回累计的内存大小
    return size
# ----------------------------------------------------------------------
# 根据给定的对象判断是否为标量值（scalar）。
def is_scalar(val: object) -> bool:
    """
    Return True if given object is scalar.

    Parameters
    ----------
    val : object
        This includes:

        - numpy array scalar (e.g. np.int64)
        - Python builtin numerics
        - Python builtin byte arrays and strings
        - None
        - datetime.datetime
        - datetime.timedelta
        - Period
        - decimal.Decimal
        - Interval
        - DateOffset
        - Fraction
        - Number.

    Returns
    -------
    bool
        Return True if given object is scalar.

    See Also
    --------
    api.types.is_list_like : Check if the input is list-like.
    api.types.is_integer : Check if the input is an integer.
    api.types.is_float : Check if the input is a float.
    api.types.is_bool : Check if the input is a boolean.

    Examples
    --------
    >>> import datetime
    >>> dt = datetime.datetime(2018, 10, 3)
    >>> pd.api.types.is_scalar(dt)
    True

    >>> pd.api.types.is_scalar([2, 3])
    False

    >>> pd.api.types.is_scalar({0: 1, 2: 3})
    False

    >>> pd.api.types.is_scalar((0, 2))
    False

    pandas supports PEP 3141 numbers:

    >>> from fractions import Fraction
    >>> pd.api.types.is_scalar(Fraction(3, 5))
    True
    """

    # 使用 C 优化的检查开始
    if (cnp.PyArray_IsAnyScalar(val)
            # 对于 Python 3 的 bytearray，PyArray_IsAnyScalar 总是 False
            or PyDate_Check(val)
            or PyDelta_Check(val)
            or PyTime_Check(val)
            # 与 numpy 不同，None 在这里被认为是标量；参见 np.isscalar
            or val is C_NA
            or val is None):
        return True

    # 接下来使用 C 优化的检查排除常见的非标量值，然后再使用非优化的检查。
    if PySequence_Check(val):
        # 例如：list、tuple
        # 包括 np.ndarray 和 Series，这些在 PyNumber_Check 下也可能返回 True
        return False

    # 注意：PyNumber_Check 包括 Decimal、Fraction、numbers.Number
    return (PyNumber_Check(val)
            or is_period_object(val)
            or isinstance(val, Interval)
            or is_offset_object(val))


# ----------------------------------------------------------------------
# 获取 NumPy 标量的 itemsize，如果不是 NumPy 标量则返回 -1。
cdef int64_t get_itemsize(object val):
    """
    Get the itemsize of a NumPy scalar, -1 if not a NumPy scalar.

    Parameters
    ----------
    val : object

    Returns
    -------
    is_ndarray : bool
    """
    if cnp.PyArray_CheckScalar(val):
        return cnp.PyArray_DescrFromScalar(val).itemsize
    else:
        return -1


# ----------------------------------------------------------------------
# 检查对象是否是迭代器（iterator）。
def is_iterator(obj: object) -> bool:
    """
    Check if the object is an iterator.

    This is intended for generators, not list-like objects.

    Parameters
    ----------
    obj : The object to check

    Returns
    -------
    is_iter : bool
        Whether `obj` is an iterator.

    Examples
    --------
    >>> import datetime
    >>> from pandas.api.types import is_iterator
    """
    # 检查对象是否为迭代器
    >>> is_iterator((x for x in []))
    # 生成器表达式是迭代器，因此返回 True
    True
    >>> is_iterator([1, 2, 3])
    # 列表不是迭代器，因此返回 False
    False
    >>> is_iterator(datetime.datetime(2017, 1, 1))
    # datetime 对象不是迭代器，因此返回 False
    False
    >>> is_iterator("foo")
    # 字符串不是迭代器，因此返回 False
    False
    >>> is_iterator(1)
    # 整数不是迭代器，因此返回 False
    False
    """
    # 调用 CPython 的 PyIter_Check 函数来检查对象是否为迭代器，并返回检查结果
    return PyIter_Check(obj)
def item_from_zerodim(val: object) -> object:
    """
    If the value is a zerodim array, return the item it contains.

    Parameters
    ----------
    val : object
        Input value which may be a zerodim array.

    Returns
    -------
    object
        The contained item if val is zerodim, otherwise returns val itself.
    """
    # Check if val is a zerodim array using a C library function
    if cnp.PyArray_IsZeroDim(val):
        # Convert zerodim array to scalar and return
        return cnp.PyArray_ToScalar(cnp.PyArray_DATA(val), val)
    # Return val if not zerodim
    return val


@cython.wraparound(False)
@cython.boundscheck(False)
def fast_unique_multiple_list_gen(object gen, bint sort=True) -> list:
    """
    Generate a list of unique values from a generator of lists.

    Parameters
    ----------
    gen : generator object
        Generator that yields lists from which unique values are extracted.
    sort : bool
        Whether to sort the resulting list of unique values.

    Returns
    -------
    list
        A list containing unique values from the generator.
    """
    cdef:
        list buf
        Py_ssize_t j, n
        list uniques = []
        set table = set()
        object val

    # Iterate through the generator of lists
    for buf in gen:
        n = len(buf)
        # Iterate through each element in the current list
        for j in range(n):
            val = buf[j]
            # Check if the element is already in the set of unique values
            if val not in table:
                # If not, add it to the set and append to the result list
                table.add(val)
                uniques.append(val)
    # Optionally, sort the list of unique values
    if sort:
        try:
            uniques.sort()
        except TypeError:
            pass

    return uniques


@cython.wraparound(False)
@cython.boundscheck(False)
def dicts_to_array(dicts: list, columns: list):
    """
    Convert a list of dictionaries into a numpy array based on specified columns.

    Parameters
    ----------
    dicts : list
        List of dictionaries where each dictionary represents a row.
    columns : list
        List of column names to extract from dictionaries.

    Returns
    -------
    ndarray[object, ndim=2]
        Numpy array where each row corresponds to a dictionary from 'dicts'
        and columns correspond to 'columns'.
    """
    cdef:
        Py_ssize_t i, j, k, n
        ndarray[object, ndim=2] result
        dict row
        object col, onan = np.nan

    k = len(columns)
    n = len(dicts)

    # Initialize result array with the appropriate dimensions
    result = np.empty((n, k), dtype="O")

    # Iterate over each dictionary in the list
    for i in range(n):
        row = dicts[i]
        # Iterate over each column name
        for j in range(k):
            col = columns[j]
            # Check if the column exists in the current dictionary row
            if col in row:
                result[i, j] = row[col]
            else:
                result[i, j] = onan  # Assign NaN if column not found

    return result


def fast_zip(list ndarrays) -> ndarray[object]:
    """
    For zipping multiple ndarrays into an ndarray of tuples.

    Parameters
    ----------
    ndarrays : list
        List of ndarrays to be zipped together into tuples.

    Returns
    -------
    ndarray[object]
        Numpy array where each element is a tuple containing elements from
        corresponding positions of input ndarrays.
    """
    cdef:
        Py_ssize_t i, j, k, n
        ndarray[object, ndim=1] result
        flatiter it
        object val, tup

    k = len(ndarrays)  # Number of ndarrays to zip
    n = len(ndarrays[0])  # Length of the first ndarray

    # Initialize result array with object dtype
    result = np.empty(n, dtype=object)

    # Initialize tuples on the first pass
    arr = ndarrays[0]
    it = <flatiter>PyArray_IterNew(arr)
    for i in range(n):
        val = PyArray_GETITEM(arr, PyArray_ITER_DATA(it))
        tup = PyTuple_New(k)

        PyTuple_SET_ITEM(tup, 0, val)
        Py_INCREF(val)
        result[i] = tup
        PyArray_ITER_NEXT(it)

    return result
    # 遍历范围从1到k-1的整数，其中k为数组ndarrays的长度
    for j in range(1, k):
        # 获取当前索引j处的数组arr
        arr = ndarrays[j]
        # 创建一个新的迭代器it，用于迭代处理arr数组
        it = <flatiter>PyArray_IterNew(arr)
        # 检查当前数组arr的长度是否等于预期长度n，如果不等则抛出数值错误异常
        if len(arr) != n:
            raise ValueError("all arrays must be same length")

        # 遍历长度为n的范围，处理arr数组中的每个元素
        for i in range(n):
            # 获取迭代器it当前位置的元素值，并存储在变量val中
            val = PyArray_GETITEM(arr, PyArray_ITER_DATA(it))
            # 将val作为元组result的第i个元素的第j个值存储
            PyTuple_SET_ITEM(result[i], j, val)
            # 增加val的引用计数，以确保其在存储期间不被释放
            Py_INCREF(val)
            # 将迭代器it移到下一个位置，准备处理下一个元素
            PyArray_ITER_NEXT(it)

    # 返回填充好的result元组，其中包含了所有ndarrays中数据的组合
    return result
# 定义函数：生成反向索引器
def get_reverse_indexer(const intp_t[:] indexer, Py_ssize_t length) -> ndarray:
    """
    Reverse indexing operation.

    Given `indexer`, make `indexer_inv` of it, such that::

        indexer_inv[indexer[x]] = x

    Parameters
    ----------
    indexer : np.ndarray[np.intp]
        Input array of indices.
    length : int
        Length of the output array.

    Returns
    -------
    np.ndarray[np.intp]
        Array where values from `indexer` are mapped to their indices.

    Notes
    -----
    If indexer is not unique, only the first occurrence is accounted.
    """
    cdef:
        Py_ssize_t i, n = len(indexer)  # 获取索引器的长度
        ndarray[intp_t, ndim=1] rev_indexer  # 初始化反向索引器数组
        intp_t idx  # 索引变量

    rev_indexer = np.empty(length, dtype=np.intp)  # 创建指定长度的空数组
    rev_indexer[:] = -1  # 初始化数组为 -1
    for i in range(n):
        idx = indexer[i]  # 获取当前索引器的值
        if idx != -1:
            rev_indexer[idx] = i  # 在反向索引器中，根据索引器的值映射到索引

    return rev_indexer  # 返回生成的反向索引器数组


# 定义函数：检查数组中是否包含无穷大数值
@cython.wraparound(False)
@cython.boundscheck(False)
def has_infs(const floating[:] arr) -> bool:
    cdef:
        Py_ssize_t i, n = len(arr)  # 获取数组长度
        floating inf, neginf, val  # 定义特殊数值和当前值
        bint ret = False  # 返回值，默认为 False

    inf = np.inf  # 正无穷大
    neginf = -inf  # 负无穷大
    with nogil:
        for i in range(n):
            val = arr[i]  # 获取数组中的当前值
            if val == inf or val == neginf:
                ret = True  # 如果当前值为无穷大，则设置返回值为 True 并中断循环
                break
    return ret  # 返回结果：数组中是否存在无穷大数值的布尔值


# 定义函数：检查数组中是否仅包含整数或 NaN
@cython.boundscheck(False)
@cython.wraparound(False)
def has_only_ints_or_nan(const floating[:] arr) -> bool:
    cdef:
        floating val  # 当前值
        intp_t i  # 当前索引

    for i in range(len(arr)):
        val = arr[i]  # 获取数组中的当前值
        if (val != val) or (val == <int64_t>val):  # 检查是否为 NaN 或整数
            continue  # 如果是 NaN 或整数，则继续下一个循环
        else:
            return False  # 如果不是 NaN 或整数，则返回 False

    return True  # 如果数组中仅包含整数或 NaN，则返回 True


# 定义函数：将可能的索引数组转换为切片对象
def maybe_indices_to_slice(ndarray[intp_t, ndim=1] indices, int max_len):
    """
    Convert possibly indices array to slice.

    Parameters
    ----------
    indices : np.ndarray[np.intp]
        Input array of indices.
    max_len : int
        Maximum length for the slice.

    Returns
    -------
    slice or np.ndarray[np.intp]
        Slice object or indices array based on conditions.

    Notes
    -----
    - If `indices` is empty, returns a zero-length slice.
    - Handles cases where `indices` is a single index, out of bounds, or non-sequential.
    """
    cdef:
        Py_ssize_t i, n = len(indices)  # 获取索引数组的长度
        intp_t k, vstart, vlast, v  # 定义临时变量

    if n == 0:
        return slice(0, 0)  # 如果索引数组为空，则返回零长度的切片对象

    vstart = indices[0]  # 获取第一个索引值
    if vstart < 0 or max_len <= vstart:
        return indices  # 如果第一个索引值小于 0 或者超过最大长度，则返回原始索引数组

    if n == 1:
        return slice(vstart, <intp_t>(vstart + 1))  # 如果只有一个索引值，则返回以该索引值为起点的切片对象

    vlast = indices[n - 1]  # 获取最后一个索引值
    if vlast < 0 or max_len <= vlast:
        return indices  # 如果最后一个索引值小于 0 或者超过最大长度，则返回原始索引数组

    k = indices[1] - indices[0]  # 计算索引之间的差值
    if k == 0:
        return indices  # 如果差值为 0，则返回原始索引数组
    else:
        for i in range(2, n):
            v = indices[i]  # 获取当前索引值
            if v - indices[i - 1] != k:
                return indices  # 如果索引值不是等差数列，则返回原始索引数组

        if k > 0:
            return slice(vstart, <intp_t>(vlast + 1), k)  # 返回起始、终止和步长为 k 的切片对象
        else:
            if vlast == 0:
                return slice(vstart, None, k)  # 如果最后一个索引为 0，则返回从 vstart 开始，步长为 k 的切片对象
            else:
                return slice(vstart, <intp_t>(vlast - 1), k)  # 否则返回从 vstart 开始，终止为 vlast-1，步长为 k 的切片对象


# 定义函数：将可能的布尔值数组转换为切片对象
@cython.wraparound(False)
@cython.boundscheck(False)
def maybe_booleans_to_slice(ndarray[uint8_t, ndim=1] mask):
    """
    Convert possibly boolean mask array to slice.

    Parameters
    ----------
    mask : np.ndarray[np.uint8]
        Input boolean mask array.

    Returns
    -------
    np.ndarray[np.bool_] or slice
        Boolean mask or slice object based on conditions.

    Notes
    -----
    - If `mask` is empty or all False, returns empty boolean array.
    - Handles cases where `mask` starts and stops with True values.
    """
    cdef:
        Py_ssize_t i, n = len(mask)  # 获取布尔值数组的长度
        Py_ssize_t start = 0, end = 0  # 起始和结束索引
        bint started = False, finished = False  # 标记变量

    for i in range(n):
        if mask[i]:
            if finished:
                return mask.view(np.bool_)  # 如果已经完成，则返回布尔值数组
            if not started:
                started = True
                start = i  # 设置起始索引
        else:
            if finished:
                continue

            if started:
                end = i  # 设置结束索引
                finished = True

    # 根据起始和结束索引生成切片对象
    if started:
        return slice(start, end)
    else:
        return mask.view(np.bool_)  # 如果没有找到起始索引，则返回布尔值数组
    # 如果未开始索引，则返回一个空切片对象
    if not started:
        return slice(0, 0)
    # 如果未完成索引，返回从开始位置到最后的切片对象
    if not finished:
        return slice(start, None)
    # 否则，返回从开始位置到结束位置的切片对象
    else:
        return slice(start, end)
@cython.wraparound(False)
@cython.boundscheck(False)
# 定义一个函数，用于比较两个 N 维对象数组的每个元素，考虑 NaN 的位置
def array_equivalent_object(ndarray left, ndarray right) -> bool:
    """
    Perform an element by element comparison on N-d object arrays
    taking into account nan positions.
    """
    # left 和 right 都是对象类型的 ndarray，但我们不能在这里做注释说明，以免限制其维度。
    cdef:
        Py_ssize_t i, n = left.size  # 获取 left 数组的大小
        object x, y  # 定义对象 x 和 y
        cnp.broadcast mi = cnp.PyArray_MultiIterNew2(left, right)  # 创建一个广播迭代器，用于同时迭代 left 和 right

    # Caller is responsible for checking left.shape == right.shape
    # 调用者负责检查 left 和 right 的形状是否相同

    for i in range(n):
        # Analogous to: x = left[i]
        # 获取当前迭代器位置上 left 和 right 的元素
        x = <object>(<PyObject**>cnp.PyArray_MultiIter_DATA(mi, 0))[0]
        y = <object>(<PyObject**>cnp.PyArray_MultiIter_DATA(mi, 1))[0]

        # we are either not equal or both nan
        # I think None == None will be true here
        try:
            if PyArray_Check(x) and PyArray_Check(y):
                if x.shape != y.shape:
                    return False
                if x.dtype == y.dtype == object:
                    if not array_equivalent_object(x, y):
                        return False
                else:
                    # Circular import isn't great, but so it goes.
                    # TODO: could use np.array_equal?
                    from pandas.core.dtypes.missing import array_equivalent

                    if not array_equivalent(x, y):
                        return False

            elif (x is C_NA) ^ (y is C_NA):
                return False
            elif not (
                PyObject_RichCompareBool(x, y, Py_EQ)
                or is_matching_na(x, y, nan_matches_none=True)
            ):
                return False
        except (ValueError, TypeError):
            # Avoid raising ValueError when comparing Numpy arrays to other types
            if cnp.PyArray_IsAnyScalar(x) != cnp.PyArray_IsAnyScalar(y):
                # Only compare scalars to scalars and non-scalars to non-scalars
                return False
            elif (not (cnp.PyArray_IsPythonScalar(x) or cnp.PyArray_IsPythonScalar(y))
                  and not (isinstance(x, type(y)) or isinstance(y, type(x)))):
                # Check if non-scalars have the same type
                return False
            elif check_na_tuples_nonequal(x, y):
                # We have tuples where one Side has a NA and the other side does not
                # Only condition we may end up with a TypeError
                return False
            raise

        cnp.PyArray_MultiIter_NEXT(mi)  # 移动广播迭代器到下一个位置

    return True  # 如果所有元素比较都相等，则返回 True


ctypedef fused int6432_t:
    int64_t
    int32_t

@cython.wraparound(False)
@cython.boundscheck(False)
# 定义一个函数，用于比较 1 维整数数组的每个元素，通常用于索引器比较
def is_range_indexer(const int6432_t[:] left, Py_ssize_t n) -> bool:
    """
    Perform an element by element comparison on 1-d integer arrays, meant for indexer
    comparisons
    """
    cdef:
        Py_ssize_t i  # 定义索引变量 i

    if left.size != n:
        return False  # 如果数组 left 的大小不等于 n，则返回 False

    for i in range(n):
        # 比较 left[i] 是否等于 i
        if left[i] != i:
            return False  # 如果不相等，则返回 False

    return True  # 如果所有元素都与其索引相等，则返回 True
    # 返回布尔值 True，表示函数执行成功
    return True
# 设置 Cython 函数的 wraparound 参数为 False，避免负数索引的循环优化
# 设置 Cython 函数的 boundscheck 参数为 False，避免边界检查的性能开销
@cython.wraparound(False)
@cython.boundscheck(False)
def is_sequence_range(const int6432_t[:] sequence, int64_t step) -> bool:
    """
    Check if sequence is equivalent to a range with the specified step.
    """
    # 使用 Py_ssize_t 类型的变量 i 和 n 来存储序列的长度
    cdef:
        Py_ssize_t i, n = len(sequence)
        int6432_t first_element

    # 如果步长为 0，则直接返回 False
    if step == 0:
        return False
    # 如果序列长度为 0，则认为是等差数列，返回 True
    if n == 0:
        return True

    # 获取序列的第一个元素作为初始值
    first_element = sequence[0]
    # 遍历序列中的元素，检查是否符合等差数列的定义
    for i in range(1, n):
        if sequence[i] != first_element + i * step:
            return False
    # 若所有元素符合等差数列的条件，则返回 True
    return True


# 定义一个 fused type，支持两种 ndarray 对象类型：一维和二维的 object 类型数组
ctypedef fused ndarr_object:
    ndarray[object, ndim=1]
    ndarray[object, ndim=2]

# TODO: get rid of this in StringArray and modify
#  and go through ensure_string_array instead


# 设置 Cython 函数的 wraparound 参数为 False，避免负数索引的循环优化
# 设置 Cython 函数的 boundscheck 参数为 False，避免边界检查的性能开销
@cython.wraparound(False)
@cython.boundscheck(False)
def convert_nans_to_NA(ndarr_object arr) -> ndarray:
    """
    Helper for StringArray that converts null values that
    are not pd.NA(e.g. np.nan, None) to pd.NA. Assumes elements
    have already been validated as null.
    """
    # 使用 Py_ssize_t 类型的变量 i, m, n 来存储循环中的索引和维度信息
    cdef:
        Py_ssize_t i, m, n
        object val
        ndarr_object result

    # 将输入数组 arr 转换为 object 类型的 numpy 数组 result
    result = np.asarray(arr, dtype="object")

    # 如果 arr 是二维数组
    if arr.ndim == 2:
        m, n = arr.shape[0], arr.shape[1]
        # 遍历二维数组中的每个元素
        for i in range(m):
            for j in range(n):
                # 获取当前元素的值
                val = arr[i, j]
                # 如果当前元素不是字符串类型，则将其转换为 C_NA
                if not isinstance(val, str):
                    result[i, j] = <object>C_NA
    else:
        # 如果 arr 是一维数组
        n = len(arr)
        # 遍历一维数组中的每个元素
        for i in range(n):
            # 获取当前元素的值
            val = arr[i]
            # 如果当前元素不是字符串类型，则将其转换为 C_NA
            if not isinstance(val, str):
                result[i] = <object>C_NA

    # 返回处理后的结果数组
    return result


# 设置 Cython 函数的 wraparound 参数为 False，避免负数索引的循环优化
# 设置 Cython 函数的 boundscheck 参数为 False，避免边界检查的性能开销
cpdef ndarray[object] ensure_string_array(
        arr,
        object na_value=np.nan,
        bint convert_na_value=True,
        bint copy=True,
        bint skipna=True,
):
    """
    Returns a new numpy array with object dtype and only strings and na values.

    Parameters
    ----------
    arr : array-like
        The values to be converted to str, if needed.
    na_value : Any, default np.nan
        The value to use for na. For example, np.nan or pd.NA.
    convert_na_value : bool, default True
        If False, existing na values will be used unchanged in the new array.
    copy : bool, default True
        Whether to ensure that a new array is returned.
    skipna : bool, default True
        Whether or not to coerce nulls to their stringified form
        (e.g. if False, NaN becomes 'nan').

    Returns
    -------
    np.ndarray[object]
        An array with the input array's elements casted to str or nan-like.
    """
    # 使用 Py_ssize_t 类型的变量 i 和 n 来存储循环中的索引和数组长度
    cdef:
        Py_ssize_t i = 0, n = len(arr)
        bint already_copied = True
        ndarray[object] newarr
    # 检查对象是否具有 to_numpy 方法
    if hasattr(arr, "to_numpy"):
        # 检查对象是否具有 dtype 属性，并且其类型为日期时间类型
        if hasattr(arr, "dtype") and arr.dtype.kind in "mM":
            # 将 DataFrame 排除在外的 dtype 检查
            # GH#41409 TODO: 这里可能不是最佳的放置位置
            # 将 arr 转换为字符串类型的对象数组，并用指定的 na_value 填充缺失值
            out = arr.astype(str).astype(object)
            out[arr.isna()] = na_value
            return out
        # 将 arr 转换为对象数组
        arr = arr.to_numpy(dtype=object)
    # 如果 arr 不是数组，则将其转换为对象数组
    elif not util.is_array(arr):
        arr = np.array(arr, dtype="object")

    # 将 arr 转换为对象类型的 numpy 数组
    result = np.asarray(arr, dtype="object")

    # 如果需要复制数组且 result 和 arr 指向同一内存位置或共享内存，则进行复制
    if copy and (result is arr or np.shares_memory(arr, result)):
        # GH#54654
        result = result.copy()
    elif not copy and result is arr:
        already_copied = False
    elif not copy and not result.flags.writeable:
        # 特殊情况，result 是视图的情况
        already_copied = False

    # 如果 arr 的 dtype 类型是字符串类型，则直接返回 result
    if issubclass(arr.dtype.type, np.str_):
        return result

    # 如果 arr 的 dtype 类型是浮点数类型，则执行非优化路径
    if arr.dtype.kind == "f":  # non-optimized path
        # 遍历数组的每个元素
        for i in range(n):
            val = arr[i]

            # 如果还没有复制过 result，则进行复制操作
            if not already_copied:
                result = result.copy()
                already_copied = True

            # 如果元素不为空，则转换为字符串并赋值给 result[i]
            if not checknull(val):
                # 对于浮点数，f"{val}" 与 str(val) 并不总是等价的
                result[i] = str(val)
            else:
                # 如果需要转换 NA 值，则使用 na_value
                if convert_na_value:
                    val = na_value
                # 如果需要跳过 NA 值，则直接赋值 val，否则转换为字符串赋值
                if skipna:
                    result[i] = val
                else:
                    result[i] = f"{val}"

        return result

    # 将 arr 转换为对象类型的新 numpy 数组
    newarr = np.asarray(arr, dtype=object)
    # 遍历新数组的每个元素
    for i in range(n):
        val = newarr[i]

        # 如果元素是字符串类型，则跳过
        if isinstance(val, str):
            continue

        # 如果还没有复制过 result，则进行复制操作
        elif not already_copied:
            result = result.copy()
            already_copied = True

        # 如果元素不为空，则根据类型进行相应的转换和赋值操作
        if not checknull(val):
            # 如果元素是字节类型，则根据讨论的预期行为进行解码
            if isinstance(val, bytes):
                # GH#49658 讨论了这里的期望行为
                result[i] = val.decode()
            # 如果元素不是浮点数对象，则使用 f"{val}" 替代 str(val)
            elif not util.is_float_object(val):
                result[i] = f"{val}"
            else:
                # 对于浮点数，f"{val}" 与 str(val) 并不总是等价的
                result[i] = str(val)
        else:
            # 如果需要转换 NA 值，则使用 na_value
            if convert_na_value:
                val = na_value
            # 如果需要跳过 NA 值，则直接赋值 val，否则转换为字符串赋值
            if skipna:
                result[i] = val
            else:
                result[i] = f"{val}"

    return result
# 检查给定的对象是否全为类数组类型，返回布尔值
def is_all_arraylike(obj: list) -> bool:
    """
    Should we treat these as levels of a MultiIndex, as opposed to Index items?
    """
    # 声明变量和类型
    cdef:
        Py_ssize_t i, n = len(obj)  # 获取列表长度
        object val  # 声明对象变量
        bint all_arrays = True  # 布尔变量，用于标识是否全部为数组类型

    # 导入必要的模块和类
    from pandas.core.dtypes.generic import (
        ABCIndex,
        ABCMultiIndex,
        ABCSeries,
    )

    # 遍历对象列表
    for i in range(n):
        val = obj[i]  # 获取当前索引处的值
        # 检查当前值是否为列表、ABC系列、ABC索引，或者是否为数组类型（util.is_array(val)为真）
        # 同时排除ABC多重索引（ABCMultiIndex）
        if (not (isinstance(val, (list, ABCSeries, ABCIndex)) or util.is_array(val))
                or isinstance(val, ABCMultiIndex)):
            # TODO: EA?
            # 排除元组和冻结集合，因为它们可能包含在索引中
            all_arrays = False  # 将标志置为假
            break  # 结束循环

    return all_arrays  # 返回是否全部为数组类型的布尔值


# ------------------------------------------------------------------------------
# Groupby-related functions

# TODO: could do even better if we know something about the data. eg, index has
# 1-min data, binner has 5-min data, then bins are just strides in index. This
# is a general, O(max(len(values), len(binner))) method.
@cython.boundscheck(False)
@cython.wraparound(False)
def generate_bins_dt64(ndarray[int64_t, ndim=1] values, const int64_t[:] binner,
                       object closed="left", bint hasnans=False):
    """
    Int64 (datetime64) version of generic python version in ``groupby.py``.
    """
    # 声明变量和类型
    cdef:
        Py_ssize_t lenidx, lenbin, i, j, bc  # 声明整型变量
        ndarray[int64_t, ndim=1] bins  # 声明int64类型的一维数组
        int64_t r_bin, nat_count  # 声明int64类型变量和计数器
        bint right_closed = closed == "right"  # 根据闭合位置判断是否右闭合

    nat_count = 0  # 初始化NaT计数为0
    if hasnans:
        # 如果存在NaT，创建掩码并计算NaT的数量
        mask = values == NPY_NAT
        nat_count = np.sum(mask)
        values = values[~mask]  # 从值中移除NaT

    lenidx = len(values)  # 获取值的长度
    lenbin = len(binner)  # 获取binner数组的长度

    # 检查值和binner的长度是否有效
    if lenidx <= 0 or lenbin <= 0:
        raise ValueError("Invalid length for values or for binner")

    # 检查binner是否适合数据
    if values[0] < binner[0]:
        raise ValueError("Values falls before first bin")

    if values[lenidx - 1] > binner[lenbin - 1]:
        raise ValueError("Values falls after last bin")

    # 创建长度为lenbin-1的空int64类型数组
    bins = np.empty(lenbin - 1, dtype=np.int64)

    j = 0  # values数组的索引
    bc = 0  # bin计数器

    # 线性扫描
    if right_closed:
        # 如果右闭合，进行如下循环
        for i in range(0, lenbin - 1):
            r_bin = binner[i + 1]
            # 计算当前bin中的值的数量，然后前进到下一个bin
            while j < lenidx and values[j] <= r_bin:
                j += 1
            bins[bc] = j  # 将计数值存入bins数组
            bc += 1  # bin计数器加一
    else:
        # 如果左闭合，进行如下循环
        for i in range(0, lenbin - 1):
            r_bin = binner[i + 1]
            # 计算当前bin中的值的数量，然后前进到下一个bin
            while j < lenidx and values[j] < r_bin:
                j += 1
            bins[bc] = j  # 将计数值存入bins数组
            bc += 1  # bin计数器加一

    if nat_count > 0:
        # 将bins数组中的值向右移动NaT的数量
        bins = bins + nat_count
        bins = np.insert(bins, 0, nat_count)  # 在bins数组开头插入NaT的数量

    return bins  # 返回bins数组


@cython.boundscheck(False)
@cython.wraparound(False)
def get_level_sorter(
    ndarray[int64_t, ndim=1] codes, const intp_t[:] starts
) -> ndarray:
    """
    Placeholder function. Implementation details are not provided.
    """
    Argsort for a single level of a multi-index, keeping the order of higher
    levels unchanged. `starts` points to starts of same-key indices w.r.t
    to leading levels; equivalent to:
        np.hstack([codes[starts[i]:starts[i+1]].argsort(kind='mergesort')
            + starts[i] for i in range(len(starts) - 1)])


    Parameters
    ----------
    codes : np.ndarray[int64_t, ndim=1]
        输入参数，一个一维的 NumPy 整数数组，表示要排序的数据。
    starts : np.ndarray[intp, ndim=1]
        输入参数，一个一维的 NumPy 整数数组，表示每个同一键索引的起始位置。

    Returns
    -------
    np.ndarray[np.int, ndim=1]
        返回值，一个一维的 NumPy 整数数组，包含排序后的索引值。
    """
    cdef:
        Py_ssize_t i, l, r
        ndarray[intp_t, ndim=1] out = cnp.PyArray_EMPTY(1, codes.shape, cnp.NPY_INTP, 0)


    for i in range(len(starts) - 1):
        l, r = starts[i], starts[i + 1]
        out[l:r] = l + codes[l:r].argsort(kind="mergesort")


    return out
# 设置 Cython 编译器选项：禁用边界检查
@cython.boundscheck(False)
# 设置 Cython 编译器选项：禁用包裹模式
@cython.wraparound(False)
# 定义一个函数 count_level_2d，接受一个二维 uint8_t 类型的 ndarray（名称为 mask）、一个 intp_t 类型的一维数组（名称为 labels）、一个 Py_ssize_t 类型的常量（名称为 max_bin）
def count_level_2d(ndarray[uint8_t, ndim=2, cast=True] mask,
                   const intp_t[:] labels,
                   Py_ssize_t max_bin,
                   ):
    # 定义变量 i, j, k, n 以及一个二维 int64_t 类型的 ndarray（名称为 counts）
    cdef:
        Py_ssize_t i, j, k, n
        ndarray[int64_t, ndim=2] counts

    # 从 mask 的 shape 属性中获取 n 和 k 的值
    n, k = (<object>mask).shape

    # 初始化一个全零的二维数组 counts，形状为 (n, max_bin)，数据类型为 int64_t
    counts = np.zeros((n, max_bin), dtype="i8")
    
    # 使用 nogil 块进行无 GIL 的并行处理
    with nogil:
        # 遍历 n
        for i in range(n):
            # 遍历 k
            for j in range(k):
                # 如果 mask[i, j] 的值为真
                if mask[i, j]:
                    # 增加 counts[i, labels[j]] 的计数
                    counts[i, labels[j]] += 1

    # 返回 counts 数组
    return counts


# 设置 Cython 编译器选项：禁用包裹模式
@cython.wraparound(False)
# 设置 Cython 编译器选项：禁用边界检查
@cython.boundscheck(False)
# 定义一个函数 generate_slices，接受一个 intp_t 类型的一维数组（名称为 labels）和一个 Py_ssize_t 类型的常量（名称为 ngroups）
def generate_slices(const intp_t[:] labels, Py_ssize_t ngroups):
    # 定义变量 i, group_size, n, start，以及两个一维 int64_t 类型的数组（名称为 starts 和 ends）
    cdef:
        Py_ssize_t i, group_size, n, start
        intp_t lab
        int64_t[::1] starts, ends

    # 获取 labels 数组的长度赋值给 n
    n = len(labels)

    # 初始化一个全零的一维数组 starts 和 ends，长度为 ngroups，数据类型为 int64_t
    starts = np.zeros(ngroups, dtype=np.int64)
    ends = np.zeros(ngroups, dtype=np.int64)

    # 初始化 start 和 group_size
    start = 0
    group_size = 0
    
    # 使用 nogil 块进行无 GIL 的并行处理
    with nogil:
        # 遍历 n
        for i in range(n):
            # 获取 labels[i] 的值赋给 lab
            lab = labels[i]
            # 如果 lab 小于 0
            if lab < 0:
                # 增加 start 的值
                start += 1
            else:
                # 增加 group_size 的值
                group_size += 1
                # 如果 i 是 n - 1 或者 lab 不等于 labels[i + 1]
                if i == n - 1 or lab != labels[i + 1]:
                    # 设置 starts[lab] 和 ends[lab] 的值
                    starts[lab] = start
                    ends[lab] = start + group_size
                    # 增加 start 的值
                    start += group_size
                    # 重置 group_size
                    group_size = 0

    # 返回 starts 和 ends 数组的 numpy 数组形式
    return np.asarray(starts), np.asarray(ends)


# 定义一个函数 indices_fast，接受一个一维 intp_t 类型的 ndarray（名称为 index）、一个 int64_t 类型的一维数组（名称为 labels）、一个列表（名称为 keys）、一个列表，其中元素是 int64_t 类型的 ndarray（名称为 sorted_labels），返回一个字典
def indices_fast(ndarray[intp_t, ndim=1] index, const int64_t[:] labels, list keys,
                 list sorted_labels) -> dict:
    """
    Parameters
    ----------
    index : ndarray[intp]
    labels : ndarray[int64]
    keys : list
    sorted_labels : list[ndarray[int64]]
    """
    # 定义变量 i, j, k, lab, cur, start, n，以及一个空字典 result 和一个 object 类型的变量 tup
    cdef:
        Py_ssize_t i, j, k, lab, cur, start, n = len(labels)
        dict result = {}
        object tup

    # 获取 keys 列表的长度赋值给 k
    k = len(keys)

    # 从第一个非空条目开始
    j = 0
    for j in range(0, n):
        # 如果 labels[j] 不等于 -1
        if labels[j] != -1:
            # 跳出循环
            break
    else:
        # 返回空字典 result
        return result
    # 初始化 cur 和 start
    cur = labels[j]
    start = j

    # 遍历从 j+1 到 n 的范围
    for i in range(j+1, n):
        # 获取 labels[i] 的值赋给 lab
        lab = labels[i]

        # 如果 lab 不等于 cur
        if lab != cur:
            # 如果 lab 不等于 -1
            if lab != -1:
                # 如果 k 等于 1
                if k == 1:
                    # 当 k = 1 时，不返回元组作为键
                    tup = keys[0][sorted_labels[0][i - 1]]
                else:
                    # 创建一个 PyTuple 对象 tup
                    tup = PyTuple_New(k)
                    # 遍历 k
                    for j in range(k):
                        # 获取 keys[j][sorted_labels[j][i - 1]] 的值赋给 val
                        val = keys[j][sorted_labels[j][i - 1]]
                        # 设置 PyTuple 的第 j 个元素为 val
                        PyTuple_SET_ITEM(tup, j, val)
                        # 增加 val 的引用计数
                        Py_INCREF(val)
                # 设置 result[tup] 的值为 index[start:i]
                result[tup] = index[start:i]
            # 设置 start 的值为 i
            start = i
        # 设置 cur 的值为 lab

        cur = lab

    # 如果 k 等于 1
    if k == 1:
        # 当 k = 1 时，不返回元组作为键
        tup = keys[0][sorted_labels[0][n - 1]]
    else:
        # 创建一个 PyTuple 对象 tup
        tup = PyTuple_New(k)
        # 遍历 k
        for j in range(k):
            # 获取 keys[j][sorted_labels[j][n - 1]] 的值赋给 val
            val = keys[j][sorted_labels[j][n - 1]]
            # 设置 PyTuple 的第 j 个元素为 val
            PyTuple_SET_ITEM(tup, j, val)
            # 增加 val 的引用计数
            Py_INCREF(val)
    # 设置 result[tup] 的值为 index[start:]
    result[tup] = index[start:]

    # 返回 result 字典
    return result


# 从 core.common 导入快速推断检查功能
def is_float(obj: object) -> bool:
    """
    Return True if given object is float.

    Returns
    -------
    bool

    Examples
    --------
    >>> pd.api.types.is_float(1.0)
    True

    >>> pd.api.types.is_float(1)
    False
    """
    return util.is_float_object(obj)


def is_integer(obj: object) -> bool:
    """
    Return True if given object is integer.

    Returns
    -------
    bool

    Examples
    --------
    >>> pd.api.types.is_integer(1)
    True

    >>> pd.api.types.is_integer(1.0)
    False
    """
    return util.is_integer_object(obj)


def is_int_or_none(obj) -> bool:
    """
    Return True if given object is integer or None.

    Returns
    -------
    bool
    """
    return obj is None or util.is_integer_object(obj)


def is_bool(obj: object) -> bool:
    """
    Return True if given object is boolean.

    Returns
    -------
    bool

    Examples
    --------
    >>> pd.api.types.is_bool(True)
    True

    >>> pd.api.types.is_bool(1)
    False
    """
    return util.is_bool_object(obj)


def is_complex(obj: object) -> bool:
    """
    Return True if given object is complex.

    Returns
    -------
    bool

    Examples
    --------
    >>> pd.api.types.is_complex(1 + 1j)
    True

    >>> pd.api.types.is_complex(1)
    False
    """
    return util.is_complex_object(obj)


cpdef bint is_decimal(object obj):
    """
    Return True if given object is of type Decimal.

    Returns
    -------
    bool
    """
    return isinstance(obj, Decimal)


def is_list_like(obj: object, allow_sets: bool = True) -> bool:
    """
    Check if the object is list-like.

    Objects that are considered list-like are for example Python
    lists, tuples, sets, NumPy arrays, and Pandas Series.

    Strings and datetime objects, however, are not considered list-like.

    Parameters
    ----------
    obj : object
        Object to check.
    allow_sets : bool, default True
        If this parameter is False, sets will not be considered list-like.

    Returns
    -------
    bool
        Whether `obj` has list-like properties.

    Examples
    --------
    >>> import datetime
    >>> from pandas.api.types import is_list_like
    >>> is_list_like([1, 2, 3])
    True
    >>> is_list_like({1, 2, 3})
    True
    >>> is_list_like(datetime.datetime(2017, 1, 1))
    False
    >>> is_list_like("foo")
    False
    >>> is_list_like(1)
    False
    >>> is_list_like(np.array([2]))
    True
    >>> is_list_like(np.array(2))
    False
    """
    return c_is_list_like(obj, allow_sets)


cdef bint c_is_list_like(object obj, bint allow_sets) except -1:
    """
    Check if the object is list-like in C level.

    This function provides efficient checks for object types
    like NumPy arrays and lists.

    Parameters
    ----------
    obj : object
        Object to check.
    allow_sets : bint
        Whether sets should be considered list-like.

    Returns
    -------
    bint
        Whether `obj` has list-like properties.
    """
    # first, performance short-cuts for the most common cases
    if util.is_array(obj):
        # exclude zero-dimensional numpy arrays, effectively scalars
        return not cnp.PyArray_IsZeroDim(obj)
    elif isinstance(obj, list):
        return True
    # then the generic implementation
    return (
        # 检查对象是否可迭代，并且不是类对象（排除类对象）
        getattr(obj, "__iter__", None) is not None and not isinstance(obj, type)
        # 排除字符串、字节串和泛型别名作为类列表的情况
        # 排除具有 __iter__ 方法的泛型别名
        and not isinstance(obj, (str, bytes, _GenericAlias))
        # 排除零维度的鸭子数组，即实际上的标量
        and not (hasattr(obj, "ndim") and obj.ndim == 0)
        # 如果 allow_sets 为 False，则排除集合类型
        and not (allow_sets is False and isinstance(obj, abc.Set))
    )
# 检查给定对象是否为 pyarrow 的 Array 或 ChunkedArray 类型
def is_pyarrow_array(obj):
    """
    Return True if given object is a pyarrow Array or ChunkedArray.

    Returns
    -------
    bool
    """
    # 如果 pyarrow 已安装，则检查对象是否为 pyarrow 的 Array 或 ChunkedArray 类型
    if PYARROW_INSTALLED:
        return isinstance(obj, (pa.Array, pa.ChunkedArray))
    # 如果 pyarrow 未安装，则返回 False
    return False


# 映射不同类型名称到通用的数据类型分类
_TYPE_MAP = {
    "categorical": "categorical",                   # 类别型数据
    "category": "categorical",                      # 类别型数据
    "int8": "integer",                              # 8位整数
    "int16": "integer",                             # 16位整数
    "int32": "integer",                             # 32位整数
    "int64": "integer",                             # 64位整数
    "i": "integer",                                 # 整数
    "uint8": "integer",                             # 8位无符号整数
    "uint16": "integer",                            # 16位无符号整数
    "uint32": "integer",                            # 32位无符号整数
    "uint64": "integer",                            # 64位无符号整数
    "u": "integer",                                 # 整数
    "float32": "floating",                          # 32位浮点数
    "float64": "floating",                          # 64位浮点数
    "float128": "floating",                         # 128位浮点数
    "float256": "floating",                         # 256位浮点数
    "f": "floating",                                # 浮点数
    "complex64": "complex",                         # 64位复数
    "complex128": "complex",                        # 128位复数
    "complex256": "complex",                        # 256位复数
    "c": "complex",                                 # 复数
    "string": "string",                             # 字符串
    str: "string",                                  # 字符串
    "S": "bytes",                                   # 字节串
    "U": "string",                                  # 字符串
    "bool": "boolean",                              # 布尔值
    "b": "boolean",                                 # 布尔值
    "datetime64[ns]": "datetime64",                 # 时间日期类型
    "M": "datetime64",                              # 时间日期类型
    "timedelta64[ns]": "timedelta64",               # 时间间隔类型
    "m": "timedelta64",                             # 时间间隔类型
    "interval": "interval",                         # 时间间隔类型
    Period: "period",                               # 时间段类型
    datetime: "datetime64",                         # 时间日期类型
    date: "date",                                   # 日期类型
    time: "time",                                   # 时间类型
    timedelta: "timedelta64",                       # 时间间隔类型
    Decimal: "decimal",                             # 十进制数类型
    bytes: "bytes",                                 # 字节串
}


# 定义一个内部类 Seen，用于跟踪在类型转换过程中遇到的元素类型
@cython.internal
cdef class Seen:
    """
    Class for keeping track of the types of elements
    encountered when trying to perform type conversions.
    """

    cdef:
        bint int_             # seen_int (整数)
        bint nat_             # seen nat (自然数)
        bint bool_            # seen_bool (布尔值)
        bint null_            # seen_null (空值)
        bint nan_             # seen_np.nan (NaN值)
        bint uint_            # seen_uint (无符号整数)
        bint sint_            # seen_sint (有符号整数)
        bint float_           # seen_float (浮点数)
        bint object_          # seen_object (对象)
        bint complex_         # seen_complex (复数)
        bint datetime_        # seen_datetime (日期时间)
        bint coerce_numeric   # coerce data to numeric (强制将数据转换为数值型)
        bint timedelta_       # seen_timedelta (时间间隔)
        bint datetimetz_      # seen_datetimetz (带时区的日期时间)
        bint period_          # seen_period (时间段)
        bint interval_        # seen_interval (时间间隔)
        bint str_             # seen_str (字符串)
    # 初始化 Seen 实例的构造函数
    def __cinit__(self, bint coerce_numeric=False):
        """
        Initialize a Seen instance.

        Parameters
        ----------
        coerce_numeric : bool, default False
            Whether or not to force conversion to a numeric data type if
            initial methods to convert to numeric fail.
        """
        # 初始化各种数据类型的标志位
        self.int_ = False
        self.nat_ = False
        self.bool_ = False
        self.null_ = False
        self.nan_ = False
        self.uint_ = False
        self.sint_ = False
        self.float_ = False
        self.object_ = False
        self.complex_ = False
        self.datetime_ = False
        self.timedelta_ = False
        self.datetimetz_ = False
        self.period_ = False
        self.interval_ = False
        self.str_ = False
        # 设置是否强制将数据转换为数值类型的选项
        self.coerce_numeric = coerce_numeric

    # 检查是否存在 uint64 类型的数据转换冲突
    cdef bint check_uint64_conflict(self) except -1:
        """
        Check whether we can safely convert a uint64 array to a numeric dtype.

        There are two cases when conversion to numeric dtype with a uint64
        array is not safe (and will therefore not be performed)

        1) A NaN element is encountered.

           uint64 cannot be safely cast to float64 due to truncation issues
           at the extreme ends of the range.

        2) A negative number is encountered.

           There is no numerical dtype that can hold both negative numbers
           and numbers greater than INT64_MAX. Hence, at least one number
           will be improperly cast if we convert to a numeric dtype.

        Returns
        -------
        bool
            Whether or not we should return the original input array to avoid
            data truncation.

        Raises
        ------
        ValueError
            uint64 elements were detected, and at least one of the
            two conflict cases was also detected. However, we are
            trying to force conversion to a numeric dtype.
        """
        return (self.uint_ and (self.null_ or self.sint_)
                and not self.coerce_numeric)

    # 设置标志表明遇到了空值
    cdef saw_null(self):
        """
        Set flags indicating that a null value was encountered.
        """
        self.null_ = True
        self.float_ = True
    # 定义一个 C 扩展函数 saw_int，用于设置整数值被发现的标志位
    def saw_int(self, object val):
        """
        Set flags indicating that an integer value was encountered.

        In addition to setting a flag that an integer was seen, we
        also set two flags depending on the type of integer seen:

        1) sint_ : a signed numpy integer type or a negative (signed) number in the
                   range of [-2**63, 0) was encountered
        2) uint_ : an unsigned numpy integer type or a positive number in the range of
                   [2**63, 2**64) was encountered

        Parameters
        ----------
        val : Python int
            Value with which to set the flags.
        """
        # 设置整数值已被发现的标志位
        self.int_ = True
        # 设置 sint_ 标志位，检查是否是有符号整数或在 [-2**63, 0) 范围内的负数
        self.sint_ = (
            self.sint_
            or (oINT64_MIN <= val < 0)  # 检查是否在有符号整数范围内
            or isinstance(val, cnp.signedinteger)  # 检查是否是 numpy 的有符号整数类型
        )
        # 设置 uint_ 标志位，检查是否是无符号整数或在 [2**63, 2**64) 范围内的正数
        self.uint_ = (
            self.uint_
            or (oINT64_MAX < val <= oUINT64_MAX)  # 检查是否在无符号整数范围内
            or isinstance(val, cnp.unsignedinteger)  # 检查是否是 numpy 的无符号整数类型
        )

    @property
    def numeric_(self):
        # 返回 numeric_ 属性，指示当前对象是否包含复数、浮点数或整数
        return self.complex_ or self.float_ or self.int_

    @property
    def is_bool(self):
        # 返回 is_bool 属性，表示当前对象是否为布尔类型，排除 NaN 和 null 值
        # 即，不是 (任何不是布尔类型)
        return self.is_bool_or_na and not (self.nan_ or self.null_)

    @property
    def is_bool_or_na(self):
        # 返回 is_bool_or_na 属性，表示当前对象是否为布尔类型或缺失值
        # 即，不是 (任何不是布尔类型或缺失值)
        return self.bool_ and not (
            self.datetime_ or self.datetimetz_ or self.nat_ or self.timedelta_
            or self.period_ or self.interval_ or self.numeric_ or self.object_
        )
# 尝试根据类型推断并返回相应的标签字符串
cdef object _try_infer_map(object dtype):
    """
    If its in our map, just return the dtype.
    如果在我们的映射中找到，直接返回 dtype。
    """
    cdef:
        object val  # 用于存储属性值的变量
        str attr   # 用于迭代属性名称的变量
    for attr in ["type", "kind", "name", "base"]:
        # 在 ArrowDtype 情况下，检查 type 是否在 kind 之前是有意义的
        val = getattr(dtype, attr, None)
        # 如果找到与 _TYPE_MAP 中的值相匹配的 val，则返回映射后的结果
        if val in _TYPE_MAP:
            return _TYPE_MAP[val]
    # 如果未找到匹配项，则返回 None
    return None


def infer_dtype(value: object, skipna: bool = True) -> str:
    """
    Return a string label of the type of a scalar or list-like of values.

    Parameters
    ----------
    value : scalar, list, ndarray, or pandas type
        The input data to infer the dtype.
    skipna : bool, default True
        Ignore NaN values when inferring the type.

    Returns
    -------
    str
        Describing the common type of the input data.
    Results can include:

    - string
    - bytes
    - floating
    - integer
    - mixed-integer
    - mixed-integer-float
    - decimal
    - complex
    - categorical
    - boolean
    - datetime64
    - datetime
    - date
    - timedelta64
    - timedelta
    - time
    - period
    - mixed
    - unknown-array

    Raises
    ------
    TypeError
        If ndarray-like but cannot infer the dtype

    See Also
    --------
    api.types.is_scalar : Check if the input is a scalar.
    api.types.is_list_like : Check if the input is list-like.
    api.types.is_integer : Check if the input is an integer.
    api.types.is_float : Check if the input is a float.
    api.types.is_bool : Check if the input is a boolean.

    Notes
    -----
    - 'mixed' is the catchall for anything that is not otherwise
      specialized
    - 'mixed-integer-float' are floats and integers
    - 'mixed-integer' are integers mixed with non-integers
    - 'unknown-array' is the catchall for something that *is* an array (has
      a dtype attribute), but has a dtype unknown to pandas (e.g. external
      extension array)

    Examples
    --------
    >>> from pandas.api.types import infer_dtype
    >>> infer_dtype(['foo', 'bar'])
    'string'

    >>> infer_dtype(['a', np.nan, 'b'], skipna=True)
    'string'

    >>> infer_dtype(['a', np.nan, 'b'], skipna=False)
    'mixed'

    >>> infer_dtype([b'foo', b'bar'])
    'bytes'

    >>> infer_dtype([1, 2, 3])
    'integer'

    >>> infer_dtype([1, 2, 3.5])
    'mixed-integer-float'

    >>> infer_dtype([1.0, 2.0, 3.5])
    'floating'

    >>> infer_dtype(['a', 1])
    'mixed-integer'

    >>> from decimal import Decimal
    >>> infer_dtype([Decimal(1), Decimal(2.0)])
    'decimal'

    >>> infer_dtype([True, False])
    'boolean'

    >>> infer_dtype([True, False, np.nan])
    'boolean'

    >>> infer_dtype([pd.Timestamp('20130101')])
    'datetime'

    >>> import datetime
    >>> infer_dtype([datetime.date(2013, 1, 1)])
    'date'

    >>> infer_dtype([np.datetime64('2013-01-01')])
    'datetime64'

    >>> infer_dtype([datetime.timedelta(0, 1, 1)])
    'timedelta'
    """
    pass  # 函数体未实现，因此使用 pass 保持其结构完整
    # 定义 Cython 局部变量：循环索引 i 和元素个数 n
    cdef:
        Py_ssize_t i, n
        # 用于存储当前值的对象 val
        object val
        # 存储数组的 ndarray 对象 values
        ndarray values
        # 标记是否已经看到过 pd.NaT 值的布尔变量 seen_pdnat，默认为 False
        bint seen_pdnat = False
        # 标记是否已经看到过有效值的布尔变量 seen_val，默认为 False
        bint seen_val = False
        # 平坦迭代器对象 it，用于遍历 ndarray
        flatiter it

    # 如果 value 是数组类型，则直接将 values 设置为 value
    if util.is_array(value):
        values = value
    # 如果 value 类型具有 inferred_type 属性且 skipna 参数为 False，则返回 inferred_type 属性的值
    elif hasattr(type(value), "inferred_type") and skipna is False:
        # 索引类型，如果可能则使用缓存的属性，否则填充缓存
        return value.inferred_type
    # 如果 value 具有 dtype 属性
    elif hasattr(value, "dtype"):
        # 尝试推断 dtype 类型，如果成功则返回推断值
        inferred = _try_infer_map(value.dtype)
        if inferred is not None:
            return inferred
        # 如果 value.dtype 不符合 PyArray_DescrCheck 的要求，则返回 "unknown-array"
        elif not cnp.PyArray_DescrCheck(value.dtype):
            return "unknown-array"
        # 将 Series/Index 展开为 ndarray
        values = np.asarray(value)
    else:
        # 如果 value 不是列表，则转换为列表
        if not isinstance(value, list):
            value = list(value)
        # 如果 value 为空，则返回 "empty"
        if not value:
            return "empty"

        # 导入构建对象数组的函数
        from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
        # 使用 value 构建对象数组 values
        values = construct_1d_object_array_from_listlike(value)

    # 尝试推断 values 的 dtype 类型
    inferred = _try_infer_map(values.dtype)
    if inferred is not None:
        # 如果推断成功则返回推断值
        return inferred

    # 如果 values 的类型编号不是 NPY_OBJECT（即不是对象类型）
    if values.descr.type_num != NPY_OBJECT:
        # 转换 values 为对象类型 ndarray
        values = values.astype(object)

    # 获取 values 的元素个数
    n = cnp.PyArray_SIZE(values)
    # 如果 values 为空则返回 "empty"
    if n == 0:
        return "empty"

    # 迭代直到找到第一个有效值，用于决定调用哪个 is_foo_array 函数
    it = PyArray_IterNew(values)
    for i in range(n):
        # 使用 PyArray_GETITEM 和 PyArray_ITER_NEXT 快速获取值 val
        val = PyArray_GETITEM(values, PyArray_ITER_DATA(it))
        PyArray_ITER_NEXT(it)

        # 不使用 checknull 来保持 np.datetime64('nat') 和 np.timedelta64('nat') 的原样
        # 如果 val 为 None 或者是 NaN 或者是 C_NA（特殊值），则跳过
        if val is None or util.is_nan(val) or val is C_NA:
            pass
        # 如果 val 是 NaT（pandas 中的缺失时间戳），则标记 seen_pdnat 为 True
        elif val is NaT:
            seen_pdnat = True
        else:
            # 否则标记 seen_val 为 True 并且跳出循环
            seen_val = True
            break

    # 如果所有值都是 nan/NaT，则返回 "datetime"
    if seen_val is False and seen_pdnat is True:
        return "datetime"
        # float/object nan is handled in latter logic
    # 如果所有值都是 nan/NaT 且 skipna 参数为 True，则返回 "empty"
    if seen_val is False and skipna:
        return "empty"

    # 如果 val 是 datetime64 类型，并且 values 是 datetime64 数组，则返回 "datetime64"
    if cnp.is_datetime64_object(val):
        if is_datetime64_array(values, skipna=skipna):
            return "datetime64"

    # 如果 val 是 timedelta 类型，并且 values 是 timedelta/timedelta64 数组，则返回 "timedelta"
    elif is_timedelta(val):
        if is_timedelta_or_timedelta64_array(values, skipna=skipna):
            return "timedelta"
    # 如果值是整数对象
    elif util.is_integer_object(val):
        # 在这里顺序很重要；这个检查必须在 is_timedelta 检查之后进行，
        # 否则 numpy 的 timedelta64 对象会通过这里

        # 如果值是整数数组
        if is_integer_array(values, skipna=skipna):
            return "integer"
        # 如果值是混合整数和浮点数数组
        elif is_integer_float_array(values, skipna=skipna):
            # 如果值是整数和缺失值混合数组
            if is_integer_na_array(values, skipna=skipna):
                return "integer-na"
            else:
                return "mixed-integer-float"
        return "mixed-integer"

    # 如果值是 Python 的 datetime 对象
    elif PyDateTime_Check(val):
        # 如果值是日期时间数组
        if is_datetime_array(values, skipna=skipna):
            return "datetime"
        # 如果值是日期数组
        elif is_date_array(values, skipna=skipna):
            return "date"

    # 如果值是 Python 的 date 对象
    elif PyDate_Check(val):
        # 如果值是日期数组
        if is_date_array(values, skipna=skipna):
            return "date"

    # 如果值是 Python 的 time 对象
    elif PyTime_Check(val):
        # 如果值是时间数组
        if is_time_array(values, skipna=skipna):
            return "time"

    # 如果值是 decimal.Decimal 对象
    elif is_decimal(val):
        # 如果值是 decimal 数组
        if is_decimal_array(values, skipna=skipna):
            return "decimal"

    # 如果值是复数对象
    elif util.is_complex_object(val):
        # 如果值是复数数组
        if is_complex_array(values):
            return "complex"

    # 如果值是浮点数对象
    elif util.is_float_object(val):
        # 如果值是浮点数数组
        if is_float_array(values):
            return "floating"
        # 如果值是混合整数和浮点数数组
        elif is_integer_float_array(values, skipna=skipna):
            # 如果值是整数和缺失值混合数组
            if is_integer_na_array(values, skipna=skipna):
                return "integer-na"
            else:
                return "mixed-integer-float"

    # 如果值是布尔对象
    elif util.is_bool_object(val):
        # 如果值是布尔数组
        if is_bool_array(values, skipna=skipna):
            return "boolean"

    # 如果值是字符串对象
    elif isinstance(val, str):
        # 如果值是字符串数组
        if is_string_array(values, skipna=skipna):
            return "string"

    # 如果值是 bytes 对象
    elif isinstance(val, bytes):
        # 如果值是 bytes 数组
        if is_bytes_array(values, skipna=skipna):
            return "bytes"

    # 如果值是 period.Period 对象
    elif is_period_object(val):
        # 如果值是 period 数组
        if is_period_array(values, skipna=skipna):
            return "period"

    # 如果值是 Interval 对象
    elif isinstance(val, Interval):
        # 如果值是 Interval 数组
        if is_interval_array(values):
            return "interval"

    # 重置数组迭代器到起始位置
    cnp.PyArray_ITER_RESET(it)
    # 遍历数组
    for i in range(n):
        # 获取数组中的元素值
        val = PyArray_GETITEM(values, PyArray_ITER_DATA(it))
        # 移动到下一个数组元素
        PyArray_ITER_NEXT(it)

        # 如果值是整数对象，则返回混合整数类型
        if util.is_integer_object(val):
            return "mixed-integer"

    # 如果没有匹配的类型，则返回混合类型
    return "mixed"
# 定义一个函数，用于检查对象是否为 timedelta 类型
cdef bint is_timedelta(object o):
    return PyDelta_Check(o) or cnp.is_timedelta64_object(o)

# 定义一个 Cython 类 Validator
@cython.internal
cdef class Validator:

    cdef:
        Py_ssize_t n
        cnp.dtype dtype
        bint skipna

    # 初始化方法，接受参数 n, dtype 和 skipna
    def __cinit__(self, Py_ssize_t n, cnp.dtype dtype=np.dtype(np.object_),
                  bint skipna=False):
        self.n = n
        self.dtype = dtype
        self.skipna = skipna

    # 验证方法，接受 ndarray 类型的 values 参数
    cdef bint validate(self, ndarray values) except -1:
        if not self.n:
            return False

        if self.is_array_typed():
            # 如果 ndarray 已经是所需的 dtype 类型，则返回 True
            return True
        elif self.dtype.type_num == NPY_OBJECT:
            if self.skipna:
                return self._validate_skipna(values)
            else:
                return self._validate(values)
        else:
            return False

    # 内部方法，用于验证 ndarray 中的值
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef bint _validate(self, ndarray values) except -1:
        cdef:
            Py_ssize_t i
            Py_ssize_t n = values.size
            flatiter it = PyArray_IterNew(values)

        for i in range(n):
            # 使用 PyArray_GETITEM 和 PyArray_ITER_NEXT 更快地获取值
            val = PyArray_GETITEM(values, PyArray_ITER_DATA(it))
            PyArray_ITER_NEXT(it)
            if not self.is_valid(val):
                return False

        return True

    # 内部方法，用于跳过缺失值后验证 ndarray 中的值
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef bint _validate_skipna(self, ndarray values) except -1:
        cdef:
            Py_ssize_t i
            Py_ssize_t n = values.size
            flatiter it = PyArray_IterNew(values)

        for i in range(n):
            # 使用 PyArray_GETITEM 和 PyArray_ITER_NEXT 更快地获取值
            val = PyArray_GETITEM(values, PyArray_ITER_DATA(it))
            PyArray_ITER_NEXT(it)
            if not self.is_valid_skipna(val):
                return False

        return True

    # 内部方法，用于检查值是否有效
    cdef bint is_valid(self, object value) except -1:
        return self.is_value_typed(value)

    # 内部方法，用于跳过缺失值后检查值是否有效
    cdef bint is_valid_skipna(self, object value) except -1:
        return self.is_valid(value) or self.is_valid_null(value)

    # 内部方法，用于检查值是否为指定类型
    cdef bint is_value_typed(self, object value) except -1:
        raise NotImplementedError(f"{type(self).__name__} child class "
                                  "must define is_value_typed")

    # 内部方法，用于检查值是否为缺失值
    cdef bint is_valid_null(self, object value) except -1:
        return value is None or value is C_NA or util.is_nan(value)
        # TODO: include decimal NA?

    # 内部方法，用于检查 ndarray 是否为指定类型
    cdef bint is_array_typed(self) except -1:
        return False


# 定义一个 BoolValidator 类，继承自 Validator 类
@cython.internal
cdef class BoolValidator(Validator):
    # 重写父类方法，用于检查值是否为布尔类型
    cdef bint is_value_typed(self, object value) except -1:
        return util.is_bool_object(value)

    # 重写父类方法，用于检查 ndarray 是否为布尔类型
    cdef bint is_array_typed(self) except -1:
        return cnp.PyDataType_ISBOOL(self.dtype)
# 定义一个 Cython 函数，用于检查给定的 ndarray 是否为布尔类型数组
cpdef bint is_bool_array(ndarray values, bint skipna=False):
    # 创建 BoolValidator 实例来验证数组的布尔值
    cdef:
        BoolValidator validator = BoolValidator(len(values),
                                                values.dtype,
                                                skipna=skipna)
    # 调用 validator 的 validate 方法来验证数组是否为布尔类型
    return validator.validate(values)


# 声明一个 Cython 内部类 IntegerValidator，继承自 Validator
@cython.internal
cdef class IntegerValidator(Validator):
    
    # 定义一个方法，用于检查单个值是否为整数类型
    cdef bint is_value_typed(self, object value) except -1:
        return util.is_integer_object(value)

    # 定义一个方法，用于检查数组是否为整数类型
    cdef bint is_array_typed(self) except -1:
        return cnp.PyDataType_ISINTEGER(self.dtype)


# 注意：此函数仅用于测试目的在 Python 中暴露
# 定义一个 Cython 函数，用于检查给定的 ndarray 是否为整数类型数组
cpdef bint is_integer_array(ndarray values, bint skipna=True):
    # 创建 IntegerValidator 实例来验证数组的整数值
    cdef:
        IntegerValidator validator = IntegerValidator(len(values),
                                                      values.dtype,
                                                      skipna=skipna)
    # 调用 validator 的 validate 方法来验证数组是否为整数类型
    return validator.validate(values)


# 声明一个 Cython 内部类 IntegerNaValidator，继承自 Validator
@cython.internal
cdef class IntegerNaValidator(Validator):
    
    # 定义一个方法，用于检查单个值是否为整数或 NaN 类型
    cdef bint is_value_typed(self, object value) except -1:
        return (util.is_integer_object(value)
                or (util.is_nan(value) and util.is_float_object(value)))

# 声明一个 Cython 函数，用于检查给定的 ndarray 是否为整数或 NaN 类型数组
cdef bint is_integer_na_array(ndarray values, bint skipna=True):
    # 创建 IntegerNaValidator 实例来验证数组的整数或 NaN 值
    cdef:
        IntegerNaValidator validator = IntegerNaValidator(len(values),
                                                          values.dtype, skipna=skipna)
    # 调用 validator 的 validate 方法来验证数组是否为整数或 NaN 类型
    return validator.validate(values)


# 声明一个 Cython 内部类 IntegerFloatValidator，继承自 Validator
@cython.internal
cdef class IntegerFloatValidator(Validator):
    
    # 定义一个方法，用于检查单个值是否为整数或浮点数类型
    cdef bint is_value_typed(self, object value) except -1:
        return util.is_integer_object(value) or util.is_float_object(value)

    # 定义一个方法，用于检查数组是否为整数或浮点数类型
    cdef bint is_array_typed(self) except -1:
        return cnp.PyDataType_ISINTEGER(self.dtype)


# 声明一个 Cython 函数，用于检查给定的 ndarray 是否为整数或浮点数类型数组
cdef bint is_integer_float_array(ndarray values, bint skipna=True):
    # 创建 IntegerFloatValidator 实例来验证数组的整数或浮点数值
    cdef:
        IntegerFloatValidator validator = IntegerFloatValidator(len(values),
                                                                values.dtype,
                                                                skipna=skipna)
    # 调用 validator 的 validate 方法来验证数组是否为整数或浮点数类型
    return validator.validate(values)


# 声明一个 Cython 内部类 FloatValidator，继承自 Validator
@cython.internal
cdef class FloatValidator(Validator):
    
    # 定义一个方法，用于检查单个值是否为浮点数类型
    cdef bint is_value_typed(self, object value) except -1:
        return util.is_float_object(value)

    # 定义一个方法，用于检查数组是否为浮点数类型
    cdef bint is_array_typed(self) except -1:
        return cnp.PyDataType_ISFLOAT(self.dtype)


# 注意：此函数仅用于测试目的在 Python 中暴露
# 定义一个 Cython 函数，用于检查给定的 ndarray 是否为浮点数类型数组
cpdef bint is_float_array(ndarray values):
    # 创建 FloatValidator 实例来验证数组的浮点数值
    cdef:
        FloatValidator validator = FloatValidator(len(values), values.dtype)
    # 调用 validator 的 validate 方法来验证数组是否为浮点数类型
    return validator.validate(values)


# 声明一个 Cython 内部类 ComplexValidator，继承自 Validator
@cython.internal
cdef class ComplexValidator(Validator):
    
    # 定义一个方法，用于检查单个值是否为复数类型
    cdef bint is_value_typed(self, object value) except -1:
        return (
            util.is_complex_object(value)
            or (util.is_float_object(value) and is_nan(value))
        )

    # 定义一个方法，用于检查数组是否为复数类型
    cdef bint is_array_typed(self) except -1:
        return cnp.PyDataType_ISCOMPLEX(self.dtype)


# 声明一个 Cython 函数，用于检查给定的 ndarray 是否为复数类型数组
cdef bint is_complex_array(ndarray values):
    cdef:
        ComplexValidator validator = ComplexValidator(len(values), values.dtype)
    # 创建一个复杂验证器对象，使用给定的值数量和数据类型初始化
    return validator.validate(values)
    # 使用创建的验证器对象对给定的值进行验证，并返回验证结果
@cython.internal
cdef class DecimalValidator(Validator):
    # 定义 DecimalValidator 类，继承自 Validator
    cdef bint is_value_typed(self, object value) except -1:
        # 检查值是否为 decimal 类型的方法，返回布尔值或者 -1（异常情况）
        return is_decimal(value)


cdef bint is_decimal_array(ndarray values, bint skipna=False):
    # 检查给定的 ndarray 是否包含 decimal 类型的数组
    cdef:
        # 创建 DecimalValidator 实例，用于验证数组
        DecimalValidator validator = DecimalValidator(
            len(values), values.dtype, skipna=skipna
        )
    return validator.validate(values)


@cython.internal
cdef class StringValidator(Validator):
    # 定义 StringValidator 类，继承自 Validator
    cdef bint is_value_typed(self, object value) except -1:
        # 检查值是否为字符串类型的方法，返回布尔值或者 -1（异常情况）
        return isinstance(value, str)

    cdef bint is_array_typed(self) except -1:
        # 检查数组是否为字符串类型的方法，返回布尔值或者 -1（异常情况）
        return self.dtype.type_num == cnp.NPY_UNICODE


cpdef bint is_string_array(ndarray values, bint skipna=False):
    # 检查给定的 ndarray 是否包含字符串类型的数组
    cdef:
        # 创建 StringValidator 实例，用于验证数组
        StringValidator validator = StringValidator(len(values),
                                                    values.dtype,
                                                    skipna=skipna)
    return validator.validate(values)


@cython.internal
cdef class BytesValidator(Validator):
    # 定义 BytesValidator 类，继承自 Validator
    cdef bint is_value_typed(self, object value) except -1:
        # 检查值是否为字节类型的方法，返回布尔值或者 -1（异常情况）
        return isinstance(value, bytes)

    cdef bint is_array_typed(self) except -1:
        # 检查数组是否为字节类型的方法，返回布尔值或者 -1（异常情况）
        return self.dtype.type_num == cnp.NPY_STRING


cdef bint is_bytes_array(ndarray values, bint skipna=False):
    # 检查给定的 ndarray 是否包含字节类型的数组
    cdef:
        # 创建 BytesValidator 实例，用于验证数组
        BytesValidator validator = BytesValidator(len(values), values.dtype,
                                                  skipna=skipna)
    return validator.validate(values)


@cython.internal
cdef class TemporalValidator(Validator):
    # 定义 TemporalValidator 类，继承自 Validator
    cdef:
        # 定义 all_generic_na 变量，用于标记是否存在通用的空值
        bint all_generic_na

    def __cinit__(self, Py_ssize_t n, cnp.dtype dtype=np.dtype(np.object_),
                  bint skipna=False):
        # 初始化方法，设置对象的属性值
        self.n = n
        self.dtype = dtype
        self.skipna = skipna
        self.all_generic_na = True

    cdef bint is_valid(self, object value) except -1:
        # 判断给定的值是否有效的方法，考虑空值情况
        return self.is_value_typed(value) or self.is_valid_null(value)

    cdef bint is_valid_null(self, object value) except -1:
        # 抽象方法，子类必须实现，用于判断是否为有效的空值
        raise NotImplementedError(f"{type(self).__name__} child class "
                                  "must define is_valid_null")

    cdef bint is_valid_skipna(self, object value) except -1:
        # 判断给定的值是否有效（考虑跳过空值情况）
        cdef:
            bint is_typed_null = self.is_valid_null(value)
            bint is_generic_null = value is None or util.is_nan(value)
        if not is_generic_null:
            self.all_generic_na = False
        return self.is_value_typed(value) or is_typed_null or is_generic_null

    cdef bint _validate_skipna(self, ndarray values) except -1:
        """
        If we _only_ saw non-dtype-specific NA values, even if they are valid
        for this dtype, we do not infer this dtype.
        """
        # 验证跳过空值的方法，特别考虑只有非 dtype 特定 NA 值的情况
        return Validator._validate_skipna(self, values) and not self.all_generic_na


@cython.internal
cdef class DatetimeValidator(TemporalValidator):
    # 定义 DatetimeValidator 类，继承自 TemporalValidator
    cdef bint is_value_typed(self, object value) except -1:
        # 检查值是否为日期时间类型的方法
        return PyDateTime_Check(value)
    # 定义一个 C 语言级别的函数，检查给定的值是否为有效的空值（Null）
    cdef bint is_valid_null(self, object value) except -1:
        # 调用 is_null_datetime64 函数来检查值是否为 datetime64 类型的空值
        return is_null_datetime64(value)
# 根据传入的 ndarray values 和可选参数 skipna（默认为 True），检查是否为日期时间数组
cpdef bint is_datetime_array(ndarray values, bint skipna=True):
    # 创建 DatetimeValidator 对象，用于验证日期时间数组的有效性
    cdef:
        DatetimeValidator validator = DatetimeValidator(len(values),
                                                        skipna=skipna)
    # 调用 validator 对象的 validate 方法，验证数值数组是否符合日期时间格式
    return validator.validate(values)


@cython.internal
# 定义一个 Cython 内部类 Datetime64Validator，继承自 DatetimeValidator
cdef class Datetime64Validator(DatetimeValidator):
    # 定义 is_value_typed 方法，检查对象是否为 datetime64 类型的数据
    cdef bint is_value_typed(self, object value) except -1:
        return cnp.is_datetime64_object(value)


# 注意：仅供测试时使用，暴露给 Python 环境
# 根据传入的 ndarray values 和可选参数 skipna（默认为 True），检查是否为 datetime64 类型的数组
cpdef bint is_datetime64_array(ndarray values, bint skipna=True):
    # 创建 Datetime64Validator 对象，用于验证 datetime64 数组的有效性
    cdef:
        Datetime64Validator validator = Datetime64Validator(len(values),
                                                            skipna=skipna)
    # 调用 validator 对象的 validate 方法，验证数值数组是否符合 datetime64 格式
    return validator.validate(values)


@cython.internal
# 定义一个 Cython 内部类 AnyDatetimeValidator，继承自 DatetimeValidator
cdef class AnyDatetimeValidator(DatetimeValidator):
    # 定义 is_value_typed 方法，检查对象是否为 datetime64 或者是无时区的 datetime 类型
    cdef bint is_value_typed(self, object value) except -1:
        return cnp.is_datetime64_object(value) or (
            PyDateTime_Check(value) and value.tzinfo is None
        )


# 根据传入的 ndarray values 和可选参数 skipna（默认为 True），检查是否为 datetime 或 datetime64 类型的数组
cdef bint is_datetime_or_datetime64_array(ndarray values, bint skipna=True):
    # 创建 AnyDatetimeValidator 对象，用于验证是否为 datetime 或 datetime64 类型的数组
    cdef:
        AnyDatetimeValidator validator = AnyDatetimeValidator(len(values),
                                                              skipna=skipna)
    # 调用 validator 对象的 validate 方法，验证数值数组是否符合 datetime 或 datetime64 格式
    return validator.validate(values)


# 注意：仅供测试时使用，暴露给 Python 环境
# 检查给定的 ndarray values 是否为单一时区的 datetime 数组
def is_datetime_with_singletz_array(values: ndarray) -> bool:
    """
    Check values have the same tzinfo attribute.
    Doesn't check values are datetime-like types.
    """
    # 定义变量 i、j 和 n，分别表示循环计数、数组长度和当前处理的值
    cdef:
        Py_ssize_t i = 0, j, n = len(values)
        object base_val, base_tz, val, tz

    # 若数组长度为 0，则直接返回 False
    if n == 0:
        return False

    # 遍历数组找到第一个非 NaT、非空且非 NaN 的值，并获取其 tzinfo 属性作为基准时区
    for i in range(n):
        base_val = values[i]
        if base_val is not NaT and base_val is not None and not util.is_nan(base_val):
            base_tz = getattr(base_val, "tzinfo", None)
            break

    # 继续遍历数组，比较每个值的 tzinfo 属性与基准时区是否一致
    for j in range(i, n):
        # 获取当前值的 tzinfo 属性，并与基准时区比较
        # 如果遇到 NaT，则跳过，因为 NaT 可以与有时区的 datetime 共存
        val = values[j]
        if val is not NaT and val is not None and not util.is_nan(val):
            tz = getattr(val, "tzinfo", None)
            if not tz_compare(base_tz, tz):
                return False

    # 注意：只有当出现了有时区的 datetime 时，才会调用此函数，因此 base_tz 在此处始终已设置
    return True


@cython.internal
# 定义一个 Cython 内部类 TimedeltaValidator，继承自 TemporalValidator
cdef class TimedeltaValidator(TemporalValidator):
    # 定义 is_value_typed 方法，检查对象是否为 timedelta 类型的数据
    cdef bint is_value_typed(self, object value) except -1:
        return PyDelta_Check(value)

    # 定义 is_valid_null 方法，检查对象是否为 null timedelta64 类型的数据
    cdef bint is_valid_null(self, object value) except -1:
        return is_null_timedelta64(value)


@cython.internal
# 定义一个 Cython 内部类 AnyTimedeltaValidator，继承自 TimedeltaValidator
cdef class AnyTimedeltaValidator(TimedeltaValidator):
    # 定义 is_value_typed 方法，检查对象是否为 timedelta 类型的数据
    cdef bint is_value_typed(self, object value) except -1:
        return is_timedelta(value)


# 注意：仅供测试时使用，暴露给 Python 环境
# 根据传入的 ndarray values 和可选参数 skipna（默认为 True），检查是否为 timedelta 或 timedelta64 类型的数组
cpdef bint is_timedelta_or_timedelta64_array(ndarray values, bint skipna=True):
    """
    Placeholder function for checking if the ndarray values contain timedelta or timedelta64 types.
    Actual implementation is omitted in the provided code snippet.
    """
    """
    使用时间间隔和/或自然值/空值进行推断。
    """
    cdef:
        # 创建一个名为 `validator` 的 `AnyTimedeltaValidator` 对象，
        # 该对象用于验证给定的 `values`，长度为 `values` 的长度，并可以选择跳过 NaN 值。
        AnyTimedeltaValidator validator = AnyTimedeltaValidator(len(values),
                                                                skipna=skipna)
    # 返回通过 `validator` 对象验证后的结果
    return validator.validate(values)
# 定义一个 Cython 内部类 DateValidator，继承自 Validator
@cython.internal
cdef class DateValidator(Validator):
    
    # 定义 is_value_typed 方法，检查传入的值是否为日期类型
    cdef bint is_value_typed(self, object value) except -1:
        return PyDate_Check(value)


# 注意：此函数仅供测试时使用
# 定义一个 Cython 公开函数 is_date_array，用于检查 ndarray 是否包含日期类型的数组
cpdef bint is_date_array(ndarray values, bint skipna=False):
    cdef:
        # 创建一个 DateValidator 实例，用于验证数组的日期值
        DateValidator validator = DateValidator(len(values), skipna=skipna)
    return validator.validate(values)


# 定义一个 Cython 内部类 TimeValidator，继承自 Validator
@cython.internal
cdef class TimeValidator(Validator):
    
    # 定义 is_value_typed 方法，检查传入的值是否为时间类型
    cdef bint is_value_typed(self, object value) except -1:
        return PyTime_Check(value)


# 注意：此函数仅供测试时使用
# 定义一个 Cython 公开函数 is_time_array，用于检查 ndarray 是否包含时间类型的数组
cpdef bint is_time_array(ndarray values, bint skipna=False):
    cdef:
        # 创建一个 TimeValidator 实例，用于验证数组的时间值
        TimeValidator validator = TimeValidator(len(values), skipna=skipna)
    return validator.validate(values)


# FIXME: 实际使用 skipna 参数
# 定义一个 Cython 函数 is_period_array，用于检查 ndarray 是否包含周期对象（Period）的数组
cdef bint is_period_array(ndarray values, bint skipna=True):
    """
    Is this an ndarray of Period objects (or NaT) with a single `freq`?
    """
    # values 应为 object-dtype，但 ndarray[object] 假定为1D，而这里可能是2D。
    cdef:
        Py_ssize_t i, N = values.size
        int dtype_code = -10000  # 即 c_FreqGroup.FR_UND
        object val
        flatiter it

    if N == 0:
        return False

    # 创建一个 PyArray_IterNew 迭代器
    it = PyArray_IterNew(values)
    for i in range(N):
        # 使用 PyArray_GETITEM 和 PyArray_ITER_NEXT 更快地获取值，等效于 `val = values[i]`
        val = PyArray_GETITEM(values, PyArray_ITER_DATA(it))
        PyArray_ITER_NEXT(it)

        # 检查是否为 Period 对象
        if is_period_object(val):
            if dtype_code == -10000:
                dtype_code = val._dtype._dtype_code
            elif dtype_code != val._dtype._dtype_code:
                # 频率不匹配
                return False
        elif checknull_with_nat(val):
            pass
        else:
            # 不是 Period 或 NaT 类似的对象
            return False

    if dtype_code == -10000:
        # 全部为 NaT，没有实际的 Period 对象
        return False
    return True


# 注意：此函数仅供测试时使用
# 定义一个 Cython 公开函数 is_interval_array，用于检查 ndarray 是否包含区间（Interval）对象的数组
cpdef bint is_interval_array(ndarray values):
    """
    Is this an ndarray of Interval (or np.nan) with a single dtype?
    """
    cdef:
        Py_ssize_t i, n = len(values)
        str closed = None
        bint numeric = False
        bint dt64 = False
        bint td64 = False
        object val

    if len(values) == 0:
        return False
    # 循环遍历范围为 0 到 n-1 的索引
    for i in range(n):
        # 获取第 i 个元素的值
        val = values[i]

        # 如果该值是 Interval 类型的实例
        if isinstance(val, Interval):
            # 如果 closed 尚未赋值，则将其设为 val 的 closed 属性值
            if closed is None:
                closed = val.closed
                # 检查 val.left 是否为浮点数对象或整数对象
                numeric = (
                    util.is_float_object(val.left)
                    or util.is_integer_object(val.left)
                )
                # 检查 val.left 是否为 timedelta64 类型
                td64 = is_timedelta(val.left)
                # 检查 val.left 是否为 datetime64 类型
                dt64 = PyDateTime_Check(val.left)
            # 否则，如果 val 的 closed 属性与已有的 closed 值不一致，则返回 False
            elif val.closed != closed:
                # 不匹配的 closed 属性
                return False
            # 如果 numeric 为 True，则验证 val.left 是否为浮点数对象或整数对象
            elif numeric:
                if not (
                    util.is_float_object(val.left)
                    or util.is_integer_object(val.left)
                ):
                    # 即 datetime64 或 timedelta64
                    return False
            # 如果 td64 为 True，则验证 val.left 是否为 timedelta64 类型
            elif td64:
                if not is_timedelta(val.left):
                    return False
            # 如果 dt64 为 True，则验证 val.left 是否为 datetime64 类型
            elif dt64:
                if not PyDateTime_Check(val.left):
                    return False
            else:
                # 如果以上条件都不满足，则引发 ValueError 异常
                raise ValueError(val)
        # 如果 val 是 NaN 或 None，则跳过当前循环
        elif util.is_nan(val) or val is None:
            pass
        else:
            # 如果 val 不是 Interval 类型的实例，且不是 NaN 或 None，则返回 False
            return False

    # 如果 closed 仍为 None，则表示所有的值都是 NaN，没有实际的 Intervals 存在，返回 False
    if closed is None:
        # 我们看到全是 NaN，没有实际的 Intervals
        return False
    # 如果以上条件都不满足，则返回 True
    return True
# 禁用 Cython 对数组边界检查和负索引的检查，优化性能
@cython.boundscheck(False)
@cython.wraparound(False)
def maybe_convert_numeric(
    # 尝试将对象数组转换为数值数组
    ndarray[object, ndim=1] values,
    # 应被解释为 NaN 的值的集合
    set na_values,
    # 是否将空数组对象视为 NaN，默认为 True
    bint convert_empty=True,
    # 是否强制将元素转换为数值，失败时将元素设为 NaN，默认为 False
    bint coerce_numeric=False,
    # 是否将转换后的值返回为带遮罩的可空数组，默认为 False
    bint convert_to_masked_nullable=False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Convert object array to a numeric array if possible.

    Parameters
    ----------
    values : ndarray[object]
        Array of object elements to convert.
    na_values : set
        Set of values that should be interpreted as NaN.
    convert_empty : bool, default True
        If an empty array-like object is encountered, whether to interpret
        that element as NaN or not. If set to False, a ValueError will be
        raised if such an element is encountered and 'coerce_numeric' is False.
    coerce_numeric : bool, default False
        If initial attempts to convert to numeric have failed, whether to
        force conversion to numeric via alternative methods or by setting the
        element to NaN. Otherwise, an Exception will be raised when such an
        element is encountered.

        This boolean also has an impact on how conversion behaves when a
        numeric array has no suitable numerical dtype to return (i.e. uint64,
        int32, uint8). If set to False, the original object array will be
        returned. Otherwise, a ValueError will be raised.
    convert_to_masked_nullable : bool, default False
        Whether to return a mask for the converted values. This also disables
        upcasting for ints with nulls to float64.
    Returns
    -------
    np.ndarray
        Array of converted object values to numerical ones.

    Optional[np.ndarray]
        If convert_to_masked_nullable is True,
        returns a boolean mask for the converted values, otherwise returns None.
    """
    if len(values) == 0:
        return (np.array([], dtype="i8"), None)

    # 通过快速路径尝试转换整数 - 基于第一个值进行转换
    cdef:
        object val = values[0]

    if util.is_integer_object(val):
        try:
            # 尝试将所有值转换为整数类型
            maybe_ints = values.astype("i8")
            # 如果所有元素转换后仍与原始值相等，则返回转换后的数组和 None
            if (maybe_ints == values).all():
                return (maybe_ints, None)
        except (ValueError, OverflowError, TypeError):
            pass

    # 否则，遍历数组进行完整的推断
    # 定义可能是整数的变量
    cdef:
        int maybe_int
        # 定义循环变量 i 和数组长度 n，使用 values 的大小
        Py_ssize_t i, n = values.size
        # 创建一个 Seen 对象，用于跟踪已见的数据类型，使用 coerce_numeric 方法初始化
        Seen seen = Seen(coerce_numeric)
        # 创建一个空的浮点数数组，形状与 values 相同
        ndarray[float64_t, ndim=1] floats = cnp.PyArray_EMPTY(
            1, values.shape, cnp.NPY_FLOAT64, 0
        )
        # 创建一个空的复数数组，形状与 values 相同
        ndarray[complex128_t, ndim=1] complexes = cnp.PyArray_EMPTY(
            1, values.shape, cnp.NPY_COMPLEX128, 0
        )
        # 创建一个空的 64 位整数数组，形状与 values 相同
        ndarray[int64_t, ndim=1] ints = cnp.PyArray_EMPTY(
            1, values.shape, cnp.NPY_INT64, 0
        )
        # 创建一个空的 64 位无符号整数数组，形状与 values 相同
        ndarray[uint64_t, ndim=1] uints = cnp.PyArray_EMPTY(
            1, values.shape, cnp.NPY_UINT64, 0
        )
        # 创建一个空的 8 位无符号整数数组，形状与 values 相同，用作布尔值数组的掩码
        ndarray[uint8_t, ndim=1] bools = cnp.PyArray_EMPTY(
            1, values.shape, cnp.NPY_UINT8, 0
        )
        # 创建一个由 0 组成的 8 位无符号整数数组，长度为 n，用作数据掩码
        ndarray[uint8_t, ndim=1] mask = np.zeros(n, dtype="u1")
        # 定义浮点数变量 fval
        float64_t fval
        # 布尔变量，指示是否允许将整数转换为可空的掩码值
        bint allow_null_in_int = convert_to_masked_nullable

    # 如果 Seen 对象检测到存在 uint64 冲突，返回原始 values 和空值
    if seen.check_uint64_conflict():
        return (values, None)

    # 如果允许在整数中包含空值，并且 Seen 对象表明只见过 float 的空值但未见过整数和布尔值
    # 则标记已见过 float 值
    # 预期可能出现未见过的整数，因此返回浮点数数组
    if allow_null_in_int and seen.null_ and not seen.int_ and not seen.bool_:
        seen.float_ = True

    # 如果 Seen 对象表明存在复数类型数据，返回复数数组和空值
    if seen.complex_:
        return (complexes, None)
    # 如果 Seen 对象表明存在浮点数类型数据
    elif seen.float_:
        # 如果允许空值，并且将整数转换为可空的掩码值
        if seen.null_ and convert_to_masked_nullable:
            return (floats, mask.view(np.bool_))
        # 否则返回浮点数数组和空值
        return (floats, None)
    # 如果 Seen 对象表明存在整数类型数据
    elif seen.int_:
        # 如果允许空值，并且将整数转换为可空的掩码值
        if seen.null_ and convert_to_masked_nullable:
            # 如果见过无符号整数，返回无符号整数数组和对应掩码的布尔值视图
            if seen.uint_:
                return (uints, mask.view(np.bool_))
            # 否则返回整数数组和对应掩码的布尔值视图
            else:
                return (ints, mask.view(np.bool_))
        # 如果见过无符号整数，返回无符号整数数组和空值
        if seen.uint_:
            return (uints, None)
        # 否则返回整数数组和空值
        else:
            return (ints, None)
    # 如果 Seen 对象表明存在布尔类型数据
    elif seen.bool_:
        # 如果允许在整数中包含空值，返回布尔值数组的布尔值视图和对应掩码的布尔值视图
        if allow_null_in_int:
            return (bools.view(np.bool_), mask.view(np.bool_))
        # 否则返回布尔值数组的布尔值视图和空值
        return (bools.view(np.bool_), None)
    # 如果 Seen 对象表明存在无符号整数类型数据
    elif seen.uint_:
        # 返回无符号整数数组和空值
        return (uints, None)
    # 否则返回整数数组和空值
    return (ints, None)
@cython.boundscheck(False)
@cython.wraparound(False)
# 定义一个函数，用于根据条件将对象数组转换为适当的数据类型
def maybe_convert_objects(ndarray[object] objects,
                          *,
                          bint try_float=False,
                          bint safe=False,
                          bint convert_numeric=True,  # 注意：默认值不同！
                          bint convert_to_nullable_dtype=False,
                          bint convert_non_numeric=False,
                          object dtype_if_all_nat=None) -> "ArrayLike":
    """
    Type inference function-- convert object array to proper dtype

    Parameters
    ----------
    objects : ndarray[object]
        Array of object elements to convert.
    try_float : bool, default False
        If an array-like object contains only float or NaN values is
        encountered, whether to convert and return an array of float dtype.
    safe : bool, default False
        Whether to upcast numeric type (e.g. int cast to float). If set to
        True, no upcasting will be performed.
    convert_numeric : bool, default True
        Whether to convert numeric entries.
    convert_to_nullable_dtype : bool, default False
        If an array-like object contains only integer or boolean values (and NaN) is
        encountered, whether to convert and return a Boolean/IntegerArray.
    convert_non_numeric : bool, default False
        Whether to convert datetime, timedelta, period, interval types.
    dtype_if_all_nat : np.dtype, ExtensionDtype, or None, default None
        Dtype to cast to if we have all-NaT.

    Returns
    -------
    np.ndarray or ExtensionArray
        Array of converted object values to more specific dtypes if applicable.
    """
    cdef:
        Py_ssize_t i, n, itemsize_max = 0
        ndarray[float64_t] floats
        ndarray[complex128_t] complexes
        ndarray[int64_t] ints
        ndarray[uint64_t] uints
        ndarray[uint8_t] bools
        ndarray[uint8_t] mask
        Seen seen = Seen()
        object val
        float64_t fnan = NaN

    if dtype_if_all_nat is not None:
        # 在实际情况下，我们不希望在没有 convert_non_numeric 的情况下传递 dtype_if_all_nat，
        # 因此禁止这种情况以避免需要在下面处理它。
        if not convert_non_numeric:
            raise ValueError(
                "Cannot specify 'dtype_if_all_nat' without convert_non_numeric=True"
            )

    n = len(objects)

    # 初始化各种数据类型的空数组，用于存储转换后的值
    floats = cnp.PyArray_EMPTY(1, objects.shape, cnp.NPY_FLOAT64, 0)
    complexes = cnp.PyArray_EMPTY(1, objects.shape, cnp.NPY_COMPLEX128, 0)
    ints = cnp.PyArray_EMPTY(1, objects.shape, cnp.NPY_INT64, 0)
    uints = cnp.PyArray_EMPTY(1, objects.shape, cnp.NPY_UINT64, 0)
    bools = cnp.PyArray_EMPTY(1, objects.shape, cnp.NPY_UINT8, 0)
    mask = np.full(n, False)

    # 尝试强制类型转换日期时间，但必须保持相同的时区
    # 如果 seen.datetimetz_ 为真
    if seen.datetimetz_:
        # 如果 objects 是带有单个时区的 datetime 数组
        if is_datetime_with_singletz_array(objects):
            # 导入 DatetimeIndex 类
            from pandas import DatetimeIndex

            try:
                # 尝试创建 DatetimeIndex 对象
                dti = DatetimeIndex(objects)
            except OutOfBoundsDatetime:
                # 处理 OutOfBoundsDatetime 异常，例如 test_to_datetime_cache_coerce_50_lines_outofbounds
                pass
            else:
                # 将 DatetimeIndex 对象的数据解包为 DatetimeArray
                return dti._data
        # 标记已经处理过对象类型为 object_
        seen.object_ = True

    # 否则，如果 seen.datetime_ 为真
    elif seen.datetime_:
        # 如果 objects 是 datetime 或 datetime64 数组
        if is_datetime_or_datetime64_array(objects):
            # 导入 DatetimeIndex 类
            from pandas import DatetimeIndex

            try:
                # 尝试创建 DatetimeIndex 对象
                dti = DatetimeIndex(objects)
            except OutOfBoundsDatetime:
                # 处理 OutOfBoundsDatetime 异常
                pass
            else:
                # 将 DatetimeIndex 对象的数据解包为 ndarray[datetime64[ns]]
                return dti._data._ndarray
        # 标记已经处理过对象类型为 object_
        seen.object_ = True

    # 否则，如果 seen.timedelta_ 为真
    elif seen.timedelta_:
        # 如果 objects 是 timedelta 或 timedelta64 数组
        if is_timedelta_or_timedelta64_array(objects):
            # 导入 TimedeltaIndex 类
            from pandas import TimedeltaIndex

            try:
                # 尝试创建 TimedeltaIndex 对象
                tdi = TimedeltaIndex(objects)
            except OutOfBoundsTimedelta:
                # 处理 OutOfBoundsTimedelta 异常
                pass
            else:
                # 将 TimedeltaIndex 对象的数据解包为 ndarray[timedelta64[ns]]
                return tdi._data._ndarray
        # 标记已经处理过对象类型为 object_
        seen.object_ = True

    # 否则，如果 seen.period_ 为真
    elif seen.period_:
        # 如果 objects 是 period 数组
        if is_period_array(objects):
            # 导入 PeriodIndex 类
            from pandas import PeriodIndex
            # 创建 PeriodIndex 对象
            pi = PeriodIndex(objects)

            # 将 PeriodIndex 对象的数据解包为 PeriodArray
            return pi._data
        # 标记已经处理过对象类型为 object_
        seen.object_ = True

    # 否则，如果 seen.str_ 为真
    elif seen.str_:
        # 如果正在使用 pyarrow 的 string dtype 并且 objects 是 string 数组（跳过 NaN 值）
        if using_pyarrow_string_dtype() and is_string_array(objects, skipna=True):
            # 导入 StringDtype 类
            from pandas.core.arrays.string_ import StringDtype

            # 创建 storage="pyarrow_numpy" 的 StringDtype 对象
            dtype = StringDtype(storage="pyarrow_numpy")
            # 使用 dtype 构造 array 类型，并从 objects 创建数组
            return dtype.construct_array_type()._from_sequence(objects, dtype=dtype)

        # 否则，如果 convert_to_nullable_dtype 为真且 objects 是 string 数组（跳过 NaN 值）
        elif convert_to_nullable_dtype and is_string_array(objects, skipna=True):
            # 导入 StringDtype 类
            from pandas.core.arrays.string_ import StringDtype

            # 创建默认 StringDtype 对象
            dtype = StringDtype()
            # 使用 dtype 构造 array 类型，并从 objects 创建数组
            return dtype.construct_array_type()._from_sequence(objects, dtype=dtype)

        # 标记已经处理过对象类型为 object_
        seen.object_ = True
    
    # 否则，如果 seen.interval_ 为真
    elif seen.interval_:
        # 如果 objects 是 interval 数组
        if is_interval_array(objects):
            # 导入 IntervalIndex 类
            from pandas import IntervalIndex
            # 创建 IntervalIndex 对象
            ii = IntervalIndex(objects)

            # 将 IntervalIndex 对象的数据解包为 IntervalArray
            return ii._data

        # 标记已经处理过对象类型为 object_
        seen.object_ = True
    # 如果出现了 `seen.nat_`，则执行以下逻辑
    elif seen.nat_:
        # 如果未见过对象、数字或布尔类型，则所有值为 NaT、None 或 nan（至少有一个 NaT）
        # 参见 GH#49340 讨论所需的行为
        dtype = dtype_if_all_nat
        if cnp.PyArray_DescrCheck(dtype):
            # 检查 dtype 是否为 np.dtype 类型的实例
            if dtype.kind not in "mM":
                # 如果 dtype 的种类不在 'mM' 中，抛出 ValueError
                raise ValueError(dtype)
            else:
                # 创建一个空数组 res，用 NPY_NAT 填充
                res = np.empty((<object>objects).shape, dtype=dtype)
                res[:] = NPY_NAT
                return res
        elif dtype is not None:
            # 如果 dtype 不为 None，则根据 dtype 创建相应的数组类型
            cls = dtype.construct_array_type()
            # 从空序列创建对象 obj
            obj = cls._from_sequence([], dtype=dtype)
            # 创建一个全为 -1 的 taker 数组
            taker = -np.ones((<object>objects).shape, dtype=np.intp)
            return obj.take(taker, allow_fill=True)
        else:
            # 如果无法猜测 dtype 类型，则将 seen.object_ 置为 True
            seen.object_ = True
    else:
        # 如果未见过 NaT，则将 seen.object_ 置为 True
        seen.object_ = True

    # 如果不需要转换为数字，则直接返回 objects
    if not convert_numeric:
        # 注意：这里将 "bool" 视为数字。因为 np.array(list_of_items) 会像处理数字一样处理布尔值条目。
        return objects

    # 如果出现了 `seen.bool_`，则执行以下逻辑
    if seen.bool_:
        if convert_to_nullable_dtype and seen.is_bool_or_na:
            # 如果需要转换为可空 dtype，并且出现了布尔值或缺失值
            from pandas.core.arrays import BooleanArray
            return BooleanArray(bools.view(np.bool_), mask)
        elif seen.is_bool:
            # 如果出现了布尔类型，返回布尔数组的视图
            # is_bool 属性排除了所有其他可能性
            return bools.view(np.bool_)
        seen.object_ = True
    # 如果没有发现对象类型为非空
    if not seen.object_:
        # 设定结果为 None
        result = None
        # 如果安全模式未开启
        if not safe:
            # 如果发现了 null 或者 NaN
            if seen.null_ or seen.nan_:
                # 如果发现了复数类型
                if seen.complex_:
                    result = complexes
                # 如果发现了浮点数类型
                elif seen.float_:
                    result = floats
                # 如果发现了整数或者无符号整数类型
                elif seen.int_ or seen.uint_:
                    # 如果需要转换为可空数据类型
                    if convert_to_nullable_dtype:
                        # 在 IntegerArray 中封装
                        if seen.uint_:
                            result = uints
                        else:
                            result = ints
                    else:
                        result = floats
                # 如果发现了 NaN
                elif seen.nan_:
                    result = floats
            # 如果未发现 null 或者 NaN
            else:
                # 如果发现了复数类型
                if seen.complex_:
                    result = complexes
                # 如果发现了浮点数类型
                elif seen.float_:
                    result = floats
                # 如果发现了整数类型
                elif seen.int_:
                    if seen.uint_:
                        result = uints
                    else:
                        result = ints
        # 如果安全模式开启
        else:
            # 不将整数转换为浮点数等
            if seen.null_:
                # 如果发现了复数类型
                if seen.complex_:
                    # 如果不是整数类型
                    if not seen.int_:
                        result = complexes
                # 如果发现了浮点数或者 NaN
                elif seen.float_ or seen.nan_:
                    # 如果不是整数类型
                    if not seen.int_:
                        result = floats
            # 如果未发现 null
            else:
                # 如果发现了复数类型
                if seen.complex_:
                    # 如果不是整数类型
                    if not seen.int_:
                        result = complexes
                # 如果发现了浮点数或者 NaN
                elif seen.float_ or seen.nan_:
                    # 如果不是整数类型
                    if not seen.int_:
                        result = floats
                # 如果发现了整数类型
                elif seen.int_:
                    if seen.uint_:
                        result = uints
                    else:
                        result = ints

        # TODO: 在进行 itemsize 检查之后再执行这些操作？

        # 如果结果为整数数组或者无符号整数数组，并且需要转换为可空数据类型
        if (result is ints or result is uints) and convert_to_nullable_dtype:
            from pandas.core.arrays import IntegerArray

            # 将这些值设为 1，以确保与 IntegerDtype._internal_fill_value 匹配
            result[mask] = 1
            result = IntegerArray(result, mask)
        # 如果结果为浮点数数组，并且需要转换为可空数据类型
        elif result is floats and convert_to_nullable_dtype:
            from pandas.core.arrays import FloatingArray

            # 将这些值设为 1.0，以确保与 FloatingDtype._internal_fill_value 匹配
            result[mask] = 1.0
            result = FloatingArray(result, mask)

        # 如果结果为整数数组、无符号整数数组、浮点数数组或者复数数组
        if result is uints or result is ints or result is floats or result is complexes:
            # 当所有值都为 NumPy 标量时，转换为最大 itemsize 的数据类型
            if itemsize_max > 0 and itemsize_max != result.dtype.itemsize:
                result = result.astype(result.dtype.kind + str(itemsize_max))
            # 返回结果
            return result
        # 如果结果不为空，则返回结果
        elif result is not None:
            return result

    # 返回对象数组，表示未发现匹配结果
    return objects
class _NoDefault(Enum):
    # 将其定义为枚举类型
    # 1) 因为它通过 pickle 正确地往返 (参见 GH#40397)
    # 2) 因为 mypy 不理解单例模式
    no_default = "NO_DEFAULT"

    def __repr__(self) -> str:
        return "<no_default>"


# 注意：no_default 在 pandas.api.extensions 中公开为公共 API
no_default = _NoDefault.no_default  # 表示默认值的标志。
NoDefault = Literal[_NoDefault.no_default]


@cython.boundscheck(False)
@cython.wraparound(False)
def map_infer_mask(
    ndarray arr,
    object f,
    const uint8_t[:] mask,
    *,
    bint convert=True,
    object na_value=no_default,
    cnp.dtype dtype=np.dtype(object)
) -> "ArrayLike":
    """
    替代 np.vectorize，支持 pandas-friendly 的 dtype 推断。

    Parameters
    ----------
    arr : ndarray
        输入的数组
    f : function
        应用于数组元素的函数
    mask : ndarray
        uint8 类型的数组，指示不应用 `f` 的值。
    convert : bool, default True
        是否调用 `maybe_convert_objects` 来转换生成的 ndarray。
    na_value : Any, optional
        用于遮罩值的结果值。默认情况下使用输入值。
    dtype : numpy.dtype
        结果 ndarray 的 numpy 数据类型。

    Returns
    -------
    np.ndarray 或 ExtensionArray
    """
    cdef:
        Py_ssize_t i
        Py_ssize_t n = len(arr)
        object val

        ndarray result = np.empty(n, dtype=dtype)

        flatiter arr_it = PyArray_IterNew(arr)
        flatiter result_it = PyArray_IterNew(result)

    for i in range(n):
        if mask[i]:
            if na_value is no_default:
                val = PyArray_GETITEM(arr, PyArray_ITER_DATA(arr_it))
            else:
                val = na_value
        else:
            val = PyArray_GETITEM(arr, PyArray_ITER_DATA(arr_it))
            val = f(val)

            if cnp.PyArray_IsZeroDim(val):
                # 解开 0 维数组，GH#690
                val = val.item()

        PyArray_SETITEM(result, PyArray_ITER_DATA(result_it), val)

        PyArray_ITER_NEXT(arr_it)
        PyArray_ITER_NEXT(result_it)

    if convert:
        return maybe_convert_objects(result)
    else:
        return result


@cython.boundscheck(False)
@cython.wraparound(False)
def map_infer(
    ndarray arr, object f, *, bint convert=True, bint ignore_na=False
) -> "ArrayLike":
    """
    替代 np.vectorize，支持 pandas-friendly 的 dtype 推断。

    Parameters
    ----------
    arr : ndarray
        输入的数组
    f : function
        应用于数组元素的函数
    convert : bint
        是否转换结果数组
    ignore_na : bint
        如果为 True，则不对 NA 值应用 f 函数

    Returns
    -------
    np.ndarray 或 ExtensionArray
    """
    cdef:
        Py_ssize_t i, n
        ndarray[object] result
        object val

    n = len(arr)
    result = cnp.PyArray_EMPTY(1, arr.shape, cnp.NPY_OBJECT, 0)
    # 遍历数组 arr 中的元素，范围是从 0 到 n-1
    for i in range(n):
        # 如果 ignore_na 为 True 并且 arr[i] 是空值（根据 checknull 函数判断）
        if ignore_na and checknull(arr[i]):
            # 将 result[i] 设置为 arr[i]，并跳过当前循环
            result[i] = arr[i]
            continue
        
        # 计算 arr[i] 的函数 f 的返回值
        val = f(arr[i])

        # 如果返回值 val 是零维数组
        if cnp.PyArray_IsZeroDim(val):
            # 解包零维数组，GH#690 的问题
            val = val.item()

        # 将计算得到的 val 存储到 result[i] 中
        result[i] = val

    # 如果 convert 为 True，则调用 maybe_convert_objects 函数处理 result 后返回
    if convert:
        return maybe_convert_objects(result)
    else:
        # 否则直接返回 result
        return result
# 将列表的列表转换为对象数组

def to_object_array(rows: object, min_width: int = 0) -> ndarray:
    """
    Convert a list of lists into an object array.

    Parameters
    ----------
    rows : 2-d array (N, K)
        List of lists to be converted into an array.
    min_width : int
        Minimum width of the object array. If a list
        in `rows` contains fewer than `width` elements,
        the remaining elements in the corresponding row
        will all be `NaN`.

    Returns
    -------
    np.ndarray[object, ndim=2]
        Object array representation of the input lists.
    """
    cdef:
        Py_ssize_t i, j, n, k, tmp
        ndarray[object, ndim=2] result
        list row

    rows = list(rows)  # 将输入的行转换为列表
    n = len(rows)  # 行数

    k = min_width  # 初始列宽度为给定的最小宽度
    for i in range(n):
        tmp = len(rows[i])  # 当前行的元素个数
        if tmp > k:
            k = tmp  # 更新最大列宽度

    result = np.empty((n, k), dtype=object)  # 创建空的对象数组

    for i in range(n):
        row = list(rows[i])  # 当前行的列表表示

        for j in range(len(row)):
            result[i, j] = row[j]  # 填充对象数组

    return result  # 返回转换后的对象数组


def tuples_to_object_array(ndarray[object] tuples):
    """
    Convert a 2-d array of tuples into an object array.

    Parameters
    ----------
    tuples : np.ndarray[object]
        2-d array of tuples to be converted into an object array.

    Returns
    -------
    np.ndarray[object, ndim=2]
        Object array representation of the input tuples.
    """
    cdef:
        Py_ssize_t i, j, n, k
        ndarray[object, ndim=2] result
        tuple tup

    n = len(tuples)  # 元组数组的行数
    k = len(tuples[0])  # 元组数组的第一行的元素个数
    result = np.empty((n, k), dtype=object)  # 创建空的对象数组
    for i in range(n):
        tup = tuples[i]
        for j in range(k):
            result[i, j] = tup[j]  # 填充对象数组

    return result  # 返回转换后的对象数组


def to_object_array_tuples(rows: object) -> np.ndarray:
    """
    Convert a list of tuples into an object array. Any subclass of
    tuple in `rows` will be casted to tuple.

    Parameters
    ----------
    rows : 2-d array (N, K)
        List of tuples to be converted into an array.

    Returns
    -------
    np.ndarray[object, ndim=2]
        Object array representation of the input tuples.
    """
    cdef:
        Py_ssize_t i, j, n, k, tmp
        ndarray[object, ndim=2] result
        tuple row

    rows = list(rows)  # 将输入的行转换为列表
    n = len(rows)  # 行数

    k = 0  # 初始列宽度为0
    for i in range(n):
        tmp = 1 if checknull(rows[i]) else len(rows[i])  # 检查是否有空值并确定当前行的元素个数
        if tmp > k:
            k = tmp  # 更新最大列宽度

    result = np.empty((n, k), dtype=object)  # 创建空的对象数组

    try:
        for i in range(n):
            row = rows[i]  # 当前行的元组表示
            for j in range(len(row)):
                result[i, j] = row[j]  # 填充对象数组
    except TypeError:
        # 若出现类型错误，例如期望元组但得到列表，将任何子类都强制转换为元组
        for i in range(n):
            row = (rows[i],) if checknull(rows[i]) else tuple(rows[i])  # 将子类强制转换为元组
            for j in range(len(row)):
                result[i, j] = row[j]  # 填充对象数组

    return result  # 返回转换后的对象数组


@cython.wraparound(False)
@cython.boundscheck(False)
def fast_multiget(dict mapping, object[:] keys, default=np.nan) -> "ArrayLike":
    """
    Efficiently retrieve values from a mapping for multiple keys.

    Parameters
    ----------
    mapping : dict
        Dictionary-like object containing key-value pairs.
    keys : Array of objects
        Array of keys to retrieve values from the mapping.
    default : object, optional
        Default value to use if a key is not found in the mapping.

    Returns
    -------
    ArrayLike
        Array of retrieved values corresponding to the keys.
    """
    cdef:
        Py_ssize_t i, n = len(keys)
        object val
        ndarray[object] output = np.empty(n, dtype="O")

    if n == 0:
        # 对于空数组的特殊处理，例如 Series
        return np.empty(0, dtype="f8")

    for i in range(n):
        val = keys[i]  # 当前键
        if val in mapping:
            output[i] = mapping[val]  # 如果键存在于映射中，将对应的值存入输出数组
        else:
            output[i] = default  # 否则存入默认值

    return maybe_convert_objects(output)  # 返回转换后的对象数组
def is_bool_list(obj: list) -> bool:
    """
    Check if this list contains only bool or np.bool_ objects.

    This is appreciably faster than checking `np.array(obj).dtype == bool`

    obj1 = [True, False] * 100
    obj2 = obj1 * 100
    obj3 = obj2 * 100
    obj4 = [True, None] + obj1

    for obj in [obj1, obj2, obj3, obj4]:
        %timeit is_bool_list(obj)
        %timeit np.array(obj).dtype.kind == "b"

    340 ns ± 8.22 ns
    8.78 µs ± 253 ns

    28.8 µs ± 704 ns
    813 µs ± 17.8 µs

    3.4 ms ± 168 µs
    78.4 ms ± 1.05 ms

    48.1 ns ± 1.26 ns
    8.1 µs ± 198 ns
    """
    cdef:
        object item  # 声明一个变量 item，用于遍历 obj 中的元素

    for item in obj:
        if not util.is_bool_object(item):  # 调用 util 模块中的函数检查 item 是否为布尔类型
            return False  # 如果有任何一个元素不是布尔类型，则返回 False

    # Note: we return True for empty list
    return True  # 如果列表为空，则返回 True


cpdef ndarray eq_NA_compat(ndarray[object] arr, object key):
    """
    Check for `arr == key`, treating all values as not-equal to pd.NA.

    key is assumed to have `not isna(key)`
    """
    cdef:
        ndarray[uint8_t, cast=True] result = cnp.PyArray_EMPTY(
            arr.ndim, arr.shape, cnp.NPY_BOOL, 0
        )  # 创建一个与 arr 维度和形状相同的空的布尔类型的 ndarray
        Py_ssize_t i  # 声明一个 Py_ssize_t 类型的变量 i，用于遍历 arr
        object item  # 声明一个变量 item，用于存储 arr 中的每个元素

    for i in range(len(arr)):
        item = arr[i]
        if item is C_NA:  # 如果 item 是 C_NA（假设常量），则设置 result[i] 为 False
            result[i] = False
        else:
            result[i] = item == key  # 否则，设置 result[i] 为 item 是否等于 key

    return result  # 返回包含比较结果的 ndarray


def dtypes_all_equal(list types not None) -> bool:
    """
    Faster version for:

    first = types[0]
    all(is_dtype_equal(first, t) for t in types[1:])

    And assuming all elements in the list are np.dtype/ExtensionDtype objects

    See timings at https://github.com/pandas-dev/pandas/pull/44594
    """
    first = types[0]  # 取出列表中的第一个元素作为比较对象
    for t in types[1:]:
        if t is first:  # 如果 t 与第一个元素相同，直接跳过（快速路径优化）
            continue
        try:
            if not t == first:  # 如果 t 不等于第一个元素，则返回 False
                return False
        except (TypeError, AttributeError):
            return False  # 如果比较中出现 TypeError 或 AttributeError 异常，返回 False
    else:
        return True  # 如果所有元素都与第一个元素相同，则返回 True


def is_np_dtype(object dtype, str kinds=None) -> bool:
    """
    Optimized check for `isinstance(dtype, np.dtype)` with
    optional `and dtype.kind in kinds`.

    dtype = np.dtype("m8[ns]")

    In [7]: %timeit isinstance(dtype, np.dtype)
    117 ns ± 1.91 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)

    In [8]: %timeit is_np_dtype(dtype)
    64 ns ± 1.51 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)

    In [9]: %timeit is_timedelta64_dtype(dtype)
    209 ns ± 6.96 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)

    In [10]: %timeit is_np_dtype(dtype, "m")
    93.4 ns ± 1.11 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)
    """
    if not cnp.PyArray_DescrCheck(dtype):  # 如果 dtype 不是 np.dtype 类型，则返回 False
        # i.e. not isinstance(dtype, np.dtype)
        return False
    if kinds is None:
        return True  # 如果没有提供 kinds 参数，则直接返回 True
    return dtype.kind in kinds  # 否则，检查 dtype 的 kind 是否在 kinds 中
```