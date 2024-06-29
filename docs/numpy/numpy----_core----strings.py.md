# `.\numpy\numpy\_core\strings.py`

```
"""
This module contains a set of functions for vectorized string
operations.
"""

import sys
import numpy as np
from numpy import (
    equal, not_equal, less, less_equal, greater, greater_equal,
    add, multiply as _multiply_ufunc,
)
from numpy._core.multiarray import _vec_string
from numpy._core.umath import (
    isalpha,
    isdigit,
    isspace,
    isalnum,
    islower,
    isupper,
    istitle,
    isdecimal,
    isnumeric,
    str_len,
    find as _find_ufunc,
    rfind as _rfind_ufunc,
    index as _index_ufunc,
    rindex as _rindex_ufunc,
    count as _count_ufunc,
    startswith as _startswith_ufunc,
    endswith as _endswith_ufunc,
    _lstrip_whitespace,
    _lstrip_chars,
    _rstrip_whitespace,
    _rstrip_chars,
    _strip_whitespace,
    _strip_chars,
    _replace,
    _expandtabs_length,
    _expandtabs,
    _center,
    _ljust,
    _rjust,
    _zfill,
    _partition,
    _partition_index,
    _rpartition,
    _rpartition_index,
)

# Exported symbols from this module
__all__ = [
    # UFuncs
    "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
    "add", "multiply", "isalpha", "isdigit", "isspace", "isalnum", "islower",
    "isupper", "istitle", "isdecimal", "isnumeric", "str_len", "find",
    "rfind", "index", "rindex", "count", "startswith", "endswith", "lstrip",
    "rstrip", "strip", "replace", "expandtabs", "center", "ljust", "rjust",
    "zfill", "partition", "rpartition",

    # _vec_string - Will gradually become ufuncs as well
    "upper", "lower", "swapcase", "capitalize", "title",

    # _vec_string - Will probably not become ufuncs
    "mod", "decode", "encode", "translate",

    # Removed from namespace until behavior has been crystalized
    # "join", "split", "rsplit", "splitlines",
]

# Maximum value for int64 data type
MAX = np.iinfo(np.int64).max


def _get_num_chars(a):
    """
    Helper function that returns the number of characters per field in
    a string or unicode array.  This is to abstract out the fact that
    for a unicode array this is itemsize / 4.
    """
    if issubclass(a.dtype.type, np.str_):
        # For numpy string arrays, calculate number of characters per field
        return a.itemsize // 4
    # For other types, return the itemsize
    return a.itemsize


def _to_bytes_or_str_array(result, output_dtype_like):
    """
    Helper function to cast a result back into an array
    with the appropriate dtype if an object array must be used
    as an intermediary.
    """
    # Convert output_dtype_like to numpy array
    output_dtype_like = np.asarray(output_dtype_like)
    if result.size == 0:
        # If result is empty, convert directly to output_dtype_like's dtype
        return result.astype(output_dtype_like.dtype)
    # Convert result to numpy array, considering output_dtype_like's dtype
    ret = np.asarray(result.tolist())
    if isinstance(output_dtype_like.dtype, np.dtypes.StringDType):
        # If output_dtype_like is string type, cast ret accordingly
        return ret.astype(type(output_dtype_like.dtype))
    # Otherwise, cast ret based on output_dtype_like's dtype and itemsize
    return ret.astype(type(output_dtype_like.dtype)(_get_num_chars(ret)))


def _clean_args(*args):
    """
    Helper function for delegating arguments to Python string
    functions.

    Many of the Python string operations that have optional arguments
    are handled by this function.
    """
    # 创建一个空列表，用于存储处理后的参数
    newargs = []
    # 遍历传入的参数列表 args
    for chk in args:
        # 如果当前参数是 None，则停止遍历
        if chk is None:
            break
        # 将非 None 的参数添加到新的参数列表 newargs 中
        newargs.append(chk)
    # 返回处理后的参数列表 newargs
    return newargs
# 定义一个函数，用于实现字符串的多重拼接操作，即将数组 `a` 中的每个元素与数组 `i` 中对应位置的整数倍拼接成新的字符串数组。

def multiply(a, i):
    """
    Return (a * i), that is string multiple concatenation,
    element-wise.

    Values in ``i`` of less than 0 are treated as 0 (which yields an
    empty string).

    Parameters
    ----------
    a : array_like, with ``StringDType``, ``bytes_`` or ``str_`` dtype
        Input array containing strings or bytes to be multiplied.

    i : array_like, with any integer dtype
        Input array containing integers representing the number of times
        each string in `a` should be repeated.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types. Contains the result of element-wise
        string multiplication.

    Raises
    ------
    TypeError
        If `i` contains a dtype that is not an integer.

    MemoryError
        If the resulting repeated strings exceed system memory.

    Examples
    --------
    Provides examples of how the function operates with different inputs.
    """
    # 将输入的 `a` 和 `i` 转换为 NumPy 数组
    a = np.asanyarray(a)
    i = np.asanyarray(i)
    
    # 检查 `i` 是否为整数类型，若不是则抛出类型错误
    if not np.issubdtype(i.dtype, np.integer):
        raise TypeError(f"unsupported type {i.dtype} for operand 'i'")
    
    # 将 `i` 中小于 0 的值设为 0，确保不会产生空字符串
    i = np.maximum(i, 0)

    # 如果 `a` 的 dtype 是 'T'，即字符串类型，则直接使用乘法操作
    if a.dtype.char == "T":
        return a * i

    # 计算 `a` 中每个字符串的长度
    a_len = str_len(a)

    # 确保进行字符串拼接不会导致内存溢出
    if np.any(a_len > sys.maxsize / np.maximum(i, 1)):
        raise MemoryError("repeated string is too long")

    # 计算拼接后的字符串数组的长度
    buffersizes = a_len * i

    # 创建一个与 `a` 类型相同、形状为 `buffersizes.shape`、dtype 为 `out_dtype` 的空数组
    out_dtype = f"{a.dtype.char}{buffersizes.max()}"
    out = np.empty_like(a, shape=buffersizes.shape, dtype=out_dtype)

    # 调用内部函数 `_multiply_ufunc` 完成实际的拼接操作
    return _multiply_ufunc(a, i, out=out)


# 定义一个函数，实现字符串格式化操作，即将数组 `a` 中的每个元素与数组 `values` 中对应位置的值进行格式化插入。

def mod(a, values):
    """
    Return (a % i), that is pre-Python 2.6 string formatting
    (interpolation), element-wise for a pair of array_likes of str
    or unicode.

    Parameters
    ----------
    a : array_like, with `np.bytes_` or `np.str_` dtype
        Input array containing strings or bytes to be formatted.

    values : array_like of values
       These values will be element-wise interpolated into the string.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types. Contains the result of element-wise
        string formatting.
    """
    # 调用 `_vec_string` 函数进行字符串格式化操作，最终返回格式化后的数组
    return _to_bytes_or_str_array(
        _vec_string(a, np.object_, '__mod__', (values,)), a)


# 定义一个函数，用于在数组 `a` 中的每个元素中查找子字符串 `sub`，返回子字符串在每个元素中第一次出现的索引位置。

def find(a, sub, start=0, end=None):
    """
    For each element, return the lowest index in the string where
    substring ``sub`` is found, such that ``sub`` is contained in the
    range [``start``, ``end``).

    Parameters
    ----------
    a : array_like, with ``StringDType``, ``bytes_`` or ``str_`` dtype
        Input array containing strings or bytes to search within.

    sub : array_like, with `np.bytes_` or `np.str_` dtype
        The substring to search for.

    start, end : array_like, with any integer dtype, optional
        The range to look in, interpreted as in slice notation.

    Returns
    -------

    """

    # 返回一个包含每个元素中子字符串 `sub` 第一次出现的最低索引位置的数组
    return _vec_string(a, sub, 'find', (start, end))
    -------
    y : ndarray
        Output array of ints
    输出整数数组 `y`

    See Also
    --------
    str.find
    相关函数：str.find

    Examples
    --------
    >>> a = np.array(["NumPy is a Python library"])
    >>> np.strings.find(a, "Python")
    array([11])
    示例：在数组 `a` 中查找子字符串 "Python"，返回位置索引数组 [11]

    """
    end = end if end is not None else MAX
    # 如果 `end` 不为 None，则使用 `end`；否则使用预定义的 `MAX` 值
    return _find_ufunc(a, sub, start, end)
    # 调用 `_find_ufunc` 函数，传递参数 `a`, `sub`, `start`, `end`，并返回其结果
# 返回数组中每个元素中指定子字符串最后出现的索引位置，查找范围为[start, end)。
def rfind(a, sub, start=0, end=None):
    end = end if end is not None else MAX  # 如果未提供end参数，则使用默认值MAX
    return _rfind_ufunc(a, sub, start, end)  # 调用内部的rfind函数进行实际查找


# 类似于find，但在子字符串未找到时引发ValueError异常。
def index(a, sub, start=0, end=None):
    end = end if end is not None else MAX  # 如果未提供end参数，则使用默认值MAX
    return _index_ufunc(a, sub, start, end)  # 调用内部的index函数进行实际查找


# 类似于rfind，但在子字符串未找到时引发ValueError异常。
def rindex(a, sub, start=0, end=None):
    end = end if end is not None else MAX  # 如果未提供end参数，则使用默认值MAX
    return _rindex_ufunc(a, sub, start, end)  # 调用内部的rindex函数进行实际查找


# 返回数组中每个元素中指定子字符串出现的非重叠次数，查找范围为[start, end)。
def count(a, sub, start=0, end=None):
    end = end if end is not None else MAX  # 如果未提供end参数，则使用默认值MAX
    return _count_ufunc(a, sub, start, end)  # 调用内部的count函数进行实际计数
    # 使用 numpy 库中的字符串处理函数 np.strings.count 对输入的字符串数组进行计数操作
    array([3, 1, 1])
    # 对字符串数组 c 中的每个字符串计算包含 'aA' 的次数，返回结果数组
    >>> np.strings.count(c, 'aA')
    array([3, 1, 0])
    # 对字符串数组 c 中的每个字符串，在指定范围 [start, end) 内计算包含 'A' 的次数，返回结果数组
    >>> np.strings.count(c, 'A', start=1, end=4)
    array([2, 1, 1])
    # 对字符串数组 c 中的每个字符串，在指定范围 [start, end) 内计算包含 'A' 的次数，返回结果数组
    >>> np.strings.count(c, 'A', start=1, end=3)
    # 如果 end 未指定，则使用预定义的 MAX 值作为结束位置
    array([1, 0, 0])

    """
    # 调用内部函数 _count_ufunc，传入参数 a, sub, start, end，并返回其结果
    end = end if end is not None else MAX
    return _count_ufunc(a, sub, start, end)
# 检查字符串数组中每个元素是否以指定前缀开头，返回布尔数组
def startswith(a, prefix, start=0, end=None):
    """
    Returns a boolean array which is `True` where the string element
    in ``a`` starts with ``prefix``, otherwise `False`.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        输入数组，元素可以是字符串类型，字节类型或者字符串

    prefix : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        要检查的前缀，可以是字符串类型，字节类型或者字符串

    start, end : array_like, with any integer dtype
        开始和结束位置的索引，用于指定检查的范围

    Returns
    -------
    out : ndarray
        布尔类型的输出数组，表示每个元素是否以指定前缀开头

    See Also
    --------
    str.startswith
        Python 内置函数 str.startswith 的文档参考

    """
    end = end if end is not None else MAX  # 如果未指定 end 参数，则使用默认值 MAX
    return _startswith_ufunc(a, prefix, start, end)


# 检查字符串数组中每个元素是否以指定后缀结尾，返回布尔数组
def endswith(a, suffix, start=0, end=None):
    """
    Returns a boolean array which is `True` where the string element
    in ``a`` ends with ``suffix``, otherwise `False`.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        输入数组，元素可以是字符串类型，字节类型或者字符串

    suffix : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        要检查的后缀，可以是字符串类型，字节类型或者字符串

    start, end : array_like, with any integer dtype
        开始和结束位置的索引，用于指定检查的范围

    Returns
    -------
    out : ndarray
        布尔类型的输出数组，表示每个元素是否以指定后缀结尾

    See Also
    --------
    str.endswith
        Python 内置函数 str.endswith 的文档参考

    Examples
    --------
    示例用法演示如何使用 np.strings.endswith 函数：
    >>> s = np.array(['foo', 'bar'])
    >>> s
    array(['foo', 'bar'], dtype='<U3')
    >>> np.strings.endswith(s, 'ar')
    array([False,  True])
    >>> np.strings.endswith(s, 'a', start=1, end=2)
    array([False,  True])

    """
    end = end if end is not None else MAX  # 如果未指定 end 参数，则使用默认值 MAX
    return _endswith_ufunc(a, suffix, start, end)


# 对字节数组进行解码，返回相应的字符串数组
def decode(a, encoding=None, errors=None):
    r"""
    Calls :meth:`bytes.decode` element-wise.

    The set of available codecs comes from the Python standard library,
    and may be extended at runtime.  For more information, see the
    :mod:`codecs` module.

    Parameters
    ----------
    a : array_like, with ``bytes_`` dtype
        输入数组，元素必须是字节类型

    encoding : str, optional
       字符串编码的名称，可选参数

    errors : str, optional
       指定如何处理解码错误的字符串，可选参数

    Returns
    -------
    out : ndarray
        解码后的字符串数组

    See Also
    --------
    :py:meth:`bytes.decode`
        Python 内置函数 bytes.decode 的文档参考

    Notes
    -----
    结果的数据类型取决于指定的编码格式。

    Examples
    --------
    示例用法演示如何使用 np.strings.decode 函数：
    >>> c = np.array([b'\x81\xc1\x81\xc1\x81\xc1', b'@@\x81\xc1@@',
    ...               b'\x81\x82\xc2\xc1\xc2\x82\x81'])
    >>> c
    array([b'\x81\xc1\x81\xc1\x81\xc1', b'@@\x81\xc1@@',
           b'\x81\x82\xc2\xc1\xc2\x82\x81'], dtype='|S7')
    >>> np.strings.decode(c, encoding='cp037')
    array(['aAaAaA', '  aA  ', 'abBABba'], dtype='<U7')

    """
    return _to_bytes_or_str_array(
        _vec_string(a, np.object_, 'decode', _clean_args(encoding, errors)),
        np.str_(''))
    Calls :meth:`str.encode` element-wise.
    # 调用 `str.encode` 方法对每个元素进行编码操作。

    The set of available codecs comes from the Python standard library,
    and may be extended at runtime. For more information, see the
    :mod:`codecs` module.
    # 可用的编解码器集合来自于 Python 标准库，并且可以在运行时进行扩展。有关更多信息，请参见 `codecs` 模块。

    Parameters
    ----------
    a : array_like, with ``StringDType`` or ``str_`` dtype
    # 参数 `a`：类似数组，其元素可以是 `StringDType` 或 `str_` 类型。

    encoding : str, optional
       The name of an encoding
    # 参数 `encoding`：字符串，可选，表示要使用的编码名称。

    errors : str, optional
       Specifies how to handle encoding errors
    # 参数 `errors`：字符串，可选，指定如何处理编码错误。

    Returns
    -------
    out : ndarray
    # 返回值 `out`：返回一个 NumPy 数组。

    See Also
    --------
    str.encode
    # 参见：`str.encode`

    Notes
    -----
    The type of the result will depend on the encoding specified.
    # 返回结果的类型取决于指定的编码。

    Examples
    --------
    >>> a = np.array(['aAaAaA', '  aA  ', 'abBABba'])
    >>> np.strings.encode(a, encoding='cp037')
    array([b'\x81\xc1\x81\xc1\x81\xc1', b'@@\x81\xc1@@',
       b'\x81\x82\xc2\xc1\xc2\x82\x81'], dtype='|S7')
    # 示例：对数组 `a` 中的字符串进行编码，使用编码器 `cp037`，返回编码后的 NumPy 数组。
       
    """
    return _to_bytes_or_str_array(
        _vec_string(a, np.object_, 'encode', _clean_args(encoding, errors)),
        np.bytes_(b''))
    # 调用 `_vec_string` 函数，对数组 `a` 的每个元素进行编码，返回字节或字符串数组。
    # `_clean_args` 函数用于清理并处理编码和错误处理参数。
    # 最终返回一个字节类型的 NumPy 数组。
# 将函数定义为一个名为 expandtabs 的函数，用于处理数组中的字符串元素，将其中的制表符替换为一定数量的空格
def expandtabs(a, tabsize=8):
    """
    Return a copy of each string element where all tab characters are
    replaced by one or more spaces.

    Calls :meth:`str.expandtabs` element-wise.

    Return a copy of each string element where all tab characters are
    replaced by one or more spaces, depending on the current column
    and the given `tabsize`. The column number is reset to zero after
    each newline occurring in the string. This doesn't understand other
    non-printing characters or escape sequences.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Input array
    tabsize : int, optional
        Replace tabs with `tabsize` number of spaces.  If not given defaults
        to 8 spaces.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input type

    See Also
    --------
    str.expandtabs

    Examples
    --------
    >>> a = np.array(['\t\tHello\tworld'])
    >>> np.strings.expandtabs(a, tabsize=4)  # doctest: +SKIP
    array(['        Hello   world'], dtype='<U21')  # doctest: +SKIP

    """
    # 将输入的 a 转换为 numpy 数组
    a = np.asanyarray(a)
    # 将输入的 tabsize 转换为 numpy 数组
    tabsize = np.asanyarray(tabsize)

    # 如果 a 的 dtype 是 "T" 类型
    if a.dtype.char == "T":
        # 调用内部函数 _expandtabs 处理 a
        return _expandtabs(a, tabsize)

    # 计算每个字符串元素经过扩展后的长度
    buffersizes = _expandtabs_length(a, tabsize)
    # 构造输出的 dtype，长度为 buffersizes 中的最大值
    out_dtype = f"{a.dtype.char}{buffersizes.max()}"
    # 创建一个和 a 相同形状的空数组，dtype 为 out_dtype
    out = np.empty_like(a, shape=buffersizes.shape, dtype=out_dtype)
    # 调用 _expandtabs 函数处理 a，并将结果写入 out 中
    return _expandtabs(a, tabsize, out=out)



# 将函数定义为一个名为 center 的函数，用于将数组中的字符串元素居中显示，可指定填充字符和输出宽度
def center(a, width, fillchar=' '):
    """
    Return a copy of `a` with its elements centered in a string of
    length `width`.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype

    width : array_like, with any integer dtype
        The length of the resulting strings, unless ``width < str_len(a)``.
    fillchar : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Optional padding character to use (default is space).

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types

    See Also
    --------
    str.center

    Notes
    -----
    While it is possible for ``a`` and ``fillchar`` to have different dtypes,
    passing a non-ASCII character in ``fillchar`` when ``a`` is of dtype "S"
    is not allowed, and a ``ValueError`` is raised.

    Examples
    --------
    >>> c = np.array(['a1b2','1b2a','b2a1','2a1b']); c
    array(['a1b2', '1b2a', 'b2a1', '2a1b'], dtype='<U4')
    >>> np.strings.center(c, width=9)
    array(['   a1b2  ', '   1b2a  ', '   b2a1  ', '   2a1b  '], dtype='<U9')
    >>> np.strings.center(c, width=9, fillchar='*')
    array(['***a1b2**', '***1b2a**', '***b2a1**', '***2a1b**'], dtype='<U9')
    >>> np.strings.center(c, width=1)
    array(['a1b2', '1b2a', 'b2a1', '2a1b'], dtype='<U4')

    """
    # 将输入的 a 转换为 numpy 数组
    a = np.asanyarray(a)
    # 将 fillchar 转换为与 a 的数据类型相同的 NumPy 数组
    fillchar = np.asanyarray(fillchar, dtype=a.dtype)
    
    # 检查 fillchar 是否包含多个字符，如果是，则引发 TypeError 异常
    if np.any(str_len(fillchar) != 1):
        raise TypeError(
            "The fill character must be exactly one character long")
    
    # 如果 a 的数据类型为 'T'（字符串类型），则调用 _center 函数进行居中处理并返回结果
    if a.dtype.char == "T":
        return _center(a, width, fillchar)
    
    # 计算 a 的字符串长度与指定宽度的最大值，并用于输出的数据类型
    width = np.maximum(str_len(a), width)
    out_dtype = f"{a.dtype.char}{width.max()}"
    
    # 根据 a 的形状、指定宽度的形状和 fillchar 的形状，确定输出数组的形状
    shape = np.broadcast_shapes(a.shape, width.shape, fillchar.shape)
    
    # 创建一个与 a 类型相同的空数组，用于存储结果，并指定其形状和数据类型
    out = np.empty_like(a, shape=shape, dtype=out_dtype)
    
    # 调用 _center 函数，将 a、width、fillchar 和输出数组 out 传递给它，并返回结果
    return _center(a, width, fillchar, out=out)
# 定义函数 ljust，用于将数组 `a` 中的元素左对齐，填充至长度为 `width` 的字符串中
def ljust(a, width, fillchar=' '):
    """
    Return an array with the elements of `a` left-justified in a
    string of length `width`.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        输入数组，可以是字符串数组或字节串数组

    width : array_like, with any integer dtype
        结果字符串的长度，除非 `width < str_len(a)`。

    fillchar : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        可选参数，用于填充的字符（默认为空格）。

    Returns
    -------
    out : ndarray
        返回一个数组，包含了左对齐后的字符串，其数据类型为 `StringDType`、`bytes_` 或 `str_`，取决于输入类型。

    See Also
    --------
    str.ljust

    Notes
    -----
    虽然 `a` 和 `fillchar` 可能具有不同的数据类型，
    但在 `a` 的数据类型为 "S" 时，如果 `fillchar` 中包含非 ASCII 字符，
    则会引发 `ValueError` 异常。

    Examples
    --------
    >>> c = np.array(['aAaAaA', '  aA  ', 'abBABba'])
    >>> np.strings.ljust(c, width=3)
    array(['aAaAaA', '  aA  ', 'abBABba'], dtype='<U7')
    >>> np.strings.ljust(c, width=9)
    array(['aAaAaA   ', '  aA     ', 'abBABba  '], dtype='<U9')

    """
    # 将 `a` 转换为任意数组
    a = np.asanyarray(a)
    # 将 `fillchar` 转换为与 `a` 相同的数据类型的数组
    fillchar = np.asanyarray(fillchar, dtype=a.dtype)

    # 如果 `fillchar` 中有长度不为 1 的元素，则引发 TypeError 异常
    if np.any(str_len(fillchar) != 1):
        raise TypeError(
            "The fill character must be exactly one character long")

    # 如果 `a` 的数据类型为 "T"，则调用 `_ljust` 函数来执行左对齐操作
    if a.dtype.char == "T":
        return _ljust(a, width, fillchar)

    # 计算字符串数组 `a` 中每个字符串的最大长度与 `width` 的最大值作为新的 `width`
    width = np.maximum(str_len(a), width)
    # 计算输出数组的形状，以适应 `a`、`width` 和 `fillchar`
    shape = np.broadcast_shapes(a.shape, width.shape, fillchar.shape)
    # 根据结果数据类型创建一个新的输出数组
    out_dtype = f"{a.dtype.char}{width.max()}"
    out = np.empty_like(a, shape=shape, dtype=out_dtype)
    # 调用 `_ljust` 函数执行左对齐操作，并将结果存储在 `out` 中
    return _ljust(a, width, fillchar, out=out)



# 定义函数 rjust，用于将数组 `a` 中的元素右对齐，填充至长度为 `width` 的字符串中
def rjust(a, width, fillchar=' '):
    """
    Return an array with the elements of `a` right-justified in a
    string of length `width`.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        输入数组，可以是字符串数组或字节串数组

    width : array_like, with any integer dtype
        结果字符串的长度，除非 `width < str_len(a)`。

    fillchar : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        可选参数，用于填充的字符（默认为空格）。

    Returns
    -------
    out : ndarray
        返回一个数组，包含了右对齐后的字符串，其数据类型为 `StringDType`、`bytes_` 或 `str_`，取决于输入类型。

    See Also
    --------
    str.rjust

    Notes
    -----
    虽然 `a` 和 `fillchar` 可能具有不同的数据类型，
    但在 `a` 的数据类型为 "S" 时，如果 `fillchar` 中包含非 ASCII 字符，
    则会引发 `ValueError` 异常。

    Examples
    --------
    >>> a = np.array(['aAaAaA', '  aA  ', 'abBABba'])
    >>> np.strings.rjust(a, width=3)
    array(['aAaAaA', '  aA  ', 'abBABba'], dtype='<U7')
    >>> np.strings.rjust(a, width=9)
    array(['   aAaAaA', '     aA  ', '  abBABba'], dtype='<U9')

    """
    # 将 `a` 转换为任意数组
    a = np.asanyarray(a)
    # 将 fillchar 转换为与数组 a 相同的数据类型的 NumPy 数组
    fillchar = np.asanyarray(fillchar, dtype=a.dtype)

    # 如果 fillchar 中有任何一个元素的长度不等于 1，则抛出 TypeError 异常
    if np.any(str_len(fillchar) != 1):
        raise TypeError(
            "The fill character must be exactly one character long")

    # 如果数组 a 的数据类型是 'T'（字符串类型），则调用 _rjust 函数右对齐数组 a
    if a.dtype.char == "T":
        return _rjust(a, width, fillchar)

    # 计算数组 a 和 width 的最大字符串长度
    width = np.maximum(str_len(a), width)

    # 计算广播后的形状，以适应 a 的形状、width 的形状和 fillchar 的形状
    shape = np.broadcast_shapes(a.shape, width.shape, fillchar.shape)

    # 根据计算得到的形状和数据类型创建一个与 a 相同类型的空数组 out
    out_dtype = f"{a.dtype.char}{width.max()}"
    out = np.empty_like(a, shape=shape, dtype=out_dtype)

    # 调用 _rjust 函数，使用计算得到的数组 a、width 和 fillchar 进行右对齐操作，结果存入 out
    return _rjust(a, width, fillchar, out=out)
# 定义一个函数 zfill，用于将数字字符串左侧填充零。如果有符号前缀（+/-），则在符号字符之后填充零。
def zfill(a, width):
    """
    Return the numeric string left-filled with zeros. A leading
    sign prefix (``+``/``-``) is handled by inserting the padding
    after the sign character rather than before.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        输入参数 `a`，可以是数组形式，数据类型可以是字符串、字节串或字符串数组
    width : array_like, with any integer dtype
        输入参数 `width`，可以是任意整数数据类型的数组，指定要填充到的字符串宽度

    Returns
    -------
    out : ndarray
        返回值是一个数组，数据类型为 `StringDType`、`bytes_` 或 `str_`，具体取决于输入类型

    See Also
    --------
    str.zfill

    Examples
    --------
    >>> np.strings.zfill(['1', '-1', '+1'], 3)
    array(['001', '-01', '+01'], dtype='<U3')

    """
    # 将输入参数 `a` 转换为任意数组
    a = np.asanyarray(a)

    # 如果输入数组 `a` 的数据类型字符为 "T"，则调用内部函数 `_zfill` 处理
    if a.dtype.char == "T":
        return _zfill(a, width)

    # 计算 `a` 中每个元素的字符串长度，并与 `width` 取最大值
    width = np.maximum(str_len(a), width)
    # 广播 `a` 和 `width` 的形状，得到输出数组的形状
    shape = np.broadcast_shapes(a.shape, width.shape)
    # 输出数组的数据类型，由 `a` 的数据类型字符和 `width` 最大值组成
    out_dtype = f"{a.dtype.char}{width.max()}"
    # 创建一个和 `a` 类型相同、形状为 `shape`、数据类型为 `out_dtype` 的空数组
    out = np.empty_like(a, shape=shape, dtype=out_dtype)
    # 调用内部函数 `_zfill` 处理并返回结果
    return _zfill(a, width, out=out)


# 定义一个函数 lstrip，用于去除每个元素开头的指定字符集合
def lstrip(a, chars=None):
    """
    For each element in `a`, return a copy with the leading characters
    removed.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        输入参数 `a`，可以是数组形式，数据类型可以是字符串、字节串或字符串数组
    chars : scalar with the same dtype as ``a``, optional
       The ``chars`` argument is a string specifying the set of
       characters to be removed. If ``None``, the ``chars``
       argument defaults to removing whitespace. The ``chars`` argument
       is not a prefix or suffix; rather, all combinations of its
       values are stripped.

    Returns
    -------
    out : ndarray
        返回值是一个数组，数据类型为 `StringDType`、`bytes_` 或 `str_`，具体取决于输入类型

    See Also
    --------
    str.lstrip

    Examples
    --------
    >>> c = np.array(['aAaAaA', '  aA  ', 'abBABba'])
    >>> c
    array(['aAaAaA', '  aA  ', 'abBABba'], dtype='<U7')
    # The 'a' variable is unstripped from c[1] because of leading whitespace.
    >>> np.strings.lstrip(c, 'a')
    array(['AaAaA', '  aA  ', 'bBABba'], dtype='<U7')
    >>> np.strings.lstrip(c, 'A') # leaves c unchanged
    array(['aAaAaA', '  aA  ', 'abBABba'], dtype='<U7')
    >>> (np.strings.lstrip(c, ' ') == np.strings.lstrip(c, '')).all()
    np.False_
    >>> (np.strings.lstrip(c, ' ') == np.strings.lstrip(c)).all()
    np.True_

    """
    # 如果 `chars` 参数为 `None`，则调用内部函数 `_lstrip_whitespace` 去除空白字符
    if chars is None:
        return _lstrip_whitespace(a)
    # 否则调用内部函数 `_lstrip_chars` 去除指定的字符集合 `chars`
    return _lstrip_chars(a, chars)


# 定义一个函数 rstrip，用于去除每个元素末尾的指定字符集合
def rstrip(a, chars=None):
    """
    For each element in `a`, return a copy with the trailing characters
    removed.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        输入参数 `a`，可以是数组形式，数据类型可以是字符串、字节串或字符串数组
    chars : scalar with the same dtype as ``a``, optional
       The ``chars`` argument is a string specifying the set of
       characters to be removed. If ``None``, the ``chars``
       argument defaults to removing whitespace. The ``chars`` argument
       is not a prefix or suffix; rather, all combinations of its
       values are stripped.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types

    See Also
    --------
    str.rstrip

    Examples
    --------
    >>> c = np.array(['aAaAaA', 'abBABba'])
    >>> c
    array(['aAaAaA', 'abBABba'], dtype='<U7')
    >>> np.strings.rstrip(c, 'a')
    array(['aAaAaA', 'abBABb'], dtype='<U7')
    >>> np.strings.rstrip(c, 'A')
    array(['aAaAa', 'abBABba'], dtype='<U7')

    """
    # 如果 chars 参数为 None，则调用 _rstrip_whitespace 函数去除字符串末尾的空白字符
    if chars is None:
        return _rstrip_whitespace(a)
    # 否则，调用 _rstrip_chars 函数去除字符串末尾指定的字符集合
    return _rstrip_chars(a, chars)
def strip(a, chars=None):
    """
    For each element in `a`, return a copy with the leading and
    trailing characters removed.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Input array.
    chars : scalar with the same dtype as ``a``, optional
        The ``chars`` argument is a string specifying the set of
        characters to be removed. If ``None``, the ``chars``
        argument defaults to removing whitespace. The ``chars`` argument
        is not a prefix or suffix; rather, all combinations of its
        values are stripped.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types

    See Also
    --------
    str.strip

    Examples
    --------
    >>> c = np.array(['aAaAaA', '  aA  ', 'abBABba'])
    >>> c
    array(['aAaAaA', '  aA  ', 'abBABba'], dtype='<U7')
    >>> np.strings.strip(c)
    array(['aAaAaA', 'aA', 'abBABba'], dtype='<U7')
    # 'a' unstripped from c[1] because of leading whitespace.
    >>> np.strings.strip(c, 'a')
    array(['AaAaA', '  aA  ', 'bBABb'], dtype='<U7')
    # 'A' unstripped from c[1] because of trailing whitespace.
    >>> np.strings.strip(c, 'A')
    array(['aAaAa', '  aA  ', 'abBABba'], dtype='<U7')

    """
    # 如果 `chars` 参数为 None，则调用 _strip_whitespace 函数去除空白字符
    if chars is None:
        return _strip_whitespace(a)
    # 否则，调用 _strip_chars 函数去除指定的字符集合
    return _strip_chars(a, chars)


def upper(a):
    """
    Return an array with the elements converted to uppercase.

    Calls :meth:`str.upper` element-wise.

    For 8-bit strings, this method is locale-dependent.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Input array.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types

    See Also
    --------
    str.upper

    Examples
    --------
    >>> c = np.array(['a1b c', '1bca', 'bca1']); c
    array(['a1b c', '1bca', 'bca1'], dtype='<U5')
    >>> np.strings.upper(c)
    array(['A1B C', '1BCA', 'BCA1'], dtype='<U5')

    """
    # 将输入数组 `a` 转换为 ndarray 类型
    a_arr = np.asarray(a)
    # 调用 _vec_string 函数，将元素逐个转换为大写形式
    return _vec_string(a_arr, a_arr.dtype, 'upper')


def lower(a):
    """
    Return an array with the elements converted to lowercase.

    Call :meth:`str.lower` element-wise.

    For 8-bit strings, this method is locale-dependent.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Input array.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types

    See Also
    --------
    str.lower

    Examples
    --------
    >>> c = np.array(['A1B C', '1BCA', 'BCA1']); c
    array(['A1B C', '1BCA', 'BCA1'], dtype='<U5')
    >>> np.strings.lower(c)
    array(['a1b c', '1bca', 'bca1'], dtype='<U5')

    """
    # 将输入数组 `a` 转换为 ndarray 类型
    a_arr = np.asarray(a)
    # 调用 _vec_string 函数，将元素逐个转换为小写形式
    return _vec_string(a_arr, a_arr.dtype, 'lower')


def swapcase(a):
    """
    Return an array with uppercase characters converted to lowercase
    and vice versa.

    Call :meth:`str.swapcase` element-wise.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Input array.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types

    See Also
    --------
    str.swapcase

    Examples
    --------
    >>> c = np.array(['aAaAaA', '  aA  ', 'abBABba']); c
    array(['aAaAaA', '  aA  ', 'abBABba'], dtype='<U7')
    >>> np.strings.swapcase(c)
    array(['AaAaAa', '  Aa  ', 'ABbabBA'], dtype='<U7')

    """
    # 将输入数组 `a` 转换为 ndarray 类型
    a_arr = np.asarray(a)
    # 调用 _vec_string 函数，将大写转换为小写，小写转换为大写
    return _vec_string(a_arr, a_arr.dtype, 'swapcase')
    """
    将字符串中的大写字符转换为小写字符，小写字符转换为大写字符，并返回副本。
    
    调用 :meth:`str.swapcase` 来实现逐元素转换。
    
    对于8位字符串，此方法依赖于本地化设置。
    
    Parameters
    ----------
    a : array-like，具有 ``StringDType``, ``bytes_``, 或 ``str_`` 数据类型
        输入数组。
    
    Returns
    -------
    out : ndarray
        输出数组，数据类型为 ``StringDType``, ``bytes_`` 或 ``str_``，取决于输入类型。
    
    See Also
    --------
    str.swapcase
    
    Examples
    --------
    >>> c=np.array(['a1B c','1b Ca','b Ca1','cA1b'],'S5'); c
    array(['a1B c', '1b Ca', 'b Ca1', 'cA1b'],
        dtype='|S5')
    >>> np.strings.swapcase(c)
    array(['A1b C', '1B cA', 'B cA1', 'Ca1B'],
        dtype='|S5')
    
    """
    # 将输入的数组转换为 numpy 数组
    a_arr = np.asarray(a)
    # 调用内部函数 _vec_string，传入输入数组、数组的数据类型和操作类型 'swapcase'，返回处理后的数组
    return _vec_string(a_arr, a_arr.dtype, 'swapcase')
def capitalize(a):
    """
    Return a copy of ``a`` with only the first character of each element
    capitalized.

    Calls :meth:`str.capitalize` element-wise.

    For byte strings, this method is locale-dependent.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Input array of strings to capitalize.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types

    See Also
    --------
    str.capitalize

    Examples
    --------
    >>> c = np.array(['a1b2','1b2a','b2a1','2a1b'],'S4'); c
    array(['a1b2', '1b2a', 'b2a1', '2a1b'],
        dtype='|S4')
    >>> np.strings.capitalize(c)
    array(['A1b2', '1b2a', 'B2a1', '2a1b'],
        dtype='|S4')

    """
    # 将输入数组转换为 NumPy 数组
    a_arr = np.asarray(a)
    # 调用内部函数 _vec_string 进行字符串首字母大写处理
    return _vec_string(a_arr, a_arr.dtype, 'capitalize')


def title(a):
    """
    Return element-wise title cased version of string or unicode.

    Title case words start with uppercase characters, all remaining cased
    characters are lowercase.

    Calls :meth:`str.title` element-wise.

    For 8-bit strings, this method is locale-dependent.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Input array.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types

    See Also
    --------
    str.title

    Examples
    --------
    >>> c=np.array(['a1b c','1b ca','b ca1','ca1b'],'S5'); c
    array(['a1b c', '1b ca', 'b ca1', 'ca1b'],
        dtype='|S5')
    >>> np.strings.title(c)
    array(['A1B C', '1B Ca', 'B Ca1', 'Ca1B'],
        dtype='|S5')

    """
    # 将输入数组转换为 NumPy 数组
    a_arr = np.asarray(a)
    # 调用内部函数 _vec_string 进行字符串标题格式处理
    return _vec_string(a_arr, a_arr.dtype, 'title')


def replace(a, old, new, count=-1):
    """
    For each element in ``a``, return a copy of the string with
    occurrences of substring ``old`` replaced by ``new``.

    Parameters
    ----------
    a : array_like, with ``bytes_`` or ``str_`` dtype

    old, new : array_like, with ``bytes_`` or ``str_`` dtype

    count : array_like, with ``int_`` dtype
        If the optional argument ``count`` is given, only the first
        ``count`` occurrences are replaced.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types

    See Also
    --------
    str.replace
    
    Examples
    --------
    >>> a = np.array(["That is a mango", "Monkeys eat mangos"])
    >>> np.strings.replace(a, 'mango', 'banana')
    array(['That is a banana', 'Monkeys eat bananas'], dtype='<U19')

    >>> a = np.array(["The dish is fresh", "This is it"])
    >>> np.strings.replace(a, 'is', 'was')
    array(['The dwash was fresh', 'Thwas was it'], dtype='<U19')
    
    """
    # 将输入数组转换为 NumPy 通用数组
    arr = np.asanyarray(a)
    # 获取数组的数据类型
    a_dt = arr.dtype
    # 将旧数组转换为NumPy数组，如果没有指定dtype，则使用a_dt
    old = np.asanyarray(old, dtype=getattr(old, 'dtype', a_dt))
    
    # 将新数组转换为NumPy数组，如果没有指定dtype，则使用a_dt
    new = np.asanyarray(new, dtype=getattr(new, 'dtype', a_dt))
    
    # 将count转换为NumPy数组
    count = np.asanyarray(count)
    
    # 如果数组arr的dtype的字符代码为"T"（代表'timedelta'类型），则调用_replace函数进行替换操作并返回结果
    if arr.dtype.char == "T":
        return _replace(arr, old, new, count)
    
    # 获取np.int64类型的最大值
    max_int64 = np.iinfo(np.int64).max
    
    # 调用_count_ufunc函数计算在数组arr中与old匹配的元素数量，限制计数不超过max_int64
    counts = _count_ufunc(arr, old, 0, max_int64)
    
    # 根据count数组的值对counts数组进行调整：如果count小于0，则保留原有counts值；否则取counts和count中较小的值
    counts = np.where(count < 0, counts, np.minimum(counts, count))
    
    # 根据替换操作计算输出数组的缓冲区大小，buffersizes的每个元素表示对应位置的替换后字符串的长度变化
    buffersizes = str_len(arr) + counts * (str_len(new) - str_len(old))
    
    # 构建输出数组的dtype，字符代码为原始arr的dtype的字符代码，并根据buffersizes的最大值确定dtype的长度
    out_dtype = f"{arr.dtype.char}{buffersizes.max()}"
    
    # 根据out_dtype创建一个与arr形状相同且dtype为out_dtype的空数组out
    out = np.empty_like(arr, shape=buffersizes.shape, dtype=out_dtype)
    
    # 调用_replace函数，使用old、new和counts替换arr中的元素，并将结果存入out数组中
    return _replace(arr, old, new, counts, out=out)
# 返回一个字符串，该字符串是序列 `seq` 中字符串连接而成的结果。
# 对每个元素调用 :meth:`str.join` 方法进行连接。
def _join(sep, seq):
    # 调用 _vec_string 函数，将 sep 和 seq 转换为字节或字符串数组，然后进行连接操作
    return _to_bytes_or_str_array(
        _vec_string(sep, np.object_, 'join', (seq,)), seq)


# 对于每个元素 `a`，使用 `sep` 作为分隔符字符串，返回字符串中的单词列表。
# 对每个元素调用 :meth:`str.split` 方法进行分割。
def _split(a, sep=None, maxsplit=None):
    # 这将返回一个大小不同的列表数组，因此我们将其保留为对象数组
    return _vec_string(
        a, np.object_, 'split', [sep] + _clean_args(maxsplit))


# 对于每个元素 `a`，使用 `sep` 作为分隔符字符串，从右向左返回字符串中的单词列表。
# 对每个元素调用 :meth:`str.rsplit` 方法进行分割。
# 与 `split` 类似，不同之处在于从右侧开始分割。
def _rsplit(a, sep=None, maxsplit=None):
    return _vec_string(
        a, np.object_, 'rsplit', [sep] + _clean_args(maxsplit))


这些函数是针对 NumPy 数组中包含的字符串元素进行操作的工具函数，使用了 NumPy 的字符串处理功能。
    """
    # 返回一个数组，其中包含不同大小的列表，因此我们将其保留为对象数组
    # 这个数组可能包含不同长度的列表，因此数据类型被指定为对象数组
    return _vec_string(
        a, np.object_, 'rsplit', [sep] + _clean_args(maxsplit))
    """
def _splitlines(a, keepends=None):
    """
    对 `a` 中的每个元素进行操作，返回每个元素按行划分后的列表。

    调用 :meth:`str.splitlines` 方法逐个元素处理。

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        输入数组，元素类型为字符串或字节流
    keepends : bool, optional
        如果为 True，则结果列表中保留行结束符；默认情况下不包含行结束符。

    Returns
    -------
    out : ndarray
        包含列表对象的数组

    See Also
    --------
    str.splitlines

    """
    # 调用 `_vec_string` 函数对每个元素进行 `splitlines` 操作，传递 `keepends` 参数
    return _vec_string(
        a, np.object_, 'splitlines', _clean_args(keepends))


def partition(a, sep):
    """
    对 `a` 中的每个元素围绕 `sep` 进行分区。

    对 `a` 中的每个元素，在第一次出现 `sep` 的位置进行分割，并返回一个三元组，
    包含分隔符之前的部分、分隔符本身和分隔符之后的部分。如果找不到分隔符，
    则三元组的第一项将包含整个字符串，第二和第三项将为空字符串。

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        输入数组
    sep : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        用于分割每个字符串元素的分隔符。

    Returns
    -------
    out : 3-tuple:
        - 元素类型为 ``StringDType``, ``bytes_`` 或 ``str_`` 的数组，包含分隔符之前的部分
        - 元素类型为 ``StringDType``, ``bytes_`` 或 ``str_`` 的数组，包含分隔符本身
        - 元素类型为 ``StringDType``, ``bytes_`` 或 ``str_`` 的数组，包含分隔符之后的部分

    See Also
    --------
    str.partition

    Examples
    --------
    >>> x = np.array(["Numpy is nice!"])
    >>> np.strings.partition(x, " ")
    (array(['Numpy'], dtype='<U5'),
     array([' '], dtype='<U1'),
     array(['is nice!'], dtype='<U8'))

    """
    # 将 `a` 转换为 `ndarray`
    a = np.asanyarray(a)
    # 当问题围绕视图问题解决时，切换到 copy=False
    sep = np.array(sep, dtype=a.dtype, copy=True, subok=True)
    if a.dtype.char == "T":
        return _partition(a, sep)

    # 在 `a` 中寻找 `sep` 的位置
    pos = _find_ufunc(a, sep, 0, MAX)
    a_len = str_len(a)
    sep_len = str_len(sep)

    not_found = pos < 0
    buffersizes1 = np.where(not_found, a_len, pos)
    buffersizes3 = np.where(not_found, 0, a_len - pos - sep_len)

    # 计算输出的 dtype
    out_dtype = ",".join([f"{a.dtype.char}{n}" for n in (
        buffersizes1.max(),
        1 if np.all(not_found) else sep_len.max(),
        buffersizes3.max(),
    )])
    shape = np.broadcast_shapes(a.shape, sep.shape)
    # 创建输出数组
    out = np.empty_like(a, shape=shape, dtype=out_dtype)
    return _partition_index(a, sep, pos, out=(out["f0"], out["f1"], out["f2"]))
    occurrence of ``sep``, and return a 3-tuple containing the part
    before the separator, the separator itself, and the part after
    the separator. If the separator is not found, the third item of
    the tuple will contain the whole string, and the first and second
    ones will be the empty string.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Input array
    sep : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Separator to split each string element in ``a``.

    Returns
    -------
    out : 3-tuple:
        - array with ``StringDType``, ``bytes_`` or ``str_`` dtype with the
          part before the separator
        - array with ``StringDType``, ``bytes_`` or ``str_`` dtype with the
          separator
        - array with ``StringDType``, ``bytes_`` or ``str_`` dtype with the
          part after the separator

    See Also
    --------
    str.rpartition

    Examples
    --------
    >>> a = np.array(['aAaAaA', '  aA  ', 'abBABba'])
    >>> np.strings.rpartition(a, 'A')
    (array(['aAaAa', '  a', 'abB'], dtype='<U5'),
     array(['A', 'A', 'A'], dtype='<U1'),
     array(['', '  ', 'Bba'], dtype='<U3'))

    """
    # 将输入的数组 a 转换为 NumPy 数组
    a = np.asanyarray(a)
    # TODO switch to copy=False when issues around views are fixed
    # 将分隔符数组 sep 转换为与 a 相同类型的 NumPy 数组副本
    sep = np.array(sep, dtype=a.dtype, copy=True, subok=True)
    # 如果 a 的数据类型是 'T'，调用 _rpartition 函数处理
    if a.dtype.char == "T":
        return _rpartition(a, sep)

    # 使用 _rfind_ufunc 函数查找 sep 在 a 中最后出现的位置
    pos = _rfind_ufunc(a, sep, 0, MAX)
    # 计算 a 和 sep 的长度
    a_len = str_len(a)
    sep_len = str_len(sep)

    # 判断是否找到分隔符，如果没找到则设置相应的缓冲区大小为 0 或者 a 和 sep 的长度
    not_found = pos < 0
    buffersizes1 = np.where(not_found, 0, pos)
    buffersizes3 = np.where(not_found, a_len, a_len - pos - sep_len)

    # 构建输出的数据类型，包括部分之前的长度、分隔符长度和部分之后的长度
    out_dtype = ",".join([f"{a.dtype.char}{n}" for n in (
        buffersizes1.max(),
        1 if np.all(not_found) else sep_len.max(),
        buffersizes3.max(),
    )])
    # 广播输入数组 a 和 sep 的形状
    shape = np.broadcast_shapes(a.shape, sep.shape)
    # 创建一个与 a 类型相同的空数组作为输出
    out = np.empty_like(a, shape=shape, dtype=out_dtype)
    # 调用 _rpartition_index 函数进行分区操作，将结果存储在 out 中的对应字段
    return _rpartition_index(
        a, sep, pos, out=(out["f0"], out["f1"], out["f2"]))
def translate(a, table, deletechars=None):
    """
    对于 `a` 中的每个元素，返回一个新字符串副本，其中删除了在可选参数 `deletechars` 中出现的所有字符，并将剩余字符通过给定的转换表映射。

    逐元素调用 :meth:`str.translate`。

    Parameters
    ----------
    a : array-like, with `np.bytes_` or `np.str_` dtype
        输入数组，元素类型为 `np.bytes_` 或 `np.str_`

    table : str of length 256
        长度为 256 的字符串转换表

    deletechars : str, optional
        要删除的字符集合，可选参数

    Returns
    -------
    out : ndarray
        输出的字符串数组，根据输入类型为 str 或 unicode

    See Also
    --------
    str.translate

    Examples
    --------
    >>> a = np.array(['a1b c', '1bca', 'bca1'])
    >>> table = a[0].maketrans('abc', '123')
    >>> deletechars = ' '
    >>> np.char.translate(a, table, deletechars)
    array(['112 3', '1231', '2311'], dtype='<U5')
    """
    # 将输入 `a` 转换为 ndarray 类型
    a_arr = np.asarray(a)
    # 如果 `a_arr` 的元素类型是字符串类型
    if issubclass(a_arr.dtype.type, np.str_):
        # 调用 _vec_string 函数处理字符串类型的数组 `a_arr`，并返回结果
        return _vec_string(
            a_arr, a_arr.dtype, 'translate', (table,))
    else:
        # 否则，调用 _vec_string 函数处理非字符串类型的数组 `a_arr`，并返回结果
        return _vec_string(
            a_arr,
            a_arr.dtype,
            'translate',
            [table] + _clean_args(deletechars)
        )
```