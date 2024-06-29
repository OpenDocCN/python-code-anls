# `.\numpy\numpy\_core\defchararray.py`

```py
"""
This module contains a set of functions for vectorized string
operations and methods.

.. note::
   The `chararray` class exists for backwards compatibility with
   Numarray, it is not recommended for new development. Starting from numpy
   1.4, if one needs arrays of strings, it is recommended to use arrays of
   `dtype` `object_`, `bytes_` or `str_`, and use the free functions
   in the `numpy.char` module for fast vectorized string operations.
   
一些方法只有在你的 Python 版本中相应的字符串方法可用时才有效。

`defchararray` 的首选别名是 `numpy.char`。
"""
import functools  # 导入 functools 模块

import numpy as np  # 导入 numpy 模块
from .._utils import set_module  # 从上层目录的 _utils 模块中导入 set_module 函数
from .numerictypes import bytes_, str_, character  # 从 numerictypes 模块中导入 bytes_, str_, character 类
from .numeric import ndarray, array as narray, asarray as asnarray  # 从 numeric 模块中导入 ndarray, narray, asnarray 函数
from numpy._core.multiarray import compare_chararrays  # 从 numpy._core.multiarray 模块中导入 compare_chararrays 函数
from numpy._core import overrides  # 从 numpy._core 模块中导入 overrides 函数
from numpy.strings import *  # 导入 numpy.strings 中的所有方法
from numpy.strings import (  # 导入 numpy.strings 中的 multiply, partition, rpartition 方法
    multiply as strings_multiply,
    partition as strings_partition,
    rpartition as strings_rpartition,
)
from numpy._core.strings import (  # 从 numpy._core.strings 模块中导入 _split, _rsplit, _splitlines, _join 函数
    _split as split,
    _rsplit as rsplit,
    _splitlines as splitlines,
    _join as join,
)

__all__ = [  # 定义 __all__ 列表
    'equal', 'not_equal', 'greater_equal', 'less_equal',
    'greater', 'less', 'str_len', 'add', 'multiply', 'mod', 'capitalize',
    'center', 'count', 'decode', 'encode', 'endswith', 'expandtabs',
    'find', 'index', 'isalnum', 'isalpha', 'isdigit', 'islower', 'isspace',
    'istitle', 'isupper', 'join', 'ljust', 'lower', 'lstrip', 'partition',
    'replace', 'rfind', 'rindex', 'rjust', 'rpartition', 'rsplit',
    'rstrip', 'split', 'splitlines', 'startswith', 'strip', 'swapcase',
    'title', 'translate', 'upper', 'zfill', 'isnumeric', 'isdecimal',
    'array', 'asarray', 'compare_chararrays', 'chararray'
    ]  # 定义包含所有方法和属性的 __all__ 列表

array_function_dispatch = functools.partial(  # 定义 array_function_dispatch 函数
    overrides.array_function_dispatch, module='numpy.char')  # 从 overrides 模块中导入 array_function_dispatch 函数

def _binary_op_dispatcher(x1, x2):
    return (x1, x2)  # 返回元组 (x1, x2)

@array_function_dispatch(_binary_op_dispatcher)  # 根据 array_function_dispatch 装饰器调用 _binary_op_dispatcher 函数
def equal(x1, x2):
    """
    Return (x1 == x2) element-wise.

    Unlike `numpy.equal`, this comparison is performed by first
    stripping whitespace characters from the end of the string.  This
    behavior is provided for backward-compatibility with numarray.

    Parameters
    ----------
    x1, x2 : array_like of str or unicode
        Input arrays of the same shape.

    Returns
    -------
    out : ndarray
        Output array of bools.

    Examples
    --------
    >>> y = "aa "
    >>> x = "aa"
    >>> np.char.equal(x, y)
    array(True)    

    See Also
    --------
    not_equal, greater_equal, less_equal, greater, less
    """
    return compare_chararrays(x1, x2, '==', True)  # 返回比较结果所得到的数组

@array_function_dispatch(_binary_op_dispatcher)  # 根据 array_function_dispatch 装饰器调用 _binary_op_dispatcher 函数
def not_equal(x1, x2):
    """
    Return (x1 != x2) element-wise.

    Unlike `numpy.not_equal`, this comparison is performed by first
    stripping whitespace characters from the end of the string.  This
    behavior is provided for backward-compatibility with numarray.
    
    Parameters
    ----------
    x1, x2 : array_like of str or unicode
        Input arrays of the same shape.
        
    Returns
    -------
    out : ndarray
        Output array of bools.
        
    See Also
    --------
    equal, greater_equal, less_equal, greater, less
    
    Examples
    --------
    >>> x1 = np.array(['a', 'b', 'c'])
    >>> np.char.not_equal(x1, 'b')
    array([ True, False,  True])
    
    """
    # 使用 compare_chararrays 函数比较两个字符串数组 x1 和 x2 的不等性，返回布尔类型的数组
    return compare_chararrays(x1, x2, '!=', True)
# 使用 array_function_dispatch 装饰器将函数注册到二进制操作调度器中
@array_function_dispatch(_binary_op_dispatcher)
# 定义函数 greater_equal，用于比较两个数组中元素是否 x1 >= x2，返回布尔值数组
def greater_equal(x1, x2):
    """
    Return (x1 >= x2) element-wise.

    Unlike `numpy.greater_equal`, this comparison is performed by
    first stripping whitespace characters from the end of the string.
    This behavior is provided for backward-compatibility with
    numarray.

    Parameters
    ----------
    x1, x2 : array_like of str or unicode
        Input arrays of the same shape.

    Returns
    -------
    out : ndarray
        Output array of bools.

    See Also
    --------
    equal, not_equal, less_equal, greater, less

    Examples
    --------
    >>> x1 = np.array(['a', 'b', 'c'])
    >>> np.char.greater_equal(x1, 'b')
    array([False,  True,  True])
    
    """
    # 调用 compare_chararrays 函数，比较 x1 和 x2 的字符数组，采用 '>=' 操作符，返回结果
    return compare_chararrays(x1, x2, '>=', True)


@array_function_dispatch(_binary_op_dispatcher)
# 定义函数 less_equal，用于比较两个数组中元素是否 x1 <= x2，返回布尔值数组
def less_equal(x1, x2):
    """
    Return (x1 <= x2) element-wise.

    Unlike `numpy.less_equal`, this comparison is performed by first
    stripping whitespace characters from the end of the string.  This
    behavior is provided for backward-compatibility with numarray.

    Parameters
    ----------
    x1, x2 : array_like of str or unicode
        Input arrays of the same shape.

    Returns
    -------
    out : ndarray
        Output array of bools.

    See Also
    --------
    equal, not_equal, greater_equal, greater, less

    Examples
    --------
    >>> x1 = np.array(['a', 'b', 'c'])
    >>> np.char.less_equal(x1, 'b')
    array([ True,  True, False])
    
    """
    # 调用 compare_chararrays 函数，比较 x1 和 x2 的字符数组，采用 '<=' 操作符，返回结果
    return compare_chararrays(x1, x2, '<=', True)


@array_function_dispatch(_binary_op_dispatcher)
# 定义函数 greater，用于比较两个数组中元素是否 x1 > x2，返回布尔值数组
def greater(x1, x2):
    """
    Return (x1 > x2) element-wise.

    Unlike `numpy.greater`, this comparison is performed by first
    stripping whitespace characters from the end of the string.  This
    behavior is provided for backward-compatibility with numarray.

    Parameters
    ----------
    x1, x2 : array_like of str or unicode
        Input arrays of the same shape.

    Returns
    -------
    out : ndarray
        Output array of bools.

    See Also
    --------
    equal, not_equal, greater_equal, less_equal, less
    
    Examples
    --------
    >>> x1 = np.array(['a', 'b', 'c'])
    >>> np.char.greater(x1, 'b')
    array([False, False,  True])
    
    """
    # 调用 compare_chararrays 函数，比较 x1 和 x2 的字符数组，采用 '>' 操作符，返回结果
    return compare_chararrays(x1, x2, '>', True)


@array_function_dispatch(_binary_op_dispatcher)
# 定义函数 less，用于比较两个数组中元素是否 x1 < x2，返回布尔值数组
def less(x1, x2):
    """
    Return (x1 < x2) element-wise.

    Unlike `numpy.greater`, this comparison is performed by first
    stripping whitespace characters from the end of the string.  This
    behavior is provided for backward-compatibility with numarray.

    Parameters
    ----------
    x1, x2 : array_like of str or unicode
        Input arrays of the same shape.

    Returns
    -------
    out : ndarray
        Output array of bools.

    See Also
    --------
    equal, not_equal, greater_equal, less_equal, greater

    Examples
    --------

    """
    # 调用 compare_chararrays 函数，比较 x1 和 x2 的字符数组，采用 '<' 操作符，返回结果
    return compare_chararrays(x1, x2, '<', True)
    # 创建一个包含字符串数组 ['a', 'b', 'c'] 的 NumPy 数组 x1
    x1 = np.array(['a', 'b', 'c'])
    # 使用 NumPy 的 char.less 函数比较 x1 中的每个元素是否小于 'b'，返回布尔数组
    np.char.less(x1, 'b')
    array([True, False, False])
    
    # 调用一个未在当前上下文中定义的函数 compare_chararrays，传入参数 x1, x2, '<', True
    # 该函数似乎用于比较两个字符数组，返回结果可能是两个数组的比较结果
    return compare_chararrays(x1, x2, '<', True)
def multiply(a, i):
    """
    Return (a * i), that is string multiple concatenation,
    element-wise.

    Values in ``i`` of less than 0 are treated as 0 (which yields an
    empty string).

    Parameters
    ----------
    a : array_like, with `np.bytes_` or `np.str_` dtype
        Input array of strings or bytes.

    i : array_like, with any integer dtype
        Multiplier array. Each element specifies how many times
        to repeat the corresponding element in `a`.

    Returns
    -------
    out : ndarray
        Output array of str or unicode, depending on input types.
        The shape of `out` matches the shape of `a`.

    Notes
    -----
    This function is a wrapper around np.strings.multiply and raises
    ValueError if `i` contains non-integer values.

    Examples
    --------
    >>> a = np.array(["a", "b", "c"])
    >>> np.strings.multiply(a, 3)
    array(['aaa', 'bbb', 'ccc'], dtype='<U3')
    >>> i = np.array([1, 2, 3])
    >>> np.strings.multiply(a, i)
    array(['a', 'bb', 'ccc'], dtype='<U3')
    >>> np.strings.multiply(np.array(['a']), i)
    array(['a', 'aa', 'aaa'], dtype='<U3')
    >>> a = np.array(['a', 'b', 'c', 'd', 'e', 'f']).reshape((2, 3))
    >>> np.strings.multiply(a, 3)
    array([['aaa', 'bbb', 'ccc'],
           ['ddd', 'eee', 'fff']], dtype='<U3')
    >>> np.strings.multiply(a, i)
    array([['a', 'bb', 'ccc'],
           ['d', 'ee', 'fff']], dtype='<U3')

    """
    try:
        # Call np.strings.multiply to perform element-wise string multiplication
        return strings_multiply(a, i)
    except TypeError:
        # Raise ValueError if `i` contains non-integer values
        raise ValueError("Can only multiply by integers")


def partition(a, sep):
    """
    Partition each element in `a` around `sep`.

    Calls :meth:`str.partition` element-wise.

    For each element in `a`, split the element as the first
    occurrence of `sep`, and return 3 strings containing the part
    before the separator, the separator itself, and the part after
    the separator. If the separator is not found, return 3 strings
    containing the string itself, followed by two empty strings.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Input array containing strings or bytes.
    sep : {str, unicode}
        Separator to split each string element in `a`.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types. The output array will have an extra
        dimension with 3 elements per input element.

    Examples
    --------
    >>> x = np.array(["Numpy is nice!"])
    >>> np.char.partition(x, " ")
    array([['Numpy', ' ', 'is nice!']], dtype='<U8')
    
    See Also
    --------
    str.partition

    """
    # Use np.stack to partition each element of `a` around `sep` and stack the results
    return np.stack(strings_partition(a, sep), axis=-1)


def rpartition(a, sep):
    """
    Partition (split) each element around the right-most separator.

    Calls :meth:`str.rpartition` element-wise.

    For each element in `a`, split the element as the last
    occurrence of `sep`, and return 3 strings containing the part
    before the separator, the separator itself, and the part after
    the separator. If the separator is not found, return 3 strings
    containing the entire string followed by two empty strings.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Input array containing strings or bytes.
    sep : {str, unicode}
        Separator to split each string element in `a`.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types. The output array will have an extra
        dimension with 3 elements per input element.

    """
    # 使用 np.stack 将字符串数组 a 中每个元素根据右边最后出现的分隔符 sep 进行分割，并将结果堆叠成一个多维数组
    return np.stack(strings_rpartition(a, sep), axis=-1)
# 设置模块名为 "numpy.char"
@set_module("numpy.char")
# 定义 chararray 类，继承自 ndarray 类
class chararray(ndarray):
    """
    chararray(shape, itemsize=1, unicode=False, buffer=None, offset=0,
              strides=None, order=None)

    Provides a convenient view on arrays of string and unicode values.

    .. note::
       The `chararray` class exists for backwards compatibility with
       Numarray, it is not recommended for new development. Starting from numpy
       1.4, if one needs arrays of strings, it is recommended to use arrays of
       `dtype` `~numpy.object_`, `~numpy.bytes_` or `~numpy.str_`, and use
       the free functions in the `numpy.char` module for fast vectorized
       string operations.

    Versus a NumPy array of dtype `~numpy.bytes_` or `~numpy.str_`, this
    class adds the following functionality:

    1) values automatically have whitespace removed from the end
       when indexed

    2) comparison operators automatically remove whitespace from the
       end when comparing values

    3) vectorized string operations are provided as methods
       (e.g. `.endswith`) and infix operators (e.g. ``"+", "*", "%"``)

    chararrays should be created using `numpy.char.array` or
    `numpy.char.asarray`, rather than this constructor directly.

    This constructor creates the array, using `buffer` (with `offset`
    and `strides`) if it is not ``None``. If `buffer` is ``None``, then
    constructs a new array with `strides` in "C order", unless both
    ``len(shape) >= 2`` and ``order='F'``, in which case `strides`
    is in "Fortran order".

    Methods
    -------
    astype
    argsort
    copy
    count
    decode
    dump
    dumps
    encode
    endswith
    expandtabs
    fill
    find
    flatten
    getfield
    index
    isalnum
    isalpha
    isdecimal
    isdigit
    islower
    isnumeric
    isspace
    istitle
    isupper
    item
    join
    ljust
    lower
    lstrip
    nonzero
    put
    ravel
    repeat
    replace
    reshape
    resize
    rfind
    rindex
    rjust
    rsplit
    rstrip
    searchsorted
    setfield
    setflags
    sort
    split
    splitlines
    squeeze
    startswith
    strip
    swapaxes
    swapcase
    take
    title
    tofile
    tolist
    tostring
    translate
    transpose
    upper
    view
    zfill

    Parameters
    ----------
    shape : tuple
        Shape of the array.
    itemsize : int, optional
        Length of each array element, in number of characters. Default is 1.
    unicode : bool, optional
        Are the array elements of type unicode (True) or string (False).
        Default is False.
    buffer : object exposing the buffer interface or str, optional
        Memory address of the start of the array data.  Default is None,
        in which case a new array is created.
    offset : int, optional
        Fixed stride displacement from the beginning of an axis?
        Default is 0. Needs to be >=0.
    """
    strides : array_like of ints, optional
        Strides for the array (see `~numpy.ndarray.strides` for
        full description). Default is None.
    order : {'C', 'F'}, optional
        The order in which the array data is stored in memory: 'C' ->
        "row major" order (the default), 'F' -> "column major"
        (Fortran) order.

    Examples
    --------
    >>> charar = np.char.chararray((3, 3))
    >>> charar[:] = 'a'
    >>> charar
    chararray([[b'a', b'a', b'a'],
               [b'a', b'a', b'a'],
               [b'a', b'a', b'a']], dtype='|S1')

    >>> charar = np.char.chararray(charar.shape, itemsize=5)
    >>> charar[:] = 'abc'
    >>> charar
    chararray([[b'abc', b'abc', b'abc'],
               [b'abc', b'abc', b'abc'],
               [b'abc', b'abc', b'abc']], dtype='|S5')

    """
    # 定义一个新的特殊数组类型 chararray，继承自 ndarray
    def __new__(subtype, shape, itemsize=1, unicode=False, buffer=None,
                offset=0, strides=None, order='C'):
        # 如果 unicode 为 True，则数据类型为 str_，否则为 bytes_
        if unicode:
            dtype = str_
        else:
            dtype = bytes_

        # 将 itemsize 强制转换为 Python 的整数类型，因为使用 NumPy 整数类型会导致 itemsize.itemsize 作为新数组中字符串的大小。
        itemsize = int(itemsize)

        if isinstance(buffer, str):
            # Unicode 对象没有缓冲区接口
            filler = buffer
            buffer = None
        else:
            filler = None

        if buffer is None:
            # 创建一个新的 ndarray 对象，使用指定的形状、数据类型和存储顺序（默认 'C'）
            self = ndarray.__new__(subtype, shape, (dtype, itemsize),
                                   order=order)
        else:
            # 创建一个新的 ndarray 对象，使用指定的形状、数据类型、缓冲区、偏移量、步幅和存储顺序
            self = ndarray.__new__(subtype, shape, (dtype, itemsize),
                                   buffer=buffer,
                                   offset=offset, strides=strides,
                                   order=order)
        if filler is not None:
            # 将数组的所有元素填充为指定的 filler 值
            self[...] = filler

        return self

    # 当调用 ufunc（和其他一些函数）时，如果输出是类似字符串的数组，则返回 chararray；否则返回 ndarray
    def __array_wrap__(self, arr, context=None, return_scalar=False):
        if arr.dtype.char in "SUbc":
            return arr.view(type(self))
        return arr

    # 在初始化时，检查数据类型是否为 'VSUbc'，否则抛出 ValueError 异常
    def __array_finalize__(self, obj):
        if self.dtype.char not in 'VSUbc':
            raise ValueError("Can only create a chararray from string data.")

    # 获取数组的元素，如果是字符类型，则去除右侧的空格并返回；否则返回原始值
    def __getitem__(self, obj):
        val = ndarray.__getitem__(self, obj)
        if isinstance(val, character):
            return val.rstrip()
        return val

    # 实现说明：这个类的大多数方法都是直接委托给本模块中的自由函数。
    # 然而，那些返回字符串数组的方法应该返回 chararray，因此需要额外的包装。
    def __eq__(self, other):
        """
        Return (self == other) element-wise.

        See Also
        --------
        equal
        """
        return equal(self, other)

    def __ne__(self, other):
        """
        Return (self != other) element-wise.

        See Also
        --------
        not_equal
        """
        return not_equal(self, other)

    def __ge__(self, other):
        """
        Return (self >= other) element-wise.

        See Also
        --------
        greater_equal
        """
        return greater_equal(self, other)

    def __le__(self, other):
        """
        Return (self <= other) element-wise.

        See Also
        --------
        less_equal
        """
        return less_equal(self, other)

    def __gt__(self, other):
        """
        Return (self > other) element-wise.

        See Also
        --------
        greater
        """
        return greater(self, other)

    def __lt__(self, other):
        """
        Return (self < other) element-wise.

        See Also
        --------
        less
        """
        return less(self, other)

    def __add__(self, other):
        """
        Return (self + other), that is string concatenation,
        element-wise for a pair of array_likes of str or unicode.

        See Also
        --------
        add
        """
        return add(self, other)

    def __radd__(self, other):
        """
        Return (other + self), that is string concatenation,
        element-wise for a pair of array_likes of `bytes_` or `str_`.

        See Also
        --------
        add
        """
        return add(other, self)

    def __mul__(self, i):
        """
        Return (self * i), that is string multiple concatenation,
        element-wise.

        See Also
        --------
        multiply
        """
        return asarray(multiply(self, i))

    def __rmul__(self, i):
        """
        Return (self * i), that is string multiple concatenation,
        element-wise.

        See Also
        --------
        multiply
        """
        return asarray(multiply(self, i))

    def __mod__(self, i):
        """
        Return (self % i), that is pre-Python 2.6 string formatting
        (interpolation), element-wise for a pair of array_likes of `bytes_`
        or `str_`.

        See Also
        --------
        mod
        """
        return asarray(mod(self, i))

    def __rmod__(self, other):
        """
        Not implemented for the right-hand operand.

        See Also
        --------
        NotImplemented
        """
        return NotImplemented
    def argsort(self, axis=-1, kind=None, order=None):
        """
        Return the indices that sort the array lexicographically.

        For full documentation see `numpy.argsort`, for which this method is
        in fact merely a "thin wrapper."

        Examples
        --------
        >>> c = np.array(['a1b c', '1b ca', 'b ca1', 'Ca1b'], 'S5')
        >>> c = c.view(np.char.chararray); c
        chararray(['a1b c', '1b ca', 'b ca1', 'Ca1b'],
              dtype='|S5')
        >>> c[c.argsort()]
        chararray(['1b ca', 'Ca1b', 'a1b c', 'b ca1'],
              dtype='|S5')

        """
        # 使用 __array__ 方法将对象转换为 ndarray 后调用其 argsort 方法
        return self.__array__().argsort(axis, kind, order)
    argsort.__doc__ = ndarray.argsort.__doc__

    def capitalize(self):
        """
        Return a copy of `self` with only the first character of each element
        capitalized.

        See Also
        --------
        char.capitalize

        """
        # 调用 numpy.char.capitalize 函数，将每个元素的首字母大写
        return asarray(capitalize(self))

    def center(self, width, fillchar=' '):
        """
        Return a copy of `self` with its elements centered in a
        string of length `width`.

        See Also
        --------
        center
        """
        # 调用 numpy.char.center 函数，将每个元素居中在长度为 width 的字符串中
        return asarray(center(self, width, fillchar))

    def count(self, sub, start=0, end=None):
        """
        Returns an array with the number of non-overlapping occurrences of
        substring `sub` in the range [`start`, `end`].

        See Also
        --------
        char.count

        """
        # 调用 numpy.char.count 函数，统计每个元素中子字符串 sub 的出现次数
        return count(self, sub, start, end)

    def decode(self, encoding=None, errors=None):
        """
        Calls ``bytes.decode`` element-wise.

        See Also
        --------
        char.decode

        """
        # 调用 numpy.char.decode 函数，对每个元素进行 bytes 解码
        return decode(self, encoding, errors)

    def encode(self, encoding=None, errors=None):
        """
        Calls :meth:`str.encode` element-wise.

        See Also
        --------
        char.encode

        """
        # 调用 numpy.char.encode 函数，对每个元素进行字符串编码
        return encode(self, encoding, errors)

    def endswith(self, suffix, start=0, end=None):
        """
        Returns a boolean array which is `True` where the string element
        in `self` ends with `suffix`, otherwise `False`.

        See Also
        --------
        char.endswith

        """
        # 调用 numpy.char.endswith 函数，判断每个元素是否以 suffix 结尾
        return endswith(self, suffix, start, end)

    def expandtabs(self, tabsize=8):
        """
        Return a copy of each string element where all tab characters are
        replaced by one or more spaces.

        See Also
        --------
        char.expandtabs

        """
        # 调用 numpy.char.expandtabs 函数，将每个元素中的制表符替换为空格
        return asarray(expandtabs(self, tabsize))

    def find(self, sub, start=0, end=None):
        """
        For each element, return the lowest index in the string where
        substring `sub` is found.

        See Also
        --------
        char.find

        """
        # 调用 numpy.char.find 函数，返回每个元素中子字符串 sub 的最低索引位置
        return find(self, sub, start, end)
    def index(self, sub, start=0, end=None):
        """
        在字符串中查找子字符串 `sub`，如果找到则返回第一次出现的位置索引；如果未找到，抛出 `ValueError` 异常。

        See Also
        --------
        char.index
        """
        return index(self, sub, start, end)

    def isalnum(self):
        """
        如果字符串中的所有字符都是字母数字并且至少有一个字符，则返回 True；否则返回 False。

        See Also
        --------
        char.isalnum
        """
        return isalnum(self)

    def isalpha(self):
        """
        如果字符串中的所有字符都是字母并且至少有一个字符，则返回 True；否则返回 False。

        See Also
        --------
        char.isalpha
        """
        return isalpha(self)

    def isdigit(self):
        """
        如果字符串中的所有字符都是数字并且至少有一个字符，则返回 True；否则返回 False。

        See Also
        --------
        char.isdigit
        """
        return isdigit(self)

    def islower(self):
        """
        如果字符串中所有大小写字符都是小写并且至少有一个大小写字符，则返回 True；否则返回 False。

        See Also
        --------
        char.islower
        """
        return islower(self)

    def isspace(self):
        """
        如果字符串中只包含空白字符并且至少有一个字符，则返回 True；否则返回 False。

        See Also
        --------
        char.isspace
        """
        return isspace(self)

    def istitle(self):
        """
        如果字符串是标题格式（即每个单词首字母大写）并且至少有一个字符，则返回 True；否则返回 False。

        See Also
        --------
        char.istitle
        """
        return istitle(self)

    def isupper(self):
        """
        如果字符串中所有大小写字符都是大写并且至少有一个大小写字符，则返回 True；否则返回 False。

        See Also
        --------
        char.isupper
        """
        return isupper(self)

    def join(self, seq):
        """
        将序列 `seq` 中的字符串连接成一个字符串并返回。

        See Also
        --------
        char.join
        """
        return join(self, seq)

    def ljust(self, width, fillchar=' '):
        """
        返回一个数组，其中 `self` 中的元素左对齐在长度为 `width` 的字符串中。

        See Also
        --------
        char.ljust
        """
        return asarray(ljust(self, width, fillchar))
    def lower(self):
        """
        Return an array with the elements of `self` converted to
        lowercase.

        See Also
        --------
        char.lower
        """
        # 将 `self` 中的每个元素转换为小写，并返回转换后的数组
        return asarray(lower(self))

    def lstrip(self, chars=None):
        """
        For each element in `self`, return a copy with the leading characters
        removed.

        See Also
        --------
        char.lstrip
        """
        # 对 `self` 中的每个元素进行去除开头指定字符后返回副本的操作
        return lstrip(self, chars)

    def partition(self, sep):
        """
        Partition each element in `self` around `sep`.

        See Also
        --------
        partition
        """
        # 对 `self` 中的每个元素使用 `sep` 进行分割，并返回分割后的数组
        return asarray(partition(self, sep))

    def replace(self, old, new, count=None):
        """
        For each element in `self`, return a copy of the string with all
        occurrences of substring `old` replaced by `new`.

        See Also
        --------
        char.replace
        """
        # 对 `self` 中的每个元素进行字符串替换操作，返回替换后的副本
        return replace(self, old, new, count if count is not None else -1)

    def rfind(self, sub, start=0, end=None):
        """
        For each element in `self`, return the highest index in the string
        where substring `sub` is found, such that `sub` is contained
        within [`start`, `end`].

        See Also
        --------
        char.rfind
        """
        # 返回 `self` 中每个元素中指定子串 `sub` 的最高索引，范围为 [`start`, `end`]
        return rfind(self, sub, start, end)

    def rindex(self, sub, start=0, end=None):
        """
        Like `rfind`, but raises :exc:`ValueError` when the substring `sub` is
        not found.

        See Also
        --------
        char.rindex
        """
        # 类似 `rfind`，但是当子串 `sub` 未找到时会引发 :exc:`ValueError`
        return rindex(self, sub, start, end)

    def rjust(self, width, fillchar=' '):
        """
        Return an array with the elements of `self`
        right-justified in a string of length `width`.

        See Also
        --------
        char.rjust
        """
        # 返回 `self` 中的每个元素右对齐到指定 `width` 的字符串中，并返回数组
        return asarray(rjust(self, width, fillchar))

    def rpartition(self, sep):
        """
        Partition each element in `self` around `sep`.

        See Also
        --------
        rpartition
        """
        # 对 `self` 中的每个元素使用 `sep` 进行分割，并返回分割后的数组
        return asarray(rpartition(self, sep))

    def rsplit(self, sep=None, maxsplit=None):
        """
        For each element in `self`, return a list of the words in
        the string, using `sep` as the delimiter string.

        See Also
        --------
        char.rsplit
        """
        # 对 `self` 中的每个元素按照 `sep` 分割字符串，并返回单词列表
        return rsplit(self, sep, maxsplit)

    def rstrip(self, chars=None):
        """
        For each element in `self`, return a copy with the trailing
        characters removed.

        See Also
        --------
        char.rstrip
        """
        # 对 `self` 中的每个元素进行去除结尾指定字符后返回副本的操作
        return rstrip(self, chars)

    def split(self, sep=None, maxsplit=None):
        """
        For each element in `self`, return a list of the words in the
        string, using `sep` as the delimiter string.

        See Also
        --------
        char.split
        """
        # 对 `self` 中的每个元素按照 `sep` 分割字符串，并返回单词列表
        return split(self, sep, maxsplit)
    def splitlines(self, keepends=None):
        """
        对于 `self` 中的每个元素，返回一个列表，其中包含元素的行，以行界限分割。

        See Also
        --------
        char.splitlines

        """
        return splitlines(self, keepends)

    def startswith(self, prefix, start=0, end=None):
        """
        返回一个布尔数组，其中 `self` 中的字符串元素在 `prefix` 开头时为 `True`，否则为 `False`。

        See Also
        --------
        char.startswith

        """
        return startswith(self, prefix, start, end)

    def strip(self, chars=None):
        """
        对于 `self` 中的每个元素，返回一个副本，其中移除了前导和尾随的字符。

        See Also
        --------
        char.strip

        """
        return strip(self, chars)

    def swapcase(self):
        """
        对于 `self` 中的每个元素，返回一个字符串副本，其中大写字符转换为小写，小写字符转换为大写。

        See Also
        --------
        char.swapcase

        """
        return asarray(swapcase(self))

    def title(self):
        """
        对于 `self` 中的每个元素，返回一个首字母大写的版本：单词以大写字母开头，其余的字母都小写。

        See Also
        --------
        char.title

        """
        return asarray(title(self))

    def translate(self, table, deletechars=None):
        """
        对于 `self` 中的每个元素，返回一个字符串副本，其中删除了在可选参数 `deletechars` 中出现的所有字符，
        并且剩余的字符通过给定的转换表进行映射。

        See Also
        --------
        char.translate

        """
        return asarray(translate(self, table, deletechars))

    def upper(self):
        """
        返回一个将 `self` 中的元素转换为大写的数组。

        See Also
        --------
        char.upper

        """
        return asarray(upper(self))

    def zfill(self, width):
        """
        返回一个数值字符串，左侧填充零，长度为 `width`。

        See Also
        --------
        char.zfill

        """
        return asarray(zfill(self, width))

    def isnumeric(self):
        """
        对于 `self` 中的每个元素，如果元素仅包含数字字符，则返回 `True`。

        See Also
        --------
        char.isnumeric

        """
        return isnumeric(self)

    def isdecimal(self):
        """
        对于 `self` 中的每个元素，如果元素仅包含十进制数字字符，则返回 `True`。

        See Also
        --------
        char.isdecimal

        """
        return isdecimal(self)
# 设置模块名称为 "numpy.char"
@set_module("numpy.char")
def array(obj, itemsize=None, copy=True, unicode=None, order=None):
    """
    创建一个 `~numpy.char.chararray`。

    .. note::
       该类用于向后兼容 numarray。
       新代码（不考虑 numarray 兼容性）应该使用 `bytes_` 或 `str_` 类型的数组，
       并使用 :mod:`numpy.char` 中的自由函数进行快速向量化的字符串操作。

    与 dtype 为 `bytes_` 或 `str_` 的 NumPy 数组相比，这个类增加了以下功能：

    1) 在索引时自动去除值末尾的空白字符

    2) 在比较值时，比较运算符会自动去除末尾的空白字符

    3) 提供作为方法的向量化字符串操作（例如 `chararray.endswith <numpy.char.chararray.endswith>`）
       和中缀操作符（例如 ``+, *, %``）

    Parameters
    ----------
    obj : str 或类似 unicode 的数组

    itemsize : int, optional
        `itemsize` 是结果数组中每个标量的字符数。如果 `itemsize` 为 None，
        并且 `obj` 是对象数组或 Python 列表，则 `itemsize` 将自动确定。
        如果提供了 `itemsize` 并且 `obj` 的类型是 str 或 unicode，则 `obj` 字符串将被分块为 `itemsize` 长度的片段。

    copy : bool, optional
        如果为 True（默认），则复制对象。否则，只有当 ``__array__`` 返回复制时，
        如果 obj 是嵌套序列或需要满足其他任何要求（`itemsize`、unicode、`order` 等）时才会进行复制。

    unicode : bool, optional
        当为 True 时，生成的 `~numpy.char.chararray` 可以包含 Unicode 字符；
        当为 False 时，只能包含 8 位字符。如果 unicode 为 None，并且 `obj` 是以下之一：

        - `~numpy.char.chararray`,
        - 类型为 :class:`str_` 或 :class:`bytes_` 的 ndarray
        - Python 的 :class:`str` 或 :class:`bytes` 对象，

        那么输出数组的 Unicode 设置将自动确定。

    order : {'C', 'F', 'A'}, optional
        指定数组的顺序。如果 order 为 'C'（默认），则数组将按 C 连续顺序排列（最后一个索引变化最快）。
        如果 order 为 'F'，则返回的数组将按 Fortran 连续顺序排列（第一个索引变化最快）。
        如果 order 为 'A'，则返回的数组可以是任何顺序（C 连续、Fortran 连续或甚至是不连续的）。
    """
    # 检查对象是否为 bytes 或者 str 类型
    if isinstance(obj, (bytes, str)):
        # 如果未提供 unicode 参数，则根据 obj 类型设定 unicode 变量
        if unicode is None:
            if isinstance(obj, str):
                unicode = True
            else:
                unicode = False

        # 如果未提供 itemsize 参数，则设定为 obj 的长度
        if itemsize is None:
            itemsize = len(obj)
        
        # 计算数组的形状，假设每个元素长度为 itemsize
        shape = len(obj) // itemsize
        
        # 创建一个 chararray 对象，使用给定的参数
        return chararray(shape, itemsize=itemsize, unicode=unicode,
                         buffer=obj, order=order)

    # 如果对象是 list 或 tuple 类型，则将其转换为 ndarray
    if isinstance(obj, (list, tuple)):
        obj = asarray(obj)

    # 如果对象是 ndarray 并且其元素类型是 character 的子类
    if isinstance(obj, ndarray) and issubclass(obj.dtype.type, character):
        # 如果 obj 不是 chararray 类型，则将其视图转换为 chararray
        if not isinstance(obj, chararray):
            obj = obj.view(chararray)

        # 如果未提供 itemsize 参数，则根据 obj 的 itemsize 设定
        if itemsize is None:
            itemsize = obj.itemsize
            # 对于 Unicode，需要将 itemsize 除以单个 Unicode 字符的大小（NumPy 默认为 4）
            if issubclass(obj.dtype.type, str_):
                itemsize //= 4

        # 如果未提供 unicode 参数，则根据 obj 的 dtype 设定
        if unicode is None:
            if issubclass(obj.dtype.type, str_):
                unicode = True
            else:
                unicode = False

        # 根据 unicode 参数设定 dtype 类型
        if unicode:
            dtype = str_
        else:
            dtype = bytes_

        # 如果提供了 order 参数，则使用 asnarray 函数重新生成数组
        if order is not None:
            obj = asarray(obj, order=order)

        # 如果需要复制，或者 itemsize 不等于 obj 的 itemsize，或者类型不匹配，则进行数据类型转换
        if (copy or
                (itemsize != obj.itemsize) or
                (not unicode and isinstance(obj, str_)) or
                (unicode and isinstance(obj, bytes_))):
            obj = obj.astype((dtype, int(itemsize)))

        return obj

    # 如果对象是 ndarray 并且其元素类型是 object 的子类
    if isinstance(obj, ndarray) and issubclass(obj.dtype.type, object):
        # 如果未提供 itemsize 参数，则将输入数组转换为列表，以便 ndarray 构造函数自动确定 itemsize
        if itemsize is None:
            obj = obj.tolist()
            # 继续默认情况处理

    # 根据 unicode 参数设定 dtype 类型
    if unicode:
        dtype = str_
    else:
        dtype = bytes_

    # 如果未提供 itemsize 参数，则使用 narray 函数生成数组
    if itemsize is None:
        val = array(obj, dtype=dtype, order=order, subok=True)
    else:
        val = array(obj, dtype=(dtype, itemsize), order=order, subok=True)

    # 返回 val 视图，类型为 chararray
    return val.view(chararray)
# 将输入对象转换为 `~numpy.char.chararray`，仅在必要时复制数据。
@set_module("numpy.char")
def asarray(obj, itemsize=None, unicode=None, order=None):
    """
    Convert the input to a `~numpy.char.chararray`, copying the data only if
    necessary.

    Versus a NumPy array of dtype `bytes_` or `str_`, this
    class adds the following functionality:

    1) values automatically have whitespace removed from the end
       when indexed

    2) comparison operators automatically remove whitespace from the
       end when comparing values

    3) vectorized string operations are provided as methods
       (e.g. `chararray.endswith <numpy.char.chararray.endswith>`)
       and infix operators (e.g. ``+``, ``*``, ``%``)

    Parameters
    ----------
    obj : array of str or unicode-like
        输入的字符串数组或类似 unicode 的对象

    itemsize : int, optional
        `itemsize` 是结果数组中每个标量的字符数。如果 `itemsize` 为 None，并且 `obj` 是对象数组或 Python 列表，
        `itemsize` 将自动确定。如果提供了 `itemsize` 并且 `obj` 的类型为 str 或 unicode，则 `obj` 字符串将被分成 `itemsize` 段。

    unicode : bool, optional
        当为 True 时，生成的 `~numpy.char.chararray` 可包含 Unicode 字符；当为 False 时，仅包含 8 位字符。
        如果 unicode 为 None，并且 `obj` 是以下类型之一：

        - `~numpy.char.chararray`
        - 类型为 `str_` 或 `unicode_` 的 ndarray
        - Python 的 str 或 unicode 对象

        输出数组的 Unicode 设置将自动确定。

    order : {'C', 'F'}, optional
        指定数组的顺序。如果 order 为 'C'（默认），则数组将以 C 连续顺序存储（最后一个索引变化最快）。
        如果 order 为 'F'，则返回的数组将以 Fortran 连续顺序存储（第一个索引变化最快）。

    Examples
    --------
    >>> np.char.asarray(['hello', 'world'])
    chararray(['hello', 'world'], dtype='<U5')
    
    """
    # 调用 numpy 的 array 函数，返回相应的 chararray 对象
    return array(obj, itemsize, copy=False,
                 unicode=unicode, order=order)
```