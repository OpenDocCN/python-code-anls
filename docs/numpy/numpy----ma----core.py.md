# `.\numpy\numpy\ma\core.py`

```
"""
numpy.ma : 处理缺失或无效值的包。

该包最初由Lawrence Livermore National Laboratory的Paul F. Dubois为numarray编写。
在2006年，Pierre Gerard-Marchant（乔治亚大学）彻底重写了该包，使MaskedArray类成为ndarray的子类，并改进了对结构化数组的支持。

版权 1999年，2000年，2001年加利福尼亚大学理事会。
可无限制重新分发。

* 适配为numpy_core 2005年由Travis Oliphant和Paul Dubois（主要）完成。
* 2006年由Pierre Gerard-Marchant（pgmdevlist_AT_gmail_DOT_com）将基础的`ndarray`子类化。
* Reggie Dugard（reggie_AT_merfinllc_DOT_com）建议的改进。

.. moduleauthor:: Pierre Gerard-Marchant

"""

# pylint: disable-msg=E1002
# 导入内置模块和函数
import builtins
import inspect
import operator
import warnings
import textwrap
import re
# 导入 functools 模块中的 reduce 函数
from functools import reduce
# 导入类型提示模块中的 Dict 类型
from typing import Dict

# 导入 numpy 库，并分别导入 numpy 的子模块
import numpy as np
import numpy._core.umath as umath
import numpy._core.numerictypes as ntypes
# 导入 numpy 的 multiarray 模块并重命名为 mu
from numpy._core import multiarray as mu
# 从 numpy 中导入 ndarray, amax, amin, iscomplexobj, bool_, _NoValue, angle 函数
from numpy import ndarray, amax, amin, iscomplexobj, bool_, _NoValue, angle
# 从 numpy 中导入 array 函数并重命名为 narray, expand_dims, iinfo, finfo
from numpy import array as narray, expand_dims, iinfo, finfo
# 从 numpy 的 _core.numeric 模块中导入 normalize_axis_tuple 函数
from numpy._core.numeric import normalize_axis_tuple
# 导入 numpy 的 _utils._inspect 模块中的 getargspec, formatargspec 函数
from numpy._utils._inspect import getargspec, formatargspec
# 从 numpy 的 _utils 模块中导入 set_module 函数

from numpy._utils import set_module

# 定义 __all__ 变量，指定公开的接口列表
__all__ = [
    'MAError', 'MaskError', 'MaskType', 'MaskedArray', 'abs', 'absolute',
    'add', 'all', 'allclose', 'allequal', 'alltrue', 'amax', 'amin',
    'angle', 'anom', 'anomalies', 'any', 'append', 'arange', 'arccos',
    'arccosh', 'arcsin', 'arcsinh', 'arctan', 'arctan2', 'arctanh',
    'argmax', 'argmin', 'argsort', 'around', 'array', 'asanyarray',
    'asarray', 'bitwise_and', 'bitwise_or', 'bitwise_xor', 'bool_', 'ceil',
    'choose', 'clip', 'common_fill_value', 'compress', 'compressed',
    'concatenate', 'conjugate', 'convolve', 'copy', 'correlate', 'cos', 'cosh',
    'count', 'cumprod', 'cumsum', 'default_fill_value', 'diag', 'diagonal',
    'diff', 'divide', 'empty', 'empty_like', 'equal', 'exp',
    'expand_dims', 'fabs', 'filled', 'fix_invalid', 'flatten_mask',
    'flatten_structured_array', 'floor', 'floor_divide', 'fmod',
    'frombuffer', 'fromflex', 'fromfunction', 'getdata', 'getmask',
    'getmaskarray', 'greater', 'greater_equal', 'harden_mask', 'hypot',
    'identity', 'ids', 'indices', 'inner', 'innerproduct', 'isMA',
    'isMaskedArray', 'is_mask', 'is_masked', 'isarray', 'left_shift',
    'less', 'less_equal', 'log', 'log10', 'log2',
    'logical_and', 'logical_not', 'logical_or', 'logical_xor', 'make_mask',
    'make_mask_descr', 'make_mask_none', 'mask_or', 'masked',
    'masked_array', 'masked_equal', 'masked_greater',
    'masked_greater_equal', 'masked_inside', 'masked_invalid',
    'masked_less', 'masked_less_equal', 'masked_not_equal',
    'masked_object', 'masked_outside', 'masked_print_option',
    'masked_singleton', 'masked_values', 'masked_where', 'max', 'maximum',
    # 定义一个列表，包含各种数学和数组操作的函数名字符串
    [
        'maximum_fill_value', 'mean', 'min', 'minimum', 'minimum_fill_value',
        'mod', 'multiply', 'mvoid', 'ndim', 'negative', 'nomask', 'nonzero',
        'not_equal', 'ones', 'ones_like', 'outer', 'outerproduct', 'power', 'prod',
        'product', 'ptp', 'put', 'putmask', 'ravel', 'remainder',
        'repeat', 'reshape', 'resize', 'right_shift', 'round', 'round_',
        'set_fill_value', 'shape', 'sin', 'sinh', 'size', 'soften_mask',
        'sometrue', 'sort', 'sqrt', 'squeeze', 'std', 'subtract', 'sum',
        'swapaxes', 'take', 'tan', 'tanh', 'trace', 'transpose', 'true_divide',
        'var', 'where', 'zeros', 'zeros_like',
    ]
MaskType = np.bool
nomask = MaskType(0)

class MaskedArrayFutureWarning(FutureWarning):
    pass

def _deprecate_argsort_axis(arr):
    """
    Adjust the axis passed to argsort, warning if necessary

    Parameters
    ----------
    arr
        The array which argsort was called on

    np.ma.argsort has a long-term bug where the default of the axis argument
    is wrong (gh-8701), which now must be kept for backwards compatibility.
    Thankfully, this only makes a difference when arrays are 2- or more-
    dimensional, so we only need a warning then.
    """
    if arr.ndim <= 1:
        # 如果数组维度小于等于1，不需要警告，但是切换到 -1，避免意外情况，
        # 例如标量轴的子类实现可能性更大。
        return -1
    else:
        # 2017-04-11, Numpy 1.13.0, gh-8701: 在默认的轴参数上发出警告
        warnings.warn(
            "In the future the default for argsort will be axis=-1, not the "
            "current None, to match its documentation and np.argsort. "
            "Explicitly pass -1 or None to silence this warning.",
            MaskedArrayFutureWarning, stacklevel=3)
        return None


def doc_note(initialdoc, note):
    """
    Adds a Notes section to an existing docstring.

    """
    if initialdoc is None:
        return
    if note is None:
        return initialdoc

    notesplit = re.split(r'\n\s*?Notes\n\s*?-----', inspect.cleandoc(initialdoc))
    notedoc = "\n\nNotes\n-----\n%s\n" % inspect.cleandoc(note)

    return ''.join(notesplit[:1] + [notedoc] + notesplit[1:])


def get_object_signature(obj):
    """
    Get the signature from obj

    """
    try:
        sig = formatargspec(*getargspec(obj))
    except TypeError:
        sig = ''
    return sig


###############################################################################
#                              Exceptions                                     #
###############################################################################


class MAError(Exception):
    """
    Class for masked array related errors.

    """
    pass


class MaskError(MAError):
    """
    Class for mask related errors.

    """
    pass


###############################################################################
#                           Filling options                                   #
###############################################################################


# b: boolean - c: complex - f: floats - i: integer - O: object - S: string
default_filler = {'b': True,
                  'c': 1.e20 + 0.0j,
                  'f': 1.e20,
                  'i': 999999,
                  'O': '?',
                  'S': b'N/A',
                  'u': 999999,
                  'V': b'???',
                  'U': 'N/A'
                  }

# Add datetime64 and timedelta64 types
for v in ["Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns", "ps",
          "fs", "as"]:
    # 将 datetime64 和 timedelta64 类型添加到 default_filler 字典中
    default_filler["M8[" + v + "]"] = np.datetime64("NaT", v)
    # 为默认填充器字典添加键为 "m8[" + v + "]" 的条目，其值为 np.timedelta64("NaT", v)
    default_filler["m8[" + v + "]"] = np.timedelta64("NaT", v)
# 定义包含所有浮点数类型的列表
float_types_list = [np.half, np.single, np.double, np.longdouble,
                    np.csingle, np.cdouble, np.clongdouble]

# 初始化最小值和最大值字典，键为类型，值为整数
_minvals: Dict[type, int] = {}
_maxvals: Dict[type, int] = {}

# 遍历 NumPy 类型字典中的所有标量类型
for sctype in ntypes.sctypeDict.values():
    # 获取当前标量类型的 NumPy 数据类型对象
    scalar_dtype = np.dtype(sctype)

    # 根据数据类型的种类分类处理
    if scalar_dtype.kind in "Mm":
        # 对于日期时间类型，获取其整数表示的最小值和最大值
        info = np.iinfo(np.int64)
        min_val, max_val = info.min, info.max
    elif np.issubdtype(scalar_dtype, np.integer):
        # 对于整数类型，获取其具体类型的最小值和最大值
        info = np.iinfo(sctype)
        min_val, max_val = info.min, info.max
    elif np.issubdtype(scalar_dtype, np.floating):
        # 对于浮点数类型，获取其具体类型的最小值和最大值
        info = np.finfo(sctype)
        min_val, max_val = info.min, info.max
    elif scalar_dtype.kind == "b":
        # 对于布尔类型，设置最小值和最大值为 0 和 1
        min_val, max_val = 0, 1
    else:
        # 对于其他类型，最小值和最大值置为 None
        min_val, max_val = None, None

    # 将最小值和最大值存入对应的字典中
    _minvals[sctype] = min_val
    _maxvals[sctype] = max_val

# 设置最大填充值为最小值字典的副本，并更新浮点数类型的键值对为负无穷
max_filler = _minvals
max_filler.update([(k, -np.inf) for k in float_types_list[:4]])
max_filler.update([(k, complex(-np.inf, -np.inf)) for k in float_types_list[-3:]])

# 设置最小填充值为最大值字典的副本，并更新浮点数类型的键值对为正无穷
min_filler = _maxvals
min_filler.update([(k,  +np.inf) for k in float_types_list[:4]])
min_filler.update([(k, complex(+np.inf, +np.inf)) for k in float_types_list[-3:]])

# 删除不再需要的浮点数类型列表
del float_types_list

def _recursive_fill_value(dtype, f):
    """
    Recursively produce a fill value for `dtype`, calling f on scalar dtypes
    """
    if dtype.names is not None:
        # 如果数据类型是结构化类型，递归处理每个字段
        # 在这里使用 `array` 包装确保使用 NumPy 的转换规则，
        # 对于整数转换，允许使用 99999 作为 int8 的填充值。
        vals = tuple(
                np.array(_recursive_fill_value(dtype[name], f))
                for name in dtype.names)
        return np.array(vals, dtype=dtype)[()]  # 从 0 维降为 void 标量
    elif dtype.subdtype:
        # 如果数据类型是子类型，递归处理其子类型和形状
        subtype, shape = dtype.subdtype
        subval = _recursive_fill_value(subtype, f)
        return np.full(shape, subval)
    else:
        # 否则直接返回 f 函数应用后的结果
        return f(dtype)

def _get_dtype_of(obj):
    """ Convert the argument for *_fill_value into a dtype """
    if isinstance(obj, np.dtype):
        # 如果对象是 NumPy 数据类型，则直接返回
        return obj
    elif hasattr(obj, 'dtype'):
        # 如果对象有 dtype 属性，则返回其 dtype
        return obj.dtype
    else:
        # 否则将对象转换为数组并返回其数据类型
        return np.asanyarray(obj).dtype

def default_fill_value(obj):
    """
    Return the default fill value for the argument object.

    The default filling value depends on the datatype of the input
    array or the type of the input scalar:

       ========  ========
       datatype  default
       ========  ========
       bool      True
       int       999999
       float     1.e20
       complex   1.e20+0j
       object    '?'
       string    'N/A'
       ========  ========

    For structured types, a structured scalar is returned, with each field the
    default fill value for its type.

    For subarray types, the fill value is an array of the same size containing
    the default scalar fill value.

    Parameters
    ----------
    obj : array_like or scalar
        Input object to determine the default fill value.
    """
    返回给定数组数据类型或标量的默认填充值。

    Parameters
    ----------
    obj : ndarray, dtype or scalar
        要返回默认填充值的数组数据类型或标量。

    Returns
    -------
    fill_value : scalar
        默认填充值。

    Examples
    --------
    >>> np.ma.default_fill_value(1)
    999999
    >>> np.ma.default_fill_value(np.array([1.1, 2., np.pi]))
    1e+20
    >>> np.ma.default_fill_value(np.dtype(complex))
    (1e+20+0j)

    """
    # 定义一个函数，根据数据类型返回标量的默认填充值
    def _scalar_fill_value(dtype):
        if dtype.kind in 'Mm':
            return default_filler.get(dtype.str[1:], '?')
        else:
            return default_filler.get(dtype.kind, '?')

    # 获取对象的数据类型
    dtype = _get_dtype_of(obj)
    # 返回递归调用的填充值函数的结果
    return _recursive_fill_value(dtype, _scalar_fill_value)
# 定义一个函数 `_extremum_fill_value`，用于计算对象的极值填充值
def _extremum_fill_value(obj, extremum, extremum_name):

    # 定义一个内部函数 `_scalar_fill_value`，用于返回指定数据类型的填充值
    def _scalar_fill_value(dtype):
        try:
            return extremum[dtype.type]
        except KeyError as e:
            # 若数据类型不适合计算极值，则抛出 TypeError 异常
            raise TypeError(
                f"Unsuitable type {dtype} for calculating {extremum_name}."
            ) from None

    # 获取对象的数据类型
    dtype = _get_dtype_of(obj)
    # 调用递归函数 `_recursive_fill_value` 计算填充值，并使用 `_scalar_fill_value` 函数获取标量填充值
    return _recursive_fill_value(dtype, _scalar_fill_value)


def minimum_fill_value(obj):
    """
    Return the maximum value that can be represented by the dtype of an object.

    This function is useful for calculating a fill value suitable for
    taking the minimum of an array with a given dtype.

    Parameters
    ----------
    obj : ndarray, dtype or scalar
        An object that can be queried for it's numeric type.

    Returns
    -------
    val : scalar
        The maximum representable value.

    Raises
    ------
    TypeError
        If `obj` isn't a suitable numeric type.

    See Also
    --------
    maximum_fill_value : The inverse function.
    set_fill_value : Set the filling value of a masked array.
    MaskedArray.fill_value : Return current fill value.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> a = np.int8()
    >>> ma.minimum_fill_value(a)
    127
    >>> a = np.int32()
    >>> ma.minimum_fill_value(a)
    2147483647

    An array of numeric data can also be passed.

    >>> a = np.array([1, 2, 3], dtype=np.int8)
    >>> ma.minimum_fill_value(a)
    127
    >>> a = np.array([1, 2, 3], dtype=np.float32)
    >>> ma.minimum_fill_value(a)
    inf

    """
    return _extremum_fill_value(obj, min_filler, "minimum")


def maximum_fill_value(obj):
    """
    Return the minimum value that can be represented by the dtype of an object.

    This function is useful for calculating a fill value suitable for
    taking the maximum of an array with a given dtype.

    Parameters
    ----------
    obj : ndarray, dtype or scalar
        An object that can be queried for it's numeric type.

    Returns
    -------
    val : scalar
        The minimum representable value.

    Raises
    ------
    TypeError
        If `obj` isn't a suitable numeric type.

    See Also
    --------
    minimum_fill_value : The inverse function.
    set_fill_value : Set the filling value of a masked array.
    MaskedArray.fill_value : Return current fill value.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> a = np.int8()
    >>> ma.maximum_fill_value(a)
    -128
    >>> a = np.int32()
    >>> ma.maximum_fill_value(a)
    -2147483648

    An array of numeric data can also be passed.

    >>> a = np.array([1, 2, 3], dtype=np.int8)
    >>> ma.maximum_fill_value(a)
    -128
    >>> a = np.array([1, 2, 3], dtype=np.float32)
    >>> ma.maximum_fill_value(a)
    -inf

    """
    return _extremum_fill_value(obj, max_filler, "maximum")


def _recursive_set_fill_value(fillvalue, dt):
    """
    Create a fill value for a structured dtype.

    Parameters
    ----------
    fillvalue : scalar
        Value to use for filling.
    dt : dtype
        Data type descriptor.

    Returns
    -------
    fill_value : scalar
        The fill value created.

    """
    # 在结构化数据类型中创建填充值的递归函数
    # 返回指定数据类型的填充值
    return np.full_like(np.empty((), dtype=dt), fillvalue, dtype=dt)
    # 将 fillvalue 调整为与结构化 dtype 中字段数目相同的大小
    fillvalue = np.resize(fillvalue, len(dt.names))
    
    # 初始化一个空列表，用于存储最终输出的值
    output_value = []
    
    # 遍历结构化 dtype 中的每个字段名和对应的填充值
    for (fval, name) in zip(fillvalue, dt.names):
        # 获取当前字段的 dtype
        cdtype = dt[name]
        
        # 如果当前字段是一个子 dtype，则将 cdtype 调整为其主类型
        if cdtype.subdtype:
            cdtype = cdtype.subdtype[0]
        
        # 如果当前字段有子字段，则递归调用 _recursive_set_fill_value 处理子字段
        if cdtype.names is not None:
            output_value.append(tuple(_recursive_set_fill_value(fval, cdtype)))
        else:
            # 否则，将填充值转换为指定的 dtype，并将其作为标量添加到输出列表中
            output_value.append(np.array(fval, dtype=cdtype).item())
    
    # 返回输出列表作为一个元组，表示结构化填充值的完整结构
    return tuple(output_value)
# 私有函数，验证给定的 `fill_value` 是否与给定的 dtype 兼容

def _check_fill_value(fill_value, ndtype):
    """
    Private function validating the given `fill_value` for the given dtype.

    If fill_value is None, it is set to the default corresponding to the dtype.

    If fill_value is not None, its value is forced to the given dtype.

    The result is always a 0d array.
    """

    # 将 ndtype 转换为 numpy 的 dtype 对象
    ndtype = np.dtype(ndtype)

    # 如果 fill_value 为 None，则设置为与 dtype 对应的默认值
    if fill_value is None:
        fill_value = default_fill_value(ndtype)

    # 如果 ndtype 具有字段（即结构化 dtype），则执行以下操作
    elif ndtype.names is not None:
        # 如果 fill_value 是 ndarray 或 np.void 类型
        if isinstance(fill_value, (np.ndarray, np.void)):
            try:
                # 尝试将 fill_value 转换为指定的 dtype
                fill_value = np.asarray(fill_value, dtype=ndtype)
            except ValueError as e:
                err_msg = "Unable to transform %s to dtype %s"
                raise ValueError(err_msg % (fill_value, ndtype)) from e
        else:
            # 如果 fill_value 不是 ndarray 或 np.void 类型，转换为 object 类型
            fill_value = np.asarray(fill_value, dtype=object)
            # 递归设置 fill_value 的值，并转换为指定的 dtype
            fill_value = np.array(_recursive_set_fill_value(fill_value, ndtype),
                                  dtype=ndtype)
    else:
        # 如果 ndtype 没有字段，即非结构化 dtype 的情况
        if isinstance(fill_value, str) and (ndtype.char not in 'OSVU'):
            # 如果 fill_value 是字符串且 ndtype 的字符类型不在 'OSVU' 中，则抛出 TypeError
            err_msg = "Cannot set fill value of string with array of dtype %s"
            raise TypeError(err_msg % ndtype)
        else:
            # 否则尝试将 fill_value 转换为指定的 dtype
            try:
                fill_value = np.asarray(fill_value, dtype=ndtype)
            except (OverflowError, ValueError) as e:
                # 如果转换过程中出现 OverflowError 或 ValueError，则抛出 TypeError
                err_msg = "Cannot convert fill_value %s to dtype %s"
                raise TypeError(err_msg % (fill_value, ndtype)) from e

    # 返回一个 0 维数组作为结果
    return np.array(fill_value)


def set_fill_value(a, fill_value):
    """
    Set the filling value of a, if a is a masked array.

    This function changes the fill value of the masked array `a` in place.
    If `a` is not a masked array, the function returns silently, without
    doing anything.

    Parameters
    ----------
    a : array_like
        Input array.
    fill_value : dtype
        Filling value. A consistency test is performed to make sure
        the value is compatible with the dtype of `a`.

    Returns
    -------
    None
        Nothing returned by this function.

    See Also
    --------
    maximum_fill_value : Return the default fill value for a dtype.
    MaskedArray.fill_value : Return current fill value.
    MaskedArray.set_fill_value : Equivalent method.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> a = np.arange(5)
    >>> a
    array([0, 1, 2, 3, 4])
    >>> a = ma.masked_where(a < 3, a)
    >>> a
    masked_array(data=[--, --, --, 3, 4],
                 mask=[ True,  True,  True, False, False],
           fill_value=999999)
    """
    # 如果 `a` 是一个掩码数组（MaskedArray）对象，则设置其填充值为指定的 `fill_value`。
    def set_fill_value(a, fill_value):
        """
        >>> ma.set_fill_value(a, -999)
        >>> a
        masked_array(data=[--, --, --, 3, 4],
                     mask=[ True,  True,  True, False, False],
               fill_value=-999)
    
        如果 `a` 不是一个掩码数组，则不会发生任何改变。
    
        >>> a = list(range(5))
        >>> a
        [0, 1, 2, 3, 4]
        >>> ma.set_fill_value(a, 100)
        >>> a
        [0, 1, 2, 3, 4]
    
        >>> a = np.arange(5)
        >>> a
        array([0, 1, 2, 3, 4])
        >>> ma.set_fill_value(a, 100)
        >>> a
        array([0, 1, 2, 3, 4])
        """
        # 如果 `a` 是掩码数组，则调用其方法设置填充值为 `fill_value`
        if isinstance(a, MaskedArray):
            a.set_fill_value(fill_value)
        # 如果 `a` 不是掩码数组，则函数直接返回
        return
# 返回数组 a 的填充值（如果有的话），否则返回该类型的默认填充值
def get_fill_value(a):
    if isinstance(a, MaskedArray):
        # 如果 a 是 MaskedArray 类型，则返回其填充值
        result = a.fill_value
    else:
        # 否则调用 default_fill_value 函数获取默认填充值
        result = default_fill_value(a)
    return result


# 返回两个 MaskedArray 的公共填充值（如果有的话）
# 如果 a.fill_value == b.fill_value，则返回填充值，否则返回 None
def common_fill_value(a, b):
    t1 = get_fill_value(a)
    t2 = get_fill_value(b)
    if t1 == t2:
        # 如果两个数组的填充值相同，则返回填充值 t1（或 t2）
        return t1
    return None


# 将输入数组 a 转换为 ndarray，将掩盖值替换为 fill_value
# 如果 a 不是 MaskedArray，则返回 a 本身
# 如果 a 是 MaskedArray 且没有掩盖值，则返回 a.data
# 如果 a 是 MaskedArray 并且 fill_value 为 None，则将 fill_value 设置为 a.fill_value
def filled(a, fill_value=None):
    if hasattr(a, 'filled'):
        # 如果 a 有 filled 方法，则调用其 filled 方法
        return a.filled(fill_value)
    elif isinstance(a, ndarray):
        # 如果 a 是 ndarray，则直接返回 a
        return a
    elif isinstance(a, dict):
        # 如果 a 是字典，则转换为对象数组
        return np.array(a, 'O')
    else:
        # 否则将 a 转换为 ndarray
        return np.array(a)


# 从一组（掩盖）数组中返回 MaskedArray 的最年轻的子类
# 如果有多个同级子类，取第一个出现的
def get_masked_subclass(*arrays):
    if len(arrays) == 1:
        arr = arrays[0]
        if isinstance(arr, MaskedArray):
            # 如果数组 arr 是 MaskedArray 类型，则返回其类型
            rcls = type(arr)
        else:
            # 否则返回默认的 MaskedArray 类型
            rcls = MaskedArray
    else:
        # 创建一个列表，包含所有数组元素的类型
        arrcls = [type(a) for a in arrays]
        # 取第一个数组元素的类型作为初始类型
        rcls = arrcls[0]
        # 如果初始类型不是 MaskedArray 的子类，则将其替换为 MaskedArray
        if not issubclass(rcls, MaskedArray):
            rcls = MaskedArray
        # 遍历其余数组元素的类型
        for cls in arrcls[1:]:
            # 如果某个数组元素的类型是当前 rcls 类型的子类，则更新 rcls
            if issubclass(cls, rcls):
                rcls = cls
    # 如果 rcls 的名称是 'MaskedConstant'，则返回 MaskedArray 类型
    # 否则返回 rcls 类型
    if rcls.__name__ == 'MaskedConstant':
        return MaskedArray
    return rcls
# 返回一个Masked Array的数据作为一个ndarray
def getdata(a, subok=True):
    """
    Return the data of a masked array as an ndarray.

    Return the data of `a` (if any) as an ndarray if `a` is a ``MaskedArray``,
    else return `a` as a ndarray or subclass (depending on `subok`) if not.

    Parameters
    ----------
    a : array_like
        Input ``MaskedArray``, alternatively a ndarray or a subclass thereof.
    subok : bool
        Whether to force the output to be a `pure` ndarray (False) or to
        return a subclass of ndarray if appropriate (True, default).

    Returns
    -------
    data : ndarray
        The data of the input `a` as an ndarray.

    See Also
    --------
    getmask : Return the mask of a masked array, or nomask.
    getmaskarray : Return the mask of a masked array, or full array of False.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> a = ma.masked_equal([[1,2],[3,4]], 2)
    >>> a
    masked_array(
      data=[[1, --],
            [3, 4]],
      mask=[[False,  True],
            [False, False]],
      fill_value=2)
    >>> ma.getdata(a)
    array([[1, 2],
           [3, 4]])

    Equivalently use the ``MaskedArray`` `data` attribute.

    >>> a.data
    array([[1, 2],
           [3, 4]])

    """
    try:
        # 尝试获取MaskedArray的数据
        data = a._data
    except AttributeError:
        # 如果失败，将输入转换为ndarray
        data = np.array(a, copy=None, subok=subok)
    if not subok:
        # 如果不需要返回子类的ndarray，则返回一个普通的ndarray视图
        return data.view(ndarray)
    return data


get_data = getdata


# 返回带有无效数据掩码和填充值的输入数组
def fix_invalid(a, mask=nomask, copy=True, fill_value=None):
    """
    Return input with invalid data masked and replaced by a fill value.

    Invalid data means values of `nan`, `inf`, etc.

    Parameters
    ----------
    a : array_like
        Input array, a (subclass of) ndarray.
    mask : sequence, optional
        Mask. Must be convertible to an array of booleans with the same
        shape as `data`. True indicates a masked (i.e. invalid) data.
    copy : bool, optional
        Whether to use a copy of `a` (True) or to fix `a` in place (False).
        Default is True.
    fill_value : scalar, optional
        Value used for fixing invalid data. Default is None, in which case
        the ``a.fill_value`` is used.

    Returns
    -------
    b : MaskedArray
        The input array with invalid entries fixed.

    Notes
    -----
    A copy is performed by default.

    Examples
    --------
    >>> x = np.ma.array([1., -1, np.nan, np.inf], mask=[1] + [0]*3)
    >>> x
    masked_array(data=[--, -1.0, nan, inf],
                 mask=[ True, False, False, False],
           fill_value=1e+20)
    >>> np.ma.fix_invalid(x)
    masked_array(data=[--, -1.0, --, --],
                 mask=[ True, False,  True,  True],
           fill_value=1e+20)

    >>> fixed = np.ma.fix_invalid(x)
    >>> fixed.data
    array([ 1.e+00, -1.e+00,  1.e+20,  1.e+20])
    >>> x.data
    array([ 1., -1., nan, inf])

    """
    # 将输入转换为MaskedArray，如果已经是则不复制，可以使用掩码
    a = masked_array(a, copy=copy, mask=mask, subok=True)
    # 找到所有无效数据的掩码
    invalid = np.logical_not(np.isfinite(a._data))
    if not invalid.any():
        # 如果没有无效数据，直接返回输入数组
        return a
    # 将掩码应用于输入数组的掩码
    a._mask |= invalid
    # 如果 fill_value 为 None，则将其设为数组 a 的填充值
    if fill_value is None:
        fill_value = a.fill_value
    # 将数组 a 中索引为 invalid 的位置替换为 fill_value
    a._data[invalid] = fill_value
    # 返回更新后的数组 a
    return a
def is_string_or_list_of_strings(val):
    # 检查 val 是否为字符串或字符串列表，列表不能为空且所有元素都必须是字符串
    return (isinstance(val, str) or
            (isinstance(val, list) and val and
             builtins.all(isinstance(s, str) for s in val)))

###############################################################################
#                                  Ufuncs                                     #
###############################################################################


ufunc_domain = {}
ufunc_fills = {}


class _DomainCheckInterval:
    """
    Define a valid interval, so that :

    ``domain_check_interval(a,b)(x) == True`` where
    ``x < a`` or ``x > b``.

    """

    def __init__(self, a, b):
        "domain_check_interval(a,b)(x) = true where x < a or y > b"
        if a > b:
            (a, b) = (b, a)
        self.a = a
        self.b = b

    def __call__(self, x):
        "Execute the call behavior."
        # nans at masked positions cause RuntimeWarnings, even though
        # they are masked. To avoid this we suppress warnings.
        with np.errstate(invalid='ignore'):
            return umath.logical_or(umath.greater(x, self.b),
                                    umath.less(x, self.a))


class _DomainTan:
    """
    Define a valid interval for the `tan` function, so that:

    ``domain_tan(eps) = True`` where ``abs(cos(x)) < eps``

    """

    def __init__(self, eps):
        "domain_tan(eps) = true where abs(cos(x)) < eps)"
        self.eps = eps

    def __call__(self, x):
        "Executes the call behavior."
        with np.errstate(invalid='ignore'):
            return umath.less(umath.absolute(umath.cos(x)), self.eps)


class _DomainSafeDivide:
    """
    Define a domain for safe division.

    """

    def __init__(self, tolerance=None):
        self.tolerance = tolerance

    def __call__(self, a, b):
        # Delay the selection of the tolerance to here in order to reduce numpy
        # import times. The calculation of these parameters is a substantial
        # component of numpy's import time.
        if self.tolerance is None:
            self.tolerance = np.finfo(float).tiny
        # don't call ma ufuncs from __array_wrap__ which would fail for scalars
        a, b = np.asarray(a), np.asarray(b)
        with np.errstate(all='ignore'):
            return umath.absolute(a) * self.tolerance >= umath.absolute(b)


class _DomainGreater:
    """
    DomainGreater(v)(x) is True where x <= v.

    """

    def __init__(self, critical_value):
        "DomainGreater(v)(x) = true where x <= v"
        self.critical_value = critical_value

    def __call__(self, x):
        "Executes the call behavior."
        with np.errstate(invalid='ignore'):
            return umath.less_equal(x, self.critical_value)


class _DomainGreaterEqual:
    """
    DomainGreaterEqual(v)(x) is True where x < v.

    """

    def __init__(self, critical_value):
        "DomainGreaterEqual(v)(x) = true where x < v"
        self.critical_value = critical_value
    # 定义一个特殊方法 `__call__`，使对象能够像函数一样被调用
    def __call__(self, x):
        "Executes the call behavior."
        # 在上下文管理器中，忽略无效值错误（NaN，无穷大等）
        with np.errstate(invalid='ignore'):
            # 使用 numpy 库中的 umath 模块，比较 x 和对象属性 `self.critical_value` 的大小关系
            return umath.less(x, self.critical_value)
class _MaskedUFunc:
    # 定义一个封装了通用函数的类
    def __init__(self, ufunc):
        # 初始化方法，接受一个通用函数对象，并保存其文档字符串和名称
        self.f = ufunc
        self.__doc__ = ufunc.__doc__
        self.__name__ = ufunc.__name__

    def __str__(self):
        # 返回对象的字符串表示形式，指明这是一个被掩码的通用函数的版本
        return f"Masked version of {self.f}"


class _MaskedUnaryOperation(_MaskedUFunc):
    """
    Defines masked version of unary operations, where invalid values are
    pre-masked.

    Parameters
    ----------
    mufunc : callable
        The function for which to define a masked version. Made available
        as ``_MaskedUnaryOperation.f``.
    fill : scalar, optional
        Filling value, default is 0.
    domain : class instance
        Domain for the function. Should be one of the ``_Domain*``
        classes. Default is None.

    """

    def __init__(self, mufunc, fill=0, domain=None):
        # 初始化方法，接受一个一元操作函数和填充值，以及一个操作域
        super().__init__(mufunc)
        self.fill = fill
        self.domain = domain
        # 将函数的操作域和填充值添加到全局映射表中
        ufunc_domain[mufunc] = domain
        ufunc_fills[mufunc] = fill

    def __call__(self, a, *args, **kwargs):
        """
        Execute the call behavior.

        """
        # 获取输入数据的有效数据部分
        d = getdata(a)
        
        # 处理操作域的情况
        if self.domain is not None:
            # 情况 1.1. : 有操作域的函数
            # 在掩码位置处的 NaN 值会引发 RuntimeWarnings，即使它们被掩码了。为了避免这种情况，我们禁止警告。
            with np.errstate(divide='ignore', invalid='ignore'):
                result = self.f(d, *args, **kwargs)
            # 创建一个掩码
            m = ~umath.isfinite(result)
            m |= self.domain(d)
            m |= getmask(a)
        else:
            # 情况 1.2. : 没有操作域的函数
            # 获取结果和掩码
            with np.errstate(divide='ignore', invalid='ignore'):
                result = self.f(d, *args, **kwargs)
            m = getmask(a)

        if not result.ndim:
            # 情况 2.1. : 结果是标量
            if m:
                return masked
            return result

        if m is not nomask:
            # 情况 2.2. : 结果是数组
            # 我们需要用输入数据填充无效数据
            # 这种方式在 C 语言中很简单：跳过元素并保留原始值，但在 Python 中必须这样做

            # 如果结果的数据类型低于输入数据的数据类型（如相等）
            try:
                np.copyto(result, d, where=m)
            except TypeError:
                pass
        # 转换为掩码子类的视图
        masked_result = result.view(get_masked_subclass(a))
        masked_result._mask = m
        masked_result._update_from(a)
        return masked_result


class _MaskedBinaryOperation(_MaskedUFunc):
    """
    Define masked version of binary operations, where invalid
    values are pre-masked.

    Parameters
    ----------
    mbfunc : function
        The function for which to define a masked version. Made available
        as ``_MaskedBinaryOperation.f``.

    """

    # 定义二元操作的掩码版本，其中无效值被预掩码
    domain : class instance
        默认的域名，用于该函数。应该是 ``_Domain*`` 类之一。默认为 None。
    fillx : scalar, optional
        第一个参数的填充值，默认为 0。
    filly : scalar, optional
        第二个参数的填充值，默认为 0。

    """

    def __init__(self, mbfunc, fillx=0, filly=0):
        """
        初始化函数，接受一个 mbfunc 函数作为参数。

        abfunc(fillx, filly) 必须已定义。

        abfunc(x, filly) = x，对所有 x 使得 reduce 可用。
        """
        super().__init__(mbfunc)
        self.fillx = fillx  # 设置 fillx 属性
        self.filly = filly  # 设置 filly 属性
        ufunc_domain[mbfunc] = None  # 将 mbfunc 映射到默认域名 None
        ufunc_fills[mbfunc] = (fillx, filly)  # 将 mbfunc 的填充值设为 (fillx, filly)

    def __call__(self, a, b, *args, **kwargs):
        """
        执行调用行为，计算函数的结果并处理掩码。

        """
        # 获取数据作为 ndarray
        (da, db) = (getdata(a), getdata(b))
        # 计算结果
        with np.errstate():
            np.seterr(divide='ignore', invalid='ignore')
            result = self.f(da, db, *args, **kwargs)
        # 获取结果的掩码
        (ma, mb) = (getmask(a), getmask(b))
        if ma is nomask:
            if mb is nomask:
                m = nomask
            else:
                m = umath.logical_or(getmaskarray(a), mb)
        elif mb is nomask:
            m = umath.logical_or(ma, getmaskarray(b))
        else:
            m = umath.logical_or(ma, mb)

        # 情况 1. : 标量
        if not result.ndim:
            if m:
                return masked  # 如果存在掩码，返回 masked
            return result  # 否则返回结果本身

        # 情况 2. : 数组
        # 在存在掩码的情况下，将结果重置为 da
        if m is not nomask and m.any():
            # 如果出现任何错误，放弃操作；无法保证掩码值的正确性
            try:
                np.copyto(result, da, casting='unsafe', where=m)
            except Exception:
                pass

        # 转换为 MaskedArray 的子类
        masked_result = result.view(get_masked_subclass(a, b))
        masked_result._mask = m
        if isinstance(a, MaskedArray):
            masked_result._update_from(a)
        elif isinstance(b, MaskedArray):
            masked_result._update_from(b)
        return masked_result
    def reduce(self, target, axis=0, dtype=None):
        """
        Reduce `target` along the given `axis`.
        函数用于沿指定轴向减少 `target` 的值。

        """
        # 获取 `target` 的掩码子类类型
        tclass = get_masked_subclass(target)
        # 获取 `target` 的掩码
        m = getmask(target)
        # 使用 `self.filly` 填充 `target`，返回填充后的数组
        t = filled(target, self.filly)
        # 如果 `t` 是标量，则将其转换为一维数组
        if t.shape == ():
            t = t.reshape(1)
            # 如果 `m` 不为空掩码，则复制掩码并调整形状为一维数组
            if m is not nomask:
                m = make_mask(m, copy=True)
                m.shape = (1,)

        # 如果 `m` 为空掩码
        if m is nomask:
            # 对 `t` 沿着指定轴向进行减少操作
            tr = self.f.reduce(t, axis)
            mr = nomask  # 结果掩码为空掩码
        else:
            # 对 `t` 沿着指定轴向进行减少操作，并指定数据类型 `dtype`
            tr = self.f.reduce(t, axis, dtype=dtype)
            # 计算结果的掩码
            mr = umath.logical_and.reduce(m, axis)

        # 如果 `tr` 是标量
        if not tr.shape:
            # 如果结果的掩码为真，则返回 `masked`，否则返回 `tr`
            if mr:
                return masked
            else:
                return tr
        # 将结果 `tr` 视图转换为 `tclass` 类型的对象
        masked_tr = tr.view(tclass)
        # 设置结果对象的掩码为 `mr`
        masked_tr._mask = mr
        return masked_tr

    def outer(self, a, b):
        """
        Return the function applied to the outer product of a and b.
        返回函数应用于 `a` 和 `b` 的外积。

        """
        # 获取 `a` 和 `b` 的数据
        (da, db) = (getdata(a), getdata(b))
        # 计算 `a` 和 `b` 的外积并赋值给 `d`
        d = self.f.outer(da, db)
        # 获取 `a` 和 `b` 的掩码
        ma = getmask(a)
        mb = getmask(b)
        # 如果 `a` 和 `b` 都没有掩码
        if ma is nomask and mb is nomask:
            m = nomask  # 结果掩码为空掩码
        else:
            # 获取 `a` 和 `b` 的掩码数组，并计算它们的逻辑或外积
            ma = getmaskarray(a)
            mb = getmaskarray(b)
            m = umath.logical_or.outer(ma, mb)
        # 如果结果的掩码维度为零且结果为真，则返回 `masked`
        if (not m.ndim) and m:
            return masked
        # 如果结果的掩码不为空，则将 `da` 复制到 `d` 中，条件为 `m`
        if m is not nomask:
            np.copyto(d, da, where=m)
        # 如果 `d` 是标量，则返回 `d`
        if not d.shape:
            return d
        # 将结果 `d` 视图转换为 `a` 和 `b` 的掩码子类类型的对象
        masked_d = d.view(get_masked_subclass(a, b))
        # 设置结果对象的掩码为 `m`
        masked_d._mask = m
        return masked_d

    def accumulate(self, target, axis=0):
        """Accumulate `target` along `axis` after filling with y fill
        value.
        在使用 `y` 填充值填充 `target` 后沿 `axis` 累积 `target` 的值。

        """
        # 获取 `target` 的掩码子类类型
        tclass = get_masked_subclass(target)
        # 使用 `self.filly` 填充 `target`，返回填充后的数组
        t = filled(target, self.filly)
        # 对 `t` 沿着指定轴向累积操作
        result = self.f.accumulate(t, axis)
        # 将累积结果 `result` 视图转换为 `tclass` 类型的对象
        masked_result = result.view(tclass)
        return masked_result
class _DomainedBinaryOperation(_MaskedUFunc):
    """
    Define binary operations that have a domain, like divide.

    They have no reduce, outer or accumulate.

    Parameters
    ----------
    mbfunc : function
        The function for which to define a masked version. Made available
        as ``_DomainedBinaryOperation.f``.
    domain : class instance
        Default domain for the function. Should be one of the ``_Domain*``
        classes.
    fillx : scalar, optional
        Filling value for the first argument, default is 0.
    filly : scalar, optional
        Filling value for the second argument, default is 0.

    """

    def __init__(self, dbfunc, domain, fillx=0, filly=0):
        """abfunc(fillx, filly) must be defined.
           abfunc(x, filly) = x for all x to enable reduce.
        """
        # 继承父类的初始化方法，传入二元函数
        super().__init__(dbfunc)
        # 设置操作的域和填充值
        self.domain = domain
        self.fillx = fillx
        self.filly = filly
        # 将函数和其默认域添加到全局字典中
        ufunc_domain[dbfunc] = domain
        ufunc_fills[dbfunc] = (fillx, filly)

    def __call__(self, a, b, *args, **kwargs):
        "Execute the call behavior."
        # 获取数据
        (da, db) = (getdata(a), getdata(b))
        # 计算结果
        with np.errstate(divide='ignore', invalid='ignore'):
            result = self.f(da, db, *args, **kwargs)
        # 获取掩码，由源掩码和无效值组合而成
        m = ~umath.isfinite(result)
        m |= getmask(a)
        m |= getmask(b)
        # 应用操作的域
        domain = ufunc_domain.get(self.f, None)
        if domain is not None:
            m |= domain(da, db)
        # 处理标量情况
        if not m.ndim:
            if m:
                return masked
            else:
                return result
        # 当掩码为真时，尝试恢复原始数据da
        try:
            np.copyto(result, 0, casting='unsafe', where=m)
            masked_da = umath.multiply(m, da)
            if np.can_cast(masked_da.dtype, result.dtype, casting='safe'):
                result += masked_da
        except Exception:
            pass

        # 转换为（子类的）MaskedArray
        masked_result = result.view(get_masked_subclass(a, b))
        masked_result._mask = m
        # 根据情况更新MaskedArray
        if isinstance(a, MaskedArray):
            masked_result._update_from(a)
        elif isinstance(b, MaskedArray):
            masked_result._update_from(b)
        return masked_result


# Unary ufuncs
# 定义一元操作的函数
exp = _MaskedUnaryOperation(umath.exp)
conjugate = _MaskedUnaryOperation(umath.conjugate)
sin = _MaskedUnaryOperation(umath.sin)
cos = _MaskedUnaryOperation(umath.cos)
arctan = _MaskedUnaryOperation(umath.arctan)
arcsinh = _MaskedUnaryOperation(umath.arcsinh)
sinh = _MaskedUnaryOperation(umath.sinh)
cosh = _MaskedUnaryOperation(umath.cosh)
# 创建一个针对 tanh 函数的封装的对象
tanh = _MaskedUnaryOperation(umath.tanh)
# 创建对 abs 函数和 absolute 函数的封装对象
abs = absolute = _MaskedUnaryOperation(umath.absolute)
# 创建一个针对 angle 函数的封装对象
angle = _MaskedUnaryOperation(angle)
# 创建一个针对 fabs 函数的封装对象
fabs = _MaskedUnaryOperation(umath.fabs)
# 创建一个针对 negative 函数的封装对象
negative = _MaskedUnaryOperation(umath.negative)
# 创建一个针对 floor 函数的封装对象
floor = _MaskedUnaryOperation(umath.floor)
# 创建一个针对 ceil 函数的封装对象
ceil = _MaskedUnaryOperation(umath.ceil)
# 创建一个针对 around 函数的封装对象
around = _MaskedUnaryOperation(np.around)
# 创建一个针对 logical_not 函数的封装对象
logical_not = _MaskedUnaryOperation(umath.logical_not)

# Domained unary ufuncs
# 创建一个针对 sqrt 函数的封装对象，设置默认值为 0.0，限定参数必须大于等于 0.0
sqrt = _MaskedUnaryOperation(umath.sqrt, 0.0, _DomainGreaterEqual(0.0))
# 创建一个针对 log 函数的封装对象，设置默认值为 1.0，限定参数必须大于 0.0
log = _MaskedUnaryOperation(umath.log, 1.0, _DomainGreater(0.0))
# 创建一个针对 log2 函数的封装对象，设置默认值为 1.0，限定参数必须大于 0.0
log2 = _MaskedUnaryOperation(umath.log2, 1.0, _DomainGreater(0.0))
# 创建一个针对 log10 函数的封装对象，设置默认值为 1.0，限定参数必须大于 0.0
log10 = _MaskedUnaryOperation(umath.log10, 1.0, _DomainGreater(0.0))
# 创建一个针对 tan 函数的封装对象，设置默认值为 0.0，限定参数必须大于 1e-35
tan = _MaskedUnaryOperation(umath.tan, 0.0, _DomainTan(1e-35))
# 创建一个针对 arcsin 函数的封装对象，设置默认值为 0.0，限定参数在 -1.0 到 1.0 之间
arcsin = _MaskedUnaryOperation(umath.arcsin, 0.0, _DomainCheckInterval(-1.0, 1.0))
# 创建一个针对 arccos 函数的封装对象，设置默认值为 0.0，限定参数在 -1.0 到 1.0 之间
arccos = _MaskedUnaryOperation(umath.arccos, 0.0, _DomainCheckInterval(-1.0, 1.0))
# 创建一个针对 arccosh 函数的封装对象，设置默认值为 1.0，限定参数必须大于等于 1.0
arccosh = _MaskedUnaryOperation(umath.arccosh, 1.0, _DomainGreaterEqual(1.0))
# 创建一个针对 arctanh 函数的封装对象，设置默认值为 0.0，限定参数在 -1.0+1e-15 到 1.0-1e-15 之间
arctanh = _MaskedUnaryOperation(umath.arctanh, 0.0, _DomainCheckInterval(-1.0 + 1e-15, 1.0 - 1e-15))

# Binary ufuncs
# 创建一个针对 add 函数的封装对象
add = _MaskedBinaryOperation(umath.add)
# 创建一个针对 subtract 函数的封装对象
subtract = _MaskedBinaryOperation(umath.subtract)
# 创建一个针对 multiply 函数的封装对象，设置默认值为 1 和 1
multiply = _MaskedBinaryOperation(umath.multiply, 1, 1)
# 创建一个针对 arctan2 函数的封装对象，设置默认值为 0.0 和 1.0
arctan2 = _MaskedBinaryOperation(umath.arctan2, 0.0, 1.0)
# 创建一个针对 equal 函数的封装对象
equal = _MaskedBinaryOperation(umath.equal)
# 禁用 equal 函数的 reduce 方法
equal.reduce = None
# 创建一个针对 not_equal 函数的封装对象
not_equal = _MaskedBinaryOperation(umath.not_equal)
# 禁用 not_equal 函数的 reduce 方法
not_equal.reduce = None
# 创建一个针对 less_equal 函数的封装对象
less_equal = _MaskedBinaryOperation(umath.less_equal)
# 禁用 less_equal 函数的 reduce 方法
less_equal.reduce = None
# 创建一个针对 greater_equal 函数的封装对象
greater_equal = _MaskedBinaryOperation(umath.greater_equal)
# 禁用 greater_equal 函数的 reduce 方法
greater_equal.reduce = None
# 创建一个针对 less 函数的封装对象
less = _MaskedBinaryOperation(umath.less)
# 禁用 less 函数的 reduce 方法
less.reduce = None
# 创建一个针对 greater 函数的封装对象
greater = _MaskedBinaryOperation(umath.greater)
# 禁用 greater 函数的 reduce 方法
greater.reduce = None
# 创建一个针对 logical_and 函数的封装对象
logical_and = _MaskedBinaryOperation(umath.logical_and)
# 创建一个对 logical_and 函数的封装对象，设置默认值为 1 和 1，并启用其 reduce 方法
alltrue = _MaskedBinaryOperation(umath.logical_and, 1, 1).reduce
# 创建一个针对 logical_or 函数的封装对象
logical_or = _MaskedBinaryOperation(umath.logical_or)
# 获取 logical_or 函数的 reduce 方法
sometrue = logical_or.reduce
# 创建一个针对 logical_xor 函数的封装对象
logical_xor = _MaskedBinaryOperation(umath.logical_xor)
# 创建一个针对 bitwise_and 函数的封装对象
bitwise_and = _MaskedBinaryOperation(umath.bitwise_and)
# 创建一个针对 bitwise_or 函数的封装对象
bitwise_or = _MaskedBinaryOperation(umath.bitwise_or)
# 创建一个针对 bitwise_xor 函数的封装对象
bitwise_xor = _MaskedBinaryOperation(umath.bitwise_xor)
# 创建一个针对 hypot 函数的封装对象
hypot = _MaskedBinaryOperation(umath.hypot)

# Domained binary ufuncs
# 创建一个针对 divide 函数的封装对象，设置默认域为 _DomainSafeDivide()，并指定两个参数的默认值为 0 和 1
divide = _DomainedBinaryOperation(umath.divide, _DomainSafeDivide(), 0, 1)
# 创建一个针对 true_divide 函数的封装对象，设置默认域为 _DomainSafeDivide()，并指定两个参数的默认值为 0 和 1
true_divide = _DomainedBinaryOperation(umath.true_divide, _DomainSafeDivide(), 0, 1)
# 创建一个针对 floor_divide 函数的封装对象，设置默认域为 _DomainSafeDivide()，并指定两个参数的默认值为 0 和 1
floor_divide = _DomainedBinaryOperation(umath.floor_divide, _DomainSafeDivide(), 0, 1)
# 创建一个针对 remainder 函数的封装对象，设置默认域为 _DomainSafeDivide()，并指定两个参数的默认值为 0 和 1
remainder = _DomainedBinaryOperation(umath.remainder, _DomainSafeDivide(), 0, 1)
fmod = _DomainedBinaryOperation(umath.fmod, _DomainSafeDivide(), 0, 1)
# 创建一个 _DomainedBinaryOperation 对象 fmod，使用 umath.fmod 函数作为操作函数，
# _DomainSafeDivide() 作为域安全除法，0 和 1 作为默认值。

mod = _DomainedBinaryOperation(umath.mod, _DomainSafeDivide(), 0, 1)
# 创建一个 _DomainedBinaryOperation 对象 mod，使用 umath.mod 函数作为操作函数，
# _DomainSafeDivide() 作为域安全除法，0 和 1 作为默认值。

###############################################################################
#                        Mask creation functions                              #
###############################################################################


def _replace_dtype_fields_recursive(dtype, primitive_dtype):
    "Private function allowing recursion in _replace_dtype_fields."
    _recurse = _replace_dtype_fields_recursive

    # Do we have some name fields ?
    if dtype.names is not None:
        descr = []
        for name in dtype.names:
            field = dtype.fields[name]
            if len(field) == 3:
                # Prepend the title to the name
                name = (field[-1], name)
            descr.append((name, _recurse(field[0], primitive_dtype)))
        new_dtype = np.dtype(descr)

    # Is this some kind of composite a la (float,2)
    elif dtype.subdtype:
        descr = list(dtype.subdtype)
        descr[0] = _recurse(dtype.subdtype[0], primitive_dtype)
        new_dtype = np.dtype(tuple(descr))

    # this is a primitive type, so do a direct replacement
    else:
        new_dtype = primitive_dtype

    # preserve identity of dtypes
    if new_dtype == dtype:
        new_dtype = dtype

    return new_dtype
# 递归替换 dtype 中的字段，将其转换为 primitive_dtype 类型的描述列表，并返回新的 dtype 对象。


def _replace_dtype_fields(dtype, primitive_dtype):
    """
    Construct a dtype description list from a given dtype.

    Returns a new dtype object, with all fields and subtypes in the given type
    recursively replaced with `primitive_dtype`.

    Arguments are coerced to dtypes first.
    """
    dtype = np.dtype(dtype)
    primitive_dtype = np.dtype(primitive_dtype)
    return _replace_dtype_fields_recursive(dtype, primitive_dtype)
# 从给定的 dtype 构造一个 dtype 描述列表。
# 返回一个新的 dtype 对象，其中给定类型中的所有字段和子类型都递归替换为 primitive_dtype。
# 首先将参数强制转换为 dtypes。


def make_mask_descr(ndtype):
    """
    Construct a dtype description list from a given dtype.

    Returns a new dtype object, with the type of all fields in `ndtype` to a
    boolean type. Field names are not altered.

    Parameters
    ----------
    ndtype : dtype
        The dtype to convert.

    Returns
    -------
    result : dtype
        A dtype that looks like `ndtype`, the type of all fields is boolean.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> dtype = np.dtype({'names':['foo', 'bar'],
    ...                   'formats':[np.float32, np.int64]})
    >>> dtype
    dtype([('foo', '<f4'), ('bar', '<i8')])
    >>> ma.make_mask_descr(dtype)
    dtype([('foo', '|b1'), ('bar', '|b1')])
    >>> ma.make_mask_descr(np.float32)
    dtype('bool')

    """
    return _replace_dtype_fields(ndtype, MaskType)
# 从给定的 dtype 构造一个 dtype 描述列表。
# 返回一个新的 dtype 对象，其中 `ndtype` 中所有字段的类型都被转换为布尔类型。
# 字段名不会改变。


def getmask(a):
    """
    Return the mask of a masked array, or nomask.

    Return the mask of `a` as an ndarray if `a` is a `MaskedArray` and the
    mask is not `nomask`, else return `nomask`. To guarantee a full array
    of booleans of the same shape as a, use `getmaskarray`.

    Parameters
    ----------
    a : MaskedArray
        Input masked array.

    Returns
    -------
    mask : ndarray
        Mask of the input array `a`, or `nomask` if `a` is not a `MaskedArray`.

    """
    # 返回输入的 MaskedArray 对象 `a` 的掩码（mask）属性，如果不存在则返回 `nomask`
    a : array_like
        Input `MaskedArray` for which the mask is required.

    See Also
    --------
    getdata : Return the data of a masked array as an ndarray.
    getmaskarray : Return the mask of a masked array, or full array of False.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> a = ma.masked_equal([[1,2],[3,4]], 2)
    >>> a
    masked_array(
      data=[[1, --],
            [3, 4]],
      mask=[[False,  True],
            [False, False]],
      fill_value=2)
    >>> ma.getmask(a)
    array([[False,  True],
           [False, False]])

    Equivalently use the `MaskedArray` `mask` attribute.

    >>> a.mask
    array([[False,  True],
           [False, False]])

    Result when mask == `nomask`

    >>> b = ma.masked_array([[1,2],[3,4]])
    >>> b
    masked_array(
      data=[[1, 2],
            [3, 4]],
      mask=False,
      fill_value=999999)
    >>> ma.nomask
    False
    >>> ma.getmask(b) == ma.nomask
    True
    >>> b.mask == ma.nomask
    True

    """
    # 返回 `a` 对象的 `_mask` 属性，如果不存在则返回 `nomask`
    return getattr(a, '_mask', nomask)
# 将 `getmask` 函数赋值给 `get_mask` 变量
get_mask = getmask

# 定义函数 `getmaskarray`，返回输入 `arr` 的掩码（mask）作为一个 ndarray，或者一个形状与 `arr` 相同的全 False 布尔数组。
def getmaskarray(arr):
    """
    Return the mask of a masked array, or full boolean array of False.

    Return the mask of `arr` as an ndarray if `arr` is a `MaskedArray` and
    the mask is not `nomask`, else return a full boolean array of False of
    the same shape as `arr`.

    Parameters
    ----------
    arr : array_like
        Input `MaskedArray` for which the mask is required.

    See Also
    --------
    getmask : Return the mask of a masked array, or nomask.
    getdata : Return the data of a masked array as an ndarray.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> a = ma.masked_equal([[1,2],[3,4]], 2)
    >>> a
    masked_array(
      data=[[1, --],
            [3, 4]],
      mask=[[False,  True],
            [False, False]],
      fill_value=2)
    >>> ma.getmaskarray(a)
    array([[False,  True],
           [False, False]])

    Result when mask == ``nomask``

    >>> b = ma.masked_array([[1,2],[3,4]])
    >>> b
    masked_array(
      data=[[1, 2],
            [3, 4]],
      mask=False,
      fill_value=999999)
    >>> ma.getmaskarray(b)
    array([[False, False],
           [False, False]])

    """
    # 获得输入 `arr` 的掩码（mask）
    mask = getmask(arr)
    # 如果掩码为 `nomask`，则创建一个形状与 `arr` 相同的全 False 布尔数组
    if mask is nomask:
        mask = make_mask_none(np.shape(arr), getattr(arr, 'dtype', None))
    return mask


# 定义函数 `is_mask`，判断输入 `m` 是否是有效的标准掩码
def is_mask(m):
    """
    Return True if m is a valid, standard mask.

    This function does not check the contents of the input, only that the
    type is MaskType. In particular, this function returns False if the
    mask has a flexible dtype.

    Parameters
    ----------
    m : array_like
        Array to test.

    Returns
    -------
    result : bool
        True if `m.dtype.type` is MaskType, False otherwise.

    See Also
    --------
    ma.isMaskedArray : Test whether input is an instance of MaskedArray.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> m = ma.masked_equal([0, 1, 0, 2, 3], 0)
    >>> m
    masked_array(data=[--, 1, --, 2, 3],
                 mask=[ True, False,  True, False, False],
           fill_value=0)
    >>> ma.is_mask(m)
    False
    >>> ma.is_mask(m.mask)
    True

    Input must be an ndarray (or have similar attributes)
    for it to be considered a valid mask.

    >>> m = [False, True, False]
    >>> ma.is_mask(m)
    False
    >>> m = np.array([False, True, False])
    >>> m
    array([False,  True, False])
    >>> ma.is_mask(m)
    True

    Arrays with complex dtypes don't return True.

    >>> dtype = np.dtype({'names':['monty', 'pithon'],
    ...                   'formats':[bool, bool]})
    >>> dtype
    dtype([('monty', '|b1'), ('pithon', '|b1')])
    >>> m = np.array([(True, False), (False, True), (True, False)],
    ...              dtype=dtype)
    >>> m
    array([( True, False), (False,  True), ( True, False)],
          dtype=[('monty', '?'), ('pithon', '?')])
    >>> ma.is_mask(m)
    False

    """
    # 尝试获取输入 `m` 的 dtype 类型，判断是否为 MaskType
    try:
        return m.dtype.type is MaskType
    except AttributeError:
        # 如果发生 AttributeError 异常，则返回 False
        return False
# 将输入的掩码数组 `m` 收缩至 `nomask`（一个特殊的掩码），如果可能的话
def _shrink_mask(m):
    if m.dtype.names is None and not m.any():  # 如果 `m` 的数据类型没有字段名且所有值均为 False
        return nomask  # 返回 `nomask`
    else:
        return m  # 否则返回原始的掩码数组 `m`

# 创建一个布尔掩码数组，从输入的数组 `m` 中生成
def make_mask(m, copy=False, shrink=True, dtype=MaskType):
    """
    Create a boolean mask from an array.

    Return `m` as a boolean mask, creating a copy if necessary or requested.
    The function can accept any sequence that is convertible to integers,
    or ``nomask``.  Does not require that contents must be 0s and 1s, values
    of 0 are interpreted as False, everything else as True.

    Parameters
    ----------
    m : array_like
        Potential mask.
    copy : bool, optional
        Whether to return a copy of `m` (True) or `m` itself (False).
    shrink : bool, optional
        Whether to shrink `m` to ``nomask`` if all its values are False.
    dtype : dtype, optional
        Data-type of the output mask. By default, the output mask has a
        dtype of MaskType (bool). If the dtype is flexible, each field has
        a boolean dtype. This is ignored when `m` is ``nomask``, in which
        case ``nomask`` is always returned.

    Returns
    -------
    result : ndarray
        A boolean mask derived from `m`.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> m = [True, False, True, True]
    >>> ma.make_mask(m)
    array([ True, False,  True,  True])
    >>> m = [1, 0, 1, 1]
    >>> ma.make_mask(m)
    array([ True, False,  True,  True])
    >>> m = [1, 0, 2, -3]
    >>> ma.make_mask(m)
    array([ True, False,  True,  True])

    Effect of the `shrink` parameter.

    >>> m = np.zeros(4)
    >>> m
    array([0., 0., 0., 0.])
    >>> ma.make_mask(m)
    False
    >>> ma.make_mask(m, shrink=False)
    array([False, False, False, False])

    Using a flexible `dtype`.

    >>> m = [1, 0, 1, 1]
    >>> n = [0, 1, 0, 0]
    >>> arr = []
    >>> for man, mouse in zip(m, n):
    ...     arr.append((man, mouse))
    >>> arr
    [(1, 0), (0, 1), (1, 0), (1, 0)]
    >>> dtype = np.dtype({'names':['man', 'mouse'],
    ...                   'formats':[np.int64, np.int64]})
    >>> arr = np.array(arr, dtype=dtype)
    >>> arr
    array([(1, 0), (0, 1), (1, 0), (1, 0)],
          dtype=[('man', '<i8'), ('mouse', '<i8')])
    >>> ma.make_mask(arr, dtype=dtype)
    array([(True, False), (False, True), (True, False), (True, False)],
          dtype=[('man', '|b1'), ('mouse', '|b1')])

    """
    if m is nomask:  # 如果 `m` 是 `nomask`
        return nomask  # 直接返回 `nomask`

    # 确保输入的 dtype 是有效的掩码描述符
    dtype = make_mask_descr(dtype)

    # legacy boolean special case: "existence of fields implies true"
    # 旧版布尔特例情况：如果存在字段，则全部置为 True
    if isinstance(m, ndarray) and m.dtype.fields and dtype == np.bool:
        return np.ones(m.shape, dtype=dtype)  # 返回一个形状与 `m` 相同的全为 True 的数组

    # 填充缺失数据，转换为 ndarray 类型
    copy = None if not copy else True  # 如果 copy 为 False，则设置为 None；否则设置为 True
    result = np.array(filled(m, True), copy=copy, dtype=dtype, subok=True)
    # Bas les masques !
    # 构建完整的掩码数组
    # 如果条件 shrink 为真，则调用 _shrink_mask 函数处理 result
    if shrink:
        result = _shrink_mask(result)
    # 返回处理后的 result 变量作为函数的结果
    return result
# 组合两个掩码数组，使用逻辑或操作符
def mask_or(m1, m2, copy=False, shrink=True):
    """
    Combine two masks with the ``logical_or`` operator.

    The result may be a view on `m1` or `m2` if the other is `nomask`
    (i.e. False).

    Parameters
    ----------
    m1, m2 : array_like
        Input masks.
    copy : bool, optional
        If copy is False and one of the inputs is `nomask`, return a view
        of the other input mask. Defaults to False.
    shrink : bool, optional
        Whether to shrink the output to `nomask` if all its values are
        False. Defaults to True.

    Returns
    -------
    mask : output mask
        The result masks values that are masked in either `m1` or `m2`.

    Raises
    ------
    ValueError
        If `m1` and `m2` have different flexible dtypes.

    Examples
    --------
    >>> m1 = np.ma.make_mask([0, 1, 1, 0])
    >>> m2 = np.ma.make_mask([1, 0, 0, 0])
    >>> np.ma.mask_or(m1, m2)
    array([ True,  True,  True, False])

    """
    # 如果 m1 是 nomask 或者 False，则根据 m2 的类型创建一个掩码数组
    if (m1 is nomask) or (m1 is False):
        dtype = getattr(m2, 'dtype', MaskType)
        return make_mask(m2, copy=copy, shrink=shrink, dtype=dtype)
    # 检查是否 m2 是 None 或者 False
    if (m2 is nomask) or (m2 is False):
        # 获取 m1 的数据类型，如果没有则使用 MaskType
        dtype = getattr(m1, 'dtype', MaskType)
        # 使用 make_mask 函数创建一个掩码，根据需要复制、缩小和指定数据类型
        return make_mask(m1, copy=copy, shrink=shrink, dtype=dtype)
    
    # 如果 m1 和 m2 是同一个对象，并且都是掩码类型，则直接返回 m1
    if m1 is m2 and is_mask(m1):
        return m1
    
    # 获取 m1 和 m2 的数据类型
    (dtype1, dtype2) = (getattr(m1, 'dtype', None), getattr(m2, 'dtype', None))
    
    # 如果 m1 和 m2 的数据类型不同，则抛出数值错误异常
    if dtype1 != dtype2:
        raise ValueError("Incompatible dtypes '%s'<>'%s'" % (dtype1, dtype2))
    
    # 如果 dtype1 具有字段名（即结构化数据类型），则执行以下操作
    if dtype1.names is not None:
        # 根据 m1 和 m2 进行广播形状，创建一个新的掩码数组
        newmask = np.empty(np.broadcast(m1, m2).shape, dtype1)
        # 使用 _recursive_mask_or 函数递归计算 m1 和 m2 的或逻辑，存储到 newmask 中
        _recursive_mask_or(m1, m2, newmask)
        return newmask
    
    # 否则，创建一个新的掩码，基于 m1 和 m2 的逻辑或运算结果
    return make_mask(umath.logical_or(m1, m2), copy=copy, shrink=shrink)
# 定义一个函数，用于将输入的掩码数组展平为完全展开的版本，其中嵌套字段被展开。
def flatten_mask(mask):
    """
    Returns a completely flattened version of the mask, where nested fields
    are collapsed.

    Parameters
    ----------
    mask : array_like
        Input array, which will be interpreted as booleans.

    Returns
    -------
    flattened_mask : ndarray of bools
        The flattened input.

    Examples
    --------
    >>> mask = np.array([0, 0, 1])
    >>> np.ma.flatten_mask(mask)
    array([False, False,  True])

    >>> mask = np.array([(0, 0), (0, 1)], dtype=[('a', bool), ('b', bool)])
    >>> np.ma.flatten_mask(mask)
    array([False, False, False,  True])

    >>> mdtype = [('a', bool), ('b', [('ba', bool), ('bb', bool)])]
    >>> mask = np.array([(0, (0, 0)), (0, (0, 1))], dtype=mdtype)
    >>> np.ma.flatten_mask(mask)
    array([False, False, False, False, False,  True])

    """

    # 定义一个内部函数，用于展平掩码并返回一个（可能是嵌套的）布尔序列。
    def _flatmask(mask):
        "Flatten the mask and returns a (maybe nested) sequence of booleans."
        # 获取掩码的字段名
        mnames = mask.dtype.names
        # 如果存在字段名，则递归地展平每个字段
        if mnames is not None:
            return [flatten_mask(mask[name]) for name in mnames]
        else:
            return mask

    # 定义一个内部函数，生成序列的展平版本。
    def _flatsequence(sequence):
        "Generates a flattened version of the sequence."
        try:
            # 遍历序列中的每个元素
            for element in sequence:
                # 如果元素是可迭代的，则递归地展平元素
                if hasattr(element, '__iter__'):
                    yield from _flatsequence(element)
                else:
                    yield element
        except TypeError:
            # 如果遇到类型错误，则直接返回该序列
            yield sequence

    # 将输入的掩码转换为NumPy数组
    mask = np.asarray(mask)
    # 使用内部函数展平掩码的结果序列
    flattened = _flatsequence(_flatmask(mask))
    # 将展平后的序列转换为布尔类型的NumPy数组并返回
    return np.array([_ for _ in flattened], dtype=bool)


# 定义一个函数，用于检查给定轴上是否存在掩码值。
def _check_mask_axis(mask, axis, keepdims=np._NoValue):
    "Check whether there are masked values along the given axis"
    # 如果掩码不等于nomask，则在给定轴上检查所有掩码值是否为真
    kwargs = {} if keepdims is np._NoValue else {'keepdims': keepdims}
    if mask is not nomask:
        return mask.all(axis=axis, **kwargs)
    # 如果掩码等于nomask，则返回nomask
    return nomask


###############################################################################
#                             Masking functions                               #
###############################################################################

# 定义一个函数，根据条件对数组进行掩码处理。
def masked_where(condition, a, copy=True):
    """
    Mask an array where a condition is met.

    Return `a` as an array masked where `condition` is True.
    Any masked values of `a` or `condition` are also masked in the output.

    Parameters
    ----------
    condition : array_like
        Masking condition.  When `condition` tests floating point values for
        equality, consider using ``masked_values`` instead.
    a : array_like
        Array to mask.
    copy : bool
        If True (default) make a copy of `a` in the result.  If False modify
        `a` in place and return a view.

    Returns
    -------
    result : MaskedArray
        The result of masking `a` where `condition` is True.

    See Also
    --------
    masked_values : Mask using floating point equality.
    masked_equal : Mask where equal to a given value.

    """

    # 返回根据条件对数组进行掩码处理后的结果。
    pass  # 函数体实现由于省略号而被略过，没有实际的代码。
    # 创建一个掩码，其中条件不等于给定值。
    masked_not_equal : Mask where *not* equal to a given value.
    # 创建一个掩码，其中条件小于或等于给定值。
    masked_less_equal : Mask where less than or equal to a given value.
    # 创建一个掩码，其中条件大于或等于给定值。
    masked_greater_equal : Mask where greater than or equal to a given value.
    # 创建一个掩码，其中条件小于给定值。
    masked_less : Mask where less than a given value.
    # 创建一个掩码，其中条件大于给定值。
    masked_greater : Mask where greater than a given value.
    # 创建一个掩码，其中条件在给定区间内。
    masked_inside : Mask inside a given interval.
    # 创建一个掩码，其中条件在给定区间外。
    masked_outside : Mask outside a given interval.
    # 创建一个掩码，其中条件是无效值（NaN或无穷大）。
    masked_invalid : Mask invalid values (NaNs or infs).
    
    Examples
    --------
    >>> import numpy.ma as ma
    >>> a = np.arange(4)
    >>> a
    array([0, 1, 2, 3])
    >>> ma.masked_where(a <= 2, a)
    masked_array(data=[--, --, --, 3],
                 mask=[ True,  True,  True, False],
           fill_value=999999)
    
    Mask array `b` conditional on `a`.
    
    >>> b = ['a', 'b', 'c', 'd']
    >>> ma.masked_where(a == 2, b)
    masked_array(data=['a', 'b', --, 'd'],
                 mask=[False, False,  True, False],
           fill_value='N/A',
                dtype='<U1')
    
    Effect of the `copy` argument.
    
    >>> c = ma.masked_where(a <= 2, a)
    >>> c
    masked_array(data=[--, --, --, 3],
                 mask=[ True,  True,  True, False],
           fill_value=999999)
    >>> c[0] = 99
    >>> c
    masked_array(data=[99, --, --, 3],
                 mask=[False,  True,  True, False],
           fill_value=999999)
    >>> a
    array([0, 1, 2, 3])
    >>> c = ma.masked_where(a <= 2, a, copy=False)
    >>> c[0] = 99
    >>> c
    masked_array(data=[99, --, --, 3],
                 mask=[False,  True,  True, False],
           fill_value=999999)
    >>> a
    array([99,  1,  2,  3])
    
    When `condition` or `a` contain masked values.
    
    >>> a = np.arange(4)
    >>> a = ma.masked_where(a == 2, a)
    >>> a
    masked_array(data=[0, 1, --, 3],
                 mask=[False, False,  True, False],
           fill_value=999999)
    >>> b = np.arange(4)
    >>> b = ma.masked_where(b == 0, b)
    >>> b
    masked_array(data=[--, 1, 2, 3],
                 mask=[ True, False, False, False],
           fill_value=999999)
    >>> ma.masked_where(a == 3, b)
    masked_array(data=[--, 1, --, --],
                 mask=[ True, False,  True,  True],
           fill_value=999999)
    
    """
    # 确保条件是有效的标准类型掩码。
    cond = make_mask(condition, shrink=False)
    # 创建一个数组 `a` 的副本，并确保其是 NumPy 数组。
    a = np.array(a, copy=copy, subok=True)
    
    (cshape, ashape) = (cond.shape, a.shape)
    # 检查条件和输入数组 `a` 的形状是否一致，如果不一致则引发错误。
    if cshape and cshape != ashape:
        raise IndexError("Inconsistent shape between the condition and the input"
                         " (got %s and %s)" % (cshape, ashape))
    # 如果数组 `a` 具有 `_mask` 属性，则将条件和 `a` 的现有掩码进行或运算。
    if hasattr(a, '_mask'):
        cond = mask_or(cond, a._mask)
        cls = type(a)
    else:
        cls = MaskedArray
    # 将数组 `a` 视图化为 `cls` 类型。
    result = a.view(cls)
    # 将结果的掩码设置为经过缩减处理的 `cond`。
    result.mask = _shrink_mask(cond)
    # 当 `a` 是 MaskedArray 且没有掩码视图时，处理结构化掩码的赋值。
    # 由于布尔值没有视图，当 'a' 是带有 nomask 的 MaskedArray 时
    # 如果不进行复制操作，并且对象 a 具有 '_mask' 属性，并且其掩码为 nomask（即无掩码）时
    if not copy and hasattr(a, '_mask') and getmask(a) is nomask:
        # 将结果的掩码更新到对象 a 的掩码上（视图方式）
        a._mask = result._mask.view()
    # 返回处理后的结果对象
    return result
# 根据条件将数组中大于给定值的元素进行掩码处理
def masked_greater(x, value, copy=True):
    """
    Mask an array where greater than a given value.

    This function is a shortcut to ``masked_where``, with
    `condition` = (x > value).

    See Also
    --------
    masked_where : Mask where a condition is met.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> a = np.arange(4)
    >>> a
    array([0, 1, 2, 3])
    >>> ma.masked_greater(a, 2)
    masked_array(data=[0, 1, 2, --],
                 mask=[False, False, False,  True],
           fill_value=999999)

    """
    return masked_where(greater(x, value), x, copy=copy)


# 根据条件将数组中大于等于给定值的元素进行掩码处理
def masked_greater_equal(x, value, copy=True):
    """
    Mask an array where greater than or equal to a given value.

    This function is a shortcut to ``masked_where``, with
    `condition` = (x >= value).

    See Also
    --------
    masked_where : Mask where a condition is met.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> a = np.arange(4)
    >>> a
    array([0, 1, 2, 3])
    >>> ma.masked_greater_equal(a, 2)
    masked_array(data=[0, 1, --, --],
                 mask=[False, False,  True,  True],
           fill_value=999999)

    """
    return masked_where(greater_equal(x, value), x, copy=copy)


# 根据条件将数组中小于给定值的元素进行掩码处理
def masked_less(x, value, copy=True):
    """
    Mask an array where less than a given value.

    This function is a shortcut to ``masked_where``, with
    `condition` = (x < value).

    See Also
    --------
    masked_where : Mask where a condition is met.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> a = np.arange(4)
    >>> a
    array([0, 1, 2, 3])
    >>> ma.masked_less(a, 2)
    masked_array(data=[--, --, 2, 3],
                 mask=[ True,  True, False, False],
           fill_value=999999)

    """
    return masked_where(less(x, value), x, copy=copy)


# 根据条件将数组中小于等于给定值的元素进行掩码处理
def masked_less_equal(x, value, copy=True):
    """
    Mask an array where less than or equal to a given value.

    This function is a shortcut to ``masked_where``, with
    `condition` = (x <= value).

    See Also
    --------
    masked_where : Mask where a condition is met.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> a = np.arange(4)
    >>> a
    array([0, 1, 2, 3])
    >>> ma.masked_less_equal(a, 2)
    masked_array(data=[--, --, --, 3],
                 mask=[ True,  True,  True, False],
           fill_value=999999)

    """
    return masked_where(less_equal(x, value), x, copy=copy)


# 根据条件将数组中不等于给定值的元素进行掩码处理
def masked_not_equal(x, value, copy=True):
    """
    Mask an array where *not* equal to a given value.

    This function is a shortcut to ``masked_where``, with
    `condition` = (x != value).

    See Also
    --------
    masked_where : Mask where a condition is met.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> a = np.arange(4)
    >>> a
    array([0, 1, 2, 3])
    >>> ma.masked_not_equal(a, 2)
    masked_array(data=[--, --, 2, --],
                 mask=[ True,  True, False,  True],
           fill_value=999999)

    """
    return masked_where(not_equal(x, value), x, copy=copy)
    # 返回一个根据条件遮罩处理的数组，将不等于给定值的元素遮盖掉
    return masked_where(not_equal(x, value), x, copy=copy)
# 掩码数组中将超出给定区间的值标记为遮蔽值。

# 快捷方式使用 `masked_where` 函数，其中条件 `condition` 对于 `x` 超出区间 [v1,v2] (x < v1)|(x > v2) 为真。
# 边界 `v1` 和 `v2` 可以以任意顺序给出。

# 查看更多
# --------
# masked_where : 满足条件时进行掩码处理。

# 注意
# -----
# 数组 `x` 已预填充其填充值。

# 示例
# --------
# >>> import numpy.ma as ma
# >>> x = [0.31, 1.2, 0.01, 0.2, -0.4, -1.1]
# >>> ma.masked_outside(x, -0.3, 0.3)
# masked_array(data=[--, --, 0.01, 0.2, --, --],
#              mask=[ True,  True, False, False,  True,  True],
#        fill_value=1e+20)

# 边界 `v1` 和 `v2` 的顺序不影响结果。

>>> ma.masked_outside(x, 0.3, -0.3)
masked_array(data=[--, --, 0.01, 0.2, --, --],
             mask=[ True,  True, False, False,  True,  True],
       fill_value=1e+20)
    # 创建一个布尔条件，判断数组 xf 中的元素是否小于 v1 或者大于 v2
    condition = (xf < v1) | (xf > v2)
    # 根据给定的条件，对数组 x 进行遮盖操作，即将符合条件的元素替换为特定值或标记，返回结果数组
    return masked_where(condition, x, copy=copy)
# 对输入数组 `x` 进行遮罩操作，将所有等于 `value` 的元素遮罩起来
def masked_object(x, value, copy=True, shrink=True):
    """
    Mask the array `x` where the data are exactly equal to value.

    This function is similar to `masked_values`, but only suitable
    for object arrays: for floating point, use `masked_values` instead.

    Parameters
    ----------
    x : array_like
        Array to mask
    value : object
        Comparison value
    copy : {True, False}, optional
        Whether to return a copy of `x`.
    shrink : {True, False}, optional
        Whether to collapse a mask full of False to nomask

    Returns
    -------
    result : MaskedArray
        The result of masking `x` where equal to `value`.

    See Also
    --------
    masked_where : Mask where a condition is met.
    masked_equal : Mask where equal to a given value (integers).
    masked_values : Mask using floating point equality.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> food = np.array(['green_eggs', 'ham'], dtype=object)
    >>> # don't eat spoiled food
    >>> eat = ma.masked_object(food, 'green_eggs')
    >>> eat
    masked_array(data=[--, 'ham'],
                 mask=[ True, False],
           fill_value='green_eggs',
                dtype=object)
    >>> # plain ol` ham is boring
    >>> fresh_food = np.array(['cheese', 'ham', 'pineapple'], dtype=object)
    >>> eat = ma.masked_object(fresh_food, 'green_eggs')
    >>> eat
    masked_array(data=['cheese', 'ham', 'pineapple'],
                 mask=False,
           fill_value='green_eggs',
                dtype=object)

    Note that `mask` is set to ``nomask`` if possible.

    >>> eat
    masked_array(data=['cheese', 'ham', 'pineapple'],
                 mask=False,
           fill_value='green_eggs',
                dtype=object)

    """
    # 检查输入的数组是否已经是 MaskedArray 类型
    if isMaskedArray(x):
        # 如果是 MaskedArray，则使用内部数据和指定的 value 进行相等性比较
        condition = umath.equal(x._data, value)
        mask = x._mask
    else:
        # 如果不是 MaskedArray，则将数组转换为 ndarray 并与 value 进行比较
        condition = umath.equal(np.asarray(x), value)
        # 初始化一个空的 mask
        mask = nomask
    # 创建一个新的 mask，将 condition 转换为 mask，并根据 shrink 参数处理 mask
    mask = mask_or(mask, make_mask(condition, shrink=shrink))
    # 返回一个新的 MaskedArray，其中数据为 x，mask 为上述生成的 mask，填充值为 value
    return masked_array(x, mask=mask, copy=copy, fill_value=value)
    # 使用 `filled` 函数生成一个新的数组 `xnew`，将原始数组 `x` 中与 `value` 近似相等的值填充为 `value`
    xnew = filled(x, value)
    
    # 如果 `xnew` 的数据类型是浮点数，使用 `isclose` 函数生成一个布尔掩码 `mask`，表示与 `value` 近似相等的位置
    if np.issubdtype(xnew.dtype, np.floating):
        mask = np.isclose(xnew, value, atol=atol, rtol=rtol)
    else:
        # 如果 `xnew` 的数据类型不是浮点数，使用 `umath.equal` 函数生成一个布尔掩码 `mask`，表示与 `value` 相等的位置
        mask = umath.equal(xnew, value)
    
    # 创建一个带有掩码的 `masked_array` 对象 `ret`，掩码为 `mask`，复制输入数组 `x`，并使用 `value` 作为填充值
    ret = masked_array(xnew, mask=mask, copy=copy, fill_value=value)
    
    # 如果 `shrink` 为真，调用 `shrink_mask` 方法来收缩掩码 `ret`，使其尽可能小
    if shrink:
        ret.shrink_mask()
    
    # 返回构建好的 `masked_array` 对象 `ret`
    return ret
def masked_invalid(a, copy=True):
    """
    Mask an array where invalid values occur (NaNs or infs).

    This function is a shortcut to ``masked_where``, with
    `condition` = ~(np.isfinite(a)). Any pre-existing mask is conserved.
    Only applies to arrays with a dtype where NaNs or infs make sense
    (i.e. floating point types), but accepts any array_like object.

    See Also
    --------
    masked_where : Mask where a condition is met.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> a = np.arange(5, dtype=float)
    >>> a[2] = np.nan
    >>> a[3] = np.inf
    >>> a
    array([ 0.,  1., nan, inf,  4.])
    >>> ma.masked_invalid(a)
    masked_array(data=[0.0, 1.0, --, --, 4.0],
                 mask=[False, False,  True,  True, False],
           fill_value=1e+20)

    """
    # 将输入转换为 numpy 数组（如果已经是，则不复制）
    a = np.array(a, copy=None, subok=True)
    # 使用 np.isfinite(a) 的反向作为条件，创建一个掩码并应用于数组 a，返回结果
    res = masked_where(~(np.isfinite(a)), a, copy=copy)
    # 如果结果的掩码为 nomask，则创建一个形状和数据类型匹配的新掩码
    if res._mask is nomask:
        res._mask = make_mask_none(res.shape, res.dtype)
    return res

###############################################################################
#                            Printing options                                 #
###############################################################################


class _MaskedPrintOption:
    """
    Handle the string used to represent missing data in a masked array.

    """

    def __init__(self, display):
        """
        Create the masked_print_option object.

        """
        # 初始化 _MaskedPrintOption 对象，设置显示值和启用状态
        self._display = display
        self._enabled = True

    def display(self):
        """
        Display the string to print for masked values.

        """
        # 返回用于打印掩码值的字符串显示值
        return self._display

    def set_display(self, s):
        """
        Set the string to print for masked values.

        """
        # 设置用于打印掩码值的字符串显示值为 s
        self._display = s

    def enabled(self):
        """
        Is the use of the display value enabled?

        """
        # 返回显示值是否启用
        return self._enabled

    def enable(self, shrink=1):
        """
        Set the enabling shrink to `shrink`.

        """
        # 设置显示值的启用状态为 shrink
        self._enabled = shrink

    def __str__(self):
        return str(self._display)

    __repr__ = __str__

# if you single index into a masked location you get this object.
# 设置默认的掩码打印选项为 '--'
masked_print_option = _MaskedPrintOption('--')


def _recursive_printoption(result, mask, printopt):
    """
    Puts printoptions in result where mask is True.

    Private function allowing for recursion

    """
    # 如果结果具有字段名，则递归地应用打印选项
    names = result.dtype.names
    if names is not None:
        for name in names:
            curdata = result[name]
            curmask = mask[name]
            _recursive_printoption(curdata, curmask, printopt)
    else:
        # 将 printopt 应用于 mask 为 True 的位置
        np.copyto(result, printopt, where=mask)
    return

# For better or worse, these end in a newline
# 预定义一些打印模板
_legacy_print_templates = dict(
    long_std=textwrap.dedent("""\
        masked_%(name)s(data =
         %(data)s,
        %(nlen)s        mask =
         %(mask)s,
        %(nlen)s  fill_value = %(fill)s)
        """),

用于定义长格式标准掩码函数的模板字符串，使用`textwrap.dedent`去除字符串的缩进。


    long_flx=textwrap.dedent("""\
        masked_%(name)s(data =
         %(data)s,
        %(nlen)s        mask =
         %(mask)s,
        %(nlen)s  fill_value = %(fill)s,
        %(nlen)s       dtype = %(dtype)s)
        """),

用于定义长格式灵活掩码函数的模板字符串，允许指定数据类型，使用`textwrap.dedent`去除字符串的缩进。


    short_std=textwrap.dedent("""\
        masked_%(name)s(data = %(data)s,
        %(nlen)s        mask = %(mask)s,
        %(nlen)s  fill_value = %(fill)s)
        """),

用于定义短格式标准掩码函数的模板字符串，使用`textwrap.dedent`去除字符串的缩进。


    short_flx=textwrap.dedent("""\
        masked_%(name)s(data = %(data)s,
        %(nlen)s        mask = %(mask)s,
        %(nlen)s  fill_value = %(fill)s,
        %(nlen)s       dtype = %(dtype)s)
        """)

用于定义短格式灵活掩码函数的模板字符串，允许指定数据类型，使用`textwrap.dedent`去除字符串的缩进。
def _recursive_filled(a, mask, fill_value):
    """
    Recursively fill `a` with `fill_value`.
    
    Parameters
    ----------
    a : structured array
        The structured array to be filled recursively.
    mask : ndarray
        The mask indicating which elements to fill.
    fill_value : structured array
        The value to fill into `a`.

    """
    # 获取结构化数组的字段名
    names = a.dtype.names
    # 遍历每个字段
    for name in names:
        # 获取当前字段的数据
        current = a[name]
        # 如果当前字段还有子字段，则递归填充子字段
        if current.dtype.names is not None:
            _recursive_filled(current, mask[name], fill_value[name])
        else:
            # 否则，直接拷贝填充值到当前字段
            np.copyto(current, fill_value[name], where=mask[name])


def flatten_structured_array(a):
    """
    Flatten a structured array.
    
    Parameters
    ----------
    a : structured array
        The structured array to be flattened.

    Returns
    -------
    output : masked array or ndarray
        A flattened masked array if the input is a masked array, otherwise a
        standard ndarray.

    """
    
    def flatten_sequence(iterable):
        """
        Flattens a compound of nested iterables.
        
        Parameters
        ----------
        iterable : iterable
            The iterable to be flattened.

        """
        for elm in iter(iterable):
            if hasattr(elm, '__iter__'):
                yield from flatten_sequence(elm)
            else:
                yield elm
    
    # 将输入转换为 ndarray
    a = np.asanyarray(a)
    # 记录初始形状
    inishape = a.shape
    # 将数组展平
    a = a.ravel()
    # 如果输入是 MaskedArray 类型
    if isinstance(a, MaskedArray):
        # 对每个元素展平并生成新的 MaskedArray
        out = np.array([tuple(flatten_sequence(d.item())) for d in a._data])
        out = out.view(MaskedArray)
        # 处理新的 mask
        out._mask = np.array([tuple(flatten_sequence(d.item()))
                              for d in getmaskarray(a)])
    else:
        # 否则，对每个元素展平生成 ndarray
        out = np.array([tuple(flatten_sequence(d.item())) for d in a])
    
    # 如果初始形状大于 1 维，重构输出形状
    if len(inishape) > 1:
        newshape = list(out.shape)
        newshape[0] = inishape
        out.shape = tuple(flatten_sequence(newshape))
    
    return out


def _arraymethod(funcname, onmask=True):
    """
    Return a class method wrapper around a basic array method.
    
    Creates a class method which returns a masked array, where the new
    `_data` array is the output of the corresponding basic method called
    on the original `_data`.
    
    If `onmask` is True, the new mask is the output of the method called
    on the initial mask. Otherwise, the new mask is just a reference
    to the initial mask.
    
    Parameters
    ----------
    funcname : str
        Name of the function to apply on data.
    onmask : bool
        Whether the mask must be processed also (True) or left
        alone (False). Default is True. Make available as `_onmask`
        attribute.

    Returns
    -------
    method : instancemethod
        Class method wrapper of the specified basic array method.

    """
    # 这里实现一个类方法，返回一个封装了基本数组方法的 MaskedArray
    pass  # 实际功能在代码中实现
    # 定义一个装饰器方法 `wrapped_method`，接受任意数量的位置参数 `args` 和关键字参数 `params`
    def wrapped_method(self, *args, **params):
        # 使用反射调用 `self._data` 对象的 `funcname` 方法，并传递参数 `args` 和 `params`
        result = getattr(self._data, funcname)(*args, **params)
        # 将结果转换为当前类的视图
        result = result.view(type(self))
        # 从当前对象更新 `result` 对象的属性
        result._update_from(self)
        # 获取当前对象的掩码
        mask = self._mask
        # 如果 `onmask` 为假（False），则将当前对象的掩码设置到 `result` 对象
        if not onmask:
            result.__setmask__(mask)
        # 否则，如果 `mask` 不是 `nomask`
        elif mask is not nomask:
            # `__setmask__` 方法会创建一个副本，这里不希望这样，因此直接赋值
            result._mask = getattr(mask, funcname)(*args, **params)
        # 返回装饰后的方法 `wrapped_method`
        return result
    # 使用反射获取 `ndarray` 或 `np` 模块中的 `funcname` 方法的文档字符串
    methdoc = getattr(ndarray, funcname, None) or getattr(np, funcname, None)
    # 如果获取到了文档字符串，则将其赋值给 `wrapped_method` 方法的文档字符串
    if methdoc is not None:
        wrapped_method.__doc__ = methdoc.__doc__
    # 将 `wrapped_method` 方法的名称设置为 `funcname`
    wrapped_method.__name__ = funcname
    # 返回装饰后的 `wrapped_method` 方法
    return wrapped_method
# 定义一个名为 MaskedIterator 的类，用于迭代遍历掩码数组（masked arrays）
class MaskedIterator:
    """
    Flat iterator object to iterate over masked arrays.

    A `MaskedIterator` iterator is returned by ``x.flat`` for any masked array
    `x`. It allows iterating over the array as if it were a 1-D array,
    either in a for-loop or by calling its `next` method.

    Iteration is done in C-contiguous style, with the last index varying the
    fastest. The iterator can also be indexed using basic slicing or
    advanced indexing.

    See Also
    --------
    MaskedArray.flat : Return a flat iterator over an array.
    MaskedArray.flatten : Returns a flattened copy of an array.

    Notes
    -----
    `MaskedIterator` is not exported by the `ma` module. Instead of
    instantiating a `MaskedIterator` directly, use `MaskedArray.flat`.

    Examples
    --------
    >>> x = np.ma.array(arange(6).reshape(2, 3))
    >>> fl = x.flat
    >>> type(fl)
    <class 'numpy.ma.MaskedIterator'>
    >>> for item in fl:
    ...     print(item)
    ...
    0
    1
    2
    3
    4
    5

    Extracting more than a single element b indexing the `MaskedIterator`
    returns a masked array:

    >>> fl[2:4]
    masked_array(data = [2 3],
                 mask = False,
           fill_value = 999999)

    """

    # 初始化方法，接受一个掩码数组 ma 作为参数
    def __init__(self, ma):
        # 将参数 ma 赋值给实例变量 self.ma
        self.ma = ma
        # 获取 ma 的数据部分的 flat 迭代器
        self.dataiter = ma._data.flat

        # 检查 ma 的掩码是否为 nomask（没有掩码）
        if ma._mask is nomask:
            # 如果没有掩码，将 self.maskiter 设为 None
            self.maskiter = None
        else:
            # 如果有掩码，获取 ma 的掩码部分的 flat 迭代器
            self.maskiter = ma._mask.flat

    # 定义迭代器的 __iter__ 方法，返回自身迭代器
    def __iter__(self):
        return self

    # 定义索引获取方法 __getitem__，接受索引 indx 作为参数
    def __getitem__(self, indx):
        # 使用 dataiter 的 __getitem__ 方法获取数据，并根据 ma 的类型创建相应的结果
        result = self.dataiter.__getitem__(indx).view(type(self.ma))
        # 如果存在 maskiter
        if self.maskiter is not None:
            # 获取 maskiter 的相应索引处的值
            _mask = self.maskiter.__getitem__(indx)
            # 如果 _mask 是 ndarray 类型
            if isinstance(_mask, ndarray):
                # 设置 _mask 的形状以匹配结果的形状，这对矩阵是必要的
                _mask.shape = result.shape
                # 将 _mask 赋给结果的掩码
                result._mask = _mask
            # 如果 _mask 是 np.void 类型
            elif isinstance(_mask, np.void):
                # 返回一个带有掩码的 mvoid 对象，使用 _mask 和 ma 的 hardmask 属性
                return mvoid(result, mask=_mask, hardmask=self.ma._hardmask)
            # 如果 _mask 为 True，表示单个标量被掩盖了
            elif _mask:
                return masked
        # 返回结果
        return result

    # 定义索引设置方法 __setitem__，接受索引 index 和值 value 作为参数
    def __setitem__(self, index, value):
        # 使用 getdata 函数获取 value 的数据部分，并设置给 dataiter 的相应索引
        self.dataiter[index] = getdata(value)
        # 如果存在 maskiter
        if self.maskiter is not None:
            # 使用 getmaskarray 函数获取 value 的掩码数组部分，并设置给 maskiter 的相应索引
            self.maskiter[index] = getmaskarray(value)
    def __next__(self):
        """
        返回下一个值，或者引发 StopIteration 异常。

        示例
        --------
        >>> x = np.ma.array([3, 2], mask=[0, 1])
        >>> fl = x.flat
        >>> next(fl)
        3
        >>> next(fl)
        masked
        >>> next(fl)
        Traceback (most recent call last):
          ...
        StopIteration

        """
        # 从 self.dataiter 中获取下一个数据项
        d = next(self.dataiter)
        # 如果 self.maskiter 不为 None，则从中获取下一个掩码项
        if self.maskiter is not None:
            m = next(self.maskiter)
            # 如果掩码项是 numpy 的 void 类型
            if isinstance(m, np.void):
                # 返回一个带有给定数据和掩码的 mvoid 对象
                return mvoid(d, mask=m, hardmask=self.ma._hardmask)
            # 如果掩码项是 True，表示当前数据项被掩盖
            elif m:
                # 返回 masked 表示数据项被掩盖
                return masked
        # 若没有掩码或者掩码项为 False，则返回原始数据项
        return d
# 设置模块名为 "numpy.ma"
@set_module("numpy.ma")
# 定义 MaskedArray 类，继承自 ndarray
class MaskedArray(ndarray):
    """
    An array class with possibly masked values.

    Masked values of True exclude the corresponding element from any
    computation.

    Construction::

      x = MaskedArray(data, mask=nomask, dtype=None, copy=False, subok=True,
                      ndmin=0, fill_value=None, keep_mask=True, hard_mask=None,
                      shrink=True, order=None)

    Parameters
    ----------
    data : array_like
        Input data.
    mask : sequence, optional
        Mask. Must be convertible to an array of booleans with the same
        shape as `data`. True indicates a masked (i.e. invalid) data.
    dtype : dtype, optional
        Data type of the output.
        If `dtype` is None, the type of the data argument (``data.dtype``)
        is used. If `dtype` is not None and different from ``data.dtype``,
        a copy is performed.
    copy : bool, optional
        Whether to copy the input data (True), or to use a reference instead.
        Default is False.
    subok : bool, optional
        Whether to return a subclass of `MaskedArray` if possible (True) or a
        plain `MaskedArray`. Default is True.
    ndmin : int, optional
        Minimum number of dimensions. Default is 0.
    fill_value : scalar, optional
        Value used to fill in the masked values when necessary.
        If None, a default based on the data-type is used.
    keep_mask : bool, optional
        Whether to combine `mask` with the mask of the input data, if any
        (True), or to use only `mask` for the output (False). Default is True.
    hard_mask : bool, optional
        Whether to use a hard mask or not. With a hard mask, masked values
        cannot be unmasked. Default is False.
    shrink : bool, optional
        Whether to force compression of an empty mask. Default is True.
    order : {'C', 'F', 'A'}, optional
        Specify the order of the array.  If order is 'C', then the array
        will be in C-contiguous order (last-index varies the fastest).
        If order is 'F', then the returned array will be in
        Fortran-contiguous order (first-index varies the fastest).
        If order is 'A' (default), then the returned array may be
        in any order (either C-, Fortran-contiguous, or even discontiguous),
        unless a copy is required, in which case it will be C-contiguous.

    Examples
    --------

    The ``mask`` can be initialized with an array of boolean values
    with the same shape as ``data``.

    >>> data = np.arange(6).reshape((2, 3))
    >>> np.ma.MaskedArray(data, mask=[[False, True, False],
    ...                               [False, False, True]])
    masked_array(
      data=[[0, --, 2],
            [3, 4, --]],
      mask=[[False,  True, False],
            [False, False,  True]],
      fill_value=999999)

    Alternatively, the ``mask`` can be initialized to homogeneous boolean
    array with the same shape as ``data`` by passing in a scalar
    """
    # 设置 __array_priority__ 为 15，指定该类在数组运算中的优先级
    __array_priority__ = 15
    
    # 定义默认的掩码为 nomask
    _defaultmask = nomask
    
    # 定义默认的硬掩码为 False
    _defaulthardmask = False
    
    # 设置基类为 ndarray
    _baseclass = ndarray
    
    # 每个轴上打印数组时的最大元素数目限制
    # 对于一维数组，此处的限制需要更多的值
    _print_width = 100
    _print_width_1d = 1500
    
    def _update_from(self, obj):
        """
        从 obj 中复制一些属性到 self 对象中。
    
        """
        # 如果 obj 是 ndarray 类型，则将其类型赋给 _baseclass
        if isinstance(obj, ndarray):
            _baseclass = type(obj)
        else:
            _baseclass = ndarray
        # 我们需要复制 _basedict 以避免向后传播
        _optinfo = {}
        # 更新 _optinfo 使用 obj 的 _optinfo，_basedict 属性
        _optinfo.update(getattr(obj, '_optinfo', {}))
        _optinfo.update(getattr(obj, '_basedict', {}))
        # 如果 obj 不是 MaskedArray 类型，则更新 _optinfo 使用 obj 的 __dict__
        if not isinstance(obj, MaskedArray):
            _optinfo.update(getattr(obj, '__dict__', {}))
        # 构建一个字典 _dict，包含以下属性的值
        _dict = dict(_fill_value=getattr(obj, '_fill_value', None),
                     _hardmask=getattr(obj, '_hardmask', False),
                     _sharedmask=getattr(obj, '_sharedmask', False),
                     _isfield=getattr(obj, '_isfield', False),
                     _baseclass=getattr(obj, '_baseclass', _baseclass),
                     _optinfo=_optinfo,
                     _basedict=_optinfo)
        # 将 _dict 中的内容更新到 self 对象的 __dict__ 中
        self.__dict__.update(_dict)
        # 同时将 _optinfo 中的内容也更新到 self 对象的 __dict__ 中
        self.__dict__.update(_optinfo)
        return
    def __array_wrap__(self, obj, context=None, return_scalar=False):
        """
        Special hook for ufuncs.

        Wraps the numpy array and sets the mask according to context.

        """
        if obj is self:  # 如果操作是原地操作
            result = obj  # 则结果就是对象本身
        else:
            result = obj.view(type(self))  # 否则，将对象视图转换为当前对象类型的视图
            result._update_from(self)  # 更新结果对象的属性从当前对象中

        if context is not None:
            result._mask = result._mask.copy()  # 复制结果对象的掩码

            func, args, out_i = context  # 获取函数、参数和输出索引
            # args 有时会包含输出 (gh-10459)，我们不希望这样
            input_args = args[:func.nin]  # 只取函数需要的输入参数部分

            # 计算输入参数的掩码并组合
            m = reduce(mask_or, [getmaskarray(arg) for arg in input_args])

            # 获取函数的定义域掩码
            domain = ufunc_domain.get(func, None)
            if domain is not None:
                # 取函数的定义域，并确保其为 ndarray
                with np.errstate(divide='ignore', invalid='ignore'):
                    d = filled(domain(*input_args), True)

                if d.any():
                    # 在定义域错误的地方填充结果值
                    try:
                        # 二元函数的定义域: 取最后一个值
                        fill_value = ufunc_fills[func][-1]
                    except TypeError:
                        # 一元函数的定义域: 直接使用此值
                        fill_value = ufunc_fills[func]
                    except KeyError:
                        # 未识别的定义域，使用 fill_value 代替
                        fill_value = self.fill_value

                    np.copyto(result, fill_value, where=d)  # 根据定义域填充结果值

                    # 更新掩码
                    if m is nomask:
                        m = d
                    else:
                        # 不要直接修改，以免出现反向传播风险
                        m = (m | d)

            # 确保掩码具有正确的大小
            if result is not self and result.shape == () and m:
                return masked
            else:
                result._mask = m
                result._sharedmask = False

        return result

    # setitem 可能会在整数数组中放入 NaN 或偶尔溢出浮点数。
    # 但这可能发生在屏蔽值中，因此避免正常的警告（通常也适用于屏蔽计算）。
    @np.errstate(over='ignore', invalid='ignore')
    # 定义特殊方法 __setitem__，用于设置对象中特定索引的值
    def __setitem__(self, indx, value):
        """
        x.__setitem__(i, y) <==> x[i]=y

        Set item described by index. If value is masked, masks those
        locations.

        设置由索引描述的元素。如果值被掩盖了，会屏蔽那些位置。

        """
        # 如果对象本身是被屏蔽的对象，则抛出异常
        if self is masked:
            raise MaskError('Cannot alter the masked element.')
        
        # 获取对象的数据和掩码
        _data = self._data
        _mask = self._mask
        
        # 如果索引是字符串类型，则直接设置数据和掩码
        if isinstance(indx, str):
            _data[indx] = value
            # 如果原本没有掩码，则创建一个适当形状和数据类型的掩码
            if _mask is nomask:
                self._mask = _mask = make_mask_none(self.shape, self.dtype)
            # 设置索引位置的掩码
            _mask[indx] = getmask(value)
            return

        # 获取数据的数据类型
        _dtype = _data.dtype

        # 如果值是被屏蔽的，则处理掩码
        if value is masked:
            # 如果原本没有掩码，则创建一个适当形状和数据类型的掩码
            if _mask is nomask:
                _mask = self._mask = make_mask_none(self.shape, _dtype)
            # 设置掩码的值
            if _dtype.names is not None:
                _mask[indx] = tuple([True] * len(_dtype.names))
            else:
                _mask[indx] = True
            return

        # 获取新值的数据部分
        dval = getattr(value, '_data', value)
        # 获取新值的掩码部分
        mval = getmask(value)
        
        # 如果数据类型有字段名，并且掩码是没有屏蔽的，设置为字段名的长度
        if _dtype.names is not None and mval is nomask:
            mval = tuple([False] * len(_dtype.names))
        
        # 如果原本没有掩码，则设置数据，然后设置掩码
        if _mask is nomask:
            _data[indx] = dval
            if mval is not nomask:
                _mask = self._mask = make_mask_none(self.shape, _dtype)
                _mask[indx] = mval
        # 如果没有硬屏蔽
        elif not self._hardmask:
            # 设置数据，然后设置掩码
            if (isinstance(indx, masked_array) and
                    not isinstance(value, masked_array)):
                _data[indx.data] = dval
            else:
                _data[indx] = dval
                _mask[indx] = mval
        # 如果索引有数据类型，并且数据类型是掩码类型
        elif hasattr(indx, 'dtype') and (indx.dtype == MaskType):
            indx = indx * umath.logical_not(_mask)
            _data[indx] = dval
        else:
            # 如果数据类型有字段名
            if _dtype.names is not None:
                err_msg = "Flexible 'hard' masks are not yet supported."
                raise NotImplementedError(err_msg)
            # 合并掩码
            mindx = mask_or(_mask[indx], mval, copy=True)
            dindx = self._data[indx]
            if dindx.size > 1:
                np.copyto(dindx, dval, where=~mindx)
            elif mindx is nomask:
                dindx = dval
            _data[indx] = dindx
            _mask[indx] = mindx
        return

    # 定义属性方法 dtype，用于获取对象的数据类型
    @property
    def dtype(self):
        return super().dtype

    # 定义 dtype 的 setter 方法
    @dtype.setter
    # 定义属性 dtype 的 setter 方法，用于设置 MaskedArray 对象的数据类型
    def dtype(self, dtype):
        # 调用父类的 dtype 属性的 setter 方法来设置数据类型
        super(MaskedArray, type(self)).dtype.__set__(self, dtype)
        # 如果存在遮盖数组（mask），则将其转换为新的数据类型描述符，并视为 ndarray
        if self._mask is not nomask:
            self._mask = self._mask.view(make_mask_descr(dtype), ndarray)
            # 尝试重置遮盖数组的形状（如果不是 void 类型的话）。
            # 如果数据类型的更改无法兼容，则会引发 ValueError 异常。
            try:
                self._mask.shape = self.shape
            except (AttributeError, TypeError):
                pass

    # 定义属性 shape 的 getter 方法，返回 MaskedArray 对象的形状
    @property
    def shape(self):
        return super().shape

    # 定义属性 shape 的 setter 方法，用于设置 MaskedArray 对象的形状
    @shape.setter
    def shape(self, shape):
        # 调用父类的 shape 属性的 setter 方法来设置形状
        super(MaskedArray, type(self)).shape.__set__(self, shape)
        # 如果存在遮盖数组（mask），则将其形状设置为新的形状
        # 注意：不能使用 self._mask，因为在设置形状时可能还不存在遮盖数组。
        if getmask(self) is not nomask:
            self._mask.shape = self.shape
    # 设置掩码
    def __setmask__(self, mask, copy=False):
        """
        Set the mask.

        """
        # 获取数据类型
        idtype = self.dtype
        # 获取当前掩码
        current_mask = self._mask
        # 如果掩码为空，则设置成True
        if mask is masked:
            mask = True

        # 如果当前掩码为空
        if current_mask is nomask:
            # 确保掩码已设置
            # 如果没有需要执行的操作，则什么也不做。
            if mask is nomask:
                return
            # 将当前掩码设置成与数据形状和数据类型相同的空掩码
            current_mask = self._mask = make_mask_none(self.shape, idtype)

        # 如果没有命名字段
        if idtype.names is None:
            # 硬掩码：不取消遮罩数据
            if self._hardmask:
                current_mask |= mask
            # 软掩码：将所有值都设置为False
            # 如果显然与兼容标量，则使用快速更新方法
            elif isinstance(mask, (int, float, np.bool, np.number)):
                current_mask[...] = mask
            # 否则，回退到更慢的通用方法
            else:
                current_mask.flat = mask
        else:
            # 有命名字段
            mdtype = current_mask.dtype
            mask = np.asarray(mask)
            # 掩码是单个值
            if not mask.ndim:
                # 它是布尔值：创建一个记录
                if mask.dtype.kind == 'b':
                    mask = np.array(tuple([mask.item()] * len(mdtype)),
                                    dtype=mdtype)
                # 它是记录：确保数据类型正确
                else:
                    mask = mask.astype(mdtype)
            # 掩码是序列
            else:
                # 确保新掩码是具有正确数据类型的数组
                try:
                    copy = None if not copy else True
                    mask = np.array(mask, copy=copy, dtype=mdtype)
                # 或者假设它是一个布尔/整数序列
                except TypeError:
                    mask = np.array([tuple([m] * len(mdtype)) for m in mask],
                                    dtype=mdtype)
            # 硬掩码：不取消遮罩数据
            if self._hardmask:
                for n in idtype.names:
                    current_mask[n] |= mask[n]
            # 软掩码：将所有值都设置为False
            # 如果显然与兼容标量，则使用快速更新方法
            elif isinstance(mask, (int, float, np.bool, np.number)):
                current_mask[...] = mask
            # 否则，回退到更慢的通用方法
            else:
                current_mask.flat = mask
        # 如果需要，重新调整形状
        if current_mask.shape:
            current_mask.shape = self.shape
        return

    _set_mask = __setmask__

    @property
    def mask(self):
        """
        返回当前的掩码。

        我们可以尝试强制重塑，但在某些情况下这将不起作用。
        返回一个视图，以便无法在原地更改 dtype 和 shape。
        这仍然通过身份保留 nomask。
        """
        return self._mask.view()

    @mask.setter
    def mask(self, value):
        """
        设置当前对象的掩码。

        Parameters:
        ----------
        value : ndarray
            要设置的掩码值。

        """
        self.__setmask__(value)

    @property
    def recordmask(self):
        """
        如果数组没有命名字段，则获取或设置数组的掩码。对于结构化数组，
        返回一个布尔值的 ndarray，其中条目为 ``True`` 如果 **所有** 字段都被掩盖，
        否则为 ``False``。

        Examples:
        --------
        >>> x = np.ma.array([(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)],
        ...         mask=[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)],
        ...        dtype=[('a', int), ('b', int)])
        >>> x.recordmask
        array([False, False,  True, False, False])

        Returns:
        -------
        ndarray
            如果数组没有命名字段，则返回掩码，否则返回经过展平的结构化数组的所有字段的布尔值。

        """
        _mask = self._mask.view(ndarray)
        if _mask.dtype.names is None:
            return _mask
        return np.all(flatten_structured_array(_mask), axis=-1)

    @recordmask.setter
    def recordmask(self, mask):
        """
        设置记录级别的掩码（暂不可用）。

        Raises:
        ------
        NotImplementedError
            暂未实现设置记录级别的掩码的功能。

        """
        raise NotImplementedError("Coming soon: setting the mask per records!")

    def harden_mask(self):
        """
        强制掩码变为硬掩码，阻止通过赋值取消掩码。

        Returns:
        -------
        self
            修改后的当前对象。

        See Also
        --------
        ma.MaskedArray.hardmask
        ma.MaskedArray.soften_mask

        """
        self._hardmask = True
        return self

    def soften_mask(self):
        """
        强制掩码变为软掩码（默认状态），允许通过赋值取消掩码。

        Returns:
        -------
        self
            修改后的当前对象。

        See Also
        --------
        ma.MaskedArray.hardmask
        ma.MaskedArray.harden_mask

        """
        self._hardmask = False
        return self

    @property
    def hardmask(self):
        """
        Specifies whether values can be unmasked through assignments.

        By default, assigning definite values to masked array entries will
        unmask them.  When `hardmask` is ``True``, the mask will not change
        through assignments.

        See Also
        --------
        ma.MaskedArray.harden_mask
        ma.MaskedArray.soften_mask

        Examples
        --------
        >>> x = np.arange(10)
        >>> m = np.ma.masked_array(x, x>5)
        >>> assert not m.hardmask

        Since `m` has a soft mask, assigning an element value unmasks that
        element:

        >>> m[8] = 42
        >>> m
        masked_array(data=[0, 1, 2, 3, 4, 5, --, --, 42, --],
                     mask=[False, False, False, False, False, False,
                           True, True, False, True],
               fill_value=999999)

        After hardening, the mask is not affected by assignments:

        >>> hardened = np.ma.harden_mask(m)
        >>> assert m.hardmask and hardened is m
        >>> m[:] = 23
        >>> m
        masked_array(data=[23, 23, 23, 23, 23, 23, --, --, 23, --],
                     mask=[False, False, False, False, False, False,
                           True, True, False, True],
               fill_value=999999)

        """
        return self._hardmask


```    
    def unshare_mask(self):
        """
        Copy the mask and set the `sharedmask` flag to ``False``.

        Whether the mask is shared between masked arrays can be seen from
        the `sharedmask` property. `unshare_mask` ensures the mask is not
        shared. A copy of the mask is only made if it was shared.

        See Also
        --------
        sharedmask

        """
        if self._sharedmask:
            # 如果 mask 是共享的，则复制一份并设置 sharedmask 标志为 False
            self._mask = self._mask.copy()
            self._sharedmask = False
        return self



    @property
    def sharedmask(self):
        """ Share status of the mask (read-only). """
        return self._sharedmask



    def shrink_mask(self):
        """
        Reduce a mask to nomask when possible.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Examples
        --------
        >>> x = np.ma.array([[1,2 ], [3, 4]], mask=[0]*4)
        >>> x.mask
        array([[False, False],
               [False, False]])
        >>> x.shrink_mask()
        masked_array(
          data=[[1, 2],
                [3, 4]],
          mask=False,
          fill_value=999999)
        >>> x.mask
        False

        """
        # 调用 _shrink_mask 函数将 mask 缩小到 nomask 的可能程度
        self._mask = _shrink_mask(self._mask)
        return self



    @property
    def baseclass(self):
        """ Class of the underlying data (read-only). """
        return self._baseclass
    def _get_data(self):
        """
        Returns the underlying data, as a view of the masked array.

        If the underlying data is a subclass of :class:`numpy.ndarray`, it is
        returned as such.

        >>> x = np.ma.array(np.matrix([[1, 2], [3, 4]]), mask=[[0, 1], [1, 0]])
        >>> x.data
        matrix([[1, 2],
                [3, 4]])

        The type of the data can be accessed through the :attr:`baseclass`
        attribute.
        """
        # 返回被屏蔽数组的底层数据的视图
        return ndarray.view(self, self._baseclass)

    _data = property(fget=_get_data)
    data = property(fget=_get_data)

    @property
    def flat(self):
        """ Return a flat iterator, or set a flattened version of self to value. """
        # 返回一个扁平化迭代器
        return MaskedIterator(self)

    @flat.setter
    def flat(self, value):
        # 将数组扁平化后，用给定的值设置其值
        y = self.ravel()
        y[:] = value

    @property
    def fill_value(self):
        """
        The filling value of the masked array is a scalar. When setting, None
        will set to a default based on the data type.

        Examples
        --------
        >>> for dt in [np.int32, np.int64, np.float64, np.complex128]:
        ...     np.ma.array([0, 1], dtype=dt).get_fill_value()
        ...
        np.int64(999999)
        np.int64(999999)
        np.float64(1e+20)
        np.complex128(1e+20+0j)

        >>> x = np.ma.array([0, 1.], fill_value=-np.inf)
        >>> x.fill_value
        np.float64(-inf)
        >>> x.fill_value = np.pi
        >>> x.fill_value
        np.float64(3.1415926535897931)

        Reset to default:

        >>> x.fill_value = None
        >>> x.fill_value
        np.float64(1e+20)

        """
        # 如果填充值为 None，则根据数据类型设置默认值
        if self._fill_value is None:
            self._fill_value = _check_fill_value(None, self.dtype)

        # 临时解决方案，处理 str 和 bytes 标量无法用 () 索引的问题，
        # 其他 numpy 标量可以。参见 issues #7259 和 #7267。
        # 当 #7267 修复后，可以移除此 if 块。
        if isinstance(self._fill_value, ndarray):
            return self._fill_value[()]
        return self._fill_value

    @fill_value.setter
    def fill_value(self, value=None):
        # 检查填充值并根据数据类型进行设置
        target = _check_fill_value(value, self.dtype)
        if not target.ndim == 0:
            # 2019-11-12, 1.18.0
            warnings.warn(
                "Non-scalar arrays for the fill value are deprecated. Use "
                "arrays with scalar values instead. The filled function "
                "still supports any array as `fill_value`.",
                DeprecationWarning, stacklevel=2)

        _fill_value = self._fill_value
        if _fill_value is None:
            # 如果未定义填充值属性，则创建该属性
            self._fill_value = target
        else:
            # 不覆盖属性，只填充它（用于传播）
            _fill_value[()] = target

    # 兼容性保留
    get_fill_value = fill_value.fget
    set_fill_value = fill_value.fset
    def filled(self, fill_value=None):
        """
        Return a copy of self, with masked values filled with a given value.
        **However**, if there are no masked values to fill, self will be
        returned instead as an ndarray.

        Parameters
        ----------
        fill_value : array_like, optional
            The value to use for invalid entries. Can be scalar or non-scalar.
            If non-scalar, the resulting ndarray must be broadcastable over
            input array. Default is None, in which case, the `fill_value`
            attribute of the array is used instead.

        Returns
        -------
        filled_array : ndarray
            A copy of ``self`` with invalid entries replaced by *fill_value*
            (be it the function argument or the attribute of ``self``), or
            ``self`` itself as an ndarray if there are no invalid entries to
            be replaced.

        Notes
        -----
        The result is **not** a MaskedArray!

        Examples
        --------
        >>> x = np.ma.array([1,2,3,4,5], mask=[0,0,1,0,1], fill_value=-999)
        >>> x.filled()
        array([   1,    2, -999,    4, -999])
        >>> x.filled(fill_value=1000)
        array([   1,    2, 1000,    4, 1000])
        >>> type(x.filled())
        <class 'numpy.ndarray'>

        Subclassing is preserved. This means that if, e.g., the data part of
        the masked array is a recarray, `filled` returns a recarray:

        >>> x = np.array([(-1, 2), (-3, 4)], dtype='i8,i8').view(np.recarray)
        >>> m = np.ma.array(x, mask=[(True, False), (False, True)])
        >>> m.filled()
        rec.array([(999999,      2), (    -3, 999999)],
                  dtype=[('f0', '<i8'), ('f1', '<i8')])
        """
        # 获取当前实例的掩码
        m = self._mask
        # 如果掩码为空（即没有被屏蔽的值），返回数据本身作为 ndarray
        if m is nomask:
            return self._data

        # 如果 fill_value 为 None，则使用实例的填充值
        if fill_value is None:
            fill_value = self.fill_value
        else:
            # 否则，检查并适配填充值的类型与实例数据类型一致
            fill_value = _check_fill_value(fill_value, self.dtype)

        # 如果实例是特殊的单例掩码对象，直接返回填充值的 ndarray
        if self is masked_singleton:
            return np.asanyarray(fill_value)

        # 如果掩码的 dtype 具有字段名（例如，结构化数据），则复制数据进行填充
        if m.dtype.names is not None:
            result = self._data.copy('K')
            _recursive_filled(result, self._mask, fill_value)
        # 如果掩码为空（没有任何屏蔽值），直接返回数据本身
        elif not m.any():
            return self._data
        else:
            result = self._data.copy('K')
            try:
                # 将填充值复制到数据的掩码位置
                np.copyto(result, fill_value, where=m)
            except (TypeError, AttributeError):
                # 处理填充值为对象数组的情况
                fill_value = np.array(fill_value, dtype=object)
                d = result.astype(object)
                result = np.choose(m, (d, fill_value))
            except IndexError:
                # 处理填充值为标量的情况
                if self._data.shape:
                    raise
                elif m:
                    result = np.array(fill_value, dtype=self.dtype)
                else:
                    result = self._data
        return result
    # 返回所有未掩码数据的一维数组。
    def compressed(self):
        """
        Return all the non-masked data as a 1-D array.

        Returns
        -------
        data : ndarray
            A new `ndarray` holding the non-masked data is returned.

        Notes
        -----
        The result is **not** a MaskedArray!

        Examples
        --------
        >>> x = np.ma.array(np.arange(5), mask=[0]*2 + [1]*3)
        >>> x.compressed()
        array([0, 1])
        >>> type(x.compressed())
        <class 'numpy.ndarray'>

        N-D arrays are compressed to 1-D.

        >>> arr = [[1, 2], [3, 4]]
        >>> mask = [[1, 0], [0, 1]]
        >>> x = np.ma.array(arr, mask=mask)
        >>> x.compressed()
        array([2, 3])

        """
        # 将多维数组压缩为一维数组
        data = ndarray.ravel(self._data)
        # 如果存在掩码，则根据掩码条件压缩数据
        if self._mask is not nomask:
            data = data.compress(np.logical_not(ndarray.ravel(self._mask)))
        # 返回压缩后的数据
        return data
    def compress(self, condition, axis=None, out=None):
        """
        Return `a` where condition is ``True``.

        If condition is a `~ma.MaskedArray`, missing values are considered
        as ``False``.

        Parameters
        ----------
        condition : var
            Boolean 1-d array selecting which entries to return. If len(condition)
            is less than the size of a along the axis, then output is truncated
            to length of condition array.
        axis : {None, int}, optional
            Axis along which the operation must be performed.
        out : {None, ndarray}, optional
            Alternative output array in which to place the result. It must have
            the same shape as the expected output but the type will be cast if
            necessary.

        Returns
        -------
        result : MaskedArray
            A :class:`~ma.MaskedArray` object.

        Notes
        -----
        Please note the difference with :meth:`compressed` !
        The output of :meth:`compress` has a mask, the output of
        :meth:`compressed` does not.

        Examples
        --------
        >>> x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
        >>> x
        masked_array(
          data=[[1, --, 3],
                [--, 5, --],
                [7, --, 9]],
          mask=[[False,  True, False],
                [ True, False,  True],
                [False,  True, False]],
          fill_value=999999)
        >>> x.compress([1, 0, 1])
        masked_array(data=[1, 3],
                     mask=[False, False],
               fill_value=999999)

        >>> x.compress([1, 0, 1], axis=1)
        masked_array(
          data=[[1, 3],
                [--, --],
                [7, 9]],
          mask=[[False, False],
                [ True,  True],
                [False, False]],
          fill_value=999999)

        """
        # 获取基本组件
        (_data, _mask) = (self._data, self._mask)

        # 将 condition 强制转换为普通的 ndarray，并忽略缺失值
        condition = np.asarray(condition)

        # 使用 _data 的 compress 方法根据 condition 进行数据筛选，生成新的 MaskedArray 对象
        _new = _data.compress(condition, axis=axis, out=out).view(type(self))

        # 从当前对象继承更新 _new
        _new._update_from(self)

        # 如果 _mask 不是 nomask，则使用 _mask 的 compress 方法根据 condition 进行掩码筛选
        if _mask is not nomask:
            _new._mask = _mask.compress(condition, axis=axis)

        # 返回处理后的新对象 _new
        return _new
    def _insert_masked_print(self):
        """
        Replace masked values with masked_print_option, casting all innermost
        dtypes to object.
        """
        # 检查是否启用了打印掩码选项
        if masked_print_option.enabled():
            # 获取当前对象的掩码
            mask = self._mask
            # 如果掩码为 nomask，则结果为原始数据
            if mask is nomask:
                res = self._data
            else:
                # 将数据转换为对象数组，以便进行填充操作
                data = self._data
                # 对于大型数组，为避免昂贵的对象数据类型转换，提取转换前的角落数据
                print_width = (self._print_width if self.ndim > 1
                               else self._print_width_1d)
                for axis in range(self.ndim):
                    if data.shape[axis] > print_width:
                        ind = print_width // 2
                        arr = np.split(data, (ind, -ind), axis=axis)
                        data = np.concatenate((arr[0], arr[2]), axis=axis)
                        arr = np.split(mask, (ind, -ind), axis=axis)
                        mask = np.concatenate((arr[0], arr[2]), axis=axis)

                # 替换数据类型字段为对象类型
                rdtype = _replace_dtype_fields(self.dtype, "O")
                res = data.astype(rdtype)
                # 递归应用打印选项到结果和掩码
                _recursive_printoption(res, mask, masked_print_option)
        else:
            # 否则，使用填充值填充数据
            res = self.filled(self.fill_value)
        return res

    def __str__(self):
        # 返回插入掩码打印后的对象的字符串表示
        return str(self._insert_masked_print())

    def _delegate_binop(self, other):
        # 这里模拟了 forward_binop_should_defer 的逻辑
        if isinstance(other, type(self)):
            return False
        array_ufunc = getattr(other, "__array_ufunc__", False)
        if array_ufunc is False:
            other_priority = getattr(other, "__array_priority__", -1000000)
            # 根据数组优先级判断是否延迟操作
            return self.__array_priority__ < other_priority
        else:
            # 如果 array_ufunc 不为 None，则在 ufunc 内部调用它
            # 返回 None 明确告诉我们不调用 ufunc，即延迟操作
            return array_ufunc is None

    def __eq__(self, other):
        """Check whether other equals self elementwise.

        When either of the elements is masked, the result is masked as well,
        but the underlying boolean data are still set, with self and other
        considered equal if both are masked, and unequal otherwise.

        For structured arrays, all fields are combined, with masked values
        ignored. The result is masked if all fields were masked, with self
        and other considered equal only if both were fully masked.
        """
        # 调用 _comparison 方法，比较 self 和 other 是否相等
        return self._comparison(other, operator.eq)
    def __ne__(self, other):
        """Check whether other does not equal self elementwise.

        When either of the elements is masked, the result is masked as well,
        but the underlying boolean data are still set, with self and other
        considered equal if both are masked, and unequal otherwise.

        For structured arrays, all fields are combined, with masked values
        ignored. The result is masked if all fields were masked, with self
        and other considered equal only if both were fully masked.
        """
        # 调用 _comparison 方法，使用 operator.ne 操作符进行元素级别的不等比较
        return self._comparison(other, operator.ne)

    # All other comparisons:
    def __le__(self, other):
        # 调用 _comparison 方法，使用 operator.le 操作符进行元素级别的小于等于比较
        return self._comparison(other, operator.le)

    def __lt__(self, other):
        # 调用 _comparison 方法，使用 operator.lt 操作符进行元素级别的小于比较
        return self._comparison(other, operator.lt)

    def __ge__(self, other):
        # 调用 _comparison 方法，使用 operator.ge 操作符进行元素级别的大于等于比较
        return self._comparison(other, operator.ge)

    def __gt__(self, other):
        # 调用 _comparison 方法，使用 operator.gt 操作符进行元素级别的大于比较
        return self._comparison(other, operator.gt)

    def __add__(self, other):
        """
        Add self to other, and return a new masked array.

        """
        # 如果可以委托二元操作，则返回 Not Implemented
        if self._delegate_binop(other):
            return NotImplemented
        # 调用 add 函数，将 self 和 other 相加
        return add(self, other)

    def __radd__(self, other):
        """
        Add other to self, and return a new masked array.

        """
        # 类似于 __rsub__ 和 __rdiv__，使用原始顺序：从 `other + self` 得到这里
        return add(other, self)

    def __sub__(self, other):
        """
        Subtract other from self, and return a new masked array.

        """
        # 如果可以委托二元操作，则返回 Not Implemented
        if self._delegate_binop(other):
            return NotImplemented
        # 调用 subtract 函数，将 self 减去 other
        return subtract(self, other)

    def __rsub__(self, other):
        """
        Subtract self from other, and return a new masked array.

        """
        return subtract(other, self)

    def __mul__(self, other):
        "Multiply self by other, and return a new masked array."
        # 如果可以委托二元操作，则返回 Not Implemented
        if self._delegate_binop(other):
            return NotImplemented
        # 调用 multiply 函数，将 self 乘以 other
        return multiply(self, other)

    def __rmul__(self, other):
        """
        Multiply other by self, and return a new masked array.

        """
        # 类似于 __rsub__ 和 __rdiv__，使用原始顺序：从 `other * self` 得到这里
        return multiply(other, self)

    def __div__(self, other):
        """
        Divide other into self, and return a new masked array.

        """
        # 如果可以委托二元操作，则返回 Not Implemented
        if self._delegate_binop(other):
            return NotImplemented
        # 调用 divide 函数，将 self 除以 other
        return divide(self, other)

    def __truediv__(self, other):
        """
        Divide other into self, and return a new masked array.

        """
        # 如果可以委托二元操作，则返回 Not Implemented
        if self._delegate_binop(other):
            return NotImplemented
        # 调用 true_divide 函数，将 self 除以 other
        return true_divide(self, other)

    def __rtruediv__(self, other):
        """
        Divide self into other, and return a new masked array.

        """
        return true_divide(other, self)
    def __floordiv__(self, other):
        """
        Divide other into self, and return a new masked array.

        """
        # 如果需要委托操作给其他对象，则返回未实现
        if self._delegate_binop(other):
            return NotImplemented
        # 调用 floor_divide 函数执行整数除法操作
        return floor_divide(self, other)

    def __rfloordiv__(self, other):
        """
        Divide self into other, and return a new masked array.

        """
        # 调用 floor_divide 函数将 other 除以 self
        return floor_divide(other, self)

    def __pow__(self, other):
        """
        Raise self to the power other, masking the potential NaNs/Infs

        """
        # 如果需要委托操作给其他对象，则返回未实现
        if self._delegate_binop(other):
            return NotImplemented
        # 调用 power 函数执行幂运算操作
        return power(self, other)

    def __rpow__(self, other):
        """
        Raise other to the power self, masking the potential NaNs/Infs

        """
        # 调用 power 函数将 other 的 self 次幂
        return power(other, self)

    def __iadd__(self, other):
        """
        Add other to self in-place.

        """
        # 获取 other 的掩码
        m = getmask(other)
        # 如果 self 的掩码是空的，则根据 other 的掩码创建新的掩码
        if self._mask is nomask:
            if m is not nomask and m.any():
                self._mask = make_mask_none(self.shape, self.dtype)
                self._mask += m
        # 否则，将 other 的掩码加到 self 的掩码中
        else:
            if m is not nomask:
                self._mask += m
        # 获取 other 的数据，根据 self 的掩码条件置零
        other_data = getdata(other)
        other_data = np.where(self._mask, other_data.dtype.type(0), other_data)
        # 将处理后的数据加到 self 的数据中
        self._data.__iadd__(other_data)
        return self

    def __isub__(self, other):
        """
        Subtract other from self in-place.

        """
        # 获取 other 的掩码
        m = getmask(other)
        # 如果 self 的掩码是空的，则根据 other 的掩码创建新的掩码
        if self._mask is nomask:
            if m is not nomask and m.any():
                self._mask = make_mask_none(self.shape, self.dtype)
                self._mask += m
        # 否则，将 other 的掩码加到 self 的掩码中
        elif m is not nomask:
            self._mask += m
        # 获取 other 的数据，根据 self 的掩码条件置零
        other_data = getdata(other)
        other_data = np.where(self._mask, other_data.dtype.type(0), other_data)
        # 将处理后的数据减去 self 的数据
        self._data.__isub__(other_data)
        return self

    def __imul__(self, other):
        """
        Multiply self by other in-place.

        """
        # 获取 other 的掩码
        m = getmask(other)
        # 如果 self 的掩码是空的，则根据 other 的掩码创建新的掩码
        if self._mask is nomask:
            if m is not nomask and m.any():
                self._mask = make_mask_none(self.shape, self.dtype)
                self._mask += m
        # 否则，将 other 的掩码加到 self 的掩码中
        elif m is not nomask:
            self._mask += m
        # 获取 other 的数据，根据 self 的掩码条件置一
        other_data = getdata(other)
        other_data = np.where(self._mask, other_data.dtype.type(1), other_data)
        # 将处理后的数据乘到 self 的数据中
        self._data.__imul__(other_data)
        return self
    def __idiv__(self, other):
        """
        Divide self by other in-place.

        """
        # 获取其他对象的数据
        other_data = getdata(other)
        # 对 self._data 和 other_data 进行安全的域除法运算
        dom_mask = _DomainSafeDivide().__call__(self._data, other_data)
        # 获取其他对象的掩码
        other_mask = getmask(other)
        # 计算新的掩码，合并其他对象的掩码和域掩码
        new_mask = mask_or(other_mask, dom_mask)
        
        # 下面的4行代码控制域填充
        if dom_mask.any():
            (_, fval) = ufunc_fills[np.divide]
            # 将 other_data 在域掩码为 True 处填充为指定的值 fval
            other_data = np.where(
                    dom_mask, other_data.dtype.type(fval), other_data)
        
        # 更新 self._mask，将新的掩码合并进去
        self._mask |= new_mask
        
        # 将 other_data 在 self._mask 为 True 处填充为 1
        other_data = np.where(self._mask, other_data.dtype.type(1), other_data)
        
        # 在 self._data 上执行原位除法操作
        self._data.__idiv__(other_data)
        
        # 返回 self 对象
        return self

    def __ifloordiv__(self, other):
        """
        Floor divide self by other in-place.

        """
        # 获取其他对象的数据
        other_data = getdata(other)
        # 对 self._data 和 other_data 进行安全的域除法运算
        dom_mask = _DomainSafeDivide().__call__(self._data, other_data)
        # 获取其他对象的掩码
        other_mask = getmask(other)
        # 计算新的掩码，合并其他对象的掩码和域掩码
        new_mask = mask_or(other_mask, dom_mask)
        
        # 下面的3行代码控制域填充
        if dom_mask.any():
            (_, fval) = ufunc_fills[np.floor_divide]
            # 将 other_data 在域掩码为 True 处填充为指定的值 fval
            other_data = np.where(
                    dom_mask, other_data.dtype.type(fval), other_data)
        
        # 更新 self._mask，将新的掩码合并进去
        self._mask |= new_mask
        
        # 将 other_data 在 self._mask 为 True 处填充为 1
        other_data = np.where(self._mask, other_data.dtype.type(1), other_data)
        
        # 在 self._data 上执行原位地向下取整除法操作
        self._data.__ifloordiv__(other_data)
        
        # 返回 self 对象
        return self

    def __itruediv__(self, other):
        """
        True divide self by other in-place.

        """
        # 获取其他对象的数据
        other_data = getdata(other)
        # 对 self._data 和 other_data 进行安全的域除法运算
        dom_mask = _DomainSafeDivide().__call__(self._data, other_data)
        # 获取其他对象的掩码
        other_mask = getmask(other)
        # 计算新的掩码，合并其他对象的掩码和域掩码
        new_mask = mask_or(other_mask, dom_mask)
        
        # 下面的3行代码控制域填充
        if dom_mask.any():
            (_, fval) = ufunc_fills[np.true_divide]
            # 将 other_data 在域掩码为 True 处填充为指定的值 fval
            other_data = np.where(
                    dom_mask, other_data.dtype.type(fval), other_data)
        
        # 更新 self._mask，将新的掩码合并进去
        self._mask |= new_mask
        
        # 将 other_data 在 self._mask 为 True 处填充为 1
        other_data = np.where(self._mask, other_data.dtype.type(1), other_data)
        
        # 在 self._data 上执行原位真除法操作
        self._data.__itruediv__(other_data)
        
        # 返回 self 对象
        return self

    def __ipow__(self, other):
        """
        Raise self to the power other, in place.

        """
        # 获取其他对象的数据
        other_data = getdata(other)
        
        # 将 self._mask 为 True 处的 other_data 填充为 1
        other_data = np.where(self._mask, other_data.dtype.type(1), other_data)
        
        # 获取其他对象的掩码
        other_mask = getmask(other)
        
        # 在忽略除以零和无效值的情况下，执行 self._data 的原位幂运算
        with np.errstate(divide='ignore', invalid='ignore'):
            self._data.__ipow__(other_data)
        
        # 标记无效值
        invalid = np.logical_not(np.isfinite(self._data))
        
        # 如果有无效值，则更新 self._mask
        if invalid.any():
            if self._mask is not nomask:
                self._mask |= invalid
            else:
                self._mask = invalid
            
            # 将 self._data 中的无效值复制为指定的填充值
            np.copyto(self._data, self.fill_value, where=invalid)
        
        # 计算新的掩码，合并其他对象的掩码和无效值掩码
        new_mask = mask_or(other_mask, invalid)
        
        # 更新 self._mask，将新的掩码合并进去
        self._mask = mask_or(self._mask, new_mask)
        
        # 返回 self 对象
        return self
    def __float__(self):
        """
        Convert to float.

        """
        # 检查数组大小是否大于1，如果是则引发类型错误
        if self.size > 1:
            raise TypeError("Only length-1 arrays can be converted "
                            "to Python scalars")
        # 如果数组元素被屏蔽（masked），发出警告并返回 NaN
        elif self._mask:
            warnings.warn("Warning: converting a masked element to nan.", stacklevel=2)
            return np.nan
        # 否则将数组的单个元素转换为 float 并返回
        return float(self.item())

    def __int__(self):
        """
        Convert to int.

        """
        # 检查数组大小是否大于1，如果是则引发类型错误
        if self.size > 1:
            raise TypeError("Only length-1 arrays can be converted "
                            "to Python scalars")
        # 如果数组元素被屏蔽（masked），则引发 MaskError
        elif self._mask:
            raise MaskError('Cannot convert masked element to a Python int.')
        # 否则将数组的单个元素转换为 int 并返回
        return int(self.item())

    @property
    def imag(self):
        """
        The imaginary part of the masked array.

        This property is a view on the imaginary part of this `MaskedArray`.

        See Also
        --------
        real

        Examples
        --------
        >>> x = np.ma.array([1+1.j, -2j, 3.45+1.6j], mask=[False, True, False])
        >>> x.imag
        masked_array(data=[1.0, --, 1.6],
                     mask=[False,  True, False],
               fill_value=1e+20)

        """
        # 获取当前数组的虚部视图，并设置相同的屏蔽（mask）
        result = self._data.imag.view(type(self))
        result.__setmask__(self._mask)
        return result

    # 保留兼容性
    get_imag = imag.fget

    @property
    def real(self):
        """
        The real part of the masked array.

        This property is a view on the real part of this `MaskedArray`.

        See Also
        --------
        imag

        Examples
        --------
        >>> x = np.ma.array([1+1.j, -2j, 3.45+1.6j], mask=[False, True, False])
        >>> x.real
        masked_array(data=[1.0, --, 3.45],
                     mask=[False,  True, False],
               fill_value=1e+20)

        """
        # 获取当前数组的实部视图，并设置相同的屏蔽（mask）
        result = self._data.real.view(type(self))
        result.__setmask__(self._mask)
        return result

    # 保留兼容性
    get_real = real.fget
    def ravel(self, order='C'):
        """
        Returns a 1D version of self, as a view.

        Parameters
        ----------
        order : {'C', 'F', 'A', 'K'}, optional
            The elements of `a` are read using this index order. 'C' means to
            index the elements in C-like order, with the last axis index
            changing fastest, back to the first axis index changing slowest.
            'F' means to index the elements in Fortran-like index order, with
            the first index changing fastest, and the last index changing
            slowest. Note that the 'C' and 'F' options take no account of the
            memory layout of the underlying array, and only refer to the order
            of axis indexing.  'A' means to read the elements in Fortran-like
            index order if `m` is Fortran *contiguous* in memory, C-like order
            otherwise.  'K' means to read the elements in the order they occur
            in memory, except for reversing the data when strides are negative.
            By default, 'C' index order is used.
            (Masked arrays currently use 'A' on the data when 'K' is passed.)

        Returns
        -------
        MaskedArray
            Output view is of shape ``(self.size,)`` (or
            ``(np.ma.product(self.shape),)``).

        Examples
        --------
        >>> x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
        >>> x
        masked_array(
          data=[[1, --, 3],
                [--, 5, --],
                [7, --, 9]],
          mask=[[False,  True, False],
                [ True, False,  True],
                [False,  True, False]],
          fill_value=999999)
        >>> x.ravel()
        masked_array(data=[1, --, 3, --, 5, --, 7, --, 9],
                     mask=[False,  True, False,  True, False,  True, False,  True,
                           False],
               fill_value=999999)

        """
        # 如果 order 参数是 'k', 'K', 'a', 'A' 中的一个，将其替换为 'F' 或 'C'
        if order in "kKaA":
            # 如果数据的存储顺序是 Fortran 连续的，则使用 'F'，否则使用 'C'
            order = "F" if self._data.flags.fnc else "C"
        # 将 ndarray.ravel 应用于 self._data，按指定顺序进行展平，并将结果视为当前类型的对象
        r = ndarray.ravel(self._data, order=order).view(type(self))
        # 从原始对象中更新结果对象的属性
        r._update_from(self)
        # 如果存在遮罩数据，则对遮罩数据进行相同顺序的展平和重塑，否则将 r._mask 设置为 nomask
        if self._mask is not nomask:
            r._mask = ndarray.ravel(self._mask, order=order).reshape(r.shape)
        else:
            r._mask = nomask
        # 返回处理后的结果对象
        return r
    def reshape(self, *s, **kwargs):
        """
        Give a new shape to the array without changing its data.

        Returns a masked array containing the same data, but with a new shape.
        The result is a view on the original array; if this is not possible, a
        ValueError is raised.

        Parameters
        ----------
        shape : int or tuple of ints
            The new shape should be compatible with the original shape. If an
            integer is supplied, then the result will be a 1-D array of that
            length.
        order : {'C', 'F'}, optional
            Determines whether the array data should be viewed as in C
            (row-major) or FORTRAN (column-major) order.

        Returns
        -------
        reshaped_array : array
            A new view on the array.

        See Also
        --------
        reshape : Equivalent function in the masked array module.
        numpy.ndarray.reshape : Equivalent method on ndarray object.
        numpy.reshape : Equivalent function in the NumPy module.

        Notes
        -----
        The reshaping operation cannot guarantee that a copy will not be made,
        to modify the shape in place, use ``a.shape = s``

        Examples
        --------
        >>> x = np.ma.array([[1,2],[3,4]], mask=[1,0,0,1])
        >>> x
        masked_array(
          data=[[--, 2],
                [3, --]],
          mask=[[ True, False],
                [False,  True]],
          fill_value=999999)
        >>> x = x.reshape((4,1))
        >>> x
        masked_array(
          data=[[--],
                [2],
                [3],
                [--]],
          mask=[[ True],
                [False],
                [False],
                [ True]],
          fill_value=999999)

        """
        # Update kwargs with 'order' set to 'C' if not provided
        kwargs.update(order=kwargs.get('order', 'C'))
        # Reshape the underlying data array, returning a new view of type 'self'
        result = self._data.reshape(*s, **kwargs).view(type(self))
        # Update the result with attributes from the current instance
        result._update_from(self)
        # Reshape the mask if it exists and assign to result's mask attribute
        mask = self._mask
        if mask is not nomask:
            result._mask = mask.reshape(*s, **kwargs)
        return result

    def resize(self, newshape, refcheck=True, order=False):
        """
        .. warning::

            This method does nothing, except raise a ValueError exception. A
            masked array does not own its data and therefore cannot safely be
            resized in place. Use the `numpy.ma.resize` function instead.

        This method is difficult to implement safely and may be deprecated in
        future releases of NumPy.

        """
        # Error message indicating the inability to resize a masked array
        errmsg = "A masked array does not own its data "\
                 "and therefore cannot be resized.\n" \
                 "Use the numpy.ma.resize function instead."
        # Raise a ValueError with the error message
        raise ValueError(errmsg)
    def put(self, indices, values, mode='raise'):
        """
        Set storage-indexed locations to corresponding values.

        Sets self._data.flat[n] = values[n] for each n in indices.
        If `values` is shorter than `indices` then it will repeat.
        If `values` has some masked values, the initial mask is updated
        in consequence, else the corresponding values are unmasked.

        Parameters
        ----------
        indices : 1-D array_like
            Target indices, interpreted as integers.
        values : array_like
            Values to place in self._data copy at target indices.
        mode : {'raise', 'wrap', 'clip'}, optional
            Specifies how out-of-bounds indices will behave.
            'raise' : raise an error.
            'wrap' : wrap around.
            'clip' : clip to the range.

        Notes
        -----
        `values` can be a scalar or length 1 array.

        Examples
        --------
        >>> x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
        >>> x
        masked_array(
          data=[[1, --, 3],
                [--, 5, --],
                [7, --, 9]],
          mask=[[False,  True, False],
                [ True, False,  True],
                [False,  True, False]],
          fill_value=999999)
        >>> x.put([0,4,8],[10,20,30])
        >>> x
        masked_array(
          data=[[10, --, 3],
                [--, 20, --],
                [7, --, 30]],
          mask=[[False,  True, False],
                [ True, False,  True],
                [False,  True, False]],
          fill_value=999999)

        >>> x.put(4,999)
        >>> x
        masked_array(
          data=[[10, --, 3],
                [--, 999, --],
                [7, --, 30]],
          mask=[[False,  True, False],
                [ True, False,  True],
                [False,  True, False]],
          fill_value=999999)

        """
        # 如果启用了硬屏蔽并且存在初始屏蔽，则清除落在屏蔽数据上的值/索引
        if self._hardmask and self._mask is not nomask:
            # 获取屏蔽后的索引
            mask = self._mask[indices]
            # 深复制索引和值
            indices = narray(indices, copy=None)
            values = narray(values, copy=None, subok=True)
            # 调整值的大小以匹配索引的形状
            values.resize(indices.shape)
            # 只保留非屏蔽的索引和对应的值
            indices = indices[~mask]
            values = values[~mask]

        # 将指定索引处的值设置为对应的数值
        self._data.put(indices, values, mode=mode)

        # 如果自身和值均未被屏蔽，则直接返回
        if self._mask is nomask and getmask(values) is nomask:
            return

        # 更新掩码
        m = getmaskarray(self)

        # 如果值未被屏蔽，则将对应索引处的屏蔽标记设为False
        if getmask(values) is nomask:
            m.put(indices, False, mode=mode)
        else:
            # 否则，将对应索引处的屏蔽标记设为值的屏蔽标记
            m.put(indices, values._mask, mode=mode)
        # 生成并设置新的掩码
        m = make_mask(m, copy=False, shrink=True)
        self._mask = m
        return
    def ids(self):
        """
        Return the addresses of the data and mask areas.

        Parameters
        ----------
        None

        Examples
        --------
        >>> x = np.ma.array([1, 2, 3], mask=[0, 1, 1])
        >>> x.ids()
        (166670640, 166659832) # may vary

        If the array has no mask, the address of `nomask` is returned. This address
        is typically not close to the data in memory:

        >>> x = np.ma.array([1, 2, 3])
        >>> x.ids()
        (166691080, 3083169284) # may vary

        """
        # 如果数组没有掩码，返回数据区域地址和 `nomask` 的地址
        if self._mask is nomask:
            return (self.ctypes.data, id(nomask))
        # 否则返回数据区域地址和掩码区域地址
        return (self.ctypes.data, self._mask.ctypes.data)

    def iscontiguous(self):
        """
        Return a boolean indicating whether the data is contiguous.

        Parameters
        ----------
        None

        Examples
        --------
        >>> x = np.ma.array([1, 2, 3])
        >>> x.iscontiguous()
        True

        `iscontiguous` returns one of the flags of the masked array:

        >>> x.flags
          C_CONTIGUOUS : True
          F_CONTIGUOUS : True
          OWNDATA : False
          WRITEABLE : True
          ALIGNED : True
          WRITEBACKIFCOPY : False

        """
        # 返回数据是否连续存储的布尔值
        return self.flags['CONTIGUOUS']

    def all(self, axis=None, out=None, keepdims=np._NoValue):
        """
        Returns True if all elements evaluate to True.

        The output array is masked where all the values along the given axis
        are masked: if the output would have been a scalar and that all the
        values are masked, then the output is `masked`.

        Refer to `numpy.all` for full documentation.

        See Also
        --------
        numpy.ndarray.all : corresponding function for ndarrays
        numpy.all : equivalent function

        Examples
        --------
        >>> np.ma.array([1,2,3]).all()
        True
        >>> a = np.ma.array([1,2,3], mask=True)
        >>> (a.all() is np.ma.masked)
        True

        """
        # 根据给定轴检查掩码，处理输出或者返回掩码对象
        kwargs = {} if keepdims is np._NoValue else {'keepdims': keepdims}

        mask = _check_mask_axis(self._mask, axis, **kwargs)
        if out is None:
            # 如果没有指定输出，创建填充后的数组并应用掩码
            d = self.filled(True).all(axis=axis, **kwargs).view(type(self))
            if d.ndim:
                d.__setmask__(mask)
            elif mask:
                return masked
            return d
        # 否则，在指定的输出上执行填充后的 all 操作，并处理掩码
        self.filled(True).all(axis=axis, out=out, **kwargs)
        if isinstance(out, MaskedArray):
            if out.ndim or mask:
                out.__setmask__(mask)
        return out
    # 定义一个方法 `any`，用于判断数组中是否有任何元素评估为 True
    def any(self, axis=None, out=None, keepdims=np._NoValue):
        """
        如果 `a` 的任何元素评估为 True，则返回 True。

        在计算过程中，掩码值被视为 False。

        完整文档请参考 `numpy.any`。

        另请参阅
        --------
        numpy.ndarray.any : ndarrays 对应的函数
        numpy.any : 等效函数

        """
        # 根据 `keepdims` 的值设置关键字参数
        kwargs = {} if keepdims is np._NoValue else {'keepdims': keepdims}

        # 根据指定的轴检查掩码，获取掩码数组
        mask = _check_mask_axis(self._mask, axis, **kwargs)
        
        # 如果没有提供输出数组 `out`
        if out is None:
            # 将填充了 False 的数组进行 `any` 计算，并转换为当前对象类型的视图
            d = self.filled(False).any(axis=axis, **kwargs).view(type(self))
            # 如果结果数组 `d` 的维度不为 0
            if d.ndim:
                # 将掩码应用到结果数组 `d`
                d.__setmask__(mask)
            # 如果有掩码存在或者结果为空，返回掩码数组，否则返回结果数组 `d`
            elif mask:
                d = masked
            return d
        
        # 如果提供了输出数组 `out`，在指定的轴上计算 `any`，并将结果存储在 `out` 中
        self.filled(False).any(axis=axis, out=out, **kwargs)
        
        # 如果 `out` 是 `MaskedArray` 类型
        if isinstance(out, MaskedArray):
            # 如果 `out` 的维度不为 0 或者存在掩码，则将掩码应用到 `out` 中
            if out.ndim or mask:
                out.__setmask__(mask)
        
        # 返回输出数组 `out`
        return out
    def nonzero(self):
        """
        Return the indices of unmasked elements that are not zero.

        Returns a tuple of arrays, one for each dimension, containing the
        indices of the non-zero elements in that dimension. The corresponding
        non-zero values can be obtained with::

            a[a.nonzero()]

        To group the indices by element, rather than dimension, use
        instead::

            np.transpose(a.nonzero())

        The result of this is always a 2d array, with a row for each non-zero
        element.

        Parameters
        ----------
        None

        Returns
        -------
        tuple_of_arrays : tuple
            Indices of elements that are non-zero.

        See Also
        --------
        numpy.nonzero :
            Function operating on ndarrays.
        flatnonzero :
            Return indices that are non-zero in the flattened version of the input
            array.
        numpy.ndarray.nonzero :
            Equivalent ndarray method.
        count_nonzero :
            Counts the number of non-zero elements in the input array.

        Examples
        --------
        >>> import numpy.ma as ma
        >>> x = ma.array(np.eye(3))
        >>> x
        masked_array(
          data=[[1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.]],
          mask=False,
          fill_value=1e+20)
        >>> x.nonzero()
        (array([0, 1, 2]), array([0, 1, 2]))

        Masked elements are ignored.

        >>> x[1, 1] = ma.masked
        >>> x
        masked_array(
          data=[[1.0, 0.0, 0.0],
                [0.0, --, 0.0],
                [0.0, 0.0, 1.0]],
          mask=[[False, False, False],
                [False,  True, False],
                [False, False, False]],
          fill_value=1e+20)
        >>> x.nonzero()
        (array([0, 2]), array([0, 2]))

        Indices can also be grouped by element.

        >>> np.transpose(x.nonzero())
        array([[0, 0],
               [2, 2]])

        A common use for ``nonzero`` is to find the indices of an array, where
        a condition is True.  Given an array `a`, the condition `a` > 3 is a
        boolean array and since False is interpreted as 0, ma.nonzero(a > 3)
        yields the indices of the `a` where the condition is true.

        >>> a = ma.array([[1,2,3],[4,5,6],[7,8,9]])
        >>> a > 3
        masked_array(
          data=[[False, False, False],
                [ True,  True,  True],
                [ True,  True,  True]],
          mask=False,
          fill_value=True)
        >>> ma.nonzero(a > 3)
        (array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))

        The ``nonzero`` method of the condition array can also be called.

        >>> (a > 3).nonzero()
        (array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))

        """
        # 将填充后的数组转换为非零元素的索引
        return np.asarray(self.filled(0)).nonzero()
    # 定义一个方法 `trace`，计算数组的迹(trace)，即对角线元素之和
    def trace(self, offset=0, axis1=0, axis2=1, dtype=None, out=None):
        """
        (this docstring should be overwritten)
        """
        #!!!: implement out + test!
        # 获取当前数组的掩码（如果有的话）
        m = self._mask
        # 如果数组没有掩码（即没有被遮罩的元素）
        if m is nomask:
            # 调用父类 ndarray 的 trace 方法进行计算
            result = super().trace(offset=offset, axis1=axis1, axis2=axis2,
                                   out=out)
            # 将结果转换为指定的数据类型 dtype
            return result.astype(dtype)
        else:
            # 否则，调用自身的 diagonal 方法获取对角线元素
            D = self.diagonal(offset=offset, axis1=axis1, axis2=axis2)
            # 将对角线元素转换为指定的数据类型 dtype，并用 0 填充掩码位置，然后沿最后一个轴求和
            return D.astype(dtype).filled(0).sum(axis=-1, out=out)
    # 将 trace 方法的文档字符串设为 ndarray.trace 方法的文档字符串
    trace.__doc__ = ndarray.trace.__doc__

    # 定义一个方法 `dot`，计算两个数组的点积
    def dot(self, b, out=None, strict=False):
        """
        a.dot(b, out=None)

        Masked dot product of two arrays. Note that `out` and `strict` are
        located in different positions than in `ma.dot`. In order to
        maintain compatibility with the functional version, it is
        recommended that the optional arguments be treated as keyword only.
        At some point that may be mandatory.

        .. versionadded:: 1.10.0

        Parameters
        ----------
        b : masked_array_like
            Inputs array.
        out : masked_array, optional
            Output argument. This must have the exact kind that would be
            returned if it was not used. In particular, it must have the
            right type, must be C-contiguous, and its dtype must be the
            dtype that would be returned for `ma.dot(a,b)`. This is a
            performance feature. Therefore, if these conditions are not
            met, an exception is raised, instead of attempting to be
            flexible.
        strict : bool, optional
            Whether masked data are propagated (True) or set to 0 (False)
            for the computation. Default is False.  Propagating the mask
            means that if a masked value appears in a row or column, the
            whole row or column is considered masked.

            .. versionadded:: 1.10.2

        See Also
        --------
        numpy.ma.dot : equivalent function

        """
        # 调用 numpy.ma 中的 dot 函数，计算两个数组的点积，传入参数包括 out 和 strict
        return dot(self, b, out=out, strict=strict)
    def sum(self, axis=None, dtype=None, out=None, keepdims=np._NoValue):
        """
        Return the sum of the array elements over the given axis.

        Masked elements are set to 0 internally.

        Refer to `numpy.sum` for full documentation.

        See Also
        --------
        numpy.ndarray.sum : corresponding function for ndarrays
        numpy.sum : equivalent function

        Examples
        --------
        >>> x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
        >>> x
        masked_array(
          data=[[1, --, 3],
                [--, 5, --],
                [7, --, 9]],
          mask=[[False,  True, False],
                [ True, False,  True],
                [False,  True, False]],
          fill_value=999999)
        >>> x.sum()
        25
        >>> x.sum(axis=1)
        masked_array(data=[4, 5, 16],
                     mask=[False, False, False],
               fill_value=999999)
        >>> x.sum(axis=0)
        masked_array(data=[8, 5, 12],
                     mask=[False, False, False],
               fill_value=999999)
        >>> print(type(x.sum(axis=0, dtype=np.int64)[0]))
        <class 'numpy.int64'>

        """
        # 根据 keepdims 参数确定 kwargs 字典的设置，若未指定则为空字典
        kwargs = {} if keepdims is np._NoValue else {'keepdims': keepdims}

        # 获取当前对象的掩码
        _mask = self._mask
        # 根据指定的轴和 kwargs 更新掩码
        newmask = _check_mask_axis(_mask, axis, **kwargs)

        # 如果未指定输出
        if out is None:
            # 对填充了 0 的数组求和，返回结果
            result = self.filled(0).sum(axis, dtype=dtype, **kwargs)
            # 获取结果的维度信息
            rndim = getattr(result, 'ndim', 0)
            # 如果结果具有维度，则将其视为当前对象的类型，并设置新的掩码
            if rndim:
                result = result.view(type(self))
                result.__setmask__(newmask)
            # 如果有新的掩码，则返回 masked 对象
            elif newmask:
                result = masked
            return result
        
        # 如果有指定输出
        # 对填充了 0 的数组求和，结果写入指定的输出数组
        result = self.filled(0).sum(axis, dtype=dtype, out=out, **kwargs)
        # 如果输出是 MaskedArray 类型
        if isinstance(out, MaskedArray):
            # 获取输出的掩码
            outmask = getmask(out)
            # 如果输出掩码不存在，则创建一个新的与输出形状相同的掩码
            if outmask is nomask:
                outmask = out._mask = make_mask_none(out.shape)
            # 将新的掩码展平并赋给输出掩码
            outmask.flat = newmask
        return out
    def cumsum(self, axis=None, dtype=None, out=None):
        """
        Return the cumulative sum of the array elements over the given axis.

        Masked values are set to 0 internally during the computation.
        However, their position is saved, and the result will be masked at
        the same locations.

        Refer to `numpy.cumsum` for full documentation.

        Notes
        -----
        The mask is lost if `out` is not a valid :class:`ma.MaskedArray` !

        Arithmetic is modular when using integer types, and no error is
        raised on overflow.

        See Also
        --------
        numpy.ndarray.cumsum : corresponding function for ndarrays
        numpy.cumsum : equivalent function

        Examples
        --------
        >>> marr = np.ma.array(np.arange(10), mask=[0,0,0,1,1,1,0,0,0,0])
        >>> marr.cumsum()
        masked_array(data=[0, 1, 3, --, --, --, 9, 16, 24, 33],
                     mask=[False, False, False,  True,  True,  True, False, False,
                           False, False],
               fill_value=999999)

        """
        # 使用 filled 方法将被遮蔽的值填充为 0，然后对数组元素沿指定轴进行累积求和
        result = self.filled(0).cumsum(axis=axis, dtype=dtype, out=out)
        # 如果指定了输出对象 out
        if out is not None:
            # 如果 out 是 MaskedArray 类型的实例，则将当前对象的遮蔽信息设置给 out
            if isinstance(out, MaskedArray):
                out.__setmask__(self.mask)
            # 返回输出对象 out
            return out
        # 将结果视图转换为当前对象的类型，并设置遮蔽信息为当前对象的遮蔽信息
        result = result.view(type(self))
        result.__setmask__(self._mask)
        # 返回处理后的结果
        return result

    def prod(self, axis=None, dtype=None, out=None, keepdims=np._NoValue):
        """
        Return the product of the array elements over the given axis.

        Masked elements are set to 1 internally for computation.

        Refer to `numpy.prod` for full documentation.

        Notes
        -----
        Arithmetic is modular when using integer types, and no error is raised
        on overflow.

        See Also
        --------
        numpy.ndarray.prod : corresponding function for ndarrays
        numpy.prod : equivalent function
        """
        # 根据 keepdims 参数的值构造关键字参数字典
        kwargs = {} if keepdims is np._NoValue else {'keepdims': keepdims}

        # 获取当前对象的遮蔽信息
        _mask = self._mask
        # 根据轴和关键字参数确定新的遮蔽信息
        newmask = _check_mask_axis(_mask, axis, **kwargs)
        
        # 如果没有指定输出对象 out
        if out is None:
            # 使用 filled 方法将被遮蔽的元素填充为 1，然后对数组元素沿指定轴进行累积乘积计算
            result = self.filled(1).prod(axis, dtype=dtype, **kwargs)
            # 获取结果的维度数
            rndim = getattr(result, 'ndim', 0)
            # 如果结果有维度
            if rndim:
                # 将结果视图转换为当前对象的类型，并设置遮蔽信息为新的遮蔽信息
                result = result.view(type(self))
                result.__setmask__(newmask)
            # 如果新的遮蔽信息存在，则将结果设为遮蔽
            elif newmask:
                result = masked
            # 返回处理后的结果
            return result
        
        # 如果有指定输出对象 out
        # 使用 filled 方法将被遮蔽的元素填充为 1，然后对数组元素沿指定轴进行累积乘积计算，并将结果存入 out
        result = self.filled(1).prod(axis, dtype=dtype, out=out, **kwargs)
        # 如果输出对象是 MaskedArray 类型的实例
        if isinstance(out, MaskedArray):
            # 获取输出对象的遮蔽信息
            outmask = getmask(out)
            # 如果输出对象的遮蔽信息为 nomask，则设置输出对象的遮蔽信息为全空遮蔽信息
            if outmask is nomask:
                outmask = out._mask = make_mask_none(out.shape)
            # 将输出对象的遮蔽信息扁平化为新的遮蔽信息
            outmask.flat = newmask
        # 返回输出对象 out
        return out
    product = prod
    # 返回沿给定轴的数组元素的累积乘积。
    def cumprod(self, axis=None, dtype=None, out=None):
        """
        返回沿着给定轴的数组元素的累积乘积。

        在计算过程中，屏蔽值在内部被设为1。
        然而，它们的位置被保存，结果将在相同位置处被屏蔽。

        完整文档请参考 `numpy.cumprod`。

        注意
        -----
        如果 `out` 不是有效的 MaskedArray，则屏蔽信息将丢失。

        当使用整数类型时，算术运算是模数的，并且在溢出时不会引发错误。

        另请参阅
        --------
        numpy.ndarray.cumprod : ndarrays 的对应函数
        numpy.cumprod : 等效函数
        """
        # 调用 filled 方法，将屏蔽的值填充为1，并计算累积乘积
        result = self.filled(1).cumprod(axis=axis, dtype=dtype, out=out)
        # 如果提供了输出数组 out
        if out is not None:
            # 如果 out 是 MaskedArray 类型的实例
            if isinstance(out, MaskedArray):
                # 将当前对象的屏蔽信息设置到 out 中
                out.__setmask__(self._mask)
            # 返回输出数组 out
            return out
        # 将结果转换为当前对象的类型
        result = result.view(type(self))
        # 将当前对象的屏蔽信息设置到结果中
        result.__setmask__(self._mask)
        # 返回结果
        return result
    def mean(self, axis=None, dtype=None, out=None, keepdims=np._NoValue):
        """
        Returns the average of the array elements along given axis.

        Masked entries are ignored, and result elements which are not
        finite will be masked.

        Refer to `numpy.mean` for full documentation.

        See Also
        --------
        numpy.ndarray.mean : corresponding function for ndarrays
        numpy.mean : Equivalent function
        numpy.ma.average : Weighted average.

        Examples
        --------
        >>> a = np.ma.array([1,2,3], mask=[False, False, True])
        >>> a
        masked_array(data=[1, 2, --],
                     mask=[False, False,  True],
               fill_value=999999)
        >>> a.mean()
        1.5

        """
        # 如果 keepdims 未指定，则将 kwargs 设为空字典；否则设置 keepdims 为指定值
        kwargs = {} if keepdims is np._NoValue else {'keepdims': keepdims}
        # 如果数组没有遮盖（mask），直接调用父类 ndarray 的 mean 方法计算平均值
        if self._mask is nomask:
            result = super().mean(axis=axis, dtype=dtype, **kwargs)[()]
        else:
            # 判断是否需要使用 float16 类型进行结果计算
            is_float16_result = False
            if dtype is None:
                if issubclass(self.dtype.type, (ntypes.integer, ntypes.bool)):
                    dtype = mu.dtype('f8')
                elif issubclass(self.dtype.type, ntypes.float16):
                    dtype = mu.dtype('f4')
                    is_float16_result = True
            # 计算总和，使用指定的 dtype
            dsum = self.sum(axis=axis, dtype=dtype, **kwargs)
            # 计算非遮盖（non-masked）元素的数量
            cnt = self.count(axis=axis, **kwargs)
            # 如果 cnt 是标量且为 0，则结果设为遮盖状态
            if cnt.shape == () and (cnt == 0):
                result = masked
            elif is_float16_result:
                # 如果结果需要为 float16 类型，使用特定的类型转换计算平均值
                result = self.dtype.type(dsum * 1. / cnt)
            else:
                # 否则直接计算平均值
                result = dsum * 1. / cnt
        # 如果指定了输出数组 out，则将计算结果赋值给 out，并根据情况设置遮盖掩码
        if out is not None:
            out.flat = result
            if isinstance(out, MaskedArray):
                outmask = getmask(out)
                if outmask is nomask:
                    outmask = out._mask = make_mask_none(out.shape)
                outmask.flat = getmask(result)
            return out
        # 返回计算得到的平均值结果
        return result
    def anom(self, axis=None, dtype=None):
        """
        计算沿指定轴线的异常值（与算术平均值的偏差）。

        返回一个异常值数组，其形状与输入数组相同，其中算术平均值沿指定轴计算。

        Parameters
        ----------
        axis : int, optional
            进行异常值计算的轴线。
            默认情况下，使用扁平化数组的平均值作为参考。
        dtype : dtype, optional
            在计算方差时使用的数据类型。对于整数类型的数组，默认为 float32；
            对于浮点类型的数组，使用相同的类型。

        Returns
        -------
        masked_array
            带有异常值的数组对象，包括数据、掩码和填充值。

        See Also
        --------
        mean : 计算数组的平均值。

        Examples
        --------
        >>> a = np.ma.array([1,2,3])
        >>> a.anom()
        masked_array(data=[-1.,  0.,  1.],
                     mask=False,
                     fill_value=1e+20)

        """
        # 计算沿指定轴的平均值
        m = self.mean(axis, dtype)
        # 如果未指定轴，则直接减去整体平均值 m
        if not axis:
            return self - m
        # 否则，在指定轴上进行广播，然后减去相应的平均值 m
        else:
            return self - expand_dims(m, axis)
    def var(self, axis=None, dtype=None, out=None, ddof=0,
            keepdims=np._NoValue, mean=np._NoValue):
        """
        Returns the variance of the array elements along given axis.

        Masked entries are ignored, and result elements which are not
        finite will be masked.

        Refer to `numpy.var` for full documentation.

        See Also
        --------
        numpy.ndarray.var : corresponding function for ndarrays
        numpy.var : Equivalent function
        """
        kwargs = {}  # 创建一个空的关键字参数字典

        if keepdims is not np._NoValue:
            kwargs['keepdims'] = keepdims  # 如果 keepdims 不是默认值，则在 kwargs 中设置 'keepdims' 键

        # 简单情况：无掩码，正常处理
        if self._mask is nomask:

            if mean is not np._NoValue:
                kwargs['mean'] = mean  # 如果 mean 不是默认值，则在 kwargs 中设置 'mean' 键

            ret = super().var(axis=axis, dtype=dtype, out=out, ddof=ddof,
                              **kwargs)[()]  # 调用父类的 var 方法计算方差，并获取结果作为标量
            if out is not None:
                if isinstance(out, MaskedArray):
                    out.__setmask__(nomask)  # 如果输出是 MaskedArray 类型，则清除其掩码
                return out  # 返回输出对象
            return ret  # 返回计算得到的方差值

        # 存在掩码数据
        cnt = self.count(axis=axis, **kwargs) - ddof  # 计算未被掩码遮盖的数据点数

        if mean is not np._NoValue:
            danom = self - mean  # 使用指定的均值计算偏差
        else:
            danom = self - self.mean(axis, dtype, keepdims=True)  # 使用数组的均值计算偏差

        if iscomplexobj(self):
            danom = umath.absolute(danom) ** 2  # 如果数组是复数类型，则计算绝对值的平方
        else:
            danom *= danom  # 否则计算偏差的平方
        dvar = divide(danom.sum(axis, **kwargs), cnt).view(type(self))  # 计算方差并将结果转换为 self 类型的视图
        # 如果输出是多维的，则根据条件设置掩码
        if dvar.ndim:
            dvar._mask = mask_or(self._mask.all(axis, **kwargs), (cnt <= 0))  # 根据掩码和数据点数设置 dvar 的掩码
            dvar._update_from(self)  # 更新 dvar 对象的信息
        elif getmask(dvar):
            # 如果标量被掩码遮盖，则确保返回掩码
            dvar = masked
            if out is not None:
                if isinstance(out, MaskedArray):
                    out.flat = 0
                    out.__setmask__(True)  # 设置输出的掩码
                elif out.dtype.kind in 'biu':
                    errmsg = "Masked data information would be lost in one or "\
                             "more location."
                    raise MaskError(errmsg)  # 抛出异常，因为数据会丢失
                else:
                    out.flat = np.nan  # 将输出设置为 NaN
                return out  # 返回输出对象
        # 如果有显式输出
        if out is not None:
            # 设置数据
            out.flat = dvar
            # 如果需要，设置掩码
            if isinstance(out, MaskedArray):
                out.__setmask__(dvar.mask)  # 设置输出的掩码
            return out  # 返回输出对象
        return dvar  # 返回计算得到的方差对象
    var.__doc__ = np.var.__doc__  # 将 var 方法的文档字符串设置为 numpy.var 方法的文档字符串
    def std(self, axis=None, dtype=None, out=None, ddof=0,
            keepdims=np._NoValue, mean=np._NoValue):
        """
        Returns the standard deviation of the array elements along given axis.

        Masked entries are ignored.

        Refer to `numpy.std` for full documentation.

        See Also
        --------
        numpy.ndarray.std : corresponding function for ndarrays
        numpy.std : Equivalent function
        """
        # 如果 keepdims 不是默认值 np._NoValue，则创建一个包含 keepdims 参数的字典
        kwargs = {} if keepdims is np._NoValue else {'keepdims': keepdims}

        # 计算方差，传递给 var 方法
        dvar = self.var(axis, dtype, out, ddof, **kwargs)
        # 如果方差为 masked，则直接返回
        if dvar is not masked:
            # 如果指定了输出 out，则直接将 out 数组平方根值写入 out，unsafe 转换
            if out is not None:
                np.power(out, 0.5, out=out, casting='unsafe')
                return out
            # 否则计算方差的平方根并返回
            dvar = sqrt(dvar)
        return dvar

    def round(self, decimals=0, out=None):
        """
        Return each element rounded to the given number of decimals.

        Refer to `numpy.around` for full documentation.

        See Also
        --------
        numpy.ndarray.round : corresponding function for ndarrays
        numpy.around : equivalent function

        Examples
        --------
        >>> import numpy.ma as ma
        >>> x = ma.array([1.35, 2.5, 1.5, 1.75, 2.25, 2.75],
        ...              mask=[0, 0, 0, 1, 0, 0])
        >>> ma.round(x)
        masked_array(data=[1.0, 2.0, 2.0, --, 2.0, 3.0],
                     mask=[False, False, False,  True, False, False],
                fill_value=1e+20)

        """
        # 调用 self._data 上的 round 方法，返回一个新的 MaskedArray 对象
        result = self._data.round(decimals=decimals, out=out).view(type(self))
        # 如果结果的维度大于 0，则更新 _mask 和从 self 更新结果
        if result.ndim > 0:
            result._mask = self._mask
            result._update_from(self)
        # 如果 self 被屏蔽，则返回 masked
        elif self._mask:
            result = masked
        # 如果没有指定输出 out，则返回结果
        if out is None:
            return result
        # 如果 out 是 MaskedArray 类型，则设置其掩码为 self._mask
        if isinstance(out, MaskedArray):
            out.__setmask__(self._mask)
        # 返回指定的输出 out
        return out
    def argmin(self, axis=None, fill_value=None, out=None, *,
                keepdims=np._NoValue):
        """
        Return array of indices to the minimum values along the given axis.

        Parameters
        ----------
        axis : {None, integer}
            如果为 None，则返回沿着扁平化数组的最小值索引；否则沿指定的轴返回最小值索引。
            Specifies the axis along which the minimum is found. If None, the index is into the flattened array.
        fill_value : scalar or None, optional
            用于填充掩码值的值。如果为 None，则使用 minimum_fill_value(self._data) 的输出。
            Value used to fill in masked values. If None, the output of minimum_fill_value(self._data) is used.
        out : {None, array}, optional
            可以放置结果的数组。其类型被保留，必须具有正确的形状来容纳输出。
            Array into which the result can be placed. Its type is preserved and it must be of the right shape to hold the output.
        keepdims : {np._NoValue, bool}, optional
            如果为 np._NoValue，则不保留维度信息；否则按指定的布尔值保留维度信息。
            If np._NoValue, dimensions are not kept; otherwise, dimensions are kept based on the specified boolean value.

        Returns
        -------
        ndarray or scalar
            如果输入是多维的，则返回沿指定轴的最小值的索引组成的新 ndarray；否则返回一个标量，表示沿指定轴的最小值的索引。
            If input is multi-dimensional, returns a new ndarray of indices to the minimum values along the given axis. Otherwise, returns a scalar of index to the minimum value along the given axis.

        Examples
        --------
        >>> x = np.ma.array(np.arange(4), mask=[1,1,0,0])
        >>> x.shape = (2,2)
        >>> x
        masked_array(
          data=[[--, --],
                [2, 3]],
          mask=[[ True,  True],
                [False, False]],
          fill_value=999999)
        >>> x.argmin(axis=0, fill_value=-1)
        array([0, 0])
        >>> x.argmin(axis=0, fill_value=9)
        array([1, 1])

        """
        # 如果 fill_value 为 None，则使用 minimum_fill_value(self) 函数计算填充值
        if fill_value is None:
            fill_value = minimum_fill_value(self)
        # 将掩码数组中的填充值填充为指定的 fill_value，并将其视图转换为普通的 ndarray 对象
        d = self.filled(fill_value).view(ndarray)
        # 确定是否保留维度信息，若 keepdims 为 np._NoValue 则不保留，否则根据布尔值决定是否保留
        keepdims = False if keepdims is np._NoValue else bool(keepdims)
        # 调用 ndarray 对象的 argmin 方法来计算沿指定轴的最小值索引，返回结果
        return d.argmin(axis, out=out, keepdims=keepdims)
    def argmax(self, axis=None, fill_value=None, out=None, *,
                keepdims=np._NoValue):
        """
        Returns array of indices of the maximum values along the given axis.
        Masked values are treated as if they had the value fill_value.

        Parameters
        ----------
        axis : {None, integer}
            If None, the index is into the flattened array, otherwise along
            the specified axis
        fill_value : scalar or None, optional
            Value used to fill in the masked values.  If None, the output of
            maximum_fill_value(self._data) is used instead.
        out : {None, array}, optional
            Array into which the result can be placed. Its type is preserved
            and it must be of the right shape to hold the output.

        Returns
        -------
        index_array : {integer_array}

        Examples
        --------
        >>> a = np.arange(6).reshape(2,3)
        >>> a.argmax()
        5
        >>> a.argmax(0)
        array([1, 1, 1])
        >>> a.argmax(1)
        array([2, 2])

        """
        # 如果 fill_value 为 None，则使用 self._data 的最大填充值
        if fill_value is None:
            fill_value = maximum_fill_value(self._data)
        # 使用填充后的数据（没有掩码）创建一个 ndarray 对象
        d = self.filled(fill_value).view(ndarray)
        # 如果 keepdims 是 np._NoValue，则将其设为 False，否则转换为布尔值
        keepdims = False if keepdims is np._NoValue else bool(keepdims)
        # 调用 ndarray 的 argmax 方法获取最大值的索引数组
        return d.argmax(axis, out=out, keepdims=keepdims)

    def partition(self, *args, **kwargs):
        # 发出警告，表明 'partition' 方法将忽略掩码
        warnings.warn("Warning: 'partition' will ignore the 'mask' "
                      f"of the {self.__class__.__name__}.",
                      stacklevel=2)
        # 调用父类的 partition 方法并返回结果
        return super().partition(*args, **kwargs)

    def argpartition(self, *args, **kwargs):
        # 发出警告，表明 'argpartition' 方法将忽略掩码
        warnings.warn("Warning: 'argpartition' will ignore the 'mask' "
                      f"of the {self.__class__.__name__}.",
                      stacklevel=2)
        # 调用父类的 argpartition 方法并返回结果
        return super().argpartition(*args, **kwargs)

    def take(self, indices, axis=None, out=None, mode='raise'):
        """
        """
        # 获取 self 对象的数据和掩码
        (_data, _mask) = (self._data, self._mask)
        # 获取 indices 的掩码情况
        maskindices = getmask(indices)
        # 如果 indices 有掩码，则用 0 填充
        if maskindices is not nomask:
            indices = indices.filled(0)
        # 如果 out 为 None，则使用 _data.take 获取数据，然后转换为 cls 类型
        if out is None:
            out = _data.take(indices, axis=axis, mode=mode)[...].view(cls)
        else:
            # 否则使用 np.take 将数据放入 out 中
            np.take(_data, indices, axis=axis, mode=mode, out=out)
        # 获取输出的掩码
        if isinstance(out, MaskedArray):
            if _mask is nomask:
                outmask = maskindices
            else:
                outmask = _mask.take(indices, axis=axis, mode=mode)
                outmask |= maskindices
            # 设置输出对象的掩码
            out.__setmask__(outmask)
        # 将 0d 数组恢复为标量，以保持与 ndarray.take 的一致性
        return out[()]

    # Array methods
    # 使用 _arraymethod 方法生成 'copy' 方法
    copy = _arraymethod('copy')
    # 使用 _arraymethod 方法生成 'diagonal' 方法
    diagonal = _arraymethod('diagonal')
    # 定义一系列方法，通过字符串参数调用 _arraymethod 函数生成对应的方法
    flatten = _arraymethod('flatten')
    repeat = _arraymethod('repeat')
    squeeze = _arraymethod('squeeze')
    swapaxes = _arraymethod('swapaxes')
    # 属性 T 返回数组的转置
    T = property(fget=lambda self: self.transpose())
    # 定义 transpose 方法，通过字符串参数调用 _arraymethod 函数生成对应的方法
    transpose = _arraymethod('transpose')

    @property
    def mT(self):
        """
        Return the matrix-transpose of the masked array.

        The matrix transpose is the transpose of the last two dimensions, even
        if the array is of higher dimension.

        .. versionadded:: 2.0

        Returns
        -------
        result: MaskedArray
            The masked array with the last two dimensions transposed

        Raises
        ------
        ValueError
            If the array is of dimension less than 2.

        See Also
        --------
        ndarray.mT:
            Equivalent method for arrays
        """
        # 如果数组的维度小于 2，则抛出 ValueError 异常
        if self.ndim < 2:
            raise ValueError("matrix transpose with ndim < 2 is undefined")

        # 如果数组没有掩码，则返回转置后的 MaskedArray 对象
        if self._mask is nomask:
            return masked_array(data=self._data.mT)
        else:
            # 如果数组有掩码，则返回带有掩码的转置后的 MaskedArray 对象
            return masked_array(data=self.data.mT, mask=self.mask.mT)


    def tolist(self, fill_value=None):
        """
        Return the data portion of the masked array as a hierarchical Python list.

        Data items are converted to the nearest compatible Python type.
        Masked values are converted to `fill_value`. If `fill_value` is None,
        the corresponding entries in the output list will be ``None``.

        Parameters
        ----------
        fill_value : scalar, optional
            The value to use for invalid entries. Default is None.

        Returns
        -------
        result : list
            The Python list representation of the masked array.

        Examples
        --------
        >>> x = np.ma.array([[1,2,3], [4,5,6], [7,8,9]], mask=[0] + [1,0]*4)
        >>> x.tolist()
        [[1, None, 3], [None, 5, None], [7, None, 9]]
        >>> x.tolist(-999)
        [[1, -999, 3], [-999, 5, -999], [7, -999, 9]]

        """
        _mask = self._mask
        # 如果没有掩码，则直接返回数据部分转换为 Python 列表的结果
        if _mask is nomask:
            return self._data.tolist()
        # 如果指定了 fill_value，则将掩码填充后再转换为列表
        if fill_value is not None:
            return self.filled(fill_value).tolist()
        # 对于结构化数组，将数据转换为对象类型列表，并根据掩码设置对应位置为 None
        names = self.dtype.names
        if names:
            result = self._data.astype([(_, object) for _ in names])
            for n in names:
                result[n][_mask[n]] = None
            return result.tolist()
        # 对于标准数组，将数据转换为对象类型数组，并根据掩码设置对应位置为 None
        if _mask is nomask:
            return [None]
        # 对于一般情况下的掩码数组，将数据部分转换为对象类型数组，并根据掩码设置对应位置为 None
        inishape = self.shape
        result = np.array(self._data.ravel(), dtype=object)
        result[_mask.ravel()] = None
        result.shape = inishape
        return result.tolist()
    def tostring(self, fill_value=None, order='C'):
        r"""
        A compatibility alias for `tobytes`, with exactly the same behavior.

        Despite its name, it returns `bytes` not `str`\ s.

        .. deprecated:: 1.19.0
        """
        # 发出警告，告知函数已被弃用，建议使用`tobytes()`代替
        warnings.warn(
            "tostring() is deprecated. Use tobytes() instead.",
            DeprecationWarning, stacklevel=2)

        # 调用`tobytes()`方法并返回结果
        return self.tobytes(fill_value, order=order)

    def tobytes(self, fill_value=None, order='C'):
        """
        Return the array data as a string containing the raw bytes in the array.

        The array is filled with a fill value before the string conversion.

        .. versionadded:: 1.9.0

        Parameters
        ----------
        fill_value : scalar, optional
            Value used to fill in the masked values. Default is None, in which
            case `MaskedArray.fill_value` is used.
        order : {'C','F','A'}, optional
            Order of the data item in the copy. Default is 'C'.

            - 'C'   -- C order (row major).
            - 'F'   -- Fortran order (column major).
            - 'A'   -- Any, current order of array.
            - None  -- Same as 'A'.

        See Also
        --------
        numpy.ndarray.tobytes
        tolist, tofile

        Notes
        -----
        As for `ndarray.tobytes`, information about the shape, dtype, etc.,
        but also about `fill_value`, will be lost.

        Examples
        --------
        >>> x = np.ma.array(np.array([[1, 2], [3, 4]]), mask=[[0, 1], [1, 0]])
        >>> x.tobytes()
        b'\x01\x00\x00\x00\x00\x00\x00\x00?B\x0f\x00\x00\x00\x00\x00?B\x0f\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00'

        """
        # 返回填充后的数组数据的原始字节字符串表示
        return self.filled(fill_value).tobytes(order=order)

    def tofile(self, fid, sep="", format="%s"):
        """
        Save a masked array to a file in binary format.

        .. warning::
          This function is not implemented yet.

        Raises
        ------
        NotImplementedError
            When `tofile` is called.

        """
        # 抛出未实现错误，因为该功能尚未被实现
        raise NotImplementedError("MaskedArray.tofile() not implemented yet.")
    # 将一个掩码数组转换为灵活类型的数组。

    # 返回的灵活类型数组将有两个字段：

    # * ``_data`` 字段存储数组的 ``_data`` 部分。
    # * ``_mask`` 字段存储数组的 ``_mask`` 部分。

    def toflex(self):
        """
        Transforms a masked array into a flexible-type array.

        The flexible type array that is returned will have two fields:

        * the ``_data`` field stores the ``_data`` part of the array.
        * the ``_mask`` field stores the ``_mask`` part of the array.

        Parameters
        ----------
        None

        Returns
        -------
        record : ndarray
            A new flexible-type `ndarray` with two fields: the first element
            containing a value, the second element containing the corresponding
            mask boolean. The returned record shape matches self.shape.

        Notes
        -----
        A side-effect of transforming a masked array into a flexible `ndarray` is
        that meta information (``fill_value``, ...) will be lost.

        Examples
        --------
        >>> x = np.ma.array([[1,2,3],[4,5,6],[7,8,9]], mask=[0] + [1,0]*4)
        >>> x
        masked_array(
          data=[[1, --, 3],
                [--, 5, --],
                [7, --, 9]],
          mask=[[False,  True, False],
                [ True, False,  True],
                [False,  True, False]],
          fill_value=999999)
        >>> x.toflex()
        array([[(1, False), (2,  True), (3, False)],
               [(4,  True), (5, False), (6,  True)],
               [(7, False), (8,  True), (9, False)]],
              dtype=[('_data', '<i8'), ('_mask', '?')])

        """
        # 获取基本的数据类型
        ddtype = self.dtype
        # 确保有一个掩码
        _mask = self._mask
        if _mask is None:
            _mask = make_mask_none(self.shape, ddtype)
        # 获取掩码的数据类型
        mdtype = self._mask.dtype

        # 创建一个形状与原始数组相同，包含两个字段的新数组
        record = np.ndarray(shape=self.shape,
                            dtype=[('_data', ddtype), ('_mask', mdtype)])
        # 将数据和掩码分别赋给新数组的字段
        record['_data'] = self._data
        record['_mask'] = self._mask
        # 返回新的灵活类型数组
        return record
    torecords = toflex

    # Pickling
    def __getstate__(self):
        """Return the internal state of the masked array, for pickling
        purposes.

        """
        # 根据对象的标志位确定字符格式
        cf = 'CF'[self.flags.fnc]
        # 获取数据部分的内部状态
        data_state = super().__reduce__()[2]
        # 返回数据状态和掩码的字节表示，以及填充值
        return data_state + (getmaskarray(self).tobytes(cf), self._fill_value)

    def __setstate__(self, state):
        """Restore the internal state of the masked array, for
        pickling purposes.  ``state`` is typically the output of the
        ``__getstate__`` output, and is a 5-tuple:

        - class name
        - a tuple giving the shape of the data
        - a typecode for the data
        - a binary string for the data
        - a binary string for the mask.

        """
        # 解析状态元组
        (_, shp, typ, isf, raw, msk, flv) = state
        # 通过调用父类方法恢复数据部分的内部状态
        super().__setstate__((shp, typ, isf, raw))
        # 恢复掩码数组的状态
        self._mask.__setstate__((shp, make_mask_descr(typ), isf, msk))
        # 恢复填充值
        self.fill_value = flv
    # 定义 __reduce__ 方法，用于在序列化（pickling）时返回一个包含三个元素的元组，用于重建 MaskedArray 对象。
    def __reduce__(self):
        """Return a 3-tuple for pickling a MaskedArray.

        """
        # 返回元组，包含一个重建函数 _mareconstruct，
        # 一个元组，包含类信息、基类信息、版本信息等参数，
        # 以及调用 self.__getstate__() 返回的对象状态。
        return (_mareconstruct,
                (self.__class__, self._baseclass, (0,), 'b',),
                self.__getstate__())

    # 定义 __deepcopy__ 方法，用于深度复制 MaskedArray 对象。
    def __deepcopy__(self, memo=None):
        # 导入深度复制函数
        from copy import deepcopy
        # 创建一个新的 MaskedArray 对象的副本
        copied = MaskedArray.__new__(type(self), self, copy=True)
        # 如果 memo 参数为 None，则创建一个空字典 memo 用于记录已复制的对象
        if memo is None:
            memo = {}
        # 将当前对象的 ID 和其副本对应起来，记录到 memo 字典中
        memo[id(self)] = copied
        # 遍历当前对象的 __dict__ 属性，深度复制每个键值对并存储到副本的 __dict__ 中
        for (k, v) in self.__dict__.items():
            copied.__dict__[k] = deepcopy(v, memo)
        # 根据 numpy 的文档，对于可能包含复合类型的对象数组（object type），需要直接使用 deepcopy()
        # 不能依赖于普通的复制语义来正确处理这些对象
        if self.dtype.hasobject:
            copied._data[...] = deepcopy(copied._data)
        # 返回深度复制后的副本对象
        return copied
# 定义一个名为 _mareconstruct 的内部函数，用于从 pickle 中存储的信息构建一个新的 MaskedArray 对象。
def _mareconstruct(subtype, baseclass, baseshape, basetype,):
    """Internal function that builds a new MaskedArray from the
    information stored in a pickle.

    """
    # 使用 baseclass 创建一个新的 ndarray 对象 _data，具有给定的 baseshape 和 basetype
    _data = ndarray.__new__(baseclass, baseshape, basetype)
    # 使用 ndarray 创建一个新的 ndarray 对象 _mask，其描述符由 make_mask_descr(basetype) 生成
    _mask = ndarray.__new__(ndarray, baseshape, make_mask_descr(basetype))
    # 返回一个新的 subtype 对象，其中包含 _data 作为数据，_mask 作为掩码，basetype 作为数据类型
    return subtype.__new__(subtype, _data, mask=_mask, dtype=basetype,)


class mvoid(MaskedArray):
    """
    Fake a 'void' object to use for masked array with structured dtypes.
    """

    def __new__(self, data, mask=nomask, dtype=None, fill_value=None,
                hardmask=False, copy=False, subok=True):
        # 根据 copy 参数决定是否复制数据
        copy = None if not copy else True
        # 将 data 转换为 ndarray 对象 _data，使用指定的 dtype，复制和子类化行为由 subok 控制
        _data = np.array(data, copy=copy, subok=subok, dtype=dtype)
        # 将 _data 视图化为当前类对象 self
        _data = _data.view(self)
        # 设置 _data 的硬掩码属性
        _data._hardmask = hardmask
        # 如果 mask 不是 nomask，则处理掩码
        if mask is not nomask:
            if isinstance(mask, np.void):
                # 如果 mask 是 np.void 类型，则直接赋值给 _data 的掩码属性
                _data._mask = mask
            else:
                try:
                    # 尝试将 mask 转换为 np.void 类型
                    mdtype = make_mask_descr(dtype)
                    _data._mask = np.array(mask, dtype=mdtype)[()]
                except TypeError:
                    # 如果转换失败，则转换为 void 类型
                    _data._mask = np.void(mask)
        # 如果 fill_value 不为 None，则设置 _data 的填充值属性
        if fill_value is not None:
            _data.fill_value = fill_value
        # 返回创建的 _data 对象
        return _data

    @property
    def _data(self):
        # 返回父类 MaskedArray 的 _data 属性的内容
        return super()._data[()]

    def __getitem__(self, indx):
        """
        Get the index.

        """
        m = self._mask
        # 如果 m[indx] 是 ndarray 类型，则处理多维字段的情况
        if isinstance(m[indx], ndarray):
            # 创建一个新的 masked_array 对象，使用 self._data[indx] 作为数据，m[indx] 作为掩码
            return masked_array(
                data=self._data[indx], mask=m[indx],
                fill_value=self._fill_value[indx],
                hard_mask=self._hardmask)
        # 如果 m 不是 nomask，并且 m[indx] 是 True，则返回 masked
        if m is not nomask and m[indx]:
            return masked
        # 否则返回 self._data[indx]
        return self._data[indx]

    def __setitem__(self, indx, value):
        # 设置 self._data[indx] 的值为 value
        self._data[indx] = value
        # 如果 self._hardmask 为 True，则将 self._mask[indx] 的掩码与 value 的掩码合并
        if self._hardmask:
            self._mask[indx] |= getattr(value, "_mask", False)
        else:
            # 否则直接将 self._mask[indx] 设为 value 的掩码
            self._mask[indx] = getattr(value, "_mask", False)

    def __str__(self):
        m = self._mask
        # 如果 m 是 nomask，则返回 self._data 的字符串表示形式
        if m is nomask:
            return str(self._data)

        # 替换 self._data 的 dtype 中的字段类型为 "O"
        rdtype = _replace_dtype_fields(self._data.dtype, "O")
        # 将 super() 的 _data 属性视图化为 ndarray 对象
        data_arr = super()._data
        # 将 data_arr 转换为 rdtype 类型的数组 res
        res = data_arr.astype(rdtype)
        # 使用 _recursive_printoption 输出 res、self._mask 和 masked_print_option
        _recursive_printoption(res, self._mask, masked_print_option)
        # 返回 res 的字符串表示形式
        return str(res)

    __repr__ = __str__
    def __iter__(self):
        """
        定义 mvoid 对象的迭代器方法。

        Returns
        -------
        yield from _data 或者 yield masked/d
            如果没有掩码 (_mask is nomask)，则逐个产生 _data 中的元素。
            否则，对 _data 和 _mask 进行 zip，根据掩码条件产生相应的值或 masked 对象。
        """
        (_data, _mask) = (self._data, self._mask)
        if _mask is nomask:
            yield from _data
        else:
            for (d, m) in zip(_data, _mask):
                if m:
                    yield masked
                else:
                    yield d

    def __len__(self):
        """
        返回 mvoid 对象的长度。

        Returns
        -------
        int
            _data 的长度
        """
        return self._data.__len__()

    def filled(self, fill_value=None):
        """
        使用给定值填充掩码字段，返回一个填充后的副本。

        Parameters
        ----------
        fill_value : array_like, optional
            用于填充无效条目的值。可以是标量或非标量。如果是非标量，则填充的数组应该与输入数组可广播。默认为 None，
            此时使用 `fill_value` 属性。

        Returns
        -------
        filled_void
            一个 `np.void` 对象，填充后的副本。

        See Also
        --------
        MaskedArray.filled
        """
        return asarray(self).filled(fill_value)[()]

    def tolist(self):
        """
        将 mvoid 对象转换为元组。

        掩码字段被替换为 None。

        Returns
        -------
        returned_tuple
            字段组成的元组。
        """
        _mask = self._mask
        if _mask is nomask:
            return self._data.tolist()
        result = []
        for (d, m) in zip(self._data, self._mask):
            if m:
                result.append(None)
            else:
                # .item() 确保返回标准的 Python 对象
                result.append(d.item())
        return tuple(result)
##############################################################################
#                                Shortcuts                                   #
##############################################################################

# 定义函数，用于检测输入是否为 MaskedArray 的实例
def isMaskedArray(x):
    """
    Test whether input is an instance of MaskedArray.

    This function returns True if `x` is an instance of MaskedArray
    and returns False otherwise.  Any object is accepted as input.

    Parameters
    ----------
    x : object
        Object to test.

    Returns
    -------
    result : bool
        True if `x` is a MaskedArray.

    See Also
    --------
    isMA : Alias to isMaskedArray.
    isarray : Alias to isMaskedArray.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> a = np.eye(3, 3)
    >>> a
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> m = ma.masked_values(a, 0)
    >>> m
    masked_array(
      data=[[1.0, --, --],
            [--, 1.0, --],
            [--, --, 1.0]],
      mask=[[False,  True,  True],
            [ True, False,  True],
            [ True,  True, False]],
      fill_value=0.0)
    >>> ma.isMaskedArray(a)
    False
    >>> ma.isMaskedArray(m)
    True
    >>> ma.isMaskedArray([0, 1, 2])
    False

    """
    return isinstance(x, MaskedArray)

# 定义别名函数，与 isMaskedArray 等价
isarray = isMaskedArray
# 向后兼容性别名
isMA = isMaskedArray  # backward compatibility


class MaskedConstant(MaskedArray):
    # 单一的 np.ma.masked 实例
    __singleton = None

    @classmethod
    def __has_singleton(cls):
        # 第二种情况确保 `cls.__singleton` 不只是超类单例的视图
        return cls.__singleton is not None and type(cls.__singleton) is cls

    def __new__(cls):
        if not cls.__has_singleton():
            # 定义被屏蔽的单例为一个 float，以提高优先级
            # 注意有时与类型比较可能会有些棘手
            data = np.array(0.)
            mask = np.array(True)

            # 防止任何修改
            data.flags.writeable = False
            mask.flags.writeable = False

            # 不要回退到 MaskedArray.__new__(MaskedConstant)，因为
            # 这可能会使其混淆 - 通过这种方式，构建完全在我们的控制之下
            cls.__singleton = MaskedArray(data, mask=mask).view(cls)

        return cls.__singleton
    # 当数组对象被finalize时调用的特殊方法，用于处理继承自父类的属性以及特定情况下的处理逻辑
    def __array_finalize__(self, obj):
        # 如果数组对象没有单例模式，则继承父类的finalize方法
        if not self.__has_singleton():
            # 处理在__new__中的`.view`情况，需要将属性正常复制
            return super().__array_finalize__(obj)
        # 如果当前对象就是单例模式，一般情况下不会发生，为安全起见，不做处理
        elif self is self.__singleton:
            pass
        else:
            # 在其他情况下，将对象下降级为MaskedArray，以避免重复的maskedconstant
            self.__class__ = MaskedArray
            MaskedArray.__array_finalize__(self, obj)

    # 对数组进行包装的特殊方法，返回一个MaskedArray对象
    def __array_wrap__(self, obj, context=None, return_scalar=False):
        return self.view(MaskedArray).__array_wrap__(obj, context)

    # 返回对象的字符串表示形式，基于masked_print_option._display的值
    def __str__(self):
        return str(masked_print_option._display)

    # 返回对象的“官方”字符串表示形式，如果是MaskedConstant的单例，则返回'masked'，否则返回对象的默认表示
    def __repr__(self):
        if self is MaskedConstant.__singleton:
            return 'masked'
        else:
            # 如果是子类或出现异常情况，明确表明是对象的默认表示形式
            return object.__repr__(self)

    # 对对象进行格式化处理的特殊方法，覆盖了ndarray.__format__方法，不支持格式字符以避免不确定性
    def __format__(self, format_spec):
        # 尝试使用object.__format__方法进行格式化，不支持格式字符会引发TypeError异常
        try:
            return object.__format__(self, format_spec)
        except TypeError:
            # 给出警告，表示格式字符串对MaskedConstant无效，将来可能会引发错误或产生不同的行为
            warnings.warn(
                "Format strings passed to MaskedConstant are ignored, but in future may "
                "error or produce different behavior",
                FutureWarning, stacklevel=2
            )
            # 返回默认的格式化结果，忽略格式字符串
            return object.__format__(self, "")

    # 自定义的__reduce__方法，用于序列化对象，返回一个元组，第一个元素是类，第二个元素是空元组
    def __reduce__(self):
        """Override of MaskedArray's __reduce__.
        """
        return (self.__class__, ())

    # 就地操作不起作用，需要覆盖以避免修改只读的数据和掩码数组
    def __iop__(self, other):
        return self
    # 将所有的就地操作方法指向__iop__，表示这些操作对MaskedConstant对象没有实际效果
    __iadd__ = \
    __isub__ = \
    __imul__ = \
    __ifloordiv__ = \
    __itruediv__ = \
    __ipow__ = \
        __iop__
    del __iop__  # 删除__iop__方法的引用，避免误用

    # 返回对象的浅拷贝，对于maskedconstant来说，作为标量，拷贝操作无需复制数据
    def copy(self, *args, **kwargs):
        """ Copy is a no-op on the maskedconstant, as it is a scalar """
        return self

    # 返回对象的浅拷贝，与copy方法功能一致，保留接口一致性
    def __copy__(self):
        return self

    # 返回对象的深拷贝，与copy方法功能一致，保留接口一致性
    def __deepcopy__(self, memo):
        return self
    # 定义 __setattr__ 方法，用于设置对象属性
    def __setattr__(self, attr, value):
        # 检查是否存在单例对象
        if not self.__has_singleton():
            # 如果没有单例对象，允许初始化单例
            return super().__setattr__(attr, value)
        elif self is self.__singleton:
            # 如果当前对象是单例对象，则抛出属性错误异常
            raise AttributeError(
                f"attributes of {self!r} are not writeable")
        else:
            # 如果出现重复实例，可能是从 __array_finalize__ 方法中进入，
            # 在那里我们设置 __class__ 属性
            return super().__setattr__(attr, value)
masked = masked_singleton = MaskedConstant()


# 创建一个名为 masked 的变量，并将其设为 MaskedConstant 的实例，
# 同时将 masked_singleton 也设置为这个实例的引用
masked = masked_singleton = MaskedConstant()



masked_array = MaskedArray


# 创建一个名为 masked_array 的变量，并将其设为 MaskedArray 类的引用
masked_array = MaskedArray



def array(data, dtype=None, copy=False, order=None,
          mask=nomask, fill_value=None, keep_mask=True,
          hard_mask=False, shrink=True, subok=True, ndmin=0):
    """
    Shortcut to MaskedArray.

    The options are in a different order for convenience and backwards
    compatibility.

    """
    return MaskedArray(data, mask=mask, dtype=dtype, copy=copy,
                       subok=subok, keep_mask=keep_mask,
                       hard_mask=hard_mask, fill_value=fill_value,
                       ndmin=ndmin, shrink=shrink, order=order)


# 定义一个名为 array 的函数，提供创建 MaskedArray 的快捷方式
# 参数顺序不同于原始 MaskedArray，为方便和向后兼容性而设定
def array(data, dtype=None, copy=False, order=None,
          mask=nomask, fill_value=None, keep_mask=True,
          hard_mask=False, shrink=True, subok=True, ndmin=0):
    """
    Shortcut to MaskedArray.

    The options are in a different order for convenience and backwards
    compatibility.

    """
    # 调用 MaskedArray 构造函数并返回结果
    return MaskedArray(data, mask=mask, dtype=dtype, copy=copy,
                       subok=subok, keep_mask=keep_mask,
                       hard_mask=hard_mask, fill_value=fill_value,
                       ndmin=ndmin, shrink=shrink, order=order)



array.__doc__ = masked_array.__doc__


# 将 array 函数的文档字符串设为 masked_array 类的文档字符串
array.__doc__ = masked_array.__doc__



def is_masked(x):
    """
    Determine whether input has masked values.

    Accepts any object as input, but always returns False unless the
    input is a MaskedArray containing masked values.

    Parameters
    ----------
    x : array_like
        Array to check for masked values.

    Returns
    -------
    result : bool
        True if `x` is a MaskedArray with masked values, False otherwise.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> x = ma.masked_equal([0, 1, 0, 2, 3], 0)
    >>> x
    masked_array(data=[--, 1, --, 2, 3],
                 mask=[ True, False,  True, False, False],
           fill_value=0)
    >>> ma.is_masked(x)
    True
    >>> x = ma.masked_equal([0, 1, 0, 2, 3], 42)
    >>> x
    masked_array(data=[0, 1, 0, 2, 3],
                 mask=False,
           fill_value=42)
    >>> ma.is_masked(x)
    False

    Always returns False if `x` isn't a MaskedArray.

    >>> x = [False, True, False]
    >>> ma.is_masked(x)
    False
    >>> x = 'a string'
    >>> ma.is_masked(x)
    False

    """
    m = getmask(x)
    if m is nomask:
        return False
    elif m.any():
        return True
    return False


# 定义一个名为 is_masked 的函数，用于检查输入是否含有掩码值
def is_masked(x):
    """
    Determine whether input has masked values.

    Accepts any object as input, but always returns False unless the
    input is a MaskedArray containing masked values.

    Parameters
    ----------
    x : array_like
        Array to check for masked values.

    Returns
    -------
    result : bool
        True if `x` is a MaskedArray with masked values, False otherwise.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> x = ma.masked_equal([0, 1, 0, 2, 3], 0)
    >>> x
    masked_array(data=[--, 1, --, 2, 3],
                 mask=[ True, False,  True, False, False],
           fill_value=0)
    >>> ma.is_masked(x)
    True
    >>> x = ma.masked_equal([0, 1, 0, 2, 3], 42)
    >>> x
    masked_array(data=[0, 1, 0, 2, 3],
                 mask=False,
           fill_value=42)
    >>> ma.is_masked(x)
    False

    Always returns False if `x` isn't a MaskedArray.

    >>> x = [False, True, False]
    >>> ma.is_masked(x)
    False
    >>> x = 'a string'
    >>> ma.is_masked(x)
    False

    """
    # 获取输入对象 x 的掩码
    m = getmask(x)
    # 如果掩码为 nomask，则返回 False
    if m is nomask:
        return False
    # 如果掩码中有任何值，则返回 True
    elif m.any():
        return True
    # 否则返回 False
    return False



##############################################################################
#                             Extrema functions                              #
##############################################################################


# 分割线，标识极值函数的开始
##############################################################################
#                             Extrema functions                              #
##############################################################################



class _extrema_operation(_MaskedUFunc):
    """
    Generic class for maximum/minimum functions.

    .. note::
      This is the base class for `_maximum_operation` and
      `_minimum_operation`.

    """
    def __init__(self, ufunc, compare, fill_value):
        super().__init__(ufunc)
        self.compare = compare
        self.fill_value_func = fill_value

    def __call__(self, a, b):
        "Executes the call behavior."

        return where(self.compare(a, b), a, b)


# 定义一个名为 _extrema_operation 的类，用于实现最大/最小函数的通用操作
# 继承自 _MaskedUFunc 类
class _extrema_operation(_MaskedUFunc):
    """
    Generic class for maximum/minimum functions.

    .. note::
      This is the base class for `_maximum_operation` and
      `_minimum_operation`.

    """
    # 构造函数，初始化操作符和填充值函数
    def __init__(self, ufunc, compare, fill_value):
        super().__init__(ufunc)
        self.compare = compare
        self.fill_value_func = fill_value

    # 调用实例时执行的方法
    def __call__(self, a, b):
        "Executes the call behavior."

        # 根据比较结果选择返回 a 或 b
        return where(self.compare(a, b), a, b)
    def reduce(self, target, axis=np._NoValue):
        "Reduce target along the given axis."
        # 将目标转换为数组，确保是 numpy 数组
        target = narray(target, copy=None, subok=True)
        # 获取目标数组的掩码
        m = getmask(target)

        # 如果未指定 axis 且目标数组维度大于 1
        if axis is np._NoValue and target.ndim > 1:
            # 2017-05-06, Numpy 1.13.0: 在默认未来版本中会对 axis 进行警告
            warnings.warn(
                f"In the future the default for ma.{self.__name__}.reduce will be axis=0, "
                f"not the current None, to match np.{self.__name__}.reduce. "
                "Explicitly pass 0 or None to silence this warning.",
                MaskedArrayFutureWarning, stacklevel=2)
            # 将 axis 设置为 None，以匹配未来的 numpy 行为
            axis = None

        # 根据是否指定 axis，设置参数字典
        if axis is not np._NoValue:
            kwargs = dict(axis=axis)
        else:
            kwargs = dict()

        # 如果目标数组没有掩码
        if m is nomask:
            # 调用 self.f.reduce 方法进行数组的约简操作
            t = self.f.reduce(target, **kwargs)
        else:
            # 将目标数组中的掩码填充后，转换为与原类型相同的视图
            target = target.filled(
                self.fill_value_func(target)).view(type(target))
            # 对填充后的数组进行约简操作
            t = self.f.reduce(target, **kwargs)
            # 对掩码数组进行逻辑与运算约简操作
            m = umath.logical_and.reduce(m, **kwargs)
            # 如果 t 具有 '_mask' 属性，设置其掩码属性为 m
            if hasattr(t, '_mask'):
                t._mask = m
            # 否则，如果 m 为真，则将 t 设置为 masked
            elif m:
                t = masked
        # 返回约简后的结果 t
        return t

    def outer(self, a, b):
        "Return the function applied to the outer product of a and b."
        # 获取数组 a 和 b 的掩码
        ma = getmask(a)
        mb = getmask(b)
        # 如果 a 和 b 均无掩码
        if ma is nomask and mb is nomask:
            m = nomask
        else:
            # 获取 a 和 b 的掩码数组，对其进行逻辑或运算得到 m
            ma = getmaskarray(a)
            mb = getmaskarray(b)
            m = logical_or.outer(ma, mb)
        # 对 a 和 b 执行外积运算，结果转换为 MaskedArray 类型
        result = self.f.outer(filled(a), filled(b))
        if not isinstance(result, MaskedArray):
            result = result.view(MaskedArray)
        # 设置结果的掩码属性为 m
        result._mask = m
        # 返回带有掩码的结果
        return result
def min(obj, axis=None, out=None, fill_value=None, keepdims=np._NoValue):
    # 创建一个空字典，如果 keepdims 不等于 np._NoValue，则添加 'keepdims' 键
    kwargs = {} if keepdims is np._NoValue else {'keepdims': keepdims}

    try:
        # 尝试调用对象的 min 方法，使用给定的参数
        return obj.min(axis=axis, fill_value=fill_value, out=out, **kwargs)
    except (AttributeError, TypeError):
        # 如果对象没有 min 方法，或者方法不接受 fill_value 参数
        # 将对象转换为数组并调用其 min 方法，使用给定的参数
        return asanyarray(obj).min(axis=axis, fill_value=fill_value,
                                   out=out, **kwargs)
min.__doc__ = MaskedArray.min.__doc__

def max(obj, axis=None, out=None, fill_value=None, keepdims=np._NoValue):
    # 创建一个空字典，如果 keepdims 不等于 np._NoValue，则添加 'keepdims' 键
    kwargs = {} if keepdims is np._NoValue else {'keepdims': keepdims}

    try:
        # 尝试调用对象的 max 方法，使用给定的参数
        return obj.max(axis=axis, fill_value=fill_value, out=out, **kwargs)
    except (AttributeError, TypeError):
        # 如果对象没有 max 方法，或者方法不接受 fill_value 参数
        # 将对象转换为数组并调用其 max 方法，使用给定的参数
        return asanyarray(obj).max(axis=axis, fill_value=fill_value,
                                   out=out, **kwargs)
max.__doc__ = MaskedArray.max.__doc__


def ptp(obj, axis=None, out=None, fill_value=None, keepdims=np._NoValue):
    # 创建一个空字典，如果 keepdims 不等于 np._NoValue，则添加 'keepdims' 键
    kwargs = {} if keepdims is np._NoValue else {'keepdims': keepdims}
    try:
        # 尝试调用对象的 ptp 方法，使用给定的参数
        return obj.ptp(axis, out=out, fill_value=fill_value, **kwargs)
    except (AttributeError, TypeError):
        # 如果对象没有 ptp 方法，或者方法不接受 fill_value 参数
        # 将对象转换为数组并调用其 ptp 方法，使用给定的参数
        return asanyarray(obj).ptp(axis=axis, fill_value=fill_value,
                                   out=out, **kwargs)
ptp.__doc__ = MaskedArray.ptp.__doc__


##############################################################################
#           Definition of functions from the corresponding methods           #
##############################################################################

class _frommethod:
    """
    Define functions from existing MaskedArray methods.

    Parameters
    ----------
    methodname : str
        Name of the method to transform.

    """

    def __init__(self, methodname, reversed=False):
        # 将对象的名称设置为给定的方法名称
        self.__name__ = methodname
        # 设置对象的文档字符串为从对应方法获取的文档
        self.__doc__ = self.getdoc()
        # 设置是否反转标志
        self.reversed = reversed

    def getdoc(self):
        "Return the doc of the function (from the doc of the method)."
        # 获取 MaskedArray 类或 np 模块中的对应方法
        meth = getattr(MaskedArray, self.__name__, None) or\
            getattr(np, self.__name__, None)
        # 获取方法的签名信息
        signature = self.__name__ + get_object_signature(meth)
        if meth is not None:
            # 如果方法存在，构建函数的文档字符串，包括方法的签名和文档
            doc = """    %s\n%s""" % (
                signature, getattr(meth, '__doc__', None))
            return doc
    # 定义一个特殊方法 __call__，允许实例对象像函数一样被调用
    def __call__(self, a, *args, **params):
        # 如果设置了反转标志，调整参数顺序以匹配函数的预期顺序
        if self.reversed:
            args = list(args)
            a, args[0] = args[0], a
        
        # 将参数 a 转换为 NumPy 数组
        marr = asanyarray(a)
        
        # 获取当前对象的方法名
        method_name = self.__name__
        
        # 从 marr 对象的类型中获取方法对象，如果不存在则为 None
        method = getattr(type(marr), method_name, None)
        
        # 如果对象方法不存在，则从 NumPy 模块中获取对应的函数
        if method is None:
            method = getattr(np, method_name)
        
        # 调用获取到的方法或函数，传入 marr 和其他参数
        return method(marr, *args, **params)
# 创建名为 'all' 的函数对象，使用 _frommethod 函数获取相关方法
all = _frommethod('all')

# 创建名为 'anomalies' 和 'anom' 的函数对象，使用 _frommethod 函数获取相关方法
anomalies = anom = _frommethod('anom')

# 创建名为 'any' 的函数对象，使用 _frommethod 函数获取相关方法
any = _frommethod('any')

# 创建名为 'compress' 的函数对象，使用 _frommethod 函数获取相关方法，并反转参数
compress = _frommethod('compress', reversed=True)

# 创建名为 'cumprod' 的函数对象，使用 _frommethod 函数获取相关方法
cumprod = _frommethod('cumprod')

# 创建名为 'cumsum' 的函数对象，使用 _frommethod 函数获取相关方法
cumsum = _frommethod('cumsum')

# 创建名为 'copy' 的函数对象，使用 _frommethod 函数获取相关方法
copy = _frommethod('copy')

# 创建名为 'diagonal' 的函数对象，使用 _frommethod 函数获取相关方法
diagonal = _frommethod('diagonal')

# 创建名为 'harden_mask' 的函数对象，使用 _frommethod 函数获取相关方法
harden_mask = _frommethod('harden_mask')

# 创建名为 'ids' 的函数对象，使用 _frommethod 函数获取相关方法
ids = _frommethod('ids')

# 使用 _extrema_operation 函数调用 umath.maximum 方法和 greater 参数，创建名为 'maximum' 的函数对象
maximum = _extrema_operation(umath.maximum, greater, maximum_fill_value)

# 创建名为 'mean' 的函数对象，使用 _frommethod 函数获取相关方法
mean = _frommethod('mean')

# 使用 _extrema_operation 函数调用 umath.minimum 方法和 less 参数，创建名为 'minimum' 的函数对象
minimum = _extrema_operation(umath.minimum, less, minimum_fill_value)

# 创建名为 'nonzero' 的函数对象，使用 _frommethod 函数获取相关方法
nonzero = _frommethod('nonzero')

# 创建名为 'prod' 和 'product' 的函数对象，使用 _frommethod 函数获取相关方法
prod = _frommethod('prod')

# 创建名为 'ravel' 的函数对象，使用 _frommethod 函数获取相关方法
ravel = _frommethod('ravel')

# 创建名为 'repeat' 的函数对象，使用 _frommethod 函数获取相关方法
repeat = _frommethod('repeat')

# 创建名为 'shrink_mask' 的函数对象，使用 _frommethod 函数获取相关方法
shrink_mask = _frommethod('shrink_mask')

# 创建名为 'soften_mask' 的函数对象，使用 _frommethod 函数获取相关方法
soften_mask = _frommethod('soften_mask')

# 创建名为 'std' 的函数对象，使用 _frommethod 函数获取相关方法
std = _frommethod('std')

# 创建名为 'sum' 的函数对象，使用 _frommethod 函数获取相关方法
sum = _frommethod('sum')

# 创建名为 'swapaxes' 的函数对象，使用 _frommethod 函数获取相关方法
swapaxes = _frommethod('swapaxes')

# 创建名为 'trace' 的函数对象，使用 _frommethod 函数获取相关方法
trace = _frommethod('trace')

# 创建名为 'var' 的函数对象，使用 _frommethod 函数获取相关方法
var = _frommethod('var')

# 创建名为 'count' 的函数对象，使用 _frommethod 函数获取相关方法
count = _frommethod('count')

# 定义名为 'take' 的函数，接受参数 a, indices, axis=None, out=None, mode='raise'
def take(a, indices, axis=None, out=None, mode='raise'):
    """
    Returns element-wise base array raised to power from second array.

    This is the masked array version of `numpy.power`. For details see
    `numpy.power`.

    See Also
    --------
    numpy.power

    Notes
    -----
    The *out* argument to `numpy.power` is not supported, `third` has to be
    None.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> x = [11.2, -3.973, 0.801, -1.41]
    >>> mask = [0, 0, 0, 1]
    >>> masked_x = ma.masked_array(x, mask)
    >>> masked_x
    masked_array(data=[11.2, -3.973, 0.801, --],
             mask=[False, False, False,  True],
       fill_value=1e+20)
    >>> ma.power(masked_x, 2)
    masked_array(data=[125.43999999999998, 15.784728999999999,
                   0.6416010000000001, --],
             mask=[False, False, False,  True],
       fill_value=1e+20)
    >>> y = [-0.5, 2, 0, 17]
    >>> masked_y = ma.masked_array(y, mask)
    >>> masked_y
    masked_array(data=[-0.5, 2.0, 0.0, --],
             mask=[False, False, False,  True],
       fill_value=1e+20)
    >>> ma.power(masked_x, masked_y)
    masked_array(data=[0.2988071523335984, 15.784728999999999, 1.0, --],
             mask=[False, False, False,  True],
       fill_value=1e+20)

    """
    # 如果 third 不为 None，抛出 MaskError 异常
    if third is not None:
        raise MaskError("3-argument power not supported.")
    # 获取参数 a 和 b 的掩码
    ma = getmask(a)
    mb = getmask(b)
    # 合并掩码
    m = mask_or(ma, mb)
    # 获取参数 a 和 b 的数据
    fa = getdata(a)
    fb = getdata(b)
    # 确定结果的类型（保持子类）
    if isinstance(a, MaskedArray):
        basetype = type(a)
    else:
        basetype = MaskedArray
    # 计算结果并视为 MaskedArray 或其子类
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(m, fa, umath.power(fa, fb)).view(basetype)
    # 更新结果对象
    result._update_from(a)
    # 找出结果中包含 NaN 或 Inf 的无效数据位置
    invalid = np.logical_not(np.isfinite(result.view(ndarray)))
    
    # 添加初始掩码
    if m is not nomask:
        # 如果 m 不是 nomask，则处理结果中的掩码
        if not result.ndim:
            # 如果结果的维度为零，则返回掩码
            return masked
        # 将 m 和 invalid 合并作为结果的掩码
        result._mask = np.logical_or(m, invalid)
    
    # 修复无效部分的数据
    if invalid.any():
        # 如果存在无效数据
        if not result.ndim:
            # 如果结果的维度为零，则返回掩码
            return masked
        elif result._mask is nomask:
            # 如果结果的掩码是 nomask，则将 invalid 设置为结果的掩码
            result._mask = invalid
        # 将无效数据位置的值填充为结果的填充值
        result._data[invalid] = result.fill_value
    
    # 返回修正后的结果
    return result
argmin = _frommethod('argmin')  # 从方法名'argmin'创建一个函数对象argmin
argmax = _frommethod('argmax')  # 从方法名'argmax'创建一个函数对象argmax

def argsort(a, axis=np._NoValue, kind=None, order=None, endwith=True,
            fill_value=None, *, stable=None):
    "Function version of the eponymous method."
    a = np.asanyarray(a)  # 将输入参数a转换为numpy数组

    # 2017-04-11, Numpy 1.13.0, gh-8701: warn on axis default
    if axis is np._NoValue:
        axis = _deprecate_argsort_axis(a)  # 如果未指定axis，通过_deprecate_argsort_axis函数设置默认值

    if isinstance(a, MaskedArray):
        return a.argsort(axis=axis, kind=kind, order=order, endwith=endwith,
                         fill_value=fill_value, stable=None)  # 对MaskedArray对象进行排序操作
    else:
        return a.argsort(axis=axis, kind=kind, order=order, stable=None)  # 对普通数组进行排序操作
argsort.__doc__ = MaskedArray.argsort.__doc__  # 将argsort函数的文档字符串设置为MaskedArray.argsort方法的文档字符串

def sort(a, axis=-1, kind=None, order=None, endwith=True, fill_value=None, *,
         stable=None):
    """
    Return a sorted copy of the masked array.

    Equivalent to creating a copy of the array
    and applying the  MaskedArray ``sort()`` method.

    Refer to ``MaskedArray.sort`` for the full documentation

    See Also
    --------
    MaskedArray.sort : equivalent method

    Examples
    --------
    >>> import numpy.ma as ma
    >>> x = [11.2, -3.973, 0.801, -1.41]
    >>> mask = [0, 0, 0, 1]
    >>> masked_x = ma.masked_array(x, mask)
    >>> masked_x
    masked_array(data=[11.2, -3.973, 0.801, --],
                 mask=[False, False, False,  True],
           fill_value=1e+20)
    >>> ma.sort(masked_x)
    masked_array(data=[-3.973, 0.801, 11.2, --],
                 mask=[False, False, False,  True],
           fill_value=1e+20)
    """
    a = np.array(a, copy=True, subok=True)  # 将输入a转换为numpy数组，并确保复制和子类保留
    if axis is None:
        a = a.flatten()  # 如果axis为None，则将数组扁平化
        axis = 0

    if isinstance(a, MaskedArray):
        a.sort(axis=axis, kind=kind, order=order, endwith=endwith,
               fill_value=fill_value, stable=stable)  # 对MaskedArray对象进行排序操作
    else:
        a.sort(axis=axis, kind=kind, order=order, stable=stable)  # 对普通数组进行排序操作
    return a


def compressed(x):
    """
    Return all the non-masked data as a 1-D array.

    This function is equivalent to calling the "compressed" method of a
    `ma.MaskedArray`, see `ma.MaskedArray.compressed` for details.

    See Also
    --------
    ma.MaskedArray.compressed : Equivalent method.

    Examples
    --------

    Create an array with negative values masked:

    >>> import numpy as np
    >>> x = np.array([[1, -1, 0], [2, -1, 3], [7, 4, -1]])
    >>> masked_x = np.ma.masked_array(x, mask=x < 0)
    >>> masked_x
    masked_array(
      data=[[1, --, 0],
            [2, --, 3],
            [7, 4, --]],
      mask=[[False,  True, False],
            [False,  True, False],
            [False, False,  True]],
      fill_value=999999)

    Compress the masked array into a 1-D array of non-masked values:

    >>> np.ma.compressed(masked_x)
    array([1, 0, 2, 3, 7, 4])

    """
    return asanyarray(x).compressed()  # 返回输入数组x中未被屏蔽的数据的压缩版本的1-D数组


def concatenate(arrays, axis=0):
    """
    Concatenate a sequence of arrays along the given axis.

    Parameters
    ----------
    arrays : sequence of array_like
        The arrays must have the same shape along all but the concatenation axis.
    # 使用给定的序列数组进行连接操作，要求除了指定轴（默认是第一个轴）外，其它维度形状必须一致。
    arrays : sequence of array_like
        The arrays must have the same shape, except in the dimension
        corresponding to `axis` (the first, by default).
    # 沿着指定的轴进行连接操作，默认为第0轴。
    axis : int, optional
        The axis along which the arrays will be joined. Default is 0.

    Returns
    -------
    # 返回一个 MaskedArray 对象，其中保留了任何屏蔽（masked）的条目。
    result : MaskedArray
        The concatenated array with any masked entries preserved.

    See Also
    --------
    # 在顶层 NumPy 模块中的等效函数 numpy.concatenate。
    numpy.concatenate : Equivalent function in the top-level NumPy module.

    Examples
    --------
    # 示例：导入 numpy.ma 模块，创建带有屏蔽值的数组，并进行连接操作。
    >>> import numpy.ma as ma
    >>> a = ma.arange(3)
    >>> a[1] = ma.masked
    >>> b = ma.arange(2, 5)
    >>> a
    masked_array(data=[0, --, 2],
                 mask=[False,  True, False],
           fill_value=999999)
    >>> b
    masked_array(data=[2, 3, 4],
                 mask=False,
           fill_value=999999)
    >>> ma.concatenate([a, b])
    masked_array(data=[0, --, 2, 2, 3, 4],
                 mask=[False,  True, False, False, False, False],
           fill_value=999999)

    """
    # 对给定的 arrays 序列中的每个数组调用 getdata 函数，并沿指定轴进行连接，形成一个新的数组 d。
    d = np.concatenate([getdata(a) for a in arrays], axis)
    # 获取 arrays 序列中数组的子类，并将连接后的数据 d 视图转换为该子类的实例。
    rcls = get_masked_subclass(*arrays)
    data = d.view(rcls)
    # 检查 arrays 序列中是否有数组带有非空屏蔽。
    for x in arrays:
        if getmask(x) is not nomask:
            break
    else:
        # 若所有数组均无屏蔽，则直接返回数据。
        return data
    # 若存在屏蔽，则连接所有数组的屏蔽，并按照指定轴重新整形为与数据 d 相同的形状。
    dm = np.concatenate([getmaskarray(a) for a in arrays], axis)
    dm = dm.reshape(d.shape)

    # 如果决定保留 '_shrinkmask' 选项，需要检查所有屏蔽是否都为 True，并检查 dm.any()。
    data._mask = _shrink_mask(dm)
    return data
# 从输入参数中提取对角线或构建对角线数组的函数
def diag(v, k=0):
    # 创建一个视图对象，用于返回对角线数组，支持掩码数组
    output = np.diag(v, k).view(MaskedArray)
    # 如果输入数组有掩码，则设置对应的掩码数组
    if getmask(v) is not nomask:
        output._mask = np.diag(v._mask, k)
    return output


# 对整数进行左移位操作的函数
def left_shift(a, n):
    # 获取输入数组的掩码
    m = getmask(a)
    # 如果没有掩码，则直接对数组执行左移位操作
    if m is nomask:
        d = umath.left_shift(filled(a), n)
        return masked_array(d)
    else:
        # 如果有掩码，则先填充数组后再执行左移位操作，并将掩码传递给结果数组
        d = umath.left_shift(filled(a, 0), n)
        return masked_array(d, mask=m)


# 对整数进行右移位操作的函数
def right_shift(a, n):
    # 获取输入数组的掩码
    m = getmask(a)
    # 如果没有掩码，则直接对数组执行右移位操作
    if m is nomask:
        d = umath.right_shift(filled(a), n)
        return masked_array(d)
    else:
        # 如果有掩码，则先填充数组后再执行右移位操作，并将掩码传递给结果数组
        d = umath.right_shift(filled(a, 0), n)
        return masked_array(d, mask=m)


# 将指定位置的值设置为相应值的函数
def put(a, indices, values, mode='raise'):
    # 由于参数顺序不同，无法使用 'frommethod'，因此直接略过注释
    # 尝试调用对象 a 的 put 方法，传入 indices、values 和 mode 参数
    try:
        return a.put(indices, values, mode=mode)
    # 如果对象 a 没有 put 方法，则转换为 NumPy 数组后再调用 put 方法
    except AttributeError:
        return np.asarray(a).put(indices, values, mode=mode)
# 改变数组中的元素根据条件和输入值。
def putmask(a, mask, values):  # , mode='raise'):
    """
    Changes elements of an array based on conditional and input values.

    This is the masked array version of `numpy.putmask`, for details see
    `numpy.putmask`.

    See Also
    --------
    numpy.putmask

    Notes
    -----
    Using a masked array as `values` will **not** transform a `ndarray` into
    a `MaskedArray`.

    Examples
    --------
    >>> arr = [[1, 2], [3, 4]]
    >>> mask = [[1, 0], [0, 0]]
    >>> x = np.ma.array(arr, mask=mask)
    >>> np.ma.putmask(x, x < 4, 10*x)
    >>> x
    masked_array(
      data=[[--, 20],
            [30, 4]],
      mask=[[ True, False],
            [False, False]],
      fill_value=999999)
    >>> x.data
    array([[10, 20],
           [30,  4]])

    """
    # 如果 `a` 不是 MaskedArray 类型，则视图化为 MaskedArray 类型
    if not isinstance(a, MaskedArray):
        a = a.view(MaskedArray)
    # 获取 `values` 的数据和掩码
    (valdata, valmask) = (getdata(values), getmask(values))
    # 如果 `a` 的掩码为 nomask
    if getmask(a) is nomask:
        # 如果 `values` 的掩码不为 nomask，则将其复制到 `a` 的掩码中
        if valmask is not nomask:
            a._sharedmask = True
            a._mask = make_mask_none(a.shape, a.dtype)
            np.copyto(a._mask, valmask, where=mask)
    # 如果 `a` 使用硬掩码
    elif a._hardmask:
        # 如果 `values` 的掩码不为 nomask，则创建临时掩码并更新 `a` 的掩码
        if valmask is not nomask:
            m = a._mask.copy()
            np.copyto(m, valmask, where=mask)
            a.mask |= m
    else:
        # 如果 `values` 的掩码为 nomask，则创建其掩码数组
        if valmask is nomask:
            valmask = getmaskarray(values)
        # 将 `values` 的掩码复制到 `a` 的掩码中
        np.copyto(a._mask, valmask, where=mask)
    # 将 `values` 的数据复制到 `a` 的数据中
    np.copyto(a._data, valdata, where=mask)
    return


def transpose(a, axes=None):
    """
    Permute the dimensions of an array.

    This function is exactly equivalent to `numpy.transpose`.

    See Also
    --------
    numpy.transpose : Equivalent function in top-level NumPy module.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> x = ma.arange(4).reshape((2,2))
    >>> x[1, 1] = ma.masked
    >>> x
    masked_array(
      data=[[0, 1],
            [2, --]],
      mask=[[False, False],
            [False,  True]],
      fill_value=999999)

    >>> ma.transpose(x)
    masked_array(
      data=[[0, 2],
            [1, --]],
      mask=[[False, False],
            [False,  True]],
      fill_value=999999)
    """
    # 尝试使用 `a` 的 `transpose` 方法，并传递轴参数
    try:
        return a.transpose(axes)
    # 如果 `a` 没有 `transpose` 方法，则转换为数组后再进行转置，并视图化为 MaskedArray 类型
    except AttributeError:
        return np.asarray(a).transpose(axes).view(MaskedArray)


def reshape(a, new_shape, order='C'):
    """
    Returns an array containing the same data with a new shape.

    Refer to `MaskedArray.reshape` for full documentation.

    See Also
    --------
    MaskedArray.reshape : equivalent function

    """
    # 尝试使用 `a` 的 `reshape` 方法，并传递新形状和顺序参数
    try:
        return a.reshape(new_shape, order=order)
    # 如果 `a` 没有 `reshape` 方法，则转换为数组后再进行重塑，并视图化为 MaskedArray 类型
    except AttributeError:
        _tmp = np.asarray(a).reshape(new_shape, order=order)
        return _tmp.view(MaskedArray)


def resize(x, new_shape):
    """
    Change shape and size of array in-place.

    Returns a view to `x` with updated shape and size.

    Parameters
    ----------
    new_shape : tuple
        The new shape of the array.

    """
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    # 获取输入数组 `x` 的掩码
    m = getmask(x)
    # 如果掩码不是 nomask，则调整掩码 `m` 的大小为新形状 `new_shape`
    if m is not nomask:
        m = np.resize(m, new_shape)
    # 调用 `numpy.resize` 来调整数组 `x` 的大小，并返回一个视图，其类型是输入 `x` 的掩码子类
    result = np.resize(x, new_shape).view(get_masked_subclass(x))
    # 如果结果数组的维度不为零，则将其掩码设置为 `m`
    if result.ndim:
        result._mask = m
    # 返回调整大小后的掩码数组 `result`
    return result
def ndim(obj):
    """
    maskedarray version of the numpy function.

    """
    # 返回输入对象的维数
    return np.ndim(getdata(obj))

# 将ndim函数的文档字符串设置为np.ndim函数的文档字符串
ndim.__doc__ = np.ndim.__doc__


def shape(obj):
    "maskedarray version of the numpy function."
    # 返回输入对象的形状
    return np.shape(getdata(obj))

# 将shape函数的文档字符串设置为np.shape函数的文档字符串
shape.__doc__ = np.shape.__doc__


def size(obj, axis=None):
    "maskedarray version of the numpy function."
    # 返回输入对象的元素数量
    return np.size(getdata(obj), axis)

# 将size函数的文档字符串设置为np.size函数的文档字符串
size.__doc__ = np.size.__doc__


def diff(a, /, n=1, axis=-1, prepend=np._NoValue, append=np._NoValue):
    """
    Calculate the n-th discrete difference along the given axis.
    The first difference is given by ``out[i] = a[i+1] - a[i]`` along
    the given axis, higher differences are calculated by using `diff`
    recursively.
    Preserves the input mask.

    Parameters
    ----------
    a : array_like
        Input array
    n : int, optional
        The number of times values are differenced. If zero, the input
        is returned as-is.
    axis : int, optional
        The axis along which the difference is taken, default is the
        last axis.
    prepend, append : array_like, optional
        Values to prepend or append to `a` along axis prior to
        performing the difference.  Scalar values are expanded to
        arrays with length 1 in the direction of axis and the shape
        of the input array in along all other axes.  Otherwise the
        dimension and shape must match `a` except along axis.

    Returns
    -------
    diff : MaskedArray
        The n-th differences. The shape of the output is the same as `a`
        except along `axis` where the dimension is smaller by `n`. The
        type of the output is the same as the type of the difference
        between any two elements of `a`. This is the same as the type of
        `a` in most cases. A notable exception is `datetime64`, which
        results in a `timedelta64` output array.

    See Also
    --------
    numpy.diff : Equivalent function in the top-level NumPy module.

    Notes
    -----
    Type is preserved for boolean arrays, so the result will contain
    `False` when consecutive elements are the same and `True` when they
    differ.

    For unsigned integer arrays, the results will also be unsigned. This
    should not be surprising, as the result is consistent with
    calculating the difference directly:

    >>> u8_arr = np.array([1, 0], dtype=np.uint8)
    >>> np.ma.diff(u8_arr)
    masked_array(data=[255],
                 mask=False,
           fill_value=np.int64(999999),
                dtype=uint8)
    >>> u8_arr[1,...] - u8_arr[0,...]
    255

    If this is not desirable, then the array should be cast to a larger
    integer type first:

    >>> i16_arr = u8_arr.astype(np.int16)
    >>> np.ma.diff(i16_arr)
    masked_array(data=[-1],
                 mask=False,
           fill_value=np.int64(999999),
                dtype=int16)

    Examples
    --------
    >>> a = np.array([1, 2, 3, 4, 7, 0, 2, 3])
    >>> x = np.ma.masked_where(a < 2, a)
    """
    # 计算沿指定轴向的n阶离散差分
    return np.ma.diff(a, n=n, axis=axis, prepend=prepend, append=append)

# 将diff函数的文档字符串设置为np.ma.diff函数的文档字符串
diff.__doc__ = np.ma.diff.__doc__
    """
    Calculate the n-th discrete difference along the given axis.

    Parameters:
    - a : array_like
        Input array.
    - n : int, optional
        The number of times values are differenced. Default is 1.
    - axis : int, optional
        The axis along which the difference is taken, default is -1 (the last axis).
    - prepend, append : array_like, optional
        Values to prepend or append to `a` along axis prior to taking the difference.

    Returns:
    - diff : ndarray
        The n-th differences. The shape of the output is the same as `a` except along `axis` where the dimension is smaller by `n`.

    Notes:
    - If `n == 0`, the input array `a` is returned as-is.
    - Raises a ValueError if `n` is negative.
    - Uses masked arrays (numpy.ma) to handle masked input arrays and preserves masks in output.
    """

    if n == 0:
        # If n is 0, return the input array as it is
        return a

    if n < 0:
        # Raise an error if n is negative
        raise ValueError("order must be non-negative but got " + repr(n))

    # Convert input array to masked array to handle masked values
    a = np.ma.asanyarray(a)

    if a.ndim == 0:
        # Raise an error if the input array is scalar (0-dimensional)
        raise ValueError("diff requires input that is at least one dimensional")

    combined = []

    # Handle prepend parameter
    if prepend is not np._NoValue:
        prepend = np.ma.asanyarray(prepend)
        if prepend.ndim == 0:
            shape = list(a.shape)
            shape[axis] = 1
            prepend = np.broadcast_to(prepend, tuple(shape))
        combined.append(prepend)

    combined.append(a)

    # Handle append parameter
    if append is not np._NoValue:
        append = np.ma.asanyarray(append)
        if append.ndim == 0:
            shape = list(a.shape)
            shape[axis] = 1
            append = np.broadcast_to(append, tuple(shape))
        combined.append(append)

    if len(combined) > 1:
        # Concatenate arrays in combined along the specified axis
        a = np.ma.concatenate(combined, axis)

    # Compute the n-th differences using numpy's diff function
    return np.diff(a, n, axis)
# Return a masked array based on the condition, choosing elements from `x` or `y`.
def where(condition, x=_NoValue, y=_NoValue):
    """
    Return a masked array with elements from `x` or `y`, depending on condition.

    .. note::
        When only `condition` is provided, this function is identical to
        `nonzero`. The rest of this documentation covers only the case where
        all three arguments are provided.

    Parameters
    ----------
    condition : array_like, bool
        Where True, yield `x`, otherwise yield `y`.
    x, y : array_like, optional
        Values from which to choose. `x`, `y` and `condition` need to be
        broadcastable to some shape.

    Returns
    -------
    out : MaskedArray
        An masked array with `masked` elements where the condition is masked,
        elements from `x` where `condition` is True, and elements from `y`
        elsewhere.

    See Also
    --------
    numpy.where : Equivalent function in the top-level NumPy module.
    nonzero : The function that is called when x and y are omitted

    Examples
    --------
    >>> x = np.ma.array(np.arange(9.).reshape(3, 3), mask=[[0, 1, 0],
    ...                                                    [1, 0, 1],
    ...                                                    [0, 1, 0]])
    >>> x
    masked_array(
      data=[[0.0, --, 2.0],
            [--, 4.0, --],
            [6.0, --, 8.0]],
      mask=[[False,  True, False],
            [ True, False,  True],
            [False,  True, False]],
      fill_value=1e+20)
    >>> np.ma.where(x > 5, x, -3.1416)
    masked_array(
      data=[[-3.1416, --, -3.1416],
            [--, -3.1416, --],
            [6.0, --, 8.0]],
      mask=[[False,  True, False],
            [ True, False,  True],
            [False,  True, False]],
      fill_value=1e+20)

    """

    # handle the single-argument case
    # Count how many arguments are missing (either x or y)
    missing = (x is _NoValue, y is _NoValue).count(True)
    # If exactly one argument is missing, raise an error
    if missing == 1:
        raise ValueError("Must provide both 'x' and 'y' or neither.")
    # If both or neither arguments are missing, use nonzero function instead
    if missing == 2:
        return nonzero(condition)

    # Convert condition to filled array, defaulting to False for masked values
    cf = filled(condition, False)
    # Get data arrays for x and y
    xd = getdata(x)
    yd = getdata(y)

    # Get mask arrays for condition, x, and y
    cm = getmaskarray(condition)
    xm = getmaskarray(x)
    ym = getmaskarray(y)

    # Handle cases where one of x or y is masked but not both
    if x is masked and y is not masked:
        xd = np.zeros((), dtype=yd.dtype)
        xm = np.ones((), dtype=ym.dtype)
    elif y is masked and x is not masked:
        yd = np.zeros((), dtype=xd.dtype)
        ym = np.ones((), dtype=xm.dtype)

    # Create final masked array based on condition, x, and y
    data = np.where(cf, xd, yd)
    mask = np.where(cf, xm, ym)
    # 使用 np.where 函数根据条件数组 cm 来选择是使用空的数组还是 mask 数组本身的元素，生成新的 mask
    mask = np.where(cm, np.ones((), dtype=mask.dtype), mask)

    # 调用 _shrink_mask 函数，对 mask 进行处理以兼容旧版本，可能是减少 mask 的尺寸或者进行其他处理
    mask = _shrink_mask(mask)

    # 创建并返回一个带有数据和掩码的掩码数组
    return masked_array(data, mask=mask)
# 返回一个数组，通过使用索引数组从选择数组中构造新数组
def choose(indices, choices, out=None, mode='raise'):
    """
    Use an index array to construct a new array from a list of choices.

    Given an array of integers and a list of n choice arrays, this method
    will create a new array that merges each of the choice arrays.  Where a
    value in `index` is i, the new array will have the value that choices[i]
    contains in the same place.

    Parameters
    ----------
    indices : ndarray of ints
        This array must contain integers in ``[0, n-1]``, where n is the
        number of choices.
    choices : sequence of arrays
        Choice arrays. The index array and all of the choices should be
        broadcastable to the same shape.
    out : array, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and `dtype`.
    mode : {'raise', 'wrap', 'clip'}, optional
        Specifies how out-of-bounds indices will behave.

        * 'raise' : raise an error
        * 'wrap' : wrap around
        * 'clip' : clip to the range

    Returns
    -------
    merged_array : array

    See Also
    --------
    choose : equivalent function

    Examples
    --------
    >>> choice = np.array([[1,1,1], [2,2,2], [3,3,3]])
    >>> a = np.array([2, 1, 0])
    >>> np.ma.choose(a, choice)
    masked_array(data=[3, 2, 1],
                 mask=False,
           fill_value=999999)

    """
    # 定义一个函数，根据输入返回填充的数组或者 True（如果是 masked）
    def fmask(x):
        "Returns the filled array, or True if masked."
        if x is masked:
            return True
        return filled(x)

    # 定义一个函数，根据输入返回掩码，True（如果是 masked）或者 False（如果是 nomask）
    def nmask(x):
        "Returns the mask, True if ``masked``, False if ``nomask``."
        if x is masked:
            return True
        return getmask(x)

    # 获取索引数组的填充值
    c = filled(indices, 0)
    # 获取选择数组的掩码列表
    masks = [nmask(x) for x in choices]
    # 获取选择数组的填充值列表
    data = [fmask(x) for x in choices]
    # 构建输出的掩码
    outputmask = np.choose(c, masks, mode=mode)
    outputmask = make_mask(mask_or(outputmask, getmask(indices)),
                           copy=False, shrink=True)
    # 根据索引数组获取选择数组的值
    d = np.choose(c, data, mode=mode, out=out).view(MaskedArray)
    if out is not None:
        # 如果提供了输出数组，并且它是 MaskedArray 类型，则设置掩码
        if isinstance(out, MaskedArray):
            out.__setmask__(outputmask)
        return out
    # 否则设置 MaskedArray 的掩码
    d.__setmask__(outputmask)
    return d


def round_(a, decimals=0, out=None):
    """
    Return a copy of a, rounded to 'decimals' places.

    When 'decimals' is negative, it specifies the number of positions
    to the left of the decimal point.  The real and imaginary parts of
    complex numbers are rounded separately. Nothing is done if the
    array is not of float type and 'decimals' is greater than or equal
    to 0.

    Parameters
    ----------
    decimals : int
        Number of decimals to round to. May be negative.
    out : array_like
        Existing array to use for output.
        If not given, returns a default copy of a.

    Notes
    -----
    """
    # 如果未提供输出对象 out，则使用 numpy 的 round 函数对数组 a 进行四舍五入操作，并将结果返回
    if out is None:
        return np.round(a, decimals, out)
    else:
        # 如果提供了输出对象 out，则使用 numpy 的 round 函数对数组 a 进行四舍五入操作，并将结果写入 out
        np.round(getdata(a), decimals, out)
        # 如果输出对象 out 具有 '_mask' 属性，则将其掩码设置为数组 a 的掩码
        if hasattr(out, '_mask'):
            out._mask = getmask(a)
        # 返回输出对象 out
        return out
# 将内置函数 round 赋值给 round_，以便稍后使用
round = round_

# 定义一个内部函数，用于在数组中屏蔽包含屏蔽值的整个一维向量
def _mask_propagate(a, axis):
    """
    Mask whole 1-d vectors of an array that contain masked values.
    """
    # 将 a 转换为数组（如果它不是数组的话）
    a = array(a, subok=False)
    # 获取数组 a 的屏蔽（mask）
    m = getmask(a)
    # 如果没有屏蔽或者屏蔽数组是空的，或者 axis 为 None，则直接返回数组 a
    if m is nomask or not m.any() or axis is None:
        return a
    # 复制 a 的屏蔽，以便进行修改
    a._mask = a._mask.copy()
    # 规范化轴元组，确保轴在合适的范围内
    axes = normalize_axis_tuple(axis, a.ndim)
    # 对每个轴进行操作
    for ax in axes:
        # 将 a 的屏蔽与沿轴 ax 的任何屏蔽合并，并保持维度
        a._mask |= m.any(axis=ax, keepdims=True)
    # 返回处理后的数组 a
    return a


# 在此处包含 masked dot，以避免从 extras.py 中导入时的问题。注意它不包含在 __all__ 中，
# 而是从 extras 中导出，以避免向后兼容性问题。
def dot(a, b, strict=False, out=None):
    """
    Return the dot product of two arrays.

    This function is the equivalent of `numpy.dot` that takes masked values
    into account. Note that `strict` and `out` are in different position
    than in the method version. In order to maintain compatibility with the
    corresponding method, it is recommended that the optional arguments be
    treated as keyword only.  At some point that may be mandatory.

    Parameters
    ----------
    a, b : masked_array_like
        Inputs arrays.
    strict : bool, optional
        Whether masked data are propagated (True) or set to 0 (False) for
        the computation. Default is False.  Propagating the mask means that
        if a masked value appears in a row or column, the whole row or
        column is considered masked.
    out : masked_array, optional
        Output argument. This must have the exact kind that would be returned
        if it was not used. In particular, it must have the right type, must be
        C-contiguous, and its dtype must be the dtype that would be returned
        for `dot(a,b)`. This is a performance feature. Therefore, if these
        conditions are not met, an exception is raised, instead of attempting
        to be flexible.

        .. versionadded:: 1.10.2

    See Also
    --------
    numpy.dot : Equivalent function for ndarrays.

    Examples
    --------
    >>> a = np.ma.array([[1, 2, 3], [4, 5, 6]], mask=[[1, 0, 0], [0, 0, 0]])
    >>> b = np.ma.array([[1, 2], [3, 4], [5, 6]], mask=[[1, 0], [0, 0], [0, 0]])
    >>> np.ma.dot(a, b)
    masked_array(
      data=[[21, 26],
            [45, 64]],
      mask=[[False, False],
            [False, False]],
      fill_value=999999)
    >>> np.ma.dot(a, b, strict=True)
    masked_array(
      data=[[--, --],
            [--, 64]],
      mask=[[ True,  True],
            [ True, False]],
      fill_value=999999)

    """
    # 如果 strict 为 True，则处理屏蔽值传播
    if strict is True:
        # 如果 a 或 b 的维度为 0，则不做任何操作
        if np.ndim(a) == 0 or np.ndim(b) == 0:
            pass
        # 如果 b 的维度为 1，则分别对 a 和 b 进行屏蔽传播
        elif b.ndim == 1:
            a = _mask_propagate(a, a.ndim - 1)
            b = _mask_propagate(b, b.ndim - 1)
        # 否则对 a 和 b 的最后两个维度进行屏蔽传播
        else:
            a = _mask_propagate(a, a.ndim - 1)
            b = _mask_propagate(b, b.ndim - 2)
    # 获取 a 和 b 的非屏蔽数组
    am = ~getmaskarray(a)
    bm = ~getmaskarray(b)
    # 如果输出对象 out 为 None，则执行以下操作
    if out is None:
        # 计算填充后的数组 a 和 b 的点积
        d = np.dot(filled(a, 0), filled(b, 0))
        # 计算 a 和 b 的掩码的逻辑非
        m = ~np.dot(am, bm)
        # 如果 d 的维度为标量，则转换为 NumPy 数组
        if np.ndim(d) == 0:
            d = np.asarray(d)
        # 创建一个基于 a 和 b 类型的掩码子类视图 r
        r = d.view(get_masked_subclass(a, b))
        # 将 m 应用为 r 的掩码
        r.__setmask__(m)
        # 返回结果 r
        return r
    # 如果输出对象 out 不为 None，则执行以下操作
    else:
        # 将填充后的数组 a 和 b 的点积存储到 out._data 中
        d = np.dot(filled(a, 0), filled(b, 0), out._data)
        # 如果 out 的掩码形状与 d 的形状不同，则创建一个新的掩码数组
        if out.mask.shape != d.shape:
            out._mask = np.empty(d.shape, MaskType)
        # 计算 a 和 b 的掩码并存储到 out._mask 中
        np.dot(am, bm, out._mask)
        # 对 out._mask 执行逻辑非操作
        np.logical_not(out._mask, out._mask)
        # 返回输出对象 out
        return out
# 返回两个浮点类型数组 a 和 b 的内积（点积）

def inner(a, b):
    """
    Returns the inner product of a and b for arrays of floating point types.

    Like the generic NumPy equivalent the product sum is over the last dimension
    of a and b. The first argument is not conjugated.

    """
    # 将数组 a 和 b 中的缺失值填充为 0，返回填充后的数组 fa 和 fb
    fa = filled(a, 0)
    fb = filled(b, 0)
    # 如果 fa 是零维数组，则将其形状改为 (1,)
    if fa.ndim == 0:
        fa.shape = (1,)
    # 如果 fb 是零维数组，则将其形状改为 (1,)
    if fb.ndim == 0:
        fb.shape = (1,)
    # 计算 fa 和 fb 的内积，并将结果视为 MaskedArray 类型返回
    return np.inner(fa, fb).view(MaskedArray)

# 将 inner 函数的文档字符串设置为 np.inner 函数文档字符串的注释版本，说明 "Masked" 值被替换为 0
inner.__doc__ = doc_note(np.inner.__doc__,
                         "Masked values are replaced by 0.")

# 将 inner 函数命名为 innerproduct，作为 inner 函数的别名
innerproduct = inner


# 返回两个一维序列 a 和 b 的外积（叉积），并视情况屏蔽掉部分元素
def outer(a, b):
    "maskedarray version of the numpy function."
    # 将数组 a 和 b 中的缺失值填充为 0，并将它们展平为一维数组，分别得到 fa 和 fb
    fa = filled(a, 0).ravel()
    fb = filled(b, 0).ravel()
    # 计算 fa 和 fb 的外积，并存储在 d 中
    d = np.outer(fa, fb)
    # 获取 a 和 b 的掩码信息
    ma = getmask(a)
    mb = getmask(b)
    # 如果 a 和 b 都没有掩码，则直接返回带掩码的数组 d
    if ma is nomask and mb is nomask:
        return masked_array(d)
    # 否则，获取 a 和 b 的掩码数组，计算新的掩码 m，并返回带掩码的数组 d
    ma = getmaskarray(a)
    mb = getmaskarray(b)
    m = make_mask(1 - np.outer(1 - ma, 1 - mb), copy=False)
    return masked_array(d, mask=m)

# 将 outer 函数的文档字符串设置为 np.outer 函数文档字符串的注释版本，说明 "Masked" 值被替换为 0
outer.__doc__ = doc_note(np.outer.__doc__,
                         "Masked values are replaced by 0.")

# 将 outer 函数命名为 outerproduct，作为 outer 函数的别名
outerproduct = outer


# 用于 ma.correlate 和 ma.convolve 的辅助函数
def _convolve_or_correlate(f, a, v, mode, propagate_mask):
    """
    Helper function for ma.correlate and ma.convolve
    """
    if propagate_mask:
        # 计算带有掩码的结果，掩码表示由任意一对元素中的无效元素贡献的结果
        mask = (
            f(getmaskarray(a), np.ones(np.shape(v), dtype=bool), mode=mode)
          | f(np.ones(np.shape(a), dtype=bool), getmaskarray(v), mode=mode)
        )
        # 计算数据部分，忽略掩码的影响，得到数据 data
        data = f(getdata(a), getdata(v), mode=mode)
    else:
        # 计算不带掩码的结果，掩码表示没有任何有效元素对贡献的结果
        mask = ~f(~getmaskarray(a), ~getmaskarray(v), mode=mode)
        # 将 a 和 v 中的缺失值填充为 0，并计算数据部分，得到数据 data
        data = f(filled(a, 0), filled(v, 0), mode=mode)

    # 返回带有掩码的数组，数据部分为 data，掩码部分为 mask
    return masked_array(data, mask=mask)


# 返回两个一维序列 a 和 v 的交叉相关（correlation）
def correlate(a, v, mode='valid', propagate_mask=True):
    """
    Cross-correlation of two 1-dimensional sequences.

    Parameters
    ----------
    a, v : array_like
        Input sequences.
    mode : {'valid', 'same', 'full'}, optional
        Refer to the `np.convolve` docstring.  Note that the default
        is 'valid', unlike `convolve`, which uses 'full'.
    propagate_mask : bool
        If True, then a result element is masked if any masked element contributes towards it.
        If False, then a result element is only masked if no non-masked element
        contribute towards it

    Returns
    -------
    out : MaskedArray
        Discrete cross-correlation of `a` and `v`.

    See Also
    --------
    numpy.correlate : Equivalent function in the top-level NumPy module.
    """
    # 调用 _convolve_or_correlate 函数，计算带有掩码的交叉相关，并返回结果
    return _convolve_or_correlate(np.correlate, a, v, mode, propagate_mask)


# 返回两个一维序列 a 和 v 的离散线性卷积（convolution）
def convolve(a, v, mode='full', propagate_mask=True):
    """
    Returns the discrete, linear convolution of two one-dimensional sequences.

    Parameters
    ----------
    a, v : array_like
        Input sequences.
    mode : {'full', 'valid', 'same'}, optional
        Refer to the `np.convolve` docstring. Note that the default is 'full'.
    propagate_mask : bool
        If True, then a result element is masked if any masked element contributes towards it.
        If False, then a result element is only masked if no non-masked element
        contribute towards it

    Returns
    -------
    out : MaskedArray
        Discrete linear convolution of `a` and `v`.
    """
    # 调用 _convolve_or_correlate 函数，计算带有掩码的离散线性卷积，并返回结果
    return _convolve_or_correlate(np.convolve, a, v, mode, propagate_mask)
    mode : {'valid', 'same', 'full'}, optional
        # 模式参数，用于指定卷积操作的方式，可选值为'valid', 'same', 'full'，具体含义可参考np.convolve文档字符串。
    propagate_mask : bool
        # 是否传播掩码的布尔值。如果为True，则如果任何掩码元素包含在某个结果元素的求和中，则结果将被掩码。
        # 如果为False，则仅当没有非掩码单元对其进行贡献时，结果元素才会被掩码。

    Returns
    -------
    out : MaskedArray
        # 返回掩码数组，表示输入数组a和v的离散线性卷积结果。

    See Also
    --------
    numpy.convolve : Equivalent function in the top-level NumPy module.
    """
    # 调用内部函数_convolve_or_correlate，使用np.convolve函数对数组a和v进行卷积或相关操作，返回结果。
    return _convolve_or_correlate(np.convolve, a, v, mode, propagate_mask)
# 定义函数 allequal，用于比较两个数组 a 和 b 是否在指定条件下全部相等
def allequal(a, b, fill_value=True):
    """
    Return True if all entries of a and b are equal, using
    fill_value as a truth value where either or both are masked.

    Parameters
    ----------
    a, b : array_like
        Input arrays to compare.
    fill_value : bool, optional
        Whether masked values in a or b are considered equal (True) or not
        (False).

    Returns
    -------
    y : bool
        Returns True if the two arrays are equal within the given
        tolerance, False otherwise. If either array contains NaN,
        then False is returned.

    See Also
    --------
    all, any
    numpy.ma.allclose

    Examples
    --------
    >>> a = np.ma.array([1e10, 1e-7, 42.0], mask=[0, 0, 1])
    >>> a
    masked_array(data=[10000000000.0, 1e-07, --],
                 mask=[False, False,  True],
           fill_value=1e+20)

    >>> b = np.array([1e10, 1e-7, -42.0])
    >>> b
    array([  1.00000000e+10,   1.00000000e-07,  -4.20000000e+01])
    >>> np.ma.allequal(a, b, fill_value=False)
    False
    >>> np.ma.allequal(a, b)
    True

    """
    # 获取 a 和 b 的掩码并合并
    m = mask_or(getmask(a), getmask(b))
    # 如果合并后的掩码为 nomask，则直接比较数据内容
    if m is nomask:
        x = getdata(a)
        y = getdata(b)
        d = umath.equal(x, y)
        return d.all()
    # 如果合并后的掩码不为 nomask，根据 fill_value 参数处理
    elif fill_value:
        x = getdata(a)
        y = getdata(b)
        d = umath.equal(x, y)
        # 创建一个带有掩码的数组，并将其转换为填充 True 后进行全局比较
        dm = array(d, mask=m, copy=False)
        return dm.filled(True).all(None)
    else:
        return False
    masked_array(data=[10000000000.0, 1e-07, --],
                 mask=[False, False,  True],
           fill_value=1e+20)
    >>> b = np.ma.array([1e10, 1e-8, -42.0], mask=[0, 0, 1])
    >>> np.ma.allclose(a, b)
    False

    >>> a = np.ma.array([1e10, 1e-8, 42.0], mask=[0, 0, 1])
    >>> b = np.ma.array([1.00001e10, 1e-9, -42.0], mask=[0, 0, 1])
    >>> np.ma.allclose(a, b)
    True
    >>> np.ma.allclose(a, b, masked_equal=False)
    False

    Masked values are not compared directly.

    >>> a = np.ma.array([1e10, 1e-8, 42.0], mask=[0, 0, 1])
    >>> b = np.ma.array([1.00001e10, 1e-9, 42.0], mask=[0, 0, 1])
    >>> np.ma.allclose(a, b)
    True
    >>> np.ma.allclose(a, b, masked_equal=False)
    False

    """
    x = masked_array(a, copy=False)  # 创建一个不复制的掩码数组 `x`，使用 `a` 作为数据源
    y = masked_array(b, copy=False)  # 创建一个不复制的掩码数组 `y`，使用 `b` 作为数据源

    # 确保 `y` 是非精确类型，以避免 abs(MIN_INT)；这将导致稍后对 `x` 的类型转换。
    # 注意：我们明确允许时间增量，这曾经有效。这可能会被弃用。参见 gh-18286。
    # 时间增量在 `atol` 是整数或时间增量时有效。
    # 尽管如此，默认的公差可能不太有用。
    if y.dtype.kind != "m":
        dtype = np.result_type(y, 1.)
        if y.dtype != dtype:
            y = masked_array(y, dtype=dtype, copy=False)

    m = mask_or(getmask(x), getmask(y))  # 计算 `x` 和 `y` 的掩码的逻辑或
    xinf = np.isinf(masked_array(x, copy=False, mask=m)).filled(False)  # 检查 `x` 中的无穷值，并填充掩码
    # 如果有任何无穷值，它们应该出现在同一位置。
    if not np.all(xinf == filled(np.isinf(y), False)):
        return False
    # 没有任何无穷值
    if not np.any(xinf):
        d = filled(less_equal(absolute(x - y), atol + rtol * absolute(y)),
                   masked_equal)
        return np.all(d)

    if not np.all(filled(x[xinf] == y[xinf], masked_equal)):
        return False
    x = x[~xinf]
    y = y[~xinf]

    d = filled(less_equal(absolute(x - y), atol + rtol * absolute(y)),
               masked_equal)

    return np.all(d)
def asarray(a, dtype=None, order=None):
    """
    Convert the input to a masked array of the given data-type.

    No copy is performed if the input is already an `ndarray`. If `a` is
    a subclass of `MaskedArray`, a base class `MaskedArray` is returned.

    Parameters
    ----------
    a : array_like
        Input data, in any form that can be converted to a masked array. This
        includes lists, lists of tuples, tuples, tuples of tuples, tuples
        of lists, ndarrays and masked arrays.
    dtype : dtype, optional
        By default, the data-type is inferred from the input data.
    order : {'C', 'F'}, optional
        Whether to use row-major ('C') or column-major ('FORTRAN') memory
        representation.  Default is 'C'.

    Returns
    -------
    out : MaskedArray
        Masked array interpretation of `a`.

    See Also
    --------
    asanyarray : Similar to `asarray`, but conserves subclasses.

    Examples
    --------
    >>> x = np.arange(10.).reshape(2, 5)
    >>> x
    array([[0., 1., 2., 3., 4.],
           [5., 6., 7., 8., 9.]])
    >>> np.ma.asarray(x)
    masked_array(
      data=[[0., 1., 2., 3., 4.],
            [5., 6., 7., 8., 9.]],
      mask=False,
      fill_value=1e+20)
    >>> type(np.ma.asarray(x))
    <class 'numpy.ma.MaskedArray'>

    """
    # 设置默认的存储顺序为 'C'（行优先）
    order = order or 'C'
    # 调用 masked_array 函数，将输入数组转换为 MaskedArray
    return masked_array(a, dtype=dtype, copy=False, keep_mask=True,
                        subok=False, order=order)


def asanyarray(a, dtype=None):
    """
    Convert the input to a masked array, conserving subclasses.

    If `a` is a subclass of `MaskedArray`, its class is conserved.
    No copy is performed if the input is already an `ndarray`.

    Parameters
    ----------
    a : array_like
        Input data, in any form that can be converted to an array.
    dtype : dtype, optional
        By default, the data-type is inferred from the input data.

    Returns
    -------
    out : MaskedArray
        MaskedArray interpretation of `a`.

    See Also
    --------
    asarray : Similar to `asanyarray`, but does not conserve subclass.

    Examples
    --------
    >>> x = np.arange(10.).reshape(2, 5)
    >>> x
    array([[0., 1., 2., 3., 4.],
           [5., 6., 7., 8., 9.]])
    >>> np.ma.asanyarray(x)
    masked_array(
      data=[[0., 1., 2., 3., 4.],
            [5., 6., 7., 8., 9.]],
      mask=False,
      fill_value=1e+20)
    >>> type(np.ma.asanyarray(x))
    <class 'numpy.ma.MaskedArray'>

    """
    # 解决问题 #8666 的临时解决方案，以保留对象的身份。理想情况下，底部行应该自动处理这个问题。
    if isinstance(a, MaskedArray) and (dtype is None or dtype == a.dtype):
        return a
    # 调用 masked_array 函数，将输入数组转换为 MaskedArray，并尽可能保留子类信息
    return masked_array(a, dtype=dtype, copy=False, keep_mask=True, subok=True)
#                               Pickling                                     #
##############################################################################

# 定义一个未实现的函数，用于从文件中读取数据并构建 MaskedArray 对象
def fromfile(file, dtype=float, count=-1, sep=''):
    raise NotImplementedError(
        "fromfile() not yet implemented for a MaskedArray.")

# 从灵活类型的数组构建一个 MaskedArray 对象
def fromflex(fxarray):
    """
    Build a masked array from a suitable flexible-type array.

    The input array has to have a data-type with ``_data`` and ``_mask``
    fields. This type of array is output by `MaskedArray.toflex`.

    Parameters
    ----------
    fxarray : ndarray
        The structured input array, containing ``_data`` and ``_mask``
        fields. If present, other fields are discarded.

    Returns
    -------
    result : MaskedArray
        The constructed masked array.

    See Also
    --------
    MaskedArray.toflex : Build a flexible-type array from a masked array.

    Examples
    --------
    >>> x = np.ma.array(np.arange(9).reshape(3, 3), mask=[0] + [1, 0] * 4)
    >>> rec = x.toflex()
    >>> rec
    array([[(0, False), (1,  True), (2, False)],
           [(3,  True), (4, False), (5,  True)],
           [(6, False), (7,  True), (8, False)]],
          dtype=[('_data', '<i8'), ('_mask', '?')])
    >>> x2 = np.ma.fromflex(rec)
    >>> x2
    masked_array(
      data=[[0, --, 2],
            [--, 4, --],
            [6, --, 8]],
      mask=[[False,  True, False],
            [ True, False,  True],
            [False,  True, False]],
      fill_value=999999)

    Extra fields can be present in the structured array but are discarded:

    >>> dt = [('_data', '<i4'), ('_mask', '|b1'), ('field3', '<f4')]
    >>> rec2 = np.zeros((2, 2), dtype=dt)
    >>> rec2
    array([[(0, False, 0.), (0, False, 0.)],
           [(0, False, 0.), (0, False, 0.)]],
          dtype=[('_data', '<i4'), ('_mask', '?'), ('field3', '<f4')])
    >>> y = np.ma.fromflex(rec2)
    >>> y
    masked_array(
      data=[[0, 0],
            [0, 0]],
      mask=[[False, False],
            [False, False]],
      fill_value=np.int64(999999),
      dtype=int32)

    """
    return masked_array(fxarray['_data'], mask=fxarray['_mask'])

# 用于将 numpy 中的函数转换为 numpy.ma 中的函数
class _convert2ma:

    """
    Convert functions from numpy to numpy.ma.

    Parameters
    ----------
        _methodname : string
            Name of the method to transform.

    """
    __doc__ = None

    # 初始化函数，接收函数名、numpy 返回类型、numpy.ma 返回类型以及可选参数
    def __init__(self, funcname, np_ret, np_ma_ret, params=None):
        self._func = getattr(np, funcname)  # 获取 numpy 中的函数对象
        self.__doc__ = self.getdoc(np_ret, np_ma_ret)  # 获取函数文档注释
        self._extras = params or {}  # 存储额外的参数信息
    def getdoc(self, np_ret, np_ma_ret):
        """
        Return the docstring of the function with modifications.

        Retrieves the docstring of the function (`self._func`) and modifies it
        by replacing the return type information specific to `np` with the
        corresponding return type for `np.ma`.

        Parameters
        ----------
        np_ret : str
            The return type string in the docstring of `np` method.
        np_ma_ret : str
            The desired return type string for `np.ma` method.

        Returns
        -------
        doc : str or None
            Modified docstring with the updated return type information.
            Returns None if `self._func` does not have a docstring.
        """
        doc = getattr(self._func, '__doc__', None)  # Retrieve docstring of self._func
        sig = get_object_signature(self._func)  # Obtain signature of self._func
        if doc:
            doc = self._replace_return_type(doc, np_ret, np_ma_ret)
            # Prepend function signature to the docstring
            if sig:
                sig = "%s%s\n" % (self._func.__name__, sig)
            doc = sig + doc  # Combine signature and docstring
        return doc

    def _replace_return_type(self, doc, np_ret, np_ma_ret):
        """
        Replace the return type documentation in the docstring.

        Checks if the `np_ret` return type exists in the docstring and replaces
        it with `np_ma_ret`.

        Parameters
        ----------
        doc : str
            The original docstring of the function.
        np_ret : str
            The return type string of `np` method to be replaced.
        np_ma_ret : str
            The return type string of `np.ma` method.

        Returns
        -------
        str
            The docstring with replaced return type information.

        Raises
        ------
        RuntimeError
            If `np_ret` is not found in the docstring, indicating an incorrect
            or missing return type information in the docstring.
        """
        if np_ret not in doc:
            raise RuntimeError(
                f"Failed to replace `{np_ret}` with `{np_ma_ret}`. "
                f"The documentation string for return type, {np_ret}, is not "
                f"found in the docstring for `np.{self._func.__name__}`. "
                f"Fix the docstring for `np.{self._func.__name__}` or "
                "update the expected string for return type."
            )

        return doc.replace(np_ret, np_ma_ret)

    def __call__(self, *args, **params):
        """
        Invoke the function with additional or modified parameters.

        Common parameters between `params` and `self._extras` are identified
        and removed from `params`. The function (`self._func`) is then called
        with the remaining parameters. The result is converted to a `MaskedArray`
        view.

        Parameters
        ----------
        *args
            Positional arguments passed to the function.
        **params
            Keyword arguments passed to the function.

        Returns
        -------
        MaskedArray
            Result of invoking `self._func` with modified parameters.

        Notes
        -----
        Modifies `result` based on common parameters like `fill_value` and `hardmask`.
        """
        _extras = self._extras
        common_params = set(params).intersection(_extras)

        # Remove common parameters from `params`
        for p in common_params:
            _extras[p] = params.pop(p)

        # Invoke the function with modified parameters
        result = self._func.__call__(*args, **params).view(MaskedArray)

        # Handle special cases based on common parameters
        if "fill_value" in common_params:
            result.fill_value = _extras.get("fill_value", None)
        if "hardmask" in common_params:
            result._hardmask = bool(_extras.get("hard_mask", False))

        return result
arange = _convert2ma(
    'arange',
    params=dict(fill_value=None, hardmask=False),
    np_ret='arange : ndarray',
    np_ma_ret='arange : MaskedArray',
)
# 将函数名 'arange' 和参数传递给 _convert2ma 函数，生成对应的 ndarray 或 MaskedArray 结果

clip = _convert2ma(
    'clip',
    params=dict(fill_value=None, hardmask=False),
    np_ret='clipped_array : ndarray',
    np_ma_ret='clipped_array : MaskedArray',
)
# 将函数名 'clip' 和参数传递给 _convert2ma 函数，生成对应的 ndarray 或 MaskedArray 结果

empty = _convert2ma(
    'empty',
    params=dict(fill_value=None, hardmask=False),
    np_ret='out : ndarray',
    np_ma_ret='out : MaskedArray',
)
# 将函数名 'empty' 和参数传递给 _convert2ma 函数，生成对应的 ndarray 或 MaskedArray 结果

empty_like = _convert2ma(
    'empty_like',
    np_ret='out : ndarray',
    np_ma_ret='out : MaskedArray',
)
# 将函数名 'empty_like' 传递给 _convert2ma 函数，生成对应的 ndarray 或 MaskedArray 结果

frombuffer = _convert2ma(
    'frombuffer',
    np_ret='out : ndarray',
    np_ma_ret='out: MaskedArray',
)
# 将函数名 'frombuffer' 传递给 _convert2ma 函数，生成对应的 ndarray 或 MaskedArray 结果

fromfunction = _convert2ma(
   'fromfunction',
   np_ret='fromfunction : any',
   np_ma_ret='fromfunction: MaskedArray',
)
# 将函数名 'fromfunction' 传递给 _convert2ma 函数，生成对应的 any 类型或 MaskedArray 结果

identity = _convert2ma(
    'identity',
    params=dict(fill_value=None, hardmask=False),
    np_ret='out : ndarray',
    np_ma_ret='out : MaskedArray',
)
# 将函数名 'identity' 和参数传递给 _convert2ma 函数，生成对应的 ndarray 或 MaskedArray 结果

indices = _convert2ma(
    'indices',
    params=dict(fill_value=None, hardmask=False),
    np_ret='grid : one ndarray or tuple of ndarrays',
    np_ma_ret='grid : one MaskedArray or tuple of MaskedArrays',
)
# 将函数名 'indices' 和参数传递给 _convert2ma 函数，生成对应的 ndarray 或 MaskedArray 结果

ones = _convert2ma(
    'ones',
    params=dict(fill_value=None, hardmask=False),
    np_ret='out : ndarray',
    np_ma_ret='out : MaskedArray',
)
# 将函数名 'ones' 和参数传递给 _convert2ma 函数，生成对应的 ndarray 或 MaskedArray 结果

ones_like = _convert2ma(
    'ones_like',
    np_ret='out : ndarray',
    np_ma_ret='out : MaskedArray',
)
# 将函数名 'ones_like' 传递给 _convert2ma 函数，生成对应的 ndarray 或 MaskedArray 结果

squeeze = _convert2ma(
    'squeeze',
    params=dict(fill_value=None, hardmask=False),
    np_ret='squeezed : ndarray',
    np_ma_ret='squeezed : MaskedArray',
)
# 将函数名 'squeeze' 和参数传递给 _convert2ma 函数，生成对应的 ndarray 或 MaskedArray 结果

zeros = _convert2ma(
    'zeros',
    params=dict(fill_value=None, hardmask=False),
    np_ret='out : ndarray',
    np_ma_ret='out : MaskedArray',
)
# 将函数名 'zeros' 和参数传递给 _convert2ma 函数，生成对应的 ndarray 或 MaskedArray 结果

zeros_like = _convert2ma(
    'zeros_like',
    np_ret='out : ndarray',
    np_ma_ret='out : MaskedArray',
)
# 将函数名 'zeros_like' 传递给 _convert2ma 函数，生成对应的 ndarray 或 MaskedArray 结果


def append(a, b, axis=None):
    """Append values to the end of an array.

    .. versionadded:: 1.9.0

    Parameters
    ----------
    a : array_like
        Values are appended to a copy of this array.
    b : array_like
        These values are appended to a copy of `a`.  It must be of the
        correct shape (the same shape as `a`, excluding `axis`).  If `axis`
        is not specified, `b` can be any shape and will be flattened
        before use.
    axis : int, optional
        The axis along which `v` are appended.  If `axis` is not given,
        both `a` and `b` are flattened before use.

    Returns
    -------
    append : MaskedArray
        A copy of `a` with `b` appended to `axis`.  Note that `append`
        does not occur in-place: a new array is allocated and filled.  If
        `axis` is None, the result is a flattened array.

    See Also
    --------
    numpy.append : Equivalent function in the top-level NumPy module.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> a = ma.masked_values([1, 2, 3], 2)
    >>> b = ma.masked_values([[4, 5, 6], [7, 8, 9]], 7)
    >>> ma.append(a, b)
    """
    pass
# append 函数用于在数组末尾添加值的功能说明和参数说明
    # 创建一个掩码数组，用于处理数据和缺失值的数组
    masked_array(data=[1, --, 3, 4, 5, 6, --, 8, 9],
                 mask=[False,  True, False, False, False, False,  True, False,
                       False],
           fill_value=999999)
    """
    # 沿指定轴连接数组 a 和 b
    return concatenate([a, b], axis)
```