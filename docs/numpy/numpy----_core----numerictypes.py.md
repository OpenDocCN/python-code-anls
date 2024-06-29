# `.\numpy\numpy\_core\numerictypes.py`

```py
"""
numerictypes: Define the numeric type objects

This module is designed so "from numerictypes import \\*" is safe.
Exported symbols include:

  Dictionary with all registered number types (including aliases):
    sctypeDict

  Type objects (not all will be available, depends on platform):
      see variable sctypes for which ones you have

    Bit-width names

    int8 int16 int32 int64 int128
    uint8 uint16 uint32 uint64 uint128
    float16 float32 float64 float96 float128 float256
    complex32 complex64 complex128 complex192 complex256 complex512
    datetime64 timedelta64

    c-based names

    bool

    object_

    void, str_

    byte, ubyte,
    short, ushort
    intc, uintc,
    intp, uintp,
    int_, uint,
    longlong, ulonglong,

    single, csingle,
    double, cdouble,
    longdouble, clongdouble,

   As part of the type-hierarchy:    xx -- is bit-width

   generic
     +-> bool                                   (kind=b)
     +-> number
     |   +-> integer
     |   |   +-> signedinteger     (intxx)      (kind=i)
     |   |   |     byte
     |   |   |     short
     |   |   |     intc
     |   |   |     intp
     |   |   |     int_
     |   |   |     longlong
     |   |   \\-> unsignedinteger  (uintxx)     (kind=u)
     |   |         ubyte
     |   |         ushort
     |   |         uintc
     |   |         uintp
     |   |         uint
     |   |         ulonglong
     |   +-> inexact
     |       +-> floating          (floatxx)    (kind=f)
     |       |     half
     |       |     single
     |       |     double
     |       |     longdouble
     |       \\-> complexfloating  (complexxx)  (kind=c)
     |             csingle
     |             cdouble
     |             clongdouble
     +-> flexible
     |   +-> character
     |   |     bytes_                           (kind=S)
     |   |     str_                             (kind=U)
     |   |
     |   \\-> void                              (kind=V)
     \\-> object_ (not used much)               (kind=O)

"""
import numbers  # 导入标准库中的 numbers 模块，用于数字类型相关的操作和判断
import warnings  # 导入警告模块，用于发出警告消息

from . import multiarray as ma  # 导入当前包中的 multiarray 模块，并使用别名 ma
from .multiarray import (  # 从 multiarray 模块中导入以下内容：
        ndarray, array, dtype, datetime_data, datetime_as_string,
        busday_offset, busday_count, is_busday, busdaycalendar
        )
from .._utils import set_module  # 从上级包中的 _utils 模块中导入 set_module 函数

# we add more at the bottom
__all__ = [  # 设置模块中可以被导入的公开对象列表
    'ScalarType', 'typecodes', 'issubdtype', 'datetime_data', 
    'datetime_as_string', 'busday_offset', 'busday_count', 
    'is_busday', 'busdaycalendar', 'isdtype'
]

# we don't need all these imports, but we need to keep them for compatibility
# for users using np._core.numerictypes.UPPER_TABLE
from ._string_helpers import (  # 从 _string_helpers 模块中导入以下内容：
    english_lower, english_upper, english_capitalize, LOWER_TABLE, UPPER_TABLE
)

from ._type_aliases import (  # 从 _type_aliases 模块中导入以下内容：
    sctypeDict, allTypes, sctypes
)
from ._dtype import _kind_name  # 从 _dtype 模块中导入 _kind_name 函数

# we don't export these for import *, but we do want them accessible
# as numerictypes.bool, etc.
from builtins import bool, int, float, complex, object, str, bytes  # 从内置模块 builtins 中导入标准类型对象
# 我们稍后会用到这个变量
generic = allTypes['generic']

# 定义所有标量类型的排序列表
# 这些类型按照精度从低到高排序
genericTypeRank = ['bool', 'int8', 'uint8', 'int16', 'uint16',
                   'int32', 'uint32', 'int64', 'uint64', 'int128',
                   'uint128', 'float16',
                   'float32', 'float64', 'float80', 'float96', 'float128',
                   'float256',
                   'complex32', 'complex64', 'complex128', 'complex160',
                   'complex192', 'complex256', 'complex512', 'object']

@set_module('numpy')
def maximum_sctype(t):
    """
    根据输入的类型，返回同种类中最高精度的标量类型。

    .. deprecated:: 2.0
        建议改用显式的 dtype，如 int64 或 float64。

    Parameters
    ----------
    t : dtype 或 dtype 指定符
        输入的数据类型。可以是一个 `dtype` 对象或可以转换为 `dtype` 的对象。

    Returns
    -------
    out : dtype
        与 `t` 同类 (`dtype.kind`) 中最高精度的数据类型。

    See Also
    --------
    obj2sctype, mintypecode, sctype2char
    dtype

    Examples
    --------
    >>> from numpy._core.numerictypes import maximum_sctype
    >>> maximum_sctype(int)
    <class 'numpy.int64'>
    >>> maximum_sctype(np.uint8)
    <class 'numpy.uint64'>
    >>> maximum_sctype(complex)
    <class 'numpy.complex256'> # 结果可能会有所不同

    >>> maximum_sctype(str)
    <class 'numpy.str_'>

    >>> maximum_sctype('i2')
    <class 'numpy.int64'>
    >>> maximum_sctype('f4')
    <class 'numpy.float128'> # 结果可能会有所不同

    """

    # 在 NumPy 2.0 中已弃用，2023-07-11
    warnings.warn(
        "`maximum_sctype` 已弃用。建议改用显式的 dtype，如 int64 或 float64。"
        " (在 NumPy 2.0 中弃用)",
        DeprecationWarning,
        stacklevel=2
    )

    # 获取对象的标量类型
    g = obj2sctype(t)
    if g is None:
        return t
    t = g
    # 获取基本类型的名称
    base = _kind_name(dtype(t))
    # 如果基本类型存在于 sctypes 中，则返回最高精度的数据类型
    if base in sctypes:
        return sctypes[base][-1]
    else:
        return t


@set_module('numpy')
def issctype(rep):
    """
    判断给定对象是否表示标量数据类型。

    Parameters
    ----------
    rep : 任意类型
        如果 `rep` 是标量数据类型的实例，则返回 True。否则返回 False。

    Returns
    -------
    out : bool
        检查 `rep` 是否为标量数据类型的布尔结果。

    See Also
    --------
    issubsctype, issubdtype, obj2sctype, sctype2char

    Examples
    --------
    >>> from numpy._core.numerictypes import issctype
    >>> issctype(np.int32)
    True
    >>> issctype(list)
    False
    >>> issctype(1.1)
    False

    字符串也是标量类型：

    >>> issctype(np.dtype('str'))
    True

    """
    # 如果 rep 不是类型或 dtype 的实例，则返回 False
    if not isinstance(rep, (type, dtype)):
        return False
    try:
        # 尝试获取 rep 的标量类型
        res = obj2sctype(rep)
        # 如果 res 存在且不等于 object_，则返回 True
        if res and res != object_:
            return True
        return False
    except Exception:
        return False


@set_module('numpy')
def obj2sctype(rep, default=None):
    """
    将输入转换为其对应的标量数据类型。
    # 返回对象的标量数据类型或者其在 NumPy 中的等效类型

    Parameters
    ----------
    rep : any
        要获取其类型的对象。
    default : any, optional
        如果提供，则用于无法确定类型的对象。如果未提供，则对于这些对象返回 None。

    Returns
    -------
    dtype : dtype or Python type
        `rep` 的数据类型。

    See Also
    --------
    sctype2char, issctype, issubsctype, issubdtype

    Examples
    --------
    >>> from numpy._core.numerictypes import obj2sctype
    >>> obj2sctype(np.int32)
    <class 'numpy.int32'>
    >>> obj2sctype(np.array([1., 2.]))
    <class 'numpy.float64'>
    >>> obj2sctype(np.array([1.j]))
    <class 'numpy.complex128'>

    >>> obj2sctype(dict)
    <class 'numpy.object_'>
    >>> obj2sctype('string')

    >>> obj2sctype(1, default=list)
    <class 'list'>
    
    """
    # 防止抽象类被向上转型
    if isinstance(rep, type) and issubclass(rep, generic):
        return rep
    # 从数组中提取 dtype
    if isinstance(rep, ndarray):
        return rep.dtype.type
    # 如果转换为 dtype 失败则返回默认值
    try:
        res = dtype(rep)
    except Exception:
        return default
    else:
        return res.type
# 设置模块为 'numpy'
@set_module('numpy')
# 定义函数，判断一个类是否是另一个类的子类
def issubclass_(arg1, arg2):
    """
    Determine if a class is a subclass of a second class.

    `issubclass_` is equivalent to the Python built-in ``issubclass``,
    except that it returns False instead of raising a TypeError if one
    of the arguments is not a class.

    Parameters
    ----------
    arg1 : class
        Input class. True is returned if `arg1` is a subclass of `arg2`.
    arg2 : class or tuple of classes.
        Input class. If a tuple of classes, True is returned if `arg1` is a
        subclass of any of the tuple elements.

    Returns
    -------
    out : bool
        Whether `arg1` is a subclass of `arg2` or not.

    See Also
    --------
    issubsctype, issubdtype, issctype

    Examples
    --------
    >>> np.issubclass_(np.int32, int)
    False
    >>> np.issubclass_(np.int32, float)
    False
    >>> np.issubclass_(np.float64, float)
    True

    """
    try:
        # 调用内置函数 issubclass 检查 arg1 是否是 arg2 的子类
        return issubclass(arg1, arg2)
    except TypeError:
        # 若 TypeError 异常发生，返回 False，不抛出异常
        return False


# 设置模块为 'numpy'
@set_module('numpy')
# 定义函数，判断第一个参数是否是第二个参数的子类
def issubsctype(arg1, arg2):
    """
    Determine if the first argument is a subclass of the second argument.

    Parameters
    ----------
    arg1, arg2 : dtype or dtype specifier
        Data-types.

    Returns
    -------
    out : bool
        The result.

    See Also
    --------
    issctype, issubdtype, obj2sctype

    Examples
    --------
    >>> from numpy._core import issubsctype
    >>> issubsctype('S8', str)
    False
    >>> issubsctype(np.array([1]), int)
    True
    >>> issubsctype(np.array([1]), float)
    False

    """
    # 使用 obj2sctype 函数获取 arg1 和 arg2 的数据类型，然后比较它们是否为子类关系
    return issubclass(obj2sctype(arg1), obj2sctype(arg2))


# 定义异常类 _PreprocessDTypeError
class _PreprocessDTypeError(Exception):
    pass


# 定义函数 _preprocess_dtype，预处理 dtype 参数
def _preprocess_dtype(dtype):
    """
    Preprocess dtype argument by:
      1. fetching type from a data type
      2. verifying that types are built-in NumPy dtypes
    """
    # 如果 dtype 是 masked array 的数据类型，提取其类型
    if isinstance(dtype, ma.dtype):
        dtype = dtype.type
    # 如果 dtype 是 ndarray 或者不在 allTypes.values() 中，抛出 _PreprocessDTypeError 异常
    if isinstance(dtype, ndarray) or dtype not in allTypes.values():
        raise _PreprocessDTypeError()
    # 返回处理后的 dtype
    return dtype


# 设置模块为 'numpy'
@set_module('numpy')
# 定义函数 isdtype，判断提供的 dtype 是否是指定数据类型种类的
def isdtype(dtype, kind):
    """
    Determine if a provided dtype is of a specified data type ``kind``.

    This function only supports built-in NumPy's data types.
    Third-party dtypes are not yet supported.

    Parameters
    ----------
    dtype : dtype
        The input dtype.
    kind : dtype or str or tuple of dtypes/strs.
        dtype or dtype kind. Allowed dtype kinds are:
        * ``'bool'`` : boolean kind
        * ``'signed integer'`` : signed integer data types
        * ``'unsigned integer'`` : unsigned integer data types
        * ``'integral'`` : integer data types
        * ``'real floating'`` : real-valued floating-point data types
        * ``'complex floating'`` : complex floating-point data types
        * ``'numeric'`` : numeric data types

    Returns
    -------
    out : bool

    See Also
    --------
    issubdtype

    Examples
    --------
    >>> import numpy as np

    """
    # 返回 obj2sctype 函数处理后的 dtype 是否符合指定的数据类型种类 kind
    return issubclass(obj2sctype(dtype), obj2sctype(kind))
    # 检查是否 np.float32 是 np.float64 的子类型，返回 False
    >>> np.isdtype(np.float32, np.float64)
    False
    # 检查是否 np.float32 是 "real floating" 中的类型，返回 True
    >>> np.isdtype(np.float32, "real floating")
    True
    # 检查 np.complex128 是否是 "real floating" 或 "complex floating" 中的类型，返回 True
    >>> np.isdtype(np.complex128, ("real floating", "complex floating"))
    True
    
    """
    尝试对 dtype 进行预处理，确保其为 NumPy 的 dtype 类型
    """
    try:
        dtype = _preprocess_dtype(dtype)
    except _PreprocessDTypeError:
        # 如果出现 _PreprocessDTypeError，则抛出 TypeError，指示 dtype 参数必须是 NumPy 的 dtype 类型
        raise TypeError(
            "dtype argument must be a NumPy dtype, "
            f"but it is a {type(dtype)}."
        ) from None
    
    # 如果 kind 是一个 tuple，则将其赋值给 input_kinds，否则将 kind 转为 tuple 并赋值给 input_kinds
    input_kinds = kind if isinstance(kind, tuple) else (kind,)
    
    # 创建一个空集合 processed_kinds，用于存储处理后的数据类型
    processed_kinds = set()
    
    # 遍历 input_kinds 中的每一个 kind
    for kind in input_kinds:
        # 如果 kind 是 "bool"，则将 allTypes["bool"] 添加到 processed_kinds 中
        if kind == "bool":
            processed_kinds.add(allTypes["bool"])
        # 如果 kind 是 "signed integer"，则将 sctypes["int"] 中的所有元素添加到 processed_kinds 中
        elif kind == "signed integer":
            processed_kinds.update(sctypes["int"])
        # 如果 kind 是 "unsigned integer"，则将 sctypes["uint"] 中的所有元素添加到 processed_kinds 中
        elif kind == "unsigned integer":
            processed_kinds.update(sctypes["uint"])
        # 如果 kind 是 "integral"，则将 sctypes["int"] 和 sctypes["uint"] 中的所有元素添加到 processed_kinds 中
        elif kind == "integral":
            processed_kinds.update(sctypes["int"] + sctypes["uint"])
        # 如果 kind 是 "real floating"，则将 sctypes["float"] 中的所有元素添加到 processed_kinds 中
        elif kind == "real floating":
            processed_kinds.update(sctypes["float"])
        # 如果 kind 是 "complex floating"，则将 sctypes["complex"] 中的所有元素添加到 processed_kinds 中
        elif kind == "complex floating":
            processed_kinds.update(sctypes["complex"])
        # 如果 kind 是 "numeric"，则将 sctypes["int"]、sctypes["uint"]、sctypes["float"] 和 sctypes["complex"] 中的所有元素添加到 processed_kinds 中
        elif kind == "numeric":
            processed_kinds.update(
                sctypes["int"] + sctypes["uint"] +
                sctypes["float"] + sctypes["complex"]
            )
        # 如果 kind 是字符串但不是已知的类型名，则抛出 ValueError
        elif isinstance(kind, str):
            raise ValueError(
                "kind argument is a string, but"
                f" {repr(kind)} is not a known kind name."
            )
        else:
            # 否则，尝试对 kind 进行预处理，将其添加到 processed_kinds 中
            try:
                kind = _preprocess_dtype(kind)
            except _PreprocessDTypeError:
                # 如果预处理失败，则抛出 TypeError，指示 kind 参数必须是 NumPy 的 dtype 类型或字符串
                raise TypeError(
                    "kind argument must be comprised of "
                    "NumPy dtypes or strings only, "
                    f"but is a {type(kind)}."
                ) from None
            processed_kinds.add(kind)
    
    # 返回判断 dtype 是否在 processed_kinds 中的结果
    return dtype in processed_kinds
# 使用装饰器设置模块名称为'numpy'
@set_module('numpy')
# 定义函数issubdtype，用于检查第一个参数是否在类型层次结构中低于或等于第二个参数
def issubdtype(arg1, arg2):
    # 如果arg1不是generic的子类，则将其转换为dtype对象的类型
    if not issubclass_(arg1, generic):
        arg1 = dtype(arg1).type
    # 如果arg2不是generic的子类，则将其转换为dtype对象的类型
    if not issubclass_(arg2, generic):
        arg2 = dtype(arg2).type
    # 返回arg1是否为arg2的子类
    return issubclass(arg1, arg2)


# 使用装饰器设置模块名称为'numpy'
@set_module('numpy')
# 定义函数sctype2char，返回标量dtype的字符串表示形式
def sctype2char(sctype):
    # 将sctype转换为其对应的标量类型
    sctype = obj2sctype(sctype)
    # 如果无法识别sctype的类型，则引发ValueError异常
    if sctype is None:
        raise ValueError("unrecognized type")
    # 如果sctype不在sctypeDict的值中，则引发KeyError异常（兼容性）
    if sctype not in sctypeDict.values():
        raise KeyError(sctype)
    # 返回sctype对应的dtype对象的字符表示
    return dtype(sctype).char


# 定义函数_scalar_type_key，作为sorted函数的key函数
def _scalar_type_key(typ):
    # 获取typ对应的dtype对象
    dt = dtype(typ)
    # 返回一个元组，包含dtype的种类的小写形式和项目大小
    return (dt.kind.lower(), dt.itemsize)
# 定义标量类型列表，包括 int、float、complex、bool、bytes、str、memoryview
ScalarType = [int, float, complex, bool, bytes, str, memoryview]
# 将从 sctypeDict 中提取的类型值去重并排序，然后添加到 ScalarType 列表末尾
ScalarType += sorted(set(sctypeDict.values()), key=_scalar_type_key)
# 将 ScalarType 列表转换为元组，使其成为不可变对象
ScalarType = tuple(ScalarType)

# 现在将确定的类型添加到当前模块的全局命名空间中
for key in allTypes:
    globals()[key] = allTypes[key]
    # 将 key 添加到模块的 __all__ 列表中，以便该类型在 import * 时被导入
    __all__.append(key)

# 删除循环中的临时变量 key，以免污染命名空间
del key

# 定义类型码字典，将不同类型映射为对应的字符码
typecodes = {
    'Character': 'c',
    'Integer': 'bhilqnp',
    'UnsignedInteger': 'BHILQNP',
    'Float': 'efdg',
    'Complex': 'FDG',
    'AllInteger': 'bBhHiIlLqQnNpP',
    'AllFloat': 'efdgFDG',
    'Datetime': 'Mm',
    'All': '?bhilqnpBHILQNPefdgFDGSUVOMm'
}

# 向后兼容性 --- 废弃的名称
# 正式废弃：Numpy 1.20.0, 2020-10-19 (参见 numpy/__init__.py)
# 将 sctypeDict 赋值给 typeDict，用于向后兼容
typeDict = sctypeDict

# 定义函数 _register_types()，用于注册数值类型到 numbers 模块的类型系统中
def _register_types():
    # 将 integer 注册为 numbers.Integral 的子类
    numbers.Integral.register(integer)
    # 将 inexact 注册为 numbers.Complex 的子类
    numbers.Complex.register(inexact)
    # 将 floating 注册为 numbers.Real 的子类
    numbers.Real.register(floating)
    # 将 number 注册为 numbers.Number 的子类

# 执行 _register_types() 函数，注册数值类型到 numbers 模块的类型系统中
_register_types()
```