# `.\numpy\numpy\lib\_type_check_impl.py`

```py
"""Automatically adapted for numpy Sep 19, 2005 by convertcode.py

"""
import functools  # 导入 functools 模块，用于创建偏函数
__all__ = ['iscomplexobj', 'isrealobj', 'imag', 'iscomplex',
           'isreal', 'nan_to_num', 'real', 'real_if_close',
           'typename', 'mintypecode',
           'common_type']

from .._utils import set_module  # 导入 set_module 函数，用于设置模块名称
import numpy._core.numeric as _nx  # 导入 numpy 内部的 numeric 模块别名为 _nx
from numpy._core.numeric import asarray, asanyarray, isnan, zeros  # 从 numpy 内部的 numeric 模块导入特定函数
from numpy._core import overrides, getlimits  # 从 numpy 内部的 core 模块导入 overrides 和 getlimits 函数
from ._ufunclike_impl import isneginf, isposinf  # 从当前包的 _ufunclike_impl 模块导入 isneginf 和 isposinf 函数

# 创建 array_function_dispatch 的偏函数，指定 module 为 'numpy'
array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')


# 定义 mintypecode 函数，并用 set_module 装饰器设置模块名称为 'numpy'
@set_module('numpy')
def mintypecode(typechars, typeset='GDFgdf', default='d'):
    """
    Return the character for the minimum-size type to which given types can
    be safely cast.

    The returned type character must represent the smallest size dtype such
    that an array of the returned type can handle the data from an array of
    all types in `typechars` (or if `typechars` is an array, then its
    dtype.char).

    Parameters
    ----------
    typechars : list of str or array_like
        If a list of strings, each string should represent a dtype.
        If array_like, the character representation of the array dtype is used.
    typeset : str or list of str, optional
        The set of characters that the returned character is chosen from.
        The default set is 'GDFgdf'.
    default : str, optional
        The default character, this is returned if none of the characters in
        `typechars` matches a character in `typeset`.

    Returns
    -------
    typechar : str
        The character representing the minimum-size type that was found.

    See Also
    --------
    dtype

    Examples
    --------
    >>> np.mintypecode(['d', 'f', 'S'])
    'd'
    >>> x = np.array([1.1, 2-3.j])
    >>> np.mintypecode(x)
    'D'

    >>> np.mintypecode('abceh', default='G')
    'G'

    """
    # 生成 typechars 中每个类型的字符表示，并以列表形式存储在 typecodes 中
    typecodes = ((isinstance(t, str) and t) or asarray(t).dtype.char
                 for t in typechars)
    # 计算 typecodes 和 typeset 的交集，并转换为集合 intersection
    intersection = set(t for t in typecodes if t in typeset)
    # 如果 intersection 为空集，则返回默认字符 default
    if not intersection:
        return default
    # 如果 intersection 包含 'F' 和 'd'，则返回 'D'
    if 'F' in intersection and 'd' in intersection:
        return 'D'
    # 否则，返回 intersection 中字典序最小的字符
    return min(intersection, key=_typecodes_by_elsize.index)


# 定义 _real_dispatcher 函数，用于作为 real 函数的分发器
def _real_dispatcher(val):
    return (val,)


# 使用 array_function_dispatch 装饰器，将 _real_dispatcher 函数注册为 real 函数的分发实现
@array_function_dispatch(_real_dispatcher)
def real(val):
    """
    Return the real part of the complex argument.

    Parameters
    ----------
    val : array_like
        Input array.

    Returns
    -------
    out : ndarray or scalar
        The real component of the complex argument. If `val` is real, the type
        of `val` is used for the output.  If `val` has complex elements, the
        returned type is float.

    See Also
    --------
    real_if_close, imag, angle

    Examples
    --------
    >>> a = np.array([1+2j, 3+4j, 5+6j])
    >>> a.real
    array([1.,  3.,  5.])
    >>> a.real = 9
    >>> a

    """
    # 返回 val 的实部，如果 val 是复数，则返回浮点数类型的实部
    return _nx.real(val)
    # 创建一个复数数组，包含三个元素，每个元素有实部和虚部
    >>> array([9.+2.j,  9.+4.j,  9.+6.j])
    # 将复数数组 a 的实部部分替换为给定的整数数组
    >>> a.real = np.array([9, 8, 7])
    # 打印数组 a 的当前状态，实部已经被更新，虚部保持不变
    >>> a
    # 计算给定复数的实部，返回结果为包含实数部分的 numpy 数组
    >>> array([9.+2.j,  8.+4.j,  7.+6.j])
    # 计算复数 1+1j 的实部，返回结果为实数部分的浮点数
    >>> np.real(1 + 1j)
    
    """
    # 尝试获取参数 val 的实部，若无法获取则将 val 转换为 numpy 数组后再获取其实部
    try:
        return val.real
    except AttributeError:
        return asanyarray(val).real
# 定义一个辅助函数 _imag_dispatcher，接受一个参数 val，返回一个包含 val 的元组
def _imag_dispatcher(val):
    return (val,)

# 使用 array_function_dispatch 装饰器，将 _imag_dispatcher 作为分派函数
@array_function_dispatch(_imag_dispatcher)
# 定义 imag 函数，用于返回复数参数的虚部
def imag(val):
    """
    返回复数参数的虚部。

    Parameters
    ----------
    val : array_like
        输入数组。

    Returns
    -------
    out : ndarray or scalar
        复数参数的虚部。如果 val 是实数，则输出与 val 相同的类型。如果 val 包含复数元素，返回的类型是 float。

    See Also
    --------
    real, angle, real_if_close

    Examples
    --------
    >>> a = np.array([1+2j, 3+4j, 5+6j])
    >>> a.imag
    array([2.,  4.,  6.])
    >>> a.imag = np.array([8, 10, 12])
    >>> a
    array([1. +8.j,  3.+10.j,  5.+12.j])
    >>> np.imag(1 + 1j)
    1.0

    """
    try:
        return val.imag  # 尝试返回 val 的虚部属性
    except AttributeError:
        return asanyarray(val).imag  # 如果 val 没有 imag 属性，将其转换为数组后再获取虚部


# 定义一个辅助函数 _is_type_dispatcher，接受一个参数 x，返回一个包含 x 的元组
def _is_type_dispatcher(x):
    return (x,)

# 使用 array_function_dispatch 装饰器，将 _is_type_dispatcher 作为分派函数
@array_function_dispatch(_is_type_dispatcher)
# 定义 iscomplex 函数，返回一个布尔数组，指示输入元素是否为复数
def iscomplex(x):
    """
    返回一个布尔数组，其中元素为 True 表示输入元素是复数。

    检测的是输入元素是否具有非零虚部，而不是输入类型是否为复数。

    Parameters
    ----------
    x : array_like
        输入数组。

    Returns
    -------
    out : ndarray of bools
        输出布尔数组。

    See Also
    --------
    isreal
    iscomplexobj : 如果 x 是复数类型或复数数组，则返回 True。

    Examples
    --------
    >>> np.iscomplex([1+1j, 1+0j, 4.5, 3, 2, 2j])
    array([ True, False, False, False, False,  True])

    """
    ax = asanyarray(x)  # 将 x 转换为数组
    if issubclass(ax.dtype.type, _nx.complexfloating):  # 如果数组的类型是复数浮点类型
        return ax.imag != 0  # 返回虚部非零的布尔数组
    res = zeros(ax.shape, bool)  # 创建与 ax 形状相同的布尔数组
    return res[()]  # 将其转换为标量（如果需要）


# 使用 array_function_dispatch 装饰器，将 _is_type_dispatcher 作为分派函数
@array_function_dispatch(_is_type_dispatcher)
# 定义 isreal 函数，返回一个布尔数组，指示输入元素是否为实数
def isreal(x):
    """
    返回一个布尔数组，其中元素为 True 表示输入元素是实数。

    如果元素具有复数类型且虚部为零，则该元素的返回值为 True。

    Parameters
    ----------
    x : array_like
        输入数组。

    Returns
    -------
    out : ndarray, bool
        与 x 相同形状的布尔数组。

    Notes
    -----
    对于字符串或对象数组，isreal 可能会产生意外结果（见示例）。

    See Also
    --------
    iscomplex
    isrealobj : 如果 x 不是复数类型，则返回 True。

    Examples
    --------
    >>> a = np.array([1+1j, 1+0j, 4.5, 3, 2, 2j], dtype=complex)
    >>> np.isreal(a)
    array([False,  True,  True,  True,  True, False])

    函数不适用于字符串数组。

    >>> a = np.array([2j, "a"], dtype="U")
    >>> np.isreal(a)  # 警告：不适用于非逐元素比较
    False

    对于 ``dtype=object`` 的输入数组，即使其中任何元素是复数，也会返回 True。

    >>> a = np.array([1, "2", 3+4j], dtype=object)
    >>> np.isreal(a)
    array([ True,  True,  True])

    """
    isreal should not be used with object arrays
    
    # 创建一个包含复数对象的 NumPy 数组
    a = np.array([1+2j, 2+1j], dtype=object)
    # 检查数组中每个元素是否为实数，但不建议在对象数组上使用该函数
    np.isreal(a)
    # 返回一个布尔数组，表示每个元素是否为实数
    
    """
    返回 x 的虚部是否全为零的布尔值数组
    """
    return imag(x) == 0
@array_function_dispatch(_is_type_dispatcher)
# 使用 array_function_dispatch 装饰器来分派函数调用，以便根据输入类型调用合适的函数
def iscomplexobj(x):
    """
    Check for a complex type or an array of complex numbers.

    The type of the input is checked, not the value. Even if the input
    has an imaginary part equal to zero, `iscomplexobj` evaluates to True.

    Parameters
    ----------
    x : any
        The input can be of any type and shape.

    Returns
    -------
    iscomplexobj : bool
        The return value, True if `x` is of a complex type or has at least
        one complex element.

    See Also
    --------
    isrealobj, iscomplex

    Examples
    --------
    >>> np.iscomplexobj(1)
    False
    >>> np.iscomplexobj(1+0j)
    True
    >>> np.iscomplexobj([3, 1+0j, True])
    True

    """
    try:
        dtype = x.dtype
        type_ = dtype.type
    except AttributeError:
        type_ = asarray(x).dtype.type
    return issubclass(type_, _nx.complexfloating)


@array_function_dispatch(_is_type_dispatcher)
# 使用 array_function_dispatch 装饰器来分派函数调用，以便根据输入类型调用合适的函数
def isrealobj(x):
    """
    Return True if x is a not complex type or an array of complex numbers.

    The type of the input is checked, not the value. So even if the input
    has an imaginary part equal to zero, `isrealobj` evaluates to False
    if the data type is complex.

    Parameters
    ----------
    x : any
        The input can be of any type and shape.

    Returns
    -------
    y : bool
        The return value, False if `x` is of a complex type.

    See Also
    --------
    iscomplexobj, isreal

    Notes
    -----
    The function is only meant for arrays with numerical values but it
    accepts all other objects. Since it assumes array input, the return
    value of other objects may be True.

    >>> np.isrealobj('A string')
    True
    >>> np.isrealobj(False)
    True
    >>> np.isrealobj(None)
    True

    Examples
    --------
    >>> np.isrealobj(1)
    True
    >>> np.isrealobj(1+0j)
    False
    >>> np.isrealobj([3, 1+0j, True])
    False

    """
    return not iscomplexobj(x)

#-----------------------------------------------------------------------------

def _getmaxmin(t):
    # 从 numpy._core 导入 getlimits 函数
    from numpy._core import getlimits
    # 调用 getlimits.finfo 函数，传入类型 t
    f = getlimits.finfo(t)
    # 返回 f 的最大值和最小值
    return f.max, f.min


def _nan_to_num_dispatcher(x, copy=None, nan=None, posinf=None, neginf=None):
    # 返回一个元组 (x,)，用于分派 nan_to_num 函数的调用
    return (x,)


@array_function_dispatch(_nan_to_num_dispatcher)
# 使用 array_function_dispatch 装饰器来分派函数调用，以便根据输入类型调用合适的函数
def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
    """
    Replace NaN with zero and infinity with large finite numbers (default
    behaviour) or with the numbers defined by the user using the `nan`,
    `posinf` and/or `neginf` keywords.

    If `x` is inexact, NaN is replaced by zero or by the user defined value in
    `nan` keyword, infinity is replaced by the largest finite floating point
    values representable by ``x.dtype`` or by the user defined value in
    `posinf` keyword and -infinity is replaced by the most negative finite
    floating point values representable by ``x.dtype`` or by the user defined
    value in `neginf` keyword.

    """
    # 将输入数据 `x` 转换为 NumPy 数组，保留子类信息并根据 `copy` 参数决定是否复制数据
    x = _nx.array(x, subok=True, copy=copy)
    # 获取数组 `x` 的数据类型
    xtype = x.dtype.type
    # 检查 x 是否为标量（即维度为 0）
    isscalar = (x.ndim == 0)
    
    # 如果 x 的类型不是浮点数或复数类型的子类，则根据是否是标量返回 x 的值或单个元素
    if not issubclass(xtype, _nx.inexact):
        return x[()] if isscalar else x
    
    # 检查 x 的类型是否为复数类型的子类
    iscomplex = issubclass(xtype, _nx.complexfloating)
    
    # 如果 x 是复数类型，则将 x 分成实部和虚部，否则将 x 放入一个元组中
    dest = (x.real, x.imag) if iscomplex else (x,)
    
    # 获取 x.real 的数据类型的最大值和最小值
    maxf, minf = _getmaxmin(x.real.dtype)
    
    # 如果 posinf 不为 None，则将 maxf 设为 posinf
    if posinf is not None:
        maxf = posinf
    
    # 如果 neginf 不为 None，则将 minf 设为 neginf
    if neginf is not None:
        minf = neginf
    
    # 遍历 dest 中的元素 d
    for d in dest:
        # 找到 d 中的 NaN 值的索引
        idx_nan = isnan(d)
        # 找到 d 中的正无穷值的索引
        idx_posinf = isposinf(d)
        # 找到 d 中的负无穷值的索引
        idx_neginf = isneginf(d)
        # 将 NaN 的位置替换为 nan
        _nx.copyto(d, nan, where=idx_nan)
        # 将正无穷的位置替换为 maxf
        _nx.copyto(d, maxf, where=idx_posinf)
        # 将负无穷的位置替换为 minf
        _nx.copyto(d, minf, where=idx_neginf)
    
    # 根据是否是标量返回 x 的值或单个元素
    return x[()] if isscalar else x
#-----------------------------------------------------------------------------
# 定义一个私有函数 _real_if_close_dispatcher，用于分发输入参数 a 和 tol
def _real_if_close_dispatcher(a, tol=None):
    # 返回一个包含参数 a 的元组
    return (a,)

# 使用 array_function_dispatch 装饰器，将 _real_if_close_dispatcher 注册为 real_if_close 的分发函数
@array_function_dispatch(_real_if_close_dispatcher)
def real_if_close(a, tol=100):
    """
    如果输入的数组是复数且所有虚部接近零，则返回实部。

    “接近零”的定义是 `tol` 乘以 `a` 的类型的机器 epsilon。

    Parameters
    ----------
    a : array_like
        输入数组。
    tol : float
        数组中元素的虚部的机器 epsilon 的容差。如果容差 <=1，则使用绝对容差。

    Returns
    -------
    out : ndarray
        如果 `a` 是实数，则输出与 `a` 相同的类型。如果 `a` 包含复数元素，则输出类型为 float。

    See Also
    --------
    real, imag, angle

    Notes
    -----
    机器 epsilon 因计算机和数据类型而异，但大多数平台上的 Python 浮点数的机器 epsilon 等于
    2.2204460492503131e-16。您可以使用 'np.finfo(float).eps' 打印出浮点数的机器 epsilon。

    Examples
    --------
    >>> np.finfo(float).eps
    2.2204460492503131e-16 # 可能会有所不同

    >>> np.real_if_close([2.1 + 4e-14j, 5.2 + 3e-15j], tol=1000)
    array([2.1, 5.2])
    >>> np.real_if_close([2.1 + 4e-13j, 5.2 + 3e-15j], tol=1000)
    array([2.1+4.e-13j, 5.2 + 3e-15j])

    """

    # 将输入参数转换为 ndarray 类型
    a = asanyarray(a)
    # 获取数组元素的数据类型
    type_ = a.dtype.type
    # 如果数据类型不是复数类型，则直接返回数组 a
    if not issubclass(type_, _nx.complexfloating):
        return a
    # 如果指定的容差 tol 大于 1，则使用数据类型的机器 epsilon 乘以 tol 作为新的容差值
    if tol > 1:
        f = getlimits.finfo(type_)
        tol = f.eps * tol
    # 如果数组 a 的所有元素的虚部的绝对值均小于 tol，则将数组 a 转换为实部数组
    if _nx.all(_nx.absolute(a.imag) < tol):
        a = a.real
    # 返回处理后的数组 a
    return a


#-----------------------------------------------------------------------------
# 将数据类型代码映射到其描述的字典
_namefromtype = {'S1': 'character',
                 '?': 'bool',
                 'b': 'signed char',
                 'B': 'unsigned char',
                 'h': 'short',
                 'H': 'unsigned short',
                 'i': 'integer',
                 'I': 'unsigned integer',
                 'l': 'long integer',
                 'L': 'unsigned long integer',
                 'q': 'long long integer',
                 'Q': 'unsigned long long integer',
                 'f': 'single precision',
                 'd': 'double precision',
                 'g': 'long precision',
                 'F': 'complex single precision',
                 'D': 'complex double precision',
                 'G': 'complex long double precision',
                 'S': 'string',
                 'U': 'unicode',
                 'V': 'void',
                 'O': 'object'
                 }

# 使用 set_module 装饰器将函数 typename 设置为属于 'numpy' 模块的函数
@set_module('numpy')
def typename(char):
    """
    根据给定的数据类型代码返回描述信息。

    Parameters
    ----------
    char : str
        数据类型代码。

    Returns
    -------
    out : str
        给定数据类型代码的描述信息。

    See Also
    --------
    dtype

    """
    Examples
    --------
    # 定义一个类型字符列表，包含多种数据类型的表示符号
    >>> typechars = ['S1', '?', 'B', 'D', 'G', 'F', 'I', 'H', 'L', 'O', 'Q',
    ...              'S', 'U', 'V', 'b', 'd', 'g', 'f', 'i', 'h', 'l', 'q']
    # 遍历每个类型字符，并输出其对应的类型名称
    >>> for typechar in typechars:
    ...     print(typechar, ' : ', np.typename(typechar))
    ...
    S1  :  character
    ?  :  bool
    B  :  unsigned char
    D  :  complex double precision
    G  :  complex long double precision
    F  :  complex single precision
    I  :  unsigned integer
    H  :  unsigned short
    L  :  unsigned long integer
    O  :  object
    Q  :  unsigned long long integer
    S  :  string
    U  :  unicode
    V  :  void
    b  :  signed char
    d  :  double precision
    g  :  long precision
    f  :  single precision
    i  :  integer
    h  :  short
    l  :  long integer
    q  :  long long integer
    """
    # 根据字符 char 查询并返回对应的类型名称 _namefromtype[char]
    return _namefromtype[char]
# Define two lists that represent possible data types for arrays: floating point types and complex types respectively.
array_type = [[_nx.float16, _nx.float32, _nx.float64, _nx.longdouble],
              [None, _nx.complex64, _nx.complex128, _nx.clongdouble]]

# Define a dictionary that maps each data type to its precision level.
array_precision = {_nx.float16: 0,
                   _nx.float32: 1,
                   _nx.float64: 2,
                   _nx.longdouble: 3,
                   _nx.complex64: 1,
                   _nx.complex128: 2,
                   _nx.clongdouble: 3}


# Dispatcher function that returns all input arrays.
def _common_type_dispatcher(*arrays):
    return arrays


# Decorated function that computes the common scalar type among input arrays.
@array_function_dispatch(_common_type_dispatcher)
def common_type(*arrays):
    """
    Return a scalar type which is common to the input arrays.

    The return type will always be an inexact (i.e. floating point) scalar
    type, even if all the arrays are integer arrays. If one of the inputs is
    an integer array, the minimum precision type that is returned is a
    64-bit floating point dtype.

    All input arrays except int64 and uint64 can be safely cast to the
    returned dtype without loss of information.

    Parameters
    ----------
    array1, array2, ... : ndarrays
        Input arrays.

    Returns
    -------
    out : data type code
        Data type code.

    See Also
    --------
    dtype, mintypecode

    Examples
    --------
    >>> np.common_type(np.arange(2, dtype=np.float32))
    <class 'numpy.float32'>
    >>> np.common_type(np.arange(2, dtype=np.float32), np.arange(2))
    <class 'numpy.float64'>
    >>> np.common_type(np.arange(4), np.array([45, 6.j]), np.array([45.0]))
    <class 'numpy.complex128'>

    """
    # Initialize variables to track if complex types are present and the maximum precision found.
    is_complex = False
    precision = 0
    
    # Iterate through each array to determine its type and update precision accordingly.
    for a in arrays:
        t = a.dtype.type  # Get the type of the array's elements.
        if iscomplexobj(a):  # Check if the array contains complex numbers.
            is_complex = True
        if issubclass(t, _nx.integer):  # Check if the type is a subclass of integer.
            p = 2  # Set precision to that of a double for integer types.
        else:
            p = array_precision.get(t, None)  # Retrieve precision level from array_precision dictionary.
            if p is None:
                raise TypeError("can't get common type for non-numeric array")  # Raise error for non-numeric arrays.
        precision = max(precision, p)  # Update maximum precision encountered.
    
    # Determine and return the common type based on whether complex types were encountered.
    if is_complex:
        return array_type[1][precision]  # Return the complex type with the maximum precision.
    else:
        return array_type[0][precision]  # Return the floating point type with the maximum precision.
```