# `.\numpy\numpy\_core\multiarray.py`

```py
"""
Create the numpy._core.multiarray namespace for backward compatibility. 
In v1.16 the multiarray and umath c-extension modules were merged into 
a single _multiarray_umath extension module. So we replicate the old 
namespace by importing from the extension module.
"""

# 导入必要的模块
import functools
from . import overrides
from . import _multiarray_umath
from ._multiarray_umath import *  # noqa: F403

# 这些导入是为了向后兼容所需的，请勿更改。问题 gh-15518

# _get_ndarray_c_version 是半公开的，故意未添加到 __all__ 中
from ._multiarray_umath import (
    _flagdict, from_dlpack, _place, _reconstruct,
    _vec_string, _ARRAY_API, _monotonicity, _get_ndarray_c_version,
    _get_madvise_hugepage, _set_madvise_hugepage,
    _get_promotion_state, _set_promotion_state
)

# 定义模块级别的公共接口列表，用于控制模块导出的公共符号
__all__ = [
    '_ARRAY_API', 'ALLOW_THREADS', 'BUFSIZE', 'CLIP', 'DATETIMEUNITS',
    'ITEM_HASOBJECT', 'ITEM_IS_POINTER', 'LIST_PICKLE', 'MAXDIMS',
    'MAY_SHARE_BOUNDS', 'MAY_SHARE_EXACT', 'NEEDS_INIT', 'NEEDS_PYAPI',
    'RAISE', 'USE_GETITEM', 'USE_SETITEM', 'WRAP',
    '_flagdict', 'from_dlpack', '_place', '_reconstruct', '_vec_string',
    '_monotonicity', 'add_docstring', 'arange', 'array', 'asarray',
    'asanyarray', 'ascontiguousarray', 'asfortranarray', 'bincount',
    'broadcast', 'busday_count', 'busday_offset', 'busdaycalendar', 'can_cast',
    'compare_chararrays', 'concatenate', 'copyto', 'correlate', 'correlate2',
    'count_nonzero', 'c_einsum', 'datetime_as_string', 'datetime_data',
    'dot', 'dragon4_positional', 'dragon4_scientific', 'dtype',
    'empty', 'empty_like', 'error', 'flagsobj', 'flatiter', 'format_longfloat',
    'frombuffer', 'fromfile', 'fromiter', 'fromstring',
    'get_handler_name', 'get_handler_version', 'inner', 'interp',
    'interp_complex', 'is_busday', 'lexsort', 'matmul', 'vecdot',
    'may_share_memory', 'min_scalar_type', 'ndarray', 'nditer', 'nested_iters',
    'normalize_axis_index', 'packbits', 'promote_types', 'putmask',
    'ravel_multi_index', 'result_type', 'scalar', 'set_datetimeparse_function',
    'set_legacy_print_mode',
    'set_typeDict', 'shares_memory', 'typeinfo',
    'unpackbits', 'unravel_index', 'vdot', 'where', 'zeros',
    '_get_promotion_state', '_set_promotion_state'
]

# 为了向后兼容，确保 pickle 从这里导入这些函数
_reconstruct.__module__ = 'numpy._core.multiarray'
scalar.__module__ = 'numpy._core.multiarray'

# 设置以下函数的模块名，以确保它们属于 numpy 的命名空间
from_dlpack.__module__ = 'numpy'
arange.__module__ = 'numpy'
array.__module__ = 'numpy'
asarray.__module__ = 'numpy'
asanyarray.__module__ = 'numpy'
ascontiguousarray.__module__ = 'numpy'
asfortranarray.__module__ = 'numpy'
datetime_data.__module__ = 'numpy'
empty.__module__ = 'numpy'
frombuffer.__module__ = 'numpy'
fromfile.__module__ = 'numpy'
fromiter.__module__ = 'numpy'
frompyfunc.__module__ = 'numpy'
fromstring.__module__ = 'numpy'
may_share_memory.__module__ = 'numpy'
nested_iters.__module__ = 'numpy'
# 将以下对象的模块名称设置为 'numpy'
promote_types.__module__ = 'numpy'
zeros.__module__ = 'numpy'
_get_promotion_state.__module__ = 'numpy'
_set_promotion_state.__module__ = 'numpy'
normalize_axis_index.__module__ = 'numpy'

# 由于 NumPy 的 C 函数不支持内省，因此无法验证调度器的签名。
# 使用 functools.partial 创建一个函数，用于从调度器生成数组函数，并设置相关参数。
array_function_from_c_func_and_dispatcher = functools.partial(
    overrides.array_function_from_dispatcher,
    module='numpy', docs_from_dispatcher=True, verify=False)

# 使用 array_function_from_c_func_and_dispatcher 装饰器创建 empty_like 函数，
# 该函数返回一个与给定数组具有相同形状和类型的新数组。
@array_function_from_c_func_and_dispatcher(_multiarray_umath.empty_like)
def empty_like(
    prototype, dtype=None, order=None, subok=None, shape=None, *, device=None
):
    """
    empty_like(prototype, dtype=None, order='K', subok=True, shape=None, *,
               device=None)

    Return a new array with the same shape and type as a given array.

    Parameters
    ----------
    prototype : array_like
        The shape and data-type of `prototype` define these same attributes
        of the returned array.
    dtype : data-type, optional
        Overrides the data type of the result.

        .. versionadded:: 1.6.0
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result. 'C' means C-order,
        'F' means F-order, 'A' means 'F' if `prototype` is Fortran
        contiguous, 'C' otherwise. 'K' means match the layout of `prototype`
        as closely as possible.

        .. versionadded:: 1.6.0
    subok : bool, optional.
        If True, then the newly created array will use the sub-class
        type of `prototype`, otherwise it will be a base-class array. Defaults
        to True.
    shape : int or sequence of ints, optional.
        Overrides the shape of the result. If order='K' and the number of
        dimensions is unchanged, will try to keep order, otherwise,
        order='C' is implied.

        .. versionadded:: 1.17.0
    device : str, optional
        The device on which to place the created array. Default: None.
        For Array-API interoperability only, so must be ``"cpu"`` if passed.

        .. versionadded:: 2.0.0

    Returns
    -------
    out : ndarray
        Array of uninitialized (arbitrary) data with the same
        shape and type as `prototype`.

    See Also
    --------
    ones_like : Return an array of ones with shape and type of input.
    zeros_like : Return an array of zeros with shape and type of input.
    full_like : Return a new array with shape of input filled with value.
    empty : Return a new uninitialized array.

    Notes
    -----
    Unlike other array creation functions (e.g. `zeros_like`, `ones_like`,
    `full_like`), `empty_like` does not initialize the values of the array,
    and may therefore be marginally faster. However, the values stored in the
    newly allocated array are arbitrary. For reproducible behavior, be sure
    to set each element of the array before reading.

    Examples
    --------
    >>> a = ([1,2,3], [4,5,6])                         # a is array-like
    # 创建一个与给定数组 `a` 具有相同形状和数据类型的未初始化数组
    >>> np.empty_like(a)
    array([[-1073741821, -1073741821,           3],    # 未初始化的值
           [          0,           0, -1073741821]])
    
    # 创建一个新的数组 `a`，包含特定的浮点数值
    >>> a = np.array([[1., 2., 3.],[4.,5.,6.]])
    
    # 创建一个与数组 `a` 具有相同形状和数据类型的未初始化数组
    >>> np.empty_like(a)
    array([[ -2.00000715e+000,   1.48219694e-323,  -2.00000572e+000], # 未初始化的值
           [  4.38791518e-305,  -2.00000715e+000,   4.17269252e-309]])
    
    """   # NOQA
    # 返回一个名为 `prototype` 的对象，通常是一个元组
    return (prototype,)
@array_function_from_c_func_and_dispatcher(_multiarray_umath.concatenate)
# 使用装饰器定义该函数作为从 C 函数 `_multiarray_umath.concatenate` 和调度器生成的数组函数
def concatenate(arrays, axis=None, out=None, *, dtype=None, casting=None):
    """
    concatenate(
        (a1, a2, ...), 
        axis=0, 
        out=None, 
        dtype=None, 
        casting="same_kind"
    )

    Join a sequence of arrays along an existing axis.

    Parameters
    ----------
    a1, a2, ... : sequence of array_like
        The arrays must have the same shape, except in the dimension
        corresponding to `axis` (the first, by default).
    axis : int, optional
        The axis along which the arrays will be joined.  If axis is None,
        arrays are flattened before use.  Default is 0.
    out : ndarray, optional
        If provided, the destination to place the result. The shape must be
        correct, matching that of what concatenate would have returned if no
        out argument were specified.
    dtype : str or dtype
        If provided, the destination array will have this dtype. Cannot be
        provided together with `out`.

        .. versionadded:: 1.20.0

    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur. Defaults to 'same_kind'.
        For a description of the options, please see :term:`casting`.
        
        .. versionadded:: 1.20.0

    Returns
    -------
    res : ndarray
        The concatenated array.

    See Also
    --------
    ma.concatenate : Concatenate function that preserves input masks.
    array_split : Split an array into multiple sub-arrays of equal or
                  near-equal size.
    split : Split array into a list of multiple sub-arrays of equal size.
    hsplit : Split array into multiple sub-arrays horizontally (column wise).
    vsplit : Split array into multiple sub-arrays vertically (row wise).
    dsplit : Split array into multiple sub-arrays along the 3rd axis (depth).
    stack : Stack a sequence of arrays along a new axis.
    block : Assemble arrays from blocks.
    hstack : Stack arrays in sequence horizontally (column wise).
    vstack : Stack arrays in sequence vertically (row wise).
    dstack : Stack arrays in sequence depth wise (along third dimension).
    column_stack : Stack 1-D arrays as columns into a 2-D array.

    Notes
    -----
    When one or more of the arrays to be concatenated is a MaskedArray,
    this function will return a MaskedArray object instead of an ndarray,
    but the input masks are *not* preserved. In cases where a MaskedArray
    is expected as input, use the ma.concatenate function from the masked
    array module instead.

    Examples
    --------
    >>> a = np.array([[1, 2], [3, 4]])
    >>> b = np.array([[5, 6]])
    >>> np.concatenate((a, b), axis=0)
    array([[1, 2],
           [3, 4],
           [5, 6]])
    >>> np.concatenate((a, b.T), axis=1)
    array([[1, 2, 5],
           [3, 4, 6]])
    >>> np.concatenate((a, b), axis=None)
    array([1, 2, 3, 4, 5, 6])
    """
    # 实现数组沿指定轴的连接操作，返回连接后的数组
    pass
    # 如果给定的输出参数 `out` 不为 None，则执行以下代码块
    if out is not None:
        # 将输入的数组 `arrays` 转换为列表，以便进行修改
        arrays = list(arrays)
        # 将 `out` 参数添加到 `arrays` 列表末尾
        arrays.append(out)
    # 返回经过处理的数组列表 `arrays`
    return arrays
# 定义一个用于计算两个数组内积的函数，底层使用 C 函数实现
@array_function_from_c_func_and_dispatcher(_multiarray_umath.inner)
def inner(a, b):
    """
    inner(a, b, /)

    Inner product of two arrays.

    Ordinary inner product of vectors for 1-D arrays (without complex
    conjugation), in higher dimensions a sum product over the last axes.

    Parameters
    ----------
    a, b : array_like
        If `a` and `b` are nonscalar, their last dimensions must match.

    Returns
    -------
    out : ndarray
        If `a` and `b` are both
        scalars or both 1-D arrays then a scalar is returned; otherwise
        an array is returned.
        ``out.shape = (*a.shape[:-1], *b.shape[:-1])``

    Raises
    ------
    ValueError
        If both `a` and `b` are nonscalar and their last dimensions have
        different sizes.

    See Also
    --------
    tensordot : Sum products over arbitrary axes.
    dot : Generalised matrix product, using second last dimension of `b`.
    einsum : Einstein summation convention.

    Notes
    -----
    For vectors (1-D arrays) it computes the ordinary inner-product::

        np.inner(a, b) = sum(a[:]*b[:])

    More generally, if ``ndim(a) = r > 0`` and ``ndim(b) = s > 0``::

        np.inner(a, b) = np.tensordot(a, b, axes=(-1,-1))

    or explicitly::

        np.inner(a, b)[i0,...,ir-2,j0,...,js-2]
             = sum(a[i0,...,ir-2,:]*b[j0,...,js-2,:])

    In addition `a` or `b` may be scalars, in which case::

       np.inner(a,b) = a*b

    Examples
    --------
    Ordinary inner product for vectors:

    >>> a = np.array([1,2,3])
    >>> b = np.array([0,1,0])
    >>> np.inner(a, b)
    2

    Some multidimensional examples:

    >>> a = np.arange(24).reshape((2,3,4))
    >>> b = np.arange(4)
    >>> c = np.inner(a, b)
    >>> c.shape
    (2, 3)
    >>> c
    array([[ 14,  38,  62],
           [ 86, 110, 134]])

    >>> a = np.arange(2).reshape((1,1,2))
    >>> b = np.arange(6).reshape((3,2))
    >>> c = np.inner(a, b)
    >>> c.shape
    (1, 1, 3)
    >>> c
    array([[[1, 3, 5]]])

    An example where `b` is a scalar:

    >>> np.inner(np.eye(2), 7)
    array([[7., 0.],
           [0., 7.]])

    """
    # 直接返回参数 a 和 b，作为该函数的结果
    return (a, b)


# 定义一个从 C 函数和分发器生成的数组函数，用于根据条件从 x 或 y 中选择元素
@array_function_from_c_func_and_dispatcher(_multiarray_umath.where)
def where(condition, x=None, y=None):
    """
    where(condition, [x, y], /)

    Return elements chosen from `x` or `y` depending on `condition`.

    .. note::
        When only `condition` is provided, this function is a shorthand for
        ``np.asarray(condition).nonzero()``. Using `nonzero` directly should be
        preferred, as it behaves correctly for subclasses. The rest of this
        documentation covers only the case where all three arguments are
        provided.

    Parameters
    ----------
    condition : array_like, bool
        Where True, yield `x`, otherwise yield `y`.
    x, y : array_like
        Values from which to choose. `x`, `y` and `condition` need to be
        broadcastable to some shape.

    Returns
    -------
    """
    out : ndarray
        返回的数组，其元素来自于 `x`，其中 `condition` 为 True，而 `y` 中的元素则来自于其他位置。

    See Also
    --------
    choose
    nonzero : 当 x 和 y 被省略时调用的函数

    Notes
    -----
    如果所有的数组都是 1-D，`where` 等效于::

        [xv if c else yv
         for c, xv, yv in zip(condition, x, y)]

    Examples
    --------
    >>> a = np.arange(10)
    >>> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> np.where(a < 5, a, 10*a)
    array([ 0,  1,  2,  3,  4, 50, 60, 70, 80, 90])

    这也适用于多维数组：

    >>> np.where([[True, False], [True, True]],
    ...          [[1, 2], [3, 4]],
    ...          [[9, 8], [7, 6]])
    array([[1, 8],
           [3, 4]])

    x、y 和 condition 的形状会被一起广播：

    >>> x, y = np.ogrid[:3, :4]
    >>> np.where(x < y, x, 10 + y)  # x 和 10+y 都会被广播
    array([[10,  0,  0,  0],
           [10, 11,  1,  1],
           [10, 11, 12,  2]])

    >>> a = np.array([[0, 1, 2],
    ...               [0, 2, 4],
    ...               [0, 3, 6]])
    >>> np.where(a < 4, a, -1)  # -1 会被广播
    array([[ 0,  1,  2],
           [ 0,  2, -1],
           [ 0,  3, -1]])
    """
    return (condition, x, y)
# 从 _multiarray_umath.lexsort 函数和分发器创建一个 array_function
@array_function_from_c_func_and_dispatcher(_multiarray_umath.lexsort)
# 定义 lexsort 函数，用于多键稳定间接排序
def lexsort(keys, axis=None):
    """
    lexsort(keys, axis=-1)

    执行使用一系列键的间接稳定排序。

    给定多个排序键，lexsort 返回一个整数索引数组，
    描述了按多个键排序的顺序。序列中的最后一个键用于主排序顺序，
    次要排序使用倒数第二个键，依此类推。

    Parameters
    ----------
    keys : (k, m, n, ...) array-like
        要排序的 k 个键。最后一个键（例如，如果 keys 是二维数组，则为最后一行）
        是主要的排序键。沿着零轴的每个 keys 的元素必须是相同形状的数组对象。
    axis : int, optional
        要间接排序的轴。默认情况下，对每个序列的最后一个轴进行排序。
        独立地沿着 axis 切片排序；参见最后一个示例。

    Returns
    -------
    indices : (m, n, ...) ndarray of ints
        沿指定轴排序键的索引数组。

    See Also
    --------
    argsort : 间接排序。
    ndarray.sort : 原地排序。
    sort : 返回数组的排序副本。

    Examples
    --------
    按姓氏和名字排序。

    >>> surnames =    ('Hertz',    'Galilei', 'Hertz')
    >>> first_names = ('Heinrich', 'Galileo', 'Gustav')
    >>> ind = np.lexsort((first_names, surnames))
    >>> ind
    array([1, 2, 0])

    >>> [surnames[i] + ", " + first_names[i] for i in ind]
    ['Galilei, Galileo', 'Hertz, Gustav', 'Hertz, Heinrich']

    根据两个数字键排序，首先按照 ``a`` 的元素，然后按照 ``b`` 的元素进行排序。

    >>> a = [1, 5, 1, 4, 3, 4, 4]  # 第一个序列
    >>> b = [9, 4, 0, 4, 0, 2, 1]  # 第二个序列
    >>> ind = np.lexsort((b, a))  # 先按 `a` 排序，然后按 `b` 排序
    >>> ind
    array([2, 0, 4, 6, 5, 3, 1])
    >>> [(a[i], b[i]) for i in ind]
    [(1, 0), (1, 9), (3, 0), (4, 1), (4, 2), (4, 4), (5, 4)]

    与 `argsort` 进行比较，它会独立地对每个键进行排序。

    >>> np.argsort((b, a), kind='stable')
    array([[2, 4, 6, 5, 1, 3, 0],
           [0, 2, 4, 3, 5, 6, 1]])

    要使用 `argsort` 按字典顺序排序，我们需要提供一个结构化数组。

    >>> x = np.array([(ai, bi) for ai, bi in zip(a, b)],
    ...              dtype = np.dtype([('x', int), ('y', int)]))
    >>> np.argsort(x)  # 或者 np.argsort(x, order=('x', 'y'))
    array([2, 0, 4, 6, 5, 3, 1])

    keys 的零轴始终对应于键序列，因此 2D 数组与其他键序列一样对待。

    >>> arr = np.asarray([b, a])
    >>> ind2 = np.lexsort(arr)
    >>> np.testing.assert_equal(ind2, ind)

    因此，`axis` 参数指的是每个键的轴，而不是 keys 参数本身的轴。
    例如，数组 ``arr`` 被视为
    """
    # 如果 keys 是一个元组，则直接返回 keys 自身
    if isinstance(keys, tuple):
        return keys
    # 否则，将 keys 包装成一个元组并返回
    else:
        return (keys,)
# 从 _multiarray_umath.can_cast 导入 C 函数并创建 array_function 装饰器，使 can_cast 函数与它关联
@array_function_from_c_func_and_dispatcher(_multiarray_umath.can_cast)
# 定义函数 can_cast，用于确定是否可以按照给定的类型转换规则从 from_ 类型到 to 类型进行转换
def can_cast(from_, to, casting=None):
    """
    can_cast(from_, to, casting='safe')

    Returns True if cast between data types can occur according to the
    casting rule.

    Parameters
    ----------
    from_ : dtype, dtype specifier, NumPy scalar, or array
        Data type, NumPy scalar, or array to cast from.
    to : dtype or dtype specifier
        Data type to cast to.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur.

        * 'no' means the data types should not be cast at all.
        * 'equiv' means only byte-order changes are allowed.
        * 'safe' means only casts which can preserve values are allowed.
        * 'same_kind' means only safe casts or casts within a kind,
          like float64 to float32, are allowed.
        * 'unsafe' means any data conversions may be done.

    Returns
    -------
    out : bool
        True if cast can occur according to the casting rule.

    Notes
    -----
    .. versionchanged:: 1.17.0
       Casting between a simple data type and a structured one is possible only
       for "unsafe" casting.  Casting to multiple fields is allowed, but
       casting from multiple fields is not.

    .. versionchanged:: 1.9.0
       Casting from numeric to string types in 'safe' casting mode requires
       that the string dtype length is long enough to store the maximum
       integer/float value converted.

    .. versionchanged:: 2.0
       This function does not support Python scalars anymore and does not
       apply any value-based logic for 0-D arrays and NumPy scalars.

    See also
    --------
    dtype, result_type

    Examples
    --------
    Basic examples

    >>> np.can_cast(np.int32, np.int64)
    True
    >>> np.can_cast(np.float64, complex)
    True
    >>> np.can_cast(complex, float)
    False

    >>> np.can_cast('i8', 'f8')
    True
    >>> np.can_cast('i8', 'f4')
    False
    >>> np.can_cast('i4', 'S4')
    False

    """
    # 返回输入的 from_ 参数，这里仅作为示例，实际函数实现应该根据输入参数进行相应的类型转换判断逻辑
    return (from_,)


# 从 _multiarray_umath.min_scalar_type 导入 C 函数并创建 array_function 装饰器，使 min_scalar_type 函数与它关联
@array_function_from_c_func_and_dispatcher(_multiarray_umath.min_scalar_type)
# 定义函数 min_scalar_type，用于确定标量值 a 的最小数据类型
def min_scalar_type(a):
    """
    min_scalar_type(a, /)

    For scalar ``a``, returns the data type with the smallest size
    and smallest scalar kind which can hold its value.  For non-scalar
    array ``a``, returns the vector's dtype unmodified.

    Floating point values are not demoted to integers,
    and complex values are not demoted to floats.

    Parameters
    ----------
    a : scalar or array_like
        The value whose minimal data type is to be found.

    Returns
    -------
    out : dtype
        The minimal data type.

    Notes
    -----
    .. versionadded:: 1.6.0

    See Also
    --------
    result_type, promote_types, dtype, can_cast

    Examples
    --------
    >>> np.min_scalar_type(10)
    dtype('uint8')

    >>> np.min_scalar_type(-260)
    dtype('int16')


    """
    # 调用 NumPy 库中的函数 np.min_scalar_type，返回适合表示给定标量值的最小数据类型
    >>> np.min_scalar_type(3.1)
    dtype('float16')
    
    # 调用 NumPy 库中的函数 np.min_scalar_type，返回适合表示给定标量值的最小数据类型
    >>> np.min_scalar_type(1e50)
    dtype('float64')
    
    # 调用 NumPy 库中的函数 np.min_scalar_type，返回适合表示给定数组的最小数据类型
    >>> np.min_scalar_type(np.arange(4, dtype='f8'))
    dtype('float64')
    
    """
    函数定义：接收参数 a，并返回以元组形式包含 a 的结果。
    """
    return (a,)
# 使用 _multiarray_umath.result_type 函数和相关的调度器装饰器，定义了 result_type 函数
@array_function_from_c_func_and_dispatcher(_multiarray_umath.result_type)
def result_type(*arrays_and_dtypes):
    """
    result_type(*arrays_and_dtypes)

    Returns the type that results from applying the NumPy
    type promotion rules to the arguments.

    Type promotion in NumPy works similarly to the rules in languages
    like C++, with some slight differences.  When both scalars and
    arrays are used, the array's type takes precedence and the actual value
    of the scalar is taken into account.

    For example, calculating 3*a, where a is an array of 32-bit floats,
    intuitively should result in a 32-bit float output.  If the 3 is a
    32-bit integer, the NumPy rules indicate it can't convert losslessly
    into a 32-bit float, so a 64-bit float should be the result type.
    By examining the value of the constant, '3', we see that it fits in
    an 8-bit integer, which can be cast losslessly into the 32-bit float.

    Parameters
    ----------
    arrays_and_dtypes : list of arrays and dtypes
        The operands of some operation whose result type is needed.

    Returns
    -------
    out : dtype
        The result type.

    See also
    --------
    dtype, promote_types, min_scalar_type, can_cast

    Notes
    -----
    .. versionadded:: 1.6.0

    The specific algorithm used is as follows.

    Categories are determined by first checking which of boolean,
    integer (int/uint), or floating point (float/complex) the maximum
    kind of all the arrays and the scalars are.

    If there are only scalars or the maximum category of the scalars
    is higher than the maximum category of the arrays,
    the data types are combined with :func:`promote_types`
    to produce the return value.

    Otherwise, `min_scalar_type` is called on each scalar, and
    the resulting data types are all combined with :func:`promote_types`
    to produce the return value.

    The set of int values is not a subset of the uint values for types
    with the same number of bits, something not reflected in
    :func:`min_scalar_type`, but handled as a special case in `result_type`.

    Examples
    --------
    >>> np.result_type(3, np.arange(7, dtype='i1'))
    dtype('int8')

    >>> np.result_type('i4', 'c8')
    dtype('complex128')

    >>> np.result_type(3.0, -2)
    dtype('float64')

    """
    # 直接返回输入参数的结果类型
    return arrays_and_dtypes


# 使用 _multiarray_umath.dot 函数和相关的调度器装饰器，定义了 dot 函数
@array_function_from_c_func_and_dispatcher(_multiarray_umath.dot)
def dot(a, b, out=None):
    """
    dot(a, b, out=None)

    Dot product of two arrays. Specifically,

    - If both `a` and `b` are 1-D arrays, it is inner product of vectors
      (without complex conjugation).

    - If both `a` and `b` are 2-D arrays, it is matrix multiplication,
      but using :func:`matmul` or ``a @ b`` is preferred.

    - If either `a` or `b` is 0-D (scalar), it is equivalent to
      :func:`multiply` and using ``numpy.multiply(a, b)`` or ``a * b`` is
      preferred.

    """
    # 返回数组 a 和 b 的点积结果
    pass  # 这里是一个占位符，表示函数体暂未实现
    # 计算两个数组的点积（dot product）。
    # 如果 `a` 是 N 维数组，`b` 是 1 维数组，则是 `a` 的最后一个轴和 `b` 的点积。
    # 如果 `a` 是 N 维数组，`b` 是 M 维数组（其中 M >= 2），则是 `a` 的最后一个轴和 `b` 的倒数第二个轴的点积。
    # 使用优化的 BLAS 库（参见 `numpy.linalg`）进行计算。
    
    def dot(a, b, out=None):
        """
        Parameters
        ----------
        a : array_like
            第一个参数。
        b : array_like
            第二个参数。
        out : ndarray, optional
            输出参数。必须具有与 `dot(a, b)` 返回类型相同的确切类型，必须是 C 连续的，
            其 dtype 必须是 `dot(a, b)` 返回的 dtype。这是一个性能特性。因此，如果不满足这些条件，
            将引发异常，而不是尝试灵活处理。
    
        Returns
        -------
        output : ndarray
            返回 `a` 和 `b` 的点积。如果 `a` 和 `b` 都是标量或都是 1 维数组，则返回标量；否则返回数组。
            如果给定了 `out`，则返回 `out`。
    
        Raises
        ------
        ValueError
            如果 `a` 的最后一个维度大小与 `b` 的倒数第二个维度大小不同。
    
        See Also
        --------
        vdot : 复共轭点积。
        tensordot : 在任意轴上进行求和积。
        einsum : Einstein 求和约定。
        matmul : '@' 操作符作为带有 out 参数的方法。
        linalg.multi_dot : 链式点积。
    
        Examples
        --------
        >>> np.dot(3, 4)
        12
    
        两个参数都不是复共轭的情况：
    
        >>> np.dot([2j, 3j], [2j, 3j])
        (-13+0j)
    
        对于二维数组，它是矩阵乘积：
    
        >>> a = [[1, 0], [0, 1]]
        >>> b = [[4, 1], [2, 2]]
        >>> np.dot(a, b)
        array([[4, 1],
               [2, 2]])
    
        >>> a = np.arange(3*4*5*6).reshape((3,4,5,6))
        >>> b = np.arange(3*4*5*6)[::-1].reshape((5,4,6,3))
        >>> np.dot(a, b)[2,3,2,1,2,2]
        499128
        >>> sum(a[2,3,2,:] * b[1,2,:,2])
        499128
    
        """
        return (a, b, out)
# 使用 `_multiarray_umath.vdot` 函数和分派器创建一个 `vdot` 函数，并装饰它
@array_function_from_c_func_and_dispatcher(_multiarray_umath.vdot)
def vdot(a, b):
    """
    vdot(a, b, /)

    Return the dot product of two vectors.

    The vdot(`a`, `b`) function handles complex numbers differently than
    dot(`a`, `b`).  If the first argument is complex the complex conjugate
    of the first argument is used for the calculation of the dot product.

    Note that `vdot` handles multidimensional arrays differently than `dot`:
    it does *not* perform a matrix product, but flattens input arguments
    to 1-D vectors first. Consequently, it should only be used for vectors.

    Parameters
    ----------
    a : array_like
        If `a` is complex the complex conjugate is taken before calculation
        of the dot product.
    b : array_like
        Second argument to the dot product.

    Returns
    -------
    output : ndarray
        Dot product of `a` and `b`.  Can be an int, float, or
        complex depending on the types of `a` and `b`.

    See Also
    --------
    dot : Return the dot product without using the complex conjugate of the
          first argument.

    Examples
    --------
    >>> a = np.array([1+2j,3+4j])
    >>> b = np.array([5+6j,7+8j])
    >>> np.vdot(a, b)
    (70-8j)
    >>> np.vdot(b, a)
    (70+8j)

    Note that higher-dimensional arrays are flattened!

    >>> a = np.array([[1, 4], [5, 6]])
    >>> b = np.array([[4, 1], [2, 2]])
    >>> np.vdot(a, b)
    30
    >>> np.vdot(b, a)
    30
    >>> 1*4 + 4*1 + 5*2 + 6*2
    30

    """
    # 直接返回输入参数 a, b，实际上并未计算点积，只是为了示例目的
    return (a, b)


# 使用 `_multiarray_umath.bincount` 函数和分派器创建一个 `bincount` 函数，并装饰它
@array_function_from_c_func_and_dispatcher(_multiarray_umath.bincount)
def bincount(x, weights=None, minlength=None):
    """
    bincount(x, /, weights=None, minlength=0)

    Count number of occurrences of each value in array of non-negative ints.

    The number of bins (of size 1) is one larger than the largest value in
    `x`. If `minlength` is specified, there will be at least this number
    of bins in the output array (though it will be longer if necessary,
    depending on the contents of `x`).
    Each bin gives the number of occurrences of its index value in `x`.
    If `weights` is specified the input array is weighted by it, i.e. if a
    value ``n`` is found at position ``i``, ``out[n] += weight[i]`` instead
    of ``out[n] += 1``.

    Parameters
    ----------
    x : array_like, 1 dimension, nonnegative ints
        Input array.
    weights : array_like, optional
        Weights, array of the same shape as `x`.
    minlength : int, optional
        A minimum number of bins for the output array.

        .. versionadded:: 1.6.0

    Returns
    -------
    out : ndarray of ints
        The result of binning the input array.
        The length of `out` is equal to ``np.amax(x)+1``.

    Raises
    ------
    ValueError
        If the input is not 1-dimensional, or contains elements with negative
        values, or if `minlength` is negative.
    TypeError
        If the type of the input is float or complex.

    See Also

    """
    # 直接返回输入参数 x，实际上并未进行 binning 的计算，只是为了示例目的
    return x
    # 计算输入数组中每个整数值的出现次数，并返回结果数组
    histogram, digitize, unique

    # 示例1：计算从0到4的整数数组的频次
    >>> np.bincount(np.arange(5))
    array([1, 1, 1, 1, 1])

    # 示例2：计算指定数组中每个整数值的频次
    >>> np.bincount(np.array([0, 1, 1, 3, 2, 1, 7]))
    array([1, 3, 1, 1, 0, 0, 0, 1])

    # 示例3：验证结果数组的大小是否等于数组中最大值加一
    >>> x = np.array([0, 1, 1, 3, 2, 1, 7, 23])
    >>> np.bincount(x).size == np.amax(x)+1
    True

    # 如果输入数组的数据类型不是整数，则会引发 TypeError
    The input array needs to be of integer dtype, otherwise a
    TypeError is raised:
    >>> np.bincount(np.arange(5, dtype=float))
    Traceback (most recent call last):
      ...
    TypeError: Cannot cast array data from dtype('float64') to dtype('int64')
    according to the rule 'safe'

    # 使用 weights 关键字可以对数组的可变大小块执行加权求和
    A possible use of ``bincount`` is to perform sums over
    variable-size chunks of an array, using the ``weights`` keyword.
    >>> w = np.array([0.3, 0.5, 0.2, 0.7, 1., -0.6]) # weights
    >>> x = np.array([0, 1, 1, 2, 2, 2])
    >>> np.bincount(x, weights=w)
    array([ 0.3,  0.7,  1.1])

    """
    return (x, weights)
# 将 _multiarray_umath.ravel_multi_index 函数从 C 函数和分发器转换为数组函数装饰器
@array_function_from_c_func_and_dispatcher(_multiarray_umath.ravel_multi_index)
def ravel_multi_index(multi_index, dims, mode=None, order=None):
    """
    ravel_multi_index(multi_index, dims, mode='raise', order='C')

    Converts a tuple of index arrays into an array of flat
    indices, applying boundary modes to the multi-index.

    Parameters
    ----------
    multi_index : tuple of array_like
        A tuple of integer arrays, one array for each dimension.
    dims : tuple of ints
        The shape of array into which the indices from ``multi_index`` apply.
    mode : {'raise', 'wrap', 'clip'}, optional
        Specifies how out-of-bounds indices are handled.  Can specify
        either one mode or a tuple of modes, one mode per index.

        * 'raise' -- raise an error (default)
        * 'wrap' -- wrap around
        * 'clip' -- clip to the range

        In 'clip' mode, a negative index which would normally
        wrap will clip to 0 instead.
    order : {'C', 'F'}, optional
        Determines whether the multi-index should be viewed as
        indexing in row-major (C-style) or column-major
        (Fortran-style) order.

    Returns
    -------
    raveled_indices : ndarray
        An array of indices into the flattened version of an array
        of dimensions ``dims``.

    See Also
    --------
    unravel_index

    Notes
    -----
    .. versionadded:: 1.6.0

    Examples
    --------
    >>> arr = np.array([[3,6,6],[4,5,1]])
    >>> np.ravel_multi_index(arr, (7,6))
    array([22, 41, 37])
    >>> np.ravel_multi_index(arr, (7,6), order='F')
    array([31, 41, 13])
    >>> np.ravel_multi_index(arr, (4,6), mode='clip')
    array([22, 23, 19])
    >>> np.ravel_multi_index(arr, (4,4), mode=('clip','wrap'))
    array([12, 13, 13])

    >>> np.ravel_multi_index((3,1,4,1), (6,7,8,9))
    1621
    """
    return multi_index


# 将 _multiarray_umath.unravel_index 函数从 C 函数和分发器转换为数组函数装饰器
@array_function_from_c_func_and_dispatcher(_multiarray_umath.unravel_index)
def unravel_index(indices, shape=None, order=None):
    """
    unravel_index(indices, shape, order='C')

    Converts a flat index or array of flat indices into a tuple
    of coordinate arrays.

    Parameters
    ----------
    indices : array_like
        An integer array whose elements are indices into the flattened
        version of an array of dimensions ``shape``. Before version 1.6.0,
        this function accepted just one index value.
    shape : tuple of ints
        The shape of the array to use for unraveling ``indices``.

        .. versionchanged:: 1.16.0
            Renamed from ``dims`` to ``shape``.

    order : {'C', 'F'}, optional
        Determines whether the indices should be viewed as indexing in
        row-major (C-style) or column-major (Fortran-style) order.

        .. versionadded:: 1.6.0

    Returns
    -------
    unraveled_coords : tuple of ndarray
        Each array in the tuple has the same shape as the ``indices``
        array.

    See Also
    --------
    ravel_multi_index

    Examples
    --------
    >>> indices = np.array([22, 41, 37])
    >>> shape = (7, 6)
    >>> np.unravel_index(indices, shape)
    (array([3, 6, 6]), array([4, 5, 1]))
    
    >>> indices = 1621
    >>> shape = (6, 7, 8, 9)
    >>> np.unravel_index(indices, shape)
    (3, 1, 4, 1)
    """
    return indices
    # 使用 numpy 库中的函数 unravel_index 来根据给定的索引和形状返回多维数组中的坐标。
    # 示例1: 在形状为 (7, 6) 的二维数组中，根据一维索引 [22, 41, 37] 返回的坐标。
    # 结果为 (array([3, 6, 6]), array([4, 5, 1]))，分别对应三个索引在二维数组中的位置。
    >>> np.unravel_index([22, 41, 37], (7, 6))
    
    # 示例2: 在以列序优先 ('F') 排序的形状为 (7, 6) 的二维数组中，根据一维索引 [31, 41, 13] 返回的坐标。
    # 结果同样为 (array([3, 6, 6]), array([4, 5, 1]))，因为列序优先的顺序下索引的解析顺序不同但解析的坐标相同。
    >>> np.unravel_index([31, 41, 13], (7, 6), order='F')
    
    # 示例3: 在形状为 (6, 7, 8, 9) 的四维数组中，根据一维索引 1621 返回的坐标。
    # 结果为 (3, 1, 4, 1)，对应索引 1621 在四维数组中的位置。
    >>> np.unravel_index(1621, (6, 7, 8, 9))
    
    """
    返回一个包含示例返回值的元组，表示解析索引后的坐标。
    """
    return (indices,)
# 使用装饰器将 copyto 函数与 C 函数 _multiarray_umath.copyto 绑定
@array_function_from_c_func_and_dispatcher(_multiarray_umath.copyto)
# 定义函数 copyto，用于将源数组 src 的值复制到目标数组 dst 中，支持必要的广播
def copyto(dst, src, casting=None, where=None):
    """
    copyto(dst, src, casting='same_kind', where=True)

    Copies values from one array to another, broadcasting as necessary.

    Raises a TypeError if the `casting` rule is violated, and if
    `where` is provided, it selects which elements to copy.

    .. versionadded:: 1.7.0

    Parameters
    ----------
    dst : ndarray
        The array into which values are copied.
    src : array_like
        The array from which values are copied.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur when copying.

        * 'no' means the data types should not be cast at all.
        * 'equiv' means only byte-order changes are allowed.
        * 'safe' means only casts which can preserve values are allowed.
        * 'same_kind' means only safe casts or casts within a kind,
          like float64 to float32, are allowed.
        * 'unsafe' means any data conversions may be done.
    where : array_like of bool, optional
        A boolean array which is broadcasted to match the dimensions
        of `dst`, and selects elements to copy from `src` to `dst`
        wherever it contains the value True.

    Examples
    --------
    >>> A = np.array([4, 5, 6])
    >>> B = [1, 2, 3]
    >>> np.copyto(A, B)
    >>> A
    array([1, 2, 3])

    >>> A = np.array([[1, 2, 3], [4, 5, 6]])
    >>> B = [[4, 5, 6], [7, 8, 9]]
    >>> np.copyto(A, B)
    >>> A
    array([[4, 5, 6],
           [7, 8, 9]])

    """
    return (dst, src, where)


# 使用装饰器将 putmask 函数与 C 函数 _multiarray_umath.putmask 绑定
@array_function_from_c_func_and_dispatcher(_multiarray_umath.putmask)
# 定义函数 putmask，根据条件和输入值改变数组元素
def putmask(a, /, mask, values):
    """
    putmask(a, mask, values)

    Changes elements of an array based on conditional and input values.

    Sets ``a.flat[n] = values[n]`` for each n where ``mask.flat[n]==True``.

    If `values` is not the same size as `a` and `mask` then it will repeat.
    This gives behavior different from ``a[mask] = values``.

    Parameters
    ----------
    a : ndarray
        Target array.
    mask : array_like
        Boolean mask array. It has to be the same shape as `a`.
    values : array_like
        Values to put into `a` where `mask` is True. If `values` is smaller
        than `a` it will be repeated.

    See Also
    --------
    place, put, take, copyto

    Examples
    --------
    >>> x = np.arange(6).reshape(2, 3)
    >>> np.putmask(x, x>2, x**2)
    >>> x
    array([[ 0,  1,  2],
           [ 9, 16, 25]])

    If `values` is smaller than `a` it is repeated:

    >>> x = np.arange(5)
    >>> np.putmask(x, x>1, [-33, -44])
    >>> x
    array([  0,   1, -33, -44, -33])

    """
    return (a, mask, values)


# 使用装饰器将 packbits 函数与 C 函数 _multiarray_umath.packbits 绑定
@array_function_from_c_func_and_dispatcher(_multiarray_umath.packbits)
# 定义函数 packbits，将布尔数组转换为按位打包的数组
def packbits(a, axis=None, bitorder='big'):
    """
    packbits(a, /, axis=None, bitorder='big')

    """

# 使用装饰器将 packbits 函数与 C 函数 _multiarray_umath.packbits 绑定
@array_function_from_c_func_and_dispatcher(_multiarray_umath.packbits)
# 定义函数 packbits，将布尔数组转换为按位打包的数组
def packbits(a, axis=None, bitorder='big'):
    """
    packbits(a, /, axis=None, bitorder='big')

    """
    # 将二进制值数组的元素打包成 uint8 数组中的位。
    
    # 结果通过在末尾插入零位进行填充至完整字节。
    
    # Parameters 参数
    # ----------
    # a : array_like
    #     整数或布尔值数组，其元素应打包成位。
    # axis : int, optional
    #     进行位打包的维度。
    #     ``None`` 表示打包扁平化数组。
    # bitorder : {'big', 'little'}, optional
    #     输入位的顺序。'big' 将模拟 bin(val)，
    #     ``[0, 0, 0, 0, 0, 0, 1, 1] => 3 = 0b00000011``，
    #     'little' 将反转顺序，所以 ``[1, 1, 0, 0, 0, 0, 0, 0] => 3``。
    #     默认为 'big'。
    #
    #     .. versionadded:: 1.17.0
    #
    # Returns 返回值
    # -------
    # packed : ndarray
    #     uint8 类型的数组，其元素表示输入元素的逻辑（0 或非零）值对应的位。
    #     `packed` 的形状与输入的维度相同（除非 `axis` 是 None，在这种情况下输出是 1-D）。
    #
    # See Also 参见
    # --------
    # unpackbits: 将 uint8 数组的元素解包成二进制输出数组。
    
    # Examples 示例
    # --------
    # >>> a = np.array([[[1,0,1],
    # ...                [0,1,0]],
    # ...               [[1,1,0],
    # ...                [0,0,1]]])
    # >>> b = np.packbits(a, axis=-1)
    # >>> b
    # array([[[160],
    #         [ 64]],
    #        [[192],
    #         [ 32]]], dtype=uint8)
    #
    # 注意，在二进制中，160 = 1010 0000，64 = 0100 0000，192 = 1100 0000，
    # 和 32 = 0010 0000。
    
    def packbits(a, axis=None, bitorder='big'):
        return (a,)
# 使用 C 函数和调度器创建一个 array_function，将其关联到 _multiarray_umath.unpackbits 函数
@array_function_from_c_func_and_dispatcher(_multiarray_umath.unpackbits)
def unpackbits(a, axis=None, count=None, bitorder='big'):
    """
    unpackbits(a, /, axis=None, count=None, bitorder='big')

    从 uint8 类型的数组 `a` 中解包元素到一个二进制数值的输出数组中。

    每个 `a` 的元素表示一个应该被解包到一个二进制数值输出数组中的位字段。
    输出数组的形状可以是 1-D（如果 `axis` 是 ``None``）或者与输入数组相同，
    且沿指定轴进行解包。

    Parameters
    ----------
    a : ndarray, uint8 type
       输入数组。
    axis : int, optional
        进行位解包的维度。
        ``None`` 意味着对扁平化的数组进行解包。
    count : int or None, optional
        沿着 `axis` 解包的元素数量，提供的目的是撤销尺寸不是八的倍数的打包效果。
        非负数意味着只解包 `count` 位。
        负数意味着从末尾裁剪掉这么多位。
        ``None`` 意味着解包整个数组（默认）。
        比可用位数大的计数将会在输出中添加零填充。
        负计数不能超过可用的位数。

        .. versionadded:: 1.17.0

    bitorder : {'big', 'little'}, optional
        返回位的顺序。'big' 将模拟 bin(val)，``3 = 0b00000011 => [0, 0, 0, 0, 0, 0, 1, 1]``，
        'little' 将顺序反转为 ``[1, 1, 0, 0, 0, 0, 0, 0]``。
        默认为 'big'。

        .. versionadded:: 1.17.0

    Returns
    -------
    unpacked : ndarray, uint8 type
       元素是二进制值（0 或 1）。

    See Also
    --------
    packbits : 将二进制值数组的元素打包到 uint8 数组中的位。

    Examples
    --------
    >>> a = np.array([[2], [7], [23]], dtype=np.uint8)
    >>> a
    array([[ 2],
           [ 7],
           [23]], dtype=uint8)
    >>> b = np.unpackbits(a, axis=1)
    >>> b
    array([[0, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 1, 1, 1],
           [0, 0, 0, 1, 0, 1, 1, 1]], dtype=uint8)
    >>> c = np.unpackbits(a, axis=1, count=-3)
    >>> c
    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0]], dtype=uint8)

    >>> p = np.packbits(b, axis=0)
    >>> np.unpackbits(p, axis=0)
    array([[0, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 1, 1, 1],
           [0, 0, 0, 1, 0, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    >>> np.array_equal(b, np.unpackbits(p, axis=0, count=b.shape[0]))
    True

    """
    return (a,)


# 使用 C 函数和调度器创建一个 array_function，将其关联到 _multiarray_umath.shares_memory 函数
@array_function_from_c_func_and_dispatcher(_multiarray_umath.shares_memory)
def shares_memory(a, b, max_work=None):
    """
    shares_memory(a, b, /, max_work=None)
    # 检查两个数组是否共享内存空间的函数定义

    Determine if two arrays share memory.
    # 确定两个数组是否共享内存空间。

    .. warning::
    # 警告信息开始

       This function can be exponentially slow for some inputs, unless
       `max_work` is set to a finite number or ``MAY_SHARE_BOUNDS``.
       If in doubt, use `numpy.may_share_memory` instead.
    # 对于某些输入，该函数可能会呈指数级增长地慢，除非将 `max_work` 设置为有限的数值或 ``MAY_SHARE_BOUNDS``。
    # 如果不确定，请使用 `numpy.may_share_memory` 替代。

    Parameters
    ----------
    a, b : ndarray
        Input arrays
    # 输入参数：a、b - 数组类型

    max_work : int, optional
        Effort to spend on solving the overlap problem (maximum number
        of candidate solutions to consider). The following special
        values are recognized:
        # 可选参数：max_work - 解决重叠问题的尝试次数（考虑的候选解的最大数量）。
        # 下列特殊值被识别：

        max_work=MAY_SHARE_EXACT  (default)
            The problem is solved exactly. In this case, the function returns
            True only if there is an element shared between the arrays. Finding
            the exact solution may take extremely long in some cases.
        # max_work=MAY_SHARE_EXACT  (默认)
        # 精确解决问题。在这种情况下，如果数组之间存在共享元素，则函数返回 True。在某些情况下，找到精确解可能需要非常长的时间。

        max_work=MAY_SHARE_BOUNDS
            Only the memory bounds of a and b are checked.
        # max_work=MAY_SHARE_BOUNDS
        # 只检查 a 和 b 的内存边界。

    Raises
    ------
    numpy.exceptions.TooHardError
        Exceeded max_work.
    # 抛出异常：numpy.exceptions.TooHardError - 超过了 max_work。

    Returns
    -------
    out : bool
    # 返回结果：布尔值

    See Also
    --------
    may_share_memory
    # 参见：may_share_memory 函数

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4])
    >>> np.shares_memory(x, np.array([5, 6, 7]))
    False
    >>> np.shares_memory(x[::2], x)
    True
    >>> np.shares_memory(x[::2], x[1::2])
    False

    Checking whether two arrays share memory is NP-complete, and
    runtime may increase exponentially in the number of
    dimensions. Hence, `max_work` should generally be set to a finite
    number, as it is possible to construct examples that take
    extremely long to run:
    # 检查两个数组是否共享内存是 NP 完全问题，且运行时间可能在维度数量上呈指数增长。
    # 因此，`max_work` 通常应设置为有限的数值，因为可能构造需要运行极长时间的示例：

    >>> from numpy.lib.stride_tricks import as_strided
    >>> x = np.zeros([192163377], dtype=np.int8)
    >>> x1 = as_strided(
    ...     x, strides=(36674, 61119, 85569), shape=(1049, 1049, 1049))
    >>> x2 = as_strided(
    ...     x[64023025:], strides=(12223, 12224, 1), shape=(1049, 1049, 1))
    >>> np.shares_memory(x1, x2, max_work=1000)
    Traceback (most recent call last):
    ...
    numpy.exceptions.TooHardError: Exceeded max_work
    # 跟踪到最近的调用：numpy.exceptions.TooHardError - 超过了 max_work。

    Running ``np.shares_memory(x1, x2)`` without `max_work` set takes
    around 1 minute for this case. It is possible to find problems
    that take still significantly longer.
    # 在没有设置 `max_work` 的情况下运行 ``np.shares_memory(x1, x2)`` 大约需要 1 分钟。可能会找到需要更长时间的问题。

    """
    return (a, b)
    # 返回两个输入的元组作为结果
# 使用装饰器将 _multiarray_umath.may_share_memory 转换为 array_function，并使用其相关的分发器
@array_function_from_c_func_and_dispatcher(_multiarray_umath.may_share_memory)
# 定义函数 may_share_memory，用于判断两个数组是否可能共享内存
def may_share_memory(a, b, max_work=None):
    """
    may_share_memory(a, b, /, max_work=None)

    Determine if two arrays might share memory

    A return of True does not necessarily mean that the two arrays
    share any element.  It just means that they *might*.

    Only the memory bounds of a and b are checked by default.

    Parameters
    ----------
    a, b : ndarray
        Input arrays
    max_work : int, optional
        Effort to spend on solving the overlap problem.  See
        `shares_memory` for details.  Default for ``may_share_memory``
        is to do a bounds check.

    Returns
    -------
    out : bool

    See Also
    --------
    shares_memory

    Examples
    --------
    >>> np.may_share_memory(np.array([1,2]), np.array([5,8,9]))
    False
    >>> x = np.zeros([3, 4])
    >>> np.may_share_memory(x[:,0], x[:,1])
    True

    """
    # 直接返回输入的数组 a 和 b，用于判断它们是否可能共享内存
    return (a, b)


# 使用装饰器将 _multiarray_umath.is_busday 转换为 array_function，并使用其相关的分发器
@array_function_from_c_func_and_dispatcher(_multiarray_umath.is_busday)
# 定义函数 is_busday，用于计算给定日期是否是有效的工作日
def is_busday(dates, weekmask=None, holidays=None, busdaycal=None, out=None):
    """
    is_busday(
        dates, 
        weekmask='1111100', 
        holidays=None, 
        busdaycal=None, 
        out=None
    )

    Calculates which of the given dates are valid days, and which are not.

    .. versionadded:: 1.7.0

    Parameters
    ----------
    dates : array_like of datetime64[D]
        The array of dates to process.
    weekmask : str or array_like of bool, optional
        A seven-element array indicating which of Monday through Sunday are
        valid days. May be specified as a length-seven list or array, like
        [1,1,1,1,1,0,0]; a length-seven string, like '1111100'; or a string
        like "Mon Tue Wed Thu Fri", made up of 3-character abbreviations for
        weekdays, optionally separated by white space. Valid abbreviations
        are: Mon Tue Wed Thu Fri Sat Sun
    holidays : array_like of datetime64[D], optional
        An array of dates to consider as invalid dates.  They may be
        specified in any order, and NaT (not-a-time) dates are ignored.
        This list is saved in a normalized form that is suited for
        fast calculations of valid days.
    busdaycal : busdaycalendar, optional
        A `busdaycalendar` object which specifies the valid days. If this
        parameter is provided, neither weekmask nor holidays may be
        provided.
    out : array of bool, optional
        If provided, this array is filled with the result.

    Returns
    -------
    out : array of bool
        An array with the same shape as ``dates``, containing True for
        each valid day, and False for each invalid day.

    See Also
    --------
    busdaycalendar : An object that specifies a custom set of valid days.
    busday_offset : Applies an offset counted in valid days.
    busday_count : Counts how many valid days are in a half-open date range.

    Examples
    --------
    """
    # 直接返回输入的日期数组 dates，用于判断每个日期是否是有效的工作日
    return (dates, weekmask, holidays, busdaycal, out)
    # 使用 numpy 库中的 is_busday 函数来判断给定日期是否是工作日
    # dates: 要检查的日期列表
    # weekmask: 定义工作日的掩码，例如 '1111100' 表示周一至周五为工作日
    # holidays: 节假日列表，这些日期被视为非工作日
    # out: 可选参数，用于指定输出结果的存储位置
    
    return (dates, weekmask, holidays, out)
# 定义一个装饰器，将 C 函数和调度器包装成数组函数，功能与 _multiarray_umath.busday_offset 相同
@array_function_from_c_func_and_dispatcher(_multiarray_umath.busday_offset)
# busday_offset 函数定义，用于计算工作日偏移量
def busday_offset(dates, offsets, roll=None, weekmask=None, holidays=None,
                  busdaycal=None, out=None):
    """
    busday_offset(
        dates, 
        offsets, 
        roll='raise', 
        weekmask='1111100', 
        holidays=None, 
        busdaycal=None, 
        out=None
    )

    First adjusts the date to fall on a valid day according to
    the ``roll`` rule, then applies offsets to the given dates
    counted in valid days.

    .. versionadded:: 1.7.0

    Parameters
    ----------
    dates : array_like of datetime64[D]
        要处理的日期数组。
    offsets : array_like of int
        偏移量数组，与 dates 进行广播。
    roll : {'raise', 'nat', 'forward', 'following', 'backward', 'preceding', \
        'modifiedfollowing', 'modifiedpreceding'}, optional
        处理不落在有效日期的日期的规则。默认为 'raise'。

        * 'raise' 表示对无效日期抛出异常。
        * 'nat' 表示对无效日期返回 NaT (not-a-time)。
        * 'forward' 和 'following' 表示取稍后的第一个有效日期。
        * 'backward' 和 'preceding' 表示取稍早的第一个有效日期。
        * 'modifiedfollowing' 表示取稍后的第一个有效日期，除非跨越月边界，在这种情况下取稍早的第一个有效日期。
        * 'modifiedpreceding' 表示取稍早的第一个有效日期，除非跨越月边界，在这种情况下取稍后的第一个有效日期。
    weekmask : str or array_like of bool, optional
        一个长度为七的数组，指示星期一到星期日哪些是有效日期。可以指定为长度为七的列表或数组，如 [1,1,1,1,1,0,0]；
        一个长度为七的字符串，如 '1111100'；或一个字符串，如 "Mon Tue Wed Thu Fri"，由星期几的三字母缩写组成，
        可以用空格分隔。有效的缩写包括：Mon Tue Wed Thu Fri Sat Sun
    holidays : array_like of datetime64[D], optional
        作为无效日期考虑的日期数组。可以以任何顺序指定，NaT (not-a-time) 日期将被忽略。
        此列表以适合快速计算有效日期的规范化形式保存。
    busdaycal : busdaycalendar, optional
        指定有效日期的 busdaycalendar 对象。如果提供了此参数，则 weekmask 和 holidays 不能提供。
    out : array of datetime64[D], optional
        如果提供，将用结果填充的数组。

    Returns
    -------
    out : array of datetime64[D]
        通过广播 dates 和 offsets 得到的形状的数组，包含应用了偏移量的日期。
    """
    """
    See Also
    --------
    busdaycalendar : An object that specifies a custom set of valid days.
        定义了一组自定义有效日期的对象。
    is_busday : Returns a boolean array indicating valid days.
        返回一个布尔数组，指示哪些日期是有效的工作日。
    busday_count : Counts how many valid days are in a half-open date range.
        统计半开区间日期范围内的有效工作日数量。

    Examples
    --------
    >>> # First business day in October 2011 (not accounting for holidays)
    ... np.busday_offset('2011-10', 0, roll='forward')
    np.datetime64('2011-10-03')
    >>> # Last business day in February 2012 (not accounting for holidays)
    ... np.busday_offset('2012-03', -1, roll='forward')
    np.datetime64('2012-02-29')
    >>> # Third Wednesday in January 2011
    ... np.busday_offset('2011-01', 2, roll='forward', weekmask='Wed')
    np.datetime64('2011-01-19')
    >>> # 2012 Mother's Day in Canada and the U.S.
    ... np.busday_offset('2012-05', 1, roll='forward', weekmask='Sun')
    np.datetime64('2012-05-13')

    >>> # First business day on or after a date
    ... np.busday_offset('2011-03-20', 0, roll='forward')
    np.datetime64('2011-03-21')
    >>> np.busday_offset('2011-03-22', 0, roll='forward')
    np.datetime64('2011-03-22')
    >>> # First business day after a date
    ... np.busday_offset('2011-03-20', 1, roll='backward')
    np.datetime64('2011-03-21')
    >>> np.busday_offset('2011-03-22', 1, roll='backward')
    np.datetime64('2011-03-23')
    """
    返回 (dates, offsets, weekmask, holidays, out)
# 从 C 函数和调度器生成的数组函数装饰器，将 C 函数 _multiarray_umath.busday_count 转换为数组函数
@array_function_from_c_func_and_dispatcher(_multiarray_umath.busday_count)
def busday_count(begindates, enddates, weekmask=None, holidays=None,
                 busdaycal=None, out=None):
    """
    busday_count(
        begindates, 
        enddates, 
        weekmask='1111100', 
        holidays=[], 
        busdaycal=None, 
        out=None
    )

    Counts the number of valid days between `begindates` and
    `enddates`, not including the day of `enddates`.

    If ``enddates`` specifies a date value that is earlier than the
    corresponding ``begindates`` date value, the count will be negative.

    .. versionadded:: 1.7.0

    Parameters
    ----------
    begindates : array_like of datetime64[D]
        The array of the first dates for counting.
    enddates : array_like of datetime64[D]
        The array of the end dates for counting, which are excluded
        from the count themselves.
    weekmask : str or array_like of bool, optional
        A seven-element array indicating which of Monday through Sunday are
        valid days. May be specified as a length-seven list or array, like
        [1,1,1,1,1,0,0]; a length-seven string, like '1111100'; or a string
        like "Mon Tue Wed Thu Fri", made up of 3-character abbreviations for
        weekdays, optionally separated by white space. Valid abbreviations
        are: Mon Tue Wed Thu Fri Sat Sun
    holidays : array_like of datetime64[D], optional
        An array of dates to consider as invalid dates.  They may be
        specified in any order, and NaT (not-a-time) dates are ignored.
        This list is saved in a normalized form that is suited for
        fast calculations of valid days.
    busdaycal : busdaycalendar, optional
        A `busdaycalendar` object which specifies the valid days. If this
        parameter is provided, neither weekmask nor holidays may be
        provided.
    out : array of int, optional
        If provided, this array is filled with the result.

    Returns
    -------
    out : array of int
        An array with a shape from broadcasting ``begindates`` and ``enddates``
        together, containing the number of valid days between
        the begin and end dates.

    See Also
    --------
    busdaycalendar : An object that specifies a custom set of valid days.
    is_busday : Returns a boolean array indicating valid days.
    busday_offset : Applies an offset counted in valid days.

    Examples
    --------
    >>> # Number of weekdays in January 2011
    ... np.busday_count('2011-01', '2011-02')
    21
    >>> # Number of weekdays in 2011
    >>> np.busday_count('2011', '2012')
    260
    >>> # Number of Saturdays in 2011
    ... np.busday_count('2011', '2012', weekmask='Sat')
    53
    """
    # 直接返回函数参数，用于展示函数签名和文档示例
    return (begindates, enddates, weekmask, holidays, out)


# 从 C 函数和调度器生成的数组函数装饰器，将 C 函数 _multiarray_umath.datetime_as_string 转换为数组函数
@array_function_from_c_func_and_dispatcher(
    _multiarray_umath.datetime_as_string)
def datetime_as_string(arr, unit=None, timezone=None, casting=None):
    """
    """
    # 将数组中的日期时间转换为字符串表示形式的函数
    def datetime_as_string(arr, unit=None, timezone='naive', casting='same_kind'):
        # arr: datetime64 类型的数组，需要格式化的 UTC 时间戳数组
        # unit: 字符串，表示日期时间的精度单位，可以是 None, 'auto'，或者参考 datetime 单位的字符串
        # timezone: 字符串 {'naive', 'UTC', 'local'} 或者 tzinfo 对象，指定显示日期时间时使用的时区信息
        #           'naive' 表示没有指定时区，'UTC' 表示使用 UTC 时区并以 'Z' 结尾，'local' 表示转换为本地时区并带有 +-#### 的时区偏移量
        #           如果是 tzinfo 对象，则与 'local' 类似，但使用指定的时区
        # casting: 字符串 {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}，指定在不同日期时间单位之间转换时的类型转换规则
    
        # 返回值
        # str_arr: 与 arr 具有相同形状的字符串数组
    
        return (arr,)
    
    
    这段代码是一个函数定义，用于将给定的日期时间数组（以 numpy 的 datetime64 类型表示）转换为字符串数组。它支持多种参数配置，包括日期时间精度、时区信息和类型转换规则。
```