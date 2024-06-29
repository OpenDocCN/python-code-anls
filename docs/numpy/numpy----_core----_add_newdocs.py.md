# `.\numpy\numpy\_core\_add_newdocs.py`

```
# 导入需要的函数和模块，用于更新 C 扩展模块中对象的文档字符串而无需重新编译
from numpy._core.function_base import add_newdoc
from numpy._core.overrides import array_function_like_doc

###############################################################################
#
# flatiter
#
# flatiter needs a toplevel description
#
###############################################################################

# 添加新文档说明给 'numpy._core' 中的 'flatiter' 对象
add_newdoc('numpy._core', 'flatiter',
    """
    Flat iterator object to iterate over arrays.

    A `flatiter` iterator is returned by ``x.flat`` for any array `x`.
    It allows iterating over the array as if it were a 1-D array,
    either in a for-loop or by calling its `next` method.

    Iteration is done in row-major, C-style order (the last
    index varying the fastest). The iterator can also be indexed using
    basic slicing or advanced indexing.

    See Also
    --------
    ndarray.flat : Return a flat iterator over an array.
    ndarray.flatten : Returns a flattened copy of an array.

    Notes
    -----
    A `flatiter` iterator can not be constructed directly from Python code
    by calling the `flatiter` constructor.

    Examples
    --------
    >>> x = np.arange(6).reshape(2, 3)
    >>> fl = x.flat
    >>> type(fl)
    <class 'numpy.flatiter'>
    >>> for item in fl:
    ...     print(item)
    ...
    0
    1
    2
    3
    4
    5

    >>> fl[2:4]
    array([2, 3])

    """)

# 添加新文档说明给 'numpy._core' 中的 'flatiter' 对象的 'base' 属性
add_newdoc('numpy._core', 'flatiter', ('base',
    """
    A reference to the array that is iterated over.

    Examples
    --------
    >>> x = np.arange(5)
    >>> fl = x.flat
    >>> fl.base is x
    True

    """))

# 添加新文档说明给 'numpy._core' 中的 'flatiter' 对象的 'coords' 属性
add_newdoc('numpy._core', 'flatiter', ('coords',
    """
    An N-dimensional tuple of current coordinates.

    Examples
    --------
    >>> x = np.arange(6).reshape(2, 3)
    >>> fl = x.flat
    >>> fl.coords
    (0, 0)
    >>> next(fl)
    0
    >>> fl.coords
    (0, 1)

    """))

# 添加新文档说明给 'numpy._core' 中的 'flatiter' 对象的 'index' 属性
add_newdoc('numpy._core', 'flatiter', ('index',
    """
    Current flat index into the array.

    Examples
    --------
    >>> x = np.arange(6).reshape(2, 3)
    >>> fl = x.flat
    >>> fl.index
    0
    >>> next(fl)
    0
    >>> fl.index
    1

    """))

# 添加新文档说明给 'numpy._core' 中的 'flatiter' 对象的 '__array__' 方法
add_newdoc('numpy._core', 'flatiter', ('__array__',
    """__array__(type=None) Get array from iterator

    """))

# 添加新文档说明给 'numpy._core' 中的 'flatiter' 对象的 'copy' 方法
add_newdoc('numpy._core', 'flatiter', ('copy',
    """
    copy()

    Get a copy of the iterator as a 1-D array.

    Examples
    --------
    >>> x = np.arange(6).reshape(2, 3)
    >>> x
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> fl = x.flat
    >>> fl.copy()
    array([0, 1, 2, 3, 4, 5])

    """))
###############################################################################
#
# nditer
#
###############################################################################

# 向numpy._core模块添加新文档字符串，描述nditer函数的用途和参数
add_newdoc('numpy._core', 'nditer',
    """
    nditer(op, flags=None, op_flags=None, op_dtypes=None, order='K',
        casting='safe', op_axes=None, itershape=None, buffersize=0)

    Efficient multi-dimensional iterator object to iterate over arrays.
    To get started using this object, see the
    :ref:`introductory guide to array iteration <arrays.nditer>`.

    Parameters
    ----------
    op : ndarray or sequence of array_like
        The array(s) to iterate over.

    flags : sequence of str, optional
          Flags to control the behavior of the iterator.

          * ``buffered`` enables buffering when required.
          * ``c_index`` causes a C-order index to be tracked.
          * ``f_index`` causes a Fortran-order index to be tracked.
          * ``multi_index`` causes a multi-index, or a tuple of indices
            with one per iteration dimension, to be tracked.
          * ``common_dtype`` causes all the operands to be converted to
            a common data type, with copying or buffering as necessary.
          * ``copy_if_overlap`` causes the iterator to determine if read
            operands have overlap with write operands, and make temporary
            copies as necessary to avoid overlap. False positives (needless
            copying) are possible in some cases.
          * ``delay_bufalloc`` delays allocation of the buffers until
            a reset() call is made. Allows ``allocate`` operands to
            be initialized before their values are copied into the buffers.
          * ``external_loop`` causes the ``values`` given to be
            one-dimensional arrays with multiple values instead of
            zero-dimensional arrays.
          * ``grow_inner`` allows the ``value`` array sizes to be made
            larger than the buffer size when both ``buffered`` and
            ``external_loop`` is used.
          * ``ranged`` allows the iterator to be restricted to a sub-range
            of the iterindex values.
          * ``refs_ok`` enables iteration of reference types, such as
            object arrays.
          * ``reduce_ok`` enables iteration of ``readwrite`` operands
            which are broadcasted, also known as reduction operands.
          * ``zerosize_ok`` allows `itersize` to be zero.
    """
    # 操作标志列表，每个操作数都有对应的标志列表。至少需要指定其中之一：readonly、readwrite、writeonly。
    # 
    # * readonly 表示该操作数只会被读取。
    # * readwrite 表示该操作数将被读取和写入。
    # * writeonly 表示该操作数只会被写入。
    # * no_broadcast 阻止操作数进行广播。
    # * contig 强制操作数数据是连续的。
    # * aligned 强制操作数数据是对齐的。
    # * nbo 强制操作数数据是本机字节顺序的。
    # * copy 如果需要，允许临时的只读复制。
    # * updateifcopy 如果需要，允许临时的读写复制。
    # * allocate 导致数组在 op 参数中为 None 时被分配。
    # * no_subtype 阻止 allocate 操作数使用子类型。
    # * arraymask 指示该操作数是写入操作数时要使用的掩码。
    # * writemasked 指示只有在选择的 arraymask 操作数为 True 时才会写入的元素。
    # * overlap_assume_elementwise 可用于标记仅按迭代器顺序访问的操作数，以允许在存在 copy_if_overlap 时进行更少保守的复制。
    op_flags : list of list of str, optional

    # 操作数的必需数据类型或数据类型元组。如果启用了复制或缓冲，数据将转换为/从其原始类型。
    op_dtypes : dtype or tuple of dtype(s), optional

    # 控制迭代顺序的选项。
    # 'C' 表示 C 顺序，'F' 表示 Fortran 顺序，
    # 'A' 表示如果所有数组都是 Fortran 连续的，则使用 'F' 顺序，否则使用 'C' 顺序，
    # 'K' 表示尽可能接近数组元素在内存中出现的顺序。
    # 这也会影响 'allocate' 操作数的元素内存顺序，因为它们被分配为与迭代顺序兼容。
    # 默认为 'K'。
    order : {'C', 'F', 'A', 'K'}, optional
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        控制在复制或缓冲时可能发生的数据转换类型。
        将其设置为 'unsafe' 不建议，因为可能会对累积操作产生不利影响。

        * 'no' 表示不进行任何数据类型转换。
        * 'equiv' 表示只允许字节顺序的改变。
        * 'safe' 表示只允许能够保留值的转换。
        * 'same_kind' 表示只允许安全的转换或者同一种类型之间的转换，如 float64 到 float32。
        * 'unsafe' 表示可以进行任何数据类型的转换。
    op_axes : list of list of ints, optional
        如果提供，是每个操作数的整数列表或 None。
        每个操作数的轴列表是一个从迭代器的维度到操作数的维度的映射。
        可以将 -1 放在条目中，使该维度被视为 `newaxis`。
    itershape : tuple of ints, optional
        迭代器的期望形状。这允许在 op_axes 映射的维度不对应于不同操作数的维度时，
        为 ``allocate`` 操作数获取不等于 1 的值。
    buffersize : int, optional
        当启用缓冲时，控制临时缓冲区的大小。设置为 0 表示默认值。

    Attributes
    ----------
    dtypes : tuple of dtype(s)
        `value` 中提供的值的数据类型。如果启用了缓冲，则可能与操作数的数据类型不同。
        仅在迭代器关闭之前有效。
    finished : bool
        迭代操作数是否已完成。
    has_delayed_bufalloc : bool
        如果为 True，则迭代器是使用 ``delay_bufalloc`` 标志创建的，并且尚未对其调用 `reset()` 函数。
    has_index : bool
        如果为 True，则迭代器是使用 ``c_index`` 或 ``f_index`` 标志之一创建的，
        并且可以使用属性 `index` 来检索索引。如果访问并且 `has_index` 为 False，则引发 ValueError。
    has_multi_index : bool
        如果为 True，则迭代器是使用 ``multi_index`` 标志创建的，并且可以使用属性 `multi_index` 来检索它。
    index
        当使用 ``c_index`` 或 ``f_index`` 标志时，此属性提供对索引的访问。
        如果访问并且 ``has_index`` 为 False，则引发 ValueError。
    iterationneedsapi : bool
        迭代是否需要访问 Python API，例如如果其中一个操作数是对象数组。
    iterindex : int
        与迭代顺序匹配的索引。
    itersize : int
        迭代器的大小。
    itviews
        `operands` 在内存中的结构化视图，匹配重新排序和优化的迭代器访问模式。
        仅在迭代器关闭之前有效。
    multi_index
        # 如果使用了 `multi_index` 标志，此属性提供对索引的访问。如果访问了 `multi_index` 但 `has_multi_index` 为 False，则引发 ValueError 异常。
        When the ``multi_index`` flag was used, this property
        provides access to the index. Raises a ValueError if accessed
        accessed and ``has_multi_index`` is False.

    ndim : int
        # 迭代器的维度。
        The dimensions of the iterator.

    nop : int
        # 迭代器操作数的数量。
        The number of iterator operands.

    operands : tuple of operand(s)
        # 要迭代的数组。仅在迭代器关闭之前有效。
        The array(s) to be iterated over. Valid only before the iterator is
        closed.

    shape : tuple of ints
        # 形状元组，迭代器的形状。
        Shape tuple, the shape of the iterator.

    value
        # 当前迭代中 `operands` 的值。通常是一个数组标量的元组，但如果使用了 `external_loop` 标志，则是一个一维数组的元组。
        Value of ``operands`` at current iteration. Normally, this is a
        tuple of array scalars, but if the flag ``external_loop`` is used,
        it is a tuple of one dimensional arrays.

    Notes
    -----
    # `nditer` 取代了 `flatiter`。`nditer` 背后的迭代器实现也通过 NumPy C API 公开。
    `nditer` supersedes `flatiter`.  The iterator implementation behind
    `nditer` is also exposed by the NumPy C API.

    # Python 接口提供两种迭代方式：一种遵循 Python 迭代器协议，另一种模仿 C 风格的 do-while 模式。在大多数情况下，原生的 Python 方法更好，但如果需要迭代器的坐标或索引，请使用 C 风格的模式。
    The Python exposure supplies two iteration interfaces, one which follows
    the Python iterator protocol, and another which mirrors the C-style
    do-while pattern.  The native Python approach is better in most cases, but
    if you need the coordinates or index of an iterator, use the C-style pattern.

    Examples
    --------
    # 下面是如何编写一个 `iter_add` 函数，使用 Python 迭代器协议：
    Here is how we might write an ``iter_add`` function, using the
    Python iterator protocol:

    >>> def iter_add_py(x, y, out=None):
    ...     addop = np.add
    ...     it = np.nditer([x, y, out], [],
    ...                 [['readonly'], ['readonly'], ['writeonly','allocate']])
    ...     with it:
    ...         for (a, b, c) in it:
    ...             addop(a, b, out=c)
    ...         return it.operands[2]

    # 下面是相同的函数，但遵循 C 风格的模式：
    Here is the same function, but following the C-style pattern:

    >>> def iter_add(x, y, out=None):
    ...    addop = np.add
    ...    it = np.nditer([x, y, out], [],
    ...                [['readonly'], ['readonly'], ['writeonly','allocate']])
    ...    with it:
    ...        while not it.finished:
    ...            addop(it[0], it[1], out=it[2])
    ...            it.iternext()
    ...        return it.operands[2]

    # 下面是一个外积函数的示例：
    Here is an example outer product function:

    >>> def outer_it(x, y, out=None):
    ...     mulop = np.multiply
    ...     it = np.nditer([x, y, out], ['external_loop'],
    ...             [['readonly'], ['readonly'], ['writeonly', 'allocate']],
    ...             op_axes=[list(range(x.ndim)) + [-1] * y.ndim,
    ...                      [-1] * x.ndim + list(range(y.ndim)),
    ...                      None])
    ...     with it:
    ...         for (a, b, c) in it:
    ...             mulop(a, b, out=c)
    ...         return it.operands[2]

    >>> a = np.arange(2)+1
    >>> b = np.arange(3)+1
    >>> outer_it(a,b)
    array([[1, 2, 3],
           [2, 4, 6]])

    # 下面是一个操作类似于 "lambda" ufunc 的示例函数：
    Here is an example function which operates like a "lambda" ufunc:

    >>> def luf(lamdaexpr, *args, **kwargs):
    ...    '''luf(lambdaexpr, op1, ..., opn, out=None, order='K', casting='safe', buffersize=0)'''
    ...    nargs = len(args)
    # 准备操作数，将 'out' 参数传递给 op，其余参数传递给 args
    op = (kwargs.get('out',None),) + args
    # 创建一个迭代器对象来遍历 op 的内容
    it = np.nditer(op, ['buffered','external_loop'],
            # 定义每个操作数的访问模式和属性列表
            [['writeonly','allocate','no_broadcast']] +
            [['readonly','nbo','aligned']] * nargs,
            # 指定迭代顺序，默认是 'K'
            order=kwargs.get('order','K'),
            # 指定类型转换方式，默认是 'safe'
            casting=kwargs.get('casting','safe'),
            # 指定缓冲区大小，默认是 0
            buffersize=kwargs.get('buffersize',0))
    # 循环遍历迭代器，直到迭代完成
    while not it.finished:
        # 对当前迭代的数据应用 lambda 表达式，并将结果存入 it[0] 中
        it[0] = lamdaexpr(*it[1:])
        # 移动到下一个迭代
        it.iternext()
    # 返回迭代器的第一个操作数
    return it.operands[0]
# nditer methods

# 添加新文档条目到 numpy._core 的 nditer 函数
add_newdoc('numpy._core', 'nditer', ('copy',
    """
    copy()

    Get a copy of the iterator in its current state.

    Examples
    --------
    >>> x = np.arange(10)
    >>> y = x + 1
    >>> it = np.nditer([x, y])
    >>> next(it)
    (array(0), array(1))
    >>> it2 = it.copy()
    >>> next(it2)
    (array(1), array(2))

    """))

# 添加新文档条目到 numpy._core 的 nditer 函数
add_newdoc('numpy._core', 'nditer', ('operands',
    """
    operands[`Slice`]

    The array(s) to be iterated over. Valid only before the iterator is closed.
    """))

# 添加新文档条目到 numpy._core 的 nditer 函数
add_newdoc('numpy._core', 'nditer', ('debug_print',
    """
    debug_print()

    Print the current state of the `nditer` instance and debug info to stdout.

    """))

# 添加新文档条目到 numpy._core 的 nditer 函数
add_newdoc('numpy._core', 'nditer', ('enable_external_loop',
    """
    enable_external_loop()

    When the "external_loop" was not used during construction, but
    is desired, this modifies the iterator to behave as if the flag
    was specified.

    """))

# 添加新文档条目到 numpy._core 的 nditer 函数
add_newdoc('numpy._core', 'nditer', ('iternext',
    """
    iternext()

    Check whether iterations are left, and perform a single internal iteration
    without returning the result.  Used in the C-style pattern do-while
    pattern.  For an example, see `nditer`.

    Returns
    -------
    iternext : bool
        Whether or not there are iterations left.

    """))

# 添加新文档条目到 numpy._core 的 nditer 函数
add_newdoc('numpy._core', 'nditer', ('remove_axis',
    """
    remove_axis(i, /)

    Removes axis `i` from the iterator. Requires that the flag "multi_index"
    be enabled.

    """))

# 添加新文档条目到 numpy._core 的 nditer 函数
add_newdoc('numpy._core', 'nditer', ('remove_multi_index',
    """
    remove_multi_index()

    When the "multi_index" flag was specified, this removes it, allowing
    the internal iteration structure to be optimized further.

    """))

# 添加新文档条目到 numpy._core 的 nditer 函数
add_newdoc('numpy._core', 'nditer', ('reset',
    """
    reset()

    Reset the iterator to its initial state.

    """))

# 添加新文档条目到 numpy._core 的 nested_iters 函数
add_newdoc('numpy._core', 'nested_iters',
    """
    nested_iters(op, axes, flags=None, op_flags=None, op_dtypes=None, \
    order="K", casting="safe", buffersize=0)

    Create nditers for use in nested loops

    Create a tuple of `nditer` objects which iterate in nested loops over
    different axes of the op argument. The first iterator is used in the
    outermost loop, the last in the innermost loop. Advancing one will change
    the subsequent iterators to point at its new element.

    Parameters
    ----------
    op : ndarray or sequence of array_like
        The array(s) to iterate over.

    axes : list of list of int
        Each item is used as an "op_axes" argument to an nditer

    flags, op_flags, op_dtypes, order, casting, buffersize (optional)
        See `nditer` parameters of the same name

    Returns
    -------
    iters : tuple of nditer
        An nditer for each item in `axes`, outermost first

    See Also
    --------
    nditer

    Examples
    --------

    Basic usage. Note how y is the "flattened" version of
    """)
    # 创建一个3D NumPy数组 `a`，形状为 (2, 3, 2)，包含从0到11的整数
    a = np.arange(12).reshape(2, 3, 2)
    
    # 使用 `np.nested_iters` 函数创建一个嵌套迭代器 `i` 和 `j`，分别迭代轴 [1] 和 [0, 2]
    i, j = np.nested_iters(a, [[1], [0, 2]], flags=["multi_index"])
    
    # 迭代器 `i` 循环，输出每次迭代的多重索引 `i.multi_index`
    for x in i:
        print(i.multi_index)
        # 迭代器 `j` 循环，输出每次迭代的多重索引 `j.multi_index` 和对应的元素 `y`
        for y in j:
            print('', j.multi_index, y)
# 添加新文档到 `numpy._core` 的 `nditer` 函数中，提供了 `close` 方法的文档字符串
add_newdoc('numpy._core', 'nditer', ('close',
    """
    close()

    Resolve all writeback semantics in writeable operands.

    .. versionadded:: 1.15.0

    See Also
    --------

    :ref:`nditer-context-manager`

    """))

###############################################################################
#
# broadcast
#
###############################################################################

# 添加新文档到 `numpy._core` 的 `broadcast` 函数中，提供了函数的概述和用法说明
add_newdoc('numpy._core', 'broadcast',
    """
    Produce an object that mimics broadcasting.

    Parameters
    ----------
    in1, in2, ... : array_like
        Input parameters.

    Returns
    -------
    b : broadcast object
        Broadcast the input parameters against one another, and
        return an object that encapsulates the result.
        Amongst others, it has ``shape`` and ``nd`` properties, and
        may be used as an iterator.

    See Also
    --------
    broadcast_arrays
    broadcast_to
    broadcast_shapes

    Examples
    --------

    Manually adding two vectors, using broadcasting:

    >>> x = np.array([[1], [2], [3]])
    >>> y = np.array([4, 5, 6])
    >>> b = np.broadcast(x, y)

    >>> out = np.empty(b.shape)
    >>> out.flat = [u+v for (u,v) in b]
    >>> out
    array([[5.,  6.,  7.],
           [6.,  7.,  8.],
           [7.,  8.,  9.]])

    Compare against built-in broadcasting:

    >>> x + y
    array([[5, 6, 7],
           [6, 7, 8],
           [7, 8, 9]])

    """)

# 添加新文档到 `numpy._core` 的 `broadcast` 函数中，描述了 `index` 属性的作用和使用示例
add_newdoc('numpy._core', 'broadcast', ('index',
    """
    current index in broadcasted result

    Examples
    --------
    >>> x = np.array([[1], [2], [3]])
    >>> y = np.array([4, 5, 6])
    >>> b = np.broadcast(x, y)
    >>> b.index
    0
    >>> next(b), next(b), next(b)
    ((1, 4), (1, 5), (1, 6))
    >>> b.index
    3

    """))

# 添加新文档到 `numpy._core` 的 `broadcast` 函数中，描述了 `iters` 属性的作用和使用示例
add_newdoc('numpy._core', 'broadcast', ('iters',
    """
    tuple of iterators along ``self``'s "components."

    Returns a tuple of `numpy.flatiter` objects, one for each "component"
    of ``self``.

    See Also
    --------
    numpy.flatiter

    Examples
    --------
    >>> x = np.array([1, 2, 3])
    >>> y = np.array([[4], [5], [6]])
    >>> b = np.broadcast(x, y)
    >>> row, col = b.iters
    >>> next(row), next(col)
    (1, 4)

    """))

# 添加新文档到 `numpy._core` 的 `broadcast` 函数中，描述了 `ndim` 属性的作用和使用示例
add_newdoc('numpy._core', 'broadcast', ('ndim',
    """
    Number of dimensions of broadcasted result. Alias for `nd`.

    .. versionadded:: 1.12.0

    Examples
    --------
    >>> x = np.array([1, 2, 3])
    >>> y = np.array([[4], [5], [6]])
    >>> b = np.broadcast(x, y)
    >>> b.ndim
    2

    """))

# 添加新文档到 `numpy._core` 的 `broadcast` 函数中，描述了 `nd` 属性的作用和使用示例
add_newdoc('numpy._core', 'broadcast', ('nd',
    """
    Number of dimensions of broadcasted result. For code intended for NumPy
    1.12.0 and later the more consistent `ndim` is preferred.

    Examples
    --------
    >>> x = np.array([1, 2, 3])
    >>> y = np.array([[4], [5], [6]])
    >>> b = np.broadcast(x, y)
    >>> b.nd
    2

    """))

# 添加新文档到 `numpy._core` 的 `broadcast` 函数中，描述了 `numiter` 属性的作用
add_newdoc('numpy._core', 'broadcast', ('numiter',
    """
    """))
    Number of iterators possessed by the broadcasted result.  # 描述变量 `numiter` 的含义，表示广播结果的迭代器数量

    Examples  # 示例开始
    --------  # 分隔示例和代码的标记
    >>> x = np.array([1, 2, 3])  # 创建一个NumPy数组 x，包含元素 [1, 2, 3]
    >>> y = np.array([[4], [5], [6]])  # 创建一个NumPy二维数组 y，包含元素 [[4], [5], [6]]
    >>> b = np.broadcast(x, y)  # 创建一个广播对象 b，将数组 x 和 y 进行广播操作
    >>> b.numiter  # 访问广播对象 b 的 numiter 属性，应该得到值 2
    2  # 输出结果应为 2

    """"))  # 示例结束
# 在 numpy._core 模块中为 broadcast 函数添加新的文档信息，描述其 shape 属性的含义和用法示例
add_newdoc('numpy._core', 'broadcast', ('shape',
    """
    Shape of broadcasted result.

    Examples
    --------
    >>> x = np.array([1, 2, 3])
    >>> y = np.array([[4], [5], [6]])
    >>> b = np.broadcast(x, y)
    >>> b.shape
    (3, 3)

    """))

# 在 numpy._core 模块中为 broadcast 函数添加新的文档信息，描述其 size 属性的含义和用法示例
add_newdoc('numpy._core', 'broadcast', ('size',
    """
    Total size of broadcasted result.

    Examples
    --------
    >>> x = np.array([1, 2, 3])
    >>> y = np.array([[4], [5], [6]])
    >>> b = np.broadcast(x, y)
    >>> b.size
    9

    """))

# 在 numpy._core 模块中为 broadcast 函数添加新的文档信息，描述其 reset 方法的含义、参数和用法示例
add_newdoc('numpy._core', 'broadcast', ('reset',
    """
    reset()

    Reset the broadcasted result's iterator(s).

    Parameters
    ----------
    None

    Returns
    -------
    None

    Examples
    --------
    >>> x = np.array([1, 2, 3])
    >>> y = np.array([[4], [5], [6]])
    >>> b = np.broadcast(x, y)
    >>> b.index
    0
    >>> next(b), next(b), next(b)
    ((1, 4), (2, 4), (3, 4))
    >>> b.index
    3
    >>> b.reset()
    >>> b.index
    0

    """))

###############################################################################
#
# numpy functions
#
###############################################################################

# 在 numpy._core.multiarray 模块中为 array 函数添加新的文档信息，描述其含义和用法
add_newdoc('numpy._core.multiarray', 'array',
    """
    array(object, dtype=None, *, copy=True, order='K', subok=False, ndmin=0,
          like=None)

    Create an array.

    Parameters
    ----------
    object : array_like
        An array, any object exposing the array interface, an object whose
        ``__array__`` method returns an array, or any (nested) sequence.
        If object is a scalar, a 0-dimensional array containing object is
        returned.
    dtype : data-type, optional
        The desired data-type for the array. If not given, NumPy will try to use
        a default ``dtype`` that can represent the values (by applying promotion
        rules when necessary.)
    copy : bool, optional
        If ``True`` (default), then the array data is copied. If ``None``,
        a copy will only be made if ``__array__`` returns a copy, if obj is
        a nested sequence, or if a copy is needed to satisfy any of the other
        requirements (``dtype``, ``order``, etc.). Note that any copy of
        the data is shallow, i.e., for arrays with object dtype, the new
        array will point to the same objects. See Examples for `ndarray.copy`.
        For ``False`` it raises a ``ValueError`` if a copy cannot be avoided.
        Default: ``True``.

    """)
    # order 参数用于指定数组的存储顺序，可选值为 {'K', 'A', 'C', 'F'}
    # 如果 object 不是数组，则新创建的数组默认按照 C 顺序（行优先）存储，除非指定了 'F'，则按照 Fortran 顺序（列优先）存储
    # 如果 object 是数组，则具体行为如下表所示
    order : {'K', 'A', 'C', 'F'}, optional
        Specify the memory layout of the array. If object is not an array, the
        newly created array will be in C order (row major) unless 'F' is
        specified, in which case it will be in Fortran order (column major).
        If object is an array the following holds.

        ===== ========= ===================================================
        order  no copy                     copy=True
        ===== ========= ===================================================
        'K'   unchanged F & C order preserved, otherwise most similar order
        'A'   unchanged F order if input is F and not C, otherwise C order
        'C'   C order   C order
        'F'   F order   F order
        ===== ========= ===================================================

        When ``copy=None`` and a copy is made for other reasons, the result is
        the same as if ``copy=True``, with some exceptions for 'A', see the
        Notes section. The default order is 'K'.
    
    # subok 参数用于控制返回的数组是否保留子类，True 表示保留子类，False 表示返回基类数组（默认行为）
    subok : bool, optional
        If True, then sub-classes will be passed-through, otherwise
        the returned array will be forced to be a base-class array (default).
    
    # ndmin 参数用于指定返回数组的最小维度，会在数组形状前面添加必要数量的 1
    ndmin : int, optional
        Specifies the minimum number of dimensions that the resulting
        array should have.  Ones will be prepended to the shape as
        needed to meet this requirement.
    
    # ${ARRAY_FUNCTION_LIKE} 是一个占位符，可能指代某种数组函数或特性，在文档中被标记为从版本 1.20.0 开始添加的内容
    ${ARRAY_FUNCTION_LIKE}

        .. versionadded:: 1.20.0

    # 返回值为一个 ndarray 数组对象，满足上述指定的要求
    Returns
    -------
    out : ndarray
        An array object satisfying the specified requirements.

    # 下面的部分列出了一些与 np.array 相关的函数，可以参考
    See Also
    --------
    empty_like : Return an empty array with shape and type of input.
    ones_like : Return an array of ones with shape and type of input.
    zeros_like : Return an array of zeros with shape and type of input.
    full_like : Return a new array with shape of input filled with value.
    empty : Return a new uninitialized array.
    ones : Return a new array setting values to one.
    zeros : Return a new array setting values to zero.
    full : Return a new array of given shape filled with value.
    copy: Return an array copy of the given object.

    # 提供了关于函数行为的一些额外说明
    Notes
    -----
    When order is 'A' and ``object`` is an array in neither 'C' nor 'F' order,
    and a copy is forced by a change in dtype, then the order of the result is
    not necessarily 'C' as expected. This is likely a bug.

    # 给出了一些函数使用的示例
    Examples
    --------
    >>> np.array([1, 2, 3])
    array([1, 2, 3])

    Upcasting:

    >>> np.array([1, 2, 3.0])
    array([ 1.,  2.,  3.])

    More than one dimension:

    >>> np.array([[1, 2], [3, 4]])
    array([[1, 2],
           [3, 4]])

    Minimum dimensions 2:

    >>> np.array([1, 2, 3], ndmin=2)
    array([[1, 2, 3]])

    Type provided:

    >>> np.array([1, 2, 3], dtype=complex)
    array([ 1.+0.j,  2.+0.j,  3.+0.j])

    Data-type consisting of more than one element:

    >>> x = np.array([(1,2),(3,4)],dtype=[('a','<i4'),('b','<i4')])
    >>> x['a']
    array([1, 3])
    Creating an array from sub-classes:

    >>> np.array(np.asmatrix('1 2; 3 4'))
    array([[1, 2],
           [3, 4]])

    >>> np.array(np.asmatrix('1 2; 3 4'), subok=True)
    matrix([[1, 2],
            [3, 4]])

    """.replace(
        "${ARRAY_FUNCTION_LIKE}",
        array_function_like_doc,
    ))


注释：


# 使用子类创建数组：

# 使用 np.asmatrix 将字符串 '1 2; 3 4' 转换为矩阵，然后用 np.array 转换为数组
>>> np.array(np.asmatrix('1 2; 3 4'))
array([[1, 2],
       [3, 4]])

# 使用 subok=True 参数，将 np.asmatrix 创建的矩阵作为子类，用 np.array 转换为矩阵
>>> np.array(np.asmatrix('1 2; 3 4'), subok=True)
matrix([[1, 2],
        [3, 4]])

""".replace(
    "${ARRAY_FUNCTION_LIKE}",
    array_function_like_doc,
))


这段代码展示了如何使用 `np.array` 函数从子类（如 `np.asmatrix` 创建的矩阵）创建数组，并演示了不同参数下的输出效果。
# 将文档添加到指定的 NumPy 多维数组核心模块和函数名称中
add_newdoc('numpy._core.multiarray', 'asarray',
    """
    asarray(a, dtype=None, order=None, *, device=None, copy=None, like=None)

    Convert the input to an array.

    Parameters
    ----------
    a : array_like
        Input data, in any form that can be converted to an array.  This
        includes lists, lists of tuples, tuples, tuples of tuples, tuples
        of lists and ndarrays.
    dtype : data-type, optional
        By default, the data-type is inferred from the input data.
    order : {'C', 'F', 'A', 'K'}, optional
        Memory layout.  'A' and 'K' depend on the order of input array a.
        'C' row-major (C-style),
        'F' column-major (Fortran-style) memory representation.
        'A' (any) means 'F' if `a` is Fortran contiguous, 'C' otherwise
        'K' (keep) preserve input order
        Defaults to 'K'.
    device : str, optional
        The device on which to place the created array. Default: None.
        For Array-API interoperability only, so must be ``"cpu"`` if passed.

        .. versionadded:: 2.0.0
    copy : bool, optional
        If ``True``, then the object is copied. If ``None`` then the object is
        copied only if needed, i.e. if ``__array__`` returns a copy, if obj
        is a nested sequence, or if a copy is needed to satisfy any of
        the other requirements (``dtype``, ``order``, etc.).
        For ``False`` it raises a ``ValueError`` if a copy cannot be avoided.
        Default: ``None``.

        .. versionadded:: 2.0.0
    ${ARRAY_FUNCTION_LIKE}

        .. versionadded:: 1.20.0

    Returns
    -------
    out : ndarray
        Array interpretation of ``a``.  No copy is performed if the input
        is already an ndarray with matching dtype and order.  If ``a`` is a
        subclass of ndarray, a base class ndarray is returned.

    See Also
    --------
    asanyarray : Similar function which passes through subclasses.
    ascontiguousarray : Convert input to a contiguous array.
    asfortranarray : Convert input to an ndarray with column-major
                     memory order.
    asarray_chkfinite : Similar function which checks input for NaNs and Infs.
    fromiter : Create an array from an iterator.
    fromfunction : Construct an array by executing a function on grid
                   positions.

    Examples
    --------
    Convert a list into an array:

    >>> a = [1, 2]
    >>> np.asarray(a)
    array([1, 2])

    Existing arrays are not copied:

    >>> a = np.array([1, 2])
    >>> np.asarray(a) is a
    True

    If `dtype` is set, array is copied only if dtype does not match:

    >>> a = np.array([1, 2], dtype=np.float32)
    >>> np.shares_memory(np.asarray(a, dtype=np.float32), a)
    True
    >>> np.shares_memory(np.asarray(a, dtype=np.float64), a)
    False

    Contrary to `asanyarray`, ndarray subclasses are not passed through:

    >>> issubclass(np.recarray, np.ndarray)
    True
    >>> a = np.array([(1., 2), (3., 4)], dtype='f4,i4').view(np.recarray)
    """
    # 使用 NumPy 的 asarray 函数将变量 a 转换为 NumPy 数组，并检查返回的数组是否与原始变量 a 是同一个对象
    >>> np.asarray(a) is a
    # 返回 False，表明 np.asarray(a) 创建了一个新的 NumPy 数组对象，而不是原始变量 a 的引用

    # 使用 NumPy 的 asanyarray 函数将变量 a 转换为 NumPy 的任意数组，然后检查返回的数组是否与原始变量 a 是同一个对象
    >>> np.asanyarray(a) is a
    # 返回 True，表明 np.asanyarray(a) 返回的是原始变量 a 的引用，没有创建新的对象

    """.replace(
        "${ARRAY_FUNCTION_LIKE}",
        array_function_like_doc,
    ))
# 将新文档添加到指定的numpy._core.multiarray模块下的asanyarray函数
add_newdoc('numpy._core.multiarray', 'asanyarray',
    """
    asanyarray(a, dtype=None, order=None, *, like=None)

    Convert the input to an ndarray, but pass ndarray subclasses through.

    Parameters
    ----------
    a : array_like
        Input data, in any form that can be converted to an array.  This
        includes scalars, lists, lists of tuples, tuples, tuples of tuples,
        tuples of lists, and ndarrays.
    dtype : data-type, optional
        By default, the data-type is inferred from the input data.
    order : {'C', 'F', 'A', 'K'}, optional
        Memory layout.  'A' and 'K' depend on the order of input array a.
        'C' row-major (C-style),
        'F' column-major (Fortran-style) memory representation.
        'A' (any) means 'F' if `a` is Fortran contiguous, 'C' otherwise
        'K' (keep) preserve input order
        Defaults to 'C'.
    device : str, optional
        The device on which to place the created array. Default: None.
        For Array-API interoperability only, so must be ``"cpu"`` if passed.

        .. versionadded:: 2.1.0

    copy : bool, optional
        If ``True``, then the object is copied. If ``None`` then the object is
        copied only if needed, i.e. if ``__array__`` returns a copy, if obj
        is a nested sequence, or if a copy is needed to satisfy any of
        the other requirements (``dtype``, ``order``, etc.).
        For ``False`` it raises a ``ValueError`` if a copy cannot be avoided.
        Default: ``None``.

        .. versionadded:: 2.1.0

    array_function_like_doc：str
        Document string to insert for ${ARRAY_FUNCTION_LIKE}

        .. versionadded:: 1.20.0

    Returns
    -------
    out : ndarray or an ndarray subclass
        Array interpretation of `a`.  If `a` is an ndarray or a subclass
        of ndarray, it is returned as-is and no copy is performed.

    See Also
    --------
    asarray : Similar function which always returns ndarrays.
    ascontiguousarray : Convert input to a contiguous array.
    asfortranarray : Convert input to an ndarray with column-major
                     memory order.
    asarray_chkfinite : Similar function which checks input for NaNs and
                        Infs.
    fromiter : Create an array from an iterator.
    fromfunction : Construct an array by executing a function on grid
                   positions.

    Examples
    --------
    Convert a list into an array:

    >>> a = [1, 2]
    >>> np.asanyarray(a)
    array([1, 2])

    Instances of `ndarray` subclasses are passed through as-is:

    >>> a = np.array([(1., 2), (3., 4)], dtype='f4,i4').view(np.recarray)
    >>> np.asanyarray(a) is a
    True

    """.replace(
        "${ARRAY_FUNCTION_LIKE}",
        array_function_like_doc,
    ))
    # dtype : str or dtype object, optional
    # 返回数组的数据类型，可以是字符串或dtype对象，可选项
    ${ARRAY_FUNCTION_LIKE}
        # 添加于1.20.0版本
        # 返回
        # -------
        # ndarray
        # 与`a`具有相同形状和内容的连续数组，如果指定了dtype，则为指定类型
        # 另见
        # --------
        # asfortranarray：将输入转换为列主内存顺序的ndarray。
        # require：返回满足要求的ndarray。
        # ndarray.flags：有关数组的内存布局的信息。

    返回
    -------
    out：ndarray
        连续数组，与`a`具有相同的形状和内容，如果指定了dtype则是指定类型。

    另见
    --------
    asfortranarray：将输入转换为具有列主内存顺序的ndarray。
                     memory order.
    require：返回满足要求的ndarray。
    ndarray.flags：有关数组的内存布局的信息。

    示例
    --------
    从Fortran连续数组开始：

    >>> x = np.ones((2, 3), order='F')
    >>> x.flags['F_CONTIGUOUS']
    True

    调用``ascontiguousarray``生成一个C连续的副本：

    >>> y = np.ascontiguousarray(x)
    >>> y.flags['C_CONTIGUOUS']
    True
    >>> np.may_share_memory(x, y)
    False

    现在，从C连续数组开始：

    >>> x = np.ones((2, 3), order='C')
    >>> x.flags['C_CONTIGUOUS']
    True

    然后，调用``ascontiguousarray``返回相同的对象：

    >>> y = np.ascontiguousarray(x)
    >>> x is y
    True

    注意：此函数返回至少具有一个维度（1-d）的数组，因此它不会保留0-d数组。
    """
# 向numpy._core.multiarray模块添加新文档条目'asfortranarray'
add_newdoc('numpy._core.multiarray', 'asfortranarray',
    """
    asfortranarray(a, dtype=None, *, like=None)

    Return an array (ndim >= 1) laid out in Fortran order in memory.

    Parameters
    ----------
    a : array_like
        Input array.
    dtype : str or dtype object, optional
        By default, the data-type is inferred from the input data.
    ${ARRAY_FUNCTION_LIKE}  # 插入数组函数相似的文档部分

        .. versionadded:: 1.20.0

    Returns
    -------
    out : ndarray
        The input `a` in Fortran, or column-major, order.

    See Also
    --------
    ascontiguousarray : Convert input to a contiguous (C order) array.
    asanyarray : Convert input to an ndarray with either row or
        column-major memory order.
    require : Return an ndarray that satisfies requirements.
    ndarray.flags : Information about the memory layout of the array.

    Examples
    --------
    Starting with a C-contiguous array:

    >>> x = np.ones((2, 3), order='C')
    >>> x.flags['C_CONTIGUOUS']
    True

    Calling ``asfortranarray`` makes a Fortran-contiguous copy:

    >>> y = np.asfortranarray(x)
    >>> y.flags['F_CONTIGUOUS']
    True
    >>> np.may_share_memory(x, y)
    False

    Now, starting with a Fortran-contiguous array:

    >>> x = np.ones((2, 3), order='F')
    >>> x.flags['F_CONTIGUOUS']
    True

    Then, calling ``asfortranarray`` returns the same object:

    >>> y = np.asfortranarray(x)
    >>> x is y
    True

    Note: This function returns an array with at least one-dimension (1-d)
    so it will not preserve 0-d arrays.

    """.replace(
        "${ARRAY_FUNCTION_LIKE}",
        array_function_like_doc,
    ))

# 向numpy._core.multiarray模块添加新文档条目'empty'
add_newdoc('numpy._core.multiarray', 'empty',
    """
    empty(shape, dtype=float, order='C', *, device=None, like=None)

    Return a new array of given shape and type, without initializing entries.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the empty array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        Desired output data-type for the array, e.g, `numpy.int8`. Default is
        `numpy.float64`.
    order : {'C', 'F'}, optional, default: 'C'
        Whether to store multi-dimensional data in row-major
        (C-style) or column-major (Fortran-style) order in
        memory.
    device : str, optional
        The device on which to place the created array. Default: None.
        For Array-API interoperability only, so must be ``"cpu"`` if passed.

        .. versionadded:: 2.0.0
    ${ARRAY_FUNCTION_LIKE}  # 插入数组函数相似的文档部分

        .. versionadded:: 1.20.0

    Returns
    -------
    out : ndarray
        Array of uninitialized (arbitrary) data of the given shape, dtype, and
        order.  Object arrays will be initialized to None.

    See Also
    --------
    empty_like : Return an empty array with shape and type of input.
    ones : Return a new array setting values to one.
    zeros : Return a new array setting values to zero.

    """
    # 返回一个给定形状的新数组，数组的值未初始化
    # 
    # 注意：
    # 不像其他数组创建函数（例如 `zeros`, `ones`, `full`），`empty` 不会初始化数组的值，
    # 因此可能会稍微快一些。然而，新分配的数组中存储的值是任意的。为了可重现的行为，
    # 在读取数组之前，请确保设置数组的每个元素。
    # 
    # 示例：
    # >>> np.empty([2, 2])
    # array([[ -9.74499359e+001,   6.69583040e-309],
    #        [  2.13182611e-314,   3.06959433e-309]])         # 未初始化
    # 
    # >>> np.empty([2, 2], dtype=int)
    # array([[-1073741821, -1067949133],
    #        [  496041986,    19249760]])                     # 未初始化
    """.replace(
        "${ARRAY_FUNCTION_LIKE}",
        array_function_like_doc,
    ))
# 添加新的文档字符串到numpy._core.multiarray模块中的scalar函数
add_newdoc('numpy._core.multiarray', 'scalar',
    """
    scalar(dtype, obj)

    Return a new scalar array of the given type initialized with obj.

    This function is meant mainly for pickle support. `dtype` must be a
    valid data-type descriptor. If `dtype` corresponds to an object
    descriptor, then `obj` can be any object, otherwise `obj` must be a
    string. If `obj` is not given, it will be interpreted as None for object
    type and as zeros for all other types.

    """)

# 添加新的文档字符串到numpy._core.multiarray模块中的zeros函数
add_newdoc('numpy._core.multiarray', 'zeros',
    """
    zeros(shape, dtype=float, order='C', *, like=None)

    Return a new array of given shape and type, filled with zeros.

    Parameters
    ----------
    shape : int or tuple of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        The desired data-type for the array, e.g., `numpy.int8`.  Default is
        `numpy.float64`.
    order : {'C', 'F'}, optional, default: 'C'
        Whether to store multi-dimensional data in row-major
        (C-style) or column-major (Fortran-style) order in
        memory.
    ${ARRAY_FUNCTION_LIKE}

        .. versionadded:: 1.20.0

    Returns
    -------
    out : ndarray
        Array of zeros with the given shape, dtype, and order.

    See Also
    --------
    zeros_like : Return an array of zeros with shape and type of input.
    empty : Return a new uninitialized array.
    ones : Return a new array setting values to one.
    full : Return a new array of given shape filled with value.

    Examples
    --------
    >>> np.zeros(5)
    array([ 0.,  0.,  0.,  0.,  0.])

    >>> np.zeros((5,), dtype=int)
    array([0, 0, 0, 0, 0])

    >>> np.zeros((2, 1))
    array([[ 0.],
           [ 0.]])

    >>> s = (2,2)
    >>> np.zeros(s)
    array([[ 0.,  0.],
           [ 0.,  0.]])

    >>> np.zeros((2,), dtype=[('x', 'i4'), ('y', 'i4')]) # custom dtype
    array([(0, 0), (0, 0)],
          dtype=[('x', '<i4'), ('y', '<i4')])

    """.replace(
        "${ARRAY_FUNCTION_LIKE}",
        array_function_like_doc,
    ))

# 添加新的文档字符串到numpy._core.multiarray模块中的set_typeDict函数
add_newdoc('numpy._core.multiarray', 'set_typeDict',
    """set_typeDict(dict)

    Set the internal dictionary that can look up an array type using a
    registered code.

    """)

# 添加新的文档字符串到numpy._core.multiarray模块中的fromstring函数
add_newdoc('numpy._core.multiarray', 'fromstring',
    """
    fromstring(string, dtype=float, count=-1, *, sep, like=None)

    A new 1-D array initialized from text data in a string.

    Parameters
    ----------
    string : str
        A string containing the data.
    dtype : data-type, optional
        The data type of the array; default: float.  For binary input data,
        the data must be in exactly this format. Most builtin numeric types are
        supported and extension types may be supported.

        .. versionadded:: 1.18.0
            Complex dtypes.
    # count : int, optional
    #     从数据中读取这么多个 `dtype` 类型的元素。如果为负数（默认），则从数据的长度确定读取的数量。

    # sep : str, optional
    #     数据中分隔数字的字符串；元素之间的额外空白也会被忽略。
    # 
    #     .. deprecated:: 1.14
    #         传递 ``sep=''``，即默认情况，已弃用，因为会触发此函数的弃用的二进制模式。
    #         此模式将 `string` 解释为二进制字节，而不是包含十进制数字的 ASCII 文本，
    #         这个操作更好地写成 ``frombuffer(string, dtype, count)``。
    #         如果 `string` 包含 Unicode 文本，则 `fromstring` 的二进制模式将首先使用 utf-8 编码它，
    #         这不会产生合理的结果。

    ${ARRAY_FUNCTION_LIKE}
        .. versionadded:: 1.20.0

    Returns
    -------
    arr : ndarray
        构造的数组。

    Raises
    ------
    ValueError
        如果字符串大小不正确，无法满足请求的 `dtype` 和 `count`。

    See Also
    --------
    frombuffer, fromfile, fromiter

    Examples
    --------
    >>> np.fromstring('1 2', dtype=int, sep=' ')
    array([1, 2])
    >>> np.fromstring('1, 2', dtype=int, sep=',')
    array([1, 2])
# 添加新文档到numpy._core.multiarray模块中，函数名为compare_chararrays
add_newdoc('numpy._core.multiarray', 'compare_chararrays',
    """
    compare_chararrays(a1, a2, cmp, rstrip)

    Performs element-wise comparison of two string arrays using the
    comparison operator specified by `cmp`.

    Parameters
    ----------
    a1, a2 : array_like
        Arrays to be compared.
    cmp : {"<", "<=", "==", ">=", ">", "!="}
        Type of comparison.
    rstrip : Boolean
        If True, the spaces at the end of Strings are removed before the comparison.

    Returns
    -------
    out : ndarray
        The output array of type Boolean with the same shape as a and b.

    Raises
    ------
    ValueError
        If `cmp` is not valid.
    TypeError
        If at least one of `a` or `b` is a non-string array

    Examples
    --------
    >>> a = np.array(["a", "b", "cde"])
    >>> b = np.array(["a", "a", "dec"])
    >>> np.char.compare_chararrays(a, b, ">", True)
    array([False,  True, False])

    """)

# 添加新文档到numpy._core.multiarray模块中，函数名为fromiter
add_newdoc('numpy._core.multiarray', 'fromiter',
    """
    fromiter(iter, dtype, count=-1, *, like=None)

    Create a new 1-dimensional array from an iterable object.

    Parameters
    ----------
    iter : iterable object
        An iterable object providing data for the array.
    dtype : data-type
        The data-type of the returned array.

        .. versionchanged:: 1.23
            Object and subarray dtypes are now supported (note that the final
            result is not 1-D for a subarray dtype).

    count : int, optional
        The number of items to read from *iterable*.  The default is -1,
        which means all data is read.
    like : ndarray, optional
        ${ARRAY_FUNCTION_LIKE}

        .. versionadded:: 1.20.0

    Returns
    -------
    out : ndarray
        The output array.

    Notes
    -----
    Specify `count` to improve performance.  It allows ``fromiter`` to
    pre-allocate the output array, instead of resizing it on demand.

    Examples
    --------
    >>> iterable = (x*x for x in range(5))
    >>> np.fromiter(iterable, float)
    array([  0.,   1.,   4.,   9.,  16.])

    A carefully constructed subarray dtype will lead to higher dimensional
    results:

    >>> iterable = ((x+1, x+2) for x in range(5))
    >>> np.fromiter(iterable, dtype=np.dtype((int, 2)))
    array([[1, 2],
           [2, 3],
           [3, 4],
           [4, 5],
           [5, 6]])

    """.replace(
        "${ARRAY_FUNCTION_LIKE}",
        array_function_like_doc,
    ))

# 添加新文档到numpy._core.multiarray模块中，函数名为fromfile
add_newdoc('numpy._core.multiarray', 'fromfile',
    """
    fromfile(file, dtype=float, count=-1, sep='', offset=0, *, like=None)

    Construct an array from data in a text or binary file.

    A highly efficient way of reading binary data with a known data-type,
    as well as parsing simply formatted text files.  Data written using the
    `tofile` method can be read using this function.

    Parameters
    ----------
    file : file-like object
        The file object or file path from which to read the data.
    dtype : data-type, optional
        The data-type of the returned array. Defaults to float.
    count : int, optional
        The number of items to read. Default is -1, which means all data is read.
    sep : str, optional
        Separator between items, applicable only for text files. Default is ''.
    offset : int, optional
        The starting position from which to read the data. Default is 0.
    like : ndarray, optional
        ${ARRAY_FUNCTION_LIKE}

    Returns
    -------
    out : ndarray
        The constructed array.

    """.replace(
        "${ARRAY_FUNCTION_LIKE}",
        array_function_like_doc,
    ))
    # file : file or str or Path
    #     Open file object or filename.
    #     文件参数可以是文件对象、文件名字符串或路径对象。

    # dtype : data-type
    #     Data type of the returned array.
    #     返回数组的数据类型。
    #     对于二进制文件，该参数确定文件中项的大小和字节顺序。
    #     大多数内置的数值类型都被支持，可能还支持扩展类型。

    #     .. versionadded:: 1.18.0
    #         复杂数据类型现在也被支持。

    # count : int
    #     Number of items to read. ``-1`` means all items (i.e., the complete
    #     file).
    #     要读取的项的数量。``-1`` 表示读取所有项（即完整文件）。

    # sep : str
    #     Separator between items if file is a text file.
    #     如果文件是文本文件，则项之间的分隔符。
    #     空字符串（""）分隔符表示文件应被视为二进制文件。
    #     分隔符中的空格（" "）匹配零个或多个空白字符。
    #     分隔符只包含空格时，至少要匹配一个空白字符。

    # offset : int
    #     The offset (in bytes) from the file's current position. Defaults to 0.
    #     Only permitted for binary files.
    #     文件当前位置的偏移量（以字节为单位）。默认为 0。
    #     仅允许用于二进制文件。

    #     .. versionadded:: 1.17.0

    # See also
    # --------
    # load, save
    # ndarray.tofile
    # loadtxt : More flexible way of loading data from a text file.
    # 参见：
    # load, save
    # ndarray.tofile
    # loadtxt：从文本文件加载数据的更灵活的方法。

    # Notes
    # -----
    # Do not rely on the combination of `tofile` and `fromfile` for
    # data storage, as the binary files generated are not platform
    # independent.  In particular, no byte-order or data-type information is
    # saved.  Data can be stored in the platform independent ``.npy`` format
    # using `save` and `load` instead.
    # 注意：
    # 不要依赖 `tofile` 和 `fromfile` 的组合来进行数据存储，
    # 因为生成的二进制文件不是平台独立的。
    # 特别是，没有保存字节顺序或数据类型信息。
    # 可以使用 `save` 和 `load` 来以平台独立的 `.npy` 格式存储数据。

    # Examples
    # --------
    # Construct an ndarray:
    # 构建一个 ndarray：

    # >>> dt = np.dtype([('time', [('min', np.int64), ('sec', np.int64)]),
    # ...                ('temp', float)])
    # >>> x = np.zeros((1,), dtype=dt)
    # >>> x['time']['min'] = 10; x['temp'] = 98.25
    # >>> x
    # array([((10, 0), 98.25)],
    #       dtype=[('time', [('min', '<i8'), ('sec', '<i8')]), ('temp', '<f8')])

    # Save the raw data to disk:
    # 将原始数据保存到磁盘：

    # >>> import tempfile
    # >>> fname = tempfile.mkstemp()[1]
    # >>> x.tofile(fname)

    # Read the raw data from disk:
    # 从磁盘读取原始数据：

    # >>> np.fromfile(fname, dtype=dt)
    # array([((10, 0), 98.25)],
    #       dtype=[('time', [('min', '<i8'), ('sec', '<i8')]), ('temp', '<f8')])

    # The recommended way to store and load data:
    # 推荐的存储和加载数据的方式：

    # >>> np.save(fname, x)
    # >>> np.load(fname + '.npy')
    # array([((10, 0), 98.25)],
    #       dtype=[('time', [('min', '<i8'), ('sec', '<i8')]), ('temp', '<f8')])
# 添加新的文档字符串到指定模块和函数，描述函数frombuffer的作用和用法
add_newdoc('numpy._core.multiarray', 'frombuffer',
    """
    frombuffer(buffer, dtype=float, count=-1, offset=0, *, like=None)

    Interpret a buffer as a 1-dimensional array.

    Parameters
    ----------
    buffer : buffer_like
        An object that exposes the buffer interface.
    dtype : data-type, optional
        Data-type of the returned array; default: float.
    count : int, optional
        Number of items to read. ``-1`` means all data in the buffer.
    offset : int, optional
        Start reading the buffer from this offset (in bytes); default: 0.
    like : object, optional
        If specified, interpret buffer as being similar to this object.
        
        .. versionadded:: 1.20.0

    Returns
    -------
    out : ndarray
        The constructed 1-dimensional array.

    See also
    --------
    ndarray.tobytes
        Inverse of this operation, construct Python bytes from the raw data
        bytes in the array.

    Notes
    -----
    If the buffer has data that is not in machine byte-order, this should
    be specified as part of the data-type, e.g.::

      >>> dt = np.dtype(int)
      >>> dt = dt.newbyteorder('>')
      >>> np.frombuffer(buf, dtype=dt) # doctest: +SKIP

    The data of the resulting array will not be byteswapped, but will be
    interpreted correctly.

    This function creates a view into the original object.  This should be safe
    in general, but it may make sense to copy the result when the original
    object is mutable or untrusted.

    Examples
    --------
    >>> s = b'hello world'
    >>> np.frombuffer(s, dtype='S1', count=5, offset=6)
    array([b'w', b'o', b'r', b'l', b'd'], dtype='|S1')

    >>> np.frombuffer(b'\\x01\\x02', dtype=np.uint8)
    array([1, 2], dtype=uint8)
    >>> np.frombuffer(b'\\x01\\x02\\x03\\x04\\x05', dtype=np.uint8, count=3)
    array([1, 2, 3], dtype=uint8)

    """.replace(
        "${ARRAY_FUNCTION_LIKE}",
        array_function_like_doc,
    ))

# 添加新的文档字符串到指定模块和函数，描述函数from_dlpack的作用和用法
add_newdoc('numpy._core.multiarray', 'from_dlpack',
    """
    from_dlpack(x, /)

    Create a NumPy array from an object implementing the ``__dlpack__``
    protocol. Generally, the returned NumPy array is a read-only view
    of the input object. See [1]_ and [2]_ for more details.

    Parameters
    ----------
    x : object
        A Python object that implements the ``__dlpack__`` and
        ``__dlpack_device__`` methods.

    Returns
    -------
    out : ndarray
        The NumPy array constructed from the input object.

    References
    ----------
    .. [1] Array API documentation,
       https://data-apis.org/array-api/latest/design_topics/data_interchange.html#syntax-for-data-interchange-with-dlpack

    .. [2] Python specification for DLPack,
       https://dmlc.github.io/dlpack/latest/python_spec.html

    Examples
    --------
    >>> import torch  # doctest: +SKIP
    >>> x = torch.arange(10)  # doctest: +SKIP
    >>> # create a view of the torch tensor "x" in NumPy
    >>> y = np.from_dlpack(x)  # doctest: +SKIP
    """)

# 添加新的文档字符串到指定模块和函数，描述函数correlate的作用和用法
add_newdoc('numpy._core.multiarray', 'correlate',
    """cross_correlate(a,v, mode=0)""")

# 添加新的文档字符串到指定模块和函数，描述函数arange的作用和用法
add_newdoc('numpy._core.multiarray', 'arange',
    """
    Return evenly spaced values within a given interval.

    """)
    arange([start,] stop[, step,], dtype=None, *, device=None, like=None)
    # 定义函数 arange，用于生成指定区间内均匀分布的数值数组。

    Return evenly spaced values within a given interval.
    # 返回给定区间内均匀分布的数值。

    ``arange`` can be called with a varying number of positional arguments:
    # ``arange`` 函数可以使用不同数量的位置参数进行调用：

    * ``arange(stop)``: Values are generated within the half-open interval
      ``[0, stop)`` (in other words, the interval including `start` but
      excluding `stop`).
    # 当只有一个参数 stop 时，生成从 0 开始到 stop-1 的数值数组。

    * ``arange(start, stop)``: Values are generated within the half-open
      interval ``[start, stop)``.
    # 当有两个参数 start 和 stop 时，生成从 start 开始到 stop-1 的数值数组。

    * ``arange(start, stop, step)`` Values are generated within the half-open
      interval ``[start, stop)``, with spacing between values given by
      ``step``.
    # 当有三个参数 start、stop 和 step 时，按照指定的步长 step 在区间 [start, stop) 内生成数值数组。

    For integer arguments the function is roughly equivalent to the Python
    built-in :py:class:`range`, but returns an ndarray rather than a ``range``
    instance.
    # 对于整数参数，函数大致等同于 Python 内置的 range 函数，但返回一个 ndarray 而不是一个 ``range`` 实例。

    When using a non-integer step, such as 0.1, it is often better to use
    `numpy.linspace`.
    # 当使用非整数步长（如 0.1）时，建议使用 `numpy.linspace`。

    See the Warning sections below for more information.
    # 查看下面的警告部分获取更多信息。

    Parameters
    ----------
    start : integer or real, optional
        Start of interval.  The interval includes this value.  The default
        start value is 0.
    # start：整数或实数，可选参数，区间的起始值，包含在区间内。默认起始值为 0。

    stop : integer or real
        End of interval.  The interval does not include this value, except
        in some cases where `step` is not an integer and floating point
        round-off affects the length of `out`.
    # stop：整数或实数，区间的结束值。区间不包含此值，但在 `step` 不是整数且浮点数舍入影响 `out` 长度的情况下除外。

    step : integer or real, optional
        Spacing between values.  For any output `out`, this is the distance
        between two adjacent values, ``out[i+1] - out[i]``.  The default
        step size is 1.  If `step` is specified as a position argument,
        `start` must also be given.
    # step：整数或实数，可选参数，数值之间的间隔。对于输出的任何 `out` 数组，这是两个相邻值之间的距离 ``out[i+1] - out[i]``。默认步长为 1。如果将 `step` 指定为位置参数，则必须同时给出 `start`。

    dtype : dtype, optional
        The type of the output array.  If `dtype` is not given, infer the data
        type from the other input arguments.
    # dtype：dtype 类型，可选参数，输出数组的数据类型。如果未提供 `dtype`，则从其他输入参数推断数据类型。

    device : str, optional
        The device on which to place the created array. Default: None.
        For Array-API interoperability only, so must be ``"cpu"`` if passed.
    # device：字符串，可选参数，要放置创建的数组的设备。默认为 None。仅用于 Array-API 互操作性，因此如果传递了此参数，则必须为 ``"cpu"``。

        .. versionadded:: 2.0.0

    like : object, optional
        ${ARRAY_FUNCTION_LIKE}
    # like：对象，可选参数，类似于数组函数。

        .. versionadded:: 1.20.0

    Returns
    -------
    arange : ndarray
        Array of evenly spaced values.
        # 返回：ndarray，均匀间隔数值的数组。

        For floating point arguments, the length of the result is
        ``ceil((stop - start)/step)``.  Because of floating point overflow,
        this rule may result in the last element of `out` being greater
        than `stop`.
        # 对于浮点数参数，结果数组的长度为 ``ceil((stop - start)/step)``。由于浮点数溢出，此规则可能导致 `out` 的最后一个元素大于 `stop`。

    Warnings
    --------
    The length of the output might not be numerically stable.
    # 输出数组的长度可能不是数值稳定的。

    Another stability issue is due to the internal implementation of
    `numpy.arange`.
    # 另一个稳定性问题源于 `numpy.arange` 的内部实现。

    The actual step value used to populate the array is
    ``dtype(start + step) - dtype(start)`` and not `step`. Precision loss
    can occur here, due to casting or due to using floating points when
    `start` is much larger than `step`. This can lead to unexpected
    # 用于填充数组的实际步长值是 ``dtype(start + step) - dtype(start)`` 而不是 `step`。由于类型转换或在 `start` 远大于 `step` 时使用浮点数，可能会发生精度损失。这可能导致意外结果。
    behaviour. For example::

      >>> np.arange(0, 5, 0.5, dtype=int)
      array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
      >>> np.arange(-3, 3, 0.5, dtype=int)
      array([-3, -2, -1,  0,  1,  2,  3,  4,  5,  6,  7,  8])

    In such cases, the use of `numpy.linspace` should be preferred.

    The built-in :py:class:`range` generates :std:doc:`Python built-in integers
    that have arbitrary size <python:c-api/long>`, while `numpy.arange`
    produces `numpy.int32` or `numpy.int64` numbers. This may result in
    incorrect results for large integer values::

      >>> power = 40
      >>> modulo = 10000
      >>> x1 = [(n ** power) % modulo for n in range(8)]
      >>> x2 = [(n ** power) % modulo for n in np.arange(8)]
      >>> print(x1)
      [0, 1, 7776, 8801, 6176, 625, 6576, 4001]  # correct
      >>> print(x2)
      [0, 1, 7776, 7185, 0, 5969, 4816, 3361]  # incorrect

    See Also
    --------
    numpy.linspace : Evenly spaced numbers with careful handling of endpoints.
    numpy.ogrid: Arrays of evenly spaced numbers in N-dimensions.
    numpy.mgrid: Grid-shaped arrays of evenly spaced numbers in N-dimensions.
    :ref:`how-to-partition`

    Examples
    --------
    >>> np.arange(3)
    array([0, 1, 2])
    >>> np.arange(3.0)
    array([ 0.,  1.,  2.])
    >>> np.arange(3,7)
    array([3, 4, 5, 6])
    >>> np.arange(3,7,2)
    array([3, 5])

    """.replace(
        "${ARRAY_FUNCTION_LIKE}",
        array_function_like_doc,
    ))


注释：

# 此段代码示例了使用 numpy.arange() 函数创建等差数组的不同用法和行为。
# numpy.arange(start, stop, step, dtype=None) 函数生成从 start 到 stop（不包含）的数字序列，步长为 step。
# 当 step 不是整数时，生成的数组类型会根据输入情况而定，可以是整数或浮点数。
# 注意，对于大整数值，使用 numpy.arange() 可能会导致不正确的结果，应当考虑使用 numpy.linspace 替代。
# 将新文档添加到 numpy._core.multiarray 模块中，函数名为 '_get_ndarray_c_version'
add_newdoc('numpy._core.multiarray', '_get_ndarray_c_version',
    """_get_ndarray_c_version()

    Return the compile time NPY_VERSION (formerly called NDARRAY_VERSION) number.

    """)

# 将新文档添加到 numpy._core.multiarray 模块中，函数名为 '_reconstruct'
add_newdoc('numpy._core.multiarray', '_reconstruct',
    """_reconstruct(subtype, shape, dtype)

    Construct an empty array. Used by Pickles.

    """)

# 将新文档添加到 numpy._core.multiarray 模块中，函数名为 'promote_types'
add_newdoc('numpy._core.multiarray', 'promote_types',
    """
    promote_types(type1, type2)

    Returns the data type with the smallest size and smallest scalar
    kind to which both ``type1`` and ``type2`` may be safely cast.
    The returned data type is always considered "canonical", this mainly
    means that the promoted dtype will always be in native byte order.

    This function is symmetric, but rarely associative.

    Parameters
    ----------
    type1 : dtype or dtype specifier
        First data type.
    type2 : dtype or dtype specifier
        Second data type.

    Returns
    -------
    out : dtype
        The promoted data type.

    Notes
    -----
    Please see `numpy.result_type` for additional information about promotion.

    .. versionadded:: 1.6.0

    Starting in NumPy 1.9, promote_types function now returns a valid string
    length when given an integer or float dtype as one argument and a string
    dtype as another argument. Previously it always returned the input string
    dtype, even if it wasn't long enough to store the max integer/float value
    converted to a string.

    .. versionchanged:: 1.23.0

    NumPy now supports promotion for more structured dtypes.  It will now
    remove unnecessary padding from a structure dtype and promote included
    fields individually.

    See Also
    --------
    result_type, dtype, can_cast

    Examples
    --------
    >>> np.promote_types('f4', 'f8')
    dtype('float64')

    >>> np.promote_types('i8', 'f4')
    dtype('float64')

    >>> np.promote_types('>i8', '<c8')
    dtype('complex128')

    >>> np.promote_types('i4', 'S8')
    dtype('S11')

    An example of a non-associative case:

    >>> p = np.promote_types
    >>> p('S', p('i1', 'u1'))
    dtype('S6')
    >>> p(p('S', 'i1'), 'u1')
    dtype('S4')

    """)

# 将新文档添加到 numpy._core.multiarray 模块中，函数名为 'c_einsum'
add_newdoc('numpy._core.multiarray', 'c_einsum',
    """
    c_einsum(subscripts, *operands, out=None, dtype=None, order='K',
           casting='safe')

    *This documentation shadows that of the native python implementation of the `einsum` function,
    except all references and examples related to the `optimize` argument (v 0.12.0) have been removed.*

    Evaluates the Einstein summation convention on the operands.

    Using the Einstein summation convention, many common multi-dimensional,
    linear algebraic array operations can be represented in a simple fashion.
    In *implicit* mode `einsum` computes these values.

    In *explicit* mode, `einsum` provides further flexibility to compute
    other array operations that might not be considered classical Einstein
    """
    summation operations, by disabling, or forcing summation over specified
    subscript labels.

    See the notes and examples for clarification.

    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation as comma separated list of
        subscript labels. An implicit (classical Einstein summation)
        calculation is performed unless the explicit indicator '->' is
        included as well as subscript labels of the precise output form.
    operands : list of array_like
        These are the arrays for the operation.
    out : ndarray, optional
        If provided, the calculation is done into this array.
    dtype : {data-type, None}, optional
        If provided, forces the calculation to use the data type specified.
        Note that you may have to also give a more liberal `casting`
        parameter to allow the conversions. Default is None.
    order : {'C', 'F', 'A', 'K'}, optional
        Controls the memory layout of the output. 'C' means it should
        be C contiguous. 'F' means it should be Fortran contiguous,
        'A' means it should be 'F' if the inputs are all 'F', 'C' otherwise.
        'K' means it should be as close to the layout of the inputs as
        is possible, including arbitrarily permuted axes.
        Default is 'K'.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur.  Setting this to
        'unsafe' is not recommended, as it can adversely affect accumulations.

          * 'no' means the data types should not be cast at all.
          * 'equiv' means only byte-order changes are allowed.
          * 'safe' means only casts which can preserve values are allowed.
          * 'same_kind' means only safe casts or casts within a kind,
            like float64 to float32, are allowed.
          * 'unsafe' means any data conversions may be done.

        Default is 'safe'.
    optimize : {False, True, 'greedy', 'optimal'}, optional
        Controls if intermediate optimization should occur. No optimization
        will occur if False and True will default to the 'greedy' algorithm.
        Also accepts an explicit contraction list from the ``np.einsum_path``
        function. See ``np.einsum_path`` for more details. Defaults to False.

    Returns
    -------
    output : ndarray
        The calculation based on the Einstein summation convention.

    See Also
    --------
    einsum_path, dot, inner, outer, tensordot, linalg.multi_dot

    Notes
    -----
    .. versionadded:: 1.6.0

    The Einstein summation convention can be used to compute
    many multi-dimensional, linear algebraic array operations. `einsum`
    provides a succinct way of representing these.

    A non-exhaustive list of these operations,
    which can be computed by `einsum`, is shown below along with examples:

    * Trace of an array, :py:func:`numpy.trace`.
    * Return a diagonal, :py:func:`numpy.diag`.
    # Array axis summations, :py:func:`numpy.sum`.
    # Transpositions and permutations, :py:func:`numpy.transpose`.
    # Matrix multiplication and dot product, :py:func:`numpy.matmul` :py:func:`numpy.dot`.
    # Vector inner and outer products, :py:func:`numpy.inner` :py:func:`numpy.outer`.
    # Broadcasting, element-wise and scalar multiplication, :py:func:`numpy.multiply`.
    # Tensor contractions, :py:func:`numpy.tensordot`.
    # Chained array operations, in efficient calculation order, :py:func:`numpy.einsum_path`.
    
    # The subscripts string is a comma-separated list of subscript labels,
    # where each label refers to a dimension of the corresponding operand.
    # Whenever a label is repeated it is summed, so ``np.einsum('i,i', a, b)``
    # is equivalent to :py:func:`np.inner(a,b) <numpy.inner>`. If a label
    # appears only once, it is not summed, so ``np.einsum('i', a)`` produces a
    # view of ``a`` with no changes. A further example ``np.einsum('ij,jk', a, b)``
    # describes traditional matrix multiplication and is equivalent to
    # :py:func:`np.matmul(a,b) <numpy.matmul>`. Repeated subscript labels in one
    # operand take the diagonal. For example, ``np.einsum('ii', a)`` is equivalent
    # to :py:func:`np.trace(a) <numpy.trace>`.
    
    # In *implicit mode*, the chosen subscripts are important
    # since the axes of the output are reordered alphabetically.  This
    # means that ``np.einsum('ij', a)`` doesn't affect a 2D array, while
    # ``np.einsum('ji', a)`` takes its transpose. Additionally,
    # ``np.einsum('ij,jk', a, b)`` returns a matrix multiplication, while,
    # ``np.einsum('ij,jh', a, b)`` returns the transpose of the
    # multiplication since subscript 'h' precedes subscript 'i'.
    
    # In *explicit mode* the output can be directly controlled by
    # specifying output subscript labels.  This requires the
    # identifier '->' as well as the list of output subscript labels.
    # This feature increases the flexibility of the function since
    # summing can be disabled or forced when required. The call
    # ``np.einsum('i->', a)`` is like :py:func:`np.sum(a) <numpy.sum>`
    # if ``a`` is a 1-D array, and ``np.einsum('ii->i', a)``
    # is like :py:func:`np.diag(a) <numpy.diag>` if ``a`` is a square 2-D array.
    # The difference is that `einsum` does not allow broadcasting by default.
    # Additionally ``np.einsum('ij,jh->ih', a, b)`` directly specifies the
    # order of the output subscript labels and therefore returns matrix
    # multiplication, unlike the example above in implicit mode.
    
    # To enable and control broadcasting, use an ellipsis.  Default
    # NumPy-style broadcasting is done by adding an ellipsis
    # to the left of each term, like ``np.einsum('...ii->...i', a)``.
    # ``np.einsum('...i->...', a)`` is like
    # :py:func:`np.sum(a, axis=-1) <numpy.sum>` for array ``a`` of any shape.
    # To take the trace along the first and last axes,
    # you can do ``np.einsum('i...i', a)``, or to do a matrix-matrix
    product with the left-most indices instead of rightmost, one can do
    ``np.einsum('ij...,jk...->ik...', a, b)``.



    When there is only one operand, no axes are summed, and no output
    parameter is provided, a view into the operand is returned instead
    of a new array.  Thus, taking the diagonal as ``np.einsum('ii->i', a)``
    produces a view (changed in version 1.10.0).



    `einsum` also provides an alternative way to provide the subscripts
    and operands as ``einsum(op0, sublist0, op1, sublist1, ..., [sublistout])``.
    If the output shape is not provided in this format `einsum` will be
    calculated in implicit mode, otherwise it will be performed explicitly.
    The examples below have corresponding `einsum` calls with the two
    parameter methods.



    .. versionadded:: 1.10.0



    Views returned from einsum are now writeable whenever the input array
    is writeable. For example, ``np.einsum('ijk...->kji...', a)`` will now
    have the same effect as :py:func:`np.swapaxes(a, 0, 2) <numpy.swapaxes>`
    and ``np.einsum('ii->i', a)`` will return a writeable view of the diagonal
    of a 2D array.



    Examples
    --------
    >>> a = np.arange(25).reshape(5,5)
    >>> b = np.arange(5)
    >>> c = np.arange(6).reshape(2,3)



    Trace of a matrix:



    >>> np.einsum('ii', a)
    60
    >>> np.einsum(a, [0,0])
    60
    >>> np.trace(a)
    60



    Extract the diagonal (requires explicit form):



    >>> np.einsum('ii->i', a)
    array([ 0,  6, 12, 18, 24])
    >>> np.einsum(a, [0,0], [0])
    array([ 0,  6, 12, 18, 24])
    >>> np.diag(a)
    array([ 0,  6, 12, 18, 24])



    Sum over an axis (requires explicit form):



    >>> np.einsum('ij->i', a)
    array([ 10,  35,  60,  85, 110])
    >>> np.einsum(a, [0,1], [0])
    array([ 10,  35,  60,  85, 110])
    >>> np.sum(a, axis=1)
    array([ 10,  35,  60,  85, 110])



    For higher dimensional arrays summing a single axis can be done with ellipsis:



    >>> np.einsum('...j->...', a)
    array([ 10,  35,  60,  85, 110])
    >>> np.einsum(a, [Ellipsis,1], [Ellipsis])
    array([ 10,  35,  60,  85, 110])



    Compute a matrix transpose, or reorder any number of axes:



    >>> np.einsum('ji', c)
    array([[0, 3],
           [1, 4],
           [2, 5]])
    >>> np.einsum('ij->ji', c)
    array([[0, 3],
           [1, 4],
           [2, 5]])
    >>> np.einsum(c, [1,0])
    array([[0, 3],
           [1, 4],
           [2, 5]])
    >>> np.transpose(c)
    array([[0, 3],
           [1, 4],
           [2, 5]])



    Vector inner products:



    >>> np.einsum('i,i', b, b)
    30
    >>> np.einsum(b, [0], b, [0])
    30
    >>> np.inner(b,b)
    30



    Matrix vector multiplication:



    >>> np.einsum('ij,j', a, b)
    array([ 30,  80, 130, 180, 230])
    >>> np.einsum(a, [0,1], b, [1])
    array([ 30,  80, 130, 180, 230])
    >>> np.dot(a, b)
    array([ 30,  80, 130, 180, 230])
    >>> np.einsum('...j,j', a, b)
    array([ 30,  80, 130, 180, 230])
    Broadcasting and scalar multiplication:

    >>> np.einsum('..., ...', 3, c)
    # 对两个数组进行广播和标量乘法运算，生成新的数组
    array([[ 0,  3,  6],
           [ 9, 12, 15]])
    >>> np.einsum(',ij', 3, c)
    # 使用逗号和矩阵乘法描述，进行广播和标量乘法运算，生成新的数组
    array([[ 0,  3,  6],
           [ 9, 12, 15]])
    >>> np.einsum(3, [Ellipsis], c, [Ellipsis])
    # 对两个数组进行广播和标量乘法运算，生成新的数组
    array([[ 0,  3,  6],
           [ 9, 12, 15]])
    >>> np.multiply(3, c)
    # 对两个数组进行逐元素乘法，生成新的数组
    array([[ 0,  3,  6],
           [ 9, 12, 15]])

    Vector outer product:

    >>> np.einsum('i,j', np.arange(2)+1, b)
    # 计算向量的外积
    array([[0, 1, 2, 3, 4],
           [0, 2, 4, 6, 8]])
    >>> np.einsum(np.arange(2)+1, [0], b, [1])
    # 计算向量的外积
    array([[0, 1, 2, 3, 4],
           [0, 2, 4, 6, 8]])
    >>> np.outer(np.arange(2)+1, b)
    # 计算向量的外积
    array([[0, 1, 2, 3, 4],
           [0, 2, 4, 6, 8]])

    Tensor contraction:

    >>> a = np.arange(60.).reshape(3,4,5)
    >>> b = np.arange(24.).reshape(4,3,2)
    >>> np.einsum('ijk,jil->kl', a, b)
    # 计算张量的收缩
    array([[ 4400.,  4730.],
           [ 4532.,  4874.],
           [ 4664.,  5018.],
           [ 4796.,  5162.],
           [ 4928.,  5306.]])
    >>> np.einsum(a, [0,1,2], b, [1,0,3], [2,3])
    # 计算张量的收缩
    array([[ 4400.,  4730.],
           [ 4532.,  4874.],
           [ 4664.,  5018.],
           [ 4796.,  5162.],
           [ 4928.,  5306.]])
    >>> np.tensordot(a,b, axes=([1,0],[0,1]))
    # 计算张量的收缩
    array([[ 4400.,  4730.],
           [ 4532.,  4874.],
           [ 4664.,  5018.],
           [ 4796.,  5162.],
           [ 4928.,  5306.]])

    Writeable returned arrays (since version 1.10.0):

    >>> a = np.zeros((3, 3))
    >>> np.einsum('ii->i', a)[:] = 1
    # 对数组进行操作，将对角线元素设置为1
    >>> a
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])

    Example of ellipsis use:

    >>> a = np.arange(6).reshape((3,2))
    >>> b = np.arange(12).reshape((4,3))
    >>> np.einsum('ki,jk->ij', a, b)
    # 计算张量的收缩，使用省略号
    array([[10, 28, 46, 64],
           [13, 40, 67, 94]])
    >>> np.einsum('ki,...k->i...', a, b)
    # 计算张量的收缩，使用省略号
    array([[10, 28, 46, 64],
           [13, 40, 67, 94]])
    >>> np.einsum('k...,jk', a, b)
    # 计算张量的收缩，使用省略号
    array([[10, 28, 46, 64],
           [13, 40, 67, 94]])
##############################################################################
#
# Documentation for ndarray attributes and methods
#
##############################################################################


##############################################################################
#
# ndarray object
#
##############################################################################


add_newdoc('numpy._core.multiarray', 'ndarray',
    """
    ndarray(shape, dtype=float, buffer=None, offset=0,
            strides=None, order=None)

    An array object represents a multidimensional, homogeneous array
    of fixed-size items.  An associated data-type object describes the
    format of each element in the array (its byte-order, how many bytes it
    occupies in memory, whether it is an integer, a floating point number,
    or something else, etc.)

    Arrays should be constructed using `array`, `zeros` or `empty` (refer
    to the See Also section below).  The parameters given here refer to
    a low-level method (`ndarray(...)`) for instantiating an array.

    For more information, refer to the `numpy` module and examine the
    methods and attributes of an array.

    Parameters
    ----------
    (for the __new__ method; see Notes below)

    shape : tuple of ints
        Shape of created array.
    dtype : data-type, optional
        Any object that can be interpreted as a numpy data type.
    buffer : object exposing buffer interface, optional
        Used to fill the array with data.
    offset : int, optional
        Offset of array data in buffer.
    strides : tuple of ints, optional
        Strides of data in memory.
    order : {'C', 'F'}, optional
        Row-major (C-style) or column-major (Fortran-style) order.

    Attributes
    ----------
    T : ndarray
        Transpose of the array.
    data : buffer
        The array's elements, in memory.
    dtype : dtype object
        Describes the format of the elements in the array.
    flags : dict
        Dictionary containing information related to memory use, e.g.,
        'C_CONTIGUOUS', 'OWNDATA', 'WRITEABLE', etc.
    flat : numpy.flatiter object
        Flattened version of the array as an iterator.  The iterator
        allows assignments, e.g., ``x.flat = 3`` (See `ndarray.flat` for
        assignment examples; TODO).
    imag : ndarray
        Imaginary part of the array.
    real : ndarray
        Real part of the array.
    size : int
        Number of elements in the array.
    itemsize : int
        The memory use of each array element in bytes.
    nbytes : int
        The total number of bytes required to store the array data,
        i.e., ``itemsize * size``.
    ndim : int
        The array's number of dimensions.
    shape : tuple of ints
        Shape of the array.
    """
)
    strides : tuple of ints
        # 每个元素在内存中移动到下一个元素所需的步长。例如，在 C 顺序中，一个连续的 int16 类型的 (3, 4) 数组的步长是 (8, 2)。
        # 这意味着在内存中从一个元素移动到下一个元素需要跳过 2 个字节，从一行移到下一行需要一次跳过 8 个字节（2 * 4）。

    ctypes : ctypes object
        # 包含数组与 ctypes 交互所需属性的类。

    base : ndarray
        # 如果数组是另一个数组的视图，则该数组是它的“base”（除非该数组本身也是视图）。
        # “base” 数组实际存储数组数据的位置。

    See Also
    --------
    array : 构造数组。
    zeros : 创建每个元素为零的数组。
    empty : 创建数组，但不改变其分配的内存（即，其中包含“垃圾”）。
    dtype : 创建数据类型。
    numpy.typing.NDArray : 与其 `dtype.type <numpy.dtype.type>` 相关的 `generic <generic type>` 的 ndarray 别名。

    Notes
    -----
    使用 `__new__` 有两种创建数组的模式：

    1. 如果 `buffer` 是 None，则仅使用 `shape`、`dtype` 和 `order`。
    2. 如果 `buffer` 是一个公开缓冲区接口的对象，则解释所有关键字。

    不需要 `__init__` 方法，因为数组在 `__new__` 方法之后已完全初始化。

    Examples
    --------
    这些示例说明了低级 `ndarray` 构造函数。查看上面的 `See Also` 部分可以找到更简单的构建 ndarray 的方法。

    第一种模式，`buffer` 是 None：

    >>> np.ndarray(shape=(2,2), dtype=float, order='F')
    array([[0.0e+000, 0.0e+000],  # 随机值
           [     nan, 2.5e-323]])

    第二种模式：

    >>> np.ndarray((2,), buffer=np.array([1,2,3]),
    ...            offset=np.int_().itemsize,
    ...            dtype=int)  # offset = 1*itemsize，即跳过第一个元素
    array([2, 3])
# 添加新的文档字符串到 'numpy._core.multiarray' 的 'ndarray' 上的 '__array_interface__' 属性
add_newdoc('numpy._core.multiarray', 'ndarray', ('__array_interface__',
    """Array protocol: Python side."""))

# 添加新的文档字符串到 'numpy._core.multiarray' 的 'ndarray' 上的 '__array_priority__' 属性
add_newdoc('numpy._core.multiarray', 'ndarray', ('__array_priority__',
    """Array priority."""))

# 添加新的文档字符串到 'numpy._core.multiarray' 的 'ndarray' 上的 '__array_struct__' 属性
add_newdoc('numpy._core.multiarray', 'ndarray', ('__array_struct__',
    """Array protocol: C-struct side."""))

# 添加新的文档字符串到 'numpy._core.multiarray' 的 'ndarray' 上的 '__dlpack__' 属性
add_newdoc('numpy._core.multiarray', 'ndarray', ('__dlpack__',
    """a.__dlpack__(*, stream=None)

    DLPack Protocol: Part of the Array API."""))

# 添加新的文档字符串到 'numpy._core.multiarray' 的 'ndarray' 上的 '__dlpack_device__' 属性
add_newdoc('numpy._core.multiarray', 'ndarray', ('__dlpack_device__',
    """a.__dlpack_device__()

    DLPack Protocol: Part of the Array API."""))

# 添加新的文档字符串到 'numpy._core.multiarray' 的 'ndarray' 上的 'base' 属性
add_newdoc('numpy._core.multiarray', 'ndarray', ('base',
    """
    Base object if memory is from some other object.

    Examples
    --------
    The base of an array that owns its memory is None:

    >>> x = np.array([1,2,3,4])
    >>> x.base is None
    True

    Slicing creates a view, whose memory is shared with x:

    >>> y = x[2:]
    >>> y.base is x
    True

    """))

# 添加新的文档字符串到 'numpy._core.multiarray' 的 'ndarray' 上的 'ctypes' 属性
add_newdoc('numpy._core.multiarray', 'ndarray', ('ctypes',
    """
    An object to simplify the interaction of the array with the ctypes
    module.

    This attribute creates an object that makes it easier to use arrays
    when calling shared libraries with the ctypes module. The returned
    object has, among others, data, shape, and strides attributes (see
    Notes below) which themselves return ctypes objects that can be used
    as arguments to a shared library.

    Parameters
    ----------
    None

    Returns
    -------
    c : Python object
        Possessing attributes data, shape, strides, etc.

    See Also
    --------
    numpy.ctypeslib

    Notes
    -----
    Below are the public attributes of this object which were documented
    in "Guide to NumPy" (we have omitted undocumented public attributes,
    as well as documented private attributes):

    .. autoattribute:: numpy._core._internal._ctypes.data
        :noindex:

    .. autoattribute:: numpy._core._internal._ctypes.shape
        :noindex:

    .. autoattribute:: numpy._core._internal._ctypes.strides
        :noindex:

    .. automethod:: numpy._core._internal._ctypes.data_as
        :noindex:

    .. automethod:: numpy._core._internal._ctypes.shape_as
        :noindex:

    .. automethod:: numpy._core._internal._ctypes.strides_as
        :noindex:

    If the ctypes module is not available, then the ctypes attribute
    of array objects still returns something useful, but ctypes objects
    are not returned and errors may be raised instead. In particular,
    the object will still have the ``as_parameter`` attribute which will
    return an integer equal to the data attribute.

    Examples
    --------
    >>> import ctypes
    >>> x = np.array([[0, 1], [2, 3]], dtype=np.int32)

"""))
    # 输出数组 x 的内容
    >>> x
    array([[0, 1],
           [2, 3]], dtype=int32)
    # 获取数组 x 的内存地址
    >>> x.ctypes.data
    31962608 # 可能会有所不同
    # 将数组 x 视为 ctypes 中 c_uint32 类型的指针
    >>> x.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    <__main__.LP_c_uint object at 0x7ff2fc1fc200> # 可能会有所不同
    # 获取 c_uint32 类型指针指向的内容，此时数组起始位置为 0
    >>> x.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)).contents
    c_uint(0)
    # 将数组 x 视为 ctypes 中 c_uint64 类型的指针
    >>> x.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)).contents
    c_ulong(4294967296)
    # 获取数组 x 的形状信息
    >>> x.ctypes.shape
    <numpy._core._internal.c_long_Array_2 object at 0x7ff2fc1fce60> # 可能会有所不同
    # 获取数组 x 的跨度信息
    >>> x.ctypes.strides
    <numpy._core._internal.c_long_Array_2 object at 0x7ff2fc1ff320> # 可能会有所不同
# 添加新的文档字符串到 numpy._core.multiarray.ndarray 中的 'data' 属性
add_newdoc('numpy._core.multiarray', 'ndarray', ('data',
    """Python buffer object pointing to the start of the array's data."""))


# 添加新的文档字符串到 numpy._core.multiarray.ndarray 中的 'dtype' 属性
add_newdoc('numpy._core.multiarray', 'ndarray', ('dtype',
    """
    Data-type of the array's elements.

    .. warning::

        Setting ``arr.dtype`` is discouraged and may be deprecated in the
        future.  Setting will replace the ``dtype`` without modifying the
        memory (see also `ndarray.view` and `ndarray.astype`).

    Parameters
    ----------
    None

    Returns
    -------
    d : numpy dtype object

    See Also
    --------
    ndarray.astype : Cast the values contained in the array to a new data-type.
    ndarray.view : Create a view of the same data but a different data-type.
    numpy.dtype

    Examples
    --------
    >>> x
    array([[0, 1],
           [2, 3]])
    >>> x.dtype
    dtype('int32')
    >>> type(x.dtype)
    <type 'numpy.dtype'>

    """))


# 添加新的文档字符串到 numpy._core.multiarray.ndarray 中的 'imag' 属性
add_newdoc('numpy._core.multiarray', 'ndarray', ('imag',
    """
    The imaginary part of the array.

    Examples
    --------
    >>> x = np.sqrt([1+0j, 0+1j])
    >>> x.imag
    array([ 0.        ,  0.70710678])
    >>> x.imag.dtype
    dtype('float64')

    """))


# 添加新的文档字符串到 numpy._core.multiarray.ndarray 中的 'itemsize' 属性
add_newdoc('numpy._core.multiarray', 'ndarray', ('itemsize',
    """
    Length of one array element in bytes.

    Examples
    --------
    >>> x = np.array([1,2,3], dtype=np.float64)
    >>> x.itemsize
    8
    >>> x = np.array([1,2,3], dtype=np.complex128)
    >>> x.itemsize
    16

    """))


# 添加新的文档字符串到 numpy._core.multiarray.ndarray 中的 'flags' 属性
add_newdoc('numpy._core.multiarray', 'ndarray', ('flags',
    """
    Information about the memory layout of the array.

    Attributes
    ----------
    C_CONTIGUOUS (C)
        The data is in a single, C-style contiguous segment.
    F_CONTIGUOUS (F)
        The data is in a single, Fortran-style contiguous segment.
    OWNDATA (O)
        The array owns the memory it uses or borrows it from another object.
    WRITEABLE (W)
        The data area can be written to.  Setting this to False locks
        the data, making it read-only.  A view (slice, etc.) inherits WRITEABLE
        from its base array at creation time, but a view of a writeable
        array may be subsequently locked while the base array remains writeable.
        (The opposite is not true, in that a view of a locked array may not
        be made writeable.  However, currently, locking a base object does not
        lock any views that already reference it, so under that circumstance it
        is possible to alter the contents of a locked array via a previously
        created writeable view onto it.)  Attempting to change a non-writeable
        array raises a RuntimeError exception.
    ALIGNED (A)
        The data and all elements are aligned appropriately for the hardware.

    """))
    WRITEBACKIFCOPY (X)
        This array is a copy of some other array. The C-API function
        PyArray_ResolveWritebackIfCopy must be called before deallocating
        to the base array will be updated with the contents of this array.
    FNC
        F_CONTIGUOUS and not C_CONTIGUOUS.
    FORC
        F_CONTIGUOUS or C_CONTIGUOUS (one-segment test).
    BEHAVED (B)
        ALIGNED and WRITEABLE.
    CARRAY (CA)
        BEHAVED and C_CONTIGUOUS.
    FARRAY (FA)
        BEHAVED and F_CONTIGUOUS and not C_CONTIGUOUS.

    Notes
    -----
    The `flags` object can be accessed dictionary-like (as in ``a.flags['WRITEABLE']``),
    or by using lowercased attribute names (as in ``a.flags.writeable``). Short flag
    names are only supported in dictionary access.

    Only the WRITEBACKIFCOPY, WRITEABLE, and ALIGNED flags can be
    changed by the user, via direct assignment to the attribute or dictionary
    entry, or by calling `ndarray.setflags`.

    The array flags cannot be set arbitrarily:

    - WRITEBACKIFCOPY can only be set ``False``.
    - ALIGNED can only be set ``True`` if the data is truly aligned.
    - WRITEABLE can only be set ``True`` if the array owns its own memory
      or the ultimate owner of the memory exposes a writeable buffer
      interface or is a string.

    Arrays can be both C-style and Fortran-style contiguous simultaneously.
    This is clear for 1-dimensional arrays, but can also be true for higher
    dimensional arrays.

    Even for contiguous arrays a stride for a given dimension
    ``arr.strides[dim]`` may be *arbitrary* if ``arr.shape[dim] == 1``
    or the array has no elements.
    It does *not* generally hold that ``self.strides[-1] == self.itemsize``
    for C-style contiguous arrays or ``self.strides[0] == self.itemsize`` for
    Fortran-style contiguous arrays is true.
# 添加新的文档字符串到指定的numpy._core.multiarray.ndarray属性
add_newdoc('numpy._core.multiarray', 'ndarray', ('flat',
    """
    A 1-D iterator over the array.

    This is a `numpy.flatiter` instance, which acts similarly to, but is not
    a subclass of, Python's built-in iterator object.

    See Also
    --------
    flatten : Return a copy of the array collapsed into one dimension.

    flatiter

    Examples
    --------
    >>> x = np.arange(1, 7).reshape(2, 3)
    >>> x
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> x.flat[3]
    4
    >>> x.T
    array([[1, 4],
           [2, 5],
           [3, 6]])
    >>> x.T.flat[3]
    5
    >>> type(x.flat)
    <class 'numpy.flatiter'>

    An assignment example:

    >>> x.flat = 3; x
    array([[3, 3, 3],
           [3, 3, 3]])
    >>> x.flat[[1,4]] = 1; x
    array([[3, 1, 3],
           [3, 1, 3]])

    """))


# 添加新的文档字符串到指定的numpy._core.multiarray.ndarray属性
add_newdoc('numpy._core.multiarray', 'ndarray', ('nbytes',
    """
    Total bytes consumed by the elements of the array.

    Notes
    -----
    Does not include memory consumed by non-element attributes of the
    array object.

    See Also
    --------
    sys.getsizeof
        Memory consumed by the object itself without parents in case view.
        This does include memory consumed by non-element attributes.

    Examples
    --------
    >>> x = np.zeros((3,5,2), dtype=np.complex128)
    >>> x.nbytes
    480
    >>> np.prod(x.shape) * x.itemsize
    480

    """))


# 添加新的文档字符串到指定的numpy._core.multiarray.ndarray属性
add_newdoc('numpy._core.multiarray', 'ndarray', ('ndim',
    """
    Number of array dimensions.

    Examples
    --------
    >>> x = np.array([1, 2, 3])
    >>> x.ndim
    1
    >>> y = np.zeros((2, 3, 4))
    >>> y.ndim
    3

    """))


# 添加新的文档字符串到指定的numpy._core.multiarray.ndarray属性
add_newdoc('numpy._core.multiarray', 'ndarray', ('real',
    """
    The real part of the array.

    Examples
    --------
    >>> x = np.sqrt([1+0j, 0+1j])
    >>> x.real
    array([ 1.        ,  0.70710678])
    >>> x.real.dtype
    dtype('float64')

    See Also
    --------
    numpy.real : equivalent function

    """))


# 添加新的文档字符串到指定的numpy._core.multiarray.ndarray属性
add_newdoc('numpy._core.multiarray', 'ndarray', ('shape',
    """
    Tuple of array dimensions.

    The shape property is usually used to get the current shape of an array,
    but may also be used to reshape the array in-place by assigning a tuple of
    array dimensions to it.  As with `numpy.reshape`, one of the new shape
    dimensions can be -1, in which case its value is inferred from the size of
    the array and the remaining dimensions. Reshaping an array in-place will
    fail if a copy is required.

    .. warning::

        Setting ``arr.shape`` is discouraged and may be deprecated in the
        future.  Using `ndarray.reshape` is the preferred approach.

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4])
    >>> x.shape
    (4,)
    >>> y = np.zeros((2, 3, 4))
    >>> y.shape
    (2, 3, 4)
    >>> y.shape = (3, 8)
    >>> y
    array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

    """))
    # 将数组 y 的形状改为 (3, 6)，但修改后数组的总大小必须保持不变，否则会引发 ValueError 异常
    >>> y.shape = (3, 6)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ValueError: total size of new array must be unchanged
    
    # 使用 np.zeros((4,2)) 创建的数组，在切片 [::2] 的结果上尝试修改形状为 (-1,)，但这种直接赋值修改形状的方式是不兼容的，应使用 `.reshape()` 方法创建带有期望形状的副本
    >>> np.zeros((4,2))[::2].shape = (-1,)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    AttributeError: Incompatible shape for in-place modification. Use
    `.reshape()` to make a copy with the desired shape.
    
    # 参考
    # numpy.shape：与 `shape` 设置器功能相同的获取器函数。
    # numpy.reshape：类似于设置 `shape` 的函数。
    # ndarray.reshape：类似于设置 `shape` 的方法。
    See Also
    --------
    numpy.shape : Equivalent getter function.
    numpy.reshape : Function similar to setting ``shape``.
    ndarray.reshape : Method similar to setting ``shape``.
add_newdoc('numpy._core.multiarray', 'ndarray', ('size',
    """
    Array中的元素数量。

    等同于 ``np.prod(a.shape)``，即数组维度的乘积。

    注意
    -----
    `a.size` 返回一个标准的任意精度的 Python 整数。使用其他方法（如建议的 ``np.prod(a.shape)`` 返回的是 ``np.int_`` 的实例）可能不会如此，并且在进一步计算中使用这个值可能导致溢出固定大小整数类型。

    示例
    --------
    >>> x = np.zeros((3, 5, 2), dtype=np.complex128)
    >>> x.size
    30
    >>> np.prod(x.shape)
    30

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('strides',
    """
    在遍历数组时，每个维度移动的字节步长元组。

    数组 `a` 中元素 ``(i[0], i[1], ..., i[n])`` 的字节偏移量是::

        offset = sum(np.array(i) * a.strides)

    关于 strides 的更详细解释可以在 :ref:`arrays.ndarray` 中找到。

    .. warning::

        不推荐设置 ``arr.strides``，将来可能会被弃用。应优先使用 `numpy.lib.stride_tricks.as_strided` 安全地创建相同数据的新视图。

    注意
    -----
    假设一个存储为连续内存块的 32 位整数数组（每个 4 字节）::

      x = np.array([[0, 1, 2, 3, 4],
                    [5, 6, 7, 8, 9]], dtype=np.int32)

    数组的 strides 告诉我们沿特定轴移动到下一个位置时需要跳过多少字节。例如，移动到下一列需要跳过 4 字节（1 个值），移动到下一行相同位置需要跳过 20 字节（5 个值）。因此，数组 `x` 的 strides 是 ``(20, 4)``。

    参见
    --------
    numpy.lib.stride_tricks.as_strided

    示例
    --------
    >>> y = np.reshape(np.arange(2*3*4), (2,3,4))
    >>> y
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],
           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]])
    >>> y.strides
    (48, 16, 4)
    >>> y[1,1,1]
    17
    >>> offset=sum(y.strides * np.array((1,1,1)))
    >>> offset/y.itemsize
    17

    >>> x = np.reshape(np.arange(5*6*7*8), (5,6,7,8)).transpose(2,3,1,0)
    >>> x.strides
    (32, 4, 224, 1344)
    >>> i = np.array([3,5,2,2])
    >>> offset = sum(i * x.strides)
    >>> x[3,5,2,2]
    813
    >>> offset / x.itemsize
    813

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('T',
    """
    转置数组的视图。

    等同于 ``self.transpose()``。

    示例
    --------
    >>> a = np.array([[1, 2], [3, 4]])
    >>> a
    array([[1, 2],
           [3, 4]])
    >>> a.T
    array([[1, 3],
           [2, 4]])

    >>> a = np.array([1, 2, 3, 4])

    """))
    # 输出数组 a 的内容
    >>> a
    # 输出数组 a 的转置，对于一维数组，转置仍然是自身
    >>> a.T

    # 参考链接，指向 transpose 函数的相关信息
    See Also
    --------
    transpose
##############################################################################
#
# ndarray methods
#
##############################################################################

add_newdoc('numpy._core.multiarray', 'ndarray', ('mT',
    """
    View of the matrix transposed array.

    The matrix transpose is the transpose of the last two dimensions, even
    if the array is of higher dimension.

    .. versionadded:: 2.0

    Raises
    ------
    ValueError
        If the array is of dimension less than 2.

    Examples
    --------
    >>> a = np.array([[1, 2], [3, 4]])
    >>> a
    array([[1, 2],
           [3, 4]])
    >>> a.mT
    array([[1, 3],
           [2, 4]])

    >>> a = np.arange(8).reshape((2, 2, 2))
    >>> a
    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])
    >>> a.mT
    array([[[0, 2],
            [1, 3]],
           [[4, 6],
            [5, 7]]])

    """))

add_newdoc('numpy._core.multiarray', 'ndarray', ('__array__',
    """
    a.__array__([dtype], *, copy=None)

    For ``dtype`` parameter it returns a new reference to self if
    ``dtype`` is not given or it matches array's data type.
    A new array of provided data type is returned if ``dtype``
    is different from the current data type of the array.
    For ``copy`` parameter it returns a new reference to self if
    ``copy=False`` or ``copy=None`` and copying isn't enforced by ``dtype``
    parameter. The method returns a new array for ``copy=True``, regardless of
    ``dtype`` parameter.

    A more detailed explanation of the ``__array__`` interface
    can be found in :ref:`dunder_array.interface`.

    """))

add_newdoc('numpy._core.multiarray', 'ndarray', ('__array_finalize__',
    """
    a.__array_finalize__(obj, /)

    Present so subclasses can call super. Does nothing.

    """))

add_newdoc('numpy._core.multiarray', 'ndarray', ('__array_wrap__',
    """
    a.__array_wrap__(array[, context], /)

    Returns a view of `array` with the same type as self.

    """))

add_newdoc('numpy._core.multiarray', 'ndarray', ('__copy__',
    """
    a.__copy__()

    Used if :func:`copy.copy` is called on an array. Returns a copy of the array.

    Equivalent to ``a.copy(order='K')``.

    """))

add_newdoc('numpy._core.multiarray', 'ndarray', ('__class_getitem__',
    """
    a.__class_getitem__(item, /)

    Return a parametrized wrapper around the `~numpy.ndarray` type.

    .. versionadded:: 1.22

    Returns
    -------
    alias : types.GenericAlias
        A parametrized `~numpy.ndarray` type.

    Examples
    --------
    >>> from typing import Any
    >>> import numpy as np

    >>> np.ndarray[Any, np.dtype[Any]]
    numpy.ndarray[typing.Any, numpy.dtype[typing.Any]]

    See Also
    --------
    :pep:`585` : Type hinting generics in standard collections.

    """))
    # 导入 numpy 库中的 typing 模块，用于类型提示
    numpy.typing.NDArray : An ndarray alias :term:`generic <generic type>`
                        w.r.t. its `dtype.type <numpy.dtype.type>`.
    # 定义了 numpy.typing.NDArray 别名，表示一种 ndarray 类型的泛型，与其 dtype.type 相关联
    # 该注释说明了 NDArray 是一个泛型类型，与其 dtype 的类型相关联
# 向 numpy 的 ndarray 类型添加新的文档字符串，用于描述 __deepcopy__ 方法的作用
add_newdoc('numpy._core.multiarray', 'ndarray', ('__deepcopy__',
    """
    a.__deepcopy__(memo, /)

    Used if :func:`copy.deepcopy` is called on an array.

    """))


# 向 numpy 的 ndarray 类型添加新的文档字符串，用于描述 __reduce__ 方法的作用
add_newdoc('numpy._core.multiarray', 'ndarray', ('__reduce__',
    """
    a.__reduce__()

    For pickling.

    """))


# 向 numpy 的 ndarray 类型添加新的文档字符串，用于描述 __setstate__ 方法的作用
add_newdoc('numpy._core.multiarray', 'ndarray', ('__setstate__',
    """
    a.__setstate__(state, /)

    For unpickling.

    The `state` argument must be a sequence that contains the following
    elements:

    Parameters
    ----------
    version : int
        optional pickle version. If omitted defaults to 0.
    shape : tuple
    dtype : data-type
    isFortran : bool
    rawdata : string or list
        a binary string with the data (or a list if 'a' is an object array)

    """))


# 向 numpy 的 ndarray 类型添加新的文档字符串，用于描述 all 方法的作用
add_newdoc('numpy._core.multiarray', 'ndarray', ('all',
    """
    a.all(axis=None, out=None, keepdims=False, *, where=True)

    Returns True if all elements evaluate to True.

    Refer to `numpy.all` for full documentation.

    See Also
    --------
    numpy.all : equivalent function

    """))


# 向 numpy 的 ndarray 类型添加新的文档字符串，用于描述 any 方法的作用
add_newdoc('numpy._core.multiarray', 'ndarray', ('any',
    """
    a.any(axis=None, out=None, keepdims=False, *, where=True)

    Returns True if any of the elements of `a` evaluate to True.

    Refer to `numpy.any` for full documentation.

    See Also
    --------
    numpy.any : equivalent function

    """))


# 向 numpy 的 ndarray 类型添加新的文档字符串，用于描述 argmax 方法的作用
add_newdoc('numpy._core.multiarray', 'ndarray', ('argmax',
    """
    a.argmax(axis=None, out=None, *, keepdims=False)

    Return indices of the maximum values along the given axis.

    Refer to `numpy.argmax` for full documentation.

    See Also
    --------
    numpy.argmax : equivalent function

    """))


# 向 numpy 的 ndarray 类型添加新的文档字符串，用于描述 argmin 方法的作用
add_newdoc('numpy._core.multiarray', 'ndarray', ('argmin',
    """
    a.argmin(axis=None, out=None, *, keepdims=False)

    Return indices of the minimum values along the given axis.

    Refer to `numpy.argmin` for detailed documentation.

    See Also
    --------
    numpy.argmin : equivalent function

    """))


# 向 numpy 的 ndarray 类型添加新的文档字符串，用于描述 argsort 方法的作用
add_newdoc('numpy._core.multiarray', 'ndarray', ('argsort',
    """
    a.argsort(axis=-1, kind=None, order=None)

    Returns the indices that would sort this array.

    Refer to `numpy.argsort` for full documentation.

    See Also
    --------
    numpy.argsort : equivalent function

    """))


# 向 numpy 的 ndarray 类型添加新的文档字符串，用于描述 argpartition 方法的作用
add_newdoc('numpy._core.multiarray', 'ndarray', ('argpartition',
    """
    a.argpartition(kth, axis=-1, kind='introselect', order=None)

    Returns the indices that would partition this array.

    Refer to `numpy.argpartition` for full documentation.

    .. versionadded:: 1.8.0

    See Also
    --------
    numpy.argpartition : equivalent function

    """))


# 向 numpy 的 ndarray 类型添加新的文档字符串，用于描述 astype 方法的作用
add_newdoc('numpy._core.multiarray', 'ndarray', ('astype',
    """
    a.astype(dtype, order='K', casting='unsafe', subok=True, copy=True)

    Copy of the array, cast to a specified type.

    Parameters
    ----------

    """))
    # 数据类型：字符串或数据类型，用于将数组转换为的类型码或数据类型。
    dtype : str or dtype
    # 排序方式：{'C'，'F'，'A'，'K'}，可选参数，控制结果的内存布局顺序。
    # 'C' 表示C顺序，'F' 表示Fortran顺序，'A' 表示如果所有数组都是Fortran连续的，则为'F'顺序，否则为'C'顺序，
    # 'K' 表示尽可能接近数组元素在内存中出现的顺序。默认为 'K'。
    order : {'C', 'F', 'A', 'K'}, optional
    # 强制类型转换：{'no', 'equiv', 'safe', 'same_kind', 'unsafe'}，可选参数，控制可能发生的数据类型转换类型。
    # 默认为 'unsafe'，以保持向后兼容性。
    # * 'no' 表示根本不应转换数据类型。
    # * 'equiv' 表示仅允许字节顺序更改。
    # * 'safe' 表示仅允许可以保留值的转换。
    # * 'same_kind' 表示仅允许安全转换或同一类别内的转换，例如从float64到float32。
    # * 'unsafe' 表示可能执行任何数据转换。
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
    # 是否传递子类：bool，可选参数。如果为True，则子类将被传递（默认行为），否则返回的数组将被强制为基类数组。
    subok : bool, optional
    # 复制选项：bool，可选参数。默认情况下，astype始终返回新分配的数组。如果设置为false，并且满足dtype，order和subok的要求，
    # 则返回输入数组而不是副本。
    copy : bool, optional

    # 返回值：ndarray
    # 除非copy为False且满足返回输入数组的其他条件（请参见copy输入参数的描述），否则arr_t是与输入数组相同形状的新数组，
    # 其dtype和order由dtype，order给出。
    Returns
    -------
    arr_t : ndarray

    # 注意事项：
    # .. versionchanged:: 1.17.0
    #    仅在“unsafe”转换中才可能在简单数据类型和结构化数据类型之间进行转换。
    #    允许向多个字段转换，但不允许从多个字段转换。
    Notes
    -----
    .. versionchanged:: 1.17.0
       Casting between a simple data type and a structured one is possible only
       for "unsafe" casting.  Casting to multiple fields is allowed, but
       casting from multiple fields is not.

    # .. versionchanged:: 1.9.0
    #    在“safe”转换模式下，从数字到字符串类型的转换需要字符串dtype的长度足够长，
    #    能够存储最大整数/浮点值转换后的值。
    #    Raises
    #    ------
    #    ComplexWarning
    #        When casting from complex to float or int. To avoid this,
    #        one should use ``a.real.astype(t)``.
    .. versionchanged:: 1.9.0
       在“safe”转换模式下，从数字到字符串类型的转换需要字符串dtype的长度足够长，
       能够存储最大整数/浮点值转换后的值。

    # 异常情况：
    # ComplexWarning
    #     当从复数到浮点数或整数的转换时会触发此警告。要避免此问题，
    #     应使用 ``a.real.astype(t)``。
    Raises
    ------
    ComplexWarning
        当从复数到浮点数或整数的转换时。为避免此问题，应使用 ``a.real.astype(t)``。

    # 示例：
    # >>> x = np.array([1, 2, 2.5])
    # >>> x
    # array([1. ,  2. ,  2.5])
    # >>> x.astype(int)
    # array([1, 2, 2])
    Examples
    --------
    >>> x = np.array([1, 2, 2.5])
    >>> x
    array([1. ,  2. ,  2.5])

    >>> x.astype(int)
    array([1, 2, 2])
add_newdoc('numpy._core.multiarray', 'ndarray', ('byteswap',
    """
    a.byteswap(inplace=False)

    交换数组元素的字节顺序

    通过返回一个交换字节顺序的数组来在低端和大端数据表示之间切换，可选择地原地交换。
    字节串数组不会被交换。复数的实部和虚部会分别交换。

    参数
    ----------
    inplace : bool, 可选
        如果为 ``True``，则原地交换字节顺序，默认为 ``False``。

    返回
    -------
    out : ndarray
        交换字节顺序的数组。如果 `inplace` 是 ``True``，则返回自身的视图。

    示例
    --------
    >>> A = np.array([1, 256, 8755], dtype=np.int16)
    >>> list(map(hex, A))
    ['0x1', '0x100', '0x2233']
    >>> A.byteswap(inplace=True)
    array([  256,     1, 13090], dtype=int16)
    >>> list(map(hex, A))
    ['0x100', '0x1', '0x3322']

    字节串数组不会被交换

    >>> A = np.array([b'ceg', b'fac'])
    >>> A.byteswap()
    array([b'ceg', b'fac'], dtype='|S3')

    ``A.view(A.dtype.newbyteorder()).byteswap()`` 会生成具有相同值但在内存中表示不同的数组

    >>> A = np.array([1, 2, 3])
    >>> A.view(np.uint8)
    array([1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0,
           0, 0], dtype=uint8)
    >>> A.view(A.dtype.newbyteorder()).byteswap(inplace=True)
    array([1, 2, 3])
    >>> A.view(np.uint8)
    array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0,
           0, 3], dtype=uint8)

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('choose',
    """
    a.choose(choices, out=None, mode='raise')

    使用索引数组从一组选择中构建新数组。

    详细文档请参见 `numpy.choose`。

    另请参阅
    --------
    numpy.choose : 等效函数

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('clip',
    """
    a.clip(min=None, max=None, out=None, **kwargs)

    返回其值限制在 ``[min, max]`` 之间的数组。
    必须给出 max 或 min 中的一个。

    详细文档请参见 `numpy.clip`。

    另请参阅
    --------
    numpy.clip : 等效函数

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('compress',
    """
    a.compress(condition, axis=None, out=None)

    返回沿给定轴选定的该数组的切片。

    详细文档请参见 `numpy.compress`。

    另请参阅
    --------
    numpy.compress : 等效函数

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('conj',
    """
    a.conj()

    对所有元素进行复共轭。

    详细文档请参见 `numpy.conjugate`。

    另请参阅
    --------
    numpy.conjugate : 等效函数

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('conjugate',
    """
    a.conjugate()

    返回复共轭，逐元素操作。

    """))
    # 调用 numpy 模块中的 `conjugate` 函数，具体文档参见该函数的完整说明。
    # 
    # See Also 段落：
    # --------
    # numpy.conjugate : 等效的函数
    """
    Refer to `numpy.conjugate` for full documentation.

    See Also
    --------
    numpy.conjugate : equivalent function
    """
add_newdoc('numpy._core.multiarray', 'ndarray', ('copy',
    """
    a.copy(order='C')

    返回数组的副本。

    参数
    ----------
    order : {'C', 'F', 'A', 'K'}, 可选
        控制副本的内存布局。'C' 表示 C-order，
        'F' 表示 F-order，'A' 表示如果 `a` 是 Fortran 连续则使用 'F'，
        否则使用 'C'。'K' 表示尽可能匹配 `a` 的布局。（注意，这个函数
        和 :func:`numpy.copy` 很相似，但是它们对于 order= 参数有不同的默认值，
        并且这个函数总是通过子类。）

    另请参阅
    --------
    numpy.copy : 具有不同默认行为的类似函数
    numpy.copyto

    注意
    -----
    这个函数是创建数组副本的首选方法。函数 :func:`numpy.copy` 也很相似，
    但它的默认行为是使用 'K' 作为顺序，并且默认情况下不会通过子类。

    示例
    --------
    >>> x = np.array([[1,2,3],[4,5,6]], order='F')

    >>> y = x.copy()

    >>> x.fill(0)

    >>> x
    array([[0, 0, 0],
           [0, 0, 0]])

    >>> y
    array([[1, 2, 3],
           [4, 5, 6]])

    >>> y.flags['C_CONTIGUOUS']
    True

    对于包含 Python 对象的数组（例如 dtype=object），
    复制是浅层的。新数组将包含相同的对象，如果该对象是可变的，
    这可能会导致意外行为：

    >>> a = np.array([1, 'm', [2, 3, 4]], dtype=object)
    >>> b = a.copy()
    >>> b[2][0] = 10
    >>> a
    array([1, 'm', list([10, 3, 4])], dtype=object)

    若要确保复制 `object` 数组中的所有元素，请使用 `copy.deepcopy`：

    >>> import copy
    >>> a = np.array([1, 'm', [2, 3, 4]], dtype=object)
    >>> c = copy.deepcopy(a)
    >>> c[2][0] = 10
    >>> c
    array([1, 'm', list([10, 3, 4])], dtype=object)
    >>> a
    array([1, 'm', list([2, 3, 4])], dtype=object)

    """))

add_newdoc('numpy._core.multiarray', 'ndarray', ('cumprod',
    """
    a.cumprod(axis=None, dtype=None, out=None)

    返回沿指定轴的元素的累积乘积。

    参考 `numpy.cumprod` 获取完整文档。

    另请参阅
    --------
    numpy.cumprod : 等效函数

    """))

add_newdoc('numpy._core.multiarray', 'ndarray', ('cumsum',
    """
    a.cumsum(axis=None, dtype=None, out=None)

    返回沿指定轴的元素的累积和。

    参考 `numpy.cumsum` 获取完整文档。

    另请参阅
    --------
    numpy.cumsum : 等效函数

    """))

add_newdoc('numpy._core.multiarray', 'ndarray', ('diagonal',
    """
    a.diagonal(offset=0, axis1=0, axis2=1)

    返回指定的对角线元素。在 NumPy 1.9 中，返回的数组是只读视图，
    而不是像以前的 NumPy 版本中的副本。在将来的版本中，只读限制将被移除。

    参考 :func:`numpy.diagonal` 获取完整文档。

    另请参阅
    --------
    numpy.diagonal

    """))
    # 打印多个分隔线，用于标记一段注释的开始或结束
    --------
    # 打印字符串 "numpy.diagonal : equivalent function"
    numpy.diagonal : equivalent function

    # 打印一个多行字符串，可能用于文档或注释
    """))
# 为 numpy.ndarray 类添加新的文档字符串，描述 dot 方法的功能
add_newdoc('numpy._core.multiarray', 'ndarray', ('dot'))

# 为 numpy.ndarray 类添加新的文档字符串，描述 dump 方法的功能及参数
add_newdoc('numpy._core.multiarray', 'ndarray', ('dump',
    """
    a.dump(file)

    Dump a pickle of the array to the specified file.
    The array can be read back with pickle.load or numpy.load.

    Parameters
    ----------
    file : str or Path
        A string naming the dump file.

        .. versionchanged:: 1.17.0
            `pathlib.Path` objects are now accepted.

    """))

# 为 numpy.ndarray 类添加新的文档字符串，描述 dumps 方法的功能及返回值
add_newdoc('numpy._core.multiarray', 'ndarray', ('dumps',
    """
    a.dumps()

    Returns the pickle of the array as a string.
    pickle.loads will convert the string back to an array.

    Parameters
    ----------
    None

    """))

# 为 numpy.ndarray 类添加新的文档字符串，描述 fill 方法的功能及参数
add_newdoc('numpy._core.multiarray', 'ndarray', ('fill',
    """
    a.fill(value)

    Fill the array with a scalar value.

    Parameters
    ----------
    value : scalar
        All elements of `a` will be assigned this value.

    Examples
    --------
    >>> a = np.array([1, 2])
    >>> a.fill(0)
    >>> a
    array([0, 0])
    >>> a = np.empty(2)
    >>> a.fill(1)
    >>> a
    array([1.,  1.])

    Fill expects a scalar value and always behaves the same as assigning
    to a single array element.  The following is a rare example where this
    distinction is important:

    >>> a = np.array([None, None], dtype=object)
    >>> a[0] = np.array(3)
    >>> a
    array([array(3), None], dtype=object)
    >>> a.fill(np.array(3))
    >>> a
    array([array(3), array(3)], dtype=object)

    Where other forms of assignments will unpack the array being assigned:

    >>> a[...] = np.array(3)
    >>> a
    array([3, 3], dtype=object)

    """))

# 为 numpy.ndarray 类添加新的文档字符串，描述 flatten 方法的功能及参数
add_newdoc('numpy._core.multiarray', 'ndarray', ('flatten',
    """
    a.flatten(order='C')

    Return a copy of the array collapsed into one dimension.

    Parameters
    ----------
    order : {'C', 'F', 'A', 'K'}, optional
        'C' means to flatten in row-major (C-style) order.
        'F' means to flatten in column-major (Fortran-
        style) order. 'A' means to flatten in column-major
        order if `a` is Fortran *contiguous* in memory,
        row-major order otherwise. 'K' means to flatten
        `a` in the order the elements occur in memory.
        The default is 'C'.

    Returns
    -------
    y : ndarray
        A copy of the input array, flattened to one dimension.

    See Also
    --------
    ravel : Return a flattened array.
    flat : A 1-D flat iterator over the array.

    Examples
    --------
    >>> a = np.array([[1,2], [3,4]])
    >>> a.flatten()
    array([1, 2, 3, 4])
    >>> a.flatten('F')
    array([1, 3, 2, 4])

    """))

# 为 numpy.ndarray 类添加新的文档字符串，描述 getfield 方法的功能及参数
add_newdoc('numpy._core.multiarray', 'ndarray', ('getfield',
    """
    a.getfield(dtype, offset=0)

    Returns a field of the given array as a certain type.

    A field is a view of the array data with a given data-type. The values in
    the view are determined by the given type and the offset into the current
    """))
    # 获取数组的字段（视图），返回一个新的数组，该数组包含了从指定偏移开始的指定数据类型的视图。
    # 视图的偏移量必须使得视图的数据类型适合于数组的数据类型；例如，如果数组的dtype是complex128，则每个元素占16字节。
    # 如果使用32位整数（4字节）来创建视图，则偏移量必须在0到12字节之间。
    #
    # 参数
    # ------
    # dtype : str 或 dtype
    #     视图的数据类型。视图的dtype大小不能大于数组本身的dtype大小。
    # offset : int
    #     开始元素视图之前要跳过的字节数。
    #
    # 示例
    # --------
    # >>> x = np.diag([1.+1.j]*2)
    # >>> x[1, 1] = 2 + 4.j
    # >>> x
    # array([[1.+1.j,  0.+0.j],
    #        [0.+0.j,  2.+4.j]])
    # >>> x.getfield(np.float64)
    # array([[1.,  0.],
    #        [0.,  2.]])
    #
    # 通过选择8字节的偏移量，我们可以选择数组的复数部分来创建视图：
    #
    # >>> x.getfield(np.float64, offset=8)
    # array([[1.,  0.],
    #        [0.,  4.]])
    """
# 向 numpy._core.multiarray 添加新的文档字符串，描述 ndarray 类的 item 方法
add_newdoc('numpy._core.multiarray', 'ndarray', ('item',
    """
    a.item(*args)

    Copy an element of an array to a standard Python scalar and return it.

    Parameters
    ----------
    \\*args : Arguments (variable number and type)

        * none: in this case, the method only works for arrays
          with one element (`a.size == 1`), which element is
          copied into a standard Python scalar object and returned.

        * int_type: this argument is interpreted as a flat index into
          the array, specifying which element to copy and return.

        * tuple of int_types: functions as does a single int_type argument,
          except that the argument is interpreted as an nd-index into the
          array.

    Returns
    -------
    z : Standard Python scalar object
        A copy of the specified element of the array as a suitable
        Python scalar

    Notes
    -----
    When the data type of `a` is longdouble or clongdouble, item() returns
    a scalar array object because there is no available Python scalar that
    would not lose information. Void arrays return a buffer object for item(),
    unless fields are defined, in which case a tuple is returned.

    `item` is very similar to a[args], except, instead of an array scalar,
    a standard Python scalar is returned. This can be useful for speeding up
    access to elements of the array and doing arithmetic on elements of the
    array using Python's optimized math.

    Examples
    --------
    >>> np.random.seed(123)
    >>> x = np.random.randint(9, size=(3, 3))
    >>> x
    array([[2, 2, 6],
           [1, 3, 6],
           [1, 0, 1]])
    >>> x.item(3)
    1
    >>> x.item(7)
    0
    >>> x.item((0, 1))
    2
    >>> x.item((2, 2))
    1

    For an array with object dtype, elements are returned as-is.

    >>> a = np.array([np.int64(1)], dtype=object)
    >>> a.item() #return np.int64
    np.int64(1)

    """))


# 向 numpy._core.multiarray 添加新的文档字符串，描述 ndarray 类的 max 方法
add_newdoc('numpy._core.multiarray', 'ndarray', ('max',
    """
    a.max(axis=None, out=None, keepdims=False, initial=<no value>, where=True)

    Return the maximum along a given axis.

    Refer to `numpy.amax` for full documentation.

    See Also
    --------
    numpy.amax : equivalent function

    """))


# 向 numpy._core.multiarray 添加新的文档字符串，描述 ndarray 类的 mean 方法
add_newdoc('numpy._core.multiarray', 'ndarray', ('mean',
    """
    a.mean(axis=None, dtype=None, out=None, keepdims=False, *, where=True)

    Returns the average of the array elements along given axis.

    Refer to `numpy.mean` for full documentation.

    See Also
    --------
    numpy.mean : equivalent function

    """))


# 向 numpy._core.multiarray 添加新的文档字符串，描述 ndarray 类的 min 方法
add_newdoc('numpy._core.multiarray', 'ndarray', ('min',
    """
    a.min(axis=None, out=None, keepdims=False, initial=<no value>, where=True)

    Return the minimum along a given axis.

    Refer to `numpy.amin` for full documentation.

    See Also
    --------
    numpy.amin : equivalent function

    """))


# 向 numpy._core.multiarray 添加新的文档字符串，描述 ndarray 类的 nonzero 方法
add_newdoc('numpy._core.multiarray', 'ndarray', ('nonzero',
    """
    a.nonzero()

    Return the indices of the elements that are non-zero.

    """))
    # 导入 numpy 库中的 nonzero 函数，用于返回数组中非零元素的索引
    from numpy import (nonzero as nonzeros)

    # 返回 numpy.nonzero 函数的文档字符串，描述如何使用该函数
    """
    Return the indices of the elements that are non-zero.

    Refer to `numpy.nonzero` for full documentation.

    See Also
    --------
    numpy.nonzero : equivalent function
    """
# 向给定的 numpy ndarray 类添加新的文档字符串
add_newdoc('numpy._core.multiarray', 'ndarray', ('prod',
    """
    a.prod(axis=None, dtype=None, out=None, keepdims=False,
        initial=1, where=True)

    返回数组元素沿给定轴的乘积

    详细文档参考 `numpy.prod`。

    参见
    --------
    numpy.prod : 等效函数

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('put',
    """
    a.put(indices, values, mode='raise')

    对所有 `n` 在 indices 中的 `a.flat[n]` 设置为 values[n]。

    详细文档参考 `numpy.put`。

    参见
    --------
    numpy.put : 等效函数

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('ravel',
    """
    a.ravel([order])

    返回扁平化的数组。

    详细文档参考 `numpy.ravel`。

    参见
    --------
    numpy.ravel : 等效函数

    ndarray.flat : 数组上的扁平迭代器。

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('repeat',
    """
    a.repeat(repeats, axis=None)

    重复数组的元素。

    详细文档参考 `numpy.repeat`。

    参见
    --------
    numpy.repeat : 等效函数

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('reshape',
    """
    a.reshape(shape, /, *, order='C', copy=None)

    返回包含相同数据的具有新形状的数组。

    详细文档参考 `numpy.reshape`。

    参见
    --------
    numpy.reshape : 等效函数

    注意
    -----
    与自由函数 `numpy.reshape` 不同，这个 `ndarray` 上的方法允许将形状参数的元素作为单独的参数传递。
    例如，``a.reshape(10, 11)`` 等效于 ``a.reshape((10, 11))``。

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('resize',
    """
    a.resize(new_shape, refcheck=True)

    原地改变数组的形状和大小。

    参数
    ----------
    new_shape : int 元组，或 `n` 个整数
        重新调整后的数组形状。
    refcheck : bool, 可选
        如果为 False，将不检查引用计数。默认为 True。

    返回
    -------
    None

    异常
    ------
    ValueError
        如果 `a` 不拥有自己的数据或存在对其的引用或视图，并且数据内存必须更改。
        仅适用于 PyPy：如果必须更改数据内存，将始终引发此异常，因为没有可靠的方法确定是否存在对其的引用或视图。
    SystemError
        如果指定了 `order` 关键字参数。这种行为是 NumPy 中的一个 bug。

    参见
    --------
    resize : 返回具有指定形状的新数组。

    注意
    -----
    如果必要，此函数将重新分配数据区域的空间。

    只有连续数组（内存中连续的数据元素）才能调整大小。

    引用计数检查的目的是确保你
    do not use this array as a buffer for another Python object and then
    reallocate the memory. However, reference counts can increase in
    other ways so if you are sure that you have not shared the memory
    for this array with another Python object, then you may safely set
    `refcheck` to False.

    Examples
    --------
    Shrinking an array: array is flattened (in the order that the data are
    stored in memory), resized, and reshaped:

    >>> a = np.array([[0, 1], [2, 3]], order='C')
    >>> a.resize((2, 1))
    >>> a
    array([[0],
           [1]])

    >>> a = np.array([[0, 1], [2, 3]], order='F')
    >>> a.resize((2, 1))
    >>> a
    array([[0],
           [2]])

    Enlarging an array: as above, but missing entries are filled with zeros:

    >>> b = np.array([[0, 1], [2, 3]])
    >>> b.resize(2, 3) # new_shape parameter doesn't have to be a tuple
    >>> b
    array([[0, 1, 2],
           [3, 0, 0]])

    Referencing an array prevents resizing...

    >>> c = a
    >>> a.resize((1, 1))
    Traceback (most recent call last):
    ...
    ValueError: cannot resize an array that references or is referenced ...

    Unless `refcheck` is False:

    >>> a.resize((1, 1), refcheck=False)
    >>> a
    array([[0]])
    >>> c
    array([[0]])
add_newdoc('numpy._core.multiarray', 'ndarray', ('round',
    """
    a.round(decimals=0, out=None)

    Return `a` with each element rounded to the given number of decimals.

    Refer to `numpy.around` for full documentation.

    See Also
    --------
    numpy.around : equivalent function

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('searchsorted',
    """
    a.searchsorted(v, side='left', sorter=None)

    Find indices where elements of v should be inserted in a to maintain order.

    For full documentation, see `numpy.searchsorted`

    See Also
    --------
    numpy.searchsorted : equivalent function

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('setfield',
    """
    a.setfield(val, dtype, offset=0)

    Put a value into a specified place in a field defined by a data-type.

    Place `val` into `a`'s field defined by `dtype` and beginning `offset`
    bytes into the field.

    Parameters
    ----------
    val : object
        Value to be placed in field.
    dtype : dtype object
        Data-type of the field in which to place `val`.
    offset : int, optional
        The number of bytes into the field at which to place `val`.

    Returns
    -------
    None

    See Also
    --------
    getfield

    Examples
    --------
    >>> x = np.eye(3)
    >>> x.getfield(np.float64)
    array([[1.,  0.,  0.],
           [0.,  1.,  0.],
           [0.,  0.,  1.]])
    >>> x.setfield(3, np.int32)
    >>> x.getfield(np.int32)
    array([[3, 3, 3],
           [3, 3, 3],
           [3, 3, 3]], dtype=int32)
    >>> x
    array([[1.0e+000, 1.5e-323, 1.5e-323],
           [1.5e-323, 1.0e+000, 1.5e-323],
           [1.5e-323, 1.5e-323, 1.0e+000]])
    >>> x.setfield(np.eye(3), np.int32)
    >>> x
    array([[1.,  0.,  0.],
           [0.,  1.,  0.],
           [0.,  0.,  1.]])

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('setflags',
    """
    a.setflags(write=None, align=None, uic=None)

    Set array flags WRITEABLE, ALIGNED, WRITEBACKIFCOPY,
    respectively.

    These Boolean-valued flags affect how numpy interprets the memory
    area used by `a` (see Notes below). The ALIGNED flag can only
    be set to True if the data is actually aligned according to the type.
    The WRITEBACKIFCOPY flag can never be set
    to True. The flag WRITEABLE can only be set to True if the array owns its
    own memory, or the ultimate owner of the memory exposes a writeable buffer
    interface, or is a string. (The exception for string is made so that
    unpickling can be done without copying memory.)

    Parameters
    ----------
    write : bool, optional
        Describes whether or not `a` can be written to.
    align : bool, optional
        Describes whether or not `a` is aligned properly for its type.
    uic : bool, optional
        Describes whether or not `a` is a copy of another "base" array.

    Notes
    -----
    Array flags provide information about how the memory area used
    """))
    # 循环中的每个数组标志位于如何解释数组的上下文中。有7个布尔标志在使用中，只有三个可以由用户更改：
    # WRITEBACKIFCOPY、WRITEABLE 和 ALIGNED。

    # WRITEABLE (W)：数据区域可以被写入；

    # ALIGNED (A)：数据和步幅适合硬件（由编译器确定）；

    # WRITEBACKIFCOPY (X)：这个数组是某个其他数组的副本（由.base引用）。当调用C-API函数
    # PyArray_ResolveWritebackIfCopy时，基础数组将使用此数组的内容更新。

    # 所有标志可以使用单个（大写）字母以及完整名称访问。

    # 示例
    # --------
    # >>> y = np.array([[3, 1, 7],
    # ...               [2, 0, 0],
    # ...               [8, 5, 9]])
    # >>> y
    # array([[3, 1, 7],
    #        [2, 0, 0],
    #        [8, 5, 9]])
    # >>> y.flags
    #   C_CONTIGUOUS : True
    #   F_CONTIGUOUS : False
    #   OWNDATA : True
    #   WRITEABLE : True
    #   ALIGNED : True
    #   WRITEBACKIFCOPY : False
    # >>> y.setflags(write=0, align=0)
    # >>> y.flags
    #   C_CONTIGUOUS : True
    #   F_CONTIGUOUS : False
    #   OWNDATA : True
    #   WRITEABLE : False
    #   ALIGNED : False
    #   WRITEBACKIFCOPY : False
    # >>> y.setflags(uic=1)
    # Traceback (most recent call last):
    #   File "<stdin>", line 1, in <module>
    # ValueError: cannot set WRITEBACKIFCOPY flag to True
# 在 `numpy._core.multiarray` 模块中为 `ndarray` 类型的对象添加新的文档字符串 'sort'
add_newdoc('numpy._core.multiarray', 'ndarray', ('sort',
    """
    a.sort(axis=-1, kind=None, order=None)

    Sort an array in-place. Refer to `numpy.sort` for full documentation.

    Parameters
    ----------
    axis : int, optional
        要排序的轴。默认为 -1，表示沿着最后一个轴排序。
    kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
        排序算法。默认为 'quicksort'。注意，'stable' 和 'mergesort' 在实现上都使用了 timsort，
        具体实现会根据数据类型变化而变化。为了向后兼容，保留 'mergesort' 选项。

        .. versionchanged:: 1.15.0
           添加了 'stable' 选项。

    order : str or list of str, optional
        当 `a` 是一个带有字段定义的数组时，此参数指定首先比较哪些字段、第二个字段等等。可以指定单个字段为字符串，
        不需要指定所有字段，但未指定的字段仍将按照它们在 dtype 中出现的顺序使用来解决平局。

    See Also
    --------
    numpy.sort : 返回数组的排序副本。
    numpy.argsort : 间接排序。
    numpy.lexsort : 多关键字的间接稳定排序。
    numpy.searchsorted : 在排序数组中查找元素。
    numpy.partition: 部分排序。

    Notes
    -----
    参见 `numpy.sort` 获取不同排序算法的注意事项。

    Examples
    --------
    >>> a = np.array([[1,4], [3,1]])
    >>> a.sort(axis=1)
    >>> a
    array([[1, 4],
           [1, 3]])
    >>> a.sort(axis=0)
    >>> a
    array([[1, 3],
           [1, 4]])

    使用 `order` 关键字来指定结构化数组排序时使用的字段：

    >>> a = np.array([('a', 2), ('c', 1)], dtype=[('x', 'S1'), ('y', int)])
    >>> a.sort(order='y')
    >>> a
    array([(b'c', 1), (b'a', 2)],
          dtype=[('x', 'S1'), ('y', '<i8')])

    """))

# 在 `numpy._core.multiarray` 模块中为 `ndarray` 类型的对象添加新的文档字符串 'partition'
add_newdoc('numpy._core.multiarray', 'ndarray', ('partition',
    """
    a.partition(kth, axis=-1, kind='introselect', order=None)

    将数组中的元素部分排序，使得第 k 个位置的元素在排序后的数组中处于应有的位置。
    在输出数组中，所有小于第 k 个元素的元素位于这个元素的左侧，所有等于或大于的元素位于右侧。
    输出数组中第 k 个元素两侧分区中的元素顺序是不确定的。

    .. versionadded:: 1.8.0

    Parameters
    ----------
    kth : int or sequence of ints
        指定每个轴上要分隔的索引或索引序列。
    axis : int, optional
        要排序的轴。默认为 -1，表示沿着最后一个轴排序。
    kind : {'introselect'}, optional
        部分排序算法。默认为 'introselect'。
    order : str or list of str, optional
        当 `a` 是一个带有字段定义的数组时，此参数指定首先比较哪些字段、第二个字段等等。可以指定单个字段为字符串，
        不需要指定所有字段，但未指定的字段仍将按照它们在 dtype 中出现的顺序使用来解决平局。

    """))
    # kth参数：指定要进行分区的元素索引。被指定的元素将移动到最终排序位置，小于它的元素将放在它前面，大于等于它的元素将放在它后面。
    # 如果提供了一个索引序列，将同时对所有指定索引的元素进行分区。
    # 注意：在1.22.0版本之后，作为索引传递布尔值已被弃用。

    axis : int, optional
        # 排序的轴向，默认为-1，即沿着最后一个轴进行排序。

    kind : {'introselect'}, optional
        # 选择算法，默认为'introselect'。

    order : str or list of str, optional
        # 当数组`a`具有定义的字段时，此参数指定首先比较哪些字段、第二个字段等。
        # 可以将单个字段指定为字符串，不需要指定所有字段，但未指定的字段仍将按照它们在dtype中出现的顺序使用，用于打破平局。

    See Also
    --------
    numpy.partition : 返回数组的分区副本。
    argpartition : 间接分区。
    sort : 完全排序。

    Notes
    -----
    参见 ``np.partition`` 获取有关不同算法的注释。

    Examples
    --------
    >>> a = np.array([3, 4, 2, 1])
    >>> a.partition(3)
    >>> a
    array([2, 1, 3, 4]) # 结果可能有所不同

    >>> a.partition((1, 3))
    >>> a
    array([1, 2, 3, 4])
# 向numpy.ndarray类型的对象添加新的文档字符串，描述其squeeze方法的作用
add_newdoc('numpy._core.multiarray', 'ndarray', ('squeeze',
    """
    a.squeeze(axis=None)

    Remove axes of length one from `a`.

    Refer to `numpy.squeeze` for full documentation.

    See Also
    --------
    numpy.squeeze : equivalent function

    """))


# 向numpy.ndarray类型的对象添加新的文档字符串，描述其std方法的作用
add_newdoc('numpy._core.multiarray', 'ndarray', ('std',
    """
    a.std(axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True)

    Returns the standard deviation of the array elements along given axis.

    Refer to `numpy.std` for full documentation.

    See Also
    --------
    numpy.std : equivalent function

    """))


# 向numpy.ndarray类型的对象添加新的文档字符串，描述其sum方法的作用
add_newdoc('numpy._core.multiarray', 'ndarray', ('sum',
    """
    a.sum(axis=None, dtype=None, out=None, keepdims=False, initial=0, where=True)

    Return the sum of the array elements over the given axis.

    Refer to `numpy.sum` for full documentation.

    See Also
    --------
    numpy.sum : equivalent function

    """))


# 向numpy.ndarray类型的对象添加新的文档字符串，描述其swapaxes方法的作用
add_newdoc('numpy._core.multiarray', 'ndarray', ('swapaxes',
    """
    a.swapaxes(axis1, axis2)

    Return a view of the array with `axis1` and `axis2` interchanged.

    Refer to `numpy.swapaxes` for full documentation.

    See Also
    --------
    numpy.swapaxes : equivalent function

    """))


# 向numpy.ndarray类型的对象添加新的文档字符串，描述其take方法的作用
add_newdoc('numpy._core.multiarray', 'ndarray', ('take',
    """
    a.take(indices, axis=None, out=None, mode='raise')

    Return an array formed from the elements of `a` at the given indices.

    Refer to `numpy.take` for full documentation.

    See Also
    --------
    numpy.take : equivalent function

    """))


# 向numpy.ndarray类型的对象添加新的文档字符串，描述其tofile方法的作用
add_newdoc('numpy._core.multiarray', 'ndarray', ('tofile',
    """
    a.tofile(fid, sep="", format="%s")

    Write array to a file as text or binary (default).

    Data is always written in 'C' order, independent of the order of `a`.
    The data produced by this method can be recovered using the function
    fromfile().

    Parameters
    ----------
    fid : file or str or Path
        An open file object, or a string containing a filename.

        .. versionchanged:: 1.17.0
            `pathlib.Path` objects are now accepted.

    sep : str
        Separator between array items for text output.
        If "" (empty), a binary file is written, equivalent to
        ``file.write(a.tobytes())``.
    format : str
        Format string for text file output.
        Each entry in the array is formatted to text by first converting
        it to the closest Python type, and then using "format" % item.

    Notes
    -----
    This is a convenience function for quick storage of array data.
    Information on endianness and precision is lost, so this method is not a
    good choice for files intended to archive data or transport data between
    machines with different endianness. Some of these problems can be overcome
    by outputting the data as text files, at the expense of speed and file
    size.

    When fid is a file object, array contents are directly written to the
    specified file.

    """))
    # 导入标准库中的sys模块
    import sys
    # 将fileinput这个模块导入到当前的命名空间中
    import fileinput as m
    # 导入os模块中的fstat()和stat()方法
    from os import fstat as stat, stat as stat
    # 从codecs模块导入编码解码器函数
    from codecs import open as _open
    # 从codecs模块导入编码解码器函数
    from codecs import open as _open
    # 从codecs模块导入编码解码器函数
    from codecs import open as _open
add_newdoc('numpy._core.multiarray', 'ndarray', ('tolist',
    """
    a.tolist()

    将数组转换为一个深度为 ``a.ndim`` 的嵌套列表，其中包含 Python 标量。

    返回数组数据的副本，作为（嵌套的）Python 列表。
    数据项通过 `~numpy.ndarray.item` 函数转换为最接近的兼容的内置 Python 类型。

    如果 ``a.ndim`` 是 0，则由于嵌套列表的深度为 0，它实际上不是列表，而是一个简单的 Python 标量。

    参数
    ----------
    none

    返回
    -------
    y : object, 或者对象列表，或者对象的嵌套列表，等等
        可能嵌套的数组元素列表。

    注意
    -----
    可以通过 ``a = np.array(a.tolist())`` 重新创建数组，尽管有时可能会丢失精度。

    示例
    --------
    对于一维数组，``a.tolist()`` 几乎与 ``list(a)`` 相同，
    但 ``tolist`` 将 numpy 标量转换为 Python 标量：

    >>> a = np.uint32([1, 2])
    >>> a_list = list(a)
    >>> a_list
    [1, 2]
    >>> type(a_list[0])
    <class 'numpy.uint32'>
    >>> a_tolist = a.tolist()
    >>> a_tolist
    [1, 2]
    >>> type(a_tolist[0])
    <class 'int'>

    另外，对于二维数组，``tolist`` 递归应用：

    >>> a = np.array([[1, 2], [3, 4]])
    >>> list(a)
    [array([1, 2]), array([3, 4])]
    >>> a.tolist()
    [[1, 2], [3, 4]]

    递归的基本情况是 0 维数组：

    >>> a = np.array(1)
    >>> list(a)
    Traceback (most recent call last):
      ...
    TypeError: iteration over a 0-d array
    >>> a.tolist()
    1
    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('tobytes', """
    a.tobytes(order='C')

    构造包含数组原始数据字节的 Python 字节对象。

    构造一个 Python 字节对象，显示数据内存的原始内容的副本。默认情况下，以 C 顺序生成字节对象。
    此行为由 ``order`` 参数控制。

    .. versionadded:: 1.9.0

    参数
    ----------
    order : {'C', 'F', 'A'}, 可选
        控制字节对象的内存布局。'C' 表示 C 顺序，'F' 表示 Fortran 顺序，
        'A'（缩写 *Any*）表示如果 `a` 是 Fortran 连续的则使用 'F'，否则使用 'C'。默认为 'C'。

    返回
    -------
    s : bytes
        显示 `a` 原始数据的 Python 字节对象。

    参见
    --------
    frombuffer
        该操作的反向操作，从 Python 字节构造一个一维数组。

    示例
    --------
    >>> x = np.array([[0, 1], [2, 3]], dtype='<u2')
    >>> x.tobytes()
    b'\\x00\\x00\\x01\\x00\\x02\\x00\\x03\\x00'
    >>> x.tobytes('C') == x.tobytes()
    True
    >>> x.tobytes('F')
    b'\\x00\\x00\\x02\\x00\\x01\\x00\\x03\\x00'

    """))


add_newdoc('numpy._core.multiarray', 'ndarray', ('tostring', r"""
    a.tostring(order='C')

    与 `~ndarray.tobytes` 完全相同行为的兼容别名。

    """))
    Despite its name, it returns :class:`bytes` not :class:`str`\ s.

    .. deprecated:: 1.19.0
    """
    尽管它的名字中含有`str`，但它返回的是 :class:`bytes` 而不是 :class:`str`。

    这段文档标记已经过时，从版本 1.19.0 开始不推荐使用。
    """
# 添加新的文档字符串到 numpy._core.multiarray.ndarray 的 'trace' 属性
add_newdoc('numpy._core.multiarray', 'ndarray', ('trace',
    """
    a.trace(offset=0, axis1=0, axis2=1, dtype=None, out=None)

    返回数组沿对角线的和。

    详细文档请参考 `numpy.trace`。

    另请参阅
    --------
    numpy.trace : 等效函数

    """))


# 添加新的文档字符串到 numpy._core.multiarray.ndarray 的 'transpose' 属性
add_newdoc('numpy._core.multiarray', 'ndarray', ('transpose',
    """
    a.transpose(*axes)

    返回数组轴的转置视图。

    详细文档请参考 `numpy.transpose`。

    参数
    ----------
    axes : None、整数元组或 n 个整数

     * None 或无参数：反转轴的顺序。
     
     * 整数元组：元组中的第 j 个位置的 i 表示数组的第 i 个轴变成转置数组的第 j 个轴。
     
     * n 个整数：与包含相同整数的 n 元组相同（此形式仅作为对元组形式的“便利”替代）。

    返回
    -------
    p : ndarray
        数组的轴适当排列的视图。

    另请参阅
    --------
    transpose : 等效函数。
    ndarray.T : 返回数组转置的数组属性。
    ndarray.reshape : 在不改变数据的情况下为数组提供新形状。

    示例
    --------
    >>> a = np.array([[1, 2], [3, 4]])
    >>> a
    array([[1, 2],
           [3, 4]])
    >>> a.transpose()
    array([[1, 3],
           [2, 4]])
    >>> a.transpose((1, 0))
    array([[1, 3],
           [2, 4]])
    >>> a.transpose(1, 0)
    array([[1, 3],
           [2, 4]])

    >>> a = np.array([1, 2, 3, 4])
    >>> a
    array([1, 2, 3, 4])
    >>> a.transpose()
    array([1, 2, 3, 4])

    """))


# 添加新的文档字符串到 numpy._core.multiarray.ndarray 的 'var' 属性
add_newdoc('numpy._core.multiarray', 'ndarray', ('var',
    """
    a.var(axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True)

    返回数组元素沿给定轴的方差。

    详细文档请参考 `numpy.var`。

    另请参阅
    --------
    numpy.var : 等效函数

    """))


# 添加新的文档字符串到 numpy._core.multiarray.ndarray 的 'view' 属性
add_newdoc('numpy._core.multiarray', 'ndarray', ('view',
    """
    a.view([dtype][, type])

    返回具有相同数据的数组的新视图。

    .. note::
        传递 None 作为 ``dtype`` 与省略该参数不同，因为前者调用 ``dtype(None)``，其是 ``dtype('float64')`` 的别名。

    参数
    ----------
    dtype : 数据类型或 ndarray 子类，可选
        返回视图的数据类型描述符，例如 float32 或 int16。
        省略此参数将使视图具有与 `a` 相同的数据类型。
        此参数还可以指定为 ndarray 子类，从而指定返回对象的类型（这相当于设置 ``type`` 参数）。
    type : Python 类型，可选
        返回视图的类型，例如 ndarray 或 matrix。再次省略该参数将保留类型。

    注记
    -----
    # 使用 a.view() 方法有两种不同的方式：
    
    # a.view(some_dtype) 或者 a.view(dtype=some_dtype) 构造一个使用不同数据类型的数组内存视图。
    # 这可能会导致内存字节的重新解释。
    
    # a.view(ndarray_subclass) 或者 a.view(type=ndarray_subclass) 返回一个 `ndarray_subclass` 的实例，
    # 它查看同一个数组（相同的形状、数据类型等）。这不会导致内存的重新解释。
    
    # 对于 a.view(some_dtype)，如果 some_dtype 的每个条目的字节数与之前的数据类型不同
    # （例如，将常规数组转换为结构化数组），那么 a 的最后一个轴必须是连续的。
    # 这个轴将在结果中被重新调整大小。
    
    # .. versionchanged:: 1.23.0
    #    现在只有最后一个轴需要是连续的。之前整个数组必须是 C 连续的。
    
    # 示例
    # -------
    # 创建一个结构化数组并使用不同类型和数据类型查看数组数据：
    
    # 创建一个结构化数组的视图，以便可以用于计算
    
    # 改变视图会改变底层数组
    
    # 使用视图将数组转换为 recarray：
    
    # 视图共享数据：
    
    # 改变数据类型大小的视图（每个条目的字节数）通常不适用于由切片、转置、Fortran 排序等定义的数组。
    
    # 然而，对于最后一个轴是连续的数组，即使其余轴不是 C 连续的，改变数据类型的视图也是完全可以的：
    <BLANKLINE>
    # 创建一个多维数组（矩阵），数据类型为 int16（即16位整数）
    np.array([[[2312, 2826],
               [5396, 5910]]], dtype=int16)

    """))
##############################################################################
#
# umath functions
#
##############################################################################

# 定义一个新的文档字符串并添加到 numpy._core.umath 模块中的 frompyfunc 函数
add_newdoc('numpy._core.umath', 'frompyfunc',
    """
    frompyfunc(func, /, nin, nout, *[, identity])

    Takes an arbitrary Python function and returns a NumPy ufunc.

    Can be used, for example, to add broadcasting to a built-in Python
    function (see Examples section).

    Parameters
    ----------
    func : Python function object
        An arbitrary Python function.
    nin : int
        The number of input arguments.
    nout : int
        The number of objects returned by `func`.
    identity : object, optional
        The value to use for the `~numpy.ufunc.identity` attribute of the resulting
        object. If specified, this is equivalent to setting the underlying
        C ``identity`` field to ``PyUFunc_IdentityValue``.
        If omitted, the identity is set to ``PyUFunc_None``. Note that this is
        _not_ equivalent to setting the identity to ``None``, which implies the
        operation is reorderable.

    Returns
    -------
    out : ufunc
        Returns a NumPy universal function (``ufunc``) object.

    See Also
    --------
    vectorize : Evaluates pyfunc over input arrays using broadcasting rules of numpy.

    Notes
    -----
    The returned ufunc always returns PyObject arrays.

    Examples
    --------
    Use frompyfunc to add broadcasting to the Python function ``oct``:

    >>> oct_array = np.frompyfunc(oct, 1, 1)
    >>> oct_array(np.array((10, 30, 100)))
    array(['0o12', '0o36', '0o144'], dtype=object)
    >>> np.array((oct(10), oct(30), oct(100))) # for comparison
    array(['0o12', '0o36', '0o144'], dtype='<U5')

    """)


##############################################################################
#
# compiled_base functions
#
##############################################################################

# 将文档字符串添加到 numpy._core.multiarray 模块中的 add_docstring 函数
add_newdoc('numpy._core.multiarray', 'add_docstring',
    """
    add_docstring(obj, docstring)

    Add a docstring to a built-in obj if possible.
    If the obj already has a docstring raise a RuntimeError
    If this routine does not know how to add a docstring to the object
    raise a TypeError
    """)

# 将新的文档字符串添加到 numpy._core.umath 模块中的 _add_newdoc_ufunc 函数
add_newdoc('numpy._core.umath', '_add_newdoc_ufunc',
    """
    add_ufunc_docstring(ufunc, new_docstring)

    Replace the docstring for a ufunc with new_docstring.
    This method will only work if the current docstring for
    the ufunc is NULL. (At the C level, i.e. when ufunc->doc is NULL.)

    Parameters
    ----------
    ufunc : numpy.ufunc
        A ufunc whose current doc is NULL.
    new_docstring : string
        The new docstring for the ufunc.

    Notes
    -----
    This method allocates memory for new_docstring on
    the heap. Technically this creates a memory leak, since this
    memory will not be reclaimed until the end of the program
    even if the ufunc itself is removed. However this will only
    be a problem if the user is repeatedly creating ufuncs with
    no documentation, adding documentation via add_newdoc_ufunc,
    and then throwing away the ufunc.
    """



# 多行字符串，包含了对于 ufunc 的文档注释说明。这段文档描述了即使 ufunc 本身被移除，
# 其文档信息仍然可以保留。然而，这只有在用户重复创建没有文档的 ufunc，然后通过
# add_newdoc_ufunc 方法添加文档，并且之后丢弃了 ufunc 的情况下才会成为问题。
# 将新的文档添加到numpy._core.multiarray模块，定义函数get_handler_name
add_newdoc('numpy._core.multiarray', 'get_handler_name',
    """
    get_handler_name(a: ndarray) -> str,None

    Return the name of the memory handler used by `a`. If not provided, return
    the name of the memory handler that will be used to allocate data for the
    next `ndarray` in this context. May return None if `a` does not own its
    memory, in which case you can traverse ``a.base`` for a memory handler.
    """)

# 将新的文档添加到numpy._core.multiarray模块，定义函数get_handler_version
add_newdoc('numpy._core.multiarray', 'get_handler_version',
    """
    get_handler_version(a: ndarray) -> int,None

    Return the version of the memory handler used by `a`. If not provided,
    return the version of the memory handler that will be used to allocate data
    for the next `ndarray` in this context. May return None if `a` does not own
    its memory, in which case you can traverse ``a.base`` for a memory handler.
    """)

# 将新的文档添加到numpy._core._multiarray_umath模块，定义函数_array_converter
add_newdoc('numpy._core._multiarray_umath', '_array_converter',
    """
    _array_converter(*array_likes)

    Helper to convert one or more objects to arrays.  Integrates machinery
    to deal with the ``result_type`` and ``__array_wrap__``.

    The reason for this is that e.g. ``result_type`` needs to convert to arrays
    to find the ``dtype``.  But converting to an array before calling
    ``result_type`` would incorrectly "forget" whether it was a Python int,
    float, or complex.
    """)

# 将新的文档添加到numpy._core._multiarray_umath模块，定义常量scalar_input
add_newdoc(
    'numpy._core._multiarray_umath', '_array_converter', ('scalar_input',
    """
    A tuple which indicates for each input whether it was a scalar that
    was coerced to a 0-D array (and was not already an array or something
    converted via a protocol like ``__array__()``).
    """))

# 将新的文档添加到numpy._core._multiarray_umath模块，定义函数as_arrays
add_newdoc('numpy._core._multiarray_umath', '_array_converter', ('as_arrays',
    """
    as_arrays(/, subok=True, pyscalars="convert_if_no_array")

    Return the inputs as arrays or scalars.

    Parameters
    ----------
    subok : True or False, optional
        Whether array subclasses are preserved.
    pyscalars : {"convert", "preserve", "convert_if_no_array"}, optional
        To allow NEP 50 weak promotion later, it may be desirable to preserve
        Python scalars.  As default, these are preserved unless all inputs
        are Python scalars.  "convert" enforces an array return.
    """))

# 将新的文档添加到numpy._core._multiarray_umath模块，定义函数result_type
add_newdoc('numpy._core._multiarray_umath', '_array_converter', ('result_type',
    """result_type(/, extra_dtype=None, ensure_inexact=False)

    Find the ``result_type`` just as ``np.result_type`` would, but taking
    into account that the original inputs (before converting to an array) may
    have been Python scalars with weak promotion.

    Parameters
    ----------
    extra_dtype : dtype instance or class
        An additional DType or dtype instance to promote (e.g. could be used
        to ensure the result precision is at least float32).
    """)
    ensure_inexact : True or False
        # 确保结果为非精确浮点数（或复数），替换 arr * 1. 或 result_type(..., 0.0) 模式的操作。
        当设为 True 时，确保替换上述模式得到浮点数（或复数）结果。
add_newdoc('numpy._core._multiarray_umath', '_array_converter', ('wrap',
    """
    wrap(arr, /, to_scalar=None)

    Call ``__array_wrap__`` on ``arr`` if ``arr`` is not the same subclass
    as the input the ``__array_wrap__`` method was retrieved from.

    Parameters
    ----------
    arr : ndarray
        The object to be wrapped. Normally an ndarray or subclass,
        although for backward compatibility NumPy scalars are also accepted
        (these will be converted to a NumPy array before being passed on to
        the ``__array_wrap__`` method).
    to_scalar : {True, False, None}, optional
        When ``True`` will convert a 0-d array to a scalar via ``result[()]``
        (with a fast-path for non-subclasses).  If ``False`` the result should
        be an array-like (as ``__array_wrap__`` is free to return a non-array).
        By default (``None``), a scalar is returned if all inputs were scalar.
    """))


add_newdoc('numpy._core.multiarray', '_get_madvise_hugepage',
    """
    _get_madvise_hugepage() -> bool

    Get use of ``madvise (2)`` MADV_HUGEPAGE support when
    allocating the array data. Returns the currently set value.
    See `global_state` for more information.
    """)

add_newdoc('numpy._core.multiarray', '_set_madvise_hugepage',
    """
    _set_madvise_hugepage(enabled: bool) -> bool

    Set  or unset use of ``madvise (2)`` MADV_HUGEPAGE support when
    allocating the array data. Returns the previously set value.
    See `global_state` for more information.
    """)


##############################################################################
#
# Documentation for ufunc attributes and methods
#
##############################################################################


##############################################################################
#
# ufunc object
#
##############################################################################

add_newdoc('numpy._core', 'ufunc',
    """
    Functions that operate element by element on whole arrays.

    To see the documentation for a specific ufunc, use `info`.  For
    example, ``np.info(np.sin)``.  Because ufuncs are written in C
    (for speed) and linked into Python with NumPy's ufunc facility,
    Python's help() function finds this page whenever help() is called
    on a ufunc.

    A detailed explanation of ufuncs can be found in the docs for :ref:`ufuncs`.

    **Calling ufuncs:** ``op(*x[, out], where=True, **kwargs)``

    Apply `op` to the arguments `*x` elementwise, broadcasting the arguments.

    The broadcasting rules are:

    * Dimensions of length 1 may be prepended to either array.
    * Arrays may be repeated along dimensions of length 1.

    Parameters
    ----------
    *x : array_like
        Input arrays.

    """)
    # out参数：用于指定结果的存储位置，可以是一个数组对象或者多个数组组成的元组。
    # 如果提供了out参数，则它必须具有与输入广播后相匹配的形状。
    # 如果作为关键字参数提供的是一个数组元组，则其长度必须等于输出的数量；使用None表示未初始化的输出将由ufunc分配。
    out : ndarray, None, or tuple of ndarray and None, optional
    
    # where参数：指定条件，它会被广播到输入数组上。在条件为True的位置，out数组将设置为ufunc的结果；其他位置，out数组将保留其原始值。
    # 注意，如果通过默认的out=None创建了未初始化的out数组，在其中条件为False的位置将保持未初始化状态。
    where : array_like, optional
    
    # **kwargs：其他关键字参数，详见ufunc文档。
    **kwargs
    
    # 返回值：返回一个数组或数组元组r。
    # 如果提供了out参数，则返回out；否则，r将被分配并可能包含未初始化的值。
    # 如果函数有多个输出，则结果将是一个数组元组。
    Returns
    -------
    r : ndarray or tuple of ndarray
##############################################################################
#
# ufunc attributes
#
##############################################################################

# 添加新的文档字符串到 numpy._core.ufunc 对象，定义了 'identity' 属性
add_newdoc('numpy._core', 'ufunc', ('identity',
    """
    The identity value.

    Data attribute containing the identity element for the ufunc,
    if it has one. If it does not, the attribute value is None.

    Examples
    --------
    >>> np.add.identity
    0
    >>> np.multiply.identity
    1
    >>> np.power.identity
    1
    >>> print(np.exp.identity)
    None
    """))

# 添加新的文档字符串到 numpy._core.ufunc 对象，定义了 'nargs' 属性
add_newdoc('numpy._core', 'ufunc', ('nargs',
    """
    The number of arguments.

    Data attribute containing the number of arguments the ufunc takes, including
    optional ones.

    Notes
    -----
    Typically this value will be one more than what you might expect
    because all ufuncs take  the optional "out" argument.

    Examples
    --------
    >>> np.add.nargs
    3
    >>> np.multiply.nargs
    3
    >>> np.power.nargs
    3
    >>> np.exp.nargs
    2
    """))

# 添加新的文档字符串到 numpy._core.ufunc 对象，定义了 'nin' 属性
add_newdoc('numpy._core', 'ufunc', ('nin',
    """
    The number of inputs.

    Data attribute containing the number of arguments the ufunc treats as input.

    Examples
    --------
    >>> np.add.nin
    2
    >>> np.multiply.nin
    2
    >>> np.power.nin
    2
    >>> np.exp.nin
    1
    """))

# 添加新的文档字符串到 numpy._core.ufunc 对象，定义了 'nout' 属性
add_newdoc('numpy._core', 'ufunc', ('nout',
    """
    The number of outputs.

    Data attribute containing the number of arguments the ufunc treats as output.

    Notes
    -----
    Since all ufuncs can take output arguments, this will always be at least 1.

    Examples
    --------
    >>> np.add.nout
    1
    >>> np.multiply.nout
    1
    >>> np.power.nout
    1
    >>> np.exp.nout
    1

    """))

# 添加新的文档字符串到 numpy._core.ufunc 对象，定义了 'ntypes' 属性
add_newdoc('numpy._core', 'ufunc', ('ntypes',
    """
    The number of types.

    The number of numerical NumPy types - of which there are 18 total - on which
    the ufunc can operate.

    See Also
    --------
    numpy.ufunc.types

    Examples
    --------
    >>> np.add.ntypes
    18
    >>> np.multiply.ntypes
    18
    >>> np.power.ntypes
    17
    >>> np.exp.ntypes
    7
    >>> np.remainder.ntypes
    14

    """))

# 添加新的文档字符串到 numpy._core.ufunc 对象，定义了 'types' 属性
add_newdoc('numpy._core', 'ufunc', ('types',
    """
    Returns a list with types grouped input->output.

    Data attribute listing the data-type "Domain-Range" groupings the ufunc can
    deliver. The data-types are given using the character codes.

    See Also
    --------
    numpy.ufunc.ntypes

    Examples
    --------
    >>> np.add.types
    ['??->?', 'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l',
    'LL->L', 'qq->q', 'QQ->Q', 'ff->f', 'dd->d', 'gg->g', 'FF->F', 'DD->D',
    'GG->G', 'OO->O']

    >>> np.multiply.types
    ['??->?', 'bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l',
    'LL->L', 'qq->q', 'QQ->Q', 'ff->f', 'dd->d', 'gg->g', 'FF->F', 'DD->D',
    'GG->G', 'OO->O']

    >>> np.power.types
    # 定义一个包含多个字符串的列表，每个字符串描述了 NumPy 中的数据类型转换规则
    ['bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l', 'LL->L',
    'qq->q', 'QQ->Q', 'ff->f', 'dd->d', 'gg->g', 'FF->F', 'DD->D', 'GG->G',
    'OO->O']
    
    # 调用 NumPy 模块的 exp 对象的 types 属性，返回描述数据类型转换规则的字符串列表
    >>> np.exp.types
    ['f->f', 'd->d', 'g->g', 'F->F', 'D->D', 'G->G', 'O->O']
    
    # 调用 NumPy 模块的 remainder 对象的 types 属性，返回描述数据类型转换规则的字符串列表
    >>> np.remainder.types
    ['bb->b', 'BB->B', 'hh->h', 'HH->H', 'ii->i', 'II->I', 'll->l', 'LL->L',
    'qq->q', 'QQ->Q', 'ff->f', 'dd->d', 'gg->g', 'OO->O']
# 向 numpy._core 模块添加新的文档字符串，关于 ufunc 对象的定义和使用说明
add_newdoc('numpy._core', 'ufunc', ('signature',
    """
    Definition of the core elements a generalized ufunc operates on.

    The signature determines how the dimensions of each input/output array
    are split into core and loop dimensions:

    1. Each dimension in the signature is matched to a dimension of the
       corresponding passed-in array, starting from the end of the shape tuple.
    2. Core dimensions assigned to the same label in the signature must have
       exactly matching sizes, no broadcasting is performed.
    3. The core dimensions are removed from all inputs and the remaining
       dimensions are broadcast together, defining the loop dimensions.

    Notes
    -----
    Generalized ufuncs are used internally in many linalg functions, and in
    the testing suite; the examples below are taken from these.
    For ufuncs that operate on scalars, the signature is None, which is
    equivalent to '()' for every argument.

    Examples
    --------
    >>> np.linalg._umath_linalg.det.signature
    '(m,m)->()'
    >>> np.matmul.signature
    '(n?,k),(k,m?)->(n?,m?)'
    >>> np.add.signature is None
    True  # equivalent to '(),()->()'
    """))

##############################################################################
#
# ufunc methods
#
##############################################################################

# 向 numpy._core 模块的 ufunc 对象添加新的文档字符串，描述 reduce 方法的功能和用法
add_newdoc('numpy._core', 'ufunc', ('reduce',
    """
    reduce(array, axis=0, dtype=None, out=None, keepdims=False, initial=<no value>, where=True)

    Reduces `array`'s dimension by one, by applying ufunc along one axis.

    Let :math:`array.shape = (N_0, ..., N_i, ..., N_{M-1})`.  Then
    :math:`ufunc.reduce(array, axis=i)[k_0, ..,k_{i-1}, k_{i+1}, .., k_{M-1}]` =
    the result of iterating `j` over :math:`range(N_i)`, cumulatively applying
    ufunc to each :math:`array[k_0, ..,k_{i-1}, j, k_{i+1}, .., k_{M-1}]`.
    For a one-dimensional array, reduce produces results equivalent to:
    ::

     r = op.identity # op = ufunc
     for i in range(len(A)):
       r = op(r, A[i])
     return r

    For example, add.reduce() is equivalent to sum().

    Parameters
    ----------
    array : array_like
        The array to act on.
    axis : None or int or tuple of ints, optional
        # 指定沿着哪个轴或哪些轴进行数组的约简操作。
        # 默认情况下 (`axis` = 0)，在输入数组的第一个维度上进行约简。
        # `axis` 可以是负数，此时它从最后一个轴开始计数。

        .. versionadded:: 1.7.0
        # 添加于版本 1.7.0

        If this is None, a reduction is performed over all the axes.
        # 如果 `axis` 是 None，则在所有的轴上进行约简操作。
        If this is a tuple of ints, a reduction is performed on multiple
        axes, instead of a single axis or all the axes as before.
        # 如果 `axis` 是一个整数元组，则在多个轴上进行约简操作，而不是在单个轴或所有轴上进行。

        For operations which are either not commutative or not associative,
        doing a reduction over multiple axes is not well-defined. The
        ufuncs do not currently raise an exception in this case, but will
        likely do so in the future.
        # 对于不是交换律或结合律的操作，在多个轴上进行约简操作没有明确定义。
        # 目前的 ufuncs 在这种情况下不会引发异常，但将来可能会引发异常。

    dtype : data-type code, optional
        # 执行操作所使用的数据类型。如果提供了 `out` 参数，则默认为 `out` 的数据类型；否则默认为 `array` 的数据类型。
        # 对于一些情况（例如 `numpy.add.reduce` 处理整数或布尔输入时），可能会提升数据类型以保留精度。

    out : ndarray, None, or tuple of ndarray and None, optional
        # 存储结果的位置。如果未提供或为 None，则返回一个新分配的数组。
        # 为与 `ufunc.__call__` 保持一致，如果作为关键字参数给出，则可能会被包装在一个元组中（长度为1）。

        .. versionchanged:: 1.13.0
           允许使用元组作为关键字参数。

    keepdims : bool, optional
        # 如果设置为 True，则约简的轴将作为大小为一的维度保留在结果中。
        # 使用此选项，结果将正确地广播到原始 `array` 对象。

        .. versionadded:: 1.7.0
           添加于版本 1.7.0

    initial : scalar, optional
        # 开始约简的初始值。
        # 如果 ufunc 没有单位元或数据类型为 object，则默认为 None；否则默认为 ufunc.identity。
        # 如果给定 `None`，则使用约简的第一个元素；如果约简为空，则抛出错误。

        .. versionadded:: 1.15.0
           添加于版本 1.15.0

    where : array_like of bool, optional
        # 一个布尔数组，广播以匹配 `array` 的维度，并选择要包含在约简中的元素。
        # 对于像 `minimum` 这样没有定义单位元的 ufuncs，还必须传入 `initial`。

        .. versionadded:: 1.17.0
           添加于版本 1.17.0

    Returns
    -------
    r : ndarray
        # 约简后的数组。如果提供了 `out`，则 `r` 是它的引用。

    Examples
    --------
    >>> np.multiply.reduce([2,3,5])
    30
    # 简单数组的例子，使用 `multiply` 函数对数组进行约简操作。

    A multi-dimensional array example:

    >>> X = np.arange(8).reshape((2,2,2))
    >>> X
    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])
    >>> np.add.reduce(X, 0)
    array([[ 4,  6],
           [ 8, 10]])
    # 多维数组的例子，使用 `add` 函数对数组在第一个轴上进行约简操作。
    # 使用 NumPy 中的 np.add.reduce 函数进行数组的逐元素加和操作，沿指定的轴进行求和，默认轴为0
    >>> np.add.reduce(X)
    array([[ 4,  6],
           [ 8, 10]])
    
    # 指定轴为1进行逐元素加和操作
    >>> np.add.reduce(X, 1)
    array([[ 2,  4],
           [10, 12]])
    
    # 指定轴为2进行逐元素加和操作
    >>> np.add.reduce(X, 2)
    array([[ 1,  5],
           [ 9, 13]])
    
    # 使用 initial 关键字参数指定初始值进行加和操作
    >>> np.add.reduce([10], initial=5)
    15
    
    # 在指定轴上进行加和操作，并初始化为指定的初始值
    >>> np.add.reduce(np.ones((2, 2, 2)), axis=(0, 2), initial=10)
    array([14., 14.])
    
    # 在排除 NaN 值的情况下进行加和操作
    >>> a = np.array([10., np.nan, 10])
    >>> np.add.reduce(a, where=~np.isnan(a))
    20.0
    
    # 允许对空数组进行加和操作，当 ufunc 没有身份元素时会失败
    >>> np.minimum.reduce([], initial=np.inf)
    inf
    
    # 在指定条件下进行最小值的逐元素减少操作
    >>> np.minimum.reduce([[1., 2.], [3., 4.]], initial=10., where=[True, False])
    array([ 1., 10.])
    
    # 对空数组进行减少操作会引发 ValueError
    >>> np.minimum.reduce([])
    Traceback (most recent call last):
        ...
    ValueError: zero-size array to reduction operation minimum which has no identity
add_newdoc('numpy._core', 'ufunc', ('accumulate',
    """
    accumulate(array, axis=0, dtype=None, out=None)

    累积应用操作符到所有元素的结果。

    对于一维数组，accumulate 的结果等同于：

      r = np.empty(len(A))      # 创建一个空数组，长度与 A 相同
      t = op.identity           # op 是应用于 A 元素的 ufunc
      for i in range(len(A)):
          t = op(t, A[i])       # 使用 op 累积计算
          r[i] = t              # 将累积结果存入 r
      return r

    例如，add.accumulate() 等同于 np.cumsum()。

    对于多维数组，accumulate 只沿着一个轴应用（默认为第零轴；参见下面的示例），因此如果想要沿多个轴累积，则需要重复使用。

    参数
    ----------
    array : array_like
        待操作的数组。
    axis : int, optional
        沿着哪个轴应用累积操作；默认为零轴。
    dtype : data-type code, optional
        用于表示中间结果的数据类型。如果提供了输出数组，则默认使用输出数组的数据类型；如果没有提供输出数组，则使用输入数组的数据类型。
    out : ndarray, None 或 ndarray 和 None 组成的元组, optional
        存储结果的位置。如果没有提供或为 None，则返回一个新分配的数组。为了与 `ufunc.__call__` 的一致性，如果作为关键字给出，则可能会被包装在一个包含一个元素的元组中。

        .. versionchanged:: 1.13.0
           允许关键字参数为元组。

    返回
    -------
    r : ndarray
        累积值。如果提供了 `out`，则 `r` 是对 `out` 的引用。

    示例
    --------
    一维数组示例：

    >>> np.add.accumulate([2, 3, 5])
    array([ 2,  5, 10])
    >>> np.multiply.accumulate([2, 3, 5])
    array([ 2,  6, 30])

    二维数组示例：

    >>> I = np.eye(2)
    >>> I
    array([[1.,  0.],
           [0.,  1.]])

    沿着轴 0（行）累积，沿着列向下：

    >>> np.add.accumulate(I, 0)
    array([[1.,  0.],
           [1.,  1.]])
    >>> np.add.accumulate(I) # 没有指定轴，默认为零轴
    array([[1.,  0.],
           [1.,  1.]])

    沿着轴 1（列）累积，沿着行向右：

    >>> np.add.accumulate(I, 1)
    array([[1.,  1.],
           [0.,  1.]])

    """))

add_newdoc('numpy._core', 'ufunc', ('reduceat',
    """
    reduceat(array, indices, axis=0, dtype=None, out=None)

    在指定切片上沿单个轴执行（局部）reduce 操作。

    对于每个 `i` 在 `range(len(indices))` 中，`reduceat` 计算
    ``ufunc.reduce(array[indices[i]:indices[i+1]])``，这变成了最终结果中 `axis` 方向上的第 `i` 个广义“行”（例如，在 2-D 数组中，如果 `axis = 0`，它将成为第 `i` 行，但如果 `axis = 1`，它将成为第 `i` 列）。有三个例外情况：
    # 当 ``i = len(indices) - 1``（即最后一个索引时），
    # ``indices[i+1] = array.shape[axis]``。

    # 如果 ``indices[i] >= indices[i + 1]``，第i个广义“行”就是 ``array[indices[i]]``。

    # 如果 ``indices[i] >= len(array)`` 或 ``indices[i] < 0``，会引发错误。

    # 输出的形状取决于 `indices` 的大小，可能比 `array` 大（如果 ``len(indices) > array.shape[axis]``）。

    # array : array_like
    #     要操作的数组。
    # indices : array_like
    #     成对的索引，以逗号分隔（不是冒号），指定要减少的切片。
    # axis : int, optional
    #     应用 reduceat 的轴。
    # dtype : data-type code, optional
    #     执行操作时使用的数据类型。默认为 `out` 的类型（如果给出），否则为 `array` 的数据类型（尽管会上转以保留一些情况下的精度，比如对整数或布尔输入进行 `numpy.add.reduce`）。
    # out : ndarray, None 或者 ndarray 和 None 的元组，可选
    #     存储结果的位置。如果未提供或为 None，则返回新分配的数组。与 `ufunc.__call__` 一致，如果作为关键字给出，则可能会被包装在一个 1 元素元组中。
    # 
    #     .. versionchanged:: 1.13.0
    #        允许关键字参数为元组。
    # 
    # 返回
    # -------
    # r : ndarray
    #     减少的值。如果提供了 `out`，则 `r` 是 `out` 的引用。
    # 
    # Notes
    # -----
    # 描述性示例：
    # 
    # 如果 `array` 是 1-D，则函数 `ufunc.accumulate(array)` 等同于 ``ufunc.reduceat(array, indices)[::2]``，其中 `indices` 是 ``range(len(array) - 1)``，其中每隔一个元素放置一个零：
    # ``indices = zeros(2 * len(array) - 1)``,
    # ``indices[1::2] = range(1, len(array))``。
    # 
    # 不要被这个属性的名称所误导：`reduceat(array)` 不一定比 `array` 小。
    # 
    # Examples
    # --------
    # 对四个连续值进行累加：
    # 
    # >>> np.add.reduceat(np.arange(8), [0, 4, 1, 5, 2, 6, 3, 7])[::2]
    # array([ 6, 10, 14, 18])
    # 
    # 2-D 示例：
    # 
    # >>> x = np.linspace(0, 15, 16).reshape(4, 4)
    # >>> x
    # array([[ 0.,  1.,  2.,  3.],
    #        [ 4.,  5.,  6.,  7.],
    #        [ 8.,  9., 10., 11.],
    #        [12., 13., 14., 15.]])
    # 
    # ::
    # 
    #  # 减少以使结果具有以下五行：
    #  # [row1 + row2 + row3]
    #  # [row4]
    #  # [row2]
    #  # [row3]
    #  # [row1 + row2 + row3 + row4]
    # 
    # >>> np.add.reduceat(x, [0, 3, 1, 2, 0])
    # array([[12., 15., 18., 21.],
    #        [12., 13., 14., 15.],
    #        [ 4.,  5.,  6.,  7.],
    #        [ 8.,  9., 10., 11.],
    #        [24., 28., 32., 36.]])
    # 
    # ::
    # 
    #  # 减少以使结果具有以下两列：
    #  # [col1 * col2 * col3, col4]
    # 
    # >>> np.multiply.reduceat(x, [0, 3], 1)
    array([[   0.,     3.],
           [ 120.,     7.],
           [ 720.,    11.],
           [2184.,    15.]])



# 创建一个包含四行两列的NumPy数组，表示一组二维数据
# 第一列包含值 [0., 120., 720., 2184.]，第二列包含值 [3., 7., 11., 15.]
add_newdoc('numpy._core', 'ufunc', ('outer',
    r"""
    outer(A, B, /, **kwargs)

    Apply the ufunc `op` to all pairs (a, b) with a in `A` and b in `B`.

    Let ``M = A.ndim``, ``N = B.ndim``. Then the result, `C`, of
    ``op.outer(A, B)`` is an array of dimension M + N such that:

    .. math:: C[i_0, ..., i_{M-1}, j_0, ..., j_{N-1}] =
       op(A[i_0, ..., i_{M-1}], B[j_0, ..., j_{N-1}])

    For `A` and `B` one-dimensional, this is equivalent to::

      r = empty(len(A),len(B))
      for i in range(len(A)):
          for j in range(len(B)):
              r[i,j] = op(A[i], B[j])  # op = ufunc in question

    Parameters
    ----------
    A : array_like
        First array
    B : array_like
        Second array
    kwargs : any
        Arguments to pass on to the ufunc. Typically `dtype` or `out`.
        See `ufunc` for a comprehensive overview of all available arguments.

    Returns
    -------
    r : ndarray
        Output array

    See Also
    --------
    numpy.outer : A less powerful version of ``np.multiply.outer``
                  that `ravel`\ s all inputs to 1D. This exists
                  primarily for compatibility with old code.

    tensordot : ``np.tensordot(a, b, axes=((), ()))`` and
                ``np.multiply.outer(a, b)`` behave same for all
                dimensions of a and b.

    Examples
    --------
    >>> np.multiply.outer([1, 2, 3], [4, 5, 6])
    array([[ 4,  5,  6],
           [ 8, 10, 12],
           [12, 15, 18]])

    A multi-dimensional example:

    >>> A = np.array([[1, 2, 3], [4, 5, 6]])
    >>> A.shape
    (2, 3)
    >>> B = np.array([[1, 2, 3, 4]])
    >>> B.shape
    (1, 4)
    >>> C = np.multiply.outer(A, B)
    >>> C.shape; C
    (2, 3, 1, 4)
    array([[[[ 1,  2,  3,  4]],
            [[ 2,  4,  6,  8]],
            [[ 3,  6,  9, 12]]],
           [[[ 4,  8, 12, 16]],
            [[ 5, 10, 15, 20]],
            [[ 6, 12, 18, 24]]]])

    """))


注释：


# 向 numpy._core.ufunc 添加新的文档条目，名称为 'outer'
add_newdoc('numpy._core', 'ufunc', ('outer',
    r"""
    outer(A, B, /, **kwargs)

    Apply the ufunc `op` to all pairs (a, b) with a in `A` and b in `B`.

    计算两个数组 A 和 B 中所有可能的元素对 (a, b) 的操作结果。

    Let ``M = A.ndim``, ``N = B.ndim``. Then the result, `C`, of
    ``op.outer(A, B)`` is an array of dimension M + N such that:

    结果数组 `C` 的维度为 M + N，其中 M 是数组 A 的维度，N 是数组 B 的维度，满足以下关系：

    .. math:: C[i_0, ..., i_{M-1}, j_0, ..., j_{N-1}] =
       op(A[i_0, ..., i_{M-1}], B[j_0, ..., j_{N-1}])

    对于一维数组 A 和 B，这相当于以下操作：

    r = empty(len(A),len(B))
    for i in range(len(A)):
        for j in range(len(B)):
            r[i,j] = op(A[i], B[j])  # op 是被应用的 ufunc

    Parameters
    ----------
    A : array_like
        第一个数组
    B : array_like
        第二个数组
    kwargs : 任意
        传递给 ufunc 的参数。通常是 `dtype` 或 `out`。
        参见 `ufunc` 获取所有可用参数的详细说明。

    Returns
    -------
    r : ndarray
        输出数组

    See Also
    --------
    numpy.outer : `np.multiply.outer` 的一个较不强大的版本，
                  将所有输入展平为 1D。主要用于与旧代码的兼容性。

    tensordot : ``np.tensordot(a, b, axes=((), ()))`` 和
                ``np.multiply.outer(a, b)`` 对于 a 和 b 的所有维度行为相同。

    Examples
    --------
    >>> np.multiply.outer([1, 2, 3], [4, 5, 6])
    array([[ 4,  5,  6],
           [ 8, 10, 12],
           [12, 15, 18]])

    多维示例：

    >>> A = np.array([[1, 2, 3], [4, 5, 6]])
    >>> A.shape
    (2, 3)
    >>> B = np.array([[1, 2, 3, 4]])
    >>> B.shape
    (1, 4)
    >>> C = np.multiply.outer(A, B)
    >>> C.shape; C
    (2, 3, 1, 4)
    array([[[[ 1,  2,  3,  4]],
            [[ 2,  4,  6,  8]],
            [[ 3,  6,  9, 12]]],
           [[[ 4,  8, 12, 16]],
            [[ 5, 10, 15, 20]],
            [[ 6, 12, 18, 24]]]])

    """))
    Examples
    --------
    # 创建一个包含四个元素的 NumPy 数组
    >>> a = np.array([1, 2, 3, 4])
    # 对数组中索引为 0 和 1 的元素取负值
    >>> np.negative.at(a, [0, 1])
    # 输出变更后的数组
    >>> a
    array([-1, -2,  3,  4])

    # 对数组中索引为 0, 1 和 2 的元素进行增量操作，索引为 2 的元素增量两次
    >>> a = np.array([1, 2, 3, 4])
    >>> np.add.at(a, [0, 1, 2, 2], 1)
    # 输出变更后的数组
    >>> a
    array([2, 3, 5, 4])

    # 将第二个数组 b 中索引为 0 和 1 的元素加到第一个数组 a 中对应的索引位置上
    # 并将结果存储在第一个数组 a 中
    >>> a = np.array([1, 2, 3, 4])
    >>> b = np.array([1, 2])
    >>> np.add.at(a, [0, 1], b)
    # 输出变更后的数组
    >>> a
    array([2, 4, 3, 4])
# 将新的文档添加到指定模块和函数的文档中
add_newdoc('numpy._core', 'ufunc', ('resolve_dtypes',
    """
    resolve_dtypes(dtypes, *, signature=None, casting=None, reduction=False)

    Find the dtypes NumPy will use for the operation.  Both input and
    output dtypes are returned and may differ from those provided.

    .. note::

        This function always applies NEP 50 rules since it is not provided
        any actual values.  The Python types ``int``, ``float``, and
        ``complex`` thus behave weak and should be passed for "untyped"
        Python input.

    Parameters
    ----------
    dtypes : tuple of dtypes, None, or literal int, float, complex
        The input dtypes for each operand.  Output operands can be
        None, indicating that the dtype must be found.
    signature : tuple of DTypes or None, optional
        If given, enforces exact DType (classes) of the specific operand.
        The ufunc ``dtype`` argument is equivalent to passing a tuple with
        only output dtypes set.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        The casting mode when casting is necessary.  This is identical to
        the ufunc call casting modes.
    reduction : boolean
        If given, the resolution assumes a reduce operation is happening
        which slightly changes the promotion and type resolution rules.
        `dtypes` is usually something like ``(None, np.dtype("i2"), None)``
        for reductions (first input is also the output).

        .. note::

            The default casting mode is "same_kind", however, as of
            NumPy 1.24, NumPy uses "unsafe" for reductions.

    Returns
    -------
    dtypes : tuple of dtypes
        The dtypes which NumPy would use for the calculation.  Note that
        dtypes may not match the passed in ones (casting is necessary).


    Examples
    --------
    This API requires passing dtypes, define them for convenience:

    >>> int32 = np.dtype("int32")
    >>> float32 = np.dtype("float32")

    The typical ufunc call does not pass an output dtype.  `numpy.add` has two
    inputs and one output, so leave the output as ``None`` (not provided):

    >>> np.add.resolve_dtypes((int32, float32, None))
    (dtype('float64'), dtype('float64'), dtype('float64'))

    The loop found uses "float64" for all operands (including the output), the
    first input would be cast.

    ``resolve_dtypes`` supports "weak" handling for Python scalars by passing
    ``int``, ``float``, or ``complex``:

    >>> np.add.resolve_dtypes((float32, float, None))
    (dtype('float32'), dtype('float32'), dtype('float32'))

    Where the Python ``float`` behaves samilar to a Python value ``0.0``
    in a ufunc call.  (See :ref:`NEP 50 <NEP50>` for details.)

    """
))

# 定义一个函数 `_resolve_dtypes_and_context`，作用与 `numpy.ufunc.resolve_dtypes` 函数相同，详见其参数信息
add_newdoc('numpy._core', 'ufunc', ('_resolve_dtypes_and_context',
    """
    _resolve_dtypes_and_context(dtypes, *, signature=None, casting=None, reduction=False)

    See `numpy.ufunc.resolve_dtypes` for parameter information.  This
    """
))
    function is considered *unstable*.  You may use it, but the returned
    information is NumPy version specific and expected to change.
    Large API/ABI changes are not expected, but a new NumPy version is
    expected to require updating code using this functionality.

    This function is designed to be used in conjunction with
    `numpy.ufunc._get_strided_loop`.  The calls are split to mirror the C API
    and allow future improvements.

    Returns
    -------
    dtypes : tuple of dtypes
        返回一个元组，包含多个 dtype 对象，表示返回结果中的数据类型。
    call_info :
        返回一个 PyCapsule 对象，其中包含所有获取低级 C 调用所需信息的封装。
        参考 `numpy.ufunc._get_strided_loop` 获取更多信息。
# 向 numpy._core 模块添加新的文档
add_newdoc('numpy._core', 'ufunc', ('_get_strided_loop',
    """
    _get_strided_loop(call_info, /, *, fixed_strides=None)

    This function fills in the ``call_info`` capsule to include all
    information necessary to call the low-level strided loop from NumPy.

    See notes for more information.

    Parameters
    ----------
    call_info : PyCapsule
        The PyCapsule returned by `numpy.ufunc._resolve_dtypes_and_context`.
    fixed_strides : tuple of int or None, optional
        A tuple with fixed byte strides of all input arrays.  NumPy may use
        this information to find specialized loops, so any call must follow
        the given stride.  Use ``None`` to indicate that the stride is not
        known (or not fixed) for all calls.

    Notes
    -----
    Together with `numpy.ufunc._resolve_dtypes_and_context` this function
    gives low-level access to the NumPy ufunc loops.
    The first function does general preparation and returns the required
    information. It returns this as a C capsule with the version specific
    name ``numpy_1.24_ufunc_call_info``.
    The NumPy 1.24 ufunc call info capsule has the following layout::

        typedef struct {
            PyArrayMethod_StridedLoop *strided_loop;
            PyArrayMethod_Context *context;
            NpyAuxData *auxdata;

            /* Flag information (expected to change) */
            npy_bool requires_pyapi;  /* GIL is required by loop */

            /* Loop doesn't set FPE flags; if not set check FPE flags */
            npy_bool no_floatingpoint_errors;
        } ufunc_call_info;

    Note that the first call only fills in the ``context``.  The call to
    ``_get_strided_loop`` fills in all other data.  The main thing to note is
    that the new-style loops return 0 on success, -1 on failure.  They are
    passed context as new first input and ``auxdata`` as (replaced) last.

    Only the ``strided_loop`` signature is considered guaranteed stable
    for NumPy bug-fix releases.  All other API is tied to the experimental
    API versioning.

    The reason for the split call is that cast information is required to
    decide what the fixed-strides will be.

    NumPy ties the lifetime of the ``auxdata`` information to the capsule.
    """))



##############################################################################
#
# Documentation for dtype attributes and methods
#
##############################################################################

##############################################################################
#
# dtype object
#
##############################################################################

# 向 numpy._core.multiarray 模块添加新的文档
add_newdoc('numpy._core.multiarray', 'dtype',
    """
    dtype(dtype, align=False, copy=False, [metadata])

    Create a data type object.

    A numpy array is homogeneous, and contains elements described by a
    dtype object. A dtype object can be constructed from different
    combinations of fundamental numeric types.
    """)
    Parameters
    ----------
    dtype
        要转换为数据类型对象的对象。
    align : bool, optional
        是否添加填充以匹配类似 C 结构的输出。仅当 `obj` 是字典或逗号分隔的字符串时可以为 ``True``。
        如果正在创建结构化的 dtype，这也会设置一个粘性对齐标志 ``isalignedstruct``。
    copy : bool, optional
        是否创建数据类型对象的新副本。如果 ``False``，结果可能只是内置数据类型对象的引用。
    metadata : dict, optional
        带有 dtype 元数据的可选字典。

    See also
    --------
    result_type

    Examples
    --------
    使用数组标量类型：

    >>> np.dtype(np.int16)
    dtype('int16')

    结构化类型，包含一个字段名 'f1'，其类型为 int16：

    >>> np.dtype([('f1', np.int16)])
    dtype([('f1', '<i2')])

    结构化类型，一个名为 'f1' 的字段，它本身包含一个具有一个字段的结构化类型：

    >>> np.dtype([('f1', [('f1', np.int16)])])
    dtype([('f1', [('f1', '<i2')])])

    结构化类型，两个字段：第一个字段包含无符号整数，第二个字段为 int32：

    >>> np.dtype([('f1', np.uint64), ('f2', np.int32)])
    dtype([('f1', '<u8'), ('f2', '<i4')])

    使用数组协议类型字符串：

    >>> np.dtype([('a','f8'),('b','S10')])
    dtype([('a', '<f8'), ('b', 'S10')])

    使用逗号分隔的字段格式。形状为 (2,3)：

    >>> np.dtype("i4, (2,3)f8")
    dtype([('f0', '<i4'), ('f1', '<f8', (2, 3))])

    使用元组。``int`` 是固定类型，3 是字段的形状。``void`` 是灵活类型，在此处大小为 10：

    >>> np.dtype([('hello',(np.int64,3)),('world',np.void,10)])
    dtype([('hello', '<i8', (3,)), ('world', 'V10')])

    将 ``int16`` 分成 2 个 ``int8``，称为 x 和 y。0 和 1 是字节偏移量：

    >>> np.dtype((np.int16, {'x':(np.int8,0), 'y':(np.int8,1)}))
    dtype((numpy.int16, [('x', 'i1'), ('y', 'i1')]))

    使用字典。两个字段名为 'gender' 和 'age'：

    >>> np.dtype({'names':['gender','age'], 'formats':['S1',np.uint8]})
    dtype([('gender', 'S1'), ('age', 'u1')])

    字节偏移量，在此处为 0 和 25：

    >>> np.dtype({'surname':('S25',0),'age':(np.uint8,25)})
    dtype([('surname', 'S25'), ('age', 'u1')])
# 向 numpy._core.multiarray 的 dtype 类添加新文档条目，具体是 'alignment' 属性
add_newdoc('numpy._core.multiarray', 'dtype', ('alignment',
    """
    This attribute represents the required memory alignment (in bytes) of
    the data-type according to the compiler.

    For more details, refer to the C-API section of the NumPy manual.

    Examples
    --------

    >>> x = np.dtype('i4')
    >>> x.alignment
    4

    >>> x = np.dtype(float)
    >>> x.alignment
    8

    """))

# 向 numpy._core.multiarray 的 dtype 类添加新文档条目，具体是 'byteorder' 属性
add_newdoc('numpy._core.multiarray', 'dtype', ('byteorder',
    """
    A character indicating the byte-order of this data-type object.

    Possible values include:

    '='  native byte-order
    '<'  little-endian byte-order
    '>'  big-endian byte-order
    '|'  byte-order not applicable

    All built-in data-type objects have byteorder either '=' or '|'.

    Examples
    --------

    >>> dt = np.dtype('i2')
    >>> dt.byteorder
    '='
    >>> np.dtype('i1').byteorder
    '|'
    >>> np.dtype('S2').byteorder
    '|'
    >>> sys_is_le = sys.byteorder == 'little'
    >>> native_code = '<' if sys_is_le else '>'
    >>> swapped_code = '>' if sys_is_le else '<'
    >>> dt = np.dtype(native_code + 'i2')
    >>> dt.byteorder
    '='
    >>> dt = np.dtype(swapped_code + 'i2')
    >>> dt.byteorder == swapped_code
    True

    """))

# 向 numpy._core.multiarray 的 dtype 类添加新文档条目，具体是 'char' 属性
add_newdoc('numpy._core.multiarray', 'dtype', ('char',
    """A unique character code representing each of the 21 different built-in types.

    Examples
    --------

    >>> x = np.dtype(float)
    >>> x.char
    'd'

    """))

# 向 numpy._core.multiarray 的 dtype 类添加新文档条目，具体是 'descr' 属性
add_newdoc('numpy._core.multiarray', 'dtype', ('descr',
    """
    Description of the data-type according to the `__array_interface__`.

    This format is required by the 'descr' key in the `__array_interface__` attribute.

    Warning: This attribute is specific to `__array_interface__`. Passing it directly
    to `numpy.dtype` may not accurately reconstruct certain dtypes (e.g., scalar and
    subarray dtypes).

    Examples
    --------

    >>> x = np.dtype(float)
    >>> x.descr
    [('', '<f8')]

    >>> dt = np.dtype([('name', np.str_, 16), ('grades', np.float64, (2,))])
    >>> dt.descr
    [('name', '<U16'), ('grades', '<f8', (2,))]

    """))

# 向 numpy._core.multiarray 的 dtype 类添加新文档条目，具体是 'fields' 属性
add_newdoc('numpy._core.multiarray', 'dtype', ('fields',
    """
    Dictionary of named fields defined for this data type, or ``None`` if no
    fields are defined.

    The dictionary keys are field names. Each dictionary entry is a tuple
    describing the field as follows:

    (dtype, offset[, title])

    Offset is a C int, typically signed and 32 bits. If present, the optional
    title can be any object (typically a string).

    """))
    or unicode then it will also be a key in the fields dictionary,
    # 如果数据类型是字符串或Unicode，则它也将成为字段字典中的一个键

    otherwise it's meta-data). Notice also that the first two elements
    # 否则，它是元数据）。注意，元组的前两个元素可以直接作为 ndarray.getfield 和 ndarray.setfield 方法的参数。

    of the tuple can be passed directly as arguments to the
    # 元组的前两个元素可以直接作为参数传递给 ndarray.getfield 和 ndarray.setfield 方法。

    ``ndarray.getfield`` and ``ndarray.setfield`` methods.
    # ndarray.getfield 和 ndarray.setfield 方法。

    See Also
    --------
    ndarray.getfield, ndarray.setfield

    Examples
    --------
    >>> dt = np.dtype([('name', np.str_, 16), ('grades', np.float64, (2,))])
    >>> print(dt.fields)
    {'grades': (dtype(('float64',(2,))), 16), 'name': (dtype('|S16'), 0)}

    """))
    # 示例代码块结束
# 添加新的文档字符串到numpy._core.multiarray.dtype中，关于dtype对象的特性flags
add_newdoc('numpy._core.multiarray', 'dtype', ('flags',
    """
    Bit-flags describing how this data type is to be interpreted.

    Bit-masks are in ``numpy._core.multiarray`` as the constants
    `ITEM_HASOBJECT`, `LIST_PICKLE`, `ITEM_IS_POINTER`, `NEEDS_INIT`,
    `NEEDS_PYAPI`, `USE_GETITEM`, `USE_SETITEM`. A full explanation
    of these flags is in C-API documentation; they are largely useful
    for user-defined data-types.

    The following example demonstrates that operations on this particular
    dtype requires Python C-API.

    Examples
    --------

    >>> x = np.dtype([('a', np.int32, 8), ('b', np.float64, 6)])
    >>> x.flags
    16
    >>> np._core.multiarray.NEEDS_PYAPI
    16

    """))

# 添加新的文档字符串到numpy._core.multiarray.dtype中，关于dtype对象的特性hasobject
add_newdoc('numpy._core.multiarray', 'dtype', ('hasobject',
    """
    Boolean indicating whether this dtype contains any reference-counted
    objects in any fields or sub-dtypes.

    Recall that what is actually in the ndarray memory representing
    the Python object is the memory address of that object (a pointer).
    Special handling may be required, and this attribute is useful for
    distinguishing data types that may contain arbitrary Python objects
    and data-types that won't.

    """))

# 添加新的文档字符串到numpy._core.multiarray.dtype中，关于dtype对象的特性isbuiltin
add_newdoc('numpy._core.multiarray', 'dtype', ('isbuiltin',
    """
    Integer indicating how this dtype relates to the built-in dtypes.

    Read-only.

    =  ========================================================================
    0  if this is a structured array type, with fields
    1  if this is a dtype compiled into numpy (such as ints, floats etc)
    2  if the dtype is for a user-defined numpy type
       A user-defined type uses the numpy C-API machinery to extend
       numpy to handle a new array type. See
       :ref:`user.user-defined-data-types` in the NumPy manual.
    =  ========================================================================

    Examples
    --------
    >>> dt = np.dtype('i2')
    >>> dt.isbuiltin
    1
    >>> dt = np.dtype('f8')
    >>> dt.isbuiltin
    1
    >>> dt = np.dtype([('field1', 'f8')])
    >>> dt.isbuiltin
    0

    """))

# 添加新的文档字符串到numpy._core.multiarray.dtype中，关于dtype对象的特性isnative
add_newdoc('numpy._core.multiarray', 'dtype', ('isnative',
    """
    Boolean indicating whether the byte order of this dtype is native
    to the platform.

    """))

# 添加新的文档字符串到numpy._core.multiarray.dtype中，关于dtype对象的特性isalignedstruct
add_newdoc('numpy._core.multiarray', 'dtype', ('isalignedstruct',
    """
    Boolean indicating whether the dtype is a struct which maintains
    field alignment. This flag is sticky, so when combining multiple
    structs together, it is preserved and produces new dtypes which
    are also aligned.

    """))

# 添加新的文档字符串到numpy._core.multiarray.dtype中，关于dtype对象的特性itemsize
add_newdoc('numpy._core.multiarray', 'dtype', ('itemsize',
    """
    The element size of this data-type object.

    For 18 of the 21 types this number is fixed by the data-type.
    For the flexible data-types, this number can be anything.

    Examples
    --------

    >>> arr = np.array([[1, 2], [3, 4]])
    >>> arr.dtype
    dtype('int64')
    >>> arr.itemsize
    8
    # 创建一个自定义的 NumPy 数据类型 `dt`，包含两个字段：'name' 是长度为 16 的字符串类型，'grades' 是一个包含两个 float64 元素的数组类型
    dt = np.dtype([('name', np.str_, 16), ('grades', np.float64, (2,))])
    # 访问数据类型 `dt` 的每个元素所占用的字节数
    dt.itemsize
    # 返回值为 80，表示数据类型 `dt` 的总字节数
# 定义一个新的文档字符串，描述了数据类型的字符代码（kind），指示数据的一般种类
add_newdoc('numpy._core.multiarray', 'dtype', ('kind',
    """
    A character code (one of 'biufcmMOSUV') identifying the general kind of data.

    =  ======================
    b  boolean
    i  signed integer
    u  unsigned integer
    f  floating-point
    c  complex floating-point
    m  timedelta
    M  datetime
    O  object
    S  (byte-)string
    U  Unicode
    V  void
    =  ======================

    Examples
    --------

    >>> dt = np.dtype('i4')
    >>> dt.kind
    'i'
    >>> dt = np.dtype('f8')
    >>> dt.kind
    'f'
    >>> dt = np.dtype([('field1', 'f8')])
    >>> dt.kind
    'V'

    """))

# 定义一个新的文档字符串，描述了数据类型的元数据（metadata）字段，可以是 None 或只读字典
add_newdoc('numpy._core.multiarray', 'dtype', ('metadata',
    """
    Either ``None`` or a readonly dictionary of metadata (mappingproxy).

    The metadata field can be set using any dictionary at data-type
    creation. NumPy currently has no uniform approach to propagating
    metadata; although some array operations preserve it, there is no
    guarantee that others will.

    .. warning::

        Although used in certain projects, this feature was long undocumented
        and is not well supported. Some aspects of metadata propagation
        are expected to change in the future.

    Examples
    --------

    >>> dt = np.dtype(float, metadata={"key": "value"})
    >>> dt.metadata["key"]
    'value'
    >>> arr = np.array([1, 2, 3], dtype=dt)
    >>> arr.dtype.metadata
    mappingproxy({'key': 'value'})

    Adding arrays with identical datatypes currently preserves the metadata:

    >>> (arr + arr).dtype.metadata
    mappingproxy({'key': 'value'})

    But if the arrays have different dtype metadata, the metadata may be
    dropped:

    >>> dt2 = np.dtype(float, metadata={"key2": "value2"})
    >>> arr2 = np.array([3, 2, 1], dtype=dt2)
    >>> (arr + arr2).dtype.metadata is None
    True  # The metadata field is cleared so None is returned
    """))

# 定义一个新的文档字符串，描述了数据类型的名称（name），表示数据类型的位宽名称
add_newdoc('numpy._core.multiarray', 'dtype', ('name',
    """
    A bit-width name for this data-type.

    Un-sized flexible data-type objects do not have this attribute.

    Examples
    --------

    >>> x = np.dtype(float)
    >>> x.name
    'float64'
    >>> x = np.dtype([('a', np.int32, 8), ('b', np.float64, 6)])
    >>> x.name
    'void640'

    """))

# 定义一个新的文档字符串，描述了数据类型的字段名称（names），表示数据类型的有序字段名称列表
add_newdoc('numpy._core.multiarray', 'dtype', ('names',
    """
    Ordered list of field names, or ``None`` if there are no fields.

    The names are ordered according to increasing byte offset. This can be
    used, for example, to walk through all of the named fields in offset order.

    Examples
    --------
    >>> dt = np.dtype([('name', np.str_, 16), ('grades', np.float64, (2,))])
    >>> dt.names
    ('name', 'grades')

    """))

# 定义一个新的文档字符串，描述了数据类型的编号（num），表示每种内置类型的唯一编号
add_newdoc('numpy._core.multiarray', 'dtype', ('num',
    """
    A unique number for each of the 21 different built-in types.

    These are roughly ordered from least-to-most precision.

    Examples
    --------

    >>> dt = np.dtype(str)
    >>> dt.num
    19
    # 创建一个新的 NumPy 数据类型 `dt`，其元素类型为 float
    >>> dt = np.dtype(float)
    # 访问 NumPy 数据类型对象 `dt` 的 `num` 属性，该属性代表元素的标识符
    >>> dt.num
    # 返回值 12 表示元素类型的标识符，对于 float 类型而言，标识符是 12
    12
add_newdoc('numpy._core.multiarray', 'dtype', ('shape',
    """
    Shape tuple of the sub-array if this data type describes a sub-array,
    and ``()`` otherwise.

    Examples
    --------

    >>> dt = np.dtype(('i4', 4))
    >>> dt.shape
    (4,)

    >>> dt = np.dtype(('i4', (2, 3)))
    >>> dt.shape
    (2, 3)

    """))

add_newdoc('numpy._core.multiarray', 'dtype', ('ndim',
    """
    Number of dimensions of the sub-array if this data type describes a
    sub-array, and ``0`` otherwise.

    .. versionadded:: 1.13.0

    Examples
    --------
    >>> x = np.dtype(float)
    >>> x.ndim
    0

    >>> x = np.dtype((float, 8))
    >>> x.ndim
    1

    >>> x = np.dtype(('i4', (3, 4)))
    >>> x.ndim
    2

    """))

add_newdoc('numpy._core.multiarray', 'dtype', ('str',
    """The array-protocol typestring of this data-type object."""))

add_newdoc('numpy._core.multiarray', 'dtype', ('subdtype',
    """
    Tuple ``(item_dtype, shape)`` if this `dtype` describes a sub-array, and
    None otherwise.

    The *shape* is the fixed shape of the sub-array described by this
    data type, and *item_dtype* the data type of the array.

    If a field whose dtype object has this attribute is retrieved,
    then the extra dimensions implied by *shape* are tacked on to
    the end of the retrieved array.

    See Also
    --------
    dtype.base

    Examples
    --------
    >>> x = numpy.dtype('8f')
    >>> x.subdtype
    (dtype('float32'), (8,))

    >>> x =  numpy.dtype('i2')
    >>> x.subdtype
    >>>

    """))

add_newdoc('numpy._core.multiarray', 'dtype', ('base',
    """
    Returns dtype for the base element of the subarrays,
    regardless of their dimension or shape.

    See Also
    --------
    dtype.subdtype

    Examples
    --------
    >>> x = numpy.dtype('8f')
    >>> x.base
    dtype('float32')

    >>> x =  numpy.dtype('i2')
    >>> x.base
    dtype('int16')

    """))

add_newdoc('numpy._core.multiarray', 'dtype', ('type',
    """The type object used to instantiate a scalar of this data-type."""))

##############################################################################
#
# dtype methods
#
##############################################################################

add_newdoc('numpy._core.multiarray', 'dtype', ('newbyteorder',
    """
    newbyteorder(new_order='S', /)

    Return a new dtype with a different byte order.

    Changes are also made in all fields and sub-arrays of the data type.

    Parameters
    ----------
    new_order : string, optional
        Byte order to force; a value from the byte order specifications
        below.  The default value ('S') results in swapping the current
        byte order.  `new_order` codes can be any of:

        * 'S' - swap dtype from current to opposite endian
        * {'<', 'little'} - little endian
        * {'>', 'big'} - big endian
        * {'=', 'native'} - native order
        * {'|', 'I'} - ignore (no change to byte order)

    Returns
    -------
    A new dtype object with the specified byte order.
    """
))
    new_dtype : dtype
        New dtype object with the given change to the byte order.
    # 定义函数签名和文档字符串，描述返回一个新的数据类型对象，其字节顺序按指定修改。

    Notes
    -----
    Changes are also made in all fields and sub-arrays of the data type.
    # 注意事项部分说明：数据类型的所有字段和子数组的字节顺序也会被修改。

    Examples
    --------
    >>> import sys
    >>> sys_is_le = sys.byteorder == 'little'
    # 导入 sys 模块并检查系统的字节顺序是否为 little endian

    >>> native_code = '<' if sys_is_le else '>'
    # 如果系统是 little endian，则 native_code 为 '<'，否则为 '>'

    >>> swapped_code = '>' if sys_is_le else '<'
    # 如果系统是 little endian，则 swapped_code 为 '>'，否则为 '<'

    >>> native_dt = np.dtype(native_code+'i2')
    # 创建一个基于系统字节顺序的新数据类型对象 native_dt，数据类型为 int16

    >>> swapped_dt = np.dtype(swapped_code+'i2')
    # 创建一个基于相反字节顺序的新数据类型对象 swapped_dt，数据类型为 int16

    >>> native_dt.newbyteorder('S') == swapped_dt
    True
    # 检查 native_dt 的字节顺序修改为 'S' 后是否等于 swapped_dt

    >>> native_dt.newbyteorder() == swapped_dt
    True
    # 检查 native_dt 默认字节顺序修改后是否等于 swapped_dt

    >>> native_dt == swapped_dt.newbyteorder('S')
    True
    # 检查 native_dt 是否等于 swapped_dt 修改字节顺序为 'S' 后的结果

    >>> native_dt == swapped_dt.newbyteorder('=')
    True
    # 检查 native_dt 是否等于 swapped_dt 修改字节顺序为 '=' 后的结果

    >>> native_dt == swapped_dt.newbyteorder('N')
    True
    # 检查 native_dt 是否等于 swapped_dt 修改字节顺序为 'N' 后的结果

    >>> native_dt == native_dt.newbyteorder('|')
    True
    # 检查 native_dt 修改字节顺序为 '|' 后是否等于自身

    >>> np.dtype('<i2') == native_dt.newbyteorder('<')
    True
    # 检查 '<i2' 数据类型是否等于 native_dt 修改为 '<' 后的结果

    >>> np.dtype('<i2') == native_dt.newbyteorder('L')
    True
    # 检查 '<i2' 数据类型是否等于 native_dt 修改为 'L' 后的结果

    >>> np.dtype('>i2') == native_dt.newbyteorder('>')
    True
    # 检查 '>i2' 数据类型是否等于 native_dt 修改为 '>' 后的结果

    >>> np.dtype('>i2') == native_dt.newbyteorder('B')
    True
    # 检查 '>i2' 数据类型是否等于 native_dt 修改为 'B' 后的结果
add_newdoc('numpy._core.multiarray', 'dtype', ('__class_getitem__',
    """
    __class_getitem__(item, /)

    Return a parametrized wrapper around the `~numpy.dtype` type.

    .. versionadded:: 1.22

    Returns
    -------
    alias : types.GenericAlias
        A parametrized `~numpy.dtype` type.

    Examples
    --------
    >>> import numpy as np

    >>> np.dtype[np.int64]
    numpy.dtype[numpy.int64]

    See Also
    --------
    :pep:`585` : Type hinting generics in standard collections.

    """))

add_newdoc('numpy._core.multiarray', 'dtype', ('__ge__',
    """
    __ge__(value, /)

    Return ``self >= value``.

    Equivalent to ``np.can_cast(value, self, casting="safe")``.

    See Also
    --------
    can_cast : Returns True if cast between data types can occur according to
               the casting rule.

    """))

add_newdoc('numpy._core.multiarray', 'dtype', ('__le__',
    """
    __le__(value, /)

    Return ``self <= value``.

    Equivalent to ``np.can_cast(self, value, casting="safe")``.

    See Also
    --------
    can_cast : Returns True if cast between data types can occur according to
               the casting rule.

    """))

add_newdoc('numpy._core.multiarray', 'dtype', ('__gt__',
    """
    __ge__(value, /)

    Return ``self > value``.

    Equivalent to
    ``self != value and np.can_cast(value, self, casting="safe")``.

    See Also
    --------
    can_cast : Returns True if cast between data types can occur according to
               the casting rule.

    """))

add_newdoc('numpy._core.multiarray', 'dtype', ('__lt__',
    """
    __lt__(value, /)

    Return ``self < value``.

    Equivalent to
    ``self != value and np.can_cast(self, value, casting="safe")``.

    See Also
    --------
    can_cast : Returns True if cast between data types can occur according to
               the casting rule.

    """))

##############################################################################
#
# Datetime-related Methods
#
##############################################################################

add_newdoc('numpy._core.multiarray', 'busdaycalendar',
    """
    busdaycalendar(weekmask='1111100', holidays=None)

    A business day calendar object that efficiently stores information
    defining valid days for the busday family of functions.

    The default valid days are Monday through Friday ("business days").
    A busdaycalendar object can be specified with any set of weekly
    valid days, plus an optional "holiday" dates that always will be invalid.

    Once a busdaycalendar object is created, the weekmask and holidays
    cannot be modified.

    .. versionadded:: 1.7.0

    Parameters
    ----------
    weekmask : str, optional
        A string of length 7 representing valid days. '1' represents a valid day
        and '0' an invalid day. By default, Monday through Friday are valid days.
    holidays : array_like of datetime objects, optional
        An array of dates that are not valid days.

    """)
    weekmask : str or array_like of bool, optional
        # 定义参数 weekmask，表示一周中哪些天是有效的工作日
        A seven-element array indicating which of Monday through Sunday are
        # 一个包含七个元素的数组，指示从周一到周日哪些是有效的工作日
        valid days. May be specified as a length-seven list or array, like
        # 可以指定为长度为七的列表或数组，如 [1,1,1,1,1,0,0]
        [1,1,1,1,1,0,0]; a length-seven string, like '1111100'; or a string
        # 或长度为七的字符串，如 '1111100'，或者像 "Mon Tue Wed Thu Fri" 这样的字符串
        like "Mon Tue Wed Thu Fri", made up of 3-character abbreviations for
        # 如 "Mon Tue Wed Thu Fri"，包含星期几的缩写，可选地用空格分隔
        weekdays, optionally separated by white space. Valid abbreviations
        # 有效的缩写包括：Mon Tue Wed Thu Fri Sat Sun
        are: Mon Tue Wed Thu Fri Sat Sun
    holidays : array_like of datetime64[D], optional
        # 定义参数 holidays，表示无效的日期列表，无论它们是周几
        An array of dates to consider as invalid dates, no matter which
        # 一个日期数组，被视为无效日期，无论它们是周几
        weekday they fall upon.  Holiday dates may be specified in any
        # 节假日日期可以以任何顺序指定
        order, and NaT (not-a-time) dates are ignored.  This list is
        # NaT（非时间）日期将被忽略。该列表以适合快速计算有效日期的规范形式保存
        saved in a normalized form that is suited for fast calculations
        # 保存在适合快速计算有效日期的标准化形式中
        of valid days.

    Returns
    -------
    out : busdaycalendar
        # 返回值是一个 busdaycalendar 对象，包含指定的 weekmask 和 holidays 值
        A business day calendar object containing the specified
        # 一个业务日历对象，包含指定的 weekmask 和 holidays 值
        weekmask and holidays values.

    See Also
    --------
    is_busday : Returns a boolean array indicating valid days.
    # 参见 is_busday：返回一个布尔数组，指示有效的工作日
    busday_offset : Applies an offset counted in valid days.
    # busday_offset：应用以有效工作日计算的偏移量
    busday_count : Counts how many valid days are in a half-open date range.
    # busday_count：计算半开放日期范围内有多少有效的工作日

    Attributes
    ----------
    weekmask : (copy) seven-element array of bool
    # 属性 weekmask：（复制）包含七个布尔元素的数组
    holidays : (copy) sorted array of datetime64[D]
    # 属性 holidays：（复制）已排序的 datetime64[D] 类型的数组

    Notes
    -----
    Once a busdaycalendar object is created, you cannot modify the
    # 一旦创建了 busdaycalendar 对象，就无法修改 weekmask 或 holidays
    weekmask or holidays.  The attributes return copies of internal data.
    # 这些属性返回内部数据的副本

    Examples
    --------
    >>> # Some important days in July
    ... bdd = np.busdaycalendar(
    ...             holidays=['2011-07-01', '2011-07-04', '2011-07-17'])
    # 创建一个 busdaycalendar 对象，指定了七月份的一些重要日期作为节假日
    >>> # Default is Monday to Friday weekdays
    ... bdd.weekmask
    # 查看默认的 weekmask，表示默认的工作日是周一到周五
    array([ True,  True,  True,  True,  True, False, False])
    >>> # Any holidays already on the weekend are removed
    ... bdd.holidays
    # 查看去除了已经在周末的节假日
    array(['2011-07-01', '2011-07-04'], dtype='datetime64[D]')
    """
add_newdoc('numpy._core.multiarray', 'busdaycalendar', ('weekmask',
    """A copy of the seven-element boolean mask indicating valid days."""))

add_newdoc('numpy._core.multiarray', 'busdaycalendar', ('holidays',
    """A copy of the holiday array indicating additional invalid days."""))

add_newdoc('numpy._core.multiarray', 'normalize_axis_index',
    """
    normalize_axis_index(axis, ndim, msg_prefix=None)

    Normalizes an axis index, `axis`, such that is a valid positive index into
    the shape of array with `ndim` dimensions. Raises an AxisError with an
    appropriate message if this is not possible.

    Used internally by all axis-checking logic.

    .. versionadded:: 1.13.0

    Parameters
    ----------
    axis : int
        The un-normalized index of the axis. Can be negative
    ndim : int
        The number of dimensions of the array that `axis` should be normalized
        against
    msg_prefix : str
        A prefix to put before the message, typically the name of the argument

    Returns
    -------
    normalized_axis : int
        The normalized axis index, such that `0 <= normalized_axis < ndim`

    Raises
    ------
    AxisError
        If the axis index is invalid, when `-ndim <= axis < ndim` is false.

    Examples
    --------
    >>> from numpy.lib.array_utils import normalize_axis_index
    >>> normalize_axis_index(0, ndim=3)
    0
    >>> normalize_axis_index(1, ndim=3)
    1
    >>> normalize_axis_index(-1, ndim=3)
    2

    >>> normalize_axis_index(3, ndim=3)
    Traceback (most recent call last):
    ...
    numpy.exceptions.AxisError: axis 3 is out of bounds for array ...
    >>> normalize_axis_index(-4, ndim=3, msg_prefix='axes_arg')
    Traceback (most recent call last):
    ...
    numpy.exceptions.AxisError: axes_arg: axis -4 is out of bounds ...
    """)

add_newdoc('numpy._core.multiarray', 'datetime_data',
    """
    datetime_data(dtype, /)

    Get information about the step size of a date or time type.

    The returned tuple can be passed as the second argument of `numpy.datetime64` and
    `numpy.timedelta64`.

    Parameters
    ----------
    dtype : dtype
        The dtype object, which must be a `datetime64` or `timedelta64` type.

    Returns
    -------
    unit : str
        The :ref:`datetime unit <arrays.dtypes.dateunits>` on which this dtype
        is based.
    count : int
        The number of base units in a step.

    Examples
    --------
    >>> dt_25s = np.dtype('timedelta64[25s]')
    >>> np.datetime_data(dt_25s)
    ('s', 25)
    >>> np.array(10, dt_25s).astype('timedelta64[s]')
    array(250, dtype='timedelta64[s]')

    The result can be used to construct a datetime that uses the same units
    as a timedelta

    >>> np.datetime64('2010', np.datetime_data(dt_25s))
    np.datetime64('2010-01-01T00:00:00','25s')
    """)


##############################################################################
#
# Documentation for `generic` attributes and methods
#
##############################################################################

add_newdoc('numpy._core.numerictypes', 'generic',
    """
    Base class for numpy scalar types.

    Class from which most (all?) numpy scalar types are derived.  For
    consistency, exposes the same API as `ndarray`, despite many
    consequent attributes being either "get-only," or completely irrelevant.
    This is the class from which it is strongly suggested users should derive
    custom scalar types.

    """)

# Attributes

def refer_to_array_attribute(attr, method=True):
    # 构建描述标量方法或属性的文档字符串模板
    docstring = """
    Scalar {} identical to the corresponding array attribute.

    Please see `ndarray.{}`.
    """
    # 返回属性名和格式化后的文档字符串
    return attr, docstring.format("method" if method else "attribute", attr)


add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('T', method=False))
# 添加关于标量的文档，引用相应的数组属性

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('base', method=False))
# 添加关于标量的文档，引用相应的数组属性

add_newdoc('numpy._core.numerictypes', 'generic', ('data',
    """Pointer to start of data."""))
# 添加关于标量的文档，描述数据起始指针

add_newdoc('numpy._core.numerictypes', 'generic', ('dtype',
    """Get array data-descriptor."""))
# 添加关于标量的文档，描述数组的数据描述符

add_newdoc('numpy._core.numerictypes', 'generic', ('flags',
    """The integer value of flags."""))
# 添加关于标量的文档，描述标量的整数标志值

add_newdoc('numpy._core.numerictypes', 'generic', ('flat',
    """A 1-D view of the scalar."""))
# 添加关于标量的文档，描述标量的一维视图

add_newdoc('numpy._core.numerictypes', 'generic', ('imag',
    """The imaginary part of the scalar."""))
# 添加关于标量的文档，描述标量的虚部

add_newdoc('numpy._core.numerictypes', 'generic', ('itemsize',
    """The length of one element in bytes."""))
# 添加关于标量的文档，描述一个元素的字节长度

add_newdoc('numpy._core.numerictypes', 'generic', ('ndim',
    """The number of array dimensions."""))
# 添加关于标量的文档，描述数组的维数

add_newdoc('numpy._core.numerictypes', 'generic', ('real',
    """The real part of the scalar."""))
# 添加关于标量的文档，描述标量的实部

add_newdoc('numpy._core.numerictypes', 'generic', ('shape',
    """Tuple of array dimensions."""))
# 添加关于标量的文档，描述数组的维度元组

add_newdoc('numpy._core.numerictypes', 'generic', ('size',
    """The number of elements in the gentype."""))
# 添加关于标量的文档，描述标量的元素数量

add_newdoc('numpy._core.numerictypes', 'generic', ('strides',
    """Tuple of bytes steps in each dimension."""))
# 添加关于标量的文档，描述每个维度的字节步长元组

# Methods

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('all'))
# 添加关于标量的文档，引用数组方法 "all"

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('any'))
# 添加关于标量的文档，引用数组方法 "any"

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('argmax'))
# 添加关于标量的文档，引用数组方法 "argmax"

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('argmin'))
# 添加关于标量的文档，引用数组方法 "argmin"

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('argsort'))
# 添加关于标量的文档，引用数组方法 "argsort"

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('astype'))
# 添加关于标量的文档，引用数组方法 "astype"

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('byteswap'))
# 添加关于标量的文档，引用数组方法 "byteswap"

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('choose'))
# 添加关于标量的文档，引用数组方法 "choose"
# 向 numpy._core.numerictypes 模块的 generic 函数添加新文档，参考数组属性 'clip'
add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('clip'))

# 向 numpy._core.numerictypes 模块的 generic 函数添加新文档，参考数组属性 'compress'
add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('compress'))

# 向 numpy._core.numerictypes 模块的 generic 函数添加新文档，参考数组属性 'conjugate'
add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('conjugate'))

# 向 numpy._core.numerictypes 模块的 generic 函数添加新文档，参考数组属性 'copy'
add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('copy'))

# 向 numpy._core.numerictypes 模块的 generic 函数添加新文档，参考数组属性 'cumprod'
add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('cumprod'))

# 向 numpy._core.numerictypes 模块的 generic 函数添加新文档，参考数组属性 'cumsum'
add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('cumsum'))

# 向 numpy._core.numerictypes 模块的 generic 函数添加新文档，参考数组属性 'diagonal'
add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('diagonal'))

# 向 numpy._core.numerictypes 模块的 generic 函数添加新文档，参考数组属性 'dump'
add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('dump'))

# 向 numpy._core.numerictypes 模块的 generic 函数添加新文档，参考数组属性 'dumps'
add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('dumps'))

# 向 numpy._core.numerictypes 模块的 generic 函数添加新文档，参考数组属性 'fill'
add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('fill'))

# 向 numpy._core.numerictypes 模块的 generic 函数添加新文档，参考数组属性 'flatten'
add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('flatten'))

# 向 numpy._core.numerictypes 模块的 generic 函数添加新文档，参考数组属性 'getfield'
add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('getfield'))

# 向 numpy._core.numerictypes 模块的 generic 函数添加新文档，参考数组属性 'item'
add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('item'))

# 向 numpy._core.numerictypes 模块的 generic 函数添加新文档，参考数组属性 'max'
add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('max'))

# 向 numpy._core.numerictypes 模块的 generic 函数添加新文档，参考数组属性 'mean'
add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('mean'))

# 向 numpy._core.numerictypes 模块的 generic 函数添加新文档，参考数组属性 'min'
add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('min'))

# 向 numpy._core.numerictypes 模块的 generic 函数添加新文档，参考数组属性 'nonzero'
add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('nonzero'))

# 向 numpy._core.numerictypes 模块的 generic 函数添加新文档，参考数组属性 'prod'
add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('prod'))

# 向 numpy._core.numerictypes 模块的 generic 函数添加新文档，参考数组属性 'put'
add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('put'))

# 向 numpy._core.numerictypes 模块的 generic 函数添加新文档，参考数组属性 'ravel'
add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('ravel'))

# 向 numpy._core.numerictypes 模块的 generic 函数添加新文档，参考数组属性 'repeat'
add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('repeat'))

# 向 numpy._core.numerictypes 模块的 generic 函数添加新文档，参考数组属性 'reshape'
add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('reshape'))

# 向 numpy._core.numerictypes 模块的 generic 函数添加新文档，参考数组属性 'resize'
add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('resize'))

# 向 numpy._core.numerictypes 模块的 generic 函数添加新文档，参考数组属性 'round'
add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('round'))

# 向 numpy._core.numerictypes 模块的 generic 函数添加新文档，参考数组属性 'searchsorted'
add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('searchsorted'))

# 向 numpy._core.numerictypes 模块的 generic 函数添加新文档，参考数组属性 'setfield'
add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('setfield'))

# 向 numpy._core.numerictypes 模块的 generic 函数添加新文档，参考数组属性 'setflags'
add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('setflags'))

# 向 numpy._core.numerictypes 模块的 generic 函数添加新文档，参考数组属性 'sort'
add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('sort'))

# 向 numpy._core.numerictypes 模块的 generic 函数添加新文档，参考数组属性 'squeeze'
add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('squeeze'))

# 向 numpy._core.numerictypes 模块的 generic 函数添加新文档，参考数组属性 'std'
add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('std'))
add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('sum'))
# 添加新的文档字符串到 numpy._core.numerictypes.generic，引用数组属性中的 'sum'

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('swapaxes'))
# 添加新的文档字符串到 numpy._core.numerictypes.generic，引用数组属性中的 'swapaxes'

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('take'))
# 添加新的文档字符串到 numpy._core.numerictypes.generic，引用数组属性中的 'take'

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('tofile'))
# 添加新的文档字符串到 numpy._core.numerictypes.generic，引用数组属性中的 'tofile'

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('tolist'))
# 添加新的文档字符串到 numpy._core.numerictypes.generic，引用数组属性中的 'tolist'

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('tostring'))
# 添加新的文档字符串到 numpy._core.numerictypes.generic，引用数组属性中的 'tostring'

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('trace'))
# 添加新的文档字符串到 numpy._core.numerictypes.generic，引用数组属性中的 'trace'

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('transpose'))
# 添加新的文档字符串到 numpy._core.numerictypes.generic，引用数组属性中的 'transpose'

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('var'))
# 添加新的文档字符串到 numpy._core.numerictypes.generic，引用数组属性中的 'var'

add_newdoc('numpy._core.numerictypes', 'generic',
           refer_to_array_attribute('view'))
# 添加新的文档字符串到 numpy._core.numerictypes.generic，引用数组属性中的 'view'

add_newdoc('numpy._core.numerictypes', 'number', ('__class_getitem__',
    """
    __class_getitem__(item, /)

    Return a parametrized wrapper around the `~numpy.number` type.

    .. versionadded:: 1.22

    Returns
    -------
    alias : types.GenericAlias
        A parametrized `~numpy.number` type.

    Examples
    --------
    >>> from typing import Any
    >>> import numpy as np

    >>> np.signedinteger[Any]
    numpy.signedinteger[typing.Any]

    See Also
    --------
    :pep:`585` : Type hinting generics in standard collections.

    """))
# 添加新的文档字符串到 numpy._core.numerictypes.number，定义 __class_getitem__ 方法的用法和返回值信息

##############################################################################
#
# Documentation for scalar type abstract base classes in type hierarchy
#
##############################################################################

add_newdoc('numpy._core.numerictypes', 'number',
    """
    Abstract base class of all numeric scalar types.

    """)
# 添加新的文档字符串到 numpy._core.numerictypes.number，定义所有数值标量类型的抽象基类

add_newdoc('numpy._core.numerictypes', 'integer',
    """
    Abstract base class of all integer scalar types.

    """)
# 添加新的文档字符串到 numpy._core.numerictypes.integer，定义所有整数标量类型的抽象基类

add_newdoc('numpy._core.numerictypes', 'signedinteger',
    """
    Abstract base class of all signed integer scalar types.

    """)
# 添加新的文档字符串到 numpy._core.numerictypes.signedinteger，定义所有有符号整数标量类型的抽象基类

add_newdoc('numpy._core.numerictypes', 'unsignedinteger',
    """
    Abstract base class of all unsigned integer scalar types.

    """)
# 添加新的文档字符串到 numpy._core.numerictypes.unsignedinteger，定义所有无符号整数标量类型的抽象基类

add_newdoc('numpy._core.numerictypes', 'inexact',
    """
    Abstract base class of all numeric scalar types with a (potentially)
    inexact representation of the values in its range, such as
    floating-point numbers.

    """)
# 添加新的文档字符串到 numpy._core.numerictypes.inexact，定义所有具有可能不精确表示值范围的数值标量类型的抽象基类，如浮点数

add_newdoc('numpy._core.numerictypes', 'floating',
    """
    Abstract base class of all floating-point scalar types.

    """)
# 添加新的文档字符串到 numpy._core.numerictypes.floating，定义所有浮点数标量类型的抽象基类

add_newdoc('numpy._core.numerictypes', 'complexfloating',
    """
    Abstract base class of all complex number scalar types that are made up of
    floating-point numbers.

    """)
# 添加新的文档字符串到 numpy._core.numerictypes.complexfloating，定义所有由浮点数构成的复数标量类型的抽象基类

add_newdoc('numpy._core.numerictypes', 'flexible',
    """
    Abstract base class of all scalar types without predefined length.

    """)
# 添加新的文档字符串到 numpy._core.numerictypes.flexible，定义所有没有预定义长度的标量类型的抽象基类
    The actual size of these types depends on the specific `numpy.dtype`
    instantiation.
# 将文档添加到指定的 NumPy 模块和函数
add_newdoc('numpy._core.numerictypes', 'character',
    """
    Abstract base class of all character string scalar types.

    """)

# 创建 StringDType 类的文档字符串
add_newdoc('numpy._core.multiarray', 'StringDType',
    """
    StringDType(*, na_object=np._NoValue, coerce=True)

    Create a StringDType instance.

    StringDType can be used to store UTF-8 encoded variable-width strings in
    a NumPy array.

    Parameters
    ----------
    na_object : object, optional
        Object used to represent missing data. If unset, the array will not
        use a missing data sentinel.
    coerce : bool, optional
        Whether or not items in an array-like passed to an array creation
        function that are neither a str or str subtype should be coerced to
        str. Defaults to True. If set to False, creating a StringDType
        array from an array-like containing entries that are not already
        strings will raise an error.

    Examples
    --------

    >>> from numpy.dtypes import StringDType
    >>> np.array(["hello", "world"], dtype=StringDType())
    array(["hello", "world"], dtype=StringDType())

    >>> arr = np.array(["hello", None, "world"],
    ...                dtype=StringDType(na_object=None))
    >>> arr
    array(["hello", None, "world"], dtype=StringDType(na_object=None))
    >>> arr[1] is None
    True

    >>> arr = np.array(["hello", np.nan, "world"],
    ...                dtype=StringDType(na_object=np.nan))
    >>> np.isnan(arr)
    array([False, True, False])

    >>> np.array([1.2, object(), "hello world"],
    ...          dtype=StringDType(coerce=True))
    ValueError: StringDType only allows string data when string coercion
    is disabled.

    >>> np.array(["hello", "world"], dtype=StringDType(coerce=True))
    array(["hello", "world"], dtype=StringDType(coerce=True))
    """)
```