# `.\numpy\numpy\lib\_index_tricks_impl.py`

```
import functools  # 导入 functools 模块，用于创建偏函数
import sys  # 导入 sys 模块，提供对解释器相关的操作
import math  # 导入 math 模块，提供数学函数
import warnings  # 导入 warnings 模块，用于处理警告

import numpy as np  # 导入 NumPy 库，命名为 np
from .._utils import set_module  # 导入 set_module 函数，可能用于设置模块
import numpy._core.numeric as _nx  # 导入 NumPy 内核的 numeric 模块，命名为 _nx
from numpy._core.numeric import ScalarType, array  # 导入 ScalarType 和 array 类型
from numpy._core.numerictypes import issubdtype  # 导入 issubdtype 函数，用于类型检查

import numpy.matrixlib as matrixlib  # 导入 NumPy 的 matrixlib 模块
from numpy._core.multiarray import ravel_multi_index, unravel_index  # 导入 ravel_multi_index 和 unravel_index 函数
from numpy._core import overrides, linspace  # 导入 overrides 和 linspace 函数
from numpy.lib.stride_tricks import as_strided  # 导入 as_strided 函数，用于创建视图
from numpy.lib._function_base_impl import diff  # 导入 diff 函数，用于计算数组的差分

# 创建一个偏函数，用于调度数组函数
array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')

# 定义公开的函数和对象列表
__all__ = [
    'ravel_multi_index', 'unravel_index', 'mgrid', 'ogrid', 'r_', 'c_',
    's_', 'index_exp', 'ix_', 'ndenumerate', 'ndindex', 'fill_diagonal',
    'diag_indices', 'diag_indices_from'
]

def _ix__dispatcher(*args):
    return args

# 使用 array_function_dispatch 装饰器，将 _ix__dispatcher 函数注册为 ix_ 的调度器
@array_function_dispatch(_ix__dispatcher)
def ix_(*args):
    """
    Construct an open mesh from multiple sequences.

    This function takes N 1-D sequences and returns N outputs with N
    dimensions each, such that the shape is 1 in all but one dimension
    and the dimension with the non-unit shape value cycles through all
    N dimensions.

    Using `ix_` one can quickly construct index arrays that will index
    the cross product. ``a[np.ix_([1,3],[2,5])]`` returns the array
    ``[[a[1,2] a[1,5]], [a[3,2] a[3,5]]]``.

    Parameters
    ----------
    args : 1-D sequences
        Each sequence should be of integer or boolean type.
        Boolean sequences will be interpreted as boolean masks for the
        corresponding dimension (equivalent to passing in
        ``np.nonzero(boolean_sequence)``).

    Returns
    -------
    out : tuple of ndarrays
        N arrays with N dimensions each, with N the number of input
        sequences. Together these arrays form an open mesh.

    See Also
    --------
    ogrid, mgrid, meshgrid

    Examples
    --------
    >>> a = np.arange(10).reshape(2, 5)
    >>> a
    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])
    >>> ixgrid = np.ix_([0, 1], [2, 4])
    >>> ixgrid
    (array([[0],
           [1]]), array([[2, 4]]))
    >>> ixgrid[0].shape, ixgrid[1].shape
    ((2, 1), (1, 2))
    >>> a[ixgrid]
    array([[2, 4],
           [7, 9]])

    >>> ixgrid = np.ix_([True, True], [2, 4])
    >>> a[ixgrid]
    array([[2, 4],
           [7, 9]])
    >>> ixgrid = np.ix_([True, True], [False, False, True, False, True])
    >>> a[ixgrid]
    array([[2, 4],
           [7, 9]])

    """
    out = []
    nd = len(args)
    # 遍历参数列表，并使用 enumerate 获取索引和对应的值
    for k, new in enumerate(args):
        # 检查 new 是否不是 numpy 的 ndarray 类型
        if not isinstance(new, _nx.ndarray):
            # 将 new 转换为 numpy 数组
            new = np.asarray(new)
            # 如果 new 是空数组，则显式指定为 _nx.intp 类型以避免默认的浮点数类型
            if new.size == 0:
                new = new.astype(_nx.intp)
        # 检查 new 是否为一维数组，否则抛出 ValueError
        if new.ndim != 1:
            raise ValueError("Cross index must be 1 dimensional")
        # 如果 new 的数据类型是布尔型，将其转换为非零元素的索引
        if issubdtype(new.dtype, _nx.bool):
            new, = new.nonzero()
        # 将 new 调整为特定形状，以符合输出要求
        new = new.reshape((1,)*k + (new.size,) + (1,)*(nd-k-1))
        # 将调整后的 new 添加到输出列表 out 中
        out.append(new)
    # 返回所有调整后的数组组成的元组作为结果
    return tuple(out)
# 定义一个名为 nd_grid 的类，用于构建多维度的“网格”。

class nd_grid:
    """
    Construct a multi-dimensional "meshgrid".

    ``grid = nd_grid()`` creates an instance which will return a mesh-grid
    when indexed.  The dimension and number of the output arrays are equal
    to the number of indexing dimensions.  If the step length is not a
    complex number, then the stop is not inclusive.

    However, if the step length is a **complex number** (e.g. 5j), then the
    integer part of its magnitude is interpreted as specifying the
    number of points to create between the start and stop values, where
    the stop value **is inclusive**.

    If instantiated with an argument of ``sparse=True``, the mesh-grid is
    open (or not fleshed out) so that only one-dimension of each returned
    argument is greater than 1.

    Parameters
    ----------
    sparse : bool, optional
        Whether the grid is sparse or not. Default is False.

    Notes
    -----
    Two instances of `nd_grid` are made available in the NumPy namespace,
    `mgrid` and `ogrid`, approximately defined as::

        mgrid = nd_grid(sparse=False)
        ogrid = nd_grid(sparse=True)

    Users should use these pre-defined instances instead of using `nd_grid`
    directly.
    """

    def __init__(self, sparse=False):
        # 初始化方法，设定类的实例属性 sparse，默认为 False
        self.sparse = sparse
    def __getitem__(self, key):
        try:
            # 初始化一个空列表用于存储步长大小
            size = []
            # 初始化一个包含 0 的列表，模仿 `np.arange` 的行为，并确保数据类型至少和 `np.int_` 一样大
            num_list = [0]
            # 遍历索引器 `key` 中的每一个维度
            for k in range(len(key)):
                # 获取当前维度的步长、起始值和终止值
                step = key[k].step
                start = key[k].start
                stop = key[k].stop
                # 如果起始值为 None，则设为 0
                if start is None:
                    start = 0
                # 如果步长为 None，则设为 1
                if step is None:
                    step = 1
                # 如果步长是复数类型，则取其绝对值作为步长大小，并将其转换为整数
                if isinstance(step, (_nx.complexfloating, complex)):
                    step = abs(step)
                    size.append(int(step))
                else:
                    # 否则，根据起始值和终止值计算出步长大小，并向上取整
                    size.append(
                        int(math.ceil((stop - start) / (step*1.0))))
                # 将起始值、终止值、步长依次加入 `num_list` 中
                num_list += [start, stop, step]
            # 根据 `num_list` 中的数据类型确定结果类型
            typ = _nx.result_type(*num_list)
            # 如果是稀疏矩阵
            if self.sparse:
                # 使用 `_nx.arange` 创建每个维度的索引数组，并指定数据类型
                nn = [_nx.arange(_x, dtype=_t)
                      for _x, _t in zip(size, (typ,)*len(size))]
            else:
                # 否则，使用 `_nx.indices` 创建索引数组
                nn = _nx.indices(size, typ)
            # 遍历索引器 `key` 中的每一个维度
            for k, kk in enumerate(key):
                # 获取当前维度的步长和起始值
                step = kk.step
                start = kk.start
                # 如果起始值为 None，则设为 0
                if start is None:
                    start = 0
                # 如果步长为 None，则设为 1
                if step is None:
                    step = 1
                # 如果步长是复数类型，则取其绝对值作为步长，并计算新的步长值
                if isinstance(step, (_nx.complexfloating, complex)):
                    step = int(abs(step))
                    # 如果步长不为 1，则计算新的步长值
                    if step != 1:
                        step = (kk.stop - start) / float(step - 1)
                # 对当前维度的索引数组进行相应的调整
                nn[k] = (nn[k]*step+start)
            # 如果是稀疏矩阵
            if self.sparse:
                # 创建一个与维度数相同的新轴对象列表
                slobj = [_nx.newaxis]*len(size)
                # 遍历每个维度
                for k in range(len(size)):
                    # 将当前维度的切片对象设为整个范围
                    slobj[k] = slice(None, None)
                    # 对索引数组进行切片操作
                    nn[k] = nn[k][tuple(slobj)]
                    # 恢复原始的新轴对象
                    slobj[k] = _nx.newaxis
                # 返回调整后的索引数组作为元组，表示 ogrid 的结果
                return tuple(nn)
            # 否则，返回调整后的索引数组，表示 mgrid 的结果
            return nn
        # 捕获可能出现的异常：索引错误或类型错误
        except (IndexError, TypeError):
            # 获取步长、终止值和起始值
            step = key.step
            stop = key.stop
            start = key.start
            # 如果起始值为 None，则设为 0
            if start is None:
                start = 0
            # 如果步长是复数类型，则取其绝对值作为步长大小，并计算新的步长值
            if isinstance(step, (_nx.complexfloating, complex)):
                # 防止创建整数数组
                step_float = abs(step)
                step = length = int(step_float)
                # 如果步长不为 1，则计算新的步长值
                if step != 1:
                    step = (key.stop-start)/float(step-1)
                # 确定结果类型
                typ = _nx.result_type(start, stop, step_float)
                # 使用 `_nx.arange` 创建调整后的索引数组，并返回结果
                return _nx.arange(0, length, 1, dtype=typ)*step + start
            else:
                # 否则，使用 `_nx.arange` 创建调整后的索引数组，并返回结果
                return _nx.arange(start, stop, step)
class MGridClass(nd_grid):
    """
    An instance which returns a dense multi-dimensional "meshgrid".

    An instance which returns a dense (or fleshed out) mesh-grid
    when indexed, so that each returned argument has the same shape.
    The dimensions and number of the output arrays are equal to the
    number of indexing dimensions.  If the step length is not a complex
    number, then the stop is not inclusive.

    However, if the step length is a **complex number** (e.g. 5j), then
    the integer part of its magnitude is interpreted as specifying the
    number of points to create between the start and stop values, where
    the stop value **is inclusive**.

    Returns
    -------
    mesh-grid : ndarray
        A single array, containing a set of `ndarray`\ s all of the same
        dimensions stacked along the first axis.

    See Also
    --------
    ogrid : like `mgrid` but returns open (not fleshed out) mesh grids
    meshgrid: return coordinate matrices from coordinate vectors
    r_ : array concatenator
    :ref:`how-to-partition`

    Examples
    --------
    >>> np.mgrid[0:5, 0:5]
    array([[[0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4]],
           [[0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4]]])
    >>> np.mgrid[-1:1:5j]
    array([-1. , -0.5,  0. ,  0.5,  1. ])

    >>> np.mgrid[0:4].shape
    (4,)
    >>> np.mgrid[0:4, 0:5].shape
    (2, 4, 5)
    >>> np.mgrid[0:4, 0:5, 0:6].shape
    (3, 4, 5, 6)

    """

    def __init__(self):
        # 调用父类的初始化方法，设置稀疏模式为 False
        super().__init__(sparse=False)


mgrid = MGridClass()


class OGridClass(nd_grid):
    """
    An instance which returns an open multi-dimensional "meshgrid".

    An instance which returns an open (i.e. not fleshed out) mesh-grid
    when indexed, so that only one dimension of each returned array is
    greater than 1.  The dimension and number of the output arrays are
    equal to the number of indexing dimensions.  If the step length is
    not a complex number, then the stop is not inclusive.

    However, if the step length is a **complex number** (e.g. 5j), then
    the integer part of its magnitude is interpreted as specifying the
    number of points to create between the start and stop values, where
    the stop value **is inclusive**.

    Returns
    -------
    mesh-grid : ndarray or tuple of ndarrays
        If the input is a single slice, returns an array.
        If the input is multiple slices, returns a tuple of arrays, with
        only one dimension not equal to 1.

    See Also
    --------
    mgrid : like `ogrid` but returns dense (or fleshed out) mesh grids
    meshgrid: return coordinate matrices from coordinate vectors
    r_ : array concatenator
    :ref:`how-to-partition`

    Examples
    --------
    >>> from numpy import ogrid
    >>> ogrid[-1:1:5j]
    """

    # 没有初始化方法的定义，继承了父类的初始化行为
    array([-1. , -0.5,  0. ,  0.5,  1. ])
    >>> ogrid[0:5, 0:5]
    (array([[0],
            [1],
            [2],
            [3],
            [4]]),
     array([[0, 1, 2, 3, 4]]))

    """

    # 创建一个包含指定值的一维数组
    # 然后生成一个二维的 meshgrid 结果
    # 第一个数组包含从 0 到 4 的行索引
    # 第二个数组包含从 0 到 4 的列索引
    # 这两个数组可以用来表示一个 5x5 的格点

    def __init__(self):
        # 调用父类的构造函数，并设置稀疏模式为 True
        super().__init__(sparse=True)
# 创建一个名为ogrid的OGridClass类的实例对象
ogrid = OGridClass()


class AxisConcatenator:
    """
    Translates slice objects to concatenation along an axis.

    For detailed documentation on usage, see `r_`.
    """
    # 定义静态方法concatenate，用于数组的连接操作
    concatenate = staticmethod(_nx.concatenate)
    # 定义静态方法makemat，用于生成矩阵对象
    makemat = staticmethod(matrixlib.matrix)

    def __init__(self, axis=0, matrix=False, ndmin=1, trans1d=-1):
        # 初始化函数，设置对象的轴向、矩阵标志、最小维数和1维转置
        self.axis = axis
        self.matrix = matrix
        self.trans1d = trans1d
        self.ndmin = ndmin

    def __len__(self):
        # 返回对象的长度，这里总是返回0
        return 0

# 在这里使用单独的类而不是简单地使用r_ = concatentor(0)等的原因是为了确保在帮助文档中能正确显示doc字符串


class RClass(AxisConcatenator):
    """
    Translates slice objects to concatenation along the first axis.

    This is a simple way to build up arrays quickly. There are two use cases.

    1. If the index expression contains comma separated arrays, then stack
       them along their first axis.
    2. If the index expression contains slice notation or scalars then create
       a 1-D array with a range indicated by the slice notation.

    If slice notation is used, the syntax ``start:stop:step`` is equivalent
    to ``np.arange(start, stop, step)`` inside of the brackets. However, if
    ``step`` is an imaginary number (i.e. 100j) then its integer portion is
    interpreted as a number-of-points desired and the start and stop are
    inclusive. In other words ``start:stop:stepj`` is interpreted as
    ``np.linspace(start, stop, step, endpoint=1)`` inside of the brackets.
    After expansion of slice notation, all comma separated sequences are
    concatenated together.

    Optional character strings placed as the first element of the index
    expression can be used to change the output. The strings 'r' or 'c' result
    in matrix output. If the result is 1-D and 'r' is specified a 1 x N (row)
    matrix is produced. If the result is 1-D and 'c' is specified, then a N x 1
    (column) matrix is produced. If the result is 2-D then both provide the
    same matrix result.

    A string integer specifies which axis to stack multiple comma separated
    arrays along. A string of two comma-separated integers allows indication
    of the minimum number of dimensions to force each entry into as the
    second integer (the axis to concatenate along is still the first integer).

    A string with three comma-separated integers allows specification of the
    axis to concatenate along, the minimum number of dimensions to force the
    entries to, and which axis should contain the start of the arrays which
    are less than the specified number of dimensions. In other words the third
    integer allows you to specify where the 1's should be placed in the shape
    of the arrays that have their shapes upgraded. By default, they are placed
    in the front of the shape tuple. The third argument allows you to specify
    """

    # 这里没有代码需要注释
    """
    定义一个名为 `r_` 的类，用于数组或矩阵的连接操作。

    """
    
    def __init__(self):
        """
        类的初始化方法。

        Parameters
        ----------
        Not a function, so takes no parameters  # 此方法没有参数

        """
        # 调用父类 AxisConcatenator 的初始化方法，将 axis 参数设为 0
        AxisConcatenator.__init__(self, 0)
`
# 创建一个空的 RClass 实例
r_ = RClass()


class CClass(AxisConcatenator):
    """
    Translates slice objects to concatenation along the second axis.

    This is short-hand for ``np.r_['-1,2,0', index expression]``, which is
    useful because of its common occurrence. In particular, arrays will be
    stacked along their last axis after being upgraded to at least 2-D with
    1's post-pended to the shape (column vectors made out of 1-D arrays).

    See Also
    --------
    column_stack : Stack 1-D arrays as columns into a 2-D array.
    r_ : For more detailed documentation.

    Examples
    --------
    >>> np.c_[np.array([1,2,3]), np.array([4,5,6])]
    array([[1, 4],
           [2, 5],
           [3, 6]])
    >>> np.c_[np.array([[1,2,3]]), 0, 0, np.array([[4,5,6]])]
    array([[1, 2, 3, ..., 4, 5, 6]])

    """

    def __init__(self):
        # 使用 AxisConcatenator 的初始化方法初始化当前对象，指定连接的轴为 -1，至少为二维数组，1 维数组转换为列向量
        AxisConcatenator.__init__(self, -1, ndmin=2, trans1d=0)


# 创建一个 CClass 的实例
c_ = CClass()


@set_module('numpy')
class ndenumerate:
    """
    Multidimensional index iterator.

    Return an iterator yielding pairs of array coordinates and values.

    Parameters
    ----------
    arr : ndarray
      Input array.

    See Also
    --------
    ndindex, flatiter

    Examples
    --------
    >>> a = np.array([[1, 2], [3, 4]])
    >>> for index, x in np.ndenumerate(a):
    ...     print(index, x)
    (0, 0) 1
    (0, 1) 2
    (1, 0) 3
    (1, 1) 4

    """

    def __init__(self, arr):
        # 将输入数组转换为 ndarray，并使用 flat 属性创建迭代器
        self.iter = np.asarray(arr).flat

    def __next__(self):
        """
        Standard iterator method, returns the index tuple and array value.

        Returns
        -------
        coords : tuple of ints
            The indices of the current iteration.
        val : scalar
            The array element of the current iteration.

        """
        # 返回当前迭代的索引元组和数组值
        return self.iter.coords, next(self.iter)

    def __iter__(self):
        # 返回迭代器本身
        return self


@set_module('numpy')
class ndindex:
    """
    An N-dimensional iterator object to index arrays.

    Given the shape of an array, an `ndindex` instance iterates over
    the N-dimensional index of the array. At each iteration a tuple
    of indices is returned, the last dimension is iterated over first.

    Parameters
    ----------
    shape : ints, or a single tuple of ints
        The size of each dimension of the array can be passed as
        individual parameters or as the elements of a tuple.

    See Also
    --------
    ndenumerate, flatiter

    Examples
    --------
    Dimensions as individual arguments

    >>> for index in np.ndindex(3, 2, 1):
    ...     print(index)
    (0, 0, 0)
    (0, 1, 0)
    (1, 0, 0)
    (1, 1, 0)
    (2, 0, 0)
    (2, 1, 0)

    Same dimensions - but in a tuple ``(3, 2, 1)``

    >>> for index in np.ndindex((3, 2, 1)):
    ...     print(index)
    (0, 0, 0)
    (0, 1, 0)
    (1, 0, 0)
    (1, 1, 0)
    (2, 0, 0)
    (2, 1, 0)

    """
    # 初始化方法，接受任意数量的形状参数
    def __init__(self, *shape):
        # 如果参数长度为1且为元组，则将其解包为形状参数
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        # 创建一个只含有一个元素的零数组，并根据给定形状和步幅创建一个视图
        x = as_strided(_nx.zeros(1), shape=shape,
                       strides=_nx.zeros_like(shape))
        # 使用 NumPy 的迭代器创建对象，并指定迭代器的属性和顺序
        self._it = _nx.nditer(x, flags=['multi_index', 'zerosize_ok'],
                              order='C')

    # 返回迭代器自身，用于迭代
    def __iter__(self):
        return self

    # 递增多维索引的方法（已废弃，请勿使用）
    def ndincr(self):
        """
        Increment the multi-dimensional index by one.

        This method is for backward compatibility only: do not use.

        .. deprecated:: 1.20.0
            This method has been advised against since numpy 1.8.0, but only
            started emitting DeprecationWarning as of this version.
        """
        # 发出警告信息表明此方法已废弃
        warnings.warn(
            "`ndindex.ndincr()` is deprecated, use `next(ndindex)` instead",
            DeprecationWarning, stacklevel=2)
        # 调用迭代器的下一个方法，更新索引
        next(self)

    # 标准迭代器方法，更新索引并返回当前迭代的索引元组
    def __next__(self):
        """
        Standard iterator method, updates the index and returns the index
        tuple.

        Returns
        -------
        val : tuple of ints
            Returns a tuple containing the indices of the current
            iteration.

        """
        # 调用迭代器的下一个方法，更新索引
        next(self._it)
        # 返回当前迭代的多维索引
        return self._it.multi_index
# 定义一个类 IndexExpression，用于构建数组的索引元组。
class IndexExpression:
    """
    A nicer way to build up index tuples for arrays.

    .. note::
       Use one of the two predefined instances ``index_exp`` or `s_`
       rather than directly using `IndexExpression`.

    For any index combination, including slicing and axis insertion,
    ``a[indices]`` is the same as ``a[np.index_exp[indices]]`` for any
    array `a`. However, ``np.index_exp[indices]`` can be used anywhere
    in Python code and returns a tuple of slice objects that can be
    used in the construction of complex index expressions.

    Parameters
    ----------
    maketuple : bool
        If True, always returns a tuple.

    See Also
    --------
    s_ : Predefined instance without tuple conversion:
       `s_ = IndexExpression(maketuple=False)`.
       The ``index_exp`` is another predefined instance that
       always returns a tuple:
       `index_exp = IndexExpression(maketuple=True)`.

    Notes
    -----
    You can do all this with :class:`slice` plus a few special objects,
    but there's a lot to remember and this version is simpler because
    it uses the standard array indexing syntax.

    Examples
    --------
    >>> np.s_[2::2]
    slice(2, None, 2)
    >>> np.index_exp[2::2]
    (slice(2, None, 2),)

    >>> np.array([0, 1, 2, 3, 4])[np.s_[2::2]]
    array([2, 4])

    """

    # 初始化方法，设置是否返回元组
    def __init__(self, maketuple):
        self.maketuple = maketuple

    # 获取索引项的方法
    def __getitem__(self, item):
        # 如果 maketuple 为 True 且 item 不是元组，则返回元组包装的 item
        if self.maketuple and not isinstance(item, tuple):
            return (item,)
        else:
            return item


# 预定义两个实例：maketuple 为 True 的 index_exp 和 maketuple 为 False 的 s_
index_exp = IndexExpression(maketuple=True)
s_ = IndexExpression(maketuple=False)

# End contribution from Konrad.


# The following functions complement those in twodim_base, but are
# applicable to N-dimensions.


# 定义函数 _fill_diagonal_dispatcher，用于返回一个元组 (a,)
def _fill_diagonal_dispatcher(a, val, wrap=None):
    return (a,)


# 使用装饰器 array_function_dispatch 将 _fill_diagonal_dispatcher 函数与 fill_diagonal 关联起来
@array_function_dispatch(_fill_diagonal_dispatcher)
# 定义函数 fill_diagonal，用于填充任意维度数组的主对角线
def fill_diagonal(a, val, wrap=False):
    """Fill the main diagonal of the given array of any dimensionality.

    For an array `a` with ``a.ndim >= 2``, the diagonal is the list of
    values ``a[i, ..., i]`` with indices ``i`` all identical.  This function
    modifies the input array in-place without returning a value.

    Parameters
    ----------
    a : array, at least 2-D.
      Array whose diagonal is to be filled in-place.
    val : scalar or array_like
      Value(s) to write on the diagonal. If `val` is scalar, the value is
      written along the diagonal. If array-like, the flattened `val` is
      written along the diagonal, repeating if necessary to fill all
      diagonal entries.
    """
    # 如果数组维度小于2，则抛出值错误异常，数组必须至少是二维的
    if a.ndim < 2:
        raise ValueError("array must be at least 2-d")
    # 初始化结束索引为None
    end = None
    # 如果数组维度为2
    if a.ndim == 2:
        # 对于二维数组的常见情况，使用显式且快速的公式。
        # 对于矩形数组，我们接受这种情况。
        step = a.shape[1] + 1
        # 如果wrap选项为False，计算结束索引，以避免对高矩阵进行对角线包装
        if not wrap:
            end = a.shape[1] * a.shape[1]
    else:
        # 对于维度大于2的情况，仅当数组的所有维度相等时，才适用步进公式，因此我们首先进行检查。
        if not np.all(diff(a.shape) == 0):
            raise ValueError("All dimensions of input must be of equal length")
        # 计算步进值，通过累积乘积计算
        step = 1 + (np.cumprod(a.shape[:-1])).sum()

    # 将值写入对角线位置
    a.flat[:end:step] = val
# 设置函数的模块名称为 'numpy'
@set_module('numpy')
# 定义函数 diag_indices，返回用于访问数组主对角线的索引
def diag_indices(n, ndim=2):
    """
    Return the indices to access the main diagonal of an array.

    This returns a tuple of indices that can be used to access the main
    diagonal of an array `a` with ``a.ndim >= 2`` dimensions and shape
    (n, n, ..., n). For ``a.ndim = 2`` this is the usual diagonal, for
    ``a.ndim > 2`` this is the set of indices to access ``a[i, i, ..., i]``
    for ``i = [0..n-1]``.

    Parameters
    ----------
    n : int
      The size, along each dimension, of the arrays for which the returned
      indices can be used.

    ndim : int, optional
      The number of dimensions.

    See Also
    --------
    diag_indices_from

    Notes
    -----
    .. versionadded:: 1.4.0

    Examples
    --------
    Create a set of indices to access the diagonal of a (4, 4) array:

    >>> di = np.diag_indices(4)
    >>> di
    (array([0, 1, 2, 3]), array([0, 1, 2, 3]))
    >>> a = np.arange(16).reshape(4, 4)
    >>> a
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> a[di] = 100
    >>> a
    array([[100,   1,   2,   3],
           [  4, 100,   6,   7],
           [  8,   9, 100,  11],
           [ 12,  13,  14, 100]])

    Now, we create indices to manipulate a 3-D array:

    >>> d3 = np.diag_indices(2, 3)
    >>> d3
    (array([0, 1]), array([0, 1]), array([0, 1]))

    And use it to set the diagonal of an array of zeros to 1:

    >>> a = np.zeros((2, 2, 2), dtype=int)
    >>> a[d3] = 1
    >>> a
    array([[[1, 0],
            [0, 0]],
           [[0, 0],
            [0, 1]]])

    """
    # 创建一个包含 n 个元素的索引数组
    idx = np.arange(n)
    # 返回一个包含 ndim 维度的元组，每个维度都是 idx 数组的副本
    return (idx,) * ndim


# 定义内部函数 _diag_indices_from，返回接收数组的对角线索引
def _diag_indices_from(arr):
    return (arr,)


# 注册函数 diag_indices_from，使用数组函数分发装饰器
@array_function_dispatch(_diag_indices_from)
# 定义函数 diag_indices_from，返回访问 n 维数组主对角线的索引
def diag_indices_from(arr):
    """
    Return the indices to access the main diagonal of an n-dimensional array.

    See `diag_indices` for full details.

    Parameters
    ----------
    arr : array, at least 2-D

    See Also
    --------
    diag_indices

    Notes
    -----
    .. versionadded:: 1.4.0

    Examples
    --------
    
    Create a 4 by 4 array.

    >>> a = np.arange(16).reshape(4, 4)
    >>> a
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    
    Get the indices of the diagonal elements.

    >>> di = np.diag_indices_from(a)
    >>> di
    (array([0, 1, 2, 3]), array([0, 1, 2, 3]))

    >>> a[di]
    array([ 0,  5, 10, 15])

    This is simply syntactic sugar for diag_indices.

    >>> np.diag_indices(a.shape[0])
    (array([0, 1, 2, 3]), array([0, 1, 2, 3]))

    """

    # 如果输入数组的维度小于 2，抛出 ValueError 异常
    if not arr.ndim >= 2:
        raise ValueError("input array must be at least 2-d")
    # 对于超过 2 维的情况，只有所有维度长度相等的数组才能使用步幅公式，因此先进行检查
    if not np.all(diff(arr.shape) == 0):
        raise ValueError("All dimensions of input must be of equal length")
    # 返回一个包含给定数组形状和维度的对角线索引的元组
    return diag_indices(arr.shape[0], arr.ndim)
```