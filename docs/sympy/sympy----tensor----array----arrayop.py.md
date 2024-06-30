# `D:\src\scipysrc\sympy\sympy\tensor\array\arrayop.py`

```
import itertools  # 导入 itertools 模块，用于高效的迭代工具
from collections.abc import Iterable  # 导入 Iterable 抽象基类，用于判断对象是否可迭代

from sympy.core._print_helpers import Printable  # 导入 Printable 类，用于协助打印输出
from sympy.core.containers import Tuple  # 导入 Tuple 类，用于处理元组
from sympy.core.function import diff  # 导入 diff 函数，用于求微分
from sympy.core.singleton import S  # 导入 S 单例对象，用于表示数学中的常量

from sympy.core.sympify import _sympify  # 导入 _sympify 函数，用于将输入转换为 SymPy 对象

from sympy.tensor.array.ndim_array import NDimArray  # 导入 NDimArray 类，多维数组的基类
from sympy.tensor.array.dense_ndim_array import DenseNDimArray, ImmutableDenseNDimArray  # 导入密集型多维数组相关类
from sympy.tensor.array.sparse_ndim_array import SparseNDimArray  # 导入稀疏型多维数组类


def _arrayfy(a):
    from sympy.matrices import MatrixBase  # 从 sympy.matrices 模块导入 MatrixBase 类，用于处理矩阵

    if isinstance(a, NDimArray):
        return a  # 如果 a 是 NDimArray 类型，则直接返回 a
    if isinstance(a, (MatrixBase, list, tuple, Tuple)):
        return ImmutableDenseNDimArray(a)  # 如果 a 是 MatrixBase 类型、列表或元组，则返回 ImmutableDenseNDimArray(a)
    return a  # 否则返回 a 本身


def tensorproduct(*args):
    """
    Tensor product among scalars or array-like objects.

    The equivalent operator for array expressions is ``ArrayTensorProduct``,
    which can be used to keep the expression unevaluated.

    Examples
    ========

    >>> from sympy.tensor.array import tensorproduct, Array
    >>> from sympy.abc import x, y, z, t
    >>> A = Array([[1, 2], [3, 4]])
    >>> B = Array([x, y])
    >>> tensorproduct(A, B)
    [[[x, y], [2*x, 2*y]], [[3*x, 3*y], [4*x, 4*y]]]
    >>> tensorproduct(A, x)
    [[x, 2*x], [3*x, 4*x]]
    >>> tensorproduct(A, B, B)
    [[[[x**2, x*y], [x*y, y**2]], [[2*x**2, 2*x*y], [2*x*y, 2*y**2]]], [[[3*x**2, 3*x*y], [3*x*y, 3*y**2]], [[4*x**2, 4*x*y], [4*x*y, 4*y**2]]]]

    Applying this function on two matrices will result in a rank 4 array.

    >>> from sympy import Matrix, eye
    >>> m = Matrix([[x, y], [z, t]])
    >>> p = tensorproduct(eye(3), m)
    >>> p
    [[[[x, y], [z, t]], [[0, 0], [0, 0]], [[0, 0], [0, 0]]], [[[0, 0], [0, 0]], [[x, y], [z, t]], [[0, 0], [0, 0]]], [[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[x, y], [z, t]]]]

    See Also
    ========

    sympy.tensor.array.expressions.array_expressions.ArrayTensorProduct

    """
    from sympy.tensor.array import SparseNDimArray, ImmutableSparseNDimArray  # 导入稀疏型多维数组相关类

    if len(args) == 0:
        return S.One  # 如果参数个数为 0，则返回数学常量 S.One
    if len(args) == 1:
        return _arrayfy(args[0])  # 如果参数个数为 1，则将参数转换为多维数组并返回
    from sympy.tensor.array.expressions.array_expressions import _CodegenArrayAbstract  # 导入数组表达式相关类
    from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct  # 导入数组张量积相关类
    from sympy.tensor.array.expressions.array_expressions import _ArrayExpr  # 导入数组表达式相关类
    from sympy.matrices.expressions.matexpr import MatrixSymbol  # 导入矩阵符号相关类
    if any(isinstance(arg, (_ArrayExpr, _CodegenArrayAbstract, MatrixSymbol)) for arg in args):
        return ArrayTensorProduct(*args)  # 如果参数中有数组表达式、代码生成数组抽象类或矩阵符号类的实例，则返回它们的数组张量积
    if len(args) > 2:
        return tensorproduct(tensorproduct(args[0], args[1]), *args[2:])  # 如果参数个数大于 2，则递归调用 tensorproduct 函数

    # length of args is 2:
    a, b = map(_arrayfy, args)  # 将 args 中的每个元素都转换为多维数组类型

    if not isinstance(a, NDimArray) or not isinstance(b, NDimArray):
        return a*b  # 如果 a 或 b 不是 NDimArray 类型，则返回它们的乘积
    # 如果 a 和 b 都是 SparseNDimArray 类型的实例，则执行以下逻辑
    if isinstance(a, SparseNDimArray) and isinstance(b, SparseNDimArray):
        # 计算数组 b 的长度
        lp = len(b)
        # 使用生成器表达式生成一个新的稀疏数组字典 new_array，
        # 字典的键为 k1*lp + k2，值为 v1*v2，其中 k1 和 k2 分别来自 a 和 b 的稀疏数组元素
        new_array = {k1*lp + k2: v1*v2 for k1, v1 in a._sparse_array.items() for k2, v2 in b._sparse_array.items()}
        # 返回一个新的 ImmutableSparseNDimArray 对象，使用 new_array 和 a、b 的形状来初始化
        return ImmutableSparseNDimArray(new_array, a.shape + b.shape)
    
    # 如果 a 和 b 不都是 SparseNDimArray 类型的实例，则执行以下逻辑
    # 计算 a 和 b 中所有元素的乘积，生成一个产品列表 product_list
    product_list = [i*j for i in Flatten(a) for j in Flatten(b)]
    # 返回一个新的 ImmutableDenseNDimArray 对象，使用 product_list 和 a、b 的形状来初始化
    return ImmutableDenseNDimArray(product_list, a.shape + b.shape)
def _util_contraction_diagonal(array, *contraction_or_diagonal_axes):
    # 将输入的 array 转换为数组格式
    array = _arrayfy(array)

    # 验证 contraction_or_diagonal_axes 参数的有效性：
    taken_dims = set()
    for axes_group in contraction_or_diagonal_axes:
        # 检查每个 axes_group 是否为可迭代对象
        if not isinstance(axes_group, Iterable):
            raise ValueError("collections of contraction/diagonal axes expected")

        # 获取当前 axes_group 对应的维度
        dim = array.shape[axes_group[0]]

        # 检查每个维度是否已经被使用过
        for d in axes_group:
            if d in taken_dims:
                raise ValueError("dimension specified more than once")
            # 检查所选维度是否具有相同的长度
            if dim != array.shape[d]:
                raise ValueError("cannot contract or diagonalize between axes of different dimension")
            taken_dims.add(d)

    # 获取数组的秩（rank）
    rank = array.rank()

    # 计算剩余维度的形状
    remaining_shape = [dim for i, dim in enumerate(array.shape) if i not in taken_dims]

    # 计算累积形状（用于计算绝对位置）
    cum_shape = [0]*rank
    _cumul = 1
    for i in range(rank):
        cum_shape[rank - i - 1] = _cumul
        _cumul *= int(array.shape[rank - i - 1])

    # DEFINITION: by absolute position it is meant the position along the one
    # dimensional array containing all the tensor components.

    # 可能的未来工作：将绝对位置的计算移动到一个类方法中。

    # 确定未收缩索引的绝对位置：
    remaining_indices = [[cum_shape[i]*j for j in range(array.shape[i])]
                         for i in range(rank) if i not in taken_dims]

    # 确定收缩索引的绝对位置：
    summed_deltas = []
    for axes_group in contraction_or_diagonal_axes:
        lidx = []
        for js in range(array.shape[axes_group[0]]):
            lidx.append(sum(cum_shape[ig] * js for ig in axes_group))
        summed_deltas.append(lidx)

    # 返回结果：原始数组、未收缩索引的绝对位置、剩余维度的形状、收缩索引的绝对位置
    return array, remaining_indices, remaining_shape, summed_deltas



def tensorcontraction(array, *contraction_axes):
    """
    Contraction of an array-like object on the specified axes.

    The equivalent operator for array expressions is ``ArrayContraction``,
    which can be used to keep the expression unevaluated.

    Examples
    ========

    >>> from sympy import Array, tensorcontraction
    >>> from sympy import Matrix, eye
    >>> tensorcontraction(eye(3), (0, 1))
    3
    >>> A = Array(range(18), (3, 2, 3))
    >>> A
    [[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]], [[12, 13, 14], [15, 16, 17]]]
    >>> tensorcontraction(A, (0, 2))
    [21, 30]

    Matrix multiplication may be emulated with a proper combination of
    ``tensorcontraction`` and ``tensorproduct``

    >>> from sympy import tensorproduct
    >>> from sympy.abc import a,b,c,d,e,f,g,h
    >>> m1 = Matrix([[a, b], [c, d]])
    >>> m2 = Matrix([[e, f], [g, h]])
    >>> p = tensorproduct(m1, m2)
    >>> p
    [[[[a*e, a*f], [a*g, a*h]], [[b*e, b*f], [b*g, b*h]]], [[[c*e, c*f], [c*g, c*h]], [[d*e, d*f], [d*g, d*h]]]]
    >>> tensorcontraction(p, (1, 2))
    [[a*e + b*g, a*f + b*h], [c*e + d*g, c*f + d*h]]
    >>> m1*m2
    Matrix([
    [a*e + b*g, a*f + b*h],
    """

    # 函数文档字符串：对数组对象在指定轴上的收缩操作进行描述
    # 数组表达式的等效操作符为 ``ArrayContraction``，可用于保持表达式的未评估状态
    # 包含多个示例展示如何使用 `tensorcontraction` 函数进行数组操作
    pass
    # 导入必要的函数和类
    from sympy.tensor.array.expressions.array_expressions import _array_contraction
    from sympy.tensor.array.expressions.array_expressions import _CodegenArrayAbstract
    from sympy.tensor.array.expressions.array_expressions import _ArrayExpr
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    
    # 检查 array 是否属于 _ArrayExpr、_CodegenArrayAbstract 或 MatrixSymbol 类型
    if isinstance(array, (_ArrayExpr, _CodegenArrayAbstract, MatrixSymbol)):
        # 如果是，则调用 _array_contraction 函数处理并返回结果
        return _array_contraction(array, *contraction_axes)

    # 否则，调用 _util_contraction_diagonal 函数处理 array 和 contraction_axes，得到以下变量
    array, remaining_indices, remaining_shape, summed_deltas = _util_contraction_diagonal(array, *contraction_axes)

    # 计算收缩后的数组：
    #
    # 1. 对所有未收缩索引进行外部循环。
    #    未收缩索引由剩余索引的绝对位置的组合积确定。
    # 2. 对所有收缩索引进行内部循环。
    #    对外部循环中绝对收缩索引和绝对未收缩索引的值进行求和。
    contracted_array = []
    for icontrib in itertools.product(*remaining_indices):
        index_base_position = sum(icontrib)
        isum = S.Zero
        for sum_to_index in itertools.product(*summed_deltas):
            idx = array._get_tuple_index(index_base_position + sum(sum_to_index))
            isum += array[idx]

        contracted_array.append(isum)

    # 如果剩余索引的数量为零，确保 contracted_array 的长度为 1，然后返回第一个元素
    if len(remaining_indices) == 0:
        assert len(contracted_array) == 1
        return contracted_array[0]

    # 否则，根据 array 的类型创建一个新的数组对象，并使用 contracted_array 和 remaining_shape 初始化
    return type(array)(contracted_array, remaining_shape)
    # 定义了一个函数，用于对数组对象在指定的轴上进行对角化操作
    """
    Diagonalization of an array-like object on the specified axes.

    This is equivalent to multiplying the expression by Kronecker deltas
    uniting the axes.

    The diagonal indices are put at the end of the axes.

    The equivalent operator for array expressions is ``ArrayDiagonal``, which
    can be used to keep the expression unevaluated.
    """

    # 检查是否有轴的长度小于等于1，如果有则引发值错误异常
    if any(len(i) <= 1 for i in diagonal_axes):
        raise ValueError("need at least two axes to diagonalize")

    # 导入必要的类和函数
    from sympy.tensor.array.expressions.array_expressions import _ArrayExpr
    from sympy.tensor.array.expressions.array_expressions import _CodegenArrayAbstract
    from sympy.tensor.array.expressions.array_expressions import ArrayDiagonal, _array_diagonal
    from sympy.matrices.expressions.matexpr import MatrixSymbol

    # 如果数组是特定类型（_ArrayExpr、_CodegenArrayAbstract、MatrixSymbol）的实例，则调用相应的函数处理后返回结果
    if isinstance(array, (_ArrayExpr, _CodegenArrayAbstract, MatrixSymbol)):
        return _array_diagonal(array, *diagonal_axes)

    # 使用 ArrayDiagonal 类的 _validate 方法验证输入的 array 和 diagonal_axes
    ArrayDiagonal._validate(array, *diagonal_axes)

    # 调用 _util_contraction_diagonal 函数处理 array 和 diagonal_axes，返回处理后的结果
    array, remaining_indices, remaining_shape, diagonal_deltas = _util_contraction_diagonal(array, *diagonal_axes)

    # 计算对角化后的数组：
    #
    # 1. 对所有未对角化的索引进行外部循环。
    #    未对角化的索引由剩余索引的绝对位置的组合乘积确定。
    # 2. 对所有对角线索引进行内部循环。
    #    它附加绝对对角化索引和外部循环的绝对未对角化索引的值。
    diagonalized_array = []
    diagonal_shape = [len(i) for i in diagonal_deltas]
    # 遍历剩余索引的笛卡尔积，生成每个组合的贡献者索引
    for icontrib in itertools.product(*remaining_indices):
        # 计算当前贡献者的基础位置索引
        index_base_position = sum(icontrib)
        isum = []
        # 遍历对角线增量的笛卡尔积，生成每个组合的对角线索引增量
        for sum_to_index in itertools.product(*diagonal_deltas):
            # 计算当前元素在数组中的具体索引
            idx = array._get_tuple_index(index_base_position + sum(sum_to_index))
            # 将对应索引处的元素添加到列表中
            isum.append(array[idx])

        # 将列表转换为与原数组类型相同的对象，并重新塑形为对应的对角线形状
        isum = type(array)(isum).reshape(*diagonal_shape)
        # 将对角线化后的子数组添加到对角线化数组列表中
        diagonalized_array.append(isum)

    # 返回一个与原数组类型相同的对象，其中包含对角线化后的数组和剩余形状的组合
    return type(array)(diagonalized_array, remaining_shape + diagonal_shape)
def permutedims(expr, perm=None, index_order_old=None, index_order_new=None):
    """
    Permutes the indices of an array.

    Parameter specifies the permutation of the indices.

    The equivalent operator for array expressions is ``PermuteDims``, which can
    be used to keep the expression unevaluated.

    Examples
    ========

    >>> from sympy.abc import x, y, z, t  # 导入符号变量 x, y, z, t
    >>> from sympy import sin  # 导入正弦函数 sin
    >>> from sympy import Array, permutedims  # 导入数组类 Array 和排列维度函数 permutedims
    >>> a = Array([[x, y, z], [t, sin(x), 0]])  # 创建一个二维数组 a
    >>> a  # 显示数组 a
    [[x, y, z], [t, sin(x), 0]]
    >>> permutedims(a, (1, 0))  # 对数组 a 进行维度置换，将第一维和第二维交换
    [[x, t], [y, sin(x)], [z, 0]]

    If the array is of second order, ``transpose`` can be used:

    >>> from sympy import transpose  # 导入转置函数 transpose
    >>> transpose(a)  # 对数组 a 进行转置操作
    [[x, t], [y, sin(x)], [z, 0]]

    Examples on higher dimensions:

    >>> b = Array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # 创建一个三维数组 b
    """
    # 导入需要的模块和类
    from sympy.tensor.array import SparseNDimArray

    from sympy.tensor.array.expressions.array_expressions import _ArrayExpr
    from sympy.tensor.array.expressions.array_expressions import _CodegenArrayAbstract
    from sympy.tensor.array.expressions.array_expressions import _permute_dims
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    from sympy.tensor.array.expressions import PermuteDims
    from sympy.tensor.array.expressions.array_expressions import get_rank

    # 从给定参数中获取置换对象
    perm = PermuteDims._get_permutation_from_arguments(perm, index_order_old, index_order_new, get_rank(expr))

    # 如果表达式是 _ArrayExpr、_CodegenArrayAbstract 或 MatrixSymbol 的实例，则调用 _permute_dims 函数进行置换操作
    if isinstance(expr, (_ArrayExpr, _CodegenArrayAbstract, MatrixSymbol)):
        return _permute_dims(expr, perm)

    # 如果表达式不是 NDimArray 的实例，则将其转换为 ImmutableDenseNDimArray 类型
    if not isinstance(expr, NDimArray):
        expr = ImmutableDenseNDimArray(expr)

    # 导入 Permutation 类并确保 perm 是 Permutation 的实例
    from sympy.combinatorics import Permutation
    if not isinstance(perm, Permutation):
        perm = Permutation(list(perm))

    # 检查置换对象的大小是否与表达式的秩相符
    if perm.size != expr.rank():
        raise ValueError("wrong permutation size")

    # 获取置换对象的逆置换并计算新的形状
    iperm = ~perm
    new_shape = perm(expr.shape)

    # 如果表达式是 SparseNDimArray 的实例，则根据新索引形成新的 SparseNDimArray
    if isinstance(expr, SparseNDimArray):
        return type(expr)({tuple(perm(expr._get_tuple_index(k))): v
                           for k, v in expr._sparse_array.items()}, new_shape)

    # 计算所有索引的跨度
    indices_span = perm([range(i) for i in expr.shape])

    # 创建新数组，并对表达式中的元素进行重新排列
    new_array = [None]*len(expr)
    import itertools  # 导入 itertools 模块
    for i, idx in enumerate(itertools.product(*indices_span)):
        t = iperm(idx)  # 应用逆置换
        new_array[i] = expr[t]

    # 返回新的表达式对象，类型与原表达式相同
    return type(expr)(new_array, new_shape)
# 定义一个名为 Flatten 的类，继承自 Printable 类
class Flatten(Printable):
    """
    Flatten an iterable object to a list in a lazy-evaluation way.

    Notes
    =====

    This class is an iterator with which the memory cost can be economised.
    Optimisation has been considered to ameliorate the performance for some
    specific data types like DenseNDimArray and SparseNDimArray.

    Examples
    ========

    >>> from sympy.tensor.array.arrayop import Flatten
    >>> from sympy.tensor.array import Array
    >>> A = Array(range(6)).reshape(2, 3)
    >>> Flatten(A)
    Flatten([[0, 1, 2], [3, 4, 5]])
    >>> [i for i in Flatten(A)]
    [0, 1, 2, 3, 4, 5]
    """

    # 初始化方法，接受一个可迭代对象 iterable 作为参数
    def __init__(self, iterable):
        # 导入需要的类
        from sympy.matrices.matrixbase import MatrixBase
        from sympy.tensor.array import NDimArray

        # 如果 iterable 不是 Iterable 或 MatrixBase 的实例，抛出异常
        if not isinstance(iterable, (Iterable, MatrixBase)):
            raise NotImplementedError("Data type not yet supported")

        # 如果 iterable 是 list 的实例，则将其转换为 NDimArray 对象
        if isinstance(iterable, list):
            iterable = NDimArray(iterable)

        # 设置类的内部属性 _iter 为 iterable，_idx 初始化为 0
        self._iter = iterable
        self._idx = 0

    # 返回迭代器自身
    def __iter__(self):
        return self

    # 实现迭代器的下一个元素的方法
    def __next__(self):
        # 导入需要的类
        from sympy.matrices.matrixbase import MatrixBase

        # 如果 _idx 小于 _iter 的长度
        if len(self._iter) > self._idx:
            # 如果 _iter 是 DenseNDimArray 的实例
            if isinstance(self._iter, DenseNDimArray):
                # 取出 _iter 的 _array 中索引为 _idx 的元素
                result = self._iter._array[self._idx]

            # 如果 _iter 是 SparseNDimArray 的实例
            elif isinstance(self._iter, SparseNDimArray):
                # 如果 _idx 在 _iter 的 _sparse_array 中
                if self._idx in self._iter._sparse_array:
                    # 取出 _iter 的 _sparse_array 中索引为 _idx 的元素
                    result = self._iter._sparse_array[self._idx]
                else:
                    result = 0  # 否则返回 0

            # 如果 _iter 是 MatrixBase 的实例
            elif isinstance(self._iter, MatrixBase):
                # 取出 _iter 中索引为 _idx 的元素
                result = self._iter[self._idx]

            # 如果 _iter 有 '__next__' 属性，说明是可迭代对象
            elif hasattr(self._iter, '__next__'):
                # 获取 _iter 的下一个元素
                result = next(self._iter)

            else:
                # 默认情况下取出 _iter 中索引为 _idx 的元素
                result = self._iter[self._idx]

        else:
            # 如果 _idx 大于等于 _iter 的长度，抛出 StopIteration 异常
            raise StopIteration

        # _idx 加一，准备获取下一个元素
        self._idx += 1
        return result

    # Python 2 兼容方法，调用 __next__() 方法
    def next(self):
        return self.__next__()

    # 返回对象的字符串表示，用于 sympy 的打印输出
    def _sympystr(self, printer):
        return type(self).__name__ + '(' + printer._print(self._iter) + ')'
```