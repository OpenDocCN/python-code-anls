# `D:\src\scipysrc\sympy\sympy\tensor\array\ndim_array.py`

```
# 导入 SymPy 库中的各个模块和类
from sympy.core.basic import Basic
from sympy.core.containers import (Dict, Tuple)
from sympy.core.expr import Expr
from sympy.core.kind import Kind, NumberKind, UndefinedKind
from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.external.gmpy import SYMPY_INTS
from sympy.printing.defaults import Printable

# 导入 itertools 库，用于迭代操作
import itertools
# 导入 collections.abc 库中的 Iterable 类
from collections.abc import Iterable

# 定义 ArrayKind 类，表示 SymPy 中的 N 维数组的类型
class ArrayKind(Kind):
    """
    Kind for N-dimensional array in SymPy.

    This kind represents the multidimensional array that algebraic
    operations are defined. Basic class for this kind is ``NDimArray``,
    but any expression representing the array can have this.

    Parameters
    ==========

    element_kind : Kind
        Kind of the element. Default is :obj:NumberKind `<sympy.core.kind.NumberKind>`,
        which means that the array contains only numbers.

    Examples
    ========

    Any instance of array class has ``ArrayKind``.

    >>> from sympy import NDimArray
    >>> NDimArray([1,2,3]).kind
    ArrayKind(NumberKind)

    Although expressions representing an array may be not instance of
    array class, it will have ``ArrayKind`` as well.

    >>> from sympy import Integral
    >>> from sympy.tensor.array import NDimArray
    >>> from sympy.abc import x
    >>> intA = Integral(NDimArray([1,2,3]), x)
    >>> isinstance(intA, NDimArray)
    False
    >>> intA.kind
    ArrayKind(NumberKind)

    Use ``isinstance()`` to check for ``ArrayKind` without specifying
    the element kind. Use ``is`` with specifying the element kind.

    >>> from sympy.tensor.array import ArrayKind
    >>> from sympy.core import NumberKind
    >>> boolA = NDimArray([True, False])
    >>> isinstance(boolA.kind, ArrayKind)
    True
    >>> boolA.kind is ArrayKind(NumberKind)
    False

    See Also
    ========

    shape : Function to return the shape of objects with ``MatrixKind``.

    """
    # 创建新的 ArrayKind 实例
    def __new__(cls, element_kind=NumberKind):
        obj = super().__new__(cls, element_kind)
        obj.element_kind = element_kind
        return obj

    # 返回 ArrayKind 的字符串表示形式
    def __repr__(self):
        return "ArrayKind(%s)" % self.element_kind

    # 类方法，用于将一组 ArrayKind 实例合并为一个 ArrayKind
    @classmethod
    def _union(cls, kinds) -> 'ArrayKind':
        elem_kinds = {e.kind for e in kinds}
        if len(elem_kinds) == 1:
            elemkind, = elem_kinds
        else:
            elemkind = UndefinedKind
        return ArrayKind(elemkind)


# 定义 NDimArray 类，表示 SymPy 中的 N 维数组
class NDimArray(Printable):
    """N-dimensional array.

    Examples
    ========

    Create an N-dim array of zeros:

    >>> from sympy import MutableDenseNDimArray
    >>> a = MutableDenseNDimArray.zeros(2, 3, 4)
    >>> a
    [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]

    Create an N-dim array from a list;

    >>> a = MutableDenseNDimArray([[2, 3], [4, 5]])
    >>> a
    [[2, 3], [4, 5]]

    >>> b = MutableDenseNDimArray([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]])
    >>> b
    """

    # NDimArray 类的主要功能和示例说明
    pass  # 此处省略了类的具体实现，未提供额外的代码内容
    # 设置 `_diff_wrt` 变量为 True，表示此类具有对不同变量的微分能力
    _diff_wrt = True
    
    # 设置 `is_scalar` 变量为 False，表明此类不是标量
    is_scalar = False
    
    # 定义类的构造函数 `__new__`，返回一个不可变的 N 维数组
    def __new__(cls, iterable, shape=None, **kwargs):
        # 导入必要的类和函数
        from sympy.tensor.array import ImmutableDenseNDimArray
        # 调用 ImmutableDenseNDimArray 构造函数
        return ImmutableDenseNDimArray(iterable, shape, **kwargs)
    
    # 定义类的 `__getitem__` 方法，抛出未实现错误
    def __getitem__(self, index):
        raise NotImplementedError("A subclass of NDimArray should implement __getitem__")
    
    # 定义类的 `_parse_index` 方法，用于解析索引
    def _parse_index(self, index):
        # 如果索引是整数或者符号整数
        if isinstance(index, (SYMPY_INTS, Integer)):
            # 如果索引超过了循环大小，则引发值错误
            if index >= self._loop_size:
                raise ValueError("Only a tuple index is accepted")
            return index
    
        # 如果数组大小为0，则引发值错误
        if self._loop_size == 0:
            raise ValueError("Index not valid with an empty array")
    
        # 如果索引的维度与数组的秩不匹配，则引发值错误
        if len(index) != self._rank:
            raise ValueError('Wrong number of array axes')
    
        real_index = 0
        # 检查输入的索引是否在当前索引范围内
        for i in range(self._rank):
            if (index[i] >= self.shape[i]) or (index[i] < -self.shape[i]):
                raise ValueError('Index ' + str(index) + ' out of border')
            if index[i] < 0:
                real_index += 1
            real_index = real_index*self.shape[i] + index[i]
    
        return real_index
    
    # 定义类的 `_get_tuple_index` 方法，用于获取元组索引
    def _get_tuple_index(self, integer_index):
        index = []
        # 遍历数组的形状
        for sh in reversed(self.shape):
            # 将整数索引转换为元组索引
            index.append(integer_index % sh)
            integer_index //= sh
        index.reverse()
        return tuple(index)
    
    # 定义类的 `_check_symbolic_index` 方法，用于检查符号索引
    def _check_symbolic_index(self, index):
        # 检查索引是否含有符号数值
        tuple_index = (index if isinstance(index, tuple) else (index,))
        if any((isinstance(i, Expr) and (not i.is_number)) for i in tuple_index):
            for i, nth_dim in zip(tuple_index, self.shape):
                # 如果索引小于0或大于数组维度，则引发值错误
                if ((i < 0) == True) or ((i >= nth_dim) == True):
                    raise ValueError("index out of range")
            # 导入 Indexed 类并返回索引对象
            from sympy.tensor import Indexed
            return Indexed(self, *tuple_index)
        return None
    
    # 定义类的 `_setter_iterable_check` 方法，用于检查设置的可迭代对象
    def _setter_iterable_check(self, value):
        # 导入必要的类
        from sympy.matrices.matrixbase import MatrixBase
        # 如果值是可迭代的、矩阵基类或者 N 维数组，则引发未实现错误
        if isinstance(value, (Iterable, MatrixBase, NDimArray)):
            raise NotImplementedError
    
    # 定义类的类方法
    @classmethod
    # 定义一个类方法，用于递归地确定可迭代对象的形状
    def _scan_iterable_shape(cls, iterable):
        # 内部函数f用于递归扫描指针指向的对象
        def f(pointer):
            # 如果指针不是可迭代的，则将其视为单个元素的列表，并返回空的形状元组
            if not isinstance(pointer, Iterable):
                return [pointer], ()

            # 如果指针是空的可迭代对象，则返回空列表和表示零长度的元组
            if len(pointer) == 0:
                return [], (0,)

            # 否则，递归调用f函数来处理指针中的每个元素，并整理结果
            result = []
            elems, shapes = zip(*[f(i) for i in pointer])
            # 检查所有元素的形状是否一致，如果不一致则抛出异常
            if len(set(shapes)) != 1:
                raise ValueError("could not determine shape unambiguously")
            for i in elems:
                result.extend(i)
            # 返回结果列表和形状元组
            return result, (len(shapes),)+shapes[0]

        # 初始调用内部函数f并返回其结果
        return f(iterable)

    # 类方法，处理创建N维数组的输入参数
    def _handle_ndarray_creation_inputs(cls, iterable=None, shape=None, **kwargs):
        # 导入必要的模块
        from sympy.matrices.matrixbase import MatrixBase
        from sympy.tensor.array import SparseNDimArray

        # 如果未提供形状，则根据iterable类型做不同处理
        if shape is None:
            if iterable is None:
                shape = ()
                iterable = ()
            # 如果iterable是SparseNDimArray类型，则直接返回其形状和稀疏数组
            elif isinstance(iterable, SparseNDimArray):
                return iterable._shape, iterable._sparse_array

            # 如果iterable是NDimArray类型，则获取其形状
            elif isinstance(iterable, NDimArray):
                shape = iterable.shape

            # 如果iterable是可迭代对象（包括numpy数组），则调用_scan_iterable_shape方法获取形状和整理后的可迭代对象
            elif isinstance(iterable, Iterable):
                iterable, shape = cls._scan_iterable_shape(iterable)

            # 如果iterable是MatrixBase类型，则获取其形状
            elif isinstance(iterable, MatrixBase):
                shape = iterable.shape

            # 否则，形状为空，将iterable视为单个元素的元组
            else:
                shape = ()
                iterable = (iterable,)

        # 如果iterable是字典类型且提供了形状，则对字典进行重新组织以支持基于多维索引的访问
        if isinstance(iterable, (Dict, dict)) and shape is not None:
            new_dict = iterable.copy()
            for k in new_dict:
                if isinstance(k, (tuple, Tuple)):
                    new_key = 0
                    for i, idx in enumerate(k):
                        new_key = new_key * shape[i] + idx
                    iterable[new_key] = iterable[k]
                    del iterable[k]

        # 如果形状中包含SymPy整数类型，则转换为元组形式
        if isinstance(shape, (SYMPY_INTS, Integer)):
            shape = (shape,)

        # 检查形状中的每个维度是否都是SymPy整数或Python整数类型，否则抛出类型错误
        if not all(isinstance(dim, (SYMPY_INTS, Integer)) for dim in shape):
            raise TypeError("Shape should contain integers only.")

        # 返回最终确定的形状元组和整理后的可迭代对象
        return tuple(shape), iterable

    # 实现特殊方法__len__()，返回数组中元素的数量
    def __len__(self):
        """Overload common function len(). Returns number of elements in array.

        Examples
        ========

        >>> from sympy import MutableDenseNDimArray
        >>> a = MutableDenseNDimArray.zeros(3, 3)
        >>> a
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        >>> len(a)
        9

        """
        return self._loop_size

    # 定义一个属性方法
    @property
    def shape(self):
        """
        返回数组的形状（维度）。

        示例
        ========

        >>> from sympy import MutableDenseNDimArray
        >>> a = MutableDenseNDimArray.zeros(3, 3)
        >>> a.shape
        (3, 3)

        """
        return self._shape

    def rank(self):
        """
        返回数组的秩（rank）。

        示例
        ========

        >>> from sympy import MutableDenseNDimArray
        >>> a = MutableDenseNDimArray.zeros(3,4,5,6,3)
        >>> a.rank()
        5

        """
        return self._rank

    def diff(self, *args, **kwargs):
        """
        计算数组中每个元素的导数。

        示例
        ========

        >>> from sympy import ImmutableDenseNDimArray
        >>> from sympy.abc import x, y
        >>> M = ImmutableDenseNDimArray([[x, y], [1, x*y]])
        >>> M.diff(x)
        [[1, 0], [0, y]]

        """
        from sympy.tensor.array.array_derivatives import ArrayDerivative
        kwargs.setdefault('evaluate', True)
        return ArrayDerivative(self.as_immutable(), *args, **kwargs)

    def _eval_derivative(self, base):
        # 类型为 (base: scalar, self: array)
        return self.applyfunc(lambda x: base.diff(x))

    def _eval_derivative_n_times(self, s, n):
        return Basic._eval_derivative_n_times(self, s, n)

    def applyfunc(self, f):
        """对 N 维数组的每个元素应用函数 f。

        示例
        ========

        >>> from sympy import ImmutableDenseNDimArray
        >>> m = ImmutableDenseNDimArray([i*2+j for i in range(2) for j in range(2)], (2, 2))
        >>> m
        [[0, 1], [2, 3]]
        >>> m.applyfunc(lambda i: 2*i)
        [[0, 2], [4, 6]]
        """
        from sympy.tensor.array import SparseNDimArray
        from sympy.tensor.array.arrayop import Flatten

        if isinstance(self, SparseNDimArray) and f(S.Zero) == 0:
            return type(self)({k: f(v) for k, v in self._sparse_array.items() if f(v) != 0}, self.shape)

        return type(self)(map(f, Flatten(self)), self.shape)

    def _sympystr(self, printer):
        def f(sh, shape_left, i, j):
            if len(shape_left) == 1:
                return "["+", ".join([printer._print(self[self._get_tuple_index(e)]) for e in range(i, j)])+"]"

            sh //= shape_left[0]
            return "[" + ", ".join([f(sh, shape_left[1:], i+e*sh, i+(e+1)*sh) for e in range(shape_left[0])]) + "]"

        if self.rank() == 0:
            return printer._print(self[()])

        return f(self._loop_size, self.shape, 0, self._loop_size)
    def tolist(self):
        """
        将 MutableDenseNDimArray 转换为一维列表

        Examples
        ========

        >>> from sympy import MutableDenseNDimArray
        >>> a = MutableDenseNDimArray([1, 2, 3, 4], (2, 2))
        >>> a
        [[1, 2], [3, 4]]
        >>> b = a.tolist()
        >>> b
        [[1, 2], [3, 4]]
        """

        def f(sh, shape_left, i, j):
            # 如果剩余的形状长度为1，返回当前切片的元素列表
            if len(shape_left) == 1:
                return [self[self._get_tuple_index(e)] for e in range(i, j)]
            result = []
            sh //= shape_left[0]
            # 对于当前维度的每个索引，递归调用f函数并添加结果到result中
            for e in range(shape_left[0]):
                result.append(f(sh, shape_left[1:], i+e*sh, i+(e+1)*sh))
            return result

        return f(self._loop_size, self.shape, 0, self._loop_size)

    def __add__(self, other):
        from sympy.tensor.array.arrayop import Flatten

        # 如果other不是NDimArray的实例，返回NotImplemented
        if not isinstance(other, NDimArray):
            return NotImplemented

        # 如果形状不匹配，引发数值错误
        if self.shape != other.shape:
            raise ValueError("array shape mismatch")
        # 使用Flatten函数将两个数组扁平化后逐元素相加，返回一个新的数组实例
        result_list = [i+j for i,j in zip(Flatten(self), Flatten(other))]

        return type(self)(result_list, self.shape)

    def __sub__(self, other):
        from sympy.tensor.array.arrayop import Flatten

        # 如果other不是NDimArray的实例，返回NotImplemented
        if not isinstance(other, NDimArray):
            return NotImplemented

        # 如果形状不匹配，引发数值错误
        if self.shape != other.shape:
            raise ValueError("array shape mismatch")
        # 使用Flatten函数将两个数组扁平化后逐元素相减，返回一个新的数组实例
        result_list = [i-j for i,j in zip(Flatten(self), Flatten(other))]

        return type(self)(result_list, self.shape)

    def __mul__(self, other):
        from sympy.matrices.matrixbase import MatrixBase
        from sympy.tensor.array import SparseNDimArray
        from sympy.tensor.array.arrayop import Flatten

        # 如果other是可迭代对象、NDimArray或MatrixBase的实例，引发数值错误
        if isinstance(other, (Iterable, NDimArray, MatrixBase)):
            raise ValueError("scalar expected, use tensorproduct(...) for tensorial product")

        # 将other转换为符号表达式
        other = sympify(other)
        # 如果self是SparseNDimArray的实例
        if isinstance(self, SparseNDimArray):
            # 如果other为零，返回一个空的SparseNDimArray实例
            if other.is_zero:
                return type(self)({}, self.shape)
            # 返回一个乘以other的每个值的SparseNDimArray实例
            return type(self)({k: other*v for (k, v) in self._sparse_array.items()}, self.shape)

        # 使用Flatten函数将数组扁平化后，每个元素乘以other，返回一个新的数组实例
        result_list = [i*other for i in Flatten(self)]
        return type(self)(result_list, self.shape)

    def __rmul__(self, other):
        from sympy.matrices.matrixbase import MatrixBase
        from sympy.tensor.array import SparseNDimArray
        from sympy.tensor.array.arrayop import Flatten

        # 如果other是可迭代对象、NDimArray或MatrixBase的实例，引发数值错误
        if isinstance(other, (Iterable, NDimArray, MatrixBase)):
            raise ValueError("scalar expected, use tensorproduct(...) for tensorial product")

        # 将other转换为符号表达式
        other = sympify(other)
        # 如果self是SparseNDimArray的实例
        if isinstance(self, SparseNDimArray):
            # 如果other为零，返回一个空的SparseNDimArray实例
            if other.is_zero:
                return type(self)({}, self.shape)
            # 返回一个乘以other的每个值的SparseNDimArray实例
            return type(self)({k: other*v for (k, v) in self._sparse_array.items()}, self.shape)

        # 使用Flatten函数将数组扁平化后，other乘以每个元素，返回一个新的数组实例
        result_list = [other*i for i in Flatten(self)]
        return type(self)(result_list, self.shape)
    def __truediv__(self, other):
        # 导入所需模块和类
        from sympy.matrices.matrixbase import MatrixBase
        from sympy.tensor.array import SparseNDimArray
        from sympy.tensor.array.arrayop import Flatten

        # 如果 other 是可迭代对象、NDimArray 或 MatrixBase 的实例，则引发数值错误
        if isinstance(other, (Iterable, NDimArray, MatrixBase)):
            raise ValueError("scalar expected")

        # 将 other 转换为 SymPy 表达式
        other = sympify(other)
        
        # 如果 self 是 SparseNDimArray 类型且 other 不为零，则返回对应操作后的 SparseNDimArray 对象
        if isinstance(self, SparseNDimArray) and other != S.Zero:
            return type(self)({k: v/other for (k, v) in self._sparse_array.items()}, self.shape)

        # 对 self 进行展平后，将每个元素除以 other，然后重新构造相同类型的对象返回
        result_list = [i/other for i in Flatten(self)]
        return type(self)(result_list, self.shape)

    def __rtruediv__(self, other):
        # 未实现的操作，抛出未实现错误
        raise NotImplementedError('unsupported operation on NDimArray')

    def __neg__(self):
        # 导入所需模块和类
        from sympy.tensor.array import SparseNDimArray
        from sympy.tensor.array.arrayop import Flatten

        # 如果 self 是 SparseNDimArray 类型，则返回对应操作后的 SparseNDimArray 对象
        if isinstance(self, SparseNDimArray):
            return type(self)({k: -v for (k, v) in self._sparse_array.items()}, self.shape)

        # 对 self 进行展平后，将每个元素取负值，然后重新构造相同类型的对象返回
        result_list = [-i for i in Flatten(self)]
        return type(self)(result_list, self.shape)

    def __iter__(self):
        # 定义迭代器函数
        def iterator():
            # 如果 self 的形状不为空，则迭代每一个索引并返回相应元素
            if self._shape:
                for i in range(self._shape[0]):
                    yield self[i]
            else:
                # 如果形状为空，则返回单个元素的迭代器
                yield self[()]

        return iterator()

    def __eq__(self, other):
        """
        NDimArray instances can be compared to each other.
        Instances equal if they have same shape and data.

        Examples
        ========

        >>> from sympy import MutableDenseNDimArray
        >>> a = MutableDenseNDimArray.zeros(2, 3)
        >>> b = MutableDenseNDimArray.zeros(2, 3)
        >>> a == b
        True
        >>> c = a.reshape(3, 2)
        >>> c == b
        False
        >>> a[0,0] = 1
        >>> b[0,0] = 2
        >>> a == b
        False
        """
        # 导入所需模块和类
        from sympy.tensor.array import SparseNDimArray
        
        # 如果 other 不是 NDimArray 类型，则返回 False
        if not isinstance(other, NDimArray):
            return False

        # 如果 self 和 other 的形状不同，则返回 False
        if not self.shape == other.shape:
            return False

        # 如果 self 和 other 都是 SparseNDimArray 类型，则比较它们的稀疏数组内容是否相等
        if isinstance(self, SparseNDimArray) and isinstance(other, SparseNDimArray):
            return dict(self._sparse_array) == dict(other._sparse_array)

        # 否则，比较 self 和 other 的展平结果是否相同
        return list(self) == list(other)

    def __ne__(self, other):
        # 返回非相等的结果
        return not self == other

    def _eval_transpose(self):
        # 如果数组的秩不是 2，则引发值错误
        if self.rank() != 2:
            raise ValueError("array rank not 2")
        # 导入排列维度操作函数并返回其结果
        from .arrayop import permutedims
        return permutedims(self, (1, 0))

    def transpose(self):
        # 返回调用 _eval_transpose 方法的结果
        return self._eval_transpose()

    def _eval_conjugate(self):
        # 导入展平操作函数
        from sympy.tensor.array.arrayop import Flatten

        # 对数组中每个元素取共轭，并返回相同类型的对象
        return self.func([i.conjugate() for i in Flatten(self)], self.shape)

    def conjugate(self):
        # 返回调用 _eval_conjugate 方法的结果
        return self._eval_conjugate()

    def _eval_adjoint(self):
        # 返回数组的转置共轭
        return self.transpose().conjugate()

    def adjoint(self):
        # 返回调用 _eval_adjoint 方法的结果
        return self._eval_adjoint()
    # 对给定的切片 `s` 进行扩展，使其符合给定的维度 `dim`
    def _slice_expand(self, s, dim):
        # 如果 `s` 不是切片对象，则将其转换为只包含 `s` 的元组
        if not isinstance(s, slice):
                return (s,)
        # 获取切片的起始、结束和步长，确保其在给定维度 `dim` 内
        start, stop, step = s.indices(dim)
        # 返回一个列表，包含按照给定步长扩展后的所有索引值
        return [start + i*step for i in range((stop-start)//step)]

    # 为数组访问操作获取切片数据和扩展后的索引
    def _get_slice_data_for_array_access(self, index):
        # 对每一个索引 `i` 和对应的维度 `dim`，进行切片扩展操作
        sl_factors = [self._slice_expand(i, dim) for (i, dim) in zip(index, self.shape)]
        # 使用 itertools 的 product 函数，生成所有可能的索引组合 `eindices`
        eindices = itertools.product(*sl_factors)
        return sl_factors, eindices

    # 为数组赋值操作获取切片数据、赋值的值以及扩展后的索引偏移
    def _get_slice_data_for_array_assignment(self, index, value):
        # 如果赋值的值 `value` 不是 NDimArray 类型，则将其转换为当前对象的类型
        if not isinstance(value, NDimArray):
            value = type(self)(value)
        # 获取数组访问操作所需的切片数据和扩展后的所有索引 `eindices`
        sl_factors, eindices = self._get_slice_data_for_array_access(index)
        # 计算每个维度的最小索引值作为切片的偏移量 `slice_offsets`
        slice_offsets = [min(i) if isinstance(i, list) else None for i in sl_factors]
        # 返回转换后的值 `value`、所有可能的索引组合 `eindices` 和切片的偏移量 `slice_offsets`
        # TODO: 添加对赋值给 `value` 的维度检查？
        return value, eindices, slice_offsets

    # 类方法：检查特殊边界条件，确保平坦列表 `flat_list` 符合给定的数组形状 `shape`
    @classmethod
    def _check_special_bounds(cls, flat_list, shape):
        # 如果数组形状是空元组 `()`，但是平坦列表 `flat_list` 的长度不为1，则抛出 ValueError 异常
        if shape == () and len(flat_list) != 1:
            raise ValueError("arrays without shape need one scalar value")
        # 如果数组形状是 `(0,)`，但是平坦列表 `flat_list` 的长度大于0，则抛出 ValueError 异常
        if shape == (0,) and len(flat_list) > 0:
            raise ValueError("if array shape is (0,) there cannot be elements")

    # 检查用于 `__getitem__` 操作的索引是否符合要求
    def _check_index_for_getitem(self, index):
        # 如果索引 `index` 是整数、SYMPY_INTS、Integer 类型或者切片对象，则将其转换为包含 `index` 的元组
        if isinstance(index, (SYMPY_INTS, Integer, slice)):
            index = (index,)
        # 如果索引的长度小于数组的秩（rank），则用 `None` 填充，直到其长度等于数组的秩
        if len(index) < self.rank():
            index = tuple(index) + \
                          tuple(slice(None) for i in range(len(index), self.rank()))
        # 如果索引的长度大于数组的秩，则抛出 ValueError 异常
        if len(index) > self.rank():
            raise ValueError('Dimension of index greater than rank of array')

        return index
class ImmutableNDimArray(NDimArray, Basic):
    # 设定操作优先级为11.0，用于指定运算符的优先级顺序
    _op_priority = 11.0

    # 实现 __hash__ 方法，用于返回对象的哈希值
    def __hash__(self):
        return Basic.__hash__(self)

    # 返回当前对象自身，因为它已经是不可变的
    def as_immutable(self):
        return self

    # 抛出 NotImplementedError 异常，表明该方法为抽象方法，需要在子类中实现
    def as_mutable(self):
        raise NotImplementedError("abstract method")
```