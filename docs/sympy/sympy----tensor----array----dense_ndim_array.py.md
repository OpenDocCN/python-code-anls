# `D:\src\scipysrc\sympy\sympy\tensor\array\dense_ndim_array.py`

```
import functools  # 导入 functools 模块，用于高阶函数操作
from typing import List  # 导入 List 类型提示

from sympy.core.basic import Basic  # 导入 SymPy 的基础类 Basic
from sympy.core.containers import Tuple  # 导入 SymPy 的元组容器类 Tuple
from sympy.core.singleton import S  # 导入 SymPy 的单例类 S
from sympy.core.sympify import _sympify  # 导入 SymPy 的符号化函数
from sympy.tensor.array.mutable_ndim_array import MutableNDimArray  # 导入 SymPy 的可变多维数组类 MutableNDimArray
from sympy.tensor.array.ndim_array import NDimArray, ImmutableNDimArray, ArrayKind  # 导入 SymPy 的多维数组相关类
from sympy.utilities.iterables import flatten  # 导入 SymPy 的扁平化函数

class DenseNDimArray(NDimArray):  # 定义 DenseNDimArray 类，继承自 NDimArray

    _array: List[Basic]  # 类属性 _array 是一个 Basic 类型的列表

    def __new__(self, *args, **kwargs):
        return ImmutableDenseNDimArray(*args, **kwargs)  # 创建一个 ImmutableDenseNDimArray 实例并返回

    @property
    def kind(self) -> ArrayKind:
        return ArrayKind._union(self._array)  # 返回数组的种类，使用 _array 的元素作为参数调用 ArrayKind._union 方法

    def __getitem__(self, index):
        """
        允许从 N 维数组中获取元素。

        Examples
        ========

        >>> from sympy import MutableDenseNDimArray
        >>> a = MutableDenseNDimArray([0, 1, 2, 3], (2, 2))
        >>> a
        [[0, 1], [2, 3]]
        >>> a[0, 0]
        0
        >>> a[1, 1]
        3
        >>> a[0]
        [0, 1]
        >>> a[1]
        [2, 3]


        符号索引：

        >>> from sympy.abc import i, j
        >>> a[i, j]
        [[0, 1], [2, 3]][i, j]

        替换 `i` 和 `j` 获取元素 `(1, 1)`:

        >>> a[i, j].subs({i: 1, j: 1})
        3

        """
        syindex = self._check_symbolic_index(index)  # 检查是否有符号索引
        if syindex is not None:
            return syindex

        index = self._check_index_for_getitem(index)  # 检查用于获取元素的索引

        if isinstance(index, tuple) and any(isinstance(i, slice) for i in index):
            sl_factors, eindices = self._get_slice_data_for_array_access(index)  # 获取用于数组访问的切片数据
            array = [self._array[self._parse_index(i)] for i in eindices]  # 根据索引获取数组中的元素
            nshape = [len(el) for i, el in enumerate(sl_factors) if isinstance(index[i], slice)]  # 计算新形状
            return type(self)(array, nshape)  # 返回相同类型的新实例，用 array 和 nshape 初始化
        else:
            index = self._parse_index(index)  # 解析索引
            return self._array[index]  # 返回数组中指定索引的元素

    @classmethod
    def zeros(cls, *shape):
        list_length = functools.reduce(lambda x, y: x*y, shape, S.One)  # 计算零数组的总长度
        return cls._new(([0]*list_length,), shape)  # 使用零元组和形状创建新实例

    def tomatrix(self):
        """
        将 MutableDenseNDimArray 转换为 Matrix。只能转换二维数组，否则会引发错误。

        Examples
        ========

        >>> from sympy import MutableDenseNDimArray
        >>> a = MutableDenseNDimArray([1 for i in range(9)], (3, 3))
        >>> b = a.tomatrix()
        >>> b
        Matrix([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]])

        """
        from sympy.matrices import Matrix  # 导入 SymPy 的 Matrix 类

        if self.rank() != 2:  # 如果数组的秩不为2，则引发 ValueError
            raise ValueError('Dimensions must be of size of 2')

        return Matrix(self.shape[0], self.shape[1], self._array)  # 返回一个新的 Matrix 对象，使用数组的形状和元素初始化
    # 定义一个方法 reshape，用于改变 MutableDenseNDimArray 实例的形状
    def reshape(self, *newshape):
        """
        Returns MutableDenseNDimArray instance with new shape. Elements number
        must be        suitable to new shape. The only argument of method sets
        new shape.

        Examples
        ========

        >>> from sympy import MutableDenseNDimArray
        >>> a = MutableDenseNDimArray([1, 2, 3, 4, 5, 6], (2, 3))
        >>> a.shape
        (2, 3)
        >>> a
        [[1, 2, 3], [4, 5, 6]]
        >>> b = a.reshape(3, 2)
        >>> b.shape
        (3, 2)
        >>> b
        [[1, 2], [3, 4], [5, 6]]

        """
        
        # 计算新形状下的总元素个数
        new_total_size = functools.reduce(lambda x,y: x*y, newshape)
        # 检查新形状的总元素个数是否与当前数组的元素总数相等，不相等则引发 ValueError 异常
        if new_total_size != self._loop_size:
            raise ValueError('Expecting reshape size to %d but got prod(%s) = %d' % (
                self._loop_size, str(newshape), new_total_size))

        # 创建并返回一个新的 MutableDenseNDimArray 实例，使用当前数组的数据和新的形状
        # 这里直接调用 type(self) 来创建相同类型的新对象，传入 self._array（当前数据）和 newshape（新形状）
        return type(self)(self._array, newshape)
class ImmutableDenseNDimArray(DenseNDimArray, ImmutableNDimArray): # type: ignore
    # 定义一个不可变的多维数组类，继承自DenseNDimArray和ImmutableNDimArray
    def __new__(cls, iterable, shape=None, **kwargs):
        # 创建新的实例方法，接受可迭代对象和形状作为输入参数
        return cls._new(iterable, shape, **kwargs)

    @classmethod
    def _new(cls, iterable, shape, **kwargs):
        # 私有类方法，用于实际创建新的实例
        shape, flat_list = cls._handle_ndarray_creation_inputs(iterable, shape, **kwargs)
        # 处理多维数组的创建输入，确定形状和扁平化的列表
        shape = Tuple(*map(_sympify, shape))
        # 将形状转换为元组，并对其进行符号化处理
        cls._check_special_bounds(flat_list, shape)
        # 检查特殊边界情况，确保数组边界正确性
        flat_list = flatten(flat_list)
        # 扁平化列表
        flat_list = Tuple(*flat_list)
        # 转换为元组形式
        self = Basic.__new__(cls, flat_list, shape, **kwargs)
        # 调用基础类的构造方法创建实例
        self._shape = shape
        # 设置形状属性
        self._array = list(flat_list)
        # 设置数组属性为扁平化列表的列表形式
        self._rank = len(shape)
        # 设置数组的秩（维度）
        self._loop_size = functools.reduce(lambda x,y: x*y, shape, 1)
        # 计算数组的总元素个数并设置为循环大小
        return self
        # 返回创建的实例对象

    def __setitem__(self, index, value):
        # 禁止修改不可变的N维数组，抛出TypeError异常
        raise TypeError('immutable N-dim array')

    def as_mutable(self):
        # 返回一个可变的版本的多维数组实例
        return MutableDenseNDimArray(self)

    def _eval_simplify(self, **kwargs):
        # 对多维数组进行简化操作，使用Sympy的简化函数
        from sympy.simplify.simplify import simplify
        return self.applyfunc(simplify)

class MutableDenseNDimArray(DenseNDimArray, MutableNDimArray):

    def __new__(cls, iterable=None, shape=None, **kwargs):
        # 创建新的实例方法，接受可迭代对象和形状作为输入参数
        return cls._new(iterable, shape, **kwargs)

    @classmethod
    def _new(cls, iterable, shape, **kwargs):
        # 私有类方法，用于实际创建新的实例
        shape, flat_list = cls._handle_ndarray_creation_inputs(iterable, shape, **kwargs)
        # 处理多维数组的创建输入，确定形状和扁平化的列表
        flat_list = flatten(flat_list)
        # 扁平化列表
        self = object.__new__(cls)
        # 使用object类的构造方法创建实例
        self._shape = shape
        # 设置形状属性
        self._array = list(flat_list)
        # 设置数组属性为扁平化列表的列表形式
        self._rank = len(shape)
        # 设置数组的秩（维度）
        self._loop_size = functools.reduce(lambda x,y: x*y, shape) if shape else len(flat_list)
        # 计算数组的总元素个数并设置为循环大小，如果形状不存在则使用扁平化列表长度
        return self
        # 返回创建的实例对象

    def __setitem__(self, index, value):
        """Allows to set items to MutableDenseNDimArray.

        Examples
        ========

        >>> from sympy import MutableDenseNDimArray
        >>> a = MutableDenseNDimArray.zeros(2,  2)
        >>> a[0,0] = 1
        >>> a[1,1] = 1
        >>> a
        [[1, 0], [0, 1]]

        """
        if isinstance(index, tuple) and any(isinstance(i, slice) for i in index):
            # 如果索引是元组且包含任何切片对象
            value, eindices, slice_offsets = self._get_slice_data_for_array_assignment(index, value)
            # 获取用于数组赋值的切片数据
            for i in eindices:
                other_i = [ind - j for ind, j in zip(i, slice_offsets) if j is not None]
                self._array[self._parse_index(i)] = value[other_i]
                # 根据切片数据更新数组元素值
        else:
            index = self._parse_index(index)
            # 解析索引
            self._setter_iterable_check(value)
            # 检查可迭代值的有效性
            value = _sympify(value)
            # 符号化值
            self._array[index] = value
            # 更新数组元素值为符号化的值

    def as_immutable(self):
        # 返回一个不可变的版本的多维数组实例
        return ImmutableDenseNDimArray(self)

    @property
    def free_symbols(self):
        # 返回数组中所有元素的自由符号集合
        return {i for j in self._array for i in j.free_symbols}
```