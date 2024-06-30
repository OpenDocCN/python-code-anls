# `D:\src\scipysrc\sympy\sympy\tensor\array\sparse_ndim_array.py`

```
from sympy.core.basic import Basic
from sympy.core.containers import (Dict, Tuple)
from sympy.core.singleton import S
from sympy.core.sympify import _sympify
from sympy.tensor.array.mutable_ndim_array import MutableNDimArray
from sympy.tensor.array.ndim_array import NDimArray, ImmutableNDimArray
from sympy.utilities.iterables import flatten

import functools

class SparseNDimArray(NDimArray):
    # 继承自 NDimArray 的稀疏多维数组类

    def __new__(self, *args, **kwargs):
        # 返回一个不可变的稀疏多维数组对象
        return ImmutableSparseNDimArray(*args, **kwargs)

    def __getitem__(self, index):
        """
        Get an element from a sparse N-dim array.

        Examples
        ========

        >>> from sympy import MutableSparseNDimArray
        >>> a = MutableSparseNDimArray(range(4), (2, 2))
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

        Symbolic indexing:

        >>> from sympy.abc import i, j
        >>> a[i, j]
        [[0, 1], [2, 3]][i, j]

        Replace `i` and `j` to get element `(0, 0)`:

        >>> a[i, j].subs({i: 0, j: 0})
        0

        """
        syindex = self._check_symbolic_index(index)
        if syindex is not None:
            return syindex

        index = self._check_index_for_getitem(index)

        # `index` is a tuple with one or more slices:
        if isinstance(index, tuple) and any(isinstance(i, slice) for i in index):
            # 获取用于数组访问的切片数据
            sl_factors, eindices = self._get_slice_data_for_array_access(index)
            # 从稀疏数组中获取数据，不存在的元素使用 S.Zero
            array = [self._sparse_array.get(self._parse_index(i), S.Zero) for i in eindices]
            # 计算新的形状
            nshape = [len(el) for i, el in enumerate(sl_factors) if isinstance(index[i], slice)]
            return type(self)(array, nshape)
        else:
            # 解析索引并从稀疏数组中获取值，不存在的元素使用 S.Zero
            index = self._parse_index(index)
            return self._sparse_array.get(index, S.Zero)

    @classmethod
    def zeros(cls, *shape):
        """
        Return a sparse N-dim array of zeros.
        返回一个零值的稀疏多维数组。
        """
        return cls({}, shape)

    def tomatrix(self):
        """
        Converts MutableDenseNDimArray to Matrix. Can convert only 2-dim array, else will raise error.

        Examples
        ========

        >>> from sympy import MutableSparseNDimArray
        >>> a = MutableSparseNDimArray([1 for i in range(9)], (3, 3))
        >>> b = a.tomatrix()
        >>> b
        Matrix([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]])
        """
        from sympy.matrices import SparseMatrix
        if self.rank() != 2:
            # 如果数组的维度不是2，抛出错误
            raise ValueError('Dimensions must be of size of 2')

        # 创建一个稀疏矩阵，使用稀疏数组的数据
        mat_sparse = {}
        for key, value in self._sparse_array.items():
            mat_sparse[self._get_tuple_index(key)] = value

        return SparseMatrix(self.shape[0], self.shape[1], mat_sparse)
    # 定义 reshape 方法，用于改变对象的形状
    def reshape(self, *newshape):
        # 计算新形状下的总元素个数
        new_total_size = functools.reduce(lambda x,y: x*y, newshape)
        # 如果新形状的总元素个数与当前对象的元素总数不一致，抛出数值错误异常
        if new_total_size != self._loop_size:
            raise ValueError("Invalid reshape parameters " + newshape)

        # 返回一个新的同类对象，使用当前对象的稀疏数组数据和新的形状
        return type(self)(self._sparse_array, newshape)
class ImmutableSparseNDimArray(SparseNDimArray, ImmutableNDimArray): # type: ignore
    # 不可变稀疏多维数组的定义，继承自SparseNDimArray和ImmutableNDimArray类

    def __new__(cls, iterable=None, shape=None, **kwargs):
        # 创建新的实例方法，用于初始化对象
        shape, flat_list = cls._handle_ndarray_creation_inputs(iterable, shape, **kwargs)
        # 处理多维数组的形状和数据输入，返回处理后的形状和扁平化后的列表

        shape = Tuple(*map(_sympify, shape))
        # 将形状中的元素转换为符号表达式

        cls._check_special_bounds(flat_list, shape)
        # 检查特殊边界情况，确保数据的有效性

        loop_size = functools.reduce(lambda x,y: x*y, shape) if shape else len(flat_list)
        # 计算循环大小，即数据总数，若形状未指定则使用扁平化列表的长度

        # Sparse array:
        if isinstance(flat_list, (dict, Dict)):
            sparse_array = Dict(flat_list)
            # 如果输入的数据是字典类型，则直接使用该字典作为稀疏数组
        else:
            sparse_array = {}
            for i, el in enumerate(flatten(flat_list)):
                if el != 0:
                    sparse_array[i] = _sympify(el)
            # 否则，遍历扁平化的列表，将非零元素存储在稀疏数组中

        sparse_array = Dict(sparse_array)
        # 将稀疏数组转换为字典对象

        self = Basic.__new__(cls, sparse_array, shape, **kwargs)
        # 调用父类Basic的__new__方法创建实例

        self._shape = shape
        self._rank = len(shape)
        self._loop_size = loop_size
        self._sparse_array = sparse_array
        # 初始化对象的形状、秩、循环大小和稀疏数组属性

        return self
        # 返回创建的对象实例

    def __setitem__(self, index, value):
        raise TypeError("immutable N-dim array")
        # 设置元素赋值的方法，抛出不可变多维数组的类型错误异常

    def as_mutable(self):
        return MutableSparseNDimArray(self)
        # 返回该不可变多维数组对象对应的可变多维数组对象


class MutableSparseNDimArray(MutableNDimArray, SparseNDimArray):
    # 可变稀疏多维数组的定义，继承自MutableNDimArray和SparseNDimArray类

    def __new__(cls, iterable=None, shape=None, **kwargs):
        # 创建新的实例方法，用于初始化对象
        shape, flat_list = cls._handle_ndarray_creation_inputs(iterable, shape, **kwargs)
        # 处理多维数组的形状和数据输入，返回处理后的形状和扁平化后的列表

        self = object.__new__(cls)
        # 调用基类object的__new__方法创建实例对象

        self._shape = shape
        self._rank = len(shape)
        self._loop_size = functools.reduce(lambda x,y: x*y, shape) if shape else len(flat_list)
        # 初始化对象的形状、秩、循环大小属性，与不可变多维数组相同

        # Sparse array:
        if isinstance(flat_list, (dict, Dict)):
            self._sparse_array = dict(flat_list)
            return self
            # 如果输入的数据是字典类型，则直接使用该字典作为稀疏数组
        else:
            self._sparse_array = {}
            for i, el in enumerate(flatten(flat_list)):
                if el != 0:
                    self._sparse_array[i] = _sympify(el)
            # 否则，遍历扁平化的列表，将非零元素存储在稀疏数组中

        return self
        # 返回创建的对象实例
    # 定义 `__setitem__` 方法，允许设置 MutableDenseNDimArray 的项

    if isinstance(index, tuple) and any(isinstance(i, slice) for i in index):
        # 如果索引是元组，并且其中有任何一个是切片对象，则执行以下操作
        value, eindices, slice_offsets = self._get_slice_data_for_array_assignment(index, value)
        # 调用内部方法，获取用于数组赋值的切片数据

        for i in eindices:
            # 遍历有效索引集合
            other_i = [ind - j for ind, j in zip(i, slice_offsets) if j is not None]
            # 计算出其他索引值
            other_value = value[other_i]
            # 获取其他值
            complete_index = self._parse_index(i)
            # 解析索引为完整索引
            if other_value != 0:
                # 如果其他值不为零
                self._sparse_array[complete_index] = other_value
                # 在稀疏数组中存储该值
            elif complete_index in self._sparse_array:
                # 否则，如果完整索引存在于稀疏数组中
                self._sparse_array.pop(complete_index)
                # 从稀疏数组中移除该项

    else:
        # 否则，对单一索引执行以下操作
        index = self._parse_index(index)
        # 解析索引
        value = _sympify(value)
        # 将值转换为 SymPy 表达式
        if value == 0 and index in self._sparse_array:
            # 如果值为零且索引存在于稀疏数组中
            self._sparse_array.pop(index)
            # 从稀疏数组中移除该项
        else:
            # 否则
            self._sparse_array[index] = value
            # 在稀疏数组中设置该值
```