# `D:\src\scipysrc\scipy\scipy\sparse\_dia.py`

```
"""Sparse DIAgonal format"""

__docformat__ = "restructuredtext en"

__all__ = ['dia_array', 'dia_matrix', 'isspmatrix_dia']

import numpy as np  # 导入 NumPy 库

from .._lib._util import copy_if_needed  # 导入局部库函数 copy_if_needed
from ._matrix import spmatrix  # 导入稀疏矩阵基类 spmatrix
from ._base import issparse, _formats, _spbase, sparray  # 导入稀疏矩阵相关基础函数和类
from ._data import _data_matrix  # 导入数据矩阵基类 _data_matrix
from ._sputils import (
    isshape, upcast_char, getdtype, get_sum_dtype, validateaxis, check_shape  # 导入稀疏矩阵工具函数
)
from ._sparsetools import dia_matvec  # 导入 DIA 格式稀疏矩阵特定的工具函数 dia_matvec


class _dia_base(_data_matrix):
    _format = 'dia'  # 设定稀疏矩阵格式为 'dia'

    def __repr__(self):
        _, fmt = _formats[self.format]  # 获取格式字符串
        sparse_cls = 'array' if isinstance(self, sparray) else 'matrix'  # 确定稀疏类别
        d = self.data.shape[0]  # 获取对角线数目
        return (
            f"<{fmt} sparse {sparse_cls} of dtype '{self.dtype}'\n"
            f"\twith {self.nnz} stored elements ({d} diagonals) and shape {self.shape}>"  # 构建对象表示形式字符串
        )

    def _data_mask(self):
        """Returns a mask of the same shape as self.data, where
        mask[i,j] is True when data[i,j] corresponds to a stored element."""
        num_rows, num_cols = self.shape  # 获取矩阵的行数和列数
        offset_inds = np.arange(self.data.shape[1])  # 生成偏移索引的数组
        row = offset_inds - self.offsets[:, None]  # 计算行索引偏移量
        mask = (row >= 0)  # 生成掩码，标记存储元素的位置
        mask &= (row < num_rows)
        mask &= (offset_inds < num_cols)
        return mask  # 返回生成的掩码

    def count_nonzero(self, axis=None):
        if axis is not None:
            raise NotImplementedError(
                "count_nonzero over an axis is not implemented for DIA format"
            )  # DIA 格式不支持沿轴的非零计数操作
        mask = self._data_mask()  # 获取数据掩码
        return np.count_nonzero(self.data[mask])  # 计算数据掩码中的非零元素数量

    count_nonzero.__doc__ = _spbase.count_nonzero.__doc__  # 设置 count_nonzero 方法的文档字符串

    def _getnnz(self, axis=None):
        if axis is not None:
            raise NotImplementedError("_getnnz over an axis is not implemented "
                                      "for DIA format")  # DIA 格式不支持沿轴的 nnz 计算操作
        M, N = self.shape  # 获取矩阵的行数和列数
        nnz = 0
        for k in self.offsets:
            if k > 0:
                nnz += min(M, N - k)
            else:
                nnz += min(M + k, N)
        return int(nnz)  # 返回计算得到的非零元素数量

    _getnnz.__doc__ = _spbase._getnnz.__doc__  # 设置 _getnnz 方法的文档字符串
    # 验证和调整轴参数，确保它在有效范围内
    validateaxis(axis)

    # 如果指定了轴并且轴数值为负数，将其调整为对应的正数值
    if axis is not None and axis < 0:
        axis += 2

    # 获取求和结果的数据类型，根据当前对象的数据类型确定
    res_dtype = get_sum_dtype(self.dtype)

    # 获取当前对象的行数和列数
    num_rows, num_cols = self.shape

    # 初始化返回值
    ret = None

    # 如果轴为0，则按列求和
    if axis == 0:
        # 获取有效数据的掩码
        mask = self._data_mask()
        # 计算按列求和的结果
        x = (self.data * mask).sum(axis=0)
        # 如果结果的长度与列数相等，则直接使用计算结果
        if x.shape[0] == num_cols:
            res = x
        # 否则创建一个新的全零数组，将计算结果复制到对应位置
        else:
            res = np.zeros(num_cols, dtype=x.dtype)
            res[:x.shape[0]] = x
        # 将结果转换为相应的容器对象，并指定数据类型
        ret = self._ascontainer(res, dtype=res_dtype)

    # 如果轴不是0，则按行求和
    else:
        # 初始化一个全零数组，用于存储行求和的结果
        row_sums = np.zeros((num_rows, 1), dtype=res_dtype)
        # 创建一个全一数组，用于计算行求和
        one = np.ones(num_cols, dtype=res_dtype)
        # 调用特定函数计算行求和，将结果存储到row_sums中
        dia_matvec(num_rows, num_cols, len(self.offsets),
                   self.data.shape[1], self.offsets, self.data, one, row_sums)

        # 将行求和结果转换为相应的容器对象
        row_sums = self._ascontainer(row_sums)

        # 如果轴参数为None，则返回行求和结果的总和
        if axis is None:
            return row_sums.sum(dtype=dtype, out=out)

        # 否则按指定的轴求和，并转换为相应的容器对象
        ret = self._ascontainer(row_sums.sum(axis=axis))

    # 如果指定了输出对象，并且其形状与计算结果不匹配，则抛出数值错误
    if out is not None and out.shape != ret.shape:
        raise ValueError("dimensions do not match")

    # 返回最终的求和结果，根据需要指定轴和数据类型
    return ret.sum(axis=(), dtype=dtype, out=out)
    
sum.__doc__ = _spbase.sum.__doc__
    # 如果参数 `other` 不是 `_dia_base` 类型的对象，则委托给 `other` 对象处理
    if not isinstance(other, _dia_base):
        return other._add_sparse(self)

    # 快速路径：处理稀疏结构完全相等的情况
    if np.array_equal(self.offsets, other.offsets):
        # 返回一个新的对象，其数据是两个对象数据相加后的结果
        return self._with_data(self.data + other.data)

    # 找到偏移量的并集，这些偏移量将会被排序并且唯一
    new_offsets = np.union1d(self.offsets, other.offsets)
    self_idx = np.searchsorted(new_offsets, self.offsets)
    other_idx = np.searchsorted(new_offsets, other.offsets)

    self_d = self.data.shape[1]
    other_d = other.data.shape[1]

    # 快速路径：处理稀疏结构的情况，其中最终的偏移量是现有偏移量的一个置换，并且对角线长度匹配
    if self_d == other_d and len(new_offsets) == len(self.offsets):
        # 从 `self.data` 中获取新数据，以 `_invert_index(self_idx)` 索引的方式
        new_data = self.data[_invert_index(self_idx)]
        # 将 `other.data` 添加到相应的位置
        new_data[other_idx, :] += other.data
    elif self_d == other_d and len(new_offsets) == len(other.offsets):
        # 从 `other.data` 中获取新数据，以 `_invert_index(other_idx)` 索引的方式
        new_data = other.data[_invert_index(other_idx)]
        # 将 `self.data` 添加到相应的位置
        new_data[self_idx, :] += self.data
    else:
        # 计算结果的最大对角线长度
        d = min(self.shape[0] + new_offsets[-1], self.shape[1])

        # 将所有对角线加到新分配的数据数组中
        new_data = np.zeros(
            (len(new_offsets), d),
            dtype=np.result_type(self.data, other.data),
        )
        # 将 `self.data` 的部分添加到新数据数组中
        new_data[self_idx, :self_d] += self.data[:, :d]
        # 将 `other.data` 的部分添加到新数据数组中
        new_data[other_idx, :other_d] += other.data[:, :d]

    # 返回一个 `_dia_container` 对象，其包含新数据和偏移量，并保持原有形状
    return self._dia_container((new_data, new_offsets), shape=self.shape)

# 使用标量 `other` 乘以当前对象的数据，并返回新对象
def _mul_scalar(self, other):
    return self._with_data(self.data * other)

# 将向量 `other` 与当前对象进行矩阵乘法，并返回结果向量
def _matmul_vector(self, other):
    x = other

    # 创建一个零向量 `y`，其长度为当前对象的行数，数据类型是两者中较大的数据类型
    y = np.zeros(self.shape[0], dtype=upcast_char(self.dtype.char,
                                                   x.dtype.char))

    # 获取当前对象数据的列数 `L`
    L = self.data.shape[1]

    # 获取当前对象的形状 `(M, N)`
    M, N = self.shape

    # 调用 `dia_matvec` 函数进行对角线矩阵-向量乘法
    dia_matvec(M, N, len(self.offsets), L, self.offsets, self.data,
               x.ravel(), y.ravel())

    # 返回乘法结果向量 `y`
    return y
    def _setdiag(self, values, k=0):
        # 获取矩阵的行数 M 和列数 N
        M, N = self.shape

        if values.ndim == 0:
            # 如果 values 是标量，进行广播处理
            values_n = np.inf
        else:
            # 否则，获取 values 的长度
            values_n = len(values)

        if k < 0:
            # 负对角线的情况下，计算有效长度 n 和索引范围
            n = min(M + k, N, values_n)
            min_index = 0
            max_index = n
        else:
            # 非负对角线的情况下，计算有效长度 n 和索引范围
            n = min(M, N - k, values_n)
            min_index = k
            max_index = k + n

        if values.ndim != 0:
            # 如果 values 不是标量，截取前 n 个值
            values = values[:n]

        # 获取当前数据矩阵的行数和列数
        data_rows, data_cols = self.data.shape

        # 如果 k 已存在于 offsets 中
        if k in self.offsets:
            # 如果 max_index 超出当前数据列数，扩展数据矩阵
            if max_index > data_cols:
                data = np.zeros((data_rows, max_index), dtype=self.data.dtype)
                data[:, :data_cols] = self.data
                self.data = data
            # 将 values 插入对应偏移的位置
            self.data[self.offsets == k, min_index:max_index] = values
        else:
            # 如果 k 不存在于 offsets 中，添加 k 到 offsets
            self.offsets = np.append(self.offsets, self.offsets.dtype.type(k))
            # 计算新的数据矩阵的行数和列数，并扩展数据矩阵
            m = max(max_index, data_cols)
            data = np.zeros((data_rows + 1, m), dtype=self.data.dtype)
            data[:-1, :data_cols] = self.data
            data[-1, min_index:max_index] = values
            self.data = data

    def todia(self, copy=False):
        # 如果 copy 为 True，则返回当前对象的副本
        if copy:
            return self.copy()
        else:
            # 否则，返回当前对象的引用
            return self

    todia.__doc__ = _spbase.todia.__doc__

    def transpose(self, axes=None, copy=False):
        if axes is not None and axes != (1, 0):
            # 如果指定了 axes 参数并且不是 (1, 0)，抛出异常
            raise ValueError("Sparse arrays/matrices do not support "
                             "an 'axes' parameter because swapping "
                             "dimensions is the only logical permutation.")

        num_rows, num_cols = self.shape
        max_dim = max(self.shape)

        # 反转对角线偏移量
        offsets = -self.offsets

        # 重新对齐数据矩阵
        r = np.arange(len(offsets), dtype=np.intc)[:, None]
        c = np.arange(num_rows, dtype=np.intc) - (offsets % max_dim)[:, None]
        pad_amount = max(0, max_dim-self.data.shape[1])
        data = np.hstack((self.data, np.zeros((self.data.shape[0], pad_amount),
                                              dtype=self.data.dtype)))
        data = data[r, c]
        # 返回转置后的对象
        return self._dia_container((data, offsets), shape=(
            num_cols, num_rows), copy=copy)

    transpose.__doc__ = _spbase.transpose.__doc__

    def diagonal(self, k=0):
        # 获取矩阵的行数和列数
        rows, cols = self.shape
        # 如果 k 超出了有效范围，返回空数组
        if k <= -rows or k >= cols:
            return np.empty(0, dtype=self.data.dtype)
        # 查找偏移量为 k 的索引
        idx, = np.nonzero(self.offsets == k)
        first_col = max(0, k)
        last_col = min(rows + k, cols)
        result_size = last_col - first_col
        # 如果没有找到偏移量为 k 的索引，返回填充过的零数组
        if idx.size == 0:
            return np.zeros(result_size, dtype=self.data.dtype)
        # 否则，返回找到的结果，并根据需要填充零
        result = self.data[idx[0], first_col:last_col]
        padding = result_size - len(result)
        if padding > 0:
            result = np.pad(result, (0, padding), mode='constant')
        return result
    # 将 diagonal 方法的文档字符串设置为 _spbase.diagonal 方法的文档字符串
    diagonal.__doc__ = _spbase.diagonal.__doc__

    # 将稀疏矩阵转换为 CSC（压缩稀疏列）格式
    def tocsc(self, copy=False):
        # 如果稀疏矩阵没有非零元素，则返回一个形状相同的空 CSC 容器
        if self.nnz == 0:
            return self._csc_container(self.shape, dtype=self.dtype)

        num_rows, num_cols = self.shape
        num_offsets, offset_len = self.data.shape
        offset_inds = np.arange(offset_len)

        # 计算行索引
        row = offset_inds - self.offsets[:,None]
        mask = (row >= 0)
        mask &= (row < num_rows)
        mask &= (offset_inds < num_cols)
        mask &= (self.data != 0)

        # 获取适合索引类型的类型
        idx_dtype = self._get_index_dtype(maxval=max(self.shape))
        # 计算 CSC 格式的 indptr
        indptr = np.zeros(num_cols + 1, dtype=idx_dtype)
        indptr[1:offset_len+1] = np.cumsum(mask.sum(axis=0)[:num_cols])
        if offset_len < num_cols:
            indptr[offset_len+1:] = indptr[offset_len]
        # 获取列索引和数据
        indices = row.T[mask.T].astype(idx_dtype, copy=False)
        data = self.data.T[mask.T]
        # 返回 CSC 格式的稀疏矩阵容器
        return self._csc_container((data, indices, indptr), shape=self.shape,
                                   dtype=self.dtype)

    # 将 tocsr 方法的文档字符串设置为 _spbase.tocsr 方法的文档字符串
    tocsc.__doc__ = _spbase.tocsc.__doc__

    # 将稀疏矩阵转换为 COO（坐标格式）格式
    def tocoo(self, copy=False):
        num_rows, num_cols = self.shape
        num_offsets, offset_len = self.data.shape
        offset_inds = np.arange(offset_len)

        # 计算行索引和列索引
        row = offset_inds - self.offsets[:,None]
        mask = (row >= 0)
        mask &= (row < num_rows)
        mask &= (offset_inds < num_cols)
        mask &= (self.data != 0)
        row = row[mask]
        col = np.tile(offset_inds, num_offsets)[mask.ravel()]
        # 获取适合索引类型的类型
        idx_dtype = self._get_index_dtype(
            arrays=(self.offsets,), maxval=max(self.shape)
        )
        row = row.astype(idx_dtype, copy=False)
        col = col.astype(idx_dtype, copy=False)
        data = self.data[mask]
        # 返回 COO 格式的稀疏矩阵容器
        return self._coo_container(
            (data, (row, col)), shape=self.shape, dtype=self.dtype, copy=False
        )

    # 将 tocoo 方法的文档字符串设置为 _spbase.tocoo 方法的文档字符串
    tocoo.__doc__ = _spbase.tocoo.__doc__

    # 由 _data_matrix 需要使用
    def _with_data(self, data, copy=True):
        """返回一个具有与 self 相同的稀疏结构但数据不同的矩阵。
        默认情况下复制结构数组。
        """
        if copy:
            # 如果 copy=True，则复制结构数组并返回新的 _dia_container
            return self._dia_container(
                (data, self.offsets.copy()), shape=self.shape
            )
        else:
            # 如果 copy=False，则直接使用现有结构数组返回新的 _dia_container
            return self._dia_container(
                (data, self.offsets), shape=self.shape
            )
    # 定义一个方法 resize，接受可变数量的形状参数
    def resize(self, *shape):
        # 调用 check_shape 函数验证并标准化形状参数
        shape = check_shape(shape)
        # 将验证后的形状参数分配给 M 和 N
        M, N = shape
        # 更新对象的数据，只保留前 N 列，不处理扩展 N 的情况
        self.data = self.data[:, :N]

        # 如果 M 大于当前对象的行数，并且存在被隐藏的值需要清除
        if (M > self.shape[0] and
                np.any(self.offsets + self.shape[0] < self.data.shape[1])):
            # 明确清除之前隐藏的值
            # 创建一个掩码来标记需要清除的值
            mask = (self.offsets[:, None] + self.shape[0] <=
                    np.arange(self.data.shape[1]))
            # 根据掩码清除数据中符合条件的值
            self.data[mask] = 0

        # 更新对象的形状属性
        self._shape = shape

    # 将 _spbase.resize 方法的文档字符串赋值给 resize 方法的文档字符串
    resize.__doc__ = _spbase.resize.__doc__
# 辅助函数，用于反转索引数组
def _invert_index(idx):
    # 创建一个与 idx 相同形状的全零数组
    inv = np.zeros_like(idx)
    # 将从 0 到 len(idx) 的整数序列按顺序赋值给 inv 的 idx 索引位置
    inv[idx] = np.arange(len(idx))
    return inv


# 检查对象 x 是否为 dia_matrix 类型的函数
def isspmatrix_dia(x):
    """
    Is `x` of dia_matrix type?

    Parameters
    ----------
    x
        object to check for being a dia matrix

    Returns
    -------
    bool
        True if `x` is a dia matrix, False otherwise

    Examples
    --------
    >>> from scipy.sparse import dia_array, dia_matrix, coo_matrix, isspmatrix_dia
    >>> isspmatrix_dia(dia_matrix([[5]]))
    True
    >>> isspmatrix_dia(dia_array([[5]]))
    False
    >>> isspmatrix_dia(coo_matrix([[5]]))
    False
    """
    return isinstance(x, dia_matrix)


# dia_array 类，继承自 _dia_base 和 sparray，用于 DIAgonal 存储的稀疏数组
class dia_array(_dia_base, sparray):
    """
    Sparse array with DIAgonal storage.

    This can be instantiated in several ways:
        dia_array(D)
            where D is a 2-D ndarray

        dia_array(S)
            with another sparse array or matrix S (equivalent to S.todia())

        dia_array((M, N), [dtype])
            to construct an empty array with shape (M, N),
            dtype is optional, defaulting to dtype='d'.

        dia_array((data, offsets), shape=(M, N))
            where the ``data[k,:]`` stores the diagonal entries for
            diagonal ``offsets[k]`` (See example below)

    Attributes
    ----------
    dtype : dtype
        Data type of the array
    shape : 2-tuple
        Shape of the array
    ndim : int
        Number of dimensions (this is always 2)
    nnz
    size
    data
        DIA format data array of the array
    offsets
        DIA format offset array of the array
    T

    Notes
    -----

    Sparse arrays can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.
    Sparse arrays with DIAgonal storage do not support slicing.

    Examples
    --------

    >>> import numpy as np
    >>> from scipy.sparse import dia_array
    >>> dia_array((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

    >>> data = np.array([[1, 2, 3, 4]]).repeat(3, axis=0)
    >>> offsets = np.array([0, -1, 2])
    >>> dia_array((data, offsets), shape=(4, 4)).toarray()
    array([[1, 0, 3, 0],
           [1, 2, 0, 4],
           [0, 2, 3, 0],
           [0, 0, 3, 4]])

    >>> from scipy.sparse import dia_array
    >>> n = 10
    >>> ex = np.ones(n)
    >>> data = np.array([ex, 2 * ex, ex])
    >>> offsets = np.array([-1, 0, 1])
    >>> dia_array((data, offsets), shape=(n, n)).toarray()
    array([[2., 1., 0., ..., 0., 0., 0.],
           [1., 2., 1., ..., 0., 0., 0.],
           [0., 1., 2., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 2., 1., 0.],
           [0., 0., 0., ..., 1., 2., 1.],
           [0., 0., 0., ..., 0., 1., 2.]])
    """


# dia_matrix 类，继承自 spmatrix 和 _dia_base，用于 DIAgonal 存储的稀疏矩阵
class dia_matrix(spmatrix, _dia_base):
    """
    Sparse matrix with DIAgonal storage.

    """
    Sparse matrix with DIAgonal storage.

    This can be instantiated in several ways:
        dia_matrix(D)
            where D is a 2-D ndarray

        dia_matrix(S)
            with another sparse array or matrix S (equivalent to S.todia())

        dia_matrix((M, N), [dtype])
            to construct an empty matrix with shape (M, N),
            dtype is optional, defaulting to dtype='d'.

        dia_matrix((data, offsets), shape=(M, N))
            where the ``data[k,:]`` stores the diagonal entries for
            diagonal ``offsets[k]`` (See example below)

    Attributes
    ----------
    dtype : dtype
        Data type of the matrix
    shape : 2-tuple
        Shape of the matrix
    ndim : int
        Number of dimensions (this is always 2)
    nnz
    size
    data
        DIA format data array of the matrix
    offsets
        DIA format offset array of the matrix
    T

    Notes
    -----

    Sparse matrices can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.
    Sparse matrices with DIAgonal storage do not support slicing.

    Examples
    --------

    >>> import numpy as np
    >>> from scipy.sparse import dia_matrix
    >>> dia_matrix((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

    >>> data = np.array([[1, 2, 3, 4]]).repeat(3, axis=0)
    >>> offsets = np.array([0, -1, 2])
    >>> dia_matrix((data, offsets), shape=(4, 4)).toarray()
    array([[1, 0, 3, 0],
           [1, 2, 0, 4],
           [0, 2, 3, 0],
           [0, 0, 3, 4]])

    >>> from scipy.sparse import dia_matrix
    >>> n = 10
    >>> ex = np.ones(n)
    >>> data = np.array([ex, 2 * ex, ex])
    >>> offsets = np.array([-1, 0, 1])
    >>> dia_matrix((data, offsets), shape=(n, n)).toarray()
    array([[2., 1., 0., ..., 0., 0., 0.],
           [1., 2., 1., ..., 0., 0., 0.],
           [0., 1., 2., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 2., 1., 0.],
           [0., 0., 0., ..., 1., 2., 1.],
           [0., 0., 0., ..., 0., 1., 2.]])
```