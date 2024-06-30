# `D:\src\scipysrc\scipy\scipy\sparse\_index.py`

```
"""Indexing mixin for sparse array/matrix classes.
"""
# 导入必要的库
import numpy as np
# 导入类型检查函数
from ._sputils import isintlike
# 导入稀疏数组的基类和判断函数
from ._base import sparray, issparse

# 整数类型的元组，包括 Python 的 int 和 numpy 的 integer
INT_TYPES = (int, np.integer)


def _broadcast_arrays(a, b):
    """
    Same as np.broadcast_arrays(a, b) but old writeability rules.

    NumPy >= 1.17.0 transitions broadcast_arrays to return
    read-only arrays. Set writeability explicitly to avoid warnings.
    Retain the old writeability rules, as our Cython code assumes
    the old behavior.
    """
    # 使用 np.broadcast_arrays 函数广播数组 a 和 b
    x, y = np.broadcast_arrays(a, b)
    # 设置数组的可写性与参数 a 和 b 保持一致，避免警告
    x.flags.writeable = a.flags.writeable
    y.flags.writeable = b.flags.writeable
    return x, y


class IndexMixin:
    """
    This class provides common dispatching and validation logic for indexing.
    """
    def _asindices(self, idx, length):
        """Convert `idx` to a valid index for an axis with a given length.

        Subclasses that need special validation can override this method.
        """
        try:
            # 将 idx 转换为 numpy 数组 x
            x = np.asarray(idx)
        except (ValueError, TypeError, MemoryError) as e:
            # 捕获可能的异常并抛出更具体的 IndexError
            raise IndexError('invalid index') from e

        # 检查 x 的维度是否为 1 或 2
        if x.ndim not in (1, 2):
            raise IndexError('Index dimension must be 1 or 2')

        # 若 x 的大小为 0，则直接返回 x
        if x.size == 0:
            return x

        # 检查索引是否超出范围
        max_indx = x.max()
        if max_indx >= length:
            raise IndexError('index (%d) out of range' % max_indx)

        min_indx = x.min()
        # 如果最小索引小于 0，则根据长度调整为非负索引
        if min_indx < 0:
            if min_indx < -length:
                raise IndexError('index (%d) out of range' % min_indx)
            # 如果 x 是原始索引数组或者不拥有数据，则复制一份以避免原地修改
            if x is idx or not x.flags.owndata:
                x = x.copy()
            x[x < 0] += length
        return x

    def _getrow(self, i):
        """Return a copy of row i of the matrix, as a (1 x n) row vector.
        """
        # 获取矩阵的形状
        M, N = self.shape
        # 将 i 转换为整数
        i = int(i)
        # 检查行索引 i 是否超出范围
        if i < -M or i >= M:
            raise IndexError('index (%d) out of range' % i)
        # 如果 i 为负数，调整为非负索引
        if i < 0:
            i += M
        # 调用 _get_intXslice 方法获取指定行的数据，并返回作为行向量
        return self._get_intXslice(i, slice(None))

    def _getcol(self, i):
        """Return a copy of column i of the matrix, as a (m x 1) column vector.
        """
        # 获取矩阵的形状
        M, N = self.shape
        # 将 i 转换为整数
        i = int(i)
        # 检查列索引 i 是否超出范围
        if i < -N or i >= N:
            raise IndexError('index (%d) out of range' % i)
        # 如果 i 为负数，调整为非负索引
        if i < 0:
            i += N
        # 调用 _get_sliceXint 方法获取指定列的数据，并返回作为列向量
        return self._get_sliceXint(slice(None), i)

    def _get_int(self, idx):
        raise NotImplementedError()

    def _get_slice(self, idx):
        raise NotImplementedError()

    def _get_array(self, idx):
        raise NotImplementedError()

    def _get_intXint(self, row, col):
        raise NotImplementedError()

    def _get_intXarray(self, row, col):
        raise NotImplementedError()

    def _get_intXslice(self, row, col):
        raise NotImplementedError()

    def _get_sliceXint(self, row, col):
        raise NotImplementedError()

    def _get_sliceXslice(self, row, col):
        raise NotImplementedError()
    # 定义一个私有方法，用于获取以切片形式访问数组的元素，抛出未实现错误
    def _get_sliceXarray(self, row, col):
        raise NotImplementedError()

    # 定义一个私有方法，用于获取以整数形式访问数组的元素，抛出未实现错误
    def _get_arrayXint(self, row, col):
        raise NotImplementedError()

    # 定义一个私有方法，用于获取以数组形式访问数组的元素，抛出未实现错误
    def _get_arrayXslice(self, row, col):
        raise NotImplementedError()

    # 定义一个私有方法，用于获取以数组形式访问列的元素，抛出未实现错误
    def _get_columnXarray(self, row, col):
        raise NotImplementedError()

    # 定义一个私有方法，用于获取以数组形式访问数组的元素，抛出未实现错误
    def _get_arrayXarray(self, row, col):
        raise NotImplementedError()

    # 定义一个私有方法，用于设置以整数形式访问数组的元素，抛出未实现错误
    def _set_int(self, idx, x):
        raise NotImplementedError()

    # 定义一个私有方法，用于设置以数组形式访问数组的元素，抛出未实现错误
    def _set_array(self, idx, x):
        raise NotImplementedError()

    # 定义一个私有方法，用于设置以整数索引访问数组的元素，抛出未实现错误
    def _set_intXint(self, row, col, x):
        raise NotImplementedError()

    # 定义一个私有方法，用于设置以数组形式访问数组的元素，抛出未实现错误
    def _set_arrayXarray(self, row, col, x):
        raise NotImplementedError()

    # 定义一个私有方法，用于设置稀疏数组的元素，如无法处理则转为稠密数组再设置
    def _set_arrayXarray_sparse(self, row, col, x):
        # 将稀疏数组 x 转换为稠密数组
        x = np.asarray(x.toarray(), dtype=self.dtype)
        # 广播 x 以匹配 row 的维度
        x, _ = _broadcast_arrays(x, row)
        # 调用自身的 _set_arrayXarray 方法，设置数组的元素
        self._set_arrayXarray(row, col, x)
def _compatible_boolean_index(idx, desired_ndim):
    """Check for boolean array or array-like. peek before asarray for array-like"""
    # 检查是否为布尔数组或类似数组
    # 使用 `hasattr` 函数检查是否具有 `ndim` 属性，指示兼容的数组并检查数据类型
    if not hasattr(idx, 'ndim'):
        # 如果没有 `ndim` 属性，尝试查看第一个元素是否为布尔值
        try:
            ix = next(iter(idx), None)
            # 遍历 `desired_ndim` 次，查看第一个元素是否为布尔值
            for _ in range(desired_ndim):
                if isinstance(ix, bool):
                    break
                ix = next(iter(ix), None)
            else:
                return None
        except TypeError:
            return None
        # 由于第一个元素是布尔值，构建数组并检查所有元素
        idx = np.asanyarray(idx)

    # 如果数组的数据类型是布尔类型，则返回该数组
    if idx.dtype.kind == 'b':
        return idx
    return None
```