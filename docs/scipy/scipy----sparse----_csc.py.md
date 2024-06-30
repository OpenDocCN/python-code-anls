# `D:\src\scipysrc\scipy\scipy\sparse\_csc.py`

```
"""
Compressed Sparse Column matrix format
"""
__docformat__ = "restructuredtext en"

__all__ = ['csc_array', 'csc_matrix', 'isspmatrix_csc']

import numpy as np  # 导入 NumPy 库

from ._matrix import spmatrix  # 从 _matrix 模块中导入 spmatrix 类
from ._base import _spbase, sparray  # 从 _base 模块中导入 _spbase 和 sparray 类
from ._sparsetools import csc_tocsr, expandptr  # 从 _sparsetools 模块中导入 csc_tocsr 和 expandptr 函数
from ._sputils import upcast  # 从 _sputils 模块中导入 upcast 函数
from ._compressed import _cs_matrix  # 从 _compressed 模块中导入 _cs_matrix 类

class _csc_base(_cs_matrix):
    _format = 'csc'  # 设置格式为 'csc'

    def transpose(self, axes=None, copy=False):
        if axes is not None and axes != (1, 0):
            raise ValueError("Sparse arrays/matrices do not support "
                             "an 'axes' parameter because swapping "
                             "dimensions is the only logical permutation.")

        M, N = self.shape  # 获取矩阵的形状

        # 返回转置后的 CSR 容器
        return self._csr_container((self.data, self.indices,
                                    self.indptr), (N, M), copy=copy)

    transpose.__doc__ = _spbase.transpose.__doc__  # 设置 transpose 方法的文档字符串为 _spbase.transpose 的文档字符串

    def __iter__(self):
        yield from self.tocsr()  # 迭代时返回转换为 CSR 格式的结果

    def tocsc(self, copy=False):
        if copy:
            return self.copy()  # 如果 copy 参数为 True，则返回副本
        else:
            return self  # 否则返回自身

    tocsc.__doc__ = _spbase.tocsc.__doc__  # 设置 tocsc 方法的文档字符串为 _spbase.tocsc 的文档字符串

    def tocsr(self, copy=False):
        M,N = self.shape  # 获取矩阵的形状
        idx_dtype = self._get_index_dtype((self.indptr, self.indices),
                                    maxval=max(self.nnz, N))  # 获取适合索引的数据类型
        indptr = np.empty(M + 1, dtype=idx_dtype)  # 创建空的 indptr 数组
        indices = np.empty(self.nnz, dtype=idx_dtype)  # 创建空的 indices 数组
        data = np.empty(self.nnz, dtype=upcast(self.dtype))  # 创建空的 data 数组

        csc_tocsr(M, N,
                  self.indptr.astype(idx_dtype),
                  self.indices.astype(idx_dtype),
                  self.data,
                  indptr,
                  indices,
                  data)  # 调用 csc_tocsr 函数进行格式转换

        A = self._csr_container(
            (data, indices, indptr),
            shape=self.shape, copy=False
        )  # 创建 CSR 格式的容器 A
        A.has_sorted_indices = True  # 设置 A 的 sorted indices 为 True
        return A  # 返回转换后的 CSR 格式的对象

    tocsr.__doc__ = _spbase.tocsr.__doc__  # 设置 tocsr 方法的文档字符串为 _spbase.tocsr 的文档字符串

    def nonzero(self):
        # CSC 不能使用 _cs_matrix 的 .nonzero 方法，因为它返回转置后的索引排序。

        # 从 _cs_matrix.tocoo 获取行和列索引
        major_dim, minor_dim = self._swap(self.shape)  # 获取主要维度和次要维度
        minor_indices = self.indices  # 获取次要索引
        major_indices = np.empty(len(minor_indices), dtype=self.indices.dtype)  # 创建空的主要索引数组
        expandptr(major_dim, self.indptr, major_indices)  # 扩展主要索引

        row, col = self._swap((major_indices, minor_indices))  # 交换行和列索引

        # 去除显式的零值
        nz_mask = self.data != 0  # 创建非零值掩码
        row = row[nz_mask]  # 应用非零值掩码到行索引
        col = col[nz_mask]  # 应用非零值掩码到列索引

        # 按照 C 风格顺序排序
        ind = np.argsort(row, kind='mergesort')  # 对行索引进行排序
        row = row[ind]  # 根据排序后的索引重新排列行索引
        col = col[ind]  # 根据排序后的索引重新排列列索引

        return row, col  # 返回行索引和列索引

    nonzero.__doc__ = _cs_matrix.nonzero.__doc__  # 设置 nonzero 方法的文档字符串为 _cs_matrix.nonzero 的文档字符串
    def _getrow(self, i):
        """Returns a copy of row i of the matrix, as a (1 x n)
        CSR matrix (row vector).
        """
        # 获取矩阵的行数 M 和列数 N
        M, N = self.shape
        # 将 i 转换为整数类型
        i = int(i)
        # 处理负数索引，转换为正数索引
        if i < 0:
            i += M
        # 检查索引是否超出范围
        if i < 0 or i >= M:
            raise IndexError('index (%d) out of range' % i)
        # 调用 _get_submatrix 方法获取指定行的 CSR 格式的子矩阵并返回
        return self._get_submatrix(minor=i).tocsr()

    def _getcol(self, i):
        """Returns a copy of column i of the matrix, as a (m x 1)
        CSC matrix (column vector).
        """
        # 获取矩阵的行数 M 和列数 N
        M, N = self.shape
        # 将 i 转换为整数类型
        i = int(i)
        # 处理负数索引，转换为正数索引
        if i < 0:
            i += N
        # 检查索引是否超出范围
        if i < 0 or i >= N:
            raise IndexError('index (%d) out of range' % i)
        # 调用 _get_submatrix 方法获取指定列的 CSC 格式的子矩阵并返回
        return self._get_submatrix(major=i, copy=True)

    def _get_intXarray(self, row, col):
        # 调用 _major_index_fancy 方法获取指定列的子矩阵，再调用 _get_submatrix 获取指定行的子矩阵并返回
        return self._major_index_fancy(col)._get_submatrix(minor=row)

    def _get_intXslice(self, row, col):
        # 检查列的步长是否为 1 或者 None，如果是则直接获取子矩阵，否则先对列进行切片再获取子矩阵并返回
        if col.step in (1, None):
            return self._get_submatrix(major=col, minor=row, copy=True)
        return self._major_slice(col)._get_submatrix(minor=row)

    def _get_sliceXint(self, row, col):
        # 检查行的步长是否为 1 或者 None，如果是则直接获取子矩阵，否则先对行进行切片再获取子矩阵并返回
        if row.step in (1, None):
            return self._get_submatrix(major=col, minor=row, copy=True)
        return self._get_submatrix(major=col)._minor_slice(row)

    def _get_sliceXarray(self, row, col):
        # 调用 _major_index_fancy 方法获取指定列的子矩阵，再调用 _minor_slice 获取指定行的子矩阵并返回
        return self._major_index_fancy(col)._minor_slice(row)

    def _get_arrayXint(self, row, col):
        # 调用 _get_submatrix 获取指定列的子矩阵，再调用 _minor_index_fancy 获取指定行的子矩阵并返回
        return self._get_submatrix(major=col)._minor_index_fancy(row)

    def _get_arrayXslice(self, row, col):
        # 调用 _major_slice 方法获取指定列的子矩阵，再调用 _minor_index_fancy 获取指定行的子矩阵并返回
        return self._major_slice(col)._minor_index_fancy(row)

    # these functions are used by the parent class (_cs_matrix)
    # to remove redundancy between csc_array and csr_matrix
    @staticmethod
    def _swap(x):
        """swap the members of x if this is a column-oriented matrix
        """
        # 如果这是一个列向量矩阵，则交换 x 的成员顺序并返回
        return x[1], x[0]
# 判断对象 `x` 是否为 csc_matrix 类型的稀疏矩阵
def isspmatrix_csc(x):
    """Is `x` of csc_matrix type?

    Parameters
    ----------
    x
        要检查是否为 csc 矩阵类型的对象

    Returns
    -------
    bool
        如果 `x` 是 csc 矩阵则返回 True，否则返回 False

    Examples
    --------
    >>> from scipy.sparse import csc_array, csc_matrix, coo_matrix, isspmatrix_csc
    >>> isspmatrix_csc(csc_matrix([[5]]))
    True
    >>> isspmatrix_csc(csc_array([[5]]))
    False
    >>> isspmatrix_csc(coo_matrix([[5]]))
    False
    """
    return isinstance(x, csc_matrix)


# 这个命名空间类用于将数组与矩阵分离，通过 isinstance 判断
class csc_array(_csc_base, sparray):
    """
    压缩稀疏列数组 (Compressed Sparse Column array)。

    可以通过以下几种方式进行实例化：
        csc_array(D)
            其中 D 是一个二维 ndarray

        csc_array(S)
            使用另一个稀疏数组或矩阵 S 进行实例化（等同于 S.tocsc()）

        csc_array((M, N), [dtype])
            创建一个指定形状 (M, N) 的空数组
            dtype 是可选的，默认为 dtype='d'。

        csc_array((data, (row_ind, col_ind)), [shape=(M, N)])
            其中 ``data``、``row_ind`` 和 ``col_ind`` 满足关系 ``a[row_ind[k], col_ind[k]] = data[k]``。

        csc_array((data, indices, indptr), [shape=(M, N)])
            标准的 CSC 表示，其中第 i 列的行索引存储在 ``indices[indptr[i]:indptr[i+1]]`` 中，
            对应的值存储在 ``data[indptr[i]:indptr[i+1]]`` 中。如果未提供 shape 参数，数组维度将从索引数组推断出来。

    属性
    ----------
    dtype : dtype
        数组的数据类型
    shape : 2-tuple
        数组的形状
    ndim : int
        数组的维数（始终为 2）
    nnz
    size
    data
        数组的 CSC 格式数据数组
    indices
        数组的 CSC 格式索引数组
    indptr
        数组的 CSC 格式索引指针数组
    has_sorted_indices
    has_canonical_format
    T

    注意
    -----

    稀疏数组可以用于算术运算：它们支持加法、减法、乘法、除法和矩阵乘方运算。

    CSC 格式的优点
        - 高效的算术运算（CSC + CSC，CSC * CSC 等）
        - 高效的列切片操作
        - 快速的矩阵向量乘法（CSR、BSR 可能更快）

    CSC 格式的缺点
      - 较慢的行切片操作（考虑使用 CSR）
      - 修改稀疏结构昂贵（考虑使用 LIL 或 DOK）

    规范格式
      - 每列内的索引按行排序。
      - 没有重复条目。

    示例
    --------

    >>> import numpy as np
    >>> from scipy.sparse import csc_array
    >>> csc_array((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)
    # 创建稀疏矩阵的压缩列格式 (CSC) 的数组，从给定的数据、行索引和列索引数组构建
    row = np.array([0, 2, 2, 0, 1, 2])
    # 列索引数组，指示每个数据元素所在的列位置
    col = np.array([0, 0, 1, 2, 2, 2])
    # 数据数组，包含每个非零元素的值
    data = np.array([1, 2, 3, 4, 5, 6])
    # 调用 csc_array 函数创建压缩列格式的稀疏矩阵，并将其转换为密集数组（数组形式）
    csc_array((data, (row, col)), shape=(3, 3)).toarray()
    
    # 创建稀疏矩阵的压缩列格式 (CSC) 的数组，从给定的数据、列索引、指针数组构建
    indptr = np.array([0, 2, 3, 6])
    # 列索引数组，指示每个数据元素所在的列位置
    indices = np.array([0, 2, 2, 0, 1, 2])
    # 数据数组，包含每个非零元素的值
    data = np.array([1, 2, 3, 4, 5, 6])
    # 调用 csc_array 函数创建压缩列格式的稀疏矩阵，并将其转换为密集数组（数组形式）
    csc_array((data, indices, indptr), shape=(3, 3)).toarray()
# 定义一个压缩稀疏列（CSC）矩阵的类，继承自 spmatrix 和 _csc_base
class csc_matrix(spmatrix, _csc_base):
    """
    压缩稀疏列矩阵。

    可以通过多种方式进行实例化：
        csc_matrix(D)
            其中 D 是一个二维 ndarray

        csc_matrix(S)
            使用另一个稀疏数组或矩阵 S 实例化（等同于 S.tocsc()）

        csc_matrix((M, N), [dtype])
            构造一个形状为 (M, N) 的空矩阵，dtype 是可选的，默认为 dtype='d'。

        csc_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
            其中 ``data``、``row_ind`` 和 ``col_ind`` 满足关系 ``a[row_ind[k], col_ind[k]] = data[k]``。

        csc_matrix((data, indices, indptr), [shape=(M, N)])
            标准的 CSC 表示，其中列 i 的行索引存储在 ``indices[indptr[i]:indptr[i+1]]``，
            对应的值存储在 ``data[indptr[i]:indptr[i+1]]``。如果未提供 shape 参数，则从索引数组中推断矩阵维度。

    属性
    ----------
    dtype : dtype
        矩阵的数据类型
    shape : 2 元组
        矩阵的形状
    ndim : int
        维度数（始终为 2）
    nnz
    size
    data
        矩阵的 CSC 格式数据数组
    indices
        矩阵的 CSC 格式索引数组
    indptr
        矩阵的 CSC 格式索引指针数组
    has_sorted_indices
    has_canonical_format
    T

    注记
    -----

    稀疏矩阵可用于算术运算：支持加法、减法、乘法、除法和矩阵幂运算。

    CSC 格式的优势
        - 高效的算术运算：CSC + CSC，CSC * CSC 等
        - 高效的列切片操作
        - 快速的矩阵向量乘法（CSR、BSR 可能更快）

    CSC 格式的劣势
      - 缓慢的行切片操作（考虑使用 CSR）
      - 更改稀疏结构昂贵（考虑使用 LIL 或 DOK）

    规范格式
      - 在每列内部，索引按行排序。
      - 不存在重复条目。

    示例
    --------

    >>> import numpy as np
    >>> from scipy.sparse import csc_matrix
    >>> csc_matrix((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

    >>> row = np.array([0, 2, 2, 0, 1, 2])
    >>> col = np.array([0, 0, 1, 2, 2, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> csc_matrix((data, (row, col)), shape=(3, 3)).toarray()
    array([[1, 0, 4],
           [0, 0, 5],
           [2, 3, 6]])

    >>> indptr = np.array([0, 2, 3, 6])
    >>> indices = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> csc_matrix((data, indices, indptr), shape=(3, 3)).toarray()
    array([[1, 0, 4],
           [0, 0, 5],
           [2, 3, 6]])

    """
```