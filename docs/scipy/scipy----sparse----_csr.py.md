# `D:\src\scipysrc\scipy\scipy\sparse\_csr.py`

```
"""
Compressed Sparse Row matrix format
"""

__docformat__ = "restructuredtext en"

__all__ = ['csr_array', 'csr_matrix', 'isspmatrix_csr']

import numpy as np  # 导入 NumPy 库

from ._matrix import spmatrix  # 从 _matrix 模块导入 spmatrix 类
from ._base import _spbase, sparray  # 从 _base 模块导入 _spbase 类和 sparray 类
from ._sparsetools import (csr_tocsc, csr_tobsr, csr_count_blocks,  # 导入 sparse 相关工具函数
                           get_csr_submatrix, csr_sample_values)
from ._sputils import upcast  # 导入 upcast 函数

from ._compressed import _cs_matrix  # 从 _compressed 模块导入 _cs_matrix 类


class _csr_base(_cs_matrix):
    _format = 'csr'  # 设置格式为 'csr'

    def transpose(self, axes=None, copy=False):
        if axes is not None and axes != (1, 0):
            raise ValueError("Sparse arrays/matrices do not support "
                             "an 'axes' parameter because swapping "
                             "dimensions is the only logical permutation.")

        if self.ndim == 1:
            return self.copy() if copy else self  # 如果是一维数组，返回拷贝或本身
        M, N = self.shape
        # 返回 CSC 格式的容器对象，转置 CSR 矩阵到 CSC 格式
        return self._csc_container((self.data, self.indices,
                                    self.indptr), shape=(N, M), copy=copy)

    transpose.__doc__ = _spbase.transpose.__doc__  # 设置 transpose 方法的文档字符串

    def tolil(self, copy=False):
        if self.ndim != 2:
            raise ValueError("Cannot convert a 1d sparse array to lil format")
        lil = self._lil_container(self.shape, dtype=self.dtype)

        self.sum_duplicates()
        ptr, ind, dat = self.indptr, self.indices, self.data
        rows, data = lil.rows, lil.data

        for n in range(self.shape[0]):
            start = ptr[n]
            end = ptr[n+1]
            rows[n] = ind[start:end].tolist()
            data[n] = dat[start:end].tolist()

        return lil  # 返回 LIL 格式的稀疏矩阵

    tolil.__doc__ = _spbase.tolil.__doc__  # 设置 tolil 方法的文档字符串

    def tocsr(self, copy=False):
        if copy:
            return self.copy()  # 如果 copy 为真，则返回拷贝
        else:
            return self  # 否则返回自身，即原对象

    tocsr.__doc__ = _spbase.tocsr.__doc__  # 设置 tocsr 方法的文档字符串

    def tocsc(self, copy=False):
        if self.ndim != 2:
            raise ValueError("Cannot convert a 1d sparse array to csc format")
        M, N = self.shape
        idx_dtype = self._get_index_dtype((self.indptr, self.indices),
                                          maxval=max(self.nnz, M))
        indptr = np.empty(N + 1, dtype=idx_dtype)
        indices = np.empty(self.nnz, dtype=idx_dtype)
        data = np.empty(self.nnz, dtype=upcast(self.dtype))

        # 调用 C 语言实现的 csr_tocsc 函数，将 CSR 格式转换为 CSC 格式
        csr_tocsc(M, N,
                  self.indptr.astype(idx_dtype),
                  self.indices.astype(idx_dtype),
                  self.data,
                  indptr,
                  indices,
                  data)

        A = self._csc_container((data, indices, indptr), shape=self.shape)
        A.has_sorted_indices = True
        return A  # 返回 CSC 格式的稀疏矩阵对象

    tocsc.__doc__ = _spbase.tocsc.__doc__  # 设置 tocsc 方法的文档字符串
    def tobsr(self, blocksize=None, copy=True):
        # 如果数组维度不是二维，则无法转换为BSR格式，抛出数值错误异常
        if self.ndim != 2:
            raise ValueError("Cannot convert a 1d sparse array to bsr format")
        
        # 如果未指定块大小，从_spfuncs模块导入estimate_blocksize函数并返回其结果
        if blocksize is None:
            from ._spfuncs import estimate_blocksize
            return self.tobsr(blocksize=estimate_blocksize(self))
        
        # 如果块大小为(1,1)，将数据、索引和行指针封装成BSR容器对象并返回
        elif blocksize == (1,1):
            arg1 = (self.data.reshape(-1,1,1),self.indices,self.indptr)
            return self._bsr_container(arg1, shape=self.shape, copy=copy)
        
        # 对于其他块大小
        else:
            R,C = blocksize
            M,N = self.shape
            
            # 检查块大小的合法性：R和C必须大于等于1，并且M和N必须能整除R和C
            if R < 1 or C < 1 or M % R != 0 or N % C != 0:
                raise ValueError(f'invalid blocksize {blocksize}')
            
            # 计算块的数量
            blks = csr_count_blocks(M,N,R,C,self.indptr,self.indices)
            
            # 获取适当的索引数据类型
            idx_dtype = self._get_index_dtype((self.indptr, self.indices),
                                        maxval=max(N//C, blks))
            
            # 创建用于存储结果的数组和索引结构
            indptr = np.empty(M//R+1, dtype=idx_dtype)
            indices = np.empty(blks, dtype=idx_dtype)
            data = np.zeros((blks,R,C), dtype=self.dtype)
            
            # 调用csr_tobsr函数执行CSR到BSR格式的转换
            csr_tobsr(M, N, R, C,
                      self.indptr.astype(idx_dtype),
                      self.indices.astype(idx_dtype),
                      self.data,
                      indptr, indices, data.ravel())
            
            # 返回封装了结果的BSR容器对象
            return self._bsr_container(
                (data, indices, indptr), shape=self.shape
            )

    # 将tobsr方法的文档字符串设置为_spbase.tobsr的文档字符串
    tobsr.__doc__ = _spbase.tobsr.__doc__

    # 这些函数被父类(_cs_matrix)使用，用于在csc_matrix和csr_matrix之间消除冗余
    @staticmethod
    def _swap(x):
        """swap the members of x if this is a column-oriented matrix
        """
        return x

    # 实现迭代器协议，返回稀疏矩阵的迭代器对象
    def __iter__(self):
        if self.ndim == 1:
            # 如果数组是一维的，生成零值和数据的迭代器
            zero = self.dtype.type(0)
            u = 0
            for v, d in zip(self.indices, self.data):
                for _ in range(v - u):
                    yield zero
                yield d
                u = v + 1
            for _ in range(self.shape[0] - u):
                yield zero
            return
        
        # 如果数组是二维的
        indptr = np.zeros(2, dtype=self.indptr.dtype)
        # 返回1维数组(sparray)或2维行(sparray)，根据是否是sparray实例决定返回形状
        shape = self.shape[1:] if isinstance(self, sparray) else (1, self.shape[1])
        i0 = 0
        for i1 in self.indptr[1:]:
            indptr[1] = i1 - i0
            indices = self.indices[i0:i1]
            data = self.data[i0:i1]
            # 返回新创建的同类型对象，包含数据、索引和指针
            yield self.__class__((data, indices, indptr), shape=shape, copy=True)
            i0 = i1
    def _getrow(self, i):
        """Returns a copy of row i of the matrix, as a (1 x n)
        CSR matrix (row vector).
        """
        # 如果是一维数组，返回整个数组的 reshape 成 (1 x 数组长度) 的形式
        if self.ndim == 1:
            if i not in (0, -1):
                raise IndexError(f'index ({i}) out of range')
            return self.reshape((1, self.shape[0]), copy=True)

        # 获取矩阵的行数 M 和列数 N
        M, N = self.shape
        # 将 i 转换为整数
        i = int(i)
        # 处理负数索引，将其转换为正数索引
        if i < 0:
            i += M
        # 检查索引是否超出范围
        if i < 0 or i >= M:
            raise IndexError('index (%d) out of range' % i)
        # 调用外部函数获取 CSR 格式的子矩阵的数据
        indptr, indices, data = get_csr_submatrix(
            M, N, self.indptr, self.indices, self.data, i, i + 1, 0, N)
        # 返回以 CSR 矩阵格式表示的行向量的副本
        return self.__class__((data, indices, indptr), shape=(1, N),
                              dtype=self.dtype, copy=False)

    def _getcol(self, i):
        """Returns a copy of column i. A (m x 1) sparse array (column vector).
        """
        # 对于一维数组，不支持获取列向量，抛出 ValueError 异常
        if self.ndim == 1:
            raise ValueError("getcol not provided for 1d arrays. Use indexing A[j]")
        # 获取矩阵的行数 M 和列数 N
        M, N = self.shape
        # 将 i 转换为整数
        i = int(i)
        # 处理负数索引，将其转换为正数索引
        if i < 0:
            i += N
        # 检查索引是否超出范围
        if i < 0 or i >= N:
            raise IndexError('index (%d) out of range' % i)
        # 调用外部函数获取 CSR 格式的子矩阵的数据
        indptr, indices, data = get_csr_submatrix(
            M, N, self.indptr, self.indices, self.data, 0, M, i, i + 1)
        # 返回以 CSR 矩阵格式表示的列向量的副本
        return self.__class__((data, indices, indptr), shape=(M, 1),
                              dtype=self.dtype, copy=False)

    def _get_int(self, idx):
        # 找到第一个匹配索引 idx 的位置，返回对应的值；如果没有找到，则返回数据类型的零值
        spot = np.flatnonzero(self.indices == idx)
        if spot.size:
            return self.data[spot[0]]
        return self.data.dtype.type(0)

    def _get_slice(self, idx):
        # 如果 idx 是 slice(None)，返回矩阵的副本
        if idx == slice(None):
            return self.copy()
        # 如果步长为 1 或者为 None，则获取子矩阵的副本并 reshape 成一维数组
        if idx.step in (1, None):
            ret = self._get_submatrix(0, idx, copy=True)
            return ret.reshape(ret.shape[-1])
        # 否则调用 _minor_slice 处理其他情况
        return self._minor_slice(idx)

    def _get_array(self, idx):
        # 获取索引数组 idx 的数据类型
        idx_dtype = self._get_index_dtype(self.indices)
        # 将 idx 转换为指定的数据类型
        idx = np.asarray(idx, dtype=idx_dtype)
        # 如果 idx 为空数组，返回一个空的稀疏数组
        if idx.size == 0:
            return self.__class__([], dtype=self.dtype)

        # 获取矩阵的行数 M 和列数 N
        M, N = 1, self.shape[0]
        # 创建用于存储行和列的数组
        row = np.zeros_like(idx, dtype=idx_dtype)
        col = np.asarray(idx, dtype=idx_dtype)
        # 创建用于存储值的数组
        val = np.empty(row.size, dtype=self.dtype)
        # 调用外部函数获取 CSR 格式的子矩阵的数据
        csr_sample_values(M, N, self.indptr, self.indices, self.data,
                          row.size, row, col, val)

        # 根据列的形状重新调整值数组的形状，并返回稀疏数组的副本
        new_shape = col.shape if col.shape[0] > 1 else (col.shape[0],)
        return self.__class__(val.reshape(new_shape))

    def _get_intXarray(self, row, col):
        # 返回指定行的行向量，并调用 _minor_index_fancy 处理列索引数组
        return self._getrow(row)._minor_index_fancy(col)
    # 根据行索引和列索引（列为整数范围），获取稀疏矩阵的子矩阵
    def _get_intXslice(self, row, col):
        # 如果列的步长是1或者为None，则调用_get_submatrix方法返回子矩阵的副本
        if col.step in (1, None):
            return self._get_submatrix(row, col, copy=True)
        
        # TODO: 一旦这个操作变快了，取消下面这行的注释：
        # return self._getrow(row)._minor_slice(col)

        M, N = self.shape
        # 获取列索引的起始、结束和步长信息
        start, stop, stride = col.indices(N)

        # 获取行指针数组中对应行的索引范围
        ii, jj = self.indptr[row:row+2]
        row_indices = self.indices[ii:jj]  # 行索引数组
        row_data = self.data[ii:jj]         # 数据数组

        # 根据步长正负值和范围，确定有效的行索引
        if stride > 0:
            ind = (row_indices >= start) & (row_indices < stop)
        else:
            ind = (row_indices <= start) & (row_indices > stop)

        # 如果步长绝对值大于1，进一步检查行索引的间隔是否符合步长要求
        if abs(stride) > 1:
            ind &= (row_indices - start) % stride == 0

        # 根据起始、步长，计算有效的行索引和对应的数据
        row_indices = (row_indices[ind] - start) // stride
        row_data = row_data[ind]
        row_indptr = np.array([0, len(row_indices)])

        # 如果步长为负数，需要对数据和索引进行反转处理
        if stride < 0:
            row_data = row_data[::-1]
            row_indices = abs(row_indices[::-1])

        # 计算子矩阵的形状
        shape = (1, max(0, int(np.ceil(float(stop - start) / stride))))
        # 返回新的稀疏矩阵对象，包含有效的数据、索引和行指针
        return self.__class__((row_data, row_indices, row_indptr), shape=shape,
                              dtype=self.dtype, copy=False)

    # 根据行范围和列索引（列为整数），获取稀疏矩阵的子矩阵
    def _get_sliceXint(self, row, col):
        # 如果行的步长是1或者为None，则调用_get_submatrix方法返回子矩阵的副本
        if row.step in (1, None):
            return self._get_submatrix(row, col, copy=True)
        
        # 否则，调用_major_slice方法获取行的切片，再调用_get_submatrix方法获取子矩阵
        return self._major_slice(row)._get_submatrix(minor=col)

    # 根据行范围和列数组索引，获取稀疏矩阵的子矩阵
    def _get_sliceXarray(self, row, col):
        # 调用_major_slice方法获取行的切片，再调用_minor_index_fancy方法获取列的子矩阵
        return self._major_slice(row)._minor_index_fancy(col)

    # 根据行数组索引和列整数索引，获取稀疏矩阵的子矩阵
    def _get_arrayXint(self, row, col):
        # 调用_major_index_fancy方法获取行的子矩阵，再调用_get_submatrix方法获取列的子矩阵
        return self._major_index_fancy(row)._get_submatrix(minor=col)

    # 根据行数组索引和列范围索引，获取稀疏矩阵的子矩阵
    def _get_arrayXslice(self, row, col):
        # 如果列的步长不是1或者为None，则生成列的索引数组，并调用_get_arrayXarray方法获取子矩阵
        if col.step not in (1, None):
            col = np.arange(*col.indices(self.shape[1]))
            return self._get_arrayXarray(row, col)
        
        # 否则，调用_major_index_fancy方法获取行的子矩阵，再调用_get_submatrix方法获取列的子矩阵
        return self._major_index_fancy(row)._get_submatrix(minor=col)

    # 根据索引和数据，设置稀疏矩阵的元素
    def _set_int(self, idx, x):
        # 调用_set_many方法设置稀疏矩阵的多个元素
        self._set_many(0, idx, x)

    # 根据索引和广播的数据，设置稀疏矩阵的元素
    def _set_array(self, idx, x):
        # 对广播后的数据进行形状处理，调用_set_many方法设置稀疏矩阵的多个元素
        x = np.broadcast_to(x, idx.shape)
        self._set_many(np.zeros_like(idx), idx, x)
# 检查参数 `x` 是否为 CSR 稀疏矩阵类型
def isspmatrix_csr(x):
    # 使用 isinstance 判断 x 是否为 csr_matrix 类型
    return isinstance(x, csr_matrix)


# 这个命名空间类将数组与矩阵分开，通过 isinstance
class csr_array(_csr_base, sparray):
    """
    压缩稀疏行数组（Compressed Sparse Row array）。

    可以通过多种方式进行实例化：
        csr_array(D)
            其中 D 是一个二维 ndarray

        csr_array(S)
            使用另一个稀疏数组或矩阵 S 进行实例化（等同于 S.tocsr()）

        csr_array((M, N), [dtype])
            构造一个形状为 (M, N) 的空数组
            dtype 是可选的，默认为 dtype='d'。

        csr_array((data, (row_ind, col_ind)), [shape=(M, N)])
            其中 ``data``, ``row_ind`` 和 ``col_ind`` 满足关系 ``a[row_ind[k], col_ind[k]] = data[k]``。

        csr_array((data, indices, indptr), [shape=(M, N)])
            标准的 CSR 表示，其中第 i 行的列索引存储在 ``indices[indptr[i]:indptr[i+1]]`` 中，
            对应的值存储在 ``data[indptr[i]:indptr[i+1]]`` 中。
            如果未提供 shape 参数，则从索引数组中推断数组的维度。

    属性
    ----------
    dtype : dtype
        数组的数据类型
    shape : 2-tuple
        数组的形状
    ndim : int
        维度数（始终为 2）
    nnz
    size
    data
        数组的 CSR 格式数据数组
    indices
        数组的 CSR 格式索引数组
    indptr
        数组的 CSR 格式指针索引数组
    has_sorted_indices
    has_canonical_format
    T

    注
    -----

    稀疏数组可用于算术运算：支持加法、减法、乘法、除法和矩阵乘法等操作。

    CSR 格式的优势
      - 高效的算术运算：CSR + CSR，CSR * CSR 等
      - 高效的行切片操作
      - 快速的矩阵向量乘积

    CSR 格式的劣势
      - 较慢的列切片操作（考虑使用 CSC 格式）
      - 更改稀疏结构的开销较大（考虑使用 LIL 或 DOK 格式）

    规范格式
        - 在每行内，索引按列排序。
        - 不存在重复条目。

    示例
    --------

    >>> import numpy as np
    >>> from scipy.sparse import csr_array
    >>> csr_array((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

    >>> row = np.array([0, 0, 1, 2, 2, 2])
    >>> col = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> csr_array((data, (row, col)), shape=(3, 3)).toarray()
    array([[1, 0, 2],
           [0, 0, 3],
           [4, 5, 6]])

    >>> indptr = np.array([0, 2, 3, 6])
    >>> indices = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> csr_array((data, indices, indptr), shape=(3, 3)).toarray()
    array([[1, 0, 2],
           [0, 0, 3],
           [4, 5, 6]])

    Duplicate entries are summed together:

    >>> row = np.array([0, 1, 2, 0])
    >>> col = np.array([0, 1, 1, 0])
    >>> data = np.array([1, 2, 4, 8])
    >>> csr_array((data, (row, col)), shape=(3, 3)).toarray()
    array([[9, 0, 0],
           [0, 2, 0],
           [0, 4, 0]])

    As an example of how to construct a CSR array incrementally,
    the following snippet builds a term-document array from texts:

    >>> docs = [["hello", "world", "hello"], ["goodbye", "cruel", "world"]]
    >>> indptr = [0]
    >>> indices = []
    >>> data = []
    >>> vocabulary = {}
    >>> for d in docs:
    ...     for term in d:
    ...         # 如果术语(term)不在词汇表中，则将其添加，并分配一个唯一的索引
    ...         index = vocabulary.setdefault(term, len(vocabulary))
    ...         # 添加术语在词汇表中的索引到索引数组
    ...         indices.append(index)
    ...         # 添加术语的频率或计数到数据数组
    ...         data.append(1)
    ...     # 每个文档结束时，更新indptr以表示新文档开始的索引
    ...     indptr.append(len(indices))
    ...
    >>> csr_array((data, indices, indptr), dtype=int).toarray()
    array([[2, 1, 0, 0],
           [0, 1, 1, 1]])
class csr_matrix(spmatrix, _csr_base):
    """
    Compressed Sparse Row matrix.

    This can be instantiated in several ways:
        csr_matrix(D)
            where D is a 2-D ndarray

        csr_matrix(S)
            with another sparse array or matrix S (equivalent to S.tocsr())

        csr_matrix((M, N), [dtype])
            to construct an empty matrix with shape (M, N)
            dtype is optional, defaulting to dtype='d'.

        csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
            where ``data``, ``row_ind`` and ``col_ind`` satisfy the
            relationship ``a[row_ind[k], col_ind[k]] = data[k]``.

        csr_matrix((data, indices, indptr), [shape=(M, N)])
            is the standard CSR representation where the column indices for
            row i are stored in ``indices[indptr[i]:indptr[i+1]]`` and their
            corresponding values are stored in ``data[indptr[i]:indptr[i+1]]``.
            If the shape parameter is not supplied, the matrix dimensions
            are inferred from the index arrays.

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
        CSR format data array of the matrix
    indices
        CSR format index array of the matrix
    indptr
        CSR format index pointer array of the matrix
    has_sorted_indices
    has_canonical_format
    T

    Notes
    -----

    Sparse matrices can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    Advantages of the CSR format
      - efficient arithmetic operations CSR + CSR, CSR * CSR, etc.
      - efficient row slicing
      - fast matrix vector products

    Disadvantages of the CSR format
      - slow column slicing operations (consider CSC)
      - changes to the sparsity structure are expensive (consider LIL or DOK)

    Canonical Format
        - Within each row, indices are sorted by column.
        - There are no duplicate entries.

    Examples
    --------

    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>> csr_matrix((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

    >>> row = np.array([0, 0, 1, 2, 2, 2])
    >>> col = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> csr_matrix((data, (row, col)), shape=(3, 3)).toarray()
    array([[1, 0, 2],
           [0, 0, 3],
           [4, 5, 6]])

    >>> indptr = np.array([0, 2, 3, 6])
    >>> indices = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6])
    >>> csr_matrix((data, indices, indptr), shape=(3, 3)).toarray()
    array([[1, 0, 2],
           [0, 0, 3],
           [4, 5, 6]])

    Duplicate entries are summed together:

    >>> row = np.array([0, 1, 2, 0])
    >>> col = np.array([0, 1, 1, 0])
    """
    # 创建一个包含整数数据的 NumPy 数组
    data = np.array([1, 2, 4, 8])
    # 使用给定的数据数组 `data`、行索引数组 `row`、列索引数组 `col` 构建一个 CSR 矩阵，并将其转换为普通的二维数组
    csr_matrix((data, (row, col)), shape=(3, 3)).toarray()
    
    # 作为逐步构建 CSR 矩阵的示例，以下代码段从文本构建了一个术语-文档矩阵：
    # 文档列表
    docs = [["hello", "world", "hello"], ["goodbye", "cruel", "world"]]
    # 行指针数组，起始指向零
    indptr = [0]
    # 列索引数组
    indices = []
    # 数据数组
    data = []
    # 术语到索引的映射字典
    vocabulary = {}
    
    # 遍历每个文档
    for d in docs:
        # 遍历每个术语
        for term in d:
            # 如果术语不在词汇表中，则将其加入，并分配一个新的索引
            index = vocabulary.setdefault(term, len(vocabulary))
            # 将索引加入到索引数组中
            indices.append(index)
            # 将数据（默认为1）加入到数据数组中
            data.append(1)
        # 将当前索引数组的长度加入到行指针数组中，表示当前文档处理结束
        indptr.append(len(indices))
    
    # 使用数据数组 `data`、列索引数组 `indices`、行指针数组 `indptr` 构建一个 CSR 矩阵，并将其转换为普通的二维数组
    csr_matrix((data, indices, indptr), dtype=int).toarray()
```