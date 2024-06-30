# `D:\src\scipysrc\scipy\scipy\sparse\_bsr.py`

```
"""Compressed Block Sparse Row format"""

__docformat__ = "restructuredtext en"

__all__ = ['bsr_array', 'bsr_matrix', 'isspmatrix_bsr']

# 导入警告模块，用于在需要时发出警告
from warnings import warn

# 导入 NumPy 库，用于处理数组和数学运算
import numpy as np

# 导入工具函数，处理复制操作
from scipy._lib._util import copy_if_needed

# 导入稀疏矩阵的基类和数据处理类
from ._matrix import spmatrix
from ._data import _data_matrix, _minmax_mixin

# 导入压缩格式矩阵的基类
from ._compressed import _cs_matrix

# 导入稀疏矩阵的基础模块，包括类型判断和基础操作
from ._base import issparse, _formats, _spbase, sparray

# 导入稀疏矩阵的实用工具函数，如数据获取、类型转换和类型检查等
from ._sputils import (isshape, getdtype, getdata, to_native, upcast,
                       check_shape)

# 导入稀疏矩阵的 C 扩展工具函数，用于具体的矩阵运算和操作
from . import _sparsetools

# 导入特定于 BSR 格式的 C 扩展工具函数，用于矩阵向量乘法、矩阵乘法等操作
from ._sparsetools import (bsr_matvec, bsr_matvecs, csr_matmat_maxnnz,
                           bsr_matmat, bsr_transpose, bsr_sort_indices,
                           bsr_tocsr)

# 定义 BSR 格式矩阵的基类，继承自 _cs_matrix 和 _minmax_mixin
class _bsr_base(_cs_matrix, _minmax_mixin):
    # 指定格式为 BSR
    _format = 'bsr'


这段代码是 Python 中用于处理压缩块稀疏行（Block Sparse Row，BSR）格式的稀疏矩阵相关的定义和导入模块。
    def check_format(self, full_check=True):
        """Check whether the array/matrix respects the BSR format.

        Parameters
        ----------
        full_check : bool, optional
            If `True`, run rigorous check, scanning arrays for valid values.
            Note that activating those check might copy arrays for casting,
            modifying indices and index pointers' inplace.
            If `False`, run basic checks on attributes. O(1) operations.
            Default is `True`.
        """
        # 获取数组的形状
        M,N = self.shape
        # 获取块的大小
        R,C = self.blocksize

        # 检查索引指针数组的数据类型是否为整数
        if self.indptr.dtype.kind != 'i':
            warn(f"indptr array has non-integer dtype ({self.indptr.dtype.name})",
                 stacklevel=2)
        # 检查索引数组的数据类型是否为整数
        if self.indices.dtype.kind != 'i':
            warn(f"indices array has non-integer dtype ({self.indices.dtype.name})",
                 stacklevel=2)

        # 检查数组的形状是否正确
        if self.indices.ndim != 1 or self.indptr.ndim != 1:
            raise ValueError("indices, and indptr should be 1-D")
        # 检查数据数组是否为三维
        if self.data.ndim != 3:
            raise ValueError("data should be 3-D")

        # 检查索引指针的值
        if len(self.indptr) != M//R + 1:
            raise ValueError("index pointer size (%d) should be (%d)" %
                                (len(self.indptr), M//R + 1))
        if self.indptr[0] != 0:
            raise ValueError("index pointer should start with 0")

        # 检查索引和数据数组的大小
        if len(self.indices) != len(self.data):
            raise ValueError("indices and data should have the same size")
        # 检查索引指针的最后一个值
        if self.indptr[-1] > len(self.indices):
            raise ValueError("Last value of index pointer should be less than "
                                "the size of index and data arrays")

        # 调用 prune 方法
        self.prune()

        # 如果 full_check 为 True，进行更严格的格式检查
        if full_check:
            # 检查格式的有效性（开销较大）
            if self.nnz > 0:
                if self.indices.max() >= N//C:
                    raise ValueError("column index values must be < %d (now max %d)"
                                     % (N//C, self.indices.max()))
                if self.indices.min() < 0:
                    raise ValueError("column index values must be >= 0")
                if np.diff(self.indptr).min() < 0:
                    raise ValueError("index pointer values must form a "
                                        "non-decreasing sequence")

            # 获取适合索引数组的数据类型
            idx_dtype = self._get_index_dtype((self.indices, self.indptr))
            # 将索引指针数组转换为指定数据类型
            self.indptr = np.asarray(self.indptr, dtype=idx_dtype)
            # 将索引数组转换为指定数据类型
            self.indices = np.asarray(self.indices, dtype=idx_dtype)
            # 将数据数组转换为本地数据类型
            self.data = to_native(self.data)

        # 如果未按排序顺序排列索引，则进行排序
        # if not self.has_sorted_indices():
        #    warn('Indices were not in sorted order. Sorting indices.')
        #    self.sort_indices(check_first=False)

    @property
    # 返回矩阵的块大小，即矩阵的列数和行数，作为元组返回
    def blocksize(self) -> tuple:
        """Block size of the matrix."""
        return self.data.shape[1:]

    # 获取非零元素个数，如果指定了axis，则抛出NotImplementedError
    def _getnnz(self, axis=None):
        if axis is not None:
            raise NotImplementedError("_getnnz over an axis is not implemented "
                                      "for BSR format")
        # 获取块的行数和列数
        R,C = self.blocksize
        # 计算非零元素的总数
        return int(self.indptr[-1] * R * C)

    # 将_spbase._getnnz的文档字符串赋值给_getnnz方法的文档字符串
    _getnnz.__doc__ = _spbase._getnnz.__doc__

    # 计算矩阵中非零元素的数量，如果指定了axis，则抛出NotImplementedError
    def count_nonzero(self, axis=None):
        if axis is not None:
            raise NotImplementedError(
                "count_nonzero over axis is not implemented for BSR format."
            )
        # 返回经过去重处理后的数据中的非零元素数量
        return np.count_nonzero(self._deduped_data())

    # 将_spbase.count_nonzero的文档字符串赋值给count_nonzero方法的文档字符串
    count_nonzero.__doc__ = _spbase.count_nonzero.__doc__

    # 返回对象的字符串表示形式，包括格式、稀疏类型、数据类型、非零元素数量和形状等信息
    def __repr__(self):
        _, fmt = _formats[self.format]
        sparse_cls = 'array' if isinstance(self, sparray) else 'matrix'
        b = 'x'.join(str(x) for x in self.blocksize)
        return (
            f"<{fmt} sparse {sparse_cls} of dtype '{self.dtype}'\n"
            f"\twith {self.nnz} stored elements (blocksize={b}) and shape {self.shape}>"
        )

    # 返回矩阵的对角线元素数组，可以通过指定k来获取主对角线以及偏移对角线的元素
    def diagonal(self, k=0):
        rows, cols = self.shape
        if k <= -rows or k >= cols:
            return np.empty(0, dtype=self.data.dtype)
        R, C = self.blocksize
        # 创建一个空数组y用来存放对角线元素
        y = np.zeros(min(rows + min(k, 0), cols - max(k, 0)),
                     dtype=upcast(self.dtype))
        # 调用_csr.bsr_diagonal函数填充对角线元素到y中
        _sparsetools.bsr_diagonal(k, rows // R, cols // C, R, C,
                                  self.indptr, self.indices,
                                  np.ravel(self.data), y)
        return y

    # 将_spbase.diagonal的文档字符串赋值给diagonal方法的文档字符串
    diagonal.__doc__ = _spbase.diagonal.__doc__

    ##########################
    # NotImplemented methods #
    ##########################

    # 抛出NotImplementedError，表示该方法未实现
    def __getitem__(self,key):
        raise NotImplementedError

    # 抛出NotImplementedError，表示该方法未实现
    def __setitem__(self,key,val):
        raise NotImplementedError

    ######################
    # Arithmetic methods #
    ######################

    # 将矩阵转换为COO格式后执行_dense的加法操作
    def _add_dense(self, other):
        return self.tocoo(copy=False)._add_dense(other)

    # 将矩阵与向量相乘，返回结果向量
    def _matmul_vector(self, other):
        M,N = self.shape
        R,C = self.blocksize

        result = np.zeros(self.shape[0], dtype=upcast(self.dtype, other.dtype))

        # 调用_sparsetools.bsr_matvec函数执行矩阵与向量的乘法操作
        bsr_matvec(M//R, N//C, R, C,
            self.indptr, self.indices, self.data.ravel(),
            other, result)

        return result

    # 将矩阵与多个向量相乘，返回结果矩阵
    def _matmul_multivector(self,other):
        R,C = self.blocksize
        M,N = self.shape
        n_vecs = other.shape[1]  # number of column vectors

        result = np.zeros((M,n_vecs), dtype=upcast(self.dtype,other.dtype))

        # 调用_sparsetools.bsr_matvecs函数执行矩阵与多个向量的乘法操作
        bsr_matvecs(M//R, N//C, n_vecs, R, C,
                self.indptr, self.indices, self.data.ravel(),
                other.ravel(), result.ravel())

        return result
    def _matmul_sparse(self, other):
        # 获取自身稀疏矩阵的形状
        M, K1 = self.shape
        # 获取另一个稀疏矩阵的形状
        K2, N = other.shape

        # 获取自身稀疏矩阵的块大小
        R, n = self.blocksize

        # 转换到指定格式
        if other.format == "bsr":
            # 如果另一个矩阵已经是 BSR 格式，获取其块大小的第二个维度
            C = other.blocksize[1]
        else:
            # 否则，默认块大小为 1
            C = 1

        # 如果另一个矩阵是 CSR 格式且当前块大小为 1，则轻量级转换为 BSR 格式
        if other.format == "csr" and n == 1:
            other = other.tobsr(blocksize=(n,C), copy=False)  # 转换格式（轻量级）
        else:
            other = other.tobsr(blocksize=(n,C))  # 转换格式

        # 确定索引数据类型
        idx_dtype = self._get_index_dtype((self.indptr, self.indices,
                                           other.indptr, other.indices))

        # 计算稀疏矩阵乘积后的非零元素数量的上限
        bnnz = csr_matmat_maxnnz(M//R, N//C,
                                 self.indptr.astype(idx_dtype),
                                 self.indices.astype(idx_dtype),
                                 other.indptr.astype(idx_dtype),
                                 other.indices.astype(idx_dtype))

        # 重新确定索引数据类型，考虑到乘积后的非零元素数量上限
        idx_dtype = self._get_index_dtype((self.indptr, self.indices,
                                           other.indptr, other.indices),
                                          maxval=bnnz)

        # 创建 BSR 格式所需的索引和数据数组
        indptr = np.empty(self.indptr.shape, dtype=idx_dtype)
        indices = np.empty(bnnz, dtype=idx_dtype)
        data = np.empty(R*C*bnnz, dtype=upcast(self.dtype, other.dtype))

        # 执行 BSR 格式的矩阵乘法
        bsr_matmat(bnnz, M//R, N//C, R, C, n,
                   self.indptr.astype(idx_dtype),
                   self.indices.astype(idx_dtype),
                   np.ravel(self.data),
                   other.indptr.astype(idx_dtype),
                   other.indices.astype(idx_dtype),
                   np.ravel(other.data),
                   indptr,
                   indices,
                   data)

        # 将数据重新形状为合适的维度
        data = data.reshape(-1, R, C)

        # TODO: 消除零元素

        # 返回 BSR 格式的稀疏矩阵
        return self._bsr_container(
            (data, indices, indptr), shape=(M, N), blocksize=(R, C)
        )

    ######################
    # Conversion methods #
    ######################

    def tobsr(self, blocksize=None, copy=False):
        """Convert this array/matrix into Block Sparse Row Format.

        With copy=False, the data/indices may be shared between this
        array/matrix and the resultant bsr_array/bsr_matrix.

        If blocksize=(R, C) is provided, it will be used for determining
        block size of the bsr_array/bsr_matrix.
        """
        # 如果指定了新的块大小，则先转换为 CSR 格式，再转换为 BSR 格式
        if blocksize not in [None, self.blocksize]:
            return self.tocsr().tobsr(blocksize=blocksize)
        # 如果 copy=True，则返回当前对象的副本
        if copy:
            return self.copy()
        else:
            return self
    def tocoo(self, copy=True):
        """Convert this array/matrix to COOrdinate format.

        When copy=False the data array will be shared between
        this array/matrix and the resultant coo_array/coo_matrix.
        """

        # 获取当前数组的形状信息
        M,N = self.shape
        # 获取块的大小信息
        R,C = self.blocksize

        # 计算每个行的非零元素个数的差分
        indptr_diff = np.diff(self.indptr)
        # 如果差分数据类型的字节大小大于np.intp的字节大小，检查是否会发生溢出
        if indptr_diff.dtype.itemsize > np.dtype(np.intp).itemsize:
            # 将差分转换为np.intp类型，并检查是否有不匹配的情况
            indptr_diff_limited = indptr_diff.astype(np.intp)
            if np.any(indptr_diff_limited != indptr_diff):
                raise ValueError("Matrix too big to convert")
            indptr_diff = indptr_diff_limited

        # 获取适合索引的数据类型
        idx_dtype = self._get_index_dtype(maxval=max(M, N))

        # 根据行的块大小重复生成行索引
        row = (R * np.arange(M//R, dtype=idx_dtype)).repeat(indptr_diff)
        row = row.repeat(R*C).reshape(-1,R,C)
        row += np.tile(np.arange(R, dtype=idx_dtype).reshape(-1,1), (1,C))
        row = row.reshape(-1)

        # 根据列的块大小重复生成列索引
        col = ((C * self.indices).astype(idx_dtype, copy=False)
               .repeat(R*C).reshape(-1,R,C))
        col += np.tile(np.arange(C, dtype=idx_dtype), (R,1))
        col = col.reshape(-1)

        # 获取数据，并按需复制
        data = self.data.reshape(-1)
        if copy:
            data = data.copy()

        # 返回COO格式的数组或矩阵
        return self._coo_container(
            (data, (row, col)), shape=self.shape
        )
    def transpose(self, axes=None, copy=False):
        # 如果指定了 axes 参数且不是 (1, 0)，则抛出 ValueError 异常
        if axes is not None and axes != (1, 0):
            raise ValueError("Sparse matrices do not support "
                              "an 'axes' parameter because swapping "
                              "dimensions is the only logical permutation.")

        # 从 self.blocksize 中获取 R 和 C
        R, C = self.blocksize
        # 获取矩阵的形状 M 和 N
        M, N = self.shape
        # 计算每个块的数量 NBLK
        NBLK = self.nnz//(R*C)

        # 如果 nnz（非零元素数量）为 0，则返回一个形状为 (N, M) 的 BSR（块压缩行）容器
        if self.nnz == 0:
            return self._bsr_container((N, M), blocksize=(C, R),
                                       dtype=self.dtype, copy=copy)

        # 初始化用于存储 BSR 格式的转置结果的数据结构
        indptr = np.empty(N//C + 1, dtype=self.indptr.dtype)
        indices = np.empty(NBLK, dtype=self.indices.dtype)
        data = np.empty((NBLK, C, R), dtype=self.data.dtype)

        # 调用底层函数 bsr_transpose 完成 BSR 格式的转置操作
        bsr_transpose(M//R, N//C, R, C,
                      self.indptr, self.indices, self.data.ravel(),
                      indptr, indices, data.ravel())

        # 返回转置后的 BSR 容器，形状为 (N, M)
        return self._bsr_container((data, indices, indptr),
                                   shape=(N, M), copy=copy)

    # 设置 transpose 方法的文档字符串为 _spbase.transpose 的文档字符串
    transpose.__doc__ = _spbase.transpose.__doc__

    ##############################################################
    # methods that examine or modify the internal data structure #
    ##############################################################

    def eliminate_zeros(self):
        """Remove zero elements in-place."""

        # 如果 nnz（非零元素数量）为 0，则直接返回，没有需要处理的内容
        if not self.nnz:
            return  # nothing to do

        # 获取块的大小 R 和 C
        R, C = self.blocksize
        # 获取矩阵的形状 M 和 N
        M, N = self.shape

        # 根据非零块的标记删除零元素，同时调整 self.indptr 和 self.indices
        mask = (self.data != 0).reshape(-1,R*C).sum(axis=1)  # nonzero blocks
        nonzero_blocks = mask.nonzero()[0]
        self.data[:len(nonzero_blocks)] = self.data[nonzero_blocks]
        _sparsetools.csr_eliminate_zeros(M//R, N//C, self.indptr,
                                         self.indices, mask)
        self.prune()

    def sum_duplicates(self):
        """Eliminate duplicate array/matrix entries by adding them together

        The is an *in place* operation
        """

        # 如果已经处于 canonical 格式，则直接返回，不进行重复项合并
        if self.has_canonical_format:
            return

        # 对 indices 进行排序，确保相同索引的元素相邻
        self.sort_indices()
        R, C = self.blocksize
        M, N = self.shape

        # 合并重复的条目，以得到 canonical 格式的 CSR 矩阵
        n_row = M // R
        nnz = 0
        row_end = 0
        for i in range(n_row):
            jj = row_end
            row_end = self.indptr[i+1]
            while jj < row_end:
                j = self.indices[jj]
                x = self.data[jj]
                jj += 1
                while jj < row_end and self.indices[jj] == j:
                    x += self.data[jj]
                    jj += 1
                self.indices[nnz] = j
                self.data[nnz] = x
                nnz += 1
            self.indptr[i+1] = nnz

        # 调用 prune 方法，修剪零元素并更新 nnz
        self.prune()  # nnz may have changed
        # 设置标志表示已处于 canonical 格式
        self.has_canonical_format = True
    def sort_indices(self):
        """
        Sort the indices of this array/matrix *in place*
        """
        # 如果已经排序过，直接返回
        if self.has_sorted_indices:
            return

        # 从 self.blocksize 中获取块大小 R, C
        R, C = self.blocksize
        # 从 self.shape 中获取数组/矩阵的形状 M, N
        M, N = self.shape

        # 调用 C 库函数 bsr_sort_indices 对索引进行排序
        bsr_sort_indices(M // R, N // C, R, C, self.indptr, self.indices, self.data.ravel())

        # 设置排序完成的标志
        self.has_sorted_indices = True

    def prune(self):
        """
        Remove empty space after all non-zero elements.
        """
        # 从 self.blocksize 中获取块大小 R, C
        R, C = self.blocksize
        # 从 self.shape 中获取数组/矩阵的形状 M, N
        M, N = self.shape

        # 检查 self.indptr 的长度是否合法
        if len(self.indptr) != M // R + 1:
            raise ValueError("index pointer has invalid length")

        # 获取最后一个索引指针的值，即非零元素的数量
        bnnz = self.indptr[-1]

        # 检查 self.indices 数组是否包含足够的元素
        if len(self.indices) < bnnz:
            raise ValueError("indices array has too few elements")
        # 检查 self.data 数组是否包含足够的元素
        if len(self.data) < bnnz:
            raise ValueError("data array has too few elements")

        # 截断 self.data 和 self.indices 数组，移除多余的空间
        self.data = self.data[:bnnz]
        self.indices = self.indices[:bnnz]

    # utility functions
    def _binopt(self, other, op, in_shape=None, out_shape=None):
        """
        Apply the binary operation fn to two sparse matrices.
        """
        # 将 other 转换为当前类的实例，使用相同的块大小
        other = self.__class__(other, blocksize=self.blocksize)

        # 根据操作符 op 动态获取对应的函数 fn，例如 bsr_plus_bsr
        fn = getattr(_sparsetools, self.format + op + self.format)

        # 从 self.blocksize 中获取块大小 R, C
        R, C = self.blocksize

        # 计算合并后的最大非零元素数量
        max_bnnz = len(self.data) + len(other.data)
        # 根据索引类型确定适当的数据类型
        idx_dtype = self._get_index_dtype((self.indptr, self.indices,
                                           other.indptr, other.indices),
                                          maxval=max_bnnz)
        # 创建空的索引指针数组和索引数组
        indptr = np.empty(self.indptr.shape, dtype=idx_dtype)
        indices = np.empty(max_bnnz, dtype=idx_dtype)

        # 如果操作是布尔运算，则创建布尔类型的数据数组，否则根据数据类型进行类型提升
        bool_ops = ['_ne_', '_lt_', '_gt_', '_le_', '_ge_']
        if op in bool_ops:
            data = np.empty(R * C * max_bnnz, dtype=np.bool_)
        else:
            data = np.empty(R * C * max_bnnz, dtype=upcast(self.dtype, other.dtype))

        # 调用 C 库函数 fn 进行二进制操作
        fn(self.shape[0] // R, self.shape[1] // C, R, C,
           self.indptr.astype(idx_dtype),
           self.indices.astype(idx_dtype),
           self.data,
           other.indptr.astype(idx_dtype),
           other.indices.astype(idx_dtype),
           np.ravel(other.data),
           indptr,
           indices,
           data)

        # 获取实际的非零元素数量
        actual_bnnz = indptr[-1]
        # 截断 indices 和 data 数组，移除多余的空间
        indices = indices[:actual_bnnz]
        data = data[:R * C * actual_bnnz]

        # 如果实际的非零元素数量小于 max_bnnz 的一半，则复制 indices 和 data 数组
        if actual_bnnz < max_bnnz / 2:
            indices = indices.copy()
            data = data.copy()

        # 将 data 数组重新形状为 (-1, R, C)
        data = data.reshape(-1, R, C)

        # 返回一个新的实例，表示二进制操作的结果
        return self.__class__((data, indices, indptr), shape=self.shape)

    # needed by _data_matrix
    # 定义一个方法 `_with_data`，接受参数 `data` 和 `copy=True`
    def _with_data(self, data, copy=True):
        """Returns a matrix with the same sparsity structure as self,
        but with different data.  By default the structure arrays
        (i.e. .indptr and .indices) are copied.
        """
        # 如果 `copy` 参数为 True，复制结构数组 `.indptr` 和 `.indices`，并用 `data` 替换原数据
        if copy:
            return self.__class__((data, self.indices.copy(), self.indptr.copy()),
                                   shape=self.shape, dtype=data.dtype)
        # 如果 `copy` 参数为 False，直接使用 `data` 替换原数据，保留原有结构数组
        else:
            return self.__class__((data, self.indices, self.indptr),
                                   shape=self.shape, dtype=data.dtype)
#    # these functions are used by the parent class
#    # to remove redundancy between bsc_matrix and bsr_matrix
#    def _swap(self,x):
#        """swap the members of x if this is a column-oriented matrix
#        """
#        return (x[0],x[1])

# 检查对象 `x` 是否为 `bsr_matrix` 类型
def isspmatrix_bsr(x):
    """Is `x` of a bsr_matrix type?

    Parameters
    ----------
    x
        object to check for being a bsr matrix

    Returns
    -------
    bool
        True if `x` is a bsr matrix, False otherwise

    Examples
    --------
    >>> from scipy.sparse import bsr_array, bsr_matrix, csr_matrix, isspmatrix_bsr
    >>> isspmatrix_bsr(bsr_matrix([[5]]))
    True
    >>> isspmatrix_bsr(bsr_array([[5]]))
    False
    >>> isspmatrix_bsr(csr_matrix([[5]]))
    False
    """
    return isinstance(x, bsr_matrix)


# This namespace class separates array from matrix with isinstance
class bsr_array(_bsr_base, sparray):
    """
    Block Sparse Row format sparse array.

    This can be instantiated in several ways:
        bsr_array(D, [blocksize=(R,C)])
            where D is a 2-D ndarray.

        bsr_array(S, [blocksize=(R,C)])
            with another sparse array or matrix S (equivalent to S.tobsr())

        bsr_array((M, N), [blocksize=(R,C), dtype])
            to construct an empty sparse array with shape (M, N)
            dtype is optional, defaulting to dtype='d'.

        bsr_array((data, ij), [blocksize=(R,C), shape=(M, N)])
            where ``data`` and ``ij`` satisfy ``a[ij[0, k], ij[1, k]] = data[k]``

        bsr_array((data, indices, indptr), [shape=(M, N)])
            is the standard BSR representation where the block column
            indices for row i are stored in ``indices[indptr[i]:indptr[i+1]]``
            and their corresponding block values are stored in
            ``data[ indptr[i]: indptr[i+1] ]``. If the shape parameter is not
            supplied, the array dimensions are inferred from the index arrays.

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
        BSR format data array of the array
    indices
        BSR format index array of the array
    indptr
        BSR format index pointer array of the array
    blocksize
        Block size
    has_sorted_indices : bool
        Whether indices are sorted
    has_canonical_format : bool
    T

    Notes
    -----
    Sparse arrays can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    **Summary of BSR format**

    The Block Sparse Row (BSR) format is very similar to the Compressed
    Sparse Row (CSR) format. BSR is appropriate for sparse matrices with dense
    sub matrices like the last example below. Such sparse block matrices often
    arise in vector-valued finite element discretizations. In such cases, BSR is
    """
    considerably more efficient than CSR and CSC for many sparse arithmetic
    operations.

    **Blocksize**

    The blocksize (R,C) must evenly divide the shape of the sparse array (M,N).
    That is, R and C must satisfy the relationship ``M % R = 0`` and
    ``N % C = 0``.

    If no blocksize is specified, a simple heuristic is applied to determine
    an appropriate blocksize.

    **Canonical Format**

    In canonical format, there are no duplicate blocks and indices are sorted
    per row.

    **Limitations**

    Block Sparse Row format sparse arrays do not support slicing.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import bsr_array
    >>> bsr_array((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

    >>> row = np.array([0, 0, 1, 2, 2, 2])
    >>> col = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3 ,4, 5, 6])
    >>> bsr_array((data, (row, col)), shape=(3, 3)).toarray()
    array([[1, 0, 2],
           [0, 0, 3],
           [4, 5, 6]])

    >>> indptr = np.array([0, 2, 3, 6])
    >>> indices = np.array([0, 2, 2, 0, 1, 2])
    >>> data = np.array([1, 2, 3, 4, 5, 6]).repeat(4).reshape(6, 2, 2)
    >>> bsr_array((data,indices,indptr), shape=(6, 6)).toarray()
    array([[1, 1, 0, 0, 2, 2],
           [1, 1, 0, 0, 2, 2],
           [0, 0, 0, 0, 3, 3],
           [0, 0, 0, 0, 3, 3],
           [4, 4, 5, 5, 6, 6],
           [4, 4, 5, 5, 6, 6]])

    """
class bsr_matrix(spmatrix, _bsr_base):
    """
    Block Sparse Row format sparse matrix.

    This can be instantiated in several ways:
        bsr_matrix(D, [blocksize=(R,C)])
            where D is a 2-D ndarray.

        bsr_matrix(S, [blocksize=(R,C)])
            with another sparse array or matrix S (equivalent to S.tobsr())

        bsr_matrix((M, N), [blocksize=(R,C), dtype])
            to construct an empty sparse matrix with shape (M, N)
            dtype is optional, defaulting to dtype='d'.

        bsr_matrix((data, ij), [blocksize=(R,C), shape=(M, N)])
            where ``data`` and ``ij`` satisfy ``a[ij[0, k], ij[1, k]] = data[k]``

        bsr_matrix((data, indices, indptr), [shape=(M, N)])
            is the standard BSR representation where the block column
            indices for row i are stored in ``indices[indptr[i]:indptr[i+1]]``
            and their corresponding block values are stored in
            ``data[ indptr[i]: indptr[i+1] ]``. If the shape parameter is not
            supplied, the matrix dimensions are inferred from the index arrays.

    Attributes
    ----------
    dtype : dtype
        数据矩阵的数据类型
    shape : 2-tuple
        矩阵的形状
    ndim : int
        维度数（始终为2）
    nnz
        非零元素的数量
    size
        元素总数
    data
        矩阵的BSR格式数据数组
    indices
        矩阵的BSR格式索引数组
    indptr
        矩阵的BSR格式索引指针数组
    blocksize
        块大小
    has_sorted_indices : bool
        索引是否已排序
    has_canonical_format : bool
        是否为规范格式
    T

    Notes
    -----
    稀疏矩阵可用于算术运算：支持加法、减法、乘法、除法和矩阵幂运算。

    **BSR格式概述**

    块稀疏行（BSR）格式与压缩稀疏行（CSR）格式非常相似。对于具有密集子矩阵的稀疏矩阵，如下面的最后一个示例，BSR格式非常合适。这种稀疏块矩阵经常出现在向量值有限元离散化中。在这种情况下，BSR对于许多稀疏算术运算比CSR和CSC更有效。

    **块大小**

    块大小（R,C）必须均匀地划分稀疏矩阵的形状（M,N）。也就是说，R和C必须满足关系 ``M % R = 0`` 和 ``N % C = 0``。

    如果未指定块大小，则将应用简单的启发式方法来确定适当的块大小。

    **规范格式**

    在规范格式中，不存在重复的块，每行的索引都是排序的。

    **限制**

    块稀疏行格式的稀疏矩阵不支持切片操作。

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import bsr_matrix
    >>> bsr_matrix((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

    >>> row = np.array([0, 0, 1, 2, 2, 2])
    # 创建一个稀疏块状矩阵（BSR 矩阵）并将其转换为稠密数组
    col = np.array([0, 2, 2, 0, 1, 2])
    # 定义矩阵的数据部分
    data = np.array([1, 2, 3 ,4, 5, 6])
    # 使用数据和行索引构建 BSR 矩阵，然后将其转换为稠密数组
    bsr_matrix((data, (row, col)), shape=(3, 3)).toarray()
    
    # 创建一个稀疏块状矩阵（BSR 矩阵），指定行指针和列索引
    indptr = np.array([0, 2, 3, 6])
    # 指定每个数据块对应的列索引
    indices = np.array([0, 2, 2, 0, 1, 2])
    # 将数据重复多次，并按照给定形状重新排列以构建 BSR 矩阵，然后将其转换为稠密数组
    data = np.array([1, 2, 3, 4, 5, 6]).repeat(4).reshape(6, 2, 2)
    bsr_matrix((data, indices, indptr), shape=(6, 6)).toarray()
```