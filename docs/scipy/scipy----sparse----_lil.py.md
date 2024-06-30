# `D:\src\scipysrc\scipy\scipy\sparse\_lil.py`

```
"""
List of Lists sparse matrix class
"""

__docformat__ = "restructuredtext en"

__all__ = ['lil_array', 'lil_matrix', 'isspmatrix_lil']

from bisect import bisect_left

import numpy as np

from ._matrix import spmatrix
from ._base import _spbase, sparray, issparse
from ._index import IndexMixin, INT_TYPES, _broadcast_arrays
from ._sputils import (getdtype, isshape, isscalarlike, upcast_scalar,
                       check_shape, check_reshape_kwargs)
from . import _csparsetools


class _lil_base(_spbase, IndexMixin):
    _format = 'lil'

    def __init__(self, arg1, shape=None, dtype=None, copy=False, *, maxprint=None):
        _spbase.__init__(self, arg1, maxprint=maxprint)
        self.dtype = getdtype(dtype, arg1, default=float)

        # First get the shape
        if issparse(arg1):
            if arg1.format == "lil" and copy:
                A = arg1.copy()
            else:
                A = arg1.tolil()

            if dtype is not None:
                newdtype = getdtype(dtype)
                A = A.astype(newdtype, copy=False)

            self._shape = check_shape(A.shape)  # 获取稀疏矩阵的形状并检查
            self.dtype = A.dtype  # 设置矩阵数据类型
            self.rows = A.rows  # 获取行指示器列表
            self.data = A.data  # 获取数据列表
        elif isinstance(arg1, tuple):
            if isshape(arg1):
                if shape is not None:
                    raise ValueError('invalid use of shape parameter')
                M, N = arg1
                self._shape = check_shape((M, N))  # 检查给定的形状参数
                self.rows = np.empty((M,), dtype=object)  # 创建行指示器的空数组
                self.data = np.empty((M,), dtype=object)  # 创建数据的空数组
                for i in range(M):
                    self.rows[i] = []  # 初始化每行的空列表
                    self.data[i] = []  # 初始化每行的空列表
            else:
                raise TypeError('unrecognized lil_array constructor usage')  # 报错：不认识的构造函数用法
        else:
            # assume A is dense
            try:
                A = self._ascontainer(arg1)  # 尝试将输入转换为适合的容器类型
            except TypeError as e:
                raise TypeError('unsupported matrix type') from e  # 报错：不支持的矩阵类型
            if isinstance(self, sparray) and A.ndim != 2:
                raise ValueError(f"LIL arrays don't support {A.ndim}D input. Use 2D")  # 报错：LIL 矩阵不支持指定维度的输入
            A = self._csr_container(A, dtype=dtype).tolil()  # 将输入转换为 CSR 容器再转换为 LIL 格式

            self._shape = check_shape(A.shape)  # 检查并设置稀疏矩阵的形状
            self.dtype = getdtype(A.dtype)  # 获取并设置数据类型
            self.rows = A.rows  # 获取行指示器列表
            self.data = A.data  # 获取数据列表

    def __iadd__(self, other):
        self[:,:] = self + other  # 实现就地加法操作
        return self

    def __isub__(self, other):
        self[:,:] = self - other  # 实现就地减法操作
        return self

    def __imul__(self, other):
        if isscalarlike(other):
            self[:,:] = self * other  # 实现就地乘法操作（当 other 是标量时）
            return self
        else:
            return NotImplemented  # 返回未实现错误

    def __itruediv__(self, other):
        if isscalarlike(other):
            self[:,:] = self / other  # 实现就地除法操作（当 other 是标量时）
            return self
        else:
            return NotImplemented  # 返回未实现错误

    # Whenever the dimensions change, empty lists should be created for each
    # row
"""
    def _getnnz(self, axis=None):
        # 如果未指定轴，则计算所有行中非零元素的总数
        if axis is None:
            return sum([len(rowvals) for rowvals in self.data])
        
        # 如果轴小于0，则转换为正向轴索引
        if axis < 0:
            axis += 2
        
        # 如果轴为0，则计算每列中非零元素的数量
        if axis == 0:
            out = np.zeros(self.shape[1], dtype=np.intp)
            for row in self.rows:
                out[row] += 1
            return out
        
        # 如果轴为1，则计算每行中非零元素的数量
        elif axis == 1:
            return np.array([len(rowvals) for rowvals in self.data], dtype=np.intp)
        
        # 抛出轴超出范围的异常
        else:
            raise ValueError('axis out of bounds')

    # 将_getnnz方法的文档字符串设置为_spbase._getnnz方法的文档字符串
    _getnnz.__doc__ = _spbase._getnnz.__doc__

    def count_nonzero(self, axis=None):
        # 如果未指定轴，则计算所有行中非零元素的总数
        if axis is None:
            return sum(np.count_nonzero(rowvals) for rowvals in self.data)

        # 如果轴小于0，则转换为正向轴索引
        if axis < 0:
            axis += 2
        
        # 如果轴为0，则计算每列中非零元素的数量，并存储在数组out中
        if axis == 0:
            out = np.zeros(self.shape[1], dtype=np.intp)
            for row, data in zip(self.rows, self.data):
                mask = [c for c, d in zip(row, data) if d != 0]
                out[mask] += 1
            return out
        
        # 如果轴为1，则计算每行中非零元素的数量，并返回结果数组
        elif axis == 1:
            return np.array(
                [np.count_nonzero(rowvals) for rowvals in self.data], dtype=np.intp,
            )
        
        # 抛出轴超出范围的异常
        else:
            raise ValueError('axis out of bounds')

    # 将count_nonzero方法的文档字符串设置为_spbase.count_nonzero方法的文档字符串
    count_nonzero.__doc__ = _spbase.count_nonzero.__doc__

    def getrowview(self, i):
        """Returns a view of the 'i'th row (without copying).
        """
        # 创建一个新的稀疏矩阵对象new，包含单行数据和行索引
        new = self._lil_container((1, self.shape[1]), dtype=self.dtype)
        new.rows[0] = self.rows[i]
        new.data[0] = self.data[i]
        return new

    def getrow(self, i):
        """Returns a copy of the 'i'th row.
        """
        # 获取稀疏矩阵的形状
        M, N = self.shape
        
        # 如果索引i为负数，则转换为正向索引
        if i < 0:
            i += M
        
        # 检查索引i是否超出范围，如果超出则抛出异常
        if i < 0 or i >= M:
            raise IndexError('row index out of bounds')
        
        # 创建一个新的稀疏矩阵对象new，包含复制的单行数据和行索引
        new = self._lil_container((1, N), dtype=self.dtype)
        new.rows[0] = self.rows[i][:]
        new.data[0] = self.data[i][:]
        return new

    def __getitem__(self, key):
        # 对于简单的(int, int)索引，使用快速路径
        if (isinstance(key, tuple) and len(key) == 2 and
                isinstance(key[0], INT_TYPES) and
                isinstance(key[1], INT_TYPES)):
            # 调用_get_intXint方法处理索引并返回结果
            return self._get_intXint(*key)
        
        # 对于其他情况，使用普通路径处理索引
        return IndexMixin.__getitem__(self, key)

    def _asindices(self, idx, N):
        # LIL算法会自动处理索引范围，此处无需再次检查
        try:
            x = np.asarray(idx)
        except (ValueError, TypeError, MemoryError) as e:
            raise IndexError('invalid index') from e
        
        # 确保索引数组的维度不超过2
        if x.ndim not in (1, 2):
            raise IndexError('Index dimension must be <= 2')
        
        return x

    def _get_intXint(self, row, col):
        # 调用_csparsetools模块的函数lil_get1处理稀疏矩阵中指定位置(row, col)的元素
        v = _csparsetools.lil_get1(self.shape[0], self.shape[1], self.rows,
                                   self.data, row, col)
        return self.dtype.type(v)
    # 根据指定的行和列索引范围获取稀疏矩阵中的元素，返回一个新的稀疏矩阵
    def _get_sliceXint(self, row, col):
        # 将行索引转换为指定范围内的索引列表
        row = range(*row.indices(self.shape[0]))
        # 调用_get_row_ranges方法获取指定行和列范围的稀疏矩阵切片
        return self._get_row_ranges(row, slice(col, col+1))

    # 根据指定的行和列索引获取稀疏矩阵中的元素，返回一个新的稀疏矩阵
    def _get_arrayXint(self, row, col):
        # 如果行索引是多余的维度，则压缩成一维数组
        row = row.squeeze()
        # 调用_get_row_ranges方法获取指定行和列范围的稀疏矩阵切片
        return self._get_row_ranges(row, slice(col, col+1))

    # 根据指定的行和列索引获取稀疏矩阵中的元素，返回一个新的稀疏矩阵
    def _get_intXslice(self, row, col):
        # 调用_get_row_ranges方法获取指定行和列范围的稀疏矩阵切片
        return self._get_row_ranges((row,), col)

    # 根据指定的行和列索引范围获取稀疏矩阵中的元素，返回一个新的稀疏矩阵
    def _get_sliceXslice(self, row, col):
        # 将行索引转换为指定范围内的索引列表
        row = range(*row.indices(self.shape[0]))
        # 调用_get_row_ranges方法获取指定行和列范围的稀疏矩阵切片
        return self._get_row_ranges(row, col)

    # 根据指定的行和列索引范围获取稀疏矩阵中的元素，返回一个新的稀疏矩阵
    def _get_arrayXslice(self, row, col):
        # 调用_get_row_ranges方法获取指定行和列范围的稀疏矩阵切片
        return self._get_row_ranges(row, col)

    # 根据指定的行和列索引获取稀疏矩阵中的元素，返回一个新的稀疏矩阵
    def _get_intXarray(self, row, col):
        # 将行索引转换为NumPy数组，确保数据类型和维度匹配
        row = np.array(row, dtype=col.dtype, ndmin=1)
        # 调用_get_columnXarray方法获取指定行和列索引的稀疏矩阵切片
        return self._get_columnXarray(row, col)

    # 根据指定的行和列索引范围获取稀疏矩阵中的元素，返回一个新的稀疏矩阵
    def _get_sliceXarray(self, row, col):
        # 将行索引转换为指定范围内的索引数组
        row = np.arange(*row.indices(self.shape[0]))
        # 调用_get_columnXarray方法获取指定行和列索引的稀疏矩阵切片
        return self._get_columnXarray(row, col)

    # 根据指定的行和列索引获取稀疏矩阵中的元素，返回一个新的稀疏矩阵
    def _get_columnXarray(self, row, col):
        # 调用_broadcast_arrays方法确保行和列索引数组的广播形状匹配
        row, col = _broadcast_arrays(row[:,None], col)
        # 调用_get_arrayXarray方法获取指定行和列索引的稀疏矩阵切片
        return self._get_arrayXarray(row, col)

    # 根据指定的行和列索引获取稀疏矩阵中的元素，返回一个新的稀疏矩阵
    def _get_arrayXarray(self, row, col):
        # 准备索引以供内存视图使用
        i, j = map(np.atleast_2d, _prepare_index_for_memoryview(row, col))
        # 创建新的稀疏矩阵容器
        new = self._lil_container(i.shape, dtype=self.dtype)
        # 调用_csparsetools模块中的lil_fancy_get函数，实现高效的稀疏矩阵元素获取
        _csparsetools.lil_fancy_get(self.shape[0], self.shape[1],
                                    self.rows, self.data,
                                    new.rows, new.data,
                                    i, j)
        return new

    # 根据指定的行和列索引范围获取稀疏矩阵中的元素，返回一个新的稀疏矩阵
    def _get_row_ranges(self, rows, col_slice):
        """
        针对列索引为切片的情况进行快速索引。

        通过按列顺序更高效地跳过零元素，优化性能比暴力方法更高。

        Parameters
        ----------
        rows : sequence or range
            索引的行。如果是range类型，必须在有效范围内。
        col_slice : slice
            索引的列

        """
        # 计算列切片的起始、结束和步长
        j_start, j_stop, j_stride = col_slice.indices(self.shape[1])
        # 创建列索引范围的range对象
        col_range = range(j_start, j_stop, j_stride)
        # 计算新稀疏矩阵的列数
        nj = len(col_range)
        # 创建新的稀疏矩阵容器
        new = self._lil_container((len(rows), nj), dtype=self.dtype)

        # 调用_csparsetools模块中的lil_get_row_ranges函数，实现快速稀疏矩阵行范围获取
        _csparsetools.lil_get_row_ranges(self.shape[0], self.shape[1],
                                         self.rows, self.data,
                                         new.rows, new.data,
                                         rows,
                                         j_start, j_stop, j_stride, nj)

        return new

    # 将指定的值插入到稀疏矩阵中的指定位置
    def _set_intXint(self, row, col, x):
        # 调用_csparsetools模块中的lil_insert函数，实现稀疏矩阵的元素插入
        _csparsetools.lil_insert(self.shape[0], self.shape[1], self.rows,
                                 self.data, row, col, x)
    def _set_arrayXarray(self, row, col, x):
        # 将 row、col、x 转换为至少是二维的数组
        i, j, x = map(np.atleast_2d, _prepare_index_for_memoryview(row, col, x))
        # 调用底层 C 函数实现稀疏矩阵的设置操作
        _csparsetools.lil_fancy_set(self.shape[0], self.shape[1],
                                    self.rows, self.data,
                                    i, j, x)

    def _set_arrayXarray_sparse(self, row, col, x):
        # 如果 x 是稀疏矩阵，则转换为密集矩阵
        x = np.asarray(x.toarray(), dtype=self.dtype)
        # 对 x 和 row 进行广播，以匹配维度
        x, _ = _broadcast_arrays(x, row)
        # 调用 _set_arrayXarray 方法实现稀疏矩阵的设置
        self._set_arrayXarray(row, col, x)

    def __setitem__(self, key, x):
        if isinstance(key, tuple) and len(key) == 2:
            row, col = key
            # 对简单的 (int, int) 索引进行快速处理
            if isinstance(row, INT_TYPES) and isinstance(col, INT_TYPES):
                # 将 x 转换为相应类型
                x = self.dtype.type(x)
                if x.size > 1:
                    raise ValueError("Trying to assign a sequence to an item")
                # 调用 _set_intXint 方法实现简单索引的设置
                return self._set_intXint(row, col, x)
            # 对于整个稀疏矩阵的赋值进行快速处理
            if (isinstance(row, slice) and isinstance(col, slice) and
                    row == slice(None) and col == slice(None) and
                    issparse(x) and x.shape == self.shape):
                # 将 x 转换为 LIL 格式的容器
                x = self._lil_container(x, dtype=self.dtype)
                self.rows = x.rows
                self.data = x.data
                return
        # 其他情况按照普通路径处理
        IndexMixin.__setitem__(self, key, x)

    def _mul_scalar(self, other):
        if other == 0:
            # 乘以零：返回零矩阵
            new = self._lil_container(self.shape, dtype=self.dtype)
        else:
            # 提升标量类型并复制当前对象
            res_dtype = upcast_scalar(self.dtype, other)
            new = self.copy()
            new = new.astype(res_dtype)
            # 将每个元素乘以标量
            for j, rowvals in enumerate(new.data):
                new.data[j] = [val*other for val in rowvals]
        return new

    def __truediv__(self, other):           # self / other
        if isscalarlike(other):
            # 复制当前对象并设置数据类型
            new = self.copy()
            new.dtype = np.result_type(self, other)
            # 将每个元素除以标量
            for j, rowvals in enumerate(new.data):
                new.data[j] = [val/other for val in rowvals]
            return new
        else:
            # 转换为 CSR 格式并进行除法操作
            return self.tocsr() / other

    def copy(self):
        M, N = self.shape
        # 创建一个与当前对象相同形状的新对象
        new = self._lil_container(self.shape, dtype=self.dtype)
        # 使用底层 C 函数快速复制行范围
        _csparsetools.lil_get_row_ranges(M, N, self.rows, self.data,
                                         new.rows, new.data, range(M),
                                         0, N, 1, N)
        return new

    copy.__doc__ = _spbase.copy.__doc__
    # 根据给定的参数调整数组的形状
    def reshape(self, *args, **kwargs):
        # 检查并返回有效的形状
        shape = check_shape(args, self.shape)
        # 检查并返回重塑操作的顺序和是否复制的参数
        order, copy = check_reshape_kwargs(kwargs)

        # 如果形状与当前形状相同，则根据复制参数决定返回副本或者自身
        if shape == self.shape:
            if copy:
                return self.copy()
            else:
                return self

        # 根据给定的形状创建新的稀疏矩阵容器
        new = self._lil_container(shape, dtype=self.dtype)

        # 根据指定的顺序进行重塑操作
        if order == 'C':  # C顺序，按行展开
            ncols = self.shape[1]
            for i, row in enumerate(self.rows):
                for col, j in enumerate(row):
                    # 根据C顺序的索引计算新的坐标
                    new_r, new_c = np.unravel_index(i * ncols + j, shape)
                    new[new_r, new_c] = self[i, j]
        elif order == 'F':  # Fortran顺序，按列展开
            nrows = self.shape[0]
            for i, row in enumerate(self.rows):
                for col, j in enumerate(row):
                    # 根据Fortran顺序的索引计算新的坐标
                    new_r, new_c = np.unravel_index(i + j * nrows, shape, order)
                    new[new_r, new_c] = self[i, j]
        else:
            raise ValueError("'order' must be 'C' or 'F'")

        # 返回重塑后的新稀疏矩阵
        return new

    reshape.__doc__ = _spbase.reshape.__doc__

    # 调整稀疏矩阵的大小
    def resize(self, *shape):
        # 检查并返回有效的形状
        shape = check_shape(shape)
        new_M, new_N = shape
        M, N = self.shape

        # 如果新形状的行数小于当前形状的行数，删除多余的行
        if new_M < M:
            self.rows = self.rows[:new_M]
            self.data = self.data[:new_M]
        # 如果新形状的行数大于当前形状的行数，扩展行数并初始化新行
        elif new_M > M:
            self.rows = np.resize(self.rows, new_M)
            self.data = np.resize(self.data, new_M)
            for i in range(M, new_M):
                self.rows[i] = []
                self.data[i] = []

        # 如果新形状的列数小于当前形状的列数，删除多余的列
        if new_N < N:
            for row, data in zip(self.rows, self.data):
                trunc = bisect_left(row, new_N)
                del row[trunc:]
                del data[trunc:]

        # 更新稀疏矩阵的形状
        self._shape = shape

    resize.__doc__ = _spbase.resize.__doc__

    # 将稀疏矩阵转换为密集数组
    def toarray(self, order=None, out=None):
        # 处理并返回转换为数组的参数
        d = self._process_toarray_args(order, out)
        for i, row in enumerate(self.rows):
            for pos, j in enumerate(row):
                # 将稀疏矩阵的数据复制到数组中
                d[i, j] = self.data[i][pos]
        return d

    toarray.__doc__ = _spbase.toarray.__doc__

    # 返回稀疏矩阵的转置
    def transpose(self, axes=None, copy=False):
        # 将稀疏矩阵转换为CSR格式，然后返回其转置的LIL格式
        return self.tocsr(copy=copy).transpose(axes=axes, copy=False).tolil(copy=False)

    transpose.__doc__ = _spbase.transpose.__doc__

    # 返回稀疏矩阵的LIL格式表示
    def tolil(self, copy=False):
        if copy:
            # 如果复制标志为True，则返回稀疏矩阵的副本
            return self.copy()
        else:
            # 否则返回稀疏矩阵本身
            return self

    tolil.__doc__ = _spbase.tolil.__doc__
    # 将稀疏矩阵转换为 CSR 格式
    def tocsr(self, copy=False):
        # 获取矩阵的行数 M 和列数 N
        M, N = self.shape
        # 如果行数 M 或列数 N 为零，则返回一个空的 CSR 容器
        if M == 0 or N == 0:
            return self._csr_container((M, N), dtype=self.dtype)

        # 构造 indptr 数组
        if M * N <= np.iinfo(np.int32).max:
            # 快速路径：已知不需要使用 64 位索引
            idx_dtype = np.int32
            # 创建一个空的 indptr 数组，数据类型为 idx_dtype
            indptr = np.empty(M + 1, dtype=idx_dtype)
            # 设置起始位置
            indptr[0] = 0
            # 获取每行非零元素的数量
            _csparsetools.lil_get_lengths(self.rows, indptr[1:])
            # 计算累积和，得到每行结束的索引位置
            np.cumsum(indptr, out=indptr)
            # 获取非零元素的总数
            nnz = indptr[-1]
        else:
            # 计算合适的索引数据类型
            idx_dtype = self._get_index_dtype(maxval=N)
            # 创建一个空的长度数组，数据类型为 idx_dtype
            lengths = np.empty(M, dtype=idx_dtype)
            # 获取每行非零元素的数量
            _csparsetools.lil_get_lengths(self.rows, lengths)
            # 计算非零元素的总数
            nnz = lengths.sum(dtype=np.int64)
            # 再次计算合适的索引数据类型
            idx_dtype = self._get_index_dtype(maxval=max(N, nnz))
            # 创建一个空的 indptr 数组，数据类型为 idx_dtype
            indptr = np.empty(M + 1, dtype=idx_dtype)
            # 设置起始位置
            indptr[0] = 0
            # 计算累积和，得到每行结束的索引位置
            np.cumsum(lengths, dtype=idx_dtype, out=indptr[1:])

        # 创建空的 indices 数组，数据类型为 idx_dtype
        indices = np.empty(nnz, dtype=idx_dtype)
        # 创建空的 data 数组，数据类型为 self.dtype
        data = np.empty(nnz, dtype=self.dtype)
        # 将稀疏矩阵的行压平成 indices 数组
        _csparsetools.lil_flatten_to_array(self.rows, indices)
        # 将稀疏矩阵的数据压平成 data 数组
        _csparsetools.lil_flatten_to_array(self.data, data)

        # 初始化 CSR 矩阵，使用 data、indices 和 indptr 组成的元组
        return self._csr_container((data, indices, indptr), shape=self.shape)

    # 将 tocsr 方法的文档字符串设置为 _spbase.tocsr 方法的文档字符串
    tocsr.__doc__ = _spbase.tocsr.__doc__
# 将索引和数据数组转换为适合传递给 Cython 的 fancy getset 例程的形式。
# 这些转换是必要的，因为要确保整数索引数组属于接受的类型之一，
# 并确保数组是可写的，以便 Cython 内存视图支持不会对其产生影响。
def _prepare_index_for_memoryview(i, j, x=None):
    if i.dtype > j.dtype:
        j = j.astype(i.dtype)  # 如果 j 的数据类型比 i 大，转换 j 的数据类型为 i 的数据类型
    elif i.dtype < j.dtype:
        i = i.astype(j.dtype)  # 如果 i 的数据类型比 j 小，转换 i 的数据类型为 j 的数据类型

    if not i.flags.writeable or i.dtype not in (np.int32, np.int64):
        i = i.astype(np.intp)  # 如果 i 不可写或者数据类型不是 int32 或 int64，则转换为 np.intp
    if not j.flags.writeable or j.dtype not in (np.int32, np.int64):
        j = j.astype(np.intp)  # 如果 j 不可写或者数据类型不是 int32 或 int64，则转换为 np.intp

    if x is not None:
        if not x.flags.writeable:
            x = x.copy()  # 如果 x 不可写，则复制一份可写的副本
        return i, j, x
    else:
        return i, j  # 如果 x 是 None，则返回重新格式化的 i 和 j


# 检查 x 是否为 lil_matrix 类型的函数
def isspmatrix_lil(x):
    """
    Is `x` of lil_matrix type?

    Parameters
    ----------
    x
        object to check for being a lil matrix

    Returns
    -------
    bool
        True if `x` is a lil matrix, False otherwise

    Examples
    --------
    >>> from scipy.sparse import lil_array, lil_matrix, coo_matrix, isspmatrix_lil
    >>> isspmatrix_lil(lil_matrix([[5]]))
    True
    >>> isspmatrix_lil(lil_array([[5]]))
    False
    >>> isspmatrix_lil(coo_matrix([[5]]))
    False
    """
    return isinstance(x, lil_matrix)


# 该命名空间类将数组与矩阵分开，使用 isinstance 来实现
class lil_array(_lil_base, sparray):
    """
    Row-based LIst of Lists sparse array.

    This is a structure for constructing sparse arrays incrementally.
    Note that inserting a single item can take linear time in the worst case;
    to construct the array efficiently, make sure the items are pre-sorted by
    index, per row.

    This can be instantiated in several ways:
        lil_array(D)
            where D is a 2-D ndarray

        lil_array(S)
            with another sparse array or matrix S (equivalent to S.tolil())

        lil_array((M, N), [dtype])
            to construct an empty array with shape (M, N)
            dtype is optional, defaulting to dtype='d'.

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
        LIL format data array of the array
    rows
        LIL format row index array of the array
    T

    Notes
    -----
    Sparse arrays can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    Advantages of the LIL format
        - supports flexible slicing
        - changes to the array sparsity structure are efficient
    """
    pass  # 此处为占位符，表示该类暂时不包含额外的方法或属性实现
    Disadvantages of the LIL format
        - arithmetic operations LIL + LIL are slow (consider CSR or CSC)
        - slow column slicing (consider CSC)
        - slow matrix vector products (consider CSR or CSC)

    Intended Usage
        - LIL is a convenient format for constructing sparse arrays
        - once an array has been constructed, convert to CSR or
          CSC format for fast arithmetic and matrix vector operations
        - consider using the COO format when constructing large arrays

    Data Structure
        - An array (``self.rows``) of rows, each of which is a sorted
          list of column indices of non-zero elements.
        - The corresponding nonzero values are stored in similar
          fashion in ``self.data``.
# 定义了一个稀疏矩阵的类 `lil_matrix`，基于行的链表稀疏矩阵格式。

class lil_matrix(spmatrix, _lil_base):
    """
    Row-based LIst of Lists sparse matrix.

    This is a structure for constructing sparse matrices incrementally.
    Note that inserting a single item can take linear time in the worst case;
    to construct the matrix efficiently, make sure the items are pre-sorted by
    index, per row.

    This can be instantiated in several ways:
        lil_matrix(D)
            where D is a 2-D ndarray

        lil_matrix(S)
            with another sparse array or matrix S (equivalent to S.tolil())

        lil_matrix((M, N), [dtype])
            to construct an empty matrix with shape (M, N)
            dtype is optional, defaulting to dtype='d'.

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
        LIL format data array of the matrix
    rows
        LIL format row index array of the matrix
    T

    Notes
    -----
    Sparse matrices can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    Advantages of the LIL format
        - supports flexible slicing
        - changes to the matrix sparsity structure are efficient

    Disadvantages of the LIL format
        - arithmetic operations LIL + LIL are slow (consider CSR or CSC)
        - slow column slicing (consider CSC)
        - slow matrix vector products (consider CSR or CSC)

    Intended Usage
        - LIL is a convenient format for constructing sparse matrices
        - once a matrix has been constructed, convert to CSR or
          CSC format for fast arithmetic and matrix vector operations
        - consider using the COO format when constructing large matrices

    Data Structure
        - An array (``self.rows``) of rows, each of which is a sorted
          list of column indices of non-zero elements.
        - The corresponding nonzero values are stored in similar
          fashion in ``self.data``.

    """
```