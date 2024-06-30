# `D:\src\scipysrc\scipy\scipy\sparse\_compressed.py`

```
"""
Base class for sparse matrix formats using compressed storage.
"""
__all__ = []  # 导出的模块列表为空

from warnings import warn  # 导入警告模块中的warn函数
import itertools  # 导入迭代工具模块
import operator  # 导入操作符模块

import numpy as np  # 导入NumPy库，并用np作为别名
from scipy._lib._util import _prune_array, copy_if_needed  # 从SciPy库中导入_prune_array和copy_if_needed函数

from ._base import _spbase, issparse, sparray, SparseEfficiencyWarning  # 从当前包中导入基础模块、判断稀疏矩阵函数、稀疏数组类和稀疏效率警告
from ._data import _data_matrix, _minmax_mixin  # 从数据模块中导入数据矩阵类和最大最小混合类
from . import _sparsetools  # 从当前包中导入稀疏工具模块
from ._sparsetools import (get_csr_submatrix, csr_sample_offsets, csr_todense,  # 从稀疏工具模块中导入多个函数
                           csr_sample_values, csr_row_index, csr_row_slice,
                           csr_column_index1, csr_column_index2)
from ._index import IndexMixin  # 从索引模块中导入索引混合类
from ._sputils import (upcast, upcast_char, to_native, isdense, isshape,  # 从工具模块中导入多个实用函数
                       getdtype, isscalarlike, isintlike, downcast_intp_index,
                       get_sum_dtype, check_shape, is_pydata_spmatrix)


class _cs_matrix(_data_matrix, _minmax_mixin, IndexMixin):
    """
    base array/matrix class for compressed row- and column-oriented arrays/matrices
    """

    def _getnnz(self, axis=None):
        """
        Return the number of nonzero entries in the matrix.
        
        Parameters
        ----------
        axis : {None, int}, optional
            Axis along which to count nonzero entries. If None, the total number
            of nonzero entries is returned. If axis is an integer, 0 refers to 
            rows and 1 refers to columns.
        
        Returns
        -------
        nnz : int or numpy.ndarray
            Number of nonzero entries. If axis is None or an integer, returns 
            either a scalar or a 1-D array depending on the input.

        Raises
        ------
        ValueError
            If axis is out of bounds.
        """
        if axis is None:
            return int(self.indptr[-1])  # 返回最后一个指针值，表示非零元素的数量
        elif self.ndim == 1:
            if axis in (0, -1):
                return int(self.indptr[-1])  # 返回最后一个指针值，表示非零元素的数量
            raise ValueError('axis out of bounds')  # 报错，轴越界
        else:
            if axis < 0:
                axis += 2
            axis, _ = self._swap((axis, 1 - axis))
            _, N = self._swap(self.shape)
            if axis == 0:
                return np.bincount(downcast_intp_index(self.indices), minlength=N)
                # 返回每行（或列）非零元素的数量，使用downcast_intp_index进行下标转换
            elif axis == 1:
                return np.diff(self.indptr)  # 返回每列（或行）非零元素的数量
            raise ValueError('axis out of bounds')  # 报错，轴越界

    _getnnz.__doc__ = _spbase._getnnz.__doc__  # 继承_spbase中_getnnz的文档字符串

    def count_nonzero(self, axis=None):
        """
        Count the number of non-zero entries in the matrix.

        Parameters
        ----------
        axis : {None, int}, optional
            Axis along which to count nonzero entries. If None, the total number
            of nonzero entries is returned. If axis is an integer, 0 refers to 
            rows and 1 refers to columns.

        Returns
        -------
        nnz : int or numpy.ndarray
            Number of nonzero entries along the specified axis. If axis is None 
            or an integer, returns either a scalar or a 1-D array depending on 
            the input.

        Raises
        ------
        ValueError
            If axis is out of bounds.
        """
        self.sum_duplicates()  # 合并重复条目
        if axis is None:
            return np.count_nonzero(self.data)  # 统计所有非零元素的数量

        if self.ndim == 1:
            if axis not in (0, -1):
                raise ValueError('axis out of bounds')  # 报错，轴越界
            return np.count_nonzero(self.data)  # 统计所有非零元素的数量

        if axis < 0:
            axis += 2
        axis, _ = self._swap((axis, 1 - axis))
        if axis == 0:
            _, N = self._swap(self.shape)
            mask = self.data != 0
            idx = self.indices if mask.all() else self.indices[mask]
            return np.bincount(downcast_intp_index(idx), minlength=N)
            # 返回每行（或列）非零元素的数量，使用downcast_intp_index进行下标转换
        elif axis == 1:
            if self.data.all():
                return np.diff(self.indptr)
            pairs = itertools.pairwise(self.indptr)
            return np.array([np.count_nonzero(self.data[i:j]) for i, j in pairs])
            # 返回每列（或行）非零元素的数量
        else:
            raise ValueError('axis out of bounds')  # 报错，轴越界

    count_nonzero.__doc__ = _spbase.count_nonzero.__doc__  # 继承_spbase中count_nonzero的文档字符串
    def check_format(self, full_check=True):
        """Check whether the array/matrix respects the CSR or CSC format.

        Parameters
        ----------
        full_check : bool, optional
            If `True`, run rigorous check, scanning arrays for valid values.
            Note that activating those check might copy arrays for casting,
            modifying indices and index pointers' inplace.
            If `False`, run basic checks on attributes. O(1) operations.
            Default is `True`.
        """
        # index arrays should have integer data types
        # 检查索引指针的数据类型是否为整数型
        if self.indptr.dtype.kind != 'i':
            warn(f"indptr array has non-integer dtype ({self.indptr.dtype.name})",
                 stacklevel=3)
        # 检查索引数组的数据类型是否为整数型
        if self.indices.dtype.kind != 'i':
            warn(f"indices array has non-integer dtype ({self.indices.dtype.name})",
                 stacklevel=3)

        # check array shapes
        # 检查数组的形状是否为一维
        for x in [self.data.ndim, self.indices.ndim, self.indptr.ndim]:
            if x != 1:
                raise ValueError('data, indices, and indptr should be 1-D')

        # check index pointer. Use _swap to determine proper bounds
        # 检查索引指针的大小，使用 _swap 方法确定正确的边界
        M, N = self._swap(self._shape_as_2d)

        # check if index pointer size matches M + 1
        # 检查索引指针的长度是否为 M + 1
        if (len(self.indptr) != M + 1):
            raise ValueError(f"index pointer size {len(self.indptr)} should be {M + 1}")
        # check if index pointer starts with 0
        # 检查索引指针的第一个值是否为 0
        if (self.indptr[0] != 0):
            raise ValueError("index pointer should start with 0")

        # check index and data arrays
        # 检查索引和数据数组是否具有相同的大小
        if (len(self.indices) != len(self.data)):
            raise ValueError("indices and data should have the same size")
        # 检查索引指针的最后一个值是否小于索引和数据数组的大小
        if (self.indptr[-1] > len(self.indices)):
            raise ValueError("Last value of index pointer should be less than "
                             "the size of index and data arrays")

        # prune the matrix
        # 调整矩阵
        self.prune()

        if full_check:
            # check format validity (more expensive)
            # 检查格式的有效性（更昂贵的检查）
            if self.nnz > 0:
                # check if indices are within bounds
                # 检查索引是否在范围内
                if self.indices.max() >= N:
                    raise ValueError(f"indices must be < {N}")
                # check if indices are non-negative
                # 检查索引是否非负
                if self.indices.min() < 0:
                    raise ValueError("indices must be >= 0")
                # check if indptr is non-decreasing
                # 检查 indptr 是否为非递减序列
                if np.diff(self.indptr).min() < 0:
                    raise ValueError("indptr must be a non-decreasing sequence")

            # determine appropriate dtype for indices and set
            # 确定索引的适当数据类型并设置
            idx_dtype = self._get_index_dtype((self.indptr, self.indices))
            self.indptr = np.asarray(self.indptr, dtype=idx_dtype)
            self.indices = np.asarray(self.indices, dtype=idx_dtype)
            # convert data to native format
            # 将数据转换为本机格式
            self.data = to_native(self.data)

        # if not self.has_sorted_indices():
        #    warn('Indices were not in sorted order.  Sorting indices.')
        #    self.sort_indices()
        #    assert(self.has_sorted_indices())
        # TODO check for duplicates?

    #######################
    # Boolean comparisons #
    #######################
    # 定义了一个方法，用于在稀疏矩阵对象中执行二元运算，针对标量类型的操作。结果返回一个新的稀疏数组，保持规范形式。
    def _scalar_binopt(self, other, op):
        """Scalar version of self._binopt, for cases in which no new nonzeros
        are added. Produces a new sparse array in canonical form.
        """
        # 合并重复的条目，确保稀疏矩阵数据结构的一致性
        self.sum_duplicates()
        # 使用给定的操作函数 op 对当前对象的数据和另一个标量 other 进行运算，生成新的稀疏数组
        res = self._with_data(op(self.data, other), copy=True)
        # 清除结果中的零条目，保持数据结构的稀疏性质
        res.eliminate_zeros()
        # 返回经过运算后的稀疏数组对象
        return res

    # 定义了等于操作符的方法，用于比较当前稀疏矩阵对象与其他对象的相等性
    def __eq__(self, other):
        # 如果 other 是标量类型
        # Scalar other.
        if isscalarlike(other):
            # 如果 other 是 NaN，返回一个布尔类型的稀疏矩阵对象
            if np.isnan(other):
                return self.__class__(self.shape, dtype=np.bool_)

            # 如果 other 等于 0
            if other == 0:
                # 发出警告，使用 == 比较稀疏矩阵与 0 是低效的，建议使用 !=
                warn("Comparing a sparse matrix with 0 using == is inefficient"
                     ", try using != instead.", SparseEfficiencyWarning,
                     stacklevel=3)
                # 创建一个全部为真的布尔类型稀疏矩阵对象
                all_true = self.__class__(np.ones(self.shape, dtype=np.bool_))
                # 对当前对象与 other 进行非相等比较，并返回结果
                inv = self._scalar_binopt(other, operator.ne)
                return all_true - inv
            else:
                # 对当前对象与 other 进行相等比较，并返回结果
                return self._scalar_binopt(other, operator.eq)
        
        # 如果 other 是密集矩阵类型
        # Dense other.
        elif isdense(other):
            # 将当前稀疏矩阵转换为密集矩阵后，执行相等比较，并返回结果
            return self.todense() == other
        
        # 如果 other 是 Pydata 稀疏矩阵类型（未实现）
        # Pydata sparse other.
        elif is_pydata_spmatrix(other):
            # 返回未实现的结果
            return NotImplemented
        
        # 如果 other 是稀疏矩阵类型
        # Sparse other.
        elif issparse(other):
            # 发出警告，使用 == 比较稀疏矩阵是低效的，建议使用 !=
            warn("Comparing sparse matrices using == is inefficient, try using"
                 " != instead.", SparseEfficiencyWarning, stacklevel=3)
            # TODO 稀疏广播
            # 如果两个稀疏矩阵的形状不同，返回 False
            if self.shape != other.shape:
                return False
            # 如果两个稀疏矩阵的存储格式不同，将 other 转换为当前稀疏矩阵的格式
            elif self.format != other.format:
                other = other.asformat(self.format)
            # 对当前对象与 other 进行非相等比较，并返回结果
            res = self._binopt(other, '_ne_')
            # 创建一个全部为真的布尔类型稀疏矩阵对象
            all_true = self.__class__(np.ones(self.shape, dtype=np.bool_))
            return all_true - res
        
        # 如果 other 类型不在已知的类型之中，返回未实现的结果
        else:
            return NotImplemented
    def __ne__(self, other):
        # 如果其他对象是标量
        if isscalarlike(other):
            # 如果其他对象是 NaN，发出警告
            if np.isnan(other):
                warn("Comparing a sparse matrix with nan using != is"
                     " inefficient", SparseEfficiencyWarning, stacklevel=3)
                # 返回一个全为 True 的稀疏矩阵
                all_true = self.__class__(np.ones(self.shape, dtype=np.bool_))
                return all_true
            # 如果其他对象不等于 0，发出警告
            elif other != 0:
                warn("Comparing a sparse matrix with a nonzero scalar using !="
                     " is inefficient, try using == instead.",
                     SparseEfficiencyWarning, stacklevel=3)
                # 返回一个全为 True 的稀疏矩阵减去与其他对象相等的稀疏矩阵
                all_true = self.__class__(np.ones(self.shape), dtype=np.bool_)
                inv = self._scalar_binopt(other, operator.eq)
                return all_true - inv
            else:
                # 使用 _scalar_binopt 方法比较与其他对象不等的情况
                return self._scalar_binopt(other, operator.ne)
        # 如果其他对象是密集矩阵
        elif isdense(other):
            # 将稀疏矩阵转换为密集矩阵后比较不等
            return self.todense() != other
        # 如果其他对象是 Pydata 稀疏矩阵
        elif is_pydata_spmatrix(other):
            # 返回未实现的操作
            return NotImplemented
        # 如果其他对象是稀疏矩阵
        elif issparse(other):
            # TODO 稀疏广播
            # 如果形状不同，返回 True
            if self.shape != other.shape:
                return True
            # 如果格式不同，将其他对象转换为相同格式
            elif self.format != other.format:
                other = other.asformat(self.format)
            # 使用 _binopt 方法比较不等
            return self._binopt(other, '_ne_')
        else:
            # 返回未实现的操作
            return NotImplemented

    def _inequality(self, other, op, op_name, bad_scalar_msg):
        # 如果其他对象是标量
        if isscalarlike(other):
            # 如果其他对象等于 0，并且操作名是 '_le_' 或 '_ge_'，抛出未实现错误
            if 0 == other and op_name in ('_le_', '_ge_'):
                raise NotImplementedError(" >= and <= don't work with 0.")
            # 如果操作 op(0, other) 返回 True，发出警告
            elif op(0, other):
                warn(bad_scalar_msg, SparseEfficiencyWarning, stacklevel=3)
                # 创建一个填充了其他对象值的稀疏矩阵
                other_arr = np.empty(self.shape, dtype=np.result_type(other))
                other_arr.fill(other)
                other_arr = self.__class__(other_arr)
                return self._binopt(other_arr, op_name)
            else:
                # 使用 _scalar_binopt 方法比较标量操作
                return self._scalar_binopt(other, op)
        # 如果其他对象是密集矩阵
        elif isdense(other):
            # 使用操作 op 比较稀疏矩阵和密集矩阵
            return op(self.todense(), other)
        # 如果其他对象是稀疏矩阵
        elif issparse(other):
            # TODO 稀疏广播
            # 如果形状不同，引发值错误
            if self.shape != other.shape:
                raise ValueError("inconsistent shapes")
            # 如果格式不同，将其他对象转换为相同格式
            elif self.format != other.format:
                other = other.asformat(self.format)
            # 如果操作名不是 '_ge_' 或 '_le_'，使用 _binopt 方法比较
            if op_name not in ('_ge_', '_le_'):
                return self._binopt(other, op_name)

            # 发出警告，比较稀疏矩阵使用 '_ge_' 或 '_le_' 是低效的
            warn("Comparing sparse matrices using >= and <= is inefficient, "
                 "using <, >, or !=, instead.",
                 SparseEfficiencyWarning, stacklevel=3)
            # 返回一个全为 True 的稀疏矩阵减去与其他对象比较的结果
            all_true = self.__class__(np.ones(self.shape, dtype=np.bool_))
            res = self._binopt(other, '_gt_' if op_name == '_le_' else '_lt_')
            return all_true - res
        else:
            # 返回未实现的操作
            return NotImplemented
    # 定义小于比较运算符的特殊方法，调用通用的比较函数进行处理
    def __lt__(self, other):
        return self._inequality(other, operator.lt, '_lt_',
                                "Comparing a sparse matrix with a scalar "
                                "greater than zero using < is inefficient, "
                                "try using >= instead.")

    # 定义大于比较运算符的特殊方法，调用通用的比较函数进行处理
    def __gt__(self, other):
        return self._inequality(other, operator.gt, '_gt_',
                                "Comparing a sparse matrix with a scalar "
                                "less than zero using > is inefficient, "
                                "try using <= instead.")

    # 定义小于等于比较运算符的特殊方法，调用通用的比较函数进行处理
    def __le__(self, other):
        return self._inequality(other, operator.le, '_le_',
                                "Comparing a sparse matrix with a scalar "
                                "greater than zero using <= is inefficient, "
                                "try using > instead.")

    # 定义大于等于比较运算符的特殊方法，调用通用的比较函数进行处理
    def __ge__(self, other):
        return self._inequality(other, operator.ge, '_ge_',
                                "Comparing a sparse matrix with a scalar "
                                "less than zero using >= is inefficient, "
                                "try using < instead.")

    #################################
    # Arithmetic operator overrides #
    #################################

    # 实现稀疏矩阵与密集矩阵相加的函数
    def _add_dense(self, other):
        # 检查矩阵形状是否兼容，如果不兼容则抛出异常
        if other.shape != self.shape:
            raise ValueError(f'Incompatible shapes ({self.shape} and {other.shape})')
        # 确定数据类型和存储顺序
        dtype = upcast_char(self.dtype.char, other.dtype.char)
        order = self._swap('CF')[0]
        # 复制密集矩阵的数据到新数组
        result = np.array(other, dtype=dtype, order=order, copy=True)
        # 根据稀疏矩阵的格式，将其转换为密集矩阵并加到结果中
        y = result if result.flags.c_contiguous else result.T
        M, N = self._swap(self._shape_as_2d)
        csr_todense(M, N, self.indptr, self.indices, self.data, y)
        # 返回结果作为特定容器的对象，确保不复制数据
        return self._container(result, copy=False)

    # 实现稀疏矩阵与稀疏矩阵相加的函数
    def _add_sparse(self, other):
        return self._binopt(other, '_plus_')

    # 实现稀疏矩阵与稀疏矩阵相减的函数
    def _sub_sparse(self, other):
        return self._binopt(other, '_minus_')

    ###########################
    # Multiplication handlers #
    ###########################

    # 实现稀疏矩阵与向量相乘的函数
    def _matmul_vector(self, other):
        M, N = self._shape_as_2d

        # 创建输出数组
        result = np.zeros(M, dtype=upcast_char(self.dtype.char, other.dtype.char))

        # 根据稀疏矩阵的格式调用相应的矩阵向量乘法函数
        fn = getattr(_sparsetools, self.format + '_matvec')
        fn(M, N, self.indptr, self.indices, self.data, other, result)

        # 如果稀疏矩阵是一维的，返回结果的第一个元素，否则返回整个结果数组
        return result[0] if self.ndim == 1 else result
    def _matmul_multivector(self, other):
        M, N = self._shape_as_2d
        n_vecs = other.shape[-1]  # 获取列向量的数量

        result = np.zeros((M, n_vecs),
                          dtype=upcast_char(self.dtype.char, other.dtype.char))

        # 根据稀疏矩阵的格式选择对应的乘法运算函数（csr_matvecs 或 csc_matvecs）
        fn = getattr(_sparsetools, self.format + '_matvecs')
        fn(M, N, n_vecs, self.indptr, self.indices, self.data,
           other.ravel(), result.ravel())

        if self.ndim == 1:
            return result.reshape((n_vecs,))
        return result

    def _matmul_sparse(self, other):
        M, K1 = self._shape_as_2d
        # 如果 other 是 1 维数组，则视为列向量处理
        o_ndim = other.ndim
        if o_ndim == 1:
            # 将 1 维数组转换为 2 维列向量（转换为 CSC 格式）
            other = other.reshape((1, other.shape[0])).T  # 注意：转换为 CSC 格式
        K2, N = other._shape if other.ndim == 2 else (other.shape[0], 1)

        # 确定新的形状: (M, N), (M,), (N,) 或 ()
        new_shape = ()
        if self.ndim == 2:
            new_shape += (M,)
        if o_ndim == 2:
            new_shape += (N,)

        # 获取主要维度，并将 other 转换为当前格式的稀疏矩阵
        major_dim = self._swap((M, N))[0]
        other = self.__class__(other)  # 转换为当前格式的稀疏矩阵

        # 获取索引类型，用于稀疏矩阵乘法
        idx_dtype = self._get_index_dtype((self.indptr, self.indices,
                                           other.indptr, other.indices))

        # 调用稀疏矩阵乘法的最大非零元素个数函数
        fn = getattr(_sparsetools, self.format + '_matmat_maxnnz')
        nnz = fn(M, N,
                 np.asarray(self.indptr, dtype=idx_dtype),
                 np.asarray(self.indices, dtype=idx_dtype),
                 np.asarray(other.indptr, dtype=idx_dtype),
                 np.asarray(other.indices, dtype=idx_dtype))
        if nnz == 0:
            if new_shape == ():
                return np.array(0, dtype=upcast(self.dtype, other.dtype))
            return self.__class__(new_shape, dtype=upcast(self.dtype, other.dtype))

        # 获取稀疏矩阵乘法的索引和数据
        idx_dtype = self._get_index_dtype((self.indptr, self.indices,
                                           other.indptr, other.indices),
                                          maxval=nnz)

        indptr = np.empty(major_dim + 1, dtype=idx_dtype)
        indices = np.empty(nnz, dtype=idx_dtype)
        data = np.empty(nnz, dtype=upcast(self.dtype, other.dtype))

        # 调用稀疏矩阵乘法函数
        fn = getattr(_sparsetools, self.format + '_matmat')
        fn(M, N, np.asarray(self.indptr, dtype=idx_dtype),
           np.asarray(self.indices, dtype=idx_dtype),
           self.data,
           np.asarray(other.indptr, dtype=idx_dtype),
           np.asarray(other.indices, dtype=idx_dtype),
           other.data,
           indptr, indices, data)

        # 根据新的形状返回结果
        if new_shape == ():
            return np.array(data[0])
        return self.__class__((data, indices, indptr), shape=new_shape)
    def diagonal(self, k=0):
        # 获取稀疏矩阵的行数和列数
        rows, cols = self.shape
        # 如果 k 超出了有效范围，返回一个空数组
        if k <= -rows or k >= cols:
            return np.empty(0, dtype=self.data.dtype)
        # 根据当前稀疏矩阵的格式调用相应的 C 扩展函数
        fn = getattr(_sparsetools, self.format + "_diagonal")
        # 创建一个空数组 y，用于存放对角线元素
        y = np.empty(min(rows + min(k, 0), cols - max(k, 0)),
                     dtype=upcast(self.dtype))
        # 调用 C 扩展函数计算对角线元素并存入数组 y
        fn(k, self.shape[0], self.shape[1], self.indptr, self.indices,
           self.data, y)
        # 返回计算得到的对角线数组 y
        return y

    # 设置 diagonal 方法的文档字符串为 _spbase.diagonal 的文档字符串
    diagonal.__doc__ = _spbase.diagonal.__doc__

    #####################
    # Other binary ops  #
    #####################

    def _maximum_minimum(self, other, npop, op_name, dense_check):
        # 如果 other 是标量类型
        if isscalarlike(other):
            # 如果应用操作后会导致稀疏矩阵变成稠密矩阵，则发出警告
            if dense_check(other):
                warn("Taking maximum (minimum) with > 0 (< 0) number results"
                     " to a dense matrix.", SparseEfficiencyWarning,
                     stacklevel=3)
                # 创建一个填充了标量值 other 的数组，并转换成相同的稀疏矩阵类型
                other_arr = np.empty(self.shape, dtype=np.asarray(other).dtype)
                other_arr.fill(other)
                other_arr = self.__class__(other_arr)
                # 调用 _binopt 方法执行对应的二元操作
                return self._binopt(other_arr, op_name)
            else:
                # 合并重复的非零元素
                self.sum_duplicates()
                # 执行对应的操作，并创建新的稀疏矩阵
                new_data = npop(self.data, np.asarray(other))
                mat = self.__class__((new_data, self.indices, self.indptr),
                                     dtype=new_data.dtype, shape=self.shape)
                return mat
        # 如果 other 是密集数组
        elif isdense(other):
            # 将稀疏矩阵转换成密集数组执行对应的操作
            return npop(self.todense(), other)
        # 如果 other 是稀疏矩阵
        elif issparse(other):
            # 调用 _binopt 方法执行对应的二元操作
            return self._binopt(other, op_name)
        else:
            # 如果操作数不兼容，则抛出 ValueError
            raise ValueError("Operands not compatible.")

    def maximum(self, other):
        # 调用 _maximum_minimum 方法执行 maximum 操作
        return self._maximum_minimum(other, np.maximum,
                                     '_maximum_', lambda x: np.asarray(x) > 0)

    # 设置 maximum 方法的文档字符串为 _spbase.maximum 的文档字符串
    maximum.__doc__ = _spbase.maximum.__doc__

    def minimum(self, other):
        # 调用 _maximum_minimum 方法执行 minimum 操作
        return self._maximum_minimum(other, np.minimum,
                                     '_minimum_', lambda x: np.asarray(x) < 0)

    # 设置 minimum 方法的文档字符串为 _spbase.minimum 的文档字符串
    minimum.__doc__ = _spbase.minimum.__doc__

    #####################
    # Reduce operations #
    #####################
    # 定义一个方法用于计算数组/矩阵沿指定轴的和。如果轴为 None，则对行和列求和，返回一个标量。
    def sum(self, axis=None, dtype=None, out=None):
        """Sum the array/matrix over the given axis.  If the axis is None, sum
        over both rows and columns, returning a scalar.
        """
        # 如果 self 的维度为 2，且没有 blocksize 属性，并且 axis 在 self._swap(((1, -1), (0, -2)))[0] 中
        if (self.ndim == 2 and not hasattr(self, 'blocksize') and
                axis in self._swap(((1, -1), (0, -2)))[0]):
            # 获取求和后结果的数据类型
            res_dtype = get_sum_dtype(self.dtype)
            # 创建一个 dtype 为 res_dtype 的全零数组，长度为 self.indptr 的长度减 1
            ret = np.zeros(len(self.indptr) - 1, dtype=res_dtype)

            # 调用 _minor_reduce 方法，使用 np.add 函数进行次要轴的归约操作
            major_index, value = self._minor_reduce(np.add)
            # 将归约后的值存入结果数组对应的位置
            ret[major_index] = value
            # 将结果数组转换为相应的容器类型
            ret = self._ascontainer(ret)
            # 如果 axis 是奇数，则对结果进行转置
            if axis % 2 == 1:
                ret = ret.T

            # 如果指定了输出数组 out，并且其形状与 ret 的形状不匹配，则引发 ValueError
            if out is not None and out.shape != ret.shape:
                raise ValueError('dimensions do not match')

            # 返回沿指定轴求和后的结果，指定 dtype 和 out 参数
            return ret.sum(axis=(), dtype=dtype, out=out)
        else:
            # 如果不满足上述条件，调用 _spbase.sum 方法进行求和操作，传递 axis、dtype 和 out 参数
            # _spbase 处理 axis 在 {None, -2, -1, 0, 1} 时的情况
            return _spbase.sum(self, axis=axis, dtype=dtype, out=out)

    # 将 sum 方法的文档字符串设置为 _spbase.sum 方法的文档字符串
    sum.__doc__ = _spbase.sum.__doc__

    # 定义一个方法用于在次要轴上使用 ufunc 函数对非零元素进行归约操作
    def _minor_reduce(self, ufunc, data=None):
        """Reduce nonzeros with a ufunc over the minor axis when non-empty

        Can be applied to a function of self.data by supplying data parameter.

        Warning: this does not call sum_duplicates()

        Returns
        -------
        major_index : array of ints
            Major indices where nonzero

        value : array of self.dtype
            Reduce result for nonzeros in each major_index
        """
        # 如果未提供 data 参数，则默认使用 self.data
        if data is None:
            data = self.data
        # 找到非零元素所在的主要索引
        major_index = np.flatnonzero(np.diff(self.indptr))
        # 使用 ufunc 函数对 data 在 self.indptr[major_index] 处进行归约操作
        value = ufunc.reduceat(data,
                               downcast_intp_index(self.indptr[major_index]))
        # 返回主要索引和归约后的值
        return major_index, value

    #######################
    # Getting and Setting #
    #######################

    # 定义一个方法用于获取整数索引对应位置的元素值
    def _get_intXint(self, row, col):
        M, N = self._swap(self.shape)
        major, minor = self._swap((row, col))
        # 获取 CSR 格式子矩阵的行指针、列索引和数据
        indptr, indices, data = get_csr_submatrix(
            M, N, self.indptr, self.indices, self.data,
            major, major + 1, minor, minor + 1)
        # 对获取的子矩阵数据进行求和，指定数据类型为 self.dtype
        return data.sum(dtype=self.dtype)

    # 定义一个方法用于获取切片索引对应位置的子矩阵
    def _get_sliceXslice(self, row, col):
        major, minor = self._swap((row, col))
        # 如果主要轴和次要轴的步长均为 1 或为 None，则调用 _get_submatrix 方法获取子矩阵副本
        if major.step in (1, None) and minor.step in (1, None):
            return self._get_submatrix(major, minor, copy=True)
        # 否则，先进行主要轴的切片操作，再进行次要轴的切片操作
        return self._major_slice(major)._minor_slice(minor)
    def _get_arrayXarray(self, row, col):
        # 获取稀疏矩阵中指定行列位置的元素
        idx_dtype = self.indices.dtype  # 获取稀疏矩阵索引数据类型
        M, N = self._swap(self.shape)  # 获取矩阵形状的转置值
        major, minor = self._swap((row, col))  # 获取主要和次要索引，进行转置
        major = np.asarray(major, dtype=idx_dtype)  # 将主要索引转换为指定的数据类型
        minor = np.asarray(minor, dtype=idx_dtype)  # 将次要索引转换为指定的数据类型

        val = np.empty(major.size, dtype=self.dtype)  # 创建一个空的数组来存储结果
        csr_sample_values(M, N, self.indptr, self.indices, self.data,
                          major.size, major.ravel(), minor.ravel(), val)  # 调用Cython函数获取稀疏矩阵中指定位置的元素值
        if major.ndim == 1:
            return self._ascontainer(val)  # 如果主要索引是一维的，则返回包装后的结果
        return self.__class__(val.reshape(major.shape))  # 否则，返回一个新的稀疏矩阵对象

    def _get_columnXarray(self, row, col):
        # 获取稀疏矩阵中指定列位置的元素
        major, minor = self._swap((row, col))  # 获取主要和次要索引，进行转置
        return self._major_index_fancy(major)._minor_index_fancy(minor)  # 调用方法获取指定列位置的元素

    def _major_index_fancy(self, idx):
        """Index along the major axis where idx is an array of ints.
        """
        idx_dtype = self._get_index_dtype((self.indptr, self.indices))  # 获取索引数据类型
        indices = np.asarray(idx, dtype=idx_dtype).ravel()  # 将输入的索引转换为指定的数据类型，并展平为一维数组

        N = self._swap(self._shape_as_2d)[1]  # 获取转置后的二维矩阵的列数
        M = len(indices)  # 获取索引数组的长度
        new_shape = self._swap((M, N)) if self.ndim == 2 else (M,)  # 根据稀疏矩阵的维度确定新的形状
        if M == 0:
            return self.__class__(new_shape, dtype=self.dtype)  # 如果索引数组长度为0，则返回一个空的稀疏矩阵对象

        row_nnz = (self.indptr[indices + 1] - self.indptr[indices]).astype(idx_dtype)  # 计算每行非零元素的数量
        res_indptr = np.zeros(M + 1, dtype=idx_dtype)  # 创建结果矩阵的行指针数组
        np.cumsum(row_nnz, out=res_indptr[1:])  # 计算累积和得到行指针数组

        nnz = res_indptr[-1]  # 获取结果矩阵的非零元素总数
        res_indices = np.empty(nnz, dtype=idx_dtype)  # 创建结果矩阵的列索引数组
        res_data = np.empty(nnz, dtype=self.dtype)  # 创建结果矩阵的数据数组
        csr_row_index(
            M,
            indices,
            self.indptr.astype(idx_dtype, copy=False),
            self.indices.astype(idx_dtype, copy=False),
            self.data,
            res_indices,
            res_data
        )  # 调用Cython函数获取稀疏矩阵中指定行索引的元素值

        return self.__class__((res_data, res_indices, res_indptr),
                              shape=new_shape, copy=False)  # 返回新的稀疏矩阵对象，包含结果数据、列索引和行指针
    # 定义一个方法 `_major_slice`，用于按照主轴索引，其中 idx 是一个切片对象
    def _major_slice(self, idx, copy=False):
        """Index along the major axis where idx is a slice object.
        """
        # 如果 idx 是全范围切片，则根据需要返回对象的副本或原对象本身
        if idx == slice(None):
            return self.copy() if copy else self

        # 获取矩阵的行列数，以及切片的起始、结束和步长
        M, N = self._swap(self._shape_as_2d)
        start, stop, step = idx.indices(M)
        # 计算切片后的行数 M
        M = len(range(start, stop, step))
        # 根据矩阵维度确定新的形状
        new_shape = self._swap((M, N)) if self.ndim == 2 else (M,)
        # 如果切片后行数为 0，则返回相应形状的空对象
        if M == 0:
            return self.__class__(new_shape, dtype=self.dtype)

        # 计算 `row_nnz` 需要的切片范围
        start0, stop0 = start, stop
        if stop == -1 and start >= 0:
            stop0 = None
        start1, stop1 = start + 1, stop + 1

        # 计算每行的非零元素个数，存储在 row_nnz 中
        row_nnz = self.indptr[start1:stop1:step] - \
            self.indptr[start0:stop0:step]
        # 确定索引的数据类型
        idx_dtype = self.indices.dtype
        # 创建结果数组的指针索引 res_indptr，长度为 M+1
        res_indptr = np.zeros(M+1, dtype=idx_dtype)
        # 计算累积非零元素个数，存储在 res_indptr 中
        np.cumsum(row_nnz, out=res_indptr[1:])

        # 根据步长判断是单步还是多步切片
        if step == 1:
            # 如果步长为 1，则直接进行切片操作获取所有索引和数据
            all_idx = slice(self.indptr[start], self.indptr[stop])
            res_indices = np.array(self.indices[all_idx], copy=copy)
            res_data = np.array(self.data[all_idx], copy=copy)
        else:
            # 如果步长不为 1，则根据 CSR 矩阵的行切片函数处理数据
            nnz = res_indptr[-1]
            res_indices = np.empty(nnz, dtype=idx_dtype)
            res_data = np.empty(nnz, dtype=self.dtype)
            csr_row_slice(start, stop, step, self.indptr, self.indices,
                          self.data, res_indices, res_data)

        # 返回切片后的新对象，形状为 new_shape，不进行复制
        return self.__class__((res_data, res_indices, res_indptr),
                              shape=new_shape, copy=False)
    def _minor_index_fancy(self, idx):
        """Index along the minor axis where idx is an array of ints.
        """
        # 获取适合索引数据类型，确保与self.indices和self.indptr兼容
        idx_dtype = self._get_index_dtype((self.indices, self.indptr))
        # 将self.indices转换为idx_dtype类型的数组，无需复制
        indices = self.indices.astype(idx_dtype, copy=False)
        # 将self.indptr转换为idx_dtype类型的数组，无需复制
        indptr = self.indptr.astype(idx_dtype, copy=False)

        # 将idx转换为idx_dtype类型的数组，并展平为一维数组
        idx = np.asarray(idx, dtype=idx_dtype).ravel()

        # 获取矩阵的形状(M, N)
        M, N = self._swap(self._shape_as_2d)
        # 确定idx的长度
        k = len(idx)
        # 根据矩阵的维度确定新的形状
        new_shape = self._swap((M, k)) if self.ndim == 2 else (k,)
        # 如果k为0，则返回一个新的空矩阵对象
        if k == 0:
            return self.__class__(new_shape, dtype=self.dtype)

        # pass 1: 计算idx条目数并计算新的indptr
        col_offsets = np.zeros(N, dtype=idx_dtype)
        res_indptr = np.empty_like(self.indptr, dtype=idx_dtype)
        # 调用Cython函数csr_column_index1，对指定的列进行索引
        csr_column_index1(
            k,
            idx,
            M,
            N,
            indptr,
            indices,
            col_offsets,
            res_indptr,
        )

        # pass 2: 复制选定idx的索引和数据
        # 将idx按照值的大小排序，并转换为idx_dtype类型的数组
        col_order = np.argsort(idx).astype(idx_dtype, copy=False)
        # 计算非零元素的数量
        nnz = res_indptr[-1]
        # 创建存储索引和数据的数组
        res_indices = np.empty(nnz, dtype=idx_dtype)
        res_data = np.empty(nnz, dtype=self.dtype)
        # 调用Cython函数csr_column_index2，复制选定的索引和数据
        csr_column_index2(col_order, col_offsets, len(self.indices),
                          indices, self.data, res_indices, res_data)
        # 返回新的矩阵对象，包含复制后的数据
        return self.__class__((res_data, res_indices, res_indptr),
                              shape=new_shape, copy=False)

    def _minor_slice(self, idx, copy=False):
        """Index along the minor axis where idx is a slice object.
        """
        # 如果idx是slice(None)，返回自身的副本或者本身（根据copy参数）
        if idx == slice(None):
            return self.copy() if copy else self

        # 获取矩阵的形状(M, N)
        M, N = self._swap(self._shape_as_2d)
        # 获取slice对象的起始、终止和步长
        start, stop, step = idx.indices(N)
        # 计算slice对象的长度N
        N = len(range(start, stop, step))
        # 如果长度为0，返回一个新的零矩阵对象
        if N == 0:
            return self.__class__(self._swap((M, N)), dtype=self.dtype)
        # 如果步长为1，调用_get_submatrix方法获取子矩阵
        if step == 1:
            return self._get_submatrix(minor=idx, copy=copy)
        # 否则，调用_minor_index_fancy方法进行索引
        # TODO: 在此处避免使用fancy索引
        return self._minor_index_fancy(np.arange(start, stop, step))

    def _get_submatrix(self, major=None, minor=None, copy=False):
        """Return a submatrix of this matrix.

        major, minor: None, int, or slice with step 1
        """
        # 获取矩阵的形状(M, N)
        M, N = self._swap(self._shape_as_2d)
        # 处理主轴和次轴的slice对象，返回起始和终止索引
        i0, i1 = _process_slice(major, M)
        j0, j1 = _process_slice(minor, N)

        # 如果切片从0到M和0到N，返回自身的副本或者本身（根据copy参数）
        if i0 == 0 and j0 == 0 and i1 == M and j1 == N:
            return self.copy() if copy else self

        # 调用Cython函数get_csr_submatrix，获取CSR格式的子矩阵的indptr、indices和data
        indptr, indices, data = get_csr_submatrix(
            M, N, self.indptr, self.indices, self.data, i0, i1, j0, j1)

        # 计算子矩阵的形状，并根据矩阵的维度进行调整
        shape = self._swap((i1 - i0, j1 - j0))
        if self.ndim == 1:
            shape = (shape[1],)
        # 返回包含子矩阵数据的新矩阵对象
        return self.__class__((data, indices, indptr), shape=shape,
                              dtype=self.dtype, copy=False)

    def _set_intXint(self, row, col, x):
        # 交换行列索引，确保与内部表示方式一致
        i, j = self._swap((row, col))
        # 调用_set_many方法，设置(i, j)位置上的值为x
        self._set_many(i, j, x)
    # 将稀疏矩阵的部分元素设置为给定值
    def _set_arrayXarray(self, row, col, x):
        # 根据给定的行和列索引，确定实际的行列位置
        i, j = self._swap((row, col))
        # 调用 _set_many 方法设置多个元素的数值
        self._set_many(i, j, x)

    # 将稀疏矩阵的部分元素设置为给定值（对稀疏矩阵特别优化）
    def _set_arrayXarray_sparse(self, row, col, x):
        # 清除将要被覆盖的现有条目
        self._zero_many(*self._swap((row, col)))

        # 获取行列的维度信息
        M, N = row.shape  # 与 col.shape 匹配
        # 判断是否需要进行行或列广播
        broadcast_row = M != 1 and x.shape[0] == 1
        broadcast_col = N != 1 and x.shape[1] == 1
        # 获取稀疏矩阵 x 的行和列索引
        r, c = x.row, x.col

        # 将 x 的数据转换为特定数据类型的 NumPy 数组
        x = np.asarray(x.data, dtype=self.dtype)
        # 如果 x 是空的，则直接返回
        if x.size == 0:
            return

        # 如果需要行广播，对行索引 r 进行重复和扩展
        if broadcast_row:
            r = np.repeat(np.arange(M), len(r))
            c = np.tile(c, M)
            x = np.tile(x, M)
        # 如果需要列广播，对列索引 c 进行重复和扩展
        if broadcast_col:
            r = np.repeat(r, N)
            c = np.tile(np.arange(N), len(c))
            x = np.repeat(x, N)
        
        # 只在新的稀疏结构中分配条目
        # 根据重排后的行列索引 i, j，调用 _set_many 方法设置多个元素的数值
        i, j = self._swap((row[r, c], col[r, c]))
        self._set_many(i, j, x)
    # 如果数组的形状中有任何一个维度为0，则直接返回，不做任何操作
    def _setdiag(self, values, k):
        # 如果数组是一维的，则无法设置对角线元素，抛出未实现错误
        if self.ndim == 1:
            raise NotImplementedError('diagonals cant be set in 1d arrays')

        # 获取数组的形状
        M, N = self.shape
        # 判断是否需要广播输入的值
        broadcast = (values.ndim == 0)

        # 根据 k 的值确定对角线的位置
        if k < 0:
            # 计算对角线的起始索引和结束索引
            if broadcast:
                max_index = min(M + k, N)
            else:
                max_index = min(M + k, N, len(values))
            i = np.arange(-k, max_index - k, dtype=self.indices.dtype)
            j = np.arange(max_index, dtype=self.indices.dtype)

        else:
            # 计算对角线的起始索引和结束索引
            if broadcast:
                max_index = min(M, N - k)
            else:
                max_index = min(M, N - k, len(values))
            i = np.arange(max_index, dtype=self.indices.dtype)
            j = np.arange(k, k + max_index, dtype=self.indices.dtype)

        # 如果输入的值不是广播形式，则截取相应长度的值
        if not broadcast:
            values = values[:len(i)]

        # 将输入的值转换为至少为一维的 numpy 数组，并将其展平为一维数组
        x = np.atleast_1d(np.asarray(values, dtype=self.dtype)).ravel()
        # 如果展平后的 x 和 i 的形状不匹配，则将 x 广播到与 i 相同的形状
        if x.squeeze().shape != i.squeeze().shape:
            x = np.broadcast_to(x, i.shape)
        # 如果 x 的大小为0，则直接返回，不做任何操作
        if x.size == 0:
            return

        # 将 M 和 N 互换，获取行压缩稀疏矩阵的偏移量
        M, N = self._swap((M, N))
        i, j = self._swap((i, j))
        n_samples = x.size
        offsets = np.empty(n_samples, dtype=self.indices.dtype)
        # 调用 csr_sample_offsets 函数，计算偏移量并返回结果
        ret = csr_sample_offsets(M, N, self.indptr, self.indices, n_samples,
                                 i, j, offsets)
        # 如果返回值为1，则执行去重操作后再次调用 csr_sample_offsets 函数
        if ret == 1:
            self.sum_duplicates()
            csr_sample_offsets(M, N, self.indptr, self.indices, n_samples,
                               i, j, offsets)
        # 如果偏移量数组中不存在 -1，则将 x 的值赋给对应偏移量位置的数组元素
        if -1 not in offsets:
            self.data[offsets] = x
            return

        # 创建一个布尔掩码，标记偏移量小于等于 -1 的位置
        mask = (offsets <= -1)
        # 如果掩码的总和小于非零元素的数量乘以 0.001，则执行以下操作
        if mask.sum() < self.nnz * 0.001:
            # 创建新的条目，将新的 i、j 和 x[mask] 插入数组中
            i = i[mask]
            j = j[mask]
            self._insert_many(i, j, x[mask])
            # 替换已有条目，更新偏移量中非 mask 部分的数据
            mask = ~mask
            self.data[offsets[mask]] = x[mask]
        else:
            # 将稀疏矩阵转换为 COO 格式，并调用其 _setdiag 方法设置对角线元素
            coo = self.tocoo()
            coo._setdiag(values, k)
            # 将 COO 格式转换为压缩格式，更新 self 的各个属性
            arrays = coo._coo_to_compressed(self._swap)
            self.indptr, self.indices, self.data, _ = arrays
    def _prepare_indices(self, i, j):
        # 获取矩阵的行数 M 和列数 N
        M, N = self._swap(self._shape_as_2d)

        def check_bounds(indices, bound):
            # 检查索引是否在合法范围内
            idx = indices.max()
            if idx >= bound:
                raise IndexError('index (%d) out of range (>= %d)' %
                                 (idx, bound))
            idx = indices.min()
            if idx < -bound:
                raise IndexError('index (%d) out of range (< -%d)' %
                                 (idx, bound))

        # 将输入的 i 和 j 转换为至少一维的 NumPy 数组，并展平
        i = np.atleast_1d(np.asarray(i, dtype=self.indices.dtype)).ravel()
        j = np.atleast_1d(np.asarray(j, dtype=self.indices.dtype)).ravel()
        # 检查 i 和 j 的边界
        check_bounds(i, M)
        check_bounds(j, N)
        # 返回处理后的 i, j, M, N
        return i, j, M, N

    def _set_many(self, i, j, x):
        """Sets value at each (i, j) to x

        Here (i,j) index major and minor respectively, and must not contain
        duplicate entries.
        """
        # 准备索引 i, j, M, N
        i, j, M, N = self._prepare_indices(i, j)
        # 将 x 转换为至少一维的 NumPy 数组，并展平
        x = np.atleast_1d(np.asarray(x, dtype=self.dtype)).ravel()

        # 获取样本数目
        n_samples = x.size
        # 创建偏移量数组
        offsets = np.empty(n_samples, dtype=self.indices.dtype)
        # 调用 csr_sample_offsets 函数进行采样偏移量计算
        ret = csr_sample_offsets(M, N, self.indptr, self.indices, n_samples,
                                 i, j, offsets)
        if ret == 1:
            # 如果返回值为 1，调用 self.sum_duplicates() 函数处理重复值
            self.sum_duplicates()
            # 再次调用 csr_sample_offsets 获取偏移量
            csr_sample_offsets(M, N, self.indptr, self.indices, n_samples,
                               i, j, offsets)

        if -1 not in offsets:
            # 如果偏移量中没有 -1，表示只影响现有的非零元素
            # 将 x 的值赋给这些位置
            self.data[offsets] = x
            return

        else:
            # 如果偏移量中有 -1，表示会改变稀疏结构
            warn(f"Changing the sparsity structure of a {self.__class__.__name__} is"
                 " expensive. lil and dok are more efficient.",
                 SparseEfficiencyWarning, stacklevel=3)
            # 替换可能的位置
            mask = offsets > -1
            self.data[offsets[mask]] = x[mask]
            # 只保留插入的元素
            mask = ~mask
            i = i[mask]
            i[i < 0] += M
            j = j[mask]
            j[j < 0] += N
            self._insert_many(i, j, x[mask])

    def _zero_many(self, i, j):
        """Sets value at each (i, j) to zero, preserving sparsity structure.

        Here (i,j) index major and minor respectively.
        """
        # 准备索引 i, j, M, N
        i, j, M, N = self._prepare_indices(i, j)

        # 获取样本数目
        n_samples = len(i)
        # 创建偏移量数组
        offsets = np.empty(n_samples, dtype=self.indices.dtype)
        # 调用 csr_sample_offsets 函数进行采样偏移量计算
        ret = csr_sample_offsets(M, N, self.indptr, self.indices, n_samples,
                                 i, j, offsets)
        if ret == 1:
            # 如果返回值为 1，调用 self.sum_duplicates() 函数处理重复值
            self.sum_duplicates()
            # 再次调用 csr_sample_offsets 获取偏移量
            csr_sample_offsets(M, N, self.indptr, self.indices, n_samples,
                               i, j, offsets)

        # 仅将零赋给现有的稀疏结构位置
        self.data[offsets[offsets > -1]] = 0
    def _insert_many(self, i, j, x):
        """Inserts new nonzero at each (i, j) with value x

        Here (i,j) index major and minor respectively.
        i, j and x must be non-empty, 1d arrays.
        Inserts each major group (e.g. all entries per row) at a time.
        Maintains has_sorted_indices property.
        Modifies i, j, x in place.
        """
        # 对 i 进行稳定排序，以确保稳定性处理重复值
        order = np.argsort(i, kind='mergesort')  # stable for duplicates
        i = i.take(order, mode='clip')  # 按排序顺序重新排列 i
        j = j.take(order, mode='clip')  # 按排序顺序重新排列 j
        x = x.take(order, mode='clip')  # 按排序顺序重新排列 x

        do_sort = self.has_sorted_indices

        # 更新索引数据类型
        idx_dtype = self._get_index_dtype((self.indices, self.indptr),
                                          maxval=(self.indptr[-1] + x.size))
        self.indptr = np.asarray(self.indptr, dtype=idx_dtype)
        self.indices = np.asarray(self.indices, dtype=idx_dtype)
        i = np.asarray(i, dtype=idx_dtype)
        j = np.asarray(j, dtype=idx_dtype)

        # 按主索引分块整合旧数据和新数据
        indices_parts = []
        data_parts = []
        ui, ui_indptr = np.unique(i, return_index=True)
        ui_indptr = np.append(ui_indptr, len(j))
        new_nnzs = np.diff(ui_indptr)
        prev = 0
        for c, (ii, js, je) in enumerate(zip(ui, ui_indptr, ui_indptr[1:])):
            # 旧条目
            start = self.indptr[prev]
            stop = self.indptr[ii]
            indices_parts.append(self.indices[start:stop])
            data_parts.append(self.data[start:stop])

            # 处理重复的 j：保留最后的设置
            uj, uj_indptr = np.unique(j[js:je][::-1], return_index=True)
            if len(uj) == je - js:
                indices_parts.append(j[js:je])
                data_parts.append(x[js:je])
            else:
                indices_parts.append(j[js:je][::-1][uj_indptr])
                data_parts.append(x[js:je][::-1][uj_indptr])
                new_nnzs[c] = len(uj)

            prev = ii

        # 剩余的旧条目
        start = self.indptr[ii]
        indices_parts.append(self.indices[start:])
        data_parts.append(self.data[start:])

        # 更新属性
        self.indices = np.concatenate(indices_parts)
        self.data = np.concatenate(data_parts)
        nnzs = np.empty(self.indptr.shape, dtype=idx_dtype)
        nnzs[0] = idx_dtype(0)
        indptr_diff = np.diff(self.indptr)
        indptr_diff[ui] += new_nnzs
        nnzs[1:] = indptr_diff
        self.indptr = np.cumsum(nnzs, out=nnzs)

        if do_sort:
            # TODO: 只在必要时排序
            self.has_sorted_indices = False
            self.sort_indices()

        self.check_format(full_check=False)

    ######################
    # Conversion methods #
    ######################
    ##############################################################
    # methods that examine or modify the internal data structure #
    ##############################################################

    def eliminate_zeros(self):
        """Remove zero entries from the array/matrix

        This is an *in place* operation.
        """
        # 获取矩阵维度
        M, N = self._swap(self._shape_as_2d)
        # 调用 C 扩展函数，从 CSR 格式中删除零元素
        _sparsetools.csr_eliminate_zeros(M, N, self.indptr, self.indices, self.data)
        # 调整数据结构以反映零元素的删除
        self.prune()  # nnz may have changed

    @property
    def has_canonical_format(self) -> bool:
        """Whether the array/matrix has sorted indices and no duplicates

        Returns
            - True: if the above applies
            - False: otherwise

        has_canonical_format implies has_sorted_indices, so if the latter flag
        is False, so will the former be; if the former is found True, the
        latter flag is also set.
        """
        # 首先检查是否缓存了结果
        if not getattr(self, '_has_sorted_indices', True):
            # 如果索引未排序，则不符合规范格式
            self._has_canonical_format = False
        elif not hasattr(self, '_has_canonical_format'):
            # 调用 C 扩展函数，检查 CSR 矩阵是否具有规范格式
            self.has_canonical_format = bool(
                _sparsetools.csr_has_canonical_format(
                    len(self.indptr) - 1, self.indptr, self.indices)
                )
        return self._has_canonical_format

    @has_canonical_format.setter
    def has_canonical_format(self, val: bool):
        # 设置是否具有规范格式的标志
        self._has_canonical_format = bool(val)
        if val:
            # 如果具有规范格式，则也应设置索引已排序的标志
            self.has_sorted_indices = True
    def sum_duplicates(self):
        """
        消除重复条目，将它们相加合并

        这是一个原地操作。
        """
        # 如果已经是规范格式，则直接返回
        if self.has_canonical_format:
            return
        # 对索引进行排序
        self.sort_indices()

        # 获取矩阵的行数和列数
        M, N = self._swap(self._shape_as_2d)
        # 调用底层函数，将重复的条目相加
        _sparsetools.csr_sum_duplicates(M, N, self.indptr, self.indices, self.data)

        # 剪枝操作，因为 nnz 可能已经改变
        self.prune()
        # 设置为已经是规范格式
        self.has_canonical_format = True

    @property
    def has_sorted_indices(self) -> bool:
        """
        检查索引是否已排序

        Returns:
            - True: 如果数组/矩阵的索引按排序顺序排列
            - False: 否则
        """
        # 首先检查缓存中是否有结果
        if not hasattr(self, '_has_sorted_indices'):
            self._has_sorted_indices = bool(
                _sparsetools.csr_has_sorted_indices(
                    len(self.indptr) - 1, self.indptr, self.indices)
                )
        return self._has_sorted_indices

    @has_sorted_indices.setter
    def has_sorted_indices(self, val: bool):
        # 设置索引排序的状态
        self._has_sorted_indices = bool(val)


    def sorted_indices(self):
        """
        返回一个索引已排序的数组/矩阵的副本
        """
        # 复制当前对象
        A = self.copy()
        # 对副本进行索引排序
        A.sort_indices()
        return A

        # 一个具有线性复杂度的替代方法如下
        # 尽管前一个选项通常更快
        # return self.toother().toother()

    def sort_indices(self):
        """
        原地对数组/矩阵的索引进行排序
        """

        # 如果索引尚未排序，则执行排序操作
        if not self.has_sorted_indices:
            _sparsetools.csr_sort_indices(len(self.indptr) - 1, self.indptr,
                                          self.indices, self.data)
            # 标记为已排序
            self.has_sorted_indices = True

    def prune(self):
        """
        移除所有非零元素后的空间
        """
        # 获取主维度（行数）
        major_dim = self._swap(self._shape_as_2d)[0]

        # 检查索引指针的长度是否有效
        if len(self.indptr) != major_dim + 1:
            raise ValueError('index pointer has invalid length')
        # 检查索引数组是否少于 nnz 元素
        if len(self.indices) < self.nnz:
            raise ValueError('indices array has fewer than nnz elements')
        # 检查数据数组是否少于 nnz 元素
        if len(self.data) < self.nnz:
            raise ValueError('data array has fewer than nnz elements')

        # 对索引数组进行剪枝，保留 nnz 长度的部分
        self.indices = _prune_array(self.indices[:self.nnz])
        # 对数据数组进行剪枝，保留 nnz 长度的部分
        self.data = _prune_array(self.data[:self.nnz])
    def resize(self, *shape):
        # 使用 check_shape 函数验证并规范化形状，允许一维数组如果是 sparray 类型
        shape = check_shape(shape, allow_1d=isinstance(self, sparray))

        # 如果对象有 blocksize 属性，则执行以下操作
        if hasattr(self, 'blocksize'):
            bm, bn = self.blocksize
            # 计算新的行数和列数，以及余数
            new_M, rm = divmod(shape[0], bm)
            new_N, rn = divmod(shape[1], bn)
            # 如果余数不为零，则抛出 ValueError 异常
            if rm or rn:
                raise ValueError(f"shape must be divisible into {self.blocksize}"
                                 f" blocks. Got {shape}")
            # 计算当前的块数 M 和 N
            M, N = self.shape[0] // bm, self.shape[1] // bn
        else:
            # 如果没有 blocksize 属性，根据 shape 的维度进行调整
            new_M, new_N = self._swap(shape if len(shape)>1 else (1, shape[0]))
            M, N = self._swap(self._shape_as_2d)

        # 如果新的行数 new_M 小于当前的 M
        if new_M < M:
            # 调整 indices, data 和 indptr，截断多余的部分
            self.indices = self.indices[:self.indptr[new_M]]
            self.data = self.data[:self.indptr[new_M]]
            self.indptr = self.indptr[:new_M + 1]
        # 如果新的行数 new_M 大于当前的 M
        elif new_M > M:
            # 调整 indptr，以适应新的行数，并填充最后一部分
            self.indptr = np.resize(self.indptr, new_M + 1)
            self.indptr[M + 1:].fill(self.indptr[M])

        # 如果新的列数 new_N 小于当前的 N
        if new_N < N:
            # 创建一个掩码，用于筛选出 indices 小于 new_N 的部分
            mask = self.indices < new_N
            # 如果不是所有的 indices 都满足条件
            if not np.all(mask):
                # 根据掩码调整 indices 和 data
                self.indices = self.indices[mask]
                self.data = self.data[mask]
                # 使用 _minor_reduce 方法进行次要的减少操作，并更新 indptr
                major_index, val = self._minor_reduce(np.add, mask)
                self.indptr.fill(0)
                self.indptr[1:][major_index] = val
                np.cumsum(self.indptr, out=self.indptr)

        # 更新对象的形状属性
        self._shape = shape

    # 将 resize 方法的文档字符串设置为 _spbase.resize 方法的文档字符串
    resize.__doc__ = _spbase.resize.__doc__

    ###################
    # utility methods #
    ###################

    # needed by _data_matrix
    # 返回一个具有与 self 相同稀疏结构但数据不同的矩阵
    def _with_data(self, data, copy=True):
        """Returns a matrix with the same sparsity structure as self,
        but with different data.  By default the structure arrays
        (i.e. .indptr and .indices) are copied.
        """
        if copy:
            # 如果 copy=True，复制结构数组（indptr 和 indices），返回新矩阵对象
            return self.__class__((data, self.indices.copy(),
                                   self.indptr.copy()),
                                  shape=self.shape,
                                  dtype=data.dtype)
        else:
            # 如果 copy=False，直接使用当前的 indptr 和 indices，返回新矩阵对象
            return self.__class__((data, self.indices, self.indptr),
                                  shape=self.shape, dtype=data.dtype)
    def _binopt(self, other, op):
        """apply the binary operation fn to two sparse matrices."""
        # 创建一个当前类的实例，以确保other是与self相同类型的稀疏矩阵
        other = self.__class__(other)

        # 获取稀疏矩阵工具库_sparsetools中对应格式的操作函数，例如csr_plus_csr、csr_minus_csr等
        fn = getattr(_sparsetools, self.format + op + self.format)

        # 计算两个稀疏矩阵的非零元素总数的最大可能值
        maxnnz = self.nnz + other.nnz
        # 确定索引数据类型，确保适当存储所有可能的索引
        idx_dtype = self._get_index_dtype((self.indptr, self.indices,
                                           other.indptr, other.indices),
                                          maxval=maxnnz)
        # 创建空的索引指针数组和索引数组
        indptr = np.empty(self.indptr.shape, dtype=idx_dtype)
        indices = np.empty(maxnnz, dtype=idx_dtype)

        # 根据操作类型，确定数据数组的数据类型
        bool_ops = ['_ne_', '_lt_', '_gt_', '_le_', '_ge_']
        if op in bool_ops:
            data = np.empty(maxnnz, dtype=np.bool_)
        else:
            data = np.empty(maxnnz, dtype=upcast(self.dtype, other.dtype))

        # 获取self矩阵的形状并调整为2维形式
        M, N = self._shape_as_2d
        # 调用稀疏矩阵工具库中的函数fn，执行二进制操作
        fn(M, N,
           np.asarray(self.indptr, dtype=idx_dtype),
           np.asarray(self.indices, dtype=idx_dtype),
           self.data,
           np.asarray(other.indptr, dtype=idx_dtype),
           np.asarray(other.indices, dtype=idx_dtype),
           other.data,
           indptr, indices, data)

        # 根据操作结果创建新的稀疏矩阵A，并进行修剪处理
        A = self.__class__((data, indices, indptr), shape=self.shape)
        A.prune()

        # 返回处理后的稀疏矩阵A
        return A

    def _divide_sparse(self, other):
        """
        Divide this matrix by a second sparse matrix.
        """
        # 检查要除的稀疏矩阵是否与当前矩阵具有相同形状
        if other.shape != self.shape:
            raise ValueError('inconsistent shapes')

        # 调用_binopt方法执行稀疏矩阵的逐元素除法操作
        r = self._binopt(other, '_eldiv_')

        # 如果结果r的数据类型是浮点数类型
        if np.issubdtype(r.dtype, np.inexact):
            # 对于浮点数类型，除法操作可能会导致在稀疏模式之外的条目为空，因此需要手动填充这些条目
            # 所有在other稀疏矩阵稀疏模式之外的位置被填充为NaN，而在其内部的位置要么是零，要么由eldiv定义
            out = np.empty(self.shape, dtype=self.dtype)
            out.fill(np.nan)
            coords = other.nonzero()
            if self.ndim == 1:
                coords = (coords[-1],)
            out[coords] = 0
            # 将结果矩阵r转换为COO格式，并将其数据写入out数组中
            r = r.tocoo()
            out[r.coords] = r.data
            # 返回处理后的结果矩阵out，封装到self的容器中
            return self._container(out)
        else:
            # 对于整数类型，结果直接返回
            out = r
            return out
# 构建对角线稀疏矩阵的 CSR（压缩稀疏行）表示，存储在 self._csr_container 中

def _make_diagonal_csr(data, is_array=False):
    """build diagonal csc_array/csr_array => self._csr_container

    Parameter `data` should be a raveled numpy array holding the
    values on the diagonal of the resulting sparse matrix.
    """
    from ._csr import csr_array, csr_matrix
    # 如果 is_array 为 False，使用 csr_matrix；否则使用 csr_array
    csr_array = csr_array if is_array else csr_matrix

    N = len(data)  # 获取数据数组的长度 N
    indptr = np.arange(N + 1)  # 创建一个长度为 N+1 的递增数组作为 indptr（指针）
    indices = indptr[:-1]  # indices 是 indptr 的前 N 项

    return csr_array((data, indices, indptr), shape=(N, N))  # 返回使用 data, indices, indptr 构建的 CSR 矩阵


def _process_slice(sl, num):
    # 处理切片 sl 和总数 num 的情况，返回有效的起始和结束索引

    if sl is None:
        i0, i1 = 0, num  # 如果 sl 为 None，则默认从 0 到 num
    elif isinstance(sl, slice):
        i0, i1, stride = sl.indices(num)  # 获取切片的起始、结束和步长
        if stride != 1:
            raise ValueError('slicing with step != 1 not supported')  # 如果步长不为 1，抛出异常
        i0 = min(i0, i1)  # 确保 i0 <= i1，当 i0 > i1 时返回空切片
    elif isintlike(sl):
        if sl < 0:
            sl += num  # 如果 sl 为负数，将其转换为正数索引
        i0, i1 = sl, sl + 1  # 如果 sl 是整数，计算起始和结束索引
        if i0 < 0 or i1 > num:
            raise IndexError(f'index out of bounds: 0 <= {i0} < {i1} <= {num}')  # 如果索引超出范围，抛出 IndexError
    else:
        raise TypeError('expected slice or scalar')  # 如果 sl 类型既不是 slice 也不是整数，抛出 TypeError

    return i0, i1  # 返回有效的起始和结束索引
```