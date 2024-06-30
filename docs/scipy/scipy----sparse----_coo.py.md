# `D:\src\scipysrc\scipy\scipy\sparse\_coo.py`

```
`
""" A sparse matrix in COOrdinate or 'triplet' format"""

__docformat__ = "restructuredtext en"

__all__ = ['coo_array', 'coo_matrix', 'isspmatrix_coo']

import math
from warnings import warn

import numpy as np

from .._lib._util import copy_if_needed
from ._matrix import spmatrix
from ._sparsetools import coo_tocsr, coo_todense, coo_matvec
from ._base import issparse, SparseEfficiencyWarning, _spbase, sparray
from ._data import _data_matrix, _minmax_mixin
from ._sputils import (upcast_char, to_native, isshape, getdtype,
                       getdata, downcast_intp_index, get_index_dtype,
                       check_shape, check_reshape_kwargs)

import operator


class _coo_base(_data_matrix, _minmax_mixin):
    _format = 'coo'

    @property
    def row(self):
        # 如果数组维度大于1，则返回行坐标，否则创建和列坐标相同形状的零数组并返回
        if self.ndim > 1:
            return self.coords[-2]
        result = np.zeros_like(self.col)
        result.setflags(write=False)
        return result


    @row.setter
    def row(self, new_row):
        # 如果数组维度小于2，则无法设置行属性
        if self.ndim < 2:
            raise ValueError('cannot set row attribute of a 1-dimensional sparse array')
        # 将新的行坐标转换为与当前行坐标相同的数据类型，并更新坐标信息
        new_row = np.asarray(new_row, dtype=self.coords[-2].dtype)
        self.coords = self.coords[:-2] + (new_row,) + self.coords[-1:]

    @property
    def col(self):
        # 返回列坐标
        return self.coords[-1]

    @col.setter
    def col(self, new_col):
        # 将新的列坐标转换为与当前列坐标相同的数据类型，并更新坐标信息
        new_col = np.asarray(new_col, dtype=self.coords[-1].dtype)
        self.coords = self.coords[:-1] + (new_col,)

    def reshape(self, *args, **kwargs):
        is_array = isinstance(self, sparray)
        # 检查并确定新形状是否符合要求
        shape = check_shape(args, self.shape, allow_1d=is_array)
        order, copy = check_reshape_kwargs(kwargs)

        # 如果不需要 reshape 则直接返回自身或其副本
        if shape == self.shape:
            if copy:
                return self.copy()
            else:
                return self

        # 当减少维度时，需要特别注意索引溢出问题，这是为什么不能简单调用
        # `np.ravel_multi_index()` 后跟 `np.unravel_index()` 的原因
        flat_coords = _ravel_coords(self.coords, self.shape, order=order)
        if len(shape) == 2:
            if order == 'C':
                new_coords = divmod(flat_coords, shape[1])
            else:
                new_coords = divmod(flat_coords, shape[0])[::-1]
        else:
            new_coords = np.unravel_index(flat_coords, shape, order=order)

        # 根据需要处理数据的复制
        if copy:
            new_data = self.data.copy()
        else:
            new_data = self.data

        # 返回一个新的 COO 矩阵实例，保留原始数据而不进行复制
        return self.__class__((new_data, new_coords), shape=shape, copy=False)

    reshape.__doc__ = _spbase.reshape.__doc__
    `
        # 返回稀疏矩阵在指定轴上的非零元素数量
        def _getnnz(self, axis=None):
            # 如果未指定轴或者指定轴为0且矩阵维度为1，则返回数据数组的长度
            if axis is None or (axis == 0 and self.ndim == 1):
                nnz = len(self.data)
                # 检查索引数组的长度与数据数组长度是否一致
                if any(len(idx) != nnz for idx in self.coords):
                    raise ValueError('all index and data arrays must have the '
                                     'same length')
    
                # 检查数据数组和所有索引数组是否都是一维数组
                if self.data.ndim != 1 or any(idx.ndim != 1 for idx in self.coords):
                    raise ValueError('row, column, and data arrays must be 1-D')
    
                return int(nnz)
    
            # 如果轴为负数，则转换为对应正数轴
            if axis < 0:
                axis += self.ndim
            # 检查轴是否超出矩阵维度范围
            if axis >= self.ndim:
                raise ValueError('axis out of bounds')
            # 对于超过2维的COO数组，暂不支持按轴计算非零元素数量
            if self.ndim > 2:
                raise NotImplementedError('per-axis nnz for COO arrays with >2 '
                                          'dimensions is not supported')
            # 使用np.bincount统计非零元素在指定轴上的分布
            return np.bincount(downcast_intp_index(self.coords[1 - axis]),
                               minlength=self.shape[1 - axis])
    
        _getnnz.__doc__ = _spbase._getnnz.__doc__
    
        # 返回稀疏矩阵在指定轴上的非零元素数量
        def count_nonzero(self, axis=None):
            # 确保矩阵数据结构中没有重复元素
            self.sum_duplicates()
            # 如果未指定轴，则返回数据数组中非零元素的数量
            if axis is None:
                return np.count_nonzero(self.data)
    
            # 如果轴为负数，则转换为对应正数轴
            if axis < 0:
                axis += self.ndim
            # 检查轴是否超出矩阵维度范围
            if axis < 0 or axis >= self.ndim:
                raise ValueError('axis out of bounds')
            # 使用掩码获取数据数组中非零元素对应的索引，并按轴计算其分布
            mask = self.data != 0
            coord = self.coords[1 - axis][mask]
            return np.bincount(downcast_intp_index(coord), minlength=self.shape[1 - axis])
    
        count_nonzero.__doc__ = _spbase.count_nonzero.__doc__
    
        # 检查稀疏矩阵数据结构的一致性
        def _check(self):
            """ Checks data structure for consistency """
            # 检查索引数组数量是否与矩阵维度一致
            if self.ndim != len(self.coords):
                raise ValueError('mismatching number of index arrays for shape; '
                                 f'got {len(self.coords)}, expected {self.ndim}')
    
            # 索引数组应该具有整数数据类型
            for i, idx in enumerate(self.coords):
                if idx.dtype.kind != 'i':
                    warn(f'index array {i} has non-integer dtype ({idx.dtype.name})',
                         stacklevel=3)
    
            # 确定索引数组的数据类型，并将其转换为相应的类型
            idx_dtype = self._get_index_dtype(self.coords, maxval=max(self.shape))
            self.coords = tuple(np.asarray(idx, dtype=idx_dtype)
                                 for idx in self.coords)
            # 将数据数组转换为本机数据类型
            self.data = to_native(self.data)
    
            # 如果矩阵中存在非零元素，则进一步检查索引范围是否合法
            if self.nnz > 0:
                for i, idx in enumerate(self.coords):
                    if idx.max() >= self.shape[i]:
                        raise ValueError(f'axis {i} index {idx.max()} exceeds '
                                         f'matrix dimension {self.shape[i]}')
                    if idx.min() < 0:
                        raise ValueError(f'negative axis {i} index: {idx.min()}')
    def transpose(self, axes=None, copy=False):
        # 如果未指定轴，则默认为逆序的当前维度范围
        if axes is None:
            axes = range(self.ndim)[::-1]
        # 如果是稀疏矩阵，并且指定了轴
        elif isinstance(self, sparray):
            # 检查轴数与矩阵维度是否匹配
            if len(axes) != self.ndim:
                raise ValueError("axes don't match matrix dimensions")
            # 检查轴是否有重复
            if len(set(axes)) != self.ndim:
                raise ValueError("repeated axis in transpose")
        # 对于稀疏矩阵，不支持 'axes' 参数，因为交换维度是唯一合乎逻辑的置换
        elif axes != (1, 0):
            raise ValueError("Sparse matrices do not support an 'axes' "
                             "parameter because swapping dimensions is the "
                             "only logical permutation.")

        # 计算置换后的形状
        permuted_shape = tuple(self._shape[i] for i in axes)
        # 计算置换后的坐标
        permuted_coords = tuple(self.coords[i] for i in axes)
        # 返回一个新的类实例，其中数据、坐标和形状都进行了置换
        return self.__class__((self.data, permuted_coords),
                              shape=permuted_shape, copy=copy)

    # 将 transpose 方法的文档字符串设置为 _spbase.transpose 的文档字符串
    transpose.__doc__ = _spbase.transpose.__doc__

    def resize(self, *shape) -> None:
        # 检查是否是数组
        is_array = isinstance(self, sparray)
        # 检查并修正形状
        shape = check_shape(shape, allow_1d=is_array)

        # 检查是否增加了维度
        if len(shape) > self.ndim:
            # 将坐标展平，并保留在新形状下的最大尺寸
            flat_coords = _ravel_coords(self.coords, self.shape)
            max_size = math.prod(shape)
            self.coords = np.unravel_index(flat_coords[:max_size], shape)
            self.data = self.data[:max_size]
            self._shape = shape
            return

        # 检查是否减少了维度
        if len(shape) < self.ndim:
            # 重新定义形状，最后一个轴用于展平数组，其余轴用 1 填充
            tmp_shape = (
                self._shape[:len(shape) - 1]  # 原始形状去掉最后一个轴
                + (-1,)  # 最后一个轴用于展平数组
                + (1,) * (self.ndim - len(shape))  # 用 1 填充剩余轴
            )
            tmp = self.reshape(tmp_shape)
            self.coords = tmp.coords[:len(shape)]
            self._shape = tmp.shape[:len(shape)]

        # 处理现有维度的截断
        is_truncating = any(old > new for old, new in zip(self.shape, shape))
        if is_truncating:
            # 创建掩码以标识超出新形状的坐标
            mask = np.logical_and.reduce([
                idx < size for idx, size in zip(self.coords, shape)
            ])
            # 如果掩码有不全为真的情况，更新坐标和数据
            if not mask.all():
                self.coords = tuple(idx[mask] for idx in self.coords)
                self.data = self.data[mask]

        # 更新形状
        self._shape = shape

    # 将 resize 方法的文档字符串设置为 _spbase.resize 的文档字符串
    resize.__doc__ = _spbase.resize.__doc__
    def tocsc(self, copy=False):
        """Convert this array/matrix to Compressed Sparse Column format
        
        Duplicate entries will be summed together.
        
        Examples
        --------
        >>> from numpy import array
        >>> from scipy.sparse import coo_array
        >>> row  = array([0, 0, 1, 3, 1, 0, 0])
        >>> col  = array([0, 2, 1, 3, 1, 0, 0])
        >>> data = array([1, 1, 1, 1, 1, 1, 1])
        >>> A = coo_array((data, (row, col)), shape=(4, 4)).tocsc()
        >>> A.toarray()
        array([[3, 0, 1, 0],
               [0, 2, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 1]])

        """
        # 如果数组维度不是2，则抛出异常，无法将1维稀疏数组转换为CSC格式
        if self.ndim != 2:
            raise ValueError("Cannot convert a 1d sparse array to csc format")
        # 如果稀疏数组中非零元素个数为0，则返回一个空的CSC格式的容器
        if self.nnz == 0:
            return self._csc_container(self.shape, dtype=self.dtype)
        else:
            # 导入CSC格式数组类
            from ._csc import csc_array
            # 调用内部方法将COO格式转换为压缩稀疏列（CSC）格式的必要元素
            indptr, indices, data, shape = self._coo_to_compressed(csc_array._swap)

            # 创建CSC格式的容器对象
            x = self._csc_container((data, indices, indptr), shape=shape)
            # 如果稀疏数组没有处于规范格式，则合并重复的条目
            if not self.has_canonical_format:
                x.sum_duplicates()
            return x
    # 将稀疏矩阵转换为压缩稀疏行 (CSR) 格式

    # 如果稀疏矩阵中没有非零元素，则返回一个空的 CSR 容器
    if self.nnz == 0:
        return self._csr_container(self.shape, dtype=self.dtype)
    else:
        # 导入 csr_array 函数用于压缩 COO 格式到 CSR 格式的转换
        from ._csr import csr_array
        # 调用 _coo_to_compressed 方法，将 COO 格式的数组转换为压缩格式的数组
        arrays = self._coo_to_compressed(csr_array._swap, copy=copy)
        indptr, indices, data, shape = arrays

        # 使用转换后的数组创建 CSR 格式的矩阵
        x = self._csr_container((data, indices, indptr), shape=self.shape)
        # 如果稀疏矩阵没有按照规范格式排列，则调用 sum_duplicates 方法
        if not self.has_canonical_format:
            x.sum_duplicates()
        return x

def _coo_to_compressed(self, swap, copy=False):
    """将 (shape, coords, data) 转换为 (indptr, indices, data, shape)"""
    M, N = swap(self._shape_as_2d)
    # 将 idx_dtype 类型从 intc 转换为 int32，用于 pythran。
    # 在 scipy/optimize/tests/test__numdiff.py::test_group_columns 中进行了测试
    idx_dtype = self._get_index_dtype(self.coords, maxval=max(self.nnz, N))

    if self.ndim == 1:
        # 复制或使用原始数据创建 indices 数组
        indices = self.coords[0].copy() if copy else self.coords[0]
        nnz = len(indices)
        # 创建包含非零元素索引的指针数组 indptr
        indptr = np.array([0, nnz], dtype=idx_dtype)
        # 复制或使用原始数据创建数据数组 data
        data = self.data.copy() if copy else self.data
        return indptr, indices, data, self.shape

    # ndim == 2
    # 交换并转换 major 和 minor 坐标数组的数据类型为 idx_dtype
    major, minor = swap(self.coords)
    nnz = len(major)
    major = major.astype(idx_dtype, copy=False)
    minor = minor.astype(idx_dtype, copy=False)

    # 创建 indptr 数组，其长度为 M+1，数据类型为 idx_dtype
    indptr = np.empty(M + 1, dtype=idx_dtype)
    # 创建 indices 数组，数据类型为 idx_dtype，形状与 minor 相同
    indices = np.empty_like(minor, dtype=idx_dtype)
    # 创建数据数组 data，数据类型为 self.dtype，形状与原始数据相同
    data = np.empty_like(self.data, dtype=self.dtype)

    # 调用 coo_tocsr 函数进行 COO 到 CSR 的转换
    coo_tocsr(M, N, nnz, major, minor, self.data, indptr, indices, data)
    return indptr, indices, data, self.shape

def tocoo(self, copy=False):
    # 如果 copy 参数为 True，则返回当前对象的副本
    if copy:
        return self.copy()
    else:
        # 否则返回当前对象本身
        return self

# 将 tocoo 方法的文档字符串设置为 _spbase.tocoo 方法的文档字符串
tocoo.__doc__ = _spbase.tocoo.__doc__
    # 将稀疏矩阵转换为 DIA 格式，支持复制选项
    def todia(self, copy=False):
        # 如果稀疏数组的维度不是2，则无法转换为 dia 格式，抛出数值错误
        if self.ndim != 2:
            raise ValueError("Cannot convert a 1d sparse array to dia format")
        
        # 合并重复的元素
        self.sum_duplicates()
        
        # 计算每个非零元素所在的对角线索引
        ks = self.col - self.row  # 每个非零元素所在的对角线
        diags, diag_idx = np.unique(ks, return_inverse=True)

        if len(diags) > 100:
            # 如果对角线数量超过100，可能是不期望的，是否应该在 todia() 中添加 maxdiags 参数？
            warn("Constructing a DIA matrix with %d diagonals "
                 "is inefficient" % len(diags),
                 SparseEfficiencyWarning, stacklevel=2)

        # 初始化并填充数据数组
        if self.data.size == 0:
            data = np.zeros((0, 0), dtype=self.dtype)
        else:
            data = np.zeros((len(diags), self.col.max()+1), dtype=self.dtype)
            data[diag_idx, self.col] = self.data

        # 返回用于包装 DIA 格式数据的容器
        return self._dia_container((data, diags), shape=self.shape)

    # 使用 _spbase 中的 todia 方法的文档字符串作为 todia 方法的文档字符串
    todia.__doc__ = _spbase.todia.__doc__

    # 将稀疏矩阵转换为 DOK 格式，支持复制选项
    def todok(self, copy=False):
        # 合并重复的元素
        self.sum_duplicates()
        
        # 创建一个 DOK 格式的容器
        dok = self._dok_container(self.shape, dtype=self.dtype)
        
        # 确保一维坐标不是元组
        if self.ndim == 1:
            coords = self.coords[0]
        else:
            coords = zip(*self.coords)

        # 将坐标和数据存入 DOK 容器的字典中
        dok._dict = dict(zip(coords, self.data))
        
        # 返回 DOK 容器
        return dok

    # 使用 _spbase 中的 todok 方法的文档字符串作为 todok 方法的文档字符串
    todok.__doc__ = _spbase.todok.__doc__

    # 提取对角线元素到数组中
    def diagonal(self, k=0):
        # 如果稀疏数组的维度不是2，则无法提取对角线，抛出数值错误
        if self.ndim != 2:
            raise ValueError("diagonal requires two dimensions")
        
        # 获取数组的行数和列数
        rows, cols = self.shape
        
        # 如果 k 小于等于 -rows 或者大于等于 cols，则返回一个空数组
        if k <= -rows or k >= cols:
            return np.empty(0, dtype=self.data.dtype)
        
        # 创建一个用于存放对角线元素的数组
        diag = np.zeros(min(rows + min(k, 0), cols - max(k, 0)),
                        dtype=self.dtype)
        
        # 创建一个布尔掩码，标识哪些元素是对角线上的
        diag_mask = (self.row + k) == self.col

        # 如果数组已经处于规范格式，则直接使用行和数据
        if self.has_canonical_format:
            row = self.row[diag_mask]
            data = self.data[diag_mask]
        else:
            # 否则，从 coords 中提取对应的索引，并通过 _sum_duplicates 合并重复的元素
            inds = tuple(idx[diag_mask] for idx in self.coords)
            (row, _), data = self._sum_duplicates(inds, self.data[diag_mask])
        
        # 将对角线元素放入 diag 数组中
        diag[row + min(k, 0)] = data

        # 返回对角线数组
        return diag

    # 使用 _data_matrix 中的 diagonal 方法的文档字符串作为 diagonal 方法的文档字符串
    diagonal.__doc__ = _data_matrix.diagonal.__doc__
    # 设置对角线上的值
    def _setdiag(self, values, k):
        # 检查数组维度是否为二维，若不是则抛出数值错误异常
        if self.ndim != 2:
            raise ValueError("setting a diagonal requires two dimensions")
        
        # 获取矩阵的行数 M 和列数 N
        M, N = self.shape
        
        # 如果 values 的维度不为空且长度为零，则直接返回
        if values.ndim and not len(values):
            return
        
        # 索引数据类型为行索引的数据类型
        idx_dtype = self.row.dtype

        # 确定要保留的三元组以及新元素的放置位置
        full_keep = self.col - self.row != k
        
        # 根据 k 的正负确定新元素的行和列
        if k < 0:
            max_index = min(M+k, N)
            if values.ndim:
                max_index = min(max_index, len(values))
            keep = np.logical_or(full_keep, self.col >= max_index)
            new_row = np.arange(-k, -k + max_index, dtype=idx_dtype)
            new_col = np.arange(max_index, dtype=idx_dtype)
        else:
            max_index = min(M, N-k)
            if values.ndim:
                max_index = min(max_index, len(values))
            keep = np.logical_or(full_keep, self.row >= max_index)
            new_row = np.arange(max_index, dtype=idx_dtype)
            new_col = np.arange(k, k + max_index, dtype=idx_dtype)

        # 定义数据数组，包含要添加的条目
        if values.ndim:
            new_data = values[:max_index]
        else:
            new_data = np.empty(max_index, dtype=self.dtype)
            new_data[:] = values

        # 更新内部结构
        self.coords = (np.concatenate((self.row[keep], new_row)),
                       np.concatenate((self.col[keep], new_col)))
        self.data = np.concatenate((self.data[keep], new_data))
        self.has_canonical_format = False

    # _data_matrix 需要使用的方法
    def _with_data(self, data, copy=True):
        """返回一个与 self 具有相同稀疏结构但数据不同的矩阵。
        
        默认情况下，索引数组会被复制。
        """
        if copy:
            # 复制索引数组
            coords = tuple(idx.copy() for idx in self.coords)
        else:
            coords = self.coords
        return self.__class__((data, coords), shape=self.shape, dtype=data.dtype)

    # 合并重复的条目
    def sum_duplicates(self) -> None:
        """通过将重复的条目相加来消除它们
        
        这是一个就地操作。
        """
        if self.has_canonical_format:
            return
        # 调用 _sum_duplicates 方法进行合并重复条目
        summed = self._sum_duplicates(self.coords, self.data)
        self.coords, self.data = summed
        self.has_canonical_format = True
    def _sum_duplicates(self, coords, data):
        # 假设 coords 不是规范格式。
        if len(data) == 0:
            return coords, data
        # 根据行和列对 coords 进行排序。这对应于 C-order，
        # 我们依赖 argmin/argmax 返回第一个索引，与 numpy 处理相同（在存在并列的情况下）。
        order = np.lexsort(coords[::-1])
        coords = tuple(idx[order] for idx in coords)
        data = data[order]
        # 创建一个布尔掩码，标记出不重复的坐标索引
        unique_mask = np.logical_or.reduce([
            idx[1:] != idx[:-1] for idx in coords
        ])
        unique_mask = np.append(True, unique_mask)
        coords = tuple(idx[unique_mask] for idx in coords)
        # 使用 unique_inds 对 data 进行按块求和操作，使用 self.dtype 作为结果的数据类型
        unique_inds, = np.nonzero(unique_mask)
        data = np.add.reduceat(data, unique_inds, dtype=self.dtype)
        return coords, data

    def eliminate_zeros(self):
        """从数组/矩阵中删除零条目

        这是一个 *原地* 操作
        """
        # 创建一个布尔掩码，标记出非零元素
        mask = self.data != 0
        self.data = self.data[mask]
        self.coords = tuple(idx[mask] for idx in self.coords)

    #######################
    # 算术处理器 #
    #######################

    def _add_dense(self, other):
        if other.shape != self.shape:
            raise ValueError(f'不兼容的形状 ({self.shape} 和 {other.shape})')
        # 根据 self.dtype 和 other.dtype 的字符码进行类型提升
        dtype = upcast_char(self.dtype.char, other.dtype.char)
        # 使用其他数组创建一个拷贝并指定类型
        result = np.array(other, dtype=dtype, copy=True)
        # 判断 result 是否列优先，获取 M 和 N 的值
        fortran = int(result.flags.f_contiguous)
        M, N = self._shape_as_2d
        # 调用 coo_todense 函数填充 result 数组
        coo_todense(M, N, self.nnz, self.row, self.col, self.data,
                    result.ravel('A'), fortran)
        return self._container(result, copy=False)

    def _matmul_vector(self, other):
        result_shape = self.shape[0] if self.ndim > 1 else 1
        # 创建一个结果数组，用于存储乘积的结果
        result = np.zeros(result_shape,
                          dtype=upcast_char(self.dtype.char, other.dtype.char))

        if self.ndim == 2:
            col = self.col
            row = self.row
        elif self.ndim == 1:
            col = self.coords[0]
            row = np.zeros_like(col)
        else:
            raise NotImplementedError(
                f"coo_matvec 对于 ndim={self.ndim} 尚未实现")

        # 调用 coo_matvec 函数计算 COO 格式的矩阵向量乘积
        coo_matvec(self.nnz, row, col, self.data, other, result)
        # 如果 self 是 sparray 类型并且结果形状为 1，则返回结果数组的第一个元素
        if isinstance(self, sparray) and result_shape == 1:
            return result[0]
        return result
    # 定义一个方法用于多向量与稀疏矩阵的乘法操作
    def _matmul_multivector(self, other):
        # 确定结果的数据类型为两个操作数的类型中更高级别的数据类型
        result_dtype = upcast_char(self.dtype.char, other.dtype.char)
        # 根据稀疏矩阵的维度进行不同的处理
        if self.ndim == 2:
            # 如果稀疏矩阵是二维的，结果的形状为 (other.shape[1], self.shape[0])
            result_shape = (other.shape[1], self.shape[0])
            # 获取稀疏矩阵的列索引
            col = self.col
            # 获取稀疏矩阵的行索引
            row = self.row
        elif self.ndim == 1:
            # 如果稀疏矩阵是一维的，结果的形状为 (other.shape[1],)
            result_shape = (other.shape[1],)
            # 获取稀疏矩阵的坐标数组的第一个元素作为列索引
            col = self.coords[0]
            # 创建一个与 col 同样形状的零数组作为行索引
            row = np.zeros_like(col)
        else:
            # 如果稀疏矩阵的维度不是 1 或 2，则抛出未实现的错误
            raise NotImplementedError(
                f"coo_matvec not implemented for ndim={self.ndim}")

        # 创建一个结果数组，形状为 result_shape，数据类型为 result_dtype
        result = np.zeros(result_shape, dtype=result_dtype)
        # 对 other 的转置进行遍历，使用稀疏矩阵-向量乘法的函数进行计算，并将结果存入 result 中
        for i, other_col in enumerate(other.T):
            coo_matvec(self.nnz, row, col, self.data, other_col, result[i:i + 1])
        # 返回结果的转置视图，其类型与 other 的类型相同
        return result.T.view(type=type(other))
# 定义一个函数 _ravel_coords，用于将多维坐标映射成扁平化的索引值
def _ravel_coords(coords, shape, order='C'):
    """Like np.ravel_multi_index, but avoids some overflow issues."""
    # 如果坐标列表长度为1，直接返回唯一的坐标值
    if len(coords) == 1:
        return coords[0]
    
    # 如果坐标列表长度为2，分别处理行优先（'C'）和列优先（'F'）的情况
    if len(coords) == 2:
        nrows, ncols = shape
        row, col = coords
        # 根据指定的顺序处理坐标映射问题，避免溢出
        if order == 'C':
            maxval = (ncols * max(0, nrows - 1) + max(0, ncols - 1))
            idx_dtype = get_index_dtype(maxval=maxval)
            return np.multiply(ncols, row, dtype=idx_dtype) + col
        elif order == 'F':
            maxval = (nrows * max(0, ncols - 1) + max(0, nrows - 1))
            idx_dtype = get_index_dtype(maxval=maxval)
            return np.multiply(nrows, col, dtype=idx_dtype) + row
        else:
            # 如果顺序不是 'C' 或 'F'，则抛出错误
            raise ValueError("'order' must be 'C' or 'F'")
    
    # 对于其他长度的坐标列表，使用 numpy 的 ravel_multi_index 函数进行映射
    return np.ravel_multi_index(coords, shape, order=order)


# 判断对象是否为 COO 稀疏矩阵类型的函数
def isspmatrix_coo(x):
    """Is `x` of coo_matrix type?

    Parameters
    ----------
    x
        object to check for being a coo matrix

    Returns
    -------
    bool
        True if `x` is a coo matrix, False otherwise

    Examples
    --------
    >>> from scipy.sparse import coo_array, coo_matrix, csr_matrix, isspmatrix_coo
    >>> isspmatrix_coo(coo_matrix([[5]]))
    True
    >>> isspmatrix_coo(coo_array([[5]]))
    False
    >>> isspmatrix_coo(csr_matrix([[5]]))
    False
    """
    return isinstance(x, coo_matrix)


# 定义 COO 格式稀疏数组的类，继承自 _coo_base 和 sparray
class coo_array(_coo_base, sparray):
    """
    A sparse array in COOrdinate format.

    Also known as the 'ijv' or 'triplet' format.

    This can be instantiated in several ways:
        coo_array(D)
            where D is an ndarray

        coo_array(S)
            with another sparse array or matrix S (equivalent to S.tocoo())

        coo_array(shape, [dtype])
            to construct an empty sparse array with shape `shape`
            dtype is optional, defaulting to dtype='d'.

        coo_array((data, coords), [shape])
            to construct from existing data and index arrays:
                1. data[:]       the entries of the sparse array, in any order
                2. coords[i][:]  the axis-i coordinates of the data entries

            Where ``A[coords] = data``, and coords is a tuple of index arrays.
            When shape is not specified, it is inferred from the index arrays.

    Attributes
    ----------
    dtype : dtype
        Data type of the sparse array
    shape : tuple of integers
        Shape of the sparse array
    ndim : int
        Number of dimensions of the sparse array
    nnz
    size
    data
        COO format data array of the sparse array
    coords
        COO format tuple of index arrays
    has_canonical_format : bool
        Whether the matrix has sorted coordinates and no duplicates
    format
    T

    Notes
    -----

    Sparse arrays can be used in arithmetic operations: they support
    """
    addition, subtraction, multiplication, division, and matrix power.

    Advantages of the COO format
        - facilitates fast conversion among sparse formats
        - permits duplicate entries (see example)
        - very fast conversion to and from CSR/CSC formats

    Disadvantages of the COO format
        - does not directly support:
            + arithmetic operations
            + slicing

    Intended Usage
        - COO is a fast format for constructing sparse arrays
        - Once a COO array has been constructed, convert to CSR or
          CSC format for fast arithmetic and matrix vector operations
        - By default when converting to CSR or CSC format, duplicate (i,j)
          entries will be summed together.  This facilitates efficient
          construction of finite element matrices and the like. (see example)

    Canonical format
        - Entries and coordinates sorted by row, then column.
        - There are no duplicate entries (i.e. duplicate (i,j) locations)
        - Data arrays MAY have explicit zeros.

    Examples
    --------

    >>> # Constructing an empty sparse array
    >>> import numpy as np
    >>> from scipy.sparse import coo_array
    >>> coo_array((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

    >>> # Constructing a sparse array using ijv format
    >>> row  = np.array([0, 3, 1, 0])
    >>> col  = np.array([0, 3, 1, 2])
    >>> data = np.array([4, 5, 7, 9])
    >>> coo_array((data, (row, col)), shape=(4, 4)).toarray()
    array([[4, 0, 9, 0],
           [0, 7, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 5]])

    >>> # Constructing a sparse array with duplicate coordinates
    >>> row  = np.array([0, 0, 1, 3, 1, 0, 0])
    >>> col  = np.array([0, 2, 1, 3, 1, 0, 0])
    >>> data = np.array([1, 1, 1, 1, 1, 1, 1])
    >>> coo = coo_array((data, (row, col)), shape=(4, 4))
    >>> # Duplicate coordinates are maintained until implicitly or explicitly summed
    >>> np.max(coo.data)
    1
    >>> coo.toarray()
    array([[3, 0, 1, 0],
           [0, 2, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 1]])
class coo_matrix(spmatrix, _coo_base):
    """
    A sparse matrix in COOrdinate format.

    Also known as the 'ijv' or 'triplet' format.

    This can be instantiated in several ways:
        coo_matrix(D)
            where D is a 2-D ndarray

        coo_matrix(S)
            with another sparse array or matrix S (equivalent to S.tocoo())

        coo_matrix((M, N), [dtype])
            to construct an empty matrix with shape (M, N)
            dtype is optional, defaulting to dtype='d'.

        coo_matrix((data, (i, j)), [shape=(M, N)])
            to construct from three arrays:
                1. data[:]   the entries of the matrix, in any order
                2. i[:]      the row indices of the matrix entries
                3. j[:]      the column indices of the matrix entries

            Where ``A[i[k], j[k]] = data[k]``.  When shape is not
            specified, it is inferred from the index arrays

    Attributes
    ----------
    dtype : dtype
        Data type of the matrix
    shape : 2-tuple
        Shape of the matrix
    ndim : int
        Number of dimensions (this is always 2)
    nnz
        Number of non-zero elements in the matrix
    size
        Number of elements in the matrix
    data
        COO format data array of the matrix
    row
        COO format row index array of the matrix
    col
        COO format column index array of the matrix
    has_canonical_format : bool
        Whether the matrix has sorted indices and no duplicates
    format
        Format of the sparse matrix representation
    T
        Transpose of the matrix

    Notes
    -----

    Sparse matrices can be used in arithmetic operations: they support
    addition, subtraction, multiplication, division, and matrix power.

    Advantages of the COO format
        - facilitates fast conversion among sparse formats
        - permits duplicate entries (see example)
        - very fast conversion to and from CSR/CSC formats

    Disadvantages of the COO format
        - does not directly support:
            + arithmetic operations
            + slicing

    Intended Usage
        - COO is a fast format for constructing sparse matrices
        - Once a COO matrix has been constructed, convert to CSR or
          CSC format for fast arithmetic and matrix vector operations
        - By default when converting to CSR or CSC format, duplicate (i,j)
          entries will be summed together.  This facilitates efficient
          construction of finite element matrices and the like. (see example)

    Canonical format
        - Entries and coordinates sorted by row, then column.
        - There are no duplicate entries (i.e. duplicate (i,j) locations)
        - Data arrays MAY have explicit zeros.

    Examples
    --------

    >>> # Constructing an empty matrix
    >>> import numpy as np
    >>> from scipy.sparse import coo_matrix
    >>> coo_matrix((3, 4), dtype=np.int8).toarray()
    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]], dtype=int8)

    >>> # Constructing a matrix using ijv format
    >>> row  = np.array([0, 3, 1, 0])
    >>> col  = np.array([0, 3, 1, 2])
    >>> data = np.array([4, 5, 7, 9])
    >>> coo_matrix((data, (row, col)), shape=(4, 4)).toarray()
    array([[4, 0, 9, 0],
           [0, 7, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 5]])

    >>> # Constructing a matrix with duplicate coordinates
    >>> row  = np.array([0, 0, 1, 3, 1, 0, 0])
    >>> col  = np.array([0, 2, 1, 3, 1, 0, 0])
    >>> data = np.array([1, 1, 1, 1, 1, 1, 1])
    >>> coo = coo_matrix((data, (row, col)), shape=(4, 4))
    >>> # Duplicate coordinates are maintained until implicitly or explicitly summed
    >>> np.max(coo.data)
    1
    >>> coo.toarray()
    array([[3, 0, 1, 0],
           [0, 2, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 1]])

    """

    # 定义一个自定义类的 __setstate__ 方法
    def __setstate__(self, state):
        # 如果状态字典中不包含 'coords' 键
        if 'coords' not in state:
            # 为了与之前版本的属性兼容，使用 'row' 和 'col' 存储 nnz（非零元素）的坐标
            state['coords'] = (state.pop('row'), state.pop('col'))
        # 更新对象的状态字典
        self.__dict__.update(state)
```