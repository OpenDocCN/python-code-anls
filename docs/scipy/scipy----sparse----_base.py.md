# `D:\src\scipysrc\scipy\scipy\sparse\_base.py`

```
# 导入 NumPy 库，并将其命名为 np
import numpy as np

# 从 _sputils 模块中导入多个函数和类
from ._sputils import (asmatrix, check_reshape_kwargs, check_shape,
                       get_sum_dtype, isdense, isscalarlike,
                       matrix, validateaxis, getdtype)

# 从 _matrix 模块中导入 spmatrix 类
from ._matrix import spmatrix

# 定义该模块中公开的对象列表
__all__ = ['isspmatrix', 'issparse', 'sparray',
           'SparseWarning', 'SparseEfficiencyWarning']

# 定义警告基类 SparseWarning
class SparseWarning(Warning):
    pass

# 定义格式警告类 SparseFormatWarning，继承自 SparseWarning
class SparseFormatWarning(SparseWarning):
    pass

# 定义效率警告类 SparseEfficiencyWarning，继承自 SparseWarning
class SparseEfficiencyWarning(SparseWarning):
    pass

# 定义稀疏矩阵可能的格式及其描述的字典
_formats = {'csc': [0, "Compressed Sparse Column"],
            'csr': [1, "Compressed Sparse Row"],
            'dok': [2, "Dictionary Of Keys"],
            'lil': [3, "List of Lists"],
            'dod': [4, "Dictionary of Dictionaries"],
            'sss': [5, "Symmetric Sparse Skyline"],
            'coo': [6, "COOrdinate"],
            'lba': [7, "Linpack BAnded"],
            'egd': [8, "Ellpack-itpack Generalized Diagonal"],
            'dia': [9, "DIAgonal"],
            'bsr': [10, "Block Sparse Row"],
            'msr': [11, "Modified compressed Sparse Row"],
            'bsc': [12, "Block Sparse Column"],
            'msc': [13, "Modified compressed Sparse Column"],
            'ssk': [14, "Symmetric SKyline"],
            'nsk': [15, "Nonsymmetric SKyline"],
            'jad': [16, "JAgged Diagonal"],
            'uss': [17, "Unsymmetric Sparse Skyline"],
            'vbr': [18, "Variable Block Row"],
            'und': [19, "Undefined"]
            }

# 包含保留零点的一元通用函数集合
_ufuncs_with_fixed_point_at_zero = frozenset([
        np.sin, np.tan, np.arcsin, np.arctan, np.sinh, np.tanh, np.arcsinh,
        np.arctanh, np.rint, np.sign, np.expm1, np.log1p, np.deg2rad,
        np.rad2deg, np.floor, np.ceil, np.trunc, np.sqrt])

# 最大打印条目数
MAXPRINT = 50

# _spbase 类，为所有稀疏数组提供基类，不能被实例化
class _spbase:
    """ This class provides a base class for all sparse arrays.  It
    cannot be instantiated.  Most of the work is provided by subclasses.
    """

    # 数组优先级
    __array_priority__ = 10.1
    # 数组格式，默认为未定义
    _format = 'und'  # undefined

    # 返回数组维度
    @property
    def ndim(self) -> int:
        return len(self._shape)

    # 返回数组的形状作为二维元组
    @property
    def _shape_as_2d(self):
        s = self._shape
        return (1, s[-1]) if len(s) == 1 else s

    # 返回 BSR 格式的容器类型
    @property
    def _bsr_container(self):
        from ._bsr import bsr_array
        return bsr_array

    # 返回 COO 格式的容器类型
    @property
    def _coo_container(self):
        from ._coo import coo_array
        return coo_array

    # 返回 CSC 格式的容器类型
    @property
    def _csc_container(self):
        from ._csc import csc_array
        return csc_array

    # 返回 CSR 格式的容器类型
    @property
    def _csr_container(self):
        from ._csr import csr_array
        return csr_array

    # 返回 DIA 格式的容器类型
    @property
    def _dia_container(self):
        from ._dia import dia_array
        return dia_array

    # 返回 DOK 格式的容器类型
    @property
    def _dok_container(self):
        from ._dok import dok_array
        return dok_array
    # 返回._lil模块中的lil_array函数对象
    def _lil_container(self):
        from ._lil import lil_array
        return lil_array

    # 初始化函数，设置对象属性和检查实例化条件
    def __init__(self, arg1, *, maxprint=None):
        self._shape = None
        # 检查类名是否为'_spbase'，如果是则引发值错误异常
        if self.__class__.__name__ == '_spbase':
            raise ValueError("This class is not intended"
                             " to be instantiated directly.")
        # 如果arg1是sparray的实例且是标量，则引发值错误异常
        if isinstance(self, sparray) and np.isscalar(arg1):
            raise ValueError(
                "scipy sparse array classes do not support instantiation from a scalar"
            )
        # 设置最大打印数量，默认为MAXPRINT，如果maxprint为None则使用默认值
        self.maxprint = MAXPRINT if maxprint is None else maxprint

    # shape属性的getter方法，返回对象的形状信息
    @property
    def shape(self):
        return self._shape

    # 对稀疏数组/矩阵进行重新形状化的方法
    def reshape(self, *args, **kwargs):
        """reshape(self, shape, order='C', copy=False)

        给稀疏数组/矩阵一个新的形状，但不改变其数据。

        Parameters
        ----------
        shape : 长度为2的整数元组
            新的形状应与原始形状兼容。
        order : {'C', 'F'}, 可选
            使用此索引顺序读取元素。'C'表示使用类似C的索引顺序读取和写入元素；例如，首先读取整个第一行，然后第二行，依此类推。
            'F'表示使用类似Fortran的索引顺序读取和写入元素；例如，首先读取整个第一列，然后第二列，依此类推。
        copy : bool, 可选
            指示是否应尽可能复制self的属性。根据所使用的稀疏数组类型，复制属性的程度有所不同。

        Returns
        -------
        reshaped : 稀疏数组/矩阵
            具有给定`shape`的稀疏数组/矩阵，不一定与当前对象的格式相同。

        See Also
        --------
        numpy.reshape : ndarray的NumPy实现的'reshape'

        """
        # 如果形状已经匹配，则不必执行实际的重新形状化
        # 否则，默认转换为COO格式并使用其reshape
        is_array = isinstance(self, sparray)
        # 检查形状，允许1维数组
        shape = check_shape(args, self.shape, allow_1d=is_array)
        # 检查reshape方法的关键字参数
        order, copy = check_reshape_kwargs(kwargs)
        # 如果形状与当前形状相同，则根据copy标志返回副本或者原对象
        if shape == self.shape:
            if copy:
                return self.copy()
            else:
                return self

        # 将对象转换为COO格式，然后进行reshape操作
        return self.tocoo(copy=copy).reshape(shape, order=order, copy=False)
    def resize(self, shape):
        """Resize the array/matrix in-place to dimensions given by ``shape``

        Any elements that lie within the new shape will remain at the same
        indices, while non-zero elements lying outside the new shape are
        removed.

        Parameters
        ----------
        shape : (int, int)
            number of rows and columns in the new array/matrix

        Notes
        -----
        The semantics are not identical to `numpy.ndarray.resize` or
        `numpy.resize`. Here, the same data will be maintained at each index
        before and after reshape, if that index is within the new bounds. In
        numpy, resizing maintains contiguity of the array, moving elements
        around in the logical array but not within a flattened representation.

        We give no guarantees about whether the underlying data attributes
        (arrays, etc.) will be modified in place or replaced with new objects.
        """
        # 抛出未实现错误，提示该方法尚未在具体子类中实现
        raise NotImplementedError(
            f'{type(self).__name__}.resize is not implemented')

    def astype(self, dtype, casting='unsafe', copy=True):
        """Cast the array/matrix elements to a specified type.

        Parameters
        ----------
        dtype : string or numpy dtype
            Typecode or data-type to which to cast the data.
        casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
            Controls what kind of data casting may occur.
            Defaults to 'unsafe' for backwards compatibility.
            'no' means the data types should not be cast at all.
            'equiv' means only byte-order changes are allowed.
            'safe' means only casts which can preserve values are allowed.
            'same_kind' means only safe casts or casts within a kind,
            like float64 to float32, are allowed.
            'unsafe' means any data conversions may be done.
        copy : bool, optional
            If `copy` is `False`, the result might share some memory with this
            array/matrix. If `copy` is `True`, it is guaranteed that the result and
            this array/matrix do not share any memory.
        """
        # 获取确切的数据类型对象
        dtype = getdtype(dtype)
        # 如果当前对象的数据类型不等于目标数据类型
        if self.dtype != dtype:
            # 将当前对象转换为 CSR 格式，然后进行类型转换，并按照指定的参数创建格式化对象
            return self.tocsr().astype(
                dtype, casting=casting, copy=copy).asformat(self.format)
        # 如果需要复制对象
        elif copy:
            # 返回当前对象的一个复制副本
            return self.copy()
        else:
            # 直接返回当前对象自身
            return self

    @classmethod
    def _ascontainer(cls, X, **kwargs):
        # 如果当前类是 sparray 的子类
        if issubclass(cls, sparray):
            # 将输入 X 转换为 numpy 数组，并使用指定的关键字参数
            return np.asarray(X, **kwargs)
        else:
            # 将输入 X 转换为 matrix 对象，并使用指定的关键字参数
            return asmatrix(X, **kwargs)

    @classmethod
    def _container(cls, X, **kwargs):
        # 如果当前类是 sparray 的子类
        if issubclass(cls, sparray):
            # 将输入 X 转换为 numpy 数组，并使用指定的关键字参数
            return np.array(X, **kwargs)
        else:
            # 将输入 X 转换为 matrix 对象，并使用指定的关键字参数
            return matrix(X, **kwargs)
    def _asfptype(self):
        """Upcast array to a floating point format (if necessary)"""

        # 定义浮点类型的字符表示形式
        fp_types = ['f', 'd', 'F', 'D']

        # 检查当前数组的数据类型是否已经是浮点类型之一，如果是则直接返回数组本身
        if self.dtype.char in fp_types:
            return self
        else:
            # 遍历浮点类型列表，找到第一个大于等于当前数组数据类型的浮点类型，并将数组转换为该类型
            for fp_type in fp_types:
                if self.dtype <= np.dtype(fp_type):
                    return self.astype(fp_type)

            # 如果未找到合适的浮点类型，则抛出类型错误异常
            raise TypeError(
                f'cannot upcast [{self.dtype.name}] to a floating point format'
            )

    def __iter__(self):
        # 实现迭代器接口，返回数组的每一行
        for r in range(self.shape[0]):
            yield self[r]

    def _getmaxprint(self):
        """Maximum number of elements to display when printed."""
        # 返回对象的最大打印元素数
        return self.maxprint

    def count_nonzero(self, axis=None):
        """Number of non-zero entries, equivalent to

        np.count_nonzero(a.toarray(), axis=axis)

        Unlike the nnz property, which return the number of stored
        entries (the length of the data attribute), this method counts the
        actual number of non-zero entries in data.

        Duplicate entries are summed before counting.

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None} optional
            Count nonzeros for the whole array, or along a specified axis.

            .. versionadded:: 1.15.0

        Returns
        -------
        numpy array
            A reduced array (no axis `axis`) holding the number of nonzero values
            for each of the indices of the nonaxis dimensions.

        Notes
        -----
        If you want to count nonzero and explicit zero stored values (e.g. nnz)
        along an axis, two fast idioms are provided by `numpy` functions for the
        common CSR, CSC, COO formats.

        For the major axis in CSR (rows) and CSC (cols) use `np.diff`:

            >>> import numpy as np
            >>> import scipy as sp
            >>> A = sp.sparse.csr_array([[4, 5, 0], [7, 0, 0]])
            >>> major_axis_stored_values = np.diff(A.indptr)  # -> np.array([2, 1])

        For the minor axis in CSR (cols) and CSC (rows) use `numpy.bincount` with
        minlength ``A.shape[1]`` for CSR and ``A.shape[0]`` for CSC:

            >>> csr_minor_stored_values = np.bincount(A.indices, minlength=A.shape[1])

        For COO, use the minor axis approach for either `axis`:

            >>> A = A.tocoo()
            >>> coo_axis0_stored_values = np.bincount(A.coords[0], minlength=A.shape[1])
            >>> coo_axis1_stored_values = np.bincount(A.coords[1], minlength=A.shape[0])

        Examples
        --------

            >>> A = sp.sparse.csr_array([[4, 5, 0], [7, 0, 0]])
            >>> A.count_nonzero(axis=0)
            array([2, 1, 0])
        """
        # 获取当前类的名称
        clsname = self.__class__.__name__
        # 抛出未实现错误，提示该方法尚未在当前类中实现
        raise NotImplementedError(f"count_nonzero not implemented for {clsname}.")
    def _getnnz(self, axis=None):
        """Number of stored values, including explicit zeros.

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None} optional
            Report stored values for the whole array, or along a specified axis.

        See also
        --------
        count_nonzero : Number of non-zero entries
        """
        clsname = self.__class__.__name__
        # 抛出未实现错误，显示当前类名
        raise NotImplementedError(f"getnnz not implemented for {clsname}.")

    @property
    def nnz(self) -> int:
        """Number of stored values, including explicit zeros.

        See also
        --------
        count_nonzero : Number of non-zero entries
        """
        return self._getnnz()

    @property
    def size(self) -> int:
        """Number of stored values.

        See also
        --------
        count_nonzero : Number of non-zero values.
        """
        return self._getnnz()

    @property
    def format(self) -> str:
        """Format string for matrix."""
        return self._format

    @property
    def T(self):
        """Transpose."""
        return self.transpose()

    @property
    def real(self):
        # 返回实部
        return self._real()

    @property
    def imag(self):
        # 返回虚部
        return self._imag()

    def __repr__(self):
        _, format_name = _formats[self.format]
        sparse_cls = 'array' if isinstance(self, sparray) else 'matrix'
        # 返回对象的字符串表示形式，包括格式、稀疏类别、数据类型和形状信息
        return (
            f"<{format_name} sparse {sparse_cls} of dtype '{self.dtype}'\n"
            f"\twith {self.nnz} stored elements and shape {self.shape}>"
        )

    def __str__(self):
        maxprint = self._getmaxprint()

        A = self.tocoo()

        # 辅助函数，输出 "(i,j)  v"
        def tostr(coords, data):
            pairs = zip(zip(*(c.tolist() for c in coords)), data)
            return '\n'.join(f'  {idx}\t{val}' for idx, val in pairs)

        out = repr(self)
        if self.nnz == 0:
            return out

        out += '\n  Coords\tValues\n'
        if self.nnz > maxprint:
            half = maxprint // 2
            # 输出稀疏矩阵的坐标和数据值，根据最大打印数量分段显示
            out += tostr(tuple(c[:half] for c in A.coords), A.data[:half])
            out += "\n  :\t:\n"
            half = maxprint - half
            out += tostr(tuple(c[-half:] for c in A.coords), A.data[-half:])
        else:
            out += tostr(A.coords, A.data)

        return out

    def __bool__(self):  # Simple -- other ideas?
        if self.shape == (1, 1):
            return self.nnz != 0
        else:
            # 如果矩阵不是1x1的，则抛出值错误，表明无法确定其真实值
            raise ValueError("The truth value of an array with more than one "
                             "element is ambiguous. Use a.any() or a.all().")
    __nonzero__ = __bool__

    # 应该返回稀疏矩阵的长度，暂时抛出异常来提醒需要定义该行为
    # What should len(sparse) return? For consistency with dense matrices,
    # perhaps it should be the number of rows?  But for some uses the number of
    # non-zeros is more important.  For now, raise an exception!
    def __len__(self):
        # 抛出类型错误，因为稀疏数组长度不明确，建议使用 getnnz() 或 shape[0]
        raise TypeError("sparse array length is ambiguous; use getnnz()"
                        " or shape[0]")

    def asformat(self, format, copy=False):
        """Return this array/matrix in the passed format.

        Parameters
        ----------
        format : {str, None}
            The desired sparse format ("csr", "csc", "lil", "dok", "array", ...)
            or None for no conversion.
        copy : bool, optional
            If True, the result is guaranteed to not share data with self.

        Returns
        -------
        A : This array/matrix in the passed format.
        """
        # 如果 format 是 None 或者已经是当前格式，根据 copy 参数决定返回拷贝或者原对象
        if format is None or format == self.format:
            if copy:
                return self.copy()
            else:
                return self
        else:
            try:
                # 尝试获取对应的格式转换方法，如 tocsr()、tocsc() 等
                convert_method = getattr(self, 'to' + format)
            except AttributeError as e:
                # 如果格式转换方法不存在，则抛出值错误
                raise ValueError(f'Format {format} is unknown.') from e

            # 转发 copy 参数给转换方法，如果支持的话
            try:
                return convert_method(copy=copy)
            except TypeError:
                return convert_method()

    ###################################################################
    #  NOTE: All arithmetic operations use csr_matrix by default.
    # Therefore a new sparse array format just needs to define a
    # .tocsr() method to provide arithmetic support. Any of these
    # methods can be overridden for efficiency.
    ####################################################################

    def multiply(self, other):
        """Point-wise multiplication by another array/matrix."""
        # 如果 other 是标量，则执行点乘标量操作；否则将当前对象转换为 CSR 格式再执行点乘操作
        if isscalarlike(other):
            return self._mul_scalar(other)
        return self.tocsr().multiply(other)

    def maximum(self, other):
        """Element-wise maximum between this and another array/matrix."""
        # 将当前对象转换为 CSR 格式后，执行元素级最大值操作
        return self.tocsr().maximum(other)

    def minimum(self, other):
        """Element-wise minimum between this and another array/matrix."""
        # 将当前对象转换为 CSR 格式后，执行元素级最小值操作
        return self.tocsr().minimum(other)

    def dot(self, other):
        """Ordinary dot product

        Examples
        --------
        >>> import numpy as np
        >>> from scipy.sparse import csr_array
        >>> A = csr_array([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
        >>> v = np.array([1, 0, -1])
        >>> A.dot(v)
        array([ 1, -3, -1], dtype=int64)

        """
        # 如果 other 是标量，则执行标量乘法；否则执行矩阵乘法
        if np.isscalar(other):
            return self * other
        else:
            return self @ other

    def power(self, n, dtype=None):
        """Element-wise power."""
        # 将当前对象转换为 CSR 格式后，执行元素的 n 次幂操作
        return self.tocsr().power(n, dtype=dtype)

    def __eq__(self, other):
        # 将当前对象转换为 CSR 格式后，执行相等比较操作
        return self.tocsr().__eq__(other)

    def __ne__(self, other):
        # 将当前对象转换为 CSR 格式后，执行不等比较操作
        return self.tocsr().__ne__(other)

    def __lt__(self, other):
        # 将当前对象转换为 CSR 格式后，执行小于比较操作
        return self.tocsr().__lt__(other)

    def __gt__(self, other):
        # 将当前对象转换为 CSR 格式后，执行大于比较操作
        return self.tocsr().__gt__(other)
    # 定义小于等于操作符重载方法，返回是否小于等于另一个对象
    def __le__(self, other):
        # 将当前对象转换为 CSR 格式后，调用其小于等于操作符方法
        return self.tocsr().__le__(other)

    # 定义大于等于操作符重载方法，返回是否大于等于另一个对象
    def __ge__(self, other):
        # 将当前对象转换为 CSR 格式后，调用其大于等于操作符方法
        return self.tocsr().__ge__(other)

    # 定义绝对值操作符重载方法，返回当前对象的绝对值
    def __abs__(self):
        # 返回当前对象转换为 CSR 格式后的绝对值
        return abs(self.tocsr())

    # 定义四舍五入操作符重载方法，返回当前对象的四舍五入值
    def __round__(self, ndigits=0):
        # 返回当前对象转换为 CSR 格式后的四舍五入值
        return round(self.tocsr(), ndigits=ndigits)

    # 定义稀疏矩阵与稀疏矩阵相加的方法
    def _add_sparse(self, other):
        # 将当前对象转换为 CSR 格式后，调用其与另一个稀疏矩阵相加的方法
        return self.tocsr()._add_sparse(other)

    # 定义稀疏矩阵与密集矩阵相加的方法
    def _add_dense(self, other):
        # 将当前对象转换为 COO 格式后，调用其与另一个密集矩阵相加的方法
        return self.tocoo()._add_dense(other)

    # 定义稀疏矩阵与稀疏矩阵相减的方法
    def _sub_sparse(self, other):
        # 将当前对象转换为 CSR 格式后，调用其与另一个稀疏矩阵相减的方法
        return self.tocsr()._sub_sparse(other)

    # 定义稀疏矩阵与密集矩阵相减的方法
    def _sub_dense(self, other):
        # 返回当前对象转换为密集矩阵后与另一个密集矩阵相减的结果
        return self.todense() - other

    # 定义密集矩阵与稀疏矩阵相减的方法
    def _rsub_dense(self, other):
        # 注意：对于无符号类型，无法简单替换为 other + (-self)
        # 返回另一个密集矩阵减去当前对象转换为密集矩阵的结果
        return other - self.todense()

    # 定义加法操作符重载方法，实现稀疏矩阵与标量、稀疏矩阵、密集矩阵的加法
    def __add__(self, other):  # self + other
        if isscalarlike(other):
            if other == 0:
                return self.copy()
            raise NotImplementedError('adding a nonzero scalar to a '
                                      'sparse array is not supported')
        elif issparse(other):
            if other.shape != self.shape:
                raise ValueError("inconsistent shapes")
            return self._add_sparse(other)
        elif isdense(other):
            other = np.broadcast_to(other, self.shape)
            return self._add_dense(other)
        else:
            return NotImplemented

    # 定义反向加法操作符重载方法，实现标量与稀疏矩阵的加法
    def __radd__(self,other):  # other + self
        return self.__add__(other)

    # 定义减法操作符重载方法，实现稀疏矩阵与标量、稀疏矩阵、密集矩阵的减法
    def __sub__(self, other):  # self - other
        if isscalarlike(other):
            if other == 0:
                return self.copy()
            raise NotImplementedError('subtracting a nonzero scalar from a '
                                      'sparse array is not supported')
        elif issparse(other):
            if other.shape != self.shape:
                raise ValueError("inconsistent shapes")
            return self._sub_sparse(other)
        elif isdense(other):
            other = np.broadcast_to(other, self.shape)
            return self._sub_dense(other)
        else:
            return NotImplemented

    # 定义反向减法操作符重载方法，实现标量与稀疏矩阵的减法
    def __rsub__(self,other):  # other - self
        if isscalarlike(other):
            if other == 0:
                return -self.copy()
            raise NotImplementedError('subtracting a sparse array from a '
                                      'nonzero scalar is not supported')
        elif isdense(other):
            other = np.broadcast_to(other, self.shape)
            return self._rsub_dense(other)
        else:
            return NotImplemented

    # 定义乘法操作符重载方法，返回当前对象与另一个对象的乘积
    def __mul__(self, other):
        return self.multiply(other)

    # 定义反向乘法操作符重载方法，返回当前对象与另一个对象的乘积
    def __rmul__(self, other):  # other * self
        return self.multiply(other)

    # 默认使用 CSR 格式处理乘法操作的方法
    def _mul_scalar(self, other):
        # 将当前对象转换为 CSR 格式后，调用其与标量相乘的方法
        return self.tocsr()._mul_scalar(other)

    # 定义向量矩阵乘法的方法
    def _matmul_vector(self, other):
        # 将当前对象转换为 CSR 格式后，调用其与向量相乘的方法
        return self.tocsr()._matmul_vector(other)
    # 使用稀疏矩阵转换为 CSR 格式后执行多向量乘法
    def _matmul_multivector(self, other):
        return self.tocsr()._matmul_multivector(other)

    # 使用稀疏矩阵转换为 CSR 格式后执行稀疏矩阵乘法
    def _matmul_sparse(self, other):
        return self.tocsr()._matmul_sparse(other)

    # 根据输入类型分发右矩阵乘法操作
    def _rmatmul_dispatch(self, other):
        if isscalarlike(other):
            # 如果是标量，执行标量乘法
            return self._mul_scalar(other)
        else:
            # 否则，尝试对输入进行转置操作，若不可转置则转换为 NumPy 数组再进行转置
            try:
                tr = other.transpose()
            except AttributeError:
                tr = np.asarray(other).transpose()
            # 执行转置后的左矩阵乘法分发操作，并返回结果的转置
            ret = self.transpose()._matmul_dispatch(tr)
            if ret is NotImplemented:
                return NotImplemented
            return ret.transpose()

    #######################
    # matmul (@) operator #
    #######################

    # 定义 '@' 操作符重载方法
    def __matmul__(self, other):
        if isscalarlike(other):
            # 如果是标量，则抛出异常
            raise ValueError("Scalar operands are not allowed, "
                             "use '*' instead")
        # 否则，调用右矩阵乘法分发方法
        return self._matmul_dispatch(other)

    # 定义反向 '@' 操作符重载方法
    def __rmatmul__(self, other):
        if isscalarlike(other):
            # 如果是标量，则抛出异常
            raise ValueError("Scalar operands are not allowed, "
                             "use '*' instead")
        # 否则，调用右矩阵乘法分发方法
        return self._rmatmul_dispatch(other)

    ####################
    # Other Arithmetic #
    ####################

    # 执行除法操作
    def _divide(self, other, true_divide=False, rdivide=False):
        if isscalarlike(other):
            if rdivide:
                if true_divide:
                    # 如果需要真实除法，则执行真实除法操作
                    return np.true_divide(other, self.todense())
                else:
                    # 否则执行普通除法操作
                    return np.divide(other, self.todense())

            # 如果需要整除并且可以将当前类型转换为 np.float64，则执行类型转换后的乘法
            if true_divide and np.can_cast(self.dtype, np.float64):
                return self.astype(np.float64)._mul_scalar(1./other)
            else:
                # 否则执行普通乘法操作
                r = self._mul_scalar(1./other)

                # 确定标量的数据类型，并将结果转换为当前类型
                scalar_dtype = np.asarray(other).dtype
                if (np.issubdtype(self.dtype, np.integer) and
                        np.issubdtype(scalar_dtype, np.integer)):
                    return r.astype(self.dtype)
                else:
                    return r

        elif isdense(other):
            if not rdivide:
                if true_divide:
                    # 如果需要真实除法，则计算 reciprocal 并乘以 self
                    recip = np.true_divide(1., other)
                else:
                    # 否则执行普通除法操作
                    recip = np.divide(1., other)
                return self.multiply(recip)
            else:
                if true_divide:
                    # 如果需要真实除法，则执行真实除法操作
                    return np.true_divide(other, self.todense())
                else:
                    # 否则执行普通除法操作
                    return np.divide(other, self.todense())
        elif issparse(other):
            if rdivide:
                # 如果需要右矩阵乘法，则调用其 _divide 方法进行处理
                return other._divide(self, true_divide, rdivide=False)

            # 将当前矩阵转换为 CSR 格式后，执行稀疏矩阵除法操作
            self_csr = self.tocsr()
            if true_divide and np.can_cast(self.dtype, np.float64):
                return self_csr.astype(np.float64)._divide_sparse(other)
            else:
                return self_csr._divide_sparse(other)
        else:
            # 若无法处理，则返回 NotImplemented
            return NotImplemented
    def __truediv__(self, other):
        # 实现真实除法运算
        return self._divide(other, true_divide=True)

    def __div__(self, other):
        # 总是执行真实除法
        return self._divide(other, true_divide=True)

    def __rtruediv__(self, other):
        # 实现此方法作为其逆操作将过于神奇 —— 返回未实现
        return NotImplemented

    def __rdiv__(self, other):
        # 实现此方法作为其逆操作将过于神奇 —— 返回未实现
        return NotImplemented

    def __neg__(self):
        # 返回自身的负值的 CSR（压缩稀疏行）表示
        return -self.tocsr()

    def __iadd__(self, other):
        # 返回未实现，不支持原地加法操作
        return NotImplemented

    def __isub__(self, other):
        # 返回未实现，不支持原地减法操作
        return NotImplemented

    def __imul__(self, other):
        # 返回未实现，不支持原地乘法操作
        return NotImplemented

    def __idiv__(self, other):
        # 转发到 __itruediv__ 方法
        return self.__itruediv__(other)

    def __itruediv__(self, other):
        # 返回未实现，不支持原地真实除法操作
        return NotImplemented

    def __pow__(self, *args, **kwargs):
        # 调用 power 方法进行幂运算
        return self.power(*args, **kwargs)

    def transpose(self, axes=None, copy=False):
        """
        Reverses the dimensions of the sparse array/matrix.

        Parameters
        ----------
        axes : None, optional
            This argument is in the signature *solely* for NumPy
            compatibility reasons. Do not pass in anything except
            for the default value.
        copy : bool, optional
            Indicates whether or not attributes of `self` should be
            copied whenever possible. The degree to which attributes
            are copied varies depending on the type of sparse array/matrix
            being used.

        Returns
        -------
        p : `self` with the dimensions reversed.

        Notes
        -----
        If `self` is a `csr_array` or a `csc_array`, then this will return a
        `csc_array` or a `csr_array`, respectively.

        See Also
        --------
        numpy.transpose : NumPy's implementation of 'transpose' for ndarrays
        """
        # 调用 tocsr 方法转换为 CSR 格式后进行转置操作
        return self.tocsr(copy=copy).transpose(axes=axes, copy=False)

    def conjugate(self, copy=True):
        """Element-wise complex conjugation.

        If the array/matrix is of non-complex data type and `copy` is False,
        this method does nothing and the data is not copied.

        Parameters
        ----------
        copy : bool, optional
            If True, the result is guaranteed to not share data with self.

        Returns
        -------
        A : The element-wise complex conjugate.

        """
        # 如果数据类型为复数类型，则调用 tocsr 方法返回 CSR 格式的复共轭
        if np.issubdtype(self.dtype, np.complexfloating):
            return self.tocsr(copy=copy).conjugate(copy=False)
        # 否则，如果 copy 参数为 True，返回复制的 self；否则返回 self
        elif copy:
            return self.copy()
        else:
            return self

    def conj(self, copy=True):
        # 调用 conjugate 方法进行复共轭操作
        return self.conjugate(copy=copy)

    # 将 conj 方法的文档字符串设置为 conjugate 方法的文档字符串
    conj.__doc__ = conjugate.__doc__

    def _real(self):
        # 返回自身的 CSR 格式的实部
        return self.tocsr()._real()

    def _imag(self):
        # 返回自身的 CSR 格式的虚部
        return self.tocsr()._imag()
    def nonzero(self):
        """Nonzero indices of the array/matrix.

        Returns a tuple of arrays (row,col) containing the indices
        of the non-zero elements of the array.

        Examples
        --------
        >>> from scipy.sparse import csr_array
        >>> A = csr_array([[1,2,0],[0,0,3],[4,0,5]])
        >>> A.nonzero()
        (array([0, 0, 1, 2, 2]), array([0, 1, 2, 0, 2]))

        """

        # 将稀疏矩阵转换为 COOrdinate 格式
        A = self.tocoo()
        # 创建一个布尔掩码，标识非零元素的位置
        nz_mask = A.data != 0
        # 返回包含非零元素索引的元组 (行索引数组, 列索引数组)
        return tuple(idx[nz_mask] for idx in A.coords)

    def _getcol(self, j):
        """Returns a copy of column j of the array, as an (m x 1) sparse
        array (column vector).
        """
        if self.ndim == 1:
            raise ValueError("getcol not provided for 1d arrays. Use indexing A[j]")
        # 子类应该重写此方法以提高效率。
        # 用一个 (n x 1) 列向量 'a' 后乘，该向量除了 a_j = 1 外全为零
        N = self.shape[-1]
        if j < 0:
            j += N
        if j < 0 or j >= N:
            raise IndexError("index out of bounds")
        # 创建一个列选择器，选择第 j 列
        col_selector = self._csc_container(([1], [[j], [0]]),
                                           shape=(N, 1), dtype=self.dtype)
        # 返回结果，self 与列选择器的乘积
        result = self @ col_selector
        return result

    def _getrow(self, i):
        """Returns a copy of row i of the array, as a (1 x n) sparse
        array (row vector).
        """
        if self.ndim == 1:
            raise ValueError("getrow not meaningful for a 1d array")
        # 子类应该重写此方法以提高效率。
        # 用一个 (1 x m) 行向量 'a' 前乘，该向量除了 a_i = 1 外全为零
        M = self.shape[0]
        if i < 0:
            i += M
        if i < 0 or i >= M:
            raise IndexError("index out of bounds")
        # 创建一个行选择器，选择第 i 行
        row_selector = self._csr_container(([1], [[0], [i]]),
                                           shape=(1, M), dtype=self.dtype)
        # 返回结果，行选择器与 self 的乘积
        return row_selector @ self
    #     # See gh-10362 for details.

    def todense(self, order=None, out=None):
        """
        Return a dense representation of this sparse array.

        Parameters
        ----------
        order : {'C', 'F'}, optional
            Whether to store multi-dimensional data in C (row-major)
            or Fortran (column-major) order in memory. The default
            is 'None', which provides no ordering guarantees.
            Cannot be specified in conjunction with the `out`
            argument.

        out : ndarray, 2-D, optional
            If specified, uses this array as the output buffer
            instead of allocating a new array to return. The
            provided array must have the same shape and dtype as
            the sparse array on which you are calling the method.

        Returns
        -------
        arr : ndarray, 2-D
            An array with the same shape and containing the same
            data represented by the sparse array, with the requested
            memory order. If `out` was passed, the same object is
            returned after being modified in-place to contain the
            appropriate values.
        """
        # 调用 `toarray` 方法获取稀疏数组的密集表示，并用 `_ascontainer` 包装后返回
        return self._ascontainer(self.toarray(order=order, out=out))

    def toarray(self, order=None, out=None):
        """
        Return a dense ndarray representation of this sparse array/matrix.

        Parameters
        ----------
        order : {'C', 'F'}, optional
            Whether to store multidimensional data in C (row-major)
            or Fortran (column-major) order in memory. The default
            is 'None', which provides no ordering guarantees.
            Cannot be specified in conjunction with the `out`
            argument.

        out : ndarray, 2-D, optional
            If specified, uses this array as the output buffer
            instead of allocating a new array to return. The provided
            array must have the same shape and dtype as the sparse
            array/matrix on which you are calling the method. For most
            sparse types, `out` is required to be memory contiguous
            (either C or Fortran ordered).

        Returns
        -------
        arr : ndarray, 2-D
            An array with the same shape and containing the same
            data represented by the sparse array/matrix, with the requested
            memory order. If `out` was passed, the same object is
            returned after being modified in-place to contain the
            appropriate values.
        """
        # 将稀疏数组转换为 COO 格式，并调用其 `toarray` 方法返回密集表示
        return self.tocoo(copy=False).toarray(order=order, out=out)

    # Any sparse array format deriving from _spbase must define one of
    # tocsr or tocoo. The other conversion methods may be implemented for
    # efficiency, but are not required.
    def tocsr(self, copy=False):
        """
        Convert this array/matrix to Compressed Sparse Row format.

        With copy=False, the data/indices may be shared between this array/matrix and
        the resultant csr_array/matrix.
        """
        # 调用 tocoo 方法将数组/矩阵转换为 COO 格式，然后将其转换为 CSR 格式
        return self.tocoo(copy=copy).tocsr(copy=False)

    def todok(self, copy=False):
        """
        Convert this array/matrix to Dictionary Of Keys format.

        With copy=False, the data/indices may be shared between this array/matrix and
        the resultant dok_array/matrix.
        """
        # 调用 tocoo 方法将数组/矩阵转换为 COO 格式，然后将其转换为 DOK 格式
        return self.tocoo(copy=copy).todok(copy=False)

    def tocoo(self, copy=False):
        """
        Convert this array/matrix to COOrdinate format.

        With copy=False, the data/indices may be shared between this array/matrix and
        the resultant coo_array/matrix.
        """
        # 直接返回数组/矩阵的 COO 格式表示
        return self.tocsr(copy=False).tocoo(copy=copy)

    def tolil(self, copy=False):
        """
        Convert this array/matrix to List of Lists format.

        With copy=False, the data/indices may be shared between this array/matrix and
        the resultant lil_array/matrix.
        """
        # 调用 tocsr 方法将数组/矩阵转换为 CSR 格式，然后将其转换为 LIL 格式
        return self.tocsr(copy=False).tolil(copy=copy)

    def todia(self, copy=False):
        """
        Convert this array/matrix to sparse DIAgonal format.

        With copy=False, the data/indices may be shared between this array/matrix and
        the resultant dia_array/matrix.
        """
        # 调用 tocoo 方法将数组/矩阵转换为 COO 格式，然后将其转换为 DIA 格式
        return self.tocoo(copy=copy).todia(copy=False)

    def tobsr(self, blocksize=None, copy=False):
        """
        Convert this array/matrix to Block Sparse Row format.

        With copy=False, the data/indices may be shared between this array/matrix and
        the resultant bsr_array/matrix.

        When blocksize=(R, C) is provided, it will be used for construction of
        the bsr_array/matrix.
        """
        # 调用 tocsr 方法将数组/矩阵转换为 CSR 格式，然后将其转换为 BSR 格式
        return self.tocsr(copy=False).tobsr(blocksize=blocksize, copy=copy)

    def tocsc(self, copy=False):
        """
        Convert this array/matrix to Compressed Sparse Column format.

        With copy=False, the data/indices may be shared between this array/matrix and
        the resultant csc_array/matrix.
        """
        # 调用 tocsr 方法将数组/矩阵转换为 CSR 格式，然后将其转换为 CSC 格式
        return self.tocsr(copy=copy).tocsc(copy=False)

    def copy(self):
        """
        Returns a copy of this array/matrix.

        No data/indices will be shared between the returned value and current
        array/matrix.
        """
        # 创建当前数组/矩阵的深拷贝并返回
        return self.__class__(self, copy=True)
    def sum(self, axis=None, dtype=None, out=None):
        """
        Sum the array/matrix elements over a given axis.

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None} optional
            Axis along which the sum is computed. The default is to
            compute the sum of all the array/matrix elements, returning a scalar
            (i.e., `axis` = `None`).
        dtype : dtype, optional
            The type of the returned array/matrix and of the accumulator in which
            the elements are summed.  The dtype of `a` is used by default
            unless `a` has an integer dtype of less precision than the default
            platform integer.  In that case, if `a` is signed then the platform
            integer is used while if `a` is unsigned then an unsigned integer
            of the same precision as the platform integer is used.

            .. versionadded:: 0.18.0

        out : np.matrix, optional
            Alternative output matrix in which to place the result. It must
            have the same shape as the expected output, but the type of the
            output values will be cast if necessary.

            .. versionadded:: 0.18.0

        Returns
        -------
        sum_along_axis : np.matrix
            A matrix with the same shape as `self`, with the specified
            axis removed.

        See Also
        --------
        numpy.matrix.sum : NumPy's implementation of 'sum' for matrices

        """

        # Validate axis parameter
        validateaxis(axis)

        # Determine the dtype of the result based on the input dtype
        res_dtype = get_sum_dtype(self.dtype)

        # Handle 1-dimensional arrays
        if self.ndim == 1:
            if axis not in (None, -1, 0):
                raise ValueError("axis must be None, -1 or 0")
            # Calculate sum along the axis using matrix multiplication with a row of ones
            ret = (self @ np.ones(self.shape, dtype=res_dtype)).astype(dtype)

            # Handle optional output array 'out'
            if out is not None:
                if any(dim != 1 for dim in out.shape):
                    raise ValueError("dimensions do not match")
                out[...] = ret
            return ret

        # For 2-dimensional arrays (matrices)
        M, N = self.shape

        if axis is None:
            # Sum over both rows and columns
            return (self @ self._ascontainer(np.ones((N, 1), dtype=res_dtype))).sum(dtype=dtype, out=out)

        if axis < 0:
            axis += 2

        # Handle summing along axis 0 or axis 1
        if axis == 0:
            # Sum over columns
            ret = self._ascontainer(np.ones((1, M), dtype=res_dtype)) @ self
        else:
            # Sum over rows
            ret = self @ self._ascontainer(np.ones((N, 1), dtype=res_dtype))

        # Check if 'out' parameter is provided and dimensions match
        if out is not None and out.shape != ret.shape:
            raise ValueError("dimensions do not match")

        return ret.sum(axis=axis, dtype=dtype, out=out)
    # 定义一个方法，计算沿指定轴的算术平均值
    def mean(self, axis=None, dtype=None, out=None):
        """
        Compute the arithmetic mean along the specified axis.

        Returns the average of the array/matrix elements. The average is taken
        over all elements in the array/matrix by default, otherwise over the
        specified axis. `float64` intermediate and return values are used
        for integer inputs.

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None} optional
            Axis along which the mean is computed. The default is to compute
            the mean of all elements in the array/matrix (i.e., `axis` = `None`).
        dtype : data-type, optional
            Type to use in computing the mean. For integer inputs, the default
            is `float64`; for floating point inputs, it is the same as the
            input dtype.

            .. versionadded:: 0.18.0

        out : np.matrix, optional
            Alternative output matrix in which to place the result. It must
            have the same shape as the expected output, but the type of the
            output values will be cast if necessary.

            .. versionadded:: 0.18.0

        Returns
        -------
        m : np.matrix
            Mean value along the specified axis.

        See Also
        --------
        numpy.matrix.mean : NumPy's implementation of 'mean' for matrices

        """
        # 调用辅助函数，验证轴参数的有效性
        validateaxis(axis)

        # 确定结果的数据类型
        res_dtype = self.dtype.type
        integral = (np.issubdtype(self.dtype, np.integer) or
                    np.issubdtype(self.dtype, np.bool_))

        # 根据输入数据类型确定输出数据类型
        if dtype is None:
            if integral:
                res_dtype = np.float64
        else:
            res_dtype = np.dtype(dtype).type

        # 中间求和过程的数据类型
        inter_dtype = np.float64 if integral else res_dtype
        inter_self = self.astype(inter_dtype)

        # 如果数组是一维的
        if self.ndim == 1:
            if axis not in (None, -1, 0):
                raise ValueError("axis must be None, -1 or 0")
            # 计算一维数组的平均值并返回
            res = inter_self / self.shape[-1]
            return res.sum(dtype=res_dtype, out=out)

        # 如果 axis 参数为 None
        if axis is None:
            # 计算整个数组的平均值并返回
            return (inter_self / (self.shape[0] * self.shape[1]))\
                .sum(dtype=res_dtype, out=out)

        # 处理负数轴参数
        if axis < 0:
            axis += 2

        # 现在 axis 只能是 0 或 1
        if axis == 0:
            # 沿着行的方向计算平均值并返回
            return (inter_self * (1.0 / self.shape[0])).sum(
                axis=0, dtype=res_dtype, out=out)
        else:
            # 沿着列的方向计算平均值并返回
            return (inter_self * (1.0 / self.shape[1])).sum(
                axis=1, dtype=res_dtype, out=out)
    def diagonal(self, k=0):
        """
        Returns the kth diagonal of the array/matrix.

        Parameters
        ----------
        k : int, optional
            Which diagonal to get, corresponding to elements a[i, i+k].
            Default: 0 (the main diagonal).

            .. versionadded:: 1.0

        See also
        --------
        numpy.diagonal : Equivalent numpy function.

        Examples
        --------
        >>> from scipy.sparse import csr_array
        >>> A = csr_array([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
        >>> A.diagonal()
        array([1, 0, 5])
        >>> A.diagonal(k=1)
        array([2, 3])
        """
        # 调用 tocsr 方法将稀疏矩阵转换为 CSR 格式，并获取其第 k 条对角线上的元素
        return self.tocsr().diagonal(k=k)

    def trace(self, offset=0):
        """
        Returns the sum along diagonals of the sparse array/matrix.

        Parameters
        ----------
        offset : int, optional
            Which diagonal to get, corresponding to elements a[i, i+offset].
            Default: 0 (the main diagonal).

        """
        # 调用 diagonal 方法获取第 offset 条对角线上的元素，并返回它们的总和
        return self.diagonal(k=offset).sum()

    def setdiag(self, values, k=0):
        """
        Set diagonal or off-diagonal elements of the array/matrix.

        Parameters
        ----------
        values : array_like
            New values of the diagonal elements.

            Values may have any length. If the diagonal is longer than values,
            then the remaining diagonal entries will not be set. If values are
            longer than the diagonal, then the remaining values are ignored.

            If a scalar value is given, all of the diagonal is set to it.

        k : int, optional
            Which off-diagonal to set, corresponding to elements a[i,i+k].
            Default: 0 (the main diagonal).

        """
        M, N = self.shape
        # 检查 k 是否超出数组维度，如果是则抛出 ValueError 异常
        if (k > 0 and k >= N) or (k < 0 and -k >= M):
            raise ValueError("k exceeds array dimensions")
        # 调用内部方法 _setdiag 来设置对角线或非对角线元素
        self._setdiag(np.asarray(values), k)

    def _setdiag(self, values, k):
        """
        This part of the implementation gets overridden by the
        different formats.
        """
        M, N = self.shape
        if k < 0:
            if values.ndim == 0:
                # 广播操作，将单个值 values 设置到对角线上
                max_index = min(M+k, N)
                for i in range(max_index):
                    self[i - k, i] = values
            else:
                # 将 values 数组中的值设置到对角线上
                max_index = min(M+k, N, len(values))
                if max_index <= 0:
                    return
                for i, v in enumerate(values[:max_index]):
                    self[i - k, i] = v
        else:
            if values.ndim == 0:
                # 广播操作，将单个值 values 设置到对角线上
                max_index = min(M, N-k)
                for i in range(max_index):
                    self[i, i + k] = values
            else:
                # 将 values 数组中的值设置到对角线上
                max_index = min(M, N-k, len(values))
                if max_index <= 0:
                    return
                for i, v in enumerate(values[:max_index]):
                    self[i, i + k] = v
    # 处理稀疏数组的转换为数组的参数
    def _process_toarray_args(self, order, out):
        # 如果输出数组 out 不为空
        if out is not None:
            # 如果指定了 order 参数，则抛出数值错误异常
            if order is not None:
                raise ValueError('order cannot be specified if out '
                                 'is not None')
            # 如果输出数组的形状或数据类型与稀疏数组 self 不匹配，则抛出数值错误异常
            if out.shape != self.shape or out.dtype != self.dtype:
                raise ValueError('out array must be same dtype and shape as '
                                 'sparse array')
            # 将输出数组 out 的所有元素设为 0
            out[...] = 0.
            # 返回输出数组 out
            return out
        else:
            # 如果输出数组 out 为空，则创建一个新的全零数组，形状和数据类型与稀疏数组 self 相同
            return np.zeros(self.shape, dtype=self.dtype, order=order)

    # 获取适合数组的索引数据类型
    def _get_index_dtype(self, arrays=(), maxval=None, check_contents=False):
        """
        Determine index dtype for array.

        This wraps _sputils.get_index_dtype, providing compatibility for both
        array and matrix API sparse matrices. Matrix API sparse matrices would
        attempt to downcast the indices - which can be computationally
        expensive and undesirable for users. The array API changes this
        behaviour.

        See discussion: https://github.com/scipy/scipy/issues/16774

        The get_index_dtype import is due to implementation details of the test
        suite. It allows the decorator ``with_64bit_maxval_limit`` to mock a
        lower int32 max value for checks on the matrix API's downcasting
        behaviour.
        """
        # 导入 _sputils 模块中的 get_index_dtype 函数
        from ._sputils import get_index_dtype

        # 对于数组 API，不检查内容
        return get_index_dtype(arrays,
                               maxval,
                               (check_contents and not isinstance(self, sparray)))
# 定义一个命名空间类，用于区分稀疏数组（sparray）和稀疏矩阵（spmatrix）
class sparray:
    """A namespace class to separate sparray from spmatrix"""


# 将 sparray 类的文档字符串设置为 _spbase 类的文档字符串
sparray.__doc__ = _spbase.__doc__


def issparse(x):
    """Is `x` of a sparse array or sparse matrix type?

    Parameters
    ----------
    x
        object to check for being a sparse array or sparse matrix

    Returns
    -------
    bool
        True if `x` is a sparse array or a sparse matrix, False otherwise

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csr_array, csr_matrix, issparse
    >>> issparse(csr_matrix([[5]]))
    True
    >>> issparse(csr_array([[5]]))
    True
    >>> issparse(np.array([[5]]))
    False
    >>> issparse(5)
    False
    """
    # 检查对象 x 是否为 _spbase 类型的实例，即稀疏数组或稀疏矩阵
    return isinstance(x, _spbase)


def isspmatrix(x):
    """Is `x` of a sparse matrix type?

    Parameters
    ----------
    x
        object to check for being a sparse matrix

    Returns
    -------
    bool
        True if `x` is a sparse matrix, False otherwise

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csr_array, csr_matrix, isspmatrix
    >>> isspmatrix(csr_matrix([[5]]))
    True
    >>> isspmatrix(csr_array([[5]]))
    False
    >>> isspmatrix(np.array([[5]]))
    False
    >>> isspmatrix(5)
    False
    """
    # 检查对象 x 是否为 spmatrix 类型的实例，即稀疏矩阵
    return isinstance(x, spmatrix)
```