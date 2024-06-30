# `D:\src\scipysrc\scipy\scipy\sparse\_data.py`

```
"""Base class for sparse matrice with a .data attribute

    subclasses must provide a _with_data() method that
    creates a new matrix with the same sparsity pattern
    as self but with a different data array

"""

import math  # 导入 math 模块
import numpy as np  # 导入 numpy 库

from ._base import _spbase, sparray, _ufuncs_with_fixed_point_at_zero  # 导入自定义模块和类
from ._sputils import isscalarlike, validateaxis  # 导入自定义工具函数

__all__ = []  # 初始化空的 __all__ 列表，用于定义模块的公开接口

# TODO implement all relevant operations
# use .data.__methods__() instead of /=, *=, etc.
class _data_matrix(_spbase):
    def __init__(self, arg1, *, maxprint=None):
        _spbase.__init__(self, arg1, maxprint=maxprint)  # 调用基类 _spbase 的初始化方法

    @property
    def dtype(self):
        return self.data.dtype  # 返回存储在 self.data 中的数据类型

    @dtype.setter
    def dtype(self, newtype):
        self.data.dtype = newtype  # 设置 self.data 的数据类型为 newtype

    def _deduped_data(self):
        if hasattr(self, 'sum_duplicates'):
            self.sum_duplicates()  # 如果存在 sum_duplicates 方法，则调用它
        return self.data  # 返回 self.data 的数据

    def __abs__(self):
        return self._with_data(abs(self._deduped_data()))  # 返回一个新的对象，其数据为 self._deduped_data() 的绝对值

    def __round__(self, ndigits=0):
        return self._with_data(np.around(self._deduped_data(), decimals=ndigits))  # 返回一个新对象，其数据为 self._deduped_data() 的四舍五入值

    def _real(self):
        return self._with_data(self.data.real)  # 返回一个新对象，其数据为 self.data 的实部

    def _imag(self):
        return self._with_data(self.data.imag)  # 返回一个新对象，其数据为 self.data 的虚部

    def __neg__(self):
        if self.dtype.kind == 'b':
            raise NotImplementedError('negating a boolean sparse array is not '
                                      'supported')  # 如果数据类型为布尔类型，则抛出未实现错误
        return self._with_data(-self.data)  # 返回一个新对象，其数据为 self.data 的相反数

    def __imul__(self, other):  # self *= other
        if isscalarlike(other):  # 如果 other 是标量类型
            self.data *= other  # 将 self.data 乘以 other
            return self  # 返回自身对象
        return NotImplemented  # 否则返回未实现错误

    def __itruediv__(self, other):  # self /= other
        if isscalarlike(other):  # 如果 other 是标量类型
            recip = 1.0 / other  # 计算 other 的倒数
            self.data *= recip  # 将 self.data 乘以 recip
            return self  # 返回自身对象
        else:
            return NotImplemented  # 否则返回未实现错误

    def astype(self, dtype, casting='unsafe', copy=True):
        dtype = np.dtype(dtype)  # 将 dtype 转换为 numpy 的数据类型对象
        if self.dtype != dtype:  # 如果当前数据类型不等于指定的 dtype
            matrix = self._with_data(
                self.data.astype(dtype, casting=casting, copy=True),  # 使用指定的 dtype 将 self.data 转换为新的数据对象
                copy=True
            )
            return matrix._with_data(matrix._deduped_data(), copy=False)  # 返回转换后的新对象，不复制数据
        elif copy:
            return self.copy()  # 返回当前对象的复制
        else:
            return self  # 否则返回当前对象本身

    astype.__doc__ = _spbase.astype.__doc__  # 设置 astype 方法的文档字符串

    def conjugate(self, copy=True):
        if np.issubdtype(self.dtype, np.complexfloating):  # 如果数据类型为复数类型
            return self._with_data(self.data.conjugate(), copy=copy)  # 返回一个新对象，其数据为 self.data 的共轭
        elif copy:
            return self.copy()  # 返回当前对象的复制
        else:
            return self  # 否则返回当前对象本身

    conjugate.__doc__ = _spbase.conjugate.__doc__  # 设置 conjugate 方法的文档字符串

    def copy(self):
        return self._with_data(self.data.copy(), copy=True)  # 返回一个当前对象数据的复制

    copy.__doc__ = _spbase.copy.__doc__  # 设置 copy 方法的文档字符串
    def power(self, n, dtype=None):
        """
        This function performs element-wise power.

        Parameters
        ----------
        n : scalar
            n is a non-zero scalar (nonzero avoids dense ones creation)
            If zero power is desired, special case it to use `np.ones`

        dtype : If dtype is not specified, the current dtype will be preserved.

        Raises
        ------
        NotImplementedError : if n is a zero scalar
            If zero power is desired, special case it to use
            ``np.ones(A.shape, dtype=A.dtype)``
        """
        # 检查输入的 n 是否为标量
        if not isscalarlike(n):
            raise NotImplementedError("input is not scalar")
        # 如果 n 为零，则抛出错误，不支持零次幂，建议使用 `np.ones` 替代
        if not n:
            raise NotImplementedError(
                "zero power is not supported as it would densify the matrix.\n"
                "Use `np.ones(A.shape, dtype=A.dtype)` for this case."
            )

        # 获取对象的去重数据
        data = self._deduped_data()
        # 如果指定了 dtype，则将数据转换为指定的数据类型
        if dtype is not None:
            data = data.astype(dtype)
        # 返回对数据进行 n 次幂操作后的新对象
        return self._with_data(data ** n)

    ###########################
    # Multiplication handlers #
    ###########################

    def _mul_scalar(self, other):
        # 返回使用标量乘法后的新对象
        return self._with_data(self.data * other)
# 为 _ufuncs_with_fixed_point_at_zero 中的每个 numpy 一元 ufunc 添加方法到 _data_matrix。
for npfunc in _ufuncs_with_fixed_point_at_zero:
    name = npfunc.__name__

    def _create_method(op):
        # 创建一个方法，对 _deduped_data() 执行 op 操作，并返回新的 _data_matrix 对象
        def method(self):
            result = op(self._deduped_data())
            return self._with_data(result, copy=True)

        # 设置方法的文档字符串，描述其为逐元素操作
        method.__doc__ = (f"Element-wise {name}.\n\n"
                          f"See `numpy.{name}` for more information.")
        # 设置方法的名称
        method.__name__ = name

        return method

    # 将创建的方法设置为 _data_matrix 的属性，方法名为 npfunc.__name__
    setattr(_data_matrix, name, _create_method(npfunc))


# 查找缺失的索引位置
def _find_missing_index(ind, n):
    for k, a in enumerate(ind):
        if k != a:
            return k

    # 增加 k 的值，继续查找可能的缺失位置
    k += 1
    if k < n:
        return k
    else:
        return -1


# _minmax_mixin 类，用于混合最小值和最大值方法
class _minmax_mixin:
    """Mixin for min and max methods.

    These are not implemented for dia_matrix, hence the separate class.
    """

    # 根据轴向执行最小值或最大值操作
    def _min_or_max_axis(self, axis, min_or_max):
        N = self.shape[axis]
        # 若数组尺寸为零，则引发 ValueError 异常
        if N == 0:
            raise ValueError("zero-size array to reduction operation")
        M = self.shape[1 - axis]
        # 获取索引数据类型
        idx_dtype = self._get_index_dtype(maxval=M)

        # 将 dia_matrix 转换为 csc_matrix 或 csr_matrix
        mat = self.tocsc() if axis == 0 else self.tocsr()
        mat.sum_duplicates()

        # 执行次要减少操作，获取主索引和值
        major_index, value = mat._minor_reduce(min_or_max)
        not_full = np.diff(mat.indptr)[major_index] < N
        value[not_full] = min_or_max(value[not_full], 0)

        # 创建值的布尔掩码
        mask = value != 0
        major_index = np.compress(mask, major_index)
        value = np.compress(mask, value)

        # 如果 self 是 sparray 类的实例
        if isinstance(self, sparray):
            coords = (major_index,)
            shape = (M,)
            return self._coo_container((value, coords), shape=shape, dtype=self.dtype)

        # 根据轴向返回 _coo_container 对象
        if axis == 0:
            return self._coo_container(
                (value, (np.zeros(len(value), dtype=idx_dtype), major_index)),
                dtype=self.dtype, shape=(1, M)
            )
        else:
            return self._coo_container(
                (value, (major_index, np.zeros(len(value), dtype=idx_dtype))),
                dtype=self.dtype, shape=(M, 1)
            )
    # 定义一个函数，用于在稀疏数组上执行最小值或最大值操作，支持指定轴向的操作结果
    def _min_or_max(self, axis, out, min_or_max):
        # 如果指定了输出参数，则抛出异常，稀疏数组不支持输出参数
        if out is not None:
            raise ValueError("Sparse arrays do not support an 'out' parameter.")

        # 验证轴向参数的有效性
        validateaxis(axis)

        # 如果数组是一维的
        if self.ndim == 1:
            # 如果指定的轴不是 None、0 或 -1，抛出异常
            if axis not in (None, 0, -1):
                raise ValueError("axis out of range")
            axis = None  # 避免调用特殊的一维轴向情况，对一维数组无影响

        # 如果 axis 为 None
        if axis is None:
            # 如果数组的任意一个维度为零，则抛出异常
            if 0 in self.shape:
                raise ValueError("zero-size array to reduction operation")

            # 使用数组元素类型的零值
            zero = self.dtype.type(0)

            # 如果数组没有非零元素，则返回零值
            if self.nnz == 0:
                return zero

            # 对去重后的数据进行扁平化后执行最小值或最大值操作
            m = min_or_max.reduce(self._deduped_data().ravel())

            # 如果稀疏数组的非零元素个数不等于数组元素总数，则再次比较结果与零值
            if self.nnz != math.prod(self.shape):
                m = min_or_max(zero, m)

            # 返回最终的最小值或最大值结果
            return m

        # 如果 axis 小于 0，则进行调整
        if axis < 0:
            axis += 2

        # 如果 axis 为 0 或 1
        if (axis == 0) or (axis == 1):
            # 调用内部函数，对指定轴向执行最小值或最大值操作
            return self._min_or_max_axis(axis, min_or_max)
        else:
            # 如果 axis 超出范围，抛出异常
            raise ValueError("axis out of range")

    # 定义一个函数，用于在稀疏数组的指定轴向上执行最小值或最大值操作，返回结果的索引
    def _arg_min_or_max_axis(self, axis, argmin_or_argmax, compare):
        # 如果数组在指定轴向上的大小为 0，则抛出异常
        if self.shape[axis] == 0:
            raise ValueError("Cannot apply the operation along a zero-sized dimension.")

        # 如果 axis 小于 0，则进行调整
        if axis < 0:
            axis += 2

        # 使用数组元素类型的零值
        zero = self.dtype.type(0)

        # 如果 axis 为 0，则转换为列压缩格式 (CSC)；否则转换为行压缩格式 (CSR)
        mat = self.tocsc() if axis == 0 else self.tocsr()

        # 移除重复条目
        mat.sum_duplicates()

        # 交换矩阵的形状信息
        ret_size, line_size = mat._swap(mat.shape)

        # 创建一个整数类型的全零数组作为返回结果
        ret = np.zeros(ret_size, dtype=int)

        # 找到非零行的索引
        nz_lines, = np.nonzero(np.diff(mat.indptr))

        # 遍历非零行
        for i in nz_lines:
            # 获取当前行的指针范围
            p, q = mat.indptr[i:i + 2]

            # 获取当前行的数据和对应的列索引
            data = mat.data[p:q]
            indices = mat.indices[p:q]

            # 执行最小值或最大值索引操作
            extreme_index = argmin_or_argmax(data)

            # 获取最小值或最大值
            extreme_value = data[extreme_index]

            # 如果比较结果为真或当前行的元素数等于行的大小
            if compare(extreme_value, zero) or q - p == line_size:
                ret[i] = indices[extreme_index]
            else:
                # 查找缺失索引
                zero_ind = _find_missing_index(indices, line_size)

                # 如果极值等于零
                if extreme_value == zero:
                    ret[i] = min(extreme_index, zero_ind)
                else:
                    ret[i] = zero_ind

        # 如果数组是 sparray 类型，则直接返回结果
        if isinstance(self, sparray):
            return ret

        # 如果 axis 为 1，则将结果重新形状为列向量
        if axis == 1:
            ret = ret.reshape(-1, 1)

        # 返回包装后的结果
        return self._ascontainer(ret)
    # 求取稀疏矩阵沿指定轴的最小或最大值的索引，不支持 'out' 参数
    def _arg_min_or_max(self, axis, out, argmin_or_argmax, compare):
        # 如果指定了 'out' 参数，则抛出异常
        if out is not None:
            raise ValueError("Sparse types do not support an 'out' parameter.")

        # 验证轴的有效性
        validateaxis(axis)

        # 如果矩阵是一维的，且指定了轴，则检查轴是否在有效范围内
        if self.ndim == 1:
            if axis not in (None, 0, -1):
                raise ValueError("axis out of range")
            axis = None  # 避免调用特殊的轴情况，对一维矩阵无影响

        # 如果指定了轴，则调用对应轴上的最小或最大值计算方法
        if axis is not None:
            return self._arg_min_or_max_axis(axis, argmin_or_argmax, compare)

        # 如果矩阵的形状中有任一维度为0，则无法应用操作
        if 0 in self.shape:
            raise ValueError("Cannot apply the operation to an empty matrix.")

        # 如果矩阵中没有非零元素，则直接返回0
        if self.nnz == 0:
            return 0

        # 零元素的表示
        zero = self.dtype.type(0)
        # 将矩阵转换为COO格式，确保没有重复的元素并且索引已排序
        mat = self.tocoo()
        mat.sum_duplicates()
        # 找到最小或最大值的索引
        extreme_index = argmin_or_argmax(mat.data)
        extreme_value = mat.data[extreme_index]
        num_col = mat.shape[-1]

        # 如果最小值小于零或最大值大于零，则不需要担心隐式零元素
        if compare(extreme_value, zero):
            # 将行和列索引线性化为一个整数索引，以避免溢出和运行时错误
            return int(mat.row[extreme_index]) * num_col + int(mat.col[extreme_index])

        # 对于极少数情况进行便宜测试，矩阵中没有隐式零元素
        size = math.prod(self.shape)
        if size == mat.nnz:
            return int(mat.row[extreme_index]) * num_col + int(mat.col[extreme_index])

        # 此时，任何隐式零元素都可能是最小或最大值
        # 在 sum_duplicates() 后，'row' 和 'col' 数组保证以 C 顺序排序，即线性化索引是有序的
        linear_indices = mat.row * num_col + mat.col
        # 查找缺失的索引，即隐式零元素的首个位置
        first_implicit_zero_index = _find_missing_index(linear_indices, size)
        # 如果最值是零，则返回最小的索引
        if extreme_value == zero:
            return min(first_implicit_zero_index, extreme_index)
        # 否则返回第一个隐式零元素的索引
        return first_implicit_zero_index
    def max(self, axis=None, out=None):
        """
        Return the maximum of the array/matrix or maximum along an axis.
        This takes all elements into account, not just the non-zero ones.

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None} optional
            Axis along which the maximum is computed. The default is to
            compute the maximum over all elements, returning
            a scalar (i.e., `axis` = `None`).

        out : None, optional
            This argument is in the signature *solely* for NumPy
            compatibility reasons. Do not pass in anything except for
            the default value, as this argument is not used.

        Returns
        -------
        amax : coo_matrix or scalar
            Maximum of `a`. If `axis` is None, the result is a scalar value.
            If `axis` is given, the result is a sparse.coo_matrix of dimension
            ``a.ndim - 1``.

        See Also
        --------
        min : The minimum value of a sparse array/matrix along a given axis.
        numpy.matrix.max : NumPy's implementation of 'max' for matrices

        """
        # 调用私有方法 _min_or_max，传入 np.maximum 函数，以实现最大值计算
        return self._min_or_max(axis, out, np.maximum)

    def min(self, axis=None, out=None):
        """
        Return the minimum of the array/matrix or maximum along an axis.
        This takes all elements into account, not just the non-zero ones.

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None} optional
            Axis along which the minimum is computed. The default is to
            compute the minimum over all elements, returning
            a scalar (i.e., `axis` = `None`).

        out : None, optional
            This argument is in the signature *solely* for NumPy
            compatibility reasons. Do not pass in anything except for
            the default value, as this argument is not used.

        Returns
        -------
        amin : coo_matrix or scalar
            Minimum of `a`. If `axis` is None, the result is a scalar value.
            If `axis` is given, the result is a sparse.coo_matrix of dimension
            ``a.ndim - 1``.

        See Also
        --------
        max : The maximum value of a sparse array/matrix along a given axis.
        numpy.matrix.min : NumPy's implementation of 'min' for matrices

        """
        # 调用私有方法 _min_or_max，传入 np.minimum 函数，以实现最小值计算
        return self._min_or_max(axis, out, np.minimum)
    def nanmax(self, axis=None, out=None):
        """
        Return the maximum of the array/matrix or maximum along an axis, ignoring any
        NaNs. This takes all elements into account, not just the non-zero
        ones.

        .. versionadded:: 1.11.0

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None} optional
            Axis along which the maximum is computed. The default is to
            compute the maximum over all elements, returning
            a scalar (i.e., `axis` = `None`).

        out : None, optional
            This argument is in the signature *solely* for NumPy
            compatibility reasons. Do not pass in anything except
            for the default value, as this argument is not used.

        Returns
        -------
        amax : coo_matrix or scalar
            Maximum of `a`. If `axis` is None, the result is a scalar value.
            If `axis` is given, the result is a sparse.coo_matrix of dimension
            ``a.ndim - 1``.

        See Also
        --------
        nanmin : The minimum value of a sparse array/matrix along a given axis,
                 ignoring NaNs.
        max : The maximum value of a sparse array/matrix along a given axis,
              propagating NaNs.
        numpy.nanmax : NumPy's implementation of 'nanmax'.

        """
        return self._min_or_max(axis, out, np.fmax)

    def nanmin(self, axis=None, out=None):
        """
        Return the minimum of the array/matrix or minimum along an axis, ignoring any
        NaNs. This takes all elements into account, not just the non-zero
        ones.

        .. versionadded:: 1.11.0

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None} optional
            Axis along which the minimum is computed. The default is to
            compute the minimum over all elements, returning
            a scalar (i.e., `axis` = `None`).

        out : None, optional
            This argument is in the signature *solely* for NumPy
            compatibility reasons. Do not pass in anything except for
            the default value, as this argument is not used.

        Returns
        -------
        amin : coo_matrix or scalar
            Minimum of `a`. If `axis` is None, the result is a scalar value.
            If `axis` is given, the result is a sparse.coo_matrix of dimension
            ``a.ndim - 1``.

        See Also
        --------
        nanmax : The maximum value of a sparse array/matrix along a given axis,
                 ignoring NaNs.
        min : The minimum value of a sparse array/matrix along a given axis,
              propagating NaNs.
        numpy.nanmin : NumPy's implementation of 'nanmin'.

        """
        return self._min_or_max(axis, out, np.fmin)
    def argmax(self, axis=None, out=None):
        """Return indices of maximum elements along an axis.

        Implicit zero elements are also taken into account. If there are
        several maximum values, the index of the first occurrence is returned.

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None}, optional
            Axis along which the argmax is computed. If None (default), index
            of the maximum element in the flatten data is returned.
        out : None, optional
            This argument is in the signature *solely* for NumPy
            compatibility reasons. Do not pass in anything except for
            the default value, as this argument is not used.

        Returns
        -------
        ind : numpy.matrix or int
            Indices of maximum elements. If matrix, its size along `axis` is 1.
        """
        # 调用内部方法 `_arg_min_or_max`，使用 `np.argmax` 函数找到最大元素的索引
        return self._arg_min_or_max(axis, out, np.argmax, np.greater)

    def argmin(self, axis=None, out=None):
        """Return indices of minimum elements along an axis.

        Implicit zero elements are also taken into account. If there are
        several minimum values, the index of the first occurrence is returned.

        Parameters
        ----------
        axis : {-2, -1, 0, 1, None}, optional
            Axis along which the argmin is computed. If None (default), index
            of the minimum element in the flatten data is returned.
        out : None, optional
            This argument is in the signature *solely* for NumPy
            compatibility reasons. Do not pass in anything except for
            the default value, as this argument is not used.

        Returns
        -------
         ind : numpy.matrix or int
            Indices of minimum elements. If matrix, its size along `axis` is 1.
        """
        # 调用内部方法 `_arg_min_or_max`，使用 `np.argmin` 函数找到最小元素的索引
        return self._arg_min_or_max(axis, out, np.argmin, np.less)
```