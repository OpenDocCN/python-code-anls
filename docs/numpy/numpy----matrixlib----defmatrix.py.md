# `.\numpy\numpy\matrixlib\defmatrix.py`

```
__all__ = ['matrix', 'bmat', 'asmatrix']

# 导入所需模块和库
import sys
import warnings
import ast

# 从模块中导入函数和类
from .._utils import set_module

# 导入核心数值计算模块及其函数
import numpy._core.numeric as N
from numpy._core.numeric import concatenate, isscalar

# 虽然 matrix_power 不在 __all__ 中，但为了向后兼容性，我们仍从 numpy.linalg 中导入
from numpy.linalg import matrix_power


def _convert_from_string(data):
    # 清理数据中的 '[]' 字符
    for char in '[]':
        data = data.replace(char, '')

    # 按行分割数据
    rows = data.split(';')
    newdata = []
    count = 0

    # 遍历每一行数据
    for row in rows:
        # 按逗号分割每行数据
        trow = row.split(',')
        newrow = []

        # 遍历每列数据
        for col in trow:
            # 按空格分割数据并进行字面评估，将结果扩展到新行中
            temp = col.split()
            newrow.extend(map(ast.literal_eval, temp))

        # 检查第一行的列数，确保所有行列数相同
        if count == 0:
            Ncols = len(newrow)
        elif len(newrow) != Ncols:
            raise ValueError("Rows not the same size.")

        count += 1
        newdata.append(newrow)

    return newdata


@set_module('numpy')
def asmatrix(data, dtype=None):
    """
    Interpret the input as a matrix.

    Unlike `matrix`, `asmatrix` does not make a copy if the input is already
    a matrix or an ndarray.  Equivalent to ``matrix(data, copy=False)``.

    Parameters
    ----------
    data : array_like
        Input data.
    dtype : data-type
       Data-type of the output matrix.

    Returns
    -------
    mat : matrix
        `data` interpreted as a matrix.

    Examples
    --------
    >>> x = np.array([[1, 2], [3, 4]])

    >>> m = np.asmatrix(x)

    >>> x[0,0] = 5

    >>> m
    matrix([[5, 2],
            [3, 4]])

    """
    # 调用 matrix 函数，将数据解释为矩阵，不进行拷贝操作
    return matrix(data, dtype=dtype, copy=False)


@set_module('numpy')
class matrix(N.ndarray):
    """
    matrix(data, dtype=None, copy=True)

    Returns a matrix from an array-like object, or from a string of data.

    A matrix is a specialized 2-D array that retains its 2-D nature
    through operations.  It has certain special operators, such as ``*``
    (matrix multiplication) and ``**`` (matrix power).

    .. note:: It is no longer recommended to use this class, even for linear
              algebra. Instead use regular arrays. The class may be removed
              in the future.

    Parameters
    ----------
    data : array_like or string
       If `data` is a string, it is interpreted as a matrix with commas
       or spaces separating columns, and semicolons separating rows.
    dtype : data-type
       Data-type of the output matrix.
    copy : bool
       If `data` is already an `ndarray`, then this flag determines
       whether the data is copied (the default), or whether a view is
       constructed.

    See Also
    --------
    array

    Examples
    --------
    >>> a = np.matrix('1 2; 3 4')
    >>> a
    matrix([[1, 2],
            [3, 4]])

    >>> np.matrix([[1, 2], [3, 4]])
    matrix([[1, 2],
            [3, 4]])

    """
    # 设置此类的数组优先级
    __array_priority__ = 10.0
    # 覆盖默认的构造方法 __new__，用于创建新的 matrix 对象
    def __new__(subtype, data, dtype=None, copy=True):
        # 发出警告，不推荐使用 matrix 子类表示矩阵或处理线性代数，建议使用普通的 ndarray
        warnings.warn('the matrix subclass is not the recommended way to '
                      'represent matrices or deal with linear algebra (see '
                      'https://docs.scipy.org/doc/numpy/user/'
                      'numpy-for-matlab-users.html). '
                      'Please adjust your code to use regular ndarray.',
                      PendingDeprecationWarning, stacklevel=2)
        # 如果输入数据已经是 matrix 类型，则根据条件返回原数据或者转换后的副本
        if isinstance(data, matrix):
            dtype2 = data.dtype
            if (dtype is None):
                dtype = dtype2
            if (dtype2 == dtype) and (not copy):
                return data
            return data.astype(dtype)

        # 如果输入数据是 ndarray 类型，则根据 dtype 参数和复制选项创建视图或副本
        if isinstance(data, N.ndarray):
            if dtype is None:
                intype = data.dtype
            else:
                intype = N.dtype(dtype)
            new = data.view(subtype)
            if intype != data.dtype:
                return new.astype(intype)
            if copy: return new.copy()
            else: return new

        # 如果输入数据是字符串，则调用 _convert_from_string 函数转换为数组
        if isinstance(data, str):
            data = _convert_from_string(data)

        # 将数据转换为数组 arr，根据其维度进行处理
        copy = None if not copy else True
        arr = N.array(data, dtype=dtype, copy=copy)
        ndim = arr.ndim
        shape = arr.shape

        # 检查矩阵维度是否为2，如果不是则引发 ValueError
        if (ndim > 2):
            raise ValueError("matrix must be 2-dimensional")
        elif ndim == 0:
            shape = (1, 1)
        elif ndim == 1:
            shape = (1, shape[0])

        # 矩阵默认的存储顺序为 'C' (按行优先)，如果数组是列优先则设置为 'F'
        order = 'C'
        if (ndim == 2) and arr.flags.fortran:
            order = 'F'

        # 如果数组不是连续存储则进行复制操作
        if not (order or arr.flags.contiguous):
            arr = arr.copy()

        # 使用 ndarray 的构造方法创建新的 matrix 对象并返回
        ret = N.ndarray.__new__(subtype, shape, arr.dtype,
                                buffer=arr,
                                order=order)
        return ret

    # 定义 __array_finalize__ 方法，用于在数组 finalization 期间执行必要的处理
    def __array_finalize__(self, obj):
        # 设定标志位 _getitem 为 False
        self._getitem = False
        # 如果 obj 是 matrix 类型并且其 _getitem 标志为真，则返回
        if (isinstance(obj, matrix) and obj._getitem): return
        ndim = self.ndim
        # 如果矩阵维度为2，则直接返回
        if (ndim == 2):
            return
        # 如果矩阵维度大于2，则调整 shape 为符合条件的新形状
        if (ndim > 2):
            newshape = tuple([x for x in self.shape if x > 1])
            ndim = len(newshape)
            if ndim == 2:
                self.shape = newshape
                return
            elif (ndim > 2):
                raise ValueError("shape too large to be a matrix.")
        else:
            newshape = self.shape
        # 处理特殊情况下的维度调整
        if ndim == 0:
            self.shape = (1, 1)
        elif ndim == 1:
            self.shape = (1, newshape[0])
        return
    # 在实例化对象中通过索引访问元素时调用的特殊方法
    def __getitem__(self, index):
        # 标记正在执行 __getitem__ 操作
        self._getitem = True

        try:
            # 尝试使用基类 N.ndarray 的 __getitem__ 方法获取元素
            out = N.ndarray.__getitem__(self, index)
        finally:
            # 无论如何，标记 __getitem__ 操作结束
            self._getitem = False

        # 如果返回的不是 N.ndarray 类型的对象，直接返回
        if not isinstance(out, N.ndarray):
            return out

        # 如果返回的是 0 维数组，返回其零维数组中的元素
        if out.ndim == 0:
            return out[()]

        # 如果返回的是 1 维数组
        if out.ndim == 1:
            sh = out.shape[0]
            # 尝试获取索引的长度，用于判断是否应该返回列向量
            try:
                n = len(index)
            except Exception:
                n = 0

            # 如果索引长度大于 1 且第二个索引是标量，将数组形状调整为 sh x 1
            if n > 1 and isscalar(index[1]):
                out.shape = (sh, 1)
            else:
                # 否则将数组形状调整为 1 x sh
                out.shape = (1, sh)

        # 返回处理后的数组
        return out

    # 在对象与其他对象相乘时调用的特殊方法
    def __mul__(self, other):
        # 如果 other 是 N.ndarray, list 或 tuple 类型
        if isinstance(other, (N.ndarray, list, tuple)) :
            # 将 self 与 other 做点乘运算，并返回结果
            return N.dot(self, asmatrix(other))
        
        # 如果 other 是标量或者没有 __rmul__ 方法的对象
        if isscalar(other) or not hasattr(other, '__rmul__') :
            # 将 self 与 other 做点乘运算，并返回结果
            return N.dot(self, other)

        # 如果不支持的操作，返回 NotImplemented
        return NotImplemented

    # 在其他对象与对象相乘时调用的特殊方法
    def __rmul__(self, other):
        # 返回 other 与 self 的点乘结果
        return N.dot(other, self)

    # 实现对象就地乘法的特殊方法
    def __imul__(self, other):
        # 将对象自身乘以 other，并将结果赋值给自身
        self[:] = self * other
        return self

    # 实现对象的乘方运算的特殊方法
    def __pow__(self, other):
        # 调用 matrix_power 函数计算 self 的 other 次方，并返回结果
        return matrix_power(self, other)

    # 实现对象就地乘方运算的特殊方法
    def __ipow__(self, other):
        # 将对象自身做 other 次方运算，并将结果赋值给自身
        self[:] = self ** other
        return self

    # 在其他对象与对象乘方运算时调用的特殊方法
    def __rpow__(self, other):
        # 不支持的操作，返回 NotImplemented
        return NotImplemented

    # 对象在进行轴向对齐操作时调用的辅助函数
    def _align(self, axis):
        """A convenience function for operations that need to preserve axis
        orientation.
        """
        # 如果 axis 为 None，返回矩阵的左上角元素
        if axis is None:
            return self[0, 0]
        # 如果 axis 为 0，返回自身
        elif axis==0:
            return self
        # 如果 axis 为 1，返回自身的转置
        elif axis==1:
            return self.transpose()
        else:
            # 如果 axis 值不合法，抛出 ValueError 异常
            raise ValueError("unsupported axis")

    # 对象在进行轴向折叠操作时调用的辅助函数
    def _collapse(self, axis):
        """A convenience function for operations that want to collapse
        to a scalar like _align, but are using keepdims=True
        """
        # 如果 axis 为 None，返回矩阵的左上角元素
        if axis is None:
            return self[0, 0]
        else:
            # 其他情况返回自身
            return self

    # 重写基类 tolist 方法以支持矩阵对象的转换为列表操作
    def tolist(self):
        """
        Return the matrix as a (possibly nested) list.

        See `ndarray.tolist` for full documentation.

        See Also
        --------
        ndarray.tolist

        Examples
        --------
        >>> x = np.matrix(np.arange(12).reshape((3,4))); x
        matrix([[ 0,  1,  2,  3],
                [ 4,  5,  6,  7],
                [ 8,  9, 10, 11]])
        >>> x.tolist()
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]

        """
        # 调用 __array__ 方法将矩阵转换为 ndarray 后，再调用 ndarray 的 tolist 方法
        return self.__array__().tolist()

    # To preserve orientation of result...
    # 返回矩阵元素沿指定轴的和。
    def sum(self, axis=None, dtype=None, out=None):
        """
        Returns the sum of the matrix elements, along the given axis.

        Refer to `numpy.sum` for full documentation.

        See Also
        --------
        numpy.sum

        Notes
        -----
        This is the same as `ndarray.sum`, except that where an `ndarray` would
        be returned, a `matrix` object is returned instead.

        Examples
        --------
        >>> x = np.matrix([[1, 2], [4, 3]])
        >>> x.sum()
        10
        >>> x.sum(axis=1)
        matrix([[3],
                [7]])
        >>> x.sum(axis=1, dtype='float')
        matrix([[3.],
                [7.]])
        >>> out = np.zeros((2, 1), dtype='float')
        >>> x.sum(axis=1, dtype='float', out=np.asmatrix(out))
        matrix([[3.],
                [7.]])

        """
        return N.ndarray.sum(self, axis, dtype, out, keepdims=True)._collapse(axis)


    # 返回可能重新塑形的矩阵。
    def squeeze(self, axis=None):
        """
        Return a possibly reshaped matrix.

        Refer to `numpy.squeeze` for more documentation.

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Selects a subset of the axes of length one in the shape.
            If an axis is selected with shape entry greater than one,
            an error is raised.

        Returns
        -------
        squeezed : matrix
            The matrix, but as a (1, N) matrix if it had shape (N, 1).

        See Also
        --------
        numpy.squeeze : related function

        Notes
        -----
        If `m` has a single column then that column is returned
        as the single row of a matrix.  Otherwise `m` is returned.
        The returned matrix is always either `m` itself or a view into `m`.
        Supplying an axis keyword argument will not affect the returned matrix
        but it may cause an error to be raised.

        Examples
        --------
        >>> c = np.matrix([[1], [2]])
        >>> c
        matrix([[1],
                [2]])
        >>> c.squeeze()
        matrix([[1, 2]])
        >>> r = c.T
        >>> r
        matrix([[1, 2]])
        >>> r.squeeze()
        matrix([[1, 2]])
        >>> m = np.matrix([[1, 2], [3, 4]])
        >>> m.squeeze()
        matrix([[1, 2],
                [3, 4]])

        """
        return N.ndarray.squeeze(self, axis=axis)
    def flatten(self, order='C'):
        """
        Return a flattened copy of the matrix.

        All `N` elements of the matrix are placed into a single row.

        Parameters
        ----------
        order : {'C', 'F', 'A', 'K'}, optional
            'C' means to flatten in row-major (C-style) order. 'F' means to
            flatten in column-major (Fortran-style) order. 'A' means to
            flatten in column-major order if `m` is Fortran *contiguous* in
            memory, row-major order otherwise. 'K' means to flatten `m` in
            the order the elements occur in memory. The default is 'C'.

        Returns
        -------
        y : matrix
            A copy of the matrix, flattened to a `(1, N)` matrix where `N`
            is the number of elements in the original matrix.

        See Also
        --------
        ravel : Return a flattened array.
        flat : A 1-D flat iterator over the matrix.

        Examples
        --------
        >>> m = np.matrix([[1,2], [3,4]])
        >>> m.flatten()
        matrix([[1, 2, 3, 4]])
        >>> m.flatten('F')
        matrix([[1, 3, 2, 4]])

        """
        # 调用 ndarray 类的 flatten 方法来对矩阵进行展平操作
        return N.ndarray.flatten(self, order=order)

    def mean(self, axis=None, dtype=None, out=None):
        """
        Returns the average of the matrix elements along the given axis.

        Refer to `numpy.mean` for full documentation.

        See Also
        --------
        numpy.mean

        Notes
        -----
        Same as `ndarray.mean` except that, where that returns an `ndarray`,
        this returns a `matrix` object.

        Examples
        --------
        >>> x = np.matrix(np.arange(12).reshape((3, 4)))
        >>> x
        matrix([[ 0,  1,  2,  3],
                [ 4,  5,  6,  7],
                [ 8,  9, 10, 11]])
        >>> x.mean()
        5.5
        >>> x.mean(0)
        matrix([[4., 5., 6., 7.]])
        >>> x.mean(1)
        matrix([[ 1.5],
                [ 5.5],
                [ 9.5]])

        """
        # 调用 ndarray 类的 mean 方法来计算矩阵元素的平均值，并返回一个 matrix 对象
        return N.ndarray.mean(self, axis, dtype, out, keepdims=True)._collapse(axis)
    def std(self, axis=None, dtype=None, out=None, ddof=0):
        """
        返回沿着给定轴的数组元素的标准差。

        参考 `numpy.std` 获取完整文档。

        See Also
        --------
        numpy.std

        Notes
        -----
        这与 `ndarray.std` 相同，不同之处在于 `ndarray` 返回时，返回的是一个 `matrix` 对象。

        Examples
        --------
        >>> x = np.matrix(np.arange(12).reshape((3, 4)))
        >>> x
        matrix([[ 0,  1,  2,  3],
                [ 4,  5,  6,  7],
                [ 8,  9, 10, 11]])
        >>> x.std()
        3.4520525295346629 # 可能有所不同
        >>> x.std(0)
        matrix([[ 3.26598632,  3.26598632,  3.26598632,  3.26598632]]) # 可能有所不同
        >>> x.std(1)
        matrix([[ 1.11803399],
                [ 1.11803399],
                [ 1.11803399]])

        """
        return N.ndarray.std(self, axis, dtype, out, ddof, keepdims=True)._collapse(axis)

    def var(self, axis=None, dtype=None, out=None, ddof=0):
        """
        返回沿着给定轴的矩阵元素的方差。

        参考 `numpy.var` 获取完整文档。

        See Also
        --------
        numpy.var

        Notes
        -----
        这与 `ndarray.var` 相同，不同之处在于 `ndarray` 返回时，返回的是一个 `matrix` 对象。

        Examples
        --------
        >>> x = np.matrix(np.arange(12).reshape((3, 4)))
        >>> x
        matrix([[ 0,  1,  2,  3],
                [ 4,  5,  6,  7],
                [ 8,  9, 10, 11]])
        >>> x.var()
        11.916666666666666
        >>> x.var(0)
        matrix([[ 10.66666667,  10.66666667,  10.66666667,  10.66666667]]) # 可能有所不同
        >>> x.var(1)
        matrix([[1.25],
                [1.25],
                [1.25]])

        """
        return N.ndarray.var(self, axis, dtype, out, ddof, keepdims=True)._collapse(axis)

    def prod(self, axis=None, dtype=None, out=None):
        """
        返回沿着给定轴的数组元素的乘积。

        参考 `prod` 获取完整文档。

        See Also
        --------
        prod, ndarray.prod

        Notes
        -----
        与 `ndarray.prod` 相同，不同之处在于 `ndarray` 返回时，返回的是一个 `matrix` 对象。

        Examples
        --------
        >>> x = np.matrix(np.arange(12).reshape((3,4))); x
        matrix([[ 0,  1,  2,  3],
                [ 4,  5,  6,  7],
                [ 8,  9, 10, 11]])
        >>> x.prod()
        0
        >>> x.prod(0)
        matrix([[  0,  45, 120, 231]])
        >>> x.prod(1)
        matrix([[   0],
                [ 840],
                [7920]])

        """
        return N.ndarray.prod(self, axis, dtype, out, keepdims=True)._collapse(axis)
    def any(self, axis=None, out=None):
        """
        Test whether any array element along a given axis evaluates to True.

        Refer to `numpy.any` for full documentation.

        Parameters
        ----------
        axis : int, optional
            Axis along which logical OR is performed
        out : ndarray, optional
            Output to existing array instead of creating new one, must have
            same shape as expected output

        Returns
        -------
        any : bool, ndarray
            Returns a single bool if `axis` is ``None``; otherwise,
            returns `ndarray`

        """
        # 调用 ndarray 类的 any 方法，用于检查数组元素在指定轴向上是否有 True 值，并保持维度信息
        return N.ndarray.any(self, axis, out, keepdims=True)._collapse(axis)

    def all(self, axis=None, out=None):
        """
        Test whether all matrix elements along a given axis evaluate to True.

        Parameters
        ----------
        See `numpy.all` for complete descriptions

        See Also
        --------
        numpy.all

        Notes
        -----
        This is the same as `ndarray.all`, but it returns a `matrix` object.

        Examples
        --------
        >>> x = np.matrix(np.arange(12).reshape((3,4))); x
        matrix([[ 0,  1,  2,  3],
                [ 4,  5,  6,  7],
                [ 8,  9, 10, 11]])
        >>> y = x[0]; y
        matrix([[0, 1, 2, 3]])
        >>> (x == y)
        matrix([[ True,  True,  True,  True],
                [False, False, False, False],
                [False, False, False, False]])
        >>> (x == y).all()
        False
        >>> (x == y).all(0)
        matrix([[False, False, False, False]])
        >>> (x == y).all(1)
        matrix([[ True],
                [False],
                [False]])

        """
        # 调用 ndarray 类的 all 方法，用于检查矩阵元素在指定轴向上是否全部为 True，并保持维度信息
        return N.ndarray.all(self, axis, out, keepdims=True)._collapse(axis)

    def max(self, axis=None, out=None):
        """
        Return the maximum value along an axis.

        Parameters
        ----------
        See `amax` for complete descriptions

        See Also
        --------
        amax, ndarray.max

        Notes
        -----
        This is the same as `ndarray.max`, but returns a `matrix` object
        where `ndarray.max` would return an ndarray.

        Examples
        --------
        >>> x = np.matrix(np.arange(12).reshape((3,4))); x
        matrix([[ 0,  1,  2,  3],
                [ 4,  5,  6,  7],
                [ 8,  9, 10, 11]])
        >>> x.max()
        11
        >>> x.max(0)
        matrix([[ 8,  9, 10, 11]])
        >>> x.max(1)
        matrix([[ 3],
                [ 7],
                [11]])

        """
        # 调用 ndarray 类的 max 方法，返回沿指定轴向的最大值，并保持维度信息
        return N.ndarray.max(self, axis, out, keepdims=True)._collapse(axis)
    def argmax(self, axis=None, out=None):
        """
        返回沿着指定轴的最大值的索引。

        返回沿着指定轴的最大值的第一个出现的索引。如果 axis 是 None，则索引是针对展平的矩阵。

        Parameters
        ----------
        axis : int, optional
            沿着哪个轴查找最大值的索引。默认为 None。
        out : ndarray, optional
            结果的放置位置。默认为 None。

        See Also
        --------
        numpy.argmax

        Notes
        -----
        这与 `ndarray.argmax` 相同，但返回一个 `matrix` 对象，而 `ndarray.argmax` 返回一个 `ndarray`。

        Examples
        --------
        >>> x = np.matrix(np.arange(12).reshape((3,4))); x
        matrix([[ 0,  1,  2,  3],
                [ 4,  5,  6,  7],
                [ 8,  9, 10, 11]])
        >>> x.argmax()
        11
        >>> x.argmax(0)
        matrix([[2, 2, 2, 2]])
        >>> x.argmax(1)
        matrix([[3],
                [3],
                [3]])

        """
        return N.ndarray.argmax(self, axis, out)._align(axis)

    def min(self, axis=None, out=None):
        """
        返回沿着指定轴的最小值。

        Parameters
        ----------
        axis : int, optional
            沿着哪个轴查找最小值。默认为 None。
        out : ndarray, optional
            结果的放置位置。默认为 None。

        See Also
        --------
        numpy.amin, ndarray.min

        Notes
        -----
        这与 `ndarray.min` 相同，但返回一个 `matrix` 对象，而 `ndarray.min` 返回一个 `ndarray`。

        Examples
        --------
        >>> x = -np.matrix(np.arange(12).reshape((3,4))); x
        matrix([[  0,  -1,  -2,  -3],
                [ -4,  -5,  -6,  -7],
                [ -8,  -9, -10, -11]])
        >>> x.min()
        -11
        >>> x.min(0)
        matrix([[ -8,  -9, -10, -11]])
        >>> x.min(1)
        matrix([[ -3],
                [ -7],
                [-11]])

        """
        return N.ndarray.min(self, axis, out, keepdims=True)._collapse(axis)

    def argmin(self, axis=None, out=None):
        """
        返回沿着指定轴的最小值的索引。

        返回沿着指定轴的最小值的第一个出现的索引。如果 axis 是 None，则索引是针对展平的矩阵。

        Parameters
        ----------
        axis : int, optional
            沿着哪个轴查找最小值的索引。默认为 None。
        out : ndarray, optional
            结果的放置位置。默认为 None。

        See Also
        --------
        numpy.argmin

        Notes
        -----
        这与 `ndarray.argmin` 相同，但返回一个 `matrix` 对象，而 `ndarray.argmin` 返回一个 `ndarray`。

        Examples
        --------
        >>> x = -np.matrix(np.arange(12).reshape((3,4))); x
        matrix([[  0,  -1,  -2,  -3],
                [ -4,  -5,  -6,  -7],
                [ -8,  -9, -10, -11]])
        >>> x.argmin()
        11
        >>> x.argmin(0)
        matrix([[2, 2, 2, 2]])
        >>> x.argmin(1)
        matrix([[3],
                [3],
                [3]])

        """
        return N.ndarray.argmin(self, axis, out)._align(axis)
    def ptp(self, axis=None, out=None):
        """
        Peak-to-peak (maximum - minimum) value along the given axis.

        Refer to `numpy.ptp` for full documentation.

        See Also
        --------
        numpy.ptp

        Notes
        -----
        Same as `ndarray.ptp`, except, where that would return an `ndarray` object,
        this returns a `matrix` object.

        Examples
        --------
        >>> x = np.matrix(np.arange(12).reshape((3,4))); x
        matrix([[ 0,  1,  2,  3],
                [ 4,  5,  6,  7],
                [ 8,  9, 10, 11]])
        >>> x.ptp()
        11
        >>> x.ptp(0)
        matrix([[8, 8, 8, 8]])
        >>> x.ptp(1)
        matrix([[3],
                [3],
                [3]])

        """
        return N.ptp(self, axis, out)._align(axis)



    @property
    def I(self):
        """
        Returns the (multiplicative) inverse of invertible `self`.

        Parameters
        ----------
        None

        Returns
        -------
        ret : matrix object
            If `self` is non-singular, `ret` is such that ``ret * self`` ==
            ``self * ret`` == ``np.matrix(np.eye(self[0,:].size))`` all return
            ``True``.

        Raises
        ------
        numpy.linalg.LinAlgError: Singular matrix
            If `self` is singular.

        See Also
        --------
        linalg.inv

        Examples
        --------
        >>> m = np.matrix('[1, 2; 3, 4]'); m
        matrix([[1, 2],
                [3, 4]])
        >>> m.getI()
        matrix([[-2. ,  1. ],
                [ 1.5, -0.5]])
        >>> m.getI() * m
        matrix([[ 1.,  0.], # may vary
                [ 0.,  1.]])

        """
        M, N = self.shape
        if M == N:
            from numpy.linalg import inv as func
        else:
            from numpy.linalg import pinv as func
        return asmatrix(func(self))



    @property
    def A(self):
        """
        Return `self` as an `ndarray` object.

        Equivalent to ``np.asarray(self)``.

        Parameters
        ----------
        None

        Returns
        -------
        ret : ndarray
            `self` as an `ndarray`

        Examples
        --------
        >>> x = np.matrix(np.arange(12).reshape((3,4))); x
        matrix([[ 0,  1,  2,  3],
                [ 4,  5,  6,  7],
                [ 8,  9, 10, 11]])
        >>> x.getA()
        array([[ 0,  1,  2,  3],
               [ 4,  5,  6,  7],
               [ 8,  9, 10, 11]])

        """
        return self.__array__()



    @property
    def H(self):
        """
        Return the conjugate transpose of `self`.

        Equivalent to ``np.asarray(self).conj().T``.

        Parameters
        ----------
        None

        Returns
        -------
        ret : matrix
            Conjugate transpose of `self`

        Examples
        --------
        >>> x = np.matrix([[1+1j, 2+2j], [3+3j, 4+4j]])
        >>> x.H
        matrix([[1.-1.j, 3.-3.j],
                [2.-2.j, 4.-4.j]])

        """
        return self.__array__().conj().T
    def A1(self):
        """
        Return `self` as a flattened `ndarray`.

        Equivalent to ``np.asarray(x).ravel()``

        Parameters
        ----------
        None

        Returns
        -------
        ret : ndarray
            `self`, 1-D, as an `ndarray`

        Examples
        --------
        >>> x = np.matrix(np.arange(12).reshape((3,4))); x
        matrix([[ 0,  1,  2,  3],
                [ 4,  5,  6,  7],
                [ 8,  9, 10, 11]])
        >>> x.getA1()
        array([ 0,  1,  2, ...,  9, 10, 11])


        """
        # 返回当前对象作为一个扁平化的 ndarray
        return self.__array__().ravel()


    def ravel(self, order='C'):
        """
        Return a flattened matrix.

        Refer to `numpy.ravel` for more documentation.

        Parameters
        ----------
        order : {'C', 'F', 'A', 'K'}, optional
            The elements of `m` are read using this index order. 'C' means to
            index the elements in C-like order, with the last axis index
            changing fastest, back to the first axis index changing slowest.
            'F' means to index the elements in Fortran-like index order, with
            the first index changing fastest, and the last index changing
            slowest. Note that the 'C' and 'F' options take no account of the
            memory layout of the underlying array, and only refer to the order
            of axis indexing.  'A' means to read the elements in Fortran-like
            index order if `m` is Fortran *contiguous* in memory, C-like order
            otherwise.  'K' means to read the elements in the order they occur
            in memory, except for reversing the data when strides are negative.
            By default, 'C' index order is used.

        Returns
        -------
        ret : matrix
            Return the matrix flattened to shape `(1, N)` where `N`
            is the number of elements in the original matrix.
            A copy is made only if necessary.

        See Also
        --------
        matrix.flatten : returns a similar output matrix but always a copy
        matrix.flat : a flat iterator on the array.
        numpy.ravel : related function which returns an ndarray

        """
        # 调用 N.ndarray.ravel 方法将当前对象扁平化，按指定顺序排列
        return N.ndarray.ravel(self, order=order)

    @property
    def T(self):
        """
        Returns the transpose of the matrix.

        Does *not* conjugate!  For the complex conjugate transpose, use ``.H``.

        Parameters
        ----------
        None

        Returns
        -------
        ret : matrix object
            The (non-conjugated) transpose of the matrix.

        See Also
        --------
        transpose, getH

        Examples
        --------
        >>> m = np.matrix('[1, 2; 3, 4]')
        >>> m
        matrix([[1, 2],
                [3, 4]])
        >>> m.getT()
        matrix([[1, 3],
                [2, 4]])

        """
        # 返回当前对象的转置矩阵
        return self.transpose()

    @property
    # 为对象的复共轭转置提供访问器方法

    # getT 访问器方法返回转置后的矩阵，已经废弃，保留仅用于兼容性
    getT = T.fget

    # getA 访问器方法返回矩阵自身的副本，已经废弃，保留仅用于兼容性
    getA = A.fget

    # getA1 访问器方法返回矩阵的第一个基类，已经废弃，保留仅用于兼容性
    getA1 = A1.fget

    # getH 访问器方法返回矩阵的复共轭转置
    getH = H.fget

    # getI 访问器方法返回矩阵的逆矩阵，已经废弃，保留仅用于兼容性
    getI = I.fget
# 如果输入对象是字符串，则根据全局或局部字典创建矩阵对象
def bmat(obj, ldict=None, gdict=None):
    """
    Build a matrix object from a string, nested sequence, or array.

    Parameters
    ----------
    obj : str or array_like
        Input data. If a string, variables in the current scope may be
        referenced by name.
    ldict : dict, optional
        A dictionary that replaces local operands in current frame.
        Ignored if `obj` is not a string or `gdict` is None.
    gdict : dict, optional
        A dictionary that replaces global operands in current frame.
        Ignored if `obj` is not a string.

    Returns
    -------
    out : matrix
        Returns a matrix object, which is a specialized 2-D array.

    See Also
    --------
    block :
        A generalization of this function for N-d arrays, that returns normal
        ndarrays.

    Examples
    --------
    >>> A = np.asmatrix('1 1; 1 1')
    >>> B = np.asmatrix('2 2; 2 2')
    >>> C = np.asmatrix('3 4; 5 6')
    >>> D = np.asmatrix('7 8; 9 0')

    All the following expressions construct the same block matrix:

    >>> np.bmat([[A, B], [C, D]])
    matrix([[1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 4, 7, 8],
            [5, 6, 9, 0]])
    >>> np.bmat(np.r_[np.c_[A, B], np.c_[C, D]])
    matrix([[1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 4, 7, 8],
            [5, 6, 9, 0]])
    >>> np.bmat('A,B; C,D')
    matrix([[1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 4, 7, 8],
            [5, 6, 9, 0]])

    """
    if isinstance(obj, str):
        # 如果对象是字符串，则根据全局或局部字典创建矩阵对象
        if gdict is None:
            # 获取上一个调用帧的全局和局部字典
            frame = sys._getframe().f_back
            glob_dict = frame.f_globals
            loc_dict = frame.f_locals
        else:
            glob_dict = gdict
            loc_dict = ldict

        # 调用内部函数 _from_string 将字符串转换为矩阵对象并返回
        return matrix(_from_string(obj, glob_dict, loc_dict))

    if isinstance(obj, (tuple, list)):
        # 如果对象是元组或列表，则根据内容创建矩阵对象
        arr_rows = []
        for row in obj:
            if isinstance(row, N.ndarray):  # 如果行是数组而不是二维数组
                # 直接拼接对象返回矩阵对象
                return matrix(concatenate(obj, axis=-1))
            else:
                # 将每行的数组对象拼接并存储
                arr_rows.append(concatenate(row, axis=-1))
        # 拼接所有行数组对象并返回矩阵对象
        return matrix(concatenate(arr_rows, axis=0))
    if isinstance(obj, N.ndarray):
        # 如果对象是数组，则直接返回矩阵对象
        return matrix(obj)
```