# `D:\src\scipysrc\scipy\scipy\linalg\_misc.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
from numpy.linalg import LinAlgError  # 导入 NumPy 中的线性代数错误类
from .blas import get_blas_funcs  # 从本地的 blas 模块中导入 get_blas_funcs 函数
from .lapack import get_lapack_funcs  # 从本地的 lapack 模块中导入 get_lapack_funcs 函数

__all__ = ['LinAlgError', 'LinAlgWarning', 'norm']  # 定义模块导出的公共接口

class LinAlgWarning(RuntimeWarning):
    """
    The warning emitted when a linear algebra related operation is close
    to fail conditions of the algorithm or loss of accuracy is expected.
    """
    pass  # 定义线性代数警告类，用于指示可能的算法失败或精度损失情况

def norm(a, ord=None, axis=None, keepdims=False, check_finite=True):
    """
    Matrix or vector norm.

    This function is able to return one of eight different matrix norms,
    or one of an infinite number of vector norms (described below), depending
    on the value of the ``ord`` parameter. For tensors with rank different from
    1 or 2, only `ord=None` is supported.

    Parameters
    ----------
    a : array_like
        Input array. If `axis` is None, `a` must be 1-D or 2-D, unless `ord`
        is None. If both `axis` and `ord` are None, the 2-norm of
        ``a.ravel`` will be returned.
    ord : {int, inf, -inf, 'fro', 'nuc', None}, optional
        Order of the norm (see table under ``Notes``). inf means NumPy's
        `inf` object.
    axis : {int, 2-tuple of ints, None}, optional
        If `axis` is an integer, it specifies the axis of `a` along which to
        compute the vector norms. If `axis` is a 2-tuple, it specifies the
        axes that hold 2-D matrices, and the matrix norms of these matrices
        are computed. If `axis` is None then either a vector norm (when `a`
        is 1-D) or a matrix norm (when `a` is 2-D) is returned.
    keepdims : bool, optional
        If this is set to True, the axes which are normed over are left in the
        result as dimensions with size one. With this option the result will
        broadcast correctly against the original `a`.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    n : float or ndarray
        Norm of the matrix or vector(s).

    Notes
    -----
    For values of ``ord <= 0``, the result is, strictly speaking, not a
    mathematical 'norm', but it may still be useful for various numerical
    purposes.

    The following norms can be calculated:

    =====  ============================  ==========================
    ord    norm for matrices             norm for vectors
    =====  ============================  ==========================
    None   Frobenius norm                2-norm
    'fro'  Frobenius norm                --
    'nuc'  nuclear norm                  --
    inf    max(sum(abs(a), axis=1))      max(abs(a))
    -inf   min(sum(abs(a), axis=1))      min(abs(a))
    0      --                            sum(a != 0)
    1      max(sum(abs(a), axis=0))      as below
    -1     min(sum(abs(a), axis=0))      as below
    ```
    计算给定数组或矩阵的范数，具体范数的计算方式取决于参数 `ord` 的选择。

    Parameters
    ----------
    a : array_like
        输入的数组或矩阵。如果 `axis` 为 None，则 `a` 必须是一维或二维的数组或矩阵，除非 `ord` 为 None。
        如果 `axis` 和 `ord` 都为 None，则返回 `a.ravel` 的二范数。
    ord : {int, inf, -inf, 'fro', 'nuc', None}, optional
        范数的阶数（详见下面的说明表）。inf 表示 NumPy 中的无穷大对象。
    axis : {int, 2 元组的整数, None}, optional
        如果 `axis` 是整数，则指定 `a` 中计算向量范数的轴。
        如果 `axis` 是一个二元组，则指定包含二维矩阵的轴，并计算这些矩阵的矩阵范数。
        如果 `axis` 是 None，则根据 `a` 的维度情况返回向量范数或矩阵范数。
    keepdims : bool, optional
        如果为 True，则保留计算范数的轴，结果中将保留维度为 1 的尺寸。
        使用此选项，结果将正确地广播到原始的 `a`。
    check_finite : bool, optional
        是否检查输入矩阵是否只包含有限数字。禁用此选项可能会提高性能，但如果输入包含无穷大或 NaN，可能会导致问题（崩溃、非终止）。

    Returns
    -------
    n : float or ndarray
        数组或矩阵的范数值。

    Notes
    -----
    对于 `ord <= 0` 的值，严格来说，结果不是数学上的 '范数'，但在各种数值计算目的中仍然有用。

    可以计算以下范数：

    =====  ============================  ==========================
    ord    矩阵的范数                     向量的范数
    =====  ============================  ==========================
    None   Frobenius 范数                2-范数
    'fro'  Frobenius 范数                --
    'nuc'  核范数                        --
    inf    max(sum(abs(a), axis=1))      max(abs(a))
    -inf   min(sum(abs(a), axis=1))      min(abs(a))
    0      --                            sum(a != 0)
    1      max(sum(abs(a), axis=0))      如下
    -1     min(sum(abs(a), axis=0))      如下
    ```
    返回给定数组或矩阵的范数，具体的计算方式取决于 `ord` 参数的值。
    """
    """
        2      2-norm (largest sing. value)  as below
        -2     smallest singular value       as below
        other  --                            sum(abs(a)**ord)**(1./ord)
        =====  ============================  ==========================
    
        The Frobenius norm is given by [1]_:
    
            :math:`||A||_F = [\\sum_{i,j} abs(a_{i,j})^2]^{1/2}`
    
        The nuclear norm is the sum of the singular values.
    
        Both the Frobenius and nuclear norm orders are only defined for
        matrices.
    
        References
        ----------
        .. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*,
               Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15
    
        Examples
        --------
        >>> import numpy as np
        >>> from scipy.linalg import norm
        >>> a = np.arange(9) - 4.0
        >>> a
        array([-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])
        >>> b = a.reshape((3, 3))
        >>> b
        array([[-4., -3., -2.],
               [-1.,  0.,  1.],
               [ 2.,  3.,  4.]])
    
        >>> norm(a)
        7.745966692414834
        >>> norm(b)
        7.745966692414834
        >>> norm(b, 'fro')
        7.745966692414834
        >>> norm(a, np.inf)
        4
        >>> norm(b, np.inf)
        9
        >>> norm(a, -np.inf)
        0
        >>> norm(b, -np.inf)
        2
    
        >>> norm(a, 1)
        20
        >>> norm(b, 1)
        7
        >>> norm(a, -1)
        -4.6566128774142013e-010
        >>> norm(b, -1)
        6
        >>> norm(a, 2)
        7.745966692414834
        >>> norm(b, 2)
        7.3484692283495345
    
        >>> norm(a, -2)
        0
        >>> norm(b, -2)
        1.8570331885190563e-016
        >>> norm(a, 3)
        5.8480354764257312
        >>> norm(a, -3)
        0
    
        """
        # Differs from numpy only in non-finite handling and the use of blas.
        # 如果需要检查数组是否有非有限元素，使用np.asarray_chkfinite进行处理
        if check_finite:
            a = np.asarray_chkfinite(a)
        else:
            # 否则，将数组a转换为NumPy数组
            a = np.asarray(a)
    
        # 如果数组a不为空且元素类型为浮点数或复数，并且未指定轴向和保持维度形状
        if a.size and a.dtype.char in 'fdFD' and axis is None and not keepdims:
    
            # 如果ord参数为None或2，并且数组a为一维数组
            if ord in (None, 2) and (a.ndim == 1):
                # 使用BLAS库进行快速和稳定的欧几里德范数计算
                nrm2 = get_blas_funcs('nrm2', dtype=a.dtype, ilp64='preferred')
                return nrm2(a)
    
            # 如果数组a为二维数组
            if a.ndim == 2:
                # 使用LAPACK库进行一些快速矩阵范数计算
                # 但某些情况下，Frobenius范数计算较慢
                lange_args = None
                # 确保在用户使用轴关键字将范数应用于转置时也能正常工作
                if ord == 1:
                    if np.isfortran(a):
                        lange_args = '1', a
                    elif np.isfortran(a.T):
                        lange_args = 'i', a.T
                elif ord == np.inf:
                    if np.isfortran(a):
                        lange_args = 'i', a
                    elif np.isfortran(a.T):
                        lange_args = '1', a.T
                if lange_args:
                    # 使用LAPACK库的lange函数计算指定的范数
                    lange = get_lapack_funcs('lange', dtype=a.dtype, ilp64='preferred')
                    return lange(*lange_args)
    
        # 对于其他情况，使用NumPy的linalg.norm函数计算范数
        return np.linalg.norm(a, ord=ord, axis=axis, keepdims=keepdims)
# 定义函数 `_datacopied`，用于严格检查 `arr` 是否不与 `original` 共享任何数据，
# 假设 `arr = asarray(original)`
def _datacopied(arr, original):
    # 如果 `arr` 和 `original` 是同一个对象，则返回 False，表示它们共享数据
    if arr is original:
        return False
    # 如果 `original` 不是 NumPy 数组类型且具有 `__array__` 属性，则返回 False，
    # 表示它们可能共享数据
    if not isinstance(original, np.ndarray) and hasattr(original, '__array__'):
        return False
    # 检查 `arr` 的数据是否不共享任何基础数据
    return arr.base is None
```