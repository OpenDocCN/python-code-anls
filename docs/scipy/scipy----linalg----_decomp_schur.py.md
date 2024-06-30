# `D:\src\scipysrc\scipy\scipy\linalg\_decomp_schur.py`

```
"""Schur decomposition functions."""
# 导入所需的库
import numpy as np
# 从 numpy 中导入特定的函数和变量
from numpy import asarray_chkfinite, single, asarray, array
# 从 numpy.linalg 中导入线性代数运算函数
from numpy.linalg import norm

# 从本地模块中导入所需的异常和函数
from ._misc import LinAlgError, _datacopied
from .lapack import get_lapack_funcs
from ._decomp import eigvals

# 定义公开的函数和变量列表
__all__ = ['schur', 'rsf2csf']

# 双精度类型列表
_double_precision = ['i', 'l', 'd']


def schur(a, output='real', lwork=None, overwrite_a=False, sort=None,
          check_finite=True):
    """
    Compute Schur decomposition of a matrix.

    The Schur decomposition is::

        A = Z T Z^H

    where Z is unitary and T is either upper-triangular, or for real
    Schur decomposition (output='real'), quasi-upper triangular. In
    the quasi-triangular form, 2x2 blocks describing complex-valued
    eigenvalue pairs may extrude from the diagonal.

    Parameters
    ----------
    a : (M, M) array_like
        Matrix to decompose
    output : {'real', 'complex'}, optional
        Construct the real or complex Schur decomposition (for real matrices).
    lwork : int, optional
        Work array size. If None or -1, it is automatically computed.
    overwrite_a : bool, optional
        Whether to overwrite data in a (may improve performance).
    sort : {None, callable, 'lhp', 'rhp', 'iuc', 'ouc'}, optional
        Specifies whether the upper eigenvalues should be sorted. A callable
        may be passed that, given a eigenvalue, returns a boolean denoting
        whether the eigenvalue should be sorted to the top-left (True).
        If output='real', the callable should have two arguments, the first
        one being the real part of the eigenvalue, the second one being
        the imaginary part.
        Alternatively, string parameters may be used::

            'lhp'   Left-hand plane (x.real < 0.0)
            'rhp'   Right-hand plane (x.real > 0.0)
            'iuc'   Inside the unit circle (x*x.conjugate() <= 1.0)
            'ouc'   Outside the unit circle (x*x.conjugate() > 1.0)

        Defaults to None (no sorting).
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    T : (M, M) ndarray
        Schur form of A. It is real-valued for the real Schur decomposition.
    Z : (M, M) ndarray
        An unitary Schur transformation matrix for A.
        It is real-valued for the real Schur decomposition.
    sdim : int
        If and only if sorting was requested, a third return value will
        contain the number of eigenvalues satisfying the sort condition.

    Raises
    ------
    """
    # 如果输出参数不是'real'、'complex'、'r'或'c'中的一个，则抛出数值错误异常
    if output not in ['real', 'complex', 'r', 'c']:
        raise ValueError("argument must be 'real', or 'complex'")
    
    # 如果检查有限性，将数组a转换为一个检查有限性后的数组a1；否则，将数组a转换为普通数组a1
    if check_finite:
        a1 = asarray_chkfinite(a)
    else:
        a1 = asarray(a)
    
    # 如果a1的数据类型是整数类型，将其转换为长整型数组a1
    if np.issubdtype(a1.dtype, np.integer):
        a1 = asarray(a, dtype=np.dtype("long"))
    
    # 如果数组a1不是二维的或者不是方阵，抛出数值错误异常
    if len(a1.shape) != 2 or (a1.shape[0] != a1.shape[1]):
        raise ValueError('expected square matrix')
    
    # 获取数组a1的数据类型字符
    typ = a1.dtype.char
    
    # 如果输出参数为'complex'或'c'，但是数组a1的数据类型字符不是'F'或'D'，则根据情况转换a1的数据类型
    if output in ['complex', 'c'] and typ not in ['F', 'D']:
        if typ in _double_precision:
            a1 = a1.astype('D')
        else:
            a1 = a1.astype('F')
    
    # 处理空矩阵的情况，返回与a1相同形状的空数组t0和z0；如果有排序函数sort，则返回第三个参数为0
    if a1.size == 0:
        t0, z0 = schur(np.eye(2, dtype=a1.dtype))
        if sort is None:
            return (np.empty_like(a1, dtype=t0.dtype),
                    np.empty_like(a1, dtype=z0.dtype))
        else:
            return (np.empty_like(a1, dtype=t0.dtype),
                    np.empty_like(a1, dtype=z0.dtype), 0)
    
    # 如果overwrite_a为True，或者_a数组是a的复制，返回用于计算GEES函数的LAPACK函数gees
    overwrite_a = overwrite_a or (_datacopied(a1, a))
    gees, = get_lapack_funcs(('gees',), (a1,))
    
    # 如果未指定lwork或lwork为-1，获取计算GEES函数所需的最佳工作数组长度
    if lwork is None or lwork == -1:
        result = gees(lambda x: None, a1, lwork=-1)
        lwork = result[-2][0].real.astype(np.int_)
    # 如果排序参数为 None，则使用默认的排序方式
    if sort is None:
        # 设定排序类型为 0
        sort_t = 0
        # 定义一个空的排序函数，始终返回 None
        def sfunction(x):
            return None
    else:
        # 否则设定排序类型为 1
        sort_t = 1
        # 如果排序参数是可调用的函数，则直接使用它作为排序函数
        if callable(sort):
            sfunction = sort
        # 如果排序参数是字符串 'lhp'，则定义排序函数为判断实部小于 0 的条件
        elif sort == 'lhp':
            def sfunction(x):
                return x.real < 0.0
        # 如果排序参数是字符串 'rhp'，则定义排序函数为判断实部大于等于 0 的条件
        elif sort == 'rhp':
            def sfunction(x):
                return x.real >= 0.0
        # 如果排序参数是字符串 'iuc'，则定义排序函数为判断绝对值小于等于 1 的条件
        elif sort == 'iuc':
            def sfunction(x):
                return abs(x) <= 1.0
        # 如果排序参数是字符串 'ouc'，则定义排序函数为判断绝对值大于 1 的条件
        elif sort == 'ouc':
            def sfunction(x):
                return abs(x) > 1.0
        # 如果排序参数不是以上指定的值，则抛出 ValueError 异常
        else:
            raise ValueError("'sort' parameter must either be 'None', or a "
                             "callable, or one of ('lhp','rhp','iuc','ouc')")

    # 使用指定的排序函数进行广义特征值求解
    result = gees(sfunction, a1, lwork=lwork, overwrite_a=overwrite_a,
                  sort_t=sort_t)

    # 获取广义特征值求解的返回信息
    info = result[-1]
    # 根据返回信息判断是否有错误情况，并抛出相应的异常
    if info < 0:
        raise ValueError(f'illegal value in {-info}-th argument of internal gees')
    elif info == a1.shape[0] + 1:
        raise LinAlgError('Eigenvalues could not be separated for reordering.')
    elif info == a1.shape[0] + 2:
        raise LinAlgError('Leading eigenvalues do not satisfy sort condition.')
    elif info > 0:
        raise LinAlgError("Schur form not found. Possibly ill-conditioned.")

    # 如果排序参数为 None，则返回特征值和特征向量的部分结果
    if sort is None:
        return result[0], result[-3]
    # 否则返回特征值、特征向量以及排序类型的完整结果
    else:
        return result[0], result[-3], result[1]
eps = np.finfo(float).eps
feps = np.finfo(single).eps
_array_kind = {'b': 0, 'h': 0, 'B': 0, 'i': 0, 'l': 0,
               'f': 0, 'd': 0, 'F': 1, 'D': 1}
_array_precision = {'i': 1, 'l': 1, 'f': 0, 'd': 1, 'F': 0, 'D': 1}
_array_type = [['f', 'd'], ['F', 'D']]

# 函数用于确定一组数组的公共类型
def _commonType(*arrays):
    kind = 0
    precision = 0
    for a in arrays:
        t = a.dtype.char
        kind = max(kind, _array_kind[t])
        precision = max(precision, _array_precision[t])
    return _array_type[kind][precision]

# 函数用于将一组数组强制转换为指定类型
def _castCopy(type, *arrays):
    cast_arrays = ()
    for a in arrays:
        if a.dtype.char == type:
            cast_arrays = cast_arrays + (a.copy(),)
        else:
            cast_arrays = cast_arrays + (a.astype(type),)
    if len(cast_arrays) == 1:
        return cast_arrays[0]
    else:
        return cast_arrays

# 函数用于将实数舒尔形式转换为复数舒尔形式
def rsf2csf(T, Z, check_finite=True):
    """
    Convert real Schur form to complex Schur form.

    Convert a quasi-diagonal real-valued Schur form to the upper-triangular
    complex-valued Schur form.

    Parameters
    ----------
    T : (M, M) array_like
        Real Schur form of the original array
    Z : (M, M) array_like
        Schur transformation matrix
    check_finite : bool, optional
        Whether to check that the input arrays contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    T : (M, M) ndarray
        Complex Schur form of the original array
    Z : (M, M) ndarray
        Schur transformation matrix corresponding to the complex form

    See Also
    --------
    schur : Schur decomposition of an array

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import schur, rsf2csf
    >>> A = np.array([[0, 2, 2], [0, 1, 2], [1, 0, 1]])
    >>> T, Z = schur(A)
    >>> T
    array([[ 2.65896708,  1.42440458, -1.92933439],
           [ 0.        , -0.32948354, -0.49063704],
           [ 0.        ,  1.31178921, -0.32948354]])
    >>> Z
    array([[0.72711591, -0.60156188, 0.33079564],
           [0.52839428, 0.79801892, 0.28976765],
           [0.43829436, 0.03590414, -0.89811411]])
    >>> T2 , Z2 = rsf2csf(T, Z)
    >>> T2
    array([[2.65896708+0.j, -1.64592781+0.743164187j, -1.21516887+1.00660462j],
           [0.+0.j , -0.32948354+8.02254558e-01j, -0.82115218-2.77555756e-17j],
           [0.+0.j , 0.+0.j, -0.32948354-0.802254558j]])
    >>> Z2
    array([[0.72711591+0.j,  0.28220393-0.31385693j,  0.51319638-0.17258824j],
           [0.52839428+0.j,  0.24720268+0.41635578j, -0.68079517-0.15118243j],
           [0.43829436+0.j, -0.76618703+0.01873251j, -0.03063006+0.46857912j]])

    """
    # 检查输入数组是否包含有限数值，根据需要执行检查
    if check_finite:
        Z, T = map(asarray_chkfinite, (Z, T))
    else:
        Z, T = map(asarray, (Z, T))
    # 对于输入的矩阵 Z 和 T，分别进行迭代处理
    for ind, X in enumerate([Z, T]):
        # 检查矩阵的维度是否为2且是否为方阵
        if X.ndim != 2 or X.shape[0] != X.shape[1]:
            # 如果不符合条件，抛出数值错误异常
            raise ValueError("Input '{}' must be square.".format('ZT'[ind]))

    # 检查矩阵 T 和 Z 的行数是否相等
    if T.shape[0] != Z.shape[0]:
        # 如果不相等，构造错误消息并抛出数值错误异常
        message = f"Input array shapes must match: Z: {Z.shape} vs. T: {T.shape}"
        raise ValueError(message)
    
    # 获取矩阵 T 的维度大小
    N = T.shape[0]
    # 确定矩阵 Z 和 T 以及一个包含数值3.0的数组的公共类型
    t = _commonType(Z, T, array([3.0], 'F'))
    # 将矩阵 Z 和 T 转换为公共类型 t 的副本
    Z, T = _castCopy(t, Z, T)

    # 对矩阵进行循环，进行相似变换以实现对角化
    for m in range(N-1, 0, -1):
        # 计算 T 矩阵中特定元素的绝对值是否大于给定精度 eps 乘以相关元素的和
        if abs(T[m, m-1]) > eps*(abs(T[m-1, m-1]) + abs(T[m, m])):
            # 计算特征值问题得到的特征值 mu
            mu = eigvals(T[m-1:m+1, m-1:m+1]) - T[m, m]
            # 计算 Givens 变换的相关参数
            r = norm([mu[0], T[m, m-1]])
            c = mu[0] / r
            s = T[m, m-1] / r
            # 构造 Givens 矩阵 G
            G = array([[c.conj(), s], [-s, c]], dtype=t)

            # 更新 T 的子块
            T[m-1:m+1, m-1:] = G.dot(T[m-1:m+1, m-1:])
            # 更新 T 的另一子块
            T[:m+1, m-1:m+1] = T[:m+1, m-1:m+1].dot(G.conj().T)
            # 更新 Z 的子块
            Z[:, m-1:m+1] = Z[:, m-1:m+1].dot(G.conj().T)

        # 将 T 的特定位置置为零
        T[m, m-1] = 0.0
    
    # 返回经过相似变换后的 T 和 Z 矩阵
    return T, Z
```