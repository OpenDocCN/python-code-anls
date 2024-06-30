# `D:\src\scipysrc\scipy\scipy\linalg\_decomp_qz.py`

```
import warnings  # 导入警告模块

import numpy as np  # 导入 NumPy 库
from numpy import asarray_chkfinite  # 从 NumPy 中导入 asarray_chkfinite 函数
from ._misc import LinAlgError, _datacopied, LinAlgWarning  # 导入自定义模块中的异常和警告类
from .lapack import get_lapack_funcs  # 从 lapack 模块导入 get_lapack_funcs 函数

__all__ = ['qz', 'ordqz']  # 模块级别变量，指定了在 from module import * 中导入的内容

_double_precision = ['i', 'l', 'd']  # 双精度数据类型列表

def _select_function(sort):
    if callable(sort):
        # 如果 sort 是可调用对象，则直接使用用户提供的函数
        sfunction = sort
    elif sort == 'lhp':
        # 如果 sort 是字符串 'lhp'，则选择 _lhp 函数
        sfunction = _lhp
    elif sort == 'rhp':
        # 如果 sort 是字符串 'rhp'，则选择 _rhp 函数
        sfunction = _rhp
    elif sort == 'iuc':
        # 如果 sort 是字符串 'iuc'，则选择 _iuc 函数
        sfunction = _iuc
    elif sort == 'ouc':
        # 如果 sort 是字符串 'ouc'，则选择 _ouc 函数
        sfunction = _ouc
    else:
        # 如果 sort 参数不是预期的值，则抛出 ValueError 异常
        raise ValueError("sort parameter must be None, a callable, or "
                         "one of ('lhp','rhp','iuc','ouc')")

    return sfunction  # 返回选定的函数对象

def _lhp(x, y):
    out = np.empty_like(x, dtype=bool)  # 创建一个与 x 同类型的空数组，用于存储布尔值
    nonzero = (y != 0)  # 找出 y 中非零元素的位置
    # 处理 (x, y) = (0, 0) 的情况
    out[~nonzero] = False
    # 计算 x 和 y 中非零元素位置上 x/y 的实部是否小于 0，结果存储在 out 数组中
    out[nonzero] = (np.real(x[nonzero]/y[nonzero]) < 0.0)
    return out  # 返回布尔数组 out

def _rhp(x, y):
    out = np.empty_like(x, dtype=bool)  # 创建一个与 x 同类型的空数组，用于存储布尔值
    nonzero = (y != 0)  # 找出 y 中非零元素的位置
    # 处理 (x, y) = (0, 0) 的情况
    out[~nonzero] = False
    # 计算 x 和 y 中非零元素位置上 x/y 的实部是否大于 0，结果存储在 out 数组中
    out[nonzero] = (np.real(x[nonzero]/y[nonzero]) > 0.0)
    return out  # 返回布尔数组 out

def _iuc(x, y):
    out = np.empty_like(x, dtype=bool)  # 创建一个与 x 同类型的空数组，用于存储布尔值
    nonzero = (y != 0)  # 找出 y 中非零元素的位置
    # 处理 (x, y) = (0, 0) 的情况
    out[~nonzero] = False
    # 计算 x 和 y 中非零元素位置上 abs(x/y) 是否小于 1.0，结果存储在 out 数组中
    out[nonzero] = (abs(x[nonzero]/y[nonzero]) < 1.0)
    return out  # 返回布尔数组 out

def _ouc(x, y):
    out = np.empty_like(x, dtype=bool)  # 创建一个与 x 同类型的空数组，用于存储布尔值
    xzero = (x == 0)  # 找出 x 中为零的位置
    yzero = (y == 0)  # 找出 y 中为零的位置
    # 处理 x 和 y 同时为零的情况
    out[xzero & yzero] = False
    # 处理 x 为非零而 y 为零的情况
    out[~xzero & yzero] = True
    # 计算 x 和 y 中非零元素位置上 abs(x/y) 是否大于 1.0，结果存储在 out 数组中
    out[~yzero] = (abs(x[~yzero]/y[~yzero]) > 1.0)
    return out  # 返回布尔数组 out

def _qz(A, B, output='real', lwork=None, sort=None, overwrite_a=False,
        overwrite_b=False, check_finite=True):
    if sort is not None:
        # 如果 sort 参数不是 None，则抛出 ValueError 异常
        raise ValueError("The 'sort' input of qz() has to be None and will be "
                         "removed in a future release. Use ordqz instead.")

    if output not in ['real', 'complex', 'r', 'c']:
        # 如果 output 参数不是指定的几种字符串，则抛出 ValueError 异常
        raise ValueError("argument must be 'real', or 'complex'")

    if check_finite:
        # 如果 check_finite 为 True，则使用 asarray_chkfinite 函数处理 A 和 B
        a1 = asarray_chkfinite(A)
        b1 = asarray_chkfinite(B)
    else:
        # 如果 check_finite 为 False，则直接转换 A 和 B 为 NumPy 数组
        a1 = np.asarray(A)
        b1 = np.asarray(B)

    a_m, a_n = a1.shape  # 获取 A 数组的形状
    b_m, b_n = b1.shape  # 获取 B 数组的形状
    if not (a_m == a_n == b_m == b_n):
        # 如果 A 和 B 不是方阵，则抛出 ValueError 异常
        raise ValueError("Array dimensions must be square and agree")

    typa = a1.dtype.char  # 获取 A 数组的数据类型字符
    if output in ['complex', 'c'] and typa not in ['F', 'D']:
        if typa in _double_precision:
            a1 = a1.astype('D')  # 将 A 转换为复数类型
            typa = 'D'
        else:
            a1 = a1.astype('F')  # 将 A 转换为复数类型
            typa = 'F'
    typb = b1.dtype.char  # 获取 B 数组的数据类型字符
    if output in ['complex', 'c'] and typb not in ['F', 'D']:
        if typb in _double_precision:
            b1 = b1.astype('D')  # 将 B 转换为复数类型
            typb = 'D'
        else:
            b1 = b1.astype('F')  # 将 B 转换为复数类型
            typb = 'F'

    overwrite_a = overwrite_a or (_datacopied(a1, A))  # 检查是否需要复制 A
    overwrite_b = overwrite_b or (_datacopied(b1, B))  # 检查是否需要复制 B
    gges, = get_lapack_funcs(('gges',), (a1, b1))

获取 LAPACK 函数 `gges`，用于求解广义特征值问题。


    if lwork is None or lwork == -1:
        # 获取最优工作数组大小
        result = gges(lambda x: None, a1, b1, lwork=-1)
        lwork = result[-2][0].real.astype(int)

如果 `lwork` 为 `None` 或者 `-1`，则调用 `gges` 函数以获取最优工作数组大小，并将其转换为整数类型。


    def sfunction(x):
        return None

定义一个简单的函数 `sfunction`，用于作为 `gges` 函数的参数。


    result = gges(sfunction, a1, b1, lwork=lwork, overwrite_a=overwrite_a,
                  overwrite_b=overwrite_b, sort_t=0)

调用 `gges` 函数来求解广义特征值问题，使用前面定义的 `sfunction`，并传递其他参数如 `lwork`、`overwrite_a`、`overwrite_b` 和 `sort_t`。


    info = result[-1]

从 `gges` 的返回结果中获取最后一个元素 `info`，用于表示求解的状态信息。


    if info < 0:
        raise ValueError(f"Illegal value in argument {-info} of gges")
    elif info > 0 and info <= a_n:
        warnings.warn("The QZ iteration failed. (a,b) are not in Schur "
                      "form, but ALPHAR(j), ALPHAI(j), and BETA(j) should be "
                      f"correct for J={info-1},...,N", LinAlgWarning,
                      stacklevel=3)
    elif info == a_n+1:
        raise LinAlgError("Something other than QZ iteration failed")
    elif info == a_n+2:
        raise LinAlgError("After reordering, roundoff changed values of some "
                          "complex eigenvalues so that leading eigenvalues "
                          "in the Generalized Schur form no longer satisfy "
                          "sort=True. This could also be due to scaling.")
    elif info == a_n+3:
        raise LinAlgError("Reordering failed in <s,d,c,z>tgsen")

根据 `info` 的值进行不同的处理：如果小于 `0`，则抛出 `ValueError`；如果大于 `0` 且小于等于 `a_n`，则发出警告；如果等于 `a_n+1`、`a_n+2` 或 `a_n+3`，则分别抛出 `LinAlgError`。


    return result, gges.typecode

返回 `gges` 函数的计算结果 `result` 和其类型码 `typecode`。
# QZ 分解函数，用于一对矩阵的广义特征值分解。

def qz(A, B, output='real', lwork=None, sort=None, overwrite_a=False,
       overwrite_b=False, check_finite=True):
    """
    QZ decomposition for generalized eigenvalues of a pair of matrices.

    The QZ, or generalized Schur, decomposition for a pair of n-by-n
    matrices (A,B) is::

        (A,B) = (Q @ AA @ Z*, Q @ BB @ Z*)

    where AA, BB is in generalized Schur form if BB is upper-triangular
    with non-negative diagonal and AA is upper-triangular, or for real QZ
    decomposition (``output='real'``) block upper triangular with 1x1
    and 2x2 blocks. In this case, the 1x1 blocks correspond to real
    generalized eigenvalues and 2x2 blocks are 'standardized' by making
    the corresponding elements of BB have the form::

        [ a 0 ]
        [ 0 b ]

    and the pair of corresponding 2x2 blocks in AA and BB will have a complex
    conjugate pair of generalized eigenvalues. If (``output='complex'``) or
    A and B are complex matrices, Z' denotes the conjugate-transpose of Z.
    Q and Z are unitary matrices.

    Parameters
    ----------
    A : (N, N) array_like
        2-D array to decompose
    B : (N, N) array_like
        2-D array to decompose
    output : {'real', 'complex'}, optional
        Construct the real or complex QZ decomposition for real matrices.
        Default is 'real'.
    lwork : int, optional
        Work array size. If None or -1, it is automatically computed.
    sort : {None, callable, 'lhp', 'rhp', 'iuc', 'ouc'}, optional
        NOTE: THIS INPUT IS DISABLED FOR NOW. Use ordqz instead.

        Specifies whether the upper eigenvalues should be sorted. A callable
        may be passed that, given a eigenvalue, returns a boolean denoting
        whether the eigenvalue should be sorted to the top-left (True). For
        real matrix pairs, the sort function takes three real arguments
        (alphar, alphai, beta). The eigenvalue
        ``x = (alphar + alphai*1j)/beta``. For complex matrix pairs or
        output='complex', the sort function takes two complex arguments
        (alpha, beta). The eigenvalue ``x = (alpha/beta)``.  Alternatively,
        string parameters may be used:

            - 'lhp'   Left-hand plane (x.real < 0.0)
            - 'rhp'   Right-hand plane (x.real > 0.0)
            - 'iuc'   Inside the unit circle (x*x.conjugate() < 1.0)
            - 'ouc'   Outside the unit circle (x*x.conjugate() > 1.0)

        Defaults to None (no sorting).
    overwrite_a : bool, optional
        Whether to overwrite data in a (may improve performance)
    overwrite_b : bool, optional
        Whether to overwrite data in b (may improve performance)
    check_finite : bool, optional
        If true checks the elements of `A` and `B` are finite numbers. If
        false does no checking and passes matrix through to
        underlying algorithm.

    Returns
    -------
    AA : (N, N) ndarray
        Generalized Schur form of A.
    BB : (N, N) ndarray
        Generalized Schur form of B.
    """
    Q : (N, N) ndarray
        左 Schur 向量。
    Z : (N, N) ndarray
        右 Schur 向量。

    See Also
    --------
    ordqz

    Notes
    -----
    Q 在 Matlab 中相应函数的结果中是转置的。

    .. versionadded:: 0.11.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import qz

    >>> A = np.array([[1, 2, -1], [5, 5, 5], [2, 4, -8]])
    >>> B = np.array([[1, 1, -3], [3, 1, -1], [5, 6, -2]])

    计算 QZ 分解。QZ 分解不是唯一的，所以取决于使用的底层库，下面输出中的系数符号可能会有差异。

    >>> AA, BB, Q, Z = qz(A, B)
    >>> AA
    array([[-1.36949157, -4.05459025,  7.44389431],
           [ 0.        ,  7.65653432,  5.13476017],
           [ 0.        , -0.65978437,  2.4186015 ]])  # 可能有所不同
    >>> BB
    array([[ 1.71890633, -1.64723705, -0.72696385],
           [ 0.        ,  8.6965692 , -0.        ],
           [ 0.        ,  0.        ,  2.27446233]])  # 可能有所不同
    >>> Q
    array([[-0.37048362,  0.1903278 ,  0.90912992],
           [-0.90073232,  0.16534124, -0.40167593],
           [ 0.22676676,  0.96769706, -0.11017818]])  # 可能有所不同
    >>> Z
    array([[-0.67660785,  0.63528924, -0.37230283],
           [ 0.70243299,  0.70853819, -0.06753907],
           [ 0.22088393, -0.30721526, -0.92565062]])  # 可能有所不同

    验证 QZ 分解。在实际输出中，我们只需要使用 ``Z`` 的转置。

    >>> Q @ AA @ Z.T  # 应该等于 A
    array([[ 1.,  2., -1.],
           [ 5.,  5.,  5.],
           [ 2.,  4., -8.]])
    >>> Q @ BB @ Z.T  # 应该等于 B
    array([[ 1.,  1., -3.],
           [ 3.,  1., -1.],
           [ 5.,  6., -2.]])

    重复分解，但使用 ``output='complex'``。

    >>> AA, BB, Q, Z = qz(A, B, output='complex')

    为了输出简洁，我们使用 ``np.set_printoptions()`` 设置 NumPy 数组的输出精度为 3，并将微小值显示为 0。

    >>> np.set_printoptions(precision=3, suppress=True)
    >>> AA
    array([[-1.369+0.j   ,  2.248+4.237j,  4.861-5.022j],
           [ 0.   +0.j   ,  7.037+2.922j,  0.794+4.932j],
           [ 0.   +0.j   ,  0.   +0.j   ,  2.655-1.103j]])  # 可能有所不同
    >>> BB
    array([[ 1.719+0.j   , -1.115+1.j   , -0.763-0.646j],
           [ 0.   +0.j   ,  7.24 +0.j   , -3.144+3.322j],
           [ 0.   +0.j   ,  0.   +0.j   ,  2.732+0.j   ]])  # 可能有所不同
    >>> Q
    array([[ 0.326+0.175j, -0.273-0.029j, -0.886-0.052j],
           [ 0.794+0.426j, -0.093+0.134j,  0.402-0.02j ],
           [-0.2  -0.107j, -0.816+0.482j,  0.151-0.167j]])  # 可能有所不同
    >>> Z
    array([[ 0.596+0.32j , -0.31 +0.414j,  0.393-0.347j],
           [-0.619-0.332j, -0.479+0.314j,  0.154-0.393j],
           [-0.195-0.104j,  0.576+0.27j ,  0.715+0.187j]])  # 可能有所不同

    对于复数数组，我们必须使用 ``Z.conj().T`` 作为以下表达式的一部分。
    expressions to verify the decomposition.

    >>> Q @ AA @ Z.conj().T  # 应该是 A
    array([[ 1.-0.j,  2.-0.j, -1.-0.j],
           [ 5.+0.j,  5.+0.j,  5.-0.j],
           [ 2.+0.j,  4.+0.j, -8.+0.j]])
    >>> Q @ BB @ Z.conj().T  # 应该是 B
    array([[ 1.+0.j,  1.+0.j, -3.+0.j],
           [ 3.-0.j,  1.-0.j, -1.+0.j],
           [ 5.+0.j,  6.+0.j, -2.+0.j]])

    """
    # 输出为实数时的变量顺序
    # AA, BB, sdim, alphar, alphai, beta, vsl, vsr, work, info
    # 输出为复数时的变量顺序
    # AA, BB, sdim, alpha, beta, vsl, vsr, work, info
    # 调用_qz函数进行广义特征值问题的求解
    result, _ = _qz(A, B, output=output, lwork=lwork, sort=sort,
                    overwrite_a=overwrite_a, overwrite_b=overwrite_b,
                    check_finite=check_finite)
    # 返回结果中的特定变量
    return result[0], result[1], result[-4], result[-3]
# 定义 QZ 分解函数，用于对一对矩阵进行 QZ 分解并进行重新排序

def ordqz(A, B, sort='lhp', output='real', overwrite_a=False,
          overwrite_b=False, check_finite=True):
    """QZ decomposition for a pair of matrices with reordering.

    Parameters
    ----------
    A : (N, N) array_like
        2-D array to decompose
    B : (N, N) array_like
        2-D array to decompose
    sort : {callable, 'lhp', 'rhp', 'iuc', 'ouc'}, optional
        Specifies whether the upper eigenvalues should be sorted. A
        callable may be passed that, given an ordered pair ``(alpha,
        beta)`` representing the eigenvalue ``x = (alpha/beta)``,
        returns a boolean denoting whether the eigenvalue should be
        sorted to the top-left (True). For the real matrix pairs
        ``beta`` is real while ``alpha`` can be complex, and for
        complex matrix pairs both ``alpha`` and ``beta`` can be
        complex. The callable must be able to accept a NumPy
        array. Alternatively, string parameters may be used:

            - 'lhp'   Left-hand plane (x.real < 0.0)
            - 'rhp'   Right-hand plane (x.real > 0.0)
            - 'iuc'   Inside the unit circle (x*x.conjugate() < 1.0)
            - 'ouc'   Outside the unit circle (x*x.conjugate() > 1.0)

        With the predefined sorting functions, an infinite eigenvalue
        (i.e., ``alpha != 0`` and ``beta = 0``) is considered to lie in
        neither the left-hand nor the right-hand plane, but it is
        considered to lie outside the unit circle. For the eigenvalue
        ``(alpha, beta) = (0, 0)``, the predefined sorting functions
        all return `False`.
    output : str {'real','complex'}, optional
        Construct the real or complex QZ decomposition for real matrices.
        Default is 'real'.
    overwrite_a : bool, optional
        If True, the contents of A are overwritten.
    overwrite_b : bool, optional
        If True, the contents of B are overwritten.
    check_finite : bool, optional
        If true checks the elements of `A` and `B` are finite numbers. If
        false does no checking and passes matrix through to
        underlying algorithm.

    Returns
    -------
    AA : (N, N) ndarray
        Generalized Schur form of A.
    BB : (N, N) ndarray
        Generalized Schur form of B.
    alpha : (N,) ndarray
        alpha = alphar + alphai * 1j. See notes.
    beta : (N,) ndarray
        See notes.
    Q : (N, N) ndarray
        The left Schur vectors.
    Z : (N, N) ndarray
        The right Schur vectors.

    See Also
    --------
    qz

    Notes
    -----
    On exit, ``(ALPHAR(j) + ALPHAI(j)*i)/BETA(j), j=1,...,N``, will be the
    generalized eigenvalues.  ``ALPHAR(j) + ALPHAI(j)*i`` and
    ``BETA(j),j=1,...,N`` are the diagonals of the complex Schur form (S,T)
    that would result if the 2-by-2 diagonal blocks of the real generalized
    Schur form of (A,B) were further reduced to triangular form using complex
    unitary transformations. If ALPHAI(j) is zero, then the jth eigenvalue is
    real.

    """
    """
    Perform generalized Schur decomposition of two matrix pairs (A, B).

    This function computes the generalized Schur decomposition of matrix pairs (A, B),
    where A and B are given matrices. It returns orthogonal matrices QQ and ZZ such that
    QQ^H * A * ZZ and QQ^H * B * ZZ are in generalized Schur form (AAA, BBB). It also
    computes alpha and beta, which are used to construct AAA and BBB.

    Parameters
    ----------
    A : array_like, shape (n, n)
        The first input matrix.
    B : array_like, shape (n, n)
        The second input matrix.
    sort : {'lhp', 'rhp', 'iuc', 'ouc', None}, optional
        Specifies the sort order of the generalized Schur decomposition.
        - 'lhp': Left half-plane eigenvalues first.
        - 'rhp': Right half-plane eigenvalues first.
        - 'iuc': Eigenvalues inside the unit circle first.
        - 'ouc': Eigenvalues outside the unit circle first.
        - None: No sorting; eigenvalues returned in arbitrary order.
    output : {'real', 'complex', 'r', 'c'}, optional
        Determines the output format. 'real' returns real values, 'complex' returns complex values.
    overwrite_a, overwrite_b : bool, optional
        Whether to overwrite data in A and B (may improve performance).
    check_finite : bool, optional
        Whether to check that input matrices contain only finite numbers.
    
    Returns
    -------
    AAA : ndarray, shape (n, n)
        Generalized Schur form of A.
    BBB : ndarray, shape (n, n)
        Generalized Schur form of B.
    alpha : ndarray
        Array of scalars used to form AAA.
    beta : ndarray
        Array of scalars used to form BBB.
    QQ : ndarray, shape (n, n)
        Orthogonal matrix QQ.
    ZZ : ndarray, shape (n, n)
        Orthogonal matrix ZZ.

    Raises
    ------
    ValueError
        If an illegal value is encountered in the arguments of the underlying LAPACK function.

    Notes
    -----
    - This function wraps LAPACK's tgsen routine to perform the generalized Schur decomposition.
    - The sort parameter affects the order of eigenvalues returned in AAA and BBB.
    - The function may modify A and B if overwrite_a or overwrite_b is set to True.

    .. versionadded:: 0.17.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import ordqz
    >>> A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
    >>> B = np.array([[0, 6, 0, 0], [5, 0, 2, 1], [5, 2, 6, 6], [4, 7, 7, 7]])
    >>> AA, BB, alpha, beta, Q, Z = ordqz(A, B, sort='lhp')

    Since we have sorted for left half plane eigenvalues, negatives come first

    >>> (alpha/beta).real < 0
    array([ True,  True, False, False], dtype=bool)

    """
```