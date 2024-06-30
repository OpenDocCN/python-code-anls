# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_expm_multiply.py`

```
# 导入警告模块中的警告函数
from warnings import warn
# 导入 NumPy 库，用于科学计算
import numpy as np

# 导入 SciPy 线性代数模块和稀疏矩阵线性代数模块
import scipy.linalg
import scipy.sparse.linalg

# 从 SciPy 的 QR 分解模块中导入 qr 函数
from scipy.linalg._decomp_qr import qr
# 从 SciPy 稀疏矩阵工具模块中导入判断是否为 PyData 稀疏矩阵的函数
from scipy.sparse._sputils import is_pydata_spmatrix
# 从 SciPy 稀疏矩阵线性代数模块中导入线性算子转换函数
from scipy.sparse.linalg import aslinearoperator
# 从 SciPy 稀疏矩阵线性代数接口模块中导入单位算子类
from scipy.sparse.linalg._interface import IdentityOperator
# 从 SciPy 稀疏矩阵线性代数接口模块中导入一范数估计函数
from scipy.sparse.linalg._onenormest import onenormest

# 模块全局变量，指定模块中公开的函数名
__all__ = ['expm_multiply']

# 定义函数 _exact_inf_norm(A)，计算 A 的无穷范数
def _exact_inf_norm(A):
    # 兼容性函数，应逐渐被移除
    if scipy.sparse.issparse(A):
        # 如果 A 是稀疏矩阵，则计算每行绝对值的和的最大值
        return max(abs(A).sum(axis=1).flat)
    elif is_pydata_spmatrix(A):
        # 如果 A 是 PyData 稀疏矩阵，则计算每行绝对值的和的最大值
        return max(abs(A).sum(axis=1))
    else:
        # 否则，使用 NumPy 计算 A 的无穷范数
        return np.linalg.norm(A, np.inf)

# 定义函数 _exact_1_norm(A)，计算 A 的一范数
def _exact_1_norm(A):
    # 兼容性函数，应逐渐被移除
    if scipy.sparse.issparse(A):
        # 如果 A 是稀疏矩阵，则计算每列绝对值的和的最大值
        return max(abs(A).sum(axis=0).flat)
    elif is_pydata_spmatrix(A):
        # 如果 A 是 PyData 稀疏矩阵，则计算每列绝对值的和的最大值
        return max(abs(A).sum(axis=0))
    else:
        # 否则，使用 NumPy 计算 A 的一范数
        return np.linalg.norm(A, 1)

# 定义函数 _trace(A)，计算 A 的迹
def _trace(A):
    # 兼容性函数，如果 A 是 PyData 稀疏矩阵，则将其转换为 SciPy 稀疏矩阵后计算迹
    if is_pydata_spmatrix(A):
        return A.to_scipy_sparse().trace()
    else:
        # 否则，直接计算 A 的迹
        return A.trace()

# 定义函数 traceest(A, m3, seed=None)，使用随机矩阵-向量乘积估计 A 的迹
def traceest(A, m3, seed=None):
    """Estimate `np.trace(A)` using `3*m3` matrix-vector products.

    The result is not deterministic.

    Parameters
    ----------
    A : LinearOperator
        Linear operator whose trace will be estimated. Has to be square.
    m3 : int
        Number of matrix-vector products divided by 3 used to estimate the
        trace.
    seed : optional
        Seed for `numpy.random.default_rng`.
        Can be provided to obtain deterministic results.

    Returns
    -------
    trace : LinearOperator.dtype
        Estimate of the trace

    Notes
    -----
    This is the Hutch++ algorithm given in [1]_.

    References
    ----------
    .. [1] Meyer, Raphael A., Cameron Musco, Christopher Musco, and David P.
       Woodruff. "Hutch++: Optimal Stochastic Trace Estimation." In Symposium
       on Simplicity in Algorithms (SOSA), pp. 142-155. Society for Industrial
       and Applied Mathematics, 2021
       https://doi.org/10.1137/1.9781611976496.16

    """
    # 使用指定种子创建随机数生成器对象
    rng = np.random.default_rng(seed)
    # 检查 A 的形状是否为二维且为方阵
    if len(A.shape) != 2 or A.shape[-1] != A.shape[-2]:
        raise ValueError("Expected A to be like a square matrix.")
    n = A.shape[-1]
    # 创建随机矩阵 S，元素为 +1 或 -1
    S = rng.choice([-1.0, +1.0], [n, m3])
    # 对 A 与 S 的乘积进行 QR 分解，经济模式（只返回必要的部分）
    Q, _ = qr(A.matmat(S), overwrite_a=True, mode='economic')
    # 计算 Q 转置与 A 乘积的迹
    trQAQ = np.trace(Q.conj().T @ A.matmat(Q))
    # 创建随机矩阵 G，元素为 +1 或 -1
    G = rng.choice([-1, +1], [n, m3])
    # 计算右侧表达式，用 G 减去 Q 与 Q 转置乘积的结果
    right = G - Q@(Q.conj().T @ G)
    # 计算右侧表达式与 A 的乘积的迹
    trGAG = np.trace(right.conj().T @ A.matmat(right))
    # 返回估计的迹值
    return trQAQ + trGAG/m3

# 定义函数 _ident_like(A)，兼容性函数，应逐渐被移除
def _ident_like(A):
    # 检查输入的矩阵 A 是否为稀疏矩阵（Scipy 的稀疏矩阵类型）
    if scipy.sparse.issparse(A):
        # 创建一个对角线格式的单位稀疏矩阵
        out = scipy.sparse.eye(A.shape[0], A.shape[1], dtype=A.dtype)
        # 如果 A 是稀疏矩阵，则返回一个与 A 具有相同格式的稀疏对角线数组
        if isinstance(A, scipy.sparse.spmatrix):
            return out.asformat(A.format)
        # 否则返回一个对角线数组，并转换为 A 的格式
        return scipy.sparse.dia_array(out).asformat(A.format)
    # 如果 A 是 PyDataSparse 稀疏矩阵类型
    elif is_pydata_spmatrix(A):
        import sparse
        # 返回一个稀疏单位矩阵，与 A 相同的形状和数据类型
        return sparse.eye(A.shape[0], A.shape[1], dtype=A.dtype)
    # 如果 A 是 Scipy 的线性操作符（LinearOperator）类型
    elif isinstance(A, scipy.sparse.linalg.LinearOperator):
        # 返回一个表示单位操作符的对象，与 A 具有相同的形状和数据类型
        return IdentityOperator(A.shape, dtype=A.dtype)
    # 如果 A 不属于以上任何类型，则默认返回一个单位矩阵，与 A 具有相同的形状和数据类型
    else:
        return np.eye(A.shape[0], A.shape[1], dtype=A.dtype)
# 计算矩阵指数 expm(A) 作用于矩阵或向量 B 的结果
def expm_multiply(A, B, start=None, stop=None, num=None,
                  endpoint=None, traceA=None):
    """
    Compute the action of the matrix exponential of A on B.

    Parameters
    ----------
    A : transposable linear operator
        The operator whose exponential is of interest.
    B : ndarray
        The matrix or vector to be multiplied by the matrix exponential of A.
    start : scalar, optional
        The starting time point of the sequence.
    stop : scalar, optional
        The end time point of the sequence, unless `endpoint` is set to False.
        In that case, the sequence consists of all but the last of ``num + 1``
        evenly spaced time points, so that `stop` is excluded.
        Note that the step size changes when `endpoint` is False.
    num : int, optional
        Number of time points to use.
    endpoint : bool, optional
        If True, `stop` is the last time point.  Otherwise, it is not included.
    traceA : scalar, optional
        Trace of `A`. If not given the trace is estimated for linear operators,
        or calculated exactly for sparse matrices. It is used to precondition
        `A`, thus an approximate trace is acceptable.
        For linear operators, `traceA` should be provided to ensure performance
        as the estimation is not guaranteed to be reliable for all cases.

        .. versionadded:: 1.9.0

    Returns
    -------
    expm_A_B : ndarray
         The result of the action :math:`e^{t_k A} B`.

    Warns
    -----
    UserWarning
        If `A` is a linear operator and ``traceA=None`` (default).

    Notes
    -----
    The optional arguments defining the sequence of evenly spaced time points
    are compatible with the arguments of `numpy.linspace`.

    The output ndarray shape is somewhat complicated so I explain it here.
    The ndim of the output could be either 1, 2, or 3.
    It would be 1 if you are computing the expm action on a single vector
    at a single time point.
    It would be 2 if you are computing the expm action on a vector
    at multiple time points, or if you are computing the expm action
    on a matrix at a single time point.
    It would be 3 if you want the action on a matrix with multiple
    columns at multiple time points.
    If multiple time points are requested, expm_A_B[0] will always
    be the action of the expm at the first time point,
    regardless of whether the action is on a vector or a matrix.

    References
    ----------
    .. [1] Awad H. Al-Mohy and Nicholas J. Higham (2011)
           "Computing the Action of the Matrix Exponential,
           with an Application to Exponential Integrators."
           SIAM Journal on Scientific Computing,
           33 (2). pp. 488-511. ISSN 1064-8275
           http://eprints.ma.man.ac.uk/1591/
    """
    """
    If all arguments (start, stop, num, endpoint) are None, use simple matrix exponentiation method.
    Otherwise, use interval-based matrix exponentiation method.
    
    Parameters:
    - A: array_like or sparse matrix, shape (m, m)
        Matrix to be exponentiated.
    - B: array_like, shape (m,)
        Vector to be multiplied.
    - start : float or None
        Time-point to start interval.
    - stop : float or None
        Time-point to stop interval.
    - num : int or None
        Number of samples to compute within the interval.
    - endpoint : bool, optional
        If True, stop is the last sample. Otherwise, stop is not included.
    - traceA : bool, optional
        If True, return the trace of A in addition to the result.
    
    Returns:
    - X : ndarray, shape (m, len(B))
        Computed result.
    
    References:
    - [2] Nicholas J. Higham and Awad H. Al-Mohy (2010)
          "Computing Matrix Functions."
          Acta Numerica,
          19. 159-208. ISSN 0962-4929
          http://eprints.ma.man.ac.uk/1451/
    
    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_matrix
    >>> from scipy.sparse.linalg import expm, expm_multiply
    >>> A = csc_matrix([[1, 0], [0, 1]])
    >>> A.toarray()
    array([[1, 0],
           [0, 1]], dtype=int64)
    >>> B = np.array([np.exp(-1.), np.exp(-2.)])
    >>> B
    array([ 0.36787944,  0.13533528])
    >>> expm_multiply(A, B, start=1, stop=2, num=3, endpoint=True)
    array([[ 1.        ,  0.36787944],
           [ 1.64872127,  0.60653066],
           [ 2.71828183,  1.        ]])
    >>> expm(A).dot(B)                  # Verify 1st timestep
    array([ 1.        ,  0.36787944])
    >>> expm(1.5*A).dot(B)              # Verify 2nd timestep
    array([ 1.64872127,  0.60653066])
    >>> expm(2*A).dot(B)                # Verify 3rd timestep
    array([ 2.71828183,  1.        ])
    """
    if all(arg is None for arg in (start, stop, num, endpoint)):
        # Use the simple matrix exponentiation method
        X = _expm_multiply_simple(A, B, traceA=traceA)
    else:
        # Use the interval-based matrix exponentiation method
        X, status = _expm_multiply_interval(A, B, start, stop, num,
                                            endpoint, traceA=traceA)
    return X
def _expm_multiply_simple(A, B, t=1.0, traceA=None, balance=False):
    """
    Compute the action of the matrix exponential at a single time point.

    Parameters
    ----------
    A : transposable linear operator
        The operator whose exponential is of interest.
    B : ndarray
        The matrix to be multiplied by the matrix exponential of A.
    t : float
        A time point.
    traceA : scalar, optional
        Trace of `A`. If not given the trace is estimated for linear operators,
        or calculated exactly for sparse matrices. It is used to precondition
        `A`, thus an approximate trace is acceptable.
    balance : bool
        Indicates whether or not to apply balancing.

    Returns
    -------
    F : ndarray
        :math:`e^{t A} B`

    Notes
    -----
    This is algorithm (3.2) in Al-Mohy and Higham (2011).

    """
    # 如果启用了平衡选项，则抛出未实现的错误
    if balance:
        raise NotImplementedError

    # 检查 A 是否为二维矩阵且为方阵
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be like a square matrix')

    # 检查 A 和 B 的形状是否兼容
    if A.shape[1] != B.shape[0]:
        raise ValueError('shapes of matrices A {} and B {} are incompatible'
                         .format(A.shape, B.shape))

    # 创建与 A 类型相同的单位矩阵
    ident = _ident_like(A)

    # 检查 A 是否为线性操作符
    is_linear_operator = isinstance(A, scipy.sparse.linalg.LinearOperator)

    # 获取 A 的大小
    n = A.shape[0]

    # 确定 B 的维度
    if len(B.shape) == 1:
        n0 = 1
    elif len(B.shape) == 2:
        n0 = B.shape[1]
    else:
        raise ValueError('expected B to be like a matrix or a vector')

    # 设置浮点数精度
    u_d = 2**-53
    tol = u_d

    # 如果未提供 traceA，则估算或计算 A 的迹
    if traceA is None:
        if is_linear_operator:
            warn("Trace of LinearOperator not available, it will be estimated."
                 " Provide `traceA` to ensure performance.", stacklevel=3)
        traceA = traceest(A, m3=1) if is_linear_operator else _trace(A)

    # 计算 mu，用于对 A 进行预处理
    mu = traceA / float(n)
    A = A - mu * ident

    # 计算 A 的 1-范数
    A_1_norm = onenormest(A) if is_linear_operator else _exact_1_norm(A)

    # 如果 t * A 的 1-范数为零，则初始化 m_star 和 s
    if t * A_1_norm == 0:
        m_star, s = 0, 1
    else:
        ell = 2
        norm_info = LazyOperatorNormInfo(t * A, A_1_norm=t * A_1_norm, ell=ell)
        m_star, s = _fragment_3_1(norm_info, n0, tol, ell=ell)

    # 调用核心计算函数来计算 e^{t A} B
    return _expm_multiply_simple_core(A, B, t, mu, m_star, s, tol, balance)


def _expm_multiply_simple_core(A, B, t, mu, m_star, s, tol=None, balance=False):
    """
    A helper function.
    """
    # 如果启用了平衡选项，则抛出未实现的错误
    if balance:
        raise NotImplementedError

    # 如果未指定公差 tol，则设置默认浮点数精度
    if tol is None:
        u_d = 2 ** -53
        tol = u_d

    # 初始化 F 为 B
    F = B

    # 计算 eta
    eta = np.exp(t * mu / float(s))

    # 迭代 s 次进行矩阵乘法运算
    for i in range(s):
        c1 = _exact_inf_norm(B)
        for j in range(m_star):
            coeff = t / float(s * (j + 1))
            B = coeff * A.dot(B)
            c2 = _exact_inf_norm(B)
            F = F + B
            if c1 + c2 <= tol * _exact_inf_norm(F):
                break
            c1 = c2

        # 更新 F
        F = eta * F
        B = F
    # 返回变量 F 的值作为函数的返回结果
    return F
# 这个表格用于计算边界值。
# 它们似乎难以计算，涉及到方程的符号操作，然后进行数值根查找。
_theta = {
        # 前30个值来自《计算矩阵函数》的表A.3。
        1: 2.29e-16,
        2: 2.58e-8,
        3: 1.39e-5,
        4: 3.40e-4,
        5: 2.40e-3,
        6: 9.07e-3,
        7: 2.38e-2,
        8: 5.00e-2,
        9: 8.96e-2,
        10: 1.44e-1,
        # 第11项
        11: 2.14e-1,
        12: 3.00e-1,
        13: 4.00e-1,
        14: 5.14e-1,
        15: 6.41e-1,
        16: 7.81e-1,
        17: 9.31e-1,
        18: 1.09,
        19: 1.26,
        20: 1.44,
        # 第21项
        21: 1.62,
        22: 1.82,
        23: 2.01,
        24: 2.22,
        25: 2.43,
        26: 2.64,
        27: 2.86,
        28: 3.08,
        29: 3.31,
        30: 3.54,
        # 其余值来自《计算矩阵指数的作用》的表3.1。
        35: 4.7,
        40: 6.0,
        45: 7.2,
        50: 8.5,
        55: 9.9,
        }

def _onenormest_matrix_power(A, p,
        t=2, itmax=5, compute_v=False, compute_w=False):
    """
    高效估计 A^p 的1-范数。

    Parameters
    ----------
    A : ndarray
        要计算其幂的矩阵。
    p : int
        非负整数幂。
    t : int, optional
        控制精度与时间、内存使用之间的权衡的正参数。
        较大的值需要更长时间和更多内存，但提供更精确的输出。
    itmax : int, optional
        最多使用这么多次迭代。
    compute_v : bool, optional
        如果为True，则请求一个范数最大化的线性运算符输入向量。
    compute_w : bool, optional
        如果为True，则请求一个范数最大化的线性运算符输出向量。

    Returns
    -------
    est : float
        稀疏矩阵1-范数的一个低估计值。
    v : ndarray, optional
        向量 v 满足 ||Av||_1 == est*||v||_1。
        它可以被视为线性运算符的输入，产生具有特别大范数的输出。
    w : ndarray, optional
        向量 Av 具有相对较大的1-范数。
        它可以被视为线性运算符的输出，在范数上比输入要大。

    """
    # 最终将其转换为 _onenormest 模块中的 API 函数，
    # 并删除其下划线，
    # 但在 expm_multiply 进入 scipy 之前暂时保留。
    from scipy.sparse.linalg._onenormest import onenormest
    return onenormest(aslinearoperator(A) ** p)

class LazyOperatorNormInfo:
    """
    操作符的信息是延迟计算的。

    这些信息包括操作符的精确1-范数，
    以及操作符幂的1-范数的估计。
    这使用了《计算矩阵的作用》(2011)中的符号表示法。

    """
    This class is specialized enough to probably not be of general interest
    outside of this module.

    """

    # 定义一个特定的类，用于操作符及其规范化相关信息
    class NormEstimator:
        
        def __init__(self, A, A_1_norm=None, ell=2, scale=1):
            """
            提供操作符及其规范化相关信息。

            Parameters
            ----------
            A : linear operator
                感兴趣的操作符。
            A_1_norm : float, optional
                A 的确切1-范数。
            ell : int, optional
                控制范数估计质量的技术参数。
            scale : int, optional
                如果指定，则返回 scale*A 的范数而不是 A。

            """
            self._A = A
            self._A_1_norm = A_1_norm
            self._ell = ell
            self._d = {}  # 空字典，用于存储估计的 d_p 值
            self._scale = scale  # 初始化规模参数

        def set_scale(self, scale):
            """
            设置规模参数。
            """
            self._scale = scale

        def onenorm(self):
            """
            计算确切的1-范数。
            """
            if self._A_1_norm is None:
                self._A_1_norm = _exact_1_norm(self._A)
            return self._scale * self._A_1_norm

        def d(self, p):
            """
            惰性估计 :math:`d_p(A) ~= || A^p ||^(1/p)`，其中 :math:`||.||` 是1-范数。
            """
            if p not in self._d:
                est = _onenormest_matrix_power(self._A, p, self._ell)
                self._d[p] = est ** (1.0 / p)
            return self._scale * self._d[p]

        def alpha(self, p):
            """
            惰性计算 max(d(p), d(p+1))。
            """
            return max(self.d(p), self.d(p + 1))
# 计算成本除以 m 的辅助函数，用于计算边界

def _compute_cost_div_m(m, p, norm_info):
    """
    A helper function for computing bounds.

    This is equation (3.10).
    It measures cost in terms of the number of required matrix products.

    Parameters
    ----------
    m : int
        A valid key of _theta.
        _theta 中的有效键值。
    p : int
        A matrix power.
        矩阵的幂次。
    norm_info : LazyOperatorNormInfo
        Information about 1-norms of related operators.
        有关相关算子的 1-范数信息。

    Returns
    -------
    cost_div_m : int
        Required number of matrix products divided by m.
        所需的矩阵乘积数量除以 m。

    """
    return int(np.ceil(norm_info.alpha(p) / _theta[m]))


# 计算最大的正整数 p，使得 p*(p-1) <= m_max + 1

def _compute_p_max(m_max):
    """
    Compute the largest positive integer p such that p*(p-1) <= m_max + 1.

    Do this in a slightly dumb way, but safe and not too slow.

    Parameters
    ----------
    m_max : int
        A count related to bounds.
        与边界相关的计数。

    """
    sqrt_m_max = np.sqrt(m_max)
    p_low = int(np.floor(sqrt_m_max))
    p_high = int(np.ceil(sqrt_m_max + 1))
    return max(p for p in range(p_low, p_high+1) if p*(p-1) <= m_max + 1)


# 用于 _expm_multiply_* 函数的辅助函数

def _fragment_3_1(norm_info, n0, tol, m_max=55, ell=2):
    """
    A helper function for the _expm_multiply_* functions.

    Parameters
    ----------
    norm_info : LazyOperatorNormInfo
        Information about norms of certain linear operators of interest.
        某些线性算子的范数信息。
    n0 : int
        Number of columns in the _expm_multiply_* B matrix.
        _expm_multiply_* 函数中 B 矩阵的列数。
    tol : float
        Expected to be
        :math:`2^{-24}` for single precision or
        :math:`2^{-53}` for double precision.
        预期为单精度的 :math:`2^{-24}` 或双精度的 :math:`2^{-53}`。
    m_max : int
        A value related to a bound.
        与边界相关的值。
    ell : int
        The number of columns used in the 1-norm approximation.
        This is usually taken to be small, maybe between 1 and 5.
        在 1-范数近似中使用的列数，通常取小值，可能在 1 到 5 之间。

    Returns
    -------
    best_m : int
        Related to bounds for error control.
        与误差控制边界相关。
    best_s : int
        Amount of scaling.
        缩放量。

    Notes
    -----
    This is code fragment (3.1) in Al-Mohy and Higham (2011).
    The discussion of default values for m_max and ell
    is given between the definitions of equation (3.11)
    and the definition of equation (3.12).
    这是 Al-Mohy 和 Higham (2011) 中的代码片段 (3.1)。
    在方程 (3.11) 的定义和方程 (3.12) 的定义之间讨论了 m_max 和 ell 的默认值。

    """
    if ell < 1:
        raise ValueError('expected ell to be a positive integer')
    best_m = None
    best_s = None
    if _condition_3_13(norm_info.onenorm(), n0, m_max, ell):
        for m, theta in _theta.items():
            s = int(np.ceil(norm_info.onenorm() / theta))
            if best_m is None or m * s < best_m * best_s:
                best_m = m
                best_s = s
    else:
        # Equation (3.11).
        for p in range(2, _compute_p_max(m_max) + 1):
            for m in range(p*(p-1)-1, m_max+1):
                if m in _theta:
                    s = _compute_cost_div_m(m, p, norm_info)
                    if best_m is None or m * s < best_m * best_s:
                        best_m = m
                        best_s = s
        best_s = max(best_s, 1)
    return best_m, best_s


# _expm_multiply_* 函数的辅助函数，检查条件 (3.13)

def _condition_3_13(A_1_norm, n0, m_max, ell):
    """
    A helper function for the _expm_multiply_* functions.
    
    """
    Parameters
    ----------
    A_1_norm : float
        A precomputed 1-norm of matrix A.
        矩阵 A 的预计算的 1-范数。
    n0 : int
        Number of columns in the _expm_multiply_* B matrix.
        _expm_multiply_* B 矩阵中的列数。
    m_max : int
        A value related to a bound.
        与某个界限相关的值。
    ell : int
        The number of columns used in the 1-norm approximation.
        This is usually taken to be small, maybe between 1 and 5.
        在 1-范数近似中使用的列数，通常较小，可能在 1 到 5 之间。

    Returns
    -------
    value : bool
        Indicates whether or not the condition has been met.
        指示条件是否已满足的布尔值。

    Notes
    -----
    This is condition (3.13) in Al-Mohy and Higham (2011).
    这是 Al-Mohy 和 Higham (2011) 中的条件 (3.13)。

    """

    # This is the rhs of equation (3.12).
    # 计算方程式 (3.12) 的右侧。
    p_max = _compute_p_max(m_max)
    a = 2 * ell * p_max * (p_max + 3)

    # Evaluate the condition (3.13).
    # 计算条件 (3.13)。
    b = _theta[m_max] / float(n0 * m_max)
    return A_1_norm <= a * b
# 如果设置了平衡参数，抛出未实现错误，暂不支持平衡操作
if balance:
    raise NotImplementedError

# 检查矩阵 A 是否为二维且为方阵
if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
    raise ValueError('expected A to be like a square matrix')

# 检查矩阵 A 和矩阵 B 的形状是否兼容
if A.shape[1] != B.shape[0]:
    raise ValueError('shapes of matrices A {} and B {} are incompatible'
                     .format(A.shape, B.shape))

# 根据 A 的形状生成类似单位矩阵的标识矩阵
ident = _ident_like(A)

# 检查 A 是否为线性操作符
is_linear_operator = isinstance(A, scipy.sparse.linalg.LinearOperator)

# 获取矩阵 A 的维度大小
n = A.shape[0]

# 确定矩阵 B 的列数
if len(B.shape) == 1:
    n0 = 1
elif len(B.shape) == 2:
    n0 = B.shape[1]
else:
    # 如果 B 的形状既不是向量也不是矩阵，则抛出错误
    raise ValueError('expected B to be like a matrix or a vector')

# 设置机器精度的二分之一作为容差值
u_d = 2**-53
tol = u_d

# 如果未提供 traceA 参数，则根据情况估算或计算 A 的迹
if traceA is None:
    if is_linear_operator:
        # 如果 A 是线性操作符，警告迹的估算将会被执行，建议提供确切的迹以提升性能
        warn("Trace of LinearOperator not available, it will be estimated."
             " Provide `traceA` to ensure performance.", stacklevel=3)
    # 如果 A 不是线性操作符，根据 A 的类型计算其迹
    traceA = traceest(A, m3=5) if is_linear_operator else _trace(A)
    # 计算平均轨迹 mu，即 traceA 除以样本数 n 的浮点数结果
    mu = traceA / float(n)

    # 准备获取 linspace 的样本点，尝试保留 linspace 的默认设置。
    linspace_kwargs = {'retstep': True}
    # 如果 num 不为 None，则设置 linspace 的样本数
    if num is not None:
        linspace_kwargs['num'] = num
    # 如果 endpoint 不为 None，则设置 linspace 的结束点设置
    if endpoint is not None:
        linspace_kwargs['endpoint'] = endpoint
    # 使用 numpy 的 linspace 函数获取样本点 samples 和步长 step
    samples, step = np.linspace(start, stop, **linspace_kwargs)

    # 将 linspace 的输出转换为出版物使用的符号表示。
    nsamples = len(samples)
    if nsamples < 2:
        raise ValueError('at least two time points are required')
    q = nsamples - 1
    h = step
    t_0 = samples[0]
    t_q = samples[q]

    # 定义输出的 ndarray。
    # 使用 ndim=3 的形状，使得最后两个索引可以参与三级 BLAS 操作。
    X_shape = (nsamples,) + B.shape
    X = np.empty(X_shape, dtype=np.result_type(A.dtype, B.dtype, float))
    t = t_q - t_0
    # 计算 A - mu * ident，其中 ident 是单位矩阵
    A = A - mu * ident
    # 计算 A 的 1-范数
    A_1_norm = onenormest(A) if is_linear_operator else _exact_1_norm(A)
    ell = 2
    # 创建 LazyOperatorNormInfo 对象，用于存储 A 的相关信息
    norm_info = LazyOperatorNormInfo(t * A, A_1_norm=t * A_1_norm, ell=ell)
    # 根据条件判断 m_star 和 s 的值
    if t * A_1_norm == 0:
        m_star, s = 0, 1
    else:
        m_star, s = _fragment_3_1(norm_info, n0, tol, ell=ell)

    # 计算在初始时间点 t_0 处的 expm 操作
    X[0] = _expm_multiply_simple_core(A, B, t_0, mu, m_star, s)

    # 计算其余时间点处的 expm 操作
    if q <= s:
        # 如果 q 小于等于 s，根据 status_only 的值返回相应结果
        if status_only:
            return 0
        else:
            return _expm_multiply_interval_core_0(A, X,
                    h, mu, q, norm_info, tol, ell, n0)
    elif not (q % s):
        # 如果 q 整除 s，根据 status_only 的值返回相应结果
        if status_only:
            return 1
        else:
            return _expm_multiply_interval_core_1(A, X,
                    h, mu, m_star, s, q, tol)
    elif (q % s):
        # 如果 q 不整除 s，根据 status_only 的值返回相应结果
        if status_only:
            return 2
        else:
            return _expm_multiply_interval_core_2(A, X,
                    h, mu, m_star, s, q, tol)
    else:
        # 如果以上条件都不满足，则抛出异常
        raise Exception('internal error')
# 定义一个辅助函数，处理当 q <= s 的情况
def _expm_multiply_interval_core_0(A, X, h, mu, q, norm_info, tol, ell, n0):
    """
    A helper function, for the case q <= s.
    """

    # 计算应用于大小为 t/q 的区间的新值 m_star 和 s
    if norm_info.onenorm() == 0:
        m_star, s = 0, 1
    else:
        # 设置标准化信息的缩放比例为 1/q
        norm_info.set_scale(1./q)
        # 调用 _fragment_3_1 函数计算 m_star 和 s
        m_star, s = _fragment_3_1(norm_info, n0, tol, ell=ell)
        # 恢复标准化信息的原始缩放比例
        norm_info.set_scale(1)

    # 循环执行 q 次
    for k in range(q):
        # 调用 _expm_multiply_simple_core 函数计算下一个时间步长的解
        X[k+1] = _expm_multiply_simple_core(A, X[k], h, mu, m_star, s)
    
    # 返回更新后的 X 和一个指示结果的值 0
    return X, 0


# 定义一个辅助函数，处理当 q > s 且 q % s == 0 的情况
def _expm_multiply_interval_core_1(A, X, h, mu, m_star, s, q, tol):
    """
    A helper function, for the case q > s and q % s == 0.
    """
    # 计算 d，即每个子区间的大小
    d = q // s
    input_shape = X.shape[1:]
    K_shape = (m_star + 1, ) + input_shape
    # 创建一个空的数组 K 用于存储中间结果
    K = np.empty(K_shape, dtype=X.dtype)
    
    # 循环执行 s 次
    for i in range(s):
        Z = X[i*d]
        K[0] = Z
        high_p = 0
        # 循环执行 d 次
        for k in range(1, d+1):
            F = K[0]
            c1 = _exact_inf_norm(F)
            # 循环执行 m_star 次
            for p in range(1, m_star+1):
                if p > high_p:
                    K[p] = h * A.dot(K[p-1]) / float(p)
                coeff = float(pow(k, p))
                F = F + coeff * K[p]
                inf_norm_K_p_1 = _exact_inf_norm(K[p])
                c2 = coeff * inf_norm_K_p_1
                if c1 + c2 <= tol * _exact_inf_norm(F):
                    break
                c1 = c2
            X[k + i*d] = np.exp(k*h*mu) * F
    
    # 返回更新后的 X 和一个指示结果的值 1
    return X, 1


# 定义一个辅助函数，处理当 q > s 且 q % s > 0 的情况
def _expm_multiply_interval_core_2(A, X, h, mu, m_star, s, q, tol):
    """
    A helper function, for the case q > s and q % s > 0.
    """
    # 计算 d，即每个子区间的大小
    d = q // s
    j = q // d
    r = q - d * j
    input_shape = X.shape[1:]
    K_shape = (m_star + 1, ) + input_shape
    # 创建一个空的数组 K 用于存储中间结果
    K = np.empty(K_shape, dtype=X.dtype)
    
    # 循环执行 j + 1 次
    for i in range(j + 1):
        Z = X[i*d]
        K[0] = Z
        high_p = 0
        if i < j:
            effective_d = d
        else:
            effective_d = r
        # 循环执行 effective_d 次
        for k in range(1, effective_d+1):
            F = K[0]
            c1 = _exact_inf_norm(F)
            # 循环执行 m_star 次
            for p in range(1, m_star+1):
                if p == high_p + 1:
                    K[p] = h * A.dot(K[p-1]) / float(p)
                    high_p = p
                coeff = float(pow(k, p))
                F = F + coeff * K[p]
                inf_norm_K_p_1 = _exact_inf_norm(K[p])
                c2 = coeff * inf_norm_K_p_1
                if c1 + c2 <= tol * _exact_inf_norm(F):
                    break
                c1 = c2
            X[k + i*d] = np.exp(k*h*mu) * F
    
    # 返回更新后的 X 和一个指示结果的值 2
    return X, 2
```