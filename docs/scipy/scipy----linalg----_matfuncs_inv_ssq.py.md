# `D:\src\scipysrc\scipy\scipy\linalg\_matfuncs_inv_ssq.py`

```
"""
Matrix functions that use Pade approximation with inverse scaling and squaring.

"""
import warnings  # 导入警告模块

import numpy as np  # 导入NumPy库

from scipy.linalg._matfuncs_sqrtm import SqrtmError, _sqrtm_triu  # 导入开方矩阵函数相关的错误和函数
from scipy.linalg._decomp_schur import schur, rsf2csf  # 导入Schur分解相关函数
from scipy.linalg._matfuncs import funm  # 导入矩阵函数相关函数
from scipy.linalg import svdvals, solve_triangular  # 导入奇异值、三角解法函数
from scipy.sparse.linalg._interface import LinearOperator  # 导入线性操作接口类
from scipy.sparse.linalg import onenormest  # 导入稀疏矩阵1-范数估计函数
import scipy.special  # 导入SciPy特殊函数模块


class LogmRankWarning(UserWarning):  # 定义日志矩阵警告基类
    pass


class LogmExactlySingularWarning(LogmRankWarning):  # 定义精确奇异警告类
    pass


class LogmNearlySingularWarning(LogmRankWarning):  # 定义近似奇异警告类
    pass


class LogmError(np.linalg.LinAlgError):  # 定义日志矩阵错误类，继承自NumPy线性代数错误类
    pass


class FractionalMatrixPowerError(np.linalg.LinAlgError):  # 定义分数矩阵幂错误类，继承自NumPy线性代数错误类
    pass


# TODO 当SciPy操作符更加成熟时，重构或移动此类
class _MatrixM1PowerOperator(LinearOperator):
    """
    A representation of the linear operator (A - I)^p.
    """

    def __init__(self, A, p):
        if A.ndim != 2 or A.shape[0] != A.shape[1]:  # 检查A是否为方阵
            raise ValueError('expected A to be like a square matrix')  # 抛出值错误
        if p < 0 or p != int(p):  # 检查p是否为非负整数
            raise ValueError('expected p to be a non-negative integer')  # 抛出值错误
        self._A = A  # 设置对象的矩阵_A
        self._p = p  # 设置对象的幂次_p
        self.ndim = A.ndim  # 设置对象的维度
        self.shape = A.shape  # 设置对象的形状

    def _matvec(self, x):
        for i in range(self._p):  # 执行_p次迭代
            x = self._A.dot(x) - x  # 计算(A - I)^p * x
        return x  # 返回计算结果

    def _rmatvec(self, x):
        for i in range(self._p):  # 执行_p次迭代
            x = x.dot(self._A) - x  # 计算x * (A - I)^p
        return x  # 返回计算结果

    def _matmat(self, X):
        for i in range(self._p):  # 执行_p次迭代
            X = self._A.dot(X) - X  # 计算(A - I)^p * X
        return X  # 返回计算结果

    def _adjoint(self):
        return _MatrixM1PowerOperator(self._A.T, self._p)  # 返回对象的伴随操作


# TODO 当SciPy操作符更加成熟时，重构或移动此函数
def _onenormest_m1_power(A, p,
        t=2, itmax=5, compute_v=False, compute_w=False):
    """
    Efficiently estimate the 1-norm of (A - I)^p.

    Parameters
    ----------
    A : ndarray
        Matrix whose 1-norm of a power is to be computed.
    p : int
        Non-negative integer power.
    t : int, optional
        A positive parameter controlling the tradeoff between
        accuracy versus time and memory usage.
        Larger values take longer and use more memory
        but give more accurate output.
    itmax : int, optional
        Use at most this many iterations.
    compute_v : bool, optional
        Request a norm-maximizing linear operator input vector if True.
    compute_w : bool, optional
        Request a norm-maximizing linear operator output vector if True.

    Returns
    -------
    est : float
        An underestimate of the 1-norm of the sparse matrix.
    v : ndarray, optional
        The vector such that ||Av||_1 == est*||v||_1.
        It can be thought of as an input to the linear operator
        that gives an output with particularly large norm.
    """
    pass  # 函数未实现，仅提供了文档字符串
    # 定义一个可选参数 w，类型为 ndarray
    w : ndarray, optional
        # Av 的向量，其 1-范数相对较大。
        # 可以看作是线性操作符的输出，
        # 在范数上相对于输入较大。

    """
    # 调用 onenormest 函数，计算线性操作符 _MatrixM1PowerOperator(A, p) 的 1-范数估计值。
    # 参数 t 控制精度，itmax 控制迭代次数，compute_v 和 compute_w 控制是否计算 v 和 w。
    return onenormest(_MatrixM1PowerOperator(A, p),
            t=t, itmax=itmax, compute_v=compute_v, compute_w=compute_w)
    ```
# 计算复数的展向数（scalar unwinding number），参见文献 [1] 的公式 Eq. (5.3)
def _unwindk(z):
    # 使用 Eq. (5.3) 计算复数 z 的展向数，应该等于 (z - log(exp(z)) / (2 pi i)
    return int(np.ceil((z.imag - np.pi) / (2*np.pi)))


# 计算 r = a^(1 / (2^k)) - 1 的辅助函数，参见文献 [1] 的算法 (2)
def _briggs_helper_function(a, k):
    # 如果 k 小于 0 或者 k 不是整数，则抛出 ValueError
    if k < 0 or int(k) != k:
        raise ValueError('expected a nonnegative integer k')
    # 当 k 等于 0 时，直接返回 a - 1
    elif k == 0:
        return a - 1
    # 当 k 等于 1 时，返回 np.sqrt(a) - 1
    elif k == 1:
        return np.sqrt(a) - 1
    else:
        k_hat = k
        # 如果 a 的幅角大于等于 π/2，则对 a 进行平方根处理，并将 k_hat 置为 k - 1
        if np.angle(a) >= np.pi / 2:
            a = np.sqrt(a)
            k_hat = k - 1
        z0 = a - 1
        a = np.sqrt(a)
        r = 1 + a
        # 循环计算 r 的值，避免相减引起的取消问题
        for j in range(1, k_hat):
            a = np.sqrt(a)
            r = r * (1 + a)
        r = z0 / r
        return r


# 计算分数幂矩阵的超对角线条目，参见文献 [1] 的公式 Eq. (5.6)
def _fractional_power_superdiag_entry(l1, l2, t12, p):
    # 计算分数幂矩阵的超对角线条目，使用 Eq. (5.6)
    pass


这些注释提供了每个函数的目的、参数、返回值以及参考文献的引用，以便理解每个函数在何种背景下设计和使用。
    # 如果 l1 等于 l2，则根据公式计算超对角线上的矩阵幂
    if l1 == l2:
        f12 = t12 * p * l1**(p-1)
    # 如果 l2 和 l1 相差较大，则根据公式计算超对角线上的矩阵幂
    elif abs(l2 - l1) > abs(l1 + l2) / 2:
        f12 = t12 * ((l2**p) - (l1**p)) / (l2 - l1)
    else:
        # 使用文献引用中的公式 (5.5)
        # 计算 z 值
        z = (l2 - l1) / (l2 + l1)
        # 计算 l1 和 l2 的自然对数
        log_l1 = np.log(l1)
        log_l2 = np.log(l2)
        # 计算 z 的反双曲正切值
        arctanh_z = np.arctanh(z)
        # 计算临时变量 tmp_a，tmp_b 和 tmp_c
        tmp_a = t12 * np.exp((p/2)*(log_l2 + log_l1))
        # 调用 _unwindk 函数计算 tmp_u
        tmp_u = _unwindk(log_l2 - log_l1)
        # 根据 tmp_u 的值计算 tmp_b
        if tmp_u:
            tmp_b = p * (arctanh_z + np.pi * 1j * tmp_u)
        else:
            tmp_b = p * arctanh_z
        # 计算 tmp_c
        tmp_c = 2 * np.sinh(tmp_b) / (l2 - l1)
        # 计算最终结果 f12
        f12 = tmp_a * tmp_c
    # 返回计算得到的超对角线上的矩阵幂
    return f12
# 计算矩阵对数的超对角线条目
def _logm_superdiag_entry(l1, l2, t12):
    """
    Compute a superdiagonal entry of a matrix logarithm.

    This is like Eq. (11.28) in [1]_, except the determination of whether
    l1 and l2 are sufficiently far apart has been modified.

    Parameters
    ----------
    l1 : complex
        矩阵的一个对角线条目。
    l2 : complex
        矩阵的一个对角线条目。
    t12 : complex
        矩阵的一个超对角线条目。

    Returns
    -------
    f12 : complex
        矩阵对数的一个超对角线条目。

    Notes
    -----
    Care has been taken to return a real number if possible when
    all of the inputs are real numbers.

    References
    ----------
    .. [1] Nicholas J. Higham (2008)
           "Functions of Matrices: Theory and Computation"
           ISBN 978-0-898716-46-7

    """
    # 如果 l1 和 l2 相等
    if l1 == l2:
        f12 = t12 / l1
    # 如果 l1 和 l2 不相等且它们之间的距离足够大
    elif abs(l2 - l1) > abs(l1 + l2) / 2:
        f12 = t12 * (np.log(l2) - np.log(l1)) / (l2 - l1)
    # 否则
    else:
        # 计算 z 和 u
        z = (l2 - l1) / (l2 + l1)
        u = _unwindk(np.log(l2) - np.log(l1))
        # 如果 u 存在
        if u:
            f12 = t12 * 2 * (np.arctanh(z) + np.pi*1j*u) / (l2 - l1)
        # 如果 u 不存在
        else:
            f12 = t12 * 2 * np.arctanh(z) / (l2 - l1)
    return f12


# 逆缩放和平方辅助函数，用于Pade逼近
def _inverse_squaring_helper(T0, theta):
    """
    A helper function for inverse scaling and squaring for Pade approximation.

    Parameters
    ----------
    T0 : (N, N) array_like upper triangular
        参与逆缩放和平方的矩阵。
    theta : indexable
        必须提供 theta[1] .. theta[7] 的值。
        这些值表示与正在计算的矩阵函数相关的Pade逼近的界限。
        例如，计算矩阵对数和计算分数矩阵幂需要不同的 theta 值。

    Returns
    -------
    R : (N, N) array_like upper triangular
        T0 的零或多个矩阵平方根的组合，减去单位矩阵。
    s : non-negative integer
        取的平方根的数量。
    m : positive integer
        Pade逼近的阶数。

    Notes
    -----
    This subroutine appears as a chunk of lines within
    a couple of published algorithms; for example it appears
    as lines 4--35 in algorithm (3.1) of [1]_, and
    as lines 3--34 in algorithm (4.1) of [2]_.
    The instances of 'goto line 38' in algorithm (3.1) of [1]_
    probably mean 'goto line 36' and have been interpreted accordingly.

    References
    ----------
    .. [1] Nicholas J. Higham and Lijing Lin (2013)
           "An Improved Schur-Pade Algorithm for Fractional Powers
           of a Matrix and their Frechet Derivatives."

    .. [2] Awad H. Al-Mohy and Nicholas J. Higham (2012)
           "Improved Inverse Scaling and Squaring Algorithms
           for the Matrix Logarithm."
           SIAM Journal on Scientific Computing, 34 (4). C152-C169.
           ISSN 1095-7197

    """
    # 检查输入的上三角方阵 T0 是否是二维且为方阵
    if len(T0.shape) != 2 or T0.shape[0] != T0.shape[1]:
        raise ValueError('expected an upper triangular square matrix')
    # 获取方阵的维度
    n, n = T0.shape
    # 将 T0 赋值给 T
    T = T0

    # 寻找 s0，使得某个对角矩阵的谱半径不超过 theta[7]
    # 注意，由于 theta[7] < 1，若 T 的任何对角元素为零，此搜索将不会终止
    s0 = 0
    tmp_diag = np.diag(T)
    # 如果 T 的对角元素不全为非零，则抛出异常
    if np.count_nonzero(tmp_diag) != n:
        raise Exception('Diagonal entries of T must be nonzero')
    # 循环直至某个条件不满足
    while np.max(np.absolute(tmp_diag - 1), initial=0.) > theta[7]:
        tmp_diag = np.sqrt(tmp_diag)
        s0 += 1

    # 对 T 进行 s0 次上三角矩阵的平方根操作
    for i in range(s0):
        T = _sqrtm_triu(T)

    # 此段流程控制略显奇怪，因为我正在翻译带有出版物中的 GOTO 的算法描述
    s = s0
    k = 0
    # 计算 T 的 2 和 3 阶范数的 m1 次幂，然后取平方根和立方根
    d2 = _onenormest_m1_power(T, 2) ** (1/2)
    d3 = _onenormest_m1_power(T, 3) ** (1/3)
    # 取 d2 和 d3 的最大值
    a2 = max(d2, d3)
    m = None
    # 遍历 theta[1] 和 theta[2]，找到满足条件的 m 值
    for i in (1, 2):
        if a2 <= theta[i]:
            m = i
            break
    # 如果 m 仍为 None，则进入循环
    while m is None:
        if s > s0:
            d3 = _onenormest_m1_power(T, 3) ** (1/3)
        # 计算 T 的 4 阶范数的 m1 次幂，然后取其 4 次根
        d4 = _onenormest_m1_power(T, 4) ** (1/4)
        # 取 d3 和 d4 的最大值
        a3 = max(d3, d4)
        # 如果 a3 <= theta[7]，则确定 m 为某个 j1
        if a3 <= theta[7]:
            j1 = min(i for i in (3, 4, 5, 6, 7) if a3 <= theta[i])
            if j1 <= 6:
                m = j1
                break
            # 若 a3 / 2 <= theta[5] 且 k < 2，则进行一些操作
            elif a3 / 2 <= theta[5] and k < 2:
                k += 1
                T = _sqrtm_triu(T)
                s += 1
                continue
        # 计算 T 的 5 阶范数的 m1 次幂，然后取其 5 次根
        d5 = _onenormest_m1_power(T, 5) ** (1/5)
        # 取 d4 和 d5 的最大值
        a4 = max(d4, d5)
        # 取 eta 为 a3 和 a4 的最小值
        eta = min(a3, a4)
        # 遍历 theta[6] 和 theta[7]，找到满足条件的 m 值
        for i in (6, 7):
            if eta <= theta[i]:
                m = i
                break
        # 如果找到 m，则跳出循环
        if m is not None:
            break
        # 否则对 T 进行平方根操作，增加 s
        T = _sqrtm_triu(T)
        s += 1

    # 这里减去单位矩阵是多余的，因为对角线将被替换以提高数值精度，
    # 但这种表述应有助于理解 R 的含义
    R = T - np.identity(n)

    # 如果 T0 的对角元素都为正实数，则进行替换对角和第一超对角线
    # 使用具有更少减法取消的公式
    has_principal_branch = all(x.real > 0 or x.imag != 0 for x in np.diag(T0))
    if has_principal_branch:
        for j in range(n):
            a = T0[j, j]
            r = _briggs_helper_function(a, s)
            R[j, j] = r
        p = np.exp2(-s)
        for j in range(n-1):
            l1 = T0[j, j]
            l2 = T0[j+1, j+1]
            t12 = T0[j, j+1]
            f12 = _fractional_power_superdiag_entry(l1, l2, t12, p)
            R[j, j+1] = f12

    # 返回 T-I 矩阵，平方根数量和 Pade 程度
    # 检查矩阵 R 是否为上三角矩阵，如果不是则抛出异常
    if not np.array_equal(R, np.triu(R)):
        raise Exception('R is not upper triangular')
    # 如果 R 是上三角矩阵，则返回 R, s, m
    return R, s, m
# A helper function to compute a specific constant used in the Pade approximation.
def _fractional_power_pade_constant(i, t):
    # 如果 i 小于 1，则引发数值错误异常
    if i < 1:
        raise ValueError('expected a positive integer i')
    # 如果 t 不在 -1 和 1 之间（不包括），则引发数值错误异常
    if not (-1 < t < 1):
        raise ValueError('expected -1 < t < 1')
    # 如果 i 等于 1，返回 -t
    if i == 1:
        return -t
    # 如果 i 是偶数
    elif i % 2 == 0:
        j = i // 2
        return (-j + t) / (2 * (2*j - 1))
    # 如果 i 是奇数
    elif i % 2 == 1:
        j = (i - 1) // 2
        return (-j - t) / (2 * (2*j + 1))
    # 如果 i 的值异常，引发异常
    else:
        raise Exception(f'unnexpected value of i, i = {i}')


def _fractional_power_pade(R, t, m):
    """
    Evaluate the Pade approximation of a fractional matrix power.

    Evaluate the degree-m Pade approximation of R
    to the fractional matrix power t using the continued fraction
    in bottom-up fashion using algorithm (4.1) in [1]_.

    Parameters
    ----------
    R : (N, N) array_like
        Upper triangular matrix whose fractional power to evaluate.
    t : float
        Fractional power between -1 and 1 exclusive.
    m : positive integer
        Degree of Pade approximation.

    Returns
    -------
    U : (N, N) array_like
        The degree-m Pade approximation of R to the fractional power t.
        This matrix will be upper triangular.

    References
    ----------
    .. [1] Nicholas J. Higham and Lijing lin (2011)
           "A Schur-Pade Algorithm for Fractional Powers of a Matrix."
           SIAM Journal on Matrix Analysis and Applications,
           32 (3). pp. 1056-1078. ISSN 0895-4798

    """
    # 如果 m 小于 1 或者 m 不是整数，则引发数值错误异常
    if m < 1 or int(m) != m:
        raise ValueError('expected a positive integer m')
    # 如果 t 不在 -1 和 1 之间（不包括），则引发数值错误异常
    if not (-1 < t < 1):
        raise ValueError('expected -1 < t < 1')
    # 将 R 转换为 NumPy 数组
    R = np.asarray(R)
    # 如果 R 不是二维数组或者不是方阵，引发数值错误异常
    if len(R.shape) != 2 or R.shape[0] != R.shape[1]:
        raise ValueError('expected an upper triangular square matrix')
    # 获取 R 的大小
    n, n = R.shape
    # 创建单位矩阵 ident
    ident = np.identity(n)
    # 计算 Y，使用 Pade 近似的常数乘以 R
    Y = R * _fractional_power_pade_constant(2*m, t)
    # 从 2*m-1 到 1 的反向迭代
    for j in range(2*m - 1, 0, -1):
        # 计算 rhs，使用 Pade 近似的常数乘以 R
        rhs = R * _fractional_power_pade_constant(j, t)
        # 解方程 ident + Y，得到 Y
        Y = solve_triangular(ident + Y, rhs)
    # 计算 U
    U = ident + Y
    # 如果 U 不等于它的上三角形式，引发异常
    if not np.array_equal(U, np.triu(U)):
        raise Exception('U is not upper triangular')
    # 返回结果矩阵 U
    return U


def _remainder_matrix_power_triu(T, t):
    """
    Compute a fractional power of an upper triangular matrix.

    The fractional power is restricted to fractions -1 < t < 1.
    This uses algorithm (3.1) of [1]_.
    The Pade approximation itself uses algorithm (4.1) of [2]_.

    Parameters
    ----------
    T : (N, N) array_like
        Upper triangular matrix whose fractional power to evaluate.
    t : float
        Fractional power between -1 and 1 exclusive.

    Returns
    -------
    X : (N, N) array_like
        The fractional power of the matrix.

    References
    ----------
    .. [1] Nicholas J. Higham and Lijing Lin (2013)
           "An Improved Schur-Pade Algorithm for Fractional Powers
           of a Matrix and their Frechet Derivatives."

    """
    # 定义一个映射，将整数映射到对应的 theta 值，用于 Schur-Pade 算法中的参数选择
    m_to_theta = {
            1: 1.51e-5,
            2: 2.24e-3,
            3: 1.88e-2,
            4: 6.04e-2,
            5: 1.24e-1,
            6: 2.00e-1,
            7: 2.79e-1,
            }
    
    # 获取矩阵 T 的形状，并将其赋值给 n
    n, n = T.shape
    
    # 复制矩阵 T 到 T0
    T0 = T
    
    # 提取 T0 的对角线元素，并赋值给 T0_diag
    T0_diag = np.diag(T0)
    
    # 如果 T0 是对角矩阵（所有非对角线元素均为零），则使用 T0_diag 的 t 次幂作为 U
    if np.array_equal(T0, np.diag(T0_diag)):
        U = np.diag(T0_diag ** t)
    else:
        # 否则，调用 _inverse_squaring_helper 函数，计算逆平方辅助函数返回的 R, s, m
        R, s, m = _inverse_squaring_helper(T0, m_to_theta)
    
        # 计算 Pade 近似值。
        # 注意这个函数期望输入是逆平方辅助函数返回的矩阵的负数
        U = _fractional_power_pade(-R, t, m)
    
        # 撤销逆缩放和平方操作。
        # 如果 T0 的主分支不存在，则可能需要更简单的处理；
        # 当 T0 的对角元素为负数且虚部为 0 时会发生这种情况。
        eivals = np.diag(T0)
        has_principal_branch = all(x.real > 0 or x.imag != 0 for x in eivals)
        for i in range(s, -1, -1):
            if i < s:
                U = U.dot(U)
            else:
                if has_principal_branch:
                    # 计算对角元素的幂次 p，更新 U 的对角线
                    p = t * np.exp2(-i)
                    U[np.diag_indices(n)] = T0_diag ** p
                    
                    # 更新超对角线元素
                    for j in range(n-1):
                        l1 = T0[j, j]
                        l2 = T0[j+1, j+1]
                        t12 = T0[j, j+1]
                        f12 = _fractional_power_superdiag_entry(l1, l2, t12, p)
                        U[j, j+1] = f12
    
    # 检查 U 是否为上三角矩阵，若不是则抛出异常
    if not np.array_equal(U, np.triu(U)):
        raise Exception('U is not upper triangular')
    
    # 返回计算结果 U
    return U
def _remainder_matrix_power(A, t):
    """
    Compute the fractional power of a matrix, for fractions -1 < t < 1.

    This uses algorithm (3.1) of [1]_.
    The Pade approximation itself uses algorithm (4.1) of [2]_.

    Parameters
    ----------
    A : (N, N) array_like
        Matrix whose fractional power to evaluate.
    t : float
        Fractional power between -1 and 1 exclusive.

    Returns
    -------
    X : (N, N) array_like
        The fractional power of the matrix.

    References
    ----------
    .. [1] Nicholas J. Higham and Lijing Lin (2013)
           "An Improved Schur-Pade Algorithm for Fractional Powers
           of a Matrix and their Frechet Derivatives."

    .. [2] Nicholas J. Higham and Lijing lin (2011)
           "A Schur-Pade Algorithm for Fractional Powers of a Matrix."
           SIAM Journal on Matrix Analysis and Applications,
           32 (3). pp. 1056-1078. ISSN 0895-4798

    """
    # 将输入的矩阵转换为 NumPy 数组
    A = np.asarray(A)
    # 检查矩阵的维度是否为二维且为方阵
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('input must be a square array')

    # 获取矩阵的行数和列数
    n, n = A.shape

    # 如果矩阵已经是上三角矩阵，则不需要转换
    if np.array_equal(A, np.triu(A)):
        Z = None
        T = A
    else:
        # 如果矩阵是实数类型，尝试进行施密特正交化（Schur 分解）
        if np.isrealobj(A):
            T, Z = schur(A)
            # 如果 Schur 分解得到的上三角矩阵不是上三角形式，则进行正交化
            if not np.array_equal(T, np.triu(T)):
                T, Z = rsf2csf(T, Z)
        else:
            # 如果矩阵是复数类型，直接进行 Schur 分解
            T, Z = schur(A, output='complex')

    # 检查上三角矩阵的对角线是否有零元素，因为这会影响倒数放大和平方操作
    T_diag = np.diag(T)
    if np.count_nonzero(T_diag) != n:
        raise FractionalMatrixPowerError(
                'cannot use inverse scaling and squaring to find '
                'the fractional matrix power of a singular matrix')

    # 如果上三角矩阵是实数类型且对角线上有负数，则将其转换为复数类型
    if np.isrealobj(T) and np.min(T_diag) < 0:
        T = T.astype(complex)

    # 计算上三角矩阵的分数幂，并根据需要进行去三角化
    U = _remainder_matrix_power_triu(T, t)
    if Z is not None:
        ZH = np.conjugate(Z).T
        return Z.dot(U).dot(ZH)
    else:
        return U


def _fractional_matrix_power(A, p):
    """
    Compute the fractional power of a matrix.

    See the fractional_matrix_power docstring in matfuncs.py for more info.

    """
    # 将输入的矩阵转换为 NumPy 数组
    A = np.asarray(A)
    # 检查矩阵的维度是否为二维且为方阵
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected a square matrix')
    # 如果指数 p 是整数，则调用 NumPy 的标准幂函数
    if p == int(p):
        return np.linalg.matrix_power(A, int(p))
    # 计算矩阵的奇异值
    s = svdvals(A)
    # 如果矩阵是奇异的，则无法使用倒数放大和平方操作
    # 这是因为重复进行平方根操作的过程
    # 如果矩阵分解无法收敛到单位矩阵，则执行以下操作
    if s[-1]:
        # 计算相对于矩阵求逆的条件数，
        # 并根据条件数决定选择 floor(p) 还是 ceil(p)
        k2 = s[0] / s[-1]
        p1 = p - np.floor(p)
        p2 = p - np.ceil(p)
        if p1 * k2 ** (1 - p1) <= -p2 * k2:
            a = int(np.floor(p))
            b = p1
        else:
            a = int(np.ceil(p))
            b = p2
        try:
            # 计算余子矩阵的幂次方
            R = _remainder_matrix_power(A, b)
            # 计算矩阵的幂次方
            Q = np.linalg.matrix_power(A, a)
            # 返回 Q 和 R 的乘积
            return Q.dot(R)
        except np.linalg.LinAlgError:
            pass
    # 如果 p 是负数，则放弃计算。
    # 如果 p 是非负数，则退化到通用的 funm 计算。
    if p < 0:
        # 创建与 A 类型相同的空矩阵，并填充 NaN
        X = np.empty_like(A)
        X.fill(np.nan)
        return X
    else:
        p1 = p - np.floor(p)
        a = int(np.floor(p))
        b = p1
        # 使用 funm 函数计算函数的矩阵
        R, info = funm(A, lambda x: pow(x, b), disp=False)
        # 计算矩阵的幂次方
        Q = np.linalg.matrix_power(A, a)
        # 返回 Q 和 R 的乘积
        return Q.dot(R)
# 定义一个函数，用于计算上三角矩阵的对数。

"""
Compute matrix logarithm of an upper triangular matrix.

The matrix logarithm is the inverse of
expm: expm(logm(`T`)) == `T`

Parameters
----------
T : (N, N) array_like
    Upper triangular matrix whose logarithm to evaluate

Returns
-------
logm : (N, N) ndarray
    Matrix logarithm of `T`

References
----------
.. [1] Awad H. Al-Mohy and Nicholas J. Higham (2012)
       "Improved Inverse Scaling and Squaring Algorithms
       for the Matrix Logarithm."
       SIAM Journal on Scientific Computing, 34 (4). C152-C169.
       ISSN 1095-7197

.. [2] Nicholas J. Higham (2008)
       "Functions of Matrices: Theory and Computation"
       ISBN 978-0-898716-46-7

.. [3] Nicholas J. Higham and Lijing lin (2011)
       "A Schur-Pade Algorithm for Fractional Powers of a Matrix."
       SIAM Journal on Matrix Analysis and Applications,
       32 (3). pp. 1056-1078. ISSN 0895-4798
"""

# 将输入矩阵 `T` 转换为 NumPy 数组
T = np.asarray(T)

# 检查输入矩阵的形状，确保是一个正方形的上三角矩阵
if len(T.shape) != 2 or T.shape[0] != T.shape[1]:
    raise ValueError('expected an upper triangular square matrix')
n, n = T.shape

# 构造一个类型适当的 T0，根据 T 的 dtype 和其谱的情况
T_diag = np.diag(T)
keep_it_real = np.isrealobj(T) and np.min(T_diag, initial=0.) >= 0
if keep_it_real:
    T0 = T
else:
    T0 = T.astype(complex)

# 定义 Table (2.1) 中给出的边界
theta = (None,
        1.59e-5, 2.31e-3, 1.94e-2, 6.21e-2,
        1.28e-1, 2.06e-1, 2.88e-1, 3.67e-1,
        4.39e-1, 5.03e-1, 5.60e-1, 6.09e-1,
        6.52e-1, 6.89e-1, 7.21e-1, 7.49e-1)

# 调用 _inverse_squaring_helper 函数，返回 R, s, m
R, s, m = _inverse_squaring_helper(T0, theta)

# 计算 U = 2**s r_m(T - I)，使用偏分式展开 (1.1)
# 需要使用与 degree-m 高斯-勒让德积分相对应的节点和权重
nodes, weights = scipy.special.p_roots(m)
nodes = nodes.real
if nodes.shape != (m,) or weights.shape != (m,):
    raise Exception('internal error')
nodes = 0.5 + 0.5 * nodes
weights = 0.5 * weights
ident = np.identity(n)
U = np.zeros_like(R)
for alpha, beta in zip(weights, nodes):
    # 使用 solve_triangular 解方程 ident + beta*R，alpha*R
    U += solve_triangular(ident + beta*R, alpha*R)
# 将 U 乘以 2 的 s 次方
U *= np.exp2(s)

# 如果 T0 的主分支不存在，则跳过此步骤；
# 当 T0 的对角线条目有负实部且虚部为 0 时会发生这种情况
has_principal_branch = all(x.real > 0 or x.imag != 0 for x in np.diag(T0))
    # 如果存在主分支

        # 重新计算 U 的对角线条目。
        U[np.diag_indices(n)] = np.log(np.diag(T0))

        # 重新计算 U 的超对角线条目。
        # 当新版本 np.diagonal() 可用时，应更新此代码的索引。
        for i in range(n-1):
            l1 = T0[i, i]
            l2 = T0[i+1, i+1]
            t12 = T0[i, i+1]
            U[i, i+1] = _logm_superdiag_entry(l1, l2, t12)

    # 返回上三角矩阵 U 的对数矩阵。
    if not np.array_equal(U, np.triu(U)):
        raise Exception('U 不是上三角矩阵')
    return U
# 定义一个函数，用于处理非奇异的上三角矩阵，并确保其对数的可计算性
def _logm_force_nonsingular_triangular_matrix(T, inplace=False):
    # 输入的矩阵应为上三角矩阵。
    # tri_eps 是一个经验值，不是机器精度。
    tri_eps = 1e-20
    # 计算矩阵 T 对角线元素的绝对值
    abs_diag = np.absolute(np.diag(T))
    # 如果存在对角线元素为零，则表示输入矩阵确实是奇异的
    if np.any(abs_diag == 0):
        # 发出警告，指出输入矩阵确实是奇异的
        exact_singularity_msg = 'The logm input matrix is exactly singular.'
        warnings.warn(exact_singularity_msg, LogmExactlySingularWarning, stacklevel=3)
        # 如果不是原地操作，则复制矩阵 T
        if not inplace:
            T = T.copy()
        n = T.shape[0]
        # 对所有对角线元素为零的情况，添加一个小量 tri_eps
        for i in range(n):
            if not T[i, i]:
                T[i, i] = tri_eps
    # 如果对角线元素小于 tri_eps，则表示矩阵接近奇异
    elif np.any(abs_diag < tri_eps):
        # 发出警告，指出输入矩阵可能接近奇异
        near_singularity_msg = 'The logm input matrix may be nearly singular.'
        warnings.warn(near_singularity_msg, LogmNearlySingularWarning, stacklevel=3)
    # 返回处理后的矩阵 T
    return T


# 定义一个函数，计算矩阵的对数
def _logm(A):
    """
    Compute the matrix logarithm.

    See the logm docstring in matfuncs.py for more info.

    Notes
    -----
    In this function we look at triangular matrices that are similar
    to the input matrix. If any diagonal entry of such a triangular matrix
    is exactly zero then the original matrix is singular.
    The matrix logarithm does not exist for such matrices,
    but in such cases we will pretend that the diagonal entries that are zero
    are actually slightly positive by an ad-hoc amount, in the interest
    of returning something more useful than NaN. This will cause a warning.

    """
    # 将输入转换为 NumPy 数组
    A = np.asarray(A)
    # 确保输入矩阵是方阵
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected a square matrix')

    # 如果输入矩阵的数据类型是整数，则将其复制为浮点数类型
    if issubclass(A.dtype.type, np.integer):
        A = np.asarray(A, dtype=float)

    # 检查输入矩阵是否为实数类型
    keep_it_real = np.isrealobj(A)
    try:
        # 如果 A 是上三角矩阵，则调用处理函数 _logm_force_nonsingular_triangular_matrix
        if np.array_equal(A, np.triu(A)):
            A = _logm_force_nonsingular_triangular_matrix(A)
            # 如果处理后的矩阵的最小对角线元素小于 0，则将其转换为复数类型
            if np.min(np.diag(A), initial=0.) < 0:
                A = A.astype(complex)
            return _logm_triu(A)
        else:
            # 如果矩阵不是上三角，则进行 Schur 分解
            if keep_it_real:
                T, Z = schur(A)
                # 如果 T 不是上三角矩阵，则转换为复数 Schur 形式
                if not np.array_equal(T, np.triu(T)):
                    T, Z = rsf2csf(T, Z)
            else:
                T, Z = schur(A, output='complex')
            # 处理 Schur 分解后的矩阵 T
            T = _logm_force_nonsingular_triangular_matrix(T, inplace=True)
            U = _logm_triu(T)
            ZH = np.conjugate(Z).T
            # 返回计算得到的矩阵对数
            return Z.dot(U).dot(ZH)
    except (SqrtmError, LogmError):
        # 如果计算出现错误，则返回一个填充了 NaN 的矩阵
        X = np.empty_like(A)
        X.fill(np.nan)
        return X
```