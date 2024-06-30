# `D:\src\scipysrc\scipy\scipy\optimize\_trustregion_constr\projections.py`

```
"""Basic linear factorizations needed by the solver."""

# 导入需要的库
from scipy.sparse import (bmat, csc_matrix, eye, issparse)
from scipy.sparse.linalg import LinearOperator
import scipy.linalg
import scipy.sparse.linalg
try:
    # 尝试导入额外的库
    from sksparse.cholmod import cholesky_AAt
    sksparse_available = True
except ImportError:
    # 如果导入失败，则发出警告并标记库不可用
    import warnings
    sksparse_available = False
import numpy as np
from warnings import warn

__all__ = [
    'orthogonality',
    'projections',
]

# 定义函数 orthogonality
def orthogonality(A, g):
    """Measure orthogonality between a vector and the null space of a matrix.

    Compute a measure of orthogonality between the null space
    of the (possibly sparse) matrix ``A`` and a given vector ``g``.

    The formula is a simplified (and cheaper) version of formula (3.13)
    from [1]_.
    ``orth =  norm(A g, ord=2)/(norm(A, ord='fro')*norm(g, ord=2))``.

    References
    ----------
    .. [1] Gould, Nicholas IM, Mary E. Hribar, and Jorge Nocedal.
           "On the solution of equality constrained quadratic
            programming problems arising in optimization."
            SIAM Journal on Scientific Computing 23.4 (2001): 1376-1395.
    """
    # 计算向量的范数
    norm_g = np.linalg.norm(g)
    # 计算矩阵 A 的 Frobenius 范数
    if issparse(A):
        norm_A = scipy.sparse.linalg.norm(A, ord='fro')
    else:
        norm_A = np.linalg.norm(A, ord='fro')

    # 检查范数是否为零
    if norm_g == 0 or norm_A == 0:
        return 0

    # 计算 A*g 的范数
    norm_A_g = np.linalg.norm(A.dot(g))
    # 计算正交性度量值
    orth = norm_A_g / (norm_A * norm_g)
    return orth


def normal_equation_projections(A, m, n, orth_tol, max_refin, tol):
    """Return linear operators for matrix A using ``NormalEquation`` approach.
    """
    # Cholesky 分解
    factor = cholesky_AAt(A)

    # z = x - A.T inv(A A.T) A x
    def null_space(x):
        v = factor(A.dot(x))
        z = x - A.T.dot(v)

        # 迭代细化以改善舍入误差
        # 描述在 [2]_ 中的算法 5.1.
        k = 0
        while orthogonality(A, z) > orth_tol:
            if k >= max_refin:
                break
            # z_next = z - A.T inv(A A.T) A z
            v = factor(A.dot(z))
            z = z - A.T.dot(v)
            k += 1

        return z

    # z = inv(A A.T) A x
    def least_squares(x):
        return factor(A.dot(x))

    # z = A.T inv(A A.T) x
    def row_space(x):
        return A.T.dot(factor(x))

    return null_space, least_squares, row_space


def augmented_system_projections(A, m, n, orth_tol, max_refin, tol):
    """Return linear operators for matrix A - ``AugmentedSystem``."""
    # 构造增广系统
    K = csc_matrix(bmat([[eye(n), A.T], [A, None]]))
    # LU 分解
    # TODO: 使用对称不定分解解决系统两倍速度更快（由于对称性）。
    try:
        solve = scipy.sparse.linalg.factorized(K)
    except RuntimeError:
        # 如果运行时错误，警告并返回使用密集的SVD分解进行因子分解
        warn("Singular Jacobian matrix. Using dense SVD decomposition to "
             "perform the factorizations.",
             stacklevel=3)
        return svd_factorization_projections(A.toarray(),
                                             m, n, orth_tol,
                                             max_refin, tol)

    # z = x - A.T inv(A A.T) A x
    # is computed solving the extended system:
    # [I A.T] * [ z ] = [x]
    # [A  O ]   [aux]   [0]
    def null_space(x):
        # v = [x]
        #     [0]
        v = np.hstack([x, np.zeros(m)])
        # lu_sol = [ z ]
        #          [aux]
        lu_sol = solve(v)
        z = lu_sol[:n]

        # Iterative refinement to improve roundoff
        # errors described in [2]_, algorithm 5.2.
        k = 0
        while orthogonality(A, z) > orth_tol:
            if k >= max_refin:
                break
            # new_v = [x] - [I A.T] * [ z ]
            #         [0]   [A  O ]   [aux]
            new_v = v - K.dot(lu_sol)
            # [I A.T] * [delta  z ] = new_v
            # [A  O ]   [delta aux]
            lu_update = solve(new_v)
            #  [ z ] += [delta  z ]
            #  [aux]    [delta aux]
            lu_sol += lu_update
            z = lu_sol[:n]
            k += 1

        # return z = x - A.T inv(A A.T) A x
        return z

    # z = inv(A A.T) A x
    # is computed solving the extended system:
    # [I A.T] * [aux] = [x]
    # [A  O ]   [ z ]   [0]
    def least_squares(x):
        # v = [x]
        #     [0]
        v = np.hstack([x, np.zeros(m)])
        # lu_sol = [aux]
        #          [ z ]
        lu_sol = solve(v)
        # return z = inv(A A.T) A x
        return lu_sol[n:m+n]

    # z = A.T inv(A A.T) x
    # is computed solving the extended system:
    # [I A.T] * [ z ] = [0]
    # [A  O ]   [aux]   [x]
    def row_space(x):
        # v = [0]
        #     [x]
        v = np.hstack([np.zeros(n), x])
        # lu_sol = [ z ]
        #          [aux]
        lu_sol = solve(v)
        # return z = A.T inv(A A.T) x
        return lu_sol[:n]

    return null_space, least_squares, row_space
def qr_factorization_projections(A, m, n, orth_tol, max_refin, tol):
    """Return linear operators for matrix A using ``QRFactorization`` approach.
    """
    # 使用 QR 分解计算矩阵 A.T 的正交分解 Q, 上三角矩阵 R 和置换矩阵 P
    Q, R, P = scipy.linalg.qr(A.T, pivoting=True, mode='economic')

    # 检查 R 矩阵的最后一行的无穷范数是否小于给定的阈值 tol
    if np.linalg.norm(R[-1, :], np.inf) < tol:
        # 若条件满足，警告用户使用 SVD 分解进行因子分解
        warn('Singular Jacobian matrix. Using SVD decomposition to ' +
             'perform the factorizations.',
             stacklevel=3)
        return svd_factorization_projections(A, m, n,
                                             orth_tol,
                                             max_refin,
                                             tol)

    # 计算零空间操作符 null_space(x)
    def null_space(x):
        # 计算 v = P inv(R) Q.T x
        aux1 = Q.T.dot(x)
        aux2 = scipy.linalg.solve_triangular(R, aux1, lower=False)
        v = np.zeros(m)
        v[P] = aux2
        # 计算 z = x - A.T dot(v)
        z = x - A.T.dot(v)

        # 迭代细化以改进舍入误差，参考文献 [2]，算法 5.1
        k = 0
        while orthogonality(A, z) > orth_tol:
            if k >= max_refin:
                break
            # 再次计算 v = P inv(R) Q.T z
            aux1 = Q.T.dot(z)
            aux2 = scipy.linalg.solve_triangular(R, aux1, lower=False)
            v[P] = aux2
            # 更新 z = z - A.T dot(v)
            z = z - A.T.dot(v)
            k += 1

        return z

    # 计算最小二乘空间操作符 least_squares(x)
    def least_squares(x):
        # 计算 z = P inv(R) Q.T x
        aux1 = Q.T.dot(x)
        aux2 = scipy.linalg.solve_triangular(R, aux1, lower=False)
        z = np.zeros(m)
        z[P] = aux2
        return z

    # 计算行空间操作符 row_space(x)
    def row_space(x):
        # 计算 z = Q inv(R.T) P.T x
        aux1 = x[P]
        aux2 = scipy.linalg.solve_triangular(R, aux1,
                                             lower=False,
                                             trans='T')
        z = Q.dot(aux2)
        return z

    return null_space, least_squares, row_space
    # 返回三个函数：null_space, least_squares, row_space
    def solve_svd(A, s, U, Vt):
        # 定义最小二乘法函数，计算 z = inv(A A.T) A x
        def least_squares(x):
            # 计算 aux1 = Vt.dot(x)
            aux1 = Vt.dot(x)
            # 计算 aux2 = 1/s * aux1
            aux2 = 1/s * aux1
            # 计算 z = U.dot(aux2)
            z = U.dot(aux2)
            return z
    
        # 定义行空间函数，计算 z = A.T inv(A A.T) x
        def row_space(x):
            # 计算 aux1 = U.T.dot(x)
            aux1 = U.T.dot(x)
            # 计算 aux2 = 1/s * aux1
            aux2 = 1/s * aux1
            # 计算 z = Vt.T.dot(aux2)
            z = Vt.T.dot(aux2)
            return z
    
        # 返回三个函数作为结果
        return null_space, least_squares, row_space
# 定义函数 projections，返回与给定矩阵 A 相关的三个线性算子

def projections(A, method=None, orth_tol=1e-12, max_refin=3, tol=1e-15):
    """Return three linear operators related with a given matrix A.

    Parameters
    ----------
    A : sparse matrix (or ndarray), shape (m, n)
        Matrix ``A`` used in the projection.
    method : string, optional
        Method used for compute the given linear
        operators. Should be one of:

            - 'NormalEquation': The operators
               will be computed using the
               so-called normal equation approach
               explained in [1]_. In order to do
               so the Cholesky factorization of
               ``(A A.T)`` is computed. Exclusive
               for sparse matrices.
            - 'AugmentedSystem': The operators
               will be computed using the
               so-called augmented system approach
               explained in [1]_. Exclusive
               for sparse matrices.
            - 'QRFactorization': Compute projections
               using QR factorization. Exclusive for
               dense matrices.
            - 'SVDFactorization': Compute projections
               using SVD factorization. Exclusive for
               dense matrices.

    orth_tol : float, optional
        Tolerance for iterative refinements.
    max_refin : int, optional
        Maximum number of iterative refinements.
    tol : float, optional
        Tolerance for singular values.

    Returns
    -------
    Z : LinearOperator, shape (n, n)
        Null-space operator. For a given vector ``x``,
        the null space operator is equivalent to apply
        a projection matrix ``P = I - A.T inv(A A.T) A``
        to the vector. It can be shown that this is
        equivalent to project ``x`` into the null space
        of A.
    LS : LinearOperator, shape (m, n)
        Least-squares operator. For a given vector ``x``,
        the least-squares operator is equivalent to apply a
        pseudoinverse matrix ``pinv(A.T) = inv(A A.T) A``
        to the vector. It can be shown that this vector
        ``pinv(A.T) x`` is the least_square solution to
        ``A.T y = x``.
    Y : LinearOperator, shape (n, m)
        Row-space operator. For a given vector ``x``,
        the row-space operator is equivalent to apply a
        projection matrix ``Q = A.T inv(A A.T)``
        to the vector.  It can be shown that this
        vector ``y = Q x``  the minimum norm solution
        of ``A y = x``.

    Notes
    -----
    Uses iterative refinements described in [1]
    during the computation of ``Z`` in order to
    cope with the possibility of large roundoff errors.

    References
    ----------
    .. [1] Gould, Nicholas IM, Mary E. Hribar, and Jorge Nocedal.
        "On the solution of equality constrained quadratic
        programming problems arising in optimization."
        SIAM Journal on Scientific Computing 23.4 (2001): 1376-1395.
    """
    # 获取矩阵 A 的形状（行数 m 和列数 n）
    m, n = np.shape(A)

    # 空矩阵的因子分解
    # 如果矩阵 A 是稀疏矩阵，则转换为压缩稀疏列（CSC）格式
    if m*n == 0:
        A = csc_matrix(A)

    # 检查参数
    if issparse(A):
        # 如果方法未指定，则默认为"AugmentedSystem"
        if method is None:
            method = "AugmentedSystem"
        # 如果指定的方法不在允许的列表中，则引发数值错误异常
        if method not in ("NormalEquation", "AugmentedSystem"):
            raise ValueError("Method not allowed for sparse matrix.")
        # 如果方法为"NormalEquation"但 scikit-sparse 不可用，则发出警告并使用"AugmentedSystem"方法代替
        if method == "NormalEquation" and not sksparse_available:
            warnings.warn("Only accepts 'NormalEquation' option when "
                          "scikit-sparse is available. Using "
                          "'AugmentedSystem' option instead.",
                          ImportWarning, stacklevel=3)
            method = 'AugmentedSystem'
    else:
        # 如果方法未指定，则默认为"QRFactorization"
        if method is None:
            method = "QRFactorization"
        # 如果指定的方法不在允许的列表中，则引发数值错误异常
        if method not in ("QRFactorization", "SVDFactorization"):
            raise ValueError("Method not allowed for dense array.")

    # 根据选择的方法调用不同的投影函数
    if method == 'NormalEquation':
        null_space, least_squares, row_space \
            = normal_equation_projections(A, m, n, orth_tol, max_refin, tol)
    elif method == 'AugmentedSystem':
        null_space, least_squares, row_space \
            = augmented_system_projections(A, m, n, orth_tol, max_refin, tol)
    elif method == "QRFactorization":
        null_space, least_squares, row_space \
            = qr_factorization_projections(A, m, n, orth_tol, max_refin, tol)
    elif method == "SVDFactorization":
        null_space, least_squares, row_space \
            = svd_factorization_projections(A, m, n, orth_tol, max_refin, tol)

    # 创建线性算子 Z, LS, Y 分别对应于 null_space, least_squares, row_space
    Z = LinearOperator((n, n), null_space)
    LS = LinearOperator((m, n), least_squares)
    Y = LinearOperator((n, m), row_space)

    # 返回三个线性算子作为结果
    return Z, LS, Y
```