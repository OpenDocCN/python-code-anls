# `D:\src\scipysrc\scipy\scipy\optimize\_lsq\common.py`

```
# 导入数学模块中的copysign函数
from math import copysign

# 导入numpy库，并使用别名np
import numpy as np

# 导入线性代数模块中的norm函数
from numpy.linalg import norm

# 导入scipy线性代数模块中的cho_factor和cho_solve函数，以及LinAlgError异常类
from scipy.linalg import cho_factor, cho_solve, LinAlgError

# 导入scipy稀疏矩阵模块中的issparse函数，以及线性操作符和转换为线性操作符的函数
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator, aslinearoperator

# 定义EPS为float类型的机器精度
EPS = np.finfo(float).eps


# 与信赖域问题相关的函数
def intersect_trust_region(x, s, Delta):
    """Find the intersection of a line with the boundary of a trust region.

    This function solves the quadratic equation with respect to t
    ||(x + s*t)||**2 = Delta**2.

    Parameters
    ----------
    x : ndarray, shape (n,)
        Point from which the line starts.
    s : ndarray, shape (n,)
        Direction vector of the line.
    Delta : float
        Radius of the trust region.

    Returns
    -------
    t_neg, t_pos : tuple of float
        Negative and positive roots.

    Raises
    ------
    ValueError
        If `s` is zero or `x` is not within the trust region.
    """
    # 计算向量s的内积，作为二次方程的系数a
    a = np.dot(s, s)
    if a == 0:
        # 如果a为零，抛出值错误
        raise ValueError("`s` is zero.")

    # 计算向量x和向量s的内积，作为二次方程的系数b
    b = np.dot(x, s)

    # 计算 ||x||^2 - Delta^2，作为二次方程的常数项c
    c = np.dot(x, x) - Delta**2
    if c > 0:
        # 如果c大于零，说明点x不在信赖域内，抛出值错误
        raise ValueError("`x` is not within the trust region.")

    # 计算判别式的平方根
    d = np.sqrt(b*b - a*c)  # Root from one fourth of the discriminant.

    # 避免数值上的损失，参考"Numerical Recipes"
    q = -(b + copysign(d, b))
    t1 = q / a
    t2 = c / q

    # 返回两个根，确保 t1 < t2
    if t1 < t2:
        return t1, t2
    else:
        return t2, t1


def solve_lsq_trust_region(n, m, uf, s, V, Delta, initial_alpha=None,
                           rtol=0.01, max_iter=10):
    """Solve a trust-region problem arising in least-squares minimization.

    This function implements a method described by J. J. More [1]_ and used
    in MINPACK, but it relies on a single SVD of Jacobian instead of series
    of Cholesky decompositions. Before running this function, compute:
    ``U, s, VT = svd(J, full_matrices=False)``.

    Parameters
    ----------
    n : int
        Number of variables.
    m : int
        Number of residuals.
    uf : ndarray
        Computed as U.T.dot(f).
    s : ndarray
        Singular values of J.
    V : ndarray
        Transpose of VT.
    Delta : float
        Radius of a trust region.
    initial_alpha : float, optional
        Initial guess for alpha, which might be available from a previous
        iteration. If None, determined automatically.
    rtol : float, optional
        Stopping tolerance for the root-finding procedure. Namely, the
        solution ``p`` will satisfy ``abs(norm(p) - Delta) < rtol * Delta``.
    max_iter : int, optional
        Maximum allowed number of iterations for the root-finding procedure.

    Returns
    -------
    p : ndarray, shape (n,)
        Found solution of a trust-region problem.
    alpha : float
        Positive value such that (J.T*J + alpha*I)*p = -J.T*f.
        Sometimes called Levenberg-Marquardt parameter.
    n_iter : int
        Number of iterations made by root-finding procedure. Zero means
        that Gauss-Newton step was selected as the solution.

    References
    ----------
    [1] J. J. More, "The Levenberg-Marquardt algorithm: implementation and theory,"
        in Numerical Analysis, ed. G. A. Watson, Springer-Verlag, 1978, pp. 105-116.
    """
    # 这个函数解决最小二乘问题中的信赖域问题

    # 先验条件检查
    if initial_alpha is None:
        # 如果未提供初始alpha，则自动确定
        pass

    # 返回三个变量：p是信赖域问题的解，alpha是Levenberg-Marquardt参数，n_iter是迭代次数
    return p, alpha, n_iter
    """
    def phi_and_derivative(alpha, suf, s, Delta):
        """Function of which to find zero.

        It is defined as "norm of regularized (by alpha) least-squares
        solution minus `Delta`". Refer to [1]_.
        """
        # 计算正则化最小二乘解的范数减去 Delta
        denom = s**2 + alpha
        p_norm = norm(suf / denom)
        # 定义 phi 为 p_norm 减去 Delta
        phi = p_norm - Delta
        # 计算 phi 对 alpha 的导数
        phi_prime = -np.sum(suf ** 2 / denom**3) / p_norm
        return phi, phi_prime

    # 计算 suf = s 乘以 uf
    suf = s * uf

    # 检查矩阵 J 是否满秩并尝试高斯-牛顿步骤。
    if m >= n:
        # 设置阈值为 EPS * m * s 的第一个元素
        threshold = EPS * m * s[0]
        # 判断 s 的最后一个元素是否大于阈值
        full_rank = s[-1] > threshold
    else:
        full_rank = False

    if full_rank:
        # 计算 p = -V 乘以 (uf / s)，并检查其范数是否小于等于 Delta
        p = -V.dot(uf / s)
        if norm(p) <= Delta:
            return p, 0.0, 0

    # 计算 alpha 的上界
    alpha_upper = norm(suf) / Delta

    if full_rank:
        # 计算 phi 和 phi 对 alpha 的导数，初始 alpha 的下界
        phi, phi_prime = phi_and_derivative(0.0, suf, s, Delta)
        alpha_lower = -phi / phi_prime
    else:
        alpha_lower = 0.0

    if initial_alpha is None or not full_rank and initial_alpha == 0:
        # 如果初始 alpha 未指定或者不是满秩且初始 alpha 为 0，选择一个合适的 alpha
        alpha = max(0.001 * alpha_upper, (alpha_lower * alpha_upper)**0.5)
    else:
        alpha = initial_alpha

    # 开始迭代求解
    for it in range(max_iter):
        if alpha < alpha_lower or alpha > alpha_upper:
            alpha = max(0.001 * alpha_upper, (alpha_lower * alpha_upper)**0.5)

        # 计算 phi 和 phi 对 alpha 的导数
        phi, phi_prime = phi_and_derivative(alpha, suf, s, Delta)

        if phi < 0:
            alpha_upper = alpha

        ratio = phi / phi_prime
        alpha_lower = max(alpha_lower, alpha - ratio)
        alpha -= (phi + Delta) * ratio / Delta

        if np.abs(phi) < rtol * Delta:
            break

    # 计算 p = -V 乘以 (suf / (s**2 + alpha))，并将 p 的范数调整为 Delta
    p = -V.dot(suf / (s**2 + alpha))

    # 调整 p 的范数为 Delta，以防止 p 超出信任区域（可能会在后续引起问题）
    p *= Delta / norm(p)

    return p, alpha, it + 1
# 解决二维一般信赖域问题的函数。
def solve_trust_region_2d(B, g, Delta):
    """Solve a general trust-region problem in 2 dimensions.

    The problem is reformulated as a 4th order algebraic equation,
    the solution of which is found by numpy.roots.

    Parameters
    ----------
    B : ndarray, shape (2, 2)
        Symmetric matrix, defines a quadratic term of the function.
    g : ndarray, shape (2,)
        Defines a linear term of the function.
    Delta : float
        Radius of a trust region.

    Returns
    -------
    p : ndarray, shape (2,)
        Found solution.
    newton_step : bool
        Whether the returned solution is the Newton step which lies within
        the trust region.
    """
    try:
        R, lower = cho_factor(B)  # 分解对称矩阵B，使用Cholesky分解
        p = -cho_solve((R, lower), g)  # 解线性方程 Bp = -g，得到步长p
        if np.dot(p, p) <= Delta**2:  # 判断步长p的范数是否在信赖域半径内
            return p, True  # 返回步长p和True表示是在信赖域内的牛顿步
    except LinAlgError:
        pass

    # 计算用于求解代数方程的系数
    a = B[0, 0] * Delta**2
    b = B[0, 1] * Delta**2
    c = B[1, 1] * Delta**2
    d = g[0] * Delta
    f = g[1] * Delta

    coeffs = np.array(
        [-b + d, 2 * (a - c + f), 6 * b, 2 * (-a + c + f), -b - d])
    t = np.roots(coeffs)  # 解代数方程 coeffs[0]*t**4 + ... + coeffs[4] = 0
    t = np.real(t[np.isreal(t)])  # 获取实数解

    # 构造可能的解向量p
    p = Delta * np.vstack((2 * t / (1 + t**2), (1 - t**2) / (1 + t**2)))
    value = 0.5 * np.sum(p * B.dot(p), axis=0) + np.dot(g, p)  # 计算每个解的函数值
    i = np.argmin(value)  # 找到最小函数值对应的索引
    p = p[:, i]  # 获取最小函数值对应的解向量p

    return p, False  # 返回解向量p和False表示不是在信赖域内的牛顿步


# 更新信赖域半径基于成本减少的函数。
def update_tr_radius(Delta, actual_reduction, predicted_reduction,
                     step_norm, bound_hit):
    """Update the radius of a trust region based on the cost reduction.

    Returns
    -------
    Delta : float
        New radius.
    ratio : float
        Ratio between actual and predicted reductions.
    """
    if predicted_reduction > 0:  # 如果预测的减少大于0
        ratio = actual_reduction / predicted_reduction  # 计算实际减少与预测减少的比值
    elif predicted_reduction == actual_reduction == 0:  # 如果预测和实际减少都为0
        ratio = 1  # 比值设为1
    else:
        ratio = 0  # 否则比值设为0

    if ratio < 0.25:  # 如果比值小于0.25
        Delta = 0.25 * step_norm  # 将信赖域半径设为步长范数的四分之一
    elif ratio > 0.75 and bound_hit:  # 如果比值大于0.75并且超出信赖域边界
        Delta *= 2.0  # 将信赖域半径扩大为原来的两倍

    return Delta, ratio  # 返回更新后的信赖域半径和比值


# 构建和最小化一维二次函数的函数。
def build_quadratic_1d(J, g, s, diag=None, s0=None):
    """Parameterize a multivariate quadratic function along a line.

    The resulting univariate quadratic function is given as follows::

        f(t) = 0.5 * (s0 + s*t).T * (J.T*J + diag) * (s0 + s*t) +
               g.T * (s0 + s*t)

    Parameters
    ----------
    J : ndarray, sparse matrix or LinearOperator shape (m, n)
        Jacobian matrix, affects the quadratic term.
    g : ndarray, shape (n,)
        Gradient, defines the linear term.
    s : ndarray, shape (n,)
        Direction vector of a line.
    diag : None or ndarray with shape (n,), optional
        Addition diagonal part, affects the quadratic term.
        If None, assumed to be 0.
    s0 : None or ndarray with shape (n,), optional
        Initial point. If None, assumed to be 0.

    Returns
    -------
    a : float
        Coefficient for t**2.
    """
    # 计算向量 v = J.dot(s)，其中 J 是 Jacobian 矩阵，s 是输入向量
    v = J.dot(s)
    # 计算 a = 0.5 * v^T * v，即向量 v 的二次型
    a = np.dot(v, v)
    # 如果提供了 diag 参数，计算 a += 0.5 * s^T * diag * s，其中 diag 是对角线矩阵
    if diag is not None:
        a += np.dot(s * diag, s)
    a *= 0.5  # 最终将 a 乘以 0.5

    # 计算 b = g^T * s，即向量 g 和向量 s 的内积
    b = np.dot(g, s)

    # 如果提供了 s0 参数
    if s0 is not None:
        # 计算 u = J.dot(s0)，其中 J 是 Jacobian 矩阵，s0 是额外的输入向量
        u = J.dot(s0)
        # 计算 b += u^T * v，即向量 u 和向量 v 的内积
        b += np.dot(u, v)
        # 计算 c = 0.5 * u^T * u + g^T * s0，即向量 u 的二次型和向量 g 和向量 s0 的内积
        c = 0.5 * np.dot(u, u) + np.dot(g, s0)
        # 如果提供了 diag 参数，进一步计算 b += s0^T * diag * s 和 c += 0.5 * s0^T * diag * s0
        if diag is not None:
            b += np.dot(s0 * diag, s)
            c += 0.5 * np.dot(s0 * diag, s0)
        return a, b, c  # 返回 a, b, c 作为结果
    else:
        return a, b  # 如果未提供 s0 参数，则仅返回 a 和 b
def minimize_quadratic_1d(a, b, lb, ub, c=0):
    """Minimize a 1-D quadratic function subject to bounds.

    The free term `c` is 0 by default. Bounds must be finite.

    Returns
    -------
    t : float
        Minimum point.
    y : float
        Minimum value.
    """
    # Initialize possible minimum points within bounds
    t = [lb, ub]
    # Check if the quadratic term is non-zero
    if a != 0:
        # Calculate the extremum point of the quadratic function
        extremum = -0.5 * b / a
        # Add the extremum point to the list of candidates if it lies within bounds
        if lb < extremum < ub:
            t.append(extremum)
    # Convert the list of points to a numpy array
    t = np.asarray(t)
    # Evaluate the quadratic function at all candidate points
    y = t * (a * t + b) + c
    # Find the index of the point that minimizes the function
    min_index = np.argmin(y)
    # Return the minimum point and its corresponding function value
    return t[min_index], y[min_index]


def evaluate_quadratic(J, g, s, diag=None):
    """Compute values of a quadratic function arising in least squares.

    The function is 0.5 * s.T * (J.T * J + diag) * s + g.T * s.

    Parameters
    ----------
    J : ndarray, sparse matrix or LinearOperator, shape (m, n)
        Jacobian matrix, affects the quadratic term.
    g : ndarray, shape (n,)
        Gradient, defines the linear term.
    s : ndarray, shape (k, n) or (n,)
        Array containing steps as rows.
    diag : ndarray, shape (n,), optional
        Addition diagonal part, affects the quadratic term.
        If None, assumed to be 0.

    Returns
    -------
    values : ndarray with shape (k,) or float
        Values of the function. If `s` was 2-D, then ndarray is
        returned, otherwise, float is returned.
    """
    # Check if `s` is 1-dimensional or 2-dimensional
    if s.ndim == 1:
        # Compute J*s and its dot product with itself
        Js = J.dot(s)
        q = np.dot(Js, Js)
        # Add diagonal term if provided
        if diag is not None:
            q += np.dot(s * diag, s)
    else:
        # Compute J*s.T and the sum of squared elements
        Js = J.dot(s.T)
        q = np.sum(Js**2, axis=0)
        # Add diagonal term if provided
        if diag is not None:
            q += np.sum(diag * s**2, axis=1)

    # Compute the linear term g*s
    l = np.dot(s, g)

    # Return the evaluated quadratic function value
    return 0.5 * q + l


# Utility functions to work with bound constraints.


def in_bounds(x, lb, ub):
    """Check if a point lies within bounds."""
    return np.all((x >= lb) & (x <= ub))


def step_size_to_bound(x, s, lb, ub):
    """Compute a min_step size required to reach a bound.

    The function computes a positive scalar t, such that x + s * t is on
    the bound.

    Returns
    -------
    step : float
        Computed step. Non-negative value.
    hits : ndarray of int with shape of x
        Each element indicates whether a corresponding variable reaches the
        bound:

             *  0 - the bound was not hit.
             * -1 - the lower bound was hit.
             *  1 - the upper bound was hit.
    """
    # Find non-zero elements in `s`
    non_zero = np.nonzero(s)
    s_non_zero = s[non_zero]
    # Initialize steps array with infinity
    steps = np.empty_like(x)
    steps.fill(np.inf)
    # Calculate step sizes to reach lower and upper bounds
    with np.errstate(over='ignore'):
        steps[non_zero] = np.maximum((lb - x)[non_zero] / s_non_zero,
                                     (ub - x)[non_zero] / s_non_zero)
    # Find the minimum step size required
    min_step = np.min(steps)
    # Determine which bounds are hit by the computed step sizes
    return min_step, np.equal(steps, min_step) * np.sign(s).astype(int)


def find_active_constraints(x, lb, ub, rtol=1e-10):
    """Determine which constraints are active in a given point.

    The threshold is computed using `rtol` and the absolute value of the
    closest bound.

    Returns
    -------
    ```
    # 创建一个与数组 x 相同形状的整数类型的零数组，用于表示约束是否激活的情况：
    #   *  0 - 约束未激活。
    #   * -1 - 约束的下界激活。
    #   *  1 - 约束的上界激活。
    active = np.zeros_like(x, dtype=int)
    
    # 如果相对容差 rtol 为零，则直接根据 lb 和 ub 设置约束的激活状态，并返回结果。
    if rtol == 0:
        active[x <= lb] = -1  # 将小于等于下界 lb 的部分设置为下界激活状态 -1
        active[x >= ub] = 1   # 将大于等于上界 ub 的部分设置为上界激活状态 1
        return active
    
    # 计算当前值 x 距离下界 lb 和上界 ub 的距离：
    lower_dist = x - lb
    upper_dist = ub - x
    
    # 根据相对容差 rtol 计算下界和上界的阈值：
    lower_threshold = rtol * np.maximum(1, np.abs(lb))
    upper_threshold = rtol * np.maximum(1, np.abs(ub))
    
    # 判断哪些约束的下界应该激活：
    lower_active = (np.isfinite(lb) &   # 下界 lb 是有限值
                    (lower_dist <= np.minimum(upper_dist, lower_threshold)))
    # 将符合条件的约束设置为下界激活状态 -1
    active[lower_active] = -1
    
    # 判断哪些约束的上界应该激活：
    upper_active = (np.isfinite(ub) &   # 上界 ub 是有限值
                    (upper_dist <= np.minimum(lower_dist, upper_threshold)))
    # 将符合条件的约束设置为上界激活状态 1
    active[upper_active] = 1
    
    # 返回最终的约束激活状态数组 active
    return active
def reflective_transformation(y, lb, ub):
    """Compute reflective transformation and its gradient."""

    # 检查是否输入向量 y 在界限 lb 和 ub 内
    if in_bounds(y, lb, ub):
        # 如果在界限内，则直接返回 y 和其梯度向量（全为1）
        return y, np.ones_like(y)

    # 检查界限 lb 和 ub 中哪些是有限的
    lb_finite = np.isfinite(lb)
    ub_finite = np.isfinite(ub)

    # 复制输入向量 y 到新变量 x
    x = y.copy()

    # 创建一个布尔向量，标记 lb 有限且 ub 无限的情况下 x 是否小于 lb
    g_negative = np.zeros_like(y, dtype=bool)

    # 处理 lb 有限且 ub 无限的情况
    mask = lb_finite & ~ub_finite
    x[mask] = np.maximum(y[mask], 2 * lb[mask] - y[mask])
    g_negative[mask] = y[mask] < lb[mask]

    # 处理 lb 无限且 ub 有限的情况
    mask = ~lb_finite & ub_finite
    x[mask] = np.minimum(y[mask], 2 * ub[mask] - y[mask])
    g_negative[mask] = y[mask] > ub[mask]

    # 处理 lb 和 ub 都有限的情况，计算界限差异向量 d
    mask = lb_finite & ub_finite
    d = ub - lb
    # 计算数组 y 中满足掩码 mask 的元素减去 lb[mask] 的余数，除以 2*d[mask]，并存储在数组 t 中
    t = np.remainder(y[mask] - lb[mask], 2 * d[mask])
    
    # 根据计算得到的 t 和给定条件，更新数组 x 中满足掩码 mask 的元素的值
    x[mask] = lb[mask] + np.minimum(t, 2 * d[mask] - t)
    
    # 根据比较 t 和 d[mask] 的结果，更新数组 g_negative 中的元素
    g_negative[mask] = t > d[mask]
    
    # 创建一个与数组 y 相同形状的数组 g，并将所有元素初始化为 1
    g = np.ones_like(y)
    
    # 根据 g_negative 数组的值，将 g 中满足条件的元素设为 -1
    g[g_negative] = -1
    
    # 返回更新后的数组 x 和 g
    return x, g
# Functions to display algorithm's progress.

# 打印非线性算法迭代的表头
def print_header_nonlinear():
    print("{:^15}{:^15}{:^15}{:^15}{:^15}{:^15}"
          .format("Iteration", "Total nfev", "Cost", "Cost reduction",
                  "Step norm", "Optimality"))

# 打印非线性算法每次迭代的信息
def print_iteration_nonlinear(iteration, nfev, cost, cost_reduction,
                              step_norm, optimality):
    if cost_reduction is None:
        cost_reduction = " " * 15
    else:
        cost_reduction = f"{cost_reduction:^15.2e}"

    if step_norm is None:
        step_norm = " " * 15
    else:
        step_norm = f"{step_norm:^15.2e}"

    print("{:^15}{:^15}{:^15.4e}{}{}{:^15.2e}"
          .format(iteration, nfev, cost, cost_reduction,
                  step_norm, optimality))

# 打印线性算法迭代的表头
def print_header_linear():
    print("{:^15}{:^15}{:^15}{:^15}{:^15}"
          .format("Iteration", "Cost", "Cost reduction", "Step norm",
                  "Optimality"))

# 打印线性算法每次迭代的信息
def print_iteration_linear(iteration, cost, cost_reduction, step_norm,
                           optimality):
    if cost_reduction is None:
        cost_reduction = " " * 15
    else:
        cost_reduction = f"{cost_reduction:^15.2e}"

    if step_norm is None:
        step_norm = " " * 15
    else:
        step_norm = f"{step_norm:^15.2e}"

    print(f"{iteration:^15}{cost:^15.4e}{cost_reduction}{step_norm}{optimality:^15.2e}")


# Simple helper functions.

# 计算最小二乘代价函数的梯度
def compute_grad(J, f):
    """Compute gradient of the least-squares cost function."""
    if isinstance(J, LinearOperator):
        return J.rmatvec(f)
    else:
        return J.T.dot(f)

# 根据雅可比矩阵计算变量的缩放比例
def compute_jac_scale(J, scale_inv_old=None):
    """Compute variables scale based on the Jacobian matrix."""
    if issparse(J):
        scale_inv = np.asarray(J.power(2).sum(axis=0)).ravel()**0.5
    else:
        scale_inv = np.sum(J**2, axis=0)**0.5

    if scale_inv_old is None:
        scale_inv[scale_inv == 0] = 1
    else:
        scale_inv = np.maximum(scale_inv, scale_inv_old)

    return 1 / scale_inv, scale_inv

# 返回左乘对角矩阵的线性操作器
def left_multiplied_operator(J, d):
    """Return diag(d) J as LinearOperator."""
    J = aslinearoperator(J)

    def matvec(x):
        return d * J.matvec(x)

    def matmat(X):
        return d[:, np.newaxis] * J.matmat(X)

    def rmatvec(x):
        return J.rmatvec(x.ravel() * d)

    return LinearOperator(J.shape, matvec=matvec, matmat=matmat,
                          rmatvec=rmatvec)

# 返回右乘对角矩阵的线性操作器
def right_multiplied_operator(J, d):
    """Return J diag(d) as LinearOperator."""
    J = aslinearoperator(J)

    def matvec(x):
        return J.matvec(np.ravel(x) * d)

    def matmat(X):
        return J.matmat(X * d[:, np.newaxis])

    def rmatvec(x):
        return d * J.rmatvec(x)

    return LinearOperator(J.shape, matvec=matvec, matmat=matmat,
                          rmatvec=rmatvec)

# 返回正则化最小二乘中出现的矩阵的线性操作器
def regularized_lsq_operator(J, diag):
    """Return a matrix arising in regularized least squares as LinearOperator.

    The matrix is
        [ J ]
        [ D ]
    """
    where D is diagonal matrix with elements from `diag`.
    """
    # 将 J 转换为线性操作符（如果尚未是的话）
    J = aslinearoperator(J)
    # 获取 J 的形状，m 是行数，n 是列数
    m, n = J.shape

    # 定义向量乘法函数 matvec(x)
    def matvec(x):
        # 对输入向量 x 进行操作，返回一个水平堆叠的数组
        # 包括 J 对 x 的乘法结果和 diag 和 x 的乘积
        return np.hstack((J.matvec(x), diag * x))

    # 定义反向向量乘法函数 rmatvec(x)
    def rmatvec(x):
        # 将输入向量 x 分解为两部分 x1 和 x2
        x1 = x[:m]   # x1 包含前 m 个元素
        x2 = x[m:]   # x2 包含从第 m 个元素开始的剩余元素
        # 返回 J 对 x1 的反向乘法结果加上 diag 与 x2 的乘积
        return J.rmatvec(x1) + diag * x2

    # 返回一个线性操作符对象，其形状为 (m + n, n)，包括定义好的 matvec 和 rmatvec 函数
    return LinearOperator((m + n, n), matvec=matvec, rmatvec=rmatvec)
# 计算 J diag(d) 的结果。
# 如果 `copy` 参数为 True 并且 J 不是 LinearOperator 类型，则复制 J。
def right_multiply(J, d, copy=True):
    if copy and not isinstance(J, LinearOperator):
        J = J.copy()

    # 如果 J 是稀疏矩阵，则按照指定的索引位置乘以对应的 d 值。
    # 这是 scikit-learn 的常用方法。
    elif issparse(J):
        J.data *= d.take(J.indices, mode='clip')

    # 如果 J 是 LinearOperator 类型，则调用特定的函数进行乘法操作。
    elif isinstance(J, LinearOperator):
        J = right_multiplied_operator(J, d)

    # 否则，直接将 J 乘以 d。
    else:
        J *= d

    return J


# 计算 diag(d) J 的结果。
# 如果 `copy` 参数为 True 并且 J 不是 LinearOperator 类型，则复制 J。
def left_multiply(J, d, copy=True):
    if copy and not isinstance(J, LinearOperator):
        J = J.copy()

    # 如果 J 是稀疏矩阵，则按照每行的非零元素乘以对应的 d 值。
    # 这是 scikit-learn 的常用方法。
    elif issparse(J):
        J.data *= np.repeat(d, np.diff(J.indptr))

    # 如果 J 是 LinearOperator 类型，则调用特定的函数进行乘法操作。
    elif isinstance(J, LinearOperator):
        J = left_multiplied_operator(J, d)

    # 否则，将 J 的每一列乘以 d 中对应的值。
    else:
        J *= d[:, np.newaxis]

    return J


# 检查非线性最小二乘法的终止条件。
def check_termination(dF, F, dx_norm, x_norm, ratio, ftol, xtol):
    ftol_satisfied = dF < ftol * F and ratio > 0.25
    xtol_satisfied = dx_norm < xtol * (xtol + x_norm)

    # 根据满足的终止条件返回相应的代码。
    if ftol_satisfied and xtol_satisfied:
        return 4
    elif ftol_satisfied:
        return 2
    elif xtol_satisfied:
        return 3
    else:
        return None


# 为鲁棒损失函数的 Jacobian 和残差进行缩放。
# 在原地修改数组。
def scale_for_robust_loss_function(J, f, rho):
    # 计算 Jacobian 的缩放因子。
    J_scale = rho[1] + 2 * rho[2] * f**2
    J_scale[J_scale < EPS] = EPS  # 如果缩放因子小于 EPS，则设置为 EPS。
    J_scale **= 0.5  # 对缩放因子取平方根。

    # 将残差 f 缩放为 rho[1] / J_scale 的倍数。
    f *= rho[1] / J_scale

    # 返回经过左乘 Jacobian 缩放后的结果和缩放后的残差。
    return left_multiply(J, J_scale, copy=False), f
```