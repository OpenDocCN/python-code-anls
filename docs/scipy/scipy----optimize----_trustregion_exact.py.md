# `D:\src\scipysrc\scipy\scipy\optimize\_trustregion_exact.py`

```
# 导入必要的库
import numpy as np
from scipy.linalg import (norm, get_lapack_funcs, solve_triangular,
                          cho_solve)
# 导入本地的信任区域最优化子模块
from ._trustregion import (_minimize_trust_region, BaseQuadraticSubproblem)

# 导出的符号列表
__all__ = ['_minimize_trustregion_exact',
           'estimate_smallest_singular_value',
           'singular_leading_submatrix',
           'IterativeSubproblem']

# 函数定义：几乎精确信任区域优化算法的最小化
def _minimize_trustregion_exact(fun, x0, args=(), jac=None, hess=None,
                                **trust_region_options):
    """
    使用几乎精确的信任区域算法最小化一个或多个变量的标量函数。

    选项
    -------
    initial_trust_radius : float
        初始信任区域半径。
    max_trust_radius : float
        信任区域半径的最大值。不会提议超过此值的步骤。
    eta : float
        对于提议步骤的信任区域相关接受严格性。
    gtol : float
        梯度范数必须小于 ``gtol`` 才能成功终止。
    """

    # 如果没有提供雅可比矩阵，则抛出错误
    if jac is None:
        raise ValueError('Jacobian is required for trust region '
                         'exact minimization.')
    # 如果提供的 Hessian 矩阵不可调用，则抛出错误
    if not callable(hess):
        raise ValueError('Hessian matrix is required for trust region '
                         'exact minimization.')
    
    # 调用信任区域最优化函数进行最小化
    return _minimize_trust_region(fun, x0, args=args, jac=jac, hess=hess,
                                  subproblem=IterativeSubproblem,
                                  **trust_region_options)


# 函数定义：估计最小奇异值
def estimate_smallest_singular_value(U):
    """给定上三角矩阵 ``U``，估计最小奇异值及其对应的右奇异向量，在 O(n**2) 操作内完成。

    参数
    ----------
    U : ndarray
        方形上三角矩阵。

    返回
    -------
    s_min : float
        提供矩阵的估计最小奇异值。
    z_min : ndarray
        估计的右奇异向量。

    注意
    -----
    此过程基于 [1]_，分两步进行。首先，找到一个从 {+1, -1} 中选择分量的向量 ``e``，
    使得系统 ``U.T w = e`` 的解 ``w`` 尽可能大。接下来估计 ``U v = w``。
    最小奇异值接近于 ``norm(w)/norm(v)``，右奇异向量接近于 ``v/norm(v)``。

    当矩阵条件数更差时，估计将更好。

    参考文献
    ----------
    .. [1] Cline, A. K., Moler, C. B., Stewart, G. W., Wilkinson, J. H.
           An estimate for the condition number of a matrix.  1979.
           SIAM Journal on Numerical Analysis, 16(2), 368-375.
    """

    # 将输入矩阵 U 至少转换为二维数组
    U = np.atleast_2d(U)
    m, n = U.shape

    # 如果矩阵不是方形，则抛出错误
    if m != n:
        raise ValueError("A square triangular matrix should be provided.")

    # 选择分量为 {+1, -1} 的向量 `e`
    # 初始化向量 p，长度为 n，所有元素初始化为 0
    p = np.zeros(n)
    # 初始化向量 w，长度为 n，元素未初始化
    w = np.empty(n)

    # 根据 Golub 和 Van Loan 的算法 3.5.1，页码 142，参考文献 [2]
    # 实现的目标是确保解向量 `w` 满足 `U.T w = e` 中的 `w` 尽可能大，
    # 此处的实现是针对下三角矩阵进行了调整。

    # 对于每一个 k，执行以下循环
    for k in range(n):
        # 计算 wp 和 wm
        wp = (1 - p[k]) / U.T[k, k]
        wm = (-1 - p[k]) / U.T[k, k]
        # 计算 pp 和 pm
        pp = p[k+1:] + U.T[k+1:, k] * wp
        pm = p[k+1:] + U.T[k+1:, k] * wm

        # 判断条件，选择较大的 wp 或 wm，并更新 p
        if abs(wp) + norm(pp, 1) >= abs(wm) + norm(pm, 1):
            w[k] = wp
            p[k+1:] = pp
        else:
            w[k] = wm
            p[k+1:] = pm

    # 解系统 `U v = w`，使用反向替换法进行求解
    v = solve_triangular(U, w)

    # 计算向量 v 和 w 的范数
    v_norm = norm(v)
    w_norm = norm(w)

    # 计算最小奇异值 s_min
    s_min = w_norm / v_norm

    # 计算关联向量 z_min
    z_min = v / v_norm

    # 返回最小奇异值和关联向量作为结果
    return s_min, z_min
def gershgorin_bounds(H):
    """
    给定一个方阵 ``H``，计算其特征值的上下界（格雷戈尔金界）。

    参考文献
    ----------
    .. [1] Conn, A. R., Gould, N. I., & Toint, P. L.
           Trust region methods. 2000. Siam. pp. 19.
    """

    # 提取矩阵 ``H`` 的对角线元素
    H_diag = np.diag(H)
    # 计算对角线元素的绝对值
    H_diag_abs = np.abs(H_diag)
    # 计算每行非对角线元素绝对值的和
    H_row_sums = np.sum(np.abs(H), axis=1)
    # 计算特征值的下界（最小值）
    lb = np.min(H_diag + H_diag_abs - H_row_sums)
    # 计算特征值的上界（最大值）
    ub = np.max(H_diag - H_diag_abs + H_row_sums)

    return lb, ub


def singular_leading_submatrix(A, U, k):
    """
    计算使得矩阵 ``A`` 前 ``k`` 行列成为奇异矩阵的项。

    参数
    ----------
    A : ndarray
        非正定对称矩阵。
    U : ndarray
        矩阵 ``A`` 的不完全Cholesky分解得到的上三角矩阵。
    k : int
        正整数，表示要使得前 ``k`` 行列为首个非正定的子矩阵。

    返回
    -------
    delta : float
        需要添加到矩阵 ``A`` 的 ``(k, k)`` 元素的量，使得其成为奇异矩阵。
    v : ndarray
        满足 ``v.T B v = 0`` 的向量 ``v``，其中 ``B`` 是加入 ``delta`` 后的矩阵 ``A``。

    """

    # 计算 delta
    delta = np.sum(U[:k-1, k-1]**2) - A[k-1, k-1]

    n = len(A)

    # 初始化 v
    v = np.zeros(n)
    v[k-1] = 1

    # 通过解三角系统计算 v 的其余部分值。
    if k != 1:
        v[:k-1] = solve_triangular(U[:k-1, :k-1], -U[:k-1, k-1])

    return delta, v


class IterativeSubproblem(BaseQuadraticSubproblem):
    """几乎精确迭代方法求解的二次子问题。

    注记
    -----
    此子问题求解器基于 [1]_, [2]_ 和 [3]_，它们实现了类似的算法。
    算法主要基于 [1]_，但也借鉴了 [2]_ 和 [3]_ 的思想。

    参考文献
    ----------
    .. [1] A.R. Conn, N.I. Gould, and P.L. Toint, "Trust region methods",
           Siam, pp. 169-200, 2000.
    .. [2] J. Nocedal and  S. Wright, "Numerical optimization",
           Springer Science & Business Media. pp. 83-91, 2006.
    .. [3] J.J. More and D.C. Sorensen, "Computing a trust region step",
           SIAM Journal on Scientific and Statistical Computing, vol. 4(3),
           pp. 553-572, 1983.
    """

    # UPDATE_COEFF 在参考文献 [1]_ 的公式 7.3.14 (p. 190) 中被命名为 "theta"。
    # 根据推荐，其值被固定为 0.01。
    UPDATE_COEFF = 0.01

    EPS = np.finfo(float).eps
    def __init__(self, x, fun, jac, hess, hessp=None,
                 k_easy=0.1, k_hard=0.2):
        # 调用父类构造函数初始化
        super().__init__(x, fun, jac, hess)

        # 当信赖域在两次连续计算中缩小（``tr_radius < previous_tr_radius``）时，
        # 可以重复使用下界 ``lambda_lb``，有助于收敛。初始时，``previous_tr_radius``
        # 被设为 -1，``lambda_lb`` 被设为 None 表示没有先前的值。
        self.previous_tr_radius = -1
        self.lambda_lb = None

        # 迭代次数计数器
        self.niter = 0

        # ``k_easy`` 和 ``k_hard`` 是用于确定迭代子问题求解的停止准则的参数。
        # 可以查看参考文献 _[1] 的 194-197 页获取更详细的描述。
        self.k_easy = k_easy
        self.k_hard = k_hard

        # 获取 Lapack 中进行 Cholesky 分解的函数。
        # 实现的 SciPy 封装不会返回方法所需的不完全因子化。
        self.cholesky, = get_lapack_funcs(('potrf',), (self.hess,))

        # 获取 Hessian 矩阵的信息
        self.dimension = len(self.hess)
        self.hess_gershgorin_lb,\
            self.hess_gershgorin_ub = gershgorin_bounds(self.hess)
        self.hess_inf = norm(self.hess, np.inf)  # Hessian 的无穷范数
        self.hess_fro = norm(self.hess, 'fro')   # Hessian 的 Frobenius 范数

        # 一个常数，用于指示小于此值的向量在反向替换中不可靠。
        # 它基于 Golub, G. H., Van Loan, C. F. (2013) 的书籍 "Matrix computations".
        # 第四版. JHU press., p.165. 这里的 EPS 是一个较小的常数。
        self.CLOSE_TO_ZERO = self.dimension * self.EPS * self.hess_inf
    def _initial_values(self, tr_radius):
        """Given a trust radius, return a good initial guess for
        the damping factor, the lower bound and the upper bound.
        The values were chosen accordingly to the guidelines on
        section 7.3.8 (p. 192) from [1]_.
        """
        
        # 计算上限阻尼因子
        lambda_ub = max(0, self.jac_mag/tr_radius + min(-self.hess_gershgorin_lb,
                                                        self.hess_fro,
                                                        self.hess_inf))
        
        # 计算下限阻尼因子
        lambda_lb = max(0, -min(self.hess.diagonal()),
                        self.jac_mag/tr_radius - min(self.hess_gershgorin_ub,
                                                     self.hess_fro,
                                                     self.hess_inf))
        
        # 根据之前的信息改进边界
        if tr_radius < self.previous_tr_radius:
            lambda_lb = max(self.lambda_lb, lambda_lb)
        
        # 初始猜测阻尼因子
        if lambda_lb == 0:
            lambda_initial = 0
        else:
            lambda_initial = max(np.sqrt(lambda_lb * lambda_ub),
                                 lambda_lb + self.UPDATE_COEFF*(lambda_ub-lambda_lb))
        
        return lambda_initial, lambda_lb, lambda_ub
```