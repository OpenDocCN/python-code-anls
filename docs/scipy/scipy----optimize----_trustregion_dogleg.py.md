# `D:\src\scipysrc\scipy\scipy\optimize\_trustregion_dogleg.py`

```
"""Dog-leg trust-region optimization."""
# 导入必要的库
import numpy as np
import scipy.linalg
# 导入内部模块，这里假设位于当前目录下的 _trustregion.py 文件中
from ._trustregion import (_minimize_trust_region, BaseQuadraticSubproblem)

__all__ = []

# 定义私有函数 _minimize_dogleg，使用 dog-leg 信赖域算法最小化标量函数
def _minimize_dogleg(fun, x0, args=(), jac=None, hess=None,
                     **trust_region_options):
    """
    Minimization of scalar function of one or more variables using
    the dog-leg trust-region algorithm.

    Options
    -------
    initial_trust_radius : float
        Initial trust-region radius.
    max_trust_radius : float
        Maximum value of the trust-region radius. No steps that are longer
        than this value will be proposed.
    eta : float
        Trust region related acceptance stringency for proposed steps.
    gtol : float
        Gradient norm must be less than `gtol` before successful
        termination.

    """
    # 如果未提供 Jacobi 矩阵，抛出值错误
    if jac is None:
        raise ValueError('Jacobian is required for dogleg minimization')
    # 如果未提供可调用的 Hessian 矩阵，抛出值错误
    if not callable(hess):
        raise ValueError('Hessian is required for dogleg minimization')
    # 调用 _minimize_trust_region 函数进行最小化操作，使用 DoglegSubproblem 子类
    return _minimize_trust_region(fun, x0, args=args, jac=jac, hess=hess,
                                  subproblem=DoglegSubproblem,
                                  **trust_region_options)


# 定义 DoglegSubproblem 类，继承自 BaseQuadraticSubproblem 类
class DoglegSubproblem(BaseQuadraticSubproblem):
    """Quadratic subproblem solved by the dogleg method"""

    # 实现 Cauchy point 方法，返回沿最陡下降方向的 Cauchy 点
    def cauchy_point(self):
        """
        The Cauchy point is minimal along the direction of steepest descent.
        """
        # 如果尚未计算 Cauchy 点
        if self._cauchy_point is None:
            g = self.jac
            Bg = self.hessp(g)
            # 计算 Cauchy 点
            self._cauchy_point = -(np.dot(g, g) / np.dot(g, Bg)) * g
        return self._cauchy_point

    # 实现 Newton point 方法，返回近似函数的全局最小值点
    def newton_point(self):
        """
        The Newton point is a global minimum of the approximate function.
        """
        # 如果尚未计算 Newton 点
        if self._newton_point is None:
            g = self.jac
            B = self.hess
            # 使用 Cholesky 分解求解 Newton 点
            cho_info = scipy.linalg.cho_factor(B)
            self._newton_point = -scipy.linalg.cho_solve(cho_info, g)
        return self._newton_point
    # 定义 solve 方法，用于通过狗腿信赖区域算法最小化函数
    def solve(self, trust_radius):
        """
        Minimize a function using the dog-leg trust-region algorithm.

        This algorithm requires function values and first and second derivatives.
        It also performs a costly Hessian decomposition for most iterations,
        and the Hessian is required to be positive definite.

        Parameters
        ----------
        trust_radius : float
            We are allowed to wander only this far away from the origin.

        Returns
        -------
        p : ndarray
            The proposed step.
        hits_boundary : bool
            True if the proposed step is on the boundary of the trust region.

        Notes
        -----
        The Hessian is required to be positive definite.

        References
        ----------
        .. [1] Jorge Nocedal and Stephen Wright,
               Numerical Optimization, second edition,
               Springer-Verlag, 2006, page 73.
        """

        # 计算牛顿点（Newton point）。
        # 这是二次模型函数的最优点。
        # 如果它在信赖区域内，则返回该点。
        p_best = self.newton_point()
        if scipy.linalg.norm(p_best) < trust_radius:
            hits_boundary = False
            return p_best, hits_boundary

        # 计算柯西点（Cauchy point）。
        # 这是沿最陡下降方向的预测最优点。
        p_u = self.cauchy_point()

        # 如果柯西点在信赖区域外，
        # 则返回路径与边界相交的点。
        p_u_norm = scipy.linalg.norm(p_u)
        if p_u_norm >= trust_radius:
            p_boundary = p_u * (trust_radius / p_u_norm)
            hits_boundary = True
            return p_boundary, hits_boundary

        # 计算信赖区域边界与连接柯西点和牛顿点的线段的交点。
        # 这需要解一个二次方程。
        # ||p_u + t*(p_best - p_u)||**2 == trust_radius**2
        # 使用二次公式找到正时间 t 的解。
        _, tb = self.get_boundaries_intersections(p_u, p_best - p_u,
                                                  trust_radius)
        p_boundary = p_u + tb * (p_best - p_u)
        hits_boundary = True
        return p_boundary, hits_boundary
```