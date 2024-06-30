# `D:\src\scipysrc\scipy\scipy\optimize\_trustregion_ncg.py`

```
# 导入数学库
import math

# 导入必要的科学计算库
import numpy as np
import scipy.linalg

# 从 _trustregion 模块中导入所需的函数和类
from ._trustregion import (_minimize_trust_region, BaseQuadraticSubproblem)

# 空的 __all__ 列表，表明当前模块没有公开的对象
__all__ = []

# 定义一个函数，使用 Newton 共轭梯度信任域算法来最小化一个或多个变量的标量函数
def _minimize_trust_ncg(fun, x0, args=(), jac=None, hess=None, hessp=None,
                        **trust_region_options):
    """
    Minimization of scalar function of one or more variables using
    the Newton conjugate gradient trust-region algorithm.

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
    # 如果没有提供 Jacobi 矩阵，抛出 ValueError 异常
    if jac is None:
        raise ValueError('Jacobian is required for Newton-CG trust-region '
                         'minimization')
    # 如果没有提供 Hessian 或者 Hessian 向量乘积，抛出 ValueError 异常
    if hess is None and hessp is None:
        raise ValueError('Either the Hessian or the Hessian-vector product '
                         'is required for Newton-CG trust-region minimization')
    
    # 调用 _minimize_trust_region 函数进行最小化操作，使用 CGSteihaugSubproblem 类来解决子问题
    return _minimize_trust_region(fun, x0, args=args, jac=jac, hess=hess,
                                  hessp=hessp, subproblem=CGSteihaugSubproblem,
                                  **trust_region_options)

# 定义一个用共轭梯度方法求解的二次子问题类
class CGSteihaugSubproblem(BaseQuadraticSubproblem):
    """Quadratic subproblem solved by a conjugate gradient method"""
```