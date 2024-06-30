# `D:\src\scipysrc\scipy\scipy\optimize\_trustregion_krylov.py`

```
# 导入模块中的函数和对象，用于最小化信任域方法
from ._trustregion import (_minimize_trust_region)
# 导入 trlib 模块中的函数，用于获取三角形子问题
from ._trlib import (get_trlib_quadratic_subproblem)

# 模块中公开的函数和对象列表
__all__ = ['_minimize_trust_krylov']

# 定义一个使用几乎精确的信任域算法最小化标量函数的函数
def _minimize_trust_krylov(fun, x0, args=(), jac=None, hess=None, hessp=None,
                           inexact=True, **trust_region_options):
    """
    Minimization of a scalar function of one or more variables using
    a nearly exact trust-region algorithm that only requires matrix
    vector products with the hessian matrix.

    .. versionadded:: 1.0.0

    Options
    -------
    inexact : bool, optional
        Accuracy to solve subproblems. If True requires less nonlinear
        iterations, but more vector products.
    """

    # 如果未提供 Jacobi 矩阵，则抛出 ValueError 异常
    if jac is None:
        raise ValueError('Jacobian is required for trust region ',
                         'exact minimization.')
    # 如果既未提供 Hessian 矩阵也未提供 Hessian 向量积，则抛出 ValueError 异常
    if hess is None and hessp is None:
        raise ValueError('Either the Hessian or the Hessian-vector product '
                         'is required for Krylov trust-region minimization')

    # tol_rel 指定在 Krylov 子空间迭代中相对于初始梯度范数的终止容差。

    # - tol_rel_i 指定内部收敛的容差。
    # - tol_rel_b 指定边界收敛的容差。
    #   在非线性规划应用中，不需要像内部情况那样精确地解决边界情况。

    # - 设置 tol_rel_i=-2 导致在 Krylov 子空间迭代中出现强迫序列，如果最终
    #   信任域保持不活跃，则导致二次收敛。
    # - 设置 tol_rel_b=-3 导致在 Krylov 子空间迭代中出现强迫序列，只要迭代达到
    #   信任域边界，就导致超线性收敛。

    # 有关详细信息，请参阅 _trlib/trlib_krylov.h 中 trlib_krylov_min 的文档。
    #
    # 在 CUTEst 库的无约束子集上测试了这些参数选择的最优性。

    # 如果选择使用不精确求解子问题，则调用 _minimize_trust_region 函数，
    # 并传递所需的参数和选项
    if inexact:
        return _minimize_trust_region(fun, x0, args=args, jac=jac,
                                      hess=hess, hessp=hessp,
                                      subproblem=get_trlib_quadratic_subproblem(
                                          tol_rel_i=-2.0, tol_rel_b=-3.0,
                                          disp=trust_region_options.get('disp', False)
                                          ),
                                      **trust_region_options)
    else:
        # 如果不是简化的信任域方法，则调用 _minimize_trust_region 函数
        return _minimize_trust_region(fun, x0, args=args, jac=jac,
                                      hess=hess, hessp=hessp,
                                      # 使用 get_trlib_quadratic_subproblem 函数获取二次子问题求解器
                                      subproblem=get_trlib_quadratic_subproblem(
                                          tol_rel_i=1e-8, tol_rel_b=1e-6,
                                          # 设置显示选项为 trust_region_options 中的 disp 值，缺省为 False
                                          disp=trust_region_options.get('disp', False)
                                          ),
                                      **trust_region_options)
```