# `D:\src\scipysrc\scipy\scipy\optimize\_trlib\__init__.py`

```
# 导入TRLIBQuadraticSubproblem类，这个类来自._trlib模块
from ._trlib import TRLIBQuadraticSubproblem

# 定义公开的模块成员列表，包括TRLIBQuadraticSubproblem类和get_trlib_quadratic_subproblem函数
__all__ = ['TRLIBQuadraticSubproblem', 'get_trlib_quadratic_subproblem']

# 定义函数get_trlib_quadratic_subproblem，可以接受三个参数：tol_rel_i、tol_rel_b和disp，默认值分别为-2.0、-3.0和False
def get_trlib_quadratic_subproblem(tol_rel_i=-2.0, tol_rel_b=-3.0, disp=False):
    # 定义内部函数subproblem_factory，接受五个参数：x、fun、jac、hess和hessp
    def subproblem_factory(x, fun, jac, hess, hessp):
        # 使用TRLIBQuadraticSubproblem类创建一个子问题实例，传入参数x、fun、jac、hess、hessp以及tol_rel_i、tol_rel_b和disp
        return TRLIBQuadraticSubproblem(x, fun, jac, hess, hessp,
                                        tol_rel_i=tol_rel_i,
                                        tol_rel_b=tol_rel_b,
                                        disp=disp)
    
    # 返回内部函数subproblem_factory，这样外部调用get_trlib_quadratic_subproblem函数时，实际上返回的是subproblem_factory函数
    return subproblem_factory
```