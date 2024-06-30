# `D:\src\scipysrc\sympy\sympy\solvers\__init__.py`

```
# 导入 SymPy 求解器模块，用于解各种方程和问题
from sympy.core.assumptions import check_assumptions, failing_assumptions

# 导入 SymPy 求解器函数：solve 等
from .solvers import solve, solve_linear_system, solve_linear_system_LU, \
    solve_undetermined_coeffs, nsolve, solve_linear, checksol, \
    det_quick, inv_quick

# 导入 SymPy 算法：解丢番图方程
from .diophantine import diophantine

# 导入 SymPy 递推求解器函数：rsolve 等
from .recurr import rsolve, rsolve_poly, rsolve_ratio, rsolve_hyper

# 导入 SymPy 常微分方程求解函数：dsolve 等
from .ode import checkodesol, classify_ode, dsolve, \
    homogeneous_order

# 导入 SymPy 多项式系统求解函数：solve_poly_system 等
from .polysys import solve_poly_system, solve_triangulated

# 导入 SymPy 偏微分方程求解函数：pdsolve 等
from .pde import pde_separate, pde_separate_add, pde_separate_mul, \
    pdsolve, classify_pde, checkpdesol

# 导入 SymPy 工具函数：ode_order
from .deutils import ode_order

# 导入 SymPy 不等式求解函数：reduce_inequalities 等
from .inequalities import reduce_inequalities, reduce_abs_inequality, \
    reduce_abs_inequalities, solve_poly_inequality, solve_rational_inequalities, solve_univariate_inequality

# 导入 SymPy 分解函数：decompogen
from .decompogen import decompogen

# 导入 SymPy 解集函数：solveset 等
from .solveset import solveset, linsolve, linear_eq_to_matrix, nonlinsolve, substitution

# 导入 SymPy 线性规划函数：lpmin, lpmax, linprog
from .simplex import lpmin, lpmax, linprog

# 定义 Complexes 为 SymPy 单例集合 S 中的 Complexes 集合
# 避免在 sympy/sets/__init__.py 中引起循环导入问题
from ..core.singleton import S
Complexes = S.Complexes

# 导出的所有函数和变量列表，用于模块级别的导入
__all__ = [
    'solve', 'solve_linear_system', 'solve_linear_system_LU',
    'solve_undetermined_coeffs', 'nsolve', 'solve_linear', 'checksol',
    'det_quick', 'inv_quick', 'check_assumptions', 'failing_assumptions',

    'diophantine',

    'rsolve', 'rsolve_poly', 'rsolve_ratio', 'rsolve_hyper',

    'checkodesol', 'classify_ode', 'dsolve', 'homogeneous_order',

    'solve_poly_system', 'solve_triangulated',

    'pde_separate', 'pde_separate_add', 'pde_separate_mul', 'pdsolve',
    'classify_pde', 'checkpdesol',

    'ode_order',

    'reduce_inequalities', 'reduce_abs_inequality', 'reduce_abs_inequalities',
    'solve_poly_inequality', 'solve_rational_inequalities',
    'solve_univariate_inequality',

    'decompogen',

    'solveset', 'linsolve', 'linear_eq_to_matrix', 'nonlinsolve',
    'substitution',

    'Complexes',

    'lpmin', 'lpmax', 'linprog'
]
```