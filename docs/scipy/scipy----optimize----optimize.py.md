# `D:\src\scipysrc\scipy\scipy\optimize\optimize.py`

```
# 导入 _sub_module_deprecation 函数，用于处理子模块废弃警告
# 导入以下函数和类至当前命名空间，用于未来版本 SciPy v2.0.0 的优化操作
from scipy._lib.deprecation import _sub_module_deprecation

# 定义模块中公开的函数和类名列表，用于控制命名空间的导出
__all__ = [  # noqa: F822
    'OptimizeResult',        # 优化结果类
    'OptimizeWarning',       # 优化警告类
    'approx_fprime',         # 数值梯度估计函数
    'bracket',               # 寻找最小值区间函数
    'brent',                 # Brent 方法优化函数
    'brute',                 # 蛮力搜索优化函数
    'check_grad',            # 梯度检查函数
    'fmin',                  # 一维函数最小化函数
    'fmin_bfgs',             # BFGS 方法最小化函数
    'fmin_cg',               # 共轭梯度方法最小化函数
    'fmin_ncg',              # 牛顿共轭梯度方法最小化函数
    'fmin_powell',           # Powell 方法最小化函数
    'fminbound',             # 有约束的单变量最小化函数
    'golden',                # 黄金分割法最小化函数
    'line_search',           # 线搜索函数
    'rosen',                 # Rosenbrock 函数
    'rosen_der',             # Rosenbrock 函数的梯度
    'rosen_hess',            # Rosenbrock 函数的 Hessian 矩阵
    'rosen_hess_prod',       # Rosenbrock 函数 Hessian 矩阵乘向量的结果
    'show_options',          # 显示优化选项函数
    'zeros',                 # 寻找函数零点函数
]


def __dir__():
    # 返回当前模块公开的所有函数和类名列表
    return __all__


def __getattr__(name):
    # 当访问未定义的属性时，通过 _sub_module_deprecation 函数提供优化模块的废弃警告
    return _sub_module_deprecation(sub_package="optimize", module="optimize",
                                   private_modules=["_optimize"], all=__all__,
                                   attribute=name)
```