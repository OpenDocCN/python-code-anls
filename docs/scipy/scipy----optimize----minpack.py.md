# `D:\src\scipysrc\scipy\scipy\optimize\minpack.py`

```
# 导入 _sub_module_deprecation 函数从 scipy._lib.deprecation 模块中
# 此文件不适用于公共使用，并将在 SciPy v2.0.0 中删除。
# 请使用 `scipy.optimize` 命名空间来导入以下包含的函数。

from scipy._lib.deprecation import _sub_module_deprecation

# 指定要导出的符号列表，防止 linter F822 警告
__all__ = [  # noqa: F822
    'OptimizeResult',   # 优化结果对象
    'OptimizeWarning',  # 优化警告对象
    'curve_fit',        # 曲线拟合函数
    'fixed_point',      # 固定点函数
    'fsolve',           # 非线性方程求解函数
    'least_squares',    # 最小二乘函数
    'leastsq',          # 最小二乘函数的旧称呼
    'zeros',            # 返回给定形状和类型的新数组，用零填充
]

# 定义 __dir__() 函数，返回 __all__ 列表，用于对象的 dir() 调用
def __dir__():
    return __all__

# 定义 __getattr__(name) 函数，用于动态获取属性
def __getattr__(name):
    # 使用 _sub_module_deprecation 函数进行子模块废弃处理
    # 参数说明：
    # - sub_package="optimize": 子模块名称为 optimize
    # - module="minpack": 主模块名称为 minpack
    # - private_modules=["_minpack_py"]: 私有模块名称列表，包括 _minpack_py
    # - all=__all__: 所有可用的符号列表
    # - attribute=name: 要获取的属性名称
    return _sub_module_deprecation(sub_package="optimize", module="minpack",
                                   private_modules=["_minpack_py"], all=__all__,
                                   attribute=name)
```