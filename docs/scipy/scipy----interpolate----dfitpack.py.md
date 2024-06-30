# `D:\src\scipysrc\scipy\scipy\interpolate\dfitpack.py`

```
# 引入 `_sub_module_deprecation` 函数，用于处理子模块已废弃警告
# 这个文件不是用于公共使用的，在 SciPy v2.0.0 中将被移除
# 使用 `scipy.interpolate` 命名空间来导入下面列出的函数

# 导入 `_sub_module_deprecation` 函数，从 `scipy._lib.deprecation` 子模块中
from scipy._lib.deprecation import _sub_module_deprecation

# 定义模块中可导出的全部符号列表，排除 F822 类型的 noqa 格式错误
__all__ = [
    'bispeu',            # 双三次插值的欧式方法
    'bispev',            # 双三次插值的扩展方法
    'curfit',            # 曲线拟合
    'dblint',            # 双线性插值
    'fpchec',            # 浮点数检查
    'fpcurf0',           # 浮点数曲线拟合
    'fpcurf1',           # 浮点数曲线拟合（一次）
    'fpcurfm1',          # 浮点数曲线拟合（负一次）
    'parcur',            # 曲面拟合
    'parder',            # 导数计算
    'pardeu',            # 欧式导数计算
    'pardtc',            # 时间相关导数计算
    'percur',            # 周期曲线拟合
    'regrid_smth',       # 重新网格化平滑处理
    'regrid_smth_spher', # 重新网格化平滑处理（球面）
    'spalde',            # 样条插值求导数
    'spherfit_lsq',      # 最小二乘球面拟合
    'spherfit_smth',     # 平滑球面拟合
    'splder',            # 样条导数计算
    'splev',             # 样条插值计算
    'splint',            # 样条积分
    'sproot',            # 样条根
    'surfit_lsq',        # 最小二乘曲面拟合
    'surfit_smth',       # 平滑曲面拟合
    'types',             # 类型定义
]

# 定义 `__dir__` 函数，返回模块的全部符号列表
def __dir__():
    return __all__

# 定义 `__getattr__` 函数，当获取未定义的属性时调用 `_sub_module_deprecation` 处理函数
def __getattr__(name):
    return _sub_module_deprecation(sub_package="interpolate", module="dfitpack",
                                   private_modules=["_dfitpack"], all=__all__,
                                   attribute=name)
```