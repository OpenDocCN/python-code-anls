# `D:\src\scipysrc\scipy\scipy\optimize\tnc.py`

```
# 此文件不适用于公共使用，并将在 SciPy v2.0.0 中删除。
# 使用 `scipy.optimize` 命名空间来导入下面列出的函数。

# 从 scipy._lib.deprecation 模块中导入 _sub_module_deprecation 函数
from scipy._lib.deprecation import _sub_module_deprecation

# 定义一个列表，包含了本模块中需要导出的公共符号名
__all__ = [
    'OptimizeResult',  # 导出优化结果类 OptimizeResult
    'fmin_tnc',         # 导出优化函数 fmin_tnc
    'zeros',            # 导出 zeros 函数
]


# 定义一个特殊方法 __dir__()，返回模块中所有需要导出的符号名列表
def __dir__():
    return __all__


# 定义一个特殊方法 __getattr__(name)，用于处理动态获取模块属性的请求
def __getattr__(name):
    # 调用 _sub_module_deprecation 函数，向用户发出模块已废弃警告，建议使用 scipy.optimize
    return _sub_module_deprecation(sub_package="optimize", module="tnc",
                                   private_modules=["_tnc"], all=__all__,
                                   attribute=name)
```