# `D:\src\scipysrc\scipy\scipy\special\sf_error.py`

```
# 导入特定的警告和错误类，此文件不适合公共使用，并将在 SciPy v2.0.0 版本中删除。
from scipy._lib.deprecation import _sub_module_deprecation

# 定义公开的变量列表，用于模块导入时的显示
__all__ = [  # noqa: F822
    'SpecialFunctionWarning',
    'SpecialFunctionError'
]

# 定义特殊方法 __dir__()，返回模块的公开变量列表
def __dir__():
    return __all__

# 定义特殊方法 __getattr__(name)，用于动态获取模块中的属性，如果属性不存在，则调用 _sub_module_deprecation 函数
# 函数参数说明：
# - sub_package="special": 子模块名为 "special"
# - module="sf_error": 模块名为 "sf_error"
# - private_modules=["_sf_error"]: 私有模块列表包含 "_sf_error"
# - all=__all__: 所有公开属性的列表
# - attribute=name: 欲获取的属性名
def __getattr__(name):
    return _sub_module_deprecation(sub_package="special", module="sf_error",
                                   private_modules=["_sf_error"], all=__all__,
                                   attribute=name)
```