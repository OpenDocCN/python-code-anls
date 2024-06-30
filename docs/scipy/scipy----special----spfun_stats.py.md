# `D:\src\scipysrc\scipy\scipy\special\spfun_stats.py`

```
# 以下代码不适合公开使用，并且将在 SciPy v2.0.0 中移除。
# 使用 `scipy.special` 命名空间来导入下面包含的函数。

# 从 scipy._lib.deprecation 模块中导入 _sub_module_deprecation 函数
from scipy._lib.deprecation import _sub_module_deprecation

# 定义 __all__ 列表，包含 'multigammaln' 字符串，禁止 Flake8 F822 警告
__all__ = ['multigammaln']  # noqa: F822


# 定义 __dir__() 函数，返回 __all__ 列表
def __dir__():
    return __all__


# 定义 __getattr__(name) 函数，当调用未定义的属性时，触发 _sub_module_deprecation 函数
def __getattr__(name):
    return _sub_module_deprecation(sub_package="special", module="spfun_stats",
                                   private_modules=["_spfun_stats"], all=__all__,
                                   attribute=name)
```