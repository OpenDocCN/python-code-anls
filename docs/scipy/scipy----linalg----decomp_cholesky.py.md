# `D:\src\scipysrc\scipy\scipy\linalg\decomp_cholesky.py`

```
# 引入需要的函数和类，从 SciPy 库中的 _lib.deprecation 子模块中导入 _sub_module_deprecation 函数
from scipy._lib.deprecation import _sub_module_deprecation

# 指定当前模块中可以被外部调用的公共接口列表
__all__ = [  # noqa: F822
    'cholesky', 'cho_factor', 'cho_solve', 'cholesky_banded',
    'cho_solve_banded', 'LinAlgError', 'get_lapack_funcs'
]

# 定义一个特殊方法 __dir__()，返回模块中的所有公共接口列表
def __dir__():
    return __all__

# 定义一个特殊方法 __getattr__()，用于在当前模块中动态获取指定名称的属性
def __getattr__(name):
    # 使用 _sub_module_deprecation 函数处理对模块中特定属性的访问，标记为过时，并提供相关替代信息
    return _sub_module_deprecation(sub_package="linalg", module="decomp_cholesky",
                                   private_modules=["_decomp_cholesky"], all=__all__,
                                   attribute=name)
```