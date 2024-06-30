# `D:\src\scipysrc\scipy\scipy\linalg\decomp_schur.py`

```
# 导入_scipy._lib.deprecation模块中的_sub_module_deprecation函数，用于处理子模块的废弃警告
# 设置__all__变量，包含当前模块中公开的函数和异常类名列表
__all__ = [  # noqa: F822
    'schur', 'rsf2csf', 'norm', 'LinAlgError', 'get_lapack_funcs', 'eigvals',
]

# 定义__dir__()函数，返回模块中公开的函数和异常类名列表
def __dir__():
    return __all__

# 定义__getattr__(name)函数，处理对模块中不存在的属性的访问
def __getattr__(name):
    # 调用_sub_module_deprecation函数，向用户发出关于模块废弃情况的警告
    return _sub_module_deprecation(sub_package="linalg", module="decomp_schur",
                                   private_modules=["_decomp_schur"], all=__all__,
                                   attribute=name)
```