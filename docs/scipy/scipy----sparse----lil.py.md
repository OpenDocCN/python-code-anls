# `D:\src\scipysrc\scipy\scipy\sparse\lil.py`

```
# 导入 `_sub_module_deprecation` 函数，用于处理子模块过时警告
from scipy._lib.deprecation import _sub_module_deprecation

# 定义 __all__ 列表，指定了当前模块中公开的接口名称，用于 `__dir__()` 和 `__getattr__()` 函数
__all__ = [
    'isspmatrix_lil',   # 指定 `isspmatrix_lil` 为公开接口
    'lil_array',        # 指定 `lil_array` 为公开接口
    'lil_matrix',       # 指定 `lil_matrix` 为公开接口
]

# 定义 `__dir__()` 函数，返回当前模块的公开接口列表
def __dir__():
    return __all__

# 定义 `__getattr__(name)` 函数，当获取未定义的属性时调用，返回过时警告处理函数 `_sub_module_deprecation` 的结果
def __getattr__(name):
    return _sub_module_deprecation(
        sub_package="sparse",   # 子包名称为 "sparse"
        module="lil",           # 模块名称为 "lil"
        private_modules=["_lil"],   # 私有模块列表包括 "_lil"
        all=__all__,            # 所有公开接口名称列表
        attribute=name          # 当前属性名称
    )
```