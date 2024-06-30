# `D:\src\scipysrc\scipy\scipy\linalg\misc.py`

```
# 本文件不适用于公共使用，并将在 SciPy v2.0.0 版本中移除。
# 请使用 `scipy.linalg` 命名空间来导入以下列出的函数。

# 从 scipy._lib.deprecation 模块中导入 _sub_module_deprecation 函数
from scipy._lib.deprecation import _sub_module_deprecation

# 定义 __all__ 列表，包含了将会被导出的符号名
__all__ = [
    'LinAlgError', 'LinAlgWarning', 'norm', 'get_blas_funcs',
    'get_lapack_funcs'
]


# 定义 __dir__() 函数，返回 __all__ 列表，用于控制 dir() 方法的输出
def __dir__():
    return __all__


# 定义 __getattr__(name) 函数，当访问的属性不存在时被调用
def __getattr__(name):
    # 调用 _sub_module_deprecation 函数，标记模块 `linalg` 的 `misc` 子模块中的私有模块 `_misc` 为过时
    return _sub_module_deprecation(
        sub_package="linalg",  # 子包名为 "linalg"
        module="misc",  # 模块名为 "misc"
        private_modules=["_misc"],  # 需要标记为过时的私有模块列表，这里只有 "_misc"
        all=__all__,  # 可导出的符号名列表，传递给 _sub_module_deprecation 函数
        attribute=name  # 访问的属性名
    )
```