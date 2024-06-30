# `D:\src\scipysrc\scipy\scipy\io\matlab\mio5_params.py`

```
# 此文件不适用于公共使用，并将在 SciPy v2.0.0 中删除。
# 使用 `scipy.io.matlab` 命名空间来导入下面列出的函数。

# 从 scipy._lib.deprecation 模块中导入 _sub_module_deprecation 函数
from scipy._lib.deprecation import _sub_module_deprecation

# 定义 __all__ 列表，包含允许导出的符号名称
__all__ = [
    'MatlabFunction', 'MatlabObject', 'MatlabOpaque', 'mat_struct',
]

# 定义 __dir__() 函数，返回 __all__ 列表，用于指定当前模块的可导入内容
def __dir__():
    return __all__


# 定义 __getattr__(name) 函数，处理动态属性访问，返回属性的 deprecation 警告
def __getattr__(name):
    return _sub_module_deprecation(
        sub_package="io.matlab",
        module="mio5_params",
        private_modules=["_mio5_params"],
        all=__all__,
        attribute=name
    )
```