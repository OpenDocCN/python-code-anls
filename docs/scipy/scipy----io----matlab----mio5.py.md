# `D:\src\scipysrc\scipy\scipy\io\matlab\mio5.py`

```
# 导入模块中的子模块过时警告函数
# 该文件不适用于公共使用，并将在 SciPy v2.0.0 中移除。
# 使用 `scipy.io.matlab` 命名空间来导入下面列出的函数。

# 从 scipy._lib.deprecation 模块中导入 _sub_module_deprecation 函数
from scipy._lib.deprecation import _sub_module_deprecation

# 定义 __all__ 列表，包含了要导出的类和函数名
__all__ = [
    'MatWriteError', 'MatReadError', 'MatReadWarning', 'MatlabObject',
    'MatlabFunction', 'mat_struct', 'varmats_from_mat',
]

# 定义 __dir__() 函数，返回 __all__ 列表，用于模块导出时的列表显示
def __dir__():
    return __all__

# 定义 __getattr__(name) 函数，处理动态获取属性的行为
def __getattr__(name):
    # 调用 _sub_module_deprecation 函数，生成关于子模块已过时的警告信息
    # sub_package 参数指定为 "io.matlab"，module 参数指定为 "mio5"
    # private_modules 参数指定为 ["_mio5"]，all 参数为 __all__ 列表
    # attribute 参数为当前属性名 name
    return _sub_module_deprecation(sub_package="io.matlab", module="mio5",
                                   private_modules=["_mio5"], all=__all__,
                                   attribute=name)
```