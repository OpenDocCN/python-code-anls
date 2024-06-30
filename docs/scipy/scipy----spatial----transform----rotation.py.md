# `D:\src\scipysrc\scipy\scipy\spatial\transform\rotation.py`

```
# 此文件不适用于公共使用，并将在 SciPy v2.0.0 中移除。
# 请使用 `scipy.spatial` 命名空间来导入下面列出的函数。

# 从 scipy._lib.deprecation 模块中导入 _sub_module_deprecation 函数
from scipy._lib.deprecation import _sub_module_deprecation

# 定义 __all__ 变量，指定公开的模块成员列表
__all__ = [  # noqa: F822
    'Rotation',  # 将 Rotation 添加到 __all__ 列表中
    'Slerp',      # 将 Slerp 添加到 __all__ 列表中
]


# 定义 __dir__() 函数，返回模块中公开的所有成员列表
def __dir__():
    return __all__


# 定义 __getattr__(name) 函数，处理动态属性访问的行为
def __getattr__(name):
    return _sub_module_deprecation(
        sub_package="spatial.transform",  # 子包名称为 "spatial.transform"
        module="rotation",                # 模块名称为 "rotation"
        private_modules=["_rotation"],    # 私有模块列表为 ["_rotation"]
        all=__all__,                      # 全部公开成员列表为 __all__
        attribute=name                    # 请求的属性名称
    )
```