# `.\numpy\numpy\core\_utils.py`

```py
# 导入警告模块，用于生成警告信息
import warnings

# 定义一个函数，用于发出警告信息
def _raise_warning(attr: str, submodule: str = None) -> None:
    # 定义新旧模块的名称
    new_module = "numpy._core"
    old_module = "numpy.core"

    # 如果有子模块，添加到模块名称中
    if submodule is not None:
        new_module = f"{new_module}.{submodule}"
        old_module = f"{old_module}.{submodule}"

    # 发出警告，说明旧模块已弃用并重命名
    warnings.warn(
        f"{old_module} is deprecated and has been renamed to {new_module}. "
        "The numpy._core namespace contains private NumPy internals and its "
        "use is discouraged, as NumPy internals can change without warning in "
        "any release. In practice, most real-world usage of numpy.core is to "
        "access functionality in the public NumPy API. If that is the case, "
        "use the public NumPy API. If not, you are using NumPy internals. "
        "If you would still like to access an internal attribute, "
        f"use {new_module}.{attr}.",
        DeprecationWarning, 
        stacklevel=3  # 设置警告的堆栈级别
    )
```