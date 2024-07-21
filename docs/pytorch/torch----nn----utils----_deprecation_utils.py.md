# `.\pytorch\torch\nn\utils\_deprecation_utils.py`

```py
# mypy: allow-untyped-defs
# 引入必要的库和模块
import importlib
import warnings
from typing import Callable, List

# 定义警告消息模板，用于标记弃用的旧模块和推荐使用的新模块
_MESSAGE_TEMPLATE = (
    r"Usage of '{old_location}' is deprecated; please use '{new_location}' instead."
)

# 定义一个函数，用于实现延迟导入已弃用的模块或功能
def lazy_deprecated_import(
    all: List[str],
    old_module: str,
    new_module: str,
) -> Callable:
    r"""Import utility to lazily import deprecated packages / modules / functional.

    The old_module and new_module are also used in the deprecation warning defined
    by the `_MESSAGE_TEMPLATE`.

    Args:
        all: The list of the functions that are imported. Generally, the module's
            __all__ list of the module.
        old_module: Old module location
        new_module: New module location / Migrated location

    Returns:
        Callable to assign to the `__getattr__`

    Usage:

        # In the `torch/nn/quantized/functional.py`
        from torch.nn.utils._deprecation_utils import lazy_deprecated_import
        _MIGRATED_TO = "torch.ao.nn.quantized.functional"
        __getattr__ = lazy_deprecated_import(
            all=__all__,
            old_module=__name__,
            new_module=_MIGRATED_TO)
    """

    # 构建警告消息，使用给定的旧模块和新模块位置信息
    warning_message = _MESSAGE_TEMPLATE.format(
        old_location=old_module, new_location=new_module
    )

    # 定义一个名为getattr_dunder的内部函数，用于按需导入模块或功能
    def getattr_dunder(name):
        if name in all:
            # 发出运行时警告，以确保它不会被默认忽略
            warnings.warn(warning_message, RuntimeWarning)
            # 导入新模块
            package = importlib.import_module(new_module)
            # 返回新模块中对应名称的属性
            return getattr(package, name)
        # 如果请求的属性不在新模块中，引发属性错误
        raise AttributeError(f"Module {new_module!r} has no attribute {name!r}.")

    # 返回getattr_dunder函数对象，该对象可用于赋值给__getattr__
    return getattr_dunder
```