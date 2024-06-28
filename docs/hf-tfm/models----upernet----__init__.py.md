# `.\models\upernet\__init__.py`

```
# 导入类型检查模块
from typing import TYPE_CHECKING

# 导入自定义的异常和模块延迟加载工具
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块的导入结构字典
_import_structure = {
    "configuration_upernet": ["UperNetConfig"],  # 导入 UperNetConfig 配置
}

# 检查是否可用 Torch 库，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 可用，则扩展导入结构字典，导入 modeling_upernet 模块中的特定类
    _import_structure["modeling_upernet"] = [
        "UperNetForSemanticSegmentation",
        "UperNetPreTrainedModel",
    ]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 导入 UperNetConfig 配置类
    from .configuration_upernet import UperNetConfig

    # 再次检查 Torch 是否可用，若不可用则捕获异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 Torch 可用，则导入 modeling_upernet 模块中的特定类
        from .modeling_upernet import UperNetForSemanticSegmentation, UperNetPreTrainedModel

# 如果不是类型检查模式
else:
    import sys

    # 动态地将当前模块替换为一个 LazyModule 实例，实现模块的延迟加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```