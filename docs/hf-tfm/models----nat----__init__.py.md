# `.\transformers\models\nat\__init__.py`

```
# 导入类型检查模块
from typing import TYPE_CHECKING
# 导入可选依赖未可用异常和延迟模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义需要导入的模块结构
_import_structure = {"configuration_nat": ["NAT_PRETRAINED_CONFIG_ARCHIVE_MAP", "NatConfig"]}

# 尝试检查是否存在 torch 库，若不存在则引发可选依赖未可用异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
# 处理可选依赖未可用异常
except OptionalDependencyNotAvailable:
    pass
# 若存在 torch 库，则执行以下代码块
else:
    # 更新模块结构，包含额外的模块
    _import_structure["modeling_nat"] = [
        "NAT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "NatForImageClassification",
        "NatModel",
        "NatPreTrainedModel",
        "NatBackbone",
    ]

# 若是类型检查模式
if TYPE_CHECKING:
    # 从配置模块中导入相关变量和类
    from .configuration_nat import NAT_PRETRAINED_CONFIG_ARCHIVE_MAP, NatConfig

    # 尝试检查是否存在 torch 库，若不存在则引发可选依赖未可用异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    # 处理可选依赖未可用异常
    except OptionalDependencyNotAvailable:
        pass
    # 若存在 torch 库，则执行以下代码块
    else:
        # 从建模模块中导入相关变量和类
        from .modeling_nat import (
            NAT_PRETRAINED_MODEL_ARCHIVE_LIST,
            NatBackbone,
            NatForImageClassification,
            NatModel,
            NatPreTrainedModel,
        )

# 若不是类型检查模式
else:
    # 导入 sys 模块
    import sys

    # 将当前模块替换为 LazyModule 的实例
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```