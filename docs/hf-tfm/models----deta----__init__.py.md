# `.\models\deta\__init__.py`

```
# 引入类型检查标记，如果支持类型检查，表示当前环境可能用于类型检查
from typing import TYPE_CHECKING

# 引入自定义的异常和模块加载延迟工具
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义模块的导入结构字典，用于延迟加载模块
_import_structure = {
    "configuration_deta": ["DETA_PRETRAINED_CONFIG_ARCHIVE_MAP", "DetaConfig"],
}

# 检查是否支持视觉处理模块，若不支持则抛出可选依赖不可用异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果支持视觉处理，添加视觉处理模块到导入结构字典中
    _import_structure["image_processing_deta"] = ["DetaImageProcessor"]

# 检查是否支持torch模块，若不支持则抛出可选依赖不可用异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果支持torch模块，添加模型相关的deta模块到导入结构字典中
    _import_structure["modeling_deta"] = [
        "DETA_PRETRAINED_MODEL_ARCHIVE_LIST",
        "DetaForObjectDetection",
        "DetaModel",
        "DetaPreTrainedModel",
    ]

# 如果是类型检查环境
if TYPE_CHECKING:
    # 从配置模块中导入预训练配置映射和DetaConfig类
    from .configuration_deta import DETA_PRETRAINED_CONFIG_ARCHIVE_MAP, DetaConfig

    # 检查是否支持视觉处理模块，若不支持则抛出可选依赖不可用异常
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 在类型检查环境下，从视觉处理模块导入DetaImageProcessor类
        from .image_processing_deta import DetaImageProcessor

    # 检查是否支持torch模块，若不支持则抛出可选依赖不可用异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 在类型检查环境下，从模型处理模块导入相关类和列表
        from .modeling_deta import (
            DETA_PRETRAINED_MODEL_ARCHIVE_LIST,
            DetaForObjectDetection,
            DetaModel,
            DetaPreTrainedModel,
        )

# 如果不是类型检查环境，即运行时环境
else:
    import sys

    # 将当前模块替换为延迟加载模块，使得导入时真正加载模块内容
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```