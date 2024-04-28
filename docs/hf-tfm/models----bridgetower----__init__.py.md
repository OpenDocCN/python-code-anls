# `.\transformers\models\bridgetower\__init__.py`

```
# 导入类型检查模块
from typing import TYPE_CHECKING
# 导入可选依赖未安装异常和延迟加载模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义导入结构字典，包含不同模块的导入信息
_import_structure = {
    "configuration_bridgetower": [
        "BRIDGETOWER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "BridgeTowerConfig",
        "BridgeTowerTextConfig",
        "BridgeTowerVisionConfig",
    ],
    "processing_bridgetower": ["BridgeTowerProcessor"],
}

# 检查视觉处理模块是否可用，若不可用则抛出可选依赖未安装异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，将视觉处理模块加入导入结构字典
    _import_structure["image_processing_bridgetower"] = ["BridgeTowerImageProcessor"]

# 检查PyTorch是否可用，若不可用则抛出可选依赖未安装异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，将PyTorch模型处理模块加入导入结构字典
    _import_structure["modeling_bridgetower"] = [
        "BRIDGETOWER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "BridgeTowerForContrastiveLearning",
        "BridgeTowerForImageAndTextRetrieval",
        "BridgeTowerForMaskedLM",
        "BridgeTowerModel",
        "BridgeTowerPreTrainedModel",
    ]

# 若为类型检查模式，进行更多的导入
if TYPE_CHECKING:
    # 从配置模块中导入特定类和映射
    from .configuration_bridgetower import (
        BRIDGETOWER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        BridgeTowerConfig,
        BridgeTowerTextConfig,
        BridgeTowerVisionConfig,
    )
    # 从处理模块中导入特定类
    from .processing_bridgetower import BridgeTowerProcessor

    # 若视觉处理模块可用，从图片处理模块中导入特定类
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_bridgetower import BridgeTowerImageProcessor

    # 若PyTorch可用，从模型处理模块中导入特定类
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_bridgetower import (
            BRIDGETOWER_PRETRAINED_MODEL_ARCHIVE_LIST,
            BridgeTowerForContrastiveLearning,
            BridgeTowerForImageAndTextRetrieval,
            BridgeTowerForMaskedLM,
            BridgeTowerModel,
            BridgeTowerPreTrainedModel,
        )

# 若不是类型检查模式，将当前模块设置为延迟加载模式
else:
    import sys
    # 用延迟加载模块替换当前模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```