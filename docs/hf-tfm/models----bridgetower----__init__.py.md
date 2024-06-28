# `.\models\bridgetower\__init__.py`

```py
# 从 typing 模块中导入 TYPE_CHECKING 类型检查器
from typing import TYPE_CHECKING

# 从当前目录的 utils 模块中导入必要的异常和工具函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义一个字典结构 _import_structure，用于组织模块和对应的导入项
_import_structure = {
    "configuration_bridgetower": [
        "BRIDGETOWER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "BridgeTowerConfig",
        "BridgeTowerTextConfig",
        "BridgeTowerVisionConfig",
    ],
    "processing_bridgetower": ["BridgeTowerProcessor"],
}

# 尝试导入图像处理模块，如果 is_vision_available 返回 False，则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果成功导入，将图像处理模块添加到 _import_structure 中
    _import_structure["image_processing_bridgetower"] = ["BridgeTowerImageProcessor"]

# 尝试导入 Torch 模块，如果 is_torch_available 返回 False，则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果成功导入，将模型处理模块添加到 _import_structure 中
    _import_structure["modeling_bridgetower"] = [
        "BRIDGETOWER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "BridgeTowerForContrastiveLearning",
        "BridgeTowerForImageAndTextRetrieval",
        "BridgeTowerForMaskedLM",
        "BridgeTowerModel",
        "BridgeTowerPreTrainedModel",
    ]

# 如果是类型检查环境
if TYPE_CHECKING:
    # 从相关模块中导入配置和处理类
    from .configuration_bridgetower import (
        BRIDGETOWER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        BridgeTowerConfig,
        BridgeTowerTextConfig,
        BridgeTowerVisionConfig,
    )
    from .processing_bridgetower import BridgeTowerProcessor

    # 尝试导入图像处理模块，如果 is_vision_available 返回 False，则跳过导入
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_bridgetower import BridgeTowerImageProcessor

    # 尝试导入 Torch 模块，如果 is_torch_available 返回 False，则跳过导入
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

# 如果不是类型检查环境，则将当前模块设置为 LazyModule，用于延迟导入模块
else:
    import sys

    # 将当前模块替换为 LazyModule 对象，支持延迟加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```