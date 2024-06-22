# `.\models\flava\__init__.py`

```py
# 引入类型检查模块
from typing import TYPE_CHECKING

# 引入必要的依赖
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义待导入的结构
_import_structure = {
    "configuration_flava": [
        "FLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "FlavaConfig",
        "FlavaImageCodebookConfig",
        "FlavaImageConfig",
        "FlavaMultimodalConfig",
        "FlavaTextConfig",
    ],
}

# 检查视觉库是否可用，若不可用则抛出异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果视觉库可用，则添加特征提取相关模块结构
    _import_structure["feature_extraction_flava"] = ["FlavaFeatureExtractor"]
    _import_structure["image_processing_flava"] = ["FlavaImageProcessor"]
    _import_structure["processing_flava"] = ["FlavaProcessor"]

# 检查 Torch 库是否可用，若不可用则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 库可用，则添加建模相关模块结构
    _import_structure["modeling_flava"] = [
        "FLAVA_PRETRAINED_MODEL_ARCHIVE_LIST",
        "FlavaForPreTraining",
        "FlavaImageCodebook",
        "FlavaImageModel",
        "FlavaModel",
        "FlavaMultimodalModel",
        "FlavaPreTrainedModel",
        "FlavaTextModel",
    ]

# 如果是类型检查，引入相应的模块
if TYPE_CHECKING:
    from .configuration_flava import (
        FLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP,
        FlavaConfig,
        FlavaImageCodebookConfig,
        FlavaImageConfig,
        FlavaMultimodalConfig,
        FlavaTextConfig,
    )

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .feature_extraction_flava import FlavaFeatureExtractor
        from .image_processing_flava import FlavaImageProcessor
        from .processing_flava import FlavaProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_flava import (
            FLAVA_PRETRAINED_MODEL_ARCHIVE_LIST,
            FlavaForPreTraining,
            FlavaImageCodebook,
            FlavaImageModel,
            FlavaModel,
            FlavaMultimodalModel,
            FlavaPreTrainedModel,
            FlavaTextModel,
        )

# 如果不是类型检查，引入系统模块
else:
    import sys
    # 将当前模块注册到sys.modules中，使用懒加载模式
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```