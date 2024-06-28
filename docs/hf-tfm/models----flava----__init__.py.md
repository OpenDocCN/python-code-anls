# `.\models\flava\__init__.py`

```py
# 版权声明及导入必要的类型检查
# Meta Platforms 作者和 The HuggingFace Team 版权声明
# Apache License, Version 2.0 版权许可，可以在指定条件下使用此文件
# 如果未按许可条件使用，可能会出现限制和法律责任
from typing import TYPE_CHECKING

# 导入异常处理相关依赖
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义导入结构，包含需要导入的模块和类
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

# 检查是否存在视觉处理相关的依赖，若不存在则引发异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 添加视觉特征提取相关的导入结构
    _import_structure["feature_extraction_flava"] = ["FlavaFeatureExtractor"]
    _import_structure["image_processing_flava"] = ["FlavaImageProcessor"]
    _import_structure["processing_flava"] = ["FlavaProcessor"]

# 检查是否存在 Torch 相关的依赖，若不存在则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 添加模型相关的导入结构
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

# 如果是类型检查阶段，则导入具体模块
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

else:
    # 如果不是类型检查阶段，则直接导入 sys 模块
    import sys
    # 将当前模块注册到 sys.modules 中，使用 LazyModule 进行延迟加载
    sys.modules[__name__] = _LazyModule(
        __name__,  # 当前模块的名称
        globals()["__file__"],  # 当前模块的文件路径
        _import_structure,  # 导入结构的定义
        module_spec=__spec__  # 当前模块的规范对象
    )
```