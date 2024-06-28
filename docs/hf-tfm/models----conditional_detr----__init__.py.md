# `.\models\conditional_detr\__init__.py`

```
# 导入类型检查模块
from typing import TYPE_CHECKING

# 导入自定义的异常类，用于处理缺失的可选依赖
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义导入结构字典，包含各模块及其对应的导入项
_import_structure = {
    "configuration_conditional_detr": [
        "CONDITIONAL_DETR_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "ConditionalDetrConfig",
        "ConditionalDetrOnnxConfig",
    ]
}

# 检查视觉处理模块是否可用，若不可用则抛出异常处理
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加特征提取和图像处理模块到导入结构中
    _import_structure["feature_extraction_conditional_detr"] = ["ConditionalDetrFeatureExtractor"]
    _import_structure["image_processing_conditional_detr"] = ["ConditionalDetrImageProcessor"]

# 检查 Torch 是否可用，若不可用则抛出异常处理
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 可用，则添加建模相关模块到导入结构中
    _import_structure["modeling_conditional_detr"] = [
        "CONDITIONAL_DETR_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ConditionalDetrForObjectDetection",
        "ConditionalDetrForSegmentation",
        "ConditionalDetrModel",
        "ConditionalDetrPreTrainedModel",
    ]

# 如果是类型检查模式，则进行额外的导入操作
if TYPE_CHECKING:
    from .configuration_conditional_detr import (
        CONDITIONAL_DETR_PRETRAINED_CONFIG_ARCHIVE_MAP,
        ConditionalDetrConfig,
        ConditionalDetrOnnxConfig,
    )

    # 检查视觉处理模块是否可用，若可用则导入特征提取和图像处理模块
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .feature_extraction_conditional_detr import ConditionalDetrFeatureExtractor
        from .image_processing_conditional_detr import ConditionalDetrImageProcessor

    # 检查 Torch 是否可用，若可用则导入建模相关模块
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_conditional_detr import (
            CONDITIONAL_DETR_PRETRAINED_MODEL_ARCHIVE_LIST,
            ConditionalDetrForObjectDetection,
            ConditionalDetrForSegmentation,
            ConditionalDetrModel,
            ConditionalDetrPreTrainedModel,
        )

# 如果不是类型检查模式，则将当前模块定义为懒加载模块
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```