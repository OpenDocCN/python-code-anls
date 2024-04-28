# `.\models\conditional_detr\__init__.py`

```
# 版权声明及许可证信息

# 导入模块
from typing import TYPE_CHECKING
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义模块导入结构
_import_structure = {
    "configuration_conditional_detr": [
        "CONDITIONAL_DETR_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "ConditionalDetrConfig",
        "ConditionalDetrOnnxConfig",
    ]
}

# 检查视觉相关依赖是否可用，不可用则引发异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，添加相关模块到导入结构中
    _import_structure["feature_extraction_conditional_detr"] = ["ConditionalDetrFeatureExtractor"]
    _import_structure["image_processing_conditional_detr"] = ["ConditionalDetrImageProcessor"]

# 检查 Torch 相关依赖是否可用，不可用则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，添加相关模块到导入结构中
    _import_structure["modeling_conditional_detr"] = [
        "CONDITIONAL_DETR_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ConditionalDetrForObjectDetection",
        "ConditionalDetrForSegmentation",
        "ConditionalDetrModel",
        "ConditionalDetrPreTrainedModel",
    ]

# 如果是类型检查阶段
if TYPE_CHECKING:
    # 导入相关模块，若视觉相关依赖可用，引入相关模块
    from .configuration_conditional_detr import (
        CONDITIONAL_DETR_PRETRAINED_CONFIG_ARCHIVE_MAP,
        ConditionalDetrConfig,
        ConditionalDetrOnnxConfig,
    )

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .feature_extraction_conditional_detr import ConditionalDetrFeatureExtractor
        from .image_processing_conditional_detr import ConditionalDetrImageProcessor

    # 若 Torch 相关依赖可用，引入相关模块
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

# 如果不是类型检查阶段
else:
    import sys

    # 将模块名替换为懒加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```