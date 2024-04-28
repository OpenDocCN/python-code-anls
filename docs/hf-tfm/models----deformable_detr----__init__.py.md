# `.\models\deformable_detr\__init__.py`

```
# 导入必要的模块和函数
from typing import TYPE_CHECKING

from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义需要导入的模块和函数的结构
_import_structure = {
    "configuration_deformable_detr": ["DEFORMABLE_DETR_PRETRAINED_CONFIG_ARCHIVE_MAP", "DeformableDetrConfig"],
}

# 检查是否视觉模块可用，若不可用则抛出异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若视觉模块可用，则导入相应的模块和函数
    _import_structure["feature_extraction_deformable_detr"] = ["DeformableDetrFeatureExtractor"]
    _import_structure["image_processing_deformable_detr"] = ["DeformableDetrImageProcessor"]

# 检查是否 torch 模块可用，若不可用则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若 torch 模块可用，则导入相应的模块和函数
    _import_structure["modeling_deformable_detr"] = [
        "DEFORMABLE_DETR_PRETRAINED_MODEL_ARCHIVE_LIST",
        "DeformableDetrForObjectDetection",
        "DeformableDetrModel",
        "DeformableDetrPreTrainedModel",
    ]

# 如果是类型检查模式，则导入其他模块和函数
if TYPE_CHECKING:
    from .configuration_deformable_detr import DEFORMABLE_DETR_PRETRAINED_CONFIG_ARCHIVE_MAP, DeformableDetrConfig

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .feature_extraction_deformable_detr import DeformableDetrFeatureExtractor
        from .image_processing_deformable_detr import DeformableDetrImageProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_deformable_detr import (
            DEFORMABLE_DETR_PRETRAINED_MODEL_ARCHIVE_LIST,
            DeformableDetrForObjectDetection,
            DeformableDetrModel,
            DeformableDetrPreTrainedModel,
        )

# 如果不是类型检查模式，则使用懒加载模块
else:
    import sys

    # 将当前模块定义为懒加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```