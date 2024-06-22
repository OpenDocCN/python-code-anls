# `.\models\detr\__init__.py`

```py
# 引入类型检查
from typing import TYPE_CHECKING
# 引入可选依赖未安装的异常
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义导入结构
_import_structure = {"configuration_detr": ["DETR_PRETRAINED_CONFIG_ARCHIVE_MAP", "DetrConfig", "DetrOnnxConfig"]}

# 如果视觉包不可用，则抛出可选依赖不可用异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果视觉包可用，则添加以下导入结构
    _import_structure["feature_extraction_detr"] = ["DetrFeatureExtractor"]
    _import_structure["image_processing_detr"] = ["DetrImageProcessor"]

# 如果 torch 包不可用，则抛出可选依赖不可用异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 torch 包可用，则添加以下导入结构
    _import_structure["modeling_detr"] = [
        "DETR_PRETRAINED_MODEL_ARCHIVE_LIST",
        "DetrForObjectDetection",
        "DetrForSegmentation",
        "DetrModel",
        "DetrPreTrainedModel",
    ]

# 如果是类型检查阶段，则进行以下导入
if TYPE_CHECKING:
    from .configuration_detr import DETR_PRETRAINED_CONFIG_ARCHIVE_MAP, DetrConfig, DetrOnnxConfig

    # 如果视觉包可用，则进行以下导入
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .feature_extraction_detr import DetrFeatureExtractor
        from .image_processing_detr import DetrImageProcessor

    # 如果 torch 包可用，则进行以下导入
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_detr import (
            DETR_PRETRAINED_MODEL_ARCHIVE_LIST,
            DetrForObjectDetection,
            DetrForSegmentation,
            DetrModel,
            DetrPreTrainedModel,
        )

# 如果不是类型检查阶段，则进行 Lazy 模块导入
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```