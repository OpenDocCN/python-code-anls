# `.\models\vilt\__init__.py`

```
# 导入必要的模块和函数
from typing import TYPE_CHECKING

# 导入自定义的异常类，用于处理可选依赖不可用的情况
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义模块的导入结构，包括各个子模块的名称和导入项
_import_structure = {"configuration_vilt": ["VILT_PRETRAINED_CONFIG_ARCHIVE_MAP", "ViltConfig"]}

# 检查视觉功能是否可用，若不可用则抛出自定义异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加视觉特征提取、图像处理和处理模块到导入结构中
    _import_structure["feature_extraction_vilt"] = ["ViltFeatureExtractor"]
    _import_structure["image_processing_vilt"] = ["ViltImageProcessor"]
    _import_structure["processing_vilt"] = ["ViltProcessor"]

# 检查是否可用的 PyTorch 包，若不可用则抛出自定义异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加模型相关的各类模块到导入结构中
    _import_structure["modeling_vilt"] = [
        "VILT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ViltForImageAndTextRetrieval",
        "ViltForImagesAndTextClassification",
        "ViltForTokenClassification",
        "ViltForMaskedLM",
        "ViltForQuestionAnswering",
        "ViltLayer",
        "ViltModel",
        "ViltPreTrainedModel",
    ]

# 如果是类型检查阶段，则导入具体的配置和模型类
if TYPE_CHECKING:
    from .configuration_vilt import VILT_PRETRAINED_CONFIG_ARCHIVE_MAP, ViltConfig

    # 检查视觉功能是否可用，若可用则导入相应的特征提取、图像处理和处理模块
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .feature_extraction_vilt import ViltFeatureExtractor
        from .image_processing_vilt import ViltImageProcessor
        from .processing_vilt import ViltProcessor

    # 检查 PyTorch 包是否可用，若可用则导入模型相关的各类模块
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_vilt import (
            VILT_PRETRAINED_MODEL_ARCHIVE_LIST,
            ViltForImageAndTextRetrieval,
            ViltForImagesAndTextClassification,
            ViltForMaskedLM,
            ViltForQuestionAnswering,
            ViltForTokenClassification,
            ViltLayer,
            ViltModel,
            ViltPreTrainedModel,
        )

# 如果不是类型检查阶段，则设置当前模块为懒加载模块
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```