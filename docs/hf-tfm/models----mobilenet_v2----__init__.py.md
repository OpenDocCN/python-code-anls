# `.\models\mobilenet_v2\__init__.py`

```py
# 引入类型检查模块
from typing import TYPE_CHECKING

# 引入自定义的异常类和延迟加载模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义模块导入结构的字典，包含各个模块及其对应的导入内容
_import_structure = {
    "configuration_mobilenet_v2": [
        "MOBILENET_V2_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "MobileNetV2Config",
        "MobileNetV2OnnxConfig",
    ],
}

# 检查视觉模块是否可用，若不可用则抛出异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，添加视觉特征提取器和图像处理器到导入结构中
    _import_structure["feature_extraction_mobilenet_v2"] = ["MobileNetV2FeatureExtractor"]
    _import_structure["image_processing_mobilenet_v2"] = ["MobileNetV2ImageProcessor"]

# 检查 Torch 是否可用，若不可用则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若 Torch 可用，添加模型相关的导入内容到导入结构中
    _import_structure["modeling_mobilenet_v2"] = [
        "MOBILENET_V2_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MobileNetV2ForImageClassification",
        "MobileNetV2ForSemanticSegmentation",
        "MobileNetV2Model",
        "MobileNetV2PreTrainedModel",
        "load_tf_weights_in_mobilenet_v2",
    ]

# 如果当前是类型检查模式
if TYPE_CHECKING:
    # 从配置模块中导入所需的配置映射和配置类
    from .configuration_mobilenet_v2 import (
        MOBILENET_V2_PRETRAINED_CONFIG_ARCHIVE_MAP,
        MobileNetV2Config,
        MobileNetV2OnnxConfig,
    )

    # 再次检查视觉模块是否可用，若不可用则抛出异常
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若可用，从特征提取和图像处理模块中导入相应类
        from .feature_extraction_mobilenet_v2 import MobileNetV2FeatureExtractor
        from .image_processing_mobilenet_v2 import MobileNetV2ImageProcessor

    # 再次检查 Torch 是否可用，若不可用则抛出异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若 Torch 可用，从模型相关模块中导入相应类和函数
        from .modeling_mobilenet_v2 import (
            MOBILENET_V2_PRETRAINED_MODEL_ARCHIVE_LIST,
            MobileNetV2ForImageClassification,
            MobileNetV2ForSemanticSegmentation,
            MobileNetV2Model,
            MobileNetV2PreTrainedModel,
            load_tf_weights_in_mobilenet_v2,
        )

# 如果不是类型检查模式，将当前模块设为延迟加载模块
else:
    import sys

    # 使用 LazyModule 将当前模块设为延迟加载模块，指定导入结构和模块规范
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```