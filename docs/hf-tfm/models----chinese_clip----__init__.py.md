# `.\models\chinese_clip\__init__.py`

```py
# 导入必要的模块和函数
from typing import TYPE_CHECKING
# 从当前包中导入自定义异常和模块惰性加载类
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义模块的导入结构，包括不同功能模块的导入列表
_import_structure = {
    "configuration_chinese_clip": [
        "CHINESE_CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "ChineseCLIPConfig",
        "ChineseCLIPOnnxConfig",
        "ChineseCLIPTextConfig",
        "ChineseCLIPVisionConfig",
    ],
    "processing_chinese_clip": ["ChineseCLIPProcessor"],
}

# 检查视觉处理模块是否可用，若不可用则抛出自定义异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则添加视觉特征提取和图像处理模块到导入结构中
    _import_structure["feature_extraction_chinese_clip"] = ["ChineseCLIPFeatureExtractor"]
    _import_structure["image_processing_chinese_clip"] = ["ChineseCLIPImageProcessor"]

# 检查是否Torch模块可用，若不可用则抛出自定义异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若Torch可用，则添加模型相关模块到导入结构中
    _import_structure["modeling_chinese_clip"] = [
        "CHINESE_CLIP_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ChineseCLIPModel",
        "ChineseCLIPPreTrainedModel",
        "ChineseCLIPTextModel",
        "ChineseCLIPVisionModel",
    ]

# 如果类型检查开启，导入相关配置和处理模块
if TYPE_CHECKING:
    from .configuration_chinese_clip import (
        CHINESE_CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP,
        ChineseCLIPConfig,
        ChineseCLIPOnnxConfig,
        ChineseCLIPTextConfig,
        ChineseCLIPVisionConfig,
    )
    from .processing_chinese_clip import ChineseCLIPProcessor

    # 检查视觉处理模块是否可用，若不可用则跳过导入
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若可用，则导入视觉特征提取和图像处理模块
        from .feature_extraction_chinese_clip import ChineseCLIPFeatureExtractor, ChineseCLIPImageProcessor

    # 检查Torch模块是否可用，若不可用则跳过导入
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若Torch可用，则导入模型相关模块
        from .modeling_chinese_clip import (
            CHINESE_CLIP_PRETRAINED_MODEL_ARCHIVE_LIST,
            ChineseCLIPModel,
            ChineseCLIPPreTrainedModel,
            ChineseCLIPTextModel,
            ChineseCLIPVisionModel,
        )

# 若不是类型检查模式，则将当前模块定义为惰性加载模块
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```