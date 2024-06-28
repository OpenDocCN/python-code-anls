# `.\models\imagegpt\__init__.py`

```
# 版权声明和许可证信息，指明代码版权归HuggingFace团队所有，使用Apache License, Version 2.0许可证
#
# from typing import TYPE_CHECKING 导入类型检查相关模块

# 从 utils 模块中导入 OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available 函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义模块导入结构的字典 _import_structure
_import_structure = {
    "configuration_imagegpt": ["IMAGEGPT_PRETRAINED_CONFIG_ARCHIVE_MAP", "ImageGPTConfig", "ImageGPTOnnxConfig"]
}

# 尝试检测视觉处理是否可用，若不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若视觉处理可用，则向 _import_structure 添加 feature_extraction_imagegpt 和 image_processing_imagegpt 模块及其对应的函数列表
    _import_structure["feature_extraction_imagegpt"] = ["ImageGPTFeatureExtractor"]
    _import_structure["image_processing_imagegpt"] = ["ImageGPTImageProcessor"]

# 尝试检测是否可用 Torch 库，若不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若 Torch 可用，则向 _import_structure 添加 modeling_imagegpt 模块及其对应的函数列表
    _import_structure["modeling_imagegpt"] = [
        "IMAGEGPT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ImageGPTForCausalImageModeling",
        "ImageGPTForImageClassification",
        "ImageGPTModel",
        "ImageGPTPreTrainedModel",
        "load_tf_weights_in_imagegpt",
    ]

# 如果当前是类型检查模式
if TYPE_CHECKING:
    # 从 configuration_imagegpt 模块导入 IMAGEGPT_PRETRAINED_CONFIG_ARCHIVE_MAP, ImageGPTConfig, ImageGPTOnnxConfig 类和常量
    from .configuration_imagegpt import IMAGEGPT_PRETRAINED_CONFIG_ARCHIVE_MAP, ImageGPTConfig, ImageGPTOnnxConfig

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 从 feature_extraction_imagegpt 模块导入 ImageGPTFeatureExtractor 类
        from .feature_extraction_imagegpt import ImageGPTFeatureExtractor
        # 从 image_processing_imagegpt 模块导入 ImageGPTImageProcessor 类

        from .image_processing_imagegpt import ImageGPTImageProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 从 modeling_imagegpt 模块导入相关类和函数
        from .modeling_imagegpt import (
            IMAGEGPT_PRETRAINED_MODEL_ARCHIVE_LIST,
            ImageGPTForCausalImageModeling,
            ImageGPTForImageClassification,
            ImageGPTModel,
            ImageGPTPreTrainedModel,
            load_tf_weights_in_imagegpt,
        )

# 如果不是类型检查模式，则动态设置当前模块为懒加载模块，使用 _LazyModule 包装，并指定模块导入结构 _import_structure
else:
    import sys
    # 将当前模块设置为 _LazyModule 类的实例，以支持延迟加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```