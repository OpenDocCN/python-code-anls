# `.\transformers\models\mobilenet_v2\__init__.py`

```py
# 版权声明和许可证信息
# 这是HuggingFace团队版权所有的代码，受到Apache许可证版本2.0的保护
# 在许可证要求的条件下，用户可以在某些限制范围内使用此代码

# 导入Python的类型提示模块
from typing import TYPE_CHECKING

# 从HuggingFace的其他模块导入用于处理依赖项和模块的工具
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义一个字典，指定要导入的结构
_import_structure = {
    "configuration_mobilenet_v2": [
        "MOBILENET_V2_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "MobileNetV2Config",
        "MobileNetV2OnnxConfig",
    ],
}

# 尝试检查视觉处理依赖是否可用
try:
    if not is_vision_available():  # 如果视觉依赖不可用
        raise OptionalDependencyNotAvailable()  # 引发一个异常
except OptionalDependencyNotAvailable:  # 如果引发了OptionalDependencyNotAvailable异常
    pass  # 什么都不做，保持空处理
else:  # 如果没有引发异常，说明视觉依赖可用
    _import_structure["feature_extraction_mobilenet_v2"] = ["MobileNetV2FeatureExtractor"]
    _import_structure["image_processing_mobilenet_v2"] = ["MobileNetV2ImageProcessor"]

# 尝试检查PyTorch依赖是否可用
try:
    if not is_torch_available():  # 如果PyTorch依赖不可用
        raise OptionalDependencyNotAvailable()  # 引发一个异常
except OptionalDependencyNotAvailable:  # 如果引发了OptionalDependencyNotAvailable异常
    pass  # 什么都不做，保持空处理
else:  # 如果没有引发异常，说明PyTorch依赖可用
    _import_structure["modeling_mobilenet_v2"] = [
        "MOBILENET_V2_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MobileNetV2ForImageClassification",
        "MobileNetV2ForSemanticSegmentation",
        "MobileNetV2Model",
        "MobileNetV2PreTrainedModel",
        "load_tf_weights_in_mobilenet_v2",
    ]

# 如果在类型检查上下文中，执行以下导入
if TYPE_CHECKING:
    from .configuration_mobilenet_v2 import (  # 导入配置相关的内容
        MOBILENET_V2_PRETRAINED_CONFIG_ARCHIVE_MAP,
        MobileNetV2Config,
        MobileNetV2OnnxConfig,
    )

    try:  # 尝试检查视觉依赖
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:  # 如果视觉依赖不可用
        pass
    else:  # 如果视觉依赖可用
        from .feature_extraction_mobilenet_v2 import MobileNetV2FeatureExtractor  # 导入特征提取器
        from .image_processing_mobilenet_v2 import MobileNetV2ImageProcessor  # 导入图像处理器

    try:  # 尝试检查PyTorch依赖
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()  # 如果PyTorch依赖不可用
    except OptionalDependencyNotAvailable:  # 如果引发了异常
        pass  # 什么都不做
    else:  # 如果PyTorch依赖可用
        from .modeling_mobilenet_v2 import (  # 导入模型相关的内容
            MOBILENET_V2_PRETRAINED_MODEL_ARCHIVE_LIST,
            MobileNetV2ForImageClassification,
            MobileNetV2ForSemanticSegmentation,
            MobileNetV2Model,
            MobileNetV2PreTrainedModel,
            load_tf_weights_in_mobilenet_v2,
        )

# 如果不是类型检查上下文，则设置当前模块为惰性加载模块
else:
    import sys  # 导入sys模块

    # 使用惰性加载的方式将当前模块设置为LazyModule
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```