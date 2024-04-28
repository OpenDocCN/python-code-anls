# `.\transformers\models\mobilenet_v1\__init__.py`

```
# 导入类型检查模块
from typing import TYPE_CHECKING

# 导入可选依赖未安装异常和懒加载模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义导入结构字典，包含模块的导入结构
_import_structure = {
    "configuration_mobilenet_v1": [
        "MOBILENET_V1_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "MobileNetV1Config",
        "MobileNetV1OnnxConfig",
    ],
}

# 检查视觉库是否可用，若不可用则引发可选依赖未安装异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果视觉库可用，添加特征提取模块和图像处理模块到导入结构字典
    _import_structure["feature_extraction_mobilenet_v1"] = ["MobileNetV1FeatureExtractor"]
    _import_structure["image_processing_mobilenet_v1"] = ["MobileNetV1ImageProcessor"]

# 检查PyTorch库是否可用，若不可用则引发可选依赖未安装异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果PyTorch库可用，添加模型建模模块到导入结构字典
    _import_structure["modeling_mobilenet_v1"] = [
        "MOBILENET_V1_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MobileNetV1ForImageClassification",
        "MobileNetV1Model",
        "MobileNetV1PreTrainedModel",
        "load_tf_weights_in_mobilenet_v1",
    ]

# 如果是类型检查环境
if TYPE_CHECKING:
    # 导入 Mobilenet V1 配置相关的类
    from .configuration_mobilenet_v1 import (
        MOBILENET_V1_PRETRAINED_CONFIG_ARCHIVE_MAP,
        MobileNetV1Config,
        MobileNetV1OnnxConfig,
    )

    # 检查视觉库是否可用，若不可用则忽略导入
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果视觉库可用，导入特征提取器和图像处理器
        from .feature_extraction_mobilenet_v1 import MobileNetV1FeatureExtractor
        from .image_processing_mobilenet_v1 import MobileNetV1ImageProcessor

    # 检查PyTorch库是否可用，若不可用则忽略导入
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果PyTorch库可用，导入Mobilenet V1模型相关的类和函数
        from .modeling_mobilenet_v1 import (
            MOBILENET_V1_PRETRAINED_MODEL_ARCHIVE_LIST,
            MobileNetV1ForImageClassification,
            MobileNetV1Model,
            MobileNetV1PreTrainedModel,
            load_tf_weights_in_mobilenet_v1,
        )

# 如果不是类型检查环境
else:
    # 导入sys模块
    import sys

    # 将当前模块替换为懒加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```