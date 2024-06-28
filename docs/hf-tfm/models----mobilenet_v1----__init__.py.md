# `.\models\mobilenet_v1\__init__.py`

```
# 版权声明和许可信息
#
# 根据 Apache License, Version 2.0 授权使用本代码
# 除非遵循许可，否则不得使用本文件
# 可以从以下链接获取许可的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 如果法律要求或书面同意，软件会基于"原样"分发，
# 没有任何明示或暗示的担保或条件。
# 有关具体的语言权利和限制，请参阅许可证。
from typing import TYPE_CHECKING

# 从工具包导入必要的异常和模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义需要导入的模块结构
_import_structure = {
    "configuration_mobilenet_v1": [
        "MOBILENET_V1_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "MobileNetV1Config",
        "MobileNetV1OnnxConfig",
    ],
}

# 检查视觉功能是否可用，若不可用则引发异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 添加特征提取器和图像处理器到导入结构中
    _import_structure["feature_extraction_mobilenet_v1"] = ["MobileNetV1FeatureExtractor"]
    _import_structure["image_processing_mobilenet_v1"] = ["MobileNetV1ImageProcessor"]

# 检查 Torch 是否可用，若不可用则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 添加模型相关模块到导入结构中
    _import_structure["modeling_mobilenet_v1"] = [
        "MOBILENET_V1_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MobileNetV1ForImageClassification",
        "MobileNetV1Model",
        "MobileNetV1PreTrainedModel",
        "load_tf_weights_in_mobilenet_v1",
    ]

# 如果是类型检查阶段，导入特定的配置和模块
if TYPE_CHECKING:
    from .configuration_mobilenet_v1 import (
        MOBILENET_V1_PRETRAINED_CONFIG_ARCHIVE_MAP,
        MobileNetV1Config,
        MobileNetV1OnnxConfig,
    )

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入特征提取器和图像处理器
        from .feature_extraction_mobilenet_v1 import MobileNetV1FeatureExtractor
        from .image_processing_mobilenet_v1 import MobileNetV1ImageProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入模型相关模块
        from .modeling_mobilenet_v1 import (
            MOBILENET_V1_PRETRAINED_MODEL_ARCHIVE_LIST,
            MobileNetV1ForImageClassification,
            MobileNetV1Model,
            MobileNetV1PreTrainedModel,
            load_tf_weights_in_mobilenet_v1,
        )

# 在非类型检查阶段，使用 LazyModule 加载导入结构
else:
    import sys

    # 将当前模块替换为 LazyModule 的实例
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```