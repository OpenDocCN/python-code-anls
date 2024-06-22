# `.\transformers\models\yolos\__init__.py`

```py
# 导入类型检查模块
from typing import TYPE_CHECKING

# 导入必要的依赖
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义模块导入结构
_import_structure = {"configuration_yolos": ["YOLOS_PRETRAINED_CONFIG_ARCHIVE_MAP", "YolosConfig", "YolosOnnxConfig"]}

# 检查视觉库是否可用
try:
    if not is_vision_available():
        # 视觉库不可用则引发可选依赖不可用异常
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 定义模块导入结构中的其他模块
    _import_structure["feature_extraction_yolos"] = ["YolosFeatureExtractor"]
    _import_structure["image_processing_yolos"] = ["YolosImageProcessor"]

# 检查 Torch 库是否可用
try:
    if not is_torch_available():
        # Torch 库不可用则引发可选依赖不可用异常
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 定义模块导入结构中的其他模块
    _import_structure["modeling_yolos"] = [
        "YOLOS_PRETRAINED_MODEL_ARCHIVE_LIST",
        "YolosForObjectDetection",
        "YolosModel",
        "YolosPreTrainedModel",
    ]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 导入配置模块相关内容
    from .configuration_yolos import YOLOS_PRETRAINED_CONFIG_ARCHIVE_MAP, YolosConfig, YolosOnnxConfig

    # 检查视觉库是否可用
    try:
        if not is_vision_available():
            # 视觉库不可用则引发可选依赖不可用异常
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入特征提取和图像处理模块相关内容
        from .feature_extraction_yolos import YolosFeatureExtractor
        from .image_processing_yolos import YolosImageProcessor

    # 检查 Torch 库是否可用
    try:
        if not is_torch_available():
            # Torch 库不可用则引发可选依赖不可用异常
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 YOLOS 模型相关内容
        from .modeling_yolos import (
            YOLOS_PRETRAINED_MODEL_ARCHIVE_LIST,
            YolosForObjectDetection,
            YolosModel,
            YolosPreTrainedModel,
        )

# 如果不是类型检查模式
else:
    # 导入 sys 模块
    import sys

    # 定义模块为延迟加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```