# `.\models\convnext\__init__.py`

```py
# 引入类型检查模块，用于检查类型是否可用
from typing import TYPE_CHECKING

# 从工具模块中引入必要的依赖项和函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tf_available,
    is_torch_available,
    is_vision_available,
)

# 定义一个字典结构，用于存储模块导入的结构
_import_structure = {
    "configuration_convnext": ["CONVNEXT_PRETRAINED_CONFIG_ARCHIVE_MAP", "ConvNextConfig", "ConvNextOnnxConfig"]
}

# 检查视觉处理模块是否可用，若不可用则抛出OptionalDependencyNotAvailable异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则向导入结构中添加特征提取和图像处理的相关内容
    _import_structure["feature_extraction_convnext"] = ["ConvNextFeatureExtractor"]
    _import_structure["image_processing_convnext"] = ["ConvNextImageProcessor"]

# 检查是否torch可用，若不可用则抛出OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则向导入结构中添加模型相关的内容
    _import_structure["modeling_convnext"] = [
        "CONVNEXT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ConvNextForImageClassification",
        "ConvNextModel",
        "ConvNextPreTrainedModel",
        "ConvNextBackbone",
    ]

# 检查是否tf可用，若不可用则抛出OptionalDependencyNotAvailable异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则向导入结构中添加TensorFlow模型相关的内容
    _import_structure["modeling_tf_convnext"] = [
        "TFConvNextForImageClassification",
        "TFConvNextModel",
        "TFConvNextPreTrainedModel",
    ]

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 从配置文件导入相关的配置信息和类定义
    from .configuration_convnext import CONVNEXT_PRETRAINED_CONFIG_ARCHIVE_MAP, ConvNextConfig, ConvNextOnnxConfig

    try:
        # 检查视觉处理模块是否可用
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若可用，则从相应模块导入特征提取和图像处理类
        from .feature_extraction_convnext import ConvNextFeatureExtractor
        from .image_processing_convnext import ConvNextImageProcessor

    try:
        # 检查torch是否可用
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若可用，则从模型定义模块导入相关类和配置信息
        from .modeling_convnext import (
            CONVNEXT_PRETRAINED_MODEL_ARCHIVE_LIST,
            ConvNextBackbone,
            ConvNextForImageClassification,
            ConvNextModel,
            ConvNextPreTrainedModel,
        )

    try:
        # 检查tf是否可用
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果条件不满足，则从当前目录下的 model_tf_convnext 模块中导入以下类和函数
        from .modeling_tf_convnext import TFConvNextForImageClassification, TFConvNextModel, TFConvNextPreTrainedModel
else:
    # 导入 sys 模块，用于操作 Python 解释器的系统功能
    import sys

    # 将当前模块名(__name__)作为键，以 _LazyModule 对象作为值，赋给 sys.modules 字典
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```