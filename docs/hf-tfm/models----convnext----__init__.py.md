# `.\models\convnext\__init__.py`

```py
# 2022年版权声明
#
# 根据 Apache 许可证 2.0 版本许可，除非符合许可证的使用，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"按原样"分发，没有任何明示或暗示的担保或条件
# 请查看许可证以获取关于具体语言规定的权限和限制
from typing import TYPE_CHECKING

# 导入所需模块/功能
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tf_available,
    is_torch_available,
    is_vision_available,
)

# 定义导入结构
_import_structure = {
    "configuration_convnext": ["CONVNEXT_PRETRAINED_CONFIG_ARCHIVE_MAP", "ConvNextConfig", "ConvNextOnnxConfig"]
}

# 如果视觉相关库不可用，则抛出异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 导入特征提取相关功能
    _import_structure["feature_extraction_convnext"] = ["ConvNextFeatureExtractor"]
    # 导入图像处理相关功能
    _import_structure["image_processing_convnext"] = ["ConvNextImageProcessor"]

# 如果 PyTorch 不可用，则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 导入模型构建相关功能
    _import_structure["modeling_convnext"] = [
        "CONVNEXT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ConvNextForImageClassification",
        "ConvNextModel",
        "ConvNextPreTrainedModel",
        "ConvNextBackbone",
    ]

# 如果 TensorFlow 不可用，则抛出异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 导入 TensorFlow 模型构建相关功能
    _import_structure["modeling_tf_convnext"] = [
        "TFConvNextForImageClassification",
        "TFConvNextModel",
        "TFConvNextPreTrainedModel",
    ]

# 如果是类型检查，则导入额外的模块/功能
if TYPE_CHECKING:
    from .configuration_convnext import CONVNEXT_PRETRAINED_CONFIG_ARCHIVE_MAP, ConvNextConfig, ConvNextOnnxConfig
    
    # 如果视觉相关库可用，则导入额外的特征提取和图像处理模块/功能
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .feature_extraction_convnext import ConvNextFeatureExtractor
        from .image_processing_convnext import ConvNextImageProcessor

    # 如果 PyTorch 可用，则导入额外的模型构建模块/功能
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_convnext import (
            CONVNEXT_PRETRAINED_MODEL_ARCHIVE_LIST,
            ConvNextBackbone,
            ConvNextForImageClassification,
            ConvNextModel,
            ConvNextPreTrainedModel,
        )

    # 如果 TensorFlow 可用，则...
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果不满足之前的条件，即不是 PyTorch，则从当前目录的 modeling_tf_convnext 模块中导入指定的类
    from .modeling_tf_convnext import TFConvNextForImageClassification, TFConvNextModel, TFConvNextPreTrainedModel
# 否则，当条件不满足时执行以下操作
import sys  # 导入sys模块，用于系统级操作

# 将当前模块注册到sys.modules中，以当前模块名为键，值为一个LazyModule对象，
# 这个LazyModule对象将模块名、文件路径和导入结构作为参数
sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```