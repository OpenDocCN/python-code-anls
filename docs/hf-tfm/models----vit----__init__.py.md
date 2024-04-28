# `.\transformers\models\vit\__init__.py`

```
# 版权声明和许可信息
# 版权声明和许可信息，指定了代码的版权和许可信息
# 根据 Apache 许可证 2.0 版本，使用此文件需要遵守许可证规定
# 可以在 http://www.apache.org/licenses/LICENSE-2.0 获取许可证的副本
from typing import TYPE_CHECKING
# 导入必要的模块和函数

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_torch_available,
    is_vision_available,
)
# 从自定义的工具模块中导入必要的函数和类

_import_structure = {"configuration_vit": ["VIT_PRETRAINED_CONFIG_ARCHIVE_MAP", "ViTConfig", "ViTOnnxConfig"]}
# 定义一个字典，包含了模块和对应的导入内容列表

try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["feature_extraction_vit"] = ["ViTFeatureExtractor"]
    _import_structure["image_processing_vit"] = ["ViTImageProcessor"]
# 检查视觉模块是否可用，如果可用则导入特征提取和图像处理模块

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_vit"] = [
        "VIT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ViTForImageClassification",
        "ViTForMaskedImageModeling",
        "ViTModel",
        "ViTPreTrainedModel",
    ]
# 检查 PyTorch 是否可用，如果可用则导入模型相关模块

try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_vit"] = [
        "TFViTForImageClassification",
        "TFViTModel",
        "TFViTPreTrainedModel",
    ]
# 检查 TensorFlow 是否可用，如果可用则导入 TensorFlow 模型相关模块

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_vit"] = [
        "FlaxViTForImageClassification",
        "FlaxViTModel",
        "FlaxViTPreTrainedModel",
    ]
# 检查 Flax 是否可用，如果可用则导入 Flax 模型相关模块

if TYPE_CHECKING:
    from .configuration_vit import VIT_PRETRAINED_CONFIG_ARCHIVE_MAP, ViTConfig, ViTOnnxConfig
    # 如果是类型检查阶段，导入配置相关的模块和类

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .feature_extraction_vit import ViTFeatureExtractor
        from .image_processing_vit import ViTImageProcessor
    # 如果视觉模块可用，导入特征提取和图像处理模块

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_vit import (
            VIT_PRETRAINED_MODEL_ARCHIVE_LIST,
            ViTForImageClassification,
            ViTForMaskedImageModeling,
            ViTModel,
            ViTPreTrainedModel,
        )
    # 如果 PyTorch 可用，导入模型相关模块
    # 尝试检查是否有可用的 TensorFlow 库
    try:
        # 如果 TensorFlow 不可用，则引发 OptionalDependencyNotAvailable 异常
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    # 捕获 OptionalDependencyNotAvailable 异常
    except OptionalDependencyNotAvailable:
        # 什么也不做，继续执行下一步
        pass
    # 如果没有发生异常
    else:
        # 从 modeling_tf_vit 模块中导入指定的类
        from .modeling_tf_vit import TFViTForImageClassification, TFViTModel, TFViTPreTrainedModel

    # 尝试检查是否有可用的 Flax 库
    try:
        # 如果 Flax 不可用，则引发 OptionalDependencyNotAvailable 异常
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    # 捕获 OptionalDependencyNotAvailable 异常
    except OptionalDependencyNotAvailable:
        # 什么也不做，继续执行下一步
        pass
    # 如果没有发生异常
    else:
        # 从 modeling_flax_vit 模块中导入指定的类
        from .modeling_flax_vit import FlaxViTForImageClassification, FlaxViTModel, FlaxViTPreTrainedModel
# 如果不在主模块中，则导入sys模块
import sys
# 将当前模块添加到sys.modules字典中，使用_LazyModule延迟加载模块
sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```