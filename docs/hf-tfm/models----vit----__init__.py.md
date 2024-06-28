# `.\models\vit\__init__.py`

```
# 引入类型检查相关模块
from typing import TYPE_CHECKING

# 从当前包的工具模块中导入需要的函数和异常类
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_torch_available,
    is_vision_available,
)

# 定义模块的导入结构，包含各模块对应的导入内容列表
_import_structure = {"configuration_vit": ["VIT_PRETRAINED_CONFIG_ARCHIVE_MAP", "ViTConfig", "ViTOnnxConfig"]}

# 检查视觉处理模块是否可用，若不可用则抛出异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果模块可用，则导入特征提取和图像处理模块
    _import_structure["feature_extraction_vit"] = ["ViTFeatureExtractor"]
    _import_structure["image_processing_vit"] = ["ViTImageProcessor"]

# 检查是否 Torch 可用，若不可用则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 可用，则导入 PyTorch 模型相关模块
    _import_structure["modeling_vit"] = [
        "VIT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ViTForImageClassification",
        "ViTForMaskedImageModeling",
        "ViTModel",
        "ViTPreTrainedModel",
    ]

# 检查是否 TensorFlow 可用，若不可用则抛出异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 TensorFlow 可用，则导入 TensorFlow 模型相关模块
    _import_structure["modeling_tf_vit"] = [
        "TFViTForImageClassification",
        "TFViTModel",
        "TFViTPreTrainedModel",
    ]

# 检查是否 Flax 可用，若不可用则抛出异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Flax 可用，则导入 Flax 模型相关模块
    _import_structure["modeling_flax_vit"] = [
        "FlaxViTForImageClassification",
        "FlaxViTModel",
        "FlaxViTPreTrainedModel",
    ]

# 如果当前是类型检查环境
if TYPE_CHECKING:
    # 从配置文件模块中导入所需的配置映射和配置类
    from .configuration_vit import VIT_PRETRAINED_CONFIG_ARCHIVE_MAP, ViTConfig, ViTOnnxConfig

    # 检查视觉处理模块是否可用，若不可用则抛出异常
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果模块可用，则从特征提取和图像处理模块导入相应类
        from .feature_extraction_vit import ViTFeatureExtractor
        from .image_processing_vit import ViTImageProcessor

    # 检查是否 Torch 可用，若不可用则抛出异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 Torch 可用，则从 PyTorch 模型相关模块导入相应类
        from .modeling_vit import (
            VIT_PRETRAINED_MODEL_ARCHIVE_LIST,
            ViTForImageClassification,
            ViTForMaskedImageModeling,
            ViTModel,
            ViTPreTrainedModel,
        )
    # 尝试检查是否安装了 TensorFlow，如果未安装则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    # 捕获 OptionalDependencyNotAvailable 异常，表示 TensorFlow 不可用
    except OptionalDependencyNotAvailable:
        # 如果捕获到异常，则不执行任何操作，继续执行后续代码
        pass
    else:
        # 如果未捕获异常，表示 TensorFlow 可用，导入相关模块
        from .modeling_tf_vit import TFViTForImageClassification, TFViTModel, TFViTPreTrainedModel

    # 尝试检查是否安装了 Flax，如果未安装则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    # 捕获 OptionalDependencyNotAvailable 异常，表示 Flax 不可用
    except OptionalDependencyNotAvailable:
        # 如果捕获到异常，则不执行任何操作，继续执行后续代码
        pass
    else:
        # 如果未捕获异常，表示 Flax 可用，导入相关模块
        from .modeling_flax_vit import FlaxViTForImageClassification, FlaxViTModel, FlaxViTPreTrainedModel
else:
    # 如果不在以上任何情况下，则执行以下操作
    import sys
    # 导入系统模块 sys，用于访问系统相关的功能

    # 将当前模块注册到 sys.modules 中，使用 _LazyModule 包装
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
    # __name__: 当前模块的名称
    # globals()["__file__"]: 当前模块的文件路径
    # _import_structure: 导入结构，可能是一个导入相关的结构或函数
    # __spec__: 可能是当前模块的规范对象，指定模块的元数据信息
```