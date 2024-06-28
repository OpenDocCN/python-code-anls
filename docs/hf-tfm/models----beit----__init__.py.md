# `.\models\beit\__init__.py`

```py
# 引入类型检查模块，用于条件类型检查
from typing import TYPE_CHECKING

# 从工具模块中引入必要的依赖，包括自定义的异常和延迟加载模块
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_torch_available,
    is_vision_available,
)

# 定义一个字典，用于存储导入结构，包含待导入模块的名称和对应的成员列表
_import_structure = {"configuration_beit": ["BEIT_PRETRAINED_CONFIG_ARCHIVE_MAP", "BeitConfig", "BeitOnnxConfig"]}

# 检查视觉处理模块是否可用，若不可用则抛出自定义的依赖不可用异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则添加视觉特征提取模块和图像处理模块到导入结构中
    _import_structure["feature_extraction_beit"] = ["BeitFeatureExtractor"]
    _import_structure["image_processing_beit"] = ["BeitImageProcessor"]

# 检查 Torch 是否可用，若不可用则抛出自定义的依赖不可用异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则添加 BEIT 模型相关模块到导入结构中
    _import_structure["modeling_beit"] = [
        "BEIT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "BeitForImageClassification",
        "BeitForMaskedImageModeling",
        "BeitForSemanticSegmentation",
        "BeitModel",
        "BeitPreTrainedModel",
        "BeitBackbone",
    ]

# 检查 Flax 是否可用，若不可用则抛出自定义的依赖不可用异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则添加 Flax BEIT 模型相关模块到导入结构中
    _import_structure["modeling_flax_beit"] = [
        "FlaxBeitForImageClassification",
        "FlaxBeitForMaskedImageModeling",
        "FlaxBeitModel",
        "FlaxBeitPreTrainedModel",
    ]

# 若在类型检查环境下，则添加详细的导入语句以满足类型检查的需求
if TYPE_CHECKING:
    # 导入 BEIT 配置相关的类和常量
    from .configuration_beit import BEIT_PRETRAINED_CONFIG_ARCHIVE_MAP, BeitConfig, BeitOnnxConfig

    try:
        # 检查视觉处理模块是否可用
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入视觉特征提取和图像处理模块
        from .feature_extraction_beit import BeitFeatureExtractor
        from .image_processing_beit import BeitImageProcessor

    try:
        # 检查 Torch 是否可用
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 BEIT 模型相关的类和常量
        from .modeling_beit import (
            BEIT_PRETRAINED_MODEL_ARCHIVE_LIST,
            BeitBackbone,
            BeitForImageClassification,
            BeitForMaskedImageModeling,
            BeitForSemanticSegmentation,
            BeitModel,
            BeitPreTrainedModel,
        )

    try:
        # 检查 Flax 是否可用
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 从当前目录中导入模块
        from .modeling_flax_beit import (
            FlaxBeitForImageClassification,
            FlaxBeitForMaskedImageModeling,
            FlaxBeitModel,
            FlaxBeitPreTrainedModel,
        )
else:
    # 导入 sys 模块，用于动态配置模块信息
    import sys
    # 将当前模块 (__name__) 的内容替换为 _LazyModule 的实例，
    # 以延迟加载模块内容，传入模块名、模块文件名、导入结构和模块规范
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```