# `.\transformers\models\beit\__init__.py`

```
# 导入类型检查模块
from typing import TYPE_CHECKING
# 导入必要的依赖和模块
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_torch_available,
    is_vision_available,
)

# 定义模块导入结构字典
_import_structure = {"configuration_beit": ["BEIT_PRETRAINED_CONFIG_ARCHIVE_MAP", "BeitConfig", "BeitOnnxConfig"]}

# 检查视觉处理模块是否可用
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 添加视觉特征提取模块
    _import_structure["feature_extraction_beit"] = ["BeitFeatureExtractor"]
    # 添加 BEIT 图像处理模块
    _import_structure["image_processing_beit"] = ["BeitImageProcessor"]

# 检查是否可用 torch 模块
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 添加 BEIT 模型相关模块
    _import_structure["modeling_beit"] = [
        "BEIT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "BeitForImageClassification",
        "BeitForMaskedImageModeling",
        "BeitForSemanticSegmentation",
        "BeitModel",
        "BeitPreTrainedModel",
        "BeitBackbone",
    ]

# 检查是否可用 flax 模块
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 添加 flax BEIT 模型相关模块
    _import_structure["modeling_flax_beit"] = [
        "FlaxBeitForImageClassification",
        "FlaxBeitForMaskedImageModeling",
        "FlaxBeitModel",
        "FlaxBeitPreTrainedModel",
    ]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 导入 BEIT 配置相关内容
    from .configuration_beit import BEIT_PRETRAINED_CONFIG_ARCHIVE_MAP, BeitConfig, BeitOnnxConfig

    try:
        # 检查视觉处理模块是否可用
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 BEIT 特征提取器和图像处理器
        from .feature_extraction_beit import BeitFeatureExtractor
        from .image_processing_beit import BeitImageProcessor

    try:
        # 检查 torch 模块是否可用
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 BEIT 模型相关内容
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
        # 检查 flax 模块是否可用
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 否则，从当前目录的模块中导入以下模型
    from .modeling_flax_beit import (
        FlaxBeitForImageClassification,  # 导入用于图像分类的 FlaxBeitForImageClassification 模型
        FlaxBeitForMaskedImageModeling,  # 导入用于图像掩模建模的 FlaxBeitForMaskedImageModeling 模型
        FlaxBeitModel,  # 导入 FlaxBeitModel 模型
        FlaxBeitPreTrainedModel,  # 导入预训练模型的 FlaxBeitPreTrainedModel
    )
# 如果不在主模块中，即不在顶层执行的情况下，导入 sys 模块
else:
    # 导入 sys 模块，用于处理系统相关的功能
    import sys

    # 使用 sys.modules 字典，将当前模块对象（__name__）的引用指向 _LazyModule 类的实例
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```