# `.\models\dpt\__init__.py`

```
# 版权声明和许可证信息
# 版权声明和许可证信息

from typing import TYPE_CHECKING
# 引入类型检查模块

from ...file_utils import _LazyModule, is_tokenizers_available, is_torch_available, is_vision_available
# 从文件工具模块中导入_LazyModule、is_tokenizers_available、is_torch_available、is_vision_available函数
from ...utils import OptionalDependencyNotAvailable
# 从实用工具模块中导入OptionalDependencyNotAvailable类

_import_structure = {"configuration_dpt": ["DPT_PRETRAINED_CONFIG_ARCHIVE_MAP", "DPTConfig"]}
# 构建导入结构字典

try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["feature_extraction_dpt"] = ["DPTFeatureExtractor"]
    _import_structure["image_processing_dpt"] = ["DPTImageProcessor"]
# 检查视觉模块是否可用，若可用则向导入结构字典中添加特征提取和图像处理模块

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_dpt"] = [
        "DPT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "DPTForDepthEstimation",
        "DPTForSemanticSegmentation",
        "DPTModel",
        "DPTPreTrainedModel",
    ]
# 检查torch模块是否可用，若可用则向导入结构字典中添加深度估计、语义分割、模型等模块

if TYPE_CHECKING:
    from .configuration_dpt import DPT_PRETRAINED_CONFIG_ARCHIVE_MAP, DPTConfig
    # 如果是类型检查阶段，则需要导入配置模块的特定内容

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .feature_extraction_dpt import DPTFeatureExtractor
        from .image_processing_dpt import DPTImageProcessor
        # 如果视觉模块可用，导入特征提取和图像处理模块

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_dpt import (
            DPT_PRETRAINED_MODEL_ARCHIVE_LIST,
            DPTForDepthEstimation,
            DPTForSemanticSegmentation,
            DPTModel,
            DPTPreTrainedModel,
        )
        # 如果torch模块可用，导入深度估计、语义分割、模型等模块

else:
    import sys
    # 如果不是类型检查阶段，则导入sys模块

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
    # 将模块指向_LazyModule类实例，延迟加载导入结构字典中的模块
```