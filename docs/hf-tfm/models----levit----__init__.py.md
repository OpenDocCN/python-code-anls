# `.\transformers\models\levit\__init__.py`

```
# 定义版权声明和许可证信息
# 使用 import 语句导入模块
from typing import TYPE_CHECKING
# 从工具包中导入OptionalDependencyNotAvailable、_LazyModule、is_torch_available、is_vision_available
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义需要导入的模块结构
_import_structure = {"configuration_levit": ["LEVIT_PRETRAINED_CONFIG_ARCHIVE_MAP", "LevitConfig", "LevitOnnxConfig"]}

# 检查视觉处理模块是否可用，不可用则抛出OptionalDependencyNotAvailable异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["feature_extraction_levit"] = ["LevitFeatureExtractor"]
    _import_structure["image_processing_levit"] = ["LevitImageProcessor"]

# 检查torch模块是否可用，不可用则抛出OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_levit"] = [
        "LEVIT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "LevitForImageClassification",
        "LevitForImageClassificationWithTeacher",
        "LevitModel",
        "LevitPreTrainedModel",
    ]

# 如果是类型检查阶段
if TYPE_CHECKING:
    # 从configuration_levit模块导入特定的类和变量
    from .configuration_levit import LEVIT_PRETRAINED_CONFIG_ARCHIVE_MAP, LevitConfig, LevitOnnxConfig

    # 检查视觉处理模块是否可用，不可用则抛出OptionalDependencyNotAvailable异常
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 从feature_extraction_levit模块导入特定类
        from .feature_extraction_levit import LevitFeatureExtractor
        # 从image_processing_levit模块导入特定类
        from .image_processing_levit import LevitImageProcessor

    # 检查torch模块是否可用，不可用则抛出OptionalDependencyNotAvailable异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 从modeling_levit模块导入特定类和变量
        from .modeling_levit import (
            LEVIT_PRETRAINED_MODEL_ARCHIVE_LIST,
            LevitForImageClassification,
            LevitForImageClassificationWithTeacher,
            LevitModel,
            LevitPreTrainedModel,
        )
# 如果不是类型检查阶段
else:
    # 导入sys模块
    import sys
    # 将当前模块设为懒加载模块LazyModule的实例
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```