# `.\models\levit\__init__.py`

```
# 版权声明和许可信息，指明版权归属和使用许可
# 详情请查阅Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
#
# 如果依据许可法律要求或以书面形式同意，软件将按“原样”分发，不附任何明示或暗示的保证或条件
# 请参阅许可协议以了解特定的语言版本

from typing import TYPE_CHECKING

# 从自定义模块中导入所需函数和类，用以检查环境是否支持特定功能
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义模块导入的结构字典，初始化一些模块路径
_import_structure = {"configuration_levit": ["LEVIT_PRETRAINED_CONFIG_ARCHIVE_MAP", "LevitConfig", "LevitOnnxConfig"]}

# 检查视觉处理功能是否可用，若不可用则引发OptionalDependencyNotAvailable异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加相关模块到导入结构字典中
    _import_structure["feature_extraction_levit"] = ["LevitFeatureExtractor"]
    _import_structure["image_processing_levit"] = ["LevitImageProcessor"]

# 检查是否支持PyTorch环境，若不支持则引发OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果支持PyTorch，添加相关模块到导入结构字典中
    _import_structure["modeling_levit"] = [
        "LEVIT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "LevitForImageClassification",
        "LevitForImageClassificationWithTeacher",
        "LevitModel",
        "LevitPreTrainedModel",
    ]

# 如果是类型检查模式，导入具体的模块和类以进行类型检查
if TYPE_CHECKING:
    from .configuration_levit import LEVIT_PRETRAINED_CONFIG_ARCHIVE_MAP, LevitConfig, LevitOnnxConfig

    # 检查视觉处理功能是否可用，若不可用则跳过
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入视觉特征提取和图像处理相关模块
        from .feature_extraction_levit import LevitFeatureExtractor
        from .image_processing_levit import LevitImageProcessor

    # 检查是否支持PyTorch环境，若不支持则跳过
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入模型相关的PyTorch模块
        from .modeling_levit import (
            LEVIT_PRETRAINED_MODEL_ARCHIVE_LIST,
            LevitForImageClassification,
            LevitForImageClassificationWithTeacher,
            LevitModel,
            LevitPreTrainedModel,
        )

# 如果不是类型检查模式，将当前模块注册为_LazyModule的懒加载模块
else:
    import sys

    # 将当前模块重新指定为_LazyModule的实例，用于延迟导入
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```