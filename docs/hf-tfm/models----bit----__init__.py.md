# `.\models\bit\__init__.py`

```
# 版权声明及许可证信息
# 2022 年版权归 HuggingFace 团队所有。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本许可，除非符合许可证的要求，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按"原样"分发，不提供任何明示或暗示的担保或条件。
# 有关详细信息，请参阅许可证。
from typing import TYPE_CHECKING

# 从工具包中导入异常处理和模块延迟加载工具
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义导入结构，初始化空字典
_import_structure = {"configuration_bit": ["BIT_PRETRAINED_CONFIG_ARCHIVE_MAP", "BitConfig", "BitOnnxConfig"]}

# 检查是否可用 Torch 库，若不可用则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加模型相关的导入结构到_import_structure字典中
    _import_structure["modeling_bit"] = [
        "BIT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "BitForImageClassification",
        "BitModel",
        "BitPreTrainedModel",
        "BitBackbone",
    ]

# 检查是否可用 Vision 库，若不可用则引发异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加图像处理相关的导入结构到_import_structure字典中
    _import_structure["image_processing_bit"] = ["BitImageProcessor"]

# 如果是类型检查阶段，则从相应模块中导入特定类和常量
if TYPE_CHECKING:
    from .configuration_bit import BIT_PRETRAINED_CONFIG_ARCHIVE_MAP, BitConfig, BitOnnxConfig

    # 检查是否可用 Torch 库，若不可用则引发异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，从模型相关模块中导入特定类
        from .modeling_bit import (
            BIT_PRETRAINED_MODEL_ARCHIVE_LIST,
            BitBackbone,
            BitForImageClassification,
            BitModel,
            BitPreTrainedModel,
        )

    # 检查是否可用 Vision 库，若不可用则引发异常
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，从图像处理相关模块中导入特定类
        from .image_processing_bit import BitImageProcessor

# 如果不是类型检查阶段，则将当前模块设置为懒加载模块
else:
    import sys

    # 将当前模块替换为懒加载模块，用于延迟导入指定的结构
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```