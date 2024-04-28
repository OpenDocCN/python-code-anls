# `.\models\idefics\__init__.py`

```py
# 版权声明和许可证信息
# 版权声明和许可证信息，指明代码版权和许可证信息
# 依赖导入
# 导入必要的依赖，包括类型检查和一些自定义的工具函数
from typing import TYPE_CHECKING
# 导入 LazyModule 类，用于延迟加载模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义模块导入结构
_import_structure = {"configuration_idefics": ["IDEFICS_PRETRAINED_CONFIG_ARCHIVE_MAP", "IdeficsConfig"]}

# 检查是否有视觉处理库可用，如果不可用则抛出异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果视觉处理库可用，则添加相关模块到导入结构中
    _import_structure["image_processing_idefics"] = ["IdeficsImageProcessor"]

# 检查是否有 Torch 库可用，如果不可用则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 库可用，则添加相关模块到导入结构中
    _import_structure["modeling_idefics"] = [
        "IDEFICS_PRETRAINED_MODEL_ARCHIVE_LIST",
        "IdeficsForVisionText2Text",
        "IdeficsModel",
        "IdeficsPreTrainedModel",
    ]
    _import_structure["processing_idefics"] = ["IdeficsProcessor"]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 导入配置和模型相关模块
    from .configuration_idefics import IDEFICS_PRETRAINED_CONFIG_ARCHIVE_MAP, IdeficsConfig

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入图像处理相关模块
        from .image_processing_idefics import IdeficsImageProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入模型和处理相关模块
        from .modeling_idefics import (
            IDEFICS_PRETRAINED_MODEL_ARCHIVE_LIST,
            IdeficsForVisionText2Text,
            IdeficsModel,
            IdeficsPreTrainedModel,
        )
        from .processing_idefics import IdeficsProcessor

# 如果不是类型检查模式
else:
    import sys

    # 将当前模块设置为 LazyModule 类的实例
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```