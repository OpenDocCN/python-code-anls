# `.\models\swin2sr\__init__.py`

```
# 版权声明和许可证信息，声明代码版权归 HuggingFace 团队所有，使用 Apache License 2.0 许可证发布
# 可以在符合许可证的情况下使用此文件。许可证详细信息可在 http://www.apache.org/licenses/LICENSE-2.0 获取
#
# 如果不符合适用法律或未经书面同意，则根据"AS IS"基础分发软件，无任何明示或暗示的担保或条件
from typing import TYPE_CHECKING

# 导入 OptionalDependencyNotAvailable 异常类、_LazyModule 类以及检查 torch 和 vision 是否可用的函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义导入结构的字典，包含模块到需要导入的类、函数的映射
_import_structure = {
    "configuration_swin2sr": ["SWIN2SR_PRETRAINED_CONFIG_ARCHIVE_MAP", "Swin2SRConfig"],
}

# 检查是否可以导入 torch
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()  # 如果 torch 不可用则抛出 OptionalDependencyNotAvailable 异常
except OptionalDependencyNotAvailable:
    pass  # 如果出现 OptionalDependencyNotAvailable 异常则不执行后续代码
else:
    # 如果 torch 可用，则添加 modeling_swin2sr 模块到导入结构中
    _import_structure["modeling_swin2sr"] = [
        "SWIN2SR_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Swin2SRForImageSuperResolution",
        "Swin2SRModel",
        "Swin2SRPreTrainedModel",
    ]

# 检查是否可以导入 vision
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()  # 如果 vision 不可用则抛出 OptionalDependencyNotAvailable 异常
except OptionalDependencyNotAvailable:
    pass  # 如果出现 OptionalDependencyNotAvailable 异常则不执行后续代码
else:
    # 如果 vision 可用，则添加 image_processing_swin2sr 模块到导入结构中
    _import_structure["image_processing_swin2sr"] = ["Swin2SRImageProcessor"]

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 导入 configuration_swin2sr 模块中的特定类和变量
    from .configuration_swin2sr import SWIN2SR_PRETRAINED_CONFIG_ARCHIVE_MAP, Swin2SRConfig

    # 检查是否可以导入 torch
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()  # 如果 torch 不可用则抛出 OptionalDependencyNotAvailable 异常
    except OptionalDependencyNotAvailable:
        pass  # 如果出现 OptionalDependencyNotAvailable 异常则不执行后续代码
    else:
        # 导入 modeling_swin2sr 模块中的特定类和变量
        from .modeling_swin2sr import (
            SWIN2SR_PRETRAINED_MODEL_ARCHIVE_LIST,
            Swin2SRForImageSuperResolution,
            Swin2SRModel,
            Swin2SRPreTrainedModel,
        )

    # 检查是否可以导入 vision
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()  # 如果 vision 不可用则抛出 OptionalDependencyNotAvailable 异常
    except OptionalDependencyNotAvailable:
        pass  # 如果出现 OptionalDependencyNotAvailable 异常则不执行后续代码
    else:
        # 导入 image_processing_swin2sr 模块中的特定类
        from .image_processing_swin2sr import Swin2SRImageProcessor

# 如果不在类型检查模式下，则将当前模块映射到 _LazyModule，延迟导入模块，以及动态导入 _import_structure 中定义的模块
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```