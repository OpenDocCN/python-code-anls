# `.\transformers\models\swin2sr\__init__.py`

```
# 导入类型检查工具
from typing import TYPE_CHECKING
# 引入必要的依赖和模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义模块导入结构
_import_structure = {
    "configuration_swin2sr": ["SWIN2SR_PRETRAINED_CONFIG_ARCHIVE_MAP", "Swin2SRConfig"],
}

# 检查是否有torch可用，若不可用则抛出OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 添加torch可用时的模块导入配置
    _import_structure["modeling_swin2sr"] = [
        "SWIN2SR_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Swin2SRForImageSuperResolution",
        "Swin2SRModel",
        "Swin2SRPreTrainedModel",
    ]

# 检查是否有vision可用，若不可用则抛出OptionalDependencyNotAvailable异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 添加vision可用时的模块导入配置
    _import_structure["image_processing_swin2sr"] = ["Swin2SRImageProcessor"]

# 如果需要进行类型检查
if TYPE_CHECKING:
    # 导入类型检查需要的配置和模型
    from .configuration_swin2sr import SWIN2SR_PRETRAINED_CONFIG_ARCHIVE_MAP, Swin2SRConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_swin2sr import (
            SWIN2SR_PRETRAINED_MODEL_ARCHIVE_LIST,
            Swin2SRForImageSuperResolution,
            Swin2SRModel,
            Swin2SRPreTrainedModel,
        )

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_swin2sr import Swin2SRImageProcessor
# 如果不需要进行类型检查
else:
    import sys
    # 创建LazyModule，指定模块的名字、文件、导入结构和规范
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```