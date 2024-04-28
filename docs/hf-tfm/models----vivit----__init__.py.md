# `.\transformers\models\vivit\__init__.py`

```
# flake8: noqa
# 用于忽略 flake8 对该模块的检查，并且还能保留其他警告

# 版权声明
# 版权归 2023 年的 HuggingFace 团队所有

# 导入类型检查模块
from typing import TYPE_CHECKING

# 依赖于 isort 来合并导入的模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义导入结构
_import_structure = {
    "configuration_vivit": ["VIVIT_PRETRAINED_CONFIG_ARCHIVE_MAP", "VivitConfig"],
}

# 检查是否可用视觉处理库
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["image_processing_vivit"] = ["VivitImageProcessor"]

# 检查是否可用 torch 库
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_vivit"] = [
        "VIVIT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "VivitModel",
        "VivitPreTrainedModel",
        "VivitForVideoClassification",
    ]

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 导入配置 VivitConfig
    from .configuration_vivit import VIVIT_PRETRAINED_CONFIG_ARCHIVE_MAP, VivitConfig

    # 检查是否可用视觉处理库
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 VivitImageProcessor
        from .image_processing_vivit import VivitImageProcessor

    # 检查是否可用 torch 库
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 VivitForVideoClassification, VivitModel, VivitPreTrainedModel
        from .modeling_vivit import (
            VIVIT_PRETRAINED_MODEL_ARCHIVE_LIST,
            VivitForVideoClassification,
            VivitModel,
            VivitPreTrainedModel,
        )

# 如果不在类型检查模式下
else:
    import sys
    # 将该模块交给 LazyModule 包装
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```