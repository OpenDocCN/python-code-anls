# `.\models\efficientnet\__init__.py`

```py
# flake8: noqa
# 忽略 flake8 对本模块的检查，因为无法只忽略 "F401 '...' imported but unused" 警告，需要保留其他警告

# 版权声明
# 2023年 HuggingFace团队。保留所有权利。
#
# 根据 Apache 许可证版本 2.0（“许可证”）许可。
# 除非符合许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证的一份副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是“按原样”分发的，
# 没有任何明示或暗示的担保或条件。请查看特定语言下的许可证条款和限制。
from typing import TYPE_CHECKING

# 依靠 isort 来合并导入
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义导入结构
_import_structure = {
    "configuration_efficientnet": [
        "EFFICIENTNET_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "EfficientNetConfig",
        "EfficientNetOnnxConfig",
    ]
}

# 如果视觉可用，则导入图片处理模块的结构
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["image_processing_efficientnet"] = ["EfficientNetImageProcessor"]

# 如果 torch 可用，则导入模型处理模块的结构
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_efficientnet"] = [
        "EFFICIENTNET_PRETRAINED_MODEL_ARCHIVE_LIST",
        "EfficientNetForImageClassification",
        "EfficientNetModel",
        "EfficientNetPreTrainedModel",
    ]

# 类型检查时导入
if TYPE_CHECKING:
    from .configuration_efficientnet import (
        EFFICIENTNET_PRETRAINED_CONFIG_ARCHIVE_MAP,
        EfficientNetConfig,
        EfficientNetOnnxConfig,
    )

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_efficientnet import EfficientNetImageProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_efficientnet import (
            EFFICIENTNET_PRETRAINED_MODEL_ARCHIVE_LIST,
            EfficientNetForImageClassification,
            EfficientNetModel,
            EfficientNetPreTrainedModel,
        )

# 非类型检查时导入
else:
    import sys

    # 将当前模块设置为懒加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```