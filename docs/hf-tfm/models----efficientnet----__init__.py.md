# `.\models\efficientnet\__init__.py`

```
# flake8: noqa
# 在本模块中无法忽略“F401 '...' imported but unused”警告，但需要保留其他警告。因此，完全禁用对本模块的检查。

# 版权 2023 年 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）许可；
# 除非符合许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件，
# 没有任何形式的明示或暗示的保证或条件。
# 有关特定语言的权限，请参阅许可证。

from typing import TYPE_CHECKING

# 使用 isort 来合并导入
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义模块导入结构
_import_structure = {
    "configuration_efficientnet": [
        "EFFICIENTNET_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "EfficientNetConfig",
        "EfficientNetOnnxConfig",
    ]
}

# 如果视觉处理可用，导入图像处理的 EfficientNet
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["image_processing_efficientnet"] = ["EfficientNetImageProcessor"]

# 如果 Torch 可用，导入 EfficientNet 的模型处理
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

# 如果类型检查开启，导入必要的配置和模型类
if TYPE_CHECKING:
    from .configuration_efficientnet import (
        EFFICIENTNET_PRETRAINED_CONFIG_ARCHIVE_MAP,
        EfficientNetConfig,
        EfficientNetOnnxConfig,
    )

    # 如果视觉处理可用，导入图像处理的 EfficientNet
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_efficientnet import EfficientNetImageProcessor

    # 如果 Torch 可用，导入 EfficientNet 的模型处理
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

# 如果类型检查未开启，使用 LazyModule 封装模块的导入
else:
    import sys

    # 将当前模块替换为 LazyModule 对象，用于延迟导入
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```