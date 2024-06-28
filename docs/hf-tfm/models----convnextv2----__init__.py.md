# `.\models\convnextv2\__init__.py`

```py
# flake8: noqa
# 无法在本模块中忽略 "F401 '...' imported but unused" 警告，以保留其他警告。因此，完全不检查本模块。

# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# 根据 Apache 许可证版本 2.0（“许可证”）进行许可；
# 除非符合许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件根据“原样”分发，
# 没有任何明示或暗示的保证或条件。
# 有关特定语言的权限，请参阅许可证。

from typing import TYPE_CHECKING

# 依赖于 isort 来合并导入项
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
    is_tf_available,
)

# 定义导入结构
_import_structure = {
    "configuration_convnextv2": [
        "CONVNEXTV2_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "ConvNextV2Config",
    ]
}

try:
    # 如果没有 Torch 可用，则引发 OptionalDependencyNotAvailable 异常
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 可用，则添加以下模型定义到导入结构中
    _import_structure["modeling_convnextv2"] = [
        "CONVNEXTV2_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ConvNextV2ForImageClassification",
        "ConvNextV2Model",
        "ConvNextV2PreTrainedModel",
        "ConvNextV2Backbone",
    ]

try:
    # 如果没有 TensorFlow 可用，则引发 OptionalDependencyNotAvailable 异常
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 TensorFlow 可用，则添加以下 TensorFlow 模型定义到导入结构中
    _import_structure["modeling_tf_convnextv2"] = [
        "TFConvNextV2ForImageClassification",
        "TFConvNextV2Model",
        "TFConvNextV2PreTrainedModel",
    ]

if TYPE_CHECKING:
    # 如果是类型检查阶段，则导入以下类型相关的定义
    from .configuration_convnextv2 import (
        CONVNEXTV2_PRETRAINED_CONFIG_ARCHIVE_MAP,
        ConvNextV2Config,
    )

    try:
        # 如果没有 Torch 可用，则引发 OptionalDependencyNotAvailable 异常
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 Torch 可用，则导入以下 Torch 模型相关的定义
        from .modeling_convnextv2 import (
            CONVNEXTV2_PRETRAINED_MODEL_ARCHIVE_LIST,
            ConvNextV2Backbone,
            ConvNextV2ForImageClassification,
            ConvNextV2Model,
            ConvNextV2PreTrainedModel,
        )

    try:
        # 如果没有 TensorFlow 可用，则引发 OptionalDependencyNotAvailable 异常
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 TensorFlow 可用，则导入以下 TensorFlow 模型相关的定义
        from .modeling_tf_convnextv2 import (
            TFConvNextV2ForImageClassification,
            TFConvNextV2Model,
            TFConvNextV2PreTrainedModel,
        )

else:
    # 如果不是类型检查阶段，则创建懒加载模块
    import sys

    # 使用懒加载模块将当前模块注册到 sys.modules 中
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```