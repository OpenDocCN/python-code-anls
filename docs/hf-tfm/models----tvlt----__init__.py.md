# `.\models\tvlt\__init__.py`

```
# flake8: noqa
# 禁用 flake8 检查此模块，因为无法忽略 "F401 '...' imported but unused" 警告，同时保留其他警告。

# Copyright 2023 The HuggingFace Team. All rights reserved.
# 版权声明，保留所有权利。

# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache 许可证版本 2.0 许可；您不得使用此文件，除非遵守许可证。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# 除非适用法律要求或书面同意，否则软件按"原样"提供，无任何明示或暗示的担保或条件。
# 请参阅许可证获取特定语言的权限和限制。
from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
    is_vision_available,
)

# _import_structure 定义了模块的导入结构，包含不同模块及其导出的符号列表
_import_structure = {
    "configuration_tvlt": ["TVLT_PRETRAINED_CONFIG_ARCHIVE_MAP", "TvltConfig"],
    "feature_extraction_tvlt": ["TvltFeatureExtractor"],
    "processing_tvlt": ["TvltProcessor"],
}

# 检查是否 Torch 可用，若不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 可用，则导入 modeling_tvlt 模块的符号列表
    _import_structure["modeling_tvlt"] = [
        "TVLT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TvltModel",
        "TvltForPreTraining",
        "TvltForAudioVisualClassification",
        "TvltPreTrainedModel",
    ]

# 检查是否 Vision 可用，若不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Vision 可用，则导入 image_processing_tvlt 模块的符号列表
    _import_structure["image_processing_tvlt"] = ["TvltImageProcessor"]

# 如果 TYPE_CHECKING 为真，导入各个模块的具体符号以供类型检查使用
if TYPE_CHECKING:
    from .configuration_tvlt import TVLT_PRETRAINED_CONFIG_ARCHIVE_MAP, TvltConfig
    from .processing_tvlt import TvltProcessor
    from .feature_extraction_tvlt import TvltFeatureExtractor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tvlt import (
            TVLT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TvltForAudioVisualClassification,
            TvltForPreTraining,
            TvltModel,
            TvltPreTrainedModel,
        )

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_tvlt import TvltImageProcessor

# 如果不是 TYPE_CHECKING 环境，则将当前模块设置为一个懒加载模块 _LazyModule
else:
    import sys

    # 使用 _LazyModule 将当前模块动态注册为延迟加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```