# `.\models\maskformer\__init__.py`

```py
# 版权声明和许可信息
#
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入类型检查相关的模块
from typing import TYPE_CHECKING

# 导入相关的工具和模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义导入结构，指定各模块的导入内容
_import_structure = {
    "configuration_maskformer": ["MASKFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "MaskFormerConfig"],
    "configuration_maskformer_swin": ["MaskFormerSwinConfig"],
}

# 检查视觉相关的依赖是否可用，若不可用则抛出OptionalDependencyNotAvailable异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则导入特征提取和图像处理相关模块
    _import_structure["feature_extraction_maskformer"] = ["MaskFormerFeatureExtractor"]
    _import_structure["image_processing_maskformer"] = ["MaskFormerImageProcessor"]

# 检查Torch相关的依赖是否可用，若不可用则抛出OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则导入模型相关的模块
    _import_structure["modeling_maskformer"] = [
        "MASKFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MaskFormerForInstanceSegmentation",
        "MaskFormerModel",
        "MaskFormerPreTrainedModel",
    ]
    _import_structure["modeling_maskformer_swin"] = [
        "MaskFormerSwinBackbone",
        "MaskFormerSwinModel",
        "MaskFormerSwinPreTrainedModel",
    ]

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 导入具体的配置和模型相关内容
    from .configuration_maskformer import MASKFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, MaskFormerConfig
    from .configuration_maskformer_swin import MaskFormerSwinConfig

    try:
        # 再次检查视觉相关依赖是否可用，若不可用则忽略
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若可用，则导入特征提取和图像处理相关模块
        from .feature_extraction_maskformer import MaskFormerFeatureExtractor
        from .image_processing_maskformer import MaskFormerImageProcessor

    try:
        # 再次检查Torch相关依赖是否可用，若不可用则忽略
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若可用，则导入模型相关的模块
        from .modeling_maskformer import (
            MASKFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            MaskFormerForInstanceSegmentation,
            MaskFormerModel,
            MaskFormerPreTrainedModel,
        )
        from .modeling_maskformer_swin import (
            MaskFormerSwinBackbone,
            MaskFormerSwinModel,
            MaskFormerSwinPreTrainedModel,
        )

# 如果不是类型检查模式，则设置模块为LazyModule
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```