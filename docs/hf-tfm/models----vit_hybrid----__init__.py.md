# `.\transformers\models\vit_hybrid\__init__.py`

```py
# 这些是 Hugging Face 团队版权声明和许可信息
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

# 导入必要的类型提示
from typing import TYPE_CHECKING

# 从同目录下导入一些工具函数和判断是否有可选依赖的函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义 _import_structure 字典，用于延迟导入需要的模块
_import_structure = {"configuration_vit_hybrid": ["VIT_HYBRID_PRETRAINED_CONFIG_ARCHIVE_MAP", "ViTHybridConfig"]}

# 检查 torch 是否可用，如果不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 torch 可用，则添加相关的 modeling_vit_hybrid 模块到 _import_structure 字典
    _import_structure["modeling_vit_hybrid"] = [
        "VIT_HYBRID_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ViTHybridForImageClassification",
        "ViTHybridModel",
        "ViTHybridPreTrainedModel",
    ]

# 检查 vision 是否可用，如果不可用则引发 OptionalDependencyNotAvailable 异常    
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 vision 可用，则添加 image_processing_vit_hybrid 模块到 _import_structure 字典
    _import_structure["image_processing_vit_hybrid"] = ["ViTHybridImageProcessor"]

# 如果是类型检查，则导入相关的类型
if TYPE_CHECKING:
    from .configuration_vit_hybrid import VIT_HYBRID_PRETRAINED_CONFIG_ARCHIVE_MAP, ViTHybridConfig

    # 如果 torch 可用，则导入 modeling_vit_hybrid 中的类型
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_vit_hybrid import (
            VIT_HYBRID_PRETRAINED_MODEL_ARCHIVE_LIST,
            ViTHybridForImageClassification,
            ViTHybridModel,
            ViTHybridPreTrainedModel,
        )

    # 如果 vision 可用，则导入 image_processing_vit_hybrid 中的类型
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_vit_hybrid import ViTHybridImageProcessor

# 如果不是类型检查，则使用 _LazyModule 延迟导入需要的模块
else:
    import sys
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```