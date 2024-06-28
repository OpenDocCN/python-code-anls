# `.\models\deformable_detr\__init__.py`

```py
# 版权声明和许可证信息，指明该代码的版权归属和许可证条款
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

# 导入类型检查模块中的TYPE_CHECKING
from typing import TYPE_CHECKING

# 导入必要的依赖
# 导入自定义工具函数和类
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义模块的导入结构字典
_import_structure = {
    "configuration_deformable_detr": ["DEFORMABLE_DETR_PRETRAINED_CONFIG_ARCHIVE_MAP", "DeformableDetrConfig"],
}

# 检查视觉库是否可用，若不可用则引发OptionalDependencyNotAvailable异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则添加以下导入到_import_structure字典
    _import_structure["feature_extraction_deformable_detr"] = ["DeformableDetrFeatureExtractor"]
    _import_structure["image_processing_deformable_detr"] = ["DeformableDetrImageProcessor"]

# 检查Torch库是否可用，若不可用则引发OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则添加以下导入到_import_structure字典
    _import_structure["modeling_deformable_detr"] = [
        "DEFORMABLE_DETR_PRETRAINED_MODEL_ARCHIVE_LIST",
        "DeformableDetrForObjectDetection",
        "DeformableDetrModel",
        "DeformableDetrPreTrainedModel",
    ]

# 如果正在进行类型检查
if TYPE_CHECKING:
    # 从configuration_deformable_detr模块导入特定符号
    from .configuration_deformable_detr import DEFORMABLE_DETR_PRETRAINED_CONFIG_ARCHIVE_MAP, DeformableDetrConfig

    # 检查视觉库是否可用，若不可用则忽略导入
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若可用，则从feature_extraction_deformable_detr和image_processing_deformable_detr模块导入特定符号
        from .feature_extraction_deformable_detr import DeformableDetrFeatureExtractor
        from .image_processing_deformable_detr import DeformableDetrImageProcessor

    # 检查Torch库是否可用，若不可用则忽略导入
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若可用，则从modeling_deformable_detr模块导入特定符号
        from .modeling_deformable_detr import (
            DEFORMABLE_DETR_PRETRAINED_MODEL_ARCHIVE_LIST,
            DeformableDetrForObjectDetection,
            DeformableDetrModel,
            DeformableDetrPreTrainedModel,
        )

# 如果不是类型检查模式，则为当前模块创建一个延迟加载模块
else:
    import sys

    # 使用_LazyModule将当前模块的导入结构暴露给sys.modules
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```