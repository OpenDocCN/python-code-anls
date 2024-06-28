# `.\models\owlvit\__init__.py`

```
# 版权声明和许可证信息，指明代码版权归 HuggingFace Team 所有，使用 Apache License 2.0 许可
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

# 导入类型检查工具
from typing import TYPE_CHECKING

# 导入依赖项检查函数和懒加载模块
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
    is_vision_available,
)

# 定义模块的导入结构
_import_structure = {
    "configuration_owlvit": [
        "OWLVIT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "OwlViTConfig",
        "OwlViTOnnxConfig",
        "OwlViTTextConfig",
        "OwlViTVisionConfig",
    ],
    "processing_owlvit": ["OwlViTProcessor"],
}

# 检查是否存在视觉处理库，若不存在则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若存在视觉处理库，则添加视觉特征提取和图像处理到导入结构
    _import_structure["feature_extraction_owlvit"] = ["OwlViTFeatureExtractor"]
    _import_structure["image_processing_owlvit"] = ["OwlViTImageProcessor"]

# 检查是否存在 Torch 库，若不存在则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若存在 Torch 库，则添加模型相关的导入到导入结构
    _import_structure["modeling_owlvit"] = [
        "OWLVIT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "OwlViTModel",
        "OwlViTPreTrainedModel",
        "OwlViTTextModel",
        "OwlViTVisionModel",
        "OwlViTForObjectDetection",
    ]

# 如果是类型检查阶段，则导入配置和处理模块
if TYPE_CHECKING:
    from .configuration_owlvit import (
        OWLVIT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        OwlViTConfig,
        OwlViTOnnxConfig,
        OwlViTTextConfig,
        OwlViTVisionConfig,
    )
    from .processing_owlvit import OwlViTProcessor

    # 在类型检查阶段，若存在视觉处理库，则导入视觉特征提取和图像处理模块
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .feature_extraction_owlvit import OwlViTFeatureExtractor
        from .image_processing_owlvit import OwlViTImageProcessor

    # 在类型检查阶段，若存在 Torch 库，则导入模型相关的模块
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_owlvit import (
            OWLVIT_PRETRAINED_MODEL_ARCHIVE_LIST,
            OwlViTForObjectDetection,
            OwlViTModel,
            OwlViTPreTrainedModel,
            OwlViTTextModel,
            OwlViTVisionModel,
        )

# 如果不是类型检查阶段，则将当前模块替换为懒加载模块以支持动态导入
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```