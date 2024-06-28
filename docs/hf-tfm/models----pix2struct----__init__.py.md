# `.\models\pix2struct\__init__.py`

```
# 版权声明及许可证声明，声明代码版权及使用许可
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

# 引入类型检查标志 TYPE_CHECKING
from typing import TYPE_CHECKING

# 从 utils 模块中引入相关工具和检查 Torch 和 Vision 是否可用的函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义模块导入结构字典 _import_structure
_import_structure = {
    "configuration_pix2struct": [
        "PIX2STRUCT_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置文件映射
        "Pix2StructConfig",  # Pix2Struct 模型配置
        "Pix2StructTextConfig",  # 文本 Pix2Struct 模型配置
        "Pix2StructVisionConfig",  # 视觉 Pix2Struct 模型配置
    ],
    "processing_pix2struct": ["Pix2StructProcessor"],  # Pix2Struct 数据处理器
}

# 检查 Vision 是否可用，若不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若 Vision 可用，则将 Pix2Struct 图像处理器添加到 _import_structure 中
    _import_structure["image_processing_pix2struct"] = ["Pix2StructImageProcessor"]

# 检查 Torch 是否可用，若不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若 Torch 可用，则将 Pix2Struct 模型相关内容添加到 _import_structure 中
    _import_structure["modeling_pix2struct"] = [
        "PIX2STRUCT_PRETRAINED_MODEL_ARCHIVE_LIST",  # 预训练模型存档列表
        "Pix2StructPreTrainedModel",  # Pix2Struct 预训练模型基类
        "Pix2StructForConditionalGeneration",  # 条件生成 Pix2Struct 模型
        "Pix2StructVisionModel",  # 视觉 Pix2Struct 模型
        "Pix2StructTextModel",  # 文本 Pix2Struct 模型
    ]

# 如果是类型检查阶段，导入特定模块和类
if TYPE_CHECKING:
    from .configuration_pix2struct import (
        PIX2STRUCT_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 预训练配置文件映射
        Pix2StructConfig,  # Pix2Struct 模型配置
        Pix2StructTextConfig,  # 文本 Pix2Struct 模型配置
        Pix2StructVisionConfig,  # 视觉 Pix2Struct 模型配置
    )
    from .processing_pix2struct import Pix2StructProcessor  # Pix2Struct 数据处理器

    # 检查 Vision 是否可用，若可用则导入 Pix2Struct 图像处理器
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_pix2struct import Pix2StructImageProcessor  # Pix2Struct 图像处理器

    # 检查 Torch 是否可用，若可用则导入 Pix2Struct 模型相关内容
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_pix2struct import (
            PIX2STRUCT_PRETRAINED_MODEL_ARCHIVE_LIST,  # 预训练模型存档列表
            Pix2StructForConditionalGeneration,  # 条件生成 Pix2Struct 模型
            Pix2StructPreTrainedModel,  # Pix2Struct 预训练模型基类
            Pix2StructTextModel,  # 文本 Pix2Struct 模型
            Pix2StructVisionModel,  # 视觉 Pix2Struct 模型
        )

# 如果不是类型检查阶段，则将当前模块替换为延迟加载模块 _LazyModule
else:
    import sys

    # 使用 _LazyModule 代替当前模块，传入模块名、文件名、导入结构和模块规范
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```