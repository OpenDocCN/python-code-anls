# `.\models\pvt\__init__.py`

```py
# coding=utf-8
# 以上行指定了源代码文件的字符编码格式为 UTF-8

# 版权声明和作者信息，标识代码版权归属和作者列表
# Copyright 2023 Authors: Wenhai Wang, Enze Xie, Xiang Li, Deng-Ping Fan,
# Kaitao Song, Ding Liang, Tong Lu, Ping Luo, Ling Shao and The HuggingFace Inc. team.
# All rights reserved.
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

# 从 typing 模块中导入 TYPE_CHECKING 类型标记
from typing import TYPE_CHECKING

# 从 ...utils 中导入所需的模块和函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
    is_vision_available,
)

# 定义模块导入结构的字典，包含 configuration_pvt 模块的导入结构
_import_structure = {
    "configuration_pvt": ["PVT_PRETRAINED_CONFIG_ARCHIVE_MAP", "PvtConfig", "PvtOnnxConfig"],
}

# 检查视觉处理是否可用，若不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，将 image_processing_pvt 模块导入结构添加到 _import_structure 字典中
    _import_structure["image_processing_pvt"] = ["PvtImageProcessor"]

# 检查 Torch 是否可用，若不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，将 modeling_pvt 模块导入结构添加到 _import_structure 字典中
    _import_structure["modeling_pvt"] = [
        "PVT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "PvtForImageClassification",
        "PvtModel",
        "PvtPreTrainedModel",
    ]

# 如果当前环境为 TYPE_CHECKING，进行类型导入
if TYPE_CHECKING:
    # 从 configuration_pvt 模块导入指定的类和变量
    from .configuration_pvt import PVT_PRETRAINED_CONFIG_ARCHIVE_MAP, PvtConfig, PvtOnnxConfig

    # 在视觉处理可用时，从 image_processing_pvt 模块导入 PvtImageProcessor 类
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_pvt import PvtImageProcessor

    # 在 Torch 可用时，从 modeling_pvt 模块导入指定的类和变量
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_pvt import (
            PVT_PRETRAINED_MODEL_ARCHIVE_LIST,
            PvtForImageClassification,
            PvtModel,
            PvtPreTrainedModel,
        )

# 如果不是 TYPE_CHECKING 环境，则将当前模块注册为 LazyModule
else:
    import sys

    # 使用 _LazyModule 将当前模块设置为惰性加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```