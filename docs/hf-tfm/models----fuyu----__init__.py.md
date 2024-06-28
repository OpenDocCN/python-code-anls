# `.\models\fuyu\__init__.py`

```py
# 版权声明和版权许可声明，指明版权归属和使用许可
# Copyright 2023 AdeptAI and The HuggingFace Inc. team. All rights reserved.
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

# 引入类型检查模块中的TYPE_CHECKING类型
from typing import TYPE_CHECKING

# 引入相关的自定义工具函数和类
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

# 定义模块的导入结构，初始化空字典
_import_structure = {
    "configuration_fuyu": ["FUYU_PRETRAINED_CONFIG_ARCHIVE_MAP", "FuyuConfig"],
}

# 检查视觉处理是否可用，若不可用则引发OptionalDependencyNotAvailable异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则将视觉处理相关的模块添加到导入结构中
    _import_structure["image_processing_fuyu"] = ["FuyuImageProcessor"]
    _import_structure["processing_fuyu"] = ["FuyuProcessor"]

# 检查是否Torch可用，若不可用则引发OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则将模型相关的模块添加到导入结构中
    _import_structure["modeling_fuyu"] = [
        "FuyuForCausalLM",
        "FuyuPreTrainedModel",
    ]

# 如果是类型检查阶段
if TYPE_CHECKING:
    # 从配置模块导入相关的类和常量
    from .configuration_fuyu import FUYU_PRETRAINED_CONFIG_ARCHIVE_MAP, FuyuConfig

    # 检查视觉处理是否可用，若不可用则pass
    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若可用，则从图像处理模块导入相关类
        from .image_processing_fuyu import FuyuImageProcessor
        from .processing_fuyu import FuyuProcessor

    # 检查Torch是否可用，若不可用则pass
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若可用，则从建模模块导入相关类
        from .modeling_fuyu import (
            FuyuForCausalLM,
            FuyuPreTrainedModel,
        )

# 如果不是类型检查阶段
else:
    import sys

    # 将当前模块注册为LazyModule，使用LazyModule来延迟加载模块内容
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```