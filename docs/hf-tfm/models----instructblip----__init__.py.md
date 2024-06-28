# `.\models\instructblip\__init__.py`

```py
# 版权声明和版权信息
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

# 导入类型检查模块
from typing import TYPE_CHECKING

# 导入自定义的异常类和模块加载延迟处理类
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块的导入结构
_import_structure = {
    "configuration_instructblip": [
        "INSTRUCTBLIP_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "InstructBlipConfig",
        "InstructBlipQFormerConfig",
        "InstructBlipVisionConfig",
    ],
    "processing_instructblip": ["InstructBlipProcessor"],
}

# 检查是否可以导入 torch，如果不可以，则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可以导入 torch，则添加以下模块到导入结构中
    _import_structure["modeling_instructblip"] = [
        "INSTRUCTBLIP_PRETRAINED_MODEL_ARCHIVE_LIST",
        "InstructBlipQFormerModel",
        "InstructBlipPreTrainedModel",
        "InstructBlipForConditionalGeneration",
        "InstructBlipVisionModel",
    ]

# 如果是类型检查阶段，进行详细的导入
if TYPE_CHECKING:
    # 导入配置相关的类和变量
    from .configuration_instructblip import (
        INSTRUCTBLIP_PRETRAINED_CONFIG_ARCHIVE_MAP,
        InstructBlipConfig,
        InstructBlipQFormerConfig,
        InstructBlipVisionConfig,
    )
    # 导入处理相关的类
    from .processing_instructblip import InstructBlipProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入模型相关的类和变量
        from .modeling_instructblip import (
            INSTRUCTBLIP_PRETRAINED_MODEL_ARCHIVE_LIST,
            InstructBlipForConditionalGeneration,
            InstructBlipPreTrainedModel,
            InstructBlipQFormerModel,
            InstructBlipVisionModel,
        )

# 如果不是类型检查阶段，将模块指定给自定义的 LazyModule 类
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```