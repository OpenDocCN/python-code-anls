# `.\models\longt5\__init__.py`

```
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

from typing import TYPE_CHECKING

# 从 utils 中导入自定义的异常和模块，用于处理可选的依赖不可用情况
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_flax_available, is_torch_available

# 定义模块的导入结构，列出了各个模块和类的导入信息
_import_structure = {
    "configuration_longt5": ["LONGT5_PRETRAINED_CONFIG_ARCHIVE_MAP", "LongT5Config", "LongT5OnnxConfig"],
}

# 尝试导入 Torch 版本的 LongT5 模型和相关配置
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_longt5"] = [
        "LONGT5_PRETRAINED_MODEL_ARCHIVE_LIST",
        "LongT5EncoderModel",
        "LongT5ForConditionalGeneration",
        "LongT5Model",
        "LongT5PreTrainedModel",
    ]

# 尝试导入 Flax 版本的 LongT5 模型和相关配置
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_longt5"] = [
        "FlaxLongT5ForConditionalGeneration",
        "FlaxLongT5Model",
        "FlaxLongT5PreTrainedModel",
    ]

# 如果是在类型检查阶段，导入详细的类型信息
if TYPE_CHECKING:
    from .configuration_longt5 import LONGT5_PRETRAINED_CONFIG_ARCHIVE_MAP, LongT5Config, LongT5OnnxConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_longt5 import (
            LONGT5_PRETRAINED_MODEL_ARCHIVE_LIST,
            LongT5EncoderModel,
            LongT5ForConditionalGeneration,
            LongT5Model,
            LongT5PreTrainedModel,
        )

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_flax_longt5 import (
            FlaxLongT5ForConditionalGeneration,
            FlaxLongT5Model,
            FlaxLongT5PreTrainedModel,
        )

# 如果不是在类型检查阶段，则将当前模块设置为懒加载模块，以延迟导入各个模块
else:
    import sys

    # 使用 LazyModule 将当前模块设置为懒加载模块，根据需要导入相关模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```