# `.\models\unispeech\__init__.py`

```
# Copyright 2021 The HuggingFace Team. All rights reserved.
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

# 从 typing 模块导入 TYPE_CHECKING
from typing import TYPE_CHECKING

# 从 ...utils 中导入必要的类和函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_torch_available,
)

# 定义导入结构
_import_structure = {"configuration_unispeech": ["UNISPEECH_PRETRAINED_CONFIG_ARCHIVE_MAP", "UniSpeechConfig"]}

# 检查是否为静态类型检查
try:
    # 如果 Torch 不可用则抛出 OptionalDependencyNotAvailable 异常
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 可用，则添加相关模型到 _import_structure 中
    _import_structure["modeling_unispeech"] = [
        "UNISPEECH_PRETRAINED_MODEL_ARCHIVE_LIST",
        "UniSpeechForCTC",
        "UniSpeechForPreTraining",
        "UniSpeechForSequenceClassification",
        "UniSpeechModel",
        "UniSpeechPreTrainedModel",
    ]

# 如果在静态类型检查环境中
if TYPE_CHECKING:
    # 从 .configuration_unispeech 中导入必要的类和函数
    from .configuration_unispeech import UNISPEECH_PRETRAINED_CONFIG_ARCHIVE_MAP, UniSpeechConfig

    try:
        # 如果 Torch 不可用则抛出 OptionalDependencyNotAvailable 异常
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 从 .modeling_unispeech 中导入必要的类和函数
        from .modeling_unispeech import (
            UNISPEECH_PRETRAINED_MODEL_ARCHIVE_LIST,
            UniSpeechForCTC,
            UniSpeechForPreTraining,
            UniSpeechForSequenceClassification,
            UniSpeechModel,
            UniSpeechPreTrainedModel,
        )

# 如果不在静态类型检查环境中
else:
    import sys

    # 将当前模块设为懒加载模块 _LazyModule 的实例
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```