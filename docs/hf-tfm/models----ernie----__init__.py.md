# `.\models\ernie\__init__.py`

```py
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

# 从相对路径导入所需模块和函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tensorflow_text_available, is_torch_available

# 定义预期的模块导入结构
_import_structure = {
    "configuration_ernie": ["ERNIE_PRETRAINED_CONFIG_ARCHIVE_MAP", "ErnieConfig", "ErnieOnnxConfig"],
}

# 尝试导入 torch，如果不可用则引发异常并捕获
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 torch 可用，则添加下列模块到导入结构
    _import_structure["modeling_ernie"] = [
        "ERNIE_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ErnieForCausalLM",
        "ErnieForMaskedLM",
        "ErnieForMultipleChoice",
        "ErnieForNextSentencePrediction",
        "ErnieForPreTraining",
        "ErnieForQuestionAnswering",
        "ErnieForSequenceClassification",
        "ErnieForTokenClassification",
        "ErnieModel",
        "ErniePreTrainedModel",
    ]

# 如果是类型检查模式，导入配置和模型相关内容
if TYPE_CHECKING:
    from .configuration_ernie import ERNIE_PRETRAINED_CONFIG_ARCHIVE_MAP, ErnieConfig, ErnieOnnxConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_ernie import (
            ERNIE_PRETRAINED_MODEL_ARCHIVE_LIST,
            ErnieForCausalLM,
            ErnieForMaskedLM,
            ErnieForMultipleChoice,
            ErnieForNextSentencePrediction,
            ErnieForPreTraining,
            ErnieForQuestionAnswering,
            ErnieForSequenceClassification,
            ErnieForTokenClassification,
            ErnieModel,
            ErniePreTrainedModel,
        )

# 如果不是类型检查模式，则设置 LazyModule 来处理模块的惰性加载
else:
    import sys

    # 将当前模块设置为 LazyModule 的实例，以处理按需导入模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```