# `.\models\tapas\__init__.py`

```py
# Copyright 2020 The HuggingFace Team. All rights reserved.
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

# 从 ...utils 中导入 OptionalDependencyNotAvailable、_LazyModule、is_tf_available 和 is_torch_available
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tf_available, is_torch_available

# 定义一个字典 _import_structure，用于存储不同模块的导入结构
_import_structure = {
    "configuration_tapas": ["TAPAS_PRETRAINED_CONFIG_ARCHIVE_MAP", "TapasConfig"],
    "tokenization_tapas": ["TapasTokenizer"],
}

# 尝试导入 torch 版本的 Tapas 模块，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则将相关模块添加到 _import_structure 中
    _import_structure["modeling_tapas"] = [
        "TAPAS_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TapasForMaskedLM",
        "TapasForQuestionAnswering",
        "TapasForSequenceClassification",
        "TapasModel",
        "TapasPreTrainedModel",
        "load_tf_weights_in_tapas",
    ]

# 尝试导入 TensorFlow 版本的 Tapas 模块，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则将相关模块添加到 _import_structure 中
    _import_structure["modeling_tf_tapas"] = [
        "TF_TAPAS_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFTapasForMaskedLM",
        "TFTapasForQuestionAnswering",
        "TFTapasForSequenceClassification",
        "TFTapasModel",
        "TFTapasPreTrainedModel",
    ]

# 如果是类型检查阶段，导入具体的类型和模块
if TYPE_CHECKING:
    from .configuration_tapas import TAPAS_PRETRAINED_CONFIG_ARCHIVE_MAP, TapasConfig
    from .tokenization_tapas import TapasTokenizer

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tapas import (
            TAPAS_PRETRAINED_MODEL_ARCHIVE_LIST,
            TapasForMaskedLM,
            TapasForQuestionAnswering,
            TapasForSequenceClassification,
            TapasModel,
            TapasPreTrainedModel,
            load_tf_weights_in_tapas,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_tapas import (
            TF_TAPAS_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFTapasForMaskedLM,
            TFTapasForQuestionAnswering,
            TFTapasForSequenceClassification,
            TFTapasModel,
            TFTapasPreTrainedModel,
        )

# 如果不是类型检查阶段，则动态加载模块，并将当前模块替换为 _LazyModule 的实例
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```