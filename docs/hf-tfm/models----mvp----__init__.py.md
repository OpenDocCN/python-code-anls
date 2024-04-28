# `.\transformers\models\mvp\__init__.py`

```
# 这是一个 HuggingFace 团队的版权文件，包含了一些导入和类型检查的代码
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

# 从 utils 模块导入一些依赖检查的函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available, is_torch_available


# 定义了一个导入结构字典，包含了一些子模块
_import_structure = {
    "configuration_mvp": ["MVP_PRETRAINED_CONFIG_ARCHIVE_MAP", "MvpConfig", "MvpOnnxConfig"],
    "tokenization_mvp": ["MvpTokenizer"],
}

# 检查是否有 tokenizers 依赖，如果没有则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果有 tokenizers 依赖，则将 tokenization_mvp_fast 模块添加到导入结构中
    _import_structure["tokenization_mvp_fast"] = ["MvpTokenizerFast"]

# 检查是否有 torch 依赖，如果没有则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果有 torch 依赖，则将 modeling_mvp 模块添加到导入结构中
    _import_structure["modeling_mvp"] = [
        "MVP_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MvpForCausalLM",
        "MvpForConditionalGeneration",
        "MvpForQuestionAnswering",
        "MvpForSequenceClassification",
        "MvpModel",
        "MvpPreTrainedModel",
    ]

# 如果是类型检查的环境，则导入相应的类
if TYPE_CHECKING:
    from .configuration_mvp import MVP_PRETRAINED_CONFIG_ARCHIVE_MAP, MvpConfig, MvpOnnxConfig
    from .tokenization_mvp import MvpTokenizer

    # 检查是否有 tokenizers 依赖，如果有则导入 MvpTokenizerFast
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_mvp_fast import MvpTokenizerFast

    # 检查是否有 torch 依赖，如果有则导入 modeling_mvp 中的类
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_mvp import (
            MVP_PRETRAINED_MODEL_ARCHIVE_LIST,
            MvpForCausalLM,
            MvpForConditionalGeneration,
            MvpForQuestionAnswering,
            MvpForSequenceClassification,
            MvpModel,
            MvpPreTrainedModel,
        )

# 如果不是类型检查的环境，则使用懒加载模块
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```