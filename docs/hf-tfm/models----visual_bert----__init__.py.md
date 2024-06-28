# `.\models\visual_bert\__init__.py`

```py
# 版权声明和许可证信息
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

# 导入必要的类型检查模块
from typing import TYPE_CHECKING

# 导入自定义的异常和模块延迟加载函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义导入结构字典，包含导入的模块和变量列表
_import_structure = {"configuration_visual_bert": ["VISUAL_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "VisualBertConfig"]}

# 尝试检查是否可用 Torch 库，若不可用则引发自定义异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 可用，则添加模型相关的导入结构
    _import_structure["modeling_visual_bert"] = [
        "VISUAL_BERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "VisualBertForMultipleChoice",
        "VisualBertForPreTraining",
        "VisualBertForQuestionAnswering",
        "VisualBertForRegionToPhraseAlignment",
        "VisualBertForVisualReasoning",
        "VisualBertLayer",
        "VisualBertModel",
        "VisualBertPreTrainedModel",
    ]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 导入配置相关的类和变量
    from .configuration_visual_bert import VISUAL_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, VisualBertConfig

    # 尝试检查是否可用 Torch 库，若不可用则忽略
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入模型相关的类和变量
        from .modeling_visual_bert import (
            VISUAL_BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            VisualBertForMultipleChoice,
            VisualBertForPreTraining,
            VisualBertForQuestionAnswering,
            VisualBertForRegionToPhraseAlignment,
            VisualBertForVisualReasoning,
            VisualBertLayer,
            VisualBertModel,
            VisualBertPreTrainedModel,
        )

# 如果不是类型检查模式，则配置 LazyModule 并添加到当前模块中
else:
    import sys

    # 使用 LazyModule 来延迟加载模块的定义
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```