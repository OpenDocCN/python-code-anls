# `.\models\qdqbert\__init__.py`

```py
# 版权声明和许可信息，指出代码的版权和许可细则
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

# 导入类型检查模块的标记
from typing import TYPE_CHECKING

# 导入自定义的异常和模块懒加载工具函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块的导入结构
_import_structure = {"configuration_qdqbert": ["QDQBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "QDQBertConfig"]}

# 检查是否有torch可用，若不可用则抛出OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若torch可用，则更新模块导入结构
    _import_structure["modeling_qdqbert"] = [
        "QDQBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "QDQBertForMaskedLM",
        "QDQBertForMultipleChoice",
        "QDQBertForNextSentencePrediction",
        "QDQBertForQuestionAnswering",
        "QDQBertForSequenceClassification",
        "QDQBertForTokenClassification",
        "QDQBertLayer",
        "QDQBertLMHeadModel",
        "QDQBertModel",
        "QDQBertPreTrainedModel",
        "load_tf_weights_in_qdqbert",
    ]

# 如果当前为类型检查模式，则导入相关类型
if TYPE_CHECKING:
    from .configuration_qdqbert import QDQBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, QDQBertConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_qdqbert import (
            QDQBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            QDQBertForMaskedLM,
            QDQBertForMultipleChoice,
            QDQBertForNextSentencePrediction,
            QDQBertForQuestionAnswering,
            QDQBertForSequenceClassification,
            QDQBertForTokenClassification,
            QDQBertLayer,
            QDQBertLMHeadModel,
            QDQBertModel,
            QDQBertPreTrainedModel,
            load_tf_weights_in_qdqbert,
        )

# 如果不是类型检查模式，则进行模块的延迟加载处理
else:
    import sys

    # 将当前模块替换为懒加载模块的实例，用于延迟导入
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```