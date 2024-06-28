# `.\models\roc_bert\__init__.py`

```py
# 版权声明和许可信息
#
# 根据 Apache 许可证版本 2.0（“许可证”）授权；
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件，
# 没有任何明示或暗示的保证或条件。
# 有关许可证详细信息，请参阅许可证。
from typing import TYPE_CHECKING

# 从 utils 中导入一些依赖项
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available, is_torch_available

# 定义导入结构
_import_structure = {
    "configuration_roc_bert": ["ROC_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "RoCBertConfig"],
    "tokenization_roc_bert": ["RoCBertTokenizer"],
}

# 检查是否可用 tokenizers，如果不可用则抛出异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    pass  # 如果可用则继续

# 检查是否可用 torch，如果不可用则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加 modeling_roc_bert 到导入结构中
    _import_structure["modeling_roc_bert"] = [
        "ROC_BERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "RoCBertForCausalLM",
        "RoCBertForMaskedLM",
        "RoCBertForMultipleChoice",
        "RoCBertForPreTraining",
        "RoCBertForQuestionAnswering",
        "RoCBertForSequenceClassification",
        "RoCBertForTokenClassification",
        "RoCBertLayer",
        "RoCBertModel",
        "RoCBertPreTrainedModel",
        "load_tf_weights_in_roc_bert",
    ]

# 如果是类型检查阶段
if TYPE_CHECKING:
    # 从相应模块导入所需符号
    from .configuration_roc_bert import ROC_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, RoCBertConfig
    from .tokenization_roc_bert import RoCBertTokenizer

    # 检查 tokenizers 是否可用，如果可用则抛出异常
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        raise OptionalDependencyNotAvailable()  # 如果可用则引发异常

    # 检查 torch 是否可用，如果可用则从 modeling_roc_bert 导入符号
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_roc_bert import (
            ROC_BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            RoCBertForCausalLM,
            RoCBertForMaskedLM,
            RoCBertForMultipleChoice,
            RoCBertForPreTraining,
            RoCBertForQuestionAnswering,
            RoCBertForSequenceClassification,
            RoCBertForTokenClassification,
            RoCBertLayer,
            RoCBertModel,
            RoCBertPreTrainedModel,
            load_tf_weights_in_roc_bert,
        )

# 如果不是类型检查阶段，则导入 _LazyModule 并设置当前模块的属性为 LazyModule
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```