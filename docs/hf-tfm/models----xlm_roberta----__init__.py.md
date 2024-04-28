# `.\transformers\models\xlm_roberta\__init__.py`

```
# 引入必要的库
from typing import TYPE_CHECKING
# 引入自定义的异常和函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_sentencepiece_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块的导入结构
_import_structure = {
    "configuration_xlm_roberta": [
        "XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "XLMRobertaConfig",
        "XLMRobertaOnnxConfig",
    ],
}

# 检查是否有 sentencepiece 库可用，如果不可用则引发异常，否则将 Tokenization 模块加入导入结构
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_xlm_roberta"] = ["XLMRobertaTokenizer"]

# 检查是否有 tokenizers 库可用，如果不可用则引发异常，否则将 Tokenization Fast 模块加入导入结构
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_xlm_roberta_fast"] = ["XLMRobertaTokenizerFast"]

# 检查是否有 torch 库可用，如果不可用则引发异常，否则将 Modeling 模块加入导入结构
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_xlm_roberta"] = [
        "XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",
        "XLMRobertaForCausalLM",
        "XLMRobertaForMaskedLM",
        "XLMRobertaForMultipleChoice",
        "XLMRobertaForQuestionAnswering",
        "XLMRobertaForSequenceClassification",
        "XLMRobertaForTokenClassification",
        "XLMRobertaModel",
        "XLMRobertaPreTrainedModel",
    ]

# 检查是否有 tensorflow 库可用，如果不可用则引发异常，否则将 TF Modeling 模块加入导入结构
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_xlm_roberta"] = [
        "TF_XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFXLMRobertaForCausalLM",
        "TFXLMRobertaForMaskedLM",
        "TFXLMRobertaForMultipleChoice",
        "TFXLMRobertaForQuestionAnswering",
        "TFXLMRobertaForSequenceClassification",
        "TFXLMRobertaForTokenClassification",
        "TFXLMRobertaModel",
        "TFXLMRobertaPreTrainedModel",
    ]

# 检查是否有 flax 库可用，如果不可用则引发异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 将模块的结构信息添加到_import_structure字典中，以便后续使用
    _import_structure["modeling_flax_xlm_roberta"] = [
        # FLAX_XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST用于存储预训练模型的存档列表
        "FLAX_XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST",
        # FlaxXLMRobertaForMaskedLM用于掩码语言建模任务的FLAX-XLM-RoBERTa模型
        "FlaxXLMRobertaForMaskedLM",
        # FlaxXLMRobertaForCausalLM用于因果语言建模任务的FLAX-XLM-RoBERTa模型
        "FlaxXLMRobertaForCausalLM",
        # FlaxXLMRobertaForMultipleChoice用于多选题任务的FLAX-XLM-RoBERTa模型
        "FlaxXLMRobertaForMultipleChoice",
        # FlaxXLMRobertaForQuestionAnswering用于问答任务的FLAX-XLM-RoBERTa模型
        "FlaxXLMRobertaForQuestionAnswering",
        # FlaxXLMRobertaForSequenceClassification用于序列分类任务的FLAX-XLM-RoBERTa模型
        "FlaxXLMRobertaForSequenceClassification",
        # FlaxXLMRobertaForTokenClassification用于标记分类任务的FLAX-XLM-RoBERTa模型
        "FlaxXLMRobertaForTokenClassification",
        # FlaxXLMRobertaModel是FLAX-XLM-RoBERTa模型的基类
        "FlaxXLMRobertaModel",
        # FlaxXLMRobertaPreTrainedModel是FLAX-XLM-RoBERTa预训练模型的基类
        "FlaxXLMRobertaPreTrainedModel",
    ]
# 如果类型检查为真，导入相关的预训练配置映射和配置
if TYPE_CHECKING:
    from .configuration_xlm_roberta import (
        XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
        XLMRobertaConfig,
        XLMRobertaOnnxConfig,
    )

    # 尝试检查是否句子分词可用，如果不可用则引发 OptionalDependencyNotAvailable 异常
    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果句子分词可用，导入 XLMRobertaTokenizer
    else:
        from .tokenization_xlm_roberta import XLMRobertaTokenizer

    # 尝试检查是否 tokenizers 可用，如果不可用则引发 OptionalDependencyNotAvailable 异常
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果 tokenizers 可用，导入 XLMRobertaTokenizerFast
    else:
        from .tokenization_xlm_roberta_fast import XLMRobertaTokenizerFast

    # 尝试检查是否 torch 可用，如果不可用则引发 OptionalDependencyNotAvailable 异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果 torch 可用，导入相关模型
    else:
        from .modeling_xlm_roberta import (
            XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
            XLMRobertaForCausalLM,
            XLMRobertaForMaskedLM,
            XLMRobertaForMultipleChoice,
            XLMRobertaForQuestionAnswering,
            XLMRobertaForSequenceClassification,
            XLMRobertaForTokenClassification,
            XLMRobertaModel,
            XLMRobertaPreTrainedModel,
        )

    # 尝试检查是否 tensorflow 可用，如果不可用则引发 OptionalDependencyNotAvailable 异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果 tensorflow 可用，导入相关模型
    else:
        from .modeling_tf_xlm_roberta import (
            TF_XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFXLMRobertaForCausalLM,
            TFXLMRobertaForMaskedLM,
            TFXLMRobertaForMultipleChoice,
            TFXLMRobertaForQuestionAnswering,
            TFXLMRobertaForSequenceClassification,
            TFXLMRobertaForTokenClassification,
            TFXLMRobertaModel,
            TFXLMRobertaPreTrainedModel,
        )

    # 尝试检查是否 flax 可用，如果不可用则引发 OptionalDependencyNotAvailable 异常
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果 flax 可用，则导入相关模型
    else:
        from .modeling_flax_xlm_roberta import (
            FLAX_XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
            FlaxXLMRobertaForCausalLM,
            FlaxXLMRobertaForMaskedLM,
            FlaxXLMRobertaForMultipleChoice,
            FlaxXLMRobertaForQuestionAnswering,
            FlaxXLMRobertaForSequenceClassification,
            FlaxXLMRobertaForTokenClassification,
            FlaxXLMRobertaModel,
            FlaxXLMRobertaPreTrainedModel,
        )

# 如果类型检查为假，则导入相关的模块
else:
    import sys

    # 将当前模块设置为懒加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```