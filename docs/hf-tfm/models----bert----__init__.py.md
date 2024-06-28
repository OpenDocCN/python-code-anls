# `.\models\bert\__init__.py`

```
# 从 typing 模块导入 TYPE_CHECKING 类型检查工具
from typing import TYPE_CHECKING

# 从 ...utils 中导入必要的模块和异常类
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tensorflow_text_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义一个字典 _import_structure，用于组织各模块需要导入的内容列表
_import_structure = {
    "configuration_bert": ["BERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "BertConfig", "BertOnnxConfig"],
    "tokenization_bert": ["BasicTokenizer", "BertTokenizer", "WordpieceTokenizer"],
}

# 检查是否安装了 tokenizers 库，如果未安装则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果安装了 tokenizers，则添加 tokenization_bert_fast 模块到 _import_structure 字典
    _import_structure["tokenization_bert_fast"] = ["BertTokenizerFast"]

# 检查是否安装了 torch 库，如果未安装则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果安装了 torch，则添加 modeling_bert 模块到 _import_structure 字典
    _import_structure["modeling_bert"] = [
        "BERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "BertForMaskedLM",
        "BertForMultipleChoice",
        "BertForNextSentencePrediction",
        "BertForPreTraining",
        "BertForQuestionAnswering",
        "BertForSequenceClassification",
        "BertForTokenClassification",
        "BertLayer",
        "BertLMHeadModel",
        "BertModel",
        "BertPreTrainedModel",
        "load_tf_weights_in_bert",
    ]

# 检查是否安装了 TensorFlow 库，如果未安装则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果安装了 TensorFlow，则添加 modeling_tf_bert 模块到 _import_structure 字典
    _import_structure["modeling_tf_bert"] = [
        "TF_BERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFBertEmbeddings",
        "TFBertForMaskedLM",
        "TFBertForMultipleChoice",
        "TFBertForNextSentencePrediction",
        "TFBertForPreTraining",
        "TFBertForQuestionAnswering",
        "TFBertForSequenceClassification",
        "TFBertForTokenClassification",
        "TFBertLMHeadModel",
        "TFBertMainLayer",
        "TFBertModel",
        "TFBertPreTrainedModel",
    ]

# 检查是否安装了 TensorFlow Text 库，如果未安装则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tensorflow_text_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果安装了 TensorFlow Text，则添加 tokenization_bert_tf 模块到 _import_structure 字典
    _import_structure["tokenization_bert_tf"] = ["TFBertTokenizer"]

# 检查是否安装了 Flax 库，如果未安装则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果安装了 Flax，则继续添加相关内容，未提供完整的代码
    pass
    # 将多个模型类名添加到_import_structure字典中的"modeling_flax_bert"键下
    _import_structure["modeling_flax_bert"] = [
        "FlaxBertForCausalLM",                   # FlaxBert用于因果语言建模的模型类
        "FlaxBertForMaskedLM",                   # FlaxBert用于遮蔽语言建模的模型类
        "FlaxBertForMultipleChoice",             # FlaxBert用于多选题的模型类
        "FlaxBertForNextSentencePrediction",     # FlaxBert用于下一句预测的模型类
        "FlaxBertForPreTraining",                # FlaxBert用于预训练的模型类
        "FlaxBertForQuestionAnswering",          # FlaxBert用于问答的模型类
        "FlaxBertForSequenceClassification",     # FlaxBert用于序列分类的模型类
        "FlaxBertForTokenClassification",        # FlaxBert用于标记分类的模型类
        "FlaxBertModel",                         # FlaxBert模型的基础模型类
        "FlaxBertPreTrainedModel",               # FlaxBert预训练模型的基础模型类
    ]
# 如果在类型检查模式下
if TYPE_CHECKING:
    # 导入 BERT 配置相关的模块和类
    from .configuration_bert import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, BertConfig, BertOnnxConfig
    # 导入 BERT 的分词器相关模块和类
    from .tokenization_bert import BasicTokenizer, BertTokenizer, WordpieceTokenizer

    # 尝试检查 tokenizers 是否可用，如果不可用则抛出异常 OptionalDependencyNotAvailable
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，导入快速的 BERT 分词器
        from .tokenization_bert_fast import BertTokenizerFast

    # 尝试检查 torch 是否可用，如果不可用则抛出异常 OptionalDependencyNotAvailable
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，导入 BERT 相关的模型和类
        from .modeling_bert import (
            BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            BertForMaskedLM,
            BertForMultipleChoice,
            BertForNextSentencePrediction,
            BertForPreTraining,
            BertForQuestionAnswering,
            BertForSequenceClassification,
            BertForTokenClassification,
            BertLayer,
            BertLMHeadModel,
            BertModel,
            BertPreTrainedModel,
            load_tf_weights_in_bert,
        )

    # 尝试检查 tensorflow 是否可用，如果不可用则抛出异常 OptionalDependencyNotAvailable
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，导入 TF 版本的 BERT 相关模型和类
        from .modeling_tf_bert import (
            TF_BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFBertEmbeddings,
            TFBertForMaskedLM,
            TFBertForMultipleChoice,
            TFBertForNextSentencePrediction,
            TFBertForPreTraining,
            TFBertForQuestionAnswering,
            TFBertForSequenceClassification,
            TFBertForTokenClassification,
            TFBertLMHeadModel,
            TFBertMainLayer,
            TFBertModel,
            TFBertPreTrainedModel,
        )

    # 尝试检查 tensorflow_text 是否可用，如果不可用则抛出异常 OptionalDependencyNotAvailable
    try:
        if not is_tensorflow_text_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，导入 TF 版本的 BERT 分词器
        from .tokenization_bert_tf import TFBertTokenizer

    # 尝试检查 flax 是否可用，如果不可用则抛出异常 OptionalDependencyNotAvailable
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，导入 Flax 版本的 BERT 相关模型和类
        from .modeling_flax_bert import (
            FlaxBertForCausalLM,
            FlaxBertForMaskedLM,
            FlaxBertForMultipleChoice,
            FlaxBertForNextSentencePrediction,
            FlaxBertForPreTraining,
            FlaxBertForQuestionAnswering,
            FlaxBertForSequenceClassification,
            FlaxBertForTokenClassification,
            FlaxBertModel,
            FlaxBertPreTrainedModel,
        )

# 如果不在类型检查模式下
else:
    # 导入 sys 模块
    import sys

    # 将当前模块设置为一个 LazyModule 对象，并导入相关结构和规范
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```