# `.\transformers\models\bert\__init__.py`

```
# 版权声明和许可信息
# 从类型提示中导入 TYPE_CHECKING
from typing import TYPE_CHECKING
# 从工具模块中导入必要的函数和异常类
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tensorflow_text_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块导入结构
_import_structure = {
    "configuration_bert": ["BERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "BertConfig", "BertOnnxConfig"],
    "tokenization_bert": ["BasicTokenizer", "BertTokenizer", "WordpieceTokenizer"],
}

# 检查 tokenizers 是否可用，若不可用则抛出异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则添加 tokenization_bert_fast 到导入结构中
    _import_structure["tokenization_bert_fast"] = ["BertTokenizerFast"]

# 检查 torch 是否可用，若不可用则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则添加 modeling_bert 到导入结构中
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

# 检查 tensorflow 是否可用，若不可用则抛出异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则添加 modeling_tf_bert 到导入结构中
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

# 检查 tensorflow_text 是否可用，若不可用则抛出异常
try:
    if not is_tensorflow_text_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则添加 tokenization_bert_tf 到导入结构中
    _import_structure["tokenization_bert_tf"] = ["TFBertTokenizer"]

# 检查 flax 是否可用，若不可用则抛出异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
# 若可用，则继续处理
    # 将模块"modeling_flax_bert"下的类名添加到_import_structure字典中
    _import_structure["modeling_flax_bert"] = [
        "FlaxBertForCausalLM",
        "FlaxBertForMaskedLM",
        "FlaxBertForMultipleChoice",
        "FlaxBertForNextSentencePrediction",
        "FlaxBertForPreTraining",
        "FlaxBertForQuestionAnswering",
        "FlaxBertForSequenceClassification",
        "FlaxBertForTokenClassification",
        "FlaxBertModel",
        "FlaxBertPreTrainedModel",
    ]
# 如果 TYPE_CHECKING 为真，则导入所需的模块和类
if TYPE_CHECKING:
    # 导入 BERT 配置映射、配置类和 ONNX 配置类
    from .configuration_bert import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, BertConfig, BertOnnxConfig
    # 导入 BERT 分词器基类、基本分词器、Bert 分词器和 Wordpiece 分词器
    from .tokenization_bert import BasicTokenizer, BertTokenizer, WordpieceTokenizer

    # 尝试检查是否安装了 tokenizers 库
    try:
        # 如果 tokenizers 库不可用，则引发 OptionalDependencyNotAvailable 异常
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 tokenizers 库可用，则导入 BertTokenizerFast 类
        from .tokenization_bert_fast import BertTokenizerFast

    # 尝试检查是否安装了 PyTorch 库
    try:
        # 如果 PyTorch 库不可用，则引发 OptionalDependencyNotAvailable 异常
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 PyTorch 库可用，则导入一系列与 BERT 相关的模型和类
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

    # 尝试检查是否安装了 TensorFlow 库
    try:
        # 如果 TensorFlow 库不可用，则引发 OptionalDependencyNotAvailable 异常
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 TensorFlow 库可用，则导入一系列与 TensorFlow BERT 相关的模型和类
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

    # 尝试检查是否安装了 TensorFlow Text 库
    try:
        # 如果 TensorFlow Text 库不可用，则引发 OptionalDependencyNotAvailable 异常
        if not is_tensorflow_text_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 TensorFlow Text 库可用，则导入 TFBertTokenizer 类
        from .tokenization_bert_tf import TFBertTokenizer

    # 尝试检查是否安装了 Flax 库
    try:
        # 如果 Flax 库不可用，则引发 OptionalDependencyNotAvailable 异常
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 Flax 库可用，则导入一系列与 Flax BERT 相关的模型和类
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

# 如果 TYPE_CHECKING 为假，则动态地将当前模块设为懒加载模块
else:
    # 导入 sys 模块
    import sys

    # 将当前模块设置为懒加载模块，其导入结构由 _import_structure 定义
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```