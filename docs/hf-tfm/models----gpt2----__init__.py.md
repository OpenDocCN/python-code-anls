# `.\models\gpt2\__init__.py`

```py
# 导入必要的模块和函数
from typing import TYPE_CHECKING

# 导入 HuggingFace 工具函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_keras_nlp_available,
    is_tensorflow_text_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块导入结构
_import_structure = {
    "configuration_gpt2": ["GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPT2Config", "GPT2OnnxConfig"],
    "tokenization_gpt2": ["GPT2Tokenizer"],
}

# 检查是否存在 tokenizers 库，若不存在则引发异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 tokenizers 库，则加入 tokenization_gpt2_fast 模块
    _import_structure["tokenization_gpt2_fast"] = ["GPT2TokenizerFast"]

# 检查是否存在 torch 库，若不存在则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 torch 库，则加入 modeling_gpt2 模块
    _import_structure["modeling_gpt2"] = [
        "GPT2_PRETRAINED_MODEL_ARCHIVE_LIST",
        "GPT2DoubleHeadsModel",
        "GPT2ForQuestionAnswering",
        "GPT2ForSequenceClassification",
        "GPT2ForTokenClassification",
        "GPT2LMHeadModel",
        "GPT2Model",
        "GPT2PreTrainedModel",
        "load_tf_weights_in_gpt2",
    ]

# 检查是否存在 TensorFlow 库，若不存在则引发异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 TensorFlow 库，则加入 modeling_tf_gpt2 模块
    _import_structure["modeling_tf_gpt2"] = [
        "TF_GPT2_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFGPT2DoubleHeadsModel",
        "TFGPT2ForSequenceClassification",
        "TFGPT2LMHeadModel",
        "TFGPT2MainLayer",
        "TFGPT2Model",
        "TFGPT2PreTrainedModel",
    ]

# 检查是否存在 Keras NLP 库，若不存在则引发异常
try:
    if not is_keras_nlp_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 Keras NLP 库，则加入 tokenization_gpt2_tf 模块
    _import_structure["tokenization_gpt2_tf"] = ["TFGPT2Tokenizer"]

# 检查是否存在 Flax 库，若不存在则引发异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 Flax 库，则加入 modeling_flax_gpt2 模块
    _import_structure["modeling_flax_gpt2"] = ["FlaxGPT2LMHeadModel", "FlaxGPT2Model", "FlaxGPT2PreTrainedModel"]

# 如果是类型检查模式，则导入 GPT2 配置和分词器
if TYPE_CHECKING:
    from .configuration_gpt2 import GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP, GPT2Config, GPT2OnnxConfig
    from .tokenization_gpt2 import GPT2Tokenizer

    # 检查是否存在 tokenizers 库，若不存在则引发异常
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
```  
    # 如果当前环境支持 Torch，导入相关模块
    else:
        from .tokenization_gpt2_fast import GPT2TokenizerFast

    # 检查 Torch 是否可用，如果不可用则抛出异常 OptionalDependencyNotAvailable，否则导入相关模块
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_gpt2 import (
            GPT2_PRETRAINED_MODEL_ARCHIVE_LIST,
            GPT2DoubleHeadsModel,
            GPT2ForQuestionAnswering,
            GPT2ForSequenceClassification,
            GPT2ForTokenClassification,
            GPT2LMHeadModel,
            GPT2Model,
            GPT2PreTrainedModel,
            load_tf_weights_in_gpt2,
        )

    # 检查 TensorFlow 是否可用，如果不可用则抛出异常 OptionalDependencyNotAvailable，否则导入相关模块
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_gpt2 import (
            TF_GPT2_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFGPT2DoubleHeadsModel,
            TFGPT2ForSequenceClassification,
            TFGPT2LMHeadModel,
            TFGPT2MainLayer,
            TFGPT2Model,
            TFGPT2PreTrainedModel,
        )

    # 检查 Keras NLP 是否可用，如果不可用则抛出异常 OptionalDependencyNotAvailable，否则导入相关模块
    try:
        if not is_keras_nlp_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_gpt2_tf import TFGPT2Tokenizer

    # 检查 Flax 是否可用，如果不可用则抛出异常 OptionalDependencyNotAvailable，否则导入相关模块
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_flax_gpt2 import FlaxGPT2LMHeadModel, FlaxGPT2Model, FlaxGPT2PreTrainedModel
# 如果不满足前面的条件，即当前模块不是主模块，那么导入sys模块
import sys
# 将当前模块添加到sys.modules字典中，使用_LazyModule类延迟导入模块
sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```