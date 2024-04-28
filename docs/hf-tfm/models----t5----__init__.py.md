# `.\transformers\models\t5\__init__.py`

```py
# 版权声明以及许可证信息
# 这段代码实现了对HuggingFace团队的依赖和导入结构进行配置

# 导入依赖库，包括类型检查
from typing import TYPE_CHECKING

# 导入工具方法和模块
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_sentencepiece_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块导入结构
_import_structure = {"configuration_t5": ["T5_PRETRAINED_CONFIG_ARCHIVE_MAP", "T5Config", "T5OnnxConfig"]}

# 检查SentencePiece是否可用，如果不可用则抛出异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加T5Tokenizer到导入结构中
    _import_structure["tokenization_t5"] = ["T5Tokenizer"]

# 检查Tokenizers是否可用，如果不可用则抛出异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加T5TokenizerFast到导入结构中
    _import_structure["tokenization_t5_fast"] = ["T5TokenizerFast"]

# 检查PyTorch是否可用，如果不可用则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加T5相关模型到导入结构中
    _import_structure["modeling_t5"] = [
        "T5_PRETRAINED_MODEL_ARCHIVE_LIST",
        "T5EncoderModel",
        "T5ForConditionalGeneration",
        "T5Model",
        "T5PreTrainedModel",
        "load_tf_weights_in_t5",
        "T5ForQuestionAnswering",
        "T5ForSequenceClassification",
    ]

# 检查TensorFlow是否可用，如果不可用则抛出异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加TF版T5相关模型到导入结构中
    _import_structure["modeling_tf_t5"] = [
        "TF_T5_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFT5EncoderModel",
        "TFT5ForConditionalGeneration",
        "TFT5Model",
        "TFT5PreTrainedModel",
    ]

# 检查Flax是否可用，如果不可用则抛出异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加Flax版T5相关模型到导入结构中
    _import_structure["modeling_flax_t5"] = [
        "FlaxT5EncoderModel",
        "FlaxT5ForConditionalGeneration",
        "FlaxT5Model",
        "FlaxT5PreTrainedModel",
    ]

# 如��是类型检查模式，则从配置模块中导入相关内容
if TYPE_CHECKING:
    from .configuration_t5 import T5_PRETRAINED_CONFIG_ARCHIVE_MAP, T5Config, T5OnnxConfig

    # 如果SentencePiece可用，从tokenization_t5模块导入T5Tokenizer
    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_t5 import T5Tokenizer

    # 如果Tokenizers可用，继续执行其他类型检查相关的导入
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果 torch 不可用，尝试导入 T5TokenizerFast
    else:
        from .tokenization_t5_fast import T5TokenizerFast
    
    # 如果 torch 不可用，抛出异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    # 如果抛出异常，则不执行以下代码
    except OptionalDependencyNotAvailable:
        pass
    # 如果 torch 可用，则导入与 T5 相关的模型类
    else:
        from .modeling_t5 import (
            T5_PRETRAINED_MODEL_ARCHIVE_LIST,
            T5EncoderModel,
            T5ForConditionalGeneration,
            T5ForQuestionAnswering,
            T5ForSequenceClassification,
            T5Model,
            T5PreTrainedModel,
            load_tf_weights_in_t5,
        )
    
    # 如果 TensorFlow 不可用，抛出异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    # 如果抛出异常，则不执行以下代码
    except OptionalDependencyNotAvailable:
        pass
    # 如果 TensorFlow 可用，则导入与 TF-T5 相关的模型类
    else:
        from .modeling_tf_t5 import (
            TF_T5_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFT5EncoderModel,
            TFT5ForConditionalGeneration,
            TFT5Model,
            TFT5PreTrainedModel,
        )
    
    # 如果 Flax 不可用，抛出异常
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    # 如果抛出异常，则不执行以下代码
    except OptionalDependencyNotAvailable:
        pass
    # 如果 Flax 可用，则导入与 Flax-T5 相关的模型类
    else:
        from .modeling_flax_t5 import (
            FlaxT5EncoderModel,
            FlaxT5ForConditionalGeneration,
            FlaxT5Model,
            FlaxT5PreTrainedModel,
        )
# 如果不是前面的条件，即导入失败，则导入sys模块
import sys
# 将当前模块添加到sys模块的modules字典中
sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```