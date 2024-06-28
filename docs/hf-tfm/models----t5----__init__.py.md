# `.\models\t5\__init__.py`

```py
# 导入必要的模块和函数
from typing import TYPE_CHECKING
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_sentencepiece_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义一个字典，用于存储导入结构
_import_structure = {"configuration_t5": ["T5_PRETRAINED_CONFIG_ARCHIVE_MAP", "T5Config", "T5OnnxConfig"]}

# 检查是否存在 sentencepiece，并根据情况抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 sentencepiece，则导入 T5Tokenizer 到 tokenization_t5
    _import_structure["tokenization_t5"] = ["T5Tokenizer"]

# 检查是否存在 tokenizers，并根据情况抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 tokenizers，则导入 T5TokenizerFast 到 tokenization_t5_fast
    _import_structure["tokenization_t5_fast"] = ["T5TokenizerFast"]

# 检查是否存在 torch，并根据情况抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 torch，则导入 T5 相关的模型和函数到 modeling_t5
    _import_structure["modeling_t5"] = [
        "T5_PRETRAINED_MODEL_ARCHIVE_LIST",
        "T5EncoderModel",
        "T5ForConditionalGeneration",
        "T5Model",
        "T5PreTrainedModel",
        "load_tf_weights_in_t5",
        "T5ForQuestionAnswering",
        "T5ForSequenceClassification",
        "T5ForTokenClassification",
    ]

# 检查是否存在 tensorflow，并根据情况抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 tensorflow，则导入 T5 相关的模型和函数到 modeling_tf_t5
    _import_structure["modeling_tf_t5"] = [
        "TF_T5_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFT5EncoderModel",
        "TFT5ForConditionalGeneration",
        "TFT5Model",
        "TFT5PreTrainedModel",
    ]

# 检查是否存在 flax，并根据情况抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 flax，则导入 T5 相关的模型和函数到 modeling_flax_t5
    _import_structure["modeling_flax_t5"] = [
        "FlaxT5EncoderModel",
        "FlaxT5ForConditionalGeneration",
        "FlaxT5Model",
        "FlaxT5PreTrainedModel",
    ]

# 如果是类型检查阶段，导入必要的类型和函数定义
if TYPE_CHECKING:
    from .configuration_t5 import T5_PRETRAINED_CONFIG_ARCHIVE_MAP, T5Config, T5OnnxConfig

    # 再次检查是否存在 sentencepiece 并导入 T5Tokenizer
    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_t5 import T5Tokenizer

    # 再次检查是否存在 tokenizers 并导入 T5Tokenizer
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    # 尝试导入 T5TokenizerFast，如果 OptionalDependencyNotAvailable 异常发生则跳过
    try:
        from .tokenization_t5_fast import T5TokenizerFast
    # 如果 OptionalDependencyNotAvailable 异常发生，则什么也不做，跳过
    except OptionalDependencyNotAvailable:
        pass
    # 如果没有异常发生，则导入成功，可以继续后续操作
    else:
        # 尝试检查是否 Torch 可用，如果不可用则引发 OptionalDependencyNotAvailable 异常
        try:
            if not is_torch_available():
                raise OptionalDependencyNotAvailable()
        # 如果 OptionalDependencyNotAvailable 异常发生，则跳过
        except OptionalDependencyNotAvailable:
            pass
        # 如果没有异常发生，则 Torch 可用，继续导入相关模块
        else:
            # 导入 T5 相关的 PyTorch 模型和函数
            from .modeling_t5 import (
                T5_PRETRAINED_MODEL_ARCHIVE_LIST,
                T5EncoderModel,
                T5ForConditionalGeneration,
                T5ForQuestionAnswering,
                T5ForSequenceClassification,
                T5ForTokenClassification,
                T5Model,
                T5PreTrainedModel,
                load_tf_weights_in_t5,
            )

        # 尝试检查是否 TensorFlow 可用，如果不可用则引发 OptionalDependencyNotAvailable 异常
        try:
            if not is_tf_available():
                raise OptionalDependencyNotAvailable()
        # 如果 OptionalDependencyNotAvailable 异常发生，则跳过
        except OptionalDependencyNotAvailable:
            pass
        # 如果没有异常发生，则 TensorFlow 可用，继续导入相关模块
        else:
            # 导入 T5 相关的 TensorFlow 模型和函数
            from .modeling_tf_t5 import (
                TF_T5_PRETRAINED_MODEL_ARCHIVE_LIST,
                TFT5EncoderModel,
                TFT5ForConditionalGeneration,
                TFT5Model,
                TFT5PreTrainedModel,
            )

        # 尝试检查是否 Flax 可用，如果不可用则引发 OptionalDependencyNotAvailable 异常
        try:
            if not is_flax_available():
                raise OptionalDependencyNotAvailable()
        # 如果 OptionalDependencyNotAvailable 异常发生，则跳过
        except OptionalDependencyNotAvailable:
            pass
        # 如果没有异常发生，则 Flax 可用，继续导入相关模块
        else:
            # 导入 T5 相关的 Flax 模型和函数
            from .modeling_flax_t5 import (
                FlaxT5EncoderModel,
                FlaxT5ForConditionalGeneration,
                FlaxT5Model,
                FlaxT5PreTrainedModel,
            )
else:
    # 导入 sys 模块，用于操作 Python 解释器的系统功能
    import sys

    # 将当前模块注册到 sys.modules 中，使用 _LazyModule 进行延迟加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```