# `.\models\bart\__init__.py`

```py
# 引入类型检查模块，用于在类型检查时导入特定模块
from typing import TYPE_CHECKING

# 从工具模块中导入必要的类和函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义导入结构，将模块名映射到需要导入的类和函数列表
_import_structure = {
    "configuration_bart": ["BART_PRETRAINED_CONFIG_ARCHIVE_MAP", "BartConfig", "BartOnnxConfig"],
    "tokenization_bart": ["BartTokenizer"],
}

# 检查是否存在 Tokenizers 库，如果不存在则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 Tokenizers 库，则添加 "tokenization_bart_fast" 到导入结构中
    _import_structure["tokenization_bart_fast"] = ["BartTokenizerFast"]

# 检查是否存在 Torch 库，如果不存在则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 Torch 库，则添加 "modeling_bart" 到导入结构中
    _import_structure["modeling_bart"] = [
        "BART_PRETRAINED_MODEL_ARCHIVE_LIST",
        "BartForCausalLM",
        "BartForConditionalGeneration",
        "BartForQuestionAnswering",
        "BartForSequenceClassification",
        "BartModel",
        "BartPreTrainedModel",
        "BartPretrainedModel",
        "PretrainedBartModel",
    ]

# 检查是否存在 TensorFlow 库，如果不存在则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 TensorFlow 库，则添加 "modeling_tf_bart" 到导入结构中
    _import_structure["modeling_tf_bart"] = [
        "TFBartForConditionalGeneration",
        "TFBartForSequenceClassification",
        "TFBartModel",
        "TFBartPretrainedModel",
    ]

# 检查是否存在 Flax 库，如果不存在则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 Flax 库，则添加 "modeling_flax_bart" 到导入结构中
    _import_structure["modeling_flax_bart"] = [
        "FlaxBartDecoderPreTrainedModel",
        "FlaxBartForCausalLM",
        "FlaxBartForConditionalGeneration",
        "FlaxBartForQuestionAnswering",
        "FlaxBartForSequenceClassification",
        "FlaxBartModel",
        "FlaxBartPreTrainedModel",
    ]

# 如果在类型检查时，导入以下模块和类
if TYPE_CHECKING:
    from .configuration_bart import BART_PRETRAINED_CONFIG_ARCHIVE_MAP, BartConfig, BartOnnxConfig
    from .tokenization_bart import BartTokenizer

    # 检查是否存在 Tokenizers 库，如果不存在则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果存在 Tokenizers 库，则导入 BartTokenizerFast 类
        from .tokenization_bart_fast import BartTokenizerFast

    # 检查是否存在 Torch 库，如果不存在则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    # 尝试导入模型 BART 的相关模块和类，如果依赖项不可用则忽略
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 BART 模型的预训练模型存档列表和各种模型类
        from .modeling_bart import (
            BART_PRETRAINED_MODEL_ARCHIVE_LIST,
            BartForCausalLM,
            BartForConditionalGeneration,
            BartForQuestionAnswering,
            BartForSequenceClassification,
            BartModel,
            BartPreTrainedModel,
            BartPretrainedModel,
            PretrainedBartModel,
        )

    # 尝试导入 TensorFlow 版本的 BART 相关模块和类，如果 TensorFlow 不可用则忽略
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 TensorFlow 版本的 BART 模型的各种类
        from .modeling_tf_bart import (
            TFBartForConditionalGeneration,
            TFBartForSequenceClassification,
            TFBartModel,
            TFBartPretrainedModel,
        )

    # 尝试导入 Flax 版本的 BART 相关模块和类，如果 Flax 不可用则忽略
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 Flax 版本的 BART 模型的各种类
        from .modeling_flax_bart import (
            FlaxBartDecoderPreTrainedModel,
            FlaxBartForCausalLM,
            FlaxBartForConditionalGeneration,
            FlaxBartForQuestionAnswering,
            FlaxBartForSequenceClassification,
            FlaxBartModel,
            FlaxBartPreTrainedModel,
        )
else:
    # 导入 sys 模块
    import sys

    # 将当前模块注册到 sys.modules 中，使用 _LazyModule 包装
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```