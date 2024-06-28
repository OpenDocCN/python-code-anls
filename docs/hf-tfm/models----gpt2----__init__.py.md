# `.\models\gpt2\__init__.py`

```py
# 引入类型检查工具，用于类型检查
from typing import TYPE_CHECKING

# 引入必要的依赖模块和函数
# 从 utils 模块中导入必要的异常类、延迟加载模块、可用性检查函数等
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

# 定义模块导入结构字典，包含各模块所需导入的类或函数列表
_import_structure = {
    "configuration_gpt2": ["GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPT2Config", "GPT2OnnxConfig"],
    "tokenization_gpt2": ["GPT2Tokenizer"],
}

# 尝试导入 tokenizers 模块，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，将 tokenization_gpt2_fast 模块添加到导入结构中
    _import_structure["tokenization_gpt2_fast"] = ["GPT2TokenizerFast"]

# 尝试导入 torch 模块，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，将 modeling_gpt2 模块添加到导入结构中
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

# 尝试导入 tensorflow 模块，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，将 modeling_tf_gpt2 模块添加到导入结构中
    _import_structure["modeling_tf_gpt2"] = [
        "TF_GPT2_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFGPT2DoubleHeadsModel",
        "TFGPT2ForSequenceClassification",
        "TFGPT2LMHeadModel",
        "TFGPT2MainLayer",
        "TFGPT2Model",
        "TFGPT2PreTrainedModel",
    ]

# 尝试导入 keras_nlp 模块，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_keras_nlp_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，将 tokenization_gpt2_tf 模块添加到导入结构中
    _import_structure["tokenization_gpt2_tf"] = ["TFGPT2Tokenizer"]

# 尝试导入 flax 模块，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，将 modeling_flax_gpt2 模块添加到导入结构中
    _import_structure["modeling_flax_gpt2"] = ["FlaxGPT2LMHeadModel", "FlaxGPT2Model", "FlaxGPT2PreTrainedModel"]

# 如果在类型检查模式下，导入所需的类型定义
if TYPE_CHECKING:
    from .configuration_gpt2 import GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP, GPT2Config, GPT2OnnxConfig
    from .tokenization_gpt2 import GPT2Tokenizer

    try:
        # 在类型检查模式下，检查 tokenizers 模块的可用性
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果前面导入失败，则尝试导入 GPT2TokenizerFast
    else:
        from .tokenization_gpt2_fast import GPT2TokenizerFast

    try:
        # 检查是否存在 torch 库，如果不存在则抛出 OptionalDependencyNotAvailable 异常
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果依赖不可用，则忽略此部分代码块
        pass
    else:
        # 导入相关的 GPT-2 模型和相关类
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

    try:
        # 检查是否存在 TensorFlow 库，如果不存在则抛出 OptionalDependencyNotAvailable 异常
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果依赖不可用，则忽略此部分代码块
        pass
    else:
        # 导入相关的 TensorFlow 版本的 GPT-2 模型和相关类
        from .modeling_tf_gpt2 import (
            TF_GPT2_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFGPT2DoubleHeadsModel,
            TFGPT2ForSequenceClassification,
            TFGPT2LMHeadModel,
            TFGPT2MainLayer,
            TFGPT2Model,
            TFGPT2PreTrainedModel,
        )

    try:
        # 检查是否存在 keras_nlp 库，如果不存在则抛出 OptionalDependencyNotAvailable 异常
        if not is_keras_nlp_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果依赖不可用，则忽略此部分代码块
        pass
    else:
        # 导入 TensorFlow 版本的 GPT-2 的 tokenizer
        from .tokenization_gpt2_tf import TFGPT2Tokenizer

    try:
        # 检查是否存在 flax 库，如果不存在则抛出 OptionalDependencyNotAvailable 异常
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果依赖不可用，则忽略此部分代码块
        pass
    else:
        # 导入 Flax 版本的 GPT-2 模型和相关类
        from .modeling_flax_gpt2 import FlaxGPT2LMHeadModel, FlaxGPT2Model, FlaxGPT2PreTrainedModel
else:
    # 如果不处于前述条件分支，则执行以下操作

    import sys
    # 导入系统模块，用于操作 Python 运行时环境

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
    # 将当前模块名注册到 sys.modules 中，以 LazyModule 的形式，支持按需导入的模块加载策略
```