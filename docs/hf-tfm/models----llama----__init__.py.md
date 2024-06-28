# `.\models\llama\__init__.py`

```py
# 版权声明和许可信息，指出代码受 Apache 许可证版本 2.0 保护，详见许可证链接
# 导入必要的类型检查模块
from typing import TYPE_CHECKING

# 导入延迟加载模块和依赖检查函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_sentencepiece_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块的导入结构，用于动态导入依赖项
_import_structure = {
    "configuration_llama": ["LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP", "LlamaConfig"],
}

# 检查是否存在 SentencePiece 库，若不存在则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加 tokenization_llama 模块到导入结构中
    _import_structure["tokenization_llama"] = ["LlamaTokenizer"]

# 检查是否存在 Tokenizers 库，若不存在则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加 tokenization_llama_fast 模块到导入结构中
    _import_structure["tokenization_llama_fast"] = ["LlamaTokenizerFast"]

# 检查是否存在 PyTorch 库，若不存在则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加 modeling_llama 模块到导入结构中
    _import_structure["modeling_llama"] = [
        "LlamaForCausalLM",
        "LlamaModel",
        "LlamaPreTrainedModel",
        "LlamaForSequenceClassification",
        "LlamaForQuestionAnswering",
    ]

# 检查是否存在 Flax 库，若不存在则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加 modeling_flax_llama 模块到导入结构中
    _import_structure["modeling_flax_llama"] = ["FlaxLlamaForCausalLM", "FlaxLlamaModel", "FlaxLlamaPreTrainedModel"]

# 如果是类型检查模式，导入配置和模型相关的类和函数
if TYPE_CHECKING:
    from .configuration_llama import LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP, LlamaConfig

    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，导入 tokenization_llama 模块的 LlamaTokenizer 类
        from .tokenization_llama import LlamaTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，导入 tokenization_llama_fast 模块的 LlamaTokenizerFast 类
        from .tokenization_llama_fast import LlamaTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，导入 modeling_llama 模块的相关类
        from .modeling_llama import (
            LlamaForCausalLM,
            LlamaForQuestionAnswering,
            LlamaForSequenceClassification,
            LlamaModel,
            LlamaPreTrainedModel,
        )
    try:
        # 检查是否存在名为 is_flax_available 的函数，用于检测是否可用 Flax 库
        if not is_flax_available():
            # 如果 Flax 库不可用，则引发 OptionalDependencyNotAvailable 异常
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果 OptionalDependencyNotAvailable 异常被引发，则忽略并继续执行后续代码
        pass
    else:
        # 如果没有异常发生，则从当前包导入 Flax 模型相关类
        from .modeling_flax_llama import FlaxLlamaForCausalLM, FlaxLlamaModel, FlaxLlamaPreTrainedModel
else:
    # 如果前面的条件不满足，则执行以下代码块
    import sys
    # 导入 sys 模块，用于在运行时操作 Python 解释器的功能

    # 将当前模块注册为一个延迟加载模块的实例，将其赋值给当前模块的名称
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```