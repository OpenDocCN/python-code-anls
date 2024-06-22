# `.\transformers\models\llama\__init__.py`

```py
# 导入必要的模块和函数
from typing import TYPE_CHECKING
# 导入自定义的异常和工具函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_sentencepiece_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块的导入结构
_import_structure = {
    "configuration_llama": ["LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP", "LlamaConfig"],
}

# 尝试检查是否可用句子分割模块，若不可用则抛出异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用则导入句子分割模块的标记器
    _import_structure["tokenization_llama"] = ["LlamaTokenizer"]

# 尝试检查是否可用标记器模块，若不可用则抛出异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用则导入标记器模块的快速标记器
    _import_structure["tokenization_llama_fast"] = ["LlamaTokenizerFast"]

# 尝试检查是否可用 Torch 模块，若不可用则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用则导入 Torch 模型相关的模块和函数
    _import_structure["modeling_llama"] = [
        "LlamaForCausalLM",
        "LlamaModel",
        "LlamaPreTrainedModel",
        "LlamaForSequenceClassification",
    ]

# 尝试检查是否可用 Flax 模块，若不可用则抛出异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用则导入 Flax 模型相关的模块和函数
    _import_structure["modeling_flax_llama"] = ["FlaxLlamaForCausalLM", "FlaxLlamaModel", "FlaxLlamaPreTrainedModel"]

# 若是类型检查环境，则导入更多的模块以进行类型检查
if TYPE_CHECKING:
    from .configuration_llama import LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP, LlamaConfig

    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_llama import LlamaTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_llama_fast import LlamaTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_llama import LlamaForCausalLM, LlamaForSequenceClassification, LlamaModel, LlamaPreTrainedModel

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果条件不满足，则导入模块 .modeling_flax_llama 中的特定类和模型
    else:
        from .modeling_flax_llama import FlaxLlamaForCausalLM, FlaxLlamaModel, FlaxLlamaPreTrainedModel
否则，如果进入else分支表示当前模块不是主模块，是被导入的模块

导入sys模块，用于操作Python解释器的功能

将当前模块赋值给sys.modules内的字典，__name__作为键，_LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)为值
```