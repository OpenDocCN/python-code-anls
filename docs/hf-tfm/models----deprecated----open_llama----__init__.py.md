# `.\models\deprecated\open_llama\__init__.py`

```py
# 导入类型检查工具，用于检查类型是否可用
from typing import TYPE_CHECKING

# 导入必要的依赖项和模块
from ....utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块的导入结构
_import_structure = {
    "configuration_open_llama": ["OPEN_LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP", "OpenLlamaConfig"],
}

# 检查是否有 sentencepiece 可用，如果不可用则引发异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加 tokenization_open_llama 到导入结构中
    _import_structure["tokenization_open_llama"] = ["LlamaTokenizer"]

# 检查是否有 tokenizers 可用，如果不可用则引发异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加 tokenization_open_llama_fast 到导入结构中
    _import_structure["tokenization_open_llama_fast"] = ["LlamaTokenizerFast"]

# 检查是否有 torch 可用，如果不可用则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加 modeling_open_llama 到导入结构中
    _import_structure["modeling_open_llama"] = [
        "OpenLlamaForCausalLM",
        "OpenLlamaModel",
        "OpenLlamaPreTrainedModel",
        "OpenLlamaForSequenceClassification",
    ]


# 如果正在进行类型检查
if TYPE_CHECKING:
    # 导入配置和类型相关的模块
    from .configuration_open_llama import OPEN_LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP, OpenLlamaConfig

    try:
        # 检查是否有 sentencepiece 可用，如果不可用则忽略
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 LlamaTokenizer 类
        from transformers import LlamaTokenizer

    try:
        # 检查是否有 tokenizers 可用，如果不可用则忽略
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 LlamaTokenizerFast 类
        from transformers import LlamaTokenizerFast

    try:
        # 检查是否有 torch 可用，如果不可用则忽略
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 modeling_open_llama 模块中的类
        from .modeling_open_llama import (
            OpenLlamaForCausalLM,
            OpenLlamaForSequenceClassification,
            OpenLlamaModel,
            OpenLlamaPreTrainedModel,
        )

# 如果不是类型检查阶段，则配置 LazyModule 并将其添加到当前模块
else:
    import sys

    # 使用 LazyModule 进行模块的延迟加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```