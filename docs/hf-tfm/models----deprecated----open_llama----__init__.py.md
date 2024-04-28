# `.\models\deprecated\open_llama\__init__.py`

```py
# 版权声明
# Copyright 2023 EleutherAI and The HuggingFace Inc. team. All rights reserved.

# 引入类型检查
from typing import TYPE_CHECKING

# 引入必要的依赖和模块
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

# 检查是否安装了 sentencepiece，如果没有则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果安装了 sentencepiece，则将 tokenization_open_llama 模块添加到导入结构中
    _import_structure["tokenization_open_llama"] = ["LlamaTokenizer"]

# 检查是否安装了 tokenizers，如果没有则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果安装了 tokenizers，则将 tokenization_open_llama_fast 模块添加到导入结构中
    _import_structure["tokenization_open_llama_fast"] = ["LlamaTokenizerFast"]

# 检查是否安装了 torch，如果没有则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果安装了 torch，则将 modeling_open_llama 模块添加到导入结构中
    _import_structure["modeling_open_llama"] = [
        "OpenLlamaForCausalLM",
        "OpenLlamaModel",
        "OpenLlamaPreTrainedModel",
        "OpenLlamaForSequenceClassification",
    ]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 导入 configuration_open_llama 模块的特定函数和类
    from .configuration_open_llama import OPEN_LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP, OpenLlamaConfig
    # 导入 LlamaTokenizer 类
    from transformers import LlamaTokenizer
    # 导入 LlamaTokenizerFast 类
    from transformers import LlamaTokenizerFast
    # 导入 modeling_open_llama 模块的特定函数和类
    from .modeling_open_llama import (
        OpenLlamaForCausalLM,
        OpenLlamaForSequenceClassification,
        OpenLlamaModel,
        OpenLlamaPreTrainedModel,
    )
# 否则
else:
    # 引入 sys 模块
    import sys
    # 将当前模块设为延迟加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```