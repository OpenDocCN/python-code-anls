# `.\models\gemma\__init__.py`

```
# 导入所需模块和函数，这里从不同的模块和子模块中导入特定的内容
from typing import TYPE_CHECKING  # 导入类型检查相关的功能

from ...utils import (
    OptionalDependencyNotAvailable,  # 导入自定义的异常类
    _LazyModule,  # 导入懒加载模块的支持
    is_flax_available,  # 检查是否有Flax库可用
    is_sentencepiece_available,  # 检查是否有SentencePiece库可用
    is_tokenizers_available,  # 检查是否有Tokenizers库可用
    is_torch_available,  # 检查是否有PyTorch库可用
)

# 定义一个字典，用于描述导入结构
_import_structure = {
    "configuration_gemma": ["GEMMA_PRETRAINED_CONFIG_ARCHIVE_MAP", "GemmaConfig"],  # Gemma模型配置相关内容
}

# 检查是否有SentencePiece库可用，若不可用则引发OptionalDependencyNotAvailable异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_gemma"] = ["GemmaTokenizer"]  # 导入Gemma模型的分词器

# 检查是否有Tokenizers库可用，若不可用则引发OptionalDependencyNotAvailable异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_gemma_fast"] = ["GemmaTokenizerFast"]  # 导入Gemma模型的快速分词器

# 检查是否有PyTorch库可用，若不可用则引发OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_gemma"] = [
        "GemmaForCausalLM",  # Gemma模型的因果语言模型
        "GemmaModel",  # Gemma模型基类
        "GemmaPreTrainedModel",  # Gemma模型的预训练模型基类
        "GemmaForSequenceClassification",  # Gemma模型的序列分类模型
    ]

# 检查是否有Flax库可用，若不可用则引发OptionalDependencyNotAvailable异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_gemma"] = [
        "FlaxGemmaForCausalLM",  # Flax版本的Gemma因果语言模型
        "FlaxGemmaModel",  # Flax版本的Gemma模型基类
        "FlaxGemmaPreTrainedModel",  # Flax版本的Gemma预训练模型基类
    ]

# 如果在类型检查模式下
if TYPE_CHECKING:
    from .configuration_gemma import GEMMA_PRETRAINED_CONFIG_ARCHIVE_MAP, GemmaConfig  # 导入Gemma模型的配置映射和配置类

    # 检查是否有SentencePiece库可用，若不可用则忽略导入
    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_gemma import GemmaTokenizer  # 导入Gemma模型的分词器

    # 检查是否有Tokenizers库可用，若不可用则忽略导入
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_gemma_fast import GemmaTokenizerFast  # 导入Gemma模型的快速分词器

    # 检查是否有PyTorch库可用，若不可用则忽略导入
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_gemma import (
            GemmaForCausalLM,  # Gemma模型的因果语言模型
            GemmaForSequenceClassification,  # Gemma模型的序列分类模型
            GemmaModel,  # Gemma模型基类
            GemmaPreTrainedModel,  # Gemma模型的预训练模型基类
        )

    # 检查是否有Flax库可用，若不可用则忽略导入
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    # 如果发生 OptionalDependencyNotAvailable 异常，则什么也不做，直接 pass
    except OptionalDependencyNotAvailable:
        pass
    # 如果没有发生异常，则导入以下模块
    else:
        from .modeling_flax_gemma import (
            FlaxGemmaForCausalLM,    # 导入 FlaxGemmaForCausalLM 类
            FlaxGemmaModel,         # 导入 FlaxGemmaModel 类
            FlaxGemmaPreTrainedModel,  # 导入 FlaxGemmaPreTrainedModel 类
        )
else:
    # 如果不是以上情况，即模块不是以导入方式被调用
    import sys
    # 导入 sys 模块，用于操作 Python 解释器的系统功能

    # 将当前模块注册为懒加载模块的实例
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
    # 使用 _LazyModule 类创建一个新的模块对象，并将其注册到 sys.modules 中，以实现懒加载模块的特性
```