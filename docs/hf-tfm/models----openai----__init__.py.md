# `.\models\openai\__init__.py`

```
# 导入需要的模块和函数
from typing import TYPE_CHECKING

# 从相对路径的模块导入必要的工具和异常处理类
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块的导入结构，包含模块名和需要导入的类和函数列表
_import_structure = {
    "configuration_openai": ["OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP", "OpenAIGPTConfig"],
    "tokenization_openai": ["OpenAIGPTTokenizer"],
}

# 检查是否可用 tokenizers 模块，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则添加 tokenization_openai_fast 模块到导入结构中
    _import_structure["tokenization_openai_fast"] = ["OpenAIGPTTokenizerFast"]

# 检查是否可用 torch 模块，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则添加 modeling_openai 模块到导入结构中
    _import_structure["modeling_openai"] = [
        "OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "OpenAIGPTDoubleHeadsModel",
        "OpenAIGPTForSequenceClassification",
        "OpenAIGPTLMHeadModel",
        "OpenAIGPTModel",
        "OpenAIGPTPreTrainedModel",
        "load_tf_weights_in_openai_gpt",
    ]

# 检查是否可用 tensorflow 模块，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则添加 modeling_tf_openai 模块到导入结构中
    _import_structure["modeling_tf_openai"] = [
        "TF_OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFOpenAIGPTDoubleHeadsModel",
        "TFOpenAIGPTForSequenceClassification",
        "TFOpenAIGPTLMHeadModel",
        "TFOpenAIGPTMainLayer",
        "TFOpenAIGPTModel",
        "TFOpenAIGPTPreTrainedModel",
    ]

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 从相对路径的 configuration_openai 模块中导入特定类和常量
    from .configuration_openai import OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP, OpenAIGPTConfig
    # 从相对路径的 tokenization_openai 模块中导入特定类
    from .tokenization_openai import OpenAIGPTTokenizer

    # 尝试检查 tokenizers 模块是否可用
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，从相对路径的 tokenization_openai_fast 模块中导入特定类
        from .tokenization_openai_fast import OpenAIGPTTokenizerFast

    # 尝试检查 torch 模块是否可用
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果不是TensorFlow环境，则导入以下模块和内容
    else:
        from .modeling_openai import (
            OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST,  # 导入OpenAI GPT预训练模型存档列表
            OpenAIGPTDoubleHeadsModel,  # 导入OpenAI GPT双头模型类
            OpenAIGPTForSequenceClassification,  # 导入OpenAI GPT序列分类模型类
            OpenAIGPTLMHeadModel,  # 导入OpenAI GPT语言模型头部模型类
            OpenAIGPTModel,  # 导入OpenAI GPT模型类
            OpenAIGPTPreTrainedModel,  # 导入OpenAI GPT预训练模型基类
            load_tf_weights_in_openai_gpt,  # 导入OpenAI GPT加载TensorFlow权重的函数
        )

    try:
        # 如果TensorFlow不可用，则引发OptionalDependencyNotAvailable异常
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果捕获到OptionalDependencyNotAvailable异常，则不做任何处理
        pass
    else:
        # 否则，在TensorFlow环境中导入以下模块和内容
        from .modeling_tf_openai import (
            TF_OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST,  # 导入TensorFlow版OpenAI GPT预训练模型存档列表
            TFOpenAIGPTDoubleHeadsModel,  # 导入TensorFlow版OpenAI GPT双头模型类
            TFOpenAIGPTForSequenceClassification,  # 导入TensorFlow版OpenAI GPT序列分类模型类
            TFOpenAIGPTLMHeadModel,  # 导入TensorFlow版OpenAI GPT语言模型头部模型类
            TFOpenAIGPTMainLayer,  # 导入TensorFlow版OpenAI GPT主层模型类
            TFOpenAIGPTModel,  # 导入TensorFlow版OpenAI GPT模型类
            TFOpenAIGPTPreTrainedModel,  # 导入TensorFlow版OpenAI GPT预训练模型基类
        )
else:
    # 导入 sys 模块，用于管理 Python 解释器的系统功能
    import sys

    # 将当前模块注册到 sys.modules 字典中，使用 _LazyModule 对象作为值
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```