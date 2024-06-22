# `.\transformers\models\openai\__init__.py`

```py
# 导入所需的模块和函数
from typing import TYPE_CHECKING  # 引用TYPE_CHECKING类型提示

from ...utils import (
    OptionalDependencyNotAvailable,  # 导入OptionalDependencyNotAvailable异常
    _LazyModule,  # 导入_LazyModule类
    is_tf_available,  # 导入is_tf_available函数
    is_tokenizers_available,  # 导入is_tokenizers_available函数
    is_torch_available,  # 导入is_torch_available函数
)

# 定义一个字典变量，包含导入结构
_import_structure = {
    "configuration_openai": ["OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP", "OpenAIGPTConfig"],  # 导入configuration_openai模块的两个变量
    "tokenization_openai": ["OpenAIGPTTokenizer"],  # 导入tokenization_openai模块的一个变量
}

# 检查是否可用tokenizers库
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()  # 如果tokenizers库不可用，则抛出OptionalDependencyNotAvailable异常
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_openai_fast"] = ["OpenAIGPTTokenizerFast"]  # 导入tokenization_openai_fast模块的一个变量

# 检查是否可用torch库
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()  # 如果torch库不可用，则抛出OptionalDependencyNotAvailable异常
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_openai"] = [
        "OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST",  # 导入modeling_openai模块的一个变量
        "OpenAIGPTDoubleHeadsModel",  # 导入modeling_openai模块的一个类
        "OpenAIGPTForSequenceClassification",  # 导入modeling_openai模块的一个类
        "OpenAIGPTLMHeadModel",  # 导入modeling_openai模块的一个类
        "OpenAIGPTModel",  # 导入modeling_openai模块的一个类
        "OpenAIGPTPreTrainedModel",  # 导入modeling_openai模块的一个类
        "load_tf_weights_in_openai_gpt",  # 导入modeling_openai模块的一个函数
    ]

# 检查是否可用tensorflow库
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()  # 如果tensorflow库不可用，则抛出OptionalDependencyNotAvailable异常
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_openai"] = [
        "TF_OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST",  # 导入modeling_tf_openai模块的一个变量
        "TFOpenAIGPTDoubleHeadsModel",  # 导入modeling_tf_openai模块的一个类
        "TFOpenAIGPTForSequenceClassification",  # 导入modeling_tf_openai模块的一个类
        "TFOpenAIGPTLMHeadModel",  # 导入modeling_tf_openai模块的一个类
        "TFOpenAIGPTMainLayer",  # 导入modeling_tf_openai模块的一个类
        "TFOpenAIGPTModel",  # 导入modeling_tf_openai模块的一个类
        "TFOpenAIGPTPreTrainedModel",  # 导入modeling_tf_openai模块的一个类
    ]

# 类型检查
if TYPE_CHECKING:
    from .configuration_openai import OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP, OpenAIGPTConfig  # 导入configuration_openai模块中的两个变量
    from .tokenization_openai import OpenAIGPTTokenizer  # 导入tokenization_openai模块中的一个变量

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()  # 如果tokenizers库不可用，则抛出OptionalDependencyNotAvailable异常
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_openai_fast import OpenAIGPTTokenizerFast  # 导入tokenization_openai_fast模块中的一个变量

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()  # 如果torch库不可用，则抛出OptionalDependencyNotAvailable异常
    except OptionalDependencyNotAvailable:
        pass
    # 如果 torch 可用，那么从 .modeling_openai 中导入这些类
    else:
        from .modeling_openai import (
            # OpenAI GPT 预训练模型的存档列表
            OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST,
            # OpenAI GPT 双头模型
            OpenAIGPTDoubleHeadsModel,
            # OpenAI GPT 序列分类模型
            OpenAIGPTForSequenceClassification,
            # OpenAI GPT 语言模型头
            OpenAIGPTLMHeadModel,
            # OpenAI GPT 模型
            OpenAIGPTModel,
            # OpenAI GPT 预训练模型
            OpenAIGPTPreTrainedModel,
            # 从 TensorFlow 加载 OpenAI GPT 权重
            load_tf_weights_in_openai_gpt,
        )
    
    # 尝试加载 TensorFlow
    try:
        # 如果 TensorFlow 不可用
        if not is_tf_available():
            # 引发缺失可选依赖的异常
            raise OptionalDependencyNotAvailable()
    # 如果出现异常
    except OptionalDependencyNotAvailable:
        # 什么也不做
        pass
    # 如果 TensorFlow 可用
    else:
        # 从 .modeling_tf_openai 中导入这些类
        from .modeling_tf_openai import (
            # TensorFlow 中 OpenAI GPT 预训练模型的存档列表
            TF_OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST,
            # TensorFlow 中 OpenAI GPT 双头模型
            TFOpenAIGPTDoubleHeadsModel,
            # TensorFlow 中 OpenAI GPT 序列分类模型
            TFOpenAIGPTForSequenceClassification,
            # TensorFlow 中 OpenAI GPT 语言模型头
            TFOpenAIGPTLMHeadModel,
            # TensorFlow 中 OpenAI GPT 主层
            TFOpenAIGPTMainLayer,
            # TensorFlow 中 OpenAI GPT 模型
            TFOpenAIGPTModel,
            # TensorFlow 中 OpenAI GPT 预训练模型
            TFOpenAIGPTPreTrainedModel,
        )
# 如果条件不满足，即模块已被导入过了
else:
    # 导入 sys 模块
    import sys
    # 将当前模块添加到 sys.modules 字典中，使用 LazyModule 进行懒加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```