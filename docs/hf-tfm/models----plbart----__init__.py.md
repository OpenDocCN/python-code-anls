# `.\models\plbart\__init__.py`

```
# 导入类型检查模块
from typing import TYPE_CHECKING

# 导入所需的实用工具和依赖项
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义模块的导入结构，包括配置和模型相关内容
_import_structure = {"configuration_plbart": ["PLBART_PRETRAINED_CONFIG_ARCHIVE_MAP", "PLBartConfig"]}

# 检查是否存在 SentencePiece 库，若不可用则引发异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，将 PLBartTokenizer 添加到导入结构中
    _import_structure["tokenization_plbart"] = ["PLBartTokenizer"]

# 检查是否存在 Torch 库，若不可用则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，将 PLBart 相关模型添加到导入结构中
    _import_structure["modeling_plbart"] = [
        "PLBART_PRETRAINED_MODEL_ARCHIVE_LIST",
        "PLBartForCausalLM",
        "PLBartForConditionalGeneration",
        "PLBartForSequenceClassification",
        "PLBartModel",
        "PLBartPreTrainedModel",
    ]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 导入配置和模型相关内容
    from .configuration_plbart import PLBART_PRETRAINED_CONFIG_ARCHIVE_MAP, PLBartConfig

    # 检查是否存在 SentencePiece 库，若不可用则引发异常
    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若可用，导入 PLBartTokenizer
        from .tokenization_plbart import PLBartTokenizer

    # 检查是否存在 Torch 库，若不可用则引发异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若可用，导入 PLBart 相关模型
        from .modeling_plbart import (
            PLBART_PRETRAINED_MODEL_ARCHIVE_LIST,
            PLBartForCausalLM,
            PLBartForConditionalGeneration,
            PLBartForSequenceClassification,
            PLBartModel,
            PLBartPreTrainedModel,
        )

# 如果不是类型检查模式
else:
    # 使用懒加载模块来延迟加载依赖模块
    import sys
    # 将当前模块映射到 LazyModule，用以按需导入模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```