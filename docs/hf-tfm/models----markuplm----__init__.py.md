# `.\transformers\models\markuplm\__init__.py`

```
# 导入类型检查模块
from typing import TYPE_CHECKING
# 从utils中导入相关模块和变量
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available, is_torch_available

# 定义模块导入结构
_import_structure = {
    "configuration_markuplm": ["MARKUPLM_PRETRAINED_CONFIG_ARCHIVE_MAP", "MarkupLMConfig"],
    "feature_extraction_markuplm": ["MarkupLMFeatureExtractor"],
    "processing_markuplm": ["MarkupLMProcessor"],
    "tokenization_markuplm": ["MarkupLMTokenizer"],
}

# 尝试导入tokenizer模块，如果不可用则抛出异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_markuplm_fast"] = ["MarkupLMTokenizerFast"]

# 尝试导入torch模块，如果不可用则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_markuplm"] = [
        "MARKUPLM_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MarkupLMForQuestionAnswering",
        "MarkupLMForSequenceClassification",
        "MarkupLMForTokenClassification",
        "MarkupLMModel",
        "MarkupLMPreTrainedModel",
    ]

# 如果是类型检查模式，则进行特定模块的导入
if TYPE_CHECKING:
    from .configuration_markuplm import MARKUPLM_PRETRAINED_CONFIG_ARCHIVE_MAP, MarkupLMConfig
    from .feature_extraction_markuplm import MarkupLMFeatureExtractor
    from .processing_markuplm import MarkupLMProcessor
    from .tokenization_markuplm import MarkupLMTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_markuplm_fast import MarkupLMTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_markuplm import (
            MARKUPLM_PRETRAINED_MODEL_ARCHIVE_LIST,
            MarkupLMForQuestionAnswering,
            MarkupLMForSequenceClassification,
            MarkupLMForTokenClassification,
            MarkupLMModel,
            MarkupLMPreTrainedModel,
        )
# 否则，使用懒加载模块方法进行导入
else:
    import sys
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```