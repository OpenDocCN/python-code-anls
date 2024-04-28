# `.\models\layoutlmv3\__init__.py`

```
# 版权声明及许可协议信息

# 引入类型检查模块
from typing import TYPE_CHECKING

# 引入必要的依赖模块
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
    is_vision_available,
)

# 定义模块导入结构
_import_structure = {
    "configuration_layoutlmv3": [
        "LAYOUTLMV3_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "LayoutLMv3Config",
        "LayoutLMv3OnnxConfig",
    ],
    "processing_layoutlmv3": ["LayoutLMv3Processor"],
    "tokenization_layoutlmv3": ["LayoutLMv3Tokenizer"],
}

# 检查 tokenizers 包是否可用，若不可用则抛出异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_layoutlmv3_fast"] = ["LayoutLMv3TokenizerFast"]

# 检查 torch 包是否可用，若不可用则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_layoutlmv3"] = [
        "LAYOUTLMV3_PRETRAINED_MODEL_ARCHIVE_LIST",
        "LayoutLMv3ForQuestionAnswering",
        "LayoutLMv3ForSequenceClassification",
        "LayoutLMv3ForTokenClassification",
        "LayoutLMv3Model",
        "LayoutLMv3PreTrainedModel",
    ]

# 检查 tensorflow 包是否可用，若不可用则抛出异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_layoutlmv3"] = [
        "TF_LAYOUTLMV3_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFLayoutLMv3ForQuestionAnswering",
        "TFLayoutLMv3ForSequenceClassification",
        "TFLayoutLMv3ForTokenClassification",
        "TFLayoutLMv3Model",
        "TFLayoutLMv3PreTrainedModel",
    ]

# 检查 vision 包是否可用，若不可用则抛出异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["feature_extraction_layoutlmv3"] = ["LayoutLMv3FeatureExtractor"]
    _import_structure["image_processing_layoutlmv3"] = ["LayoutLMv3ImageProcessor"]

# 类型检查时导入特定模块
if TYPE_CHECKING:
    from .configuration_layoutlmv3 import (
        LAYOUTLMV3_PRETRAINED_CONFIG_ARCHIVE_MAP,
        LayoutLMv3Config,
        LayoutLMv3OnnxConfig,
    )
    from .processing_layoutlmv3 import LayoutLMv3Processor
    from .tokenization_layoutlmv3 import LayoutLMv3Tokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    # 尝
# 否则，如果不满足之前的条件，则导入 sys 模块
import sys
# 将当前模块添加到 sys 模块的 modules 字典中，使用 _LazyModule 对象对当前模块进行封装
sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```