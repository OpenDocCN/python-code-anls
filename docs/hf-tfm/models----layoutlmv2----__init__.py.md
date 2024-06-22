# `.\models\layoutlmv2\__init__.py`

```py
# 导入所需的模块和函数
from typing import TYPE_CHECKING
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tokenizers_available,
    is_torch_available,
    is_vision_available,
)

# 定义需要导入的模块和函数结构
_import_structure = {
    "configuration_layoutlmv2": ["LAYOUTLMV2_PRETRAINED_CONFIG_ARCHIVE_MAP", "LayoutLMv2Config"],
    "processing_layoutlmv2": ["LayoutLMv2Processor"],
    "tokenization_layoutlmv2": ["LayoutLMv2Tokenizer"],
}

# 检查是否存在 tokenizers，如果不存在则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 tokenizers，则添加 LayoutLMv2TokenizerFast 到导入结构中
    _import_structure["tokenization_layoutlmv2_fast"] = ["LayoutLMv2TokenizerFast"]

# 检查是否存在 vision，如果不存在则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 vision，则添加 LayoutLMv2FeatureExtractor 和 LayoutLMv2ImageProcessor 到导入结构中
    _import_structure["feature_extraction_layoutlmv2"] = ["LayoutLMv2FeatureExtractor"]
    _import_structure["image_processing_layoutlmv2"] = ["LayoutLMv2ImageProcessor"]

# 检查是否存在 torch，如果不存在则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在 torch，则添加 LayoutLMv2 相关模型到导入结构中
    _import_structure["modeling_layoutlmv2"] = [
        "LAYOUTLMV2_PRETRAINED_MODEL_ARCHIVE_LIST",
        "LayoutLMv2ForQuestionAnswering",
        "LayoutLMv2ForSequenceClassification",
        "LayoutLMv2ForTokenClassification",
        "LayoutLMv2Layer",
        "LayoutLMv2Model",
        "LayoutLMv2PreTrainedModel",
    ]

# 如果是类型检查阶段，则从各个子模块导入相应的内容
if TYPE_CHECKING:
    from .configuration_layoutlmv2 import LAYOUTLMV2_PRETRAINED_CONFIG_ARCHIVE_MAP, LayoutLMv2Config
    from .processing_layoutlmv2 import LayoutLMv2Processor
    from .tokenization_layoutlmv2 import LayoutLMv2Tokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_layoutlmv2_fast import LayoutLMv2TokenizerFast

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .feature_extraction_layoutlmv2 import LayoutLMv2FeatureExtractor, LayoutLMv2ImageProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入布局感知多语言模型v2相关模块
        from .modeling_layoutlmv2 import (
            # 导入预训练模型的存档列表
            LAYOUTLMV2_PRETRAINED_MODEL_ARCHIVE_LIST,
            # 导入用于问答任务的布局感知多语言模型v2
            LayoutLMv2ForQuestionAnswering,
            # 导入用于序列分类任务的布局感知多语言模型v2
            LayoutLMv2ForSequenceClassification,
            # 导入用于标记分类任务的布局感知多语言模型v2
            LayoutLMv2ForTokenClassification,
            # 导入布局感知多语言模型v2的层
            LayoutLMv2Layer,
            # 导入布局感知多语言模型v2
            LayoutLMv2Model,
            # 导入布局感知多语言模型v2的预训练模型
            LayoutLMv2PreTrainedModel,
        )
# 如果前面的条件均不满足，即不在 main 模块，则执行以下代码

# 导入 sys 模块
import sys

# 将当前模块（__name__）的属性设置为 _LazyModule 类的实例，其中包括模块的名称、文件名、导入结构和模块规范
sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```