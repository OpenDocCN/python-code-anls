# `.\models\layoutlmv2\__init__.py`

```py
# 导入必要的模块和函数
from typing import TYPE_CHECKING
# 从 utils 中导入相关函数和异常类
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tokenizers_available,
    is_torch_available,
    is_vision_available,
)

# 定义模块导入结构
_import_structure = {
    "configuration_layoutlmv2": ["LAYOUTLMV2_PRETRAINED_CONFIG_ARCHIVE_MAP", "LayoutLMv2Config"],
    "processing_layoutlmv2": ["LayoutLMv2Processor"],
    "tokenization_layoutlmv2": ["LayoutLMv2Tokenizer"],
}

# 检查是否可用 tokenizers 库，如果不可用则引发异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加 tokenization_layoutlmv2 到导入结构中
    _import_structure["tokenization_layoutlmv2_fast"] = ["LayoutLMv2TokenizerFast"]

# 检查是否可用 vision 库，如果不可用则引发异常
try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加 feature_extraction_layoutlmv2 和 image_processing_layoutlmv2 到导入结构中
    _import_structure["feature_extraction_layoutlmv2"] = ["LayoutLMv2FeatureExtractor"]
    _import_structure["image_processing_layoutlmv2"] = ["LayoutLMv2ImageProcessor"]

# 检查是否可用 torch 库，如果不可用则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加 modeling_layoutlmv2 到导入结构中
    _import_structure["modeling_layoutlmv2"] = [
        "LAYOUTLMV2_PRETRAINED_MODEL_ARCHIVE_LIST",
        "LayoutLMv2ForQuestionAnswering",
        "LayoutLMv2ForSequenceClassification",
        "LayoutLMv2ForTokenClassification",
        "LayoutLMv2Layer",
        "LayoutLMv2Model",
        "LayoutLMv2PreTrainedModel",
    ]

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 从相应的模块中导入特定的类和变量
    from .configuration_layoutlmv2 import LAYOUTLMV2_PRETRAINED_CONFIG_ARCHIVE_MAP, LayoutLMv2Config
    from .processing_layoutlmv2 import LayoutLMv2Processor
    from .tokenization_layoutlmv2 import LayoutLMv2Tokenizer

    try:
        # 再次检查 tokenizers 库是否可用
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，则导入 LayoutLMv2TokenizerFast
        from .tokenization_layoutlmv2_fast import LayoutLMv2TokenizerFast

    try:
        # 再次检查 vision 库是否可用
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，则导入 LayoutLMv2FeatureExtractor 和 LayoutLMv2ImageProcessor
        from .feature_extraction_layoutlmv2 import LayoutLMv2FeatureExtractor, LayoutLMv2ImageProcessor

    try:
        # 再次检查 torch 库是否可用
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 从当前目录下的 `modeling_layoutlmv2` 模块中导入多个类和常量
        from .modeling_layoutlmv2 import (
            LAYOUTLMV2_PRETRAINED_MODEL_ARCHIVE_LIST,
            LayoutLMv2ForQuestionAnswering,
            LayoutLMv2ForSequenceClassification,
            LayoutLMv2ForTokenClassification,
            LayoutLMv2Layer,
            LayoutLMv2Model,
            LayoutLMv2PreTrainedModel,
        )
else:
    # 导入 sys 模块，用于操作 Python 解释器的运行时环境
    import sys

    # 将当前模块(__name__)的引用映射到一个自定义的 _LazyModule 对象上，
    # 并传入当前模块的名称、文件路径、导入结构 _import_structure，
    # 同时传入模块规范 __spec__（如果有的话）来指定模块的详细规范信息。
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```