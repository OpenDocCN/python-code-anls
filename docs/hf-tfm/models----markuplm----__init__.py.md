# `.\models\markuplm\__init__.py`

```
# 版权声明和许可信息
#
# 版权所有 2022 年 HuggingFace 团队。保留所有权利。
# 
# 根据 Apache 许可证 2.0 版本（“许可证”）进行许可；
# 您只能在遵守许可证的情况下使用此文件。
# 您可以在以下网址获取许可证副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律要求或书面同意，软件
# 是基于“按原样提供”的基础上分发的，
# 没有任何明示或暗示的担保或条件。
# 有关详细信息，请参阅许可证。
from typing import TYPE_CHECKING

# 从 utils 模块导入所需的依赖和函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available, is_torch_available

# 定义导入结构
_import_structure = {
    "configuration_markuplm": ["MARKUPLM_PRETRAINED_CONFIG_ARCHIVE_MAP", "MarkupLMConfig"],
    "feature_extraction_markuplm": ["MarkupLMFeatureExtractor"],
    "processing_markuplm": ["MarkupLMProcessor"],
    "tokenization_markuplm": ["MarkupLMTokenizer"],
}

# 检查是否 tokenizers 可用，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则添加快速 tokenization 的导入
    _import_structure["tokenization_markuplm_fast"] = ["MarkupLMTokenizerFast"]

# 检查是否 torch 可用，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若可用，则添加 modeling 相关的导入
    _import_structure["modeling_markuplm"] = [
        "MARKUPLM_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MarkupLMForQuestionAnswering",
        "MarkupLMForSequenceClassification",
        "MarkupLMForTokenClassification",
        "MarkupLMModel",
        "MarkupLMPreTrainedModel",
    ]

# 如果是类型检查阶段，则从各模块导入特定类和常量
if TYPE_CHECKING:
    from .configuration_markuplm import MARKUPLM_PRETRAINED_CONFIG_ARCHIVE_MAP, MarkupLMConfig
    from .feature_extraction_markuplm import MarkupLMFeatureExtractor
    from .processing_markuplm import MarkupLMProcessor
    from .tokenization_markuplm import MarkupLMTokenizer

    # 类型检查下，如果 tokenizers 可用，则导入 fast tokenization
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_markuplm_fast import MarkupLMTokenizerFast

    # 类型检查下，如果 torch 可用，则导入 modeling 相关模块
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

# 如果不是类型检查阶段，则使用 _LazyModule 进行懒加载导入
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```