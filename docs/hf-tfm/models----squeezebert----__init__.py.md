# `.\models\squeezebert\__init__.py`

```py
# 版权声明和许可信息，说明该代码受 Apache License, Version 2.0 版权保护
#
# 如果符合许可证要求，可以使用本文件；否则，不得使用
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非法律另有规定或书面同意，否则本软件按“原样”分发，不附带任何明示或暗示的保证或条件
# 详见许可证了解更多信息
#

# 引入类型检查
from typing import TYPE_CHECKING

# 引入必要的依赖和模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available, is_torch_available

# 定义要导入的结构
_import_structure = {
    "configuration_squeezebert": [
        "SQUEEZEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "SqueezeBertConfig",
        "SqueezeBertOnnxConfig",
    ],
    "tokenization_squeezebert": ["SqueezeBertTokenizer"],
}

# 检查是否有 tokenizers 库可用，如果不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，将 tokenization_squeezebert_fast 模块添加到导入结构中
    _import_structure["tokenization_squeezebert_fast"] = ["SqueezeBertTokenizerFast"]

# 检查是否有 torch 库可用，如果不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，将 modeling_squeezebert 模块添加到导入结构中
    _import_structure["modeling_squeezebert"] = [
        "SQUEEZEBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "SqueezeBertForMaskedLM",
        "SqueezeBertForMultipleChoice",
        "SqueezeBertForQuestionAnswering",
        "SqueezeBertForSequenceClassification",
        "SqueezeBertForTokenClassification",
        "SqueezeBertModel",
        "SqueezeBertModule",
        "SqueezeBertPreTrainedModel",
    ]

# 如果是类型检查模式，引入相关模块
if TYPE_CHECKING:
    from .configuration_squeezebert import (
        SQUEEZEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        SqueezeBertConfig,
        SqueezeBertOnnxConfig,
    )
    from .tokenization_squeezebert import SqueezeBertTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_squeezebert_fast import SqueezeBertTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_squeezebert import (
            SQUEEZEBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            SqueezeBertForMaskedLM,
            SqueezeBertForMultipleChoice,
            SqueezeBertForQuestionAnswering,
            SqueezeBertForSequenceClassification,
            SqueezeBertForTokenClassification,
            SqueezeBertModel,
            SqueezeBertModule,
            SqueezeBertPreTrainedModel,
        )

# 如果不是类型检查模式，将当前模块设置为延迟加载模块
else:
    import sys

    # 将当前模块替换为延迟加载模块，其中包括导入结构和当前文件的信息
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```