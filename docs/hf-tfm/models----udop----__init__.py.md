# `.\models\udop\__init__.py`

```
# 版权声明和许可证声明，指出此文件受版权保护，并遵循Apache License 2.0
#
# from...utils中导入所需的模块和函数
from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_tokenizers_available,
    is_torch_available,
)

# 定义导入结构，用于延迟导入模块和函数
_import_structure = {
    "configuration_udop": ["UDOP_PRETRAINED_CONFIG_ARCHIVE_MAP", "UdopConfig"],
    "processing_udop": ["UdopProcessor"],
}

# 尝试导入句子分词器，若不可用则引发OptionalDependencyNotAvailable异常
try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_udop"] = ["UdopTokenizer"]

# 尝试导入tokenizers，若不可用则引发OptionalDependencyNotAvailable异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_udop_fast"] = ["UdopTokenizerFast"]

# 尝试导入torch，若不可用则引发OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_udop"] = [
        "UDOP_PRETRAINED_MODEL_ARCHIVE_LIST",
        "UdopForConditionalGeneration",
        "UdopPreTrainedModel",
        "UdopModel",
        "UdopEncoderModel",
    ]

# 如果在类型检查模式下，导入具体模块和类
if TYPE_CHECKING:
    from .configuration_udop import UDOP_PRETRAINED_CONFIG_ARCHIVE_MAP, UdopConfig
    from .processing_udop import UdopProcessor

    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_udop import UdopTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_udop_fast import UdopTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_udop import (
            UDOP_PRETRAINED_MODEL_ARCHIVE_LIST,
            UdopEncoderModel,
            UdopForConditionalGeneration,
            UdopModel,
            UdopPreTrainedModel,
        )

# 若不是类型检查模式，将该模块设置为LazyModule的实例，以支持延迟加载
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```