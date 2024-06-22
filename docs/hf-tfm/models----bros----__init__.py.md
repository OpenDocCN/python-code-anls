# `.\transformers\models\bros\__init__.py`

```py
# 导入 TYPE_CHECKING 模块，用于类型检查
from typing import TYPE_CHECKING
# 导入 OptionalDependencyNotAvailable 异常和 _LazyModule，以及 is_tokenizers_available 和 is_torch_available 函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available, is_torch_available

# 定义需要导入的模块结构
_import_structure = {
    "configuration_bros": ["BROS_PRETRAINED_CONFIG_ARCHIVE_MAP", "BrosConfig"],  # 导入配置相关的模块
}

# 检查是否 tokenizers 可用，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass  # 若抛出异常，则忽略

# 如果 tokenizers 可用，则导入 processing_bros 模块
else:
    _import_structure["processing_bros"] = ["BrosProcessor"]

# 检查是否 torch 可用，若不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass  # 若抛出异常，则忽略

# 如果 torch 可用，则导入 modeling_bros 模块
else:
    _import_structure["modeling_bros"] = [
        "BROS_PRETRAINED_MODEL_ARCHIVE_LIST",
        "BrosPreTrainedModel",
        "BrosModel",
        "BrosForTokenClassification",
        "BrosSpadeEEForTokenClassification",
        "BrosSpadeELForTokenClassification",
    ]

# 如果是类型检查，则导入相应的模块
if TYPE_CHECKING:
    from .configuration_bros import BROS_PRETRAINED_CONFIG_ARCHIVE_MAP, BrosConfig  # 导入配置相关的模块

    # 检查 tokenizers 是否可用，若可用则导入 processing_bros 模块
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass  # 若抛出异常，则忽略
    else:
        from .processing_bros import BrosProcessor

    # 检查 torch 是否可用，若可用则导入 modeling_bros 模块
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass  # 若抛出异常，则忽略
    else:
        from .modeling_bros import (
            BROS_PRETRAINED_MODEL_ARCHIVE_LIST,
            BrosForTokenClassification,
            BrosModel,
            BrosPreTrainedModel,
            BrosSpadeEEForTokenClassification,
            BrosSpadeELForTokenClassification,
        )

# 若不是类型检查，则将当前模块设置为 LazyModule
else:
    import sys

    # 将当前模块设置为 LazyModule
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```