# `.\transformers\models\splinter\__init__.py`

```py
# 引入必要的模块和类型检查工具
from typing import TYPE_CHECKING
# 引入自定义工具模块和懒加载模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available, is_torch_available

# 定义模块导入结构，包括配置、分词器和模型
_import_structure = {
    "configuration_splinter": ["SPLINTER_PRETRAINED_CONFIG_ARCHIVE_MAP", "SplinterConfig"],
    "tokenization_splinter": ["SplinterTokenizer"],
}

# 尝试检查分词器是否可用，若不可用则引发异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加快速分词器到导入结构中
    _import_structure["tokenization_splinter_fast"] = ["SplinterTokenizerFast"]

# 尝试检查 PyTorch 是否可用，若不可用则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则添加模型相关内容到导入结构中
    _import_structure["modeling_splinter"] = [
        "SPLINTER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "SplinterForQuestionAnswering",
        "SplinterForPreTraining",
        "SplinterLayer",
        "SplinterModel",
        "SplinterPreTrainedModel",
    ]

# 如果是类型检查环境
if TYPE_CHECKING:
    # 导入配置和分词器
    from .configuration_splinter import SPLINTER_PRETRAINED_CONFIG_ARCHIVE_MAP, SplinterConfig
    from .tokenization_splinter import SplinterTokenizer

    # 尝试检查分词器是否可用，若不可用则引发异常
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，则导入快速分词器
        from .tokenization_splinter_fast import SplinterTokenizerFast

    # 尝试检查 PyTorch 是否可用，若不可用则引发异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，则导入模型相关内容
        from .modeling_splinter import (
            SPLINTER_PRETRAINED_MODEL_ARCHIVE_LIST,
            SplinterForPreTraining,
            SplinterForQuestionAnswering,
            SplinterLayer,
            SplinterModel,
            SplinterPreTrainedModel,
        )

# 如果不是类型检查环境
else:
    import sys

    # 将当前模块设为懒加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```