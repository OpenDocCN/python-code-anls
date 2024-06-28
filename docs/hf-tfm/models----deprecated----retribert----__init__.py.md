# `.\models\deprecated\retribert\__init__.py`

```
# 导入必要的模块和函数
from typing import TYPE_CHECKING
# 从工具包中导入必要的异常和延迟加载模块
from ....utils import OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available, is_torch_available

# 定义模块的导入结构
_import_structure = {
    "configuration_retribert": ["RETRIBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "RetriBertConfig"],
    "tokenization_retribert": ["RetriBertTokenizer"],
}

# 检查是否存在 tokenizers 库，若不存在则抛出异常
try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若存在则添加快速 tokenization_retribert_fast 模块的导入结构
    _import_structure["tokenization_retribert_fast"] = ["RetriBertTokenizerFast"]

# 检查是否存在 torch 库，若不存在则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若存在则添加 modeling_retribert 模块的导入结构
    _import_structure["modeling_retribert"] = [
        "RETRIBERT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "RetriBertModel",
        "RetriBertPreTrainedModel",
    ]

# 如果是类型检查阶段
if TYPE_CHECKING:
    # 从相关模块导入必要的类和变量
    from .configuration_retribert import RETRIBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, RetriBertConfig
    from .tokenization_retribert import RetriBertTokenizer

    # 再次检查 tokenizers 库是否可用
    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若可用，则从 tokenization_retribert_fast 导入 RetriBertTokenizerFast 类
        from .tokenization_retribert_fast import RetriBertTokenizerFast

    # 再次检查 torch 库是否可用
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若可用，则从 modeling_retribert 导入相关类和变量
        from .modeling_retribert import (
            RETRIBERT_PRETRAINED_MODEL_ARCHIVE_LIST,
            RetriBertModel,
            RetriBertPreTrainedModel,
        )

# 如果不是类型检查阶段
else:
    # 导入 sys 模块
    import sys

    # 将当前模块替换为延迟加载模块 _LazyModule
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```