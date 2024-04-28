# `.\models\gpt_neox_japanese\__init__.py`

```py
# 导入 TYPE_CHECKING 模块，用于类型检查
from typing import TYPE_CHECKING

# 导入 LazyModule 类和 is_torch_available 函数，LazyModule 是一个延迟加载模块的类，is_torch_available 用于检查是否安装了 Torch
from ...file_utils import _LazyModule, is_torch_available
# 导入 OptionalDependencyNotAvailable 异常类
from ...utils import OptionalDependencyNotAvailable

# 定义需要导入的模块和对应的成员结构的字典
_import_structure = {
    "configuration_gpt_neox_japanese": ["GPT_NEOX_JAPANESE_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPTNeoXJapaneseConfig"],
    "tokenization_gpt_neox_japanese": ["GPTNeoXJapaneseTokenizer"],
}

# 尝试检查是否安装了 Torch，如果没有安装则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果安装了 Torch，则添加 modeling_gpt_neox_japanese 模块到导入结构中
    _import_structure["modeling_gpt_neox_japanese"] = [
        "GPT_NEOX_JAPANESE_PRETRAINED_MODEL_ARCHIVE_LIST",
        "GPTNeoXJapaneseForCausalLM",
        "GPTNeoXJapaneseLayer",
        "GPTNeoXJapaneseModel",
        "GPTNeoXJapanesePreTrainedModel",
    ]

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 导入 configuration_gpt_neox_japanese 模块的特定成员
    from .configuration_gpt_neox_japanese import GPT_NEOX_JAPANESE_PRETRAINED_CONFIG_ARCHIVE_MAP, GPTNeoXJapaneseConfig
    # 导入 tokenization_gpt_neox_japanese 模块的特定成员
    from .tokenization_gpt_neox_japanese import GPTNeoXJapaneseTokenizer

    # 尝试检查是否安装了 Torch，如果没有安装则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果安装了 Torch，则导入 modeling_gpt_neox_japanese 模块的特定成员
        from .modeling_gpt_neox_japanese import (
            GPT_NEOX_JAPANESE_PRETRAINED_MODEL_ARCHIVE_LIST,
            GPTNeoXJapaneseForCausalLM,
            GPTNeoXJapaneseLayer,
            GPTNeoXJapaneseModel,
            GPTNeoXJapanesePreTrainedModel,
        )

# 如果不在类型检查模式下
else:
    # 导入 sys 模块
    import sys

    # 将当前模块替换为 LazyModule 的实例，使用 LazyModule 可以延迟加载模块，提高性能
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)

``` 
```