# `.\models\gpt_neox_japanese\__init__.py`

```py
# 导入必要的类型检查模块
from typing import TYPE_CHECKING

# 导入懒加载模块和条件判断函数
from ...file_utils import _LazyModule, is_torch_available
from ...utils import OptionalDependencyNotAvailable

# 定义模块的导入结构
_import_structure = {
    "configuration_gpt_neox_japanese": ["GPT_NEOX_JAPANESE_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPTNeoXJapaneseConfig"],
    "tokenization_gpt_neox_japanese": ["GPTNeoXJapaneseTokenizer"],
}

# 尝试检查是否导入了 Torch，若未导入则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若成功导入 Torch，则添加额外的模型相关导入结构
    _import_structure["modeling_gpt_neox_japanese"] = [
        "GPT_NEOX_JAPANESE_PRETRAINED_MODEL_ARCHIVE_LIST",
        "GPTNeoXJapaneseForCausalLM",
        "GPTNeoXJapaneseLayer",
        "GPTNeoXJapaneseModel",
        "GPTNeoXJapanesePreTrainedModel",
    ]

# 如果类型检查开启，导入相应的类型和模块
if TYPE_CHECKING:
    from .configuration_gpt_neox_japanese import GPT_NEOX_JAPANESE_PRETRAINED_CONFIG_ARCHIVE_MAP, GPTNeoXJapaneseConfig
    from .tokenization_gpt_neox_japanese import GPTNeoXJapaneseTokenizer

    # 同样地，尝试检查是否导入了 Torch，若未导入则抛出异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若成功导入 Torch，则导入模型相关的类型和模块
        from .modeling_gpt_neox_japanese import (
            GPT_NEOX_JAPANESE_PRETRAINED_MODEL_ARCHIVE_LIST,
            GPTNeoXJapaneseForCausalLM,
            GPTNeoXJapaneseLayer,
            GPTNeoXJapaneseModel,
            GPTNeoXJapanesePreTrainedModel,
        )

# 若非类型检查模式，则直接将当前模块设置为懒加载模式
else:
    import sys

    # 动态设置当前模块为懒加载模式，使用 _LazyModule 进行懒加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```