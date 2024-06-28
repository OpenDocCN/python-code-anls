# `.\models\gptsan_japanese\__init__.py`

```
# 导入必要的模块和函数
from typing import TYPE_CHECKING

# 导入可选的异常类和延迟加载模块的帮助函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_torch_available,
)

# 定义模块的导入结构
_import_structure = {
    "configuration_gptsan_japanese": ["GPTSAN_JAPANESE_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPTSanJapaneseConfig"],
    "tokenization_gptsan_japanese": ["GPTSanJapaneseTokenizer"],
}

# 检查是否可用 Torch 库，若不可用则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则扩展导入结构中的模型和标记化模块
    _import_structure["modeling_gptsan_japanese"] = [
        "GPTSAN_JAPANESE_PRETRAINED_MODEL_ARCHIVE_LIST",
        "GPTSanJapaneseForConditionalGeneration",
        "GPTSanJapaneseModel",
        "GPTSanJapanesePreTrainedModel",
    ]
    _import_structure["tokenization_gptsan_japanese"] = [
        "GPTSanJapaneseTokenizer",
    ]

# 如果是类型检查模式，则进行详细的类型导入
if TYPE_CHECKING:
    # 导入配置和标记化模块的特定类
    from .configuration_gptsan_japanese import GPTSAN_JAPANESE_PRETRAINED_CONFIG_ARCHIVE_MAP, GPTSanJapaneseConfig
    from .tokenization_gptsan_japanese import GPTSanJapaneseTokenizer

    # 再次检查 Torch 库是否可用，若不可用则引发异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若可用，则详细导入模型相关的类
        from .modeling_gptsan_japanese import (
            GPTSAN_JAPANESE_PRETRAINED_MODEL_ARCHIVE_LIST,
            GPTSanJapaneseForConditionalGeneration,
            GPTSanJapaneseModel,
            GPTSanJapanesePreTrainedModel,
        )
        from .tokenization_gptsan_japanese import GPTSanJapaneseTokenizer

# 非类型检查模式下，设置模块的延迟加载
else:
    import sys

    # 将当前模块注册为延迟加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```