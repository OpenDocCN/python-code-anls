# `.\models\gptsan_japanese\__init__.py`

```
# 版权声明及许可信息，说明本代码的版权及使用许可
from typing import TYPE_CHECKING  # 导入类型检查模块

from ...utils import (  # 导入工具函数
    OptionalDependencyNotAvailable,  # 导入可选依赖不存在异常
    _LazyModule,  # 导入懒加载模块
    is_flax_available,  # 导入flax是否可用的函数
    is_tf_available,  # 导入tf是否可用的函数
    is_torch_available,  # 导入torch是否可用的函数
)

# 导入结构定义，包括configuration_gptsan_japanese和tokenization_gptsan_japanese
_import_structure = {
    "configuration_gptsan_japanese": ["GPTSAN_JAPANESE_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPTSanJapaneseConfig"],
    "tokenization_gptsan_japanese": ["GPTSanJapaneseTokenizer"],
}

# 检查torch是否可用，若不可用则引发OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:  # 若torch可用
    # 添加modeling_gptsan_japanese和tokenization_gptsan_japanese至import_structure
    _import_structure["modeling_gptsan_japanese"] = [
        "GPTSAN_JAPANESE_PRETRAINED_MODEL_ARCHIVE_LIST",
        "GPTSanJapaneseForConditionalGeneration",
        "GPTSanJapaneseModel",
        "GPTSanJapanesePreTrainedModel",
    ]
    _import_structure["tokenization_gptsan_japanese"] = [
        "GPTSanJapaneseTokenizer",
    ]

# 类型检查时的导入
if TYPE_CHECKING:
    from .configuration_gptsan_japanese import GPTSAN_JAPANESE_PRETRAINED_CONFIG_ARCHIVE_MAP, GPTSanJapaneseConfig
    from .tokenization_gptsan_japanese import GPTSanJapaneseTokenizer

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_gptsan_japanese import (
            GPTSAN_JAPANESE_PRETRAINED_MODEL_ARCHIVE_LIST,
            GPTSanJapaneseForConditionalGeneration,
            GPTSanJapaneseModel,
            GPTSanJapanesePreTrainedModel,
        )
        from .tokenization_gptsan_japanese import GPTSanJapaneseTokenizer


else:  # 非类型检查时
    import sys  # 导入sys模块

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)  # 设置模块对象
```