# `.\transformers\models\seamless_m4t_v2\__init__.py`

```py
# 导入必要的模块和函数
from typing import TYPE_CHECKING
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)

# 定义需要导入的结构
_import_structure = {
    "configuration_seamless_m4t_v2": ["SEAMLESS_M4T_V2_PRETRAINED_CONFIG_ARCHIVE_MAP", "SeamlessM4Tv2Config"],
}

# 检查是否需要导入 torch 模块，如果不可用则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 torch 可用则添加 modeling_seamless_m4t_v2 模块及其下的内容到 _import_structure
    _import_structure["modeling_seamless_m4t_v2"] = [
        "SEAMLESS_M4T_V2_PRETRAINED_MODEL_ARCHIVE_LIST",
        "SeamlessM4Tv2ForTextToSpeech",
        "SeamlessM4Tv2ForSpeechToSpeech",
        "SeamlessM4Tv2ForTextToText",
        "SeamlessM4Tv2ForSpeechToText",
        "SeamlessM4Tv2Model",
        "SeamlessM4Tv2PreTrainedModel",
    ]

# 如果需要类型检查，则进行相应的模块导入
if TYPE_CHECKING:
    from .configuration_seamless_m4t_v2 import SEAMLESS_M4T_V2_PRETRAINED_CONFIG_ARCHIVE_MAP, SeamlessM4Tv2Config

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_seamless_m4t_v2 import (
            SEAMLESS_M4T_V2_PRETRAINED_MODEL_ARCHIVE_LIST,
            SeamlessM4Tv2ForSpeechToSpeech,
            SeamlessM4Tv2ForSpeechToText,
            SeamlessM4Tv2ForTextToSpeech,
            SeamlessM4Tv2ForTextToText,
            SeamlessM4Tv2Model,
            SeamlessM4Tv2PreTrainedModel,
        )

# 如果不需要类型检查，则将 LazyModule 对象绑定到当前模块
else:
    import sys
    # 创建 LazyModule 对象，并将其绑定到当前模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```