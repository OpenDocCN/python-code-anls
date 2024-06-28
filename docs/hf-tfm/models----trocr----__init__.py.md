# `.\models\trocr\__init__.py`

```
# 导入必要的类型检查模块
from typing import TYPE_CHECKING

# 导入必要的依赖项和函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_speech_available,
    is_torch_available,
)

# 定义模块导入结构
_import_structure = {
    "configuration_trocr": ["TROCR_PRETRAINED_CONFIG_ARCHIVE_MAP", "TrOCRConfig"],
    "processing_trocr": ["TrOCRProcessor"],
}

# 检查是否可以使用 Torch，若不可用则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 可用，添加相关模块到导入结构中
    _import_structure["modeling_trocr"] = [
        "TROCR_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TrOCRForCausalLM",
        "TrOCRPreTrainedModel",
    ]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 导入类型检查所需的配置和处理模块
    from .configuration_trocr import TROCR_PRETRAINED_CONFIG_ARCHIVE_MAP, TrOCRConfig
    from .processing_trocr import TrOCRProcessor

    # 再次检查 Torch 是否可用，若不可用则引发异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入类型检查所需的模型处理模块
        from .modeling_trocr import TROCR_PRETRAINED_MODEL_ARCHIVE_LIST, TrOCRForCausalLM, TrOCRPreTrainedModel

# 如果不是类型检查模式
else:
    import sys

    # 将当前模块注册为 LazyModule，并使用指定的导入结构
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```