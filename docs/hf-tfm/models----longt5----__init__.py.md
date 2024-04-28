# `.\transformers\models\longt5\__init__.py`

```
# 版权声明、许可信息、依赖库检查、模型相关配置和模型导入的主文件

from typing import TYPE_CHECKING  # 导入类型提示模块

from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_flax_available, is_torch_available  # 导入必要的工具函数和模块

# 定义需要导入的模型配置的结构
_import_structure = {
    "configuration_longt5": ["LONGT5_PRETRAINED_CONFIG_ARCHIVE_MAP", "LongT5Config", "LongT5OnnxConfig"],
}

# 检查是否存在 torch 库，若不存在则触发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_longt5"] = [
        "LONGT5_PRETRAINED_MODEL_ARCHIVE_LIST",
        "LongT5EncoderModel",
        "LongT5ForConditionalGeneration",
        "LongT5Model",
        "LongT5PreTrainedModel",
    ]

# 检查是否存在 flax 库，若不存在则触发 OptionalDependencyNotAvailable 异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_longt5"] = [
        "FlaxLongT5ForConditionalGeneration",
        "FlaxLongT5Model",
        "FlaxLongT5PreTrainedModel",
    ]

# 如果是在类型检查模式下，根据不同库的可用性导入相应的模型配置和模型
if TYPE_CHECKING:
    from .configuration_longt5 import LONGT5_PRETRAINED_CONFIG_ARCHIVE_MAP, LongT5Config, LongT5OnnxConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_longt5 import (
            LONGT5_PRETRAINED_MODEL_ARCHIVE_LIST,
            LongT5EncoderModel,
            LongT5ForConditionalGeneration,
            LongT5Model,
            LongT5PreTrainedModel,
        )

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_flax_longt5 import (
            FlaxLongT5ForConditionalGeneration,
            FlaxLongT5Model,
            FlaxLongT5PreTrainedModel,
        )

# 如果不是在类型检查模式下，将当前模块注册为惰性加载模块
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```  
```