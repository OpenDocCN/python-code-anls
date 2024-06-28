# `.\models\llava\__init__.py`

```
# 导入必要的模块和函数
from typing import TYPE_CHECKING
# 从 utils 模块中导入自定义异常 OptionalDependencyNotAvailable、_LazyModule 和 is_torch_available 函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块的导入结构，包括配置和处理相关的模块和类
_import_structure = {
    "configuration_llava": ["LLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP", "LlavaConfig"],
    "processing_llava": ["LlavaProcessor"],
}

# 尝试检查是否有 torch 可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 torch 可用，则添加 modeling_llava 模块的内容到导入结构中
    _import_structure["modeling_llava"] = [
        "LLAVA_PRETRAINED_MODEL_ARCHIVE_LIST",
        "LlavaForConditionalGeneration",
        "LlavaPreTrainedModel",
    ]

# 如果 TYPE_CHECKING 为 True，即在类型检查环境下
if TYPE_CHECKING:
    # 从 configuration_llava 模块中导入特定的类和变量
    from .configuration_llava import LLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP, LlavaConfig
    # 从 processing_llava 模块中导入特定的类
    from .processing_llava import LlavaProcessor

    # 尝试检查是否有 torch 可用，如果不可用则忽略异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 torch 可用，则从 modeling_llava 模块中导入特定的类和变量
        from .modeling_llava import (
            LLAVA_PRETRAINED_MODEL_ARCHIVE_LIST,
            LlavaForConditionalGeneration,
            LlavaPreTrainedModel,
        )

# 如果不在类型检查环境下
else:
    import sys

    # 将当前模块设置为一个 LazyModule 的实例，延迟加载模块内容
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```