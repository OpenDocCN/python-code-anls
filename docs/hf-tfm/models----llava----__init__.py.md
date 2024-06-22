# `.\transformers\models\llava\__init__.py`

```py
# 版权声明和许可证信息

# 导入类型检查模块
from typing import TYPE_CHECKING

# 导入自定义的模块和函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义需要导入的结构
_import_structure = {"configuration_llava": ["LLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP", "LlavaConfig"]}

# 如果 Torch 不可用，则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 可用，则添加对应的模型和处理模块到导入结构中
    _import_structure["modeling_llava"] = [
        "LLAVA_PRETRAINED_MODEL_ARCHIVE_LIST",
        "LlavaForConditionalGeneration",
        "LlavaPreTrainedModel",
    ]
    _import_structure["processing_llava"] = ["LlavaProcessor"]

# 如果是类型检查，导入配置和相关模型
if TYPE_CHECKING:
    from .configuration_llava import LLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP, LlavaConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_llava import (
            LLAVA_PRETRAINED_MODEL_ARCHIVE_LIST,
            LlavaForConditionalGeneration,
            LlavaPreTrainedModel,
        )
        from .processing_llava import LlavaProcessor
# 如果不是类型检查，使用懒加载模块
else:
    import sys
    # 设置当前模块为懒加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```