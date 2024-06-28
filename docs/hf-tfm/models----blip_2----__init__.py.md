# `.\models\blip_2\__init__.py`

```py
# 引入必要的模块和类型检查
from typing import TYPE_CHECKING
# 引入自定义的异常，用于处理可选依赖不可用情况下的异常
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块的导入结构，包括配置和处理模块的名称列表
_import_structure = {
    "configuration_blip_2": [
        "BLIP_2_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "Blip2Config",
        "Blip2QFormerConfig",
        "Blip2VisionConfig",
    ],
    "processing_blip_2": ["Blip2Processor"],
}

# 尝试检查是否存在 Torch 库，如果不存在则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 可用，则添加建模模块的名称列表到导入结构中
    _import_structure["modeling_blip_2"] = [
        "BLIP_2_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Blip2Model",
        "Blip2QFormerModel",
        "Blip2PreTrainedModel",
        "Blip2ForConditionalGeneration",
        "Blip2VisionModel",
    ]

# 如果类型检查开启，则从配置和处理模块中导入相应的类
if TYPE_CHECKING:
    from .configuration_blip_2 import (
        BLIP_2_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Blip2Config,
        Blip2QFormerConfig,
        Blip2VisionConfig,
    )
    from .processing_blip_2 import Blip2Processor

    # 尝试检查 Torch 是否可用，不可用则跳过导入建模模块的步骤
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_blip_2 import (
            BLIP_2_PRETRAINED_MODEL_ARCHIVE_LIST,
            Blip2ForConditionalGeneration,
            Blip2Model,
            Blip2PreTrainedModel,
            Blip2QFormerModel,
            Blip2VisionModel,
        )

else:
    # 如果没有类型检查，则将当前模块注册为 LazyModule，将导入结构传递给 LazyModule
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```