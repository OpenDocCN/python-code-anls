# `.\transformers\models\bark\__init__.py`

```py
# 引入必要的模块和函数
from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)

# 定义要导入的结构
_import_structure = {
    "configuration_bark": [
        "BARK_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "BarkCoarseConfig",
        "BarkConfig",
        "BarkFineConfig",
        "BarkSemanticConfig",
    ],
    "processing_bark": ["BarkProcessor"],
}

# 检查是否有 torch 可用
try:
    if not is_torch_available():
        # 如果 torch 不可用，引发 OptionalDependencyNotAvailable 异常
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    # 如果引发了 OptionalDependencyNotAvailable 异常，则不执行以下操作
    pass
else:
    # 如果没有引发异常，则将 modeling_bark 导入到 _import_structure 中
    _import_structure["modeling_bark"] = [
        "BARK_PRETRAINED_MODEL_ARCHIVE_LIST",
        "BarkFineModel",
        "BarkSemanticModel",
        "BarkCoarseModel",
        "BarkModel",
        "BarkPreTrainedModel",
        "BarkCausalModel",
    ]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 从 configuration_bark 模块中导入所需的内容
    from .configuration_bark import (
        BARK_PRETRAINED_CONFIG_ARCHIVE_MAP,
        BarkCoarseConfig,
        BarkConfig,
        BarkFineConfig,
        BarkSemanticConfig,
    )
    # 从 processing_bark 模块中导入所需的内容
    from .processing_bark import BarkProcessor

    try:
        # 再次检查是否有 torch 可用
        if not is_torch_available():
            # 如果 torch 不可用，引发 OptionalDependencyNotAvailable 异常
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        # 如果引发了 OptionalDependencyNotAvailable 异常，则不执行以下操作
        pass
    else:
        # 如果没有引发异常，则从 modeling_bark 模块中导入所需的内容
        from .modeling_bark import (
            BARK_PRETRAINED_MODEL_ARCHIVE_LIST,
            BarkCausalModel,
            BarkCoarseModel,
            BarkFineModel,
            BarkModel,
            BarkPreTrainedModel,
            BarkSemanticModel,
        )

# 如果不是类型检查模式
else:
    # 导入 sys 模块
    import sys

    # 将当前模块重定向为 _LazyModule
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```