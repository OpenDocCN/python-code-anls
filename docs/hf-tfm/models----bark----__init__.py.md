# `.\models\bark\__init__.py`

```
# 导入类型检查模块
from typing import TYPE_CHECKING

# 导入必要的异常和模块
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)

# 定义模块导入结构
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

# 检查是否导入了 torch 模块，如果未导入，则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果导入了 torch 模块，则添加额外的模块到导入结构中
    _import_structure["modeling_bark"] = [
        "BARK_PRETRAINED_MODEL_ARCHIVE_LIST",
        "BarkFineModel",
        "BarkSemanticModel",
        "BarkCoarseModel",
        "BarkModel",
        "BarkPreTrainedModel",
        "BarkCausalModel",
    ]

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 从相关模块导入特定的类或对象
    from .configuration_bark import (
        BARK_PRETRAINED_CONFIG_ARCHIVE_MAP,
        BarkCoarseConfig,
        BarkConfig,
        BarkFineConfig,
        BarkSemanticConfig,
    )
    from .processing_bark import BarkProcessor

    # 再次检查是否导入了 torch 模块，如果未导入，则继续忽略
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果导入了 torch 模块，则从 modeling_bark 模块中导入特定的类或对象
        from .modeling_bark import (
            BARK_PRETRAINED_MODEL_ARCHIVE_LIST,
            BarkCausalModel,
            BarkCoarseModel,
            BarkFineModel,
            BarkModel,
            BarkPreTrainedModel,
            BarkSemanticModel,
        )

# 如果不是类型检查模式，则执行以下操作
else:
    # 导入 sys 模块
    import sys

    # 将当前模块替换为一个懒加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```