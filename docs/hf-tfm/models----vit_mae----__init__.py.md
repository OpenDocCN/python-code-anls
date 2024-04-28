# `.\transformers\models\vit_mae\__init__.py`

```
# 保留版权声明

# 引入必要的依赖
from typing import TYPE_CHECKING
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_torch_available,
)

# 定义需要导入的结构
_import_structure = {"configuration_vit_mae": ["VIT_MAE_PRETRAINED_CONFIG_ARCHIVE_MAP", "ViTMAEConfig"]}

# 检查是否有 torch 可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 导入 torch 模型相关结构
    _import_structure["modeling_vit_mae"] = [
        "VIT_MAE_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ViTMAEForPreTraining",
        "ViTMAELayer",
        "ViTMAEModel",
        "ViTMAEPreTrainedModel",
    ]

# 检查是否有 tensorflow 可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 导入 tensorflow 模型相关结构
    _import_structure["modeling_tf_vit_mae"] = [
        "TFViTMAEForPreTraining",
        "TFViTMAEModel",
        "TFViTMAEPreTrainedModel",
    ]

# 检查是否为类型检查阶段，如果是则导入类型检查所需的结构
if TYPE_CHECKING:
    from .configuration_vit_mae import VIT_MAE_PRETRAINED_CONFIG_ARCHIVE_MAP, ViTMAEConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_vit_mae import (
            VIT_MAE_PRETRAINED_MODEL_ARCHIVE_LIST,
            ViTMAEForPreTraining,
            ViTMAELayer,
            ViTMAEModel,
            ViTMAEPreTrainedModel,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_vit_mae import TFViTMAEForPreTraining, TFViTMAEModel, TFViTMAEPreTrainedModel

# 如果不是类型检查，将模块设为懒加载模块
else:
    import sys
    # 创建懒加载模块，添加必要的结构，并设定模块规范
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```