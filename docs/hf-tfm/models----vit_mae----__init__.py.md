# `.\models\vit_mae\__init__.py`

```py
# 引入依赖类型检查
from typing import TYPE_CHECKING

# 引入内部工具函数和异常类
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_torch_available,
)

# 定义模块导入结构
_import_structure = {"configuration_vit_mae": ["VIT_MAE_PRETRAINED_CONFIG_ARCHIVE_MAP", "ViTMAEConfig"]}

# 检查是否支持 Torch，若不支持则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若支持 Torch，则添加相关模型定义到导入结构中
    _import_structure["modeling_vit_mae"] = [
        "VIT_MAE_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ViTMAEForPreTraining",
        "ViTMAELayer",
        "ViTMAEModel",
        "ViTMAEPreTrainedModel",
    ]

# 检查是否支持 TensorFlow，若不支持则抛出异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若支持 TensorFlow，则添加相关模型定义到导入结构中
    _import_structure["modeling_tf_vit_mae"] = [
        "TFViTMAEForPreTraining",
        "TFViTMAEModel",
        "TFViTMAEPreTrainedModel",
    ]

# 如果处于类型检查模式
if TYPE_CHECKING:
    # 从特定模块导入配置和模型定义
    from .configuration_vit_mae import VIT_MAE_PRETRAINED_CONFIG_ARCHIVE_MAP, ViTMAEConfig

    try:
        # 再次检查是否支持 Torch，若不支持则抛出异常
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若支持 Torch，则从模型定义中导入相关类
        from .modeling_vit_mae import (
            VIT_MAE_PRETRAINED_MODEL_ARCHIVE_LIST,
            ViTMAEForPreTraining,
            ViTMAELayer,
            ViTMAEModel,
            ViTMAEPreTrainedModel,
        )

    try:
        # 再次检查是否支持 TensorFlow，若不支持则抛出异常
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若支持 TensorFlow，则从模型定义中导入相关类
        from .modeling_tf_vit_mae import TFViTMAEForPreTraining, TFViTMAEModel, TFViTMAEPreTrainedModel

# 如果不处于类型检查模式
else:
    # 动态创建一个懒加载模块，用于按需导入所需的模型和配置
    import sys
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```