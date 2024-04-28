# `.\transformers\models\resnet\__init__.py`

```py
# 从 typing 模块中导入 TYPE_CHECKING，用于类型检查
from typing import TYPE_CHECKING

# 从 utils 模块中导入所需函数和类
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_torch_available,
)

# 定义模块导入结构
_import_structure = {
    "configuration_resnet": ["RESNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "ResNetConfig", "ResNetOnnxConfig"]
}

# 检查是否导入了 torch 库，若未导入则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若导入了 torch 库，则将相关模块添加到导入结构中
    _import_structure["modeling_resnet"] = [
        "RESNET_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ResNetForImageClassification",
        "ResNetModel",
        "ResNetPreTrainedModel",
        "ResNetBackbone",
    ]

# 检查是否导入了 tensorflow 库，若未导入则抛出异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若导入了 tensorflow 库，则将相关模块添加到导入结构中
    _import_structure["modeling_tf_resnet"] = [
        "TF_RESNET_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFResNetForImageClassification",
        "TFResNetModel",
        "TFResNetPreTrainedModel",
    ]

# 检查是否导入了 flax 库，若未导入则抛出异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若导入了 flax 库，则将相关模块添加到导入结构中
    _import_structure["modeling_flax_resnet"] = [
        "FlaxResNetForImageClassification",
        "FlaxResNetModel",
        "FlaxResNetPreTrainedModel",
    ]

# 如果是类型检查阶段，则导入额外的模块以进行类型检查
if TYPE_CHECKING:
    # 从 configuration_resnet 模块中导入所需内容
    from .configuration_resnet import RESNET_PRETRAINED_CONFIG_ARCHIVE_MAP, ResNetConfig, ResNetOnnxConfig

    # 检查是否导入了 torch 库，若未导入则抛出异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若导入了 torch 库，则从 modeling_resnet 模块中导入所需内容
        from .modeling_resnet import (
            RESNET_PRETRAINED_MODEL_ARCHIVE_LIST,
            ResNetBackbone,
            ResNetForImageClassification,
            ResNetModel,
            ResNetPreTrainedModel,
        )

    # 检查是否导入了 tensorflow 库，若未导入则抛出异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 若导入了 tensorflow 库，则从 modeling_tf_resnet 模块中导入所需内容
        from .modeling_tf_resnet import (
            TF_RESNET_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFResNetForImageClassification,
            TFResNetModel,
            TFResNetPreTrainedModel,
        )

    # 检查是否导入了 flax 库，若未导入则抛出异常
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
```  
    # 如果不是第一个条件成立，则从当前包中导入以下模块
    from .modeling_flax_resnet import FlaxResNetForImageClassification, FlaxResNetModel, FlaxResNetPreTrainedModel
# 如果当前模块不存在，则执行这段代码
else:
    # 导入 sys 模块
    import sys
    # 创建一个懒加载模块对象，并将其设置为当前模块的实例
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```