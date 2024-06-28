# `.\models\resnet\__init__.py`

```py
# 引入类型检查模块
from typing import TYPE_CHECKING

# 从工具模块中引入必要的异常和工具函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_torch_available,
)

# 定义一个字典，包含了需要导入的结构
_import_structure = {
    "configuration_resnet": ["RESNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "ResNetConfig", "ResNetOnnxConfig"]
}

# 尝试导入 torch 相关模块，如果不可用则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加 torch 模型相关结构到导入结构字典中
    _import_structure["modeling_resnet"] = [
        "RESNET_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ResNetForImageClassification",
        "ResNetModel",
        "ResNetPreTrainedModel",
        "ResNetBackbone",
    ]

# 尝试导入 TensorFlow 相关模块，如果不可用则抛出异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加 TensorFlow 模型相关结构到导入结构字典中
    _import_structure["modeling_tf_resnet"] = [
        "TF_RESNET_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFResNetForImageClassification",
        "TFResNetModel",
        "TFResNetPreTrainedModel",
    ]

# 尝试导入 Flax 相关模块，如果不可用则抛出异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，添加 Flax 模型相关结构到导入结构字典中
    _import_structure["modeling_flax_resnet"] = [
        "FlaxResNetForImageClassification",
        "FlaxResNetModel",
        "FlaxResNetPreTrainedModel",
    ]

# 如果处于类型检查模式，从相应模块导入必要的类型和配置
if TYPE_CHECKING:
    from .configuration_resnet import RESNET_PRETRAINED_CONFIG_ARCHIVE_MAP, ResNetConfig, ResNetOnnxConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 torch 模型相关的类型和类
        from .modeling_resnet import (
            RESNET_PRETRAINED_MODEL_ARCHIVE_LIST,
            ResNetBackbone,
            ResNetForImageClassification,
            ResNetModel,
            ResNetPreTrainedModel,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 TensorFlow 模型相关的类型和类
        from .modeling_tf_resnet import (
            TF_RESNET_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFResNetForImageClassification,
            TFResNetModel,
            TFResNetPreTrainedModel,
        )

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    # 如果前面的条件不满足，导入以下模块中的指定类和函数
    from .modeling_flax_resnet import FlaxResNetForImageClassification, FlaxResNetModel, FlaxResNetPreTrainedModel
else:
    # 如果前面的条件都不满足，则执行以下代码块
    import sys
    # 导入系统模块 sys

    # 使用当前模块的名称作为键，将 _LazyModule 对象赋值给 sys.modules 中的相应条目
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
    # 这里假设 _LazyModule 是一个自定义的模块加载器类，将当前模块注册到 sys.modules 中
```