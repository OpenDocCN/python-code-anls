# `.\models\swin\__init__.py`

```
# 引入类型检查的模块
from typing import TYPE_CHECKING

# 引入异常类，用于处理可选依赖不可用的情况
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tf_available, is_torch_available

# 定义模块的导入结构，包含配置和模型相关的导入信息
_import_structure = {"configuration_swin": ["SWIN_PRETRAINED_CONFIG_ARCHIVE_MAP", "SwinConfig", "SwinOnnxConfig"]}

# 检查是否有torch可用，若不可用则抛出OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若torch可用，则添加相关的模型定义到_import_structure中
    _import_structure["modeling_swin"] = [
        "SWIN_PRETRAINED_MODEL_ARCHIVE_LIST",
        "SwinForImageClassification",
        "SwinForMaskedImageModeling",
        "SwinModel",
        "SwinPreTrainedModel",
        "SwinBackbone",
    ]

# 检查是否有tensorflow可用，若不可用则抛出OptionalDependencyNotAvailable异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若tensorflow可用，则添加相关的tensorflow模型定义到_import_structure中
    _import_structure["modeling_tf_swin"] = [
        "TF_SWIN_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFSwinForImageClassification",
        "TFSwinForMaskedImageModeling",
        "TFSwinModel",
        "TFSwinPreTrainedModel",
    ]

# 如果当前是类型检查阶段
if TYPE_CHECKING:
    # 从配置模块中导入特定的配置类和常量
    from .configuration_swin import SWIN_PRETRAINED_CONFIG_ARCHIVE_MAP, SwinConfig, SwinOnnxConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 从模型定义模块中导入特定的torch模型类
        from .modeling_swin import (
            SWIN_PRETRAINED_MODEL_ARCHIVE_LIST,
            SwinBackbone,
            SwinForImageClassification,
            SwinForMaskedImageModeling,
            SwinModel,
            SwinPreTrainedModel,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 从tensorflow模型定义模块中导入特定的tensorflow模型类
        from .modeling_tf_swin import (
            TF_SWIN_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFSwinForImageClassification,
            TFSwinForMaskedImageModeling,
            TFSwinModel,
            TFSwinPreTrainedModel,
        )

# 如果不是类型检查阶段，则执行延迟模块加载的逻辑
else:
    import sys

    # 将当前模块替换为LazyModule，以实现延迟导入
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```