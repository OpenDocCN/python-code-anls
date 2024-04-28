# `.\transformers\models\swin\__init__.py`

```
# 版权声明及许可证信息

# 引入类型检查模块
from typing import TYPE_CHECKING

# 引入可选依赖不可用异常和延迟加载模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tf_available, is_torch_available

# 定义依赖导入结构
_import_structure = {"configuration_swin": ["SWIN_PRETRAINED_CONFIG_ARCHIVE_MAP", "SwinConfig", "SwinOnnxConfig"]}

# 检查是否torch可用，不可用则抛出可选依赖不可用异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若torch可用，则扩展依赖导入结构
    _import_structure["modeling_swin"] = [
        "SWIN_PRETRAINED_MODEL_ARCHIVE_LIST",
        "SwinForImageClassification",
        "SwinForMaskedImageModeling",
        "SwinModel",
        "SwinPreTrainedModel",
        "SwinBackbone",
    ]

# 检查是否tensorflow可用，不可用则抛出可选依赖不可用异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若tensorflow可用，则扩展依赖导入结构
    _import_structure["modeling_tf_swin"] = [
        "TF_SWIN_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFSwinForImageClassification",
        "TFSwinForMaskedImageModeling",
        "TFSwinModel",
        "TFSwinPreTrainedModel",
    ]

# 如果在类型检查环境下
if TYPE_CHECKING:
    # 导入配置swin模块中的指定内容
    from .configuration_swin import SWIN_PRETRAINED_CONFIG_ARCHIVE_MAP, SwinConfig, SwinOnnxConfig
    # 如果torch可用，则导入modeling_swin模块中的指定内容
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_swin import (
            SWIN_PRETRAINED_MODEL_ARCHIVE_LIST,
            SwinBackbone,
            SwinForImageClassification,
            SwinForMaskedImageModeling,
            SwinModel,
            SwinPreTrainedModel,
        )
    # 如果tensorflow可用，则导入modeling_tf_swin模块中的指定内容
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_swin import (
            TF_SWIN_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFSwinForImageClassification,
            TFSwinForMaskedImageModeling,
            TFSwinModel,
            TFSwinPreTrainedModel,
        )

# 如果不在类型检查环境下
else:
    # 动态创建延迟加载模块
    import sys
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```