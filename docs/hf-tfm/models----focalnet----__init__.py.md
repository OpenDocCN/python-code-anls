# `.\models\focalnet\__init__.py`

```py
# 导入必要的模块和函数
from typing import TYPE_CHECKING
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块的导入结构字典
_import_structure = {"configuration_focalnet": ["FOCALNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "FocalNetConfig"]}

# 检查是否有torch可用，若不可用则抛出OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果torch可用，定义modeling_focalnet模块的导入结构列表
    _import_structure["modeling_focalnet"] = [
        "FOCALNET_PRETRAINED_MODEL_ARCHIVE_LIST",
        "FocalNetForImageClassification",
        "FocalNetForMaskedImageModeling",
        "FocalNetBackbone",
        "FocalNetModel",
        "FocalNetPreTrainedModel",
    ]

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 导入configuration_focalnet模块中的特定内容
    from .configuration_focalnet import FOCALNET_PRETRAINED_CONFIG_ARCHIVE_MAP, FocalNetConfig

    # 再次检查torch是否可用，若不可用则抛出OptionalDependencyNotAvailable异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入modeling_focalnet模块中的特定内容
        from .modeling_focalnet import (
            FOCALNET_PRETRAINED_MODEL_ARCHIVE_LIST,
            FocalNetBackbone,
            FocalNetForImageClassification,
            FocalNetForMaskedImageModeling,
            FocalNetModel,
            FocalNetPreTrainedModel,
        )

# 如果不在类型检查模式下
else:
    import sys

    # 将当前模块替换为LazyModule对象，以支持懒加载模块的特性
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```