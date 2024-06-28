# `.\models\vit_msn\__init__.py`

```py
# 导入所需模块和函数
from typing import TYPE_CHECKING
# 从当前项目的utils模块中导入异常类和LazyModule类，还有is_torch_available函数
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块的导入结构，包含了configuration_vit_msn的两个对象
_import_structure = {"configuration_vit_msn": ["VIT_MSN_PRETRAINED_CONFIG_ARCHIVE_MAP", "ViTMSNConfig"]}

# 检查是否存在torch库，若不存在则引发OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果torch可用，则在_import_structure中添加modeling_vit_msn的四个对象
    _import_structure["modeling_vit_msn"] = [
        "VIT_MSN_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ViTMSNModel",
        "ViTMSNForImageClassification",
        "ViTMSNPreTrainedModel",
    ]

# 如果是类型检查模式，导入具体的配置和模型类
if TYPE_CHECKING:
    # 从当前模块的configuration_vit_msn中导入两个对象
    from .configuration_vit_msn import VIT_MSN_PRETRAINED_CONFIG_ARCHIVE_MAP, ViTMSNConfig

    # 再次检查是否存在torch库，若不存在则引发OptionalDependencyNotAvailable异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果torch可用，则从当前模块的modeling_vit_msn中导入四个对象
        from .modeling_vit_msn import (
            VIT_MSN_PRETRAINED_MODEL_ARCHIVE_LIST,
            ViTMSNForImageClassification,
            ViTMSNModel,
            ViTMSNPreTrainedModel,
        )

# 如果不是类型检查模式，将当前模块设置为_LazyModule的实例
else:
    import sys

    # 将当前模块设为_LazyModule的实例，用于惰性加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```