# `.\models\informer\__init__.py`

```
# 导入需要的模块和函数
from typing import TYPE_CHECKING
# 从当前包中导入自定义的异常和延迟加载模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块的导入结构，包括配置和模型信息
_import_structure = {
    "configuration_informer": [
        "INFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "InformerConfig",
    ],
}

# 检查是否导入了 torch 库，如果没有则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果导入了 torch，则添加额外的模型信息到导入结构中
    _import_structure["modeling_informer"] = [
        "INFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "InformerForPrediction",
        "InformerModel",
        "InformerPreTrainedModel",
    ]

# 如果是类型检查阶段，则导入配置和模型相关的类型信息
if TYPE_CHECKING:
    from .configuration_informer import INFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, InformerConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_informer import (
            INFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            InformerForPrediction,
            InformerModel,
            InformerPreTrainedModel,
        )

# 如果不是类型检查阶段，则创建 LazyModule 对象，并将其作为当前模块的属性
else:
    import sys

    # 创建 LazyModule 对象，并设置当前模块的属性为该对象，实现延迟加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```