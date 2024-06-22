# `.\transformers\models\pegasus_x\__init__.py`

```py
# 导入类型检查库
from typing import TYPE_CHECKING

# 导入自定义模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块导入结构
_import_structure = {
    "configuration_pegasus_x": ["PEGASUS_X_PRETRAINED_CONFIG_ARCHIVE_MAP", "PegasusXConfig"],
}

# 检查是否存在 torch 库，若不存在则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若存在 torch 库，则添加相应的模型导入结构
    _import_structure["modeling_pegasus_x"] = [
        "PEGASUS_X_PRETRAINED_MODEL_ARCHIVE_LIST",
        "PegasusXForConditionalGeneration",
        "PegasusXModel",
        "PegasusXPreTrainedModel",
    ]

# 如果是类型检查，导入相关配置和模型类
if TYPE_CHECKING:
    from .configuration_pegasus_x import PEGASUS_X_PRETRAINED_CONFIG_ARCHIVE_MAP, PegasusXConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_pegasus_x import (
            PEGASUS_X_PRETRAINED_MODEL_ARCHIVE_LIST,
            PegasusXForConditionalGeneration,
            PegasusXModel,
            PegasusXPreTrainedModel,
        )

# 如果不是类型检查，将模块设置为 LazyModule，延迟导入模块
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```  
```