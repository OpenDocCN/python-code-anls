# `.\transformers\models\patchtst\__init__.py`

```
# 导入类型检查模块
from typing import TYPE_CHECKING

# 导入可选依赖未安装异常和延迟加载模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义导入结构，包含了模型配置和模型的导入结构
_import_structure = {
    "configuration_patchtst": [
        "PATCHTST_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "PatchTSTConfig",
    ],
}

# 尝试检查是否 Torch 可用，如果不可用则抛出可选依赖未安装异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Torch 可用，扩展导入结构，包含了模型相关的内容
    _import_structure["modeling_patchtst"] = [
        "PATCHTST_PRETRAINED_MODEL_ARCHIVE_LIST",
        "PatchTSTModel",
        "PatchTSTPreTrainedModel",
        "PatchTSTForPrediction",
        "PatchTSTForPretraining",
        "PatchTSTForRegression",
        "PatchTSTForClassification",
    ]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 导入配置相关的内容
    from .configuration_patchtst import PATCHTST_PRETRAINED_CONFIG_ARCHIVE_MAP, PatchTSTConfig

    # 尝试检查是否 Torch 可用，如果不可用则抛出可选依赖未安装异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入模型相关的内容
        from .modeling_patchtst import (
            PATCHTST_PRETRAINED_MODEL_ARCHIVE_LIST,
            PatchTSTForClassification,
            PatchTSTForPrediction,
            PatchTSTForPretraining,
            PatchTSTForRegression,
            PatchTSTModel,
            PatchTSTPreTrainedModel,
        )

# 如果不是类型检查模式，即在运行时
else:
    import sys

    # 将当前模块替换为 LazyModule，以延迟加载模块中的内容
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```