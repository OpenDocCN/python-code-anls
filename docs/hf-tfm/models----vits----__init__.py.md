# `.\transformers\models\vits\__init__.py`

```
# 这是一个版权声明和许可信息
# 这些注释描述了 Apache License,Version 2.0 的许可条款
from typing import TYPE_CHECKING

# 导入一些辅助函数和模块
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_speech_available,
    is_torch_available,
)

# 定义了一个导入结构字典，用于延迟导入相关模块
_import_structure = {
    "configuration_vits": [
        "VITS_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "VitsConfig",
    ],
    "tokenization_vits": ["VitsTokenizer"],
}

# 检查是否安装了 PyTorch 库，如果没有则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 PyTorch 可用，则添加 modeling_vits 相关的模块到导入结构中
    _import_structure["modeling_vits"] = [
        "VITS_PRETRAINED_MODEL_ARCHIVE_LIST",
        "VitsModel",
        "VitsPreTrainedModel",
    ]

# 如果是类型检查阶段，则导入相关模块
if TYPE_CHECKING:
    from .configuration_vits import (
        VITS_PRETRAINED_CONFIG_ARCHIVE_MAP,
        VitsConfig,
    )
    from .tokenization_vits import VitsTokenizer

    # 再次检查 PyTorch 的可用性
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_vits import (
            VITS_PRETRAINED_MODEL_ARCHIVE_LIST,
            VitsModel,
            VitsPreTrainedModel,
        )

# 如果不是类型检查阶段，则使用延迟加载机制
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```