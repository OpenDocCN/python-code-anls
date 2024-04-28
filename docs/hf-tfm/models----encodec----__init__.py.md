# `.\models\encodec\__init__.py`

```py
# 版权声明及使用许可信息
# 从依赖库中导入必要的模块和函数
from typing import TYPE_CHECKING
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)

# 定义模块的导入结构
_import_structure = {
    "configuration_encodec": [
        "ENCODEC_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "EncodecConfig",
    ],
    "feature_extraction_encodec": ["EncodecFeatureExtractor"],
}

# 检查是否存在 torch，若不存在则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若存在 torch 则添加相关模块到导入结构中
    _import_structure["modeling_encodec"] = [
        "ENCODEC_PRETRAINED_MODEL_ARCHIVE_LIST",
        "EncodecModel",
        "EncodecPreTrainedModel",
    ]

# 如果是类型检查，导入相关模块和函数
if TYPE_CHECKING:
    from .configuration_encodec import (
        ENCODEC_PRETRAINED_CONFIG_ARCHIVE_MAP,
        EncodecConfig,
    )
    from .feature_extraction_encodec import EncodecFeatureExtractor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_encodec import (
            ENCODEC_PRETRAINED_MODEL_ARCHIVE_LIST,
            EncodecModel,
            EncodecPreTrainedModel,
        )

# 如果不是类型检查，将模块注册到系统中
else:
    import sys
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```