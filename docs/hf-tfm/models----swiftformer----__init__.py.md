# `.\transformers\models\swiftformer\__init__.py`

```py
# 引入类型检查模块
from typing import TYPE_CHECKING
# 从工具模块中引入必要的异常类和函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)

# 定义模块的导入结构，包括模块名和对应的导入内容
_import_structure = {
    "configuration_swiftformer": [
        "SWIFTFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "SwiftFormerConfig",
        "SwiftFormerOnnxConfig",
    ]
}

# 尝试导入torch模块，如果未安装则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果torch可用，则导入Swiftformer模型相关的内容
    _import_structure["modeling_swiftformer"] = [
        "SWIFTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "SwiftFormerForImageClassification",
        "SwiftFormerModel",
        "SwiftFormerPreTrainedModel",
    ]

# 如果需要类型检查，则导入相关配置和模型内容
if TYPE_CHECKING:
    from .configuration_swiftformer import (
        SWIFTFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        SwiftFormerConfig,
        SwiftFormerOnnxConfig,
    )

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_swiftformer import (
            SWIFTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            SwiftFormerForImageClassification,
            SwiftFormerModel,
            SwiftFormerPreTrainedModel,
        )

# 如果不需要类型检查，则将当前模块设置为懒加载模块
else:
    # 导入sys模块
    import sys
    # 将当前模块设为懒加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```