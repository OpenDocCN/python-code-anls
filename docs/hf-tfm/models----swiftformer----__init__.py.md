# `.\models\swiftformer\__init__.py`

```
# 引入必要的模块和依赖项来支持 SwiftFormer 模型的配置和建模
from typing import TYPE_CHECKING

# 从工具包中引入异常类和模块懒加载工具
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)

# 定义模块的导入结构，包括配置和建模相关内容
_import_structure = {
    "configuration_swiftformer": [
        "SWIFTFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "SwiftFormerConfig",
        "SwiftFormerOnnxConfig",
    ]
}

# 检查是否存在 torch 库，如果不存在则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 torch 可用，则添加建模相关内容到导入结构中
    _import_structure["modeling_swiftformer"] = [
        "SWIFTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "SwiftFormerForImageClassification",
        "SwiftFormerModel",
        "SwiftFormerPreTrainedModel",
    ]

# 如果 TYPE_CHECKING 为真，则从 configuration_swiftformer 模块中导入特定的配置类和映射
if TYPE_CHECKING:
    from .configuration_swiftformer import (
        SWIFTFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        SwiftFormerConfig,
        SwiftFormerOnnxConfig,
    )

    # 再次检查 torch 是否可用，若不可用则抛出异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，则从 modeling_swiftformer 模块导入建模相关类
        from .modeling_swiftformer import (
            SWIFTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
            SwiftFormerForImageClassification,
            SwiftFormerModel,
            SwiftFormerPreTrainedModel,
        )

# 如果 TYPE_CHECKING 为假，则导入 sys 模块并将当前模块设为 _LazyModule 的懒加载实例
else:
    import sys

    # 将当前模块定义为 _LazyModule 的实例，以支持懒加载模块的导入
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```