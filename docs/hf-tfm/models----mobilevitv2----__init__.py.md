# `.\models\mobilevitv2\__init__.py`

```
# 版权声明和许可信息，标明代码版权和使用许可条件
from typing import TYPE_CHECKING

# 导入必要的异常和模块
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
    is_vision_available,
)

# 定义模块导入结构的字典
_import_structure = {
    "configuration_mobilevitv2": [
        "MOBILEVITV2_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "MobileViTV2Config",
        "MobileViTV2OnnxConfig",
    ],
}

# 检查是否存在torch模块，若不存在则抛出OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果存在torch模块，将相关模型导入结构加入_import_structure字典
    _import_structure["modeling_mobilevitv2"] = [
        "MOBILEVITV2_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MobileViTV2ForImageClassification",
        "MobileViTV2ForSemanticSegmentation",
        "MobileViTV2Model",
        "MobileViTV2PreTrainedModel",
    ]

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 导入配置相关的类和常量
    from .configuration_mobilevitv2 import (
        MOBILEVITV2_PRETRAINED_CONFIG_ARCHIVE_MAP,
        MobileViTV2Config,
        MobileViTV2OnnxConfig,
    )

    # 再次检查是否存在torch模块，若不存在则忽略
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入模型相关的类和常量
        from .modeling_mobilevitv2 import (
            MOBILEVITV2_PRETRAINED_MODEL_ARCHIVE_LIST,
            MobileViTV2ForImageClassification,
            MobileViTV2ForSemanticSegmentation,
            MobileViTV2Model,
            MobileViTV2PreTrainedModel,
        )

# 如果不在类型检查模式下，则将当前模块作为LazyModule延迟加载模块导入
else:
    import sys

    # 将当前模块替换为_LazyModule对象，实现延迟导入
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```