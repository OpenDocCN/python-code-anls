# `.\models\vitdet\__init__.py`

```py
# 版权声明和许可条款，指明版权归 HuggingFace 团队所有，使用 Apache License, Version 2.0 许可
#
# 如果未遵守许可，除非适用法律要求或书面同意，否则不得使用该文件
from typing import TYPE_CHECKING

# 从当前目录中的 utils 模块导入所需的符号
from ...utils import (
    OptionalDependencyNotAvailable,  # 导入 OptionalDependencyNotAvailable 异常类
    _LazyModule,  # 导入 _LazyModule 类
    is_torch_available,  # 导入 is_torch_available 函数
)

# 定义导入结构，包含了配置和模型相关的符号
_import_structure = {"configuration_vitdet": ["VITDET_PRETRAINED_CONFIG_ARCHIVE_MAP", "VitDetConfig"]}

# 尝试检查是否 torch 可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 torch 可用，则将模型相关的符号加入导入结构
    _import_structure["modeling_vitdet"] = [
        "VITDET_PRETRAINED_MODEL_ARCHIVE_LIST",
        "VitDetModel",
        "VitDetPreTrainedModel",
        "VitDetBackbone",
    ]

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 导入配置相关的符号和模型相关的符号（如果 torch 可用）
    from .configuration_vitdet import VITDET_PRETRAINED_CONFIG_ARCHIVE_MAP, VitDetConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_vitdet import (
            VITDET_PRETRAINED_MODEL_ARCHIVE_LIST,
            VitDetBackbone,
            VitDetModel,
            VitDetPreTrainedModel,
        )

# 如果不在类型检查模式下
else:
    import sys

    # 动态设置当前模块的 sys.modules 条目，使用 _LazyModule 进行延迟加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```