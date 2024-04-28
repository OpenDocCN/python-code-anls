# `.\models\instructblip\__init__.py`

```py
# 版权声明和许可证信息
# 版权声明和许可证信息，指明代码版权和许可证信息
#
# 引入必要的类型检查模块
from typing import TYPE_CHECKING

# 引入必要的依赖和模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块的导入结构
_import_structure = {
    "configuration_instructblip": [
        "INSTRUCTBLIP_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "InstructBlipConfig",
        "InstructBlipQFormerConfig",
        "InstructBlipVisionConfig",
    ],
    "processing_instructblip": ["InstructBlipProcessor"],
}

# 检查是否存在 torch 库，若不存在则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若存在 torch 库，则添加相关模型的导入结构
    _import_structure["modeling_instructblip"] = [
        "INSTRUCTBLIP_PRETRAINED_MODEL_ARCHIVE_LIST",
        "InstructBlipQFormerModel",
        "InstructBlipPreTrainedModel",
        "InstructBlipForConditionalGeneration",
        "InstructBlipVisionModel",
    ]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 从相关模块中导入必要的类和变量
    from .configuration_instructblip import (
        INSTRUCTBLIP_PRETRAINED_CONFIG_ARCHIVE_MAP,
        InstructBlipConfig,
        InstructBlipQFormerConfig,
        InstructBlipVisionConfig,
    )
    from .processing_instructblip import InstructBlipProcessor

    # 再次检查是否存在 torch 库，若不存在则引发异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 从相关模块中导入必要的类和变量
        from .modeling_instructblip import (
            INSTRUCTBLIP_PRETRAINED_MODEL_ARCHIVE_LIST,
            InstructBlipForConditionalGeneration,
            InstructBlipPreTrainedModel,
            InstructBlipQFormerModel,
            InstructBlipVisionModel,
        )

# 如果不是类型检查模式
else:
    import sys

    # 将当前模块设置为 LazyModule 类型
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```