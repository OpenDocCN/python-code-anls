# `.\transformers\models\align\__init__.py`

```
# 引入必要的模块和函数
from typing import TYPE_CHECKING
# 引入自定义异常和模块延迟加载工具
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)

# 定义需要延迟加载的模块和函数的结构
_import_structure = {
    "configuration_align": [
        "ALIGN_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "AlignConfig",
        "AlignTextConfig",
        "AlignVisionConfig",
    ],
    "processing_align": ["AlignProcessor"],
}

# 尝试导入 torch 模块，如果不可用则引发自定义异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 torch 可用，则添加相关模型结构到导入结构中
    _import_structure["modeling_align"] = [
        "ALIGN_PRETRAINED_MODEL_ARCHIVE_LIST",
        "AlignModel",
        "AlignPreTrainedModel",
        "AlignTextModel",
        "AlignVisionModel",
    ]

# 如果类型检查开启
if TYPE_CHECKING:
    # 从配置模块中导入相关内容
    from .configuration_align import (
        ALIGN_PRETRAINED_CONFIG_ARCHIVE_MAP,
        AlignConfig,
        AlignTextConfig,
        AlignVisionConfig,
    )
    # 从处理模块中导入相关内容
    from .processing_align import AlignProcessor

    # 尝试导入 torch 模块，如果不可用则引发自定义异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 torch 可用，则从建模模块中导入相关内容
        from .modeling_align import (
            ALIGN_PRETRAINED_MODEL_ARCHIVE_LIST,
            AlignModel,
            AlignPreTrainedModel,
            AlignTextModel,
            AlignVisionModel,
        )

# 如果不是类型检查阶段，则进行模块延迟加载
else:
    # 导入系统模块
    import sys
    # 将当前模块替换为 LazyModule 实例，实现延迟加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```