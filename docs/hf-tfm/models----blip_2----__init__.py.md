# `.\transformers\models\blip_2\__init__.py`

```py
# 导入模块类型检查
from typing import TYPE_CHECKING

# 导入可选依赖未安装异常和延迟加载模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块导入结构
_import_structure = {
    "configuration_blip_2": [
        "BLIP_2_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置文件归档映射
        "Blip2Config",  # BLIP2 配置
        "Blip2QFormerConfig",  # BLIP2 QFormer 配置
        "Blip2VisionConfig",  # BLIP2 视觉配置
    ],
    "processing_blip_2": ["Blip2Processor"],  # BLIP2 处理器
}

# 检查是否 Torch 可用，若不可用则引发可选依赖未安装异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若 Torch 可用，则加入模型建模结构
    _import_structure["modeling_blip_2"] = [
        "BLIP_2_PRETRAINED_MODEL_ARCHIVE_LIST",  # 预训练模型归档列表
        "Blip2Model",  # BLIP2 模型
        "Blip2QFormerModel",  # BLIP2 QFormer 模型
        "Blip2PreTrainedModel",  # BLIP2 预训练模型
        "Blip2ForConditionalGeneration",  # BLIP2 有条件生成模型
        "Blip2VisionModel",  # BLIP2 视觉模型
    ]

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 从配置模块中导入配置相关内容
    from .configuration_blip_2 import (
        BLIP_2_PRETRAINED_CONFIG_ARCHIVE_MAP,
        Blip2Config,
        Blip2QFormerConfig,
        Blip2VisionConfig,
    )
    # 从处理模块中导入处理器
    from .processing_blip_2 import Blip2Processor

    # 再次检查 Torch 是否可用，若不可用则引发可选依赖未安装异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 从模型建模模块中导入模型相关内容
        from .modeling_blip_2 import (
            BLIP_2_PRETRAINED_MODEL_ARCHIVE_LIST,
            Blip2ForConditionalGeneration,
            Blip2Model,
            Blip2PreTrainedModel,
            Blip2QFormerModel,
            Blip2VisionModel,
        )

# 如果不是类型检查模式
else:
    import sys

    # 将当前模块替换为延迟加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```