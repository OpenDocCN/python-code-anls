# `.\transformers\models\altclip\__init__.py`

```
# 引入类型检查模块中的 TYPE_CHECKING 常量
from typing import TYPE_CHECKING
# 从工具模块中导入自定义的异常类 OptionalDependencyNotAvailable 和懒加载模块 LazyModule
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available, is_torch_available

# 定义模块导入结构字典，包含了不同模块的导入信息
_import_structure = {
    "configuration_altclip": [
        "ALTCLIP_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "AltCLIPConfig",
        "AltCLIPTextConfig",
        "AltCLIPVisionConfig",
    ],
    "processing_altclip": ["AltCLIPProcessor"],
}

# 检查是否导入了 torch 库，如果没有则抛出自定义异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果导入了 torch 库，则添加相关模型配置和模型定义的导入信息到模块导入结构字典中
    _import_structure["modeling_altclip"] = [
        "ALTCLIP_PRETRAINED_MODEL_ARCHIVE_LIST",
        "AltCLIPPreTrainedModel",
        "AltCLIPModel",
        "AltCLIPTextModel",
        "AltCLIPVisionModel",
    ]


# 如果当前环境支持类型检查，则进行类型检查相关的导入
if TYPE_CHECKING:
    # 从配置模块中导入预训练模型配置信息
    from .configuration_altclip import (
        ALTCLIP_PRETRAINED_CONFIG_ARCHIVE_MAP,
        AltCLIPConfig,
        AltCLIPTextConfig,
        AltCLIPVisionConfig,
    )
    # 从处理模块中导入处理器类
    from .processing_altclip import AltCLIPProcessor

    # 检查是否导入了 torch 库，如果没有则抛出自定义异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果导入了 torch 库，则从模型定义模块中导入模型相关类
        from .modeling_altclip import (
            ALTCLIP_PRETRAINED_MODEL_ARCHIVE_LIST,
            AltCLIPModel,
            AltCLIPPreTrainedModel,
            AltCLIPTextModel,
            AltCLIPVisionModel,
        )


# 如果当前环境不支持类型检查，则将当前模块重定向为懒加载模块
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```