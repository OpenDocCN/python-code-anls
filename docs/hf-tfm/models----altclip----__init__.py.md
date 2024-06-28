# `.\models\altclip\__init__.py`

```
# 版权声明和许可证信息，指明版权归 The HuggingFace Team 所有，使用 Apache License, Version 2.0 许可
#
# 引入必要的模块和函数
from typing import TYPE_CHECKING
# 从 ...utils 中导入相关模块，处理依赖未安装的情况
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tokenizers_available, is_torch_available

# 定义模块的导入结构，包括配置、处理和模型的列表
_import_structure = {
    "configuration_altclip": [
        "ALTCLIP_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "AltCLIPConfig",
        "AltCLIPTextConfig",
        "AltCLIPVisionConfig",
    ],
    "processing_altclip": ["AltCLIPProcessor"],
}

# 检查是否安装了 torch，若未安装则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果安装了 torch，则将 modeling_altclip 模块添加到导入结构中
    _import_structure["modeling_altclip"] = [
        "ALTCLIP_PRETRAINED_MODEL_ARCHIVE_LIST",
        "AltCLIPPreTrainedModel",
        "AltCLIPModel",
        "AltCLIPTextModel",
        "AltCLIPVisionModel",
    ]

# 如果是类型检查阶段，则从相应模块中导入具体的类和常量
if TYPE_CHECKING:
    from .configuration_altclip import (
        ALTCLIP_PRETRAINED_CONFIG_ARCHIVE_MAP,
        AltCLIPConfig,
        AltCLIPTextConfig,
        AltCLIPVisionConfig,
    )
    from .processing_altclip import AltCLIPProcessor

    # 再次检查是否安装了 torch，若未安装则忽略
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果安装了 torch，则从 modeling_altclip 模块导入相关类和常量
        from .modeling_altclip import (
            ALTCLIP_PRETRAINED_MODEL_ARCHIVE_LIST,
            AltCLIPModel,
            AltCLIPPreTrainedModel,
            AltCLIPTextModel,
            AltCLIPVisionModel,
        )

# 如果不是类型检查阶段，则定义一个 LazyModule 并将其设置为当前模块的代理
else:
    import sys

    # 将当前模块的 sys.modules 设置为 LazyModule 对象，用于延迟加载
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```