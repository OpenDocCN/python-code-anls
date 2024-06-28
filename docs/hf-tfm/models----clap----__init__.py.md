# `.\models\clap\__init__.py`

```py
# 引入必要的模块和类型检查功能
from typing import TYPE_CHECKING
# 引入自定义的异常和懒加载模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块的导入结构，包括配置、处理和特征提取相关的模块
_import_structure = {
    "configuration_clap": [
        "CLAP_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ClapAudioConfig",
        "ClapConfig",
        "ClapTextConfig",
    ],
    "processing_clap": ["ClapProcessor"],
}

# 尝试检查是否存在 torch，如果不存在则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 torch 可用，则添加建模和特征提取模块到导入结构中
    _import_structure["modeling_clap"] = [
        "CLAP_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ClapModel",
        "ClapPreTrainedModel",
        "ClapTextModel",
        "ClapTextModelWithProjection",
        "ClapAudioModel",
        "ClapAudioModelWithProjection",
    ]
    _import_structure["feature_extraction_clap"] = ["ClapFeatureExtractor"]

# 如果是类型检查环境，引入配置和处理模块中的符号，以及特征提取和建模模块（如果 torch 可用）
if TYPE_CHECKING:
    from .configuration_clap import (
        CLAP_PRETRAINED_MODEL_ARCHIVE_LIST,
        ClapAudioConfig,
        ClapConfig,
        ClapTextConfig,
    )
    from .processing_clap import ClapProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .feature_extraction_clap import ClapFeatureExtractor
        from .modeling_clap import (
            CLAP_PRETRAINED_MODEL_ARCHIVE_LIST,
            ClapAudioModel,
            ClapAudioModelWithProjection,
            ClapModel,
            ClapPreTrainedModel,
            ClapTextModel,
            ClapTextModelWithProjection,
        )

# 如果不是类型检查环境，则设置当前模块为懒加载模块
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```