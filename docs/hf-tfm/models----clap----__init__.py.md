# `.\transformers\models\clap\__init__.py`

```
# 导入类型检查相关模块
from typing import TYPE_CHECKING
# 导入可选依赖未安装异常类
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义需要导入的模块结构
_import_structure = {
    "configuration_clap": [
        "CLAP_PRETRAINED_MODEL_ARCHIVE_LIST",  # 预训练模型存档列表
        "ClapAudioConfig",  # Clap 音频配置类
        "ClapConfig",  # Clap 配置类
        "ClapTextConfig",  # Clap 文本配置类
    ],
    "processing_clap": ["ClapProcessor"],  # Clap 处理器类
}

# 检查是否安装了 torch
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果安装了 torch，则添加模型相关模块
    _import_structure["modeling_clap"] = [
        "CLAP_PRETRAINED_MODEL_ARCHIVE_LIST",  # 预训练模型存档列表
        "ClapModel",  # Clap 模型类
        "ClapPreTrainedModel",  # Clap 预训练模型类
        "ClapTextModel",  # Clap 文本模型类
        "ClapTextModelWithProjection",  # Clap 带投影的文本模型类
        "ClapAudioModel",  # Clap 音频模型类
        "ClapAudioModelWithProjection",  # Clap 带投影的音频模型类
    ]
    # 添加特征提取相关模块
    _import_structure["feature_extraction_clap"] = ["ClapFeatureExtractor"]

# 如果是类型检查阶段
if TYPE_CHECKING:
    # 导入配置相关类
    from .configuration_clap import (
        CLAP_PRETRAINED_MODEL_ARCHIVE_LIST,
        ClapAudioConfig,
        ClapConfig,
        ClapTextConfig,
    )
    # 导入处理器类
    from .processing_clap import ClapProcessor

    # 再次检查是否安装了 torch
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入特征提取相关类和模型相关类
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

# 如果不是类型检查阶段
else:
    # 导入 sys 模块
    import sys

    # 将当前模块指定为惰性加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```