# `.\transformers\models\musicgen\__init__.py`

```py
# 导入类型检查工具
from typing import TYPE_CHECKING
# 导入可选依赖未安装异常和延迟加载模块
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块导入结构
_import_structure = {
    "configuration_musicgen": [
        "MUSICGEN_PRETRAINED_CONFIG_ARCHIVE_MAP",  # 预训练配置文件的存档映射
        "MusicgenConfig",  # 音乐生成配置
        "MusicgenDecoderConfig",  # 解码器配置
    ],
    "processing_musicgen": ["MusicgenProcessor"],  # 音乐生成处理器
}

# 尝试导入 PyTorch，如果不可用则引发可选依赖未安装异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 PyTorch 可用，添加模型相关的导入结构
    _import_structure["modeling_musicgen"] = [
        "MUSICGEN_PRETRAINED_MODEL_ARCHIVE_LIST",  # 预训练模型存档列表
        "MusicgenForConditionalGeneration",  # 用于条件生成的音乐生成模型
        "MusicgenForCausalLM",  # 用于因果语言模型的音乐生成模型
        "MusicgenModel",  # 音乐生成模型
        "MusicgenPreTrainedModel",  # 音乐生成预训练模型
    ]

# 如果是类型检查阶段
if TYPE_CHECKING:
    # 导入类型检查所需的模块
    from .configuration_musicgen import (
        MUSICGEN_PRETRAINED_CONFIG_ARCHIVE_MAP,  # 预训练配置文件的存档映射
        MusicgenConfig,  # 音乐生成配置
        MusicgenDecoderConfig,  # 解码器配置
    )
    from .processing_musicgen import MusicgenProcessor  # 音乐生成处理器

    # 尝试导入 PyTorch，如果不可用则引发可选依赖未安装异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 PyTorch 可用，导入模型相关的类型检查所需的模块
        from .modeling_musicgen import (
            MUSICGEN_PRETRAINED_MODEL_ARCHIVE_LIST,  # 预训练模型存档列表
            MusicgenForCausalLM,  # 用于因果语言模型的音乐生成模型
            MusicgenForConditionalGeneration,  # 用于条件生成的音乐生成模型
            MusicgenModel,  # 音乐生成模型
            MusicgenPreTrainedModel,  # 音乐生成预训练模型
        )

# 如果不是类型检查阶段
else:
    # 导入 sys 模块
    import sys
    # 将当前模块替换为延迟加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```