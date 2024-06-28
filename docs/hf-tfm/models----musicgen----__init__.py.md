# `.\models\musicgen\__init__.py`

```py
# 引入类型检查模块，用于在不同环境下处理类型的依赖
from typing import TYPE_CHECKING

# 引入自定义的异常类和模块延迟加载工具类
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available

# 定义模块的导入结构，包含各模块的导出变量和类
_import_structure = {
    "configuration_musicgen": [
        "MUSICGEN_PRETRAINED_CONFIG_ARCHIVE_MAP",
        "MusicgenConfig",
        "MusicgenDecoderConfig",
    ],
    "processing_musicgen": ["MusicgenProcessor"],
}

# 尝试检查是否存在 torch 库，若不存在则引发自定义的可选依赖未找到异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 若存在 torch 库，则添加相关模型建模模块到导入结构中
    _import_structure["modeling_musicgen"] = [
        "MUSICGEN_PRETRAINED_MODEL_ARCHIVE_LIST",
        "MusicgenForConditionalGeneration",
        "MusicgenForCausalLM",
        "MusicgenModel",
        "MusicgenPreTrainedModel",
    ]

# 如果处于类型检查模式，从相应模块中导入配置和处理类
if TYPE_CHECKING:
    from .configuration_musicgen import (
        MUSICGEN_PRETRAINED_CONFIG_ARCHIVE_MAP,
        MusicgenConfig,
        MusicgenDecoderConfig,
    )
    from .processing_musicgen import MusicgenProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果存在 torch 库，则从模型建模模块中导入相关类
        from .modeling_musicgen import (
            MUSICGEN_PRETRAINED_MODEL_ARCHIVE_LIST,
            MusicgenForCausalLM,
            MusicgenForConditionalGeneration,
            MusicgenModel,
            MusicgenPreTrainedModel,
        )

# 如果不处于类型检查模式，则导入 sys 模块，并将当前模块定义为延迟加载模块
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```