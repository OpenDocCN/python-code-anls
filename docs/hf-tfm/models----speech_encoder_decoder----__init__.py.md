# `.\models\speech_encoder_decoder\__init__.py`

```py
# 导入必要的模块和函数
from typing import TYPE_CHECKING
# 从相对路径导入必要的异常和类
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_flax_available, is_torch_available

# 定义导入结构，包含模型配置
_import_structure = {"configuration_speech_encoder_decoder": ["SpeechEncoderDecoderConfig"]}

# 检查是否可用 Torch 库，若不可用则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 添加 Torch 版本的模型到导入结构中
    _import_structure["modeling_speech_encoder_decoder"] = ["SpeechEncoderDecoderModel"]

# 检查是否可用 Flax 库，若不可用则抛出异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 添加 Flax 版本的模型到导入结构中
    _import_structure["modeling_flax_speech_encoder_decoder"] = ["FlaxSpeechEncoderDecoderModel"]

# 如果是类型检查阶段
if TYPE_CHECKING:
    # 导入模型配置类
    from .configuration_speech_encoder_decoder import SpeechEncoderDecoderConfig

    # 检查 Torch 是否可用，若可用则导入 Torch 版本的模型
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_speech_encoder_decoder import SpeechEncoderDecoderModel

    # 检查 Flax 是否可用，若可用则导入 Flax 版本的模型
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_flax_speech_encoder_decoder import FlaxSpeechEncoderDecoderModel

# 如果不是类型检查阶段
else:
    # 导入系统模块
    import sys

    # 将当前模块替换为 LazyModule 实例，用于惰性加载模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```