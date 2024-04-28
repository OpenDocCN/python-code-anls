# `.\transformers\models\speech_encoder_decoder\__init__.py`

```
# 引入类型检查模块
from typing import TYPE_CHECKING

# 引入自定义模块和函数，包括可选依赖未安装的错误处理
from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_flax_available, is_torch_available

# 定义模块导入结构
_import_structure = {"configuration_speech_encoder_decoder": ["SpeechEncoderDecoderConfig"]}

# 检查是否存在 PyTorch
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 PyTorch 存在，则将模型模块添加到导入结构中
    _import_structure["modeling_speech_encoder_decoder"] = ["SpeechEncoderDecoderModel"]

# 检查是否存在 Flax
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 Flax 存在，则将 Flax 模型模块添加到导入结构中
    _import_structure["modeling_flax_speech_encoder_decoder"] = ["FlaxSpeechEncoderDecoderModel"]

# 如果正在进行类型检查
if TYPE_CHECKING:
    # 导入配置模块以供类型检查使用
    from .configuration_speech_encoder_decoder import SpeechEncoderDecoderConfig

    try:
        # 如果 PyTorch 存在，导入 PyTorch 模型模块以供类型检查使用
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_speech_encoder_decoder import SpeechEncoderDecoderModel

    try:
        # 如果 Flax 存在，导入 Flax 模型模块以供类型检查使用
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_flax_speech_encoder_decoder import FlaxSpeechEncoderDecoderModel

# 如果不是在进行类型检查，而是在运行时
else:
    # 导入 sys 模块
    import sys

    # 将当前模块替换为懒加载模块，使用定义的导入结构和当前模块规范
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```