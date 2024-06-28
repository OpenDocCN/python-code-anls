# `.\models\vision_encoder_decoder\__init__.py`

```
# 版权声明和许可证信息，保留所有权利
#
# 根据 Apache 许可证版本 2.0 授权使用此文件；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发本软件，
# 不附带任何形式的担保或条件，无论是明示的还是默示的。
# 有关具体语言的权限，请参阅许可证。
#

# 从 typing 模块导入 TYPE_CHECKING 类型提示
from typing import TYPE_CHECKING

# 从 utils 模块导入所需的类和函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_torch_available,
)

# 定义模块导入结构
_import_structure = {
    "configuration_vision_encoder_decoder": ["VisionEncoderDecoderConfig", "VisionEncoderDecoderOnnxConfig"]
}

# 检查是否有 torch 可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则将 VisionEncoderDecoderModel 导入到模块导入结构中
    _import_structure["modeling_vision_encoder_decoder"] = ["VisionEncoderDecoderModel"]

# 检查是否有 TensorFlow 可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则将 TFVisionEncoderDecoderModel 导入到模块导入结构中
    _import_structure["modeling_tf_vision_encoder_decoder"] = ["TFVisionEncoderDecoderModel"]

# 检查是否有 Flax 可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则将 FlaxVisionEncoderDecoderModel 导入到模块导入结构中
    _import_structure["modeling_flax_vision_encoder_decoder"] = ["FlaxVisionEncoderDecoderModel"]

# 如果是 TYPE_CHECKING 模式
if TYPE_CHECKING:
    # 从 configuration_vision_encoder_decoder 模块中导入特定配置类
    from .configuration_vision_encoder_decoder import VisionEncoderDecoderConfig, VisionEncoderDecoderOnnxConfig

    # 检查是否有 torch 可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，则从 modeling_vision_encoder_decoder 模块中导入 VisionEncoderDecoderModel 类
        from .modeling_vision_encoder_decoder import VisionEncoderDecoderModel

    # 检查是否有 TensorFlow 可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，则从 modeling_tf_vision_encoder_decoder 模块中导入 TFVisionEncoderDecoderModel 类
        from .modeling_tf_vision_encoder_decoder import TFVisionEncoderDecoderModel

    # 检查是否有 Flax 可用，如果不可用则抛出 OptionalDependencyNotAvailable 异常
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，则从 modeling_flax_vision_encoder_decoder 模块中导入 FlaxVisionEncoderDecoderModel 类
        from .modeling_flax_vision_encoder_decoder import FlaxVisionEncoderDecoderModel

# 如果不是 TYPE_CHECKING 模式
else:
    # 导入 sys 模块
    import sys

    # 将当前模块定义为延迟加载模块的 LazyModule 对象
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```