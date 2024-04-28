# `.\transformers\models\vision_encoder_decoder\__init__.py`

```
# 版权声明和许可证信息
# 版权归 The HuggingFace Team 所有，保留所有权利。
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按"原样"分发软件
# 没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关权限和限制的具体语言

# 导入必要的模块和函数
from typing import TYPE_CHECKING
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

# 检查是否有 torch 可用，如果不可用则引发异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_vision_encoder_decoder"] = ["VisionEncoderDecoderModel"]

# 检查是否有 tensorflow 可用，如果不可用则引发异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_vision_encoder_decoder"] = ["TFVisionEncoderDecoderModel"]

# 检查是否有 flax 可用，如果不可用则引发异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_vision_encoder_decoder"] = ["FlaxVisionEncoderDecoderModel"]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 导入配置类
    from .configuration_vision_encoder_decoder import VisionEncoderDecoderConfig, VisionEncoderDecoderOnnxConfig

    # 检查是否有 torch 可用，如果不可用则引发异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 torch 模型类
        from .modeling_vision_encoder_decoder import VisionEncoderDecoderModel

    # 检查是否有 tensorflow 可用，如果不可用则引发异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 tensorflow 模型类
        from .modeling_tf_vision_encoder_decoder import TFVisionEncoderDecoderModel

    # 检查是否有 flax 可用，如果不可用则引发异常
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 导入 flax 模型类
        from .modeling_flax_vision_encoder_decoder import FlaxVisionEncoderDecoderModel

# 如果不是类型检查模式
else:
    import sys

    # 将当前模块设置为 LazyModule
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```