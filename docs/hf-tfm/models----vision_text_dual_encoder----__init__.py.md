# `.\transformers\models\vision_text_dual_encoder\__init__.py`

```py
# 版权声明和许可证信息
# 版权声明和许可证信息，指定了代码的版权和许可证信息
# 根据 Apache 许可证 2.0 版本，除非符合许可证的规定，否则不得使用此文件
# 可以在 http://www.apache.org/licenses/LICENSE-2.0 获取许可证的副本
# 根据适用法律或书面同意，分发的软件是基于“原样”分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关特定语言的权限和限制

# 导入必要的模块和函数
# 导入类型检查相关的模块
from typing import TYPE_CHECKING
# 导入自定义的异常类和函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_torch_available,
)

# 定义模块的导入结构
_import_structure = {
    "configuration_vision_text_dual_encoder": ["VisionTextDualEncoderConfig"],
    "processing_vision_text_dual_encoder": ["VisionTextDualEncoderProcessor"],
}

# 检查是否有 torch 库可用，如果不可用则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 torch 可用，则添加 VisionTextDualEncoderModel 到导入结构中
    _import_structure["modeling_vision_text_dual_encoder"] = ["VisionTextDualEncoderModel"]

# 检查是否有 flax 库可用，如果不可用则抛出异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 flax 可用，则添加 FlaxVisionTextDualEncoderModel 到导入结构中
    _import_structure["modeling_flax_vision_text_dual_encoder"] = ["FlaxVisionTextDualEncoderModel"]

# 检查是否有 tensorflow 库可用，如果不可用则抛出异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果 tensorflow 可用，则添加 TFVisionTextDualEncoderModel 到导入结构中
    _import_structure["modeling_tf_vision_text_dual_encoder"] = ["TFVisionTextDualEncoderModel"]

# 如果是类型检查模式
if TYPE_CHECKING:
    # 导入类型检查所需的模块和类
    from .configuration_vision_text_dual_encoder import VisionTextDualEncoderConfig
    from .processing_vision_text_dual_encoder import VisionTextDualEncoderProcessor

    # 检查是否有 torch 库可用，如果不可用则抛出异常
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 torch 可用，则导入 VisionTextDualEncoderModel
        from .modeling_vision_text_dual_encoder import VisionTextDualEncoderModel

    # 检查是否有 flax 库可用，如果不可用则抛出异常
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 flax 可用，则导入 FlaxVisionTextDualEncoderModel
        from .modeling_flax_vision_text_dual_encoder import FlaxVisionTextDualEncoderModel

    # 检查是否有 tensorflow 库可用，如果不可用则抛出异常
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果 tensorflow 可用，则导入 TFVisionTextDualEncoderModel
        from .modeling_tf_vision_text_dual_encoder import TFVisionTextDualEncoderModel

# 如果不是类型检查模式
else:
    import sys

    # 将当前模块设置为 LazyModule，延迟导入模块
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```