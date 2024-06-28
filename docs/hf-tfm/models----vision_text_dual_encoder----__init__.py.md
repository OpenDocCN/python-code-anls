# `.\models\vision_text_dual_encoder\__init__.py`

```
# 导入必要的模块和函数，包括一些自定义的异常和LazyModule
from typing import TYPE_CHECKING
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_torch_available,
)

# 定义模块的导入结构，用于延迟加载模块
_import_structure = {
    "configuration_vision_text_dual_encoder": ["VisionTextDualEncoderConfig"],
    "processing_vision_text_dual_encoder": ["VisionTextDualEncoderProcessor"],
}

# 检查是否可用torch，如果不可用则抛出OptionalDependencyNotAvailable异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则将VisionTextDualEncoderModel添加到_import_structure中
    _import_structure["modeling_vision_text_dual_encoder"] = ["VisionTextDualEncoderModel"]

# 检查是否可用flax，如果不可用则抛出OptionalDependencyNotAvailable异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则将FlaxVisionTextDualEncoderModel添加到_import_structure中
    _import_structure["modeling_flax_vision_text_dual_encoder"] = ["FlaxVisionTextDualEncoderModel"]

# 检查是否可用tensorflow，如果不可用则抛出OptionalDependencyNotAvailable异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果可用，则将TFVisionTextDualEncoderModel添加到_import_structure中
    _import_structure["modeling_tf_vision_text_dual_encoder"] = ["TFVisionTextDualEncoderModel"]

# 如果是类型检查模式，则从相应的模块导入具体的类
if TYPE_CHECKING:
    from .configuration_vision_text_dual_encoder import VisionTextDualEncoderConfig
    from .processing_vision_text_dual_encoder import VisionTextDualEncoderProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，则从modeling_vision_text_dual_encoder模块导入VisionTextDualEncoderModel类
        from .modeling_vision_text_dual_encoder import VisionTextDualEncoderModel

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，则从modeling_flax_vision_text_dual_encoder模块导入FlaxVisionTextDualEncoderModel类
        from .modeling_flax_vision_text_dual_encoder import FlaxVisionTextDualEncoderModel

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        # 如果可用，则从modeling_tf_vision_text_dual_encoder模块导入TFVisionTextDualEncoderModel类
        from .modeling_tf_vision_text_dual_encoder import TFVisionTextDualEncoderModel

# 如果不是类型检查模式，则将LazyModule应用到当前模块，以支持延迟导入
else:
    import sys

    # 将当前模块替换为LazyModule，用于延迟导入
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
```