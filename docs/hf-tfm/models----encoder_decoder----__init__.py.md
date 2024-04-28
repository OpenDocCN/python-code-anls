# `.\models\encoder_decoder\__init__.py`

```
# 导入类型检查模块
from typing import TYPE_CHECKING

# 导入必要的模块和函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_torch_available,
)

# 定义要导入的结构
_import_structure = {"configuration_encoder_decoder": ["EncoderDecoderConfig"]}

# 如果没有安装 torch，则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_encoder_decoder"] = ["EncoderDecoderModel"]

# 如果没有安装 tensorflow，则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_encoder_decoder"] = ["TFEncoderDecoderModel"]

# 如果没有安装 flax，则引发 OptionalDependencyNotAvailable 异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_encoder_decoder"] = ["FlaxEncoderDecoderModel"]

# 如果是类型检查，导入相应的模块和函数
if TYPE_CHECKING:
    from .configuration_encoder_decoder import EncoderDecoderConfig
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_encoder_decoder import EncoderDecoderModel

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_encoder_decoder import TFEncoderDecoderModel

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_flax_encoder_decoder import FlaxEncoderDecoderModel

# 如果不是类型检查，将模块设为 LazyModule
else:
    import sys
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```