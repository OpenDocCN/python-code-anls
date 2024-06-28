# `.\models\encoder_decoder\__init__.py`

```
# 版权声明和许可证信息

# 引入类型检查标记
from typing import TYPE_CHECKING

# 从 utils 模块中导入必要的异常和工具函数
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_torch_available,
)

# 定义模块的导入结构，包含配置模块中的 EncoderDecoderConfig
_import_structure = {"configuration_encoder_decoder": ["EncoderDecoderConfig"]}

# 检查是否支持 PyTorch，若不支持则抛出异常
try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果支持 PyTorch，则导入 EncoderDecoderModel 模型
    _import_structure["modeling_encoder_decoder"] = ["EncoderDecoderModel"]

# 检查是否支持 TensorFlow，若不支持则抛出异常
try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果支持 TensorFlow，则导入 TFEncoderDecoderModel 模型
    _import_structure["modeling_tf_encoder_decoder"] = ["TFEncoderDecoderModel"]

# 检查是否支持 Flax，若不支持则抛出异常
try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    # 如果支持 Flax，则导入 FlaxEncoderDecoderModel 模型
    _import_structure["modeling_flax_encoder_decoder"] = ["FlaxEncoderDecoderModel"]

# 如果当前为类型检查模式
if TYPE_CHECKING:
    # 从当前模块中导入 EncoderDecoderConfig 类型
    from .configuration_encoder_decoder import EncoderDecoderConfig

    # 检查是否支持 PyTorch，若支持则从 modeling_encoder_decoder 中导入 EncoderDecoderModel 类型
    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_encoder_decoder import EncoderDecoderModel

    # 检查是否支持 TensorFlow，若支持则从 modeling_tf_encoder_decoder 中导入 TFEncoderDecoderModel 类型
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_encoder_decoder import TFEncoderDecoderModel

    # 检查是否支持 Flax，若支持则从 modeling_flax_encoder_decoder 中导入 FlaxEncoderDecoderModel 类型
    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_flax_encoder_decoder import FlaxEncoderDecoderModel

# 如果不是类型检查模式，则将当前模块设为延迟加载模块
else:
    import sys

    # 动态设置当前模块为 _LazyModule 类型
    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
```