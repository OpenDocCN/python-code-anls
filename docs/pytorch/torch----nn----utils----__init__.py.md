# `.\pytorch\torch\nn\utils\__init__.py`

```
# 导入模块中的特定功能和对象，使它们可以在当前代码中直接使用

from . import parametrizations, rnn, stateless
# 从当前包中导入 parametrizations, rnn, stateless 模块或对象

from .clip_grad import clip_grad_norm, clip_grad_norm_, clip_grad_value_
# 从当前包中导入 clip_grad_norm, clip_grad_norm_, clip_grad_value_ 函数

from .convert_parameters import parameters_to_vector, vector_to_parameters
# 从当前包中导入 parameters_to_vector, vector_to_parameters 函数

from .fusion import (
    fuse_conv_bn_eval,
    fuse_conv_bn_weights,
    fuse_linear_bn_eval,
    fuse_linear_bn_weights,
)
# 从当前包中导入以下函数：fuse_conv_bn_eval, fuse_conv_bn_weights,
# fuse_linear_bn_eval, fuse_linear_bn_weights

from .init import skip_init
# 从当前包中导入 skip_init 函数

from .memory_format import (
    convert_conv2d_weight_memory_format,
    convert_conv3d_weight_memory_format,
)
# 从当前包中导入以下函数：convert_conv2d_weight_memory_format,
# convert_conv3d_weight_memory_format

from .spectral_norm import remove_spectral_norm, spectral_norm
# 从当前包中导入 remove_spectral_norm, spectral_norm 函数

from .weight_norm import remove_weight_norm, weight_norm
# 从当前包中导入 remove_weight_norm, weight_norm 函数

# 定义一个列表，包含了当前模块中所有要公开的函数和对象的名称
__all__ = [
    "clip_grad_norm",
    "clip_grad_norm_",
    "clip_grad_value_",
    "convert_conv2d_weight_memory_format",
    "convert_conv3d_weight_memory_format",
    "fuse_conv_bn_eval",
    "fuse_conv_bn_weights",
    "fuse_linear_bn_eval",
    "fuse_linear_bn_weights",
    "parameters_to_vector",
    "parametrizations",
    "remove_spectral_norm",
    "remove_weight_norm",
    "rnn",
    "skip_init",
    "spectral_norm",
    "stateless",
    "vector_to_parameters",
    "weight_norm",
]
# 定义一个特殊变量 __all__，其中包含了当前模块中所有希望在外部公开的函数和对象的名称
```