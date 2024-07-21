# `.\pytorch\torch\nn\quantized\modules\__init__.py`

```
r"""Quantized Modules.

Note::
    The `torch.nn.quantized` namespace is in the process of being deprecated.
    Please, use `torch.ao.nn.quantized` instead.
"""

# 导入量化模块的各种功能
from torch.ao.nn.quantized.modules.activation import ReLU6, Hardswish, ELU, LeakyReLU, Sigmoid, Softmax, MultiheadAttention, PReLU
from torch.ao.nn.quantized.modules.batchnorm import BatchNorm2d, BatchNorm3d
from torch.ao.nn.quantized.modules.conv import Conv1d, Conv2d, Conv3d
from torch.ao.nn.quantized.modules.conv import ConvTranspose1d, ConvTranspose2d, ConvTranspose3d
from torch.ao.nn.quantized.modules.dropout import Dropout
from torch.ao.nn.quantized.modules.embedding_ops import Embedding, EmbeddingBag
from torch.ao.nn.quantized.modules.functional_modules import FloatFunctional, FXFloatFunctional, QFunctional
from torch.ao.nn.quantized.modules.linear import Linear
from torch.ao.nn.quantized.modules.normalization import LayerNorm, GroupNorm, InstanceNorm1d, InstanceNorm2d, InstanceNorm3d
from torch.ao.nn.quantized.modules.rnn import LSTM

# 导入单独的量化模块
from torch.ao.nn.quantized.modules import MaxPool2d
from torch.ao.nn.quantized.modules import Quantize, DeQuantize

# 下面的导入语句是为了兼容直接从子模块导入的情况，
# 例如 `from torch.nn.quantized.modules.conv import ...`。
# 不需要将它们添加到 `__all__` 中。
from torch.ao.nn.quantized.modules import activation
from torch.ao.nn.quantized.modules import batchnorm
from torch.ao.nn.quantized.modules import conv
from torch.ao.nn.quantized.modules import dropout
from torch.ao.nn.quantized.modules import embedding_ops
from torch.ao.nn.quantized.modules import functional_modules
from torch.ao.nn.quantized.modules import linear
from torch.ao.nn.quantized.modules import normalization
from torch.ao.nn.quantized.modules import rnn
from torch.ao.nn.quantized.modules import utils

# 定义了可以直接导入的符号列表
__all__ = [
    'BatchNorm2d',
    'BatchNorm3d',
    'Conv1d',
    'Conv2d',
    'Conv3d',
    'ConvTranspose1d',
    'ConvTranspose2d',
    'ConvTranspose3d',
    'DeQuantize',
    'ELU',
    'Embedding',
    'EmbeddingBag',
    'GroupNorm',
    'Hardswish',
    'InstanceNorm1d',
    'InstanceNorm2d',
    'InstanceNorm3d',
    'LayerNorm',
    'LeakyReLU',
    'Linear',
    'LSTM',
    'MultiheadAttention',
    'Quantize',
    'ReLU6',
    'Sigmoid',
    'Softmax',
    'Dropout',
    'PReLU',
    # 包装模块
    'FloatFunctional',
    'FXFloatFunctional',
    'QFunctional',
]
```