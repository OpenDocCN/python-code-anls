# `Bert-VITS2\onnx_modules\V230\models_onnx.py`

```

# 导入必要的库
import math
import torch
from torch import nn
from torch.nn import functional as F

# 导入自定义的模块
import commons
import modules
from . import attentions_onnx

# 从 torch.nn 中导入一些类
from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

# 从 commons 模块中导入 init_weights 和 get_padding 函数
from commons import init_weights, get_padding
# 从 text 模块中导入 symbols, num_tones, num_languages
from .text import symbols, num_tones, num_languages

```