# `Bert-VITS2\oldVersion\V210\models.py`

```

import math  # 导入数学库
import torch  # 导入PyTorch库
from torch import nn  # 从PyTorch库中导入神经网络模块
from torch.nn import functional as F  # 从PyTorch库中导入神经网络函数模块

import commons  # 导入自定义的commons模块
import modules  # 导入自定义的modules模块
import attentions  # 导入自定义的attentions模块
import monotonic_align  # 导入自定义的monotonic_align模块

from torch.nn import Conv1d, ConvTranspose1d, Conv2d  # 从PyTorch库中导入一维卷积、一维转置卷积和二维卷积
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm  # 从PyTorch库中导入权重归一化、移除权重归一化和谱归一化
from vector_quantize_pytorch import VectorQuantize  # 从自定义的vector_quantize_pytorch模块中导入向量量化

from commons import init_weights, get_padding  # 从自定义的commons模块中导入初始化权重和获取填充

from .text import symbols, num_tones, num_languages  # 从当前目录下的text模块中导入symbols、num_tones和num_languages

```