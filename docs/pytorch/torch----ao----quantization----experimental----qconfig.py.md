# `.\pytorch\torch\ao\quantization\experimental\qconfig.py`

```
# 导入 PyTorch 库
import torch
# 从 torch.ao.quantization.qconfig 模块导入 QConfig 类
from torch.ao.quantization.qconfig import QConfig
# 从 torch.ao.quantization 模块导入 MinMaxObserver 类
from torch.ao.quantization import MinMaxObserver
# 从 torch.ao.quantization.fake_quantize 模块导入 FakeQuantize 类
from torch.ao.quantization.fake_quantize import FakeQuantize
# 从 torch.ao.quantization.experimental.fake_quantize 模块导入 APoTFakeQuantize 类
from torch.ao.quantization.experimental.fake_quantize import APoTFakeQuantize

"""
Default symmetric fake_quant for activations.
"""
# 创建用于激活的默认对称伪量化对象
default_symmetric_fake_quant = FakeQuantize.with_args(observer=MinMaxObserver,
                                                      qscheme=torch.per_tensor_symmetric,
                                                      dtype=torch.quint8)

"""
Default symmetric fake_quant for weights.
"""
# 创建用于权重的默认对称伪量化对象
default_weight_symmetric_fake_quant = FakeQuantize.with_args(observer=MinMaxObserver,
                                                             qscheme=torch.per_tensor_symmetric,
                                                             dtype=torch.qint8)

# uniform activation and weight, b=8 k=2
# 创建用于激活和权重的均匀量化配置，b=8, k=2
uniform_qconfig_8bit = QConfig(activation=default_symmetric_fake_quant,
                               weight=default_weight_symmetric_fake_quant.with_args)

# uniform activation, APoT weight, b=8 k=2
# 创建用于激活的均匀量化配置和 APoT 权重配置，b=8, k=2
apot_weight_qconfig_8bit = QConfig(activation=default_symmetric_fake_quant.with_args,
                                   weight=APoTFakeQuantize.with_args(b=8, k=2, dtype=torch.qint8))

# APoT activation and uniform weight, b=8 k=2
# 创建用于 APoT 激活和均匀权重的量化配置，b=8, k=2
apot_qconfig_8bit = QConfig(activation=APoTFakeQuantize.with_args(b=8, k=2, dtype=torch.quint8),
                            weight=APoTFakeQuantize.with_args(b=8, k=2, dtype=torch.qint8))

# uniform activation and weight, b=4 k=2
# 创建用于激活和权重的均匀量化配置，b=4, k=2
uniform_qconfig_4bit = QConfig(activation=default_symmetric_fake_quant.with_args(quant_min=0,
                                                                                 quant_max=15),
                               weight=default_weight_symmetric_fake_quant.with_args(quant_min=0,
                                                                                    quant_max=15))

# uniform activation, APoT weight, b=4 k=2
# 创建用于激活的均匀量化配置和 APoT 权重配置，b=4, k=2
apot_weight_qconfig_4bit = QConfig(activation=default_symmetric_fake_quant.with_args(quant_min=0,
                                                                                     quant_max=15),
                                   weight=APoTFakeQuantize.with_args(b=4, k=2, dtype=torch.qint8))

# APoT activation and uniform weight, b=4 k=2
# 创建用于 APoT 激活和均匀权重的量化配置，b=4, k=2
apot_qconfig_4bit = QConfig(activation=APoTFakeQuantize.with_args(b=4, k=2, dtype=torch.quint8),
                            weight=APoTFakeQuantize.with_args(b=4, k=2, dtype=torch.qint8))
```