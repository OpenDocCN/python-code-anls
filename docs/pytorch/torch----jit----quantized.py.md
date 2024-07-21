# `.\pytorch\torch\jit\quantized.py`

```py
# mypy: allow-untyped-defs
# 导入 torch 库
import torch

# 定义 QuantizedLinear 类，继承自 torch.jit.ScriptModule
class QuantizedLinear(torch.jit.ScriptModule):
    # 初始化方法，抛出 RuntimeError 异常
    def __init__(self, other):
        raise RuntimeError(
            "torch.jit.QuantizedLinear is no longer supported. Please use "
            "torch.ao.nn.quantized.dynamic.Linear instead."
        )

# FP16 weights
# 定义 QuantizedLinearFP16 类，继承自 torch.jit.ScriptModule
class QuantizedLinearFP16(torch.jit.ScriptModule):
    # 初始化方法，调用父类的初始化方法，并抛出 RuntimeError 异常
    def __init__(self, other):
        super().__init__()
        raise RuntimeError(
            "torch.jit.QuantizedLinearFP16 is no longer supported. "
            "Please use the torch.ao.nn.quantized.dynamic.Linear instead."
        )

# 定义 QuantizedRNNCellBase 类，继承自 torch.jit.ScriptModule
# 量化 RNN 单元的基础实现
class QuantizedRNNCellBase(torch.jit.ScriptModule):
    # 初始化方法，抛出 RuntimeError 异常
    def __init__(self, other):
        raise RuntimeError(
            "torch.jit.QuantizedRNNCellBase is no longer supported. "
            "Please use the torch.ao.nn.quantized.dynamic.RNNCell instead."
        )

# 定义 QuantizedRNNCell 类，继承自 QuantizedRNNCellBase
# 量化 RNN 单元的实现
class QuantizedRNNCell(QuantizedRNNCellBase):
    # 初始化方法，调用父类的初始化方法，并抛出 RuntimeError 异常
    def __init__(self, other):
        super().__init__(other)
        raise RuntimeError(
            "torch.jit.QuantizedRNNCell is no longer supported. "
            "Please use the torch.ao.nn.quantized.dynamic.RNNCell instead."
        )

# 定义 QuantizedLSTMCell 类，继承自 QuantizedRNNCellBase
# 量化 LSTM 单元的实现
class QuantizedLSTMCell(QuantizedRNNCellBase):
    # 初始化方法，调用父类的初始化方法，并抛出 RuntimeError 异常
    def __init__(self, other):
        super().__init__(other)
        raise RuntimeError(
            "torch.jit.QuantizedLSTMCell is no longer supported. "
            "Please use the torch.ao.nn.quantized.dynamic.LSTMCell instead."
        )

# 定义 QuantizedGRUCell 类，继承自 QuantizedRNNCellBase
# 量化 GRU 单元的实现
class QuantizedGRUCell(QuantizedRNNCellBase):
    # 初始化方法，调用父类的初始化方法，并抛出 RuntimeError 异常
    def __init__(self, other):
        super().__init__(other)
        raise RuntimeError(
            "torch.jit.QuantizedGRUCell is no longer supported. "
            "Please use the torch.ao.nn.quantized.dynamic.GRUCell instead."
        )

# 定义 QuantizedRNNBase 类，继承自 torch.jit.ScriptModule
# 量化 RNN 基础实现
class QuantizedRNNBase(torch.jit.ScriptModule):
    # 初始化方法，抛出 RuntimeError 异常
    def __init__(self, other, dtype=torch.int8):
        raise RuntimeError(
            "torch.jit.QuantizedRNNBase is no longer supported. "
            "Please use the torch.ao.nn.quantized.dynamic instead."
        )

# 定义 QuantizedLSTM 类，继承自 QuantizedRNNBase
# 量化 LSTM 实现
class QuantizedLSTM(QuantizedRNNBase):
    # 初始化方法，抛出 RuntimeError 异常
    def __init__(self, other, dtype):
        raise RuntimeError(
            "torch.jit.QuantizedLSTM is no longer supported. "
            "Please use the torch.ao.nn.quantized.dynamic.LSTM instead."
        )

# 定义 QuantizedGRU 类，继承自 QuantizedRNNBase
# 量化 GRU 实现
class QuantizedGRU(QuantizedRNNBase):
    # 初始化方法，抛出 RuntimeError 异常
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "torch.jit.QuantizedGRU is no longer supported. "
            "Please use the torch.ao.nn.quantized.dynamic.GRU instead."
        )

# 定义 quantize_rnn_cell_modules 函数
# 抛出 RuntimeError 异常
def quantize_rnn_cell_modules(module):
    raise RuntimeError(
        "quantize_rnn_cell_modules function is no longer supported. "
        "Please use torch.ao.quantization.quantize_dynamic API instead."
    )

# 定义 quantize_linear_modules 函数
# 抛出 RuntimeError 异常
def quantize_linear_modules(module, dtype=torch.int8):
    raise RuntimeError(
        "quantize_linear_modules function is no longer supported. "
        "Please use torch.ao.quantization.quantize_dynamic API instead."
    )
# 定义 quantize_rnn_modules 函数，用于对 RNN 模块进行量化处理
def quantize_rnn_modules(module, dtype=torch.int8):
    # 抛出运行时错误，提醒用户该函数已不再支持使用
    raise RuntimeError(
        "quantize_rnn_modules function is no longer supported. "
        "Please use torch.ao.quantization.quantize_dynamic API instead."
    )
```