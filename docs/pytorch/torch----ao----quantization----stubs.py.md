# `.\pytorch\torch\ao\quantization\stubs.py`

```
# mypy: allow-untyped-defs

# 导入 torch 的 nn 模块
from torch import nn

# 定义量化前置模块 QuantStub
class QuantStub(nn.Module):
    r"""Quantize stub module, before calibration, this is same as an observer,
    it will be swapped as `nnq.Quantize` in `convert`.

    Args:
        qconfig: quantization configuration for the tensor,
            if qconfig is not provided, we will get qconfig from parent modules
    """
    
    # 初始化函数，接收一个可选的量化配置 qconfig
    def __init__(self, qconfig=None):
        super().__init__()
        # 如果提供了 qconfig，将其赋值给当前对象的 qconfig 属性
        if qconfig:
            self.qconfig = qconfig

    # 前向传播函数，直接返回输入 x
    def forward(self, x):
        return x


# 定义去量化模块 DeQuantStub
class DeQuantStub(nn.Module):
    r"""Dequantize stub module, before calibration, this is same as identity,
    this will be swapped as `nnq.DeQuantize` in `convert`.

    Args:
        qconfig: quantization configuration for the tensor,
            if qconfig is not provided, we will get qconfig from parent modules
    """
    
    # 初始化函数，接收一个可选的量化配置 qconfig
    def __init__(self, qconfig=None):
        super().__init__()
        # 如果提供了 qconfig，将其赋值给当前对象的 qconfig 属性
        if qconfig:
            self.qconfig = qconfig

    # 前向传播函数，直接返回输入 x
    def forward(self, x):
        return x


# 定义量化包装器 QuantWrapper 类
class QuantWrapper(nn.Module):
    r"""A wrapper class that wraps the input module, adds QuantStub and
    DeQuantStub and surround the call to module with call to quant and dequant
    modules.

    This is used by the `quantization` utility functions to add the quant and
    dequant modules, before `convert` function `QuantStub` will just be observer,
    it observes the input tensor, after `convert`, `QuantStub`
    will be swapped to `nnq.Quantize` which does actual quantization. Similarly
    for `DeQuantStub`.
    """
    
    # 定义类属性：QuantStub, DeQuantStub 和 module
    quant: QuantStub
    dequant: DeQuantStub
    module: nn.Module

    # 初始化函数，接收一个 nn.Module 对象作为参数 module
    def __init__(self, module):
        super().__init__()
        # 从 module 中获取 qconfig 属性
        qconfig = getattr(module, "qconfig", None)
        # 添加 QuantStub 模块到当前对象，使用传入的 qconfig
        self.add_module('quant', QuantStub(qconfig))
        # 添加 DeQuantStub 模块到当前对象，使用传入的 qconfig
        self.add_module('dequant', DeQuantStub(qconfig))
        # 添加传入的 module 到当前对象
        self.add_module('module', module)
        # 设置当前对象的训练模式与传入 module 一致
        self.train(module.training)

    # 前向传播函数，接收输入 X
    def forward(self, X):
        # 通过 QuantStub 对输入 X 进行量化
        X = self.quant(X)
        # 对量化后的结果 X 进行 module 的前向传播
        X = self.module(X)
        # 对 module 前向传播结果进行去量化
        return self.dequant(X)
```