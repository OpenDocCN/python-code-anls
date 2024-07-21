# `.\pytorch\torch\ao\nn\quantized\modules\linear.py`

```
# mypy: allow-untyped-defs
# 允许在类型检查中使用未注释类型定义

from collections.abc import Iterable
# 导入 Iterable 抽象基类，用于支持集合类型操作

import torch
# 导入 PyTorch 库

import torch.nn as nn
# 导入 PyTorch 神经网络模块

import torch.ao.nn.intrinsic as nni
# 导入 PyTorch AO 模块的内部函数

import torch.ao.nn.intrinsic.qat as nniqat
# 导入 PyTorch AO 模块的量化自动校准函数

from torch.nn.utils.fusion import fuse_linear_bn_weights
# 导入 PyTorch 神经网络工具箱中的线性层与批量归一化融合函数

from torch.nn.utils.parametrize import type_before_parametrizations
# 导入 PyTorch 神经网络工具箱中的参数化类型

from typing import Optional
# 导入 Optional 类型提示，用于声明可选参数的类型

from .utils import _quantize_weight, _hide_packed_params_repr, WeightedQuantizedModule
# 从当前包中导入自定义工具函数和量化模块

__all__ = ['LinearPackedParams', 'Linear']
# 将 LinearPackedParams 和 Linear 添加到模块的公开接口列表中

class LinearPackedParams(torch.nn.Module):
    _version = 3
    # 类属性，指定当前类的版本号为 3

    def __init__(self, dtype=torch.qint8):
        super().__init__()
        # 调用父类的初始化方法
        self.dtype = dtype
        # 设置当前实例的数据类型属性
        if self.dtype == torch.qint8:
            # 如果数据类型为 torch.qint8
            wq = torch._empty_affine_quantized([1, 1], scale=1.0, zero_point=0, dtype=torch.qint8)
            # 创建一个 torch.qint8 类型的量化张量 wq
        elif self.dtype == torch.float16:
            # 如果数据类型为 torch.float16
            wq = torch.zeros([1, 1], dtype=torch.float)
            # 创建一个 torch.float 类型的张量 wq
        self.set_weight_bias(wq, None)  # type: ignore[possibly-undefined]
        # 调用 set_weight_bias 方法设置权重和偏置，忽略类型检查的可能未定义警告

    @torch.jit.export
    # 将方法标记为 Torch 脚本导出函数
    def set_weight_bias(self, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> None:
        # 设置权重和偏置的方法，接受 torch.Tensor 类型的 weight 和可选的 bias 参数，无返回值
        if self.dtype == torch.qint8:
            self._packed_params = torch.ops.quantized.linear_prepack(weight, bias)
            # 使用量化运算符进行线性预打包，生成 _packed_params
        elif self.dtype == torch.float16:
            self._packed_params = torch.ops.quantized.linear_prepack_fp16(weight, bias)
            # 使用量化运算符进行 FP16 线性预打包，生成 _packed_params
        else:
            raise RuntimeError('Unsupported dtype on dynamic quantized linear!')
            # 抛出运行时错误，不支持的动态量化线性层数据类型

    @torch.jit.export
    # 将方法标记为 Torch 脚本导出函数
    def _weight_bias(self):
        # 获取权重和偏置的方法，根据 dtype 返回相应的解包结果
        if self.dtype == torch.qint8:
            return torch.ops.quantized.linear_unpack(self._packed_params)
            # 使用量化运算符进行线性解包，返回解包结果
        elif self.dtype == torch.float16:
            return torch.ops.quantized.linear_unpack_fp16(self._packed_params)
            # 使用量化运算符进行 FP16 线性解包，返回解包结果
        else:
            raise RuntimeError('Unsupported dtype on dynamic quantized linear!')
            # 抛出运行时错误，不支持的动态量化线性层数据类型

    def forward(self, x):
        # 前向传播方法，输入 x 返回 x，实际中应包含具体的前向计算逻辑
        return x

    # Version 1
    #   self
    #   |--- weight : Tensor
    #   |--- bias : Tensor
    #
    # Version 2
    #   self
    #   |--- weight : Tensor
    #   |--- bias : Tensor
    #   |--- dtype : torch.dtype
    #
    # Version 3
    #   self
    #   |--- _packed_params : (Tensor, Tensor) representing (weight, bias)
    #                         of LinearPackedParams
    #   |--- dtype : torch.dtype
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # 自定义保存状态到 state_dict 的方法，继承并扩展父类的保存逻辑
        super()._save_to_state_dict(destination, prefix, keep_vars)
        # 调用父类的保存方法
        destination[prefix + 'dtype'] = self.dtype
        # 将当前实例的 dtype 属性添加到目标 state_dict
        destination[prefix + '_packed_params'] = self._weight_bias()
        # 将当前实例的 _weight_bias 方法返回值添加到目标 state_dict
    # 从状态字典中加载模型的参数和状态信息
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # 从元数据中获取版本号，如果不存在或者小于2，则使用 torch.qint8 数据类型
        version = local_metadata.get('version', None)
        if version is None or version < 2:
            self.dtype = torch.qint8
        else:
            # 如果版本号大于等于2，则从状态字典中获取 dtype，并从状态字典中移除
            self.dtype = state_dict[prefix + 'dtype']
            state_dict.pop(prefix + 'dtype')

        # 如果版本号不存在或者小于3，则设置权重和偏置
        if version is None or version < 3:
            self.set_weight_bias(state_dict[prefix + 'weight'], state_dict[prefix + 'bias'])
            # 从状态字典中移除权重和偏置
            state_dict.pop(prefix + 'weight')
            state_dict.pop(prefix + 'bias')

        # 如果版本号等于3，则从状态字典中获取打包的参数并设置权重和偏置
        if version == 3:
            weight, bias = state_dict[prefix + '_packed_params']
            state_dict.pop(prefix + '_packed_params')
            self.set_weight_bias(weight, bias)

        # 调用父类的方法来加载状态字典中的其他信息
        super()._load_from_state_dict(state_dict, prefix, local_metadata, False,
                                      missing_keys, unexpected_keys, error_msgs)


    # 返回模型对象的字符串表示形式
    def __repr__(self):
        return self._weight_bias().__repr__()
class Linear(WeightedQuantizedModule):
    r"""
    A quantized linear module with quantized tensor as inputs and outputs.
    We adopt the same interface as `torch.nn.Linear`, please see
    https://pytorch.org/docs/stable/nn.html#torch.nn.Linear for documentation.

    Similar to :class:`~torch.nn.Linear`, attributes will be randomly
    initialized at module creation time and will be overwritten later

    Attributes:
        weight (Tensor): the non-learnable quantized weights of the module of
                         shape :math:`(\text{out\_features}, \text{in\_features})`.
        bias (Tensor): the non-learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized to zero.
        scale: `scale` parameter of output Quantized Tensor, type: double
        zero_point: `zero_point` parameter for output Quantized Tensor, type: long

    Examples::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> m = nn.quantized.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> # xdoctest: +SKIP
        >>> input = torch.quantize_per_tensor(input, 1.0, 0, torch.quint8)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    _version = 3
    _FLOAT_MODULE = (nn.Linear, nn.modules.linear.NonDynamicallyQuantizableLinear)

    def __init__(self, in_features, out_features, bias_=True,
                 dtype=torch.qint8):
        super().__init__()
        # 初始化量化线性模块
        self.in_features = in_features  # 输入特征数
        self.out_features = out_features  # 输出特征数
        bias = None
        if bias_:
            bias = torch.zeros(out_features, dtype=torch.float)  # 初始化偏置为零向量，长度为输出特征数

        if dtype == torch.qint8:
            qweight = torch._empty_affine_quantized(
                [out_features, in_features], scale=1, zero_point=0, dtype=torch.qint8)
        elif dtype == torch.float16:
            qweight = torch.zeros([out_features, in_features], dtype=torch.float)  # 若dtype为float16，则初始化权重为零矩阵
        else:
            raise RuntimeError('Unsupported dtype specified for quantized Linear!')  # 不支持的dtype类型错误

        self._packed_params = LinearPackedParams(dtype)  # 初始化线性参数的打包参数对象
        self._packed_params.set_weight_bias(qweight, bias)  # 设置权重和偏置
        self.scale = 1.0  # 输出量化张量的缩放参数初始化为1.0
        self.zero_point = 0  # 输出量化张量的零点初始化为0

    def _get_name(self):
        return 'QuantizedLinear'  # 返回模块的名称字符串

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, scale={self.scale}, ' \
               f'zero_point={self.zero_point}, qscheme={self.weight().qscheme()}'
        # 返回额外的字符串表示，包括输入和输出特征数、缩放因子、零点、权重的量化方案

    def __repr__(self):
        return _hide_packed_params_repr(self, LinearPackedParams)
        # 返回模块的字符串表示形式，隐藏打包参数的具体表示
    # 实现前向传播函数，调用量化的线性运算函数
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.quantized.linear(
            x, self._packed_params._packed_params, self.scale, self.zero_point)

    # ===== 序列化方法 =====
    # 特别注意的是，我们需要将权重解包成常规的量化张量形式进行序列化。
    # 打包后的权重不应该存在于创建它们的进程之外，而应该从量化张量权重派生。
    #
    # Version 1
    #   self
    #   |--- scale : float
    #   |--- zero_point : int
    #   |--- weight : Tensor
    #   |--- bias : Tensor
    #
    # Version 2
    #   self
    #   |--- scale : float
    #   |--- zero_point : int
    #   |--- _packed_params : Module
    #        |--- weight : Tensor
    #        |--- bias : Tensor
    #
    # Version 3
    #   self
    #   |--- scale : float
    #   |--- zero_point : int
    #   |--- _packed_params : Module
    #        |--- _packed_params : (Tensor, Tensor) representing weight, bias
    #                              of LinearPackedParams C++ struct
    #
    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'scale'] = torch.tensor(self.scale)
        destination[prefix + 'zero_point'] = torch.tensor(self.zero_point)

    # ===== 反序列化方法 =====
    # 与序列化方法对应，我们必须将序列化后的量化张量权重打包成其压缩格式，以供 FBGEMM 操作使用。
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self.scale = float(state_dict[prefix + 'scale'])
        state_dict.pop(prefix + 'scale')

        self.zero_point = int(state_dict[prefix + 'zero_point'])
        state_dict.pop(prefix + 'zero_point')

        version = local_metadata.get('version', None)

        if version is None or version == 1:
            # 将参数移动到 LinearPackedParameters 子模块中
            weight = state_dict.pop(prefix + 'weight')
            bias = state_dict.pop(prefix + 'bias')
            state_dict.update({prefix + '_packed_params.weight': weight,
                               prefix + '_packed_params.bias': bias})

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, False,
            missing_keys, unexpected_keys, error_msgs)

    # 使用函数而不是属性，确保 JIT 序列化时不将其注册为属性
    def _weight_bias(self):
        return self._packed_params._weight_bias()

    # 返回权重张量
    def weight(self):
        return self._weight_bias()[0]

    # 返回偏置张量
    def bias(self):
        return self._weight_bias()[1]

    # 设置权重和偏置
    def set_weight_bias(self, w: torch.Tensor, b: Optional[torch.Tensor]) -> None:
        self._packed_params.set_weight_bias(w, b)

    @classmethod
    # 从观察到的浮点数模块创建一个量化模块

    @classmethod
    r"""Create a quantized module from an observed float module
    """
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # 检查模块是否具有 'weight_fake_quant' 属性
        if hasattr(mod, 'weight_fake_quant'):
            # 如果模块类型为 nniqat.LinearBn1d，则融合线性层和批标准化的权重和偏置
            if type_before_parametrizations(mod) == nniqat.LinearBn1d:
                mod.weight, mod.bias = fuse_linear_bn_weights(
                    mod.weight, mod.bias, mod.bn.running_mean, mod.bn.running_var,
                    mod.bn.eps, mod.bn.weight, mod.bn.bias)
            # 获取权重后处理器和激活后处理器
            weight_post_process = mod.weight_fake_quant
            activation_post_process = mod.activation_post_process
        else:
            # 如果模块没有 'weight_fake_quant' 属性，则进行类型检查和错误处理
            # 由于此函数不参与 JIT，可以忽略赋值时的类型不匹配问题
            # 同时，mypy 对于未实现可迭代对象也有问题，这些问题也需要忽略
            if not isinstance(cls._FLOAT_MODULE, Iterable):
                cls._FLOAT_MODULE = [cls._FLOAT_MODULE]  # type: ignore[assignment]
            # 拼接支持的模块类型字符串
            supported_modules = ', '.join([float_mod.__name__ for float_mod in cls._FLOAT_MODULE])  # type: ignore[attr-defined]
            # 构建错误消息，说明只支持特定的模块类型
            error_msg = f'nnq.{cls.__name__}.from_float only works for {supported_modules}, but got: {type(mod)}'
            # 断言输入模块类型在支持的模块类型之内，否则抛出错误消息
            assert type_before_parametrizations(mod) in cls._FLOAT_MODULE, error_msg.format()  # type: ignore[attr-defined]
            # 断言输入浮点模块具有 'qconfig' 属性
            assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
            # 获取激活后处理器
            activation_post_process = mod.activation_post_process
            # 如果模块类型为 nni.LinearReLU，则仅保留其第一个元素
            if type_before_parametrizations(mod) == nni.LinearReLU:
                mod = mod[0]
            # 获取权重后处理器，如果不存在 'weight_fake_quant' 属性，则使用 'qconfig' 定义的权重后处理器
            weight_post_process = mod.qconfig.weight() if not hasattr(mod, "weight_fake_quant") else mod.weight_fake_quant

        # 如果不使用预先计算的伪量化，观察器可能尚未调用
        if not use_precomputed_fake_quant:
            # 调用权重后处理器，观察权重
            weight_post_process(mod.weight)
        # 获取权重后处理器的数据类型
        dtype = weight_post_process.dtype
        # 计算激活函数的量化参数（量化比例和零点）
        act_scale, act_zp = activation_post_process.calculate_qparams()
        # 断言权重后处理器的数据类型为 torch.qint8
        assert dtype == torch.qint8, 'Weight observer must have dtype torch.qint8'
        # 对模块的权重进行量化
        qweight = _quantize_weight(mod.weight.float(), weight_post_process)
        # 创建量化线性层
        qlinear = cls(mod.in_features,
                      mod.out_features,
                      dtype=dtype)
        # 设置量化线性层的权重和偏置
        qlinear.set_weight_bias(qweight, mod.bias)
        # 设置量化线性层的量化比例和零点
        qlinear.scale = float(act_scale)
        qlinear.zero_point = int(act_zp)
        # 返回量化线性层
        return qlinear
    def from_reference(cls, ref_qlinear, output_scale, output_zero_point):
        r"""Create a (fbgemm/qnnpack) quantized module from a reference quantized module

        Args:
            ref_qlinear (Module): a reference quantized linear module, either produced by torch.ao.quantization
                          utilities or provided by the user
            output_scale (float): scale for output Tensor
            output_zero_point (int): zero point for output Tensor
        """
        # 根据给定的参考量化线性模块创建一个新的量化模块
        qlinear = cls(
            ref_qlinear.in_features,  # 使用参考模块的输入特征数初始化新模块
            ref_qlinear.out_features)  # 使用参考模块的输出特征数初始化新模块
        # 获取参考模块的量化权重
        qweight = ref_qlinear.get_quantized_weight()
        # 设置新模块的量化权重和偏置
        qlinear.set_weight_bias(qweight, ref_qlinear.bias)

        # 设置新模块的输出缩放因子
        qlinear.scale = float(output_scale)
        # 设置新模块的输出零点
        qlinear.zero_point = int(output_zero_point)
        # 返回创建的新的量化模块
        return qlinear
```