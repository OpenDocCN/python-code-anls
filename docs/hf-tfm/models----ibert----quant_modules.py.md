# `.\models\ibert\quant_modules.py`

```
# 设置文件编码为 UTF-8
# 版权声明，包括作者信息和版权信息
# 版权所有 (c) 2021, NVIDIA CORPORATION. 保留所有权利。
#
# 根据 Apache 许可证 2.0 版本使用本文件
# 除非符合许可证规定，否则不得使用本文件
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发本软件
# 没有任何明示或暗示的担保或条件
# 有关特定语言的权限，请参阅许可证

import decimal  # 导入 decimal 库

import numpy as np  # 导入 numpy 库
import torch  # 导入 torch 库
from torch import nn  # 从 torch 中导入 nn 模块
from torch.autograd import Function  # 从 torch.autograd 中导入 Function 类

from ...utils import logging  # 从相对路径中导入 logging 模块

# 获取 logger 对象，用于记录日志信息
logger = logging.get_logger(__name__)


class QuantEmbedding(nn.Module):
    """
    `torch.nn.Embedding` 的量化版本。在 `torch.nn.Embedding` 的基础上增加了量化特定的参数。

    Args:
        weight_bit (`int`, *optional*, defaults to `8`):
            权重的量化位宽。
        momentum (`float`, *optional*, defaults to `0.95`):
            更新激活量化范围的动量。
        quant_mode (`bool`, *optional*, defaults to `False`):
            是否对该层进行量化。
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        _weight=None,
        weight_bit=8,
        momentum=0.95,
        quant_mode=False,
    ):
        super().__init__()
        self.num_ = num_embeddings  # 设置 num_ 属性为 num_embeddings
        self.dim = embedding_dim  # 设置 dim 属性为 embedding_dim
        self.padding_idx = padding_idx  # 设置 padding_idx 属性
        self.max_norm = max_norm  # 设置 max_norm 属性
        self.norm_type = norm_type  # 设置 norm_type 属性
        self.scale_grad_by_freq = scale_grad_by_freq  # 设置 scale_grad_by_freq 属性
        self.sparse = sparse  # 设置 sparse 属性

        self.weight = nn.Parameter(torch.zeros([num_embeddings, embedding_dim]))  # 初始化权重参数
        self.register_buffer("weight_scaling_factor", torch.zeros(1))  # 注册缓冲区 weight_scaling_factor
        self.register_buffer("weight_integer", torch.zeros_like(self.weight))  # 注册缓冲区 weight_integer

        self.weight_bit = weight_bit  # 设置 weight_bit 属性
        self.momentum = momentum  # 设置 momentum 属性
        self.quant_mode = quant_mode  # 设置 quant_mode 属性
        self.percentile_mode = False  # 设置 percentile_mode 属性为 False
        self.weight_function = SymmetricQuantFunction.apply  # 设置 weight_function 属性为 SymmetricQuantFunction.apply
    # 定义前向传播函数，用于模型的正向计算
    def forward(self, x, positions=None, incremental_state=None):
        # 如果不处于量化模式，则直接返回原始的嵌入结果和空的状态
        if not self.quant_mode:
            return (
                nn.functional.embedding(
                    x,
                    self.weight,
                    self.padding_idx,
                    self.max_norm,
                    self.norm_type,
                    self.scale_grad_by_freq,
                    self.sparse,
                ),
                None,
            )

        # 获取模型的权重
        w = self.weight
        # 分离权重数据并进行转换
        w_transform = w.data.detach()
        # 计算权重数据的最小值，并扩展为1维张量
        w_min = w_transform.min().expand(1)
        # 计算权重数据的最大值，并扩展为1维张量
        w_max = w_transform.max().expand(1)

        # 计算权重的对称线性量化参数
        self.weight_scaling_factor = symmetric_linear_quantization_params(self.weight_bit, w_min, w_max, False)
        # 使用量化函数将浮点权重转换为整数权重
        self.weight_integer = self.weight_function(
            self.weight, self.weight_bit, self.percentile_mode, self.weight_scaling_factor
        )

        # 使用整数权重进行嵌入操作
        emb_int = nn.functional.embedding(
            x,
            self.weight_integer,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        # 返回量化后的嵌入结果乘以权重的缩放因子，以及权重的缩放因子本身
        return emb_int * self.weight_scaling_factor, self.weight_scaling_factor
class QuantAct(nn.Module):
    """
    Quantizes the given activation.

    Args:
        activation_bit (`int`):
            Bitwidth for the quantized activation.
        act_range_momentum (`float`, *optional*, defaults to `0.95`):
            Momentum for updating the activation quantization range.
        per_channel (`bool`, *optional*, defaults to `False`):
            Whether to or not use channel-wise quantization.
        channel_len (`int`, *optional*):
            Specify the channel length when set the *per_channel* True.
        quant_mode (`bool`, *optional`, defaults to `False`):
            Whether or not the layer is quantized.
    """

    def __init__(self, activation_bit, act_range_momentum=0.95, per_channel=False, channel_len=None, quant_mode=False):
        super().__init__()

        self.activation_bit = activation_bit  # 设置激活量化的位宽
        self.act_range_momentum = act_range_momentum  # 激活量化范围动量更新的动量
        self.quant_mode = quant_mode  # 层是否量化的标志
        self.per_channel = per_channel  # 是否进行通道-wise的量化
        self.percentile = False  # 百分位数是否激活的标志
        self.act_function = SymmetricQuantFunction.apply  # 使用的量化函数

        if not self.per_channel:
            # 如果不是每个通道独立量化，则注册缓冲区
            self.register_buffer("x_min", torch.zeros(1))
            self.register_buffer("x_max", torch.zeros(1))
            self.register_buffer("act_scaling_factor", torch.zeros(1))
            self.x_min -= 1e-5  # 调整最小值的初始化偏移
            self.x_max += 1e-5  # 调整最大值的初始化偏移
        else:
            # 目前不支持通道-wise模式的量化
            raise NotImplementedError("per-channel mode is not currently supported for activation.")

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(activation_bit={self.activation_bit}, "
            f"quant_mode: {self.quant_mode}, Act_min: {self.x_min.item():.2f}, "
            f"Act_max: {self.x_max.item():.2f})"
        )

    def forward(
        self,
        x,
        pre_act_scaling_factor=None,
        identity=None,
        identity_scaling_factor=None,
        specified_min=None,
        specified_max=None,
        ):
            # 根据标识(identity)是否为空来确定是否对输入进行偏移操作
            x_act = x if identity is None else identity + x
            # 如果处于训练模式，则收集运行时的统计信息
            if self.training:
                # 断言检查，确保激活量化模式下不支持百分位模式和按通道模式
                assert not self.percentile, "percentile mode is not currently supported for activation."
                assert not self.per_channel, "per-channel mode is not currently supported for activation."
                # 计算激活值张量的最小值和最大值
                x_min = x_act.data.min()
                x_max = x_act.data.max()

                # 断言检查，确保计算激活值的最小和最大时未检测到NaN值
                assert (
                    x_max.isnan().sum() == 0 and x_min.isnan().sum() == 0
                ), "NaN detected when computing min/max of the activation"

                # 初始化过程
                if self.x_min.min() > -1.1e-5 and self.x_max.max() < 1.1e-5:
                    # 更新活动范围的最小值和最大值
                    self.x_min = self.x_min + x_min
                    self.x_max = self.x_max + x_max

                # 指数移动平均 (EMA)
                # 使用动量以防止量化值在每次迭代中发生显著变化
                elif self.act_range_momentum == -1:
                    self.x_min = torch.min(self.x_min, x_min)
                    self.x_max = torch.max(self.x_max, x_max)
                else:
                    self.x_min = self.x_min * self.act_range_momentum + x_min * (1 - self.act_range_momentum)
                    self.x_max = self.x_max * self.act_range_momentum + x_max * (1 - self.act_range_momentum)

        if not self.quant_mode:
            # 如果不处于量化模式，则直接返回经过激活函数处理后的值和空的量化参数
            return x_act, None

        # 根据指定的最小值和最大值或者默认的活动范围来计算活动缩放因子
        x_min = self.x_min if specified_min is None else specified_min
        x_max = self.x_max if specified_max is None else specified_max

        # 计算对应的对称线性量化参数
        self.act_scaling_factor = symmetric_linear_quantization_params(
            self.activation_bit, x_min, x_max, per_channel=self.per_channel
        )

        if pre_act_scaling_factor is None:
            # 如果没有预先计算的激活值缩放因子，则进行输入的量化操作
            quant_act_int = self.act_function(x, self.activation_bit, self.percentile, self.act_scaling_factor)
        else:
            # 否则，使用固定点乘法进行量化操作
            quant_act_int = FixedPointMul.apply(
                x,
                pre_act_scaling_factor,
                self.activation_bit,
                self.act_scaling_factor,
                identity,
                identity_scaling_factor,
            )

        # 计算正确的输出缩放因子，用于量化后的激活值
        correct_output_scale = self.act_scaling_factor.view(-1)

        return quant_act_int * correct_output_scale, self.act_scaling_factor
# 定义一个自定义的量化线性层，继承自 `torch.nn.Module`
class QuantLinear(nn.Module):
    """
    Quantized version of `torch.nn.Linear`. Adds quantization-specific arguments on top of `torch.nn.Linear`.

    Args:
        weight_bit (`int`, *optional*, defaults to `8`):
            Bitwidth for the quantized weight.
        bias_bit (`int`, *optional*, defaults to `32`):
            Bitwidth for the quantized bias.
        per_channel (`bool`, *optional*, defaults to `False`):
            Whether or not to use channel-wise quantization.
        quant_mode (`bool`, *optional*, defaults to `False`):
            Whether or not the layer is quantized.
    """

    # 初始化函数，设置量化线性层的参数和缓冲区
    def __init__(
        self, in_features, out_features, bias=True, weight_bit=8, bias_bit=32, per_channel=False, quant_mode=False
    ):
        super().__init__()
        # 设置输入和输出特征数
        self.in_features = in_features
        self.out_features = out_features

        # 初始化权重参数，并注册缓冲区 weight_integer 用于量化后的权重存储
        self.weight = nn.Parameter(torch.zeros([out_features, in_features]))
        self.register_buffer("weight_integer", torch.zeros_like(self.weight))
        # 初始化缩放因子，对每个输出特征都有一个缩放因子
        self.register_buffer("fc_scaling_factor", torch.zeros(self.out_features))
        
        # 如果有偏置项，则初始化偏置参数，并注册缓冲区 bias_integer 用于量化后的偏置存储
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
            self.register_buffer("bias_integer", torch.zeros_like(self.bias))

        # 设置权重和偏置的位宽，量化模式，是否使用通道级量化等属性
        self.weight_bit = weight_bit
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.bias_bit = bias_bit
        self.quant_mode = quant_mode  # 设置量化模式
        self.percentile_mode = False  # 百分位模式，这里未启用
        self.weight_function = SymmetricQuantFunction.apply  # 设置权重量化函数

    # 返回对象的字符串表示，包含量化参数信息
    def __repr__(self):
        s = super().__repr__()
        s = f"({s} weight_bit={self.weight_bit}, quant_mode={self.quant_mode})"
        return s
    # 定义前向传播函数，接受输入 x 和可选的前一层激活量缩放因子 prev_act_scaling_factor
    def forward(self, x, prev_act_scaling_factor=None):
        # 如果不处于量化模式下，直接调用 PyTorch 的线性层函数进行前向传播
        if not self.quant_mode:
            return nn.functional.linear(x, weight=self.weight, bias=self.bias), None

        # 断言 prev_act_scaling_factor 是一个标量张量
        assert prev_act_scaling_factor is not None and prev_act_scaling_factor.shape == (1,), (
            "Input activation to the QuantLinear layer should be globally (non-channel-wise) quantized. "
            "Please add a QuantAct layer with `per_channel = True` before this QuantAct layer"
        )

        # 获取权重张量
        w = self.weight
        # 分离权重数据，并且不再追踪计算图
        w_transform = w.data.detach()
        
        # 如果按通道量化
        if self.per_channel:
            # 计算每个通道的最小值和最大值
            w_min, _ = torch.min(w_transform, dim=1, out=None)
            w_max, _ = torch.max(w_transform, dim=1, out=None)
        else:
            # 计算整个权重张量的最小值和最大值，并扩展为包含一个元素的张量
            w_min = w_transform.min().expand(1)
            w_max = w_transform.max().expand(1)

        # 计算量化参数，根据权重位数、最小值、最大值和是否按通道量化
        self.fc_scaling_factor = symmetric_linear_quantization_params(self.weight_bit, w_min, w_max, self.per_channel)
        # 计算量化后的整数权重
        self.weight_integer = self.weight_function(
            self.weight, self.weight_bit, self.percentile_mode, self.fc_scaling_factor
        )

        # 计算偏置项的缩放因子
        bias_scaling_factor = self.fc_scaling_factor * prev_act_scaling_factor

        # 如果存在偏置项
        if self.bias is not None:
            # 计算量化后的整数偏置项
            self.bias_integer = self.weight_function(self.bias, self.bias_bit, False, bias_scaling_factor)

        # 将 prev_act_scaling_factor 重塑为形状为 (1, -1) 的张量，并用它对输入 x 进行缩放
        prev_act_scaling_factor = prev_act_scaling_factor.view(1, -1)
        x_int = x / prev_act_scaling_factor

        # 使用量化后的整数权重和偏置项进行线性变换，并乘以偏置项的缩放因子
        return (
            nn.functional.linear(x_int, weight=self.weight_integer, bias=self.bias_integer) * bias_scaling_factor,
            bias_scaling_factor,
        )
class IntGELU(nn.Module):
    """
    Quantized version of `torch.nn.GELU`. Adds quantization-specific arguments on top of `torch.nn.GELU`.

    Args:
        quant_mode (`bool`, *optional*, defaults to `False`):
            Whether or not the layer is quantized.
        force_dequant (`str`, *optional*, defaults to `"none"`):
            Force dequantize the layer if either "gelu" or "nonlinear" is given.
    """

    def __init__(self, quant_mode=True, force_dequant="none"):
        super().__init__()
        self.quant_mode = quant_mode  # 初始化量化模式标志，默认为 True

        if force_dequant in ["nonlinear", "gelu"]:
            logger.info("Force dequantize gelu")
            self.quant_mode = False  # 如果 force_dequant 参数为 "nonlinear" 或 "gelu"，强制取消量化模式

        if not self.quant_mode:
            self.activation_fn = nn.GELU()  # 如果未使用量化模式，则使用 nn.GELU 激活函数

        self.k = 1.4142  # 常数 k，用于计算缩放因子
        self.const = 14  # 虚拟的整数常数
        self.coeff = [-0.2888, -1.769, 1]  # 系数数组 [a, b, c]，用于计算整数误差函数
        self.coeff[2] /= self.coeff[0]  # 系数归一化处理

    def int_erf(self, x_int, scaling_factor):
        b_int = torch.floor(self.coeff[1] / scaling_factor)  # 计算 b 的整数值
        c_int = torch.floor(self.coeff[2] / scaling_factor**2)  # 计算 c 的整数值
        sign = torch.sign(x_int)  # 计算 x_int 的符号

        abs_int = torch.min(torch.abs(x_int), -b_int)  # 取绝对值并截断到 -b_int
        y_int = sign * ((abs_int + b_int) ** 2 + c_int)  # 计算整数误差函数

        scaling_factor = scaling_factor**2 * self.coeff[0]  # 更新缩放因子的平方乘以系数 a

        # 避免溢出，通过右移操作
        y_int = floor_ste.apply(y_int / 2**self.const)  # 使用 floor_ste 函数进行右移处理
        scaling_factor = scaling_factor * 2**self.const  # 更新缩放因子

        return y_int, scaling_factor  # 返回整数误差函数值和更新后的缩放因子

    def forward(self, x, scaling_factor=None):
        if not self.quant_mode:
            return self.activation_fn(x), None  # 如果未使用量化模式，直接返回激活函数处理后的结果

        x_int = x / scaling_factor  # 计算 x 的整数值
        sigmoid_int, sigmoid_scaling_factor = self.int_erf(x_int, scaling_factor / self.k)  # 计算整数误差函数

        shift_int = 1.0 // sigmoid_scaling_factor  # 计算整数误差函数的偏移量

        x_int = x_int * (sigmoid_int + shift_int)  # 应用整数误差函数和偏移量对 x_int 进行处理
        scaling_factor = scaling_factor * sigmoid_scaling_factor / 2  # 更新缩放因子

        return x_int * scaling_factor, scaling_factor  # 返回处理后的整数值和更新后的缩放因子


class IntSoftmax(nn.Module):
    """
    Quantized version of `torch.nn.Softmax`. Adds quantization-specific arguments on top of `torch.nn.Softmax`.

    Args:
        output_bit (`int`):
            Bitwidth for the layer output activation.
        quant_mode (`bool`, *optional*, defaults to `False`):
            Whether or not the layer is quantized.
        force_dequant (`str`, *optional*, defaults to `"none"`):
            Force dequantize the layer if either "softmax" or "nonlinear" is given.
    """
    # 初始化函数，设置输出位数、量化模式和强制去量化模式
    def __init__(self, output_bit, quant_mode=False, force_dequant="none"):
        # 调用父类初始化函数
        super().__init__()
        # 设置输出位数
        self.output_bit = output_bit
        # 最大位数设为32
        self.max_bit = 32
        # 设置量化模式
        self.quant_mode = quant_mode

        # 如果强制去量化模式为"nonlinear"或"softmax"
        if force_dequant in ["nonlinear", "softmax"]:
            # 输出日志信息
            logger.info("Force dequantize softmax")
            # 强制取消量化模式设为False
            self.quant_mode = False

        # 初始化量化操作对象，16为输入量化位数
        self.act = QuantAct(16, quant_mode=self.quant_mode)
        # 设置常数x0为-ln2
        self.x0 = -0.6931  # -ln2
        # 设置常数const为30，用作虚拟整数常量
        self.const = 30  # dummy integer constant
        # 设置多项式系数为ax**2 + bx + c，其中a为1.0，b为0.35815147，c为0.96963238
        self.coef = [0.35815147, 0.96963238, 1.0]
        # 根据a对b和c进行归一化处理
        self.coef[1] /= self.coef[0]
        self.coef[2] /= self.coef[0]

    # 整型多项式函数
    def int_polynomial(self, x_int, scaling_factor):
        # 禁用梯度计算
        with torch.no_grad():
            # 计算系数b_int和c_int
            b_int = torch.floor(self.coef[1] / scaling_factor)
            c_int = torch.floor(self.coef[2] / scaling_factor**2)
        # 计算多项式结果z
        z = (x_int + b_int) * x_int + c_int
        # 更新缩放因子为多项式系数乘以原缩放因子的平方
        scaling_factor = self.coef[0] * scaling_factor**2
        return z, scaling_factor

    # 整型指数函数
    def int_exp(self, x_int, scaling_factor):
        # 禁用梯度计算
        with torch.no_grad():
            # 计算整数化的x0_int
            x0_int = torch.floor(self.x0 / scaling_factor)
        # 限制x_int的最小值为常数const乘以x0_int
        x_int = torch.max(x_int, self.const * x0_int)

        # 计算q和r
        q = floor_ste.apply(x_int / x0_int)
        r = x_int - x0_int * q
        # 计算指数整数和缩放因子
        exp_int, exp_scaling_factor = self.int_polynomial(r, scaling_factor)
        # 对指数整数进行修剪并缩放
        exp_int = torch.clamp(floor_ste.apply(exp_int * 2 ** (self.const - q)), min=0)
        scaling_factor = exp_scaling_factor / 2**self.const
        return exp_int, scaling_factor

    # 前向传播函数
    def forward(self, x, scaling_factor):
        # 如果非量化模式，直接返回softmax函数结果和空值
        if not self.quant_mode:
            return nn.functional.softmax(x, dim=-1), None

        # 计算整数化的输入x_int
        x_int = x / scaling_factor

        # 计算x_int的最大值和更新x_int
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max

        # 计算指数整数和指数缩放因子
        exp_int, exp_scaling_factor = self.int_exp(x_int, scaling_factor)

        # 避免溢出
        exp, exp_scaling_factor = self.act(exp_int, exp_scaling_factor)
        exp_int = exp / exp_scaling_factor

        # 计算指数整数的总和
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)
        # 计算因子
        factor = floor_ste.apply(2**self.max_bit / exp_int_sum)
        # 对指数整数进行修剪并缩放
        exp_int = floor_ste.apply(exp_int * factor / 2 ** (self.max_bit - self.output_bit))
        scaling_factor = 1 / 2**self.output_bit
        return exp_int * scaling_factor, scaling_factor
    """
    Quantized version of `torch.nn.LayerNorm`. Adds quantization-specific arguments on top of `torch.nn.LayerNorm`.

    Args:
        normalized_shape (`int` or `list` or `torch.Size`):
            Shape of the input tensor over which normalization is applied.
        eps (`float`):
            Small value added to the denominator for numerical stability.
        output_bit (`int`, *optional*, defaults to `8`):
            Bitwidth for the layer output activation.
        quant_mode (`bool`, *optional*, defaults to `False`):
            Whether or not the layer is quantized.
        force_dequant (`str`, *optional*, defaults to `"none"`):
            If set to `"layernorm"` or `"nonlinear"`, forces dequantization of the layer.

    Attributes:
        weight (`torch.nn.Parameter`):
            Learnable parameter representing the scaling factor.
        bias (`torch.nn.Parameter`):
            Learnable parameter representing the bias.
        shift (`torch.Tensor`):
            Buffer holding the shift value for dynamic adjustment.
        output_bit (`int`):
            Bitwidth for the layer output activation.
        max_bit (`int`):
            Maximum allowable bitwidth for quantization.
        dim_sqrt (`None`):
            Placeholder for the square root of the dimension, initially `None`.
        activation (`QuantAct`):
            Instance of `QuantAct` for quantization-aware activation.

    Methods:
        set_shift(self, y_int):
            Adjusts `self.shift` based on the input tensor `y_int`.
        overflow_fallback(self, y_int):
            Handles overflow during training and adjusts `self.shift` accordingly.

    Notes:
        - This class extends `torch.nn.Module` and integrates quantization-specific features.
        - It manages parameters for scaling and bias, quantization mode, and dynamic shift adjustments.
        - The `QuantAct` instance `activation` handles activation quantization within the layer.
    """

    def __init__(self, normalized_shape, eps, output_bit=8, quant_mode=False, force_dequant="none"):
        super().__init__()
        # Initialize attributes related to normalization
        self.normalized_shape = normalized_shape
        self.eps = eps

        # Initialize learnable parameters
        self.weight = nn.Parameter(torch.zeros(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

        # Manage quantization mode, with option for forced dequantization
        self.quant_mode = quant_mode
        if force_dequant in ["nonlinear", "layernorm"]:
            logger.info("Force dequantize layernorm")
            self.quant_mode = False

        # Buffer for dynamic shift adjustment
        self.register_buffer("shift", torch.zeros(1))
        
        # Configure output bitwidth and related parameters
        self.output_bit = output_bit
        self.max_bit = 32
        self.dim_sqrt = None
        
        # Quantized activation function
        self.activation = QuantAct(self.output_bit, quant_mode=self.quant_mode)

    def set_shift(self, y_int):
        """
        Adjusts `self.shift` based on the input tensor `y_int`.

        Args:
            y_int (`torch.Tensor`):
                Integer tensor representing the quantized activation values.
        """
        with torch.no_grad():
            y_sq_int = y_int**2
            var_int = torch.sum(y_sq_int, axis=2, keepdim=True)
            shift = (torch.log2(torch.sqrt(var_int / 2**self.max_bit)).ceil()).max()
            shift_old = self.shift
            self.shift = torch.max(self.shift, shift)
            logger.info(f"Dynamic shift adjustment: {int(shift_old)} -> {int(self.shift)}")

    def overflow_fallback(self, y_int):
        """
        Handles overflow during training and adjusts `self.shift` accordingly.

        Args:
            y_int (`torch.Tensor`):
                Integer tensor representing the quantized activation values.

        Returns:
            `torch.Tensor`: Tensor representing the adjusted variance after shift.
        """
        self.set_shift(y_int)  # adjusts `self.shift`
        y_int_shifted = floor_ste.apply(y_int / 2**self.shift)
        y_sq_int = y_int_shifted**2
        var_int = torch.sum(y_sq_int, axis=2, keepdim=True)
        return var_int
    # 定义前向传播函数，接受输入张量 x 和可选的缩放因子 scaling_factor
    def forward(self, x, scaling_factor=None):
        # 如果不是量化模式
        if not self.quant_mode:
            # 计算输入张量 x 沿第二个轴的均值
            mean = x.mean(axis=2, keepdim=True)
            # 对输入张量进行均值中心化
            y = x - mean
            # 计算中心化后的输入张量的方差
            var = torch.mean(y**2, axis=2, keepdim=True)
            # 根据均值和方差进行标准化处理
            x = y / torch.sqrt(self.eps + var)
            # 对标准化后的张量进行加权和偏移处理
            x = x * self.weight + self.bias
            # 返回处理后的张量和空的 scaling_factor
            return x, None

        # 如果是量化模式，并且还未计算过 feature 维度的平方根
        if self.dim_sqrt is None:
            # 计算 feature 维度的平方根并保存到 self.dim_sqrt 中
            n = torch.tensor(x.shape[2], dtype=torch.float)
            self.dim_sqrt = torch.sqrt(n).to(x.device)

        # 对输入张量 x 进行除以缩放因子的量化
        x_int = x / scaling_factor
        # 计算量化后的输入张量沿第二个轴的均值并四舍五入
        mean_int = round_ste.apply(x_int.mean(axis=2, keepdim=True))
        # 对量化后的输入张量进行均值中心化
        y_int = x_int - mean_int
        # 将中心化后的量化张量按照指定的位移因子进行向下取整操作
        y_int_shifted = floor_ste.apply(y_int / 2**self.shift)
        # 计算量化后的输入张量的平方
        y_sq_int = y_int_shifted**2
        # 计算量化后的输入张量的方差
        var_int = torch.sum(y_sq_int, axis=2, keepdim=True)

        # 如果处于训练阶段，并且检测到方差 var_int 存在溢出
        if self.training:
            # 如果方差 var_int 的最大值超过了 self.max_bit 所指定的阈值
            if var_int.max() >= 2**self.max_bit:
                # 执行溢出处理函数以获取修正后的方差 var_int
                var_int = self.overflow_fallback(y_int)
                # 断言确保修正后的方差 var_int 仍然小于 self.max_bit + 0.1
                assert var_int.max() < 2**self.max_bit + 0.1, (
                    "Error detected in overflow handling: "
                    "`var_int` exceeds `self.max_bit` (the maximum possible bit width)"
                )

        # 待替换为生成相同输出的整数平方根核函数
        std_int = floor_ste.apply(torch.sqrt(var_int)) * 2**self.shift
        # 计算因子，用于缩放输入张量 y_int
        factor = floor_ste.apply(2**31 / std_int)
        # 根据计算得到的因子对输入张量 y_int 进行进一步处理
        y_int = floor_ste.apply(y_int * factor / 2)
        # 计算缩放因子 scaling_factor，用于最终的缩放和偏移
        scaling_factor = self.dim_sqrt / 2**30

        # 缩放和偏移处理
        bias = self.bias.data.detach() / (self.weight.data.detach())
        bias_int = floor_ste.apply(bias / scaling_factor)

        y_int = y_int + bias_int
        scaling_factor = scaling_factor * self.weight
        x = y_int * scaling_factor

        # 返回处理后的张量 x 和最终的 scaling_factor
        return x, scaling_factor
# 计算给定张量中百分位数的最大值和最小值
def get_percentile_min_max(input, lower_percentile, upper_percentile, output_tensor=False):
    """
    Calculate the percentile max and min values in a given tensor

    Args:
        input (`torch.Tensor`):
            The target tensor to calculate percentile max and min.
        lower_percentile (`float`):
            If 0.1, means we return the value of the smallest 0.1% value in the tensor as percentile min.
        upper_percentile (`float`):
            If 99.9, means we return the value of the largest 0.1% value in the tensor as percentile max.
        output_tensor (`bool`, *optional*, defaults to `False`):
            If True, this function returns tensors, otherwise it returns values.

    Returns:
        `Tuple(torch.Tensor, torch.Tensor)`: Percentile min and max value of *input*
    """
    # 获取输入张量的长度
    input_length = input.shape[0]

    # 计算下分位数和上分位数的索引
    lower_index = round(input_length * (1 - lower_percentile * 0.01))
    upper_index = round(input_length * upper_percentile * 0.01)

    # 计算上分位数的值
    upper_bound = torch.kthvalue(input, k=upper_index).values

    # 如果 lower_percentile 为 0，则下分位数设为 0，否则计算下分位数的值
    if lower_percentile == 0:
        lower_bound = upper_bound * 0
        # lower_index += 1
    else:
        lower_bound = -torch.kthvalue(-input, k=lower_index).values

    # 如果不需要输出张量，将下分位数和上分位数转换为标量值
    if not output_tensor:
        lower_bound = lower_bound.item()
        upper_bound = upper_bound.item()
    return lower_bound, upper_bound


def linear_quantize(input, scale, zero_point, inplace=False):
    """
    Quantize single-precision input tensor to integers with the given scaling factor and zeropoint.

    Args:
        input (`torch.Tensor`):
            Single-precision input tensor to be quantized.
        scale (`torch.Tensor`):
            Scaling factor for quantization.
        zero_point (`torch.Tensor`):
            Shift for quantization.
        inplace (`bool`, *optional*, defaults to `False`):
            Whether to compute inplace or not.

    Returns:
        `torch.Tensor`: Linearly quantized value of *input* according to *scale* and *zero_point*.
    """
    # 根据张量维度重新调整 scale 和 zero_point，适用于卷积权重和激活函数
    if len(input.shape) == 4:
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    # 根据张量维度重新调整 scale 和 zero_point，适用于线性权重
    elif len(input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    else:
        scale = scale.view(-1)
        zero_point = zero_point.view(-1)
    # 执行量化操作：input = float / scale + zero_point
    if inplace:
        input.mul_(1.0 / scale).add_(zero_point).round_()
        return input
    return torch.round(1.0 / scale * input + zero_point)


def symmetric_linear_quantization_params(num_bits, saturation_min, saturation_max, per_channel=False):
    """
    Compute the scaling factor with the given quantization range for symmetric quantization.
    """
    # 在对称量化情况下计算缩放因子，根据给定的量化范围
    # 在这部分，我们不需要进行任何梯度计算，
    # 为了确保这一点，我们使用 torch.no_grad() 来包裹代码块

    with torch.no_grad():
        # 计算量化的范围，使用的比特数为 num_bits
        n = 2 ** (num_bits - 1) - 1

        # 如果 per_channel 为 True，执行以下操作
        if per_channel:
            # 计算每个通道的最大饱和度，并取绝对值
            scale, _ = torch.max(torch.stack([saturation_min.abs(), saturation_max.abs()], dim=1), dim=1)
            # 将 scale 限制在最小值为 1e-8，然后进行量化范围的计算
            scale = torch.clamp(scale, min=1e-8) / n

        else:
            # 计算整体的最大饱和度，并取绝对值
            scale = max(saturation_min.abs(), saturation_max.abs())
            # 将 scale 限制在最小值为 1e-8，然后进行量化范围的计算
            scale = torch.clamp(scale, min=1e-8) / n

    # 返回计算得到的量化因子 scale
    return scale
class SymmetricQuantFunction(Function):
    """
    Class to quantize the given floating-point values using symmetric quantization with given range and bitwidth.
    """

    @staticmethod
    def forward(ctx, x, k, percentile_mode, scale):
        """
        Args:
            x (`torch.Tensor`):
                Floating point tensor to be quantized.
            k (`int`):
                Quantization bitwidth.
            percentile_mode (`bool`):
                Whether or not to use percentile calibration.
            scale (`torch.Tensor`):
                Pre-calculated scaling factor for *x*. Note that the current implementation of SymmetricQuantFunction
                requires pre-calculated scaling factor.

        Returns:
            `torch.Tensor`: Symmetric-quantized value of *input*.
        """
        # Define the zero point as a tensor with value 0.0 on the same device as scale
        zero_point = torch.tensor(0.0).to(scale.device)

        # Calculate the maximum representable integer for given bitwidth k
        n = 2 ** (k - 1) - 1
        
        # Perform linear quantization with the given parameters
        new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)
        
        # Clamp the quantized values to ensure they lie within the representable range
        new_quant_x = torch.clamp(new_quant_x, -n, n - 1)

        # Store scaling factor in context for backward pass
        ctx.scale = scale
        
        return new_quant_x

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve stored scaling factor from context
        scale = ctx.scale
        
        # Adjust scale shape based on gradient output dimensions
        if len(grad_output.shape) == 4:
            scale = scale.view(-1, 1, 1, 1)
        elif len(grad_output.shape) == 2:
            scale = scale.view(-1, 1)
        else:
            scale = scale.view(-1)
        
        # Return gradient scaled by the inverse of the scaling factor, and None for other arguments
        return grad_output.clone() / scale, None, None, None, None


class floor_ste(Function):
    """
    Straight-through Estimator(STE) for torch.floor()
    """

    @staticmethod
    def forward(ctx, x):
        # Forward pass computes the floor of input tensor x
        return torch.floor(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass returns the gradient unchanged
        return grad_output.clone()


class round_ste(Function):
    """
    Straight-through Estimator(STE) for torch.round()
    """

    @staticmethod
    def forward(ctx, x):
        # Forward pass computes the round of input tensor x
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass returns the gradient unchanged
        return grad_output.clone()


def batch_frexp(inputs, max_bit=31):
    """
    Decompose the scaling factor into mantissa and twos exponent.

    Args:
        scaling_factor (`torch.Tensor`):
            Target scaling factor to decompose.

    Returns:
        ``Tuple(torch.Tensor, torch.Tensor)`: mantisa and exponent
    """

    # Get the shape of the input tensor
    shape_of_input = inputs.size()

    # Flatten the input tensor to 1D
    inputs = inputs.view(-1)

    # Use NumPy's frexp function to decompose each element of the tensor into mantissa and exponent
    output_m, output_e = np.frexp(inputs.cpu().numpy())
    
    # Quantize the mantissa and shift it to fit within max_bit range
    tmp_m = []
    for m in output_m:
        int_m_shifted = int(
            decimal.Decimal(m * (2**max_bit)).quantize(decimal.Decimal("1"), rounding=decimal.ROUND_HALF_UP)
        )
        tmp_m.append(int_m_shifted)
    output_m = np.array(tmp_m)

    # Calculate the exponent in terms of max_bit
    output_e = float(max_bit) - output_e

    # Return the quantized mantissa and exponent tensors reshaped to the original input shape
    return (
        torch.from_numpy(output_m).to(inputs.device).view(shape_of_input),
        torch.from_numpy(output_e).to(inputs.device).view(shape_of_input),
    )
class FixedPointMul(Function):
    """
    Function to perform fixed-point arithmetic that can match integer arithmetic on hardware.

    Args:
        pre_act (`torch.Tensor`):
            Input tensor.
        pre_act_scaling_factor (`torch.Tensor`):
            Scaling factor of the input tensor *pre_act*.
        bit_num (`int`):
            Quantization bitwidth.
        z_scaling_factor (`torch.Tensor`):
            Scaling factor of the output tensor.
        identity (`torch.Tensor`, *optional*):
            Identity tensor, if exists.
        identity_scaling_factor (`torch.Tensor`, *optional*):
            Scaling factor of the identity tensor *identity*, if exists.

    Returns:
        `torch.Tensor`: Output tensor(*pre_act* if *identity* is not given, otherwise the addition of *pre_act* and
        *identity*), whose scale is rescaled to *z_scaling_factor*.
    """

    @staticmethod
    def forward(
        ctx,
        pre_act,
        pre_act_scaling_factor,
        bit_num,
        z_scaling_factor,
        identity=None,
        identity_scaling_factor=None,
    ):
        # Lambda function to reshape input tensor if necessary
        if len(pre_act_scaling_factor.shape) == 3:
            reshape = lambda x: x  # noqa: E731
        else:
            reshape = lambda x: x.view(1, 1, -1)  # noqa: E731
        
        # Store identity tensor in the context
        ctx.identity = identity

        # Maximum representable integer in fixed-point representation
        n = 2 ** (bit_num - 1) - 1

        # Perform operations with gradients turned off
        with torch.no_grad():
            # Reshape scaling factors
            pre_act_scaling_factor = reshape(pre_act_scaling_factor)
            if identity is not None:
                identity_scaling_factor = reshape(identity_scaling_factor)

            # Store scaling factor of the output tensor in the context
            ctx.z_scaling_factor = z_scaling_factor

            # Quantize input tensor pre_act
            z_int = torch.round(pre_act / pre_act_scaling_factor)
            _A = pre_act_scaling_factor.type(torch.double)
            _B = (z_scaling_factor.type(torch.float)).type(torch.double)
            new_scale = _A / _B
            new_scale = reshape(new_scale)

            # Compute mantissa and exponent using batch_frexp function
            m, e = batch_frexp(new_scale)

            # Compute the output tensor in fixed-point arithmetic
            output = z_int.type(torch.double) * m.type(torch.double)
            output = torch.round(output / (2.0**e))

            # If identity tensor is provided, perform additional fixed-point arithmetic
            if identity is not None:
                wx_int = torch.round(identity / identity_scaling_factor)

                _A = identity_scaling_factor.type(torch.double)
                _B = (z_scaling_factor.type(torch.float)).type(torch.double)
                new_scale = _A / _B
                new_scale = reshape(new_scale)

                m1, e1 = batch_frexp(new_scale)
                output1 = wx_int.type(torch.double) * m1.type(torch.double)
                output1 = torch.round(output1 / (2.0**e1))

                # Sum the outputs of pre_act and identity tensors
                output = output1 + output

            # Clamp the output tensor within the range of representable integers
            return torch.clamp(output.type(torch.float), -n - 1, n)

    @staticmethod
    # 定义反向传播函数，计算梯度
    def backward(ctx, grad_output):
        # 初始化变量用于存储身份梯度
        identity_grad = None
        # 如果上下文中的身份不为None，则计算身份梯度
        if ctx.identity is not None:
            # 克隆梯度输出并除以上下文中的缩放因子，作为身份梯度
            identity_grad = grad_output.clone() / ctx.z_scaling_factor
        # 返回计算得到的梯度，其他返回值为None
        return grad_output.clone() / ctx.z_scaling_factor, None, None, None, None, identity_grad, None
```