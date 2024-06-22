# `.\models\ibert\quant_modules.py`

```py
# 设置文件编码为utf-8
# 版权声明，包括作者信息和版权信息
# 根据Apache License 2.0许可证，对代码进行许可
# 获取许可证的链接
# 根据适用法律或书面同意，根据许可证分发软件
# 根据许可证，分发的软件基于"AS IS"基础，没有任何明示或暗示的保证或条件
# 查看许可证以了解具体语言和限制

# 导入所需的库
import decimal
import numpy as np
import torch
from torch import nn
from torch.autograd import Function
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义一个量化版本的torch.nn.Embedding类，添加了量化特定参数
class QuantEmbedding(nn.Module):
    """
    Quantized version of `torch.nn.Embedding`. Adds quantization-specific arguments on top of `torch.nn.Embedding`.

    Args:
        weight_bit (`int`, *optional*, defaults to `8`):
            Bitwidth for the quantized weight.
        momentum (`float`, *optional*, defaults to `0.95`):
            Momentum for updating the activation quantization range.
        quant_mode (`bool`, *optional*, defaults to `False`):
            Whether or not the layer is quantized.
    """

    # 初始化函数，设置各种参数
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
        # 初始化各种参数
        self.num_ = num_embeddings
        self.dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        # 初始化权重参数
        self.weight = nn.Parameter(torch.zeros([num_embeddings, embedding_dim]))
        self.register_buffer("weight_scaling_factor", torch.zeros(1))
        self.register_buffer("weight_integer", torch.zeros_like(self.weight))

        # 设置量化相关参数
        self.weight_bit = weight_bit
        self.momentum = momentum
        self.quant_mode = quant_mode
        self.percentile_mode = False
        self.weight_function = SymmetricQuantFunction.apply
    # 定义前向传播函数，接受输入 x，位置 positions 和增量状态 incremental_state
    def forward(self, x, positions=None, incremental_state=None):
        # 如果不处于量化模式，则直接返回通过 nn.functional.embedding 函数计算的结果和空值
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

        # 获取权重参数 w
        w = self.weight
        # 分离权重参数的数据部分
        w_transform = w.data.detach()
        # 计算权重参数的最小值，并扩展为 1 维张量
        w_min = w_transform.min().expand(1)
        # 计算权重参数的最大值，并扩展为 1 维张量
        w_max = w_transform.max().expand(1)

        # 计算权重参数的对称线性量化因子
        self.weight_scaling_factor = symmetric_linear_quantization_params(self.weight_bit, w_min, w_max, False)
        # 使用权重函数对权重参数进行量化
        self.weight_integer = self.weight_function(
            self.weight, self.weight_bit, self.percentile_mode, self.weight_scaling_factor
        )

        # 使用量化后的整数权重参数进行嵌入操作
        emb_int = nn.functional.embedding(
            x,
            self.weight_integer,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        # 返回量化后的嵌入结果乘以权重缩放因子和权重缩放因子本身
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
        quant_mode (`bool`, *optional*, defaults to `False`):
            Whether or not the layer is quantized.
    """

    def __init__(self, activation_bit, act_range_momentum=0.95, per_channel=False, channel_len=None, quant_mode=False):
        # 初始化函数，设置激活位宽、激活范围动量、是否使用通道级量化、通道长度和量化模式
        super().__init__()

        self.activation_bit = activation_bit
        self.act_range_momentum = act_range_momentum
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.percentile = False
        self.act_function = SymmetricQuantFunction.apply

        if not self.per_channel:
            # 如果不是通道级量化，则初始化激活最小值、最大值和缩放因子
            self.register_buffer("x_min", torch.zeros(1))
            self.register_buffer("x_max", torch.zeros(1))
            self.register_buffer("act_scaling_factor", torch.zeros(1))
            self.x_min -= 1e-5
            self.x_max += 1e-5
        else:
            # 如果是通道级量化，则抛出未实现的错误
            raise NotImplementedError("per-channel mode is not currently supported for activation.")

    def __repr__(self):
        # 返回对象的字符串表示，包括激活位宽、量化模式、激活最小值和最大值
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
        # 如果没有指定身份，则将输入值赋给 x_act
        x_act = x if identity is None else identity + x
        # 如果处于训练状态，则收集运行统计信息
        if self.training:
            # 断言不支持百分位模式和每通道模式
            assert not self.percentile, "percentile mode is not currently supported for activation."
            assert not self.per_channel, "per-channel mode is not currently supported for activation."
            # 计算激活的最小值和最大值
            x_min = x_act.data.min()
            x_max = x_act.data.max()

            # 断言激活的最小值和最大值不包含 NaN
            assert (
                x_max.isnan().sum() == 0 and x_min.isnan().sum() == 0
            ), "NaN detected when computing min/max of the activation"

            # 初始化
            if self.x_min.min() > -1.1e-5 and self.x_max.max() < 1.1e-5:
                self.x_min = self.x_min + x_min
                self.x_max = self.x_max + x_max

            # 指数移动平均（EMA）
            # 使用动量防止量化值在每次迭代中发生巨大变化
            elif self.act_range_momentum == -1:
                self.x_min = torch.min(self.x_min, x_min)
                self.x_max = torch.max(self.x_max, x_max)
            else:
                self.x_min = self.x_min * self.act_range_momentum + x_min * (1 - self.act_range_momentum)
                self.x_max = self.x_max * self.act_range_momentum + x_max * (1 - self.act_range_momentum)

        # 如果不是量化模式，则返回 x_act 和空值
        if not self.quant_mode:
            return x_act, None

        # 如果没有指定最小值，则将 self.x_min 赋给 x_min
        x_min = self.x_min if specified_min is None else specified_min
        # 如果没有指定最大值，则将 self.x_max 赋给 x_max
        x_max = self.x_max if specified_max is None else specified_max

        # 计算激活的缩放因子
        self.act_scaling_factor = symmetric_linear_quantization_params(
            self.activation_bit, x_min, x_max, per_channel=self.per_channel
        )

        if pre_act_scaling_factor is None:
            # 这是用于输入量化
            quant_act_int = self.act_function(x, self.activation_bit, self.percentile, self.act_scaling_factor)
        else:
            quant_act_int = FixedPointMul.apply(
                x,
                pre_act_scaling_factor,
                self.activation_bit,
                self.act_scaling_factor,
                identity,
                identity_scaling_factor,
            )

        # 获取正确的输出比例
        correct_output_scale = self.act_scaling_factor.view(-1)

        return quant_act_int * correct_output_scale, self.act_scaling_factor
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

    def __init__(
        self, in_features, out_features, bias=True, weight_bit=8, bias_bit=32, per_channel=False, quant_mode=False
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 设置输入特征和输出特征
        self.in_features = in_features
        self.out_features = out_features

        # 初始化权重参数，并将其注册为模型的缓冲区
        self.weight = nn.Parameter(torch.zeros([out_features, in_features]))
        self.register_buffer("weight_integer", torch.zeros_like(self.weight))
        self.register_buffer("fc_scaling_factor", torch.zeros(self.out_features))
        # 如果有偏置，则初始化偏置参数，并将其注册为模型的缓冲区
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
            self.register_buffer("bias_integer", torch.zeros_like(self.bias))

        # 设置权重位宽、偏置位宽、是否使用通道级量化、是否启用量化模式等参数
        self.weight_bit = weight_bit
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.bias_bit = bias_bit
        self.quant_mode = quant_mode
        self.percentile_mode = False
        # 设置权重量化函数
        self.weight_function = SymmetricQuantFunction.apply

    def __repr__(self):
        # 调用父类的 __repr__ 方法，并添加额外的信息
        s = super().__repr__()
        s = f"({s} weight_bit={self.weight_bit}, quant_mode={self.quant_mode})"
        return s
    # 定义前向传播函数，接受输入 x 和前一层激活的缩放因子 prev_act_scaling_factor
    def forward(self, x, prev_act_scaling_factor=None):
        # 如果不处于量化模式，则直接返回线性变换结果和空的缩放因子
        if not self.quant_mode:
            return nn.functional.linear(x, weight=self.weight, bias=self.bias), None

        # 断言 prev_act_scaling_factor 是一个标量张量
        assert prev_act_scaling_factor is not None and prev_act_scaling_factor.shape == (1,), (
            "Input activation to the QuantLinear layer should be globally (non-channel-wise) quantized. "
            "Please add a QuantAct layer with `per_channel = True` before this QuantAct layer"
        )

        # 获取权重参数
        w = self.weight
        w_transform = w.data.detach()
        # 如果是按通道量化，则计算每个通道的最小值和最大值
        if self.per_channel:
            w_min, _ = torch.min(w_transform, dim=1, out=None)
            w_max, _ = torch.max(w_transform, dim=1, out=None)
        else:
            # 否则，计算全局最小值和最大值
            w_min = w_transform.min().expand(1)
            w_max = w_transform.max().expand(1)

        # 计算权重的对称线性量化参数
        self.fc_scaling_factor = symmetric_linear_quantization_params(self.weight_bit, w_min, w_max, self.per_channel)
        # 对权重进行整数化
        self.weight_integer = self.weight_function(
            self.weight, self.weight_bit, self.percentile_mode, self.fc_scaling_factor
        )

        # 计算偏置的缩放因子
        bias_scaling_factor = self.fc_scaling_factor * prev_act_scaling_factor

        # 如果存在偏置，则对偏置进行整数化
        if self.bias is not None:
            self.bias_integer = self.weight_function(self.bias, self.bias_bit, False, bias_scaling_factor)

        # 将 prev_act_scaling_factor 转换为形状为 (1, -1)
        prev_act_scaling_factor = prev_act_scaling_factor.view(1, -1)
        # 对输入 x 进行缩放
        x_int = x / prev_act_scaling_factor

        # 返回经过整数化的线性变换结果和偏置的缩放因子
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
        # 初始化函数，设置 quant_mode 和 force_dequant 属性
        super().__init__()
        self.quant_mode = quant_mode

        # 如果 force_dequant 是 "nonlinear" 或 "gelu"，则强制取消量化模式
        if force_dequant in ["nonlinear", "gelu"]:
            logger.info("Force dequantize gelu")
            self.quant_mode = False

        # 如果不是量化模式，则使用 nn.GELU() 作为激活函数
        if not self.quant_mode:
            self.activation_fn = nn.GELU()

        # 设置常数和系数
        self.k = 1.4142
        self.const = 14  # dummy integer constant
        self.coeff = [-0.2888, -1.769, 1]  # a(x+b)**2 + c
        self.coeff[2] /= self.coeff[0]

    def int_erf(self, x_int, scaling_factor):
        # 计算误差函数的整数版本
        b_int = torch.floor(self.coeff[1] / scaling_factor)
        c_int = torch.floor(self.coeff[2] / scaling_factor**2)
        sign = torch.sign(x_int)

        abs_int = torch.min(torch.abs(x_int), -b_int)
        y_int = sign * ((abs_int + b_int) ** 2 + c_int)
        scaling_factor = scaling_factor**2 * self.coeff[0]

        # 避免溢出
        y_int = floor_ste.apply(y_int / 2**self.const)
        scaling_factor = scaling_factor * 2**self.const

        return y_int, scaling_factor

    def forward(self, x, scaling_factor=None):
        # 前向传播函数
        if not self.quant_mode:
            return self.activation_fn(x), None

        x_int = x / scaling_factor
        sigmoid_int, sigmoid_scaling_factor = self.int_erf(x_int, scaling_factor / self.k)

        shift_int = 1.0 // sigmoid_scaling_factor

        x_int = x_int * (sigmoid_int + shift_int)
        scaling_factor = scaling_factor * sigmoid_scaling_factor / 2

        return x_int * scaling_factor, scaling_factor


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
    # 初始化函数，设置输出位数、量化模式和强制去量化方式
    def __init__(self, output_bit, quant_mode=False, force_dequant="none"):
        # 调用父类初始化函数
        super().__init__()
        # 设置输出位数
        self.output_bit = output_bit
        # 最大位数为32
        self.max_bit = 32
        # 设置量化模式
        self.quant_mode = quant_mode

        # 如果强制去量化方式为非线性或softmax，则关闭量化模式
        if force_dequant in ["nonlinear", "softmax"]:
            logger.info("Force dequantize softmax")
            self.quant_mode = False

        # 初始化量化激活函数
        self.act = QuantAct(16, quant_mode=self.quant_mode)
        # 初始化常数
        self.x0 = -0.6931  # -ln2
        self.const = 30  # 虚拟整数常数
        # 初始化多项式系数
        self.coef = [0.35815147, 0.96963238, 1.0]  # ax**2 + bx + c
        self.coef[1] /= self.coef[0]
        self.coef[2] /= self.coef[0]

    # 整数多项式计算函数
    def int_polynomial(self, x_int, scaling_factor):
        with torch.no_grad():
            b_int = torch.floor(self.coef[1] / scaling_factor)
            c_int = torch.floor(self.coef[2] / scaling_factor**2)
        z = (x_int + b_int) * x_int + c_int
        scaling_factor = self.coef[0] * scaling_factor**2
        return z, scaling_factor

    # 整数指数计算函数
    def int_exp(self, x_int, scaling_factor):
        with torch.no_grad():
            x0_int = torch.floor(self.x0 / scaling_factor)
        x_int = torch.max(x_int, self.const * x0_int)

        q = floor_ste.apply(x_int / x0_int)
        r = x_int - x0_int * q
        exp_int, exp_scaling_factor = self.int_polynomial(r, scaling_factor)
        exp_int = torch.clamp(floor_ste.apply(exp_int * 2 ** (self.const - q)), min=0)
        scaling_factor = exp_scaling_factor / 2**self.const
        return exp_int, scaling_factor

    # 前向传播函数
    def forward(self, x, scaling_factor):
        # 如果不是量化模式，则直接返回softmax结果
        if not self.quant_mode:
            return nn.functional.softmax(x, dim=-1), None

        # 将输入除以缩放因子得到整数输入
        x_int = x / scaling_factor

        # 计算整数输入的最大值
        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max
        exp_int, exp_scaling_factor = self.int_exp(x_int, scaling_factor)

        # 避免溢出
        exp, exp_scaling_factor = self.act(exp_int, exp_scaling_factor)
        exp_int = exp / exp_scaling_factor

        # 计算整数指数的和
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)
        factor = floor_ste.apply(2**self.max_bit / exp_int_sum)
        exp_int = floor_ste.apply(exp_int * factor / 2 ** (self.max_bit - self.output_bit))
        scaling_factor = 1 / 2**self.output_bit
        return exp_int * scaling_factor, scaling_factor
class IntLayerNorm(nn.Module):
    """
    Quantized version of `torch.nn.LayerNorm`. Adds quantization-specific arguments on top of `torch.nn.LayerNorm`.

    Args:
        output_bit (`int`, *optional*, defaults to `8`):
            Bitwidth for the layer output activation.
        quant_mode (`bool`, *optional*, defaults to `False`):
            Whether or not the layer is quantized.
        force_dequant (`str`, *optional*, defaults to `"none"`):
            Force dequantize the layer if either "layernorm" or "nonlinear" is given.
    """

    # 初始化函数，设置参数和变量
    def __init__(self, normalized_shape, eps, output_bit=8, quant_mode=False, force_dequant="none"):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        # 初始化权重和偏置参数
        self.weight = nn.Parameter(torch.zeros(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

        self.quant_mode = quant_mode
        # 如果 force_dequant 参数为 "nonlinear" 或 "layernorm"，则强制取消量化模式
        if force_dequant in ["nonlinear", "layernorm"]:
            logger.info("Force dequantize layernorm")
            self.quant_mode = False

        # 注册缓冲区变量 shift，并初始化其他变量
        self.register_buffer("shift", torch.zeros(1))
        self.output_bit = output_bit
        self.max_bit = 32
        self.dim_sqrt = None
        self.activation = QuantAct(self.output_bit, quant_mode=self.quant_mode)

    # 设置 shift 变量的值
    def set_shift(self, y_int):
        with torch.no_grad():
            y_sq_int = y_int**2
            var_int = torch.sum(y_sq_int, axis=2, keepdim=True)
            shift = (torch.log2(torch.sqrt(var_int / 2**self.max_bit)).ceil()).max()
            shift_old = self.shift
            self.shift = torch.max(self.shift, shift)
            logger.info(f"Dynamic shift adjustment: {int(shift_old)} -> {int(self.shift)}")

    # 处理溢出情况的回退函数
    def overflow_fallback(self, y_int):
        """
        This fallback function is called when overflow is detected during training time, and adjusts the `self.shift`
        to avoid overflow in the subsequent runs.
        """
        self.set_shift(y_int)  # 调整 self.shift 的值
        y_int_shifted = floor_ste.apply(y_int / 2**self.shift)
        y_sq_int = y_int_shifted**2
        var_int = torch.sum(y_sq_int, axis=2, keepdim=True)
        return var_int
    # 定义前向传播函数，接受输入 x 和缩放因子 scaling_factor
    def forward(self, x, scaling_factor=None):
        # 如果不处于量化模式
        if not self.quant_mode:
            # 计算输入 x 沿第二个维度的均值
            mean = x.mean(axis=2, keepdim=True)
            # 对 x 减去均值
            y = x - mean
            # 计算 y 的平方和的均值，即方差
            var = torch.mean(y**2, axis=2, keepdim=True)
            # 对 y 进行标准化
            x = y / torch.sqrt(self.eps + var)
            # 对标准化后的 x 进行权重和偏置的线性变换
            x = x * self.weight + self.bias
            # 返回处理后的 x 和空的 scaling_factor
            return x, None

        # 如果是第一次运行，计算特征维度的平方根
        if self.dim_sqrt is None:
            n = torch.tensor(x.shape[2], dtype=torch.float)
            self.dim_sqrt = torch.sqrt(n).to(x.device)

        # 归一化：计算均值和方差（标准差）
        x_int = x / scaling_factor
        mean_int = round_ste.apply(x_int.mean(axis=2, keepdim=True))
        y_int = x_int - mean_int
        y_int_shifted = floor_ste.apply(y_int / 2**self.shift)
        y_sq_int = y_int_shifted**2
        var_int = torch.sum(y_sq_int, axis=2, keepdim=True)

        # 训练时的溢出处理
        if self.training:
            # 如果检测到溢出
            if var_int.max() >= 2**self.max_bit:
                # 使用溢出回退函数处理 var_int
                var_int = self.overflow_fallback(y_int)
                # 断言确保 var_int 未超过 self.max_bit
                assert var_int.max() < 2**self.max_bit + 0.1, (
                    "Error detected in overflow handling: "
                    "`var_int` exceeds `self.max_bit` (the maximum possible bit width)"
                )

        # 待替换为产生相同输出的整数平方根核
        std_int = floor_ste.apply(torch.sqrt(var_int)) * 2**self.shift
        factor = floor_ste.apply(2**31 / std_int)
        y_int = floor_ste.apply(y_int * factor / 2)
        scaling_factor = self.dim_sqrt / 2**30

        # 缩放和移位
        bias = self.bias.data.detach() / (self.weight.data.detach())
        bias_int = floor_ste.apply(bias / scaling_factor)

        y_int = y_int + bias_int
        scaling_factor = scaling_factor * self.weight
        x = y_int * scaling_factor

        # 返回处理后的 x 和更新后的 scaling_factor
        return x, scaling_factor
# 计算给定张量中的百分位最大和最小值
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

    # 获取上分位数的值
    upper_bound = torch.kthvalue(input, k=upper_index).values

    # 如果下分位数为0，则下分位数为0，否则获取下分位数的值
    if lower_percentile == 0:
        lower_bound = upper_bound * 0
        # lower_index += 1
    else:
        lower_bound = -torch.kthvalue(-input, k=lower_index).values

    # 如果不需要返回张量，则将下分位数和上分位数转换为标量值
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
        zero_pint (`torch.Tensor`):
            Shift for quantization.
        inplace (`bool`, *optional*, defaults to `False`):
            Whether to compute inplace or not.

    Returns:
        `torch.Tensor`: Linearly quantized value of *input* according to *scale* and *zero_point*.
    """
    # 为卷积权重和激活函数重新调整scale和zeropoint的形状
    if len(input.shape) == 4:
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    # 为线性权重重新调整scale和zeropoint的形状
    elif len(input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    else:
        scale = scale.view(-1)
        zero_point = zero_point.view(-1)
    # 量化 = 浮点数 / scale + zero_point
    if inplace:
        input.mul_(1.0 / scale).add_(zero_point).round_()
        return input
    return torch.round(1.0 / scale * input + zero_point)


def symmetric_linear_quantization_params(num_bits, saturation_min, saturation_max, per_channel=False):
    """
    Compute the scaling factor with the given quantization range for symmetric quantization.
    Args:
        saturation_min (`torch.Tensor`):
            Lower bound for quantization range.
        saturation_max (`torch.Tensor`):
            Upper bound for quantization range.
        per_channel (`bool`, *optional*, defaults to `False`):
            Whether to or not use channel-wise quantization.

    Returns:
        `torch.Tensor`: Scaling factor that linearly quantizes the given range between *saturation_min* and
        *saturation_max*.
    """
    # in this part, we do not need any gradient computation,
    # in order to enforce this, we put torch.no_grad()
    with torch.no_grad():
        # Calculate the number of quantization levels
        n = 2 ** (num_bits - 1) - 1

        if per_channel:
            # Calculate the scaling factor for per-channel quantization
            scale, _ = torch.max(torch.stack([saturation_min.abs(), saturation_max.abs()], dim=1), dim=1)
            scale = torch.clamp(scale, min=1e-8) / n

        else:
            # Calculate the scaling factor for global quantization
            scale = max(saturation_min.abs(), saturation_max.abs())
            scale = torch.clamp(scale, min=1e-8) / n

    # Return the calculated scaling factor
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
        zero_point = torch.tensor(0.0).to(scale.device)

        n = 2 ** (k - 1) - 1
        new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)
        new_quant_x = torch.clamp(new_quant_x, -n, n - 1)

        ctx.scale = scale
        return new_quant_x

    @staticmethod
    def backward(ctx, grad_output):
        scale = ctx.scale
        if len(grad_output.shape) == 4:
            scale = scale.view(-1, 1, 1, 1)
        # reshape scale and zeropoint for linear weights
        elif len(grad_output.shape) == 2:
            scale = scale.view(-1, 1)
        else:
            scale = scale.view(-1)

        return grad_output.clone() / scale, None, None, None, None


class floor_ste(Function):
    """
    Straight-through Estimator(STE) for torch.floor()
    """

    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


class round_ste(Function):
    """
    Straight-through Estimator(STE) for torch.round()
    """

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
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

    shape_of_input = inputs.size()

    # trans the input to be a 1-d tensor
    inputs = inputs.view(-1)

    output_m, output_e = np.frexp(inputs.cpu().numpy())
    tmp_m = []
    for m in output_m:
        int_m_shifted = int(
            decimal.Decimal(m * (2**max_bit)).quantize(decimal.Decimal("1"), rounding=decimal.ROUND_HALF_UP)
        )
        tmp_m.append(int_m_shifted)
    output_m = np.array(tmp_m)

    output_e = float(max_bit) - output_e

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
        # Check if the input tensor has 3 dimensions, if not reshape it
        if len(pre_act_scaling_factor.shape) == 3:
            reshape = lambda x: x  # noqa: E731
        else:
            reshape = lambda x: x.view(1, 1, -1)  # noqa: E731
        ctx.identity = identity

        # Calculate the maximum value that can be represented with the given bitwidth
        n = 2 ** (bit_num - 1) - 1

        # Perform operations with no gradient calculation
        with torch.no_grad():
            pre_act_scaling_factor = reshape(pre_act_scaling_factor)
            if identity is not None:
                identity_scaling_factor = reshape(identity_scaling_factor)

            ctx.z_scaling_factor = z_scaling_factor

            # Perform fixed-point multiplication
            z_int = torch.round(pre_act / pre_act_scaling_factor)
            _A = pre_act_scaling_factor.type(torch.double)
            _B = (z_scaling_factor.type(torch.float)).type(torch.double)
            new_scale = _A / _B
            new_scale = reshape(new_scale)

            m, e = batch_frexp(new_scale)

            output = z_int.type(torch.double) * m.type(torch.double)
            output = torch.round(output / (2.0**e))

            if identity is not None:
                # Perform fixed-point multiplication for identity tensor
                wx_int = torch.round(identity / identity_scaling_factor)

                _A = identity_scaling_factor.type(torch.double)
                _B = (z_scaling_factor.type(torch.float)).type(torch.double)
                new_scale = _A / _B
                new_scale = reshape(new_scale)

                m1, e1 = batch_frexp(new_scale)
                output1 = wx_int.type(torch.double) * m1.type(torch.double)
                output1 = torch.round(output1 / (2.0**e1))

                # Add the result of identity tensor multiplication to the main output
                output = output1 + output

            # Clamp the output within the range of representable values
            return torch.clamp(output.type(torch.float), -n - 1, n)

    @staticmethod
    # 定义反向传播函数，接收上下文和梯度输出作为参数
    def backward(ctx, grad_output):
        # 初始化 identity_grad 为 None
        identity_grad = None
        # 如果上下文中的 identity 不为 None
        if ctx.identity is not None:
            # 将梯度输出克隆一份并除以上下文中的 z_scaling_factor，作为 identity_grad
            identity_grad = grad_output.clone() / ctx.z_scaling_factor
        # 将梯度输出克隆一份并除以上下文中的 z_scaling_factor，作为返回的梯度
        return grad_output.clone() / ctx.z_scaling_factor, None, None, None, None, identity_grad, None
```