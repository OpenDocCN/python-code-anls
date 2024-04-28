# `.\transformers\models\vits\modeling_vits.py`

```
# coding=utf-8

# 导入必要的库
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn

# 导入相关模块
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import (
    BaseModelOutput,
    ModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_vits import VitsConfig


# 创建并获取logger对象用于日志记录
logger = logging.get_logger(__name__)

# 定义一个配置变量，值为"VitsConfig"，用于文档说明
_CONFIG_FOR_DOC = "VitsConfig"

# 预训练模型列表
VITS_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/mms-tts-eng",
    # 可供查看所有VITS模型的链接
    # https://huggingface.co/models?filter=vits
    # 可供查看所有MMS模型的链接
    # https://huggingface.co/models?sort=trending&search=facebook%2Fmms-tts
]


# VitsModelOutput类，继承ModelOutput类
@dataclass
class VitsModelOutput(ModelOutput):
    """
    VITS模型的输出，包括可能的隐藏状态和注意力结果。

    """

    # 更多注释...
    Args:
        waveform (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            The final audio waveform predicted by the model.
            由模型预测得出的最终音频波形。

        sequence_lengths  (`torch.FloatTensor` of shape `(batch_size,)`):
            The length in samples of each element in the `waveform` batch.
            `waveform` 批处理中每个元素的样本长度（以样本为单位）。

        spectrogram (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_bins)`):
            The log-mel spectrogram predicted at the output of the flow model. This spectrogram is passed to the Hi-Fi GAN decoder model to obtain the final audio waveform.
            在流模型的输出处预测的对数梅尔频谱图。这个频谱图将传递给 Hi-Fi GAN 解码器模型，以获取最终的音频波形。

        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            模型在每一层输出的隐藏状态的元组，包括可选的初始嵌入层输出。隐藏状态由 `torch.FloatTensor` 组成，形状为 `(batch_size, sequence_length, hidden_size)`。

        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
            当 `output_attentions=True` 被传递或 `config.output_attentions=True` 时，返回每一层注意力权重的元组。注意力权重是 `torch.FloatTensor` 组成，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

    """

    waveform: torch.FloatTensor = None
    sequence_lengths: torch.FloatTensor = None
    spectrogram: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# 定义一个数据类，描述了VITS文本编码器模型的输出，包括潜在隐藏状态和注意力
@dataclass
class VitsTextEncoderOutput(ModelOutput):
    """
    Describes the outputs for the VITS text encoder model, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        prior_means (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            The predicted mean values of the prior distribution for the latent text variables.
        prior_log_variances (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            The predicted log-variance values of the prior distribution for the latent text variables.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attention weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: torch.FloatTensor = None
    prior_means: torch.FloatTensor = None
    prior_log_variances: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# 定义一个Torch脚本函数，实现融合的加、tanh、sigmoid和乘法操作
@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, num_channels):
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :num_channels, :])
    s_act = torch.sigmoid(in_act[:, num_channels:, :])
    acts = t_act * s_act
    return acts

# 定义一个函数，实现无约束的有理二次样条变换，表示一个单调递增的分段有理二次函数
def _unconstrained_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    reverse=False,
    tail_bound=5.0,
    min_bin_width=1e-3,
    min_bin_height=1e-3,
    min_derivative=1e-3,
):
    """
    This transformation represents a monotonically increasing piecewise rational quadratic function. Outside of the
    `tail_bound`, the transform behaves as an identity function.
    """
    Args:
        inputs (`torch.FloatTensor` of shape `(batch_size, channels, seq_len)`:
            Vits卷积流模块的隐藏状态输入的后半部分。
        unnormalized_widths (`torch.FloatTensor` of shape `(batch_size, channels, seq_len, duration_predictor_flow_bins)`):
            来自卷积流模块中卷积投影层输出的隐藏状态的前`duration_predictor_flow_bins`。
        unnormalized_heights (`torch.FloatTensor` of shape `(batch_size, channels, seq_len, duration_predictor_flow_bins)`):
            来自卷积流模块中卷积投影层输出的隐藏状态的第二个`duration_predictor_flow_bins`。
        unnormalized_derivatives (`torch.FloatTensor` of shape `(batch_size, channels, seq_len, duration_predictor_flow_bins)`):
            来自卷积流模块中卷积投影层输出的隐藏状态的第三个`duration_predictor_flow_bins`。
        reverse (`bool`, *optional*, defaults to `False`):
            模型是否以反向模式运行。
        tail_bound (`float`, *optional* defaults to 5):
            有理二次函数的上限和下限范围。超出此`tail_bound`范围，变换行为就像是一个恒等函数。
        min_bin_width (`float`, *optional*, defaults to 1e-3):
            用于分段有理二次函数的宽度维度的最小bin值。
        min_bin_height (`float`, *optional*, defaults to 1e-3):
            用于分段有理二次函数的高度维度的最小bin值。
        min_derivative (`float`, *optional*, defaults to 1e-3):
            用于分段有理二次函数的导数的最小bin值。
    Returns:
        outputs (`torch.FloatTensor` of shape `(batch_size, channels, seq_len)`:
            通过应用`tail_bound`限制后的分段有理二次函数变换的隐藏状态。
        log_abs_det (`torch.FloatTensor` of shape `(batch_size, channels, seq_len)`:
            对应于应用`tail_bound`限制后的`outputs`的绝对值的行列式的对数。
    """
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    log_abs_det = torch.zeros_like(inputs)
    constant = np.log(np.exp(1 - min_derivative) - 1)

    unnormalized_derivatives = nn.functional.pad(unnormalized_derivatives, pad=(1, 1))
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant

    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    log_abs_det[outside_interval_mask] = 0.0
    # 对输入 inputs 中属于 inside_interval_mask 掩码范围内的部分执行有理二次样条变换
    # 返回变换后的输出 outputs 和变换的对数绝对值行列式 log_abs_det
    outputs[inside_interval_mask], log_abs_det[inside_interval_mask] = _rational_quadratic_spline(
        # 输入 inputs 中属于 inside_interval_mask 掩码范围内的部分
        inputs=inputs[inside_interval_mask],
        # 对应的未归一化的宽度参数
        unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
        # 对应的未归一化的高度参数
        unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
        # 对应的未归一化的导数参数
        unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
        # 是否反向变换
        reverse=reverse,
        # 尾部约束
        tail_bound=tail_bound,
        # 最小的柱状图宽度
        min_bin_width=min_bin_width,
        # 最小的柱状图高度
        min_bin_height=min_bin_height,
        # 最小导数值
        min_derivative=min_derivative,
    )
    # 返回变换后的输出 outputs 和变换的对数绝对值行列式 log_abs_det
    return outputs, log_abs_det
def _rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    reverse,
    tail_bound,
    min_bin_width,
    min_bin_height,
    min_derivative,
):
    """
    This transformation represents a monotonically increasing piecewise rational quadratic function. Unlike the
    function `_unconstrained_rational_quadratic_spline`, the function behaves the same across the `tail_bound`.

    Args:
        inputs (`torch.FloatTensor` of shape `(batch_size, channels, seq_len)`):
            Second half of the hidden-states input to the Vits convolutional flow module.
        unnormalized_widths (`torch.FloatTensor` of shape `(batch_size, channels, seq_len, duration_predictor_flow_bins)`):
            First `duration_predictor_flow_bins` of the hidden-states from the output of the convolution projection
            layer in the convolutional flow module
        unnormalized_heights (`torch.FloatTensor` of shape `(batch_size, channels, seq_len, duration_predictor_flow_bins)`):
            Second `duration_predictor_flow_bins` of the hidden-states from the output of the convolution projection
            layer in the convolutional flow module
        unnormalized_derivatives (`torch.FloatTensor` of shape `(batch_size, channels, seq_len, duration_predictor_flow_bins)`):
            Third `duration_predictor_flow_bins` of the hidden-states from the output of the convolution projection
            layer in the convolutional flow module
        reverse (`bool`):
            Whether the model is being run in reverse mode.
        tail_bound (`float`):
            Upper and lower limit bound for the rational quadratic function. Outside of this `tail_bound`, the
            transform behaves as an identity function.
        min_bin_width (`float`):
            Minimum bin value across the width dimension for the piecewise rational quadratic function.
        min_bin_height (`float`):
            Minimum bin value across the height dimension for the piecewise rational quadratic function.
        min_derivative (`float`):
            Minimum bin value across the derivatives for the piecewise rational quadratic function.
    Returns:
        outputs (`torch.FloatTensor` of shape `(batch_size, channels, seq_len)`:
            Hidden-states as transformed by the piecewise rational quadratic function.
        log_abs_det (`torch.FloatTensor` of shape `(batch_size, channels, seq_len)`:
            Logarithm of the absolute value of the determinants corresponding to the `outputs`.
    """
    # 计算上界和下界
    upper_bound = tail_bound
    lower_bound = -tail_bound

    # 如果输入超出了定义域范围，则引发值错误
    if torch.min(inputs) < lower_bound or torch.max(inputs) > upper_bound:
        raise ValueError("Input to a transform is not within its domain")

    # 获取宽度维度的数量
    num_bins = unnormalized_widths.shape[-1]

    # 如果最小的 bin 宽度乘以 bin 的数量大于 1.0，则引发值错误
    if min_bin_width * num_bins > 1.0:
        raise ValueError(f"Minimal bin width {min_bin_width} too large for the number of bins {num_bins}")
    # 如果最小 bin 的高度乘以 bin 的数量大于 1.0，则抛出异常
    if min_bin_height * num_bins > 1.0:
        raise ValueError(f"Minimal bin height {min_bin_height} too large for the number of bins {num_bins}")

    # 使用 softmax 函数对未归一化的宽度进行归一化计算
    widths = nn.functional.softmax(unnormalized_widths, dim=-1)
    # 使用线性插值计算每个 bin 的宽度
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    # 计算累积宽度并进行填充
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = nn.functional.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    # 将累积宽度映射到指定的范围
    cumwidths = (upper_bound - lower_bound) * cumwidths + lower_bound
    cumwidths[..., 0] = lower_bound
    cumwidths[..., -1] = upper_bound
    # 计算每个 bin 的宽度
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    # 计算导数并进行 softplus 操作
    derivatives = min_derivative + nn.functional.softplus(unnormalized_derivatives)

    # 使用 softmax 函数对未归一化的高度进行归一化计算
    heights = nn.functional.softmax(unnormalized_heights, dim=-1)
    # 使用线性插值计算每个 bin 的高度
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    # 计算累积高度并进行填充
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = nn.functional.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    # 将累积高度映射到指定的范围
    cumheights = (upper_bound - lower_bound) * cumheights + lower_bound
    cumheights[..., 0] = lower_bound
    cumheights[..., -1] = upper_bound
    # 计算每个 bin 的高度
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    # 根据是否反向映射，选择使用高度还是宽度作为 bin 的定位信息
    bin_locations = cumheights if reverse else cumwidths
    # 避免与上一 bin 重叠
    bin_locations[..., -1] += 1e-6
    # 根据输入选择合适的 bin
    bin_idx = torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1
    bin_idx = bin_idx[..., None]

    # 获取当前 bin 的相关信息
    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    # 计算当前 bin 的高度与宽度之比
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    intermediate1 = input_derivatives + input_derivatives_plus_one - 2 * input_delta
    if not reverse:
        # 计算 theta 和 theta*(1-theta)，用于后续计算
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        # 计算输出值
        numerator = input_heights * (input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + intermediate1 * theta_one_minus_theta
        outputs = input_cumheights + numerator / denominator

        # 计算雅可比行列式的对数值
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2)
        )
        log_abs_det = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, log_abs_det
    else:
        # 计算二次方程的根
        intermediate2 = inputs - input_cumheights
        # 计算中间变量
        intermediate3 = intermediate2 * intermediate1
        # 计算二次方程的系数
        a = input_heights * (input_delta - input_derivatives) + intermediate3
        b = input_heights * input_derivatives - intermediate3
        c = -input_delta * intermediate2

        # 计算判别式
        discriminant = b.pow(2) - 4 * a * c
        # 如果判别式不全为非负数，抛出异常
        if not (discriminant >= 0).all():
            raise RuntimeError(f"invalid discriminant {discriminant}")

        # 计算根
        root = (2 * c) / (-b - torch.sqrt(discriminant))
        # 计算输出值
        outputs = root * input_bin_widths + input_cumwidths

        # 计算导数的分母部分
        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + intermediate1 * theta_one_minus_theta
        # 计算导数的分子部分
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        # 计算对数绝对值的行列式
        log_abs_det = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        # 返回结果
        return outputs, -log_abs_det
class VitsWaveNet(torch.nn.Module):
    # 定义一个名为 VitsWaveNet 的类，继承自 torch.nn.Module
    def __init__(self, config: VitsConfig, num_layers: int):
        # 初始化函数，接收配置参数 VitsConfig 和层数参数 num_layers
        super().__init__()
        # 调用父类的初始化函数

        self.hidden_size = config.hidden_size
        # 将隐藏层大小设为配置中的隐藏层大小
        self.num_layers = num_layers
        # 将层数设为传入的层数

        self.in_layers = torch.nn.ModuleList()
        # 初始化一个空的 ModuleList 用于存储输入层
        self.res_skip_layers = torch.nn.ModuleList()
        # 初始化一个空的 ModuleList 用于存储残差跳跃连接层
        self.dropout = nn.Dropout(config.wavenet_dropout)
        # 创建一个丢弃层

        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm
        else:
            weight_norm = nn.utils.weight_norm
        # 检查是否存在 weight_norm 方法，如果不存在则使用默认 weight_norm 方法

        if config.speaker_embedding_size != 0:
            # 如果说话者嵌入维度不为零
            cond_layer = torch.nn.Conv1d(config.speaker_embedding_size, 2 * config.hidden_size * num_layers, 1)
            # 创建一个卷积层用于处理说话者嵌入
            self.cond_layer = weight_norm(cond_layer, name="weight")
            # 对卷积层进行权重归一化

        for i in range(num_layers):
            # 循环遍历层数
            dilation = config.wavenet_dilation_rate**i
            # 计算当前层的膨胀率
            padding = (config.wavenet_kernel_size * dilation - dilation) // 2
            # 根据膨胀率计算填充大小
            in_layer = torch.nn.Conv1d(
                in_channels=config.hidden_size,
                out_channels=2 * config.hidden_size,
                kernel_size=config.wavenet_kernel_size,
                dilation=dilation,
                padding=padding,
            )
            # 创建一个卷积层作为输入层
            in_layer = weight_norm(in_layer, name="weight")
            # 对卷积层进行权重归一化
            self.in_layers.append(in_layer)
            # 将输入层添加到输入层列表中

            if i < num_layers - 1:
                # 如果不是最后一层
                res_skip_channels = 2 * config.hidden_size
            else:
                res_skip_channels = config.hidden_size
            # 计算残差跳跃连接层的输出通道数

            res_skip_layer = torch.nn.Conv1d(config.hidden_size, res_skip_channels, 1)
            # 创建一个卷积层作为残差跳跃连接层
            res_skip_layer = weight_norm(res_skip_layer, name="weight")
            # 对卷积层进行权重归一化
            self.res_skip_layers.append(res_skip_layer)
            # 将残差跳跃连接层添加到列表中
    # 前向传播函数，接收输入、填充掩码和全局条件，返回输出
    def forward(self, inputs, padding_mask, global_conditioning=None):
        # 初始化一个与输入相同大小的零张量作为输出
        outputs = torch.zeros_like(inputs)
        # 创建包含隐藏层大小的整数张量
        num_channels_tensor = torch.IntTensor([self.hidden_size])

        # 如果存在全局条件
        if global_conditioning is not None:
            # 使用条件层处理全局条件
            global_conditioning = self.cond_layer(global_conditioning)

        # 遍历每一层
        for i in range(self.num_layers):
            # 在输入上使用输入层处理得到隐藏状态
            hidden_states = self.in_layers[i](inputs)

            # 如果存在全局条件
            if global_conditioning is not None:
                # 计算全局状态偏移量
                cond_offset = i * 2 * self.hidden_size
                # 从全局条件中获取全局状态
                global_states = global_conditioning[:, cond_offset : cond_offset + 2 * self.hidden_size, :]
            else:
                # 如果不存在全局条件，则将全局状态初始化为与隐藏状态相同大小的零张量
                global_states = torch.zeros_like(hidden_states)

            # 经过融合的激活函数处理隐藏状态和全局状态，得到激活值
            acts = fused_add_tanh_sigmoid_multiply(hidden_states, global_states, num_channels_tensor[0])
            # 对激活值应用丢弃操作
            acts = self.dropout(acts)

            # 通过残差连接层处理激活值，得到残差跳连激活值
            res_skip_acts = self.res_skip_layers[i](acts)
            # 如果不是最后一层
            if i < self.num_layers - 1:
                # 截取残差激活值中前半部分，作为残差激活值
                res_acts = res_skip_acts[:, : self.hidden_size, :]
                # 对输入应用残差跳连和填充掩码的组合操作，更新输入
                inputs = (inputs + res_acts) * padding_mask
                # 更新输出，加上残差跳连激活值的后半部分
                outputs = outputs + res_skip_acts[:, self.hidden_size :, :]
            else:
                # 如果是最后一层，直接更新输出，加上残差跳连激活值
                outputs = outputs + res_skip_acts

        # 返回输出乘以填充掩码
        return outputs * padding_mask

    # 移除权重归一化
    def remove_weight_norm(self):
        # 如果说话者嵌入大小不为0，移除条件层的权重归一化
        if self.speaker_embedding_size != 0:
            torch.nn.utils.remove_weight_norm(self.cond_layer)
        # 遍历输入层，移除权重归一化
        for layer in self.in_layers:
            torch.nn.utils.remove_weight_norm(layer)
        # 遍历残差跳连层，移除权重归一化
        for layer in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(layer)
# 定义 VitsPosteriorEncoder 类，继承自 nn.Module 类
class VitsPosteriorEncoder(nn.Module):
    # 初始化方法，接受一个 VitsConfig 类型的参数
    def __init__(self, config: VitsConfig):
        super().__init__()
        # 设置输出通道数为 config.flow_size
        self.out_channels = config.flow_size

        # 创建一个 1D 的卷积层，输入通道数为 config.spectrogram_bins，输出通道数为 config.hidden_size，卷积核大小为 1
        self.conv_pre = nn.Conv1d(config.spectrogram_bins, config.hidden_size, 1)
        # 创建一个 VitsWaveNet 对象
        self.wavenet = VitsWaveNet(config, num_layers=config.posterior_encoder_num_wavenet_layers)
        # 创建一个 1D 的卷积层，输入通道数为 config.hidden_size，输出通道数为 self.out_channels * 2，卷积核大小为 1
        self.conv_proj = nn.Conv1d(config.hidden_size, self.out_channels * 2, 1)

    # 前向传播方法，接受 inputs、padding_mask 和 global_conditioning 三个参数
    def forward(self, inputs, padding_mask, global_conditioning=None):
        # 对 inputs 进行卷积操作，并乘以 padding_mask
        inputs = self.conv_pre(inputs) * padding_mask
        # 将处理后的 inputs 输入到 wavenet 中进行处理
        inputs = self.wavenet(inputs, padding_mask, global_conditioning)
        # 对处理后的结果进行卷积操作，并乘以 padding_mask
        stats = self.conv_proj(inputs) * padding_mask
        # 将卷积后的结果拆分为均值和对数标准差
        mean, log_stddev = torch.split(stats, self.out_channels, dim=1)
        # 对均值进行采样得到结果
        sampled = (mean + torch.randn_like(mean) * torch.exp(log_stddev)) * padding_mask
        # 返回采样结果、均值和对数标准差
        return sampled, mean, log_stddev


# 定义 HifiGanResidualBlock 类，继承自 nn.Module 类
# 从 transformers.models.speecht5.modeling_speecht5.HifiGanResidualBlock 复制而来
class HifiGanResidualBlock(nn.Module):
    # 初始化方法，接受 channels、kernel_size、dilation 和 leaky_relu_slope 四个参数
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), leaky_relu_slope=0.1):
        super().__init__()
        # 设置 leaky_relu_slope 属性为给定的 leaky_relu_slope 值
        self.leaky_relu_slope = leaky_relu_slope

        # 创建一组具有不同 dilation 的卷积层并加入到 con1 中
        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation[i],
                    padding=self.get_padding(kernel_size, dilation[i]),
                )
                for i in range(len(dilation))
            ]
        )
        # 创建一组 dilation 为 1 的卷积层并加入到 conv2 中
        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    stride=1,
                    dilation=1,
                    padding=self.get_padding(kernel_size, 1),
                )
                for _ in range(len(dilation))
            ]
        )

    # 计算 padding 的方法，接受 kernel_size 和 dilation 两个参数
    def get_padding(self, kernel_size, dilation=1):
        return (kernel_size * dilation - dilation) // 2

    # 添加权重归一化
    def apply_weight_norm(self):
        for layer in self.convs1:
            nn.utils.weight_norm(layer)
        for layer in self.convs2:
            nn.utils.weight_norm(layer)

    # 移除权重归一化
    def remove_weight_norm(self):
        for layer in self.convs1:
            nn.utils.remove_weight_norm(layer)
        for layer in self.convs2:
            nn.utils.remove_weight_norm(layer)

    # 前向传播方法，接受 hidden_states 一个参数
    def forward(self, hidden_states):
        # 对 conv1 和 conv2 一一对应地进行操作
        for conv1, conv2 in zip(self.convs1, self.convs2):
            # 保存 hidden_states 为 residual
            residual = hidden_states
            # 对 hidden_states 应用 leaky_relu 函数
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            # 通过 conv1 进行卷积
            hidden_states = conv1(hidden_states)
            # 对卷积结果应用 leaky_relu 函数
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            # 通过 conv2 进行卷积
            hidden_states = conv2(hidden_states)
            # 加上残差项
            hidden_states = hidden_states + residual
        # 返回最终的 hidden_states
        return hidden_states

# 定义 VitsHifiGan 类
class VitsHifiGan(nn.Module):
``` 
    # 初始化函数，接受一个 VitsConfig 对象作为参数
    def __init__(self, config: VitsConfig):
        # 调用父类的初始化函数
        super().__init__()
        # 将传入的配置对象保存到当前对象的 config 属性中
        self.config = config
        # 计算配置中残差块的数量
        self.num_kernels = len(config.resblock_kernel_sizes)
        # 计算配置中上采样率的数量
        self.num_upsamples = len(config.upsample_rates)
        # 创建一个一维卷积层，用于数据预处理
        self.conv_pre = nn.Conv1d(
            config.flow_size,  # 输入通道数为 config 中的 flow_size
            config.upsample_initial_channel,  # 输出通道数为 config 中的 upsample_initial_channel
            kernel_size=7,  # 卷积核大小为 7
            stride=1,  # 步长为 1
            padding=3,  # 填充大小为 3
        )
    
        # 创建一个存储上采样器的模块列表
        self.upsampler = nn.ModuleList()
        # 遍历配置中的上采样率和上采样核大小
        for i, (upsample_rate, kernel_size) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            # 将一个反卷积层添加到上采样器列表中
            self.upsampler.append(
                nn.ConvTranspose1d(
                    config.upsample_initial_channel // (2**i),  # 输入通道数随层数变化而减半
                    config.upsample_initial_channel // (2 ** (i + 1)),  # 输出通道数随层数变化而减半
                    kernel_size=kernel_size,  # 使用配置中的上采样核大小
                    stride=upsample_rate,  # 使用配置中的上采样率作为步长
                    padding=(kernel_size - upsample_rate) // 2,  # 使用填充保持输入输出大小一致
                )
            )
    
        # 创建一个存储残差块的模块列表
        self.resblocks = nn.ModuleList()
        # 遍历上采样器的数量
        for i in range(len(self.upsampler)):
            # 计算当前层的输出通道数
            channels = config.upsample_initial_channel // (2 ** (i + 1))
            # 遍历配置中的残差块核大小和扩张率
            for kernel_size, dilation in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
                # 将一个 HiFi-GAN 残差块添加到残差块列表中
                self.resblocks.append(HifiGanResidualBlock(channels, kernel_size, dilation, config.leaky_relu_slope))
    
        # 创建一个一维卷积层，用于最终处理输出
        self.conv_post = nn.Conv1d(channels, 1, kernel_size=7, stride=1, padding=3, bias=False)
    
        # 如果配置中的说话人嵌入大小不为 0，则创建一个条件卷积层
        if config.speaker_embedding_size != 0:
            self.cond = nn.Conv1d(config.speaker_embedding_size, config.upsample_initial_channel, 1)
    
    # 对所有上采样器和残差块应用权重归一化
    def apply_weight_norm(self):
        for layer in self.upsampler:
            nn.utils.weight_norm(layer)
        for layer in self.resblocks:
            layer.apply_weight_norm()
    
    # 移除所有上采样器和残差块的权重归一化
    def remove_weight_norm(self):
        for layer in self.upsampler:
            nn.utils.remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()
    
    # 前向传播函数，接受输入的频谱图和全局条件
    def forward(
        self, spectrogram: torch.FloatTensor, global_conditioning: Optional[torch.FloatTensor] = None
    # 将频谱图转换为语音波形的函数
    
    
        ) -> torch.FloatTensor:
            r"""
            Converts a spectrogram into a speech waveform.
    
            Args:
                spectrogram (`torch.FloatTensor` of shape `(batch_size, config.spectrogram_bins, sequence_length)`):
                    Tensor containing the spectrograms.
                global_conditioning (`torch.FloatTensor` of shape `(batch_size, config.speaker_embedding_size, 1)`, *optional*):
                    Tensor containing speaker embeddings, for multispeaker models.
    
            Returns:
                `torch.FloatTensor`: Tensor of shape shape `(batch_size, 1, num_frames)` containing the speech waveform.
            """
            
            # 使用预定义的卷积层对频谱图进行处理
            hidden_states = self.conv_pre(spectrogram)
    
            # 如果存在全局条件，将其与处理后的频谱图相加
            if global_conditioning is not None:
                hidden_states = hidden_states + self.cond(global_conditioning)
    
            # 经过多次上采样的过程
            for i in range(self.num_upsamples):
                # 使用leaky relu激活函数对特征进行处理
                hidden_states = nn.functional.leaky_relu(hidden_states, self.config.leaky_relu_slope)
                # 使用上采样器对特征进行上采样
                hidden_states = self.upsampler[i](hidden_states)
    
                # 使用残差块对特征进行卷积操作
                res_state = self.resblocks[i * self.num_kernels](hidden_states)
                for j in range(1, self.num_kernels):
                    res_state += self.resblocks[i * self.num_kernels + j](hidden_states)
                hidden_states = res_state / self.num_kernels
    
            # 使用leaky relu激活函数对特征进行处理
            hidden_states = nn.functional.leaky_relu(hidden_states)
            # 使用后卷积层对特征进行处理
            hidden_states = self.conv_post(hidden_states)
            # 使用tanh函数对特征进行处理，得到最终的语音波形
            waveform = torch.tanh(hidden_states)
            return waveform
    
    
    - 将频谱图转换为语音波形的函数
    - 参数：
        - `spectrogram`：形状为`(batch_size, config.spectrogram_bins, sequence_length)`的张量，包含频谱图
        - `global_conditioning`：形状为`(batch_size, config.speaker_embedding_size, 1)`的张量，包含全局条件，用于多说话人模型（可选）
    - 返回：
        - `torch.FloatTensor`：形状为`(batch_size, 1, num_frames)`的张量，包含语音波形
    
    - 线性卷积操作对频谱图进行处理
    - 如果`global_conditioning`存在，则与处理后的频谱图相加
    - 多次上采样的过程
    - 使用leaky relu激活函数对特征进行处理
    - 使用上采样器对特征进行上采样
    - 使用残差块对特征进行卷积操作
    - 使用leaky relu激活函数对特征进行处理
    - 使用后卷积层对特征进行处理
    - 使用tanh函数对特征进行处理，得到最终的语音波形
# 定义 VitsResidualCouplingLayer 类，继承自 nn.Module
class VitsResidualCouplingLayer(nn.Module):
    def __init__(self, config: VitsConfig):
        super().__init__()
        # 将 flow_size 除以 2 并赋值给 half_channels
        self.half_channels = config.flow_size // 2

        # 创建一个包含一个卷积层的模型，输入通道数为 half_channels，输出通道数为 hidden_size，卷积核大小为 1
        self.conv_pre = nn.Conv1d(self.half_channels, config.hidden_size, 1)
        # 创建 VitsWaveNet 模型，输入参数为 config，num_layers 属性设置为 config.prior_encoder_num_wavenet_layers
        self.wavenet = VitsWaveNet(config, num_layers=config.prior_encoder_num_wavenet_layers)
        # 创建一个包含一个卷积层的模型，输入通道数为 hidden_size，输出通道数为 half_channels，卷积核大小为 1
        self.conv_post = nn.Conv1d(config.hidden_size, self.half_channels, 1)

    # 前向传播函数，输入参数是 inputs、padding_mask、global_conditioning 和 reverse
    def forward(self, inputs, padding_mask, global_conditioning=None, reverse=False):
        # 将 inputs 拆分成两个部分，first_half 和 second_half，拆分的大小均为 half_channels
        first_half, second_half = torch.split(inputs, [self.half_channels] * 2, dim=1)
        # 将 first_half 输入到 conv_pre 模型中，并乘以 padding_mask
        hidden_states = self.conv_pre(first_half) * padding_mask
        # 将 hidden_states 输入到 wavenet 模型中，通过 wavenet 进行处理，并传入 padding_mask 和 global_conditioning
        hidden_states = self.wavenet(hidden_states, padding_mask, global_conditioning)
        # 将 hidden_states 进行第二次卷积处理，并乘以 padding_mask
        mean = self.conv_post(hidden_states) * padding_mask
        # 创建一个与 mean 大小相同的全零张量，并将其赋值给 log_stddev
        log_stddev = torch.zeros_like(mean)

        # 如果不是反向操作
        if not reverse:
            # 更新 second_half 的值为 mean + second_half * exp(log_stddev) * padding_mask
            second_half = mean + second_half * torch.exp(log_stddev) * padding_mask
            # 将 first_half 和 second_half 拼接在一起形成 outputs
            outputs = torch.cat([first_half, second_half], dim=1)
            # 按维度 [1, 2] 求 log_stddev 的和，并赋值给 log_determinant
            log_determinant = torch.sum(log_stddev, [1, 2])
            # 返回 outputs 和 log_determinant
            return outputs, log_determinant
        else:
            # 更新 second_half 的值为 (second_half - mean) * exp(-log_stddev) * padding_mask
            second_half = (second_half - mean) * torch.exp(-log_stddev) * padding_mask
            # 将 first_half 和 second_half 拼接在一起形成 outputs
            outputs = torch.cat([first_half, second_half], dim=1)
            # 返回 outputs 和 None
            return outputs, None


# 定义 VitsResidualCouplingBlock 类，继承自 nn.Module
class VitsResidualCouplingBlock(nn.Module):
    def __init__(self, config: VitsConfig):
        super().__init__()
        # 创建一个空的模型列表 flows
        self.flows = nn.ModuleList()
        # 根据 prior_encoder_num_flows 的数量，循环创建 VitsResidualCouplingLayer 模型，并添加到 flows 列表中
        # 每个 VitsResidualCouplingLayer 模型的参数为 config
        for _ in range(config.prior_encoder_num_flows):
            self.flows.append(VitsResidualCouplingLayer(config))

    # 前向传播函数，输入参数是 inputs、padding_mask、global_conditioning 和 reverse
    def forward(self, inputs, padding_mask, global_conditioning=None, reverse=False):
        # 如果不是反向操作
        if not reverse:
            # 对 flows 列表中的每个模型进行循环操作
            for flow in self.flows:
                # 将 inputs 和类似 None 的对象传入 flow 模型，并获取结果，赋值给 inputs，忽略返回值的第二个元素
                inputs, _ = flow(inputs, padding_mask, global_conditioning)
                # 对 inputs 进行维度翻转，维度为 [1]，赋值给 inputs
                inputs = torch.flip(inputs, [1])
        else:
            # 对 flows 列表进行倒序循环操作
            for flow in reversed(self.flows):
                # 对 inputs 进行维度翻转，维度为 [1]，赋值给 inputs
                inputs = torch.flip(inputs, [1])
                # 将 inputs 和类似 None 的对象传入 flow 模型，并获取结果，赋值给 inputs，忽略返回值的第二个元素
                inputs, _ = flow(inputs, padding_mask, global_conditioning, reverse=True)
        # 返回 inputs
        return inputs


class VitsDilatedDepthSeparableConv(nn.Module):
    # 初始化函数，接受配置和丢失率作为参数
    def __init__(self, config: VitsConfig, dropout_rate=0.0):
        # 调用父类初始化函数
        super().__init__()
        # 从配置中获取参数
        kernel_size = config.duration_predictor_kernel_size
        channels = config.hidden_size
        self.num_layers = config.depth_separable_num_layers

        # 初始化丢失率
        self.dropout = nn.Dropout(dropout_rate)
        # 初始化空的可变模块列表
        self.convs_dilated = nn.ModuleList()
        self.convs_pointwise = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()
        # 循环按层数初始化可变模块列表
        for i in range(self.num_layers):
            # 计算空洞卷积的膨胀率和填充值
            dilation = kernel_size**i
            padding = (kernel_size * dilation - dilation) // 2
            # 添加空洞卷积层
            self.convs_dilated.append(
                nn.Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    groups=channels,
                    dilation=dilation,
                    padding=padding,
                )
            )
            # 添加点卷积层
            self.convs_pointwise.append(nn.Conv1d(channels, channels, 1))
            # 添加LayerNorm层1
            self.norms_1.append(nn.LayerNorm(channels))
            # 添加LayerNorm层2
            self.norms_2.append(nn.LayerNorm(channels))

    # 前向传播函数，接受输入、填充掩码和全局条件参数
    def forward(self, inputs, padding_mask, global_conditioning=None):
        # 如果有全局条件参数，则将输入和全局条件相加
        if global_conditioning is not None:
            inputs = inputs + global_conditioning

        # 循环按层数进行操作
        for i in range(self.num_layers):
            # 获取当前空洞卷积层的输出
            hidden_states = self.convs_dilated[i](inputs * padding_mask)
            # 对输出进行LayerNorm和GELU激活函数操作
            hidden_states = self.norms_1[i](hidden_states.transpose(1, -1)).transpose(1, -1)
            hidden_states = nn.functional.gelu(hidden_states)
            # 对输出进行点卷积操作
            hidden_states = self.convs_pointwise[i](hidden_states)
            # 对输出再次进行LayerNorm和GELU激活函数操作
            hidden_states = self.norms_2[i](hidden_states.transpose(1, -1)).transpose(1, -1)
            hidden_states = nn.functional.gelu(hidden_states)
            # 对输出进行丢失率处理
            hidden_states = self.dropout(hidden_states)
            # 输入和处理后的输出相加
            inputs = inputs + hidden_states

        # 返回最终结果
        return inputs * padding_mask
# 定义一个名为 VitsConvFlow 的类，继承自 nn.Module
class VitsConvFlow(nn.Module):
    # 初始化方法，接收一个 VitsConfig 类型的参数 config
    def __init__(self, config: VitsConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 将配置参数中的隐藏层大小作为卷积层的过滤器通道数
        self.filter_channels = config.hidden_size
        # 将配置参数中的深度可分离通道数的一半作为 half_channels
        self.half_channels = config.depth_separable_channels // 2
        # 将配置参数中的持续时间预测器流的 bin 数作为 num_bins
        self.num_bins = config.duration_predictor_flow_bins
        # 将配置参数中的持续时间预测器尾部边界值作为 tail_bound
        self.tail_bound = config.duration_predictor_tail_bound

        # 创建一个 1D 卷积层，输入通道数为 half_channels，输出通道数为 filter_channels，卷积核大小为 1
        self.conv_pre = nn.Conv1d(self.half_channels, self.filter_channels, 1)
        # 创建一个 VitsDilatedDepthSeparableConv 类的实例
        self.conv_dds = VitsDilatedDepthSeparableConv(config)
        # 创建一个 1D 卷积层，输入通道数为 filter_channels，输出通道数为 half_channels * (num_bins * 3 - 1)，卷积核大小为 1
        self.conv_proj = nn.Conv1d(self.filter_channels, self.half_channels * (self.num_bins * 3 - 1), 1)

    # 前向传播方法，接收输入、填充掩码、全局条件和是否反转的标志作为参数
    def forward(self, inputs, padding_mask, global_conditioning=None, reverse=False):
        # 将输入张量沿通道维度分割成两部分，分别赋值给 first_half 和 second_half
        first_half, second_half = torch.split(inputs, [self.half_channels] * 2, dim=1)

        # 对第一部分进行预卷积
        hidden_states = self.conv_pre(first_half)
        # 运行深度可分离卷积操作
        hidden_states = self.conv_dds(hidden_states, padding_mask, global_conditioning)
        # 运行最终投影卷积操作并乘以填充掩码
        hidden_states = self.conv_proj(hidden_states) * padding_mask

        # 获取批量大小、通道数和长度
        batch_size, channels, length = first_half.shape
        # 将隐藏状态重新整形，并转置最后两个维度
        hidden_states = hidden_states.reshape(batch_size, channels, -1, length).permute(0, 1, 3, 2)

        # 计算未归一化的宽度、高度和导数
        unnormalized_widths = hidden_states[..., : self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_heights = hidden_states[..., self.num_bins : 2 * self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_derivatives = hidden_states[..., 2 * self.num_bins :]

        # 利用非受限有理二次样条进行拟合
        second_half, log_abs_det = _unconstrained_rational_quadratic_spline(
            second_half,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            reverse=reverse,
            tail_bound=self.tail_bound,
        )

        # 在通道维度上连接第一部分和第二部分并乘以填充掩码
        outputs = torch.cat([first_half, second_half], dim=1) * padding_mask
        # 如果不是反向操作，则计算对数行列式
        if not reverse:
            log_determinant = torch.sum(log_abs_det * padding_mask, [1, 2])
            return outputs, log_determinant
        # 如果是反向操作，则不返回对数行列式
        else:
            return outputs, None


# 定义一个名为 VitsElementwiseAffine 的类，继承自 nn.Module
class VitsElementwiseAffine(nn.Module):
    # 初始化方法，接收一个 VitsConfig 类型的参数 config
    def __init__(self, config: VitsConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 将配置参数中的深度可分离通道数作为 channels
        self.channels = config.depth_separable_channels
        # 创建一个参数，用于平移输入
        self.translate = nn.Parameter(torch.zeros(self.channels, 1))
        # 创建一个参数，用于缩放输入的对数
        self.log_scale = nn.Parameter(torch.zeros(self.channels, 1))

    # 前向传播方法，接收输入、填充掩码、全局条件和是否反转的标志作为参数
    def forward(self, inputs, padding_mask, global_conditioning=None, reverse=False):
        # 如果不是反向操作
        if not reverse:
            # 计算输出，进行平移并指数化缩放
            outputs = self.translate + torch.exp(self.log_scale) * inputs
            # 乘以填充掩码
            outputs = outputs * padding_mask
            # 计算对数行列式
            log_determinant = torch.sum(self.log_scale * padding_mask, [1, 2])
            return outputs, log_determinant
        # 如果是反向操作
        else:
            # 计算输出，进行反平移并指数化反缩放
            outputs = (inputs - self.translate) * torch.exp(-self.log_scale) * padding_mask
            return outputs, None


# 定义一个名为 VitsStochasticDurationPredictor 的类，继承自 nn.Module
class VitsStochasticDurationPredictor(nn.Module):
    # 初始化函数，传入配置参数
    def __init__(self, config):
        # 调用父类初始化函数
        super().__init__()
        # 从配置参数中获取说话者嵌入大小和隐藏层大小
        embed_dim = config.speaker_embedding_size
        filter_channels = config.hidden_size

        # 创建卷积层用于预处理
        self.conv_pre = nn.Conv1d(filter_channels, filter_channels, 1)
        # 创建卷积层用于投影
        self.conv_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        # 创建 VitsDilatedDepthSeparableConv 模块
        self.conv_dds = VitsDilatedDepthSeparableConv(
            config,
            dropout_rate=config.duration_predictor_dropout,
        )

        # 如果嵌入大小不为0，创建用于条件的卷积层
        if embed_dim != 0:
            self.cond = nn.Conv1d(embed_dim, filter_channels, 1)

        # 创建流模块列表
        self.flows = nn.ModuleList()
        # 添加 VitsElementwiseAffine 模块到流列表中
        self.flows.append(VitsElementwiseAffine(config))
        # 根据配置参数中的流数量循环添加 VitsConvFlow 模块到流列表中
        for _ in range(config.duration_predictor_num_flows):
            self.flows.append(VitsConvFlow(config))

        # 创建后处理的卷积层用于预处理
        self.post_conv_pre = nn.Conv1d(1, filter_channels, 1)
        # 创建后处理的卷积层用于投影
        self.post_conv_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        # 创建后处理的 VitsDilatedDepthSeparableConv 模块
        self.post_conv_dds = VitsDilatedDepthSeparableConv(
            config,
            dropout_rate=config.duration_predictor_dropout,
        )

        # 创建后处理的流模块列表
        self.post_flows = nn.ModuleList()
        # 添加后处理的 VitsElementwiseAffine 模块到流列表中
        self.post_flows.append(VitsElementwiseAffine(config))
        # 根据配置参数中的流数量循环添加后处理的 VitsConvFlow 模块到流列表中
        for _ in range(config.duration_predictor_num_flows):
            self.post_flows.append(VitsConvFlow(config))
class VitsDurationPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 从配置中获取持续时间预测器的卷积核大小和滤波器通道数
        kernel_size = config.duration_predictor_kernel_size
        filter_channels = config.duration_predictor_filter_channels

        # 定义丢弃层，用于随机失活
        self.dropout = nn.Dropout(config.duration_predictor_dropout)
        # 定义第一个卷积层，用于特征提取
        self.conv_1 = nn.Conv1d(config.hidden_size, filter_channels, kernel_size, padding=kernel_size // 2)
        # 定义第一个层归一化层，用于规范化特征
        self.norm_1 = nn.LayerNorm(filter_channels, eps=config.layer_norm_eps)
        # 定义第二个卷积层，用于特征提取
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        # 定义第二个层归一化层，用于规范化特征
        self.norm_2 = nn.LayerNorm(filter_channels, eps=config.layer_norm_eps)
        # 定义投影层，将特征映射到一个标量上
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        # 如果存在说话人嵌入大小，则定义条件卷积层
        if config.speaker_embedding_size != 0:
            self.cond = nn.Conv1d(config.speaker_embedding_size, config.hidden_size, 1)

    def forward(self, inputs, padding_mask, global_conditioning=None):
        # 将输入张量从计算图中分离出来，使得反向传播不会影响到它
        inputs = torch.detach(inputs)

        # 如果存在全局条件，则将全局条件加到输入张量上
        if global_conditioning is not None:
            global_conditioning = torch.detach(global_conditioning)
            inputs = inputs + self.cond(global_conditioning)

        # 第一次卷积操作
        inputs = self.conv_1(inputs * padding_mask)
        # 应用 ReLU 激活函数
        inputs = torch.relu(inputs)
        # 进行第一次层归一化
        inputs = self.norm_1(inputs.transpose(1, -1)).transpose(1, -1)
        # 应用丢弃层
        inputs = self.dropout(inputs)

        # 第二次卷积操作
        inputs = self.conv_2(inputs * padding_mask)
        # 应用 ReLU 激活函数
        inputs = torch.relu(inputs)
        # 进行第二次层归一化
        inputs = self.norm_2(inputs.transpose(1, -1)).transpose(1, -1)
        # 应用丢弃层
        inputs = self.dropout(inputs)

        # 投影层，将特征映射到一个标量上
        inputs = self.proj(inputs * padding_mask)
        return inputs * padding_mask


class VitsAttention(nn.Module):
    """Multi-headed attention with relative positional representation."""

    def __init__(self, config: VitsConfig):
        super().__init__()
        # 获取配置中的隐藏大小、注意力头数、注意力丢弃概率和窗口大小
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.dropout = config.attention_dropout
        self.window_size = config.window_size

        # 计算每个头的维度
        self.head_dim = self.embed_dim // self.num_heads
        # 用于缩放的值
        self.scaling = self.head_dim**-0.5

        # 如果隐藏大小不能被注意力头数整除，则引发错误
        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"hidden_size must be divisible by num_attention_heads (got `hidden_size`: {self.embed_dim}"
                f" and `num_attention_heads`: {self.num_heads})."
            )

        # 定义线性变换层，用于计算键、值和查询
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_bias)
        # 定义输出投影层
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_bias)

        # 如果存在窗口大小，则定义相对位置表示的参数
        if self.window_size:
            self.emb_rel_k = nn.Parameter(torch.randn(1, self.window_size * 2 + 1, self.head_dim) * self.scaling)
            self.emb_rel_v = nn.Parameter(torch.randn(1, self.window_size * 2 + 1, self.head_dim) * self.scaling)
    # 重新塑造张量的形状，将其视为 batch_size x seq_len x num_heads x head_dim，并交换维度
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 前向传播函数，用于 Transformer 模型的自注意力机制
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        
    # 获取相对位置嵌入，根据给定的长度和窗口大小进行填充和切片操作
    def _get_relative_embeddings(self, relative_embeddings, length):
        pad_length = max(length - (self.window_size + 1), 0)
        if pad_length > 0:
            relative_embeddings = nn.functional.pad(relative_embeddings, [0, 0, pad_length, pad_length, 0, 0])

        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        return relative_embeddings[:, slice_start_position:slice_end_position]

    # 将相对位置转换为绝对位置
    def _relative_position_to_absolute_position(self, x):
        batch_heads, length, _ = x.size()

        # 在列上进行填充，以进行相对位置到绝对位置的转换
        x = nn.functional.pad(x, [0, 1, 0, 0, 0, 0])

        # 在 x_flat 上进行填充，以便形成 shape 为 (length*2*length) 的张量
        x_flat = x.view([batch_heads, length * 2 * length])
        x_flat = nn.functional.pad(x_flat, [0, length - 1, 0, 0])

        # 重新塑造和切片，舍弃填充元素
        x_final = x_flat.view([batch_heads, length + 1, 2 * length - 1])
        x_final = x_final[:, :length, length - 1 :]
        return x_final

    # 将绝对位置转换为相对位置
    def _absolute_position_to_relative_position(self, x):
        batch_heads, length, _ = x.size()

        # 沿着列进行填充
        x = nn.functional.pad(x, [0, length - 1, 0, 0, 0, 0])
        x_flat = x.view([batch_heads, length * (2 * length - 1)])

        # 在变换后的元素之前添加 0，以使其在重塑后产生偏移
        x_flat = nn.functional.pad(x_flat, [length, 0, 0, 0])
        x_final = x_flat.view([batch_heads, length, 2 * length])[:, :, 1:]
        return x_final
# 前馈网络模块
class VitsFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 第一个卷积层，输入维度为 config.hidden_size，输出维度为 config.ffn_dim，核大小为 config.ffn_kernel_size
        self.conv_1 = nn.Conv1d(config.hidden_size, config.ffn_dim, config.ffn_kernel_size)
        # 第二个卷积层，输入维度为 config.ffn_dim，输出维度为 config.hidden_size，核大小为 config.ffn_kernel_size
        self.conv_2 = nn.Conv1d(config.ffn_dim, config.hidden_size, config.ffn_kernel_size)
        # dropout 层
        self.dropout = nn.Dropout(config.activation_dropout)

        # 根据 config.hidden_act 选择对应的激活函数
        if isinstance(config.hidden_act, str):
            self.act_fn = ACT2FN[config.hidden_act]
        else:
            self.act_fn = config.hidden_act

        # 根据 config.ffn_kernel_size 设置填充
        if config.ffn_kernel_size > 1:
            pad_left = (config.ffn_kernel_size - 1) // 2
            pad_right = config.ffn_kernel_size // 2
            self.padding = [pad_left, pad_right, 0, 0, 0, 0]
        else:
            self.padding = None

    def forward(self, hidden_states, padding_mask):
        # 将输入 hidden_states 从 [batch_size, hidden_size, seq_len] 转换为 [batch_size, seq_len, hidden_size]
        hidden_states = hidden_states.permute(0, 2, 1)
        # 将 padding_mask 从 [batch_size, seq_len] 转换为 [batch_size, seq_len, 1]
        padding_mask = padding_mask.permute(0, 2, 1)

        # 将 hidden_states 和 padding_mask 相乘，以屏蔽填充部分
        hidden_states = hidden_states * padding_mask
        # 如果 self.padding 不为 None，则对 hidden_states 进行填充
        if self.padding is not None:
            hidden_states = nn.functional.pad(hidden_states, self.padding)

        # 经过第一个卷积层和激活函数，再通过 dropout 层
        hidden_states = self.conv_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # 再次将 hidden_states 和 padding_mask 相乘，以屏蔽填充部分
        hidden_states = hidden_states * padding_mask
        # 如果 self.padding 不为 None，则对 hidden_states 进行填充
        if self.padding is not None:
            hidden_states = nn.functional.pad(hidden_states, self.padding)

        # 经过第二个卷积层，并将结果与 padding_mask 相乘，以屏蔽填充部分
        hidden_states = self.conv_2(hidden_states)
        hidden_states = hidden_states * padding_mask

        # 将 hidden_states 从 [batch_size, seq_len, hidden_size] 转换回 [batch_size, hidden_size, seq_len]
        hidden_states = hidden_states.permute(0, 2, 1)
        return hidden_states


# Transformer 编码器层
class VitsEncoderLayer(nn.Module):
    def __init__(self, config: VitsConfig):
        super().__init__()
        # 注意力模块
        self.attention = VitsAttention(config)
        # dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout)
        # LayerNorm 层
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 前馈网络模块
        self.feed_forward = VitsFeedForward(config)
        # 最终的 LayerNorm 层
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # 保存 hidden_states 作为残差
        residual = hidden_states
        # 经过注意力模块，得到 hidden_states 和注意力权重
        hidden_states, attn_weights = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        # 通过 dropout 层
        hidden_states = self.dropout(hidden_states)
        # 将残差和 hidden_states 相加，并通过 LayerNorm 层
        hidden_states = self.layer_norm(residual + hidden_states)

        # 保存 hidden_states 作为残差
        residual = hidden_states
        # 经过前馈网络模块
        hidden_states = self.feed_forward(hidden_states, padding_mask)
        # 通过 dropout 层
        hidden_states = self.dropout(hidden_states)
        # 将残差和 hidden_states 相加，并通过最终的 LayerNorm 层
        hidden_states = self.final_layer_norm(residual + hidden_states)

        # 返回 hidden_states，如果需要还可以返回注意力权重
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

# Transformer 编码器
class VitsEncoder(nn.Module):
        # 初始化方法，接受一个 VitsConfig 类型的参数
        def __init__(self, config: VitsConfig):
            # 调用父类的初始化方法
            super().__init__()
            # 将传入的配置参数保存在实例变量中
            self.config = config
            # 创建一个由多个 VitsEncoderLayer 组成的列表，列表的长度为配置参数中指定的层数
            self.layers = nn.ModuleList([VitsEncoderLayer(config) for _ in range(config.num_hidden_layers)])
            # 梯度检查点标志默认设置为 False
            self.gradient_checkpointing = False
            # 设置层丢弃的概率为配置参数中指定的层丢弃率
            self.layerdrop = config.layerdrop

        # 前向传播方法
        def forward(
            self,
            hidden_states: torch.FloatTensor,  # 输入参数：隐藏状态张量
            padding_mask: torch.FloatTensor,  # 输入参数：填充掩码张量
            attention_mask: Optional[torch.Tensor] = None,  # 输入参数：注意力掩码张量，可选
            output_attentions: Optional[bool] = None,  # 输入参数：是否输出注意力张量，可选
            output_hidden_states: Optional[bool] = None,  # 输入参数：是否输出隐藏状态，可选
            return_dict: Optional[bool] = None,  # 输入参数：是否以字典的形式返回结果，可选
    ) -> Union[Tuple, BaseModelOutput]:
        # 如果设置了输出隐藏状态，则初始化一个空元组，否则初始化为None
        all_hidden_states = () if output_hidden_states else None
        # 如果设置了输出注意力，则初始化一个空元组，否则初始化为None
        all_self_attentions = () if output_attentions else None

        # 扩展注意力掩码
        if attention_mask is not None:
            # 将形状为[bsz, seq_len]的注意力掩码扩展为[bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        # 将隐藏状态乘以填充掩码
        hidden_states = hidden_states * padding_mask

        # 检查是否启用了deepspeed zero3
        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

        # 遍历每一个编码器层
        for encoder_layer in self.layers:
            # 如果设置了输出隐藏状态，将当前隐藏状态添加到all_hidden_states中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 添加LayerDrop
            dropout_probability = np.random.uniform(0, 1)

            # 根据LayerDrop概率决定是否跳过当前层
            skip_the_layer = self.training and (dropout_probability < self.layerdrop)
            
            # 如果不跳过当前层或者启用了deepspeed zero3，则执行当前层的计算
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # 在deepspeed zero3下，所有GPU必须同步运行
                if self.gradient_checkpointing and self.training:
                    # 使用梯度检查点函数执行当前层的计算
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        padding_mask,
                        attention_mask,
                        output_attentions,
                    )
                else:
                    # 执行当前层的计算
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        padding_mask=padding_mask,
                        output_attentions=output_attentions,
                    )
                # 更新隐藏状态
                hidden_states = layer_outputs[0]

            # 如果跳过当前层，则将layer_outputs设置为None
            if skip_the_layer:
                layer_outputs = (None, None)

            # 如果设置了输出注意力，则将当前层的注意力添加到all_self_attentions中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 将隐藏状态再次乘以填充掩码
        hidden_states = hidden_states * padding_mask

        # 如果设置了输出隐藏状态，则将当前隐藏状态添加到all_hidden_states中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典，则返回包含不为None的元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        # 返回BaseModelOutput对象
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
class VitsTextEncoder(nn.Module):
    """
    Transformer编码器，使用相对位置表示而不是绝对位置编码。
    """

    def __init__(self, config: VitsConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 设置配置
        self.config = config
        # 创建词嵌入层，参数分别为词汇表大小、隐藏层大小、填充标记ID
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        # 创建编码器层
        self.encoder = VitsEncoder(config)
        # 创建1维卷积层，参数为隐藏层大小、flow大小乘以2（流大小的两倍）、卷积核大小为1
        self.project = nn.Conv1d(config.hidden_size, config.flow_size * 2, kernel_size=1)

    def get_input_embeddings(self):
        # 返回词嵌入层
        return self.embed_tokens

    def set_input_embeddings(self, value):
        # 设置词嵌入层
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], VitsTextEncoderOutput]:
        # 将输入的词索引转换为词嵌入，并乘以隐藏层大小的平方根
        hidden_states = self.embed_tokens(input_ids) * math.sqrt(self.config.hidden_size)

        # 调用编码器层，传入相关参数，并获取编码器的输出
        encoder_outputs = self.encoder(
            hidden_states=hidden_states,
            padding_mask=padding_mask,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果不返回字典，则直接返回编码器的最后一个隐藏状态
        last_hidden_state = encoder_outputs[0] if not return_dict else encoder_outputs.last_hidden_state

        # 将最后一个隐藏状态经过卷积处理，并乘以填充掩码
        stats = self.project(last_hidden_state.transpose(1, 2)).transpose(1, 2) * padding_mask
        # 将统计数据按flow大小分割成均值和对数方差
        prior_means, prior_log_variances = torch.split(stats, self.config.flow_size, dim=2)

        # 如果不返回字典，则组合输出结果并返回
        if not return_dict:
            outputs = (last_hidden_state, prior_means, prior_log_variances) + encoder_outputs[1:]
            return outputs

        # 返回VitsTextEncoderOutput对象，包含编码器的各种输出
        return VitsTextEncoderOutput(
            last_hidden_state=last_hidden_state,
            prior_means=prior_means,
            prior_log_variances=prior_log_variances,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class VitsPreTrainedModel(PreTrainedModel):
    """
    用于处理权重初始化以及下载和加载预训练模型的抽象类。
    """

    # 配置类为VitsConfig
    config_class = VitsConfig
    # 基础模型前缀为"vits"
    base_model_prefix = "vits"
    # 主输入名称为"input_ids"
    main_input_name = "input_ids"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 初始化神经网络模块的权重
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果是线性层
        if isinstance(module, nn.Linear):
            # 用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置，则初始化为0
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是LayerNorm层
        elif isinstance(module, nn.LayerNorm):
            # 初始化偏置为0
            module.bias.data.zero_()
            # 初始化权重为1
            module.weight.data.fill_(1.0)
        # 如果是一维卷积层
        elif isinstance(module, nn.Conv1d):
            # 使用Kaiming正态初始化权重
            nn.init.kaiming_normal_(module.weight)
            # 如果有偏置，则使用均匀分布初始化
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)
        # 如果是嵌入层
        elif isinstance(module, nn.Embedding):
            # 用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有填充索引，则将填充索引对应的权重初始化为0
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
# 文档字符串，描述了该模型是一个 PyTorch 模块
VITS_START_DOCSTRING = r"""
    # 该模型继承自 `PreTrainedModel`，请查看父类文档以了解通用方法的实现，例如下载或保存、调整输入嵌入大小、剪枝等。

    # 该模型也是 PyTorch 的 `torch.nn.Module` 的一个子类。
    # 可以将其作为常规 PyTorch 模块使用，并参考 PyTorch 文档了解所有与一般使用和行为相关的事项。

    # 参数:
    #     config (`VitsConfig`): 模型配置类，包含模型的所有参数。
    #     使用配置文件初始化不会加载与模型相关的权重，仅加载配置。
    #     请参考 `~PreTrainedModel.from_pretrained` 方法来加载模型权重。
"""

# 文档字符串，描述了模型的输入参数
VITS_INPUTS_DOCSTRING = r"""
    # 参数:
    #     input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
    #         词汇表中输入序列标记的索引。提供时，默认会忽略填充。

    #         可以使用 `AutoTokenizer` 获取索引。参见 `PreTrainedTokenizer.encode` 和
    #         `PreTrainedTokenizer.__call__` 了解详细信息。

    #         [什么是输入ID？](../glossary#input-ids)
    #     attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *可选*):
    #         避免在填充标记索引上执行卷积和注意力的掩码。掩码值选择在 `[0, 1]` 之间：

    #         - `1` 表示 **未被掩码** 的标记，
    #         - `0` 表示 **被掩码** 的标记。

    #         [什么是注意力掩码？](../glossary#attention-mask)
    #     speaker_id (`int`, *可选*):
    #         使用哪个扬声器嵌入。仅用于多扬声器模型。
    #     output_attentions (`bool`, *可选*):
    #         是否返回所有注意力层的注意力张量。参见返回张量下的 `attentions` 以获取更多详细信息。
    #     output_hidden_states (`bool`, *可选*):
    #         是否返回所有层的隐藏状态。参见返回张量下的 `hidden_states` 以获取更多详细信息。
    #     return_dict (`bool`, *可选*):
    #         是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""

# 定义 VitsModel 类，该类继承自 VitsPreTrainedModel，并使用注释中的文本作为类的描述
@add_start_docstrings(
    # 类描述：完整的 VITS 模型，用于文本到语音合成。
    "The complete VITS model, for text-to-speech synthesis.",
    # 使用之前定义的文档字符串作为类的文档
    VITS_START_DOCSTRING,
)
class VitsModel(VitsPreTrainedModel):
    # 初始化方法，接受一个VitsConfig对象作为参数
    def __init__(self, config: VitsConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 将配置信息保存到self.config中
        self.config = config
        # 创建一个VitsTextEncoder对象并保存到self.text_encoder中
        self.text_encoder = VitsTextEncoder(config)
        # 创建一个VitsResidualCouplingBlock对象并保存到self.flow中
        self.flow = VitsResidualCouplingBlock(config)
        # 创建一个VitsHifiGan对象并保存到self.decoder中
        self.decoder = VitsHifiGan(config)

        # 根据配置选择使用VitsStochasticDurationPredictor或者VitsDurationPredictor，并保存到self.duration_predictor中
        if config.use_stochastic_duration_prediction:
            self.duration_predictor = VitsStochasticDurationPredictor(config)
        else:
            self.duration_predictor = VitsDurationPredictor(config)

        # 如果配置中包含多个说话者，则创建一个nn.Embedding对象并保存到embed_speaker中
        if config.num_speakers > 1:
            self.embed_speaker = nn.Embedding(config.num_speakers, config.speaker_embedding_size)

        # 用于训练的后验编码器，保存到self.posterior_encoder中，仅在训练时使用
        self.posterior_encoder = VitsPosteriorEncoder(config)

        # 控制生成语音属性的参数
        self.speaking_rate = config.speaking_rate
        self.noise_scale = config.noise_scale
        self.noise_scale_duration = config.noise_scale_duration

        # 初始化权重并应用最终处理
        self.post_init()

    # 返回text_encoder对象
    def get_encoder(self):
        return self.text_encoder

    # 前向传播方法，接受多个输入参数，并返回VitsModelOutput对象
    @add_start_docstrings_to_model_forward(VITS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=VitsModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        speaker_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.FloatTensor] = None,
```