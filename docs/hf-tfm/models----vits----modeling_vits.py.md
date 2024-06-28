# `.\models\vits\modeling_vits.py`

```py
# coding=utf-8
# Copyright 2023 The Kakao Enterprise Authors and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch VITS model."""

# Import necessary libraries
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn

# Import modules from Hugging Face's library
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

# Get the logger instance for this module
logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "VitsConfig"

# List of pretrained model names for VITS
VITS_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/mms-tts-eng",
    # See all VITS models at https://huggingface.co/models?filter=vits
    # and all MMS models at https://huggingface.co/models?sort=trending&search=facebook%2Fmms-tts
]

# Dataclass representing the output structure of VITS model
@dataclass
class VitsModelOutput(ModelOutput):
    """
    Describes the outputs for the VITS model, with potential hidden states and attentions.
    """
    # 定义输入参数和它们的类型注释，这些参数是模型的输出结果
    Args:
        waveform (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            模型预测的最终音频波形。
        sequence_lengths (`torch.FloatTensor` of shape `(batch_size,)`):
            `waveform` 批次中每个元素的样本长度。
        spectrogram (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_bins)`):
            在流模型输出的对数梅尔频谱图。此频谱图传递给 Hi-Fi GAN 解码器模型以获取最终音频波形。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            包含模型每一层输出的隐藏状态的元组。如果模型具有嵌入层，则还包括初始嵌入输出。
            每个张量的形状为 `(batch_size, sequence_length, hidden_size)`。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            包含注意力权重的元组，每个张量形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
            这些注意力权重经过注意力 softmax 后使用，用于计算自注意力头中的加权平均值。
    """

    # 初始化各个输入参数为 None，用于后续的赋值操作
    waveform: torch.FloatTensor = None
    sequence_lengths: torch.FloatTensor = None
    spectrogram: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
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


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, num_channels):
    """
    Applies a fused operation of addition, tanh, sigmoid, and element-wise multiplication.

    Args:
        input_a (torch.FloatTensor): Input tensor A.
        input_b (torch.FloatTensor): Input tensor B.
        num_channels (int): Number of channels for splitting input tensors.

    Returns:
        torch.FloatTensor: Output tensor after applying the fused operations.
    """
    # Element-wise addition of input tensors A and B
    in_act = input_a + input_b
    # Apply tanh activation to the first `num_channels` channels
    t_act = torch.tanh(in_act[:, :num_channels, :])
    # Apply sigmoid activation to the remaining channels
    s_act = torch.sigmoid(in_act[:, num_channels:, :])
    # Element-wise multiplication of tanh and sigmoid outputs
    acts = t_act * s_act
    return acts


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

    Args:
        inputs (torch.Tensor): Input tensor to be transformed.
        unnormalized_widths (torch.Tensor): Unnormalized widths of the spline segments.
        unnormalized_heights (torch.Tensor): Unnormalized heights of the spline segments.
        unnormalized_derivatives (torch.Tensor): Unnormalized derivatives of the spline segments.
        reverse (bool, optional): If True, applies the transformation in reverse.
        tail_bound (float, optional): Bound beyond which the transform behaves as an identity function.
        min_bin_width (float, optional): Minimum width of each spline bin.
        min_bin_height (float, optional): Minimum height of each spline bin.
        min_derivative (float, optional): Minimum derivative of each spline segment.

    Returns:
        torch.Tensor: Transformed output tensor.
    """
    # Function description continues in the implementation
    # 创建一个布尔掩码，指示哪些输入值在指定的区间内
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    # 通过反转掩码来确定哪些输入值在区间外
    outside_interval_mask = ~inside_interval_mask

    # 初始化输出和对数绝对值行列式的张量，形状与输入相同
    outputs = torch.zeros_like(inputs)
    log_abs_det = torch.zeros_like(inputs)
    # 计算常数值，用于限制分段有理二次函数的行为
    constant = np.log(np.exp(1 - min_derivative) - 1)

    # 在维度上进行填充操作，确保未归一化导数的维度正确
    unnormalized_derivatives = nn.functional.pad(unnormalized_derivatives, pad=(1, 1))
    # 将第一个和最后一个未归一化导数设置为常数值
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant

    # 对区间外的输入值直接赋值为原始输入值
    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    # 对区间外的对数绝对值行列式赋值为零
    log_abs_det[outside_interval_mask] = 0.0
    # 调用 _rational_quadratic_spline 函数计算和更新输出和对数绝对行列式
    outputs[inside_interval_mask], log_abs_det[inside_interval_mask] = _rational_quadratic_spline(
        # 提供在区间内的输入数据
        inputs=inputs[inside_interval_mask],
        # 提供在区间内的未归一化宽度
        unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
        # 提供在区间内的未归一化高度
        unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
        # 提供在区间内的未归一化导数
        unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
        # 指定是否反向处理
        reverse=reverse,
        # 指定尾部边界
        tail_bound=tail_bound,
        # 指定最小箱子宽度
        min_bin_width=min_bin_width,
        # 指定最小箱子高度
        min_bin_height=min_bin_height,
        # 指定最小导数
        min_derivative=min_derivative,
    )
    # 返回更新后的输出和对数绝对行列式
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
        outputs (`torch.FloatTensor` of shape `(batch_size, channels, seq_len)`):
            Hidden-states as transformed by the piecewise rational quadratic function.
        log_abs_det (`torch.FloatTensor` of shape `(batch_size, channels, seq_len)`:
            Logarithm of the absolute value of the determinants corresponding to the `outputs`.
    """
    # 设置上界和下界为尾部限制
    upper_bound = tail_bound
    lower_bound = -tail_bound

    # 检查输入是否在定义域内
    if torch.min(inputs) < lower_bound or torch.max(inputs) > upper_bound:
        raise ValueError("Input to a transform is not within its domain")

    # 获取宽度维度的数量
    num_bins = unnormalized_widths.shape[-1]

    # 检查最小的 bin 宽度是否过大
    if min_bin_width * num_bins > 1.0:
        raise ValueError(f"Minimal bin width {min_bin_width} too large for the number of bins {num_bins}")
    # 检查最小柱高乘以柱子数量是否大于1.0，如果是则抛出值错误异常
    if min_bin_height * num_bins > 1.0:
        raise ValueError(f"Minimal bin height {min_bin_height} too large for the number of bins {num_bins}")

    # 使用 softmax 函数对未归一化的宽度进行归一化处理
    widths = nn.functional.softmax(unnormalized_widths, dim=-1)
    # 根据公式计算每个柱子的宽度
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    # 计算累积宽度并进行填充，确保第一个元素为 0.0
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = nn.functional.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    # 将累积宽度映射到指定的上下界
    cumwidths = (upper_bound - lower_bound) * cumwidths + lower_bound
    cumwidths[..., 0] = lower_bound  # 设置第一个元素为下界
    cumwidths[..., -1] = upper_bound  # 设置最后一个元素为上界
    # 计算每个柱子的实际宽度
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    # 计算导数，使用 softplus 函数对未归一化的导数进行处理
    derivatives = min_derivative + nn.functional.softplus(unnormalized_derivatives)

    # 使用 softmax 函数对未归一化的高度进行归一化处理
    heights = nn.functional.softmax(unnormalized_heights, dim=-1)
    # 根据公式计算每个柱子的高度
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    # 计算累积高度并进行填充，确保第一个元素为 0.0
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = nn.functional.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    # 将累积高度映射到指定的上下界
    cumheights = (upper_bound - lower_bound) * cumheights + lower_bound
    cumheights[..., 0] = lower_bound  # 设置第一个元素为下界
    cumheights[..., -1] = upper_bound  # 设置最后一个元素为上界
    # 计算每个柱子的实际高度
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    # 根据 reverse 参数选择要使用的柱子位置
    bin_locations = cumheights if reverse else cumwidths
    # 在最后一个位置加上微小的偏移量，以防止除以零的情况
    bin_locations[..., -1] += 1e-6
    # 根据输入的值确定每个输入点所属的柱子索引
    bin_idx = torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1
    bin_idx = bin_idx[..., None]

    # 获取每个输入点所在柱子的累积宽度和宽度
    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    # 获取每个输入点所在柱子的累积高度和高度
    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    # 计算每个柱子的斜率
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    # 获取每个输入点所在柱子的导数和导数加一
    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    # 获取每个输入点所在柱子的高度
    input_heights = heights.gather(-1, bin_idx)[..., 0]

    # 计算中间变量1
    intermediate1 = input_derivatives + input_derivatives_plus_one - 2 * input_delta
    # 如果不是反向操作，根据给定的公式计算 theta 值
    if not reverse:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        # 计算输出值
        numerator = input_heights * (input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + intermediate1 * theta_one_minus_theta
        outputs = input_cumheights + numerator / denominator

        # 计算对数绝对值行列式的值
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2)
        )
        log_abs_det = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, log_abs_det
    # 如果输入不符合特定条件，则执行以下代码块
    else:
        # 计算二次方程的根
        intermediate2 = inputs - input_cumheights
        # 计算中间变量3，即 intermediate2 乘以 intermediate1
        intermediate3 = intermediate2 * intermediate1
        # 计算二次方程的系数 a
        a = input_heights * (input_delta - input_derivatives) + intermediate3
        # 计算二次方程的系数 b
        b = input_heights * input_derivatives - intermediate3
        # 计算二次方程的常数项 c
        c = -input_delta * intermediate2

        # 计算判别式
        discriminant = b.pow(2) - 4 * a * c
        # 如果判别式有任何值小于零，抛出运行时错误
        if not (discriminant >= 0).all():
            raise RuntimeError(f"invalid discriminant {discriminant}")

        # 计算二次方程的一个根
        root = (2 * c) / (-b - torch.sqrt(discriminant))
        # 计算输出值
        outputs = root * input_bin_widths + input_cumwidths

        # 计算 theta * (1 - theta)
        theta_one_minus_theta = root * (1 - root)
        # 计算分母
        denominator = input_delta + intermediate1 * theta_one_minus_theta
        # 计算导数的分子
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        # 计算对数绝对值行列式的值
        log_abs_det = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        # 返回计算结果：输出值和对数绝对值行列式的负值
        return outputs, -log_abs_det
# 定义一个名为 VitsWaveNet 的神经网络模型类，继承自 torch.nn.Module
class VitsWaveNet(torch.nn.Module):
    # 初始化方法，接受两个参数：配置对象 config 和层数 num_layers
    def __init__(self, config: VitsConfig, num_layers: int):
        # 调用父类的初始化方法
        super().__init__()
        # 设置隐藏层大小为 config 中的 hidden_size
        self.hidden_size = config.hidden_size
        # 设置网络层数为传入的 num_layers
        self.num_layers = num_layers

        # 初始化输入层和残差跳跃连接层为 ModuleList，用于存储网络的卷积层
        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        # 使用 config 中的 wavenet_dropout 设置一个 Dropout 层
        self.dropout = nn.Dropout(config.wavenet_dropout)

        # 根据是否存在 nn.utils.parametrizations.weight_norm 决定 weight_norm 函数的赋值
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm
        else:
            weight_norm = nn.utils.weight_norm

        # 如果 config 中的 speaker_embedding_size 不为 0，则创建一个 Conv1d 来处理说话者嵌入
        if config.speaker_embedding_size != 0:
            cond_layer = torch.nn.Conv1d(config.speaker_embedding_size, 2 * config.hidden_size * num_layers, 1)
            # 将 cond_layer 应用 weight_norm
            self.cond_layer = weight_norm(cond_layer, name="weight")

        # 循环创建 num_layers 层的卷积层
        for i in range(num_layers):
            dilation = config.wavenet_dilation_rate**i
            padding = (config.wavenet_kernel_size * dilation - dilation) // 2
            # 创建一个 dilation 卷积层，用于输入数据
            in_layer = torch.nn.Conv1d(
                in_channels=config.hidden_size,
                out_channels=2 * config.hidden_size,
                kernel_size=config.wavenet_kernel_size,
                dilation=dilation,
                padding=padding,
            )
            # 应用 weight_norm 到 in_layer
            in_layer = weight_norm(in_layer, name="weight")
            self.in_layers.append(in_layer)

            # 如果不是最后一层，创建一个残差跳跃连接层
            if i < num_layers - 1:
                res_skip_channels = 2 * config.hidden_size
            else:
                res_skip_channels = config.hidden_size

            # 创建一个 1x1 的卷积层作为残差跳跃连接层
            res_skip_layer = torch.nn.Conv1d(config.hidden_size, res_skip_channels, 1)
            # 应用 weight_norm 到 res_skip_layer
            res_skip_layer = weight_norm(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)
    # 前向传播函数，用于模型的前向计算
    def forward(self, inputs, padding_mask, global_conditioning=None):
        # 初始化输出张量，形状与输入相同
        outputs = torch.zeros_like(inputs)
        # 创建一个张量，包含隐藏大小的整数值
        num_channels_tensor = torch.IntTensor([self.hidden_size])

        # 如果存在全局条件，则通过条件层处理全局条件数据
        if global_conditioning is not None:
            global_conditioning = self.cond_layer(global_conditioning)

        # 遍历每一层
        for i in range(self.num_layers):
            # 使用第i层输入层处理输入数据
            hidden_states = self.in_layers[i](inputs)

            # 如果存在全局条件，则从全局条件中选择对应层的状态
            if global_conditioning is not None:
                cond_offset = i * 2 * self.hidden_size
                global_states = global_conditioning[:, cond_offset : cond_offset + 2 * self.hidden_size, :]
            else:
                # 否则初始化全局状态为与隐藏状态形状相同的零张量
                global_states = torch.zeros_like(hidden_states)

            # 调用融合操作函数，计算激活函数的输出
            acts = fused_add_tanh_sigmoid_multiply(hidden_states, global_states, num_channels_tensor[0])
            # 对激活输出进行dropout处理
            acts = self.dropout(acts)

            # 使用残差连接和跳跃连接层处理激活输出
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.num_layers - 1:
                # 如果不是最后一层，则进行残差连接
                res_acts = res_skip_acts[:, : self.hidden_size, :]
                inputs = (inputs + res_acts) * padding_mask
                outputs = outputs + res_skip_acts[:, self.hidden_size :, :]
            else:
                # 如果是最后一层，则仅将输出增加残差跳跃连接层的输出
                outputs = outputs + res_skip_acts

        # 最后将输出乘以填充掩码，返回最终的输出张量
        return outputs * padding_mask

    # 移除所有权重归一化操作
    def remove_weight_norm(self):
        # 如果存在说话者嵌入大小，则移除条件层的权重归一化
        if self.speaker_embedding_size != 0:
            torch.nn.utils.remove_weight_norm(self.cond_layer)
        # 分别对每一层输入层和残差跳跃连接层移除权重归一化
        for layer in self.in_layers:
            torch.nn.utils.remove_weight_norm(layer)
        for layer in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(layer)
# 定义一个名为 VitsPosteriorEncoder 的神经网络模块
class VitsPosteriorEncoder(nn.Module):
    # 初始化函数，接受一个 VitsConfig 类型的参数 config
    def __init__(self, config: VitsConfig):
        super().__init__()
        # 设置输出通道数为 config.flow_size
        self.out_channels = config.flow_size

        # 使用 1 维卷积定义 conv_pre 层，输入通道数为 config.spectrogram_bins，输出通道数为 config.hidden_size，卷积核大小为 1
        self.conv_pre = nn.Conv1d(config.spectrogram_bins, config.hidden_size, 1)
        # 初始化一个 VitsWaveNet 类型的模型 wavenet，传入参数 config 和 posterior_encoder_num_wavenet_layers
        self.wavenet = VitsWaveNet(config, num_layers=config.posterior_encoder_num_wavenet_layers)
        # 使用 1 维卷积定义 conv_proj 层，输入通道数为 config.hidden_size，输出通道数为 self.out_channels * 2，卷积核大小为 1
        self.conv_proj = nn.Conv1d(config.hidden_size, self.out_channels * 2, 1)

    # 前向传播函数，接受 inputs（输入数据）、padding_mask（填充掩码）、global_conditioning（全局条件）作为参数
    def forward(self, inputs, padding_mask, global_conditioning=None):
        # 对输入数据 inputs 应用 conv_pre 层和 padding_mask，然后将结果赋值给 inputs
        inputs = self.conv_pre(inputs) * padding_mask
        # 将处理后的 inputs 输入到 wavenet 模型中进行处理，同时传入 padding_mask 和 global_conditioning
        inputs = self.wavenet(inputs, padding_mask, global_conditioning)
        # 对处理后的结果应用 conv_proj 层和 padding_mask，然后将结果赋值给 stats
        stats = self.conv_proj(inputs) * padding_mask
        # 将 stats 按照第二个维度（通道维度）拆分为均值 mean 和对数标准差 log_stddev
        mean, log_stddev = torch.split(stats, self.out_channels, dim=1)
        # 使用均值 mean 和随机生成的正态分布数据（标准差为 exp(log_stddev)）生成采样数据，并应用 padding_mask
        sampled = (mean + torch.randn_like(mean) * torch.exp(log_stddev)) * padding_mask
        # 返回采样数据 sampled、均值 mean 和对数标准差 log_stddev
        return sampled, mean, log_stddev


# 从 transformers.models.speecht5.modeling_speecht5.HifiGanResidualBlock 复制而来的类
class HifiGanResidualBlock(nn.Module):
    # 初始化函数，接受 channels（通道数）、kernel_size（卷积核大小，默认为 3）、dilation（膨胀率元组，默认为 (1, 3, 5)）、leaky_relu_slope（LeakyReLU 斜率，默认为 0.1）作为参数
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), leaky_relu_slope=0.1):
        super().__init__()
        # 设置 LeakyReLU 的斜率
        self.leaky_relu_slope = leaky_relu_slope

        # 创建多个 1 维卷积层的列表 convs1，每个卷积层的输入输出通道数相同，采用不同的膨胀率
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
        # 创建多个 1 维卷积层的列表 convs2，每个卷积层的输入输出通道数相同，都采用膨胀率为 1
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

    # 获取给定卷积核大小和膨胀率的填充数
    def get_padding(self, kernel_size, dilation=1):
        return (kernel_size * dilation - dilation) // 2

    # 对 convs1 和 convs2 中的卷积层应用权重归一化
    def apply_weight_norm(self):
        for layer in self.convs1:
            nn.utils.weight_norm(layer)
        for layer in self.convs2:
            nn.utils.weight_norm(layer)

    # 移除 convs1 和 convs2 中的卷积层的权重归一化
    def remove_weight_norm(self):
        for layer in self.convs1:
            nn.utils.remove_weight_norm(layer)
        for layer in self.convs2:
            nn.utils.remove_weight_norm(layer)

    # 前向传播函数，接受 hidden_states（隐藏状态）作为输入
    def forward(self, hidden_states):
        # 遍历 convs1 和 convs2 中的每一对卷积层
        for conv1, conv2 in zip(self.convs1, self.convs2):
            # 将隐藏状态作为残差项保存
            residual = hidden_states
            # 应用 LeakyReLU 激活函数
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            # 应用 conv1 卷积层
            hidden_states = conv1(hidden_states)
            # 再次应用 LeakyReLU 激活函数
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            # 应用 conv2 卷积层
            hidden_states = conv2(hidden_states)
            # 将残差项加到输出上，形成残差连接
            hidden_states = hidden_states + residual
        # 返回最终的隐藏状态
        return hidden_states
    # 初始化函数，接受一个VitsConfig类型的配置对象作为参数
    def __init__(self, config: VitsConfig):
        super().__init__()  # 调用父类的初始化函数

        # 将配置对象保存在实例变量中
        self.config = config

        # 计算残差块卷积核数量和上采样率的数量
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)

        # 创建一个1维卷积层，作为初始卷积层
        self.conv_pre = nn.Conv1d(
            config.flow_size,
            config.upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        # 创建一个空的模块列表，用于存放上采样的卷积层
        self.upsampler = nn.ModuleList()
        for i, (upsample_rate, kernel_size) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            # 每次迭代，向模块列表中添加一个转置卷积层
            self.upsampler.append(
                nn.ConvTranspose1d(
                    config.upsample_initial_channel // (2**i),
                    config.upsample_initial_channel // (2 ** (i + 1)),
                    kernel_size=kernel_size,
                    stride=upsample_rate,
                    padding=(kernel_size - upsample_rate) // 2,
                )
            )

        # 创建一个空的模块列表，用于存放残差块
        self.resblocks = nn.ModuleList()
        for i in range(len(self.upsampler)):
            channels = config.upsample_initial_channel // (2 ** (i + 1))
            for kernel_size, dilation in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
                # 每次迭代，向模块列表中添加一个残差块
                self.resblocks.append(HifiGanResidualBlock(channels, kernel_size, dilation, config.leaky_relu_slope))

        # 创建一个1维卷积层，作为后处理卷积层
        self.conv_post = nn.Conv1d(channels, 1, kernel_size=7, stride=1, padding=3, bias=False)

        # 如果配置中指定了说话人嵌入的大小，则创建一个条件卷积层
        if config.speaker_embedding_size != 0:
            self.cond = nn.Conv1d(config.speaker_embedding_size, config.upsample_initial_channel, 1)

    # 对模型中的上采样层应用权重归一化
    def apply_weight_norm(self):
        for layer in self.upsampler:
            nn.utils.weight_norm(layer)
        for layer in self.resblocks:
            layer.apply_weight_norm()

    # 移除模型中的上采样层的权重归一化
    def remove_weight_norm(self):
        for layer in self.upsampler:
            nn.utils.remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()

    # 前向传播函数，接受频谱图和全局条件作为输入
    def forward(
        self, spectrogram: torch.FloatTensor, global_conditioning: Optional[torch.FloatTensor] = None
    ):
        # 省略函数体，由具体的前向传播逻辑组成
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
        # 将输入的频谱图通过预处理卷积层转换
        hidden_states = self.conv_pre(spectrogram)

        # 如果提供了全局条件信息，则通过条件模块进行调节
        if global_conditioning is not None:
            hidden_states = hidden_states + self.cond(global_conditioning)

        # 多次进行上采样操作，使用LeakyReLU作为激活函数
        for i in range(self.num_upsamples):
            hidden_states = nn.functional.leaky_relu(hidden_states, self.config.leaky_relu_slope)
            hidden_states = self.upsampler[i](hidden_states)

            # 应用残差块以保留重要信息并减少训练中的梯度消失问题
            res_state = self.resblocks[i * self.num_kernels](hidden_states)
            for j in range(1, self.num_kernels):
                res_state += self.resblocks[i * self.num_kernels + j](hidden_states)
            hidden_states = res_state / self.num_kernels

        # 最终通过LeakyReLU和后处理卷积层处理得到最终的波形数据
        hidden_states = nn.functional.leaky_relu(hidden_states)
        hidden_states = self.conv_post(hidden_states)
        waveform = torch.tanh(hidden_states)
        return waveform
class VitsResidualCouplingLayer(nn.Module):
    def __init__(self, config: VitsConfig):
        super().__init__()
        # 计算每半通道数，用于定义不同层次的卷积大小
        self.half_channels = config.flow_size // 2

        # 前处理卷积层，将半通道数转换为隐藏层大小
        self.conv_pre = nn.Conv1d(self.half_channels, config.hidden_size, 1)
        # WaveNet 模型，使用给定的配置和层数
        self.wavenet = VitsWaveNet(config, num_layers=config.prior_encoder_num_wavenet_layers)
        # 后处理卷积层，将隐藏层大小转换回半通道数
        self.conv_post = nn.Conv1d(config.hidden_size, self.half_channels, 1)

    def forward(self, inputs, padding_mask, global_conditioning=None, reverse=False):
        # 将输入张量拆分为两半，分别处理
        first_half, second_half = torch.split(inputs, [self.half_channels] * 2, dim=1)
        # 使用前处理卷积层处理第一半数据，同时考虑填充掩码
        hidden_states = self.conv_pre(first_half) * padding_mask
        # 将处理后的数据输入 WaveNet 模型，考虑填充掩码和全局条件
        hidden_states = self.wavenet(hidden_states, padding_mask, global_conditioning)
        # 使用后处理卷积层处理 WaveNet 输出，同时考虑填充掩码
        mean = self.conv_post(hidden_states) * padding_mask
        # 初始化对数标准差为零张量
        log_stddev = torch.zeros_like(mean)

        if not reverse:
            # 如果不是反向模式，则执行如下操作
            # 计算第二半数据的均值修正，并考虑填充掩码和对数标准差
            second_half = mean + second_half * torch.exp(log_stddev) * padding_mask
            # 将修正后的数据拼接起来作为输出
            outputs = torch.cat([first_half, second_half], dim=1)
            # 计算对数行列式，以便可逆层的反向传播使用
            log_determinant = torch.sum(log_stddev, [1, 2])
            return outputs, log_determinant
        else:
            # 如果是反向模式，则执行如下操作
            # 计算第二半数据的均值修正反向，并考虑填充掩码和对数标准差
            second_half = (second_half - mean) * torch.exp(-log_stddev) * padding_mask
            # 将修正后的数据拼接起来作为输出
            outputs = torch.cat([first_half, second_half], dim=1)
            return outputs, None


class VitsResidualCouplingBlock(nn.Module):
    def __init__(self, config: VitsConfig):
        super().__init__()
        # 创建多个 VitsResidualCouplingLayer 层作为流
        self.flows = nn.ModuleList()
        for _ in range(config.prior_encoder_num_flows):
            self.flows.append(VitsResidualCouplingLayer(config))

    def forward(self, inputs, padding_mask, global_conditioning=None, reverse=False):
        if not reverse:
            # 如果不是反向模式，则对每个流执行正向操作
            for flow in self.flows:
                inputs, _ = flow(inputs, padding_mask, global_conditioning)
                inputs = torch.flip(inputs, [1])  # 翻转张量维度1（时间维度）
        else:
            # 如果是反向模式，则对每个流执行反向操作
            for flow in reversed(self.flows):
                inputs = torch.flip(inputs, [1])  # 翻转张量维度1（时间维度）
                inputs, _ = flow(inputs, padding_mask, global_conditioning, reverse=True)
        return inputs
    # 初始化方法，接受一个VitsConfig类型的配置和一个可选的dropout_rate参数
    def __init__(self, config: VitsConfig, dropout_rate=0.0):
        # 调用父类的初始化方法
        super().__init__()
        # 获取配置中的参数并赋值给本地变量
        kernel_size = config.duration_predictor_kernel_size
        channels = config.hidden_size
        self.num_layers = config.depth_separable_num_layers

        # 创建一个Dropout层，用于随机丢弃输入数据中的部分神经元
        self.dropout = nn.Dropout(dropout_rate)
        # 初始化一个ModuleList用于存放深度可分离卷积层
        self.convs_dilated = nn.ModuleList()
        # 初始化一个ModuleList用于存放逐点卷积层
        self.convs_pointwise = nn.ModuleList()
        # 初始化一个ModuleList用于存放LayerNorm层1
        self.norms_1 = nn.ModuleList()
        # 初始化一个ModuleList用于存放LayerNorm层2
        self.norms_2 = nn.ModuleList()
        
        # 循环创建num_layers个深度可分离卷积、逐点卷积、LayerNorm层1和LayerNorm层2
        for i in range(self.num_layers):
            # 计算当前层的膨胀系数和填充数
            dilation = kernel_size**i
            padding = (kernel_size * dilation - dilation) // 2
            # 添加一个深度可分离卷积层到ModuleList中
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
            # 添加一个逐点卷积层到ModuleList中
            self.convs_pointwise.append(nn.Conv1d(channels, channels, 1))
            # 添加一个LayerNorm层1到ModuleList中
            self.norms_1.append(nn.LayerNorm(channels))
            # 添加一个LayerNorm层2到ModuleList中
            self.norms_2.append(nn.LayerNorm(channels))

    # 前向传播方法，接受输入数据、填充遮罩和全局条件（可选），返回处理后的数据
    def forward(self, inputs, padding_mask, global_conditioning=None):
        # 如果有全局条件，则将输入数据和全局条件相加
        if global_conditioning is not None:
            inputs = inputs + global_conditioning

        # 循环进行num_layers次操作
        for i in range(self.num_layers):
            # 应用深度可分离卷积层到输入数据，并乘以填充遮罩
            hidden_states = self.convs_dilated[i](inputs * padding_mask)
            # 对卷积后的隐藏状态应用LayerNorm层1
            hidden_states = self.norms_1[i](hidden_states.transpose(1, -1)).transpose(1, -1)
            # 应用GELU激活函数
            hidden_states = nn.functional.gelu(hidden_states)
            # 应用逐点卷积层到激活后的隐藏状态
            hidden_states = self.convs_pointwise[i](hidden_states)
            # 对逐点卷积后的隐藏状态应用LayerNorm层2
            hidden_states = self.norms_2[i](hidden_states.transpose(1, -1)).transpose(1, -1)
            # 应用GELU激活函数
            hidden_states = nn.functional.gelu(hidden_states)
            # 应用Dropout层到隐藏状态
            hidden_states = self.dropout(hidden_states)
            # 将输入数据和处理后的隐藏状态相加，得到下一层的输入数据
            inputs = inputs + hidden_states

        # 返回处理后的输入数据乘以填充遮罩
        return inputs * padding_mask
# 定义一个名为 VitsConvFlow 的自定义神经网络模块，继承自 nn.Module 类
class VitsConvFlow(nn.Module):
    # 初始化函数，接受一个 VitsConfig 类型的配置对象作为参数
    def __init__(self, config: VitsConfig):
        super().__init__()
        # 设置卷积层的输出通道数为隐藏大小
        self.filter_channels = config.hidden_size
        # 设置卷积深度可分离卷积的通道数为隐藏大小的一半
        self.half_channels = config.depth_separable_channels // 2
        # 设置持续时间预测流的分箱数
        self.num_bins = config.duration_predictor_flow_bins
        # 设置持续时间预测的尾部边界
        self.tail_bound = config.duration_predictor_tail_bound

        # 定义预卷积层，输入通道数为半通道数，输出通道数为过滤器通道数
        self.conv_pre = nn.Conv1d(self.half_channels, self.filter_channels, 1)
        # 定义扩展的深度可分离卷积层
        self.conv_dds = VitsDilatedDepthSeparableConv(config)
        # 定义投影卷积层，输入通道数为过滤器通道数，输出通道数为半通道数乘以（分箱数乘以3再减1）
        self.conv_proj = nn.Conv1d(self.filter_channels, self.half_channels * (self.num_bins * 3 - 1), 1)

    # 前向传播函数，接受输入、填充掩码、全局条件（可选）、是否反向（可选）作为参数
    def forward(self, inputs, padding_mask, global_conditioning=None, reverse=False):
        # 将输入张量按通道数的一半分割成两部分
        first_half, second_half = torch.split(inputs, [self.half_channels] * 2, dim=1)

        # 对第一部分进行预卷积操作
        hidden_states = self.conv_pre(first_half)
        # 对预卷积结果进行深度可分离卷积操作
        hidden_states = self.conv_dds(hidden_states, padding_mask, global_conditioning)
        # 对深度可分离卷积结果进行投影卷积，并乘以填充掩码
        hidden_states = self.conv_proj(hidden_states) * padding_mask

        # 获取批次大小、通道数和长度
        batch_size, channels, length = first_half.shape
        # 重塑隐藏状态张量的形状，并对维度进行置换
        hidden_states = hidden_states.reshape(batch_size, channels, -1, length).permute(0, 1, 3, 2)

        # 提取未归一化的宽度、高度和导数
        unnormalized_widths = hidden_states[..., : self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_heights = hidden_states[..., self.num_bins : 2 * self.num_bins] / math.sqrt(self.filter_channels)
        unnormalized_derivatives = hidden_states[..., 2 * self.num_bins :]

        # 使用非约束有理二次样条函数对第二部分进行变换，并返回变换后的结果和对数绝对值行列式
        second_half, log_abs_det = _unconstrained_rational_quadratic_spline(
            second_half,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            reverse=reverse,
            tail_bound=self.tail_bound,
        )

        # 将第一部分和变换后的第二部分连接起来，并乘以填充掩码
        outputs = torch.cat([first_half, second_half], dim=1) * padding_mask
        # 如果不是反向传播，则计算对数行列式，并返回结果和对数行列式
        if not reverse:
            log_determinant = torch.sum(log_abs_det * padding_mask, [1, 2])
            return outputs, log_determinant
        # 如果是反向传播，则只返回结果
        else:
            return outputs, None


# 定义一个名为 VitsElementwiseAffine 的自定义神经网络模块，继承自 nn.Module 类
class VitsElementwiseAffine(nn.Module):
    # 初始化函数，接受一个 VitsConfig 类型的配置对象作为参数
    def __init__(self, config: VitsConfig):
        super().__init__()
        # 设置通道数为深度可分离卷积的通道数
        self.channels = config.depth_separable_channels
        # 定义平移参数和对数尺度参数作为可训练参数
        self.translate = nn.Parameter(torch.zeros(self.channels, 1))
        self.log_scale = nn.Parameter(torch.zeros(self.channels, 1))

    # 前向传播函数，接受输入、填充掩码、全局条件（可选）、是否反向（可选）作为参数
    def forward(self, inputs, padding_mask, global_conditioning=None, reverse=False):
        # 如果不是反向传播，则计算输出并乘以填充掩码
        if not reverse:
            outputs = self.translate + torch.exp(self.log_scale) * inputs
            outputs = outputs * padding_mask
            log_determinant = torch.sum(self.log_scale * padding_mask, [1, 2])
            return outputs, log_determinant
        # 如果是反向传播，则计算输出并返回结果
        else:
            outputs = (inputs - self.translate) * torch.exp(-self.log_scale) * padding_mask
            return outputs, None
    # 初始化函数，用于初始化对象
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 从配置中获取说话人嵌入大小作为嵌入维度
        embed_dim = config.speaker_embedding_size
        # 从配置中获取隐藏层大小作为卷积滤波器的通道数
        filter_channels = config.hidden_size

        # 定义预处理卷积层，输入和输出通道数都为 filter_channels，卷积核大小为 1
        self.conv_pre = nn.Conv1d(filter_channels, filter_channels, 1)
        # 定义投影卷积层，输入和输出通道数都为 filter_channels，卷积核大小为 1
        self.conv_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        
        # 创建 VitsDilatedDepthSeparableConv 模块，配置中包括 dropout 率
        self.conv_dds = VitsDilatedDepthSeparableConv(
            config,
            dropout_rate=config.duration_predictor_dropout,
        )

        # 如果嵌入维度不为 0，则定义条件卷积层，将嵌入维度映射到 filter_channels
        if embed_dim != 0:
            self.cond = nn.Conv1d(embed_dim, filter_channels, 1)

        # 创建流模块列表，第一个元素是 VitsElementwiseAffine 模块
        self.flows = nn.ModuleList()
        self.flows.append(VitsElementwiseAffine(config))
        
        # 根据配置循环创建多个 VitsConvFlow 模块，用于流模块列表
        for _ in range(config.duration_predictor_num_flows):
            self.flows.append(VitsConvFlow(config))

        # 定义后处理的预处理卷积层，输入通道数为 1，输出通道数为 filter_channels，卷积核大小为 1
        self.post_conv_pre = nn.Conv1d(1, filter_channels, 1)
        # 定义后处理的投影卷积层，输入和输出通道数都为 filter_channels，卷积核大小为 1
        self.post_conv_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        
        # 创建后处理的 VitsDilatedDepthSeparableConv 模块，配置中包括 dropout 率
        self.post_conv_dds = VitsDilatedDepthSeparableConv(
            config,
            dropout_rate=config.duration_predictor_dropout,
        )

        # 创建后处理流模块列表，第一个元素是 VitsElementwiseAffine 模块
        self.post_flows = nn.ModuleList()
        self.post_flows.append(VitsElementwiseAffine(config))
        
        # 根据配置循环创建多个 VitsConvFlow 模块，用于后处理流模块列表
        for _ in range(config.duration_predictor_num_flows):
            self.post_flows.append(VitsConvFlow(config))
class VitsDurationPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 从配置中获取预测器的参数
        kernel_size = config.duration_predictor_kernel_size
        filter_channels = config.duration_predictor_filter_channels

        # 定义模型的各个层和模块
        self.dropout = nn.Dropout(config.duration_predictor_dropout)
        # 第一个卷积层，用于特征提取
        self.conv_1 = nn.Conv1d(config.hidden_size, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = nn.LayerNorm(filter_channels, eps=config.layer_norm_eps)
        # 第二个卷积层，用于进一步提取特征
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = nn.LayerNorm(filter_channels, eps=config.layer_norm_eps)
        # 最后的投影层，用于预测持续时间
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        # 如果有说话者嵌入的大小，则定义条件卷积层
        if config.speaker_embedding_size != 0:
            self.cond = nn.Conv1d(config.speaker_embedding_size, config.hidden_size, 1)

    def forward(self, inputs, padding_mask, global_conditioning=None):
        # 对输入进行离散化处理，防止梯度回传到条件信息
        inputs = torch.detach(inputs)

        # 如果有全局条件信息，则将其加入到输入中
        if global_conditioning is not None:
            global_conditioning = torch.detach(global_conditioning)
            inputs = inputs + self.cond(global_conditioning)

        # 第一层卷积，激活函数，层归一化和 dropout
        inputs = self.conv_1(inputs * padding_mask)
        inputs = torch.relu(inputs)
        inputs = self.norm_1(inputs.transpose(1, -1)).transpose(1, -1)
        inputs = self.dropout(inputs)

        # 第二层卷积，激活函数，层归一化和 dropout
        inputs = self.conv_2(inputs * padding_mask)
        inputs = torch.relu(inputs)
        inputs = self.norm_2(inputs.transpose(1, -1)).transpose(1, -1)
        inputs = self.dropout(inputs)

        # 最终的投影层，用于生成持续时间预测
        inputs = self.proj(inputs * padding_mask)
        return inputs * padding_mask


class VitsAttention(nn.Module):
    """Multi-headed attention with relative positional representation."""

    def __init__(self, config: VitsConfig):
        super().__init__()
        # 从配置中获取注意力机制的参数
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.dropout = config.attention_dropout
        self.window_size = config.window_size

        # 计算每个头的维度
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5

        # 检查隐藏层维度是否可以被头数整除
        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"hidden_size must be divisible by num_attention_heads (got `hidden_size`: {self.embed_dim}"
                f" and `num_attention_heads`: {self.num_heads})."
            )

        # 定义键、值、查询和输出的线性投影层
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_bias)

        # 如果定义了窗口大小，则使用相对位置表示
        if self.window_size:
            self.emb_rel_k = nn.Parameter(torch.randn(1, self.window_size * 2 + 1, self.head_dim) * self.scaling)
            self.emb_rel_v = nn.Parameter(torch.randn(1, self.window_size * 2 + 1, self.head_dim) * self.scaling)
    # 将输入张量重新形状为指定的形状，用于多头注意力机制中
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 实现 Transformer 模型的前向传播
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # 这里是前向传播函数，接收多个输入参数并进行计算，返回输出结果

    # 获取相对位置嵌入的方法
    def _get_relative_embeddings(self, relative_embeddings, length):
        pad_length = max(length - (self.window_size + 1), 0)
        if pad_length > 0:
            # 在相对嵌入张量的长度维度上进行填充
            relative_embeddings = nn.functional.pad(relative_embeddings, [0, 0, pad_length, pad_length, 0, 0])

        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        # 切片获取相对位置嵌入的部分
        return relative_embeddings[:, slice_start_position:slice_end_position]

    # 将相对位置转换为绝对位置
    def _relative_position_to_absolute_position(self, x):
        batch_heads, length, _ = x.size()

        # 在最后一列上进行填充，以进行相对索引到绝对索引的转换
        x = nn.functional.pad(x, [0, 1, 0, 0, 0, 0])

        # 扩展额外元素以匹配形状 (len+1, 2*len-1)
        x_flat = x.view([batch_heads, length * 2 * length])
        x_flat = nn.functional.pad(x_flat, [0, length - 1, 0, 0])

        # 重塑并切片去除填充元素
        x_final = x_flat.view([batch_heads, length + 1, 2 * length - 1])
        x_final = x_final[:, :length, length - 1 :]
        return x_final

    # 将绝对位置转换为相对位置
    def _absolute_position_to_relative_position(self, x):
        batch_heads, length, _ = x.size()

        # 沿着列维度进行填充
        x = nn.functional.pad(x, [0, length - 1, 0, 0, 0, 0])
        x_flat = x.view([batch_heads, length * (2 * length - 1)])

        # 在重塑后的元素前面添加 0，以平移元素位置
        x_flat = nn.functional.pad(x_flat, [length, 0, 0, 0])
        x_final = x_flat.view([batch_heads, length, 2 * length])[:, :, 1:]
        return x_final
# 定义 VitsFeedForward 类，继承自 nn.Module，用于实现 Vits 模型的前馈网络部分
class VitsFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 第一个卷积层，输入通道数为 config.hidden_size，输出通道数为 config.ffn_dim，卷积核大小为 config.ffn_kernel_size
        self.conv_1 = nn.Conv1d(config.hidden_size, config.ffn_dim, config.ffn_kernel_size)
        # 第二个卷积层，输入通道数为 config.ffn_dim，输出通道数为 config.hidden_size，卷积核大小为 config.ffn_kernel_size
        self.conv_2 = nn.Conv1d(config.ffn_dim, config.hidden_size, config.ffn_kernel_size)
        # Dropout 层，用于随机失活，参数为 config.activation_dropout
        self.dropout = nn.Dropout(config.activation_dropout)

        # 根据配置文件中的 hidden_act 参数确定激活函数
        if isinstance(config.hidden_act, str):
            self.act_fn = ACT2FN[config.hidden_act]
        else:
            self.act_fn = config.hidden_act

        # 如果卷积核大小大于 1，则设置填充值以保证卷积操作不改变张量的维度
        if config.ffn_kernel_size > 1:
            pad_left = (config.ffn_kernel_size - 1) // 2
            pad_right = config.ffn_kernel_size // 2
            self.padding = [pad_left, pad_right, 0, 0, 0, 0]
        else:
            self.padding = None

    # 前向传播函数，接受 hidden_states 和 padding_mask 作为输入
    def forward(self, hidden_states, padding_mask):
        # 调整 hidden_states 的维度顺序，使得通道维度变为第二维度
        hidden_states = hidden_states.permute(0, 2, 1)
        padding_mask = padding_mask.permute(0, 2, 1)

        # 将 hidden_states 和 padding_mask 进行逐元素乘法
        hidden_states = hidden_states * padding_mask
        
        # 如果有设置填充值，对 hidden_states 进行填充操作
        if self.padding is not None:
            hidden_states = nn.functional.pad(hidden_states, self.padding)

        # 经过第一个卷积层、激活函数、以及 Dropout 层的处理
        hidden_states = self.conv_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # 再次经过逐元素乘法操作
        hidden_states = hidden_states * padding_mask

        # 如果有设置填充值，再次对 hidden_states 进行填充操作
        if self.padding is not None:
            hidden_states = nn.functional.pad(hidden_states, self.padding)

        # 经过第二个卷积层的处理
        hidden_states = self.conv_2(hidden_states)

        # 再次经过逐元素乘法操作
        hidden_states = hidden_states * padding_mask

        # 调整 hidden_states 的维度顺序，使得通道维度恢复到最后一维
        hidden_states = hidden_states.permute(0, 2, 1)
        
        # 返回处理后的 hidden_states
        return hidden_states


# 定义 VitsEncoderLayer 类，继承自 nn.Module，用于实现 Vits 模型的编码器层
class VitsEncoderLayer(nn.Module):
    def __init__(self, config: VitsConfig):
        super().__init__()
        # 自注意力机制层
        self.attention = VitsAttention(config)
        # Dropout 层，用于随机失活，参数为 config.hidden_dropout
        self.dropout = nn.Dropout(config.hidden_dropout)
        # LayerNorm 层，用于归一化输入数据
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 前馈网络层
        self.feed_forward = VitsFeedForward(config)
        # 最终归一化层
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # 前向传播函数，接受 hidden_states、padding_mask、attention_mask 和 output_attentions 作为输入
    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # 保存残差连接
        residual = hidden_states
        
        # 自注意力机制层的前向传播，返回处理后的 hidden_states 和注意力权重
        hidden_states, attn_weights = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )

        # 经过 Dropout 层处理
        hidden_states = self.dropout(hidden_states)
        
        # 残差连接和 LayerNorm 层的处理
        hidden_states = self.layer_norm(residual + hidden_states)

        # 保存新的残差连接
        residual = hidden_states
        
        # 前馈网络层的前向传播
        hidden_states = self.feed_forward(hidden_states, padding_mask)
        
        # 经过 Dropout 层处理
        hidden_states = self.dropout(hidden_states)
        
        # 最终归一化层的处理
        hidden_states = self.final_layer_norm(residual + hidden_states)

        # 输出结果保存在 outputs 中
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则将 attn_weights 加入到 outputs 中
        if output_attentions:
            outputs += (attn_weights,)

        # 返回 outputs
        return outputs
    # 初始化函数，用于创建一个新的VitsEncoder对象
    def __init__(self, config: VitsConfig):
        # 调用父类的初始化函数，确保继承父类的属性和方法
        super().__init__()
        # 将传入的配置对象保存在实例变量中，以便在类中的其他方法中使用
        self.config = config
        # 创建一个包含多个VitsEncoderLayer对象的模块列表，数量由配置中的num_hidden_layers指定
        self.layers = nn.ModuleList([VitsEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 设置梯度检查点标志为False，表示不使用梯度检查点功能
        self.gradient_checkpointing = False
        # 设置层丢弃率，从配置中获取
        self.layerdrop = config.layerdrop

    # 前向传播函数，定义了VitsEncoder对象的数据流向
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        padding_mask: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 定义函数的返回类型为元组或 BaseModelOutput 类型
    ) -> Union[Tuple, BaseModelOutput]:
    
        # 如果不输出隐藏状态，则初始化空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果不输出注意力权重，则初始化空元组
        all_self_attentions = () if output_attentions else None
    
        # 扩展 attention_mask 到四维张量
        if attention_mask is not None:
            # 将二维注意力掩码扩展为四维张量 [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)
    
        # 对隐藏状态应用填充掩码
        hidden_states = hidden_states * padding_mask
    
        # 检查是否启用了 DeepSpeed Zero3
        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()
    
        # 遍历所有的编码器层
        for encoder_layer in self.layers:
            # 如果输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
    
            # 添加 LayerDrop（参见 https://arxiv.org/abs/1909.11556 进行描述）
            dropout_probability = np.random.uniform(0, 1)
    
            # 计算是否跳过当前层
            skip_the_layer = self.training and (dropout_probability < self.layerdrop)
            
            # 如果不跳过当前层或者 DeepSpeed Zero3 已启用
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # 如果启用了梯度检查点且在训练模式下，则使用梯度检查点函数计算层输出
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        encoder_layer.__call__,
                        hidden_states,
                        padding_mask,
                        attention_mask,
                        output_attentions,
                    )
                else:
                    # 否则直接调用编码器层计算层输出
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        padding_mask=padding_mask,
                        output_attentions=output_attentions,
                    )
                # 更新隐藏状态为当前层的输出的第一个元素
                hidden_states = layer_outputs[0]
    
            # 如果跳过当前层，则设置层输出为 None
            if skip_the_layer:
                layer_outputs = (None, None)
    
            # 如果输出注意力权重，则将当前层的注意力权重添加到 all_self_attentions 中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
    
        # 对最终的隐藏状态再次应用填充掩码
        hidden_states = hidden_states * padding_mask
    
        # 如果输出隐藏状态，则将最终的隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
    
        # 如果不返回字典形式的结果，则返回所有非 None 的结果元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
    
        # 返回 BaseModelOutput 类型的结果
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
class VitsTextEncoder(nn.Module):
    """
    Transformer encoder that uses relative positional representation instead of absolute positional encoding.
    """

    def __init__(self, config: VitsConfig):
        super().__init__()
        self.config = config
        # 初始化词嵌入层，vocab_size为词汇表大小，hidden_size为隐藏层大小，pad_token_id为填充token的ID
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        # 初始化编码器
        self.encoder = VitsEncoder(config)
        # 项目层，使用1维卷积进行投影
        self.project = nn.Conv1d(config.hidden_size, config.flow_size * 2, kernel_size=1)

    def get_input_embeddings(self):
        # 返回输入的词嵌入层
        return self.embed_tokens

    def set_input_embeddings(self, value):
        # 设置输入的词嵌入层
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
        # 使用词嵌入层乘以sqrt(hidden_size)来得到输入的隐藏状态
        hidden_states = self.embed_tokens(input_ids) * math.sqrt(self.config.hidden_size)

        # 将隐藏状态传入编码器进行编码
        encoder_outputs = self.encoder(
            hidden_states=hidden_states,
            padding_mask=padding_mask,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取编码器的输出中的最后一个隐藏状态
        last_hidden_state = encoder_outputs[0] if not return_dict else encoder_outputs.last_hidden_state

        # 通过卷积层对最后隐藏状态进行投影，同时考虑填充mask
        stats = self.project(last_hidden_state.transpose(1, 2)).transpose(1, 2) * padding_mask
        # 将投影后的统计数据分割为先验均值和对数方差
        prior_means, prior_log_variances = torch.split(stats, self.config.flow_size, dim=2)

        if not return_dict:
            # 如果不要求返回字典，则返回元组形式的输出
            outputs = (last_hidden_state, prior_means, prior_log_variances) + encoder_outputs[1:]
            return outputs

        # 如果要求返回字典形式的输出，构建VitsTextEncoderOutput对象
        return VitsTextEncoderOutput(
            last_hidden_state=last_hidden_state,
            prior_means=prior_means,
            prior_log_variances=prior_log_variances,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class VitsPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置配置类
    config_class = VitsConfig
    # 基础模型前缀
    base_model_prefix = "vits"
    # 主要输入名称
    main_input_name = "input_ids"
    # 是否支持梯度检查点
    supports_gradient_checkpointing = True
    # 初始化神经网络模块的权重
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果是线性层
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置项，初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        
        # 如果是 LayerNorm 层
        elif isinstance(module, nn.LayerNorm):
            # 初始化偏置项为零
            module.bias.data.zero_()
            # 初始化权重为全1
            module.weight.data.fill_(1.0)
        
        # 如果是 1D 卷积层
        elif isinstance(module, nn.Conv1d):
            # 使用 Kaiming 初始化方法初始化权重
            nn.init.kaiming_normal_(module.weight)
            # 如果有偏置项，根据组、输入通道数和卷积核大小计算初始化范围并均匀分布初始化
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)
        
        # 如果是嵌入层
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果指定了填充索引，将该索引处的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
# 导入 VITS 模型的文档字符串，介绍了模型的继承关系和通用方法的使用
VITS_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`VitsConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# VITS 模型的输入文档字符串，详细描述了输入参数的含义和使用方法
VITS_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing convolution and attention on padding token indices. Mask values selected in `[0,
            1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        speaker_id (`int`, *optional*):
            Which speaker embedding to use. Only used for multispeaker models.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# 添加 VITS 模型的开始文档字符串，描述了完整的 VITS 模型用于文本到语音合成
@add_start_docstrings(
    "The complete VITS model, for text-to-speech synthesis.",
    VITS_START_DOCSTRING,
)
class VitsModel(VitsPreTrainedModel):
    # 类定义部分，继承自 VitsPreTrainedModel
    # 初始化函数，用于初始化模型对象
    def __init__(self, config: VitsConfig):
        # 调用父类的初始化方法，传入配置参数
        super().__init__(config)
        # 将配置参数保存到对象中
        self.config = config
        # 创建文本编码器对象，使用给定配置参数
        self.text_encoder = VitsTextEncoder(config)
        # 创建残差耦合块对象，使用给定配置参数
        self.flow = VitsResidualCouplingBlock(config)
        # 创建 HiFi-GAN 解码器对象，使用给定配置参数
        self.decoder = VitsHifiGan(config)

        # 根据配置决定使用随机时长预测器或固定时长预测器
        if config.use_stochastic_duration_prediction:
            self.duration_predictor = VitsStochasticDurationPredictor(config)
        else:
            self.duration_predictor = VitsDurationPredictor(config)

        # 如果配置中的说话人数量大于 1，则创建说话人嵌入对象
        if config.num_speakers > 1:
            self.embed_speaker = nn.Embedding(config.num_speakers, config.speaker_embedding_size)

        # 仅在训练时使用，创建后验编码器对象
        self.posterior_encoder = VitsPosteriorEncoder(config)

        # 初始化合成语音的参数，控制语速、噪声比例和噪声时长
        self.speaking_rate = config.speaking_rate
        self.noise_scale = config.noise_scale
        self.noise_scale_duration = config.noise_scale_duration

        # 执行初始化后处理操作
        self.post_init()

    # 返回文本编码器对象
    def get_encoder(self):
        return self.text_encoder

    # 前向传播函数，实现模型的前向计算
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