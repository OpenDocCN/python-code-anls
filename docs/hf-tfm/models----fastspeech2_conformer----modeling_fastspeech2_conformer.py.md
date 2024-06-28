# `.\models\fastspeech2_conformer\modeling_fastspeech2_conformer.py`

```py
# coding=utf-8
# Copyright 2023 The Espnet authors, IMS Toucan authors, and the HuggingFace Inc. team. All rights reserved.
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
""" PyTorch FastSpeech2Conformer model."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn

from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, logging, replace_return_docstrings
from .configuration_fastspeech2_conformer import (
    FastSpeech2ConformerConfig,
    FastSpeech2ConformerHifiGanConfig,
    FastSpeech2ConformerWithHifiGanConfig,
)

# 获取logger对象，用于日志记录
logger = logging.get_logger(__name__)

# FastSpeech2Conformer模型的预训练模型存档列表
FASTSPEECH2_CONFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "espnet/fastspeech2_conformer",
    # See all FastSpeech2Conformer models at https://huggingface.co/models?filter=fastspeech2_conformer
]

@dataclass
# FastSpeech2ConformerModelOutput类定义，用作FastSpeech2Conformer模型的输出类型
class FastSpeech2ConformerModelOutput(ModelOutput):
    """
    Output type of [`FastSpeech2ConformerModel`].
    """
    # loss 是一个可选的 torch.FloatTensor，表示生成语谱图的损失
    loss: Optional[torch.FloatTensor] = None

    # spectrogram 是一个 torch.FloatTensor，表示预测的语谱图，其形状为 (batch_size, sequence_length, num_bins)
    spectrogram: torch.FloatTensor = None

    # encoder_last_hidden_state 是一个可选的 torch.FloatTensor，表示模型编码器最后一层的隐藏状态序列，
    # 其形状为 (batch_size, sequence_length, hidden_size)
    encoder_last_hidden_state: torch.FloatTensor = None

    # encoder_hidden_states 是一个可选的元组(torch.FloatTensor)，当传递了 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回，
    # 其中包含模型编码器每一层的隐藏状态序列，形状为 (batch_size, sequence_length, hidden_size)
    encoder_hidden_states: tuple(torch.FloatTensor) = None

    # encoder_attentions 是一个可选的元组(torch.FloatTensor)，当传递了 `output_attentions=True` 或 `config.output_attentions=True` 时返回，
    # 包含模型编码器每一层的注意力权重，形状为 (batch_size, num_heads, sequence_length, sequence_length)
    encoder_attentions: tuple(torch.FloatTensor) = None

    # decoder_hidden_states 是一个可选的元组(torch.FloatTensor)，当传递了 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回，
    # 其中包含模型解码器每一层的隐藏状态序列，形状为 (batch_size, sequence_length, hidden_size)
    decoder_hidden_states: tuple(torch.FloatTensor) = None

    # decoder_attentions 是一个可选的元组(torch.FloatTensor)，当传递了 `output_attentions=True` 或 `config.output_attentions=True` 时返回，
    # 包含模型解码器每一层的注意力权重，形状为 (batch_size, num_heads, sequence_length, sequence_length)
    decoder_attentions: tuple(torch.FloatTensor) = None

    # duration_outputs 是一个可选的 torch.LongTensor，表示持续时间预测器的输出，
    # 形状为 (batch_size, max_text_length + 1)
    duration_outputs: torch.LongTensor = None

    # pitch_outputs 是一个可选的 torch.FloatTensor，表示音高预测器的输出，
    # 形状为 (batch_size, max_text_length + 1, 1)
    pitch_outputs: torch.FloatTensor = None

    # energy_outputs 是一个可选的 torch.FloatTensor，表示能量预测器的输出，
    # 形状为 (batch_size, max_text_length + 1, 1)
    energy_outputs: torch.FloatTensor = None
    # 定义可选的变量，用于存储编码器最终隐藏状态的张量
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 定义可选的变量，用于存储编码器所有隐藏状态的元组张量
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义可选的变量，用于存储编码器注意力分布的元组张量
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 定义可选的变量，用于存储解码器隐藏状态的元组张量
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义可选的变量，用于存储解码器注意力分布的元组张量
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 定义默认为None的变量，用于存储输出的持续时间预测结果的长整型张量
    duration_outputs: torch.LongTensor = None
    # 定义默认为None的变量，用于存储输出的音高预测结果的浮点数张量
    pitch_outputs: torch.FloatTensor = None
    # 定义默认为None的变量，用于存储输出的能量预测结果的浮点数张量
    energy_outputs: torch.FloatTensor = None
@dataclass
class FastSpeech2ConformerWithHifiGanOutput(FastSpeech2ConformerModelOutput):
    """
    Output type of [`FastSpeech2ConformerWithHifiGan`].

    """

    # 用于存储生成的波形数据的张量
    waveform: torch.FloatTensor = None


_CONFIG_FOR_DOC = "FastSpeech2ConformerConfig"

FASTSPEECH2_CONFORMER_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FastSpeech2ConformerConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


HIFIGAN_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FastSpeech2ConformerConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

FASTSPEECH2_CONFORMER_WITH_HIFIGAN_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`FastSpeech2ConformerWithHifiGanConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
def length_regulator(encoded_embeddings, duration_labels, speaking_speed=1.0):
    """
    Length regulator for feed-forward Transformer.

    This is the length regulator module described in `FastSpeech: Fast, Robust and Controllable Text to Speech`
    https://arxiv.org/pdf/1905.09263.pdf. The length regulator expands char or phoneme-level embedding features to
    frame-level by repeating each feature based on the corresponding predicted durations.

    Args:
        encoded_embeddings (`torch.Tensor` of shape `(batch_size, max_text_length, embedding_dim)`):
            Batch of sequences of char or phoneme embeddings.
        duration_labels (`torch.LongTensor` of shape `(batch_size, time)`):
            Batch of durations of each frame.
        speaking_speed (`float`, *optional*, defaults to 1.0):
            Value to control speed of speech.

    Returns:
        `torch.Tensor`:
            Replicated input tensor based on durations (batch_size, time*, embedding_dim).
    """

    if speaking_speed <= 0:
        raise ValueError("`speaking_speed` must be greater than 0.")
    elif speaking_speed != 1.0:
        # Adjust duration labels based on speaking speed if it's not 1.0
        duration_labels = torch.round(duration_labels.float() * speaking_speed).long()

    if duration_labels.sum() == 0:
        # Ensure at least one frame per sequence if all durations sum to zero
        duration_labels[duration_labels.sum(dim=1).eq(0)] = 1

    # Calculate the maximum length needed based on the sum of duration labels per batch
    max_len = torch.sum(duration_labels, dim=1).max()

    # Create a padded tensor to hold the expanded embeddings
    hidden_states = torch.zeros(
        (encoded_embeddings.size(0), max_len, encoded_embeddings.size(2)),
        dtype=torch.float,
        device=encoded_embeddings.device,
    )

    # Loop through each sequence in the batch and expand embeddings based on duration labels
    for i, (encoded_embedding, target_duration) in enumerate(zip(encoded_embeddings, duration_labels)):
        # Repeat each embedding based on its corresponding duration label
        repeated = torch.repeat_interleave(encoded_embedding, target_duration, dim=0)
        # Place repeated embeddings into the padded tensor
        hidden_states[i, : repeated.size(0)] = repeated

    return hidden_states


class FastSpeech2ConformerDurationPredictor(nn.Module):
    """
    Duration predictor module.

    This is a module of duration predictor described in the paper 'FastSpeech: Fast, Robust and Controllable Text to
    Speech' https://arxiv.org/pdf/1905.09263.pdf The duration predictor predicts a duration of each frame in log domain
    from the hidden embeddings of encoder.

    Note:
        The calculation domain of outputs is different between in `forward` and in `inference`. In `forward`, the
        outputs are calculated in log domain but in `inference`, those are calculated in linear domain.

    """
    def __init__(self, config: FastSpeech2ConformerConfig):
        super().__init__()

        # 初始化卷积层列表
        self.conv_layers = nn.ModuleList()
        # 设置在对数域计算时的偏移量
        self.log_domain_offset = 1.0

        # 根据配置信息循环创建持续预测器的卷积层
        for layer_idx in range(config.duration_predictor_layers):
            num_chans = config.duration_predictor_channels
            # 确定当前层的输入通道数
            input_channels = config.hidden_size if layer_idx == 0 else num_chans
            # 创建并添加预测器层对象到卷积层列表
            layer = FastSpeech2ConformerPredictorLayer(
                input_channels,
                num_chans,
                config.duration_predictor_kernel_size,
                config.duration_predictor_dropout_rate,
            )
            self.conv_layers.append(layer)

        # 创建线性层，输出维度为1，用于预测持续时间
        self.linear = nn.Linear(config.duration_predictor_channels, 1)

    def forward(self, encoder_hidden_states):
        """
        Args:
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, max_text_length, input_dim)`):
                输入序列的批次数据.
                input_dim是每个时间步的特征维度.

        Returns:
            `torch.Tensor`: 在对数域中预测的持续时间 `(batch_size, max_text_length)`.

        """
        # 调整输入张量的维度顺序为(batch_size, input_dim, max_text_length)
        hidden_states = encoder_hidden_states.transpose(1, -1)
        
        # 逐层通过卷积层处理隐藏状态
        for layer in self.conv_layers:
            hidden_states = layer(hidden_states)

        # 在对数域中计算线性层的输出，调整维度为(batch_size, max_text_length)
        hidden_states = self.linear(hidden_states.transpose(1, -1)).squeeze(-1)

        if not self.training:
            # 若非训练模式，转换回线性域并进行修剪
            hidden_states = torch.clamp(torch.round(hidden_states.exp() - self.log_domain_offset), min=0).long()

        return hidden_states
# Copied from transformers.models.speecht5.modeling_speecht5.SpeechT5BatchNormConvLayer
# 定义了一个名为 FastSpeech2ConformerBatchNormConvLayer 的类，继承自 nn.Module
class FastSpeech2ConformerBatchNormConvLayer(nn.Module):
    # 初始化方法，接受 config 和可选的 layer_id 参数
    def __init__(self, config, layer_id=0):
        super().__init__()

        # 根据 layer_id 决定输入卷积层的维度
        if layer_id == 0:
            in_conv_dim = config.num_mel_bins
        else:
            in_conv_dim = config.speech_decoder_postnet_units

        # 根据 layer_id 决定输出卷积层的维度
        if layer_id == config.speech_decoder_postnet_layers - 1:
            out_conv_dim = config.num_mel_bins
        else:
            out_conv_dim = config.speech_decoder_postnet_units

        # 创建一个 1 维卷积层，设置输入维度、输出维度、卷积核大小、步长、填充和是否包含偏置
        self.conv = nn.Conv1d(
            in_conv_dim,
            out_conv_dim,
            kernel_size=config.speech_decoder_postnet_kernel,
            stride=1,
            padding=(config.speech_decoder_postnet_kernel - 1) // 2,
            bias=False,
        )

        # 创建一个 1 维批归一化层，设置归一化的通道数
        self.batch_norm = nn.BatchNorm1d(out_conv_dim)

        # 根据 layer_id 决定是否使用激活函数 Tanh
        if layer_id < config.speech_decoder_postnet_layers - 1:
            self.activation = nn.Tanh()
        else:
            self.activation = None

        # 创建一个 Dropout 层，设置丢弃率
        self.dropout = nn.Dropout(config.speech_decoder_postnet_dropout)

    # 前向传播方法，接受 hidden_states 作为输入，返回处理后的 hidden_states
    def forward(self, hidden_states):
        # 将输入 hidden_states 经过卷积层 conv 处理
        hidden_states = self.conv(hidden_states)
        # 将卷积层的输出经过批归一化层 batch_norm 处理
        hidden_states = self.batch_norm(hidden_states)
        # 如果有激活函数 activation，则将批归一化后的结果经过激活函数处理
        if self.activation is not None:
            hidden_states = self.activation(hidden_states)
        # 将处理后的结果经过 Dropout 处理
        hidden_states = self.dropout(hidden_states)
        # 返回处理后的 hidden_states
        return hidden_states


# 定义了一个名为 FastSpeech2ConformerSpeechDecoderPostnet 的类，继承自 nn.Module
class FastSpeech2ConformerSpeechDecoderPostnet(nn.Module):
    # 初始化方法，接受 config 参数
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 创建一个线性层，将隐藏状态映射到输出特征的大小
        self.feat_out = nn.Linear(config.hidden_size, config.num_mel_bins * config.reduction_factor)
        # 创建一个由多个 FastSpeech2ConformerBatchNormConvLayer 组成的层列表
        self.layers = nn.ModuleList(
            [FastSpeech2ConformerBatchNormConvLayer(config, i) for i in range(config.speech_decoder_postnet_layers)]
        )

    # 前向传播方法，接受 hidden_states 作为输入，返回处理后的 outputs_before_postnet 和 outputs_after_postnet
    def forward(self, hidden_states: torch.Tensor):
        # 将隐藏状态通过线性层 feat_out 映射到输出特征的大小，并重塑输出形状
        outputs_before_postnet = self.feat_out(hidden_states).view(hidden_states.size(0), -1, self.config.num_mel_bins)
        # 将重塑后的输出结果转置，以便后续处理
        layer_output = outputs_before_postnet.transpose(1, 2)
        # 遍历每个层，并将 layer_output 依次经过每一层处理
        for layer in self.layers:
            layer_output = layer(layer_output)
        # 将原始输出和经过层处理后的结果进行相加，得到最终的 outputs_after_postnet
        outputs_after_postnet = outputs_before_postnet + layer_output.transpose(1, 2)
        # 返回处理后的 outputs_before_postnet 和 outputs_after_postnet
        return outputs_before_postnet, outputs_after_postnet


# 定义了一个名为 FastSpeech2ConformerPredictorLayer 的类，继承自 nn.Module
class FastSpeech2ConformerPredictorLayer(nn.Module):
    # 初始化方法，接受 input_channels、num_chans、kernel_size 和 dropout_rate 参数
    def __init__(self, input_channels, num_chans, kernel_size, dropout_rate):
        super().__init__()
        # 创建一个 1 维卷积层，设置输入通道数、输出通道数、卷积核大小、步长、填充
        self.conv = nn.Conv1d(
            input_channels,
            num_chans,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        # 创建一个 ReLU 激活函数
        self.activation = nn.ReLU()
        # 创建一个 LayerNorm 层，设置归一化的通道数
        self.layer_norm = nn.LayerNorm(num_chans)
        # 创建一个 Dropout 层，设置丢弃率
        self.dropout = nn.Dropout(dropout_rate)
    # 定义一个前向传播函数，接收隐藏状态作为输入
    def forward(self, hidden_states):
        # 使用卷积层处理隐藏状态
        hidden_states = self.conv(hidden_states)
        # 对卷积层输出应用激活函数
        hidden_states = self.activation(hidden_states)

        # 在第1维上执行层归一化操作
        hidden_states = hidden_states.transpose(1, -1)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states.transpose(1, -1)

        # 对处理后的隐藏状态应用 dropout
        hidden_states = self.dropout(hidden_states)

        # 返回处理后的隐藏状态作为输出
        return hidden_states
class FastSpeech2ConformerVariancePredictor(nn.Module):
    def __init__(
        self,
        config: FastSpeech2ConformerConfig,
        num_layers=2,
        num_chans=384,
        kernel_size=3,
        dropout_rate=0.5,
    ):
        """
        Initilize variance predictor module.

        Args:
            config (`FastSpeech2ConformerConfig`): Configuration object for the model.
            num_layers (`int`, *optional*, defaults to 2): Number of convolutional layers.
            num_chans (`int`, *optional*, defaults to 384): Number of channels of convolutional layers.
            kernel_size (`int`, *optional*, defaults to 3): Kernel size of convolutional layers.
            dropout_rate (`float`, *optional*, defaults to 0.5): Dropout rate.
        """
        super().__init__()
        # 创建包含多个卷积层的模块列表
        self.conv_layers = nn.ModuleList()
        for idx in range(num_layers):
            input_channels = config.hidden_size if idx == 0 else num_chans
            # 创建并添加一个新的卷积层到模块列表中
            layer = FastSpeech2ConformerPredictorLayer(input_channels, num_chans, kernel_size, dropout_rate)
            self.conv_layers.append(layer)
        # 创建一个线性层，用于最终预测
        self.linear = nn.Linear(num_chans, 1)

    def forward(self, encoder_hidden_states, padding_masks=None):
        """
        Calculate forward propagation.

        Args:
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, max_text_length, input_dim)`):
                Batch of input sequences.
            padding_masks (`torch.ByteTensor` of shape `(batch_size, max_text_length)`, *optional*):
                Batch of masks indicating padded part.

        Returns:
            Tensor: Batch of predicted sequences `(batch_size, max_text_length, 1)`.
        """
        # 将输入的隐藏状态进行维度转置
        hidden_states = encoder_hidden_states.transpose(1, -1)
        # 通过所有卷积层进行前向传播计算
        for layer in self.conv_layers:
            hidden_states = layer(hidden_states)

        # 对输出结果进行线性变换
        hidden_states = self.linear(hidden_states.transpose(1, 2))

        # 如果提供了填充掩码，则使用掩码将填充部分置为零
        if padding_masks is not None:
            hidden_states = hidden_states.masked_fill(padding_masks, 0.0)

        return hidden_states


class FastSpeech2ConformerVarianceEmbedding(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=384,
        kernel_size=1,
        padding=0,
        dropout_rate=0.0,
    ):
        """
        Initialize variance embedding module.

        Args:
            in_channels (`int`, *optional*, defaults to 1): Number of input channels.
            out_channels (`int`, *optional*, defaults to 384): Number of output channels.
            kernel_size (`int`, *optional*, defaults to 1): Kernel size of the convolutional layer.
            padding (`int`, *optional*, defaults to 0): Padding size of the convolutional layer.
            dropout_rate (`float`, *optional*, defaults to 0.0): Dropout rate.
        """
        super().__init__()
        # 创建一个卷积层，用于嵌入变量
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        # 创建一个丢弃层，用于随机丢弃数据
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states):
        # 将输入的隐藏状态进行维度转置
        hidden_states = hidden_states.transpose(1, 2)
        # 通过卷积层进行前向传播计算
        hidden_states = self.conv(hidden_states)
        # 通过丢弃层进行前向传播计算
        hidden_states = self.dropout(hidden_states)
        # 再次将隐藏状态的维度进行转置
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


class FastSpeech2ConformerAttention(nn.Module):
    """
    Multi-Head attention layer with relative position encoding. Details can be found in
    """
    """
    https://github.com/espnet/espnet/pull/2816. Paper: https://arxiv.org/abs/1901.02860.
    """

    # 初始化函数，创建一个 FastSpeech2ConformerAttention 对象
    def __init__(self, config: FastSpeech2ConformerConfig, module_config):
        """Construct an FastSpeech2ConformerAttention object."""
        # 调用父类的初始化方法
        super().__init__()

        # 假设 d_v 总是等于 dim_key
        # 设置注意力头的数量
        self.num_heads = module_config["num_attention_heads"]
        # 获取隐藏层大小
        self.hidden_size = config.hidden_size
        # 计算 key 的维度
        self.dim_key = self.hidden_size // self.num_heads
        # 计算每个头的维度
        self.head_dim = self.hidden_size // self.num_heads

        # 初始化 Linear 层，用于查询（query）、键（key）、值（value）和输出（output）
        self.linear_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_v = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_out = nn.Linear(self.hidden_size, self.hidden_size)

        # Dropout 层，用于注意力机制中的 dropout
        self.dropout = nn.Dropout(p=module_config["attention_dropout_rate"])

        # 用于位置编码的线性变换
        self.linear_pos = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # 学习得到的偏置参数，用于矩阵 c 和矩阵 d
        # 参见论文 https://arxiv.org/abs/1901.02860 第 3.3 节的描述
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.num_heads, self.head_dim))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.num_heads, self.head_dim))

    # 移动相对位置张量
    def shift_relative_position_tensor(self, pos_tensor):
        """
        Args:
            pos_tensor (torch.Tensor of shape (batch_size, head, time1, 2*time1-1)): Input tensor.
        """
        # 在最后一个维度上填充零，扩展张量
        zero_pad = torch.zeros((*pos_tensor.size()[:3], 1), device=pos_tensor.device, dtype=pos_tensor.dtype)
        pos_tensor_padded = torch.cat([zero_pad, pos_tensor], dim=-1)

        # 重新组织张量的形状，将最后一个维度扩展一个单位
        pos_tensor_padded = pos_tensor_padded.view(*pos_tensor.size()[:2], pos_tensor.size(3) + 1, pos_tensor.size(2))
        
        # 保留位置从 0 到 time2 的部分
        pos_tensor = pos_tensor_padded[:, :, 1:].view_as(pos_tensor)[:, :, :, : pos_tensor.size(-1) // 2 + 1]

        return pos_tensor

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pos_emb: Optional[torch.Tensor] = None,
        output_attentions: Optional[torch.Tensor] = False,
class FastSpeech2ConformerConvolutionModule(nn.Module):
    def __init__(self, config: FastSpeech2ConformerConfig, module_config):
        super().__init__()
        # kernel_size should be an odd number for 'SAME' padding
        channels = config.hidden_size
        kernel_size = module_config["kernel_size"]
        
        # 定义第一个逐点卷积层，将输入通道数变换为2倍的输出通道数
        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels, kernel_size=1, stride=1, padding=0, bias=True)
        
        # 定义深度卷积层，应用1维深度卷积，groups设置为通道数，使用SAME填充以保持长度不变
        self.depthwise_conv = nn.Conv1d(
            channels, channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2, groups=channels, bias=True
        )
        
        # 定义批标准化层，用于归一化深度卷积层的输出
        self.norm = nn.BatchNorm1d(channels)
        
        # 定义第二个逐点卷积层，将输出通道数恢复为原来的通道数
        self.pointwise_conv2 = nn.Conv1d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, hidden_states):
        """
        Compute convolution module.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch, time, channels)`): Input tensor.

        Returns:
            `torch.Tensor`: Output tensor of shape `(batch, time, channels)`.

        """
        # 交换时间维度和特征维度，将 (batch, time, channels) 转换为 (batch, channels, time)
        hidden_states = hidden_states.transpose(1, 2)

        # 应用GLU机制，将 (batch_size, 2*channels, time) 转换为 (batch_size, channels, time)
        hidden_states = self.pointwise_conv1(hidden_states)
        hidden_states = nn.functional.glu(hidden_states, dim=1)

        # 应用深度卷积
        hidden_states = self.depthwise_conv(hidden_states)
        
        # 应用批标准化
        hidden_states = self.norm(hidden_states)

        # 应用sigmoid函数，并将结果与深度卷积输出相乘
        hidden_states = hidden_states * torch.sigmoid(hidden_states)

        # 应用第二个逐点卷积层，将 (batch, channels, time) 转换回 (batch, time, channels)
        hidden_states = self.pointwise_conv2(hidden_states)

        return hidden_states.transpose(1, 2)
    # 初始化函数，用于初始化一个 FastSpeech2ConformerConfig 类的实例
    def __init__(self, config: FastSpeech2ConformerConfig, module_config):
        # 调用父类的初始化方法
        super().__init__()

        # 定义自注意力模块
        self.self_attn = FastSpeech2ConformerAttention(config, module_config)

        # 定义前馈模块
        self.feed_forward = FastSpeech2ConformerMultiLayeredConv1d(config, module_config)

        # 根据配置选择是否使用 Macaron 风格
        self.macaron_style = config.use_macaron_style_in_conformer
        if self.macaron_style:
            # 如果使用 Macaron 风格，定义额外的前馈模块和层归一化
            self.feed_forward_macaron = FastSpeech2ConformerMultiLayeredConv1d(config, module_config)
            self.ff_macaron_layer_norm = nn.LayerNorm(config.hidden_size)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0  # 否则设定前馈缩放因子为 1.0

        # 根据配置选择是否使用卷积模块
        self.use_cnn_module = config.use_cnn_in_conformer
        if self.use_cnn_module:
            # 如果使用卷积模块，定义卷积模块和两个层归一化
            self.conv_module = FastSpeech2ConformerConvolutionModule(config, module_config)
            self.conv_layer_norm = nn.LayerNorm(config.hidden_size)
            self.final_layer_norm = nn.LayerNorm(config.hidden_size)

        # 定义前馈层归一化
        self.ff_layer_norm = nn.LayerNorm(config.hidden_size)

        # 定义自注意力层归一化
        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size)

        # 定义 dropout 层
        self.dropout = nn.Dropout(module_config["dropout_rate"])

        # 定义 hidden size 大小
        self.size = config.hidden_size

        # 从模块配置中获取是否在归一化前执行操作和是否在拼接之后执行操作的标志
        self.normalize_before = module_config["normalize_before"]
        self.concat_after = module_config["concat_after"]
        if self.concat_after:
            # 如果在拼接之后执行操作，定义一个线性层用于拼接后的向量变换
            self.concat_linear = nn.Linear(config.hidden_size + config.hidden_size, config.hidden_size)

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        pos_emb: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[torch.Tensor] = False,
class FastSpeech2ConformerMultiLayeredConv1d(nn.Module):
    """
    Multi-layered conv1d for Transformer block.

    This is a module of multi-layered conv1d designed to replace positionwise feed-forward network in Transformer
    block, which is introduced in 'FastSpeech: Fast, Robust and Controllable Text to Speech'
    https://arxiv.org/pdf/1905.09263.pdf
    """

    def __init__(self, config: FastSpeech2ConformerConfig, module_config):
        """
        Initialize FastSpeech2ConformerMultiLayeredConv1d module.

        Args:
            config (`FastSpeech2ConformerConfig`): Configuration object containing model parameters.
            module_config (`dict`): Dictionary containing specific module configurations.
        """
        super().__init__()
        # Set input channels from config
        input_channels = config.hidden_size
        # Set hidden channels from module_config
        hidden_channels = module_config["linear_units"]
        # Set kernel size from config
        kernel_size = config.positionwise_conv_kernel_size
        # Define the first convolution layer
        self.conv1 = nn.Conv1d(input_channels, hidden_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        # Define the second convolution layer
        self.conv2 = nn.Conv1d(hidden_channels, input_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        # Define dropout layer with dropout rate from module_config
        self.dropout = nn.Dropout(module_config["dropout_rate"])

    def forward(self, hidden_states):
        """
        Perform forward propagation through the module.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch_size, time, input_channels).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, time, hidden_channels).
        """
        # Transpose tensor to (batch_size, input_channels, time)
        hidden_states = hidden_states.transpose(-1, 1)
        # Apply first convolution layer
        hidden_states = self.conv1(hidden_states)
        # Apply ReLU activation function
        hidden_states = torch.relu(hidden_states)
        # Apply dropout
        hidden_states = self.dropout(hidden_states)
        # Apply second convolution layer
        hidden_states = self.conv2(hidden_states)
        # Transpose tensor back to (batch_size, time, hidden_channels)
        hidden_states = hidden_states.transpose(-1, 1)
        return hidden_states


class FastSpeech2ConformerRelPositionalEncoding(nn.Module):
    """
    Relative positional encoding module (new implementation).

    Args:
        config (`FastSpeech2ConformerConfig`): Configuration object containing model parameters.
        module_config (`dict`): Dictionary containing specific module configurations.
    Details can be found in https://github.com/espnet/espnet/pull/2816. See : Appendix Batch in https://arxiv.org/abs/1901.02860
    """

    def __init__(self, config: FastSpeech2ConformerConfig, module_config):
        """
        Construct a FastSpeech2ConformerRelPositionalEncoding object.

        Args:
            config (`FastSpeech2ConformerConfig`): Configuration object containing model parameters.
            module_config (`dict`): Dictionary containing specific module configurations.
        """
        super().__init__()
        # Initialize embedding dimension from config
        self.embed_dim = config.hidden_size
        # Set input scale as square root of embedding dimension
        self.input_scale = math.sqrt(self.embed_dim)
        # Initialize dropout layer with positional dropout rate from module_config
        self.dropout = nn.Dropout(p=module_config["positional_dropout_rate"])
        # Initialize positional encoding as None initially
        self.pos_enc = None
        # Set maximum length for positional encoding
        self.max_len = 5000
        # Extend positional encoding with a tensor of zeros
        self.extend_pos_enc(torch.tensor(0.0).expand(1, self.max_len))
    def extend_pos_enc(self, x):
        """Reset the positional encodings."""
        # 如果已经存在位置编码，则检查是否需要重新初始化
        if self.pos_enc is not None:
            # self.pos_enc 包含正负两部分
            # self.pos_enc 的长度为 2 * 输入长度 - 1
            if self.pos_enc.size(1) >= x.size(1) * 2 - 1:
                # 如果当前的数据类型或设备与输入不匹配，则将位置编码转换为相应类型和设备
                if self.pos_enc.dtype != x.dtype or self.pos_enc.device != x.device:
                    self.pos_enc = self.pos_enc.to(dtype=x.dtype, device=x.device)
                return
        # 创建正位置编码和负位置编码
        pos_enc_positive = torch.zeros(x.size(1), self.embed_dim)
        pos_enc_negative = torch.zeros(x.size(1), self.embed_dim)
        # 生成位置向量，表示位置的相对关系
        position = torch.arange(0, x.size(1), dtype=torch.int64).float().unsqueeze(1)
        # 计算正弦和余弦项的分母
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2, dtype=torch.int64).float() * -(math.log(10000.0) / self.embed_dim)
        )
        # 计算正位置编码的正弦和余弦值
        pos_enc_positive[:, 0::2] = torch.sin(position * div_term)
        pos_enc_positive[:, 1::2] = torch.cos(position * div_term)
        # 计算负位置编码的正弦和余弦值
        pos_enc_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pos_enc_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # 翻转正位置编码的顺序并连接正负位置编码，以支持平移技巧
        # 参考 https://arxiv.org/abs/1901.02860
        pos_enc_positive = torch.flip(pos_enc_positive, [0]).unsqueeze(0)
        pos_enc_negative = pos_enc_negative[1:].unsqueeze(0)
        pos_enc = torch.cat([pos_enc_positive, pos_enc_negative], dim=1)
        self.pos_enc = pos_enc.to(device=x.device, dtype=x.dtype)

    def forward(self, feature_representation):
        """
        Args:
            feature_representation (`torch.Tensor` of shape (batch_size, time, `*`)):
                Input tensor.

        Returns:
            `torch.Tensor`: Encoded tensor (batch_size, time, `*`).
        """
        # 扩展或重置位置编码
        self.extend_pos_enc(feature_representation)
        # 对特征表示进行缩放
        hidden_states = feature_representation * self.input_scale
        # 计算中心索引
        center_idx = self.pos_enc.size(1) // 2
        # 提取位置编码的一部分，以便与隐藏状态匹配
        pos_emb = self.pos_enc[:, center_idx - hidden_states.size(1) + 1 : center_idx + hidden_states.size(1)]
        return self.dropout(hidden_states), self.dropout(pos_emb)
# FastSpeech2ConformerEncoder 类定义，作为 FastSpeech2 模型的编码器模块
class FastSpeech2ConformerEncoder(nn.Module):
    """
    FastSpeech2ConformerEncoder encoder module.

    Args:
        config (`FastSpeech2ConformerConfig`):
            FastSpeech2ConformerConfig instance. 模型配置参数对象
        module_config (`dict`):
            Dictionary containing the encoder or decoder module configuration from the `FastSpeech2ConformerConfig`.
            包含编码器或解码器模块配置的字典，从 FastSpeech2ConformerConfig 中获取
        use_encoder_input_layer (`bool`, *optional*, defaults to `False`):
            Input layer type. 是否使用编码器输入层类型

    """

    def __init__(
        self,
        config: FastSpeech2ConformerConfig,
        module_config,
        use_encoder_input_layer=False,
    ):
        super().__init__()

        self.embed = None
        # 如果指定了使用编码器输入层，则创建一个词嵌入层
        if use_encoder_input_layer:
            self.embed = nn.Embedding(
                num_embeddings=config.vocab_size, embedding_dim=config.hidden_size, padding_idx=0
            )

        # 创建相对位置编码器
        self.pos_enc = FastSpeech2ConformerRelPositionalEncoding(config, module_config)

        # 创建多个 Conformer 层的列表
        self.conformer_layers = nn.ModuleList(
            [FastSpeech2ConformerEncoderLayer(config, module_config) for _ in range(module_config["layers"])]
        )

    def forward(
        self,
        input_tensor: torch.LongTensor,
        attention_mask: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                输入序列标记在词汇表中的索引。默认情况下，将忽略填充标记。

                可以使用 [`AutoTokenizer`] 获得索引。有关详细信息，请参见 [`PreTrainedTokenizer.encode`] 和
                [`PreTrainedTokenizer.__call__`]。

                [什么是输入 ID？](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *可选*):
                遮罩，用于避免在填充标记索引上执行注意力计算。遮罩值在 `[0, 1]` 范围内选择：

                - 对于 **未遮罩** 的标记，为 1，
                - 对于 **遮罩** 的标记，为 0。

                [什么是注意力遮罩？](../glossary#attention-mask)
            output_hidden_states (`bool`, *可选*):
                是否返回所有层的隐藏状态。有关详细信息，请参见返回张量中的 `hidden_states`。
            output_attentions (`bool`, *可选*):
                是否返回所有注意力层的注意力张量。有关详细信息，请参见返回张量中的 `attentions`。
            return_dict (`bool`, *可选*):
                是否返回 [`~utils.ModelOutput`] 而不是简单元组。
        Returns:
            `torch.Tensor`:
                形状为 `(batch, time, attention_dim)` 的输出张量。
        """
        # 将输入张量视为特征表示
        feature_representation = input_tensor
        # 如果存在嵌入层，则使用嵌入层处理特征表示
        if self.embed is not None:
            feature_representation = self.embed(feature_representation)

        # 使用位置编码器处理特征表示和位置编码
        hidden_states, pos_emb = self.pos_enc(feature_representation)

        # 初始化存储所有隐藏状态和注意力张量的元组
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # 逐层处理Conformer模型的每个层
        for conformer_layer in self.conformer_layers:
            # 如果需要输出隐藏状态，则添加当前层的隐藏状态到存储中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 对当前层进行处理，获取其输出
            layer_outputs = conformer_layer(hidden_states, pos_emb, attention_mask, output_attentions)
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力张量，则添加当前层的注意力张量到存储中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 添加最后一层的隐藏状态到存储中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 根据 return_dict 决定返回格式
        if not return_dict:
            # 返回包含非空项的元组
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 返回格式化的 BaseModelOutput 对象
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions
        )
class FastSpeech2ConformerLoss(nn.Module):
    def __init__(self, config: FastSpeech2ConformerConfig):
        super().__init__()

        use_masking = config.use_masking
        use_weighted_masking = config.use_weighted_masking

        # 检查是否同时开启了 use_masking 和 use_weighted_masking
        if use_masking and use_weighted_masking:
            raise ValueError("Either use_masking or use_weighted_masking can be True, but not both.")

        # 设置是否使用 masking 和 weighted masking
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking

        # 根据是否使用 weighted masking 设置损失函数的缩减方式
        reduction = "none" if self.use_weighted_masking else "mean"
        # 定义 L1 损失函数
        self.l1_criterion = nn.L1Loss(reduction=reduction)
        # 定义 MSE 损失函数
        self.mse_criterion = nn.MSELoss(reduction=reduction)
        # 定义 duration 损失函数
        self.duration_criterion = nn.MSELoss(reduction=reduction)
        # 设置对数域偏移量
        self.log_domain_offset = 1.0

    def forward(
        self,
        outputs_after_postnet,
        outputs_before_postnet,
        duration_outputs,
        pitch_outputs,
        energy_outputs,
        spectrogram_labels,
        duration_labels,
        pitch_labels,
        energy_labels,
        duration_mask,
        spectrogram_mask,



class FastSpeech2ConformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 配置类
    config_class = FastSpeech2ConformerConfig
    # 基础模型前缀
    base_model_prefix = "fastspeech2_conformer"

    # 主要输入名称
    main_input_name = "input_ids"

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, (nn.LayerNorm)):
            # 将 LayerNorm 层的偏置初始化为零
            module.bias.data.zero_()
            # 将 LayerNorm 层的权重初始化为 1.0
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            # 使用 Kaiming 初始化卷积层的权重
            nn.init.kaiming_normal_(module.weight)
            # 如果存在偏置，使用均匀分布初始化
            if module.bias is not None:
                key = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-key, b=key)
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化 Embedding 层的权重
            module.weight.data.normal_()
            # 如果有 padding_idx，将对应位置的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, FastSpeech2ConformerAttention):
            # 使用 Xavier 初始化注意力机制中的位置偏置
            nn.init.xavier_uniform_(module.pos_bias_u)
            nn.init.xavier_uniform_(module.pos_bias_v)

    def _set_gradient_checkpointing(self, module, value=False):
        # 如果是 FastSpeech2ConformerEncoder 类型的模块，设置梯度检查点
        if isinstance(module, FastSpeech2ConformerEncoder):
            module.gradient_checkpointing = value



@add_start_docstrings(
    """FastSpeech2Conformer Model.""",
    FASTSPEECH2_CONFORMER_START_DOCSTRING,
)
class FastSpeech2ConformerModel(FastSpeech2ConformerPreTrainedModel):
    """
    FastSpeech 2 module.

    This is a module of FastSpeech 2 described in 'FastSpeech 2: Fast and High-Quality End-to-End Text to Speech'
    https://arxiv.org/abs/2006.04558. Instead of quantized pitch and energy, we use token-averaged value introduced in
    """
    FastPitch: Parallel Text-to-speech with Pitch Prediction. The encoder and decoder are Conformers instead of regular
    Transformers.
    """

    @replace_return_docstrings(output_type=FastSpeech2ConformerModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        spectrogram_labels: Optional[torch.FloatTensor] = None,
        duration_labels: Optional[torch.LongTensor] = None,
        pitch_labels: Optional[torch.FloatTensor] = None,
        energy_labels: Optional[torch.FloatTensor] = None,
        speaker_ids: Optional[torch.LongTensor] = None,
        lang_ids: Optional[torch.LongTensor] = None,
        speaker_embedding: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
# 从 transformers.models.speecht5.modeling_speecht5.HifiGanResidualBlock 复制的残差块类
class HifiGanResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), leaky_relu_slope=0.1):
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope

        # 第一组卷积层列表，使用不同的扩张率创建卷积层
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
        
        # 第二组卷积层列表，每个卷积层的扩张率都为1，但使用相同的填充函数
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

    # 计算卷积的填充量
    def get_padding(self, kernel_size, dilation=1):
        return (kernel_size * dilation - dilation) // 2

    # 应用权重归一化到所有卷积层
    def apply_weight_norm(self):
        for layer in self.convs1:
            nn.utils.weight_norm(layer)
        for layer in self.convs2:
            nn.utils.weight_norm(layer)

    # 移除所有卷积层的权重归一化
    def remove_weight_norm(self):
        for layer in self.convs1:
            nn.utils.remove_weight_norm(layer)
        for layer in self.convs2:
            nn.utils.remove_weight_norm(layer)

    # 前向传播函数定义
    def forward(self, hidden_states):
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = hidden_states
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)  # 应用 LeakyReLU 激活函数
            hidden_states = conv1(hidden_states)  # 第一组卷积层
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)  # 再次应用 LeakyReLU 激活函数
            hidden_states = conv2(hidden_states)  # 第二组卷积层
            hidden_states = hidden_states + residual  # 加上残差连接
        return hidden_states


# 从 transformers.models.speecht5.modeling_speecht5.SpeechT5HifiGan 复制的类，并将 SpeechT5 替换为 FastSpeech2Conformer
@add_start_docstrings(
    """HiFi-GAN vocoder.""",
    HIFIGAN_START_DOCSTRING,
)
class FastSpeech2ConformerHifiGan(PreTrainedModel):
    config_class = FastSpeech2ConformerHifiGanConfig
    main_input_name = "spectrogram"
    def __init__(self, config: FastSpeech2ConformerHifiGanConfig):
        # 调用父类的初始化方法，传入配置参数
        super().__init__(config)
        # 计算使用的残差块卷积核数量和上采样率数量
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        # 创建一个卷积层，用于预处理输入特征
        self.conv_pre = nn.Conv1d(
            config.model_in_dim,
            config.upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding=3,
        )

        # 创建上采样层，根据配置中的参数创建多个反卷积层，并添加到模块列表中
        self.upsampler = nn.ModuleList()
        for i, (upsample_rate, kernel_size) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            self.upsampler.append(
                nn.ConvTranspose1d(
                    config.upsample_initial_channel // (2**i),
                    config.upsample_initial_channel // (2 ** (i + 1)),
                    kernel_size=kernel_size,
                    stride=upsample_rate,
                    padding=(kernel_size - upsample_rate) // 2,
                )
            )

        # 创建残差块层，根据配置中的参数创建多个残差块，并添加到模块列表中
        self.resblocks = nn.ModuleList()
        for i in range(len(self.upsampler)):
            channels = config.upsample_initial_channel // (2 ** (i + 1))
            for kernel_size, dilation in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
                self.resblocks.append(HifiGanResidualBlock(channels, kernel_size, dilation, config.leaky_relu_slope))

        # 创建一个卷积层，用于后处理输出特征
        self.conv_post = nn.Conv1d(channels, 1, kernel_size=7, stride=1, padding=3)

        # 注册缓冲区，用于存储输入特征的均值和标准差
        self.register_buffer("mean", torch.zeros(config.model_in_dim))
        self.register_buffer("scale", torch.ones(config.model_in_dim))

        # 调用初始化权重的方法，初始化各个层的权重
        self.post_init()

    def _init_weights(self, module):
        """初始化权重的方法."""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # 对线性层和卷积层的权重进行正态分布初始化
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                # 如果存在偏置项，则将偏置项初始化为零
                module.bias.data.zero_()

    def apply_weight_norm(self):
        # 对预处理卷积层和所有上采样层应用权重归一化
        nn.utils.weight_norm(self.conv_pre)
        for layer in self.upsampler:
            nn.utils.weight_norm(layer)
        # 对所有残差块应用权重归一化
        for layer in self.resblocks:
            layer.apply_weight_norm()
        # 对后处理卷积层应用权重归一化
        nn.utils.weight_norm(self.conv_post)

    def remove_weight_norm(self):
        # 移除预处理卷积层和所有上采样层的权重归一化
        nn.utils.remove_weight_norm(self.conv_pre)
        for layer in self.upsampler:
            nn.utils.remove_weight_norm(layer)
        # 移除所有残差块的权重归一化
        for layer in self.resblocks:
            layer.remove_weight_norm()
        # 移除后处理卷积层的权重归一化
        nn.utils.remove_weight_norm(self.conv_post)
    def forward(self, spectrogram: torch.FloatTensor) -> torch.FloatTensor:
        r"""
        Converts a log-mel spectrogram into a speech waveform. Passing a batch of log-mel spectrograms returns a batch
        of speech waveforms. Passing a single, un-batched log-mel spectrogram returns a single, un-batched speech
        waveform.

        Args:
            spectrogram (`torch.FloatTensor`):
                Tensor containing the log-mel spectrograms. Can be batched and of shape `(batch_size, sequence_length,
                config.model_in_dim)`, or un-batched and of shape `(sequence_length, config.model_in_dim)`.

        Returns:
            `torch.FloatTensor`: Tensor containing the speech waveform. If the input spectrogram is batched, will be of
            shape `(batch_size, num_frames,)`. If un-batched, will be of shape `(num_frames,)`.
        """
        # 如果需要在前处理阶段进行归一化，则对输入的频谱图进行归一化处理
        if self.config.normalize_before:
            spectrogram = (spectrogram - self.mean) / self.scale

        # 检查输入的频谱图是否是批量数据
        is_batched = spectrogram.dim() == 3
        if not is_batched:
            # 如果输入不是批量数据，则在第0维度上增加一个维度，使其变成批量数据
            spectrogram = spectrogram.unsqueeze(0)

        # 将频谱图的维度进行转置，以符合卷积层的输入要求
        hidden_states = spectrogram.transpose(2, 1)

        # 经过预处理的卷积层
        hidden_states = self.conv_pre(hidden_states)

        # 循环执行上采样操作
        for i in range(self.num_upsamples):
            # 应用 LeakyReLU 激活函数
            hidden_states = nn.functional.leaky_relu(hidden_states, self.config.leaky_relu_slope)
            # 执行上采样操作
            hidden_states = self.upsampler[i](hidden_states)

            # 执行残差块操作
            res_state = self.resblocks[i * self.num_kernels](hidden_states)
            for j in range(1, self.num_kernels):
                res_state += self.resblocks[i * self.num_kernels + j](hidden_states)
            # 对残差块结果进行均值处理
            hidden_states = res_state / self.num_kernels

        # 应用 LeakyReLU 激活函数
        hidden_states = nn.functional.leaky_relu(hidden_states)
        # 经过后处理的卷积层
        hidden_states = self.conv_post(hidden_states)
        # 应用 Tanh 激活函数，将输出范围限制在 [-1, 1]
        hidden_states = torch.tanh(hidden_states)

        if not is_batched:
            # 如果输入不是批量数据，则去除批量维度，并将张量展平为一维音频波形
            waveform = hidden_states.squeeze(0).transpose(1, 0).view(-1)
        else:
            # 如果输入是批量数据，则去除序列长度维度，使其变为一维
            waveform = hidden_states.squeeze(1)

        return waveform
# 为 FastSpeech2ConformerWithHifiGan 类添加文档字符串，描述其作为一个文本到语音模型（生成波形）的 FastSpeech2ConformerHifiGan 语音合成器。
@add_start_docstrings(
    "The FastSpeech2ConformerModel with a FastSpeech2ConformerHifiGan vocoder head that performs text-to-speech (waveform).",
    FASTSPEECH2_CONFORMER_WITH_HIFIGAN_START_DOCSTRING,
)
class FastSpeech2ConformerWithHifiGan(PreTrainedModel):
    # 指定配置类为 FastSpeech2ConformerWithHifiGanConfig
    config_class = FastSpeech2ConformerWithHifiGanConfig

    # 初始化方法，接受一个 FastSpeech2ConformerWithHifiGanConfig 类型的 config 参数
    def __init__(self, config: FastSpeech2ConformerWithHifiGanConfig):
        # 调用父类的初始化方法，传入 config 参数
        super().__init__(config)

        # 创建 FastSpeech2ConformerModel 模型对象，使用 config.model_config 进行配置
        self.model = FastSpeech2ConformerModel(config.model_config)
        # 创建 FastSpeech2ConformerHifiGan 语音合成器对象，使用 config.vocoder_config 进行配置
        self.vocoder = FastSpeech2ConformerHifiGan(config.vocoder_config)

        # 将 config 参数保存为实例属性
        self.config = config

    # 重写 forward 方法的文档字符串，指定输出类型为 FastSpeech2ConformerWithHifiGanOutput，配置类为 FastSpeech2ConformerWithHifiGanConfig
    @replace_return_docstrings(
        output_type=FastSpeech2ConformerWithHifiGanOutput, config_class=FastSpeech2ConformerWithHifiGanConfig
    )
    # 前向传播方法，接受多个输入参数，所有参数都是 torch 张量类型，有些参数可以为空
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        spectrogram_labels: Optional[torch.FloatTensor] = None,
        duration_labels: Optional[torch.LongTensor] = None,
        pitch_labels: Optional[torch.FloatTensor] = None,
        energy_labels: Optional[torch.FloatTensor] = None,
        speaker_ids: Optional[torch.LongTensor] = None,
        lang_ids: Optional[torch.LongTensor] = None,
        speaker_embedding: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        # 以下参数没有在声明中列出，但在使用时会根据需要传入
        **kwargs,
    ):
        # 省略了具体的前向传播逻辑，需要在实际代码中查看
        pass
```