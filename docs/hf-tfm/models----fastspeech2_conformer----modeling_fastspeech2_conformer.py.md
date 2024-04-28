# `.\models\fastspeech2_conformer\modeling_fastspeech2_conformer.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache License, Version 2.0 进行授权
# 可以在 http://www.apache.org/licenses/LICENSE-2.0 获取授权副本
# 除了遵守许可证外，不得使用这个文件
# 除非法律要求或经书面同意，否则不得分发
# 分发的文件基于"原样"分发，没有任何明示或暗示的担保或条件
# 有关具体语言控制权限以及受限制的限制，请参阅许可证
""" PyTorch FastSpeech2Conformer model."""

# 导入必要的库
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn

# 导入模型输出、预训练模型等
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, logging, replace_return_docstrings
from .configuration_fastspeech2_conformer import (
    FastSpeech2ConformerConfig,
    FastSpeech2ConformerHifiGanConfig,
    FastSpeech2ConformerWithHifiGanConfig,
)

# 获取日志记录器
logger = logging.get_logger(__name__)

# FastSpeech2Conformer 模型的预训练模型存档列表
FASTSPEECH2_CONFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "espnet/fastspeech2_conformer",
    # 可在 https://huggingface.co/models?filter=fastspeech2_conformer 查看所有 FastSpeech2Conformer 模型
]


@dataclass
class FastSpeech2ConformerModelOutput(ModelOutput):
    """
    Output type of [`FastSpeech2ConformerModel`].
    # loss为生成频谱图的损失，类型为torch.FloatTensor，形状为(1,)，当提供了`labels`时返回
    loss: Optional[torch.FloatTensor] = None
    
    # spectrogram为预测的频谱图，类型为torch.FloatTensor，形状为(batch_size, sequence_length, num_bins)
    spectrogram: torch.FloatTensor = None

    # encoder_last_hidden_state为模型编码器最后一层的隐藏状态，类型为torch.FloatTensor，形状为(batch_size, sequence_length, hidden_size)，可选参数
    encoder_last_hidden_state: torch.FloatTensor = None

    # encoder_hidden_states为编码器每层的隐藏状态的元组，类型为tuple(torch.FloatTensor)，形状为(batch_size, sequence_length, hidden_size)，在`output_hidden_states=True`时返回
    encoder_hidden_states: tuple(torch.FloatTensor) = None

    # encoder_attentions为编码器每层的注意力权重的元组，类型为tuple(torch.FloatTensor)，形状为(batch_size, num_heads, sequence_length, sequence_length)，在`output_attentions=True`时返回
    encoder_attentions: tuple(torch.FloatTensor) = None

    # decoder_hidden_states为解码器每层的隐藏状态的元组，类型为tuple(torch.FloatTensor)，形状为(batch_size, sequence_length, hidden_size)，在`output_hidden_states=True`时返回
    decoder_hidden_states: tuple(torch.FloatTensor) = None

    # decoder_attentions为解码器每层的注意力权重的元组，类型为tuple(torch.FloatTensor)，形状为(batch_size, num_heads, sequence_length, sequence_length)，在`output_attentions=True`时返回
    decoder_attentions: tuple(torch.FloatTensor) = None

    # duration_outputs为持续时间预测器的输出，类型为torch.LongTensor，形状为(batch_size, max_text_length + 1)，可选参数
    duration_outputs: torch.LongTensor = None

    # pitch_outputs为音高预测器的输出，类型为torch.FloatTensor，形状为(batch_size, max_text_length + 1, 1)，可选参数
    pitch_outputs: torch.FloatTensor = None

    # energy_outputs为能量预测器的输出，类型为torch.FloatTensor，形状为(batch_size, max_text_length + 1, 1)，可选参数
    energy_outputs: torch.FloatTensor = None
    # 初始化编码器最后隐藏状态，默认为None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 初始化编码器隐藏状态，默认为None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 初始化编码器注意力权重，默认为None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 初始化解码器隐藏状态，默认为None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 初始化解码器注意力权重，默认为None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 初始化持续时间输出，数据类型为长整型，默认为None
    duration_outputs: torch.LongTensor = None
    # 初始化音高输出，数据类型为浮点型，默认为None
    pitch_outputs: torch.FloatTensor = None
    # 初始化能量输出，数据类型为浮点型，默认为None
    energy_outputs: torch.FloatTensor = None
@dataclass
class FastSpeech2ConformerWithHifiGanOutput(FastSpeech2ConformerModelOutput):
    """
    Output type of [`FastSpeech2ConformerWithHifiGan`].
    """

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
        # 根据输入的语速调整预测的持续时间
        duration_labels = torch.round(duration_labels.float() * speaking_speed).long()

    if duration_labels.sum() == 0:
        # 如果持续时间的总和为0，则将其设置为1
        duration_labels[duration_labels.sum(dim=1).eq(0)] = 1

    # 计算所需的最大长度
    max_len = torch.sum(duration_labels, dim=1).max()

    # 创建一个填充的张量来保存结果
    hidden_states = torch.zeros(
        (encoded_embeddings.size(0), max_len, encoded_embeddings.size(2)),
        dtype=torch.float,
        device=encoded_embeddings.device,
    )

    # 遍历批次并填充数据
    for i, (encoded_embedding, target_duration) in enumerate(zip(encoded_embeddings, duration_labels)):
        # 根据目标持续时间重复特征
        repeated = torch.repeat_interleave(encoded_embedding, target_duration, dim=0)
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
        # 调用父类的构造函数
        super().__init__()

        # 初始化卷积层列表
        self.conv_layers = nn.ModuleList()
        # 设置对数域的偏移量
        self.log_domain_offset = 1.0

        # 循环创建并添加预测器层到卷积层列表
        for layer_idx in range(config.duration_predictor_layers):
            # 获取当前层的通道数
            num_chans = config.duration_predictor_channels
            # 获取输入通道数，如果是第一层，则为 hidden_size，否则为 num_chans
            input_channels = config.hidden_size if layer_idx == 0 else num_chans
            # 创建 FastSpeech2ConformerPredictorLayer 层
            layer = FastSpeech2ConformerPredictorLayer(
                input_channels,
                num_chans,
                config.duration_predictor_kernel_size,
                config.duration_predictor_dropout_rate,
            )
            # 添加到卷积层列表
            self.conv_layers.append(layer)
        # 创建线性层
        self.linear = nn.Linear(config.duration_predictor_channels, 1)

    def forward(self, encoder_hidden_states):
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, max_text_length, input_dim)`):
                Batch of input sequences.
            padding_masks (`torch.ByteTensor` of shape `(batch_size, max_text_length)`, *optional*):
                Batch of masks indicating padded part.

        Returns:
            `torch.Tensor`: Batch of predicted durations in log domain `(batch_size, max_text_length)`.

        """
        # 将输入张量进行维度转置，从(batch_size, max_text_length, input_dim)变为(batch_size, input_dim, max_text_length)
        hidden_states = encoder_hidden_states.transpose(1, -1)
        # 循环遍历卷积层并计算输出
        for layer in self.conv_layers:
            hidden_states = layer(hidden_states)

        # NOTE: 在对数域中计算，输出形状为(batch_size, max_text_length)
        hidden_states = self.linear(hidden_states.transpose(1, -1)).squeeze(-1)

        # 如果不是训练阶段
        if not self.training:
            # NOTE: 在线性域中计算
            hidden_states = torch.clamp(torch.round(hidden_states.exp() - self.log_domain_offset), min=0).long()

        # 返回隐藏状态
        return hidden_states
# 从 transformers.models.speecht5.modeling_speecht5.SpeechT5BatchNormConvLayer 复制的 FastSpeech2ConformerBatchNormConvLayer 类
class FastSpeech2ConformerBatchNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()

        if layer_id == 0:
            in_conv_dim = config.num_mel_bins
        else:
            in_conv_dim = config.speech_decoder_postnet_units

        if layer_id == config.speech_decoder_postnet_layers - 1:
            out_conv_dim = config.num_mel_bins
        else:
            out_conv_dim = config.speech_decoder_postnet_units

        # 定义一维卷积层
        self.conv = nn.Conv1d(
            in_conv_dim,
            out_conv_dim,
            kernel_size=config.speech_decoder_postnet_kernel,
            stride=1,
            padding=(config.speech_decoder_postnet_kernel - 1) // 2,
            bias=False,
        )
        # 一维批归一化层
        self.batch_norm = nn.BatchNorm1d(out_conv_dim)

        if layer_id < config.speech_decoder_postnet_layers - 1:
            self.activation = nn.Tanh()
        else:
            self.activation = None

        # 随机失活层
        self.dropout = nn.Dropout(config.speech_decoder_postnet_dropout)

    def forward(self, hidden_states):
        # 进行卷积操作
        hidden_states = self.conv(hidden_states)
        # 进行批归一化操作
        hidden_states = self.batch_norm(hidden_states)
        if self.activation is not None:
            # 根据激活函数进行激活
            hidden_states = self.activation(hidden_states)
        # 进行随机失活操作
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class FastSpeech2ConformerSpeechDecoderPostnet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 线性层
        self.feat_out = nn.Linear(config.hidden_size, config.num_mel_bins * config.reduction_factor)
        # 创建 FastSpeech2ConformerBatchNormConvLayer 实例列表
        self.layers = nn.ModuleList(
            [FastSpeech2ConformerBatchNormConvLayer(config, i) for i in range(config.speech_decoder_postnet_layers)]
        )

    def forward(self, hidden_states: torch.Tensor):
        outputs_before_postnet = self.feat_out(hidden_states).view(hidden_states.size(0), -1, self.config.num_mel_bins)
        layer_output = outputs_before_postnet.transpose(1, 2)
        for layer in self.layers:
            layer_output = layer(layer_output)
        outputs_after_postnet = outputs_before_postnet + layer_output.transpose(1, 2)
        return outputs_before_postnet, outputs_after_postnet


class FastSpeech2ConformerPredictorLayer(nn.Module):
    def __init__(self, input_channels, num_chans, kernel_size, dropout_rate):
        super().__init__()
        # 一维卷积层
        self.conv = nn.Conv1d(
            input_channels,
            num_chans,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        # 激活函数
        self.activation = nn.ReLU()
        # 层归一化
        self.layer_norm = nn.LayerNorm(num_chans)
        # 随机失活层
        self.dropout = nn.Dropout(dropout_rate)
    # 传入隐藏状态，经过卷积操作后更新隐藏状态
    hidden_states = self.conv(hidden_states)
    # 经过激活函数处理隐藏状态
    hidden_states = self.activation(hidden_states)

    # 在第1维度上进行 layer norm 操作
    hidden_states = hidden_states.transpose(1, -1)
    hidden_states = self.layer_norm(hidden_states)
    hidden_states = hidden_states.transpose(1, -1)

    # 对隐藏状态进行 dropout 操作
    hidden_states = self.dropout(hidden_states)

    # 返回更新后的隐藏状态
    return hidden_states
# 定义一个名为FastSpeech2ConformerVariancePredictor的类，继承自nn.Module
class FastSpeech2ConformerVariancePredictor(nn.Module):
    # 初始化函数
    def __init__(
        self,
        config: FastSpeech2ConformerConfig,  # FastSpeech2ConformerConfig类型的参数，用于提供相关配置
        num_layers=2,  # 整型参数，卷积层的数量，默认为2
        num_chans=384,  # 整型参数，卷积层的通道数，默认为384
        kernel_size=3,  # 整型参数，卷积层的卷积核大小，默认为3
        dropout_rate=0.5,  # 浮点型参数，dropout的比例，默认为0.5
    ):
        """
        Initilize variance predictor module.

        Args:
            input_dim (`int`): Input dimension.
            num_layers (`int`, *optional*, defaults to 2): Number of convolutional layers.
            num_chans (`int`, *optional*, defaults to 384): Number of channels of convolutional layers.
            kernel_size (`int`, *optional*, defaults to 3): Kernel size of convolutional layers.
            dropout_rate (`float`, *optional*, defaults to 0.5): Dropout rate.
        """
        # 调用父类构造函数
        super().__init__()
        # 定义一个空的卷积层列表
        self.conv_layers = nn.ModuleList()
        # 遍历num_layers次
        for idx in range(num_layers):
            # 计算输入通道数
            input_channels = config.hidden_size if idx == 0 else num_chans
            # 创建一个FastSpeech2ConformerPredictorLayer实例，并添加到卷积层列表中
            layer = FastSpeech2ConformerPredictorLayer(input_channels, num_chans, kernel_size, dropout_rate)
            self.conv_layers.append(layer)
        # 定义一个线性层，输出维度为1
        self.linear = nn.Linear(num_chans, 1)

    # 前向计算函数
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
        # 将输入张量的维度进行转置
        # (batch_size, input_dim, max_text_length)
        hidden_states = encoder_hidden_states.transpose(1, -1)
        # 遍历卷积层列表
        for layer in self.conv_layers:
            # 调用卷积层的前向计算函数
            hidden_states = layer(hidden_states)

        # 将隐藏状态的维度转置
        hidden_states = self.linear(hidden_states.transpose(1, 2))

        # 如果padding_masks不���空，则将hidden_states中对应的位置的值设为0
        if padding_masks is not None:
            hidden_states = hidden_states.masked_fill(padding_masks, 0.0)

        # 返回隐藏状态
        return hidden_states


class FastSpeech2ConformerVarianceEmbedding(nn.Module):
    # 初始化函数
    def __init__(
        self,
        in_channels=1,  # 整型参数，输入通道数，默认为1
        out_channels=384,  # 整型参数，输出通道数，默认为384
        kernel_size=1,  # 整型参数，卷积核大小，默认为1
        padding=0,  # 整型参数，padding大小，默认为0
        dropout_rate=0.0,  # 整型参数，默认为0.0
    ):
        # 调用父类构造函数
        super().__init__()
        # 定义一个一维卷积层
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        # 定义一个dropout层
        self.dropout = nn.Dropout(dropout_rate)

    # 前向计算函数
    def forward(self, hidden_states):
        # 将隐藏状态的维度进行转置
        hidden_states = hidden_states.transpose(1, 2)
        # 输入一维卷积层得到输出
        hidden_states = self.conv(hidden_states)
        # 经过dropout层
        hidden_states = self.dropout(hidden_states)
        # 再次将隐藏状态维度进行转置
        hidden_states = hidden_states.transpose(1, 2)
        # 返回隐藏状态
        return hidden_states


class FastSpeech2ConformerAttention(nn.Module):
    """
    Multi-Head attention layer with relative position encoding. Details can be found in
    """

    # 可忽略的模型，文档字符串为空
    # 定义一个类，实现 FastSpeech2ConformerAttention
    # 构造一个 FastSpeech2ConformerAttention 对象
    def __init__(self, config: FastSpeech2ConformerConfig, module_config):
        # 调用父类初始化方法
        super().__init__()
        # 假设 d_v 总是等于 dim_key
        # 初始化属性值
        self.num_heads = module_config["num_attention_heads"]
        self.hidden_size = config.hidden_size
        self.dim_key = self.hidden_size // self.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.linear_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_v = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_out = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(p=module_config["attention_dropout_rate"])

        # 用于位置编码的线性变换
        self.linear_pos = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        # 下面两个可学习的偏置用于矩阵 c 和矩阵 d
        # 参考论文 https://arxiv.org/abs/1901.02860 第 3.3 节
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.num_heads, self.head_dim))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.num_heads, self.head_dim))

    def shift_relative_position_tensor(self, pos_tensor):
        """
        Args:
            pos_tensor (torch.Tensor of shape (batch_size, head, time1, 2*time1-1)): Input tensor.
        """
        # 创建与输入张量形状相同的零张量进行填充
        zero_pad = torch.zeros((*pos_tensor.size()[:3], 1), device=pos_tensor.device, dtype=pos_tensor.dtype)
        # 在最后一维上连接零张量
        pos_tensor_padded = torch.cat([zero_pad, pos_tensor], dim=-1)

        # 改变张量的形状以便后续操作
        pos_tensor_padded = pos_tensor_padded.view(*pos_tensor.size()[:2], pos_tensor.size(3) + 1, pos_tensor.size(2))
        # 仅保留从0到time2的位置信息
        pos_tensor = pos_tensor_padded[:, :, 1:].view_as(pos_tensor)[:, :, :, : pos_tensor.size(-1) // 2 + 1]

        return pos_tensor

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
        # 为了使用 'SAME' 填充，kernel_size 应该是一个奇数
        channels = config.hidden_size
        kernel_size = module_config["kernel_size"]
        # 第一个卷积层，1x1 卷积，将通道数从 channels 扩展到 2*channels
        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels, kernel_size=1, stride=1, padding=0, bias=True)
        # 第二个卷积层，深度卷积，用于捕捉序列维度的信息
        self.depthwise_conv = nn.Conv1d(
            channels, channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2, groups=channels, bias=True
        )
        # 归一化层，用于标准化深度卷积的输出
        self.norm = nn.BatchNorm1d(channels)
        # 第三个卷积层，1x1 卷积，将通道数恢复到 channels
        self.pointwise_conv2 = nn.Conv1d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, hidden_states):
        """
        Compute convolution module.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch, time, channels)`): Input tensor.

        Returns:
            `torch.Tensor`: Output tensor of shape `(batch, time, channels)`.

        """
        # exchange the temporal dimension and the feature dimension
        # 交换时间维度和特征维度
        hidden_states = hidden_states.transpose(1, 2)

        # GLU mechanism, (batch_size, 2*channel, dim)
        # GLU 机制，将输入分为两个部分，乘以门控，然后相加
        hidden_states = self.pointwise_conv1(hidden_states)
        # (batch_size, channel, dim)
        # 维度压缩
        hidden_states = nn.functional.glu(hidden_states, dim=1)

        # 1D Depthwise Conv
        # 一维深度卷积
        hidden_states = self.depthwise_conv(hidden_states)
        # 标准化
        hidden_states = self.norm(hidden_states)

        # gating mechanism
        # 门控机制
        hidden_states = hidden_states * torch.sigmoid(hidden_states)

        # 第二个 1x1 卷积
        hidden_states = self.pointwise_conv2(hidden_states)

        return hidden_states.transpose(1, 2)


class FastSpeech2ConformerEncoderLayer(nn.Module):
```  
    # 初始化方法，接受两个参数：配置对象和模块配置
    def __init__(self, config: FastSpeech2ConformerConfig, module_config):
        # 调用父类的初始化方法
        super().__init__()

        # 定义自注意力模块
        self.self_attn = FastSpeech2ConformerAttention(config, module_config)

        # 定义前馈模块
        self.feed_forward = FastSpeech2ConformerMultiLayeredConv1d(config, module_config)

        # 是否使用马卡龙风格
        self.macaron_style = config.use_macaron_style_in_conformer
        if self.macaron_style:
            # 如果使用马卡龙风格，定义前馈马卡龙模块
            self.feed_forward_macaron = FastSpeech2ConformerMultiLayeredConv1d(config, module_config)
            # 定义前馈马卡龙层归一化
            self.ff_macaron_layer_norm = nn.LayerNorm(config.hidden_size)
            # 前馈缩放比例为0.5
            self.ff_scale = 0.5
        else:
            # 前馈缩放比例为1.0
            self.ff_scale = 1.0

        # 是否使用卷积模块
        self.use_cnn_module = config.use_cnn_in_conformer
        if self.use_cnn_module:
            # 如果使用卷积模块，定义卷积模块
            self.conv_module = FastSpeech2ConformerConvolutionModule(config, module_config)
            # 定义卷积层归一化
            self.conv_layer_norm = nn.LayerNorm(config.hidden_size)
            # 定义最终层归一化
            self.final_layer_norm = nn.LayerNorm(config.hidden_size)

        # 定义前馈层归一化
        self.ff_layer_norm = nn.LayerNorm(config.hidden_size)

        # 定义自注意力层归一化
        self.self_attn_layer_norm = nn.LayerNorm(config.hidden_size)

        # 定义丢弃层
        self.dropout = nn.Dropout(module_config["dropout_rate"])
        # 定义大小
        self.size = config.hidden_size
        # 是否在之前归一化
        self.normalize_before = module_config["normalize_before"]
        # 是否在之后连接
        self.concat_after = module_config["concat_after"]
        if self.concat_after:
            # 如果在之后连接，定义连接线性层
            self.concat_linear = nn.Linear(config.hidden_size + config.hidden_size, config.hidden_size)

    # 正向传播方法，接受多个参数
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
            input_channels (`int`): Number of input channels.
            hidden_channels (`int`): Number of hidden channels.
            kernel_size (`int`): Kernel size of conv1d.
            dropout_rate (`float`): Dropout rate.
        """
        # 初始化函数
        super().__init__()
        # 从 FastSpeech2ConformerConfig 中获取隐藏层的尺寸作为输入通道数
        input_channels = config.hidden_size
        # 从 module_config 中获取线性单元数作为隐藏通道数
        hidden_channels = module_config["linear_units"]
        # 从 FastSpeech2ConformerConfig 中获取位置卷积核的大小
        kernel_size = config.positionwise_conv_kernel_size
        # 创建第一个卷积层，输入通道数为 input_channels，输出通道数为 hidden_channels
        # 卷积核大小为 kernel_size，填充方式为对称填充
        self.conv1 = nn.Conv1d(input_channels, hidden_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        # 创建第二个卷积层，输入通道数为 hidden_channels，输出通道数为 input_channels
        # 卷积核大小为 kernel_size，填充方式为对称填充
        self.conv2 = nn.Conv1d(hidden_channels, input_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        # 创建 dropout 层，使用 module_config 中的 dropout_rate
        self.dropout = nn.Dropout(module_config["dropout_rate"])

    def forward(self, hidden_states):
        """
        Calculate forward propagation.

        Args:
            hidden_states (torch.Tensor): Batch of input tensors (batch_size, time, input_channels).

        Returns:
            torch.Tensor: Batch of output tensors (batch_size, time, hidden_channels).
        """
        # 将输入的 hidden_states 调整维度顺序，转换成(batch_size, input_channels, time)
        hidden_states = hidden_states.transpose(-1, 1)
        # 通过第一个卷积层进行卷积操作
        hidden_states = self.conv1(hidden_states)
        # 使用 ReLU 激活函数
        hidden_states = torch.relu(hidden_states)
        # 对卷积结果进行 dropout 操作
        hidden_states = self.dropout(hidden_states)
        # 通过第二个卷积层进行卷积操作
        hidden_states = self.conv2(hidden_states)
        # 调整维度顺序，转换回(batch_size, time, hidden_channels)
        hidden_states = hidden_states.transpose(-1, 1)
        # 返回卷积后的结果
        return hidden_states


class FastSpeech2ConformerRelPositionalEncoding(nn.Module):
    """
    Args:
    Relative positional encoding module (new implementation). Details can be found in
    https://github.com/espnet/espnet/pull/2816. See : Appendix Batch in https://arxiv.org/abs/1901.02860
        config (`FastSpeech2ConformerConfig`):
            FastSpeech2ConformerConfig instance.
        module_config (`dict`):
            Dictionary containing the encoder or decoder module configuration from the `FastSpeech2ConformerConfig`.
    """

    def __init__(self, config: FastSpeech2ConformerConfig, module_config):
        """
        Construct an PositionalEncoding object.
        """
        # 初始化函数
        super().__init__()
        # 获取隐藏层尺寸作为嵌入维度
        self.embed_dim = config.hidden_size
        # 计算输入的缩放因子，以用于位置编码
        self.input_scale = math.sqrt(self.embed_dim)
        # 创建 dropout 层，使用 module_config 中的 positional_dropout_rate
        self.dropout = nn.Dropout(p=module_config["positional_dropout_rate"])
        # 初始化位置编码为 None
        self.pos_enc = None
        # 设定最大序列长度
        self.max_len = 5000
        # 扩展位置编码，用于处理长序列
        self.extend_pos_enc(torch.tensor(0.0).expand(1, self.max_len))
    def extend_pos_enc(self, x):
        """Reset the positional encodings."""
        # 检查是否已存在位置编码
        if self.pos_enc is not None:
            # 若位置编码已存在，则包含了正负两个部分
            # 位置编码长度为 2 * 输入长度 - 1
            if self.pos_enc.size(1) >= x.size(1) * 2 - 1:
                # 若位置编码的长度足够长，与输入长度相符或更长，则不做任何操作
                # 检查位置编码的数据类型和设备是否与输入相匹配，不匹配则转换
                if self.pos_enc.dtype != x.dtype or self.pos_enc.device != x.device:
                    self.pos_enc = self.pos_enc.to(dtype=x.dtype, device=x.device)
                return
        # 若位置编码不存在或长度不足，则需要重新生成
        # 假设 `i` 表示查询向量的位置，`j` 表示键向量的位置。当键位于左侧时（i>j），使用正相对位置，否则使用负相对位置（i<j）。
        # 初始化正相对位置编码和负相对位置编码
        pos_enc_positive = torch.zeros(x.size(1), self.embed_dim)
        pos_enc_negative = torch.zeros(x.size(1), self.embed_dim)
        # 计算位置向量
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        # 计算除数项，用于计算正弦和余弦值
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2, dtype=torch.float32) * -(math.log(10000.0) / self.embed_dim)
        )
        # 计算正弦和余弦值并填充到相应位置
        pos_enc_positive[:, 0::2] = torch.sin(position * div_term)
        pos_enc_positive[:, 1::2] = torch.cos(position * div_term)
        pos_enc_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pos_enc_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # 颠倒正相对位置编码的顺序并连接正负两部分位置编码
        # 用于支持类似于 https://arxiv.org/abs/1901.02860 中描述的偏移技巧
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
        # 扩展位置编码以适应输入长度
        self.extend_pos_enc(feature_representation)
        # 将特征表示乘以输入缩放因子
        hidden_states = feature_representation * self.input_scale
        # 计算中心索引以在位置编码中截取相应长度的位置编码
        center_idx = self.pos_enc.size(1) // 2
        pos_emb = self.pos_enc[:, center_idx - hidden_states.size(1) + 1 : center_idx + hidden_states.size(1)]
        # 返回经过 dropout 处理的隐藏状态和位置编码
        return self.dropout(hidden_states), self.dropout(pos_emb)
# 定义 FastSpeech2ConformerEncoder 类，用于实现 FastSpeech 2 模型的编码器部分
class FastSpeech2ConformerEncoder(nn.Module):
    """
    FastSpeech2ConformerEncoder encoder module.

    Args:
        config (`FastSpeech2ConformerConfig`):
            FastSpeech2ConformerConfig instance.
        module_config (`dict`):
            Dictionary containing the encoder or decoder module configuration from the `FastSpeech2ConformerConfig`.
        use_encoder_input_layer (`bool`, *optional*, defaults to `False`):
            Input layer type.
    """

    # 初始化函数，设置模型参数和各个层
    def __init__(
        self,
        config: FastSpeech2ConformerConfig,
        module_config,
        use_encoder_input_layer=False,
    ):
        super().__init__()

        # 如果使用编码器输入层，则创建嵌入层
        self.embed = None
        if use_encoder_input_layer:
            self.embed = nn.Embedding(
                num_embeddings=config.vocab_size, embedding_dim=config.hidden_size, padding_idx=0
            )

        # 创建相对位置编码层
        self.pos_enc = FastSpeech2ConformerRelPositionalEncoding(config, module_config)

        # 创建多个 Conformer 编码器层
        self.conformer_layers = nn.ModuleList(
            [FastSpeech2ConformerEncoderLayer(config, module_config) for _ in range(module_config["layers"])]
        )

    # 前向传播函数，接受输入张量和其他参数，返回编码器部分的输出
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
                输入序列标记在词汇表中的索引。默认情况下，提供填充（padding）将被忽略。

                可以使用 [`AutoTokenizer`] 获取索引。有关详细信息，请参阅 [`PreTrainedTokenizer.encode`] 和
                [`PreTrainedTokenizer.__call__`]。

                [什么是输入 ID？](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *可选*):
                避免对填充标记索引执行注意力的掩码。掩码值选取在 `[0, 1]` 范围内：

                - 对于**不被掩码**的标记，为 1，
                - 对于**被掩码**的标记，为 0。

                [什么是注意力掩码？](../glossary#attention-mask)
            output_hidden_states (`bool`, *可选*):
                是否返回所有层的隐藏状态。有关更多细节，请参阅返回张量中的 `hidden_states`。
            output_attentions (`bool`, *可选*):
                是否返回所有注意力层的注意力张量。有关更多细节，请参阅返回张量中的 `attentions`。
            return_dict (`bool`, *可选*):
                是否返回一个 [`~utils.ModelOutput`] 而不是普通元组。
        Returns:
            `torch.Tensor`:
                形状为 `(batch, time, attention_dim)` 的输出张量。
        """
        # 特征表示初始化为输入张量
        feature_representation = input_tensor
        # 如果存在嵌入层，则使用嵌入层对特征表示进行转换
        if self.embed is not None:
            feature_representation = self.embed(feature_representation)

        # 应用位置编码层
        hidden_states, pos_emb = self.pos_enc(feature_representation)

        # 初始化所有隐藏状态和自注意力张量
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # 遍历所有的Conformer层
        for conformer_layer in self.conformer_layers:
            # 如果需要输出隐藏状态，则添加当前隐藏状态到列表中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 调用Conformer层的前向传播
            layer_outputs = conformer_layer(hidden_states, pos_emb, attention_mask, output_attentions)
            hidden_states = layer_outputs[0]

            # 如果需要输出自注意力张量，则添加当前层的自注意力张量到列表中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 添加最后一层的隐藏状态到列表中（如果需要输出隐藏状态）
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要以字典形式返回结果，则返回对应的元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 否则以字典形式返回结果
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions
        )
```  
# 定义一个计算 FastSpeech2Conformer 模型损失的类
class FastSpeech2ConformerLoss(nn.Module):
    def __init__(self, config: FastSpeech2ConformerConfig):
        super().__init__()

        # 从配置中获取是否使用遮罩和加权遮罩的标志
        use_masking = config.use_masking
        use_weighted_masking = config.use_weighted_masking

        # 如果同时使用遮罩和加权遮罩，则抛出异常
        if use_masking and use_weighted_masking:
            raise ValueError("Either use_masking or use_weighted_masking can be True, but not both.")

        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking

        # 定义损失函数
        reduction = "none" if self.use_weighted_masking else "mean"
        self.l1_criterion = nn.L1Loss(reduction=reduction)
        self.mse_criterion = nn.MSELoss(reduction=reduction)
        self.duration_criterion = nn.MSELoss(reduction=reduction)
        self.log_domain_offset = 1.0

    # 前向传播函数，计算模型的损失
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
        
# FastSpeech2Conformer 预训练模型的父类
class FastSpeech2ConformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 定义 FastSpeech2Conformer 的配置类和模型名称前缀
    config_class = FastSpeech2ConformerConfig
    base_model_prefix = "fastspeech2_conformer"

    main_input_name = "input_ids"

    # 初始化权重函数
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.LayerNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                key = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-key, b=key)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_()
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, FastSpeech2ConformerAttention):
            nn.init.xavier_uniform_(module.pos_bias_u)
            nn.init.xavier_uniform_(module.pos_bias_v)

    # 设置梯度检查点函数
    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, FastSpeech2ConformerEncoder):
            module.gradient_checkpointing = value

# FastSpeech2Conformer 模型类，继承自 FastSpeech2ConformerPreTrainedModel
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
    FastPitch: Parallel Text-to-speech with Pitch Prediction. The encoder and decoder are Conformers instead of regular Transformers.
    """

    @replace_return_docstrings(output_type=FastSpeech2ConformerModelOutput, config_class=_CONFIG_FOR_DOC)
    # 定义前向传播方法
    def forward(
        # 输入的 token IDs
        input_ids: torch.LongTensor,
        # 注意力掩码
        attention_mask: Optional[torch.LongTensor] = None,
        # 频谱标签
        spectrogram_labels: Optional[torch.FloatTensor] = None,
        # 语音持续时间标签
        duration_labels: Optional[torch.LongTensor] = None,
        # 语音音高标签
        pitch_labels: Optional[torch.FloatTensor] = None,
        # 语音能量标签
        energy_labels: Optional[torch.FloatTensor] = None,
        # 说话者 IDs
        speaker_ids: Optional[torch.LongTensor] = None,
        # 语言 IDs
        lang_ids: Optional[torch.LongTensor] = None,
        # 说话者嵌入向量
        speaker_embedding: Optional[torch.FloatTensor] = None,
        # 是否返回字典输出
        return_dict: Optional[bool] = None,
        # 是否输出注意力权重
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态
        output_hidden_states: Optional[bool] = None,
# 定义 HifiGanResidualBlock 类，继承自 nn.Module
class HifiGanResidualBlock(nn.Module):
    # 初始化方法，接受 channels、kernel_size、dilation 和 leaky_relu_slope 四个参数
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), leaky_relu_slope=0.1):
        # 调用父类的初始化方法
        super().__init__()
        # 将 leaky_relu_slope 参数保存到实例属性中
        self.leaky_relu_slope = leaky_relu_slope

        # 创建一系列卷积层，放入 ModuleList 中
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

    # 计算 padding 的辅助方法
    def get_padding(self, kernel_size, dilation=1):
        return (kernel_size * dilation - dilation) // 2

    # 添加权重归一化的方法
    def apply_weight_norm(self):
        for layer in self.convs1:
            nn.utils.weight_norm(layer)
        for layer in self.convs2:
            nn.utils.weight_norm(layer)

    # 移除权重归一化的方法
    def remove_weight_norm(self):
        for layer in self.convs1:
            nn.utils.remove_weight_norm(layer)
        for layer in self.convs2:
            nn.utils.remove_weight_norm(layer)

    # 前向传播方法，接受 hidden_states 参数
    def forward(self, hidden_states):
        # 遍历两个 ModuleList 中的卷积层，对 hidden_states 进行卷积操作和残差连接
        for conv1, conv2 in zip(self.convs1, self.convs2):
            # 保存残差连接的值
            residual = hidden_states
            # 使用 LeakyReLU 激活函数
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            # 第一个卷积层
            hidden_states = conv1(hidden_states)
            # 使用 LeakyReLU 激活函数
            hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
            # 第二个卷积层
            hidden_states = conv2(hidden_states)
            # 残差连接
            hidden_states = hidden_states + residual
        # 返回处理后的 hidden_states
        return hidden_states

# 添加文档字符串
@add_start_docstrings(
    """HiFi-GAN vocoder.""",
    HIFIGAN_START_DOCSTRING,
)
# 定义 FastSpeech2ConformerHifiGan 类，继承自 PreTrainedModel
class FastSpeech2ConformerHifiGan(PreTrainedModel):
    # 指定配置类
    config_class = FastSpeech2ConformerHifiGanConfig
    # 主要输入的名称为 "spectrogram"
    main_input_name = "spectrogram"
    # 初始化方法，接受一个 FastSpeech2ConformerHifiGanConfig 类型的参数
    def __init__(self, config: FastSpeech2ConformerHifiGanConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 计算卷积层的数量
        self.num_kernels = len(config.resblock_kernel_sizes)
        # 计算上采样层的数量
        self.num_upsamples = len(config.upsample_rates)
        # 创建一个一维卷积层作为预处理器
        self.conv_pre = nn.Conv1d(
            config.model_in_dim,  # 输入通道数
            config.upsample_initial_channel,  # 输出通道数
            kernel_size=7,  # 卷积核大小
            stride=1,  # 步长
            padding=3,  # 填充
        )

        # 创建一个上采样模块列表
        self.upsampler = nn.ModuleList()
        # 遍历上采样率和卷积核大小的组合
        for i, (upsample_rate, kernel_size) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            # 添加一个一维转置卷积层到上采样模块列表中
            self.upsampler.append(
                nn.ConvTranspose1d(
                    config.upsample_initial_channel // (2**i),  # 输入通道数
                    config.upsample_initial_channel // (2 ** (i + 1)),  # 输出通道数
                    kernel_size=kernel_size,  # 卷积核大小
                    stride=upsample_rate,  # 步长
                    padding=(kernel_size - upsample_rate) // 2,  # 填充
                )
            )

        # 创建一个残差块模块列表
        self.resblocks = nn.ModuleList()
        # 遍历上采样模块列表的长度
        for i in range(len(self.upsampler)):
            channels = config.upsample_initial_channel // (2 ** (i + 1))  # 计算通道数
            # 遍历残差块的卷积核大小和膨胀率
            for kernel_size, dilation in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
                # 添加一个 HifiGanResidualBlock 到残差块模块列表中
                self.resblocks.append(HifiGanResidualBlock(channels, kernel_size, dilation, config.leaky_relu_slope))

        # 创建一个一维卷积层作为后处理器
        self.conv_post = nn.Conv1d(channels, 1, kernel_size=7, stride=1, padding=3)

        # 将均值和标准差作为缓冲区注册到模型中
        self.register_buffer("mean", torch.zeros(config.model_in_dim))
        self.register_buffer("scale", torch.ones(config.model_in_dim))

        # 初始化权重并应用最终处理
        self.post_init()

    # 初始化权重的方法
    def _init_weights(self, module):
        """Initialize the weights."""
        # 如果模块是线性层或者一维卷积层
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # 初始化权重为正态分布
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置项，初始化偏置为零
            if module.bias is not None:
                module.bias.data.zero_()

    # 应用权重归一化的方法
    def apply_weight_norm(self):
        # 对预处理器应用权重归一化
        nn.utils.weight_norm(self.conv_pre)
        # 对每个上采样层应用权重归一化
        for layer in self.upsampler:
            nn.utils.weight_norm(layer)
        # 对每个残差块应用权重归一化
        for layer in self.resblocks:
            layer.apply_weight_norm()
        # 对后处理器应用权重归一化
        nn.utils.weight_norm(self.conv_post)

    # 移除权重归一化的方法
    def remove_weight_norm(self):
        # 移除预处理器的权重归一化
        nn.utils.remove_weight_norm(self.conv_pre)
        # 移除每个上采样层的权重归一化
        for layer in self.upsampler:
            nn.utils.remove_weight_norm(layer)
        # 移除每个残差块的权重归一化
        for layer in self.resblocks:
            layer.remove_weight_norm()
        # 移除后处理器的权重归一化
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
        # 如果需要进行归一化处理
        if self.config.normalize_before:
            # 对输入的 log-mel spectrogram 进行归一化处理
            spectrogram = (spectrogram - self.mean) / self.scale

        # 检查输入是否为批处理
        is_batched = spectrogram.dim() == 3
        # 如果不是批处理，则添加批处理维度
        if not is_batched:
            spectrogram = spectrogram.unsqueeze(0)

        # 将频谱转置，以适应卷积操作的维度要求
        hidden_states = spectrogram.transpose(2, 1)

        # 应用预卷积层
        hidden_states = self.conv_pre(hidden_states)
        # 循环进行上采样操作
        for i in range(self.num_upsamples):
            hidden_states = nn.functional.leaky_relu(hidden_states, self.config.leaky_relu_slope)
            hidden_states = self.upsampler[i](hidden_states)

            # 应用残差模块
            res_state = self.resblocks[i * self.num_kernels](hidden_states)
            for j in range(1, self.num_kernels):
                res_state += self.resblocks[i * self.num_kernels + j](hidden_states)
            # 计算残差均值
            hidden_states = res_state / self.num_kernels

        # 应用激活函数
        hidden_states = nn.functional.leaky_relu(hidden_states)
        # 应用后卷积层
        hidden_states = self.conv_post(hidden_states)
        # 应用 tanh 激活函数
        hidden_states = torch.tanh(hidden_states)

        # 如果输入不是批处理，则去除批处理维度并将张量展平为 1 维音频波形
        if not is_batched:
            waveform = hidden_states.squeeze(0).transpose(1, 0).view(-1)
        else:
            # 如果输入是批处理，则去除序列长度维度，因为它将折叠到 1
            waveform = hidden_states.squeeze(1)

        # 返回音频波形张量
        return waveform
# 引入所需库
@add_start_docstrings(
    "The FastSpeech2ConformerModel with a FastSpeech2ConformerHifiGan vocoder head that performs text-to-speech (waveform).",
    FASTSPEECH2_CONFORMER_WITH_HIFIGAN_START_DOCSTRING,
)
# 定义 FastSpeech2ConformerWithHifiGan 类，继承自 PreTrainedModel 类
class FastSpeech2ConformerWithHifiGan(PreTrainedModel):
    # 设置配置类为 FastSpeech2ConformerWithHifiGanConfig
    config_class = FastSpeech2ConformerWithHifiGanConfig

    # 初始化方法
    def __init__(self, config: FastSpeech2ConformerWithHifiGanConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建 FastSpeech2ConformerModel 模型对象
        self.model = FastSpeech2ConformerModel(config.model_config)
        # 创建 FastSpeech2ConformerHifiGan 语音合成器对象
        self.vocoder = FastSpeech2ConformerHifiGan(config.vocoder_config)

        # 保存配置
        self.config = config

    # 前向传播方法
    @replace_return_docstrings(
        output_type=FastSpeech2ConformerWithHifiGanOutput, config_class=FastSpeech2ConformerWithHifiGanConfig
    )
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
```