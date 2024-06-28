# `.\models\encodec\modeling_encodec.py`

```py
# coding=utf-8
# 版权 2023 Meta Platforms, Inc. 及其关联公司以及 HuggingFace Inc. 团队保留所有权利。
#
# 根据 Apache 许可证版本 2.0 （“许可证”）获得许可；
# 除非符合许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则依据许可证分发的软件是按“原样”分发的，
# 没有任何明示或暗示的保证或条件。
# 请参阅许可证了解具体语言的权限和限制。
""" PyTorch EnCodec model."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_encodec import EncodecConfig

logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "EncodecConfig"

ENCODEC_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/encodec_24khz",
    "facebook/encodec_48khz",
    # See all EnCodec models at https://huggingface.co/models?filter=encodec
]

@dataclass
class EncodecOutput(ModelOutput):
    """
    Args:
        audio_codes (`torch.LongTensor`  of shape `(batch_size, nb_chunks, chunk_length)`, *optional*):
            Discret code embeddings computed using `model.encode`.
        audio_values (`torch.FlaotTensor` of shape `(batch_size, sequence_length)`, *optional*)
            Decoded audio values, obtained using the decoder part of Encodec.
    """
    audio_codes: torch.LongTensor = None
    audio_values: torch.FloatTensor = None


@dataclass
class EncodecEncoderOutput(ModelOutput):
    """
    Args:
        audio_codes (`torch.LongTensor`  of shape `(batch_size, nb_chunks, chunk_length)`, *optional*):
            Discret code embeddings computed using `model.encode`.
        audio_scales (`torch.Tensor` of shape `(batch_size, nb_chunks)`, *optional*):
            Scaling factor for each `audio_codes` input. This is used to unscale each chunk of audio when decoding.
    """
    audio_codes: torch.LongTensor = None
    audio_scales: torch.FloatTensor = None


@dataclass
class EncodecDecoderOutput(ModelOutput):
    """
    Args:
        audio_values (`torch.FloatTensor`  of shape `(batch_size, segment_length)`, *optional*):
            Decoded audio values, obtained using the decoder part of Encodec.
    """
    audio_values: torch.FloatTensor = None


class EncodecConv1d(nn.Module):
    """Conv1d with asymmetric or causal padding and normalization."""

    def __init__(
        self, config, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 配置参数
        self.config = config
        # 输入通道数
        self.in_channels = in_channels
        # 输出通道数
        self.out_channels = out_channels
        # 卷积核大小
        self.kernel_size = kernel_size
        # 步长
        self.stride = stride
        # 膨胀率
        self.dilation = dilation
    ):
        super().__init__()  # 调用父类的构造函数初始化
        self.causal = config.use_causal_conv  # 设置是否使用因果卷积的配置
        self.pad_mode = config.pad_mode  # 设置填充模式的配置
        self.norm_type = config.norm_type  # 设置规范化类型的配置

        if self.norm_type not in ["weight_norm", "time_group_norm"]:
            raise ValueError(
                f'self.norm_type must be one of `"weight_norm"`, `"time_group_norm"`), got {self.norm_type}'
            )  # 如果规范化类型不在支持的列表中，抛出数值错误异常

        # warn user on unusual setup between dilation and stride
        if stride > 1 and dilation > 1:
            logger.warning(
                "EncodecConv1d has been initialized with stride > 1 and dilation > 1"
                f" (kernel_size={kernel_size} stride={stride}, dilation={dilation})."
            )  # 如果步长大于1且膨胀大于1，记录警告信息

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, dilation=dilation)
        if self.norm_type == "weight_norm":
            self.conv = nn.utils.weight_norm(self.conv)  # 如果使用权重规范化，对卷积层应用权重规范化
        elif self.norm_type == "time_group_norm":
            self.norm = nn.GroupNorm(1, out_channels)  # 如果使用时间组规范化，创建时间组规范化层

    @staticmethod
    def _get_extra_padding_for_conv1d(
        hidden_states: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
    ) -> int:
        """See `pad_for_conv1d`."""
        length = hidden_states.shape[-1]  # 获取隐藏状态的长度
        n_frames = (length - kernel_size + padding_total) / stride + 1  # 计算帧数
        ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)  # 计算理想长度
        return ideal_length - length  # 返回额外的填充长度

    @staticmethod
    def _pad1d(hidden_states: torch.Tensor, paddings: Tuple[int, int], mode: str = "zero", value: float = 0.0):
        """Tiny wrapper around torch.nn.functional.pad, just to allow for reflect padding on small input.
        If this is the case, we insert extra 0 padding to the right before the reflection happens.
        """
        length = hidden_states.shape[-1]  # 获取隐藏状态的长度
        padding_left, padding_right = paddings  # 解包填充值
        if not mode == "reflect":
            return nn.functional.pad(hidden_states, paddings, mode, value)  # 如果填充模式不是反射，则使用指定模式进行填充

        max_pad = max(padding_left, padding_right)  # 获取最大填充值
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1  # 计算额外填充长度
            hidden_states = nn.functional.pad(hidden_states, (0, extra_pad))  # 在反射填充前插入额外的0填充
        padded = nn.functional.pad(hidden_states, paddings, mode, value)  # 执行填充操作
        end = padded.shape[-1] - extra_pad  # 计算有效结束位置
        return padded[..., :end]  # 返回填充后的结果，截断额外填充部分
    # 定义一个前向传播函数，接受隐藏状态作为输入
    def forward(self, hidden_states):
        # 获取卷积层的核大小
        kernel_size = self.conv.kernel_size[0]
        # 获取卷积层的步幅
        stride = self.conv.stride[0]
        # 获取卷积层的扩张率
        dilation = self.conv.dilation[0]
        # 计算考虑扩张率后的有效核大小
        kernel_size = (kernel_size - 1) * dilation + 1  # effective kernel size with dilations
        # 计算总的填充量
        padding_total = kernel_size - stride
        # 调用方法计算额外的填充量
        extra_padding = self._get_extra_padding_for_conv1d(hidden_states, kernel_size, stride, padding_total)

        if self.causal:
            # 如果是因果卷积，进行左填充
            hidden_states = self._pad1d(hidden_states, (padding_total, extra_padding), mode=self.pad_mode)
        else:
            # 如果不是因果卷积，根据奇数步长要求进行非对称填充
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            hidden_states = self._pad1d(
                hidden_states, (padding_left, padding_right + extra_padding), mode=self.pad_mode
            )

        # 对隐藏状态应用卷积操作
        hidden_states = self.conv(hidden_states)

        if self.norm_type == "time_group_norm":
            # 如果指定使用时间组归一化，对隐藏状态进行归一化处理
            hidden_states = self.norm(hidden_states)

        # 返回处理后的隐藏状态
        return hidden_states
class EncodecConvTranspose1d(nn.Module):
    """ConvTranspose1d with asymmetric or causal padding and normalization."""

    def __init__(self, config, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        self.causal = config.use_causal_conv  # 是否使用因果卷积
        self.trim_right_ratio = config.trim_right_ratio  # 右侧修剪比例
        self.norm_type = config.norm_type  # 标准化类型
        if self.norm_type not in ["weight_norm", "time_group_norm"]:
            raise ValueError(
                f'self.norm_type must be one of `"weight_norm"`, `"time_group_norm"`), got {self.norm_type}'
            )

        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride)  # 定义反卷积层
        if config.norm_type == "weight_norm":
            self.conv = nn.utils.weight_norm(self.conv)  # 如果标准化类型是 weight_norm，则对卷积层应用 weight_norm
        elif config.norm_type == "time_group_norm":
            self.norm = nn.GroupNorm(1, out_channels)  # 如果标准化类型是 time_group_norm，则使用 GroupNorm

        if not (self.causal or self.trim_right_ratio == 1.0):
            raise ValueError("`trim_right_ratio` != 1.0 only makes sense for causal convolutions")

    def forward(self, hidden_states):
        kernel_size = self.conv.kernel_size[0]  # 获取卷积核大小
        stride = self.conv.stride[0]  # 获取卷积步长
        padding_total = kernel_size - stride  # 计算总的填充量

        hidden_states = self.conv(hidden_states)  # 执行反卷积操作

        if self.norm_type == "time_group_norm":
            hidden_states = self.norm(hidden_states)  # 如果使用 time_group_norm，则对隐藏状态进行标准化

        # 只修剪固定的填充。从 `pad_for_conv1d` 多余的填充将在输出时移除。
        # 在这里移除它们需要在匹配的层传递长度。
        if self.causal:
            # 根据指定的比例修剪右侧的填充
            # 如果 trim_right_ratio = 1.0，则从右侧全部修剪
            padding_right = math.ceil(padding_total * self.trim_right_ratio)
        else:
            # 对于奇数步长需要对称填充
            padding_right = padding_total // 2

        padding_left = padding_total - padding_right

        # 取消填充
        end = hidden_states.shape[-1] - padding_right
        hidden_states = hidden_states[..., padding_left:end]
        return hidden_states


class EncodecLSTM(nn.Module):
    """
    LSTM without worrying about the hidden state, nor the layout of the data. Expects input as convolutional layout.
    """

    def __init__(self, config, dimension):
        super().__init__()
        self.lstm = nn.LSTM(dimension, dimension, config.num_lstm_layers)  # 定义 LSTM 层

    def forward(self, hidden_states):
        hidden_states = hidden_states.permute(2, 0, 1)  # 调整输入的维度顺序
        hidden_states = self.lstm(hidden_states)[0] + hidden_states  # 执行 LSTM 操作并添加到原始输入
        hidden_states = hidden_states.permute(1, 2, 0)  # 调整输出的维度顺序
        return hidden_states


class EncodecResnetBlock(nn.Module):
    """
    Residual block from SEANet model as used by EnCodec.
    """
    # 初始化函数，用于初始化 EncodecBlock 类的实例
    def __init__(self, config: EncodecConfig, dim: int, dilations: List[int]):
        super().__init__()  # 调用父类的初始化方法

        # 根据配置参数和维度计算出卷积核大小的元组
        kernel_sizes = (config.residual_kernel_size, 1)

        # 检查卷积核大小的数量是否与 dilations 列表的长度相等，若不相等则抛出异常
        if len(kernel_sizes) != len(dilations):
            raise ValueError("Number of kernel sizes should match number of dilations")

        # 根据维度和压缩比例计算隐藏层的维度
        hidden = dim // config.compress
        block = []

        # 遍历卷积核大小和 dilations 列表，构建 EncodecBlock 的每个卷积层
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            # 计算当前卷积层的输入通道数和输出通道数
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden

            # 添加 ELU 激活函数层
            block += [nn.ELU()]
            # 添加 EncodecConv1d 卷积层
            block += [EncodecConv1d(config, in_chs, out_chs, kernel_size, dilation=dilation)]

        # 将 block 列表转换为 nn.ModuleList，并赋值给 self.block
        self.block = nn.ModuleList(block)

        # 根据配置参数决定是否使用卷积作为 shortcut
        if config.use_conv_shortcut:
            self.shortcut = EncodecConv1d(config, dim, dim, kernel_size=1)
        else:
            # 否则使用恒等映射作为 shortcut
            self.shortcut = nn.Identity()

    # 前向传播函数，用于计算 EncodecBlock 的前向传播结果
    def forward(self, hidden_states):
        residual = hidden_states  # 记录初始输入作为残差连接的基准

        # 遍历 self.block 中的每个层，依次对 hidden_states 进行前向传播计算
        for layer in self.block:
            hidden_states = layer(hidden_states)

        # 将残差连接的结果与 self.shortcut 计算的结果相加，并返回最终的前向传播结果
        return self.shortcut(residual) + hidden_states
class EncodecEncoder(nn.Module):
    """SEANet encoder as used by EnCodec."""

    def __init__(self, config: EncodecConfig):
        super().__init__()
        model = [EncodecConv1d(config, config.audio_channels, config.num_filters, config.kernel_size)]
        scaling = 1

        # Downsample to raw audio scale
        for ratio in reversed(config.upsampling_ratios):
            current_scale = scaling * config.num_filters
            # Add residual layers
            for j in range(config.num_residual_layers):
                model += [EncodecResnetBlock(config, current_scale, [config.dilation_growth_rate**j, 1])]
            # Add downsampling layers
            model += [nn.ELU()]
            model += [EncodecConv1d(config, current_scale, current_scale * 2, kernel_size=ratio * 2, stride=ratio)]
            scaling *= 2

        model += [EncodecLSTM(config, scaling * config.num_filters)]
        model += [nn.ELU()]
        model += [EncodecConv1d(config, scaling * config.num_filters, config.hidden_size, config.last_kernel_size)]

        self.layers = nn.ModuleList(model)

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class EncodecDecoder(nn.Module):
    """SEANet decoder as used by EnCodec."""

    def __init__(self, config: EncodecConfig):
        super().__init__()
        scaling = int(2 ** len(config.upsampling_ratios))
        model = [EncodecConv1d(config, config.hidden_size, scaling * config.num_filters, config.kernel_size)]

        model += [EncodecLSTM(config, scaling * config.num_filters)]

        # Upsample to raw audio scale
        for ratio in config.upsampling_ratios:
            current_scale = scaling * config.num_filters
            # Add upsampling layers
            model += [nn.ELU()]
            model += [
                EncodecConvTranspose1d(config, current_scale, current_scale // 2, kernel_size=ratio * 2, stride=ratio)
            ]
            # Add residual layers
            for j in range(config.num_residual_layers):
                model += [EncodecResnetBlock(config, current_scale // 2, (config.dilation_growth_rate**j, 1))]
            scaling //= 2

        # Add final layers
        model += [nn.ELU()]
        model += [EncodecConv1d(config, config.num_filters, config.audio_channels, config.last_kernel_size)]
        self.layers = nn.ModuleList(model)

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class EncodecEuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance."""
    # 初始化函数，接受一个配置对象 config
    def __init__(self, config: EncodecConfig):
        # 调用父类的初始化函数
        super().__init__()
        
        # 创建一个全零的张量作为初始的嵌入向量，大小为 (codebook_size, codebook_dim)
        embed = torch.zeros(config.codebook_size, config.codebook_dim)
        
        # 设置对象的 codebook_size 属性
        self.codebook_size = config.codebook_size
        
        # 使用 register_buffer 方法注册一个名为 "inited" 的布尔型张量，值为 True
        self.register_buffer("inited", torch.Tensor([True]))
        
        # 使用 register_buffer 方法注册一个名为 "cluster_size" 的全零张量，大小为 (codebook_size,)
        self.register_buffer("cluster_size", torch.zeros(config.codebook_size))
        
        # 使用 register_buffer 方法注册一个名为 "embed" 的张量，初始值为 embed
        self.register_buffer("embed", embed)
        
        # 使用 register_buffer 方法注册一个名为 "embed_avg" 的张量，初始值为 embed 的克隆
        self.register_buffer("embed_avg", embed.clone())

    # 量化函数，接受隐藏状态 hidden_states 作为输入
    def quantize(self, hidden_states):
        # 将 embed 转置后进行量化计算
        embed = self.embed.t()
        
        # 计算隐藏状态的平方和，并保留维度
        scaled_states = hidden_states.pow(2).sum(1, keepdim=True)
        
        # 计算距离 dist，用于量化操作
        dist = -(scaled_states - 2 * hidden_states @ embed + embed.pow(2).sum(0, keepdim=True))
        
        # 选取距离最大的索引作为量化后的索引
        embed_ind = dist.max(dim=-1).indices
        
        # 返回量化后的索引
        return embed_ind

    # 编码函数，接受隐藏状态 hidden_states 作为输入
    def encode(self, hidden_states):
        # 获取隐藏状态的形状
        shape = hidden_states.shape
        
        # 对隐藏状态进行预处理，将其重塑为二维张量
        hidden_states = hidden_states.reshape((-1, shape[-1]))
        
        # 进行量化操作
        embed_ind = self.quantize(hidden_states)
        
        # 对量化后的索引进行后处理，恢复原始形状
        embed_ind = embed_ind.view(*shape[:-1])
        
        # 返回编码后的索引
        return embed_ind

    # 解码函数，接受量化后的索引 embed_ind 作为输入
    def decode(self, embed_ind):
        # 使用 nn.functional.embedding 对 embed_ind 进行解码，使用预先定义的 embed 作为嵌入矩阵
        quantize = nn.functional.embedding(embed_ind, self.embed)
        
        # 返回解码结果
        return quantize
class EncodecVectorQuantization(nn.Module):
    """
    Vector quantization implementation. Currently supports only euclidean distance.
    """

    def __init__(self, config: EncodecConfig):
        super().__init__()
        # 初始化时创建一个 EncodecEuclideanCodebook 对象作为 codebook
        self.codebook = EncodecEuclideanCodebook(config)

    def encode(self, hidden_states):
        # 将 hidden_states 的维度进行置换，通常用于序列数据的维度变换
        hidden_states = hidden_states.permute(0, 2, 1)
        # 调用 codebook 的 encode 方法进行向量编码
        embed_in = self.codebook.encode(hidden_states)
        return embed_in

    def decode(self, embed_ind):
        # 调用 codebook 的 decode 方法进行向量解码
        quantize = self.codebook.decode(embed_ind)
        # 再次置换维度，使其与输入 hidden_states 的维度一致
        quantize = quantize.permute(0, 2, 1)
        return quantize


class EncodecResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer."""

    def __init__(self, config: EncodecConfig):
        super().__init__()
        # 从 config 中获取相关参数
        self.codebook_size = config.codebook_size
        self.frame_rate = config.frame_rate
        self.num_quantizers = config.num_quantizers
        # 使用 ModuleList 创建多个 EncodecVectorQuantization 实例作为 layers
        self.layers = nn.ModuleList([EncodecVectorQuantization(config) for _ in range(config.num_quantizers)])

    def get_num_quantizers_for_bandwidth(self, bandwidth: Optional[float] = None) -> int:
        """Return num_quantizers based on specified target bandwidth."""
        # 根据码书大小和帧率计算每个量化器的带宽
        bw_per_q = math.log2(self.codebook_size) * self.frame_rate
        num_quantizers = self.num_quantizers
        if bandwidth is not None and bandwidth > 0.0:
            # 根据给定带宽计算最大可用的量化器数量
            num_quantizers = int(max(1, math.floor(bandwidth * 1000 / bw_per_q)))
        return num_quantizers

    def encode(self, embeddings: torch.Tensor, bandwidth: Optional[float] = None) -> torch.Tensor:
        """
        Encode a given input tensor with the specified frame rate at the given bandwidth. The RVQ encode method sets
        the appropriate number of quantizers to use and returns indices for each quantizer.
        """
        # 根据带宽计算要使用的量化器数量
        num_quantizers = self.get_num_quantizers_for_bandwidth(bandwidth)
        residual = embeddings
        all_indices = []
        # 对每个量化器层进行编码和解码
        for layer in self.layers[:num_quantizers]:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        # 将所有量化器的输出索引堆叠成一个张量返回
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation."""
        quantized_out = torch.tensor(0.0, device=codes.device)
        # 对每个量化器层进行解码
        for i, indices in enumerate(codes):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized
        return quantized_out


class EncodecPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定配置类
    config_class = EncodecConfig
    # 指定模型前缀
    base_model_prefix = "encodec"
    # 主输入名称
    main_input_name = "input_values"
    # 初始化神经网络模块的权重
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果是线性层模块
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重，均值为0，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置项，将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是LayerNorm或者GroupNorm模块
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            # 将偏置项初始化为零
            module.bias.data.zero_()
            # 将权重初始化为1.0
            module.weight.data.fill_(1.0)
        # 如果是一维卷积层模块
        elif isinstance(module, nn.Conv1d):
            # 使用Kaiming正态分布初始化权重
            nn.init.kaiming_normal_(module.weight)
            # 如果存在偏置项，根据特定公式使用均匀分布初始化
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)
        # 如果是嵌入层模块
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重，均值为0，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果指定了padding_idx，将其对应的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果是LSTM模块
        elif isinstance(module, nn.LSTM):
            # 遍历LSTM模块的命名参数
            for name, param in module.named_parameters():
                # 如果参数名中包含"weight"，使用Xavier均匀分布初始化
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                # 如果参数名中包含"bias"，将其初始化为零
                elif "bias" in name:
                    nn.init.constant_(param, 0.0)
# 定义一个多行字符串，用于存储关于 ENCODEC_START_DOCSTRING 的详细文档说明
ENCODEC_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`EncodecConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义一个多行字符串，用于存储关于 ENCODEC_INPUTS_DOCSTRING 的详细文档说明
ENCODEC_INPUTS_DOCSTRING = r"""
    Args:
        input_values (`torch.FloatTensor` of shape `(batch_size, channels, sequence_length)`, *optional*):
            Raw audio input converted to Float and padded to the approriate length in order to be encoded using chunks
            of length self.chunk_length and a stride of `config.chunk_stride`.
        padding_mask (`torch.BoolTensor` of shape `(batch_size, channels, sequence_length)`, *optional*):
            Mask to avoid computing scaling factors on padding token indices (can we avoid computing conv on these+).
            Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            <Tip warning={true}>

             `padding_mask` should always be passed, unless the input was truncated or not padded. This is because in
             order to process tensors effectively, the input audio should be padded so that `input_length % stride =
             step` with `step = chunk_length-stride`. This ensures that all chunks are of the same shape

            </Tip>

        bandwidth (`float`, *optional*):
            The target bandwidth. Must be one of `config.target_bandwidths`. If `None`, uses the smallest possible
            bandwidth. bandwidth is represented as a thousandth of what it is, e.g. 6kbps bandwidth is represented as
            `bandwidth == 6.0`
        audio_codes (`torch.LongTensor`  of shape `(batch_size, nb_chunks, chunk_length)`, *optional*):
            Discret code embeddings computed using `model.encode`.
        audio_scales (`torch.Tensor` of shape `(batch_size, nb_chunks)`, *optional*):
            Scaling factor for each `audio_codes` input.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# 应用装饰器函数 add_start_docstrings，添加了关于 EnCodec neural audio codec 模型的描述和 ENCODEC_START_DOCSTRING 的详细文档说明
@add_start_docstrings(
    "The EnCodec neural audio codec model.",
    ENCODEC_START_DOCSTRING,
)
    def __init__(self, config: EncodecConfig):
        # 调用父类的构造函数，传入配置对象
        super().__init__(config)
        # 将配置对象存储在实例中
        self.config = config

        # 创建编码器和解码器实例，使用给定的配置对象
        self.encoder = EncodecEncoder(config)
        self.decoder = EncodecDecoder(config)

        # 创建量化器实例，使用给定的配置对象
        self.quantizer = EncodecResidualVectorQuantizer(config)

        # 计算每个码书的比特数，并检查码书大小是否为2的幂
        self.bits_per_codebook = int(math.log2(self.config.codebook_size))
        if 2**self.bits_per_codebook != self.config.codebook_size:
            raise ValueError("The codebook_size must be a power of 2.")

        # 执行后续的初始化步骤
        self.post_init()

    def get_encoder(self):
        # 返回当前实例的编码器
        return self.encoder

    def get_decoder(self):
        # 返回当前实例的解码器
        return self.decoder

    def _encode_frame(
        self, input_values: torch.Tensor, bandwidth: float, padding_mask: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        使用底层的 VQVAE 对给定输入进行编码。如果 `config.normalize` 设置为 `True`，则首先对输入进行归一化。
        需要 padding_mask 来计算正确的比例。
        """
        # 获取输入张量的长度
        length = input_values.shape[-1]
        # 计算帧的持续时间，基于采样率和长度
        duration = length / self.config.sampling_rate

        # 如果配置中设置了 chunk_length_s，并且帧的持续时间超过了 chunk_length_s，则引发运行时错误
        if self.config.chunk_length_s is not None and duration > 1e-5 + self.config.chunk_length_s:
            raise RuntimeError(f"Duration of frame ({duration}) is longer than chunk {self.config.chunk_length_s}")

        scale = None
        if self.config.normalize:
            # 如果填充非零
            input_values = input_values * padding_mask
            # 计算输入的平均值（单声道）
            mono = torch.sum(input_values, 1, keepdim=True) / input_values.shape[1]
            # 计算标准差
            scale = mono.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-8
            # 对输入进行归一化
            input_values = input_values / scale

        # 使用编码器对归一化后的输入进行编码，得到嵌入
        embeddings = self.encoder(input_values)
        # 使用量化器对嵌入进行编码，得到码字
        codes = self.quantizer.encode(embeddings, bandwidth)
        # 调整码字的维度顺序
        codes = codes.transpose(0, 1)
        # 返回码字和归一化的比例
        return codes, scale

    def encode(
        self,
        input_values: torch.Tensor,
        padding_mask: torch.Tensor = None,
        bandwidth: Optional[float] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], EncodecEncoderOutput]:
        """
        Encodes the input audio waveform into discrete codes.

        Args:
            input_values (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Float values of the input audio waveform.
            padding_mask (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Padding mask used to pad the `input_values`.
            bandwidth (`float`, *optional*):
                The target bandwidth. Must be one of `config.target_bandwidths`. If `None`, uses the smallest possible
                bandwidth. bandwidth is represented as a thousandth of what it is, e.g. 6kbps bandwidth is represented
                as bandwidth == 6.0

        Returns:
            A list of frames containing the discrete encoded codes for the input audio waveform, along with rescaling
            factors for each chunk when `normalize` is True. Each frame is a tuple `(codebook, scale)`, with
            `codebook` of shape `[batch_size, num_codebooks, frames]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if bandwidth is None:
            bandwidth = self.config.target_bandwidths[0]
        if bandwidth not in self.config.target_bandwidths:
            raise ValueError(
                f"This model doesn't support the bandwidth {bandwidth}. "
                f"Select one of {self.config.target_bandwidths}."
            )

        _, channels, input_length = input_values.shape

        if channels < 1 or channels > 2:
            raise ValueError(f"Number of audio channels must be 1 or 2, but got {channels}")

        # Determine the chunk length and stride based on model configuration
        chunk_length = self.config.chunk_length
        if chunk_length is None:
            chunk_length = input_length
            stride = input_length  # No overlap between chunks if chunk_length equals input_length
        else:
            stride = self.config.chunk_stride

        # If padding mask is not provided, create a mask with all elements set to True
        if padding_mask is None:
            padding_mask = torch.ones_like(input_values).bool()

        encoded_frames = []
        scales = []

        # Check if input length is properly divisible into chunks
        step = chunk_length - stride
        if (input_length % stride) - step != 0:
            raise ValueError(
                "The input length is not properly padded for batched chunked decoding. Make sure to pad the input correctly."
            )

        # Iterate over the input audio waveform in chunks
        for offset in range(0, input_length - step, stride):
            mask = padding_mask[..., offset : offset + chunk_length].bool()
            frame = input_values[:, :, offset : offset + chunk_length]
            # Encode each chunk of audio waveform into discrete codes
            encoded_frame, scale = self._encode_frame(frame, bandwidth, mask)
            encoded_frames.append(encoded_frame)
            scales.append(scale)

        encoded_frames = torch.stack(encoded_frames)

        # Return encoded frames and scales if return_dict is False
        if not return_dict:
            return (encoded_frames, scales)

        # If return_dict is True, return an instance of EncodecEncoderOutput
        return EncodecEncoderOutput(encoded_frames, scales)
    def _linear_overlap_add(frames: List[torch.Tensor], stride: int):
        # Generic overlap add, with linear fade-in/fade-out, supporting complex scenario
        # e.g., more than 2 frames per position.
        # The core idea is to use a weight function that is a triangle,
        # with a maximum value at the middle of the chunk.
        # We use this weighting when summing the frames, and divide by the sum of weights
        # for each position at the end. Thus:
        #   - if a frame is the only one to cover a position, the weighting is a no-op.
        #   - if 2 frames cover a position:
        #          ...  ...
        #         /   \/   \
        #        /    /\    \
        #            S  T       , i.e. S offset of second frame starts, T end of first frame.
        # Then the weight function for each one is: (t - S), (T - t), with `t` a given offset.
        # After the final normalization, the weight of the second frame at position `t` is
        # (t - S) / (t - S + (T - t)) = (t - S) / (T - S), which is exactly what we want.
        #
        #   - if more than 2 frames overlap at a given point, we hope that by induction
        #      something sensible happens.

        # 检查输入帧列表是否为空
        if len(frames) == 0:
            raise ValueError("`frames` cannot be an empty list.")

        # 获取第一个帧的设备信息，数据类型和形状（去掉最后一个维度）
        device = frames[0].device
        dtype = frames[0].dtype
        shape = frames[0].shape[:-1]

        # 计算总的输出大小，考虑重叠时的步长
        total_size = stride * (len(frames) - 1) + frames[-1].shape[-1]

        # 获取第一个帧的长度
        frame_length = frames[0].shape[-1]

        # 生成时间向量，用于权重计算，使用三角形权重函数
        time_vec = torch.linspace(0, 1, frame_length + 2, device=device, dtype=dtype)[1:-1]
        weight = 0.5 - (time_vec - 0.5).abs()

        # 初始化总权重和输出张量
        sum_weight = torch.zeros(total_size, device=device, dtype=dtype)
        out = torch.zeros(*shape, total_size, device=device, dtype=dtype)
        offset: int = 0

        # 遍历每个帧并添加到输出张量中，同时累加权重
        for frame in frames:
            frame_length = frame.shape[-1]
            out[..., offset : offset + frame_length] += weight[:frame_length] * frame
            sum_weight[offset : offset + frame_length] += weight[:frame_length]
            offset += stride

        # 检查最小的权重和是否大于零，防止除以零错误
        if sum_weight.min() == 0:
            raise ValueError(f"`sum_weight` minimum element must be bigger than zero: {sum_weight}`")

        # 返回归一化后的输出张量
        return out / sum_weight
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], EncodecDecoderOutput]:
        """
        Decodes the given frames into an output audio waveform.

        Note that the output might be a bit bigger than the input. In that case, any extra steps at the end can be
        trimmed.

        Args:
            audio_codes (`torch.LongTensor`  of shape `(batch_size, nb_chunks, chunk_length)`, *optional*):
                Discrete code embeddings computed using `model.encode`.
            audio_scales (`torch.Tensor` of shape `(batch_size, nb_chunks)`, *optional*):
                Scaling factor for each `audio_codes` input.
            padding_mask (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Padding mask used to pad the `input_values`.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        """
        # Determine whether to return a dictionary output based on provided argument or default configuration
        return_dict = return_dict or self.config.return_dict

        # Retrieve the chunk length from configuration
        chunk_length = self.config.chunk_length

        # If chunk_length is not specified, decode a single frame
        if chunk_length is None:
            if len(audio_codes) != 1:
                raise ValueError(f"Expected one frame, got {len(audio_codes)}")
            # Decode the single frame using the provided audio codes and scales
            audio_values = self._decode_frame(audio_codes[0], audio_scales[0])
        else:
            decoded_frames = []

            # Decode each frame using corresponding codes and scales
            for frame, scale in zip(audio_codes, audio_scales):
                frames = self._decode_frame(frame, scale)
                decoded_frames.append(frames)

            # Combine decoded frames using linear overlap-add method
            audio_values = self._linear_overlap_add(decoded_frames, self.config.chunk_stride or 1)

        # Trim the audio waveform based on the provided padding mask
        if padding_mask is not None and padding_mask.shape[-1] < audio_values.shape[-1]:
            audio_values = audio_values[..., : padding_mask.shape[-1]]

        # Return either a tuple or EncodecDecoderOutput based on return_dict flag
        if not return_dict:
            return (audio_values,)
        return EncodecDecoderOutput(audio_values)
        return_dict = return_dict or self.config.return_dict
        # 如果 return_dict 为 None，则使用 self.config.return_dict 的值作为默认值

        if padding_mask is None:
            padding_mask = torch.ones_like(input_values).bool()
        # 如果 padding_mask 为 None，则创建一个与 input_values 维度相同的全为 True 的布尔张量作为 padding_mask

        if audio_codes is not None and audio_scales is None:
            raise ValueError("You specified `audio_codes` but did not specify the `audio_scales`")
        # 如果指定了 audio_codes 但未指定 audio_scales，则抛出 ValueError 异常

        if audio_scales is not None and audio_codes is None:
            raise ValueError("You specified `audio_scales` but did not specify the `audio_codes`")
        # 如果指定了 audio_scales 但未指定 audio_codes，则抛出 ValueError 异常

        if audio_scales is None and audio_codes is None:
            audio_codes, audio_scales = self.encode(input_values, padding_mask, bandwidth, False)
        # 如果未指定 audio_scales 和 audio_codes，则调用 self.encode 方法生成它们

        audio_values = self.decode(audio_codes, audio_scales, padding_mask, return_dict=return_dict)[0]
        # 使用 self.decode 方法解码得到 audio_values

        if not return_dict:
            return (audio_codes, audio_values)
        # 如果 return_dict 为 False，则返回 audio_codes 和 audio_values 的元组

        return EncodecOutput(audio_codes=audio_codes, audio_values=audio_values)
        # 否则，返回一个 EncodecOutput 对象，包含 audio_codes 和 audio_values
```