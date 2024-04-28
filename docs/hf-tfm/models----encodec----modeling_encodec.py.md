# `.\models\encodec\modeling_encodec.py`

```
# 设置文件编码格式为 UTF-8
# 版权声明
# 根据 Apache 授权协议许可
# 获取许可详情网址
# 根据适用法律或协议书面同意，本软件根据"原样"基础分发，没有明示或暗示的任何保证或条件
# 查看许可协议以获取更多详细信息
""" PyTorch EnCodec model."""

# 导入必要的模块
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
# 从其他文件中导入必要的函数和类
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_encodec import EncodecConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于生成通用文档的配置
_CONFIG_FOR_DOC = "EncodecConfig"

# 预训练模型列表
ENCODEC_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/encodec_24khz",
    "facebook/encodec_48khz",
    # See all EnCodec models at https://huggingface.co/models?filter=encodec
]

# 定义 EncodecOutput 类
@dataclass
class EncodecOutput(ModelOutput):
    """
    Args:
        audio_codes (`torch.FloatTensor`  of shape `(batch_size, nb_chunks, chunk_length)`, *optional*):
            Discret code embeddings computed using `model.encode`.
        audio_values (`torch.FlaotTensor` of shape `(batch_size, sequence_length)`, *optional*)
            Decoded audio values, obtained using the decoder part of Encodec.
    """

    audio_codes: torch.FloatTensor = None
    audio_values: torch.FloatTensor = None

# 定义 EncodecEncoderOutput 类
@dataclass
class EncodecEncoderOutput(ModelOutput):
    """
    Args:
        audio_codes (`torch.FloatTensor`  of shape `(batch_size, nb_chunks, chunk_length)`, *optional*):
            Discret code embeddings computed using `model.encode`.
        audio_scales (`torch.Tensor` of shape `(batch_size, nb_chunks)`, *optional*):
            Scaling factor for each `audio_codes` input. This is used to unscale each chunk of audio when decoding.
    """

    audio_codes: torch.FloatTensor = None
    audio_scales: torch.FloatTensor = None

# 定义 EncodecDecoderOutput 类
@dataclass
class EncodecDecoderOutput(ModelOutput):
    """
    Args:
        audio_values (`torch.FloatTensor`  of shape `(batch_size, segment_length)`, *optional*):
            Decoded audio values, obtained using the decoder part of Encodec.
    """

    audio_values: torch.FloatTensor = None

# 定义 EncodecConv1d 类
class EncodecConv1d(nn.Module):
    """Conv1d with asymmetric or causal padding and normalization."""

    def __init__(
        self, config, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, dilation: int = 1
        ):
        # 调用父类初始化方法
        super().__init__()
        # 从配置中获取是否使用因果卷积的设置
        self.causal = config.use_causal_conv
        # 从配置中获取填充模式
        self.pad_mode = config.pad_mode
        # 从配置中获取规范化类型
        self.norm_type = config.norm_type

        # 如果规范化类型不是“weight_norm”或“time_group_norm”，则引发 ValueError 异常
        if self.norm_type not in ["weight_norm", "time_group_norm"]:
            raise ValueError(
                f'self.norm_type must be one of `"weight_norm"`, `"time_group_norm"`), got {self.norm_type}'
            )

        # 在步幅大于 1 且膨胀大于 1 的情况下，提醒用户不寻常的设置
        if stride > 1 and dilation > 1:
            logger.warning(
                "EncodecConv1d has been initialized with stride > 1 and dilation > 1"
                f" (kernel_size={kernel_size} stride={stride}, dilation={dilation})."
            )

        # 创建 1 维卷积层
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, dilation=dilation)
        # 如果规范化类型是“weight_norm”，则将卷积层应用权重规范化
        if self.norm_type == "weight_norm":
            self.conv = nn.utils.weight_norm(self.conv)
        # 如果规范化类型是“time_group_norm”，则对输出通道应用组归一化
        elif self.norm_type == "time_group_norm":
            self.norm = nn.GroupNorm(1, out_channels)

    @staticmethod
    def _get_extra_padding_for_conv1d(
        hidden_states: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
    ) -> int:
        """See `pad_for_conv1d`."""
        # 计算卷积层之外的额外填充量
        length = hidden_states.shape[-1]
        n_frames = (length - kernel_size + padding_total) / stride + 1
        ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
        return ideal_length - length

    @staticmethod
    def _pad1d(hidden_states: torch.Tensor, paddings: Tuple[int, int], mode: str = "zero", value: float = 0.0):
        """Tiny wrapper around torch.nn.functional.pad, just to allow for reflect padding on small input.
        If this is the case, we insert extra 0 padding to the right before the reflection happens.
        """
        # 在小输入上允许反射填充的 torch.nn.functional.pad 的简单封装
        length = hidden_states.shape[-1]
        padding_left, padding_right = paddings
        if not mode == "reflect":
            return nn.functional.pad(hidden_states, paddings, mode, value)

        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            hidden_states = nn.functional.pad(hidden_states, (0, extra_pad))
        padded = nn.functional.pad(hidden_states, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    # 前向传播函数，接受隐藏状态作为输入
    def forward(self, hidden_states):
        # 获取卷积核大小
        kernel_size = self.conv.kernel_size[0]
        # 获取卷积步长
        stride = self.conv.stride[0]
        # 获取卷积扩张率
        dilation = self.conv.dilation[0]
        # 计算使用扩张率后的有效核大小
        kernel_size = (kernel_size - 1) * dilation + 1
        # 计算总填充量
        padding_total = kernel_size - stride
        # 获取额外填充量
        extra_padding = self._get_extra_padding_for_conv1d(hidden_states, kernel_size, stride, padding_total)

        if self.causal:
            # 如果是因果卷积，进行左填充
            hidden_states = self._pad1d(hidden_states, (padding_total, extra_padding), mode=self.pad_mode)
        else:
            # 如果不是因果卷积，对于奇数步长需要异步填充
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            hidden_states = self._pad1d(
                hidden_states, (padding_left, padding_right + extra_padding), mode=self.pad_mode
            )

        # 使用填充后的隐藏状态进行卷积操作
        hidden_states = self.conv(hidden_states)

        if self.norm_type == "time_group_norm":
            # 如果使用时间分组归一化，对隐藏状态进行归一化处理
            hidden_states = self.norm(hidden_states)

        # 返回处理后的隐藏状态
        return hidden_states
``` 
# 定义一个带有不对称或因果填充和规范化的 ConvTranspose1d 类
class EncodecConvTranspose1d(nn.Module):
    """ConvTranspose1d with asymmetric or causal padding and normalization."""

    # 初始化函数，接受输入通道数、输出通道数、卷积核大小和步长
    def __init__(self, config, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        # 是否使用因果卷积
        self.causal = config.use_causal_conv
        # 右侧修剪比例
        self.trim_right_ratio = config.trim_right_ratio
        # 规范化类型
        self.norm_type = config.norm_type
        # 如果规范化类型不在指定的列表中，则抛出异常
        if self.norm_type not in ["weight_norm", "time_group_norm"]:
            raise ValueError(
                f'self.norm_type must be one of `"weight_norm"`, `"time_group_norm"`), got {self.norm_type}'
            )

        # 创建 ConvTranspose1d 层
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride)
        # 根据规范化类型选择规范化操作
        if config.norm_type == "weight_norm":
            self.conv = nn.utils.weight_norm(self.conv)
        elif config.norm_type == "time_group_norm":
            self.norm = nn.GroupNorm(1, out_channels)

        # 如果不是因果卷积且修剪右侧比例不等于1.0，则抛出异常
        if not (self.causal or self.trim_right_ratio == 1.0):
            raise ValueError("`trim_right_ratio` != 1.0 only makes sense for causal convolutions")

    # 前向传播函数
    def forward(self, hidden_states):
        # 获取卷积核大小和步长
        kernel_size = self.conv.kernel_size[0]
        stride = self.conv.stride[0]
        # 计算总填充量
        padding_total = kernel_size - stride

        # 使用 ConvTranspose1d 进行卷积操作
        hidden_states = self.conv(hidden_states)

        # 如果规范化类型为"time_group_norm"，则进行规范化操作
        if self.norm_type == "time_group_norm":
            hidden_states = self.norm(hidden_states)

        # 如果是因果卷积，则根据指定比例修剪填充，否则进行对称填充
        if self.causal:
            # 根据指定比例修剪右侧填充
            padding_right = math.ceil(padding_total * self.trim_right_ratio)
        else:
            # 针对奇数步长进行不对称填充
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

    # 初始化函数，接受配置和维度参数
    def __init__(self, config, dimension):
        super().__init__()
        # 创建 LSTM 层
        self.lstm = nn.LSTM(dimension, dimension, config.num_lstm_layers)

    # 前向传播函数
    def forward(self, hidden_states):
        # 将输入数据进行维度变换
        hidden_states = hidden_states.permute(2, 0, 1)
        # 对输入数据进行 LSTM 操作，并将结果与输入相加
        hidden_states = self.lstm(hidden_states)[0] + hidden_states
        hidden_states = hidden_states.permute(1, 2, 0)
        return hidden_states


class EncodecResnetBlock(nn.Module):
    """
    Residual block from SEANet model as used by EnCodec.
    """
    # 初始化函数，接受配置、维度和扩张率列表作为参数
    def __init__(self, config: EncodecConfig, dim: int, dilations: List[int]):
        # 调用父类的初始化方法
        super().__init__()
        # 确定卷积核大小，这里是一个元组，包含了残差块和池化块的卷积核大小
        kernel_sizes = (config.residual_kernel_size, 1)
        # 如果卷积核大小的数量和扩张率列表的数量不一致，抛出数值错误异常
        if len(kernel_sizes) != len(dilations):
            raise ValueError("Number of kernel sizes should match number of dilations")

        # 计算隐藏层维度
        hidden = dim // config.compress
        # 初始化残差块
        block = []
        # 遍历卷积核大小和扩张率的组合
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            # 确定输入通道数，如果是第一层则为维度，否则为隐藏层维度
            in_chs = dim if i == 0 else hidden
            # 确定输出通道数，如果是最后一层则为维度，否则为隐藏层维度
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            # 添加ELU激活函数到块中
            block += [nn.ELU()]
            # 添加EncodecConv1d卷积层到块中
            block += [EncodecConv1d(config, in_chs, out_chs, kernel_size, dilation=dilation)]
        # 将块转换为模块列表
        self.block = nn.ModuleList(block)

        # 如果配置指定使用卷积快捷连接
        if config.use_conv_shortcut:
            # 使用EncodecConv1d作为快捷连接
            self.shortcut = EncodecConv1d(config, dim, dim, kernel_size=1)
        else:
            # 否则使用恒等映射
            self.shortcut = nn.Identity()

    # 前向传播函数，接受隐藏状态作为输入
    def forward(self, hidden_states):
        # 保存输入状态为残差
        residual = hidden_states
        # 遍历残差块中的每一层，并计算输出状态
        for layer in self.block:
            hidden_states = layer(hidden_states)

        # 返回残差块输出和快捷连接输出的和
        return self.shortcut(residual) + hidden_states
class EncodecEncoder(nn.Module):
    """SEANet encoder as used by EnCodec."""  # 创建一个SEANet编码器类，EnCodec中使用

    def __init__(self, config: EncodecConfig):
        super().__init__()
        model = [EncodecConv1d(config, config.audio_channels, config.num_filters, config.kernel_size)]  # 创建包含一个卷积层的模型
        scaling = 1  # 初始化一个缩放参数
        
        # Downsample to raw audio scale
        for ratio in reversed(config.upsampling_ratios):
            current_scale = scaling * config.num_filters  # 计算当前缩放比例
            # Add residual layers
            for j in range(config.num_residual_layers):
                model += [EncodecResnetBlock(config, current_scale, [config.dilation_growth_rate**j, 1])]  # 添加残差块
            # Add downsampling layers
            model += [nn.ELU()]  # 添加ELU激活函数
            model += [EncodecConv1d(config, current_scale, current_scale * 2, kernel_size=ratio * 2, stride=ratio)]  # 添加卷积层
            scaling *= 2  # 更新缩放参数

        model += [EncodecLSTM(config, scaling * config.num_filters)]  # 添加LSTM层
        model += [nn.ELU()]  # 添加ELU激活函数
        model += [EncodecConv1d(config, scaling * config.num_filters, config.hidden_size, config.last_kernel_size)]  # 添加卷积层

        self.layers = nn.ModuleList(model)  # 将model转换为模块列表，赋值给self.layers

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)  # 逐层对hidden_states进行前向传播
        return hidden_states  # 返回hidden_states


class EncodecDecoder(nn.Module):
    """SEANet decoder as used by EnCodec."""  # 创建一个SEANet解码器类，EnCodec中使用

    def __init__(self, config: EncodecConfig):
        super().__init__()
        scaling = int(2 ** len(config.upsampling_ratios))  # 计算缩放系数
        model = [EncodecConv1d(config, config.hidden_size, scaling * config.num_filters, config.kernel_size)]  # 创建包含一个卷积层的模型

        model += [EncodecLSTM(config, scaling * config.num_filters)]  # 添加LSTM层

        # Upsample to raw audio scale
        for ratio in config.upsampling_ratios:
            current_scale = scaling * config.num_filters  # 计算当前缩放比例
            # Add upsampling layers
            model += [nn.ELU()]  # 添加ELU激活函数
            model += [EncodecConvTranspose1d(config, current_scale, current_scale // 2, kernel_size=ratio * 2, stride=ratio)]  # 添加转置卷积层
            # Add residual layers
            for j in range(config.num_residual_layers):
                model += [EncodecResnetBlock(config, current_scale // 2, (config.dilation_growth_rate**j, 1))]  # 添加残差块
            scaling //= 2  # 更新缩放参数

        # Add final layers
        model += [nn.ELU()]  # 添加ELU激活函数
        model += [EncodecConv1d(config, config.num_filters, config.audio_channels, config.last_kernel_size)]  # 添加卷积层
        self.layers = nn.ModuleList(model)  # 将model转换为模块列表，赋值给self.layers

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)  # 逐层对hidden_states进行前向传播
        return hidden_states  # 返回hidden_states


class EncodecEuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance."""  # 创建一个使用欧几里得距离的码本类
    # 初始化编码器对象，接受一个 EncodecConfig 类型的参数
    def __init__(self, config: EncodecConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个全零张量，用于表示编码器的初始嵌入矩阵，尺寸为 (codebook_size, codebook_dim)
        embed = torch.zeros(config.codebook_size, config.codebook_dim)

        # 设置编码器的编码矩阵的大小
        self.codebook_size = config.codebook_size

        # 将一个标记张量注册为模型的缓冲区，表示编码器是否已经初始化，初始值为 True
        self.register_buffer("inited", torch.Tensor([True]))
        # 将一个全零张量注册为模型的缓冲区，表示每个聚类的大小，尺寸为 (codebook_size)
        self.register_buffer("cluster_size", torch.zeros(config.codebook_size))
        # 将初始的嵌入矩阵注册为模型的缓冲区，表示编码器的编码矩阵
        self.register_buffer("embed", embed)
        # 克隆初始的嵌入矩阵，用于表示平均嵌入矩阵
        self.register_buffer("embed_avg", embed.clone())

    # 对隐藏状态进行量化
    def quantize(self, hidden_states):
        # 转置编码矩阵
        embed = self.embed.t()
        # 对隐藏状态进行缩放并求和
        scaled_states = hidden_states.pow(2).sum(1, keepdim=True)
        # 计算欧氏距离
        dist = -(scaled_states - 2 * hidden_states @ embed + embed.pow(2).sum(0, keepdim=True))
        # 获取最大值对应的索引，即嵌入矩阵中的索引
        embed_ind = dist.max(dim=-1).indices
        return embed_ind

    # 对隐藏状态进行编码
    def encode(self, hidden_states):
        # 获取隐藏状态的形状
        shape = hidden_states.shape
        # 将隐藏状态展平
        hidden_states = hidden_states.reshape((-1, shape[-1]))
        # 进行量化
        embed_ind = self.quantize(hidden_states)
        # 将量化结果恢复成原来的形状
        embed_ind = embed_ind.view(*shape[:-1])
        return embed_ind

    # 解码编码索引
    def decode(self, embed_ind):
        # 使用嵌入索引获取编码结果
        quantize = nn.functional.embedding(embed_ind, self.embed)
        return quantize
# 定义一个编码器向量量化的类，用于实现向量量化，目前仅支持欧式距离
class EncodecVectorQuantization(nn.Module):
    """
    Vector quantization implementation. Currently supports only euclidean distance.
    """

    def __init__(self, config: EncodecConfig):
        super().__init__()
        # 初始化编码器的欧式码本
        self.codebook = EncodecEuclideanCodebook(config)

    # 对隐藏状态进行编码
    def encode(self, hidden_states):
        # 调整隐藏状态的维度顺序
        hidden_states = hidden_states.permute(0, 2, 1)
        # 使用编码器进行编码
        embed_in = self.codebook.encode(hidden_states)
        return embed_in

    # 对编码后的值进行解码
    def decode(self, embed_ind):
        # 使用编码器进行解码
        quantize = self.codebook.decode(embed_ind)
        # 调整编码后的值的维度顺序
        quantize = quantize.permute(0, 2, 1)
        return quantize


# 定义一个编码器预处理模型类
class EncodecResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer."""

    def __init__(self, config: EncodecConfig):
        super().__init__()
        # 初始化代码本大小
        self.codebook_size = config.codebook_size
        # 初始化帧率
        self.frame_rate = config.frame_rate
        # 初始化量化器数量
        self.num_quantizers = config.num_quantizers
        # 使用编码器向量量化类创建多个编码器对象列表
        self.layers = nn.ModuleList([EncodecVectorQuantization(config) for _ in range(config.num_quantizers)])

    # 根据指定目标带宽返回量化器数量
    def get_num_quantizers_for_bandwidth(self, bandwidth: Optional[float] = None) -> int:
        """Return num_quantizers based on specified target bandwidth."""
        # 根据代码本大小和帧率计算带宽
        bw_per_q = math.log2(self.codebook_size) * self.frame_rate
        num_quantizers = self.num_quantizers
        # 根据指定带宽计算量化器数量
        if bandwidth is not None and bandwidth > 0.0:
            num_quantizers = int(max(1, math.floor(bandwidth * 1000 / bw_per_q)))
        return num_quantizers

    # 对给定输入张量进行编码
    def encode(self, embeddings: torch.Tensor, bandwidth: Optional[float] = None) -> torch.Tensor:
        """
        Encode a given input tensor with the specified frame rate at the given bandwidth. The RVQ encode method sets
        the appropriate number of quantizers to use and returns indices for each quantizer.
        """
        # 获取指定带宽的量化器数量
        num_quantizers = self.get_num_quantizers_for_bandwidth(bandwidth)
        residual = embeddings
        all_indices = []
        # 遍历所有编码���进行编码
        for layer in self.layers[:num_quantizers]:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        out_indices = torch.stack(all_indices)
        return out_indices

    # 对给定的编码进行解码
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation."""
        quantized_out = torch.tensor(0.0, device=codes.device)
        # 遍历所有编码器进行解码
        for i, indices in enumerate(codes):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized
        return quantized_out


# 定义一个编码器预训练模型类
class EncodecPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置配置类为EncodecConfig
    config_class = EncodecConfig
    # 设置基础模型前缀为encodec
    base_model_prefix = "encodec"
    # 设置主要输入名称为input_values
    main_input_name = "input_values"
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果是线性层
        if isinstance(module, nn.Linear):
            # 初始化权重为正态分布
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置，则将偏置初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是 LayerNorm 或 GroupNorm
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            # 将偏置初始化为零
            module.bias.data.zero_()
            # 将权重初始化为全 1
            module.weight.data.fill_(1.0)
        # 如果是 1D 卷积层
        elif isinstance(module, nn.Conv1d):
            # 使用 kaiming 正态分布初始化权重
            nn.init.kaiming_normal_(module.weight)
            # 如果有偏置，则根据公式初始化
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)
        # 如果是嵌入层
        elif isinstance(module, nn.Embedding):
            # 初始化权重为正态分布
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有填充索引，则将填充索引对应的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果是 LSTM 层
        elif isinstance(module, nn.LSTM):
            # 遍历参数
            for name, param in module.named_parameters():
                # 如果参数名中包含 "weight"，则使用 xavier 均匀分布初始化
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                # 如果参数名中包含 "bias"，则将偏置初始化为零
                elif "bias" in name:
                    nn.init.constant_(param, 0.0)
# 定义 ENCODEC_START_DOCSTRING 常量，包含该模型的文档字符串，说明该模型继承自 PreTrainedModel，并给出了一些参数信息
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

# 定义 ENCODEC_INPUTS_DOCSTRING 常量，包含该模型的输入参数文档字符串，详细描述了输入参数及其作用
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
        audio_codes (`torch.FloatTensor`  of shape `(batch_size, nb_chunks, chunk_length)`, *optional*):
            Discret code embeddings computed using `model.encode`.
        audio_scales (`torch.Tensor` of shape `(batch_size, nb_chunks)`, *optional*):
            Scaling factor for each `audio_codes` input.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# 使用 @add_start_docstrings 装饰器添加类的文档说明及输入参数说明
@add_start_docstrings(
    "The EnCodec neural audio codec model.",
    ENCODEC_START_DOCSTRING,
)
# 定义 EncodecModel 类，继承自 EncodecPreTrainedModel
class EncodecModel(EncodecPreTrainedModel):
    # 初始化函数，接受一个 EncodecConfig 类型的配置参数
    def __init__(self, config: EncodecConfig):
        # 调用父类的初始化函数
        super().__init__(config)
        # 保存传入的配置参数
        self.config = config

        # 创建编码器和解码器对象
        self.encoder = EncodecEncoder(config)
        self.decoder = EncodecDecoder(config)

        # 创建离散向量量化器对象
        self.quantizer = EncodecResidualVectorQuantizer(config)

        # 计算每个向量编码所需比特数
        self.bits_per_codebook = int(math.log2(self.config.codebook_size))
        # 如果 codebook_size 不是 2 的幂，则抛出异常
        if 2**self.bits_per_codebook != self.config.codebook_size:
            raise ValueError("The codebook_size must be a power of 2.")

        # 初始化权重和完成后续处理
        self.post_init()

    # 返回编码器对象
    def get_encoder(self):
        return self.encoder

    # 返回解码器对象
    def get_decoder(self):
        return self.decoder

    # 编码输入张量，返回离散编码和缩放因子
    def _encode_frame(
        self, input_values: torch.Tensor, bandwidth: float, padding_mask: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encodes the given input using the underlying VQVAE. If `config.normalize` is set to `True` the input is first
        normalized. The padding mask is required to compute the correct scale.
        """
        # 获取输入张量长度
        length = input_values.shape[-1]
        # 计算持续时间
        duration = length / self.config.sampling_rate

        # 如果指定了 chunk_length_s，且持续时间超过指定值，则抛出异常
        if self.config.chunk_length_s is not None and duration > 1e-5 + self.config.chunk_length_s:
            raise RuntimeError(f"Duration of frame ({duration}) is longer than chunk {self.config.chunk_length_s}")

        scale = None
        # 如果需要归一化
        if self.config.normalize:
            # 如果填充值非零
            input_values = input_values * padding_mask
            mono = torch.sum(input_values, 1, keepdim=True) / input_values.shape[1]  # 计算平均值
            scale = mono.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-8
            input_values = input_values / scale  # 归一化

        # 使用编码器得到嵌入向量
        embeddings = self.encoder(input_values)
        # 使用离散向量量化器编码向量，并调整维度
        codes = self.quantizer.encode(embeddings, bandwidth)
        codes = codes.transpose(0, 1)
        return codes, scale

    # 对输入张量进行编码
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
            factors for each chunk when `normalize` is True. Each frames is a tuple `(codebook, scale)`, with
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

        chunk_length = self.config.chunk_length
        if chunk_length is None:
            chunk_length = input_length
            stride = input_length
        else:
            stride = self.config.chunk_stride

        if padding_mask is None:
            padding_mask = torch.ones_like(input_values).bool()  # 使用与input_values相同形状的张量来创建全为True的填充掩码

        encoded_frames = []  # 创建一个空的列表用于存储编码后的帧
        scales = []  # 创建一个空的列表用于存储缩放因子

        step = chunk_length - stride  # 计算步长
        if (input_length % stride) - step != 0:  # 检查输入长度是否被正确地填充以进行批量分块解码
            raise ValueError(
                "The input length is not properly padded for batched chunked decoding. Make sure to pad the input correctly."
            )

        for offset in range(0, input_length - step, stride):  # 遍历每一个偏移位置，以便进行编码
            mask = padding_mask[..., offset : offset + chunk_length].bool()  # 从填充掩码中提取当前帧的掩码
            frame = input_values[:, :, offset : offset + chunk_length]  # 从输入值中提取当前帧
            encoded_frame, scale = self._encode_frame(frame, bandwidth, mask)  # 对当前帧进行编码
            encoded_frames.append(encoded_frame)  # 将编码后的帧添加到列表中
            scales.append(scale)  # 将缩放因子添加到列表中

        encoded_frames = torch.stack(encoded_frames)  # 将编码后的帧堆叠在一起

        if not return_dict:
            return (encoded_frames, scales)  # 如果不返回字典，则返回编码后的帧和缩放因子的元组

        return EncodecEncoderOutput(encoded_frames, scales)  # 返回编码后的帧和缩放因子的对象
    def _linear_overlap_add(frames: List[torch.Tensor], stride: int):
        # 定义一个线性叠加函数，支持线性淡入/淡出效果，同时支持复杂情况，例如一个位置有多于两个帧。
        # 核心思想是使用一个三角形权重函数，在每个块的中间具有最大值。
        # 我们在对帧进行求和时使用这个权重，并在最后除以每个位置的权重总和。因此：
        #   - 如果一个帧是唯一覆盖一个位置的，则权重函数不起作用。
        #   - 如果有两个帧覆盖一个位置：
        #          ...  ...
        #         /   \/   \
        #        /    /\    \
        #            S  T       ，即第二帧开始的偏移量为 S，第一个帧的结束为 T。
        # 那么每个帧的权重函数分别是：(t - S)，(T - t)，其中 `t` 是给定的偏移量。
        # 在最终归一化之后，位置 `t` 上第二帧的权重为 (t - S) / (t - S + (T - t)) = (t - S) / (T - S)，这正是我们想要的。
        #
        #   - 如果有多于两个帧在给定点重叠，我们希望通过归纳出现一些合理的结果。
        if len(frames) == 0:
            raise ValueError("`frames` 不能是空列表.")

        device = frames[0].device
        dtype = frames[0].dtype
        shape = frames[0].shape[:-1]
        total_size = stride * (len(frames) - 1) + frames[-1].shape[-1]

        frame_length = frames[0].shape[-1]
        time_vec = torch.linspace(0, 1, frame_length + 2, device=device, dtype=dtype)[1:-1]
        weight = 0.5 - (time_vec - 0.5).abs()

        sum_weight = torch.zeros(total_size, device=device, dtype=dtype)
        out = torch.zeros(*shape, total_size, device=device, dtype=dtype)
        offset: int = 0

        for frame in frames:
            frame_length = frame.shape[-1]
            out[..., offset : offset + frame_length] += weight[:frame_length] * frame
            sum_weight[offset : offset + frame_length] += weight[:frame_length]
            offset += stride

        if sum_weight.min() == 0:
            raise ValueError(f"`sum_weight` 最小元素必须大于零: {sum_weight}`")

        return out / sum_weight

    def _decode_frame(self, codes: torch.Tensor, scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 转置输入的编码，以便于解码
        codes = codes.transpose(0, 1)
        # 解码编码成嵌入
        embeddings = self.quantizer.decode(codes)
        # 使用解码器解码嵌入并得到输出
        outputs = self.decoder(embeddings)
        # 如果提供了比例参数，则对输出进行缩放
        if scale is not None:
            outputs = outputs * scale.view(-1, 1, 1)
        return outputs

    def decode(
        self,
        audio_codes: torch.Tensor,
        audio_scales: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], EncodecDecoderOutput]:
        """
        Decodes the given frames into an output audio waveform.

        Note that the output might be a bit bigger than the input. In that case, any extra steps at the end can be
        trimmed.

        Args:
            audio_codes (`torch.FloatTensor`  of shape `(batch_size, nb_chunks, chunk_length)`, *optional*):
                Discret code embeddings computed using `model.encode`.
            audio_scales (`torch.Tensor` of shape `(batch_size, nb_chunks)`, *optional*):
                Scaling factor for each `audio_codes` input.
            padding_mask (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`):
                Padding mask used to pad the `input_values`.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        """
        # Determine whether to return a dictionary or plain tuple based on the value of `return_dict` or model configuration
        return_dict = return_dict or self.config.return_dict

        # Get the length of each audio chunk
        chunk_length = self.config.chunk_length
        # If chunk length is not specified
        if chunk_length is None:
            # Ensure only one frame is present
            if len(audio_codes) != 1:
                raise ValueError(f"Expected one frame, got {len(audio_codes)}")
            # Decode the single frame
            audio_values = self._decode_frame(audio_codes[0], audio_scales[0])
        else:
            # Decode each frame individually and store them
            decoded_frames = []
            for frame, scale in zip(audio_codes, audio_scales):
                frames = self._decode_frame(frame, scale)
                decoded_frames.append(frames)
            # Combine decoded frames using linear overlap-add
            audio_values = self._linear_overlap_add(decoded_frames, self.config.chunk_stride or 1)

        # Truncate audio values based on padding mask if necessary
        if padding_mask is not None and padding_mask.shape[-1] < audio_values.shape[-1]:
            audio_values = audio_values[..., : padding_mask.shape[-1]]

        # Return either a tuple or a `EncodecDecoderOutput` object depending on `return_dict`
        if not return_dict:
            return (audio_values,)
        return EncodecDecoderOutput(audio_values)

    @add_start_docstrings_to_model_forward(ENCODEC_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=EncodecOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_values: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        bandwidth: Optional[float] = None,
        audio_codes: Optional[torch.Tensor] = None,
        audio_scales: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[Tuple[torch.Tensor, torch.Tensor], EncodecOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from datasets import load_dataset
        >>> from transformers import AutoProcessor, EncodecModel

        >>> dataset = load_dataset("ashraq/esc50")
        >>> audio_sample = dataset["train"]["audio"][0]["array"]

        >>> model_id = "facebook/encodec_24khz"
        >>> model = EncodecModel.from_pretrained(model_id)
        >>> processor = AutoProcessor.from_pretrained(model_id)

        >>> inputs = processor(raw_audio=audio_sample, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> audio_codes = outputs.audio_codes
        >>> audio_values = outputs.audio_values
        ```"""
        # 设定返回值类型，默认为 self.config.return_dict
        return_dict = return_dict or self.config.return_dict

        # 如果未提供 padding_mask，则创建一个与 input_values 形状相同的全 True 的布尔类型张量
        if padding_mask is None:
            padding_mask = torch.ones_like(input_values).bool()

        # 如果提供了 audio_codes 但未提供 audio_scales，则引发 ValueError
        if audio_codes is not None and audio_scales is None:
            raise ValueError("You specified `audio_codes` but did not specify the `audio_scales`")

        # 如果提供了 audio_scales 但未提供 audio_codes，则引发 ValueError
        if audio_scales is not None and audio_codes is None:
            raise ValueError("You specified `audio_scales` but did not specify the `audio_codes`")

        # 如果未提供 audio_scales 和 audio_codes，则调用 encode 方法生成它们
        if audio_scales is None and audio_codes is None:
            audio_codes, audio_scales = self.encode(input_values, padding_mask, bandwidth, False)

        # 调用 decode 方法将音频编码解码成音频值
        audio_values = self.decode(audio_codes, audio_scales, padding_mask, return_dict=return_dict)[0]
        # 如果 return_dict 为 False，则返回元组 (audio_codes, audio_values)
        if not return_dict:
            return (audio_codes, audio_values)

        # 否则，返回 EncodecOutput 对象，包含 audio_codes 和 audio_values
        return EncodecOutput(audio_codes=audio_codes, audio_values=audio_values)
```