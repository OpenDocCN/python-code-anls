# `.\transformers\models\univnet\modeling_univnet.py`

```
# 版权声明和许可声明
""" PyTorch UnivNetModel model."""

# 导入所需的库
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

# 导入模型相关的辅助函数和类
from ...modeling_utils import ModelOutput, PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_univnet import UnivNetConfig

# 设置日志记录器
logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "UnivNetConfig"

_CHECKPOINT_FOR_DOC = "dg845/univnet-dev"

# 预训练模型列表
UNIVNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "dg845/univnet-dev",
    # See all UnivNet models at https://huggingface.co/models?filter=univnet
]

# 定义输出类，包含生成的音频波形和波形的原始未填充长度
@dataclass
class UnivNetModelOutput(ModelOutput):
    """
    Output class for the [`UnivNetModel`], which includes the generated audio waveforms and the original unpadded
    lengths of those waveforms (so that the padding can be removed by [`UnivNetModel.batch_decode`]).

    Args:
        waveforms (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Batched 1D (mono-channel) output audio waveforms.
        waveform_lengths (`torch.FloatTensor` of shape `(batch_size,)`):
            The batched length in samples of each unpadded waveform in `waveforms`.
    """

    waveforms: torch.FloatTensor = None
    waveform_lengths: torch.FloatTensor = None

# 定义用于每个位置变量卷积块（LVCBlock）内核预测网络的残差块的实现
class UnivNetKernelPredictorResidualBlock(nn.Module):
    """
    Implementation of the residual block for the kernel predictor network inside each location variable convolution
    block (LVCBlock).

    Parameters:
        config: (`UnivNetConfig`):
            Config for the `UnivNetModel` model.
    """

    def __init__(
        self,
        config: UnivNetConfig,
    ):
        super().__init__()
        # 设置残差块的参数
        self.channels = config.model_in_channels
        self.kernel_size = config.kernel_predictor_conv_size
        self.dropout_prob = config.kernel_predictor_dropout
        self.leaky_relu_slope = config.leaky_relu_slope

        padding = (self.kernel_size - 1) // 2

        # 定义模型层
        self.dropout = nn.Dropout(self.dropout_prob)
        self.conv1 = nn.Conv1d(self.channels, self.channels, self.kernel_size, padding=padding, bias=True)
        self.conv2 = nn.Conv1d(self.channels, self.channels, self.kernel_size, padding=padding, bias=True)
    # 前向传播函数，接受隐藏状态作为输入，返回处理后的隐藏状态
    def forward(self, hidden_states: torch.FloatTensor):
        # 给定的 hidden_states 应该具有形状 (batch_size, channels, seq_length)
        residual = hidden_states  # 保存输入的隐藏状态，用于后续的残差连接
        hidden_states = self.dropout(hidden_states)  # 对隐藏状态进行dropout处理
        hidden_states = self.conv1(hidden_states)  # 使用第一个卷积层处理隐藏状态
        hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)  # 使用Leaky ReLU激活函数处理隐藏状态
        hidden_states = self.conv2(hidden_states)  # 使用第二个卷积层处理隐藏状态
        hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)  # 使用Leaky ReLU激活函数处理隐藏状态
        return hidden_states + residual  # 返回处理后的隐藏状态加上之前保存的残差连接的隐藏状态

    # 对卷积层应用权重归一化
    def apply_weight_norm(self):
        nn.utils.weight_norm(self.conv1)  # 对第一个卷积层进行权重归一化处理
        nn.utils.weight_norm(self.conv2)  # 对第二个卷积层进行权重归一化处理

    # 移除卷积层的权重归一化
    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv1)  # 移除第一个卷积层的权重归一化
        nn.utils.remove_weight_norm(self.conv2)  # 移除第二个卷积层的权重归一化
# UnivNetKernelPredictor 类是实现 UnivNet 模型中位置变量卷积层 (LVCs) 的 kernel 和 bias 预测的网络
class UnivNetKernelPredictor(nn.Module):
    """
    实现用于为每个 UnivNet LVCBlock 中的位置变量卷积层 (LVCs) 提供 kernel 和 bias 的 kernel 预测网络。

    基于 [maum-ai/univnet](https://github.com/maum-ai/univnet/blob/9bb2b54838bb6d7ce767131cc7b8b61198bc7558/model/lvcnet.py#L7) 中的 KernelPredictor 实现。

    参数:
        config (UnivNetConfig):
            UnivNetModel 模型的配置。
        conv_kernel_size (int, optional, defaults to 3):
            位置变量卷积层 kernel (卷积权重张量) 的 kernel 大小。
        conv_layers (int, optional, defaults to 4):
            要输出 kernel 和 bias 的位置变量卷积层数量。
    """

    def __init__(
        self,
        config: UnivNetConfig,
        conv_kernel_size: int = 3,
        conv_layers: int = 4,
    ):
        super().__init__()

        # 设置模型中的一些参数
        self.conv_in_channels = config.model_hidden_channels
        self.conv_out_channels = 2 * config.model_hidden_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_layers = conv_layers

        # 计算 kernel 和 bias 通道数
        self.kernel_channels = (
            self.conv_in_channels * self.conv_out_channels * self.conv_kernel_size * self.conv_layers
        )
        self.bias_channels = self.conv_out_channels * self.conv_layers

        # 设置 ResNet 的一些参数
        self.resnet_in_channels = config.num_mel_bins
        self.resnet_hidden_channels = config.kernel_predictor_hidden_channels
        self.resnet_kernel_size = config.kernel_predictor_conv_size
        self.num_blocks = config.kernel_predictor_num_blocks

        self.leaky_relu_slope = config.leaky_relu_slope

        padding = (self.resnet_kernel_size - 1) // 2

        # 定义输入卷积层
        self.input_conv = nn.Conv1d(self.resnet_in_channels, self.resnet_hidden_channels, 5, padding=2, bias=True)

        # 定义 ResBlock 层
        self.resblocks = nn.ModuleList([UnivNetKernelPredictorResidualBlock(config) for _ in range(self.num_blocks)])

        # 定义输出卷积层
        self.kernel_conv = nn.Conv1d(
            self.resnet_hidden_channels, self.kernel_channels, self.resnet_kernel_size, padding=padding, bias=True
        )
        self.bias_conv = nn.Conv1d(
            self.resnet_hidden_channels, self.bias_channels, self.resnet_kernel_size, padding=padding, bias=True
        )
    def forward(self, spectrogram: torch.FloatTensor):
        """
        Maps a conditioning log-mel spectrogram to a tensor of convolutional kernels and biases, for use in location
        variable convolutional layers. Note that the input spectrogram should have shape (batch_size, input_channels,
        seq_length).

        Args:
            spectrogram (`torch.FloatTensor` of shape `(batch_size, input_channels, seq_length)`):
                Tensor containing the log-mel spectrograms.

        Returns:
            Tuple[`torch.FloatTensor, `torch.FloatTensor`]: tuple of tensors where the first element is the tensor of
            location variable convolution kernels of shape `(batch_size, self.conv_layers, self.conv_in_channels,
            self.conv_out_channels, self.conv_kernel_size, seq_length)` and the second element is the tensor of
            location variable convolution biases of shape `(batch_size, self.conv_layers. self.conv_out_channels,
            seq_length)`.
        """
        batch_size, _, seq_length = spectrogram.shape

        hidden_states = self.input_conv(spectrogram)  # 使用输入的频谱图进行卷积
        hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)  # 对卷积结果应用leaky relu激活函数

        for resblock in self.resblocks:  # 循环遍历每个残差块
            hidden_states = resblock(hidden_states)  # 应用残差块

        kernel_hidden_states = self.kernel_conv(hidden_states)  # 使用卷积层对隐藏状态进行卷积
        bias_hidden_states = self.bias_conv(hidden_states)  # 使用卷积层对隐藏状态进行卷积

        # Reshape kernels and biases to appropriate shape
        kernels = kernel_hidden_states.view(  # 重新调整卷积核的形状
            batch_size,
            self.conv_layers,
            self.conv_in_channels,
            self.conv_out_channels,
            self.conv_kernel_size,
            seq_length,
        ).contiguous()
        biases = bias_hidden_states.view(  # 重新调整偏置的形状
            batch_size,
            self.conv_layers,
            self.conv_out_channels,
            seq_length,
        ).contiguous()

        return kernels, biases  # 返回重新调整后的卷积核和偏置

    def apply_weight_norm(self):
        nn.utils.weight_norm(self.input_conv)  # 对输入卷积层应用权重归一化
        for layer in self.resblocks:  # 循环遍历每个残差块
            layer.apply_weight_norm()  # 对每个残差块应用权重归一化
        nn.utils.weight_norm(self.kernel_conv)  # 对卷积核卷积层应用权重归一化
        nn.utils.weight_norm(self.bias_conv)  # 对偏置卷积层应用权重归一化

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.input_conv)  # 移除输入卷积层的权重归一化
        for layer in self.resblocks:  # 循环遍历每个残差块
            layer.remove_weight_norm()  # 移除每个残差块的权重归一化
        nn.utils.remove_weight_norm(self.kernel_conv)  # 移除卷积核卷积层的权重归一化
        nn.utils.remove_weight_norm(self.bias_conv)  # 移除偏置卷积层的权重归一化
# 实现 UnivNet 残差网络的位置变量卷积 (LVC) 残差块
class UnivNetLvcResidualBlock(nn.Module):
    """
    实现 UnivNet 残差网络的位置变量卷积 (LVC) 残差块。

    参数:
        config (`UnivNetConfig`):
            UnivNetModel 模型的配置。
        kernel_size (`int`):
            膨胀 1D 卷积层的核大小。
        dilation (`int`):
            膨胀 1D 卷积层的膨胀率。
    """

    def __init__(
        self,
        config: UnivNetConfig,
        kernel_size: int,
        dilation: int,
    ):
        super().__init__()
        # 获取模型隐藏层通道数
        self.hidden_channels = config.model_hidden_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        # 获取 LeakyReLU 的斜率
        self.leaky_relu_slope = config.leaky_relu_slope

        # 计算卷积层的填充大小
        padding = self.dilation * (self.kernel_size - 1) // 2

        # 构建 1D 卷积层
        self.conv = nn.Conv1d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            padding=padding,
            dilation=self.dilation,
        )

    def forward(self, hidden_states, kernel, bias, hop_size=256):
        # 保存残差
        residual = hidden_states
        # 应用 LeakyReLU 激活函数
        hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
        # 应用 1D 卷积
        hidden_states = self.conv(hidden_states)
        # 再次应用 LeakyReLU 激活函数
        hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
        # 应用位置变量卷积
        hidden_states = self.location_variable_convolution(hidden_states, kernel, bias, hop_size=hop_size)
        
        # 门控激活单元
        hidden_states = torch.sigmoid(hidden_states[:, : self.hidden_channels, :]) * torch.tanh(
            hidden_states[:, self.hidden_channels :, :]
        )
        
        # 跳跃连接
        hidden_states = residual + hidden_states

        return hidden_states

    # 基于 https://github.com/maum-ai/univnet/blob/9bb2b54838bb6d7ce767131cc7b8b61198bc7558/model/lvcnet.py#L171
    def location_variable_convolution(
        self,
        hidden_states: torch.FloatTensor,
        kernel: torch.FloatTensor,
        bias: torch.FloatTensor,
        dilation: int = 1,
        hop_size: int = 256,
    ):
        # 位置变量卷积的实现

    def apply_weight_norm(self):
        # 应用权重归一化
        nn.utils.weight_norm(self.conv)

    def remove_weight_norm(self):
        # 移除权重归一化
        nn.utils.remove_weight_norm(self.conv)


# 实现 UnivNet 残差块中的位置变量卷积 (LVC) 残差块，包含用于预测 LVC 层的核和偏置的 UnivNetKernelPredictor
class UnivNetLvcBlock(nn.Module):
    """
    实现 UnivNet 残差块中的位置变量卷积 (LVC) 残差块，包含用于预测 LVC 层的核和偏置的 UnivNetKernelPredictor。

    基于 [maum-ai/univnet](https://github.com/maum-ai/univnet/blob/9bb2b54838bb6d7ce767131cc7b8b61198bc7558/model/lvcnet.py#L98) 中的 LVCBlock。
    """
    Parameters:
        config (`UnivNetConfig`):
            Config for the `UnivNetModel` model.
        layer_id (`int`):
            An integer corresponding to the index of the current LVC resnet block layer. This should be between 0 and
            `len(config.resblock_stride_sizes) - 1)` inclusive.
        lvc_hop_size (`int`, *optional*, defaults to 256):
            The hop size for the location variable convolutional layers.

    """
    
    # 初始化函数，用于初始化类的实例
    def __init__(
        self,
        config: UnivNetConfig,  # 参数：UnivNetConfig类型的config
        layer_id: int,  # 参数：整型的layer_id
        lvc_hop_size: int = 256,  # 参数：整型的lvc_hop_size，默认值为256
    ):
        super().__init__()  # 调用父类的初始化函数

        # 获取配置信息
        self.hidden_channels = config.model_hidden_channels
        self.kernel_size = config.resblock_kernel_sizes[layer_id]
        self.stride = config.resblock_stride_sizes[layer_id]
        self.dilations = config.resblock_dilation_sizes[layer_id]
        self.cond_hop_length = lvc_hop_size
        self.leaky_relu_slope = config.leaky_relu_slope
        self.num_blocks = len(self.dilations)

        # 创建一个反卷积层
        self.convt_pre = nn.ConvTranspose1d(
            self.hidden_channels,
            self.hidden_channels,
            2 * self.stride,
            stride=self.stride,
            padding=self.stride // 2 + self.stride % 2,
            output_padding=self.stride % 2,
        )

        # 创建一个用于预测卷积核的模型
        self.kernel_predictor = UnivNetKernelPredictor(config, self.kernel_size, self.num_blocks)

        # 创建一个包含多个LVC残差块的ModuleList
        self.resblocks = nn.ModuleList(
            [UnivNetLvcResidualBlock(config, self.kernel_size, self.dilations[i]) for i in range(self.num_blocks)]
        )

    # 前向传播函数
    def forward(self, hidden_states: torch.FloatTensor, spectrogram: torch.FloatTensor):
        # hidden_states: (batch_size, hidden_channels, seq_length)
        # spectrogram: (batch_size, cond_channels, cond_length)
        hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)  # 应用Leaky ReLU激活函数
        hidden_states = self.convt_pre(hidden_states)  # 将输入经过反卷积层处理

        # 获取预测的卷积核
        kernels, biases = self.kernel_predictor(spectrogram)

        # 遍历多个LVC残差块，处理hidden_states
        for i, resblock in enumerate(self.resblocks):
            kernel = kernels[:, i, :, :, :, :]  # 取出第i个卷积核
            bias = biases[:, i, :, :]  # 取出第i个偏置
            hidden_states = resblock(hidden_states, kernel, bias, hop_size=self.cond_hop_length)  # 进行残差块处理

        return hidden_states  # 返回处理后的结果

    # 应用权重归一化
    def apply_weight_norm(self):
        nn.utils.weight_norm(self.convt_pre)
        self.kernel_predictor.apply_weight_norm()
        for layer in self.resblocks:
            layer.apply_weight_norm()

    # 移除权重归一化
    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.convt_pre)
        self.kernel_predictor.remove_weight_norm()
        for layer in self.resblocks:
            layer.remove_weight_norm()
# 定义包含文档字符串的字符串，描述了该模型的继承关系和用法
UNIVNET_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`UnivNetConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义包含输入说明的字符串，描述了将噪声波形和条件谱图转换为语音波形的功能
UNIVNET_INPUTS_DOCSTRING = r"""
    Converts a noise waveform and a conditioning spectrogram to a speech waveform. Passing a batch of log-mel
    spectrograms returns a batch of speech waveforms. Passing a single, un-batched log-mel spectrogram returns a
    single, un-batched speech waveform.

    Args:
        input_features (`torch.FloatTensor`):
            Tensor containing the log-mel spectrograms. Can be batched and of shape `(batch_size, sequence_length,
            config.num_mel_channels)`, or un-batched and of shape `(sequence_length, config.num_mel_channels)`.
        noise_sequence (`torch.FloatTensor`, *optional*):
            Tensor containing a noise sequence of standard Gaussian noise. Can be batched and of shape `(batch_size,
            sequence_length, config.model_in_channels)`, or un-batched and of shape (sequence_length,
            config.model_in_channels)`. If not supplied, will be randomly generated.
        padding_mask (`torch.BoolTensor`, *optional*):
            Mask indicating which parts of each sequence are padded. Mask values are selected in `[0, 1]`:

            - 1 for tokens that are **not masked**
            - 0 for tokens that are **masked**

            The mask can be batched and of shape `(batch_size, sequence_length)` or un-batched and of shape
            `(sequence_length,)`.
        generator (`torch.Generator`, *optional*):
            A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
            deterministic.
        return_dict:
            Whether to return a [`~utils.ModelOutput`] subclass instead of a plain tuple.
"""

# 在类的文档字符串中添加额外的说明
@add_start_docstrings(
    """UnivNet GAN vocoder.""",
    UNIVNET_START_DOCSTRING,
)
# 定义类 UnivNetModel，继承自 PreTrainedModel
class UnivNetModel(PreTrainedModel):
    # 类的配置参数为 UnivNetConfig 类
    config_class = UnivNetConfig
    # 主要输入为 "input_features"
    main_input_name = "input_features"
    # 初始化函数，接受一个 UnivNetConfig 对象作为参数
    def __init__(self, config: UnivNetConfig):
        # 调用父类的初始化函数
        super().__init__(config)

        # 记录 ResNet 块的数量
        self.num_kernels = len(config.resblock_kernel_sizes)
        # 记录 Leaky ReLU 的斜率
        self.leaky_relu_slope = config.leaky_relu_slope

        # 创建一个一维卷积层，用于预处理输入特征
        self.conv_pre = nn.Conv1d(
            config.model_in_channels,
            config.model_hidden_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode="reflect",
        )

        # 初始化位置可变卷积 ResNet 块
        num_layers = len(config.resblock_stride_sizes)
        hop_length = 1
        hop_lengths = []
        # 计算每个 ResNet 块的跳跃长度
        for stride in config.resblock_stride_sizes:
            hop_length = hop_length * stride
            hop_lengths.append(hop_length)

        # 创建包含多个 UnivNetLvcBlock 实例的模块列表
        self.resblocks = nn.ModuleList(
            [
                UnivNetLvcBlock(
                    config,
                    layer_id=i,
                    lvc_hop_size=hop_lengths[i],
                )
                for i in range(num_layers)
            ]
        )

        # 创建用于后处理的一维卷积层
        self.conv_post = nn.Conv1d(config.model_hidden_channels, 1, 7, padding=3, padding_mode="reflect")

        # 初始化权重并进行最终处理
        self.post_init()

    # 前向传播函数，对输入进行处理并返回输出
    @add_start_docstrings_to_model_forward(UNIVNET_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=UnivNetModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_features: torch.FloatTensor,
        noise_sequence: Optional[torch.FloatTensor] = None,
        padding_mask: Optional[torch.FloatTensor] = None,
        generator: Optional[torch.Generator] = None,
        return_dict: Optional[bool] = None,
    ):
    
    # 初始化权重函数，对传入模块的权重进行初始化
    def _init_weights(self, module):
        """Initialize the weights."""
        # 如果模块是线性层、一维卷积层或一维转置卷积层，则初始化权重
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果模块有偏置，则将偏置初始化为零
            if module.bias is not None:
                module.bias.data.zero_()

    # 对模型的卷积层应用权重归一化
    def apply_weight_norm(self):
        # 对预处理卷积层应用权重归一化
        nn.utils.weight_norm(self.conv_pre)
        # 对每个 ResNet 块应用权重归一化
        for layer in self.resblocks:
            layer.apply_weight_norm()
        # 对后处理卷积层应用权重归一化
        nn.utils.weight_norm(self.conv_post)

    # 移除模型的卷积层的权重归一化
    def remove_weight_norm(self):
        # 移除预处理卷积层的权重归一化
        nn.utils.remove_weight_norm(self.conv_pre)
        # 移除每个 ResNet 块的权重归一化
        for layer in self.resblocks:
            layer.remove_weight_norm()
        # 移除后处理卷积层的权重归一化
        nn.utils.remove_weight_norm(self.conv_post)
```