# `.\models\univnet\modeling_univnet.py`

```
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
""" PyTorch UnivNetModel model."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ...modeling_utils import ModelOutput, PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_univnet import UnivNetConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "UnivNetConfig"

_CHECKPOINT_FOR_DOC = "dg845/univnet-dev"

UNIVNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "dg845/univnet-dev",
    # See all UnivNet models at https://huggingface.co/models?filter=univnet
]


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
        # Initialize the residual block parameters based on the provided configuration
        self.channels = config.model_in_channels  # Number of input channels for the block
        self.kernel_size = config.kernel_predictor_conv_size  # Size of the convolutional kernel
        self.dropout_prob = config.kernel_predictor_dropout  # Dropout probability
        self.leaky_relu_slope = config.leaky_relu_slope  # Slope of the Leaky ReLU activation function

        padding = (self.kernel_size - 1) // 2  # Calculate padding size for convolution

        # Dropout layer to randomly zero some of the input elements with probability self.dropout_prob
        self.dropout = nn.Dropout(self.dropout_prob)
        # First 1D convolutional layer with input channels, output channels, kernel size, and padding
        self.conv1 = nn.Conv1d(self.channels, self.channels, self.kernel_size, padding=padding, bias=True)
        # Second 1D convolutional layer with input channels, output channels, kernel size, and padding
        self.conv2 = nn.Conv1d(self.channels, self.channels, self.kernel_size, padding=padding, bias=True)
    # 对神经网络模型中的前向传播方法进行定义，接受隐藏状态作为输入参数
    def forward(self, hidden_states: torch.FloatTensor):
        # residual用于存储输入的原始隐藏状态，以便后续进行残差连接
        residual = hidden_states
        # 对输入的隐藏状态进行dropout操作，以减少过拟合风险
        hidden_states = self.dropout(hidden_states)
        # 第一层卷积操作，将dropout后的隐藏状态作为输入
        hidden_states = self.conv1(hidden_states)
        # 使用LeakyReLU激活函数对第一层卷积的输出进行非线性变换
        hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
        # 第二层卷积操作，将第一层卷积的输出作为输入
        hidden_states = self.conv2(hidden_states)
        # 再次使用LeakyReLU激活函数对第二层卷积的输出进行非线性变换
        hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)
        # 返回经过卷积和激活函数处理后的隐藏状态，加上之前保存的残差
        return hidden_states + residual
    
    # 对模型中的卷积层应用权重归一化（weight normalization）
    def apply_weight_norm(self):
        # 对第一层卷积层应用权重归一化
        nn.utils.weight_norm(self.conv1)
        # 对第二层卷积层应用权重归一化
        nn.utils.weight_norm(self.conv2)
    
    # 移除模型中的卷积层的权重归一化
    def remove_weight_norm(self):
        # 移除第一层卷积层的权重归一化
        nn.utils.remove_weight_norm(self.conv1)
        # 移除第二层卷积层的权重归一化
        nn.utils.remove_weight_norm(self.conv2)
class UnivNetKernelPredictor(nn.Module):
    """
    Implementation of the kernel predictor network which supplies the kernel and bias for the location variable
    convolutional layers (LVCs) in each UnivNet LVCBlock.

    Based on the KernelPredictor implementation in
    [maum-ai/univnet](https://github.com/maum-ai/univnet/blob/9bb2b54838bb6d7ce767131cc7b8b61198bc7558/model/lvcnet.py#L7).

    Parameters:
        config: (`UnivNetConfig`):
            Config for the `UnivNetModel` model.
        conv_kernel_size (`int`, *optional*, defaults to 3):
            The kernel size for the location variable convolutional layer kernels (convolutional weight tensor).
        conv_layers (`int`, *optional*, defaults to 4):
            The number of location variable convolutional layers to output kernels and biases for.
    """

    def __init__(
        self,
        config: UnivNetConfig,
        conv_kernel_size: int = 3,
        conv_layers: int = 4,
    ):
        super().__init__()

        # 设置卷积层输入通道数为模型隐藏通道数
        self.conv_in_channels = config.model_hidden_channels
        # 设置卷积层输出通道数为模型隐藏通道数的两倍
        self.conv_out_channels = 2 * config.model_hidden_channels
        # 设置卷积核大小为给定参数值
        self.conv_kernel_size = conv_kernel_size
        # 设置卷积层数为给定参数值
        self.conv_layers = conv_layers

        # 计算卷积核的总通道数，考虑了通道数、卷积核大小和卷积层数
        self.kernel_channels = (
            self.conv_in_channels * self.conv_out_channels * self.conv_kernel_size * self.conv_layers
        )
        # 计算偏置的总通道数，考虑了输出通道数和卷积层数
        self.bias_channels = self.conv_out_channels * self.conv_layers

        # 设置 ResNet 的输入通道数为 Mel 频谱的数量
        self.resnet_in_channels = config.num_mel_bins
        # 设置 ResNet 隐藏层的通道数为给定的隐藏通道数
        self.resnet_hidden_channels = config.kernel_predictor_hidden_channels
        # 设置 ResNet 卷积核大小为给定的卷积核大小
        self.resnet_kernel_size = config.kernel_predictor_conv_size
        # 设置 ResNet 的块数量为给定的块数
        self.num_blocks = config.kernel_predictor_num_blocks

        # 设置 Leaky ReLU 的负斜率为给定的斜率
        self.leaky_relu_slope = config.leaky_relu_slope

        # 计算卷积的填充大小，确保卷积核能够处理边界
        padding = (self.resnet_kernel_size - 1) // 2

        # 输入卷积层，接受 Mel 频谱作为输入，输出到 ResNet 的隐藏层
        self.input_conv = nn.Conv1d(self.resnet_in_channels, self.resnet_hidden_channels, 5, padding=2, bias=True)

        # 创建 ResNet 块的列表，每个块是 UnivNetKernelPredictorResidualBlock 类的实例
        self.resblocks = nn.ModuleList([UnivNetKernelPredictorResidualBlock(config) for _ in range(self.num_blocks)])

        # 输出卷积层，生成卷积核参数，以适应 LVC 的位置变量卷积层
        self.kernel_conv = nn.Conv1d(
            self.resnet_hidden_channels, self.kernel_channels, self.resnet_kernel_size, padding=padding, bias=True
        )
        # 输出卷积层，生成偏置参数，以适应 LVC 的位置变量卷积层
        self.bias_conv = nn.Conv1d(
            self.resnet_hidden_channels, self.bias_channels, self.resnet_kernel_size, padding=padding, bias=True
        )
    def forward(self, spectrogram: torch.FloatTensor):
        """
        将一个条件化的对数梅尔频谱映射到卷积核和偏置的张量，用于位置变量卷积层。注意输入的频谱应具有形状 (batch_size, input_channels, seq_length)。

        Args:
            spectrogram (`torch.FloatTensor` of shape `(batch_size, input_channels, seq_length)`):
                包含对数梅尔频谱的张量。

        Returns:
            Tuple[`torch.FloatTensor, `torch.FloatTensor`]: 一个元组，第一个元素是形状为 `(batch_size, self.conv_layers, self.conv_in_channels,
            self.conv_out_channels, self.conv_kernel_size, seq_length)` 的位置变量卷积核张量，第二个元素是形状为 `(batch_size, self.conv_layers, self.conv_out_channels,
            seq_length)` 的位置变量卷积偏置张量。
        """
        batch_size, _, seq_length = spectrogram.shape  # 获取批次大小、输入通道数和序列长度

        hidden_states = self.input_conv(spectrogram)  # 应用输入卷积层
        hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)  # 应用泄漏整流激活函数

        for resblock in self.resblocks:  # 遍历所有残差块
            hidden_states = resblock(hidden_states)  # 应用残差块

        kernel_hidden_states = self.kernel_conv(hidden_states)  # 应用核卷积层
        bias_hidden_states = self.bias_conv(hidden_states)  # 应用偏置卷积层

        # 将卷积核和偏置重塑为适当的形状
        kernels = kernel_hidden_states.view(
            batch_size,
            self.conv_layers,
            self.conv_in_channels,
            self.conv_out_channels,
            self.conv_kernel_size,
            seq_length,
        ).contiguous()
        biases = bias_hidden_states.view(
            batch_size,
            self.conv_layers,
            self.conv_out_channels,
            seq_length,
        ).contiguous()

        return kernels, biases

    def apply_weight_norm(self):
        nn.utils.weight_norm(self.input_conv)  # 对输入卷积层应用权重归一化
        for layer in self.resblocks:  # 对所有残差块应用权重归一化
            layer.apply_weight_norm()
        nn.utils.weight_norm(self.kernel_conv)  # 对核卷积层应用权重归一化
        nn.utils.weight_norm(self.bias_conv)  # 对偏置卷积层应用权重归一化

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.input_conv)  # 移除输入卷积层的权重归一化
        for layer in self.resblocks:  # 移除所有残差块的权重归一化
            layer.remove_weight_norm()
        nn.utils.remove_weight_norm(self.kernel_conv)  # 移除核卷积层的权重归一化
        nn.utils.remove_weight_norm(self.bias_conv)  # 移除偏置卷积层的权重归一化
class UnivNetLvcResidualBlock(nn.Module):
    """
    Implementation of the location variable convolution (LVC) residual block for the UnivNet residual network.

    Parameters:
        config: (`UnivNetConfig`):
            Config for the `UnivNetModel` model.
        kernel_size (`int`):
            The kernel size for the dilated 1D convolutional layer.
        dilation (`int`):
            The dilation for the dilated 1D convolutional layer.
    """

    def __init__(
        self,
        config: UnivNetConfig,
        kernel_size: int,
        dilation: int,
    ):
        super().__init__()
        self.hidden_channels = config.model_hidden_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.leaky_relu_slope = config.leaky_relu_slope

        # Calculate padding for the convolution layer
        padding = self.dilation * (self.kernel_size - 1) // 2

        # Define the 1D convolutional layer with specified parameters
        self.conv = nn.Conv1d(
            self.hidden_channels,
            self.hidden_channels,
            self.kernel_size,
            padding=padding,
            dilation=self.dilation,
        )

    def forward(self, hidden_states, kernel, bias, hop_size=256):
        # Store the input hidden_states as residual for skip connection
        residual = hidden_states

        # Apply leaky ReLU activation function to the input
        hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)

        # Perform dilated 1D convolution using the defined convolution layer
        hidden_states = self.conv(hidden_states)

        # Apply leaky ReLU activation function again
        hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)

        # Apply location variable convolution (LVC) using custom function
        hidden_states = self.location_variable_convolution(hidden_states, kernel, bias, hop_size=hop_size)

        # Apply gated activation unit: sigmoid and tanh functions
        hidden_states = torch.sigmoid(hidden_states[:, : self.hidden_channels, :]) * torch.tanh(
            hidden_states[:, self.hidden_channels :, :]
        )

        # Add the residual (skip connection) to the processed hidden states
        hidden_states = residual + hidden_states

        return hidden_states

    # Custom method for applying weight normalization to the convolution layer
    def apply_weight_norm(self):
        nn.utils.weight_norm(self.conv)

    # Custom method for removing weight normalization from the convolution layer
    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv)
    Parameters:
        config (`UnivNetConfig`):
            `UnivNetModel`模型的配置。
        layer_id (`int`):
            当前LVC ResNet块层的索引，应在0到`len(config.resblock_stride_sizes) - 1`之间（包括边界）。
        lvc_hop_size (`int`, *可选*, 默认为256):
            位置变量卷积层的跳跃步长。

    """
    
    def __init__(
        self,
        config: UnivNetConfig,
        layer_id: int,
        lvc_hop_size: int = 256,
    ):
        super().__init__()
        self.hidden_channels = config.model_hidden_channels  # 设置隐藏通道数
        self.kernel_size = config.resblock_kernel_sizes[layer_id]  # 根据层索引获取内核大小
        self.stride = config.resblock_stride_sizes[layer_id]  # 根据层索引获取步幅大小
        self.dilations = config.resblock_dilation_sizes[layer_id]  # 根据层索引获取扩张率列表
        self.cond_hop_length = lvc_hop_size  # 设置条件跳跃长度
        self.leaky_relu_slope = config.leaky_relu_slope  # 设置LeakyReLU的斜率
        self.num_blocks = len(self.dilations)  # 获取块的数量

        self.convt_pre = nn.ConvTranspose1d(
            self.hidden_channels,  # 输入通道数
            self.hidden_channels,  # 输出通道数
            2 * self.stride,  # 内核大小
            stride=self.stride,  # 步幅大小
            padding=self.stride // 2 + self.stride % 2,  # 填充大小
            output_padding=self.stride % 2,  # 输出填充大小
        )

        self.kernel_predictor = UnivNetKernelPredictor(config, self.kernel_size, self.num_blocks)  # 初始化内核预测器

        self.resblocks = nn.ModuleList(
            [UnivNetLvcResidualBlock(config, self.kernel_size, self.dilations[i]) for i in range(self.num_blocks)]
        )  # 创建LVC残差块列表

    def forward(self, hidden_states: torch.FloatTensor, spectrogram: torch.FloatTensor):
        # hidden_states: (batch_size, hidden_channels, seq_length)
        # spectrogram: (batch_size, cond_channels, cond_length)
        hidden_states = nn.functional.leaky_relu(hidden_states, self.leaky_relu_slope)  # 应用LeakyReLU激活函数
        hidden_states = self.convt_pre(hidden_states)  # 执行转置卷积预处理

        kernels, biases = self.kernel_predictor(spectrogram)  # 从谱图预测内核和偏置

        for i, resblock in enumerate(self.resblocks):
            kernel = kernels[:, i, :, :, :, :]  # 获取当前块的内核
            bias = biases[:, i, :, :]  # 获取当前块的偏置
            hidden_states = resblock(hidden_states, kernel, bias, hop_size=self.cond_hop_length)  # 执行残差块操作

        return hidden_states  # 返回处理后的隐藏状态

    def apply_weight_norm(self):
        nn.utils.weight_norm(self.convt_pre)  # 应用权重归一化到转置卷积层
        self.kernel_predictor.apply_weight_norm()  # 应用权重归一化到内核预测器
        for layer in self.resblocks:
            layer.apply_weight_norm()  # 依次应用权重归一化到每个残差块

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.convt_pre)  # 移除转置卷积层的权重归一化
        self.kernel_predictor.remove_weight_norm()  # 移除内核预测器的权重归一化
        for layer in self.resblocks:
            layer.remove_weight_norm()  # 依次移除每个残差块的权重归一化
# 包含关于 UnivNetModel 类的开始文档字符串，描述了该类的继承和基本使用方法
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

# 包含关于 UnivNetModel 类的输入文档字符串，描述了输入参数及其形状的详细信息
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

# 使用 @add_start_docstrings 装饰器添加关于 UnivNetModel 类的简要描述和详细文档字符串
@add_start_docstrings(
    """UnivNet GAN vocoder.""",
    UNIVNET_START_DOCSTRING,
)
# 定义 UnivNetModel 类，继承自 PreTrainedModel，表示一个 UnivNet GAN 声码器模型
class UnivNetModel(PreTrainedModel):
    # 指定该模型的配置类为 UnivNetConfig
    config_class = UnivNetConfig
    # 指定主要输入的名称为 "input_features"
    main_input_name = "input_features"
    def __init__(self, config: UnivNetConfig):
        super().__init__(config)

        self.num_kernels = len(config.resblock_kernel_sizes)  # 计算 ResNet 块的内核数目
        self.leaky_relu_slope = config.leaky_relu_slope  # 从配置中获取 Leaky ReLU 的斜率

        self.conv_pre = nn.Conv1d(
            config.model_in_channels,
            config.model_hidden_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode="reflect",
        )
        # 创建预处理卷积层，用于输入数据的初始处理

        # Initialize location-variable convolution ResNet Blocks.
        num_layers = len(config.resblock_stride_sizes)  # 获取 ResNet 块的层数
        hop_length = 1
        hop_lengths = []
        for stride in config.resblock_stride_sizes:
            hop_length = hop_length * stride
            hop_lengths.append(hop_length)
        # 计算每个 ResNet 块的跳跃长度，并存储在列表中

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
        # 创建 ResNet 块的列表，每个块都使用不同的位置变量卷积设置

        self.conv_post = nn.Conv1d(config.model_hidden_channels, 1, 7, padding=3, padding_mode="reflect")
        # 创建后处理卷积层，用于最终输出的处理

        # Initialize weights and apply final processing
        self.post_init()
        # 调用初始化方法，用于权重初始化和最终处理的应用

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
        # 正向传播方法，详细文档说明见装饰器函数

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        # 初始化模块的权重，适用于线性层、卷积层和转置卷积层

    def apply_weight_norm(self):
        nn.utils.weight_norm(self.conv_pre)
        for layer in self.resblocks:
            layer.apply_weight_norm()
        nn.utils.weight_norm(self.conv_post)
        # 应用权重归一化到预处理卷积层、ResNet 块和后处理卷积层

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv_pre)
        for layer in self.resblocks:
            layer.remove_weight_norm()
        nn.utils.remove_weight_norm(self.conv_post)
        # 移除预处理卷积层、ResNet 块和后处理卷积层的权重归一化
```