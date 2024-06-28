# `.\models\depth_anything\modeling_depth_anything.py`

```py
# 指定文件编码为 UTF-8

# 版权声明和信息，指出版权归 TikTok 和 HuggingFace Inc. 团队所有，保留所有权利
# 根据 Apache 许可证 2.0 版本，除非符合许可证的要求，否则不得使用此文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 如果适用法律要求或书面同意，本软件是基于“按原样”提供的，没有任何形式的明示或暗示的保证或条件
""" PyTorch Depth Anything model."""

# 引入必要的模块和类型提示
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

# 引入 HuggingFace 提供的实用函数和模块
from ...file_utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import DepthEstimatorOutput
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from ..auto import AutoBackbone

# 引入 DepthAnythingConfig 配置类
from .configuration_depth_anything import DepthAnythingConfig

# 获取 logger 实例
logger = logging.get_logger(__name__)

# 模型配置文档字符串
_CONFIG_FOR_DOC = "DepthAnythingConfig"

# 预训练模型存档列表
DEPTH_ANYTHING_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "LiheYoung/depth-anything-small-hf",
    # 查看所有 Depth Anything 模型的列表：https://huggingface.co/models?filter=depth_anything
]

# Depth Anything 模型的起始文档字符串，描述模型是 PyTorch 的 nn.Module 子类，使用方法和行为可以参考 PyTorch 文档
DEPTH_ANYTHING_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`DepthAnythingConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# Depth Anything 模型的输入参数文档字符串，描述了模型接受的输入参数的详细信息
DEPTH_ANYTHING_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`DPTImageProcessor.__call__`]
            for details.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""

class DepthAnythingReassembleLayer(nn.Module):
    """
    PyTorch 模块用于 Depth Anything 重组层的定义
    """
    def __init__(self, config, channels, factor):
        super().__init__()
        # 创建一个 1x1 的卷积层，用于投影隐藏状态的通道数
        self.projection = nn.Conv2d(in_channels=config.reassemble_hidden_size, out_channels=channels, kernel_size=1)

        # 根据因子选择上/下采样操作
        if factor > 1:
            # 如果因子大于1，则使用转置卷积进行上采样
            self.resize = nn.ConvTranspose2d(channels, channels, kernel_size=factor, stride=factor, padding=0)
        elif factor == 1:
            # 如果因子等于1，则保持形状不变
            self.resize = nn.Identity()
        elif factor < 1:
            # 如果因子小于1，则进行下采样，使用卷积核大小为3，步长为1/factor，填充为1
            self.resize = nn.Conv2d(channels, channels, kernel_size=3, stride=int(1 / factor), padding=1)

    # 从 transformers.models.dpt.modeling_dpt.DPTReassembleLayer.forward 复制过来的方法
    def forward(self, hidden_state):
        # 对隐藏状态进行投影
        hidden_state = self.projection(hidden_state)
        # 对投影后的状态进行尺寸调整（上/下采样）
        hidden_state = self.resize(hidden_state)

        return hidden_state
class DepthAnythingReassembleStage(nn.Module):
    """
    This class reassembles the hidden states of the backbone into image-like feature representations at various
    resolutions.
    
    This happens in 3 stages:
    1. Take the patch embeddings and reshape them to image-like feature representations.
    2. Project the channel dimension of the hidden states according to `config.neck_hidden_sizes`.
    3. Resizing the spatial dimensions (height, width).

    Args:
        config (`[DepthAnythingConfig]`):
            Model configuration class defining the model architecture.
    """

    def __init__(self, config):
        super().__init__()

        # Store the configuration object for the model
        self.config = config

        # Initialize a list to store reassemble layers based on config
        self.layers = nn.ModuleList()
        for channels, factor in zip(config.neck_hidden_sizes, config.reassemble_factors):
            # Append a reassemble layer to the list
            self.layers.append(DepthAnythingReassembleLayer(config, channels=channels, factor=factor))

    def forward(self, hidden_states: List[torch.Tensor], patch_height=None, patch_width=None) -> List[torch.Tensor]:
        """
        Args:
            hidden_states (`List[torch.FloatTensor]`, each of shape `(batch_size, sequence_length + 1, hidden_size)`):
                List of hidden states from the backbone.
        """
        # Initialize an empty list to store output tensors
        out = []

        # Iterate over each hidden state tensor in the input list
        for i, hidden_state in enumerate(hidden_states):
            # Remove the first token from the sequence dimension
            hidden_state = hidden_state[:, 1:]

            # Extract dimensions from the modified hidden state tensor
            batch_size, _, num_channels = hidden_state.shape

            # Reshape the tensor to image-like representation
            hidden_state = hidden_state.reshape(batch_size, patch_height, patch_width, num_channels)

            # Permute dimensions to (batch_size, num_channels, height, width)
            hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()

            # Apply the reassemble layer corresponding to the current index
            hidden_state = self.layers[i](hidden_state)

            # Append the processed tensor to the output list
            out.append(hidden_state)

        # Return the list of reassembled hidden states
        return out


class DepthAnythingPreActResidualLayer(nn.Module):
    """
    ResidualConvUnit, pre-activate residual unit.

    Args:
        config (`[DepthAnythingConfig]`):
            Model configuration class defining the model architecture.
    """

    def __init__(self, config):
        super().__init__()

        # Initialize ReLU activation function
        self.activation1 = nn.ReLU()

        # Initialize the first convolutional layer
        self.convolution1 = nn.Conv2d(
            config.fusion_hidden_size,
            config.fusion_hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        # Initialize ReLU activation function
        self.activation2 = nn.ReLU()

        # Initialize the second convolutional layer
        self.convolution2 = nn.Conv2d(
            config.fusion_hidden_size,
            config.fusion_hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # Preserve the input tensor as residual connection
        residual = hidden_state

        # Apply ReLU activation to the input tensor
        hidden_state = self.activation1(hidden_state)

        # Perform convolution operation using the first convolutional layer
        hidden_state = self.convolution1(hidden_state)

        # Apply ReLU activation to the resulting tensor
        hidden_state = self.activation2(hidden_state)

        # Perform convolution operation using the second convolutional layer
        hidden_state = self.convolution2(hidden_state)

        # Add the residual connection to the final output tensor
        return hidden_state + residual
class DepthAnythingFeatureFusionLayer(nn.Module):
    """Feature fusion layer, merges feature maps from different stages.

    Args:
        config (`[DepthAnythingConfig]`):
            Model configuration class defining the model architecture.
    """

    def __init__(self, config):
        super().__init__()

        # Projection layer to adjust feature map dimensions
        self.projection = nn.Conv2d(config.fusion_hidden_size, config.fusion_hidden_size, kernel_size=1, bias=True)

        # Residual layers for feature fusion
        self.residual_layer1 = DepthAnythingPreActResidualLayer(config)
        self.residual_layer2 = DepthAnythingPreActResidualLayer(config)

    def forward(self, hidden_state, residual=None, size=None):
        # Apply residual connection if residual is provided
        if residual is not None:
            # Resize residual tensor if shapes are different
            if hidden_state.shape != residual.shape:
                residual = nn.functional.interpolate(
                    residual, size=(hidden_state.shape[2], hidden_state.shape[3]), mode="bilinear", align_corners=False
                )
            hidden_state = hidden_state + self.residual_layer1(residual)

        # Apply the second residual layer
        hidden_state = self.residual_layer2(hidden_state)

        # Determine modifier for interpolation
        modifier = {"scale_factor": 2} if size is None else {"size": size}

        # Interpolate the hidden_state tensor
        hidden_state = nn.functional.interpolate(
            hidden_state,
            **modifier,
            mode="bilinear",
            align_corners=True,
        )

        # Project the interpolated feature map using the projection layer
        hidden_state = self.projection(hidden_state)

        return hidden_state


class DepthAnythingFeatureFusionStage(nn.Module):
    # Copied from transformers.models.dpt.modeling_dpt.DPTFeatureFusionStage.__init__ with DPT->DepthAnything
    def __init__(self, config):
        super().__init__()

        # Initialize layers list with DepthAnythingFeatureFusionLayer instances
        self.layers = nn.ModuleList()
        for _ in range(len(config.neck_hidden_sizes)):
            self.layers.append(DepthAnythingFeatureFusionLayer(config))

    def forward(self, hidden_states, size=None):
        # Reverse the order of hidden_states for processing from last to first
        hidden_states = hidden_states[::-1]

        fused_hidden_states = []

        # Process the first layer separately using the last hidden_state
        size = hidden_states[1].shape[2:]  # size for interpolation
        fused_hidden_state = self.layers[0](hidden_states[0], size=size)
        fused_hidden_states.append(fused_hidden_state)

        # Iterate through remaining layers and hidden_states in reverse order
        for idx, (hidden_state, layer) in enumerate(zip(hidden_states[1:], self.layers[1:])):
            size = hidden_states[1:][idx + 1].shape[2:] if idx != (len(hidden_states[1:]) - 1) else None

            # Apply the current layer to fused_hidden_state and current hidden_state
            fused_hidden_state = layer(fused_hidden_state, hidden_state, size=size)

            fused_hidden_states.append(fused_hidden_state)

        return fused_hidden_states


# Copied from transformers.models.dpt.modeling_dpt.DPTPreTrainedModel with DPT->DepthAnything,dpt->depth_anything
class DepthAnythingPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # Configuration class to be used with this model
    config_class = DepthAnythingConfig
    # 设置基础模型的前缀名称
    base_model_prefix = "depth_anything"
    # 设置主输入的名称为"pixel_values"
    main_input_name = "pixel_values"
    # 启用梯度检查点支持
    supports_gradient_checkpointing = True

    # 初始化模型权重的函数
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果是线性层、2D卷积层或转置卷积层，则初始化权重
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            # 使用正态分布初始化权重，均值为0，标准差为self.config.initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置，则将偏置初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是LayerNorm层，则初始化偏置为零，权重为1
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
class DepthAnythingNeck(nn.Module):
    """
    DepthAnythingNeck. A neck is a module that is normally used between the backbone and the head. It takes a list of tensors as
    input and produces another list of tensors as output. For DepthAnything, it includes 2 stages:

    * DepthAnythingReassembleStage
    * DepthAnythingFeatureFusionStage.

    Args:
        config (dict): config dict.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Initialize the reassemble stage using provided configuration
        self.reassemble_stage = DepthAnythingReassembleStage(config)

        # Initialize convolutional layers based on neck_hidden_sizes in the config
        self.convs = nn.ModuleList()
        for channel in config.neck_hidden_sizes:
            self.convs.append(nn.Conv2d(channel, config.fusion_hidden_size, kernel_size=3, padding=1, bias=False))

        # Initialize the fusion stage using provided configuration
        self.fusion_stage = DepthAnythingFeatureFusionStage(config)

    def forward(self, hidden_states: List[torch.Tensor], patch_height=None, patch_width=None) -> List[torch.Tensor]:
        """
        Args:
            hidden_states (`List[torch.FloatTensor]`, each of shape `(batch_size, sequence_length, hidden_size)` or `(batch_size, hidden_size, height, width)`):
                List of hidden states from the backbone.
        """
        # Check if hidden_states is a tuple or list of tensors
        if not isinstance(hidden_states, (tuple, list)):
            raise ValueError("hidden_states should be a tuple or list of tensors")

        # Ensure the number of hidden states matches the number of neck hidden sizes in the config
        if len(hidden_states) != len(self.config.neck_hidden_sizes):
            raise ValueError("The number of hidden states should be equal to the number of neck hidden sizes.")

        # Reassemble hidden states using the reassemble stage
        hidden_states = self.reassemble_stage(hidden_states, patch_height, patch_width)

        # Apply convolutional layers to each hidden state feature
        features = [self.convs[i](feature) for i, feature in enumerate(hidden_states)]

        # Apply fusion stage to the processed features
        output = self.fusion_stage(features)

        return output


class DepthAnythingDepthEstimationHead(nn.Module):
    """
    Output head consisting of 3 convolutional layers. It progressively halves the feature dimension and upsamples
    the predictions to the input resolution after the first convolutional layer (details can be found in the DPT paper's
    supplementary material).
    """

    def __init__(self, config):
        super().__init__()

        # Initialize head_in_index and patch_size from config
        self.head_in_index = config.head_in_index
        self.patch_size = config.patch_size

        features = config.fusion_hidden_size

        # Define convolutional layers with decreasing feature dimensions
        self.conv1 = nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(features // 2, config.head_hidden_size, kernel_size=3, stride=1, padding=1)
        self.activation1 = nn.ReLU()

        # Final convolutional layer with output dimension 1 (for depth estimation)
        self.conv3 = nn.Conv2d(config.head_hidden_size, 1, kernel_size=1, stride=1, padding=0)
        self.activation2 = nn.ReLU()
    # 在输入的隐藏状态中选择与当前头部相关的部分
    hidden_states = hidden_states[self.head_in_index]

    # 第一层卷积，用于生成深度预测
    predicted_depth = self.conv1(hidden_states)

    # 对预测的深度图进行插值操作，调整大小以适应给定的补丁尺寸
    predicted_depth = nn.functional.interpolate(
        predicted_depth,
        (int(patch_height * self.patch_size), int(patch_width * self.patch_size)),
        mode="bilinear",
        align_corners=True,
    )

    # 第二层卷积，处理调整大小后的深度预测
    predicted_depth = self.conv2(predicted_depth)

    # 应用第一个激活函数到调整后的深度预测
    predicted_depth = self.activation1(predicted_depth)

    # 第三层卷积，进一步处理激活后的深度预测
    predicted_depth = self.conv3(predicted_depth)

    # 应用第二个激活函数到最终的深度预测结果
    predicted_depth = self.activation2(predicted_depth)

    # 压缩维度，将深度预测结果从 (batch_size, 1, height, width) 变为 (batch_size, height, width)
    predicted_depth = predicted_depth.squeeze(dim=1)  # shape (batch_size, height, width)

    # 返回最终的深度预测张量
    return predicted_depth
@add_start_docstrings(
    """
    Depth Anything Model with a depth estimation head on top (consisting of 3 convolutional layers) e.g. for KITTI, NYUv2.
    """,
    DEPTH_ANYTHING_START_DOCSTRING,
)
class DepthAnythingForDepthEstimation(DepthAnythingPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化深度估计模型的主干网络
        self.backbone = AutoBackbone.from_config(config.backbone_config)
        
        # 初始化深度估计模型的特征提取网络
        self.neck = DepthAnythingNeck(config)
        
        # 初始化深度估计模型的深度估计头部网络
        self.head = DepthAnythingDepthEstimationHead(config)

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(DEPTH_ANYTHING_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DepthEstimatorOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
```