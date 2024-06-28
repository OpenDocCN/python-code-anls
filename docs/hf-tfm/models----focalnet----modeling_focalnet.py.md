# `.\models\focalnet\modeling_focalnet.py`

```
# coding=utf-8
# Copyright 2023 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch FocalNet model."""

# Import necessary modules for the FocalNet model
import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# Import utilities and functions from Hugging Face libraries
from ...activations import ACT2FN
from ...modeling_outputs import BackboneOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ...utils.backbone_utils import BackboneMixin
from .configuration_focalnet import FocalNetConfig

# Initialize logger for the current module
logger = logging.get_logger(__name__)

# General docstring for documentation purposes
_CONFIG_FOR_DOC = "FocalNetConfig"

# Base docstring for checkpoint information
_CHECKPOINT_FOR_DOC = "microsoft/focalnet-tiny"
_EXPECTED_OUTPUT_SHAPE = [1, 49, 768]

# Image classification docstring for model usage
_IMAGE_CLASS_CHECKPOINT = "microsoft/focalnet-tiny"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

# List of pretrained model archive paths
FOCALNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/focalnet-tiny",
    # See all FocalNet models at https://huggingface.co/models?filter=focalnet
]

# Define a dataclass for FocalNetEncoderOutput extending ModelOutput
@dataclass
class FocalNetEncoderOutput(ModelOutput):
    """
    FocalNet encoder's outputs, with potential hidden states.
    This dataclass inherits from ModelOutput provided by Hugging Face.
    """
    # 定义函数参数和返回值的类型注释
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的隐藏状态输出序列，形状为(batch_size, sequence_length, hidden_size)。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            模型每一层的隐藏状态组成的元组，包括初始嵌入层输出。
            形状为(batch_size, sequence_length, hidden_size)。
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            模型每一层的隐藏状态组成的元组，包括初始嵌入层输出，并且重新整形以包括空间维度。
            形状为(batch_size, hidden_size, height, width)。
    
    # 初始化函数的返回值，分别为None类型，表示初始情况下未赋值
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
# 使用 `dataclass` 装饰器定义一个数据类，表示 FocalNet 模型的输出，继承自 `ModelOutput` 类。
@dataclass
class FocalNetModelOutput(ModelOutput):
    """
    FocalNet model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`, *optional*, returned when `add_pooling_layer=True` is passed):
            Average pooling of the last layer hidden-state.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, hidden_size, height, width)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
            include the spatial dimensions.
    """

    # 定义类成员 `last_hidden_state`，表示模型最后一层的隐藏状态
    last_hidden_state: torch.FloatTensor = None
    # 定义类成员 `pooler_output`，表示最后一层隐藏状态的平均池化结果，可选，当 `add_pooling_layer=True` 时返回
    pooler_output: Optional[torch.FloatTensor] = None
    # 定义类成员 `hidden_states`，表示每一层模型的隐藏状态的元组，可选，当 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义类成员 `reshaped_hidden_states`，表示每一层模型隐藏状态的元组，且包括空间维度的重塑，可选，当 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class FocalNetMaskedImageModelingOutput(ModelOutput):
    """
    FocalNet masked image model outputs.
    """

    # 这是一个空的数据类，用于表示 FocalNet 模型处理掩膜图像后的输出
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `bool_masked_pos` is provided):
            Masked image modeling (MLM) loss.
            图像模型的掩码损失，如果提供了 `bool_masked_pos`，则返回。
        reconstruction (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Reconstructed pixel values.
            重建后的像素数值，形状为 `(batch_size, num_channels, height, width)`。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` representing hidden-states of the model at the output of each layer plus the initial embedding outputs.
            模型在每层输出和初始嵌入输出时的隐藏状态的元组。
            形状为 `(batch_size, sequence_length, hidden_size)`。
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` representing hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to include the spatial dimensions.
            模型在每层输出和初始嵌入输出时的隐藏状态的元组，已重塑以包括空间维度。
            形状为 `(batch_size, hidden_size, height, width)`。

    """

    # 可选的损失值，如果没有提供将为 None
    loss: Optional[torch.FloatTensor] = None
    # 可选的重建像素值
    reconstruction: torch.FloatTensor = None
    # 可选的隐藏状态，表示模型在每层输出和初始嵌入输出时的隐藏状态
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 可选的重塑后的隐藏状态，表示模型在每层输出和初始嵌入输出时的隐藏状态，并已重塑以包括空间维度
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
@dataclass
class FocalNetImageClassifierOutput(ModelOutput):
    """
    FocalNet outputs for image classification.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, hidden_size, height, width)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
            include the spatial dimensions.
    """

    loss: Optional[torch.FloatTensor] = None  # 损失值，用于分类或回归任务的损失
    logits: torch.FloatTensor = None  # 分类（或回归）得分，经过 SoftMax 之前的输出
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 每个层输出的隐藏状态，包括初始嵌入输出
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 每个层输出的隐藏状态，包括空间维度的重塑

class FocalNetEmbeddings(nn.Module):
    """
    Construct the patch embeddings and layernorm. Optionally, also the mask token.
    """

    def __init__(self, config, use_mask_token=False):
        super().__init__()

        self.patch_embeddings = FocalNetPatchEmbeddings(
            config=config,
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_channels=config.num_channels,
            embed_dim=config.embed_dim,
            use_conv_embed=config.use_conv_embed,
            is_stem=True,
        )  # 创建图像的补丁嵌入
        self.patch_grid = self.patch_embeddings.grid_size  # 获取补丁嵌入的网格大小
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim)) if use_mask_token else None  # 可选地创建掩码令牌

        self.norm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)  # LayerNorm 归一化层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # 随机失活层

    def forward(
        self, pixel_values: Optional[torch.FloatTensor], bool_masked_pos: Optional[torch.BoolTensor] = None
        # 前向传播方法，接收像素值和可选的掩码位置张量

        )
        # 前向传播方法，接收像素值和可选的掩码位置张量
    ) -> Tuple[torch.Tensor]:
        # 获取图像的补丁嵌入和输出维度信息
        embeddings, output_dimensions = self.patch_embeddings(pixel_values)
        # 对嵌入向量进行归一化处理
        embeddings = self.norm(embeddings)
        # 获取当前批次的大小和序列长度
        batch_size, seq_len, _ = embeddings.size()

        # 如果存在布尔类型的遮罩位置信息
        if bool_masked_pos is not None:
            # 将遮罩标记扩展到与嵌入向量相同的维度
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # 将布尔类型的遮罩位置转换成与mask_tokens相同类型的张量
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            # 使用遮罩标记替换被遮罩的视觉标记
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # 对嵌入向量应用dropout操作
        embeddings = self.dropout(embeddings)
        # 返回处理后的嵌入向量和输出维度信息
        return embeddings, output_dimensions
class FocalNetPatchEmbeddings(nn.Module):
    def __init__(
        self,
        config,
        image_size,
        patch_size,
        num_channels,
        embed_dim,
        add_norm=False,
        use_conv_embed=False,
        is_stem=False,
    ):
        super().__init__()
        # 将图像大小和补丁大小转换为元组，如果它们不是可迭代对象，则分别使用默认值
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        # 计算图像中的补丁数量
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        # 计算网格大小，即图像尺寸与补丁尺寸的整除结果
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])

        if use_conv_embed:
            # 如果选择使用卷积嵌入，则根据是否是 stem 层选择不同的卷积参数
            if is_stem:
                kernel_size = 7
                padding = 2
                stride = 4
            else:
                kernel_size = 3
                padding = 1
                stride = 2
            # 设置卷积投影层，根据参数创建卷积层对象
            self.projection = nn.Conv2d(
                num_channels, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
            )
        else:
            # 否则，使用常规的卷积设置补丁大小作为卷积核大小和步幅
            self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        if add_norm:
            # 如果指定要添加 LayerNorm，则创建 LayerNorm 层
            self.norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        else:
            # 否则，不添加标准化层
            self.norm = None

    def maybe_pad(self, pixel_values, height, width):
        # 如果图像宽度不能被补丁宽度整除，则对像素值进行填充
        if width % self.patch_size[1] != 0:
            pad_values = (0, self.patch_size[1] - width % self.patch_size[1])
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        # 如果图像高度不能被补丁高度整除，则对像素值进行填充
        if height % self.patch_size[0] != 0:
            pad_values = (0, 0, 0, self.patch_size[0] - height % self.patch_size[0])
            pixel_values = nn.functional.pad(pixel_values, pad_values)
        return pixel_values

    def forward(self, pixel_values: Optional[torch.FloatTensor]) -> Tuple[torch.Tensor, Tuple[int]]:
        # 获取输入张量的形状信息
        _, num_channels, height, width = pixel_values.shape
        # 检查通道数是否与配置中指定的数值相符合
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 对输入像素进行可能的填充，使其能够被补丁大小整除
        pixel_values = self.maybe_pad(pixel_values, height, width)
        # 使用投影层进行特征提取，得到嵌入特征张量
        embeddings = self.projection(pixel_values)
        # 获取嵌入特征张量的新形状信息
        _, _, height, width = embeddings.shape
        output_dimensions = (height, width)
        # 对嵌入特征进行展平和转置操作，以便后续处理
        embeddings = embeddings.flatten(2).transpose(1, 2)

        if self.norm is not None:
            # 如果存在 LayerNorm 层，则对嵌入特征进行标准化处理
            embeddings = self.norm(embeddings)

        return embeddings, output_dimensions


# Copied from transformers.models.beit.modeling_beit.drop_path
# 定义一个函数用于在神经网络中应用路径丢弃（Stochastic Depth），每个样本都可能执行该操作（当应用于残差块的主路径时）。
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    # 如果丢失概率为0或者当前非训练状态，则直接返回输入
    if drop_prob == 0.0 or not training:
        return input
    # 计算保留概率
    keep_prob = 1 - drop_prob
    # 创建一个与输入形状兼容的随机张量，用于随机选择保留的路径
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # 适用于不同维度的张量，而不仅仅是二维卷积网络
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # 对随机张量进行二值化处理
    # 对输入进行路径丢弃操作，并返回处理后的输出
    output = input.div(keep_prob) * random_tensor
    return output


# 从transformers.models.beit.modeling_beit.BeitDropPath复制并更改为FocalNet
class FocalNetDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 调用上面定义的drop_path函数来实现路径丢弃操作
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        # 返回当前模块的额外描述信息，这里是丢弃概率(drop_prob)
        return "p={}".format(self.drop_prob)


class FocalNetModulation(nn.Module):
    # 这里可以添加FocalNetModulation的具体实现，以进行FocalNet的特定调制
    def __init__(self, config, index, dim, focal_factor=2, bias=True, projection_dropout=0.0):
        super().__init__()

        self.dim = dim  # 设置对象的维度属性
        self.focal_window = config.focal_windows[index]  # 获取配置中的焦点窗口大小
        self.focal_level = config.focal_levels[index]  # 获取配置中的焦点级别
        self.focal_factor = focal_factor  # 设置焦点因子
        self.use_post_layernorm_in_modulation = config.use_post_layernorm_in_modulation  # 是否使用后层标准化调制
        self.normalize_modulator = config.normalize_modulator  # 是否标准化调制器

        self.projection_in = nn.Linear(dim, 2 * dim + (self.focal_level + 1), bias=bias)  # 输入投影层
        self.projection_context = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=bias)  # 上下文投影卷积层

        self.activation = nn.GELU()  # 激活函数
        self.projection_out = nn.Linear(dim, dim)  # 输出投影层
        self.projection_dropout = nn.Dropout(projection_dropout)  # 投影层的dropout
        self.focal_layers = nn.ModuleList()  # 焦点层列表

        self.kernel_sizes = []  # 焦点层的卷积核尺寸列表
        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window  # 计算每个焦点层的卷积核尺寸
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(
                        dim, dim, kernel_size=kernel_size, stride=1, groups=dim, padding=kernel_size // 2, bias=False
                    ),  # 焦点层的卷积操作
                    nn.GELU(),  # 焦点层后的激活函数
                )
            )
            self.kernel_sizes.append(kernel_size)  # 将卷积核尺寸添加到列表中
        if self.use_post_layernorm_in_modulation:
            self.layernorm = nn.LayerNorm(dim, eps=config.layer_norm_eps)  # 后层标准化层

    def forward(self, hidden_state):
        """
        Args:
            hidden_state:
                Input features with shape of (batch_size, height, width, num_channels)
        """
        num_channels = hidden_state.shape[-1]  # 获取输入张量中的通道数

        # pre linear projection
        x = self.projection_in(hidden_state).permute(0, 3, 1, 2).contiguous()  # 线性投影操作，并对张量维度进行转置和连续化处理
        q, ctx, self.gates = torch.split(x, (num_channels, num_channels, self.focal_level + 1), 1)  # 按通道数切分张量为q, ctx和门控信号

        # context aggreation
        ctx_all = 0  # 初始化上下文聚合变量
        for level in range(self.focal_level):
            ctx = self.focal_layers[level](ctx)  # 使用每个焦点层处理上下文
            ctx_all = ctx_all + ctx * self.gates[:, level : level + 1]  # 加权累积上下文特征
        ctx_global = self.activation(ctx.mean(2, keepdim=True).mean(3, keepdim=True))  # 全局上下文特征的平均池化和激活处理
        ctx_all = ctx_all + ctx_global * self.gates[:, self.focal_level :]  # 添加全局上下文特征的加权结果

        # normalize context
        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level + 1)  # 如果需要，对上下文进行标准化

        # focal modulation
        self.modulator = self.projection_context(ctx_all)  # 使用上下文调制器对输入进行调制
        x_out = q * self.modulator  # 根据调制结果对q进行调制
        x_out = x_out.permute(0, 2, 3, 1).contiguous()  # 对输出张量进行转置和连续化处理
        if self.use_post_layernorm_in_modulation:
            x_out = self.layernorm(x_out)  # 如果需要，对调制后的输出进行后层标准化处理

        # post linear projection
        x_out = self.projection_out(x_out)  # 输出层的线性投影
        x_out = self.projection_dropout(x_out)  # 输出层的dropout处理
        return x_out  # 返回最终的输出张量
# 定义一个名为 FocalNetLayer 的自定义神经网络层
class FocalNetLayer(nn.Module):
    r"""Focal Modulation Network layer (block).

    Args:
        config (`FocalNetConfig`):
            Model config.
        index (`int`):
            Layer index.
        dim (`int`):
            Number of input channels.
        input_resolution (`Tuple[int]`):
            Input resolution.
        drop_path (`float`, *optional*, defaults to 0.0):
            Stochastic depth rate.
    """

    # 初始化函数，用于设置层的各种属性和参数
    def __init__(self, config, index, dim, input_resolution, drop_path=0.0):
        super().__init__()

        self.config = config

        # 设置层特定的属性
        self.dim = dim  # 输入通道数
        self.input_resolution = input_resolution  # 输入分辨率

        # 设置通用属性
        self.drop = config.hidden_dropout_prob  # 隐藏层的 Dropout 概率
        self.use_post_layernorm = config.use_post_layernorm  # 是否使用层归一化

        # 第一个层归一化模块
        self.norm1 = nn.LayerNorm(dim, eps=config.layer_norm_eps)

        # FocalNetModulation 类的实例化，用于模块化调节
        self.modulation = FocalNetModulation(
            config=config,
            index=index,
            dim=dim,
            projection_dropout=self.drop,
        )

        # 根据 drop_path 参数选择是否应用随机深度
        self.drop_path = FocalNetDropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # 第二个层归一化模块
        self.norm2 = nn.LayerNorm(dim, eps=config.layer_norm_eps)

        # 计算 MLP 隐藏层的维度
        mlp_hidden_dim = int(dim * config.mlp_ratio)

        # 实例化 FocalNetMlp 类，定义 MLP 结构
        self.mlp = FocalNetMlp(config=config, in_features=dim, hidden_features=mlp_hidden_dim, drop=self.drop)

        # 初始化 layerscale 的 gamma 参数
        self.gamma_1 = 1.0
        self.gamma_2 = 1.0
        if config.use_layerscale:
            self.gamma_1 = nn.Parameter(config.layerscale_value * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(config.layerscale_value * torch.ones((dim)), requires_grad=True)
    # 定义前向传播函数，接收隐藏状态和输入尺寸作为参数
    def forward(self, hidden_state, input_dimensions):
        # 解包输入尺寸为高度和宽度
        height, width = input_dimensions
        # 获取隐藏状态的批大小、深度和通道数
        batch_size, _, num_channels = hidden_state.shape
        # 保存原始的隐藏状态作为快捷方式
        shortcut = hidden_state

        # Focal Modulation（集中调制）
        # 如果未使用后层归一化，则对隐藏状态进行归一化处理
        hidden_state = hidden_state if self.use_post_layernorm else self.norm1(hidden_state)
        # 将隐藏状态重新调整形状为(batch_size, height, width, num_channels)
        hidden_state = hidden_state.view(batch_size, height, width, num_channels)
        # 应用调制器（modulation）到隐藏状态，再将其展平为(batch_size, height * width, num_channels)
        hidden_state = self.modulation(hidden_state).view(batch_size, height * width, num_channels)
        # 如果使用后层归一化，则再次对隐藏状态进行归一化处理
        hidden_state = hidden_state if not self.use_post_layernorm else self.norm1(hidden_state)

        # FFN（Feed Forward Network，前馈神经网络）
        # 结合快捷方式和经过DropPath处理的隐藏状态乘以gamma_1
        hidden_state = shortcut + self.drop_path(self.gamma_1 * hidden_state)
        # 将DropPath处理后的MLP输出乘以gamma_2加回到隐藏状态上
        hidden_state = hidden_state + self.drop_path(
            self.gamma_2
            * (self.norm2(self.mlp(hidden_state)) if self.use_post_layernorm else self.mlp(self.norm2(hidden_state)))
        )

        # 返回最终的隐藏状态
        return hidden_state
# 定义 FocalNetStage 类，继承自 nn.Module，用于 FocalNet 的每个阶段处理
class FocalNetStage(nn.Module):
    # 初始化方法，接收配置、阶段索引和输入分辨率作为参数
    def __init__(self, config, index, input_resolution):
        super().__init__()

        # 将配置参数保存到实例变量中
        self.config = config
        # 计算深度列表的长度，即阶段数
        self.num_stages = len(config.depths)

        # 计算当前阶段的嵌入维度和输出维度
        embed_dim = [config.embed_dim * (2**i) for i in range(self.num_stages)]
        dim = embed_dim[index]
        out_dim = embed_dim[index + 1] if (index < self.num_stages - 1) else None
        # 如果不是最后一个阶段，则设置下采样函数
        downsample = FocalNetPatchEmbeddings if (index < self.num_stages - 1) else None

        # 根据随机深度衰减规则生成当前阶段的丢弃路径率
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]
        drop_path = dpr[sum(config.depths[:index]):sum(config.depths[:index + 1])]

        # 创建当前阶段的层列表，每一层使用 FocalNetLayer 类处理
        self.layers = nn.ModuleList(
            [
                FocalNetLayer(
                    config=config,
                    index=index,
                    dim=dim,
                    input_resolution=input_resolution,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                )
                for i in range(config.depths[index])
            ]
        )

        # 如果有下采样函数，则初始化它
        if downsample is not None:
            self.downsample = downsample(
                config=config,
                image_size=input_resolution,
                patch_size=2,
                num_channels=dim,
                embed_dim=out_dim,
                add_norm=True,
                use_conv_embed=config.use_conv_embed,
                is_stem=False,
            )
        else:
            self.downsample = None

        # 初始化指针状态为 False
        self.pointing = False

    # 前向传播方法，接收隐藏状态张量和输入尺寸元组作为参数，返回包含三个张量的元组
    def forward(self, hidden_states: torch.Tensor, input_dimensions: Tuple[int, int]) -> Tuple[torch.Tensor]:
        height, width = input_dimensions
        # 遍历所有层，逐层进行前向传播计算
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states, input_dimensions)

        # 在进行下采样之前保存当前隐藏状态
        hidden_states_before_downsampling = hidden_states
        # 如果有下采样函数，则对隐藏状态进行形状变换和下采样操作
        if self.downsample is not None:
            height, width = input_dimensions
            hidden_states = hidden_states.transpose(1, 2).reshape(
                hidden_states_before_downsampling.shape[0], -1, height, width
            )
            hidden_states, output_dimensions = self.downsample(hidden_states)
        else:
            # 如果没有下采样函数，则直接使用原始的输入尺寸作为输出尺寸
            output_dimensions = (height, width, height, width)

        # 返回阶段的输出元组，包括下采样后的隐藏状态、未下采样前的隐藏状态和输出尺寸
        stage_outputs = (hidden_states, hidden_states_before_downsampling, output_dimensions)

        return stage_outputs
    # 初始化方法，用于创建 FocalNet 对象实例
    def __init__(self, config, grid_size):
        # 调用父类的初始化方法
        super().__init__()
        # 获取深度网络层数
        self.num_stages = len(config.depths)
        # 保存配置对象
        self.config = config

        # 创建一个包含多个 FocalNetStage 实例的列表，每个实例对应一个深度网络阶段
        self.stages = nn.ModuleList(
            [
                FocalNetStage(
                    config=config,
                    index=i_layer,
                    # 设置输入分辨率，根据层次 index 和网格大小计算
                    input_resolution=(grid_size[0] // (2**i_layer), grid_size[1] // (2**i_layer)),
                )
                for i_layer in range(self.num_stages)
            ]
        )

        # 梯度检查点设为 False
        self.gradient_checkpointing = False

    # 前向传播方法，接收隐藏状态张量、输入维度、可选输出隐藏状态标志、可选输出前采样隐藏状态标志和返回字典标志
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        output_hidden_states: Optional[bool] = False,
        output_hidden_states_before_downsampling: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        ) -> Union[Tuple, FocalNetEncoderOutput]:
        # 如果需要输出隐藏状态，则初始化空元组来存储所有隐藏状态
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出隐藏状态，则初始化空元组来存储所有重塑后的隐藏状态
        all_reshaped_hidden_states = () if output_hidden_states else None

        # 如果需要输出隐藏状态，则重塑隐藏状态张量的形状
        if output_hidden_states:
            batch_size, _, hidden_size = hidden_states.shape
            # rearrange b (h w) c -> b c h w
            reshaped_hidden_state = hidden_states.view(batch_size, *input_dimensions, hidden_size)
            reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
            all_hidden_states += (hidden_states,)
            all_reshaped_hidden_states += (reshaped_hidden_state,)

        # 遍历所有阶段模块进行处理
        for i, stage_module in enumerate(self.stages):
            # 如果启用了梯度检查点且正在训练阶段，则使用梯度检查点函数来计算阶段输出
            if self.gradient_checkpointing and self.training:
                stage_outputs = self._gradient_checkpointing_func(
                    stage_module.__call__,
                    hidden_states,
                    input_dimensions,
                )
            else:
                # 否则直接调用阶段模块来计算阶段输出
                stage_outputs = stage_module(hidden_states, input_dimensions)

            # 更新隐藏状态为当前阶段的主要输出
            hidden_states = stage_outputs[0]
            # 保存当前阶段的下采样之前的隐藏状态
            hidden_states_before_downsampling = stage_outputs[1]
            # 更新输出的尺寸维度信息
            output_dimensions = stage_outputs[2]

            # 更新输入尺寸为当前输出尺寸的高度和宽度
            input_dimensions = (output_dimensions[-2], output_dimensions[-1])

            # 如果需要输出隐藏状态且输出下采样之前的隐藏状态
            if output_hidden_states and output_hidden_states_before_downsampling:
                batch_size, _, hidden_size = hidden_states_before_downsampling.shape
                # rearrange b (h w) c -> b c h w
                # 使用原始（未下采样）的高度和宽度来重塑隐藏状态张量的形状
                reshaped_hidden_state = hidden_states_before_downsampling.view(
                    batch_size, *(output_dimensions[0], output_dimensions[1]), hidden_size
                )
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states_before_downsampling,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            # 如果需要输出隐藏状态但不需要输出下采样之前的隐藏状态
            elif output_hidden_states and not output_hidden_states_before_downsampling:
                batch_size, _, hidden_size = hidden_states.shape
                # rearrange b (h w) c -> b c h w
                # 重塑隐藏状态张量的形状
                reshaped_hidden_state = hidden_states.view(batch_size, *input_dimensions, hidden_size)
                reshaped_hidden_state = reshaped_hidden_state.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states,)
                all_reshaped_hidden_states += (reshaped_hidden_state,)

        # 如果不需要以字典形式返回结果，则返回元组形式的隐藏状态和所有隐藏状态
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        # 否则以 FocalNetEncoderOutput 类的形式返回结果，包括最后的隐藏状态、所有隐藏状态和所有重塑后的隐藏状态
        return FocalNetEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            reshaped_hidden_states=all_reshaped_hidden_states,
        )
# 从transformers.models.swin.modeling_swin.SwinPreTrainedModel复制过来，并将Swin->FocalNet，swin->focalnet
class FocalNetPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 使用FocalNetConfig作为配置类
    config_class = FocalNetConfig
    # base_model_prefix指定模型前缀为"focalnet"
    base_model_prefix = "focalnet"
    # 主输入的名称为"pixel_values"
    main_input_name = "pixel_values"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        # 初始化权重：对于Linear和Conv2d层使用正态分布初始化权重，均值为0，标准差为config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 与TF版本稍有不同，TF版本使用截断正态分布进行初始化
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                # 如果存在偏置项，则将其初始化为0
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            # 如果是LayerNorm层，将偏置项初始化为0，权重初始化为1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


# FOCALNET_START_DOCSTRING是FocalNetModel的文档字符串的一部分，包含模型的基本用法和参数说明
FOCALNET_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`FocalNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# FOCALNET_INPUTS_DOCSTRING是FocalNetModel的输入参数说明文档字符串的一部分，详细描述了输入参数的类型和含义
FOCALNET_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`AutoImageProcessor.__call__`] for details.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# 使用@add_start_docstrings注解，将FocalNetModel的文档字符串合并生成
@add_start_docstrings(
    "The bare FocalNet Model outputting raw hidden-states without any specific head on top.",
    FOCALNET_START_DOCSTRING,
)
# 定义FocalNetModel类，继承自FocalNetPreTrainedModel类
class FocalNetModel(FocalNetPreTrainedModel):
    pass  # Placeholder for future model implementation
    # 初始化函数，用于初始化模型
    def __init__(self, config, add_pooling_layer=True, use_mask_token=False):
        # 调用父类的初始化函数
        super().__init__(config)
        # 将配置参数保存到对象中
        self.config = config
        # 计算深度列表的长度，确定特征数量
        self.num_stages = len(config.depths)
        self.num_features = int(config.embed_dim * 2 ** (self.num_stages - 1))

        # 创建嵌入层对象
        self.embeddings = FocalNetEmbeddings(config, use_mask_token=use_mask_token)
        # 创建编码器对象，使用嵌入层的 patch_grid 参数
        self.encoder = FocalNetEncoder(config, self.embeddings.patch_grid)

        # 创建 LayerNorm 层，设置归一化的特征数量和 epsilon 值
        self.layernorm = nn.LayerNorm(self.num_features, eps=config.layer_norm_eps)
        # 如果指定要添加池化层，则创建 AdaptiveAvgPool1d 层，用于池化操作
        self.pooler = nn.AdaptiveAvgPool1d(1) if add_pooling_layer else None

        # 初始化权重并进行最终处理
        self.post_init()

    # 获取输入嵌入的函数，返回 patch_embeddings 属性
    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    # 前向传播函数，根据输入参数进行模型的前向计算
    @add_start_docstrings_to_model_forward(FOCALNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=FocalNetModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, FocalNetModelOutput]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        # 如果未指定 output_hidden_states，则使用配置中的设定
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定 return_dict，则使用配置中的设定
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果 pixel_values 为 None，则抛出数值错误异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 调用嵌入层的前向函数，得到嵌入输出和输入维度
        embedding_output, input_dimensions = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)

        # 调用编码器的前向函数，得到编码器的输出
        encoder_outputs = self.encoder(
            embedding_output,
            input_dimensions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取序列输出并进行 LayerNorm 处理
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        # 初始化池化输出为 None
        pooled_output = None
        # 如果池化层不为 None，则进行池化操作和扁平化处理
        if self.pooler is not None:
            pooled_output = self.pooler(sequence_output.transpose(1, 2))
            pooled_output = torch.flatten(pooled_output, 1)

        # 如果不返回字典，则返回元组形式的输出
        if not return_dict:
            output = (sequence_output, pooled_output) + encoder_outputs[1:]
            return output

        # 如果返回字典，则返回 FocalNetModelOutput 类型的对象
        return FocalNetModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            reshaped_hidden_states=encoder_outputs.reshaped_hidden_states,
        )
@add_start_docstrings(
    """
    FocalNet Model with a decoder on top for masked image modeling.

    This follows the same implementation as in [SimMIM](https://arxiv.org/abs/2111.09886).

    <Tip>

    Note that we provide a script to pre-train this model on custom data in our [examples
    directory](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining).

    </Tip>
    """,
    FOCALNET_START_DOCSTRING,
)
class FocalNetForMaskedImageModeling(FocalNetPreTrainedModel):
    """
    FocalNet Model for masked image modeling, extending FocalNetPreTrainedModel.

    Inherits from FocalNetPreTrainedModel and implements a model architecture with a decoder.
    """

    def __init__(self, config):
        """
        Initializes the FocalNetForMaskedImageModeling.

        Args:
            config: FocalNet configuration class instance.
        """
        super().__init__(config)

        # Initialize FocalNet model with specified configuration
        self.focalnet = FocalNetModel(config, add_pooling_layer=False, use_mask_token=True)

        # Calculate number of stages and features for the decoder
        self.num_stages = len(config.depths)
        num_features = int(config.embed_dim * 2 ** (self.num_stages - 1))

        # Define decoder architecture using convolution and pixel shuffle
        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=num_features, out_channels=config.encoder_stride**2 * config.num_channels, kernel_size=1
            ),
            nn.PixelShuffle(config.encoder_stride),
        )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(FOCALNET_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=FocalNetMaskedImageModelingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,



@add_start_docstrings(
    """
    FocalNet Model with an image classification head on top (a linear layer on top of the pooled output) e.g. for
    ImageNet.
    """,
    FOCALNET_START_DOCSTRING,
)
class FocalNetForImageClassification(FocalNetPreTrainedModel):
    """
    FocalNet Model for image classification tasks, extending FocalNetPreTrainedModel.

    Inherits from FocalNetPreTrainedModel and implements a model architecture with an image classification head.
    """

    # Copied from transformers.models.swin.modeling_swin.SwinForImageClassification.__init__ with Swin->FocalNet, swin->focalnet
    def __init__(self, config):
        """
        Initializes the FocalNetForImageClassification.

        Args:
            config: FocalNet configuration class instance.
        """
        super().__init__(config)

        self.num_labels = config.num_labels
        self.focalnet = FocalNetModel(config)

        # Define classifier head
        self.classifier = (
            nn.Linear(self.focalnet.num_features, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(FOCALNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=FocalNetImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[Tuple, FocalNetImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 如果 return_dict 为 None，则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 FocalNet 模型进行前向传播
        outputs = self.focalnet(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取池化后的输出，通常是经过全局平均池化的结果
        pooled_output = outputs[1]

        # 对池化后的输出进行分类器的前向传播，得到分类器的输出 logits
        logits = self.classifier(pooled_output)

        # 初始化损失为 None
        loss = None
        # 如果 labels 不为 None，则计算损失
        if labels is not None:
            # 如果问题类型未定义，则根据条件自动定义问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型选择相应的损失函数
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()  # 使用均方误差损失函数
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()  # 使用交叉熵损失函数
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()  # 使用带 logits 的二元交叉熵损失函数
                loss = loss_fct(logits, labels)

        # 如果 return_dict 为 False，则返回损失和模型输出，否则返回自定义输出对象 FocalNetImageClassifierOutput
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return FocalNetImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            reshaped_hidden_states=outputs.reshaped_hidden_states,
        )
@add_start_docstrings(
    """
    FocalNet backbone, to be used with frameworks like X-Decoder.
    """,
    FOCALNET_START_DOCSTRING,
)
class FocalNetBackbone(FocalNetPreTrainedModel, BackboneMixin):
    def __init__(self, config: FocalNetConfig):
        super().__init__(config)
        super()._init_backbone(config)

        # 设置特征的维度列表，包括嵌入维度和隐藏层尺寸
        self.num_features = [config.embed_dim] + config.hidden_sizes
        # 创建 FocalNet 模型对象
        self.focalnet = FocalNetModel(config)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(FOCALNET_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BackboneOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BackboneOutput:
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = AutoImageProcessor.from_pretrained("microsoft/focalnet-tiny-lrf")
        >>> model = AutoBackbone.from_pretrained("microsoft/focalnet-tiny-lrf")

        >>> inputs = processor(image, return_tensors="pt")
        >>> outputs = model(**inputs)
        ```"""
        # 如果 return_dict 为 None，则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果 output_hidden_states 为 None，则使用配置中的默认设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # 调用 FocalNet 模型进行前向传播
        outputs = self.focalnet(pixel_values, output_hidden_states=True, return_dict=True)

        # 获取重塑后的隐藏状态
        hidden_states = outputs.reshaped_hidden_states

        feature_maps = ()
        # 遍历阶段名称和输出特征名称，获取特征映射
        for idx, stage in enumerate(self.stage_names):
            if stage in self.out_features:
                feature_maps += (hidden_states[idx],)

        # 如果不要求返回字典，则返回一个元组
        if not return_dict:
            output = (feature_maps,)
            if output_hidden_states:
                output += (outputs.hidden_states,)
            return output

        # 返回 BackboneOutput 对象，包含特征映射、隐藏状态（如果需要）、注意力（暂时为 None）
        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=None,
        )
```