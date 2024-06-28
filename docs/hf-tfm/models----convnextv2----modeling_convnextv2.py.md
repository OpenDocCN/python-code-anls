# `.\models\convnextv2\modeling_convnextv2.py`

```py
# coding=utf-8
# Copyright 2023 Meta Platforms, Inc. and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch ConvNextV2 model."""


from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BackboneOutput,
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ...utils.backbone_utils import BackboneMixin
from .configuration_convnextv2 import ConvNextV2Config


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "ConvNextV2Config"

# Base docstring
_CHECKPOINT_FOR_DOC = "facebook/convnextv2-tiny-1k-224"
_EXPECTED_OUTPUT_SHAPE = [1, 768, 7, 7]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "facebook/convnextv2-tiny-1k-224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

CONVNEXTV2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/convnextv2-tiny-1k-224",
    # See all ConvNextV2 models at https://huggingface.co/models?filter=convnextv2
]


# Copied from transformers.models.beit.modeling_beit.drop_path
# 定义函数 drop_path，实现随机深度（Stochastic Depth）机制
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    # 如果 drop_prob 为 0 或者不处于训练模式，则直接返回输入
    if drop_prob == 0.0 or not training:
        return input
    # 计算保留概率
    keep_prob = 1 - drop_prob
    # 创建与输入张量相同形状的随机张量，用于随机深度的保留路径
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # 对随机生成的张量进行向下取整操作，用于二值化处理
    output = input.div(keep_prob) * random_tensor  # 对输入张量进行按元素除法操作，并乘以随机生成的张量，用于Dropout处理
    return output  # 返回处理后的张量作为输出
# 从 transformers.models.beit.modeling_beit.BeitDropPath 复制过来的代码，将 Beit 替换为 ConvNextV2
class ConvNextV2DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 调用 drop_path 函数，对隐藏状态进行随机深度路径丢弃
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


# 从 transformers.models.convnext.modeling_convnext.ConvNextLayerNorm 复制过来的代码，将 ConvNext 替换为 ConvNextV2
class ConvNextV2LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height,
    width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data format: {self.data_format}")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            # 对 channels_last 格式的输入进行 layer_norm
            x = torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # 对 channels_first 格式的输入进行 layer_norm
            input_dtype = x.dtype
            x = x.float()
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = x.to(dtype=input_dtype)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


# 从 transformers.models.convnext.modeling_convnext.ConvNextEmbeddings 复制过来的代码，将 ConvNext 替换为 ConvNextV2
class ConvNextV2Embeddings(nn.Module):
    """This class is comparable to (and inspired by) the SwinEmbeddings class
    found in src/transformers/models/swin/modeling_swin.py.
    """

    def __init__(self, config):
        super().__init__()
        # Patch embedding layer using 2D convolution
        self.patch_embeddings = nn.Conv2d(
            config.num_channels, config.hidden_sizes[0], kernel_size=config.patch_size, stride=config.patch_size
        )
        # Layer normalization specific to ConvNeXTV2 embeddings
        self.layernorm = ConvNextV2LayerNorm(config.hidden_sizes[0], eps=1e-6, data_format="channels_first")
        self.num_channels = config.num_channels

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        num_channels = pixel_values.shape[1]
        # Check if input pixel values have the expected number of channels
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # Compute patch embeddings using the defined convolutional layer
        embeddings = self.patch_embeddings(pixel_values)
        # Apply layer normalization to the embeddings
        embeddings = self.layernorm(embeddings)
        return embeddings


class ConvNextV2Layer(nn.Module):
    """This corresponds to the `Block` class in the original implementation.

    There are two equivalent implementations: [DwConv, LayerNorm (channels_first), Conv, GELU,1x1 Conv]; all in (N, C,
    H, W) (2) [DwConv, Permute to (N, H, W, C), LayerNorm (channels_last), Linear, GELU, Linear]; Permute back

    The authors used (2) as they find it slightly faster in PyTorch.

    Args:
        config ([`ConvNextV2Config`]): Model configuration class.
        dim (`int`): Number of input channels.
        drop_path (`float`): Stochastic depth rate. Default: 0.0.
    """

    def __init__(self, config, dim, drop_path=0):
        super().__init__()
        # Depthwise convolutional layer
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        # Layer normalization specific to ConvNeXTV2 layers
        self.layernorm = ConvNextV2LayerNorm(dim, eps=1e-6)
        # Pointwise (1x1) convolutional layers implemented as linear transformations
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        # Activation function chosen from the configuration
        self.act = ACT2FN[config.hidden_act]
        # Gated residual network (GRN) layer
        self.grn = ConvNextV2GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # Drop path regularization if specified
        self.drop_path = ConvNextV2DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        input = hidden_states
        # Apply depthwise convolution
        x = self.dwconv(hidden_states)
        # Permute dimensions for compatibility with subsequent operations
        # (batch_size, num_channels, height, width) -> (batch_size, height, width, num_channels)
        x = x.permute(0, 2, 3, 1)
        # Apply layer normalization
        x = self.layernorm(x)
        # Apply first pointwise convolution followed by activation
        x = self.pwconv1(x)
        x = self.act(x)
        # Apply gated residual network (GRN) layer
        x = self.grn(x)
        # Apply second pointwise convolution
        x = self.pwconv2(x)
        # Permute dimensions back to the original form
        # (batch_size, height, width, num_channels) -> (batch_size, num_channels, height, width)
        x = x.permute(0, 3, 1, 2)

        # Add the input tensor and the output of the drop path layer
        x = input + self.drop_path(x)
        return x


# Copied from transformers.models.convnext.modeling_convnext.ConvNextStage with ConvNeXT->ConvNeXTV2, ConvNext->ConvNextV2
class ConvNextV2Stage(nn.Module):
    """Represents a stage in the ConvNeXTV2 model."""
    """ConvNeXTV2 stage, consisting of an optional downsampling layer + multiple residual blocks.

    Args:
        config ([`ConvNextV2Config`]): Model configuration class.
        in_channels (`int`): Number of input channels.
        out_channels (`int`): Number of output channels.
        depth (`int`): Number of residual blocks.
        drop_path_rates(`List[float]`): Stochastic depth rates for each layer.
    """
    
    # 定义 ConvNeXTV2 阶段的网络模块，包括可选的下采样层和多个残差块

    def __init__(self, config, in_channels, out_channels, kernel_size=2, stride=2, depth=2, drop_path_rates=None):
        super().__init__()  # 调用父类的初始化方法

        # 如果输入通道数与输出通道数不同或者步长大于1，则创建一个下采样层的序列
        if in_channels != out_channels or stride > 1:
            self.downsampling_layer = nn.Sequential(
                ConvNextV2LayerNorm(in_channels, eps=1e-6, data_format="channels_first"),  # 添加通道规范化层
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),  # 添加卷积层
            )
        else:
            self.downsampling_layer = nn.Identity()  # 否则使用恒等映射作为下采样层

        # 根据传入的深度参数，创建一个包含多个 ConvNextV2Layer 的序列
        drop_path_rates = drop_path_rates or [0.0] * depth  # 如果未提供 drop_path_rates，则初始化为0
        self.layers = nn.Sequential(
            *[ConvNextV2Layer(config, dim=out_channels, drop_path=drop_path_rates[j]) for j in range(depth)]
        )

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        hidden_states = self.downsampling_layer(hidden_states)  # 应用下采样层
        hidden_states = self.layers(hidden_states)  # 应用多个 ConvNextV2Layer 层
        return hidden_states
# 从 transformers.models.convnext.modeling_convnext.ConvNextEncoder 复制并修改为 ConvNextV2
class ConvNextV2Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化各阶段的神经网络模块列表
        self.stages = nn.ModuleList()
        # 根据深度和dropout率生成一个列表，用于每个阶段的路径丢弃率
        drop_path_rates = [
            x.tolist() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths)).split(config.depths)
        ]
        prev_chs = config.hidden_sizes[0]
        # 遍历每个阶段的配置
        for i in range(config.num_stages):
            out_chs = config.hidden_sizes[i]
            # 创建 ConvNextV2Stage 实例作为每个阶段的神经网络模块
            stage = ConvNextV2Stage(
                config,
                in_channels=prev_chs,
                out_channels=out_chs,
                stride=2 if i > 0 else 1,
                depth=config.depths[i],
                drop_path_rates=drop_path_rates[i],
            )
            self.stages.append(stage)
            prev_chs = out_chs

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutputWithNoAttention]:
        # 如果需要输出所有隐藏状态，则初始化一个空元组
        all_hidden_states = () if output_hidden_states else None

        # 遍历每个阶段的神经网络模块
        for i, layer_module in enumerate(self.stages):
            if output_hidden_states:
                # 如果需要输出所有隐藏状态，则将当前隐藏状态添加到 all_hidden_states 元组中
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 将当前隐藏状态传递给当前阶段的神经网络模块，更新隐藏状态
            hidden_states = layer_module(hidden_states)

        # 如果需要输出所有隐藏状态，则将最终隐藏状态添加到 all_hidden_states 元组中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要返回字典，则根据情况返回隐藏状态和所有隐藏状态元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        # 返回一个 BaseModelOutputWithNoAttention 对象，包含最终隐藏状态和所有隐藏状态元组
        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


# 从 transformers.models.convnext.modeling_convnext.ConvNextPreTrainedModel 复制并修改为 ConvNextV2
class ConvNextV2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 与 ConvNextV2 相关的配置类
    config_class = ConvNextV2Config
    # Base model 的前缀名称
    base_model_prefix = "convnextv2"
    # 主要输入的名称
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果是线性层或二维卷积层，初始化权重
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 使用正态分布初始化权重，均值为 0，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                # 如果存在偏置项，则将其初始化为零
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            # 如果是 LayerNorm 层，将偏置项初始化为零，权重初始化为 1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


# 开始的文档字符串，说明这是一个 PyTorch 的 nn.Module 子类
CONVNEXTV2_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    # 将其作为常规的 PyTorch 模块使用，并参考 PyTorch 文档处理所有与一般使用和行为相关的问题。

    Parameters:
        config ([`ConvNextV2Config`]): 包含模型所有参数的模型配置类。
            使用配置文件初始化模型时，不会加载与模型关联的权重，仅加载配置信息。
            可查看 [`~PreTrainedModel.from_pretrained`] 方法来加载模型权重。
"""

CONVNEXTV2_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`ConvNextImageProcessor`]. See
            [`ConvNextImageProcessor.__call__`] for details.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# 为 ConvNextV2Model 添加文档注释，描述其作为 ConvNextV2 模型的基础输出模型，不带特定的顶层头部。
# 同时继承了 CONVNEXTV2_START_DOCSTRING 中的描述。
@add_start_docstrings(
    "The bare ConvNextV2 model outputting raw features without any specific head on top.",
    CONVNEXTV2_START_DOCSTRING,
)
# 从 transformers.models.convnext.modeling_convnext.ConvNextModel 复制代码，替换为 ConvNextV2Model，CONVNEXT->CONVNEXTV2
class ConvNextV2Model(ConvNextV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # 初始化 ConvNextV2Embeddings，用于处理输入特征
        self.embeddings = ConvNextV2Embeddings(config)
        # 初始化 ConvNextV2Encoder，用于处理嵌入特征
        self.encoder = ConvNextV2Encoder(config)

        # 最终的 layernorm 层，用于标准化最后隐藏层的特征
        self.layernorm = nn.LayerNorm(config.hidden_sizes[-1], eps=config.layer_norm_eps)

        # 初始化权重并应用最终处理
        self.post_init()

    # 添加 CONVNEXTV2_INPUTS_DOCSTRING 作为 forward 方法的文档注释
    @add_start_docstrings_to_model_forward(CONVNEXTV2_INPUTS_DOCSTRING)
    # 添加代码示例的文档注释，包括 _CHECKPOINT_FOR_DOC、BaseModelOutputWithPoolingAndNoAttention 等信息
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    # 前向传播函数，接受像素值 pixel_values 作为输入，返回隐藏状态的 BaseModelOutputWithPoolingAndNoAttention
    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPoolingAndNoAttention]:
        # 如果未指定 output_hidden_states，则使用 self.config.output_hidden_states
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定 return_dict，则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果 pixel_values 为 None，则抛出 ValueError
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 将 pixel_values 传递给 embeddings，得到嵌入特征输出
        embedding_output = self.embeddings(pixel_values)

        # 将嵌入特征输出传递给 encoder，得到编码器的输出
        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 取编码器输出中的最后隐藏状态
        last_hidden_state = encoder_outputs[0]

        # 全局平均池化，将 (N, C, H, W) 的张量池化为 (N, C)
        pooled_output = self.layernorm(last_hidden_state.mean([-2, -1]))

        # 如果 return_dict 为 False，则返回元组形式的输出
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 否则，返回带有池化器输出的 BaseModelOutputWithPoolingAndNoAttention
        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )
    """
    ConvNextV2 Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """
    # 使用ConvNextV2模型进行图像分类，顶部有一个分类头部（线性层在池化特征之上），例如用于ImageNet数据集
    CONVNEXTV2_START_DOCSTRING,

class ConvNextV2ForImageClassification(ConvNextV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化模型参数
        self.num_labels = config.num_labels
        self.convnextv2 = ConvNextV2Model(config)

        # 分类器头部
        self.classifier = (
            nn.Linear(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(CONVNEXTV2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    # 前向传播函数，接受像素值、标签等参数，并返回模型输出
    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 如果 return_dict 不为 None，则使用它；否则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 convnextv2 方法处理像素值，根据 return_dict 参数返回结果
        outputs = self.convnextv2(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        # 如果 return_dict 为 True，则从 outputs 中获取 pooler_output；否则从 outputs 的第二个元素获取
        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        # 使用分类器对 pooled_output 进行分类得到 logits
        logits = self.classifier(pooled_output)

        # 初始化损失为 None
        loss = None
        # 如果 labels 不为 None，则计算损失
        if labels is not None:
            # 如果 self.config.problem_type 为 None，则根据 num_labels 确定 problem_type
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据 problem_type 计算相应的损失函数
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果 return_dict 为 False，则返回 logits 与 outputs 的其余部分作为输出
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则返回 ImageClassifierOutputWithNoAttention 对象
        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )
@add_start_docstrings(
    """
    ConvNeXT V2 backbone, to be used with frameworks like DETR and MaskFormer.
    """,
    CONVNEXTV2_START_DOCSTRING,
)
# 基于ConvNeXT V2的主干网络，用于与DETR和MaskFormer等框架配合使用
# 从transformers.models.convnext.modeling_convnext.ConvNextBackbone复制而来，修改了名称和配置
class ConvNextV2Backbone(ConvNextV2PreTrainedModel, BackboneMixin):
    def __init__(self, config):
        super().__init__(config)
        # 调用父类初始化函数
        super()._init_backbone(config)

        # 初始化嵌入层和编码器
        self.embeddings = ConvNextV2Embeddings(config)
        self.encoder = ConvNextV2Encoder(config)
        self.num_features = [config.hidden_sizes[0]] + config.hidden_sizes

        # 为输出特征的隐藏状态添加层归一化
        hidden_states_norms = {}
        for stage, num_channels in zip(self._out_features, self.channels):
            hidden_states_norms[stage] = ConvNextV2LayerNorm(num_channels, data_format="channels_first")
        self.hidden_states_norms = nn.ModuleDict(hidden_states_norms)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(CONVNEXTV2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BackboneOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # ConvNeXT V2模型的前向传播函数，接受像素值张量和一些可选的返回参数
        # 返回值类型为BackboneOutput，具体配置类为_CONFIG_FOR_DOC
        ) -> BackboneOutput:
        """
        返回：模型输出的BackboneOutput对象。

        Examples: 示例代码展示了如何使用该函数来处理图像和调用模型。

        ```
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = AutoImageProcessor.from_pretrained("facebook/convnextv2-tiny-1k-224")
        >>> model = AutoBackbone.from_pretrained("facebook/convnextv2-tiny-1k-224")

        >>> inputs = processor(image, return_tensors="pt")
        >>> outputs = model(**inputs)
        ```"""

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 确定是否使用返回字典的配置，默认使用模型配置中的设定

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 确定是否输出隐藏状态的配置，默认使用模型配置中的设定

        embedding_output = self.embeddings(pixel_values)
        # 将输入的像素值嵌入到模型的嵌入层中得到嵌入输出

        outputs = self.encoder(
            embedding_output,
            output_hidden_states=True,  # 强制输出隐藏状态
            return_dict=return_dict,    # 按需返回字典或元组
        )
        # 使用编码器对嵌入输出进行编码，并根据配置决定返回字典或元组

        hidden_states = outputs.hidden_states if return_dict else outputs[1]
        # 根据返回字典的配置选择输出的隐藏状态

        feature_maps = ()
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                hidden_state = self.hidden_states_norms[stage](hidden_state)
                # 对特定阶段的隐藏状态进行归一化处理
                feature_maps += (hidden_state,)

        if not return_dict:
            output = (feature_maps,)
            if output_hidden_states:
                output += (hidden_states,)
            return output
        # 如果不要求返回字典，则返回一个包含特征图和隐藏状态的元组

        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=hidden_states if output_hidden_states else None,
            attentions=None,
        )
        # 否则，返回一个BackboneOutput对象，包含特征图、隐藏状态和注意力信息（注意力默认为None）
```