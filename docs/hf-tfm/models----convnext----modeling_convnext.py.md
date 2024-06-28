# `.\models\convnext\modeling_convnext.py`

```py
# coding=utf-8
# Copyright 2022 Meta Platforms, Inc. and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch ConvNext model."""

# Import necessary modules and functions from PyTorch and Transformers
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# Import various components from HuggingFace Transformers library
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
from .configuration_convnext import ConvNextConfig

# Get logger instance for logging messages
logger = logging.get_logger(__name__)

# General docstring for configuration
_CONFIG_FOR_DOC = "ConvNextConfig"

# Base docstring for checkpoint
_CHECKPOINT_FOR_DOC = "facebook/convnext-tiny-224"
_EXPECTED_OUTPUT_SHAPE = [1, 768, 7, 7]

# Image classification checkpoint and expected output
_IMAGE_CLASS_CHECKPOINT = "facebook/convnext-tiny-224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

# List of pretrained model archives for ConvNext
CONVNEXT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/convnext-tiny-224",
    # See all ConvNext models at https://huggingface.co/models?filter=convnext
]

# Function definition for drop path, a form of stochastic depth regularization
# Copied from transformers.models.beit.modeling_beit.drop_path
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    # 根据输入的张量 input 和保留概率 keep_prob 计算 dropout 后的输出张量
    output = input.div(keep_prob) * random_tensor
    # 返回 dropout 后的输出张量
    return output
# 从 transformers.models.beit.modeling_beit.BeitDropPath 复制的代码，将 Beit 替换为 ConvNext
class ConvNextDropPath(nn.Module):
    """每个样本应用于残差块主路径中的丢弃路径（随机深度）。"""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class ConvNextLayerNorm(nn.Module):
    r"""支持两种数据格式的 LayerNorm：channels_last（默认）或 channels_first。
    输入数据维度的顺序。channels_last 对应形状为 (batch_size, height, width, channels) 的输入，
    而 channels_first 对应形状为 (batch_size, channels, height, width) 的输入。
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
            x = torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            input_dtype = x.dtype
            x = x.float()
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = x.to(dtype=input_dtype)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConvNextEmbeddings(nn.Module):
    """这个类类似于（并且受到启发于）src/transformers/models/swin/modeling_swin.py 中的 SwinEmbeddings 类。"""

    def __init__(self, config):
        super().__init__()
        self.patch_embeddings = nn.Conv2d(
            config.num_channels, config.hidden_sizes[0], kernel_size=config.patch_size, stride=config.patch_size
        )
        self.layernorm = ConvNextLayerNorm(config.hidden_sizes[0], eps=1e-6, data_format="channels_first")
        self.num_channels = config.num_channels
    # 定义前向传播函数，接收像素值作为输入，并返回处理后的张量
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        # 获取输入张量的通道数
        num_channels = pixel_values.shape[1]
        # 检查输入张量的通道数是否与模型配置中的通道数一致，若不一致则抛出数值错误异常
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 使用预定义的函数处理输入像素值，生成嵌入表示
        embeddings = self.patch_embeddings(pixel_values)
        # 对生成的嵌入表示进行层归一化处理
        embeddings = self.layernorm(embeddings)
        # 返回处理后的嵌入表示张量
        return embeddings
# 定义 ConvNeXT 阶段，包含可选的下采样层和多个残差块
class ConvNextStage(nn.Module):
    """ConvNeXT stage, consisting of an optional downsampling layer + multiple residual blocks.

    Args:
        config ([`ConvNextConfig`]): Model configuration class.
        in_channels (`int`): Number of input channels.
        out_channels (`int`): Number of output channels.
        depth (`int`): Number of residual blocks.
        drop_path_rates(`List[float]`): Stochastic depth rates for each layer.
    """


在这段代码中，我们定义了一个名为 `ConvNextStage` 的类，用于表示 ConvNeXT 模型的一个阶段。这个阶段包括一个可选的下采样层和多个残差块，这些块将按照指定的参数配置进行堆叠。
    # 初始化函数，用于构建一个自定义的卷积神经网络模块
    def __init__(self, config, in_channels, out_channels, kernel_size=2, stride=2, depth=2, drop_path_rates=None):
        # 调用父类的初始化方法
        super().__init__()

        # 如果输入通道数不等于输出通道数或者步长大于1，则创建一个下采样层
        if in_channels != out_channels or stride > 1:
            self.downsampling_layer = nn.Sequential(
                # 使用自定义的 ConvNextLayerNorm 类，对输入进行归一化处理
                ConvNextLayerNorm(in_channels, eps=1e-6, data_format="channels_first"),
                # 添加一个卷积层，用于下采样
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
            )
        else:
            # 如果输入通道数等于输出通道数且步长为1，则使用一个恒等映射层
            self.downsampling_layer = nn.Identity()
        
        # 如果未提供 drop_path_rates 参数，则初始化为一个与深度 depth 相同长度的全零列表
        drop_path_rates = drop_path_rates or [0.0] * depth
        
        # 创建深度为 depth 的卷积层序列
        self.layers = nn.Sequential(
            *[ConvNextLayer(config, dim=out_channels, drop_path=drop_path_rates[j]) for j in range(depth)]
        )

    # 前向传播函数，用于定义模型的数据流向
    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        # 对输入的 hidden_states 进行下采样处理
        hidden_states = self.downsampling_layer(hidden_states)
        # 将下采样后的结果通过多层卷积处理
        hidden_states = self.layers(hidden_states)
        # 返回处理后的结果张量
        return hidden_states
class ConvNextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个空的模块列表用于存放各个阶段的 ConvNextStage
        self.stages = nn.ModuleList()
        # 计算每个阶段的 drop path rates
        drop_path_rates = [
            x.tolist() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths)).split(config.depths)
        ]
        # 初始化前一阶段的输出通道数为输入的隐藏大小的第一个元素
        prev_chs = config.hidden_sizes[0]
        # 遍历创建每个阶段的 ConvNextStage
        for i in range(config.num_stages):
            out_chs = config.hidden_sizes[i]
            stage = ConvNextStage(
                config,
                in_channels=prev_chs,
                out_channels=out_chs,
                stride=2 if i > 0 else 1,
                depth=config.depths[i],
                drop_path_rates=drop_path_rates[i],
            )
            # 将创建的阶段添加到模块列表中
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

        # 遍历每个阶段的模块，并对隐藏状态进行前向传播
        for i, layer_module in enumerate(self.stages):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states = layer_module(hidden_states)

        # 如果需要输出所有隐藏状态，则添加最后一个阶段的隐藏状态到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典形式的输出，则根据需要返回隐藏状态和所有隐藏状态的元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        # 返回带有无注意力的基本模型输出
        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


class ConvNextPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ConvNextConfig
    base_model_prefix = "convnext"
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        """Initialize the weights"""
        # 初始化线性层和卷积层的权重，使用正态分布初始化，偏置初始化为零
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 与 TF 版本略有不同，使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        # 初始化层归一化层的权重，偏置初始化为零，权重初始化为 1.0
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


CONVNEXT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.
"""
    Parameters:
        config ([`ConvNextConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
@add_start_docstrings(
    """
    ConvNext 模型输出裸特征，没有特定头部添加。
    """,
    CONVNEXT_START_DOCSTRING,
)


这段代码定义了一个类 `ConvNextModel`，继承自 `ConvNextPreTrainedModel`，用于构建 ConvNext 模型。在类的初始化函数中，首先调用父类的初始化方法，然后设置了模型的配置信息和各个模块的初始化，包括 `ConvNextEmbeddings` 和 `ConvNextEncoder`。此外，还初始化了一个 `LayerNorm` 层用于最终的归一化处理。

``````
        self.embeddings = ConvNextEmbeddings(config)


这行代码初始化了 `ConvNextEmbeddings` 类的实例 `self.embeddings`，并传入了模型配置 `config`。


        self.encoder = ConvNextEncoder(config)


这行代码初始化了 `ConvNextEncoder` 类的实例 `self.encoder`，并同样传入了模型配置 `config`。


        self.layernorm = nn.LayerNorm(config.hidden_sizes[-1], eps=config.layer_norm_eps)


这行代码初始化了 `LayerNorm` 层 `self.layernorm`，其中 `config.hidden_sizes[-1]` 表示配置中定义的隐藏层的最后一个尺寸，`eps=config.layer_norm_eps` 则是配置中定义的层归一化的 epsilon 参数。
    # ConvNext模型，其顶部有一个图像分类头部（在池化特征之上的线性层），例如用于ImageNet。
    """,
    # 使用CONVNEXT_START_DOCSTRING的值作为注释的起始点
    CONVNEXT_START_DOCSTRING,
    )
    # Image classification model inheriting from a pretrained ConvNext model
    class ConvNextForImageClassification(ConvNextPreTrainedModel):
        def __init__(self, config):
            super().__init__(config)

            # Number of labels for classification
            self.num_labels = config.num_labels
            # Instantiate ConvNext model
            self.convnext = ConvNextModel(config)

            # Classifier head: either a linear layer or an identity function based on number of labels
            self.classifier = (
                nn.Linear(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity()
            )

            # Initialize weights and perform final setup
            self.post_init()

        @add_start_docstrings_to_model_forward(CONVNEXT_INPUTS_DOCSTRING)
        @add_code_sample_docstrings(
            checkpoint=_IMAGE_CLASS_CHECKPOINT,
            output_type=ImageClassifierOutputWithNoAttention,
            config_class=_CONFIG_FOR_DOC,
            expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
        )
        # Forward method for the model
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
        # 根据需要决定是否使用预定义的返回字典或者自定义的返回字典配置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用卷积神经网络模型进行前向传播
        outputs = self.convnext(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        # 根据返回字典的设置选择使用池化后的输出或者直接从输出列表中获取结果
        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        # 使用分类器模型计算输出logits
        logits = self.classifier(pooled_output)

        # 初始化损失值为None
        loss = None
        # 如果提供了标签，则计算损失
        if labels is not None:
            # 如果问题类型未定义，则根据标签类型和标签数量设置问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型选择相应的损失函数
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    # 对于回归问题，使用均方误差损失函数
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # 对于单标签分类问题，使用交叉熵损失函数
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # 对于多标签分类问题，使用带logits的二元交叉熵损失函数
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果不使用返回字典，则按照原始模型的输出方式返回
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果使用返回字典，则构建特定输出格式的对象返回
        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )
"""
ConvNeXt backbone, to be used with frameworks like DETR and MaskFormer.
"""

# 定义 ConvNeXtBackbone 类，用于与 DETR 和 MaskFormer 等框架一起使用的卷积神经网络骨干
class ConvNextBackbone(ConvNextPreTrainedModel, BackboneMixin):
    
    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 调用父类 ConvNextPreTrainedModel 的 _init_backbone 方法
        super()._init_backbone(config)

        # 初始化嵌入层和编码器
        self.embeddings = ConvNextEmbeddings(config)
        self.encoder = ConvNextEncoder(config)
        self.num_features = [config.hidden_sizes[0]] + config.hidden_sizes

        # 为输出特征的隐藏状态添加层归一化
        hidden_states_norms = {}
        # 遍历输出特征和通道数，为每个输出特征添加 ConvNextLayerNorm 层归一化
        for stage, num_channels in zip(self._out_features, self.channels):
            hidden_states_norms[stage] = ConvNextLayerNorm(num_channels, data_format="channels_first")
        self.hidden_states_norms = nn.ModuleDict(hidden_states_norms)

        # 初始化权重并应用最终处理
        self.post_init()

    # forward 方法，定义模型的前向传播过程
    @add_start_docstrings_to_model_forward(CONVNEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BackboneOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> BackboneOutput:
        """
        返回 BackBoneOutput 对象。

        Examples:

        ```
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = AutoImageProcessor.from_pretrained("facebook/convnext-tiny-224")
        >>> model = AutoBackbone.from_pretrained("facebook/convnext-tiny-224")

        >>> inputs = processor(image, return_tensors="pt")
        >>> outputs = model(**inputs)
        ```"""
        
        # 设置返回字典的默认值为 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 设置输出隐藏状态的默认值为 self.config.output_hidden_states
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        
        # 使用 self.embeddings 对象生成嵌入输出
        embedding_output = self.embeddings(pixel_values)

        # 将嵌入输出传入编码器，并设置输出隐藏状态和返回字典的选项
        outputs = self.encoder(
            embedding_output,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        # 根据是否使用返回字典来选择输出的隐藏状态
        hidden_states = outputs.hidden_states if return_dict else outputs[1]

        # 初始化特征图为空元组
        feature_maps = ()
        # 遍历阶段名称和对应的隐藏状态，如果阶段在输出特征中，则归一化隐藏状态并添加到特征图中
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                hidden_state = self.hidden_states_norms[stage](hidden_state)
                feature_maps += (hidden_state,)

        # 如果不使用返回字典，则将特征图和可能的隐藏状态作为元组输出
        if not return_dict:
            output = (feature_maps,)
            if output_hidden_states:
                output += (hidden_states,)
            return output

        # 使用 BackboneOutput 类返回特征图、隐藏状态（如果有）、注意力（未提供）
        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=hidden_states if output_hidden_states else None,
            attentions=None,
        )
```