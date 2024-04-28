# `.\transformers\models\resnet\modeling_resnet.py`

```py
# 设置编码格式为 UTF-8

# 导入必要的库和模块
# 包括 torch 库、torch 的一些子模块、自定义的模块以及一些函数
from typing import Optional

import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入自定义的模块和函数
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
from .configuration_resnet import ResNetConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 通用文档字符串
_CONFIG_FOR_DOC = "ResNetConfig"

# 基本文档字符串
_CHECKPOINT_FOR_DOC = "microsoft/resnet-50"
_EXPECTED_OUTPUT_SHAPE = [1, 2048, 7, 7]

# 图像分类文档字符串
_IMAGE_CLASS_CHECKPOINT = "microsoft/resnet-50"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tiger cat"

# 预训练的 ResNet 模型存档列表
RESNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/resnet-50",
    # 查看所有 ResNet 模型列表：https://huggingface.co/models?filter=resnet
]


# ResNet 的卷积层定义
class ResNetConvLayer(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, activation: str = "relu"
    ):
        super().__init__()
        # 定义卷积层
        self.convolution = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=False
        )
        # 定义归一化层
        self.normalization = nn.BatchNorm2d(out_channels)
        # 定义激活函数
        self.activation = ACT2FN[activation] if activation is not None else nn.Identity()

    def forward(self, input: Tensor) -> Tensor:
        # 前向传播函数
        # 进行卷积操作
        hidden_state = self.convolution(input)
        # 进行归一化操作
        hidden_state = self.normalization(hidden_state)
        # 进行激活函数操作
        hidden_state = self.activation(hidden_state)
        return hidden_state


# ResNet 的嵌入层定义
class ResNetEmbeddings(nn.Module):
    """
    ResNet 嵌入层（干部）由单个具有侵略性的卷积组成。
    """
    # 初始化函数，接受一个ResNetConfig对象作为参数
    def __init__(self, config: ResNetConfig):
        # 调用父类的初始化函数
        super().__init__()
        # 创建ResNetConvLayer对象作为embedder，设置通道数、嵌入维度、卷积核大小、步长和激活函数
        self.embedder = ResNetConvLayer(
            config.num_channels, config.embedding_size, kernel_size=7, stride=2, activation=config.hidden_act
        )
        # 创建最大池化层对象，设置卷积核大小、步长和填充
        self.pooler = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 记录通道数
        self.num_channels = config.num_channels

    # 前向传播函数，接受像素值Tensor作为输入，返回嵌入值Tensor
    def forward(self, pixel_values: Tensor) -> Tensor:
        # 获取输入像素值Tensor的通道数
        num_channels = pixel_values.shape[1]
        # 如果通道数不匹配设置的通道数，则抛出数值错误
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 对输入像素值进行嵌入
        embedding = self.embedder(pixel_values)
        # 对嵌入值进行最大池化
        embedding = self.pooler(embedding)
        # 返回最终嵌入值
        return embedding
class ResNetShortCut(nn.Module):
    """
    ResNet shortcut, used to project the residual features to the correct size. If needed, it is also used to
    downsample the input using `stride=2`.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        # 初始化函数，定义输入通道数、输出通道数和步长
        super().__init__()
        # 创建卷积层，用于调整残差特征的大小
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        # 创建批归一化层，用于规范化输出特征
        self.normalization = nn.BatchNorm2d(out_channels)

    def forward(self, input: Tensor) -> Tensor:
        # 将输入通过卷积层调整大小
        hidden_state = self.convolution(input)
        # 对调整后的特征进行规范化
        hidden_state = self.normalization(hidden_state)
        return hidden_state


class ResNetBasicLayer(nn.Module):
    """
    A classic ResNet's residual layer composed by two `3x3` convolutions.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, activation: str = "relu"):
        # 初始化函数，定义输入通道数、输出通道数、步长和激活函数
        super().__init__()
        # 根据条件是否需要应用shortcut，来创建shortcut模块或者空恒等映射
        should_apply_shortcut = in_channels != out_channels or stride != 1
        self.shortcut = (
            ResNetShortCut(in_channels, out_channels, stride=stride) if should_apply_shortcut else nn.Identity()
        )
        # 创建包含两个3x3卷积层的序列模块
        self.layer = nn.Sequential(
            ResNetConvLayer(in_channels, out_channels, stride=stride),
            ResNetConvLayer(out_channels, out_channels, activation=None),
        )
        # 获取激活函数
        self.activation = ACT2FN[activation]

    def forward(self, hidden_state):
        # 保存残差连接之前的特征
        residual = hidden_state
        # 通过卷积层序列得到新的特征
        hidden_state = self.layer(hidden_state)
        # 通过shortcut模块获取调整后的残差
        residual = self.shortcut(residual)
        # 将新特征和残差相加
        hidden_state += residual
        # 使用激活函数
        hidden_state = self.activation(hidden_state)
        return hidden_state


class ResNetBottleNeckLayer(nn.Module):
    """
    A classic ResNet's bottleneck layer composed by three `3x3` convolutions.

    The first `1x1` convolution reduces the input by a factor of `reduction` in order to make the second `3x3`
    convolution faster. The last `1x1` convolution remaps the reduced features to `out_channels`. If
    `downsample_in_bottleneck` is true, downsample will be in the first layer instead of the second layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        activation: str = "relu",
        reduction: int = 4,
        downsample_in_bottleneck: bool = False,
    # 定义一个ResNetBottleneck类，继承自nn.Module
    class ResNetBottleneck(nn.Module):
        # 初始化函数，接受以下参数：
        # in_channels: 输入通道数
        # out_channels: 输出通道数
        # stride: 步长
        # reduction: 通道压缩比例
        # activation: 激活函数名称
        # downsample_in_bottleneck: 是否在瓶颈层进行下采样
        def __init__(
            self,
            in_channels,
            out_channels,
            stride=1,
            reduction=4,
            activation="relu",
            downsample_in_bottleneck=False,
        ):
            super().__init__()
            # 判断是否需要添加shortcut层
            should_apply_shortcut = in_channels != out_channels or stride != 1
            # 计算压缩后的通道数
            reduces_channels = out_channels // reduction
            # 如果需要添加shortcut层，则创建ResNetShortCut层
            # 否则使用nn.Identity层
            self.shortcut = (
                ResNetShortCut(in_channels, out_channels, stride=stride) if should_apply_shortcut else nn.Identity()
            )
            # 定义主要的网络层，包括3个ResNetConvLayer
            self.layer = nn.Sequential(
                ResNetConvLayer(
                    in_channels, reduces_channels, kernel_size=1, stride=stride if downsample_in_bottleneck else 1
                ),
                ResNetConvLayer(reduces_channels, reduces_channels, stride=stride if not downsample_in_bottleneck else 1),
                ResNetConvLayer(reduces_channels, out_channels, kernel_size=1, activation=None),
            )
            # 获取激活函数
            self.activation = ACT2FN[activation]
    
        # 前向传播函数
        def forward(self, hidden_state):
            # 保存当前的residual输入
            residual = hidden_state
            # 通过主要网络层
            hidden_state = self.layer(hidden_state)
            # 通过shortcut层
            residual = self.shortcut(residual)
            # 将主要输出和shortcut输出相加
            hidden_state += residual
            # 应用激活函数
            hidden_state = self.activation(hidden_state)
            return hidden_state
class ResNetStage(nn.Module):
    """
    A ResNet stage composed by stacked layers.
    """

    def __init__(
        self,
        config: ResNetConfig,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        depth: int = 2,
    ):
        super().__init__()

        # 根据配置选择使用瓶颈块（bottleneck）还是基础块（basic），并初始化第一个层
        layer = ResNetBottleNeckLayer if config.layer_type == "bottleneck" else ResNetBasicLayer

        if config.layer_type == "bottleneck":
            # 如果是瓶颈块，第一个层可能需要下采样
            first_layer = layer(
                in_channels,
                out_channels,
                stride=stride,
                activation=config.hidden_act,
                downsample_in_bottleneck=config.downsample_in_bottleneck,
            )
        else:
            # 如果是基础块，第一个层不需要下采样
            first_layer = layer(in_channels, out_channels, stride=stride, activation=config.hidden_act)
        # 创建由多个层组成的序列
        self.layers = nn.Sequential(
            first_layer, *[layer(out_channels, out_channels, activation=config.hidden_act) for _ in range(depth - 1)]
        )

    def forward(self, input: Tensor) -> Tensor:
        hidden_state = input
        # 逐层进行前向传播
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class ResNetEncoder(nn.Module):
    def __init__(self, config: ResNetConfig):
        super().__init__()
        self.stages = nn.ModuleList([])
        # 根据配置的`downsample_in_first_stage`决定第一阶段的第一层是否进行下采样
        self.stages.append(
            ResNetStage(
                config,
                config.embedding_size,
                config.hidden_sizes[0],
                stride=2 if config.downsample_in_first_stage else 1,
                depth=config.depths[0],
            )
        )
        # 遍历隐藏层大小和深度，创建各个阶段的ResNetStage实例
        in_out_channels = zip(config.hidden_sizes, config.hidden_sizes[1:])
        for (in_channels, out_channels), depth in zip(in_out_channels, config.depths[1:]):
            self.stages.append(ResNetStage(config, in_channels, out_channels, depth=depth))

    def forward(
        self, hidden_state: Tensor, output_hidden_states: bool = False, return_dict: bool = True
    ) -> BaseModelOutputWithNoAttention:
        hidden_states = () if output_hidden_states else None

        # 对每个阶段进行前向传播
        for stage_module in self.stages:
            if output_hidden_states:
                hidden_states = hidden_states + (hidden_state,)

            hidden_state = stage_module(hidden_state)

        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(v for v in [hidden_state, hidden_states] if v is not None)

        # 返回模型输出的字典
        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_state,
            hidden_states=hidden_states,
        )


class ResNetPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 配置类为ResNetConfig
    config_class = ResNetConfig
    # 模型的基本前缀
    base_model_prefix = "resnet"
    
    # 主要输入名称
    main_input_name = "pixel_values"
    
    # 初始化权重的函数
    def _init_weights(self, module):
        # 如果是卷积层
        if isinstance(module, nn.Conv2d):
            # 使用 Kaiming 正态分布初始化权重
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        # 如果是批量归一化层或组归一化层
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            # 将权重初始化为 1，偏置初始化为 0
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
# 这是 ResNetModel 类的文档字符串，描述了该模型的功能和用法
RESNET_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ResNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 这是 ResNetModel 类 forward 方法的输入参数文档字符串
RESNET_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`ConvNextImageProcessor.__call__`] for details.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# 这是 ResNetModel 类的定义，它继承自 ResNetPreTrainedModel
@add_start_docstrings(
    "The bare ResNet model outputting raw features without any specific head on top.",
    RESNET_START_DOCSTRING,
)
class ResNetModel(ResNetPreTrainedModel):
    def __init__(self, config):
        # 调用父类的构造函数
        super().__init__(config)
        # 保存配置信息
        self.config = config
        # 创建 ResNetEmbeddings 实例
        self.embedder = ResNetEmbeddings(config)
        # 创建 ResNetEncoder 实例
        self.encoder = ResNetEncoder(config)
        # 创建全局平均池化层
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        # 初始化权重并应用最终处理
        self.post_init()

    # 这是 ResNetModel 类 forward 方法的定义
    @add_start_docstrings_to_model_forward(RESNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self, pixel_values: Tensor, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None
        ) -> BaseModelOutputWithPoolingAndNoAttention:
        # 定义函数的返回类型为BaseModelOutputWithPoolingAndNoAttention

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果output_hidden_states不为空，则使用它，否则使用self.config.output_hidden_states

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果return_dict不为空，则使用它，否则使用self.config.use_return_dict

        embedding_output = self.embedder(pixel_values)
        # 使用self.embedder处理像素值，得到嵌入输出

        encoder_outputs = self.encoder(
            embedding_output, output_hidden_states=output_hidden_states, return_dict=return_dict
        )
        # 使用self.encoder对嵌入输出进行编码，得到编码器输出

        last_hidden_state = encoder_outputs[0]
        # 获取编码器输出的最后一个隐藏状态

        pooled_output = self.pooler(last_hidden_state)
        # 使用self.pooler对最后一个隐藏状态进行池化计算，得到池化输出

        if not return_dict:
            # 如果return_dict为False
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
            # 返回最后一个隐藏状态、池化输出以及其他编码器输出的所有其他元素

        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )
        # 返回一个BaseModelOutputWithPoolingAndNoAttention对象，包括最后一个隐藏状态、池化输出和编码器输出的所有隐藏状态
@add_start_docstrings(
    """
    ResNet Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    RESNET_START_DOCSTRING,
)
class ResNetForImageClassification(ResNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 从配置中获取类别数量
        self.num_labels = config.num_labels
        # 创建 ResNet 模型
        self.resnet = ResNetModel(config)
        # 分类头部
        self.classifier = nn.Sequential(
            # 将特征展平
            nn.Flatten(),
            # 如果类别数量大于 0，则添加线性层，否则使用恒等映射
            nn.Linear(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity(),
        )
        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(RESNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> ImageClassifierOutputWithNoAttention:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 检查是否给定了 return_dict，如果没有，则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 ResNet 提取特征
        outputs = self.resnet(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        # 如果不返回字典，则获取池化后输出
        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        # 对池化后的输出进行分类
        logits = self.classifier(pooled_output)

        loss = None

        if labels is not None:
            # 判断问题类型，未指定则根据标签数量自动判断
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            # 处理回归问题
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            # 处理单标签分类问题
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            # 处理多标签分类问题
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果不返回字典，则拼接输出，包括 logits 和隐藏状态
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output

        # 返回 ImageClassifierOutputWithNoAttention 对象，包括损失、logits 和隐藏状态
        return ImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)
# 导入 ResNetPreTrainedModel 和 BackboneMixin 类
@add_start_docstrings(
    """
    ResNet backbone, to be used with frameworks like DETR and MaskFormer.
    """,
    RESNET_START_DOCSTRING,
)
# 定义 ResNetBackbone 类,继承自 ResNetPreTrainedModel 和 BackboneMixin 类
class ResNetBackbone(ResNetPreTrainedModel, BackboneMixin):
    # 初始化函数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        super()._init_backbone(config)

        # 设置特征图通道数
        self.num_features = [config.embedding_size] + config.hidden_sizes
        # 创建 ResNetEmbeddings 和 ResNetEncoder 对象
        self.embedder = ResNetEmbeddings(config)
        self.encoder = ResNetEncoder(config)

        # 初始化权重并进行后处理
        self.post_init()

    # 前向传播函数
    @add_start_docstrings_to_model_forward(RESNET_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BackboneOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self, pixel_values: Tensor, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None
    ) -> BackboneOutput:
        # 设置 return_dict 和 output_hidden_states 参数
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # 得到 embedding 输出
        embedding_output = self.embedder(pixel_values)

        # 通过 ResNetEncoder 得到输出
        outputs = self.encoder(embedding_output, output_hidden_states=True, return_dict=True)

        # 获取所有隐藏层输出
        hidden_states = outputs.hidden_states

        # 根据需要的特征图索引,从隐藏层输出中获取相应的特征图
        feature_maps = ()
        for idx, stage in enumerate(self.stage_names):
            if stage in self.out_features:
                feature_maps += (hidden_states[idx],)

        # 根据 return_dict 参数构建输出
        if not return_dict:
            output = (feature_maps,)
            if output_hidden_states:
                output += (outputs.hidden_states,)
            return output

        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=None,
        )
```