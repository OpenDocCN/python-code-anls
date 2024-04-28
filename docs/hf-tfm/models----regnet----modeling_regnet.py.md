# `.\transformers\models\regnet\modeling_regnet.py`

```
# 这是一个 PyTorch 中的 RegNet 模型的实现
# 版权由 Meta Platforms, Inc. 和 The HuggingFace Inc. 团队所有
# 该模型遵循 Apache License 2.0 协议

# 导入必要的模块和数据类型
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 从 HuggingFace 库中导入相关模块
from ...activations import ACT2FN
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_outputs import (
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
)
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_regnet import RegNetConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义通用文档字符串
_CONFIG_FOR_DOC = "RegNetConfig"
_CHECKPOINT_FOR_DOC = "facebook/regnet-y-040"
_EXPECTED_OUTPUT_SHAPE = [1, 1088, 7, 7]
_IMAGE_CLASS_CHECKPOINT = "facebook/regnet-y-040"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

# 定义支持的预训练模型列表
REGNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/regnet-y-040",
    # 查看所有 RegNet 模型 https://huggingface.co/models?filter=regnet
]

# 定义 RegNetConvLayer 类，实现 Conv-BN-Activation 的组合
class RegNetConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation: Optional[str] = "relu",
    ):
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=groups,
            bias=False,
        )
        self.normalization = nn.BatchNorm2d(out_channels)
        self.activation = ACT2FN[activation] if activation is not None else nn.Identity()

    def forward(self, hidden_state):
        hidden_state = self.convolution(hidden_state)
        hidden_state = self.normalization(hidden_state)
        hidden_state = self.activation(hidden_state)
        return hidden_state

# 定义 RegNetEmbeddings 类，实现 RegNet 的输入嵌入层
class RegNetEmbeddings(nn.Module):
    """
    RegNet Embedddings (stem) composed of a single aggressive convolution.
    """
    # 初始化函数，接受一个RegNetConfig对象作为参数
    def __init__(self, config: RegNetConfig):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个RegNetConvLayer对象作为embedder，指定输入通道数、嵌入维度、卷积核大小、步幅、激活函数
        self.embedder = RegNetConvLayer(
            config.num_channels, config.embedding_size, kernel_size=3, stride=2, activation=config.hidden_act
        )
        # 保存通道数到实例变量
        self.num_channels = config.num_channels

    # 前向传播函数，接受像素数值作为输入
    def forward(self, pixel_values):
        # 获取输入像素数值的通道数
        num_channels = pixel_values.shape[1]
        # 如果通道数不等于实例变量中保存的通道数，抛出数值错误异常
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 对输入像素数值进行嵌入操作得到隐藏状态
        hidden_state = self.embedder(pixel_values)
        # 返回隐藏状态
        return hidden_state
# 从transformers.models.resnet.modeling_resnet.ResNetShortCut复制的代码，将ResNet->RegNet
class RegNetShortCut(nn.Module):
    """
    RegNet shortcut，用于将残差特征投影到正确的大小。必要时，也用于使用`stride=2`对输入进行下采样。
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        # 1x1卷积层，用于将输入通道数变换为输出通道数，并可能进行下采样
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        # 批归一化层，用于归一化输出特征
        self.normalization = nn.BatchNorm2d(out_channels)

    def forward(self, input: Tensor) -> Tensor:
        # 输入特征经过卷积变换
        hidden_state = self.convolution(input)
        # 归一化变换后的特征
        hidden_state = self.normalization(hidden_state)
        return hidden_state


class RegNetSELayer(nn.Module):
    """
    压缩和激励层（SE），在[Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)中提出。
    """

    def __init__(self, in_channels: int, reduced_channels: int):
        super().__init__()
        # 自适应平均池化层，用于对输入特征进行池化
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        # SE层包含一系列卷积和激活操作，用于获取注意力权重
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, hidden_state):
        # b c h w -> b c 1 1
        # 对隐藏状态进行池化
        pooled = self.pooler(hidden_state)
        # 应用SE注意力获取权重
        attention = self.attention(pooled)
        hidden_state = hidden_state * attention
        return hidden_state


class RegNetXLayer(nn.Module):
    """
    RegNet的层，由三个`3x3`卷积组成，与具有reduction=1的ResNet瓶颈层相同。
    """

    def __init__(self, config: RegNetConfig, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        # 是否应用shortcut，根据输入输出通道数和步幅判断
        should_apply_shortcut = in_channels != out_channels or stride != 1
        # 计算分组数
        groups = max(1, out_channels // config.groups_width)
        # 构建shortcut模块，如果需要应用shortcut，则使用RegNetShortCut，否则使用nn.Identity()
        self.shortcut = (
            RegNetShortCut(in_channels, out_channels, stride=stride) if should_apply_shortcut else nn.Identity()
        )
        # RegNetXLayer的层包括一系列卷积操作和激活函数
        self.layer = nn.Sequential(
            RegNetConvLayer(in_channels, out_channels, kernel_size=1, activation=config.hidden_act),
            RegNetConvLayer(out_channels, out_channels, stride=stride, groups=groups, activation=config.hidden_act),
            RegNetConvLayer(out_channels, out_channels, kernel_size=1, activation=None),
        )
        self.activation = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        residual = hidden_state
        # 隐藏状态通过层的一系列操作
        hidden_state = self.layer(hidden_state)
        # 应用shortcut处理剩余的隐藏状态
        residual = self.shortcut(residual)
        hidden_state += residual
        # 使用激活函数处理隐藏状态
        hidden_state = self.activation(hidden_state)
        return hidden_state


class RegNetYLayer(nn.Module):
    """
    RegNet的Y层：带有Squeeze和Excitation的X层。
    """
    # 定义了一个 RegNetBlock 类，它是 RegNet 模型中的一个基本模块
    def __init__(self, config: RegNetConfig, in_channels: int, out_channels: int, stride: int = 1):
        # 调用父类（nn.Module）的构造方法
        super().__init__()
        # 判断是否需要应用捷径连接（shortcut）
        should_apply_shortcut = in_channels != out_channels or stride != 1
        # 计算分组数，最小为1
        groups = max(1, out_channels // config.groups_width)
        # 如果需要应用捷径连接，则创建一个 RegNetShortCut 模块
        # 否则使用一个恒等映射（nn.Identity）
        self.shortcut = (
            RegNetShortCut(in_channels, out_channels, stride=stride) if should_apply_shortcut else nn.Identity()
        )
        # 定义主要的卷积层
        self.layer = nn.Sequential(
            # 1x1 卷积层，先缩小通道数
            RegNetConvLayer(in_channels, out_channels, kernel_size=1, activation=config.hidden_act),
            # 3x3 分组卷积层，stride 可能为 1 或 2
            RegNetConvLayer(out_channels, out_channels, stride=stride, groups=groups, activation=config.hidden_act),
            # SE 层，通道注意力机制
            RegNetSELayer(out_channels, reduced_channels=int(round(in_channels / 4))),
            # 1x1 卷积层，恢复通道数
            RegNetConvLayer(out_channels, out_channels, kernel_size=1, activation=None),
        )
        # 激活函数
        self.activation = ACT2FN[config.hidden_act]
    
    # 前向传播
    def forward(self, hidden_state):
        # 保存残差
        residual = hidden_state
        # 通过主要的卷积层
        hidden_state = self.layer(hidden_state)
        # 应用捷径连接
        residual = self.shortcut(residual)
        # 残差连接
        hidden_state += residual
        # 激活函数
        hidden_state = self.activation(hidden_state)
        return hidden_state
class RegNetStage(nn.Module):
    """
    A RegNet stage composed by stacked layers.
    """

    def __init__(
        self,
        config: RegNetConfig,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        depth: int = 2,
    ):
        # 初始化函数，初始化 RegNetStage 类
        super().__init__()

        # 根据配置的层类型选择不同的层
        layer = RegNetXLayer if config.layer_type == "x" else RegNetYLayer

        # 创建包含多个层的序列
        self.layers = nn.Sequential(
            # 第一层进行降采样，步长为2
            layer(
                config,
                in_channels,
                out_channels,
                stride=stride,
            ),
            *[layer(config, out_channels, out_channels) for _ in range(depth - 1)],
        )

    def forward(self, hidden_state):
        # 前向传播函数
        hidden_state = self.layers(hidden_state)
        return hidden_state


class RegNetEncoder(nn.Module):
    def __init__(self, config: RegNetConfig):
        # 初始化函数，初始化 RegNetEncoder 类
        super().__init__()
        self.stages = nn.ModuleList([])
        # 根据`downsample_in_first_stage`设置，第一个阶段中的第一层可能对输入进行降采样
        self.stages.append(
            RegNetStage(
                config,
                config.embedding_size,
                config.hidden_sizes[0],
                stride=2 if config.downsample_in_first_stage else 1,
                depth=config.depths[0],
            )
        )
        in_out_channels = zip(config.hidden_sizes, config.hidden_sizes[1:])
        for (in_channels, out_channels), depth in zip(in_out_channels, config.depths[1:]):
            # 添加多个阶段到模型
            self.stages.append(RegNetStage(config, in_channels, out_channels, depth=depth))

    def forward(
        self, hidden_state: Tensor, output_hidden_states: bool = False, return_dict: bool = True
    ) -> BaseModelOutputWithNoAttention:
        hidden_states = () if output_hidden_states else None

        for stage_module in self.stages:
            if output_hidden_states:
                hidden_states = hidden_states + (hidden_state,)

            hidden_state = stage_module(hidden_state)

        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(v for v in [hidden_state, hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(last_hidden_state=hidden_state, hidden_states=hidden_states)


class RegNetPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 定义 RegNetPreTrainedModel 类，处理权重初始化、下载和加载预训练模型

    config_class = RegNetConfig
    base_model_prefix = "regnet"
    main_input_name = "pixel_values"

    # Copied from transformers.models.resnet.modeling_resnet.ResNetPreTrainedModel._init_weights
    # 初始化模型权重的函数
    def _init_weights(self, module):
        # 如果给定模块是二维卷积层
        if isinstance(module, nn.Conv2d):
            # 使用 Kaiming 初始化方法对卷积核进行初始化，采用"fan_out"模式和"relu"激活函数
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        # 如果给定模块是批归一化层或组归一化层
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            # 将批归一化或组归一化层的权重初始化为1
            nn.init.constant_(module.weight, 1)
            # 将批归一化或组归一化层的偏置初始化为0
            nn.init.constant_(module.bias, 0)
# 定义 RegNetModel 类的文档字符串开头部分
REGNET_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matters related to general usage and
    behavior.

    Parameters:
        config ([`RegNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义 RegNetModel 类的输入参数文档字符串
REGNET_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`ConvNextImageProcessor.__call__`] for details.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""

# 定义 RegNetModel 类
@add_start_docstrings(
    "The bare RegNet model outputting raw features without any specific head on top.",
    REGNET_START_DOCSTRING,
)
class RegNetModel(RegNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # 创建 RegNetEmbeddings 对象
        self.embedder = RegNetEmbeddings(config)
        # 创建 RegNetEncoder 对象
        self.encoder = RegNetEncoder(config)
        # 创建全局平均池化层
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(REGNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self, pixel_values: Tensor, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None
    ):
        # 在这里添加 forward 方法的注释
        ) -> BaseModelOutputWithPoolingAndNoAttention:
            # 设置 output_hidden_states 为 self.config.output_hidden_states，如果未提供则使用默认值
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            # 设置 return_dict 为 self.config.use_return_dict，如果未提供则使用默认值
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            # 将像素值传入 embedder 进行嵌入
            embedding_output = self.embedder(pixel_values)

            # 使用 encoder 处理嵌入输出，可以选择输出隐藏状态或字典形式
            encoder_outputs = self.encoder(
                embedding_output, output_hidden_states=output_hidden_states, return_dict=return_dict
            )

            # 获取最后一个隐藏状态
            last_hidden_state = encoder_outputs[0]

            # 使用 pooler 处理最后一个隐藏状态，得到池化输出
            pooled_output = self.pooler(last_hidden_state)

            # 如果没有设定返回字典形式的输出，则返回元组形式的隐藏状态和池化输出
            if not return_dict:
                return (last_hidden_state, pooled_output) + encoder_outputs[1:]

            # 返回 BaseModelOutputWithPoolingAndNoAttention 类型的对象，包含最后的隐藏状态、池化输出和隐藏状态列表
            return BaseModelOutputWithPoolingAndNoAttention(
                last_hidden_state=last_hidden_state,
                pooler_output=pooled_output,
                hidden_states=encoder_outputs.hidden_states,
            )
# 使用预训练的 RegNet 模型进行图像分类任务
@add_start_docstrings(
    """
    RegNet Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    REGNET_START_DOCSTRING,
)
# 从 transformers.models.resnet.modeling_resnet.ResNetForImageClassification 复制代码，并将 RESNET->REGNET,ResNet->RegNet,resnet->regnet
class RegNetForImageClassification(RegNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 设置分类标签数量
        self.num_labels = config.num_labels
        # 初始化 RegNet 模型
        self.regnet = RegNetModel(config)
        # 添加分类头部
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity(),
        )
        # 进行最终的参数初始化
        self.post_init()

    # 为模型前向传播添加文档字符串
    @add_start_docstrings_to_model_forward(REGNET_INPUTS_DOCSTRING)
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
    ):
    # 定义了一个返回 ImageClassifierOutputWithNoAttention 的函数
    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> ImageClassifierOutputWithNoAttention:
        # 如果 return_dict 没有被传入，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # 通过 regnet 获得输出结果，包括 pooler_output 和 hidden_states
        outputs = self.regnet(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)
    
        # 获取 pooler_output
        pooled_output = outputs.pooler_output if return_dict else outputs[1]
    
        # 将 pooled_output 传入分类器获得逻辑输出 logits
        logits = self.classifier(pooled_output)
    
        # 初始化 loss 为 None
        loss = None
    
        # 如果传入了标签 labels
        if labels is not None:
            # 如果之前没有设置 problem_type，则根据标签类型设置
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            
            # 根据 problem_type 计算损失
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
    
        # 如果不需要返回字典，则返回 loss, logits 和 hidden_states
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
    
        # 返回 ImageClassifierOutputWithNoAttention
        return ImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)
```