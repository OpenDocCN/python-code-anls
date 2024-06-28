# `.\models\resnet\modeling_resnet.py`

```
# 导入必要的库和模块
from typing import Optional

import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入模型输出相关的类
from ...activations import ACT2FN
from ...modeling_outputs import (
    BackboneOutput,
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
)
# 导入预训练模型相关的工具和类
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ...utils.backbone_utils import BackboneMixin
# 导入 ResNet 的配置类
from .configuration_resnet import ResNetConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "ResNetConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "microsoft/resnet-50"
_EXPECTED_OUTPUT_SHAPE = [1, 2048, 7, 7]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "microsoft/resnet-50"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tiger cat"

# 预定义的 ResNet 预训练模型列表
RESNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/resnet-50",
    # See all resnet models at https://huggingface.co/models?filter=resnet
]

# 定义 ResNetConvLayer 类，实现 ResNet 的卷积层
class ResNetConvLayer(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, activation: str = "relu"
    ):
        super().__init__()
        # 定义卷积层
        self.convolution = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=False
        )
        # 定义批标准化层
        self.normalization = nn.BatchNorm2d(out_channels)
        # 定义激活函数，从预定义的 ACT2FN 中选择
        self.activation = ACT2FN[activation] if activation is not None else nn.Identity()

    def forward(self, input: Tensor) -> Tensor:
        # 前向传播函数
        hidden_state = self.convolution(input)  # 卷积操作
        hidden_state = self.normalization(hidden_state)  # 批标准化操作
        hidden_state = self.activation(hidden_state)  # 激活函数操作
        return hidden_state

# 定义 ResNetEmbeddings 类，用于 ResNet 的嵌入（stem）部分
class ResNetEmbeddings(nn.Module):
    """
    ResNet Embeddings (stem) composed of a single aggressive convolution.
    """
    # 初始化函数，用于创建一个 ResNet 模型对象
    def __init__(self, config: ResNetConfig):
        # 调用父类的初始化函数，确保继承自父类的属性被正确初始化
        super().__init__()
        # 创建一个 ResNetConvLayer 对象作为嵌入层，配置如下：
        #   - 输入通道数为 config.num_channels
        #   - 输出特征维度为 config.embedding_size
        #   - 卷积核大小为 7x7
        #   - 步长为 2
        #   - 激活函数为 config.hidden_act
        self.embedder = ResNetConvLayer(
            config.num_channels, config.embedding_size, kernel_size=7, stride=2, activation=config.hidden_act
        )
        # 创建一个最大池化层对象，配置如下：
        #   - 池化核大小为 3x3
        #   - 步长为 2
        #   - 填充为 1
        self.pooler = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 将 config 中的通道数设置为当前对象的通道数属性
        self.num_channels = config.num_channels

    # 前向传播函数，用于定义数据从输入到输出的流程
    def forward(self, pixel_values: Tensor) -> Tensor:
        # 获取输入张量 pixel_values 的通道数
        num_channels = pixel_values.shape[1]
        # 如果输入张量的通道数与初始化时设置的通道数不一致，则抛出数值错误异常
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 将输入张量 pixel_values 通过 embedder 进行嵌入操作得到 embedding
        embedding = self.embedder(pixel_values)
        # 对 embedding 使用 pooler 进行最大池化操作
        embedding = self.pooler(embedding)
        # 返回池化后的 embedding 张量作为最终输出
        return embedding
# 定义一个经典的 ResNet 瓶颈层，由三个 3x3 的卷积组成。

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
    ):
        super().__init__()
        # 判断是否需要在瓶颈层中进行降采样
        should_apply_shortcut = in_channels != out_channels or stride != 1
        # 如果需要降采样，则使用 ResNetShortCut 类来处理，否则使用恒等映射 nn.Identity()
        self.shortcut = (
            ResNetShortCut(in_channels, out_channels, stride=stride) if should_apply_shortcut else nn.Identity()
        )
        # 第一个 3x3 卷积层，用于降维或保持维度不变
        self.layer1 = ResNetConvLayer(in_channels, out_channels // reduction, kernel_size=1, stride=1)
        # 第二个 3x3 卷积层，用于特征提取
        self.layer2 = ResNetConvLayer(out_channels // reduction, out_channels // reduction, kernel_size=3, stride=stride)
        # 第三个 1x1 卷积层，用于将特征映射回原始维度
        self.layer3 = ResNetConvLayer(out_channels // reduction, out_channels, kernel_size=1, stride=1, activation=None)
        # 激活函数，根据传入的 activation 参数选择相应的激活函数
        self.activation = ACT2FN[activation]

    def forward(self, hidden_state):
        # 将输入作为残差项备份
        residual = hidden_state
        # 依次通过各卷积层
        hidden_state = self.layer1(hidden_state)
        hidden_state = self.layer2(hidden_state)
        hidden_state = self.layer3(hidden_state)
        # 应用可能的降采样或恒等映射
        residual = self.shortcut(residual)
        # 将残差项与卷积输出相加
        hidden_state += residual
        # 应用激活函数
        hidden_state = self.activation(hidden_state)
        return hidden_state
    ):
        # 调用父类的构造方法进行初始化
        super().__init__()
        # 确定是否应用快捷方式，根据输入通道数、输出通道数和步长来判断
        should_apply_shortcut = in_channels != out_channels or stride != 1
        # 计算减少的通道数，用于残差块内的维度减少操作
        reduces_channels = out_channels // reduction
        # 如果需要应用快捷方式，则创建ResNetShortCut对象；否则创建一个恒等映射对象(nn.Identity())
        self.shortcut = (
            ResNetShortCut(in_channels, out_channels, stride=stride) if should_apply_shortcut else nn.Identity()
        )
        # 构建残差块的主要层序列
        self.layer = nn.Sequential(
            # 第一个卷积层，用于减少输入通道数或步长的卷积操作
            ResNetConvLayer(
                in_channels, reduces_channels, kernel_size=1, stride=stride if downsample_in_bottleneck else 1
            ),
            # 第二个卷积层，不进行减少的卷积操作
            ResNetConvLayer(reduces_channels, reduces_channels, stride=stride if not downsample_in_bottleneck else 1),
            # 第三个卷积层，用于将减少的通道数映射回输出通道数的卷积操作，不使用激活函数
            ResNetConvLayer(reduces_channels, out_channels, kernel_size=1, activation=None),
        )
        # 根据给定的激活函数名称选择对应的激活函数
        self.activation = ACT2FN[activation]

    def forward(self, hidden_state):
        # 将输入的隐藏状态作为残差块的输入
        residual = hidden_state
        # 经过残差块的主要层序列操作，得到输出的隐藏状态
        hidden_state = self.layer(hidden_state)
        # 对输入的残差应用快捷方式，将其映射到与主要层输出相同的维度空间
        residual = self.shortcut(residual)
        # 将残差与主要层的输出相加，形成残差块的最终输出
        hidden_state += residual
        # 对最终输出应用预定义的激活函数
        hidden_state = self.activation(hidden_state)
        # 返回处理后的隐藏状态作为残差块的最终输出
        return hidden_state
# 定义 ResNet 网络的一个阶段，由多个堆叠的层组成
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

        # 根据配置选择使用瓶颈块或基础块作为层
        layer = ResNetBottleNeckLayer if config.layer_type == "bottleneck" else ResNetBasicLayer

        # 根据配置选择不同的第一层
        if config.layer_type == "bottleneck":
            first_layer = layer(
                in_channels,
                out_channels,
                stride=stride,
                activation=config.hidden_act,
                downsample_in_bottleneck=config.downsample_in_bottleneck,
            )
        else:
            first_layer = layer(in_channels, out_channels, stride=stride, activation=config.hidden_act)
        
        # 创建包含多个层的序列容器
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
    """
    ResNet 编码器由多个 ResNet 阶段组成。
    """

    def __init__(self, config: ResNetConfig):
        super().__init__()
        self.stages = nn.ModuleList([])
        
        # 根据 `downsample_in_first_stage` 确定第一个阶段的第一层是否降采样输入
        self.stages.append(
            ResNetStage(
                config,
                config.embedding_size,
                config.hidden_sizes[0],
                stride=2 if config.downsample_in_first_stage else 1,
                depth=config.depths[0],
            )
        )
        
        # 构建其余阶段
        in_out_channels = zip(config.hidden_sizes, config.hidden_sizes[1:])
        for (in_channels, out_channels), depth in zip(in_out_channels, config.depths[1:]):
            self.stages.append(ResNetStage(config, in_channels, out_channels, depth=depth))

    def forward(
        self, hidden_state: Tensor, output_hidden_states: bool = False, return_dict: bool = True
    ) -> BaseModelOutputWithNoAttention:
        hidden_states = () if output_hidden_states else None

        # 遍历每个阶段进行前向传播
        for stage_module in self.stages:
            if output_hidden_states:
                hidden_states = hidden_states + (hidden_state,)
            
            hidden_state = stage_module(hidden_state)

        # 如果需要输出隐藏状态，则将最终的隐藏状态添加到隐藏状态元组中
        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)

        # 如果不需要返回字典，则根据情况返回隐藏状态和/或隐藏状态元组
        if not return_dict:
            return tuple(v for v in [hidden_state, hidden_states] if v is not None)

        # 返回带有最终隐藏状态和隐藏状态元组的 BaseModelOutputWithNoAttention 对象
        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_state,
            hidden_states=hidden_states,
        )


class ResNetPreTrainedModel(PreTrainedModel):
    """
    处理权重初始化和预训练模型下载和加载的抽象类。
    """

    # 指定配置类为 ResNetConfig
    config_class = ResNetConfig
    # 定义基础模型前缀为 "resnet"
    base_model_prefix = "resnet"
    
    # 定义主要输入名称为 "pixel_values"
    main_input_name = "pixel_values"
    
    # 定义初始化权重函数，接受一个模块作为参数
    def _init_weights(self, module):
        # 如果模块是 nn.Conv2d 类型，则使用 Kaiming 初始化方法初始化权重
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        # 如果模块是 nn.BatchNorm2d 或 nn.GroupNorm 类型，则将权重初始化为 1，偏置初始化为 0
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
# 定义一个多行字符串，用于描述此模型是一个 PyTorch 的 torch.nn.Module 子类，使用时需按照一般的 PyTorch 模块方式使用，
# 参考 PyTorch 文档了解一般用法和行为。
RESNET_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ResNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义一个多行字符串，用于描述输入参数的文档信息，包括像素值、是否返回所有层的隐藏状态以及返回类型的选择。
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

# 使用装饰器 @add_start_docstrings 和提供的多行字符串 RESNET_START_DOCSTRING，为 ResNetModel 类添加文档说明。
@add_start_docstrings(
    "The bare ResNet model outputting raw features without any specific head on top.",
    RESNET_START_DOCSTRING,
)
class ResNetModel(ResNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embedder = ResNetEmbeddings(config)
        self.encoder = ResNetEncoder(config)
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        # Initialize weights and apply final processing
        self.post_init()

    # 使用装饰器 @add_start_docstrings_to_model_forward 和提供的多行字符串 RESNET_INPUTS_DOCSTRING，为 forward 方法添加输入文档说明。
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
        # 指定函数的返回类型，这里返回一个带有池化和无注意力的基础模型输出
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果输出隐藏状态未指定，则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用嵌入器（embedder）处理像素值，生成嵌入输出
        embedding_output = self.embedder(pixel_values)

        # 使用编码器（encoder）处理嵌入输出，可以选择输出隐藏状态和是否返回字典
        encoder_outputs = self.encoder(
            embedding_output, output_hidden_states=output_hidden_states, return_dict=return_dict
        )

        # 获取编码器输出的最后隐藏状态
        last_hidden_state = encoder_outputs[0]

        # 使用池化器（pooler）对最后隐藏状态进行池化
        pooled_output = self.pooler(last_hidden_state)

        # 如果不返回字典，则返回元组形式的输出
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 如果需要返回字典，则构建特定的基础模型输出对象并返回
        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )
    """
    ResNet Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """
    # 使用 ResNetPreTrainedModel 作为基类，构建一个带有图像分类头部的 ResNet 模型，用于 ImageNet 等任务
    class ResNetForImageClassification(ResNetPreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            # 设置分类标签数量
            self.num_labels = config.num_labels
            # 初始化 ResNet 模型
            self.resnet = ResNetModel(config)
            # 分类头部，包括展平层和线性层，根据配置决定是否使用标签数量进行分类
            self.classifier = nn.Sequential(
                nn.Flatten(),
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
        # 前向传播函数，接收像素值、标签、是否输出隐藏状态和是否返回字典作为参数
        def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> ImageClassifierOutputWithNoAttention:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 如果 return_dict 不为 None，则使用该值；否则使用 self.config.use_return_dict 的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 ResNet 模型来计算输出，根据 return_dict 是否为 True 返回不同的输出
        outputs = self.resnet(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        # 如果 return_dict 为 False，则使用 outputs 的第二个元素作为汇集的输出；否则使用 outputs 的 pooler_output
        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        # 将汇集的输出传入分类器，得到 logits
        logits = self.classifier(pooled_output)

        # 初始化损失为 None
        loss = None

        # 如果 labels 不为 None，则开始计算损失
        if labels is not None:
            # 如果问题类型尚未确定，则根据条件确定问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            
            # 根据问题类型选择合适的损失函数和计算方式
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    # 对于单一标签的回归问题，计算损失
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    # 对于多标签的回归问题，计算损失
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # 对于单标签分类问题，使用交叉熵损失函数
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # 对于多标签分类问题，使用带 logits 的二元交叉熵损失函数
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果 return_dict 为 False，则返回 logits 和额外的 hidden states；否则返回损失和 logits
        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output
        
        # 返回 ImageClassifierOutputWithNoAttention 对象，其中包括损失、logits 和 hidden states
        return ImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)
@add_start_docstrings(
    """
    ResNet backbone, to be used with frameworks like DETR and MaskFormer.
    """,
    RESNET_START_DOCSTRING,
)
class ResNetBackbone(ResNetPreTrainedModel, BackboneMixin):
    def __init__(self, config):
        # 调用父类的初始化方法，传入配置参数
        super().__init__(config)
        # 调用基类的初始化背景方法
        super()._init_backbone(config)

        # 设置特征维度列表，包括嵌入大小和隐藏层大小
        self.num_features = [config.embedding_size] + config.hidden_sizes
        # 初始化嵌入器
        self.embedder = ResNetEmbeddings(config)
        # 初始化编码器
        self.encoder = ResNetEncoder(config)

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(RESNET_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BackboneOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self, pixel_values: Tensor, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None
    ) -> BackboneOutput:
        """
        返回模型的输出结果。
        
        Examples:
        
        ```python
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        >>> model = AutoBackbone.from_pretrained(
        ...     "microsoft/resnet-50", out_features=["stage1", "stage2", "stage3", "stage4"]
        ... )

        >>> inputs = processor(image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> feature_maps = outputs.feature_maps
        >>> list(feature_maps[-1].shape)
        [1, 2048, 7, 7]
        ```
        """
        # 确定是否返回字典格式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 确定是否输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # 对输入的像素值进行嵌入处理
        embedding_output = self.embedder(pixel_values)

        # 使用编码器处理嵌入输出，并请求输出隐藏状态
        outputs = self.encoder(embedding_output, output_hidden_states=True, return_dict=True)

        # 获取隐藏状态
        hidden_states = outputs.hidden_states

        # 初始化空的特征映射元组
        feature_maps = ()
        # 遍历阶段名称和隐藏状态，添加符合输出特征的阶段
        for idx, stage in enumerate(self.stage_names):
            if stage in self.out_features:
                feature_maps += (hidden_states[idx],)

        # 如果不返回字典格式的输出，组合输出并包含隐藏状态
        if not return_dict:
            output = (feature_maps,)
            if output_hidden_states:
                output += (outputs.hidden_states,)
            return output

        # 返回 BackboneOutput 对象，包含特征映射、隐藏状态（如果有）和注意力（为空）
        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=None,
        )
```