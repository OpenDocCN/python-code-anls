# `.\models\deprecated\van\modeling_van.py`

```py
# coding=utf-8
# 版权声明及许可信息

"""
PyTorch Visual Attention Network (VAN) model.
"""

import math
from collections import OrderedDict
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入激活函数映射
from ....activations import ACT2FN
# 导入模型输出类
from ....modeling_outputs import (
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
)
# 导入预训练模型基类
from ....modeling_utils import PreTrainedModel
# 导入工具函数：添加代码示例文档字符串、添加模型前向传播的起始文档字符串、日志记录
from ....utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
# 导入 VAN 模型配置
from .configuration_van import VanConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的配置项
_CONFIG_FOR_DOC = "VanConfig"

# 用于文档的检查点
_CHECKPOINT_FOR_DOC = "Visual-Attention-Network/van-base"
# 预期输出形状
_EXPECTED_OUTPUT_SHAPE = [1, 512, 7, 7]

# 图像分类用的检查点
_IMAGE_CLASS_CHECKPOINT = "Visual-Attention-Network/van-base"
# 预期的图像分类输出
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

# 预训练模型存档列表
VAN_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Visual-Attention-Network/van-base",
    # 查看所有 VAN 模型 https://huggingface.co/models?filter=van
]

# 从 transformers.models.convnext.modeling_convnext.drop_path 复制过来的函数
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    按样本（在残差块的主路径中应用）丢弃路径（随机深度）。

    Comment by Ross Wightman: 这与我为 EfficientNet 等网络创建的 DropConnect 实现相同，
    然而，原始名称具有误导性，因为“Drop Connect”是另一篇论文中的不同形式的 dropout…
    参见讨论: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 …
    我选择更改层和参数名称为 'drop path'，而不是将 DropConnect 作为层名称，并使用 'survival rate' 作为参数。
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # 适用于不同维度的张量，而不仅仅是 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # 二值化
    # 计算输出值，通过输入值除以保持概率得到，然后乘以一个随机生成的张量
    output = input.div(keep_prob) * random_tensor
    # 返回计算得到的输出值
    return output
# 从 transformers.models.convnext.modeling_convnext.ConvNextDropPath 复制而来，更名为 VanDropPath
class VanDropPath(nn.Module):
    """每个样本使用丢弃路径（Stochastic Depth）（当应用于残差块的主路径时）。"""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class VanOverlappingPatchEmbedder(nn.Module):
    """
    使用 patchify 操作对输入进行下采样，默认使用步幅为 4 的窗口使相邻窗口重叠一半区域。
    来自 [PVTv2: Improved Baselines with Pyramid Vision Transformer](https://arxiv.org/abs/2106.13797)。
    """

    def __init__(self, in_channels: int, hidden_size: int, patch_size: int = 7, stride: int = 4):
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=stride, padding=patch_size // 2
        )
        self.normalization = nn.BatchNorm2d(hidden_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden_state = self.convolution(input)
        hidden_state = self.normalization(hidden_state)
        return hidden_state


class VanMlpLayer(nn.Module):
    """
    带有深度卷积的 MLP，来自 [PVTv2: Improved Baselines with Pyramid Vision Transformer](https://arxiv.org/abs/2106.13797)。
    """

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        out_channels: int,
        hidden_act: str = "gelu",
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.in_dense = nn.Conv2d(in_channels, hidden_size, kernel_size=1)
        self.depth_wise = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1, groups=hidden_size)
        self.activation = ACT2FN[hidden_act]
        self.dropout1 = nn.Dropout(dropout_rate)
        self.out_dense = nn.Conv2d(hidden_size, out_channels, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.in_dense(hidden_state)
        hidden_state = self.depth_wise(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.dropout1(hidden_state)
        hidden_state = self.out_dense(hidden_state)
        hidden_state = self.dropout2(hidden_state)
        return hidden_state


class VanLargeKernelAttention(nn.Module):
    """
    基础的大核注意力（LKA）。
    """
    # 初始化函数，接受隐藏层大小作为参数
    def __init__(self, hidden_size: int):
        # 调用父类初始化方法
        super().__init__()
        # 定义深度可分离卷积层，输入和输出通道数均为 hidden_size，卷积核大小为 5x5，填充为 2
        self.depth_wise = nn.Conv2d(hidden_size, hidden_size, kernel_size=5, padding=2, groups=hidden_size)
        # 定义带孔深度可分离卷积层，输入和输出通道数均为 hidden_size，卷积核大小为 7x7，扩张率为 3，填充为 9
        self.depth_wise_dilated = nn.Conv2d(
            hidden_size, hidden_size, kernel_size=7, dilation=3, padding=9, groups=hidden_size
        )
        # 定义逐点卷积层，输入和输出通道数均为 hidden_size，卷积核大小为 1x1
        self.point_wise = nn.Conv2d(hidden_size, hidden_size, kernel_size=1)

    # 前向传播函数，接受隐藏状态张量并返回处理后的张量
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # 深度可分离卷积层的前向传播，对隐藏状态进行卷积操作
        hidden_state = self.depth_wise(hidden_state)
        # 带孔深度可分离卷积层的前向传播，对隐藏状态进行卷积操作
        hidden_state = self.depth_wise_dilated(hidden_state)
        # 逐点卷积层的前向传播，对隐藏状态进行卷积操作
        hidden_state = self.point_wise(hidden_state)
        # 返回处理后的隐藏状态张量
        return hidden_state
class VanLargeKernelAttentionLayer(nn.Module):
    """
    Computes attention using Large Kernel Attention (LKA) and attends the input.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        # 初始化一个 Large Kernel Attention 对象
        self.attention = VanLargeKernelAttention(hidden_size)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # 计算注意力权重
        attention = self.attention(hidden_state)
        # 将注意力权重应用到隐藏状态上
        attended = hidden_state * attention
        return attended


class VanSpatialAttentionLayer(nn.Module):
    """
    Van spatial attention layer composed by projection (via conv) -> act -> Large Kernel Attention (LKA) attention ->
    projection (via conv) + residual connection.
    """

    def __init__(self, hidden_size: int, hidden_act: str = "gelu"):
        super().__init__()
        # 通过卷积进行投影和激活函数
        self.pre_projection = nn.Sequential(
            OrderedDict(
                [
                    ("conv", nn.Conv2d(hidden_size, hidden_size, kernel_size=1)),
                    ("act", ACT2FN[hidden_act]),  # 使用指定的激活函数
                ]
            )
        )
        # 初始化一个 VanLargeKernelAttentionLayer 层
        self.attention_layer = VanLargeKernelAttentionLayer(hidden_size)
        # 通过卷积进行投影
        self.post_projection = nn.Conv2d(hidden_size, hidden_size, kernel_size=1)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        residual = hidden_state
        # 前向传播：投影和激活函数
        hidden_state = self.pre_projection(hidden_state)
        # 前向传播：使用注意力层
        hidden_state = self.attention_layer(hidden_state)
        # 前向传播：投影
        hidden_state = self.post_projection(hidden_state)
        # 添加残差连接
        hidden_state = hidden_state + residual
        return hidden_state


class VanLayerScaling(nn.Module):
    """
    Scales the inputs by a learnable parameter initialized by `initial_value`.
    """

    def __init__(self, hidden_size: int, initial_value: float = 1e-2):
        super().__init__()
        # 初始化一个可学习的权重参数
        self.weight = nn.Parameter(initial_value * torch.ones((hidden_size)), requires_grad=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # 通过增加维度来进行广播操作
        hidden_state = self.weight.unsqueeze(-1).unsqueeze(-1) * hidden_state
        return hidden_state


class VanLayer(nn.Module):
    """
    Van layer composed by normalization layers, large kernel attention (LKA) and a multi layer perceptron (MLP).
    """

    def __init__(
        self,
        config: VanConfig,
        hidden_size: int,
        mlp_ratio: int = 4,
        drop_path_rate: float = 0.5,
        # 省略部分
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 根据给定的 drop_path_rate 创建 VanDropPath 实例或者 nn.Identity 实例
        self.drop_path = VanDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        # 创建一个 nn.BatchNorm2d 实例，用于预处理输入数据
        self.pre_normomalization = nn.BatchNorm2d(hidden_size)
        # 创建一个 VanSpatialAttentionLayer 实例，处理输入数据的注意力机制
        self.attention = VanSpatialAttentionLayer(hidden_size, config.hidden_act)
        # 创建一个 VanLayerScaling 实例，用于缩放注意力输出
        self.attention_scaling = VanLayerScaling(hidden_size, config.layer_scale_init_value)
        # 创建一个 nn.BatchNorm2d 实例，用于处理注意力输出的后处理
        self.post_normalization = nn.BatchNorm2d(hidden_size)
        # 创建一个 VanMlpLayer 实例，处理注意力输出的 MLP 层
        self.mlp = VanMlpLayer(
            hidden_size, hidden_size * mlp_ratio, hidden_size, config.hidden_act, config.dropout_rate
        )
        # 创建一个 VanLayerScaling 实例，用于缩放 MLP 输出
        self.mlp_scaling = VanLayerScaling(hidden_size, config.layer_scale_init_value)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # 保存输入的残差连接
        residual = hidden_state
        # 对输入进行预处理
        hidden_state = self.pre_normomalization(hidden_state)
        # 应用注意力机制
        hidden_state = self.attention(hidden_state)
        # 缩放注意力输出
        hidden_state = self.attention_scaling(hidden_state)
        # 应用 drop_path 操作
        hidden_state = self.drop_path(hidden_state)
        # 添加残差连接
        hidden_state = residual + hidden_state
        # 更新残差连接
        residual = hidden_state
        # 对注意力输出进行后处理
        hidden_state = self.post_normalization(hidden_state)
        # 应用 MLP 层
        hidden_state = self.mlp(hidden_state)
        # 缩放 MLP 输出
        hidden_state = self.mlp_scaling(hidden_state)
        # 应用 drop_path 操作
        hidden_state = self.drop_path(hidden_state)
        # 添加残差连接
        hidden_state = residual + hidden_state
        # 返回处理后的输出
        return hidden_state
        hidden_state: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Perform forward pass through the VanEncoder.

        Args:
            hidden_state (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
            output_hidden_states (bool, optional): Whether to output hidden states of all stages.
            return_dict (bool, optional): Whether to return a dictionary with hidden states.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Depending on `return_dict`,
                either the final encoded tensor or a tuple containing the final tensor and a dictionary
                with hidden states from each stage.
        """
        for stage in self.stages:
            hidden_state = stage(hidden_state)

        if output_hidden_states:
            hidden_states_dict = {f"hidden_state_{i}": stage(hidden_state) for i, stage in enumerate(self.stages)}
            if return_dict:
                return hidden_state, hidden_states_dict
            else:
                return hidden_state

        return hidden_state if not return_dict else (hidden_state, {})
    ) -> Union[Tuple, BaseModelOutputWithNoAttention]:
        all_hidden_states = () if output_hidden_states else None
        # 初始化一个空元组，用于存储所有隐藏状态，如果不需要输出隐藏状态，则置为 None

        for _, stage_module in enumerate(self.stages):
            # 遍历 self.stages 中的每个阶段模块，每个模块称为 stage_module
            hidden_state = stage_module(hidden_state)
            # 对当前隐藏状态应用当前阶段模块，更新隐藏状态

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)
                # 如果需要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 元组中

        if not return_dict:
            # 如果不返回字典格式的输出
            return tuple(v for v in [hidden_state, all_hidden_states] if v is not None)
            # 返回隐藏状态和所有隐藏状态的元组，去除 None 值

        return BaseModelOutputWithNoAttention(last_hidden_state=hidden_state, hidden_states=all_hidden_states)
        # 返回一个 BaseModelOutputWithNoAttention 对象，包含最终的隐藏状态和所有隐藏状态
class VanPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 使用 VanConfig 作为该模型的配置类
    config_class = VanConfig
    # 模型的基础名称前缀为 "van"
    base_model_prefix = "van"
    # 主要输入的名称为 "pixel_values"
    main_input_name = "pixel_values"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果是 nn.Linear 模块，使用截断正态分布初始化权重
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=self.config.initializer_range)
            # 如果是 nn.Linear 模块且有偏置，则初始化偏置为常数 0
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        # 如果是 nn.LayerNorm 模块，初始化偏置为常数 0，权重为常数 1.0
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        # 如果是 nn.Conv2d 模块，使用正态分布初始化权重，偏置初始化为零
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()


# VAN_START_DOCSTRING 是一个原始字符串，用于定义关于 VAN 模型的文档字符串
VAN_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`VanConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# VAN_INPUTS_DOCSTRING 是一个原始字符串，用于定义 VAN 模型的输入文档字符串
VAN_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`ConvNextImageProcessor.__call__`] for details.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all stages. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

@add_start_docstrings(
    "The bare VAN model outputting raw features without any specific head on top. Note, VAN does not have an embedding"
    " layer.",
    VAN_START_DOCSTRING,
)
# VanModel 类继承自 VanPreTrainedModel，用于具体实现 VAN 模型
class VanModel(VanPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 设置模型配置
        self.config = config
        # 使用 VanEncoder 根据配置初始化编码器
        self.encoder = VanEncoder(config)
        # 最后的 layernorm 层，使用 nn.LayerNorm 初始化，eps 参数由 config 提供
        self.layernorm = nn.LayerNorm(config.hidden_sizes[-1], eps=config.layer_norm_eps)
        # 初始化权重并应用最终处理
        self.post_init()

    # 使用 VAN_INPUTS_DOCSTRING 注释模型前向方法的参数
    @add_start_docstrings_to_model_forward(VAN_INPUTS_DOCSTRING)
    # 将函数修饰为文档化代码示例的装饰器，指定了一些文档化参数
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    # 定义前向传播函数，接受像素值和可选的参数，返回编码器输出或元组
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor],  # 输入像素值的张量，可选
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态的布尔值，可选
        return_dict: Optional[bool] = None,  # 是否使用字典形式返回结果的布尔值，可选
    ) -> Union[Tuple, BaseModelOutputWithPoolingAndNoAttention]:  # 返回值可以是元组或指定的输出类型
        # 如果未提供输出隐藏状态的参数，则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未提供返回字典的参数，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用编码器处理输入像素值，根据参数决定是否输出隐藏状态或使用字典返回
        encoder_outputs = self.encoder(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 提取编码器输出的最后隐藏状态作为主要输出
        last_hidden_state = encoder_outputs[0]
        # 对最后隐藏状态进行全局平均池化，将高度和宽度维度降为1，保留批次和通道维度
        pooled_output = last_hidden_state.mean(dim=[-2, -1])

        # 如果不要求使用返回字典，返回编码器的最后隐藏状态和池化后的输出，以及可能的其他输出
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 如果要求使用返回字典，构造指定类型的输出对象，包括最后隐藏状态、池化输出和所有隐藏状态
        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )
# 使用 VAN 模型进行图像分类任务的定制，包含一个线性分类器作为顶层（位于池化特征之上），例如用于 ImageNet 数据集。
@add_start_docstrings(
    """
    VAN 模型，顶部附带一个图像分类头部（线性层在池化特征之上），例如适用于 ImageNet。
    """,
    VAN_START_DOCSTRING,
)
class VanForImageClassification(VanPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 初始化 VAN 模型
        self.van = VanModel(config)
        # 分类器头部
        self.classifier = (
            # 如果配置中指定的标签数大于 0，则使用线性层；否则使用恒等映射
            nn.Linear(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(VAN_INPUTS_DOCSTRING)
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
    ) -> Union[Tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 设置返回字典，如果未指定则根据配置决定是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用模型的前向传播方法，获取输出
        outputs = self.van(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        # 如果指定了使用返回字典，则从输出中获取汇聚后的特征表示
        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        # 使用分类器对汇聚后的特征表示进行分类得到 logits
        logits = self.classifier(pooled_output)

        # 初始化损失为 None
        loss = None
        # 如果提供了标签
        if labels is not None:
            # 如果问题类型未指定，则根据标签类型和类数设定问题类型
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型计算相应的损失
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.config.num_labels == 1:
                    # 对于回归问题，计算均方误差损失
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    # 对于多标签回归问题，计算均方误差损失
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # 对于单标签分类问题，计算交叉熵损失
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # 对于多标签分类问题，计算二元交叉熵损失
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果不使用返回字典，则将 logits 和额外的隐藏状态输出返回
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 使用自定义的输出类返回结果，包括损失、logits 和隐藏状态
        return ImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)
```