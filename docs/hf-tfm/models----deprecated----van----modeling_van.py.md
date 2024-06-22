# `.\models\deprecated\van\modeling_van.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 本模型代码的版权归 BNRist（清华大学）、TKLNDST（南开大学）和 HuggingFace 公司所有
# 根据 Apache 许可 2.0 版授权
# 除非符合许可，否则不得使用此文件
# 您可以在以下网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按“原样”分发软件
# 没有任何明示或暗示的担保或条件
# 有关许可的详细信息，请参阅许可证

""" PyTorch 可视化注意力网络（VAN）模型。"""

import math
from collections import OrderedDict
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 从 transformers 库中导入必要的内容
from ....activations import ACT2FN
from ....modeling_outputs import (
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
)
from ....modeling_utils import PreTrainedModel
from ....utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_van import VanConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的通用字符串
_CONFIG_FOR_DOC = "VanConfig"

# 用于文档的基本字符串
_CHECKPOINT_FOR_DOC = "Visual-Attention-Network/van-base"
_EXPECTED_OUTPUT_SHAPE = [1, 512, 7, 7]

# 图像分类文档的字符串
_IMAGE_CLASS_CHECKPOINT = "Visual-Attention-Network/van-base"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

# VAN 预训练模型存档列表
VAN_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Visual-Attention-Network/van-base",
    # 在 https://huggingface.co/models?filter=van 上查看所有 VAN 模型
]

# 从 transformers.models.convnext.modeling_convnext.drop_path 复制的函数
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    为每个样本丢弃路径（随机深度）（应用于残差块的主路径）。

    Ross Wightman 的评论：这与我为 EfficientNet 等网络创建的 DropConnect 实现相同，
    但是，原始名称有误导性，因为“Drop Connect”是另一篇论文中的不同形式的丢失连接……
    有关讨论，请参见：https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ……
    我选择更改层和参数名称为“drop path”，而不是将 DropConnect 作为层名称并使用“survival rate”作为参数。
    """
    if drop_prob == 0.0 or not training:
        # 如果丢弃概率为 0 或者不处于训练模式，则直接返回输入
        return input
    keep_prob = 1 - drop_prob
    # 计算随机张量
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # 对随机张量进行二值化
```  
    # 计算输入除以保留概率，然后乘以随机张量
    output = input.div(keep_prob) * random_tensor
    # 返回计算结果
    return output
# 定义 VanDropPath 类，用于实现在残差块的主路径中对每个样本进行随机深度（Stochastic Depth）的操作
class VanDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    # 初始化函数，设置 drop_prob 参数，默认为 None
    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        # 存储 drop_prob 参数值
        self.drop_prob = drop_prob

    # 前向传播函数，对输入的 hidden_states 进行 drop path 操作
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 调用 drop_path 函数实现 drop path 操作，根据当前是否处于训练状态来确定是否执行 drop path
        return drop_path(hidden_states, self.drop_prob, self.training)

    # 返回类的额外表示信息，显示当前对象的 drop_prob 参数值
    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


# 定义 VanOverlappingPatchEmbedder 类，用于进行输入的降采样操作，采用默认的 stride=4 实现相邻窗口的重叠
class VanOverlappingPatchEmbedder(nn.Module):
    """
    Downsamples the input using a patchify operation with a `stride` of 4 by default making adjacent windows overlap by
    half of the area. From [PVTv2: Improved Baselines with Pyramid Vision
    Transformer](https://arxiv.org/abs/2106.13797).
    """

    # 初始化函数，设置输入通道数 in_channels、隐藏层大小 hidden_size、patch 大小 patch_size 和步长 stride，默认为 7 和 4
    def __init__(self, in_channels: int, hidden_size: int, patch_size: int = 7, stride: int = 4):
        super().__init__()
        # 定义卷积层，用于实现降采样操作
        self.convolution = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=stride, padding=patch_size // 2
        )
        # 定义批归一化层，对降采样后的特征进行归一化处理
        self.normalization = nn.BatchNorm2d(hidden_size)

    # 前向传播函数，对输入进行降采样操作，并进行归一化处理
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden_state = self.convolution(input)
        hidden_state = self.normalization(hidden_state)
        return hidden_state


# 定义 VanMlpLayer 类，用于实现带有深度卷积的 MLP 层
class VanMlpLayer(nn.Module):
    """
    MLP with depth-wise convolution, from [PVTv2: Improved Baselines with Pyramid Vision
    Transformer](https://arxiv.org/abs/2106.13797).
    """

    # 初始化函数，设置输入通道数 in_channels、隐藏层大小 hidden_size、输出通道数 out_channels、隐藏层激活函数 hidden_act，默认为 "gelu"，以及 dropout_rate，默认为 0.5
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        out_channels: int,
        hidden_act: str = "gelu",
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        # 定义输入密集层，用于将输入特征进行维度变换
        self.in_dense = nn.Conv2d(in_channels, hidden_size, kernel_size=1)
        # 定义深度卷积层，实现 MLP 中的深度连接
        self.depth_wise = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1, groups=hidden_size)
        # 定义激活函数，根据 hidden_act 参数确定
        self.activation = ACT2FN[hidden_act]
        # 定义 dropout 操作，用于防止过拟合
        self.dropout1 = nn.Dropout(dropout_rate)
        # 定义输出密集层，将特征映射到输出通道数维度
        self.out_dense = nn.Conv2d(hidden_size, out_channels, kernel_size=1)
        # 再次定义 dropout 操作
        self.dropout2 = nn.Dropout(dropout_rate)

    # 前向传播函数，实现 MLP 层的前向计算过程
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.in_dense(hidden_state)
        hidden_state = self.depth_wise(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.dropout1(hidden_state)
        hidden_state = self.out_dense(hidden_state)
        hidden_state = self.dropout2(hidden_state)
        return hidden_state


# 定义 VanLargeKernelAttention 类，用于实现基本的大核注意力机制
class VanLargeKernelAttention(nn.Module):
    """
    Basic Large Kernel Attention (LKA).
    """
    # 定义一个类的初始化方法，接受隐藏层大小作为参数
    def __init__(self, hidden_size: int):
        # 调用父类的初始化方法
        super().__init__()
        # 定义深度可分离卷积层，输入和输出通道数都是 hidden_size，卷积核大小为 5x5，填充为2，组数为 hidden_size
        self.depth_wise = nn.Conv2d(hidden_size, hidden_size, kernel_size=5, padding=2, groups=hidden_size)
        # 定义带扩张的深度可分离卷积层，输入和输出通道数都是 hidden_size，卷积核大小为 7x7，扩张率为3，填充为9，组数为 hidden_size
        self.depth_wise_dilated = nn.Conv2d(
            hidden_size, hidden_size, kernel_size=7, dilation=3, padding=9, groups=hidden_size
        )
        # 定义逐点卷积层，输入和输出通道数都是 hidden_size，卷积核大小为 1x1
        self.point_wise = nn.Conv2d(hidden_size, hidden_size, kernel_size=1)
    
    # 定义类的前向传播方法，接受一个 torch.Tensor 类型的隐藏状态输入，返回一个 torch.Tensor 类型的输出
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # 对输入的隐藏状态进行深度可分离卷积操作
        hidden_state = self.depth_wise(hidden_state)
        # 对深度可分离卷积后的结果进行带扩张的深度可分离卷积操作
        hidden_state = self.depth_wise_dilated(hidden_state)
        # 对带扩张的深度可分离卷积后的结果进行逐点卷积操作
        hidden_state = self.point_wise(hidden_state)
        # 返回最终处理后的隐藏状态
        return hidden_state
class VanLargeKernelAttentionLayer(nn.Module):
    """
    Computes attention using Large Kernel Attention (LKA) and attends the input.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = VanLargeKernelAttention(hidden_size)  # 初始化 Large Kernel Attention 对象

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        attention = self.attention(hidden_state)  # 计算注意力权重
        attended = hidden_state * attention  # 利用注意力权重对隐藏状态进行加权
        return attended  # 返回加权后的隐藏状态


class VanSpatialAttentionLayer(nn.Module):
    """
    Van spatial attention layer composed by projection (via conv) -> act -> Large Kernel Attention (LKA) attention ->
    projection (via conv) + residual connection.
    """

    def __init__(self, hidden_size: int, hidden_act: str = "gelu"):
        super().__init__()
        # 前投影层，包括卷积和激活函数
        self.pre_projection = nn.Sequential(
            OrderedDict(
                [
                    ("conv", nn.Conv2d(hidden_size, hidden_size, kernel_size=1)),  # 卷积层
                    ("act", ACT2FN[hidden_act]),  # 激活函数
                ]
            )
        )
        self.attention_layer = VanLargeKernelAttentionLayer(hidden_size)  # 初始化 Large Kernel Attention 层
        self.post_projection = nn.Conv2d(hidden_size, hidden_size, kernel_size=1)  # 后投影层，卷积层

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        residual = hidden_state  # 保存残差连接
        hidden_state = self.pre_projection(hidden_state)  # 前投影
        hidden_state = self.attention_layer(hidden_state)  # 利用 Large Kernel Attention 进行注意力计算
        hidden_state = self.post_projection(hidden_state)  # 后投影
        hidden_state = hidden_state + residual  # 加上残差连接
        return hidden_state  # 返回处理后的隐藏状态


class VanLayerScaling(nn.Module):
    """
    Scales the inputs by a learnable parameter initialized by `initial_value`.
    """

    def __init__(self, hidden_size: int, initial_value: float = 1e-2):
        super().__init__()
        self.weight = nn.Parameter(initial_value * torch.ones((hidden_size)), requires_grad=True)  # 初始化可学习的参数

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # unsqueezing for broadcasting
        hidden_state = self.weight.unsqueeze(-1).unsqueeze(-1) * hidden_state  # 利用学习到的参数对隐藏状态进行缩放
        return hidden_state  # 返回缩放后的隐藏状态


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
        ```py
    # 初始化方法，继承父类初始化方法
    def __init__(self, drop_path_rate):
        super().__init__()
        # 如果dropout路径比率大于0，则设置drop_path为VanDropPath实例，否则为nn.Identity()实例
        self.drop_path = VanDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        # 初始化预归一化层，使用隐藏层大小为参数
        self.pre_normomalization = nn.BatchNorm2d(hidden_size)
        # 初始化注意力层，使用隐藏层大小和隐藏激活函数为参数
        self.attention = VanSpatialAttentionLayer(hidden_size, config.hidden_act)
        # 初始化注意力层缩放，使用隐藏层大小和层缩放初始值为参数
        self.attention_scaling = VanLayerScaling(hidden_size, config.layer_scale_init_value)
        # 初始化后归一化层，使用隐藏层大小为参数
        self.post_normalization = nn.BatchNorm2d(hidden_size)
        # 初始化多层感知机层，使用隐藏层大小、MLP比率、隐藏层大小、隐藏激活函数和dropout率为参数
        self.mlp = VanMlpLayer(
            hidden_size, hidden_size * mlp_ratio, hidden_size, config.hidden_act, config.dropout_rate
        )
        # 初始化MLP层缩放，使用隐藏层大小和层缩放初始值为参数
        self.mlp_scaling = VanLayerScaling(hidden_size, config.layer_scale_init_value)

    # 前向传播方法，输入hidden_state张量，返回张量
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # 保存残差连接
        residual = hidden_state
        # 预归一化
        hidden_state = self.pre_normomalization(hidden_state)
        # 注意力计算
        hidden_state = self.attention(hidden_state)
        # 注意力缩放
        hidden_state = self.attention_scaling(hidden_state)
        # Dropout Path
        hidden_state = self.drop_path(hidden_state)
        # 残差连接
        hidden_state = residual + hidden_state
        # 更新残差连接
        residual = hidden_state
        # 后归一化
        hidden_state = self.post_normalization(hidden_state)
        # MLP计算
        hidden_state = self.mlp(hidden_state)
        # MLP缩放
        hidden_state = self.mlp_scaling(hidden_state)
        # Dropout Path
        hidden_state = self.drop_path(hidden_state)
        # 残差连接
        hidden_state = residual + hidden_state
        # 返回结果张量
        return hidden_state
class VanStage(nn.Module):
    """
    VanStage, consisting of multiple layers.
    """

    def __init__(
        self,
        config: VanConfig,
        in_channels: int,
        hidden_size: int,
        patch_size: int,
        stride: int,
        depth: int,
        mlp_ratio: int = 4,
        drop_path_rate: float = 0.0,
    ):
        # 初始化VanStage类
        super().__init__()
        # 使用VanOverlappingPatchEmbedder类对输入进行嵌入处理
        self.embeddings = VanOverlappingPatchEmbedder(in_channels, hidden_size, patch_size, stride)
        self.layers = nn.Sequential(
            *[
                # 使用VanLayer类构建多个层组成的神经网络
                VanLayer(
                    config,
                    hidden_size,
                    mlp_ratio=mlp_ratio,
                    drop_path_rate=drop_path_rate,
                )
                for _ in range(depth)
            ]
        )
        # 对隐藏状态进行归一化处理
        self.normalization = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.embeddings(hidden_state)  # 使用嵌入器对隐藏状态进行处理
        hidden_state = self.layers(hidden_state)  # 对隐藏状态进行多层神经网络处理
        # 重新排列张量的维度
        batch_size, hidden_size, height, width = hidden_state.shape
        hidden_state = hidden_state.flatten(2).transpose(1, 2)
        hidden_state = self.normalization(hidden_state)  # 对隐藏状态进行归一化处理
        # 重新排列张量的维度
        hidden_state = hidden_state.view(batch_size, height, width, hidden_size).permute(0, 3, 1, 2)
        return hidden_state  # 返回处理后的隐藏状态


class VanEncoder(nn.Module):
    """
    VanEncoder, consisting of multiple stages.
    """

    def __init__(self, config: VanConfig):
        # 初始化VanEncoder类
        super().__init__()
        self.stages = nn.ModuleList([])
        patch_sizes = config.patch_sizes
        strides = config.strides
        hidden_sizes = config.hidden_sizes
        depths = config.depths
        mlp_ratios = config.mlp_ratios
        drop_path_rates = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]

        for num_stage, (patch_size, stride, hidden_size, depth, mlp_expantion, drop_path_rate) in enumerate(
            zip(patch_sizes, strides, hidden_sizes, depths, mlp_ratios, drop_path_rates)
        ):
            is_first_stage = num_stage == 0
            in_channels = hidden_sizes[num_stage - 1]
            if is_first_stage:
                in_channels = config.num_channels
            self.stages.append(
                VanStage(
                    config,
                    in_channels,
                    hidden_size,
                    patch_size=patch_size,
                    stride=stride,
                    depth=depth,
                    mlp_ratio=mlp_expantion,
                    drop_path_rate=drop_path_rate,
                )
            )

    def forward(
        self,
        hidden_state: torch.Tensor,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    # 定义函数，接受输入参数和返回类型注解
    ) -> Union[Tuple, BaseModelOutputWithNoAttention]:
        # 确定是否需要输出所有隐藏状态
        all_hidden_states = () if output_hidden_states else None
    
        # 遍历各个阶段的模块
        for _, stage_module in enumerate(self.stages):
            # 使用当前阶段的模块处理隐藏状态
            hidden_state = stage_module(hidden_state)
    
            # 如果需要输出所有隐藏状态
            if output_hidden_states:
                # 将当前隐藏状态添加到所有隐藏状态元组中
                all_hidden_states = all_hidden_states + (hidden_state,)
    
        # 如果不要返回字典形式的结果
        if not return_dict:
            # 返回非None的隐藏状态和所有隐藏状态
            return tuple(v for v in [hidden_state, all_hidden_states] if v is not None)
    
        # 返回BaseModelOutputWithNoAttention类型的结果
        return BaseModelOutputWithNoAttention(last_hidden_state=hidden_state, hidden_states=all_hidden_states)
class VanPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置配置类
    config_class = VanConfig
    # 设置基础模型前缀
    base_model_prefix = "van"
    # 设置主要输入名称
    main_input_name = "pixel_values"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果是线性层，使用截断正态分布初始化权重，偏置初始化为0
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        # 如果是LayerNorm层，偏置初始化为0，权重初始化为1
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        # 如果是卷积层，使用正态分布初始化权重，偏置初始化为0
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                module.bias.data.zero_()


VAN_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`VanConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

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
class VanModel(VanPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # 初始化编码器
        self.encoder = VanEncoder(config)
        # final layernorm layer
        self.layernorm = nn.LayerNorm(config.hidden_sizes[-1], eps=config.layer_norm_eps)
        # Initialize weights and apply final processing
        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(VAN_INPUTS_DOCSTRING)
    # 添加代码示例的文档字符串，用于生成 API 文档
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,  # 检查点参数用于文档生成
        output_type=BaseModelOutputWithPoolingAndNoAttention,  # 输出类型
        config_class=_CONFIG_FOR_DOC,  # 用于文档生成的配置类
        modality="vision",  # 输入模态
        expected_output=_EXPECTED_OUTPUT_SHAPE,  # 预期输出形状
    )
    # 前向传播函数，接收像素值作为输入，返回包含池化和无注意力的基础模型输出类型的结果
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor],  # 像素值，可选的浮点张量
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选的布尔值
        return_dict: Optional[bool] = None,  # 是否返回字典，可选的布尔值
    ) -> Union[Tuple, BaseModelOutputWithPoolingAndNoAttention]:  # 函数返回值的类型注释
        output_hidden_states = (  # 如果输出隐藏状态参数不为None，则使用传入的参数，否则使用配置中的输出隐藏状态设置
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # 如果返回字典参数不为None，则使用传入的参数，否则使用配置中的返回字典设置
        
        encoder_outputs = self.encoder(  # 使用编码器处理像素值，获取编码器的输出
            pixel_values,
            output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
            return_dict=return_dict,  # 是否返回字典
        )
        last_hidden_state = encoder_outputs[0]  # 获取编码器输出的最后一个隐藏状态
        # 全局平均池化，将 n c w h 的张量平均到 n c 的形状
        pooled_output = last_hidden_state.mean(dim=[-2, -1])
    
        if not return_dict:  # 如果不返回字典，则返回元组
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
    
        return BaseModelOutputWithPoolingAndNoAttention(  # 如果返回字典，则返回包含池化和无注意力的基础模型输出类型的结果
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )
# 使用 VAN 模型进行图像分类，该模型在顶部具有图像分类头（在池化特征的顶部是一个线性层），例如用于 ImageNet 数据集
@add_start_docstrings(
    """
    VAN Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    VAN_START_DOCSTRING,
)
class VanForImageClassification(VanPreTrainedModel):
    # 初始化函数
    def __init__(self, config):
        # 调用基类的初始化函数
        super().__init__(config)
        # 创建 VAN 模型
        self.van = VanModel(config)
        # 分类器头部
        self.classifier = (
            nn.Linear(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # forward 方法
    @add_start_docstrings_to_model_forward(VAN_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        # 像素值（可选）
        pixel_values: Optional[torch.FloatTensor] = None,
        # 标签（可选）
        labels: Optional[torch.LongTensor] = None,
        # 是否输出隐藏状态（可选）
        output_hidden_states: Optional[bool] = None,
        # 返回字典（可选）
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用现有的模型参数调用van方法，得到输出
        outputs = self.van(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        # 如果使用返回字典，则将outputs.pooler_output赋值给pooled_output；否则，将outputs[1]赋值给pooled_output
        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        # 将pooled_output输入到分类器中得到logits
        logits = self.classifier(pooled_output)

        # 初始化loss变量
        loss = None
        # 如果labels不为空
        if labels is not None:
            # 如果模型配置的问题类型为空
            if self.config.problem_type is None:
                # 如果模型配置的标签数为1，问题类型为回归
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                # 如果标签数大于1且标签数据类型为torch.long或torch.int，问题类型为单标签分类
                elif self.config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                # 否则，问题类型为多标签分类
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型的不同选择不同的损失计算方法
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.config.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果不使用返回字典，则将logits和outputs的后续元素组成元组返回
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果使用返回字典，返回ImageClassifierOutputWithNoAttention对象
        return ImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)
```