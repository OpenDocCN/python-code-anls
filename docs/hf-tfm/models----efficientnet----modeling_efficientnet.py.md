# `.\models\efficientnet\modeling_efficientnet.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 使用 Apache 许可证版本 2.0 授权
# 详细许可证请参见 http://www.apache.org/licenses/LICENSE-2.0

# 导入必要的库和模块
import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
# 导入模型相关输出和工具函数
from ...activations import ACT2FN
from ...modeling_outputs import (
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
)

# 设置日志
logger = logging.get_logger(__name__)

# 用以提供全局注释
_CONFIG_FOR_DOC = "EfficientNetConfig"
_CHECKPOINT_FOR_DOC = "google/efficientnet-b7"
_EXPECTED_OUTPUT_SHAPE = [1, 768, 7, 7]
_IMAGE_CLASS_CHECKPOINT = "google/efficientnet-b7"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"
EFFICIENTNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/efficientnet-b7",
]

# EfficientNet 模型相关注释
EFFICIENTNET_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`EfficientNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 输入参数注释
EFFICIENTNET_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`AutoImageProcessor.__call__`] for details.
            像素值。可以使用 [`AutoImageProcessor`] 获得像素值。详情请参阅 [`AutoImageProcessor.__call__`]。

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
            是否返回所有层的隐藏状态。有关更多详细信息，请参阅返回张量中的 `hidden_states`。

        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""

def round_filters(config: EfficientNetConfig, num_channels: int):
    r"""
    根据深度乘数对滤波器数量进行舍入。
    """
    divisor = config.depth_divisor  # 获取深度除数
    num_channels *= config.width_coefficient  # 根据宽度系数调整通道数量
    new_dim = max(divisor, int(num_channels + divisor / 2) // divisor * divisor)  # 计算新的通道数量

    # 确保向下舍入不会超过当前通道数量的10%
    if new_dim < 0.9 * num_channels:
        new_dim += divisor

    return int(new_dim)


def correct_pad(kernel_size: Union[int, Tuple], adjust: bool = True):
    r"""
    获取深度卷积的元组填充值的实用函数。

    Args:
        kernel_size (`int` or `tuple`):
            卷积层的核大小。
        adjust (`bool`, *optional*, 默认为 `True`):
            调整填充值以应用到输入的右侧和底部。
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    if adjust:
        return (correct[1] - 1, correct[1], correct[0] - 1, correct[0])  # 调整填充值
    else:
        return (correct[1], correct[1], correct[0], correct[0])


class EfficientNetEmbeddings(nn.Module):
    r"""
    对应原始实现的干细胞模块的模块。
    """

    def __init__(self, config: EfficientNetConfig):
        super().__init__()

        self.out_dim = round_filters(config, 32)  # 获取输出维度
        self.padding = nn.ZeroPad2d(padding=(0, 1, 0, 1))  # 创建一个零填充层
        self.convolution = nn.Conv2d(
            config.num_channels, self.out_dim, kernel_size=3, stride=2, padding="valid", bias=False
        )  # 创建卷积层
        self.batchnorm = nn.BatchNorm2d(self.out_dim, eps=config.batch_norm_eps, momentum=config.batch_norm_momentum)  # 创建批归一化层
        self.activation = ACT2FN[config.hidden_act]  # 获取激活函数

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.padding(pixel_values)  # 进行填充
        features = self.convolution(features)  # 进行卷积
        features = self.batchnorm(features)  # 进行批归一化
        features = self.activation(features)  # 进行激活

        return features


class EfficientNetDepthwiseConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        depth_multiplier=1,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        padding_mode="zeros",
    ):
        out_channels = in_channels * depth_multiplier
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode,
        )


class EfficientNetExpansionLayer(nn.Module):
    r"""
    这对应于原始实现中每个块的扩展阶段。
    """
    # 初始化函数，用于创建一个扩展层
    def __init__(self, config: EfficientNetConfig, in_dim: int, out_dim: int, stride: int):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个 1x1 的卷积层，用于扩展输入通道数
        self.expand_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=1,
            padding="same",  # 使用 "same" 模式填充，保持输出大小不变
            bias=False,  # 不使用偏置项
        )
        # 创建扩展层的批归一化层
        self.expand_bn = nn.BatchNorm2d(num_features=out_dim, eps=config.batch_norm_eps)
        # 创建扩展层的激活函数，根据配置选择
        self.expand_act = ACT2FN[config.hidden_act]

    # 前向传播函数，用于将输入张量进行扩展处理
    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        # Expand phase
        # 通过扩展卷积层进行通道数扩展
        hidden_states = self.expand_conv(hidden_states)
        # 批归一化处理
        hidden_states = self.expand_bn(hidden_states)
        # 激活函数处理
        hidden_states = self.expand_act(hidden_states)

        return hidden_states
# 定义 EfficientNet 模型的深度卷积层
class EfficientNetDepthwiseLayer(nn.Module):
    r"""
    This corresponds to the depthwise convolution phase of each block in the original implementation.
    """

    def __init__(
        self,
        config: EfficientNetConfig,
        in_dim: int,
        stride: int,
        kernel_size: int,
        adjust_padding: bool,
    ):
        super().__init__()
        self.stride = stride
        # 根据步长决定是否为valid或者same padding
        conv_pad = "valid" if self.stride == 2 else "same"
        # 根据kernel_size和adjust_padding计算正确的padding值
        padding = correct_pad(kernel_size, adjust=adjust_padding)

        # 深度卷积层的零填充层
        self.depthwise_conv_pad = nn.ZeroPad2d(padding=padding)
        # EfficientNetDepthwiseConv2d 模型的深度卷积层
        self.depthwise_conv = EfficientNetDepthwiseConv2d(
            in_dim, kernel_size=kernel_size, stride=stride, padding=conv_pad, bias=False
        )
        # 深度卷积层的批归一化层
        self.depthwise_norm = nn.BatchNorm2d(
            num_features=in_dim, eps=config.batch_norm_eps, momentum=config.batch_norm_momentum
        )
        # 深度卷积层的激活函数
        self.depthwise_act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        # 深度卷积
        if self.stride == 2:
            hidden_states = self.depthwise_conv_pad(hidden_states)

        hidden_states = self.depthwise_conv(hidden_states)
        hidden_states = self.depthwise_norm(hidden_states)
        hidden_states = self.depthwise_act(hidden_states)

        return hidden_states


# 定义 EfficientNet 模型的Squeeze and Excite层
class EfficientNetSqueezeExciteLayer(nn.Module):
    r"""
    This corresponds to the Squeeze and Excitement phase of each block in the original implementation.
    """

    def __init__(self, config: EfficientNetConfig, in_dim: int, expand_dim: int, expand: bool = False):
        super().__init__()
        self.dim = expand_dim if expand else in_dim
        self.dim_se = max(1, int(in_dim * config.squeeze_expansion_ratio))

        # Squeeze层，做自适应平均池化
        self.squeeze = nn.AdaptiveAvgPool2d(output_size=1)
        # Reduce层，做1x1卷积降维
        self.reduce = nn.Conv2d(
            in_channels=self.dim,
            out_channels=self.dim_se,
            kernel_size=1,
            padding="same",
        )
        # Expand层，做1x1卷积升维
        self.expand = nn.Conv2d(
            in_channels=self.dim_se,
            out_channels=self.dim,
            kernel_size=1,
            padding="same",
        )
        # Reduce层的激活函数
        self.act_reduce = ACT2FN[config.hidden_act]
        # Expand层的激活函数
        self.act_expand = nn.Sigmoid()

    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        # 保存输入
        inputs = hidden_states
        hidden_states = self.squeeze(hidden_states)
        hidden_states = self.reduce(hidden_states)
        hidden_states = self.act_reduce(hidden_states)

        hidden_states = self.expand(hidden_states)
        hidden_states = self.act_expand(hidden_states)
        # 对输入进行Squeeze and Excite
        hidden_states = torch.mul(inputs, hidden_states)

        return hidden_states


# 定义 EfficientNet 模型的最终阶段
class EfficientNetFinalBlockLayer(nn.Module):
    r"""
    This corresponds to the final phase of each block in the original implementation.
    """
    # 初始化方法，接受一系列参数来配置 EfficientNet 模块
    def __init__(
        self, config: EfficientNetConfig, in_dim: int, out_dim: int, stride: int, drop_rate: float, id_skip: bool
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 判断是否应用 dropout，条件是步长为 1 且不存在跳跃连接
        self.apply_dropout = stride == 1 and not id_skip
        # 定义用于投影的卷积层，将输入特征图维度转换为输出维度
        self.project_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=1,
            padding="same",  # 使用 "same" padding 保持特征图大小不变
            bias=False,  # 不使用偏置项
        )
        # 定义批归一化层，用于规范化投影后的特征图
        self.project_bn = nn.BatchNorm2d(
            num_features=out_dim,  # 输入通道数等于投影后的输出维度
            eps=config.batch_norm_eps,  # 批归一化的 epsilon 参数
            momentum=config.batch_norm_momentum  # 批归一化的动量参数
        )
        # 定义 dropout 层，用于在训练过程中随机丢弃部分特征
        self.dropout = nn.Dropout(p=drop_rate)

    # 前向传播方法，接受嵌入向量和隐藏状态作为输入，返回经过处理的隐藏状态
    def forward(self, embeddings: torch.FloatTensor, hidden_states: torch.FloatTensor) -> torch.Tensor:
        # 使用投影卷积层对隐藏状态进行特征映射
        hidden_states = self.project_conv(hidden_states)
        # 对投影后的特征图进行批归一化处理
        hidden_states = self.project_bn(hidden_states)

        # 如果应用 dropout，则对特征图进行 dropout 处理，并添加嵌入向量
        if self.apply_dropout:
            hidden_states = self.dropout(hidden_states)
            hidden_states = hidden_states + embeddings

        # 返回处理后的隐藏状态
        return hidden_states
```  
class EfficientNetBlock(nn.Module):
    r"""
    This corresponds to the expansion and depthwise convolution phase of each block in the original implementation.

    Args:
        config ([`EfficientNetConfig`]):
            Model configuration class.
        in_dim (`int`):
            Number of input channels.
        out_dim (`int`):
            Number of output channels.
        stride (`int`):
            Stride size to be used in convolution layers.
        expand_ratio (`int`):
            Expand ratio to set the output dimensions for the expansion and squeeze-excite layers.
        kernel_size (`int`):
            Kernel size for the depthwise convolution layer.
        drop_rate (`float`):
            Dropout rate to be used in the final phase of each block.
        id_skip (`bool`):
            Whether to apply dropout and sum the final hidden states with the input embeddings during the final phase
            of each block. Set to `True` for the first block of each stage.
        adjust_padding (`bool`):
            Whether to apply padding to only right and bottom side of the input kernel before the depthwise convolution
            operation, set to `True` for inputs with odd input sizes.
    """

    def __init__(
        self,
        config: EfficientNetConfig,
        in_dim: int,
        out_dim: int,
        stride: int,
        expand_ratio: int,
        kernel_size: int,
        drop_rate: float,
        id_skip: bool,
        adjust_padding: bool,
    ):
        super().__init__()
        # 设置扩展比例
        self.expand_ratio = expand_ratio
        # 如果扩展比例不为1，设置expand为True，否则为False
        self.expand = True if self.expand_ratio != 1 else False
        # 根据扩展比例计算扩展后的输入维度
        expand_in_dim = in_dim * expand_ratio

        # 如果需要扩展，则创建扩展层
        if self.expand:
            self.expansion = EfficientNetExpansionLayer(
                config=config, in_dim=in_dim, out_dim=expand_in_dim, stride=stride
            )

        # 创建深度可分离卷积层
        self.depthwise_conv = EfficientNetDepthwiseLayer(
            config=config,
            in_dim=expand_in_dim if self.expand else in_dim,
            stride=stride,
            kernel_size=kernel_size,
            adjust_padding=adjust_padding,
        )
        # 创建Squeeze-Excite层
        self.squeeze_excite = EfficientNetSqueezeExciteLayer(
            config=config, in_dim=in_dim, expand_dim=expand_in_dim, expand=self.expand
        )
        # 创建最终块层
        self.projection = EfficientNetFinalBlockLayer(
            config=config,
            in_dim=expand_in_dim if self.expand else in_dim,
            out_dim=out_dim,
            stride=stride,
            drop_rate=drop_rate,
            id_skip=id_skip,
        )
```py  
    # 前向传播函数，接收隐藏状态张量作为输入，并返回处理后的张量
    def forward(self, hidden_states: torch.FloatTensor) -> torch.Tensor:
        embeddings = hidden_states
        # 如果扩展比例不为1，执行扩展和深度卷积阶段
        if self.expand_ratio != 1:
            hidden_states = self.expansion(hidden_states)
        # 执行深度可分离卷积阶段
        hidden_states = self.depthwise_conv(hidden_states)

        # 执行压缩激励阶段
        hidden_states = self.squeeze_excite(hidden_states)
        # 执行投影阶段，将原始隐藏状态和处理后的隐藏状态进行投影
        hidden_states = self.projection(embeddings, hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states
# 定义 EfficientNetEncoder 类，用于前向传播嵌入到每个 EfficientNet 块中
class EfficientNetEncoder(nn.Module):
    r"""
    Forward propogates the embeddings through each EfficientNet block.

    Args:
        config ([`EfficientNetConfig`]):
            Model configuration class.
    """

    # 初始化方法，接受配置参数
    def __init__(self, config: EfficientNetConfig):
        # 调用父类初始化方法
        super().__init__()
        # 保存配置信息
        self.config = config
        # 获取深度系数
        self.depth_coefficient = config.depth_coefficient

        # 定义函数，根据深度乘数来对块重复次数进行四舍五入
        def round_repeats(repeats):
            # 围绕深度乘数对块重复次数进行四舍五入
            return int(math.ceil(self.depth_coefficient * repeats))

        # 获取基本块的数量
        num_base_blocks = len(config.in_channels)
        # 计算总块数，包括重复块
        num_blocks = sum(round_repeats(n) for n in config.num_block_repeats)

        # 当前块的编号
        curr_block_num = 0
        # 存储块的列表
        blocks = []
        # 循环基本块的数量
        for i in range(num_base_blocks):
            # 获取输入维度
            in_dim = round_filters(config, config.in_channels[i])
            # 获取输出维度
            out_dim = round_filters(config, config.out_channels[i])
            # 获取步幅
            stride = config.strides[i]
            # 获取卷积核大小
            kernel_size = config.kernel_sizes[i]
            # 获取膨胀比率
            expand_ratio = config.expand_ratios[i]

            # 根据块重复次数循环
            for j in range(round_repeats(config.num_block_repeats[i])):
                # 如果是第一个块，则进行跳跃连接
                id_skip = True if j == 0 else False
                # 如果不是第一个块，步幅为 1，否则为之前的步幅
                stride = 1 if j > 0 else stride
                # 如果不是第一个块，输入维度为输出维度，否则为之前的输入维度
                in_dim = out_dim if j > 0 else in_dim
                # 根据当前块编号来判断是否进行调整填充
                adjust_padding = False if curr_block_num in config.depthwise_padding else True
                # 计算丢弃连接率
                drop_rate = config.drop_connect_rate * curr_block_num / num_blocks

                # 创建 EfficientNet 块
                block = EfficientNetBlock(
                    config=config,
                    in_dim=in_dim,
                    out_dim=out_dim,
                    stride=stride,
                    kernel_size=kernel_size,
                    expand_ratio=expand_ratio,
                    drop_rate=drop_rate,
                    id_skip=id_skip,
                    adjust_padding=adjust_padding,
                )
                # 将块添加到列表中
                blocks.append(block)
                # 更新当前块编号
                curr_block_num += 1

        # 将块列表转换为模块列表
        self.blocks = nn.ModuleList(blocks)
        # 定义顶部卷积层
        self.top_conv = nn.Conv2d(
            in_channels=out_dim,
            out_channels=round_filters(config, 1280),
            kernel_size=1,
            padding="same",
            bias=False,
        )
        # 定义顶部批归一化层
        self.top_bn = nn.BatchNorm2d(
            num_features=config.hidden_dim, eps=config.batch_norm_eps, momentum=config.batch_norm_momentum
        )
        # 定义顶部激活函数
        self.top_activation = ACT2FN[config.hidden_act]

    # 前向传播方法
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        # 如果需要输出所有隐藏状态，则初始化一个元组用于存储所有隐藏状态
        all_hidden_states = (hidden_states,) if output_hidden_states else None

        # 遍历神经网络的所有块，并依次对隐藏状态进行处理
        for block in self.blocks:
            hidden_states = block(hidden_states)
            # 如果需要输出所有隐藏状态，则将当前隐藏状态添加到all_hidden_states中
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

        # 将隐藏状态经过顶部的卷积层、批归一化层和激活函数进行处理
        hidden_states = self.top_conv(hidden_states)
        hidden_states = self.top_bn(hidden_states)
        hidden_states = self.top_activation(hidden_states)

        # 如果不需要返回字典形式的输出，则将所有处理过的隐藏状态进行返回
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        # 返回一个包含最终隐藏状态和所有隐藏状态的 BaseModelOutputWithNoAttention 对象
        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )
```  
class EfficientNetPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = EfficientNetConfig  # 配置类为 EfficientNetConfig
    base_model_prefix = "efficientnet"  # 基础模型前缀为 "efficientnet"
    main_input_name = "pixel_values"  # 主输入名称为 "pixel_values"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


@add_start_docstrings(
    "The bare EfficientNet model outputting raw features without any specific head on top.",
    EFFICIENTNET_START_DOCSTRING,
)
class EfficientNetModel(EfficientNetPreTrainedModel):
    def __init__(self, config: EfficientNetConfig):
        super().__init__(config)  # 调用父类构造函数初始化模型
        self.config = config  # 保存配置
        self.embeddings = EfficientNetEmbeddings(config)  # 创建 EfficientNetEmbeddings 对象
        self.encoder = EfficientNetEncoder(config)  # 创建 EfficientNetEncoder 对象

        # Final pooling layer
        if config.pooling_type == "mean":  # 如果池化类型为均值
            self.pooler = nn.AvgPool2d(config.hidden_dim, ceil_mode=True)  # 创建平均池化层
        elif config.pooling_type == "max":  # 如果池化类型为最大
            self.pooler = nn.MaxPool2d(config.hidden_dim, ceil_mode=True)  # 创建最大池化层
        else:
            raise ValueError(f"config.pooling must be one of ['mean', 'max'] got {config.pooling}")  # 抛出异常

        # Initialize weights and apply final processing
        self.post_init()  # 调用后处理函数

    @add_start_docstrings_to_model_forward(EFFICIENTNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 定义一个方法，参数类型为 pixel_values: Optional[torch.FloatTensor], output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None，返回值类型为 Union[Tuple, BaseModelOutputWithPoolingAndNoAttention]
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 如果 output_hidden_states 为空，则使用配置中的 output_hidden_states
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果 return_dict 为空，则使用配置中的 use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果 pixel_values 为空，则抛出 ValueError 异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 将输入的像素值通过嵌入层得到嵌入输出
        embedding_output = self.embeddings(pixel_values)

        # 将嵌入输出传入编码器模型，得到编码器的输出，同时可以选择是否返回隐藏层状态
        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # 应用池化操作，得到池化后的输出
        last_hidden_state = encoder_outputs[0]
        pooled_output = self.pooler(last_hidden_state)
        
        # 重塑池化输出的形状，将 (batch_size, 1280, 1, 1) 转换为 (batch_size, 1280)
        pooled_output = pooled_output.reshape(pooled_output.shape[:2])

        # 如果不要求以字典的形式返回结果，则返回元组
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        # 否则以 BaseModelOutputWithPoolingAndNoAttention 类型的对象返回结果
        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )
# 创建EfficientNetForImageClassification类，该类在EfficientNetPreTrainedModel基础上增加了一个图像分类头部（即池化特征的线性层），例如用于ImageNet数据集的图像分类
@add_start_docstrings(
    """
    EfficientNet Model with an image classification head on top (a linear layer on top of the pooled features), e.g.
    for ImageNet.
    """,
    EFFICIENTNET_START_DOCSTRING,
)
class EfficientNetForImageClassification(EfficientNetPreTrainedModel):
    # 初始化EfficientNetForImageClassification类
    def __init__(self, config):
        super().__init__(config)
        # 设置类的标签数量
        self.num_labels = config.num_labels
        self.config = config
        # 创建EfficientNetModel对象
        self.efficientnet = EfficientNetModel(config)
        # 分类器头部
        self.dropout = nn.Dropout(p=config.dropout_rate)
        self.classifier = nn.Linear(config.hidden_dim, self.num_labels) if self.num_labels > 0 else nn.Identity()

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数
    @add_start_docstrings_to_model_forward(EFFICIENTNET_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 根据需要设置是否返回字典，默认为使用配置中的返回字典的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用efficientnet模型处理像素值得到输出
        outputs = self.efficientnet(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        # 如果return_dict为True，则使用pooler_output作为输出的特征向量，否则使用outputs[1]作为输出的特征向量
        pooled_output = outputs.pooler_output if return_dict else outputs[1]
        # 对输出的特征向量进行dropout操作
        pooled_output = self.dropout(pooled_output)
        # 通过分类器得到logits
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            # 如果配置问题类型为None，则根据num_labels的值设置问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型计算损失函数
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

        # 如果return_dict为False，则返回输出的logits和其他的hidden_states
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果return_dict为True，则返回ImageClassifierOutputWithNoAttention对象
        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )
```