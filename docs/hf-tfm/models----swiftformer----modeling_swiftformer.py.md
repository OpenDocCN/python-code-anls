# `.\transformers\models\swiftformer\modeling_swiftformer.py`

```py
# 添加编码信息
# 2023年 MBZUAI 和 The HuggingFace Inc. 团队 版权所有。

# 根据 Apache 许可证 2.0 进行许可，除非符合许可证的条款，否则您不得使用此文件。
# 您可以获得许可证的一份副本，网址为
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件都是基于“按原样”分发的，不附带任何明示或暗示的保证或条件。
# 请参阅许可证以获取特定语言管理权限和限制

# PyTorch SwiftFormer 模型。
import collections.abc
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2CLS
from ...modeling_outputs import (
    BaseModelOutputWithNoAttention,
    ImageClassifierOutputWithNoAttention,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from .configuration_swiftformer import SwiftFormerConfig

# 获取 logger 对象
logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "SwiftFormerConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "MBZUAI/swiftformer-xs"
_EXPECTED_OUTPUT_SHAPE = [1, 220, 7, 7]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "MBZUAI/swiftformer-xs"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

# 预训练模型列表
SWIFTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "MBZUAI/swiftformer-xs",
    # 查看所有的 SwiftFormer 模型 https://huggingface.co/models?filter=swiftformer
]

class SwiftFormerPatchEmbedding(nn.Module):
    """
    Patch Embedding Layer 由两个 2D 卷积层构成。

    输入: 形状为`[batch_size, in_channels, height, width]`的张量

    输出: 形状为`[batch_size, out_channels, height/4, width/4]`的张量
    """

    def __init__(self, config: SwiftFormerConfig):
        super().__init__()

        in_chs = config.num_channels
        out_chs = config.embed_dims[0]
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_chs, out_chs // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_chs // 2, eps=config.batch_norm_eps),
            nn.ReLU(),
            nn.Conv2d(out_chs // 2, out_chs, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_chs, eps=config.batch_norm_eps),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.patch_embedding(x)


# 从 transformers.models.beit.modeling_beit.drop_path 中复制函数
# 删除路径(随机深度)每个样本(应用于残差块的主路径时)
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """

    """
    # 如果dropout概率为0，或者不处于训练状态，则直接返回输入
    if drop_prob == 0.0 or not training:
        return input
    # 计算保留概率
    keep_prob = 1 - drop_prob
    # 计算随机张量的形状
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # 适用于不同维度的张量，而不仅仅是2D卷积神经网络
    # 生成随机张量
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    # 将随机张量进行二值化处理
    random_tensor.floor_()
    # 计算输出，采用DropConnect算法
    output = input.div(keep_prob) * random_tensor
    # 返回输出
    return output
# 从 transformers.models.beit.modeling_beit.BeitDropPath 复制代码，并将 Beit->Swiftformer
class SwiftFormerDropPath(nn.Module):
    """对每个样本进行路径丢弃（随机深度），应用于残差块的主路径。"""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class SwiftFormerEmbeddings(nn.Module):
    """
    由单个2D卷积和批量归一化层组成的嵌入层。

    输入: 形状为`[batch_size, channels, height, width]`的张量

    输出: 形状为`[batch_size, channels, height/stride, width/stride]`的张量
    """

    def __init__(self, config: SwiftFormerConfig, index: int):
        super().__init__()

        # 从配置中获取参数
        patch_size = config.down_patch_size
        stride = config.down_stride
        padding = config.down_pad
        embed_dims = config.embed_dims

        in_chans = embed_dims[index]
        embed_dim = embed_dims[index + 1]

        # 确保参数为可迭代对象
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        stride = stride if isinstance(stride, collections.abc.Iterable) else (stride, stride)
        padding = padding if isinstance(padding, collections.abc.Iterable) else (padding, padding)

        # 创建卷积和归一化层
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(embed_dim, eps=config.batch_norm_eps)

    def forward(self, x):
        # 进行前向传播
        x = self.proj(x)
        x = self.norm(x)
        return x


class SwiftFormerConvEncoder(nn.Module):
    """
    `SwiftFormerConvEncoder`，使用3*3和1*1卷积。

    输入: 形状为`[batch_size, channels, height, width]`的张量

    输出: 形状为`[batch_size, channels, height, width]`的张量
    """

    def __init__(self, config: SwiftFormerConfig, dim: int):
        super().__init__()
        hidden_dim = int(config.mlp_ratio * dim)

        # 创建卷积、归一化、激活和参数层
        self.depth_wise_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = nn.BatchNorm2d(dim, eps=config.batch_norm_eps)
        self.point_wise_conv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.point_wise_conv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.drop_path = nn.Identity()
        self.layer_scale = nn.Parameter(torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)

    def forward(self, x):
        # 执行前向传播
        input = x
        x = self.depth_wise_conv(x)
        x = self.norm(x)
        x = self.point_wise_conv1(x)
        x = self.act(x)
        x = self.point_wise_conv2(x)
        x = input + self.drop_path(self.layer_scale * x)
        return x


class SwiftFormerMlp(nn.Module):
    """
    # 实现包含 1*1 卷积的 MLP 层
    class MLP(nn.Module):
    
        # 初始化函数
        def __init__(self, config: SwiftFormerConfig, in_features: int):
            super().__init__()
            # 计算隐藏层特征数
            hidden_features = int(in_features * config.mlp_ratio)
            # 归一化层
            self.norm1 = nn.BatchNorm2d(in_features, eps=config.batch_norm_eps)
            # 第一个 1*1 卷积层
            self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
            # 激活函数，根据配置选择不同的激活函数
            act_layer = ACT2CLS[config.hidden_act]
            self.act = act_layer()
            # 第二个 1*1 卷积层
            self.fc2 = nn.Conv2d(hidden_features, in_features, 1)
            # dropout 层
            self.drop = nn.Dropout(p=0.0)
    
        # 前向传播函数
        def forward(self, x):
            # 归一化
            x = self.norm1(x)
            # 第一个 1*1 卷积
            x = self.fc1(x)
            # 激活函数
            x = self.act(x)
            # dropout
            x = self.drop(x)
            # 第二个 1*1 卷积
            x = self.fc2(x)
            # dropout
            x = self.drop(x)
            # 返回结果
            return x
# 定义了 SwiftFormerEfficientAdditiveAttention 类，用于 SwiftFormer 模型中的高效加性注意力机制
class SwiftFormerEfficientAdditiveAttention(nn.Module):
    """
    Efficient Additive Attention module for SwiftFormer.

    Input: tensor of shape `[batch_size, channels, height, width]`

    Output: tensor of shape `[batch_size, channels, height, width]`
    """

    # 初始化函数，接收 SwiftFormerConfig 类型的 config 对象和整数 dim
    def __init__(self, config: SwiftFormerConfig, dim: int = 512):
        super().__init__()

        # 使用线性变换将维度为 dim 的输入映射到维度为 dim 的输出
        self.to_query = nn.Linear(dim, dim)
        self.to_key = nn.Linear(dim, dim)

        # 使用参数初始化权重矩阵 w_g
        self.w_g = nn.Parameter(torch.randn(dim, 1))
        self.scale_factor = dim**-0.5  # 设置缩放因子为维度的倒数开平方
        self.proj = nn.Linear(dim, dim)  # 对输入进行线性变换
        self.final = nn.Linear(dim, dim)  # 对输入进行最终的线性变换

    # 前向传播函数
    def forward(self, x):
        query = self.to_query(x)  # 执行查询的线性变换
        key = self.to_key(x)  # 执行键的线性变换

        query = torch.nn.functional.normalize(query, dim=-1)  # 对查询进行标准化
        key = torch.nn.functional.normalize(key, dim=-1)  # 对键进行标准化

        query_weight = query @ self.w_g  # 计算查询的权重
        scaled_query_weight = query_weight * self.scale_factor  # 缩放查询的权重
        scaled_query_weight = scaled_query_weight.softmax(dim=-1)  # 对缩放后的查询权重进行 softmax 操作

        global_queries = torch.sum(scaled_query_weight * query, dim=1)  # 计算全局查询向量
        global_queries = global_queries.unsqueeze(1).repeat(1, key.shape[1], 1)  # 将全局查询向量扩展成与键相同的维度

        out = self.proj(global_queries * key) + query  # 计算输出
        out = self.final(out)  # 使用最终的线性变换进行处理

        return out  # 返回输出张量


# 定义了 SwiftFormerLocalRepresentation 类，用于 SwiftFormer 模型中的局部表示模块
class SwiftFormerLocalRepresentation(nn.Module):
    """
    Local Representation module for SwiftFormer that is implemented by 3*3 depth-wise and point-wise convolutions.

    Input: tensor of shape `[batch_size, channels, height, width]`

    Output: tensor of shape `[batch_size, channels, height, width]`
    """

    def __init__(self, config: SwiftFormerConfig, dim: int):
        super().__init__()

        # 定义深度可分离卷积层、批归一化层和两个点卷积层
        self.depth_wise_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = nn.BatchNorm2d(dim, eps=config.batch_norm_eps)
        self.point_wise_conv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.GELU()  # 使用 GELU 作为激活函数
        self.point_wise_conv2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.drop_path = nn.Identity()
        self.layer_scale = nn.Parameter(torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)

    # 前向传播函数
    def forward(self, x):
        input = x  # 保存输入张量
        x = self.depth_wise_conv(x)  # 执行深度可分离卷积
        x = self.norm(x)  # 对卷积结果进行批归一化
        x = self.point_wise_conv1(x)  # 执行第一个点卷积
        x = self.act(x)  # 使用 GELU 激活函数
        x = self.point_wise_conv2(x)  # 执行第二个点卷积
        x = input + self.drop_path(self.layer_scale * x)  # 添加残差连接并执行 drop path
        return x  # 返回输出张量


# 定义了 SwiftFormerEncoderBlock 类，用于 SwiftFormer 模型中的编码器块
class SwiftFormerEncoderBlock(nn.Module):
    """
    SwiftFormer Encoder Block for SwiftFormer. It consists of (1) Local representation module, (2)
    SwiftFormerEfficientAdditiveAttention, and (3) MLP block.

    Input: tensor of shape `[batch_size, channels, height, width]`

    Output: tensor of shape `[batch_size, channels,height, width]`
    """
    # 初始化函数，接受配置、维度和丢弃路径作为参数
    def __init__(self, config: SwiftFormerConfig, dim: int, drop_path: float = 0.0) -> None:
        # 调用父类的初始化函数
        super().__init__()

        # 从配置中获取层尺度初始化值和是否使用层尺度
        layer_scale_init_value = config.layer_scale_init_value
        use_layer_scale = config.use_layer_scale

        # 创建局部表示、注意力和线性变换层对象
        self.local_representation = SwiftFormerLocalRepresentation(config, dim=dim)
        self.attn = SwiftFormerEfficientAdditiveAttention(config, dim=dim)
        self.linear = SwiftFormerMlp(config, in_features=dim)

        # 创建丢弃路径对象，如果丢弃率大于 0.0，否则创建身份变换
        self.drop_path = SwiftFormerDropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # 设置是否使用层尺度
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            # 如果使用层尺度，则创建两个可训练的层尺度参数
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True
            )
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True
            )

    # 前向传播函数，接受输入张量 x 作为参数
    def forward(self, x):
        # 将输入张量通过局部表示层
        x = self.local_representation(x)
        # 获取张量的形状信息
        batch_size, channels, height, width = x.shape
        if self.use_layer_scale:
            # 如果使用层尺度，则对局部表示进行层尺度调整
            x = x + self.drop_path(
                self.layer_scale_1
                * self.attn(x.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels))
                .reshape(batch_size, height, width, channels)
                .permute(0, 3, 1, 2)
            )
            # 对局部表示进行线性变换和层尺度调整
            x = x + self.drop_path(self.layer_scale_2 * self.linear(x))

        else:
            # 如果不使用层尺度，则直接对局部表示进行注意力和线性变换
            x = x + self.drop_path(
                self.attn(x.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels))
                .reshape(batch_size, height, width, channels)
                .permute(0, 3, 1, 2)
            )
            x = x + self.drop_path(self.linear(x))
        # 返回处理后的张量
        return x
class SwiftFormerStage(nn.Module):
    """
    A Swiftformer stage consisting of a series of `SwiftFormerConvEncoder` blocks and a final
    `SwiftFormerEncoderBlock`.

    Input: tensor in shape `[batch_size, channels, height, width]`

    Output: tensor in shape `[batch_size, channels, height, width]`
    """

    def __init__(self, config: SwiftFormerConfig, index: int) -> None:
        super().__init__()

        layer_depths = config.depths  # 获取配置中的层深度列表
        dim = config.embed_dims[index]  # 获取配置中指定索引处的嵌入维度
        depth = layer_depths[index]  # 获取当前层的深度

        blocks = []  # 初始化模块列表
        for block_idx in range(depth):  # 遍历当前层的每个块
            block_dpr = config.drop_path_rate * (block_idx + sum(layer_depths[:index])) / (sum(layer_depths) - 1)
            # 计算当前块的 drop path rate

            if depth - block_idx <= 1:  # 如果是当前层的最后一个块
                blocks.append(SwiftFormerEncoderBlock(config, dim=dim, drop_path=block_dpr))
                # 添加一个 SwiftFormerEncoderBlock 到模块列表
            else:
                blocks.append(SwiftFormerConvEncoder(config, dim=dim))
                # 否则添加一个 SwiftFormerConvEncoder 到模块列表

        self.blocks = nn.ModuleList(blocks)  # 将模块列表转换为模块列表

    def forward(self, input):
        for block in self.blocks:  # 遍历模块列表中的每个块
            input = block(input)  # 将输入传递给当前块，并更新输入
        return input


class SwiftFormerEncoder(nn.Module):
    def __init__(self, config: SwiftFormerConfig) -> None:
        super().__init__()
        self.config = config  # 存储配置信息

        embed_dims = config.embed_dims  # 获取嵌入维度列表
        downsamples = config.downsamples  # 获取下采样标志列表
        layer_depths = config.depths  # 获取层深度列表

        # Transformer model
        network = []  # 初始化网络模块列表
        for i in range(len(layer_depths)):  # 遍历每个层
            stage = SwiftFormerStage(config=config, index=i)  # 创建一个 SwiftFormerStage
            network.append(stage)  # 将 SwiftFormerStage 添加到网络模块列表
            if i >= len(layer_depths) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                # 如果需要进行下采样或者嵌入维度发生变化
                network.append(SwiftFormerEmbeddings(config, index=i))
                # 添加 SwiftFormerEmbeddings 到网络模块列表
        self.network = nn.ModuleList(network)  # 将网络模块列表转换为模块列表

        self.gradient_checkpointing = False  # 初始化梯度检查点标志

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, BaseModelOutputWithNoAttention]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )  # 获取输出隐藏状态标志
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 获取返回字典标志

        all_hidden_states = (hidden_states,) if output_hidden_states else None
        # 如果需要输出隐藏状态，则初始化全部隐藏状态列表，否则设为 None

        for block in self.network:  # 遍历网络模块列表中的每个模块
            hidden_states = block(hidden_states)  # 传递隐藏状态到当前模块，并更新隐藏状态
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                # 如果需要输出隐藏状态，则将当前隐藏状态添加到全部隐藏状态列表中

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)
            # 如果不需要返回字典，则返回隐藏状态和全部隐藏状态的元组

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )  # 否则返回 Base Model Output Without Attention
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    
    # 定义一个抽象类，用于处理权重初始化以及一个简单的接口用于下载和加载预训练模型
    config_class = SwiftFormerConfig
    # 基础模型前缀
    base_model_prefix = "swiftformer"
    # 主要输入名称
    main_input_name = "pixel_values"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        # 初始化权重
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # 对于线性层和卷积层，使用截断正态分布进行权重初始化
            nn.init.trunc_normal_(module.weight, std=0.02)
            # 如果存在偏置，则将其初始化为常数0
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LayerNorm)):
            # 对于 LayerNorm 层，将偏置初始化为常数0，将权重初始化为常数1
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
SWIFTFORMER_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`SwiftFormerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

SWIFTFORMER_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`ViTImageProcessor.__call__`]
            for details.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# Define a class decorator that adds docstrings to the SwiftFormerModel class
@add_start_docstrings(
    "The bare SwiftFormer Model transformer outputting raw hidden-states without any specific head on top.",
    SWIFTFORMER_START_DOCSTRING,
)
class SwiftFormerModel(SwiftFormerPreTrainedModel):
    # Define the initialization method for the SwiftFormerModel class
    def __init__(self, config: SwiftFormerConfig):
        super().__init__(config)
        # Assign the model configuration
        self.config = config

        # Initialize the patch embedding layer
        self.patch_embed = SwiftFormerPatchEmbedding(config)
        # Initialize the encoder layer
        self.encoder = SwiftFormerEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    # Define the forward method for the SwiftFormerModel class
    @add_start_docstrings_to_model_forward(SWIFTFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[Tuple, BaseModelOutputWithNoAttention]:
        r""" """
        # 初始化函数，声明返回类型为元组或BaseModelOutputWithNoAttention

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 将output_hidden_states设置为传入参数值或者self.config.output_hidden_states的值

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 将return_dict设置为传入参数值或者self.config.use_return_dict的值

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        # 如果pixel_values为None，则引发数值错误异常

        embedding_output = self.patch_embed(pixel_values)
        # 使用self.patch_embed对pixel_values进行嵌入表示

        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 使用self.encoder对嵌入表示进行编码

        if not return_dict:
            return tuple(v for v in encoder_outputs if v is not None)
        # 如果return_dict为False，则返回encoder_outputs中非空的元素组成的元组

        return BaseModelOutputWithNoAttention(
            last_hidden_state=encoder_outputs.last_hidden_state,
            hidden_states=encoder_outputs.hidden_states,
        )
        # 返回BaseModelOutputWithNoAttention对象，指定最后隐藏状态和隐藏状态
# 导入必要的 docstrings 和预训练模型
@add_start_docstrings(
    """
    SwiftFormer Model transformer with an image classification head on top (e.g. for ImageNet).
    """,
    SWIFTFORMER_START_DOCSTRING,
)
# 定义 SwiftFormerForImageClassification 类，继承自 SwiftFormerPreTrainedModel
class SwiftFormerForImageClassification(SwiftFormerPreTrainedModel):
    def __init__(self, config: SwiftFormerConfig) -> None:
        # 调用父类的构造函数
        super().__init__(config)
        # 获取配置中的 embed_dims 参数
        embed_dims = config.embed_dims
        # 获取配置中的 num_labels 参数
        self.num_labels = config.num_labels
        # 创建 SwiftFormerModel 实例
        self.swiftformer = SwiftFormerModel(config)
        # 创建 BatchNorm2d 层，用于特征归一化
        self.norm = nn.BatchNorm2d(embed_dims[-1], eps=config.batch_norm_eps)
        # 如果 num_labels 大于 0，创建线性分类层；否则创建 Identity 层
        self.head = nn.Linear(embed_dims[-1], self.num_labels) if self.num_labels > 0 else nn.Identity()
        # 同上，创建分布式分类层
        self.dist_head = nn.Linear(embed_dims[-1], self.num_labels) if self.num_labels > 0 else nn.Identity()
        # 执行最终的参数初始化操作
        self.post_init()

    # 定义前向传播方法
    @add_start_docstrings_to_model_forward(SWIFTFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        ) -> Union[tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 检查是否应该返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 运行基础模型
        outputs = self.swiftformer(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取序列输出
        sequence_output = outputs.last_hidden_state if return_dict else outputs[0]

        # 运行分类头
        sequence_output = self.norm(sequence_output)
        sequence_output = sequence_output.flatten(2).mean(-1)
        cls_out = self.head(sequence_output)
        distillation_out = self.dist_head(sequence_output)
        logits = (cls_out + distillation_out) / 2

        # 计算损失
        loss = None
        if labels is not None:
            # 确定问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

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

        # 如果不返回字典
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回 ImageClassifierOutputWithNoAttention 类型
        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )
```