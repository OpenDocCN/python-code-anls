# `.\models\pvt_v2\modeling_pvt_v2.py`

```
# 设置文件编码为 UTF-8
# 版权声明，包含作者信息和 HuggingFace 公司信息，保留所有权利
# 根据 Apache 许可证 2.0 版本，除非符合许可证要求，否则不得使用此文件
# 可以在以下链接处获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 根据适用法律或书面同意，本软件按“原样”分发，无任何明示或暗示的担保或条件
# 详见许可证，了解特定语言的权利和限制
"""PyTorch PVTv2 模型."""

import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 从 HuggingFace 库导入一些工具和模块
from ...activations import ACT2FN
from ...modeling_outputs import BackboneOutput, BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 从 backbone_utils 模块中导入 BackboneMixin 类
from ...utils.backbone_utils import BackboneMixin
# 导入 PVTv2 配置文件
from .configuration_pvt_v2 import PvtV2Config

# 获取日志记录器
logger = logging.get_logger(__name__)

# 以下是用于文档的常量定义
_CONFIG_FOR_DOC = "PvtV2Config"

_CHECKPOINT_FOR_DOC = "OpenGVLab/pvt_v2_b0"
_EXPECTED_OUTPUT_SHAPE = [1, 256, 7, 7]

_IMAGE_CLASS_CHECKPOINT = "OpenGVLab/pvt_v2_b0"
_IMAGE_CLASS_EXPECTED_OUTPUT = "LABEL_281"  # "tabby, tabby cat" 的 ImageNet ID

# 预训练模型存档列表
PVT_V2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "OpenGVLab/pvt_v2_b0",
    "OpenGVLab/pvt_v2_b1",
    "OpenGVLab/pvt_v2_b2",
    "OpenGVLab/pvt_v2_b2_linear",
    "OpenGVLab/pvt_v2_b3",
    "OpenGVLab/pvt_v2_b4",
    "OpenGVLab/pvt_v2_b5",
    # 更多 PVT 模型请查看 https://huggingface.co/models?filter=pvt_v2
]

# 从 transformers.models.beit.modeling_beit.drop_path 复制的函数
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    每个样本都会丢弃路径（随机深度），主要用于残差块的主路径中。

    Ross Wightman 的评论：这与我为 EfficientNet 等网络创建的 DropConnect 实现相同，
    然而，原始名称具有误导性，因为 'Drop Connect' 是另一篇论文中的不同形式的 dropout……
    参见讨论：https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ……
    我选择将层和参数名称更改为 'drop path'，而不是将 DropConnect 作为层名称混合使用，并使用 'survival rate' 作为参数。
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    # 定义一个形状，确保与输入张量的维度匹配，支持不同维度的张量，而不仅仅是二维卷积神经网络
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)
    # 创建一个随机张量，形状与输入张量相同，元素值在 [keep_prob, keep_prob+1) 之间，数据类型和设备与输入张量一致
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    # 将随机张量向下取整，实现二值化操作
    random_tensor.floor_()
    # 对输入张量进行二值化处理，输出保留的概率为 keep_prob，其余元素置零
    output = input.div(keep_prob) * random_tensor
    # 返回处理后的输出张量
    return output
# 从 transformers.models.convnext.modeling_convnext.ConvNextDropPath 复制代码，并将 ConvNext 改为 Pvt
class PvtV2DropPath(nn.Module):
    """每个样本的随机深度（Drop Path，应用于残差块的主路径）。"""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class PvtV2OverlapPatchEmbeddings(nn.Module):
    """将图像转换为补丁嵌入。"""

    def __init__(self, config: PvtV2Config, layer_idx: int):
        super().__init__()
        patch_size = config.patch_sizes[layer_idx]
        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        stride = config.strides[layer_idx]
        num_channels = config.num_channels if layer_idx == 0 else config.hidden_sizes[layer_idx - 1]
        hidden_size = config.hidden_sizes[layer_idx]
        self.patch_size = patch_size
        # 使用二维卷积将输入的像素值转换为嵌入向量
        self.proj = nn.Conv2d(
            num_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        # 对嵌入向量进行层归一化
        self.layer_norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values):
        # 将像素值映射为嵌入向量
        embeddings = self.proj(pixel_values)
        _, _, height, width = embeddings.shape
        # 将嵌入向量展平以便进行后续处理，并转置维度
        embeddings = embeddings.flatten(2).transpose(1, 2)
        # 应用层归一化
        embeddings = self.layer_norm(embeddings)
        return embeddings, height, width


class PvtV2DepthWiseConv(nn.Module):
    """
    使用零填充的深度卷积（DW convolution），以融入位置信息。
    深度卷积的组数等于输入通道数，即每个输入通道一个滤波器，从而减少参数和计算成本，主要用于位置编码。
    """

    def __init__(self, config: PvtV2Config, dim: int = 768):
        super().__init__()
        # 定义深度卷积层，每个输入通道一个滤波器
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, hidden_states, height, width):
        batch_size, seq_len, num_channels = hidden_states.shape
        # 调整张量形状以适应深度卷积的输入要求
        hidden_states = hidden_states.transpose(1, 2).view(batch_size, num_channels, height, width)
        # 应用深度卷积
        hidden_states = self.dwconv(hidden_states)
        # 将输出展平以便后续处理，并转置维度
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        return hidden_states


class PvtV2SelfAttention(nn.Module):
    """高效的自注意力机制。"""
    # 初始化函数，接受配置对象、隐藏层大小、注意力头数和空间缩减比例作为参数
    def __init__(self, config: PvtV2Config, hidden_size: int, num_attention_heads: int, spatial_reduction_ratio: int):
        super().__init__()  # 调用父类的初始化方法

        # 从配置对象中获取是否使用线性注意力的标志
        self.linear_attention = config.linear_attention
        # 初始化一个空的被修剪的注意力头集合
        self.pruned_heads = set()
        # 设置隐藏层的大小
        self.hidden_size = hidden_size
        # 设置注意力头的数量
        self.num_attention_heads = num_attention_heads

        # 检查隐藏层大小是否能够被注意力头的数量整除
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})"
            )

        # 计算每个注意力头的大小
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        # 计算所有注意力头总共的大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 创建查询、键和值的线性层，用于生成注意力矩阵
        self.query = nn.Linear(self.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(self.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(self.hidden_size, self.all_head_size, bias=config.qkv_bias)
        # 创建注意力概率的dropout层
        self.attn_drop = nn.Dropout(config.attention_probs_dropout_prob)
        # 创建投影层，用于将注意力输出映射回原始隐藏状态的维度
        self.proj = nn.Linear(self.hidden_size, self.hidden_size)
        # 创建投影层的dropout层
        self.proj_drop = nn.Dropout(config.hidden_dropout_prob)

        # 设置空间缩减比例
        self.spatial_reduction_ratio = spatial_reduction_ratio
        # 如果使用线性注意力，初始化自适应平均池化层、空间缩减卷积层、层归一化和GELU激活函数
        if self.linear_attention:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.spatial_reduction = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=1, stride=1)
            self.layer_norm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
            self.act = nn.GELU()
        # 否则，如果空间缩减比例大于1，初始化空间缩减卷积层和层归一化
        elif spatial_reduction_ratio > 1:
            self.spatial_reduction = nn.Conv2d(
                self.hidden_size, self.hidden_size, kernel_size=spatial_reduction_ratio, stride=spatial_reduction_ratio
            )
            self.layer_norm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)

    # 转换隐藏状态张量为注意力分数张量的形状
    def transpose_for_scores(self, hidden_states) -> torch.Tensor:
        new_shape = hidden_states.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        hidden_states = hidden_states.view(new_shape)
        return hidden_states.permute(0, 2, 1, 3)

    # 前向传播函数，接受隐藏状态张量、高度、宽度和是否输出注意力矩阵作为输入
    def forward(
        self,
        hidden_states: torch.Tensor,
        height: int,
        width: int,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor]:
        # 获取输入张量的维度信息：batch_size为批大小，seq_len为序列长度，num_channels为通道数
        batch_size, seq_len, num_channels = hidden_states.shape
        
        # 使用self.query对隐藏状态进行查询操作，为了后续的注意力计算做准备
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        # 如果使用线性注意力
        if self.linear_attention:
            # 将隐藏状态重新排列以便空间池化
            hidden_states = hidden_states.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)
            # 对空间维度进行池化操作，并将结果重新排列以供后续处理
            hidden_states = (
                self.spatial_reduction(self.pool(hidden_states)).reshape(batch_size, num_channels, -1).permute(0, 2, 1)
            )
            # 对处理后的隐藏状态进行激活函数操作
            hidden_states = self.act(self.layer_norm(hidden_states))
        # 如果使用空间缩减比例大于1
        elif self.spatial_reduction_ratio > 1:
            # 将隐藏状态重新排列以便空间池化
            hidden_states = hidden_states.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)
            # 对空间维度进行池化操作，并将结果重新排列以供后续处理
            hidden_states = (
                self.spatial_reduction(hidden_states).reshape(batch_size, num_channels, -1).permute(0, 2, 1)
            )
            # 对处理后的隐藏状态进行LayerNorm操作
            hidden_states = self.layer_norm(hidden_states)

        # 使用self.key对隐藏状态进行键的计算
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        # 使用self.value对隐藏状态进行值的计算
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # 计算注意力分数，通过query与key的点积得到
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # 对注意力分数进行缩放
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 对注意力分数进行softmax操作，得到注意力权重
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 对注意力权重进行dropout操作，用于随机失活整个token以供关注
        attention_probs = self.attn_drop(attention_probs)
        # 计算上下文层，将注意力权重与值进行加权求和，并对结果进行重排列和reshape
        context_layer = (attention_probs @ value_layer).transpose(1, 2).reshape(batch_size, seq_len, num_channels)
        # 对上下文层进行投影操作
        context_layer = self.proj(context_layer)
        # 对投影后的上下文层进行dropout操作
        context_layer = self.proj_drop(context_layer)

        # 如果需要输出注意力权重，则返回包含注意力权重的输出
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs

    def prune_heads(self, heads):
        # 如果要剪枝的头部列表为空，则直接返回
        if len(heads) == 0:
            return
        
        # 查找可剪枝头部并获取相应索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.num_attention_heads, self.attention_head_size, self.pruned_heads
        )

        # 对query、key、value以及proj进行剪枝操作
        self.query = prune_linear_layer(self.query, index)
        self.key = prune_linear_layer(self.key, index)
        self.value = prune_linear_layer(self.value, index)
        self.proj = prune_linear_layer(self.proj, index, dim=1)

        # 更新超参数并存储已剪枝头部
        self.num_attention_heads = self.num_attention_heads - len(heads)
        self.all_head_size = self.attention_head_size * self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)
# 定义了一个私有的PvtV2BlockLayer类，继承自nn.Module类
class PvtV2BlockLayer(nn.Module):
    # 初始化方法，接受配置config、层索引layer_idx和dropout路径drop_path参数
    def __init__(self, config: PvtV2Config, layer_idx: int, drop_path: float = 0.0):
        # 调用父类的初始化方法
        super().__init__()
        
        # 从配置中获取隐藏层大小
        hidden_size: int = config.hidden_sizes[layer_idx]
        # 从配置中获取注意力头数
        num_attention_heads: int = config.num_attention_heads[layer_idx]
        # 从配置中获取空间降维比率
        spatial_reduction_ratio: int = config.sr_ratios[layer_idx]
        # 从配置中获取MLP比率
        mlp_ratio: float = config.mlp_ratios[layer_idx]
        
        # 第一个LayerNorm层，用于规范化隐藏层状态
        self.layer_norm_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        
        # PvtV2SelfAttention自注意力层，接收隐藏层大小、注意力头数和空间降维比率等参数
        self.attention = PvtV2SelfAttention(
            config=config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            spatial_reduction_ratio=spatial_reduction_ratio,
        )
        
        # 根据drop_path是否大于0来决定是否添加PvtV2DropPath或者保持Identity
        self.drop_path = PvtV2DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        # 第二个LayerNorm层，用于规范化隐藏层状态
        self.layer_norm_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        
        # 计算MLP隐藏层大小
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        
        # PvtV2ConvFeedForwardNetwork类，用于定义PVT V2的前馈神经网络，接收配置和隐藏层大小等参数
        self.mlp = PvtV2ConvFeedForwardNetwork(config=config, in_features=hidden_size, hidden_features=mlp_hidden_size)
    # 定义神经网络模型的前向传播函数，接受隐藏状态张量、图像高度、宽度和是否输出注意力矩阵作为参数
    def forward(self, hidden_states: torch.Tensor, height: int, width: int, output_attentions: bool = False):
        # 对隐藏状态进行 Layer Normalization 处理，并传递给注意力机制模块
        self_attention_outputs = self.attention(
            hidden_states=self.layer_norm_1(hidden_states),
            height=height,
            width=width,
            output_attentions=output_attentions,
        )
        # 从注意力机制模块的输出中获取注意力矩阵之外的结果
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        # 对注意力输出进行 Drop Path 处理
        attention_output = self.drop_path(attention_output)
        # 将处理后的注意力输出与原始隐藏状态相加，得到新的隐藏状态
        hidden_states = attention_output + hidden_states

        # 将新的隐藏状态传递给 MLP 模块进行处理，并返回结果
        mlp_output = self.mlp(self.layer_norm_2(hidden_states), height, width)

        # 对 MLP 输出进行 Drop Path 处理
        mlp_output = self.drop_path(mlp_output)
        # 将 MLP 处理后的输出与原始隐藏状态相加，得到当前层的最终输出
        layer_output = hidden_states + mlp_output

        # 将当前层的输出添加到输出元组中，并返回
        outputs = (layer_output,) + outputs

        return outputs
# 定义私有版本2的编码器层，继承自nn.Module类
class PvtV2EncoderLayer(nn.Module):
    def __init__(self, config: PvtV2Config, layer_idx: int):
        super().__init__()
        # 初始化：重叠补丁嵌入层，用于将输入转换为补丁嵌入表示
        self.patch_embedding = PvtV2OverlapPatchEmbeddings(
            config=config,
            layer_idx=layer_idx,
        )
        
        # Transformer块
        # 随机深度衰减规则
        drop_path_decays = torch.linspace(0, config.drop_path_rate, sum(config.depths)).tolist()
        block_layers = []
        # 构建由多个PvtV2块层组成的列表
        for block_idx in range(config.depths[layer_idx]):
            block_layers.append(
                PvtV2BlockLayer(
                    config=config,
                    layer_idx=layer_idx,
                    drop_path=drop_path_decays[sum(config.depths[:layer_idx]) + block_idx],
                )
            )
        self.blocks = nn.ModuleList(block_layers)

        # 层归一化
        self.layer_norm = nn.LayerNorm(config.hidden_sizes[layer_idx], eps=config.layer_norm_eps)

    def forward(self, hidden_states, output_attentions):
        # 如果需要输出注意力矩阵，则初始化一个空元组
        all_self_attentions = () if output_attentions else None
        
        # 第一步：获取补丁嵌入
        hidden_states, height, width = self.patch_embedding(hidden_states)
        
        # 第二步：通过所有块层处理嵌入
        for block in self.blocks:
            layer_outputs = block(hidden_states, height, width, output_attentions)
            hidden_states = layer_outputs[0]
            # 如果需要输出注意力矩阵，则将每个块层的注意力矩阵追加到元组中
            if output_attentions:
                all_self_attentions += (layer_outputs[1],)
        
        # 第三步：应用层归一化
        hidden_states = self.layer_norm(hidden_states)

        # 准备输出：仅包含隐藏状态或者隐藏状态和所有注意力矩阵
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (all_self_attentions,)

        return outputs, height, width


# 定义私有版本2的编码器，继承自nn.Module类
class PvtV2Encoder(nn.Module):
    def __init__(self, config: PvtV2Config):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False

        # 编码器层列表
        self.layers = nn.ModuleList([PvtV2EncoderLayer(config, i) for i in range(config.num_encoder_blocks)])

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        ) -> Union[Tuple, BaseModelOutput]:
        # 初始化隐藏状态和注意力矩阵的输出容器
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # 获取批处理大小并将输入像素值设置为初始隐藏状态
        batch_size = pixel_values.shape[0]
        hidden_states = pixel_values

        # 遍历所有层并处理每一层的输出
        for idx, layer in enumerate(self.layers):
            # 如果启用了梯度检查点且处于训练模式，则使用梯度检查点函数
            if self.gradient_checkpointing and self.training:
                layer_output = self._gradient_checkpointing_func(layer.__call__, hidden_states, output_attentions)
            else:
                # 否则直接调用层对象处理隐藏状态和注意力
                layer_output = layer(hidden_states, output_attentions)
            
            # 解包层输出，获取隐藏状态和尺寸信息
            outputs, height, width = layer_output
            hidden_states = outputs[0]  # 更新隐藏状态为当前层的主要输出

            # 如果需要输出注意力矩阵，则累加到全部自注意力矩阵中
            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[1],)

            # 将隐藏状态重塑回(batch_size, num_channels, height, width)的形状
            hidden_states = hidden_states.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()

            # 如果需要输出所有隐藏状态，则累加当前层处理后的隐藏状态
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要以字典形式返回结果，则以元组形式返回所有输出
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        
        # 否则以BaseModelOutput对象形式返回所有输出
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
class PvtV2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 定义配置类，指向PvtV2Config
    config_class = PvtV2Config
    # 模型基础名称前缀
    base_model_prefix = "pvt_v2"
    # 主要输入名称
    main_input_name = "pixel_values"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        # 如果是线性层，使用截断正态分布初始化权重，初始化范围为配置中的initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=self.config.initializer_range)
            # 如果有偏置，将偏置数据清零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是LayerNorm层，将偏置数据清零，权重数据填充为1.0
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        # 如果是卷积层，使用正态分布初始化权重，标准差为sqrt(2.0 / fan_out)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            # 如果有偏置，将偏置数据清零
            if module.bias is not None:
                module.bias.data.zero_()


PVT_V2_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~PvtV2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

PVT_V2_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`PvtImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Pvt-v2 encoder outputting raw hidden-states without any specific head on top.",
    PVT_V2_START_DOCSTRING,
)
# PvtV2Model类继承自PvtV2PreTrainedModel类，代表Pvt-v2编码器的原始隐藏状态输出，没有特定的输出头部
class PvtV2Model(PvtV2PreTrainedModel):
    def __init__(self, config: PvtV2Config):
        super().__init__(config)
        self.config = config

        # hierarchical Transformer encoder
        self.encoder = PvtV2Encoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 遍历需要修剪的层和对应需要修剪的注意力头
        for layer, heads in heads_to_prune.items():
            # 调用 encoder 中指定层的注意力模块，执行修剪操作
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(PVT_V2_INPUTS_DOCSTRING.format("(batch_size, channels, height, width)"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        # 根据参数或配置设置是否输出注意力权重
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 根据参数或配置设置是否输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 根据参数或配置设置是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入数据传入编码器模型进行前向传播
        encoder_outputs = self.encoder(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取编码器的序列输出
        sequence_output = encoder_outputs[0]

        if not return_dict:
            # 如果不使用返回字典，返回元组形式的序列输出和其他编码器输出
            return (sequence_output,) + encoder_outputs[1:]

        # 如果使用返回字典，则返回 BaseModelOutput 对象，包含序列输出、隐藏状态和注意力权重
        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 使用带有图片分类头部的 Pvt-v2 模型转换器类定义，该头部是在 [CLS] 标记的最终隐藏状态之上的线性层，例如适用于 ImageNet 数据集。
# 这个类继承自 PvtV2PreTrainedModel 类。
@add_start_docstrings(
    """
    Pvt-v2 模型的图片分类器，顶部带有一个线性层（放在最终隐藏状态的 [CLS] 标记之上），例如用于 ImageNet。
    """,
    PVT_V2_START_DOCSTRING,
)
class PvtV2ForImageClassification(PvtV2PreTrainedModel):
    
    def __init__(self, config: PvtV2Config) -> None:
        super().__init__(config)

        # 设置分类标签数量
        self.num_labels = config.num_labels
        # 初始化 Pvt-v2 模型
        self.pvt_v2 = PvtV2Model(config)

        # 分类器头部
        self.classifier = (
            # 如果标签数量大于 0，创建一个线性层
            nn.Linear(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(PVT_V2_INPUTS_DOCSTRING.format("(batch_size, channels, height, width)"))
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 前向传播函数定义，接受像素值张量和可选标签等参数

        pixel_values: Optional[torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 初始化返回字典，如果未指定则根据配置决定是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 PVT-v2 模型进行前向传播
        outputs = self.pvt_v2(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取序列输出（最后一层的隐藏状态）
        sequence_output = outputs[0]

        # 将形状从 (batch_size, num_channels, height, width) 转换为 (batch_size, height*width, hidden_size)
        batch_size = sequence_output.shape[0]
        sequence_output = sequence_output.permute(0, 2, 3, 1)  # 调整维度顺序
        sequence_output = sequence_output.reshape(batch_size, -1, self.config.hidden_sizes[-1])  # 重新形状化

        # 全局平均池化
        sequence_output = sequence_output.mean(dim=1)

        # 使用分类器进行预测
        logits = self.classifier(sequence_output)

        # 初始化损失为 None
        loss = None
        if labels is not None:
            # 确定问题类型（回归、单标签分类、多标签分类）
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型计算相应的损失函数
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

        # 如果不要求返回字典，则返回元组
        if not return_dict:
            output = (logits,) + outputs[1:]  # 包含 logits 和额外的输出（隐藏状态等）
            return ((loss,) + output) if loss is not None else output

        # 如果要求返回字典，则创建 ImageClassifierOutput 对象并返回
        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    PVTv2 backbone, to be used with frameworks like DETR and MaskFormer.
    """,
    PVT_V2_START_DOCSTRING,
)
class PvtV2Backbone(PvtV2Model, BackboneMixin):
    def __init__(self, config: PvtV2Config):
        super().__init__(config)
        super()._init_backbone(config)
        self.num_features = config.hidden_sizes
        """
        初始化函数，接受一个配置对象作为参数，并初始化 PVTv2 模型的骨干部分。
        设置模型的特征数为配置中的隐藏层大小。
        """

    @add_start_docstrings_to_model_forward(PVT_V2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BackboneOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BackboneOutput:
        """
        正向传播函数，接受输入像素值和一些可选参数，返回骨干网络的输出。

        Args:
            pixel_values (torch.FloatTensor): 输入的像素值张量。
            output_attentions (Optional[bool], optional): 是否输出注意力权重。默认为None。
            output_hidden_states (Optional[bool], optional): 是否输出隐藏状态。默认为None。
            return_dict (Optional[bool], optional): 是否返回字典格式的输出。默认为None。

        Returns:
            BackboneOutput: 包含特征图、隐藏状态和注意力权重的输出对象。

        Examples:
        
        ```python
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = AutoImageProcessor.from_pretrained("OpenGVLab/pvt_v2_b0")
        >>> model = AutoBackbone.from_pretrained(
        ...     "OpenGVLab/pvt_v2_b0", out_features=["stage1", "stage2", "stage3", "stage4"]
        ... )

        >>> inputs = processor(image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> feature_maps = outputs.feature_maps
        >>> list(feature_maps[-1].shape)
        [1, 256, 7, 7]
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        """
        确保返回字典的设置为配置中的默认值。
        确保输出隐藏状态的设置为配置中的默认值。
        """

        outputs = self.encoder(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,  # 总是输出隐藏状态以供后续使用
            return_dict=return_dict,
        )
        """
        使用编码器进行前向传播，传递输入像素值和其他参数。
        总是输出隐藏状态以确保能够提取特征。
        """

        hidden_states = outputs.hidden_states

        feature_maps = ()
        for idx, stage in enumerate(self.stage_names):
            if stage in self.out_features:
                feature_maps += (hidden_states[idx],)
        """
        根据设置的输出特征名称，从隐藏状态中选择对应阶段的特征图。
        """

        if not return_dict:
            output = (feature_maps,)
            if output_hidden_states:
                output += (outputs.hidden_states,)
            return output
        """
        如果不返回字典形式的输出，构建一个包含特征图和隐藏状态的元组返回。
        """

        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=None,
        )
        """
        返回包含特征图、隐藏状态和注意力权重的 BackboneOutput 对象。
        """
```