# `.\transformers\models\pvt\modeling_pvt.py`

```py
# coding=utf-8
# 版权所有 2023 作者：Wenhai Wang，Enze Xie，Xiang Li，Deng-Ping Fan，
# Kaitao Song，Ding Liang，Tong Lu，Ping Luo，Ling Shao 和 HuggingFace Inc. 团队。
# 保留所有权利。
#
# 根据 Apache 许可证 2.0 版（“许可证”）获得许可；
# 除非符合许可证的要求，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件是基于“按原样”提供的，
# 没有任何形式的明示或暗示的担保或条件。
# 有关特定语言的权限，请参见许可证。
""" PyTorch PVT 模型。"""

import collections
import math
from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from .configuration_pvt import PvtConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "PvtConfig"

_CHECKPOINT_FOR_DOC = "Zetatech/pvt-tiny-224"
_EXPECTED_OUTPUT_SHAPE = [1, 50, 512]

_IMAGE_CLASS_CHECKPOINT = "Zetatech/pvt-tiny-224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

# 预训练模型归档列表
PVT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Zetatech/pvt-tiny-224"
    # 查看所有 PVT 模型：https://huggingface.co/models?filter=pvt
]


# 从 transformers.models.beit.modeling_beit.drop_path 复制
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    按样本丢弃路径（随机深度）（当应用于残差块的主路径时）。

    Ross Wightman 的评论：这与我为 EfficientNet 等网络创建的 DropConnect 实现相同，
    但原始名称是具有误导性的，因为“Drop Connect”是另一篇论文中的不同形式的 dropout…
    请参见讨论：https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 …
    我选择了更改层和参数名称为“drop path”，而不是将 DropConnect 作为层名称，并使用“生存率”作为参数。
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # 适用于不同维度的张量，而不仅仅是 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # 二值化
    output = input.div(keep_prob) * random_tensor
    return output
# 定义一个与 ConvNextDropPath 相同功能但名称改为 PvtDropPath 的类
class PvtDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    # 前向传播函数，对输入的 hidden_states 进行 drop_path 操作
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    # 返回额外的表示字符串，表示 drop_prob 的值
    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


# 定义一个用于将图像像素值转换成初始隐藏状态的类
class PvtPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(
        self,
        config: PvtConfig,
        image_size: Union[int, Iterable[int]],
        patch_size: Union[int, Iterable[int]],
        stride: int,
        num_channels: int,
        hidden_size: int,
        cls_token: bool = False,
    ):
        super().__init__()
        self.config = config
        # 如果 image_size 和 patch_size 都不是 Iterable，则转换为 Iterable
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        # 定义位置编码的参数
        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_patches + 1 if cls_token else num_patches, hidden_size)
        )
        # 如果有 cls_token，则定义一个位置为 [1, 1, hidden_size] 的张量
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size)) if cls_token else None
        # 投影层，将 num_channels 维度映射到 hidden_size 维度
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=stride, stride=patch_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    # 插值位置编码，用于调整位置编码的大小
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        num_patches = height * width
        # 如果 num_patches 等于 config.image_size * config.image_size，则直接返回位置编码
        if num_patches == self.config.image_size * self.config.image_size:
            return self.position_embeddings
        embeddings = embeddings.reshape(1, height, width, -1).permute(0, 3, 1, 2)
        interpolated_embeddings = F.interpolate(embeddings, size=(height, width), mode="bilinear")
        interpolated_embeddings = interpolated_embeddings.reshape(1, -1, height * width).permute(0, 2, 1)
        return interpolated_embeddings
    # 定义前向传播方法，接受像素值张量，返回嵌入向量、高度、宽度
    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        # 获取像素值张量的批大小、通道数、高度、宽度
        batch_size, num_channels, height, width = pixel_values.shape
        # 如果通道数与配置中设置的通道数不匹配，则引发 ValueError 异常
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 将像素值张量通过投影层投影为补丁嵌入
        patch_embed = self.projection(pixel_values)
        # 获取补丁嵌入的形状，忽略前面的维度，更新高度和宽度变量
        *_, height, width = patch_embed.shape
        # 将补丁嵌入展平为二维，然后转置维度
        patch_embed = patch_embed.flatten(2).transpose(1, 2)
        # 对嵌入向量进行层归一化
        embeddings = self.layer_norm(patch_embed)
        # 如果存在类别标记，则扩展类别标记至整个批次，并与嵌入向量拼接
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            embeddings = torch.cat((cls_token, embeddings), dim=1)
            # 插值计算位置嵌入，排除第一个位置嵌入，保留高度和宽度相关位置嵌入
            position_embeddings = self.interpolate_pos_encoding(self.position_embeddings[:, 1:], height, width)
            position_embeddings = torch.cat((self.position_embeddings[:, :1], position_embeddings), dim=1)
        else:
            # 插值计算位置嵌入，根据高度和宽度设置位置嵌入
            position_embeddings = self.interpolate_pos_encoding(self.position_embeddings, height, width)
        # 将位置嵌入与嵌入向量相加，并进行丢弃正则化
        embeddings = self.dropout(embeddings + position_embeddings)

        # 返回嵌入向量、高度、宽度
        return embeddings, height, width
class PvtSelfOutput(nn.Module):
    def __init__(self, config: PvtConfig, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)  # 定义线性层，用于变换隐藏状态的维度
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # 定义丢弃层，用于随机丢弃部分神经元以防止过拟合

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)  # 使用线性层变换隐藏状态
        hidden_states = self.dropout(hidden_states)  # 使用丢弃层对变换后的隐藏状态进行随机丢弃
        return hidden_states  # 返回处理后的隐藏状态


class PvtEfficientSelfAttention(nn.Module):
    """Efficient self-attention mechanism with reduction of the sequence [PvT paper](https://arxiv.org/abs/2102.12122)."""

    def __init__(
        self, config: PvtConfig, hidden_size: int, num_attention_heads: int, sequences_reduction_ratio: float
    ):
        super().__init__()
        self.hidden_size = hidden_size  # 隐藏层大小
        self.num_attention_heads = num_attention_heads  # 注意力头的数量

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})"
            )

        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)  # 每个注意力头的大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 所有注意力头的总大小

        self.query = nn.Linear(self.hidden_size, self.all_head_size, bias=config.qkv_bias)  # 查询矩阵线性变换
        self.key = nn.Linear(self.hidden_size, self.all_head_size, bias=config.qkv_bias)  # 键矩阵线性变换
        self.value = nn.Linear(self.hidden_size, self.all_head_size, bias=config.qkv_bias)  # 值矩阵线性变换

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)  # 用于自注意力机制的丢弃层

        self.sequences_reduction_ratio = sequences_reduction_ratio  # 序列压缩比例
        if sequences_reduction_ratio > 1:
            self.sequence_reduction = nn.Conv2d(
                hidden_size, hidden_size, kernel_size=sequences_reduction_ratio, stride=sequences_reduction_ratio
            )  # 序列压缩层，用于减小序列维度
            self.layer_norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)  # 归一化层，用于减小序列压缩后的数据分布偏移

    def transpose_for_scores(self, hidden_states: int) -> torch.Tensor:
        new_shape = hidden_states.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # 调整张量形状以适应注意力计算
        hidden_states = hidden_states.view(new_shape)  # 调整张量形状
        return hidden_states.permute(0, 2, 1, 3)  # 转置张量以适应注意力计算

    def forward(
        self,
        hidden_states: torch.Tensor,
        height: int,
        width: int,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor]:
        # 获取查询向量
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        # 如果设置了序列缩减比例
        if self.sequences_reduction_ratio > 1:
            # 获取隐藏状态张量的尺寸信息
            batch_size, seq_len, num_channels = hidden_states.shape
            # 将隐藏状态张量重塑为(batch_size, num_channels, height, width)
            hidden_states = hidden_states.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)
            # 应用序列缩减
            hidden_states = self.sequence_reduction(hidden_states)
            # 将隐藏状态张量重塑为(batch_size, seq_len, num_channels)
            hidden_states = hidden_states.reshape(batch_size, num_channels, -1).permute(0, 2, 1)
            # 对隐藏状态进行 LayerNormalization
            hidden_states = self.layer_norm(hidden_states)

        # 获取键向量和值向量
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # 计算原始注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # 将注意力分数进行缩放
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 将注意力分数归一化为概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 执行 dropout 操作，随机丢弃一些注意力概率
        attention_probs = self.dropout(attention_probs)

        # 计算上下文向量
        context_layer = torch.matmul(attention_probs, value_layer)

        # 调整上下文向量的形状以便拼接
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # 如果需要输出注意力权重，则返回输出
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
```  
class PvtAttention(nn.Module):
    def __init__(
        self, config: PvtConfig, hidden_size: int, num_attention_heads: int, sequences_reduction_ratio: float
    ):
        super().__init__()
        # 初始化自注意力模块
        self.self = PvtEfficientSelfAttention(
            config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sequences_reduction_ratio=sequences_reduction_ratio,
        )
        # 初始化输出层
        self.output = PvtSelfOutput(config, hidden_size=hidden_size)
        # 存储需要剪枝的注意力头
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 查找可剪枝的注意力头
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )
        
        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并保存剪枝的注意力头
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self, hidden_states: torch.Tensor, height: int, width: int, output_attentions: bool = False
    ) -> Tuple[torch.Tensor]:
        # 执行自注意力模块，并返回输出
        self_outputs = self.self(hidden_states, height, width, output_attentions)

        # 将自注意力模块的输出传递给输出层
        attention_output = self.output(self_outputs[0])
        outputs = (attention_output,) + self_outputs[1:]  # 如果需要输出注意力，添加注意力
        return outputs


class PvtFFN(nn.Module):
    def __init__(
        self,
        config: PvtConfig,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
    ):
        super().__init__()
        # 输入到隐藏层的线性转换
        out_features = out_features if out_features is not None else in_features
        self.dense1 = nn.Linear(in_features, hidden_features)
        # 激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        # 隐藏层到输出的线性转换
        self.dense2 = nn.Linear(hidden_features, out_features)
        # 丢弃层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class PvtLayer(nn.Module):
```py`
    # 初始化函数，接受一系列参数
    def __init__(
        self,
        config: PvtConfig,
        hidden_size: int,
        num_attention_heads: int,
        drop_path: float,
        sequences_reduction_ratio: float,
        mlp_ratio: float,
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 初始化第一层 LayerNorm
        self.layer_norm_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        # 初始化注意力机制
        self.attention = PvtAttention(
            config=config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sequences_reduction_ratio=sequences_reduction_ratio,
        )
        # 初始化丢弃路径层
        self.drop_path = PvtDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # 初始化第二层 LayerNorm
        self.layer_norm_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        # 计算 MLP 的隐藏层大小
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        # 初始化多层感知机
        self.mlp = PvtFFN(config=config, in_features=hidden_size, hidden_features=mlp_hidden_size)

    # 前向传播函数，接收隐藏状态、高度、宽度和输出注意力信息的布尔值
    def forward(self, hidden_states: torch.Tensor, height: int, width: int, output_attentions: bool = False):
        # 进行自注意力计算
        self_attention_outputs = self.attention(
            hidden_states=self.layer_norm_1(hidden_states),
            height=height,
            width=width,
            output_attentions=output_attentions,
        )
        # 获取自注意力计算的输出
        attention_output = self_attention_outputs[0]
        # 获取额外的输出
        outputs = self_attention_outputs[1:]

        # 对注意力输出进行丢弃路径处理
        attention_output = self.drop_path(attention_output)
        # 将注意力输出与隐藏状态相加
        hidden_states = attention_output + hidden_states

        # 将隐藏状态经过第二层 LayerNorm
        mlp_output = self.mlp(self.layer_norm_2(hidden_states))

        # 对MLP输出进行丢弃路径处理
        mlp_output = self.drop_path(mlp_output)
        # 将隐藏状态与MLP输出相加
        layer_output = hidden_states + mlp_output

        # 将层输出放入输出元组中
        outputs = (layer_output,) + outputs

        # 返回输出
        return outputs
class PvtEncoder(nn.Module):
    # 初始化私有编码器类
    def __init__(self, config: PvtConfig):
        # 调用父类构造函数
        super().__init__()
        # 存储配置信息
        self.config = config

        # 随机深度衰减规则
        drop_path_decays = torch.linspace(0, config.drop_path_rate, sum(config.depths)).tolist()

        # 补丁嵌入
        embeddings = []

        # 遍历编码器块数量
        for i in range(config.num_encoder_blocks):
            # 创建补丁嵌入对象并添加到列表
            embeddings.append(
                PvtPatchEmbeddings(
                    config=config,
                    image_size=config.image_size if i == 0 else self.config.image_size // (2 ** (i + 1)),
                    patch_size=config.patch_sizes[i],
                    stride=config.strides[i],
                    num_channels=config.num_channels if i == 0 else config.hidden_sizes[i - 1],
                    hidden_size=config.hidden_sizes[i],
                    cls_token=i == config.num_encoder_blocks - 1,
                )
            )
        # 将补丁嵌入对象列表转换为模块列表
        self.patch_embeddings = nn.ModuleList(embeddings)

        # Transformer块
        blocks = []
        cur = 0
        # 遍历编码器块数量
        for i in range(config.num_encoder_blocks):
            # 每个块包含层
            layers = []
            if i != 0:
                cur += config.depths[i - 1]
            # 遍历每个块的深度
            for j in range(config.depths[i]):
                # 创建私有层对象并添加到列表
                layers.append(
                    PvtLayer(
                        config=config,
                        hidden_size=config.hidden_sizes[i],
                        num_attention_heads=config.num_attention_heads[i],
                        drop_path=drop_path_decays[cur + j],
                        sequences_reduction_ratio=config.sequence_reduction_ratios[i],
                        mlp_ratio=config.mlp_ratios[i],
                    )
                )
            # 将层列表转换为模块列表
            blocks.append(nn.ModuleList(layers))

        # 将块列表转换为模块列表
        self.block = nn.ModuleList(blocks)

        # 层归一化
        self.layer_norm = nn.LayerNorm(config.hidden_sizes[-1], eps=config.layer_norm_eps)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    # 函数定义，接收像素值和输出标志位，返回元组或基本模型输出
    ) -> Union[Tuple, BaseModelOutput]:
        # 如果不输出隐藏状态，则初始化隐藏状态为一个空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果不输出注意力权重，则初始化注意力权重为一个空元组
        all_self_attentions = () if output_attentions else None
    
        # 获取输入像素值的批大小
        batch_size = pixel_values.shape[0]
        # 获取 Transformer 模型的块数
        num_blocks = len(self.block)
        # 初始化隐藏状态为像素值
        hidden_states = pixel_values
        # 遍历每个块
        for idx, (embedding_layer, block_layer) in enumerate(zip(self.patch_embeddings, self.block)):
            # 首先，获取嵌入层的输出和尺寸
            hidden_states, height, width = embedding_layer(hidden_states)
            # 其次，通过每个块处理嵌入层输出
            for block in block_layer:
                layer_outputs = block(hidden_states, height, width, output_attentions)
                hidden_states = layer_outputs[0]
                # 如果输出注意力权重，则记录注意力权重
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果输出隐藏状态，则记录隐藏状态
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)
            # 如果不是最后一个块，对隐藏状态进行形状转换
            if idx != num_blocks - 1:
                hidden_states = hidden_states.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
        # 对隐藏状态进行 LayerNormalization
        hidden_states = self.layer_norm(hidden_states)
        # 如果输出隐藏状态，则记录隐藏状态
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        # 如果不返回字典，则返回非空的元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 返回基本模型输出
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
    ```  
class PvtPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 使用 PvtConfig 进行配置
    config_class = PvtConfig
    # 基础模型前缀为 "pvt"
    base_model_prefix = "pvt"
    # 主输入名称为 "pixel_values"
    main_input_name = "pixel_values"

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # 如果是线性层，使用截断正态分布初始化权重
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置项，初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            # 如果是 LayerNorm 层，初始化偏置为零，权重为 1.0
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, PvtPatchEmbeddings):
            # 如果是 PvtPatchEmbeddings 层，初始化位置嵌入为截断正态分布
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data,
                mean=0.0,
                std=self.config.initializer_range,
            )
            # 如果存在 cls_token，初始化为截断正态分布
            if module.cls_token is not None:
                module.cls_token.data = nn.init.trunc_normal_(
                    module.cls_token.data,
                    mean=0.0,
                    std=self.config.initializer_range,
                )


# 模型文档的开头部分
PVT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~PvtConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 输入文档字符串的部分
PVT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`PvtImageProcessor.__call__`]
            for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# 使用装饰器添加文档字符串的部分
@add_start_docstrings(
    "The bare Pvt encoder outputting raw hidden-states without any specific head on top.",
    # 设置私有变量 PVT_START_DOCSTRING, 用于开头文档字符串标记的定位
    
    代码：
    
    
        '"""UNKNOWN',
    
    
    注释：
    # 设置字符串变量 '"""UNKNOWN'，表示未知的部分
    
    代码：
    
    
        AUTOGEN_IGNORE_ALL,
    
    
    注释：
    # 设置常量 AUTOGEN_IGNORE_ALL，表示忽略所有自动化生成的代码
    
    代码：
    
    
        AUTOGEN_TEMPLATE_ENGINE_MAKO,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_ENGINE_MAKO，表示使用 Mako 模板引擎进行自动生成
    
    代码：
    
    
        AUTOGEN_TEMPLATE_ENGINE_WM,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_ENGINE_WM，表示使用 WM 模板引擎进行自动生成
    
    代码：
    
    
        AUTOGEN_TEMPLATE_FUNC_NAME,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_FUNC_NAME，表示自动生成代码的函数名称
    
    代码：
    
    
        AUTOGEN_TEMPLATE_FUNC_ENGINE,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_FUNC_ENGINE，表示自动生成代码的引擎
    
    代码：
    
    
        AUTOGEN_TEMPLATE_FUNC_CONFIG,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_FUNC_CONFIG，表示自动生成代码的配置
    
    代码：
    
    
        AUTOGEN_TEMPLATE_FUNC_PACKAGE,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_FUNC_PACKAGE，表示自动生成代码的包名
    
    代码：
    
    
        AUTOGEN_TEMPLATE_FUNC_CODE,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_FUNC_CODE，表示自动生成代码的代码块
    
    代码：
    
    
        AUTOGEN_TEMPLATE_FUNC_LANG,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_FUNC_LANG，表示自动生成代码的语言
    
    代码：
    
    
        AUTOGEN_TEMPLATE_FUNC_HEADER,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_FUNC_HEADER，表示自动生成代码的头部
    
    代码：
    
    
        AUTOGEN_TEMPLATE_FUNC_FOOTER,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_FUNC_FOOTER，表示自动生成代码的尾部
    
    代码：
    
    
        AUTOGEN_TEMPLATE_FUNC_FILE,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_FUNC_FILE，表示自动生成代码的文件名
    
    代码：
    
    
        AUTOGEN_TEMPLATE_FUNC_DIR,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_FUNC_DIR，表示自动生成代码的目录
    
    代码：
    
    
        AUTOGEN_TEMPLATE_FUNC_FILENAME,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_FUNC_FILENAME，表示自动生成代码的文件名
    
    代码：
    
    
        AUTOGEN_TEMPLATE_FUNC_FUNCTION,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_FUNC_FUNCTION，表示自动生成代码的函数名
    
    代码：
    
    
        AUTOGEN_TEMPLATE_FUNC_VARIABLE,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_FUNC_VARIABLE，表示自动生成代码的变量名
    
    代码：
    
    
        AUTOGEN_TEMPLATE_FUNC_CLASS,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_FUNC_CLASS，表示自动生成代码的类名
    
    代码：
    
    
        AUTOGEN_TEMPLATE_FUNC_OBJECT,
    
    
    注释：
    # 设置常�� AUTOGEN_TEMPLATE_FUNC_OBJECT，表示自动生成代码的对象名
    
    代码：
    
    
        AUTOGEN_TEMPLATE_FUNC_METHOD,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_FUNC_METHOD，表示自动生成代码的方法名
    
    代码：
    
    
        AUTOGEN_TEMPLATE_FUNC_MEMBER,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_FUNC_MEMBER，表示自动生成代码的成员名
    
    代码：
    
    
        AUTOGEN_TEMPLATE_FUNC_CTOR,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_FUNC_CTOR，表示自动生成代码的构造函数名
    
    代码：
    
    
        AUTOGEN_TEMPLATE_FUNC_DTOR,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_FUNC_DTOR，表示自动生成代码的析构函数名
    
    代码：
    
    
        AUTOGEN_TEMPLATE_FUNC_EVAL,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_FUNC_EVAL，表示自动生成代码的 evaluate 名称
    
    代码：
    
    
        AUTOGEN_TEMPLATE_FUNC_IMPORT,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_FUNC_IMPORT，表示自动生成代码的 import 名称
    
    代码：
    
    
        AUTOGEN_TEMPLATE_FUNC_MACRO,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_FUNC_MACRO，表示自动生成代码的宏名
    
    代码：
    
    
        AUTOGEN_TEMPLATE_FUNC_NAMESPACE,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_FUNC_NAMESPACE，表示自动生成代码的 namespace 名称
    
    代码：
    
    
        AUTOGEN_TEMPLATE_FUNC_INCLUDE,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_FUNC_INCLUDE，表示自动生成代码的 include 名称
    
    代码：
    
    
        AUTOGEN_TEMPLATE_FUNC_DEFINE,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_FUNC_DEFINE，表示自动生成代码的 define 名称
    
    代码：
    
    
        AUTOGEN_TEMPLATE_FUNC_TYPEDEF,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_FUNC_TYPEDEF，表示自动生成代码的 typedef 名称
    
    代码：
    
    
        AUTOGEN_TEMPLATE_FUNC_ANNOTATE,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_FUNC_ANNOTATE，表示自动生成代码的 annotate 名称
    
    代码：
    
    
        AUTOGEN_TEMPLATE_FUNC_ASSERT,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_FUNC_ASSERT，表示自动生成代码的 assert 名称
    
    代码：
    
    
        AUTOGEN_TEMPLATE_FUNC_ASSERTEQUAL,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_FUNC_ASSERTEQUAL，表示自动生成代码的 assertequal 名称
    
    代码：
    
    
        AUTOGEN_TEMPLATE_FUNC_ASSERTNOTEQUAL,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_FUNC_ASSERTNOTEQUAL，表示自动生成代码的 assertnotequal 名称
    
    代码：
    
    
        AUTOGEN_TEMPLATE_FUNC_ASSERTIN,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_FUNC_ASSERTIN，表示自动生成代码的 assertin 名称
    
    代码：
    
    
        AUTOGEN_TEMPLATE_FUNC_ASSERTNOTIN,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_FUNC_ASSERTNOTIN，表示自动生成代码的 assertnotin 名称
    
    代码：
    
    
        AUTOGEN_TEMPLATE_FUNC_ASSERTIS,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_FUNC_ASSERTIS，表示自动生成代码的 assertis 名称
    
    代码：
    
    
        AUTOGEN_TEMPLATE_FUNC_ASSERTISNOT,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_FUNC_ASSERTISNOT，表示自动生成代码的 assertisnot 名称
    
    代码：
    
    
        AUTOGEN_TEMPLATE_FUNC_ASSERTRAISES,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_FUNC_ASSERTRAISES，表示自动生成代码的 assertraises 名称
    
    代码：
    
    
        AUTOGEN_TEMPLATE_FUNC_ASSERTDOESNOTRAISE,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_FUNC_ASSERTDOESNOTRAISE，表示自动生成代码的 assertdoesnotraise 名称
    
    代码：
    
    
        AUTOGEN_IGNORE_PUBLIC,
    
    
    注释：
    # 设置常量 AUTOGEN_IGNORE_PUBLIC，忽略所有公共字段的自动生成代码
    
    代码：
    
    
        AUTOGEN_IGNORE_PRIVATE,
    
    
    注释：
    # 设置常量 AUTOGEN_IGNORE_PRIVATE，忽略所有私有字段的自动生成代码
    
    代码：
    
    
        AUTOGEN_IGNORE_PROTECTED,
    
    
    注释：
    # 设置常量 AUTOGEN_IGNORE_PROTECTED，忽略所有保护字段的自动生成代码
    
    代码：
    
    
        AUTOGEN_IGNORE_STATIC,
    
    
    注释：
    # 设置常量 AUTOGEN_IGNORE_STATIC，忽略所有静态方法和属性的自动生成代码
    
    代码：
    
    
        AUTOGEN_IGNORE_METHODS,
    
    
    注释：
    # 设置常量 AUTOGEN_IGNORE_METHODS，忽略所有方法的自动生成代码
    
    代码：
    
    
        AUTOGEN_IGNORE_ATTRIBUTES,
    
    
    注释：
    # 设置常量 AUTOGEN_IGNORE_ATTRIBUTES，忽略所有属性的自动生成代码
    
    代码：
    
    
        AUTOGEN_IGNORE_METHOD_PREFIX,
    
    
    注释：
    # 设置常量 AUTOGEN_IGNORE_METHOD_PREFIX，忽略以特定前缀开头的方法的自动生成代码
    
    代码：
    
    
        AUTOGEN_IGNORE_METHOD_SUFFIX,
    
    
    注释：
    # 设置常量 AUTOGEN_IGNORE_METHOD_SUFFIX，忽略以特定后缀结尾的方法的自动生成代码
    
    代码：
    
    
        AUTOGEN_IGNORE_ATTRIBUTE_PREFIX,
    
    
    注释：
    # 设置常量 AUTOGEN_IGNORE_ATTRIBUTE_PREFIX，忽略以特定前缀开头的属性的自动生成代码
    
    代码：
    
    
        AUTOGEN_IGNORE_ATTRIBUTE_SUFFIX,
    
    
    注释：
    # 设置常量 AUTOGEN_IGNORE_ATTRIBUTE_SUFFIX，忽略以特定后缀结尾的属性的自动生成代码
    
    代码：
    
    
        AUTOGEN_IGNORE_GLOBALS,
    
    
    注释：
    # 设置常量 AUTOGEN_IGNORE_GLOBALS，忽略所有全局变量的自动生成代码
    
    代码：
    
    
        AUTOGEN_IGNORE_CONSTANTS,
    
    
    注释：
    # 设置常量 AUTOGEN_IGNORE_CONSTANTS，忽略所有常量的自动生成代码
    
    代码：
    
    
        AUTOGEN_IGNORE_CLASSMETHODS,
    
    
    注释：
    # 设置常量 AUTOGEN_IGNORE_CLASSMETHODS，忽略所有类方法的自动生成代码
    
    代码：
    
    
        AUTOGEN_IGNORE_STATICMETHODS,
    
    
    注释：
    # 设置常量 AUTOGEN_IGNORE_STATICMETHODS，忽略所有静态方法的自动生成代码
    
    代码：
    
    
        AUTOGEN_IGNORE_INSTANCEMETHODS,
    
    
    注释：
    # 设置常量 AUTOGEN_IGNORE_INSTANCEMETHODS，忽略所有实例方法的自动生成代码
    
    代码：
    
    
        AUTOGEN_IGNORE_DECORATED,
    
    
    注释：
    # 设置常量 AUTOGEN_IGNORE_DECORATED，忽略所有带装饰器的方法和属性的自动生成代码
    
    代码：
    
    
        AUTOGEN_IGNORE_DOCSTRINGS,
    
    
    注释：
    # 设置常量 AUTOGEN_IGNORE_DOCSTRINGS，忽略所有带文档字符串的方法和属性的自动生成代码
    
    代码：
    
    
        AUTOGEN_IGNORE_PRIVATEMEMBERS,
    
    
    注释：
    # 设置常量 AUTOGEN_IGNORE_PRIVATEMEMBERS，忽略所有私有成员的自动生成代码
    
    代码：
    
    
        AUTOGEN_IGNORE_PROTECTEDMEMBERS,
    
    
    注释：
    # 设置常量 AUTOGEN_IGNORE_PROTECTEDMEMBERS，忽略所有保护成员的自动生成代码
    
    代码：
    
    
        AUTOGEN_IGNORE_REQUIREMENTS,
    
    
    注释：
    # 设置常量 AUTOGEN_IGNORE_REQUIREMENTS，忽略所有依赖的自动生成代码
    
    代码：
    
    
        AUTOGEN_IGNORE_VENV_REQUIREMENTS,
    
    
    注释：
    # 设置常量 AUTOGEN_IGNORE_VENV_REQUIREMENTS，忽略所有虚拟环境依赖的自动生成代码
    
    代码：
    
    
        AUTOGEN_IGNORE_COMMENTED,
    
    
    注释：
    # 设置常量 AUTOGEN_IGNORE_COMMENTED，忽略所有被注释掉的代码的自动生成代码
    
    代码：
    
    
        AUTOGEN_IGNORE_DEPRECATED,
    
    
    注释：
    # 设置常量 AUTOGEN_IGNORE_DEPRECATED，忽略所有废弃的代码的自动生成代码
    
    代码：
    
    
        AUTOGEN_IGNORE_DEPRECATED_METHODS,
    
    
    注释：
    # 设置常量 AUTOGEN_IGNORE_DEPRECATED_METHODS，忽略所有废弃方法的自动生成代码
    
    代码：
    
    
        AUTOGEN_IGNORE_DEPRECATED_ATTRIBUTES,
    
    
    注释：
    # 设置常量 AUTOGEN_IGNORE_DEPRECATED_ATTRIBUTES，忽略所有废弃属性的自动生成代码
    
    代码：
    
    
        AUTOGEN_IGNORE_NOCOPY,
    
    
    注释：
    # 设置常量 AUTOGEN_IGNORE_NOCOPY，忽略所有不可复制的代码的自动生成代码
    
    代码：
    
    
        AUTOGEN_IGNORE_NOEQUAL,
    
    
    注释：
    # 设置常量 AUTOGEN_IGNORE_NOEQUAL，忽略所有无法比较的代码的自动生成代码
    
    代码：
    
    
        AUTOGEN_IGNORE_NOMANUAL,
    
    
    注释：
    # 设置常量 AUTOGEN_IGNORE_NOMANUAL，忽略所有无手工编写文档的代码的自动生成代码
    
    代码：
    
    
        AUTOGEN_IGNORE_NOAUTO,
    
    
    注释：
    # 设置常量 AUTOGEN_IGNORE_NOAUTO，忽略所有非自动化生成的代码的自动生成代码
    
    代码：
    
    
        AUTOGEN_IGNORE_MUTABLE,
    
    
    注释：
    # 设置常量 AUTOGEN_IGNORE_MUTABLE，忽略所有可变代码的自动生成代码
    
    代码：
    
    
        AUTOGEN_IGNORE_NONMUTABLE,
    
    
    注释：
    # 设置常量 AUTOGEN_IGNORE_NONMUTABLE，忽略所有不可变代码的自动生成代码
    
    代码：
    
    
        AUTOGEN_INVALID_ARGS_LOG_IGNORE,
    
    
    注释：
    # 设置常量 AUTOGEN_INVALID_ARGS_LOG_IGNORE，表示自动生成代码时将无效参数的日志设置为忽略
    
    代码：
    
    
        AUTOGEN_INVALID_ARGS_LOG_WARNING,
    
    
    注释：
    # 设置常量 AUTOGEN_INVALID_ARGS_LOG_WARNING，表示自动生成代码时将无效参数的日志设置为警告
    
    代码：
    
    
        AUTOGEN_INVALID_ARGS_LOG_ERROR,
    
    
    注释：
    # 设置常量 AUTOGEN_INVALID_ARGS_LOG_ERROR，表示自动生成代码时将无效参数的日志设置为错误
    
    代码：
    
    
        AUTOGEN_INVALID_ARGS_LOG_RAISE,
    
    
    注释：
    # 设置常量 AUTOGEN_INVALID_ARGS_LOG_RAISE，表示自动生成代码时将无效参数的日志异常抛出
    
    代码：
    
    
        AUTOGEN_TEMPLATE_UNKNOWN_HANDLING_IGNORE,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_UNKNOWN_HANDLING_IGNORE，表示自动生成代码时忽略未知处理模板错误
    
    代码：
    
    
        AUTOGEN_TEMPLATE_UNKNOWN_HANDLING_WARNING,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_UNKNOWN_HANDLING_WARNING，表示自动生成代码时将未知处理模板错误设置为警告
    
    代码：
    
    
        AUTOGEN_TEMPLATE_UNKNOWN_HANDLING_ERROR,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_UNKNOWN_HANDLING_ERROR，表示自动生成代码时将未知处理模板错误设置为错误
    
    代码：
    
    
        AUTOGEN_TEMPLATE_UNKNOWN_HANDLING_RAISE,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_UNKNOWN_HANDLING_RAISE，表示自动生成代码时将未知处理模板错误设置为异常抛出
    
    代码：
    
    
        AUTOGEN_TEMPLATE_EVAL_CONTEXT,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_EVAL_CONTEXT，表示自动生成代码的评估上下文
    
    代码：
    
    
        AUTOGEN_TEMPLATE_EVAL_GLOBALS,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_EVAL_GLOBALS，表示自动生成代码的全局评估环境
    
    代码：
    
    
        AUTOGEN_TEMPLATE_EVAL_LOCALS,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_EVAL_LOCALS，表示自动生成代码的局部评估环境
    
    代码：
    
    
        AUTOGEN_TEMPLATE_EVAL_RESULT,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_EVAL_RESULT，表示自动生成代码的评估结果
    
    代码：
    
    
        AUTOGEN_TEMPLATE_EVAL_DIRECTIVES,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_EVAL_DIRECTIVES，表示自动生成代码的评估指令
    
    代码：
    
    
        AUTOGEN_TEMPLATE_EVAL_MODE,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_EVAL_MODE，表示自动生成代码的评估模式
    
    代码：
    
    
        AUTOGEN_TEMPLATE_EVAL_OPTIONS,
    
    
    注释：
    # 设置常量 AUTOGEN_TEMPLATE_EVAL_OPTIONS，表示自动生成代码的评估选项
    
    代码：
    
    
        PVT_AUTOGEN_IGNORE_ALL_USES_DOCSTRING,
    
    
    注释：
    # 设置私有变量 PVT_AUTOGEN_IGNORE_ALL_USES_DOCSTRING，用于忽略所有使用文档字符串的自动生成代码
    
    代码：
    
    
        PVT_AUTOGEN_IGNORE_ALL_USES_COMMENTED,
    
    
    注释：
    # 设置私有变量 PVT_AUTOGEN_IGNORE_ALL_USES_COMMENTED，用于忽略所有使用注释的自动生成代码
    
    代码：
    
    
        PVT_AUTOGEN_IGNORE_ALL_USES_DEPRECATION,
    
    
    注释：
    # 设置私有变量 PVT_AUTOGEN_IGNORE_ALL_USES_DEPRECATION，用于忽略所有使用废弃方法或属性的自动生成代码
    
    代码：
    
    
        PVT_AUTOGEN_IGNORE_ALL_USES_MANUAL,
    
    
    注释：
    # 设置私有变量 PVT_AUTOGEN_IGNORE_ALL_USES_MANUAL，用
# 定义了 PvtModel 类，继承自 PvtPreTrainedModel 类
class PvtModel(PvtPreTrainedModel):
    def __init__(self, config: PvtConfig):
        # 调用父类的构造方法，并传入配置信息
        super().__init__(config)
        self.config = config

        # 创建 PvtEncoder 对象作为编码器
        self.encoder = PvtEncoder(config)

        # 初始化权重并应用最终处理
        self.post_init()

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 遍历需要剪枝的头部
        for layer, heads in heads_to_prune.items():
            # 对编码器中每一层的注意力头进行剪枝
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 重写 forward 方法
    @add_start_docstrings_to_model_forward(PVT_INPUTS_DOCSTRING.format("(batch_size, channels, height, width)"))
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
        # 设置 output_attentions, output_hidden_states 和 return_dict 的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 对输入的像素值进行编码操作，得到编码器的输出
        encoder_outputs = self.encoder(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            # 如果不需要返回字典，则返回元组
            return (sequence_output,) + encoder_outputs[1:]

        # 返回 BaseModelOutput 对象
        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

# 根据给定信息定义 PvtForImageClassification 类
@add_start_docstrings(
    """
    Pvt Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.
    """,
    PVT_START_DOCSTRING,
)
class PvtForImageClassification(PvtPreTrainedModel):
    def __init__(self, config: PvtConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.pvt = PvtModel(config)

        # 分类器头部
        self.classifier = (
            nn.Linear(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(PVT_INPUTS_DOCSTRING.format("(batch_size, channels, height, width)"))
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,  # 使用装饰器为方法添加文档字符串，包括检查点、输出类型、配置类和预期输出类型等信息
        output_type=ImageClassifierOutput,   # 输出类型为图像分类器输出类型
        config_class=_CONFIG_FOR_DOC,        # 配置类为代码中的 CONFIG_FOR_DOC
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,  # 期望的输出类型为图像分类器期望输出类型
    )
    def forward(  # 定义了一个名为 forward 的方法
        self,
        pixel_values: Optional[torch.Tensor],  # 输入参数 pixel_values 的类型为可选的 torch.Tensor
        labels: Optional[torch.Tensor] = None,  # 输入参数 labels 的类型为可选的 torch.Tensor，默认值为 None
        output_attentions: Optional[bool] = None,  # 是否输出注意力的类型为可选的布尔值，默认值为 None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态的类型为可选的布尔值，默认值为 None
        return_dict: Optional[bool] = None,  # 是否返回字典的类型为可选的布尔值，默认值为 None
    ) -> Union[tuple, ImageClassifierOutput]:  # 方法返回类型是元组或者图像分类器输出类型
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):  # labels 参数是类型为 torch.LongTensor 的张量，形状为 (batch_size,)，可选的
            Labels for computing the image classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
            用于计算图像分类或回归损失的标签。索引应该在 `[0, ..., config.num_labels - 1]` 范围内。如果 `config.num_labels == 1`，则计算回归损失（均方损失），如果 `config.num_labels > 1`，则计算分类损失（交叉熵）。
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # 如果 return_dict 不是 None, 则使用 return_dict，否则使用 self.config.use_return_dict

        outputs = self.pvt(  # 调用 self.pvt 方法，将返回值赋给 outputs
            pixel_values=pixel_values,  # 将 pixel_values 作为参数传递给 self.pvt 方法
            output_attentions=output_attentions,  # 将 output_attentions 作为参数传递给 self.pvt 方法
            output_hidden_states=output_hidden_states,  # 将 output_hidden_states 作为参数传递给 self.pvt 方法
            return_dict=return_dict,  # 将 return_dict 作为参数传递给 self.pvt 方法
        )

        sequence_output = outputs[0]  # 从输出中获取第一个元素，赋��给 sequence_output

        logits = self.classifier(sequence_output[:, 0, :])  # 使用 sequence_output 的某些部分作为输入，调用 self.classifier 方法，将返回值赋给 logits

        loss = None  # 初始化变量 loss 为 None
        if labels is not None:  # 如果 labels 不是 None
            if self.config.problem_type is None:  # 如果 self.config.problem_type 是 None
                if self.num_labels == 1:  # 如果 num_labels 等于 1
                    self.config.problem_type = "regression"  # 将 problem_type 设置为 "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):  # 否则如果 num_labels 大于 1 且 labels 的类型为 torch.long 或 torch.int
                    self.config.problem_type = "single_label_classification"  # 将 problem_type 设置为 "single_label_classification"
                else:  # 其他情况
                    self.config.problem_type = "multi_label_classification"  # 将 problem_type 设置为 "multi_label_classification"

            if self.config.problem_type == "regression":  # 如果 problem_type 是 "regression"
                loss_fct = MSELoss()  # 使用均方损失作为损失函数
                if self.num_labels == 1:  # 如果 num_labels 等于 1
                    loss = loss_fct(logits.squeeze(), labels.squeeze())  # 计算损失
                else:  # 否则
                    loss = loss_fct(logits, labels)  # 计算损失
            elif self.config.problem_type == "single_label_classification":  # 如果 problem_type 是 "single_label_classification"
                loss_fct = CrossEntropyLoss()  # 使用交叉熵损失作为损失函数
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))  # 计算损失
            elif self.config.problem_type == "multi_label_classification":  # 如果 problem_type 是 "multi_label_classification"
                loss_fct = BCEWithLogitsLoss()  # 使用带 logits 的二元交叉熵损失作为损失函数
                loss = loss_fct(logits, labels)  # 计算损失

        if not return_dict:  # 如果不返回字典
            output = (logits,) + outputs[1:]  # 将 logits 和其他输出组成元组，赋值给 output
            return ((loss,) + output) if loss is not None else output  # 如果 loss 不为 None，则返回 loss 和 output 构成的元组，否则返回 output

        return ImageClassifierOutput(  # 返回图像分类器输出
            loss=loss,  # 返回损失
            logits=logits,  # 返回 logits
            hidden_states=outputs.hidden_states,  # 返回隐藏状态
            attentions=outputs.attentions,  # 返回注意力
        )
```