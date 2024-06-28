# `.\models\dinat\modeling_dinat.py`

```py
# coding=utf-8
# 版权 2022 年 SHI Labs 和 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可；
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件是基于“按原样”提供的，
# 没有任何明示或暗示的保证或条件。
# 有关详细信息，请参阅许可证。
""" PyTorch Dilated Neighborhood Attention Transformer model."""

# 导入必要的库
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入相关的自定义模块和函数
from ...activations import ACT2FN
from ...modeling_outputs import BackboneOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    ModelOutput,
    OptionalDependencyNotAvailable,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_natten_available,
    logging,
    replace_return_docstrings,
    requires_backends,
)
from ...utils.backbone_utils import BackboneMixin
from .configuration_dinat import DinatConfig

# 检查是否安装了 natten 库，如果安装了，导入相关函数，否则定义占位函数并抛出异常
if is_natten_available():
    from natten.functional import natten2dav, natten2dqkrpb
else:

    def natten2dqkrpb(*args, **kwargs):
        raise OptionalDependencyNotAvailable()

    def natten2dav(*args, **kwargs):
        raise OptionalDependencyNotAvailable()

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 模型配置文件的通用文档字符串
_CONFIG_FOR_DOC = "DinatConfig"

# 检查点地址的基础文档字符串
_CHECKPOINT_FOR_DOC = "shi-labs/dinat-mini-in1k-224"

# 预期输出形状的基础文档字符串
_EXPECTED_OUTPUT_SHAPE = [1, 7, 7, 512]

# 图像分类模型检查点文档字符串
_IMAGE_CLASS_CHECKPOINT = "shi-labs/dinat-mini-in1k-224"

# 图像分类预期输出文档字符串
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

# Dinat 预训练模型存档列表
DINAT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "shi-labs/dinat-mini-in1k-224",
    # 查看所有 Dinat 模型：https://huggingface.co/models?filter=dinat
]

# drop_path 和 DinatDropPath 是从 timm 库中导入的。
    # 定义函数的参数和返回类型注释，以下是函数的输入参数说明：
    #   last_hidden_state：模型最后一层的隐藏状态，类型为 torch.FloatTensor，形状为 (batch_size, sequence_length, hidden_size)
    #   hidden_states：模型各层隐藏状态的元组，每个元素类型为 torch.FloatTensor，形状为 (batch_size, sequence_length, hidden_size)，可选参数，当 output_hidden_states=True 时返回
    #   attentions：注意力权重的元组，每个元素类型为 torch.FloatTensor，形状为 (batch_size, num_heads, sequence_length, sequence_length)，可选参数，当 output_attentions=True 时返回
    #   reshaped_hidden_states：重新调整后的隐藏状态的元组，每个元素类型为 torch.FloatTensor，形状为 (batch_size, hidden_size, height, width)，可选参数，当 output_hidden_states=True 时返回
    
    last_hidden_state: torch.FloatTensor = None
    # 初始化最后一层的隐藏状态为 None，类型为 torch.FloatTensor
    
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 初始化隐藏状态元组为 None，类型为 Optional[Tuple[torch.FloatTensor, ...]]，表示可能为 None 或包含多个 torch.FloatTensor 元素的元组
    
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 初始化注意力权重元组为 None，类型为 Optional[Tuple[torch.FloatTensor, ...]]，表示可能为 None 或包含多个 torch.FloatTensor 元素的元组
    
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 初始化重新调整后的隐藏状态元组为 None，类型为 Optional[Tuple[torch.FloatTensor, ...]]，表示可能为 None 或包含多个 torch.FloatTensor 元素的元组
# 从`transformers.models.nat.modeling_nat.NatModelOutput`复制并将`Nat`改为`Dinat`的数据类定义
@dataclass
class DinatModelOutput(ModelOutput):
    """
    Dinat model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`, *optional*, returned when `add_pooling_layer=True` is passed):
            Average pooling of the last layer hidden-state.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each stage) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, hidden_size, height, width)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
            include the spatial dimensions.
    """

    # Dinat模型的最后隐藏状态
    last_hidden_state: torch.FloatTensor = None
    # 可选项，当传递`add_pooling_layer=True`时返回，最后一层隐藏状态的平均池化
    pooler_output: Optional[torch.FloatTensor] = None
    # 可选项，当传递`output_hidden_states=True`时或`config.output_hidden_states=True`时返回，模型每层的隐藏状态的元组
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 可选项，当传递`output_attentions=True`时或`config.output_attentions=True`时返回，注意力权重的元组
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    # 可选项，当传递`output_hidden_states=True`时或`config.output_hidden_states=True`时返回，包含空间维度的隐藏状态的元组
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None


# 从`transformers.models.nat.modeling_nat.NatImageClassifierOutput`复制并将`Nat`改为`Dinat`的数据类定义
@dataclass
class DinatImageClassifierOutput(ModelOutput):
    """
    Dinat outputs for image classification.
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            分类（如果 `config.num_labels==1` 则为回归）的损失值。
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            SoftMax 之前的分类（或回归，如果 `config.num_labels==1`）分数。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            模型每一层的隐藏状态，包括初始嵌入输出。

            包含形状为 `(batch_size, sequence_length, hidden_size)` 的 `torch.FloatTensor` 元组。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            注意力权重经过注意力 SoftMax 后的值，用于计算自注意力头中的加权平均值。

            包含形状为 `(batch_size, num_heads, sequence_length, sequence_length)` 的 `torch.FloatTensor` 元组。
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            模型每一层的隐藏状态，包括初始嵌入输出，且包含空间维度。

            包含形状为 `(batch_size, hidden_size, height, width)` 的 `torch.FloatTensor` 元组。
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
# 从 transformers.models.nat.modeling_nat.NatEmbeddings 复制的类，将 Nat 替换为 Dinat
class DinatEmbeddings(nn.Module):
    """
    构建补丁和位置嵌入。

    Args:
        config:
            模型配置对象，包含嵌入维度等参数。
    """

    def __init__(self, config):
        super().__init__()

        # 使用 DinatPatchEmbeddings 类构建补丁嵌入
        self.patch_embeddings = DinatPatchEmbeddings(config)

        # 应用 LayerNorm 进行归一化
        self.norm = nn.LayerNorm(config.embed_dim)
        # 应用 dropout 进行正则化
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, pixel_values: Optional[torch.FloatTensor]) -> Tuple[torch.Tensor]:
        # 生成补丁嵌入
        embeddings = self.patch_embeddings(pixel_values)
        # 对嵌入应用 LayerNorm
        embeddings = self.norm(embeddings)
        # 对归一化后的嵌入应用 dropout
        embeddings = self.dropout(embeddings)

        return embeddings


# 从 transformers.models.nat.modeling_nat.NatPatchEmbeddings 复制的类，将 Nat 替换为 Dinat
class DinatPatchEmbeddings(nn.Module):
    """
    这个类将形状为 `(batch_size, num_channels, height, width)` 的 `pixel_values` 转换为形状为
    `(batch_size, height, width, hidden_size)` 的初始隐藏状态（补丁嵌入），以供 Transformer 消费。

    Args:
        config:
            模型配置对象，包含补丁大小、通道数和嵌入维度等参数。
    """

    def __init__(self, config):
        super().__init__()
        patch_size = config.patch_size
        num_channels, hidden_size = config.num_channels, config.embed_dim
        self.num_channels = num_channels

        if patch_size == 4:
            pass
        else:
            # TODO: 支持任意的补丁大小。
            raise ValueError("Dinat 目前仅支持补丁大小为 4。")

        # 使用两个卷积层进行投影
        self.projection = nn.Sequential(
            nn.Conv2d(self.num_channels, hidden_size // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(hidden_size // 2, hidden_size, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        )

    def forward(self, pixel_values: Optional[torch.FloatTensor]) -> torch.Tensor:
        _, num_channels, height, width = pixel_values.shape
        # 检查像素值的通道维度是否与配置中设置的一致
        if num_channels != self.num_channels:
            raise ValueError(
                "确保像素值的通道维度与配置中设置的一致。"
            )
        # 应用投影来生成补丁嵌入，然后重新排列维度
        embeddings = self.projection(pixel_values)
        embeddings = embeddings.permute(0, 2, 3, 1)

        return embeddings


# 从 transformers.models.nat.modeling_nat.NatDownsampler 复制的类，将 Nat 替换为 Dinat
class DinatDownsampler(nn.Module):
    """
    卷积下采样层。

    Args:
        dim (`int`):
            输入通道数。
        norm_layer (`nn.Module`, *optional*, 默认为 `nn.LayerNorm`):
            归一化层类。
    """

    def __init__(self, dim: int, norm_layer: nn.Module = nn.LayerNorm) -> None:
        super().__init__()
        self.dim = dim
        # 使用卷积进行降维
        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # 应用归一化层
        self.norm = norm_layer(2 * dim)
    # 定义前向传播方法，接受一个形状为 [batch_size, height, width, channels] 的张量 input_feature，并返回一个形状相同的张量
    def forward(self, input_feature: torch.Tensor) -> torch.Tensor:
        # 调用 self.reduction 方法对输入张量进行维度变换，将通道维移到第二个位置，然后再次调用 permute 将通道维还原到最后一个位置
        input_feature = self.reduction(input_feature.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # 对变换后的张量 input_feature 进行规范化处理
        input_feature = self.norm(input_feature)
        # 返回处理后的张量 input_feature
        return input_feature
# 从transformers.models.beit.modeling_beit.drop_path复制而来
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    每个样本中应用在残差块主路径上的路径丢弃（随机深度）。

    注释由Ross Wightman提供：这与我为EfficientNet等网络创建的DropConnect实现相同，
    但原始名称有误导，因为'Drop Connect'是另一篇论文中不同形式的dropout...
    参见讨论：https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    我选择将层和参数名称更改为'drop path'，而不是混合使用DropConnect作为层名称并使用'survival rate'作为参数。
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # 适用于不同维度张量，而不仅仅是2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # 二值化
    output = input.div(keep_prob) * random_tensor
    return output


# 从transformers.models.beit.modeling_beit.BeitDropPath复制，将Beit改为Dinat
class DinatDropPath(nn.Module):
    """每个样本中应用在残差块主路径上的路径丢弃（随机深度）。"""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class NeighborhoodAttention(nn.Module):
    def __init__(self, config, dim, num_heads, kernel_size, dilation):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"隐藏大小（{dim}）不是注意力头数（{num_heads}）的整数倍"
            )

        self.num_attention_heads = num_heads
        self.attention_head_size = int(dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.kernel_size = kernel_size
        self.dilation = dilation

        # rpb是可学习的相对位置偏置；与Swin使用的概念相同。
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * self.kernel_size - 1), (2 * self.kernel_size - 1)))

        self.query = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    # 从transformers.models.nat.modeling_nat.NeighborhoodAttention.transpose_for_scores复制，将Nat改为Dinat
    # 将输入张量 x 进行形状转换，以便进行多头注意力计算
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 3, 1, 2, 4)

    # 实现 Transformer 的前向传播
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 通过 self.query、self.key、self.value 函数获取查询、键和值张量，并进行形状转换
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # 对查询张量应用缩放因子，以便在计算注意力权重之前缩放
        query_layer = query_layer / math.sqrt(self.attention_head_size)

        # 计算注意力分数，包括相对位置偏置
        attention_scores = natten2dqkrpb(query_layer, key_layer, self.rpb, self.kernel_size, self.dilation)

        # 将注意力分数归一化为注意力概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 使用 dropout 随机丢弃一部分注意力概率，这在 Transformer 中是标准做法
        attention_probs = self.dropout(attention_probs)

        # 计算上下文张量，结合注意力概率和值张量
        context_layer = natten2dav(attention_probs, value_layer, self.kernel_size, self.dilation)

        # 对上下文张量进行维度置换，以适应后续处理
        context_layer = context_layer.permute(0, 2, 3, 1, 4).contiguous()

        # 调整上下文张量的形状，以适应全头尺寸
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # 根据输出选项返回结果，包括上下文张量和（如果需要的话）注意力概率
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
# Copied from transformers.models.nat.modeling_nat.NeighborhoodAttentionOutput
class NeighborhoodAttentionOutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        # 定义一个全连接层，输入和输出维度都是 dim
        self.dense = nn.Linear(dim, dim)
        # 定义一个 Dropout 层，使用配置中的概率来丢弃注意力机制的概率
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将输入的 hidden_states 经过全连接层 dense
        hidden_states = self.dense(hidden_states)
        # 对经过全连接层后的 hidden_states 进行 Dropout 处理
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class NeighborhoodAttentionModule(nn.Module):
    def __init__(self, config, dim, num_heads, kernel_size, dilation):
        super().__init__()
        # 创建一个邻域注意力模块，使用给定的参数
        self.self = NeighborhoodAttention(config, dim, num_heads, kernel_size, dilation)
        # 创建一个输出层，将邻域注意力模块的输出映射到指定维度上
        self.output = NeighborhoodAttentionOutput(config, dim)
        # 初始化一个空的集合，用于存储被剪枝的注意力头索引
        self.pruned_heads = set()

    # Copied from transformers.models.nat.modeling_nat.NeighborhoodAttentionModule.prune_heads
    def prune_heads(self, heads):
        # 如果 heads 列表为空，则直接返回
        if len(heads) == 0:
            return
        # 查找可剪枝的注意力头和对应的索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储被剪枝的头
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # Copied from transformers.models.nat.modeling_nat.NeighborhoodAttentionModule.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 执行邻域注意力模块的前向传播
        self_outputs = self.self(hidden_states, output_attentions)
        # 将邻域注意力模块的输出传递给输出层，同时传入原始的 hidden_states
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出注意力权重，则将它们加入到输出中
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


# Copied from transformers.models.nat.modeling_nat.NatIntermediate with Nat->Dinat
class DinatIntermediate(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        # 定义一个线性层，将输入维度 dim 映射到 config.mlp_ratio * dim 的输出维度
        self.dense = nn.Linear(dim, int(config.mlp_ratio * dim))
        # 根据配置中的激活函数类型选择对应的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入的 hidden_states 经过线性层 dense
        hidden_states = self.dense(hidden_states)
        # 将线性层的输出经过选择的激活函数 intermediate_act_fn 处理
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
# 从transformers.models.nat.modeling_nat.NatOutput复制并将Nat->Dinat
class DinatOutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        # 使用线性层将输入维度映射到指定维度，mlp_ratio为配置参数
        self.dense = nn.Linear(int(config.mlp_ratio * dim), dim)
        # 使用指定的dropout概率创建一个dropout层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入的hidden_states通过线性层映射
        hidden_states = self.dense(hidden_states)
        # 对映射后的结果进行dropout处理
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class DinatLayer(nn.Module):
    def __init__(self, config, dim, num_heads, dilation, drop_path_rate=0.0):
        super().__init__()
        # 设置用于分块前馈的块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 设置卷积核大小
        self.kernel_size = config.kernel_size
        # 设置扩张率
        self.dilation = dilation
        # 计算窗口大小，是卷积核大小和扩张率的乘积
        self.window_size = self.kernel_size * self.dilation
        # 在LayerNorm之前应用LayerNorm进行归一化，eps是配置参数
        self.layernorm_before = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        # 使用NeighborhoodAttentionModule创建注意力层，config为配置参数
        self.attention = NeighborhoodAttentionModule(
            config, dim, num_heads, kernel_size=self.kernel_size, dilation=self.dilation
        )
        # 如果drop_path_rate大于0，创建DropPath层，否则创建Identity层
        self.drop_path = DinatDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        # 在LayerNorm之后应用LayerNorm进行归一化，eps是配置参数
        self.layernorm_after = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        # 创建DinatIntermediate层，处理中间状态
        self.intermediate = DinatIntermediate(config, dim)
        # 创建DinatOutput层，产生最终输出
        self.output = DinatOutput(config, dim)
        # 如果配置中的layer_scale_init_value大于0，则创建可训练参数，否则为None
        self.layer_scale_parameters = (
            nn.Parameter(config.layer_scale_init_value * torch.ones((2, dim)), requires_grad=True)
            if config.layer_scale_init_value > 0
            else None
        )

    def maybe_pad(self, hidden_states, height, width):
        # 获取当前窗口大小
        window_size = self.window_size
        # 默认填充值为0
        pad_values = (0, 0, 0, 0, 0, 0)
        # 如果输入的高度或宽度小于窗口大小，则进行填充
        if height < window_size or width < window_size:
            pad_l = pad_t = 0
            pad_r = max(0, window_size - width)
            pad_b = max(0, window_size - height)
            pad_values = (0, 0, pad_l, pad_r, pad_t, pad_b)
            # 对隐藏状态进行填充
            hidden_states = nn.functional.pad(hidden_states, pad_values)
        return hidden_states, pad_values

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
        #
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 获取隐藏状态的批量大小、高度、宽度和通道数
        batch_size, height, width, channels = hidden_states.size()
        # 保存隐藏状态的快捷方式
        shortcut = hidden_states

        # 对隐藏状态进行 layer normalization
        hidden_states = self.layernorm_before(hidden_states)
        # 如果隐藏状态小于卷积核大小乘以膨胀率，则进行填充
        hidden_states, pad_values = self.maybe_pad(hidden_states, height, width)

        # 获取填充后的高度和宽度
        _, height_pad, width_pad, _ = hidden_states.shape

        # 执行注意力机制，获取注意力输出
        attention_outputs = self.attention(hidden_states, output_attentions=output_attentions)

        # 从注意力输出中提取主要的注意力输出
        attention_output = attention_outputs[0]

        # 检查是否进行了填充
        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
            # 如果有填充，则裁剪注意力输出以匹配原始尺寸
            attention_output = attention_output[:, :height, :width, :].contiguous()

        # 如果存在层缩放参数，则应用第一个参数到注意力输出
        if self.layer_scale_parameters is not None:
            attention_output = self.layer_scale_parameters[0] * attention_output

        # 将注意力输出与快捷方式相加，应用 drop path 操作
        hidden_states = shortcut + self.drop_path(attention_output)

        # 对层输出进行 layer normalization
        layer_output = self.layernorm_after(hidden_states)
        # 经过中间层和输出层的处理
        layer_output = self.output(self.intermediate(layer_output))

        # 如果存在层缩放参数，则应用第二个参数到层输出
        if self.layer_scale_parameters is not None:
            layer_output = self.layer_scale_parameters[1] * layer_output

        # 将层输出与隐藏状态相加，再应用 drop path 操作
        layer_output = hidden_states + self.drop_path(layer_output)

        # 构造层输出元组，可能包含注意力权重
        layer_outputs = (layer_output, attention_outputs[1]) if output_attentions else (layer_output,)
        return layer_outputs
# 定义了一个名为 DinatStage 的自定义神经网络模块，继承自 nn.Module
class DinatStage(nn.Module):
    # 初始化函数，接收多个参数用于配置模块
    def __init__(self, config, dim, depth, num_heads, dilations, drop_path_rate, downsample):
        super().__init__()
        self.config = config  # 存储配置参数
        self.dim = dim  # 存储维度参数
        # 使用 nn.ModuleList 存储 DinatLayer 层的列表
        self.layers = nn.ModuleList(
            [
                DinatLayer(
                    config=config,
                    dim=dim,
                    num_heads=num_heads,
                    dilation=dilations[i],
                    drop_path_rate=drop_path_rate[i],
                )
                for i in range(depth)
            ]
        )

        # 如果 downsample 参数不为 None，则创建 downsample 层
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=nn.LayerNorm)
        else:
            self.downsample = None

        self.pointing = False  # 初始化 pointing 属性为 False

    # 重写 forward 方法，执行前向传播计算
    # 从 transformers.models.nat.modeling_nat.NatStage.forward 复制而来
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        _, height, width, _ = hidden_states.size()
        # 遍历 self.layers 列表中的每个 DinatLayer 层，依次计算输出
        for i, layer_module in enumerate(self.layers):
            layer_outputs = layer_module(hidden_states, output_attentions)
            hidden_states = layer_outputs[0]  # 更新 hidden_states

        hidden_states_before_downsampling = hidden_states
        # 如果存在 downsample 层，则对计算前的 hidden_states 进行下采样
        if self.downsample is not None:
            hidden_states = self.downsample(hidden_states_before_downsampling)

        # 返回计算后的 hidden_states 和计算前的 hidden_states_before_downsampling
        stage_outputs = (hidden_states, hidden_states_before_downsampling)

        # 如果需要输出注意力矩阵，则将其加入 stage_outputs 中
        if output_attentions:
            stage_outputs += layer_outputs[1:]
        return stage_outputs


# 定义了一个名为 DinatEncoder 的自定义神经网络模块，继承自 nn.Module
class DinatEncoder(nn.Module):
    # 初始化函数，接收配置参数 config
    def __init__(self, config):
        super().__init__()
        self.num_levels = len(config.depths)  # 计算深度级别数量
        self.config = config  # 存储配置参数
        # 根据配置参数创建多层 DinatStage 模块，并存储在 nn.ModuleList 中
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]
        self.levels = nn.ModuleList(
            [
                DinatStage(
                    config=config,
                    dim=int(config.embed_dim * 2**i_layer),
                    depth=config.depths[i_layer],
                    num_heads=config.num_heads[i_layer],
                    dilations=config.dilations[i_layer],
                    drop_path_rate=dpr[sum(config.depths[:i_layer]) : sum(config.depths[: i_layer + 1])],
                    downsample=DinatDownsampler if (i_layer < self.num_levels - 1) else None,
                )
                for i_layer in range(self.num_levels)
            ]
        )

    # 重写 forward 方法，执行前向传播计算
    # 从 transformers.models.nat.modeling_nat.NatEncoder.forward 复制而来，Nat->Dinat
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        output_hidden_states_before_downsampling: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, DinatEncoderOutput]:
        # 如果没有要输出的隐藏状态，则置空
        all_hidden_states = () if output_hidden_states else None
        # 如果没有要输出的重塑后的隐藏状态，则置空
        all_reshaped_hidden_states = () if output_hidden_states else None
        # 如果没有要输出的注意力权重，则置空
        all_self_attentions = () if output_attentions else None

        if output_hidden_states:
            # 重新排列隐藏状态的维度顺序：从 b h w c 到 b c h w
            reshaped_hidden_state = hidden_states.permute(0, 3, 1, 2)
            # 将当前隐藏状态添加到所有隐藏状态的元组中
            all_hidden_states += (hidden_states,)
            # 将重塑后的隐藏状态添加到所有重塑后的隐藏状态的元组中
            all_reshaped_hidden_states += (reshaped_hidden_state,)

        for i, layer_module in enumerate(self.levels):
            # 对每一层模块进行前向传播
            layer_outputs = layer_module(hidden_states, output_attentions)

            # 更新当前隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]
            # 如果需要输出隐藏状态且需要输出下采样前的隐藏状态
            hidden_states_before_downsampling = layer_outputs[1]

            if output_hidden_states and output_hidden_states_before_downsampling:
                # 重新排列下采样前的隐藏状态的维度顺序：从 b h w c 到 b c h w
                reshaped_hidden_state = hidden_states_before_downsampling.permute(0, 3, 1, 2)
                # 将下采样前的隐藏状态添加到所有隐藏状态的元组中
                all_hidden_states += (hidden_states_before_downsampling,)
                # 将重塑后的隐藏状态添加到所有重塑后的隐藏状态的元组中
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            elif output_hidden_states and not output_hidden_states_before_downsampling:
                # 重新排列当前隐藏状态的维度顺序：从 b h w c 到 b c h w
                reshaped_hidden_state = hidden_states.permute(0, 3, 1, 2)
                # 将当前隐藏状态添加到所有隐藏状态的元组中
                all_hidden_states += (hidden_states,)
                # 将重塑后的隐藏状态添加到所有重塑后的隐藏状态的元组中
                all_reshaped_hidden_states += (reshaped_hidden_state,)

            if output_attentions:
                # 将当前层的注意力权重添加到所有注意力权重的元组中
                all_self_attentions += layer_outputs[2:]

        if not return_dict:
            # 如果不返回字典，则返回非空值的元组
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        # 返回 DinatEncoderOutput 对象，包含最终的隐藏状态、所有隐藏状态、所有注意力权重和所有重塑后的隐藏状态
        return DinatEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            reshaped_hidden_states=all_reshaped_hidden_states,
        )
# DinatPreTrainedModel 类的子类，用于处理权重初始化以及下载和加载预训练模型的简单接口
class DinatPreTrainedModel(PreTrainedModel):

    # 模型的配置类，指定为 DinatConfig
    config_class = DinatConfig
    # 基础模型的前缀名称为 "dinat"
    base_model_prefix = "dinat"
    # 主要输入的名称为 "pixel_values"
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        """初始化模型的权重"""
        # 如果是 nn.Linear 或 nn.Conv2d 模块
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 使用正态分布初始化权重数据，标准差为 self.config.initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置项，则将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是 nn.LayerNorm 模块
        elif isinstance(module, nn.LayerNorm):
            # 将偏置项初始化为零
            module.bias.data.zero_()
            # 将权重初始化为全1
            module.weight.data.fill_(1.0)


# DINAT_START_DOCSTRING 是字符串常量，用于保存 DinatModel 类的文档字符串
DINAT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`DinatConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# DINAT_INPUTS_DOCSTRING 是字符串常量，用于保存 DinatModel 类的输入参数文档字符串
DINAT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`ViTImageProcessor.__call__`]
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

# add_start_docstrings 装饰器，用于给 DinatModel 类添加文档字符串
@add_start_docstrings(
    "The bare Dinat Model transformer outputting raw hidden-states without any specific head on top.",
    DINAT_START_DOCSTRING,
)
# DinatModel 类的定义，继承自 DinatPreTrainedModel 类
# 从 transformers.models.nat.modeling_nat.NatModel 复制而来，将 Nat 替换为 Dinat，NAT 替换为 DINAT
class DinatModel(DinatPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)

        requires_backends(self, ["natten"])  # 要求后端支持 "natten" 模块

        self.config = config  # 保存配置信息
        self.num_levels = len(config.depths)  # 确定金字塔层数
        self.num_features = int(config.embed_dim * 2 ** (self.num_levels - 1))  # 计算特征数量

        self.embeddings = DinatEmbeddings(config)  # 初始化嵌入层
        self.encoder = DinatEncoder(config)  # 初始化编码器

        self.layernorm = nn.LayerNorm(self.num_features, eps=config.layer_norm_eps)  # 初始化层归一化层
        self.pooler = nn.AdaptiveAvgPool1d(1) if add_pooling_layer else None  # 根据参数决定是否添加池化层

        # 初始化权重并进行最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings  # 返回输入嵌入

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)  # 剪枝模型的注意力头部

    @add_start_docstrings_to_model_forward(DINAT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=DinatModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, DinatModelOutput]:
        # 设置是否输出注意力权重，默认从模型配置中获取
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置是否输出隐藏状态，默认从模型配置中获取
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置是否返回字典格式的输出，默认从模型配置中获取
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果未提供像素值，则抛出数值错误异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 将像素值传入嵌入层进行处理
        embedding_output = self.embeddings(pixel_values)

        # 使用编码器处理嵌入输出
        encoder_outputs = self.encoder(
            embedding_output,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取编码器的序列输出，并进行 LayerNormalization
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        # 初始化池化输出为 None
        pooled_output = None
        # 如果模型有池化层，则对序列输出进行池化操作
        if self.pooler is not None:
            pooled_output = self.pooler(sequence_output.flatten(1, 2).transpose(1, 2))
            pooled_output = torch.flatten(pooled_output, 1)

        # 如果不要求以字典格式返回结果，则返回元组形式的输出
        if not return_dict:
            output = (sequence_output, pooled_output) + encoder_outputs[1:]
            return output

        # 否则，以自定义的输出对象形式返回结果，包括最后的隐藏状态、池化输出以及各层的隐藏状态和注意力权重
        return DinatModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            reshaped_hidden_states=encoder_outputs.reshaped_hidden_states,
        )
@add_start_docstrings(
    """
    Dinat Model transformer with an image classification head on top (a linear layer on top of the final hidden state
    of the [CLS] token) e.g. for ImageNet.
    """,
    DINAT_START_DOCSTRING,
)
class DinatForImageClassification(DinatPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 检查后端库是否已经加载
        requires_backends(self, ["natten"])

        # 设置分类任务的类别数目
        self.num_labels = config.num_labels
        # 初始化 DinatModel 模型
        self.dinat = DinatModel(config)

        # 分类器头部
        self.classifier = (
            nn.Linear(self.dinat.num_features, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(DINAT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=DinatImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, DinatImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 确保返回字典存在，如果未提供则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用自注意力网络模型（DINAT），传入像素值和其他选项参数
        outputs = self.dinat(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 提取汇聚后的输出特征
        pooled_output = outputs[1]

        # 将汇聚后的特征输入分类器，生成预测 logits
        logits = self.classifier(pooled_output)

        # 初始化损失值为 None
        loss = None
        # 如果提供了标签
        if labels is not None:
            # 如果问题类型未定义，则根据标签类型设置问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型计算损失
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

        # 如果不要求返回字典，则组装输出元组
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 DINAT 图像分类器输出对象，包括损失、logits、隐藏状态、注意力等
        return DinatImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            reshaped_hidden_states=outputs.reshaped_hidden_states,
        )
# 使用装饰器添加文档字符串，描述了这个类是一个用于如DETR和MaskFormer等框架的NAT骨干。
# 这里继承了DinatPreTrainedModel和BackboneMixin类。
class DinatBackbone(DinatPreTrainedModel, BackboneMixin):
    def __init__(self, config):
        super().__init__(config)
        # 调用父类的初始化方法，传递配置对象给父类
        super()._init_backbone(config)

        # 确保所需的后端库存在
        requires_backends(self, ["natten"])

        # 初始化嵌入层和编码器
        self.embeddings = DinatEmbeddings(config)
        self.encoder = DinatEncoder(config)

        # 计算每个阶段的特征维度列表，这些维度是根据配置的嵌入维度和深度计算得出的
        self.num_features = [config.embed_dim] + [int(config.embed_dim * 2**i) for i in range(len(config.depths))]

        # 为输出特征的隐藏状态添加层归一化
        hidden_states_norms = {}
        for stage, num_channels in zip(self._out_features, self.channels):
            hidden_states_norms[stage] = nn.LayerNorm(num_channels)
        self.hidden_states_norms = nn.ModuleDict(hidden_states_norms)

        # 执行后续的权重初始化和最终处理
        self.post_init()

    # 获取输入嵌入层的方法
    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    # 使用装饰器添加文档字符串，描述了这个方法的输入参数和输出类型
    # 并替换返回值的文档字符串，指定输出类型为BackboneOutput，配置类为_CONFIG_FOR_DOC
    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        """
        如果 return_dict 参数为 None，则使用 self.config.use_return_dict 决定返回值类型
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        """
        如果 output_hidden_states 参数为 None，则使用 self.config.output_hidden_states 决定是否输出隐藏状态
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        """
        如果 output_attentions 参数为 None，则使用 self.config.output_attentions 决定是否输出注意力权重
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        """
        使用 self.embeddings 将 pixel_values 转换为嵌入输出
        """
        embedding_output = self.embeddings(pixel_values)

        """
        使用 self.encoder 对嵌入输出进行编码，设置输出选项和返回值类型为字典
        """
        outputs = self.encoder(
            embedding_output,
            output_attentions=output_attentions,
            output_hidden_states=True,
            output_hidden_states_before_downsampling=True,
            return_dict=True,
        )

        """
        从编码器输出中获取重塑后的隐藏状态
        """
        hidden_states = outputs.reshaped_hidden_states

        """
        初始化空的特征图列表
        """
        feature_maps = ()
        """
        遍历阶段名称和隐藏状态，将符合条件的特征图添加到列表中
        """
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                batch_size, num_channels, height, width = hidden_state.shape
                hidden_state = hidden_state.permute(0, 2, 3, 1).contiguous()
                hidden_state = hidden_state.view(batch_size, height * width, num_channels)
                hidden_state = self.hidden_states_norms[stage](hidden_state)
                hidden_state = hidden_state.view(batch_size, height, width, num_channels)
                hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()
                feature_maps += (hidden_state,)

        """
        如果不需要返回字典形式的结果，则将特征图和可能的隐藏状态组成元组返回
        """
        if not return_dict:
            output = (feature_maps,)
            if output_hidden_states:
                output += (outputs.hidden_states,)
            return output

        """
        否则，返回包含特征图、隐藏状态和注意力权重的 BackboneOutput 对象
        """
        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )
```