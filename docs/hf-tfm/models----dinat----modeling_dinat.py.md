# `.\models\dinat\modeling_dinat.py`

```py
# 设置编码为utf-8
# 版权声明
# 根据Apache许可证2.0版授权，您可以按照许可证的规定使用这个文件
# 您可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，本软件按"原样"（AS IS）基础发布，没有任何担保或条件，无论是明示的还是隐含的
# 查看具体语言方面授权和限制的许可证
# PyTorch Dilated Neighborhood Attention Transformer模型

# 导入模块
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
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

# 判断是否含有natten依赖库
if is_natten_available():
    from natten.functional import natten2dav, natten2dqkrpb
else:

    def natten2dqkrpb(*args, **kwargs):
        raise OptionalDependencyNotAvailable()

    def natten2dav(*args, **kwargs):
        raise OptionalDependencyNotAvailable()

# 日志记录
logger = logging.get_logger(__name__)

# 通用文档字符串
_CONFIG_FOR_DOC = "DinatConfig"

# 基础文档字符串
_CHECKPOINT_FOR_DOC = "shi-labs/dinat-mini-in1k-224"
_EXPECTED_OUTPUT_SHAPE = [1, 7, 7, 512]

# 图像分类文档字符串
_IMAGE_CLASS_CHECKPOINT = "shi-labs/dinat-mini-in1k-224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

# Dinat预训练模型存档列表
DINAT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "shi-labs/dinat-mini-in1k-224",
    # 查看所有Dinat模型https://huggingface.co/models?filter=dinat
]

# drop_path 和 DinatDropPath 来源于timm库.
    # 定义函数参数及其类型注释
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`.
    
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each stage) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
    
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, hidden_size, height, width)`.
    
            Hidden-states of the model at the output of each layer plus the initial embedding outputs reshaped to
            include the spatial dimensions.
    """
    
    # 初始化变量并指定类型注释
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
# 使用 dataclass 装饰器创建一个数据类 DinatModelOutput，继承自 ModelOutput
# 该类是从 transformers.models.nat.modeling_nat.NatModelOutput 复制而来，将 Nat 替换为 Dinat
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

    # 定义各种成员变量，及其类型，初始值为 None 或 Optional
    last_hidden_state: torch.FloatTensor = None
    pooler_output: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


# 使用 dataclass 装饰器创建一个数据类 DinatImageClassifierOutput，继承自 ModelOutput
# 该类是从 transformers.models.nat.modeling_nat.NatImageClassifierOutput 复制而来，将 Nat 替换为 Dinat
class DinatImageClassifierOutput(ModelOutput):
    """
    Dinat outputs for image classification.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.  # 定义loss变量，表示分类（或回归如果config.num_labels==1）损失，当提供`labels`参数时返回
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).  # 定义logits变量，表示分类（或回归如果config.num_labels==1）分数（SoftMax前）
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`.  # 隐藏状态，模型各层输出的元组（一个用于嵌入输出，一个用于每个阶段输出）的形状为`(batch_size, sequence_length, hidden_size)`
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True` is passed):
            Tuple of `torch.FloatTensor` (one for each stage) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.  # 注意力权重，self-attention中用于计算加权平均值的元组，形状为`(batch_size, num_heads, sequence_length, sequence_length)`
        reshaped_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True` is passed):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, hidden_size, height, width)`.  # 重塑后的隐藏状态，模型各层输出包括初始嵌入输出的元组，重塑以包括空间维度，形状为`(batch_size, hidden_size, height, width)`
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    reshaped_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
# 从 transformers.models.nat.modeling_nat.NatEmbeddings 复制而来，将 Nat 改为 Dinat
class DinatEmbeddings(nn.Module):
    """
    构建补丁和位置嵌入。
    """

    def __init__(self, config):
        super().__init__()

        # 使用配置文件创建 Dinat 补丁嵌入对象
        self.patch_embeddings = DinatPatchEmbeddings(config)

        # 使用配置文件中的嵌入维度创建 LayerNorm 层
        self.norm = nn.LayerNorm(config.embed_dim)
        # 使用配置文件中的隐藏丢弃率创建 Dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, pixel_values: Optional[torch.FloatTensor]) -> Tuple[torch.Tensor]:
        # 使用 DinatPatchEmbeddings 对象对输入的 pixel_values 进行嵌入
        embeddings = self.patch_embeddings(pixel_values)
        # 对嵌入结果进行归一化处理
        embeddings = self.norm(embeddings)
        # 在嵌入结果上应用 dropout
        embeddings = self.dropout(embeddings)
        # 返回嵌入结果
        return embeddings


# 从 transformers.models.nat.modeling_nat.NatPatchEmbeddings 复制而来，将 Nat 改为 Dinat
class DinatPatchEmbeddings(nn.Module):
    """
    此类将形状为 `(batch_size, num_channels, height, width)` 的 `pixel_values` 转换为初始 `hidden_states`（补丁嵌入），
    形状为 `(batch_size, height, width, hidden_size)`，供 Transformer 使用。
    """

    def __init__(self, config):
        super().__init__()
        # 从配置文件中获取补丁尺寸、通道数和嵌入维度
        patch_size = config.patch_size
        num_channels, hidden_size = config.num_channels, config.embed_dim
        self.num_channels = num_channels

        # 检查补丁尺寸，如果不是 4，则抛出错误
        if patch_size == 4:
            pass
        else:
            # TODO: 支持任意补丁尺寸
            raise ValueError("Dinat 目前只支持补丁尺寸为 4。")

        # 使用两个卷积层创建投影序列，分别进行尺寸缩减和通道数变化
        self.projection = nn.Sequential(
            nn.Conv2d(self.num_channels, hidden_size // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(hidden_size // 2, hidden_size, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        )

    def forward(self, pixel_values: Optional[torch.FloatTensor]) -> torch.Tensor:
        # 解包 pixel_values 的形状，获取通道数、高度和宽度
        _, num_channels, height, width = pixel_values.shape
        # 检查输入的通道数是否与配置文件中的通道数匹配
        if num_channels != self.num_channels:
            raise ValueError(
                "确保像素值的通道数与配置文件中设置的通道数匹配。"
            )
        # 使用投影序列对 pixel_values 进行转换
        embeddings = self.projection(pixel_values)
        # 将维度调整为 (batch_size, height, width, hidden_size)
        embeddings = embeddings.permute(0, 2, 3, 1)
        # 返回嵌入结果
        return embeddings


# 从 transformers.models.nat.modeling_nat.NatDownsampler 复制而来，将 Nat 改为 Dinat
class DinatDownsampler(nn.Module):
    """
    卷积下采样层。

    参数：
        dim (`int`):
            输入通道数。
        norm_layer (`nn.Module`, *可选*，默认为 `nn.LayerNorm`):
            归一化层类。
    """

    def __init__(self, dim: int, norm_layer: nn.Module = nn.LayerNorm) -> None:
        super().__init__()
        # 设置输入通道数
        self.dim = dim
        # 使用输入通道数、卷积核参数和步长创建卷积层，实现通道扩展和下采样
        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # 创建归一化层，对输出通道数进行归一化
        self.norm = norm_layer(2 * dim)
    # 前向传播方法，接收一个输入特征张量并返回处理后的特征张量
    def forward(self, input_feature: torch.Tensor) -> torch.Tensor:
        # 将输入特征张量转换为通道维度在第3维度上的排列，并通过reduction层降维
        input_feature = self.reduction(input_feature.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # 对降维后的特征进行归一化处理
        input_feature = self.norm(input_feature)
        # 返回处理后的特征张量
        return input_feature
# 从transformers.models.beit.modeling_beit.drop_path中拷贝过来的函数
# 对输入进行 Stochastic Depth 操作，即随机深度
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    根据每个样本丢弃路径（应用于残差块的主路径）。
    
    Ross Wightman的注释：这与我为EfficientNet等网络创建的DropConnect实现相同，
    但是原始名称是误导性的，因为'Drop Connect'是另一篇论文中的一种不同形式的dropout...
    请参阅讨论：https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... 我选择改变
    层和参数名为'drop path'，而不是将DropConnect作为层名并使用'survival rate'作为参数。
    """
    # 如果丢弃概率为0或者不是在训练状态，则直接返回输入数据
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # 适用于不同维度张量，而不仅仅是2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # 二值化
    output = input.div(keep_prob) * random_tensor
    return output


# 从transformers.models.beit.modeling_beit.BeitDropPath中拷贝过来的类，将Beit->Dinat
class DinatDropPath(nn.Module):
    """对每个样本进行路径丢弃（应用于残差块的主路径）。"""
    
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
                f"The hidden size ({dim}) is not a multiple of the number of attention heads ({num_heads})"
            )

        self.num_attention_heads = num_heads
        self.attention_head_size = int(dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.kernel_size = kernel_size
        self.dilation = dilation

        # rpb是可学习的相对位置偏差；Swin中使用了相同的概念。
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * self.kernel_size - 1), (2 * self.kernel_size - 1)))

        self.query = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    # 从transformers.models.nat.modeling_nat.NeighborhoodAttention.transpose_for_scores中拷贝过来的代码，将Nat->Dinat
    # 将输入张量 x 进行维度变换，使其符合多头注意力机制的输入要求
    new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
    x = x.view(new_x_shape)
    # 返回维度变换后的张量，调换了部分维度顺序
    return x.permute(0, 3, 1, 2, 4)

    # 前向传播函数，接收隐藏状态张量和是否输出注意力权重的标志位
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 对隐藏状态应用 self.query、self.key、self.value 层，然后对结果进行维度变换，以匹配注意力计算的要求
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # 对 query_layer 进行缩放，以提高计算效率，同时不改变计算结果
        query_layer = query_layer / math.sqrt(self.attention_head_size)

        # 计算注意力分数，使用自定义的函数，结合相对位置偏置
        attention_scores = natten2dqkrpb(query_layer, key_layer, self.rpb, self.kernel_size, self.dilation)

        # 对注意力分数进行 softmax 归一化，得到注意力权重
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 对注意力权重进行 dropout 操作，用于随机屏蔽部分注意力
        attention_probs = self.dropout(attention_probs)

        # 根据注意力权重和 value_layer 计算上下文层，使用自定义函数
        context_layer = natten2dav(attention_probs, value_layer, self.kernel_size, self.dilation)
        # 调整上下文层的维度顺序，使其符合后续计算的要求
        context_layer = context_layer.permute(0, 2, 3, 1, 4).contiguous()
        # 对上下文层进行维度变换，使其符合多头注意力机制的输出要求
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # 根据是否输出注意力权重，返回不同的结果
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
# 定义一个邻域注意力输出模块
class NeighborhoodAttentionOutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        # 定义一个线性层，用于处理隐藏状态
        self.dense = nn.Linear(dim, dim)
        # 定义一个dropout层，用于处理注意力概率
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用线性层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 使用dropout层处理处理后的隐藏状态
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# 定义一个邻域注意力模块
class NeighborhoodAttentionModule(nn.Module):
    def __init__(self, config, dim, num_heads, kernel_size, dilation):
        super().__init__()
        # 初始化自注意力层
        self.self = NeighborhoodAttention(config, dim, num_heads, kernel_size, dilation)
        # 初始化输出层
        self.output = NeighborhoodAttentionOutput(config, dim)
        # 初始化一个空集合用于存储被修剪的注意力头部
        self.pruned_heads = set()

    # 定义修剪注意力头部的方法
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 查找可修剪的头部和索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 修剪线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储修剪的头部
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 定义前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 使用自注意力层处理隐藏状态
        self_outputs = self.self(hidden_states, output_attentions)
        # 使用输出层处理注意力输出
        attention_output = self.output(self_outputs[0], hidden_states)
        # 将注意力输出与注意力矩阵拼接，如果有输出注意力的话
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


# 定义一个DinatIntermediate类，修改自然注意力中间层为动态邻域注意力中间层
class DinatIntermediate(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        # 定义一个线性层，将隐藏状态映射到更低维度
        self.dense = nn.Linear(dim, int(config.mlp_ratio * dim))
        # 如果激活函数是字符串形式，则从预定义字典中获取相应的激活函数；否则直接使用配置中的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用线性层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 使用激活函数处理线性层输出
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
# 从transformers.models.nat.modeling_nat.NatOutput复制而来，将Nat->Dinat
class DinatOutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        # 全连接层，将MLP的输出维度调整为dim
        self.dense = nn.Linear(int(config.mlp_ratio * dim), dim)
        # Dropout层，用于防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 经过全连接层
        hidden_states = self.dense(hidden_states)
        # 经过Dropout层
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class DinatLayer(nn.Module):
    def __init__(self, config, dim, num_heads, dilation, drop_path_rate=0.0):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.kernel_size = config.kernel_size
        self.dilation = dilation
        self.window_size = self.kernel_size * self.dilation
        # 归一化层，用于层间归一化
        self.layernorm_before = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        # 自注意力模块
        self.attention = NeighborhoodAttentionModule(
            config, dim, num_heads, kernel_size=self.kernel_size, dilation=self.dilation
        )
        # DropPath层，用于随机失活一部分路径
        self.drop_path = DinatDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        # 归一化层，用于层间归一化
        self.layernorm_after = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        # Dinat中间层
        self.intermediate = DinatIntermediate(config, dim)
        # Dinat输出层
        self.output = DinatOutput(config, dim)
        # 如果配置了层比例参数，则初始化，用于缩放层间输出
        self.layer_scale_parameters = (
            nn.Parameter(config.layer_scale_init_value * torch.ones((2, dim)), requires_grad=True)
            if config.layer_scale_init_value > 0
            else None
        )

    def maybe_pad(self, hidden_states, height, width):
        window_size = self.window_size
        pad_values = (0, 0, 0, 0, 0, 0)
        # 如果输入的高度或宽度小于窗口大小，进行填充
        if height < window_size or width < window_size:
            pad_l = pad_t = 0
            pad_r = max(0, window_size - width)
            pad_b = max(0, window_size - height)
            pad_values = (0, 0, pad_l, pad_r, pad_t, pad_b)
            hidden_states = nn.functional.pad(hidden_states, pad_values)
        return hidden_states, pad_values

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
    # 定义函数，接受torch张量并返回元组类型的结果
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 获取隐藏层状态的批量大小、高度、宽度和通道数
        batch_size, height, width, channels = hidden_states.size()
        # 保存隐藏层状态的快捷方式
        shortcut = hidden_states

        # 在隐藏状态之前进行层归一化处理
        hidden_states = self.layernorm_before(hidden_states)
        # 如果隐藏状态小于卷积核大小乘以膨胀率，则进行填充
        hidden_states, pad_values = self.maybe_pad(hidden_states, height, width)

        # 获取填充后的高度和宽度
        _, height_pad, width_pad, _ = hidden_states.shape

        # 应用注意力机制
        attention_outputs = self.attention(hidden_states, output_attentions=output_attentions)

        # 获取注意力输出
        attention_output = attention_outputs[0]

        # 判断是否进行了填充
        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
            attention_output = attention_output[:, :height, :width, :].contiguous()

        # 如果存在层比例参数，则对注意力输出进行缩放
        if self.layer_scale_parameters is not None:
            attention_output = self.layer_scale_parameters[0] * attention_output

        # 将快捷方式和注意力输出相加，并应用dropout路径
        hidden_states = shortcut + self.drop_path(attention_output)

        # 对层输出进行层归一化处理
        layer_output = self.layernorm_after(hidden_states)
        # 对层输出进行输出层和中间层的处理
        layer_output = self.output(self.intermediate(layer_output))

        # 如果存在层比例参数，则对层输出进行缩放
        if self.layer_scale_parameters is not None:
            layer_output = self.layer_scale_parameters[1] * layer_output

        # 对层输出应用快捷方式和dropout路径处理
        layer_output = hidden_states + self.drop_path(layer_output)

        # 如果输出包括注意力，则返回注意力输出和层输出，否则只返回层输出
        layer_outputs = (layer_output, attention_outputs[1]) if output_attentions else (layer_output,)
        return layer_outputs
# 定义一个名为 DinatStage 的类，继承自 nn.Module
class DinatStage(nn.Module):
    # 初始化方法，接收多个参数：config、dim、depth、num_heads、dilations、drop_path_rate、downsample
    def __init__(self, config, dim, depth, num_heads, dilations, drop_path_rate, downsample):
        # 调用父类的初始化方法
        super().__init__()
        # 将参数赋给实例变量
        self.config = config
        self.dim = dim
        # 创建 nn.ModuleList，包含多个 DinatLayer 对象
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

        # 如果 downsample 参数不为 None，则创建 downsample 对象，否则为 None
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=nn.LayerNorm)
        else:
            self.downsample = None

        # 初始化 pointing 为 False
        self.pointing = False

    # 定义 forward 方法，接收参数 hidden_states: torch.Tensor 和 output_attentions: Optional[bool]
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 获取 hidden_states 的尺寸信息
        _, height, width, _ = hidden_states.size()
        # 遍历 self.layers 中的每个层，对 hidden_states 进行处理
        for i, layer_module in enumerate(self.layers):
            layer_outputs = layer_module(hidden_states, output_attentions)
            hidden_states = layer_outputs[0]

        # 将未下采样的 hidden_states 保存为 hidden_states_before_downsampling
        hidden_states_before_downsampling = hidden_states
        # 如果 self.downsample 不为 None，则对 hidden_states 进行下采样
        if self.downsample is not None:
            hidden_states = self.downsample(hidden_states_before_downsampling)

        # 将处理后的 hidden_states 和未下采样的 hidden_states 做为 stage_outputs 返回
        stage_outputs = (hidden_states, hidden_states_before_downsampling)

        # 如果 output_attentions 为 True，则在返回值中加入 layer_outputs[1:]
        if output_attentions:
            stage_outputs += layer_outputs[1:]
        return stage_outputs


# 定义一个名为 DinatEncoder 的类，继承自 nn.Module
class DinatEncoder(nn.Module):
    # 初始化方法，接收参数 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 获取 config.depths 的长度，并赋给实例变量 self.num_levels
        self.num_levels = len(config.depths)
        # 将参数 config 赋给实例变量 self.config
        self.config = config
        # 创建一个 dpr 列表，包含多个 torch.linspace() 返回值的整数部分，用于控制剔除的概率
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]
        # 创建 nn.ModuleList，包含多个 DinatStage 对象
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

    # 定义 forward 方法，接收多个参数：hidden_states、output_attentions、output_hidden_states、output_hidden_states_before_downsampling、return_dict
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        output_hidden_states_before_downsampling: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    # 函数声明，接受一个输入参数，返回一个 Tuple 或 DinatEncoderOutput 对象
    ) -> Union[Tuple, DinatEncoderOutput]:
        # 如果不输出隐藏状态，则初始化空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果不输出隐藏状态，则初始化空元组
        all_reshaped_hidden_states = () if output_hidden_states else None
        # 如果不输出注意力权重，则初始化空元组
        all_self_attentions = () if output_attentions else None

        # 如果需要输出隐藏状态
        if output_hidden_states:
            # 重新排列隐藏状态的维度顺序，从 b h w c 改为 b c h w
            reshaped_hidden_state = hidden_states.permute(0, 3, 1, 2)
            all_hidden_states += (hidden_states,)  # 将隐藏状态添加到元组中
            all_reshaped_hidden_states += (reshaped_hidden_state,)

        # 遍历每一层的模块
        for i, layer_module in enumerate(self.levels):
            # 调用每一层模块，获取输出结果
            layer_outputs = layer_module(hidden_states, output_attentions)

            # 更新隐藏状态和下采样前隐藏状态
            hidden_states = layer_outputs[0]
            hidden_states_before_downsampling = layer_outputs[1]

            # 如果输出隐藏状态和下采样前隐藏状态
            if output_hidden_states and output_hidden_states_before_downsampling:
                # 重新排列下采样前隐藏状态的维度顺序，从 b h w c 改为 b c h w
                reshaped_hidden_state = hidden_states_before_downsampling.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states_before_downsampling,)  # 将下采样前隐藏状态添加到元组中
                all_reshaped_hidden_states += (reshaped_hidden_state,)
            # 如果输出隐藏状态但不输出下采样前隐藏状态
            elif output_hidden_states and not output_hidden_states_before_downsampling:
                # 重新排列隐藏状态的维度顺序，从 b h w c 改为 b c h w
                reshaped_hidden_state = hidden_states.permute(0, 3, 1, 2)
                all_hidden_states += (hidden_states,)  # 将隐藏状态添加到元组中
                all_reshaped_hidden_states += (reshaped_hidden_state,)

            # 如果需要输出注意力权重
            if output_attentions:
                all_self_attentions += layer_outputs[2:]  # 将注意力权重添加到元组中

        # 如果不需要返回字典
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)  # 返回由不为None的值组成的元组

        # 如果需要返回字典
        return DinatEncoderOutput(
            last_hidden_state=hidden_states,  # 设置最后一个隐藏状态
            hidden_states=all_hidden_states,  # 设置所有隐藏状态
            attentions=all_self_attentions,  # 设置所有注意力权重
            reshaped_hidden_states=all_reshaped_hidden_states,  # 设置所有重排的隐藏状态
        )   # 返回 DinatEncoderOutput 对象
# 定义一个继承自PreTrainedModel的抽象类，用于处理权重初始化以及下载和加载预训练模型的简单接口
class DinatPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # Dinat模型的配置类
    config_class = DinatConfig
    # 模型的前缀
    base_model_prefix = "dinat"
    # 主输入名称
    main_input_name = "pixel_values"

    # 初始化权重的方法
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果是线性层或卷积层
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置项，初始化为0
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是LayerNorm层
        elif isinstance(module, nn.LayerNorm):
            # 初始化偏置为0，权重为1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


# Dinat模型的文档字符串
DINAT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`DinatConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# Dinat模型的输入文档字符串
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

# 添加文档字符串注释
@add_start_docstrings(
    "The bare Dinat Model transformer outputting raw hidden-states without any specific head on top.",
    DINAT_START_DOCSTRING,
)
# 从transformers.models.nat.modeling_nat.NatModel复制并修改为DinatModel
class DinatModel(DinatPreTrainedModel):
    # 初始化函数，设置配置和是否添加池化层
    def __init__(self, config, add_pooling_layer=True):
        # 调用父类的初始化函数
        super().__init__(config)

        # 检查是否需要后端支持
        requires_backends(self, ["natten"])

        # 保存配置信息
        self.config = config
        # 计算特征数量
        self.num_levels = len(config.depths)
        self.num_features = int(config.embed_dim * 2 ** (self.num_levels - 1))

        # 创建嵌入层和编码器
        self.embeddings = DinatEmbeddings(config)
        self.encoder = DinatEncoder(config)

        # 创建 LayerNorm 层和池化层
        self.layernorm = nn.LayerNorm(self.num_features, eps=config.layer_norm_eps)
        self.pooler = nn.AdaptiveAvgPool1d(1) if add_pooling_layer else None

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    # 剪枝模型的注意力头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 前向传播函数
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
    # 定义函数的输入参数和返回类型
    ) -> Union[Tuple, DinatModelOutput]:
        # 如果未指定是否输出注意力权重，则使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定是否输出隐藏状态，则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定是否返回字典，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果未提供像素值，则抛出数值错误
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 将像素值传入嵌入层
        embedding_output = self.embeddings(pixel_values)

        # 将嵌入输出传入编码器
        encoder_outputs = self.encoder(
            embedding_output,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取编码器输出的序列输出
        sequence_output = encoder_outputs[0]
        # 对序列输出进行 LayerNormalization
        sequence_output = self.layernorm(sequence_output)

        # 初始化池化输出为 None
        pooled_output = None
        # 如果存在池化层
        if self.pooler is not None:
            # 对序列输出进行展平、转置和池化
            pooled_output = self.pooler(sequence_output.flatten(1, 2).transpose(1, 2))
            # 对池化输出进行展平
            pooled_output = torch.flatten(pooled_output, 1)

        # 如果不返回字典
        if not return_dict:
            # 构建输出元组
            output = (sequence_output, pooled_output) + encoder_outputs[1:]
            # 返回输出元组
            return output

        # 返回 DinatModelOutput 对象，包括最后隐藏状态、池化输出、隐藏状态和注意力权重
        return DinatModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            reshaped_hidden_states=encoder_outputs.reshaped_hidden_states,
        )
# 使用 Dinat 模型变换器，在顶部添加一个图像分类头部（即在 [CLS] 标记的最终隐藏状态之上的线性层），例如用于 ImageNet。
@add_start_docstrings(
    """
    Dinat Model transformer with an image classification head on top (a linear layer on top of the final hidden state
    of the [CLS] token) e.g. for ImageNet.
    """,
    DINAT_START_DOCSTRING,
)
class DinatForImageClassification(DinatPreTrainedModel):
    # 初始化函数，接受配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)

        # 检查是否需要后端支持 "natten"
        requires_backends(self, ["natten"])

        # 获取标签数量
        self.num_labels = config.num_labels
        # 创建 Dinat 模型
        self.dinat = DinatModel(config)

        # 分类器头部
        self.classifier = (
            nn.Linear(self.dinat.num_features, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数
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
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 设置返回字典，如果未提供则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 DINO 模型进行前向传播
        outputs = self.dinat(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取池化后的输出
        pooled_output = outputs[1]

        # 使用分类器获取 logits
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
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

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return DinatImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            reshaped_hidden_states=outputs.reshaped_hidden_states,
        )
# 添加起始文档字符串，描述该模型是用于类似DETR和MaskFormer等框架的NAT骨干
# DINAT_START_DOCSTRING是另一个文档字符串的一部分
class DinatBackbone(DinatPreTrainedModel, BackboneMixin):
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 初始化骨干网络
        super()._init_backbone(config)

        # 检查是否需要后端支持natten
        requires_backends(self, ["natten"])

        # 初始化嵌入层和编码器
        self.embeddings = DinatEmbeddings(config)
        self.encoder = DinatEncoder(config)
        # 计算特征数
        self.num_features = [config.embed_dim] + [int(config.embed_dim * 2**i) for i in range(len(config.depths)]

        # 为输出特征的隐藏状态添加层归一化
        hidden_states_norms = {}
        for stage, num_channels in zip(self._out_features, self.channels):
            hidden_states_norms[stage] = nn.LayerNorm(num_channels)
        self.hidden_states_norms = nn.ModuleDict(hidden_states_norms)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    # 添加模型前向方法的起始文档字符串
    # DINAT_INPUTS_DOCSTRING是另一个文档字符串的一部分
    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> BackboneOutput:
        """
        返回给定输入的 BackboneOutput 对象

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = AutoImageProcessor.from_pretrained("shi-labs/nat-mini-in1k-224")
        >>> model = AutoBackbone.from_pretrained(
        ...     "shi-labs/nat-mini-in1k-224", out_features=["stage1", "stage2", "stage3", "stage4"]
        ... )

        >>> inputs = processor(image, return_tensors="pt")

        >>> outputs = model(**inputs)

        >>> feature_maps = outputs.feature_maps
        >>> list(feature_maps[-1].shape)
        [1, 512, 7, 7]
        ```py"""
        # 如果 return_dict 不为 None，则使用 return_dict，否则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果 output_hidden_states 不为 None，则使用 output_hidden_states，否则使用 self.config.output_hidden_states
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果 output_attentions 不为 None，则使用 output_attentions，否则使用 self.config.output_attentions
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        # 使用输入的像素值计算嵌入输出
        embedding_output = self.embeddings(pixel_values)

        # 使用编码器处理嵌入输出
        outputs = self.encoder(
            embedding_output,
            output_attentions=output_attentions,
            output_hidden_states=True,
            output_hidden_states_before_downsampling=True,
            return_dict=True,
        )

        # 获取处理后的隐藏状态
        hidden_states = outputs.reshaped_hidden_states

        # 初始化特征图
        feature_maps = ()
        # 遍历阶段和隐藏状态，处理隐藏状态并添加到特征图中
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                batch_size, num_channels, height, width = hidden_state.shape
                hidden_state = hidden_state.permute(0, 2, 3, 1).contiguous()
                hidden_state = hidden_state.view(batch_size, height * width, num_channels)
                hidden_state = self.hidden_states_norms[stage](hidden_state)
                hidden_state = hidden_state.view(batch_size, height, width, num_channels)
                hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()
                feature_maps += (hidden_state,)

        # 如果不需要返回字典，则返回特征图和隐藏状态
        if not return_dict:
            output = (feature_maps,)
            if output_hidden_states:
                output += (outputs.hidden_states,)
            return output

        # 返回 BackboneOutput 对象
        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )
```