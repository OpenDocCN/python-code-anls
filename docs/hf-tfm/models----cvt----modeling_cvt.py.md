# `.\models\cvt\modeling_cvt.py`

```py
# 设置文件编码为UTF-8
# 版权声明
# 根据Apache许可证2.0版本使用此文件
# 你不得使用本文件，除非符合许可证的条件
# 你可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按"原样"分发软件
# 没有任何形式的担保或条件，明示或默示
# 请查看许可证以了解特定语言规定的权限和限制。

""" PyTorch CvT model."""

# 导入所需模块
import collections.abc
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入所需函数
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_outputs import ImageClassifierOutputWithNoAttention, ModelOutput
from ...modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import logging
from .configuration_cvt import CvtConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# General docstring
# 总体文档字符串
_CONFIG_FOR_DOC = "CvtConfig"

# Base docstring
# 基本文档字符串
_CHECKPOINT_FOR_DOC = "microsoft/cvt-13"
_EXPECTED_OUTPUT_SHAPE = [1, 384, 14, 14]

# Image classification docstring
# 图像分类文档字符串
_IMAGE_CLASS_CHECKPOINT = "microsoft/cvt-13"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

# 预训练模型存档列表
CVT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/cvt-13",
    "microsoft/cvt-13-384",
    "microsoft/cvt-13-384-22k",
    "microsoft/cvt-21",
    "microsoft/cvt-21-384",
    "microsoft/cvt-21-384-22k",
    # 查看所有Cvt模型请访问https://huggingface.co/models?filter=cvt
]

@dataclass
class BaseModelOutputWithCLSToken(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        cls_token_value (`torch.FloatTensor` of shape `(batch_size, 1, hidden_size)`):
            Classification token at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
    """

    last_hidden_state: torch.FloatTensor = None
    cls_token_value: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
# Copied from transformers.models.beit.modeling_beit.drop_path
# 定义函数 drop_path，用于执行随机深度（Stochastic Depth）的路径丢弃
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    # 如果丢弃概率为 0 或者未进行训练，则直接返回输入
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    # 计算随机张量的形状，与输入张量的形状相同
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    # 生成随机张量
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    # 执行路径丢弃操作，并返回结果
    output = input.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.beit.modeling_beit.BeitDropPath
# 定义类 CvtDropPath，用于执行随机深度的路径丢弃
class CvtDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 调用 drop_path 函数执行随机深度的路径丢弃，并返回结果
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class CvtEmbeddings(nn.Module):
    """
    Construct the CvT embeddings.
    """
    # 构建了 CvT 嵌入层的类
    def __init__(self, patch_size, num_channels, embed_dim, stride, padding, dropout_rate):
        super().__init__()
        # 初始化卷积嵌入层
        self.convolution_embeddings = CvtConvEmbeddings(
            patch_size=patch_size, num_channels=num_channels, embed_dim=embed_dim, stride=stride, padding=padding
        )
        # 初始化 dropout 层
        self.dropout = nn.Dropout(dropout_rate)

    # 执行前向传播操作
    def forward(self, pixel_values):
        # 获取卷积嵌入层的输出
        hidden_state = self.convolution_embeddings(pixel_values)
        # 对卷积嵌入层的输出进行 dropout 操作
        hidden_state = self.dropout(hidden_state)
        return hidden_state


class CvtConvEmbeddings(nn.Module):
    """
    Image to Conv Embedding.
    """
    # 定义了将图像转换为卷积嵌入的类
    def __init__(self, patch_size, num_channels, embed_dim, stride, padding):
        super().__init__()
        # 将 patch_size 转换为元组形式
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        self.patch_size = patch_size
        # 定义卷积投影层
        self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        # 定义归一化层
        self.normalization = nn.LayerNorm(embed_dim)
    # 定义一个前向传播函数，接受像素值作为输入
    def forward(self, pixel_values):
        # 使用投影函数处理像素值
        pixel_values = self.projection(pixel_values)
        # 获取输入像素值的形状信息：批量大小、通道数、高度、宽度
        batch_size, num_channels, height, width = pixel_values.shape
        # 计算隐藏层大小
        hidden_size = height * width
        # 重新排列像素值张量维度："b c h w -> b (h w) c"
        pixel_values = pixel_values.view(batch_size, num_channels, hidden_size).permute(0, 2, 1)
        # 如果开启了标准化操作
        if self.normalization:
            # 对像素值进行标准化处理
            pixel_values = self.normalization(pixel_values)
        # 重新排列像素值张量维度："b (h w) c" -> "b c h w"
        pixel_values = pixel_values.permute(0, 2, 1).view(batch_size, num_channels, height, width)
        # 返回处理后的像素值张量
        return pixel_values
# 自注意力模型的卷积投影层，用于将输入特征映射到更高维空间
class CvtSelfAttentionConvProjection(nn.Module):
    def __init__(self, embed_dim, kernel_size, padding, stride):
        super().__init__()
        # 创建一个二维卷积层，用于对输入进行卷积操作
        self.convolution = nn.Conv2d(
            embed_dim,  # 输入通道数
            embed_dim,  # 输出通道数
            kernel_size=kernel_size,  # 卷积核大小
            padding=padding,  # 填充大小
            stride=stride,  # 步长大小
            bias=False,  # 是否使用偏置
            groups=embed_dim,  # 输入通道分组
        )
        # 创建一个二维批归一化层，用于规范化卷积输出
        self.normalization = nn.BatchNorm2d(embed_dim)

    def forward(self, hidden_state):
        # 对输入特征进行卷积操作
        hidden_state = self.convolution(hidden_state)
        # 对卷积输出进行批归一化
        hidden_state = self.normalization(hidden_state)
        return hidden_state


# 自注意力模型的线性投影层，用于将二维特征映射到一维空间
class CvtSelfAttentionLinearProjection(nn.Module):
    def forward(self, hidden_state):
        # 获取输入特征的形状信息
        batch_size, num_channels, height, width = hidden_state.shape
        # 计算输入特征的大小
        hidden_size = height * width
        # 将二维特征重排为一维特征
        hidden_state = hidden_state.view(batch_size, num_channels, hidden_size).permute(0, 2, 1)
        return hidden_state


# 自注意力模型的投影层，用于将输入特征映射到适合进行自注意力计算的维度
class CvtSelfAttentionProjection(nn.Module):
    def __init__(self, embed_dim, kernel_size, padding, stride, projection_method="dw_bn"):
        super().__init__()
        # 根据投影方法选择对应的投影层
        if projection_method == "dw_bn":
            self.convolution_projection = CvtSelfAttentionConvProjection(embed_dim, kernel_size, padding, stride)
        # 创建线性投影层
        self.linear_projection = CvtSelfAttentionLinearProjection()

    def forward(self, hidden_state):
        # 使用卷积投影层将输入特征映射到更高维空间
        hidden_state = self.convolution_projection(hidden_state)
        # 使用线性投影层将特征映射到一维空间
        hidden_state = self.linear_projection(hidden_state)
        return hidden_state


# 自注意力模型的主体结构，用于计算自注意力权重并应用于输入特征
class CvtSelfAttention(nn.Module):
    def __init__(
        self,
        num_heads,
        embed_dim,
        kernel_size,
        padding_q,
        padding_kv,
        stride_q,
        stride_kv,
        qkv_projection_method,
        qkv_bias,
        attention_drop_rate,
        with_cls_token=True,
        **kwargs,
        ):
        # 调用父类的构造函数
        super().__init__()
        # 定义缩放因子为 embed_dim 的负 0.5 次方
        self.scale = embed_dim**-0.5
        # 是否包含类别令牌
        self.with_cls_token = with_cls_token
        # 嵌入维度
        self.embed_dim = embed_dim
        # 注意力头的数量
        self.num_heads = num_heads

        # 创建查询的卷积投影
        self.convolution_projection_query = CvtSelfAttentionProjection(
            embed_dim,
            kernel_size,
            padding_q,
            stride_q,
            projection_method="linear" if qkv_projection_method == "avg" else qkv_projection_method,
        )
        # 创建键的卷积投影
        self.convolution_projection_key = CvtSelfAttentionProjection(
            embed_dim, kernel_size, padding_kv, stride_kv, projection_method=qkv_projection_method
        )
        # 创建值的卷积投影
        self.convolution_projection_value = CvtSelfAttentionProjection(
            embed_dim, kernel_size, padding_kv, stride_kv, projection_method=qkv_projection_method
        )

        # 线性转换，用于查询
        self.projection_query = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        # 线性转换，用于键
        self.projection_key = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        # 线性转换，用于值
        self.projection_value = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)

        # 用于注意力的丢弃层
        self.dropout = nn.Dropout(attention_drop_rate)

    # 将隐藏状态重排以适应多头注意力
    def rearrange_for_multi_head_attention(self, hidden_state):
        # 获取批大小、隐藏大小和注意力头的维度
        batch_size, hidden_size, _ = hidden_state.shape
        # 计算每个注意力头的维度
        head_dim = self.embed_dim // self.num_heads
        # 重排张量维度，使其符合多头注意力的输入格式
        # 'b t (h d) -> b h t d'
        return hidden_state.view(batch_size, hidden_size, self.num_heads, head_dim).permute(0, 2, 1, 3)
    # 定义前向传播函数，用于执行自注意力机制的操作
    def forward(self, hidden_state, height, width):
        # 如果模型包含分类标记，则将隐藏状态分割为分类标记和其余部分
        if self.with_cls_token:
            cls_token, hidden_state = torch.split(hidden_state, [1, height * width], 1)
        # 获取批量大小、隐藏状态维度和通道数
        batch_size, hidden_size, num_channels = hidden_state.shape
        # 将隐藏状态重新排列为"b (h w) c -> b c h w"的形状
        hidden_state = hidden_state.permute(0, 2, 1).view(batch_size, num_channels, height, width)

        # 计算查询、键和值的投影
        key = self.convolution_projection_key(hidden_state)
        query = self.convolution_projection_query(hidden_state)
        value = self.convolution_projection_value(hidden_state)

        # 如果模型包含分类标记，则将分类标记连接到查询、键和值中
        if self.with_cls_token:
            query = torch.cat((cls_token, query), dim=1)
            key = torch.cat((cls_token, key), dim=1)
            value = torch.cat((cls_token, value), dim=1)

        # 计算每个头部的维度
        head_dim = self.embed_dim // self.num_heads

        # 将查询、键和值投影到多头注意力的形状
        query = self.rearrange_for_multi_head_attention(self.projection_query(query))
        key = self.rearrange_for_multi_head_attention(self.projection_key(key))
        value = self.rearrange_for_multi_head_attention(self.projection_value(value))

        # 计算注意力分数
        attention_score = torch.einsum("bhlk,bhtk->bhlt", [query, key]) * self.scale
        # 对注意力分数进行 softmax 操作
        attention_probs = torch.nn.functional.softmax(attention_score, dim=-1)
        # 对注意力概率进行 dropout 操作
        attention_probs = self.dropout(attention_probs)

        # 计算上下文向量
        context = torch.einsum("bhlt,bhtv->bhlv", [attention_probs, value])
        # 重新排列上下文向量的形状为"b h t d -> b t (h d)"
        _, _, hidden_size, _ = context.shape
        context = context.permute(0, 2, 1, 3).contiguous().view(batch_size, hidden_size, self.num_heads * head_dim)
        # 返回上下文向量
        return context
class CvtSelfOutput(nn.Module):
    """
    The residual connection is defined in CvtLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """
    # 定义CvtSelfOutput类，用于处理自注意力机制模块的输出
    def __init__(self, embed_dim, drop_rate):
        # 初始化函数，接收嵌入维度和丢弃率作为参数
        super().__init__()
        # 调用父类构造函数
        self.dense = nn.Linear(embed_dim, embed_dim)
        # 创建线性层，处理嵌入维度到嵌入维度的转换
        self.dropout = nn.Dropout(drop_rate)
        # 创建丢弃层，用于丢弃一部分神经元

    def forward(self, hidden_state, input_tensor):
        # 前向传播函数，接收隐藏状态和输入张量作为参数
        hidden_state = self.dense(hidden_state)
        # 使用线性层处理隐藏状态
        hidden_state = self.dropout(hidden_state)
        # 使用丢弃层丢弃一部分神经元
        return hidden_state
        # 返回处理后的隐藏状态


class CvtAttention(nn.Module):
    #定义CvtAttention类，用于处理自注意力机制模块
    def __init__(
        self,
        num_heads,
        embed_dim,
        kernel_size,
        padding_q,
        padding_kv,
        stride_q,
        stride_kv,
        qkv_projection_method,
        qkv_bias,
        attention_drop_rate,
        drop_rate,
        with_cls_token=True,
    ):
        #初始化函数，接收多个参数，包括注意力头数、嵌入维度等
        super().__init__()
        #调用父类构造函数
        self.attention = CvtSelfAttention(
            num_heads,
            embed_dim,
            kernel_size,
            padding_q,
            padding_kv,
            stride_q,
            stride_kv,
            qkv_projection_method,
            qkv_bias,
            attention_drop_rate,
            with_cls_token,
        )
        #创建自注意力机制模块
        self.output = CvtSelfOutput(embed_dim, drop_rate)
        #创建自身输出模块
        self.pruned_heads = set()
        #初始化剪枝头集合

    def prune_heads(self, heads):
        #定义剪枝函数，接收头数作为参数
        if len(heads) == 0:
            return
        #如果头数为0，则直接返回
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )
        #找到可剪枝头的索引
        # 剪枝线性层
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        #更新参数并存储被剪枝的头索引

        #更新超参数并存储被剪枝的头
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_state, height, width):
        #定义前向传播函数，接收隐藏状态、高度和宽度作为参数
        self_output = self.attention(hidden_state, height, width)
        #使用自注意力模块处理隐藏状态
        attention_output = self.output(self_output, hidden_state)
        #使用自身输出模块处理自注意力输出
        return attention_output
        #返回注意力输出结果


class CvtIntermediate(nn.Module):
    #定义CvtIntermediate类，用于处理中间状态的模块
    def __init__(self, embed_dim, mlp_ratio):
        #初始化函数，接收嵌入维度和多层感知机比率作为参数
        super().__init__()
        #调用父类构造函数
        self.dense = nn.Linear(embed_dim, int(embed_dim * mlp_ratio))
        #创建线性层，处理嵌入维度到多层感知机输出维度的转换
        self.activation = nn.GELU()
        #创建GELU激活函数

    def forward(self, hidden_state):
        #定义前向传播函数，接收隐藏状态作为参数
        hidden_state = self.dense(hidden_state)
        #使用线性层处理隐藏状态
        hidden_state = self.activation(hidden_state)
        #使用激活函数处理中间状态
        return hidden_state
        #返回处理后的中间状态
    # 初始化函数，设置模型参数，包括嵌入维度、MLP 比例和丢弃率
    def __init__(self, embed_dim, mlp_ratio, drop_rate):
        # 继承父类的初始化函数
        super().__init__()
        # 创建线性层，将输入维度乘以 MLP 比例后映射为嵌入维度
        self.dense = nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        # 创建丢弃层，以指定丢弃率进行丢弃
        self.dropout = nn.Dropout(drop_rate)

    # 前向传播函数，接收隐藏状态和输入张量作为参数
    def forward(self, hidden_state, input_tensor):
        # 通过线性层对隐藏状态进行映射
        hidden_state = self.dense(hidden_state)
        # 通过丢弃层对映射后的隐藏状态进行丢弃
        hidden_state = self.dropout(hidden_state)
        # 将丢弃后的隐藏状态与输入张量相加得到结果
        hidden_state = hidden_state + input_tensor
        # 返回计算结果
        return hidden_state
# 定义一个CvtLayer类，包含了注意力层、归一化和多层感知器（mlp）
class CvtLayer(nn.Module):
    """
    CvtLayer composed by attention layers, normalization and multi-layer perceptrons (mlps).
    """

    def __init__(
        self,
        num_heads,
        embed_dim,
        kernel_size,
        padding_q,
        padding_kv,
        stride_q,
        stride_kv,
        qkv_projection_method,
        qkv_bias,
        attention_drop_rate,
        drop_rate,
        mlp_ratio,
        drop_path_rate,
        with_cls_token=True,
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化注意力层
        self.attention = CvtAttention(
            num_heads,
            embed_dim,
            kernel_size,
            padding_q,
            padding_kv,
            stride_q,
            stride_kv,
            qkv_projection_method,
            qkv_bias,
            attention_drop_rate,
            drop_rate,
            with_cls_token,
        )

        # 初始化中间层
        self.intermediate = CvtIntermediate(embed_dim, mlp_ratio)
        # 初始化输出层
        self.output = CvtOutput(embed_dim, mlp_ratio, drop_rate)
        # 初始化DropPath
        self.drop_path = CvtDropPath(drop_prob=drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        # 初始化之前的LayerNorm
        self.layernorm_before = nn.LayerNorm(embed_dim)
        # 初始化之后的LayerNorm
        self.layernorm_after = nn.LayerNorm(embed_dim)

    def forward(self, hidden_state, height, width):
        # 使用注意力层处理隐藏状态
        self_attention_output = self.attention(
            self.layernorm_before(hidden_state),  # in Cvt, layernorm is applied before self-attention
            height,
            width,
        )
        attention_output = self_attention_output
        attention_output = self.drop_path(attention_output)

        # 第一个残差连接
        hidden_state = attention_output + hidden_state

        # 在Cvt中，也在自注意力之后应用LayerNorm
        layer_output = self.layernorm_after(hidden_state)
        layer_output = self.intermediate(layer_output)

        # 在这里完成第二个残差连接
        layer_output = self.output(layer_output, hidden_state)
        layer_output = self.drop_path(layer_output)
        return layer_output


class CvtStage(nn.Module):
    # 初始化函数，接受配置和阶段参数
    def __init__(self, config, stage):
        # 调用父类的初始化函数
        super().__init__()
        # 存储配置和阶段参数
        self.config = config
        self.stage = stage
        # 如果配置中包含分类标记，并且该阶段需要分类标记
        if self.config.cls_token[self.stage]:
            # 初始化分类标记为一个可学习的参数，大小为(1, 1, embed_dim[-1])
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.config.embed_dim[-1]))

        # 初始化嵌入层
        self.embedding = CvtEmbeddings(
            patch_size=config.patch_sizes[self.stage],  # 补丁大小
            stride=config.patch_stride[self.stage],    # 步幅
            num_channels=config.num_channels if self.stage == 0 else config.embed_dim[self.stage - 1],  # 输入通道数
            embed_dim=config.embed_dim[self.stage],    # 嵌入维度
            padding=config.patch_padding[self.stage],  # 填充
            dropout_rate=config.drop_rate[self.stage],  # 丢弃率
        )

        # 计算每个层的丢弃路径率
        drop_path_rates = [x.item() for x in torch.linspace(0, config.drop_path_rate[self.stage], config.depth[stage])]

        # 初始化层组成的序列
        self.layers = nn.Sequential(
            *[
                CvtLayer(
                    num_heads=config.num_heads[self.stage],  # 头数
                    embed_dim=config.embed_dim[self.stage],  # 嵌入维度
                    kernel_size=config.kernel_qkv[self.stage],  # QKV核大小
                    padding_q=config.padding_q[self.stage],    # Q填充
                    padding_kv=config.padding_kv[self.stage],  # KV填充
                    stride_kv=config.stride_kv[self.stage],    # KV步幅
                    stride_q=config.stride_q[self.stage],      # Q步幅
                    qkv_projection_method=config.qkv_projection_method[self.stage],  # QKV投影方法
                    qkv_bias=config.qkv_bias[self.stage],      # QKV偏置
                    attention_drop_rate=config.attention_drop_rate[self.stage],  # 注意力丢弃率
                    drop_rate=config.drop_rate[self.stage],    # 丢弃率
                    drop_path_rate=drop_path_rates[self.stage],  # 丢弃路径率
                    mlp_ratio=config.mlp_ratio[self.stage],    # MLP比率
                    with_cls_token=config.cls_token[self.stage],  # 是否使用分类标记
                )
                for _ in range(config.depth[self.stage])  # 根据深度创建CvtLayer
            ]
        )

    # 前向传播函数
    def forward(self, hidden_state):
        cls_token = None
        # 嵌入输入的隐藏状态
        hidden_state = self.embedding(hidden_state)
        batch_size, num_channels, height, width = hidden_state.shape
        # 重排形状以适应层的输入要求
        hidden_state = hidden_state.view(batch_size, num_channels, height * width).permute(0, 2, 1)
        # 如果需要分类标记
        if self.config.cls_token[self.stage]:
            # 将分类标记扩展到与隐藏状态相同的批大小
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            # 将分类标记与隐藏状态拼接起来
            hidden_state = torch.cat((cls_token, hidden_state), dim=1)

        # 逐层进行前向传播
        for layer in self.layers:
            # 调用每个层的前向传播函数
            layer_outputs = layer(hidden_state, height, width)
            hidden_state = layer_outputs

        # 如果需要分类标记
        if self.config.cls_token[self.stage]:
            # 将隐藏状态分割为分类标记和其余部分
            cls_token, hidden_state = torch.split(hidden_state, [1, height * width], 1)
        # 重排形状以适应输出的形状
        hidden_state = hidden_state.permute(0, 2, 1).view(batch_size, num_channels, height, width)
        # 返回隐藏状态和分类标记（如果有）
        return hidden_state, cls_token
## 注释：


# CvtEncoder 类继承自 nn.Module 类，用于将图像转换为隐含表示
class CvtEncoder(nn.Module):
    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 保存配置对象
        self.config = config
        # 存储 CvtStage 对象的列表
        self.stages = nn.ModuleList([])
        # 遍历配置的深度，并为每个深度创建一个 CvtStage 对象，然后添加到 stages 列表中
        for stage_idx in range(len(config.depth)):
            self.stages.append(CvtStage(config, stage_idx))

    # 前向传播方法
    def forward(self, pixel_values, output_hidden_states=False, return_dict=True):
        # 如果设置为输出隐藏状态，则创建一个空的元组用于保存所有隐藏状态
        all_hidden_states = () if output_hidden_states else None
        # 初始化隐藏状态为像素值
        hidden_state = pixel_values

        # 初始化 cls_token 为 None
        cls_token = None
        # 遍历 stages 列表
        for _, (stage_module) in enumerate(self.stages):
            # 调用 CvtStage 对象的前向传播方法，更新隐藏状态和 cls_token
            hidden_state, cls_token = stage_module(hidden_state)
            # 如果设置为输出隐藏状态，则保存隐藏状态到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

        # 如果 return_dict 为 False，则返回一个元组，其中包含非空的隐藏状态，cls_token 和 all_hidden_states
        if not return_dict:
            return tuple(v for v in [hidden_state, cls_token, all_hidden_states] if v is not None)

        # 如果 return_dict 为 True，则返回一个 BaseModelOutputWithCLSToken 对象，其中包含最后的隐藏状态，cls_token 和 all_hidden_states
        return BaseModelOutputWithCLSToken(
            last_hidden_state=hidden_state,
            cls_token_value=cls_token,
            hidden_states=all_hidden_states,
        )


# CvtPreTrainedModel 类继承自 PreTrainedModel 类，用于处理权重初始化和预训练模型的下载和加载
class CvtPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 配置类是 CvtConfig 类
    config_class = CvtConfig
    # 基础模型前缀是 "cvt"
    base_model_prefix = "cvt"
    # 主输入名称是 "pixel_values"
    main_input_name = "pixel_values"

    # 初始化权重的方法
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果是线性层或卷积层，初始化权重和偏置
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是层归一化层，初始化权重为 1.0，偏置为 0.0
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        # 如果是 CvtStage 对象，初始化 cls_token
        elif isinstance(module, CvtStage):
            if self.config.cls_token[module.stage]:
                module.cls_token.data = nn.init.trunc_normal_(
                    torch.zeros(1, 1, self.config.embed_dim[-1]), mean=0.0, std=self.config.initializer_range
                )


# CVT_START_DOCSTRING 是一个多行字符串，用于描述 CVTPreTrainedModel 类
CVT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`CvtConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# CVT_INPUTS_DOCSTRING 是一个多行字符串，用于描述 CVTPreTrainedModel 类的参数
CVT_INPUTS_DOCSTRING = r"""
    # 参数说明：
    # pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
    #     像素值。可以使用 [`AutoImageProcessor`] 获取像素值。详见 [`CvtImageProcessor.__call__`]。
    # output_hidden_states (`bool`, *optional*):
    #     是否返回所有层的隐藏状态。更多细节见返回的张量中的 `hidden_states`。
    # return_dict (`bool`, *optional*):
    #     是否返回 [`~file_utils.ModelOutput`] 而不是一个普通的元组。
"""
Wrap the Cvt Model transformer with additional functionalities and documentation.

class CvtModel(CvtPreTrainedModel):
    """
    Initialize the Cvt Model transformer with the given configuration and optional pooling layer.
    """
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.encoder = CvtEncoder(config)  # Initialize the encoder using the provided configuration
        self.post_init()  # Perform additional post initialization steps

    """
    Prune heads of the model based on the provided dictionary of layer numbers and heads to prune in each layer.
    """
    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    """
    Forward method to process the input pixel values and return the model outputs.
    """
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithCLSToken]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        encoder_outputs = self.encoder(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutputWithCLSToken(
            last_hidden_state=sequence_output,
            cls_token_value=encoder_outputs.cls_token_value,
            hidden_states=encoder_outputs.hidden_states,
        )


class CvtForImageClassification(CvtPreTrainedModel):
    """
    Initialize the Cvt Model transformer with an image classification head on top for tasks such as ImageNet.
    """
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.cvt = CvtModel(config, add_pooling_layer=False)  # Initialize the Cvt Model without a pooling layer
        self.layernorm = nn.LayerNorm(config.embed_dim[-1])  # Apply layer normalization
        # Classifier head
        self.classifier = (
            nn.Linear(config.embed_dim[-1], config.num_labels) if config.num_labels > 0 else nn.Identity()
        )  # Initialize the classifier based on the number of labels

        # Initialize weights and apply final processing
        self.post_init()  # Perform additional post initialization steps
    # 使用装饰器为代码添加文档字符串，指定了代码的一些元数据
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    # 定义一个方法，用于模型的前向推断
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,  # 输入的像素值，默认为 None
        labels: Optional[torch.Tensor] = None,         # 目标标签，默认为 None
        output_hidden_states: Optional[bool] = None,   # 是否输出隐藏状态，默认为 None
        return_dict: Optional[bool] = None,            # 是否返回字典，默认为 None
    ) -> Union[Tuple, ImageClassifierOutputWithNoAttention]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 判断是否要返回数据字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 使用图片像素值作为输入，进行模型预测
        outputs = self.cvt(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取最后一个隐藏层的输出和cls token的输出
        sequence_output = outputs[0]
        cls_token = outputs[1]
        # 判断是否需要把cls token融入到输出序列中
        if self.config.cls_token[-1]:
            sequence_output = self.layernorm(cls_token)
        else:
            batch_size, num_channels, height, width = sequence_output.shape
            # 重排列 "b c h w -> b (h w) c"
            sequence_output = sequence_output.view(batch_size, num_channels, height * width).permute(0, 2, 1)
            sequence_output = self.layernorm(sequence_output)

        # 对序列输出进行平均
        sequence_output_mean = sequence_output.mean(dim=1)
        # 将平均后的输出传入分类器
        logits = self.classifier(sequence_output_mean)

        loss = None
        # 计算损失函数
        if labels is not None:
            # 判断问题类型
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.config.num_labels == 1:
                    # 计算回归损失
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                # 计算单标签分类损失
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                # 计算多标签分类损失
                loss = loss_fct(logits, labels)

        # 根据是否要返回数据字典进行返回
        if not return_dict:
            # 返回 (logits, hidden_states...)
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回预测结果和损失值的数据字典
        return ImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)
```