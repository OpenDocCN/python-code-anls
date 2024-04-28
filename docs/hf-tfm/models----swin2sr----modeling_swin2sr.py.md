# `.\transformers\models\swin2sr\modeling_swin2sr.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 这个代码版权归 2022 年微软研究和 HuggingFace 公司团队所有
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可；
# 除非符合许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 根据“按原样”基础分发，
# 没有任何形式的明示或暗示担保或条件。
# 有关特定语言的权限，请参阅许可证。
""" PyTorch Swin2SR Transformer 模型。"""

# 导入所需模块和库
import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

# 导入自定义模块和类
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageSuperResolutionOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_swin2sr import Swin2SRConfig

# 设置日志
logger = logging.get_logger(__name__)

# 一般文档字符串
_CONFIG_FOR_DOC = "Swin2SRConfig"

# 基本文档字符串
_CHECKPOINT_FOR_DOC = "caidas/swin2SR-classical-sr-x2-64"
_EXPECTED_OUTPUT_SHAPE = [1, 180, 488, 648]

# Swin2SR 预训练模型列表
SWIN2SR_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "caidas/swin2SR-classical-sr-x2-64",
    # 查看所有 Swin2SR 模型 https://huggingface.co/models?filter=swin2sr
]

# Swin2SR 编码器输出类
@dataclass
class Swin2SREncoderOutput(ModelOutput):
    """
    Swin2SR 编码器的输出，可能包含隐藏状态和注意力权重。

    参数:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的隐藏状态序列。
        hidden_states (`tuple(torch.FloatTensor)`, *可选*, 当 `output_hidden_states=True` 时返回或当 `config.output_hidden_states=True` 时返回):
            包含模型每一层的隐藏状态的元组，形状为 `(batch_size, sequence_length, hidden_size)`。

            每一层的隐藏状态和初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *可选*, 当 `output_attentions=True` 时返回或当 `config.output_attentions=True` 时返回):
            每一阶段的注意力权重的元组，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            在自注意力头中用于计算加权平均值的注意力权重。
    """
    # 定义名为 last_hidden_state 的变量，类型为 torch.FloatTensor，初始赋值为 None
    last_hidden_state: torch.FloatTensor = None
    # 定义名为 hidden_states 的变量，类型为可选的 Tuple[torch.FloatTensor]，初始赋值为 None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义名为 attentions 的变量，类型为可选的 Tuple[torch.FloatTensor]，初始赋值为 None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# 从transformers.models.swin.modeling_swin.window_partition中复制的函数
def window_partition(input_feature, window_size):
    """
    将给定的输入分割成窗口。
    """
    # 获取输入特征的批次大小、高度、宽度和通道数
    batch_size, height, width, num_channels = input_feature.shape
    # 将输入特征重塑为指定窗口大小的窗口
    input_feature = input_feature.view(
        batch_size, height // window_size, window_size, width // window_size, window_size, num_channels
    )
    # 对窗口进行排列以便后续处理，并将其重塑为合适的形状
    windows = input_feature.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, num_channels)
    return windows


# 从transformers.models.swin.modeling_swin.window_reverse中复制的函数
def window_reverse(windows, window_size, height, width):
    """
    合并窗口以产生更高分辨率的特征。
    """
    # 获取窗口的通道数
    num_channels = windows.shape[-1]
    # 将窗口重塑为合适的形状
    windows = windows.view(-1, height // window_size, width // window_size, window_size, window_size, num_channels)
    # 对窗口进行排列以便后续处理，并将其重塑为原始输入的形状
    windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, height, width, num_channels)
    return windows


# 从transformers.models.beit.modeling_beit.drop_path中复制的函数
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    对每个样本进行路径丢弃（随机深度）（当应用于残差块的主路径时）。

    评论由Ross Wightman提供：这与我为EfficientNet等网络创建的DropConnect实现相同，
    但是，原始名称具有误导性，因为'Drop Connect'是另一篇论文中的不同形式的dropout......
    请参阅讨论：https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956......
    我选择将层名称和参数名称更改为'drop path'，而不是将DropConnect作为层名称混合使用，并使用'survival rate'作为参数。
    """
    # 如果丢失概率为0或不在训练模式，则直接返回输入
    if drop_prob == 0.0 or not training:
        return input
    # 计算保留的概率
    keep_prob = 1 - drop_prob
    # 计算随机张量以进行二值化
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # 适应不同维度的张量，而不仅仅是2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # 二值化
    # 应用丢弃路径并返回结果
    output = input.div(keep_prob) * random_tensor
    return output


# 从transformers.models.swin.modeling_swin.SwinDropPath更名为Swin2SRDropPath
class Swin2SRDropPath(nn.Module):
    """对每个样本进行路径丢弃（随机深度）（当应用于残差块的主路径时）。"""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class Swin2SREmbeddings(nn.Module):
    """
    构造补丁和可选的位置嵌入。
    """
    ``` 
    # 初始化函数，接受配置参数，并调用父类的初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
    
        # 创建Swin2SRPatchEmbeddings对象，用于处理嵌入的补丁
        self.patch_embeddings = Swin2SRPatchEmbeddings(config)
        # 获取补丁的数量
        num_patches = self.patch_embeddings.num_patches
    
        # 如果配置中使用绝对嵌入，则创建位置嵌入参数，否则设为None
        if config.use_absolute_embeddings:
            self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))
        else:
            self.position_embeddings = None
    
        # 创建一个Dropout层，用于在模型中添加随机失活
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 获取窗口大小
        self.window_size = config.window_size
    
    # 前向传播函数，接受像素值并返回嵌入和输出维度
    def forward(self, pixel_values: Optional[torch.FloatTensor]) -> Tuple[torch.Tensor]:
        # 使用Swin2SRPatchEmbeddings处理像素值获取嵌入和输出维度
        embeddings, output_dimensions = self.patch_embeddings(pixel_values)
    
        # 如果存在位置嵌入参数，则将其加到嵌入中
        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings
    
        # 对嵌入进行随机失活处理
        embeddings = self.dropout(embeddings)
    
        # 返回处理后的嵌入和输出维度
        return embeddings, output_dimensions
class Swin2SRPatchEmbeddings(nn.Module):
    # 定义 Swin2SRPatchEmbeddings 类，继承自 nn.Module
    def __init__(self, config, normalize_patches=True):
        # 初始化函数，接受 config 和 normalize_patches 参数
        super().__init__()
        # 调用父类的初始化函数
        num_channels = config.embed_dim
        # 从 config 中获取 embed_dim 赋值给 num_channels
        image_size, patch_size = config.image_size, config.patch_size
        # 从 config 中获取 image_size 和 patch_size 赋值给 image_size 和 patch_size

        # 处理 image_size 和 patch_size
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        patches_resolution = [image_size[0] // patch_size[0], image_size[1] // patch_size[1]]
        # 计算 patches 的分辨率
        self.patches_resolution = patches_resolution
        # 保存 patches 分辨率
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        # 计算 patches 数量

        # 创建卷积层用于投影
        self.projection = nn.Conv2d(num_channels, config.embed_dim, kernel_size=patch_size, stride=patch_size)
        # 使用 nn.Conv2d 定义投影层
        self.layernorm = nn.LayerNorm(config.embed_dim) if normalize_patches else None
        # 根据 normalize_patches 参数定义归一化层或者为 None

    def forward(self, embeddings: Optional[torch.FloatTensor]) -> Tuple[torch.Tensor, Tuple[int]]:
        # 前向传播函数，输入为 embeddings，返回类型为 Tuple[torch.Tensor, Tuple[int]]
        embeddings = self.projection(embeddings)
        # 使用投影层处理输入
        _, _, height, width = embeddings.shape
        # 获取 embeddings 的高度和宽度
        output_dimensions = (height, width)
        # 保存输出的高度和宽度
        embeddings = embeddings.flatten(2).transpose(1, 2)
        # 将 embeddings 进行扁平化和转置

        if self.layernorm is not None:
            embeddings = self.layernorm(embeddings)
            # 如果有归一化层，则对 embeddings 进行归一化

        return embeddings, output_dimensions
        # 返回处理后的 embeddings 和输出的尺寸


class Swin2SRPatchUnEmbeddings(nn.Module):
    r"""Image to Patch Unembedding"""

    def __init__(self, config):
        # 初始化函数，接受 config 参数
        super().__init__()

        self.embed_dim = config.embed_dim
        # 从 config 中获取 embed_dim 赋值给 self.embed_dim

    def forward(self, embeddings, x_size):
        # 前向传播函数，输入为 embeddings 和 x_size
        batch_size, height_width, num_channels = embeddings.shape
        # 获取 embeddings 的批量大小、高度宽度、通道数
        embeddings = embeddings.transpose(1, 2).view(batch_size, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        # 对 embeddings 进行转置和变形
        return embeddings
        # 返回处理后的 embeddings


# Copied from transformers.models.swinv2.modeling_swinv2.Swinv2PatchMerging with Swinv2->Swin2SR
class Swin2SRPatchMerging(nn.Module):
    """
    Patch Merging Layer.

    Args:
        input_resolution (`Tuple[int]`):
            Resolution of input feature.
        dim (`int`):
            Number of input channels.
        norm_layer (`nn.Module`, *optional*, defaults to `nn.LayerNorm`):
            Normalization layer class.
    """

    def __init__(self, input_resolution: Tuple[int], dim: int, norm_layer: nn.Module = nn.LayerNorm) -> None:
        # 初始化函数，接受 input_resolution、dim、norm_layer 参数
        super().__init__()
        # 调用父类的初始化函数
        self.input_resolution = input_resolution
        # 保存输入分辨率
        self.dim = dim
        # 保存输入通道数
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        # 定义线性层用于降维
        self.norm = norm_layer(2 * dim)
        # 根据 norm_layer 定义归一化层

    def maybe_pad(self, input_feature, height, width):
        # 定义用于辅助填充的函数
        should_pad = (height % 2 == 1) or (width % 2 == 1)
        # 判断是否需要填充
        if should_pad:
            pad_values = (0, 0, 0, width % 2, 0, height % 2)
            # 计算填充的值
            input_feature = nn.functional.pad(input_feature, pad_values)
            # 对输入特征进行填充

        return input_feature
        # 返回填充后的特征
    def forward(self, input_feature: torch.Tensor, input_dimensions: Tuple[int, int]) -> torch.Tensor:
        # 解包输入维度元组
        height, width = input_dimensions
        # 获取输入特征的形状信息
        batch_size, dim, num_channels = input_feature.shape

        # 改变输入特征的形状为[batch_size, height, width, num_channels]
        input_feature = input_feature.view(batch_size, height, width, num_channels)
        # 如果需要，对输入进行填充以使其能够被height和width整除
        input_feature = self.maybe_pad(input_feature, height, width)
        # 对输入进行下采样，得到input_feature_0，input_feature_1，input_feature_2，input_feature_3
        input_feature_0 = input_feature[:, 0::2, 0::2, :]
        input_feature_1 = input_feature[:, 1::2, 0::2, :]
        input_feature_2 = input_feature[:, 0::2, 1::2, :]
        input_feature_3 = input_feature[:, 1::2, 1::2, :]
        # 拼接四个下采样后的特征图，得到输入特征
        input_feature = torch.cat([input_feature_0, input_feature_1, input_feature_2, input_feature_3], -1)
        # 重新调整输入特征的形状为[batch_size, height/2 * width/2, 4*num_channels]
        input_feature = input_feature.view(batch_size, -1, 4 * num_channels)  # [batch_size, height/2 * width/2, 4*C]

        # 对输入特征进行降维操作
        input_feature = self.reduction(input_feature)
        # 对输入特征进行正则化操作
        input_feature = self.norm(input_feature)

        return input_feature
# 定义一个自注意力机制模块，源自于 transformers.models.swinv2.modeling_swinv2.Swinv2SelfAttention，将Swinv2替换为Swin2SR
class Swin2SRSelfAttention(nn.Module):
    # 定义一个函数，用于调整张量形状，以匹配注意力头的数量和每个头的大小
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # 调整张量形状
        x = x.view(new_x_shape)
        # 交换维度，以便将注意力头移动到合适的位置
        return x.permute(0, 2, 1, 3)

    # 自注意力机制的前向传播
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    # 这个函数实现了多头注意力机制
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
               head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False) -> Tuple[torch.Tensor]:
        # 获取输入张量的大小信息
        batch_size, dim, num_channels = hidden_states.shape
        # 对输入张量应用Query变换得到混合查询层
        mixed_query_layer = self.query(hidden_states)
    
        # 对输入张量应用Key变换并转置得到Key层
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        # 对输入张量应用Value变换并转置得到Value层
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        # 对混合查询层应用转置操作得到查询层
        query_layer = self.transpose_for_scores(mixed_query_layer)
    
        # 计算基于余弦相似度的注意力得分
        attention_scores = nn.functional.normalize(query_layer, dim=-1) @ nn.functional.normalize(
            key_layer, dim=-1
        ).transpose(-2, -1)
        # 应用可学习的温度因子放缩注意力得分
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()
        attention_scores = attention_scores * logit_scale
        # 加上基于相对位置的偏置
        relative_position_bias_table = self.continuous_position_bias_mlp(self.relative_coords_table).view(
            -1, self.num_attention_heads
        )
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attention_scores = attention_scores + relative_position_bias.unsqueeze(0)
    
        # 如果传入了注意力掩码，则应用掩码
        if attention_mask is not None:
            mask_shape = attention_mask.shape[0]
            attention_scores = attention_scores.view(
                batch_size // mask_shape, mask_shape, self.num_attention_heads, dim, dim
            ) + attention_mask.unsqueeze(1).unsqueeze(0)
            attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(0)
            attention_scores = attention_scores.view(-1, self.num_attention_heads, dim, dim)
    
        # 将注意力得分归一化为概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    
        # 对注意力概率应用dropout
        attention_probs = self.dropout(attention_probs)
    
        # 如果传入了头掩码，则应用头掩码
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
    
        # 将注意力权重应用于Value层得到上下文层
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
    
        # 返回上下文层和注意力概率（如果需要）
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs
# 从transformers.models.swin.modeling_swin.SwinSelfOutput中复制并修改为Swin2SRSelfOutput类
class Swin2SRSelfOutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        self.dense = nn.Linear(dim, dim)  # 创建一个线性层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)  # 创建一个丢弃层

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)  # 将隐藏状态通过线性层
        hidden_states = self.dropout(hidden_states)  # 使用丢弃层进行丢弃

        return hidden_states


# 从transformers.models.swinv2.modeling_swinv2.Swinv2Attention中复制并修改为Swin2SRAttention类
class Swin2SRAttention(nn.Module):
    def __init__(self, config, dim, num_heads, window_size, pretrained_window_size=0):
        super().__init__()
        self.self = Swin2SRSelfAttention(
            config=config,
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            pretrained_window_size=pretrained_window_size
            if isinstance(pretrained_window_size, collections.abc.Iterable)  # 如果pretrained_window_size是collections.abc.Iterable的实例
            else (pretrained_window_size, pretrained_window_size),  # 则使用pretrained_window_size，否则使用(pretrained_window_size, pretrained_window_size)
        )
        self.output = Swin2SRSelfOutput(config, dim)  # 创建Swin2SRSelfOutput对象
        self.pruned_heads = set()  # 创建一个空的集合对象

    def prune_heads(self, heads):
        if len(heads) == 0:  # 如果heads的长度为0
            return  # 直接返回
        heads, index = find_pruneable_heads_and_indices(  # 调用find_pruneable_heads_and_indices函数
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 修剪线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储被修剪的头部
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(hidden_states, attention_mask, head_mask, output_attentions)  # 将隐藏状态通过self层
        attention_output = self.output(self_outputs[0], hidden_states)  # 将self输出传递给output层，得到attention_output
        outputs = (attention_output,) + self_outputs[1:]  # 如果输出了注意力，则将注意力添加到outputs中
        return outputs  # 返回outputs


# 从transformers.models.swin.modeling_swin.SwinIntermediate中复制并修改为Swin2SRIntermediate类
class Swin2SRIntermediate(nn.Module):
    # 定义一个初始化函数，接受配置和维度参数
    def __init__(self, config, dim):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个全连接层，输入维度为dim，输出维度为config.mlp_ratio * dim
        self.dense = nn.Linear(dim, int(config.mlp_ratio * dim))
        # 判断config.hidden_act是否是字符串类型
        if isinstance(config.hidden_act, str):
            # 如果是字符串类型，则将config.hidden_act映射为对应的激活函数，并赋值给self.intermediate_act_fn
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            # 如果不是字符串类型，则直接将config.hidden_act赋值给self.intermediate_act_fn
            self.intermediate_act_fn = config.hidden_act

    # 定义一个前向传播函数，接受输入hidden_states，返回输出hidden_states
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入hidden_states传入全连接层，并得到输出
        hidden_states = self.dense(hidden_states)
        # 将输出hidden_states通过激活函数self.intermediate_act_fn进行激活
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回经过全连接层和激活函数处理后的hidden_states
        return hidden_states
# 从transformers.models.swin.modeling_swin.SwinOutput复制并改名为Swin2SROutput
class Swin2SROutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        # 使用线性变换层进行特征转换
        self.dense = nn.Linear(int(config.mlp_ratio * dim), dim)
        # 使用Dropout进行正则化
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 特征转换
        hidden_states = self.dense(hidden_states)
        # Dropout操作
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# 从transformers.models.swinv2.modeling_swinv2.Swinv2Layer复制并改名为Swin2SRLayer
class Swin2SRLayer(nn.Module):
    def __init__(self, config, dim, input_resolution, num_heads, shift_size=0, pretrained_window_size=0):
        super().__init__()
        # 计算窗口大小和移动步长
        self.input_resolution = input_resolution
        window_size, shift_size = self._compute_window_shift(
            (config.window_size, config.window_size), (shift_size, shift_size)
        )
        self.window_size = window_size[0]
        self.shift_size = shift_size[0]
        # 创建Swin2SRAttention层
        self.attention = Swin2SRAttention(
            config=config,
            dim=dim,
            num_heads=num_heads,
            window_size=self.window_size,
            pretrained_window_size=pretrained_window_size
            if isinstance(pretrained_window_size, collections.abc.Iterable)
            else (pretrained_window_size, pretrained_window_size),
        )
        # LayerNorm层
        self.layernorm_before = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        # DropPath层，根据配置选择是否执行
        self.drop_path = Swin2SRDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        # Swin2SRIntermediate层
        self.intermediate = Swin2SRIntermediate(config, dim)
        # Swin2SROutput层
        self.output = Swin2SROutput(config, dim)
        # LayerNorm层
        self.layernorm_after = nn.LayerNorm(dim, eps=config.layer_norm_eps)

    def _compute_window_shift(self, target_window_size, target_shift_size) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        # 计算窗口大小和移动步长
        window_size = [r if r <= w else w for r, w in zip(self.input_resolution, target_window_size)]
        shift_size = [0 if r <= w else s for r, w, s in zip(self.input_resolution, window_size, target_shift_size)]
        return window_size, shift_size
    def get_attn_mask(self, height, width, dtype): 
        # 创建注意力掩码，用于多头自注意力机制中的窗口平移
        if self.shift_size > 0:
            # 创建一个全零的注意力掩码，形状为 (1, height, width, 1)
            img_mask = torch.zeros((1, height, width, 1), dtype=dtype)
            # 切片使得 height 分成三个部分：0 到 -window_size, -window_size 到 -shift_size, -shift_size 到 None
            height_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            # 切片使得 width 分成三个部分：0 到 -window_size, -window_size 到 -shift_size, -shift_size 到 None
            width_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            # 用于给窗口中的每个像素点标识一个编号，编号从 0 开始递增
            count = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    # 将窗口内的像素点对应的位置标识为编号 count
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1

            # 将注意力掩码划分成窗口大小的小块
            mask_windows = window_partition(img_mask, self.window_size)
            # 将窗口块展平为二维数组并重新设置形状
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            # 创建注意力掩码矩阵
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            # 将注意力掩码矩阵中非零值填充为-100，将零值填充为0
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        # 返回注意力掩码
        return attn_mask

    def maybe_pad(self, hidden_states, height, width):
        # 计算需要填充的右边和底部的像素点数
        pad_right = (self.window_size - width % self.window_size) % self.window_size
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        # 定义填充的数值和填充模式（右边和底部填充）
        pad_values = (0, 0, 0, pad_right, 0, pad_bottom)
        # 使用填充的数值填充 hidden_states 张量
        hidden_states = nn.functional.pad(hidden_states, pad_values)
        # 返回填充后的 hidden_states 张量和填充模式
        return hidden_states, pad_values

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,


注释：
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 从输入维度中获取高度和宽度
        height, width = input_dimensions
        # 获取隐藏状态的批量大小、通道数
        batch_size, _, channels = hidden_states.size()
        # 将隐藏状态保存为shortcut
        shortcut = hidden_states

        # 将隐藏状态填充为窗口大小的倍数
        hidden_states = hidden_states.view(batch_size, height, width, channels)
        hidden_states, pad_values = self.maybe_pad(hidden_states, height, width)
        _, height_pad, width_pad, _ = hidden_states.shape
        # 循环移位
        if self.shift_size > 0:
            shifted_hidden_states = torch.roll(hidden_states, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_hidden_states = hidden_states

        # 划分窗口
        hidden_states_windows = window_partition(shifted_hidden_states, self.window_size)
        hidden_states_windows = hidden_states_windows.view(-1, self.window_size * self.window_size, channels)
        # 获取注意力遮罩
        attn_mask = self.get_attn_mask(height_pad, width_pad, dtype=hidden_states.dtype)
        if attn_mask is not None:
            attn_mask = attn_mask.to(hidden_states_windows.device)

        # 进行注意力计算
        attention_outputs = self.attention(
            hidden_states_windows, attn_mask, head_mask, output_attentions=output_attentions
        )

        attention_output = attention_outputs[0]

        attention_windows = attention_output.view(-1, self.window_size, self.window_size, channels)
        shifted_windows = window_reverse(attention_windows, self.window_size, height_pad, width_pad)

        # 反向循环移位
        if self.shift_size > 0:
            attention_windows = torch.roll(shifted_windows, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attention_windows = shifted_windows

        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
            attention_windows = attention_windows[:, :height, :width, :].contiguous()

        attention_windows = attention_windows.view(batch_size, height * width, channels)
        hidden_states = self.layernorm_before(attention_windows)
        hidden_states = shortcut + self.drop_path(hidden_states)

        layer_output = self.intermediate(hidden_states)
        layer_output = self.output(layer_output)
        layer_output = hidden_states + self.drop_path(self.layernorm_after(layer_output))

        layer_outputs = (layer_output, attention_outputs[1]) if output_attentions else (layer_output,)
        return layer_outputs
class Swin2SRStage(nn.Module):
    """
    This corresponds to the Residual Swin Transformer Block (RSTB) in the original implementation.
    """

    def __init__(self, config, dim, input_resolution, depth, num_heads, drop_path, pretrained_window_size=0):
        super().__init__()
        # 存储配置信息和维度
        self.config = config
        self.dim = dim
        # 创建 Transformer 层列表
        self.layers = nn.ModuleList(
            [
                Swin2SRLayer(
                    config=config,
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    shift_size=0 if (i % 2 == 0) else config.window_size // 2,
                    pretrained_window_size=pretrained_window_size,
                )
                for i in range(depth)
            ]
        )

        # 根据连接方式不同创建不同的卷积层
        if config.resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif config.resi_connection == "3conv":
            # 为了节省参数和内存，创建了一个序列化的卷积层
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )

        # 创建 Patch Embedding 实例
        self.patch_embed = Swin2SRPatchEmbeddings(config, normalize_patches=False)

        # 创建 Patch Un-Embedding 实例
        self.patch_unembed = Swin2SRPatchUnEmbeddings(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 保存初始隐藏状态，用于残差连接
        residual = hidden_states

        height, width = input_dimensions
        # 遍历 Transformer 层并进行前向传播
        for i, layer_module in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(hidden_states, input_dimensions, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

        output_dimensions = (height, width, height, width)

        # 反向 Patch Un-Embedding 操作
        hidden_states = self.patch_unembed(hidden_states, input_dimensions)
        # 进行卷积操作
        hidden_states = self.conv(hidden_states)
        # 前向 Patch Embedding 操作
        hidden_states, _ = self.patch_embed(hidden_states)

        # 将隐藏状态与初始残差相加
        hidden_states = hidden_states + residual

        stage_outputs = (hidden_states, output_dimensions)

        # 如果需要输出注意力矩阵，则添加到输出中
        if output_attentions:
            stage_outputs += layer_outputs[1:]
        return stage_outputs


class Swin2SREncoder(nn.Module):
    # 初始化 Swin2SREncoder 类
    def __init__(self, config, grid_size):
        # 调用父类的初始化方法
        super().__init__()
        # 获取深度配置的长度，作为当前模型的阶段数
        self.num_stages = len(config.depths)
        # 保存配置信息
        self.config = config
        # 根据配置的 drop_path_rate 计算每个阶段的 drop path 概率
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]
        # 使用 nn.ModuleList 创建各个阶段的模块
        self.stages = nn.ModuleList(
            [
                # 对于每个阶段
                Swin2SRStage(
                    # 传入配置信息
                    config=config,
                    # 设置隐藏层维度
                    dim=config.embed_dim,
                    # 设置输入分辨率
                    input_resolution=(grid_size[0], grid_size[1]),
                    # 设置当前阶段的深度
                    depth=config.depths[stage_idx],
                    # 设置当前阶段的注意力头数
                    num_heads=config.num_heads[stage_idx],
                    # 设置当前阶段的 drop path 概率
                    drop_path=dpr[sum(config.depths[:stage_idx]) : sum(config.depths[: stage_idx + 1])],
                    # 设置预训练的窗口大小为 0
                    pretrained_window_size=0,
                )
                # 对于每个阶段进行迭代
                for stage_idx in range(self.num_stages)
            ]
        )
        # 关闭梯度检查点
        self.gradient_checkpointing = False
    
    # 前向传播
    def forward(
        self,
        # 输入的隐藏状态
        hidden_states: torch.Tensor,
        # 输入的尺寸
        input_dimensions: Tuple[int, int],
        # 头部掩码
        head_mask: Optional[torch.FloatTensor] = None,
        # 是否输出注意力
        output_attentions: Optional[bool] = False,
        # 是否输出隐藏状态
        output_hidden_states: Optional[bool] = False,
        # 是否返回字典格式输出
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, Swin2SREncoderOutput]:
        # 初始化空元组，用于存储中间结果
        all_input_dimensions = ()
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
    
        # 如果需要输出隐藏状态，则先添加初始的隐藏状态
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
    
        # 对于每个阶段模块进行迭代
        for i, stage_module in enumerate(self.stages):
            # 获取当前阶段的头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None
    
            # 如果开启了梯度检查点，且处于训练模式
            if self.gradient_checkpointing and self.training:
                # 使用 _gradient_checkpointing_func 进行前向传播
                layer_outputs = self._gradient_checkpointing_func(
                    stage_module.__call__, hidden_states, input_dimensions, layer_head_mask, output_attentions
                )
            else:
                # 否则直接调用stage_module的前向传播方法
                layer_outputs = stage_module(hidden_states, input_dimensions, layer_head_mask, output_attentions)
    
            # 获取当前阶段的输出隐藏状态和输出尺寸
            hidden_states = layer_outputs[0]
            output_dimensions = layer_outputs[1]
    
            # 更新输入尺寸
            input_dimensions = (output_dimensions[-2], output_dimensions[-1])
            all_input_dimensions += (input_dimensions,)
    
            # 如果需要输出隐藏状态，则添加当前阶段的隐藏状态
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
    
            # 如果需要输出注意力，则添加当前阶段的注意力
            if output_attentions:
                all_self_attentions += layer_outputs[2:]
    
        # 根据返回设置构建输出
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
    
        return Swin2SREncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
class Swin2SRPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # Swin2SRPreTrainedModel 类，用于处理权重初始化和预训练模型的下载和加载

    config_class = Swin2SRConfig
    base_model_prefix = "swin2sr"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    # 设置类变量：config_class, base_model_prefix, main_input_name, supports_gradient_checkpointing

    def _init_weights(self, module):
        """Initialize the weights"""
        # 初始化模型权重
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.trunc_normal_(module.weight.data, std=self.config.initializer_range)
            # 使用截尾正态分布初始化线性层和卷积层的权重
            if module.bias is not None:
                module.bias.data.zero_()
                # 如果有偏置项，初始化为零
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            # 如果是 LayerNorm 层，初始化偏置为零，权重为1.0


SWIN2SR_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`Swin2SRConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# Swin2SR 模型的文档字符串，介绍模型类的基本信息和使用说明

SWIN2SR_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`Swin2SRImageProcessor.__call__`] for details.
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# Swin2SR 模型的输入文档字符串，描述模型接受的输入参数及其含义

@add_start_docstrings(
    "The bare Swin2SR Model transformer outputting raw hidden-states without any specific head on top.",
    SWIN2SR_START_DOCSTRING,
)
class Swin2SRModel(Swin2SRPreTrainedModel):

# Swin2SRModel 类，继承自 Swin2SRPreTrainedModel 类，添加了一些文档字符串信息
    # 初始化成员变量
    def __init__(self, config):
        # 调用父类的构造方法
        super().__init__(config)
        # 保存配置参数
        self.config = config

        # 如果输入和输出的通道数都为3，则设置RGB均值
        if config.num_channels == 3 and config.num_channels_out == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            # 将RGB均值转换为张量，并且调整形状
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            # 设置均值为0
            self.mean = torch.zeros(1, 1, 1, 1)
        # 设置图像范围
        self.img_range = config.img_range

        # 创建第一个卷积层
        self.first_convolution = nn.Conv2d(config.num_channels, config.embed_dim, 3, 1, 1)
        # 创建嵌入层
        self.embeddings = Swin2SREmbeddings(config)
        # 创建编码器
        self.encoder = Swin2SREncoder(config, grid_size=self.embeddings.patch_embeddings.patches_resolution)

        # 归一化层
        self.layernorm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        # 解嵌层
        self.patch_unembed = Swin2SRPatchUnEmbeddings(config)
        # 后处理卷积层
        self.conv_after_body = nn.Conv2d(config.embed_dim, config.embed_dim, 3, 1, 1)

        # 初始化权重并进行最终处理
        self.post_init()

    # 获取输入嵌入
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

    # 对像素值进行填充和归一化处理
    def pad_and_normalize(self, pixel_values):
        _, _, height, width = pixel_values.size()

        # 1. 填充
        window_size = self.config.window_size
        modulo_pad_height = (window_size - height % window_size) % window_size
        modulo_pad_width = (window_size - width % window_size) % window_size
        # 通过反射填充像素值
        pixel_values = nn.functional.pad(pixel_values, (0, modulo_pad_width, 0, modulo_pad_height), "reflect")

        # 2. 归一化
        self.mean = self.mean.type_as(pixel_values)
        # 对像素值进行归一化处理
        pixel_values = (pixel_values - self.mean) * self.img_range

        return pixel_values

    # 前向传播
    @add_start_docstrings_to_model_forward(SWIN2SR_INPUTS_DOCSTRING)
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
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 定义函数read_zip的输入参数和返回类型
        ) -> Union[Tuple, BaseModelOutput]:
        # 如果output_attentions为None，则使用默认值self.config.output_attentions
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果output_hidden_states为None，则使用默认值self.config.output_hidden_states
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果return_dict为None，则使用默认值self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果需要，准备头部掩码
        # 在head_mask中，1.0表示我们保留该头部
        # attention_probs的形状为bsz x n_heads x N x N
        # 输入的head_mask的形状为[num_heads]或[num_hidden_layers x num_heads]
        # head_mask被转换为形状为[num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, len(self.config.depths))

        _, _, height, width = pixel_values.shape

        # 一些预处理：padding + normalization
        pixel_values = self.pad_and_normalize(pixel_values)

        # 第一次卷积
        embeddings = self.first_convolution(pixel_values)
        # 获得嵌入输出和输入的维度
        embedding_output, input_dimensions = self.embeddings(embeddings)

        # 编码器输出
        encoder_outputs = self.encoder(
            embedding_output,
            input_dimensions,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 序列输出
        sequence_output = encoder_outputs[0]
        # 对序列输出进行layernorm
        sequence_output = self.layernorm(sequence_output)

        # 对序列输出进行patch_unembed
        sequence_output = self.patch_unembed(sequence_output, (height, width))
        # 经过主体后的卷积输出加上嵌入
        sequence_output = self.conv_after_body(sequence_output) + embeddings

        # 如果return_dict为False，则返回输出元组
        if not return_dict:
            output = (sequence_output,) + encoder_outputs[1:]

            return output

        # 返回BaseModelOutput对象
        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
class Upsample(nn.Module):
    """上采样模块。

    Args:
        scale (`int`):
            缩放因子。支持的缩放因子：2^n 和 3。
        num_features (`int`):
            中间特征的通道数。
    """

    def __init__(self, scale, num_features):
        super().__init__()

        self.scale = scale
        if (scale & (scale - 1)) == 0:
            # 如果 scale = 2^n
            for i in range(int(math.log(scale, 2))):
                self.add_module(f"convolution_{i}", nn.Conv2d(num_features, 4 * num_features, 3, 1, 1))
                self.add_module(f"pixelshuffle_{i}", nn.PixelShuffle(2))
        elif scale == 3:
            # 如果 scale = 3
            self.convolution = nn.Conv2d(num_features, 9 * num_features, 3, 1, 1)
            self.pixelshuffle = nn.PixelShuffle(3)
        else:
            raise ValueError(f"Scale {scale} is not supported. Supported scales: 2^n and 3.")

    def forward(self, hidden_state):
        if (self.scale & (self.scale - 1)) == 0:
            for i in range(int(math.log(self.scale, 2))):
                hidden_state = self.__getattr__(f"convolution_{i}")(hidden_state)
                hidden_state = self.__getattr__(f"pixelshuffle_{i}")(hidden_state)

        elif self.scale == 3:
            hidden_state = self.convolution(hidden_state)
            hidden_state = self.pixelshuffle(hidden_state)

        return hidden_state


class UpsampleOneStep(nn.Module):
    """上采样一步模块（与 Upsample 的区别在于它始终只有 1 个卷积层 + 1 个像素混洗层）。

    用于轻量级超分辨率以节省参数。

    Args:
        scale (int):
            缩放因子。支持的缩放因子：2^n 和 3。
        in_channels (int):
            输入特征的通道数。
        out_channels (int):
            输出特征的通道数。
    """

    def __init__(self, scale, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, (scale**2) * out_channels, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)

        return x


class PixelShuffleUpsampler(nn.Module):
    def __init__(self, config, num_features):
        super().__init__()
        self.conv_before_upsample = nn.Conv2d(config.embed_dim, num_features, 3, 1, 1)
        self.activation = nn.LeakyReLU(inplace=True)
        self.upsample = Upsample(config.upscale, num_features)
        self.final_convolution = nn.Conv2d(num_features, config.num_channels_out, 3, 1, 1)

    def forward(self, sequence_output):
        x = self.conv_before_upsample(sequence_output)
        x = self.activation(x)
        x = self.upsample(x)
        x = self.final_convolution(x)

        return x


class NearestConvUpsampler(nn.Module):
    # 初始化函数，包含配置和特征数量作为参数
    def __init__(self, config, num_features):
        # 调用父类的初始化方法
        super().__init__()
        # 如果配置中的放大倍数不是4，抛出数值错误异常
        if config.upscale != 4:
            raise ValueError("The nearest+conv upsampler only supports an upscale factor of 4 at the moment.")
    
        # 创建卷积层，输入维度为config.embed_dim，输出维度为num_features，卷积核大小为3，步长为1，填充为1
        self.conv_before_upsample = nn.Conv2d(config.embed_dim, num_features, 3, 1, 1)
        # 创建激活函数层，使用LeakyReLU，并设置inplace参数为True
        self.activation = nn.LeakyReLU(inplace=True)
        # 创建卷积层，输入和输出维度都为num_features，卷积核大小为3，步长为1，填充为1
        self.conv_up1 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        # 创建卷积层，输入和输出维度都为num_features，卷积核大小为3，步长为1，填充为1
        self.conv_up2 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        # 创建卷积层，输入和输出维度都为num_features，卷积核大小为3，步长为1，填充为1
        self.conv_hr = nn.Conv2d(num_features, num_features, 3, 1, 1)
        # 创建卷积层，输入维度为num_features，输出维度为config.num_channels_out，卷积核大小为3，步长为1，填充为1
        self.final_convolution = nn.Conv2d(num_features, config.num_channels_out, 3, 1, 1)
        # 创建LeakyReLU激活函数层，设置negative_slope为0.2，并将inplace参数设置为True
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    # 前向传播函数，输入为sequence_output
    def forward(self, sequence_output):
        # 将输入的sequence_output进行卷积操作
        sequence_output = self.conv_before_upsample(sequence_output)
        # 对卷积后的结果进行激活函数操作
        sequence_output = self.activation(sequence_output)
        # 使用双线性插值对sequence_output进行上采样，并对上采样后的结果进行卷积和LeakyReLU激活函数操作
        sequence_output = self.lrelu(
            self.conv_up1(torch.nn.functional.interpolate(sequence_output, scale_factor=2, mode="nearest"))
        )
        # 再次使用双线性插值对sequence_output进行上采样，并对上采样后的结果进行卷积和LeakyReLU激活函数操作
        sequence_output = self.lrelu(
            self.conv_up2(torch.nn.functional.interpolate(sequence_output, scale_factor=2, mode="nearest"))
        )
        # 对上一步操作后的结果进行卷积和LeakyReLU激活函数操作，得到重建的图像
        reconstruction = self.final_convolution(self.lrelu(self.conv_hr(sequence_output)))
        # 返回重建的图像
        return reconstruction
class PixelShuffleAuxUpsampler(nn.Module):
    # 定义一个名为PixelShuffleAuxUpsampler的类，继承自nn.Module
    def __init__(self, config, num_features):
        # 初始化方法，接收config和num_features两个参数
        super().__init__()
        # 调用父类的初始化方法

        self.upscale = config.upscale
        # 设置self.upscale为config中的upscale属性值
        self.conv_bicubic = nn.Conv2d(config.num_channels, num_features, 3, 1, 1)
        # 创建一个2维卷积层，输入通道数为config.num_channels，输出通道数为num_features，卷积核大小为3x3，步长为1，填充为1
        self.conv_before_upsample = nn.Conv2d(config.embed_dim, num_features, 3, 1, 1)
        # 创建一个2维卷积层，输入通道数为config.embed_dim，输出通道数为num_features，卷积核大小为3x3，步长为1，填充为1
        self.activation = nn.LeakyReLU(inplace=True)
        # 创建一个LeakyReLU激活函数实例，inplace参数为True表示inplace执行
        self.conv_aux = nn.Conv2d(num_features, config.num_channels, 3, 1, 1)
        # 创建一个2维卷积层，输入通道数为num_features，输出通道数为config.num_channels，卷积核大小为3x3，步长为1，填充为1
        self.conv_after_aux = nn.Sequential(nn.Conv2d(3, num_features, 3, 1, 1), nn.LeakyReLU(inplace=True))
        # 创建一个包含两个2维卷积层和一个LeakyReLU激活函数的序列
        self.upsample = Upsample(config.upscale, num_features)
        # 创建一个Upsample实例，传入config.upscale和num_features作为参数
        self.final_convolution = nn.Conv2d(num_features, config.num_channels_out, 3, 1, 1)
        # 创建一个2维卷积层，输入通道数为num_features，输出通道数为config.num_channels_out，卷积核大小为3x3，步长为1，填充为1

    def forward(self, sequence_output, bicubic, height, width):
        # 定义前向传播方法，接收sequence_output, bicubic, height, width四个参数
        bicubic = self.conv_bicubic(bicubic)
        # 对bicubic进行卷积操作
        sequence_output = self.conv_before_upsample(sequence_output)
        # 对sequence_output进行卷积操作
        sequence_output = self.activation(sequence_output)
        # 对sequence_output进行激活函数操作
        aux = self.conv_aux(sequence_output)
        # 对sequence_output进行卷积操作，赋值给aux
        sequence_output = self.conv_after_aux(aux)
        # 对aux进行卷积操作，赋值给sequence_output
        sequence_output = (
            self.upsample(sequence_output)[:, :, : height * self.upscale, : width * self.upscale]
            + bicubic[:, :, : height * self.upscale, : width * self.upscale]
        )
        # 对sequence_output和bicubic进行像素级相加操作
        reconstruction = self.final_convolution(sequence_output)
        # 对sequence_output进行卷积操作，得到reconstruction

        return reconstruction, aux
        # 返回reconstruction和aux


@add_start_docstrings(
    """
    Swin2SR Model transformer with an upsampler head on top for image super resolution and restoration.
    """,
    SWIN2SR_START_DOCSTRING,
)
class Swin2SRForImageSuperResolution(Swin2SRPreTrainedModel):
    # 定义一个名为Swin2SRForImageSuperResolution的类，继承自Swin2SRPreTrainedModel类
    def __init__(self, config):
        # 初始化方法，接收config参数
        super().__init__(config)
        # 调用父类的初始化方法

        self.swin2sr = Swin2SRModel(config)
        # 创建一个Swin2SRModel实例，传入config作为参数
        self.upsampler = config.upsampler
        # 设置self.upsampler为config的upsampler属性值
        self.upscale = config.upscale
        # 设置self.upscale为config的upscale属性值
    # 定义一个名为 forward 的方法，用于前向传播
    def forward(
        # 输入像素值，默认为 None，数据类型为 torch.FloatTensor
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        # 头部掩码，默认为 None，数据类型为 torch.FloatTensor
        head_mask: Optional[torch.FloatTensor] = None,
        # 标签，默认为 None，数据类型为 torch.LongTensor
        labels: Optional[torch.LongTensor] = None,
        # 是否输出注意力权重，默认为 None，数据类型为 bool
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，默认为 None，数据类型为 bool
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典，默认为 None，数据类型为 bool
        return_dict: Optional[bool] = None,
```py  
```