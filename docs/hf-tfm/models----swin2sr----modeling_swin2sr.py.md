# `.\models\swin2sr\modeling_swin2sr.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 Microsoft Research 和 HuggingFace Inc. 团队所有
#
# 根据 Apache 许可证 2.0 版本使用本文件
# 除非符合许可证规定，否则不得使用本文件
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，本软件是基于“原样”提供的，不提供任何明示或暗示的保证或条件
# 请参阅许可证获取更多信息
""" PyTorch Swin2SR Transformer model."""

# 导入必要的库和模块
import collections.abc  # 导入 collections.abc 模块
import math  # 导入 math 模块
from dataclasses import dataclass  # 从 dataclasses 模块导入 dataclass 装饰器
from typing import Optional, Tuple, Union  # 导入类型提示的相关类和类型

import torch  # 导入 PyTorch 库
import torch.utils.checkpoint  # 导入 PyTorch 的 checkpoint 工具
from torch import nn  # 从 PyTorch 导入 nn 模块

# 导入模型相关的子模块和函数
from ...activations import ACT2FN  # 从 ...activations 模块导入 ACT2FN 函数
from ...modeling_outputs import BaseModelOutput, ImageSuperResolutionOutput  # 从 ...modeling_outputs 模块导入输出类
from ...modeling_utils import PreTrainedModel  # 从 ...modeling_utils 模块导入预训练模型相关类
from ...pytorch_utils import find_pruneable_heads_and_indices, meshgrid, prune_linear_layer  # 从 ...pytorch_utils 导入相关工具函数
from ...utils import (
    ModelOutput,  # 从 ...utils 模块导入 ModelOutput 类
    add_code_sample_docstrings,  # 从 ...utils 模块导入相关函数和类
    add_start_docstrings,  # 从 ...utils 模块导入相关函数和类
    add_start_docstrings_to_model_forward,  # 从 ...utils 模块导入相关函数和类
    logging,  # 从 ...utils 模块导入 logging 模块
    replace_return_docstrings,  # 从 ...utils 模块导入相关函数
)

# 导入 Swin2SR 的配置类
from .configuration_swin2sr import Swin2SRConfig  # 从当前目录下的 configuration_swin2sr 模块导入 Swin2SRConfig 类

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的一般信息
_CONFIG_FOR_DOC = "Swin2SRConfig"

# 用于文档的基本检查点信息
_CHECKPOINT_FOR_DOC = "caidas/swin2SR-classical-sr-x2-64"

# 预期的输出形状
_EXPECTED_OUTPUT_SHAPE = [1, 180, 488, 648]

# Swin2SR 预训练模型存档列表
SWIN2SR_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "caidas/swin2SR-classical-sr-x2-64",
    # 查看所有 Swin2SR 模型，请访问 https://huggingface.co/models?filter=swin2sr
]

@dataclass
class Swin2SREncoderOutput(ModelOutput):
    """
    Swin2SR 编码器的输出，可能包含隐藏状态和注意力权重。

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的隐藏状态序列输出。
        hidden_states (`tuple(torch.FloatTensor)`, *可选*, 当 `output_hidden_states=True` 传递或当 `config.output_hidden_states=True` 时返回):
            模型每层的隐藏状态的元组，包括初始嵌入的输出。

            模型每层的隐藏状态以及初始嵌入的输出。
        attentions (`tuple(torch.FloatTensor)`, *可选*, 当 `output_attentions=True` 传递或当 `config.output_attentions=True` 时返回):
            模型每阶段的注意力权重的元组。

            注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    """
    # 声明一个变量 last_hidden_state，类型为 torch.FloatTensor，初始值为 None
    last_hidden_state: torch.FloatTensor = None
    # 声明一个变量 hidden_states，类型为可选的元组，元素类型为 torch.FloatTensor
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 声明一个变量 attentions，类型为可选的元组，元素类型为 torch.FloatTensor
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# Copied from transformers.models.swin.modeling_swin.window_partition
def window_partition(input_feature, window_size):
    """
    Partitions the given input into windows.
    """
    # 获取输入特征的尺寸信息：批量大小、高度、宽度、通道数
    batch_size, height, width, num_channels = input_feature.shape
    # 将输入特征按窗口大小进行划分，重塑为新的形状
    input_feature = input_feature.view(
        batch_size, height // window_size, window_size, width // window_size, window_size, num_channels
    )
    # 对划分后的窗口进行重新排序，以便后续处理
    windows = input_feature.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, num_channels)
    return windows


# Copied from transformers.models.swin.modeling_swin.window_reverse
def window_reverse(windows, window_size, height, width):
    """
    Merges windows to produce higher resolution features.
    """
    # 确定窗口的通道数
    num_channels = windows.shape[-1]
    # 将窗口合并为更高分辨率的特征
    windows = windows.view(-1, height // window_size, width // window_size, window_size, window_size, num_channels)
    # 对合并后的特征进行重新排序，以符合原始输入的形状
    windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, height, width, num_channels)
    return windows


# Copied from transformers.models.beit.modeling_beit.drop_path
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    # 如果 drop_prob 为 0 或不处于训练模式，则直接返回输入
    if drop_prob == 0.0 or not training:
        return input
    # 计算保留的概率
    keep_prob = 1 - drop_prob
    # 创建一个与输入形状相同的随机张量
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # 将随机张量二值化
    # 应用 drop path 操作，并返回处理后的输出
    output = input.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.swin.modeling_swin.SwinDropPath with Swin->Swin2SR
class Swin2SRDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 调用 drop_path 函数来执行 drop path 操作
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class Swin2SREmbeddings(nn.Module):
    """
    Construct the patch and optional position embeddings.
    """
    # 初始化函数，接受一个配置参数config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()

        # 使用配置参数初始化Swin2SRPatchEmbeddings对象，赋值给self.patch_embeddings
        self.patch_embeddings = Swin2SRPatchEmbeddings(config)
        
        # 获取patch数目，用于后续位置编码的初始化
        num_patches = self.patch_embeddings.num_patches

        # 根据配置决定是否创建位置编码的参数
        if config.use_absolute_embeddings:
            # 创建一个形状为(1, num_patches + 1, config.embed_dim)的可学习参数，初始值为全零
            self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))
        else:
            # 如果不使用绝对位置编码，则置为None
            self.position_embeddings = None

        # 初始化一个dropout层，使用给定的dropout概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # 保存配置中的窗口大小参数
        self.window_size = config.window_size

    # 前向传播函数，接受一个可选的torch.FloatTensor类型的像素值作为输入，返回一个torch.Tensor类型的元组
    def forward(self, pixel_values: Optional[torch.FloatTensor]) -> Tuple[torch.Tensor]:
        # 调用patch_embeddings对象处理输入像素值，返回嵌入张量和输出维度信息
        embeddings, output_dimensions = self.patch_embeddings(pixel_values)

        # 如果位置编码参数不为None，则将嵌入张量和位置编码相加
        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings

        # 对嵌入张量应用dropout操作
        embeddings = self.dropout(embeddings)

        # 返回处理后的嵌入张量和输出维度信息的元组
        return embeddings, output_dimensions
class Swin2SRPatchEmbeddings(nn.Module):
    # Swin2SRPatchEmbeddings 类的定义，继承自 nn.Module
    def __init__(self, config, normalize_patches=True):
        super().__init__()
        # 初始化函数，接收配置参数和是否标准化补丁的标志

        num_channels = config.embed_dim
        # 从配置中获取嵌入维度

        image_size, patch_size = config.image_size, config.patch_size
        # 从配置中获取图像尺寸和补丁尺寸

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        # 确保图像尺寸和补丁尺寸是可迭代对象，如果不是，则转换为元组形式

        patches_resolution = [image_size[0] // patch_size[0], image_size[1] // patch_size[1]]
        # 计算补丁的分辨率，即图像被划分成的补丁数目

        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        # 设置补丁的分辨率和补丁的总数

        self.projection = nn.Conv2d(num_channels, config.embed_dim, kernel_size=patch_size, stride=patch_size)
        # 使用卷积层进行投影，将输入的通道数转换为嵌入维度，卷积核大小为补丁大小，步长为补丁大小

        self.layernorm = nn.LayerNorm(config.embed_dim) if normalize_patches else None
        # 如果需要对补丁进行标准化，则使用 LayerNorm 进行处理，否则设为 None

    def forward(self, embeddings: Optional[torch.FloatTensor]) -> Tuple[torch.Tensor, Tuple[int]]:
        # 前向传播函数，接收嵌入向量作为输入，返回嵌入后的张量和输出维度的元组

        embeddings = self.projection(embeddings)
        # 使用定义的投影层对输入的嵌入向量进行投影变换

        _, _, height, width = embeddings.shape
        # 获取投影后张量的高度和宽度信息

        output_dimensions = (height, width)
        # 记录输出的高度和宽度信息

        embeddings = embeddings.flatten(2).transpose(1, 2)
        # 将投影后的张量按照第三维度展平，然后进行转置操作

        if self.layernorm is not None:
            embeddings = self.layernorm(embeddings)
        # 如果定义了 LayerNorm 层，则对嵌入向量进行标准化处理

        return embeddings, output_dimensions
        # 返回处理后的嵌入向量和输出的尺寸信息


class Swin2SRPatchUnEmbeddings(nn.Module):
    # Swin2SRPatchUnEmbeddings 类的定义，继承自 nn.Module
    r"""Image to Patch Unembedding"""

    def __init__(self, config):
        super().__init__()
        # 初始化函数，接收配置参数

        self.embed_dim = config.embed_dim
        # 设置嵌入维度为配置中指定的值

    def forward(self, embeddings, x_size):
        # 前向传播函数，接收嵌入向量和图像尺寸作为输入

        batch_size, height_width, num_channels = embeddings.shape
        # 获取输入嵌入向量的批量大小、高度宽度乘积以及通道数

        embeddings = embeddings.transpose(1, 2).view(batch_size, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        # 将嵌入向量进行转置和视图变换，以重构原始图像尺寸

        return embeddings
        # 返回重构后的嵌入向量


# Copied from transformers.models.swinv2.modeling_swinv2.Swinv2PatchMerging with Swinv2->Swin2SR
class Swin2SRPatchMerging(nn.Module):
    # Swin2SRPatchMerging 类的定义，继承自 nn.Module
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
        super().__init__()
        # 初始化函数，接收输入特征的分辨率、输入通道数和可选的标准化层

        self.input_resolution = input_resolution
        # 设置输入特征的分辨率属性

        self.dim = dim
        # 设置输入通道数属性

        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        # 使用线性层进行维度减少，从 4*dim 到 2*dim，无偏置项

        self.norm = norm_layer(2 * dim)
        # 使用指定的标准化层对输出进行标准化处理

    def maybe_pad(self, input_feature, height, width):
        # 辅助函数，可能对输入特征进行填充，使得其高度和宽度为偶数

        should_pad = (height % 2 == 1) or (width % 2 == 1)
        # 检查输入特征的高度或宽度是否为奇数

        if should_pad:
            pad_values = (0, 0, 0, width % 2, 0, height % 2)
            # 计算需要填充的值

            input_feature = nn.functional.pad(input_feature, pad_values)
            # 使用 PyTorch 的函数进行填充操作

        return input_feature
        # 返回填充后的输入特征
    def forward(self, input_feature: torch.Tensor, input_dimensions: Tuple[int, int]) -> torch.Tensor:
        # 解包输入维度元组
        height, width = input_dimensions
        # `dim` 是输入特征的维度，即 height * width
        batch_size, dim, num_channels = input_feature.shape

        # 将输入特征重新视图化为四维张量 [batch_size, height, width, num_channels]
        input_feature = input_feature.view(batch_size, height, width, num_channels)
        
        # 如果需要，对输入进行填充使其可以被 width 和 height 整除
        input_feature = self.maybe_pad(input_feature, height, width)
        
        # 提取四个子块，每个块大小为 [batch_size, height/2, width/2, num_channels]
        input_feature_0 = input_feature[:, 0::2, 0::2, :]
        input_feature_1 = input_feature[:, 1::2, 0::2, :]
        input_feature_2 = input_feature[:, 0::2, 1::2, :]
        input_feature_3 = input_feature[:, 1::2, 1::2, :]
        
        # 将四个子块沿最后一个维度拼接，形成新的特征张量 [batch_size, height/2, width/2, 4*num_channels]
        input_feature = torch.cat([input_feature_0, input_feature_1, input_feature_2, input_feature_3], -1)
        
        # 将特征张量重新视图化为 [batch_size, height/2 * width/2, 4*num_channels]
        input_feature = input_feature.view(batch_size, -1, 4 * num_channels)

        # 使用 reduction 方法对特征张量进行降维处理
        input_feature = self.reduction(input_feature)
        
        # 使用 norm 方法对降维后的特征张量进行归一化处理
        input_feature = self.norm(input_feature)

        # 返回处理后的特征张量作为输出
        return input_feature
# 从transformers.models.swinv2.modeling_swinv2.Swinv2SelfAttention复制而来，将Swinv2改为Swin2SR
class Swin2SRSelfAttention(nn.Module):
    # 将输入张量x重新形状以用于注意力分数计算
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 自注意力机制的前向传播
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:  # 函数声明，返回类型为包含单个张量的元组
        batch_size, dim, num_channels = hidden_states.shape  # 获取隐藏状态张量的形状信息
        mixed_query_layer = self.query(hidden_states)  # 使用 query 网络对隐藏状态进行处理得到混合查询层

        key_layer = self.transpose_for_scores(self.key(hidden_states))  # 使用 key 网络对隐藏状态进行处理，然后转置以用于注意力计算
        value_layer = self.transpose_for_scores(self.value(hidden_states))  # 使用 value 网络对隐藏状态进行处理，然后转置以用于注意力计算
        query_layer = self.transpose_for_scores(mixed_query_layer)  # 对混合查询层进行转置以用于注意力计算

        # cosine attention
        attention_scores = nn.functional.normalize(query_layer, dim=-1) @ nn.functional.normalize(
            key_layer, dim=-1
        ).transpose(-2, -1)  # 计算注意力分数，使用余弦相似度进行归一化，然后进行乘积计算

        logit_scale = torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()  # 限制并指数化对数缩放参数
        attention_scores = attention_scores * logit_scale  # 缩放注意力分数

        relative_position_bias_table = self.continuous_position_bias_mlp(self.relative_coords_table).view(
            -1, self.num_attention_heads
        )  # 使用位置偏置 MLP 计算连续位置偏置表，并进行形状重塑

        # [window_height*window_width,window_height*window_width,num_attention_heads]
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )  # 根据相对位置索引选择相对位置偏置表中的偏置，并进行形状调整

        # [num_attention_heads,window_height*window_width,window_height*window_width]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # 调整相对位置偏置的维度顺序

        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)  # 对相对位置偏置进行 sigmoid 处理并乘以常数 16
        attention_scores = attention_scores + relative_position_bias.unsqueeze(0)  # 添加相对位置偏置到注意力分数中

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in Swin2SRModel forward() function)
            mask_shape = attention_mask.shape[0]  # 获取注意力掩码的形状信息
            attention_scores = attention_scores.view(
                batch_size // mask_shape, mask_shape, self.num_attention_heads, dim, dim
            ) + attention_mask.unsqueeze(1).unsqueeze(0)  # 将注意力分数调整为与掩码相匹配的形状，并应用掩码

            attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(0)  # 再次应用掩码

            attention_scores = attention_scores.view(-1, self.num_attention_heads, dim, dim)  # 调整注意力分数的形状

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)  # 对注意力分数进行 softmax 归一化

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)  # 使用 dropout 对注意力概率进行处理

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask  # 如果有头部掩码，则将其应用到注意力概率上

        context_layer = torch.matmul(attention_probs, value_layer)  # 使用注意力概率与值层进行加权求和得到上下文层
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # 调整上下文层的维度顺序

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # 计算新的上下文层形状
        context_layer = context_layer.view(new_context_layer_shape)  # 根据计算的形状调整上下文层的形状

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)  # 根据是否需要输出注意力分数来选择输出内容

        return outputs  # 返回上下文层和（如果需要）注意力分数
# 从 transformers.models.swin.modeling_swin.SwinSelfOutput 复制并修改为 Swin2SRSelfOutput 类
class Swin2SRSelfOutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        # 创建一个线性层，输入和输出维度均为 dim
        self.dense = nn.Linear(dim, dim)
        # 创建一个 dropout 层，使用 config 中指定的 dropout 概率
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将输入的 hidden_states 传入 dense 线性层
        hidden_states = self.dense(hidden_states)
        # 将经过线性层的 hidden_states 应用 dropout
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# 从 transformers.models.swinv2.modeling_swinv2.Swinv2Attention 复制并修改为 Swin2SRAttention 类
class Swin2SRAttention(nn.Module):
    def __init__(self, config, dim, num_heads, window_size, pretrained_window_size=0):
        super().__init__()
        # 初始化 self 层，即 Swin2SRSelfAttention 对象
        self.self = Swin2SRSelfAttention(
            config=config,
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            pretrained_window_size=pretrained_window_size
            if isinstance(pretrained_window_size, collections.abc.Iterable)
            else (pretrained_window_size, pretrained_window_size),
        )
        # 初始化 output 层，即 Swin2SRSelfOutput 对象
        self.output = Swin2SRSelfOutput(config, dim)
        # 初始化一个空集合，用于存储剪枝的注意力头信息
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 查找可剪枝的注意力头和其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 对线性层进行剪枝
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储已剪枝的头信息
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
        # 执行自注意力机制，并获取 self_outputs
        self_outputs = self.self(hidden_states, attention_mask, head_mask, output_attentions)
        # 将 self_outputs[0] 作为输入，hidden_states 作为辅助输入，传入 output 层
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果输出注意力信息，则将 attentions 添加到 outputs 中
        outputs = (attention_output,) + self_outputs[1:]  # 如果需要输出 attentions，则添加到 outputs 中
        return outputs


# 从 transformers.models.swin.modeling_swin.SwinIntermediate 复制并修改为 Swin2SRIntermediate 类
class Swin2SRIntermediate(nn.Module):
    # 初始化函数，用于创建一个新的神经网络层
    def __init__(self, config, dim):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个全连接层，将输入维度 dim 映射到 int(config.mlp_ratio * dim) 的输出维度
        self.dense = nn.Linear(dim, int(config.mlp_ratio * dim))
        
        # 根据配置选择隐藏层激活函数，如果配置中隐藏层激活函数是字符串，则从预定义的映射中选择对应的函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            # 否则直接使用配置中指定的隐藏层激活函数
            self.intermediate_act_fn = config.hidden_act

    # 前向传播函数，处理输入的隐藏状态张量并返回处理后的张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入的隐藏状态张量通过全连接层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 将线性变换后的张量输入到中间激活函数中进行非线性变换
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回变换后的张量作为输出
        return hidden_states
# 从transformers.models.swin.modeling_swin.SwinOutput复制并将Swin改为Swin2SR
class Swin2SROutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        # 创建一个线性层，将输入维度乘以config.mlp_ratio，输出维度为dim
        self.dense = nn.Linear(int(config.mlp_ratio * dim), dim)
        # 创建一个Dropout层，以config.hidden_dropout_prob的概率丢弃神经元
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 前向传播函数，首先通过线性层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 然后对处理后的状态应用Dropout操作
        hidden_states = self.dropout(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states


# 从transformers.models.swinv2.modeling_swinv2.Swinv2Layer复制并将Swinv2改为Swin2SR
class Swin2SRLayer(nn.Module):
    def __init__(self, config, dim, input_resolution, num_heads, shift_size=0, pretrained_window_size=0):
        super().__init__()
        # 设置输入分辨率
        self.input_resolution = input_resolution
        # 计算窗口大小和移动尺寸
        window_size, shift_size = self._compute_window_shift(
            (config.window_size, config.window_size), (shift_size, shift_size)
        )
        # 选择第一个维度的窗口大小和移动尺寸
        self.window_size = window_size[0]
        self.shift_size = shift_size[0]
        # 创建Swin2SRAttention层，传入config、dim、num_heads、window_size和pretrained_window_size参数
        self.attention = Swin2SRAttention(
            config=config,
            dim=dim,
            num_heads=num_heads,
            window_size=self.window_size,
            pretrained_window_size=pretrained_window_size
            if isinstance(pretrained_window_size, collections.abc.Iterable)
            else (pretrained_window_size, pretrained_window_size),
        )
        # 创建LayerNorm层，归一化dim维度的输入，eps为config.layer_norm_eps
        self.layernorm_before = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        # 创建Swin2SRDropPath层，如果config.drop_path_rate大于0.0则应用DropPath，否则为恒等映射
        self.drop_path = Swin2SRDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        # 创建Swin2SRIntermediate层，处理输入为config和dim的中间层
        self.intermediate = Swin2SRIntermediate(config, dim)
        # 创建Swin2SROutput层，处理输入为config和dim的输出层
        self.output = Swin2SROutput(config, dim)
        # 创建LayerNorm层，归一化dim维度的输出，eps为config.layer_norm_eps
        self.layernorm_after = nn.LayerNorm(dim, eps=config.layer_norm_eps)

    def _compute_window_shift(self, target_window_size, target_shift_size) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        # 计算窗口大小和移动尺寸的函数，返回目标窗口大小和目标移动尺寸
        window_size = [r if r <= w else w for r, w in zip(self.input_resolution, target_window_size)]
        shift_size = [0 if r <= w else s for r, w, s in zip(self.input_resolution, window_size, target_shift_size)]
        return window_size, shift_size
    ````
        # 根据窗口移动大小生成注意力掩码，用于移位窗口的多头自注意力
        def get_attn_mask(self, height, width, dtype):
            if self.shift_size > 0:
                # 创建一个全零的张量作为图像的注意力掩码
                img_mask = torch.zeros((1, height, width, 1), dtype=dtype)
                # 定义高度和宽度的切片，用于生成多个窗口
                height_slices = (
                    slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None),
                )
                width_slices = (
                    slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None),
                )
                count = 0
                # 在图像的每个窗口位置设置对应的编号
                for height_slice in height_slices:
                    for width_slice in width_slices:
                        img_mask[:, height_slice, width_slice, :] = count
                        count += 1
    
                # 将图像分块，每个块的大小为窗口大小乘以窗口大小
                mask_windows = window_partition(img_mask, self.window_size)
                mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
                # 创建注意力掩码，表示不同窗口之间的相对位置
                attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
                # 使用特定值填充掩码，0位置用0.0填充，非0位置用-100.0填充
                attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            else:
                # 如果不需要移位，返回空的注意力掩码
                attn_mask = None
            return attn_mask
    
        # 在需要时对隐藏状态进行填充，以适应窗口大小的整数倍
        def maybe_pad(self, hidden_states, height, width):
            # 计算需要右侧和底部填充的像素数，使其可以被窗口大小整除
            pad_right = (self.window_size - width % self.window_size) % self.window_size
            pad_bottom = (self.window_size - height % self.window_size) % self.window_size
            # 定义填充值的元组，格式为(top, bottom, left, right, ...)
            pad_values = (0, 0, 0, pad_right, 0, pad_bottom)
            # 对隐藏状态进行填充操作
            hidden_states = nn.functional.pad(hidden_states, pad_values)
            return hidden_states,  def forward(
            self,
            hidden_states: torch.Tensor,
            input_dimensions: Tuple[int, int],
            head_mask: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = False,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 定义函数签名，指定输入和输出类型为 torch.Tensor 的元组
        height, width = input_dimensions
        # 解包输入维度
        batch_size, _, channels = hidden_states.size()
        # 获取隐藏状态的批量大小、高度、宽度和通道数
        shortcut = hidden_states
        # 保存隐藏状态的快捷方式

        # pad hidden_states to multiples of window size
        # 将隐藏状态填充到窗口大小的倍数
        hidden_states = hidden_states.view(batch_size, height, width, channels)
        # 调整隐藏状态的形状为 [batch_size, height, width, channels]
        hidden_states, pad_values = self.maybe_pad(hidden_states, height, width)
        # 调用 maybe_pad 方法，可能对隐藏状态进行填充，同时获取填充值
        _, height_pad, width_pad, _ = hidden_states.shape
        # 解包填充后的隐藏状态的形状

        # cyclic shift
        # 循环移位操作
        if self.shift_size > 0:
            # 如果 shift_size 大于 0
            shifted_hidden_states = torch.roll(hidden_states, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # 在维度 (1, 2) 上对隐藏状态进行负向移位操作
        else:
            shifted_hidden_states = hidden_states
            # 否则，不进行移位操作，保持隐藏状态不变

        # partition windows
        # 划分窗口
        hidden_states_windows = window_partition(shifted_hidden_states, self.window_size)
        # 调用 window_partition 方法，将移位后的隐藏状态划分为窗口
        hidden_states_windows = hidden_states_windows.view(-1, self.window_size * self.window_size, channels)
        # 将划分后的窗口重新视图为 [batch_size * num_windows, window_size * window_size, channels]
        attn_mask = self.get_attn_mask(height_pad, width_pad, dtype=hidden_states.dtype)
        # 调用 get_attn_mask 方法，获取注意力掩码
        if attn_mask is not None:
            attn_mask = attn_mask.to(hidden_states_windows.device)
            # 如果注意力掩码不为空，则将其移到与 hidden_states_windows 相同的设备上

        attention_outputs = self.attention(
            hidden_states_windows, attn_mask, head_mask, output_attentions=output_attentions
        )
        # 调用 attention 方法，进行注意力计算

        attention_output = attention_outputs[0]
        # 获取注意力输出的第一个元素

        attention_windows = attention_output.view(-1, self.window_size, self.window_size, channels)
        # 将注意力输出重新视图为 [batch_size * num_windows, window_size, window_size, channels]
        shifted_windows = window_reverse(attention_windows, self.window_size, height_pad, width_pad)
        # 调用 window_reverse 方法，逆转注意力窗口

        # reverse cyclic shift
        # 逆转循环移位
        if self.shift_size > 0:
            # 如果 shift_size 大于 0
            attention_windows = torch.roll(shifted_windows, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            # 在维度 (1, 2) 上对注意力窗口进行正向移位操作
        else:
            attention_windows = shifted_windows
            # 否则，不进行移位操作，保持注意力窗口不变

        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        # 判断是否进行了填充
        if was_padded:
            attention_windows = attention_windows[:, :height, :width, :].contiguous()
            # 如果进行了填充，则截取注意力窗口的有效部分

        attention_windows = attention_windows.view(batch_size, height * width, channels)
        # 将注意力窗口重新视图为 [batch_size, height * width, channels]
        hidden_states = self.layernorm_before(attention_windows)
        # 调用 layernorm_before 方法，对注意力窗口进行层归一化处理
        hidden_states = shortcut + self.drop_path(hidden_states)
        # 将快捷方式与经过 drop_path 处理后的隐藏状态相加

        layer_output = self.intermediate(hidden_states)
        # 调用 intermediate 方法，生成中间层输出
        layer_output = self.output(layer_output)
        # 调用 output 方法，生成输出层输出
        layer_output = hidden_states + self.drop_path(self.layernorm_after(layer_output))
        # 将隐藏状态与经过 drop_path 和层归一化后的输出相加

        layer_outputs = (layer_output, attention_outputs[1]) if output_attentions else (layer_output,)
        # 如果需要输出注意力，则返回包含注意力输出的元组，否则只返回输出层输出的元组
        return layer_outputs
        # 返回层输出的元组
class Swin2SRStage(nn.Module):
    """
    This corresponds to the Residual Swin Transformer Block (RSTB) in the original implementation.
    """

    def __init__(self, config, dim, input_resolution, depth, num_heads, drop_path, pretrained_window_size=0):
        super().__init__()
        self.config = config  # 初始化模型配置参数
        self.dim = dim  # 初始化模型维度参数

        # 创建包含多个Swin2SRLayer层的ModuleList
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

        # 根据配置参数选择不同的残差连接方式
        if config.resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif config.resi_connection == "3conv":
            # 采用序列化方式创建多层卷积神经网络
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )

        # 创建Swin2SRPatchEmbeddings对象
        self.patch_embed = Swin2SRPatchEmbeddings(config, normalize_patches=False)

        # 创建Swin2SRPatchUnEmbeddings对象
        self.patch_unembed = Swin2SRPatchUnEmbeddings(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        residual = hidden_states  # 保存输入的隐藏状态作为残差

        height, width = input_dimensions  # 获取输入图像的高度和宽度
        for i, layer_module in enumerate(self.layers):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 调用Swin2SRLayer的forward方法进行前向传播
            layer_outputs = layer_module(hidden_states, input_dimensions, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]  # 更新隐藏状态输出

        output_dimensions = (height, width, height, width)  # 设置输出的图像维度

        hidden_states = self.patch_unembed(hidden_states, input_dimensions)  # 反向解嵌入处理
        hidden_states = self.conv(hidden_states)  # 应用卷积层处理隐藏状态
        hidden_states, _ = self.patch_embed(hidden_states)  # 应用图像嵌入处理

        hidden_states = hidden_states + residual  # 加上残差连接

        stage_outputs = (hidden_states, output_dimensions)  # 定义阶段输出结果

        if output_attentions:
            stage_outputs += layer_outputs[1:]  # 如果需要输出注意力，将其添加到输出结果中
        return stage_outputs  # 返回阶段输出结果
    # 初始化函数，接受配置对象和网格大小作为参数
    def __init__(self, config, grid_size):
        # 调用父类初始化方法
        super().__init__()
        # 计算阶段数量，即深度列表的长度
        self.num_stages = len(config.depths)
        # 保存配置对象
        self.config = config
        # 计算丢弃路径率数组，根据配置的丢弃路径率和各个阶段的深度
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]
        # 创建阶段列表，每个阶段是一个Swin2SRStage模块
        self.stages = nn.ModuleList(
            [
                Swin2SRStage(
                    config=config,
                    dim=config.embed_dim,
                    input_resolution=(grid_size[0], grid_size[1]),
                    depth=config.depths[stage_idx],
                    num_heads=config.num_heads[stage_idx],
                    drop_path=dpr[sum(config.depths[:stage_idx]) : sum(config.depths[: stage_idx + 1])],
                    pretrained_window_size=0,
                )
                for stage_idx in range(self.num_stages)
            ]
        )

        # 是否启用梯度检查点，默认为False
        self.gradient_checkpointing = False

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, Swin2SREncoderOutput]:
        # 初始化所有输入尺寸为空元组
        all_input_dimensions = ()
        # 如果需要输出隐藏状态，则初始化为空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力权重，则初始化为空元组
        all_self_attentions = () if output_attentions else None

        # 如果需要输出隐藏状态，则添加当前隐藏状态到all_hidden_states中
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 遍历所有阶段
        for i, stage_module in enumerate(self.stages):
            # 获取当前阶段的头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 如果启用了梯度检查点并且正在训练阶段，则使用梯度检查点函数
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    stage_module.__call__, hidden_states, input_dimensions, layer_head_mask, output_attentions
                )
            else:
                # 否则，直接调用阶段模块进行前向传播
                layer_outputs = stage_module(hidden_states, input_dimensions, layer_head_mask, output_attentions)

            # 更新隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]
            # 更新输入尺寸为当前层输出的维度
            output_dimensions = layer_outputs[1]
            input_dimensions = (output_dimensions[-2], output_dimensions[-1])
            # 将当前层的输出维度添加到所有输入尺寸中
            all_input_dimensions += (input_dimensions,)

            # 如果需要输出隐藏状态，则添加当前隐藏状态到all_hidden_states中
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # 如果需要输出注意力权重，则将当前层的注意力权重添加到all_self_attentions中
            if output_attentions:
                all_self_attentions += layer_outputs[2:]

        # 如果不需要返回字典形式的输出，则返回隐藏状态、所有隐藏状态和所有注意力权重
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        # 否则，返回Swin2SREncoderOutput对象，包含最终隐藏状态、所有隐藏状态和所有注意力权重
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

    # 指定配置类
    config_class = Swin2SRConfig
    # 基础模型前缀
    base_model_prefix = "swin2sr"
    # 主输入名称
    main_input_name = "pixel_values"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        # 初始化模块的权重
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.trunc_normal_(module.weight.data, std=self.config.initializer_range)
            # 如果存在偏置，将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            # 初始化 LayerNorm 的偏置为零，权重为1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


# Swin2SRModel 类的文档字符串
SWIN2SR_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`Swin2SRConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# Swin2SRModel 类的输入文档字符串
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


@add_start_docstrings(
    "The bare Swin2SR Model transformer outputting raw hidden-states without any specific head on top.",
    SWIN2SR_START_DOCSTRING,
)
# Swin2SRModel 类，继承自 Swin2SRPreTrainedModel，用于构建模型
class Swin2SRModel(Swin2SRPreTrainedModel):
    def __init__(self, config):
        # 调用父类构造函数初始化对象
        super().__init__(config)
        # 保存配置信息到对象属性
        self.config = config

        # 根据配置信息设置均值张量
        if config.num_channels == 3 and config.num_channels_out == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.img_range = config.img_range

        # 创建第一个卷积层
        self.first_convolution = nn.Conv2d(config.num_channels, config.embed_dim, 3, 1, 1)
        # 创建嵌入层
        self.embeddings = Swin2SREmbeddings(config)
        # 创建编码器
        self.encoder = Swin2SREncoder(config, grid_size=self.embeddings.patch_embeddings.patches_resolution)

        # 创建层归一化层
        self.layernorm = nn.LayerNorm(config.embed_dim, eps=config.layer_norm_eps)
        # 创建补丁解嵌入层
        self.patch_unembed = Swin2SRPatchUnEmbeddings(config)
        # 创建主体后的卷积层
        self.conv_after_body = nn.Conv2d(config.embed_dim, config.embed_dim, 3, 1, 1)

        # 调用后初始化方法，初始化权重并进行最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回嵌入层的补丁嵌入对象
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 遍历要修剪的头信息，对编码器中对应层的自注意力机制进行修剪
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def pad_and_normalize(self, pixel_values):
        _, _, height, width = pixel_values.size()

        # 1. 执行填充操作
        window_size = self.config.window_size
        modulo_pad_height = (window_size - height % window_size) % window_size
        modulo_pad_width = (window_size - width % window_size) % window_size
        pixel_values = nn.functional.pad(pixel_values, (0, modulo_pad_width, 0, modulo_pad_height), "reflect")

        # 2. 执行归一化操作
        self.mean = self.mean.type_as(pixel_values)
        pixel_values = (pixel_values - self.mean) * self.img_range

        return pixel_values

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
        ) -> Union[Tuple, BaseModelOutput]:
        # 如果没有显式指定，根据配置决定是否输出注意力权重
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果没有显式指定，根据配置决定是否输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果没有显式指定，根据配置决定是否使用返回字典形式
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 准备头部掩码（如果需要）
        # 在头部掩码中，1.0 表示保留该头部
        # attention_probs 的形状为 bsz x n_heads x N x N
        # 输入的 head_mask 形状为 [num_heads] 或 [num_hidden_layers x num_heads]
        # head_mask 被转换为形状 [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, len(self.config.depths))

        _, _, height, width = pixel_values.shape

        # 一些预处理：填充 + 归一化
        pixel_values = self.pad_and_normalize(pixel_values)

        # 第一个卷积层处理像素值
        embeddings = self.first_convolution(pixel_values)
        # 将卷积后的结果传递给嵌入层处理，同时获取输入维度信息
        embedding_output, input_dimensions = self.embeddings(embeddings)

        # 编码器处理嵌入输出
        encoder_outputs = self.encoder(
            embedding_output,
            input_dimensions,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 取出编码器的输出的第一个元素作为序列输出
        sequence_output = encoder_outputs[0]
        # 序列输出经过 LayerNormalization 处理
        sequence_output = self.layernorm(sequence_output)

        # 将序列输出重新映射到原始尺寸上
        sequence_output = self.patch_unembed(sequence_output, (height, width))
        # 经过主体后的卷积操作，加上初始的嵌入值
        sequence_output = self.conv_after_body(sequence_output) + embeddings

        # 如果不使用返回字典形式，则输出为包含序列输出和其他编码器输出的元组
        if not return_dict:
            output = (sequence_output,) + encoder_outputs[1:]
            return output

        # 如果使用返回字典形式，则构造 BaseModelOutput 对象返回
        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
class PixelShuffleUpsampler(nn.Module):
    """PixelShuffleUpsampler module.

    This module performs upsampling using PixelShuffle.

    Args:
        config (`object`):
            Configuration object containing parameters.
        num_features (`int`):
            Number of intermediate features.

    Attributes:
        conv_before_upsample (`nn.Conv2d`):
            Convolutional layer before upsampling.
        activation (`nn.LeakyReLU`):
            LeakyReLU activation function.
        upsample (`Upsample`):
            Upsample module.
        final_convolution (`nn.Conv2d`):
            Final convolutional layer.

    """

    def __init__(self, config, num_features):
        super().__init__()
        
        # Initialize convolution before upsampling
        self.conv_before_upsample = nn.Conv2d(config.embed_dim, num_features, 3, 1, 1)
        # Initialize activation function
        self.activation = nn.LeakyReLU(inplace=True)
        # Initialize upsampling module
        self.upsample = Upsample(config.upscale, num_features)
        # Initialize final convolutional layer
        self.final_convolution = nn.Conv2d(num_features, config.num_channels_out, 3, 1, 1)

    def forward(self, sequence_output):
        # Apply convolution before upsampling
        x = self.conv_before_upsample(sequence_output)
        # Apply activation function
        x = self.activation(x)
        # Apply upsampling using the Upsample module
        x = self.upsample(x)
        # Apply final convolutional layer
        x = self.final_convolution(x)

        return x


class NearestConvUpsampler(nn.Module):
    """NearestConvUpsampler module.

    This module performs upsampling using nearest-neighbor interpolation followed by convolution.

    Args:
        scale (`int`):
            Scale factor for upsampling.
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`):
            Number of output channels.

    Attributes:
        upsample (`nn.Upsample`):
            Upsampling layer.
        conv (`nn.Conv2d`):
            Convolutional layer.

    """
    def __init__(self, config, num_features):
        super().__init__()
        # 检查是否需要进行4倍上采样，否则抛出数值错误异常
        if config.upscale != 4:
            raise ValueError("The nearest+conv upsampler only supports an upscale factor of 4 at the moment.")

        # 第一层卷积，将输入特征维度转换为num_features，卷积核大小为3x3，填充为1
        self.conv_before_upsample = nn.Conv2d(config.embed_dim, num_features, 3, 1, 1)
        # 激活函数，使用LeakyReLU
        self.activation = nn.LeakyReLU(inplace=True)
        # 上采样卷积层1，输入和输出特征维度都为num_features，卷积核大小为3x3，填充为1
        self.conv_up1 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        # 上采样卷积层2，输入和输出特征维度都为num_features，卷积核大小为3x3，填充为1
        self.conv_up2 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        # 高分辨率恢复卷积层，输入和输出特征维度都为num_features，卷积核大小为3x3，填充为1
        self.conv_hr = nn.Conv2d(num_features, num_features, 3, 1, 1)
        # 最终卷积层，将特征维度转换为config.num_channels_out，卷积核大小为3x3，填充为1
        self.final_convolution = nn.Conv2d(num_features, config.num_channels_out, 3, 1, 1)
        # LeakyReLU激活函数，斜率为0.2，inplace操作
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, sequence_output):
        # 序列输出先经过第一层卷积
        sequence_output = self.conv_before_upsample(sequence_output)
        # 经过激活函数
        sequence_output = self.activation(sequence_output)
        # 上采样至原始大小的两倍，并经过LeakyReLU激活函数
        sequence_output = self.lrelu(
            self.conv_up1(torch.nn.functional.interpolate(sequence_output, scale_factor=2, mode="nearest"))
        )
        # 再次上采样至原始大小的四倍，并经过LeakyReLU激活函数
        sequence_output = self.lrelu(
            self.conv_up2(torch.nn.functional.interpolate(sequence_output, scale_factor=2, mode="nearest"))
        )
        # 最终的重建，经过高分辨率恢复卷积层和LeakyReLU激活函数
        reconstruction = self.final_convolution(self.lrelu(self.conv_hr(sequence_output)))
        # 返回重建的结果
        return reconstruction
# 定义像素混洗辅助上采样器模块的类，用于图像超分辨率和恢复任务
class PixelShuffleAuxUpsampler(nn.Module):
    def __init__(self, config, num_features):
        super().__init__()

        # 从配置中获取上采样比例
        self.upscale = config.upscale
        # 定义使用三通道卷积进行双三次插值的卷积层
        self.conv_bicubic = nn.Conv2d(config.num_channels, num_features, 3, 1, 1)
        # 定义用于上采样前的卷积层
        self.conv_before_upsample = nn.Conv2d(config.embed_dim, num_features, 3, 1, 1)
        # 定义激活函数为LeakyReLU
        self.activation = nn.LeakyReLU(inplace=True)
        # 定义用于辅助任务的卷积层，将序列输出映射到通道数为config.num_channels的张量
        self.conv_aux = nn.Conv2d(num_features, config.num_channels, 3, 1, 1)
        # 定义用于辅助任务后续处理的序列卷积和LeakyReLU激活函数的顺序层
        self.conv_after_aux = nn.Sequential(nn.Conv2d(3, num_features, 3, 1, 1), nn.LeakyReLU(inplace=True))
        # 定义上采样模块
        self.upsample = Upsample(config.upscale, num_features)
        # 定义最终的卷积层，将上采样后的特征映射到config.num_channels_out的输出通道
        self.final_convolution = nn.Conv2d(num_features, config.num_channels_out, 3, 1, 1)

    def forward(self, sequence_output, bicubic, height, width):
        # 对双三次插值结果进行卷积操作
        bicubic = self.conv_bicubic(bicubic)
        # 对序列输出进行上采样前的卷积操作
        sequence_output = self.conv_before_upsample(sequence_output)
        # 序列输出经过激活函数处理
        sequence_output = self.activation(sequence_output)
        # 对序列输出进行辅助任务的卷积操作
        aux = self.conv_aux(sequence_output)
        # 经过辅助任务卷积后的序列输出再次进行卷积和激活函数处理
        sequence_output = self.conv_after_aux(aux)
        # 序列输出经过上采样模块，根据指定的高度和宽度进行裁剪
        sequence_output = (
            self.upsample(sequence_output)[:, :, : height * self.upscale, : width * self.upscale]
            + bicubic[:, :, : height * self.upscale, : width * self.upscale]
        )
        # 最终将上采样后的序列输出进行最终卷积操作，生成重建图像
        reconstruction = self.final_convolution(sequence_output)

        return reconstruction, aux


# 使用添加文档字符串装饰器为Swin2SRForImageSuperResolution类添加说明
@add_start_docstrings(
    """
    Swin2SR模型的变压器，顶部带有一个上采样器头部，用于图像超分辨率和恢复。
    """,
    SWIN2SR_START_DOCSTRING,
)
class Swin2SRForImageSuperResolution(Swin2SRPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化Swin2SR模型
        self.swin2sr = Swin2SRModel(config)
        # 获取配置中的上采样器类型和上采样比例
        self.upsampler = config.upsampler
        self.upscale = config.upscale

        # 根据上采样器类型选择对应的上采样器模块
        num_features = 64
        if self.upsampler == "pixelshuffle":
            self.upsample = PixelShuffleUpsampler(config, num_features)
        elif self.upsampler == "pixelshuffle_aux":
            self.upsample = PixelShuffleAuxUpsampler(config, num_features)
        elif self.upsampler == "pixelshuffledirect":
            # 轻量级超分辨率模型，只进行一步上采样
            self.upsample = UpsampleOneStep(config.upscale, config.embed_dim, config.num_channels_out)
        elif self.upsampler == "nearest+conv":
            # 适用于真实世界超分辨率，减少伪影的最近邻插值加卷积上采样器
            self.upsample = NearestConvUpsampler(config, num_features)
        else:
            # 用于图像去噪和JPEG压缩伪影减少的最终卷积层
            self.final_convolution = nn.Conv2d(config.embed_dim, config.num_channels_out, 3, 1, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    # 使用添加文档字符串装饰器为forward方法添加输入说明
    @add_start_docstrings_to_model_forward(SWIN2SR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ImageSuperResolutionOutput, config_class=_CONFIG_FOR_DOC)
    # 定义一个方法 `forward`，用于模型的前向传播
    # 
    # 参数说明：
    # - pixel_values: 可选的 torch.FloatTensor，表示输入的像素值
    # - head_mask: 可选的 torch.FloatTensor，表示注意力头部的掩码
    # - labels: 可选的 torch.LongTensor，表示标签数据
    # - output_attentions: 可选的 bool 值，控制是否输出注意力权重
    # - output_hidden_states: 可选的 bool 值，控制是否输出隐藏状态
    # - return_dict: 可选的 bool 值，控制是否以字典形式返回结果
```