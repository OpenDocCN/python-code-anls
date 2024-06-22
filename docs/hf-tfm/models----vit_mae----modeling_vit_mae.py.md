# `.\transformers\models\vit_mae\modeling_vit_mae.py`

```py
# 引入需要的库和模块
import collections.abc  # 引入collections模块中的abc子模块，用于抽象基类
import math  # 引入math模块，用于数学运算
from copy import deepcopy  # 从copy模块中引入deepcopy函数，用于深拷贝对象
from dataclasses import dataclass  # 从dataclasses模块引入dataclass装饰器，用于定义数据类
from typing import Optional, Set, Tuple, Union  # 引入typing模块中的一些类型提示工具

import numpy as np  # 引入numpy库，用于数值计算
import torch  # 引入torch库，用于构建神经网络
import torch.utils.checkpoint  # 引入torch.utils.checkpoint模块，用于checkpointing技术
from torch import nn  # 从torch库中引入nn模块，用于构建神经网络

# 引入Hugging Face的相关工具和模块
from ...activations import ACT2FN  # 从当前目录下的activations模块中引入ACT2FN变量
from ...modeling_outputs import BaseModelOutput  # 从当前目录下的modeling_outputs模块中引入BaseModelOutput类
from ...modeling_utils import PreTrainedModel  # 从当前目录下的modeling_utils模块中引入PreTrainedModel类
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer  # 从当前目录下的pytorch_utils模块引入两个函数
from ...utils import (  # 从当前目录下的utils模块引入多个函数和类
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_vit_mae import ViTMAEConfig  # 从当前目录下的configuration_vit_mae模块引入ViTMAEConfig类

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义模型配置和检查点的文档字符串信息
_CONFIG_FOR_DOC = "ViTMAEConfig"
_CHECKPOINT_FOR_DOC = "facebook/vit-mae-base"

# 定义预训练模型的存档列表
VIT_MAE_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/vit-mae-base",
    # 可查看所有ViTMAE模型的链接
    # https://huggingface.co/models?filter=vit_mae
]

# 定义ViTMAE模型输出类，继承自ModelOutput类
@dataclass
class ViTMAEModelOutput(ModelOutput):
    """
    Class for ViTMAEModel's outputs, with potential hidden states and attentions.
    """
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Tensor indicating which patches are masked (1) and which are not (0).
        ids_restore (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Tensor containing the original index of the (shuffled) masked patches.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    # 定义变量并初始化为None
    last_hidden_state: torch.FloatTensor = None
    mask: torch.LongTensor = None
    ids_restore: torch.LongTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# 定义 ViTMAEDecoderOutput 类，用于存储 ViTMAEDecoder 的输出，包括潜在的隐藏状态和注意力权重
@dataclass
class ViTMAEDecoderOutput(ModelOutput):
    """
    Class for ViTMAEDecoder's outputs, with potential hidden states and attentions.

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, patch_size ** 2 * num_channels)`):
            Pixel reconstruction logits.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True` is passed):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# 定义 ViTMAEForPreTrainingOutput 类，用于存储 ViTMAEForPreTraining 的输出，包括潜在的隐藏状态和注意力权重
@dataclass
class ViTMAEForPreTrainingOutput(ModelOutput):
    """
    Class for ViTMAEForPreTraining's outputs, with potential hidden states and attentions.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`):
            Pixel reconstruction loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, patch_size ** 2 * num_channels)`):
            Pixel reconstruction logits.
        mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Tensor indicating which patches are masked (1) and which are not (0).
        ids_restore (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Tensor containing the original index of the (shuffled) masked patches.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True` is passed):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True` is passed):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    # 定义一个名为 logits 的变量，类型为 torch.FloatTensor，初始值为 None
    logits: torch.FloatTensor = None
    
    # 定义一个名为 mask 的变量，类型为 torch.LongTensor，初始值为 None
    mask: torch.LongTensor = None
    
    # 定义一个名为 ids_restore 的变量，类型为 torch.LongTensor，初始值为 None
    ids_restore: torch.LongTensor = None
    
    # 定义一个名为 hidden_states 的变量，类型为可选的元组，其中元素为 torch.FloatTensor 类型，初始值为 None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    
    # 定义一个名为 attentions 的变量，类型为可选的元组，其中元素为 torch.FloatTensor 类型，初始值为 None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
def get_2d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    """
    Create 2D sin/cos positional embeddings.

    Args:
        embed_dim (`int`):
            Embedding dimension.
        grid_size (`int`):
            The grid height and width.
        add_cls_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add a classification (CLS) token.

    Returns:
        (`torch.FloatTensor` of shape (grid_size*grid_size, embed_dim) or (1+grid_size*grid_size, embed_dim): the
        position embeddings (with or without classification token)
    """
    # Create an array representing the grid height
    grid_h = np.arange(grid_size, dtype=np.float32)
    # Create an array representing the grid width
    grid_w = np.arange(grid_size, dtype=np.float32)
    # Create a meshgrid of the grid height and width
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    # Stack the grid arrays along a new axis
    grid = np.stack(grid, axis=0)

    # Reshape the grid array to a 4D array
    grid = grid.reshape([2, 1, grid_size, grid_size])
    # Get 2D sin/cos positional embeddings from the grid
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    # If add_cls_token is True, concatenate zeros to represent a classification token
    if add_cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    # Return the position embeddings
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    # use half of dimensions to encode grid_h
    # Get 1D sin/cos positional embeddings from the grid height
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    # Get 1D sin/cos positional embeddings from the grid width
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    # Concatenate the embeddings along the last dimension
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    # Create an array of evenly spaced values representing omega
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    # Calculate omega values
    omega = 1.0 / 10000**omega  # (D/2,)

    # Reshape the position array
    pos = pos.reshape(-1)  # (M,)
    # Compute outer product of position and omega
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    # Compute sine and cosine of the outer product
    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    # Concatenate sine and cosine embeddings along the last dimension
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class ViTMAEEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.

    """

    def __init__(self, config):
        super().__init__()

        # Define a learnable parameter representing the CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        # Initialize patch embeddings
        self.patch_embeddings = ViTMAEPatchEmbeddings(config)
        # Get the number of patches
        self.num_patches = self.patch_embeddings.num_patches
        # Initialize position embeddings with zeros, including an extra position for CLS token
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, config.hidden_size), requires_grad=False
        )
        self.config = config
        # Initialize weights
        self.initialize_weights()
    def initialize_weights(self):
        # 初始化位置嵌入（position embeddings），使用 sin-cos 嵌入并冻结
        pos_embed = get_2d_sincos_pos_embed(
            self.position_embeddings.shape[-1], int(self.patch_embeddings.num_patches**0.5), add_cls_token=True
        )
        # 将位置嵌入数据转换为张量并复制给位置嵌入张量
        self.position_embeddings.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # 将 patch_embeddings 初始化为类似 nn.Linear 的方式（而不是 nn.Conv2d）
        w = self.patch_embeddings.projection.weight.data
        # 使用 Xavier 初始化权重
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # 使用正态分布初始化 cls_token，timm 的 trunc_normal_(std=.02) 实际上等效于 normal_(std=0.02)，因为截断太大（2.）
        torch.nn.init.normal_(self.cls_token, std=self.config.initializer_range)

    def random_masking(self, sequence, noise=None):
        """
        对每个样本进行随机掩码，通过样本排序（argsort）进行样本随机化。每个样本的随机化是通过随机噪音实现的。

        Args:
            sequence (`torch.LongTensor` of shape `(batch_size, sequence_length, dim)`)
            noise (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) 主要用于测试目的，控制随机性并保持可重现性
        """
        batch_size, seq_length, dim = sequence.shape
        len_keep = int(seq_length * (1 - self.config.mask_ratio))

        if noise is None:
            # 生成 [0, 1] 范围内的随机噪音
            noise = torch.rand(batch_size, seq_length, device=sequence.device)

        # 对每个样本的噪音进行排序
        ids_shuffle = torch.argsort(noise, dim=1)  # 升序：小的保留，大的移除
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # 保留第一个子集
        ids_keep = ids_shuffle[:, :len_keep]
        # 使用 gather 函数根据索引收集数据
        sequence_unmasked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, dim))

        # 生成二进制掩码：0 表示保留，1 表示移除
        mask = torch.ones([batch_size, seq_length], device=sequence.device)
        mask[:, :len_keep] = 0
        # 通过恢复顺序来获取二进制掩码
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return sequence_unmasked, mask, ids_restore

    def forward(self, pixel_values, noise=None):
        batch_size, num_channels, height, width = pixel_values.shape
        # 将像素值转换为 patch embeddings
        embeddings = self.patch_embeddings(pixel_values)

        # 添加位置嵌入，但不包括 cls token
        embeddings = embeddings + self.position_embeddings[:, 1:, :]

        # 掩码：长度 -> 长度 * config.mask_ratio
        embeddings, mask, ids_restore = self.random_masking(embeddings, noise)

        # 添加 cls token
        cls_token = self.cls_token + self.position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
        # 在 embeddings 中连接 cls token
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        return embeddings, mask, ids_restore
寸、通道数以及隐藏层尺寸
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size
        # 如果图像尺寸和补丁尺寸不是可迭代对象，将它们转换为元组
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        # 计算补丁数
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        # 设置类属性
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        # 定义投影层，将图像补丁映射到隐藏表示
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values):
        # 获取输入的张量维度信息
        batch_size, num_channels, height, width = pixel_values.shape
        # 检查输入张量的通道数是否与配置中的通道数相匹配
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 检查输入图像的尺寸是否与配置中的图像尺寸相匹配
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        # 将输入的像素值通过投影层映射为隐藏表示，并进行形状调整
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return x


# Copied from transformers.models.vit.modeling_vit.ViTSelfAttention ViT->ViTMAE
class ViTMAESelfAttention(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        # 调用父类构造函数
        super().__init__()
        # 检查隐藏尺寸是否是注意力头数的整数倍
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        # 初始化注意力头数和每个注意力头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 定义查询、键和值的线性转换层
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        # 定义dropout层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    # 将输入张量进行维度转换，使得最后两个维度变成多个头的形式
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # 计算新的张量形状，保留前面的维度并添加多个头的维度
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # 将张量重塑为新的形状
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)  # 调整维度顺序以便后续计算

    # Transformer 的前向传播函数
    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 计算查询向量
        mixed_query_layer = self.query(hidden_states)

        # 计算键向量，并进行维度转换以适应注意力计算
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        # 计算值向量，并进行维度转换以适应注意力计算
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        # 对混合的查询向量进行维度转换以适应注意力计算
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 计算原始的注意力分数，即查询向量和键向量的点积
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # 对注意力分数进行缩放
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 将注意力分数归一化为概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 使用 dropout 对注意力概率进行随机失活
        attention_probs = self.dropout(attention_probs)

        # 如果有头部掩码，则应用头部掩码
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文张量，即注意力概率与值向量的加权和
        context_layer = torch.matmul(attention_probs, value_layer)

        # 调整上下文张量的维度顺序
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # 重塑上下文张量的形状，以便后续处理
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # 如果需要输出注意力权重，则返回上下文张量和注意力概率
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
# 从transformers.models.vit.modeling_vit.ViTSelfOutput复制，并将ViT更改为ViTMAE
class ViTMAESelfOutput(nn.Module):
    """
    The residual connection is defined in ViTMAELayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        # 创建线性层，用于self-attention后的输出变换
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建dropout层，用于防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 线性变换
        hidden_states = self.dense(hidden_states)
        # dropout
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# 从transformers.models.vit.modeling_vit.ViTAttention复制，并将ViT更改为ViTMAE
class ViTMAEAttention(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        # 创建ViTMAESelfAttention层
        self.attention = ViTMAESelfAttention(config)
        # 创建ViTMAESelfOutput层
        self.output = ViTMAESelfOutput(config)
        # 用于存储被修剪的注意力头
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        # 找到需要修剪的注意力头并返回它们的索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # 修剪线性层
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储修剪的注意力头
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 通过self-attention层得到输出
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        # 将self-attention的输出作为输入，通过ViTMAESelfOutput层得到最终输出
        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # 如果有的话，添加注意力值
        return outputs


# 从transformers.models.vit.modeling_vit.ViTIntermediate复制，并将ViT更改为ViTMAE
class ViTMAEIntermediate(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        # 创建线性层，用于将ViTMAEAttention的输出变换为下一层的输入
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果配置中的隐藏激活函数是字符串，则使用预定义的激活函数；否则，使用配置中指定的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        # 对输入的 hidden_states 执行全连接操作
        hidden_states = self.dense(hidden_states)
        # 对全连接后的 hidden_states 执行激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回处理后的 hidden_states
        return hidden_states
# 从transformers.models.vit.modeling_vit.ViTOutput ViT->ViTMAE中复制过来的类ViTMAEOutput
class ViTMAEOutput(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        # 创建一个全连接层，输入尺寸为config.intermediate_size，输出尺寸为config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个丢弃层，根据config.hidden_dropout_prob进行丢弃
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 通过全连接层处理hidden_states
        hidden_states = self.dense(hidden_states)
        # 丢弃一部分hidden_states，以减少过拟合
        hidden_states = self.dropout(hidden_states)
        # 将处理后的hidden_states与input_tensor相加
        hidden_states = hidden_states + input_tensor
        # 返回处理后的hidden_states
        return hidden_states


# 从transformers.models.vit.modeling_vit.ViTLayer中复制过来的类ViTMAELayer，对应timm实现中的Block类
class ViTMAELayer(nn.Module):
    """这对应于timm实现中的Block类。"""

    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        # 初始化ViTMAELayer类的各种属性
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ViTMAEAttention(config)
        self.intermediate = ViTMAEIntermediate(config)
        self.output = ViTMAEOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 使用self-attention处理输入的hidden_states，并返回输出
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # 在ViTMAE中，self-attention之前应用layernorm
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # 如果输出注意力权重，则添加self-attention

        # 第一个残差连接
        hidden_states = attention_output + hidden_states

        # 在ViTMAE中，self-attention之后也应用layernorm
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # 在这里执行第二个残差连接
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


# 从transformers.models.vit.modeling_vit.ViTEncoder中复制过来的ViTMAEEncoder类
class ViTMAEEncoder(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        # 初始化ViTMAEEncoder类的各种属性
        self.config = config
        self.layer = nn.ModuleList([ViTMAELayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    # 这是一个 Transformer 模型的 forward 方法的一部分
    def forward(
        self,
        hidden_states,
        head_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        # 如果需要输出隐藏状态，则创建一个空的元组来存储所有的隐藏状态
        # 如果不需要输出隐藏状态，则设置为 None
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力权重，则创建一个空的元组来存储所有的注意力权重
        # 如果不需要输出注意力权重，则设置为 None
        all_self_attentions = () if output_attentions else None
    
        # 遍历每一个 Transformer 层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
    
            # 获取当前层的 head_mask
            layer_head_mask = head_mask[i] if head_mask is not None else None
    
            # 如果使用了梯度检查点技术并且处于训练阶段
            if self.gradient_checkpointing and self.training:
                # 使用 _gradient_checkpointing_func 方法计算当前层的输出
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                # 直接调用当前层的 forward 方法计算输出
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)
    
            # 更新隐藏状态
            hidden_states = layer_outputs[0]
    
            # 如果需要输出注意力权重，则将当前层的注意力权重添加到 all_self_attentions 中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
    
        # 如果需要输出隐藏状态，则将最后一层的隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
    
        # 如果不需要返回字典，则返回一个元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
    
        # 否则返回一个 BaseModelOutput 对象
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
class ViTMAEPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    
    # 设置config_class为ViTMAEConfig，用于配置模型
    config_class = ViTMAEConfig
    # 设置base_model_prefix为"vit"，用于指示基本模型前缀
    base_model_prefix = "vit"
    # 设置main_input_name为"pixel_values"，用于指示主输入名称
    main_input_name = "pixel_values"
    # 设置supports_gradient_checkpointing为True，支持梯度检查点
    supports_gradient_checkpointing = True

    # 初始化模型权重
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果是线性层或卷积层
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 稍微与TF版本不同，PyTorch使用正态分布初始化权重
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置项，则将其初始化为0
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是LayerNorm层
        elif isinstance(module, nn.LayerNorm):
            # 将偏置项初始化为0
            module.bias.data.zero_()
            # 将权重初始化为1
            module.weight.data.fill_(1.0)


VIT_MAE_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ViTMAEConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

VIT_MAE_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`ViTImageProcessor.__call__`]
            for details.

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
    "The bare ViTMAE Model transformer outputting raw hidden-states without any specific head on top.",
    VIT_MAE_START_DOCSTRING,
)
# ViTMAEModel类继承自ViTMAEPreTrainedModel
class ViTMAEModel(ViTMAEPreTrainedModel):
    # 初始化函数，接受配置参数并调用父类初始化函数
    def __init__(self, config):
        super().__init__(config)
        # 设置实例变量config
        self.config = config

        # 创建嵌入层对象
        self.embeddings = ViTMAEEmbeddings(config)
        # 创建编码器对象
        self.encoder = ViTMAEEncoder(config)

        # 初始化 LayerNorm 层
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    # 剪枝模型中的注意力头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 遍历需要剪枝的层和对应的注意力头
        for layer, heads in heads_to_prune.items():
            # 调用编码器对象中的剪枝函数
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 前向传播函数
    @add_start_docstrings_to_model_forward(VIT_MAE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ViTMAEModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        noise: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ViTMAEModelOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, ViTMAEModel
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
        >>> model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> last_hidden_states = outputs.last_hidden_state
        ```py"""

        # 如果 output_attentions 是 None，则使用 config 中的 output_attentions
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果 output_hidden_states 是 None，则使用 config 中的 output_hidden_states
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果 return_dict 是 None，则使用 config 中的 use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果 pixel_values 为 None，则抛出 ValueError 异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 准备需要的头部遮罩
        # head_mask 中的 1.0 表示保留该头部
        # attention_probs 的形状是 bsz x n_heads x N x N
        # 输入的 head_mask 的形状是 [num_heads] 或者 [num_hidden_layers x num_heads]
        # 并且将 head_mask 转换成形状 [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # 嵌入层输出、遮罩和恢复的 ids
        embedding_output, mask, ids_restore = self.embeddings(pixel_values, noise=noise)

        # 编码器的输出
        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 序列输出是编码器输出的第一个元素
        sequence_output = encoder_outputs[0]
        # 对序列输出进行 LayerNormalization
        sequence_output = self.layernorm(sequence_output)

        # 如果不使用 return_dict，则返回元组：序列输出、遮罩、恢复的 ids，还有其他 encoder_outputs
        if not return_dict:
            return (sequence_output, mask, ids_restore) + encoder_outputs[1:]

        # 使用 return_dict=True，则返回 ViTMAEModelOutput 对象
        return ViTMAEModelOutput(
            last_hidden_state=sequence_output,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
class ViTMAEDecoder(nn.Module):
    # ViTMAEDecoder 类定义
    def __init__(self, config, num_patches):
        # 初始化方法
        super().__init__()
        # 初始化 ViTMAEDecoder 类的父类
        self.decoder_embed = nn.Linear(config.hidden_size, config.decoder_hidden_size, bias=True)
        # 创建一个线性层，用于将隐藏状态映射到解码器隐藏大小
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_hidden_size))
        # 初始化一个参数张量，用于表示掩码令牌
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.decoder_hidden_size), requires_grad=False
        )  # fixed sin-cos embedding
        # 初始化一个参数张量，用于表示固定的正弦-余弦位置嵌入

        decoder_config = deepcopy(config)
        # 使用深拷贝复制配置
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        # 修改解码器配置参数
        self.decoder_layers = nn.ModuleList(
            [ViTMAELayer(decoder_config) for _ in range(config.decoder_num_hidden_layers)]
        )
        # 使用解码器配置创建多个 ViTMAELayer 实例，并添加到模块列表

        self.decoder_norm = nn.LayerNorm(config.decoder_hidden_size, eps=config.layer_norm_eps)
        # 使用 LayerNorm 对解码器隐藏状态进行归一化
        self.decoder_pred = nn.Linear(
            config.decoder_hidden_size, config.patch_size**2 * config.num_channels, bias=True
        )  # encoder to decoder
        # 创建线性层，用于将解码器隐藏状态映射到可输出的位置
        self.gradient_checkpointing = False
        self.config = config
        self.initialize_weights(num_patches)

    def initialize_weights(self, num_patches):
        # 初始化权重的方法
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(num_patches**0.5), add_cls_token=True
        )
        # 获取二维正弦-余弦位置嵌入
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        # 将正弦-余弦位置嵌入数据复制到模型中

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=self.config.initializer_range)
        # 使用正态分布初始化掩码令牌

    def forward(
        self,
        hidden_states,
        ids_restore,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        # 前向传播方法
    # 在这里缺少一个函数的声明，因此这一部分代码无法单独注释
    # 嵌入 tokens
    x = self.decoder_embed(hidden_states)
    
    # 将 mask tokens 添加到序列中
    mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
    x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # 没有 cls token
    x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # 重排
    x = torch.cat([x[:, :1, :], x_], dim=1)  # 添加 cls token
    
    # 添加位置嵌入
    hidden_states = x + self.decoder_pos_embed
    
    # 应用 Transformer 层（块）
    all_hidden_states = () if output_hidden_states else None
    all_self_attentions = () if output_attentions else None
    for i, layer_module in enumerate(self.decoder_layers):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
    
        # 如果启用渐变检查点并且正在训练，则使用渐变检查点函数
        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                layer_module.__call__,
                hidden_states,
                None,
                output_attentions,
            )
        else:
            layer_outputs = layer_module(hidden_states, head_mask=None, output_attentions=output_attentions)
    
        hidden_states = layer_outputs[0]
    
        if output_attentions:
            all_self_attentions = all_self_attentions + (layer_outputs[1],)
    
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)
    
    hidden_states = self.decoder_norm(hidden_states)
    
    # 预测器投影
    logits = self.decoder_pred(hidden_states)
    
    # 删除 cls token
    logits = logits[:, 1:, :]
    
    # 根据 return_dict 决定返回结果
    if not return_dict:
        return tuple(v for v in [logits, all_hidden_states, all_self_attentions] if v is not None)
    return ViTMAEDecoderOutput(
        logits=logits,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )
# 为自监督预训练的 ViTMAE 模型添加解码器的变压器模型。
# 通过提供脚本来训练自定义数据，训练脚本位于 examples 目录下的 image-pretraining 文件夹中
# 继承自 ViTMAEPreTrainedModel 类
class ViTMAEForPreTraining(ViTMAEPreTrainedModel):
    # 初始化方法，接受一个配置参数
    def __init__(self, config):
        # 调用基类的初始化方法
        super().__init__(config)
        # 保存配置
        self.config = config
        # 创建 ViT 模型
        self.vit = ViTMAEModel(config)
        # 创建解码器
        self.decoder = ViTMAEDecoder(config, num_patches=self.vit.embeddings.num_patches)
        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.vit.embeddings.patch_embeddings

    # 剪枝模型中的头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 将像素值分块化
    def patchify(self, pixel_values):
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Patchified pixel values.
        """
        patch_size, num_channels = self.config.patch_size, self.config.num_channels
        # 检查像素值的形状
        if (pixel_values.shape[2] != pixel_values.shape[3]) or (pixel_values.shape[2] % patch_size != 0):
            raise ValueError("Make sure the pixel values have a squared size that is divisible by the patch size")
        if pixel_values.shape[1] != num_channels:
            raise ValueError(
                "Make sure the number of channels of the pixel values is equal to the one set in the configuration"
            )

        # 分块化处理
        batch_size = pixel_values.shape[0]
        num_patches_one_direction = pixel_values.shape[2] // patch_size
        patchified_pixel_values = pixel_values.reshape(
            batch_size, num_channels, num_patches_one_direction, patch_size, num_patches_one_direction, patch_size
        )
        patchified_pixel_values = torch.einsum("nchpwq->nhwpqc", patchified_pixel_values)
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size, num_patches_one_direction * num_patches_one_direction, patch_size**2 * num_channels
        )
        return patchified_pixel_values
        def unpatchify(self, patchified_pixel_values):
            """
            Args:
                patchified_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                    Patchified pixel values.

            Returns:
                `torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`:
                    Pixel values.
            """
            # 获取配置中的补丁大小和通道数
            patch_size, num_channels = self.config.patch_size, self.config.num_channels
            # 计算每个方向上的补丁数量
            num_patches_one_direction = int(patchified_pixel_values.shape[1] ** 0.5)
            # 检查是否补丁数量可以被平方
            if num_patches_one_direction**2 != patchified_pixel_values.shape[1]:
                raise ValueError("Make sure that the number of patches can be squared")

            # 反补丁
            batch_size = patchified_pixel_values.shape[0]
            # 重塑张量形状以进行反补丁
            patchified_pixel_values = patchified_pixel_values.reshape(
                batch_size,
                num_patches_one_direction,
                num_patches_one_direction,
                patch_size,
                patch_size,
                num_channels,
            )
            # 使用einsum对张量进行操作并重塑形状
            patchified_pixel_values = torch.einsum("nhwpqc->nchpwq", patchified_pixel_values)
            # 最终重塑返回像素值
            pixel_values = patchified_pixel_values.reshape(
                batch_size,
                num_channels,
                num_patches_one_direction * patch_size,
                num_patches_one_direction * patch_size,
            )
            return pixel_values

        def forward_loss(self, pixel_values, pred, mask):
            """
            Args:
                pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                    Pixel values.
                pred (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                    Predicted pixel values.
                mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                    Tensor indicating which patches are masked (1) and which are not (0).

            Returns:
                `torch.FloatTensor`: Pixel reconstruction loss.
            """
            # 将输入像素值转换为目标张量
            target = self.patchify(pixel_values)
            # 如果配置中启用了像素损失归一化
            if self.config.norm_pix_loss:
                # 计算均值和方差，并对目标张量进行归一化
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.0e-6) ** 0.5

            # 计算像素值预测和目标张量之间的平方差
            loss = (pred - target) ** 2
            # 计算每个补丁的平均损失
            loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

            # 计算被掩盖的补丁的平均损失
            loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
            return loss

        @add_start_docstrings_to_model_forward(VIT_MAE_INPUTS_DOCSTRING)
        @replace_return_docstrings(output_type=ViTMAEForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
        def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            noise: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ViTMAEForPreTrainingOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, ViTMAEForPreTraining
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
        >>> model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> loss = outputs.loss
        >>> mask = outputs.mask
        >>> ids_restore = outputs.ids_restore
        ```py"""
        # 如果 return_dict 不为空，则使用 return_dict；否则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将 pixel_values 和其他参数传递给 Vit 模型进行处理
        outputs = self.vit(
            pixel_values,
            noise=noise,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从输出中提取最后隐藏层内容、还原的 ids 和 mask
        latent = outputs.last_hidden_state
        ids_restore = outputs.ids_restore
        mask = outputs.mask

        # 使用 decoder 处理 latent 和 ids_restore，得到 logits
        decoder_outputs = self.decoder(latent, ids_restore)
        logits = decoder_outputs.logits  # shape (batch_size, num_patches, patch_size*patch_size*num_channels)

        # 计算损失值
        loss = self.forward_loss(pixel_values, logits, mask)

        if not return_dict:
            # 如果不返回字典，则将结果输出为元组
            output = (logits, mask, ids_restore) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 以 ViTMAEForPreTrainingOutput 对象的形式返回结果
        return ViTMAEForPreTrainingOutput(
            loss=loss,
            logits=logits,
            mask=mask,
            ids_restore=ids_restore,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```