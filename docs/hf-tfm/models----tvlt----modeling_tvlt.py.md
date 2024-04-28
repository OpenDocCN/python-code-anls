# `.\transformers\models\tvlt\modeling_tvlt.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 使用 Apache License, Version 2.0 开源许可
""" PyTorch TVLT 模型。"""

# 导入需要的模块
import collections.abc
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
# 导入所需的自定义模块和函数

# 提供给 TVLT model 的输出类，可能包含隐藏状态和注意力
from ...activations import ACT2FN
# 提供基础模型的输出类
from ...modeling_outputs import BaseModelOutput, SequenceClassifierOutput
# 提供预训练模型的抽象基类
from ...modeling_utils import PreTrainedModel
# 导入用于剪枝的工具函数
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
# 提供一些实用函数
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 导入 TVLT 的配置文件
from .configuration_tvlt import TvltConfig

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 该模型的配置文件文档
_CONFIG_FOR_DOC = "TvltConfig"
# 预训练模型的检查点
_CHECKPOINT_FOR_DOC = "ZinengTang/tvlt-base"

# TVLT 预训练模型的列表
TVLT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "ZinengTang/tvlt-base",
    # 可以在 https://huggingface.co/ZinengTang/tvlt-base 查看所有 TVLT 模型
]

# 定义 TvltModelOutput 类，用于 TVLT 模型的输出
@dataclass
class TvltModelOutput(ModelOutput):
    """
    Class for TvltModel's outputs, with potential hidden states and attentions.
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的隐藏状态的序列。
        last_pixel_hidden_state (`torch.FloatTensor` of shape `(batch_size, pixel_sequence_length, hidden_size)`):
            模型最后一层的像素序列的隐藏状态。
        last_audio_hidden_state (`torch.FloatTensor` of shape `(batch_size, audio_sequence_length, hidden_size)`):
            模型最后一层的音频序列的隐藏状态。
        pixel_label_masks (`torch.FloatTensor` of shape `(batch_size, pixel_patch_length)`):
            表示哪些像素补丁被屏蔽（1）和哪些没有被屏蔽（0）的张量。
        audio_label_masks (`torch.FloatTensor` of shape `(batch_size, audio_patch_length)`):
            表示哪些音频补丁被屏蔽（1）和哪些没有被屏蔽（0）的张量。
        pixel_ids_restore (`torch.LongTensor` of shape `(batch_size, pixel_patch_length)`):
            包含像素屏蔽的 ID 排列的张量。
        audio_ids_restore (`torch.LongTensor` of shape `(batch_size, audio_patch_length)`):
            包含音频屏蔽的 ID 排列的张量。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            模型每一层的隐藏状态的元组。元组包含一个张量（用于嵌入的输出）和每一层的输出，形状为 `(batch_size, sequence_length, hidden_size)`。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            每一层的注意力权重的元组。每个元组元素是一个张量，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
    """

    last_hidden_state: torch.FloatTensor = None
    last_pixel_hidden_state: torch.FloatTensor = None
    last_audio_hidden_state: torch.FloatTensor = None
    pixel_label_masks: torch.LongTensor = None
    audio_label_masks: torch.LongTensor = None
    pixel_ids_restore: torch.LongTensor = None
    audio_ids_restore: torch.LongTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
```  
@dataclass
class TvltDecoderOutput(ModelOutput):
    """
    Class for TvltDecoder's outputs, with potential hidden states and attentions.

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, patch_size ** 2 * num_channels)`):
            Pixel reconstruction logits.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class TvltForPreTrainingOutput(ModelOutput):
    """
    Class for TvltForPreTraining's outputs, with potential hidden states and attentions.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`):
            Pixel reconstruction loss.
        matching_logits (`torch.FloatTensor` of shape `(batch_size, 1)`):
            Matching objective logits.
        pixel_logits (`torch.FloatTensor` of shape
            `(batch_size, pixel_patch_length, image_patch_size ** 3 * pixel_num_channels)`): Pixel reconstruction
            logits.
        audio_logits (`torch.FloatTensor` of shape
            `(batch_size, audio_patch_length, image_patch_size[0] * image_patch_size[1])`): Audio reconstruction
            logits.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    # 用于存储匹配（matching）的逻辑回归结果的张量，初始化为 None
    matching_logits: torch.FloatTensor = None
    # 用于存储像素（pixel）的逻辑回归结果的张量，初始化为 None
    pixel_logits: torch.FloatTensor = None
    # 用于存储音频（audio）的逻辑回归结果的张量，初始化为 None
    audio_logits: torch.FloatTensor = None
    # 用于存储隐藏状态（hidden states）的元组，元组中包含一个或多个张量，初始化为 None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 用于存储注意力权重（attentions）的元组，元组中包含一个或多个张量，初始化为 None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# 生成用于遮蔽像素的噪声
def generate_pixel_mask_noise(pixel_values, pixel_mask=None, mask_ratio=0.75):
    # 获取输入张量的批大小和序列长度
    batch_size, seq_len = pixel_values.shape[:2]
    # 生成[0, 1]之间的随机噪声
    noise = torch.rand((batch_size, seq_len), device=pixel_values.device)
    # 计算需要保留的部分长度
    len_keep = int(seq_len * (1 - mask_ratio))
    # 返回噪声和保留长度
    return noise, len_keep

# 生成用于遮蔽音频的噪声
def generate_audio_mask_noise(audio_values, audio_mask=None, mask_ratio=0.75, mask_type="patch-level", freq_len=8):
    # 获取输入张量的批大小和序列长度
    batch_size, seq_len = audio_values.shape[:2]
    # 根据遮蔽类型生成噪声
    if mask_type == "frame-level":
        # 将序列长度划分为块，每块包含 freq_len 个元素
        num_time_patches = seq_len // freq_len
        # 生成随机噪声，并重复扩展成与序列长度一致的张量
        noise = (
            torch.rand(batch_size, num_time_patches, device=audio_values.device)
            .unsqueeze(-1)
            .repeat(1, 1, freq_len)
            .view(batch_size, seq_len)
        )
    elif mask_type == "patch-level":
        # 生成[0, 1]之间的随机噪声
        noise = torch.rand(batch_size, seq_len, device=audio_values.device)
    # 计算需要保留的部分长度
    len_keep = int(seq_len * (1 - mask_ratio))
    # 返回噪声和保留长度
    return noise, len_keep

# 对序列执行随机遮蔽
def random_masking(sequence, noise, len_keep, attention_masks=None):
    # 获取输入张量的批大小、序列长度和隐藏维度
    batch_size, seq_len, hidden_dim = sequence.shape
    # 对噪声进行升序排序，得到索引 ids_shuffle
    ids_shuffle = torch.argsort(noise, dim=1)
    # 根据 ids_shuffle 对序列进行重新排序
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    # 保留前 len_keep 个元素
    ids_keep = ids_shuffle[:, :len_keep]
    sequence_masked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, hidden_dim))
    # 生成二进制遮蔽掩码
    label_masks = torch.ones([batch_size, seq_len], device=sequence.device)
    label_masks[:, :len_keep] = 0
    # 根据 ids_restore 对遮蔽掩码进行重新排序
    label_masks = torch.gather(label_masks, dim=1, index=ids_restore)
    # 如果有注意力掩码，则应用之
    if attention_masks is not None:
        label_masks *= attention_masks
        attention_masks = torch.gather(attention_masks, dim=1, index=ids_keep)
    # 返回遮蔽后的序列、注意力掩码和遮蔽掩码
    return sequence_masked, attention_masks, label_masks, ids_restore

# 构建像素嵌入
class TvltPixelEmbeddings(nn.Module):
    # 初始化方法
    def __init__(self, config):
        super().__init__()
        # 构建像素分块嵌入
        self.patch_embeddings = TvltPixelPatchEmbeddings(config)
        # 获取每个图像的分块数量
        self.num_patches_per_image = self.patch_embeddings.num_patches_per_image
        # 定义类型嵌入、时间嵌入和位置嵌入
        self.type_embed_v = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.temporal_embed = nn.Parameter(torch.zeros(1, config.num_frames, config.hidden_size))
        self.pos_embed_v = nn.Parameter(torch.zeros(1, self.num_patches_per_image, config.hidden_size))
        # 存储配置信息
        self.config = config
    # 前向传播函数，用于处理输入的像素数值和注意力掩码
    def forward(self, pixel_values, attention_masks=None):
        # 获取输入像素值的形状信息
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        
        # 使用 patch_embeddings 函数对像素值进行嵌入操作
        embeddings = self.patch_embeddings(pixel_values)
        
        # 在嵌入结果上添加位置编码
        embeddings += self.pos_embed_v.repeat(1, num_frames, 1)
        
        # 在嵌入结果上添加时间编码
        embeddings += torch.repeat_interleave(self.temporal_embed[:, :num_frames], self.num_patches_per_image, dim=1)
        
        # 在嵌入结果上添加类型编码
        embeddings += self.type_embed_v
        
        # 返回嵌入结果和注意力掩码
        return embeddings, attention_masks
class TvltAudioEmbeddings(nn.Module): 
    """构建音频嵌入的类，包括补丁和位置嵌入。"""

    def __init__(self, config):
        super().__init__()

        self.patch_embeddings = TvltAudioPatchEmbeddings(config) 
        self.num_patches = self.patch_embeddings.num_patches 

        self.type_embed_a = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) 
        self.num_freq_patches = config.frequency_length // config.audio_patch_size[1] 
        self.pos_embed_a = nn.Parameter(torch.zeros(1, self.num_patches // self.num_freq_patches, config.hidden_size)) 
        self.freq_embed = nn.Parameter(torch.zeros(1, self.num_freq_patches, config.hidden_size)) 

        self.num_freq_patches = config.frequency_length // config.audio_patch_size[1] 
        self.config = config 

    def forward(self, audio_values, attention_masks=None): 
        # 创建补丁嵌入
        embeddings = self.patch_embeddings(audio_values)

        num_time_patches = embeddings.size(1) // self.num_freq_patches 
        embeddings += self.freq_embed.repeat(1, num_time_patches, 1) 
        embeddings += torch.repeat_interleave(self.pos_embed_a[:, :num_time_patches], self.num_freq_patches, dim=1) 
        embeddings += self.type_embed_a 

        return embeddings, attention_masks


class TvltPixelPatchEmbeddings(nn.Module): 
    """
    此类将形状为`(batch_size, num_channels, height, width)`的`pixel_values`转换为形状为`(batch_size, seq_length, hidden_size)`的初始
    `hidden_states`（补丁嵌入），以供Transformer使用。
    """

    def __init__(self, config): 
        super().__init()
        image_size, patch_size = config.image_size, config.image_patch_size
        num_channels, hidden_size = config.num_image_channels, config.hidden_size 

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size) 
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size) 
        num_patches_per_image = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0]) 
        self.image_size = image_size 
        self.patch_size = patch_size 
        self.num_channels = num_channels 
        self.num_patches_per_image = num_patches_per_image 
        self.hidden_size = hidden_size

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size) 
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # 获取输入张量的形状信息
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        # 检查通道数是否与配置中设置的通道数相匹配
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 检查输入图像尺寸是否与模型的期望尺寸匹配
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]}."
            )

        # 对输入的图像像素值进行重塑，将其形状变为(batch_size * num_frames, num_channels, height, width)
        pixel_values = pixel_values.reshape(batch_size * num_frames, num_channels, height, width)
        # 使用projection模型对像素值进行投影，并将结果展平并转置
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        # 将投影后的结果再次重塑，变为(batch_size, num_frames * self.num_patches_per_image, self.hidden_size)
        embeddings = embeddings.reshape(batch_size, num_frames * self.num_patches_per_image, self.hidden_size)

        # 返回最终的嵌入向量张量
        return embeddings
class TvltAudioPatchEmbeddings(nn.Module):
    """
    This class turns `audio_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    # 初始化函数，接受配置对象作为参数
    def __init__(self, config):
        super().__init__()
        spectrogram_length, frequency_length, patch_size = (
            config.spectrogram_length,
            config.frequency_length,
            config.audio_patch_size,
        )
        num_channels, hidden_size = config.num_audio_channels, config.hidden_size

        spectrogram_size = (spectrogram_length, frequency_length)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (spectrogram_size[1] // patch_size[1]) * (spectrogram_size[0] // patch_size[0])
        patch_shape = (spectrogram_size[0] // patch_size[0], spectrogram_size[1] // patch_size[1])
        self.spectrogram_size = spectrogram_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.patch_shape = patch_shape

        # 使用卷积层将输入进行投影
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    # 前向传播函数，接受音频值张量并返回嵌入张量
    def forward(self, audio_values: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = audio_values.shape
        # 检查通道的数量是否与配置中设置的一致
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 检查输入音频大小是否与模型期望的大小匹配
        if height > self.spectrogram_size[0] or width != self.spectrogram_size[1]:
            raise ValueError(
                f"Input audio size ({height}*{width}) doesn't match model"
                f" ({self.spectrogram_size[0]}*{self.spectrogram_size[1]})."
            )
        # 将输入值经过投影并展平，然后进行转置操作
        embeddings = self.projection(audio_values).flatten(2).transpose(1, 2)

        return embeddings


# 以下是从transformers.models.vilt.modeling_vilt.ViltSelfAttention复制并修改为TvltSelfAttention
class TvltSelfAttention(nn.Module):
    # 初始化方法，接受配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 检查隐藏层大小是否符合注意力头数的要求
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            # 如果不符合要求，则抛出数值错误异常
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        # 设置注意力头数
        self.num_attention_heads = config.num_attention_heads
        # 计算注意力头大小
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        # 计算全部头的大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化查询、键、值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        # 初始化注意力概率的dropout层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    # 转换形状以适应注意力计算
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播方法
    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        # 计算混合的查询层
        mixed_query_layer = self.query(hidden_states)

        # 计算键和值的转置形状
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 计算原始的注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 如果存在注意力遮罩，则应用注意力遮罩
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # 将注意力分数归一化为概率
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # 对注意力概率进行dropout
        attention_probs = self.dropout(attention_probs)

        # 如果存在头遮罩，则应用头遮罩
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文层
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # 如果需要输出注意力信息，则返回上下文层和注意力概率
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
class TvltSelfOutput(nn.Module):
    """
    The residual connection is defined in TvltLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """
    # TvltSelfOutput类继承自nn.Module，用于定义Tvlt模型的自注意力机制输出层
    def __init__(self, config: TvltConfig) -> None:
        super().__init__()
        # 定义全连接层，输入和输出维度均为config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义dropout层，概率为config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 全连接层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # dropout层处理全连接层输出
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class TvltAttention(nn.Module):
    # 定义TvltAttention类，用于Tvlt模型的自注意力机制
    def __init__(self, config: TvltConfig):
        super().__init__()
        # 初始化TvltSelfAttention和TvltSelfOutput模块
        self.attention = TvltSelfAttention(config)
        self.output = TvltSelfOutput(config)
        self.pruned_heads = set()

    # 头部修剪函数
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 寻找可修剪头部和对应索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # 修剪线性层
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储修剪头部
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 前向传播函数
    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        # 通过self.attention计算self_outputs
        self_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions)

        # 通过self.output计算attention_output
        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class TvltIntermediate(nn.Module):
    # 定义TvltIntermediate类，用于Tvlt模型的中间层
    def __init__(self, config: TvltConfig) -> None:
        super().__init__()
        # 定义全连接层，输入维度为config.hidden_size，输出维度为config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
    # 定义一个方法，用于前向传播，输入参数为隐藏状态张量，输出为张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用全连接层对隐藏状态进行处理
        hidden_states = self.dense(hidden_states)
        # 使用激活函数对处理后的隐藏状态进行处理
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states
# 从transformers.models.vilt.modeling_vilt.ViltOutput复制了TvltOutput类，并将Vilt->Tvlt
class TvltOutput(nn.Module):
    def __init__(self, config: TvltConfig) -> None:
        super().__init__()
        # 创建一个线性全连接层，用于将intermediate_size映射到hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个dropout层，用于防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 通过全连接层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 在处理后的隐藏状态上应用dropout
        hidden_states = self.dropout(hidden_states)

        # 将处理后的隐藏状态与输入张量相加
        hidden_states = hidden_states + input_tensor

        return hidden_states


# 从transformers.models.vilt.modeling_vilt.ViltLayer复制了TvltLayer类，并将Vilt->Tvlt
class TvltLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config):
        super().__init__()
        # 设置feed forward的块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        # 初始化TvltAttention、TvltIntermediate、TvltOutput和LayerNorm层
        self.attention = TvltAttention(config)
        self.intermediate = TvltIntermediate(config)
        self.output = TvltOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        # 在self-attention之前应用layernorm
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        # 第一个残差连接
        hidden_states = attention_output + hidden_states.to(attention_output.device)

        # 在self-attention之后也应用layernorm
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # 第二个残差连接在这里执行
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


# 从transformers.models.vilt.modeling_vilt.ViltEncoder复制了TvltEncoder类，并将Vilt->Tvlt
class TvltEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化TvltLayer模块列表，数量为num_hidden_layers
        self.config = config
        self.layer = nn.ModuleList([TvltLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        ):
            # 如果不输出隐藏状态，则初始化为一个空元组，否则初始化为None
            all_hidden_states = () if output_hidden_states else None
            # 如果不输出注意力，则初始化为一个空元组，否则初始化为None
            all_self_attentions = () if output_attentions else None

            # 遍历每个层的模块
            for i, layer_module in enumerate(self.layer):
                # 如果输出隐藏状态，则将当前隐藏状态添加到all_hidden_states中
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                # 如果头部蒙版不为空，则获取当前层的蒙版，否则初始化为None
                layer_head_mask = head_mask[i] if head_mask is not None else None

                # 如果启用梯度检查点并且处于训练模式，则调用_gradient_checkpointing_func函数
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        layer_module.__call__,
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        output_attentions,
                    )
                else:
                    # 否则，调用当前层模块的__call__函数
                    layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, output_attentions)

                # 更新隐藏状态为当前层输出的隐藏状态
                hidden_states = layer_outputs[0]

                # 如果输出注意力，则将当前层输出的注意力添加到all_self_attentions中
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)

            # 如果输出隐藏状态，则将最终的隐藏状态添加到all_hidden_states中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果不返回字典，则返回包含非空元素的元组
            if not return_dict:
                return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
            # 否则，返回BaseModelOutput对象
            return BaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
            )
# TvltPreTrainedModel 类，用于处理权重初始化、下载和加载预训练模型的简单接口
class TvltPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # TvltConfig 作为配置类
    config_class = TvltConfig
    # 基础模型前缀为 "tvlt"
    base_model_prefix = "tvlt"
    # 主要输入名称为 "pixel_values"
    main_input_name = "pixel_values"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    # 初始化权重的方法
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果是线性层或卷积层
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 略有不同于 TF 版本，使用 normal 分布初始化权重
            # 参考：https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置，则初始化为 0
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是 LayerNorm 层
        elif isinstance(module, nn.LayerNorm):
            # 偏置初始化为 0
            module.bias.data.zero_()
            # 权重初始化为 1
            module.weight.data.fill_(1.0)


# TVLT_START_DOCSTRING 用于存储模型的文档字符串
TVLT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`TvltConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# TVLT_INPUTS_DOCSTRING 用于存储输入的文档字符串
TVLT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`):
            # 像素数值。像素数值可以使用 [`TvltProcessor`] 获得。有关详细信息，请参阅 [`TvltProcessor.__call__`]。

        audio_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            # 音频数值。音频数值可以使用 [`TvltProcessor`] 获得。有关详细信息，请参阅 [`TvltProcessor.__call__`]。

        pixel_mask (`torch.FloatTensor` of shape `(batch_size, num_pixel_patches)`):
            # 像素掩码。像素掩码可以使用 [`TvltProcessor`] 获得。有关详细信息，请参阅 [`TvltProcessor.__call__`]。

        audio_mask (`torch.FloatTensor` of shape `(batch_size, num_audio_patches)`):
            # 音频掩码。音频掩码可以使用 [`TvltProcessor`] 获得。有关详细信息，请参阅 [`TvltProcessor.__call__`]。

        pixel_values_mixed (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`):
            # Tvlt 视听匹配中混合正负样本的像素数值。可以使用 [`TvltProcessor`] 获得。有关详细信息，请参阅 [`TvltProcessor.__call__`]。

        pixel_mask_mixed (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            # pixel_values_mixed 的像素掩码。可以使用 [`TvltProcessor`] 获得。有关详细信息，请参阅 [`TvltProcessor.__call__`]。

        mask_pixel (`bool`, *optional*):
            # 是否对 MAE 任务遮罩像素。仅在 TvltForPreTraining 中设置为 True。

        mask_audio (`bool`, *optional*):
            # 是否对 MAE 任务遮罩音频。仅在 TvltForPreTraining 中设置为 True。

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多详细信息，请参阅返回张量下的 `attentions`。

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多详细信息，请参阅返回张量下的 `hidden_states`。

        return_dict (`bool`, *optional*):
            # 是否返回一个 [`~utils.ModelOutput`] 而不是一个普通元组。
"""
@add_start_docstrings(
    "The bare TVLT Model transformer outputting raw hidden-states without any specific head on top.",
    TVLT_START_DOCSTRING,
)
class TvltModel(TvltPreTrainedModel):
    # TVLT 模型的基本定义，继承自 TvltPreTrainedModel
    def __init__(self, config):
        # 初始化方法，接收配置参数并进行初始化
        super().__init__(config)
        # 保存配置，初始化各嵌入层和编码器
        self.config = config
        self.pixel_embeddings = TvltPixelEmbeddings(config)
        self.audio_embeddings = TvltAudioEmbeddings(config)
        self.encoder = TvltEncoder(config)
        self.cls_embedding = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        if config.use_mean_pooling:
            self.layernorm = None
        else:
            self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 获取像素和音频的嵌入层
        return self.pixel_embeddings.patch_embeddings, self.audio_embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        # 剪枝模型的注意力头
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(TVLT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TvltModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        # TVLT 模型的前向传播方法，接收像素和音频数据以及相关参数
        pixel_values: torch.FloatTensor,
        audio_values: torch.FloatTensor,
        pixel_mask: Optional[torch.FloatTensor] = None,
        audio_mask: Optional[torch.FloatTensor] = None,
        mask_pixel: bool = False,
        mask_audio: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
class TvltDecoder(nn.Module):
    # TVLT 解码器的定义
    def __init__(self, config):
        # 初始化方法，接收配置参数并进行初始化
        super().__init__()
        # 复制配置并设置解码器的隐藏层大小等参数
        decoder_config = deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        self.decoder_layers = nn.ModuleList(
            [TvltLayer(decoder_config) for _ in range(config.decoder_num_hidden_layers)]
        )
        # 设置解码器的 LayerNorm 层
        self.layernorm = nn.LayerNorm(config.decoder_hidden_size, eps=config.layer_norm_eps)
        # 设置梯度检查点为 False，保存配置
        self.gradient_checkpointing = False
        self.config = config

    def forward(
        self,
        # TVLT 解码器的前向传播方法，接收隐藏状态以及输出参数
        hidden_states,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        ):
            # 如果输出隐藏状态，初始化为空元组；否则为 None
            all_hidden_states = () if output_hidden_states else None
            # 如果输出注意力，初始化为空元组；否则为 None
            all_self_attentions = () if output_attentions else None
            # 遍历解码器层
            for i, layer_module in enumerate(self.decoder_layers):
                # 如果输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                # 如果启用了渐变检查点并且在训练中
                if self.gradient_checkpointing and self.training:
                    # 使用梯度检查点函数进行处理
                    layer_outputs = self._gradient_checkpointing_func(
                        layer_module.__call__,
                        hidden_states,
                        None,
                        output_attentions,
                    )
                else:
                    # 否则使用层的调用方法计算输出
                    layer_outputs = layer_module(hidden_states, output_attentions=output_attentions)

                # 更新隐藏状态为当前层输出的第一个元素
                hidden_states = layer_outputs[0]

                # 如果输出注意力，将当前层的注意力添加到 all_self_attentions 中
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)

            # 如果输出隐藏状态，将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 预测者投影，计算 logits
            logits = self.layernorm(hidden_states)

            # 如果不返回字典，返回值为元组中的非 None 元素
            if not return_dict:
                return tuple(v for v in [logits, all_hidden_states, all_self_attentions] if v is not None)
            # 否则返回 TvltDecoderOutput 对象，包括 logits、hidden_states 和 attentions
            return TvltDecoderOutput(logits=logits, hidden_states=all_hidden_states, attentions=all_self_attentions)
# 使用装饰器添加模型文档字符串，介绍了 TVLT 模型的特点以及自监督预训练的目的
@add_start_docstrings(
    "The TVLT Model transformer with the decoder on top for self-supervised pre-training.",
    TVLT_START_DOCSTRING,
)
# TvltForPreTraining 类，继承自 TvltPreTrainedModel 类
class TvltForPreTraining(TvltPreTrainedModel):
    # 初始化方法，接受一个配置参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 将配置参数保存到实例变量中
        self.config = config

        # 保存任务匹配标志和 MAE 任务标志到实例变量中
        self.task_matching = config.task_matching
        self.task_mae = config.task_mae
        # 如果既没有任务匹配标志也没有 MAE 任务标志，则抛出 ValueError 异常
        if not (self.task_matching or self.task_mae):
            raise ValueError("Must set at least one of matching task and MAE task to true")

        # 创建 TVLT 模型实例
        self.tvlt = TvltModel(config)

        # 如果有任务匹配标志，则创建匹配头实例
        if self.task_matching:
            self.matching_head = TvltMatchingHead(config)

        # 如果有 MAE 任务标志，则创建相关组件
        if self.task_mae:
            # 创建线性层，用于编码器到解码器的映射
            self.encoder_to_decoder = nn.Linear(config.hidden_size, config.decoder_hidden_size, bias=True)

            # 创建像素和音频的掩码令牌参数
            self.pixel_mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_hidden_size))
            self.audio_mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_hidden_size))

            # 创建解码器实例
            self.decoder = TvltDecoder(config)

            # 设置解码器相关参数
            decoder_hidden_size = config.decoder_hidden_size
            num_frames = config.num_frames
            num_patches_per_image = self.tvlt.pixel_embeddings.num_patches_per_image
            self.decoder_pixel_pos_embed = nn.Parameter(torch.zeros(1, num_patches_per_image, decoder_hidden_size))
            self.decoder_temporal_embed = nn.Parameter(torch.zeros(1, config.num_frames, decoder_hidden_size))
            self.decoder_pixel_type_embed = nn.Parameter(torch.zeros(1, 1, decoder_hidden_size))
            num_audio_patches = self.tvlt.audio_embeddings.num_patches
            num_freq_patches = config.frequency_length // config.audio_patch_size[1]
            self.decoder_audio_pos_embed = nn.Parameter(
                torch.zeros(1, num_audio_patches // num_freq_patches, decoder_hidden_size)
            )
            self.decoder_freq_embed = nn.Parameter(torch.zeros(1, num_freq_patches, decoder_hidden_size))
            self.decoder_audio_type_embed = nn.Parameter(torch.zeros(1, 1, decoder_hidden_size))

            # 设置像素和音频 MAE 头
            pixel_mae_output_dim = self.config.image_patch_size[0] ** 2 * self.config.num_image_channels
            self.pixel_mae_head = TvltMAEHead(config, pixel_mae_output_dim)
            audio_mae_output_dim = (
                self.config.audio_patch_size[0] * self.config.audio_patch_size[1] * self.config.num_audio_channels
            )
            self.audio_mae_head = TvltMAEHead(config, audio_mae_output_dim)

            # 保存一些参数到实例变量中
            self.num_frames = num_frames
            self.num_patches_per_image = num_patches_per_image
            self.num_freq_patches = num_freq_patches
            self.image_patch_size = config.image_patch_size
            self.audio_patch_size = config.audio_patch_size

        # 初始化权重并进行最终处理
        self.post_init()
    # 将输入的像素值按指定大小切割成块状
    def patchify_pixel(self, pixel_values):
        """
        pixel_values: [batch_size, num_frames, 3, height, width]
        """
        # 获取输入的像素值张量的形状信息
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        # 计算图像的高度和宽度方向上可以切割成的块的数量
        num_patches_height = pixel_values.shape[3] // self.image_patch_size[0]
        num_patches_width = pixel_values.shape[4] // self.image_patch_size[1]
        # 将像素值张量重塑为块状
        patchified_pixel_values = pixel_values.reshape(
            shape=(
                batch_size,
                num_frames,
                num_channels,
                num_patches_height,
                self.image_patch_size[0],
                num_patches_width,
                self.image_patch_size[1],
            )
        )
        # 使用 Einstein 求和符号重排张量的维度顺序
        patchified_pixel_values = torch.einsum("ntchpwq->nthwpqc", patchified_pixel_values)
        # 将块状像素值张量重新调整形状
        patchified_pixel_values = patchified_pixel_values.reshape(
            shape=(
                batch_size,
                num_patches_height * num_patches_width * num_frames,
                self.image_patch_size[0] * self.image_patch_size[1] * num_channels,
            )
        )
        # 返回切割后的像素值张量
        return patchified_pixel_values

    # 将输入的音频值按指定大小切割成块状
    def patchify_audio(self, audio_values):
        """
        audio_values: [batch_size, 1, height, width]
        """
        # 获取输入的音频值张量的形状信息
        batch_size, num_channels, height, width = audio_values.shape
        # 计算音频的高度和宽度方向上可以切割成的块的数量
        num_patches_height = height // self.audio_patch_size[0]
        num_patches_width = width // self.audio_patch_size[1]
        # 将音频值张量重塑为块状
        patchified_audio_values = audio_values.reshape(
            shape=(
                batch_size,
                num_channels,
                num_patches_height,
                self.audio_patch_size[0],
                num_patches_width,
                self.audio_patch_size[1],
            )
        )
        # 使用 Einstein 求和符号重排张量的维度顺序
        patchified_audio_values = torch.einsum("nchpwq->nhwpqc", patchified_audio_values)
        # 将块状音频值张量重新调整形状
        patchified_audio_values = patchified_audio_values.reshape(
            shape=(
                batch_size,
                num_patches_height * num_patches_width,
                self.audio_patch_size[0] * self.audio_patch_size[1] * num_channels,
            )
        )
        # 返回切割后的音频值张量
        return patchified_audio_values

    # 计算像素值的均方误差损失
    def pixel_mae_loss(self, pixel_values, pixel_predictions, mask):
        # 将输入的像素值切割成块状
        patchified_pixel_values = self.patchify_pixel(pixel_values)
        # 计算损失，即预测值与真实值之差的平方
        loss = (pixel_predictions - patchified_pixel_values) ** 2
        # 沿着最后一个维度计算均值，即每个块的平均损失
        loss = loss.mean(dim=-1)  # [batch_size, pixel_pixel_length], mean loss per patch
        # 将损失乘以掩码并求和，再除以掩码的总数，得到被移除块的平均损失
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        # 返回损失值
        return loss

    # 计算音频值的均方误差损失
    def audio_mae_loss(self, audio_values, audio_predictions, mask):
        # 将输入的音频值切割成块状
        patchified_audio_values = self.patchify_audio(audio_values)
        # 计算损失，即预测值与真实值之差的平方
        loss = (audio_predictions - patchified_audio_values) ** 2
        # 沿着最后一个维度计算均值，即每个块的平均损失
        loss = loss.mean(dim=-1)  # [batch_size, audio_pixel_length], mean loss per patch
        # 将损失乘以掩码并求和，再除以掩码的总数，得到被移除块的平均损失
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        # 返回损失值
        return loss
    # 将预测的掩码序列拼接到原始序列上的函数
    def concatenate_mask(self, mask_token, sequence, ids_restore):
        # 获取输入序列的批次大小、长度和维度
        batch_size, seq_length, dim = sequence.shape
        # 根据恢复索引的大小重复掩码标记，以填充到原始序列长度
        mask_tokens = mask_token.repeat(batch_size, ids_restore.shape[1] - seq_length, 1)
        # 将原始序列和掩码标记拼接起来
        padded_sequence = torch.cat([sequence, mask_tokens], dim=1)
        # 根据恢复索引将序列顺序恢复到原始顺序
        padded_sequence = torch.gather(
            padded_sequence, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, dim)
        )
        return padded_sequence
    
    # 定义模型的前向传播过程
    @add_start_docstrings_to_model_forward(TVLT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TvltForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor, # 输入的图像数据
        audio_values: torch.FloatTensor, # 输入的音频数据
        pixel_mask: Optional[torch.FloatTensor] = None, # 图像数据的掩码
        audio_mask: Optional[torch.FloatTensor] = None, # 音频数据的掩码
        labels: Optional[torch.LongTensor] = None, # 标签数据
        pixel_values_mixed: Optional[torch.FloatTensor] = None, # 混合图像数据
        pixel_mask_mixed: Optional[torch.FloatTensor] = None, # 混合图像数据的掩码
        output_attentions: Optional[bool] = None, # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None, # 是否输出隐藏状态
        return_dict: Optional[bool] = None # 是否以字典形式返回结果
    ):
# 创建一个名为 TvltPooler 的类，继承自 nn.Module
class TvltPooler(nn.Module):
    # 初始化方法，接受一个 config 参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个线性层，输入维度为 config.hidden_size，输出维度为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个激活函数层，使用双曲正切函数
        self.activation = nn.Tanh()

    # 前向传播方法，接受 hidden_states 参数
    def forward(self, hidden_states):
        # 提取 hidden_states 中的第一个 token 的数据
        first_token_tensor = hidden_states[:, 0]
        # 将第一个 token 的数据传入线性层
        pooled_output = self.dense(first_token_tensor)
        # 通过激活函数层处理得到的结果
        pooled_output = self.activation(pooled_output)
        # 返回处理后的结果
        return pooled_output


# 创建一个名为 TvltMatchingHead 的类，继承自 nn.Module
class TvltMatchingHead(nn.Module):
    # 初始化方法，接受一个 config 参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个 TvltPooler 类的实例
        self.pooler = TvltPooler(config)
        # 创建一个线性层，输入维度为 config.hidden_size，输出维度为 1
        self.fc = nn.Linear(config.hidden_size, 1)

    # 前向传播方法，接受 hidden_states 参数
    def forward(self, hidden_states):
        # 通过 pooler 处理 hidden_states，再将结果传入线性层
        hidden_states = self.fc(self.pooler(hidden_states))
        # 返回处理后的结果
        return hidden_states


# 创建一个名为 TvltMAEHead 的类，继承自 nn.Module
class TvltMAEHead(nn.Module):
    # 初始化方法，接受一个 config 参数和一个可选的 output_dim 参数
    def __init__(self, config, output_dim=None):
        # 调用父类的初始化方法
        super().__init__()
        # 将 config 参数存储到实例变量中
        self.config = config
        # 创建一个线性层，输入维度为 config.decoder_hidden_size，输出维度为 output_dim
        self.decoder = nn.Linear(config.decoder_hidden_size, output_dim)

    # 前向传播方法，接受 hidden_states 参数
    def forward(self, hidden_states):
        # 通过线性层处理 hidden_states
        hidden_states = self.decoder(hidden_states)
        # 返回处理后的结果
        return hidden_states


# 创建一个名为 TvltForAudioVisualClassification 的类，继承自 TvltPreTrainedModel 类
@add_start_docstrings(
    """
    Tvlt Model transformer with a classifier head on top (an MLP on top of the final hidden state of the [CLS] token)
    for audiovisual classification tasks, e.g. CMU-MOSEI Sentiment Analysis and Audio to Video Retrieval.
    """,
    TVLT_START_DOCSTRING,
)
class TvltForAudioVisualClassification(TvltPreTrainedModel):
    # 初始化方法，接受一个 config 参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建一个 TvltModel 类的实例
        self.tvlt = TvltModel(config)

        # 分类器头部
        self.classifier = nn.Sequential(
            # 创建一个线性层，输入维度为 config.hidden_size，输出维度为 config.hidden_size * 2
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            # 创建一个 LayerNorm 层，输入维度为 config.hidden_size * 2，指定 epsilon 参数为 config.layer_norm_eps
            nn.LayerNorm(config.hidden_size * 2, eps=config.layer_norm_eps),
            # GELU 激活函数层
            nn.GELU(),
            # 创建一个线性层，输入维度为 config.hidden_size * 2，输出维度为 config.num_labels
            nn.Linear(config.hidden_size * 2, config.num_labels),
        )
        # 将 config 参数存储到实例变量中
        self.config = config

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法，接受指定的输入和可选的参数
    @add_start_docstrings_to_model_forward(TVLT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        audio_values: torch.FloatTensor,
        pixel_mask: Optional[torch.FloatTensor] = None,
        audio_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    # 定义一个方法，接受输入的像素值、音频值和标签，返回分类器的输出
    def forward(
        pixel_values: torch.FloatTensor,
        audio_values: torch.FloatTensor,
        pixel_mask: Optional[torch.FloatTensor] = None,
        audio_mask: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.FloatTensor], SequenceClassifierOutput]:
        # 如果 return_dict 为 None，则使用配置中的 use_return_dict 值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # 调用 tvlt 模型，传入像素值、音频值，以及其它参数
        outputs = self.tvlt(
            pixel_values,
            audio_values,
            pixel_mask=pixel_mask,
            audio_mask=audio_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 从输出中取出序列输出的第一个元素
        sequence_output = outputs[0][:, 0]
        # 将序列输出传入分类器，得到 logit 值
        logits = self.classifier(sequence_output)  # rank value
    
        # 初始化损失值
        loss = None
        # 如果标签不为空
        if labels is not None:
            # 根据配置中的损失类型选择损失函数
            if self.config.loss_type == "regression":
                loss_fct = MSELoss()
                # 计算均方误差损失
                loss = loss_fct(logits, labels)
            elif self.config.loss_type == "classification":
                loss_fct = CrossEntropyLoss()
                # 计算交叉熵损失
                loss = loss_fct(logits, labels)
    
        # 如果不返回字典
        if not return_dict:
            # 组装输出结果
            output = (logits,) + outputs[4:]
            return ((loss,) + output) if loss is not None else output
    
        # 返回分类器输出包装类对象
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```