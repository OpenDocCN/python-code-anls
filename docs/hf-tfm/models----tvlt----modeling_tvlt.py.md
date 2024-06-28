# `.\models\tvlt\modeling_tvlt.py`

```
# 导入标准库和第三方库
import collections.abc  # 导入 collections.abc 模块，用于检查对象是否是可迭代的序列
import math  # 导入 math 模块，提供基本的数学函数实现
from copy import deepcopy  # 从 copy 模块中导入 deepcopy 函数，用于深度复制对象
from dataclasses import dataclass  # 从 dataclasses 模块中导入 dataclass 装饰器，用于简化类的定义
from typing import Optional, Tuple, Union  # 导入类型提示，声明可选类型、元组、联合类型

import torch  # 导入 PyTorch 库
import torch.utils.checkpoint  # 导入 PyTorch 的 checkpoint 模块，提供基于检查点的内存优化方法
from torch import nn  # 从 PyTorch 导入 nn 模块，用于神经网络的构建
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss  # 从 nn 模块中导入损失函数类

# 导入自定义模块和函数
from ...activations import ACT2FN  # 从相对路径导入 ACT2FN，用于激活函数
from ...modeling_outputs import BaseModelOutput, SequenceClassifierOutput  # 导入模型输出类
from ...modeling_utils import PreTrainedModel  # 导入预训练模型基类
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer  # 导入 PyTorch 辅助函数
from ...utils import (
    ModelOutput,  # 导入模型输出基类
    add_start_docstrings,  # 导入函数用于添加文档字符串的装饰器
    add_start_docstrings_to_model_forward,  # 导入函数用于添加模型前向传播文档字符串的装饰器
    logging,  # 导入 logging 模块，用于日志记录
    replace_return_docstrings,  # 导入函数用于替换返回值文档字符串的装饰器
)

# 导入 TVLT 模型配置类
from .configuration_tvlt import TvltConfig

# 获取全局日志记录器
logger = logging.get_logger(__name__)

# 定义用于文档的配置名称
_CONFIG_FOR_DOC = "TvltConfig"
# 定义用于文档的检查点名称
_CHECKPOINT_FOR_DOC = "ZinengTang/tvlt-base"

# 预定义 TVLT 预训练模型的存档列表
TVLT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "ZinengTang/tvlt-base",
    # 查看所有 TVLT 模型的列表地址
    # https://huggingface.co/ZinengTang/tvlt-base
]


@dataclass
# TvltModelOutput 类，用于 TvltModel 的输出，包含潜在的隐藏状态和注意力
class TvltModelOutput(ModelOutput):
    """
    Class for TvltModel's outputs, with potential hidden states and attentions.
    """
    # 定义函数参数，表示模型输出的隐藏状态及其它相关信息
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层输出的隐藏状态序列。
        last_pixel_hidden_state (`torch.FloatTensor` of shape `(batch_size, pixel_sequence_length, hidden_size)`):
            模型最后一层输出的像素序列的隐藏状态。
        last_audio_hidden_state (`torch.FloatTensor` of shape `(batch_size, audio_sequence_length, hidden_size)`):
            模型最后一层输出的音频序列的隐藏状态。
        pixel_label_masks (`torch.FloatTensor` of shape `(batch_size, pixel_patch_length)`):
            表示哪些像素补丁被掩盖（置为1），哪些未被掩盖（置为0）的张量。
        audio_label_masks (`torch.FloatTensor` of shape `(batch_size, audio_patch_length)`):
            表示哪些音频补丁被掩盖（置为1），哪些未被掩盖（置为0）的张量。
        pixel_ids_restore (`torch.LongTensor` of shape `(batch_size, pixel_patch_length)`):
            像素掩盖的id排列顺序的张量。
        audio_ids_restore (`torch.LongTensor` of shape `(batch_size, audio_patch_length)`):
            音频掩盖的id排列顺序的张量。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            元组，包含模型每层的隐藏状态张量（嵌入输出和每层的输出），形状为 `(batch_size, sequence_length, hidden_size)`。
            当参数 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            元组，包含模型每层的注意力权重张量，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
            注意力softmax后的注意力权重，用于计算自注意力头中的加权平均。
            当参数 `output_attentions=True` 或 `config.output_attentions=True` 时返回。
    """

    # 初始化函数参数，均为None，表示这些参数在调用时可以传入具体的张量数据
    last_hidden_state: torch.FloatTensor = None
    last_pixel_hidden_state: torch.FloatTensor = None
    last_audio_hidden_state: torch.FloatTensor = None
    pixel_label_masks: torch.LongTensor = None
    audio_label_masks: torch.LongTensor = None
    pixel_ids_restore: torch.LongTensor = None
    audio_ids_restore: torch.LongTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
# TvltDecoderOutput 类用于存储 TvltDecoder 模型的输出结果，可能包含隐藏状态和注意力信息

@dataclass
class TvltDecoderOutput(ModelOutput):
    """
    Class for TvltDecoder's outputs, with potential hidden states and attentions.

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, patch_size ** 2 * num_channels)`):
            Pixel reconstruction logits. 像素重构的逻辑回归输出。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs. 模型每一层输出的隐藏状态，包括初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads. 经过注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


# TvltForPreTrainingOutput 类用于存储 TvltForPreTraining 模型的输出结果，可能包含隐藏状态和注意力信息

@dataclass
class TvltForPreTrainingOutput(ModelOutput):
    """
    Class for TvltForPreTraining's outputs, with potential hidden states and attentions.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`):
            Pixel reconstruction loss. 像素重构损失。
        matching_logits (`torch.FloatTensor` of shape `(batch_size, 1)`):
            Matching objective logits. 匹配目标的逻辑回归输出。
        pixel_logits (`torch.FloatTensor` of shape
            `(batch_size, pixel_patch_length, image_patch_size ** 3 * pixel_num_channels)`): Pixel reconstruction
            logits. 像素重构的逻辑回归输出。
        audio_logits (`torch.FloatTensor` of shape
            `(batch_size, audio_patch_length, image_patch_size[0] * image_patch_size[1])`): Audio reconstruction
            logits. 音频重构的逻辑回归输出。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings and one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs. 模型每一层输出的隐藏状态，包括初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads. 经过注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    loss: Optional[torch.FloatTensor] = None
    # 定义一个变量 matching_logits，类型为 torch 的 FloatTensor，初始值为 None，用于存储匹配 logits
    matching_logits: torch.FloatTensor = None
    
    # 定义一个变量 pixel_logits，类型为 torch 的 FloatTensor，初始值为 None，用于存储像素 logits
    pixel_logits: torch.FloatTensor = None
    
    # 定义一个变量 audio_logits，类型为 torch 的 FloatTensor，初始值为 None，用于存储音频 logits
    audio_logits: torch.FloatTensor = None
    
    # 定义一个变量 hidden_states，类型为可选的元组，元素为 torch 的 FloatTensor，初始值为 None，用于存储隐藏状态
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    
    # 定义一个变量 attentions，类型为可选的元组，元素为 torch 的 FloatTensor，初始值为 None，用于存储注意力机制
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
# 生成用于像素值屏蔽的噪声，用于音频屏蔽。
def generate_pixel_mask_noise(pixel_values, pixel_mask=None, mask_ratio=0.75):
    """Generate noise for audio masking."""
    # 获取批次大小和序列长度
    batch_size, seq_len = pixel_values.shape[:2]
    # 生成在 [0, 1] 范围内的随机噪声
    noise = torch.rand((batch_size, seq_len), device=pixel_values.device)  # noise in [0, 1]
    # 计算需要保留的序列长度
    len_keep = int(seq_len * (1 - mask_ratio))
    return noise, len_keep


# 生成用于音频屏蔽的噪声。
def generate_audio_mask_noise(audio_values, audio_mask=None, mask_ratio=0.75, mask_type="patch-level", freq_len=8):
    """Generate noise for audio masking."""
    # 获取批次大小和序列长度
    batch_size, seq_len = audio_values.shape[:2]
    if mask_type == "frame-level":
        # 计算帧级别的时间片段数
        num_time_patches = seq_len // freq_len
        # 生成 [0, 1] 范围内的随机噪声并重复以匹配序列长度
        noise = (
            torch.rand(batch_size, num_time_patches, device=audio_values.device)
            .unsqueeze(-1)
            .repeat(1, 1, freq_len)
            .view(batch_size, seq_len)
        )  # noise in [0, 1]
    elif mask_type == "patch-level":
        # 生成 [0, 1] 范围内的随机噪声
        noise = torch.rand(batch_size, seq_len, device=audio_values.device)  # noise in [0, 1]
    # 计算需要保留的序列长度
    len_keep = int(seq_len * (1 - mask_ratio))
    return noise, len_keep


# 随机屏蔽，通过样本内帧级别的乱序进行随机屏蔽。通过 argsort 随机噪声进行样本内的乱序。
def random_masking(sequence, noise, len_keep, attention_masks=None):
    """
    Perform random masking by per-sample shuffling on frame-level. Per-sample shuffling is done by argsort random
    noise. sequence: [batch_size, seq_len, hidden_dim], sequence
    """
    batch_size, seq_len, hidden_dim = sequence.shape

    # 对每个样本的噪声进行排序
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    # 恢复原始顺序
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # 保留第一个子集
    ids_keep = ids_shuffle[:, :len_keep]
    # 使用乱序索引收集序列数据
    sequence_masked = torch.gather(sequence, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, hidden_dim))

    # 生成二进制屏蔽：0 表示保留，1 表示移除
    label_masks = torch.ones([batch_size, seq_len], device=sequence.device)
    label_masks[:, :len_keep] = 0
    # 使用 ids_restore 恢复原始顺序得到二进制屏蔽
    label_masks = torch.gather(label_masks, dim=1, index=ids_restore)

    if attention_masks is not None:
        # 若存在注意力屏蔽，则将其乘以二进制屏蔽
        label_masks *= attention_masks
        # 使用 ids_keep 乱序索引 attention_masks
        attention_masks = torch.gather(attention_masks, dim=1, index=ids_keep)

    return sequence_masked, attention_masks, label_masks, ids_restore


class TvltPixelEmbeddings(nn.Module):
    """Construct the patch and position embeddings."""

    def __init__(self, config):
        super().__init__()

        # 初始化像素块和位置嵌入
        self.patch_embeddings = TvltPixelPatchEmbeddings(config)
        self.num_patches_per_image = self.patch_embeddings.num_patches_per_image

        # 初始化类型嵌入向量、时间嵌入和位置嵌入向量
        self.type_embed_v = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.temporal_embed = nn.Parameter(torch.zeros(1, config.num_frames, config.hidden_size))
        self.pos_embed_v = nn.Parameter(torch.zeros(1, self.num_patches_per_image, config.hidden_size))

        self.config = config
    def forward(self, pixel_values, attention_masks=None):
        # 定义函数 forward，用于模型的前向传播计算
        # 获取输入张量的维度信息
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        
        # 通过 patch_embeddings 方法将像素值转换为补丁嵌入向量
        embeddings = self.patch_embeddings(pixel_values)
        
        # 加上位置嵌入向量，重复 num_frames 次以适应每一帧
        embeddings += self.pos_embed_v.repeat(1, num_frames, 1)
        
        # 使用 torch.repeat_interleave 方法，根据 num_patches_per_image 重复填充时间嵌入向量的部分，以适应每个补丁
        embeddings += torch.repeat_interleave(self.temporal_embed[:, :num_frames], self.num_patches_per_image, dim=1)
        
        # 加上类型嵌入向量，以适应输入数据的类型特征
        embeddings += self.type_embed_v
        
        # 返回嵌入向量和注意力掩码（可选）
        return embeddings, attention_masks
class TvltAudioEmbeddings(nn.Module):
    """Construct the patch and position embeddings."""

    def __init__(self, config):
        super().__init__()

        # 初始化音频补丁嵌入对象
        self.patch_embeddings = TvltAudioPatchEmbeddings(config)
        # 获取补丁数量
        self.num_patches = self.patch_embeddings.num_patches

        # 初始化音频类型嵌入
        self.type_embed_a = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        # 计算频率补丁数量
        self.num_freq_patches = config.frequency_length // config.audio_patch_size[1]
        # 初始化位置嵌入
        self.pos_embed_a = nn.Parameter(torch.zeros(1, self.num_patches // self.num_freq_patches, config.hidden_size))
        # 初始化频率嵌入
        self.freq_embed = nn.Parameter(torch.zeros(1, self.num_freq_patches, config.hidden_size))

        # 重新计算频率补丁数量
        self.num_freq_patches = config.frequency_length // config.audio_patch_size[1]
        # 保存配置信息
        self.config = config

    def forward(self, audio_values, attention_masks=None):
        # 创建补丁嵌入
        embeddings = self.patch_embeddings(audio_values)

        # 计算时间补丁数量
        num_time_patches = embeddings.size(1) // self.num_freq_patches
        # 添加频率嵌入到每个时间补丁
        embeddings += self.freq_embed.repeat(1, num_time_patches, 1)
        # 添加位置嵌入到每个时间补丁
        embeddings += torch.repeat_interleave(self.pos_embed_a[:, :num_time_patches], self.num_freq_patches, dim=1)
        # 添加类型嵌入
        embeddings += self.type_embed_a

        # 返回嵌入和注意力掩码（可选）
        return embeddings, attention_masks


class TvltPixelPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        
        # 初始化图像大小、补丁大小、通道数和隐藏层大小
        image_size, patch_size = config.image_size, config.image_patch_size
        num_channels, hidden_size = config.num_image_channels, config.hidden_size

        # 确保图像大小和补丁大小是迭代对象
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        # 计算每张图像的补丁数量
        num_patches_per_image = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])

        # 保存初始化参数
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches_per_image = num_patches_per_image
        self.hidden_size = hidden_size

        # 使用卷积层进行投影
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
    # 定义一个方法，用于对输入的像素值进行前向传播计算
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # 从输入的像素值张量中提取批量大小、帧数、通道数、高度和宽度
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        
        # 检查输入的像素值通道数是否与配置中指定的通道数一致，若不一致则抛出数值错误
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        
        # 检查输入图像的高度和宽度是否与模型配置中的设置一致，若不一致则抛出数值错误
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )

        # 将输入的像素值张量重塑为(batch_size * num_frames, num_channels, height, width)的形状
        pixel_values = pixel_values.reshape(batch_size * num_frames, num_channels, height, width)
        
        # 使用模型中的投影层对重塑后的像素值进行投影，并将结果展平并转置
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        
        # 将投影后的嵌入张量重新形状为(batch_size, num_frames * self.num_patches_per_image, self.hidden_size)
        embeddings = embeddings.reshape(batch_size, num_frames * self.num_patches_per_image, self.hidden_size)

        # 返回计算得到的嵌入张量作为前向传播的结果
        return embeddings
# 定义一个名为 `TvltAudioPatchEmbeddings` 的类，继承自 `nn.Module`，用于将形状为 `(batch_size, num_channels, height, width)` 的音频值转换为形状为 `(batch_size, seq_length, hidden_size)` 的初始隐藏状态（即补丁嵌入），以供 Transformer 模型使用。

    """
    This class turns `audio_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        # 从配置中获取频谱长度、频率长度和补丁大小
        spectrogram_length, frequency_length, patch_size = (
            config.spectrogram_length,
            config.frequency_length,
            config.audio_patch_size,
        )
        # 从配置中获取音频通道数和隐藏状态的大小
        num_channels, hidden_size = config.num_audio_channels, config.hidden_size

        # 定义频谱大小为元组 `(spectrogram_length, frequency_length)`
        spectrogram_size = (spectrogram_length, frequency_length)
        # 如果 `patch_size` 是可迭代对象，则保持不变；否则转换为元组 `(patch_size, patch_size)`
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        # 计算补丁数量，即 `(spectrogram_size[1] // patch_size[1]) * (spectrogram_size[0] // patch_size[0])`
        num_patches = (spectrogram_size[1] // patch_size[1]) * (spectrogram_size[0] // patch_size[0])
        # 定义补丁形状为 `(spectrogram_size[0] // patch_size[0], spectrogram_size[1] // patch_size[1])`
        patch_shape = (spectrogram_size[0] // patch_size[0], spectrogram_size[1] // patch_size[1])

        # 设置类的属性，包括频谱大小、补丁大小、音频通道数、补丁数量和补丁形状
        self.spectrogram_size = spectrogram_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.patch_shape = patch_shape

        # 使用 `nn.Conv2d` 定义投影层，将输入的音频通道转换为隐藏状态的卷积操作
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, audio_values: torch.Tensor) -> torch.Tensor:
        # 获取输入音频的形状信息 `(batch_size, num_channels, height, width)`
        batch_size, num_channels, height, width = audio_values.shape
        # 如果输入音频的通道数与设定的音频通道数不匹配，抛出数值错误
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 如果输入音频的高度大于设定的频谱高度或者宽度不等于设定的频率长度，抛出数值错误
        if height > self.spectrogram_size[0] or width != self.spectrogram_size[1]:
            raise ValueError(
                f"Input audio size ({height}*{width}) doesn't match model"
                f" ({self.spectrogram_size[0]}*{self.spectrogram_size[1]})."
            )
        # 将输入音频值投影到隐藏状态空间，并展平成形状 `(batch_size, hidden_size, seq_length)`
        embeddings = self.projection(audio_values).flatten(2).transpose(1, 2)

        # 返回嵌入后的结果
        return embeddings


# 从 `transformers.models.vilt.modeling_vilt.ViltSelfAttention` 复制到 `TvltSelfAttention`，仅修改类名
class TvltSelfAttention(nn.Module):
    # 初始化函数，接收一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        
        # 检查隐藏层大小是否能被注意力头数整除，同时检查是否有嵌入大小的属性
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            # 如果不满足条件，抛出数值错误异常
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )
        
        # 设置注意力头数和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 创建用于查询、键和值的线性层，并指定是否使用偏置
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        # 创建用于 dropout 的层，以减少注意力概率
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    # 将输入张量 x 转换为适合计算注意力分数的形状
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播函数，接收隐藏状态、注意力掩码、头掩码和是否输出注意力作为参数
    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        # 通过查询线性层生成混合查询层
        mixed_query_layer = self.query(hidden_states)

        # 使用键和值线性层生成适合计算注意力分数的键和值张量
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 计算查询与键的点积，得到原始的注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # 如果存在注意力掩码，将其应用到注意力分数上
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # 将注意力分数归一化为概率值
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # 使用 dropout 层减少注意力概率
        attention_probs = self.dropout(attention_probs)

        # 如果存在头掩码，将其应用到注意力概率上
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文张量，即注意力概率加权的值层
        context_layer = torch.matmul(attention_probs, value_layer)

        # 将上下文张量的维度重新排列为 [batch_size, seq_length, all_head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # 根据输出注意力的设置返回结果
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
# Copied from transformers.models.vilt.modeling_vilt.ViltSelfOutput with Vilt->Tvlt
class TvltSelfOutput(nn.Module):
    """
    The residual connection is defined in TvltLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: TvltConfig) -> None:
        super().__init__()
        # 定义一个全连接层，输入和输出大小都为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义一个 dropout 层，根据 config.hidden_dropout_prob 概率随机将输入设置为0
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用全连接层处理输入 hidden_states
        hidden_states = self.dense(hidden_states)
        # 对全连接层的输出进行 dropout 处理
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# Copied from transformers.models.vilt.modeling_vilt.ViltAttention with Vilt->Tvlt
class TvltAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化 TvltSelfAttention 和 TvltSelfOutput
        self.attention = TvltSelfAttention(config)
        self.output = TvltSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        # 如果 heads 列表为空，直接返回
        if len(heads) == 0:
            return
        # 调用 find_pruneable_heads_and_indices 函数获取需要修剪的头信息
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # 对 attention 和 output 中的相关层进行修剪
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储修剪后的头信息
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        # 调用 TvltSelfAttention 的 forward 方法处理 hidden_states
        self_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions)

        # 使用 TvltSelfOutput 处理 self_outputs 的第一个元素和 hidden_states
        attention_output = self.output(self_outputs[0], hidden_states)

        # 如果需要输出 attention，则将其加入 outputs
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.vilt.modeling_vilt.ViltIntermediate with Vilt->Tvlt
class TvltIntermediate(nn.Module):
    def __init__(self, config: TvltConfig) -> None:
        super().__init__()
        # 定义一个全连接层，输入大小为 config.hidden_size，输出大小为 config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 根据配置选择隐藏层激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
    # 定义一个前向传播方法，接收隐藏状态张量作为输入，并返回处理后的张量作为输出
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态张量传入全连接层，进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的结果应用中间激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)

        # 返回经过线性变换和激活函数处理后的隐藏状态张量
        return hidden_states
# Copied from transformers.models.vilt.modeling_vilt.ViltOutput with Vilt->Tvlt
class TvltOutput(nn.Module):
    def __init__(self, config: TvltConfig) -> None:
        super().__init__()
        # 定义一个全连接层，将输入维度为config.intermediate_size的向量映射到config.hidden_size的向量
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 定义一个dropout层，用于在训练过程中随机置零输入张量中的部分元素，以防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将输入hidden_states通过全连接层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对变换后的hidden_states进行dropout操作
        hidden_states = self.dropout(hidden_states)

        # 将dropout后的hidden_states与输入的input_tensor相加，实现残差连接
        hidden_states = hidden_states + input_tensor

        return hidden_states


# Copied from transformers.models.vilt.modeling_vilt.ViltLayer with Vilt->Tvlt
class TvltLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config):
        super().__init__()
        # 初始化TvltLayer类，设置一些需要的参数
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        # 初始化self.attention为TvltAttention类的实例
        self.attention = TvltAttention(config)
        # 初始化self.intermediate为TvltIntermediate类的实例
        self.intermediate = TvltIntermediate(config)
        # 初始化self.output为TvltOutput类的实例
        self.output = TvltOutput(config)
        # 初始化layernorm_before，使用nn.LayerNorm对输入向量进行归一化处理
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化layernorm_after，同样使用nn.LayerNorm对输入向量进行归一化处理
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        # 在ViLT中，先对输入hidden_states进行layernorm处理，再进行自注意力计算
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        # 取出self_attention计算后的输出
        attention_output = self_attention_outputs[0]
        # 如果需要输出注意力权重，则将注意力权重也包含在输出中
        outputs = self_attention_outputs[1:]

        # 第一个残差连接，将自注意力计算的输出与原始hidden_states相加
        hidden_states = attention_output + hidden_states.to(attention_output.device)

        # 在ViLT中，再次对输出进行layernorm处理
        layer_output = self.layernorm_after(hidden_states)
        # 将layernorm处理后的输出传递给intermediate层进行进一步的非线性变换
        layer_output = self.intermediate(layer_output)

        # 第二个残差连接，将intermediate层的输出与原始hidden_states相加
        layer_output = self.output(layer_output, hidden_states)

        # 将最终的layer_output作为输出结果，并将可能的注意力权重等其他信息也包含在outputs中返回
        outputs = (layer_output,) + outputs

        return outputs


# Copied from transformers.models.vilt.modeling_vilt.ViltEncoder with Vilt->Tvlt
class TvltEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 初始化TvltEncoder类，创建包含config.num_hidden_layers个TvltLayer层的ModuleList
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
        # 如果输出隐藏状态为真，则初始化一个空元组，否则为None
        all_hidden_states = () if output_hidden_states else None
        # 如果输出注意力分布为真，则初始化一个空元组，否则为None
        all_self_attentions = () if output_attentions else None

        # 遍历所有的Transformer层
        for i, layer_module in enumerate(self.layer):
            # 如果输出隐藏状态为真，将当前隐藏状态添加到all_hidden_states中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 如果开启了梯度检查点且处于训练状态
            if self.gradient_checkpointing and self.training:
                # 使用梯度检查点函数进行前向传播
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                # 普通的Transformer层前向传播
                layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, output_attentions)

            # 更新隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]

            # 如果输出注意力分布为真，将当前层的注意力分布添加到all_self_attentions中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果输出隐藏状态为真，将最终隐藏状态添加到all_hidden_states中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不使用返回字典形式，则返回非None的元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 否则，使用BaseModelOutput对象包装并返回
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
        Args:
            pixel_values (:obj:`torch.Tensor` of shape :obj:`(batch_size, channels, height, width)`):
                Pixel values. Pixel values are expected to be in the range [0, 1]. If the model expects a different
                range, you can rescale it accordingly before passing it to the model.
            Return: A dictionary containing the following entries:

                - **last_hidden_state** (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
                    Sequence of hidden-states at the output of the last layer of the model.
                - **pooler_output** (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, hidden_size)`):
                    Last layer hidden-state of the first token of the sequence (classification token) further processed
                    by a Linear layer and a Tanh activation function. The Linear layer weights are trained from the
                    last layer hidden-state and the bias is initialized as a zero vector.
    # 定义函数签名，描述输入参数的类型和形状
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`):
            像素数值。可以使用 [`TvltProcessor`] 获取像素数值。有关详细信息，请参见 [`TvltProcessor.__call__`]。

        audio_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            音频数值。可以使用 [`TvltProcessor`] 获取音频数值。有关详细信息，请参见 [`TvltProcessor.__call__`]。

        pixel_mask (`torch.FloatTensor` of shape `(batch_size, num_pixel_patches)`):
            像素掩码。可以使用 [`TvltProcessor`] 获取像素掩码。有关详细信息，请参见 [`TvltProcessor.__call__`]。

        audio_mask (`torch.FloatTensor` of shape `(batch_size, num_audio_patches)`):
            音频掩码。可以使用 [`TvltProcessor`] 获取音频掩码。有关详细信息，请参见 [`TvltProcessor.__call__`]。

        pixel_values_mixed (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`):
            Tvlt 视听匹配中混合了正负样本的像素数值。可以使用 [`TvltProcessor`] 获取混合像素数值。有关详细信息，请参见 [`TvltProcessor.__call__`]。

        pixel_mask_mixed (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            `pixel_values_mixed` 的像素掩码。可以使用 [`TvltProcessor`] 获取混合像素掩码。有关详细信息，请参见 [`TvltProcessor.__call__`]。

        mask_pixel (`bool`, *optional*):
            是否为 MAE 任务屏蔽像素。仅在 `TvltForPreTraining` 中设置为 True。

        mask_audio (`bool`, *optional*):
            是否为 MAE 任务屏蔽音频。仅在 `TvltForPreTraining` 中设置为 True。

        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。有关返回的张量中 `attentions` 的更多详细信息，请参阅。

        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。有关返回的张量中 `hidden_states` 的更多详细信息，请参阅。

        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""
定义 TvltModel 类，继承自 TvltPreTrainedModel 类，实现了 TVLT 模型的基础功能。

这是一个 Transformer 模型，用于处理 TVLT 相关任务，返回原始隐藏状态而不添加任何特定的输出头部。

@param config: 模型的配置对象，包含了模型的各种参数设置

初始化 TvltModel 类，设置模型的各个组件和参数。

self.pixel_embeddings = TvltPixelEmbeddings(config)
    # 初始化像素嵌入层，根据配置创建 TvltPixelEmbeddings 对象

self.audio_embeddings = TvltAudioEmbeddings(config)
    # 初始化音频嵌入层，根据配置创建 TvltAudioEmbeddings 对象

self.encoder = TvltEncoder(config)
    # 初始化编码器，根据配置创建 TvltEncoder 对象

self.cls_embedding = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
    # 创建一个可学习的参数，用于分类嵌入

if config.use_mean_pooling:
    self.layernorm = None
else:
    self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    # 如果配置要求使用均值池化，则不使用 LayerNorm；否则使用 LayerNorm 对隐藏状态进行标准化

调用 post_init 方法，用于初始化权重并进行最终处理

提供方法 get_input_embeddings，用于获取像素和音频嵌入的 patch 嵌入对象

_prune_heads 方法用于剪枝模型的注意力头部

@param heads_to_prune: 要剪枝的模型头部的字典，格式为 {layer_num: 在该层要剪枝的头部列表}，参见基类 PreTrainedModel

前向传播方法 forward，接受像素值和音频值作为输入，并可选地接受掩码和其他参数，返回 TvltModelOutput 对象

@param pixel_values: 像素值的张量输入
@param audio_values: 音频值的张量输入
@param pixel_mask: 可选的像素掩码张量
@param audio_mask: 可选的音频掩码张量
@param mask_pixel: 是否对像素值进行掩码处理
@param mask_audio: 是否对音频值进行掩码处理
@param output_attentions: 是否输出注意力权重
@param output_hidden_states: 是否输出隐藏状态
@param return_dict: 是否返回字典形式的输出结果

@return: TvltModelOutput 对象，包含前向传播的输出结果

定义 TvltDecoder 类，继承自 nn.Module 类，用于 TVLT 模型的解码器部分

@param config: 模型配置对象，包含解码器的各种参数设置

初始化 TvltDecoder 类，设置解码器的层列表和标准化层

self.decoder_layers = nn.ModuleList([TvltLayer(decoder_config) for _ in range(config.decoder_num_hidden_layers)])
    # 创建 TvltLayer 的模块列表，用于组成解码器的层

self.layernorm = nn.LayerNorm(config.decoder_hidden_size, eps=config.layer_norm_eps)
    # 创建解码器层的 LayerNorm 层，对隐藏状态进行标准化

设置梯度检查点为 False，并保存配置信息
"""
    ):
        # 如果输出隐藏状态设置为 True，则初始化空元组以保存所有隐藏状态，默认为 None
        all_hidden_states = () if output_hidden_states else None
        # 如果输出注意力权重设置为 True，则初始化空元组以保存所有自注意力权重，默认为 None
        all_self_attentions = () if output_attentions else None
        
        # 遍历 Transformer 解码器的每个层
        for i, layer_module in enumerate(self.decoder_layers):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            # 如果启用了梯度检查点且处于训练模式，则使用梯度检查点函数调用层模块
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    None,
                    output_attentions,
                )
            else:
                # 否则直接调用层模块，得到层的输出
                layer_outputs = layer_module(hidden_states, output_attentions=output_attentions)

            # 更新隐藏状态为当前层的输出的第一个元素（通常是下一层的输入）
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力权重，则将当前层的自注意力权重添加到 all_self_attentions 中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果需要输出隐藏状态，则将最终的隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 对最终的隐藏状态进行层归一化，得到最终的预测结果 logits
        logits = self.layernorm(hidden_states)

        # 如果不需要返回字典格式的结果，则按照顺序返回 logits、all_hidden_states、all_self_attentions 中不为 None 的部分
        if not return_dict:
            return tuple(v for v in [logits, all_hidden_states, all_self_attentions] if v is not None)
        
        # 否则，将结果封装成 TvltDecoderOutput 对象并返回
        return TvltDecoderOutput(logits=logits, hidden_states=all_hidden_states, attentions=all_self_attentions)
# 添加自动文档字符串以描述该类的作用和功能
@add_start_docstrings(
    "The TVLT Model transformer with the decoder on top for self-supervised pre-training.",
    TVLT_START_DOCSTRING,
)
# TvltForPreTraining 类继承自 TvltPreTrainedModel 类
class TvltForPreTraining(TvltPreTrainedModel):
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 将配置信息存储在实例中
        self.config = config

        # 从配置中获取任务匹配和任务 MAE 的标志位
        self.task_matching = config.task_matching
        self.task_mae = config.task_mae
        # 如果既没有设置任务匹配也没有设置任务 MAE，则抛出值错误异常
        if not (self.task_matching or self.task_mae):
            raise ValueError("Must set at least one of matching task and MAE task to true")

        # 创建 TVLT 模型实例
        self.tvlt = TvltModel(config)

        # 如果配置了任务匹配，则创建匹配头部实例
        if self.task_matching:
            self.matching_head = TvltMatchingHead(config)

        # 如果配置了任务 MAE，则进行以下初始化操作
        if self.task_mae:
            # 创建编码器到解码器的线性层
            self.encoder_to_decoder = nn.Linear(config.hidden_size, config.decoder_hidden_size, bias=True)

            # 创建像素级和音频级掩码标记参数
            self.pixel_mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_hidden_size))
            self.audio_mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_hidden_size))

            # 创建 TVLT 解码器实例
            self.decoder = TvltDecoder(config)

            # 从配置中获取解码器的隐藏层大小
            decoder_hidden_size = config.decoder_hidden_size

            # 从 TVLT 模型的像素嵌入中获取相关参数并创建相应的解码器位置嵌入
            num_frames = config.num_frames
            num_patches_per_image = self.tvlt.pixel_embeddings.num_patches_per_image
            self.decoder_pixel_pos_embed = nn.Parameter(torch.zeros(1, num_patches_per_image, decoder_hidden_size))
            self.decoder_temporal_embed = nn.Parameter(torch.zeros(1, config.num_frames, decoder_hidden_size))
            self.decoder_pixel_type_embed = nn.Parameter(torch.zeros(1, 1, decoder_hidden_size))

            # 从 TVLT 模型的音频嵌入中获取相关参数并创建相应的解码器位置嵌入
            num_audio_patches = self.tvlt.audio_embeddings.num_patches
            num_freq_patches = config.frequency_length // config.audio_patch_size[1]
            self.decoder_audio_pos_embed = nn.Parameter(
                torch.zeros(1, num_audio_patches // num_freq_patches, decoder_hidden_size)
            )
            self.decoder_freq_embed = nn.Parameter(torch.zeros(1, num_freq_patches, decoder_hidden_size))
            self.decoder_audio_type_embed = nn.Parameter(torch.zeros(1, 1, decoder_hidden_size))

            # 创建像素级和音频级 MAE 头部实例
            pixel_mae_output_dim = self.config.image_patch_size[0] ** 2 * self.config.num_image_channels
            self.pixel_mae_head = TvltMAEHead(config, pixel_mae_output_dim)
            audio_mae_output_dim = (
                self.config.audio_patch_size[0] * self.config.audio_patch_size[1] * self.config.num_audio_channels
            )
            self.audio_mae_head = TvltMAEHead(config, audio_mae_output_dim)

            # 存储一些与解码器相关的参数信息
            self.num_frames = num_frames
            self.num_patches_per_image = num_patches_per_image
            self.num_freq_patches = num_freq_patches
            self.image_patch_size = config.image_patch_size
            self.audio_patch_size = config.audio_patch_size

        # 执行后续的初始化步骤，包括权重初始化和最终处理
        self.post_init()
    # 将输入的像素值按照指定的图像块大小进行分块处理
    def patchify_pixel(self, pixel_values):
        """
        pixel_values: [batch_size, num_frames, 3, height, width]
        """
        # 获取输入像素值张量的维度信息
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        # 计算在高度和宽度上可以分成多少个图像块
        num_patches_height = pixel_values.shape[3] // self.image_patch_size[0]
        num_patches_width = pixel_values.shape[4] // self.image_patch_size[1]
        # 将像素值重新组织成指定形状的张量，以便后续处理
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
        # 使用 Einstein Summation Convention 进行张量乘积计算，重新排列张量维度
        patchified_pixel_values = torch.einsum("ntchpwq->nthwpqc", patchified_pixel_values)
        # 将重新排列的张量再次整形为指定形状，以便后续计算
        patchified_pixel_values = patchified_pixel_values.reshape(
            shape=(
                batch_size,
                num_patches_height * num_patches_width * num_frames,
                self.image_patch_size[0] * self.image_patch_size[1] * num_channels,
            )
        )
        return patchified_pixel_values

    # 将输入的音频值按照指定的音频块大小进行分块处理
    def patchify_audio(self, audio_values):
        """
        audio_values: [batch_size, 1, height, width]
        """
        # 获取输入音频值张量的维度信息
        batch_size, num_channels, height, width = audio_values.shape
        # 计算在高度和宽度上可以分成多少个音频块
        num_patches_height = height // self.audio_patch_size[0]
        num_patches_width = width // self.audio_patch_size[1]
        # 将音频值重新组织成指定形状的张量，以便后续处理
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
        # 使用 Einstein Summation Convention 进行张量乘积计算，重新排列张量维度
        patchified_audio_values = torch.einsum("nchpwq->nhwpqc", patchified_audio_values)
        # 将重新排列的张量再次整形为指定形状，以便后续计算
        patchified_audio_values = patchified_audio_values.reshape(
            shape=(
                batch_size,
                num_patches_height * num_patches_width,
                self.audio_patch_size[0] * self.audio_patch_size[1] * num_channels,
            )
        )
        return patchified_audio_values

    # 计算像素预测和实际像素之间的均方误差损失
    def pixel_mae_loss(self, pixel_values, pixel_predictions, mask):
        # 将输入的像素值进行分块处理
        patchified_pixel_values = self.patchify_pixel(pixel_values)
        # 计算预测像素值和分块像素值之间的平方差
        loss = (pixel_predictions - patchified_pixel_values) ** 2
        # 计算每个图像块上的平均损失
        loss = loss.mean(dim=-1)  # [batch_size, pixel_pixel_length], mean loss per patch
        # 根据掩码计算移除的图像块上的平均损失
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    # 计算音频预测和实际音频值之间的均方误差损失
    def audio_mae_loss(self, audio_values, audio_predictions, mask):
        # 将输入的音频值进行分块处理
        patchified_audio_values = self.patchify_audio(audio_values)
        # 计算预测音频值和分块音频值之间的平方差
        loss = (audio_predictions - patchified_audio_values) ** 2
        # 计算每个音频块上的平均损失
        loss = loss.mean(dim=-1)  # [batch_size, audio_pixel_length], mean loss per patch
        # 根据掩码计算移除的音频块上的平均损失
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    # 定义一个方法用于拼接掩码到序列的末尾
    def concatenate_mask(self, mask_token, sequence, ids_restore):
        # 获取序列的批大小、序列长度和维度
        batch_size, seq_length, dim = sequence.shape
        # 将掩码标记重复添加到每个样本序列末尾，以匹配恢复后的序列长度
        mask_tokens = mask_token.repeat(batch_size, ids_restore.shape[1] - seq_length, 1)
        # 在序列的末尾连接掩码标记
        padded_sequence = torch.cat([sequence, mask_tokens], dim=1)
        # 根据恢复的索引ids_restore重新排序序列，以恢复原始顺序
        padded_sequence = torch.gather(
            padded_sequence, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, dim)
        )  # unshuffle
        # 返回重新排序后的序列
        return padded_sequence

    # 定义模型的前向传播方法，此处注释通过装饰器已添加到模型前向方法的输入和输出文档字符串
    @add_start_docstrings_to_model_forward(TVLT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TvltForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        audio_values: torch.FloatTensor,
        pixel_mask: Optional[torch.FloatTensor] = None,
        audio_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values_mixed: Optional[torch.FloatTensor] = None,
        pixel_mask_mixed: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 定义一个自定义的 Transformer 模型，用于处理音频和视觉分类任务，例如 CMU-MOSEI 情感分析和音频到视频检索
@add_start_docstrings(
    """
    Tvlt Model transformer with a classifier head on top (an MLP on top of the final hidden state of the [CLS] token)
    for audiovisual classification tasks, e.g. CMU-MOSEI Sentiment Analysis and Audio to Video Retrieval.
    """,
    TVLT_START_DOCSTRING,
)
class TvltForAudioVisualClassification(TvltPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化 TvltModel，这是主要的 Transformer 模型
        self.tvlt = TvltModel(config)

        # 分类器头部网络
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),  # 线性层，扩展隐藏层大小
            nn.LayerNorm(config.hidden_size * 2, eps=config.layer_norm_eps),  # LayerNorm 层
            nn.GELU(),  # GELU 激活函数
            nn.Linear(config.hidden_size * 2, config.num_labels),  # 线性层，输出分类标签数
        )
        self.config = config

        # 初始化权重并进行最终处理
        self.post_init()

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
    ) -> Union[Tuple[torch.FloatTensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, num_labels)`, *optional*):
            Labels for computing the audiovisual loss. Indices should be in `[0, ..., num_classes-1]` where num_classes
            refers to the number of classes in audiovisual tasks.

        Return:

        Examples:
        ```python
        >>> from transformers import TvltProcessor, TvltForAudioVisualClassification
        >>> import numpy as np
        >>> import torch

        >>> num_frames = 8
        >>> images = list(np.random.randn(num_frames, 3, 224, 224))
        >>> audio = list(np.random.randn(10000))
        >>> processor = TvltProcessor.from_pretrained("ZinengTang/tvlt-base")
        >>> model = TvltForAudioVisualClassification.from_pretrained("ZinengTang/tvlt-base")
        >>> input_dict = processor(images, audio, sampling_rate=44100, return_tensors="pt")

        >>> outputs = model(**input_dict)
        >>> loss = outputs.loss
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # 若未指定 return_dict 则使用模型配置中的默认值

        outputs = self.tvlt(
            pixel_values,
            audio_values,
            pixel_mask=pixel_mask,
            audio_mask=audio_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )  # 调用 Tvlt 模型进行前向传播，获取输出

        sequence_output = outputs[0][:, 0]  # 获取序列输出的第一个位置的结果
        logits = self.classifier(sequence_output)  # 将序列输出传入分类器，得到分类 logits

        loss = None
        if labels is not None:
            if self.config.loss_type == "regression":  # 如果损失类型为回归
                loss_fct = MSELoss()
                loss = loss_fct(logits, labels)  # 计算均方误差损失
            elif self.config.loss_type == "classification":  # 如果损失类型为分类
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)  # 计算交叉熵损失

        if not return_dict:
            output = (logits,) + outputs[4:]  # 如果不返回字典，则组合输出结果
            return ((loss,) + output) if loss is not None else output  # 返回包含损失的输出

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )  # 返回包含损失、logits、隐藏状态和注意力的 SequenceClassifierOutput 对象
```