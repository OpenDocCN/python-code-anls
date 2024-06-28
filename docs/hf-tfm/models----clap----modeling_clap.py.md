# `.\models\clap\modeling_clap.py`

```
# 设置编码格式为 UTF-8
# 版权声明，指出了 LAION-AI 团队和 HuggingFace 团队对代码的所有权
# 根据 Apache 许可证版本 2.0 使用此文件，详细信息可以在指定网址获取
""" PyTorch CLAP model. """
# 导入必要的库和模块
import collections
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

# 导入来自其他路径的模块和函数
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 导入 CLAP 的配置文件
from .configuration_clap import ClapAudioConfig, ClapConfig, ClapTextConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 文档中的模型检查点
_CHECKPOINT_FOR_DOC = "laion/clap-htsat-fused"

# 预训练模型的存档列表
CLAP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "laion/clap-htsat-fused",
    "laion/clap-htsat-unfused",
    # 更多 CLAP 模型可以在指定链接中查看
]


# 从 https://github.com/LAION-AI/CLAP/blob/6ad05a971ba0622f6acee8c41993e0d02bbed639/src/open_clip/utils.py#L191 改编
def interpolate(hidden_states, ratio):
    """
    在时间域内插值数据。这用于补偿 CNN 下采样导致的分辨率降低。

    Args:
        hidden_states (`torch.FloatTensor` of shape (batch_size, time_length, classes_num)):
            输入的隐藏状态
        ratio (`int`):
            输出长度与输入长度的比率。
    """
    # 获取隐藏状态的维度信息
    (batch_size, time_length, classes_num) = hidden_states.shape
    # 将隐藏状态进行上采样
    upsampled = hidden_states[:, :, None, :].repeat(1, 1, ratio, 1)
    # 重新调整上采样后的形状
    upsampled = upsampled.reshape(batch_size, time_length * ratio, classes_num)
    return upsampled


# 从 https://github.com/LAION-AI/CLAP/blob/6ad05a971ba0622f6acee8c41993e0d02bbed639/src/open_clip/htsat.py#L249 改编
def window_partition(hidden_states, window_size):
    """
    返回调整大小后的隐藏状态。输出形状应为 `(batch_size * num_windows, window_size, window_size, num_channels)`

    Args:
        hidden_states (`torch.FloatTensor` of shape `(batch_size, height, width, num_channels)`):
            输入的隐藏状态
        window_size (`int`):
            窗口大小
    # 获取隐藏状态张量的形状信息，分别为批大小、高度、宽度、通道数
    batch_size, height, width, num_channels = hidden_states.shape
    
    # 将隐藏状态张量重塑为更小窗口大小的形状
    hidden_states = hidden_states.view(
        batch_size,                    # 新的批大小保持不变
        height // window_size,         # 将高度分割成窗口大小的部分
        window_size,                   # 窗口的高度
        width // window_size,          # 将宽度分割成窗口大小的部分
        window_size,                   # 窗口的宽度
        num_channels                   # 保持通道数不变
    )
    
    # 对重塑后的隐藏状态张量进行维度置换和连续化操作，以便形成窗口视图
    windows = hidden_states.permute(0, 1, 3, 2, 4, 5).contiguous().view(
        -1,                            # 自动计算批大小乘以新窗口数的总数
        window_size,                   # 窗口的高度
        window_size,                   # 窗口的宽度
        num_channels                   # 通道数保持不变
    )
    
    # 返回重塑后的窗口视图
    return windows
# Adapted from https://github.com/LAION-AI/CLAP/blob/6ad05a971ba0622f6acee8c41993e0d02bbed639/src/open_clip/htsat.py#L263
def window_reverse(windows, window_size, height, width):
    """
    Merges windows to produce higher resolution features.
    Args:
        windows (`torch.FloatTensor` of shape `(num_windows * batch_size, window_size, window_size, num_channels)`):
            Input windows
        window_size (`int`):
            Window size
        height (`int`):
            Height of the resized audio
        width (`int`):
            Width of the resized audio
    """
    # 获取输入窗口的最后一个维度，即通道数
    num_channels = windows.shape[-1]
    # 重新排列窗口，以生成更高分辨率的特征
    windows = windows.view(-1, height // window_size, width // window_size, window_size, window_size, num_channels)
    windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, height, width, num_channels)
    return windows


# Copied from transformers.models.roberta.modeling_roberta.create_position_ids_from_input_ids
def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        input_ids (`torch.Tensor`):
            Input tensor of token IDs
        padding_idx (`int`):
            Index of padding tokens in input_ids
        past_key_values_length (`int`, optional):
            Length of past key values, default is 0

    Returns:
        torch.Tensor:
            Tensor with position IDs corresponding to input_ids
    """
    # 创建一个 mask，标记非填充符号的位置
    mask = input_ids.ne(padding_idx).int()
    # 计算递增的位置索引，并考虑过去键值的长度
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx


# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html#CLIP-loss-function
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    Computes the contrastive loss using logits and labels.

    Args:
        logits (`torch.Tensor`):
            Logits from the model
    Returns:
        torch.Tensor:
            Computed contrastive loss
    """
    # 创建标签，长度与logits相同
    labels = torch.arange(len(logits), device=logits.device)
    return nn.functional.cross_entropy(logits, labels)


@dataclass
# Copied from transformers.models.clip.modeling_clip.CLIPTextModelOutput with CLIP->Clap
class ClapTextModelOutput(ModelOutput):
    """
    Output class for CLAP text model that includes a pooling of the last hidden states.
    Inherits from transformers.modeling_outputs.ModelOutput.
    """
    pass
    Args:
        text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The text embeddings obtained by applying the projection layer to the pooler_output.
            通过将投影层应用于池化输出得到的文本嵌入向量。
        
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
            模型最后一层输出的隐藏状态序列。

        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            模型每一层输出的隐藏状态元组，如果模型有嵌入层，则包括嵌入层输出，形状为`(batch_size, sequence_length, hidden_size)`。

        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            注意力权重元组，每层一个张量，形状为`(batch_size, num_heads, sequence_length, sequence_length)`。
@dataclass
# 定义一个数据类 ClapAudioModelOutput，用于存储 Clap 模型的输出结果，模仿原始实现的输出格式

class ClapAudioModelOutput(ModelOutput):
    """
    ClapAudio 模型的输出，模拟了原始实现的输出格式。

    Args:
        audio_embeds (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            应用投影层到汇聚输出得到的音频嵌入向量。
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的隐藏状态序列输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            元组，包含每一层的注意力权重 `torch.FloatTensor`，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            在注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            元组，包含模型的隐藏状态 `torch.FloatTensor` 输出，如果模型有嵌入层，还包含初始嵌入输出。
            形状为 `(batch_size, sequence_length, hidden_size)`。

            模型每一层的隐藏状态以及可选的初始嵌入层输出。
    """

    audio_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
# 从 transformers.models.clip.modeling_clip.CLIPOutput 复制而来，替换 CLIP 为 Clap，vision 为 audio，Vision 为 Audio，image 为 audio

class ClapOutput(ModelOutput):
    """
    Clap 模型的输出，模仿了原始实现的输出格式。
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for audio-text similarity.
        logits_per_audio:(`torch.FloatTensor` of shape `(audio_batch_size, text_batch_size)`):
            The scaled dot product scores between `audio_embeds` and `text_embeds`. This represents the audio-text
            similarity scores.
        logits_per_text:(`torch.FloatTensor` of shape `(text_batch_size, audio_batch_size)`):
            The scaled dot product scores between `text_embeds` and `audio_embeds`. This represents the text-audio
            similarity scores.
        text_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`ClapTextModel`].
        audio_embeds(`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The audio embeddings obtained by applying the projection layer to the pooled output of [`ClapAudioModel`].
        text_model_output(`BaseModelOutputWithPooling`):
            The output of the [`ClapTextModel`].
        audio_model_output(`BaseModelOutputWithPooling`):
            The output of the [`ClapAudioModel`].
    """
    # Optional: Holds the contrastive loss if computed
    loss: Optional[torch.FloatTensor] = None
    # Holds the similarity scores between audio and text embeddings
    logits_per_audio: torch.FloatTensor = None
    # Holds the similarity scores between text and audio embeddings
    logits_per_text: torch.FloatTensor = None
    # Holds the text embeddings after projection from `ClapTextModel`
    text_embeds: torch.FloatTensor = None
    # Holds the audio embeddings after projection from `ClapAudioModel`
    audio_embeds: torch.FloatTensor = None
    # Stores the output of `ClapTextModel` including pooling
    text_model_output: BaseModelOutputWithPooling = None
    # Stores the output of `ClapAudioModel` including pooling
    audio_model_output: BaseModelOutputWithPooling = None

    def to_tuple(self) -> Tuple[Any]:
        # Converts all attributes into a tuple, converting `text_model_output` and `audio_model_output` into tuples as well
        return tuple(
            self[k] if k not in ["text_model_output", "audio_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )
# Adapted from transformers.models.swin.modeling_swin.SwinDropPath
class ClapDropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks). This is a slightly
    refactored version of the `SwinDropPath` implementation.
    """

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob  # 初始化时设置 dropout 概率

    def forward(self, hidden_states):
        if self.drop_prob == 0.0 or not self.training:  # 如果 dropout 概率为0或者不在训练模式下，直接返回原始输入
            return hidden_states

        keep_prob = 1 - self.drop_prob  # 计算保留的概率
        # 根据输入 hidden_states 的维度，创建一个与之相同的随机张量
        shape = (hidden_states.shape[0],) + (1,) * (hidden_states.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=hidden_states.dtype, device=hidden_states.device)
        random_tensor.floor_()  # 将随机张量二值化
        output = hidden_states.div(keep_prob) * random_tensor  # 应用 dropout 操作
        return output


# Adapted from https://github.com/LAION-AI/CLAP/blob/6ad05a971ba0622f6acee8c41993e0d02bbed639/src/open_clip/feature_fusion.py#L133
class ClapAudioAFFBlock(nn.Module):
    r"""
    ATTENTIONAL FEATURE FUSION Block from CLAP, since in CLAP we are always in 2D mode, it is not needed to implement
    the 1D version.
    """

    def __init__(self, config: ClapAudioConfig):
        super().__init__()
        channels = config.patch_embeds_hidden_size
        downsize_ratio = config.aff_block_r
        inter_channels = int(channels // downsize_ratio)

        # 局部注意力机制模块
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),  # 1x1 卷积
            nn.BatchNorm2d(inter_channels),  # 批量归一化层
            nn.ReLU(inplace=True),  # ReLU 激活函数
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),  # 1x1 卷积
            nn.BatchNorm2d(channels),  # 批量归一化层
        )

        # 全局注意力机制模块
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 自适应平均池化层
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),  # 1x1 卷积
            nn.BatchNorm2d(inter_channels),  # 批量归一化层
            nn.ReLU(inplace=True),  # ReLU 激活函数
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),  # 1x1 卷积
            nn.BatchNorm2d(channels),  # 批量归一化层
        )

        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活函数

    def forward(self, hidden_states, residual):
        attention_input = hidden_states + residual  # 输入特征与残差连接

        # 融合层输出为局部注意力和全局注意力的加权和
        fused_layer_output = self.local_att(attention_input) + self.global_att(attention_input)
        fused_layer_output = self.sigmoid(fused_layer_output)  # 应用 Sigmoid 激活

        # 最终输出为经过加权后的输入特征与残差的线性组合
        output = 2 * hidden_states * fused_layer_output + 2 * residual * (1 - fused_layer_output)
        return output


class ClapAudioPatchEmbed(nn.Module):
    """
    This module converts the hidden states reshaped as an image to patch embeddings ready to be passed to the
    Transformer block.
    """
    # 初始化函数，接受一个 ClapAudioConfig 类型的配置对象作为参数
    def __init__(self, config: ClapAudioConfig):
        # 调用父类的初始化方法
        super().__init__()
        
        # 根据配置对象中的 spec_size 属性确定图像的尺寸，如果是整数则生成正方形尺寸，否则使用配置中的尺寸元组
        img_size = (config.spec_size, config.spec_size) if isinstance(config.spec_size, int) else config.spec_size
        
        # 根据配置对象中的 patch_size 属性确定 patch 的尺寸，如果是整数则生成正方形尺寸，否则使用配置中的尺寸元组
        patch_size = (
            (config.patch_size, config.patch_size) if isinstance(config.patch_size, int) else config.patch_size
        )
        
        # 根据配置对象中的 patch_stride 属性确定 patch 的步幅，如果是整数则生成相同步幅，否则使用配置中的步幅元组
        patch_stride = (
            (config.patch_stride, config.patch_stride) if isinstance(config.patch_stride, int) else config.patch_stride
        )

        # 将计算得到的图像尺寸和 patch 步幅存储到对象的 img_size 和 patch_stride 属性中
        self.img_size = img_size
        self.patch_stride = patch_stride

        # 计算图像网格的大小，即将图像尺寸按照 patch 步幅划分的块数
        self.grid_size = (img_size[0] // patch_stride[0], img_size[1] // patch_stride[1])
        
        # 计算总的 patch 数量，即图像网格的行数乘以列数
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # 根据配置对象中的 flatten_patch_embeds 属性确定是否展平 patch 的嵌入表示
        self.flatten = config.flatten_patch_embeds
        
        # 根据配置对象中的 enable_fusion 属性确定是否启用融合
        self.enable_fusion = config.enable_fusion

        # 根据 patch_size 和 patch_stride 计算用于卷积操作的 padding
        padding = ((patch_size[0] - patch_stride[0]) // 2, (patch_size[1] - patch_stride[1]) // 2)

        # 根据 enable_fusion 和 fusion_type 配置创建卷积核的尺寸缩放因子
        scale_factor = 4 if (self.enable_fusion) and (config.fusion_type == "channel_map") else 1

        # 创建一个卷积层，用于将 patch 的输入通道映射到隐藏表示空间，采用配置对象中的参数
        self.proj = nn.Conv2d(
            config.patch_embed_input_channels * scale_factor,  # 输入通道数为 patch 的输入通道数乘以尺寸缩放因子
            config.patch_embeds_hidden_size,  # 输出通道数为 patch 的嵌入表示的隐藏层大小
            kernel_size=patch_size,  # 卷积核尺寸为 patch_size
            stride=patch_stride,  # 步幅为 patch_stride
            padding=padding,  # 使用计算得到的 padding
        )

        # 根据配置对象中的 enable_patch_layer_norm 属性确定是否启用 patch 层的归一化
        self.norm = nn.LayerNorm(config.patch_embeds_hidden_size) if config.enable_patch_layer_norm else nn.Identity()
        
        # 如果启用融合，则创建融合模型和用于 mel 频谱的卷积层
        if self.enable_fusion:
            # 创建融合模型对象
            self.fusion_model = ClapAudioAFFBlock(config)
            
            # 创建用于 mel 频谱的卷积层，采用配置对象中的参数
            self.mel_conv2d = nn.Conv2d(
                config.patch_embed_input_channels,  # 输入通道数为 patch 的输入通道数
                config.patch_embeds_hidden_size,  # 输出通道数为 patch 的嵌入表示的隐藏层大小
                kernel_size=(patch_size[0], patch_size[1] * 3),  # 卷积核尺寸为 patch_size 的高，宽乘以3
                stride=(patch_stride[0], patch_stride[1] * 3),  # 步幅为 patch_stride 的高，宽乘以3
                padding=padding,  # 使用计算得到的 padding
            )
    # 前向传播函数，接受隐藏状态和可能的更长输入索引
    def forward(self, hidden_states, is_longer_idx=None):
        # 如果启用融合
        if self.enable_fusion:
            # 提取最后一个 mel，因为输入已经进行了转置
            global_hidden_states = hidden_states[:, 0:1, :, :]

            # 全局处理
            batch_size, num_channels, height, width = global_hidden_states.shape

            # 检查输入音频尺寸是否与模型期望的图像尺寸匹配
            if height != self.img_size[0] or width != self.img_size[1]:
                raise ValueError(
                    f"Input audio size ({height}*{width}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
                )

            # 对全局隐藏状态进行投影
            global_hidden_states = self.proj(global_hidden_states)
            output_width = global_hidden_states.size(-1)

            # 如果存在更长的输入索引
            if len(is_longer_idx) > 0:
                # 本地处理
                local_hidden_states = hidden_states[is_longer_idx, 1:, :, :].contiguous()
                batch_size, num_channels, height, width = local_hidden_states.shape
                # 重塑本地隐藏状态以便进行卷积操作
                local_hidden_states = local_hidden_states.view(batch_size * num_channels, 1, height, width)

                local_hidden_states = self.mel_conv2d(local_hidden_states)

                _, features, height, width = local_hidden_states.shape
                local_hidden_states = local_hidden_states.view(batch_size, num_channels, features, height, width)
                local_hidden_states = local_hidden_states.permute((0, 2, 3, 1, 4)).contiguous().flatten(3)

                local_width = local_hidden_states.size(-1)
                # 对本地隐藏状态进行填充，使其与全局隐藏状态的输出宽度一致
                local_hidden_states = torch.nn.functional.pad(
                    local_hidden_states, (0, output_width - local_width), "constant", 0
                )

                # 使用融合模型融合全局隐藏状态和本地隐藏状态
                global_hidden_states[is_longer_idx] = self.fusion_model(
                    global_hidden_states[is_longer_idx], local_hidden_states
                )
            # 更新隐藏状态为全局隐藏状态
            hidden_states = global_hidden_states
        else:
            # 如果未启用融合，直接进行投影
            _, _, height, width = hidden_states.shape
            # 检查输入音频尺寸是否与模型期望的图像尺寸匹配
            if height != self.img_size[0] or width != self.img_size[1]:
                raise ValueError(
                    f"Input audio size ({height}*{width}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
                )
            hidden_states = self.proj(hidden_states)

        # 如果设置了 flatten 标志，将隐藏状态展平并转置
        if self.flatten:
            hidden_states = hidden_states.flatten(2).transpose(1, 2)
        # 对隐藏状态进行归一化
        hidden_states = self.norm(hidden_states)
        # 返回最终处理后的隐藏状态
        return hidden_states
# 从 transformers.models.swin.modeling_swin.SwinSelfAttention 复制并改名为 ClapAudioSelfAttention 的类定义
class ClapAudioSelfAttention(nn.Module):
    # 初始化方法，接受 config、维度 dim、注意力头数 num_heads 和窗口大小 window_size 作为参数
    def __init__(self, config, dim, num_heads, window_size):
        super().__init__()
        # 如果维度 dim 不能被注意力头数 num_heads 整除，抛出数值错误异常
        if dim % num_heads != 0:
            raise ValueError(
                f"The hidden size ({dim}) is not a multiple of the number of attention heads ({num_heads})"
            )

        # 设置注意力头数和每个头的大小
        self.num_attention_heads = num_heads
        self.attention_head_size = int(dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.window_size = (
            window_size if isinstance(window_size, collections.abc.Iterable) else (window_size, window_size)
        )

        # 创建相对位置偏置表格参数，维度为 ((2 * window_size[0] - 1) * (2 * window_size[1] - 1)) x num_heads
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads)
        )

        # 获取窗口内每个 token 的相对位置索引对
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        # 将相对位置索引作为缓冲区注册到模块中
        self.register_buffer("relative_position_index", relative_position_index)

        # 定义查询、键、值的线性变换层，并考虑配置中的偏置
        self.query = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)

        # 定义用于注意力掩码的丢弃层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    # 将输入张量 x 转换为注意力分数矩阵的形状
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播方法，接受隐藏状态 hidden_states 和可选的注意力掩码 attention_mask 等作为输入参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        ) -> Tuple[torch.Tensor]:
        # 获取隐藏状态张量的批大小、维度和通道数
        batch_size, dim, num_channels = hidden_states.shape
        # 使用 self.query 对隐藏状态进行查询，生成混合的查询层
        mixed_query_layer = self.query(hidden_states)

        # 使用 self.key 对隐藏状态进行键的转换，并为注意力打分做准备
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        # 使用 self.value 对隐藏状态进行值的转换，并为注意力的加权求和做准备
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        # 对混合的查询层进行转置，以便与键层进行点积运算
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 计算原始的注意力分数，即查询层和键层的点积
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # 对注意力分数进行缩放，以减少梯度消失问题
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 获取相对位置偏置，根据预先计算的相对位置偏置表和索引
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        # 重新调整相对位置偏置的形状，以便与注意力分数相加
        relative_position_bias = relative_position_bias.view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        # 对相对位置偏置进行维度置换和连续性处理
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        # 将相对位置偏置添加到注意力分数中
        attention_scores = attention_scores + relative_position_bias.unsqueeze(0)

        # 如果存在注意力掩码，则应用该掩码
        if attention_mask is not None:
            # 调整注意力分数的形状以便与注意力掩码相加
            mask_shape = attention_mask.shape[0]
            attention_scores = attention_scores.view(
                batch_size // mask_shape, mask_shape, self.num_attention_heads, dim, dim
            )
            attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(0)
            attention_scores = attention_scores.view(-1, self.num_attention_heads, dim, dim)

        # 将注意力分数归一化为概率分布
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 对注意力概率进行 dropout 处理
        attention_probs = self.dropout(attention_probs)

        # 如果存在头部掩码，则将注意力概率乘以头部掩码
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算加权求和后的值层，得到上下文层
        context_layer = torch.matmul(attention_probs, value_layer)
        # 对上下文层进行维度置换和连续性处理，以便后续计算
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # 调整上下文层的形状，以便匹配预期的输出形状
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # 根据需要选择输出内容，包括注意力概率
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
# Copied from transformers.models.swin.modeling_swin.SwinSelfOutput with Swin->ClapAudio
class ClapAudioSelfOutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        # 初始化一个全连接层，输入输出维度都为 dim
        self.dense = nn.Linear(dim, dim)
        # 初始化一个 dropout 层，使用给定的 dropout 概率
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 全连接层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 使用 dropout 处理全连接层输出
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# Copied from transformers.models.swin.modeling_swin.SwinAttention with Swin->ClapAudio
class ClapAudioAttention(nn.Module):
    def __init__(self, config, dim, num_heads, window_size):
        super().__init__()
        # 初始化 self attention 层
        self.self = ClapAudioSelfAttention(config, dim, num_heads, window_size)
        # 初始化输出层
        self.output = ClapAudioSelfOutput(config, dim)
        # 初始化一个集合，用于存储需要剪枝的注意力头
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 调用剪枝函数，获取需要剪枝的注意力头和相应的索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 对 self attention 层的查询、键、值线性层进行剪枝
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        # 对输出层的全连接层进行剪枝
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储剪枝过的头
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
        # 调用 self attention 层的 forward 方法
        self_outputs = self.self(hidden_states, attention_mask, head_mask, output_attentions)
        # 通过输出层处理 self attention 层的输出和输入的隐藏状态
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出注意力权重，则将其添加到输出中
        outputs = (attention_output,) + self_outputs[1:]  # 如果需要输出注意力权重，则将其添加到输出中
        return outputs


# Copied from transformers.models.swin.modeling_swin.SwinIntermediate with Swin->ClapAudio
class ClapAudioIntermediate(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        # 初始化一个全连接层，输入维度为 dim，输出维度为 config.mlp_ratio * dim
        self.dense = nn.Linear(dim, int(config.mlp_ratio * dim))
        # 如果隐藏激活函数是字符串，则使用对应的函数映射；否则直接使用配置中的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用全连接层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 使用中间激活函数处理全连接层输出
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
# 从transformers.models.swin.modeling_swin.SwinOutput复制过来，将Swin替换为ClapAudio
class ClapAudioOutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        # 创建一个线性层，输入维度为config.mlp_ratio乘以dim的整数部分，输出维度为dim
        self.dense = nn.Linear(int(config.mlp_ratio * dim), dim)
        # 创建一个dropout层，使用config.hidden_dropout_prob作为丢弃概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入的hidden_states传递给线性层
        hidden_states = self.dense(hidden_states)
        # 对线性层的输出进行dropout处理
        hidden_states = self.dropout(hidden_states)
        # 返回处理后的hidden_states作为输出
        return hidden_states


# 从transformers.models.swin.modeling_swin.SwinLayer复制过来，将Swin替换为ClapAudio，将SwinDropPath替换为ClapDropPath
class ClapAudioLayer(nn.Module):
    def __init__(self, config, dim, input_resolution, num_heads, shift_size=0):
        super().__init__()
        # 设定前馈分块大小为config.chunk_size_feed_forward
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 设定位移大小为shift_size
        self.shift_size = shift_size
        # 设定窗口大小为config.window_size
        self.window_size = config.window_size
        # 设定输入分辨率为input_resolution
        self.input_resolution = input_resolution
        # 在LayerNorm层之前添加LayerNorm，输入维度为dim，epsilon值为config.layer_norm_eps
        self.layernorm_before = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        # 创建ClapAudioAttention对象，使用config、dim、num_heads和window_size作为参数
        self.attention = ClapAudioAttention(config, dim, num_heads, window_size=self.window_size)
        # 如果config.drop_path_rate大于0.0，则创建ClapDropPath对象，否则创建一个恒等映射层nn.Identity()
        self.drop_path = ClapDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        # 在LayerNorm层之后添加LayerNorm，输入维度为dim，epsilon值为config.layer_norm_eps
        self.layernorm_after = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        # 创建ClapAudioIntermediate对象，使用config和dim作为参数
        self.intermediate = ClapAudioIntermediate(config, dim)
        # 创建ClapAudioOutput对象，使用config和dim作为参数
        self.output = ClapAudioOutput(config, dim)

    def set_shift_and_window_size(self, input_resolution):
        if min(input_resolution) <= self.window_size:
            # 如果输入分辨率中的最小值小于等于窗口大小，则不对窗口进行分区
            self.shift_size = 0
            self.window_size = min(input_resolution)

    def get_attn_mask(self, height, width, dtype):
        if self.shift_size > 0:
            # 计算SW-MSA的注意力掩码
            img_mask = torch.zeros((1, height, width, 1), dtype=dtype)
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
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        # 返回注意力掩码
        return attn_mask
    # 对输入的隐藏状态进行可能的填充，使其能够被窗口大小整除
    def maybe_pad(self, hidden_states, height, width):
        # 计算右侧需要填充的数量，确保宽度能够被窗口大小整除
        pad_right = (self.window_size - width % self.window_size) % self.window_size
        # 计算底部需要填充的数量，确保高度能够被窗口大小整除
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        # 组成填充的数值，(top, bottom, left, right, 0, 0)，这里只填充右侧和底部
        pad_values = (0, 0, 0, pad_right, 0, pad_bottom)
        # 使用PyTorch的函数对隐藏状态进行填充操作
        hidden_states = nn.functional.pad(hidden_states, pad_values)
        # 返回填充后的隐藏状态和填充数值
        return hidden_states, pad_values

    # 前向传播函数，接受隐藏状态、输入维度、头部遮罩、输出注意力权重、始终分区等参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        always_partition: Optional[bool] = False,
        ) -> Tuple[torch.Tensor, torch.Tensor]:  
        # 函数声明，接受一个输入参数并返回一个元组，包含两个 torch.Tensor 类型的对象

        if not always_partition:  
            # 如果参数 always_partition 不为真，则执行以下操作
            self.set_shift_and_window_size(input_dimensions)  
            # 调用对象的方法设置位移和窗口大小
        else:  
            # 否则，如果 always_partition 为真，则执行以下操作
            pass  
            # 不执行任何操作，直接跳过

        height, width = input_dimensions  
        # 解包输入维度元组，获取高度和宽度

        batch_size, _, channels = hidden_states.size()  
        # 获取隐藏状态张量的批量大小、通道数等信息
        shortcut = hidden_states  
        # 将隐藏状态张量赋值给快捷变量 shortcut

        hidden_states = self.layernorm_before(hidden_states)  
        # 对隐藏状态张量应用前层归一化

        hidden_states = hidden_states.view(batch_size, height, width, channels)  
        # 将隐藏状态张量重新形状为四维张量，形状为（批量大小，高度，宽度，通道数）

        # pad hidden_states to multiples of window size
        hidden_states, pad_values = self.maybe_pad(hidden_states, height, width)  
        # 可能对隐藏状态张量进行填充，使其大小为窗口大小的倍数，并返回填充后的张量和填充值

        _, height_pad, width_pad, _ = hidden_states.shape  
        # 解包填充后的张量形状，获取填充后的高度和宽度

        # cyclic shift
        if self.shift_size > 0:  
            # 如果位移大小大于零，则执行以下操作
            shifted_hidden_states = torch.roll(hidden_states, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))  
            # 在指定维度上将隐藏状态张量进行循环位移
        else:  
            # 否则，如果位移大小不大于零，则执行以下操作
            shifted_hidden_states = hidden_states  
            # 将隐藏状态张量赋值给位移后的隐藏状态张量

        # partition windows
        hidden_states_windows = window_partition(shifted_hidden_states, self.window_size)  
        # 划分窗口，将位移后的隐藏状态张量划分为窗口
        hidden_states_windows = hidden_states_windows.view(-1, self.window_size * self.window_size, channels)  
        # 将划分后的窗口重新形状为三维张量

        attn_mask = self.get_attn_mask(height_pad, width_pad, dtype=hidden_states.dtype)  
        # 获取注意力掩码，用于注意力计算
        if attn_mask is not None:  
            # 如果注意力掩码不为空，则执行以下操作
            attn_mask = attn_mask.to(hidden_states_windows.device)  
            # 将注意力掩码移到与隐藏状态窗口相同的设备上

        attention_outputs = self.attention(
            hidden_states_windows, attn_mask, head_mask, output_attentions=output_attentions
        )  
        # 使用注意力机制计算输出，包括注意力权重和其它相关输出

        attention_output = attention_outputs[0]  
        # 获取注意力输出的第一个元素

        attention_windows = attention_output.view(-1, self.window_size, self.window_size, channels)  
        # 将注意力输出重新形状为四维张量，表示窗口形式的注意力输出

        shifted_windows = window_reverse(attention_windows, self.window_size, height_pad, width_pad)  
        # 将注意力窗口反转，逆操作

        # reverse cyclic shift
        if self.shift_size > 0:  
            # 如果位移大小大于零，则执行以下操作
            attention_windows = torch.roll(shifted_windows, shifts=(self.shift_size, self.shift_size), dims=(1, 2))  
            # 在指定维度上对注意力窗口进行反向循环位移
        else:  
            # 否则，如果位移大小不大于零，则执行以下操作
            attention_windows = shifted_windows  
            # 将反转后的注意力窗口赋值给注意力窗口

        was_padded = pad_values[3] > 0 or pad_values[5] > 0  
        # 检查是否进行了填充

        if was_padded:  
            # 如果进行了填充，则执行以下操作
            attention_windows = attention_windows[:, :height, :width, :].contiguous()  
            # 对注意力窗口进行切片，保留非填充部分，并确保内存连续性

        attention_windows = attention_windows.view(batch_size, height * width, channels)  
        # 将注意力窗口重新形状为三维张量

        hidden_states = shortcut + self.drop_path(attention_windows)  
        # 将快捷路径与注意力窗口加上 dropout 后的结果相加，作为隐藏状态的新值

        layer_output = self.layernorm_after(hidden_states)  
        # 对隐藏状态应用后层归一化
        layer_output = self.intermediate(layer_output)  
        # 对层输出应用中间层处理
        layer_output = hidden_states + self.output(layer_output)  
        # 将层输出与输出层处理后的结果相加，作为最终的层输出

        layer_outputs = (layer_output, attention_outputs[1]) if output_attentions else (layer_output,)  
        # 如果需要输出注意力，将注意力权重也作为输出之一
        return layer_outputs  
        # 返回层的输出元组
# 从transformers.models.swin.modeling_swin.SwinStage复制而来，将Swin替换为ClapAudio
class ClapAudioStage(nn.Module):
    def __init__(self, config, dim, input_resolution, depth, num_heads, drop_path, downsample):
        super().__init__()
        self.config = config
        self.dim = dim
        # 创建包含多个ClapAudioLayer的模块列表
        self.blocks = nn.ModuleList(
            [
                ClapAudioLayer(
                    config=config,
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    shift_size=0 if (i % 2 == 0) else config.window_size // 2,
                )
                for i in range(depth)
            ]
        )

        # 如果downsample不为None，则使用给定的输入分辨率和维度创建下采样层
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=nn.LayerNorm)
        else:
            self.downsample = None

        # 是否进行指向性操作的标志
        self.pointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        always_partition: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        height, width = input_dimensions
        # 对每个ClapAudioLayer进行前向传播
        for i, layer_module in enumerate(self.blocks):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states, input_dimensions, layer_head_mask, output_attentions, always_partition
            )

            hidden_states = layer_outputs[0]

        hidden_states_before_downsampling = hidden_states
        # 如果存在下采样层，则进行下采样操作
        if self.downsample is not None:
            height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2
            output_dimensions = (height, width, height_downsampled, width_downsampled)
            hidden_states = self.downsample(hidden_states_before_downsampling, input_dimensions)
        else:
            output_dimensions = (height, width, height, width)

        stage_outputs = (hidden_states, hidden_states_before_downsampling, output_dimensions)

        # 如果需要输出注意力权重，则将它们包含在输出中
        if output_attentions:
            stage_outputs += layer_outputs[1:]
        return stage_outputs


# 从transformers.models.swin.modeling_swin.SwinPatchMerging复制而来，将Swin替换为ClapAudio
class ClapAudioPatchMerging(nn.Module):
    """
    Patch Merging Layer.

    Args:
        input_resolution (`Tuple[int]`):
            输入特征的分辨率。
        dim (`int`):
            输入通道数。
        norm_layer (`nn.Module`, *optional*, defaults to `nn.LayerNorm`):
            标准化层的类。
    """
    def __init__(self, input_resolution: Tuple[int], dim: int, norm_layer: nn.Module = nn.LayerNorm) -> None:
        super().__init__()
        self.input_resolution = input_resolution  # 初始化输入分辨率
        self.dim = dim  # 初始化维度
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)  # 创建线性变换层，从4*dim到2*dim
        self.norm = norm_layer(4 * dim)  # 初始化归一化层，输入为4*dim

    def maybe_pad(self, input_feature, height, width):
        should_pad = (height % 2 == 1) or (width % 2 == 1)  # 判断是否需要填充，如果高或宽为奇数则需要
        if should_pad:
            pad_values = (0, 0, 0, width % 2, 0, height % 2)  # 计算填充值，使得宽和高都为偶数
            input_feature = nn.functional.pad(input_feature, pad_values)  # 对输入特征进行填充

        return input_feature

    def forward(self, input_feature: torch.Tensor, input_dimensions: Tuple[int, int]) -> torch.Tensor:
        height, width = input_dimensions  # 解包输入维度
        # `dim` is height * width
        batch_size, dim, num_channels = input_feature.shape  # 获取输入特征的形状信息

        input_feature = input_feature.view(batch_size, height, width, num_channels)  # 将输入特征重塑为四维张量
        # pad input to be disible by width and height, if needed
        input_feature = self.maybe_pad(input_feature, height, width)  # 如果需要，对输入特征进行填充
        # [batch_size, height/2, width/2, num_channels]
        input_feature_0 = input_feature[:, 0::2, 0::2, :]  # 提取特征的子采样部分，步长为2
        # [batch_size, height/2, width/2, num_channels]
        input_feature_1 = input_feature[:, 1::2, 0::2, :]  # 提取特征的子采样部分，步长为2
        # [batch_size, height/2, width/2, num_channels]
        input_feature_2 = input_feature[:, 0::2, 1::2, :]  # 提取特征的子采样部分，步长为2
        # [batch_size, height/2, width/2, num_channels]
        input_feature_3 = input_feature[:, 1::2, 1::2, :]  # 提取特征的子采样部分，步长为2
        # batch_size height/2 width/2 4*num_channels
        input_feature = torch.cat([input_feature_0, input_feature_1, input_feature_2, input_feature_3], -1)  # 将四个子采样特征拼接在一起
        input_feature = input_feature.view(batch_size, -1, 4 * num_channels)  # 将特征重新展平，变成三维张量

        input_feature = self.norm(input_feature)  # 对特征进行归一化
        input_feature = self.reduction(input_feature)  # 对特征进行线性变换

        return input_feature
# 定义 ClapAudioEncoder 类，继承自 nn.Module，用于音频编码器的定义和处理
class ClapAudioEncoder(nn.Module):
    # 初始化方法，接受一个配置对象 config
    def __init__(self, config):
        super().__init__()
        # 计算层数并保存到 self.num_layers 中
        self.num_layers = len(config.depths)

        # 保存配置对象到 self.config 中
        self.config = config
        # 创建 ClapAudioPatchEmbed 对象并保存到 self.patch_embed 中
        self.patch_embed = ClapAudioPatchEmbed(config)
        # 从配置中获取是否启用融合，并保存到 self.enable_fusion 中
        self.enable_fusion = config.enable_fusion
        # 从 patch_embed 中获取 patch 的步幅并保存到 self.patch_stride 中
        self.patch_stride = self.patch_embed.patch_stride
        # 从配置中获取 spec_size 并保存到 self.spec_size 中
        self.spec_size = config.spec_size
        # 计算频率比率并保存到 self.freq_ratio 中
        self.freq_ratio = config.spec_size // config.num_mel_bins

        # 计算特征数量并保存到 self.num_features 中
        self.num_features = int(config.patch_embeds_hidden_size * 2 ** (self.num_layers - 1))

        # 根据 drop_path_rate 创建一个列表，用于后续的层级设置
        drop_path_rate = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]

        # 计算 patch embed 的网格大小并保存到 grid_size 中
        grid_size = self.patch_embed.grid_size
        # 根据层数创建输入分辨率列表，并保存到 self.input_resolutions 中
        self.input_resolutions = [(grid_size[0] // (2**i), grid_size[1] // (2**i)) for i in range(self.num_layers)]

        # 创建一个 nn.ModuleList，包含多个 ClapAudioStage 层，并保存到 self.layers 中
        self.layers = nn.ModuleList(
            [
                ClapAudioStage(
                    config=config,
                    dim=int(config.patch_embeds_hidden_size * 2**i_layer),
                    input_resolution=self.input_resolutions[i_layer],
                    depth=config.depths[i_layer],
                    num_heads=config.num_attention_heads[i_layer],
                    drop_path=drop_path_rate[sum(config.depths[:i_layer]) : sum(config.depths[: i_layer + 1])],
                    downsample=ClapAudioPatchMerging if (i_layer < self.num_layers - 1) else None,
                )
                for i_layer in range(self.num_layers)
            ]
        )

        # 设置梯度检查点为 False
        self.gradient_checkpointing = False

        # 创建一个 nn.BatchNorm2d 层，用于批量归一化，并保存到 self.batch_norm 中
        self.batch_norm = nn.BatchNorm2d(config.num_mel_bins)
        # 创建一个 nn.LayerNorm 层，用于层归一化，并保存到 self.norm 中
        self.norm = nn.LayerNorm(self.num_features)
        # 从配置中获取 depths 并保存到 self.depths 中
        self.depths = config.depths
        # 创建一个 nn.AdaptiveAvgPool1d 层，用于自适应平均池化，并保存到 self.avgpool 中
        self.avgpool = nn.AdaptiveAvgPool1d(1)
    def reshape_mel2img(self, normalized_input_features):
        """
        The input is 4 normalized log mel spectrograms. It is reshape to the common shape of images. Each channel
        should represent 1 of the 4 crops of the spectrogram. For more details, refer to the [`ClapFeatureExtractor`].
        """
        # 获取输入特征的形状信息：batch_size, channels, time_length, freq_length
        _, _, time_length, freq_length = normalized_input_features.shape

        # 计算目标图像的宽度和高度
        spec_width = int(self.spec_size * self.freq_ratio)
        spec_heigth = self.spec_size // self.freq_ratio

        # 检查输入的时间长度和频率长度是否超过了目标图像的大小
        if time_length > spec_width or freq_length > spec_heigth:
            raise ValueError("the wav size should be less than or equal to the swin input size")

        # 为了避免双三次插值时的零值错误，对输入进行插值处理
        if time_length < spec_width:
            normalized_input_features = nn.functional.interpolate(
                normalized_input_features, (spec_width, freq_length), mode="bicubic", align_corners=True
            )
        if freq_length < spec_heigth:
            normalized_input_features = nn.functional.interpolate(
                normalized_input_features, (time_length, spec_heigth), mode="bicubic", align_corners=True
            )

        # 获取调整后的输入特征的新形状信息
        batch, channels, time, freq = normalized_input_features.shape

        # 将输入特征重塑为目标形状
        # batch_size, channels, spec_width, spec_heigth --> batch_size, channels * freq_ratio, spec_heigth, spec_width // freq_ratio
        normalized_input_features = normalized_input_features.reshape(
            batch, channels * self.freq_ratio, time // self.freq_ratio, freq
        )
        # 转置特征以匹配期望的维度顺序
        normalized_input_features = normalized_input_features.permute(0, 1, 3, 2).contiguous()
        # 再次重塑特征以最终形状返回
        normalized_input_features = normalized_input_features.reshape(
            batch, channels, freq * self.freq_ratio, time // self.freq_ratio
        )

        return normalized_input_features
# CLAP_START_DOCSTRING 变量，包含模型继承自 `PreTrainedModel` 的描述，建议查看超类文档以获取关于模型的通用方法，如下载、保存、调整输入嵌入大小、修剪头等。
CLAP_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`ClapConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# CLAP_TEXT_INPUTS_DOCSTRING 变量，包含描述模型文本输入参数的文档字符串，包括 input_ids、attention_mask、position_ids 等。
CLAP_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# CLAP_AUDIO_INPUTS_DOCSTRING 变量，暂时为空字符串，预留给后续可能会添加的音频输入相关的文档字符串。
CLAP_AUDIO_INPUTS_DOCSTRING = r"""
    """
    Args:
        input_features (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Input audio features. This should be returned by the [`ClapFeatureExtractor`] class that you can also
            retrieve from [`AutoFeatureExtractor`]. See [`ClapFeatureExtractor.__call__`] for details.
        is_longer (`torch.FloatTensor`, of shape `(batch_size, 1)`, *optional*):
            Whether the audio clip is longer than `max_length`. If `True`, a feature fusion will be enabled to enhance
            the features.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attention tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

CLAP_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        input_features (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Input audio features. This should be returnes by the [`ClapFeatureExtractor`] class that you can also
            retrieve from [`AutoFeatureExtractor`]. See [`ClapFeatureExtractor.__call__`] for details.
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class ClapProjectionLayer(nn.Module):
    """
    Projection layer for CLAP model.

    Args:
        config (Union[ClapAudioConfig, ClapTextConfig]): Configuration object for CLAP model.

    Attributes:
        linear1 (nn.Linear): First linear transformation layer.
        activation: Activation function applied after the first linear transformation.
        linear2 (nn.Linear): Second linear transformation layer.
    """

    def __init__(self, config: Union[ClapAudioConfig, ClapTextConfig]):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        projection_dim = config.projection_dim

        # Initialize linear layers
        self.linear1 = nn.Linear(hidden_size, projection_dim)
        self.activation = ACT2FN[config.projection_hidden_act]
        self.linear2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, hidden_states):
        """
        Perform forward pass of the projection layer.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape `(batch_size, hidden_size)`.

        Returns:
            torch.Tensor: Output tensor of shape `(batch_size, projection_dim)`.
        """
        hidden_states = self.linear1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.linear2(hidden_states)
        return hidden_states


# Copied from transformers.models.roberta.modeling_roberta.RobertaEmbeddings with Roberta->ClapText, persistent=False->persistent=True
class ClapTextEmbeddings(nn.Module):
    """
    CLAP model text embeddings.

    Inherits from nn.Module and handles the embeddings for text input.

    """
    # Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    
    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        super().__init__()
        # 初始化词嵌入层，使用 nn.Embedding 类，配置词汇大小、隐藏大小，并设置填充索引为 config.pad_token_id
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 初始化位置嵌入层，使用 nn.Embedding 类，配置最大位置嵌入大小和隐藏大小
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 初始化类型嵌入层，使用 nn.Embedding 类，配置类型词汇表大小和隐藏大小
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
    
        # self.LayerNorm 没有使用蛇形命名以保持与 TensorFlow 模型变量名的一致性，以便能够加载任何 TensorFlow 检查点文件
        # 初始化 LayerNorm 层，使用 nn.LayerNorm 类，配置隐藏大小和 eps 参数为 config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化 Dropout 层，使用 nn.Dropout 类，配置丢弃率为 config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids（1，长度位置 emb）在内存中是连续的，并在序列化时导出
        # 设置位置嵌入类型，默认为 "absolute"，或从 config 中获取 position_embedding_type 属性
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 注册缓冲区，创建位置 ID 张量，大小为 (1, config.max_position_embeddings)，持久化为 True
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=True
        )
        # 注册缓冲区，创建类型 ID 张量，大小与位置 ID 张量相同，类型为 long 型，持久化为 True
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=True
        )
    
        # End copy
        # 设置填充索引为 config.pad_token_id
        self.padding_idx = config.pad_token_id
        # 重新初始化位置嵌入层，使用 nn.Embedding 类，配置最大位置嵌入大小、隐藏大小，并设置填充索引为 self.padding_idx
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )
        ):
            # 如果未提供位置标识符，则根据输入的标记标识符创建位置标识符。任何填充的标记保持填充状态。
            if position_ids is None:
                if input_ids is not None:
                    position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
                else:
                    position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

            # 如果提供了输入标记标识符，则确定其形状
            if input_ids is not None:
                input_shape = input_ids.size()
            else:
                input_shape = inputs_embeds.size()[:-1]

            # 获取序列的长度
            seq_length = input_shape[1]

            # 将token_type_ids设置为构造函数中注册的缓冲区，通常是全零，这在自动生成时很有用，注册的缓冲区有助于用户在不传递token_type_ids的情况下跟踪模型，解决问题＃5664
            if token_type_ids is None:
                if hasattr(self, "token_type_ids"):
                    buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                    buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                    token_type_ids = buffered_token_type_ids_expanded
                else:
                    token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

            # 如果未提供inputs_embeds，则使用input_ids获取word_embeddings
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)

            # 计算最终的嵌入向量
            embeddings = inputs_embeds + token_type_embeddings
            if self.position_embedding_type == "absolute":
                position_embeddings = self.position_embeddings(position_ids)
                embeddings += position_embeddings
            embeddings = self.LayerNorm(embeddings)
            embeddings = self.dropout(embeddings)
            return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        给定直接的嵌入向量，我们无法推断哪些是填充的，因此只生成顺序位置标识符。

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        # 创建顺序的位置标识符
        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)
# 从transformers.models.bert.modeling_bert.BertSelfAttention复制并修改为ClapTextSelfAttention
class ClapTextSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 检查隐藏层大小是否能被注意力头数整除，若不满足条件且config没有embedding_size属性则引发错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 设置注意力头数和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化查询、键和值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # Dropout层用于注意力概率的随机失活
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        # 设置位置嵌入类型，默认为绝对位置嵌入
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )

        # 如果位置嵌入类型为相对键或相对键查询，则初始化距离嵌入层
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        # 标记是否为解码器
        self.is_decoder = config.is_decoder

    # 将输入张量重塑为注意力分数张量的形状
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播函数定义
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ):



# 从transformers.models.bert.modeling_bert.BertSelfOutput复制并修改为ClapTextSelfOutput
class ClapTextSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 全连接层，将隐藏状态映射回原始维度
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # LayerNorm层，用于归一化隐藏状态
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout层，用于隐藏状态的随机失活
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数定义
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 全连接层映射
        hidden_states = self.dense(hidden_states)
        # Dropout随机失活
        hidden_states = self.dropout(hidden_states)
        # LayerNorm归一化，并加上残差连接
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
# 定义 ClapTextAttention 类，继承自 nn.Module
class ClapTextAttention(nn.Module):
    # 初始化函数，接收 config 和 position_embedding_type 参数
    def __init__(self, config, position_embedding_type=None):
        # 调用父类的初始化函数
        super().__init__()
        # 创建 self 属性，调用 ClapTextSelfAttention 类，传入 config 和 position_embedding_type 参数
        self.self = ClapTextSelfAttention(config, position_embedding_type=position_embedding_type)
        # 创建 output 属性，调用 ClapTextSelfOutput 类，传入 config 参数
        self.output = ClapTextSelfOutput(config)
        # 创建 pruned_heads 属性，初始化为空集合
        self.pruned_heads = set()

    # 头部修剪函数，接收 heads 参数
    def prune_heads(self, heads):
        # 如果 heads 长度为 0，则直接返回
        if len(heads) == 0:
            return
        # 调用 find_pruneable_heads_and_indices 函数，获取可以修剪的头部及其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 修剪线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并记录修剪的头部
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 前向传播函数，接收多个输入参数，返回元组类型的张量
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 调用 self 的 forward 方法，传入多个参数，获取 self_outputs
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 调用 output 属性的 forward 方法，传入 self_outputs[0] 和 hidden_states，获取 attention_output
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出注意力，将 attentions 添加到 outputs 中
        outputs = (attention_output,) + self_outputs[1:]  # 如果需要输出注意力，将 attentions 添加到 outputs 中
        return outputs


# 从 transformers.models.bert.modeling_bert.BertIntermediate 类复制而来
class ClapTextIntermediate(nn.Module):
    # 初始化函数，接收 config 参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建 dense 属性，使用 nn.Linear 类，输入为 config.hidden_size 和 config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果 config.hidden_act 是字符串类型，则使用 ACT2FN 字典中对应的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播函数，接收 hidden_states 参数，返回 torch.Tensor 类型的张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用 dense 属性对 hidden_states 进行线性变换
        hidden_states = self.dense(hidden_states)
        # 使用 intermediate_act_fn 激活函数对 hidden_states 进行激活
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# 从 transformers.models.bert.modeling_bert.BertOutput 类复制而来
class ClapTextOutput(nn.Module):
    def __init__(self, config):
        # 调用父类构造函数进行初始化
        super().__init__()
        # 创建一个全连接层，将输入大小设为config中的中间层大小，输出大小为config中的隐藏层大小
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个LayerNorm层，输入大小为config中的隐藏层大小，设置epsilon为config中的layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个Dropout层，设置dropout概率为config中的hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用全连接层进行前向传播，将隐藏状态转换为新的表示
        hidden_states = self.dense(hidden_states)
        # 对转换后的表示进行dropout操作，以减少过拟合风险
        hidden_states = self.dropout(hidden_states)
        # 对dropout后的表示进行LayerNorm操作，并将输入张量与LayerNorm后的结果相加
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回经过处理后的隐藏状态张量
        return hidden_states
# 从transformers.models.bert.modeling_bert.BertLayer复制代码，并将Bert->ClapText
class ClapTextLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化层参数
        self.chunk_size_feed_forward = config.chunk_size_feed_forward  # 设置前馈过程的分块大小
        self.seq_len_dim = 1  # 序列长度维度设为1
        self.attention = ClapTextAttention(config)  # 创建ClapTextAttention对象
        self.is_decoder = config.is_decoder  # 是否为解码器模型
        self.add_cross_attention = config.add_cross_attention  # 是否添加交叉注意力
        if self.add_cross_attention:
            # 如果添加了交叉注意力，且不是解码器模型，则抛出错误
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 创建具有绝对位置嵌入类型的ClapTextAttention对象
            self.crossattention = ClapTextAttention(config, position_embedding_type="absolute")
        # 创建ClapTextIntermediate对象
        self.intermediate = ClapTextIntermediate(config)
        # 创建ClapTextOutput对象
        self.output = ClapTextOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        ) -> Tuple[torch.Tensor]:
        # 解码器单向自注意力的缓存键/值元组在位置1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 执行自注意力层的前向传播
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # 获取自注意力层的输出
        attention_output = self_attention_outputs[0]

        # 如果是解码器，最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            # 输出中排除最后一个元素，因为它是自注意力的缓存
            outputs = self_attention_outputs[1:-1]
            # 获取当前的键/值元组
            present_key_value = self_attention_outputs[-1]
        else:
            # 如果需要输出注意力权重，则包括自注意力层的输出
            outputs = self_attention_outputs[1:]
        
        cross_attn_present_key_value = None
        # 如果是解码器且有编码器隐藏状态作为输入
        if self.is_decoder and encoder_hidden_states is not None:
            # 如果未定义crossattention，抛出错误
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 交叉注意力的缓存键/值元组在过去键/值元组的第3,4位置
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 执行交叉注意力层的前向传播
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            # 获取交叉注意力层的输出
            attention_output = cross_attention_outputs[0]
            # 将交叉注意力层的输出添加到总输出中
            outputs = outputs + cross_attention_outputs[1:-1]

            # 将交叉注意力的键/值元组添加到当前键/值元组中
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 将注意力输出应用于前向传播的分块处理
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        # 将层输出添加到总输出中
        outputs = (layer_output,) + outputs

        # 如果是解码器，将注意力键/值作为最后一个输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        # 执行前馈网络的分块处理
        intermediate_output = self.intermediate(attention_output)
        # 应用激活函数和残差连接
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
# 从transformers.models.bert.modeling_bert.BertEncoder复制代码并修改为ClapTextEncoder
class ClapTextEncoder(nn.Module):
    # 初始化方法
    def __init__(self, config):
        super().__init__()
        # 保存配置信息
        self.config = config
        # 创建一个包含多个ClapTextLayer对象的层列表，数量由配置文件中的num_hidden_layers指定
        self.layer = nn.ModuleList([ClapTextLayer(config) for _ in range(config.num_hidden_layers)])
        # 是否使用梯度检查点，默认为False
        self.gradient_checkpointing = False

    # 前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        # 输入参数结束，这里只是声明参数，并未实际执行任何操作
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        # 如果输出隐藏状态，则初始化一个空元组用于存储所有隐藏状态
        all_hidden_states = () if output_hidden_states else None
        # 如果输出注意力权重，则初始化一个空元组用于存储所有自注意力权重
        all_self_attentions = () if output_attentions else None
        # 如果输出注意力权重且模型配置支持交叉注意力，则初始化一个空元组用于存储所有交叉注意力权重
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 如果启用梯度检查点且在训练模式下，检查是否与使用缓存同时设置。如果是，则发出警告并强制将use_cache设置为False
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # 如果不使用缓存，则初始化一个空元组来存储下一个解码器缓存
        next_decoder_cache = () if use_cache else None
        # 遍历所有解码器层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到all_hidden_states元组中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码（如果有的话）
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 获取当前层的过去键值对（如果有的话）
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 如果启用梯度检查点且在训练模式下，使用梯度检查点函数来计算当前层的输出
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                # 否则，直接调用当前层模块计算当前层的输出
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            # 更新当前隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]
            # 如果使用缓存，则将当前层的缓存添加到next_decoder_cache元组中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 如果需要输出注意力权重，则将当前层的自注意力权重添加到all_self_attentions元组中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果模型配置支持交叉注意力，则将当前层的交叉注意力权重添加到all_cross_attentions元组中
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果需要输出隐藏状态，则将最终隐藏状态添加到all_hidden_states元组中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典形式的输出，则返回一个元组，包含所有需要返回的结果，过滤掉值为None的项
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        # 否则，返回一个BaseModelOutputWithPastAndCrossAttentions对象，包含特定的输出结果
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
# Copied from transformers.models.bert.modeling_bert.BertPooler
class ClapTextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义一个全连接层，输入和输出维度均为config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义激活函数为双曲正切函数
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 从输入的hidden_states中获取第一个token对应的隐藏状态
        first_token_tensor = hidden_states[:, 0]
        # 将获取的隐藏状态输入全连接层，进行线性变换
        pooled_output = self.dense(first_token_tensor)
        # 将线性变换的结果输入激活函数，得到最终的池化输出
        pooled_output = self.activation(pooled_output)
        # 返回池化后的输出张量
        return pooled_output


class ClapPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 配置类为ClapConfig
    config_class = ClapConfig
    # 模型基础名称前缀为"clap"
    base_model_prefix = "clap"
    # 不支持梯度检查点
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        """Initialize the weights"""
        # 获取初始化因子
        factor = self.config.initializer_factor

        if isinstance(module, ClapTextEmbeddings):
            # 如果是文本嵌入模块，初始化位置嵌入和token类型嵌入的权重
            module.position_embeddings.weight.data.normal_(mean=0.0, std=factor * 0.02)
            module.token_type_embeddings.weight.data.normal_(mean=0.0, std=factor * 0.02)
        elif isinstance(module, ClapModel):
            # 如果是ClapModel，初始化logit_scale_a和logit_scale_t
            nn.init.normal_(module.logit_scale_a, std=factor * 0.02)
            nn.init.normal_(module.logit_scale_t, std=factor * 0.02)
        elif isinstance(module, nn.Embedding):
            # 如果是嵌入层，初始化权重
            module.weight.data.normal_(mean=0.0, std=factor * 0.02)
        elif isinstance(module, nn.LayerNorm):
            # 如果是LayerNorm层，初始化偏置为零，权重为1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, (nn.Conv2d, nn.Linear)):
            # 如果是卷积层或线性层，根据特定的初始化公式初始化权重
            in_proj_std = (self.config.hidden_size**-0.5) * ((2 * self.config.num_hidden_layers) ** -0.5) * factor
            nn.init.normal_(module.weight, std=in_proj_std)
            if module.bias is not None:
                module.bias.data.zero_()


class ClapAudioModel(ClapPreTrainedModel):
    # 配置类为ClapAudioConfig
    config_class = ClapAudioConfig
    # 主要输入名称为"input_features"
    main_input_name = "input_features"

    def __init__(self, config: ClapAudioConfig):
        super().__init__(config)
        # 初始化音频编码器
        self.audio_encoder = ClapAudioEncoder(config)
        # 初始化权重并进行最终处理
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        # 返回音频编码器中的投影嵌入层
        return self.audio_encoder.patch_embed.proj

    @add_start_docstrings_to_model_forward(CLAP_AUDIO_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=ClapAudioConfig)
    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        is_longer: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果 return_dict 参数不为 None，则使用其自身的值；否则使用 self.config.use_return_dict 的值

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果 output_attentions 参数不为 None，则使用其自身的值；否则使用 self.config.output_attentions 的值

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果 output_hidden_states 参数不为 None，则使用其自身的值；否则使用 self.config.output_hidden_states 的值

        return self.audio_encoder(
            input_features=input_features,
            is_longer=is_longer,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 调用 self.audio_encoder 方法，传入以下参数：
        # - input_features: 输入特征
        # - is_longer: 布尔值，指示输入是否较长
        # - output_attentions: 是否输出注意力权重，根据前面的处理得到的值
        # - output_hidden_states: 是否输出隐藏状态，根据前面的处理得到的值
        # - return_dict: 是否返回字典形式的输出结果，根据前面的处理得到的值
# 定义一个名为 ClapTextModel 的类，它继承自 ClapPreTrainedModel 类
class ClapTextModel(ClapPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in *Attention is
    all you need*_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.

    .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762
    """

    # 设置配置类为 ClapTextConfig
    config_class = ClapTextConfig

    # 从 transformers.models.bert.modeling_bert.BertModel.__init__ 复制的初始化函数，将 Bert 替换为 ClapText
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        # 初始化嵌入层为 ClapTextEmbeddings 对象
        self.embeddings = ClapTextEmbeddings(config)
        # 初始化编码器为 ClapTextEncoder 对象
        self.encoder = ClapTextEncoder(config)

        # 如果 add_pooling_layer 为 True，则初始化池化层为 ClapTextPooler 对象，否则为 None
        self.pooler = ClapTextPooler(config) if add_pooling_layer else None

        # 调用后处理函数，初始化权重并进行最终处理
        self.post_init()

    # 从 transformers.models.bert.modeling_bert.BertModel.forward 复制的前向传播函数
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 开始时增加文档字符串，未提供具体内容
        @add_start_docstrings(CLAP_START_DOCSTRING)
        class ClapModel(ClapPreTrainedModel):
            config_class = ClapConfig
    # 初始化方法，接受一个 ClapConfig 类型的参数 config
    def __init__(self, config: ClapConfig):
        # 调用父类的初始化方法，传入 config 参数
        super().__init__(config)

        # 检查 config.text_config 是否为 ClapTextConfig 类型，如果不是则抛出 ValueError 异常
        if not isinstance(config.text_config, ClapTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type ClapTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        # 检查 config.audio_config 是否为 ClapAudioConfig 类型，如果不是则抛出 ValueError 异常
        if not isinstance(config.audio_config, ClapAudioConfig):
            raise ValueError(
                "config.audio_config is expected to be of type ClapAudioConfig but is of type"
                f" {type(config.audio_config)}."
            )

        # 将 config 中的 text_config 和 audio_config 分别赋值给局部变量 text_config 和 audio_config
        text_config = config.text_config
        audio_config = config.audio_config

        # 初始化 logit_scale_a 和 logit_scale_t 为对数形式的初始值
        self.logit_scale_a = nn.Parameter(torch.tensor(math.log(config.logit_scale_init_value)))
        self.logit_scale_t = nn.Parameter(torch.tensor(math.log(config.logit_scale_init_value)))

        # 将 projection_dim 初始化为 config 中指定的 projection_dim
        self.projection_dim = config.projection_dim

        # 初始化文本模型和文本投影层
        self.text_model = ClapTextModel(text_config)
        self.text_projection = ClapProjectionLayer(text_config)

        # 初始化音频模型和音频投影层
        self.audio_model = ClapAudioModel(audio_config)
        self.audio_projection = ClapProjectionLayer(audio_config)

        # 调用自定义的 post_init 方法，用于初始化权重并进行最终处理
        self.post_init()
        ) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`ClapTextModel`].

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, ClapModel

        >>> model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        >>> tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

        >>> inputs = tokenizer(["the sound of a cat", "the sound of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        # Use CLAP model's config for some fields (if specified) instead of those of audio & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass input_ids, attention_mask, position_ids, and other relevant parameters to the text_model
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Determine pooled_output based on whether return_dict is enabled
        pooled_output = text_outputs[1] if return_dict is not None else text_outputs.pooler_output

        # Project pooled_output to obtain text_features
        text_features = self.text_projection(pooled_output)

        # Normalize text_features along the last dimension
        text_features = F.normalize(text_features, dim=-1)

        # Return the normalized text_features
        return text_features
        ) -> torch.FloatTensor:
        r"""
        Returns:
            audio_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The audio embeddings obtained by
            applying the projection layer to the pooled output of [`ClapAudioModel`].

        Examples:

        ```python
        >>> from transformers import AutoFeatureExtractor, ClapModel
        >>> import torch

        >>> model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        >>> feature_extractor = AutoFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")
        >>> random_audio = torch.rand((16_000))
        >>> inputs = feature_extractor(random_audio, return_tensors="pt")
        >>> audio_features = model.get_audio_features(**inputs)
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用音频模型获取音频输出特征
        audio_outputs = self.audio_model(
            input_features=input_features,
            is_longer=is_longer,
            return_dict=return_dict,
        )

        # 根据返回的结果是否使用字典形式确定池化输出
        pooled_output = audio_outputs[1] if not return_dict else audio_outputs.pooler_output

        # 使用音频投影层对池化输出进行投影
        audio_features = self.audio_projection(pooled_output)

        # 对音频特征进行标准化处理
        audio_features = F.normalize(audio_features, dim=-1)

        # 返回处理后的音频特征
        return audio_features

    @add_start_docstrings_to_model_forward(CLAP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ClapOutput, config_class=ClapConfig)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        is_longer: Optional[torch.BoolTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
"""
CLAP Audio Model with a projection layer on top (a linear layer on top of the pooled output).
"""

@add_start_docstrings(
    """
    CLAP Text Model with a projection layer on top (a linear layer on top of the pooled output).
    """,
    CLAP_START_DOCSTRING,
)
class ClapTextModelWithProjection(ClapPreTrainedModel):
    # 指定配置类
    config_class = ClapTextConfig

    def __init__(self, config: ClapTextConfig):
        # 调用父类初始化方法
        super().__init__(config)
        # 初始化文本模型
        self.text_model = ClapTextModel(config)
        # 初始化投影层
        self.text_projection = ClapProjectionLayer(config)
        # 初始化权重并进行最终处理
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        # 返回文本模型的词嵌入层
        return self.text_model.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        # 设置文本模型的词嵌入层
        self.text_model.embeddings.word_embeddings = value

    @add_start_docstrings_to_model_forward(CLAP_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ClapTextModelOutput, config_class=ClapTextConfig)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ClapTextModelOutput]:
        """
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, ClapTextModelWithProjection

        >>> model = ClapTextModelWithProjection.from_pretrained("laion/clap-htsat-unfused")
        >>> tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

        >>> inputs = tokenizer(["a sound of a cat", "a sound of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> text_embeds = outputs.text_embeds
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用文本模型的forward方法
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取池化后的输出
        pooled_output = text_outputs[1] if not return_dict else text_outputs.pooler_output

        # 通过投影层得到文本嵌入
        text_embeds = self.text_projection(pooled_output)

        if not return_dict:
            # 如果不返回字典，则返回元组形式的输出
            outputs = (text_embeds, text_outputs[0]) + text_outputs[2:]
            return tuple(output for output in outputs if output is not None)

        # 如果返回字典，则构造ClapTextModelOutput对象返回
        return ClapTextModelOutput(
            text_embeds=text_embeds,
            last_hidden_state=text_outputs.last_hidden_state,
            hidden_states=text_outputs.hidden_states,
            attentions=text_outputs.attentions,
        )


@add_start_docstrings(
    """
    CLAP Audio Model with a projection layer on top (a linear layer on top of the pooled output).
    """,
    CLAP_START_DOCSTRING,
)
class ClapAudioModelWithProjection(ClapPreTrainedModel):
    config_class = ClapAudioConfig  # 设置类的配置类为ClapAudioConfig

    main_input_name = "input_features"  # 主要输入名称为"input_features"

    def __init__(self, config: ClapAudioConfig):
        super().__init__(config)  # 调用父类构造函数，初始化模型配置

        self.audio_model = ClapAudioModel(config)  # 初始化音频模型
        self.audio_projection = ClapProjectionLayer(config)  # 初始化音频投影层

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.audio_model.audio_encoder.patch_embed.proj  # 返回输入嵌入的投影层

    @add_start_docstrings_to_model_forward(CLAP_AUDIO_INPUTS_DOCSTRING)  # 添加模型前向传播的文档字符串
    @replace_return_docstrings(output_type=ClapAudioModelOutput, config_class=ClapAudioConfig)  # 替换返回的文档字符串为ClapAudioModelOutput类型

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        is_longer: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ClapAudioModelOutput]:
        r"""
        前向传播函数，接受以下参数并返回相应的输出：

        - input_features (Optional[torch.FloatTensor]): 输入特征张量，默认为None。
        - is_longer (Optional[torch.BoolTensor]): 是否为较长输入张量，默认为None。
        - output_attentions (Optional[bool]): 是否输出注意力，默认为None。
        - output_hidden_states (Optional[bool]): 是否输出隐藏状态，默认为None。
        - return_dict (Optional[bool]): 是否返回字典类型输出，默认为None。

        Returns:
        Union[Tuple, ClapAudioModelOutput]: 返回音频嵌入或ClapAudioModelOutput对象。

        Examples:

        ```python
        >>> from datasets import load_dataset
        >>> from transformers import ClapAudioModelWithProjection, ClapProcessor

        >>> model = ClapAudioModelWithProjection.from_pretrained("laion/clap-htsat-fused")
        >>> processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")

        >>> dataset = load_dataset("ashraq/esc50")
        >>> audio_sample = dataset["train"]["audio"][0]["array"]

        >>> inputs = processor(audios=audio_sample, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> audio_embeds = outputs.audio_embeds
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # 设置返回字典的类型
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions  # 设置输出注意力的类型
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )  # 设置输出隐藏状态的类型

        audio_outputs = self.audio_model(
            input_features=input_features,
            is_longer=is_longer,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )  # 获取音频模型的输出

        pooled_output = audio_outputs[1] if not return_dict else audio_outputs.pooler_output  # 获取池化输出

        audio_embeds = self.audio_projection(pooled_output)  # 使用音频投影层得到音频嵌入

        if not return_dict:
            outputs = (audio_embeds, audio_outputs[0]) + audio_outputs[2:]  # 构建输出元组
            return tuple(output for output in outputs if output is not None)  # 返回非空的输出元组

        return ClapAudioModelOutput(
            audio_embeds=audio_embeds,
            last_hidden_state=audio_outputs.last_hidden_state,
            attentions=audio_outputs.attentions,
            hidden_states=audio_outputs.hidden_states,
        )  # 返回ClapAudioModelOutput对象
```