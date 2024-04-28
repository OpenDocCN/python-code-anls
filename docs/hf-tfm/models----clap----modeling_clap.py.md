# `.\transformers\models\clap\modeling_clap.py`

```
# 设置文件编码为 UTF-8
# 版权声明
""" PyTorch CLAP 模型。"""
import collections  # 导入 collections 库，用于创建特定数据结构
import math  # 导入 math 库，提供数学函数
from dataclasses import dataclass  # 导入 dataclasses 库，用于数据类的创建
from typing import Any, List, Optional, Tuple, Union  # 导入 typing 库，提供类型提示功能

import torch  # 导入 PyTorch 库
import torch.nn.functional as F  # 导入 PyTorch 中的函数模块
from torch import nn  # 导入 PyTorch 中的神经网络模块

from ...activations import ACT2FN  # 导入激活函数映射
from ...modeling_outputs import (  # 导入模型输出相关类
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from ...modeling_utils import PreTrainedModel  # 导入预训练模型相关工具函数
from ...pytorch_utils import (  # 导入 PyTorch 相关工具函数
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    meshgrid,
    prune_linear_layer,
)
from ...utils import (  # 导入通用工具函数
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_clap import ClapAudioConfig, ClapConfig, ClapTextConfig  # 导入 CLAP 模型配置类

logger = logging.get_logger(__name__)  # 获取日志记录器

_CHECKPOINT_FOR_DOC = "laion/clap-htsat-fused"  # 用于文档的检查点地址

CLAP_PRETRAINED_MODEL_ARCHIVE_LIST = [  # CLAP 预训练模型地址列表
    "laion/clap-htsat-fused",
    "laion/clap-htsat-unfused",
    # 查看所有 CLAP 模型 https://huggingface.co/models?filter=clap
]


# 从 https://github.com/LAION-AI/CLAP/blob/6ad05a971ba0622f6acee8c41993e0d02bbed639/src/open_clip/utils.py#L191 适配而来
def interpolate(hidden_states, ratio):
    """
    在时间域内插值数据。这用于补偿 CNN 下采样时分辨率的降低。

    Args:
        hidden_states (`torch.FloatTensor` of shape (batch_size, time_length, classes_num)):
            输入的隐藏状态
        ratio (`int`):
            输出长度与输入长度的比率。
    """
    (batch_size, time_length, classes_num) = hidden_states.shape  # 获取隐藏状态的形状信息
    upsampled = hidden_states[:, :, None, :].repeat(1, 1, ratio, 1)  # 沿指定维度重复张量
    upsampled = upsampled.reshape(batch_size, time_length * ratio, classes_num)  # 重新塑造张量形状
    return upsampled  # 返回上采样后的隐藏状态


# 从 https://github.com/LAION-AI/CLAP/blob/6ad05a971ba0622f6acee8c41993e0d02bbed639/src/open_clip/htsat.py#L249 适配而来
def window_partition(hidden_states, window_size):
    """
    返回调整大小后的隐藏状态。输出形状应为 `(batch_size * num_windows, window_size, window_size, num_channels)`

    Args:
        hidden_states (`torch.FloatTensor` of shape `(batch_size, height, width, num_channels)`):
            输入的隐藏状态
        window_size (`int`):
            窗口大小
```  
    """
    # 获取隐藏状态的形状信息，包括批大小、高度、宽度和通道数
    batch_size, height, width, num_channels = hidden_states.shape

    # 重新调整隐藏状态的形状，将其分割成窗口大小的子块
    hidden_states = hidden_states.view(
        batch_size, height // window_size, window_size, width // window_size, window_size, num_channels
    )
    
    # 对隐藏状态进行排列操作，以便按照指定顺序重新组织数据
    windows = hidden_states.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, num_channels)
    
    # 返回重新组织后的窗口数据
    return windows
# 从 https://github.com/LAION-AI/CLAP/blob/6ad05a971ba0622f6acee8c41993e0d02bbed639/src/open_clip/htsat.py#L263 改编
def window_reverse(windows, window_size, height, width):
    """
    合并窗口以生成更高分辨率的特征。
    Args:
        windows (`torch.FloatTensor` of shape `(num_windows * batch_size, window_size, window_size, num_channels)`):
            输入窗口
        window_size (`int`):
            窗口大小
        height (`int`):
            调整后音频的高度
        width (`int`):
            调整后音频的宽度
    """
    num_channels = windows.shape[-1]
    # 重新排列窗口以匹配输出的高分辨率特征
    windows = windows.view(-1, height // window_size, width // window_size, window_size, window_size, num_channels)
    # 转置窗口以使维度匹配
    windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, height, width, num_channels)
    return windows


# 从 transformers.models.roberta.modeling_roberta.create_position_ids_from_input_ids 复制
def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    用位置数字替换非填充符号。位置编号从 padding_idx+1 开始。填充符号被忽略。这是从 fairseq 的 `utils.make_positions` 修改而来。

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # 创建一个掩码，表示非填充的位置
    mask = input_ids.ne(padding_idx).int()
    # 根据输入的位置 id 创建递增的位置 id
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx


# 对比损失函数，从 https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html#CLIP-loss-function 改编
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    # 创建标签，与 logits 的长度相同
    labels = torch.arange(len(logits), device=logits.device)
    # 使用交叉熵损失计算对比损失
    return nn.functional.cross_entropy(logits, labels)


@dataclass
# 从 transformers.models.clip.modeling_clip.CLIPTextModelOutput 复制，将 CLIP->Clap
class ClapTextModelOutput(ModelOutput):
    """
    包含文本模型输出的基类，还包含最后隐藏状态的池化。
``` 
    Args:
        text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            模型使用 projection 层处理 pooler_output 后得到的文本嵌入。
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的隐藏状态序列。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            包含模型每一层隐藏状态的元组，形状为 `(batch_size, sequence_length, hidden_size)`。

            模型每一层的隐藏状态以及可选的初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            包含模型每一层注意力权重的元组，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            自注意力头中用于计算加权平均值的注意力权重。
    """

    # 可选参数：文本嵌入
    text_embeds: Optional[torch.FloatTensor] = None
    # 必需参数：最后一层的隐藏状态序列
    last_hidden_state: torch.FloatTensor = None
    # 可选参数：模型每一层隐藏状态的元组
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 可选参数：模型每一层注意力权重的元组
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# 用于定义Clap模型的输出，模拟原始实现的输出格式
@dataclass
class ClapAudioModelOutput(ModelOutput):
    """
    ClapAudio模型的输出，用于模仿原始实现的输出。

    Args:
        audio_embeds (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            通过将投影层应用于pooler_output得到的音频嵌入。
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            模型最后一层的输出的隐藏状态序列。
        attentions (`tuple(torch.FloatTensor)`, *optional*, 当传入`output_attentions=True`或`config.output_attentions=True`时返回):
            由多个`torch.FloatTensor`组成的元组（每层一个），形状为`(batch_size, num_heads, sequence_length, sequence_length)`。

            在注意力softmax之后的注意力权重，用于计算自注意力头中的加权平均值。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, 当传入`output_hidden_states=True`或`config.output_hidden_states=True`时返回):
            由多个`torch.FloatTensor`组成的元组（如果模型有嵌入层，则包括嵌入层输出和每个层的输出），形状为`(batch_size, sequence_length, hidden_size)`。

            每层模型的隐藏状态，以及可选的初始嵌入层输出。
    """

    audio_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
# 从transformers.models.clip.modeling_clip.CLIPOutput复制而来，将CLIP改为Clap，vision改为audio，Vision改为Audio，image改为audio
class ClapOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for audio-text similarity.
        logits_per_audio: (`torch.FloatTensor` of shape `(audio_batch_size, text_batch_size)`):
            The scaled dot product scores between `audio_embeds` and `text_embeds`. This represents the audio-text
            similarity scores.
        logits_per_text: (`torch.FloatTensor` of shape `(text_batch_size, audio_batch_size)`):
            The scaled dot product scores between `text_embeds` and `audio_embeds`. This represents the text-audio
            similarity scores.
        text_embeds: (`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of [`ClapTextModel`].
        audio_embeds: (`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The audio embeddings obtained by applying the projection layer to the pooled output of [`ClapAudioModel`].
        text_model_output: (`BaseModelOutputWithPooling`):
            The output of the [`ClapTextModel`].
        audio_model_output: (`BaseModelOutputWithPooling`):
            The output of the [`ClapAudioModel`].
    """

    # 定义可选的损失值，默认为None
    loss: Optional[torch.FloatTensor] = None
    # 定义音频文本之间的分数，默认为None
    logits_per_audio: torch.FloatTensor = None
    # 定义文本音频之间的分数，默认为None
    logits_per_text: torch.FloatTensor = None
    # 定义文本嵌入，默认为None
    text_embeds: torch.FloatTensor = None
    # 定义音频嵌入，默认为None
    audio_embeds: torch.FloatTensor = None
    # 定义文本模型的输出，默认为None
    text_model_output: BaseModelOutputWithPooling = None
    # 定义音频模型的输出，默认为None
    audio_model_output: BaseModelOutputWithPooling = None

    # 将对象转换为元组
    def to_tuple(self) -> Tuple[Any]:
        # 对每个键值对进行处理，如果键不在排除列表中，则直接返回对应值；否则调用该值的to_tuple()方法
        return tuple(
            self[k] if k not in ["text_model_output", "audio_model_output"] else getattr(self, k).to_tuple()
            for k in self.keys()
        )
# 从transformers.models.swin.modeling_swin.SwinDropPath调整而来的类，用于在残差块的主路径上对每个样本应用Drop paths（随机深度）。
# 这是`SwinDropPath`实现的略微重构版本。
class ClapDropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks). This is a slightly
    refactored version of the `SwinDropPath` implementation.
    """

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states):
        # 如果drop_prob为0或者不在训练模式下，直接返回隐藏状态
        if self.drop_prob == 0.0 or not self.training:
            return hidden_states

        keep_prob = 1 - self.drop_prob
        # 处理不同维度的张量，不仅限于2D卷积网络
        shape = (hidden_states.shape[0],) + (1,) * (hidden_states.ndim - 1)

        # 生成随机张量，其值在[keep_prob, 1)之间，与隐藏状态张量相同的数据类型和设备上
        random_tensor = keep_prob + torch.rand(shape, dtype=hidden_states.dtype, device=hidden_states.device)
        random_tensor.floor_()  # 二值化
        output = hidden_states.div(keep_prob) * random_tensor
        return output


# 从https://github.com/LAION-AI/CLAP/blob/6ad05a971ba0622f6acee8c41993e0d02bbed639/src/open_clip/feature_fusion.py#L133调整而来的类
# ATTENTIONAL FEATURE FUSION Block来自CLAP，因为在CLAP中我们总是处于2D模式，因此不需要实现1D版本。
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

        # 本地注意力层
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 全局注意力层
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden_states, residual):
        attention_input = hidden_states + residual

        # 融合本地和全局注意力的输出
        fused_layer_output = self.local_att(attention_input) + self.global_att(attention_input)
        # 对融合输出进行Sigmoid激活
        fused_layer_output = self.sigmoid(fused_layer_output)

        # 计算最终输出
        output = 2 * hidden_states * fused_layer_output + 2 * residual * (1 - fused_layer_output)
        return output


class ClapAudioPatchEmbed(nn.Module):
    """
    This module converts the hidden states reshaped as an image to patch embeddings ready to be passed to the
    Transformer block.
    """
    # 初始化函数，接受一个 ClapAudioConfig 类型的参数
    def __init__(self, config: ClapAudioConfig):
        # 调用父类的初始化函数
        super().__init__()
        
        # 根据配置中的 spec_size 确定图像大小
        img_size = (config.spec_size, config.spec_size) if isinstance(config.spec_size, int) else config.spec_size
        # 根据配置中的 patch_size 确定补丁大小
        patch_size = (
            (config.patch_size, config.patch_size) if isinstance(config.patch_size, int) else config.patch_size
        )
        # 根据配置中的 patch_stride 确定补丁步幅
        patch_stride = (
            (config.patch_stride, config.patch_stride) if isinstance(config.patch_stride, int) else config.patch_stride
        )

        # 将图像大小和补丁步幅保存到对象中
        self.img_size = img_size
        self.patch_stride = patch_stride

        # 计算网格大小，即图像大小除以补丁步幅
        self.grid_size = (img_size[0] // patch_stride[0], img_size[1] // patch_stride[1])
        # 计算补丁数量，即网格大小的乘积
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # 根据配置中的 flatten_patch_embeds 和 enable_fusion 设置对象属性
        self.flatten = config.flatten_patch_embeds
        self.enable_fusion = config.enable_fusion

        # 计算填充大小，用于卷积层的 padding 参数
        padding = ((patch_size[0] - patch_stride[0]) // 2, (patch_size[1] - patch_stride[1]) // 2)

        # 根据 enable_fusion 和 fusion_type 设置 scale_factor
        scale_factor = 4 if (self.enable_fusion) and (config.fusion_type == "channel_map") else 1

        # 创建卷积层对象，用于将输入的补丁转换为隐藏表示
        self.proj = nn.Conv2d(
            config.patch_embed_input_channels * scale_factor,
            config.patch_embeds_hidden_size,
            kernel_size=patch_size,
            stride=patch_stride,
            padding=padding,
        )

        # 根据配置中的 enable_patch_layer_norm 设置 LayerNorm 或 Identity 层
        self.norm = nn.LayerNorm(config.patch_embeds_hidden_size) if config.enable_patch_layer_norm else nn.Identity()
        
        # 如果启用了融合，则创建 ClapAudioAFFBlock 和 mel_conv2d 卷积层对象
        if self.enable_fusion:
            self.fusion_model = ClapAudioAFFBlock(config)
            self.mel_conv2d = nn.Conv2d(
                config.patch_embed_input_channels,
                config.patch_embeds_hidden_size,
                kernel_size=(patch_size[0], patch_size[1] * 3),
                stride=(patch_stride[0], patch_stride[1] * 3),
                padding=padding,
            )
    # 定义前向传播函数，接受隐藏状态和可选的较长的序列索引
    def forward(self, hidden_states, is_longer_idx=None):
        # 如果启用了融合
        if self.enable_fusion:
            # 从隐藏状态中提取最后一个 mel，因为输入已经转置
            global_hidden_states = hidden_states[:, 0:1, :, :]

            # 全局处理
            batch_size, num_channels, height, width = global_hidden_states.shape

            # 检查输入音频尺寸是否与模型匹配
            if height != self.img_size[0] or width != self.img_size[1]:
                raise ValueError(
                    f"Input audio size ({height}*{width}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
                )

            # 对全局隐藏状态进行投影
            global_hidden_states = self.proj(global_hidden_states)
            output_width = global_hidden_states.size(-1)
            
            # 如果存在较长的序列索引
            if len(is_longer_idx) > 0:
                # 本地处理
                local_hidden_states = hidden_states[is_longer_idx, 1:, :, :].contiguous()
                batch_size, num_channels, height, width = local_hidden_states.shape
                local_hidden_states = local_hidden_states.view(batch_size * num_channels, 1, height, width)

                # 对本地隐藏状态进行 mel 卷积
                local_hidden_states = self.mel_conv2d(local_hidden_states)

                _, features, height, width = local_hidden_states.shape
                local_hidden_states = local_hidden_states.view(batch_size, num_channels, features, height, width)
                local_hidden_states = local_hidden_states.permute((0, 2, 3, 1, 4)).contiguous().flatten(3)

                local_width = local_hidden_states.size(-1)
                # 对本地隐藏状态进行填充，使其与全局隐藏状态宽度相匹配
                local_hidden_states = torch.nn.functional.pad(
                    local_hidden_states, (0, output_width - local_width), "constant", 0
                )

                # 使用融合模型将全局隐藏状态和本地隐藏状态进行融合
                global_hidden_states[is_longer_idx] = self.fusion_model(
                    global_hidden_states[is_longer_idx], local_hidden_states
                )
            # 更新隐藏状态为全局隐藏状态
            hidden_states = global_hidden_states
        else:
            _, _, height, width = hidden_states.shape
            # 检查输入音频尺寸是否与模型匹配
            if height != self.img_size[0] or width != self.img_size[1]:
                raise ValueError(
                    f"Input audio size ({height}*{width}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
                )
            # 对隐藏状态进行投影
            hidden_states = self.proj(hidden_states)

        # 如果设置了 flatten 标志，则对隐藏状态进行展平并转置
        if self.flatten:
            hidden_states = hidden_states.flatten(2).transpose(1, 2)
        # 对隐藏状态进行规范化
        hidden_states = self.norm(hidden_states)
        # 返回隐藏状态
        return hidden_states
```  
# 从transformers.models.swin.modeling_swin.SwinSelfAttention复制代码并更名为ClapAudioSelfAttention
class ClapAudioSelfAttention(nn.Module):
    def __init__(self, config, dim, num_heads, window_size):
        super().__init__()
        # 如果隐藏大小不是注意力头数的倍数，则引发错误
        if dim % num_heads != 0:
            raise ValueError(
                f"The hidden size ({dim}) is not a multiple of the number of attention heads ({num_heads})"
            )

        self.num_attention_heads = num_heads
        self.attention_head_size = int(dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # 确定窗口大小
        self.window_size = (
            window_size if isinstance(window_size, collections.abc.Iterable) else (window_size, window_size)
        )

        # 创建相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads)
        )

        # 获取窗口内每个令牌的成对相对位置索引
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
        # 注册相对位置索引为缓冲区
        self.register_buffer("relative_position_index", relative_position_index)

        # 初始化查询、键、值权重
        self.query = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(self.all_head_size, self.all_head_size, bias=config.qkv_bias)

        # 初始化dropout层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    # 重塑张量形状以便计算注意力分数
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 获取隐藏状态张量的维度信息
        batch_size, dim, num_channels = hidden_states.shape
        # 计算混合查询层
        mixed_query_layer = self.query(hidden_states)

        # 计算键层并转置以供计算得分
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        # 计算值层并转置以供计算得分
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        # 对混合查询层进行转置以供计算得分
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 计算原始注意力分数，通过查询和键的点积得到
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # 对注意力分数进行缩放
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 获取相对位置偏置
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )

        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        # 将相对位置偏置添加到注意力分数上
        attention_scores = attention_scores + relative_position_bias.unsqueeze(0)

        if attention_mask is not None:
            # 应用注意力掩码
            mask_shape = attention_mask.shape[0]
            attention_scores = attention_scores.view(
                batch_size // mask_shape, mask_shape, self.num_attention_heads, dim, dim
            )
            attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(0)
            attention_scores = attention_scores.view(-1, self.num_attention_heads, dim, dim)

        # 将注意力分数归一化为概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 使用 dropout 进行正则化
        attention_probs = self.dropout(attention_probs)

        # 如果存在头部掩码，则将其应用到注意力概率上
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文层
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # 输出结果，包括上下文层和注意力概率
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
# 从transformers.models.swin.modeling_swin.SwinSelfOutput复制并修改为ClapAudioSelfOutput类
class ClapAudioSelfOutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        # 创建一个线性层，输入和输出维度均为dim
        self.dense = nn.Linear(dim, dim)
        # 创建一个dropout层，使用config中的attention_probs_dropout_prob作为dropout概率
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将hidden_states输入到线性层中
        hidden_states = self.dense(hidden_states)
        # 对输出进行dropout处理
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# 从transformers.models.swin.modeling_swin.SwinAttention复制并修改为ClapAudioAttention类
class ClapAudioAttention(nn.Module):
    def __init__(self, config, dim, num_heads, window_size):
        super().__init__()
        # 创建ClapAudioSelfAttention对象
        self.self = ClapAudioSelfAttention(config, dim, num_heads, window_size)
        # 创建ClapAudioSelfOutput对象
        self.output = ClapAudioSelfOutput(config, dim)
        # 初始化一个空集合用于存储被剪枝的头
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 找到可剪枝的头和索引
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 调用self对象的forward方法
        self_outputs = self.self(hidden_states, attention_mask, head_mask, output_attentions)
        # 将self输出传递给output对象
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出注意力，则将其添加到outputs中
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


# 从transformers.models.swin.modeling_swin.SwinIntermediate复制并修改为ClapAudioIntermediate类
class ClapAudioIntermediate(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        # 创建一个线性层，输入维度为dim，输出维度为config.mlp_ratio * dim
        self.dense = nn.Linear(dim, int(config.mlp_ratio * dim))
        # 根据config中的hidden_act设置激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将hidden_states输入到线性层中
        hidden_states = self.dense(hidden_states)
        # 使用设置的激活函数处理输出
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
# 从transformers.models.swin.modeling_swin.SwinOutput复制类定义，将Swin->ClapAudio
class ClapAudioOutput(nn.Module):
    def __init__(self, config, dim):
        super().__init__()
        # 创建一个线性层，将输入维度变换为config.mlp_ratio * dim，输出维度为dim
        self.dense = nn.Linear(int(config.mlp_ratio * dim), dim)
        # 创建一个Dropout层，以config.hidden_dropout_prob的概率丢弃输入
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 输入hidden_states经过线性层变换
        hidden_states = self.dense(hidden_states)
        # 对变换后的结果进行丢弃
        hidden_states = self.dropout(hidden_states)
        # 返回结果
        return hidden_states


# 从transformers.models.swin.modeling_swin.SwinLayer复制类定义，将SwinDropPath->ClapDropPath, Swin->ClapAudio
class ClapAudioLayer(nn.Module):
    def __init__(self, config, dim, input_resolution, num_heads, shift_size=0):
        super().__init__()
        # 设置Feed Forward模块的分块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 设置平移大小
        self.shift_size = shift_size
        # 设置窗口大小
        self.window_size = config.window_size
        # 设置输入分辨率
        self.input_resolution = input_resolution
        # 创建LayerNorm层，归一化输入向量的维度为dim
        self.layernorm_before = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        # 创建注意力机制模块
        self.attention = ClapAudioAttention(config, dim, num_heads, window_size=self.window_size)
        # 如果dropout rate大于0，则创建ClapDropPath层，否则创建Identity层
        self.drop_path = ClapDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        # 创建LayerNorm层，归一化输入向量的维度为dim
        self.layernorm_after = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        # 创建中间层，用于前馈网络
        self.intermediate = ClapAudioIntermediate(config, dim)
        # 创建输出层
        self.output = ClapAudioOutput(config, dim)

    def set_shift_and_window_size(self, input_resolution):
        # 如果输入分辨率中的最小值小于等于窗口大小，则不分区窗口
        if min(input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(input_resolution)

    def get_attn_mask(self, height, width, dtype):
        # 如果平移大小大于0，则计算自注意力机制的注意力掩码
        if self.shift_size > 0:
            # 计算自注意力机制的掩码
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
    # 可能对隐藏状态进行填充，以确保其高度和宽度能够被窗口大小整除
    def maybe_pad(self, hidden_states, height, width):
        # 计算需要向右填充的数量，确保宽度能够被窗口大小整除
        pad_right = (self.window_size - width % self.window_size) % self.window_size
        # 计算需要向下填充的数量，确保高度能够被窗口大小整除
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        # 定义填充的数值，格式为 (top, bottom, left, right, ...)
        pad_values = (0, 0, 0, pad_right, 0, pad_bottom)
        # 对隐藏状态进行填充
        hidden_states = nn.functional.pad(hidden_states, pad_values)
        # 返回填充后的隐藏状态和填充数值
        return hidden_states, pad_values

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        always_partition: Optional[bool] = False,
    # 定义函数的输入和输出类型为 torch.Tensor 的元组
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 如果不总是分区，则设置偏移和窗口大小
        if not always_partition:
            self.set_shift_and_window_size(input_dimensions)
        else:
            pass
        # 获取输入维度的高度和宽度
        height, width = input_dimensions
        # 获取隐藏状态的批量大小、通道数
        batch_size, _, channels = hidden_states.size()
        # 保存隐藏状态的快捷方式
        shortcut = hidden_states

        # 在 layernorm 之前对隐藏状态进行处理
        hidden_states = self.layernorm_before(hidden_states)

        # 将隐藏状态重塑为(batch_size, height, width, channels)的形状
        hidden_states = hidden_states.view(batch_size, height, width, channels)

        # 对隐藏状态进行填充，使其成为窗口大小的倍数
        hidden_states, pad_values = self.maybe_pad(hidden_states, height, width)

        # 获取填充后隐藏状态的维度
        _, height_pad, width_pad, _ = hidden_states.shape
        # 循环移位
        if self.shift_size > 0:
            shifted_hidden_states = torch.roll(hidden_states, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_hidden_states = hidden_states

        # 分区窗口
        hidden_states_windows = window_partition(shifted_hidden_states, self.window_size)
        hidden_states_windows = hidden_states_windows.view(-1, self.window_size * self.window_size, channels)
        # 获取注意力掩码
        attn_mask = self.get_attn_mask(height_pad, width_pad, dtype=hidden_states.dtype)
        if attn_mask is not None:
            attn_mask = attn_mask.to(hidden_states_windows.device)

        # 进行注意力计算
        attention_outputs = self.attention(
            hidden_states_windows, attn_mask, head_mask, output_attentions=output_attentions
        )

        attention_output = attention_outputs[0]

        # 将注意力输出重塑为(batch_size, height, width, channels)的形状
        attention_windows = attention_output.view(-1, self.window_size, self.window_size, channels)
        shifted_windows = window_reverse(attention_windows, self.window_size, height_pad, width_pad)

        # 反向循环移位
        if self.shift_size > 0:
            attention_windows = torch.roll(shifted_windows, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attention_windows = shifted_windows

        # 如果进行了填充，则截取注意力窗口
        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
            attention_windows = attention_windows[:, :height, :width, :].contiguous()

        # 将注意力窗口重塑为(batch_size, height * width, channels)的形状
        attention_windows = attention_windows.view(batch_size, height * width, channels)

        # 计算最终隐藏状态
        hidden_states = shortcut + self.drop_path(attention_windows)

        # 在隐藏状态上进行 layernorm
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = hidden_states + self.output(layer_output)

        # 如果需要输出注意力，则返回注意力输出
        layer_outputs = (layer_output, attention_outputs[1]) if output_attentions else (layer_output,)
        return layer_outputs
# 从 transformers.models.swin.modeling_swin 模块中复制 ClapAudioStage 类，并将其中的 Swin 替换为 ClapAudio
class ClapAudioStage(nn.Module):
    def __init__(self, config, dim, input_resolution, depth, num_heads, drop_path, downsample):
        super().__init__()
        # 保存传入的配置
        self.config = config
        # 保存传入的维度信息
        self.dim = dim
        # 创建包含多个 ClapAudioLayer 实例的模块列表，每个实例代表一个层
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

        # 如果存在下采样操作，则创建下采样层；否则设为 None
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=nn.LayerNorm)
        else:
            self.downsample = None

        # 是否指定了某个位置
        self.pointing = False

    # 前向传播函数，接收隐藏状态、输入维度、头部掩码、是否输出注意力矩阵、是否总是分区参数等，返回隐藏状态和其他信息的元组
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_dimensions: Tuple[int, int],
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        always_partition: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 获取输入维度的高度和宽度
        height, width = input_dimensions
        # 遍历所有层
        for i, layer_module in enumerate(self.blocks):
            # 如果存在头部掩码，则获取当前层的头部掩码；否则设为 None
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 调用当前层的前向传播函数
            layer_outputs = layer_module(
                hidden_states, input_dimensions, layer_head_mask, output_attentions, always_partition
            )

            # 更新隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]

        # 在进行下采样之前保存隐藏状态
        hidden_states_before_downsampling = hidden_states
        # 如果存在下采样操作，则进行下采样
        if self.downsample is not None:
            # 计算下采样后的高度和宽度
            height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2
            # 计算输出维度
            output_dimensions = (height, width, height_downsampled, width_downsampled)
            # 调用下采样层的前向传播函数
            hidden_states = self.downsample(hidden_states_before_downsampling, input_dimensions)
        else:
            # 如果不存在下采样操作，则输出维度与输入维度相同
            output_dimensions = (height, width, height, width)

        # 构造阶段输出的元组
        stage_outputs = (hidden_states, hidden_states_before_downsampling, output_dimensions)

        # 如果需要输出注意力矩阵，则将其添加到阶段输出的元组中
        if output_attentions:
            stage_outputs += layer_outputs[1:]
        # 返回阶段输出的元组
        return stage_outputs


# 从 transformers.models.swin.modeling_swin 模块中复制 ClapAudioPatchMerging 类，并将其中的 Swin 替换为 ClapAudio
class ClapAudioPatchMerging(nn.Module):
    """
    补丁合并层。

    参数:
        input_resolution (`Tuple[int]`):
            输入特征的分辨率。
        dim (`int`):
            输入通道数。
        norm_layer (`nn.Module`, *optional*, 默认为 `nn.LayerNorm`):
            标准化层类。
    """
    def __init__(self, input_resolution: Tuple[int], dim: int, norm_layer: nn.Module = nn.LayerNorm) -> None:
        super().__init__()
        # 初始化函数，接受输入分辨率和维度参数
        self.input_resolution = input_resolution
        self.dim = dim
        # 创建一个线性变换层，将输入特征维度降低为原来的一半
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        # 创建一个归一化层，用于规范化输入特征
        self.norm = norm_layer(4 * dim)

    def maybe_pad(self, input_feature, height, width):
        # 判断是否需要对输入进行填充使其能够整除高度和宽度
        should_pad = (height % 2 == 1) or (width % 2 == 1)
        if should_pad:
            # 计算填充值
            pad_values = (0, 0, 0, width % 2, 0, height % 2)
            # 对输入进行填充
            input_feature = nn.functional.pad(input_feature, pad_values)

        return input_feature

    def forward(self, input_feature: torch.Tensor, input_dimensions: Tuple[int, int]) -> torch.Tensor:
        height, width = input_dimensions
        # `dim` 是高度乘以宽度
        batch_size, dim, num_channels = input_feature.shape

        input_feature = input_feature.view(batch_size, height, width, num_channels)
        # 如果需要，对输入进行填充，使其能够整除高度和宽度
        input_feature = self.maybe_pad(input_feature, height, width)
        # 将输入特征切分成四个部分，每部分对应一个小块的特征
        # [batch_size, height/2, width/2, num_channels]
        input_feature_0 = input_feature[:, 0::2, 0::2, :]
        # [batch_size, height/2, width/2, num_channels]
        input_feature_1 = input_feature[:, 1::2, 0::2, :]
        # [batch_size, height/2, width/2, num_channels]
        input_feature_2 = input_feature[:, 0::2, 1::2, :]
        # [batch_size, height/2, width/2, num_channels]
        input_feature_3 = input_feature[:, 1::2, 1::2, :]
        # 拼接四个部分的特征，以准备进行后续的处理
        input_feature = torch.cat([input_feature_0, input_feature_1, input_feature_2, input_feature_3], -1)
        # 重新调整特征的形状
        input_feature = input_feature.view(batch_size, -1, 4 * num_channels)  # batch_size height/2*width/2 4*C

        # 对输入特征进行归一化处理
        input_feature = self.norm(input_feature)
        # 对输入特征进行维度降低操作
        input_feature = self.reduction(input_feature)

        return input_feature
class ClapAudioEncoder(nn.Module):
    # 定义 ClapAudioEncoder 类，继承自 nn.Module
    def __init__(self, config):
        # 初始化方法
        super().__init__()
        # 调用父类的初始化方法
        self.num_layers = len(config.depths)
        # 获取层数

        self.config = config
        # 保存配置参数
        self.patch_embed = ClapAudioPatchEmbed(config)
        # 创建 ClapAudioPatchEmbed 对象，用于将输入音频转换为图像表示
        self.enable_fusion = config.enable_fusion
        # 是否启用融合操作的标志
        self.patch_stride = self.patch_embed.patch_stride
        # 获取 PatchEmbed 的步幅
        self.spec_size = config.spec_size
        # 获取输入音频的大小
        self.freq_ratio = config.spec_size // config.num_mel_bins
        # 计算频率比率

        self.num_features = int(config.patch_embeds_hidden_size * 2 ** (self.num_layers - 1))
        # 计算特征维度

        drop_path_rate = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]
        # 计算 Drop Path 的比率

        grid_size = self.patch_embed.grid_size
        # 获取 PatchEmbed 的网格大小
        self.input_resolutions = [(grid_size[0] // (2**i), grid_size[1] // (2**i)) for i in range(self.num_layers)]
        # 计算输入分辨率

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
        # 创建 ClapAudioStage 层的列表

        self.gradient_checkpointing = False
        # 是否启用梯度检查点

        self.batch_norm = nn.BatchNorm2d(config.num_mel_bins)
        # 批归一化层
        self.norm = nn.LayerNorm(self.num_features)
        # 层归一化层
        self.depths = config.depths
        # 存储深度参数
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # 自适应平均池化层
    def reshape_mel2img(self, normalized_input_features):
        """
        The input is 4 normalized log mel spectrograms. It is reshape to the common shape of images. Each channel
        should represent 1 of the 4 crops of the spectrogram. For more details, refer to the [`ClapFeatureExtractor`].
        """
        # 获取输入特征的形状信息
        _, _, time_length, freq_length = normalized_input_features.shape

        # 计算图像的宽度和高度
        spec_width = int(self.spec_size * self.freq_ratio)
        spec_heigth = self.spec_size // self.freq_ratio

        # 检查输入特征的大小是否符合要求
        if time_length > spec_width or freq_length > spec_heigth:
            raise ValueError("the wav size should be less than or equal to the swin input size")

        # 避免双三次插值的零误差
        if time_length < spec_width:
            normalized_input_features = nn.functional.interpolate(
                normalized_input_features, (spec_width, freq_length), mode="bicubic", align_corners=True
            )
        if freq_length < spec_heigth:
            normalized_input_features = nn.functional.interpolate(
                normalized_input_features, (time_length, spec_heigth), mode="bicubic", align_corners=True
            )

        # 获取调整后的输入特征的形状信息
        batch, channels, time, freq = normalized_input_features.shape

        # 重塑输入特征的形状
        normalized_input_features = normalized_input_features.reshape(
            batch, channels * self.freq_ratio, time // self.freq_ratio, freq
        )
        normalized_input_features = normalized_input_features.permute(0, 1, 3, 2).contiguous()
        normalized_input_features = normalized_input_features.reshape(
            batch, channels, freq * self.freq_ratio, time // self.freq_ratio
        )

        return normalized_input_features

    def forward(
        self,
        input_features,
        is_longer: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        output_hidden_states_before_downsampling: Optional[bool] = False,
        always_partition: Optional[bool] = False,
        return_dict: Optional[bool] = True,
# 定义 CLAP_START_DOCSTRING 字符串，包含模型的继承信息和参数说明
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

# 定义 CLAP_TEXT_INPUTS_DOCSTRING 字符串，包含文本输入参数说明
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

# 定义 CLAP_AUDIO_INPUTS_DOCSTRING 字符串，包含音频输入参数说明
CLAP_AUDIO_INPUTS_DOCSTRING = r"""
    Args:
        input_features (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            输入音频特征。应该由 [`ClapFeatureExtractor`] 类返回，您也可以从 [`AutoFeatureExtractor`] 获取。有关详细信息，请参见 [`ClapFeatureExtractor.__call__`]。
        is_longer (`torch.FloatTensor`, of shape `(batch_size, 1)`, *optional*):
            音频剪辑是否长于 `max_length`。如果为 `True`，则会启用特征融合以增强特征。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回的张量下的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。有关更多详细信息，请参见返回的张量下的 `hidden_states`。
        return_dict (`bool`, *optional*):
            是否返回一个 [`~utils.ModelOutput`] 而不是一个普通的元组。
"""

CLAP_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            输入序列标记在词汇表中的索引。默认情况下，如果提供填充，则会被忽略。

            可以使用 [`AutoTokenizer`] 获得这些索引。详情请参见 [`PreTrainedTokenizer.encode`] 和
            [`PreTrainedTokenizer.__call__`]。

            [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            避免在填充的标记索引上执行注意力操作的掩码。掩码值选在 `[0, 1]` 范围内：

            - 1 表示**未被掩码**的标记，
            - 0 表示**被掩码**的标记。

            [什么是注意力掩码？](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            输入序列标记在位置嵌入中的位置索引。选在范围 `[0, config.max_position_embeddings - 1]` 内。

            [什么是位置 ID？](../glossary#position-ids)
        input_features (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            输入音频特征。这应该由 [`ClapFeatureExtractor`] 类返回，您也可以从 [`AutoFeatureExtractor`] 获取它。
            详情请参见 [`ClapFeatureExtractor.__call__`]。

        return_loss (`bool`, *optional*):
            是否返回对比损失。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。有关详细信息，请参见返回张量下的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。有关详细信息，请参见返回张量下的 `hidden_states`。
        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""


class ClapProjectionLayer(nn.Module):
    def __init__(self, config: Union[ClapAudioConfig, ClapTextConfig]):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        projection_dim = config.projection_dim

        # 定义线性层
        self.linear1 = nn.Linear(hidden_size, projection_dim)
        # 激活函数
        self.activation = ACT2FN[config.projection_hidden_act]
        # 定义第二个线性层
        self.linear2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, hidden_states):
        # 第一个线性层
        hidden_states = self.linear1(hidden_states)
        # 应用激活函数
        hidden_states = self.activation(hidden_states)
        # 第二个线性层
        hidden_states = self.linear2(hidden_states)
        return hidden_states


# 从 transformers.models.roberta.modeling_roberta.RobertaEmbeddings 复制而来，将 Roberta->ClapText，persistent=False->persistent=True
class ClapTextEmbeddings(nn.Module):
    """
    # BertEmbeddings的变体，对位置嵌入索引进行微小调整。
    
    # 从transformers.models.bert.modeling_bert.BertEmbeddings.__init__中复制而来
    def __init__(self, config):
        # 调用父类构造函数
        super().__init__()
        # 初始化词嵌入层，vocab_size为词汇表大小，hidden_size为隐藏单元大小，padding_idx用于指定填充的token id
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 初始化位置嵌入层，max_position_embeddings为最大位置嵌入大小，hidden_size为隐藏单元大小
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 初始化token类型嵌入层，type_vocab_size为token类型嵌入大小，hidden_size为隐藏单元大小
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
    
        # self.LayerNorm未使用蛇形命名以保持与TensorFlow模型变量名的一致性，并能够加载任何TensorFlow检查点文件
        # 初始化LayerNorm层，隐藏单元大小为config.hidden_size，eps用于防止除零错误
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化dropout层，使用config.hidden_dropout_prob概率进行dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb)在内存中是连续的，并在序列化时导出
        # 根据max_position_embeddings初始化position_ids，用于存储位置id
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=True
        )
        # 初始化token_type_ids，用于存储token类型id，默认为全零向量
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=True
        )
    
        # 复制结束
        # 初始化padding_idx，用于指定填充的token id
        self.padding_idx = config.pad_token_id
        # 初始化位置嵌入层，max_position_embeddings为最大位置嵌入大小，hidden_size为隐藏单元大小，使用padding_idx指定填充的token id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )
    
    # 正向传播函数
    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
        ):
        # 如果位置 id 为 None
        if position_ids is None:
            # 如果输入 id 不为 None
            if input_ids is not None:
                # 从输入的标记 id 创建位置 id。任何填充的标记仍然保持填充状态。
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                # 从输入的嵌入创建位置 id
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        # 如果输入 id 不为 None
        if input_ids is not None:
            # 获取输入 id 的形状
            input_shape = input_ids.size()
        else:
            # 获取输入嵌入的形状，去掉最后一个维度
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度
        seq_length = input_shape[1]

        # 将 token_type_ids 设置为构造函数中注册的缓冲区，其中所有值都为零，通常在自动生成时发生，
        # 注册的缓冲区有助于用户在不传递 token_type_ids 的情况下跟踪模型，解决问题 #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果输入嵌入为 None
        if inputs_embeds is None:
            # 使用 word_embeddings 获取输入 id 的嵌入
            inputs_embeds = self.word_embeddings(input_ids)
        # 获取 token_type_ids 的嵌入
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将输入嵌入和 token_type_ids 的嵌入相加
        embeddings = inputs_embeds + token_type_embeddings
        # 如果位置嵌入类型为 "absolute"
        if self.position_embedding_type == "absolute":
            # 获取位置嵌入
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        # 对 embeddings 进行 LayerNorm
        embeddings = self.LayerNorm(embeddings)
        # 对 embeddings 进行 dropout
        embeddings = self.dropout(embeddings)
        # 返回 embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        # 获取输入嵌入的形状，去掉最后一个维度
        input_shape = inputs_embeds.size()[:-1]
        # 获取序列长度
        sequence_length = input_shape[1]

        # 生成顺序位置 id
        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)
# 从transformers.models.bert.modeling_bert.BertSelfAttention复制并修改为ClapTextSelfAttention类
class ClapTextSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 如果隐藏大小不能被注意力头的数量整除，且config对象没有embedding_size属性，则引发值错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 初始化注意力头的数量和大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化查询、键和值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 初始化dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        # 初始化位置嵌入类型，默认为绝对位置嵌入
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果位置嵌入类型是相对键或相对键-查询，则初始化距离嵌入
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        # 判断是否为解码器
        self.is_decoder = config.is_decoder

    # 将张量转置为注意力分数
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播函数
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
        # 省略了前向传播的具体实现
        pass

# 从transformers.models.bert.modeling_bert.BertSelfOutput复制并修改为ClapTextSelfOutput类
class ClapTextSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化线性层、LayerNorm和dropout
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 线性变换
        hidden_states = self.dense(hidden_states)
        # dropout
        hidden_states = self.dropout(hidden_states)
        # 添加残差连接和LayerNorm
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

# 从transformers.models.bert.modeling_bert.BertAttention复制并修改
# 定义一个自定义的文本注意力模块，继承自 nn.Module
class ClapTextAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 初始化 self 层，使用 ClapTextSelfAttention 模块
        self.self = ClapTextSelfAttention(config, position_embedding_type=position_embedding_type)
        # 初始化 output 层，使用 ClapTextSelfOutput 模块
        self.output = ClapTextSelfOutput(config)
        # 初始化一个空集合用于存储被剪枝的注意力头
        self.pruned_heads = set()

    # 剪枝注意力头
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 找到可剪枝的头和索引
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

    # 前向传播函数
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
        # 使用 self 层进行前向传播
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 使用 output 层处理 self 层的输出和输入 hidden_states
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # 如果需要输出注意力，添加到输出中
        return outputs


# 从 transformers.models.bert.modeling_bert.BertIntermediate 复制而来
# 定义一个自定义的文本中间层模块，继承自 nn.Module
class ClapTextIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个线性层，将隐藏状态的大小转换为中间层的大小
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 根据配置选择激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播函数
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用线性层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 使用激活函数处理线性层的输出
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# 从 transformers.models.bert.modeling_bert.BertOutput 复制而来
# 定义一个自定义的文本输出模块，继承自 nn.Module
class ClapTextOutput(nn.Module):
    # 初始化函数，接受一个配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个全连接层，输入大小为config.intermediate_size，输出大小为config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个LayerNorm层，输入大小为config.hidden_size，eps为config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个Dropout层，概率为config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    # 前向传播函数，接受两个张量参数，返回一个张量
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将hidden_states传入全连接层，得到输出
        hidden_states = self.dense(hidden_states)
        # 对输出进行Dropout操作
        hidden_states = self.dropout(hidden_states)
        # 将Dropout后的输出与input_tensor相加，然后传入LayerNorm层
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回LayerNorm层的输出
        return hidden_states
# 从transformers.models.bert.modeling_bert.BertLayer复制并修改为ClapTextLayer
class ClapTextLayer(nn.Module):
    # 初始化ClapTextLayer类
    def __init__(self, config):
        super().__init__()
        # 定义前向传播中的前馈块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度维度
        self.seq_len_dim = 1
        # 注意力层
        self.attention = ClapTextAttention(config)
        # 是否为解码器
        self.is_decoder = config.is_decoder
        # 是否添加交叉注意力
        self.add_cross_attention = config.add_cross_attention
        # 如果添加交叉注意力
        if self.add_cross_attention:
            # 如果不是解码器则引发值错误
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 添加交叉注意力
            self.crossattention = ClapTextAttention(config, position_embedding_type="absolute")
        # 中间层
        self.intermediate = ClapTextIntermediate(config)
        # 输出层
        self.output = ClapTextOutput(config)

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:  # 函数声明，接受输入参数，返回一个包含 torch.Tensor 的元组
        # 如果过去的键/值缓存不为空，则使用过去的自注意力键/值缓存的前两个元素
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 进行自注意力计算
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]  # 获取自注意力输出

        # 如果是解码器，最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]  # 排除自注意力缓存以外的输出
            present_key_value = self_attention_outputs[-1]  # 获取当前的键/值缓存
        else:
            outputs = self_attention_outputs[1:]  # 如果我们输出注意力权重，则添加自注意力
                                                # 如果不是解码器，自注意力输出从第二个元素开始

        cross_attn_present_key_value = None  # 初始化交叉注意力的当前键/值缓存为 None
        if self.is_decoder and encoder_hidden_states is not None:  # 如果是解码器且有编码器隐藏状态
            if not hasattr(self, "crossattention"):  # 如果没有交叉注意力层，则引发 ValueError
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 如果过去的键/值缓存不为空，则使用过去的交叉注意力键/值缓存的最后两个元素
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 进行交叉注意力计算
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]  # 获取交叉注意力输出
            outputs = outputs + cross_attention_outputs[1:-1]  # 添加交叉注意力到输出

            # 将交叉注意力缓存添加到当前键/值缓存的末尾
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 将注意力输出应用于前向传播的块
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs  # 将层输出添加到输出元组中

        # 如果是解码器，则将注意力键/值作为最后一个输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)  # 将当前键/值缓存添加到输出中

        return outputs  # 返回输出元组

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)  # 应用中间层
        layer_output = self.output(intermediate_output, attention_output)  # 应用输出层
        return layer_output  # 返回层输出
# 从transformers.models.bert.modeling_bert.BertEncoder复制代码，并将Bert->ClapText
class ClapTextEncoder(nn.Module):
    # 初始化函数，接受配置参数config
    def __init__(self, config):
        super().__init__()
        # 保存配置参数
        self.config = config
        # 创建包含多个ClapTextLayer的ModuleList，数量为config中指定的隐藏层数
        self.layer = nn.ModuleList([ClapTextLayer(config) for _ in range(config.num_hidden_layers)])
        # 是否启用梯度检查点，默认为False
        self.gradient_checkpointing = False

    # 前向传播函数
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
    # 定义函数的返回类型，可以是包含 Tensor 的元组或 BaseModelOutputWithPastAndCrossAttentions 类型
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
    
    # 如果 output_hidden_states 为 True，则初始化一个空元组用于存储所有隐藏状态；否则设置为 None
    all_hidden_states = () if output_hidden_states else None
    # 如果 output_attentions 为 True，则初始化一个空元组用于存储所有自注意力机制；否则设置为 None
    all_self_attentions = () if output_attentions else None
    # 如果 output_attentions 为 True 且配置中添加了跨注意力，则初始化一个空元组用于存储所有交叉注意力机制；否则设置为 None
    all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
    
    # 如果启用了梯度检查点且正在训练，则在使用缓存时发出警告，并将 use_cache 设置为 False
    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False
    
    # 如果使用缓存，则初始化一个空元组用于存储下一个解码器缓存；否则设置为 None
    next_decoder_cache = () if use_cache else None
    # 遍历解码器的每一层
    for i, layer_module in enumerate(self.layer):
        # 如果 output_hidden_states 为 True，则将当前隐藏状态添加到所有隐藏状态中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
    
        # 如果提供了头部遮罩，则将其应用于当前层；否则设置为 None
        layer_head_mask = head_mask[i] if head_mask is not None else None
        # 如果提供了过去的键值对，则获取当前层的过去键值对；否则设置为 None
        past_key_value = past_key_values[i] if past_key_values is not None else None
    
        # 如果启用了梯度检查点且正在训练，则使用梯度检查点函数计算当前层的输出
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
        # 如果使用缓存，则将当前层的输出缓存添加到下一个解码器缓存中
        if use_cache:
            next_decoder_cache += (layer_outputs[-1],)
        # 如果输出注意力机制，则将当前层的自注意力机制添加到所有自注意力机制中
        if output_attentions:
            all_self_attentions = all_self_attentions + (layer_outputs[1],)
            # 如果配置中添加了跨注意力，则将当前层的交叉注意力机制添加到所有交叉注意力机制中
            if self.config.add_cross_attention:
                all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
    
    # 如果输出隐藏状态，则将最终隐藏状态添加到所有隐藏状态中
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)
    
    # 如果 return_dict 为 False，则返回一个包含隐藏状态、下一个解码器缓存、所有隐藏状态、所有自注意力机制和所有交叉注意力机制的元组，排除值为 None 的项
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
    # 否则，返回一个 BaseModelOutputWithPastAndCrossAttentions 类型的对象，包含最终隐藏状态、下一个解码器缓存、所有隐藏状态、所有自注意力机制和所有交叉注意力机制
    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=next_decoder_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
        cross_attentions=all_cross_attentions,
    )
# 从 transformers.models.bert.modeling_bert.BertPooler 复制过来的代码，定义了一个名为 ClapTextPooler 的类
class ClapTextPooler(nn.Module):
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 定义一个全连接层，输入和输出维度都是 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义激活函数为双曲正切函数
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过选择第一个 token 对应的隐藏状态来“池化”模型
        first_token_tensor = hidden_states[:, 0]
        # 将选择的隐藏状态通过全连接层得到池化输出
        pooled_output = self.dense(first_token_tensor)
        # 对池化输出应用激活函数
        pooled_output = self.activation(pooled_output)
        # 返回池化输出
        return pooled_output


# 定义了一个名为 ClapPreTrainedModel 的抽象类，用于处理权重初始化、下载和加载预训练模型等
class ClapPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 配置类为 ClapConfig
    config_class = ClapConfig
    # 基础模型前缀为 "clap"
    base_model_prefix = "clap"
    # 不支持梯度检查点
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        """Initialize the weights"""
        # 初始化权重的因子
        factor = self.config.initializer_factor

        if isinstance(module, ClapTextEmbeddings):
            # 如果是 ClapTextEmbeddings 类型的模块，对位置嵌入和标记类型嵌入进行初始化
            module.position_embeddings.weight.data.normal_(mean=0.0, std=factor * 0.02)
            module.token_type_embeddings.weight.data.normal_(mean=0.0, std=factor * 0.02)
        elif isinstance(module, ClapModel):
            # 如果是 ClapModel 类型的模块，对 logit_scale_a 和 logit_scale_t 进行初始化
            nn.init.normal_(module.logit_scale_a, std=factor * 0.02)
            nn.init.normal_(module.logit_scale_t, std=factor * 0.02)
        elif isinstance(module, nn.Embedding):
            # 如果是 nn.Embedding 类型的模块，对权重进行初始化
            module.weight.data.normal_(mean=0.0, std=factor * 0.02)

        elif isinstance(module, nn.LayerNorm):
            # 如果是 nn.LayerNorm 类型的模块，对偏置和权重进行初始化
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, (nn.Conv2d, nn.Linear)):
            # 如果是 nn.Conv2d 或 nn.Linear 类型的模块，对权重进行初始化
            in_proj_std = (self.config.hidden_size**-0.5) * ((2 * self.config.num_hidden_layers) ** -0.5) * factor
            nn.init.normal_(module.weight, std=in_proj_std)
            if module.bias is not None:
                module.bias.data.zero_()


# 定义了一个名为 ClapAudioModel 的类，继承自 ClapPreTrainedModel 类
class ClapAudioModel(ClapPreTrainedModel):
    # 配置类为 ClapAudioConfig
    config_class = ClapAudioConfig
    # 主要输入名称为 "input_features"
    main_input_name = "input_features"

    def __init__(self, config: ClapAudioConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 初始化音频编码器
        self.audio_encoder = ClapAudioEncoder(config)
        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        # 获取输入嵌入层
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
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        """  # 函数定义，指定返回类型为元组或BaseModelOutputWithPooling对象

        # 设置返回字典是否可用，默认根据配置决定
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 设置是否输出注意力权重，默认根据配置决定
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置是否输出隐藏状态，默认根据配置决定
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # 调用音频编码器，返回编码后的特征表示
        return self.audio_encoder(
            input_features=input_features,
            is_longer=is_longer,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
class ClapTextModel(ClapPreTrainedModel):
    """
    
    该模型既可以作为一个编码器（仅具有自注意力），也可以作为解码器，此时在自注意力层之间添加了一层交叉注意力，
    遵循 *Attention is all you need*_ 的架构，作者包括 Ashish Vaswani、Noam Shazeer、Niki Parmar、Jakob Uszkoreit、
    Llion Jones、Aidan N. Gomez、Lukasz Kaiser 和 Illia Polosukhin 在内。

    若要作为解码器，模型需要使用 `is_decoder` 参数初始化为 `True`。若要在 Seq2Seq 模型中使用，模型需要使用 `is_decoder` 参数
    和 `add_cross_attention` 参数都设置为 `True` 进行初始化；此时预期将编码器隐藏状态作为输入传递给前向传播。

    .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762

    """

    config_class = ClapTextConfig

    # 从 transformers.models.bert.modeling_bert.BertModel.__init__ 复制并将 Bert->ClapText
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        # 创建 ClapTextEmbeddings 实例
        self.embeddings = ClapTextEmbeddings(config)
        # 创建 ClapTextEncoder 实例
        self.encoder = ClapTextEncoder(config)

        # 如果 add_pooling_layer 为 True，创建 ClapTextPooler 实例，否则为 None
        self.pooler = ClapTextPooler(config) if add_pooling_layer else None

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 获取输入词嵌入
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        # 设置输入词嵌入
        self.embeddings.word_embeddings = value

    # 从 transformers.models.bert.modeling_bert.BertModel.forward 复制
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
    # 初始化函数，接受一个 ClapConfig 类型的参数，并调用父类的初始化函数
    def __init__(self, config: ClapConfig):
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

        # 将 config.text_config 和 config.audio_config 赋值给局部变量
        text_config = config.text_config
        audio_config = config.audio_config

        # 初始化 logit 缩放参数，并将其转换为可训练的参数
        self.logit_scale_a = nn.Parameter(torch.tensor(math.log(config.logit_scale_init_value)))
        self.logit_scale_t = nn.Parameter(torch.tensor(math.log(config.logit_scale_init_value)))

        # 从配置中获取投影维度
        self.projection_dim = config.projection_dim

        # 初始化文本模型和文本投影层
        self.text_model = ClapTextModel(text_config)
        self.text_projection = ClapProjectionLayer(text_config)

        # 初始化音频模型和音频投影层
        self.audio_model = ClapAudioModel(audio_config)
        self.audio_projection = ClapProjectionLayer(audio_config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 为模型的前向传播函数添加文档字符串，详细说明输入参数和输出
    @add_start_docstrings_to_model_forward(CLAP_TEXT_INPUTS_DOCSTRING)
    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
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

        # 使用文本模型处理输入特征，获取文本输出
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从文本输出中获取池化后的输出，用于投影层
        pooled_output = text_outputs[1] if return_dict is not None else text_outputs.pooler_output
        # 使用文本投影层处理池化输出，得到文本特征
        text_features = self.text_projection(pooled_output)
        # 对文本特征进行标准化
        text_features = F.normalize(text_features, dim=-1)

        # 返回文本特征
        return text_features

    @add_start_docstrings_to_model_forward(CLAP_AUDIO_INPUTS_DOCSTRING)
    def get_audio_features(
        self,
        input_features: Optional[torch.Tensor] = None,
        is_longer: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
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
        # 确定是否输出注意力权重
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 确定是否输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 确定是否以字典形式返回结果
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取音频模型的输出
        audio_outputs = self.audio_model(
            input_features=input_features,
            is_longer=is_longer,
            return_dict=return_dict,
        )

        # 获取音频模型的汇总输出
        pooled_output = audio_outputs[1] if not return_dict else audio_outputs.pooler_output

        # 将汇总输出投影到指定维度的空间
        audio_features = self.audio_projection(pooled_output)
        # 对音频特征进行标准化处理
        audio_features = F.normalize(audio_features, dim=-1)

        # 返回处理后的音频特征
        return audio_features

    # 将注释添加到模型的前向方法上
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
# 为 CLAP 文本模型添加一个在池化输出之上的投影层（线性层）的 CLAP 文本模型
@add_start_docstrings(
    """
    CLAP Text Model with a projection layer on top (a linear layer on top of the pooled output).
    """,
    CLAP_START_DOCSTRING,
)
class ClapTextModelWithProjection(ClapPreTrainedModel):
    # 使用 CLAP 文本配置类
    config_class = ClapTextConfig

    def __init__(self, config: ClapTextConfig):
        super().__init__(config)
        # 初始化 CLAP 文本模型
        self.text_model = ClapTextModel(config)
        # 初始化 CLAP 投影层
        self.text_projection = ClapProjectionLayer(config)
        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.word_embeddings

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.text_model.embeddings.word_embeddings = value

    # 对 CLAP 文本模型进行前向传播
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
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, ClapTextModelWithProjection

        >>> model = ClapTextModelWithProjection.from_pretrained("laion/clap-htsat-unfused")
        >>> tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

        >>> inputs = tokenizer(["a sound of a cat", "a sound of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> text_embeds = outputs.text_embeds
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取文本模型输出
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果不返回字典，则使用 text_outputs[1] 作为池化输出；否则使用 text_outputs.pooler_output
        pooled_output = text_outputs[1] if not return_dict else text_outputs.pooler_output

        # 对池化输出应用文本投影
        text_embeds = self.text_projection(pooled_output)

        # 如果不返回字典，则返回 (text_embeds, text_outputs[0]) + text_outputs[2:] 中非空的项
        if not return_dict:
            outputs = (text_embeds, text_outputs[0]) + text_outputs[2:]
            return tuple(output for output in outputs if output is not None)

        # 如果返回字典，则返回 CLAP 文本模型输出
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
    # 设置配置类
    config_class = ClapAudioConfig
    # 主要输入名称
    main_input_name = "input_features"

    # 初始化方法
    def __init__(self, config: ClapAudioConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建 CLAP 音频模型对象
        self.audio_model = ClapAudioModel(config)
        # 创建 CLAP 投影层对象
        self.audio_projection = ClapProjectionLayer(config)
        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入层的方法
    def get_input_embeddings(self) -> nn.Module:
        # 返回音频编码器的投影层
        return self.audio_model.audio_encoder.patch_embed.proj

    # 重写前向传播方法，并添加文档字符串
    @add_start_docstrings_to_model_forward(CLAP_AUDIO_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ClapAudioModelOutput, config_class=ClapAudioConfig)
    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        is_longer: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ClapAudioModelOutput]:
        r"""
        Returns:

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
        ```"""
        # 检查是否需要返回字典类型的结果
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 检查是否需要输出注意力权重
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 检查是否需要输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # 调用 CLAP 音频模型的前向传播方法
        audio_outputs = self.audio_model(
            input_features=input_features,
            is_longer=is_longer,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 提取池化输出
        pooled_output = audio_outputs[1] if not return_dict else audio_outputs.pooler_output

        # 对池化输出进行投影
        audio_embeds = self.audio_projection(pooled_output)

        # 如果不需要返回字典类型的结果
        if not return_dict:
            # 组装输出元组
            outputs = (audio_embeds, audio_outputs[0]) + audio_outputs[2:]
            # 返回输出元组，过滤掉为 None 的部分
            return tuple(output for output in outputs if output is not None)

        # 返回 CLAP 音频模型输出对象
        return ClapAudioModelOutput(
            audio_embeds=audio_embeds,
            last_hidden_state=audio_outputs.last_hidden_state,
            attentions=audio_outputs.attentions,
            hidden_states=audio_outputs.hidden_states,
        )
```