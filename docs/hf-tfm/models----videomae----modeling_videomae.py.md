# `.\models\videomae\modeling_videomae.py`

```
# 设置文件编码为 UTF-8
# 版权声明，指明版权归属及保留所有权利
# 根据 Apache License, Version 2.0 许可证使用本文件
# 除非符合许可证要求，否则不得使用本文件
# 可在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 本软件根据许可证“按原样”提供，无任何明示或暗示的担保或条件
# 请参阅许可证了解具体条款和限制
""" PyTorch VideoMAE (masked autoencoder) model."""

# 导入所需模块和库
import collections.abc  # 导入 collections.abc 模块
import math  # 导入 math 模块
from copy import deepcopy  # 导入 deepcopy 函数
from dataclasses import dataclass  # 导入 dataclass 装饰器
from typing import Optional, Set, Tuple, Union  # 导入类型注解相关的类和装饰器

import numpy as np  # 导入 NumPy 库并命名为 np
import torch  # 导入 PyTorch 库
import torch.utils.checkpoint  # 导入 PyTorch 的 checkpoint 功能
from torch import nn  # 从 PyTorch 导入 nn 模块
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss  # 从 nn 模块导入损失函数类

# 导入 Hugging Face 库中的相关模块和函数
from ...activations import ACT2FN  # 从 activations 模块导入 ACT2FN 函数
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput  # 从 modeling_outputs 导入输出类
from ...modeling_utils import PreTrainedModel  # 从 modeling_utils 导入预训练模型基类
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer  # 导入模型优化相关函数
from ...utils import (  # 导入通用工具函数和类
    ModelOutput,  # 导入 ModelOutput 类
    add_start_docstrings,  # 导入函数，用于向模型方法添加文档字符串
    add_start_docstrings_to_model_forward,  # 导入函数，用于向模型前向方法添加文档字符串
    logging,  # 导入 logging 模块
    replace_return_docstrings,  # 导入函数，用于替换返回文档字符串
)
from ...utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD  # 导入常量
from .configuration_videomae import VideoMAEConfig  # 导入 VideoMAE 模型的配置类


# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 用于文档的配置和检查点信息
_CONFIG_FOR_DOC = "VideoMAEConfig"
_CHECKPOINT_FOR_DOC = "MCG-NJU/videomae-base"

# 预训练模型存档列表
VIDEOMAE_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "MCG-NJU/videomae-base",
    # 可在 https://huggingface.co/models?filter=videomae 查看所有 VideoMAE 模型
]


@dataclass
class VideoMAEDecoderOutput(ModelOutput):
    """
    VideoMAEDecoder 的输出类，可能包含隐藏状态和注意力权重。

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, patch_size ** 2 * num_channels)`):
            像素重构的 logits。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, 返回时 `output_hidden_states=True` 或 `config.output_hidden_states=True`):
            一个元组，包含 `torch.FloatTensor`（嵌入层输出 + 每个层的输出）的形状为 `(batch_size, sequence_length, hidden_size)`。
            模型每一层的隐藏状态以及初始嵌入层的输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, 返回时 `output_attentions=True` 或 `config.output_attentions=True`):
            一个元组，包含 `torch.FloatTensor`（每个层的注意力权重）的形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
            经过注意力 softmax 后的注意力权重，用于计算自注意力头的加权平均值。
    """
    # 定义一个变量 logits，类型为 torch 的 FloatTensor，初始值为 None
    logits: torch.FloatTensor = None
    # 定义一个变量 hidden_states，类型为 torch 的 FloatTensor 元组，可选类型为 None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义一个变量 attentions，类型为 torch 的 FloatTensor 元组，可选类型为 None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
@dataclass
class VideoMAEForPreTrainingOutput(ModelOutput):
    """
    Class for VideoMAEForPreTraining's outputs, with potential hidden states and attentions.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`):
            Pixel reconstruction loss.
        logits (`torch.FloatTensor` of shape `(batch_size, patch_size ** 2 * num_channels)`):
            Pixel reconstruction logits.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid):
    """
    Sinusoid position encoding table.

    Args:
        n_position (int): Number of positions to encode.
        d_hid (int): Hidden dimension size.

    Returns:
        torch.FloatTensor: Sinusoid position encoding table of shape `(1, n_position, d_hid)`.
    """

    # Define a function to compute position-based angles
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    # Create a numpy array for sinusoid table initialization
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    
    # Apply sine and cosine to alternate columns of the sinusoid table
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    # Convert the numpy array to a torch tensor and add a batch dimension
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class VideoMAEEmbeddings(nn.Module):
    """
    Construct the patch and position embeddings for VideoMAE model.

    Args:
        config (object): Configuration object containing model settings.

    Attributes:
        patch_embeddings (VideoMAEPatchEmbeddings): Patch embeddings module.
        num_patches (int): Number of patches in the input.
        position_embeddings (torch.FloatTensor): Sinusoid position embeddings tensor.
        config (object): Configuration object.
    """

    def __init__(self, config):
        super().__init__()

        # Initialize patch embeddings using VideoMAEPatchEmbeddings
        self.patch_embeddings = VideoMAEPatchEmbeddings(config)
        
        # Determine the number of patches from patch embeddings
        self.num_patches = self.patch_embeddings.num_patches
        
        # Initialize fixed sin-cos position embeddings
        self.position_embeddings = get_sinusoid_encoding_table(self.num_patches, config.hidden_size)
        
        # Store the configuration object
        self.config = config
    def forward(self, pixel_values, bool_masked_pos):
        # 创建补丁嵌入
        embeddings = self.patch_embeddings(pixel_values)

        # 添加位置嵌入
        # 将位置嵌入转换为与embeddings相同类型并复制到相同设备上
        embeddings = embeddings + self.position_embeddings.type_as(embeddings).to(embeddings.device).clone().detach()

        # 只保留可见的补丁
        # ~bool_masked_pos 表示可见的补丁
        if bool_masked_pos is not None:
            batch_size, _, num_channels = embeddings.shape
            embeddings = embeddings[~bool_masked_pos]
            embeddings = embeddings.reshape(batch_size, -1, num_channels)

        return embeddings
# 视频到补丁嵌入的模块。将形状为 (batch_size, num_frames, num_channels, height, width) 的视频批次转换为
# 形状为 (batch_size, seq_len, hidden_size) 的张量，以供 Transformer 编码器使用。

class VideoMAEPatchEmbeddings(nn.Module):
    """
    Video to Patch Embedding. This module turns a batch of videos of shape (batch_size, num_frames, num_channels,
    height, width) into a tensor of shape (batch_size, seq_len, hidden_size) to be consumed by a Transformer encoder.

    The seq_len (the number of patches) equals (number of frames // tubelet_size) * (height // patch_size) * (width //
    patch_size).

    """

    def __init__(self, config):
        super().__init__()

        # 从配置中获取各种参数
        image_size = config.image_size
        patch_size = config.patch_size
        num_channels = config.num_channels
        hidden_size = config.hidden_size
        num_frames = config.num_frames
        tubelet_size = config.tubelet_size

        # 如果图像大小和补丁大小不是可迭代对象，则转换为元组
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)

        # 设置类属性
        self.image_size = image_size
        self.patch_size = patch_size
        self.tubelet_size = int(tubelet_size)

        # 计算补丁数量 seq_len，即 patches 的数量
        num_patches = (
            (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        )
        self.num_channels = num_channels
        self.num_patches = num_patches

        # 创建用于将视频像素值映射为补丁嵌入的 3D 卷积层
        self.projection = nn.Conv3d(
            in_channels=num_channels,
            out_channels=hidden_size,
            kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]),
            stride=(self.tubelet_size, patch_size[0], patch_size[1]),
        )

    def forward(self, pixel_values):
        # 获取输入张量的形状信息
        batch_size, num_frames, num_channels, height, width = pixel_values.shape

        # 检查通道数是否与配置中的一致
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )

        # 检查输入图像尺寸是否与配置中的一致
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )

        # 将像素值排列为 (batch_size, num_channels, num_frames, height, width)
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)

        # 通过投影层将像素值映射为补丁嵌入，并进行扁平化和转置以适应 Transformer 的输入要求
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)

        return embeddings
    def __init__(self, config: VideoMAEConfig) -> None:
        super().__init__()
        # 检查隐藏大小是否是注意力头数的倍数，且未定义嵌入大小
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            # 如果不是，则引发值错误异常
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        # 初始化注意力头数和每个注意力头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化查询、键、值的线性变换层
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=False)

        # 如果配置指定了 QKV 的偏置，则初始化偏置参数
        if config.qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(self.all_head_size))
            self.v_bias = nn.Parameter(torch.zeros(self.all_head_size))
        else:
            self.q_bias = None
            self.v_bias = None

        # 初始化注意力概率的 dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # 重塑张量 x 的形状以适应注意力分数计算所需的维度顺序
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ):
        # 正向传播函数定义
        # 定义函数签名和返回类型注解，可以返回包含 torch.Tensor 的元组
        ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 如果存在查询偏置 self.q_bias，则创建一个与 self.v_bias 相同形状的零张量 k_bias
        k_bias = torch.zeros_like(self.v_bias, requires_grad=False) if self.q_bias is not None else None
        # 计算键 keys，使用线性变换将 hidden_states 与 self.key.weight 相乘并加上偏置 k_bias
        keys = nn.functional.linear(input=hidden_states, weight=self.key.weight, bias=k_bias)
        # 计算值 values，使用线性变换将 hidden_states 与 self.value.weight 相乘并加上偏置 self.v_bias
        values = nn.functional.linear(input=hidden_states, weight=self.value.weight, bias=self.v_bias)
        # 计算查询 queries，使用线性变换将 hidden_states 与 self.query.weight 相乘并加上偏置 self.q_bias
        queries = nn.functional.linear(input=hidden_states, weight=self.query.weight, bias=self.q_bias)

        # 将 keys、values、queries 转换为多头注意力的格式
        key_layer = self.transpose_for_scores(keys)
        value_layer = self.transpose_for_scores(values)
        query_layer = self.transpose_for_scores(queries)

        # 计算注意力分数，即查询与键的点积
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # 对注意力分数进行缩放，以提高数值稳定性
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 对注意力分数进行 softmax 归一化，得到注意力概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 使用 dropout 随机丢弃一些注意力概率，以防止过拟合
        attention_probs = self.dropout(attention_probs)

        # 如果存在 head_mask，则将注意力概率与 head_mask 相乘，实现注意力头的屏蔽
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文向量，即注意力概率与值的加权和
        context_layer = torch.matmul(attention_probs, value_layer)

        # 将上下文向量进行维度重排，以符合模型输出的形状
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # 根据输出设置，返回上下文向量及注意力概率，或仅返回上下文向量
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
# Copied from transformers.models.vit.modeling_vit.ViTSelfOutput with ViT->VideoMAE
class VideoMAESelfOutput(nn.Module):
    """
    The residual connection is defined in VideoMAELayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: VideoMAEConfig) -> None:
        super().__init__()
        # 定义一个全连接层，将输入的隐藏状态转换为相同维度的输出
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义一个 dropout 层，用于随机断开神经元，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用全连接层处理输入的隐藏状态
        hidden_states = self.dense(hidden_states)
        # 对处理后的隐藏状态应用 dropout
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTAttention with ViT->VideoMAE
class VideoMAEAttention(nn.Module):
    def __init__(self, config: VideoMAEConfig) -> None:
        super().__init__()
        # 创建一个 VideoMAESelfAttention 实例，用于注意力机制
        self.attention = VideoMAESelfAttention(config)
        # 创建一个 VideoMAESelfOutput 实例，用于处理注意力输出
        self.output = VideoMAESelfOutput(config)
        # 存储需要剪枝的注意力头信息的集合
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        # 寻找可剪枝的注意力头和相应的索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储剪枝后的头信息
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 使用注意力层处理隐藏状态，可能输出注意力权重
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        # 使用输出层处理注意力层的输出和输入的隐藏状态
        attention_output = self.output(self_outputs[0], hidden_states)

        # 如果需要输出注意力权重，则将其添加到输出中
        outputs = (attention_output,) + self_outputs[1:]  # 如果输出注意力权重，则添加到输出中
        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTIntermediate ViT->VideoMAE
class VideoMAEIntermediate(nn.Module):
    def __init__(self, config: VideoMAEConfig) -> None:
        super().__init__()
        # 创建一个线性层，将输入隐藏状态转换为中间隐藏层的维度
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果配置中隐藏激活函数为字符串，则选择相应的激活函数；否则使用给定的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
    # 对输入的隐藏状态进行前向传播
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用全连接层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的结果应用激活函数（可能是ReLU等）
        hidden_states = self.intermediate_act_fn(hidden_states)

        # 返回处理后的隐藏状态
        return hidden_states
# Copied from transformers.models.vit.modeling_vit.ViTOutput ViT->VideoMAE
class VideoMAEOutput(nn.Module):
    def __init__(self, config: VideoMAEConfig) -> None:
        super().__init__()
        # 初始化一个全连接层，将输入特征大小转换为隐藏大小
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 初始化一个dropout层，用于在训练过程中随机置零输入张量的部分元素，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将输入的隐藏状态传入全连接层
        hidden_states = self.dense(hidden_states)
        # 对全连接层的输出进行dropout操作
        hidden_states = self.dropout(hidden_states)

        # 将dropout后的输出与输入张量相加，实现残差连接
        hidden_states = hidden_states + input_tensor

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTLayer with ViT->VideoMAE
class VideoMAELayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: VideoMAEConfig) -> None:
        super().__init__()
        # 定义用于分块前馈的块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 定义序列长度维度
        self.seq_len_dim = 1
        # 初始化注意力机制模块
        self.attention = VideoMAEAttention(config)
        # 初始化中间层模块
        self.intermediate = VideoMAEIntermediate(config)
        # 初始化输出层模块
        self.output = VideoMAEOutput(config)
        # 初始化一个LayerNorm层，用于对隐藏状态进行归一化处理
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 在VideoMAE中，先对隐藏状态进行LayerNorm处理，然后应用自注意力机制
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # 如果需要输出注意力权重，则将其添加到输出中

        # 第一个残差连接
        hidden_states = attention_output + hidden_states

        # 在VideoMAE中，还需在自注意力后再次应用LayerNorm
        layer_output = self.layernorm_after(hidden_states)
        # 经过中间层处理
        layer_output = self.intermediate(layer_output)

        # 第二个残差连接在这里实现
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTEncoder with ViT->VideoMAE
class VideoMAEEncoder(nn.Module):
    def __init__(self, config: VideoMAEConfig) -> None:
        super().__init__()
        self.config = config
        # 初始化一个由VideoMAELayer组成的模块列表，每个VideoMAELayer对应一个隐藏层
        self.layer = nn.ModuleList([VideoMAELayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
    # 定义一个方法，用于前向传播（推理阶段）的操作，输入参数包括隐藏状态、头部掩码、是否输出注意力权重、是否输出每层隐藏状态、是否返回字典形式的输出
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        # 如果需要输出每层隐藏状态，则初始化一个空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力权重，则初始化一个空元组
        all_self_attentions = () if output_attentions else None

        # 遍历每个层次的 Transformer 层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出每层隐藏状态，在 all_hidden_states 中添加当前隐藏状态
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 如果启用了梯度检查点技术并且当前处于训练模式
            if self.gradient_checkpointing and self.training:
                # 使用梯度检查点函数来调用当前层，并传入相应的参数
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                # 否则直接调用当前层的 __call__ 方法，传入隐藏状态、头部掩码和是否输出注意力权重
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            # 更新隐藏状态为当前层的输出的第一个元素（通常是最终的隐藏状态）
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力权重，在 all_self_attentions 中添加当前层的注意力权重
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果需要输出每层隐藏状态，在 all_hidden_states 中添加最终的隐藏状态
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要返回字典形式的输出，则返回一个元组，过滤掉为 None 的部分
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 否则返回一个 BaseModelOutput 对象，包括最终的隐藏状态、所有隐藏状态和所有注意力权重
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
@add_start_docstrings(
    "The bare VideoMAE Model transformer outputting raw hidden-states without any specific head on top.",
    VIDEOMAE_START_DOCSTRING,
)

模型类的装饰器，用于为 `VideoMAEModel` 添加文档字符串，并且包括了模型的描述信息和参数说明。


class VideoMAEModel(VideoMAEPreTrainedModel):

定义了 `VideoMAEModel` 类，它继承自 `VideoMAEPreTrainedModel` 类，是视频多模态自编码器（VideoMAE）的模型类。


VIDEOMAE_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`VideoMAEConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

`VIDEOMAE_START_DOCSTRING` 是一个原始字符串，用于描述 `VideoMAEModel` 类的基本信息和参数说明。它介绍了模型是如何作为 PyTorch 的 `torch.nn.Module` 子类来使用的，并提供了初始化参数的说明。


VIDEOMAE_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`VideoMAEImageProcessor.__call__`] for details.

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

`VIDEOMAE_INPUTS_DOCSTRING` 是一个原始字符串，用于描述 `VideoMAEModel` 类的输入参数。它详细说明了模型接受的各种输入参数，包括像素值、头部掩码、是否返回注意力张量和隐藏状态等。

这些注释和文档字符串为 `VideoMAEModel` 类提供了清晰的描述和参数说明，帮助用户了解如何使用和配置该模型。
    # 初始化函数，接受配置参数并调用父类的初始化方法
    def __init__(self, config):
        super().__init__(config)
        # 将配置参数保存在实例变量中
        self.config = config

        # 创建视频嵌入对象
        self.embeddings = VideoMAEEmbeddings(config)
        # 创建视频编码器对象
        self.encoder = VideoMAEEncoder(config)

        # 根据配置决定是否使用层归一化，如果使用平均池化则不需要层归一化
        if config.use_mean_pooling:
            self.layernorm = None
        else:
            # 初始化层归一化对象
            self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 初始化权重并进行最终处理
        self.post_init()

    # 返回输入嵌入对象的方法
    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    # 剪枝模型中注意力头的方法
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 遍历需要剪枝的层和对应的注意力头
        for layer, heads in heads_to_prune.items():
            # 调用编码器对象的指定层的注意力机制对象进行注意力头的剪枝操作
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 模型前向传播方法，用于处理视频输入和其他参数，返回模型输出
    @add_start_docstrings_to_model_forward(VIDEOMAE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 定义 VideoMAEDecoder 类，继承自 nn.Module，用于解码视频MAE模型
class VideoMAEDecoder(nn.Module):
    # 初始化方法
    def __init__(self, config, num_patches):
        super().__init__()

        # 计算解码器输出标签数目
        decoder_num_labels = config.num_channels * config.tubelet_size * config.patch_size**2

        # 深拷贝配置对象，设置解码器配置参数
        decoder_config = deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size

        # 创建解码器层列表，每一层使用 VideoMAELayer 类初始化
        self.decoder_layers = nn.ModuleList(
            [VideoMAELayer(decoder_config) for _ in range(config.decoder_num_hidden_layers)]
        )

        # 设置层归一化对象
        self.norm = nn.LayerNorm(config.decoder_hidden_size)

        # 根据解码器输出标签数目确定头部连接层，如果为零则使用恒等映射
        self.head = (
            nn.Linear(config.decoder_hidden_size, decoder_num_labels) if decoder_num_labels > 0 else nn.Identity()
        )

        # 是否使用梯度检查点技术，默认关闭
        self.gradient_checkpointing = False
        self.config = config

    # 前向传播方法
    def forward(
        self,
        hidden_states,
        return_token_num,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # 初始化存储所有隐藏状态和注意力分数的元组，根据输出标志初始化为 None 或空元组
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # 遍历所有解码器层进行前向传播
        for i, layer_module in enumerate(self.decoder_layers):
            # 如果需要输出隐藏状态，则将当前隐藏状态加入到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果启用梯度检查点技术并且在训练阶段，则使用 _gradient_checkpointing_func 函数调用
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    None,
                    output_attentions,
                )
            else:
                # 否则正常调用解码器层的前向传播方法
                layer_outputs = layer_module(hidden_states, head_mask=None, output_attentions=output_attentions)

            # 更新隐藏状态为解码器层的输出的第一个元素
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力分数，则将当前层的注意力分数加入到 all_self_attentions 中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 最后一层解码器的隐藏状态加入到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果 return_token_num 大于 0，则截取隐藏状态的后 return_token_num 个片段
        if return_token_num > 0:
            hidden_states = hidden_states[:, -return_token_num:]

        # 对最终隐藏状态进行归一化处理
        hidden_states = self.norm(hidden_states)

        # 使用头部连接层计算最终的 logits
        logits = self.head(hidden_states)

        # 如果 return_dict 为 False，则返回包含 logits、all_hidden_states 和 all_self_attentions 的元组
        if not return_dict:
            return tuple(v for v in [logits, all_hidden_states, all_self_attentions] if v is not None)
        # 否则返回 VideoMAEDecoderOutput 对象，包含 logits、all_hidden_states 和 all_self_attentions
        return VideoMAEDecoderOutput(logits=logits, hidden_states=all_hidden_states, attentions=all_self_attentions)


# 使用 add_start_docstrings 装饰器为 VideoMAEForPreTraining 类添加文档字符串
@add_start_docstrings(
    "The VideoMAE Model transformer with the decoder on top for self-supervised pre-training.",
    VIDEOMAE_START_DOCSTRING,
)
class VideoMAEForPreTraining(VideoMAEPreTrainedModel):
    # 类定义部分省略
    # 初始化方法，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法，传递配置对象作为参数
        super().__init__(config)
        # 将配置对象保存在实例变量中
        self.config = config

        # 创建 VideoMAEModel 对象并保存在实例变量中
        self.videomae = VideoMAEModel(config)

        # 创建一个线性层，用于编码器到解码器的映射，不使用偏置项
        self.encoder_to_decoder = nn.Linear(config.hidden_size, config.decoder_hidden_size, bias=False)
        # 创建一个可学习的参数，用于表示掩码的标记
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_hidden_size))
        
        # 使用 sinusoid 编码表生成位置嵌入
        self.position_embeddings = get_sinusoid_encoding_table(
            self.videomae.embeddings.num_patches, config.decoder_hidden_size
        )

        # 创建 VideoMAEDecoder 对象，传递配置对象和图像片段数量作为参数
        self.decoder = VideoMAEDecoder(config, num_patches=self.videomae.embeddings.num_patches)

        # 调用后初始化方法，执行权重初始化和最终处理操作
        self.post_init()

    # 将输入的视频像素值和掩码的位置信息作为输入，执行前向传播操作
    @add_start_docstrings_to_model_forward(VIDEOMAE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=VideoMAEForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        bool_masked_pos: torch.BoolTensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 定义 VideoMAEForVideoClassification 类，继承自 VideoMAEPreTrainedModel 类，用于视频分类任务的模型转换器
@add_start_docstrings(
    """VideoMAE Model transformer with a video classification head on top (a linear layer on top of the average pooled hidden
    states of all tokens) e.g. for ImageNet.""",
    VIDEOMAE_START_DOCSTRING,
)
class VideoMAEForVideoClassification(VideoMAEPreTrainedModel):
    def __init__(self, config):
        # 调用父类 VideoMAEPreTrainedModel 的初始化方法
        super().__init__(config)

        # 设定标签数量
        self.num_labels = config.num_labels
        # 初始化 VideoMAEModel 模型
        self.videomae = VideoMAEModel(config)

        # 分类器头部
        # 如果 config.use_mean_pooling 为 True，则使用 LayerNorm 对象进行归一化处理
        self.fc_norm = nn.LayerNorm(config.hidden_size) if config.use_mean_pooling else None
        # 根据标签数量初始化线性分类器或恒等映射
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # 初始化权重并进行最终处理
        self.post_init()

    # 定义前向传播函数
    @add_start_docstrings_to_model_forward(VIDEOMAE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ImageClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
```