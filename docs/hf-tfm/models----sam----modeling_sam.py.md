# `.\transformers\models\sam\modeling_sam.py`

```py
# coding=utf-8
# 版权声明与许可证信息

""" PyTorch SAM 模型。"""

# 导入必要的库
import collections
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor, nn

# 导入模型输出相关的类和函数
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 文档中的配置信息
_CONFIG_FOR_DOC = "SamConfig"
_CHECKPOINT_FOR_DOC = "facebook/sam-vit-huge"

# SAM 模型的预训练模型列表
SAM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/sam-vit-huge",
    "facebook/sam-vit-large",
    "facebook/sam-vit-base",
    # 查看所有 SAM 模型，请访问 https://huggingface.co/models?filter=sam
]

# SamVisionEncoderOutput 类，表示 SAM 视觉编码器的输出，包含通过将池化层输出应用于投影层获得的图像嵌入
@dataclass
class SamVisionEncoderOutput(ModelOutput):
    """
    Base class for sam vision model's outputs that also contains image embeddings obtained by applying the projection
    layer to the pooler_output.
"""
    # 该函数用于描述模型的输出,包括图像嵌入、最后一层的隐藏状态、所有层的隐藏状态以及注意力权重
    Args:
        # 图像嵌入,当模型初始化时带有投影层时输出,形状为 (batch_size, output_dim)
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        # 最后一层的隐藏状态,形状为 (batch_size, sequence_length, hidden_size)
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        # 所有层的隐藏状态,当 output_hidden_states=True 时输出,形状为 (batch_size, sequence_length, hidden_size)
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        # 注意力权重,当 output_attentions=True 时输出,形状为 (batch_size, num_heads, sequence_length, sequence_length)
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # 图像嵌入,可选
    image_embeds: Optional[torch.FloatTensor] = None
    # 最后一层的隐藏状态
    last_hidden_state: torch.FloatTensor = None
    # 所有层的隐藏状态,可选
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 注意力权重,可选
    attentions: Optional[Tuple[torch.FloatTensor]] = None
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput
import torch
import torch.nn as nn
from typing import Optional, Tuple

# 定义一个数据类，表示Segment-Anything模型的输出
@dataclass
class SamImageSegmentationOutput(ModelOutput):
    """
    Base class for Segment-Anything model's output

    Args:
        iou_scores (`torch.FloatTensor` of shape `(batch_size, num_masks)`):
            The iou scores of the predicted masks.
            预测掩码的IoU分数。
        pred_masks (`torch.FloatTensor` of shape `(batch_size, num_masks, height, width)`):
            The predicted low resolutions masks. Needs to be post-processed by the processor
            预测的低分辨率掩码。需要由处理器进行后处理。
        vision_hidden_states  (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the vision model at the output of each layer plus the optional initial embedding outputs.
            视觉模型每层输出的隐藏状态以及可选的初始嵌入输出。
        vision_attentions  (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
            注意力softmax后的注意力权重，用于计算自注意力头中的加权平均值。
        mask_decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
            注意力softmax后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    # 定义IoU分数和预测掩码的张量
    iou_scores: torch.FloatTensor = None
    pred_masks: torch.FloatTensor = None
    vision_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    vision_attentions: Optional[Tuple[torch.FloatTensor]] = None
    mask_decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


class SamPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """
    # 初始化方法，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类初始化方法
        super().__init__()
        # 从配置对象中获取图像大小和块大小
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size
        # 如果图像大小和块大小不是可迭代对象，则将它们转换为元组
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        # 计算图像可以划分的块数
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        
        # 将获取的属性赋值给实例变量
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        # 创建一个卷积层，用于将图像块投影到隐藏层
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    # 前向传播方法，接受像素值作为输入
    def forward(self, pixel_values):
        # 获取输入张量的维度信息
        batch_size, num_channels, height, width = pixel_values.shape
        # 如果输入张量的通道数与配置中设置的不匹配，则抛出异常
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 如果输入图像的尺寸与模型规定的尺寸不匹配，则抛出异常
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]}."
            )
        # 将输入张量通过投影层并进行维度置换，得到嵌入表示
        embeddings = self.projection(pixel_values).permute(0, 2, 3, 1)
        # 返回嵌入表示
        return embeddings
# 定义一个SamMLPBlock类，继承自nn.Module
class SamMLPBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义一个线性层，输入维度为config.hidden_size，输出维度为config.mlp_dim
        self.lin1 = nn.Linear(config.hidden_size, config.mlp_dim)
        # 定义另一个线性层，输入维度为config.mlp_dim，输出维度为config.hidden_size
        self.lin2 = nn.Linear(config.mlp_dim, config.hidden_size)
        # 根据配置中的隐藏激活函数选择对应的激活函数
        self.act = ACT2FN[config.hidden_act]

    # 前向传播方法，接收输入hidden_states，返回输出hidden_states
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过第一个线性层处理输入hidden_states
        hidden_states = self.lin1(hidden_states)
        # 通过配置中的隐藏激活函数处理输出hidden_states
        hidden_states = self.act(hidden_states)
        # 通过第二个线性层处理输出hidden_states
        hidden_states = self.lin2(hidden_states)
        # 返回处理后的hidden_states
        return hidden_states


# 从transformers.models.convnext.modeling_convnext.ConvNextLayerNorm复制到SamLayerNorm
class SamLayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height,
    width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width).
    """

    # 初始化方法，接收参数normalized_shape，eps=1e-6，data_format="channels_last"
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        # 定义参数weight为一个可训练参数，形状为normalized_shape，值为1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        # 定义参数bias为一个可训练参数，形状为normalized_shape，值为0
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        # 设置eps值
        self.eps = eps
        # 设置数据格式
        self.data_format = data_format
        # 如果data_format不是"channels_last"或"channels_first"，则抛出NotImplementedError
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data format: {self.data_format}")
        # 设置normalized_shape
        self.normalized_shape = (normalized_shape,)

    # 前向传播方法，接收输入x，返回输出x
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 如果数据格式是channels_last
        if self.data_format == "channels_last":
            # 使用torch���layer_norm函数对输入x进行层归一化处理
            x = torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # 如果数据格式是channels_first
        elif self.data_format == "channels_first":
            # 将输入x的数据类型转换为float类型
            input_dtype = x.dtype
            x = x.float()
            # 计算输入x的均值和方差
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            # 对输入x进行归一化处理
            x = (x - u) / torch.sqrt(s + self.eps)
            # 将x的数据类型重新转换为输入时的数据类型
            x = x.to(dtype=input_dtype)
            # 使用weight和bias对x进行加权和偏置处理
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
        # 返回处理后的x
        return x


# 定义一个SamAttention类，用于SAM的注意力层，可以在将查询、键和值投影到之后缩小嵌入的尺寸
class SamAttention(nn.Module):
    """
    SAM's attention layer that allows for downscaling the size of the embedding after projection to queries, keys, and
    values.
    """
    # 初始化方法，接收配置和下采样率作为参数
    def __init__(self, config, downsample_rate=None):
        # 调用父类的初始化方法
        super().__init__()
        # 从配置中获取隐藏层大小
        self.hidden_size = config.hidden_size

        # 如果未提供下采样率，则使用配置中的注意力下采样率
        downsample_rate = config.attention_downsample_rate if downsample_rate is None else downsample_rate
        # 计算内部维度
        self.internal_dim = config.hidden_size // downsample_rate
        # 获取注意力头的数量
        self.num_attention_heads = config.num_attention_heads
        # 如果内部维度不能被注意力头数量整除，则抛出数值错误
        if self.internal_dim % config.num_attention_heads != 0:
            raise ValueError("num_attention_heads must divide hidden_size.")

        # 初始化 Q、K、V 和输出的线性层
        self.q_proj = nn.Linear(self.hidden_size, self.internal_dim)
        self.k_proj = nn.Linear(self.hidden_size, self.internal_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, self.hidden_size)

    # 将隐藏状态分割成不同的注意力头部分
    def _separate_heads(self, hidden_states: Tensor, num_attention_heads: int) -> Tensor:
        # 获取隐藏状态的形状参数
        batch, point_batch_size, n_tokens, channel = hidden_states.shape
        # 计算每个注意头对应的通道数
        c_per_head = channel // num_attention_heads
        # 重塑隐藏状态
        hidden_states = hidden_states.reshape(batch * point_batch_size, n_tokens, num_attention_heads, c_per_head)
        # 转置隐藏状态
        return hidden_states.transpose(1, 2)

    # 将分开的注意力头重新合并成隐藏状态
    def _recombine_heads(self, hidden_states: Tensor, point_batch_size: int) -> Tensor:
        # 获取隐藏状态的形状参数
        batch, n_heads, n_tokens, c_per_head = hidden_states.shape
        # 转置隐藏状态
        hidden_states = hidden_states.transpose(1, 2)
        # 重塑合并的隐藏状态
        return hidden_states.reshape(batch // point_batch_size, point_batch_size, n_tokens, n_heads * c_per_head)

    # 前向传播方法，接收查询、键、值和注意力相似度作为参数
    def forward(self, query: Tensor, key: Tensor, value: Tensor, attention_similarity: Tensor = None) -> Tensor:
        # 输入投影
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        # 获取点批次大小
        point_batch_size = query.shape[1]
        # 将查询、键、值分隔成注意力头部分
        query = self._separate_heads(query, self.num_attention_heads)
        key = self._separate_heads(key, self.num_attention_heads)
        value = self._separate_heads(value, self.num_attention_heads)

        # 计算注意力权重
        _, _, _, c_per_head = query.shape
        attn = query @ key.permute(0, 1, 3, 2)
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # 如果提供了注意力相似度，则将其加入注意力权重中
        if attention_similarity is not None:
            attn = attn + attention_similarity
            attn = torch.softmax(attn, dim=-1)

        # 获取输出
        out = attn @ value
        out = self._recombine_heads(out, point_batch_size)
        out = self.out_proj(out)

        # 返回输出
        return out
class SamTwoWayAttentionBlock(nn.Module):
    def __init__(self, config, attention_downsample_rate: int = 2, skip_first_layer_pe: bool = False):
        """
        A transformer block with four layers:
            (1) self-attention of sparse inputs (2) cross attention of sparse inputs -> dense inputs (3) mlp block on
            sparse inputs (4) cross attention of dense inputs -> sparse inputs

        Arguments:
            config (`SamMaskDecoderConfig`):
                The configuration file used to instantiate the block
            attention_downsample_rate (*optionalk*, int, defaults to 2):
                The downsample ratio of the block used to reduce the inner dim of the attention.
            skip_first_layer_pe (*optional*, bool, defaults to `False`):
                Whether or not to skip the addition of the query_point_embedding on the first layer.
        """
        super().__init__()

        # 从配置中获取隐藏层大小和层标准化的 epsilon 值
        self.hidden_size = config.hidden_size
        self.layer_norm_eps = config.layer_norm_eps

        # 初始化自注意力层和层标准化
        self.self_attn = SamAttention(config, downsample_rate=1)
        self.layer_norm1 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        # 初始化从标记到图像的交叉注意力层和层标准化
        self.cross_attn_token_to_image = SamAttention(config, downsample_rate=attention_downsample_rate)
        self.layer_norm2 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        # 初始化 MLP 块和层标准化
        self.mlp = SamMLPBlock(config)
        self.layer_norm3 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        # 初始化图像到标记的交叉注意力层和层标准化
        self.layer_norm4 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.cross_attn_image_to_token = SamAttention(config, downsample_rate=attention_downsample_rate)

        # 设置是否跳过第一层的点嵌入添加
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        query_point_embedding: Tensor,
        key_point_embedding: Tensor,
        attention_similarity: Tensor,
        output_attentions: bool = False,
        # Self attention block
        # 如果需要跳过第一层位置编码，则使用 self attention 对查询进行处理
        if self.skip_first_layer_pe:
            queries = self.self_attn(query=queries, key=queries, value=queries)
        else:
            # 否则将查询与查询位置编码相加后，使用 self attention 进行处理，并将结果与原查询相加
            query = queries + query_point_embedding
            attn_out = self.self_attn(query=query, key=query, value=queries)
            queries = queries + attn_out
        # 使用 layer normalization 进行标准化
        queries = self.layer_norm1(queries)

        # Cross attention block, tokens attending to image embedding
        # 将查询与查询位置编码相加后，将键与键位置编码相加后，使用 cross attention 进行处理
        query = queries + query_point_embedding
        key = keys + key_point_embedding
        attn_out = self.cross_attn_token_to_image(
            query=query, key=key, value=keys, attention_similarity=attention_similarity
        )
        queries = queries + attn_out
        # 使用 layer normalization 进行标准化
        queries = self.layer_norm2(queries)

        # MLP block
        # 使用全连接网络进行处理
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.layer_norm3(queries)

        # Cross attention block, image embedding attending to tokens
        # 将查询与查询位置编码相加后，将键与键位置编码相加后，使用 cross attention 进行处理
        query = queries + query_point_embedding
        key = keys + key_point_embedding
        attn_out = self.cross_attn_image_to_token(query=key, key=query, value=queries)
        keys = keys + attn_out
        # 使用 layer normalization 进行标准化
        keys = self.layer_norm4(keys)

        # 组装输出结果
        outputs = (queries, keys)

        # 如果需要输出注意力分布，则将注意力分布添加到输出结果中
        if output_attentions:
            outputs = outputs + (attn_out,)
        else:
            outputs = outputs + (None,)

        return outputs
class SamTwoWayTransformer(nn.Module):
    def __init__(self, config: SamMaskDecoderConfig):
        # 初始化函数，接受一个配置参数
        super().__init__()
        self.config = config

        self.num_hidden_layers = config.num_hidden_layers
        self.layers = nn.ModuleList()

        for i in range(self.num_hidden_layers):
            # 将多个 SamTwoWayAttentionBlock 实例添加到 layers 中
            self.layers.append(SamTwoWayAttentionBlock(config, skip_first_layer_pe=(i == 0)))

        # 创建最终的 attention 层和 layer normalization
        self.final_attn_token_to_image = SamAttention(config)
        self.layer_norm_final_attn = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        point_embeddings: Tensor,
        image_embeddings: Tensor,
        image_positional_embeddings: Tensor,
        attention_similarity: Tensor,
        target_embedding=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        # 前向传播函数，接受多个输入参数，并返回结果
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        all_attentions = ()

        if image_embeddings is None:
            # 如果没有传入图像embedding，则抛出异常
            raise ValueError("You have to specify an image_embedding")

        image_embeddings = image_embeddings.flatten(2).permute(0, 2, 1).unsqueeze(1)
        image_positional_embeddings = image_positional_embeddings.flatten(2).permute(0, 2, 1).unsqueeze(1)

        # 准备查询（queries）
        queries = point_embeddings
        keys = image_embeddings

        # 应用 transformer block 和最终的 layernorm
        for layer in self.layers:
            if target_embedding is not None:
                queries += target_embedding

            # 调用每个 attention block，并收集输出的attention
            queries, keys, attention_outputs = layer(
                queries=queries,
                keys=keys,
                query_point_embedding=point_embeddings,
                key_point_embedding=image_positional_embeddings,
                attention_similarity=attention_similarity,
                output_attentions=output_attentions,
            )

            if output_attentions:
                all_attentions = all_attentions + (attention_outputs,)

        # 应用最终的 attention 层，��点到图像
        query = queries + point_embeddings
        key = keys + image_positional_embeddings

        attn_out = self.final_attn_token_to_image(query=query, key=key, value=keys)

        queries = queries + attn_out
        # 对结果进行 layer normalization
        queries = self.layer_norm_final_attn(queries)
        return queries, keys, all_attentions


class SamFeedForward(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, sigmoid_output: bool = False
        # 初始化函数，接受输入维度、隐藏层维度、输出维度、隐藏层数和是否使用sigmoid输出作为参数
    # 定义一个全连接神经网络层
    def __init__(
        self,
        input_dim, 
        hidden_dim,
        output_dim,
        num_layers,
        sigmoid_output=False
    ):
        # 调用父类的 __init__ 方法
        super().__init__()
        # 设置网络层数
        self.num_layers = num_layers
        # 使用 ReLU 作为激活函数
        self.activation = nn.ReLU()
        # 定义输入到隐藏层的线性变换
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        # 定义隐藏层到输出层的线性变换
        self.proj_out = nn.Linear(hidden_dim, output_dim)
        # 定义中间隐藏层，共 num_layers-2 层
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)])
        # 是否在输出层使用 sigmoid 函数
        self.sigmoid_output = sigmoid_output
    
    # 定义前向传播过程
    def forward(self, hidden_states):
        # 输入通过第一个线性变换和激活函数
        hidden_states = self.proj_in(hidden_states)
        hidden_states = self.activation(hidden_states)
        # 通过中间隐藏层
        for layer in self.layers:
            hidden_states = self.activation(layer(hidden_states))
        # 最后通过输出层的线性变换
        hidden_states = self.proj_out(hidden_states)
        # 如果需要 sigmoid 输出，则应用 sigmoid 函数
        if self.sigmoid_output:
            hidden_states = F.sigmoid(hidden_states)
        # 返回最终输出
        return hidden_states
class SamMaskDecoder(nn.Module):
    def __init__(self, config: SamMaskDecoderConfig):
        super().__init__()

        self.hidden_size = config.hidden_size  # 存储隐藏层大小

        self.num_multimask_outputs = config.num_multimask_outputs  # 存储多掩码输出数量
        self.num_mask_tokens = config.num_multimask_outputs + 1  # 存储掩码标记数量

        self.iou_token = nn.Embedding(1, self.hidden_size)  # 创建 IOU 标记的嵌入层
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, self.hidden_size)  # 创建掩码标记的嵌入层

        self.transformer = SamTwoWayTransformer(config)  # 创建双向转换器

        # 应该为此创建一个新类吗？
        self.upscale_conv1 = nn.ConvTranspose2d(self.hidden_size, self.hidden_size // 4, kernel_size=2, stride=2)  # 创建上采样卷积层1
        self.upscale_conv2 = nn.ConvTranspose2d(self.hidden_size // 4, self.hidden_size // 8, kernel_size=2, stride=2)  # 创建上采样卷积层2
        self.upscale_layer_norm = SamLayerNorm(self.hidden_size // 4, data_format="channels_first")  # 创建上采样层规范化层
        self.activation = nn.GELU()  # 创建 GELU 激活函数

        mlps_list = []
        for _ in range(self.num_mask_tokens):
            mlps_list += [SamFeedForward(self.hidden_size, self.hidden_size, self.hidden_size // 8, 3)]  # 创建多个前馈神经网络
        self.output_hypernetworks_mlps = nn.ModuleList(mlps_list)  # 创建输出超网络的前馈神经网络列表

        self.iou_prediction_head = SamFeedForward(
            self.hidden_size, config.iou_head_hidden_dim, self.num_mask_tokens, config.iou_head_depth
        )  # 创建 IOU 预测头部的前馈神经网络

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_positional_embeddings: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        output_attentions: Optional[bool] = None,
        attention_similarity: torch.Tensor = None,
        target_embedding: torch.Tensor = None,
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config: SamPromptEncoderConfig):
        # 调用父类的初始化函数
        super().__init__()
        # 设置遮罩输入通道数为配置中的四分之一
        self.mask_input_channels = config.mask_input_channels // 4
        # 设置激活函数为配置中指定的激活函数
        self.activation = ACT2FN[config.hidden_act]
        # 创建第一个二维卷积层，输入通道数为1，输出通道数为遮罩输入通道数，卷积核大小为2x2，步长为2
        self.conv1 = nn.Conv2d(1, self.mask_input_channels, kernel_size=2, stride=2)
        # 创建第二个二维卷积层，输入通道数为遮罩输入通道数，输出通道数为配置中的遮罩输入通道数，卷积核大小为2x2，步长为2
        self.conv2 = nn.Conv2d(self.mask_input_channels, config.mask_input_channels, kernel_size=2, stride=2)
        # 创建第三个二维卷积层，输入通道数为配置中的遮罩输入通道数，输出通道数为配置中的隐藏大小，卷积核大小为1x1
        self.conv3 = nn.Conv2d(config.mask_input_channels, config.hidden_size, kernel_size=1)
        # 创建第一个 SAM 归一化层，输入通道数为遮罩输入通道数，epsilon为配置中的层归一化epsilon，数据格式为"channels_first"
        self.layer_norm1 = SamLayerNorm(
            self.mask_input_channels, eps=config.layer_norm_eps, data_format="channels_first"
        )
        # 创建第二个 SAM 归一化层，输入通道数为遮罩输入通道数的四倍，epsilon为配置中的层归一化epsilon，数据格式为"channels_first"
        self.layer_norm2 = SamLayerNorm(
            self.mask_input_channels * 4, eps=config.layer_norm_eps, data_format="channels_first"
        )

    # 前向传播函数，接受遮罩作为输入，返回密集嵌入
    def forward(self, masks):
        # 经过第一个卷积层
        hidden_states = self.conv1(masks)
        # 经过第一个 SAM 归一化层
        hidden_states = self.layer_norm1(hidden_states)
        # 经过激活函数
        hidden_states = self.activation(hidden_states)

        # 经过第二个卷积层
        hidden_states = self.conv2(hidden_states)
        # 经过第二个 SAM 归一化层
        hidden_states = self.layer_norm2(hidden_states)
        # 经过激活函数
        hidden_states = self.activation(hidden_states)
        # 经过第三个卷积层，得到密集嵌入
        dense_embeddings = self.conv3(hidden_states)
        # 返回密集嵌入
        return dense_embeddings
class SamPromptEncoder(nn.Module):
    # 定义 SamPromptEncoder 类，继承自 nn.Module
    def __init__(self, config: SamPromptEncoderConfig, shared_patch_embedding):
        # 初始化方法
        super().__init__()
        # 共享的 patch 嵌入
        self.shared_embedding = shared_patch_embedding
        # 创建 SamMaskEmbedding 对象
        self.mask_embed = SamMaskEmbedding(config)
        # 创建 nn.Embedding 对象，用于非掩码情况下的嵌入
        self.no_mask_embed = nn.Embedding(1, config.hidden_size)

        # 图像嵌入大小
        self.image_embedding_size = (config.image_embedding_size, config.image_embedding_size)
        # 输入图像大小
        self.input_image_size = config.image_size

        # 创建用于嵌入点的 nn.ModuleList 对象
        self.point_embed = nn.ModuleList(
            [nn.Embedding(1, config.hidden_size) for i in range(config.num_point_embeddings)]
        )
        # 隐藏状态的大小
        self.hidden_size = config.hidden_size
        # 创建用于不是点的嵌入的 nn.Embedding 对象
        self.not_a_point_embed = nn.Embedding(1, config.hidden_size)

    def _embed_points(self, points: torch.Tensor, labels: torch.Tensor, pad: bool) -> torch.Tensor:
        # 嵌入点提示
        """Embeds point prompts."""
        # 将点位移到像素中心
        points = points + 0.5
        # 如果需要填充
        if pad:
            # 计算目标点形状和标签形状
            target_point_shape = (points.shape[0], points.shape[1], 1, points.shape[-1])
            target_labels_shape = (points.shape[0], points.shape[1], 1)
            # 创建填充点和标签
            padding_point = torch.zeros(target_point_shape, device=points.device)
            padding_label = -torch.ones(target_labels_shape, device=labels.device)
            # 连接点和标签
            points = torch.cat([points, padding_point], dim=2)
            labels = torch.cat([labels, padding_label], dim=2)
        # 输入形状
        input_shape = (self.input_image_size, self.input_image_size)
        # 点嵌入
        point_embedding = self.shared_embedding(points, input_shape)

        # 需要在 ONNX 导出中使用 torch.where 并扩展标签张量
        point_embedding = torch.where(labels[..., None] == -1, self.not_a_point_embed.weight, point_embedding)

        # 这是 ONNX 导出所需的。dtype、device 需要明确指定，否则 torch.onnx.export 会解释为 double
        point_embedding = torch.where(
            labels[..., None] != -10,
            point_embedding,
            torch.tensor(0.0, dtype=point_embedding.dtype, device=point_embedding.device),
        )

        point_embedding = torch.where(
            (labels == 0)[:, :, :, None],
            point_embedding + self.point_embed[0].weight[None, None, :, :],
            point_embedding,
        )

        point_embedding = torch.where(
            (labels == 1)[:, :, :, None],
            point_embedding + self.point_embed[1].weight[None, None, :, :],
            point_embedding,
        )

        return point_embedding
    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        # 将框坐标向中心像素偏移0.5
        boxes = boxes + 0.5
        # 获取批次大小和框的数量
        batch_size, nb_boxes = boxes.shape[:2]
        # 将框的坐标重新排列成(batch_size, nb_boxes, 2, 2)的形状
        coords = boxes.reshape(batch_size, nb_boxes, 2, 2)
        # 定义输入形状为(input_image_size, input_image_size)
        input_shape = (self.input_image_size, self.input_image_size)
        # 使用共享的嵌入层对角落进行嵌入
        corner_embedding = self.shared_embedding(coords, input_shape)
        # 添加点嵌入权重到相应的角落
        corner_embedding[:, :, 0, :] += self.point_embed[2].weight
        corner_embedding[:, :, 1, :] += self.point_embed[3].weight
        # 返回角落嵌入
        return corner_embedding

    def forward(
        self,
        input_points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        input_labels: Optional[torch.Tensor],
        input_boxes: Optional[torch.Tensor],
        input_masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense embeddings.

        Args:
            points (`torch.Tensor`, *optional*):
                point coordinates and labels to embed.
            boxes (`torch.Tensor`, *optional*):
                boxes to embed
            masks (`torch.Tensor`, *optional*):
                masks to embed
        """
        # 初始化稀疏嵌入为None
        sparse_embeddings = None
        batch_size = 1
        # 获取共享嵌入的设备
        target_device = self.shared_embedding.positional_embedding.device
        # 如果有点的输入
        if input_points is not None:
            # 获取批次大小和点的批次大小
            batch_size, point_batch_size = input_points.shape[:2]
            # 如果未提供标签，则抛出错误
            if input_labels is None:
                raise ValueError("If points are provided, labels must also be provided.")
            # 对点进行嵌入处理
            point_embeddings = self._embed_points(input_points, input_labels, pad=(input_boxes is None))
            sparse_embeddings = point_embeddings
        # 如果有框的输入
        if input_boxes is not None:
            # 获取批次大小
            batch_size = input_boxes.shape[0]
            # 对框进行嵌入处理
            box_embeddings = self._embed_boxes(input_boxes)
            # 如果稀疏嵌入是None，则更新为框嵌入；否则拼接框嵌入到稀疏嵌入
            if sparse_embeddings is None:
                sparse_embeddings = box_embeddings
            else:
                sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=2)
        # 如果有遮罩的输入
        if input_masks is not None:
            # 对遮罩进行嵌入处理
            dense_embeddings = self.mask_embed(input_masks)
        else:
            # 初始化稠密嵌入
            # 从无遮罩嵌入权重重塑为(batch_size, -1, 1, 1)的形状，扩展为适当大小
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                batch_size, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        # 如果稀疏嵌入是None，则初始化为全零张量
        if sparse_embeddings is None:
            sparse_embeddings = torch.zeros((batch_size, 1, 1, self.hidden_size), device=target_device)

        # 返回稀疏嵌入和稠密嵌入
        return sparse_embeddings, dense_embeddings
class SamVisionAttention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(self, config, window_size):
        super().__init__()
        input_size = (
            (config.image_size // config.patch_size, config.image_size // config.patch_size)
            if window_size == 0
            else (window_size, window_size)
        )

        # 设置注意力头的数量
        self.num_attention_heads = config.num_attention_heads
        # 计算每个注意力头的维度
        head_dim = config.hidden_size // config.num_attention_heads
        # 缩放系数，用于缩放注意力分数
        self.scale = head_dim**-0.5
        # 设置注意力机制中的 dropout 率
        self.dropout = config.attention_dropout

        # 定义 Q、K、V 的线性映射层
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=config.qkv_bias)
        # 定义输出投影层
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

        # 是否使用相对位置编码
        self.use_rel_pos = config.use_rel_pos
        if self.use_rel_pos:
            if input_size is None:
                raise ValueError("Input size must be provided if using relative positional encoding.")

            # 初始化相对位置编码
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def get_rel_pos(self, q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
        """
        Get relative positional embeddings according to the relative positions of
            query and key sizes.

        Args:
            q_size (int):
                size of the query.
            k_size (int):
                size of key k.
            rel_pos (`torch.Tensor`):
                relative position embeddings (L, channel).

        Returns:
            Extracted positional embeddings according to relative positions.
        """
        # 计算相对距离的最大值
        max_rel_dist = int(2 * max(q_size, k_size) - 1)
        # 插值处理相对位置编码
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)

        # 如果查询和键的形状不同，通过缩放短的长度来缩放坐标
        q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
        k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
        relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

        return rel_pos_resized[relative_coords.long()]

    def add_decomposed_rel_pos(
        self,
        attn: torch.Tensor,
        query: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        q_size: Tuple[int, int],
        k_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
        https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py

        Args:
            attn (`torch.Tensor`):
                attention map.
            query (`torch.Tensor`):
                query q in the attention layer with shape (batch_size, query_height * query_width, channel).
            rel_pos_h (`torch.Tensor`):
                relative position embeddings (Lh, channel) for height axis.
            rel_pos_w (`torch.Tensor`):
                relative position embeddings (Lw, channel) for width axis.
            q_size (tuple):
                spatial sequence size of query q with (query_height, query_width).
            k_size (tuple):
                spatial sequence size of key k with (key_height, key_width).

        Returns:
            attn (`torch.Tensor`):
                attention map with added relative positional embeddings.
        """
        # 获取 query 的高度和宽度
        query_height, query_width = q_size
        key_height, key_width = k_size
        # 通过相对位置 embeddings 获取高度和宽度对应的值
        relative_position_height = self.get_rel_pos(query_height, key_height, rel_pos_h)
        relative_position_width = self.get_rel_pos(query_width, key_width, rel_pos_w)

        # 获取 query 的维度信息
        batch_size, _, dim = query.shape
        # 修改 query 的形状
        reshaped_query = query.reshape(batch_size, query_height, query_width, dim)
        # 计算高度相对位置项
        rel_h = torch.einsum("bhwc,hkc->bhwk", reshaped_query, relative_position_height)
        # 计算宽度相对位置项
        rel_w = torch.einsum("bhwc,wkc->bhwk", reshaped_query, relative_position_width)
        # 重塑 attention map 的形状
        attn = attn.reshape(batch_size, query_height, query_width, key_height, key_width)
        # 添加高度和宽度相对位置项到 attention map
        attn = attn + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
        # 重新调整 attention map 的形状
        attn = attn.reshape(batch_size, query_height * query_width, key_height * key_width)
        return attn
    # 前向传播函数，用于计算自注意力机制的输出
    def forward(self, hidden_states: torch.Tensor, output_attentions=False) -> torch.Tensor:
        # 获取输入张量的形状信息，batch_size表示批大小，height表示张量高度，width表示张量宽度，_表示通道数
        batch_size, height, width, _ = hidden_states.shape
        # 使用全连接层进行qkv映射，得到qkv张量，其形状为(3, batch_size, nHead, height * width, channel)
        qkv = (
            self.qkv(hidden_states)
            .reshape(batch_size, height * width, 3, self.num_attention_heads, -1)
            .permute(2, 0, 3, 1, 4)
        )
        # 将qkv张量分解为query, key, value张量，形状为(batch_size * nHead, height * width, channel)
        query, key, value = qkv.reshape(3, batch_size * self.num_attention_heads, height * width, -1).unbind(0)

        # 计算注意力权重
        attn_weights = (query * self.scale) @ key.transpose(-2, -1)

        # 如果使用相对位置编码，则在注意力权重中添加分解的相对位置编码
        if self.use_rel_pos:
            attn_weights = self.add_decomposed_rel_pos(
                attn_weights, query, self.rel_pos_h, self.rel_pos_w, (height, width), (height, width)
            )

        # 对注意力权重进行softmax归一化，dim=-1表示对最后一个维度进行操作
        attn_weights = torch.nn.functional.softmax(attn_weights, dtype=torch.float32, dim=-1).to(query.dtype)

        # 对注意力权重进行dropout操作
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # 计算注意力输出，形状为(batch_size, num_attention_heads, height, width, -1)
        attn_output = (attn_probs @ value).reshape(batch_size, self.num_attention_heads, height, width, -1)
        # 将注意力输出的维度顺序调整为(batch_size, height, width, num_attention_heads, -1)，以便后续处理
        attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(batch_size, height, width, -1)

        # 使用全连接层进行投影，将注意力输出映射回原始维度
        attn_output = self.proj(attn_output)

        # 如果需要输出注意力权重，则将注意力输出和注意力权重一并返回，否则只返回注意力输出
        if output_attentions:
            outputs = (attn_output, attn_weights)
        else:
            outputs = (attn_output, None)

        # 返回注意力输出
        return outputs
# SamVisionLayer 类是一个 PyTorch 的 nn.Module，用于实现 Self-Attention Module (SAM) 的视觉层
class SamVisionLayer(nn.Module):
    def __init__(self, config, window_size):
        # 调用父类的构造函数
        super().__init__()
        # 创建一个 LayerNorm 层，用于对输入的 hidden_states 进行归一化
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个 SamVisionAttention 注意力模块
        self.attn = SamVisionAttention(config, window_size)
        # 创建另一个 LayerNorm 层
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个 SamMLPBlock 多层感知机模块
        self.mlp = SamMLPBlock(config)
        # 保存窗口大小
        self.window_size = window_size

    # 将输入的 hidden_states 划分为不重叠的窗口，并对不足窗口大小的部分进行填充
    def window_partition(self, hidden_states: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Args:
            hidden_states (tensor): input tokens with [batch_size, height, width, channel].
            window_size (int): window size.

        Returns:
            windows: windows after partition with [batch_size * num_windows, window_size, window_size, channel].
            (pad_height, pad_width): padded height and width before partition
        """
        # 获取输入 tensor 的形状
        batch_size, height, width, channel = hidden_states.shape

        # 计算需要填充的高度和宽度，使其能被窗口大小整除
        pad_h = (window_size - height % window_size) % window_size
        pad_w = (window_size - width % window_size) % window_size
        # 使用 F.pad 对输入 tensor 进行填充
        hidden_states = F.pad(hidden_states, (0, 0, 0, pad_w, 0, pad_h))
        # 获取填充后的高度和宽度
        pad_height, pad_width = height + pad_h, width + pad_w

        # 将输入 tensor 划分为不重叠的窗口
        hidden_states = hidden_states.reshape(
            batch_size, pad_height // window_size, window_size, pad_width // window_size, window_size, channel
        )
        # 对窗口进行重新排列，得到最终的窗口 tensor
        windows = hidden_states.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(-1, window_size, window_size, channel)
        # 返回窗口 tensor 和填充大小
        return windows, (pad_height, pad_width)

    # 将窗口 tensor 还原为原始的 hidden_states 形状
    def window_unpartition(
        self, windows: torch.Tensor, window_size: int, padding_shape: Tuple[int, int], original_shape: Tuple[int, int]
    
# 定义SamVisionNeck类，继承自nn.Module
class SamVisionNeck(nn.Module):
    # 初始化方法，接受一个SamVisionConfig类型的参数config
    def __init__(self, config: SamVisionConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 将config赋值给self.config
        self.config = config

        # 创建一个卷积层，输入通道为config.hidden_size，输出通道为config.output_channels，卷积核为1x1，无偏置
        self.conv1 = nn.Conv2d(config.hidden_size, config.output_channels, kernel_size=1, bias=False)
        # 创建一个SamLayerNorm层，输入通道为config.output_channels
        self.layer_norm1 = SamLayerNorm(config.output_channels, data_format="channels_first")
        # 创建一个卷积层，输入通道为config.output_channels，输出通道为config.output_channels，卷积核为3x3，填充为1，无偏置
        self.conv2 = nn.Conv2d(config.output_channels, config.output_channels, kernel_size=3, padding=1, bias=False)
        # 创建一个SamLayerNorm层，输入通道为config.output_channels
        self.layer_norm2 = SamLayerNorm(config.output_channels, data_format="channels_first")

    # 前向传播方法，接受hidden_states作为输入
    def forward(self, hidden_states):
        # 调整hidden_states的维度顺序，将通道维放到第二个维度，其余维度顺序前后交换
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        # 经过第一个卷积层和LayerNorm层处理
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.layer_norm1(hidden_states)

        # 经过第二个卷积层和LayerNorm层处理
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.layer_norm2(hidden_states)
        # 返回处理后的hidden_states
        return hidden_states



# 定义SamVisionEncoder类，继承自nn.Module
class SamVisionEncoder(nn.Module):
    # 初始化方法，接受一个SamVisionConfig类型的参数config
    def __init__(self, config: SamVisionConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 将config赋值给self.config
        self.config = config
        # 将config的image_size赋值给self.image_size
        self.image_size = config.image_size

        # 创建一个SamPatchEmbeddings的实例，赋值给self.patch_embed
        self.patch_embed = SamPatchEmbeddings(config)

        # 初始化pos_embed为None
        self.pos_embed = None
        # 如果config.use_abs_pos为True，创建一个形状为(1, image_size//patch_size, image_size//patch_size, hidden_size)的可训练参数
        if config.use_abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1,
                    config.image_size // config.patch_size,
                    config.image_size // config.patch_size,
                    config.hidden_size,
                )
            )

        # 创建一个空的ModuleList对象
        self.layers = nn.ModuleList()
        # 循环config.num_hidden_layers次，创建SamVisionLayer实例，添加到self.layers中
        for i in range(config.num_hidden_layers):
            layer = SamVisionLayer(
                config,
                window_size=config.window_size if i not in config.global_attn_indexes else 0,
            )
            self.layers.append(layer)

        # 创建一个SamVisionNeck的实例，赋值给self.neck
        self.neck = SamVisionNeck(config)

        # 设置gradient_checkpointing为False
        self.gradient_checkpointing = False

    # 获取patch_embed属性
    def get_input_embeddings(self):
        return self.patch_embed

    # 前向传播方法，接受可选的输入参数，并返回结果
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[Tuple, SamVisionEncoderOutput]:
        # 设置output_attentions，默认为self.config.output_attentions
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置output_hidden_states，默认为self.config.output_hidden_states
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置return_dict，默认为self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果pixel_values为None，则抛出ValueError异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 通过patch_embed方法获取hidden_states
        hidden_states = self.patch_embed(pixel_values)
        # 如果存在pos_embed，将其加到hidden_states上
        if self.pos_embed is not None:
            hidden_states = hidden_states + self.pos_embed

        # 初始化保存所有hidden_states和self-attentions的变量
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        # 对每个layer进行迭代
        for i, layer_module in enumerate(self.layers):
            # 如果需要输出hidden_states，则将当前hidden_states加入到all_hidden_states中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 判断是否采用梯度检查点和是否处于训练模式，来确定调用哪个方法获取layer_outputs
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                )
            else:
                layer_outputs = layer_module(hidden_states, output_attentions=output_attentions)

            # 更新hidden_states为当前layer的输出
            hidden_states = layer_outputs[0]

            # 如果需要输出attentions，则将当前layer的attentions加入到all_self_attentions中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果需要输出hidden_states，则将最终的hidden_states加入到all_hidden_states中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 通过neck方法对hidden_states进行处理
        hidden_states = self.neck(hidden_states)

        # 根据return_dict决定返回的内容
        if not return_dict:
            outputs = (hidden_states,)
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if output_attentions:
                outputs = outputs + (all_self_attentions,)
            return outputs

        return SamVisionEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
# 创建一个SamPreTrainedModel类并继承自PreTrainedModel类
class SamPreTrainedModel(PreTrainedModel):
    # 设置config_class属性为SamConfig类
    config_class = SamConfig
    # 设置base_model_prefix属性为"sam"
    base_model_prefix = "sam"
    # 设置main_input_name属性为"pixel_values"
    main_input_name = "pixel_values"

    # 初始化权重的方法
    def _init_weights(self, module):
        # 获取初始范围标准差
        std = self.config.initializer_range
        # 判断module类型，如果是Linear、Conv2d或ConvTranspose2d，则初始化权重和偏置
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        # 判断module类型，如果是Embedding，则初始化权重和填充索引
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

# 以下是SAM_START_DOCSTRING的内容注释，包括了模型的介绍和参数说明
SAM_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`SamConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# SAM_INPUTS_DOCSTRING暂时为空

# 将注释添加到类SamModel中，并包括了模型的介绍和参数说明
@add_start_docstrings(
    "Segment Anything Model (SAM) for generating segmentation masks, given an input image and ",
    " optional 2D location and bounding boxes.",
    SAM_START_DOCSTRING,
)
class SamModel(SamPreTrainedModel):
    # 设置_tied_weights_keys属性为["prompt_encoder.shared_embedding.positional_embedding"]
    _tied_weights_keys = ["prompt_encoder.shared_embedding.positional_embedding"]

    # 初始化方法
    def __init__(self, config):
        super().__init__(config)
        # 初始化共享图像嵌入
        self.shared_image_embedding = SamPositionalEmbedding(config.vision_config)
        # 初始化视觉编码器
        self.vision_encoder = SamVisionEncoder(config.vision_config)
        # 初始化提示编码器
        self.prompt_encoder = SamPromptEncoder(config.prompt_encoder_config, self.shared_image_embedding)
        # 初始化掩码解码器
        self.mask_decoder = SamMaskDecoder(config.mask_decoder_config)

        # 调用私有方法
        self.post_init()

    # 获取输入嵌入的方法
    def get_input_embeddings(self):
        return self.vision_encoder.get_input_embeddings()
    # 获取宽度位置嵌入
    def get_image_wide_positional_embeddings(self):
        # 获取图像嵌入大小
        size = self.config.prompt_encoder_config.image_embedding_size
        # 获取目标设备和数据类型
        target_device = self.shared_image_embedding.positional_embedding.device
        target_dtype = self.shared_image_embedding.positional_embedding.dtype
        # 创建全为1的 size x size 的张量，设备和数据类型与目标一致
        grid = torch.ones((size, size), device=target_device, dtype=target_dtype)
        # 计算纵向位置嵌入
        y_embed = grid.cumsum(dim=0) - 0.5
        # 计算横向位置嵌入
        x_embed = grid.cumsum(dim=1) - 0.5
        # 将位置嵌入标准化到[0, 1]范围
        y_embed = y_embed / size
        x_embed = x_embed / size

        # 通过共享图像嵌入层获取位置嵌入
        positional_embedding = self.shared_image_embedding(torch.stack([x_embed, y_embed], dim=-1))
        # 调整位置嵌入的维度顺序并增加一维，得到 channel x height x width 的结果
        return positional_embedding.permute(2, 0, 1).unsqueeze(0)

    # 获取图像嵌入
    @torch.no_grad()
    def get_image_embeddings(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        通过视觉编码器传递像素值来返回图像嵌入

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                输入像素值
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的注意力张量。
            output_hidden_states (`bool`, *optional*):
                是否返回所有层的隐藏状态。
            return_dict (`bool`, *optional*):
                是否返回 [`~utils.ModelOutput`] 而不是普通元组。
        """
        # 通过视觉编码器获取视觉输出
        vision_output = self.vision_encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取图像嵌入
        image_embeddings = vision_output[0]
        return image_embeddings

    # 获取提示嵌入
    @torch.no_grad()
    def get_prompt_embeddings(
        self,
        input_points: Optional[torch.FloatTensor] = None,
        input_labels: Optional[torch.LongTensor] = None,
        input_boxes: Optional[torch.FloatTensor] = None,
        input_masks: Optional[torch.LongTensor] = None,
    ):
        r"""
        返回通过输入点、标签、框和掩码通过提示编码器获得的提示嵌入。

        Args:
            input_points (`torch.FloatTensor` of shape `(batch_size, point_batch_size, num_points_per_image, 2)`):
                提示编码器的可选输入点。点的填充由处理器自动完成。`point_batch_size` 指的是我们希望模型对每个点预测的掩码数量。模型将总共输出 `point_batch_size` 次 3 个掩码。
            input_labels (`torch.LongTensor` of shape `(batch_size, point_batch_size, num_points_per_image)`):
                提示编码器的可选输入标签。标签的填充由处理器自动完成，或者可以由用户提供。
            input_boxes (`torch.FloatTensor` of shape `(batch_size, num_boxes_per_image, 4)`):
                提示编码器的可选输入框。框的填充由处理器自动完成。用户也可以手动传递输入框。
            input_masks (`torch.LongTensor` of shape `(batch_size, image_size, image_size)`):
                提示编码器的可选输入掩码。
        """
        prompt_output = self.prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
        )
        返回提示输出

    @add_start_docstrings_to_model_forward(SAM_INPUTS_DOCSTRING)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_points: Optional[torch.FloatTensor] = None,
        input_labels: Optional[torch.LongTensor] = None,
        input_boxes: Optional[torch.FloatTensor] = None,
        input_masks: Optional[torch.LongTensor] = None,
        image_embeddings: Optional[torch.FloatTensor] = None,
        multimask_output: bool = True,
        attention_similarity: Optional[torch.FloatTensor] = None,
        target_embedding: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
```