# `.\models\sam\modeling_sam.py`

```
# coding=utf-8
# 设置编码方式为 UTF-8

# 版权声明及许可证，声明代码版权及使用许可
# Copyright 2023 The Meta AI Authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch SAM model."""
# PyTorch SAM 模型的定义和实现

import collections  # 导入 collections 模块，用于高效数据容器类型
import math  # 导入 math 模块，提供数学运算函数
from dataclasses import dataclass  # 导入 dataclass 类装饰器，用于定义数据类
from typing import Dict, List, Optional, Tuple, Union  # 导入类型提示相关的库

import numpy as np  # 导入 NumPy 库，用于数值计算
import torch  # 导入 PyTorch 库，进行深度学习模型的构建和训练
import torch.nn.functional as F  # 导入 PyTorch 中的函数模块
import torch.utils.checkpoint  # 导入 PyTorch 的 checkpoint 模块，用于内存优化

from torch import Tensor, nn  # 导入 PyTorch 的张量类和神经网络模块

from ...activations import ACT2FN  # 导入激活函数
from ...modeling_outputs import BaseModelOutput  # 导入基础模型输出类
from ...modeling_utils import PreTrainedModel  # 导入预训练模型基类
from ...utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging  # 导入工具函数和日志模块
from .configuration_sam import SamConfig, SamMaskDecoderConfig, SamPromptEncoderConfig, SamVisionConfig  # 导入 SAM 模型的配置类

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 以下是文档化的字符串，用于生成文档

_CONFIG_FOR_DOC = "SamConfig"  # 配置文档化字符串，指定 SAM 的配置类
_CHECKPOINT_FOR_DOC = "facebook/sam-vit-huge"  # 检查点文档化字符串，指定 SAM 模型的预训练检查点

# SAM 模型的预训练模型存档列表
SAM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/sam-vit-huge",
    "facebook/sam-vit-large",
    "facebook/sam-vit-base",
    # 查看所有 SAM 模型的列表可访问 https://huggingface.co/models?filter=sam
]


@dataclass
class SamVisionEncoderOutput(ModelOutput):
    """
    Base class for sam vision model's outputs that also contains image embeddings obtained by applying the projection
    layer to the pooler_output.
    """
    # SAM 视觉编码器输出的基类，同时包含通过将投影层应用于池化输出获得的图像嵌入。
    # 可选参数：模型输出的图像嵌入向量，形状为(batch_size, output_dim)。仅在模型初始化时设置了 `with_projection=True` 时返回。
    image_embeds: Optional[torch.FloatTensor] = None
    
    # 必需参数：模型最后一层的隐藏状态输出，形状为(batch_size, sequence_length, hidden_size)。
    last_hidden_state: torch.FloatTensor = None
    
    # 可选参数：模型的隐藏状态输出，是一个元组，包含模型每一层的隐藏状态输出。当 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回。
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    
    # 可选参数：模型的注意力权重输出，是一个元组，包含每个注意力头的注意力权重。当 `output_attentions=True` 或 `config.output_attentions=True` 时返回。
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
@dataclass
class SamImageSegmentationOutput(ModelOutput):
    """
    Base class for Segment-Anything model's output

    Args:
        iou_scores (`torch.FloatTensor` of shape `(batch_size, num_masks)`):
            The iou scores of the predicted masks.
        pred_masks (`torch.FloatTensor` of shape `(batch_size, num_masks, height, width)`):
            The predicted low resolutions masks. Needs to be post-processed by the processor
        vision_hidden_states  (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the vision model at the output of each layer plus the optional initial embedding outputs.
        vision_attentions  (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        mask_decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    iou_scores: torch.FloatTensor = None
    pred_masks: torch.FloatTensor = None
    vision_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    vision_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    mask_decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None



class SamPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """
    # 初始化函数，用于初始化类实例
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 从配置对象中获取图像大小和patch大小
        image_size, patch_size = config.image_size, config.patch_size
        # 从配置对象中获取通道数和隐藏层大小
        num_channels, hidden_size = config.num_channels, config.hidden_size
        # 如果图像大小不是可迭代对象，则将其转换为元组
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        # 如果patch大小不是可迭代对象，则将其转换为元组
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        # 计算图像被分成的patch数量
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        
        # 将计算得到的各个属性赋值给实例变量
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        
        # 创建卷积层，用于投影输入像素值到隐藏表示空间
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    # 前向传播函数，接收像素值作为输入，返回嵌入表示
    def forward(self, pixel_values):
        # 获取输入张量的维度信息
        batch_size, num_channels, height, width = pixel_values.shape
        
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
        
        # 将输入图像通过投影层并进行维度转置，得到嵌入表示
        embeddings = self.projection(pixel_values).permute(0, 2, 3, 1)
        
        # 返回嵌入表示
        return embeddings
# 定义一个用于SAM模型中MLP块的类
class SamMLPBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个线性层，将输入维度调整为mlp_dim
        self.lin1 = nn.Linear(config.hidden_size, config.mlp_dim)
        # 创建另一个线性层，将mlp_dim维度调整回hidden_size
        self.lin2 = nn.Linear(config.mlp_dim, config.hidden_size)
        # 选择激活函数，根据配置选择相应的激活函数
        self.act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 第一个线性层的前向传播
        hidden_states = self.lin1(hidden_states)
        # 应用选择的激活函数
        hidden_states = self.act(hidden_states)
        # 第二个线性层的前向传播
        hidden_states = self.lin2(hidden_states)
        return hidden_states


# 从transformers.models.convnext.modeling_convnext.ConvNextLayerNorm复制并修改为SamLayerNorm
class SamLayerNorm(nn.Module):
    r"""支持两种数据格式（channels_last或channels_first）的LayerNorm。
    channels_last对应输入形状为(batch_size, height, width, channels)，而channels_first对应输入形状为(batch_size, channels, height, width)。
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        # 初始化可学习的权重和偏置参数
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        # 如果数据格式不是channels_last或channels_first，则抛出异常
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data format: {self.data_format}")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            # 对输入进行layer normalization，使用学习的权重和偏置参数
            x = torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # 如果数据格式为channels_first，则对输入进行自定义的layer normalization
            input_dtype = x.dtype
            x = x.float()
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = x.to(dtype=input_dtype)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class SamAttention(nn.Module):
    """
    SAM的注意力层，允许在将查询、键和值投影后缩小嵌入的大小。
    """
    # 初始化函数，接受配置参数和降采样率作为可选参数
    def __init__(self, config, downsample_rate=None):
        super().__init__()
        # 设置隐藏层大小为配置中的隐藏层大小
        self.hidden_size = config.hidden_size

        # 如果没有指定降采样率，则使用配置中的注意力降采样率
        downsample_rate = config.attention_downsample_rate if downsample_rate is None else downsample_rate

        # 根据降采样率计算内部维度
        self.internal_dim = config.hidden_size // downsample_rate
        self.num_attention_heads = config.num_attention_heads

        # 检查内部维度是否可以整除注意力头数，否则抛出数值错误
        if self.internal_dim % config.num_attention_heads != 0:
            raise ValueError("num_attention_heads must divide hidden_size.")

        # 初始化查询、键、值、输出的线性投影层
        self.q_proj = nn.Linear(self.hidden_size, self.internal_dim)
        self.k_proj = nn.Linear(self.hidden_size, self.internal_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, self.hidden_size)

    # 将隐藏状态按注意力头数分离的函数
    def _separate_heads(self, hidden_states: Tensor, num_attention_heads: int) -> Tensor:
        batch, point_batch_size, n_tokens, channel = hidden_states.shape
        c_per_head = channel // num_attention_heads
        # 重塑张量形状以便每个注意力头独立操作
        hidden_states = hidden_states.reshape(batch * point_batch_size, n_tokens, num_attention_heads, c_per_head)
        return hidden_states.transpose(1, 2)

    # 将分离的注意力头重新组合为隐藏状态的函数
    def _recombine_heads(self, hidden_states: Tensor, point_batch_size: int) -> Tensor:
        batch, n_heads, n_tokens, c_per_head = hidden_states.shape
        # 调整张量形状以将注意力头合并回原始形式
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states.reshape(batch // point_batch_size, point_batch_size, n_tokens, n_heads * c_per_head)

    # 前向传播函数，接受查询、键、值张量，可选注意力相似性张量，并返回输出张量
    def forward(self, query: Tensor, key: Tensor, value: Tensor, attention_similarity: Tensor = None) -> Tensor:
        # 对输入进行投影
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        # 获取点批次大小
        point_batch_size = query.shape[1]

        # 将查询、键、值张量分离成注意力头
        query = self._separate_heads(query, self.num_attention_heads)
        key = self._separate_heads(key, self.num_attention_heads)
        value = self._separate_heads(value, self.num_attention_heads)

        # 计算注意力权重
        _, _, _, c_per_head = query.shape
        attn = query @ key.permute(0, 1, 3, 2)  # batch_size * point_batch_size x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # 如果提供了注意力相似性张量，则加到注意力权重上
        if attention_similarity is not None:
            attn = attn + attention_similarity
            attn = torch.softmax(attn, dim=-1)

        # 计算输出
        out = attn @ value
        out = self._recombine_heads(out, point_batch_size)
        out = self.out_proj(out)

        return out
class SamTwoWayAttentionBlock(nn.Module):
    def __init__(self, config, attention_downsample_rate: int = 2, skip_first_layer_pe: bool = False):
        """
        A transformer block with four layers:
            (1) self-attention of sparse inputs
            (2) cross attention of sparse inputs -> dense inputs
            (3) MLP block on sparse inputs
            (4) cross attention of dense inputs -> sparse inputs

        Arguments:
            config (`SamMaskDecoderConfig`):
                The configuration file used to instantiate the block
            attention_downsample_rate (*optionalk*, int, defaults to 2):
                The downsample ratio of the block used to reduce the inner dim of the attention.
            skip_first_layer_pe (*optional*, bool, defaults to `False`):
                Whether or not to skip the addition of the query_point_embedding on the first layer.
        """
        super().__init__()

        # Initialize hidden size and layer normalization epsilon from configuration
        self.hidden_size = config.hidden_size
        self.layer_norm_eps = config.layer_norm_eps

        # Self-attention layer for sparse inputs
        self.self_attn = SamAttention(config, downsample_rate=1)
        self.layer_norm1 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        # Cross-attention from token to image (sparse to dense) inputs
        self.cross_attn_token_to_image = SamAttention(config, downsample_rate=attention_downsample_rate)
        self.layer_norm2 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        # MLP block on sparse inputs
        self.mlp = SamMLPBlock(config)
        self.layer_norm3 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        # Layer normalization before cross-attention from image to token (dense to sparse)
        self.layer_norm4 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.cross_attn_image_to_token = SamAttention(config, downsample_rate=attention_downsample_rate)

        # Option to skip adding query_point_embedding in the first layer
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        query_point_embedding: Tensor,
        key_point_embedding: Tensor,
        attention_similarity: Tensor,
        output_attentions: bool = False,
    ):
        # Self attention block
        if self.skip_first_layer_pe:
            # 如果需要跳过第一个自注意力层，则使用 queries 对自注意力进行处理
            queries = self.self_attn(query=queries, key=queries, value=queries)
        else:
            # 否则，将 query_point_embedding 添加到 queries 中，然后进行自注意力计算
            query = queries + query_point_embedding
            attn_out = self.self_attn(query=query, key=query, value=queries)
            queries = queries + attn_out
        # 对 queries 进行 Layer Normalization 处理
        queries = self.layer_norm1(queries)

        # Cross attention block, tokens attending to image embedding
        # 将 query_point_embedding 添加到 queries，将 key_point_embedding 添加到 keys
        query = queries + query_point_embedding
        key = keys + key_point_embedding

        # 使用 cross_attn_token_to_image 方法进行跨注意力计算，将结果添加到 queries 中
        attn_out = self.cross_attn_token_to_image(
            query=query, key=key, value=keys, attention_similarity=attention_similarity
        )
        queries = queries + attn_out

        # 对 queries 进行 Layer Normalization 处理
        queries = self.layer_norm2(queries)

        # MLP block
        # 使用 MLP 模块处理 queries
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        # 对 queries 进行 Layer Normalization 处理
        queries = self.layer_norm3(queries)

        # Cross attention block, image embedding attending to tokens
        # 将 query_point_embedding 添加到 queries，将 key_point_embedding 添加到 keys
        query = queries + query_point_embedding
        key = keys + key_point_embedding

        # 使用 cross_attn_image_to_token 方法进行跨注意力计算，将结果添加到 keys 中
        attn_out = self.cross_attn_image_to_token(query=key, key=query, value=queries)
        keys = keys + attn_out

        # 对 keys 进行 Layer Normalization 处理
        keys = self.layer_norm4(keys)

        # 输出为 (queries, keys) 元组
        outputs = (queries, keys)

        # 如果需要输出注意力权重，则将注意力权重添加到输出元组中
        if output_attentions:
            outputs = outputs + (attn_out,)
        else:
            outputs = outputs + (None,)

        # 返回最终的输出元组
        return outputs
# 定义一个双向转换器模型，继承自 nn.Module
class SamTwoWayTransformer(nn.Module):
    # 初始化函数，接收一个 SamMaskDecoderConfig 类型的配置对象作为参数
    def __init__(self, config: SamMaskDecoderConfig):
        super().__init__()
        # 保存配置对象
        self.config = config

        # 从配置中获取隐藏层数量
        self.num_hidden_layers = config.num_hidden_layers
        # 初始化一个模块列表用于保存多个双向注意力块
        self.layers = nn.ModuleList()

        # 根据隐藏层数量循环创建双向注意力块并添加到模块列表中
        for i in range(self.num_hidden_layers):
            self.layers.append(SamTwoWayAttentionBlock(config, skip_first_layer_pe=(i == 0)))

        # 创建最终的注意力层对象和对应的 LayerNorm 层
        self.final_attn_token_to_image = SamAttention(config)
        self.layer_norm_final_attn = nn.LayerNorm(config.hidden_size)

    # 前向传播函数
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
        # 如果未指定 output_attentions，则使用配置中的设定
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定 output_hidden_states，则使用配置中的设定
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定 return_dict，则使用配置中的设定
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 用于保存所有注意力值的元组
        all_attentions = ()

        # 如果 image_embeddings 为 None，则抛出 ValueError 异常
        if image_embeddings is None:
            raise ValueError("You have to specify an image_embedding")

        # 对 image_embeddings 和 image_positional_embeddings 进行形状变换和排列
        image_embeddings = image_embeddings.flatten(2).permute(0, 2, 1).unsqueeze(1)
        image_positional_embeddings = image_positional_embeddings.flatten(2).permute(0, 2, 1).unsqueeze(1)

        # 准备查询向量
        queries = point_embeddings
        keys = image_embeddings

        # 对每个双向注意力块执行变换操作并应用最终的 LayerNorm
        for layer in self.layers:
            # 如果存在 target_embedding，则将其加到 queries 中
            if target_embedding is not None:
                queries += target_embedding

            # 调用当前层的 forward 方法进行注意力计算
            queries, keys, attention_outputs = layer(
                queries=queries,
                keys=keys,
                query_point_embedding=point_embeddings,
                key_point_embedding=image_positional_embeddings,
                attention_similarity=attention_similarity,
                output_attentions=output_attentions,
            )

            # 如果 output_attentions 为 True，则将当前层的 attention_outputs 添加到 all_attentions 中
            if output_attentions:
                all_attentions = all_attentions + (attention_outputs,)

        # 应用从点到图像的最终注意力层
        query = queries + point_embeddings
        key = keys + image_positional_embeddings

        # 调用最终的注意力层进行计算
        attn_out = self.final_attn_token_to_image(query=query, key=key, value=keys)

        # 将计算得到的 attn_out 加到 queries 中，并应用 LayerNorm
        queries = queries + attn_out
        queries = self.layer_norm_final_attn(queries)

        # 返回 queries, keys 和 all_attentions（如果有）
        return queries, keys, all_attentions
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 设置神经网络的层数
        self.num_layers = num_layers
        # 指定激活函数为ReLU
        self.activation = nn.ReLU()
        # 创建输入层到隐藏层的线性映射
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        # 创建隐藏层到输出层的线性映射
        self.proj_out = nn.Linear(hidden_dim, output_dim)
        # 使用ModuleList创建隐藏层的线性映射列表
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)])
        # 根据需要设置输出是否经过sigmoid函数
        self.sigmoid_output = sigmoid_output

    def forward(self, hidden_states):
        # 输入数据经过输入层到隐藏层的线性映射
        hidden_states = self.proj_in(hidden_states)
        # 经过ReLU激活函数处理隐藏层的输出
        hidden_states = self.activation(hidden_states)
        # 遍历隐藏层列表，每一层经过线性映射和ReLU激活函数
        for layer in self.layers:
            hidden_states = self.activation(layer(hidden_states))

        # 经过隐藏层到输出层的线性映射
        hidden_states = self.proj_out(hidden_states)
        # 如果需要，对输出进行sigmoid函数处理
        if self.sigmoid_output:
            hidden_states = F.sigmoid(hidden_states)
        # 返回神经网络的输出
        return hidden_states
class SamMaskDecoder(nn.Module):
    def __init__(self, config: SamMaskDecoderConfig):
        super().__init__()

        self.hidden_size = config.hidden_size  # 从配置中获取隐藏层大小

        self.num_multimask_outputs = config.num_multimask_outputs  # 多重遮罩输出的数量
        self.num_mask_tokens = config.num_multimask_outputs + 1  # 遮罩标记的数量，包括一个IOU标记

        self.iou_token = nn.Embedding(1, self.hidden_size)  # 创建一个大小为1的嵌入层，用于IOU标记
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, self.hidden_size)  # 创建一个嵌入层，用于所有遮罩标记

        self.transformer = SamTwoWayTransformer(config)  # 创建一个SamTwoWayTransformer对象

        # 创建上采样卷积层和归一化层
        self.upscale_conv1 = nn.ConvTranspose2d(self.hidden_size, self.hidden_size // 4, kernel_size=2, stride=2)
        self.upscale_conv2 = nn.ConvTranspose2d(self.hidden_size // 4, self.hidden_size // 8, kernel_size=2, stride=2)
        self.upscale_layer_norm = SamLayerNorm(self.hidden_size // 4, data_format="channels_first")
        self.activation = nn.GELU()  # GELU激活函数

        mlps_list = []
        for _ in range(self.num_mask_tokens):
            mlps_list += [SamFeedForward(self.hidden_size, self.hidden_size, self.hidden_size // 8, 3)]
        self.output_hypernetworks_mlps = nn.ModuleList(mlps_list)  # 创建一个包含多个SamFeedForward层的模块列表

        self.iou_prediction_head = SamFeedForward(
            self.hidden_size, config.iou_head_hidden_dim, self.num_mask_tokens, config.iou_head_depth
        )  # 创建一个SamFeedForward对象，用于IOU预测头部

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
    ):
        # 此处应为前向传播方法，接收各种输入张量并进行模型的前向运算，但未提供具体实现，无法详细注释
        pass  # 占位符，表示此处未实现具体逻辑


class SamPositionalEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scale = config.hidden_size // 2  # 缩放因子为隐藏层大小的一半
        self.register_buffer("positional_embedding", self.scale * torch.randn((2, config.num_pos_feats)))  # 注册位置嵌入的缓冲区

    def forward(self, input_coords, input_shape=None):
        """Positionally encode points that are normalized to [0,1]."""
        coordinates = input_coords.clone()

        if input_shape is not None:
            coordinates[:, :, :, 0] = coordinates[:, :, :, 0] / input_shape[1]  # 归一化x坐标
            coordinates[:, :, :, 1] = coordinates[:, :, :, 1] / input_shape[0]  # 归一化y坐标

        # 假设坐标位于[0, 1]^2区域并具有d_1 x ... x d_n x 2形状
        coordinates = 2 * coordinates - 1  # 映射到[-1, 1]区间
        coordinates = coordinates.to(self.positional_embedding.dtype)  # 将坐标转换为位置嵌入的数据类型
        coordinates = coordinates @ self.positional_embedding  # 点乘位置嵌入
        coordinates = 2 * np.pi * coordinates  # 缩放角度
        # 输出d_1 x ... x d_n x 通道形状
        return torch.cat([torch.sin(coordinates), torch.cos(coordinates)], dim=-1)  # 返回正弦和余弦编码的拼接


class SamMaskEmbedding(nn.Module):
    # 此处未提供具体代码实现，无法详细注释
    pass  # 占位符，表示此处未实现具体逻辑
    # 初始化方法，接受一个配置对象作为参数
    def __init__(self, config: SamPromptEncoderConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 计算输入通道数的四分之一，并赋值给实例变量
        self.mask_input_channels = config.mask_input_channels // 4
        # 根据配置中的激活函数名称，获取对应的激活函数，并赋值给实例变量
        self.activation = ACT2FN[config.hidden_act]
        # 创建第一个二维卷积层，输入通道为1，输出通道为四分之一的输入通道数，核大小为2x2，步长为2
        self.conv1 = nn.Conv2d(1, self.mask_input_channels, kernel_size=2, stride=2)
        # 创建第二个二维卷积层，输入通道数为四分之一的输入通道数，输出通道数为配置中的输入通道数，核大小为2x2，步长为2
        self.conv2 = nn.Conv2d(self.mask_input_channels, config.mask_input_channels, kernel_size=2, stride=2)
        # 创建第三个二维卷积层，输入通道数为配置中的输入通道数，输出通道数为配置中的隐藏大小，核大小为1x1
        self.conv3 = nn.Conv2d(config.mask_input_channels, config.hidden_size, kernel_size=1)
        # 创建第一个 SAM 层归一化实例，输入通道数为四分之一的输入通道数，epsilon 使用配置中的值，数据格式为"channels_first"
        self.layer_norm1 = SamLayerNorm(
            self.mask_input_channels, eps=config.layer_norm_eps, data_format="channels_first"
        )
        # 创建第二个 SAM 层归一化实例，输入通道数为四倍的输入通道数，epsilon 使用配置中的值，数据格式为"channels_first"
        self.layer_norm2 = SamLayerNorm(
            self.mask_input_channels * 4, eps=config.layer_norm_eps, data_format="channels_first"
        )

    # 前向传播方法，接收掩码作为输入，返回密集嵌入向量
    def forward(self, masks):
        # 第一次卷积操作，将掩码传入第一个卷积层，得到隐藏状态
        hidden_states = self.conv1(masks)
        # 对第一个卷积层的输出进行 SAM 层归一化
        hidden_states = self.layer_norm1(hidden_states)
        # 对归一化后的输出应用激活函数
        hidden_states = self.activation(hidden_states)

        # 第二次卷积操作，将上一步的输出传入第二个卷积层，得到隐藏状态
        hidden_states = self.conv2(hidden_states)
        # 对第二个卷积层的输出进行 SAM 层归一化
        hidden_states = self.layer_norm2(hidden_states)
        # 对归一化后的输出应用激活函数
        hidden_states = self.activation(hidden_states)

        # 第三次卷积操作，将上一步的输出传入第三个卷积层，得到密集嵌入向量
        dense_embeddings = self.conv3(hidden_states)
        # 返回密集嵌入向量作为前向传播的结果
        return dense_embeddings
# 定义 SamPromptEncoder 类，继承自 nn.Module，用于处理 SamPrompt 编码器相关操作
class SamPromptEncoder(nn.Module):
    # 初始化方法，接受 SamPromptEncoderConfig 类型的 config 和共享的 patch embedding
    def __init__(self, config: SamPromptEncoderConfig, shared_patch_embedding):
        super().__init__()
        # 共享的 patch embedding
        self.shared_embedding = shared_patch_embedding
        # 创建 SamMaskEmbedding 对象，用于处理 mask 相关操作
        self.mask_embed = SamMaskEmbedding(config)
        # 创建一个只包含一个元素的 nn.Embedding 对象，用于处理没有 mask 的情况
        self.no_mask_embed = nn.Embedding(1, config.hidden_size)

        # 设置图像嵌入大小和输入图像大小
        self.image_embedding_size = (config.image_embedding_size, config.image_embedding_size)
        self.input_image_size = config.image_size

        # 创建一个 nn.ModuleList，包含多个 nn.Embedding 对象，用于处理点嵌入
        self.point_embed = nn.ModuleList(
            [nn.Embedding(1, config.hidden_size) for i in range(config.num_point_embeddings)]
        )
        # 隐藏状态的大小
        self.hidden_size = config.hidden_size
        # 创建一个只包含一个元素的 nn.Embedding 对象，用于处理不是点的情况
        self.not_a_point_embed = nn.Embedding(1, config.hidden_size)

    # 内部方法，用于嵌入点提示
    def _embed_points(self, points: torch.Tensor, labels: torch.Tensor, pad: bool) -> torch.Tensor:
        """Embeds point prompts."""
        # 将点位移 0.5，以将其移动到像素中心
        points = points + 0.5
        # 如果需要填充
        if pad:
            # 创建目标点形状和标签形状的张量
            target_point_shape = (points.shape[0], points.shape[1], 1, points.shape[-1])
            target_labels_shape = (points.shape[0], points.shape[1], 1)
            # 创建填充点和标签的零张量和负一标签
            padding_point = torch.zeros(target_point_shape, device=points.device)
            padding_label = -torch.ones(target_labels_shape, device=labels.device)
            # 在维度 2 上拼接点和标签
            points = torch.cat([points, padding_point], dim=2)
            labels = torch.cat([labels, padding_label], dim=2)
        # 输入形状为 (self.input_image_size, self.input_image_size)
        input_shape = (self.input_image_size, self.input_image_size)
        # 使用共享的嵌入嵌入点
        point_embedding = self.shared_embedding(points, input_shape)

        # 根据标签是否为 -1，选择不是点的嵌入或点的嵌入
        point_embedding = torch.where(labels[..., None] == -1, self.not_a_point_embed.weight, point_embedding)

        # 对于 ONNX 导出，需要使用 torch.where 扩展标签张量
        point_embedding = torch.where(
            labels[..., None] != -10,
            point_embedding,
            torch.tensor(0.0, dtype=point_embedding.dtype, device=point_embedding.device),
        )

        # 根据标签是否为 0，加上第一个点嵌入的权重
        point_embedding = torch.where(
            (labels == 0)[:, :, :, None],
            point_embedding + self.point_embed[0].weight[None, None, :, :],
            point_embedding,
        )

        # 根据标签是否为 1，加上第二个点嵌入的权重
        point_embedding = torch.where(
            (labels == 1)[:, :, :, None],
            point_embedding + self.point_embed[1].weight[None, None, :, :],
            point_embedding,
        )

        # 返回点嵌入结果
        return point_embedding
    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        # 将框的坐标加上0.5，以将坐标中心移至像素中心
        boxes = boxes + 0.5  # Shift to center of pixel
        batch_size, nb_boxes = boxes.shape[:2]
        # 将框的坐标重塑为(batch_size, nb_boxes, 2, 2)的形状
        coords = boxes.reshape(batch_size, nb_boxes, 2, 2)
        # 设置输入形状为(self.input_image_size, self.input_image_size)
        input_shape = (self.input_image_size, self.input_image_size)
        # 使用共享的嵌入层来嵌入角点的坐标
        corner_embedding = self.shared_embedding(coords, input_shape)
        # 将角点嵌入矩阵的第一个维度加上self.point_embed[2].weight
        corner_embedding[:, :, 0, :] += self.point_embed[2].weight
        # 将角点嵌入矩阵的第二个维度加上self.point_embed[3].weight
        corner_embedding[:, :, 1, :] += self.point_embed[3].weight
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
        sparse_embeddings = None
        batch_size = 1
        # 确定目标设备为self.shared_embedding.positional_embedding的设备
        target_device = self.shared_embedding.positional_embedding.device
        if input_points is not None:
            batch_size, point_batch_size = input_points.shape[:2]
            # 如果提供了points但未提供labels，则抛出异常
            if input_labels is None:
                raise ValueError("If points are provided, labels must also be provided.")
            # 使用_embed_points方法嵌入points的坐标和标签
            point_embeddings = self._embed_points(input_points, input_labels, pad=(input_boxes is None))
            sparse_embeddings = point_embeddings
        if input_boxes is not None:
            batch_size = input_boxes.shape[0]
            # 使用_embed_boxes方法嵌入boxes
            box_embeddings = self._embed_boxes(input_boxes)
            if sparse_embeddings is None:
                sparse_embeddings = box_embeddings
            else:
                sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=2)
        if input_masks is not None:
            # 使用mask_embed方法嵌入masks
            dense_embeddings = self.mask_embed(input_masks)
        else:
            # 使用no_mask_embed的权重初始化dense_embeddings
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                batch_size, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        # 如果sparse_embeddings仍为None，则初始化为全零张量
        if sparse_embeddings is None:
            sparse_embeddings = torch.zeros((batch_size, 1, 1, self.hidden_size), device=target_device)

        return sparse_embeddings, dense_embeddings
        """
        Add decomposed relative positional embeddings to the attention scores.

        Args:
            attn (`torch.Tensor`):
                Attention scores between query and key.
            query (`torch.Tensor`):
                Query tensor.
            rel_pos_h (`torch.Tensor`):
                Relative positional embeddings along height.
            rel_pos_w (`torch.Tensor`):
                Relative positional embeddings along width.
            q_size (`Tuple[int, int]`):
                Size of the query tensor (height, width).
            k_size (`Tuple[int, int]`):
                Size of the key tensor (height, width).

        Returns:
            `torch.Tensor`: Attention scores modified by relative positional embeddings.
        """

        max_rel_dist = int(2 * max(q_size[0], k_size[0]) - 1)

        # Interpolate relative position embeddings
        rel_pos_h_resized = F.interpolate(rel_pos_h.unsqueeze(0), size=max_rel_dist, mode="linear")
        rel_pos_w_resized = F.interpolate(rel_pos_w.unsqueeze(0), size=max_rel_dist, mode="linear")

        rel_pos_h_resized = rel_pos_h_resized.squeeze(0)
        rel_pos_w_resized = rel_pos_w_resized.squeeze(0)

        # Scale coordinates with maximum length if query and key sizes differ
        q_coords = torch.arange(q_size[0]).unsqueeze(1) * max(k_size[0] / q_size[0], 1.0)
        k_coords = torch.arange(k_size[0]).unsqueeze(0) * max(q_size[0] / k_size[0], 1.0)
        relative_coords_h = (q_coords - k_coords) + (k_size[0] - 1) * max(q_size[0] / k_size[0], 1.0)

        q_coords = torch.arange(q_size[1]).unsqueeze(1) * max(k_size[1] / q_size[1], 1.0)
        k_coords = torch.arange(k_size[1]).unsqueeze(0) * max(q_size[1] / k_size[1], 1.0)
        relative_coords_w = (q_coords - k_coords) + (k_size[1] - 1) * max(q_size[1] / k_size[1], 1.0)

        # Gather relative positional embeddings
        rel_pos_h = rel_pos_h_resized[relative_coords_h.long()]
        rel_pos_w = rel_pos_w_resized[relative_coords_w.long()]

        # Combine relative positional embeddings
        rel_pos = rel_pos_h + rel_pos_w.unsqueeze(0)

        # Reshape and expand relative positional embeddings
        rel_pos = rel_pos.unsqueeze(0).expand(attn.size(0), -1, -1)

        # Add relative positional embeddings to attention scores
        attn = attn + rel_pos

        return attn
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
        # 解包查询大小和键大小
        query_height, query_width = q_size
        key_height, key_width = k_size
        
        # 获取相对位置编码矩阵的高度和宽度方向上的影响
        relative_position_height = self.get_rel_pos(query_height, key_height, rel_pos_h)
        relative_position_width = self.get_rel_pos(query_width, key_width, rel_pos_w)

        # 获取查询的批次大小、通道数和维度
        batch_size, _, dim = query.shape
        
        # 重塑查询张量为四维张量，以便进行后续的张量乘积操作
        reshaped_query = query.reshape(batch_size, query_height, query_width, dim)
        
        # 计算相对位置编码对高度和宽度的影响
        rel_h = torch.einsum("bhwc,hkc->bhwk", reshaped_query, relative_position_height)
        rel_w = torch.einsum("bhwc,wkc->bhwk", reshaped_query, relative_position_width)
        
        # 重塑注意力图张量以便添加相对位置编码
        attn = attn.reshape(batch_size, query_height, query_width, key_height, key_width)
        attn = attn + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
        
        # 将注意力图重新展平为原始形状并返回
        attn = attn.reshape(batch_size, query_height * query_width, key_height * key_width)
        return attn
    def forward(self, hidden_states: torch.Tensor, output_attentions=False) -> torch.Tensor:
        # 获取隐藏状态的维度信息
        batch_size, height, width, _ = hidden_states.shape
        
        # 使用 qkv 网络处理隐藏状态，生成 qkv 张量，形状为 (3, batch_size, nHead, height * width, channel)
        qkv = (
            self.qkv(hidden_states)
            .reshape(batch_size, height * width, 3, self.num_attention_heads, -1)
            .permute(2, 0, 3, 1, 4)
        )
        
        # 将 qkv 张量按照 q, k, v 分开，形状为 (batch_size * nHead, height * width, channel)
        query, key, value = qkv.reshape(3, batch_size * self.num_attention_heads, height * width, -1).unbind(0)
        
        # 计算注意力权重，使用注意力机制中的 query 和 key 进行点积
        attn_weights = (query * self.scale) @ key.transpose(-2, -1)

        # 如果使用相对位置编码，则添加分解的相对位置信息
        if self.use_rel_pos:
            attn_weights = self.add_decomposed_rel_pos(
                attn_weights, query, self.rel_pos_h, self.rel_pos_w, (height, width), (height, width)
            )
        
        # 对注意力权重进行 softmax 归一化处理
        attn_weights = torch.nn.functional.softmax(attn_weights, dtype=torch.float32, dim=-1).to(query.dtype)
        
        # 对注意力权重进行 dropout 处理
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # 计算注意力输出，将注意力概率与 value 进行加权求和
        attn_output = (attn_probs @ value).reshape(batch_size, self.num_attention_heads, height, width, -1)
        attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(batch_size, height, width, -1)
        
        # 使用投影层进行输出转换
        attn_output = self.proj(attn_output)
        
        # 如果需要输出注意力权重，则将其包含在输出中
        if output_attentions:
            outputs = (attn_output, attn_weights)
        else:
            outputs = (attn_output, None)
        
        return outputs
    def __init__(self, config, window_size):
        """
        Initialize the SamVisionLayer module.

        Args:
            config (object): Configuration object containing parameters like hidden_size and layer_norm_eps.
            window_size (int): Size of the sliding window to partition the input.
        """
        # Call the parent class constructor to initialize it
        super().__init__()
        
        # Layer normalization applied to the input tensor
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Self-attention mechanism tailored for vision tasks
        self.attn = SamVisionAttention(config, window_size)
        
        # Layer normalization applied after self-attention
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Multilayer perceptron block for further processing
        self.mlp = SamMLPBlock(config)
        
        # Store the window size for later use
        self.window_size = window_size

    def window_partition(self, hidden_states: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Partition input tensor into non-overlapping windows with padding if necessary.

        Args:
            hidden_states (torch.Tensor): Input tensor with shape [batch_size, height, width, channel].
            window_size (int): Size of the window.

        Returns:
            windows (torch.Tensor): Tensor of windows after partitioning with shape [batch_size * num_windows, window_size, window_size, channel].
            (pad_height, pad_width) (Tuple[int, int]): Padded height and width before partitioning.
        """
        # Extract dimensions from the input tensor
        batch_size, height, width, channel = hidden_states.shape
        
        # Calculate padding required to make dimensions divisible by window_size
        pad_h = (window_size - height % window_size) % window_size
        pad_w = (window_size - width % window_size) % window_size
        
        # Apply padding to the input tensor
        hidden_states = F.pad(hidden_states, (0, 0, 0, pad_w, 0, pad_h))
        
        # Update height and width dimensions after padding
        pad_height, pad_width = height + pad_h, width + pad_w
        
        # Reshape the tensor into windows of the specified size
        hidden_states = hidden_states.reshape(
            batch_size, pad_height // window_size, window_size, pad_width // window_size, window_size, channel
        )
        
        # Permute dimensions to arrange windows properly
        windows = hidden_states.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(-1, window_size, window_size, channel)
        
        return windows, (pad_height, pad_width)

    def window_unpartition(
        self, windows: torch.Tensor, window_size: int, padding_shape: Tuple[int, int], original_shape: Tuple[int, int]
    ):
        """
        Reconstruct the original tensor from windows.

        Args:
            windows (torch.Tensor): Tensor of windows with shape [batch_size * num_windows, window_size, window_size, channel].
            window_size (int): Size of the window.
            padding_shape (Tuple[int, int]): Padded height and width before partitioning.
            original_shape (Tuple[int, int]): Original height and width of the input tensor before padding.

        Returns:
            Tensor of original shape [batch_size, height, width, channel].
        """
        # Implementation of this method is typically specific to how window_partition was implemented.
        # It reconstructs the original tensor from the windows created in window_partition.
        pass
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (tensor):
                输入的张量，包含 [batch_size * num_windows, window_size, window_size, channel] 的数据。
            window_size (int):
                窗口大小。
            padding_shape (Tuple):
                填充后的高度和宽度 (pad_height, pad_width)。
            original_shape (Tuple):
                填充前的原始高度和宽度 (height, width)。

        Returns:
            hidden_states: 没有分区的序列，维度为 [batch_size, height, width, channel]。
        """
        pad_height, pad_width = padding_shape
        height, width = original_shape
        batch_size = windows.shape[0] // (pad_height * pad_width // window_size // window_size)
        hidden_states = windows.reshape(
            batch_size, pad_height // window_size, pad_width // window_size, window_size, window_size, -1
        )
        hidden_states = (
            hidden_states.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(batch_size, pad_height, pad_width, -1)
        )

        hidden_states = hidden_states[:, :height, :width, :].contiguous()
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        # 窗口分区
        if self.window_size > 0:
            height, width = hidden_states.shape[1], hidden_states.shape[2]
            hidden_states, padding_shape = self.window_partition(hidden_states, self.window_size)

        hidden_states, attn_weights = self.attn(
            hidden_states=hidden_states,
            output_attentions=output_attentions,
        )
        # 反向窗口分区
        if self.window_size > 0:
            hidden_states = self.window_unpartition(hidden_states, self.window_size, padding_shape, (height, width))

        hidden_states = residual + hidden_states
        layernorm_output = self.layer_norm2(hidden_states)
        hidden_states = hidden_states + self.mlp(layernorm_output)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs
class SamVisionNeck(nn.Module):
    # SamVisionNeck 类，用于实现视觉模型的颈部结构
    def __init__(self, config: SamVisionConfig):
        super().__init__()
        self.config = config

        # 第一个卷积层，将输入特征映射到输出通道数
        self.conv1 = nn.Conv2d(config.hidden_size, config.output_channels, kernel_size=1, bias=False)
        # 第一个层归一化层，对输出进行通道方向的标准化
        self.layer_norm1 = SamLayerNorm(config.output_channels, data_format="channels_first")
        # 第二个卷积层，继续处理特征映射，增加网络的非线性能力
        self.conv2 = nn.Conv2d(config.output_channels, config.output_channels, kernel_size=3, padding=1, bias=False)
        # 第二个层归一化层，对输出进行通道方向的标准化
        self.layer_norm2 = SamLayerNorm(config.output_channels, data_format="channels_first")

    def forward(self, hidden_states):
        # 将输入特征的维度重新排列，以适应卷积层的输入要求
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        # 第一个卷积层的前向传播
        hidden_states = self.conv1(hidden_states)
        # 第一个层归一化层的前向传播
        hidden_states = self.layer_norm1(hidden_states)

        # 第二个卷积层的前向传播
        hidden_states = self.conv2(hidden_states)
        # 第二个层归一化层的前向传播
        hidden_states = self.layer_norm2(hidden_states)
        return hidden_states


class SamVisionEncoder(nn.Module):
    # SamVisionEncoder 类，用于实现视觉模型的编码器结构
    def __init__(self, config: SamVisionConfig):
        super().__init__()
        self.config = config
        self.image_size = config.image_size

        # 图像分块嵌入层，将图像转换为序列数据
        self.patch_embed = SamPatchEmbeddings(config)

        self.pos_embed = None
        if config.use_abs_pos:
            # 如果使用绝对位置编码，则初始化绝对位置嵌入
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1,
                    config.image_size // config.patch_size,
                    config.image_size // config.patch_size,
                    config.hidden_size,
                )
            )

        # 编码器的层列表
        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            # 创建并添加视觉层
            layer = SamVisionLayer(
                config,
                window_size=config.window_size if i not in config.global_attn_indexes else 0,
            )
            self.layers.append(layer)

        # 视觉模型的颈部结构
        self.neck = SamVisionNeck(config)

        # 是否使用梯度检查点
        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.patch_embed

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 省略部分：此处省略了 forward 方法的参数描述
    ) -> Union[Tuple, SamVisionEncoderOutput]:
        # 检查是否需要输出注意力权重，默认使用配置中的设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 检查是否需要输出隐藏状态，默认使用配置中的设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 检查是否使用返回字典格式，默认使用配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果像素值为 None，则抛出数值错误异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 将像素值传入到补丁嵌入层中
        hidden_states = self.patch_embed(pixel_values)
        # 如果存在位置编码，则将其加到隐藏状态中
        if self.pos_embed is not None:
            hidden_states = hidden_states + self.pos_embed

        # 初始化用于存储所有隐藏状态的变量，如果不需要输出隐藏状态则设为 None
        all_hidden_states = () if output_hidden_states else None
        # 初始化用于存储所有自注意力权重的变量，如果不需要输出注意力权重则设为 None
        all_self_attentions = () if output_attentions else None

        # 遍历所有编码器层
        for i, layer_module in enumerate(self.layers):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到存储列表中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果启用梯度检查点并且处于训练模式，则使用梯度检查点函数调用层模块
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                )
            else:
                # 否则直接调用层模块，传入当前隐藏状态和是否需要输出注意力权重的标志
                layer_outputs = layer_module(hidden_states, output_attentions=output_attentions)

            # 更新隐藏状态为当前层模块的输出的第一个元素（通常是新的隐藏状态）
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力权重，则将当前层的自注意力权重添加到存储列表中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果需要输出隐藏状态，则将最终的隐藏状态添加到存储列表中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 将最终的隐藏状态传入到“neck”层中进行处理
        hidden_states = self.neck(hidden_states)

        # 如果不使用返回字典格式，则构建返回的输出元组
        if not return_dict:
            outputs = (hidden_states,)
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if output_attentions:
                outputs = outputs + (all_self_attentions,)
            return outputs

        # 如果使用返回字典格式，则构建并返回 SamVisionEncoderOutput 对象
        return SamVisionEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
class SamPreTrainedModel(PreTrainedModel):
    # 配置使用的配置类
    config_class = SamConfig
    # 模型中基础模型的前缀名称
    base_model_prefix = "sam"
    # 主输入名称为像素值
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        # 从配置中获取初始化范围的标准差
        std = self.config.initializer_range
        # 如果模块是线性层、二维卷积层或反卷积层
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            # 初始化权重为正态分布
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果有偏置，则初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是嵌入层
        elif isinstance(module, nn.Embedding):
            # 初始化权重为正态分布
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果有填充索引，则将对应索引的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()



@add_start_docstrings(
    "Segment Anything Model (SAM) for generating segmentation masks, given an input image and ",
    " optional 2D location and bounding boxes.",
    SAM_START_DOCSTRING,
)
class SamModel(SamPreTrainedModel):
    # 需要共享权重的键列表
    _tied_weights_keys = ["prompt_encoder.shared_embedding.positional_embedding"]

    def __init__(self, config):
        super().__init__(config)
        # 共享图像嵌入
        self.shared_image_embedding = SamPositionalEmbedding(config.vision_config)

        # 视觉编码器
        self.vision_encoder = SamVisionEncoder(config.vision_config)
        # 提示编码器，使用共享的图像嵌入
        self.prompt_encoder = SamPromptEncoder(config.prompt_encoder_config, self.shared_image_embedding)
        # 掩码解码器
        self.mask_decoder = SamMaskDecoder(config.mask_decoder_config)

        # 进行初始化后处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回视觉编码器的输入嵌入
        return self.vision_encoder.get_input_embeddings()


注释：
    def get_image_wide_positional_embeddings(self):
        # 获取图像嵌入的位置编码，使用配置中的图像嵌入大小
        size = self.config.prompt_encoder_config.image_embedding_size
        # 获取共享的图像嵌入位置编码的设备和数据类型
        target_device = self.shared_image_embedding.positional_embedding.device
        target_dtype = self.shared_image_embedding.positional_embedding.dtype
        # 创建一个全为1的张量作为网格，设备和数据类型与位置编码相同
        grid = torch.ones((size, size), device=target_device, dtype=target_dtype)
        # 计算垂直方向的位置编码
        y_embed = grid.cumsum(dim=0) - 0.5
        # 计算水平方向的位置编码
        x_embed = grid.cumsum(dim=1) - 0.5
        # 将位置编码归一化到 [0, 1] 范围内
        y_embed = y_embed / size
        x_embed = x_embed / size

        # 使用共享的图像嵌入模型，将 x 和 y 的位置编码堆叠起来作为输入
        positional_embedding = self.shared_image_embedding(torch.stack([x_embed, y_embed], dim=-1))
        # 将通道维度放到最前面，返回的形状为 channel x height x width
        return positional_embedding.permute(2, 0, 1).unsqueeze(0)

    @torch.no_grad()
    def get_image_embeddings(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        返回通过视觉编码器处理像素值得到的图像嵌入。

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                输入的像素值
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的注意力张量。
            output_hidden_states (`bool`, *optional*):
                是否返回所有层的隐藏状态。
            return_dict (`bool`, *optional*):
                是否返回 [`~utils.ModelOutput`] 而不是简单的元组。

        """
        # 使用视觉编码器处理像素值，根据参数决定是否返回特定的信息
        vision_output = self.vision_encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 提取视觉编码器输出的图像嵌入张量
        image_embeddings = vision_output[0]
        return image_embeddings

    @torch.no_grad()
    def get_prompt_embeddings(
        self,
        input_points: Optional[torch.FloatTensor] = None,
        input_labels: Optional[torch.LongTensor] = None,
        input_boxes: Optional[torch.FloatTensor] = None,
        input_masks: Optional[torch.LongTensor] = None,
    ):
        # 返回用于提示的嵌入，接受多种类型的输入作为可选参数
        # （具体实现部分未提供，在实际代码中可能有更多的细节处理）
    ):
        r"""
        Returns the prompt embeddings by passing the input points, labels, boxes and masks through the prompt encoder.

        Args:
            input_points (`torch.FloatTensor` of shape `(batch_size, point_batch_size, num_points_per_image, 2)`):
                Optional input points for the prompt encoder. The padding of the point is automatically done by the
                processor. `point_batch_size` refers to the number of masks that we want the model to predict per
                point. The model will output `point_batch_size` times 3 masks in total.
            input_labels (`torch.LongTensor` of shape `(batch_size, point_batch_size, num_points_per_image)`):
                Optional input labels for the prompt encoder. The padding of the labels is automatically done by the
                processor, or can be fed by the user.
            input_boxes (`torch.FloatTensor` of shape `(batch_size, num_boxes_per_image, 4)`):
                Optional input boxes for the prompt encoder. The padding of the boxes is automatically done by the
                processor. users can also pass manually the input boxes.
            input_masks (`torch.LongTensor` of shape `(batch_size, image_size, image_size)`):
                Optional input masks for the prompt encoder.
        """
        # 使用 prompt_encoder 方法计算 prompt 的嵌入结果，传入参数包括输入的点、标签、框和掩码
        prompt_output = self.prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
        )
        # 返回 prompt 的嵌入结果
        return prompt_output

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