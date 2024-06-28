# `.\models\detr\modeling_detr.py`

```py
# coding=utf-8
# Copyright 2021 Facebook AI Research The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch DETR model."""

import math  # 导入数学函数库
from dataclasses import dataclass  # 导入数据类装饰器
from typing import Dict, List, Optional, Tuple, Union  # 导入类型提示模块

import torch  # 导入PyTorch库
from torch import Tensor, nn  # 导入张量和神经网络模块

from ...activations import ACT2FN  # 导入激活函数映射
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask  # 导入注意力掩码工具函数
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithCrossAttentions, Seq2SeqModelOutput  # 导入模型输出类
from ...modeling_utils import PreTrainedModel  # 导入预训练模型基类
from ...utils import (
    ModelOutput,  # 导入模型输出工具类
    add_start_docstrings,  # 导入添加文档字符串的工具函数
    add_start_docstrings_to_model_forward,  # 导入添加模型前向文档字符串的工具函数
    is_accelerate_available,  # 导入加速库可用性检查函数
    is_scipy_available,  # 导入SciPy库可用性检查函数
    is_timm_available,  # 导入Timm库可用性检查函数
    is_vision_available,  # 导入视觉库可用性检查函数
    logging,  # 导入日志模块
    replace_return_docstrings,  # 导入替换返回文档字符串的工具函数
    requires_backends,  # 导入后端需求检查函数
)
from ...utils.backbone_utils import load_backbone  # 导入加载骨干网络函数
from .configuration_detr import DetrConfig  # 导入DETR模型配置类


if is_accelerate_available():  # 如果加速库可用
    from accelerate import PartialState  # 导入部分状态
    from accelerate.utils import reduce  # 导入数据缩减函数

if is_scipy_available():  # 如果SciPy库可用
    from scipy.optimize import linear_sum_assignment  # 导入线性求解分配函数

if is_timm_available():  # 如果Timm库可用
    from timm import create_model  # 导入创建模型函数

if is_vision_available():  # 如果视觉库可用
    from transformers.image_transforms import center_to_corners_format  # 导入中心转角格式转换函数

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

_CONFIG_FOR_DOC = "DetrConfig"  # 用于文档的DETR配置类名
_CHECKPOINT_FOR_DOC = "facebook/detr-resnet-50"  # 用于文档的预训练模型名

DETR_PRETRAINED_MODEL_ARCHIVE_LIST = [  # DETR预训练模型存档列表
    "facebook/detr-resnet-50",
    # See all DETR models at https://huggingface.co/models?filter=detr
]


@dataclass
class DetrDecoderOutput(BaseModelOutputWithCrossAttentions):
    """
    Base class for outputs of the DETR decoder. This class adds one attribute to BaseModelOutputWithCrossAttentions,
    namely an optional stack of intermediate decoder activations, i.e. the output of each decoder layer, each of them
    gone through a layernorm. This is useful when training the model with auxiliary decoding losses.
    """
    # 定义一个可选参数 `intermediate_hidden_states`，类型为 `torch.FloatTensor`，形状为 `(config.decoder_layers, batch_size, num_queries, hidden_size)`
    # 如果 `config.auxiliary_loss=True`，则返回中间解码器激活值，即每个解码器层的输出，每个都经过了层归一化处理。
    intermediate_hidden_states: Optional[torch.FloatTensor] = None
# 使用 `dataclass` 装饰器定义一个数据类 `DetrModelOutput`，它是 `Seq2SeqModelOutput` 的子类
@dataclass
class DetrModelOutput(Seq2SeqModelOutput):
    """
    DETR 编码-解码模型的输出基类。这个类在 `Seq2SeqModelOutput` 的基础上增加了一个属性，
    即可选的中间解码器激活堆栈，即每个解码器层的输出，每个输出通过了一个 layernorm。
    在使用辅助解码损失训练模型时非常有用。
    """

    # 可选的中间隐藏状态，类型为 FloatTensor
    intermediate_hidden_states: Optional[torch.FloatTensor] = None


# 使用 `dataclass` 装饰器定义一个数据类 `DetrObjectDetectionOutput`，它是 `ModelOutput` 的子类
@dataclass
class DetrObjectDetectionOutput(ModelOutput):
    """
    [`DetrForObjectDetection`] 的输出类型。
    """

    # 可选的损失，类型为 FloatTensor
    loss: Optional[torch.FloatTensor] = None
    # 可选的损失字典，类型为字典
    loss_dict: Optional[Dict] = None
    # 预测的 logits，类型为 FloatTensor
    logits: torch.FloatTensor = None
    # 预测的框，类型为 FloatTensor
    pred_boxes: torch.FloatTensor = None
    # 可选的辅助输出，类型为列表中包含字典
    auxiliary_outputs: Optional[List[Dict]] = None
    # 最后的隐藏状态，类型为 FloatTensor
    last_hidden_state: Optional[torch.FloatTensor] = None
    # 解码器的隐藏状态，类型为元组中的 FloatTensor
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 解码器的注意力，类型为元组中的 FloatTensor
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 交叉注意力，类型为元组中的 FloatTensor
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器的最后隐藏状态，类型为 FloatTensor
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 编码器的隐藏状态，类型为元组中的 FloatTensor
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器的注意力，类型为元组中的 FloatTensor
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


# 使用 `dataclass` 装饰器定义一个数据类 `DetrSegmentationOutput`，它是 `ModelOutput` 的子类
@dataclass
class DetrSegmentationOutput(ModelOutput):
    """
    [`DetrForSegmentation`] 的输出类型。
    """

    # 可选的损失，类型为 FloatTensor
    loss: Optional[torch.FloatTensor] = None
    # 可选的损失字典，类型为字典
    loss_dict: Optional[Dict] = None
    # 预测的 logits，类型为 FloatTensor
    logits: torch.FloatTensor = None
    # 预测的框，类型为 FloatTensor
    pred_boxes: torch.FloatTensor = None
    # 预测的掩码，类型为 FloatTensor
    pred_masks: torch.FloatTensor = None
    # 可选的辅助输出，类型为列表中包含字典
    auxiliary_outputs: Optional[List[Dict]] = None
    # 最后的隐藏状态，类型为 FloatTensor
    last_hidden_state: Optional[torch.FloatTensor] = None
    # 解码器的隐藏状态，类型为元组中的 FloatTensor
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 解码器的注意力，类型为元组中的 FloatTensor
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 交叉注意力，类型为元组中的 FloatTensor
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器的最后隐藏状态，类型为 FloatTensor
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # 编码器的隐藏状态，类型为元组中的 FloatTensor
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 编码器的注意力，类型为元组中的 FloatTensor
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


# 以下是从 https://github.com/facebookresearch/detr/blob/master/backbone.py 复制的实用程序
class DetrFrozenBatchNorm2d(nn.Module):
    """
    固定统计量和仿射参数的 BatchNorm2d。

    从 torchvision.misc.ops 中复制粘贴，添加了 eps 在 rqsrt 前，否则除了 torchvision.models.resnet[18,34,50,101] 之外的任何模型都会产生 NaN。
    """

    def __init__(self, n):
        super().__init__()
        # 注册缓冲区：权重、偏置、运行时均值、运行时方差，都是长度为 n 的张量
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        ):
        # 加载模型状态字典时的特殊方法
        pass
    ):
        # 构建存储批次追踪数的键名
        num_batches_tracked_key = prefix + "num_batches_tracked"
        # 如果状态字典中存在批次追踪数的键名，则删除该键
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        # 调用父类的方法，从状态字典加载模型参数
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):
        # 将权重重塑为适合用户操作的形状
        weight = self.weight.reshape(1, -1, 1, 1)
        # 将偏置重塑为适合用户操作的形状
        bias = self.bias.reshape(1, -1, 1, 1)
        # 将运行时方差重塑为适合用户操作的形状
        running_var = self.running_var.reshape(1, -1, 1, 1)
        # 将运行时均值重塑为适合用户操作的形状
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        # 定义一个很小的常数 epsilon，用于数值稳定性
        epsilon = 1e-5
        # 计算 scale，用于标准化
        scale = weight * (running_var + epsilon).rsqrt()
        # 调整偏置，确保数据的中心化
        bias = bias - running_mean * scale
        # 返回经过标准化和偏置处理后的输入 x
        return x * scale + bias
def replace_batch_norm(model):
    r"""
    Recursively replace all `torch.nn.BatchNorm2d` with `DetrFrozenBatchNorm2d`.

    Args:
        model (torch.nn.Module):
            input model
    """
    # 遍历模型的所有子模块
    for name, module in model.named_children():
        # 如果当前模块是 `nn.BatchNorm2d` 类型
        if isinstance(module, nn.BatchNorm2d):
            # 创建一个新的 `DetrFrozenBatchNorm2d` 模块
            new_module = DetrFrozenBatchNorm2d(module.num_features)

            # 如果当前 `nn.BatchNorm2d` 模块的权重不在 "meta" 设备上
            if not module.weight.device == torch.device("meta"):
                # 复制权重、偏置、均值和方差到新的 `DetrFrozenBatchNorm2d` 模块
                new_module.weight.data.copy_(module.weight)
                new_module.bias.data.copy_(module.bias)
                new_module.running_mean.data.copy_(module.running_mean)
                new_module.running_var.data.copy_(module.running_var)

            # 将模型中原来的 `nn.BatchNorm2d` 模块替换为新的 `DetrFrozenBatchNorm2d` 模块
            model._modules[name] = new_module

        # 如果当前模块还有子模块，则递归替换其中的 `nn.BatchNorm2d` 模块
        if len(list(module.children())) > 0:
            replace_batch_norm(module)


class DetrConvEncoder(nn.Module):
    """
    Convolutional backbone, using either the AutoBackbone API or one from the timm library.

    nn.BatchNorm2d layers are replaced by DetrFrozenBatchNorm2d as defined above.

    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        # 根据配置选择使用 timm 库中的模型还是自定义加载的模型
        if config.use_timm_backbone:
            # 如果使用 timm 模型，则确保需要 timm 后端支持
            requires_backends(self, ["timm"])
            kwargs = {}
            if config.dilation:
                kwargs["output_stride"] = 16
            # 创建 timm 模型，并只输出特征
            backbone = create_model(
                config.backbone,
                pretrained=config.use_pretrained_backbone,
                features_only=True,
                out_indices=(1, 2, 3, 4),
                in_chans=config.num_channels,
                **kwargs,
            )
        else:
            # 否则加载自定义的模型
            backbone = load_backbone(config)

        # 使用 `replace_batch_norm` 函数将模型中所有的 `nn.BatchNorm2d` 替换为 `DetrFrozenBatchNorm2d`
        with torch.no_grad():
            replace_batch_norm(backbone)

        # 将替换后的模型设置为类属性
        self.model = backbone
        # 根据配置获取中间层的通道数信息
        self.intermediate_channel_sizes = (
            self.model.feature_info.channels() if config.use_timm_backbone else self.model.channels
        )

        # 根据模型类型和配置设置参数的梯度是否需要计算
        backbone_model_type = config.backbone if config.use_timm_backbone else config.backbone_config.model_type
        if "resnet" in backbone_model_type:
            for name, parameter in self.model.named_parameters():
                if config.use_timm_backbone:
                    # 对于 timm 模型，除了特定的几个阶段，其他参数设为不需要梯度计算
                    if "layer2" not in name and "layer3" not in name and "layer4" not in name:
                        parameter.requires_grad_(False)
                else:
                    # 对于自定义加载的模型，除了特定的几个阶段，其他参数设为不需要梯度计算
                    if "stage.1" not in name and "stage.2" not in name and "stage.3" not in name:
                        parameter.requires_grad_(False)
    # 定义一个前向传播方法，接收像素数值和像素掩码作为输入参数
    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor):
        # 如果配置要求使用timm的后端模型，则将像素值传递给模型并获取特征图列表，否则直接从模型获取特征图
        features = self.model(pixel_values) if self.config.use_timm_backbone else self.model(pixel_values).feature_maps

        # 初始化一个空列表，用于存储输出的特征图和相应的掩码
        out = []
        for feature_map in features:
            # 将像素掩码下采样至与对应特征图相同的形状
            mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
            out.append((feature_map, mask))
        # 返回包含特征图和掩码的列表
        return out
class DetrConvModel(nn.Module):
    """
    This module adds 2D position embeddings to all intermediate feature maps of the convolutional encoder.
    """

    def __init__(self, conv_encoder, position_embedding):
        super().__init__()
        self.conv_encoder = conv_encoder  # 初始化卷积编码器模型
        self.position_embedding = position_embedding  # 初始化位置嵌入模型

    def forward(self, pixel_values, pixel_mask):
        # 将像素值和像素掩码通过骨干网络传递，获取(feature_map, pixel_mask)元组的列表
        out = self.conv_encoder(pixel_values, pixel_mask)
        pos = []
        for feature_map, mask in out:
            # 位置编码
            pos.append(self.position_embedding(feature_map, mask).to(feature_map.dtype))

        return out, pos


class DetrSinePositionEmbedding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(self, embedding_dim=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.embedding_dim = embedding_dim  # 嵌入维度
        self.temperature = temperature  # 温度参数
        self.normalize = normalize  # 是否归一化
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale  # 缩放参数

    def forward(self, pixel_values, pixel_mask):
        if pixel_mask is None:
            raise ValueError("No pixel mask provided")  # 若未提供像素掩码，则引发异常
        y_embed = pixel_mask.cumsum(1, dtype=torch.float32)  # 沿着纵向累积和
        x_embed = pixel_mask.cumsum(2, dtype=torch.float32)  # 沿着横向累积和
        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * self.scale  # 归一化并乘以缩放参数
            x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * self.scale  # 归一化并乘以缩放参数

        dim_t = torch.arange(self.embedding_dim, dtype=torch.int64, device=pixel_values.device).float()  # 创建维度张量
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.embedding_dim)  # 计算温度的指数变换

        pos_x = x_embed[:, :, :, None] / dim_t  # 计算X方向的位置编码
        pos_y = y_embed[:, :, :, None] / dim_t  # 计算Y方向的位置编码
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)  # 正弦余弦变换
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)  # 正弦余弦变换
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # 合并并转置维度
        return pos


class DetrLearnedPositionEmbedding(nn.Module):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, embedding_dim=256):
        super().__init__()
        self.row_embeddings = nn.Embedding(50, embedding_dim)  # 行嵌入模块
        self.column_embeddings = nn.Embedding(50, embedding_dim)  # 列嵌入模块
    # 定义一个前向传播方法，接受像素数值和可选的像素掩码作为输入
    def forward(self, pixel_values, pixel_mask=None):
        # 获取输入张量的高度和宽度
        height, width = pixel_values.shape[-2:]
        # 在设备上创建一个张量，包含从0到width-1的整数值，用于列的位置编码
        width_values = torch.arange(width, device=pixel_values.device)
        # 在设备上创建一个张量，包含从0到height-1的整数值，用于行的位置编码
        height_values = torch.arange(height, device=pixel_values.device)
        # 使用列位置编码器计算列方向的位置嵌入
        x_emb = self.column_embeddings(width_values)
        # 使用行位置编码器计算行方向的位置嵌入
        y_emb = self.row_embeddings(height_values)
        # 创建位置张量，结合列和行的位置嵌入，形成二维位置信息
        pos = torch.cat([x_emb.unsqueeze(0).repeat(height, 1, 1), y_emb.unsqueeze(1).repeat(1, width, 1)], dim=-1)
        # 将位置张量进行维度置换，调整顺序为(位置嵌入维度, 高度, 宽度)
        pos = pos.permute(2, 0, 1)
        # 在第一维度上添加一个维度，用于批处理维度
        pos = pos.unsqueeze(0)
        # 将位置张量复制以适配输入像素值张量的批处理大小
        pos = pos.repeat(pixel_values.shape[0], 1, 1, 1)
        # 返回位置张量，其中包含了输入像素值的位置信息
        return pos
def build_position_encoding(config):
    # 根据模型配置计算位置编码的步数
    n_steps = config.d_model // 2
    # 根据位置嵌入类型选择不同的位置编码方法
    if config.position_embedding_type == "sine":
        # 如果选择使用正弦位置编码，则创建一个正弦位置嵌入对象
        # TODO find a better way of exposing other arguments
        position_embedding = DetrSinePositionEmbedding(n_steps, normalize=True)
    elif config.position_embedding_type == "learned":
        # 如果选择使用学习得到的位置编码，则创建一个学习得到的位置嵌入对象
        position_embedding = DetrLearnedPositionEmbedding(n_steps)
    else:
        # 如果选择的位置编码类型不被支持，则抛出数值错误异常
        raise ValueError(f"Not supported {config.position_embedding_type}")

    # 返回位置嵌入对象
    return position_embedding


class DetrAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.

    Here, we add position embeddings to the queries and keys (as explained in the DETR paper).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        # 确保 embed_dim 必须能被 num_heads 整除，否则抛出数值错误异常
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {num_heads})."
            )
        # 缩放因子，用于缩放注意力分数
        self.scaling = self.head_dim**-0.5

        # 初始化线性映射函数，用于 Q、K、V 的投影
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        # 将输入张量重新形状为适合多头注意力计算的形状
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def with_pos_embed(self, tensor: torch.Tensor, object_queries: Optional[Tensor], **kwargs):
        # 获取位置嵌入，或者警告使用过时的位置嵌入
        position_embeddings = kwargs.pop("position_embeddings", None)

        if kwargs:
            # 如果有未预期的关键字参数，则抛出数值错误异常
            raise ValueError(f"Unexpected arguments {kwargs.keys()}")

        if position_embeddings is not None and object_queries is not None:
            # 如果同时指定了位置嵌入和物体查询，则抛出数值错误异常
            raise ValueError(
                "Cannot specify both position_embeddings and object_queries. Please use just object_queries"
            )

        if position_embeddings is not None:
            # 如果仅指定了位置嵌入，则发出一次性警告
            logger.warning_once(
                "position_embeddings has been deprecated and will be removed in v4.34. Please use object_queries instead"
            )
            # 将物体查询设置为位置嵌入
            object_queries = position_embeddings

        # 返回带有位置嵌入的张量或者原始张量
        return tensor if object_queries is None else tensor + object_queries

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        object_queries: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        spatial_position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs,
    ):
        # 这里定义了注意力层的前向传播过程，具体实现包括 Q、K、V 的映射以及多头注意力机制等
    # 初始化函数，用于初始化一个DetrEncoderLayer对象
    def __init__(self, config: DetrConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 设置嵌入维度为config中定义的模型维度
        self.embed_dim = config.d_model
        # 初始化自注意力机制，使用DetrAttention类
        self.self_attn = DetrAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        # 初始化自注意力层的LayerNorm层，对嵌入向量进行归一化
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 设置dropout概率，用于各种dropout操作
        self.dropout = config.dropout
        # 根据配置文件选择激活函数
        self.activation_fn = ACT2FN[config.activation_function]
        # 设置激活函数的dropout概率
        self.activation_dropout = config.activation_dropout
        # 第一个全连接层，将嵌入维度映射到编码器中间维度
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        # 第二个全连接层，将编码器中间维度映射回嵌入维度
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        # 最终的LayerNorm层，对编码器输出进行最后的归一化
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
    
    # 前向传播函数，用于执行数据的前向传播
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量
        attention_mask: torch.Tensor,  # 注意力掩码张量
        object_queries: torch.Tensor = None,  # 目标查询张量，默认为空
        output_attentions: bool = False,  # 是否输出注意力权重，默认为False
        **kwargs,  # 其他关键字参数
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
                隐藏状态，形状为 `(batch, seq_len, embed_dim)` 的输入张量
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative
                values.
                注意力掩码，大小为 `(batch, 1, target_len, source_len)`，其中填充元素由非常大的负值表示
            object_queries (`torch.FloatTensor`, *optional*):
                Object queries (also called content embeddings), to be added to the hidden states.
                对象查询（也称为内容嵌入），将添加到隐藏状态中
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
                是否返回所有注意力层的注意力张量。详细信息请参见返回的张量中的 `attentions` 字段
        """
        position_embeddings = kwargs.pop("position_embeddings", None)  # 从关键字参数中弹出 `position_embeddings`

        if kwargs:
            raise ValueError(f"Unexpected arguments {kwargs.keys()}")  # 如果还有未处理的关键字参数，引发错误

        if position_embeddings is not None and object_queries is not None:
            raise ValueError(
                "Cannot specify both position_embeddings and object_queries. Please use just object_queries"
            )  # 如果同时指定了 `position_embeddings` 和 `object_queries`，则引发错误

        if position_embeddings is not None:
            logger.warning_once(
                "position_embeddings has been deprecated and will be removed in v4.34. Please use object_queries instead"
            )  # 如果指定了 `position_embeddings`，发出警告提示将在 v4.34 版本中移除 `position_embeddings`，建议使用 `object_queries`
            object_queries = position_embeddings  # 使用 `position_embeddings` 替代 `object_queries`

        residual = hidden_states  # 保存输入的隐藏状态

        # 通过自注意力机制处理隐藏状态
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            object_queries=object_queries,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)  # 对隐藏状态进行 dropout
        hidden_states = residual + hidden_states  # 加上残差连接
        hidden_states = self.self_attn_layer_norm(hidden_states)  # 对加和后的隐藏状态进行 layer normalization

        residual = hidden_states  # 保存上一层处理后的隐藏状态

        # 通过全连接层 fc1 处理隐藏状态，并使用激活函数进行非线性变换
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)  # 对处理后的隐藏状态进行 dropout

        hidden_states = self.fc2(hidden_states)  # 通过全连接层 fc2 进行线性变换
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)  # 对处理后的隐藏状态进行 dropout

        hidden_states = residual + hidden_states  # 加上残差连接
        hidden_states = self.final_layer_norm(hidden_states)  # 对加和后的隐藏状态进行 layer normalization

        if self.training:
            if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
                # 如果隐藏状态中存在无穷大或 NaN 值，进行 clamp 操作，避免数值溢出或无效操作

        outputs = (hidden_states,)  # 定义最终的输出为处理后的隐藏状态元组

        if output_attentions:
            outputs += (attn_weights,)  # 如果需要输出注意力权重，则将注意力权重加入输出元组中

        return outputs  # 返回最终的输出元组
class DetrDecoderLayer(nn.Module):
    # DETR 解码器层定义
    def __init__(self, config: DetrConfig):
        super().__init__()
        self.embed_dim = config.d_model

        # 自注意力机制
        self.self_attn = DetrAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        # 自注意力机制层归一化
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # 编码器注意力机制
        self.encoder_attn = DetrAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # 前馈神经网络层
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)

        # 最终层归一化
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        object_queries: Optional[torch.Tensor] = None,
        query_position_embeddings: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
class DetrClassificationHead(nn.Module):
    """用于句子级分类任务的头部模块。"""

    def __init__(self, input_dim: int, inner_dim: int, num_classes: int, pooler_dropout: float):
        super().__init__()
        # 全连接层
        self.dense = nn.Linear(input_dim, inner_dim)
        # Dropout 层
        self.dropout = nn.Dropout(p=pooler_dropout)
        # 输出投影层
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class DetrPreTrainedModel(PreTrainedModel):
    config_class = DetrConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"
    _no_split_modules = [r"DetrConvEncoder", r"DetrEncoderLayer", r"DetrDecoderLayer"]
    # 初始化模型权重的函数，根据模块类型不同采用不同的初始化方法
    def _init_weights(self, module):
        # 从配置中获取标准差和Xavier初始化的标准差
        std = self.config.init_std
        xavier_std = self.config.init_xavier_std

        # 如果是 DetrMHAttentionMap 类型的模块
        if isinstance(module, DetrMHAttentionMap):
            # 初始化 k_linear 和 q_linear 模块的偏置为零
            nn.init.zeros_(module.k_linear.bias)
            nn.init.zeros_(module.q_linear.bias)
            # 使用 Xavier 均匀分布初始化 k_linear 和 q_linear 模块的权重
            nn.init.xavier_uniform_(module.k_linear.weight, gain=xavier_std)
            nn.init.xavier_uniform_(module.q_linear.weight, gain=xavier_std)
        
        # 如果是 DetrLearnedPositionEmbedding 类型的模块
        elif isinstance(module, DetrLearnedPositionEmbedding):
            # 使用均匀分布初始化 row_embeddings 和 column_embeddings 的权重
            nn.init.uniform_(module.row_embeddings.weight)
            nn.init.uniform_(module.column_embeddings.weight)
        
        # 如果是 nn.Linear, nn.Conv2d, nn.BatchNorm2d 之一的模块
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            # 使用正态分布初始化权重，均值为0，标准差为 std
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果存在偏置，则将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        
        # 如果是 nn.Embedding 类型的模块
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重，均值为0，标准差为 std
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果定义了 padding_idx，则将对应位置的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
# DETR 模型的文档字符串，提供了有关模型继承自 `PreTrainedModel` 的说明，建议查看超类文档以了解库实现的通用方法（例如下载或保存模型、调整输入嵌入大小、修剪头等）。

This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

# DETR 模型同时也是 PyTorch 的 `torch.nn.Module` 子类，可以像普通的 PyTorch 模块一样使用，有关一般用法和行为，请参考 PyTorch 文档。

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

# 参数部分描述了模型的初始化参数，需要一个 `DetrConfig` 类的实例。通过配置文件初始化模型不会加载与模型关联的权重，只加载配置。可以查看 `~PreTrainedModel.from_pretrained` 方法来加载模型权重。

Parameters:
    config ([`DetrConfig`]):
        Model configuration class with all the parameters of the model. Initializing with a config file does not
        load the weights associated with the model, only the configuration. Check out the
        [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            # 像素值，表示图像的像素数据，格式为(batch_size, num_channels, height, width)

            Pixel values. Padding will be ignored by default should you provide it.
            # 默认情况下会忽略填充的像素值

        pixel_mask (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            # 像素掩码，用于避免在填充像素上执行注意力操作，形状为(batch_size, height, width)，可选参数

            Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:
            # 用于遮盖填充像素，以避免在其上执行注意力操作。掩码值在[0, 1]之间：

            - 1 for pixels that are real (i.e. **not masked**),
            # 1 表示真实像素（即未被遮盖）

            - 0 for pixels that are padding (i.e. **masked**).
            # 0 表示填充像素（即已被遮盖）

            [What are attention masks?](../glossary#attention-mask)
            # 了解注意力掩码的更多信息，请参考链接中的文档

        decoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, num_queries)`, *optional*):
            # 解码器注意力掩码，形状为(batch_size, num_queries)，可选参数
            Not used by default. Can be used to mask object queries.
            # 默认情况下不使用。可用于遮盖对象查询。

        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            # 编码器输出，形状为(tuple(torch.FloatTensor)，可选参数
            Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`)
            # 元组包括(`last_hidden_state`, 可选: `hidden_states`, 可选: `attentions`)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of
            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
            # `last_hidden_state` 形状为(batch_size, sequence_length, hidden_size)，是编码器最后一层的隐藏状态输出。在解码器的交叉注意力中使用。

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            # 输入嵌入，形状为(batch_size, sequence_length, hidden_size)，可选参数
            Optionally, instead of passing the flattened feature map (output of the backbone + projection layer), you
            # 可选择直接传递扁平化的特征图（骨干网络输出 + 投影层输出），而不是传递这些特征图的嵌入表示。

            can choose to directly pass a flattened representation of an image.
            # 可选择直接传递图像的扁平化表示。

        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
            # 解码器输入嵌入，形状为(batch_size, num_queries, hidden_size)，可选参数
            Optionally, instead of initializing the queries with a tensor of zeros, you can choose to directly pass an
            # 可选择直接传递一个嵌入表示来初始化查询，而不是用零张量初始化。

            embedded representation.
            # 嵌入表示

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关详细信息，请参见返回张量下的`attentions`。
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
            # 是否返回所有注意力层的注意力张量。有关详细信息，请参见返回张量下的`attentions`。

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关详细信息，请参见返回张量下的`hidden_states`。
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
            # 是否返回所有层的隐藏状态。有关详细信息，请参见返回张量下的`hidden_states`。

        return_dict (`bool`, *optional*):
            # 是否返回[`~utils.ModelOutput`]而不是普通的元组。
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            # 是否返回一个[`~utils.ModelOutput`]而不是普通的元组。
"""
class DetrEncoder(DetrPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`DetrEncoderLayer`].

    The encoder updates the flattened feature map through multiple self-attention layers.

    Small tweak for DETR:

    - object_queries are added to the forward pass.

    Args:
        config: DetrConfig
    """

    def __init__(self, config: DetrConfig):
        super().__init__(config)

        self.dropout = config.dropout  # 从配置中获取 dropout 率
        self.layerdrop = config.encoder_layerdrop  # 从配置中获取 encoder 层级 dropout 率

        self.layers = nn.ModuleList([DetrEncoderLayer(config) for _ in range(config.encoder_layers)])  # 创建指定数量的编码器层

        # in the original DETR, no layernorm is used at the end of the encoder, as "normalize_before" is set to False by default
        # 在原始的DETR中，编码器末端不使用layernorm，因为默认情况下"normalize_before"设置为False

        # Initialize weights and apply final processing
        self.post_init()  # 执行初始化和最终处理



    def forward(
        self,
        inputs_embeds=None,
        attention_mask=None,
        object_queries=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,



class DetrDecoder(DetrPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`DetrDecoderLayer`].

    The decoder updates the query embeddings through multiple self-attention and cross-attention layers.

    Some small tweaks for DETR:

    - object_queries and query_position_embeddings are added to the forward pass.
    - if self.config.auxiliary_loss is set to True, also returns a stack of activations from all decoding layers.

    Args:
        config: DetrConfig
    """

    def __init__(self, config: DetrConfig):
        super().__init__(config)
        self.dropout = config.dropout  # 从配置中获取 dropout 率
        self.layerdrop = config.decoder_layerdrop  # 从配置中获取 decoder 层级 dropout 率

        self.layers = nn.ModuleList([DetrDecoderLayer(config) for _ in range(config.decoder_layers)])  # 创建指定数量的解码器层
        # in DETR, the decoder uses layernorm after the last decoder layer output
        # 在DETR中，解码器在最后一层解码器输出后使用layernorm
        self.layernorm = nn.LayerNorm(config.d_model)  # 创建指定维度的layernorm层

        self.gradient_checkpointing = False  # 默认关闭梯度检查点

        # Initialize weights and apply final processing
        self.post_init()  # 执行初始化和最终处理



    def forward(
        self,
        inputs_embeds=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        object_queries=None,
        query_position_embeddings=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,



@add_start_docstrings(
    """
    The bare DETR Model (consisting of a backbone and encoder-decoder Transformer) outputting raw hidden-states without
    any specific head on top.
    """,
    DETR_START_DOCSTRING,
)
class DetrModel(DetrPreTrainedModel):
    """
    DETR模型，包括骨干和编码器-解码器Transformer，输出没有特定头部的原始隐藏状态。
    """
    # 初始化函数，接受一个DetrConfig类型的配置对象作为参数
    def __init__(self, config: DetrConfig):
        # 调用父类的初始化方法，传入配置对象
        super().__init__(config)

        # 创建backbone和位置编码
        backbone = DetrConvEncoder(config)
        object_queries = build_position_encoding(config)
        # 使用创建的backbone和位置编码创建DetrConvModel对象，并赋给self.backbone属性
        self.backbone = DetrConvModel(backbone, object_queries)

        # 创建投影层，使用nn.Conv2d进行初始化
        self.input_projection = nn.Conv2d(backbone.intermediate_channel_sizes[-1], config.d_model, kernel_size=1)

        # 创建查询位置嵌入层，使用nn.Embedding进行初始化
        self.query_position_embeddings = nn.Embedding(config.num_queries, config.d_model)

        # 创建编码器和解码器对象
        self.encoder = DetrEncoder(config)
        self.decoder = DetrDecoder(config)

        # 初始化权重并进行最终处理
        self.post_init()

    # 返回编码器对象
    def get_encoder(self):
        return self.encoder

    # 返回解码器对象
    def get_decoder(self):
        return self.decoder

    # 冻结backbone的参数，使其不可训练
    def freeze_backbone(self):
        # 遍历backbone的模型参数，并设置requires_grad为False
        for name, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(False)

    # 解冻backbone的参数，使其可训练
    def unfreeze_backbone(self):
        # 遍历backbone的模型参数，并设置requires_grad为True
        for name, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(True)

    # 前向传播函数，根据DETR的输入文档字符串进行注释
    @add_start_docstrings_to_model_forward(DETR_INPUTS_DOCSTRING)
    # 替换返回的文档字符串类型为DetrModelOutput，并使用_CONFIG_FOR_DOC作为配置类
    @replace_return_docstrings(output_type=DetrModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
"""
DETR Model (consisting of a backbone and encoder-decoder Transformer) with object detection heads on top, for tasks
such as COCO detection.
"""
# 导入所需模块和函数
@add_start_docstrings(
    """
    DETR Model (consisting of a backbone and encoder-decoder Transformer) with a segmentation head on top, for tasks
    such as COCO panoptic.
    """,
    DETR_START_DOCSTRING,
)
# DETR模型的子类，用于分割任务，例如COCO全景分割
class DetrForSegmentation(DetrPreTrainedModel):
    # 使用给定的配置初始化对象检测模型
    def __init__(self, config: DetrConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建对象检测模型实例
        self.detr = DetrForObjectDetection(config)

        # 初始化分割头部
        # 从配置中获取隐藏大小和注意力头数
        hidden_size, number_of_heads = config.d_model, config.encoder_attention_heads
        # 从对象检测模型中获取中间层通道大小
        intermediate_channel_sizes = self.detr.model.backbone.conv_encoder.intermediate_channel_sizes

        # 创建小型卷积分割头部实例
        self.mask_head = DetrMaskHeadSmallConv(
            hidden_size + number_of_heads, intermediate_channel_sizes[::-1][-3:], hidden_size
        )

        # 创建 DETR 多头注意力地图实例
        self.bbox_attention = DetrMHAttentionMap(
            hidden_size, hidden_size, number_of_heads, dropout=0.0, std=config.init_xavier_std
        )

        # 执行初始化权重和最终处理
        self.post_init()

    # 将输入参数和返回值的文档字符串添加到模型的前向方法中
    @add_start_docstrings_to_model_forward(DETR_INPUTS_DOCSTRING)
    # 替换返回值的文档字符串为分割输出类型，并指定配置类
    @replace_return_docstrings(output_type=DetrSegmentationOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[List[dict]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 定义一个函数 `_expand`，用于将给定的张量在第一维度上插入新维度，然后在该维度上重复指定次数，并将结果展平。
def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)


# 从 https://github.com/facebookresearch/detr/blob/master/models/segmentation.py 中引用的代码片段
# 定义了一个名为 `DetrMaskHeadSmallConv` 的类，用于实现一个简单的卷积头部，使用组归一化。通过 FPN 方法进行上采样。
class DetrMaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm. Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()

        # 如果 `dim` 不是 8 的倍数，抛出错误，因为 GroupNorm 的组数设置为 8
        if dim % 8 != 0:
            raise ValueError(
                "The hidden_size + number of attention heads must be divisible by 8 as the number of groups in"
                " GroupNorm is set to 8"
            )

        # 定义中间层的维度列表，依次为 `dim`, `context_dim // 2`, `context_dim // 4`, `context_dim // 8`, `context_dim // 16`, `context_dim // 64`
        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]

        # 定义卷积层和组归一化层
        self.lay1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, dim)
        self.lay2 = nn.Conv2d(dim, inter_dims[1], 3, padding=1)
        self.gn2 = nn.GroupNorm(min(8, inter_dims[1]), inter_dims[1])
        self.lay3 = nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = nn.GroupNorm(min(8, inter_dims[2]), inter_dims[2])
        self.lay4 = nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = nn.GroupNorm(min(8, inter_dims[3]), inter_dims[3])
        self.lay5 = nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.gn5 = nn.GroupNorm(min(8, inter_dims[4]), inter_dims[4])
        self.out_lay = nn.Conv2d(inter_dims[4], 1, 3, padding=1)

        # 设置类属性 `dim`
        self.dim = dim

        # 适配器层，用于将 FPN 的输出适配到不同层的输入维度
        self.adapter1 = nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.adapter3 = nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

        # 初始化所有卷积层的权重和偏置
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
    # 定义前向传播函数，接受输入参数 x（特征张量）、bbox_mask（边界框掩码张量）、fpns（特征金字塔网络列表）
    def forward(self, x: Tensor, bbox_mask: Tensor, fpns: List[Tensor]):
        # 将 x（投影后的特征图，形状为 (batch_size, d_model, height/32, width/32)）与 bbox_mask（注意力映射，
        # 形状为 (batch_size, n_queries, n_heads, height/32, width/32)）拼接起来
        x = torch.cat([_expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)], 1)

        # 经过第一层线性变换层
        x = self.lay1(x)
        # 经过第一个组归一化层
        x = self.gn1(x)
        # 经过 ReLU 激活函数
        x = nn.functional.relu(x)

        # 经过第二层线性变换层
        x = self.lay2(x)
        # 经过第二个组归一化层
        x = self.gn2(x)
        # 经过 ReLU 激活函数
        x = nn.functional.relu(x)

        # 获取当前的特征金字塔网络（fpns）中的第一个子网络
        cur_fpn = self.adapter1(fpns[0])
        # 如果当前特征金字塔网络的批次数不等于 x 的批次数，则扩展它以匹配 x 的批次数
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        # 将当前特征金字塔网络的输出与 x 插值后相加
        x = cur_fpn + nn.functional.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        # 经过第三层线性变换层
        x = self.lay3(x)
        # 经过第三个组归一化层
        x = self.gn3(x)
        # 经过 ReLU 激活函数
        x = nn.functional.relu(x)

        # 获取当前的特征金字塔网络中的第二个子网络
        cur_fpn = self.adapter2(fpns[1])
        # 如果当前特征金字塔网络的批次数不等于 x 的批次数，则扩展它以匹配 x 的批次数
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        # 将当前特征金字塔网络的输出与 x 插值后相加
        x = cur_fpn + nn.functional.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        # 经过第四层线性变换层
        x = self.lay4(x)
        # 经过第四个组归一化层
        x = self.gn4(x)
        # 经过 ReLU 激活函数
        x = nn.functional.relu(x)

        # 获取当前的特征金字塔网络中的第三个子网络
        cur_fpn = self.adapter3(fpns[2])
        # 如果当前特征金字塔网络的批次数不等于 x 的批次数，则扩展它以匹配 x 的批次数
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        # 将当前特征金字塔网络的输出与 x 插值后相加
        x = cur_fpn + nn.functional.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        # 经过第五层线性变换层
        x = self.lay5(x)
        # 经过第五个组归一化层
        x = self.gn5(x)
        # 经过 ReLU 激活函数
        x = nn.functional.relu(x)

        # 经过输出层的线性变换
        x = self.out_lay(x)
        # 返回最终的输出张量
        return x
class DetrMHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True, std=None):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        # Linear transformation for queries
        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        # Linear transformation for keys
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        # Normalization factor for scaling dot products in attention calculation
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask: Optional[Tensor] = None):
        # Linear transformation of queries
        q = self.q_linear(q)
        # Convolutional transformation of keys
        k = nn.functional.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)

        # Reshape queries and keys for multi-head attention computation
        queries_per_head = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        keys_per_head = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])

        # Compute scaled dot-product attention scores
        weights = torch.einsum("bqnc,bnchw->bqnhw", queries_per_head * self.normalize_fact, keys_per_head)

        # Apply mask to attention weights if provided
        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), torch.finfo(weights.dtype).min)

        # Apply softmax to obtain attention distributions
        weights = nn.functional.softmax(weights.flatten(2), dim=-1).view(weights.size())
        # Apply dropout
        weights = self.dropout(weights)

        return weights


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs (0 for the negative class and 1 for the positive
                 class).
    """
    # Apply sigmoid function to inputs
    inputs = inputs.sigmoid()
    # Flatten the inputs
    inputs = inputs.flatten(1)
    # Compute numerator of DICE coefficient
    numerator = 2 * (inputs * targets).sum(1)
    # Compute denominator of DICE coefficient
    denominator = inputs.sum(-1) + targets.sum(-1)
    # Compute DICE loss
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (`torch.FloatTensor` of arbitrary shape):
            The predictions for each example.
        targets (`torch.FloatTensor` with the same shape as `inputs`)
            A tensor storing the binary classification label for each element in the `inputs` (0 for the negative class
            and 1 for the positive class).
        alpha (`float`, *optional*, defaults to `0.25`):
            Optional weighting factor in the range (0,1) to balance positive vs. negative examples.
        gamma (`int`, *optional*, defaults to `2`):
            Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.

    Returns:
        Loss tensor
    """
    # 对输入进行 sigmoid 操作，将输出值限制在 (0, 1) 范围内
    prob = inputs.sigmoid()
    # 使用二元交叉熵损失函数计算损失，但保留每个样本的损失值，不进行汇总
    ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # 计算 modulating factor，用于调节损失
    p_t = prob * targets + (1 - prob) * (1 - targets)
    # 计算最终的损失值，使用 focal loss 的形式进行加权
    loss = ce_loss * ((1 - p_t) ** gamma)

    # 如果 alpha 大于等于 0，则使用 focal loss 的 alpha 权重调节损失
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # 计算最终的平均损失值，并对所有样本求和，然后除以 num_boxes 得到平均损失
    return loss.mean(1).sum() / num_boxes
    """
    This class computes the losses for DetrForObjectDetection/DetrForSegmentation. The process happens in two steps: 1)
    we compute hungarian assignment between ground truth boxes and the outputs of the model 2) we supervise each pair
    of matched ground-truth / prediction (supervise class and box).

    A note on the `num_classes` argument (copied from original repo in detr.py): "the naming of the `num_classes`
    parameter of the criterion is somewhat misleading. It indeed corresponds to `max_obj_id` + 1, where `max_obj_id` is
    the maximum id for a class in your dataset. For example, COCO has a `max_obj_id` of 90, so we pass `num_classes` to
    be 91. As another example, for a dataset that has a single class with `id` 1, you should pass `num_classes` to be 2
    (`max_obj_id` + 1). For more details on this, check the following discussion
    https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223"

    Args:
        matcher (`DetrHungarianMatcher`):
            Module able to compute a matching between targets and proposals.
        num_classes (`int`):
            Number of object categories, omitting the special no-object category.
        eos_coef (`float`):
            Relative classification weight applied to the no-object category.
        losses (`List[str]`):
            List of all the losses to be applied. See `get_loss` for a list of all available losses.
    """

    def __init__(self, matcher, num_classes, eos_coef, losses):
        super().__init__()
        self.matcher = matcher  # 初始化匹配器，用于计算目标与模型输出之间的匹配
        self.num_classes = num_classes  # 目标类别数，不包括特殊的无对象类别
        self.eos_coef = eos_coef  # 无对象类别的相对分类权重
        self.losses = losses  # 待应用的所有损失列表

        # 创建一个权重张量，用于交叉熵计算，最后一个元素用于处理无对象类别
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    # removed logging parameter, which was part of the original implementation
    def loss_labels(self, outputs, targets, indices, num_boxes):
        """
        Classification loss (NLL) targets dicts must contain the key "class_labels" containing a tensor of dim
        [nb_target_boxes]
        """
        if "logits" not in outputs:
            raise KeyError("No logits were found in the outputs")
        source_logits = outputs["logits"]  # 获取模型输出的逻辑回归结果

        idx = self._get_source_permutation_idx(indices)  # 获取源排列的索引
        target_classes_o = torch.cat([t["class_labels"][J] for t, (_, J) in zip(targets, indices)])  # 获取目标类别
        target_classes = torch.full(
            source_logits.shape[:2], self.num_classes, dtype=torch.int64, device=source_logits.device
        )
        target_classes[idx] = target_classes_o  # 将目标类别放入正确的位置

        # 计算交叉熵损失
        loss_ce = nn.functional.cross_entropy(source_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}  # 存储交叉熵损失

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        Compute the cardinality error, i.e. the absolute error in the number of predicted non-empty boxes.

        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients.
        """
        # 获取模型输出中的分类 logits
        logits = outputs["logits"]
        # 获取 logits 的设备信息
        device = logits.device
        # 计算目标长度，即每个目标包含的类标签数
        target_lengths = torch.as_tensor([len(v["class_labels"]) for v in targets], device=device)
        # 计算预测中非空盒子数量
        # 非空盒子是指预测中不是“no-object”类别（即最后一个类别）的预测
        card_pred = (logits.argmax(-1) != logits.shape[-1] - 1).sum(1)
        # 计算预测盒子数量的绝对误差
        card_err = nn.functional.l1_loss(card_pred.float(), target_lengths.float())
        # 构建 losses 字典，包含基于预测盒子数量的误差
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]. The target boxes
        are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        # 检查模型输出中是否存在预测框
        if "pred_boxes" not in outputs:
            raise KeyError("No predicted boxes found in outputs")
        # 根据索引获取源排列的索引
        idx = self._get_source_permutation_idx(indices)
        # 获取预测框和目标框
        source_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # 计算边界框的 L1 回归损失
        loss_bbox = nn.functional.l1_loss(source_boxes, target_boxes, reduction="none")

        # 构建 losses 字典，包含边界框的 L1 回归损失和 GIoU 损失
        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        # 计算广义 IoU 损失
        loss_giou = 1 - torch.diag(
            generalized_box_iou(center_to_corners_format(source_boxes), center_to_corners_format(target_boxes))
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses
    def loss_masks(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.

        Targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w].
        """
        # 检查输出中是否包含预测的 masks
        if "pred_masks" not in outputs:
            raise KeyError("No predicted masks found in outputs")

        # 获取源排列索引，用于根据预测和目标的排列顺序调整预测 masks
        source_idx = self._get_source_permutation_idx(indices)
        # 获取目标排列索引
        target_idx = self._get_target_permutation_idx(indices)
        # 获取预测的 masks
        source_masks = outputs["pred_masks"]
        # 根据源排列索引选择对应的预测 masks
        source_masks = source_masks[source_idx]
        # 获取目标 masks 列表
        masks = [t["masks"] for t in targets]
        # 使用 nested_tensor_from_tensor_list 函数将目标 masks 转换为 NestedTensor，同时获取有效区域 valid
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        # 将目标 masks 转换为与预测 masks 相同的设备类型
        target_masks = target_masks.to(source_masks)
        # 根据目标排列索引选择对应的目标 masks
        target_masks = target_masks[target_idx]

        # 将预测 masks 上采样至目标大小
        source_masks = nn.functional.interpolate(
            source_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False
        )
        # 压缩维度，将预测 masks 变为一维
        source_masks = source_masks[:, 0].flatten(1)

        # 压缩维度，将目标 masks 变为一维
        target_masks = target_masks.flatten(1)
        # 将目标 masks 变换为与预测 masks 相同的形状
        target_masks = target_masks.view(source_masks.shape)

        # 计算损失，包括 sigmoid focal loss 和 dice loss
        losses = {
            "loss_mask": sigmoid_focal_loss(source_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(source_masks, target_masks, num_boxes),
        }
        return losses

    def _get_source_permutation_idx(self, indices):
        # 根据 indices 排列预测，返回批次索引和源索引
        batch_idx = torch.cat([torch.full_like(source, i) for i, (source, _) in enumerate(indices)])
        source_idx = torch.cat([source for (source, _) in indices])
        return batch_idx, source_idx

    def _get_target_permutation_idx(self, indices):
        # 根据 indices 排列目标，返回批次索引和目标索引
        batch_idx = torch.cat([torch.full_like(target, i) for i, (_, target) in enumerate(indices)])
        target_idx = torch.cat([target for (_, target) in indices])
        return batch_idx, target_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        # 定义损失函数映射
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
        }
        # 检查所请求的损失是否在损失映射中
        if loss not in loss_map:
            raise ValueError(f"Loss {loss} not supported")
        # 返回所请求损失函数的结果
        return loss_map[loss](outputs, targets, indices, num_boxes)
    def forward(self, outputs, targets):
        """
        This performs the loss computation.

        Args:
             outputs (`dict`, *optional*):
                Dictionary of tensors, see the output specification of the model for the format.
             targets (`List[dict]`, *optional*):
                List of dicts, such that `len(targets) == batch_size`. The expected keys in each dict depends on the
                losses applied, see each loss' doc.
        """
        # Exclude auxiliary outputs from the outputs dictionary
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "auxiliary_outputs"}

        # Retrieve indices that match outputs of the last layer with targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the total number of target boxes across all samples for normalization
        num_boxes = sum(len(t["class_labels"]) for t in targets)
        # Convert num_boxes to a tensor of float type, and move it to the same device as outputs
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        world_size = 1

        # Check if acceleration is available and adjust num_boxes and world_size accordingly
        if is_accelerate_available():
            # If PartialState._shared_state is not empty, reduce num_boxes
            if PartialState._shared_state != {}:
                num_boxes = reduce(num_boxes)
                # Get the number of processes from PartialState
                world_size = PartialState().num_processes

        # Normalize num_boxes considering the number of processes
        num_boxes = torch.clamp(num_boxes / world_size, min=1).item()

        # Compute losses for each specified loss function
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # If there are auxiliary outputs, compute losses for each auxiliary output separately
        if "auxiliary_outputs" in outputs:
            for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):
                indices = self.matcher(auxiliary_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        # Skip computation of masks loss for auxiliary outputs due to cost
                        continue
                    # Append index to keys in losses dictionary for each auxiliary output
                    l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # Return computed losses
        return losses
# taken from https://github.com/facebookresearch/detr/blob/master/models/detr.py
class DetrMLPPredictionHead(nn.Module):
    """
    Very simple multi-layer perceptron (MLP, also called FFN), used to predict the normalized center coordinates,
    height and width of a bounding box w.r.t. an image.

    Copied from https://github.com/facebookresearch/detr/blob/master/models/detr.py

    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        # Define a list of linear layers with ReLU activation for the MLP
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        # Feed forward through each linear layer with ReLU activation, except the last layer
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# taken from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
class DetrHungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of this, in general, there are more
    predictions than targets. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).

    Args:
        class_cost:
            The relative weight of the classification error in the matching cost.
        bbox_cost:
            The relative weight of the L1 error of the bounding box coordinates in the matching cost.
        giou_cost:
            The relative weight of the giou loss of the bounding box in the matching cost.
    """

    def __init__(self, class_cost: float = 1, bbox_cost: float = 1, giou_cost: float = 1):
        super().__init__()
        # Ensure that the "scipy" library is available when initializing this module
        requires_backends(self, ["scipy"])

        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        # Check if all costs are non-zero; raise an error if they are all zero
        if class_cost == 0 and bbox_cost == 0 and giou_cost == 0:
            raise ValueError("All costs of the Matcher can't be 0")

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Args:
            outputs (`dict`):
                A dictionary that contains at least these entries:
                * "logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                * "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates.
            targets (`List[dict]`):
                A list of targets (len(targets) = batch_size), where each target is a dict containing:
                * "class_labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of
                  ground-truth objects in the target) containing the class labels
                * "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates.

        Returns:
            `List[Tuple]`: A list of size `batch_size`, containing tuples of (index_i, index_j) where:
            - index_i is the indices of the selected predictions (in order)
            - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds: len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        # Extract batch size and number of queries from the outputs
        batch_size, num_queries = outputs["logits"].shape[:2]

        # Flatten logits and apply softmax to get probabilities over classes
        out_prob = outputs["logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]

        # Flatten predicted boxes
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Concatenate target class labels into a single tensor
        target_ids = torch.cat([v["class_labels"] for v in targets])

        # Concatenate target boxes into a single tensor
        target_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute classification cost matrix based on negative log likelihood approximation
        class_cost = -out_prob[:, target_ids]

        # Compute L1 cost matrix between predicted and target boxes
        bbox_cost = torch.cdist(out_bbox, target_bbox, p=1)

        # Compute generalized IoU cost matrix between predicted and target boxes
        giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))

        # Combine different costs into a final cost matrix using predefined weights
        cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost

        # Reshape cost matrix to batch size x num_queries x (sum of all target boxes)
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        # Split cost matrix based on number of target boxes in each sample and perform linear sum assignment
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]

        # Return indices as a list of tuples containing selected predictions and corresponding targets
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
# below: bounding box utilities taken from https://github.com/facebookresearch/detr/blob/master/util/box_ops.py

def _upcast(t: Tensor) -> Tensor:
    """
    Protects from numerical overflows in multiplications by upcasting to the equivalent higher type.

    Args:
        t (`Tensor`): The input tensor to be upcasted.

    Returns:
        `Tensor`: The upcasted tensor.
    """
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


def box_area(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, specified by (x1, y1, x2, y2) coordinates.

    Args:
        boxes (`torch.FloatTensor` of shape `(number_of_boxes, 4)`):
            Boxes for which the area will be computed. They are expected in (x1, y1, x2, y2) format with `0 <= x1 < x2` and `0 <= y1 < y2`.

    Returns:
        `torch.FloatTensor`: A tensor containing the area for each box.
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    """
    Computes the Intersection over Union (IoU) between two sets of bounding boxes.

    Args:
        boxes1 (`Tensor`): Bounding boxes in format (x1, y1, x2, y2).
        boxes2 (`Tensor`): Bounding boxes in format (x1, y1, x2, y2).

    Returns:
        `Tensor`: IoU scores for each pair of boxes.
        `Tensor`: Union area for each pair of boxes.
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    width_height = (right_bottom - left_top).clamp(min=0)  # [N,M,2]
    inter = width_height[:, :, 0] * width_height[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Computes the Generalized Intersection over Union (IoU) between two sets of bounding boxes.

    Args:
        boxes1 (`Tensor`): Bounding boxes in format [x0, y0, x1, y1].
        boxes2 (`Tensor`): Bounding boxes in format [x0, y0, x1, y1].

    Returns:
        `Tensor`: Generalized IoU scores for each pair of boxes.
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        raise ValueError(f"boxes1 must be in [x0, y0, x1, y1] (corner) format, but got {boxes1}")
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        raise ValueError(f"boxes2 must be in [x0, y0, x1, y1] (corner) format, but got {boxes2}")
    iou, union = box_iou(boxes1, boxes2)

    top_left = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    width_height = (bottom_right - top_left).clamp(min=0)  # [N,M,2]
    area = width_height[:, :, 0] * width_height[:, :, 1]

    return iou - (area - union) / area


# below: taken from https://github.com/facebookresearch/detr/blob/master/util/misc.py#L306
def _max_by_axis(the_list):
    """
    Finds the maximum value along each axis of a list of lists.

    Args:
        the_list (`List[List[int]]`): A list of lists of integers.

    Returns:
        `List[int]`: A list containing the maximum value along each axis.
    """
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    # Placeholder for further implementation, class not fully shown
    pass
    # 初始化方法，接受张量列表和可选的掩码张量作为参数
    def __init__(self, tensors, mask: Optional[Tensor]):
        # 将传入的张量列表赋值给实例变量 tensors
        self.tensors = tensors
        # 将传入的掩码张量赋值给实例变量 mask
        self.mask = mask

    # 将 NestedTensor 对象中的张量数据移动到指定的设备上
    def to(self, device):
        # 将 self.tensors 中的张量数据移动到指定设备，并赋值给 cast_tensor
        cast_tensor = self.tensors.to(device)
        # 获取实例变量 self.mask
        mask = self.mask
        # 如果 mask 不为 None
        if mask is not None:
            # 将 mask 中的数据移动到指定设备，并赋值给 cast_mask
            cast_mask = mask.to(device)
        else:
            # 如果 mask 为 None，则将 cast_mask 设置为 None
            cast_mask = None
        # 返回一个新的 NestedTensor 对象，其中的张量和掩码都已经移动到指定设备上
        return NestedTensor(cast_tensor, cast_mask)

    # 返回 NestedTensor 对象中包含的张量和掩码
    def decompose(self):
        return self.tensors, self.mask

    # 定制对象的字符串表示，返回张量的字符串表示
    def __repr__(self):
        return str(self.tensors)
# 根据给定的张量列表创建嵌套张量
def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # 检查第一个张量的维度是否为3
    if tensor_list[0].ndim == 3:
        # 计算张量列表中每个张量的最大尺寸
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # 构建批次的形状，包括批次大小、通道数、高度和宽度
        batch_shape = [len(tensor_list)] + max_size
        batch_size, num_channels, height, width = batch_shape
        # 获取张量列表中第一个张量的数据类型和设备
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        # 创建一个全零的张量，形状为批次形状，指定数据类型和设备
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        # 创建一个全一的掩码张量，形状为(batch_size, height, width)，数据类型为布尔型，设备为指定设备
        mask = torch.ones((batch_size, height, width), dtype=torch.bool, device=device)
        # 遍历张量列表中的每个张量，以及新创建的张量和掩码
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            # 将原始图像的数据复制到新创建的张量中对应的位置
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            # 将掩码中对应图像部分的值设置为False，表示实际数据存在的位置
            m[: img.shape[1], : img.shape[2]] = False
    else:
        # 如果张量维度不为3，抛出值错误异常
        raise ValueError("Only 3-dimensional tensors are supported")
    # 返回嵌套张量对象，包括数据张量和掩码张量
    return NestedTensor(tensor, mask)
```