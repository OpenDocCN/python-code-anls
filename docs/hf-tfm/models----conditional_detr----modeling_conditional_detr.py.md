# `.\models\conditional_detr\modeling_conditional_detr.py`

```
# coding=utf-8
# Copyright 2022 Microsoft Research Asia and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch Conditional DETR model."""

# 导入所需的库和模块
import math  # 导入数学函数库
from dataclasses import dataclass  # 导入 dataclass 装饰器，用于定义数据类
from typing import Dict, List, Optional, Tuple, Union  # 导入类型提示相关模块

import torch  # 导入 PyTorch 库
from torch import Tensor, nn  # 导入张量和神经网络相关模块

# 导入辅助功能模块和实用工具
from ...activations import ACT2FN  # 导入激活函数映射
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask  # 导入用于准备4D注意力掩码的函数
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithCrossAttentions, Seq2SeqModelOutput  # 导入模型输出相关类
from ...modeling_utils import PreTrainedModel  # 导入预训练模型基类
from ...utils import (  # 导入各种实用函数和类
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_accelerate_available,
    is_scipy_available,
    is_timm_available,
    is_vision_available,
    logging,
    replace_return_docstrings,
    requires_backends,
)
from ...utils.backbone_utils import load_backbone  # 导入加载骨干网络的函数
from .configuration_conditional_detr import ConditionalDetrConfig  # 导入 Conditional DETR 模型配置类


if is_accelerate_available():  # 如果加速库可用
    from accelerate import PartialState  # 导入 PartialState 类
    from accelerate.utils import reduce  # 导入 reduce 函数

if is_scipy_available():  # 如果 scipy 库可用
    from scipy.optimize import linear_sum_assignment  # 导入 linear_sum_assignment 函数

if is_timm_available():  # 如果 timm 库可用
    from timm import create_model  # 导入 create_model 函数

if is_vision_available():  # 如果视觉库可用
    from ...image_transforms import center_to_corners_format  # 导入将中心格式转换为角点格式的函数

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

_CONFIG_FOR_DOC = "ConditionalDetrConfig"  # 用于文档的配置信息
_CHECKPOINT_FOR_DOC = "microsoft/conditional-detr-resnet-50"  # 用于文档的检查点信息

# 预训练模型存档列表
CONDITIONAL_DETR_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/conditional-detr-resnet-50",
    # 查看所有 Conditional DETR 模型：https://huggingface.co/models?filter=conditional_detr
]


@dataclass
class ConditionalDetrDecoderOutput(BaseModelOutputWithCrossAttentions):
    """
    Base class for outputs of the Conditional DETR decoder. This class adds one attribute to
    BaseModelOutputWithCrossAttentions, namely an optional stack of intermediate decoder activations, i.e. the output
    of each decoder layer, each of them gone through a layernorm. This is useful when training the model with auxiliary
    decoding losses.
    """
    pass  # 条件 DETR 解码器输出的基类，添加了中间解码器激活堆栈属性
    # 定义函数的参数及其类型说明
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            最后一个模型层的隐藏状态序列输出。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            包含模型每一层隐藏状态的元组，形状为 `(batch_size, sequence_length, hidden_size)`。
            当 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            注意力权重的元组，每个元素对应每一层的注意力权重。
            形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
            在自注意力头中用于计算加权平均后返回，当 `output_attentions=True` 或 `config.output_attentions=True` 时返回。
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            交叉注意力层的注意力权重元组，每个元素对应解码器交叉注意力层的注意力权重。
            形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
            在交叉注意力头中用于计算加权平均后返回，当同时设置了 `output_attentions=True` 和 `config.add_cross_attention=True` 或者 `config.output_attentions=True` 时返回。
        intermediate_hidden_states (`torch.FloatTensor` of shape `(config.decoder_layers, batch_size, num_queries, hidden_size)`, *optional*, returned when `config.auxiliary_loss=True`):
            中间解码器激活状态，即每个解码器层的输出，经过层归一化后的结果。
            形状为 `(config.decoder_layers, batch_size, num_queries, hidden_size)`。
            当设置了 `config.auxiliary_loss=True` 时返回。
@dataclass
class ConditionalDetrModelOutput(Seq2SeqModelOutput):
    """
    ConditionalDetr 模型的输出基类。添加了一个额外的属性 intermediate_hidden_states，
    可选地包含中间解码器激活的堆栈，即每个解码器层的输出，经过 layernorm 处理。
    在训练模型时使用辅助解码损失时非常有用。
    """

    intermediate_hidden_states: Optional[torch.FloatTensor] = None
    reference_points: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class ConditionalDetrObjectDetectionOutput(ModelOutput):
    """
    ConditionalDetr 对象检测模型的输出类型。

    """

    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict] = None
    logits: torch.FloatTensor = None
    pred_boxes: torch.FloatTensor = None
    auxiliary_outputs: Optional[List[Dict]] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class ConditionalDetrSegmentationOutput(ModelOutput):
    """
    ConditionalDetr 分割模型的输出类型。

    """

    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict] = None
    logits: torch.FloatTensor = None
    pred_boxes: torch.FloatTensor = None
    pred_masks: torch.FloatTensor = None
    auxiliary_outputs: Optional[List[Dict]] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


class ConditionalDetrFrozenBatchNorm2d(nn.Module):
    """
    ConditionalDetr 的冻结批量归一化层。

    BatchNorm2d 的批次统计信息和仿射参数被固定的版本。
    从 torchvision.misc.ops 中复制粘贴而来，添加了在求平方根前的 eps，
    否则除 torchvision.models.resnet[18,34,50,101] 之外的其他模型会产生 NaN 值。
    """
    # 初始化函数，用于创建一个新的实例
    def __init__(self, n):
        # 调用父类的初始化方法
        super().__init__()
        # 注册一个名为 "weight" 的缓冲区，包含了大小为 n 的全为 1 的张量
        self.register_buffer("weight", torch.ones(n))
        # 注册一个名为 "bias" 的缓冲区，包含了大小为 n 的全为 0 的张量
        self.register_buffer("bias", torch.zeros(n))
        # 注册一个名为 "running_mean" 的缓冲区，包含了大小为 n 的全为 0 的张量
        self.register_buffer("running_mean", torch.zeros(n))
        # 注册一个名为 "running_var" 的缓冲区，包含了大小为 n 的全为 1 的张量
        self.register_buffer("running_var", torch.ones(n))

    # 加载模型状态字典的私有方法
    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # 构建 num_batches_tracked 对应的键
        num_batches_tracked_key = prefix + "num_batches_tracked"
        # 如果 state_dict 中存在 num_batches_tracked_key，则将其删除
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        # 调用父类的 _load_from_state_dict 方法来加载状态字典
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    # 前向传播函数
    def forward(self, x):
        # 将 weight 重塑为大小为 (1, n, 1, 1) 的张量，以便与输入 x 的形状兼容
        weight = self.weight.reshape(1, -1, 1, 1)
        # 将 bias 重塑为大小为 (1, n, 1, 1) 的张量，以便与输入 x 的形状兼容
        bias = self.bias.reshape(1, -1, 1, 1)
        # 将 running_var 重塑为大小为 (1, n, 1, 1) 的张量，以便与输入 x 的形状兼容
        running_var = self.running_var.reshape(1, -1, 1, 1)
        # 将 running_mean 重塑为大小为 (1, n, 1, 1) 的张量，以便与输入 x 的形状兼容
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        # 定义一个极小值 epsilon 用于稳定计算
        epsilon = 1e-5
        # 计算缩放因子，用于标准化输入 x
        scale = weight * (running_var + epsilon).rsqrt()
        # 根据标准化的结果，调整偏置 bias
        bias = bias - running_mean * scale
        # 返回经过标准化和调整的输入 x
        return x * scale + bias
# Copied from transformers.models.detr.modeling_detr.replace_batch_norm with Detr->ConditionalDetr
def replace_batch_norm(model):
    r"""
    递归地将模型中所有的 `torch.nn.BatchNorm2d` 替换为 `ConditionalDetrFrozenBatchNorm2d`。

    Args:
        model (torch.nn.Module):
            输入的模型
    """
    # 遍历模型的每个子模块
    for name, module in model.named_children():
        # 如果当前模块是 `nn.BatchNorm2d` 类型
        if isinstance(module, nn.BatchNorm2d):
            # 创建新的 `ConditionalDetrFrozenBatchNorm2d` 模块
            new_module = ConditionalDetrFrozenBatchNorm2d(module.num_features)

            # 如果原始的 BatchNorm 参数不在设备 "meta" 上
            if not module.weight.device == torch.device("meta"):
                # 复制参数到新模块中
                new_module.weight.data.copy_(module.weight)
                new_module.bias.data.copy_(module.bias)
                new_module.running_mean.data.copy_(module.running_mean)
                new_module.running_var.data.copy_(module.running_var)

            # 将模型中的当前子模块替换为新模块
            model._modules[name] = new_module

        # 如果当前模块还有子模块，则递归调用替换函数
        if len(list(module.children())) > 0:
            replace_batch_norm(module)


# Copied from transformers.models.detr.modeling_detr.DetrConvEncoder
class ConditionalDetrConvEncoder(nn.Module):
    """
    使用 AutoBackbone API 或 timm 库中的模型作为卷积主干网络。

    所有的 nn.BatchNorm2d 层都被上述定义的 DetrFrozenBatchNorm2d 替换。
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        # 根据配置选择使用 timm 提供的模型或自定义的模型加载
        if config.use_timm_backbone:
            # 确保依赖于 timm 库的模型加载
            requires_backends(self, ["timm"])
            kwargs = {}
            # 如果配置中指定了 dilation，则设置输出步幅为 16
            if config.dilation:
                kwargs["output_stride"] = 16
            # 创建 timm 模型，仅输出特征，指定需要的层
            backbone = create_model(
                config.backbone,
                pretrained=config.use_pretrained_backbone,
                features_only=True,
                out_indices=(1, 2, 3, 4),
                in_chans=config.num_channels,
                **kwargs,
            )
        else:
            # 根据配置加载自定义的模型
            backbone = load_backbone(config)

        # 使用 `replace_batch_norm` 函数替换所有的 BatchNorm 层为 Frozen BatchNorm
        with torch.no_grad():
            replace_batch_norm(backbone)

        # 将处理过的模型设置为当前类的模型属性
        self.model = backbone
        # 根据选择的主干网络类型确定中间特征通道数
        self.intermediate_channel_sizes = (
            self.model.feature_info.channels() if config.use_timm_backbone else self.model.channels
        )

        # 根据主干网络类型和配置冻结不需要训练的参数
        backbone_model_type = config.backbone if config.use_timm_backbone else config.backbone_config.model_type
        if "resnet" in backbone_model_type:
            for name, parameter in self.model.named_parameters():
                if config.use_timm_backbone:
                    if "layer2" not in name and "layer3" not in name and "layer4" not in name:
                        parameter.requires_grad_(False)
                else:
                    if "stage.1" not in name and "stage.2" not in name and "stage.3" not in name:
                        parameter.requires_grad_(False)
    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor):
        # 将像素值通过模型以获取特征图列表
        features = self.model(pixel_values) if self.config.use_timm_backbone else self.model(pixel_values).feature_maps

        out = []
        for feature_map in features:
            # 将像素掩码下采样以匹配对应特征图的形状
            mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
            out.append((feature_map, mask))
        # 返回包含特征图和掩码元组的列表
        return out
# Copied from transformers.models.detr.modeling_detr.DetrConvModel with Detr->ConditionalDetr
class ConditionalDetrConvModel(nn.Module):
    """
    This module adds 2D position embeddings to all intermediate feature maps of the convolutional encoder.
    """

    def __init__(self, conv_encoder, position_embedding):
        super().__init__()
        self.conv_encoder = conv_encoder  # 初始化卷积编码器
        self.position_embedding = position_embedding  # 初始化位置嵌入模块

    def forward(self, pixel_values, pixel_mask):
        # 将像素值和像素掩码通过骨干网络，获取包含(feature_map, pixel_mask)元组的列表
        out = self.conv_encoder(pixel_values, pixel_mask)
        pos = []
        for feature_map, mask in out:
            # 位置编码
            pos.append(self.position_embedding(feature_map, mask).to(feature_map.dtype))

        return out, pos


class ConditionalDetrSinePositionEmbedding(nn.Module):
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
            raise ValueError("No pixel mask provided")  # 如果没有提供像素掩码则报错
        y_embed = pixel_mask.cumsum(1, dtype=torch.float32)  # 沿着y轴累积求和
        x_embed = pixel_mask.cumsum(2, dtype=torch.float32)  # 沿着x轴累积求和
        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * self.scale  # 归一化y嵌入
            x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * self.scale  # 归一化x嵌入

        dim_t = torch.arange(self.embedding_dim, dtype=torch.int64, device=pixel_values.device).float()
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.embedding_dim)  # 温度调整

        pos_x = x_embed[:, :, :, None] / dim_t  # 计算x位置编码
        pos_y = y_embed[:, :, :, None] / dim_t  # 计算y位置编码
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)  # 堆叠并展平x位置编码
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)  # 堆叠并展平y位置编码
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # 拼接位置编码并进行维度置换
        return pos


# Copied from transformers.models.detr.modeling_detr.DetrLearnedPositionEmbedding with Detr->ConditionalDetr
class ConditionalDetrLearnedPositionEmbedding(nn.Module):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """
    # 初始化函数，设置嵌入维度，默认为256
    def __init__(self, embedding_dim=256):
        # 调用父类初始化方法
        super().__init__()
        # 创建行坐标的嵌入层，将50个坐标映射到指定维度的向量空间
        self.row_embeddings = nn.Embedding(50, embedding_dim)
        # 创建列坐标的嵌入层，将50个坐标映射到指定维度的向量空间
        self.column_embeddings = nn.Embedding(50, embedding_dim)

    # 前向传播函数，接收像素值和可能的掩码
    def forward(self, pixel_values, pixel_mask=None):
        # 获取像素值的高度和宽度
        height, width = pixel_values.shape[-2:]
        # 创建一个张量，包含从0到width-1的列坐标，设备与像素值相同
        width_values = torch.arange(width, device=pixel_values.device)
        # 创建一个张量，包含从0到height-1的行坐标，设备与像素值相同
        height_values = torch.arange(height, device=pixel_values.device)
        # 使用列坐标嵌入层获取宽度方向的嵌入表示
        x_emb = self.column_embeddings(width_values)
        # 使用行坐标嵌入层获取高度方向的嵌入表示
        y_emb = self.row_embeddings(height_values)
        # 将列坐标和行坐标的嵌入表示拼接起来，形成位置编码
        pos = torch.cat([x_emb.unsqueeze(0).repeat(height, 1, 1), y_emb.unsqueeze(1).repeat(1, width, 1)], dim=-1)
        # 将位置编码的维度顺序调整为(2, height, width)
        pos = pos.permute(2, 0, 1)
        # 在最前面添加一个维度，将其形状变为(1, 2, height, width)
        pos = pos.unsqueeze(0)
        # 将位置编码复制为与输入像素值相同数量的批次，形状变为(batch_size, 2, height, width)
        pos = pos.repeat(pixel_values.shape[0], 1, 1, 1)
        # 返回包含位置编码的张量作为输出
        return pos
# 从transformers.models.detr.modeling_detr.build_position_encoding复制，并将Detr更改为ConditionalDetr
def build_position_encoding(config):
    # 根据配置计算位置编码的步数
    n_steps = config.d_model // 2
    # 根据位置编码类型选择不同的位置编码方式
    if config.position_embedding_type == "sine":
        # TODO 找到更好的方法来暴露其他参数
        # 使用ConditionalDetrSinePositionEmbedding创建正弦位置编码
        position_embedding = ConditionalDetrSinePositionEmbedding(n_steps, normalize=True)
    elif config.position_embedding_type == "learned":
        # 使用ConditionalDetrLearnedPositionEmbedding创建学习位置编码
        position_embedding = ConditionalDetrLearnedPositionEmbedding(n_steps)
    else:
        # 若位置编码类型不支持，则抛出错误
        raise ValueError(f"Not supported {config.position_embedding_type}")

    # 返回位置编码对象
    return position_embedding


# 用于生成二维坐标正弦位置编码的函数
def gen_sine_position_embeddings(pos_tensor, d_model):
    # 正弦函数的缩放因子
    scale = 2 * math.pi
    # 将模型维度除以2，得到每个坐标的维度
    dim = d_model // 2
    # 创建一个张量，表示维度信息
    dim_t = torch.arange(dim, dtype=torch.float32, device=pos_tensor.device)
    # 计算不同维度的缩放因子
    dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / dim)
    # 对输入的位置张量进行缩放
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    # 计算位置编码
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    # 使用正弦和余弦函数来编码位置
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    # 拼接编码后的位置张量
    pos = torch.cat((pos_y, pos_x), dim=2)
    # 返回生成的位置编码张量
    return pos


# 多头注意力机制，从transformers.models.detr.modeling_detr.DetrAttention复制
class DetrAttention(nn.Module):
    """
    来自《Attention Is All You Need》论文的多头注意力机制。

    在这里，我们根据DETR论文的解释，将位置编码添加到查询和键中。
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        # 设置注意力机制的参数
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        # 检查是否能够正确划分维度
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim必须能被num_heads整除 (得到 `embed_dim`: {self.embed_dim} 和 `num_heads`: {num_heads})."
            )
        # 缩放因子
        self.scaling = self.head_dim**-0.5

        # 线性映射，用于查询、键、值以及输出
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    # 重新整形张量形状以适应多头注意力机制
    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    # 定义一个方法，用于在输入张量中添加位置嵌入或对象查询
    def with_pos_embed(self, tensor: torch.Tensor, object_queries: Optional[Tensor], **kwargs):
        # 从kwargs中取出名为"position_embeddings"的参数
        position_embeddings = kwargs.pop("position_embeddings", None)

        # 如果kwargs中还有其他未知参数，则抛出异常
        if kwargs:
            raise ValueError(f"Unexpected arguments {kwargs.keys()}")

        # 如果同时指定了position_embeddings和object_queries，则抛出异常
        if position_embeddings is not None and object_queries is not None:
            raise ValueError(
                "Cannot specify both position_embeddings and object_queries. Please use just object_queries"
            )

        # 如果指定了position_embeddings，则发出警告并使用object_queries替代
        if position_embeddings is not None:
            logger.warning_once(
                "position_embeddings has been deprecated and will be removed in v4.34. Please use object_queries instead"
            )
            object_queries = position_embeddings

        # 返回原始张量或者原始张量加上对象查询（位置嵌入）
        return tensor if object_queries is None else tensor + object_queries

    # 定义一个前向传播方法，接收多个输入参数，并返回处理后的结果
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
# 定义了一个名为 ConditionalDetrAttention 的 PyTorch 模块，用于实现条件 DETR 模型中的交叉注意力机制。
class ConditionalDetrAttention(nn.Module):
    """
    Cross-Attention used in Conditional DETR 'Conditional DETR for Fast Training Convergence' paper.

    The key q_proj, k_proj, v_proj are defined outside the attention. This attention allows the dim of q, k to be
    different to v.
    """

    # 初始化函数，设置模块的参数和层
    def __init__(
        self,
        embed_dim: int,                   # 输入的嵌入维度
        out_dim: int,                     # 输出的维度
        num_heads: int,                   # 注意力头的数量
        dropout: float = 0.0,             # Dropout 概率，默认为 0.0
        bias: bool = True,                # 是否使用偏置
    ):
        super().__init__()
        self.embed_dim = embed_dim        # 设置输入的嵌入维度
        self.out_dim = out_dim            # 设置输出的维度
        self.num_heads = num_heads        # 设置注意力头的数量
        self.dropout = dropout            # 设置 Dropout 概率
        self.head_dim = embed_dim // num_heads  # 计算每个注意力头的维度
        # 检查 embed_dim 必须被 num_heads 整除，否则抛出异常
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {num_heads})."
            )
        # 计算值的每个注意力头的维度
        self.v_head_dim = out_dim // num_heads
        # 检查 out_dim 必须被 num_heads 整除，否则抛出异常
        if self.v_head_dim * num_heads != self.out_dim:
            raise ValueError(
                f"out_dim must be divisible by num_heads (got `out_dim`: {self.out_dim} and `num_heads`: {num_heads})."
            )
        # 缩放因子，用于缩放注意力分数
        self.scaling = self.head_dim**-0.5

        # 输出投影层，将注意力计算后的结果映射到指定维度的空间
        self.out_proj = nn.Linear(out_dim, out_dim, bias=bias)

    # 辅助函数，用于整理输入张量的形状以便与注意力计算兼容
    def _qk_shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 辅助函数，用于整理值张量的形状以便与注意力计算兼容
    def _v_shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.v_head_dim).transpose(1, 2).contiguous()

    # 前向传播函数，实现注意力计算和投影映射
    def forward(
        self,
        hidden_states: torch.Tensor,                  # 输入的隐藏状态张量
        attention_mask: Optional[torch.Tensor] = None, # 注意力遮罩张量（可选）
        key_states: Optional[torch.Tensor] = None,     # 键张量（可选）
        value_states: Optional[torch.Tensor] = None,   # 值张量（可选）
        output_attentions: bool = False,               # 是否输出注意力权重（默认为 False）
# Copied from transformers.models.detr.modeling_detr.DetrEncoderLayer with DetrEncoderLayer->ConditionalDetrEncoderLayer,DetrConfig->ConditionalDetrConfig
# 来自 transformers.models.detr.modeling_detr.DetrEncoderLayer 的复制，将 DetrEncoderLayer 改为 ConditionalDetrEncoderLayer，DetrConfig 改为 ConditionalDetrConfig
class ConditionalDetrEncoderLayer(nn.Module):
    # 初始化函数，根据给定的配置创建层
    def __init__(self, config: ConditionalDetrConfig):
        super().__init__()
        self.embed_dim = config.d_model                          # 设置嵌入维度
        # 自注意力层，使用给定配置中的参数初始化
        self.self_attn = DetrAttention(
            embed_dim=self.embed_dim,                           # 设置嵌入维度
            num_heads=config.encoder_attention_heads,            # 设置注意力头的数量
            dropout=config.attention_dropout,                   # 设置注意力 dropout 概率
        )
        # 自注意力层的 LayerNorm 层，标准化输入
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout                           # 设置 dropout 概率
        # 激活函数，使用给定配置中的激活函数
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout     # 设置激活函数 dropout 概率
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)  # 第一个全连接层
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)  # 第二个全连接层
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)       # 最终输出的 LayerNorm 层
    # 定义模型的前向传播函数，接收输入的隐藏状态、注意力掩码和对象查询（可选），并可能返回注意力张量
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        object_queries: torch.Tensor = None,
        output_attentions: bool = False,
        **kwargs,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): 输入层的张量，形状为 `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): 注意力掩码张量，形状为
                `(batch, 1, target_len, source_len)`，其中填充元素由非常大的负值表示。
            object_queries (`torch.FloatTensor`, *optional*):
                对象查询（也称为内容嵌入），将添加到隐藏状态中。
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的注意力张量。有关更多细节，请参见返回张量中的 `attentions`。

        """

        # 从 `kwargs` 中弹出 `position_embeddings`，如果存在的话
        position_embeddings = kwargs.pop("position_embeddings", None)

        # 如果 `kwargs` 不为空，则抛出错误，不允许有未知的额外参数
        if kwargs:
            raise ValueError(f"Unexpected arguments {kwargs.keys()}")

        # 如果同时指定了 `position_embeddings` 和 `object_queries`，则抛出错误
        if position_embeddings is not None and object_queries is not None:
            raise ValueError(
                "Cannot specify both position_embeddings and object_queries. Please use just object_queries"
            )

        # 如果存在 `position_embeddings`，发出警告并使用 `object_queries` 替代
        if position_embeddings is not None:
            logger.warning_once(
                "position_embeddings has been deprecated and will be removed in v4.34. Please use object_queries instead"
            )
            object_queries = position_embeddings

        # 保存初始的隐藏状态作为残差连接的一部分
        residual = hidden_states

        # 使用自注意力机制进行前向传播，计算新的隐藏状态和可能的注意力权重
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            object_queries=object_queries,
            output_attentions=output_attentions,
        )

        # 对新的隐藏状态应用 dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # 将残差连接应用到新的隐藏状态上
        hidden_states = residual + hidden_states

        # 对连接后的隐藏状态应用层归一化
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 保存新的隐藏状态作为残差连接的一部分
        residual = hidden_states

        # 应用激活函数和线性变换 fc1 到隐藏状态
        hidden_states = self.activation_fn(self.fc1(hidden_states))

        # 对经过 fc1 处理后的隐藏状态应用 dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)

        # 再次应用线性变换 fc2 到隐藏状态
        hidden_states = self.fc2(hidden_states)

        # 对经过 fc2 处理后的隐藏状态应用 dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # 将残差连接应用到新的隐藏状态上
        hidden_states = residual + hidden_states

        # 对连接后的隐藏状态应用最终的层归一化
        hidden_states = self.final_layer_norm(hidden_states)

        # 如果处于训练模式下，检查隐藏状态是否包含无穷大或 NaN 值，进行裁剪处理
        if self.training:
            if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # 准备输出结果，仅包含隐藏状态
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则将它们加入到输出中
        if output_attentions:
            outputs += (attn_weights,)

        # 返回最终的输出结果
        return outputs
# 定义一个名为 ConditionalDetrDecoderLayer 的类，继承自 nn.Module 类
class ConditionalDetrDecoderLayer(nn.Module):
    # 初始化方法，接受一个配置对象 config: ConditionalDetrConfig
    def __init__(self, config: ConditionalDetrConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 设置类属性 embed_dim 为 config.d_model
        self.embed_dim = config.d_model

        d_model = config.d_model
        # 初始化 Decoder Self-Attention 的投影层
        self.sa_qcontent_proj = nn.Linear(d_model, d_model)
        self.sa_qpos_proj = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model)
        self.sa_kpos_proj = nn.Linear(d_model, d_model)
        self.sa_v_proj = nn.Linear(d_model, d_model)

        # 创建 ConditionalDetrAttention 对象并赋值给 self.self_attn 属性
        self.self_attn = ConditionalDetrAttention(
            embed_dim=self.embed_dim,
            out_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        # 设置类属性 dropout 为 config.dropout
        self.dropout = config.dropout
        # 根据配置选择激活函数，并赋值给 self.activation_fn 属性
        self.activation_fn = ACT2FN[config.activation_function]
        # 设置类属性 activation_dropout 为 config.activation_dropout
        self.activation_dropout = config.activation_dropout

        # 初始化 Decoder Self-Attention 的 LayerNorm
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # 初始化 Decoder Cross-Attention 的投影层
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)

        # 创建 ConditionalDetrAttention 对象并赋值给 self.encoder_attn 属性
        self.encoder_attn = ConditionalDetrAttention(
            self.embed_dim * 2, self.embed_dim, config.decoder_attention_heads, dropout=config.attention_dropout
        )
        # 初始化 Decoder Cross-Attention 的 LayerNorm
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 初始化第一个全连接层
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        # 初始化第二个全连接层
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        # 初始化最终的 LayerNorm
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        # 设置类属性 nhead 为 config.decoder_attention_heads

    # 前向传播方法，接受一些输入参数，并返回计算结果
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        object_queries: Optional[torch.Tensor] = None,
        query_position_embeddings: Optional[torch.Tensor] = None,
        query_sine_embed: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        is_first: Optional[bool] = False,
        **kwargs,
# 定义一个名为 ConditionalDetrClassificationHead 的类，继承自 nn.Module 类
class ConditionalDetrClassificationHead(nn.Module):
    # 初始化方法，接受输入维度 input_dim，内部维度 inner_dim，类别数量 num_classes，以及池化层的 dropout 比例 pooler_dropout
    def __init__(self, input_dim: int, inner_dim: int, num_classes: int, pooler_dropout: float):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化一个全连接层，输入维度为 input_dim，输出维度为 inner_dim
        self.dense = nn.Linear(input_dim, inner_dim)
        # 初始化一个 dropout 层，dropout 比例为 pooler_dropout
        self.dropout = nn.Dropout(p=pooler_dropout)
        # 初始化一个全连接层，输入维度为 inner_dim，输出维度为 num_classes
        self.out_proj = nn.Linear(inner_dim, num_classes)
    # 对输入的隐藏状态进行 dropout 操作，以减少过拟合风险
    hidden_states = self.dropout(hidden_states)
    # 将经过 dropout 后的隐藏状态输入全连接层 dense，进行线性变换
    hidden_states = self.dense(hidden_states)
    # 对全连接层的输出应用双曲正切函数，引入非线性映射
    hidden_states = torch.tanh(hidden_states)
    # 再次对处理后的隐藏状态进行 dropout 操作，进一步减少过拟合
    hidden_states = self.dropout(hidden_states)
    # 最终通过输出投影层 out_proj 得到最终的隐藏状态表示
    hidden_states = self.out_proj(hidden_states)
    # 返回经过全连接层和激活函数处理后的隐藏状态
    return hidden_states
# 从 transformers.models.detr.modeling_detr.DetrMLPPredictionHead 复制的 MLP 类，修改了类名为 MLP
class MLP(nn.Module):
    """
    非常简单的多层感知机（MLP，也称为前馈神经网络），用于预测相对于图像的归一化中心坐标、高度和宽度的边界框。

    从 https://github.com/facebookresearch/detr/blob/master/models/detr.py 复制而来
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        # 创建多个线性层组成的层序列
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            # 对每一层应用 ReLU 激活函数，最后一层不应用激活函数
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# 从 transformers.models.detr.modeling_detr.DetrPreTrainedModel 复制的 ConditionalDetrPreTrainedModel 类
# 修改了类名为 ConditionalDetrPreTrainedModel
class ConditionalDetrPreTrainedModel(PreTrainedModel):
    # 使用 ConditionalDetrConfig 作为配置类
    config_class = ConditionalDetrConfig
    # 模型的基础名称前缀为 "model"
    base_model_prefix = "model"
    # 主要输入名称为 "pixel_values"
    main_input_name = "pixel_values"
    # 不分割的模块名称列表，用于模型初始化权重时的特殊处理
    _no_split_modules = [r"ConditionalDetrConvEncoder", r"ConditionalDetrEncoderLayer", r"ConditionalDetrDecoderLayer"]

    def _init_weights(self, module):
        # 获取初始化的标准差和 Xaiver 初始化的标准差
        std = self.config.init_std
        xavier_std = self.config.init_xavier_std

        if isinstance(module, ConditionalDetrMHAttentionMap):
            # 初始化条件化的 MH Attention Map 中的偏置为零，权重使用 Xaiver 初始化
            nn.init.zeros_(module.k_linear.bias)
            nn.init.zeros_(module.q_linear.bias)
            nn.init.xavier_uniform_(module.k_linear.weight, gain=xavier_std)
            nn.init.xavier_uniform_(module.q_linear.weight, gain=xavier_std)
        elif isinstance(module, ConditionalDetrLearnedPositionEmbedding):
            # 均匀分布初始化位置嵌入的行和列权重
            nn.init.uniform_(module.row_embeddings.weight)
            nn.init.uniform_(module.column_embeddings.weight)
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            # 略微不同于 TF 版本的初始化方法，这里使用正态分布初始化权重，偏置初始化为零
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 正态分布初始化嵌入权重
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


CONDITIONAL_DETR_START_DOCSTRING = r"""
    该模型继承自 [`PreTrainedModel`]。查看超类文档以了解库为其所有模型实现的通用方法（如下载或保存、调整输入嵌入、修剪头部等）。

    该模型还是一个 PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) 的子类。
    将其视为常规的 PyTorch 模块，并参考 PyTorch 文档以获取所有与一般用法相关的信息
"""
    and behavior.

    Parameters:
        config ([`ConditionalDetrConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
# 从transformers.models.detr.modeling_detr.DetrEncoder复制并修改为ConditionalDetrEncoder，继承自ConditionalDetrPreTrainedModel
class ConditionalDetrEncoder(ConditionalDetrPreTrainedModel):
    """
    Transformer编码器，包含config.encoder_layers个自注意力层。每一层是一个ConditionalDetrEncoderLayer。

    编码器通过多个自注意力层更新扁平化特征图。

    对于ConditionalDETR的小调整：
    - 对象查询（object_queries）在前向传播中添加。
    """
    Args:
        config: ConditionalDetrConfig
    """

    # 初始化方法，接收一个配置对象，设置网络的参数和层
    def __init__(self, config: ConditionalDetrConfig):
        # 调用父类的初始化方法，传入配置对象
        super().__init__(config)

        # 从配置对象中获取dropout和encoder_layerdrop参数
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        # 使用列表推导式创建一个包含多个ConditionalDetrEncoderLayer对象的ModuleList
        self.layers = nn.ModuleList([ConditionalDetrEncoderLayer(config) for _ in range(config.encoder_layers)])

        # 在原始的ConditionalDETR中，encoder的末尾没有使用layernorm，因为默认情况下"normalize_before"设置为False

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法定义，接收多个输入参数和关键字参数
    def forward(
        self,
        inputs_embeds=None,
        attention_mask=None,
        object_queries=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
class ConditionalDetrDecoder(ConditionalDetrPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`ConditionalDetrDecoderLayer`].

    The decoder updates the query embeddings through multiple self-attention and cross-attention layers.

    Some small tweaks for Conditional DETR:

    - object_queries and query_position_embeddings are added to the forward pass.
    - if self.config.auxiliary_loss is set to True, also returns a stack of activations from all decoding layers.

    Args:
        config: ConditionalDetrConfig
    """

    def __init__(self, config: ConditionalDetrConfig):
        super().__init__(config)
        self.dropout = config.dropout  # 初始化dropout比率
        self.layerdrop = config.decoder_layerdrop  # 初始化层间dropout比率

        # 创建多层Transformer解码器，每层为ConditionalDetrDecoderLayer类的实例
        self.layers = nn.ModuleList([ConditionalDetrDecoderLayer(config) for _ in range(config.decoder_layers)])
        
        # Conditional DETR中，在最后一个解码器层输出后使用layernorm
        self.layernorm = nn.LayerNorm(config.d_model)
        d_model = config.d_model
        self.gradient_checkpointing = False  # 梯度检查点设为False

        # query_scale是应用于f以生成变换T的前馈神经网络
        self.query_scale = MLP(d_model, d_model, d_model, 2)  # 初始化query_scale网络
        self.ref_point_head = MLP(d_model, d_model, 2, 2)  # 初始化ref_point_head网络
        
        # 对于除最后一层以外的每一层，设置ca_qpos_proj为None
        for layer_id in range(config.decoder_layers - 1):
            self.layers[layer_id + 1].ca_qpos_proj = None

        # 初始化权重并应用最终处理
        self.post_init()

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
    ):
        # 省略了forward方法的注释，因为forward方法的详细解释不应该包含在代码块内部，只需提供类的初始化和重要属性的解释即可。



@add_start_docstrings(
    """
    The bare Conditional DETR Model (consisting of a backbone and encoder-decoder Transformer) outputting raw
    hidden-states without any specific head on top.
    """,
    CONDITIONAL_DETR_START_DOCSTRING,
)
class ConditionalDetrModel(ConditionalDetrPreTrainedModel):
    def __init__(self, config: ConditionalDetrConfig):
        super().__init__(config)

        # 创建骨干网络(backbone) + 位置编码
        backbone = ConditionalDetrConvEncoder(config)
        object_queries = build_position_encoding(config)
        self.backbone = ConditionalDetrConvModel(backbone, object_queries)

        # 创建投影层
        self.input_projection = nn.Conv2d(backbone.intermediate_channel_sizes[-1], config.d_model, kernel_size=1)

        self.query_position_embeddings = nn.Embedding(config.num_queries, config.d_model)

        self.encoder = ConditionalDetrEncoder(config)
        self.decoder = ConditionalDetrDecoder(config)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_encoder(self):
        return self.encoder



# 注释结束
    # 返回当前对象的解码器
    def get_decoder(self):
        return self.decoder

    # 冻结模型的主干网络，使其参数不再更新
    def freeze_backbone(self):
        # 遍历主干网络的所有参数，并设置它们的梯度更新为 False
        for name, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(False)

    # 解冻模型的主干网络，使其参数可以更新
    def unfreeze_backbone(self):
        # 遍历主干网络的所有参数，并设置它们的梯度更新为 True
        for name, param in self.backbone.conv_encoder.model.named_parameters():
            param.requires_grad_(True)

    # 前向传播函数，接受输入并返回模型的输出
    @add_start_docstrings_to_model_forward(CONDITIONAL_DETR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ConditionalDetrModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
"""
CONDITIONAL_DETR Model (consisting of a backbone and encoder-decoder Transformer) with object detection heads on
top, for tasks such as COCO detection.
"""
@add_start_docstrings(
    """
    CONDITIONAL_DETR Model (consisting of a backbone and encoder-decoder Transformer) with object detection heads on
    top, for tasks such as COCO detection.
    """,
    CONDITIONAL_DETR_START_DOCSTRING,
)
class ConditionalDetrForObjectDetection(ConditionalDetrPreTrainedModel):
    def __init__(self, config: ConditionalDetrConfig):
        super().__init__(config)

        # CONDITIONAL DETR encoder-decoder model
        self.model = ConditionalDetrModel(config)

        # Object detection heads
        self.class_labels_classifier = nn.Linear(
            config.d_model, config.num_labels
        )  # We add one for the "no object" class
        self.bbox_predictor = ConditionalDetrMLPPredictionHead(
            input_dim=config.d_model, hidden_dim=config.d_model, output_dim=4, num_layers=3
        )

        # Initialize weights and apply final processing
        self.post_init()

    # taken from https://github.com/Atten4Vis/conditionalDETR/blob/master/models/conditional_detr.py
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    @add_start_docstrings_to_model_forward(CONDITIONAL_DETR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ConditionalDetrObjectDetectionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[List[dict]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
):
    """
    Perform forward pass of the Conditional DETR model for object detection.

    Args:
        pixel_values (torch.FloatTensor): Tensor of pixel values of shape (batch_size, sequence_length, channels).
        pixel_mask (Optional[torch.LongTensor]): Optional tensor of pixel masks with shape (batch_size, sequence_length).
        decoder_attention_mask (Optional[torch.LongTensor]): Optional tensor indicating which positions should be
            attended to by the decoder with shape (batch_size, sequence_length).
        encoder_outputs (Optional[torch.FloatTensor]): Optional tensor with encoder outputs of shape
            (batch_size, sequence_length, hidden_size).
        inputs_embeds (Optional[torch.FloatTensor]): Optional tensor of embeddings to be used as inputs to the decoder
            instead of pixel_values.
        decoder_inputs_embeds (Optional[torch.FloatTensor]): Optional tensor of embeddings to be used as inputs to the
            decoder.
        labels (Optional[List[dict]]): Optional list of dictionaries containing labels for object detection.
        output_attentions (Optional[bool]): Whether to output attentions weights.
        output_hidden_states (Optional[bool]): Whether to output hidden states.
        return_dict (Optional[bool]): Whether to return a dictionary as output.

    Returns:
        ConditionalDetrObjectDetectionOutput: Output object containing the logits and predicted boxes.

    """
    """
    CONDITIONAL_DETR Model (consisting of a backbone and encoder-decoder Transformer) with a segmentation head on top,
    for tasks such as COCO panoptic.
    """
    @add_start_docstrings(
        """
        CONDITIONAL_DETR Model (consisting of a backbone and encoder-decoder Transformer) with a segmentation head on top,
        for tasks such as COCO panoptic.
    
        """,
        CONDITIONAL_DETR_START_DOCSTRING,
    )
    class ConditionalDetrForSegmentation(ConditionalDetrPreTrainedModel):
    def __init__(self, config: ConditionalDetrConfig):
        super().__init__(config)

        # object detection model
        self.conditional_detr = ConditionalDetrForObjectDetection(config)

        # segmentation head
        # 获取配置中的隐藏大小和注意力头数
        hidden_size, number_of_heads = config.d_model, config.encoder_attention_heads
        # 从模型的编码器中提取中间通道大小
        intermediate_channel_sizes = self.conditional_detr.model.backbone.conv_encoder.intermediate_channel_sizes

        # 初始化分割头部，连接隐藏大小、注意力头数和中间通道大小的一部分
        self.mask_head = ConditionalDetrMaskHeadSmallConv(
            hidden_size + number_of_heads, intermediate_channel_sizes[::-1][-3:], hidden_size
        )

        # 初始化边界框的注意力机制，使用隐藏大小和注意力头数，指定初始化Xavier的标准差
        self.bbox_attention = ConditionalDetrMHAttentionMap(
            hidden_size, hidden_size, number_of_heads, dropout=0.0, std=config.init_xavier_std
        )

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(CONDITIONAL_DETR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ConditionalDetrSegmentationOutput, config_class=_CONFIG_FOR_DOC)
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
# 定义一个函数 _expand，用于扩展张量的维度
def _expand(tensor, length: int):
    # 在第一维度上插入一个维度，并重复该维度，以扩展张量的长度
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)

# 从 transformers.models.detr.modeling_detr.DetrMaskHeadSmallConv 复制并修改为 ConditionalDetrMaskHeadSmallConv 类
class ConditionalDetrMaskHeadSmallConv(nn.Module):
    """
    简单的卷积头部，使用组归一化。使用 FPN 方法进行上采样
    """

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()

        # 检查 dim 是否能被 8 整除，因为 GroupNorm 中的组数设置为 8
        if dim % 8 != 0:
            raise ValueError(
                "隐藏大小加注意力头的数量必须能被 8 整除，因为 GroupNorm 的组数设置为 8"
            )

        # 计算中间层的维度
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

        self.dim = dim

        # 定义适配器层，用于将 FPN 的特征映射适配到不同的中间层维度
        self.adapter1 = nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.adapter3 = nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

        # 对所有模块进行初始化，卷积层使用 Kaiming 初始化，偏置初始化为常数 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
    # 定义一个方法 `forward`，用于前向传播计算
    def forward(self, x: Tensor, bbox_mask: Tensor, fpns: List[Tensor]):
        # 将特征图 x（形状为 batch_size, d_model, heigth/32, width/32）与 bbox_mask（注意力图，形状为 batch_size, n_queries, n_heads, height/32, width/32）拼接起来
        x = torch.cat([_expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)], 1)

        # 经过第一个神经网络层 lay1
        x = self.lay1(x)
        # 经过第一个 GroupNorm 层 gn1
        x = self.gn1(x)
        # 应用 ReLU 激活函数
        x = nn.functional.relu(x)

        # 经过第二个神经网络层 lay2
        x = self.lay2(x)
        # 经过第二个 GroupNorm 层 gn2
        x = self.gn2(x)
        # 再次应用 ReLU 激活函数
        x = nn.functional.relu(x)

        # 获取当前的特征金字塔网络（FPN）的第一个分支 fpns[0]，并通过 adapter1 适配器层调整其大小
        cur_fpn = self.adapter1(fpns[0])
        # 如果当前特征金字塔分支的 batch 大小与 x 的 batch 大小不一致，则扩展它以匹配 x 的大小
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        # 将适配后的特征金字塔与 x 进行相加，并对 x 进行最近邻插值以调整大小
        x = cur_fpn + nn.functional.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        # 经过第三个神经网络层 lay3
        x = self.lay3(x)
        # 经过第三个 GroupNorm 层 gn3
        x = self.gn3(x)
        # 再次应用 ReLU 激活函数
        x = nn.functional.relu(x)

        # 获取当前的特征金字塔网络（FPN）的第二个分支 fpns[1]，并通过 adapter2 适配器层调整其大小
        cur_fpn = self.adapter2(fpns[1])
        # 如果当前特征金字塔分支的 batch 大小与 x 的 batch 大小不一致，则扩展它以匹配 x 的大小
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        # 将适配后的特征金字塔与 x 进行相加，并对 x 进行最近邻插值以调整大小
        x = cur_fpn + nn.functional.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        # 经过第四个神经网络层 lay4
        x = self.lay4(x)
        # 经过第四个 GroupNorm 层 gn4
        x = self.gn4(x)
        # 再次应用 ReLU 激活函数
        x = nn.functional.relu(x)

        # 获取当前的特征金字塔网络（FPN）的第三个分支 fpns[2]，并通过 adapter3 适配器层调整其大小
        cur_fpn = self.adapter3(fpns[2])
        # 如果当前特征金字塔分支的 batch 大小与 x 的 batch 大小不一致，则扩展它以匹配 x 的大小
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        # 将适配后的特征金字塔与 x 进行相加，并对 x 进行最近邻插值以调整大小
        x = cur_fpn + nn.functional.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        # 经过第五个神经网络层 lay5
        x = self.lay5(x)
        # 经过第五个 GroupNorm 层 gn5
        x = self.gn5(x)
        # 再次应用 ReLU 激活函数
        x = nn.functional.relu(x)

        # 最终经过输出层 out_lay
        x = self.out_lay(x)
        # 返回处理后的输出 x
        return x
# 从 transformers.models.detr.modeling_detr.DetrMHAttentionMap 复制而来，修改为 ConditionalDetrMHAttentionMap 类
class ConditionalDetrMHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True, std=None):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        # 创建线性层，用于计算查询（q）和键（k）的线性变换
        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        # 归一化因子，用于缩放每个头的注意力分数
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask: Optional[Tensor] = None):
        # 计算查询的线性变换
        q = self.q_linear(q)
        # 计算键的线性变换，并进行卷积操作
        k = nn.functional.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        # 将查询和键分割成每个头的部分
        queries_per_head = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        keys_per_head = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        # 计算加权的注意力分数
        weights = torch.einsum("bqnc,bnchw->bqnhw", queries_per_head * self.normalize_fact, keys_per_head)

        if mask is not None:
            # 对注意力分数应用掩码
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), torch.finfo(weights.dtype).min)
        # 计算注意力权重的 softmax，并应用 dropout
        weights = nn.functional.softmax(weights.flatten(2), dim=-1).view(weights.size())
        weights = self.dropout(weights)
        return weights


# 从 transformers.models.detr.modeling_detr.dice_loss 复制而来
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
    # 对输入进行 sigmoid 激活
    inputs = inputs.sigmoid()
    # 展平输入张量
    inputs = inputs.flatten(1)
    # 计算 DICE 损失的分子部分
    numerator = 2 * (inputs * targets).sum(1)
    # 计算 DICE 损失的分母部分
    denominator = inputs.sum(-1) + targets.sum(-1)
    # 计算 DICE 损失值
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


# 从 transformers.models.detr.modeling_detr.sigmoid_focal_loss 复制而来
def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    """
    # 计算 sigmoid focal 损失
    pass
    # 将输入 logits 转换为概率值，使用 sigmoid 函数
    prob = inputs.sigmoid()
    # 计算二元交叉熵损失，reduction="none" 表示不进行降维
    ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # 计算 modulating factor，用于平衡简单和困难的样本
    p_t = prob * targets + (1 - prob) * (1 - targets)
    # 计算加权的损失，通过对损失的平方进行操作
    loss = ce_loss * ((1 - p_t) ** gamma)

    # 如果 alpha 大于等于 0，则计算 alpha_t 权重，用于平衡正负样本
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # 计算最终的损失值，对每个样本的损失求均值后求和，并除以 num_boxes 得到平均损失
    return loss.mean(1).sum() / num_boxes
    # 定义 ConditionalDetrLoss 类，用于计算条件化目标检测或分割任务中的损失
    """
    This class computes the losses for ConditionalDetrForObjectDetection/ConditionalDetrForSegmentation. The process
    happens in two steps: 1) we compute hungarian assignment between ground truth boxes and the outputs of the model 2)
    we supervise each pair of matched ground-truth / prediction (supervise class and box).

    Args:
        matcher (`ConditionalDetrHungarianMatcher`):
            Module able to compute a matching between targets and proposals.
        num_classes (`int`):
            Number of object categories, omitting the special no-object category.
        focal_alpha (`float`):
            Alpha parameter in focal loss.
        losses (`List[str]`):
            List of all the losses to be applied. See `get_loss` for a list of all available losses.
    """
    # 初始化方法，接受匹配器、类别数量、焦点损失的 alpha 参数和损失列表
    # 被复制自 transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrLoss.__init__
    def __init__(self, matcher, num_classes, focal_alpha, losses):
        super().__init__()
        # 设置匹配器对象
        self.matcher = matcher
        # 设置类别数量
        self.num_classes = num_classes
        # 设置焦点损失的 alpha 参数
        self.focal_alpha = focal_alpha
        # 设置损失列表
        self.losses = losses

    # 定义损失标签方法，计算分类损失（二元焦点损失）
    # 输出中必须包含 "logits" 键
    """
    Classification loss (Binary focal loss) targets dicts must contain the key "class_labels" containing a tensor
    of dim [nb_target_boxes]
    """
    # 接受输出、目标、索引和盒子数量作为参数
    # 被复制自 transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrLoss.loss_labels
    def loss_labels(self, outputs, targets, indices, num_boxes):
        # 如果输出中没有 "logits" 键，则抛出 KeyError
        if "logits" not in outputs:
            raise KeyError("No logits were found in the outputs")
        # 获取源 logits
        source_logits = outputs["logits"]

        # 获取源排列的索引
        idx = self._get_source_permutation_idx(indices)
        # 从目标中获取目标类别
        target_classes_o = torch.cat([t["class_labels"][J] for t, (_, J) in zip(targets, indices)])
        # 创建一个全为 num_classes 的 tensor，用于表示目标类别
        target_classes = torch.full(
            source_logits.shape[:2], self.num_classes, dtype=torch.int64, device=source_logits.device
        )
        target_classes[idx] = target_classes_o

        # 创建一个全零的 tensor，用于表示目标类别的 one-hot 编码
        target_classes_onehot = torch.zeros(
            [source_logits.shape[0], source_logits.shape[1], source_logits.shape[2] + 1],
            dtype=source_logits.dtype,
            layout=source_logits.layout,
            device=source_logits.device,
        )
        # 在 target_classes_onehot 上按照 target_classes 的索引进行填充
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        # 去掉最后一维，使得 target_classes_onehot 与 source_logits 的形状一致
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        # 计算交叉熵损失（乘以类别数目）
        loss_ce = (
            sigmoid_focal_loss(source_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2)
            * source_logits.shape[1]
        )
        # 返回计算出的损失
        losses = {"loss_ce": loss_ce}

        return losses

    # 用 torch.no_grad() 修饰的方法，计算基数损失
    # 被复制自 transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrLoss.loss_cardinality
    # 定义一个方法用于计算卡迪尼尔错误，即预测的非空框数量的绝对误差
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        Compute the cardinality error, i.e. the absolute error in the number of predicted non-empty boxes.

        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients.
        """
        # 从模型输出中获取 logits（预测结果）
        logits = outputs["logits"]
        # 获取 logits 的设备信息
        device = logits.device
        # 计算目标长度，即每个目标框内的类别标签数目
        target_lengths = torch.as_tensor([len(v["class_labels"]) for v in targets], device=device)
        # 计算预测的非"no-object"类别的数量
        card_pred = (logits.argmax(-1) != logits.shape[-1] - 1).sum(1)
        # 使用 L1 损失计算卡迪尼尔错误
        card_err = nn.functional.l1_loss(card_pred.float(), target_lengths.float())
        # 构建损失字典，存储卡迪尼尔错误
        losses = {"cardinality_error": card_err}
        return losses

    # 从 transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrLoss.loss_boxes 复制而来
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]. The target boxes
        are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        # 检查输出中是否存在预测框
        if "pred_boxes" not in outputs:
            raise KeyError("No predicted boxes found in outputs")
        # 获取源索引的置换索引
        idx = self._get_source_permutation_idx(indices)
        # 获取源框（预测框）和目标框
        source_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # 计算边界框的 L1 回归损失
        loss_bbox = nn.functional.l1_loss(source_boxes, target_boxes, reduction="none")

        # 构建损失字典，存储边界框的损失，将损失平均化为每个框的损失
        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        # 计算边界框的 GIoU 损失
        loss_giou = 1 - torch.diag(
            generalized_box_iou(center_to_corners_format(source_boxes), center_to_corners_format(target_boxes))
        )
        # 将 GIoU 损失平均化为每个框的损失
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    # 从 transformers.models.detr.modeling_detr.DetrLoss.loss_masks 复制而来
    def loss_masks(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.

        Targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w].
        """
        # 检查输出中是否包含预测的 masks
        if "pred_masks" not in outputs:
            raise KeyError("No predicted masks found in outputs")

        # 获取源索引的排列顺序
        source_idx = self._get_source_permutation_idx(indices)
        # 获取目标索引的排列顺序
        target_idx = self._get_target_permutation_idx(indices)
        # 获取预测的 masks
        source_masks = outputs["pred_masks"]
        # 根据源索引重新排列预测的 masks
        source_masks = source_masks[source_idx]
        # 获取目标中的 masks 列表
        masks = [t["masks"] for t in targets]

        # 将目标 masks 转换成嵌套张量，并解析出有效区域
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        # 将目标 masks 转换为与预测 masks 相同的设备类型
        target_masks = target_masks.to(source_masks)
        # 根据目标索引重新排列目标 masks
        target_masks = target_masks[target_idx]

        # 将预测 masks 上采样到目标尺寸
        source_masks = nn.functional.interpolate(
            source_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False
        )
        # 压缩维度
        source_masks = source_masks[:, 0].flatten(1)

        # 压缩维度
        target_masks = target_masks.flatten(1)
        # 重新调整形状以匹配预测 masks
        target_masks = target_masks.view(source_masks.shape)

        # 计算损失，包括 sigmoid focal loss 和 dice loss
        losses = {
            "loss_mask": sigmoid_focal_loss(source_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(source_masks, target_masks, num_boxes),
        }
        return losses

    # Copied from transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrLoss._get_source_permutation_idx
    def _get_source_permutation_idx(self, indices):
        # 根据 indices 对预测结果进行排列
        batch_idx = torch.cat([torch.full_like(source, i) for i, (source, _) in enumerate(indices)])
        source_idx = torch.cat([source for (source, _) in indices])
        return batch_idx, source_idx

    # Copied from transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrLoss._get_target_permutation_idx
    def _get_target_permutation_idx(self, indices):
        # 根据 indices 对目标进行排列
        batch_idx = torch.cat([torch.full_like(target, i) for i, (_, target) in enumerate(indices)])
        target_idx = torch.cat([target for (_, target) in indices])
        return batch_idx, target_idx

    # Copied from transformers.models.detr.modeling_detr.DetrLoss.get_loss
    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        # 根据指定的损失类型获取相应的损失函数并计算损失
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
        }
        # 检查损失类型是否被支持
        if loss not in loss_map:
            raise ValueError(f"Loss {loss} not supported")
        return loss_map[loss](outputs, targets, indices, num_boxes)

    # Copied from transformers.models.detr.modeling_detr.DetrLoss.forward
    def forward(self, outputs, targets):
        """
        This method computes the losses for the model during training.

        Args:
             outputs (`dict`, *optional*):
                Dictionary containing tensors representing model predictions.
             targets (`List[dict]`, *optional*):
                List of dictionaries where each dictionary corresponds to target data for one sample in the batch.

        Returns:
            losses (`dict`):
                Dictionary containing computed losses.

        """
        # Exclude auxiliary outputs from outputs dictionary
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "auxiliary_outputs"}

        # Match model outputs with target data
        indices = self.matcher(outputs_without_aux, targets)

        # Calculate the total number of target boxes across all samples
        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)

        # Determine the world size for distributed training
        world_size = 1
        if is_accelerate_available():
            if PartialState._shared_state != {}:
                num_boxes = reduce(num_boxes)
                world_size = PartialState().num_processes
        
        # Normalize num_boxes and ensure it is at least 1
        num_boxes = torch.clamp(num_boxes / world_size, min=1).item()

        # Compute losses for each specified loss type
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # If there are auxiliary outputs, compute losses for each auxiliary output separately
        if "auxiliary_outputs" in outputs:
            for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):
                indices = self.matcher(auxiliary_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        # Skip computing intermediate masks losses due to computational cost
                        continue
                    # Append index suffix to loss keys to distinguish between auxiliary losses
                    l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
# 从 transformers.models.detr.modeling_detr.DetrMLPPredictionHead 复制而来，修改为 ConditionalDetrMLPPredictionHead
class ConditionalDetrMLPPredictionHead(nn.Module):
    """
    Very simple multi-layer perceptron (MLP, also called FFN), used to predict the normalized center coordinates,
    height and width of a bounding box w.r.t. an image.

    Copied from https://github.com/facebookresearch/detr/blob/master/models/detr.py
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        # 创建包含多个线性层的 ModuleList，用于构建 MLP
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            # 对输入应用 ReLU 激活函数，除了最后一层
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# 从 transformers.models.deformable_detr.modeling_deformable_detr.DeformableDetrHungarianMatcher 复制而来，修改为 ConditionalDetrHungarianMatcher
class ConditionalDetrHungarianMatcher(nn.Module):
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
        # 确保模块在后端中使用了 "scipy" 库
        requires_backends(self, ["scipy"])

        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        # 如果所有匹配器的成本都为零，则引发 ValueError 异常
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
        
        batch_size, num_queries = outputs["logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        # 将 logits 展平为 [batch_size * num_queries, num_classes] 并进行 sigmoid 操作，得到分类概率
        out_prob = outputs["logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        
        # 将 pred_boxes 展平为 [batch_size * num_queries, 4]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        # 将所有目标的类别标签拼接成一个张量
        target_ids = torch.cat([v["class_labels"] for v in targets])
        
        # 将所有目标的边界框坐标拼接成一个张量
        target_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost.
        # 计算分类损失
        alpha = 0.25
        gamma = 2.0
        # 计算负分类损失
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        # 计算正分类损失
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        # 选取目标类别对应的损失
        class_cost = pos_cost_class[:, target_ids] - neg_cost_class[:, target_ids]

        # Compute the L1 cost between boxes
        # 计算边界框之间的 L1 损失
        bbox_cost = torch.cdist(out_bbox, target_bbox, p=1)

        # Compute the giou cost between boxes
        # 计算边界框之间的 GIoU 损失
        giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))

        # Final cost matrix
        # 组合计算得到的三种损失到最终的损失矩阵中
        cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        # Perform linear sum assignment on the cost matrix split by target sizes
        # 根据目标的大小执行线性求和分配，并返回索引
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
# Copied from transformers.models.detr.modeling_detr._upcast
# 将输入张量的数据类型上溯，以防止乘法运算中的数值溢出
def _upcast(t: Tensor) -> Tensor:
    if t.is_floating_point():
        # 如果输入张量已经是浮点类型，则保持不变；否则将其转换为 float 类型
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        # 如果输入张量是整数类型，则保持不变；否则将其转换为 int 类型
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


# Copied from transformers.models.detr.modeling_detr.box_area
# 计算一组边界框的面积，这些边界框由其 (x1, y1, x2, y2) 坐标表示
def box_area(boxes: Tensor) -> Tensor:
    """
    计算一组边界框的面积，这些边界框由其 (x1, y1, x2, y2) 坐标表示。

    Args:
        boxes (`torch.FloatTensor` of shape `(number_of_boxes, 4)`):
            需要计算面积的边界框。边界框应以 (x1, y1, x2, y2) 格式给出，要求 `0 <= x1 < x2` 和 `0 <= y1 < y2`。

    Returns:
        `torch.FloatTensor`: 包含每个边界框面积的张量。
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# Copied from transformers.models.detr.modeling_detr.box_iou
# 计算两组边界框的 IoU（交并比）
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    width_height = (right_bottom - left_top).clamp(min=0)  # [N,M,2]
    inter = width_height[:, :, 0] * width_height[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


# Copied from transformers.models.detr.modeling_detr.generalized_box_iou
# 计算通用 IoU，参考 https://giou.stanford.edu/
def generalized_box_iou(boxes1, boxes2):
    """
    计算通用 IoU，参考 https://giou.stanford.edu/。边界框应以 [x0, y0, x1, y1]（角点）格式给出。

    Returns:
        `torch.FloatTensor`: 一个形状为 [N, M] 的成对矩阵，其中 N = len(boxes1)，M = len(boxes2)
    """
    # 检查是否存在不正确的边界框，避免产生无限值或非数值结果
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        raise ValueError(f"boxes1 的格式必须为 [x0, y0, x1, y1]（角点），但得到的是 {boxes1}")
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        raise ValueError(f"boxes2 的格式必须为 [x0, y0, x1, y1]（角点），但得到的是 {boxes2}")
    iou, union = box_iou(boxes1, boxes2)

    top_left = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    width_height = (bottom_right - top_left).clamp(min=0)  # [N,M,2]
    area = width_height[:, :, 0] * width_height[:, :, 1]

    return iou - (area - union) / area


# Copied from transformers.models.detr.modeling_detr._max_by_axis
# 返回给定列表中各子列表在相同索引处的最大值列表
def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes
# 定义嵌套张量类，包含多个张量和可选的遮罩张量
class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors  # 初始化张量列表
        self.mask = mask  # 初始化遮罩张量

    # 将嵌套张量移到指定设备
    def to(self, device):
        # 将张量列表转移到指定设备
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            # 如果有遮罩张量，则也将其转移到指定设备
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    # 分解嵌套张量，返回包含的张量和遮罩张量
    def decompose(self):
        return self.tensors, self.mask

    # 返回嵌套张量的字符串表示
    def __repr__(self):
        return str(self.tensors)


# 从张量列表创建嵌套张量
# 复制自 transformers.models.detr.modeling_detr.nested_tensor_from_tensor_list
def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    if tensor_list[0].ndim == 3:
        # 计算最大尺寸
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        batch_size, num_channels, height, width = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        # 创建全零张量，形状为 batch_shape，指定 dtype 和设备
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        # 创建全一遮罩张量，形状为 (batch_size, height, width)，布尔类型，指定设备
        mask = torch.ones((batch_size, height, width), dtype=torch.bool, device=device)
        # 将每个张量复制到对应的零张量片段中，并更新遮罩张量
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        # 如果不是三维张量，则引发错误
        raise ValueError("Only 3-dimensional tensors are supported")
    return NestedTensor(tensor, mask)
```