# `.\models\udop\modeling_udop.py`

```
# coding=utf-8
# 版权 2024 年 Microsoft Research 和 HuggingFace Inc. 团队所有。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）授权；
# 除非符合许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件基于“原样”分发，
# 没有任何形式的明示或暗示担保或条件。
# 有关详细信息，请参阅许可证。
""" PyTorch UDOP model."""

import collections  # 导入 collections 模块
import logging  # 导入 logging 模块
import math  # 导入 math 模块
import random  # 导入 random 模块
from abc import ABC, abstractmethod  # 从 abc 模块导入 ABC 抽象类和 abstractmethod 装饰器
from copy import deepcopy  # 导入 deepcopy 函数
from dataclasses import dataclass  # 导入 dataclass 装饰器
from typing import Any, Dict, Optional, Sequence, Tuple, Union  # 导入类型提示相关的类和函数

import torch  # 导入 PyTorch 库
from torch import Tensor, nn  # 从 torch 库导入 Tensor 和 nn 模块
from torch.nn import CrossEntropyLoss  # 从 torch.nn 模块导入 CrossEntropyLoss 类

from transformers import UdopConfig  # 导入 UdopConfig 类
from transformers.modeling_outputs import (  # 从 transformers.modeling_outputs 导入以下类
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)

from ...activations import ACT2FN  # 导入 ACT2FN 激活函数
from ...modeling_utils import PreTrainedModel  # 导入 PreTrainedModel 类
from ...pytorch_utils import (  # 从 ...pytorch_utils 导入以下函数
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from ...utils import (  # 从 ...utils 导入以下函数和类
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

logger = logging.getLogger(__name__)  # 获取当前模块的 logger 对象

UDOP_PRETRAINED_MODEL_ARCHIVE_LIST = [  # UDOP 预训练模型的列表
    "microsoft/udop-large",
    # 查看所有 UDOP 模型：https://huggingface.co/models?filter=udop
]

_CONFIG_FOR_DOC = "UdopConfig"  # 用于文档的配置名称

UDOP_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Args:
        config ([`UdopConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

UDOP_INPUTS_DOCSTRING = r"""
"""

UDOP_ENCODER_INPUTS_DOCSTRING = r"""
"""

@dataclass
class BaseModelOutputWithAttentionMask(ModelOutput):
    """
    Class for the model's outputs that may also contain a past key/values (to speed up sequential decoding). Includes
    an additional attention mask.
"""
    # 最后一层模型的隐藏状态，形状为(batch_size, sequence_length, hidden_size)，若使用了past_key_values，只输出形状为(batch_size, 1, hidden_size)的序列的最后隐藏状态。
    last_hidden_state: torch.FloatTensor = None
    
    # 注意力掩码，形状为(batch_size, sequence_length)，用于指示模型在计算注意力时要忽略的位置。
    attention_mask: torch.FloatTensor = None
    
    # 过去的键值对，类型为Optional[Tuple[Tuple[torch.FloatTensor]]]，当使用use_cache=True或者config.use_cache=True时返回，包含预先计算的隐藏状态（在自注意力块中的键和值），
    # 若config.is_encoder_decoder=True，还包含交叉注意力块中的隐藏状态。
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    
    # 隐藏状态的元组，类型为Optional[Tuple[torch.FloatTensor]]，当传递output_hidden_states=True或者config.output_hidden_states=True时返回，
    # 包含模型每一层的隐藏状态（如果模型有嵌入层，则还包括初始嵌入输出）。
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    
    # 注意力权重的元组，类型为Optional[Tuple[torch.FloatTensor]]，当传递output_attentions=True或者config.output_attentions=True时返回，
    # 包含每一层的注意力权重，形状为(batch_size, num_heads, sequence_length, sequence_length)，用于计算自注意力头中的加权平均值。
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    
    # 交叉注意力权重的元组，类型为Optional[Tuple[torch.FloatTensor]]，当传递output_attentions=True和config.add_cross_attention=True时返回，
    # 包含解码器的交叉注意力层的注意力权重，形状为(batch_size, num_heads, sequence_length, sequence_length)，用于计算交叉注意力头中的加权平均值。
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    """
    合并图像和文本嵌入，作为UDOP编码器/解码器的输入。

    首先，通过检查每个视觉补丁是否在标记边界框内，创建图像嵌入。如果是，则将视觉补丁与标记嵌入组合。
    然后，将视觉边界框与文本边界框结合起来。
    最后，将视觉边界框与文本注意力掩码结合起来。
    """

    # 计算序列的长度，即视觉补丁的数量
    sequence_length = num_patches

    # 计算OCR点的x坐标，取值范围在0到sequence_length-1之间
    ocr_points_x = torch.clip(
        torch.floor((bbox[:, :, 0] + bbox[:, :, 2]) / 2.0 * sequence_length).long(), 0, sequence_length - 1
    )

    # 计算OCR点的y坐标，取值范围在0到(sequence_length-1)*sequence_length之间
    ocr_points_y = (
        torch.clip(torch.floor((bbox[:, :, 1] + bbox[:, :, 3]) / 2.0 * sequence_length).long(), 0, sequence_length - 1)
        * sequence_length
    )

    # 组合计算得到OCR点的索引
    ocr_points = ocr_points_x + ocr_points_y

    # 确保边界框的类型为float以计算均值
    bbox = bbox.to(torch.float64)

    # 创建目标分段，判断是否边界框均值为0或1
    target_seg = (bbox.mean(-1) == 0.0) | (bbox.mean(-1) == 1.0)

    # 根据OCR点索引，重复使用视觉嵌入
    repeated_vision_embeds = torch.gather(
        image_embeddings, 1, ocr_points.unsqueeze(-1).repeat(1, 1, image_embeddings.size(-1))
    )

    # 将重复视觉嵌入中属于目标分段的部分置为0
    repeated_vision_embeds[target_seg] = 0.0

    # 将重复视觉嵌入添加到输入嵌入中
    inputs_embeds += repeated_vision_embeds

    # 创建补丁索引，全为True的布尔张量
    patch_inds = torch.full_like(image_embeddings[:, :, 0], True).bool()

    # 构造索引张量ind，用于聚合OCR点
    ind = torch.cat(
        [
            torch.arange(len(ocr_points))[:, None].repeat(1, ocr_points.size(-1))[:, :, None].to(ocr_points),
            ocr_points[:, :, None],
        ],
        dim=-1,
    )

    # 展平ind张量，以便用于后续操作
    ind = ind.flatten(0, 1)
    # 将元组列表解压缩为行和列两个分离的列表
    rows, cols = zip(*ind)
    # 根据给定的行列索引将 patch_inds 中对应位置的元素设为 False
    patch_inds[rows, cols] = False

    # 从 image_embeddings 中选择符合 patch_inds 条件的图像嵌入片段，并组成列表
    input_vision_patches = [image_embeddings[i][patch_inds[i]] for i in range(len(patch_inds))]

    # 如果 visual_bbox 为 None，则调用 get_visual_bbox 函数获取视觉边界框，并做扩展和设备适配处理
    if visual_bbox is None:
        visual_bbox = get_visual_bbox(image_size=image_size, patch_size=patch_size)
        visual_bbox = visual_bbox.unsqueeze(0).repeat(image_embeddings.size(0), 1, 1)
        visual_bbox = visual_bbox.to(image_embeddings.device)

    # 根据 patch_inds 条件选择 visual_bbox 中的子集，并组成列表
    visual_bbox = [visual_bbox[i][patch_inds[i]] for i in range(len(patch_inds))]

    # 如果 attention_mask 不为 None，则为 visual_bbox 创建对应的视觉注意力掩码列表
    if attention_mask is not None:
        visual_attention_mask = [torch.tensor([1] * len(item)).to(attention_mask) for item in visual_bbox]

    # 如果 max_len 为 0，则设为 image_embeddings 的第一维度大小；否则减去 inputs_embeds 的第一维度大小
    if max_len == 0:
        max_len = image_embeddings.size(1)
    else:
        max_len = max_len - inputs_embeds.size(1)

    # 将 input_vision_patches 中的每个张量填充到相同的最大长度，并组成张量列表
    inputs_vision_patches = torch.stack(
        [pad_sequence(item, max_len, torch.zeros_like(image_embeddings[0, 0])) for item in input_vision_patches]
    )

    # 将 visual_bbox 中的每个张量填充到相同的最大长度，并组成张量列表
    visual_bbox = torch.stack([pad_sequence(item, max_len, torch.zeros_like(bbox[0, 0])) for item in visual_bbox])

    # 如果 attention_mask 不为 None，则将 visual_attention_mask 中的每个张量填充到相同的最大长度，并组成张量列表
    if attention_mask is not None:
        visual_attention_mask = torch.stack(
            [pad_sequence(item, max_len, torch.zeros_like(attention_mask[0, 0])) for item in visual_attention_mask]
        )

    # 将 inputs_embeds 和 inputs_vision_patches 拼接在第二维度上
    inputs_embeds = torch.cat([inputs_embeds, inputs_vision_patches], 1)

    # 将 bbox 和 visual_bbox 拼接在第二维度上
    bbox = torch.cat([bbox, visual_bbox], 1)

    # 如果 attention_mask 不为 None，则将 attention_mask 和 visual_attention_mask 拼接在第二维度上
    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, visual_attention_mask], 1)

    # 返回拼接后的 inputs_embeds, bbox 和 attention_mask（如果有的话）
    return inputs_embeds, bbox, attention_mask
class UdopPatchEmbeddings(nn.Module):
    """2D Image to Patch Embeddings"""

    def __init__(self, config):
        super().__init__()
        # 初始化函数，接收配置参数config，并进行初始化设置
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        # 如果image_size和patch_size不是可迭代对象，则转换为元组形式
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        
        # 计算图像分块数
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        
        # 设置对象的属性
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        # 使用2D卷积层将图像块映射为嵌入向量
        self.proj = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values):
        # 前向传播函数，接收像素值张量，并返回嵌入向量
        batch_size, num_channels, height, width = pixel_values.shape
        
        # 检查输入图像尺寸是否符合预期
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model"
                f" ({self.image_size[0]}*{self.image_size[1]})."
            )
        
        # 使用卷积层进行嵌入向量的计算
        embeddings = self.proj(pixel_values)
        
        # 将卷积输出展平，并转置维度，以便后续处理
        embeddings = embeddings.flatten(2).transpose(1, 2)
        
        # 返回计算得到的嵌入向量
        return embeddings


class UdopPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models. Based on `T5PreTrainedModel`.
    """

    config_class = UdopConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["UdopBlock"]
    _keep_in_fp32_modules = ["wo"]

    # 从transformers.models.prophetnet.modeling_prophetnet.ProphetNetPreTrainedModel._shift_right复制而来，将ProphetNet替换为Udop
    def _shift_right(self, input_ids):
        # 获取解码器起始标记和填充标记
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        # 确保解码器起始标记被定义
        assert decoder_start_token_id is not None, (
            "self.model.config.decoder_start_token_id has to be defined. In Udop it is usually set to the"
            " pad_token_id. See Udop docs for more information"
        )

        # 将输入向右移动一位
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        # 确保填充标记被定义
        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        
        # 将标签中可能存在的-100值替换为填充标记
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        # 确保`shifted_input_ids`中的值都是非负数
        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

        # 返回向右移位后的输入标记
        return shifted_input_ids

# 从transformers.models.t5.modeling_t5.T5LayerNorm复制而来，将T5替换为Udop
# 定义一个名为 UdopLayerNorm 的自定义 PyTorch 模块，用于实现 Udop 风格的 Layer Normalization
class UdopLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        构造一个 Udop 风格的 layernorm 模块。无偏置和无均值减法。
        """
        super().__init__()
        # 使用 nn.Parameter 定义可学习的权重参数，默认为全 1 的张量
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # 设置方差的 epsilon 值，用于数值稳定性
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # Udop 使用一种仅进行缩放而不进行偏移的 layer_norm，也被称为 Root Mean Square Layer Normalization
        # https://arxiv.org/abs/1910.07467，因此计算方差时没有均值，并且没有偏置。此外，我们希望确保
        # 半精度输入的累积在 fp32 中完成

        # 计算输入张量的方差，将输入转换为 float32 类型，平方，沿着指定维度求均值，并保持维度
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        # 根据方差和 epsilon 计算归一化后的 hidden_states
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # 如果权重参数的数据类型是半精度（float16 或 bfloat16），则将 hidden_states 转换为相同数据类型
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        # 返回经过权重缩放后的 hidden_states
        return self.weight * hidden_states


# 从 transformers.models.t5.modeling_t5.T5DenseActDense 复制并修改为 Udop 风格
class UdopDenseActDense(nn.Module):
    def __init__(self, config: UdopConfig):
        super().__init__()
        # 使用 nn.Linear 定义一个线性变换层 wi，输入维度为 config.d_model，输出维度为 config.d_ff，无偏置
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        # 定义一个线性变换层 wo，输入维度为 config.d_ff，输出维度为 config.d_model，无偏置
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        # 定义一个 Dropout 层，使用 config.dropout_rate 作为丢弃概率
        self.dropout = nn.Dropout(config.dropout_rate)
        # 选择激活函数，根据 config.dense_act_fn 选择对应的激活函数
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        # 对输入的 hidden_states 进行 wi 层的线性变换
        hidden_states = self.wi(hidden_states)
        # 对变换后的 hidden_states 应用选择的激活函数
        hidden_states = self.act(hidden_states)
        # 对激活后的 hidden_states 应用 Dropout 操作
        hidden_states = self.dropout(hidden_states)
        
        # 如果 wo.weight 是 torch.Tensor，并且 hidden_states 的数据类型不等于 wo.weight 的数据类型，
        # 并且 wo.weight 的数据类型不是 torch.int8，则将 hidden_states 转换为 wo.weight 的数据类型
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        
        # 对应用完所有层操作后的 hidden_states 应用 wo 层的线性变换
        hidden_states = self.wo(hidden_states)
        
        # 返回最终的 hidden_states
        return hidden_states


# 从 transformers.models.t5.modeling_t5.T5DenseGatedActDense 复制并修改为 Udop 风格
class UdopDenseGatedActDense(nn.Module):
    def __init__(self, config: UdopConfig):
        super().__init__()
        # 定义两个线性变换层 wi_0 和 wi_1，输入维度均为 config.d_model，输出维度为 config.d_ff，无偏置
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        # 定义一个线性变换层 wo，输入维度为 config.d_ff，输出维度为 config.d_model，无偏置
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        # 定义一个 Dropout 层，使用 config.dropout_rate 作为丢弃概率
        self.dropout = nn.Dropout(config.dropout_rate)
        # 选择激活函数，根据 config.dense_act_fn 选择对应的激活函数
        self.act = ACT2FN[config.dense_act_fn]
    # 定义神经网络的前向传播方法，接收隐藏状态作为输入
    def forward(self, hidden_states):
        # 将隐藏状态通过激活函数 gelu 处理
        hidden_gelu = self.act(self.wi_0(hidden_states))
        # 将隐藏状态通过线性层 wi_1 处理
        hidden_linear = self.wi_1(hidden_states)
        # 将 gelu 处理后的结果与线性处理后的结果相乘，得到新的隐藏状态
        hidden_states = hidden_gelu * hidden_linear
        # 对新的隐藏状态进行 dropout 处理
        hidden_states = self.dropout(hidden_states)

        # 为了使 8 位量化在 google/flan-t5-xxl 模型中正常工作，self.wo 保持为 float32 类型
        # 参考 https://github.com/huggingface/transformers/issues/20287
        # 同时，确保权重不是 `int8` 类型，以防用户强制 `_keep_in_fp32_modules` 为 `None`
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            # 如果 hidden_states 的数据类型与 self.wo.weight 的数据类型不同，则将 hidden_states 转换为 self.wo.weight 的数据类型
            hidden_states = hidden_states.to(self.wo.weight.dtype)

        # 将处理后的隐藏状态传递给输出层 wo 进行处理
        hidden_states = self.wo(hidden_states)
        # 返回处理后的隐藏状态作为输出
        return hidden_states
# Copied from transformers.models.t5.modeling_t5.T5LayerFF with T5->Udop
class UdopLayerFF(nn.Module):
    def __init__(self, config: UdopConfig):
        super().__init__()
        # 根据配置选择不同类型的前向传播层：带门控激活函数或普通激活函数
        if config.is_gated_act:
            self.DenseReluDense = UdopDenseGatedActDense(config)
        else:
            self.DenseReluDense = UdopDenseActDense(config)

        # 初始化层归一化和dropout层
        self.layer_norm = UdopLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        # 应用层归一化到隐藏状态
        forwarded_states = self.layer_norm(hidden_states)
        # 经过前向传播层（带门控或普通激活函数）
        forwarded_states = self.DenseReluDense(forwarded_states)
        # 使用dropout应用到隐藏状态
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


# Copied from transformers.models.t5.modeling_t5.T5Attention with T5->Udop
class UdopAttention(nn.Module):
    def __init__(self, config: UdopConfig, has_relative_attention_bias=False):
        super().__init__()
        # 初始化注意力层参数和配置
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # 使用线性层定义查询、键、值和输出的投影
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        # 如果有相对注意力偏置，则初始化相对注意力偏置的嵌入
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()
        self.gradient_checkpointing = False

    def prune_heads(self, heads):
        # 如果需要修剪注意力头部，则根据给定的头部索引进行修剪
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
        )
        # 对线性层进行修剪
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # 更新超参数
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor - 相对位置差值的整数张量
            bidirectional: a boolean - 是否为双向注意力的标志
            num_buckets: an integer - 桶的数量，决定了输出的范围 [0, num_buckets)
            max_distance: an integer - 最大的相对距离，超过该距离的相对位置都映射到同一个桶中

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
            返回一个形状与relative_position相同的张量，包含在范围[0, num_buckets)内的int32值
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets
    # 计算相对位置偏置
    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        # 如果未指定设备，则使用预定义的相对注意力偏置权重的设备
        if device is None:
            device = self.relative_attention_bias.weight.device
        # 创建一个表示查询长度的张量，并在第二维添加维度
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        # 创建一个表示键长度的张量，并在第一维添加维度
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        # 计算相对位置差，得到形状为 (query_length, key_length) 的张量
        relative_position = memory_position - context_position  
        # 根据相对位置差进行分桶，得到形状为 (query_length, key_length) 的张量
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # 形状为 (query_length, key_length) 的相对位置差张量
            bidirectional=(not self.is_decoder),  # 是否双向
            num_buckets=self.relative_attention_num_buckets,  # 分桶数量
            max_distance=self.relative_attention_max_distance,  # 最大距离限制
        )
        # 根据分桶结果获取相对注意力偏置值，形状为 (query_length, key_length, num_heads)
        values = self.relative_attention_bias(relative_position_bucket)
        # 调整张量的维度顺序，添加额外的维度，形状变为 (1, num_heads, query_length, key_length)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        # 返回相对位置偏置张量
        return values

    # 前向传播方法
    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
# 从transformers.models.t5.modeling_t5.T5LayerSelfAttention复制过来，将T5替换为Udop
class UdopLayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        # 初始化自注意力层，使用UdopAttention代替T5中的Attention机制
        self.SelfAttention = UdopAttention(config, has_relative_attention_bias=has_relative_attention_bias)
        # 初始化层归一化模块，使用UdopLayerNorm代替T5中的LayerNorm
        self.layer_norm = UdopLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # 初始化dropout层，使用给定的dropout率
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        # 对输入的hidden_states进行层归一化处理
        normed_hidden_states = self.layer_norm(hidden_states)
        # 调用SelfAttention进行注意力计算
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # 将原始的hidden_states和dropout后的attention_output相加，得到新的hidden_states
        hidden_states = hidden_states + self.dropout(attention_output[0])
        # 如果输出注意力权重，将其包含在outputs中
        outputs = (hidden_states,) + attention_output[1:]  # 如果有输出，添加注意力权重
        return outputs


# 从transformers.models.t5.modeling_t5.T5LayerCrossAttention复制过来，将T5替换为Udop
class UdopLayerCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化跨层注意力层，使用UdopAttention代替T5中的Attention机制
        self.EncDecAttention = UdopAttention(config, has_relative_attention_bias=False)
        # 初始化层归一化模块，使用UdopLayerNorm代替T5中的LayerNorm
        self.layer_norm = UdopLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # 初始化dropout层，使用给定的dropout率
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        # 对输入的hidden_states进行层归一化处理
        normed_hidden_states = self.layer_norm(hidden_states)
        # 调用EncDecAttention进行跨层注意力计算
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        # 将原始的hidden_states和dropout后的attention_output相加，得到新的层输出
        layer_output = hidden_states + self.dropout(attention_output[0])
        # 如果输出注意力权重，将其包含在outputs中
        outputs = (layer_output,) + attention_output[1:]  # 如果有输出，添加注意力权重
        return outputs


# 从transformers.models.t5.modeling_t5.T5Block复制过来，将T5替换为Udop
class UdopBlock(nn.Module):
    # 初始化方法，用于创建一个新的UdopLayer对象
    def __init__(self, config, has_relative_attention_bias=False):
        # 调用父类的初始化方法
        super().__init__()
        # 根据配置设置是否为解码器
        self.is_decoder = config.is_decoder
        # 创建一个空的模块列表用于存储各层的UdopLayer
        self.layer = nn.ModuleList()
        # 将一个UdopLayerSelfAttention层添加到模块列表中
        self.layer.append(UdopLayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        # 如果是解码器，则添加一个UdopLayerCrossAttention层到模块列表中
        if self.is_decoder:
            self.layer.append(UdopLayerCrossAttention(config))

        # 添加一个UdopLayerFF层到模块列表中
        self.layer.append(UdopLayerFF(config))

    # 前向传播方法，定义了模型的计算流程
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
# UdopCellEmbeddings 类定义，用于生成嵌入表示单元格位置的模块
class UdopCellEmbeddings(nn.Module):
    # 初始化函数，设置最大二维位置嵌入和隐藏大小
    def __init__(self, max_2d_position_embeddings=501, hidden_size=1024):
        super(UdopCellEmbeddings, self).__init__()
        # 设置最大二维位置嵌入的数量
        self.max_2d_position_embeddings = max_2d_position_embeddings

        # 创建 X 轴和 Y 轴位置嵌入的 Embedding 层
        self.x_position_embeddings = nn.Embedding(max_2d_position_embeddings, hidden_size)
        self.y_position_embeddings = nn.Embedding(max_2d_position_embeddings, hidden_size)

    # 前向传播函数，计算嵌入表示
    def forward(self, bbox):
        # 将 bbox 的值裁剪到 [0.0, 1.0] 范围内
        bbox = torch.clip(bbox, 0.0, 1.0)
        # 将裁剪后的 bbox 转换为整数索引，乘以最大二维位置嵌入的数量
        bbox = (bbox * (self.max_2d_position_embeddings - 1)).long()
        # 获取左上角和右下角位置的 X 和 Y 轴嵌入
        left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
        upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
        right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
        lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])

        # 计算最终的嵌入表示，将四个位置的嵌入相加
        embeddings = (
            left_position_embeddings
            + upper_position_embeddings
            + right_position_embeddings
            + lower_position_embeddings
        )

        return embeddings


# UdopAttention._relative_position_bucket 的别名函数 get_relative_position_bucket
get_relative_position_bucket = UdopAttention._relative_position_bucket
# 定义增强范围的元组常量
AUGMENTATION_RANGE = (0.80, 1.25)


# RelativePositionBiasBase 类定义，用于定义相对位置偏置的基础类
class RelativePositionBiasBase(nn.Module, ABC):
    """
    相对位置偏置的基础类。

    Args:
        num_heads (`int`):
            模型中的注意力头数，将创建大小为 `num_heads` 的嵌入，将添加到每个令牌对的分数上。
        relative_attention_num_buckets (`int`, *optional*, 默认为 32):
            令牌对度量（序列中的距离、像素中的距离等）将被分桶化，该参数定义了这种桶的数量。
        bidirectional (`bool`, *optional*, 默认为 `True`):
            令牌对之间的距离是否应该是双向的。如果为 `False`，则距离(tok1, tok2) == 距离(tok2, tok1)。
        scaling_factor (`int`, *optional*, 默认为 1):
            用于缩放相对距离的因子。
        max_distance (`int`, *optional*, 默认为 128):
            所有大于此值的距离将进入同一个桶中。
        augmentation (`bool`, *optional*, 默认为 `False`):
            是否将相对距离乘以随机标量。
        expand (`bool`, *optional*, 默认为 `False`):
            是否扩展现有的预训练模型，并在后续添加中添加 prefix_bucket。
    """

    # 初始化函数，设置相对位置偏置的基础参数
    def __init__(
        self,
        num_heads=None,
        relative_attention_num_buckets=32,
        bidirectional=True,
        scaling_factor=1,
        max_distance=128,
        level="tokens",
        augmentation=False,
        prefix_bucket=False,
        expand=False,
        # 继承自 nn.Module 和 ABC 类的初始化
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)  # 调用父类的初始化函数
    ):
        # 调用父类的构造函数，初始化基类的属性
        super(RelativePositionBiasBase, self).__init__()
        # 初始化相对位置偏置基类的属性
        self.prefix_bucket = prefix_bucket
        self.augmentation = augmentation
        self.level = level
        self.max_distance = max_distance
        self.scaling_factor = scaling_factor
        self.bidirectional = bidirectional
        self.num_heads = num_heads
        self.expand = expand
        self.relative_attention_num_buckets = relative_attention_num_buckets
        # 如果使用前缀桶并且不扩展，额外增加两个头
        extra_head = 2 if prefix_bucket and not self.expand else 0
        # 创建相对注意力偏置的嵌入层，根据桶的数量和额外头的数量进行初始化
        self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets + extra_head, self.num_heads)

    @abstractmethod
    def prepare_input(
        self,
        attention_mask: Optional[Tensor] = None,
        bbox: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        # 准备输入的抽象方法，接受注意力掩码和边界框作为可选参数，返回张量
        pass

    def get_bucket(self, attention_mask: Optional[Tensor] = None, bbox: Optional[Dict[str, Any]] = None) -> Tensor:
        # 获取桶（bucket）函数，计算相对位置桶
        relative_position = self.prepare_input(attention_mask, bbox)
        # 调用函数计算相对位置桶的张量
        rp_bucket: Tensor = get_relative_position_bucket(
            relative_position,
            bidirectional=self.bidirectional,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.max_distance,
        )
        return rp_bucket

    def get_relative_position(self, positions):
        # 获取相对位置的函数，输入位置张量，返回相对位置张量
        context_position = positions[:, :, None]
        memory_position = positions[:, None, :]
        relative_position = memory_position - context_position
        # 如果启用了增强且处于训练模式，则对相对位置进行随机增强
        if self.augmentation and self.training:
            relative_position *= random.uniform(*AUGMENTATION_RANGE)
        relative_position *= self.scaling_factor

        return relative_position.to(torch.long)
    # 如果设置了self.expand并且self.prefix_bucket为True，则执行以下操作：
    # 创建一个新的Embedding层用于相对注意力偏置，长度为self.relative_attention_num_buckets + 2，每个头部有一个值
    new_bias = nn.Embedding(self.relative_attention_num_buckets + 2, self.num_heads)
    # 将旧的相对注意力偏置的权重复制到新的Embedding层中
    new_bias.weight.data[: self.relative_attention_num_buckets] = self.relative_attention_bias.weight.data
    # 将剩余部分设置为0.1
    new_bias.weight.data[self.relative_attention_num_buckets :] = 0.1
    # 将模型的相对注意力偏置设置为新的Embedding层
    self.relative_attention_bias = new_bias
    # 设置self.expand为False，表示不再需要扩展相对注意力偏置
    self.expand = False

rp_bucket = self.get_bucket(attention_mask, bbox)

# 如果self.prefix_bucket为True，则执行以下操作：
if self.prefix_bucket:
    # 如果rp_bucket的第一维度为1，而attention_mask的第一维度大于1，则将rp_bucket复制到相同的长度
    if rp_bucket.size(0) == 1 and attention_mask.size(0) > 1:
        rp_bucket = rp_bucket.repeat(attention_mask.size(0), 1, 1)
    # 基于假设前缀的边界框是负数，判断边界框是否为前缀
    is_prefix = bbox[:, :, 1] < 0
    # 统计每个样本中负数边界框的数量
    num_prefix = is_prefix.sum(-1)
    # 对于每个样本和其对应的负数边界框数量，执行以下操作：
    for idx, num_prefix_row in enumerate(num_prefix.cpu().numpy()):
        # 将rp_bucket中前num_prefix_row行和列填充为self.relative_attention_num_buckets
        rp_bucket[idx, :num_prefix_row, num_prefix_row:] = self.relative_attention_num_buckets
        rp_bucket[idx, num_prefix_row:, :num_prefix_row] = self.relative_attention_num_buckets + 1

# 使用rp_bucket作为输入，计算相对注意力偏置的值
values: Tensor = self.relative_attention_bias(rp_bucket)
# 如果计算得到的values张量维度不是4，则抛出值错误异常
if values.dim() != 4:
    raise ValueError("Wrong dimension of values tensor")
# 调整values张量的维度顺序为[0, 3, 1, 2]
values = values.permute([0, 3, 1, 2])

# 返回调整后的相对注意力偏置值张量
return values
class RelativePositionBias1D(RelativePositionBiasBase):
    def __init__(self, scaling_factor=1, max_distance=128, **kwargs):
        """
        Reimplementation of T5 relative position bias. Distance between given tokens is their distance in the sequence.
        Parameters are the same as in base class
        """
        # 调用父类的初始化方法，设置缩放因子和最大距离等参数
        super().__init__(scaling_factor=scaling_factor, max_distance=max_distance, **kwargs)

    def prepare_input(self, attention_mask: Optional[Tensor] = None, bbox: Optional[Dict[str, Any]] = None) -> Tensor:
        # 如果缩放因子不为1，抛出异常
        if self.scaling_factor != 1:
            raise ValueError("No need to scale 1d features")
        # 创建一个序列长度的张量，代表位置信息，使用与attention_mask相同的设备类型
        relative_position = self.get_relative_position(
            torch.arange(attention_mask.size(1), dtype=torch.long, device=attention_mask.device)[None, :]
        )

        return relative_position


class RelativePositionBiasHorizontal(RelativePositionBiasBase):
    def __init__(self, scaling_factor=100, max_distance=100, **kwargs):
        """
        Represents in the bucket embeddings horizontal distance between two tokens. Parameters are the same as in base
        class
        """
        # 调用父类的初始化方法，设置缩放因子和最大距离等参数
        super().__init__(scaling_factor=scaling_factor, max_distance=max_distance, **kwargs)

    def prepare_input(self, attention_mask: Optional[Tensor] = None, bbox: Optional[Dict[str, Any]] = None) -> Tensor:
        # 如果缩放因子不大于1.0，抛出异常
        if not self.scaling_factor > 1.0:
            raise ValueError("Need to scale the values of bboxes, as they are in small (0,1) range")
        # 如果bbox为None，抛出异常
        if bbox is None:
            raise ValueError("Bbox is required for horizontal relative position bias")
        # 获取bbox左侧点的x坐标位置
        horizontal_position: Tensor = bbox[:, :, [0, 2]].mean(dim=-1)

        return self.get_relative_position(horizontal_position)


class RelativePositionBiasVertical(RelativePositionBiasBase):
    def __init__(self, scaling_factor=100, max_distance=100, **kwargs):
        """
        Represents in the bucket embeddings vertical distance between two tokens. Parameters are the same as in base
        class
        """
        # 调用父类的初始化方法，设置缩放因子和最大距离等参数
        super().__init__(scaling_factor=scaling_factor, max_distance=max_distance, **kwargs)

    def prepare_input(self, attention_mask: Optional[Tensor] = None, bbox: Optional[Dict[str, Any]] = None) -> Tensor:
        # 如果缩放因子不大于1.0，抛出异常
        if not self.scaling_factor > 1.0:
            raise ValueError("Need to scale the values of bboxes, as they are in small (0,1) range")
        # 如果bbox为None，抛出异常
        if bbox is None:
            raise ValueError("Bbox is required for vertical relative position bias")
        # 获取bbox中间点的y坐标位置
        vertical_position: Tensor = bbox[:, :, [1, 3]].mean(dim=-1)

        return self.get_relative_position(vertical_position)


class RelativePositionBiasAggregated(nn.Module):
    # 初始化方法，用于创建一个新的相对位置偏置合并类实例
    def __init__(self, modules: Sequence[RelativePositionBiasBase]):
        """
        Class which sums up various computed biases.

        Args:
            modules (Sequence[RelativePositionBiasBase]):
                List of relative bias modules.
        """
        # 调用父类的初始化方法
        super().__init__()
        # 使用传入的模块列表创建一个神经网络模块列表
        self.biases = nn.ModuleList(modules)

    # 前向传播方法，计算相对位置偏置的总和
    def forward(
        self, attention_mask: Optional[Tensor] = None, bbox: Optional[Dict[str, Any]] = None
    ) -> Union[float, Tensor]:
        # 初始化输出为浮点数0.0
        output = 0.0
        # 遍历每个偏置模块
        for bias in self.biases:  # type: ignore
            # 将当前偏置模块的计算结果加到输出上
            output = bias(attention_mask, bbox) + output

        # 返回最终的输出结果，可以是浮点数或张量
        return output
# 定义了一个字典，将字符串映射到相应的相对位置偏置类上
BIAS_CLASSES = {
    "1d": RelativePositionBias1D,
    "horizontal": RelativePositionBiasHorizontal,
    "vertical": RelativePositionBiasVertical,
}


def create_relative_bias(config: UdopConfig) -> Sequence[RelativePositionBiasBase]:
    """
    创建一个空列表或一个/多个相对偏置对象。

    :param config: 模型的配置对象
    :return: 创建的偏置模块序列
    """
    bias_list = []
    # 检查配置对象是否有 'relative_bias_args' 属性
    if hasattr(config, "relative_bias_args"):
        # 遍历配置中的每个相对偏置参数
        for bias_kwargs_org in config.relative_bias_args:
            # 深拷贝相对偏置参数
            bias_kwargs = deepcopy(bias_kwargs_org)
            # 弹出 'type' 键作为偏置类型
            bias_type = bias_kwargs.pop("type")
            # 获取模型的头数（如果配置对象中有 'num_heads' 属性，则使用它；否则使用 'num_attention_heads' 属性）
            model_num_heads = config.num_heads if hasattr(config, "num_heads") else config.num_attention_heads
            # 如果偏置参数中包含 'num_heads' 键
            if "num_heads" in bias_kwargs:
                # 如果 'num_heads' 不等于模型的头数，则抛出数值错误
                if bias_kwargs["num_heads"] != model_num_heads:
                    raise ValueError("Number of heads must match num of heads in the model")
            else:
                # 否则，将 'num_heads' 设置为模型的头数
                bias_kwargs["num_heads"] = model_num_heads
            # 根据偏置类型创建相对偏置对象，并添加到偏置列表中
            bias_list.append(BIAS_CLASSES[bias_type](**bias_kwargs))  # type: ignore

    return bias_list


class UdopStack(UdopPreTrainedModel):
    """
    这个类基于 `T5Stack`，但修改以考虑图像模态以及2D位置嵌入。
    """

    def __init__(self, config, embed_tokens=None, embed_patches=None):
        super().__init__(config)

        # 初始化嵌入 tokens 和嵌入 patches
        self.embed_tokens = embed_tokens
        self.embed_patches = embed_patches
        self.is_decoder = config.is_decoder  # 是否为解码器
        self._max_length = config.max_length  # 最大长度
        self.num_layers = config.num_layers  # 层数

        # 创建 UdopBlock 模块列表，每层的第一层是否有相对注意力偏置由 'has_relative_attention_bias' 决定
        self.block = nn.ModuleList(
            [UdopBlock(config, has_relative_attention_bias=bool(i == 0)) for i in range(self.num_layers)]
        )

        # 最终层的层归一化
        self.final_layer_norm = UdopLayerNorm(config.d_model, eps=config.layer_norm_epsilon)

        # dropout 层
        self.dropout = nn.Dropout(config.dropout_rate)

        # 如果不是解码器，初始化 2D 单元嵌入
        if not self.is_decoder:
            self.cell_2d_embedding = UdopCellEmbeddings(config.max_2d_position_embeddings, config.hidden_size)

        # 获取编码器位置偏置的权重
        self.relative_bias = self._get_relative_bias(config)

        # 将编码器原始位置偏置的权重绑定到相应位置
        for bias in self.relative_bias.biases:
            if isinstance(bias, RelativePositionBias1D):
                self._tie_or_clone_weights(
                    bias.relative_attention_bias, self.block[0].layer[0].SelfAttention.relative_attention_bias
                )

    @staticmethod
    def _get_relative_bias(config: UdopConfig) -> RelativePositionBiasAggregated:
        # 创建相对位置偏置列表
        relative_bias_list = create_relative_bias(config)
        # 返回聚合的相对位置偏置对象
        return RelativePositionBiasAggregated(relative_bias_list)

    def get_input_embeddings(self):
        # 返回输入嵌入 tokens
        return self.embed_tokens

    def get_output_embeddings(self):
        # 返回输出嵌入 tokens
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        # 设置输入嵌入 tokens
        self.embed_tokens = new_embeddings
    # 定义模型的前向传播方法，接受多个参数来处理输入数据和模型配置
    def forward(
        self,
        input_ids=None,  # 输入的token ids序列
        attention_mask=None,  # 注意力遮罩，指示模型应注意的token位置
        bbox=None,  # 图像bounding box信息
        encoder_hidden_states=None,  # 编码器的隐藏状态
        encoder_attention_mask=None,  # 编码器的注意力遮罩
        inputs_embeds=None,  # 输入的嵌入表示
        pixel_values=None,  # 图像的像素值
        visual_bbox=None,  # 可视信息的bounding box
        image_embeddings=None,  # 图像嵌入
        position_bias=None,  # 位置偏置
        head_mask=None,  # 头部注意力掩码
        cross_attn_head_mask=None,  # 跨注意力头部掩码
        past_key_values=None,  # 过去的键值对
        use_cache=None,  # 是否使用缓存
        output_attentions=None,  # 是否输出注意力权重
        output_hidden_states=None,  # 是否输出隐藏状态
        return_dict=None,  # 是否返回字典形式的结果
@add_start_docstrings(
    "The bare UDOP encoder-decoder Transformer outputting raw hidden-states without any specific head on top.",
    UDOP_START_DOCSTRING,
)
class UdopModel(UdopPreTrainedModel):
    # 定义了共享权重的关键键名列表，用于权重共享
    _tied_weights_keys = [
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
        "encoder.embed_patches.proj.weight",
        "encoder.embed_patches.proj.bias",
        "encoder.relative_bias.biases.0.relative_attention_bias.weight",
        "decoder.relative_bias.biases.0.relative_attention_bias.weight",
    ]

    def __init__(self, config):
        # 调用父类构造函数初始化模型
        super(UdopModel, self).__init__(config)

        # 定义文本和图像嵌入层
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.patch_embed = UdopPatchEmbeddings(config)

        # 复制配置以用于编码器
        encoder_config = deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # 初始化编码器栈
        self.encoder = UdopStack(encoder_config, self.shared, self.patch_embed)

        # 复制配置以用于解码器
        decoder_config = deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        # 初始化解码器栈
        self.decoder = UdopStack(decoder_config, self.shared)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.shared

    # 设置输入嵌入
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    # 获取编码器
    def get_encoder(self):
        return self.encoder

    # 获取解码器
    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(UDOP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Tensor = None,
        attention_mask: Tensor = None,
        bbox: Dict[str, Any] = None,
        pixel_values: Optional[Tensor] = None,
        visual_bbox: Dict[str, Any] = None,
        decoder_input_ids: Optional[Tensor] = None,
        decoder_attention_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        encoder_outputs: Optional[Tensor] = None,
        past_key_values: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        decoder_inputs_embeds: Optional[Tensor] = None,
        decoder_head_mask: Optional[Tensor] = None,
        cross_attn_head_mask: Optional[Tensor] = None,
        use_cache=True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 此处应该继续，但是代码未完整给出
        pass


注释：这段代码定义了一个UDOP模型类，它是一个编码器-解码器的Transformer模型，用于文本生成任务，包含了共享的嵌入层和编码器解码器栈的初始化。
    This class is based on [`T5ForConditionalGeneration`], extended to deal with images and layout (2D) data."""
    # 此处是基于 `T5ForConditionalGeneration` 类扩展而来，用于处理图像和布局（2D）数据。
    UDOP_START_DOCSTRING,
    # 开始 UDOP 文档字符串
# 定义了一个继承自UdopPreTrainedModel的新模型类UdopForConditionalGeneration，用于条件生成任务
class UdopForConditionalGeneration(UdopPreTrainedModel):
    # 定义了一些共享权重的键列表，这些键将在模型中被共享使用
    _tied_weights_keys = [
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
        "encoder.embed_patches.proj.weight",
        "encoder.embed_patches.proj.bias",
        "encoder.relative_bias.biases.0.relative_attention_bias.weight",
        "decoder.relative_bias.biases.0.relative_attention_bias.weight",
        "lm_head.weight",
    ]

    # 初始化函数，接受一个配置对象config
    def __init__(self, config):
        # 调用父类的初始化方法，传入配置对象config
        super(UdopForConditionalGeneration, self).__init__(config)

        # 定义共享的文本和图像嵌入层，使用nn.Embedding创建，形状为(vocab_size, d_model)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        
        # 创建UdopPatchEmbeddings对象，用于图像块的嵌入表示
        self.patch_embed = UdopPatchEmbeddings(config)

        # 复制配置对象config以创建编码器的配置，并设置一些标志位
        encoder_config = deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # 创建编码器对象UdopStack，传入复制后的配置对象、共享的嵌入层和图像块嵌入对象
        self.encoder = UdopStack(encoder_config, self.shared, self.patch_embed)

        # 复制配置对象config以创建解码器的配置，并设置一些标志位和解码层数
        decoder_config = deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        # 创建解码器对象UdopStack，传入复制后的配置对象、共享的嵌入层
        self.decoder = UdopStack(decoder_config, self.shared)

        # 定义语言建模头部的权重，使用nn.Linear创建，输入特征大小为config.d_model，输出大小为config.vocab_size
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 执行额外的初始化操作，可能包括权重初始化和最终处理
        self.post_init()

    # 获取输入嵌入层的方法，返回共享的嵌入层对象
    def get_input_embeddings(self):
        return self.shared

    # 设置输入嵌入层的方法，接受一个新的嵌入层对象new_embeddings，并将其设置为共享的嵌入层
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        # 分别将新的嵌入层设置到编码器和解码器中
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    # 设置输出嵌入层的方法，接受一个新的嵌入层对象new_embeddings，并将其设置为语言建模头部的权重
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 获取输出嵌入层的方法，返回语言建模头部的权重对象
    def get_output_embeddings(self):
        return self.lm_head

    # 获取编码器对象的方法，返回编码器对象
    def get_encoder(self):
        return self.encoder

    # 获取解码器对象的方法，返回解码器对象
    def get_decoder(self):
        return self.decoder

    # 使用装饰器@add_start_docstrings_to_model_forward和@replace_return_docstrings修饰的方法
    # 用于向模型的前向方法添加输入文档字符串和替换返回文档字符串
    # 定义一个方法 `forward`，用于模型的前向传播
    def forward(
        self,
        input_ids: Tensor = None,
        attention_mask: Tensor = None,
        bbox: Dict[str, Any] = None,
        pixel_values: Optional[Tensor] = None,
        visual_bbox: Dict[str, Any] = None,
        decoder_input_ids: Optional[Tensor] = None,
        decoder_attention_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        encoder_outputs: Optional[Tensor] = None,
        past_key_values: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        decoder_inputs_embeds: Optional[Tensor] = None,
        decoder_head_mask: Optional[Tensor] = None,
        cross_attn_head_mask: Optional[Tensor] = None,
        use_cache=True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[Tensor] = None,
    ):
        # 如果 `past_key_values` 不为 None，则截取 `input_ids` 的最后一个位置
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # 返回一个包含各种输入的字典，用于生成模型的输入
        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
            # 使用 `kwargs.get` 获取可能存在的额外输入
            "bbox": kwargs.get("bbox", None),
            "pixel_values": kwargs.get("pixel_values", None),
            "visual_bbox": kwargs.get("visual_bbox", None),
        }

    # 从 `transformers.models.t5.modeling_t5.T5ForConditionalGeneration._reorder_cache` 复制过来的方法
    # 重新排序缓存中的过去键值，以适应束搜索的索引
    # 如果过去的键值未包含在输出中
    # 提示用户快速解码已禁用，无需重新排序
    if past_key_values is None:
        logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
        return past_key_values

    # 初始化重新排序后的解码器过去状态元组
    reordered_decoder_past = ()
    # 遍历每个层级的过去状态
    for layer_past_states in past_key_values:
        # 初始化重新排序后的层级过去状态元组
        reordered_layer_past_states = ()
        # 遍历每个层级内部的过去状态
        for layer_past_state in layer_past_states:
            # 根据束搜索的索引重新排列层级过去状态，以正确的批次索引
            reordered_layer_past_states = reordered_layer_past_states + (
                layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
            )

        # 检查重新排序后的第一个层级过去状态的形状是否与原始形状匹配
        if reordered_layer_past_states[0].shape != layer_past_states[0].shape:
            raise ValueError(
                f"reordered_layer_past_states[0] shape {reordered_layer_past_states[0].shape} and layer_past_states[0] shape {layer_past_states[0].shape} mismatched"
            )
        # 检查重新排序后的层级过去状态元组长度是否与原始长度匹配
        if len(reordered_layer_past_states) != len(layer_past_states):
            raise ValueError(
                f"length of reordered_layer_past_states {len(reordered_layer_past_states)} and length of layer_past_states {len(layer_past_states)} mismatched"
            )

        # 将重新排序后的层级过去状态元组添加到重新排序后的解码器过去状态元组中
        reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)

    # 返回重新排序后的解码器过去状态元组
    return reordered_decoder_past
# 为 UDOP 模型的编码器定义一个新的类，继承自 UdopPreTrainedModel
@add_start_docstrings(
    "The bare UDOP Model transformer outputting encoder's raw hidden-states without any specific head on top.",
    UDOP_START_DOCSTRING,
)
class UdopEncoderModel(UdopPreTrainedModel):
    # 被绑定权重的键列表，用于共享权重的层
    _tied_weights_keys = [
        "encoder.embed_tokens.weight",
        "encoder.embed_patches.proj.weight",
        "encoder.embed_patches.proj.bias",
        "encoder.relative_bias.biases.0.relative_attention_bias.weight",
    ]

    # 初始化函数，接受一个 UdopConfig 类型的参数 config
    def __init__(self, config: UdopConfig):
        super().__init__(config)

        # 文本和图像的嵌入层
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        # 图像补丁的嵌入层
        self.patch_embed = UdopPatchEmbeddings(config)

        # 深拷贝配置以创建编码器的配置
        encoder_config = deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # 创建 UdopStack 编码器层
        self.encoder = UdopStack(encoder_config, self.shared, self.patch_embed)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入层的方法
    def get_input_embeddings(self):
        return self.shared

    # 设置输入嵌入层的方法，接受新的嵌入层参数
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        # 更新编码器的输入嵌入层
        self.encoder.set_input_embeddings(new_embeddings)

    # 获取编码器的方法
    def get_encoder(self):
        return self.encoder

    # 剪枝模型头部的方法，接受一个 heads_to_prune 字典参数
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 遍历 heads_to_prune 字典中的每个层和对应要剪枝的头部列表
        for layer, heads in heads_to_prune.items():
            # 调用编码器中相应层的自注意力模块的剪枝方法
            self.encoder.block[layer].layer[0].SelfAttention.prune_heads(heads)

    # 前向传播方法，接受多个输入参数，并使用装饰器添加了输入输出文档字符串
    @add_start_docstrings_to_model_forward(UDOP_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithAttentionMask, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Tensor = None,
        bbox: Dict[str, Any] = None,
        attention_mask: Tensor = None,
        pixel_values: Optional[Tensor] = None,
        visual_bbox: Dict[str, Any] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 设置输出注意力权重，默认为模型配置中的设定，如果未显式提供则使用模型的配置值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置输出隐藏状态，默认为模型配置中的设定，如果未显式提供则使用模型的配置值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置是否返回字典形式的输出，默认为模型配置中的设定，如果未显式提供则使用模型的配置值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用编码器模型的前向方法，传入各种输入参数和配置选项
        encoder_outputs = self.encoder(
            input_ids=input_ids,                 # 输入的token IDs
            bbox=bbox,                           # 文本框的边界框信息
            visual_bbox=visual_bbox,             # 可视化边界框信息
            pixel_values=pixel_values,           # 图像像素值
            attention_mask=attention_mask,       # 注意力遮罩，控制哪些token参与注意力计算
            inputs_embeds=inputs_embeds,         # 替代token IDs的嵌入表示
            head_mask=head_mask,                 # 多头注意力掩码
            output_attentions=output_attentions, # 控制是否输出注意力权重
            output_hidden_states=output_hidden_states, # 控制是否输出隐藏状态
            return_dict=return_dict,             # 控制是否返回字典形式的输出
        )

        # 返回编码器的输出结果
        return encoder_outputs
```