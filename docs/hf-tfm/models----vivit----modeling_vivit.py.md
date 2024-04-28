# `.\transformers\models\vivit\modeling_vivit.py`

```py
# 设置代码编码为 utf-8

# 版权声明
# Copyright 2023 Google AI and The HuggingFace Inc. team. All rights reserved.
# 根据 Apache 许可证版本 2.0 授权
# 除非符合许可证要求或经书面同意，否则不得使用此文件
# 您可以在以下网址获取许可证副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除特定法律要求或书面同意外，依照原样分发的软件被分发在 "原样" 基础上，不含任何保证或条件
# 可能适用的附加许可证条款限制具体的权限
# 请参考许可证以了解相关权限和限制
""" PyTorch ViViT 模型。"""


# 导入必要的库和模块
import math
from typing import Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_vivit import VivitConfig

# 设置日志记录器
logger = logging.get_logger(__name__)

# 以下为用于文档的常量
_CHECKPOINT_FOR_DOC = "google/vivit-b-16x2-kinetics400"
_CONFIG_FOR_DOC = "VivitConfig"

VIVIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/vivit-b-16x2-kinetics400",
    # 查看所有 Vivit 模型，请访问 https://huggingface.co/models?filter=vivit
]


# 定义 VivitTubeletEmbeddings 类
class VivitTubeletEmbeddings(nn.Module):
    """
    构建 Vivit Tubelet 嵌入。

    此模块将形状为 (batch_size, num_frames, num_channels, height, width) 的视频批次转换为形状为
    (batch_size, seq_len, hidden_size) 的张量，以供 Transformer 编码器使用。

    seq_len（补丁数量）等于 (number of frames // tubelet_size[0]) * (height // tubelet_size[1]) *
    (width // tubelet_size[2])。
    """

    def __init__(self, config):
        super().__init__()
        # 初始化模块参数
        self.num_frames = config.num_frames
        self.image_size = config.image_size
        self.patch_size = config.tubelet_size
        # 计算补丁数量
        self.num_patches = (
            (self.image_size // self.patch_size[2])
            * (self.image_size // self.patch_size[1])
            * (self.num_frames // self.patch_size[0])
        )
        self.embed_dim = config.hidden_size

        # 投影层，使用三维卷积实现
        self.projection = nn.Conv3d(
            config.num_channels, config.hidden_size, kernel_size=config.tubelet_size, stride=config.tubelet_size
        )
    # 前向传播函数，用于将输入数据进行模型前向计算
    def forward(self, pixel_values):
        # 获取输入数据的形状信息
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        # 检查输入图片尺寸是否符合模型要求
        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size}*{self.image_size})."
            )

        # 将输入数据维度重新排列为(batch_size, num_channels, num_frames, height, width)
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)

        # 将像素值通过投影层进行处理
        x = self.projection(pixel_values)
        # 展平处理后的数据，并将通道维度移到第二维
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        # 返回处理后的数据
        return x
class VivitEmbeddings(nn.Module):
    """
    Vivit Embeddings.

    Creates embeddings from a video using VivitTubeletEmbeddings, adds CLS token and positional embeddings.
    """

    def __init__(self, config):
        super().__init__()

        # 定义 CLS token 参数
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        # 创建 VivitTubeletEmbeddings 对象，用于从视频中创建嵌入
        self.patch_embeddings = VivitTubeletEmbeddings(config)

        # 定义位置嵌入参数
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.patch_embeddings.num_patches + 1, config.hidden_size)
        )
        # 定义 dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, pixel_values):
        # 获取批次大小
        batch_size = pixel_values.shape[0]
        # 将像素值转换为嵌入
        embeddings = self.patch_embeddings(pixel_values)

        # 创建 CLS token
        cls_tokens = self.cls_token.tile([batch_size, 1, 1])

        # 将 CLS token 与嵌入串联起来
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # 给每个标记添加位置编码
        embeddings = embeddings + self.position_embeddings

        # 应用 dropout
        embeddings = self.dropout(embeddings)

        return embeddings


# 复制自 transformers.models.vit.modeling_vit.ViTSelfAttention，并将 ViT 改为 Vivit
class VivitSelfAttention(nn.Module):
    def __init__(self, config: VivitConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 定义 query、key、value 线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        # 定义 dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # 转换 tensor 的维度
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ):
        # 进行自注意力计算
        # 定义一个函数，接收隐藏状态作为输入，返回元组类型的结果
        # 混合查询层，使用self.query对隐藏状态进行处理
        mixed_query_layer = self.query(hidden_states)

        # 对隐藏状态使用self.key，然后调用transpose_for_scores方法进行转置
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        # 对隐藏状态使用self.value，然后调用transpose_for_scores方法进行转置
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        # 对混合查询层进行转置
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 计算"查询"和"键"之间的点积，得到原始注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # 将注意力分数除以sqrt(注意力头大小)，进行归一化
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 将注意力分数规范化为概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 进行Dropout操作，以达到随机“丢弃”部分标记进行关注的效果
        attention_probs = self.dropout(attention_probs)

        # 如果head_mask不为空，则将attention_probs与head_mask相乘
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 通过注意力概率和value_layer计算上下文层
        context_layer = torch.matmul(attention_probs, value_layer)

        # 将context_layer进行转置，并且保持基础数据内存布局
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # 根据新的上下文层形状进行重塑
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # 如果output_attentions不为空，则输出包含context_layer和attention_probs的元组
        # 否则只返回context_layer
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        # 返回结果
        return outputs
# 从transformers.models.vit.modeling_vit.ViTSelfOutput复制代码，并将ViT更改为Vivit
class VivitSelfOutput(nn.Module):
    """
    在VivitLayer中定义了残差连接，而不是在此处（与其他模型相反），这是因为在每个块之前应用了层归一化。
    """

    def __init__(self, config: VivitConfig) -> None:
        super().__init__()
        # 创建一个全连接层，输入维度和输出维度都是config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个Dropout层，丢弃概率为config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将输入hidden_states传递给全连接层，并得到输出
        hidden_states = self.dense(hidden_states)
        # 对全连接层的输出进行Dropout
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# 从transformers.models.vit.modeling_vit.ViTAttention复制代码，并将ViT更改为Vivit
class VivitAttention(nn.Module):
    def __init__(self, config: VivitConfig) -> None:
        super().__init__()
        # 创建VivitSelfAttention层
        self.attention = VivitSelfAttention(config)
        # 创建VivitSelfOutput层
        self.output = VivitSelfOutput(config)
        # 存储已经被剪枝的注意力头的索引
        self.pruned_heads = set()

    # 剪枝注意力头
    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        # 找到可剪枝的头的索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储已剪枝的头
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 调用VivitSelfAttention层的forward方法
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        # 调用VivitSelfOutput层的forward方法
        attention_output = self.output(self_outputs[0], hidden_states)

        # 如果需要输出注意力权重，则添加到输出中
        outputs = (attention_output,) + self_outputs[1:]  
        return outputs


# VivitIntermediate层
class VivitIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，输入维度是config.hidden_size，输出维度是config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 创建一个Dropout层，丢弃概率为config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 如果config.hidden_act是字符串，则将其转换为相应的激活函数，否则直接使用config.hidden_act
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
    # 定义一个前向传播的方法，接收隐藏状态作为输入
    def forward(self, hidden_states):
        # 使用全连接层对隐藏状态进行处理
        hidden_states = self.dense(hidden_states)
        # 使用激活函数对处理后的隐藏状态进行处理
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 对处理后的隐藏状态进行dropout操作
        hidden_states = self.dropout(hidden_states)

        # 返回处理后的隐藏状态
        return hidden_states
class VivitOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # 使用全连接层对隐藏状态进行线性变换
        hidden_states = self.dense(hidden_states)

        # 对线性变换后的隐藏状态进行 dropout 处理
        hidden_states = self.dropout(hidden_states)

        # 将 dropout 后的隐藏状态和输入张量相加
        hidden_states = hidden_states + input_tensor

        return hidden_states


class VivitLayer(nn.Module):
    """This corresponds to the EncoderBlock class in the scenic/vivit implementation."""

    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = VivitAttention(config)
        self.intermediate = VivitIntermediate(config)
        self.output = VivitOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        # 对输入的隐藏状态进行 layernorm 处理
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        # 第一个残差连接
        hidden_states = attention_output + hidden_states

        # 在 self-attention 后再次进行 layernorm 处理
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # 第二个残差连接
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


class VivitEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 创建多个 VivitLayer 实例，并组成 nn.ModuleList
        self.layer = nn.ModuleList([VivitLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    # 初始化隐藏状态和自注意力输出的元组（根据设置判断是否需要输出隐藏状态和自注意力）
    all_hidden_states = () if output_hidden_states else None
    all_self_attentions = () if output_attentions else None

    # 遍历网络层，处理每一层的输出
    for i, layer_module in enumerate(self.layer):
        # 如果需要输出隐藏状态，添加当前的隐藏状态到 all_hidden_states 元组
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 确定是否需要应用头部遮罩
        layer_head_mask = head_mask[i] if head_mask is not None else None

        # 根据是否启用梯度检查点和是否在训练模式下，选择执行方式
        if self.gradient_checkpointing and self.training:
            # 使用梯度检查点来调用层模块
            layer_outputs = self._gradient_checkpointing_func(
                layer_module.__call__,  # 要调用的层模块
                hidden_states,  # 当前的隐藏状态
                layer_head_mask,  # 头部遮罩
                output_attentions,  # 是否输出自注意力
            )
        else:
            # 直接调用层模块
            layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

        # 更新隐藏状态为当前层的输出的第一部分
        hidden_states = layer_outputs[0]

        # 如果需要输出自注意力，将当前层的自注意力添加到 all_self_attentions
        if output_attentions:
            all_self_attentions = all_self_attentions + (layer_outputs[1],)

    # 如果需要输出隐藏状态，添加最终的隐藏状态到 all_hidden_states
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    # 根据 return_dict 参数确定返回值
    if not return_dict:
        # 返回一个元组，包含隐藏状态、隐藏状态序列和自注意力序列（只包括非空部分）
        return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

    # 使用 BaseModelOutput 结构返回结果，包含最后的隐藏状态、隐藏状态序列和自注意力序列
    return BaseModelOutput(
        last_hidden_state=hidden_states,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
    )
# VivitPooler 类，用于池化模型隐藏状态
class VivitPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 全连接层，输入和输出维度相同
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 激活函数
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # 通过取第一个标记对应的隐藏状态来"池化"模型
        first_token_tensor = hidden_states[:, 0]
        # 经过全连接层
        pooled_output = self.dense(first_token_tensor)
        # 经过激活函数
        pooled_output = self.activation(pooled_output)
        return pooled_output


# VivitPreTrainedModel 类，用于处理权重初始化，下载和加载预训练模型
class VivitPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = VivitConfig
    base_model_prefix = "vivit"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            # 对于全连接层和三维卷积层，使用正态分布初始化权重，偏差初始化为零
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 对于嵌入层，使用正态分布初始化权重，如果存在填充索引，将填充索引位置初始化为零
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 对于 LayerNorm 层，偏差初始化为零，权重初始化为1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Parameter):
            # 对于参数，使用正态分布初始化权重
            module.data.normal_(mean=0.0, std=self.config.initializer_range)


# Vivit 模型文档字符串的开始部分
VIVIT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`VivitConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# Vivit 模型输入文档字符串
VIVIT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`):
            输入的像素值。像素值可以使用 [`VivitImageProcessor`] 获取。参见 [`VivitImageProcessor.preprocess`] 获取详情。

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            用于屏蔽自注意力模块中选定头部的掩码。掩码的值范围在 `[0, 1]`：

            - 1 表示该头部 **未被屏蔽**，
            - 0 表示该头部 **被屏蔽**。

        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。返回的张量中包含 `attentions`。有关详细信息，请参见返回的张量下的 `attentions`。

        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。返回的张量中包含 `hidden_states`。有关详细信息，请参见返回的张量下的 `hidden_states`。

        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而不是普通的元组。
# 导入必要的库
import torch
import torch.nn as nn
from ... import add_start_docstrings, replace_return_docstrings
from ...models.vit.modeling_vit import VitPreTrainedModel, ViTPreTrainedModel
from ...models.vit.modeling_vit import ViTPooler, VitEmbeddings, VitLayer, VitModel
from typing import Optional

# 为 ViViT 模型添加文档字符串，说明模型输出原始隐藏状态，没有特定的头部
@add_start_docstrings(
    "The bare ViViT Transformer model outputting raw hidden-states without any specific head on top.",
    VIVIT_START_DOCSTRING,
)
class VivitModel(VivitPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        # 初始化 ViViT 模型的嵌入层和编码器
        self.embeddings = VivitEmbeddings(config)
        self.encoder = VivitEncoder(config)

        # 初始化层归一化和池化器（如果指定添加池化层）
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = VivitPooler(config) if add_pooling_layer else None

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    # 剪枝模型中的注意力头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model.

        Args:
            heads_to_prune:
                dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # ViViT 模型的前向传播函数，添加文档字符串描述输入和输出
    @add_start_docstrings_to_model_forward(VIVIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,



# 为 ViViT 视频分类模型添加文档字符串，说明模型包含一个视频分类头
@add_start_docstrings(
    """ViViT Transformer model with a video classification head on top (a linear layer on top of the final hidden state of the
[CLS] token) e.g. for Kinetics-400.""",
    VIVIT_START_DOCSTRING,
)
class VivitForVideoClassification(VivitPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 确定类别数量
        self.num_labels = config.num_labels
        # 初始化 ViViT 模型，不添加池化层
        self.vivit = VivitModel(config, add_pooling_layer=False)

        # 分类器头部
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # 初始化权重并应用最终处理
        self.post_init()

    # ViViT 视频分类模型的前向传播函数，添加文档字符串描述输入和输出
    @add_start_docstrings_to_model_forward(VIVIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ImageClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
```