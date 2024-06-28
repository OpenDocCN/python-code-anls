# `.\models\vit_hybrid\modeling_vit_hybrid.py`

```
# coding=utf-8
# 版权 2022 Google AI、Ross Wightman、The HuggingFace Inc. team。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可；
# 除非符合许可证要求，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件，
# 没有任何明示或暗示的保证或条件。
# 有关特定语言的权限，请参阅许可证。

""" PyTorch ViT Hybrid model. """

import collections.abc
import math
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入模型输出和工具类
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging

# 加载背景工具模块
from ...utils.backbone_utils import load_backbone
from .configuration_vit_hybrid import ViTHybridConfig

# 获取记录器对象
logger = logging.get_logger(__name__)

# 用于文档的通用字符串
_CONFIG_FOR_DOC = "ViTHybridConfig"

# 用于文档的基本字符串
_CHECKPOINT_FOR_DOC = "google/vit-hybrid-base-bit-384"
_EXPECTED_OUTPUT_SHAPE = [1, 197, 768]

# 用于图像分类的文档字符串
_IMAGE_CLASS_CHECKPOINT = "google/vit-hybrid-base-bit-384"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

# ViT Hybrid 预训练模型的存档列表
VIT_HYBRID_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/vit-hybrid-base-bit-384",
    # 查看所有 ViT Hybrid 模型的列表：https://huggingface.co/models?filter=vit-hybrid
]

class ViTHybridEmbeddings(nn.Module):
    """
    构建CLS标记、位置和补丁嵌入。可选择添加掩码标记。
    """

    # 从 transformers.models.vit.modeling_vit.ViTEmbeddings.__init__ 复制而来，将 ViT 改为 ViTHybrid
    def __init__(self, config: ViTHybridConfig, use_mask_token: bool = False) -> None:
        super().__init__()

        # 定义CLS标记作为可训练参数
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        
        # 如果需要，定义掩码标记作为可训练参数；否则为None
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
        
        # 初始化补丁嵌入层
        self.patch_embeddings = ViTHybridPatchEmbeddings(config)
        
        # 计算补丁数目（用于位置编码）
        num_patches = self.patch_embeddings.num_patches
        
        # 定义位置编码作为可训练参数
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, config.hidden_size))
        
        # 定义Dropout层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # 保存配置对象
        self.config = config
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        # 计算当前嵌入向量中的补丁数量和预训练位置编码中的位置数量
        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1
        
        # 如果补丁数量和位置数量相等，并且图像的高度和宽度也相等，则直接返回位置编码
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        
        # 从位置编码中提取类别位置编码和补丁位置编码
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        
        # 获取嵌入向量的维度信息
        dim = embeddings.shape[-1]
        
        # 根据配置中的补丁大小调整图像的高度和宽度
        height = height // self.config.patch_size
        width = width // self.config.patch_size
        
        # 为了避免插值时的浮点误差，向高度和宽度添加一个小的数值
        height, width = height + 0.1, width + 0.1
        
        # 将补丁位置编码重塑为合适的形状，以便进行插值
        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        
        # 使用双三次插值对补丁位置编码进行插值
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(height / math.sqrt(num_positions), width / math.sqrt(num_positions)),
            mode="bicubic",
            align_corners=False,
        )
        
        # 检查插值后的高度和宽度是否与预期的一致，否则抛出值错误
        if int(height) != patch_pos_embed.shape[-2] or int(width) != patch_pos_embed.shape[-1]:
            raise ValueError(f"Invalid height or width: {height}, {width}")
        
        # 调整补丁位置编码的形状，并将类别位置编码与之合并
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
    ) -> torch.Tensor:
        # 获取输入张量的形状信息，分别为批量大小、通道数、高度和宽度
        batch_size, num_channels, height, width = pixel_values.shape
        # 使用 patch_embeddings 方法对输入的像素值进行嵌入处理，包括是否插值位置编码的选择
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        # 如果存在 bool_masked_pos，则执行以下操作
        if bool_masked_pos is not None:
            # 获取嵌入后张量的序列长度
            seq_length = embeddings.shape[1]
            # 将 mask_token 在批量维度和序列长度维度上进行扩展
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # 创建用于掩盖被标记视觉标记的 mask 张量，并将其类型转换为与 mask_tokens 相同的类型
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            # 将 embeddings 中被 mask 标记的部分替换为 mask_tokens，保持其它部分不变
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # 将 [CLS] 标记添加到嵌入的补丁标记中
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # 将位置编码添加到每个标记中
        if interpolate_pos_encoding:
            # 如果选择插值位置编码，则对 embeddings 应用 interpolate_pos_encoding 方法
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            # 否则，直接将预定义的位置编码添加到 embeddings 中
            embeddings = embeddings + self.position_embeddings

        # 对嵌入的张量应用 dropout 操作
        embeddings = self.dropout(embeddings)

        # 返回处理后的嵌入张量
        return embeddings
# 定义一个继承自 nn.Module 的类 ViTHybridPatchEmbeddings，用于将形状为 `(batch_size, num_channels, height, width)` 的像素值转换成形状为 `(batch_size, seq_length, hidden_size)` 的初始隐藏状态（补丁嵌入），以便供 Transformer 使用。

    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config, feature_size=None):
        super().__init__()
        
        # 从配置中获取图像大小和补丁大小
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size
        
        # 将图像大小和补丁大小转为可迭代对象，确保其为元组
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        
        # 载入指定配置的主干模型
        self.backbone = load_backbone(config)
        
        # 检查主干模型是否为 "bit" 类型，否则引发 ValueError 异常
        if self.backbone.config.model_type != "bit":
            raise ValueError(f"Backbone model type {self.backbone.model_type} is not supported.")
        
        # 获取主干模型的最终特征维度
        feature_dim = self.backbone.channels[-1]
        
        # 如果未提供特征大小，则从配置中获取主干模型的特征映射形状
        if feature_size is None:
            feature_map = config.backbone_featmap_shape
            
            # 提取特征大小，并设置特征维度为特征映射的第二维度值
            feature_size = feature_map[-2:]
            feature_dim = feature_map[1]
        else:
            # 将特征大小转为元组形式，确保其为可迭代对象
            feature_size = (
                feature_size if isinstance(feature_size, collections.abc.Iterable) else (feature_size, feature_size)
            )
            # 获取主干模型的最终特征维度
            feature_dim = self.backbone.channels[-1]
        
        # 计算网格大小，即特征大小除以补丁大小得到的元组
        self.grid_size = (feature_size[0] // patch_size[0], feature_size[1] // patch_size[1])
        # 计算补丁数量，即网格大小的乘积
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        # 设置图像大小和补丁大小
        self.image_size = image_size
        self.patch_size = patch_size
        # 设置通道数
        self.num_channels = num_channels
        
        # 定义投影层，使用二维卷积将特征维度投影到隐藏大小，卷积核大小为补丁大小，步长为补丁大小
        self.projection = nn.Conv2d(feature_dim, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        # 提取输入张量的形状信息
        _, num_channels, height, width = pixel_values.shape
        
        # 如果通道数不匹配预设的通道数，引发 ValueError 异常
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        
        # 如果不需要插值位置编码
        if not interpolate_pos_encoding:
            # 如果输入图像的高度或宽度与模型预设的图像大小不匹配，引发 ValueError 异常
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )
        
        # 使用主干模型处理输入像素值，提取最终特征映射的最后一个
        features = self.backbone(pixel_values).feature_maps[-1]
        
        # 使用投影层对特征映射进行卷积投影，然后展平为二维张量并转置维度
        embeddings = self.projection(features).flatten(2).transpose(1, 2)
        
        # 返回转换后的补丁嵌入张量
        return embeddings


# 从 transformers.models.vit.modeling_vit.ViTSelfAttention 复制并修改为 ViTHybridSelfAttention
class ViTHybridSelfAttention(nn.Module):
    def __init__(self, config: ViTHybridConfig) -> None:
        super().__init__()
        # 检查隐藏层大小是否是注意力头数的整数倍，并且是否定义了嵌入大小
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 创建用于查询、键和值的线性层，每个线性层输出的大小为 all_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        # 定义用于 dropout 的层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # 调整输入张量 x 的形状，以便适应多头注意力的计算
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 通过查询线性层处理隐藏状态，生成混合查询层
        mixed_query_layer = self.query(hidden_states)

        # 使用 transpose_for_scores 方法对键和值线性层的输出进行适应性调整
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 计算注意力分数，即查询与键的点积
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # 对注意力分数进行缩放
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 对注意力分数进行 softmax 操作，将其转换为概率分布
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 使用 dropout 对注意力概率进行随机失活处理
        attention_probs = self.dropout(attention_probs)

        # 如果指定了 head_mask，则应用 head_mask 到注意力概率上
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文层，即注意力概率与值层的乘积
        context_layer = torch.matmul(attention_probs, value_layer)

        # 调整上下文层的形状以匹配输出要求
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # 根据输出要求构建输出元组
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
# 从 transformers.models.vit.modeling_vit.ViTSelfOutput 复制而来，修改为 ViTHybridSelfOutput
class ViTHybridSelfOutput(nn.Module):
    """
    在 ViTHybridLayer 中定义残差连接，而不是像其他模型一样在此处定义，这是因为每个块前都应用了 layernorm。
    """

    def __init__(self, config: ViTHybridConfig) -> None:
        super().__init__()
        # 定义一个线性层，将输入的隐藏状态映射到相同大小的输出空间
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义一个 dropout 层，以防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 通过线性层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 对处理后的输出应用 dropout
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# 从 transformers.models.vit.modeling_vit.ViTAttention 复制而来，修改为 ViTHybridAttention
class ViTHybridAttention(nn.Module):
    def __init__(self, config: ViTHybridConfig) -> None:
        super().__init__()
        # 初始化注意力机制模块，这里使用 ViTHybridSelfAttention
        self.attention = ViTHybridSelfAttention(config)
        # 初始化输出模块，这里使用 ViTHybridSelfOutput
        self.output = ViTHybridSelfOutput(config)
        # 存储需要剪枝的注意力头信息
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        # 寻找可以剪枝的注意力头及其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # 对线性层进行剪枝
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储已剪枝的注意力头
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 使用注意力机制模块处理隐藏状态
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        # 将注意力机制模块的输出传入输出模块，生成最终的注意力输出
        attention_output = self.output(self_outputs[0], hidden_states)

        # 如果需要输出注意力权重，则将它们添加到输出元组中
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


# 从 transformers.models.vit.modeling_vit.ViTIntermediate 复制而来，修改为 ViTHybridIntermediate
class ViTHybridIntermediate(nn.Module):
    # 初始化函数，用于创建一个新的网络层对象
    def __init__(self, config: ViTHybridConfig) -> None:
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个线性层，将输入大小设置为 config.hidden_size，输出大小为 config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        
        # 检查 config.hidden_act 是否为字符串类型，若是，则从 ACT2FN 字典中获取对应的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            # 否则直接使用 config.hidden_act 作为激活函数
            self.intermediate_act_fn = config.hidden_act

    # 前向传播函数，用于定义层的计算流程
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入的 hidden_states 通过 self.dense 线性层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 将线性变换后的结果通过 self.intermediate_act_fn 激活函数进行非线性变换
        hidden_states = self.intermediate_act_fn(hidden_states)

        # 返回经过处理后的 hidden_states 结果作为输出
        return hidden_states
# Copied from transformers.models.vit.modeling_vit.ViTOutput with ViT->ViTHybrid
# 定义了一个名为 ViTHybridOutput 的类，继承自 nn.Module
class ViTHybridOutput(nn.Module):
    def __init__(self, config: ViTHybridConfig) -> None:
        super().__init__()
        # 创建一个全连接层，将输入特征维度为 config.intermediate_size 转换为 config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 定义一个 dropout 层，以 config.hidden_dropout_prob 的概率丢弃输入
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播方法，接收 hidden_states 和 input_tensor 两个张量作为输入，返回处理后的张量
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 通过全连接层处理 hidden_states
        hidden_states = self.dense(hidden_states)
        # 对处理后的 hidden_states 进行 dropout
        hidden_states = self.dropout(hidden_states)

        # 将处理后的 hidden_states 与 input_tensor 相加，实现残差连接
        hidden_states = hidden_states + input_tensor

        return hidden_states


# 定义了一个名为 ViTHybridLayer 的类，继承自 nn.Module
class ViTHybridLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: ViTHybridConfig) -> None:
        super().__init__()
        # 设置用于分块前馈的 chunk 大小和序列长度维度
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        # 初始化 self-attention、中间层和输出层
        self.attention = ViTHybridAttention(config)
        self.intermediate = ViTHybridIntermediate(config)
        self.output = ViTHybridOutput(config)
        # 设置两个 layernorm 层，分别应用在 self-attention 前后
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # 前向传播方法，接收 hidden_states、head_mask 和 output_attentions 三个参数，返回处理后的张量或元组
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 对 hidden_states 应用 layernorm_before，并传入 self-attention 进行处理
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViTHybrid, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        # 获取 self-attention 的输出张量
        attention_output = self_attention_outputs[0]
        # 如果需要输出注意力权重，则包含在 outputs 中
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # 第一个残差连接，将 attention_output 与原始 hidden_states 相加
        hidden_states = attention_output + hidden_states.to(attention_output.device)

        # 在 ViTHybrid 中，也在 self-attention 后应用 layernorm
        layer_output = self.layernorm_after(hidden_states)
        # 经过中间层处理
        layer_output = self.intermediate(layer_output)

        # 第二个残差连接，输出最终的层输出
        layer_output = self.output(layer_output, hidden_states)

        # 将 layer_output 添加到 outputs 中，并返回
        outputs = (layer_output,) + outputs

        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTEncoder with ViT->ViTHybrid
# 定义了一个名为 ViTHybridEncoder 的类，继承自 nn.Module
class ViTHybridEncoder(nn.Module):
    def __init__(self, config: ViTHybridConfig) -> None:
        super().__init__()
        # 存储配置信息
        self.config = config
        # 创建一系列 ViTHybridLayer 层，数量为 config.num_hidden_layers
        self.layer = nn.ModuleList([ViTHybridLayer(config) for _ in range(config.num_hidden_layers)])
        # 设置梯度检查点为 False
        self.gradient_checkpointing = False
    # 定义前向传播函数，接受隐藏状态、头部掩码、是否输出注意力权重、是否输出隐藏状态、是否返回字典等参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        # 如果需要输出隐藏状态，则初始化一个空元组用于存储所有的隐藏状态
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力权重，则初始化一个空元组用于存储所有的注意力权重
        all_self_attentions = () if output_attentions else None

        # 遍历每个层次的模块
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态加入到所有隐藏状态的元组中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码，如果头部掩码不为None，则获取当前层的掩码；否则为None
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 如果开启了梯度检查点并且在训练阶段，则使用梯度检查点函数对当前层进行调用
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                # 否则直接调用当前层模块，计算输出结果
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            # 更新隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力权重，则将当前层的注意力权重加入到所有注意力权重的元组中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果需要输出隐藏状态，则将最终的隐藏状态加入到所有隐藏状态的元组中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要返回字典形式的输出，则返回非None的元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 否则返回BaseModelOutput对象，包含最终的隐藏状态、所有隐藏状态和所有注意力权重
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values of the input image. This tensor represents the image in the form of batches,
            channels, height, and width.

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask indicating which heads of the self-attention mechanism to mask out. It can be provided as
            a 1D tensor for a single layer model or a 2D tensor for multi-layer models. Values are in the
            range [0, 1]:

            - 1 indicates that the head is **not masked**,
            - 0 indicates that the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to include the attention tensors from all attention layers in the output. Refer to
            the returned tensors for details on the `attentions` field.

        output_hidden_states (`bool`, *optional*):
            Whether or not to include the hidden states from all layers in the output. Refer to the returned
            tensors for details on the `hidden_states` field.

        return_dict (`bool`, *optional*):
            Whether to return a [`~utils.ModelOutput`] instead of a tuple. If True, the output will be wrapped
            in a standardized model output format for ease of use and consistency.
"""
@add_start_docstrings(
    "The bare ViT Hybrid Model transformer outputting raw hidden-states without any specific head on top.",
    VIT_START_DOCSTRING,
)
# 从transformers.models.vit.modeling_vit.ViTModel复制而来，将ViT替换为ViTHybrid
class ViTHybridModel(ViTHybridPreTrainedModel):
    def __init__(self, config: ViTHybridConfig, add_pooling_layer: bool = True, use_mask_token: bool = False):
        super().__init__(config)
        self.config = config

        # 初始化ViTHybridEmbeddings对象，使用是否mask token作为参数
        self.embeddings = ViTHybridEmbeddings(config, use_mask_token=use_mask_token)
        # 初始化ViTHybridEncoder对象
        self.encoder = ViTHybridEncoder(config)

        # 初始化LayerNorm层，使用配置中的hidden_size和layer_norm_eps参数
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 如果add_pooling_layer为True，则初始化ViTHybridPooler对象，否则设为None
        self.pooler = ViTHybridPooler(config) if add_pooling_layer else None

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self) -> ViTHybridPatchEmbeddings:
        # 返回embeddings中的patch_embeddings对象
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 遍历heads_to_prune字典，对每个层和对应的需要prune的头部列表进行操作
        for layer, heads in heads_to_prune.items():
            # 调用encoder中对应层的attention对象的prune_heads方法，进行头部修剪
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        # 如果 output_attentions 为 None，则使用模型配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果 output_hidden_states 为 None，则使用模型配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果 return_dict 为 None，则使用模型配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果 pixel_values 为空，则抛出数值错误异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 准备头部掩码（head_mask）如果需要
        # head_mask 中的 1.0 表示保留该头部
        # attention_probs 的形状为 bsz x n_heads x N x N
        # 输入的 head_mask 形状为 [num_heads] 或 [num_hidden_layers x num_heads]
        # head_mask 被转换为形状 [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # TODO: 可能有更干净的方法来转换输入（从 ImageProcessor 的角度来看）
        
        # 检查 pixel_values 的数据类型是否符合预期，若不符合，则转换为预期的数据类型
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        # 将像素值传入嵌入层，得到嵌入输出
        embedding_output = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )

        # 将嵌入输出传入编码器层
        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # 获取编码器的序列输出
        sequence_output = encoder_outputs[0]
        # 应用层归一化到序列输出
        sequence_output = self.layernorm(sequence_output)
        # 如果存在池化器，则对序列输出进行池化
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        # 如果 return_dict 为 False，则返回头部输出和编码器其他输出
        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        # 如果 return_dict 为 True，则返回包含池化输出在内的 BaseModelOutputWithPooling 对象
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 从transformers.models.vit.modeling_vit.ViTPooler复制而来，将ViT替换为ViTHybrid
class ViTHybridPooler(nn.Module):
    def __init__(self, config: ViTHybridConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # 创建一个线性层，输入输出维度相同
        self.activation = nn.Tanh()  # 使用双曲正切函数作为激活函数

    def forward(self, hidden_states):
        # 通过仅获取第一个令牌对应的隐藏状态来“汇集”模型
        first_token_tensor = hidden_states[:, 0]  # 获取第一个令牌对应的隐藏状态张量
        pooled_output = self.dense(first_token_tensor)  # 将其应用于线性层
        pooled_output = self.activation(pooled_output)  # 应用双曲正切激活函数
        return pooled_output


@add_start_docstrings(
    """
    ViT Hybrid Model transformer with an image classification head on top (a linear layer on top of the final hidden
    state of the [CLS] token) e.g. for ImageNet.
    """,
    VIT_START_DOCSTRING,
)
# 从transformers.models.vit.modeling_vit.ViTForImageClassification复制而来，将ViT替换为ViTHybrid
class ViTHybridForImageClassification(ViTHybridPreTrainedModel):
    def __init__(self, config: ViTHybridConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels  # 设置分类标签数目
        self.vit = ViTHybridModel(config, add_pooling_layer=False)  # 创建一个ViTHybridModel模型实例，不添加汇集层

        # 分类器头部
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        # 如果有分类标签，则创建一个线性层作为分类器头部；否则使用恒等映射函数Identity()

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 确保 return_dict 变量有值，如果没有提供则使用 self.config.use_return_dict 的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 ViT 模型进行前向传播
        outputs = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        # 获取模型输出中的序列输出
        sequence_output = outputs[0]

        # 对序列输出的第一个位置进行分类预测
        logits = self.classifier(sequence_output[:, 0, :])

        # 初始化损失为 None
        loss = None
        if labels is not None:
            # 将 labels 移动到与 logits 相同的设备上，以支持模型并行计算
            labels = labels.to(logits.device)
            # 根据配置确定问题类型，如果未指定则根据 num_labels 来判断
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型计算损失函数
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果不需要返回字典形式的结果，则按元组形式返回输出
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回 ImageClassifierOutput 对象，包含损失、logits、隐藏状态和注意力权重
        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```