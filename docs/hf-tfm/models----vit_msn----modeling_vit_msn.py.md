# `.\models\vit_msn\modeling_vit_msn.py`

```
# coding=utf-8
# 版权所有 2022 年 Facebook AI 和 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（"许可证"）许可；
# 除非符合许可证要求，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件基于"原样"提供，
# 没有任何形式的明示或暗示的保证或条件。
# 有关特定语言的权限，请参阅许可证。
""" PyTorch ViT MSN（masked siamese network）模型。"""

# 导入必要的库
import collections.abc
import math
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 从本地库中导入相关函数和类
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_vit_msn import ViTMSNConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 文档用配置和检查点
_CONFIG_FOR_DOC = "ViTMSNConfig"
_CHECKPOINT_FOR_DOC = "facebook/vit-msn-small"
VIT_MSN_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/vit-msn-small",
    # 查看所有 ViTMSN 模型 https://huggingface.co/models?filter=vit_msn
]

class ViTMSNEmbeddings(nn.Module):
    """
    构建 CLS 令牌、位置和补丁嵌入。可选地，也包括掩码令牌。
    """

    def __init__(self, config: ViTMSNConfig, use_mask_token: bool = False) -> None:
        super().__init__()

        # 初始化 CLS 令牌参数
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        
        # 如果使用掩码令牌，则初始化掩码令牌参数
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
        
        # 初始化补丁嵌入层
        self.patch_embeddings = ViTMSNPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        
        # 初始化位置嵌入参数
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
        
        # 初始化 dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # 保存配置信息
        self.config = config
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        # 计算当前嵌入张量中的图块数量和预训练位置编码的位置数量
        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1

        # 如果图块数量与位置数量相等，并且高度与宽度相同，则直接返回预训练的位置编码
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        
        # 从预训练的位置编码中提取类别位置编码和图块位置编码
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]

        # 获取张量的维度信息
        dim = embeddings.shape[-1]

        # 计算图块窗口的高度和宽度
        patch_window_height = height // self.config.patch_size
        patch_window_width = width // self.config.patch_size

        # 为了避免插值时的浮点数误差，向高度和宽度添加一个小数值
        patch_window_height, patch_window_width = patch_window_height + 0.1, patch_window_width + 0.1

        # 将图块位置编码重塑为合适的形状，并进行维度置换
        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        # 使用双三次插值对图块位置编码进行插值
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(
                patch_window_height / math.sqrt(num_positions),
                patch_window_width / math.sqrt(num_positions),
            ),
            mode="bicubic",
            align_corners=False,
        )

        # 再次进行维度置换和重塑，以便与类别位置编码拼接
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        # 返回拼接后的位置编码张量
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
    ) -> torch.Tensor:
        # 获取输入张量的形状信息
        batch_size, num_channels, height, width = pixel_values.shape
        # 使用 patch_embeddings 方法将像素值转换为嵌入向量
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        if bool_masked_pos is not None:
            # 获取嵌入向量的序列长度
            seq_length = embeddings.shape[1]
            # 扩展 mask_token 到与 embeddings 相同的形状，用于替换被遮盖的视觉 tokens
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # 创建一个掩码，将布尔类型的遮盖位置转换为与 mask_tokens 相同类型的张量
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            # 使用 mask 对 embeddings 进行覆盖处理，替换遮盖位置的 tokens
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # 将 [CLS] token 添加到嵌入的 patch tokens 中
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # 在第一维度上连接 cls_tokens 和 embeddings
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # 添加位置编码到每个 token
        if interpolate_pos_encoding:
            # 使用 interpolate_pos_encoding 方法对 embeddings 进行插值处理
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            # 直接添加预先计算好的位置编码到 embeddings
            embeddings = embeddings + self.position_embeddings

        # 对 embeddings 应用 dropout 操作
        embeddings = self.dropout(embeddings)

        # 返回最终的嵌入向量张量
        return embeddings
# 从transformers.models.vit.modeling_vit.ViTPatchEmbeddings复制而来，修改为ViTMSN的实现
class ViTMSNPatchEmbeddings(nn.Module):
    """
    这个类将形状为`(batch_size, num_channels, height, width)`的`pixel_values`转换为形状为`(batch_size, seq_length, hidden_size)`的初始隐藏状态（patch embeddings），
    以供Transformer使用。
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        # 将image_size和patch_size转换为元组（tuple），如果它们不是可迭代对象
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        
        # 计算patch的数量
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        # 使用Conv2d进行投影，将输入的num_channels维度转换为hidden_size维度
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        
        # 检查输入的像素值是否与配置中的num_channels匹配
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        
        # 如果不插值位置编码，检查输入图像的尺寸是否与配置中的image_size匹配
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )
        
        # 对输入的像素值进行投影，并将结果展平和转置，以生成patch embeddings
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        
        return embeddings


# 从transformers.models.vit.modeling_vit.ViTSelfAttention复制而来，修改为ViTMSN的实现
class ViTMSNSelfAttention(nn.Module):
    def __init__(self, config: ViTMSNConfig) -> None:
        super().__init__()
        # 检查隐藏层大小是否可以被注意力头数整除，并且配置中没有嵌入大小的属性
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            # 如果不符合条件，抛出数值错误异常
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        # 初始化注意力头数和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化查询、键、值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        # 初始化注意力概率的Dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # 调整张量形状以便计算注意力得分
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 计算混合查询层
        mixed_query_layer = self.query(hidden_states)

        # 计算键和值的转置以便计算注意力得分
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 计算"查询"和"键"之间的点积，得到原始注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # 将注意力分数除以sqrt(注意力头的大小)进行缩放
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 对注意力分数进行归一化，得到注意力概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 应用Dropout到注意力概率上，实际上是以一定概率将整个token置零以进行注意
        attention_probs = self.dropout(attention_probs)

        # 如果有头部遮罩，应用头部遮罩到注意力概率上
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文层，将注意力概率乘以值层
        context_layer = torch.matmul(attention_probs, value_layer)

        # 调整上下文层的形状以适应输出
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # 根据需要返回上下文层和注意力概率，或者仅返回上下文层
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
# 从 transformers.models.vit.modeling_vit.ViTSelfOutput 复制并修改为 ViT->ViTMSN
class ViTMSNSelfOutput(nn.Module):
    """
    The residual connection is defined in ViTMSNLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """
    
    def __init__(self, config: ViTMSNConfig) -> None:
        super().__init__()
        # 定义一个全连接层，输入和输出维度都为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义一个 Dropout 层，使用 config.hidden_dropout_prob 的概率进行随机失活
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将输入的 hidden_states 应用全连接层 self.dense
        hidden_states = self.dense(hidden_states)
        # 对全连接层的输出应用 Dropout
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# 从 transformers.models.vit.modeling_vit.ViTAttention 复制并修改为 ViT->ViTMSN
class ViTMSNAttention(nn.Module):
    def __init__(self, config: ViTMSNConfig) -> None:
        super().__init__()
        # 初始化 ViTMSNSelfAttention 和 ViTMSNSelfOutput 层
        self.attention = ViTMSNSelfAttention(config)
        self.output = ViTMSNSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        # 找到可裁剪的注意力头部并索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # 裁剪线性层
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储已裁剪的头部
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 将输入的 hidden_states 通过注意力层 self.attention 进行处理
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        # 将注意力层的输出通过 self.output 层进行处理
        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # 如果有需要，添加注意力信息到输出中
        return outputs


# 从 transformers.models.vit.modeling_vit.ViTIntermediate 复制并修改为 ViT->ViTMSN
class ViTMSNIntermediate(nn.Module):
    def __init__(self, config: ViTMSNConfig) -> None:
        super().__init__()
        # 定义一个全连接层，输入维度为 config.hidden_size，输出维度为 config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 根据配置选择激活函数，存储在 self.intermediate_act_fn 中
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
    # 定义一个前向传播方法，接受隐藏状态作为输入张量，并返回处理后的张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用全连接层对隐藏状态进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的结果应用激活函数，例如ReLU等
        hidden_states = self.intermediate_act_fn(hidden_states)

        # 返回处理后的隐藏状态张量作为输出
        return hidden_states
# Copied from transformers.models.vit.modeling_vit.ViTOutput with ViT->ViTMSN
class ViTMSNOutput(nn.Module):
    def __init__(self, config: ViTMSNConfig) -> None:
        super().__init__()
        # 定义一个全连接层，将输入特征维度转换为配置中指定的隐藏层大小
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 定义一个dropout层，用于随机置零输入张量的部分元素，以防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将输入的隐藏状态通过全连接层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对变换后的隐藏状态进行dropout处理
        hidden_states = self.dropout(hidden_states)

        # 将dropout后的隐藏状态与输入张量相加，实现残差连接
        hidden_states = hidden_states + input_tensor

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTLayer with ViT->ViTMSN
class ViTMSNLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: ViTMSNConfig) -> None:
        super().__init__()
        # 定义块大小用于分块前馈网络的处理
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度维度，默认为1，用于处理自注意力
        self.seq_len_dim = 1
        # 定义注意力层，使用ViTMSNAttention类处理自注意力机制
        self.attention = ViTMSNAttention(config)
        # 定义中间层，使用ViTMSNIntermediate类处理中间层操作
        self.intermediate = ViTMSNIntermediate(config)
        # 定义输出层，使用ViTMSNOutput类处理输出层操作
        self.output = ViTMSNOutput(config)
        # 定义前层归一化层，使用LayerNorm对隐藏状态进行归一化
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 定义后层归一化层，同样使用LayerNorm对隐藏状态进行归一化
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 在ViTMSN中，先对隐藏状态进行前层归一化
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # 如果输出注意力权重，将其加入到输出元组中

        # 第一个残差连接
        hidden_states = attention_output + hidden_states

        # 在ViTMSN中，也会在自注意力后进行后层归一化
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # 第二个残差连接
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTEncoder with ViT->ViTMSN
class ViTMSNEncoder(nn.Module):
    def __init__(self, config: ViTMSNConfig) -> None:
        super().__init__()
        self.config = config
        # 使用ViTMSNLayer构建编码器的多层堆叠
        self.layer = nn.ModuleList([ViTMSNLayer(config) for _ in range(config.num_hidden_layers)])
        # 是否使用梯度检查点，默认为False
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        ) -> Union[tuple, BaseModelOutput]:
        # 如果不输出隐藏状态，则初始化一个空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果不输出注意力权重，则初始化一个空元组
        all_self_attentions = () if output_attentions else None

        # 遍历每个 Transformer 层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前层的隐藏状态加入到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 如果启用了梯度检查点并且处于训练状态
            if self.gradient_checkpointing and self.training:
                # 使用梯度检查点函数进行前向传播计算
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                # 普通情况下直接调用当前层的前向传播方法
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            # 更新隐藏状态为当前层的输出的第一个元素（通常是最终隐藏状态）
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力权重，则将当前层的注意力权重加入到 all_self_attentions 中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果需要输出隐藏状态，则将最终的隐藏状态加入到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不以字典形式返回结果，则返回所有非空元素的元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 以 BaseModelOutput 对象形式返回结果，包含最终隐藏状态、所有隐藏状态、所有注意力权重
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
class ViTMSNPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 使用 ViTMSNConfig 作为模型配置类
    config_class = ViTMSNConfig
    # 模型基础名称前缀
    base_model_prefix = "vit"
    # 主要输入名称为 pixel_values
    main_input_name = "pixel_values"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    # todo: Resort to https://github.com/facebookresearch/msn/blob/main/src/deit.py#L200-#L211
    # when creating pre-training scripts.
    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 与 TF 版本略有不同，TF 使用截断正态分布进行初始化
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            # 初始化 LayerNorm 的权重
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


VIT_MSN_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ViTMSNConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

VIT_MSN_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`ViTImageProcessor.__call__`]
            for details.

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
        interpolate_pos_encoding (`bool`, *optional*):
            Whether to interpolate the pre-trained position encodings.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""
# 为 ViTMSNModel 类添加文档字符串，描述该模型输出原始隐藏状态而不带特定的输出头部
@add_start_docstrings(
    "The bare ViTMSN Model outputting raw hidden-states without any specific head on top.",
    VIT_MSN_START_DOCSTRING,
)
class ViTMSNModel(ViTMSNPreTrainedModel):
    def __init__(self, config: ViTMSNConfig, use_mask_token: bool = False):
        # 调用父类的初始化方法，传入配置对象
        super().__init__(config)
        # 保存配置对象
        self.config = config

        # 初始化嵌入层对象
        self.embeddings = ViTMSNEmbeddings(config, use_mask_token=use_mask_token)
        # 初始化编码器对象
        self.encoder = ViTMSNEncoder(config)

        # 初始化 LayerNorm 层，用于归一化隐藏状态向量
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 调用后处理方法，用于权重初始化和最终处理
        self.post_init()

    def get_input_embeddings(self) -> ViTMSNPatchEmbeddings:
        # 返回嵌入层的 patch_embeddings 属性，用于获取输入嵌入
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 遍历需要剪枝的层和头部的字典
        for layer, heads in heads_to_prune.items():
            # 在编码器的指定层中，调用注意力头部的剪枝方法
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(VIT_MSN_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置是否输出注意力权重，默认为模型配置中的输出设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置是否输出隐藏状态，默认为模型配置中的输出设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 设置是否返回字典格式的输出，默认为模型配置中的设置使用返回字典

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        # 如果输入的像素值为 None，则抛出值错误异常

        # Prepare head mask if needed
        # 准备需要的头部掩码
        # 1.0 in head_mask indicate we keep the head
        # head_mask 中的 1.0 表示我们保留该头部
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        # 根据输入的头部掩码参数获取头部掩码，确保其形状符合模型的隐藏层数量和序列长度

        embedding_output = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )
        # 将像素值输入到嵌入层，根据 bool_masked_pos 和 interpolate_pos_encoding 参数进行相应的处理

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 将嵌入输出传入编码器，获取编码器的输出结果

        sequence_output = encoder_outputs[0]
        # 从编码器输出中获取序列输出
        sequence_output = self.layernorm(sequence_output)
        # 序列输出经过 LayerNorm 处理

        if not return_dict:
            head_outputs = (sequence_output,)
            return head_outputs + encoder_outputs[1:]
        # 如果不要求返回字典格式，则返回头部输出和编码器其他输出

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
        # 返回模型的基础输出，包括最后的隐藏状态、隐藏状态列表和注意力权重列表
# 注意：我们尚未为分类头部准备权重。此类用于希望对基础模型（ViTMSNModel）进行微调的用户。
@add_start_docstrings(
    """
    在顶部具有图像分类头的 ViTMSN 模型，例如用于 ImageNet。
    """,
    VIT_MSN_START_DOCSTRING,
)
class ViTMSNForImageClassification(ViTMSNPreTrainedModel):
    def __init__(self, config: ViTMSNConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.vit = ViTMSNModel(config)

        # 分类器头部
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(VIT_MSN_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ImageClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
```