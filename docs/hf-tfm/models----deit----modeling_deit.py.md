# `.\models\deit\modeling_deit.py`

```
# 设置文件编码为 utf-8
# 版权声明，版权属于 Facebook AI Research (FAIR)、Ross Wightman 和 The HuggingFace Inc. 团队
#
# 根据 Apache 许可证，禁止未经许可使用该文件
# 可以从以下网址获取许可证副本: http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则不得使用此文件
# 根据许可证规定，软件以“按现状”基础分发，没有任何明示或暗示的担保或条件
# 请查看许可证，了解特定语言的权限和限制

""" PyTorch DeiT 模型。"""

# 导入所需的模块
import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入自定义模块和类
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
    MaskedImageModelingOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_deit import DeiTConfig

# 获取 logger
logger = logging.get_logger(__name__)

# 用于文档的通用字符串
_CONFIG_FOR_DOC = "DeiTConfig"

# 基本文档字符串
_CHECKPOINT_FOR_DOC = "facebook/deit-base-distilled-patch16-224"
_EXPECTED_OUTPUT_SHAPE = [1, 198, 768]

# 图像分类文档字符串
_IMAGE_CLASS_CHECKPOINT = "facebook/deit-base-distilled-patch16-224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

# 预训练模型档案列表
DEIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/deit-base-distilled-patch16-224",
    # 查看所有 DeiT 模型列表: https://huggingface.co/models?filter=deit
]

# DeiTEmbeddings 类，构造 CLS token、蒸馏 token、位置和补丁嵌入。也可以选择添加 mask token
class DeiTEmbeddings(nn.Module):
    
    def __init__(self, config: DeiTConfig, use_mask_token: bool = False) -> None:
        super().__init__()
        
        # 初始化参数：CLS token、蒸馏 token、mask token、位置嵌入和补丁嵌入
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.distillation_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
        self.patch_embeddings = DeiTPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 2, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    # 定义一个方法，用于前向传播
    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        # 调用patch_embeddings方法，将像素值转换为嵌入向量
        embeddings = self.patch_embeddings(pixel_values)
        # 获取嵌入向量的尺寸信息：批次大小、序列长度、嵌入维度
        batch_size, seq_length, _ = embeddings.size()

        # 如果存在掩码位置信息
        if bool_masked_pos is not None:
            # 将掩码令牌进行扩展以匹配嵌入向量的尺寸
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # 创建掩码，用于替换受掩码影响的视觉令牌
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            # 将受掩码影响的嵌入向量替换为掩码令牌
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # 扩展CLS令牌以匹配当前批次的大小
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # 扩展蒸馏令牌以匹配当前批次的大小
        distillation_tokens = self.distillation_token.expand(batch_size, -1, -1)
        # 将CLS令牌、蒸馏令牌以及嵌入向量拼接在一起
        embeddings = torch.cat((cls_tokens, distillation_tokens, embeddings), dim=1)
        # 添加位置嵌入到嵌入向量中
        embeddings = embeddings + self.position_embeddings
        # 对嵌入向量进行dropout操作
        embeddings = self.dropout(embeddings)
        # 返回处理后的嵌入向量
        return embeddings
class DeiTPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        # 初始化 DeiTPatchEmbeddings 类
        super().__init__()
        # 从配置中获取图像大小和补丁大小
        image_size, patch_size = config.image_size, config.patch_size
        # 从配置中获取通道数和隐藏大小
        num_channels, hidden_size = config.num_channels, config.hidden_size

        # 如果图像大小是可迭代的，则保持不变；否则将其转换为元组
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        # 如果补丁大小是可迭代的，则保持不变；否则将其转换为元组
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        # 计算补丁数量
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        # 设置类的属性
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        # 创建卷积层进行投影
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # 获取像素值的形状
        batch_size, num_channels, height, width = pixel_values.shape
        # 如果通道数与配置中的通道数不匹配，则引发 ValueError
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 如果高度或宽度与图像大小不匹配，则引发 ValueError
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        # 对像素值进行投影，并展平后转置，以匹配 Transformer 输入形状
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        # 返回结果张量
        return x


# Copied from transformers.models.vit.modeling_vit.ViTSelfAttention with ViT->DeiT
class DeiTSelfAttention(nn.Module):
    def __init__(self, config: DeiTConfig) -> None:
        # 初始化 DeiTSelfAttention 类
        super().__init__()
        # 如果隐藏大小不是注意力头数的倍数且配置中没有嵌入大小，则引发 ValueError
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        # 设置注意力头数和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 创建查询、键、值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        # 创建 dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    # 对输入的张量进行维度转换，使其适用于注意力得分计算
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # 根据注意力头的数量和大小等参数，改变张量的形状
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播函数
    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 使用查询层对隐藏状态进行处理
        mixed_query_layer = self.query(hidden_states)

        # 使用键层对隐藏状态进行处理，并通过转置得到需要的形状
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        # 使用值层对隐藏状态进行处理，并通过转置得到需要的形状
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        # 使用处理后的查询层对隐藏状态进行处理，并通过转置得到需要的形状
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 通过点积或者内积得到原始的注意力得分
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # 对注意力得分进行缩放
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 将注意力得分规范化成概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 使用dropout函数对注意力概率进行dropout操作
        attention_probs = self.dropout(attention_probs)

        # 如果存在head_mask，则进行头部掩码操作
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 使用注意力概率对值层进行加权求和得到上下文层
        context_layer = torch.matmul(attention_probs, value_layer)

        # 对上下文层进行形状转换
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # 返回上下文层和注意力得分
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
# 从transformers.models.vit.modeling_vit.ViTSelfOutput复制，将ViT->DeiT
class DeiTSelfOutput(nn.Module):
    """
    在这里定义残差连接，而不是像其他模型一样在这里定义（这是由于在每个块之前应用layernorm）。
    """

    def __init__(self, config: DeiTConfig) -> None:
        super().__init__()
        # 创建一个全连接层，将输入的隐藏状态转换为指定大小的输出
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个Dropout层，对输入的隐藏状态进行随机丢弃一部分值
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用全连接层对隐藏状态进行转换
        hidden_states = self.dense(hidden_states)
        # 对转换后的隐藏状态进行随机丢弃
        hidden_states = self.dropout(hidden_states)
        # 返回转换和丢弃后的隐藏状态
        return hidden_states


# 从transformers.models.vit.modeling_vit.ViTAttention复制，将ViT->DeiT
class DeiTAttention(nn.Module):
    def __init__(self, config: DeiTConfig) -> None:
        super().__init__()
        # 创建一个DeiTSelfAttention层
        self.attention = DeiTSelfAttention(config)
        # 创建一个DeiTSelfOutput层
        self.output = DeiTSelfOutput(config)
        # 创建一个空集合，用于存储要剪枝的头部
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        # 根据给定的头部和注意力头部的数量等信息，找到要剪枝的头部以及其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储剪枝的头部
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # 如果有输出的话，加上注意力
        return outputs


# 从transformers.models.vit.modeling_vit.ViTIntermediate复制，将ViT->DeiT
class DeiTIntermediate(nn.Module):
    def __init__(self, config: DeiTConfig) -> None:
        super().__init__()
        # 创建一个全连接层，将输入的隐藏状态转换为中间大小的输出
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果隐藏激活函数是字符串，则使用对应的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
    # 前向传播函数，接受隐藏状态张量作为输入，并返回处理后的隐藏状态张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用全连接层对隐藏状态张量进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的隐藏状态张量应用激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)

        # 返回处理后的隐藏状态张量
        return hidden_states
# 从 transformers.models.vit.modeling_vit.ViTOutput 复制并更名为 DeiTOutput，这是 DeiT 模型的输出层
class DeiTOutput(nn.Module):
    def __init__(self, config: DeiTConfig) -> None:
        super().__init__()
        # 使用全连接层将隐藏状态转换为 DeiT 模型的输出维度
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 使用 dropout 进行正则化
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态通过全连接层
        hidden_states = self.dense(hidden_states)
        # 使用 dropout 进行正则化
        hidden_states = self.dropout(hidden_states)

        # 将全连接层输出与输入张量相加，实现残差连接
        hidden_states = hidden_states + input_tensor

        return hidden_states


# 从 transformers.models.vit.modeling_vit.ViTLayer 复制并更名为 DeiTLayer，对应于 DeiT 模型中的 Block 类
class DeiTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: DeiTConfig) -> None:
        super().__init__()
        # 分块前馈网络的块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度维度
        self.seq_len_dim = 1
        # 多头注意力机制
        self.attention = DeiTAttention(config)
        # 中间层
        self.intermediate = DeiTIntermediate(config)
        # 输出层
        self.output = DeiTOutput(config)
        # 在自注意力之前应用 Layernorm
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 在自注意力之后应用 Layernorm
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 对隐藏状态应用 Layernorm，与自注意力结合
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # 在 DeiT 中，在自注意力之前应用 Layernorm
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # 如果输出注意力权重，则添加自注意力权重

        # 第一个残差连接
        hidden_states = attention_output + hidden_states

        # 在 DeiT 中，还在自注意力之后应用 Layernorm
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # 第二个残差连接在这里执行
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


# 从 transformers.models.vit.modeling_vit.ViTEncoder 复制并更名为 DeiTEncoder
class DeiTEncoder(nn.Module):
    def __init__(self, config: DeiTConfig) -> None:
        super().__init__()
        # DeiT 配置
        self.config = config
        # DeiT 层的列表
        self.layer = nn.ModuleList([DeiTLayer(config) for _ in range(config.num_hidden_layers)])
        # 梯度检查点
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    # 定义函数，指定返回类型为元组或 BaseModelOutput 类型
    ) -> Union[tuple, BaseModelOutput]:
        # 如果输出隐藏状态为真，则初始化空元组，否则初始化为 None
        all_hidden_states = () if output_hidden_states else None
        # 如果输出注意力权重为真，则初始化空元组，否则初始化为 None
        all_self_attentions = () if output_attentions else None

        # 遍历每一层的模块
        for i, layer_module in enumerate(self.layer):
            # 如果输出隐藏状态为真，则用当前隐藏状态更新 all_hidden_states
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果 head_mask 不为 None，则获取当前层的屏蔽头信息，否则初始化为 None
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 如果启用了梯度检查点并且处于训练模式，则调用梯度检查点函数
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                # 否则调用当前层的模块，得到当前层的输出
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            # 用当前层的输出更新隐藏状态
            hidden_states = layer_outputs[0]

            # 如果输出注意力权重为真，则用当前层的注意力权重更新 all_self_attentions
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果输出隐藏状态为真，则用当前隐藏状态更新 all_hidden_states
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果 return_dict 为假，则返回满足条件的值的元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 否则返回 BaseModelOutput 对象
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
# 定义了一个继承自 PreTrainedModel 的抽象类，用于处理权重初始化和下载/加载预训练模型的简单接口
class DeiTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    # 配置文件的类，用于初始化模型的参数
    config_class = DeiTConfig
    # 模型前缀
    base_model_prefix = "deit"
    # 主输入的名称
    main_input_name = "pixel_values"
    # 是否支持梯度检查点
    supports_gradient_checkpointing = True
    # 需要忽略拆分的模块
    _no_split_modules = ["DeiTLayer"]

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 对线性层和卷积层的权重进行初始化
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            # 如果存在偏置，将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            # 将 LayerNorm 的偏置初始化为零，权重初始化为 1.0
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


DEIT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`DeiTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

DEIT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`DeiTImageProcessor.__call__`] for details.

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

@add_start_docstrings(
    "The bare DeiT Model transformer outputting raw hidden-states without any specific head on top.",
    # 定义 API 文档字符串的起始标记
    DEIT_START_DOCSTRING,
)
# 定义一个名为DeiTModel的类，并继承自DeiTPreTrainedModel
class DeiTModel(DeiTPreTrainedModel):
    # 初始化函数，接受DeiTConfig类型的config参数，以及add_pooling_layer和use_mask_token两个布尔类型的可选参数
    def __init__(self, config: DeiTConfig, add_pooling_layer: bool = True, use_mask_token: bool = False) -> None:
        # 调用父类的初始化函数
        super().__init__(config)
        # 将config赋值给实例变量self.config
        self.config = config

        # 创建DeiTEmbeddings对象并赋值给self.embeddings
        self.embeddings = DeiTEmbeddings(config, use_mask_token=use_mask_token)
        # 创建DeiTEncoder对象并赋值给self.encoder
        self.encoder = DeiTEncoder(config)

        # 创建具有 LayerNorm 的神经网络层，并赋值给self.layernorm
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 如果add_pooling_layer为True，则创建DeiTPooler对象并赋值给self.pooler；否则赋值为None
        self.pooler = DeiTPooler(config) if add_pooling_layer else None

        # 初始化权重并应用最终处理
        self.post_init()

    # 定义获取输入嵌入的函数，返回DeiTPatchEmbeddings对象
    def get_input_embeddings(self) -> DeiTPatchEmbeddings:
        return self.embeddings.patch_embeddings

    # 定义剪枝模型头部的函数
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 定义前向传播函数并添加模型文档字符串和代码示例文档字符串
    @add_start_docstrings_to_model_forward(DEIT_INPUTS_DOCSTRING)
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
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        # 根据参数决定是否返回注意力矩阵
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 根据参数决定是否返回所有隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 根据参数决定是否返回字典对象
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果像素值为空，则抛出异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 如果需要，准备头部遮罩
        # 头部遮罩中的1.0表示保留该头部
        # 注意力矩阵的形状为bsz x n_heads x N x N
        # 输入的头部遮罩的形状为[num_heads]或[num_hidden_layers x num_heads]
        # 头部遮罩会被转换成形状[num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # 如果像素值的数据类型与期望的数据类型不符，则将像素值转换成期望的数据类型
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        # 将像素值和可选的布尔遮罩作为输入，计算嵌入输出
        embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)

        # 调用编码器，计算编码器的输出
        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0] # 获取序列输出
        sequence_output = self.layernorm(sequence_output) # 对序列输出进行LayerNorm处理
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None # 对序列输出进行池化操作

        if not return_dict:
            # 如果不需要返回字典对象，则返回序列输出，及可选的池化输出和编码器的其他输出
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        # 如果需要返回字典对象，则返回序列输出、可选的池化输���和编码器的隐藏状态和注意力矩阵
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 定义一个名为DeiTPooler的类，继承自nn.Module
class DeiTPooler(nn.Module):
    def __init__(self, config: DeiTConfig):
        super().__init__()
        # 定义一个全连接层，输入维度为config.hidden_size，输出维度为config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义一个激活函数为Tanh
        self.activation = nn.Tanh()

    # 定义前向传播函数
    def forward(self, hidden_states):
        # 通过取第一个token对应的隐藏状态来"池化"模型
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# 定义一个名为DeiTForMaskedImageModeling的类，继承自DeiTPreTrainedModel
class DeiTForMaskedImageModeling(DeiTPreTrainedModel):
    # 初始化函数
    def __init__(self, config: DeiTConfig) -> None:
        super().__init__(config)

        # 实例化DeiTModel，并禁用添加池化层，使用遮罩token
        self.deit = DeiTModel(config, add_pooling_layer=False, use_mask_token=True)

        # 定义一个包含卷积层和像素重排的顺序容器
        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=config.hidden_size,
                out_channels=config.encoder_stride**2 * config.num_channels,
                kernel_size=1,
            ),
            nn.PixelShuffle(config.encoder_stride),
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # 定义前向传播函数，接收输入像素值、遮罩位置、头部遮罩、是否输出注意力权重、是否输出隐藏状态、是否返回字典
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        

# 定义一个名为DeiTForImageClassification的类，继承自DeiTPreTrainedModel
class DeiTForImageClassification(DeiTPreTrainedModel):
    # 初始化函数
    def __init__(self, config: DeiTConfig) -> None:
        super().__init__(config)

        # 获取标签数量
        self.num_labels = config.num_labels
        # 实例化DeiTModel，并禁用添加池化层
        self.deit = DeiTModel(config, add_pooling_layer=False)

        # 定义分类器头部，如果标签数量大于0，则是一个全连接层；否则是一个恒等映射
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # 初始化权重并应用最终处理
        self.post_init()
    # 使用装饰器替换返回文档字符串，设置输出类型为ImageClassifierOutput，配置类为_CONFIG_FOR_DOC
    @replace_return_docstrings(output_type=ImageClassifierOutput, config_class=_CONFIG_FOR_DOC)
    # 前向传播函数，接收像素数值、头部掩码、标签、输出注意力、隐藏状态等参数，并定义为可选的张量
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,  # 输入像素值，默认为None
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码，默认为None
        labels: Optional[torch.Tensor] = None,  # 标签，默认为None
        output_attentions: Optional[bool] = None,  # 输出注意力，默认为None
        output_hidden_states: Optional[bool] = None,  # 输出隐藏状态，默认为None
        return_dict: Optional[bool] = None,  # 返回字典，默认为None
# 定义了一个名为`DeiTForImageClassificationWithTeacherOutput`的数据类，用于存储`DeiTForImageClassificationWithTeacher`模型的输出结果
@dataclass
class DeiTForImageClassificationWithTeacherOutput(ModelOutput):
    """
    Output type of [`DeiTForImageClassificationWithTeacher`].

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Prediction scores as the average of the cls_logits and distillation logits.
        cls_logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Prediction scores of the classification head (i.e. the linear layer on top of the final hidden state of the
            class token).
        distillation_logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Prediction scores of the distillation head (i.e. the linear layer on top of the final hidden state of the
            distillation token).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    logits: torch.FloatTensor = None
    cls_logits: torch.FloatTensor = None
    distillation_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# 定义了一个`DeiTForImageClassificationWithTeacher`类，继承自`DeiTPreTrainedModel`类
@add_start_docstrings(
    """
    DeiT Model transformer with image classification heads on top (a linear layer on top of the final hidden state of
    the [CLS] token and a linear layer on top of the final hidden state of the distillation token) e.g. for ImageNet.

    .. warning::

           This model supports inference-only. Fine-tuning with distillation (i.e. with a teacher) is not yet
           supported.
    """,
    DEIT_START_DOCSTRING,
)
class DeiTForImageClassificationWithTeacher(DeiTPreTrainedModel):
    # 初始化函数，接收一个 DeiTConfig 类型的参数
    def __init__(self, config: DeiTConfig) -> None:
        # 调用父类的初始化函数
        super().__init__(config)

        # 保存配置中的类别数目
        self.num_labels = config.num_labels
        # 创建 DeiT 模型，不添加池化层
        self.deit = DeiTModel(config, add_pooling_layer=False)

        # 分类器头部
        # 如果类别数目大于0，则创建线性全连接层作为分类器，否则创建一个恒等映射（等于没有层）
        self.cls_classifier = (
            nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )
        self.distillation_classifier = (
            nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # 重写父类的 forward 方法
    @add_start_docstrings_to_model_forward(DEIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=DeiTForImageClassificationWithTeacherOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, DeiTForImageClassificationWithTeacherOutput]:
        # 如果 return_dict 参数为 None，则使用配置中的 use_return_dict 属性
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 DeiTModel 的 forward 方法，得到输出结果
        outputs = self.deit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取序列输出
        sequence_output = outputs[0]

        # 通过线性全连接层对序列输出的第一个位置进行分类
        cls_logits = self.cls_classifier(sequence_output[:, 0, :])
        # 通过线性全连接层对序列输出的第二个位置进行蒸馏分类
        distillation_logits = self.distillation_classifier(sequence_output[:, 1, :])

        # 在推理过程中，返回两个分类器预测结果的平均值
        logits = (cls_logits + distillation_logits) / 2

        # 如果不要求返回字典，则返回一个元组，包含预测结果和 DeiTModel 的其他输出
        if not return_dict:
            output = (logits, cls_logits, distillation_logits) + outputs[1:]
            return output

        # 否则，返回 DeiTForImageClassificationWithTeacherOutput 类型的对象，包含预测结果和 DeiTModel 的其他输出
        return DeiTForImageClassificationWithTeacherOutput(
            logits=logits,
            cls_logits=cls_logits,
            distillation_logits=distillation_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```