# `.\models\deit\modeling_deit.py`

```py
# 设置编码格式为 UTF-8
# 版权声明，版权归Facebook AI Research (FAIR)，Ross Wightman，The HuggingFace Inc. team所有
#
# 根据Apache许可证2.0版授权使用此文件；
# 除非符合许可证的要求，否则您不能使用本文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按"原样"分发，
# 没有任何形式的担保或条件，包括但不限于明示或暗示的担保或条件。
# 有关详细信息，请参阅许可证。
""" PyTorch DeiT模型。"""

import collections.abc  # 导入collections.abc模块
import math  # 导入math模块
from dataclasses import dataclass  # 从dataclasses模块导入dataclass装饰器
from typing import Optional, Set, Tuple, Union  # 导入类型提示相关内容

import torch  # 导入PyTorch库
import torch.utils.checkpoint  # 导入PyTorch的checkpoint工具
from torch import nn  # 从PyTorch导入神经网络模块
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss  # 从PyTorch导入损失函数类

from ...activations import ACT2FN  # 导入激活函数映射表
from ...modeling_outputs import (  # 导入模型输出相关类
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
    MaskedImageModelingOutput,
)
from ...modeling_utils import PreTrainedModel  # 导入预训练模型基类
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer  # 导入模型剪枝相关工具
from ...utils import (  # 导入实用工具函数和类
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_deit import DeiTConfig  # 导入DeiT模型的配置类

logger = logging.get_logger(__name__)  # 获取日志记录器

# 通用文档字符串
_CONFIG_FOR_DOC = "DeiTConfig"

# 基础文档字符串
_CHECKPOINT_FOR_DOC = "facebook/deit-base-distilled-patch16-224"
_EXPECTED_OUTPUT_SHAPE = [1, 198, 768]

# 图像分类文档字符串
_IMAGE_CLASS_CHECKPOINT = "facebook/deit-base-distilled-patch16-224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

DEIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/deit-base-distilled-patch16-224",
    # 查看所有DeiT模型：https://huggingface.co/models?filter=deit
]


class DeiTEmbeddings(nn.Module):
    """
    构建CLS令牌、蒸馏令牌、位置和补丁嵌入。可选地，还包括掩码令牌。
    """

    def __init__(self, config: DeiTConfig, use_mask_token: bool = False) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))  # 定义CLS令牌参数
        self.distillation_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))  # 定义蒸馏令牌参数
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None  # 如果使用掩码令牌，则定义掩码令牌参数
        self.patch_embeddings = DeiTPatchEmbeddings(config)  # 初始化补丁嵌入层
        num_patches = self.patch_embeddings.num_patches  # 获取补丁数
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 2, config.hidden_size))  # 定义位置嵌入参数
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # 定义dropout层
    # 定义一个方法 `forward`，用于模型前向传播，接受像素值张量 `pixel_values` 和可选的布尔类型掩码张量 `bool_masked_pos`，返回处理后的张量
    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        # 使用 `patch_embeddings` 方法将像素值张量转换为嵌入张量
        embeddings = self.patch_embeddings(pixel_values)
        # 获取嵌入张量的批处理大小、序列长度和嵌入维度
        batch_size, seq_length, _ = embeddings.size()

        # 如果存在掩码张量，则用 `mask_token` 替换掩码的视觉标记
        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # 创建掩码张量，并将其转换为与 `mask_tokens` 相同的数据类型
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            # 用 `mask` 控制张量 `embeddings` 中的掩码视觉标记部分，保持未掩码部分不变
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # 将 `cls_token` 扩展为与批处理大小相同的形状
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # 将 `distillation_token` 扩展为与批处理大小相同的形状
        distillation_tokens = self.distillation_token.expand(batch_size, -1, -1)
        # 在维度1上连接 `cls_tokens`、`distillation_tokens` 和 `embeddings`
        embeddings = torch.cat((cls_tokens, distillation_tokens, embeddings), dim=1)
        # 将位置嵌入加到 `embeddings` 中
        embeddings = embeddings + self.position_embeddings
        # 应用 dropout 操作到 `embeddings` 中
        embeddings = self.dropout(embeddings)
        # 返回处理后的嵌入张量 `embeddings`
        return embeddings
# 定义一个名为 DeiTPatchEmbeddings 的类，继承自 nn.Module，用于将形状为 `(batch_size, num_channels, height, width)` 的像素值转换为形状为 `(batch_size, seq_length, hidden_size)` 的初始隐藏状态（patch embeddings），以供 Transformer 模型使用。
class DeiTPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        # 从配置中获取图像尺寸和patch尺寸
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        # 如果图像尺寸和patch尺寸不是可迭代对象，则转换为元组
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        # 计算图像中的patch数量
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        # 使用 nn.Conv2d 定义投影层，将图像的每个patch投影到隐藏空间
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # 获取输入张量的形状信息
        batch_size, num_channels, height, width = pixel_values.shape
        # 如果输入通道数与配置中的不匹配，则抛出异常
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 如果输入图像尺寸与配置中的不匹配，则抛出异常
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        # 对输入的像素值进行投影并展平，然后转置以匹配 Transformer 的输入格式
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return x


# 从 transformers.models.vit.modeling_vit.ViTSelfAttention 复制并修改为 DeiT
class DeiTSelfAttention(nn.Module):
    def __init__(self, config: DeiTConfig) -> None:
        super().__init__()
        # 如果隐藏大小不能被注意力头数整除，并且配置中没有嵌入大小的属性，则抛出异常
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        # 设置注意力头数和每个注意力头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 定义查询、键和值的线性映射层，用于生成查询、键和值的表示
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        # 定义用于注意力概率的 dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    # 对输入张量进行形状转换，将最后两个维度重新组织为多个注意头的形式
    new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
    x = x.view(new_x_shape)
    return x.permute(0, 2, 1, 3)

def forward(
    self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
    # 通过查询函数处理隐藏状态，生成混合查询层
    mixed_query_layer = self.query(hidden_states)

    # 将键层进行分组，以适应多头注意力的计算需求
    key_layer = self.transpose_for_scores(self.key(hidden_states))
    # 将值层进行分组，以适应多头注意力的计算需求
    value_layer = self.transpose_for_scores(self.value(hidden_states))
    # 将查询层进行分组，以适应多头注意力的计算需求
    query_layer = self.transpose_for_scores(mixed_query_layer)

    # 计算原始的注意力分数，通过查询和键的点积得到
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

    # 对注意力分数进行除以注意力头大小的平方根的缩放
    attention_scores = attention_scores / math.sqrt(self.attention_head_size)

    # 对注意力分数进行 softmax 操作，将其归一化为注意力概率
    attention_probs = nn.functional.softmax(attention_scores, dim=-1)

    # 对注意力概率进行 dropout 操作，以减少过拟合风险
    attention_probs = self.dropout(attention_probs)

    # 如果给定了头部掩码，则将注意力概率与掩码相乘，实现头部掩蔽
    if head_mask is not None:
        attention_probs = attention_probs * head_mask

    # 计算上下文层，通过注意力概率加权值层的值
    context_layer = torch.matmul(attention_probs, value_layer)

    # 调整上下文层的维度顺序，使其恢复到原始输入的维度形状
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    # 重新组织上下文层的形状，将多头注意力的结果合并
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(new_context_layer_shape)

    # 根据是否需要输出注意力权重，返回相应的结果
    outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

    return outputs
# Copied from transformers.models.vit.modeling_vit.ViTSelfOutput with ViT->DeiT
class DeiTSelfOutput(nn.Module):
    """
    The residual connection is defined in DeiTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: DeiTConfig) -> None:
        super().__init__()
        # 定义线性层，用于变换隐藏状态的维度
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义 Dropout 层，用于随机置零隐藏状态中的部分元素，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用线性层变换隐藏状态
        hidden_states = self.dense(hidden_states)
        # 对变换后的隐藏状态进行 Dropout 处理
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTAttention with ViT->DeiT
class DeiTAttention(nn.Module):
    def __init__(self, config: DeiTConfig) -> None:
        super().__init__()
        # 初始化自注意力层，用于计算注意力分布
        self.attention = DeiTSelfAttention(config)
        # 初始化自注意力输出层，用于处理自注意力层的输出结果
        self.output = DeiTSelfOutput(config)
        # 初始化一个空集合，用于记录剪枝过的注意力头
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        # 查找可剪枝的注意力头并获取索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并记录剪枝过的注意力头
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 进行自注意力计算
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        # 使用自注意力输出层处理自注意力计算的结果和原始隐藏状态
        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # 如果需要输出注意力分布，则添加到输出中
        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTIntermediate with ViT->DeiT
class DeiTIntermediate(nn.Module):
    def __init__(self, config: DeiTConfig) -> None:
        super().__init__()
        # 初始化线性层，用于变换隐藏状态的维度至中间维度
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            # 如果配置中的隐藏激活函数是字符串，则使用预定义的激活函数字典中对应的函数
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            # 否则直接使用配置中的激活函数
            self.intermediate_act_fn = config.hidden_act
    # 前向传播函数，接收隐藏状态张量作为输入，返回处理后的隐藏状态张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态张量通过全连接层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 应用激活函数到线性变换后的隐藏状态张量
        hidden_states = self.intermediate_act_fn(hidden_states)

        # 返回处理后的隐藏状态张量作为输出
        return hidden_states
# Copied from transformers.models.vit.modeling_vit.ViTOutput with ViT->DeiT
class DeiTOutput(nn.Module):
    def __init__(self, config: DeiTConfig) -> None:
        super().__init__()
        # 定义一个全连接层，将中间尺寸的特征转换为隐藏层尺寸
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 定义一个用于随机失活的层，以防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将输入的隐藏状态进行全连接层的变换
        hidden_states = self.dense(hidden_states)
        # 对全连接层的输出进行随机失活
        hidden_states = self.dropout(hidden_states)

        # 将全连接层的输出与输入张量相加，实现残差连接
        hidden_states = hidden_states + input_tensor

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTLayer with ViT->DeiT
class DeiTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: DeiTConfig) -> None:
        super().__init__()
        # 定义用于分块的前馈传播的块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度维度设置为1
        self.seq_len_dim = 1
        # 使用DeiTAttention类来定义注意力机制
        self.attention = DeiTAttention(config)
        # 使用DeiTIntermediate类定义中间层结构
        self.intermediate = DeiTIntermediate(config)
        # 使用DeiTOutput类定义输出层结构
        self.output = DeiTOutput(config)
        # 在隐藏层上应用LayerNorm，设置eps为config中的层归一化参数
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 同样在隐藏层上应用LayerNorm，设置eps为config中的层归一化参数
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 对输入的隐藏状态应用LayerNorm，然后传入注意力机制
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in DeiT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        # 获取注意力机制的输出
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # 如果需要输出注意力权重，也一并返回

        # 第一个残差连接
        hidden_states = attention_output + hidden_states

        # 在DeiT中，还会在自注意力后应用LayerNorm
        layer_output = self.layernorm_after(hidden_states)
        # 将LayerNorm后的输出传入中间层
        layer_output = self.intermediate(layer_output)

        # 第二个残差连接
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTEncoder with ViT->DeiT
class DeiTEncoder(nn.Module):
    def __init__(self, config: DeiTConfig) -> None:
        super().__init__()
        self.config = config
        # 创建一个由多个DeiTLayer组成的层列表，列表长度由config中的层数决定
        self.layer = nn.ModuleList([DeiTLayer(config) for _ in range(config.num_hidden_layers)])
        # 梯度检查点设为False，用于控制是否使用梯度检查点来节省内存
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        ) -> Union[tuple, BaseModelOutput]:
        # 如果不输出隐藏状态，则初始化为空元组；否则为None，以便后续累积隐藏状态
        all_hidden_states = () if output_hidden_states else None
        # 如果不输出注意力权重，则初始化为空元组；否则为None，以便后续累积注意力权重
        all_self_attentions = () if output_attentions else None

        # 遍历模型的每一层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态加入到所有隐藏状态的元组中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码（如果存在）
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 如果启用了梯度检查点且处于训练模式，则使用梯度检查点功能
            if self.gradient_checkpointing and self.training:
                # 调用梯度检查点函数，用于推断当前层的输出
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                # 否则，直接调用当前层的前向传播函数
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            # 更新隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力权重，则将当前层的注意力权重加入到所有注意力权重的元组中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果需要输出隐藏状态，则将最终的隐藏状态加入到所有隐藏状态的元组中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要以字典形式返回结果，则以元组形式返回相应的结果组件
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 否则，以BaseModelOutput对象的形式返回结果，其中包含最终的隐藏状态、所有隐藏状态和所有注意力权重
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
class DeiTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 使用 DeiTConfig 作为配置类
    config_class = DeiTConfig
    # 基础模型前缀为 "deit"
    base_model_prefix = "deit"
    # 主要输入名称为 "pixel_values"
    main_input_name = "pixel_values"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 不需要分割的模块列表
    _no_split_modules = ["DeiTLayer"]

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 将输入向上转换为 `fp32`，然后转换回所需的 `dtype`，以避免在 `half` 模式下出现 `trunc_normal_cpu` 未实现的问题
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                # 如果存在偏置，则将其数据清零
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            # 如果是 LayerNorm 模块，则将偏置数据清零，权重数据填充为 1.0
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
    DEIT_START_DOCSTRING,



# 插入一个名为 DEIT_START_DOCSTRING 的常量，用于开始一个文档字符串的标记
)
# 定义一个新的类 DeiTModel，继承自 DeiTPreTrainedModel
class DeiTModel(DeiTPreTrainedModel):
    # 初始化方法，接受配置 config 和两个可选参数 add_pooling_layer 和 use_mask_token
    def __init__(self, config: DeiTConfig, add_pooling_layer: bool = True, use_mask_token: bool = False) -> None:
        # 调用父类的初始化方法
        super().__init__(config)
        # 将配置信息保存到实例变量中
        self.config = config

        # 初始化嵌入层，使用 DeiTEmbeddings 类，并根据 use_mask_token 参数确定是否使用掩码标记
        self.embeddings = DeiTEmbeddings(config, use_mask_token=use_mask_token)
        # 初始化编码器，使用 DeiTEncoder 类
        self.encoder = DeiTEncoder(config)

        # 初始化层归一化，使用 nn.LayerNorm 类，设置归一化尺寸和 epsilon 值
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 如果 add_pooling_layer 为 True，则初始化池化层，使用 DeiTPooler 类；否则设为 None
        self.pooler = DeiTPooler(config) if add_pooling_layer else None

        # 执行初始化后处理方法
        self.post_init()

    # 获取输入嵌入层的方法
    def get_input_embeddings(self) -> DeiTPatchEmbeddings:
        return self.embeddings.patch_embeddings

    # 剪枝模型头部的方法
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 遍历需要剪枝的层及其对应的头部列表
        for layer, heads in heads_to_prune.items():
            # 在编码器的指定层上，调用注意力机制的剪枝方法
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 使用文档字符串装饰器添加模型前向方法的说明文档
    @add_start_docstrings_to_model_forward(DEIT_INPUTS_DOCSTRING)
    # 使用代码示例文档字符串装饰器添加示例代码和参数说明
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    # 模型的前向传播方法
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
        # 根据输入参数或模型配置决定是否输出注意力权重
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 根据输入参数或模型配置决定是否输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 根据输入参数或模型配置决定是否使用返回字典形式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果像素值为空，则抛出数值错误异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 准备头部遮罩（head_mask），用于多头注意力机制的控制
        # head_mask 中 1.0 表示保留该头部的注意力权重
        # attention_probs 的形状为 bsz x n_heads x N x N
        # 输入的 head_mask 形状为 [num_heads] 或 [num_hidden_layers x num_heads]
        # head_mask 被转换为 [num_hidden_layers x batch x num_heads x seq_length x seq_length] 的形状
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # TODO: 可能有更清晰的方式来转换输入（来自 `ImageProcessor` 的一侧？）
        # 检查像素值的数据类型是否与预期的数据类型一致，如果不一致则进行转换
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        # 将像素值和可选的布尔掩码位置作为输入，进行嵌入编码处理
        embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)

        # 将嵌入输出作为编码器的输入，进行编码器的前向传播
        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 提取编码器的序列输出
        sequence_output = encoder_outputs[0]
        # 应用层归一化到序列输出
        sequence_output = self.layernorm(sequence_output)
        # 如果存在池化层，则将序列输出池化为池化输出
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        # 如果不使用返回字典，则返回头部输出和编码器的其他输出
        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        # 如果使用返回字典，则返回包含编码器输出的 BaseModelOutputWithPooling 对象
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# Copied from transformers.models.vit.modeling_vit.ViTPooler with ViT->DeiT
class DeiTPooler(nn.Module):
    def __init__(self, config: DeiTConfig):
        super().__init__()
        # 初始化一个全连接层，输入和输出维度都是 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 激活函数使用 Tanh
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # 通过取第一个 token 对应的隐藏状态来"池化"模型
        first_token_tensor = hidden_states[:, 0]
        # 将第一个 token 的隐藏状态作为输入，经过全连接层得到池化输出
        pooled_output = self.dense(first_token_tensor)
        # 应用 Tanh 激活函数
        pooled_output = self.activation(pooled_output)
        # 返回池化后的输出
        return pooled_output


@add_start_docstrings(
    """DeiT Model with a decoder on top for masked image modeling, as proposed in [SimMIM](https://arxiv.org/abs/2111.09886).

    <Tip>

    Note that we provide a script to pre-train this model on custom data in our [examples
    directory](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining).

    </Tip>
    """,
    DEIT_START_DOCSTRING,
)
# DeiTForMaskedImageModeling 类，继承自 DeiTPreTrainedModel
class DeiTForMaskedImageModeling(DeiTPreTrainedModel):
    def __init__(self, config: DeiTConfig) -> None:
        super().__init__(config)

        # 初始化 DeiTModel，设置 add_pooling_layer=False 和 use_mask_token=True
        self.deit = DeiTModel(config, add_pooling_layer=False, use_mask_token=True)

        # 定义解码器，使用卷积层和像素混洗层
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

    @add_start_docstrings_to_model_forward(DEIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MaskedImageModelingOutput, config_class=_CONFIG_FOR_DOC)
    # 定义前向传播函数，接收像素值、布尔掩码位置、头部掩码等参数
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,



@add_start_docstrings(
    """
    DeiT Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.
    """,
    DEIT_START_DOCSTRING,
)
# DeiTForImageClassification 类，继承自 DeiTPreTrainedModel
class DeiTForImageClassification(DeiTPreTrainedModel):
    def __init__(self, config: DeiTConfig) -> None:
        super().__init__(config)

        # 设置类别数量
        self.num_labels = config.num_labels
        # 初始化 DeiTModel，不使用额外的池化层
        self.deit = DeiTModel(config, add_pooling_layer=False)

        # 分类器头部，根据类别数量确定输出维度
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(DEIT_INPUTS_DOCSTRING)
    # 使用装饰器替换返回文档字符串，指定输出类型为ImageClassifierOutput，配置类为_CONFIG_FOR_DOC
    @replace_return_docstrings(output_type=ImageClassifierOutput, config_class=_CONFIG_FOR_DOC)
    # 定义神经网络的前向传播函数，接受以下参数：
    # pixel_values: 可选的 torch.Tensor，表示像素值
    # head_mask: 可选的 torch.Tensor，表示头部屏蔽（mask）
    # labels: 可选的 torch.Tensor，表示标签数据
    # output_attentions: 可选的 bool 值，控制是否输出注意力权重
    # output_hidden_states: 可选的 bool 值，控制是否输出隐藏状态
    # return_dict: 可选的 bool 值，控制是否返回结果字典
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 数据类，用于保存`DeiTForImageClassificationWithTeacher`模型的输出结果
@dataclass
class DeiTForImageClassificationWithTeacherOutput(ModelOutput):
    """
    `DeiTForImageClassificationWithTeacher`的输出类型。

    Args:
        logits (`torch.FloatTensor`，形状为 `(batch_size, config.num_labels)`):
            预测分数，是`cls_logits`和`distillation_logits`的平均值。
        cls_logits (`torch.FloatTensor`，形状为 `(batch_size, config.num_labels)`):
            分类头部的预测分数（即最终隐藏状态的类令牌上的线性层）。
        distillation_logits (`torch.FloatTensor`，形状为 `(batch_size, config.num_labels)`):
            蒸馏头部的预测分数（即最终隐藏状态的蒸馏令牌上的线性层）。
        hidden_states (`tuple(torch.FloatTensor)`，*可选*，当传递 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回):
            一个元组，包含 `torch.FloatTensor`（一个用于嵌入的输出 + 每个层的输出），形状为 `(batch_size, sequence_length, hidden_size)`。
            模型在每一层输出的隐藏状态，以及初始嵌入的输出。
        attentions (`tuple(torch.FloatTensor)`，*可选*，当传递 `output_attentions=True` 或 `config.output_attentions=True` 时返回):
            一个元组，包含 `torch.FloatTensor`（每个层一个），形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
            经过注意力 softmax 后的注意力权重，在自注意力头中用于计算加权平均值。

    """

@add_start_docstrings(
    """
    带有图像分类头的DeiT模型转换器（在[CLS]令牌的最终隐藏状态上有一个线性层，以及在蒸馏令牌的最终隐藏状态上有一个线性层），例如用于ImageNet。

    .. warning::

           此模型仅支持推断。尚不支持使用蒸馏进行微调（即带有教师）。
    """,
    DEIT_START_DOCSTRING,
)
class DeiTForImageClassificationWithTeacher(DeiTPreTrainedModel):
    """
    带有教师的DeiT模型，用于图像分类。
    """
    def __init__(self, config: DeiTConfig) -> None:
        # 调用父类的初始化方法，传入配置对象
        super().__init__(config)

        # 设置分类标签数量
        self.num_labels = config.num_labels
        # 使用给定配置初始化 DeiT 模型，不添加池化层
        self.deit = DeiTModel(config, add_pooling_layer=False)

        # 分类器头部
        # 如果标签数量大于零，则使用线性层作为分类器，否则使用恒等映射
        self.cls_classifier = (
            nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )
        # 同上，为蒸馏分类器设置线性层或恒等映射
        self.distillation_classifier = (
            nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # 初始化权重并应用最终处理
        self.post_init()

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
        # 如果 return_dict 为 None，则使用配置对象中的 use_return_dict 属性
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入像素值和其他参数传递给 DeiT 模型进行前向计算
        outputs = self.deit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取序列输出（通常是最后一层的输出）
        sequence_output = outputs[0]

        # 对序列输出的第一个位置进行分类预测
        cls_logits = self.cls_classifier(sequence_output[:, 0, :])
        # 对序列输出的第二个位置进行蒸馏分类预测
        distillation_logits = self.distillation_classifier(sequence_output[:, 1, :])

        # 在推断时，返回两个分类器预测结果的平均值作为最终 logits
        logits = (cls_logits + distillation_logits) / 2

        # 如果不要求返回字典，则返回一个包含 logits 和所有输出的元组
        if not return_dict:
            output = (logits, cls_logits, distillation_logits) + outputs[1:]
            return output

        # 如果要求返回字典，则构建 DeiTForImageClassificationWithTeacherOutput 对象并返回
        return DeiTForImageClassificationWithTeacherOutput(
            logits=logits,
            cls_logits=cls_logits,
            distillation_logits=distillation_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```