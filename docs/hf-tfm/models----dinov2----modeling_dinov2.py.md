# `.\models\dinov2\modeling_dinov2.py`

```py
# 设置文件编码格式为 utf-8

# 引用标准库中的模块
import collections.abc
import math
from typing import Dict, List, Optional, Set, Tuple, Union

# 引用第三方库中的模块
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 引用 Hugging Face 提供的模块和函数
from ...activations import ACT2FN
from ...modeling_outputs import (
    BackboneOutput,
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ...utils.backbone_utils import BackboneMixin
from .configuration_dinov2 import Dinov2Config

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 通用文档字符串
_CONFIG_FOR_DOC = "Dinov2Config"

# 基本文档字符串
_CHECKPOINT_FOR_DOC = "facebook/dinov2-base"
_EXPECTED_OUTPUT_SHAPE = [1, 257, 768]

# 图像分类文档字符串
_IMAGE_CLASS_CHECKPOINT = "facebook/dinov2-small-imagenet1k-1-layer"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

# 预训练模型列表
DINOV2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/dinov2-base",
    # See all DINOv2 models at https://huggingface.co/models?filter=dinov2
]

# Dinov2Embeddings 类，用于构建 CLS token、mask token、位置和 patch embeddings
class Dinov2Embeddings(nn.Module):
    def __init__(self, config: Dinov2Config) -> None:
        # 调用父类构造函数
        super().__init__()

        # 创建一个可学习的参数 cls_token，形状为 (1, 1, config.hidden_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        # 创建一个可学习的参数 mask_token，形状为 (1, config.hidden_size)
        self.mask_token = nn.Parameter(torch.zeros(1, config.hidden_size))
        # 创建 Dinov2PatchEmbeddings 对象
        self.patch_embeddings = Dinov2PatchEmbeddings(config)
        # 获取 patch 的数量
        num_patches = self.patch_embeddings.num_patches
        # 创建一个可学习的参数 position_embeddings，形状为 (1, num_patches + 1, config.hidden_size)
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, config.hidden_size))
        # 创建一个丢弃层对象
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 设置配置属性
        self.config = config
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        # 这个方法允许插值预训练的位置编码，以便在更高分辨率的图像上使用模型。

        # 计算嵌入中的块数和位置编码中的位置数
        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1
        # 如果块数和位置数相等，并且高度等于宽度，则返回位置编码
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        # 获取位置编码中的类别位置编码和块位置编码
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        # 计算新的高度和宽度，以便对位置编码进行插值
        height = height // self.config.patch_size
        width = width // self.config.patch_size
        # 为了避免插值中的浮点错误，添加一个很小的数值
        # 参见：https://github.com/facebookresearch/dino/issues/8
        height, width = height + 0.1, width + 0.1
        # 更改块位置编码张量的形状和维度顺序
        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        target_dtype = patch_pos_embed.dtype
        # 对块位置编码进行插值
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.to(dtype=torch.float32),
            scale_factor=(float(height / math.sqrt(num_positions)), float(width / math.sqrt(num_positions))),
            mode="bicubic",
            align_corners=False,
        ).to(dtype=target_dtype)
        # 检查插值后的形状是否匹配
        if int(height) != patch_pos_embed.shape[-2] or int(width) != patch_pos_embed.shape[-1]:
            raise ValueError("Width or height does not match with the interpolated position embeddings")
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape
        target_dtype = self.patch_embeddings.projection.weight.dtype
        embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))

        if bool_masked_pos is not None:
            # 根据掩码位置更新嵌入值
            embeddings = torch.where(
                bool_masked_pos.unsqueeze(-1), self.mask_token.to(embeddings.dtype).unsqueeze(0), embeddings
            )

        # 将[CLS]标记添加到嵌入的块标记中
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # 向每个标记添加位置编码
        embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)

        embeddings = self.dropout(embeddings)

        return embeddings
class Dinov2PatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        # 初始化方法，将图片像素值转换为 patch embeddings
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # 前向传播方法，转换像素值为 embeddings
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings


# Copied from transformers.models.vit.modeling_vit.ViTSelfAttention with ViT->Dinov2
class Dinov2SelfAttention(nn.Module):
    def __init__(self, config: Dinov2Config) -> None:
        # 初始化方法，构建 Dinov2SelfAttention 模块
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # 调整形状适应计算 attention scores
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    # 定义一个前向传播函数，接受隐藏状态、头部掩码和是否输出注意力权重作为输入，返回上下文层和注意力权重（如果需要）
    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 用隐藏状态计算查询层
        mixed_query_layer = self.query(hidden_states)

        # 用隐藏状态计算键层，并转置为得分矩阵计算
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        # 用隐藏状态计算值层，并转置为得分矩阵计算
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        # 转置查询层为得分矩阵计算
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 通过“查询”和“键”之间的点积，得到原始注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # 用数学函数计算注意力分数除以平方根头部大小的值
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 将注意力分数归一化为概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 这实际上是删除完整的令牌以进行注意，这可能看起来有点不寻常，但源自原始Transformer论文
        attention_probs = self.dropout(attention_probs)

        # 如果需要，对头部进行掩码
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文层，将注意力概率矩阵与值层相乘
        context_layer = torch.matmul(attention_probs, value_layer)

        # 将上下文层转置，并重塑形状
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # 如果需要输出注意力，返回上下文层和注意力概率，否则只返回上下文层
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
# 从 transformers.models.vit.modeling_vit.ViTSelfOutput 复制过来的类，将 ViT 改为 Dinov2
class Dinov2SelfOutput(nn.Module):
    """
    The residual connection is defined in Dinov2Layer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        # 创建一个全连接层，输入和输出维度都是 hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个 dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将 hidden_states 输入到全连接层
        hidden_states = self.dense(hidden_states)
        # 对全连接层的输出应用 dropout
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# 从 transformers.models.vit.modeling_vit.ViTAttention 复制过来的类，将 ViT 改为 Dinov2
class Dinov2Attention(nn.Module):
    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        # 创建一个 Dinov2SelfAttention 层
        self.attention = Dinov2SelfAttention(config)
        # 创建一个 Dinov2SelfOutput 层
        self.output = Dinov2SelfOutput(config)
        # 初始化一个空的集合用来存储需要裁剪的头部
        self.pruned_heads = set()

    # 对头部进行裁剪
    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        # 找到需要裁剪的头部
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # 裁剪线性层
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储裁剪的头部
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 获取 self attention 层的输出
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        # 将 self attention 层的输出应用到 self output 层
        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # 如果需要输出注意力，则将其添加到输出中
        return outputs


# 层缩放类，根据给定的配置参数初始化一个 lambda1 的可学习参数
class Dinov2LayerScale(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        # 创建一个可训练的参数 lambda1，初始值为 config.layerscale_value * hidden_size
        self.lambda1 = nn.Parameter(config.layerscale_value * torch.ones(config.hidden_size))

    # 前向传播函数
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # 通过 lambda1 对隐藏状态进行缩放操作
        return hidden_state * self.lambda1


# 从 transformers.models.beit.modeling_beit.drop_path 复制过来的 drop_path 函数
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    # 为每个样本丢弃路径 (Stochastic Depth)（当应用于残差块的主路径时）

    # 评论来自 Ross Wightman: 这与我为 EfficientNet 等网络创建的 DropConnect 实现相同，
    # 但是，原始名称是有误导的，因为 'Drop Connect' 是另一篇论文中的一种不同形式的 dropout...
    # 参见讨论: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... 我选择将层和参数名称更改为 'drop path'，
    # 而不是将 DropConnect 作为层名称并使用 'survival rate' 作为参数.
    if drop_prob == 0.0 or not training:
        # 如果丢弃概率为 0 或者不在训练阶段，则直接返回输入
        return input
    # 计算保留概率
    keep_prob = 1 - drop_prob
    # 计算形状，适用于不同维度的张量，而不仅仅是 2D 卷积网络
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)
    # 生成随机张量，其值为保留概率加上一个服从均匀分布的随机数
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    # 将随机张量二值化
    random_tensor.floor_()
    # 根据保留概率对输入进行缩放，并乘以随机张量
    output = input.div(keep_prob) * random_tensor
    # 返回处理后的输出
    return output
# 从transformers.models.beit.modeling_beit.BeitDropPath中复制代码
class Dinov2DropPath(nn.Module):
    """对每个样本应用DropPath（随机深度）（当应用于残差块的主路径时）。"""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class Dinov2MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        in_features = out_features = config.hidden_size
        hidden_features = int(config.hidden_size * config.mlp_ratio)
        # 第一个全连接层，输入维度为config.hidden_size，输出维度为hidden_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        # 根据配置选择激活函数
        if isinstance(config.hidden_act, str):
            self.activation = ACT2FN[config.hidden_act]
        else:
            self.activation = config.hidden_act
        # 第二个全连接层，输入维度为hidden_features，输出维度为config.hidden_size
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # 第一个全连接层
        hidden_state = self.fc1(hidden_state)
        # 激活函数
        hidden_state = self.activation(hidden_state)
        # 第二个全连接层
        hidden_state = self.fc2(hidden_state)
        return hidden_state


class Dinov2SwiGLUFFN(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        in_features = out_features = config.hidden_size
        hidden_features = int(config.hidden_size * config.mlp_ratio)
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8

        # 输入权重层，输入维度为config.hidden_size，输出维度为2 * hidden_features
        self.weights_in = nn.Linear(in_features, 2 * hidden_features, bias=True)
        # 输出权重层，输入维度为hidden_features，输出维度为config.hidden_size
        self.weights_out = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # 输入权重层
        hidden_state = self.weights_in(hidden_state)
        # 将输出切分为两部分
        x1, x2 = hidden_state.chunk(2, dim=-1)
        # 使用SiLU激活函数并进行乘法操作
        hidden = nn.functional.silu(x1) * x2
        # 输出权重层
        return self.weights_out(hidden)


class Dinov2Layer(nn.Module):
    """这对应于原始实现中的Block类。"""

    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()

        # 第一个LayerNorm层
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dinov2Attention层
        self.attention = Dinov2Attention(config)
        # Dinov2LayerScale层
        self.layer_scale1 = Dinov2LayerScale(config)
        # Dinov2DropPath层
        self.drop_path1 = Dinov2DropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()

        # 第二个LayerNorm层
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 根据配置选择MLP或SwiGLUFFN
        if config.use_swiglu_ffn:
            self.mlp = Dinov2SwiGLUFFN(config)
        else:
            self.mlp = Dinov2MLP(config)
        # Dinov2LayerScale层
        self.layer_scale2 = Dinov2LayerScale(config)
        # Dinov2DropPath层
        self.drop_path2 = Dinov2DropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
    # 定义前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 使用 self-attention 层处理隐藏状态（在 Dinov2 中，layernorm 应用在 self-attention 之前）
        self_attention_outputs = self.attention(
            self.norm1(hidden_states),  # 在 Dinov2 中，layernorm 在 self-attention 之前应用
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        # 对注意力输出进行缩放
        attention_output = self.layer_scale1(attention_output)
        outputs = self_attention_outputs[1:]  # 如果输出注意力权重，添加 self attentions

        # 第一个残差连接
        hidden_states = attention_output + hidden_states

        # 在 Dinov2 中，layernorm 也应用在 self-attention 之后
        layer_output = self.norm2(hidden_states)
        layer_output = self.mlp(layer_output)
        layer_output = self.layer_scale2(layer_output)

        # 第二个残差连接
        layer_output = layer_output + hidden_states

        outputs = (layer_output,) + outputs

        return outputs
# 从transformers.models.vit.modeling_vit.ViTEncoder 复制代码，并将 ViT 改为 Dinov2
class Dinov2Encoder(nn.Module):
    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        # 初始化配置参数
        self.config = config
        # 使用 DINoV2 层组成的模块列表
        self.layer = nn.ModuleList([Dinov2Layer(config) for _ in range(config.num_hidden_layers)])
        # 渐变检查点默认为关闭
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        # 如果输出隐藏状态为真，则初始化 all_hidden_states 为空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果输出注意力权重矩阵为真，则初始化 all_self_attentions 为空元组
        all_self_attentions = () if output_attentions else None

        # 遍历 DINoV2 层列表
        for i, layer_module in enumerate(self.layer):
            # 如果输出隐藏状态为真，则将当前隐状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            # 获取当前层的头部遮罩
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 如果启用渐变检查点并且处于训练状态，则使用渐变检查点函数
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                # 否则，直接调用当前层模块计算输出
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            # 更新当前隐状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]

            # 如果输出注意力权重矩阵为真，则将当前层的注意力权重矩阵添加到 all_self_attentions 中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果输出隐藏状态为真，则将最终隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典，则返回相应的元组或 BaseModelOutput
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

# Dinov2PreTrainedModel 类用于处理权重初始化、预训练模型的下载和加载
class Dinov2PreTrainedModel(PreTrainedModel):
    config_class = Dinov2Config
    # DINoV2 模型前缀
    base_model_prefix = "dinov2"
    # 主输入名称为 "pixel_values"
    main_input_name = "pixel_values"
    # 支持渐变检查点的开关为真
    supports_gradient_checkpointing = True
    # 初始化模型权重的方法
    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        # 如果是线性层或卷积层
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 将权重数据升级为`fp32`，然后再转回期望的`dtype`，以避免`half`模式下的`trunc_normal_cpu`未实现的问题
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            # 如果有偏置，将偏置数据置零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是 LayerNorm 层
        elif isinstance(module, nn.LayerNorm):
            # 将偏置数据置零
            module.bias.data.zero_()
            # 将权重数据置为 1.0
            module.weight.data.fill_(1.0)
        # 如果是 Dinov2Embeddings 类型
        elif isinstance(module, Dinov2Embeddings):
            # 初始化位置嵌入的数据
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)
            # 初始化 cls_token 的数据
            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.cls_token.dtype)
# DINOV2_INPUTS_DOCSTRING 用于描述 DINOv2 模型的输入参数及其含义
DINOV2_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            # 像素数值。可以使用 [`AutoImageProcessor`] 获取像素值。参见 [`BitImageProcessor.preprocess`] 获取详细信息。

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于使自注意模块中选择的头部失效的掩码。掩码值在 `[0, 1]` 范围内：

            - 1 表示头部**不被掩盖**，
            - 0 表示头部**被掩盖**。

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。查看返回的张量下的 `attentions` 以获取更多详细信息。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。查看返回的张量下的 `hidden_states` 以获取更多详细信息。
        return_dict (`bool`, *optional*):
            # 是否返回一个 [`~utils.ModelOutput`] 而不是一个普通元组。
# 定义了一个名为Dinov2Model的类，继承自Dinov2PreTrainedModel
@add_start_docstrings(
    "The bare DINOv2 Model transformer outputting raw hidden-states without any specific head on top.", # 添加模型介绍文档
    DINOV2_START_DOCSTRING, # 使用预定义的模型文档字符串
)
class Dinov2Model(Dinov2PreTrainedModel):
    # 初始化函数，接受一个Dinov2Config类型的参数
    def __init__(self, config: Dinov2Config):
        super().__init__(config) # 调用父类的初始化函数
        self.config = config # 将传入的config参数赋值给self.config
        # 初始化嵌入层和编码器
        self.embeddings = Dinov2Embeddings(config) # 创建嵌入层对象
        self.encoder = Dinov2Encoder(config) # 创建编码器对象
        # 应用LayerNorm
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) # 初始化LayerNorm对象
        # 初始化权重并应用最终处理
        self.post_init() # 调用自定义的post_init方法进行初始化

    # 获取输入嵌入
    def get_input_embeddings(self) -> Dinov2PatchEmbeddings:
        return self.embeddings.patch_embeddings # 返回嵌入层的patch_embeddings

    # 修剪模型头部
    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads) # 针对给定层和头部进行修剪操作

    # 前向传播函数
    @add_start_docstrings_to_model_forward(DINOV2_BASE_INPUTS_DOCSTRING) # 添加模型前向传播文档
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC, # 添加代码示例文档
        output_type=BaseModelOutputWithPooling, # 输出类型文档
        config_class=_CONFIG_FOR_DOC, # 配置文档
        modality="vision", # 模态文档
        expected_output=_EXPECTED_OUTPUT_SHAPE, # 期望输出形状文档
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None, # 可选的像素值输入
        bool_masked_pos: Optional[torch.Tensor] = None, # 可选的bool掩码位置输入
        head_mask: Optional[torch.Tensor] = None, # 可选的头部掩码输入
        output_attentions: Optional[bool] = None, # 可选的输出注意力权重
        output_hidden_states: Optional[bool] = None, # 可选的输出隐藏态
        return_dict: Optional[bool] = None, # 可选的返回字典
        ) -> Union[Tuple, BaseModelOutputWithPooling]:
            # 如果未指定output_attentions，则使用配置中的output_attentions
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            # 如果未指定output_hidden_states，则使用配置中的output_hidden_states
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            # 如果未指定return_dict，则使用配置中的use_return_dict
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            # 如果pixel_values为空，则抛出数值错误
            if pixel_values is None:
                raise ValueError("You have to specify pixel_values")

            # 准备头部蒙版（如果需要）
            # head_mask中的1.0表示保留该头部
            # attention_probs的形状为bsz x n_heads x N x N
            # 输入的head_mask形状为[num_heads]或[num_hidden_layers x num_heads]
            # 并且head_mask被转换为形状[num_hidden_layers x batch x num_heads x seq_length x seq_length]
            head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

            # 使用embeddings模块将像素值转换为嵌入输出
            embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)

            # 使用encoder模块处理嵌入输出
            encoder_outputs = self.encoder(
                embedding_output,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            # 获取编码器输出的序列输出
            sequence_output = encoder_outputs[0]
            # 对序列输出进行layernorm处理
            sequence_output = self.layernorm(sequence_output)
            # 获取池化输出
            pooled_output = sequence_output[:, 0, :]

            # 如果不需要返回字典类型的输出
            if not return_dict:
                # 获取头部输出及编码器输出的其他部分
                head_outputs = (sequence_output, pooled_output)
                return head_outputs + encoder_outputs[1:]

            # 返回包含池化输出、最后隐藏状态、隐藏状态和注意力的字典类型输出
            return BaseModelOutputWithPooling(
                last_hidden_state=sequence_output,
                pooler_output=pooled_output,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )
# 添加类的文档字符串，描述了模型的结构和用途
@add_start_docstrings(
    """
    Dinov2 Model transformer with an image classification head on top (a linear layer on top of the final hidden state
    of the [CLS] token) e.g. for ImageNet.
    """,
    DINOV2_START_DOCSTRING,
)
class Dinov2ForImageClassification(Dinov2PreTrainedModel):
    # 初始化方法，接受一个Dinov2Config类型的参数
    def __init__(self, config: Dinov2Config) -> None:
        # 调用父类的初始化方法
        super().__init__(config)

        # 保存标签数目
        self.num_labels = config.num_labels
        # 创建Dinov2Model对象
        self.dinov2 = Dinov2Model(config)

        # 分类器层
        self.classifier = (
            # 根据config.num_labels设置线性层或恒等映射
            nn.Linear(config.hidden_size * 2, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法，接受多个torch.Tensor类型的参数
    @add_start_docstrings_to_model_forward(DINOV2_INPUTS_DOCSTRING)
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
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 确定是否返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 DINOv2 模型进行前向传播
        outputs = self.dinov2(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取序列输出
        sequence_output = outputs[0]  # batch_size, sequence_length, hidden_size

        # 获取分类标记和补丁标记
        cls_token = sequence_output[:, 0]
        patch_tokens = sequence_output[:, 1:]

        # 构建线性层的输入
        linear_input = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)

        # 通过分类器获取 logits
        logits = self.classifier(linear_input)

        loss = None
        if labels is not None:
            # 将标签移动到正确的设备以启用模型并行处理
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

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

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回图像分类器输出
        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用Dinov2预训练模型和BackboneMixin创建Dinov2Backbone类，用于与DETR和MaskFormer等框架一起使用
@add_start_docstrings(
    """
    Dinov2 backbone, to be used with frameworks like DETR and MaskFormer.
    """,
    DINOV2_START_DOCSTRING,
)
class Dinov2Backbone(Dinov2PreTrainedModel, BackboneMixin):
    # 初始化函数，接受配置参数config
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)
        # 调用父类的_init_backbone方法
        super()._init_backbone(config)

        # 设置特征数量为隐藏层大小，长度为隐藏层数加1
        self.num_features = [config.hidden_size for _ in range(config.num_hidden_layers + 1)]
        # 创建Dinov2Embeddings对象
        self.embeddings = Dinov2Embeddings(config)
        # 创建Dinov2Encoder对象
        self.encoder = Dinov2Encoder(config)

        # 创建LayerNorm对象，用于归一化隐藏层输出
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入的方法，返回Dinov2PatchEmbeddings对象
    def get_input_embeddings(self) -> Dinov2PatchEmbeddings:
        return self.embeddings.patch_embeddings

    # 前向传播方法，接受像素值张量和一些可选参数
    @add_start_docstrings_to_model_forward(DINOV2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BackboneOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BackboneOutput:
        """
        返回BackboneOutput对象:

        示例:

        ```python
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        >>> model = AutoBackbone.from_pretrained(
        ...     "facebook/dinov2-base", out_features=["stage2", "stage5", "stage8", "stage11"]
        ... )

        >>> inputs = processor(image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> feature_maps = outputs.feature_maps
        >>> list(feature_maps[-1].shape)
        [1, 768, 16, 16]
        ```py"""
        如果return_dict不为None，则使用return_dict，否则使用self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        使用输入的像素值计算嵌入输出
        embedding_output = self.embeddings(pixel_values)

        使用编码器处理嵌入输出，输出隐藏状态和注意力权重
        outputs = self.encoder(
            embedding_output, output_hidden_states=True, output_attentions=output_attentions, return_dict=return_dict
        )

        如果return_dict为True，则将隐藏状态设置为outputs.hidden_states，否则设置为outputs[1]
        hidden_states = outputs.hidden_states if return_dict else outputs[1]

        初始化特征图为空元组
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            遍历阶段名称和隐藏状态
            如果阶段在输出特征中
            if stage in self.out_features:
                如果配置中应用了layernorm，则对隐藏状态进行layernorm处理
                if self.config.apply_layernorm:
                    hidden_state = self.layernorm(hidden_state)
                如果配置中需要重塑隐藏状态
                if self.config.reshape_hidden_states:
                    修正隐藏状态的形状
                    hidden_state = hidden_state[:, 1:]
                    # 这实际上是原始实现中的一个bug，我们在这里复制了它，因为通常顺序是高度，宽度
                    batch_size, _, height, width = pixel_values.shape
                    patch_size = self.config.patch_size
                    hidden_state = hidden_state.reshape(batch_size, height // patch_size, width // patch_size, -1)
                    hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()
                将处理后的隐藏状态添加到特征图中
                feature_maps += (hidden_state,)

        如果return_dict为False
        if not return_dict:
            如果输出隐藏状态为True，则输出为特征图和outputs[1:]
            if output_hidden_states:
                output = (feature_maps,) + outputs[1:]
            否则输出为特征图和outputs[2:]
            else:
                output = (feature_maps,) + outputs[2:]
            返回输出
            return output

        返回BackboneOutput对象，包含特征图、隐藏状态和注意力权重
        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
        )
```