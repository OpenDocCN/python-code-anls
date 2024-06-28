# `.\models\dinov2\modeling_dinov2.py`

```py
# 设置文件编码为 UTF-8
# 版权声明及保留所有权利声明
#
# 根据 Apache 许可证 2.0 版本进行许可
# 除非符合许可证中的要求，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件按"原样"分发，
# 不提供任何明示或暗示的保证或条件
# 请参阅许可证以了解具体的法律条款和条件
""" PyTorch DINOv2 模型."""

# 导入必要的库和模块
import collections.abc
import math
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入模型输出类
from ...activations import ACT2FN
from ...modeling_outputs import (
    BackboneOutput,
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
)
# 导入模型基类和相关工具函数
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 导入背景模块相关的工具函数
from ...utils.backbone_utils import BackboneMixin
# 导入 DINOv2 配置类
from .configuration_dinov2 import Dinov2Config

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 用于文档字符串的常规配置名称
_CONFIG_FOR_DOC = "Dinov2Config"

# 用于文档字符串的基础检查点名称
_CHECKPOINT_FOR_DOC = "facebook/dinov2-base"
# 预期的输出形状
_EXPECTED_OUTPUT_SHAPE = [1, 257, 768]

# 图像分类模型文档字符串中的检查点名称
_IMAGE_CLASS_CHECKPOINT = "facebook/dinov2-small-imagenet1k-1-layer"
# 预期的图像分类输出
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

# DINOv2 预训练模型的存档列表
DINOV2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/dinov2-base",
    # 查看所有 DINOv2 模型：https://huggingface.co/models?filter=dinov2
]


class Dinov2Embeddings(nn.Module):
    """
    构建 CLS 令牌、掩码令牌、位置和补丁嵌入的模块。
    """

    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()

        # 定义 CLS 令牌作为可学习的参数
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        # 定义掩码令牌作为可学习的参数，初始为全零向量
        self.mask_token = nn.Parameter(torch.zeros(1, config.hidden_size))
        # 使用 Dinov2PatchEmbeddings 类构建补丁嵌入
        self.patch_embeddings = Dinov2PatchEmbeddings(config)
        # 获取补丁数量并为每个补丁和位置添加嵌入向量
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, config.hidden_size))
        # 使用配置中的隐藏层丢弃率定义丢弃层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 保存配置对象
        self.config = config
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method interpolates the pre-trained position encodings for higher resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        # Determine the number of patches and positions minus one
        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1

        # If the number of patches matches the number of positions and height equals width, return the original position embeddings
        if num_patches == num_positions and height == width:
            return self.position_embeddings

        # Select the [CLS] token positional embedding and the patch positional embeddings
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]

        # Get the dimension of the embeddings
        dim = embeddings.shape[-1]

        # Calculate the effective height and width based on the configuration's patch size
        height = height // self.config.patch_size
        width = width // self.config.patch_size

        # Add a small number to avoid floating-point errors during interpolation
        height, width = height + 0.1, width + 0.1

        # Reshape patch_pos_embed to match the spatial dimensions of the patches
        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        # Determine the target data type for patch_pos_embed
        target_dtype = patch_pos_embed.dtype

        # Perform bicubic interpolation on patch_pos_embed to match the new height and width
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.to(dtype=torch.float32),
            scale_factor=(float(height / math.sqrt(num_positions)), float(width / math.sqrt(num_positions))),
            mode="bicubic",
            align_corners=False,
        ).to(dtype=target_dtype)

        # Verify that the interpolated dimensions match the expected height and width
        if int(height) != patch_pos_embed.shape[-2] or int(width) != patch_pos_embed.shape[-1]:
            raise ValueError("Width or height does not match with the interpolated position embeddings")

        # Reshape patch_pos_embed back to the original token dimensions and concatenate with class_pos_embed
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape
        target_dtype = self.patch_embeddings.projection.weight.dtype

        # Convert pixel_values to the target_dtype and pass through patch_embeddings
        embeddings = self.patch_embeddings(pixel_values.to(dtype=target_dtype))

        # Conditionally replace embeddings with mask_token where bool_masked_pos is True
        if bool_masked_pos is not None:
            embeddings = torch.where(
                bool_masked_pos.unsqueeze(-1), self.mask_token.to(embeddings.dtype).unsqueeze(0), embeddings
            )

        # Expand the [CLS] token across the batch and concatenate with embeddings
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # Add interpolated positional encoding to each token embedding
        embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)

        # Apply dropout to the embeddings
        embeddings = self.dropout(embeddings)

        return embeddings
# 定义一个用于将像素值转换为初始隐藏状态（补丁嵌入）的模块，以便Transformer处理。
class Dinov2PatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        # 从配置中提取图像大小和补丁大小
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        # 将图像大小和补丁大小转为可迭代对象（tuple），如果不是的话
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        
        # 计算图像中的补丁数量
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        
        # 设置模块的属性
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        # 使用卷积层进行投影，将通道数转换为隐藏大小
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # 检查输入像素值的通道数是否与配置中指定的通道数相匹配
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        
        # 使用投影层进行补丁嵌入，然后展平和转置以生成最终的嵌入表示
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings


# 从transformers.models.vit.modeling_vit.ViTSelfAttention中复制并修改为Dinov2
class Dinov2SelfAttention(nn.Module):
    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        # 检查隐藏大小是否是注意力头数的倍数，如果没有提供embedding_size属性的话
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        # 设置注意力头数和每个注意力头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 定义查询、键、值的线性变换层，并支持是否使用偏置
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        # 定义dropout层，用于注意力概率的dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # 重新排列张量形状，以便进行多头注意力的计算
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    # 定义一个方法用于前向传播，接受隐藏状态、头部掩码（可选的张量）、是否输出注意力矩阵作为参数，并返回一个元组
    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 生成混合查询层，使用 self.query 对隐藏状态进行处理
        mixed_query_layer = self.query(hidden_states)

        # 生成键层，先使用 self.key 对隐藏状态进行处理，再根据注意力头大小重新排列维度
        key_layer = self.transpose_for_scores(self.key(hidden_states))

        # 生成值层，先使用 self.value 对隐藏状态进行处理，同样根据注意力头大小重新排列维度
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # 对混合查询层也进行维度重排
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 计算注意力分数，使用 torch.matmul 计算 "查询" 和 "键" 的点积
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # 将注意力分数除以 sqrt(注意力头大小)，进行归一化处理
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 对注意力分数进行 softmax 操作，将其转换为注意力概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 对注意力概率进行 dropout 操作，以减少过拟合风险
        attention_probs = self.dropout(attention_probs)

        # 如果存在头部掩码，则将注意力概率与掩码相乘
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文层，使用注意力概率和值层的乘积
        context_layer = torch.matmul(attention_probs, value_layer)

        # 对上下文层进行维度重排，将注意力头的结果合并到一起
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # 根据是否需要输出注意力矩阵，构造输出元组
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        # 返回最终的输出结果
        return outputs
# 从transformers.models.vit.modeling_vit.ViTSelfOutput复制到Dinov2SelfOutput，并将ViT改为Dinov2
class Dinov2SelfOutput(nn.Module):
    """
    The residual connection is defined in Dinov2Layer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        # 使用配置中的隐藏层大小定义线性层
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 使用配置中的隐藏层dropout概率定义dropout层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将输入的隐藏状态传递给线性层
        hidden_states = self.dense(hidden_states)
        # 对线性层的输出应用dropout
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# 从transformers.models.vit.modeling_vit.ViTAttention复制到Dinov2Attention，并将ViT改为Dinov2
class Dinov2Attention(nn.Module):
    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        # 初始化自注意力层和输出层
        self.attention = Dinov2SelfAttention(config)
        self.output = Dinov2SelfOutput(config)
        # 初始化一个空集合，用于存储需要剪枝的注意力头
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        # 调用辅助函数找到可剪枝的注意力头索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储剪枝后的注意力头
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 将隐藏状态传递给自注意力层进行处理
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        # 将自注意力层的输出传递给输出层，与原始输入结合形成注意力输出
        attention_output = self.output(self_outputs[0], hidden_states)

        # 如果需要输出注意力权重，则将它们添加到输出中
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# 从transformers.models.beit.modeling_beit.drop_path复制
def drop_path(input: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    实现DropPath操作，用于随机关闭网络中的路径，以增强模型的泛化能力。
    """
    # 如果 drop_prob 为 0 或者不处于训练模式，则直接返回输入
    if drop_prob == 0.0 or not training:
        return input
    
    # 计算保留节点的概率
    keep_prob = 1 - drop_prob
    
    # 确定随机张量的形状，适用于不同维度的张量，而不仅仅是二维卷积网络
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)
    
    # 生成与输入张量相同形状的随机张量，其值在 [keep_prob, 1.0) 范围内
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    
    # 将随机张量向下取整，实现二值化操作
    random_tensor.floor_()
    
    # 对输入张量进行调整，以实现随机丢弃路径的效果
    output = input.div(keep_prob) * random_tensor
    
    # 返回调整后的输出张量
    return output
# Copied from transformers.models.beit.modeling_beit.BeitDropPath
# 定义 Dinov2DropPath 类，用于实现每个样本的随机深度路径丢弃（在残差块的主路径上应用）。
class Dinov2DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob  # 初始化丢弃概率

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 调用 drop_path 函数来执行随机深度路径丢弃操作
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        # 返回描述实例额外信息的字符串，这里是丢弃概率
        return "p={}".format(self.drop_prob)


class Dinov2MLP(nn.Module):
    # 定义 Dinov2MLP 类，用于实现多层感知机（MLP）部分的前向传播。
    def __init__(self, config) -> None:
        super().__init__()
        in_features = out_features = config.hidden_size
        hidden_features = int(config.hidden_size * config.mlp_ratio)
        # 第一个全连接层，输入特征数为隐藏大小，输出特征数为隐藏大小乘以 MLP 比率
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        # 激活函数根据配置选择或者直接使用配置中的激活函数
        if isinstance(config.hidden_act, str):
            self.activation = ACT2FN[config.hidden_act]
        else:
            self.activation = config.hidden_act
        # 第二个全连接层，输入特征数为隐藏大小乘以 MLP 比率，输出特征数为隐藏大小
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # MLP 的前向传播过程，依次经过第一层全连接、激活函数、第二层全连接
        hidden_state = self.fc1(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = self.fc2(hidden_state)
        return hidden_state


class Dinov2SwiGLUFFN(nn.Module):
    # 定义 Dinov2SwiGLUFFN 类，用于实现基于 SwiGLU 的前馈神经网络部分的前向传播。
    def __init__(self, config) -> None:
        super().__init__()
        in_features = out_features = config.hidden_size
        hidden_features = int(config.hidden_size * config.mlp_ratio)
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8

        # 输入权重层，输入特征数为隐藏大小，输出特征数为隐藏大小乘以 2
        self.weights_in = nn.Linear(in_features, 2 * hidden_features, bias=True)
        # 输出权重层，输入特征数为隐藏大小乘以 2/3，输出特征数为隐藏大小
        self.weights_out = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # SwiGLU 前馈神经网络的前向传播过程，经过输入权重层、切片操作、激活函数、输出权重层
        hidden_state = self.weights_in(hidden_state)
        x1, x2 = hidden_state.chunk(2, dim=-1)
        hidden = nn.functional.silu(x1) * x2
        return self.weights_out(hidden)


class Dinov2Layer(nn.Module):
    """This corresponds to the Block class in the original implementation."""
    # 定义 Dinov2Layer 类，对应原始实现中的块（Block）类。

    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()

        # 层归一化层，用于规范隐藏状态，epsilon 为层归一化的小数
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dinov2Attention 注意力机制
        self.attention = Dinov2Attention(config)
        # 层缩放，用于缩放层的输出
        self.layer_scale1 = Dinov2LayerScale(config)
        # Dinov2DropPath 类，用于随机深度路径丢弃
        self.drop_path = Dinov2DropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()

        # 层归一化层，用于规范隐藏状态，epsilon 为层归一化的小数
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 根据配置选择使用 SwiGLU 前馈神经网络或者 MLP
        if config.use_swiglu_ffn:
            self.mlp = Dinov2SwiGLUFFN(config)
        else:
            self.mlp = Dinov2MLP(config)
        # 层缩放，用于缩放层的输出
        self.layer_scale2 = Dinov2LayerScale(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    # 定义函数返回类型，可以返回两种可能的元组类型，每种元组包含两个 torch.Tensor 对象
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 对隐藏状态进行第一次 LayerNorm 处理，并传递给自注意力模块
        self_attention_outputs = self.attention(
            self.norm1(hidden_states),  # Dinov2 中在自注意力之前应用 LayerNorm
            head_mask,
            output_attentions=output_attentions,
        )
        # 获取自注意力模块的输出
        attention_output = self_attention_outputs[0]

        # 对自注意力输出应用第一个缩放层
        attention_output = self.layer_scale1(attention_output)
        outputs = self_attention_outputs[1:]  # 如果需要输出注意力权重，则将其添加到 outputs 中

        # 第一个残差连接
        hidden_states = self.drop_path(attention_output) + hidden_states

        # Dinov2 中在自注意力之后再次应用 LayerNorm
        layer_output = self.norm2(hidden_states)
        # 经过 MLP (多层感知机) 的前馈网络处理
        layer_output = self.mlp(layer_output)
        # 对 MLP 输出应用第二个缩放层
        layer_output = self.layer_scale2(layer_output)

        # 第二个残差连接
        layer_output = self.drop_path(layer_output) + hidden_states

        # 将最终的层输出与可能的注意力权重输出打包成元组
        outputs = (layer_output,) + outputs

        # 返回处理后的输出
        return outputs
# 从 transformers.models.vit.modeling_vit.ViTEncoder 复制而来，将 ViT 修改为 Dinov2
class Dinov2Encoder(nn.Module):
    # Dinov2Encoder 类的构造函数，接受一个 Dinov2Config 类型的参数 config
    def __init__(self, config: Dinov2Config) -> None:
        super().__init__()
        # 保存传入的配置参数
        self.config = config
        # 使用 Dinov2Layer 类构建的 nn.ModuleList，构建包含 config.num_hidden_layers 个层的层列表
        self.layer = nn.ModuleList([Dinov2Layer(config) for _ in range(config.num_hidden_layers)])
        # 是否启用梯度检查点，默认为 False
        self.gradient_checkpointing = False

    # Dinov2Encoder 的前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码张量，可选
        output_attentions: bool = False,  # 是否输出注意力权重，默认为 False
        output_hidden_states: bool = False,  # 是否输出所有隐藏状态，默认为 False
        return_dict: bool = True,  # 是否返回字典格式的输出，默认为 True
    ) -> Union[tuple, BaseModelOutput]:  # 返回值可以是元组或 BaseModelOutput 类型

        # 如果要输出隐藏状态，则初始化一个空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果要输出注意力权重，则初始化一个空元组
        all_self_attentions = () if output_attentions else None

        # 遍历每一层的 module
        for i, layer_module in enumerate(self.layer):
            # 如果要输出隐藏状态，则将当前的隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 如果启用梯度检查点且在训练阶段，使用梯度检查点函数进行计算
            if self.gradient_checkpointing and self.training:
                # 调用梯度检查点函数进行前向传播计算
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                # 否则，直接调用当前层的 forward 方法进行前向传播计算
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            # 更新隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]

            # 如果要输出注意力权重，则将当前层的注意力权重输出添加到 all_self_attentions 中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果要输出隐藏状态，则将最终的隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要以字典格式返回结果，则返回不为 None 的值的元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 否则，返回一个 BaseModelOutput 类型的对象，包含最终的隐藏状态、所有隐藏状态和所有注意力权重
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


# Dinov2PreTrainedModel 类，继承自 PreTrainedModel 类
class Dinov2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # Dinov2PreTrainedModel 类的配置类为 Dinov2Config
    config_class = Dinov2Config
    # 基础模型前缀名称为 "dinov2"
    base_model_prefix = "dinov2"
    # 主要输入名称为 "pixel_values"
    main_input_name = "pixel_values"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 初始化模型的权重
    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        # 如果是线性层或者卷积层
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 将权重数据先转换为 float32 类型，避免在 half 类型下使用 `trunc_normal_cpu` 时出现问题，
            # 然后再转换回原始的 dtype
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            # 如果存在偏置项，将其数据初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是 LayerNorm 层
        elif isinstance(module, nn.LayerNorm):
            # 将偏置项数据初始化为零
            module.bias.data.zero_()
            # 将权重数据初始化为全 1
            module.weight.data.fill_(1.0)
        # 如果是 Dinov2Embeddings 类型的模块
        elif isinstance(module, Dinov2Embeddings):
            # 初始化位置嵌入的数据
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)
            # 初始化类别令牌的数据
            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.cls_token.dtype)
# DINOV2_INPUTS_DOCSTRING 是一个字符串变量，用于存储模型输入参数的文档字符串模板。
DINOV2_INPUTS_DOCSTRING = r"""
# 以下是模型的输入参数说明：

Args:
    pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
        像素值。像素值可以使用 [`AutoImageProcessor`] 获取。详见 [`BitImageProcessor.preprocess`]。

    bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, sequence_length)`):
        布尔掩码位置。指示哪些补丁是掩码的（1），哪些不是（0）。仅在预训练中相关。

    head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
        用于掩盖自注意力模块中选定头部的掩码。掩码值在 `[0, 1]` 中选择：

        - 1 表示头部**未被掩盖**，
        - 0 表示头部**被掩盖**。

    output_attentions (`bool`, *optional*):
        是否返回所有注意力层的注意力张量。返回的张量中的 `attentions` 有更多细节。

    output_hidden_states (`bool`, *optional*):
        是否返回所有层的隐藏状态。返回的张量中的 `hidden_states` 有更多细节。

    return_dict (`bool`, *optional*):
        是否返回一个 [`~utils.ModelOutput`] 而不是一个普通元组。
"""
    # 定义函数签名和参数说明
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            像素值。像素值可以通过 [`AutoImageProcessor`] 获取。详见 [`BitImageProcessor.preprocess`]。

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            自注意力模块中选定头部的掩码。掩码值在 `[0, 1]` 范围内：

            - 1 表示头部 **未被掩码**，
            - 0 表示头部 **已被掩码**。

        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。详见返回的张量中的 `attentions`。

        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。详见返回的张量中的 `hidden_states`。

        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而非普通元组。
"""
@add_start_docstrings(
    "The bare DINOv2 Model transformer outputting raw hidden-states without any specific head on top.",
    DINOV2_START_DOCSTRING,
)
"""
class Dinov2Model(Dinov2PreTrainedModel):
    """
    DINOv2 模型类，继承自预训练模型基类 Dinov2PreTrainedModel。
    """
    def __init__(self, config: Dinov2Config):
        """
        初始化方法，设置模型配置信息。

        Args:
            config (Dinov2Config): 模型的配置对象。
        """
        super().__init__(config)
        self.config = config

        # 初始化模型的嵌入层和编码器
        self.embeddings = Dinov2Embeddings(config)
        self.encoder = Dinov2Encoder(config)

        # 初始化层归一化层
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 执行初始化权重和最终处理
        self.post_init()

    def get_input_embeddings(self) -> Dinov2PatchEmbeddings:
        """
        返回输入嵌入层对象。

        Returns:
            Dinov2PatchEmbeddings: 输入嵌入层对象。
        """
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        对模型的注意力头进行修剪。

        Args:
            heads_to_prune (Dict[int, List[int]]): 要在每层修剪的注意力头的字典。

        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(DINOV2_BASE_INPUTS_DOCSTRING)
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
        bool_masked_pos: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[Tuple, BaseModelOutputWithPooling]:
        # 确定是否输出注意力权重
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 确定是否输出隐藏层状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 确定是否返回字典形式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            # 如果像素值为None，则抛出数值错误异常
            raise ValueError("You have to specify pixel_values")

        # 准备头部掩码（如果需要）
        # head_mask中的1.0表示保留该头部
        # attention_probs的形状为 bsz x n_heads x N x N
        # 输入的head_mask的形状为[num_heads]或[num_hidden_layers x num_heads]
        # head_mask被转换为形状[num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # 将像素值嵌入到模型中，如果有bool_masked_pos，则指定其位置
        embedding_output = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)

        # 编码器处理嵌入的输出
        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取编码器的序列输出
        sequence_output = encoder_outputs[0]
        # 对序列输出进行层归一化
        sequence_output = self.layernorm(sequence_output)
        # 提取汇总输出，通常是序列输出的第一个位置
        pooled_output = sequence_output[:, 0, :]

        if not return_dict:
            # 如果不需要返回字典形式的输出，则返回头部输出和编码器的其他输出状态
            head_outputs = (sequence_output, pooled_output)
            return head_outputs + encoder_outputs[1:]

        # 如果需要返回字典形式的输出，则返回BaseModelOutputWithPooling对象
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 使用装饰器为类添加文档字符串，描述该类是基于Dinov2模型的转换器，带有顶部的图像分类头部
# 顶部的图像分类头部指的是在[CLS]标记的最终隐藏状态之上的线性层，例如用于ImageNet分类
@add_start_docstrings(
    """
    Dinov2 Model transformer with an image classification head on top (a linear layer on top of the final hidden state
    of the [CLS] token) e.g. for ImageNet.
    """,
    DINOV2_START_DOCSTRING,
)
class Dinov2ForImageClassification(Dinov2PreTrainedModel):
    def __init__(self, config: Dinov2Config) -> None:
        super().__init__(config)

        # 初始化函数，调用父类构造函数初始化配置
        self.num_labels = config.num_labels
        # 创建Dinov2模型实例
        self.dinov2 = Dinov2Model(config)

        # 分类器头部
        # 如果配置中的标签数大于0，则创建一个线性层作为分类器，输入大小为两倍的隐藏状态大小，输出大小为标签数
        # 否则创建一个身份映射层
        self.classifier = (
            nn.Linear(config.hidden_size * 2, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # 使用装饰器为前向方法添加文档字符串，描述输入和输出的格式，参考检查点和期望输出类型
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
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 如果 return_dict 不为 None，则使用给定的 return_dict；否则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用模型的 forward 方法，传入像素值 pixel_values 和其他参数，获取模型输出
        outputs = self.dinov2(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 提取模型输出中的序列输出（通常是经过特征提取后的表示）
        sequence_output = outputs[0]  # batch_size, sequence_length, hidden_size

        # 提取序列输出中的 CLS token，通常用于整体序列的表示
        cls_token = sequence_output[:, 0]

        # 提取除了 CLS token 以外的所有 patch token 的表示
        patch_tokens = sequence_output[:, 1:]

        # 构造线性层的输入，将 CLS token 和 patch token 的均值拼接在一起
        linear_input = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)

        # 将拼接后的表示输入分类器，得到预测的 logits
        logits = self.classifier(linear_input)

        # 初始化损失为 None
        loss = None
        if labels is not None:
            # 将 labels 移动到正确的设备上以支持模型并行计算
            labels = labels.to(logits.device)

            # 如果问题类型为 None，则根据标签和标签数量确定问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型计算相应的损失
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    # 如果只有一个标签，则计算回归损失
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    # 否则计算多标签的回归损失
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # 如果是单标签分类问题，则使用交叉熵损失函数
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # 如果是多标签分类问题，则使用带 logits 的二元交叉熵损失函数
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果不需要返回字典形式的输出，则按照元组形式返回输出和损失（如果存在损失）
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 否则以 ImageClassifierOutput 的形式返回输出
        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用自定义文档字符串描述 Dinov2 的骨干模型，适用于 DETR 和 MaskFormer 等框架
@add_start_docstrings(
    """
    Dinov2 backbone, to be used with frameworks like DETR and MaskFormer.
    """,
    DINOV2_START_DOCSTRING,
)
class Dinov2Backbone(Dinov2PreTrainedModel, BackboneMixin):
    def __init__(self, config):
        # 调用父类的初始化方法，传入配置
        super().__init__(config)
        # 初始化骨干模型
        super()._init_backbone(config)

        # 计算特征数量列表，每层的特征数量都是隐藏大小 hidden_size
        self.num_features = [config.hidden_size for _ in range(config.num_hidden_layers + 1)]
        
        # 初始化嵌入层
        self.embeddings = Dinov2Embeddings(config)
        # 初始化编码器
        self.encoder = Dinov2Encoder(config)

        # 初始化 LayerNorm 层，用于归一化隐藏状态
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 初始化权重并进行最终处理
        self.post_init()

    def get_input_embeddings(self) -> Dinov2PatchEmbeddings:
        # 返回嵌入层中的 patch_embeddings
        return self.embeddings.patch_embeddings

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
        定义方法签名和返回类型注解，方法返回类型为 BackboneOutput。

        返回方法的输出结果，通常用于说明方法的功能。

        Examples: 示例用法，展示如何使用此方法的代码片段。

        ```
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
        ```"""
        # 初始化 return_dict 变量，如果外部未提供则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 初始化 output_hidden_states 变量，如果外部未提供则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 初始化 output_attentions 变量，如果外部未提供则使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        # 将输入像素值传递给嵌入层处理，生成嵌入输出
        embedding_output = self.embeddings(pixel_values)

        # 将嵌入输出传递给编码器，获取编码器的输出
        outputs = self.encoder(
            embedding_output, output_hidden_states=True, output_attentions=output_attentions, return_dict=return_dict
        )

        # 根据是否使用 return_dict，选择合适的隐藏状态
        hidden_states = outputs.hidden_states if return_dict else outputs[1]

        # 初始化特征图为空元组
        feature_maps = ()
        # 遍历阶段名称和隐藏状态，为每个阶段生成特征图
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                # 如果配置要求，对隐藏状态应用层归一化
                if self.config.apply_layernorm:
                    hidden_state = self.layernorm(hidden_state)
                # 如果配置要求，重塑隐藏状态的形状
                if self.config.reshape_hidden_states:
                    hidden_state = hidden_state[:, 1:]
                    # 原始实现中存在的 bug，这里修复了它，通常是高度、宽度的顺序
                    batch_size, _, height, width = pixel_values.shape
                    patch_size = self.config.patch_size
                    hidden_state = hidden_state.reshape(batch_size, height // patch_size, width // patch_size, -1)
                    hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()
                # 将生成的特征图添加到特征图元组中
                feature_maps += (hidden_state,)

        # 如果不使用 return_dict，则根据输出隐藏状态构建输出
        if not return_dict:
            if output_hidden_states:
                output = (feature_maps,) + outputs[1:]
            else:
                output = (feature_maps,) + outputs[2:]
            return output

        # 如果使用 return_dict，则构建 BackboneOutput 对象并返回
        return BackboneOutput(
            feature_maps=feature_maps,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
        )
```