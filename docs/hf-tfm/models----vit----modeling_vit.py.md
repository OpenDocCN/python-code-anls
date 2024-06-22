# `.\transformers\models\vit\modeling_vit.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 2021年由 Google AI、Ross Wightman 和 The HuggingFace Inc. 团队保留所有权利。
# 根据 Apache 许可证 2.0 版本（“许可证”）授权；
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据“原样”分发软件，
# 没有任何明示或暗示的担保或条件。
# 请查看许可证以获取特定语言的权限和限制。
""" PyTorch ViT model."""

# 导入必要的库
import collections.abc
import math
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入自定义的激活函数映射
from ...activations import ACT2FN
# 导入模型输出相关的类
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
    MaskedImageModelingOutput,
)
# 导入模型工具函数
from ...modeling_utils import PreTrainedModel
# 导入 PyTorch 工具函数
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
# 导入工具函数
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 导入 ViT 配置类
from .configuration_vit import ViTConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的配置信息
_CONFIG_FOR_DOC = "ViTConfig"
# 用于文档的检查点信息
_CHECKPOINT_FOR_DOC = "google/vit-base-patch16-224-in21k"
# 预期输出形状
_EXPECTED_OUTPUT_SHAPE = [1, 197, 768]

# 图像分类检查点
_IMAGE_CLASS_CHECKPOINT = "google/vit-base-patch16-224"
# 预期图像分类输出
_IMAGE_CLASS_EXPECTED_OUTPUT = "Egyptian cat"

# 预训练模型存档列表
VIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/vit-base-patch16-224",
    # 查看所有 ViT 模型 https://huggingface.co/models?filter=vit
]

# ViTEmbeddings 类，用于构建 CLS 令牌、位置和补丁嵌入。可选地，还包括掩码令牌。
class ViTEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, config: ViTConfig, use_mask_token: bool = False) -> None:
        super().__init__()

        # 定义 CLS 令牌参数
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        # 如果使用掩码令牌，则定义掩码令牌参数
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
        # 初始化补丁嵌入
        self.patch_embeddings = ViTPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        # 定义位置嵌入参数
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, config.hidden_size))
        # 定义丢弃层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        # 获取嵌入的补丁数量
        num_patches = embeddings.shape[1] - 1
        # 获取位置编码的位置数量
        num_positions = self.position_embeddings.shape[1] - 1
        # 如果补丁数量和位置数量相等，并且高度等于宽度，则直接返回位置编码
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        # 获取类别位置编码
        class_pos_embed = self.position_embeddings[:, 0]
        # 获取补丁位置编码
        patch_pos_embed = self.position_embeddings[:, 1:]
        # 获取嵌入的维度
        dim = embeddings.shape[-1]
        # 计算高度和宽度的初始值
        h0 = height // self.config.patch_size
        w0 = width // self.config.patch_size
        # 为了避免插值时的浮点误差，添加一个小数
        # 参考：https://github.com/facebookresearch/dino/issues/8
        h0, w0 = h0 + 0.1, w0 + 0.1
        # 重塑补丁位置编码的形状
        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
        # 转置补丁位置编码
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        # 插值补丁位置编码
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
            mode="bicubic",
            align_corners=False,
        )
        # 断言高度和宽度与补丁位置编码的形状匹配
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        # 转置补丁位置编码
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        # 拼接类别位置编码和补丁位置编码
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    # 定义函数的返回类型为 torch.Tensor
    ) -> torch.Tensor:
        # 获取输入张量的维度信息
        batch_size, num_channels, height, width = pixel_values.shape
        # 使用 patch_embeddings 函数对像素值进行嵌入处理，同时插值位置编码
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        # 如果存在掩码位置信息
        if bool_masked_pos is not None:
            # 获取嵌入张量的序列长度
            seq_length = embeddings.shape[1]
            # 创建掩码标记张量
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # 用 mask_tokens 替换掩码的视觉标记
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # 添加 [CLS] 标记到嵌入的补丁标记
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # 为每个标记添加位置编码
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings

        # 对嵌入进行 dropout 处理
        embeddings = self.dropout(embeddings)

        # 返回处理后的嵌入张量
        return embeddings
class ViTPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        # 初始化函数，接受配置参数，用于将像素值转换为补丁嵌入
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        # 将图像大小和补丁大小转换为可迭代对象
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        # 使用卷积层将通道数转换为隐藏大小
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        # 前向传播函数，将像素值转换为嵌入
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings


class ViTSelfAttention(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        # 初始化函数，接受配置参数，用于自注意力机制
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化查询、键、值线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    # 将输入张量 x 进行维度转换，将最后两个维度拆分成多个维度
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 实现自注意力机制的前向传播
    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 计算混合查询层
        mixed_query_layer = self.query(hidden_states)

        # 计算键层和值层，并进行维度转换
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 计算原始注意力分数，即查询与键的点积
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # 对注意力分数进行缩放
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 将注意力分数归一化为概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 使用 dropout 进行注意力概率的随机失活
        attention_probs = self.dropout(attention_probs)

        # 如果有头部掩码，则应用头部掩码
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文层，即注意力概率与值层的乘积
        context_layer = torch.matmul(attention_probs, value_layer)

        # 调整上下文层的维度
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # 根据是否输出注意力权重，返回不同的输出结果
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
class ViTSelfOutput(nn.Module):
    """
    The residual connection is defined in ViTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        # 定义一个全连接层，输入和输出维度都是 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义一个 dropout 层，概率为 config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将 hidden_states 输入到全连接层中
        hidden_states = self.dense(hidden_states)
        # 对全连接层的输出进行 dropout
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class ViTAttention(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        # 创建 ViTSelfAttention 对象
        self.attention = ViTSelfAttention(config)
        # 创建 ViTSelfOutput 对象
        self.output = ViTSelfOutput(config)
        # 初始化一个空集合，用于存储需要剪枝的头部
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        # 找到需要剪枝的头部和对应的索引
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
        # 调用 ViTSelfAttention 的 forward 方法
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        # 将 self_outputs[0] 和 hidden_states 输入到 ViTSelfOutput 中
        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # 如果需要输出注意力，将注意力添加到输出中
        return outputs


class ViTIntermediate(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        # 定义一个全连接层，输入维度为 config.hidden_size，输出维度为 config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 根据配置选择激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将 hidden_states 输入到全连接层中
        hidden_states = self.dense(hidden_states)
        # 使用激活函数处理全连接层的输出
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class ViTOutput(nn.Module):
    # 初始化函数，接受一个ViTConfig类型的参数，并调用父类的初始化函数
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        # 创建一个全连接层，输入大小为config.intermediate_size，输出大小为config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个Dropout层，概率为config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，接受两个torch.Tensor类型的参数，返回一个torch.Tensor类型的结果
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将hidden_states输入到全连接层中得到输出
        hidden_states = self.dense(hidden_states)
        # 对输出进行Dropout操作
        hidden_states = self.dropout(hidden_states)

        # 将Dropout后的输出与input_tensor相加
        hidden_states = hidden_states + input_tensor

        # 返回相加后的结果
        return hidden_states
class ViTLayer(nn.Module):
    """这对应于 timm 实现中的 Block 类。"""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ViTAttention(config)
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # 在 ViT 中，self-attention 之前应用 layernorm
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # 如果输出注意力权重，则添加 self attentions

        # 第一个残差连接
        hidden_states = attention_output + hidden_states

        # 在 ViT 中，self-attention 之后也应用 layernorm
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # 第二个残差连接在这里完成
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


class ViTEncoder(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([ViTLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    # 定义函数的返回类型为元组或BaseModelOutput类型
    ) -> Union[tuple, BaseModelOutput]:
        # 如果输出隐藏状态为True，则初始化一个空元组，否则为None
        all_hidden_states = () if output_hidden_states else None
        # 如果输出注意力权重为True，则初始化一个空元组，否则为None
        all_self_attentions = () if output_attentions else None

        # 遍历每个层的module
        for i, layer_module in enumerate(self.layer):
            # 如果输出隐藏状态为True，则将当前隐藏状态添加到all_hidden_states中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 如果启用梯度检查点且处于训练模式，则使用梯度检查点函数
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                # 否则直接调用当前层的module
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            # 更新隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]

            # 如果输出注意力权重为True，则将当前层的注意力权重添加到all_self_attentions中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果输出隐藏状态为True，则将最终隐藏状态添加到all_hidden_states中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典，则返回非空的hidden_states, all_hidden_states, all_self_attentions元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 否则返回BaseModelOutput对象
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
class ViTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置配置类为 ViTConfig
    config_class = ViTConfig
    # 设置基础模型前缀为 "vit"
    base_model_prefix = "vit"
    # 设置主输入名称为 "pixel_values"
    main_input_name = "pixel_values"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 不进行模块拆分的列表
    _no_split_modules = ["ViTEmbeddings", "ViTLayer"]

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        # 如果是线性层或卷积层，使用截断正态分布初始化权重，偏置数据为零
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 将输入转换为 fp32，并返回到所需的 dtype 以避免在 half 中无法实现 trunc_normal_cpu
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是 LayerNorm 层，偏置数据为零，权重数据填充为 1.0
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        # 如果是 ViTEmbeddings 层，对位置嵌入和类别标记进行初始化
        elif isinstance(module, ViTEmbeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.cls_token.dtype)


VIT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ViTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

VIT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            # 像素值。可以使用 [`AutoImageProcessor`] 获得像素值。有关详细信息，请参阅 [`ViTImageProcessor.__call__`]

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于将自注意力模块的选定头部置零的掩码。掩码值选定在 `[0, 1]` 之间：

            # - 1 表示头部被 **不屏蔽**，
            # - 0 表示头部被 **屏蔽**。

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回的张量中的 `attentions`。

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多详细信息，请参见返回的张量中的 `hidden_states`。

        interpolate_pos_encoding (`bool`, *optional*):
            # 是否插值预训练的位置编码。

        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通的元组。
"""
从ViTConfig继承的ViT模型转换器，输出没有特定头部的原始隐藏状态。

ViT模型的文档字符串，

类的定义开始
class ViTModel(ViTPreTrainedModel):

    def __init__(self, config: ViTConfig, add_pooling_layer: bool = True, use_mask_token: bool = False):
        # 初始化方法, 接收ViTConfig对象作为参数
        super().__init__(config)
        self.config = config

        # 创建ViTEmbeddings对象和ViTEncoder对象
        self.embeddings = ViTEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = ViTEncoder(config)

        # 初始化LayerNorm和ViTPooler对象
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViTPooler(config) if add_pooling_layer else None

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self) -> ViTPatchEmbeddings:
        # 返回patch_embeddings属性
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        剪枝模型的头部。heads_to_prune：{层编号: 需要在此层剪枝的头部列表} 的字典格式。参照基类PreTrainedModel。
        """
        # 遍历需要剪枝的头部进行操作
        for layer, heads in heads_to_prune.items():
            # 调用encoder中对应层的attention的prune_heads方法来剪枝
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 增加前向传播函数的文档字符串
    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
    # 增加代码示例的文档字符串
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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        # Generate the output of the embeddings module
        embedding_output = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )

        # Encode the embedding outputs using the transformer encoder
        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Apply layer normalization to the sequence output
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        
        # Generate a pooled output using the pooler layer if it exists
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        # Return output based on whether return_dict is True or False
        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
class ViTPooler(nn.Module):
    # 初始化 ViTPooler 类
    def __init__(self, config: ViTConfig):
        super().__init__()
        # 创建全连接层，输入和输出维度均为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建激活函数，使用 Tanh 函数
        self.activation = nn.Tanh()

    # 前向传播函数
    def forward(self, hidden_states):
        # 通过选择第一个 token 对应的隐藏状态来"池化"模型
        first_token_tensor = hidden_states[:, 0]
        # 对第一个 token 的隐藏状态进行线性变换
        pooled_output = self.dense(first_token_tensor)
        # 使用激活函数激活输出
        pooled_output = self.activation(pooled_output)
        # 返回池化后的输出
        return pooled_output


# 通过添加文档字符串进行模型说明
@add_start_docstrings(
    """ViT Model with a decoder on top for masked image modeling, as proposed in [SimMIM](https://arxiv.org/abs/2111.09886).

    <Tip>

    Note that we provide a script to pre-train this model on custom data in our [examples
    directory](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining).

    </Tip>
    """,
    VIT_START_DOCSTRING,
)
class ViTForMaskedImageModeling(ViTPreTrainedModel):
    # 初始化 ViTForMaskedImageModeling 类
    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)

        # 创建 ViT 模型，不添加池化层，使用掩码标记
        self.vit = ViTModel(config, add_pooling_layer=False, use_mask_token=True)

        # 创建解码器层，包括 1 个卷积层和像素分解层
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

    # 设置前向传播函数的输入输出说明
    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MaskedImageModelingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 前向传播函数的具体实现，在该函数中完成模型的前向计算

# 通过���加文档字符串进行模型说明
@add_start_docstrings(
    """
    ViT Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.

    <Tip>

        Note that it's possible to fine-tune ViT on higher resolution images than the ones it has been trained on, by
        setting `interpolate_pos_encoding` to `True` in the forward of the model. This will interpolate the pre-trained
        position embeddings to the higher resolution.

    </Tip>
    """,
    VIT_START_DOCSTRING,
)
class ViTForImageClassification(ViTPreTrainedModel):
    # 初始化 ViTForImageClassification 类
    # 初始化函数，接受一个 ViTConfig 类型的参数并调用父类初始化函数
    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)

        # 将配置中的标签数量赋值给当前类的 num_labels 属性
        self.num_labels = config.num_labels
        # 使用 ViTModel 类初始化一个 ViT 模型，关闭额外的池化层
        self.vit = ViTModel(config, add_pooling_layer=False)

        # 分类器头部
        # 如果配置中标签数量大于零，则使用 nn.Linear 创建一个线性分类器，否则使用 nn.Identity 创建一个恒等映射
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数，包含多个可选参数
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
        # 检查是否需要返回字典类型的结果，若未指定则使用配置中的默认设置
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

        # 获取序列输出
        sequence_output = outputs[0]

        # 将序列输出的首个标记位置的输出传入分类器，得到分类结果
        logits = self.classifier(sequence_output[:, 0, :])

        # 初始化损失值
        loss = None
        # 若存在标签
        if labels is not None:
            # 将标签移到正确的设备上以启用模型并行处理
            labels = labels.to(logits.device)
            # 若问题类型未指定，则根据标签数量确定问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型计算损失值
            if self.config.problem_type == "regression":
                # 使用均方误差损失函数
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    # 若标签数量为1，计算标量回归损失
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    # 否则计算回归损失
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # 使用交叉熵损失函数
                loss_fct = CrossEntropyLoss()
                # 计算单标签分类损失
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # 使用带有逻辑回归的二元交叉熵损失函数
                loss_fct = BCEWithLogitsLoss()
                # 计算多标签分类损失
                loss = loss_fct(logits, labels)

        # 若不需要返回字典类型的结果，则返回输出元组
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回图像分类器输出
        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```