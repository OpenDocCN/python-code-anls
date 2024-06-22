# `.\transformers\models\vit_msn\modeling_vit_msn.py`

```py
# 设置文件编码为 UTF-8

# 导入所需模块和库
# collections.abc 用于类型提示
# math 用于数学运算
# typing 用于类型提示
import collections.abc
import math
from typing import Dict, List, Optional, Set, Tuple, Union

# 导入 PyTorch 库
import torch
import torch.utils.checkpoint
# 导入 PyTorch 中的各种损失函数
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入相关模块和函数
# activiations 激活函数相关
from ...activations import ACT2FN
# modeling_outputs 输出相关
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
# modeling_utils 模型相关工具函数
from ...modeling_utils import PreTrainedModel
# pytorch_utils PyTorch 相关工具函数
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
# utils 常用工具函数
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
# 导入 ViTMSN 相关配置
from .configuration_vit_msn import ViTMSNConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的配置和检查点说明
_CONFIG_FOR_DOC = "ViTMSNConfig"
_CHECKPOINT_FOR_DOC = "facebook/vit-msn-small"
# 预训练模型存档列表
VIT_MSN_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/vit-msn-small",
    # 查看所有 ViTMSN 模型：https://huggingface.co/models?filter=vit_msn
]

# ViTMSNEmbeddings 类用于构建 CLS 令牌、位置和补丁嵌入，以及可选的掩码令牌
class ViTMSNEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """

    # 初始化函数
    def __init__(self, config: ViTMSNConfig, use_mask_token: bool = False) -> None:
        super().__init__()

        # 定义 CLS 令牌参数
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        # 如果使用掩码令牌，则定义掩码令牌参数
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
        # 创建补丁嵌入对象
        self.patch_embeddings = ViTMSNPatchEmbeddings(config)
        # 获取补丁数量
        num_patches = self.patch_embeddings.num_patches
        # 定义位置嵌入参数
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
        # 定义丢弃层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 保存配置信息
        self.config = config
``` 
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        插值预训练位置编码，以便在更高分辨率的图像上使用模型。

        来源:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        # 获取嵌入的补丁数量和位置编码的数量
        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1
        # 如果补丁数量等于位置编码数量且图像高度和宽度相等，则直接返回位置编码
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        # 分离类别位置编码和补丁位置编码
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        # 计算补丁窗口的高度和宽度
        patch_window_height = height // self.config.patch_size
        patch_window_width = width // self.config.patch_size
        # 为了避免插值时的浮点误差，添加一个小数
        patch_window_height, patch_window_width = patch_window_height + 0.1, patch_window_width + 0.1
        # 重新形状补丁位置编码以便进行插值
        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        # 使用双三次插值对补丁位置编码进行插值
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(
                patch_window_height / math.sqrt(num_positions),
                patch_window_width / math.sqrt(num_positions),
            ),
            mode="bicubic",
            align_corners=False,
        )
        # 重新排列补丁位置编码的维度
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        # 将类别位置编码和插值后的补丁位置编码拼接在一起
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    # 定义函数，接收像素值作为输入，返回嵌入表示
    def forward(self, pixel_values: torch.Tensor, bool_masked_pos: torch.Tensor = None, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        # 获取输入张量的形状信息
        batch_size, num_channels, height, width = pixel_values.shape
        # 对像素值进行补丁嵌入处理
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        # 如果存在被掩盖位置的标记
        if bool_masked_pos is not None:
            # 获取嵌入张量的长度
            seq_length = embeddings.shape[1]
            # 根据掩码位置，创建相应数量的掩盖标记
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # 根据掩盖位置替换已嵌入的视觉标记
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

        # 对嵌入进行丢弃操作
        embeddings = self.dropout(embeddings)

        # 返回处理后的嵌入张量
        return embeddings
# 从transformers.models.vit.modeling_vit.ViTPatchEmbeddings复制并改名为ViTMSNPatchEmbeddings
class ViTMSNPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
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

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
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


# 从transformers.models.vit.modeling_vit.ViTSelfAttention复制并改名为ViTMSNSelfAttention
class ViTMSNSelfAttention(nn.Module):
    # 初始化函数，接受一个 ViTMSNConfig 对象作为配置参数
    def __init__(self, config: ViTMSNConfig) -> None:
        # 调用父类初始化函数
        super().__init__()
        
        # 检查隐藏层大小是否是注意力头数量的整数倍并且是否包含嵌入大小属性
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        # 初始化注意力头数量、每个注意力头的大小以及总的头大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化查询、键、值线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        # 初始化dropout层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    # 通过修改张量的形状，将张量准备用于计算注意力分数
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播函数，接收隐藏状态、头掩码和是否输出注意力权重作为输入
    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 计算混合的查询层
        mixed_query_layer = self.query(hidden_states)

        # 计算键层和值层，然后将其准备为计算注意力分数
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 计算原始的注意力分数，即查询和键的点积
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # 对注意力分数进行缩放
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 将注意力分数标准化为概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 使用dropout层来进行随机失活，以减少过拟合
        attention_probs = self.dropout(attention_probs)

        # 如果需要，对头进行掩码操作
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文层，即注意力概率和值的加权和
        context_layer = torch.matmul(attention_probs, value_layer)

        # 将上下文层的维度进行调整
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # 根据是否需要输出注意力权重，返回不同的结果
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
        # 创建一个全连接层，输入和输出尺寸均为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个dropout层，以 config.hidden_dropout_prob 为丢弃概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 通过全连接层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 通过dropout层进行丢弃
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# 从 transformers.models.vit.modeling_vit.ViTAttention 复制并修改为 ViT->ViTMSN
class ViTMSNAttention(nn.Module):
    def __init__(self, config: ViTMSNConfig) -> None:
        super().__init__()
        # 创建 ViTMSNSelfAttention 层和 ViTMSNSelfOutput 层
        self.attention = ViTMSNSelfAttention(config)
        self.output = ViTMSNSelfOutput(config)
        # 存储需要剔除的头的集合
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储已剔除的头
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 通过 ViTMSNSelfAttention 层获取 self_outputs
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        # 通过 ViTMSNSelfOutput 层将 self_outputs[0] 和 hidden_states 输出为 attention_output
        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # 如果输出了，加上注意力
        return outputs


# 从 transformers.models.vit.modeling_vit.ViTIntermediate 复制并修改为 ViT->ViTMSN
class ViTMSNIntermediate(nn.Module):
    def __init__(self, config: ViTMSNConfig) -> None:
        super().__init__()
        # 创建一个全连接层，输入尺寸为 config.hidden_size，输出尺寸为 config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果 hidden_act 是字符串类型，则使用对应的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            # 否则使用配置中的隐藏激活函数
            self.intermediate_act_fn = config.hidden_act
    # 前向传播计算
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过全连接层进行变换
        hidden_states = self.dense(hidden_states)
        # 应用中间激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回计算结果
        return hidden_states
# 定义一个 ViTMSNOutput 模块，继承自 nn.Module
# 这个模块用于在 ViT-MSN 模型中进行最后的输出处理
class ViTMSNOutput(nn.Module):
    def __init__(self, config: ViTMSNConfig) -> None:
        # 调用父类 nn.Module 的初始化方法
        super().__init__()
        # 定义一个线性层，将中间层的隐藏状态映射到最终的隐藏状态
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 定义一个 Dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用线性层将中间层的隐藏状态映射到最终的隐藏状态
        hidden_states = self.dense(hidden_states)
        # 使用 Dropout 层对最终的隐藏状态进行处理
        hidden_states = self.dropout(hidden_states)
        # 将最终的隐藏状态与输入张量相加，得到最终的输出
        hidden_states = hidden_states + input_tensor
        return hidden_states


# 定义一个 ViTMSNLayer 模块，继承自 nn.Module
# 这个模块对应于 timm 实现中的 Block 类
class ViTMSNLayer(nn.Module):
    def __init__(self, config: ViTMSNConfig) -> None:
        # 调用父类 nn.Module 的初始化方法
        super().__init__()
        # 设置 chunk_size_feed_forward 和 seq_len_dim
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        # 定义 ViTMSNAttention、ViTMSNIntermediate 和 ViTMSNOutput 模块
        self.attention = ViTMSNAttention(config)
        self.intermediate = ViTMSNIntermediate(config)
        self.output = ViTMSNOutput(config)
        # 定义两个 LayerNorm 层，分别用于 self-attention 前后
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 在 self-attention 之前应用 LayerNorm
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),
            head_mask,
            output_attentions=output_attentions,
        )
        # 获取 self-attention 的输出
        attention_output = self_attention_outputs[0]
        # 将 self-attention 的其他输出也返回
        outputs = self_attention_outputs[1:]

        # 进行第一个残差连接
        hidden_states = attention_output + hidden_states

        # 在 self-attention 之后再次应用 LayerNorm
        layer_output = self.layernorm_after(hidden_states)
        # 通过中间层处理
        layer_output = self.intermediate(layer_output)
        # 进行第二个残差连接
        layer_output = self.output(layer_output, hidden_states)

        # 将最终的输出和其他输出一起返回
        outputs = (layer_output,) + outputs

        return outputs


# 定义一个 ViTMSNEncoder 模块，继承自 nn.Module
# 这个模块用于构建 ViT-MSN 模型的编码器部分
class ViTMSNEncoder(nn.Module):
    def __init__(self, config: ViTMSNConfig) -> None:
        # 调用父类 nn.Module 的初始化方法
        super().__init__()
        # 保存配置信息
        self.config = config
        # 定义多个 ViTMSNLayer 模块组成的层列表
        self.layer = nn.ModuleList([ViTMSNLayer(config) for _ in range(config.num_hidden_layers)])
        # 是否启用梯度检查点
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 这里省略了具体的前向计算逻辑
        # 该部分代码将在下一个回答中提供
    # 设置返回结果类型为元组或者 BaseModelOutput 类型
    ) -> Union[tuple, BaseModelOutput]:
        # 如果不输出隐藏状态，则设置空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果不输出注意力权重，则设置空元组
        all_self_attentions = () if output_attentions else None

        # 遍历每个层次的模块
        for i, layer_module in enumerate(self.layer):
            # 如果输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            # 获取当前层的头部屏蔽情况
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 如果使用梯度检查点并且正在训练，则使用梯度检查点方法进行模块调用
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                # 否则直接调用当前层的模块
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            # 更新隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]

            # 如果输出注意力权重，则将当前层的注意力权重添加到 all_self_attentions 中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果输出隐藏状态，则将最终隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典，则返回存在值的元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 否则返回 BaseModelOutput 类型对象
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

    # 设置配置类为 ViTMSNConfig
    config_class = ViTMSNConfig
    # 设置基础模型前缀为 "vit"
    base_model_prefix = "vit"
    # 设置主要输入名称为 "pixel_values"
    main_input_name = "pixel_values"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    # todo: 为创建预训练脚本，请参考 https://github.com/facebookresearch/msn/blob/main/src/deit.py#L200-#L211
    # 在这里需要初始化模型权重
    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        # 如果是 Linear 或 Conv2d 模块
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 使用正态分布初始化权重，平均值为 0，标准差为配置中的initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置项，将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是 LayerNorm 模块
        elif isinstance(module, nn.LayerNorm):
            # 将偏置项初始化为零
            module.bias.data.zero_()
            # 将权重初始化为全1
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

# 输入文档字符串
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
# 导入 add_start_docstrings 和 VIT_MSN_START_DOCSTRING 装饰器，以及 BaseModelOutput 和 _CONFIG_FOR_DOC
@add_start_docstrings(
    "The bare ViTMSN Model outputting raw hidden-states without any specific head on top.",
    VIT_MSN_START_DOCSTRING,
)
# 定义 ViTMSNModel 类，继承自 ViTMSNPreTrainedModel
class ViTMSNModel(ViTMSNPreTrainedModel):
    # 初始化函数，接受配置参数以及是否使用掩码标记
    def __init__(self, config: ViTMSNConfig, use_mask_token: bool = False):
        # 调用父类的初始化方法
        super().__init__(config)
        # 保存配置
        self.config = config

        # 创建嵌入层
        self.embeddings = ViTMSNEmbeddings(config, use_mask_token=use_mask_token)
        # 创建编码器
        self.encoder = ViTMSNEncoder(config)

        # 创建归一化层
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self) -> ViTMSNPatchEmbeddings:
        return self.embeddings.patch_embeddings

    # 裁剪注意力头
    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 前向传播函数
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
    ) -> Union[tuple, BaseModelOutput]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).

        Returns:

        Examples:

        ```py
        >>> from transformers import AutoImageProcessor, ViTMSNModel
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/vit-msn-small")
        >>> model = ViTMSNModel.from_pretrained("facebook/vit-msn-small")
        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> with torch.no_grad():
        ...     outputs = model(**inputs)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
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

        embedding_output = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        if not return_dict:
            head_outputs = (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 警告：我们尚未为分类头部准备好权重。这个类是为那些有兴趣对基础模型（ViTMSNModel）进行微调的用户而设的。
@add_start_docstrings(
    """
    在顶部带有图像分类头的 ViTMSN 模型，例如用于 ImageNet。
    """,
    VIT_MSN_START_DOCSTRING,
)
class ViTMSNForImageClassification(ViTMSNPreTrainedModel):
    def __init__(self, config: ViTMSNConfig) -> None:
        # 调用父类的初始化方法
        super().__init__(config)

        # 获取类别数
        self.num_labels = config.num_labels
        # 创建 ViTMSN 模型
        self.vit = ViTMSNModel(config)

        # 分类器头部
        # 如果类别数大于 0，则创建一个线性分类器，否则创建一个 Identity 层
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # 初始化权重并应用最终处理
        self.post_init()

    # 重写前向传播方法，添加输入和输出的文档说明
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