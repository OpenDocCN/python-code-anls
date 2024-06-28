# `.\models\yolos\modeling_yolos.py`

```
# 设置文件编码为UTF-8
# 版权声明，指出2022年版权归华中科技大学电信学院和HuggingFace团队所有
# 根据Apache许可证2.0版，除非符合许可证的规定，否则禁止使用此文件
# 可以在以下链接找到完整的许可证内容：http://www.apache.org/licenses/LICENSE-2.0
""" PyTorch YOLOS 模型."""

# 导入必要的模块
import collections.abc  # 引入集合抽象基类
import math  # 引入数学库
from dataclasses import dataclass  # 引入数据类
from typing import Dict, List, Optional, Set, Tuple, Union  # 引入类型提示

import torch  # 引入PyTorch
import torch.utils.checkpoint  # 引入PyTorch的checkpoint功能
from torch import Tensor, nn  # 从PyTorch中引入张量和神经网络模块

# 导入相关的模型输出和工具函数
from ...activations import ACT2FN  # 从activations模块导入激活函数映射
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling  # 导入基础模型输出类
from ...modeling_utils import PreTrainedModel  # 导入预训练模型基类
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer  # 导入模型剪枝工具函数
from ...utils import (  # 导入各种实用函数和类
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_accelerate_available,
    is_scipy_available,
    is_vision_available,
    logging,
    replace_return_docstrings,
    requires_backends,
)
from .configuration_yolos import YolosConfig  # 导入YOLOS模型的配置类

# 如果scipy可用，则导入线性求和分配优化工具
if is_scipy_available():
    from scipy.optimize import linear_sum_assignment

# 如果vision模块可用，则导入图像转换相关函数
if is_vision_available():
    from transformers.image_transforms import center_to_corners_format

# 如果accelerate可用，则导入部分状态和相关的降维函数
if is_accelerate_available():
    from accelerate import PartialState
    from accelerate.utils import reduce

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

# 用于文档的配置信息
_CONFIG_FOR_DOC = "YolosConfig"

# 用于文档的检查点信息
_CHECKPOINT_FOR_DOC = "hustvl/yolos-small"

# 预期的输出形状
_EXPECTED_OUTPUT_SHAPE = [1, 3401, 384]

# YOLOS预训练模型存档列表
YOLOS_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "hustvl/yolos-small",
    # 查看所有YOLOS模型，请访问：https://huggingface.co/models?filter=yolos
]

@dataclass
class YolosObjectDetectionOutput(ModelOutput):
    """
    [`YolosForObjectDetection`] 的输出类型。
    """
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)):
            Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
            bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
            scale-invariant IoU loss.
        loss_dict (`Dict`, *optional*):
            A dictionary containing the individual losses. Useful for logging.
        logits (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes + 1)`):
            Classification logits (including no-object) for all queries.
        pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)`):
            Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height). These
            values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
            possible padding). You can use [`~YolosImageProcessor.post_process`] to retrieve the unnormalized bounding
            boxes.
        auxiliary_outputs (`list[Dict]`, *optional*):
            Optional, only returned when auxilary losses are activated (i.e. `config.auxiliary_loss` is set to `True`)
            and labels are provided. It is a list of dictionaries containing the two above keys (`logits` and
            `pred_boxes`) for each decoder layer.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of
            the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    # Optional variables initialized to None, indicating they may or may not be present
    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict] = None
    logits: torch.FloatTensor = None
    pred_boxes: torch.FloatTensor = None
    auxiliary_outputs: Optional[List[Dict]] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# 定义一个名为 YolosEmbeddings 的类，继承自 nn.Module，用于构建 CLS token、检测 token、位置和补丁嵌入。
class YolosEmbeddings(nn.Module):
    """
    Construct the CLS token, detection tokens, position and patch embeddings.
    构建 CLS token、检测 token、位置和补丁嵌入。
    """

    def __init__(self, config: YolosConfig) -> None:
        super().__init__()

        # 定义一个可学习参数，用于表示 CLS token，形状为 [1, 1, hidden_size]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        
        # 定义一个可学习参数，用于表示检测 token，形状为 [1, num_detection_tokens, hidden_size]
        self.detection_tokens = nn.Parameter(torch.zeros(1, config.num_detection_tokens, config.hidden_size))
        
        # 使用 YolosPatchEmbeddings 类构建补丁嵌入对象
        self.patch_embeddings = YolosPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        
        # 定义一个可学习参数，用于表示位置嵌入，形状为 [1, num_patches + num_detection_tokens + 1, hidden_size]
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, num_patches + config.num_detection_tokens + 1, config.hidden_size)
        )

        # 定义一个 dropout 层，用于在训练过程中随机断开一些神经元连接，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # 创建一个 InterpolateInitialPositionEmbeddings 的实例，用于插值初始位置嵌入
        self.interpolation = InterpolateInitialPositionEmbeddings(config)
        
        # 保存配置参数，以便后续使用
        self.config = config

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        
        # 使用补丁嵌入对象处理输入的像素值，得到嵌入向量
        embeddings = self.patch_embeddings(pixel_values)

        batch_size, seq_len, _ = embeddings.size()

        # 将 [CLS] token 和检测 token 添加到嵌入的补丁 token 中
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        detection_tokens = self.detection_tokens.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings, detection_tokens), dim=1)

        # 添加位置编码到每个 token 中
        # 可能需要对现有的位置嵌入进行插值处理
        position_embeddings = self.interpolation(self.position_embeddings, (height, width))

        embeddings = embeddings + position_embeddings

        # 对嵌入向量应用 dropout 层
        embeddings = self.dropout(embeddings)

        return embeddings


# 定义一个名为 InterpolateInitialPositionEmbeddings 的类，继承自 nn.Module，用于插值初始位置嵌入。
class InterpolateInitialPositionEmbeddings(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
    # 定义一个前向传播函数，用于处理位置嵌入和图像大小参数，返回一个张量
    def forward(self, pos_embed, img_size=(800, 1344)) -> torch.Tensor:
        # 从位置嵌入中提取类别位置嵌入，形状为(batch_size, hidden_size)
        cls_pos_embed = pos_embed[:, 0, :]
        # 添加一个维度使其形状变为(batch_size, 1, hidden_size)，用于后续拼接
        cls_pos_embed = cls_pos_embed[:, None]
        # 从位置嵌入中提取检测位置嵌入，形状为(batch_size, num_detection_tokens, hidden_size)
        det_pos_embed = pos_embed[:, -self.config.num_detection_tokens :, :]
        # 从位置嵌入中提取除类别和检测外的其余位置嵌入，形状为(batch_size, seq_len - num_detection_tokens - 1, hidden_size)
        patch_pos_embed = pos_embed[:, 1 : -self.config.num_detection_tokens, :]
        # 将形状为(batch_size, hidden_size, seq_len - num_detection_tokens - 1)的张量转置为(batch_size, seq_len - num_detection_tokens - 1, hidden_size)
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        # 获取批次大小、隐藏大小和序列长度
        batch_size, hidden_size, seq_len = patch_pos_embed.shape

        # 计算图像中的分块高度和宽度
        patch_height, patch_width = (
            self.config.image_size[0] // self.config.patch_size,
            self.config.image_size[1] // self.config.patch_size,
        )
        # 将位置嵌入重塑为(batch_size, hidden_size, patch_height, patch_width)
        patch_pos_embed = patch_pos_embed.view(batch_size, hidden_size, patch_height, patch_width)

        # 重新调整分块的位置嵌入大小至新的分块高度和宽度
        height, width = img_size
        new_patch_heigth, new_patch_width = height // self.config.patch_size, width // self.config.patch_size
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed, size=(new_patch_heigth, new_patch_width), mode="bicubic", align_corners=False
        )
        # 展平重新调整后的位置嵌入，形状变为(batch_size, seq_len - num_detection_tokens - 1, hidden_size)
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)
        # 拼接类别位置嵌入、分块位置嵌入和检测位置嵌入，形状为(batch_size, seq_len, hidden_size)
        scale_pos_embed = torch.cat((cls_pos_embed, patch_pos_embed, det_pos_embed), dim=1)
        # 返回拼接后的位置嵌入张量
        return scale_pos_embed
class InterpolateMidPositionEmbeddings(nn.Module):
    """
    模块用于在Transformer模型中插值中间位置的位置嵌入。

    Args:
        config: 模型配置参数对象

    Attributes:
        config: 存储模型配置参数的对象

    Methods:
        forward(pos_embed, img_size=(800, 1344)): 前向传播方法，用于计算位置嵌入的插值结果。

    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    def forward(self, pos_embed, img_size=(800, 1344)) -> torch.Tensor:
        """
        执行前向传播计算插值后的位置嵌入。

        Args:
            pos_embed: 位置嵌入张量，形状为(batch_size, seq_length, hidden_size, seq_len)
            img_size: 图像大小元组，默认为(800, 1344)

        Returns:
            scale_pos_embed: 插值后的位置嵌入张量，形状为(batch_size, seq_length, hidden_size)

        """
        # 提取CLS位置嵌入，保持维度
        cls_pos_embed = pos_embed[:, :, 0, :]
        cls_pos_embed = cls_pos_embed[:, None]
        
        # 提取检测标记位置嵌入
        det_pos_embed = pos_embed[:, :, -self.config.num_detection_tokens :, :]
        
        # 提取补丁位置嵌入，并转置张量的最后两个维度
        patch_pos_embed = pos_embed[:, :, 1 : -self.config.num_detection_tokens, :]
        patch_pos_embed = patch_pos_embed.transpose(2, 3)
        
        # 获取补丁嵌入张量的形状信息
        depth, batch_size, hidden_size, seq_len = patch_pos_embed.shape
        
        # 将补丁嵌入张量重塑为(batch_size * depth, hidden_size, patch_height, patch_width)
        patch_height, patch_width = (
            self.config.image_size[0] // self.config.patch_size,
            self.config.image_size[1] // self.config.patch_size,
        )
        patch_pos_embed = patch_pos_embed.view(depth * batch_size, hidden_size, patch_height, patch_width)
        
        # 插值新的补丁位置嵌入至目标图像大小
        height, width = img_size
        new_patch_height, new_patch_width = height // self.config.patch_size, width // self.config.patch_size
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed, size=(new_patch_height, new_patch_width), mode="bicubic", align_corners=False
        )
        
        # 将插值后的补丁位置嵌入张量重塑为(batch_size, depth, new_patch_height * new_patch_width, hidden_size)
        patch_pos_embed = (
            patch_pos_embed.flatten(2)
            .transpose(1, 2)
            .contiguous()
            .view(depth, batch_size, new_patch_height * new_patch_width, hidden_size)
        )
        
        # 拼接不同部分的位置嵌入张量：CLS位置嵌入 + 插值后的补丁位置嵌入 + 检测标记位置嵌入
        scale_pos_embed = torch.cat((cls_pos_embed, patch_pos_embed, det_pos_embed), dim=2)
        
        return scale_pos_embed


class YolosPatchEmbeddings(nn.Module):
    """
    此类将输入的`pixel_values`（形状为(batch_size, num_channels, height, width)）转换为Transformer模型消费的初始隐藏状态（补丁嵌入），
    形状为(batch_size, seq_length, hidden_size)。

    Args:
        config: 模型配置参数对象

    Attributes:
        image_size: 图像大小元组
        patch_size: 补丁大小元组
        num_channels: 输入图像的通道数
        num_patches: 图像中的补丁数量

    Methods:
        __init__(config): 初始化方法，设置类属性和卷积投影
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        # 确保图像大小和补丁大小是可迭代对象
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        
        # 计算图像中的补丁数量
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        
        # 设置类属性
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        
        # 定义卷积投影层，将输入图像转换为补丁嵌入
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
    # 定义一个方法 `forward`，接收一个名为 `pixel_values` 的张量作为输入，并返回一个张量作为输出
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # 获取输入张量的批大小、通道数、高度和宽度
        batch_size, num_channels, height, width = pixel_values.shape
        # 如果输入张量的通道数不等于预设的 `self.num_channels`，抛出数值错误异常
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )

        # 对输入张量应用投影层 `self.projection`，并将结果扁平化为二维，然后交换维度 1 和 2
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        # 返回处理后的嵌入张量
        return embeddings
# 从transformers.models.vit.modeling_vit.ViTSelfAttention复制代码到YolosSelfAttention并替换ViT为Yolos
class YolosSelfAttention(nn.Module):
    def __init__(self, config: YolosConfig) -> None:
        super().__init__()
        # 检查隐藏大小是否是注意力头数的倍数，若不是则抛出数值错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 创建query、key、value线性层，用于计算注意力分数
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        # 定义Dropout层用于注意力概率的dropout操作
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # 将输入张量x转换为注意力分数矩阵的形状
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 计算混合的查询层
        mixed_query_layer = self.query(hidden_states)

        # 计算键值对应的注意力分数矩阵
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 计算原始的注意力分数，通过query和key的点积得到
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # 对注意力分数进行缩放，除以注意力头大小的平方根
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 对注意力分数进行softmax归一化得到注意力概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 使用Dropout对注意力概率进行随机置零处理
        attention_probs = self.dropout(attention_probs)

        # 如果有头掩码，则将头掩码应用到注意力概率上
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文向量，通过注意力概率加权求和value层
        context_layer = torch.matmul(attention_probs, value_layer)

        # 重新排列上下文向量的形状以便后续处理
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # 根据需要返回上下文向量和注意力概率，或仅返回上下文向量
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
# Copied from transformers.models.vit.modeling_vit.ViTSelfOutput with ViT->Yolos
class YolosSelfOutput(nn.Module):
    """
    The residual connection is defined in YolosLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: YolosConfig) -> None:
        super().__init__()
        # 定义一个全连接层，输入和输出维度都是 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义一个 dropout 层，根据配置中的 dropout 概率进行丢弃操作
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将输入的 hidden_states 通过全连接层 self.dense 进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对经过全连接层的输出进行 dropout 操作
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTAttention with ViT->Yolos
class YolosAttention(nn.Module):
    def __init__(self, config: YolosConfig) -> None:
        super().__init__()
        # 初始化 YolosSelfAttention 层
        self.attention = YolosSelfAttention(config)
        # 初始化 YolosSelfOutput 层
        self.output = YolosSelfOutput(config)
        # 初始化一个空集合，用于存储待剪枝的注意力头部索引
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        # 如果待剪枝的头部集合为空，则直接返回
        if len(heads) == 0:
            return
        # 调用辅助函数 find_pruneable_heads_and_indices 寻找可剪枝的头部及其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储已剪枝的头部索引
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 调用 self.attention 进行注意力计算，返回自注意力的输出
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        # 将自注意力的输出 self_outputs[0] 和输入 hidden_states 传入 self.output 层
        attention_output = self.output(self_outputs[0], hidden_states)

        # 如果需要输出注意力权重，则在 outputs 中包含它们
        outputs = (attention_output,) + self_outputs[1:]  # 如果输出注意力权重，则添加它们到输出中
        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTIntermediate with ViT->Yolos
class YolosIntermediate(nn.Module):
    def __init__(self, config: YolosConfig) -> None:
        super().__init__()
        # 定义一个全连接层，输入维度是 config.hidden_size，输出维度是 config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果 config.hidden_act 是字符串类型，则选择对应的激活函数，否则直接使用给定的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
    # 定义一个类方法 `forward`，用于执行前向传播操作，接收隐藏状态作为输入，并返回处理后的隐藏状态
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入的隐藏状态通过全连接层 `self.dense` 进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的隐藏状态应用激活函数 `self.intermediate_act_fn` 进行非线性变换
        hidden_states = self.intermediate_act_fn(hidden_states)
    
        # 返回经过线性变换和激活函数处理后的隐藏状态作为结果
        return hidden_states
# 从transformers.models.vit.modeling_vit.ViTOutput复制并将ViT改为Yolos
class YolosOutput(nn.Module):
    def __init__(self, config: YolosConfig) -> None:
        super().__init__()
        # 创建一个全连接层，输入大小为config.intermediate_size，输出大小为config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个dropout层，以config.hidden_dropout_prob的概率丢弃输入
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将输入hidden_states传递到全连接层中进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对全连接层的输出应用dropout操作
        hidden_states = self.dropout(hidden_states)

        # 将dropout后的输出与输入tensor相加，实现残差连接
        hidden_states = hidden_states + input_tensor

        return hidden_states


# 从transformers.models.vit.modeling_vit.ViTLayer复制并将ViT改为Yolos
class YolosLayer(nn.Module):
    """这对应于timm实现中的Block类。"""

    def __init__(self, config: YolosConfig) -> None:
        super().__init__()
        # 设置用于分块前馈的chunk大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 设置序列长度的维度为1
        self.seq_len_dim = 1
        # 创建一个YolosAttention实例
        self.attention = YolosAttention(config)
        # 创建一个YolosIntermediate实例
        self.intermediate = YolosIntermediate(config)
        # 创建一个YolosOutput实例
        self.output = YolosOutput(config)
        # 创建一个LayerNorm层，在隐藏大小为config.hidden_size时使用config.layer_norm_eps作为epsilon
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个LayerNorm层，在隐藏大小为config.hidden_size时使用config.layer_norm_eps作为epsilon
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 在Yolos中，layernorm在self-attention之前应用
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # 如果输出注意力权重，将其添加到输出中

        # 第一个残差连接
        hidden_states = attention_output + hidden_states

        # 在Yolos中，layernorm也在self-attention之后应用
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # 第二个残差连接在这里完成
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs
    # 初始化函数，接受一个 YolosConfig 对象作为配置参数
    def __init__(self, config: YolosConfig) -> None:
        # 调用父类的初始化方法
        super().__init__()
        # 将配置参数保存到对象属性中
        self.config = config
        # 创建一个由多个 YolosLayer 对象组成的列表，并保存到对象属性中
        self.layer = nn.ModuleList([YolosLayer(config) for _ in range(config.num_hidden_layers)])
        # 设置梯度检查点标志为 False
        self.gradient_checkpointing = False

        # 计算序列长度，用于中间位置嵌入的初始化
        seq_length = (
            1 + (config.image_size[0] * config.image_size[1] // config.patch_size**2) + config.num_detection_tokens
        )
        # 如果配置中指定使用中间位置嵌入，则创建相应的可训练参数
        self.mid_position_embeddings = (
            nn.Parameter(
                torch.zeros(
                    config.num_hidden_layers - 1,
                    1,
                    seq_length,
                    config.hidden_size,
                )
            )
            if config.use_mid_position_embeddings
            else None
        )

        # 如果配置中指定使用中间位置嵌入，则创建相应的插值器对象
        self.interpolation = InterpolateMidPositionEmbeddings(config) if config.use_mid_position_embeddings else None

    # 前向传播函数，接受多个输入参数，并返回一个联合类型的输出
    def forward(
        self,
        hidden_states: torch.Tensor,
        height,
        width,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        # 如果需要输出隐藏状态，则初始化空元组用于保存所有隐藏状态
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力权重，则初始化空元组用于保存所有注意力权重
        all_self_attentions = () if output_attentions else None

        # 如果配置中指定使用中间位置嵌入，则根据输入的高度和宽度进行插值计算
        if self.config.use_mid_position_embeddings:
            interpolated_mid_position_embeddings = self.interpolation(self.mid_position_embeddings, (height, width))

        # 遍历所有层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 如果启用梯度检查点并且处于训练阶段，则调用梯度检查点函数获取层的输出
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                # 否则，直接调用层模块进行前向传播计算
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            # 更新隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]

            # 如果配置中指定使用中间位置嵌入，并且不是最后一层，则将中间位置嵌入添加到当前隐藏状态中
            if self.config.use_mid_position_embeddings:
                if i < (self.config.num_hidden_layers - 1):
                    hidden_states = hidden_states + interpolated_mid_position_embeddings[i]

            # 如果需要输出注意力权重，则将当前层的注意力权重添加到 all_self_attentions 中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果需要输出隐藏状态，则将最终隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典形式的结果，则将结果打包成元组并返回
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 否则，返回一个 BaseModelOutput 对象，包含最终的隐藏状态、所有隐藏状态和所有注意力权重
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
@add_start_docstrings(
    "The bare YOLOS Model transformer outputting raw hidden-states without any specific head on top.",
    YOLOS_START_DOCSTRING,
)
class YolosModel(YolosPreTrainedModel):
    """
    YolosModel extends YolosPreTrainedModel to implement a transformer model without specific heads on top.

    Inherits from YolosPreTrainedModel and utilizes the provided YOLOS_START_DOCSTRING for detailed documentation.
    """
    # 初始化函数，接受一个YolosConfig类型的配置参数和一个布尔值参数add_pooling_layer，默认为True
    def __init__(self, config: YolosConfig, add_pooling_layer: bool = True):
        # 调用父类的初始化方法
        super().__init__(config)
        # 将传入的配置参数保存到实例变量中
        self.config = config

        # 初始化YolosEmbeddings对象，用于处理嵌入
        self.embeddings = YolosEmbeddings(config)
        # 初始化YolosEncoder对象，用于编码器
        self.encoder = YolosEncoder(config)

        # 初始化LayerNorm层，用于层归一化，设置epsilon参数为config中的layer_norm_eps
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # 如果add_pooling_layer为True，则初始化YolosPooler对象，用于池化
        # 否则将self.pooler设置为None
        self.pooler = YolosPooler(config) if add_pooling_layer else None

        # 调用post_init方法，用于初始化权重和应用最终处理
        self.post_init()

    # 返回输入嵌入层对象YolosPatchEmbeddings
    def get_input_embeddings(self) -> YolosPatchEmbeddings:
        return self.embeddings.patch_embeddings

    # 私有方法，用于修剪模型中的注意力头
    # heads_to_prune参数为一个字典，表示每个层需要修剪的注意力头列表
    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model.

        Args:
            heads_to_prune (`dict` of {layer_num: list of heads to prune in this layer}):
                See base class `PreTrainedModel`.
        """
        # 遍历heads_to_prune字典中的每一层和对应需要修剪的头部列表
        for layer, heads in heads_to_prune.items():
            # 调用self.encoder.layer[layer].attention.prune_heads方法进行修剪
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 前向传播函数，接受多个可选的输入参数，并返回模型输出
    @add_start_docstrings_to_model_forward(YOLOS_INPUTS_DOCSTRING)
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
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,

        # 接受像素值张量作为输入，可选参数，默认为None
        head_mask: Optional[torch.Tensor] = None,
        # 输出注意力权重的标志，可选参数，默认为None
        output_attentions: Optional[bool] = None,
        # 输出隐藏状态的标志，可选参数，默认为None
        output_hidden_states: Optional[bool] = None,
        # 返回字典类型的输出结果标志，可选参数，默认为None
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        # 设置输出注意力矩阵的选项，如果未指定则使用配置中的默认设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置输出隐藏状态的选项，如果未指定则使用配置中的默认设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置返回字典的选项，如果未指定则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果未提供像素值，则抛出数值错误异常
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 准备头部屏蔽（head mask）如果需要
        # head_mask 中的 1.0 表示我们保留该头部
        # attention_probs 的形状为 bsz x n_heads x N x N
        # 输入的 head_mask 的形状为 [num_heads] 或 [num_hidden_layers x num_heads]
        # 并且 head_mask 被转换为形状 [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # 将像素值嵌入到嵌入层中
        embedding_output = self.embeddings(pixel_values)

        # 将嵌入输出传入编码器（encoder）
        encoder_outputs = self.encoder(
            embedding_output,
            height=pixel_values.shape[-2],
            width=pixel_values.shape[-1],
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取编码器的序列输出
        sequence_output = encoder_outputs[0]
        # 应用层归一化到序列输出
        sequence_output = self.layernorm(sequence_output)
        # 如果存在汇聚器（pooler），则将序列输出传入汇聚器
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        # 如果不使用返回字典，则返回头部输出和编码器其他输出
        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        # 如果使用返回字典，则返回包含序列输出、汇聚器输出以及编码器其他输出的 BaseModelOutputWithPooling 对象
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
class YolosPooler(nn.Module):
    # 定义 YolosPooler 类，继承自 nn.Module
    def __init__(self, config: YolosConfig):
        # 初始化函数，接收一个 YolosConfig 类型的参数 config
        super().__init__()
        # 创建一个线性层，输入和输出维度均为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 激活函数为双曲正切函数
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # 前向传播函数
        # 我们通过简单地选择与第一个标记对应的隐藏状态来“池化”模型。
        first_token_tensor = hidden_states[:, 0]
        # 将第一个标记的隐藏状态输入到线性层中
        pooled_output = self.dense(first_token_tensor)
        # 应用激活函数到线性层的输出
        pooled_output = self.activation(pooled_output)
        # 返回池化后的输出
        return pooled_output


@add_start_docstrings(
    """
    YOLOS Model (consisting of a ViT encoder) with object detection heads on top, for tasks such as COCO detection.
    """,
    YOLOS_START_DOCSTRING,
)
class YolosForObjectDetection(YolosPreTrainedModel):
    # 定义 YolosForObjectDetection 类，继承自 YolosPreTrainedModel 类
    def __init__(self, config: YolosConfig):
        # 初始化函数，接收一个 YolosConfig 类型的参数 config
        super().__init__(config)

        # YOLOS (ViT) 编码器模型
        self.vit = YolosModel(config, add_pooling_layer=False)

        # 目标检测头部
        # 我们为“无对象”类别添加一个头部
        self.class_labels_classifier = YolosMLPPredictionHead(
            input_dim=config.hidden_size, hidden_dim=config.hidden_size, output_dim=config.num_labels + 1, num_layers=3
        )
        self.bbox_predictor = YolosMLPPredictionHead(
            input_dim=config.hidden_size, hidden_dim=config.hidden_size, output_dim=4, num_layers=3
        )

        # 初始化权重并应用最终处理
        self.post_init()

    # 参考自 https://github.com/facebookresearch/detr/blob/master/models/detr.py
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # 这是一个解决方案，使 torchscript 可以正常工作，因为 torchscript
        # 不支持具有非同构值的字典，例如同时包含张量和列表的字典。
        return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    @add_start_docstrings_to_model_forward(YOLOS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=YolosObjectDetectionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[List[Dict]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # YolosForObjectDetection 的前向传播函数
        pass  # 在此函数中执行具体的前向传播操作，但没有提供具体实现

# 从 transformers.models.detr.modeling_detr.dice_loss 复制而来
def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs (0 for the negative class and 1 for the positive
                 class).
    """
    # 对输入进行 sigmoid 激活
    inputs = inputs.sigmoid()
    # 将输入扁平化处理
    inputs = inputs.flatten(1)
    # 计算每个样本的分子部分：2 * (inputs * targets) 的按行求和
    numerator = 2 * (inputs * targets).sum(1)
    # 计算每个样本的分母部分：inputs 和 targets 的按最后一个维度求和
    denominator = inputs.sum(-1) + targets.sum(-1)
    # 计算损失值：1 减去 (numerator + 1) 除以 (denominator + 1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    # 返回所有样本损失值的平均值
    return loss.sum() / num_boxes
# Copied from transformers.models.detr.modeling_detr.sigmoid_focal_loss
def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (`torch.FloatTensor` of arbitrary shape):
            The predictions for each example.
        targets (`torch.FloatTensor` with the same shape as `inputs`)
            A tensor storing the binary classification label for each element in the `inputs` (0 for the negative class
            and 1 for the positive class).
        alpha (`float`, *optional*, defaults to `0.25`):
            Optional weighting factor in the range (0,1) to balance positive vs. negative examples.
        gamma (`int`, *optional*, defaults to `2`):
            Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.

    Returns:
        Loss tensor
    """
    # 将预测值通过sigmoid函数转换为概率
    prob = inputs.sigmoid()
    # 计算二元交叉熵损失，不进行缩减
    ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # 计算调节因子 p_t
    p_t = prob * targets + (1 - prob) * (1 - targets)
    # 计算焦点损失
    loss = ce_loss * ((1 - p_t) ** gamma)

    # 如果 alpha 大于等于 0，应用 alpha 调节损失
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # 返回损失的均值，并对所有盒子的损失进行求和后再除以盒子数量
    return loss.mean(1).sum() / num_boxes


# Copied from transformers.models.detr.modeling_detr.DetrLoss with Detr->Yolos
class YolosLoss(nn.Module):
    """
    This class computes the losses for YolosForObjectDetection/YolosForSegmentation. The process happens in two steps: 1)
    we compute hungarian assignment between ground truth boxes and the outputs of the model 2) we supervise each pair
    of matched ground-truth / prediction (supervise class and box).

    A note on the `num_classes` argument (copied from original repo in detr.py): "the naming of the `num_classes`
    parameter of the criterion is somewhat misleading. It indeed corresponds to `max_obj_id` + 1, where `max_obj_id` is
    the maximum id for a class in your dataset. For example, COCO has a `max_obj_id` of 90, so we pass `num_classes` to
    be 91. As another example, for a dataset that has a single class with `id` 1, you should pass `num_classes` to be 2
    (`max_obj_id` + 1). For more details on this, check the following discussion
    https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223"


    Args:
        matcher (`YolosHungarianMatcher`):
            Module able to compute a matching between targets and proposals.
        num_classes (`int`):
            Number of object categories, omitting the special no-object category.
        eos_coef (`float`):
            Relative classification weight applied to the no-object category.
        losses (`List[str]`):
            List of all the losses to be applied. See `get_loss` for a list of all available losses.
    """
    pass
    # 初始化函数，接收匹配器、类别数、EOS（End of Sequence）系数和损失函数列表作为参数
    def __init__(self, matcher, num_classes, eos_coef, losses):
        # 调用父类的初始化方法
        super().__init__()
        # 将参数赋值给对象的属性
        self.matcher = matcher
        self.num_classes = num_classes
        self.eos_coef = eos_coef
        self.losses = losses
        # 创建一个全为1的张量，长度为类别数+1，最后一个元素赋值为EOS系数
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        # 将这个权重张量注册为缓冲区，使其能够被保存到模型的状态中
        self.register_buffer("empty_weight", empty_weight)

    # 使用的是负对数似然损失（NLL），计算分类损失
    def loss_labels(self, outputs, targets, indices, num_boxes):
        """
        Classification loss (NLL) targets dicts must contain the key "class_labels" containing a tensor of dim
        [nb_target_boxes]
        """
        # 检查输出中是否存在"logits"键
        if "logits" not in outputs:
            raise KeyError("No logits were found in the outputs")
        # 获取模型输出的分类预测值
        source_logits = outputs["logits"]

        # 根据匹配索引重新排序目标类别标签，以匹配源 logits 的顺序
        idx = self._get_source_permutation_idx(indices)
        target_classes_o = torch.cat([t["class_labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            source_logits.shape[:2], self.num_classes, dtype=torch.int64, device=source_logits.device
        )
        target_classes[idx] = target_classes_o

        # 计算交叉熵损失，使用 self.empty_weight 作为类别权重
        loss_ce = nn.functional.cross_entropy(source_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}

        return losses

    # 用于计算基数误差（cardinality error），即预测的非空框的数量与目标之间的绝对误差
    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        Compute the cardinality error, i.e. the absolute error in the number of predicted non-empty boxes.

        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients.
        """
        # 获取模型输出的 logits
        logits = outputs["logits"]
        device = logits.device
        # 获取每个目标的类别标签张量长度
        target_lengths = torch.as_tensor([len(v["class_labels"]) for v in targets], device=device)
        # 计算预测的非空框数量，即 logits.argmax(-1) 不是最后一个类别的数量
        card_pred = (logits.argmax(-1) != logits.shape[-1] - 1).sum(1)
        # 使用 L1 损失函数计算基数误差
        card_err = nn.functional.l1_loss(card_pred.float(), target_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]. The target boxes
        are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        # 检查输出中是否存在预测的边界框
        if "pred_boxes" not in outputs:
            raise KeyError("No predicted boxes found in outputs")
        
        # 根据索引获取重新排列后的源边界框
        idx = self._get_source_permutation_idx(indices)
        source_boxes = outputs["pred_boxes"][idx]
        
        # 获取目标边界框，并且将它们拼接成一个张量
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # 计算 L1 回归损失
        loss_bbox = nn.functional.l1_loss(source_boxes, target_boxes, reduction="none")

        losses = {}
        # 将 L1 损失求和并进行归一化
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        # 计算 GIoU 损失
        loss_giou = 1 - torch.diag(
            generalized_box_iou(center_to_corners_format(source_boxes), center_to_corners_format(target_boxes))
        )
        # 将 GIoU 损失求和并进行归一化
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.

        Targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w].
        """
        # 检查输出中是否存在预测的掩码
        if "pred_masks" not in outputs:
            raise KeyError("No predicted masks found in outputs")

        # 获取源索引和目标索引，根据它们重新排列预测的掩码
        source_idx = self._get_source_permutation_idx(indices)
        target_idx = self._get_target_permutation_idx(indices)
        source_masks = outputs["pred_masks"]
        source_masks = source_masks[source_idx]
        
        # 获取目标掩码，并将它们解压缩成嵌套张量
        masks = [t["masks"] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(source_masks)
        target_masks = target_masks[target_idx]

        # 将预测掩码插值到目标大小
        source_masks = nn.functional.interpolate(
            source_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False
        )
        source_masks = source_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(source_masks.shape)
        
        losses = {
            # 计算 sigmoid focal 损失
            "loss_mask": sigmoid_focal_loss(source_masks, target_masks, num_boxes),
            # 计算 dice 损失
            "loss_dice": dice_loss(source_masks, target_masks, num_boxes),
        }
        return losses

    def _get_source_permutation_idx(self, indices):
        # 根据索引创建批次索引和源索引
        batch_idx = torch.cat([torch.full_like(source, i) for i, (source, _) in enumerate(indices)])
        source_idx = torch.cat([source for (source, _) in indices])
        return batch_idx, source_idx
    def _get_target_permutation_idx(self, indices):
        # 根据给定的索引重新排列目标
        # 创建一个张量，其中每个目标都被填充为其索引 i
        batch_idx = torch.cat([torch.full_like(target, i) for i, (_, target) in enumerate(indices)])
        # 创建一个张量，包含所有目标的索引
        target_idx = torch.cat([target for (_, target) in indices])
        return batch_idx, target_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        # 定义损失函数映射
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
        }
        if loss not in loss_map:
            # 如果损失函数不在映射中，则抛出错误
            raise ValueError(f"Loss {loss} not supported")
        # 调用相应的损失函数并返回结果
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        """
        This performs the loss computation.

        Args:
             outputs (`dict`, *optional*):
                Dictionary of tensors, see the output specification of the model for the format.
             targets (`List[dict]`, *optional*):
                List of dicts, such that `len(targets) == batch_size`. The expected keys in each dict depends on the
                losses applied, see each loss' doc.
        """
        # 剔除辅助输出，保留主要输出
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "auxiliary_outputs"}

        # 获取输出和目标之间的匹配关系
        indices = self.matcher(outputs_without_aux, targets)

        # 计算所有节点上的平均目标框数，用于归一化
        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        world_size = 1
        if is_accelerate_available():
            # 如果加速可用，并且部分状态不为空，则进行一些处理
            if PartialState._shared_state != {}:
                num_boxes = reduce(num_boxes)
                world_size = PartialState().num_processes
        # 限制目标框数除以世界大小后的值，并转为标量
        num_boxes = torch.clamp(num_boxes / world_size, min=1).item()

        # 计算所有请求的损失
        losses = {}
        for loss in self.losses:
            # 获取损失并更新到损失字典中
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # 如果存在辅助损失，则对每个中间层的输出进行相同的处理
        if "auxiliary_outputs" in outputs:
            for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):
                # 获取辅助输出和目标之间的匹配关系
                indices = self.matcher(auxiliary_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        # 忽略中间层的掩码损失计算，因为成本太高
                        continue
                    # 获取损失并更新到损失字典中，使用带有索引后缀的键
                    l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
# 从 transformers.models.detr.modeling_detr.DetrMLPPredictionHead 复制并改名为 YolosMLPPredictionHead
class YolosMLPPredictionHead(nn.Module):
    """
    简单的多层感知机（MLP，也称为 FFN），用于预测边界框相对于图像的归一化中心坐标、高度和宽度。

    从 https://github.com/facebookresearch/detr/blob/master/models/detr.py 复制而来
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        # 创建一个由多个线性层组成的模块列表
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        # 通过多个线性层和 ReLU 激活函数进行前向传播
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# 从 transformers.models.detr.modeling_detr.DetrHungarianMatcher 复制并改名为 YolosHungarianMatcher
class YolosHungarianMatcher(nn.Module):
    """
    这个类计算网络预测与目标之间的匹配。

    为了效率考虑，目标不包括无对象。因此，一般情况下预测数量比目标多。
    在这种情况下，我们对最佳预测进行 1 对 1 的匹配，而其余预测则不匹配（因此视为非对象）。

    Args:
        class_cost:
            匹配成本中分类误差的相对权重。
        bbox_cost:
            匹配成本中边界框坐标 L1 误差的相对权重。
        giou_cost:
            匹配成本中边界框 giou 损失的相对权重。
    """

    def __init__(self, class_cost: float = 1, bbox_cost: float = 1, giou_cost: float = 1):
        super().__init__()
        # 确保依赖库 scipy 被加载
        requires_backends(self, ["scipy"])

        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        # 如果所有匹配成本都为 0，则抛出错误
        if class_cost == 0 and bbox_cost == 0 and giou_cost == 0:
            raise ValueError("All costs of the Matcher can't be 0")

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Args:
            outputs (`dict`):
                A dictionary that contains at least these entries:
                * "logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                * "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates.
            targets (`List[dict]`):
                A list of targets (len(targets) = batch_size), where each target is a dict containing:
                * "class_labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of
                  ground-truth objects in the target) containing the class labels
                * "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates.

        Returns:
            `List[Tuple]`: A list of size `batch_size`, containing tuples of (index_i, index_j) where:
            - index_i is the indices of the selected predictions (in order)
            - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds: len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        # Extract batch size and number of queries from the logits tensor shape
        batch_size, num_queries = outputs["logits"].shape[:2]

        # Flatten logits and apply softmax to get the output probabilities
        out_prob = outputs["logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]

        # Flatten predicted boxes tensor
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Concatenate target class labels from all targets
        target_ids = torch.cat([v["class_labels"] for v in targets])

        # Concatenate target box coordinates from all targets
        target_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute classification cost using negative log likelihood approximation
        class_cost = -out_prob[:, target_ids]

        # Compute L1 cost between predicted boxes and target boxes
        bbox_cost = torch.cdist(out_bbox, target_bbox, p=1)

        # Compute generalized IoU (giou) cost between predicted and target boxes
        giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))

        # Combine costs into a final cost matrix using pre-defined coefficients (self.bbox_cost, self.class_cost, self.giou_cost)
        cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost

        # Reshape cost matrix to match batch size and number of queries
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        # Split cost matrix based on target sizes and apply linear sum assignment to find optimal assignment
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]

        # Return indices as a list of tuples, each tuple representing (index_i, index_j)
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
# Copied from transformers.models.detr.modeling_detr._upcast
def _upcast(t: Tensor) -> Tensor:
    # 如果输入张量是浮点型，则根据需要将其类型提升到更高的浮点类型，以防止数值溢出
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        # 如果输入张量是整型，则根据需要将其类型提升到更高的整型类型
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


# Copied from transformers.models.detr.modeling_detr.box_area
def box_area(boxes: Tensor) -> Tensor:
    """
    计算一组边界框的面积，这些边界框由其 (x1, y1, x2, y2) 坐标指定。

    Args:
        boxes (`torch.FloatTensor` of shape `(number_of_boxes, 4)`):
            待计算面积的边界框。这些边界框应该以 (x1, y1, x2, y2) 格式提供，其中 `0 <= x1 < x2` 且 `0 <= y1 < y2`。

    Returns:
        `torch.FloatTensor`: 包含每个边界框面积的张量。
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# Copied from transformers.models.detr.modeling_detr.box_iou
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    width_height = (right_bottom - left_top).clamp(min=0)  # [N,M,2]
    inter = width_height[:, :, 0] * width_height[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


# Copied from transformers.models.detr.modeling_detr.generalized_box_iou
def generalized_box_iou(boxes1, boxes2):
    """
    使用 https://giou.stanford.edu/ 中的广义 IoU 计算。边界框应该以 [x0, y0, x1, y1] (左上角和右下角) 格式提供。

    Returns:
        `torch.FloatTensor`: 一个形状为 [N, M] 的成对矩阵，其中 N = len(boxes1)，M = len(boxes2)
    """
    # 检查是否存在退化边界框，这会导致无穷大 / 无效结果，因此进行早期检查
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        raise ValueError(f"boxes1 必须以 [x0, y0, x1, y1] (左上角和右下角) 格式提供，但是提供了 {boxes1}")
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        raise ValueError(f"boxes2 必须以 [x0, y0, x1, y1] (左上角和右下角) 格式提供，但是提供了 {boxes2}")
    iou, union = box_iou(boxes1, boxes2)

    top_left = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    width_height = (bottom_right - top_left).clamp(min=0)  # [N,M,2]
    area = width_height[:, :, 0] * width_height[:, :, 1]

    return iou - (area - union) / area


# Copied from transformers.models.detr.modeling_detr._max_by_axis
def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes
# 定义一个名为 NestedTensor 的类，用于处理包含张量和可选遮罩的嵌套张量数据结构
class NestedTensor(object):
    # 初始化方法，接收张量列表和可选的遮罩张量作为参数
    def __init__(self, tensors, mask: Optional[Tensor]):
        # 将输入的张量列表赋值给实例变量 tensors
        self.tensors = tensors
        # 将输入的遮罩张量赋值给实例变量 mask
        self.mask = mask

    # 转换方法，将嵌套张量对象的张量数据移动到指定设备上
    def to(self, device):
        # 将嵌套张量中的张量数据转移到指定的设备上，并保存为新的张量对象
        cast_tensor = self.tensors.to(device)
        # 获取当前对象的遮罩张量
        mask = self.mask
        # 如果存在遮罩张量，则将其也转移到指定的设备上
        if mask is not None:
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        # 返回一个新的 NestedTensor 对象，其张量和遮罩均已转移到指定设备上
        return NestedTensor(cast_tensor, cast_mask)

    # 解构方法，返回嵌套张量对象中的张量和遮罩
    def decompose(self):
        return self.tensors, self.mask

    # 字符串表示方法，返回嵌套张量对象的张量的字符串表示
    def __repr__(self):
        return str(self.tensors)


# 从输入的张量列表中创建嵌套张量对象的函数，要求输入的张量必须是三维的
# 引自 transformers.models.detr.modeling_detr.nested_tensor_from_tensor_list
def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # 检查输入张量列表中第一个张量的维度是否为三维
    if tensor_list[0].ndim == 3:
        # 计算张量列表中所有张量的最大尺寸
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # 构建批次的形状，包括批次大小和每个张量的最大尺寸
        batch_shape = [len(tensor_list)] + max_size
        # 解包批次形状
        batch_size, num_channels, height, width = batch_shape
        # 获取第一个张量的数据类型和设备信息
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        # 创建全零张量，其形状与批次形状相同，指定数据类型和设备
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        # 创建全一遮罩张量，形状为批次大小、高度和宽度，数据类型为布尔型，指定设备
        mask = torch.ones((batch_size, height, width), dtype=torch.bool, device=device)
        # 将每个输入张量复制到对应位置的全零张量中，并更新遮罩张量
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        # 如果输入张量不是三维的，则抛出 ValueError 异常
        raise ValueError("Only 3-dimensional tensors are supported")
    # 返回一个新的 NestedTensor 对象，其中包含填充后的张量和遮罩张量
    return NestedTensor(tensor, mask)
```