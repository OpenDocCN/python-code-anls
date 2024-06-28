# `.\models\layoutlmv2\modeling_layoutlmv2.py`

```
# coding=utf-8
# 版权 2021 Microsoft Research The HuggingFace Inc. team. 保留所有权利。
#
# 根据 Apache 许可证 2.0 版本授权使用此文件；
# 您不得在未遵守许可证的情况下使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件
# 根据许可证“按原样”分发，
# 没有任何明示或暗示的担保或条件。
# 有关特定语言的详细信息，请参见许可证。
""" PyTorch LayoutLMv2 模型。"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入自定义的激活函数映射
from ...activations import ACT2FN
# 导入模型输出类
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
# 导入模型工具类
from ...modeling_utils import PreTrainedModel
# 导入 PyTorch 实用工具
from ...pytorch_utils import apply_chunking_to_forward
# 导入通用工具函数
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_detectron2_available,
    logging,
    replace_return_docstrings,
    requires_backends,
)
# 导入 LayoutLMv2 配置类
from .configuration_layoutlmv2 import LayoutLMv2Config

# 检查是否有 detectron2 可用（软依赖）
if is_detectron2_available():
    import detectron2
    from detectron2.modeling import META_ARCH_REGISTRY

# 获取 logger 实例
logger = logging.get_logger(__name__)

# 用于文档的检查点路径
_CHECKPOINT_FOR_DOC = "microsoft/layoutlmv2-base-uncased"
# 用于文档的配置类名称
_CONFIG_FOR_DOC = "LayoutLMv2Config"

# LayoutLMv2 预训练模型存档列表
LAYOUTLMV2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/layoutlmv2-base-uncased",
    "microsoft/layoutlmv2-large-uncased",
    # 查看所有 LayoutLMv2 模型：https://huggingface.co/models?filter=layoutlmv2
]

class LayoutLMv2Embeddings(nn.Module):
    """从词、位置和标记类型嵌入构建嵌入。"""
    # 初始化函数，接受一个配置对象 `config`
    def __init__(self, config):
        # 调用父类的初始化方法
        super(LayoutLMv2Embeddings, self).__init__()
        
        # 创建词嵌入层，使用 nn.Embedding
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        
        # 创建位置嵌入层，用于编码位置信息
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # 创建二维空间坐标位置嵌入层，用于编码 X 方向的位置信息
        self.x_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        
        # 创建二维空间坐标位置嵌入层，用于编码 Y 方向的位置信息
        self.y_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        
        # 创建二维空间坐标位置嵌入层，用于编码 H (高度) 的位置信息
        self.h_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)
        
        # 创建二维空间坐标位置嵌入层，用于编码 W (宽度) 的位置信息
        self.w_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)
        
        # 创建标记类型嵌入层，用于编码标记类型信息
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 创建 LayerNorm 层，用于标准化隐藏状态向量
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # 创建 Dropout 层，用于随机失活以防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 注册一个缓冲区 "position_ids"，用于保存位置编码的张量
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    # 计算空间位置嵌入的私有方法，接受一个 bbox 张量作为输入
    def _calc_spatial_position_embeddings(self, bbox):
        try:
            # 使用 X 位置嵌入层编码 bbox 的左边界位置信息
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            
            # 使用 Y 位置嵌入层编码 bbox 的上边界位置信息
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            
            # 使用 X 位置嵌入层编码 bbox 的右边界位置信息
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            
            # 使用 Y 位置嵌入层编码 bbox 的下边界位置信息
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            # 如果 bbox 的坐标值不在预期范围内（0-1000），抛出异常
            raise IndexError("The `bbox` coordinate values should be within 0-1000 range.") from e

        # 使用 H 位置嵌入层编码 bbox 的高度信息（下边界 - 上边界）
        h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        
        # 使用 W 位置嵌入层编码 bbox 的宽度信息（右边界 - 左边界）
        w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])

        # 将所有位置嵌入张量拼接在一起，形成空间位置嵌入张量
        spatial_position_embeddings = torch.cat(
            [
                left_position_embeddings,
                upper_position_embeddings,
                right_position_embeddings,
                lower_position_embeddings,
                h_position_embeddings,
                w_position_embeddings,
            ],
            dim=-1,
        )
        # 返回计算得到的空间位置嵌入张量
        return spatial_position_embeddings
        # LayoutLMv2SelfAttention 类的初始化函数
        def __init__(self, config):
            # 调用父类的初始化方法
            super().__init__()
            # 检查 hidden_size 是否能被 num_attention_heads 整除，如果不行且没有 embedding_size 属性，则引发 ValueError
            if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
                raise ValueError(
                    f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                    f"heads ({config.num_attention_heads})"
                )
            # 是否使用快速 QKV 模式
            self.fast_qkv = config.fast_qkv
            # 注意力头的数量
            self.num_attention_heads = config.num_attention_heads
            # 每个注意力头的大小
            self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
            # 所有注意力头的总大小
            self.all_head_size = self.num_attention_heads * self.attention_head_size

            # 是否具有相对注意力偏置
            self.has_relative_attention_bias = config.has_relative_attention_bias
            # 是否具有空间注意力偏置
            self.has_spatial_attention_bias = config.has_spatial_attention_bias

            # 如果使用快速 QKV 模式，则定义线性变换和偏置参数
            if config.fast_qkv:
                self.qkv_linear = nn.Linear(config.hidden_size, 3 * self.all_head_size, bias=False)
                self.q_bias = nn.Parameter(torch.zeros(1, 1, self.all_head_size))
                self.v_bias = nn.Parameter(torch.zeros(1, 1, self.all_head_size))
            else:
                # 否则分别定义查询、键、值的线性变换
                self.query = nn.Linear(config.hidden_size, self.all_head_size)
                self.key = nn.Linear(config.hidden_size, self.all_head_size)
                self.value = nn.Linear(config.hidden_size, self.all_head_size)

            # 定义 dropout 层
            self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # 将输入张量变形以便进行注意力计算
        def transpose_for_scores(self, x):
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
            x = x.view(*new_x_shape)
            return x.permute(0, 2, 1, 3)

        # 计算 QKV 的函数，根据是否使用快速 QKV 模式选择不同的处理方式
        def compute_qkv(self, hidden_states):
            if self.fast_qkv:
                # 使用快速 QKV 模式，进行线性变换并分割得到 Q、K、V
                qkv = self.qkv_linear(hidden_states)
                q, k, v = torch.chunk(qkv, 3, dim=-1)
                # 如果 Q 的维度与 q_bias 的维度相同，则直接加上偏置；否则进行维度调整后再加上偏置
                if q.ndimension() == self.q_bias.ndimension():
                    q = q + self.q_bias
                    v = v + self.v_bias
                else:
                    _sz = (1,) * (q.ndimension() - 1) + (-1,)
                    q = q + self.q_bias.view(*_sz)
                    v = v + self.v_bias.view(*_sz)
            else:
                # 否则分别计算 Q、K、V
                q = self.query(hidden_states)
                k = self.key(hidden_states)
                v = self.value(hidden_states)
            return q, k, v

        # 前向传播函数，接收隐藏状态、注意力掩码、头部掩码等输入，返回处理后的结果
        def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            output_attentions=False,
            rel_pos=None,
            rel_2d_pos=None,
        ):
            # 使用给定的隐藏状态计算查询、键和值
            q, k, v = self.compute_qkv(hidden_states)

            # (B, L, H*D) -> (B, H, L, D)
            # 将查询、键、值张量重新排列为注意力头的形状
            query_layer = self.transpose_for_scores(q)
            key_layer = self.transpose_for_scores(k)
            value_layer = self.transpose_for_scores(v)

            # 缩放查询张量，以确保稳定的注意力分数
            query_layer = query_layer / math.sqrt(self.attention_head_size)

            # 计算注意力分数
            # [BSZ, NAT, L, L]
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

            # 如果存在相对位置注意力偏置，添加到注意力分数中
            if self.has_relative_attention_bias:
                attention_scores += rel_pos

            # 如果存在空间注意力偏置，添加到注意力分数中
            if self.has_spatial_attention_bias:
                attention_scores += rel_2d_pos

            # 对注意力分数进行掩码处理，将不需要的位置置为极小值
            attention_scores = attention_scores.float().masked_fill_(
                attention_mask.to(torch.bool), torch.finfo(attention_scores.dtype).min
            )

            # 计算注意力权重，通过 softmax 函数归一化
            attention_probs = nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32).type_as(value_layer)

            # 使用 dropout 进行注意力权重的随机丢弃
            # 这实际上是丢弃整个待注意的标记，这在传统 Transformer 论文中是正常的做法
            attention_probs = self.dropout(attention_probs)

            # 如果指定了头部掩码，应用头部掩码
            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            # 计算上下文向量，通过加权值层乘以值层得到
            context_layer = torch.matmul(attention_probs, value_layer)

            # 调整上下文向量的维度，使其符合输出的所有头部大小
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)

            # 返回模型的输出，包括上下文向量和注意力权重（如果指定输出注意力权重）
            outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
            return outputs
# 定义 LayoutLMv2Attention 类，继承自 nn.Module
class LayoutLMv2Attention(nn.Module):
    # 初始化方法，接收配置参数 config
    def __init__(self, config):
        super().__init__()
        # 初始化 self 属性为 LayoutLMv2SelfAttention 类的实例，传入配置参数 config
        self.self = LayoutLMv2SelfAttention(config)
        # 初始化 output 属性为 LayoutLMv2SelfOutput 类的实例，传入配置参数 config
        self.output = LayoutLMv2SelfOutput(config)

    # 前向传播方法
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        # 调用 self 属性（LayoutLMv2SelfAttention 实例）的 forward 方法
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )
        # 调用 output 属性（LayoutLMv2SelfOutput 实例）的 forward 方法
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出注意力权重，将其加入 outputs 中
        outputs = (attention_output,) + self_outputs[1:]  # 如果要输出注意力权重，则添加到输出中
        return outputs


# 定义 LayoutLMv2SelfOutput 类，继承自 nn.Module
class LayoutLMv2SelfOutput(nn.Module):
    # 初始化方法，接收配置参数 config
    def __init__(self, config):
        super().__init__()
        # 初始化 dense 属性为 nn.Linear 类的实例，实现线性变换，输入输出维度为 hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 初始化 LayerNorm 属性为 nn.LayerNorm 类的实例，实现层归一化，输入维度为 hidden_size
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化 dropout 属性为 nn.Dropout 类的实例，实现随机失活，丢弃概率为 hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播方法
    def forward(self, hidden_states, input_tensor):
        # 经过全连接层 dense，实现线性变换
        hidden_states = self.dense(hidden_states)
        # 经过 dropout 层，实现随机失活
        hidden_states = self.dropout(hidden_states)
        # 输入 hidden_states 与 input_tensor 的残差连接，经过 LayerNorm 层，实现层归一化
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# 从 transformers.models.bert.modeling_bert.BertIntermediate 复制过来，替换 Bert 为 LayoutLMv2
class LayoutLMv2Intermediate(nn.Module):
    # 初始化方法，接收配置参数 config
    def __init__(self, config):
        super().__init__()
        # 初始化 dense 属性为 nn.Linear 类的实例，实现线性变换，输入维度为 hidden_size，输出维度为 intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果 hidden_act 是字符串类型，使用 ACT2FN 字典中对应的激活函数，否则直接使用配置中的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播方法，接收隐藏状态 hidden_states，返回经过线性变换和激活函数处理后的 hidden_states
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# 从 transformers.models.bert.modeling_bert.BertOutput 复制过来，替换 Bert 为 LayoutLMv2
class LayoutLMv2Output(nn.Module):
    # 初始化方法，接收配置参数 config
    def __init__(self, config):
        super().__init__()
        # 初始化 dense 属性为 nn.Linear 类的实例，实现线性变换，输入维度为 intermediate_size，输出维度为 hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 初始化 LayerNorm 属性为 nn.LayerNorm 类的实例，实现层归一化，输入维度为 hidden_size
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化 dropout 属性为 nn.Dropout 类的实例，实现随机失活，丢弃概率为 hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播方法，接收 hidden_states 和 input_tensor，返回经过线性变换、层归一化和随机失活处理后的 hidden_states
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# 定义 LayoutLMv2Layer 类，继承自 nn.Module，具体内容未完整给出，故未添加进一步的注释
class LayoutLMv2Layer(nn.Module):
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 设置前向传播中的块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度维度设为1（通常用于处理序列数据）
        self.seq_len_dim = 1
        # 创建 LayoutLMv2Attention 对象
        self.attention = LayoutLMv2Attention(config)
        # 创建 LayoutLMv2Intermediate 对象
        self.intermediate = LayoutLMv2Intermediate(config)
        # 创建 LayoutLMv2Output 对象
        self.output = LayoutLMv2Output(config)

    # 前向传播函数，处理输入的隐藏状态和其他可选参数
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        # 使用 self.attention 对象处理隐藏状态和注意力掩码等参数
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )
        # 获取注意力输出，通常是元组的第一个元素
        attention_output = self_attention_outputs[0]

        # 如果需要输出注意力权重，将注意力输出添加到 outputs 中
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # 将前向传播函数 apply_chunking_to_forward 应用于 attention_output
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        # 将处理后的层输出添加到 outputs 中
        outputs = (layer_output,) + outputs

        # 返回最终的输出结果
        return outputs

    # feed_forward_chunk 方法，处理注意力输出并返回层输出
    def feed_forward_chunk(self, attention_output):
        # 使用 self.intermediate 处理注意力输出
        intermediate_output = self.intermediate(attention_output)
        # 使用 self.output 处理 intermediate_output 和 attention_output，并返回层输出
        layer_output = self.output(intermediate_output, attention_output)
        # 返回层输出
        return layer_output
def relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    """
    Adapted from Mesh Tensorflow:
    https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
    Translate relative position to a bucket number for relative attention. The relative position is defined as
    memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
    position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for small
    absolute relative_position and larger buckets for larger absolute relative_positions. All relative positions
    >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket. This should
    allow for more graceful generalization to longer sequences than the model has been trained on.

    Args:
        relative_position: an int32 Tensor - 相对位置，表示从注意位置到被注意位置的距离（以标记为单位）
        bidirectional: a boolean - 是否双向关注
        num_buckets: an integer - 桶的数量，用于映射相对位置到桶号
        max_distance: an integer - 最大距离限制，超过此距离的相对位置映射到同一个桶

    Returns:
        a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        返回一个形状与relative_position相同的Tensor，包含范围在[0, num_buckets)内的int32值
    """

    ret = 0  # 初始化返回值

    if bidirectional:
        num_buckets //= 2  # 如果是双向的注意力，桶的数量减半
        ret += (relative_position > 0).long() * num_buckets  # 根据相对位置的正负决定加的桶数
        n = torch.abs(relative_position)  # 取相对位置的绝对值
    else:
        n = torch.max(-relative_position, torch.zeros_like(relative_position))  # 若为单向注意力，取负相对位置的最大值或0

    # 现在n的范围为[0, inf)

    # 将一半的桶用于精确增量的位置
    max_exact = num_buckets // 2
    is_small = n < max_exact  # 判断是否为小范围的相对位置

    # 另一半的桶用于对数级别更大的位置范围，直到max_distance
    val_if_large = max_exact + (
        torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
    ).to(torch.long)
    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))  # 确保不超过桶的上限

    ret += torch.where(is_small, n, val_if_large)  # 根据是否为小范围选择对应的值加到返回值上
    return ret  # 返回计算得到的桶号
    # 初始化函数，接收一个配置参数对象
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 保存配置对象到实例变量中
        self.config = config
        # 创建一个由多个 LayoutLMv2Layer 组成的模块列表
        self.layer = nn.ModuleList([LayoutLMv2Layer(config) for _ in range(config.num_hidden_layers)])

        # 检查是否有相对注意力偏置
        self.has_relative_attention_bias = config.has_relative_attention_bias
        # 检查是否有空间注意力偏置
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

        # 如果有相对注意力偏置，创建相对位置偏置线性层
        if self.has_relative_attention_bias:
            self.rel_pos_bins = config.rel_pos_bins
            self.max_rel_pos = config.max_rel_pos
            self.rel_pos_bias = nn.Linear(self.rel_pos_bins, config.num_attention_heads, bias=False)

        # 如果有空间注意力偏置，创建空间位置偏置线性层（x 和 y 方向各一个）
        if self.has_spatial_attention_bias:
            self.max_rel_2d_pos = config.max_rel_2d_pos
            self.rel_2d_pos_bins = config.rel_2d_pos_bins
            self.rel_pos_x_bias = nn.Linear(self.rel_2d_pos_bins, config.num_attention_heads, bias=False)
            self.rel_pos_y_bias = nn.Linear(self.rel_2d_pos_bins, config.num_attention_heads, bias=False)

        # 梯度检查点，默认为 False
        self.gradient_checkpointing = False

    # 计算一维位置嵌入
    def _calculate_1d_position_embeddings(self, position_ids):
        # 计算相对位置矩阵
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
        # 将相对位置矩阵映射到桶中，并返回相对位置嵌入
        rel_pos = relative_position_bucket(
            rel_pos_mat,
            num_buckets=self.rel_pos_bins,
            max_distance=self.max_rel_pos,
        )
        # 使用相对位置偏置线性层进行映射和转置操作
        rel_pos = self.rel_pos_bias.weight.t()[rel_pos].permute(0, 3, 1, 2)
        # 保证数据连续性，并返回结果
        rel_pos = rel_pos.contiguous()
        return rel_pos

    # 计算二维位置嵌入
    def _calculate_2d_position_embeddings(self, bbox):
        # 提取边界框的 x 和 y 坐标
        position_coord_x = bbox[:, :, 0]
        position_coord_y = bbox[:, :, 3]
        # 计算 x 和 y 方向上的相对位置矩阵
        rel_pos_x_2d_mat = position_coord_x.unsqueeze(-2) - position_coord_x.unsqueeze(-1)
        rel_pos_y_2d_mat = position_coord_y.unsqueeze(-2) - position_coord_y.unsqueeze(-1)
        # 将二维相对位置矩阵映射到桶中，并返回二维相对位置嵌入（x 和 y 方向分别处理）
        rel_pos_x = relative_position_bucket(
            rel_pos_x_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        rel_pos_y = relative_position_bucket(
            rel_pos_y_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        # 使用相对位置偏置线性层进行映射和转置操作（x 和 y 方向分别处理）
        rel_pos_x = self.rel_pos_x_bias.weight.t()[rel_pos_x].permute(0, 3, 1, 2)
        rel_pos_y = self.rel_pos_y_bias.weight.t()[rel_pos_y].permute(0, 3, 1, 2)
        # 保证数据连续性，并将 x 和 y 方向上的嵌入相加作为最终的二维相对位置嵌入
        rel_pos_x = rel_pos_x.contiguous()
        rel_pos_y = rel_pos_y.contiguous()
        rel_2d_pos = rel_pos_x + rel_pos_y
        return rel_2d_pos

    # 前向传播函数，处理模型的输入并返回输出
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        bbox=None,
        position_ids=None,
    ):
        ):
            # 如果输出隐藏状态被设置为真，则初始化一个空元组用于存储所有隐藏状态
            all_hidden_states = () if output_hidden_states else None
            # 如果输出注意力权重被设置为真，则初始化一个空元组用于存储所有自注意力权重
            all_self_attentions = () if output_attentions else None

            # 如果模型支持相对位置注意力偏置，则计算一维位置嵌入
            rel_pos = self._calculate_1d_position_embeddings(position_ids) if self.has_relative_attention_bias else None
            # 如果模型支持空间注意力偏置，则计算二维位置嵌入
            rel_2d_pos = self._calculate_2d_position_embeddings(bbox) if self.has_spatial_attention_bias else None

            # 遍历模型的每一个层，并进行相应操作
            for i, layer_module in enumerate(self.layer):
                # 如果需要输出隐藏状态，则将当前隐藏状态添加到所有隐藏状态元组中
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                # 如果有头部掩码，则从给定的掩码列表中选择当前层的头部掩码
                layer_head_mask = head_mask[i] if head_mask is not None else None

                # 如果启用了梯度检查点且处于训练模式下，则使用梯度检查点函数进行前向传播
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        layer_module.__call__,
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        output_attentions,
                        rel_pos=rel_pos,
                        rel_2d_pos=rel_2d_pos,
                    )
                else:
                    # 否则，直接调用当前层模块进行前向传播
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        output_attentions,
                        rel_pos=rel_pos,
                        rel_2d_pos=rel_2d_pos,
                    )

                # 更新隐藏状态为当前层的输出的第一个元素（即隐藏状态）
                hidden_states = layer_outputs[0]
                # 如果需要输出注意力权重，则将当前层的注意力权重添加到所有自注意力权重元组中
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)

            # 如果需要输出隐藏状态，则将最终的隐藏状态添加到所有隐藏状态元组中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果不需要以字典形式返回结果，则按顺序返回隐藏状态、所有隐藏状态、所有自注意力权重
            if not return_dict:
                return tuple(
                    v
                    for v in [
                        hidden_states,
                        all_hidden_states,
                        all_self_attentions,
                    ]
                    if v is not None
                )
            # 否则，以 BaseModelOutput 类的形式返回结果，包括最终隐藏状态、所有隐藏状态和所有自注意力权重
            return BaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
            )
class LayoutLMv2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 使用 LayoutLMv2Config 类作为配置类
    config_class = LayoutLMv2Config
    # 预训练模型存档映射列表
    pretrained_model_archive_map = LAYOUTLMV2_PRETRAINED_MODEL_ARCHIVE_LIST
    # 基础模型前缀名称
    base_model_prefix = "layoutlmv2"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # 对线性层的权重进行初始化，使用正态分布，标准差为配置文件中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                # 如果存在偏置项，则将其初始化为零
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 对嵌入层的权重进行初始化，使用正态分布，标准差为配置文件中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                # 如果指定了填充索引，则将填充索引位置的权重初始化为零
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 对 LayerNorm 层的偏置项初始化为零，权重初始化为 1.0
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


def my_convert_sync_batchnorm(module, process_group=None):
    # 与 `nn.modules.SyncBatchNorm.convert_sync_batchnorm` 相同，但允许从 `detectron2.layers.FrozenBatchNorm2d` 转换
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        # 将普通 BatchNorm 转换为 SyncBatchNorm
        return nn.modules.SyncBatchNorm.convert_sync_batchnorm(module, process_group)
    module_output = module
    if isinstance(module, detectron2.layers.FrozenBatchNorm2d):
        # 如果是 FrozenBatchNorm2d，则创建对应的 SyncBatchNorm
        module_output = torch.nn.SyncBatchNorm(
            num_features=module.num_features,
            eps=module.eps,
            affine=True,
            track_running_stats=True,
            process_group=process_group,
        )
        # 设置权重和偏置项
        module_output.weight = torch.nn.Parameter(module.weight)
        module_output.bias = torch.nn.Parameter(module.bias)
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = torch.tensor(0, dtype=torch.long, device=module.running_mean.device)
    for name, child in module.named_children():
        # 递归调用，对子模块进行转换
        module_output.add_module(name, my_convert_sync_batchnorm(child, process_group))
    del module
    return module_output


class LayoutLMv2VisualBackbone(nn.Module):
    # 这里是 LayoutLMv2 的视觉骨干网络定义的开始
    # 初始化函数，接受一个配置参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 获取检测器的配置信息
        self.cfg = config.get_detectron2_config()
        
        # 获取模型的元架构（meta architecture）
        meta_arch = self.cfg.MODEL.META_ARCHITECTURE
        # 根据元架构从注册表中获取对应的模型，并使用配置初始化模型
        model = META_ARCH_REGISTRY.get(meta_arch)(self.cfg)
        
        # 断言模型的主干是 FPN（特征金字塔网络）
        assert isinstance(model.backbone, detectron2.modeling.backbone.FPN)
        # 将模型的主干赋值给当前对象的属性
        self.backbone = model.backbone
        
        # 断言像素均值和像素标准差的长度相等
        assert len(self.cfg.MODEL.PIXEL_MEAN) == len(self.cfg.MODEL.PIXEL_STD)
        num_channels = len(self.cfg.MODEL.PIXEL_MEAN)
        
        # 将像素均值作为缓冲区的一部分注册到当前对象
        self.register_buffer(
            "pixel_mean",
            torch.Tensor(self.cfg.MODEL.PIXEL_MEAN).view(num_channels, 1, 1),
            persistent=False,
        )
        
        # 将像素标准差作为缓冲区的一部分注册到当前对象
        self.register_buffer(
            "pixel_std", torch.Tensor(self.cfg.MODEL.PIXEL_STD).view(num_channels, 1, 1), persistent=False
        )
        
        # 设置输出特征的关键字为 "p2"
        self.out_feature_key = "p2"
        
        # 如果启用了确定性算法，则使用平均池化替代自适应平均池化
        if torch.are_deterministic_algorithms_enabled():
            logger.warning("using `AvgPool2d` instead of `AdaptiveAvgPool2d`")
            input_shape = (224, 224)
            backbone_stride = self.backbone.output_shape()[self.out_feature_key].stride
            
            # 根据配置计算池化层的输出大小，并创建平均池化层
            self.pool = nn.AvgPool2d(
                (
                    math.ceil(math.ceil(input_shape[0] / backbone_stride) / config.image_feature_pool_shape[0]),
                    math.ceil(math.ceil(input_shape[1] / backbone_stride) / config.image_feature_pool_shape[1]),
                )
            )
        else:
            # 否则使用自适应平均池化层根据配置的形状创建池化层
            self.pool = nn.AdaptiveAvgPool2d(config.image_feature_pool_shape[:2])
        
        # 如果配置的图像特征池化形状长度为2，则添加主干输出特征的通道数到配置中
        if len(config.image_feature_pool_shape) == 2:
            config.image_feature_pool_shape.append(self.backbone.output_shape()[self.out_feature_key].channels)
        
        # 断言主干输出特征的通道数与配置中的相匹配
        assert self.backbone.output_shape()[self.out_feature_key].channels == config.image_feature_pool_shape[2]

    # 前向传播函数，接收图像作为输入并返回处理后的特征
    def forward(self, images):
        # 如果输入是张量，则直接使用，否则获取其张量表示
        images_input = ((images if torch.is_tensor(images) else images.tensor) - self.pixel_mean) / self.pixel_std
        # 将输入图像特征传递给主干网络进行特征提取
        features = self.backbone(images_input)
        # 从提取的特征中选择指定的输出特征并进行池化，然后展平并转置以便后续处理
        features = self.pool(features[self.out_feature_key]).flatten(start_dim=2).transpose(1, 2).contiguous()
        # 返回处理后的特征
        return features
    # 同步批归一化操作的方法定义
    def synchronize_batch_norm(self):
        # 检查当前环境是否支持分布式训练，并且已经初始化，且进程的排名大于-1
        if not (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_rank() > -1
        ):
            # 如果不满足条件，抛出运行时错误
            raise RuntimeError("Make sure torch.distributed is set up properly.")

        # 获取当前进程的排名
        self_rank = torch.distributed.get_rank()
        # 获取当前节点的 GPU 数量
        node_size = torch.cuda.device_count()
        # 获取整个分布式环境中的进程总数
        world_size = torch.distributed.get_world_size()
        # 检查进程总数是否可以被节点数整除
        if not (world_size % node_size == 0):
            # 如果不能整除，抛出运行时错误
            raise RuntimeError("Make sure the number of processes can be divided by the number of nodes")

        # 计算每个节点的全局排名列表
        node_global_ranks = [list(range(i * node_size, (i + 1) * node_size)) for i in range(world_size // node_size)]
        # 创建用于同步批归一化的分组列表
        sync_bn_groups = [
            torch.distributed.new_group(ranks=node_global_ranks[i]) for i in range(world_size // node_size)
        ]
        # 计算当前进程所在节点的索引
        node_rank = self_rank // node_size

        # 调用自定义的同步批归一化函数，将模型的骨干网络同步到对应的分组中
        self.backbone = my_convert_sync_batchnorm(self.backbone, process_group=sync_bn_groups[node_rank])
LAYOUTLMV2_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`LayoutLMv2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

LAYOUTLMV2_INPUTS_DOCSTRING = r"""
"""


class LayoutLMv2Pooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


@add_start_docstrings(
    "The bare LayoutLMv2 Model transformer outputting raw hidden-states without any specific head on top.",
    LAYOUTLMV2_START_DOCSTRING,
)
class LayoutLMv2Model(LayoutLMv2PreTrainedModel):
    def __init__(self, config):
        requires_backends(self, "detectron2")  # 检查是否有 detectron2 后端支持
        super().__init__(config)
        self.config = config
        self.has_visual_segment_embedding = config.has_visual_segment_embedding
        self.embeddings = LayoutLMv2Embeddings(config)  # 初始化模型的嵌入层

        self.visual = LayoutLMv2VisualBackbone(config)  # 初始化视觉骨干网络
        self.visual_proj = nn.Linear(config.image_feature_pool_shape[-1], config.hidden_size)  # 图像特征投影层
        if self.has_visual_segment_embedding:
            self.visual_segment_embedding = nn.Parameter(nn.Embedding(1, config.hidden_size).weight[0])  # 可视化片段嵌入
        self.visual_LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # 视觉层归一化
        self.visual_dropout = nn.Dropout(config.hidden_dropout_prob)  # 视觉层dropout

        self.encoder = LayoutLMv2Encoder(config)  # 初始化编码器
        self.pooler = LayoutLMv2Pooler(config)  # 初始化池化器

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings  # 返回输入的嵌入层

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value  # 设置输入的嵌入层
    # 计算文本输入的嵌入向量
    def _calc_text_embeddings(self, input_ids, bbox, position_ids, token_type_ids, inputs_embeds=None):
        # 如果有传入 input_ids，则获取其形状
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            # 否则获取 inputs_embeds 的形状，排除最后一个维度
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列的长度
        seq_length = input_shape[1]

        # 如果 position_ids 为空，则创建一个从 0 到 seq_length-1 的序列作为 position_ids
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # 如果 token_type_ids 为空，则创建与 input_ids 相同形状的全零张量
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # 如果 inputs_embeds 为空，则使用 input_ids 从 word_embeddings 中获取嵌入向量
        if inputs_embeds is None:
            inputs_embeds = self.embeddings.word_embeddings(input_ids)
        
        # 获取位置嵌入向量和空间位置嵌入向量
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        spatial_position_embeddings = self.embeddings._calc_spatial_position_embeddings(bbox)
        
        # 获取 token_type_ids 对应的嵌入向量
        token_type_embeddings = self.embeddings.token_type_embeddings(token_type_ids)

        # 计算最终的输入嵌入向量，包括 word embeddings、位置 embeddings、空间位置 embeddings 和 token_type embeddings
        embeddings = inputs_embeds + position_embeddings + spatial_position_embeddings + token_type_embeddings
        
        # 对 embeddings 进行 LayerNorm 处理
        embeddings = self.embeddings.LayerNorm(embeddings)
        
        # 对 embeddings 进行 dropout 处理
        embeddings = self.embeddings.dropout(embeddings)
        
        # 返回计算得到的 embeddings
        return embeddings

    # 计算图像输入的嵌入向量
    def _calc_img_embeddings(self, image, bbox, position_ids):
        # 通过 visual 方法获取视觉特征，并通过 visual_proj 进行投影
        visual_embeddings = self.visual_proj(self.visual(image))
        
        # 获取位置嵌入向量和空间位置嵌入向量
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        spatial_position_embeddings = self.embeddings._calc_spatial_position_embeddings(bbox)
        
        # 将视觉特征 embeddings、位置 embeddings 和空间位置 embeddings 相加
        embeddings = visual_embeddings + position_embeddings + spatial_position_embeddings
        
        # 如果模型具有视觉分段嵌入，则将其加到 embeddings 中
        if self.has_visual_segment_embedding:
            embeddings += self.visual_segment_embedding
        
        # 对 embeddings 进行 visual_LayerNorm 处理
        embeddings = self.visual_LayerNorm(embeddings)
        
        # 对 embeddings 进行 visual_dropout 处理
        embeddings = self.visual_dropout(embeddings)
        
        # 返回计算得到的 embeddings
        return embeddings
    @add_start_docstrings_to_model_forward(LAYOUTLMV2_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    # 用于模型前向传播的函数定义，添加了输入文档字符串和返回值文档字符串的装饰器
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        image: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 如果指定了input_ids和inputs_embeds，则抛出异常，因为不能同时指定两者
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        # 如果指定了input_ids，则返回其大小
        elif input_ids is not None:
            return input_ids.size()
        # 如果指定了inputs_embeds，则返回其除最后一维之外的大小
        elif inputs_embeds is not None:
            return inputs_embeds.size()[:-1]
        else:
            # 如果既未指定input_ids也未指定inputs_embeds，则抛出异常，要求至少指定其中之一
            raise ValueError("You have to specify either input_ids or inputs_embeds")
@add_start_docstrings(
    """
    LayoutLMv2 Model with a sequence classification head on top (a linear layer on top of the concatenation of the
    final hidden state of the [CLS] token, average-pooled initial visual embeddings and average-pooled final visual
    embeddings, e.g. for document image classification tasks such as the
    [RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/) dataset.
    """,
    LAYOUTLMV2_START_DOCSTRING,
)
class LayoutLMv2ForSequenceClassification(LayoutLMv2PreTrainedModel):
    """
    LayoutLMv2模型，顶部带有序列分类头（一个线性层，位于[CLS] token的最终隐藏状态、平均池化的初始视觉嵌入和平均池化的最终视觉嵌入的连接处），
    例如用于文档图像分类任务，如[RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/)数据集。
    """

    def __init__(self, config):
        """
        初始化函数，配置LayoutLMv2序列分类模型。

        Args:
            config (LayoutLMv2Config): 模型配置对象

        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layoutlmv2 = LayoutLMv2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 3, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        获取输入嵌入层（word embeddings）。

        Returns:
            torch.nn.Embedding: LayoutLMv2模型的词嵌入层对象

        """
        return self.layoutlmv2.embeddings.word_embeddings

    @add_start_docstrings_to_model_forward(LAYOUTLMV2_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        image: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        前向传播函数，执行LayoutLMv2序列分类模型的前向计算。

        Args:
            input_ids (torch.LongTensor, optional): 输入token的ID张量，默认为None
            bbox (torch.LongTensor, optional): 边界框信息的张量，默认为None
            image (torch.FloatTensor, optional): 图像特征的张量，默认为None
            attention_mask (torch.FloatTensor, optional): 注意力掩码的张量，默认为None
            token_type_ids (torch.LongTensor, optional): token类型ID的张量，默认为None
            position_ids (torch.LongTensor, optional): 位置ID的张量，默认为None
            head_mask (torch.FloatTensor, optional): 头部掩码的张量，默认为None
            inputs_embeds (torch.FloatTensor, optional): 输入嵌入的张量，默认为None
            labels (torch.LongTensor, optional): 标签的张量，默认为None
            output_attentions (bool, optional): 是否输出注意力，默认为None
            output_hidden_states (bool, optional): 是否输出隐藏状态，默认为None
            return_dict (bool, optional): 是否返回字典格式的输出，默认为None

        Returns:
            SequenceClassifierOutput: 序列分类任务的输出对象

        """
        # 省略部分代码...

@add_start_docstrings(
    """
    LayoutLMv2 Model with a token classification head on top (a linear layer on top of the text part of the hidden
    states) e.g. for sequence labeling (information extraction) tasks such as
    [FUNSD](https://guillaumejaume.github.io/FUNSD/), [SROIE](https://rrc.cvc.uab.es/?ch=13),
    [CORD](https://github.com/clovaai/cord) and [Kleister-NDA](https://github.com/applicaai/kleister-nda).
    """,
    LAYOUTLMV2_START_DOCSTRING,
)
class LayoutLMv2ForTokenClassification(LayoutLMv2PreTrainedModel):
    """
    LayoutLMv2模型，顶部带有标记分类头（一个线性层，位于隐藏状态的文本部分的顶部），
    例如用于序列标记任务（信息提取），如FUNSD, SROIE, CORD和Kleister-NDA。
    """

    def __init__(self, config):
        """
        初始化函数，配置LayoutLMv2标记分类模型。

        Args:
            config (LayoutLMv2Config): 模型配置对象

        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layoutlmv2 = LayoutLMv2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        获取输入嵌入层（word embeddings）。

        Returns:
            torch.nn.Embedding: LayoutLMv2模型的词嵌入层对象

        """
        return self.layoutlmv2.embeddings.word_embeddings

    @add_start_docstrings_to_model_forward(LAYOUTLMV2_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 使用装饰器替换返回文档字符串，设置输出类型为 TokenClassifierOutput，配置类为 _CONFIG_FOR_DOC
    @replace_return_docstrings(output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    # 前向传播函数，接受多个输入参数和可选参数
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入 token IDs，类型为长整型张量，可选
        bbox: Optional[torch.LongTensor] = None,  # 边界框信息，类型为长整型张量，可选
        image: Optional[torch.FloatTensor] = None,  # 图像数据，类型为浮点数张量，可选
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力掩码，类型为浮点数张量，可选
        token_type_ids: Optional[torch.LongTensor] = None,  # token 类型 IDs，类型为长整型张量，可选
        position_ids: Optional[torch.LongTensor] = None,  # 位置 IDs，类型为长整型张量，可选
        head_mask: Optional[torch.FloatTensor] = None,  # 头部掩码，类型为浮点数张量，可选
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入嵌入，类型为浮点数张量，可选
        labels: Optional[torch.LongTensor] = None,  # 标签，类型为长整型张量，可选
        output_attentions: Optional[bool] = None,  # 是否输出注意力信息，可选布尔值
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选布尔值
        return_dict: Optional[bool] = None,  # 是否返回字典形式的输出，可选布尔值
"""
LayoutLMv2 Model with a span classification head on top for extractive question-answering tasks such as
[DocVQA](https://rrc.cvc.uab.es/?ch=17) (a linear layer on top of the text part of the hidden-states output to
compute `span start logits` and `span end logits`).
"""
# 带有用于提取式问答任务的跨度分类头部的 LayoutLMv2 模型，例如 [DocVQA](https://rrc.cvc.uab.es/?ch=17)。
# 这个模型在隐藏状态输出的文本部分上增加了线性层，用于计算 `span start logits` 和 `span end logits`。

# 引用 LayoutLMv2 的起始文档字符串
LAYOUTLMV2_START_DOCSTRING = """

class LayoutLMv2ForQuestionAnswering(LayoutLMv2PreTrainedModel):
    def __init__(self, config, has_visual_segment_embedding=True):
        # 调用 LayoutLMv2PreTrainedModel 的初始化方法
        super().__init__(config)
        # 设置模型需要输出的标签数
        self.num_labels = config.num_labels
        # 根据输入的配置，决定是否包含视觉段落嵌入
        config.has_visual_segment_embedding = has_visual_segment_embedding
        # 创建 LayoutLMv2Model 对象
        self.layoutlmv2 = LayoutLMv2Model(config)
        # 创建用于问答任务的线性输出层，输入大小为隐藏状态的大小，输出大小为标签数
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回 LayoutLMv2 模型中的词嵌入层
        return self.layoutlmv2.embeddings.word_embeddings

    # 引用 LAYOUTLMV2_INPUTS_DOCSTRING，添加到模型前向方法的文档字符串中
    @add_start_docstrings_to_model_forward(LAYOUTLMV2_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 替换模型前向方法的返回文档字符串，使用 QuestionAnsweringModelOutput 类型，引用 _CONFIG_FOR_DOC 配置类
    @replace_return_docstrings(output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        image: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
"""
```