# `.\models\layoutlmv2\modeling_layoutlmv2.py`

```
# 设置文件编码为 UTF-8
# 版权声明及许可协议
# 版权所有 2021 年微软研究院和 HuggingFace 公司。保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”提供的软件
# 没有任何担保或条件，无论是明示的还是默示的。
# 有关许可证的特定语言，请参阅许可证。
""" PyTorch LayoutLMv2 模型。"""

import math  # 导入数学模块
from typing import Optional, Tuple, Union  # 导入类型提示相关模块

import torch  # 导入 PyTorch
import torch.utils.checkpoint  # 导入 PyTorch 工具模块
from torch import nn  # 导入 PyTorch 中的神经网络模块
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss  # 导入损失函数

from ...activations import ACT2FN  # 导入激活函数
from ...modeling_outputs import (  # 导入模型输出
    BaseModelOutput,
    BaseModelOutputWithPooling,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel  # 导入预训练模型基类
from ...pytorch_utils import apply_chunking_to_forward  # 导入工具函数
from ...utils import (  # 导入通用工具函数
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_detectron2_available,
    logging,
    replace_return_docstrings,
    requires_backends,
)
from .configuration_layoutlmv2 import LayoutLMv2Config  # 导入配置文件

# 检查 Detectron2 是否可用
if is_detectron2_available():
    import detectron2  # 导入 Detectron2
    from detectron2.modeling import META_ARCH_REGISTRY  # 导入 Detectron2 的模型注册

logger = logging.get_logger(__name__)  # 获取日志记录器

_CHECKPOINT_FOR_DOC = "microsoft/layoutlmv2-base-uncased"  # 模型检查点
_CONFIG_FOR_DOC = "LayoutLMv2Config"  # 配置文件

# LayoutLMv2 可用的预训练模型列表
LAYOUTLMV2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/layoutlmv2-base-uncased",
    "microsoft/layoutlmv2-large-uncased",
    # 更多 LayoutLMv2 模型请访问：https://huggingface.co/models?filter=layoutlmv2
]


class LayoutLMv2Embeddings(nn.Module):
    """构建来自单词、位置和标记类型嵌入的嵌入。"""
```  
    # 初始化函数，接受配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super(LayoutLMv2Embeddings, self).__init__()
        # 初始化词嵌入层，词汇量为config.vocab_size，嵌入维度为config.hidden_size，设置填充索引为config.pad_token_id
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 初始化位置嵌入层，位置嵌入的最大位置数为config.max_position_embeddings，嵌入维度为config.hidden_size
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # 初始化 X 坐标位置嵌入层，X 坐标最大位置数为config.max_2d_position_embeddings，嵌入维度为config.coordinate_size
        self.x_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        # 初始化 Y 坐标位置嵌入层，Y 坐标最大位置数为config.max_2d_position_embeddings，嵌入维度为config.coordinate_size
        self.y_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        # 初始化高度位置嵌入层，高度最大位置数为config.max_2d_position_embeddings，嵌入维度为config.shape_size
        self.h_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)
        # 初始化宽度位置嵌入层，宽度最大位置数为config.max_2d_position_embeddings，嵌入维度为config.shape_size
        self.w_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)
        # 初始化令牌类型嵌入层，令牌类型数为config.type_vocab_size，嵌入维度为config.hidden_size
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 初始化 LayerNorm 层，输入维度为config.hidden_size，设置 epsilon 为config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化 dropout 层，设置 dropout 概率为config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 注册一个持久性缓冲区，用于存储位置索引，大小为(config.max_position_embeddings, )
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    # 计算空间位置嵌入的私有方法，接受边界框坐标bbox作为输入
    def _calc_spatial_position_embeddings(self, bbox):
        # 尝试获取左上角 X 坐标位置嵌入
        left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
        # 尝试获取左上角 Y 坐标位置嵌入
        upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
        # 尝试获取右下角 X 坐标位置嵌入
        right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
        # 尝试获取右下角 Y 坐标位置嵌入
        lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])

        # 获取边界框高度，计算高度位置嵌入
        h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        # 获取边界框宽度，计算宽度位置嵌入
        w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])

        # 将各种位置嵌入拼接起来作为空间位置嵌入
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
        # 返回空间位置嵌入
        return spatial_position_embeddings
# 定义 LayoutLMv2SelfAttention 类，继承自 nn.Module
class LayoutLMv2SelfAttention(nn.Module):
    # 初始化函数，接受参数 config
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 如果隐藏层大小不能被注意力头数整除，并且 config 没有 "embedding_size" 属性，抛出数值错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        # 初始化快速 qkv 标记，注意力头数，注意力头大小，全部头的大小
        self.fast_qkv = config.fast_qkv
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化是否有相对注意力偏置，是否有空间注意力偏置
        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

        # 根据是否使用快速 qkv，初始化 qkv 线性层或者查询、键、值线性层
        if config.fast_qkv:
            self.qkv_linear = nn.Linear(config.hidden_size, 3 * self.all_head_size, bias=False)
            self.q_bias = nn.Parameter(torch.zeros(1, 1, self.all_head_size))
            self.v_bias = nn.Parameter(torch.zeros(1, 1, self.all_head_size))
        else:
            self.query = nn.Linear(config.hidden_size, self.all_head_size)
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 初始化 dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    # 定义转置函数，用于转置输入张量的形状以便计算注意力分数
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 定义计算 qkv 函数，根据是否使用快速 qkv 返回查询、键、值张量
    def compute_qkv(self, hidden_states):
        if self.fast_qkv:
            qkv = self.qkv_linear(hidden_states)
            q, k, v = torch.chunk(qkv, 3, dim=-1)
            if q.ndimension() == self.q_bias.ndimension():
                q = q + self.q_bias
                v = v + self.v_bias
            else:
                _sz = (1,) * (q.ndimension() - 1) + (-1,)
                q = q + self.q_bias.view(*_sz)
                v = v + self.v_bias.view(*_sz)
        else:
            q = self.query(hidden_states)
            k = self.key(hidden_states)
            v = self.value(hidden_states)
        return q, k, v

    # 前向传播函数，接受输入张量 hidden_states 等参数，计算自注意力
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
        # 通过隐藏状态计算 Query、Key、Value，并分别赋值给 q, k, v
        q, k, v = self.compute_qkv(hidden_states)

        # 将 (B, L, H*D) 的张量转置为 (B, H, L, D)，作为查询、键、值的张量
        query_layer = self.transpose_for_scores(q)
        key_layer = self.transpose_for_scores(k)
        value_layer = self.transpose_for_scores(v)

        # 对查询张量进行缩放操作，用以调整注意力分布
        query_layer = query_layer / math.sqrt(self.attention_head_size)
        
        # 计算注意力分数矩阵，使用查询张量和键张量进行点积
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        
        # 如果存在相对位置注意力偏置，加上相对位置偏置张量
        if self.has_relative_attention_bias:
            attention_scores += rel_pos
        
        # 如果存在空间注意力偏置，加上空间位置偏置张量
        if self.has_spatial_attention_bias:
            attention_scores += rel_2d_pos
        
        # 对注意力分数矩阵进行掩码填充操作，将 attention_mask 转换为布尔型后进行填充
        attention_scores = attention_scores.float().masked_fill_(
            attention_mask.to(torch.bool), torch.finfo(attention_scores.dtype).min
        )
        
        # 对注意力分数矩阵进行 softmax 操作，得到 attention_probs
        attention_probs = nn.functional.softmax(attention_scores, dim=-1, dtype=torch.float32).type_as(value_layer)

        # 对 attention_probs 进行 Dropout 操作
        attention_probs = self.dropout(attention_probs)

        # 如果存在 head_mask，对 attention_probs 进行 mask 处理
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 将注意力概率矩阵与值张量相乘，得到上下文张量
        context_layer = torch.matmul(attention_probs, value_layer)
        
        # 调整上下文张量的形状，转换为 (B, L, H, D) 的形式
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # 返回上下文张量和注意力概率，如果需要输出注意力分布则返回；否则只返回上下文张量
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs
# LayoutLMv2Attention 类
class LayoutLMv2Attention(nn.Module):
    # 构造函数，初始化模型
    def __init__(self, config):
        super().__init__()
        # 创建 LayoutLMv2SelfAttention 实例
        self.self = LayoutLMv2SelfAttention(config)
        # 创建 LayoutLMv2SelfOutput 实例
        self.output = LayoutLMv2SelfOutput(config)

    # 前向传播函数
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        # 调用 LayoutLMv2SelfAttention 实例的 forward() 方法
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )
        # 调用 LayoutLMv2SelfOutput 实例的 forward() 方法
        attention_output = self.output(self_outputs[0], hidden_states)
        # 将 attention_output 和 self_outputs 的其它元素合并到 outputs 中
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        # 返回 outputs
        return outputs

# LayoutLMv2SelfOutput 类
class LayoutLMv2SelfOutput(nn.Module):
    # 构造函数，初始化模型
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，将 hidden_state 转化为 config.hidden_size 的维度
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个 LayerNorm 层，用于归一化数据
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个 dropout 层，用于随机丢弃数据防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数
    def forward(self, hidden_states, input_tensor):
        # 使用全连接层将 hidden_states 转化为 config.hidden_size 维度
        hidden_states = self.dense(hidden_states)
        # 使用 dropout 层进行随机丢弃
        hidden_states = self.dropout(hidden_states)
        # 使用 LayerNorm 进行归一化处理
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的结果
        return hidden_states

# LayoutLMv2Intermediate 类
class LayoutLMv2Intermediate(nn.Module):
    # 构造函数，初始化模型
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，将 hidden_state 转化为 config.intermediate_size 的维度
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果 hidden_act 是字符串，则调用 ACT2FN 字典获取对应的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播函数
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用全连接层将 hidden_states 转化为 config.intermediate_size 维度
        hidden_states = self.dense(hidden_states)
        # 使用 intermediate_act_fn 指定的激活函数处理 hidden_states
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回处理后的结果
        return hidden_states

# LayoutLMv2Output 类
class LayoutLMv2Output(nn.Module):
    # 构造函数，初始化模型
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，将 config.intermediate_size 转化为 config.hidden_size 的维度
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个 LayerNorm 层，用于归一化数据
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个 dropout 层，用于随机丢弃数据防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用全连接层将 hidden_states 转化为 config.hidden_size 维度
        hidden_states = self.dense(hidden_states)
        # 使用 dropout 层进行随机丢弃
        hidden_states = self.dropout(hidden_states)
        # 使用 LayerNorm 进行归一化处理
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的结果
        return hidden_states

# LayoutLMv2Layer 类
class LayoutLMv2Layer(nn.Module):
    # 初始化LayoutLMv2TransformerLayer类的实例，初始化部分参数并调用父类的初始化方法
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        # 创建LayoutLMv2Attention、LayoutLMv2Intermediate和LayoutLMv2Output实例
        self.attention = LayoutLMv2Attention(config)
        self.intermediate = LayoutLMv2Intermediate(config)
        self.output = LayoutLMv2Output(config)
    
    # 前向传播方法，接收隐藏状态、注意力掩码等参数
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        # 调用自注意力方法对隐藏状态进行处理
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )
        attention_output = self_attention_outputs[0]
    
        outputs = self_attention_outputs[1:]  # 如果输出注意力权重，则添加自注意力信息
    
        # 对注意力输出进行分块处理
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs
    
        return outputs
    
    # 实现feed_forward_chunk方法对注意力输出进行处理
    def feed_forward_chunk(self, attention_output):
        # 通过中间层处理注意力输出
        intermediate_output = self.intermediate(attention_output)
        # 输出通过输出层处理后得到最终层输出
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
# 将相对位置转换为相对注意力的桶编号
def relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    """
    Adapted from Mesh Tensorflow:
    https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
    将相对位置转换为相对注意力的桶编号。相对位置被定义为 memory_position - query_position，即从关注位置到被关注位置的距离（以令牌计）。如果 bidirectional=False，则正的相对位置无效。我们对较小的绝对相对位置使用较小的桶，对较大的绝对相对位置使用较大的桶。所有大于等于 max_distance 的相对位置映射到同一个桶。所有小于等于 -max_distance 的相对位置映射到同一个桶。这应该能够更好地适应比模型训练样本长度更长的序列。

    参数:
        relative_position: 一个 int32 张量
        bidirectional: 一个布尔值，表示注意力是否是双向的
        num_buckets: 一个整数，表示桶的数量
        max_distance: 一个整数，表示最大距离

    返回:
        一个形状与 relative_position 相同的张量，包含范围为 [0, num_buckets) 的 int32 值
    """

    # 初始化结果为 0
    ret = 0
    # 如果是双向注意力，则减少桶的数量
    if bidirectional:
        num_buckets //= 2
        # 将 relative_position 大于 0 的位置映射到对应桶
        ret += (relative_position > 0).long() * num_buckets
        # 计算绝对值
        n = torch.abs(relative_position)
    else:
        # 计算负的相对位置的最大值
        n = torch.max(-relative_position, torch.zeros_like(relative_position))
    # 现在 n 的范围为 [0, inf)

    # 一半的桶用于精确增量的位置
    max_exact = num_buckets // 2
    # 判断是否是较小的相对位置
    is_small = n < max_exact

    # 另一半的桶用于按对数增加的相对位置，最大距离为 max_distance
    val_if_large = max_exact + (
        torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
    ).to(torch.long)
    # 将 val_if_large 中大于 num_buckets-1 的位置映射到 num_buckets-1
    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

    # 将结果加到 ret 中，根据 is_small 判断要加 n 或者 val_if_large
    ret += torch.where(is_small, n, val_if_large)
    return ret


class LayoutLMv2Encoder(nn.Module):
    # 省略不表
    # 初始化方法
    def __init__(self, config):
        # 调用父类初始化方法
        super().__init__()
        # 保存配置信息
        self.config = config
        # 创建指定数量的 LayoutLMv2Layer 模块，存储在 ModuleList 中
        self.layer = nn.ModuleList([LayoutLMv2Layer(config) for _ in range(config.num_hidden_layers)])

        # 判断是否有相对注意力偏置
        self.has_relative_attention_bias = config.has_relative_attention_bias
        # 判断是否有空间注意力偏置
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

        # 如果有相对注意力偏置
        if self.has_relative_attention_bias:
            # 保存配置中的相对位置桶数和最大相对位置
            self.rel_pos_bins = config.rel_pos_bins
            self.max_rel_pos = config.max_rel_pos
            # 创建相对注意力偏置的线性层
            self.rel_pos_bias = nn.Linear(self.rel_pos_bins, config.num_attention_heads, bias=False)

        # 如果有空间注意力偏置
        if self.has_spatial_attention_bias:
            # 保存配置中的最大 2D 相对位置和 2D 相对位置桶数
            self.max_rel_2d_pos = config.max_rel_2d_pos
            self.rel_2d_pos_bins = config.rel_2d_pos_bins
            # 创建 x 方向的相对位置偏置的线性层
            self.rel_pos_x_bias = nn.Linear(self.rel_2d_pos_bins, config.num_attention_heads, bias=False)
            # 创建 y 方向的相对位置偏置的线性层
            self.rel_pos_y_bias = nn.Linear(self.rel_2d_pos_bins, config.num_attention_heads, bias=False)

        # 初始化是否使用渐变检查点
        self.gradient_checkpointing = False

    # 计算一维位置嵌入
    def _calculate_1d_position_embeddings(self, position_ids):
        # 计算相对位置矩阵
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
        # 使用相对位置桶函数处理相对位置矩阵
        rel_pos = relative_position_bucket(
            rel_pos_mat,
            num_buckets=self.rel_pos_bins,
            max_distance=self.max_rel_pos,
        )
        # 计算相对位置偏置
        rel_pos = self.rel_pos_bias.weight.t()[rel_pos].permute(0, 3, 1, 2)
        rel_pos = rel_pos.contiguous()
        return rel_pos

    # 计算二维位置嵌入
    def _calculate_2d_position_embeddings(self, bbox):
        # 提取边界框的 x 和 y 坐标
        position_coord_x = bbox[:, :, 0]
        position_coord_y = bbox[:, :, 3]
        # 计算 x 方向和 y 方向上的相对位置矩阵
        rel_pos_x_2d_mat = position_coord_x.unsqueeze(-2) - position_coord_x.unsqueeze(-1)
        rel_pos_y_2d_mat = position_coord_y.unsqueeze(-2) - position_coord_y.unsqueeze(-1)
        # 使用相对位置桶函数处理 x 方向上的相对位置矩阵
        rel_pos_x = relative_position_bucket(
            rel_pos_x_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        # 使用相对位置桶函数处理 y 方向上的相对位置矩阵
        rel_pos_y = relative_position_bucket(
            rel_pos_y_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        # 计算 x 方向和 y 方向上的相对位置偏置
        rel_pos_x = self.rel_pos_x_bias.weight.t()[rel_pos_x].permute(0, 3, 1, 2)
        rel_pos_y = self.rel_pos_y_bias.weight.t()[rel_pos_y].permute(0, 3, 1, 2)
        rel_pos_x = rel_pos_x.contiguous()
        rel_pos_y = rel_pos_y.contiguous()
        # 组合 x 方向和 y 方向上的相对位置偏置
        rel_2d_pos = rel_pos_x + rel_pos_y
        return rel_2d_pos

    # 前向传播方法
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
        # 如果不要输出隐藏状态，则初始化空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果不要输出注意力权重，则初始化空元组
        all_self_attentions = () if output_attentions else None

        # 如果模型有相对位置注意力偏置，则计算1D位置嵌入
        rel_pos = self._calculate_1d_position_embeddings(position_ids) if self.has_relative_attention_bias else None
        # 如果模型有空间注意力偏置，则计算2D位置嵌入
        rel_2d_pos = self._calculate_2d_position_embeddings(bbox) if self.has_spatial_attention_bias else None

        # 遍历每个层进行前向计算
        for i, layer_module in enumerate(self.layer):
            # 如果要输出隐藏状态，则将当前隐藏状态添加到元组中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 如果使用梯度检查点并且在训练模式下
            if self.gradient_checkpointing and self.training:
                # 使用梯度检查点函数进行前向传播计算
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
                # 否则直接调用当前层的前向传播函数
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                    rel_pos=rel_pos,
                    rel_2d_pos=rel_2d_pos,
                )

            # 更新当前隐藏状态
            hidden_states = layer_outputs[0]
            # 如果要输出注意力权重，则将当前层的注意力权重添加到元组中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果要输出隐藏状态，则将当前隐藏状态添加到元组中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典，则以元组形式返回隐藏状态、所有隐藏状态和所有注意力权重
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
        # 否则以类的实例形式返回结果
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

    # 指定配置类
    config_class = LayoutLMv2Config
    # 预训练模型存档映射表
    pretrained_model_archive_map = LAYOUTLMV2_PRETRAINED_MODEL_ARCHIVE_LIST
    # 基础模型前缀
    base_model_prefix = "layoutlmv2"

    def _init_weights(self, module):
        """Initialize the weights"""
        # 初始化权重
        if isinstance(module, nn.Linear):
            # 对线性层的权重进行初始化，使用正态分布
            # 与 TF 版本略有不同，TF 版本使用截断正态分布进行初始化
            # 参考：https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置项，则将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 对嵌入层的权重进行初始化，使用正态分布
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在填充索引，则将对应索引的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 对层归一化层的权重进行初始化，偏置初始化为零，权重初始化为全1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


def my_convert_sync_batchnorm(module, process_group=None):
    # 与 `nn.modules.SyncBatchNorm.convert_sync_batchnorm` 相同，但允许从 `detectron2.layers.FrozenBatchNorm2d` 转换
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        return nn.modules.SyncBatchNorm.convert_sync_batchnorm(module, process_group)
    module_output = module
    if isinstance(module, detectron2.layers.FrozenBatchNorm2d):
        # 如果输入模块是 `detectron2.layers.FrozenBatchNorm2d`，则转换为 `torch.nn.SyncBatchNorm`
        module_output = torch.nn.SyncBatchNorm(
            num_features=module.num_features,
            eps=module.eps,
            affine=True,
            track_running_stats=True,
            process_group=process_group,
        )
        # 复制权重和偏置
        module_output.weight = torch.nn.Parameter(module.weight)
        module_output.bias = torch.nn.Parameter(module.bias)
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = torch.tensor(0, dtype=torch.long, device=module.running_mean.device)
    for name, child in module.named_children():
        # 递归调用以处理子模块
        module_output.add_module(name, my_convert_sync_batchnorm(child, process_group))
    # 删除原始模块
    del module
    return module_output


class LayoutLMv2VisualBackbone(nn.Module):
    # 初始化对象，传入配置参数
    def __init__(self, config):
        # 调用父类初始化方法
        super().__init__()
        # 获取配置文件中的检测器配置
        self.cfg = config.get_detectron2_config()
        # 获取模型的元架构
        meta_arch = self.cfg.MODEL.META_ARCHITECTURE
        # 根据元架构选择模型
        model = META_ARCH_REGISTRY.get(meta_arch)(self.cfg)
        assert isinstance(model.backbone, detectron2.modeling.backbone.FPN)
        # 设置对象的背骨结构
        self.backbone = model.backbone

        # 确保像素均值和像素标准差的长度相同
        assert len(self.cfg.MODEL.PIXEL_MEAN) == len(self.cfg.MODEL.PIXEL_STD)
        num_channels = len(self.cfg.MODEL.PIXEL_MEAN)
        # 向对象注册像素均值的缓冲区
        self.register_buffer(
            "pixel_mean",
            torch.Tensor(self.cfg.MODEL.PIXEL_MEAN).view(num_channels, 1, 1),
            persistent=False,
        )
        # 向对象注册像素标准差的缓冲区
        self.register_buffer(
            "pixel_std", torch.Tensor(self.cfg.MODEL.PIXEL_STD).view(num_channels, 1, 1), persistent=False
        )
        # 设置输出特征的键值
        self.out_feature_key = "p2"
        # 如果启用了确定性算法，则使用平均池化替代自适应平均池化
        if torch.are_deterministic_algorithms_enabled():
            logger.warning("using `AvgPool2d` instead of `AdaptiveAvgPool2d`")
            input_shape = (224, 224)
            backbone_stride = self.backbone.output_shape()[self.out_feature_key].stride
            # 设置池化层为平均池化层
            self.pool = nn.AvgPool2d(
                (
                    math.ceil(math.ceil(input_shape[0] / backbone_stride) / config.image_feature_pool_shape[0]),
                    math.ceil(math.ceil(input_shape[1] / backbone_stride) / config.image_feature_pool_shape[1]),
                )
            )
        else:
            # 设置池化层为自适应平均池化层
            self.pool = nn.AdaptiveAvgPool2d(config.image_feature_pool_shape[:2])
        # 如果配置参数中的图像特征池化形状长度为2，将通道数添加到配置参数中
        if len(config.image_feature_pool_shape) == 2:
            config.image_feature_pool_shape.append(self.backbone.output_shape()[self.out_feature_key].channels)
        # 确保背骨结构输出的特征通道数与配置参数中的图像特征池化形状相符
        assert self.backbone.output_shape()[self.out_feature_key].channels == config.image_feature_pool_shape[2]

    # 前向传播函数
    def forward(self, images):
        # 将输入图像减去像素均值并除以像素标准差
        images_input = ((images if torch.is_tensor(images) else images.tensor) - self.pixel_mean) / self.pixel_std
        # 使用背骨结构处理图像输入，获取特征
        features = self.backbone(images_input)
        features = features[self.out_feature_key]
        # 对特征进行池化、拉平、转置操作
        features = self.pool(features).flatten(start_dim=2).transpose(1, 2).contiguous()
        return features
    # 同步批标准化操作
    def synchronize_batch_norm(self):
        # 检查是否分布式环境已设置好
        if not (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_rank() > -1
        ):
            raise RuntimeError("Make sure torch.distributed is set up properly.")
    
        # 获取当前进程的排名
        self_rank = torch.distributed.get_rank()
        # 获取节点上的 GPU 数量
        node_size = torch.cuda.device_count()
        # 获取分布式环境中的进程数量
        world_size = torch.distributed.get_world_size()
        # 检查进程数量是否可以整除节点数量
        if not (world_size % node_size == 0):
            raise RuntimeError("Make sure the number of processes can be divided by the number of nodes")
    
        # 根据节点数量构建节点全局排名列表
        node_global_ranks = [list(range(i * node_size, (i + 1) * node_size)) for i in range(world_size // node_size)]
        # 创建同步批标准化的进程组
        sync_bn_groups = [
            torch.distributed.new_group(ranks=node_global_ranks[i]) for i in range(world_size // node_size)
        ]
        # 获取当前进程所在节点的排名
        node_rank = self_rank // node_size
    
        # 对神经网络的主干模块进行同步批标准化转换
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
        # 初始化一个全连接层，用于池化操作
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 初始化一个激活函数，用于池化操作后的激活
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # 我们通过简单地取第一个标记对应的隐藏状态来对模型进行“池化”。
        first_token_tensor = hidden_states[:, 0]
        # 通过全连接层进行池化操作
        pooled_output = self.dense(first_token_tensor)
        # 对池化后的输出应用激活函数
        pooled_output = self.activation(pooled_output)
        # 返回池化后的输出
        return pooled_output


@add_start_docstrings(
    "The bare LayoutLMv2 Model transformer outputting raw hidden-states without any specific head on top.",
    LAYOUTLMV2_START_DOCSTRING,
)
class LayoutLMv2Model(LayoutLMv2PreTrainedModel):
    def __init__(self, config):
        # 检查是否需要后端支持
        requires_backends(self, "detectron2")
        super().__init__(config)
        self.config = config
        # 是否具有视觉分段嵌入
        self.has_visual_segment_embedding = config.has_visual_segment_embedding
        # 初始化嵌入层
        self.embeddings = LayoutLMv2Embeddings(config)

        # 初始化视觉模块
        self.visual = LayoutLMv2VisualBackbone(config)
        # 初始化视觉投影层
        self.visual_proj = nn.Linear(config.image_feature_pool_shape[-1], config.hidden_size)
        if self.has_visual_segment_embedding:
            # 如果有视觉分段嵌入，则初始化相应的参数
            self.visual_segment_embedding = nn.Parameter(nn.Embedding(1, config.hidden_size).weight[0])
        # 初始化视觉LayerNorm层
        self.visual_LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化视觉dropout层
        self.visual_dropout = nn.Dropout(config.hidden_dropout_prob)

        # 初始化编码器
        self.encoder = LayoutLMv2Encoder(config)
        # 初始化池化层
        self.pooler = LayoutLMv2Pooler(config)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 获取输入嵌入层
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        # 设置输入嵌入层
        self.embeddings.word_embeddings = value
    # 计算文本输入的嵌入
    def _calc_text_embeddings(self, input_ids, bbox, position_ids, token_type_ids, inputs_embeds=None):
        # 如果存在输入的 token IDs
        if input_ids is not None:
            input_shape = input_ids.size()
        # 如果不存在输入的 token IDs，使用嵌入的维度
        else:
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度
        seq_length = input_shape[1]

        # 如果位置 IDs 不存在，则创建一个序列长的位置 IDs 张量
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        # 如果 token 类型 IDs 不存在，则创建一个与 input_ids 大小相同的全零张量
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # 如果输入嵌入不存在，则使用词嵌入层的 word_embeddings 获取嵌入
        if inputs_embeds is None:
            inputs_embeds = self.embeddings.word_embeddings(input_ids)
        # 获取位置嵌入、空间位置嵌入和 token 类型嵌入
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        spatial_position_embeddings = self.embeddings._calc_spatial_position_embeddings(bbox)
        token_type_embeddings = self.embeddings.token_type_embeddings(token_type_ids)

        # 计算嵌入并添加 LayerNorm 和 dropout
        embeddings = inputs_embeds + position_embeddings + spatial_position_embeddings + token_type_embeddings
        embeddings = self.embeddings.LayerNorm(embeddings)
        embeddings = self.embeddings.dropout(embeddings)
        # 返回计算后的嵌入
        return embeddings

    # 计算图像输入的嵌入
    def _calc_img_embeddings(self, image, bbox, position_ids):
        # 获取视觉输入的嵌入并进行视觉映射
        visual_embeddings = self.visual_proj(self.visual(image))
        # 获取位置嵌入、空间位置嵌入
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        spatial_position_embeddings = self.embeddings._calc_spatial_position_embeddings(bbox)
        # 将嵌入相加
        embeddings = visual_embeddings + position_embeddings + spatial_position_embeddings
        # 如果存在视觉分段嵌入，则加上
        if self.has_visual_segment_embedding:
            embeddings += self.visual_segment_embedding
        # 添加 LayerNorm 和 dropout
        embeddings = self.visual_LayerNorm(embeddings)
        embeddings = self.visual_dropout(embeddings)
        # 返回计算后的嵌入
        return embeddings
    # 计算视觉边界框（visual bounding box）的坐标
    def _calc_visual_bbox(self, image_feature_pool_shape, bbox, device, final_shape):
        # 计算 x 轴方向的视觉边界框坐标
        visual_bbox_x = torch.div(
            torch.arange(
                0,
                1000 * (image_feature_pool_shape[1] + 1),
                1000,
                device=device,
                dtype=bbox.dtype,
            ),
            self.config.image_feature_pool_shape[1],
            rounding_mode="floor",
        )
        # 计算 y 轴方向的视觉边界框坐标
        visual_bbox_y = torch.div(
            torch.arange(
                0,
                1000 * (self.config.image_feature_pool_shape[0] + 1),
                1000,
                device=device,
                dtype=bbox.dtype,
            ),
            self.config.image_feature_pool_shape[0],
            rounding_mode="floor",
        )
        # 根据 x 和 y 轴方向的坐标，构建四个边界框的坐标
        visual_bbox = torch.stack(
            [
                visual_bbox_x[:-1].repeat(image_feature_pool_shape[0], 1),
                visual_bbox_y[:-1].repeat(image_feature_pool_shape[1], 1).transpose(0, 1),
                visual_bbox_x[1:].repeat(image_feature_pool_shape[0], 1),
                visual_bbox_y[1:].repeat(image_feature_pool_shape[1], 1).transpose(0, 1),
            ],
            dim=-1,
        ).view(-1, bbox.size(-1))

        # 将计算的边界框坐标复制到最终形状
        visual_bbox = visual_bbox.repeat(final_shape[0], 1, 1)

        return visual_bbox

    # 获取输入张量的形状
    def _get_input_shape(self, input_ids=None, inputs_embeds=None):
        # 检查输入参数是否正确
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            return input_ids.size()
        elif inputs_embeds is not None:
            return inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

    # 模型前向传播函数
    @add_start_docstrings_to_model_forward(LAYOUTLMV2_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
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
# 使用add_start_docstrings装饰器添加模型说明文档
@add_start_docstrings(
    """
    LayoutLMv2 Model with a sequence classification head on top (a linear layer on top of the concatenation of the
    final hidden state of the [CLS] token, average-pooled initial visual embeddings and average-pooled final visual
    embeddings, e.g. for document image classification tasks such as the
    [RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/) dataset.
    """,
    LAYOUTLMV2_START_DOCSTRING,
)
# 定义LayoutLMv2ForSequenceClassification类，继承自LayoutLMv2PreTrainedModel类
class LayoutLMv2ForSequenceClassification(LayoutLMv2PreTrainedModel):
    # 初始化函数，接受config参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)
        # 将config中的num_labels赋值给self.num_labels
        self.num_labels = config.num_labels
        # 创建LayoutLMv2Model对象，并赋值给self.layoutlmv2
        self.layoutlmv2 = LayoutLMv2Model(config)
        # 创建一个dropout层，并赋值给self.dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 创建一个线性层，并赋值给self.classifier
        self.classifier = nn.Linear(config.hidden_size * 3, config.num_labels)

        # 调用post_init函数初始化权重和进行最终处理
        self.post_init()

    # 获取输入embedding的函数
    def get_input_embeddings(self):
        return self.layoutlmv2.embeddings.word_embeddings

    # 使用add_start_docstrings_to_model_forward装饰器添加模型前向传播的说明文档
    @add_start_docstrings_to_model_forward(LAYOUTLMV2_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 使用replace_return_docstrings装饰器调整返回值的说明文档格式
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    # 定义前向传播函数，接受多个输入参数
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
# 使用add_start_docstrings装饰器添加模型说明文档
@add_start_docstrings(
    """
    LayoutLMv2 Model with a token classification head on top (a linear layer on top of the text part of the hidden
    states) e.g. for sequence labeling (information extraction) tasks such as
    [FUNSD](https://guillaumejaume.github.io/FUNSD/), [SROIE](https://rrc.cvc.uab.es/?ch=13),
    [CORD](https://github.com/clovaai/cord) and [Kleister-NDA](https://github.com/applicaai/kleister-nda).
    """,
    LAYOUTLMV2_START_DOCSTRING,
)
# 定义LayoutLMv2ForTokenClassification类，继承自LayoutLMv2PreTrainedModel类
class LayoutLMv2ForTokenClassification(LayoutLMv2PreTrainedModel):
    # 初始化函数，接受config参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)
        # 将config中的num_labels赋值给self.num_labels
        self.num_labels = config.num_labels
        # 创建LayoutLMv2Model对象，并赋值给self.layoutlmv2
        self.layoutlmv2 = LayoutLMv2Model(config)
        # 创建一个dropout层，并赋值给self.dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 创建一个线性层，并赋值给self.classifier
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 调用post_init函数初始化权重和进行最终处理
        self.post_init()

    # 获取输入embedding的函数
    def get_input_embeddings(self):
        return self.layoutlmv2.embeddings.word_embeddings

    # 使用add_start_docstrings_to_model_forward装饰器添加模型前向传播的说明文档
    @add_start_docstrings_to_model_forward(LAYOUTLMV2_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 用于替换返回文档字符串，指定输出类型为TokenClassifierOutput，配置类为_CONFIG_FOR_DOC
    @replace_return_docstrings(output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    # 正向传播函数
    def forward(
        # 输入的标记 ID，类型为可选的长整型张量
        input_ids: Optional[torch.LongTensor] = None,
        # 边界框，类型为可选的长整型张量
        bbox: Optional[torch.LongTensor] = None,
        # 图像，类型为可选的浮点张量
        image: Optional[torch.FloatTensor] = None,
        # 注意力掩码，类型为可选的浮点张量
        attention_mask: Optional[torch.FloatTensor] = None,
        # 标记类型 ID，类型为可选的长整型张量
        token_type_ids: Optional[torch.LongTensor] = None,
        # 位置 ID，类型为可选的长整型张量
        position_ids: Optional[torch.LongTensor] = None,
        # 头掩码，类型为可选的浮点张量
        head_mask: Optional[torch.FloatTensor] = None,
        # 输入嵌入，类型为可选的浮点张量
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # 标签，类型为可选的长整型张量
        labels: Optional[torch.LongTensor] = None,
        # 是否输出注意力权重，类型为可选的布尔值
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，类型为可选的布尔值
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典，类型为可选的布尔值
        return_dict: Optional[bool] = None,
# 添加文档字符串，描述 LayoutLMv2 模型及其在 extractive question-answering 任务中的用途
@add_start_docstrings(
    """
    LayoutLMv2 Model with a span classification head on top for extractive question-answering tasks such as
    [DocVQA](https://rrc.cvc.uab.es/?ch=17) (a linear layer on top of the text part of the hidden-states output to
    compute `span start logits` and `span end logits`).
    """,
    LAYOUTLMV2_START_DOCSTRING,
)
# 定义 LayoutLMv2 模型类，继承自 LayoutLMv2PreTrainedModel
class LayoutLMv2ForQuestionAnswering(LayoutLMv2PreTrainedModel):
    # 初始化方法
    def __init__(self, config, has_visual_segment_embedding=True):
        super().__init__(config)
        # 设置类属性 num_labels
        self.num_labels = config.num_labels
        # 设置 config 中的属性 has_visual_segment_embedding
        config.has_visual_segment_embedding = has_visual_segment_embedding
        # 创建 LayoutLMv2Model 模型实例
        self.layoutlmv2 = LayoutLMv2Model(config)
        # 创建线性层，用于计算 span 的开始和结束 logits
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.layoutlmv2.embeddings.word_embeddings

    # 定义前向传播方法，并添加对应的文档字符串
    @add_start_docstrings_to_model_forward(LAYOUTLMV2_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
```