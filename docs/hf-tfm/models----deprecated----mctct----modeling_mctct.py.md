# `.\models\deprecated\mctct\modeling_mctct.py`

```py
# coding=utf-8
# 版权所有 2022 年 HuggingFace Inc. 团队保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（"许可证"）授权；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按"原样"提供，
# 不提供任何明示或暗示的担保或条件。
# 有关更多详细信息，请参阅许可证。
""" PyTorch M-CTC-T 模型。"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ....activations import ACT2FN
from ....file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ....integrations.deepspeed import is_deepspeed_zero3_enabled
from ....modeling_attn_mask_utils import _prepare_4d_attention_mask
from ....modeling_outputs import BaseModelOutput, CausalLMOutput
from ....modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from ....utils import logging
from .configuration_mctct import MCTCTConfig

logger = logging.get_logger(__name__)

_HIDDEN_STATES_START_POSITION = 1

_CONFIG_FOR_DOC = "MCTCTConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "speechbrain/m-ctc-t-large"
_EXPECTED_OUTPUT_SHAPE = [1, 195, 1536]

# CTC docstring
_CTC_EXPECTED_OUTPUT = '"Mr. Quilter is the apostle of the middle classes, and we\'re glad to welcome his gospel."'
_CTC_EXPECTED_LOSS = 1885.65

MCTCT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "speechbrain/m-ctc-t-large",
    # See all M-CTC-T models at https://huggingface.co/models?filter=mctct
]


class MCTCTConv1dSubsampler(nn.Module):
    """
    Convolutional subsampler: a stack of 1D convolution (along temporal dimension) followed by non-linear activation
    via gated linear units (https://arxiv.org/abs/1911.08460)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config  # 保存配置信息到对象属性中
        self.glu_dim = config.conv_glu_dim  # 从配置中获取 GLU 操作的维度

        self.dropout = nn.Dropout(config.conv_dropout)  # 使用配置中的 dropout 概率初始化 dropout 层

        self.num_layers = config.num_conv_layers  # 获取卷积层的数量
        self.in_channels = config.input_feat_per_channel * config.input_channels  # 计算输入通道数乘以每个通道的特征数

        if self.num_layers > 1:
            if config.conv_channels is None:
                raise ValueError(
                    "Need to specify `conv_channels` configuration in `MCTCTConfig` to use multiple convolution"
                    " layers."
                )
            self.mid_channels = config.conv_channels  # 如果有多于一层卷积，则从配置中获取中间层的通道数
        else:
            self.mid_channels = None  # 如果只有一层卷积，则中间层通道数为 None

        self.out_channels = config.hidden_size * 2  # 计算输出通道数，考虑到 GLU 操作会减半
        self.kernel_size = config.conv_kernel  # 获取卷积核大小的配置
        self.stride = config.conv_stride  # 获取卷积步长的配置

        # NOTE: MCTCT 模型原理上只使用一个卷积核。我为了灵活性允许多层卷积，但不确定模型定义应该限制为单层。
        # 这在考虑像 forward() 函数第1行的填充时尤其重要。
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                self.in_channels if i == 0 else self.mid_channels[i-1],  # 输入通道数根据层次确定
                self.mid_channels[i] if i < self.num_layers - 1 else self.out_channels,  # 输出通道数根据层次确定
                kernel_size=k,
                stride=self.stride[i],
                padding="valid",  # 使用有效填充（无填充）
            )
            for i, k in enumerate(self.kernel_size)  # 遍历卷积核大小的配置列表
        )

    def forward(self, input_features):
        # NOTE: 参考 __init__ 中的注释，目前只计算填充，就像只有一层卷积一样。
        padding = sum([size // 2 for size in self.kernel_size])  # 计算填充大小使卷积后大小不变

        input_features = torch.nn.functional.pad(input_features, (0, 0, padding, padding), "constant", 0)  # 对输入特征进行填充
        hidden_states = input_features.transpose(1, 2).contiguous()  # 调整输入特征的维度顺序，变为 Batch x Frame x Time

        for conv in self.conv_layers:
            hidden_states = conv(hidden_states)  # 执行卷积操作
            hidden_states = nn.functional.glu(hidden_states, dim=self.glu_dim)  # 执行 GLU 激活函数
            hidden_states = self.dropout(hidden_states)  # 应用 dropout

        hidden_states = hidden_states.transpose(1, 2).contiguous()  # 调整隐藏状态的维度顺序，变为 Batch x Time x Frame
        return hidden_states  # 返回处理后的隐藏状态作为输出
# 定义 MCTCTEmbeddings 类，用于构建来自单词、位置和标记类型嵌入的总体嵌入。
class MCTCTEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    # 初始化方法，接受一个配置对象 config
    def __init__(self, config):
        super().__init__()
        # 使用 nn.Embedding 创建单词嵌入层，vocab_size 表示词汇表大小，hidden_size 表示隐藏层大小，padding_idx 表示填充标记的索引
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 使用 nn.Embedding 创建位置嵌入层，max_position_embeddings 表示最大位置嵌入数量
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 使用 nn.Embedding 创建标记类型嵌入层，type_vocab_size 表示标记类型数量
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 创建 MCTCTLayerNorm 实例作为 LayerNorm 层
        # self.LayerNorm 不使用 snake_case，以保持与 TensorFlow 模型变量名一致，便于加载任何 TensorFlow 检查点文件
        self.LayerNorm = MCTCTLayerNorm()
        # 使用 nn.Dropout 创建丢弃层，hidden_dropout_prob 表示丢弃概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 注册 position_ids 缓冲区，存储位置 id，torch.arange 创建从 0 到 max_position_embeddings-1 的张量
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 注册 token_type_ids 缓冲区，存储标记类型 id，创建与 position_ids 相同大小的全零张量
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
            persistent=False,
        )

    # 前向传播方法，接受多个输入参数并返回嵌入张量
    def forward(
        self, input_features=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        # 获取输入特征的形状
        input_shape = input_features.size() if input_features is not None else inputs_embeds.size()[:-1]

        # 获取序列长度
        seq_length = input_shape[1]

        # 如果 position_ids 为 None，则使用注册的 position_ids 缓冲区中的值
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # 如果 token_type_ids 为 None，则使用注册的 token_type_ids 缓冲区中的值
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果 inputs_embeds 为 None，则使用 word_embeddings 对 input_features 进行嵌入
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_features)

        # 使用 token_type_embeddings 对 token_type_ids 进行嵌入
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将单词嵌入和标记类型嵌入相加，得到最终嵌入张量
        embeddings = inputs_embeds + token_type_embeddings

        # 对嵌入张量进行 LayerNorm 处理
        embeddings = self.LayerNorm(embeddings)
        # 对嵌入张量进行丢弃处理
        embeddings = self.dropout(embeddings)
        # 返回处理后的嵌入张量
        return embeddings
    # 初始化函数，接受配置参数并初始化模型的各种属性
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 检查隐藏大小是否是注意力头数的倍数，同时没有嵌入大小属性
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            # 如果不是倍数关系，则抛出数值错误异常
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 初始化注意力头数和每个注意力头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.attention_head_dim
        # 计算所有注意力头的总大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化查询、键、值的线性变换层，用于注意力机制中的线性变换
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=False)

        # 初始化 dropout 层，用于注意力概率的 dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # 初始化最大位置嵌入大小和距离嵌入层
        self.max_position_embeddings = config.max_position_embeddings
        self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        # 是否为解码器模型的标志
        self.is_decoder = config.is_decoder

    # 将输入张量变换为注意力分数张量的形状
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 重塑输入张量为指定形状的张量
    def reshape_fortran(self, x, shape):
        if len(x.shape) > 0:
            x = x.permute(*reversed(range(len(x.shape))))
        return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))

    # 实现相对位置嵌入旋转的函数
    def relative_position_embedding_rotate(self, scores):
        # 注意：需要重新评估是否真正需要重新实现这部分代码，
        # 或者完全重构的原因是由于代码的其他部分。添加这部分和重塑 fortrain 代码似乎非常不理想。
        
        # 将张量维度重新排列，调整位置以适应后续操作
        scores = scores.permute(0, 2, 3, 1)  # 例如 [10, 1839, 14, 4]

        batch, hidden_state, seq_len, heads = scores.shape

        # 在第二维度上拼接零张量，扩展张量尺寸
        scores = torch.cat((scores, torch.zeros((batch, seq_len, seq_len, heads), device=scores.device)), dim=1)

        # 调用重塑函数，将张量重新整形为指定形状
        scores = self.reshape_fortran(scores, [batch, (hidden_state + seq_len) * seq_len, 1, heads])

        # 保留部分张量尺寸，截取需要的部分张量
        scores = scores[:, : (seq_len + hidden_state - 1) * seq_len]

        # 再次调用重塑函数，将张量调整为另一种指定形状
        scores = self.reshape_fortran(scores, [batch, hidden_state + seq_len - 1, seq_len, heads])

        halfpoint = hidden_state // 2
        # 调整张量顺序，使得维度重新排列
        scores = scores[:, halfpoint : halfpoint + seq_len].transpose(1, 2)  # 例如 [10, 14, 14, 4]

        # 返回重新排列后的张量，调整张量顺序以适应后续操作
        return scores.permute(0, 3, 1, 2)

    # 前向传播函数，定义了模型的计算流程
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        ):
            # 使用 self.query 对隐藏状态进行查询操作，得到混合查询层
            mixed_query_layer = self.query(hidden_states)
            # 将混合查询层的结果除以 sqrt(attention_head_size)，用于缩放
            mixed_query_layer = mixed_query_layer / math.sqrt(self.attention_head_size)

            # 使用 self.key 对隐藏状态进行键生成，并转置以匹配注意力分数计算的要求
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            # 使用 self.value 对隐藏状态进行值生成，并转置以匹配注意力分数计算的要求
            value_layer = self.transpose_for_scores(self.value(hidden_states))

            # 将混合查询层转置以匹配注意力分数计算的要求
            query_layer = self.transpose_for_scores(mixed_query_layer)

            # 计算注意力分数，使用 torch.matmul 进行矩阵乘法
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

            # 获取相对位置嵌入权重
            positional_embedding = self.distance_embedding.weight
            # 使用 einsum 函数计算相对位置分数
            relative_position_scores = torch.einsum("lh, bche -> bcle", positional_embedding, query_layer.transpose(2, 3))

            # 对相对位置分数应用旋转操作
            relative_position_scores = self.relative_position_embedding_rotate(relative_position_scores)
            # 将相对位置分数添加到注意力分数中
            attention_scores = attention_scores + relative_position_scores

            # 如果存在注意力掩码，则将其应用到注意力分数中
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask

            # 对注意力分数进行 softmax 归一化得到注意力概率
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)

            # 使用 dropout 函数对注意力概率进行随机失活
            attention_probs = self.dropout(attention_probs)

            # 如果存在头部掩码，则将其应用到注意力概率中
            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            # 计算上下文层，将注意力概率与值层相乘得到加权后的上下文表示
            context_layer = torch.matmul(attention_probs, value_layer)

            # 对上下文层进行维度变换和展平操作
            context_layer = context_layer.permute(0, 2, 1, 3).flatten(start_dim=-2)

            # 根据是否需要输出注意力权重决定输出内容
            outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

            # 返回输出内容
            return outputs
class MCTCTLayerNorm(nn.Module):
    # 定义一个自定义的 LayerNorm 模块，用于单例权重和偏置参数的标准化
    def __init__(self):
        super().__init__()
        # 初始化单例权重参数为1，可学习的模型参数
        self.singleton_weight = nn.Parameter(torch.ones(1))
        # 初始化单例偏置参数为0，可学习的模型参数
        self.singleton_bias = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states):
        # 根据单例权重和偏置参数对输入的 hidden_states 进行线性变换和偏置
        return (hidden_states * self.singleton_weight) + self.singleton_bias


class MCTCTSelfOutput(nn.Module):
    # 定义 Transformer 模型中的 SelfOutput 层，包括线性映射、LayerNorm 和 dropout 操作
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 使用线性映射将隐藏状态映射到相同的维度
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        # 使用 LayerNorm 进行归一化处理
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 使用 dropout 进行随机失活以防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # 对隐藏状态进行线性映射
        hidden_states = self.dense(hidden_states)
        # 对映射结果进行 dropout 操作
        hidden_states = self.dropout(hidden_states)
        # 对 dropout 后的结果进行 LayerNorm 处理并与输入张量相加
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class MCTCTAttention(nn.Module):
    # 定义 Transformer 模型中的 Attention 层，包括 SelfAttention 和 SelfOutput
    def __init__(self, config):
        super().__init__()
        # 初始化 SelfAttention 层
        self.self = MCTCTSelfAttention(config)
        # 初始化 SelfOutput 层
        self.output = MCTCTSelfOutput(config)
        # 初始化用于存储被修剪的 attention head 的集合
        self.pruned_heads = set()

    def prune_heads(self, heads):
        # 根据传入的 head 索引列表修剪模型的 attention heads
        if len(heads) == 0:
            return
        # 寻找可以修剪的 heads 和对应的索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 对 SelfAttention 和 SelfOutput 中的线性层进行修剪
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并记录修剪的 heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        # 前向传播函数，首先通过 SelfAttention 层获取输出，然后经过 SelfOutput 层得到最终输出
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # 如果有需要，添加注意力权重信息

        return outputs


class MCTCTIntermediate(nn.Module):
    # 定义 Transformer 模型中的 Intermediate 层，包括线性映射和激活函数
    def __init__(self, config):
        super().__init__()
        # 使用线性映射将隐藏状态映射到中间层的维度
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        # 根据配置选择相应的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
    # 定义一个类方法 `forward`，用于处理输入的隐藏状态数据
    def forward(self, hidden_states):
        # 将隐藏状态数据输入全连接层 `dense` 进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的隐藏状态数据应用激活函数 `intermediate_act_fn`
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回处理后的隐藏状态数据
        return hidden_states
class MCTCTOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个全连接层，将输入特征从 config.intermediate_size 转换为 config.hidden_size，且不使用偏置
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        # 初始化 LayerNorm 层，对隐藏状态的每个特征进行归一化，eps 是归一化过程中的稳定性参数
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化 Dropout 层，以 config.hidden_dropout_prob 的概率随机丢弃隐藏状态中的特征
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # 将隐藏状态经过全连接层变换
        hidden_states = self.dense(hidden_states)
        # 对变换后的隐藏状态进行 Dropout 处理
        hidden_states = self.dropout(hidden_states)
        # 将经过 Dropout 处理后的隐藏状态与输入张量进行残差连接，并经过 LayerNorm 处理
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class MCTCTLayer(nn.Module):
    def __init__(self, config: MCTCTConfig):
        super().__init__()

        # 设置序列长度维度为1，用于后续的块状处理
        self.seq_len_dim = 1
        # 设置前馈传播的块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward

        # 初始化 MCTCTIntermediate 层和 MCTCTAttention 层
        self.intermediate = MCTCTIntermediate(config)
        self.attention = MCTCTAttention(config)
        # 标记是否为解码器
        self.is_decoder = config.is_decoder
        # 初始化 MCTCTOutput 层
        self.output = MCTCTOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        # 使用注意力层处理隐藏状态，获取自注意力输出和其它附加信息
        self_attention_outputs = self.attention(
            hidden_states, attention_mask, head_mask, output_attentions=output_attentions
        )
        # 获取自注意力的输出
        attention_output = self_attention_outputs[0]
        # 如果输出注意力权重，将其加入输出中
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # 将注意力输出应用于前馈传播的块状处理，按照指定的块大小和序列长度维度进行处理
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )

        # 将块状处理后的输出添加到总体输出中
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        # 将注意力输出输入到中间层进行处理
        intermediate_output = self.intermediate(attention_output)
        # 将中间层的输出输入到输出层进行处理
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class MCTCTPreTrainedModel(PreTrainedModel):
    """
    一个处理权重初始化、预训练模型下载和加载的抽象类。
    """

    # 指定配置类
    config_class = MCTCTConfig
    # 模型名称前缀
    base_model_prefix = "mctct"
    # 主输入名称
    main_input_name = "input_features"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    def _init_weights(self, module):
        """Initialize the weights"""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            # 对于线性层，使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果存在偏置项，初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 对于嵌入层，使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果定义了padding_idx，将对应的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 对于LayerNorm层，初始化偏置为零，权重为1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, MCTCTLayerNorm):
            # 对于自定义的LayerNorm，初始化权重为1，偏置为零
            module.singleton_weight.data.fill_(1.0)
            module.singleton_bias.data.zero_()
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # 对于线性层和一维卷积层，使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果存在偏置项，初始化为零
            if module.bias is not None:
                module.bias.data.zero_()

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """
        dilation = 1
        for _, kernel_sz, stride in zip(
            range(self.config.num_conv_layers), self.config.conv_kernel, self.config.conv_stride
        ):
            padding = kernel_sz // 2
            # 计算卷积层输出长度
            input_lengths = input_lengths + 2 * padding - dilation * (kernel_sz - 1) - 1
            input_lengths = torch.div(input_lengths, stride, rounding_mode="trunc") + 1

        return input_lengths

    def _get_feature_vector_attention_mask(self, feature_vector_length, attention_mask):
        # 如果注意力掩码的维度大于2，将其转换为2维
        if len(attention_mask.shape) > 2:
            attention_mask = attention_mask[:, :, -1]

        # 计算通过特征向量长度生成的注意力掩码
        subsampled_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1))
        bsz = attention_mask.size()[0]
        attention_mask = torch.zeros(
            (bsz, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )

        # 设置所有输出长度前的位置为1，表示需要关注这些位置
        attention_mask[(torch.arange(bsz, device=attention_mask.device), subsampled_lengths - 1)] = 1
        # 对注意力掩码进行累积操作，确保在输出长度之前的位置都被关注到
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).long()
        return attention_mask
MCTCT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`MCTCTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

MCTCT_INPUTS_DOCSTRING = r"""
    Args:
        input_features (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`Wav2Vec2CTCTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
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
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""

class MCTCTEncoder(MCTCTPreTrainedModel):
    def __init__(self, config: MCTCTConfig):
        super().__init__(config)
        self.hidden_dropout_prob = config.hidden_dropout_prob  # 初始化隐藏层的dropout概率

        self.layer_norm = MCTCTLayerNorm()  # 初始化层归一化
        self.conv = MCTCTConv1dSubsampler(config)  # 使用给定配置初始化一维卷积子采样器
        self.layers = nn.ModuleList([MCTCTLayer(config) for _ in range(config.num_hidden_layers)])  # 使用给定配置初始化多层MCTCTLayer组成的模块列表

        self.gradient_checkpointing = False  # 初始化梯度检查点为False

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor,
        head_mask: torch.Tensor,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    # 定义一个字符串，描述了一个模型的基本特性：M-CTC-T Model，输出原始的隐藏状态而没有额外的头部处理。
    # 这是模型文档字符串的开始。
    "The bare M-CTC-T Model transformer outputting raw hidden-states without any specific head on top.",
    MCTCT_START_DOCSTRING,
# 定义一个 MCTCTModel 类，继承自 MCTCTPreTrainedModel 类
class MCTCTModel(MCTCTPreTrainedModel):
    def __init__(self, config):
        # 调用父类的构造方法，传入配置参数
        super().__init__(config)
        # 将配置参数保存在对象的属性中
        self.config = config

        # 初始化编码器（encoder）部分
        self.encoder = MCTCTEncoder(config)

        # 执行额外的初始化操作和最终处理
        self.post_init()

    # 使用装饰器添加模型前向传播方法的文档字符串
    @add_start_docstrings_to_model_forward(MCTCT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 使用装饰器添加示例代码的文档字符串
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    # 定义模型的前向传播方法
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        # 如果未显式提供输出注意力的设置，则使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未显式提供输出隐藏状态的设置，则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未显式提供返回字典的设置，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果输入特征为空，则引发数值错误
        if input_features is None:
            raise ValueError("You have to specify input_features.")

        # 将输入特征传递给编码器，获取编码器的输出
        encoder_outputs = self.encoder(
            input_features,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取编码器输出中的序列输出
        sequence_output = encoder_outputs[0]

        # 如果不使用返回字典，则返回一个包含序列输出和其它编码器输出的元组
        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        # 如果使用返回字典，则返回一个包含序列输出、隐藏状态和注意力的 BaseModelOutput 对象
        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# 使用装饰器添加模型描述文档字符串，并继承自 MCTCTPreTrainedModel 类
@add_start_docstrings(
    """MCTCT Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).""",
    MCTCT_START_DOCSTRING,
)
class MCTCTForCTC(MCTCTPreTrainedModel):
    # 初始化方法，接受一个配置参数并调用父类的初始化方法
    def __init__(self, config):
        super().__init__(config)

        # 使用给定的配置参数初始化 MCTCTModel 类的实例
        self.mctct = MCTCTModel(config)

        # 如果配置中未定义词汇表大小，则抛出值错误异常
        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `MCTCTForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )

        # 从配置中获取隐藏层大小作为输出隐藏层大小
        output_hidden_size = config.hidden_size

        # 创建一个线性层，用于CTC（Connectionist Temporal Classification）任务的输出
        self.ctc_head = nn.Linear(output_hidden_size, config.vocab_size)

        # 调用后续初始化方法，用于权重初始化和最终处理
        self.post_init()

    # 前向传播方法，接受多个输入参数并返回预测结果或损失
    @add_start_docstrings_to_model_forward(MCTCT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_CTC_EXPECTED_OUTPUT,
        expected_loss=_CTC_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        ) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """
        # Determine if the return_dict should be set based on the model's configuration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Perform masked connectionist temporal classification on the input features
        outputs = self.mctct(
            input_features,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Extract hidden states from the model's outputs
        hidden_states = outputs[0]

        # Compute logits using the CTC head
        logits = self.ctc_head(hidden_states)

        loss = None
        if labels is not None:
            if labels.max() >= self.config.vocab_size:
                # Raise an error if any label value exceeds the vocabulary size
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # Compute input_lengths from attention_mask
            attention_mask = (
                attention_mask
                if attention_mask is not None
                else torch.ones(input_features.shape[:-1], dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
            
            # Create a mask for valid labels and calculate target_lengths
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # Log-probabilities for the CTC loss calculation
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            # Disable CuDNN for CTC loss computation
            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        if not return_dict:
            # Prepare output tuple if return_dict is False
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        # Return structured output using CausalLMOutput if return_dict is True
        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
```