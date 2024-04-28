# `.\models\esm\modeling_esm.py`

```py
# 设置文件编码为 UTF-8
# 版权声明和许可信息
# 版权所有 2022 年 Meta 公司和 HuggingFace Inc. 团队保留所有权利。
#
# 根据 Apache 许可证 2.0 版（“许可证”）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件是基于“按原样”的基础分发的，
# 没有任何形式的担保或条件。
# 有关详细信息，请参阅许可证。
""" PyTorch ESM 模型。"""

# 导入必要的库
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入帮助函数和模型输出
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import logging

# 导入配置
from .configuration_esm import EsmConfig

# 设置日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置
_CHECKPOINT_FOR_DOC = "facebook/esm2_t6_8M_UR50D"
_CONFIG_FOR_DOC = "EsmConfig"

# ESM 预训练模型档案列表
ESM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/esm2_t6_8M_UR50D",
    "facebook/esm2_t12_35M_UR50D",
    # 这不是所有 ESM 模型的完整列表！
    # 查看所有 ESM 模型 https://huggingface.co/models?filter=esm
]

# 旋转函数，将输入向量 x 旋转 180 度
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

# 应用旋转位置嵌入，基于余弦和正弦参数对输入张量进行变换
def apply_rotary_pos_emb(x, cos, sin):
    cos = cos[:, :, : x.shape[-2], :]
    sin = sin[:, :, : x.shape[-2], :]

    return (x * cos) + (rotate_half(x) * sin)

# GELU 激活函数的实现
def gelu(x):
    """
    这是原始 ESM 代码库中的 gelu 实现。使用 F.gelu 得到微妙错误的结果。
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

# 使最后两个维度对称化的函数，用于接触预测
def symmetrize(x):
    "使最后两个维度对称化，用于接触预测。"
    return x + x.transpose(-1, -2)

# 执行平均乘积修正的函数，用于接触预测
def average_product_correct(x):
    "执行平均乘积修正，用于接触预测。"
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)

    avg = a1 * a2
    avg.div_(a12)  # in-place 减少内存占用
    normalized = x - avg
    return normalized

# 旋转嵌入类，基于 RoFormer 中的旋转位置嵌入
class RotaryEmbedding(torch.nn.Module):
    """
    基于 RoFormer 中的旋转位置嵌入的旋转位置嵌入。
    查询和键由依赖于它们相对位置的旋转矩阵进行变换。
    """
    # 初始化函数，接受一个整数参数 dim
    def __init__(self, dim: int):
        # 调用父类初始化方法
        super().__init__()
        # 生成并保存逆频率缓冲区（非可训练）
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        # 将逆频率缓冲区保存在模型的缓冲区中
        self.register_buffer("inv_freq", inv_freq)

        # 缓存序列长度、余弦值表和正弦值表
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    # 更新余弦值和正弦值表的函数
    def _update_cos_sin_tables(self, x, seq_dimension=2):
        # 获取输入张量的序列长度
        seq_len = x.shape[seq_dimension]

        # 如果序列长度发生变化，或者当前设备与缓存中的设备不同（可能是由于追踪等原因）
        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device:
            # 更新缓存的序列长度
            self._seq_len_cached = seq_len
            # 生成频率张量
            t = torch.arange(x.shape[seq_dimension], device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            # 生成余弦和正弦值张量
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            # 缓存余弦和正弦值张量
            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]

        # 返回缓存的余弦和正弦值张量
        return self._cos_cached, self._sin_cached

    # 前向传播函数，接受两个张量参数 q 和 k，返回两个张量的元组
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 更新余弦值和正弦值表
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension=-2)

        # 应用旋转位置嵌入到输入张量，并返回结果元组
        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        )
class EsmContactPredictionHead(nn.Module):
    """Performs symmetrization, apc, and computes a logistic regression on the output features"""

    def __init__(
        self,
        in_features: int,
        bias=True,
        eos_idx: int = 2,
    ):
        # 初始化类，并设置属性
        super().__init__()
        self.in_features = in_features
        self.eos_idx = eos_idx
        self.regression = nn.Linear(in_features, 1, bias)  # 创建线性层用于逻辑回归
        self.activation = nn.Sigmoid()  # 创建激活函数

    def forward(self, tokens, attentions):
        # 移除 eos 标记的注意力
        eos_mask = tokens.ne(self.eos_idx).to(attentions)
        eos_mask = eos_mask.unsqueeze(1) * eos_mask.unsqueeze(2)
        attentions = attentions * eos_mask[:, None, None, :, :]
        attentions = attentions[..., :-1, :-1]
        # 移除 cls 标记的注意力
        attentions = attentions[..., 1:, 1:]
        batch_size, layers, heads, seqlen, _ = attentions.size()
        attentions = attentions.view(batch_size, layers * heads, seqlen, seqlen)

        # 特征：批大小 x 通道 x 令牌 x 令牌（对称）
        attentions = attentions.to(
            self.regression.weight.device
        )  # 注意力始终为 float32，可能需要转换为 float16
        attentions = average_product_correct(symmetrize(attentions))  # 对注意力进行一些处理
        attentions = attentions.permute(0, 2, 3, 1)  # 调整张量维度
        return self.activation(self.regression(attentions).squeeze(3))  # 返回逻辑回归结果


class EsmEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config):
        # 初始化类，并设置属性
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        if config.emb_layer_norm_before:
            self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.layer_norm = None
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )
        self.token_dropout = config.token_dropout
        self.mask_token_id = config.mask_token_id

    def forward(
        self, input_ids=None, attention_mask=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        # 如果位置编码为空，则根据输入的标记 ID 创建位置编码。任何填充的标记保持填充状态。
        if position_ids is None:
            if input_ids is not None:
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        # 如果输入的嵌入是空的，则使用词嵌入层将输入的标记 ID 转换为嵌入向量
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # 注意：如果我们将来要支持 ESM-1（而不是 1b！），则需要在这里支持一个嵌入缩放因子。
        embeddings = inputs_embeds

        # Matt: ESM 有一种稍微不同的方式处理 MLM 中的掩码。如果 token_dropout 标志为 False，则处理方式与 BERT/RoBERTa 相同。
        # 如果将其设置为 True，则会将掩码的标记视为被选择用于输入的 dropout 并将其清零。
        # 当掩码的标记不存在时，通过缩放嵌入，补偿掉这种 "mask-dropout"，缩放因子为 (训练期间未掩码的标记的分数) / (样本中未掩码的标记的分数)。
        # 这类似于在评估期间放弃值时，dropout 层如何缩小输出（或者等效地，在训练期间如何放大未放弃的输出）。
        if self.token_dropout:
            # 将掩码标记位置的嵌入值设置为 0.0
            embeddings = embeddings.masked_fill((input_ids == self.mask_token_id).unsqueeze(-1), 0.0)
            mask_ratio_train = 0.15 * 0.8  # 在所有 ESM 模型训练运行中硬编码的比率
            src_lengths = attention_mask.sum(-1)
            mask_ratio_observed = (input_ids == self.mask_token_id).sum(-1).float() / src_lengths
            # 缩放嵌入以补偿 mask-dropout
            embeddings = (embeddings * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]).to(
                embeddings.dtype
            )

        # 如果位置编码类型是 "absolute"，则添加绝对位置编码到嵌入向量中
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings

        # 如果存在层归一化，则应用层归一化
        if self.layer_norm is not None:
            embeddings = self.layer_norm(embeddings)
        # 如果存在注意力掩码，则将注意力掩码应用到嵌入向量中
        if attention_mask is not None:
            embeddings = (embeddings * attention_mask.unsqueeze(-1)).to(embeddings.dtype)
        # Matt: 我认为这一行代码从 BERT 复制过来时出现了错误，暂时禁用它。
        # embeddings = self.dropout(embeddings)
        # 返回嵌入向量
        return embeddings
    # 从输入的嵌入向量中创建位置ID，由于直接提供了嵌入向量，无法推断哪些是填充的，因此只能生成顺序的位置ID
    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        # 获取输入嵌入向量的形状
        input_shape = inputs_embeds.size()[:-1]
        # 获取序列长度
        sequence_length = input_shape[1]

        # 生成位置ID，从padding_idx + 1到sequence_length + padding_idx + 1的连续整数，类型为long，在输入嵌入向量的设备上
        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        # 将位置ID扩展成与输入形状相同的张量并增加一个维度
        return position_ids.unsqueeze(0).expand(input_shape)
# 定义了一个自注意力机制的类 EsmSelfAttention，继承自 nn.Module 类
class EsmSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        # 调用父类的构造函数
        super().__init__()
        # 如果隐藏层大小不能被注意力头的数量整除，且配置中没有嵌入大小，则引发 ValueError
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 初始化注意力头的数量
        self.num_attention_heads = config.num_attention_heads
        # 计算每个注意力头的大小
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        # 计算所有注意力头的总大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化 query、key、value 的线性变换层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 初始化 dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # 设置位置嵌入类型，默认为绝对位置嵌入
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果位置嵌入类型为相对关键词或相对关键词查询，则初始化距离嵌入层
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            # 初始化最大位置嵌入数
            self.max_position_embeddings = config.max_position_embeddings
            # 初始化距离嵌入层
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        # 如果位置嵌入类型为旋转，则初始化旋转嵌入对象
        elif self.position_embedding_type == "rotary":
            self.rotary_embeddings = RotaryEmbedding(dim=self.attention_head_size)

        # 设置是否为解码器
        self.is_decoder = config.is_decoder

    # 将输入张量转换为分数张量
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # 计算新的张量形状
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # 重新调整张量形状
        x = x.view(new_x_shape)
        # 交换张量维度顺序，以得到注意力分数张量
        return x.permute(0, 2, 1, 3)

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    # 初始化函数，设置配置参数
    def __init__(self, config):
        # 调用父类初始化函数
        super().__init__()
        # 初始化自注意力对象
        self.self = EsmSelfAttention(config)
        # 初始化自注意力输出对象
        self.output = EsmSelfOutput(config)
        # 存储需要剪枝的头部索引
        self.pruned_heads = set()
        # 初始化 LayerNorm 层
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # 剪枝头部函数
    def prune_heads(self, heads):
        # 如果剪枝头部列表为空，则直接返回
        if len(heads) == 0:
            return
        # 查找可剪枝的头部及其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        # 剪枝自注意力查询权重
        self.self.query = prune_linear_layer(self.self.query, index)
        # 剪枝自注意力键权重
        self.self.key = prune_linear_layer(self.self.key, index)
        # 剪枝自注意力值权重
        self.self.value = prune_linear_layer(self.self.value, index)
        # 剪枝自注意力输出层
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储剪枝头部
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 前向传播函数
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # 对隐藏状态进行 LayerNorm 处理
        hidden_states_ln = self.LayerNorm(hidden_states)
        # 调用自注意力模块
        self_outputs = self.self(
            hidden_states_ln,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 自注意力输出结果传入自注意力输出层
        attention_output = self.output(self_outputs[0], hidden_states)
        # 返回结果，如果需要输出注意力分布，也一起返回
        outputs = (attention_output,) + self_outputs[1:]  # 如果需要输出注意力分布，则添加到输出中
        return outputs
# 定义 EsmIntermediate 类，继承自 nn.Module
class EsmIntermediate(nn.Module):
    # 初始化函数，接受一个 config 对象参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个全连接层，输入维度为 config.hidden_size，输出维度为 config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    # 前向传播函数，接受 hidden_states 作为输入张量，返回输出张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过全连接层处理 hidden_states
        hidden_states = self.dense(hidden_states)
        # 经过 gelu 激活函数处理
        hidden_states = gelu(hidden_states)
        # 返回处理后的 hidden_states
        return hidden_states

# 定义 EsmOutput 类，继承自 nn.Module
class EsmOutput(nn.Module):
    # 初始化函数，接受一个 config 对象参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个全连接层，输入维度为 config.intermediate_size，输出维度为 config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个 dropout 层，使用 config.hidden_dropout_prob 作为丢弃概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，接受 hidden_states 和 input_tensor 作为输入，返回输出张量
    def forward(self, hidden_states, input_tensor):
        # 通过全连接层处理 hidden_states
        hidden_states = self.dense(hidden_states)
        # 经过 dropout 处理
        hidden_states = self.dropout(hidden_states)
        # 将处理后的 hidden_states 与 input_tensor 相加
        hidden_states = hidden_states + input_tensor
        # 返回处理后的 hidden_states
        return hidden_states

# 定义 EsmLayer 类，继承自 nn.Module
class EsmLayer(nn.Module):
    # 初始化函数，接受一个 config 对象参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 设置 chunk_size_feed_forward 和 seq_len_dim 属性
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        # 创建 EsmAttention 对象
        self.attention = EsmAttention(config)
        # 设置 is_decoder 和 add_cross_attention 属性
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        # 添加交叉注意力机制
        if self.add_cross_attention:
            if not self.is_decoder:
                # 如果不是解码器模型，则抛出异常
                raise RuntimeError(f"{self} should be used as a decoder model if cross attention is added")
            # 创建交叉注意力对象
            self.crossattention = EsmAttention(config)
        # 创建 EsmIntermediate、EsmOutput 和 LayerNorm 层
        self.intermediate = EsmIntermediate(config)
        self.output = EsmOutput(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # 前向传播函数，接受多个参数，返回输出结果
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        ):
        # 如果之前有缓存的自注意力键值对，那么获取uni-directional自注意力缓存的键/值元组在位置1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 对自注意力的输出进行计算，包括注意力、头部掩码等
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # 获取自注意力的输出
        attention_output = self_attention_outputs[0]

        # 如果是解码器，最后的输出是自注意力缓存的元组
        if self.is_decoder:
            # 获取除了自注意力缓存之外的所有输出
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            # 如果不是解码器，输出包括自注意力权重
            outputs = self_attention_outputs[1:]
            
        cross_attn_present_key_value = None
        # 如果是解码器并且有编码器的隐藏状态
        if self.is_decoder and encoder_hidden_states is not None:
            # 如果没有交叉注意力，则抛出属性错误
            if not hasattr(self, "crossattention"):
                raise AttributeError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated"
                    " with cross-attention layers by setting `config.add_cross_attention=True`"
                )
            # 获取交叉注意力缓存的键值对在过去键值对元组的第3和第4位置
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 对交叉注意力的输出进行计算，包括注意力、头部掩码、编码器隐藏状态等
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            # 获取交叉注意力的输出
            attention_output = cross_attention_outputs[0]
            # 将交叉注意力的输出添加到输出中
            outputs = outputs + cross_attention_outputs[1:-1]
            # 将交叉注意力缓存添加到现有的键值对中
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 将注意力输出传入前馈网络进行计算
        layer_output = self.feed_forward_chunk(attention_output)

        outputs = (layer_output,) + outputs

        # 如果是解码器，将注意力键/值对作为最后的输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)
        return outputs

    # 对注意力输出进行前馈网络的计算
    def feed_forward_chunk(self, attention_output):
        # 对注意力输出进行LayerNorm处理
        attention_output_ln = self.LayerNorm(attention_output)
        # 通过中间层进行转换
        intermediate_output = self.intermediate(attention_output_ln)
        # 通过输出层获取最终输出
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
# 定义 EsmEncoder 类，继承自 nn.Module
class EsmEncoder(nn.Module):
    # 初始化方法，接受一个 config 参数
    def __init__(self, config):
        # 调用父类 nn.Module 的初始化方法
        super().__init__()
        # 将传入的 config 参数赋值给对象的 config 属性
        self.config = config
        # 创建一个 nn.ModuleList 对象，包含 config.num_hidden_layers 个 EsmLayer 对象
        self.layer = nn.ModuleList([EsmLayer(config) for _ in range(config.num_hidden_layers)])
        # 创建一个 LayerNorm 层，对隐藏状态进行归一化
        self.emb_layer_norm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 设置梯度检查点为 False
        self.gradient_checkpointing = False

    # 前向传播方法，接受多个输入参数
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
# 定义 EsmPooler 类，继承自 nn.Module
class EsmPooler(nn.Module):
    # 初始化方法，接受一个 config 参数
    def __init__(self, config):
        # 调用父类 nn.Module 的初始化方法
        super().__init__()
        # 创建一个线性层，输入输出维度均为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个 Tanh 激活函数
        self.activation = nn.Tanh()

    # 前向传播方法，接受一个输入参数 hidden_states，返回一个张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 提取 hidden_states 中的第一个 token 对应的隐藏状态
        first_token_tensor = hidden_states[:, 0]
        # 将提取的隐藏状态通过线性层和激活函数得到池化输出
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        # 返回池化输出
        return pooled_output


# 定义 EsmPreTrainedModel 类，继承自 PreTrainedModel 类
class EsmPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    # 设置 config_class 属性为 EsmConfig 类
    config_class = EsmConfig
    # 设置 base_model_prefix 属性为 "esm"
    base_model_prefix = "esm"
    # 设置 supports_gradient_checkpointing 属性为 True
    supports_gradient_checkpointing = True
    # 设置 _no_split_modules 属性为不可拆分的模块列表
    _no_split_modules = ["EsmLayer", "EsmFoldTriangularSelfAttentionBlock", "EsmEmbeddings"]

    # 初始化权重方法，接受一个 module 参数
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果 module 类型为 nn.Linear
        if isinstance(module, nn.Linear):
            # 对权重进行正态分布初始化
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置项，将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果 module 类型为 nn.Embedding
        elif isinstance(module, nn.Embedding):
            # 对权重进行正态分布初始化
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在填充索引，将其对应的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果 module 类型为 nn.LayerNorm
        elif isinstance(module, nn.LayerNorm):
            # 将偏置项初始化为零，权重初始化为 1.0
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

# 设置 ESM_START_DOCSTRING 字符串
ESM_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    # 使用它作为常规的 PyTorch 模块，并参考 PyTorch 文档，了解一切与通用使用和行为相关的事项。

    # 参数:
    #     config ([`EsmConfig`]): 包含模型所有参数的模型配置类。
    #         使用配置文件初始化不会加载与模型相关的权重，只会加载配置信息。
    #         查看 [`~PreTrainedModel.from_pretrained`] 方法来加载模型权重。
"""

ESM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare ESM Model transformer outputting raw hidden-states without any specific head on top.",
    ESM_START_DOCSTRING,
)
class EsmModel(EsmPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    # 定义 ESM 模型类，继承自 PreTrainedModel
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        # 初始化 ESM Embeddings，Encoder，和 Pooler
        self.embeddings = EsmEmbeddings(config)
        self.encoder = EsmEncoder(config)

        # 如果需要添加 Pooling Layer 则初始化 ESM Pooler
        self.pooler = EsmPooler(config) if add_pooling_layer else None

        # 初始化 ESM Contact Prediction Head
        self.contact_head = EsmContactPredictionHead(
            in_features=config.num_hidden_layers * config.num_attention_heads, bias=True
        )

        # 初始化权重并进行最终处理
        self.post_init()

    # 获取输入的嵌入层
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入的嵌入层
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # 剪枝模型的头部
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 前向传播方法
    @add_start_docstrings_to_model_forward(ESM_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 预测联系人函数，接受输入 tokens 和 attention_mask
    def predict_contacts(self, tokens, attention_mask):
        # 调用模型，传入 tokens 和 attention_mask，返回模型输出的注意力矩阵
        attns = self(tokens, attention_mask=attention_mask, return_dict=True, output_attentions=True).attentions
        # 将每个注意力矩阵堆叠起来，维度与原始模型布局相匹配
        attns = torch.stack(attns, dim=1)  # Matches the original model layout
        # 在原始模型中，对于填充标记的注意力完全被清零。
        # 这在大多数情况下并不重要，因为其他标记不会关注它们，
        # 但是对于接触预测任务而言很重要，因为它接受注意力作为输入，
        # 所以我们在这里必须模仿那种行为。
        # 将注意力矩阵乘以 attention_mask，以模拟原始模型中的效果
        attns *= attention_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        attns *= attention_mask.unsqueeze(1).unsqueeze(2).unsqueeze(4)
        # 调用联系人头部模型，传入 tokens 和调整后的 attentions，返回联系人预测结果
        return self.contact_head(tokens, attns)
# 使用指定的文档字符串构建一个带有语言建模头的 ESM 模型
@add_start_docstrings("""ESM Model with a `language modeling` head on top.""", ESM_START_DOCSTRING)
class EsmForMaskedLM(EsmPreTrainedModel):
    # 可以共享参数的关键字列表
    _tied_weights_keys = ["lm_head.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)

        # 如果配置为解码器，发出警告信息
        if config.is_decoder:
            logger.warning(
                "If you want to use `EsmForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 构建 ESM 模型，不包含池化层
        self.esm = EsmModel(config, add_pooling_layer=False)
        # 构建 ESM 语言模型头
        self.lm_head = EsmLMHead(config)

        # 初始化模型权重
        self.init_weights()

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.lm_head.decoder

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    # 定义前向传播方法，包含输入和输出的文档注释、代码样例注释
    @add_start_docstrings_to_model_forward(ESM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="<mask>",
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        # 确定是否返回字典格式的结果，如果未指定，则使用模型配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 ESM 模型处理输入，得到输出
        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 提取序列输出
        sequence_output = outputs[0]
        # 使用语言模型头部进行预测
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        # 如果有标签，则计算掩码语言模型损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            labels = labels.to(prediction_scores.device)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果不返回字典格式的结果，则返回元组
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 返回掩码语言模型的输出
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # 预测联系的函数
    def predict_contacts(self, tokens, attention_mask):
        return self.esm.predict_contacts(tokens, attention_mask=attention_mask)
class EsmLMHead(nn.Module):
    """ESM Head for masked language modeling."""

    def __init__(self, config):
        # 初始化 ESM 头部，用于掩码语言建模
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # 定义全连接层
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # 定义 LayerNorm 层

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)  # 定义线性层
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))  # 设置偏置向量

    def forward(self, features, **kwargs):
        x = self.dense(features)  # 全连接层操作
        x = gelu(x)  # 执行 GELU 激活函数
        x = self.layer_norm(x)  # LayerNorm 操作

        # 用偏置投影回词汇表大小
        x = self.decoder(x) + self.bias
        return x


@add_start_docstrings(
    """
    ESM Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    ESM_START_DOCSTRING,
)
class EsmForSequenceClassification(EsmPreTrainedModel):
    def __init__(self, config):
        # 初始化 ESM 序列分类/回归模型
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.esm = EsmModel(config, add_pooling_layer=False)  # 创建 ESM 模型对象
        self.classifier = EsmClassificationHead(config)  # 创建分类器对象

        self.init_weights()  # 初始化权重

    @add_start_docstrings_to_model_forward(ESM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    def forward(
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple, SequenceClassifierOutput]:
        """
        此函数定义了模型的前向传播过程。
    
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            用于计算序列分类/回归损失的标签。索引应在 `[0, ..., config.num_labels - 1]` 范围内。
            如果 `config.num_labels == 1`，则计算回归损失（均方损失）；
            如果 `config.num_labels > 1`，则计算分类损失（交叉熵损失）。
    
        """
        # 确定是否返回字典形式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # 使用 ESM 模型处理输入
        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取序列输出
        sequence_output = outputs[0]
        # 对序列输出进行分类
        logits = self.classifier(sequence_output)
    
        # 初始化损失
        loss = None
        # 如果存在标签，则计算损失
        if labels is not None:
            # 将标签转移到与 logits 相同的设备上
            labels = labels.to(logits.device)
    
            # 根据问题类型确定损失函数
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
                    # 对于单一标签回归问题，计算均方损失
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # 对于单一标签分类问题，使用交叉熵损失
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # 对于多标签分类问题，使用带 logits 的二元交叉熵损失
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
    
        # 如果不要求返回字典形式的输出，则返回元组
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
    
        # 返回 SequenceClassifierOutput 类型的对象，包括损失、logits、隐藏状态和注意力权重
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 定义一个带有标记分类头的 ESM 模型，用于命名实体识别等任务
class EsmForTokenClassification(EsmPreTrainedModel):
    def __init__(self, config):
        # 调用父类构造函数
        super().__init__(config)
        # 获取标签数量
        self.num_labels = config.num_labels

        # 创建 ESM 模型，不包含池化层
        self.esm = EsmModel(config, add_pooling_layer=False)
        # 随机失活层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 线性分类器
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重
        self.init_weights()

    # 前向传播函数
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 如果 return_dict 没提供则使用配置中定义的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取模型输出
        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0] # 获取序列输出

        sequence_output = self.dropout(sequence_output) # 序列输出经过随机失活层
        logits = self.classifier(sequence_output) # 序列输出经过分类器

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.to(logits.device) # 将标签移到相同设备
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1)) # 计算交叉熵损失

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output # 返回输出

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# 用于句子级分类任务的头部模块
class EsmClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    # 初始化模型，传入配置参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个全连接层，输入和输出大小都为隐藏层大小
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个dropout层，以隐藏层dropout概率作为参数
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 创建一个全连接层，输入大小为隐藏层大小，输出大小为标签的数量
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
    
    # 前向传播方法
    def forward(self, features, **kwargs):
        # 取features中的第一个token，即"<s>"，放入x中
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        # 对x进行dropout
        x = self.dropout(x)
        # 将x输入到全连接层中
        x = self.dense(x)
        # 对x使用双曲正切函数
        x = torch.tanh(x)
        # 对x进行dropout
        x = self.dropout(x)
        # 将x输入到输出的全连接层中
        x = self.out_proj(x)
        # 返回结果
        return x
# 从输入的 input_ids 中创建位置编号，替换非填充符号为它们对应的位置编号。位置编号从 padding_idx+1 开始。填充符号将被忽略。这是修改自 fairseq 的 `utils.make_positions`。

# 参数:
# - input_ids: torch.Tensor 输入的张量
# - padding_idx: int 填充符号的索引
# - past_key_values_length: int 过去的键值对长度，默认为0

def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    # 这里一系列的强制转换和类型转换被精心平衡，以便既能在 ONNX 导出中工作，也能在 XLA 中工作。
    # 创建一个与 input_ids 相同大小的张量，其中非填充符号的位置为1，填充符号的位置为0
    mask = input_ids.ne(padding_idx).int()
    # 计算累积非填充符号的数量，并将类型转换为与 mask 张量相同的类型，然后加上过去的键值对长度，最后乘以 mask 张量
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    # 最后将得到的位置编号转换为长整型，并将填充符号加回去
    return incremental_indices.long() + padding_idx
```