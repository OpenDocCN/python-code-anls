# `.\models\esm\modeling_esm.py`

```py
# 设置文件编码为 UTF-8

# 版权声明和许可证信息

# 导入必要的库和模块
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入各种辅助函数和类
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
from .configuration_esm import EsmConfig

# 获取 logger 实例
logger = logging.get_logger(__name__)

# 文档中使用的模型检查点名称
_CHECKPOINT_FOR_DOC = "facebook/esm2_t6_8M_UR50D"

# 文档中使用的配置文件名称
_CONFIG_FOR_DOC = "EsmConfig"

# ESM 预训练模型存档列表
ESM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/esm2_t6_8M_UR50D",
    "facebook/esm2_t12_35M_UR50D",
    # This is not a complete list of all ESM models!
    # See all ESM models at https://huggingface.co/models?filter=esm
]

# 定义一个函数，将输入张量沿着最后一个维度分成两部分，然后交换这两部分的顺序
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

# 应用旋转位置嵌入到输入张量中
def apply_rotary_pos_emb(x, cos, sin):
    # 限制余弦和正弦嵌入的长度与输入张量的前两个维度一致
    cos = cos[:, :, : x.shape[-2], :]
    sin = sin[:, :, : x.shape[-2], :]

    # 返回应用旋转位置嵌入后的结果张量
    return (x * cos) + (rotate_half(x) * sin)

# 实现原始 ESM 仓库中的 GELU 激活函数
def gelu(x):
    """
    This is the gelu implementation from the original ESM repo. Using F.gelu yields subtly wrong results.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

# 使张量在最后两个维度上对称化，用于接触预测
def symmetrize(x):
    "Make layer symmetric in final two dimensions, used for contact prediction."
    return x + x.transpose(-1, -2)

# 执行平均产品修正，用于接触预测
def average_product_correct(x):
    "Perform average product correct, used for contact prediction."
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)

    avg = a1 * a2
    avg.div_(a12)  # in-place to reduce memory
    normalized = x - avg
    return normalized

# 定义旋转嵌入类，基于 RoFormer 中的旋转位置嵌入实现
class RotaryEmbedding(torch.nn.Module):
    """
    Rotary position embeddings based on those in
    [RoFormer](https://huggingface.co/docs/transformers/model_doc/roformer). Query and keys are transformed by rotation
    matrices which depend on their relative positions.
    """
    def __init__(self, dim: int):
        super().__init__()
        # 生成并保存反频率缓冲区（非可训练）
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        inv_freq = inv_freq
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_dimension=2):
        seq_len = x.shape[seq_dimension]

        # 如果序列长度发生变化，或者在新设备上（可能由于追踪等原因），则重置表格
        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device:
            self._seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dimension], device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]

        return self._cos_cached, self._sin_cached

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 更新余弦和正弦表格，使用 k 张量的序列维度作为参数
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension=-2)

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
        super().__init__()
        self.in_features = in_features
        self.eos_idx = eos_idx
        # 定义一个线性层，用于执行 logistic 回归
        self.regression = nn.Linear(in_features, 1, bias)
        # 定义激活函数为 Sigmoid
        self.activation = nn.Sigmoid()

    def forward(self, tokens, attentions):
        # 移除 EOS 标记的注意力
        eos_mask = tokens.ne(self.eos_idx).to(attentions)
        eos_mask = eos_mask.unsqueeze(1) * eos_mask.unsqueeze(2)
        attentions = attentions * eos_mask[:, None, None, :, :]
        attentions = attentions[..., :-1, :-1]
        # 移除 CLS 标记的注意力
        attentions = attentions[..., 1:, 1:]
        batch_size, layers, heads, seqlen, _ = attentions.size()
        attentions = attentions.view(batch_size, layers * heads, seqlen, seqlen)

        # 特征：批次 x 通道 x 标记 x 标记（对称）
        attentions = attentions.to(
            self.regression.weight.device
        )  # 注意力始终是 float32，可能需要转换为 float16
        # 对注意力矩阵进行对称化处理和平均产品校正
        attentions = average_product_correct(symmetrize(attentions))
        # 将维度重新排列以匹配线性层的输入要求
        attentions = attentions.permute(0, 2, 3, 1)
        return self.activation(self.regression(attentions).squeeze(3))


class EsmEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config):
        super().__init__()
        # 词嵌入层，根据配置参数创建
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        if config.emb_layer_norm_before:
            # 如果配置中指定在嵌入之前进行层归一化，则初始化层归一化层
            self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.layer_norm = None
        # Dropout 层，用于随机置零输入张量的元素
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 位置嵌入类型，绝对或相对
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 在内存中创建位置 IDs 张量，用于序列中每个位置的索引
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

        self.padding_idx = config.pad_token_id
        # 位置嵌入层，根据配置参数创建
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )
        # 是否启用 token dropout
        self.token_dropout = config.token_dropout
        # Mask token 的 ID
        self.mask_token_id = config.mask_token_id

    def forward(
        self, input_ids=None, attention_mask=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
        ):
            # 如果未提供位置编码，但提供了输入的token ids，则从token ids创建位置编码，保留任何填充的token。
            if position_ids is None:
                if input_ids is not None:
                    position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
                else:
                    position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

            # 如果未提供输入的嵌入表示，则使用word_embeddings来生成。
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)

            # 如果希望在未来支持ESM-1（而不是1b！），则可能需要在这里支持一个embedding_scale因子。
            embeddings = inputs_embeds

            # Matt: ESM在MLM中有一个处理masking的略微不同的选项。如果token_dropout标志为False，则处理方式与BERT/RoBERTa相同。
            # 如果设置为True，则掩码token被视为已选择进行输入的dropout并将其置零。
            # 当训练期间没有掩码的token时，通过将嵌入乘以 (训练期间未掩码token的分数) / (样本中未掩码token的分数)，来补偿"mask-dropout"。
            # 这类似于dropout层在评估期间未实际丢弃值时缩减输出（或者在训练期间增加未丢弃的输出）的方式。
            if self.token_dropout:
                embeddings = embeddings.masked_fill((input_ids == self.mask_token_id).unsqueeze(-1), 0.0)
                mask_ratio_train = 0.15 * 0.8  # 在所有ESM模型训练中硬编码的比率
                src_lengths = attention_mask.sum(-1)
                mask_ratio_observed = (input_ids == self.mask_token_id).sum(-1).float() / src_lengths
                embeddings = (embeddings * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]).to(
                    embeddings.dtype
                )

            # 如果位置嵌入类型为"absolute"，则添加绝对位置嵌入到嵌入表示中。
            if self.position_embedding_type == "absolute":
                position_embeddings = self.position_embeddings(position_ids)
                embeddings = embeddings + position_embeddings

            # 如果层归一化函数不为None，则对嵌入表示进行层归一化。
            if self.layer_norm is not None:
                embeddings = self.layer_norm(embeddings)

            # 如果存在注意力遮罩，则将其应用于嵌入表示。
            if attention_mask is not None:
                embeddings = (embeddings * attention_mask.unsqueeze(-1)).to(embeddings.dtype)

            # Matt: 我认为这行代码从BERT中复制过来时出现了错误，暂时禁用它。
            # embeddings = self.dropout(embeddings)

            # 返回最终的嵌入表示。
            return embeddings
    # 根据输入的嵌入张量生成位置 ID。由于我们直接提供了嵌入向量，无法推断哪些是填充的，因此生成连续的位置 ID。

    # 获取输入嵌入张量的形状，去除最后一个维度（通常是 batch 维度）
    input_shape = inputs_embeds.size()[:-1]
    # 获取序列长度，即嵌入张量的第二个维度大小
    sequence_length = input_shape[1]

    # 使用 torch.arange 生成从 self.padding_idx + 1 到 sequence_length + self.padding_idx + 1 的整数序列
    # 结果类型为 long 型，设备为 inputs_embeds 的设备
    position_ids = torch.arange(
        self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
    )

    # 将 position_ids 在第0维度上增加一个维度，并在各维度上重复 input_shape 次数，以便与 inputs_embeds 的形状匹配
    return position_ids.unsqueeze(0).expand(input_shape)
# 定义一个自注意力模块，继承自 nn.Module
class EsmSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 检查隐藏大小是否能够被注意力头数整除，或者检查是否有嵌入大小属性
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 初始化注意力头数和每个注意力头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 定义用于查询、键、值的线性变换层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 定义用于dropout的层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        # 设置位置嵌入的类型，默认为绝对位置编码
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        
        # 如果位置编码类型是相对键或者相对键查询，则初始化距离嵌入层
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        
        # 如果位置编码类型是旋转类型，则初始化旋转嵌入层
        elif self.position_embedding_type == "rotary":
            self.rotary_embeddings = RotaryEmbedding(dim=self.attention_head_size)

        # 标记是否为解码器
        self.is_decoder = config.is_decoder

    # 将输入张量转换为分数矩阵
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
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
    def __init__(self, config):
        super().__init__()
        # 初始化 self 属性为 EsmSelfAttention 对象，使用给定的配置参数
        self.self = EsmSelfAttention(config)
        # 初始化 output 属性为 EsmSelfOutput 对象，使用给定的配置参数
        self.output = EsmSelfOutput(config)
        # 初始化 pruned_heads 为一个空集合，用于存储被剪枝的注意力头部索引
        self.pruned_heads = set()
        # 初始化 LayerNorm 属性为 nn.LayerNorm，使用给定的 hidden_size 和 layer_norm_eps 参数
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 调用 find_pruneable_heads_and_indices 函数找到可剪枝的头部及其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储被剪枝的头部索引
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

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
        # 对输入的 hidden_states 进行 LayerNorm 处理
        hidden_states_ln = self.LayerNorm(hidden_states)
        # 调用 self.self 的 forward 方法进行自注意力计算
        self_outputs = self.self(
            hidden_states_ln,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 将自注意力的输出与原始 hidden_states 应用 self.output 进行最终的输出
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出注意力信息，则将其加入到输出中
        outputs = (attention_output,) + self_outputs[1:]  # 如果需要输出注意力信息，则加入
        return outputs
class EsmIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个全连接层，将输入维度为 config.hidden_size 转换为 config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用全连接层进行前向传播，将隐藏状态 hidden_states 映射到 intermediate_size 的维度
        hidden_states = self.dense(hidden_states)
        # 使用 GELU 激活函数处理 hidden_states
        hidden_states = gelu(hidden_states)
        return hidden_states


class EsmOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个全连接层，将 intermediate_size 的输入转换为 hidden_size 的输出
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 初始化一个 dropout 层，以概率 config.hidden_dropout_prob 随机将输出置为 0
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # 使用全连接层进行前向传播，将 hidden_states 映射回 hidden_size 的维度
        hidden_states = self.dense(hidden_states)
        # 对输出进行 dropout 处理
        hidden_states = self.dropout(hidden_states)
        # 将 dropout 处理后的输出与输入 input_tensor 相加，实现残差连接
        hidden_states = hidden_states + input_tensor
        return hidden_states


class EsmLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 设置用于分块前向传播的 chunk_size_feed_forward
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 设置序列长度的维度为 1
        self.seq_len_dim = 1
        # 初始化自注意力层
        self.attention = EsmAttention(config)
        # 设置是否为解码器的标志位
        self.is_decoder = config.is_decoder
        # 设置是否添加跨层注意力的标志位
        self.add_cross_attention = config.add_cross_attention
        # 如果添加了跨层注意力，确保模型被用作解码器模型
        if self.add_cross_attention:
            if not self.is_decoder:
                raise RuntimeError(f"{self} should be used as a decoder model if cross attention is added")
            # 初始化跨层注意力层
            self.crossattention = EsmAttention(config)
        # 初始化中间层和输出层
        self.intermediate = EsmIntermediate(config)
        self.output = EsmOutput(config)
        # 初始化层归一化层
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

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
        # 对隐藏状态进行自注意力计算
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            )
        # 从自注意力输出中提取隐藏状态
        attention_output = self_attention_outputs[0]
        # 如果添加了跨层注意力，计算跨层注意力输出
        if self.add_cross_attention:
            cross_attention_outputs = self.crossattention(
                attention_output,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions=output_attentions,
                )
            attention_output = cross_attention_outputs[0]
        # 应用中间层和输出层处理
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        # 应用层归一化层处理输出
        layer_output = self.LayerNorm(layer_output + attention_output)
        outputs = (layer_output,) + self_attention_outputs[1:]  # 添加注意力输出信息
        if output_attentions:
            outputs = outputs + cross_attention_outputs[1:]  # 添加跨层注意力输出信息
        return outputs
        # 如果过去的键/值对存在，只保留自注意力缓存的前两个位置的值
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 执行自注意力计算，传入隐藏状态、注意力掩码、头掩码等参数
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # 获取自注意力计算的输出
        attention_output = self_attention_outputs[0]

        # 如果当前实例为解码器，最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            # 只保留自注意力计算输出中除了最后一个元组的部分
            outputs = self_attention_outputs[1:-1]
            # 提取自注意力的当前键/值对
            present_key_value = self_attention_outputs[-1]
        else:
            # 否则，包含所有自注意力计算输出（如果输出注意力权重的话）
            outputs = self_attention_outputs[1:]

        # 初始化交叉注意力的当前键/值对为 None
        cross_attn_present_key_value = None
        # 如果当前实例是解码器且存在编码器的隐藏状态
        if self.is_decoder and encoder_hidden_states is not None:
            # 检查是否存在交叉注意力层，如果没有则引发错误
            if not hasattr(self, "crossattention"):
                raise AttributeError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated"
                    " with cross-attention layers by setting `config.add_cross_attention=True`"
                )

            # 如果过去的键/值对存在，只保留交叉注意力缓存的最后两个位置的值
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 执行交叉注意力计算，传入自注意力的输出、注意力掩码、头掩码、编码器的隐藏状态等参数
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            # 获取交叉注意力计算的输出
            attention_output = cross_attention_outputs[0]
            # 将交叉注意力计算输出中除了最后一个元组的部分添加到自注意力计算输出中
            outputs = outputs + cross_attention_outputs[1:-1]

            # 将交叉注意力的当前键/值对添加到自注意力的当前键/值对中的最后两个位置
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 使用注意力输出执行前馈网络的计算
        layer_output = self.feed_forward_chunk(attention_output)

        # 将前馈网络的输出添加到输出元组中
        outputs = (layer_output,) + outputs

        # 如果当前实例是解码器，将注意力的键/值对作为输出元组的最后一个元素返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)
        return outputs
    ```
class EsmEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # 初始化模型配置
        self.layer = nn.ModuleList([EsmLayer(config) for _ in range(config.num_hidden_layers)])  # 创建多层ESM层
        self.emb_layer_norm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # 对嵌入进行层归一化
        self.gradient_checkpointing = False  # 梯度检查点标记为False，表示不使用梯度检查点

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
        # 返回字典类型的结果



class EsmPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # 密集连接层
        self.activation = nn.Tanh()  # 激活函数为Tanh

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过获取第一个token对应的隐藏状态来“池化”模型
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)  # 使用线性层进行池化
        pooled_output = self.activation(pooled_output)  # 应用Tanh激活函数
        return pooled_output



class EsmPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = EsmConfig  # 配置类为EsmConfig
    base_model_prefix = "esm"  # 基础模型前缀为"esm"
    supports_gradient_checkpointing = True  # 支持梯度检查点

    _no_split_modules = ["EsmLayer", "EsmFoldTriangularSelfAttentionBlock", "EsmEmbeddings"]
    # 不拆分的模块列表

    # 初始化权重的函数，根据不同类型的模块进行初始化
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化线性层的权重，偏置置零
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化嵌入层的权重，偏置置零
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 层归一化的权重初始化，偏置置零，缩放参数置1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


ESM_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    # 将其作为常规的 PyTorch 模块使用，并参考 PyTorch 文档以获取有关一般用法和行为的所有信息。

    Parameters:
        # config ([`EsmConfig`]): 模型配置类，包含模型的所有参数。
        # 初始化时使用配置文件不会加载与模型相关的权重，只加载配置信息。
        # 可以查看 [`~PreTrainedModel.from_pretrained`] 方法来加载模型权重。
# 定义了一个原始的 ESM 模型类，继承自 EsmPreTrainedModel
@add_start_docstrings(
    "The bare ESM Model transformer outputting raw hidden-states without any specific head on top.",
    ESM_START_DOCSTRING,
)
class EsmModel(EsmPreTrainedModel):
    """
    ESM 模型类，可以作为编码器（只包含自注意力）或解码器使用，后者则在自注意力层之间添加了一层交叉注意力，遵循了 Ashish Vaswani 等人在《Attention is all you need》中描述的架构。
    """
    """
    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    # 根据给定的配置初始化模型，可选择添加一个池化层
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        # 初始化嵌入层
        self.embeddings = EsmEmbeddings(config)
        # 初始化编码器
        self.encoder = EsmEncoder(config)

        # 如果设置了添加池化层，则初始化池化层，否则为None
        self.pooler = EsmPooler(config) if add_pooling_layer else None

        # 初始化联系预测头部
        self.contact_head = EsmContactPredictionHead(
            in_features=config.num_hidden_layers * config.num_attention_heads, bias=True
        )

        # 执行初始化后的权重和最终处理
        self.post_init()

    # 返回输入嵌入层
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入嵌入层
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # 剪枝模型中的注意力头部
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 前向传播函数，接受多种输入和参数
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
    # 预测联系人函数，接受 tokens 和 attention_mask 作为输入
    def predict_contacts(self, tokens, attention_mask):
        # 使用模型进行推断，返回注意力矩阵列表
        attns = self(tokens, attention_mask=attention_mask, return_dict=True, output_attentions=True).attentions
        # 将注意力矩阵堆叠起来，以匹配原始模型的布局
        attns = torch.stack(attns, dim=1)  # Matches the original model layout
        
        # 在原始模型中，对于填充的 token，其注意力被完全置零。
        # 大多数情况下这不会有影响，因为其他 token 不会关注它们，
        # 但对于需要将注意力作为输入的联系人预测任务而言，这一点非常重要，
        # 因此我们需要在这里模仿这种处理方式。
        
        # 将注意力矩阵乘以 attention_mask，以将填充的 token 的注意力置零
        attns *= attention_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        attns *= attention_mask.unsqueeze(1).unsqueeze(2).unsqueeze(4)
        
        # 使用联系人头部模型进行联系人预测，并返回结果
        return self.contact_head(tokens, attns)
# 定义一个 EsmForMaskedLM 类，继承自 EsmPreTrainedModel 类，并添加了语言建模头部
@add_start_docstrings("""ESM Model with a `language modeling` head on top.""", ESM_START_DOCSTRING)
class EsmForMaskedLM(EsmPreTrainedModel):
    # 定义了与 lm_head.decoder.weight 相关的权重绑定键
    _tied_weights_keys = ["lm_head.decoder.weight"]

    # 初始化方法，接收一个配置对象 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 如果配置中 is_decoder 为 True，则发出警告信息
        if config.is_decoder:
            logger.warning(
                "If you want to use `EsmForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 创建 EsmModel 对象，不添加池化层
        self.esm = EsmModel(config, add_pooling_layer=False)
        # 创建 EsmLMHead 对象
        self.lm_head = EsmLMHead(config)

        # 初始化模型权重
        self.init_weights()

    # 返回 lm_head.decoder 对象，用于输出嵌入
    def get_output_embeddings(self):
        return self.lm_head.decoder

    # 设置 lm_head.decoder 的新嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    # 前向传播方法，接收多个输入参数并返回输出
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
        # 以下为输入参数的详细说明
    ):
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        # 根据参数 `return_dict` 确定是否返回字典类型的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用模型 `esm` 进行前向传播，传入各种输入参数
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
        # 获取模型输出的序列输出
        sequence_output = outputs[0]
        # 对序列输出进行预测得到预测分数
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        # 如果存在标签，则计算掩码语言建模损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            # 将标签移动到与预测分数相同的设备上
            labels = labels.to(prediction_scores.device)
            # 计算掩码语言建模的损失
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果不返回字典类型的输出，则组织最终输出格式
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 返回掩码语言建模任务的输出，包括损失、预测分数、隐藏状态和注意力权重
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def predict_contacts(self, tokens, attention_mask):
        # 调用模型 `esm` 的方法进行接触预测
        return self.esm.predict_contacts(tokens, attention_mask=attention_mask)
class EsmLMHead(nn.Module):
    """ESM Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        # 定义一个全连接层，将输入特征空间映射到隐藏大小的空间
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # Layer normalization，对输入进行归一化处理
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 用于输出，将隐藏大小映射回词汇表大小的线性层，无偏置
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 定义一个偏置参数，长度为词汇表大小，用于模型输出的偏移
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, features, **kwargs):
        # 前向传播函数
        x = self.dense(features)  # 全连接层映射
        x = gelu(x)  # 使用 GELU 激活函数
        x = self.layer_norm(x)  # Layer normalization 归一化处理

        # 用线性层映射回词汇表大小，并加上偏置
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
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        # ESM 模型主体部分，不添加池化层
        self.esm = EsmModel(config, add_pooling_layer=False)
        # 分类头部，用于序列分类任务
        self.classifier = EsmClassificationHead(config)

        self.init_weights()  # 初始化模型权重

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
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 如果 return_dict 为 None，则使用 self.config.use_return_dict 决定是否返回字典形式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 ESM 模型进行前向传播，获取模型的输出
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
        # 从模型输出中获取序列输出
        sequence_output = outputs[0]
        # 将序列输出输入分类器，得到预测 logits
        logits = self.classifier(sequence_output)

        # 初始化损失为 None
        loss = None
        # 如果存在 labels，则计算损失
        if labels is not None:
            # 将 labels 移动到 logits 所在的设备上
            labels = labels.to(logits.device)

            # 根据问题类型确定问题类型（"regression", "single_label_classification", "multi_label_classification"）
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
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果不要求返回字典形式的输出，则按元组形式返回结果
        if not return_dict:
            output = (logits,) + outputs[2:]  # 将 logits 和其他输出组成元组
            return ((loss,) + output) if loss is not None else output  # 如果有损失，则将损失与输出一起返回，否则只返回输出

        # 返回 SequenceClassifierOutput 对象，包括损失、logits、隐藏状态和注意力权重
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    ESM Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    ESM_START_DOCSTRING,
)



class EsmForTokenClassification(EsmPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # 初始化 ESM 模型，不添加池化层
        self.esm = EsmModel(config, add_pooling_layer=False)
        
        # Dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # 分类器，将隐藏状态映射到标签数的线性层
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化模型权重
        self.init_weights()



    @add_start_docstrings_to_model_forward(ESM_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
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
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        
        # 确定是否返回字典类型的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取 ESM 模型的输出
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

        # 应用 Dropout 层
        sequence_output = self.dropout(sequence_output)
        
        # 使用分类器将序列输出映射到标签空间
        logits = self.classifier(sequence_output)

        # 计算损失
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()

            # 将标签移到与 logits 相同的设备上
            labels = labels.to(logits.device)
            # 计算交叉熵损失
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不返回字典，则以元组形式返回输出
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TokenClassifierOutput 对象
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



class EsmClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    # 初始化函数，用于创建一个新的神经网络模型实例
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个全连接层，输入和输出维度都是 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个 Dropout 层，使用 config.hidden_dropout_prob 作为丢弃概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 创建一个全连接层，输入维度是 config.hidden_size，输出维度是 config.num_labels
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    # 前向传播函数，定义了数据从输入到输出的流程
    def forward(self, features, **kwargs):
        # 取 features 的第一个位置的数据，通常表示起始 token（如 [CLS]）
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        # 对取出的数据应用 Dropout，随机部分神经元失活，防止过拟合
        x = self.dropout(x)
        # 将数据通过全连接层 self.dense 进行线性变换
        x = self.dense(x)
        # 对变换后的数据应用双曲正切函数进行非线性变换
        x = torch.tanh(x)
        # 再次应用 Dropout 层，进一步随机失活神经元
        x = self.dropout(x)
        # 将数据通过全连接层 self.out_proj 进行线性变换，得到最终的输出结果
        x = self.out_proj(x)
        # 返回神经网络模型的输出结果
        return x
def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        input_ids: torch.Tensor, input tensor containing token IDs
        padding_idx: int, the index of padding tokens in input_ids
        past_key_values_length: int, optional, length of past key values for incremental processing

    Returns:
        torch.Tensor, tensor of position IDs corresponding to input_ids
    """
    # 创建一个掩码，标记非填充符号的位置为1，填充符号为0
    mask = input_ids.ne(padding_idx).int()
    # 计算每个非填充符号的位置编号，位置编号从 padding_idx+1 开始，乘以掩码以忽略填充符号
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    # 将位置编号转换为长整型并加上 padding_idx，得到最终的位置 ID
    return incremental_indices.long() + padding_idx
```