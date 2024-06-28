# `.\models\deberta\modeling_deberta.py`

```py
# coding=utf-8
# 版权 2020 年 Microsoft 和 Hugging Face Inc. 团队。
#
# 根据 Apache 许可证 2.0 版本（"许可证"）授权;
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按"原样"分发，
# 没有任何明示或暗示的保证或条件。
# 有关详细信息，请参阅许可证。

""" PyTorch DeBERTa 模型。"""

from collections.abc import Sequence
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import softmax_backward_data
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_deberta import DebertaConfig

# 获取 logger 实例
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = "DebertaConfig"
_CHECKPOINT_FOR_DOC = "microsoft/deberta-base"

# Masked LM 的文档字符串
_CHECKPOINT_FOR_MASKED_LM = "lsanochkin/deberta-large-feedback"
_MASKED_LM_EXPECTED_OUTPUT = "' Paris'"
_MASKED_LM_EXPECTED_LOSS = "0.54"

# QuestionAnswering 的文档字符串
_CHECKPOINT_FOR_QA = "Palak/microsoft_deberta-large_squad"
_QA_EXPECTED_OUTPUT = "' a nice puppet'"
_QA_EXPECTED_LOSS = 0.14
_QA_TARGET_START_INDEX = 12
_QA_TARGET_END_INDEX = 14

# 预训练模型存档列表
DEBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/deberta-base",
    "microsoft/deberta-large",
    "microsoft/deberta-xlarge",
    "microsoft/deberta-base-mnli",
    "microsoft/deberta-large-mnli",
    "microsoft/deberta-xlarge-mnli",
]


class ContextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 线性层，用于池化隐藏状态
        self.dense = nn.Linear(config.pooler_hidden_size, config.pooler_hidden_size)
        # 稳定的 Dropout 层，用于池化层输出
        self.dropout = StableDropout(config.pooler_dropout)
        self.config = config

    def forward(self, hidden_states):
        # 通过获取第一个 token 对应的隐藏状态来"池化"模型。

        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = ACT2FN[self.config.pooler_hidden_act](pooled_output)
        return pooled_output

    @property
    def output_dim(self):
        # 返回输出维度，即配置中的隐藏大小
        return self.config.hidden_size


class XSoftmax(torch.autograd.Function):
    """
    优化了内存的 Masked Softmax 实现
    """
    @staticmethod
    def forward(self, input, mask, dim):
        # 设置对象的维度属性为指定的 softmax 维度
        self.dim = dim
        # 计算反转后的掩码，将 mask 张量转换为布尔类型，然后取反
        rmask = ~(mask.to(torch.bool))

        # 使用最小值填充输入张量中掩码位置的元素
        output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
        # 在指定维度上应用 softmax 操作
        output = torch.softmax(output, self.dim)
        # 将 softmax 结果中掩码位置的元素置为 0
        output.masked_fill_(rmask, 0)
        # 保存输出张量以备反向传播使用
        self.save_for_backward(output)
        # 返回经 softmax 处理后的输出张量
        return output

    @staticmethod
    def backward(self, grad_output):
        # 从保存的张量中获取输出
        (output,) = self.saved_tensors
        # 调用 softmax 反向传播函数计算输入的梯度
        inputGrad = softmax_backward_data(self, grad_output, output, self.dim, output)
        # 返回输入梯度及其余两个 None
        return inputGrad, None, None

    @staticmethod
    def symbolic(g, self, mask, dim):
        # 导入符号化帮助函数和符号化操作集
        import torch.onnx.symbolic_helper as sym_help
        from torch.onnx.symbolic_opset9 import masked_fill, softmax

        # 将 mask 转换为 long 类型并取其相反值作为 r_mask
        mask_cast_value = g.op("Cast", mask, to_i=sym_help.cast_pytorch_to_onnx["Long"])
        r_mask = g.op(
            "Cast",
            g.op("Sub", g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64)), mask_cast_value),
            to_i=sym_help.cast_pytorch_to_onnx["Bool"],
        )
        # 使用最小值填充 self 中 r_mask 的位置
        output = masked_fill(
            g, self, r_mask, g.op("Constant", value_t=torch.tensor(torch.finfo(self.type().dtype()).min))
        )
        # 在 dim 维度上应用 softmax
        output = softmax(g, output, dim)
        # 将输出中 r_mask 的位置置为 0
        return masked_fill(g, output, r_mask, g.op("Constant", value_t=torch.tensor(0, dtype=torch.bool)))
class DropoutContext(object):
    # 定义一个 DropoutContext 类，用于保存 dropout 相关的上下文信息
    def __init__(self):
        # 初始化 dropout 概率为 0
        self.dropout = 0
        # 初始化掩码为 None
        self.mask = None
        # 初始化缩放系数为 1
        self.scale = 1
        # 是否重用掩码，默认为 True
        self.reuse_mask = True


def get_mask(input, local_context):
    # 根据传入的 local_context 类型判断是否为 DropoutContext 类型
    if not isinstance(local_context, DropoutContext):
        # 如果不是 DropoutContext 类型，则将 local_context 视为 dropout 概率
        dropout = local_context
        # 掩码初始化为 None
        mask = None
    else:
        # 如果是 DropoutContext 类型，从 local_context 中获取 dropout 概率
        dropout = local_context.dropout
        # 将 dropout 乘以缩放系数
        dropout *= local_context.scale
        # 如果允许重用掩码，则获取 local_context 中的掩码；否则初始化掩码为 None
        mask = local_context.mask if local_context.reuse_mask else None

    # 如果 dropout 大于 0 且掩码为 None，则根据输入张量 input 的形状生成掩码
    if dropout > 0 and mask is None:
        # 使用 Bernoulli 分布生成与 input 同样形状的掩码，并转换为布尔型
        mask = (1 - torch.empty_like(input).bernoulli_(1 - dropout)).to(torch.bool)

    # 如果 local_context 是 DropoutContext 类型且其掩码为 None，则将生成的掩码保存到 local_context 中
    if isinstance(local_context, DropoutContext):
        if local_context.mask is None:
            local_context.mask = mask

    # 返回生成的掩码和 dropout 概率
    return mask, dropout


class XDropout(torch.autograd.Function):
    """Optimized dropout function to save computation and memory by using mask operation instead of multiplication."""

    @staticmethod
    def forward(ctx, input, local_ctx):
        # 调用 get_mask 函数获取掩码和 dropout 概率
        mask, dropout = get_mask(input, local_ctx)
        # 计算缩放系数
        ctx.scale = 1.0 / (1 - dropout)
        # 如果 dropout 概率大于 0，则将输入张量 input 中掩码为 True 的元素置为 0，并乘以缩放系数
        if dropout > 0:
            # 保存掩码以备反向传播使用
            ctx.save_for_backward(mask)
            return input.masked_fill(mask, 0) * ctx.scale
        else:
            # 如果 dropout 概率为 0，则直接返回输入张量 input
            return input

    @staticmethod
    def backward(ctx, grad_output):
        # 如果缩放系数大于 1，则从 ctx 中恢复保存的掩码，并将梯度乘以缩放系数
        if ctx.scale > 1:
            (mask,) = ctx.saved_tensors
            return grad_output.masked_fill(mask, 0) * ctx.scale, None
        else:
            # 如果缩放系数不大于 1，则直接返回梯度
            return grad_output, None

    @staticmethod
    def symbolic(g: torch._C.Graph, input: torch._C.Value, local_ctx: Union[float, DropoutContext]) -> torch._C.Value:
        from torch.onnx import symbolic_opset12

        dropout_p = local_ctx
        # 如果 local_ctx 是 DropoutContext 类型，则从中获取 dropout 概率
        if isinstance(local_ctx, DropoutContext):
            dropout_p = local_ctx.dropout
        # 在训练时使用 StableDropout，故设置 train=True
        train = True
        # TODO: 应检查 opset_version 是否大于 12，暂无法良好实现，如在 https://github.com/pytorch/pytorch/issues/78391 修复后，执行：
        # if opset_version < 12:
        #   return torch.onnx.symbolic_opset9.dropout(g, input, dropout_p, train)
        # 使用 symbolic_opset12 中的 dropout 符号化函数
        return symbolic_opset12.dropout(g, input, dropout_p, train)


class StableDropout(nn.Module):
    """
    Optimized dropout module for stabilizing the training

    Args:
        drop_prob (float): the dropout probabilities
    """

    def __init__(self, drop_prob):
        super().__init__()
        # 初始化 dropout 概率
        self.drop_prob = drop_prob
        # 计数器初始化为 0
        self.count = 0
        # 上下文栈初始化为 None
        self.context_stack = None

    def forward(self, x):
        """
        Call the module

        Args:
            x (`torch.tensor`): The input tensor to apply dropout
        """
        # 如果处于训练模式且 dropout 概率大于 0，则调用 XDropout 的 apply 方法应用 dropout
        if self.training and self.drop_prob > 0:
            return XDropout.apply(x, self.get_context())
        # 否则直接返回输入张量 x
        return x
    # 重置对象的计数器和上下文堆栈
    def clear_context(self):
        self.count = 0
        self.context_stack = None
    
    # 初始化上下文堆栈，设置重用掩码和缩放比例
    def init_context(self, reuse_mask=True, scale=1):
        # 如果上下文堆栈为空，则初始化为空列表
        if self.context_stack is None:
            self.context_stack = []
        # 重置计数器
        self.count = 0
        # 遍历上下文堆栈中的每个上下文对象，并设置其重用掩码和缩放比例
        for c in self.context_stack:
            c.reuse_mask = reuse_mask
            c.scale = scale
    
    # 获取当前上下文对象，并设置丢弃概率
    def get_context(self):
        # 如果上下文堆栈不为空
        if self.context_stack is not None:
            # 如果计数超出了堆栈长度，添加一个新的丢弃上下文对象到堆栈中
            if self.count >= len(self.context_stack):
                self.context_stack.append(DropoutContext())
            # 获取当前计数对应的上下文对象
            ctx = self.context_stack[self.count]
            ctx.dropout = self.drop_prob  # 设置丢弃概率
            self.count += 1  # 计数器自增
            return ctx  # 返回获取的上下文对象
        else:
            return self.drop_prob  # 如果上下文堆栈为空，则返回丢弃概率本身
# 定义一个 Deberta 模型的中间层，继承自 nn.Module 类
class DebertaIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，将输入维度从 config.hidden_size 转换为 config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果 config.hidden_act 是字符串，则使用预定义的激活函数 ACT2FN[config.hidden_act]
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            # 否则直接使用 config.hidden_act 作为激活函数
            self.intermediate_act_fn = config.hidden_act
    # 定义一个前向传播方法，接受隐藏状态张量作为输入，并返回处理后的张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态张量传递给全连接层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的张量应用中间激活函数，例如ReLU等
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回经过线性变换和激活函数处理后的张量作为输出
        return hidden_states
# 定义一个名为 DebertaOutput 的神经网络模块，继承自 nn.Module 类
class DebertaOutput(nn.Module):
    # 初始化方法，接受一个 config 参数
    def __init__(self, config):
        super().__init__()
        # 创建一个线性层，输入大小为 config.intermediate_size，输出大小为 config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个 DebertaLayerNorm 层，输入大小为 config.hidden_size，使用给定的 layer_norm_eps 进行归一化
        self.LayerNorm = DebertaLayerNorm(config.hidden_size, config.layer_norm_eps)
        # 创建一个 StableDropout 层，使用给定的 hidden_dropout_prob 进行稳定的随机失活
        self.dropout = StableDropout(config.hidden_dropout_prob)
        # 保存 config 参数到当前对象中
        self.config = config

    # 前向传播方法，接受 hidden_states 和 input_tensor 作为输入
    def forward(self, hidden_states, input_tensor):
        # 将 hidden_states 输入至 self.dense 线性层
        hidden_states = self.dense(hidden_states)
        # 对 hidden_states 进行 dropout 处理
        hidden_states = self.dropout(hidden_states)
        # 将处理后的 hidden_states 和 input_tensor 相加，然后输入至 self.LayerNorm 层
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的 hidden_states
        return hidden_states


# 定义一个名为 DebertaLayer 的神经网络模块，继承自 nn.Module 类
class DebertaLayer(nn.Module):
    # 初始化方法，接受一个 config 参数
    def __init__(self, config):
        super().__init__()
        # 创建一个 DebertaAttention 层，使用给定的 config 参数
        self.attention = DebertaAttention(config)
        # 创建一个 DebertaIntermediate 层，使用给定的 config 参数
        self.intermediate = DebertaIntermediate(config)
        # 创建一个 DebertaOutput 层，使用给定的 config 参数
        self.output = DebertaOutput(config)

    # 前向传播方法，接受多个参数，包括 hidden_states、attention_mask 等
    def forward(
        self,
        hidden_states,
        attention_mask,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
        output_attentions=False,
    ):
        # 将 hidden_states 等参数输入至 self.attention 层进行处理，获取 attention_output
        attention_output = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        # 如果设置了 output_attentions 参数为 True，则从 attention_output 中获取 att_matrix
        if output_attentions:
            attention_output, att_matrix = attention_output
        # 将 attention_output 输入至 self.intermediate 层进行处理，获取 intermediate_output
        intermediate_output = self.intermediate(attention_output)
        # 将 intermediate_output 和 attention_output 输入至 self.output 层进行处理，获取 layer_output
        layer_output = self.output(intermediate_output, attention_output)
        # 如果设置了 output_attentions 参数为 True，则返回 layer_output 和 att_matrix
        if output_attentions:
            return (layer_output, att_matrix)
        else:
            # 否则，返回 layer_output
            return layer_output


# 定义一个名为 DebertaEncoder 的神经网络模块，继承自 nn.Module 类
class DebertaEncoder(nn.Module):
    """Modified BertEncoder with relative position bias support"""

    # 初始化方法，接受一个 config 参数
    def __init__(self, config):
        super().__init__()
        # 创建一个 nn.ModuleList，其中包含 config.num_hidden_layers 个 DebertaLayer 层对象
        self.layer = nn.ModuleList([DebertaLayer(config) for _ in range(config.num_hidden_layers)])
        # 检查是否需要支持相对位置偏置
        self.relative_attention = getattr(config, "relative_attention", False)
        # 如果启用了相对位置偏置
        if self.relative_attention:
            # 获取最大的相对位置距离，并设置为 max_relative_positions，如果小于 1，则使用 config.max_position_embeddings
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            # 创建一个相对位置嵌入的 Embedding 层，大小为 max_relative_positions * 2，维度为 config.hidden_size
            self.rel_embeddings = nn.Embedding(self.max_relative_positions * 2, config.hidden_size)
        # 禁用梯度检查点
        self.gradient_checkpointing = False

    # 获取相对位置嵌入的方法
    def get_rel_embedding(self):
        # 如果启用了相对位置注意力，则返回 rel_embeddings 的权重，否则返回 None
        rel_embeddings = self.rel_embeddings.weight if self.relative_attention else None
        return rel_embeddings

    # 获取注意力掩码的方法，输入参数 attention_mask
    def get_attention_mask(self, attention_mask):
        # 如果 attention_mask 的维度小于等于 2
        if attention_mask.dim() <= 2:
            # 对 attention_mask 进行扩展，增加一个维度在第二和第三个位置
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # 通过扩展的 attention_mask 和自身的点积来生成新的 attention_mask
            attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
        # 如果 attention_mask 的维度为 3
        elif attention_mask.dim() == 3:
            # 在第二个位置增加一个维度
            attention_mask = attention_mask.unsqueeze(1)

        # 返回处理后的 attention_mask
        return attention_mask
    # 如果启用了相对位置注意力并且未提供相对位置参数，则根据查询状态和隐藏状态的维度构建相对位置信息
    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        if self.relative_attention and relative_pos is None:
            q = query_states.size(-2) if query_states is not None else hidden_states.size(-2)
            relative_pos = build_relative_position(q, hidden_states.size(-2), hidden_states.device)
        return relative_pos

    # Transformer 模型的前向传播函数
    def forward(
        self,
        hidden_states,
        attention_mask,
        output_hidden_states=True,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        return_dict=True,
    ):
        # 获取注意力掩码
        attention_mask = self.get_attention_mask(attention_mask)
        # 获取相对位置信息
        relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)

        # 初始化保存所有隐藏状态和注意力分数的空元组
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # 如果 hidden_states 是 Sequence 对象，则取其第一个元素作为 next_kv，否则直接使用 hidden_states
        if isinstance(hidden_states, Sequence):
            next_kv = hidden_states[0]
        else:
            next_kv = hidden_states
        
        # 获取相对位置嵌入
        rel_embeddings = self.get_rel_embedding()

        # 遍历每个 Transformer 层
        for i, layer_module in enumerate(self.layer):
            # 如果输出隐藏状态，则将当前隐藏状态加入 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果启用了梯度检查点和处于训练模式，则调用梯度检查点函数，否则直接调用当前层的 forward 方法
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    next_kv,
                    attention_mask,
                    query_states,
                    relative_pos,
                    rel_embeddings,
                    output_attentions,
                )
            else:
                hidden_states = layer_module(
                    next_kv,
                    attention_mask,
                    query_states=query_states,
                    relative_pos=relative_pos,
                    rel_embeddings=rel_embeddings,
                    output_attentions=output_attentions,
                )

            # 如果输出注意力分数，则解压 hidden_states 为 hidden_states 和 att_m
            if output_attentions:
                hidden_states, att_m = hidden_states

            # 更新 query_states
            if query_states is not None:
                query_states = hidden_states
                # 如果 hidden_states 是 Sequence 对象，则更新 next_kv 为下一个隐藏状态
                if isinstance(hidden_states, Sequence):
                    next_kv = hidden_states[i + 1] if i + 1 < len(self.layer) else None
            else:
                next_kv = hidden_states

            # 如果输出注意力分数，则将当前层的 att_m 加入 all_attentions 中
            if output_attentions:
                all_attentions = all_attentions + (att_m,)

        # 最后一个 Transformer 层的隐藏状态加入 all_hidden_states
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典形式的结果，则按顺序返回非空的结果元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        # 返回字典形式的 BaseModelOutput 结果
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )
# 使用给定的 query_size 和 device 创建一个长为 query_size 的长整型张量 q_ids
q_ids = torch.arange(query_size, dtype=torch.long, device=device)

# 使用给定的 key_size 和 device 创建一个长为 key_size 的长整型张量 k_ids
k_ids = torch.arange(key_size, dtype=torch.long, device=device)

# 计算相对位置张量 rel_pos_ids，其维度为 [query_size, key_size]，其中每个元素为对应的 query 和 key 的相对位置差值
rel_pos_ids = q_ids[:, None] - k_ids.view(1, -1).repeat(query_size, 1)

# 截取 rel_pos_ids 的子集，保留前 query_size 行，其维度变为 [1, query_size, key_size]
rel_pos_ids = rel_pos_ids[:query_size, :]

# 将 rel_pos_ids 维度扩展为 [1, query_size, key_size]，以符合函数返回的期望输出形状
rel_pos_ids = rel_pos_ids.unsqueeze(0)
return rel_pos_ids

# 将输入的 c2p_pos 张量扩展为 [query_layer.size(0), query_layer.size(1), query_layer.size(2), relative_pos.size(-1)]
@torch.jit.script
def c2p_dynamic_expand(c2p_pos, query_layer, relative_pos):
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), query_layer.size(2), relative_pos.size(-1)])

# 将输入的 c2p_pos 张量扩展为 [query_layer.size(0), query_layer.size(1), key_layer.size(-2), key_layer.size(-2)]
@torch.jit.script
def p2c_dynamic_expand(c2p_pos, query_layer, key_layer):
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), key_layer.size(-2), key_layer.size(-2)])

# 将输入的 pos_index 张量扩展为 [p2c_att.size()[0], p2c_att.size()[1], pos_index.size(-2), key_layer.size(-2)]
@torch.jit.script
def pos_dynamic_expand(pos_index, p2c_att, key_layer):
    return pos_index.expand(p2c_att.size()[:2] + (pos_index.size(-2), key_layer.size(-2)))

class DisentangledSelfAttention(nn.Module):
    """
    Disentangled self-attention module

    Parameters:
        config (`str`):
            A model config class instance with the configuration to build a new model. The schema is similar to
            *BertConfig*, for more details, please refer [`DebertaConfig`]

    """
    def __init__(self, config):
        super().__init__()
        # 检查隐藏层大小是否是注意力头数的倍数，如果不是则抛出数值错误异常
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        
        # 初始化注意力头数和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # 创建输入到投影层的线性变换，输出维度是注意力头数的三倍
        self.in_proj = nn.Linear(config.hidden_size, self.all_head_size * 3, bias=False)
        
        # 初始化注意力头的偏置参数
        self.q_bias = nn.Parameter(torch.zeros((self.all_head_size), dtype=torch.float))
        self.v_bias = nn.Parameter(torch.zeros((self.all_head_size), dtype=torch.float))
        
        # 根据配置初始化位置注意力类型列表
        self.pos_att_type = config.pos_att_type if config.pos_att_type is not None else []
        
        # 是否启用相对位置注意力和对话头机制
        self.relative_attention = getattr(config, "relative_attention", False)
        self.talking_head = getattr(config, "talking_head", False)
        
        # 如果启用了对话头机制，则初始化对话头的线性投影层
        if self.talking_head:
            self.head_logits_proj = nn.Linear(config.num_attention_heads, config.num_attention_heads, bias=False)
            self.head_weights_proj = nn.Linear(config.num_attention_heads, config.num_attention_heads, bias=False)
        
        # 如果启用了相对位置注意力，则根据配置初始化相关参数
        if self.relative_attention:
            # 最大相对位置，默认为配置中的最大相对位置或者位置嵌入的最大位置数
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            # 使用稳定的 Dropout 初始化位置 Dropout 层
            self.pos_dropout = StableDropout(config.hidden_dropout_prob)
            
            # 如果 pos_att_type 包含 "c2p"，则初始化位置投影层
            if "c2p" in self.pos_att_type:
                self.pos_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
            # 如果 pos_att_type 包含 "p2c"，则初始化位置查询投影层
            if "p2c" in self.pos_att_type:
                self.pos_q_proj = nn.Linear(config.hidden_size, self.all_head_size)
        
        # 初始化注意力概率的稳定 Dropout
        self.dropout = StableDropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        # 调整张量形状以便计算注意力分数，将最后一维划分为注意力头数和其余部分
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, -1)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
    # 计算解缠注意力偏置
    def disentangled_att_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        # 如果未提供相对位置信息，则根据查询层和键层的大小构建相对位置张量
        if relative_pos is None:
            q = query_layer.size(-2)
            relative_pos = build_relative_position(q, key_layer.size(-2), query_layer.device)
        
        # 如果相对位置张量的维度为2，则扩展为4维张量
        if relative_pos.dim() == 2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        # 如果相对位置张量的维度为3，则在第二维度上进行扩展
        elif relative_pos.dim() == 3:
            relative_pos = relative_pos.unsqueeze(1)
        # 如果相对位置张量的维度不为2或3或4，则引发异常
        elif relative_pos.dim() != 4:
            raise ValueError(f"Relative position ids must be of dim 2 or 3 or 4. {relative_pos.dim()}")

        # 限制注意力跨度为查询层和键层大小的较小值，同时不超过最大相对位置范围
        att_span = min(max(query_layer.size(-2), key_layer.size(-2)), self.max_relative_positions)
        # 将相对位置张量转换为长整型，并移动到查询层所在的设备上
        relative_pos = relative_pos.long().to(query_layer.device)
        # 从相对位置嵌入中选择与限制注意力跨度相关的子集，扩展为三维张量
        rel_embeddings = rel_embeddings[
            self.max_relative_positions - att_span : self.max_relative_positions + att_span, :
        ].unsqueeze(0)

        # 初始化注意力分数
        score = 0

        # 如果位置注意力类型包含"c2p"
        if "c2p" in self.pos_att_type:
            # 使用位置投影对相对位置嵌入进行处理，并转换以适应注意力计算的需求
            pos_key_layer = self.pos_proj(rel_embeddings)
            pos_key_layer = self.transpose_for_scores(pos_key_layer)
            # 计算内容到位置的注意力分数
            c2p_att = torch.matmul(query_layer, pos_key_layer.transpose(-1, -2))
            # 对相对位置进行限幅处理，确保不超出有效范围
            c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
            # 使用动态索引扩展相对位置，获取相应的注意力分数
            c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_dynamic_expand(c2p_pos, query_layer, relative_pos))
            # 累加内容到位置的注意力分数
            score += c2p_att

        # 如果位置注意力类型包含"p2c"
        if "p2c" in self.pos_att_type:
            # 使用位置查询投影对相对位置嵌入进行处理，并转换以适应注意力计算的需求
            pos_query_layer = self.pos_q_proj(rel_embeddings)
            pos_query_layer = self.transpose_for_scores(pos_query_layer)
            # 根据比例因子对位置查询层进行归一化
            pos_query_layer /= torch.sqrt(torch.tensor(pos_query_layer.size(-1), dtype=torch.float) * scale_factor)
            # 如果查询层和键层的大小不同，则重新构建相对位置张量
            if query_layer.size(-2) != key_layer.size(-2):
                r_pos = build_relative_position(key_layer.size(-2), key_layer.size(-2), query_layer.device)
            else:
                r_pos = relative_pos
            # 对位置到内容的相对位置进行限幅处理
            p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span * 2 - 1)
            # 计算位置到内容的注意力分数
            p2c_att = torch.matmul(key_layer, pos_query_layer.transpose(-1, -2).to(dtype=key_layer.dtype))
            # 使用动态索引扩展相对位置，获取相应的注意力分数，并进行转置以匹配注意力计算的形状
            p2c_att = torch.gather(
                p2c_att, dim=-1, index=p2c_dynamic_expand(p2c_pos, query_layer, key_layer)
            ).transpose(-1, -2)

            # 如果查询层和键层的大小不同，则进一步处理位置索引
            if query_layer.size(-2) != key_layer.size(-2):
                pos_index = relative_pos[:, :, :, 0].unsqueeze(-1)
                p2c_att = torch.gather(p2c_att, dim=-2, index=pos_dynamic_expand(pos_index, p2c_att, key_layer))
            # 累加位置到内容的注意力分数
            score += p2c_att

        # 返回最终的注意力分数
        return score
    # DebertaEmbeddings 类定义，用于构建来自单词、位置和标记类型嵌入的嵌入层
    """Construct the embeddings from word, position and token_type embeddings."""
    
    # 初始化方法
    def __init__(self, config):
        super().__init__()
        
        # 从配置中获取填充标记 ID，默认为 0
        pad_token_id = getattr(config, "pad_token_id", 0)
        
        # 确定嵌入层的维度大小，默认为 config.hidden_size
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)
        
        # 创建单词嵌入层，vocab_size 是词汇表大小，embedding_size 是嵌入向量的维度，padding_idx 是填充标记的索引
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embedding_size, padding_idx=pad_token_id)
        
        # 是否使用位置偏置输入，默认为 True
        self.position_biased_input = getattr(config, "position_biased_input", True)
        
        # 如果不使用位置偏置输入，则位置嵌入层设为 None；否则创建位置嵌入层，max_position_embeddings 是最大位置嵌入数量
        if not self.position_biased_input:
            self.position_embeddings = None
        else:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.embedding_size)
        
        # 如果配置中有标记类型大小（type_vocab_size 大于 0），则创建标记类型嵌入层
        if config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, self.embedding_size)
        
        # 如果嵌入层大小不等于隐藏层大小，则使用线性变换将其投影到隐藏层大小
        if self.embedding_size != config.hidden_size:
            self.embed_proj = nn.Linear(self.embedding_size, config.hidden_size, bias=False)
        
        # 创建 DebertaLayerNorm 层，用于归一化隐藏层输出
        self.LayerNorm = DebertaLayerNorm(config.hidden_size, config.layer_norm_eps)
        
        # 创建稳定的 Dropout 层，用于隐藏层的随机失活
        self.dropout = StableDropout(config.hidden_dropout_prob)
        
        # 保存配置信息
        self.config = config
        
        # 注册缓冲区，position_ids 是一个持久化的缓冲区，torch.arange 生成 0 到 max_position_embeddings-1 的序列
        # expand((1, -1)) 将其扩展为形状为 (1, max_position_embeddings) 的张量
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
    # 定义前向传播函数，接受多个输入参数并返回嵌入向量表示
    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, mask=None, inputs_embeds=None):
        # 如果传入了 input_ids，则获取其形状
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            # 否则，获取 inputs_embeds 的形状（除去最后一个维度）
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度
        seq_length = input_shape[1]

        # 如果 position_ids 为空，则使用预定义的位置编码（截取到序列长度）
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # 如果 token_type_ids 为空，则创建零填充的张量，与 input_shape 相同的大小
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果 inputs_embeds 为空，则使用 word_embeddings 从 input_ids 中获取嵌入
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # 如果存在 position_embeddings 属性，则从 position_ids 中获取位置嵌入
        if self.position_embeddings is not None:
            position_embeddings = self.position_embeddings(position_ids.long())
        else:
            # 否则，创建与 inputs_embeds 相同形状的零填充张量
            position_embeddings = torch.zeros_like(inputs_embeds)

        # 初始化 embeddings 为 inputs_embeds
        embeddings = inputs_embeds

        # 如果启用了 position_biased_input，则将 position_embeddings 添加到 embeddings 中
        if self.position_biased_input:
            embeddings += position_embeddings

        # 如果配置中定义了 type_vocab_size（表示 token 类型的数量），则将 token_type_embeddings 添加到 embeddings 中
        if self.config.type_vocab_size > 0:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings

        # 如果嵌入大小不等于隐藏层大小，则使用 embed_proj 对 embeddings 进行投影
        if self.embedding_size != self.config.hidden_size:
            embeddings = self.embed_proj(embeddings)

        # 对 embeddings 应用 LayerNorm 规范化
        embeddings = self.LayerNorm(embeddings)

        # 如果存在 mask 参数，则对 embeddings 应用 mask
        if mask is not None:
            # 如果 mask 的维度与 embeddings 不同，则根据 mask 的维度进行调整
            if mask.dim() != embeddings.dim():
                if mask.dim() == 4:
                    mask = mask.squeeze(1).squeeze(1)
                mask = mask.unsqueeze(2)
            # 将 mask 转换为 embeddings 的数据类型
            mask = mask.to(embeddings.dtype)

            # 将 embeddings 应用 mask
            embeddings = embeddings * mask

        # 对 embeddings 应用 dropout
        embeddings = self.dropout(embeddings)

        # 返回处理后的 embeddings
        return embeddings
# DebertaPreTrainedModel 类定义，继承自 PreTrainedModel，用于处理模型权重初始化、预训练模型下载和加载的抽象类
class DebertaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # DebertaPreTrainedModel 类的配置类，指定为 DebertaConfig
    config_class = DebertaConfig

    # 基础模型名称前缀为 "deberta"
    base_model_prefix = "deberta"

    # 在加载模型时忽略的键列表，预期外的键为 "position_embeddings"
    _keys_to_ignore_on_load_unexpected = ["position_embeddings"]

    # 支持梯度检查点的标志
    supports_gradient_checkpointing = True

    # 初始化模型权重的方法
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # 如果是线性层，使用正态分布初始化权重，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置项，初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 如果是嵌入层，使用正态分布初始化权重，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在填充索引，将填充索引对应的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


这段代码定义了一个抽象类 `DebertaPreTrainedModel`，用于处理权重初始化和预训练模型的下载和加载。注释详细解释了类的各个部分和方法的作用。
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列标记在词汇表中的索引。
            # 可以使用 `AutoTokenizer` 获取这些索引。参见 `PreTrainedTokenizer.encode` 和 `PreTrainedTokenizer.__call__`。
            # 详细信息请参阅 [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 遮罩，用于避免对填充标记索引执行注意力操作。
            # 遮罩值为 0 或 1：
            # - 1 表示 **未遮罩** 的标记，
            # - 0 表示 **遮罩** 的标记。
            # 详细信息请参阅 [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 段标记索引，指示输入的第一部分和第二部分。
            # 索引选取在 `[0, 1]` 范围内：
            # - 0 对应 *句子 A* 的标记，
            # - 1 对应 *句子 B* 的标记。
            # 详细信息请参阅 [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 每个输入序列标记在位置嵌入中的位置索引。
            # 索引选取在 `[0, config.max_position_embeddings - 1]` 范围内。
            # 详细信息请参阅 [What are position IDs?](../glossary#position-ids)
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选项，可以直接传递嵌入表示代替 `input_ids`。
            # 如果需要更多控制如何将 *input_ids* 索引转换为相关向量，则此选项非常有用。
            # 这比模型内部的嵌入查找矩阵更为灵活。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。
            # 返回的张量中的 `attentions` 部分有关更多详细信息。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。
            # 返回的张量中的 `hidden_states` 部分有关更多详细信息。
        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通的元组。
"""
DeBERTa 模型的基础类，输出原始的隐藏状态，没有特定的输出头部。

Args:
    config (DebertaConfig): 包含模型配置的对象实例

Attributes:
    embeddings (DebertaEmbeddings): DeBERTa 模型的嵌入层
    encoder (DebertaEncoder): DeBERTa 模型的编码器
    z_steps (int): 用于某些特定功能的步骤计数
    config (DebertaConfig): 模型的配置对象

Raises:
    NotImplementedError: 当尝试调用未实现的修剪功能时抛出异常

Methods:
    get_input_embeddings: 返回模型的输入嵌入层
    set_input_embeddings: 设置模型的输入嵌入层
    _prune_heads: 修剪模型的头部，但此功能在 DeBERTa 模型中尚未实现
    forward: DeBERTa 模型的前向传播方法，接受多个输入参数和配置选项

"""
@add_start_docstrings(
    "The bare DeBERTa Model transformer outputting raw hidden-states without any specific head on top.",
    DEBERTA_START_DOCSTRING,
)
class DebertaModel(DebertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化嵌入层和编码器
        self.embeddings = DebertaEmbeddings(config)
        self.encoder = DebertaEncoder(config)
        self.z_steps = 0
        self.config = config
        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回模型的输入嵌入层（词嵌入）
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        # 设置模型的输入嵌入层（词嵌入）
        self.embeddings.word_embeddings = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 抛出未实现异常，因为 DeBERTa 模型中未实现修剪功能
        raise NotImplementedError("The prune function is not implemented in DeBERTa model.")

    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        # 如果未指定 output_attentions，则使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定 output_hidden_states，则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定 return_dict，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果同时指定了 input_ids 和 inputs_embeds，则抛出 ValueError 异常
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        # 如果指定了 input_ids，则检查 padding 和 attention_mask
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            # 获取 input_ids 的形状
            input_shape = input_ids.size()
        # 如果指定了 inputs_embeds，则获取其形状除去最后一维的部分
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            # 如果既未指定 input_ids 也未指定 inputs_embeds，则抛出 ValueError 异常
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 获取输入数据所在设备的信息
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # 如果未提供 attention_mask，则创建一个全为 1 的 mask，形状与输入数据一致，放置在相同设备上
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        # 如果未提供 token_type_ids，则创建一个全为 0 的 tensor，数据类型为 long，形状与输入数据一致，放置在相同设备上
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # 通过 embeddings 层处理输入数据，得到嵌入的输出
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        # 将嵌入输出传递给 encoder 层进行编码，获取编码器的输出
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )
        # 获取编码器的编码层输出
        encoded_layers = encoder_outputs[1]

        # 如果设置了 z_steps 大于 1，则执行多步的自注意力操作
        if self.z_steps > 1:
            # 获取倒数第二层的隐藏状态
            hidden_states = encoded_layers[-2]
            # 复制编码器的最后一层以形成层数组
            layers = [self.encoder.layer[-1] for _ in range(self.z_steps)]
            # 获取编码器的最后一层的查询状态
            query_states = encoded_layers[-1]
            # 获取相对位置编码的嵌入
            rel_embeddings = self.encoder.get_rel_embedding()
            # 获取注意力 mask
            attention_mask = self.encoder.get_attention_mask(attention_mask)
            # 获取相对位置编码
            rel_pos = self.encoder.get_rel_pos(embedding_output)
            # 对于除第一层外的每一层，执行自注意力操作
            for layer in layers[1:]:
                query_states = layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=False,
                    query_states=query_states,
                    relative_pos=rel_pos,
                    rel_embeddings=rel_embeddings,
                )
                # 将查询状态添加到编码层列表中
                encoded_layers.append(query_states)

        # 获取编码层的最后一层作为序列输出
        sequence_output = encoded_layers[-1]

        # 如果不需要返回字典，则返回编码器输出的元组
        if not return_dict:
            return (sequence_output,) + encoder_outputs[(1 if output_hidden_states else 2):]

        # 如果需要返回字典，则构造一个 BaseModelOutput 对象，并返回
        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions,
        )
# 使用装饰器为类添加文档字符串，描述此类为基于DeBERTa模型的语言建模头部模型
@add_start_docstrings("""DeBERTa Model with a `language modeling` head on top.""", DEBERTA_START_DOCSTRING)
class DebertaForMaskedLM(DebertaPreTrainedModel):
    # 定义权重共享的键名列表，这些键名指定了需要共享权重的模型参数
    _tied_weights_keys = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]

    def __init__(self, config):
        # 调用父类的初始化方法，传入配置参数config
        super().__init__(config)

        # 创建DeBERTa模型实例，并传入配置参数
        self.deberta = DebertaModel(config)
        # 创建DeBERTa的MLM头部实例，并传入配置参数
        self.cls = DebertaOnlyMLMHead(config)

        # 调用本类的后初始化方法，用于初始化权重并进行最终处理
        self.post_init()

    # 返回MLM头部的输出嵌入，这里是预测的词汇表解码器
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置MLM头部的输出嵌入为新的嵌入张量
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 使用装饰器为forward方法添加文档字符串，描述其输入与输出
    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 添加代码示例文档字符串，包括模型检查点、输出类型、配置类等信息
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_MASKED_LM,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="[MASK]",
        expected_output=_MASKED_LM_EXPECTED_OUTPUT,
        expected_loss=_MASKED_LM_EXPECTED_LOSS,
    )
    # 前向传播函数，接收多种输入参数，并返回一个字典或单个张量，根据return_dict的布尔值来决定返回类型
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        # 初始化 return_dict，若未提供则使用 self.config.use_return_dict 的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 DeBERTa 模型进行前向传播
        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取模型输出的序列表示
        sequence_output = outputs[0]

        # 将序列表示作为输入，通过分类层获取预测分数
        prediction_scores = self.cls(sequence_output)

        # 初始化 masked_lm_loss
        masked_lm_loss = None
        # 如果提供了 labels，则计算 masked language modeling 的损失
        if labels is not None:
            # 定义损失函数为交叉熵损失
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            # 计算 masked language modeling 的损失
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果 return_dict 为 False，则按非字典形式返回输出
        if not return_dict:
            # 组装输出为元组
            output = (prediction_scores,) + outputs[1:]
            # 返回损失和输出序列，如果存在损失
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 如果 return_dict 为 True，则以 MaskedLMOutput 对象形式返回结果
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
class DebertaPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)

        # 定义一个全连接层，将隐藏状态转换为指定的嵌入大小
        self.dense = nn.Linear(config.hidden_size, self.embedding_size)
        
        # 根据配置选择激活函数，用于转换层的输出
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        
        # 对转换后的输出进行 Layer Normalization
        self.LayerNorm = nn.LayerNorm(self.embedding_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        # 将隐藏状态输入全连接层进行线性转换
        hidden_states = self.dense(hidden_states)
        
        # 应用预定义的激活函数对转换后的状态进行非线性变换
        hidden_states = self.transform_act_fn(hidden_states)
        
        # 对转换后的状态进行 Layer Normalization
        hidden_states = self.LayerNorm(hidden_states)
        
        return hidden_states


class DebertaLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # 创建一个预测头变换模块，根据配置参数进行初始化
        self.transform = DebertaPredictionHeadTransform(config)

        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)
        
        # 定义一个线性层，用于预测每个 token 的分数，输出维度为词汇表大小，无偏置项
        self.decoder = nn.Linear(self.embedding_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # 将线性层的偏置与参数进行关联，以便与 `resize_token_embeddings` 正确调整偏置大小
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        # 将隐藏状态输入变换模块进行预测头的变换
        hidden_states = self.transform(hidden_states)
        
        # 输入变换后的隐藏状态进行预测头的线性预测
        hidden_states = self.decoder(hidden_states)
        
        return hidden_states


# 从 transformers.models.bert.BertOnlyMLMHead 复制并更名为 DebertaOnlyMLMHead
class DebertaOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # 创建一个预测模块，用于 MLM 头的预测
        self.predictions = DebertaLMPredictionHead(config)

    def forward(self, sequence_output):
        # 将序列输出输入预测模块进行 MLM 头的预测
        prediction_scores = self.predictions(sequence_output)
        
        return prediction_scores


@add_start_docstrings(
    """
    DeBERTa Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    DEBERTA_START_DOCSTRING,
)
class DebertaForSequenceClassification(DebertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
        # 获取类别数
        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        # 创建 DeBERTa 模型，并初始化上下文池化器
        self.deberta = DebertaModel(config)
        self.pooler = ContextPooler(config)
        
        # 获取池化后的输出维度
        output_dim = self.pooler.output_dim

        # 创建一个线性层，用于分类任务的输出，输出维度为类别数
        self.classifier = nn.Linear(output_dim, num_labels)
        
        # 获取或设置分类器的 dropout 概率
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 获取输入嵌入层
        return self.deberta.get_input_embeddings()
    # 设置新的输入嵌入到DeBERTa模型中
    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

    # 为模型的forward方法添加文档字符串，描述输入参数的格式和含义
    # 包括batch_size（批量大小）和sequence_length（序列长度）
    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 添加代码示例的文档字符串，包括模型的checkpoint（检查点）、输出类型、配置类等信息
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义模型的forward方法
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,                  # 输入的token IDs（可选）
        attention_mask: Optional[torch.Tensor] = None,            # 注意力掩码（可选）
        token_type_ids: Optional[torch.Tensor] = None,            # token类型IDs（可选）
        position_ids: Optional[torch.Tensor] = None,              # 位置IDs（可选）
        inputs_embeds: Optional[torch.Tensor] = None,             # 嵌入的输入（可选）
        labels: Optional[torch.Tensor] = None,                    # 标签（可选）
        output_attentions: Optional[bool] = None,                 # 是否输出注意力（可选）
        output_hidden_states: Optional[bool] = None,              # 是否输出隐藏状态（可选）
        return_dict: Optional[bool] = None,                       # 是否返回字典格式的输出（可选）
@add_start_docstrings(
    """
    DeBERTa Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    DEBERTA_START_DOCSTRING,
)
class DebertaForTokenClassification(DebertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # 初始化 DeBERTa 模型
        self.deberta = DebertaModel(config)
        # Dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 线性分类器，将隐藏状态映射到标签空间
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 确定是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 DeBERTa 模型的前向传播
        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取序列输出
        sequence_output = outputs[0]

        # 应用 Dropout
        sequence_output = self.dropout(sequence_output)
        # 应用线性分类器获取 logits
        logits = self.classifier(sequence_output)

        # 计算损失
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 根据是否返回字典构建返回值
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TokenClassifierOutput 对象，包含损失、logits、隐藏状态和注意力权重
        return TokenClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    DEBERTA_START_DOCSTRING,


注释：

# 将隐藏状态输出作为基础层，计算“span起始logits”和“span结束logits”的上层层次。
# DEBERTA_START_DOCSTRING是预定义的常量或字符串，可能用于文档字符串或注释的格式化。
)
class DebertaForQuestionAnswering(DebertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.deberta = DebertaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_QA,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_QA_EXPECTED_OUTPUT,
        expected_loss=_QA_EXPECTED_LOSS,
        qa_target_start_index=_QA_TARGET_START_INDEX,
        qa_target_end_index=_QA_TARGET_END_INDEX,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入的token ID序列，可选的Tensor类型
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，指示模型在计算中应忽略的位置，可选的Tensor类型
        token_type_ids: Optional[torch.Tensor] = None,  # token类型ID，用于区分两个句子或段落，可选的Tensor类型
        position_ids: Optional[torch.Tensor] = None,  # 位置ID，标识输入token的位置信息，可选的Tensor类型
        inputs_embeds: Optional[torch.Tensor] = None,  # 嵌入输入，替代输入IDs的嵌入表示，可选的Tensor类型
        start_positions: Optional[torch.Tensor] = None,  # 答案的起始位置索引，用于训练和评估，可选的Tensor类型
        end_positions: Optional[torch.Tensor] = None,  # 答案的结束位置索引，用于训练和评估，可选的Tensor类型
        output_attentions: Optional[bool] = None,  # 是否返回注意力权重，可选的布尔类型
        output_hidden_states: Optional[bool] = None,  # 是否返回隐藏状态，可选的布尔类型
        return_dict: Optional[bool] = None,  # 是否返回字典形式的输出，可选的布尔类型
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        # 确定是否要返回字典格式的输出，根据配置或参数设定
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 DeBERTa 模型进行前向推断
        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取模型输出中的序列输出
        sequence_output = outputs[0]

        # 将序列输出送入问答输出层获取 logits
        logits = self.qa_outputs(sequence_output)
        # 将 logits 拆分为起始和结束 logits
        start_logits, end_logits = logits.split(1, dim=-1)
        # 去除多余的维度并保持连续性
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        # 如果提供了起始和结束位置的标签，则计算损失
        if start_positions is not None and end_positions is not None:
            # 如果在多 GPU 上运行，扩展维度以匹配模型输出
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 忽略超出模型输入范围的位置
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 定义交叉熵损失函数，并计算起始和结束位置的损失
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        # 如果不需要返回字典格式的输出，则按元组方式返回结果
        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 如果需要返回字典格式的输出，则构建 QuestionAnsweringModelOutput 对象并返回
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```