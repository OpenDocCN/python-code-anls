# `.\models\deberta_v2\modeling_deberta_v2.py`

```
# coding=utf-8
# Copyright 2020 Microsoft and the Hugging Face Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch DeBERTa-v2 model."""

from collections.abc import Sequence
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import softmax_backward_data
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_deberta_v2 import DebertaV2Config


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "DebertaV2Config"
_CHECKPOINT_FOR_DOC = "microsoft/deberta-v2-xlarge"
_QA_TARGET_START_INDEX = 2
_QA_TARGET_END_INDEX = 9

DEBERTA_V2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/deberta-v2-xlarge",
    "microsoft/deberta-v2-xxlarge",
    "microsoft/deberta-v2-xlarge-mnli",
    "microsoft/deberta-v2-xxlarge-mnli",
]


# Copied from transformers.models.deberta.modeling_deberta.ContextPooler
class ContextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用线性层将输入特征从 pooler_hidden_size 转换到 pooler_hidden_size
        self.dense = nn.Linear(config.pooler_hidden_size, config.pooler_hidden_size)
        # 添加稳定的 dropout 层，减少过拟合
        self.dropout = StableDropout(config.pooler_dropout)
        self.config = config

    def forward(self, hidden_states):
        # 通过取第一个 token 的隐藏状态来池化模型
        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token)
        # 将池化后的输出通过线性层得到最终的池化输出
        pooled_output = self.dense(context_token)
        # 使用激活函数 ACT2FN[self.config.pooler_hidden_act] 处理池化输出
        pooled_output = ACT2FN[self.config.pooler_hidden_act](pooled_output)
        return pooled_output

    @property
    def output_dim(self):
        # 返回输出维度为隐藏大小
        return self.config.hidden_size


# Copied from transformers.models.deberta.modeling_deberta.XSoftmax with deberta->deberta_v2
class XSoftmax(torch.autograd.Function):
    """
    Masked Softmax which is optimized for saving memory
    """
    # XSoftmax 是一个优化内存的掩码 Softmax 函数
    Args:
        input (`torch.tensor`): The input tensor that will apply softmax.
        mask (`torch.IntTensor`):
            The mask matrix where 0 indicate that element will be ignored in the softmax calculation.
        dim (int): The dimension along which softmax will be applied.

    Example:

    ```python
    >>> import torch
    >>> from transformers.models.deberta_v2.modeling_deberta_v2 import XSoftmax

    >>> # Make a tensor
    >>> x = torch.randn([4, 20, 100])

    >>> # Create a mask
    >>> mask = (x > 0).int()

    >>> # Specify the dimension to apply softmax
    >>> dim = -1

    >>> y = XSoftmax.apply(x, mask, dim)
    ```

    @staticmethod
    def forward(self, input, mask, dim):
        # Set the dimension for softmax calculation
        self.dim = dim
        # Invert the mask to create a reverse mask (rmask)
        rmask = ~(mask.to(torch.bool))

        # Replace ignored elements with the minimum value of the input tensor's dtype
        output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
        # Apply softmax along the specified dimension
        output = torch.softmax(output, self.dim)
        # Zero out the softmax values corresponding to ignored elements
        output.masked_fill_(rmask, 0)
        # Save the output tensor for backward computation
        self.save_for_backward(output)
        return output

    @staticmethod
    def backward(self, grad_output):
        # Retrieve the saved output tensor
        (output,) = self.saved_tensors
        # Compute gradient of input with respect to softmax output
        inputGrad = softmax_backward_data(self, grad_output, output, self.dim, output)
        return inputGrad, None, None

    @staticmethod
    def symbolic(g, self, mask, dim):
        import torch.onnx.symbolic_helper as sym_help
        from torch.onnx.symbolic_opset9 import masked_fill, softmax

        # Cast mask to long and create reverse mask (r_mask)
        mask_cast_value = g.op("Cast", mask, to_i=sym_help.cast_pytorch_to_onnx["Long"])
        r_mask = g.op(
            "Cast",
            g.op("Sub", g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64)), mask_cast_value),
            to_i=sym_help.cast_pytorch_to_onnx["Bool"],
        )
        # Fill ignored elements with the minimum value of tensor's dtype
        output = masked_fill(
            g, self, r_mask, g.op("Constant", value_t=torch.tensor(torch.finfo(self.type().dtype()).min))
        )
        # Apply softmax along specified dimension
        output = softmax(g, output, dim)
        # Fill ignored elements of softmax output with zero
        return masked_fill(g, output, r_mask, g.op("Constant", value_t=torch.tensor(0, dtype=torch.bool)))
# Copied from transformers.models.deberta.modeling_deberta.DropoutContext
# 定义了一个名为 DropoutContext 的类，用于管理 Dropout 相关的上下文信息
class DropoutContext(object):
    def __init__(self):
        # 初始化 dropout 参数为 0
        self.dropout = 0
        # 初始化 mask 为 None
        self.mask = None
        # 初始化 scale 参数为 1
        self.scale = 1
        # 初始化 reuse_mask 参数为 True，表示可以重复使用 mask
        self.reuse_mask = True


# Copied from transformers.models.deberta.modeling_deberta.get_mask
# 定义了一个名为 get_mask 的函数，用于根据不同的上下文获取 dropout mask
def get_mask(input, local_context):
    # 如果 local_context 不是 DropoutContext 类型，则将其作为 dropout 参数处理
    if not isinstance(local_context, DropoutContext):
        dropout = local_context
        mask = None
    else:
        # 如果 local_context 是 DropoutContext 类型，则获取其中的 dropout 和 scale 参数
        dropout = local_context.dropout
        dropout *= local_context.scale
        # 根据 reuse_mask 参数决定是否重用 mask
        mask = local_context.mask if local_context.reuse_mask else None

    # 如果 dropout 大于 0 且 mask 为 None，则生成一个新的 dropout mask
    if dropout > 0 and mask is None:
        mask = (1 - torch.empty_like(input).bernoulli_(1 - dropout)).to(torch.bool)

    # 如果 local_context 是 DropoutContext 类型且其 mask 为 None，则更新其 mask
    if isinstance(local_context, DropoutContext):
        if local_context.mask is None:
            local_context.mask = mask

    # 返回生成的 mask 和 dropout 参数
    return mask, dropout


# Copied from transformers.models.deberta.modeling_deberta.XDropout
# 定义了一个名为 XDropout 的自定义 PyTorch 函数，优化了 dropout 操作以节省计算和内存
class XDropout(torch.autograd.Function):
    """Optimized dropout function to save computation and memory by using mask operation instead of multiplication."""

    @staticmethod
    def forward(ctx, input, local_ctx):
        # 调用 get_mask 函数获取 mask 和 dropout 参数
        mask, dropout = get_mask(input, local_ctx)
        # 计算 scale 参数用于反向传播时的缩放
        ctx.scale = 1.0 / (1 - dropout)
        # 如果 dropout 大于 0，则应用 dropout mask，并对输入进行缩放
        if dropout > 0:
            ctx.save_for_backward(mask)
            return input.masked_fill(mask, 0) * ctx.scale
        else:
            # 如果 dropout 等于 0，则直接返回输入
            return input

    @staticmethod
    def backward(ctx, grad_output):
        # 如果 scale 大于 1，则恢复被 dropout 的梯度
        if ctx.scale > 1:
            (mask,) = ctx.saved_tensors
            return grad_output.masked_fill(mask, 0) * ctx.scale, None
        else:
            # 如果 scale 不大于 1，则直接返回梯度
            return grad_output, None

    @staticmethod
    def symbolic(g: torch._C.Graph, input: torch._C.Value, local_ctx: Union[float, DropoutContext]) -> torch._C.Value:
        from torch.onnx import symbolic_opset12

        # 根据 local_ctx 类型决定 dropout 参数
        dropout_p = local_ctx
        if isinstance(local_ctx, DropoutContext):
            dropout_p = local_ctx.dropout
        # 使用符号运算创建 ONNX 图中的 dropout 操作
        # 这里固定使用 opset12 版本的 dropout 符号操作
        train = True  # StableDropout 只在训练时调用此函数。
        return symbolic_opset12.dropout(g, input, dropout_p, train)


# Copied from transformers.models.deberta.modeling_deberta.StableDropout
# 定义了一个名为 StableDropout 的 PyTorch 模块，用于稳定化训练时的 dropout 操作
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
        # 如果处于训练模式且 dropout 概率大于0，则应用自定义的 XDropout 操作
        if self.training and self.drop_prob > 0:
            return XDropout.apply(x, self.get_context())
        # 否则直接返回输入张量 x
        return x

    def clear_context(self):
        """
        Clear the context stack and reset count to zero.
        """
        # 将计数器 count 设为 0，清空上下文栈 context_stack
        self.count = 0
        self.context_stack = None

    def init_context(self, reuse_mask=True, scale=1):
        """
        Initialize the context stack with optional parameters.

        Args:
            reuse_mask (bool, optional): Whether to reuse mask for dropout. Defaults to True.
            scale (int, optional): Scaling factor for dropout. Defaults to 1.
        """
        # 如果上下文栈 context_stack 为空，则初始化为空列表
        if self.context_stack is None:
            self.context_stack = []
        # 将计数器 count 设为 0
        self.count = 0
        # 遍历上下文栈 context_stack，设置每个上下文对象的复用掩码和缩放因子
        for c in self.context_stack:
            c.reuse_mask = reuse_mask
            c.scale = scale

    def get_context(self):
        """
        Get the current dropout context from the context stack or create a new one.

        Returns:
            DropoutContext: Current or newly created dropout context.
        """
        # 如果上下文栈 context_stack 不为空
        if self.context_stack is not None:
            # 如果计数器 count 大于或等于上下文栈 context_stack 的长度，添加新的 DropoutContext 对象到栈中
            if self.count >= len(self.context_stack):
                self.context_stack.append(DropoutContext())
            # 获取当前计数器对应的上下文对象 ctx
            ctx = self.context_stack[self.count]
            # 设置该上下文对象的 dropout 属性为当前实例的 drop_prob
            ctx.dropout = self.drop_prob
            # 计数器 count 加一
            self.count += 1
            # 返回获取到的上下文对象 ctx
            return ctx
        else:
            # 如果上下文栈 context_stack 为空，则直接返回当前实例的 drop_prob
            return self.drop_prob
# Copied from transformers.models.deberta.modeling_deberta.DebertaSelfOutput with DebertaLayerNorm->LayerNorm
class DebertaV2SelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个全连接层，输入和输出维度都是 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 初始化 LayerNorm 层，输入维度是 config.hidden_size，使用 config.layer_norm_eps 作为 epsilon 参数
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        # 初始化一个稳定的 Dropout 层，使用 config.hidden_dropout_prob 作为 dropout 概率
        self.dropout = StableDropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # 将 hidden_states 输入全连接层 self.dense 中
        hidden_states = self.dense(hidden_states)
        # 对全连接层的输出进行 dropout 处理
        hidden_states = self.dropout(hidden_states)
        # 将 dropout 处理后的 hidden_states 和 input_tensor 相加，并通过 LayerNorm 层处理
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.deberta.modeling_deberta.DebertaAttention with Deberta->DebertaV2
class DebertaV2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个 DisentangledSelfAttention 对象
        self.self = DisentangledSelfAttention(config)
        # 初始化一个 DebertaV2SelfOutput 对象
        self.output = DebertaV2SelfOutput(config)
        # 保存配置信息
        self.config = config

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
    ):
        # 调用 self.self 的 forward 方法进行自注意力计算
        self_output = self.self(
            hidden_states,
            attention_mask,
            output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        # 如果需要输出注意力矩阵，解包 self_output
        if output_attentions:
            self_output, att_matrix = self_output
        # 如果 query_states 为 None，则使用 hidden_states 作为 query_states
        if query_states is None:
            query_states = hidden_states
        # 调用 self.output 的 forward 方法，将 self_output 和 query_states 作为输入
        attention_output = self.output(self_output, query_states)

        if output_attentions:
            return (attention_output, att_matrix)
        else:
            return attention_output


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->DebertaV2
class DebertaV2Intermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个全连接层，输入维度是 config.hidden_size，输出维度是 config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 根据 config.hidden_act 的类型选择对应的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将 hidden_states 输入全连接层 self.dense
        hidden_states = self.dense(hidden_states)
        # 使用选择的激活函数对全连接层的输出进行激活
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.deberta.modeling_deberta.DebertaOutput with DebertaLayerNorm->LayerNorm
class DebertaV2Output(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个全连接层，输入维度是 config.intermediate_size，输出维度是 config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 初始化 LayerNorm 层，输入维度是 config.hidden_size，使用 config.layer_norm_eps 作为 epsilon 参数
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        # 初始化一个稳定的 Dropout 层，使用 config.hidden_dropout_prob 作为 dropout 概率
        self.dropout = StableDropout(config.hidden_dropout_prob)
        # 保存配置信息
        self.config = config
    # 定义神经网络的前向传播函数，接收隐藏状态和输入张量作为参数
    def forward(self, hidden_states, input_tensor):
        # 将隐藏状态通过全连接层进行变换
        hidden_states = self.dense(hidden_states)
        # 对变换后的隐藏状态应用丢弃(dropout)操作
        hidden_states = self.dropout(hidden_states)
        # 将丢弃后的隐藏状态与输入张量相加，并通过层归一化处理
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的隐藏状态
        return hidden_states
# 从transformers.models.deberta.modeling_deberta.DebertaLayer复制而来，Deberta->DebertaV2
class DebertaV2Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化注意力层，使用DebertaV2Attention类
        self.attention = DebertaV2Attention(config)
        # 初始化中间层，使用DebertaV2Intermediate类
        self.intermediate = DebertaV2Intermediate(config)
        # 初始化输出层，使用DebertaV2Output类
        self.output = DebertaV2Output(config)

    def forward(
        self,
        hidden_states,
        attention_mask,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
        output_attentions=False,
    ):
        # 调用注意力层的前向传播函数
        attention_output = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        # 如果需要输出注意力矩阵，则解包注意力输出
        if output_attentions:
            attention_output, att_matrix = attention_output
        # 经过中间层的前向传播
        intermediate_output = self.intermediate(attention_output)
        # 经过输出层的前向传播，得到最终层的输出
        layer_output = self.output(intermediate_output, attention_output)
        # 如果需要输出注意力矩阵，则返回层输出和注意力矩阵
        if output_attentions:
            return (layer_output, att_matrix)
        else:
            return layer_output


class ConvLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 获取卷积核大小，默认为3
        kernel_size = getattr(config, "conv_kernel_size", 3)
        # 获取卷积组数，默认为1
        groups = getattr(config, "conv_groups", 1)
        # 获取卷积激活函数，默认为"tanh"
        self.conv_act = getattr(config, "conv_act", "tanh")
        # 定义一维卷积层
        self.conv = nn.Conv1d(
            config.hidden_size, config.hidden_size, kernel_size, padding=(kernel_size - 1) // 2, groups=groups
        )
        # 初始化LayerNorm层
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        # 初始化稳定的Dropout层
        self.dropout = StableDropout(config.hidden_dropout_prob)
        # 保存配置信息
        self.config = config

    def forward(self, hidden_states, residual_states, input_mask):
        # 执行卷积操作，要求hidden_states的维度为[batch_size, seq_length, hidden_size]
        out = self.conv(hidden_states.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        # 生成掩码，用于遮盖无效位置的输出
        rmask = (1 - input_mask).bool()
        out.masked_fill_(rmask.unsqueeze(-1).expand(out.size()), 0)
        # 应用激活函数到卷积输出，并加上稳定的Dropout
        out = ACT2FN[self.conv_act](self.dropout(out))

        # 计算LayerNorm的输入，即残差连接后的结果
        layer_norm_input = residual_states + out
        # 对LayerNorm层进行归一化处理
        output = self.LayerNorm(layer_norm_input).to(layer_norm_input)

        # 如果输入掩码为空，则直接使用输出；否则，根据掩码遮盖输出结果
        if input_mask is None:
            output_states = output
        else:
            if input_mask.dim() != layer_norm_input.dim():
                # 如果输入掩码维度与LayerNorm输入维度不同，则调整掩码维度
                if input_mask.dim() == 4:
                    input_mask = input_mask.squeeze(1).squeeze(1)
                input_mask = input_mask.unsqueeze(2)

            input_mask = input_mask.to(output.dtype)
            # 对输出应用掩码
            output_states = output * input_mask

        return output_states


class DebertaV2Encoder(nn.Module):
    """Modified BertEncoder with relative position bias support"""
    # 初始化函数，接收一个配置对象作为参数
    def __init__(self, config):
        # 调用父类初始化方法
        super().__init__()

        # 创建包含多个 DebertaV2Layer 层的 ModuleList，数量由配置中的 num_hidden_layers 决定
        self.layer = nn.ModuleList([DebertaV2Layer(config) for _ in range(config.num_hidden_layers)])
        
        # 检查是否启用相对注意力机制
        self.relative_attention = getattr(config, "relative_attention", False)

        # 如果启用相对注意力机制
        if self.relative_attention:
            # 获取最大相对位置数，如果未设置或小于1，则使用默认的最大位置嵌入数
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings

            # 获取位置桶的数目，如果大于0，则根据桶数计算位置嵌入的大小
            self.position_buckets = getattr(config, "position_buckets", -1)
            pos_ebd_size = self.max_relative_positions * 2

            # 如果设置了位置桶数，则重新计算位置嵌入的大小
            if self.position_buckets > 0:
                pos_ebd_size = self.position_buckets * 2

            # 创建相对位置嵌入层，使用 nn.Embedding，大小为 pos_ebd_size × config.hidden_size
            self.rel_embeddings = nn.Embedding(pos_ebd_size, config.hidden_size)

        # 解析配置中的 norm_rel_ebd 字符串，去除首尾空格并转换为小写，以列表形式保存到 self.norm_rel_ebd
        self.norm_rel_ebd = [x.strip() for x in getattr(config, "norm_rel_ebd", "none").lower().split("|")]

        # 如果 norm_rel_ebd 中包含 "layer_norm"，则创建 LayerNorm 层用于归一化相对位置嵌入
        if "layer_norm" in self.norm_rel_ebd:
            self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=True)

        # 如果配置中指定了卷积核大小大于0，则创建 ConvLayer，否则将 self.conv 设为 None
        self.conv = ConvLayer(config) if getattr(config, "conv_kernel_size", 0) > 0 else None
        
        # 梯度检查点默认为 False
        self.gradient_checkpointing = False

    # 获取相对位置嵌入，如果未启用相对注意力或相对位置嵌入不存在，则返回 None
    def get_rel_embedding(self):
        rel_embeddings = self.rel_embeddings.weight if self.relative_attention else None
        # 如果相对位置嵌入存在且需要进行 LayerNorm，则对相对位置嵌入进行归一化处理
        if rel_embeddings is not None and ("layer_norm" in self.norm_rel_ebd):
            rel_embeddings = self.LayerNorm(rel_embeddings)
        return rel_embeddings

    # 获取注意力掩码，用于屏蔽无效的注意力位置
    def get_attention_mask(self, attention_mask):
        # 如果 attention_mask 的维度不大于2，则扩展其维度以适应多头注意力计算的需求
        if attention_mask.dim() <= 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
        # 如果 attention_mask 的维度为3，则在第1维上再次扩展以适应多头注意力计算的需求
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)

        return attention_mask

    # 获取相对位置编码，根据输入的隐藏状态和查询状态生成相对位置编码
    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        # 如果启用相对注意力且未提供相对位置编码，则根据参数构建相对位置编码
        if self.relative_attention and relative_pos is None:
            q = query_states.size(-2) if query_states is not None else hidden_states.size(-2)
            relative_pos = build_relative_position(
                q,
                hidden_states.size(-2),
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
                device=hidden_states.device,
            )
        return relative_pos

    # 前向传播函数，接收多个参数用于模型的计算，并返回模型输出
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
        # 如果输入的 attention_mask 的维度小于等于2，则直接使用该输入作为 input_mask
        if attention_mask.dim() <= 2:
            input_mask = attention_mask
        else:
            # 否则，计算 attention_mask 沿着倒数第二维的和是否大于0，生成 input_mask
            input_mask = attention_mask.sum(-2) > 0
        # 获取处理后的 attention_mask
        attention_mask = self.get_attention_mask(attention_mask)
        # 获取相对位置编码
        relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)

        # 初始化用于存储所有隐藏状态和注意力的元组，如果不需要输出则为 None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # 如果 hidden_states 是 Sequence 类型，则取其第一个作为 next_kv
        if isinstance(hidden_states, Sequence):
            next_kv = hidden_states[0]
        else:
            next_kv = hidden_states
        # 获取相对位置编码的嵌入
        rel_embeddings = self.get_rel_embedding()
        # 输出状态初始化为 next_kv
        output_states = next_kv
        # 遍历每一个 transformer 层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前 output_states 添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (output_states,)

            # 如果启用了梯度检查点且正在训练阶段，则使用梯度检查点函数进行前向传播
            if self.gradient_checkpointing and self.training:
                output_states = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    next_kv,
                    attention_mask,
                    query_states,
                    relative_pos,
                    rel_embeddings,
                    output_attentions,
                )
            else:
                # 否则，直接调用当前层进行前向传播
                output_states = layer_module(
                    next_kv,
                    attention_mask,
                    query_states=query_states,
                    relative_pos=relative_pos,
                    rel_embeddings=rel_embeddings,
                    output_attentions=output_attentions,
                )

            # 如果需要输出注意力权重，则从 output_states 中解包出注意力权重 att_m
            if output_attentions:
                output_states, att_m = output_states

            # 如果是第一个层并且存在卷积操作，则将卷积操作应用到输出状态上
            if i == 0 and self.conv is not None:
                output_states = self.conv(hidden_states, output_states, input_mask)

            # 如果 query_states 不为 None，则更新 query_states 为当前输出状态
            if query_states is not None:
                query_states = output_states
                # 如果 hidden_states 是 Sequence 类型，则更新 next_kv 为下一个 hidden_states
                if isinstance(hidden_states, Sequence):
                    next_kv = hidden_states[i + 1] if i + 1 < len(self.layer) else None
            else:
                # 否则，更新 next_kv 为当前输出状态
                next_kv = output_states

            # 如果需要输出注意力权重，则将当前层的注意力权重 att_m 添加到 all_attentions 中
            if output_attentions:
                all_attentions = all_attentions + (att_m,)

        # 如果需要输出隐藏状态，则将最后一个 output_states 添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (output_states,)

        # 如果不需要返回字典格式的输出，则将结果打包为元组返回
        if not return_dict:
            return tuple(v for v in [output_states, all_hidden_states, all_attentions] if v is not None)
        # 否则，返回 BaseModelOutput 类型的对象，包含最终的隐藏状态、所有隐藏状态和所有注意力权重
        return BaseModelOutput(
            last_hidden_state=output_states, hidden_states=all_hidden_states, attentions=all_attentions
        )
@torch.jit.script
# 从transformers.models.deberta.modeling_deberta.c2p_dynamic_expand复制而来，用于扩展C2P位置编码
def c2p_dynamic_expand(c2p_pos, query_layer, relative_pos):
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), query_layer.size(2), relative_pos.size(-1)])


@torch.jit.script
# 从transformers.models.deberta.modeling_deberta.p2c_dynamic_expand复制而来，用于扩展P2C位置编码
def p2c_dynamic_expand(c2p_pos, query_layer, key_layer):
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), key_layer.size(-2), key_layer.size(-2)])


@torch.jit.script
# 从transformers.models.deberta.modeling_deberta.pos_dynamic_expand复制而来，用于扩展位置索引
def pos_dynamic_expand(pos_index, p2c_att, key_layer):
    return pos_index.expand(p2c_att.size()[:2] + (pos_index.size(-2), key_layer.size(-2)))


class DisentangledSelfAttention(nn.Module):
    """
    Disentangled self-attention module

    Parameters:
        config (`DebertaV2Config`):
            A model config class instance with the configuration to build a new model. The schema is similar to
            *BertConfig*, for more details, please refer [`DebertaV2Config`]

    """
    # 初始化函数，接收一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        
        # 检查隐藏层大小是否能被注意力头数整除，否则抛出数值错误异常
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        
        # 设置注意力头数
        self.num_attention_heads = config.num_attention_heads
        
        # 计算每个注意力头的大小
        _attention_head_size = config.hidden_size // config.num_attention_heads
        self.attention_head_size = getattr(config, "attention_head_size", _attention_head_size)
        
        # 计算所有注意力头的总大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # 初始化查询、键、值的线性投影层
        self.query_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.key_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.value_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)

        # 是否共享注意力键
        self.share_att_key = getattr(config, "share_att_key", False)
        
        # 位置注意力类型
        self.pos_att_type = config.pos_att_type if config.pos_att_type is not None else []
        
        # 是否使用相对位置注意力
        self.relative_attention = getattr(config, "relative_attention", False)

        # 如果使用相对位置注意力
        if self.relative_attention:
            # 设置位置桶数和最大相对位置
            self.position_buckets = getattr(config, "position_buckets", -1)
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            
            # 如果最大相对位置小于1，则使用配置的最大位置嵌入数
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            
            # 设置位置嵌入的大小
            self.pos_ebd_size = self.max_relative_positions
            
            # 如果位置桶数大于0，则将位置嵌入的大小设为位置桶数
            if self.position_buckets > 0:
                self.pos_ebd_size = self.position_buckets
            
            # 初始化位置嵌入的稳定dropout
            self.pos_dropout = StableDropout(config.hidden_dropout_prob)
            
            # 如果不共享注意力键
            if not self.share_att_key:
                # 如果是"c2p"类型的位置注意力，初始化位置键的线性投影层
                if "c2p" in self.pos_att_type:
                    self.pos_key_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
                # 如果是"p2c"类型的位置注意力，初始化位置查询的线性投影层
                if "p2c" in self.pos_att_type:
                    self.pos_query_proj = nn.Linear(config.hidden_size, self.all_head_size)

        # 初始化注意力概率的稳定dropout
        self.dropout = StableDropout(config.attention_probs_dropout_prob)

    # 将输入张量 x 转置以适应多头注意力的形状
    def transpose_for_scores(self, x, attention_heads):
        new_x_shape = x.size()[:-1] + (attention_heads, -1)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))

    # 前向传播函数
    def forward(
        self,
        hidden_states,           # 输入的隐藏状态张量
        attention_mask,          # 注意力掩码张量
        output_attentions=False, # 是否输出注意力
        query_states=None,       # 查询状态张量（可选）
        relative_pos=None,       # 相对位置（可选）
        rel_embeddings=None,     # 相对位置嵌入（可选）
# 从 transformers.models.deberta.modeling_deberta.DebertaEmbeddings 复制而来，修改了 DebertaLayerNorm->LayerNorm
class DebertaV2Embeddings(nn.Module):
    """从单词、位置和令牌类型嵌入构造嵌入。"""

    def __init__(self, config):
        super().__init__()
        # 获取填充令牌ID，若无则默认为0
        pad_token_id = getattr(config, "pad_token_id", 0)
        # 获取嵌入大小，默认为隐藏大小
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)
        # 创建单词嵌入层，大小为词汇表大小 x 嵌入大小，使用填充ID作为padding_idx
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embedding_size, padding_idx=pad_token_id)

        # 是否使用位置偏置输入，默认为True
        self.position_biased_input = getattr(config, "position_biased_input", True)
        if not self.position_biased_input:
            self.position_embeddings = None
        else:
            # 创建位置嵌入层，大小为最大位置嵌入数 x 嵌入大小
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.embedding_size)

        # 如果类型词汇大小大于0，则创建令牌类型嵌入层，大小为类型词汇大小 x 嵌入大小
        if config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, self.embedding_size)

        # 如果嵌入大小不等于隐藏大小，则创建线性投影层，将嵌入大小映射到隐藏大小，无偏置
        if self.embedding_size != config.hidden_size:
            self.embed_proj = nn.Linear(self.embedding_size, config.hidden_size, bias=False)
        
        # 创建LayerNorm层，对隐藏大小进行归一化，使用给定的层归一化epsilon值
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        # 创建稳定Dropout层，使用给定的隐藏丢弃概率
        self.dropout = StableDropout(config.hidden_dropout_prob)
        # 保存配置信息
        self.config = config

        # position_ids (1, len position emb) 在内存中是连续的，并且在序列化时被导出
        # 创建位置ID张量，大小为1 x 最大位置嵌入数，使用torch.arange扩展而来，不持久化
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
    # 定义前向传播方法，接收多个输入参数：input_ids, token_type_ids, position_ids, mask, inputs_embeds
    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, mask=None, inputs_embeds=None):
        # 如果 input_ids 不为 None，则获取其形状
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            # 否则获取 inputs_embeds 的形状，去除最后一个维度
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度，即 input_shape 的第二个维度
        seq_length = input_shape[1]

        # 如果 position_ids 为 None，则使用预定义的 self.position_ids，并截取到与序列长度相同的部分
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # 如果 token_type_ids 为 None，则创建与 input_shape 相同形状的零张量
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果 inputs_embeds 为 None，则使用 self.word_embeddings 对 input_ids 进行嵌入处理
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # 如果存在 self.position_embeddings，则根据 position_ids 获取位置嵌入
        if self.position_embeddings is not None:
            position_embeddings = self.position_embeddings(position_ids.long())
        else:
            # 否则创建与 inputs_embeds 相同形状的零张量作为位置嵌入
            position_embeddings = torch.zeros_like(inputs_embeds)

        # 将嵌入向量初始化为 inputs_embeds
        embeddings = inputs_embeds
        # 如果开启了位置偏置输入 self.position_biased_input，则加上位置嵌入
        if self.position_biased_input:
            embeddings += position_embeddings
        # 如果配置中的 type_vocab_size 大于 0，则加上 token_type_embeddings
        if self.config.type_vocab_size > 0:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings

        # 如果嵌入大小不等于隐藏大小 self.embedding_size != self.config.hidden_size，则通过 embed_proj 进行投影
        if self.embedding_size != self.config.hidden_size:
            embeddings = self.embed_proj(embeddings)

        # 经过 LayerNorm 归一化处理
        embeddings = self.LayerNorm(embeddings)

        # 如果 mask 不为 None，则对 embeddings 应用 mask
        if mask is not None:
            # 如果 mask 的维度不等于 embeddings 的维度
            if mask.dim() != embeddings.dim():
                # 如果 mask 的维度为 4，则进行挤压操作
                if mask.dim() == 4:
                    mask = mask.squeeze(1).squeeze(1)
                # 将 mask 的维度扩展到与 embeddings 相同
                mask = mask.unsqueeze(2)
            # 将 mask 转换为与 embeddings 相同的数据类型
            mask = mask.to(embeddings.dtype)
            # 应用 mask 到 embeddings 上
            embeddings = embeddings * mask

        # 经过 dropout 处理
        embeddings = self.dropout(embeddings)
        # 返回 embeddings
        return embeddings
# 从transformers.models.deberta.modeling_deberta.DebertaPreTrainedModel复制而来，将Deberta改为DebertaV2
class DebertaV2PreTrainedModel(PreTrainedModel):
    """
    用于处理权重初始化、预训练模型下载和加载的抽象类。
    """
    
    # 配置类指定为DebertaV2Config
    config_class = DebertaV2Config
    # 基础模型前缀为"deberta"
    base_model_prefix = "deberta"
    # 加载时忽略的键名列表
    _keys_to_ignore_on_load_unexpected = ["position_embeddings"]
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """初始化权重。"""
        if isinstance(module, nn.Linear):
            # 与TF版本稍有不同，使用正态分布进行初始化
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
                

DEBERTA_START_DOCSTRING = r"""
    DeBERTa模型由何鹏程、刘晓东、高建峰、陈伟柱在论文《DeBERTa: Decoding-enhanced BERT with Disentangled
    Attention》中提出。它在BERT/RoBERTa的基础上进行了两项改进，即解耦注意力和增强的掩码解码器。通过这两项改进，
    在使用80GB预训练数据的大多数任务上超越了BERT/RoBERTa。

    这个模型也是PyTorch的torch.nn.Module子类。
    使用时可以像普通的PyTorch Module一样使用，并参考PyTorch文档处理一切一般使用和行为相关的事项。
    

    参数:
        config ([`DebertaV2Config`]): 包含模型所有参数的配置类。
            使用配置文件初始化模型时不会加载模型的权重，只会加载配置信息。
            查看 [`~PreTrainedModel.from_pretrained`] 方法来加载模型权重。
"""

DEBERTA_INPUTS_DOCSTRING = r"""
    # 输入
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列的token索引，在词汇表中
            Indices of input sequence tokens in the vocabulary.
            
            # 可以使用`AutoTokenizer`获取这些索引。详见`PreTrainedTokenizer.encode`和`PreTrainedTokenizer.__call__`
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 注意力掩码，避免在填充的token索引上进行注意力计算
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 分段token索引，指示输入的第一部分和第二部分。索引在`[0, 1]`中选择：

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 每个输入序列token在位置嵌入中的位置索引。在范围`[0, config.max_position_embeddings - 1]`中选择。

            [What are position IDs?](../glossary#position-ids)
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选项，直接传递嵌入表示，而不是传递`input_ids`。在想要更多控制如何将`input_ids`索引转换为关联向量时有用。

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回张量下的`attentions`。
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多详细信息，请参见返回张量下的`hidden_states`。
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.

        return_dict (`bool`, *optional*):
            # 是否返回`~utils.ModelOutput`而不是普通元组。
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""
@add_start_docstrings(
    "The bare DeBERTa Model transformer outputting raw hidden-states without any specific head on top.",
    DEBERTA_START_DOCSTRING,
)
# 从transformers.models.deberta.modeling_deberta.DebertaModel复制而来，将Deberta更改为DebertaV2
class DebertaV2Model(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化模型的嵌入层和编码器
        self.embeddings = DebertaV2Embeddings(config)
        self.encoder = DebertaV2Encoder(config)
        self.z_steps = 0  # 初始化 z_steps 为 0
        self.config = config  # 保存模型配置
        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回模型的嵌入层中的词嵌入
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        # 设置模型的嵌入层中的词嵌入
        self.embeddings.word_embeddings = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        剪枝模型中的注意力头。
        heads_to_prune: 要剪枝的头部字典 {层号: 要在此层中剪枝的头部列表}，参见基类PreTrainedModel
        """
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

        # DeBERTa模型的前向传播函数，接受多种输入参数和控制标志
        # DEBERTA_INPUTS_DOCSTRING 格式化字符串，描述了输入的文档字符串
        # _CHECKPOINT_FOR_DOC 检查点用于文档，BaseModelOutput 输出类型，_CONFIG_FOR_DOC 配置类
        pass
        ) -> Union[Tuple, BaseModelOutput]:
        # 如果用户没有指定是否输出注意力权重，使用配置中的默认设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果用户没有指定是否输出隐藏状态，使用配置中的默认设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果用户没有指定是否返回字典格式的输出，使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果既指定了输入的 token IDs 又指定了嵌入向量，抛出数值错误
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            # 如果指定了输入的 token IDs，则检查是否存在填充并且没有给出注意力遮罩的警告
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            # 获取输入的 token IDs 的形状
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            # 如果指定了嵌入向量，则获取其形状，排除最后一个维度（用于批处理）
            input_shape = inputs_embeds.size()[:-1]
        else:
            # 如果既未指定 token IDs 也未指定嵌入向量，抛出数值错误
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 确定使用的设备是 token IDs 的设备还是嵌入向量的设备
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # 如果没有给出注意力遮罩，则创建一个全为 1 的注意力遮罩张量
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        # 如果没有给出 token 类型 IDs，则创建一个全为 0 的 token 类型 IDs 张量
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # 将输入传递到嵌入层，获取嵌入输出
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        # 将嵌入输出传递到编码器层，并返回编码器的输出
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )
        # 获取编码器的编码层输出
        encoded_layers = encoder_outputs[1]

        # 如果设置了多步更新 z_steps 大于 1
        if self.z_steps > 1:
            # 获取倒数第二层的隐藏状态
            hidden_states = encoded_layers[-2]
            # 复制编码器最后一层，次数为 z_steps
            layers = [self.encoder.layer[-1] for _ in range(self.z_steps)]
            # 获取查询状态
            query_states = encoded_layers[-1]
            # 获取相对嵌入
            rel_embeddings = self.encoder.get_rel_embedding()
            # 获取注意力遮罩
            attention_mask = self.encoder.get_attention_mask(attention_mask)
            # 获取相对位置
            rel_pos = self.encoder.get_rel_pos(embedding_output)
            # 对于除了第一层的每一层
            for layer in layers[1:]:
                # 运行
                Those .g There Med J Give Read Simple Here Engage in Perhaps they had been
@add_start_docstrings("""DeBERTa Model with a `language modeling` head on top.""", DEBERTA_START_DOCSTRING)
class DebertaV2ForMaskedLM(DebertaV2PreTrainedModel):
    _tied_weights_keys = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        # 初始化 DeBERTa V2 模型
        self.deberta = DebertaV2Model(config)
        # 初始化仅包含 MLM 头部的模型
        self.cls = DebertaV2OnlyMLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_output_embeddings(self):
        # 返回输出嵌入的解码器部分
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        # 设置新的输出嵌入到解码器部分
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="[MASK]",
    )
    # 从 transformers.models.deberta.modeling_deberta.DebertaForMaskedLM.forward 复制而来，将 Deberta 改为 DebertaV2
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

        # 根据是否返回字典设置返回结果
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

        # 获取模型输出的序列输出
        sequence_output = outputs[0]
        # 通过分类器预测下一个词的分数
        prediction_scores = self.cls(sequence_output)

        # 初始化masked_lm_loss为None
        masked_lm_loss = None
        # 如果提供了labels，则计算masked language modeling损失
        if labels is not None:
            # 使用交叉熵损失函数，忽略标签为-100的token（padding token）
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果不要求返回字典，则返回元组形式的输出
        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 如果要求返回字典，则返回MaskedLMOutput对象
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 从transformers.models.deberta.modeling_deberta.DebertaPredictionHeadTransform复制而来，将Deberta改为DebertaV2
class DebertaV2PredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)

        # 定义一个全连接层，将隐藏状态的维度映射到嵌入大小
        self.dense = nn.Linear(config.hidden_size, self.embedding_size)
        # 根据配置文件中的激活函数名称或对象选择变换函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # 应用LayerNorm到嵌入维度上，使用配置中的层标准化参数eps
        self.LayerNorm = nn.LayerNorm(self.embedding_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        # 全连接层映射
        hidden_states = self.dense(hidden_states)
        # 应用激活函数变换
        hidden_states = self.transform_act_fn(hidden_states)
        # 应用LayerNorm
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# 从transformers.models.deberta.modeling_deberta.DebertaLMPredictionHead复制而来，将Deberta改为DebertaV2
class DebertaV2LMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化DebertaV2PredictionHeadTransform，用于预测头部的变换
        self.transform = DebertaV2PredictionHeadTransform(config)

        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)
        # 输出权重与输入嵌入相同，但每个标记有一个仅输出的偏置项
        self.decoder = nn.Linear(self.embedding_size, config.vocab_size, bias=False)

        # 初始化一个参数化的偏置项，与每个标记的词汇表大小相对应
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # 需要一个链接以确保偏置项能够与`resize_token_embeddings`正确调整大小
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        # 应用预测头部的变换
        hidden_states = self.transform(hidden_states)
        # 应用线性层进行最终预测
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# 从transformers.models.bert.BertOnlyMLMHead复制而来，将bert改为deberta
class DebertaV2OnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化DebertaV2LMPredictionHead，用于唯一的MLM头部
        self.predictions = DebertaV2LMPredictionHead(config)

    def forward(self, sequence_output):
        # 应用MLM预测头部，生成预测分数
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


@add_start_docstrings(
    """
    在顶部有一个序列分类/回归头部的DeBERTa模型变换器（池化输出之上的线性层），例如用于GLUE任务。
    """,
    DEBERTA_START_DOCSTRING,
)
class DebertaV2ForSequenceClassification(DebertaV2PreTrainedModel):
    # 用于序列分类的DeBERTa模型变换器，继承自DebertaV2PreTrainedModel
    # 初始化函数，接受一个配置参数config作为输入
    def __init__(self, config):
        # 调用父类的初始化函数，将config传递给父类
        super().__init__(config)

        # 从配置参数中获取num_labels，如果没有指定，则默认为2
        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        # 创建一个DebertaV2Model对象，使用给定的config作为参数
        self.deberta = DebertaV2Model(config)
        
        # 创建一个ContextPooler对象，使用给定的config作为参数
        self.pooler = ContextPooler(config)
        
        # 获取ContextPooler的输出维度作为输出维度
        output_dim = self.pooler.output_dim

        # 创建一个线性层用于分类，输入维度为output_dim，输出维度为num_labels
        self.classifier = nn.Linear(output_dim, num_labels)
        
        # 获取配置参数中的cls_dropout，如果未指定，则使用config中的hidden_dropout_prob作为默认值
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        
        # 创建一个稳定的Dropout层，使用上一步得到的drop_out作为参数
        self.dropout = StableDropout(drop_out)

        # 调用post_init函数，用于初始化权重并进行最终处理
        self.post_init()

    # 返回DebertaV2Model对象的输入嵌入层
    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    # 设置DebertaV2Model对象的输入嵌入层为新的嵌入层new_embeddings
    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

    # 使用DebertaForSequenceClassification.forward的文档字符串作为注释
    # 包括Deberta输入的描述和代码示例的描述
    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 从transformers.models.deberta.modeling_deberta.DebertaForSequenceClassification.forward复制并修改为DebertaV2
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
"""
DeBERTa Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
Named-Entity-Recognition (NER) tasks.
"""
# 从 transformers.models.deberta.modeling_deberta.DebertaForTokenClassification 复制并修改为 DebertaV2
class DebertaV2ForTokenClassification(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels  # 初始化分类标签数量

        self.deberta = DebertaV2Model(config)  # 使用 DebertaV2Model 初始化 DeBERTa 模型
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # Dropout 层，用于防止过拟合
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)  # 分类器，线性层映射到标签数量维度

        # 初始化权重并应用最终处理
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 DeBERTa 模型的 forward 方法
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

        sequence_output = outputs[0]  # 取出模型输出的序列输出

        sequence_output = self.dropout(sequence_output)  # 应用 Dropout
        logits = self.classifier(sequence_output)  # 应用分类器线性层

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # 定义交叉熵损失函数
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))  # 计算损失

        if not return_dict:
            output = (logits,) + outputs[1:]  # 输出 logits 和其它附加输出
            return ((loss,) + output) if loss is not None else output

        # 返回 TokenClassifierOutput，包含损失、logits、隐藏状态和注意力
        return TokenClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )


@add_start_docstrings(
    """
    """
    DeBERTa Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    DEBERTA_START_DOCSTRING,



# 描述 DeBERTa 模型，该模型用于提取式问答任务（如 SQuAD），在隐藏状态输出的基础上添加一个用于计算起始位置和结束位置 logit 的线性层作为分类头。
# DEBERTA_START_DOCSTRING 用于引用关于 DeBERTa 模型的文档字符串的常量或变量，可能包含了模型的详细描述和用法说明。
    )
    # 关闭括号，用于结束类定义中的一些参数和装饰器的定义

class DebertaV2ForQuestionAnswering(DebertaV2PreTrainedModel):
    # 定义一个新的类，继承自DebertaV2PreTrainedModel

    def __init__(self, config):
        # 初始化函数，接受一个配置参数config

        super().__init__(config)
        # 调用父类的初始化方法

        self.num_labels = config.num_labels
        # 设置类属性num_labels为config中的num_labels字段值

        self.deberta = DebertaV2Model(config)
        # 创建一个DebertaV2Model实例，传入config作为配置参数，并将其赋值给self.deberta

        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        # 创建一个线性层，将隐藏大小为config.hidden_size映射到标签数为config.num_labels的输出空间

        # Initialize weights and apply final processing
        self.post_init()
        # 调用类中的post_init方法，用于初始化权重和应用最终处理

    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        qa_target_start_index=_QA_TARGET_START_INDEX,
        qa_target_end_index=_QA_TARGET_END_INDEX,
    )
    # 添加文档字符串和代码示例，用于模型的前向传播，根据特定的格式化字符串和样例

    # 从transformers.models.deberta.modeling_deberta.DebertaForQuestionAnswering.forward中复制并将Deberta改为DebertaV2
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
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
        # 初始化 return_dict 变量，如果未提供则使用模型配置中的默认设置
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

        # 获取模型的序列输出
        sequence_output = outputs[0]

        # 将序列输出传递给 QA 输出层得到 logits
        logits = self.qa_outputs(sequence_output)
        # 将 logits 拆分为开始位置和结束位置的 logits
        start_logits, end_logits = logits.split(1, dim=-1)
        # 去除不必要的维度并保持连续性
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        # 初始化总损失为 None
        total_loss = None
        # 如果提供了起始位置和结束位置的标签，则计算损失
        if start_positions is not None and end_positions is not None:
            # 如果在多 GPU 下运行，添加一个维度以匹配 logits 的维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 忽略超出模型输入的位置
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 使用交叉熵损失函数，忽略指定的索引
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        # 如果 return_dict 为 False，则返回一个包含损失和 logits 的元组
        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 如果 return_dict 为 True，则返回一个 QuestionAnsweringModelOutput 对象
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    DeBERTa Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    DEBERTA_START_DOCSTRING,
)
class DebertaV2ForMultipleChoice(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 2)  # 获取配置中的标签数量，默认为2
        self.num_labels = num_labels

        self.deberta = DebertaV2Model(config)  # 初始化DeBERTa模型
        self.pooler = ContextPooler(config)  # 初始化上下文池化器
        output_dim = self.pooler.output_dim  # 获取池化器的输出维度

        self.classifier = nn.Linear(output_dim, 1)  # 创建线性层，用于多选分类任务的分类
        drop_out = getattr(config, "cls_dropout", None)  # 获取配置中的dropout值
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out  # 如果未指定，则使用默认的隐藏层dropout概率
        self.dropout = StableDropout(drop_out)  # 创建稳定的dropout层

        self.init_weights()  # 初始化模型权重

    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()  # 获取输入的嵌入层

    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)  # 设置新的输入嵌入层

    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
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
    ):
        """
        前向传播方法，接受多个输入参数并返回模型输出。

        Args:
            input_ids (Optional[torch.Tensor], optional): 输入的token IDs张量. Defaults to None.
            attention_mask (Optional[torch.Tensor], optional): 注意力掩码张量. Defaults to None.
            token_type_ids (Optional[torch.Tensor], optional): token类型IDs张量. Defaults to None.
            position_ids (Optional[torch.Tensor], optional): 位置IDs张量. Defaults to None.
            inputs_embeds (Optional[torch.Tensor], optional): 输入的嵌入张量. Defaults to None.
            labels (Optional[torch.Tensor], optional): 标签张量. Defaults to None.
            output_attentions (Optional[bool], optional): 是否输出注意力权重. Defaults to None.
            output_hidden_states (Optional[bool], optional): 是否输出隐藏状态. Defaults to None.
            return_dict (Optional[bool], optional): 是否返回字典形式的输出. Defaults to None.

        Returns:
            MultipleChoiceModelOutput: 包含模型输出的命名元组。
        """
        # TODO: Implement forward pass logic here
        pass
    ) -> Union[Tuple, MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 确保返回字典不为None时使用配置中的返回字典设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 计算选择题个数，如果input_ids不为None，则取其第二维度的大小作为选择题数目
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 将输入张量展平，以便用于模型输入
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 调用DeBERTa模型进行推断
        outputs = self.deberta(
            flat_input_ids,
            position_ids=flat_position_ids,
            token_type_ids=flat_token_type_ids,
            attention_mask=flat_attention_mask,
            inputs_embeds=flat_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从模型输出中获取编码层的结果
        encoder_layer = outputs[0]
        # 使用池化器对编码层结果进行池化
        pooled_output = self.pooler(encoder_layer)
        # 对池化结果应用dropout
        pooled_output = self.dropout(pooled_output)
        # 将池化后的结果送入分类器得到logits
        logits = self.classifier(pooled_output)
        # 将logits重塑为(batch_size, num_choices)形状
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        # 如果有标签，则计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 如果不使用返回字典，则返回输出元组
        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 使用MultipleChoiceModelOutput对象包装结果并返回
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```