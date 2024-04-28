# `.\models\deberta\modeling_deberta.py`

```
# 设置编码为 UTF-8
# 版权声明，采用 Apache 2.0 协议
# 导入必要的模块和类
from collections.abc import Sequence  # 导入序列抽象基类
from typing import Optional, Tuple, Union  # 导入类型提示

import torch  # 导入 PyTorch 模块
import torch.utils.checkpoint  # 导入 PyTorch 的检查点工具
from torch import nn  # 导入 PyTorch 的神经网络模块
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss  # 导入损失函数

from ...activations import ACT2FN  # 导入激活函数
from ...modeling_outputs import (  # 导入模型输出类
    BaseModelOutput,
    MaskedLMOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel  # 导入预训练模型基类
from ...pytorch_utils import softmax_backward_data  # 导入反向传播函数
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging  # 导入工具函数和日志记录器
from .configuration_deberta import DebertaConfig  # 导入 DeBERTa 配置类

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器
_CONFIG_FOR_DOC = "DebertaConfig"  # 用于文档的配置文件
_CHECKPOINT_FOR_DOC = "microsoft/deberta-base"  # 用于文档的检查点

# Masked LM 文档字符串相关信息
_CHECKPOINT_FOR_MASKED_LM = "lsanochkin/deberta-large-feedback"  # 用于文档的掩码语言建模模型检查点
_MASKED_LM_EXPECTED_OUTPUT = "' Paris'"  # 期望的掩码语言建模输出
_MASKED_LM_EXPECTED_LOSS = "0.54"  # 期望的掩码语言建模损失

# 问答文档字符串相关信息
_CHECKPOINT_FOR_QA = "Palak/microsoft_deberta-large_squad"  # 用于文档的问答模型检查点
_QA_EXPECTED_OUTPUT = "' a nice puppet'"  # 期望的问答模型输出
_QA_EXPECTED_LOSS = 0.14  # 期望的问答模型损失
_QA_TARGET_START_INDEX = 12  # 问答模型目标开始索引
_QA_TARGET_END_INDEX = 14  # 问答模型目标结束索引

# 预训练模型存档列表
DEBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/deberta-base",
    "microsoft/deberta-large",
    "microsoft/deberta-xlarge",
    "microsoft/deberta-base-mnli",
    "microsoft/deberta-large-mnli",
    "microsoft/deberta-xlarge-mnli",
]

# 上下文池化器类，用于提取上下文特征
class ContextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.pooler_hidden_size, config.pooler_hidden_size)  # 全连接层
        self.dropout = StableDropout(config.pooler_dropout)  # 稳定的 dropout 层
        self.config = config  # 配置对象

    def forward(self, hidden_states):
        # 我们通过简单地选择对应于第一个标记的隐藏状态来“池化”模型。

        context_token = hidden_states[:, 0]  # 提取第一个标记的隐藏状态
        context_token = self.dropout(context_token)  # 应用 dropout
        pooled_output = self.dense(context_token)  # 全连接层
        pooled_output = ACT2FN[self.config.pooler_hidden_act](pooled_output)  # 激活函数
        return pooled_output  # 返回池化后的输出

    @property
    def output_dim(self):
        return self.config.hidden_size  # 返回输出维度


class XSoftmax(torch.autograd.Function):
    """
    用于内存优化的掩码 Softmax
    """
    Args:
        input (`torch.tensor`): The input tensor that will apply softmax.
        mask (`torch.IntTensor`):
            The mask matrix where 0 indicate that element will be ignored in the softmax calculation.
        dim (int): The dimension that will apply softmax

    Example:

    ```python
    >>> import torch
    >>> from transformers.models.deberta.modeling_deberta import XSoftmax

    >>> # Make a tensor
    >>> x = torch.randn([4, 20, 100])

    >>> # Create a mask
    >>> mask = (x > 0).int()

    >>> # Specify the dimension to apply softmax
    >>> dim = -1

    >>> y = XSoftmax.apply(x, mask, dim)
    ```"""

# 定义一个静态方法，用于实现 softmax 操作
class XSoftmax:
    
    @staticmethod
    def forward(self, input, mask, dim):
        # 设置维度
        self.dim = dim
        # 对 mask 取反
        rmask = ~(mask.to(torch.bool))

        # 根据 mask 进行填充
        output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
        # 对填充后的结果进行 softmax 操作
        output = torch.softmax(output, self.dim)
        # 根据 mask 进行填充
        output.masked_fill_(rmask, 0)
        # 保存计算结果以备反向传播使用
        self.save_for_backward(output)
        return output

    @staticmethod
    def backward(self, grad_output):
        # 获取保存的计算结果
        (output,) = self.saved_tensors
        # 计算输入的梯度
        inputGrad = softmax_backward_data(self, grad_output, output, self.dim, output)
        return inputGrad, None, None

    @staticmethod
    def symbolic(g, self, mask, dim):
        import torch.onnx.symbolic_helper as sym_help
        from torch.onnx.symbolic_opset9 import masked_fill, softmax

        # 将 mask 转换为 long 类型
        mask_cast_value = g.op("Cast", mask, to_i=sym_help.cast_pytorch_to_onnx["Long"])
        # 取 mask 的反
        r_mask = g.op(
            "Cast",
            g.op("Sub", g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64)), mask_cast_value),
            to_i=sym_help.cast_pytorch_to_onnx["Bool"],
        )
        # 根据 mask 进行填充
        output = masked_fill(
            g, self, r_mask, g.op("Constant", value_t=torch.tensor(torch.finfo(self.type().dtype()).min))
        )
        # 对填充后的结果进行 softmax 操作
        output = softmax(g, output, dim)
        return masked_fill(g, output, r_mask, g.op("Constant", value_t=torch.tensor(0, dtype=torch.bool)))
class DropoutContext(object):
    # 定义 Dropout 上下文类
    def __init__(self):
        self.dropout = 0
        self.mask = None
        self.scale = 1
        self.reuse_mask = True


def get_mask(input, local_context):
    # 获取掩码值并应用到输入数据中
    if not isinstance(local_context, DropoutContext):
        dropout = local_context
        mask = None
    else:
        dropout = local_context.dropout
        dropout *= local_context.scale
        mask = local_context.mask if local_context.reuse_mask else None

    # 如果有 dropout 并且没有掩码值，根据概率生成掩码值
    if dropout > 0 and mask is None:
        mask = (1 - torch.empty_like(input).bernoulli_(1 - dropout)).to(torch.bool)

    # 如果是 DropoutContext 类型，存储掩码值
    if isinstance(local_context, DropoutContext):
        if local_context.mask is None:
            local_context.mask = mask

    # 返回掩码值和 dropout 值
    return mask, dropout


class XDropout(torch.autograd.Function):
    """Optimized dropout function to save computation and memory by using mask operation instead of multiplication."""

    # 前向传播函数
    @staticmethod
    def forward(ctx, input, local_ctx):
        mask, dropout = get_mask(input, local_ctx)
        ctx.scale = 1.0 / (1 - dropout)
        # 如果有 dropout，应用掩码值到输入并缩放
        if dropout > 0:
            ctx.save_for_backward(mask)
            return input.masked_fill(mask, 0) * ctx.scale
        else:
            return input

    # 反向传播函数
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.scale > 1:
            (mask,) = ctx.saved_tensors
            return grad_output.masked_fill(mask, 0) * ctx.scale, None
        else:
            return grad_output, None

    # 符号化函数
    @staticmethod
    def symbolic(g: torch._C.Graph, input: torch._C.Value, local_ctx: Union[float, DropoutContext]) -> torch._C.Value:
        from torch.onnx import symbolic_opset12

        dropout_p = local_ctx
        if isinstance(local_ctx, DropoutContext):
            dropout_p = local_ctx.dropout
        # 只有在训练中才调用此函数
        train = True
        # 返回符号化的 Dropout 函数
        return symbolic_opset12.dropout(g, input, dropout_p, train)


class StableDropout(nn.Module):
    """
    Optimized dropout module for stabilizing the training

    Args:
        drop_prob (float): the dropout probabilities
    """

    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob
        self.count = 0
        self.context_stack = None

    def forward(self, x):
        """
        Call the module

        Args:
            x (`torch.tensor`): The input tensor to apply dropout
        """
        # 如果在训练并且有 dropout 概率，应用 XDropout 模块
        if self.training and self.drop_prob > 0:
            return XDropout.apply(x, self.get_context())
        return x
    # 清除上下文信息的方法
    def clear_context(self):
        # 将计数器归零
        self.count = 0
        # 清空上下文栈
        self.context_stack = None

    # 初始化上下文信息的方法
    def init_context(self, reuse_mask=True, scale=1):
        # 如果上下文栈为空，则初始化为一个空列表
        if self.context_stack is None:
            self.context_stack = []
        # 将计数器归零
        self.count = 0
        # 遍历上下文栈中的每个上下文对象
        for c in self.context_stack:
            # 设置重用掩码为传入参数中的值
            c.reuse_mask = reuse_mask
            # 设置比例为传入参数中的值
            c.scale = scale

    # 获取上下文信息的方法
    def get_context(self):
        # 如果上下文栈不为空
        if self.context_stack is not None:
            # 如果计数器超出上下文栈的长度
            if self.count >= len(self.context_stack):
                # 在上下文栈中添加一个新的DropoutContext对象
                self.context_stack.append(DropoutContext())
            # 获取当前计数器指向的上下文对象
            ctx = self.context_stack[self.count]
            # 设置当前上下文对象的丢弃概率为预设值
            ctx.dropout = self.drop_prob
            # 计数器加一
            self.count += 1
            # 返回当前上下文对象
            return ctx
        else:
            # 如果上下文栈为空，则返回预设的丢弃概率值
            return self.drop_prob
class DebertaLayerNorm(nn.Module):
    """LayerNorm module in the TF style (epsilon inside the square root)."""

    def __init__(self, size, eps=1e-12):
        # 初始化 DebertaLayerNorm 类
        super().__init__()
        # 添加可训练参数 weight 和 bias
        self.weight = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # 保存输入的数据类型
        input_type = hidden_states.dtype
        # 将输入数据类型转换为 float
        hidden_states = hidden_states.float()
        # 计算数据的均值
        mean = hidden_states.mean(-1, keepdim=True)
        # 计算方差
        variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
        # 进行 LayerNorm 操作
        hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
        # 恢复数据类型
        hidden_states = hidden_states.to(input_type)
        # 线性变换和偏置操作
        y = self.weight * hidden_states + self.bias
        # 返回结果
        return y


class DebertaSelfOutput(nn.Module):
    def __init__(self, config):
        # 初始化 DebertaSelfOutput 类
        super().__init__()
        # 线性变换层
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # Deberta 风格的 LayerNorm
        self.LayerNorm = DebertaLayerNorm(config.hidden_size, config.layer_norm_eps)
        # 使用稳定的 Dropout
        self.dropout = StableDropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # 线性变换
        hidden_states = self.dense(hidden_states)
        # Dropout 操作
        hidden_states = self.dropout(hidden_states)
        # LayerNorm 操作
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回结果
        return hidden_states


class DebertaAttention(nn.Module):
    def __init__(self, config):
        # 初始化 DebertaAttention 类
        super().__init__()
        # 自注意力层
        self.self = DisentangledSelfAttention(config)
        # SelfOutput 层
        self.output = DebertaSelfOutput(config)
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
        # SelfAttention 操作
        self_output = self.self(
            hidden_states,
            attention_mask,
            output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        # 如果需要输出注意力分��矩阵
        if output_attentions:
            self_output, att_matrix = self_output
        # 如果没有提供查询状态，则使用隐藏状态
        if query_states is None:
            query_states = hidden_states
        # 输出注意力结果
        attention_output = self.output(self_output, query_states)

        # 如果需要输出注意力分布矩阵
        if output_attentions:
            return (attention_output, att_matrix)
        else:
            return attention_output


# 从 transformers.models.bert.modeling_bert.BertIntermediate 复制并修改为 DebertaIntermediate
class DebertaIntermediate(nn.Module):
    def __init__(self, config):
        # 初始化 DebertaIntermediate 类
        super().__init__()
        # 线性变换层
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
    # 前向传播函数，接受隐藏状态张量作为输入，返回处理后的隐藏状态张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用全连接层对隐藏状态张量进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的隐藏状态张量应用激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回处理后的隐藏状态张量
        return hidden_states
# 定义一个 DebertaOutput 类，继承自 nn.Module 类
class DebertaOutput(nn.Module):
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个全连接层，输入大小为配置对象中的 intermediate_size，输出大小为配置对象中的 hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个 DebertaLayerNorm 层，用于层归一化，输入大小为配置对象中的 hidden_size
        self.LayerNorm = DebertaLayerNorm(config.hidden_size, config.layer_norm_eps)
        # 创建一个稳定的 Dropout 层，用于随机断开连接，概率为配置对象中的 hidden_dropout_prob
        self.dropout = StableDropout(config.hidden_dropout_prob)
        # 将配置对象保存到类属性中
        self.config = config

    # 前向传播函数，接受隐藏状态和输入张量作为参数
    def forward(self, hidden_states, input_tensor):
        # 将隐藏状态输入全连接层进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换的结果进行 Dropout
        hidden_states = self.dropout(hidden_states)
        # 将 Dropout 后的结果与输入张量相加，并输入到层归一化层中
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回归一化后的结果
        return hidden_states


# 定义一个 DebertaLayer 类，继承自 nn.Module 类
class DebertaLayer(nn.Module):
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个 DebertaAttention 层
        self.attention = DebertaAttention(config)
        # 创建一个 DebertaIntermediate 层
        self.intermediate = DebertaIntermediate(config)
        # 创建一个 DebertaOutput 层
        self.output = DebertaOutput(config)

    # 前向传播函数，接受隐藏状态、注意力掩码等参数
    def forward(
        self,
        hidden_states,
        attention_mask,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
        output_attentions=False,
    ):
        # 将隐藏状态输入注意力层
        attention_output = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        # 如果输出注意力权重，则将注意力输出和注意力矩阵分开
        if output_attentions:
            attention_output, att_matrix = attention_output
        # 将注意力输出输入到中间层
        intermediate_output = self.intermediate(attention_output)
        # 将中间层输出输入到输出层
        layer_output = self.output(intermediate_output, attention_output)
        # 如果输出注意力权重，则返回层输出和注意力矩阵
        if output_attentions:
            return (layer_output, att_matrix)
        # 否则，只返回层输出
        else:
            return layer_output


# 定义一个 DebertaEncoder 类，继承自 nn.Module 类
class DebertaEncoder(nn.Module):
    """Modified BertEncoder with relative position bias support"""

    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建多个 DebertaLayer 层，根据配置对象中的隐藏层数量
        self.layer = nn.ModuleList([DebertaLayer(config) for _ in range(config.num_hidden_layers)])
        # 是否支持相对位置偏置
        self.relative_attention = getattr(config, "relative_attention", False)
        # 如果支持相对位置偏置
        if self.relative_attention:
            # 获取最大相对位置偏置数量，如果未设置则使用最大位置嵌入数量
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            # 创建一个相对位置嵌入层
            self.rel_embeddings = nn.Embedding(self.max_relative_positions * 2, config.hidden_size)
        # 梯度检查点标志，默认为 False
        self.gradient_checkpointing = False

    # 获取相对位置嵌入
    def get_rel_embedding(self):
        rel_embeddings = self.rel_embeddings.weight if self.relative_attention else None
        return rel_embeddings

    # 获取注意力掩码
    def get_attention_mask(self, attention_mask):
        # 如果注意力掩码维度小于等于2，则扩展维度以匹配注意力张量
        if attention_mask.dim() <= 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
        # 如果注意力掩码维度为3，则在第二个维度上添加一个维度
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)

        return attention_mask
    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        # 如果开启了相对位置编码并且未传入相对位置编码，根据查询状态的大小和隐藏状态的大小产生相对位置编码
        if self.relative_attention and relative_pos is None:
            q = query_states.size(-2) if query_states is not None else hidden_states.size(-2)
            relative_pos = build_relative_position(q, hidden_states.size(-2), hidden_states.device)
        # 返回相对位置编码
        return relative_pos

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
        # 获取相对位置编码
        relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)

        # 初始化输出的隐藏状态和注意力
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # 如果隐藏状态是一个序列，取第一个隐藏状态作为下一个键值对隐藏状态
        if isinstance(hidden_states, Sequence):
            next_kv = hidden_states[0]
        else:
            next_kv = hidden_states
        # 获取相对位置embedding
        rel_embeddings = self.get_rel_embedding()
        # 遍历所有层
        for i, layer_module in enumerate(self.layer):
            # 如果输出隐藏状态为True，则将当前隐藏状态添加到输出隐藏状态中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果开启了渐进检查点，且处于训练模式
            if self.gradient_checkpointing and self.training:
                # 使用梯度检查点调用当前层，得到下一个键值对隐藏状态
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
                # 调用当前层，得到下一个键值对隐藏状态
                hidden_states = layer_module(
                    next_kv,
                    attention_mask,
                    query_states=query_states,
                    relative_pos=relative_pos,
                    rel_embeddings=rel_embeddings,
                    output_attentions=output_attentions,
                )

            # 如果输出注意力为True，则从隐藏状态中分离出注意力矩阵
            if output_attentions:
                hidden_states, att_m = hidden_states

            # 如果传入了查询状态，则将当前隐藏状态作为查询状态
            if query_states is not None:
                query_states = hidden_states
                # 如果隐藏状态是一个序列，则取下一个隐藏状态
                if isinstance(hidden_states, Sequence):
                    next_kv = hidden_states[i + 1] if i + 1 < len(self.layer) else None
            else:
                # 否则，将当前隐藏状态作为下一个键值对隐藏状态
                next_kv = hidden_states

            # 如果输出注意力为True，则将注意力矩阵添加到输出注意力中
            if output_attentions:
                all_attentions = all_attentions + (att_m,)

        # 如果输出隐藏状态为True，则将最后的隐藏状态添加到输出隐藏状态中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果返回字典为True，则返回BaseModelOutput对象
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )
# 根据查询和键构建相对位置

# 假设查询的绝对位置范围为 (0, query_size)，键的绝对位置范围为 (0, key_size)，
# 则从查询到键的相对位置为 R_{q → k} = P_q - P_k

# 参数：
#   query_size (int): 查询的长度
#   key_size (int): 键的长度
# 返回：
#   `torch.LongTensor`: 形状为 [1, query_size, key_size] 的张量
def build_relative_position(query_size, key_size, device):
    # 创建一个设备相关的长整型张量，范围从 0 到 query_size-1，表示查询的位置索引
    q_ids = torch.arange(query_size, dtype=torch.long, device=device)
    # 创建一个设备相关的长整型张量，范围从 0 到 key_size-1，表示键的位置索引
    k_ids = torch.arange(key_size, dtype=torch.long, device=device)
    # 计算查询到键的相对位置索引
    rel_pos_ids = q_ids[:, None] - k_ids.view(1, -1).repeat(query_size, 1)
    # 截取相对位置索引张量的部分，保留查询的长度，即 [0, query_size-1] 行和所有列
    rel_pos_ids = rel_pos_ids[:query_size, :]
    # 在第 0 维度上添加一个维度，以匹配输出张量的形状 [1, query_size, key_size]
    rel_pos_ids = rel_pos_ids.unsqueeze(0)
    return rel_pos_ids


# 动态扩展从子到父的位置索引张量
@torch.jit.script
def c2p_dynamic_expand(c2p_pos, query_layer, relative_pos):
    # 在最后一个维度上复制子到父位置索引张量，以匹配查询层的形状
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), query_layer.size(2), relative_pos.size(-1)])


# 动态扩展从父到子的位置索引张量
@torch.jit.script
def p2c_dynamic_expand(c2p_pos, query_layer, key_layer):
    # 在最后两个维度上复制子到父位置索引张量，以匹配查询层和键层的形状
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), key_layer.size(-2), key_layer.size(-2)])


# 动态扩展位置索引张量
@torch.jit.script
def pos_dynamic_expand(pos_index, p2c_att, key_layer):
    # 在前两个维度上复制位置索引张量，以匹配 p2c_att 和键层的形状
    return pos_index.expand(p2c_att.size()[:2] + (pos_index.size(-2), key_layer.size(-2)))


# 分解的自注意力模块
class DisentangledSelfAttention(nn.Module):
    """
    分解的自注意力模块

    参数:
        config (`str`):
            包含用于构建新模型的配置的模型配置类实例。模式与 *BertConfig* 类似，更多细节请参考 [`DebertaConfig`]
    """
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 检查隐藏层大小是否是注意力头的数量的整数倍，如果不是则抛出数值错误异常
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        # 设置注意力头的数量
        self.num_attention_heads = config.num_attention_heads
        # 计算每个注意力头的大小
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        # 计算所有头的总大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # 输入层的线性变换
        self.in_proj = nn.Linear(config.hidden_size, self.all_head_size * 3, bias=False)
        # 查询（query）偏置
        self.q_bias = nn.Parameter(torch.zeros((self.all_head_size), dtype=torch.float))
        # 值（value）偏置
        self.v_bias = nn.Parameter(torch.zeros((self.all_head_size), dtype=torch.float))
        # 设置位置注意力类型
        self.pos_att_type = config.pos_att_type if config.pos_att_type is not None else []

        # 是否使用相对注意力
        self.relative_attention = getattr(config, "relative_attention", False)
        # 是否使用“talking head”注意力
        self.talking_head = getattr(config, "talking_head", False)

        # 如果使用“talking head”注意力
        if self.talking_head:
            # 设置头权重的投影
            self.head_logits_proj = nn.Linear(config.num_attention_heads, config.num_attention_heads, bias=False)
            self.head_weights_proj = nn.Linear(config.num_attention_heads, config.num_attention_heads, bias=False)

        # 如果使用相对注意力
        if self.relative_attention:
            # 获取最大的相对位置
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                # 如果最大相对位置小于1，则设置为配置对象中的最大位置嵌入数量
                self.max_relative_positions = config.max_position_embeddings
            # 位置丢弃，用于相对位置注意力
            self.pos_dropout = StableDropout(config.hidden_dropout_prob)

            # 如果位置类型中包括“c2p”
            if "c2p" in self.pos_att_type:
                # 设置位置投影
                self.pos_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
            # 如果位置类型中包括“p2c”
            if "p2c" in self.pos_att_type:
                # 设置查询位置投影
                self.pos_q_proj = nn.Linear(config.hidden_size, self.all_head_size)

        # 注意力头的概率丢弃
        self.dropout = StableDropout(config.attention_probs_dropout_prob)

    # 为得分调整张量的形状
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, -1)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播函数
    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
    # 计算注意力偏置，根据查询、键值、相对位置、相对位置嵌入和缩放因子
    def disentangled_att_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        # 如果未提供相对位置信息，则根据查询和键值大小构建相对位置信息
        if relative_pos is None:
            q = query_layer.size(-2)
            relative_pos = build_relative_position(q, key_layer.size(-2), query_layer.device)
        # 如果相对位置维度为 2，则增加两个维度
        if relative_pos.dim() == 2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        # 如果相对位置维度为 3，则在第二维度处增加一个维度
        elif relative_pos.dim() == 3:
            relative_pos = relative_pos.unsqueeze(1)
        # 如果相对位置维度不为 4，则引发数值异常
        elif relative_pos.dim() != 4:
            raise ValueError(f"Relative position ids must be of dim 2 or 3 or 4. {relative_pos.dim()}")

        # 计算最大注意力范围
        att_span = min(max(query_layer.size(-2), key_layer.size(-2)), self.max_relative_positions)
        # 将相对位置转换为长整型，并移动到相同设备上
        relative_pos = relative_pos.long().to(query_layer.device)
        # 从相对位置嵌入中选取特定范围的内容并增加一个维度
        rel_embeddings = rel_embeddings[
            self.max_relative_positions - att_span : self.max_relative_positions + att_span, :
        ].unsqueeze(0)

        # 初始化分数变量
        score = 0

        # content->position
        # 如果位置注意力类型包含"c2p"
        if "c2p" in self.pos_att_type:
            # 对位置信息进行投影
            pos_key_layer = self.pos_proj(rel_embeddings)
            pos_key_layer = self.transpose_for_scores(pos_key_layer)
            # 矩阵相乘计算注意力
            c2p_att = torch.matmul(query_layer, pos_key_layer.transpose(-1, -2))
            # 对相对位置进行裁剪
            c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
            # 根据裁剪的位置选择相应的内容进行加权求和
            c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_dynamic_expand(c2p_pos, query_layer, relative_pos))
            score += c2p_att

        # position->content
        # 如果位置注意力类型包含"p2c"
        if "p2c" in self.pos_att_type:
            # 对查询进行位置投影
            pos_query_layer = self.pos_q_proj(rel_embeddings)
            pos_query_layer = self.transpose_for_scores(pos_query_layer)
            # 对位置查询进行归一化
            pos_query_layer /= torch.sqrt(torch.tensor(pos_query_layer.size(-1), dtype=torch.float) * scale_factor)
            # 如果查询和键值长度不相等，则重新构建相对位置信息
            if query_layer.size(-2) != key_layer.size(-2):
                r_pos = build_relative_position(key_layer.size(-2), key_layer.size(-2), query_layer.device)
            else:
                r_pos = relative_pos
            # 对相对位置进行裁剪
            p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span * 2 - 1)
            # 矩阵相乘计算注意力，根据裁剪的位置选择相应的内容进行加权求和并转置
            p2c_att = torch.matmul(key_layer, pos_query_layer.transpose(-1, -2).to(dtype=key_layer.dtype))
            p2c_att = torch.gather(
                p2c_att, dim=-1, index=p2c_dynamic_expand(p2c_pos, query_layer, key_layer)
            ).transpose(-1, -2)

            # 如果查询和键值长度不相等，则根据相对位置信息对注意力进行进一步调整
            if query_layer.size(-2) != key_layer.size(-2):
                pos_index = relative_pos[:, :, :,
class DebertaEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        # 初始化 DebertaEmbeddings 类
        super().__init__()
        # 获取配置中的 pad_token_id，默认为 0
        pad_token_id = getattr(config, "pad_token_id", 0)
        # 获取配置中的 embedding_size，默认为 config.hidden_size
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)
        # 创建词嵌入层，vocab_size 表示词汇表大小
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embedding_size, padding_idx=pad_token_id)

        # 获取配置中的 position_biased_input，默认为 True
        self.position_biased_input = getattr(config, "position_biased_input", True)
        # 如果不使用位置偏置输入
        if not self.position_biased_input:
            # 置位置嵌入为 None
            self.position_embeddings = None
        else:
            # 创建位置嵌入层，max_position_embeddings 表示最大位置嵌入数
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.embedding_size)

        # 如果配置中的 type_vocab_size 大于 0
        if config.type_vocab_size > 0:
            # 创建类型嵌入层，type_vocab_size 表示类型嵌入大小
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, self.embedding_size)

        # 如果嵌入大小不等于隐藏大小
        if self.embedding_size != config.hidden_size:
            # 创建线性变换层，将嵌入大小映射到隐藏大小
            self.embed_proj = nn.Linear(self.embedding_size, config.hidden_size, bias=False)
        # 创建 DebertaLayerNorm 层，用于层标准化
        self.LayerNorm = DebertaLayerNorm(config.hidden_size, config.layer_norm_eps)
        # 创建稳定的 Dropout 层
        self.dropout = StableDropout(config.hidden_dropout_prob)
        # 保存配置
        self.config = config

        # 位置标识符（1，位置嵌入长度）在内存中是连续的，并且在序列化时被导出
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
    # 定义前向传播函数，用于生成模型的嵌入表示
    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, mask=None, inputs_embeds=None):
        # 如果传入了 input_ids 参数
        if input_ids is not None:
            # 获取 input_ids 的形状
            input_shape = input_ids.size()
        else:
            # 否则获取 inputs_embeds 的形状，并去掉最后一维
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列的长度
        seq_length = input_shape[1]

        # 如果未提供 position_ids，则从预先定义的位置嵌入矩阵中选择与序列长度相匹配的部分
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # 如果未提供 token_type_ids，则创建与输入形状相同的零张量
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果未提供 inputs_embeds，则通过调用 word_embeddings 方法根据 input_ids 获取输入的嵌入向量
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # 如果存在位置嵌入模块，则根据 position_ids 获取位置嵌入向量，否则创建一个与输入嵌入向量形状相同的零张量
        if self.position_embeddings is not None:
            position_embeddings = self.position_embeddings(position_ids.long())
        else:
            position_embeddings = torch.zeros_like(inputs_embeds)

        # 将输入嵌入向量与位置嵌入向量相加
        embeddings = inputs_embeds
        if self.position_biased_input:
            embeddings += position_embeddings

        # 如果配置中的类型词汇大小大于0，则根据 token_type_ids 获取类型词嵌入向量，并将其与嵌入向量相加
        if self.config.type_vocab_size > 0:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings

        # 如果嵌入向量的大小与隐藏大小不相等，则通过 embed_proj 方法将嵌入向量投影到隐藏大小
        if self.embedding_size != self.config.hidden_size:
            embeddings = self.embed_proj(embeddings)

        # 对嵌入向量进行 LayerNorm 归一化处理
        embeddings = self.LayerNorm(embeddings)

        # 如果提供了掩码，则将掩码应用于嵌入向量
        if mask is not None:
            # 如果掩码维度与嵌入向量维度不同，则根据情况对掩码进行调整
            if mask.dim() != embeddings.dim():
                if mask.dim() == 4:
                    mask = mask.squeeze(1).squeeze(1)
                mask = mask.unsqueeze(2)
            # 将掩码转换为嵌入向量的数据类型，并将其应用于嵌入向量
            mask = mask.to(embeddings.dtype)
            embeddings = embeddings * mask

        # 对嵌入向量进行 dropout 处理
        embeddings = self.dropout(embeddings)
        # 返回嵌入向量
        return embeddings
class DebertaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    
    # 设置配置类为 DebertaConfig
    config_class = DebertaConfig
    # 设置基础模型前缀为 "deberta"
    base_model_prefix = "deberta"
    # 加载时忽略的键列表
    _keys_to_ignore_on_load_unexpected = ["position_embeddings"]
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights."""
        # 若为线性层
        if isinstance(module, nn.Linear):
            # 初始化权重为正态分布，均值为0，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 若存在偏置，则初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 若为嵌入层
        elif isinstance(module, nn.Embedding):
            # 初始化权重为正态分布，均值为0，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 若存在填充索引，则将填充索引对应的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


DEBERTA_START_DOCSTRING = r"""
    The DeBERTa model was proposed in [DeBERTa: Decoding-enhanced BERT with Disentangled
    Attention](https://arxiv.org/abs/2006.03654) by Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen. It's build
    on top of BERT/RoBERTa with two improvements, i.e. disentangled attention and enhanced mask decoder. With those two
    improvements, it out perform BERT/RoBERTa on a majority of tasks with 80GB pretraining data.

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.


    Parameters:
        config ([`DebertaConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

DEBERTA_INPUTS_DOCSTRING = r"""
    Args:
        # 输入序列标记在词汇表中的索引
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        # 避免在填充标记索引上执行注意力的掩码
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        # 段标记索引，指示输入的第一部分和第二部分。索引在`[0,1]`中选择
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        # 指示每个输入序列标记在位置嵌入中的位置索引。选择范围为`[0, config.max_position_embeddings - 1]`
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        # 输入嵌入。
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
            model's internal embedding lookup matrix.
        # 是否返回所有注意力层的注意力张量
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        # 是否返回所有层的隐藏状态
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        # 是否返回一个`~utils.ModelOutput`而不是普通元组
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
# 使用 add_start_docstrings 装饰器添加模型文档字符串，指定 DeBERTa 模型的配置信息
@add_start_docstrings(
    "The bare DeBERTa Model transformer outputting raw hidden-states without any specific head on top.",
    DEBERTA_START_DOCSTRING,
)
# 定义 DebertaModel 类，继承自 DebertaPreTrainedModel
class DebertaModel(DebertaPreTrainedModel):
    # 初始化方法，接受配置参数
    def __init__(self, config):
        super().__init__(config)
        # 初始化模型的嵌入层和编码器
        self.embeddings = DebertaEmbeddings(config)
        self.encoder = DebertaEncoder(config)
        # 初始化步骤数和配置信息
        self.z_steps = 0
        self.config = config
        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入的方法
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入嵌入的方法
    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    # 剪枝模型头部的方法
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError("The prune function is not implemented in DeBERTa model.")

    # 使用 add_start_docstrings_to_model_forward 装饰器添加模型前向方法的文档字符串
    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 使用 add_code_sample_docstrings 装饰器添加模型前向方法的代码示例文档字符串
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义模型前向方法，接受多个输入参数
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
        # 指定函数的返回类型，可以是元组或者BaseModelOutput类型

        # 如果output_attentions未指定，则使用配置中的output_attentions值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果output_hidden_states未指定，则使用配置中的output_hidden_states值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果return_dict未指定，则使用配置中的use_return_dict值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果同时指定了input_ids和inputs_embeds，则抛出数值错误
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        # 如果指定了input_ids
        elif input_ids is not None:
            # 如果input_ids存在padding并且没有attention_mask，则警告
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            # 获取input_ids的形状
            input_shape = input_ids.size()
        # 如果指定了inputs_embeds
        elif inputs_embeds is not None:
            # 获取inputs_embeds的形状，去掉最后一维（通常是embedding维度）
            input_shape = inputs_embeds.size()[:-1]
        # 如果既没有指定input_ids也没有指定inputs_embeds，则抛出数值错误
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 获取input_ids或inputs_embeds所在的设备
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # 如果attention_mask为空，则创建全1的attention_mask，形状和设备与input_ids或inputs_embeds匹配
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        # 如果token_type_ids为空，则创建全0的token_type_ids，数据类型为long，形状和设备与input_ids或inputs_embeds匹配
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # 通过embeddings模块获取embedding输出
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        # 通过encoder模块获取编码器输出
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )
        # 获取编码器的所有输出层
        encoded_layers = encoder_outputs[1]

        # 如果z_steps大于1
        if self.z_steps > 1:
            # 获取倒数第二层的隐藏状态
            hidden_states = encoded_layers[-2]
            # 创建包含多个self.encoder.layer[-1]的列表
            layers = [self.encoder.layer[-1] for _ in range(self.z_steps)]
            # 获取倒数第一层的查询状态
            query_states = encoded_layers[-1]
            # 获取关系嵌入
            rel_embeddings = self.encoder.get_rel_embedding()
            # 获取注意力掩码
            attention_mask = self.encoder.get_attention_mask(attention_mask)
            # 获取相对位置嵌入
            rel_pos = self.encoder.get_rel_pos(embedding_output)
            # 遍历除了第一层的所有层进行计算
            for layer in layers[1:]:
                query_states = layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=False,
                    query_states=query_states,
                    relative_pos=rel_pos,
                    rel_embeddings=rel_embeddings,
                )
                # 添加查询状态到编码层列表中
                encoded_layers.append(query_states)

        # 获取序列输出
        sequence_output = encoded_layers[-1]

        # 如果不返回字典，则返回元组
        if not return_dict:
            return (sequence_output,) + encoder_outputs[(1 if output_hidden_states else 2) :]

        # 返回BaseModelOutput类型的结果
        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions,
        )
# 给 DeBERTa 模型添加一个语言建模头部
@add_start_docstrings("""DeBERTa Model with a `language modeling` head on top.""", DEBERTA_START_DOCSTRING)
# 创建 DebertaForMaskedLM 类，继承自 DebertaPreTrainedModel
class DebertaForMaskedLM(DebertaPreTrainedModel):
    # 定义 tied_weights_keys 列表
    _tied_weights_keys = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]

    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建 DebertaModel 对象
        self.deberta = DebertaModel(config)
        # 创建 DebertaOnlyMLMHead 对象
        self.cls = DebertaOnlyMLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 前向传播方法
    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_MASKED_LM,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="[MASK]",
        expected_output=_MASKED_LM_EXPECTED_OUTPUT,
        expected_loss=_MASKED_LM_EXPECTED_LOSS,
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
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional`):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        # 指定返回dict是由输入参数还是由配置文件决定
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用DeBERTa模型，获取输出
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
        # 通过全连接层获取预测得分
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        # 如果提供了标签，则计算masked language modeling损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果不需要返回dict，则构造输出元组，并返回
        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 返回MaskedLMOutput对象，包括损失、logits、hidden_states和attentions
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
class DebertaPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 从配置中获取embedding_size，如果没有则使用hidden_size
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)

        # 初始化一个全连接层，将hidden_size映射到embedding_size
        self.dense = nn.Linear(config.hidden_size, self.embedding_size)
        # 根据配置中的hidden_act选择激活函数，并将其保存为transform_act_fn
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # 初始化LayerNorm，输入维度为embedding_size
        self.LayerNorm = nn.LayerNorm(self.embedding_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        # 将hidden_states经过全连接层映射到embedding_size
        hidden_states = self.dense(hidden_states)
        # 使用transform_act_fn激活函数处理hidden_states
        hidden_states = self.transform_act_fn(hidden_states)
        # 使用LayerNorm对hidden_states进行归一化处理
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class DebertaLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个DebertaPredictionHeadTransform对象
        self.transform = DebertaPredictionHeadTransform(config)

        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)
        # 输出权重与输入嵌入相同，但每个标记有一个只有输出的偏置
        self.decoder = nn.Linear(self.embedding_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # 链接这两个变量，以便偏置在resize_token_embeddings时能正确调整大小
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        # 将hidden_states传递给transform处理
        hidden_states = self.transform(hidden_states)
        # 使用decoder对hidden_states进行处理
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# 从transformers.models.bert.BertOnlyMLMHead中复制代码，将bert改为deberta
class DebertaOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个DebertaLMPredictionHead对象
        self.predictions = DebertaLMPredictionHead(config)

    def forward(self, sequence_output):
        # 将sequence_output传递给predictions处理
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


@add_start_docstrings(
    """
    DeBERTa Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    DEBERTA_START_DOCSTRING,
)
# 声明一个类DebertaForSequenceClassification继承自DebertaPreTrainedModel
class DebertaForSequenceClassification(DebertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        # 初始化一个DebertaModel对象
        self.deberta = DebertaModel(config)
        # 初始化一个ContextPooler对象
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        # 初始化一个全连接层，将output_dim映射到num_labels
        self.classifier = nn.Linear(output_dim, num_labels)
        # 获取配置中的cls_dropout，如果没有则使用config.hidden_dropout_prob
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        # 初始化一个StableDropout对象
        self.dropout = StableDropout(drop_out)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()
    # 设置输入 embeddings，替换原有的 embeddings
    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

    # 添加模型前向传播的文档字符串和示例代码文档字符串
    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 模型前向传播函数，接受多种输入参数
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入的词 ID
        attention_mask: Optional[torch.Tensor] = None,  # 注意力遮罩
        token_type_ids: Optional[torch.Tensor] = None,  # 标记类型 ID
        position_ids: Optional[torch.Tensor] = None,  # 位置 ID
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入的嵌入向量
        labels: Optional[torch.Tensor] = None,  # 标签
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 返回结果的类型
# 为 DebertaForTokenClassification 类添加文档字符串和基类
@add_start_docstrings(
    """
    DeBERTa Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    DEBERTA_START_DOCSTRING,
)
class DebertaForTokenClassification(DebertaPreTrainedModel):
    # 初始化函数
    def __init__(self, config):
        # 调用基类的初始化函数
        super().__init__(config)
        # 设置标签数量
        self.num_labels = config.num_labels

        # 创建一个 DebertaModel 对象
        self.deberta = DebertaModel(config)
        # 创建一个丢弃层对象
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 创建一个线性层对象
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数
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

        # 调用 DebertaModel 的前向传播函数
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

        sequence_output = outputs[0]

        # 对输出进行丢弃
        sequence_output = self.dropout(sequence_output)
        # 使用线性层进行分类
        logits = self.classifier(sequence_output)

        loss = None
        # 如果有标签，则计算损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不使用返回字典，则返回输出
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 使用 TokenClassifierOutput 返回输出
        return TokenClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )


# 为 DebertaForTokenClassification 类添加文档字符串和基类
@add_start_docstrings(
    """
    DeBERTa Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    """,
    提供了对隐藏状态输出的额外层，用于计算`起始标记logits`和`结束标记logits`。
    """,
    DEBERTA_START_DOCSTRING,
)

# 定义 DebertaForQuestionAnswering 类，继承自 DebertaPreTrainedModel 类
class DebertaForQuestionAnswering(DebertaPreTrainedModel):
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置 num_labels 属性为配置中的 num_labels
        self.num_labels = config.num_labels

        # 创建 DebertaModel 对象
        self.deberta = DebertaModel(config)
        # 创建一个线性层来处理问题回答的输出
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 定义前向传播方法，对输入进行处理并返回结果
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
    # 定义一个方法，输入为input_ids、attention_mask、token_type_ids、position_ids、inputs_embeds、output_attentions、output_hidden_states，输出为Tuple或者QuestionAnsweringModelOutput
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        
        # 如果return_dict不为空，则使用return_dict；否则，使用self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 对输入的参数进行计算得到outputs
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

        # 从outputs中获取sequence_output
        sequence_output = outputs[0]

        # 通过sequence_output计算logits
        logits = self.qa_outputs(sequence_output)
        
        # 将logits分割为start_logits和end_logits
        start_logits, end_logits = logits.split(1, dim=-1)
        
        # 对start_logits和end_logits进行处理
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        
        # 如果start_positions和end_positions都不为空
        if start_positions is not None and end_positions is not None:
            # 如果我们在多GPU上，增加一个维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 有时候start/end位置超出了我们的模型输入，我们忽略这些位置
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 定义交叉熵损失函数，忽略索引为ignored_index的loss
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```