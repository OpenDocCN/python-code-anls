# `.\models\deberta_v2\modeling_deberta_v2.py`

```py
# 包含方法和类的名称空间
from collections.abc import Sequence
from typing import Optional, Tuple, Union

# 导入 PyTorch 和相关模块
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss

# 导入自定义的模块和函数
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

# 设置日志记录器
logger = logging.get_logger(__name__)

# 设置文档中的配置示例和检查点示例
_CONFIG_FOR_DOC = "DebertaV2Config"
_CHECKPOINT_FOR_DOC = "microsoft/deberta-v2-xlarge"
_QA_TARGET_START_INDEX = 2
_QA_TARGET_END_INDEX = 9

# 预训练模型的清单列表
DEBERTA_V2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/deberta-v2-xlarge",
    "microsoft/deberta-v2-xxlarge",
    "microsoft/deberta-v2-xlarge-mnli",
    "microsoft/deberta-v2-xxlarge-mnli",
]

# 定义上下文池模块
# 从 transformers.models.deberta.modeling_deberta.ContextPooler 复制而来
class ContextPooler(nn.Module):
    def __init__(self, config):
        # 调用父类的构造函数
        super().__init__()
        # 定义线性层和稳定的 Dropout 层
        self.dense = nn.Linear(config.pooler_hidden_size, config.pooler_hidden_size)
        self.dropout = StableDropout(config.pooler_dropout)
        self.config = config

    def forward(self, hidden_states):
        # 通过选择第一个令牌的隐藏状态进行汇总模型
        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = ACT2FN[self.config.pooler_hidden_act](pooled_output)
        return pooled_output

    @property
    def output_dim(self):
        return self.config.hidden_size


# 定义 XSoftmax 类
# 从 transformers.models.deberta.modeling_deberta.XSoftmax 复制而来
class XSoftmax(torch.autograd.Function):
    """
    用以节省内存的 Softmax 层
    """

    def __init__(self, dim):
        # 调用父类的构造函数
        super(XSoftmax, self).__init__()
        self.dim = dim

    # 静态方法，创建一个 XSoftmax 层
    @staticmethod
    def forward(ctx, input, mask):
        if mask is not None:
            input = input.masked_fill(~mask, float('-inf'))
        input = input.softmax(dim=-1)
        ctx.save_for_backward(input)
        return input

    # 后向传播
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output - (input * grad_output).sum(dim=-1, keepdim=True) * input
        return grad_input, None

    # 创建 XSoftmax 对象
    @staticmethod
    def apply(input, mask):
        return XSoftmax(input.dim(), mask)(input)

    # 返回模块中的输入维度
    def extra_repr(self):
        return 'dim={dim}'.format(dim=self.dim)


class SinusoidalPositionalEmbedding(nn.Module):
    """
    正弦位置嵌入
    """

    def __init__(
        self,
        num_positions: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        left_pad: bool = False,
    ):
        # 调用父类的构造函数
        super().__init__()

        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.left_pad = left_pad

        # 创建正弦位置向量
        self.weight = nn.Parameter(torch.zeros(num_positions, embedding_dim))

        # 初始化正弦位置向量
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input: torch.Tensor):
        bsz, seq_len = input.size()[:2]

        # 按浮点除以 10000，生成正弦和余弦函数的输入（用于计算正弦位置向量）
        positions = torch.arange(seq_len, dtype=torch.float32, device=input.device).unsqueeze(-1)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2, dtype=torch.float32, device=input.device) * -(math.log(10000.0) / self.embedding_dim))

        # 计算正弦和余弦函数的值
        pos_emb = torch.zeros_like(input, device=input.device)
        pos_emb[..., 0::2] = torch.sin(positions * div_term)
        pos_emb[..., 1::2] = torch.cos(positions * div_term)

        # 返回正弦位置向量
        return pos_emb

    @torch.jit.export
    def compute_position_encodings(self, inputs):
        max_positions = self.weight.size(0) - 1 + self.left_pad
        mask = inputs.ne(self.padding_idx).int()
        return self.weight.index_select(0, inputs.clamp(0, max_positions)), mask


def _init_weights(module):
    """
    初始化所有线性层的权重
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()
    if isinstance(module, LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


# 以下是 DebertaV2Model 的类定义
class DebertaV2Model(PreTrainedModel):

    def __init__(self, config: DebertaV2Config):
        super().__init__(config)
        self.config = config
        self.embeddings = DebertaV2Embeddings(config)
        self.encoder = DebertaV2Encoder(config)
        self.encoder_lf = None
        self.pooler = ContextPooler(config)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # 定义一个返回字典，默认返回 None
        if return_dict is None:
            return_dict = self.config.use_return_dict

        # 初始化输入嵌入和 attention_mask
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 定义 attention_mask，默认为全 1 的矩阵
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=self.device)

        # 定义 token_type_ids，默认为全 0 的矩阵
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.device)

        # 定义 position_ids，默认为从 0 开始的索引
        if position_ids is None:
            if self.config.sinusoidal_pos_embds:
                position_ids = self.embeddings.position_ids.unsqueeze(0).expand(input_shape)
            else:
                position_ids = torch.arange(input_shape[1], dtype=torch.long, device=self.device)
                position_ids = position_ids.unsqueeze(0).expand(input_shape)

        # 定义嵌入
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        # 前向传播
        hidden_states = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 汇总隐藏状态
        pooled_output = self.pooler(hidden_states)

        # 构建返回值
        if not return_dict:
            return (hidden_states, pooled_output)
        else:
            return BaseModelOutput(last_hidden_state=hidden_states, pooler_output=pooled_output)

    def init_weights(self):
        self.apply(_init_weights)


class DebertaV2StableDropout(nn.Module):
    """
    基于重参数化的 dropout 层
    """

    def __init__(self, p, dim=-1):
        super().__init__()
        self.p = p
        self.dim = dim

    def forward(self, input: torch.Tensor):
        if not self.training or self.p == 0:
            return input
        return torch.utils.checkpoint.dropout(input, self.p, self.training, self.dim)


class DebertaV2Dropout(nn.Module):
    """
    稳定训练 dropout 层
    """

    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training and self.p > 0:
            mask = torch.ones((x.size(-1),), dtype=x.dtype, layout=x.layout, device=x.device).bernoulli_(1 - self.p)
            mask = mask / (1 - self.p)
            mask = mask.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            return mask * x
        return x

class DebertaV2Attention(nn.Module):
    """
    DeBERTa-v2 注意力机制
    """

    def __init__(self, config, is_decoder=False):
        super().__init__()
        self.is_decoder = is_decoder
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = DebertaV2Dropout(config.attention_dropout)
        self.layer_norm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.output_attentions = config.output_attentions

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        query_layer = query_layer / math.sqrt(self.attention_head_size)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if attention_mask is not None:
            if attention_mask.dim() == 3:
                # [bsz x qlen x klen]
                attention_mask = attention_mask[:, None, :, :]
            elif attention_mask.dim() == 2:
                # [bsz x seq_len]
                attention_mask = attention_mask[:, None, None, :]
                attention_mask = attention_mask.expand(-1, 1, hidden_states.size(1), -1)
            elif attention_mask.dim() == 1:
                # [bsz]
                attention_mask = attention_mask[:, None, None, None]
                attention_mask = attention_mask.expand(-1, 1, hidden_states.size(1), hidden_states.size(1))
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose(0, 1)

        context_layer = context_layer.reshape(hidden_states.size(0), -1, self.num_attention_heads * self.attention_head_size)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x


class DebertaV2FeedForward(nn.Module):
    """
    DeBERTa-v2 前馈神经网络
    """

    def __init__(self, config):
        super
    Args:
        input (`torch.tensor`): The input tensor that will apply softmax.
        mask (`torch.IntTensor`):
            The mask matrix where 0 indicate that element will be ignored in the softmax calculation.
        dim (int): The dimension that will apply softmax

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
    ```py"""

class XSoftmax:
    @staticmethod
    def forward(self, input, mask, dim):
        # Save dimension for later use
        self.dim = dim
        # Invert the mask to get valid elements to use in softmax calculation
        rmask = ~(mask.to(torch.bool))
        # Set masked elements to minimum value of input tensor
        output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
        # Apply softmax along specified dimension
        output = torch.softmax(output, self.dim)
        # Set masked elements to 0 in the output
        output.masked_fill_(rmask, 0)
        # Save output for backward pass
        self.save_for_backward(output)
        return output

    @staticmethod
    def backward(self, grad_output):
        # Retrieve saved output tensor
        (output,) = self.saved_tensors
        # Calculate gradient with respect to input tensor
        inputGrad = softmax_backward_data(self, grad_output, output, self.dim, output)
        return inputGrad, None, None

    @staticmethod
    def symbolic(g, self, mask, dim):
        import torch.onnx.symbolic_helper as sym_help
        from torch.onnx.symbolic_opset9 import masked_fill, softmax

        # Cast mask to Long and compute reverse mask
        mask_cast_value = g.op("Cast", mask, to_i=sym_help.cast_pytorch_to_onnx["Long"])
        r_mask = g.op(
            "Cast",
            g.op("Sub", g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64)), mask_cast_value),
            to_i=sym_help.cast_pytorch_to_onnx["Bool"],
        )
        # Set masked elements in input tensor to minimum value
        output = masked_fill(
            g, self, r_mask, g.op("Constant", value_t=torch.tensor(torch.finfo(self.type().dtype()).min))
        )
        # Apply softmax and replace masked elements with 0
        output = softmax(g, output, dim)
        return masked_fill(g, output, r_mask, g.op("Constant", value_t=torch.tensor(0, dtype=torch.bool)))
# 从 transformers.models.deberta.modeling_deberta.DropoutContext 复制代码，定义了一个 DropoutContext 类
class DropoutContext(object):
    def __init__(self):
        self.dropout = 0  # 初始化 dropout 为 0
        self.mask = None  # 初始化 mask 为 None
        self.scale = 1  # 初始化 scale 为 1
        self.reuse_mask = True  # 初始化 reuse_mask 为 True


# 从 transformers.models.deberta.modeling_deberta.get_mask 复制代码，定义了一个函数 get_mask
def get_mask(input, local_context):
    if not isinstance(local_context, DropoutContext):  # 检查 local_context 是否为 DropoutContext 类型
        dropout = local_context  # 如果不是 DropoutContext 类型，则使用 local_context 作为 dropout
        mask = None  # 将 mask 设置为 None
    else:
        dropout = local_context.dropout  # 如果是 DropoutContext 类型，使用 local_context 中的 dropout
        dropout *= local_context.scale  # 将 dropout 乘以 scale
        mask = local_context.mask if local_context.reuse_mask else None  # 根据reuse_mask决定是否重用mask

    if dropout > 0 and mask is None:  # 如果 dropout 大于 0 且 mask 是 None
        mask = (1 - torch.empty_like(input).bernoulli_(1 - dropout)).to(torch.bool)  # 创建 mask

    if isinstance(local_context, DropoutContext):  # 如果 local_context 是 DropoutContext 类型
        if local_context.mask is None:  # 如果 local_context 中的 mask 为 None
            local_context.mask = mask  # 更新 local_context 中的 mask

    return mask, dropout  # 返回 mask 和 dropout


# 从 transformers.models.deberta.modeling_deberta.XDropout 复制代码，定义了一个 XDropout 类
class XDropout(torch.autograd.Function):
    """Optimized dropout function to save computation and memory by using mask operation instead of multiplication."""

    @staticmethod
    def forward(ctx, input, local_ctx):
        mask, dropout = get_mask(input, local_ctx)  # 调用 get_mask 函数获取 mask 和 dropout
        ctx.scale = 1.0 / (1 - dropout)  # 计算 scale
        if dropout > 0:  # 如果 dropout 大于 0
            ctx.save_for_backward(mask)  # 保存 mask
            return input.masked_fill(mask, 0) * ctx.scale  # 返回处理后的输入
        else:
            return input  # 如果 dropout 为 0，则返回原始输入

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.scale > 1:  # 如果 scale 大于 1
            (mask,) = ctx.saved_tensors  # 获取保存的 mask
            return grad_output.masked_fill(mask, 0) * ctx.scale, None  # 返回梯度处理结果和 None
        else:
            return grad_output, None  # 如果 scale 不大于 1，则返回梯度和 None

    @staticmethod
    def symbolic(g: torch._C.Graph, input: torch._C.Value, local_ctx: Union[float, DropoutContext]) -> torch._C.Value:
        from torch.onnx import symbolic_opset12

        dropout_p = local_ctx  # 设置dropout_p为local_ctx
        if isinstance(local_ctx, DropoutContext):  # 如果local_ctx是DropoutContext类型
            dropout_p = local_ctx.dropout  # 使用local_ctx中的dropout
        train = True  # 训练模式为True
        return symbolic_opset12.dropout(g, input, dropout_p, train)  # 返回symbolic_opset12.dropout的结果


# 从 transformers.models.deberta.modeling_deberta.StableDropout 复制代码，定义了一个 StableDropout 类
class StableDropout(nn.Module):
    """
    Optimized dropout module for stabilizing the training

    Args:
        drop_prob (float): the dropout probabilities
    """

    def __init__(self, drop_prob):
        super().__init__()  # 调用父类构造函数
        self.drop_prob = drop_prob  # 初始化 drop_prob
        self.count = 0  # 初始化 count 为 0
        self.context_stack = None  # 初始化 context_stack 为 None
    # 定义一个方法，用于对输入的张量进行前向传播操作
    def forward(self, x):
        """
        Call the module
    
        Args:
            x (`torch.tensor`): The input tensor to apply dropout
        """
        # 如果处于训练状态并且丢弃概率大于0，则调用XDropout.apply方法对输入张量进行处理
        if self.training and self.drop_prob > 0:
            return XDropout.apply(x, self.get_context())
        # 如果不处于训练状态或者丢弃概率为0，则直接返回输入张量
        return x
    
    # 清除上下文信息的方法，将计数器和上下文栈重置为初始状态
    def clear_context(self):
        self.count = 0
        self.context_stack = None
    
    # 初始化上下文信息的方法，设置重用掩码和比例，并创建上下文栈
    def init_context(self, reuse_mask=True, scale=1):
        if self.context_stack is None:
            self.context_stack = []
        self.count = 0
        for c in self.context_stack:
            c.reuse_mask = reuse_mask
            c.scale = scale
    
    # 获取上下文信息的方法，如果上下文栈不为空，则获取上下文信息并更新计数器
    def get_context(self):
        if self.context_stack is not None:
            if self.count >= len(self.context_stack):
                self.context_stack.append(DropoutContext())
            ctx = self.context_stack[self.count]
            ctx.dropout = self.drop_prob
            self.count += 1
            return ctx
        # 如果上下文栈为空，则返回丢弃概率值
        else:
            return self.drop_prob
# 从 transformers.models.deberta.modeling_deberta.DebertaSelfOutput 复制并修改为 DebertaLayerNorm->LayerNorm
class DebertaV2SelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个全连接层，将输入维度转换为 hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 初始化一个 LayerNorm 层，对 hidden_size 维度的数据进行归一化
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        # 初始化一个固定概率下降的 dropout 层
        self.dropout = StableDropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # 将隐藏状态输入全连接层得到新的隐藏状态
        hidden_states = self.dense(hidden_states)
        # 对隐藏状态进行 dropout
        hidden_states = self.dropout(hidden_states)
        # 对隐藏状态进行 LayerNorm 处理，并与输入向量相加
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# 从 transformers.models.deberta.modeling_deberta.DebertaAttention 复制并修改为 Deberta->DebertaV2
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
        # 调用 self 对象的 forward 方法进行自注意力操作
        self_output = self.self(
            hidden_states,
            attention_mask,
            output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        if output_attentions:
            # 如果输出注意力矩阵，则返回注意力矩阵和自注意力输出
            self_output, att_matrix = self_output
        if query_states is None:
            query_states = hidden_states
        # 对自注意力结果和查询状态进行输出操作
        attention_output = self.output(self_output, query_states)

        if output_attentions:
            # 如果输出注意力矩阵，则返回注意力输出和注意力矩阵
            return (attention_output, att_matrix)
        else:
            # 否则，只返回注意力输出
            return attention_output


# 从 transformers.models.bert.modeling_bert.BertIntermediate 复制并修改为 Bert->DebertaV2
class DebertaV2Intermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个全连接层，将 hidden_size 维度转换为 intermediate_size 维度
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 判断 hidden_act 是否为字符串，如果是则获取对应的激活函数，否则直接获取激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态输入全连接层得到新的隐藏状态
        hidden_states = self.dense(hidden_states)
        # 对隐藏状态使用激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# 从 transformers.models.deberta.modeling_deberta.DebertaOutput 复制并修改为 DebertaLayerNorm->LayerNorm
class DebertaV2Output(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个全连接层，将 intermediate_size 维度转换为 hidden_size 维度
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 初始化一个 LayerNorm 层，对 hidden_size 维度的数据进行归一化
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        # 初始化一个固定概率下降的 dropout 层
        self.dropout = StableDropout(config.hidden_dropout_prob)
        # 保存配置信息
        self.config = config
    # 前向传播函数，用于神经网络的前向计算
    def forward(self, hidden_states, input_tensor):
        # 使用全连接层对隐藏状态进行变换
        hidden_states = self.dense(hidden_states)
        # 对变换后的隐藏状态进行 Dropout 处理，以减少过拟合
        hidden_states = self.dropout(hidden_states)
        # 将经过全连接层和 Dropout 处理后的隐藏状态与输入张量相加，并进行 LayerNorm 处理
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的隐藏状态
        return hidden_states
# 从transformers.models.deberta.modeling_deberta.DebertaLayer复制代码并将Deberta->DebertaV2
class DebertaV2Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = DebertaV2Attention(config)  # 初始化DebertaV2Attention对象
        self.intermediate = DebertaV2Intermediate(config)  # 初始化DebertaV2Intermediate对象
        self.output = DebertaV2Output(config)  # 初始化DebertaV2Output对象

    def forward(
        self,
        hidden_states,
        attention_mask,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
        output_attentions=False,
    ):
        attention_output = self.attention(  # 调用attention方法
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        if output_attentions:
            attention_output, att_matrix = attention_output  # 如果output_attentions为True，更新attention_output和att_matrix
        intermediate_output = self.intermediate(attention_output)  # 调用intermediate方法
        layer_output = self.output(intermediate_output, attention_output)  # 调用output方法
        if output_attentions:
            return (layer_output, att_matrix)  # 如果output_attentions为True，返回layer_output和att_matrix
        else:
            return layer_output  # 如果output_attentions为False，只返回layer_output


class ConvLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        kernel_size = getattr(config, "conv_kernel_size", 3)  # 获取config的conv_kernel_size属性，默认为3
        groups = getattr(config, "conv_groups", 1)  # 获取config的conv_groups属性，默认为1
        self.conv_act = getattr(config, "conv_act", "tanh")  # 获取config的conv_act属性，默认为"tanh"
        self.conv = nn.Conv1d(  # 创建一个一维卷积层
            config.hidden_size, config.hidden_size, kernel_size, padding=(kernel_size - 1) // 2, groups=groups
        )
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)  # 初始化LayerNorm对象
        self.dropout = StableDropout(config.hidden_dropout_prob)  # 初始化StableDropout对象
        self.config = config  # 设置config属性

    def forward(self, hidden_states, residual_states, input_mask):
        out = self.conv(hidden_states.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()  # 一维卷积操作
        rmask = (1 - input_mask).bool()  # 计算input_mask的逻辑反
        out.masked_fill_(rmask.unsqueeze(-1).expand(out.size()), 0)  # 根据rmask设置out中的部分值为0
        out = ACT2FN[self.conv_act](self.dropout(out))  # 根据conv_act进行激活和dropout操作

        layer_norm_input = residual_states + out  # 计算layer_norm_input
        output = self.LayerNorm(layer_norm_input).to(layer_norm_input)  # 进行LayerNorm操作

        if input_mask is None:  # 如果input_mask为None
            output_states = output  # 则直接输出output
        else:
            if input_mask.dim() != layer_norm_input.dim():  # 如果input_mask的维度和layer_norm_input的维度不一致
                if input_mask.dim() == 4:  # 如果input_mask的维度为4
                    input_mask = input_mask.squeeze(1).squeeze(1)  # 压缩input_mask的两个维度
                input_mask = input_mask.unsqueeze(2)  # 在第二个维度上扩展input_mask
            input_mask = input_mask.to(output.dtype)  # 转换input_mask的数据类型为output的数据类型
            output_states = output * input_mask  # 将output和input_mask进行乘法操作

        return output_states  # 返回output_states


class DebertaV2Encoder(nn.Module):
    """Modified BertEncoder with relative position bias support"""  # 有相对位置偏差支持的修改后的BertEncoder
    # 初始化函数，接受配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()

        # 创建一个包含多个 DebertaV2Layer 层的 ModuleList
        self.layer = nn.ModuleList([DebertaV2Layer(config) for _ in range(config.num_hidden_layers)])
        # 检查是否开启相对位置编码
        self.relative_attention = getattr(config, "relative_attention", False)

        # 如果开启相对位置编码
        if self.relative_attention:
            # 设置最大相对位置
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings

            # 设置位置桶的数量
            self.position_buckets = getattr(config, "position_buckets", -1)
            pos_ebd_size = self.max_relative_positions * 2

            # 如果启用位置桶，重新设置位置嵌入的大小
            if self.position_buckets > 0:
                pos_ebd_size = self.position_buckets * 2

            # 创建相对位置嵌入的 Embedding 层
            self.rel_embeddings = nn.Embedding(pos_ebd_size, config.hidden_size)

        # 读取相对位置嵌入层的规范化方式
        self.norm_rel_ebd = [x.strip() for x in getattr(config, "norm_rel_ebd", "none").lower().split("|")]

        # 如果采用 LayerNorm 规范化
        if "layer_norm" in self.norm_rel_ebd:
            # 初始化 LayerNorm 层
            self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=True)

        # 如果设置卷积层
        self.conv = ConvLayer(config) if getattr(config, "conv_kernel_size", 0) > 0 else None
        # 是否开启梯度检查点
        self.gradient_checkpointing = False

    # 获取相对位置嵌入
    def get_rel_embedding(self):
        # 如果开启相对位置编码，获取相对位置嵌入权重
        rel_embeddings = self.rel_embeddings.weight if self.relative_attention else None
        # 如果相对位置嵌入存在且采用 LayerNorm 规范化
        if rel_embeddings is not None and ("layer_norm" in self.norm_rel_ebd):
            # 对相对位置嵌入进行规范化
            rel_embeddings = self.LayerNorm(rel_embeddings)
        return rel_embeddings

    # 获取注意力遮盖
    def get_attention_mask(self, attention_mask):
        # 如果注意力遮盖维度不大于2
        if attention_mask.dim() <= 2:
            # 增加维度使其适应计算
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # 生成注意力遮盖矩阵
            attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
        # 如果注意力遮盖维度为3
        elif attention_mask.dim() == 3:
            # 增加维度以匹配计算需求
            attention_mask = attention_mask.unsqueeze(1)

        return attention_mask

    # 获取相对位置
    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        # 如果开启相对位置编码且相对位置矩阵不存在
        if self.relative_attention and relative_pos is None:
            # 获取查询序列的长度或使用隐藏状态的长度
            q = query_states.size(-2) if query_states is not None else hidden_states.size(-2)
            # 构建相对位置
            relative_pos = build_relative_position(
                q,
                hidden_states.size(-2),
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
                device=hidden_states.device,
            )
        return relative_pos

    # 前向传播函数
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
            # 检查是否需要使用注意力遮罩
            if attention_mask.dim() <= 2:
                # 如果注意力遮罩的维度小于等于2，则使用原始的attention_mask
                input_mask = attention_mask
            else:
                # 如果注意力遮罩的维度大于2，则对最后两维求和，生成input_mask
                input_mask = attention_mask.sum(-2) > 0
            # 获取注意力遮罩
            attention_mask = self.get_attention_mask(attention_mask)
            # 获取相对位置编码
            relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)

            # 如果需要输出隐藏状态，则初始化空的all_hidden_states元组；否则置为None
            all_hidden_states = () if output_hidden_states else None
            # 如果需要输出注意力，则初始化空的all_attentions元组；否则置为None
            all_attentions = () if output_attentions else None

            # 如果hidden_states是可迭代的序列，则取其第一个元素作为next_kv；否则直接将hidden_states赋值给next_kv
            if isinstance(hidden_states, Sequence):
                next_kv = hidden_states[0]
            else:
                next_kv = hidden_states
            # 获取相对位置编码的嵌入
            rel_embeddings = self.get_rel_embedding()
            # 将output_states初始化为next_kv
            output_states = next_kv
            # 遍历每一个层
            for i, layer_module in enumerate(self.layer):
                # 如果需要输出隐藏状态，则添加当前output_states到all_hidden_states
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (output_states,)

                # 如果支持梯度检查点并且处于训练状态，则调用_gradient_checkpointing_func函数
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
                    # 否则直接调用layer_module进行前向传播
                    output_states = layer_module(
                        next_kv,
                        attention_mask,
                        query_states=query_states,
                        relative_pos=relative_pos,
                        rel_embeddings=rel_embeddings,
                        output_attentions=output_attentions,
                    )

                # 如果需要输出注意力，则从output_states中提取注意力
                if output_attentions:
                    output_states, att_m = output_states

                # 如果是第一个层并且存在卷积操作，则对output_states进行卷积
                if i == 0 and self.conv is not None:
                    output_states = self.conv(hidden_states, output_states, input_mask)

                # 如果query_states不为None，则更新query_states和next_kv
                if query_states is not None:
                    query_states = output_states
                    if isinstance(hidden_states, Sequence):
                        next_kv = hidden_states[i + 1] if i + 1 < len(self.layer) else None
                else:
                    next_kv = output_states

                # 如果需要输出注意力，则添加当前att_m到all_attentions
                if output_attentions:
                    all_attentions = all_attentions + (att_m,)

            # 如果需要输出隐藏状态，则添加当前output_states到all_hidden_states
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (output_states,)

            # 如果不需要返回字典，则返回output_states, all_hidden_states, all_attentions中非None的元素
            if not return_dict:
                return tuple(v for v in [output_states, all_hidden_states, all_attentions] if v is not None)
            # 否则返回BaseModelOutput对象
            return BaseModelOutput(
                last_hidden_state=output_states, hidden_states=all_hidden_states, attentions=all_attentions
            )
# 根据相对位置、桶大小和最大位置创建位置桶
def make_log_bucket_position(relative_pos, bucket_size, max_position):
    # 计算相对位置的符号
    sign = torch.sign(relative_pos)
    # 计算绝对位置
    mid = bucket_size // 2
    abs_pos = torch.where(
        (relative_pos < mid) & (relative_pos > -mid),
        torch.tensor(mid - 1).type_as(relative_pos),
        torch.abs(relative_pos),
    )
    # 计算对数位置
    log_pos = (
        torch.ceil(torch.log(abs_pos / mid) / torch.log(torch.tensor((max_position - 1) / mid)) * (mid - 1)) + mid
    )
    # 根据条件返回位置桶
    bucket_pos = torch.where(abs_pos <= mid, relative_pos.type_as(log_pos), log_pos * sign)
    return bucket_pos


def build_relative_position(query_size, key_size, bucket_size=-1, max_position=-1, device=None):
    """
    根据查询和键构建相对位置

    假设查询的绝对位置 \\(P_q\\) 在 (0, query_size) 范围内，并且键的绝对位置 \\(P_k\\) 在 (0, key_size) 范围内，
    则从查询到键的相对位置为 \\(R_{q \\rightarrow k} = P_q - P_k\\)

    Args:
        query_size (int): 查询的长度
        key_size (int): 键的长度
        bucket_size (int): 位置桶的大小
        max_position (int): 可允许的最大绝对位置
        device (`torch.device`): 创建张量所在的设备

    Return:
        `torch.LongTensor`: 形状为 [1, query_size, key_size] 的张量
    """

    # 创建查询和键的索引
    q_ids = torch.arange(0, query_size, device=device)
    k_ids = torch.arange(0, key_size, device=device)
    # 计算相对位置
    rel_pos_ids = q_ids[:, None] - k_ids[None, :]
    if bucket_size > 0 and max_position > 0:
        # 根据位置桶大小和最大位置进行处理
        rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
    rel_pos_ids = rel_pos_ids.to(torch.long)
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = rel_pos_ids.unsqueeze(0)
    return rel_pos_ids


@torch.jit.script
# 从 transformers.models.deberta.modeling_deberta.c2p_dynamic_expand 复制而来
def c2p_dynamic_expand(c2p_pos, query_layer, relative_pos):
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), query_layer.size(2), relative_pos.size(-1)])


@torch.jit.script
# 从 transformers.models.deberta.modeling_deberta.p2c_dynamic_expand 复制而来
def p2c_dynamic_expand(c2p_pos, query_layer, key_layer):
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), key_layer.size(-2), key_layer.size(-2)])


@torch.jit.script
# 从 transformers.models.deberta.modeling_deberta.pos_dynamic_expand 复制而来
def pos_dynamic_expand(pos_index, p2c_att, key_layer):
    return pos_index.expand(p2c_att.size()[:2] + (pos_index.size(-2), key_layer.size(-2))


class DisentangledSelfAttention(nn.Module):
    """
    分解自注意力模块

    Parameters:
        config (`DebertaV2Config`):
            使用指定配置构建一个新模型。其架构类似于 *BertConfig*，更多细节请参考 [`DebertaV2Config`]

    """
    # 初始化函数，用于初始化注意力机制的参数
    def __init__(self, config):
        # 调用父类初始化函数
        super().__init__()
        # 检查隐藏层大小是否可以整除注意力头的数量，若不能则引发数值错误
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        # 设置注意力头的数量
        self.num_attention_heads = config.num_attention_heads
        # 计算每个注意力头的大小
        _attention_head_size = config.hidden_size // config.num_attention_heads
        # 获取注意力头的大小，若未指定则使用计算结果
        self.attention_head_size = getattr(config, "attention_head_size", _attention_head_size)
        # 计算所有注意力头的总大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # 用线性变换层构建查询、键、值的投影矩阵
        self.query_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.key_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.value_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)

        # 检查是否共享注意力机制的键
        self.share_att_key = getattr(config, "share_att_key", False)
        # 获取位置注意力类型，默认为空列表
        self.pos_att_type = config.pos_att_type if config.pos_att_type is not None else []
        # 检查是否使用相对位置编码
        self.relative_attention = getattr(config, "relative_attention", False)

        # 如果使用相对位置编码
        if self.relative_attention:
            # 获取位置桶的数量，默认为-1
            self.position_buckets = getattr(config, "position_buckets", -1)
            # 获取最大相对位置，默认为-1
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            # 若最大相对位置小于1，则设置为最大位置编码
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            # 计算位置编码的维度
            self.pos_ebd_size = self.max_relative_positions
            if self.position_buckets > 0:
                self.pos_ebd_size = self.position_buckets

            # 使用稳定的Dropout层进行位置编码的丢弃
            self.pos_dropout = StableDropout(config.hidden_dropout_prob)

            # 若不共享注意力机制的键，则分别构建位置编码的投影矩阵
            if not self.share_att_key:
                if "c2p" in self.pos_att_type:
                    self.pos_key_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
                if "p2c" in self.pos_att_type:
                    self.pos_query_proj = nn.Linear(config.hidden_size, self.all_head_size)

        # 使用稳定的Dropout层进行注意力概率的丢弃
        self.dropout = StableDropout(config.attention_probs_dropout_prob)

    # 对输入张量进行维度变换，以便计算注意力分数
    def transpose_for_scores(self, x, attention_heads):
        # 计算新的张量形状
        new_x_shape = x.size()[:-1] + (attention_heads, -1)
        # 改变张量形状
        x = x.view(new_x_shape)
        # 将张量维度进行转置以便计算注意力分数
        return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))

    # 前向传播函数，用于计算注意力机制的输出
    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
# 从transformers.models.deberta.modeling_deberta.DebertaEmbeddings拷贝代码，并使用DebertaLayerNorm->LayerNorm进行替换
class DebertaV2Embeddings(nn.Module):
    """构建来自词嵌入、位置和标记类型嵌入的嵌入。"""

    def __init__(self, config):
        super().__init__()
        # 获取配置中的pad_token_id，如果不存在则默认为0
        pad_token_id = getattr(config, "pad_token_id", 0)
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)
        # 使用词汇大小和嵌入大小创建词嵌入
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embedding_size, padding_idx=pad_token_id)

        # 获取配置中的position_biased_input，如果不存在则默认为True
        self.position_biased_input = getattr(config, "position_biased_input", True)
        if not self.position_biased_input:
            self.position_embeddings = None
        else:
            # 使用最大位置嵌入和嵌入大小创建位置嵌入
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.embedding_size)

        # 如果配置中的type_vocab_size大于0，则使用类型词汇大小和嵌入大小创建标记类型嵌入
        if config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, self.embedding_size)

        # 如果嵌入大小不等于隐藏层大小，则使用嵌入大小和隐藏层大小创建线性投影
        if self.embedding_size != config.hidden_size:
            self.embed_proj = nn.Linear(self.embedding_size, config.hidden_size, bias=False)
        # 使用隐藏层大小和层归一化epsilon创建层归一化
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        # 使用稳定的dropout概率创建dropout
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.config = config

        # position_ids(1, len position emb)在内存中是连续的，并在序列化时导出
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
    # 定义一个前向传播函数，接收输入的token id、token类型id、位置id、mask和输入的嵌入向量
    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, mask=None, inputs_embeds=None):
        # 如果传入了token id，则获取其形状
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            # 否则获取输入嵌入向量的形状
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # 如果位置id为空，则使用预定义的位置id，截取到与序列长度相同的部分
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # 如果token类型id为空，则创建与输入形状相同的零矩阵
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果输入嵌入向量为空，则使用word_embeddings对token id进行嵌入
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # 如果存在位置嵌入，则使用position_embeddings进行嵌入，否则创建一个与输入嵌入向量相同形状的零矩阵
        if self.position_embeddings is not None:
            position_embeddings = self.position_embeddings(position_ids.long())
        else:
            position_embeddings = torch.zeros_like(inputs_embeds)

          # 合并token嵌入和位置嵌入，如果position_biased_input为真则加上位置嵌入
        embeddings = inputs_embeds
        if self.position_biased_input:
            embeddings += position_embeddings
        # 如果类型词汇表大小大于0，则使用token_type_embeddings对token类型id进行嵌入
        if self.config.type_vocab_size > 0:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings

        # 如果嵌入大小不等于隐藏层大小，则对嵌入进行线性投影
        if self.embedding_size != self.config.hidden_size:
            embeddings = self.embed_proj(embeddings)

        # 对嵌入向量进行LayerNorm
        embeddings = self.LayerNorm(embeddings)

        # 如果存在mask，则对嵌入向量应用mask
        if mask is not None:
            if mask.dim() != embeddings.dim():
                if mask.dim() == 4:
                    mask = mask.squeeze(1).squeeze(1)
                mask = mask.unsqueeze(2)
            mask = mask.to(embeddings.dtype)

            embeddings = embeddings * mask

        # 对嵌入向量应用dropout
        embeddings = self.dropout(embeddings)

        # 返回嵌入向量
        return embeddings
# 这是一个继承自 PreTrainedModel 的类，用于处理权重初始化和提供下载与加载预训练模型的简单接口
class DebertaV2PreTrainedModel(PreTrainedModel):
    """
    这是一个用于处理权重初始化和提供预训练模型下载/加载接口的抽象类
    """

    # 设置配置类为 DebertaV2Config
    config_class = DebertaV2Config
    # 设置基础模型的前缀名称
    base_model_prefix = "deberta"
    # 指定在加载模型时忽略的键
    _keys_to_ignore_on_load_unexpected = ["position_embeddings"]
    # 指示是否支持梯度检查点
    supports_gradient_checkpointing = True

    # 这个方法用于初始化权重
    def _init_weights(self, module):
        """初始化权重"""
        # 如果模块是线性层
        if isinstance(module, nn.Linear):
            # 使用正态分布来初始化权重，均值为0.0，标准差为配置文件中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果该线性层有偏置，则将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是嵌入层
        elif isinstance(module, nn.Embedding):
            # 用正态分布初始化嵌入权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果嵌入层有填充索引，将填充的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


# 这是关于 DeBERTa 模型的文档字符串，描述了该模型的背景和用途
DEBERTA_START_DOCSTRING = r"""
    DeBERTa 模型是在 [DeBERTa: Decoding-enhanced BERT with Disentangled
    Attention](https://arxiv.org/abs/2006.03654) 中提出的，由 Pengcheng He、Xiaodong Liu、Jianfeng Gao、Weizhu Chen 等人开发。
    它基于 BERT/RoBERTa 进行了两个改进：解耦注意力和增强的遮罩解码器。通过这两个改进，
    它在绝大多数任务上优于 BERT/RoBERTa，并且进行了 80GB 的预训练数据。

    这个模型也是 PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) 的子类。
    可以像普通 PyTorch 模块一样使用，并参考 PyTorch 文档来了解其通用用法和行为。

    参数：
        config ([`DebertaV2Config`]): 模型配置类，包含模型的所有参数。
        使用配置文件初始化不会加载模型关联的权重，只会加载配置。要加载模型权重，请查看 [`~PreTrainedModel.from_pretrained`] 方法。
"""

# 文档字符串描述了 DeBERTa 的输入方式和格式
DEBERTA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列中词汇表中的标记的索引。
            # 可以使用 [`AutoTokenizer`] 获取索引。有关详细信息，请参见 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`]。

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 避免在填充标记索引上执行注意力的掩码。选择在 `[0, 1]` 中的掩码值：

            - 1 表示 **未掩码** 的标记，
            - 0 表示 **已掩码** 的标记。

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 段标记索引，用于指示输入的第一部分和第二部分。索引在 `[0, 1]` 中选择：

            - 0 对应于 *句子 A* 的标记，
            - 1 对应于 *句子 B* 的标记。

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 输入序列中每个标记的位置索引，用于位置嵌入。在范围 `[0, config.max_position_embeddings - 1]` 中选择。

            [What are position IDs?](../glossary#position-ids)
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选地，您可以选择直接传递嵌入表示而不是传递 `input_ids`。如果您希望更多地控制如何将 *input_ids* 索引转换为关联向量，
            # 而不是使用模型的内部嵌入查找矩阵，则此选项很有用。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关详细信息，请参见返回的张量下的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关详细信息，请参见返回的张量下的 `hidden_states`。
        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通元组。
```  
"""
定义 DebertaV2Model 类，继承自 DebertaV2PreTrainedModel 类
@add_start_docstrings 注释装饰器，用于添加模型文档字符串
class DebertaV2Model(DebertaV2PreTrainedModel):
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 初始化模型的嵌入层
        self.embeddings = DebertaV2Embeddings(config)
        # 初始化模型的编码器层
        self.encoder = DebertaV2Encoder(config)
        # 设置 z_steps 为 0
        self.z_steps = 0
        # 保存模型配置
        self.config = config
        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入嵌入层
    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    # 剪枝模型的头部
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError("The prune function is not implemented in DeBERTa model.")

    # 模型的前向传播函数
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
    # 函数定义，接受输入参数并返回模型输出
    ) -> Union[Tuple, BaseModelOutput]:
        # 检查是否需要输出注意力权重
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 检查是否需要输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 检查是否返回字典形式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果同时指定了 input_ids 和 inputs_embeds，则引发错误
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        # 如果指定了 input_ids
        elif input_ids is not None:
            # 如果存在 padding 并且没有给出 attention_mask，则发出警告
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            # 获取 input_ids 的形状
            input_shape = input_ids.size()
        # 如果指定了 inputs_embeds
        elif inputs_embeds is not None:
            # 获取 inputs_embeds 的形状
            input_shape = inputs_embeds.size()[:-1]
        else:
            # 如果既没有指定 input_ids 也没有指定 inputs_embeds，则引发错误
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 获取输入所在设备
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # 如果没有给出 attention_mask，则创建全 1 的注意力掩码
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        # 如果没有给出 token_type_ids，则创建全 0 的 token_type_ids
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # 获取嵌入输出
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )

        # 获取编码器输出
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )
        # 获取编码器编码的层数
        encoded_layers = encoder_outputs[1]

        # 如果 z_steps 大于 1
        if self.z_steps > 1:
            # 获取倒数第二层的隐藏状态
            hidden_states = encoded_layers[-2]
            # 复制 self.z_steps 次最后一层编码器
            layers = [self.encoder.layer[-1] for _ in range(self.z_steps)]
            # 获取查询状态
            query_states = encoded_layers[-1]
            # 获取相对位置嵌入
            rel_embeddings = self.encoder.get_rel_embedding()
            # 获取注意力掩码
            attention_mask = self.encoder.get_attention_mask(attention_mask)
            # 获取相对位置编码
            rel_pos = self.encoder.get_rel_pos(embedding_output)
            # 对于除第一次外的每一次迭代
            for layer in layers[1:]:
                # 使用当前层进行计算
                query_states = layer(
                    hidden_states,
                    attention_mask,
                    output_attentions=False,
                    query_states=query_states,
                    relative_pos=rel_pos,
                    rel_embeddings=rel_embeddings,
                )
                # 将结果添加到编码层中
                encoded_layers.append(query_states)

        # 获取序列输出
        sequence_output = encoded_layers[-1]

        # 如果不需要返回字典形式的输出
        if not return_dict:
            # 返回元组形式的输出
            return (sequence_output,) + encoder_outputs[(1 if output_hidden_states else 2) :]

        # 返回字典形式的 BaseModelOutput
        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions,
        )
# 为DeBERTa模型添加一个基于语言建模的头部
@add_start_docstrings("""DeBERTa Model with a `language modeling` head on top.""", DEBERTA_START_DOCSTRING)
class DebertaV2ForMaskedLM(DebertaV2PreTrainedModel):
    # 定义共享权重的键值对
    _tied_weights_keys = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]

    def __init__(self, config):
        # 调用父类初始化方法
        super().__init__(config)

        # 实例化DeBERTaV2模型
        self.deberta = DebertaV2Model(config)
        # 实例化仅包含MLM头部的DeBERTaV2模型
        self.cls = DebertaV2OnlyMLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出embedding
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出embedding
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 为模型的前向传播添加文档字符串
    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="[MASK]",
    )
    # 从transformers.models.deberta.modeling_deberta.DebertaForMaskedLM.forward复制过来，并更改Deberta为DebertaV2
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
        # 如果 return_dict 参数不为 None，则使用传入的值，否则使用配置中的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 DeBERTa 模型进行前向传播
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
        # 使用分类器层得到预测的词汇分数
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        # 如果存在标签，则计算 masked language modeling loss
        if labels is not None:
            # 定义交叉熵损失函数，-100 索引表示填充标记
            loss_fct = CrossEntropyLoss()
            # 计算 masked language modeling loss
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果不需要返回字典，则将输出组合成元组返回
        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 如果需要返回字典，则构造 MaskedLMOutput 对象返回
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 从transformers.models.deberta.modeling_deberta.DebertaPredictionHeadTransform复制，将Deberta->DebertaV2
class DebertaV2PredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)

        # 使用线性层进行转换
        self.dense = nn.Linear(config.hidden_size, self.embedding_size)
        if isinstance(config.hidden_act, str):
            # 如果config.hidden_act是字符串，使用ACT2FN中定义的函数
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            # 否则使用config.hidden_act
            self.transform_act_fn = config.hidden_act
        # LayerNorm层
        self.LayerNorm = nn.LayerNorm(self.embedding_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        # 通过线性层进行转换
        hidden_states = self.dense(hidden_states)
        # 使用激活函数进行转换
        hidden_states = self.transform_act_fn(hidden_states)
        # 使用LayerNorm进行转换
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

# 从transformers.models.deberta.modeling_deberta.DebertaLMPredictionHead复制，将Deberta->DebertaV2
class DebertaV2LMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用DebertaV2PredictionHeadTransform对数据进行转换
        self.transform = DebertaV2PredictionHeadTransform(config)

        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)
        # 输出权重与输入嵌入相同，但是每个令牌都有一个仅用于输出的偏置
        self.decoder = nn.Linear(self.embedding_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # 需要一个链接来正确调整`resize_token_embeddings`的偏置大小
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        # 使用DebertaV2PredictionHeadTransform对数据进行转换
        hidden_states = self.transform(hidden_states)
        # 使用解码器进行转换
        hidden_states = self.decoder(hidden_states)
        return hidden_states

# 从transformers.models.bert.BertOnlyMLMHead复制，将bert->deberta
class DebertaV2OnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 包含DebertaV2LMPredictionHead
        self.predictions = DebertaV2LMPredictionHead(config)

    def forward(self, sequence_output):
        # 使用DebertaV2LMPredictionHead进行预测
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

@add_start_docstrings(
    """
    在顶部有一个序列分类/回归头���池化输出的线性层），例如GLUE任务
    """,
    DEBERTA_START_DOCSTRING,
)
# 基于DebertaV2PreTrainedModel的DebertaV2ForSequenceClassification类
class DebertaV2ForSequenceClassification(DebertaV2PreTrainedModel):
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化函数，传入配置对象
        super().__init__(config)

        # 从配置对象中获取 num_labels 属性，如果不存在则默认为 2
        num_labels = getattr(config, "num_labels", 2)
        # 将获取到的 num_labels 赋值给 self.num_labels
        self.num_labels = num_labels

        # 创建一个 DebertaV2Model 模型对象，使用传入的配置对象作为参数
        self.deberta = DebertaV2Model(config)
        # 创建一个 ContextPooler 模型对象，使用传入的配置对象作为参数
        self.pooler = ContextPooler(config)
        # 获取 ContextPooler 输出维度，赋值给 output_dim 变量
        output_dim = self.pooler.output_dim

        # 创建一个线性层，输入维度为 output_dim，输出维度为 num_labels
        self.classifier = nn.Linear(output_dim, num_labels)
        # 获取配置对象中的 cls_dropout 属性，如果不存在则使用配置对象中的 hidden_dropout_prob 属性
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        # 创建一个稳定的 Dropout 层，使用 drop_out 作为 dropout 概率
        self.dropout = StableDropout(drop_out)

        # 初始化模型权重并进行最终处理
        self.post_init()

    # 获取输入嵌入层对象
    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    # 设置输入嵌入层对象
    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

    # 前向传播函数
    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 从 transformers.models.deberta.modeling_deberta.DebertaForSequenceClassification.forward 复制而来，将 Deberta 改为 DebertaV2
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
# 添加模型文档的起始字符串，描述了该模型是在DeBERTa模型的基础上添加了一个标记分类头部，用于命名实体识别（NER）等任务
# 使用了DeBERTaV2的起始文档字符串
@add_start_docstrings(
    """
    DeBERTa Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    DEBERTA_START_DOCSTRING,
)
# 从transformers.models.deberta.modeling_deberta.DebertaForTokenClassification复制过来并将Deberta->DebertaV2
class DebertaV2ForTokenClassification(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.deberta = DebertaV2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 为模型的forward方法添加模型输入的起始注释字符串，描述了输入参数的形状和类型
    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 为模型的forward方法添加代码示例注释字符串，描述了checkpoint、output_type和config_class的信息
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
        # 设置返回字典为与配置中的使用返回字典一致
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用DebertaV2模型进行前向传播
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

        # 对输出进行dropout处理
        sequence_output = self.dropout(sequence_output)
        # 经过线性变换得到logits
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # 计算交叉熵损失
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果return_dict为False，则返回一个元组
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果return_dict为True，则返回TokenClassifierOutput
        return TokenClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

# 添加模型文档的起始字符串
@add_start_docstrings(
    """
    # DeBERTa 模型，顶部带有一个用于抽取式问答任务（如 SQuAD）的跨度分类头（在隐藏状态输出的基础上添加线性层来计算“跨度起始对数”和“跨度结束对数”）
    """,
    # DEBERTA_START_DOCSTRING 是一个文档字符串的起始标记
# 定义 DebertaV2ForQuestionAnswering 类，继承自 DebertaV2PreTrainedModel 类
class DebertaV2ForQuestionAnswering(DebertaV2PreTrainedModel):
    # 初始化函数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)
        # 设置标签数量
        self.num_labels = config.num_labels

        # 创建 DebertaV2Model 对象
        self.deberta = DebertaV2Model(config)
        # 创建线性层，用于问题回答
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数
    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        qa_target_start_index=_QA_TARGET_START_INDEX,
        qa_target_end_index=_QA_TARGET_END_INDEX,
    )
    # 从 transformers.models.deberta.modeling_deberta.DebertaForQuestionAnswering.forward 复制而来，将 Deberta 替换为 DebertaV2
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
    # 定义 forward 方法，用于从输入数据中预测问答任务的开始和结束位置
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
        # 判断是否使用返回字典的形式
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # 将输入数据传入 DeBERTa 模型，获取序列输出
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
    
        # 将序列输出传入 QA 输出层，获取开始和结束位置的 logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
    
        # 计算损失函数
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 如果 start_positions 和 end_positions 的维度大于 1，则去掉最后一个维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 将 start_positions 和 end_positions 限制在序列长度之内
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
    
            # 使用交叉熵损失函数计算总损失
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
    
        # 根据 return_dict 的值返回不同格式的输出
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
# 基于 DeBERTa 模型构建多选题分类模型，顶部附加了一个线性层（在池化输出之上）和 softmax 函数，用于例如 RocStories/SWAG 任务。
@add_start_docstrings(
    """
    DeBERTa 模型与多选题分类头部结合（在池化输出之上的线性层和 softmax 函数），例如 RocStories/SWAG 任务。
    """,
    DEBERTA_START_DOCSTRING,
)
class DebertaV2ForMultipleChoice(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 获取标签数量，默认为 2
        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        # 初始化 DeBERTa 模型
        self.deberta = DebertaV2Model(config)
        # 初始化池化器
        self.pooler = ContextPooler(config)
        # 获取池化输出的维度
        output_dim = self.pooler.output_dim

        # 初始化分类器
        self.classifier = nn.Linear(output_dim, 1)
        # 获取分类器的 dropout 比率，默认为隐藏层 dropout 比率
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        # 初始化 dropout 层
        self.dropout = StableDropout(drop_out)

        # 初始化权重
        self.init_weights()

    # 获取输入词嵌入
    def get_input_embeddings(self):
        return self.deberta.get_input_embeddings()

    # 设置输入词嵌入
    def set_input_embeddings(self, new_embeddings):
        self.deberta.set_input_embeddings(new_embeddings)

    @add_start_docstrings_to_model_forward(DEBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 前向传播函数，接受多个输入参数，返回结果字典
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
        # 定义函数，接受输入并返回多项选择模型输出
        ) -> Union[Tuple, MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 设置返回字典，如果未指定则使用配置中的设定
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 获取选择数，如果输入id不为空，则获取其第二维的大小，否则获取输入嵌入的第二维大小
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 将输入id扁平化为二维张量，如果输入id不为空，否则设置为None
        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 使用DeBERTa模型处理扁平化的输入和其它参数，获取输出
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

        # 获取编码器层的输出
        encoder_layer = outputs[0]
        # 使用池化层处理编码器输出
        pooled_output = self.pooler(encoder_layer)
        # 使用dropout层处理池化输出
        pooled_output = self.dropout(pooled_output)
        # 使用分类器处理池化输出，得到logits
        logits = self.classifier(pooled_output)
        # 重塑logits的形状
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        # 如果传入了标签
        if labels is not None:
            # 定义交叉熵损失函数
            loss_fct = CrossEntropyLoss()
            # 计算损失
            loss = loss_fct(reshaped_logits, labels)

        # 如果不返回字典
        if not return_dict:
            # 返回输出和损失
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回多项选择模型输出
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```