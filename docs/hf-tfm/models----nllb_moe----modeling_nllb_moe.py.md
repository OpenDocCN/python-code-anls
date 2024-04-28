# `.\transformers\models\nllb_moe\modeling_nllb_moe.py`

```
# 设置文件编码格式为UTF-8
# 版权声明
# 根据Apache许可证版本2.0授权使用该文件
# 可以在遵循许可证的情况下使用，否则不得使用该文件
# 可以在以下网址获取许可证的一份副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，本软件是基于“按原样”基础分发的，
# 没有任何形式的保证或条件，无论是明示的还是隐含的
# 请参阅许可证以了解特定语言下的权限和限制

""" PyTorch NLLB-MoE model."""

# 导入模块
import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

# 导入自定义模块
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
    MoEModelOutput,
    MoEModelOutputWithPastAndCrossAttentions,
    Seq2SeqMoEModelOutput,
    Seq2SeqMoEOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_nllb_moe import NllbMoeConfig

# 获取logger对象
logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "NllbMoeConfig"
_CHECKPOINT_FOR_DOC = "hf-internal-testing/dummy-nllb-moe-2-experts"
_REAL_CHECKPOINT_FOR_DOC = "facebook/nllb-moe-54b"

# 预训练权重的ids和相应的url字典
NLLB_MOE_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/nllb-moe-54b",
    # 查看所有NLLB-MOE模型，请访问https://huggingface.co/models?filter=nllb-moe
]

# 从transformers.models.bart.modeling_bart.shift_tokens_right复制而来
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

# 从transformers.models.roberta.modeling_roberta.create_position_ids_from_input_ids复制而来
def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    # 这个函数用于创建一个递增的位置索引序列，忽略掉输入序列中的填充符号。它修改自 fairseq 库的 `utils.make_positions` 函数。
    #
    # 参数:
    #     x: torch.Tensor x: 输入序列
    #
    # 返回值:
    #     torch.Tensor: 递增的位置索引序列
    def get_incremental_state(input_ids, padding_idx, past_key_values_length):
        # 首先创建一个掩码张量，用于指示哪些位置是有效的输入(不是填充符号)
        mask = input_ids.ne(padding_idx).int()
        # 根据掩码张量计算每个位置的递增索引值
        # 1. 沿着序列维度累加掩码张量得到累加索引
        # 2. 将累加索引加上过去的 key-value 对的长度得到最终的递增索引
        # 3. 将索引值乘以掩码张量，保证只有有效位置的索引值有效
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        # 将索引值转换为 long 类型并加上填充符号的索引值
        return incremental_indices.long() + padding_idx
# 定义一个计算负载均衡损失函数的函数
def load_balancing_loss_func(router_probs: torch.Tensor, expert_indices: torch.Tensor) -> float:
    # 如果 router_probs 为 None，则返回 0 作为损失
    if router_probs is None:
        return 0
    
    # 获取专家的数量
    num_experts = router_probs.shape[-1]
    
    # 如果 expert_indices 的数据类型不是 torch.int64，则将其转换为 torch.int64
    if expert_indices.dtype != torch.int64:
        expert_indices = expert_indices.to(torch.int64)
    
    # 如果 expert_indices 的维度为 2，则在最后一个维度上增加一个维度
    if len(expert_indices.shape) == 2:
        expert_indices = expert_indices.unsqueeze(2)
    
    # 使用 one-hot 编码将 expert_indices 转换为掩码矩阵
    expert_mask = torch.nn.functional.one_hot(expert_indices, num_experts)
    
    # 对于给定的 token，确定它是否被路由到给定的专家
    expert_mask = torch.max(expert_mask, axis=-2).values
    
    # 将 expert_mask 转换为 float32 类型
    expert_mask = expert_mask.to(torch.float32)
    
    # 计算每个专家上的 token 数
    tokens_per_group_and_expert = torch.mean(expert_mask, axis=-2)
    
    # 计算每个专家的路由概率
    router_prob_per_group_and_expert = torch.mean(router_probs, axis=-2)
    
    # 计算并返回负载均衡损失
    return torch.mean(tokens_per_group_and_expert * router_prob_per_group_and_expert) * (num_experts**2)


# 定义一个生成正弦波位置编码的模块
class NllbMoeSinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.offset = 2
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.make_weights(num_positions + self.offset, embedding_dim, padding_idx)

    # 生成位置编码权重
    def make_weights(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        emb_weights = self.get_embedding(num_embeddings, embedding_dim, padding_idx)
        if hasattr(self, "weights"):
            # 将权重移动到正确的数据类型和设备
            emb_weights = emb_weights.to(dtype=self.weights.dtype, device=self.weights.device)

        # 将权重注册为模块的缓冲区
        self.register_buffer("weights", emb_weights, persistent=False)

    # 静态方法，用于获取位置编码
    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """
        构建正弦嵌入。
    
        这与 tensor2tensor 中的实现相匹配，但与 "Attention Is All You Need" 第 3.5 节的描述略有不同。
        """
        # 计算嵌入向量一半的维度
        half_dim = embedding_dim // 2
        # 计算嵌入向量的权重矩阵
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        # 如果嵌入向量的维度是奇数，则在最后一列填充 0 
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        # 如果指定了填充 index，则将该 index 对应的嵌入向量全部置为 0
        if padding_idx is not None:
            emb[padding_idx, :] = 0
    
        return emb.to(torch.get_default_dtype())
    
    @torch.no_grad()
    def forward(
        self, input_ids: torch.Tensor = None, inputs_embeds: torch.Tensor = None, past_key_values_length: int = 0
    ):
        # 如果给定了 input_ids
        if input_ids is not None:
            bsz, seq_len = input_ids.size()
            # 从 input_ids 生成 position_ids，将所有填充的 token 保持填充状态
            position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length).to(
                input_ids.device
            )
        # 否则, 如果给定了 inputs_embeds
        else:
            bsz, seq_len = inputs_embeds.size()[:-1]
            # 使用 inputs_embeds 生成 position_ids
            position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds, past_key_values_length)
    
        # 如果超出了当前嵌入权重矩阵的最大位置，则生成新的嵌入权重矩阵
        max_pos = self.padding_idx + 1 + seq_len + past_key_values_length
        if max_pos > self.weights.size(0):
            self.make_weights(max_pos + self.offset, self.embedding_dim, self.padding_idx)
    
        # 根据 position_ids 从嵌入权重矩阵中选择对应位置的嵌入向量并返回，detach() 用于截断反向传播
        return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, self.weights.shape[-1]).detach()
    
    def create_position_ids_from_inputs_embeds(self, inputs_embeds, past_key_values_length):
        """
        直接提供嵌入向量，因此无法推断哪些是填充的，因此只生成顺序的 position ids。
    
        Args:
            inputs_embeds: torch.Tensor
    
        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]
    
        # 生成 sequential 的 position ids
        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape).contiguous() + past_key_values_length
class NllbMoeTop2Router(nn.Module):
    """
    Router using tokens choose top-2 experts assignment.

    This router uses the same mechanism as in NLLB-MoE from the fairseq repository. Items are sorted by router_probs
    and then routed to their choice of expert until the expert's expert_capacity is reached. **There is no guarantee
    that each token is processed by an expert**, or that each expert receives at least one token.

    The router combining weights are also returned to make sure that the states that are not updated will be masked.

    """

    def __init__(self, config: NllbMoeConfig):
        super().__init__()
        # 初始化路由器，设定参数
        self.num_experts = config.num_experts
        self.expert_capacity = config.expert_capacity
        self.classifier = nn.Linear(config.hidden_size, self.num_experts, bias=config.router_bias)
        self.router_ignore_padding_tokens = config.router_ignore_padding_tokens
        self.dtype = getattr(torch, config.router_dtype)

        self.second_expert_policy = config.second_expert_policy
        self.normalize_router_prob_before_dropping = config.normalize_router_prob_before_dropping
        self.batch_prioritized_routing = config.batch_prioritized_routing
        self.moe_eval_capacity_token_fraction = config.moe_eval_capacity_token_fraction

    def _cast_classifier(self):
        r"""
        `bitsandbytes` `Linear8bitLt` layers does not support manual casting Therefore we need to check if they are an
        instance of the `Linear8bitLt` class by checking special attributes.
        """
        # 检查分类器是否支持手动转换类型，如果不支持，则将其转换为指定类型
        if not (hasattr(self.classifier, "SCB") or hasattr(self.classifier, "CB")):
            self.classifier = self.classifier.to(self.dtype)

    def normalize_router_probabilities(self, router_probs, top_1_mask, top_2_mask):
        # 根据路由概率和掩码计算归一化后的路由概率
        top_1_max_probs = (router_probs * top_1_mask).sum(dim=1)
        top_2_max_probs = (router_probs * top_2_mask).sum(dim=1)
        denom_s = torch.clamp(top_1_max_probs + top_2_max_probs, min=torch.finfo(router_probs.dtype).eps)
        top_1_max_probs = top_1_max_probs / denom_s
        top_2_max_probs = top_2_max_probs / denom_s
        return top_1_max_probs, top_2_max_probs

    def route_tokens(
        self,
        router_logits: torch.Tensor,
        input_dtype: torch.dtype = torch.float32,
        padding_mask: Optional[torch.LongTensor] = None,
    # 定义一个名为forward的方法，用于前向传播
    def forward(self, hidden_states: torch.Tensor, padding_mask: Optional[torch.LongTensor] = None) -> Tuple:
        r"""
        The hidden states are reshaped to simplify the computation of the router probabilities (combining weights for
        each experts.)

        Args:
            hidden_states (`torch.Tensor`):
                (batch_size, sequence_length, hidden_dim) from which router probabilities are computed.
        Returns:
            top_1_mask (`torch.Tensor` of shape (batch_size, sequence_length)):
                Index tensor of shape [batch_size, sequence_length] corresponding to the expert selected for each token
                using the top1 probabilities of the router.
            router_probabilities (`torch.Tensor` of shape (batch_size, sequence_length, nump_experts)):
                Tensor of shape (batch_size, sequence_length, num_experts) corresponding to the probabilities for each
                token and expert. Used for routing tokens to experts.
            router_logits (`torch.Tensor` of shape (batch_size, sequence_length))):
                Logits tensor of shape (batch_size, sequence_length, num_experts) corresponding to raw router logits.
                This is used later for computing router z-loss.
        """
        # 设置输入张量的数据类型
        self.input_dtype = hidden_states.dtype
        # 获取输入张量的维度信息
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        # 重新调整隐藏状态张量的形状，以简化路由器概率的计算
        hidden_states = hidden_states.reshape((batch_size * sequence_length), hidden_dim)
        # 将隐藏状态张量的数据类型转换为 self.dtype
        hidden_states = hidden_states.to(self.dtype)
        # 转换分类器的数据类型
        self._cast_classifier()
        # 通过分类器获取路由器的logits
        router_logits = self.classifier(hidden_states)
        # 使用路由器logits和输入数据类型来路由token和计算routing probabilities
        top_1_mask, router_probs = self.route_tokens(router_logits, self.input_dtype, padding_mask)
        # 返回top 1 mask和router probabilities
        return top_1_mask, router_probs
class NllbMoeDenseActDense(nn.Module):
    def __init__(self, config: NllbMoeConfig, ffn_dim: int):
        super().__init__()
        # 第一个全连接层，输入维度为模型配置的维度，输出维度为指定的MLP隐藏层维度
        self.fc1 = nn.Linear(config.d_model, ffn_dim)
        # 第二个全连接层，输入维度为MLP隐藏层维度，输出维度为模型配置的维度
        self.fc2 = nn.Linear(ffn_dim, config.d_model)
        # Dropout层，用于防止过拟合，以概率config.activation_dropout对隐藏层进行丢弃
        self.dropout = nn.Dropout(config.activation_dropout)
        # 激活函数，根据模型配置选择相应的激活函数
        self.act = ACT2FN[config.activation_function]

    def forward(self, hidden_states):
        # 第一层全连接操作，将输入的hidden_states进行线性变换
        hidden_states = self.fc1(hidden_states)
        # 使用激活函数进行非线性变换
        hidden_states = self.act(hidden_states)
        # 对隐藏层进行dropout操作
        hidden_states = self.dropout(hidden_states)
        # 如果满足一定条件，则将hidden_states转换为与第二个全连接层的权重相同的数据类型
        if (
            isinstance(self.fc2.weight, torch.Tensor)
            and hidden_states.dtype != self.fc2.weight.dtype
            and (self.fc2.weight.dtype != torch.int8 and self.fc2.weight.dtype != torch.uint8)
        ):
            hidden_states = hidden_states.to(self.fc2.weight.dtype)
        # 第二层全连接操作，将hidden_states进行线性变换
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class NllbMoeSparseMLP(nn.Module):
    r"""
    Implementation of the NLLB-MoE sparse MLP module.
    """

    def __init__(self, config: NllbMoeConfig, ffn_dim: int, expert_class: nn.Module = NllbMoeDenseActDense):
        super().__init__()
        # NLLB-MoE模型的Top-2路由器
        self.router = NllbMoeTop2Router(config)
        # MoE Token的Dropout率
        self.moe_token_dropout = config.moe_token_dropout
        # Token的Dropout层，以moe_token_dropout的概率丢弃token
        self.token_dropout = nn.Dropout(self.moe_token_dropout)
        # 专家的数量
        self.num_experts = config.num_experts

        # 用于存储多个专家模型的ModuleDict
        self.experts = nn.ModuleDict()
        # 初始化所有专家模型
        for idx in range(self.num_experts):
            # 为每个专家模型创建一个独立的实例，并添加到ModuleDict中
            self.experts[f"expert_{idx}"] = expert_class(config, ffn_dim)
    # 定义前向传播函数，接受隐藏状态和填充掩码作为输入
    def forward(self, hidden_states: torch.Tensor, padding_mask: Optional[torch.Tensor] = False):
        r"""
        The goal of this forward pass is to have the same number of operation as the equivalent `NllbMoeDenseActDense`
        (mlp) layer. This means that all of the hidden states should be processed at most twice ( since we are using a
        top_2 gating mecanism). This means that we keep the complexity to O(batch_size x sequence_length x hidden_dim)
        instead of O(num_experts x batch_size x sequence_length x hidden_dim).

        1- Get the `router_probs` from the `router`. The shape of the `router_mask` is `(batch_size X sequence_length,
        num_expert)` and corresponds to the boolean version of the `router_probs`. The inputs are masked using the
        `router_mask`.

        2- Dispatch the hidden_states to its associated experts. The router probabilities are used to weight the
        contribution of each experts when updating the masked hidden states.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_dim)`):
                The hidden states
            padding_mask (`torch.Tensor`, *optional*, defaults to `False`):
                Attention mask. Can be in the causal form or not.

        Returns:
            hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_dim)`):
                Updated hidden states
            router_logits (`torch.Tensor` of shape `(batch_size, sequence_length, num_experts)`):
                Needed for computing the loss

        """
        # 获取隐藏状态的形状信息
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        # 调用路由器的前向传播函数，获取路由概率
        top_1_mask, router_probs = self.router(hidden_states, padding_mask)
        # 将路由概率转换为布尔类型的掩码
        router_mask = router_probs.bool()
        # 将隐藏状态重新塑形，方便后续计算
        hidden_states = hidden_states.reshape((batch_size * sequence_length), hidden_dim)
        # 使用 Einsum 函数将隐藏状态按照路由掩码分派到对应的专家
        masked_hidden_states = torch.einsum("bm,be->ebm", hidden_states, router_mask)
        # 遍历专家，并更新掩码后的隐藏状态
        for idx, expert in enumerate(self.experts.values()):
            # 获取当前专家对应的 token 索引
            token_indices = router_mask[:, idx]
            # 根据路由概率获取组合权重
            combining_weights = router_probs[token_indices, idx]
            # 调用专家的前向传播函数，获取专家输出
            expert_output = expert(masked_hidden_states[idx, token_indices])
            # 如果存在 token dropout，则在训练时进行处理
            if self.moe_token_dropout > 0:
                if self.training:
                    expert_output = self.token_dropout(expert_output)
                else:
                    expert_output *= 1 - self.moe_token_dropout
            # 更新掩码后的隐藏状态
            masked_hidden_states[idx, token_indices] = torch.einsum("b,be->be", combining_weights, expert_output)
        # 将所有专家的输出加总，重新塑形为原始形状
        hidden_states = masked_hidden_states.sum(dim=0).reshape(batch_size, sequence_length, hidden_dim)

        # 获取最大路由概率对应的专家索引，用于计算损失
        top_1_expert_index = torch.argmax(top_1_mask, dim=-1)
        # 返回更新后的隐藏状态和路由概率以及最大路由概率对应的专家索引
        return hidden_states, (router_probs, top_1_expert_index)
# 从transformers.models.bart.modeling_bart.BartAttention复制而来，将Bart改为NllbMoe，key_value_states改为encoder_hidden_states
class NllbMoeAttention(nn.Module):
    """来自《Attention Is All You Need》论文的多头注意力机制"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[NllbMoeConfig] = None,
    ):
        super().__init__()
        # 初始化注意力机制的参数
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        # 检查参数是否合法
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim必须能被num_heads整除（得到的 `embed_dim`: {self.embed_dim}"
                f" 和 `num_heads`: {num_heads}）."
            )
        # 设置缩放因子
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        # 初始化线性层
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    # 将张量重塑为适合多头注意力的形状
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
# NllbMoeEncoderLayer类
class NllbMoeEncoderLayer(nn.Module):
    def __init__(self, config: NllbMoeConfig, is_sparse: bool = False):
        super().__init__()
        # 初始化编码器层参数
        self.embed_dim = config.d_model
        self.is_sparse = is_sparse
        # 初始化自注意力机制、层标准化和前馈神经网络
        self.self_attn = NllbMoeAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.attn_dropout = nn.Dropout(config.dropout)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        if not self.is_sparse:
            self.ffn = NllbMoeDenseActDense(config, ffn_dim=config.encoder_ffn_dim)
        else:
            self.ffn = NllbMoeSparseMLP(config, ffn_dim=config.encoder_ffn_dim)
        self.ff_layer_norm = nn.LayerNorm(config.d_model)
        self.ff_dropout = nn.Dropout(config.activation_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
        output_router_logits: bool = False,
    # 定义一个前向传播的函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
        output_router_logits: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`):
                attention mask of size `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very
                large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        # 保存输入的hidden_states作为残差连接
        residual = hidden_states
        # 通过层归一化处理输入
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 执行self-attention操作, 获得输出hidden_states, 注意力权重attn_weights以及可选的其他输出
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # 对self-attention输出应用dropout
        hidden_states = self.attn_dropout(hidden_states)
        # 将self-attention输出与残差相加
        hidden_states = residual + hidden_states
    
        # 保存当前hidden_states作为新的残差连接
        residual = hidden_states
        # 通过层归一化处理hidden_states
        hidden_states = self.ff_layer_norm(hidden_states)
        # 如果是稀疏模式, 执行前馈网络计算, 获得hidden_states和router_states
        if self.is_sparse:
            hidden_states, router_states = self.ffn(hidden_states, attention_mask)
        # 如果不是稀疏模式, 只执行前馈网络计算, router_states设为None
        else:
            hidden_states, router_states = self.ffn(hidden_states), None
        # 对前馈网络输出应用dropout
        hidden_states = self.ff_dropout(hidden_states)
        # 将前馈网络输出与残差相加
        hidden_states = residual + hidden_states
    
        # 如果hidden_states中存在inf或nan, 则将其限制在合理范围内
        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
    
        # 构建输出, 第一个元素是hidden_states
        outputs = (hidden_states,)
        # 如果需要, 添加注意力权重attn_weights到输出
        if output_attentions:
            outputs += (attn_weights,)
        # 如果需要, 添加router_states到输出
        if output_router_logits:
            outputs += (router_states,)
    
        return outputs
class NllbMoeDecoderLayer(nn.Module):
    def __init__(self, config: NllbMoeConfig, is_sparse: bool = False):
        super().__init__()
        self.embed_dim = config.d_model
        self.is_sparse = is_sparse
        # 初始化自注意力机制，用于解码器自身的注意力
        self.self_attn = NllbMoeAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        # 初始化自注意力机制的Dropout层
        self.attn_dropout = nn.Dropout(config.dropout)

        # 初始化自注意力机制的Layer Norm层
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 初始化交叉注意力机制，用于解码器和编码器之间的注意力
        self.cross_attention = NllbMoeAttention(
            self.embed_dim, config.decoder_attention_heads, config.attention_dropout, is_decoder=True
        )
        # 初始化交叉注意力机制的Layer Norm层
        self.cross_attention_layer_norm = nn.LayerNorm(self.embed_dim)
        # 根据是否稀疏设置前馈网络
        if not self.is_sparse:
            # 初始化前馈网络（非稀疏版本）
            self.ffn = NllbMoeDenseActDense(config, ffn_dim=config.decoder_ffn_dim)
        else:
            # 初始化前馈网络（稀疏版本）
            self.ffn = NllbMoeSparseMLP(config, ffn_dim=config.decoder_ffn_dim)
        # 初始化前馈网络的Layer Norm层
        self.ff_layer_norm = nn.LayerNorm(config.d_model)
        # 初始化前馈网络的Dropout层
        self.ff_dropout = nn.Dropout(config.activation_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = True,
class NllbMoePreTrainedModel(PreTrainedModel):
    config_class = NllbMoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["NllbMoeEncoderLayer", "NllbMoeDecoderLayer"]

    def _init_weights(self, module):
        """Initialize the weights"""
        # 根据模块类型初始化权重
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            # 如果是线性层，使用正态分布初始化权重，偏置初始化为零
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 如果是嵌入层，使用正态分布初始化权重，偏置初始化为零
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


NLLB_MOE_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    # 将其用作普通的PyTorch模块，并参考PyTorch文档了解所有与一般用法和行为相关的事项。

    # 参数:
    #     config ([`NllbMoeConfig`]):
    #         包含模型所有参数的模型配置类。使用配置文件初始化只会加载与模型相关的配置，不会加载模型的权重。
    #         查看[`~PreTrainedModel.from_pretrained`]方法以加载模型权重。
# 创建示例
NLLB_MOE_GENERATION_EXAMPLE = r"""
    Translation example:
    """
    """python
    >>> from transformers import AutoTokenizer, NllbMoeForConditionalGeneration

    >>> model = NllbMoeForConditionalGeneration.from_pretrained("facebook/nllb-moe-54b")
    >>> tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-moe-54b")

    >>> text_to_translate = "Life is like a box of chocolates"
    >>> model_inputs = tokenizer(text_to_translate, return_tensors="pt")

    >>> # translate to French
    >>> gen_tokens = model.generate(**model_inputs, forced_bos_token_id=tokenizer.get_lang_id("eng_Latn"))
    >>> print(tokenizer.batch_decode(gen_tokens, skip_special_tokens=True))
    """
"""

NLLB_MOE_INPUTS_DOCSTRING = r"""
"""

# 定义 NllbMoeEncoder 类，继承自 NllbMoePreTrainedModel
class NllbMoeEncoder(NllbMoePreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`NllbMoeEncoderLayer`].

    Args:
        config:
            NllbMoeConfig
        embed_tokens (nn.Embedding):
            output embedding
    """

    # 初始化方法
    def __init__(self, config: NllbMoeConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        # 初始化一些属性
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        # 如果有传入自定义的嵌入词汇表，使用自定义的
        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = NllbMoeSinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
            self.padding_idx,
        )
        sparse_step = config.encoder_sparse_step
        self.layers = nn.ModuleList()
        # 创建编码器层
        for i in range(config.encoder_layers):
            is_sparse = (i + 1) % sparse_step == 0 if sparse_step > 0 else False
            self.layers.append(NllbMoeEncoderLayer(config, is_sparse))

        self.layer_norm = nn.LayerNorm(config.d_model)
        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 定义 NllbMoeDecoder 类，用于实现 NLLB (No Language Left Behind) Mixture-of-Experts 解码器
    class NllbMoeDecoder(BaseModelOutputWithPastAndCrossAttentions):
        """
        实现 NLLB (No Language Left Behind) Mixture-of-Experts 解码器，主要包含以下功能:
        1. 使用 nn.Embedding 层对输入进行词嵌入
        2. 使用 NllbMoeSinusoidalPositionalEmbedding 层增加位置编码
        3. 使用多个 NllbMoeDecoderLayer 层进行解码处理
        4. 最后使用 nn.LayerNorm 层进行归一化
        
        Args:
            config (NllbMoeConfig):
                解码器的配置参数
            embed_tokens (nn.Embedding):
                输出词嵌入层
        """
    
        def __init__(self, config: NllbMoeConfig, embed_tokens: Optional[nn.Embedding] = None):
            # 调用父类构造函数
            super().__init__(config)
            # 设置 dropout 和 layerdrop 参数
            self.dropout = config.dropout
            self.layerdrop = config.decoder_layerdrop
            # 设置 padding 索引和最大目标位置
            self.padding_idx = config.pad_token_id
            self.max_target_positions = config.max_position_embeddings
            # 计算缩放因子
            self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
            
            # 创建词嵌入层
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
            
            # 如果提供了外部词嵌入层，则使用它
            if embed_tokens is not None:
                self.embed_tokens.weight = embed_tokens.weight
    
            # 创建位置编码层
            self.embed_positions = NllbMoeSinusoidalPositionalEmbedding(
                config.max_position_embeddings,
                config.d_model,
                self.padding_idx,
            )
    
            # 创建多个解码层
            sparse_step = config.decoder_sparse_step
            self.layers = nn.ModuleList()
            for i in range(config.decoder_layers):
                is_sparse = (i + 1) % sparse_step == 0 if sparse_step > 0 else False
                self.layers.append(NllbMoeDecoderLayer(config, is_sparse))
    
            # 创建最终的归一化层
            self.layer_norm = nn.LayerNorm(config.d_model)
    
            # 是否启用梯度检查点
            self.gradient_checkpointing = False
            
            # 初始化权重并应用最终处理
            self.post_init()
    
        def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_router_logits: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ):
            # 此处省略了forward 方法的具体实现
            pass
# 加载 NllbMoe 预训练模型的文档字符串
@add_start_docstrings(
    "The bare NllbMoe Model outputting raw hidden-states without any specific head on top.",
    NLLB_MOE_START_DOCSTRING,
)
# 定义 NllbMoeModel 类，继承自 NllbMoePreTrainedModel
class NllbMoeModel(NllbMoePreTrainedModel):
    # 定义需要共享权重的键
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    # 初始化函数
    def __init__(self, config: NllbMoeConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # 根据配置获取填充token ID和词表大小
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        # 创建共享的词嵌入层
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        # 创建编码器和解码器
        self.encoder = NllbMoeEncoder(config, self.shared)
        self.decoder = NllbMoeDecoder(config, self.shared)

        # 进行模型权重初始化和后处理
        self.post_init()

    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.shared

    # 设置输入嵌入层
    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    # 绑定编码器和解码器的词嵌入层
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    # 获取编码器
    def get_encoder(self):
        return self.encoder

    # 获取解码器
    def get_decoder(self):
        return self.decoder

    # 定义模型前向传播
    @add_start_docstrings_to_model_forward(NLLB_MOE_INPUTS_DOCSTRING)
    @add_start_docstrings_to_model_forward(NLLB_MOE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqMoEModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,

# 加载 NllbMoe 生成模型的文档字符串
@add_start_docstrings(
    "The NllbMoe Model with a language modeling head. Can be used for summarization.", NLLB_MOE_START_DOCSTRING
)
# 定义 NllbMoeForConditionalGeneration 类，继承自 NllbMoePreTrainedModel
class NllbMoeForConditionalGeneration(NllbMoePreTrainedModel):
    # 设置模型前缀
    base_model_prefix = "model"
    # 定义需要共享权重的键
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]
    # 使用给定的配置初始化模型，继承了父类的初始化方法
    def __init__(self, config: NllbMoeConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建 NllbMoeModel 模型实例
        self.model = NllbMoeModel(config)
        # 创建线性层，用于最后的输出，将模型的隐藏状态映射到词汇表大小的向量
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 设置路由器损失的权重
        self.router_z_loss_coef = config.router_z_loss_coef
        # 设置路由器辅助损失的权重
        self.router_aux_loss_coef = config.router_aux_loss_coef
        # 初始化权重并应用最终处理
        self.post_init()

    # 获取编码器
    def get_encoder(self):
        return self.model.get_encoder()

    # 获取解码器
    def get_decoder(self):
        return self.model.get_decoder()

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 前向传播方法，接收各种输入，返回输出
    @add_start_docstrings_to_model_forward(NLLB_MOE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqMoEOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(NLLB_MOE_GENERATION_EXAMPLE)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 解包路由器输出的方法
    def _unpack_router_logits(self, router_outputs):
        # 初始化总路由器 logits 和总专家索引的列表
        total_router_logits = []
        total_expert_indexes = []
        # 遍历每个路由器输出
        for router_output in router_outputs:
            # 如果路由器输出不为空
            if router_output is not None:
                # 获取路由器 logits 和专家索引
                router_logits, expert_indexes = router_output
                # 将路由器 logits 和专家索引添加到总列表中
                total_router_logits.append(router_logits)
                total_expert_indexes.append(expert_indexes)

        # 将所有路由器 logits 拼接在一起，如果有的话
        total_router_logits = torch.cat(total_router_logits, dim=1) if len(total_router_logits) > 0 else None
        # 将所有专家索引堆叠在一起，如果有的话
        total_expert_indexes = torch.stack(total_expert_indexes, dim=1) if len(total_expert_indexes) > 0 else None
        # 返回总路由器 logits 和总专家索引
        return total_router_logits, total_expert_indexes

    # 从 transfomers.models.switch_transformers.SwitchTransformersForConditionalGeneration.prepare_inputs_for_generation 复制而来
    # 为生成准备输入数据，这是一个类方法
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,  # 解码器输入的标识符序列
        past_key_values=None,  # 存储了过去的键值对，用于生成时的状态保持
        attention_mask=None,  # 注意力遮罩，指定哪些位置需要被注意到
        head_mask=None,  # 多头注意力的头部掩码
        decoder_head_mask=None,  # 解码器的多头注意力头部掩码
        cross_attn_head_mask=None,  # 交叉注意力的多头注意力头部掩码
        use_cache=None,  # 控制是否使用缓存
        encoder_outputs=None,  # 编码器输出
        **kwargs,  # 其他关键字参数
    ):
        # 如果使用过去的键值对
        if past_key_values is not None:
            # 获取过去的长度
            past_length = past_key_values[0][0].shape[2]

            # 如果解码器输入的长度大于过去的长度
            if decoder_input_ids.shape[1] > past_length:
                # 移除前缀长度为过去的长度
                remove_prefix_length = past_length
            else:
                # 否则默认只保留最后一个标识符
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            # 更新解码器输入的标识符序列，仅保留最近输入的部分
            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        # 返回准备好的输入数据
        return {
            "input_ids": None,  # 因为已经提供了编码器输出，所以不需要输入标识符
            "encoder_outputs": encoder_outputs,  # 编码器输出
            "past_key_values": past_key_values,  # 过去的键值对
            "decoder_input_ids": decoder_input_ids,  # 更新后的解码器输入标识符序列
            "attention_mask": attention_mask,  # 注意力遮罩
            "head_mask": head_mask,  # 头部掩码
            "decoder_head_mask": decoder_head_mask,  # 解码器头部掩码
            "cross_attn_head_mask": cross_attn_head_mask,  # 交叉注意力头部掩码
            "use_cache": use_cache,  # 控制是否使用缓存
        }

    # 静态方法：重新排序缓存中的内容
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        # 重新排序过去的键值对
        reordered_past = ()
        # 遍历每一层的过去键值对
        for layer_past in past_key_values:
            # 对每一层的过去状态按照beam_idx重新排序，并添加到结果中
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重新排序后的过去键值对
        return reordered_past
```