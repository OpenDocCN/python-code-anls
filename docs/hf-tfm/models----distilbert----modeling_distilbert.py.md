# `.\models\distilbert\modeling_distilbert.py`

```py
# 设置脚本编码格式为 UTF-8
# 版权声明信息包括当前年份和版权方信息
# 根据 Apache License, Version 2.0 许可证使用此文件，并遵循许可证的规定
# 可以在以下链接获取许可证的副本: http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则将根据“ AS IS” 基础分发此软件，没有任何担保或条件，无论是明示或暗示的
# 请参见许可证以查看特定语言的许可证给予的权限和限制

"""
 从 Facebook, Inc XLM 模型 (https://github.com/facebookresearch/XLM) 和 HuggingFace PyTorch Google AI Bert 模型 (https://github.com/google-research/bert) 中部分改编的 PyTorch DistilBERT 模型
"""

# 导入所需的库和模块
import math
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import get_activation
from ...configuration_utils import PretrainedConfig
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from .configuration_distilbert import DistilBertConfig

# 检查是否存在特定版本的 Flash Attention
if is_flash_attn_2_available():
    # 导入 Flash Attention 相关函数和模块
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input

# 获取 logger 对象
logger = logging.get_logger(__name__)
# 用于文档的模型检查点和配置
_CHECKPOINT_FOR_DOC = "distilbert-base-uncased"
_CONFIG_FOR_DOC = "DistilBertConfig"

# 预训练的 DistilBERT 模型归档列表
DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "distilbert-base-uncased",
    "distilbert-base-uncased-distilled-squad",
    "distilbert-base-cased",
    "distilbert-base-cased-distilled-squad",
    "distilbert-base-german-cased",
    "distilbert-base-multilingual-cased",
    "distilbert-base-uncased-finetuned-sst-2-english",
    # 可以查看所有 DistilBERT 模型：https://huggingface.co/models?filter=distilbert
]

# 架构的 UTILS 和 BUILDING BLOCKS #
# 从 transformers.models.llama.modeling_llama._get_unpad_data 复制函数
def _get_unpad_data(attention_mask):
    # 计算每个序列在批次中的长度
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    # 找到非零元素的索引
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    # 获取批次中序列长度的最大值
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    # 计算累积序列长度，并在左侧填充一个额外的零，以便与索引对齐
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    # 返回结果：索引、累积序列长度、批次中的最大序列长度
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )
# 创建 sinusoidal embeddings 的函数，用于生成位置编码
def create_sinusoidal_embeddings(n_pos: int, dim: int, out: torch.Tensor):
    # 如果启用了 DeepSpeed 的 zero3 功能
    if is_deepspeed_zero3_enabled():
        # 导入 deepspeed 库
        import deepspeed
        # 使用 DeepSpeed 的 zero.GatheredParameters 将参数分布到所有的 GPU
        with deepspeed.zero.GatheredParameters(out, modifier_rank=0):
            # 如果当前进程的 rank 为 0
            if torch.distributed.get_rank() == 0:
                # 调用内部函数 _create_sinusoidal_embeddings 生成 sinusoidal embeddings
                _create_sinusoidal_embeddings(n_pos=n_pos, dim=dim, out=out)
    else:
        # 否则直接调用内部函数 _create_sinusoidal_embeddings 生成 sinusoidal embeddings
        _create_sinusoidal_embeddings(n_pos=n_pos, dim=dim, out=out)


# 内部函数，用于生成 sinusoidal embeddings
def _create_sinusoidal_embeddings(n_pos: int, dim: int, out: torch.Tensor):
    # 生成位置编码矩阵
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    # 将输出张量设置为不需要梯度
    out.requires_grad = False
    # 使用 sine 函数生成 sinusoidal embeddings
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    # 使用 cosine 函数生成 sinusoidal embeddings
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    # 将输出张量从计算图中分离
    out.detach_()


# Embeddings 类，用于处理词嵌入和位置编码
class Embeddings(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        # 词嵌入层，根据配置参数创建
        self.word_embeddings = nn.Embedding(config.vocab_size, config.dim, padding_idx=config.pad_token_id)
        # 位置编码层，根据配置参数创建
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.dim)
        # 如果配置参数中启用了 sinusoidal_pos_embds
        if config.sinusoidal_pos_embds:
            # 调用 create_sinusoidal_embeddings 函数生成 sinusoidal embeddings
            create_sinusoidal_embeddings(
                n_pos=config.max_position_embeddings, dim=config.dim, out=self.position_embeddings.weight
            )
        # LayerNorm 层，用于层标准化
        self.LayerNorm = nn.LayerNorm(config.dim, eps=1e-12)
        # Dropout 层，用于随机失活
        self.dropout = nn.Dropout(config.dropout)
        # 注册位置编码的位置标识，用于后续的位置编码
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
    def forward(self, input_ids: torch.Tensor, input_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters:
            input_ids (torch.Tensor):
                torch.tensor(bs, max_seq_length) The token ids to embed.
            input_embeds (*optional*, torch.Tensor):
                The pre-computed word embeddings. Can only be passed if the input ids are `None`.

        Returns: 
            torch.tensor(bs, max_seq_length, dim) The embedded tokens (plus position embeddings, no token_type embeddings)
        """
        if input_ids is not None:
            input_embeds = self.word_embeddings(input_ids)  # (bs, max_seq_length, dim)

        seq_length = input_embeds.size(1)

        # Setting the position-ids to the registered buffer in constructor, it helps
        # when tracing the model without passing position-ids, solves
        # isues similar to issue #5664
        if hasattr(self, "position_ids"):
            position_ids = self.position_ids[:, :seq_length]
        else:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)  # (max_seq_length)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (bs, max_seq_length)

        position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)

        embeddings = input_embeds + position_embeddings  # (bs, max_seq_length, dim)
        embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
        return embeddings
# 定义一个多头自注意力机制的类，继承自 nn.Module
class MultiHeadSelfAttention(nn.Module):
    # 初始化函数，接受一个预训练配置的参数
    def __init__(self, config: PretrainedConfig):
        # 调用父类的初始化函数
        super().__init__()
        # 将传入的配置参数保存在类中
        self.config = config

        # 从配置中获取多头数和维度信息
        self.n_heads = config.n_heads
        self.dim = config.dim
        # 使用配置中的注意力丢弃率创建一个丢弃层
        self.dropout = nn.Dropout(p=config.attention_dropout)
        # 默认不是因果关系的注意力
        self.is_causal = False

        # 确保多头数能够均匀划分维度
        if self.dim % self.n_heads != 0:
            # 如果不能均匀划分，则引发 ValueError 异常
            raise ValueError(f"self.n_heads: {self.n_heads} must divide self.dim: {self.dim} evenly")

        # 分别创建线性层来处理查询、键和值
        self.q_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.k_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.v_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        # 创建一个输出线性层来整合多头自注意力输出
        self.out_lin = nn.Linear(in_features=config.dim, out_features=config.dim)

        # 初始化一个空集合来保存被剪枝的头
        self.pruned_heads: Set[int] = set()
        # 计算每个注意力头的大小
        self.attention_head_size = self.dim // self.n_heads

    # 头剪枝函数，接受要剪枝的头列表
    def prune_heads(self, heads: List[int]):
        # 如果没有要剪枝的头，直接返回
        if len(heads) == 0:
            return
        # 找到可剪枝的头和对应的索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.attention_head_size, self.pruned_heads
        )
        # 剪枝线性层
        self.q_lin = prune_linear_layer(self.q_lin, index)
        self.k_lin = prune_linear_layer(self.k_lin, index)
        self.v_lin = prune_linear_layer(self.v_lin, index)
        self.out_lin = prune_linear_layer(self.out_lin, index, dim=1)
        # 更新超参数
        self.n_heads = self.n_heads - len(heads)
        self.dim = self.attention_head_size * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 前向传播函数，接受查询、键、值、掩码等参数
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Parameters:
            query: torch.tensor(bs, seq_length, dim) 查询张量
            key: torch.tensor(bs, seq_length, dim) 键张量
            value: torch.tensor(bs, seq_length, dim) 值张量
            mask: torch.tensor(bs, seq_length) 掩码张量

        Returns:
            weights: torch.tensor(bs, n_heads, seq_length, seq_length) 注意力权重张量 context: torch.tensor(bs,
            seq_length, dim) 上下文化层。 可选项：仅在 `output_attentions=True` 时返回
        """
        bs, q_length, dim = query.size()  # 获取查询张量的形状信息
        k_length = key.size(1)  # 获取键张量的长度
        # assert dim == self.dim, f'Dimensions do not match: {dim} input vs {self.dim} configured'
        # assert key.size() == value.size()

        dim_per_head = self.dim // self.n_heads  # 计算每个头的维度

        mask_reshp = (bs, 1, 1, k_length)  # 重新整形掩码张量的形状

        def shape(x: torch.Tensor) -> torch.Tensor:
            """separate heads"""
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)  # 将张量分割成多个头

        def unshape(x: torch.Tensor) -> torch.Tensor:
            """group heads"""
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)  # 将多个头合并成一个张量

        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head) 对查询进行线性变换
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head) 对键进行线性变换
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head) 对值进行线性变换

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head) 缩放查询张量
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length) 计算注意力分数
        mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length) 扩展掩码张量
        scores = scores.masked_fill(
            mask, torch.tensor(torch.finfo(scores.dtype).min)
        )  # (bs, n_heads, q_length, k_length) 对掩码位置进行填充

        weights = nn.functional.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length) 计算注意力权重
        weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length) 对注意力权重进行 dropout

        # 如果需要，对头部进行掩码
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head) 计算上下文化张量
        context = unshape(context)  # (bs, q_length, dim) 合并多个头
        context = self.out_lin(context)  # (bs, q_length, dim) 最终输出层线性变换

        if output_attentions:
            return (context, weights)  # 如果需要输出注意力权重，返回上下文化张量和注意力权重
        else:
            return (context,)  # 否则，仅返回上下文化张量
# 定义 DistilBertFlashAttention2 类，继承自 MultiHeadSelfAttention
class DistilBertFlashAttention2(MultiHeadSelfAttention):
    """
    DistilBert flash attention module. This module inherits from `MultiHeadSelfAttention` as the weights of the module
    stays untouched. The only required change would be on the forward pass where it needs to correctly call the public
    API of flash attention and deal with padding tokens in case the input contains any of them.
    """

    # 重写 __init__ 方法，继承自父类 MultiHeadSelfAttention 的 __init__ 方法
    def __init__(self, *args, **kwargs):
        # 调用父类的初始化方法
        super().__init__(*args, **kwargs)

        # 临时解决方案，等 Flash Attention 版本更新到 2.1 可以移除
        # 判断 Flash Attention 的版本是否大于等于 2.1，不大于则使用旧的顶部对齐的掩码
        # 参考：https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    # 重写 forward 方法
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Parameters:
            query: torch.tensor(bs, seq_length, dim)  # 查询张量，形状为(batch_size, seq_length, dim)
            key: torch.tensor(bs, seq_length, dim)  # 键张量，形状为(batch_size, seq_length, dim)
            value: torch.tensor(bs, seq_length, dim)  # 值张量，形状为(batch_size, seq_length, dim)
            mask: torch.tensor(bs, seq_length)  # 掩码张量，形状为(batch_size, seq_length)

        Returns:
            weights: torch.tensor(bs, n_heads, seq_length, seq_length) Attention weights context: torch.tensor(bs,
            seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`  # 权重张量和上下文张量
        """
        batch_size, q_length, dim = query.size()  # 获取查询张量的维度信息

        dim_per_head = self.dim // self.n_heads  # 计算每个头部的维度

        def reshape(x: torch.Tensor) -> torch.Tensor:
            """separate heads"""
            return x.view(batch_size, -1, self.n_heads, dim_per_head)  # 对输入张量的维度进行重组和分隔，以便于多头注意力计算

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        query_states = reshape(self.q_lin(query))  # 将查询张量线性变换并进行重组
        key_states = reshape(self.k_lin(key))  # 将键张量线性变换并进行重组
        value_states = reshape(self.v_lin(value))  # 将值张量线性变换并进行重组

        attn_dropout = self.config.attention_dropout if self.training else 0.0  # 获取注意力层的dropout率

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        if query_states.dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()  # 获取自动混合精度训练的GPU数据类型
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype  # 获取预量化数据类型
            else:
                target_dtype = self.q_lin.weight.dtype  # 获取线性变换层的权重数据类型

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )  # 输出警告，提醒输入隐藏状态可能被静默地转换为float32，并将输入张量重新转换为正确的数据类型

            query_states = query_states.to(target_dtype)  # 转换查询张量的数据类型
            key_states = key_states.to(target_dtype)  # 转换键张量的数据类型
            value_states = value_states.to(target_dtype)  # 转换值张量的数据类型

        attn_weights = self._flash_attention_forward(
            query_states, key_states, value_states, mask, q_length, dropout=attn_dropout
        )  # 执行闪回式注意力前向传播，得到注意力权重

        attn_weights_reshaped = attn_weights.reshape(batch_size, q_length, self.n_heads * dim_per_head)  # 重新整形注意力权重张量
        attn_output = self.out_lin(attn_weights_reshaped)  # 线性变换得到最终的上下文张量

        if output_attentions:  # 如果需要输出注意力权重
            return (attn_output, attn_weights)  # 返回上下文张量和
    # 从transformers.models.llama.modeling_llama.LlamaFlashAttention2._flash_attention_forward 复制而来，将causal=True改为causal=False
    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        调用Flash Attention的前向方法 - 如果输入的隐藏状态包含至少一个填充令牌，则先取消填充输入，然后计算注意力分数并填充最终的注意力分数。

        Args:
            query_states (`torch.Tensor`):
                要传递给Flash Attention API的输入查询状态
            key_states (`torch.Tensor`):
                要传递给Flash Attention API的输入键状态
            value_states (`torch.Tensor`):
                要传递给Flash Attention API的输入值状态
            attention_mask (`torch.Tensor`):
                填充掩码 - 对应大小为`（batch_size，seq_len）`的张量，其中0表示填充令牌的位置，1表示非填充令牌的位置。
            dropout (`int`, *optional*):
                注意力的丢弃率
            softmax_scale (`float`, *optional*):
                在应用softmax之前对QK^T进行缩放。默认为1 / sqrt(head_dim)
        """
        如果闪存注意力不使用左上角掩码:
            causal = self.is_causal
        else:
            # TODO: 一旦闪存注意力升级到2.1版本，删除`query_length != 1`的检查。有关详细信息，请查看LlamaFlashAttention2 __init__中的注释。
            causal = self.is_causal and query_length != 1

        # 序列中至少包含一个填充令牌
        如果注意力掩码不为None：
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        否则:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        返回attn_output
    # 从transformers.models.llama.modeling_llama.LlamaFlashAttention2._upad_input复制的函数，将num_heads更名为n_heads
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        # 从注意力掩码中获取解压数据的索引、当前序列长度和批次中最大序列长度
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        # 获取批次大小、键值序列长度、键值头数和头维度
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape
    
        # 对键值层进行索引操作，重新组织形状为(batch_size * kv_seq_len, num_key_value_heads, head_dim)的键值层
        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        # 对值层进行索引操作，重新组织形状为(batch_size * kv_seq_len, num_key_value_heads, head_dim)的值层
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        # 如果查询长度等于键值序列长度
        if query_length == kv_seq_len:
            # 对查询层进行索引操作，重新组织形状为(batch_size * kv_seq_len, self.n_heads, head_dim)的查询层
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.n_heads, head_dim), indices_k
            )
            # 保存查询序列长度、当前序列长度和最大序列长度，并使用与键相同的索引
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        # 如果查询长度等于1
        elif query_length == 1:
            # 将最大序列长度设置为1，创建一个张量表示序列长度（长度为batch_size），并使用设备和数据类型来自查询层的设备和数据类型
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            # 创建索引以与查询序列长度相同，并将查询层维度中的单维度去除
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # 对注意力掩码进行切片以获取最后查询长度个标记的掩码
            attention_mask = attention_mask[:, -query_length:]
            # 对输入序列进行解压操作，并获取解压后的查询层、索引、当前序列长度和最大序列长度
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)
    
        # 返回更新后的查询层、键层、值层、查询索引、序列长度元组和最大序列长度元组
        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
class FFN(nn.Module):
    def __init__(self, config: PretrainedConfig):
        # 初始化 FFN 类
        super().__init__()
        # 初始化丢弃层
        self.dropout = nn.Dropout(p=config.dropout)
        # 设定前馈块的分块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 设置序列长度的维度
        self.seq_len_dim = 1
        # 初始化第一个全连接层
        self.lin1 = nn.Linear(in_features=config.dim, out_features=config.hidden_dim)
        # 初始化第二个全连接层
        self.lin2 = nn.Linear(in_features=config.hidden_dim, out_features=config.dim)
        # 初始化激活函数
        self.activation = get_activation(config.activation)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 应用分块方法执行前向传播
        return apply_chunking_to_forward(self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, input)

    def ff_chunk(self, input: torch.Tensor) -> torch.Tensor:
        # 执行前馈块操作
        x = self.lin1(input)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x


DISTILBERT_ATTENTION_CLASSES = {
    "eager": MultiHeadSelfAttention,
    "flash_attention_2": DistilBertFlashAttention2,
}

class TransformerBlock(nn.Module):
    def __init__(self, config: PretrainedConfig):
        # 初始化 TransformerBlock 类
        super().__init__()

        # 如果维度不能被多头注意力头数整除，则抛出异常
        if config.dim % config.n_heads != 0:
            raise ValueError(f"config.n_heads {config.n_heads} must divide config.dim {config.dim} evenly")

        # 选择多头注意力实现类，并传入配置
        self.attention = DISTILBERT_ATTENTION_CLASSES[config._attn_implementation](config)
        # 使用 LayerNorm 对自注意力输出进行归一化
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)
        # 初始化前馈神经网络
        self.ffn = FFN(config)
        # 对输出应用 LayerNorm 进行归一化
        self.output_layer_norm = nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        ```
        ) -> Tuple[torch.Tensor, ...]:
        """
        Parameters:
            x: torch.tensor(bs, seq_length, dim) - 输入张量，形状为(bs, seq_length, dim)
            attn_mask: torch.tensor(bs, seq_length) - 注意力掩码张量，形状为(bs, seq_length)

        Returns:
            sa_weights: torch.tensor(bs, n_heads, seq_length, seq_length) - 自注意力权重张量，形状为(bs, n_heads, seq_length, seq_length)，注意力权重
            ffn_output: torch.tensor(bs, seq_length, dim) - 输出张量，形状为(bs, seq_length, dim)，变换器块的上下文化输出
        """
        # Self-Attention
        # 自注意力
        sa_output = self.attention(
            query=x,  # 查询张量
            key=x,  # 键张量
            value=x,  # 值张量
            mask=attn_mask,  # 注意力掩码
            head_mask=head_mask,  # 头部掩码
            output_attentions=output_attentions,  # 是否输出注意力权重
        )
        if output_attentions:
            sa_output, sa_weights = sa_output  # 如果输出注意力权重，解包注意力输出和注意力权重
            # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
        else:  # 处理返回元组的情况
            if type(sa_output) != tuple:
                raise TypeError(f"sa_output must be a tuple but it is {type(sa_output)} type")
            # 抛出类型错误异常

            sa_output = sa_output[0]  # 获取注意力输出元组的第一个元素
        sa_output = self.sa_layer_norm(sa_output + x)  # 注意力输出与输入张量相加后进行 layer normalization

        # Feed Forward Network
        # 前馈网络
        ffn_output = self.ffn(sa_output)  # 输入注意力输出进行前馈网络计算
        # (bs, seq_length, dim)
        ffn_output: torch.Tensor = self.output_layer_norm(ffn_output + sa_output)  # 前馈网络输出与注意力输出相加后进行 layer normalization
        # (bs, seq_length, dim)

        output = (ffn_output,)  # 输出元组
        if output_attentions:
            output = (sa_weights,) + output  # 如果输出注意力权重，将注意力权重添加到输出元组中
        return output  # 返回输出元组
# 定义一个 Transformer 类，继承自 nn.Module
class Transformer(nn.Module):
    # 初始化方法，接受一个预训练配置对象作为参数
    def __init__(self, config: PretrainedConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 设置层数为配置对象中的层数
        self.n_layers = config.n_layers
        # 使用列表推导式创建包含多个 TransformerBlock 对象的 ModuleList
        self.layer = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        # 设置梯度检查点为 False
        self.gradient_checkpointing = False

    # 前向传播方法
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: Optional[bool] = None,
    ) -> Union[BaseModelOutput, Tuple[torch.Tensor, ...]]:  # docstyle-ignore
        """
        Parameters:
            x: torch.tensor(bs, seq_length, dim) Input sequence embedded.
            attn_mask: torch.tensor(bs, seq_length) Attention mask on the sequence.

        Returns:
            hidden_state: torch.tensor(bs, seq_length, dim) Sequence of hidden states in the last (top)
            layer all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
                Tuple of length n_layers with the hidden states from each layer.
                Optional: only if output_hidden_states=True
            all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
                Tuple of length n_layers with the attention weights from each layer
                Optional: only if output_attentions=True
        """
        # 初始化存储所有隐藏状态和注意力权重的变量
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # 初始化隐藏状态为输入序列
        hidden_state = x
        # 遍历每个层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到存储中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

            # 梯度检查点和训练模式下，使用梯度检查点函数计算层输出
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_state,
                    attn_mask,
                    head_mask[i],
                    output_attentions,
                )
            else:
                # 否则直接调用层模块计算层输出
                layer_outputs = layer_module(
                    hidden_state,
                    attn_mask,
                    head_mask[i],
                    output_attentions,
                )

            # 更新隐藏状态为当前层输出的最后一个值
            hidden_state = layer_outputs[-1]

            # 如果需要输出注意力权重，则将当前层的注意力权重添加到存储中
            if output_attentions:
                if len(layer_outputs) != 2:
                    raise ValueError(f"The length of the layer_outputs should be 2, but it is {len(layer_outputs)}")

                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
            else:
                if len(layer_outputs) != 1:
                    raise ValueError(f"The length of the layer_outputs should be 1, but it is {len(layer_outputs)}")

        # 添加最后一层的隐藏状态到存储中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        # 如果不需要返回字典形式的结果，则返回非空的元组
        if not return_dict:
            return tuple(v for v in [hidden_state, all_hidden_states, all_attentions] if v is not None)
        # 否则返回包含最后隐藏状态、所有隐藏状态和注意力权重的字典形式结果
        return BaseModelOutput(
            last_hidden_state=hidden_state, hidden_states=all_hidden_states, attentions=all_attentions
        )
# 为编码器和任务特定模型提供接口
class DistilBertPreTrainedModel(PreTrainedModel):
    """
    一个处理权重初始化和一个简单接口用于下载和加载预训练模型的抽象类。
    """

    config_class = DistilBertConfig
    load_tf_weights = None
    base_model_prefix = "distilbert"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True

    def _init_weights(self, module: nn.Module):
        """初始化权重。"""
        if isinstance(module, nn.Linear):
            # 与 TF 版本略有不同，TF 版本使用截断正态分布进行初始化
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


DISTILBERT_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`DistilBertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

DISTILBERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列标记在词汇表中的索引。
            # 可以使用 [`AutoTokenizer`] 获取索引。参见 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`] 了解详情。
            # [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 遮蔽填充标记索引，避免在这些位置执行注意力计算。遮蔽值在 `[0, 1]` 之间：

            # - 1 表示 **未被遮蔽** 的标记，
            # - 0 表示 **被遮蔽** 的标记。

            # [什么是注意力遮蔽？](../glossary#attention-mask)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于屏蔽自注意力模块中选定头部的遮蔽。遮蔽值在 `[0, 1]` 之间：

            # - 1 表示头部 **未被遮蔽**，
            # - 0 表示头部 **被遮蔽**。

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选地，可以直接传递嵌入表示而不是传递 `input_ids`。如果您想要更多控制如何将 `input_ids` 索引转换为相关向量，
            # 则这很有用，而不是使用模型的内部嵌入查找矩阵。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回张量中的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多详细信息，请参见返回张量中的 `hidden_states`。
        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""
定义 DistilBERT 模型类，输出原始隐藏状态而不带任何特定的头部
"""
@add_start_docstrings(
    "The bare DistilBERT encoder/transformer outputting raw hidden-states without any specific head on top.",
    DISTILBERT_START_DOCSTRING,
)
class DistilBertModel(DistilBertPreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)

        # 初始化 Embeddings 层
        self.embeddings = Embeddings(config)  # Embeddings
        # 初始化 Transformer 层作为编码器
        self.transformer = Transformer(config)  # Encoder
        # 检查是否使用 Flash Attention 2
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        # 初始化权重并应用最终处理
        self.post_init()

    def get_position_embeddings(self) -> nn.Embedding:
        """
        返回位置嵌入
        """
        return self.embeddings.position_embeddings
    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        调整模型的位置嵌入矩阵大小，如果 `new_num_position_embeddings != config.max_position_embeddings`。

        参数:
            new_num_position_embeddings (`int`):
                新的位置嵌入矩阵的数量。如果位置嵌入是可学习的，增加大小将在末尾添加新初始化的向量，而减小大小将从末尾删除向量。
                如果位置嵌入不是可学习的（例如正弦位置嵌入），增加大小将在末尾按照位置编码算法添加正确的向量，而减小大小将从末尾删除向量。
        """
        num_position_embeds_diff = new_num_position_embeddings - self.config.max_position_embeddings

        # 如果长度保持不变，则无需调整大小
        if num_position_embeds_diff == 0:
            return

        logger.info(f"设置 `config.max_position_embeddings={new_num_position_embeddings}`...")
        self.config.max_position_embeddings = new_num_position_embeddings

        old_position_embeddings_weight = self.embeddings.position_embeddings.weight.clone()

        self.embeddings.position_embeddings = nn.Embedding(self.config.max_position_embeddings, self.config.dim)

        if self.config.sinusoidal_pos_embds:
            create_sinusoidal_embeddings(
                n_pos=self.config.max_position_embeddings, dim=self.config.dim, out=self.position_embeddings.weight
            )
        else:
            with torch.no_grad():
                if num_position_embeds_diff > 0:
                    self.embeddings.position_embeddings.weight[:-num_position_embeds_diff] = nn.Parameter(
                        old_position_embeddings_weight
                    )
                else:
                    self.embeddings.position_embeddings.weight = nn.Parameter(
                        old_position_embeddings_weight[:num_position_embeds_diff]
                    )
        # 将位置嵌入移动到正确的设备
        self.embeddings.position_embeddings.to(self.device)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        self.embeddings.word_embeddings = new_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[List[int]]):
        """
        剪枝模型的注意力头。heads_to_prune: {层号: 要在此层中剪枝的头列表} 参见基类 PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.transformer.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format("batch_size, num_choices"))
    # 添加代码示例文档字符串，包括检查点、输出类型和配置类
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义前向传播函数，接受多个输入参数并返回模型输出
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入的 token ID 张量，默认为 None
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码张量，默认为 None
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码张量，默认为 None
        inputs_embeds: Optional[torch.Tensor] = None,  # 嵌入输入张量，默认为 None
        output_attentions: Optional[bool] = None,  # 是否输出注意力，默认为 None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，默认为 None
        return_dict: Optional[bool] = None,  # 是否返回字典，默认为 None
    ) -> Union[BaseModelOutput, Tuple[torch.Tensor, ...]]:  # 返回值类型为模型输出或元组

        # 如果未指定输出注意力，则使用配置中的输出注意力设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定输出隐藏状态，则使用配置中的输出隐藏状态设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定返回字典，则使用配置中的返回字典设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果同时指定了 input_ids 和 inputs_embeds，则引发 ValueError
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        # 如果指定了 input_ids，则检查填充和无注意力掩码的警告，并获取输入形状
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        # 如果指定了 inputs_embeds，则获取输入形状
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 根据 input_ids 或 inputs_embeds 的设备获取设备信息
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # 如果需要，准备头部掩码
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # 获取嵌入层的输出，形状为 (bs, seq_length, dim)
        embeddings = self.embeddings(input_ids, inputs_embeds)

        # 如果使用了 flash attention 2，则根据条件设置注意力掩码
        if self._use_flash_attention_2:
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 如果未指定注意力掩码，则创建全为 1 的注意力掩码张量
            if attention_mask is None:
                attention_mask = torch.ones(input_shape, device=device)  # (bs, seq_length)

        # 返回变换器的输出，传入嵌入层输出、注意力掩码、头部掩码等参数
        return self.transformer(
            x=embeddings,
            attn_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
# 使用 DistilBert 模型，在其顶部添加一个用于掩码语言建模的头部
@add_start_docstrings(
    """DistilBert Model with a `masked language modeling` head on top.""",
    DISTILBERT_START_DOCSTRING,
)
class DistilBertForMaskedLM(DistilBertPreTrainedModel):
    # 定义共享权重的键
    _tied_weights_keys = ["vocab_projector.weight"]

    def __init__(self, config: PretrainedConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # 获取激活函数
        self.activation = get_activation(config.activation)

        # 初始化 DistilBert 模型
        self.distilbert = DistilBertModel(config)
        # 初始化词汇转换层
        self.vocab_transform = nn.Linear(config.dim, config.dim)
        # 初始化词汇层归一化
        self.vocab_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        # 初始化词汇投影层
        self.vocab_projector = nn.Linear(config.dim, config.vocab_size)

        # 初始化权重并应用最终处理
        self.post_init()

        # 定义掩码语言建模的损失函数
        self.mlm_loss_fct = nn.CrossEntropyLoss()

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings
        """
        # 返回位置嵌入
        return self.distilbert.get_position_embeddings()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end. If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the
                size will add correct vectors at the end following the position encoding algorithm, whereas reducing
                the size will remove vectors from the end.
        """
        # 调整模型的位置嵌入
        self.distilbert.resize_position_embeddings(new_num_position_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        # 返回输出词汇投影层
        return self.vocab_projector

    def set_output_embeddings(self, new_embeddings: nn.Module):
        # 设置新的输出词汇投影层
        self.vocab_projector = new_embeddings

    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format("batch_size, num_choices"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[MaskedLMOutput, Tuple[torch.Tensor, ...]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional`):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        # 确定是否返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 DistilBERT 模型进行前向传播
        dlbrt_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取隐藏状态
        hidden_states = dlbrt_output[0]  # (bs, seq_length, dim)
        # 将隐藏状态转换为预测 logits
        prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
        # 对 logits 应用激活函数
        prediction_logits = self.activation(prediction_logits)  # (bs, seq_length, dim)
        # 对 logits 应用层归一化
        prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        # 对 logits 应用投影层
        prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)

        mlm_loss = None
        # 如果存在标签，则计算 MLM 损失
        if labels is not None:
            mlm_loss = self.mlm_loss_fct(prediction_logits.view(-1, prediction_logits.size(-1)), labels.view(-1))

        # 如果不返回字典，则返回输出
        if not return_dict:
            output = (prediction_logits,) + dlbrt_output[1:]
            return ((mlm_loss,) + output) if mlm_loss is not None else output

        # 返回 MaskedLMOutput 对象
        return MaskedLMOutput(
            loss=mlm_loss,
            logits=prediction_logits,
            hidden_states=dlbrt_output.hidden_states,
            attentions=dlbrt_output.attentions,
        )
# 使用 DistilBert 模型进行序列分类/回归任务的模型转换器，顶部有一个线性层（在池化输出之上）用于 GLUE 任务
class DistilBertForSequenceClassification(DistilBertPreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 获取标签数量
        self.num_labels = config.num_labels
        self.config = config

        # 初始化 DistilBert 模型
        self.distilbert = DistilBertModel(config)
        # 初始化预分类器
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        # 初始化分类器
        self.classifier = nn.Linear(config.dim, config.num_labels)
        # 初始化 dropout 层
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_position_embeddings(self) -> nn.Embedding:
        """
        返回位置嵌入
        """
        return self.distilbert.get_position_embeddings()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        如果 `new_num_position_embeddings != config.max_position_embeddings`，则调整模型的位置嵌入。

        参数:
            new_num_position_embeddings (`int`):
                新的位置嵌入矩阵的数量。如果位置嵌入是可学习的，增加大小将在末尾添加新初始化的向量，而减小大小将从末尾删除向量。
                如果位置嵌入不是可学习的（例如正弦位置嵌入），增加大小将按照位置编码算法在末尾添加正确的向量，而减小大小将从末尾删除向量。
        """
        self.distilbert.resize_position_embeddings(new_num_position_embeddings)

    # 将注释添加到模型的前向方法
    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[SequenceClassifierOutput, Tuple[torch.Tensor, ...]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 确定是否返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取 DistilBERT 模型的输出
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)

        loss = None
        if labels is not None:
            # 确定问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型计算损失
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

        # 如果不返回字典，则返回输出元组
        if not return_dict:
            output = (logits,) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回 SequenceClassifierOutput 对象
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )
# 使用 DistilBert 模型进行抽取式问答任务的模型，包含一个用于计算 `span start logits` 和 `span end logits` 的线性层
class DistilBertForQuestionAnswering(DistilBertPreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # 初始化 DistilBert 模型
        self.distilbert = DistilBertModel(config)
        # 初始化用于问答任务的线性层
        self.qa_outputs = nn.Linear(config.dim, config.num_labels)
        # 如果标签数不为 2，则抛出异常
        if config.num_labels != 2:
            raise ValueError(f"config.num_labels should be 2, but it is {config.num_labels}")

        # 初始化 Dropout 层
        self.dropout = nn.Dropout(config.qa_dropout)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取位置嵌入
    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings
        """
        return self.distilbert.get_position_embeddings()

    # 调整位置嵌入的大小
    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end. If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the
                size will add correct vectors at the end following the position encoding algorithm, whereas reducing
                the size will remove vectors from the end.
        """
        self.distilbert.resize_position_embeddings(new_num_position_embeddings)

    # 前向传播方法
    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format("batch_size, num_choices"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,



# 使用 DistilBert 模型进行标记分类任务的模型，包含一个用于输出隐藏状态的线性层
class DistilBertForTokenClassification(DistilBertPreTrainedModel):
    # 初始化方法，接受一个预训练配置对象作为参数
    def __init__(self, config: PretrainedConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置类属性 num_labels 为配置对象的 num_labels 属性
        self.num_labels = config.num_labels

        # 创建一个 DistilBertModel 对象并赋值给属性 distilbert
        self.distilbert = DistilBertModel(config)
        # 创建一个丢弃层对象并赋值给属性 dropout
        self.dropout = nn.Dropout(config.dropout)
        # 创建一个线性层对象并赋值给属性 classifier
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 调用 post_init 方法进行权重初始化和最终处理
        # Initialize weights and apply final processing
        self.post_init()

    # 返回位置嵌入（位置编码）的方法
    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings
        """
        return self.distilbert.get_position_embeddings()

    # 调整位置嵌入（位置编码）的方法，根据输入的新的位置嵌入数量
    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end. If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the
                size will add correct vectors at the end following the position encoding algorithm, whereas reducing
                the size will remove vectors from the end.
        """
        # 调用 distilbert 对象的 resize_position_embeddings 方法来调整位置嵌入
        self.distilbert.resize_position_embeddings(new_num_position_embeddings)

    # 前向传播方法，接受多个输入参数并返回模型的预测结果
    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[TokenClassifierOutput, Tuple[torch.Tensor, ...]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 确保返回的结果是一个字典类型，如果未指定，则使用模型配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 DistilBERT 模型处理输入
        outputs = self.distilbert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取模型输出的序列表示
        sequence_output = outputs[0]

        # 对序列表示进行 dropout 处理
        sequence_output = self.dropout(sequence_output)
        # 使用分类器对序列表示进行分类，得到分类的 logits
        logits = self.classifier(sequence_output)

        loss = None
        # 如果存在标签，则计算损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # 计算交叉熵损失
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不需要返回字典，则返回元组形式的输出
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典，则返回 TokenClassifierOutput 对象
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用 DistilBERT 模型，在其顶部添加一个多项选择分类头部（一个线性层和 softmax），用于例如 RocStories/SWAG 任务
class DistilBertForMultipleChoice(DistilBertPreTrainedModel):
    # 初始化函数，接受一个配置参数
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)

        # 初始化 DistilBERT 模型
        self.distilbert = DistilBertModel(config)
        # 添加一个线性层处理器
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        # 添加一个线性分类器
        self.classifier = nn.Linear(config.dim, 1)
        # 添加一个 dropout 层
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取位置嵌入的函数
    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings
        """
        return self.distilbert.get_position_embeddings()

    # 调整位置嵌入的函数
    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`)
                The number of new position embeddings. If position embeddings are learned, increasing the size will add
                newly initialized vectors at the end, whereas reducing the size will remove vectors from the end. If
                position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the size will
                add correct vectors at the end following the position encoding algorithm, whereas reducing the size
                will remove vectors from the end.
        """
        self.distilbert.resize_position_embeddings(new_num_position_embeddings)

    # 前向传播函数
    @add_start_docstrings_to_model_forward(
        DISTILBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @replace_return_docstrings(output_type=MultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
```