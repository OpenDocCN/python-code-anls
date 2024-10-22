# `.\chatglm4v-9b\modeling_chatglm.py`

```py
# PyTorch GLM-4V 模型的文档字符串
""" PyTorch GLM-4V model. """
# 导入数学库
import math
# 导入系统库
import sys
# 导入 PyTorch 库
import torch
# 导入用于检查点的工具
import torch.utils.checkpoint
# 导入 PyTorch 的功能性模块
import torch.nn.functional as F
# 从 PyTorch 导入 nn 模块
from torch import nn
# 从 nn 模块导入多种损失函数
from torch.nn import CrossEntropyLoss, LayerNorm, MSELoss, BCEWithLogitsLoss
# 从 nn.utils 导入跳过初始化的工具
from torch.nn.utils import skip_init
# 导入类型提示相关的模块
from typing import Optional, Tuple, Union, List, Dict, Any

# 从 transformers 导入模型输出相关的类
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
# 从 transformers 导入预训练模型的基类
from transformers.modeling_utils import PreTrainedModel
# 从 transformers 导入日志记录和可用性检查
from transformers.utils import logging, is_torch_npu_available
# 从生成模块导入 logits 处理器
from transformers.generation.logits_process import LogitsProcessor
# 从生成工具导入生成相关的类
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerationConfig, ModelOutput

# 导入视觉模型
from .visual import EVA2CLIPModel
# 导入 ChatGLM 配置
from .configuration_chatglm import ChatGLMConfig

# 尝试导入 Flash Attention 相关工具
try:
    from transformers.utils import is_flash_attn_greater_or_equal_2_10, is_flash_attn_2_available

    # 如果 Flash Attention 2 可用，导入相关函数
    if is_flash_attn_2_available():
        from flash_attn import flash_attn_func, flash_attn_varlen_func
        from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
# 捕获导入异常
except:
    pass

# 设置 JIT 融合内核所需的标志
# 如果不是在 macOS 上并且不支持 NPU，则设置 JIT 配置
if sys.platform != 'darwin' and not is_torch_npu_available():
    torch._C._jit_set_profiling_mode(False)  # 禁用 JIT 轮廓模式
    torch._C._jit_set_profiling_executor(False)  # 禁用 JIT 轮廓执行器
    torch._C._jit_override_can_fuse_on_cpu(True)  # 允许在 CPU 上融合
    torch._C._jit_override_can_fuse_on_gpu(True)  # 允许在 GPU 上融合

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义语言和视觉的标记类型
LANGUAGE_TOKEN_TYPE = 0
VISION_TOKEN_TYPE = 1

# 定义文档检查点和配置
_CHECKPOINT_FOR_DOC = "THUDM/ChatGLM"
_CONFIG_FOR_DOC = "ChatGLMConfig"


# 默认初始化函数
def default_init(cls, *args, **kwargs):
    # 使用给定的参数初始化类
    return cls(*args, **kwargs)


# 定义无效分数的 logits 处理器
class InvalidScoreLogitsProcessor(LogitsProcessor):
    # 重写调用方法
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 检查分数是否存在 NaN 或 Inf
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            # 将分数置为零
            scores.zero_()
            # 设置特定索引的分数
            scores[..., 198] = 5e4
        # 返回处理后的分数
        return scores


# 定义前缀编码器
class PrefixEncoder(torch.nn.Module):
    """
    用于编码前缀的 torch.nn 模型
    输入形状: (batch-size, prefix-length)
    输出形状: (batch-size, prefix-length, 2*layers*hidden)
    """
    # 初始化方法，接受一个 ChatGLMConfig 配置对象
    def __init__(self, config: ChatGLMConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 从配置中获取前缀投影的设置
        self.prefix_projection = config.prefix_projection
        # 如果启用了前缀投影
        if self.prefix_projection:
            # 计算用于编码前缀的键值对的大小
            kv_size = config.num_layers * config.kv_channels * config.multi_query_group_num * 2
            # 创建嵌入层，输入大小为 pre_seq_len，输出大小为 kv_size
            self.embedding = torch.nn.Embedding(config.pre_seq_len, kv_size)
            # 创建一个包含两个线性层和一个 Tanh 激活函数的顺序网络
            self.trans = torch.nn.Sequential(
                # 第一层线性变换，输入大小为 kv_size，输出大小为 hidden_size
                torch.nn.Linear(kv_size, config.hidden_size),
                # 应用 Tanh 激活函数
                torch.nn.Tanh(),
                # 第二层线性变换，输入大小为 hidden_size，输出大小为 kv_size
                torch.nn.Linear(config.hidden_size, kv_size)
            )
        else:
            # 如果没有启用前缀投影，直接创建嵌入层
            self.embedding = torch.nn.Embedding(config.pre_seq_len,
                                                config.num_layers * config.kv_channels * config.multi_query_group_num * 2)

    # 前向传播方法，接受一个前缀张量
    def forward(self, prefix: torch.Tensor):
        # 如果启用了前缀投影
        if self.prefix_projection:
            # 将前缀张量通过嵌入层进行嵌入
            prefix_tokens = self.embedding(prefix)
            # 通过转换网络获取过去的键值对
            past_key_values = self.trans(prefix_tokens)
        else:
            # 如果没有前缀投影，直接通过嵌入层获取过去的键值对
            past_key_values = self.embedding(prefix)
        # 返回过去的键值对
        return past_key_values
# 定义一个函数用于沿最后一个维度拆分张量
def split_tensor_along_last_dim(
        tensor: torch.Tensor,  # 输入的张量
        num_partitions: int,  # 拆分张量的分区数
        contiguous_split_chunks: bool = False,  # 是否要求每个分块在内存中是连续的
) -> List[torch.Tensor]:  # 返回类型为张量列表
    """拆分张量沿其最后一个维度。

    参数：
        tensor: 输入张量。
        num_partitions: 拆分张量的分区数
        contiguous_split_chunks: 如果为 True，则使每个块在内存中连续。

    返回：
        张量列表
    """
    # 获取张量的最后维度索引
    last_dim = tensor.dim() - 1
    # 计算每个分区的大小
    last_dim_size = tensor.size()[last_dim] // num_partitions
    # 使用 torch.split 函数进行拆分
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # 注意：torch.split 默认不创建连续的张量
    if contiguous_split_chunks:  # 如果需要连续的分块
        # 返回每个分块的连续版本
        return tuple(chunk.contiguous() for chunk in tensor_list)

    # 返回拆分后的张量列表
    return tensor_list


# 定义一个旋转嵌入类，继承自 nn.Module
class RotaryEmbedding(nn.Module):
    # 初始化函数，设置参数
    def __init__(self, dim, rope_ratio=1, original_impl=False, device=None, dtype=None):
        super().__init__()  # 调用父类初始化
        # 计算反频率并在 buffer 中注册
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).to(dtype=dtype) / dim))
        self.register_buffer("inv_freq", inv_freq)  # 注册反频率
        self.dim = dim  # 保存维度信息
        self.original_impl = original_impl  # 保存原始实现标志
        self.rope_ratio = rope_ratio  # 保存旋转比例

    # 实现方法，根据序列长度和维度生成嵌入
    def impl(self, seq_length: int, dim: int, device: torch.device, dtype: torch.dtype):
        base = 10000 * self.rope_ratio  # 计算基础值
        # 计算反频率
        inv_freq = 1.0 / (
                base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        # 创建序列的张量
        seq = torch.arange(seq_length, device=inv_freq.device, dtype=torch.float32)
        # 计算频率的外积
        freqs = torch.outer(seq, inv_freq)
        # 第一个部分是偶数向量分量，第二个部分是奇数向量分量，
        # 维度大小为 2 * dim
        emb = torch.cat((freqs, freqs), dim=-1)  # 将频率拼接
        return emb  # 返回嵌入

    # 前向实现函数，定义前向传播的逻辑
    def forward_impl(
            self, seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000
    ):
        """增强的 Transformer，带有旋转位置嵌入。

        来源于: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
        transformers/rope/__init__.py。MIT 许可证:
        https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license。
        """
        # 计算旋转嵌入的基础 $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        base = base * self.rope_ratio
        # 计算每个位置的频率 $\theta_i$，用于位置嵌入
        theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=torch.float, device=device) / n_elem))

        # 创建位置索引 `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, dtype=torch.float, device=device)

        # 计算位置索引与频率的外积
        idx_theta = torch.outer(seq_idx, theta).float()

        # 堆叠余弦和正弦值，形成位置嵌入的缓存
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

        # 处理数据类型，模拟 complex32 的行为，避免结果不同
        if dtype in (torch.float16, torch.bfloat16, torch.int8):
            # 将缓存转换为 bfloat16 或 half，根据数据类型
            cache = cache.bfloat16() if dtype == torch.bfloat16 else cache.half()
        # 返回计算得到的缓存
        return cache

    def forward(self, max_seq_len, offset=0):
        # 如果使用原始实现，则调用原始的前向传播方法
        if self.original_impl:
            return self.forward_impl(
                max_seq_len, self.dim, dtype=self.inv_freq.dtype, device=self.inv_freq.device
            )
        # 否则调用自定义实现的前向传播方法
        else:
            return self.impl(max_seq_len, self.dim, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
# 使用 Torch JIT 编译器将此函数编译为高效的 Torch 脚本
@torch.jit.script
def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # x: [b, np, sq, hn]，其中 b 是批量大小，np 是序列数，sq 是序列长度，hn 是隐藏维度
    b, np, sq, hn = x.size(0), x.size(1), x.size(2), x.size(3)
    # 计算旋转维度，rope_cache 的最后一维的大小乘以 2
    rot_dim = rope_cache.shape[-2] * 2
    # 将 x 分为旋转部分和其他部分
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # 截断 rope_cache 以支持可变大小
    rope_cache = rope_cache[:, :sq]
    # 将 x 重塑为 [b, np, sq, rot_dim / 2, 2] 的形状
    xshaped = x.reshape(b, np, sq, rot_dim // 2, 2)
    # 将 rope_cache 视图重塑为 [b, 1, sq, xshaped 的最后一维, 2]
    rope_cache = rope_cache.view(-1, 1, sq, xshaped.size(3), 2)
    # 计算输出，使用旋转位置编码的公式
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    # 将输出展平，去掉最后一个维度的 3 维信息
    x_out2 = x_out2.flatten(3)
    # 将处理后的输出与未处理部分连接，沿最后一个维度拼接
    return torch.cat((x_out2, x_pass), dim=-1)


# 定义 RMSNorm 类，继承自 torch.nn.Module
class RMSNorm(torch.nn.Module):
    # 初始化 RMSNorm 类的构造函数
    def __init__(self, normalized_shape, eps=1e-5, device=None, dtype=None, **kwargs):
        # 调用父类构造函数
        super().__init__()
        # 创建可学习的权重参数，形状为 normalized_shape
        self.weight = torch.nn.Parameter(torch.empty(normalized_shape, device=device, dtype=dtype))
        # 设置 epsilon 值以避免除以零
        self.eps = eps

    # 定义前向传播方法
    def forward(self, hidden_states: torch.Tensor):
        # 获取输入的 dtype
        input_dtype = hidden_states.dtype
        # 计算方差，取平方后求均值，保持维度
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        # 规范化 hidden_states，乘以方差的平方根的倒数
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        # 返回加权的隐藏状态，转换回原始的 dtype
        return (self.weight * hidden_states).to(input_dtype)


# 定义 CoreAttention 类，继承自 torch.nn.Module
class CoreAttention(torch.nn.Module):
    # 初始化 CoreAttention 类的构造函数
    def __init__(self, config: ChatGLMConfig, layer_number):
        # 调用父类构造函数
        super(CoreAttention, self).__init__()

        # 根据配置设置是否应用查询键层的缩放
        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        # 设置注意力 softmax 的数据类型为 FP32
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        # 如果应用查询键层缩放，则强制使用 FP32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        # 确保层号至少为 1
        self.layer_number = max(1, layer_number)

        # 计算投影大小
        projection_size = config.kv_channels * config.num_attention_heads

        # 每个注意力头和每个分区的大小
        self.hidden_size_per_partition = projection_size
        # 每个注意力头的隐藏维度
        self.hidden_size_per_attention_head = projection_size // config.num_attention_heads
        # 每个分区的注意力头数量
        self.num_attention_heads_per_partition = config.num_attention_heads

        coeff = None
        # 计算规范化因子，使用每个注意力头的隐藏大小的平方根
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        # 如果应用查询键层缩放，则调整规范化因子
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff
        # 存储缩放系数
        self.coeff = coeff

        # 初始化注意力 dropout
        self.attention_dropout = torch.nn.Dropout(config.attention_dropout)

# 定义 SdpaAttention 类，继承自 CoreAttention
class SdpaAttention(CoreAttention):
    # 定义前向传播函数，接受查询层、键层、值层和注意力掩码作为输入
    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        # 如果没有注意力掩码且查询层的最后一维与键层的最后一维相同
        if attention_mask is None and query_layer.shape[2] == key_layer.shape[2]:
            # 使用缩放点积注意力计算上下文层，设置为因果模式，并根据训练状态设置丢弃率
            context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer,
                                                                                 is_causal=True,
                                                                                 dropout_p=self.config.attention_dropout if self.training else 0.0)
        else:
            # 如果存在注意力掩码
            if attention_mask is not None:
                # 反转注意力掩码
                attention_mask = ~attention_mask
            # 使用缩放点积注意力计算上下文层，传入注意力掩码和丢弃率
            context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer,
                                                                                 attention_mask,
                                                                                 dropout_p=self.config.attention_dropout if self.training else 0.0)
        # 转置上下文层的第1维和第2维，并确保内存连续
        context_layer = context_layer.transpose(1, 2).contiguous()
        # 生成新的上下文层形状，将最后两个维度替换为分区后的隐藏层大小
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        # 按新的形状重塑上下文层
        context_layer = context_layer.reshape(*new_context_layer_shape)
        # 返回处理后的上下文层
        return context_layer
# 获取未填充的注意力数据
def _get_unpad_data(attention_mask):
    # 计算每个样本的序列长度，使用 int32 类型
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    # 找到注意力掩码中非零的索引，并扁平化
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    # 计算批次中最长的序列长度
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    # 计算累计序列长度，并在开头填充一个零
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    # 返回索引、累计序列长度和最大序列长度
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# 从 transformers.models.llama.modeling_llama.LlamaFlashAttention2 复制而来
class FlashAttention2(CoreAttention):
    def __init__(self, *args, **kwargs):
        # 初始化基类
        super().__init__(*args, **kwargs)
        # 检查 Flash Attention 的版本以决定是否使用左上角掩码
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(self, query_states, key_states, value_states, attention_mask):
        # 转置查询状态以符合 Flash Attention 的要求
        query_states = query_states.transpose(1, 2)
        # 转置键状态以符合 Flash Attention 的要求
        key_states = key_states.transpose(1, 2)
        # 转置值状态以符合 Flash Attention 的要求
        value_states = value_states.transpose(1, 2)
        # 获取批次大小和查询长度
        batch_size, query_length = query_states.shape[:2]
        # 根据 Flash Attention 的配置决定 causal 标志
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: 一旦 Flash Attention 对 RoCm 的版本提升到 2.1，移除 `query_length != 1` 的检查
            causal = self.is_causal and query_length != 1
        # 设置 dropout 概率，根据训练状态决定
        dropout = self.config.attention_dropout if self.training else 0.0
        # 如果存在注意力掩码，则进行处理
        if attention_mask is not None:
            # 调用输入处理函数以获取未填充的输入
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            # 解包累计序列长度
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            # 解包最大序列长度
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            # 调用 Flash Attention 函数进行计算，使用未填充的状态
            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=None,
                causal=causal,
            )

            # 将未填充的注意力输出填充为最终输出
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            # 如果没有注意力掩码，直接计算注意力输出
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=None, causal=causal
            )
        # 重塑输出的形状以符合批次大小和查询长度
        attn_output = attn_output.reshape(batch_size, query_length, self.hidden_size_per_partition).contiguous()
        # 返回最终的注意力输出
        return attn_output
    # 更新输入的查询层、键层和值层，并处理注意力掩码和查询长度
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        # 获取未填充数据的索引、当前序列长度和批次中最大序列长度
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        # 获取键层的批次大小、键值序列长度、键值头数和头维度
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape
    
        # 根据索引调整键层的形状
        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        # 根据索引调整值层的形状
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        # 如果查询长度等于键值序列长度
        if query_length == kv_seq_len:
            # 根据索引调整查询层的形状
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_attention_heads_per_partition, head_dim),
                indices_k
            )
            # 设置当前序列长度和最大序列长度为键的值
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        # 如果查询长度为1
        elif query_length == 1:
            # 最大序列长度设为1
            max_seqlen_in_batch_q = 1
            # 生成当前序列长度的范围
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # 这里有一个内存拷贝，性能较差。
            # 获取索引
            indices_q = cu_seqlens_q[:-1]
            # 压缩查询层的维度
            query_layer = query_layer.squeeze(1)
        else:
            # 根据查询长度调整注意力掩码（假设是左填充）
            attention_mask = attention_mask[:, -query_length:]
            # 去填充输入并获取相应的查询层和索引等
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)
    
        # 返回更新后的查询层、键层、值层及其相关信息
        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
# 定义核心注意力类的字典映射
CORE_ATTENTION_CLASSES = {
    "eager": CoreAttention,  # 将 "eager" 映射到 CoreAttention 类
    "sdpa": SdpaAttention,   # 将 "sdpa" 映射到 SdpaAttention 类
    "flash_attention_2": FlashAttention2  # 将 "flash_attention_2" 映射到 FlashAttention2 类
}

# 定义自注意力类，继承自 PyTorch 的模块
class SelfAttention(torch.nn.Module):
    """并行自注意力层抽象类。

    自注意力层接受大小为 [s, b, h] 的输入并返回相同大小的输出。
    """

    # 初始化方法
    def __init__(self, config: ChatGLMConfig, layer_number, device=None):
        super(SelfAttention, self).__init__()  # 调用父类初始化方法
        self.layer_number = max(1, layer_number)  # 确保层编号至少为1

        # 计算投影大小
        self.projection_size = config.kv_channels * config.num_attention_heads

        # 每个注意力头和每个分区的值
        self.hidden_size_per_attention_head = self.projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads  # 每个分区的注意力头数量

        self.multi_query_attention = config.multi_query_attention  # 是否使用多查询注意力
        self.qkv_hidden_size = 3 * self.projection_size  # QKV的隐藏大小
        self.original_rope = config.original_rope  # 原始旋转位置编码配置
        if self.multi_query_attention:  # 如果使用多查询注意力
            self.num_multi_query_groups_per_partition = config.multi_query_group_num  # 每个分区的多查询组数量
            self.qkv_hidden_size = (  # 更新QKV的隐藏大小
                    self.projection_size + 2 * self.hidden_size_per_attention_head * config.multi_query_group_num
            )
        # 定义线性层以计算QKV
        self.query_key_value = nn.Linear(config.hidden_size, self.qkv_hidden_size,
                                         bias=config.add_bias_linear or config.add_qkv_bias,
                                         device=device, **_config_to_kwargs(config)
                                         )

        # 实例化核心注意力
        self.core_attention = CoreAttention(config, self.layer_number)

        # 定义输出层
        self.dense = nn.Linear(self.projection_size, config.hidden_size, bias=config.add_bias_linear,
                               device=device, **_config_to_kwargs(config)
                               )

    # 分配内存的方法
    def _allocate_memory(self, inference_max_sequence_len, batch_size, device=None, dtype=None):
        if self.multi_query_attention:  # 根据是否使用多查询注意力设置头数
            num_attention_heads = self.num_multi_query_groups_per_partition
        else:
            num_attention_heads = self.num_attention_heads_per_partition  # 设置为每个分区的注意力头数量
        return torch.empty(  # 返回空的张量用于存储注意力值
            inference_max_sequence_len,
            batch_size,
            num_attention_heads,
            self.hidden_size_per_attention_head,
            dtype=dtype,
            device=device,
        )

    # 前向传播方法
    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True
# 定义配置转化为关键字参数的方法
def _config_to_kwargs(args):
    common_kwargs = {
        "dtype": args.torch_dtype,  # 将 PyTorch 数据类型作为关键字参数
    }
    return common_kwargs

# 定义多层感知器类，继承自 PyTorch 的模块
class MLP(torch.nn.Module):
    """多层感知器。

    MLP 将输入的隐藏状态 h 投影到 4*h 的隐藏维度，进行非线性变换，然后将状态投影回 h 的隐藏维度。
    """
    # 初始化 MLP 类，接受配置和可选设备参数
    def __init__(self, config: ChatGLMConfig, device=None):
        # 调用父类的初始化方法
        super(MLP, self).__init__()
    
        # 设置是否添加线性层的偏置
        self.add_bias = config.add_bias_linear
    
        # 创建一个线性层，将输入从隐层大小映射到 4h
        # 使用 SWIGLU 时，输出宽度加倍，详见相关文献
        self.dense_h_to_4h = nn.Linear(
            config.hidden_size,  # 输入特征数
            config.ffn_hidden_size * 2,  # 输出特征数
            bias=self.add_bias,  # 是否使用偏置
            device=device,  # 指定设备
            **_config_to_kwargs(config)  # 其他配置参数
        )
    
        # 定义 SWIGLU 激活函数
        def swiglu(x):
            # 将输入张量分成两部分
            x = torch.chunk(x, 2, dim=-1)
            # 返回激活函数的输出
            return F.silu(x[0]) * x[1]
    
        # 设置激活函数为 SWIGLU
        self.activation_func = swiglu
    
        # 创建一个线性层，将 4h 的输出映射回隐层大小
        self.dense_4h_to_h = nn.Linear(
            config.ffn_hidden_size,  # 输入特征数
            config.hidden_size,  # 输出特征数
            bias=self.add_bias,  # 是否使用偏置
            device=device,  # 指定设备
            **_config_to_kwargs(config)  # 其他配置参数
        )
    
    # 前向传播方法，处理隐藏状态
    def forward(self, hidden_states):
        # 将隐藏状态通过第一层线性变换
        # 输出形状为 [s, b, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        # 应用 SWIGLU 激活函数
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # 将激活后的结果通过第二层线性变换
        # 输出形状为 [s, b, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        # 返回最终输出
        return output
# 定义一个单一的变换器层，继承自 PyTorch 的 Module 类
class GLMBlock(torch.nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    # 初始化方法，设置层配置、层号和设备
    def __init__(self, config: ChatGLMConfig, layer_number, device=None):
        # 调用父类构造函数
        super(GLMBlock, self).__init__()
        # 保存层号
        self.layer_number = layer_number

        # 从配置中获取是否在层归一化后应用残差连接
        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm

        # 从配置中获取是否使用32位浮点的残差连接
        self.fp32_residual_connection = config.fp32_residual_connection

        # 根据配置选择层归一化函数
        LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
        # 对输入数据应用层归一化
        self.input_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon, device=device,
                                             dtype=config.torch_dtype)

        # 自注意力层
        self.self_attention = SelfAttention(config, layer_number, device=device)
        # 隐藏层的丢弃率
        self.hidden_dropout = config.hidden_dropout

        # 在注意力输出后应用层归一化
        self.post_attention_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon, device=device,
                                                      dtype=config.torch_dtype)

        # 多层感知机
        self.mlp = MLP(config, device=device)

    # 前向传播方法
    def forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True,
    ):
        # hidden_states: [s, b, h]

        # 在变换器层开始应用层归一化
        layernorm_output = self.input_layernorm(hidden_states)
        # 进行自注意力计算
        attention_output, kv_cache = self.self_attention(
            layernorm_output,
            attention_mask,
            rotary_pos_emb,
            kv_cache=kv_cache,
            use_cache=use_cache
        )

        # 残差连接
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # 应用丢弃，准备进行层归一化的输入
        layernorm_input = torch.nn.functional.dropout(attention_output, p=self.hidden_dropout, training=self.training)
        layernorm_input = residual + layernorm_input

        # 自注意力后的层归一化
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # 通过多层感知机计算输出
        mlp_output = self.mlp(layernorm_output)

        # 第二次残差连接
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        # 应用丢弃并计算最终输出
        output = torch.nn.functional.dropout(mlp_output, p=self.hidden_dropout, training=self.training)
        output = residual + output

        # 返回输出和键值缓存
        return output, kv_cache


# 定义变换器类，继承自 PyTorch 的 Module 类
class GLMTransformer(torch.nn.Module):
    """Transformer class."""
    # 初始化方法，接收配置和设备参数
        def __init__(self, config: ChatGLMConfig, device=None):
            # 调用父类的初始化方法
            super(GLMTransformer, self).__init__()
    
            # 设置 FP32 残差连接的配置
            self.fp32_residual_connection = config.fp32_residual_connection
            # 设置后层归一化的配置
            self.post_layer_norm = config.post_layer_norm
    
            # 获取层数
            # Number of layers.
            self.num_layers = config.num_layers
    
            # 定义构建层的方法
            # Transformer layers.
            def build_layer(layer_number):
                # 创建 GLMBlock 层实例
                return GLMBlock(config, layer_number, device=device)
    
            # 构建多个层并放入 ModuleList
            self.layers = torch.nn.ModuleList([build_layer(i + 1) for i in range(self.num_layers)])
    
            # 如果需要后层归一化
            if self.post_layer_norm:
                # 根据配置选择层归一化类型
                LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
                # 创建最终的层归一化实例，作为输出前的归一化
                # Final layer norm before output.
                self.final_layernorm = LayerNormFunc(config.hidden_size, eps=config.layernorm_epsilon, device=device,
                                                     dtype=config.torch_dtype)
    
            # 初始化梯度检查点标志为 False
            self.gradient_checkpointing = False
    
        # 获取指定层的方法
        def _get_layer(self, layer_number):
            # 返回对应编号的层
            return self.layers[layer_number]
    
        # 前向传播方法
        def forward(
                # 输入的隐藏状态
                self, hidden_states, 
                # 注意力掩码
                attention_mask, 
                # 旋转位置嵌入
                rotary_pos_emb, 
                # 可选的键值缓存
                kv_caches=None,
                # 是否使用缓存的标志，默认 True
                use_cache: Optional[bool] = True,
                # 是否输出隐藏状态的标志，默认 False
                output_hidden_states: Optional[bool] = False,
    ):
        # 如果 kv_caches 为空，则为每层初始化为 None
        if not kv_caches:
            kv_caches = [None for _ in range(self.num_layers)]
        # 如果使用缓存，则初始化 presents 为空元组，否则为 None
        presents = () if use_cache else None
        # 如果开启梯度检查点并处于训练模式
        if self.gradient_checkpointing and self.training:
            # 如果使用缓存，则发出警告并禁用缓存
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # 初始化所有自注意力的集合为 None
        all_self_attentions = None
        # 如果需要输出隐藏状态，则初始化为一个空元组，否则为 None
        all_hidden_states = () if output_hidden_states else None
        # 遍历每一层
        for index in range(self.num_layers):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到所有隐藏状态中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层
            layer = self._get_layer(index)
            # 如果开启梯度检查点并处于训练模式
            if self.gradient_checkpointing and self.training:
                # 使用检查点函数来计算当前层的输出
                layer_ret = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_caches[index],
                    use_cache,
                    use_reentrant=False
                )
            else:
                # 直接调用当前层计算输出
                layer_ret = layer(
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                    kv_cache=kv_caches[index],
                    use_cache=use_cache
                )
            # 解包当前层的输出和缓存
            hidden_states, kv_cache = layer_ret
            # 如果使用缓存，则将当前缓存添加到 presents 中
            if use_cache:
                presents = presents + (kv_cache,)

        # 如果需要输出隐藏状态，则将最后的隐藏状态添加到所有隐藏状态中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 最终的层归一化
        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        # 返回隐藏状态、缓存、所有隐藏状态和所有自注意力
        return hidden_states, presents, all_hidden_states, all_self_attentions
# 定义一个抽象类，处理权重初始化和预训练模型的下载与加载接口
class ChatGLMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    # 指示模型是否可以并行化
    is_parallelizable = False
    # 指示模型是否支持梯度检查点
    supports_gradient_checkpointing = True
    # 配置类为 ChatGLMConfig
    config_class = ChatGLMConfig
    # 基础模型前缀为 "transformer"
    base_model_prefix = "transformer"
    # 不可分割的模块列表
    _no_split_modules = ["GLMBlock"]
    # 支持 flash attention 2
    _supports_flash_attn_2 = True
    # 支持 SDPA
    _supports_sdpa = True

    # 初始化权重的方法
    def _init_weights(self, module: nn.Module):
        """Initialize the weights."""
        return

    # 获取输入的掩码
    def get_masks(self, input_embeds, past_key_values, padding_mask=None):
        # 获取批大小、序列长度和嵌入维度
        batch_size, seq_length, embed_size = input_embeds.shape
        # 创建全1的注意力掩码
        full_attention_mask = torch.ones(batch_size, seq_length, seq_length, device=input_embeds.device)
        # 变为下三角矩阵
        full_attention_mask.tril_()
        # 初始化过去的长度
        past_length = 0
        # 如果有过去的键值对，获取过去的长度
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]
        # 如果过去的长度存在，拼接注意力掩码
        if past_length:
            full_attention_mask = torch.cat((torch.ones(batch_size, seq_length, past_length,
                                                        device=input_embeds.device), full_attention_mask), dim=-1)
        # 如果有填充掩码，进行相应的操作
        if padding_mask is not None:
            full_attention_mask = full_attention_mask * padding_mask.unsqueeze(1)
        # 如果没有过去的长度且有填充掩码，调整全注意力掩码
        if not past_length and padding_mask is not None:
            full_attention_mask -= padding_mask.unsqueeze(-1) - 1
        # 将注意力掩码转为布尔类型
        full_attention_mask = (full_attention_mask < 0.5).bool()
        # 增加维度以适应后续操作
        full_attention_mask.unsqueeze_(1)
        # 返回最终的注意力掩码
        return full_attention_mask

    # 获取位置 ID
    def get_position_ids(self, input_ids, device):
        # 获取批大小和序列长度
        batch_size, seq_length = input_ids.shape
        # 创建位置 ID 并扩展到批大小
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
        return position_ids

    # 获取多模态位置 ID
    def get_multimodal_position_ids(self, input_ids, device):
        # 获取批大小和序列长度
        batch_size, seq_length = input_ids.shape
        # 创建位置 ID 并扩展到批大小
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)

# 定义嵌入类，继承自 torch.nn.Module
class Embedding(torch.nn.Module):
    """Language model embeddings."""

    # 初始化方法
    def __init__(self, config: ChatGLMConfig, device=None):
        super(Embedding, self).__init__()

        # 获取隐藏层大小
        self.hidden_size = config.hidden_size
        # 创建单词嵌入层（并行）
        self.word_embeddings = nn.Embedding(
            config.padded_vocab_size,
            self.hidden_size,
            dtype=config.torch_dtype,
            device=device
        )
        # 是否使用 fp32 残差连接
        self.fp32_residual_connection = config.fp32_residual_connection

    # 前向传播方法
    def forward(self, input_ids):
        # 获取单词嵌入
        words_embeddings = self.word_embeddings(input_ids)
        # 设置嵌入值
        embeddings = words_embeddings
        # 如果设置了 fp32 残差连接，将嵌入转换为浮点型
        if self.fp32_residual_connection:
            embeddings = embeddings.float()
        # 返回嵌入
        return embeddings


# 检查图像列表是否为空
def is_empty(images_list: Optional[List[List[torch.Tensor]]]):
    # 检查 images_list 是否为 None 或者为空列表
    if images_list is None or len(images_list) == 0:
        # 如果是，返回 True
        return True
    # 遍历 images_list 中的每个 image_list
    for image_list in images_list:
        # 如果 image_list 不是 None
        if image_list is not None:
            # 返回 False，表示存在有效的 image_list
            return False
    # 如果所有 image_list 都是 None，返回 True
    return True
# 定义 ChatGLMModel 类，继承自 ChatGLMPreTrainedModel
class ChatGLMModel(ChatGLMPreTrainedModel):
    # 初始化方法，接受配置、设备和空初始化标志
    def __init__(self, config: ChatGLMConfig, device=None, empty_init=True):
        # 调用父类的初始化方法，传入配置
        super().__init__(config)
        # 根据空初始化标志选择初始化方法
        if empty_init:
            init_method = skip_init
        else:
            init_method = default_init
        # 初始化关键字参数字典
        init_kwargs = {}
        # 如果设备不为 None，将其加入初始化参数
        if device is not None:
            init_kwargs["device"] = device
        # 使用初始化方法创建嵌入层
        self.embedding = init_method(Embedding, config, **init_kwargs)
        # 获取层数配置
        self.num_layers = config.num_layers
        # 获取多查询组数配置
        self.multi_query_group_num = config.multi_query_group_num
        # 获取 KV 通道数配置
        self.kv_channels = config.kv_channels

        # 旋转位置嵌入
        self.seq_length = config.seq_length
        # 根据注意力头数或 KV 通道数计算旋转维度
        rotary_dim = (
            config.hidden_size // config.num_attention_heads if config.kv_channels is None else config.kv_channels
        )

        # 创建旋转位置嵌入对象
        self.rotary_pos_emb = RotaryEmbedding(rotary_dim // 2, rope_ratio=config.rope_ratio,
                                              original_impl=config.original_rope,
                                              device=device, dtype=config.torch_dtype)
        # 使用初始化方法创建 GLMTransformer 编码器
        self.encoder = init_method(GLMTransformer, config, **init_kwargs)
        # 使用初始化方法创建输出层
        self.output_layer = init_method(nn.Linear, config.hidden_size, config.padded_vocab_size, bias=False,
                                        dtype=config.torch_dtype, **init_kwargs)
        # 获取预序列长度配置
        self.pre_seq_len = config.pre_seq_len
        # 获取前缀投影配置
        self.prefix_projection = config.prefix_projection
        # 如果预序列长度不为 None
        if self.pre_seq_len is not None:
            # 将所有参数的 requires_grad 设置为 False
            for param in self.parameters():
                param.requires_grad = False
            # 创建前缀令牌的张量
            self.prefix_tokens = torch.arange(self.pre_seq_len).long()
            # 创建前缀编码器
            self.prefix_encoder = PrefixEncoder(config)
            # 初始化 Dropout 层
            self.dropout = torch.nn.Dropout(0.1)

        # 创建视觉模型
        self.vision = EVA2CLIPModel(config)

    # 获取输入嵌入的方法
    def get_input_embeddings(self):
        return self.embedding.word_embeddings

    # 设置输入嵌入的方法
    def set_input_embeddings(self, value):
        self.embedding.word_embeddings = value

    # 获取提示的方法，接受批大小、设备和数据类型
    def get_prompt(self, batch_size, device, dtype=torch.half):
        # 扩展前缀令牌的维度以匹配批大小
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
        # 通过前缀编码器处理前缀令牌并转换数据类型
        past_key_values = self.prefix_encoder(prefix_tokens).type(dtype)
        # 重新排列过去的关键值的维度
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.pre_seq_len,
            self.num_layers * 2,
            self.multi_query_group_num,
            self.kv_channels
        )
        # 应用 Dropout 层
        past_key_values = self.dropout(past_key_values)
        # 调整维度顺序并分割
        past_key_values = past_key_values.permute([2, 1, 0, 3, 4]).split(2)
        # 返回处理后的过去关键值
        return past_key_values
    # 定义一个前向传播函数，接受多个输入参数
        def forward(
                # 输入 ID，类型为长整型张量，默认为 None
                self,
                input_ids: torch.LongTensor = None,
                # 输入图像，类型为张量，默认为 None
                images: torch.Tensor = None,
                # 位置 ID，类型为可选张量，默认为 None
                position_ids: Optional[torch.Tensor] = None,
                # 注意力掩码，类型为可选布尔张量，默认为 None
                attention_mask: Optional[torch.BoolTensor] = None,
                # 完整注意力掩码，类型为可选布尔张量，默认为 None
                full_attention_mask: Optional[torch.BoolTensor] = None,
                # 过去的键值对，类型为可选元组，默认为 None
                past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
                # 输入嵌入，类型为可选张量，默认为 None
                inputs_embeds: Optional[torch.Tensor] = None,
                # 是否使用缓存，类型为可选布尔值，默认为 None
                use_cache: Optional[bool] = None,
                # 是否输出隐藏状态，类型为可选布尔值，默认为 None
                output_hidden_states: Optional[bool] = None,
                # 是否以字典格式返回结果，类型为可选布尔值，默认为 None
                return_dict: Optional[bool] = None,
# 将历史对话转换为提示字符串，包含用户和助手的对话内容
def _history_to_prompt(history, query):
    # 初始化提示字符串为空
    prompt = ''
    # 标记是否已有历史查询
    flag = False
    # 遍历历史对话，索引和内容
    for i, (old_query, response) in enumerate(history):
        # 添加用户查询和助手响应到提示中，依据标记决定用户标签的添加
        prompt += ('<|user|>' if flag else '') + old_query + "<|assistant|>" + response + "<|endoftext|>"
        # 更新标记为 True，表示已有查询
        flag = True
    # 添加最新查询到提示中，依据标记决定用户标签的添加
    prompt += '{}{}<|assistant|>'.format('<|user|>' if flag else '', query)
    # 返回最终的提示字符串
    return prompt


# 自定义的条件生成模型类，继承自预训练模型
class ChatGLMForConditionalGeneration(ChatGLMPreTrainedModel):
    # 初始化模型，设置配置和其他参数
    def __init__(self, config: ChatGLMConfig, empty_init=True, device=None):
        # 调用父类的初始化方法
        super().__init__(config)

        # 设置最大序列长度
        self.max_sequence_length = config.max_length
        # 初始化变换模型
        self.transformer = ChatGLMModel(config, empty_init=empty_init, device=device)
        # 保存配置对象
        self.config = config

    # 更新生成过程中的模型参数
    def _update_model_kwargs_for_generation(
            self,
            outputs: ModelOutput,
            model_kwargs: Dict[str, Any],
            is_encoder_decoder: bool = False,
    ) -> Dict[str, Any]:
        # 从模型输出提取过去的键值对
        cache_name, cache = self._extract_past_from_model_output(outputs)
        # 将缓存添加到模型参数中
        model_kwargs[cache_name] = cache

        # 更新注意力掩码
        if "attention_mask" in model_kwargs:
            # 获取当前注意力掩码
            attention_mask = model_kwargs["attention_mask"]
            # 连接新创建的掩码，将其追加到当前掩码的末尾
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        # 更新位置 ID
        if "position_ids" in model_kwargs:
            # 获取当前的位置 ID
            position_ids = model_kwargs["position_ids"]
            # 复制最后一个位置 ID，并加 1
            new_position_id = position_ids[..., -1:].clone()
            new_position_id += 1
            # 将新的位置 ID 追加到当前的位置 ID
            model_kwargs["position_ids"] = torch.cat(
                [position_ids, new_position_id], dim=-1
            )

        # 标记为非首次前向传递
        model_kwargs["is_first_forward"] = False
        # 返回更新后的模型参数
        return model_kwargs

    # 准备生成所需的输入
    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,
            images: Optional[torch.Tensor] = None,
            past_key_values: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            is_first_forward: bool = True,
            **kwargs
    ) -> dict:  # 定义函数返回类型为字典
        # 如果 past 不为 None，只处理输入 ID 的最后一个 token
        if position_ids is None:  # 如果没有提供位置 ID
            # 获取输入 ID 的位置 ID，设备与输入 ID 相同
            position_ids = self.get_position_ids(input_ids, device=input_ids.device)  
        if attention_mask is not None:  # 如果提供了注意力掩码
            # 从配置中获取图像的大小
            image_size: int = self.config.vision_config['image_size']  
            # 从配置中获取补丁的大小
            patch_size: int = self.config.vision_config['patch_size']  
            # 计算图像中补丁的数量
            num_patches = (image_size // patch_size // 2) ** 2  
            new_attention_masks = []  # 初始化新的注意力掩码列表

            # 如果没有图像，使用默认的 ID
            eoi_token_pos = 6  # 结束 token 的位置
            boi_token_pos = 4  # 开始 token 的位置

            # 遍历输入 ID 的每个 token
            for i in range(len(input_ids)):  
                # 将当前输入 ID 转换为列表
                input_id = input_ids[i].tolist()  
                # 如果图像不为空，获取 BOI 和 EOI token 的位置
                if not is_empty(images):  
                    boi_token_pos, eoi_token_pos = input_id.index(self.config.boi_token_id), input_id.index(
                        self.config.eoi_token_id)  
                # 确保 EOI 和 BOI token 之间的距离为 2
                assert eoi_token_pos - boi_token_pos == 2  
                # 生成新的注意力掩码并添加到列表
                new_attention_masks.append(torch.cat(
                    (attention_mask[i, :boi_token_pos + 1], attention_mask.new_ones(num_patches),  # 添加 BOI 前的掩码和新的补丁掩码
                     attention_mask[i, eoi_token_pos:])  # 添加 EOI 之后的掩码
                ))  
            # 将新的注意力掩码列表堆叠为张量
            attention_mask = torch.stack(new_attention_masks, dim=0)  
        if not is_first_forward:  # 如果不是第一次前向传播
            if past_key_values is not None:  # 如果过去的键值对不为 None
                # 只保留 position_ids 的最后一个元素
                position_ids = position_ids[..., -1:]  
                # 只保留 input_ids 的最后一个元素
                input_ids = input_ids[:, -1:]  
        # 返回一个字典，包含输入 ID、图像、过去的键值、位置 ID、注意力掩码及其他参数
        return {  
            "input_ids": input_ids,  # 输入 ID
            "images": images,  # 图像
            "past_key_values": past_key_values,  # 过去的键值
            "position_ids": position_ids,  # 位置 ID
            "attention_mask": attention_mask,  # 注意力掩码
            "return_last_logit": True,  # 是否返回最后的 logit
            "use_cache": use_cache  # 是否使用缓存
        }  

    def forward(  # 定义前向传播函数
            self,
            input_ids: Optional[torch.Tensor] = None,  # 可选的输入 ID 张量
            images: List[List[torch.Tensor]] = None,  # 可选的图像列表
            position_ids: Optional[torch.Tensor] = None,  # 可选的位置 ID 张量
            attention_mask: Optional[torch.Tensor] = None,  # 可选的注意力掩码
            past_key_values: Optional[Tuple[torch.FloatTensor]] = None,  # 可选的过去键值对
            inputs_embeds: Optional[torch.Tensor] = None,  # 可选的输入嵌入
            labels: Optional[torch.Tensor] = None,  # 可选的标签
            use_cache: Optional[bool] = None,  # 是否使用缓存
            output_attentions: Optional[bool] = None,  # 是否输出注意力
            output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
            return_dict: Optional[bool] = None,  # 是否以字典形式返回
            return_last_logit: Optional[bool] = False,  # 是否返回最后的 logit
    # 在方法的括号结束位置，可能是某个函数定义的一部分
        ):
            # 根据配置选择是否使用缓存，默认使用类配置中的值
            use_cache = use_cache if use_cache is not None else self.config.use_cache
            # 根据配置选择是否返回字典格式，默认使用类配置中的值
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
            # 调用 transformer 进行前向传播，传入多个输入参数
            transformer_outputs = self.transformer(
                # 输入的 token ID
                input_ids=input_ids,
                # 输入的图像数据
                images=images,
                # 位置编码的 ID
                position_ids=position_ids,
                # 注意力掩码
                attention_mask=attention_mask,
                # 过去的键值对
                past_key_values=past_key_values,
                # 输入的嵌入表示
                inputs_embeds=inputs_embeds,
                # 是否使用缓存
                use_cache=use_cache,
                # 是否输出隐藏状态
                output_hidden_states=output_hidden_states,
                # 是否返回字典格式
                return_dict=return_dict,
            )
    
            # 从 transformer 输出中获取隐藏状态
            hidden_states = transformer_outputs[0]
            # 如果需要返回最后的 logit，则只保留最后一个时间步的隐藏状态
            if return_last_logit:
                hidden_states = hidden_states[:, -1:]
            # 通过输出层获取语言模型的 logits
            lm_logits = self.transformer.output_layer(hidden_states)
    
            # 初始化损失为 None
            loss = None
            # 如果标签存在，则计算损失
            if labels is not None:
                # 创建新的标签列表
                new_labels = []
                # 遍历每个输入 ID
                for i in range(len(input_ids)):
                    # 将当前输入 ID 转换为列表
                    input_id = input_ids[i].tolist()
                    # 获取 BOI 和 EOI 令牌的位置
                    boi_token_pos, eoi_token_pos = input_id.index(self.config.boi_token_id), input_id.index(
                        self.config.eoi_token_id)
                    # 确保 EOI 和 BOI 之间的间隔为 2
                    assert eoi_token_pos - boi_token_pos == 2
    
                    # 构建新的标签，包含 BOI 之前的标签和 EOI 之后的标签
                    new_labels.append(torch.cat(
                        (
                            labels[i, :boi_token_pos + 1],
                            torch.tensor([-100]).to(labels.device).to(labels.dtype).repeat(1600),
                            labels[i, eoi_token_pos:])))
    
                # 将新的标签列表转换为张量
                labels = torch.stack(new_labels, dim=0)
                # 将 logits 转换为 float32 类型
                lm_logits = lm_logits.to(torch.float32)
                # 将 logits 和标签分别向左移动一个时间步
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # 定义交叉熵损失函数，忽略 -100 的标签
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                # 计算损失
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
                # 将 logits 和损失转换为与隐藏状态相同的类型
                lm_logits = lm_logits.to(hidden_states.dtype)
                loss = loss.to(hidden_states.dtype)
    
            # 如果不返回字典格式，组合输出结果
            if not return_dict:
                output = (lm_logits,) + transformer_outputs[1:]
                # 如果存在损失，返回损失和输出结果
                return ((loss,) + output) if loss is not None else output
    
            # 返回包含损失、logits 和其他 transformer 输出的 CausalLMOutputWithPast 对象
            return CausalLMOutputWithPast(
                loss=loss,
                logits=lm_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
            )
    
        # 定义静态方法，用于重新排序缓存
        @staticmethod
        def _reorder_cache(
                # past 包含过去的键值对
                past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], 
                # beam 的索引
                beam_idx: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:  # 指定函数返回类型为元组，其中包含元组，元组内包含两个 torch.Tensor
        """  # 函数文档字符串，描述该函数的功能
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or  # 该函数用于重新排序 `past_key_values` 缓存，适用于 beam_search 或 beam_sample 调用
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct  # 这个操作是为了在每次生成步骤中，将 `past_key_values` 与正确的 beam_idx 匹配
        beam_idx at every generation step.  # 在每次生成步骤中匹配 beam_idx
        
        Output shares the same memory storage as `past`.  # 输出与 `past` 共享相同的内存存储
        """
        return tuple(  # 返回一个元组，包含处理后的每层过去的键值
            (  # 开始一个元组，包含两个处理后的 torch.Tensor
                layer_past[0].index_select(0, beam_idx.to(layer_past[0].device)),  # 从 layer_past 的第一个 tensor 中选择对应 beam_idx 的元素，并转移到相应设备
                layer_past[1].index_select(0, beam_idx.to(layer_past[1].device)),  # 从 layer_past 的第二个 tensor 中选择对应 beam_idx 的元素，并转移到相应设备
            )  # 结束当前层的元组
            for layer_past in past  # 遍历 past 中的每个 layer_past
        )  # 结束 tuple 的构造
# 定义一个用于序列分类的模型类，继承自预训练模型类
class ChatGLMForSequenceClassification(ChatGLMPreTrainedModel):
    # 初始化方法，接受配置对象、空初始化标志和设备参数
    def __init__(self, config: ChatGLMConfig, empty_init=True, device=None):
        # 调用父类的初始化方法，传入配置
        super().__init__(config)

        # 获取配置中的标签数量
        self.num_labels = config.num_labels
        # 创建变换器模型，使用配置和传入的参数
        self.transformer = ChatGLMModel(config, empty_init=empty_init, device=device)

        # 创建一个线性分类头，输入维度为隐藏层大小，输出维度为标签数量
        self.classifier_head = nn.Linear(config.hidden_size, config.num_labels, bias=True, dtype=torch.half)
        # 如果配置中指定了分类头的 dropout 率，则创建 Dropout 层
        if config.classifier_dropout is not None:
            self.dropout = nn.Dropout(config.classifier_dropout)
        # 否则，将 dropout 设置为 None
        else:
            self.dropout = None
        # 将配置对象保存到实例变量中
        self.config = config

    # 定义前向传播方法，处理输入和可选参数
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            full_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            inputs_embeds: Optional[torch.LongTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    # 返回类型可以是元组或带过去状态的序列分类输出
) -> Union[Tuple[torch.Tensor, ...], SequenceClassifierOutputWithPast]:
    # 如果 return_dict 为 None，则使用配置中的 use_return_dict 值
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # 调用变换器模型，传入相关参数以获取输出
    transformer_outputs = self.transformer(
        input_ids=input_ids,  # 输入的 ID 列表
        position_ids=position_ids,  # 输入的位置信息
        attention_mask=attention_mask,  # 注意力掩码
        full_attention_mask=full_attention_mask,  # 完整的注意力掩码
        past_key_values=past_key_values,  # 过去的键值对
        inputs_embeds=inputs_embeds,  # 输入的嵌入表示
        use_cache=use_cache,  # 是否使用缓存
        output_hidden_states=output_hidden_states,  # 是否输出隐藏状态
        return_dict=return_dict,  # 是否返回字典格式的输出
    )

    # 获取变换器输出的隐藏状态
    hidden_states = transformer_outputs[0]
    # 取最后一个隐藏状态作为池化的隐藏状态
    pooled_hidden_states = hidden_states[-1]
    # 如果设置了 dropout，则对池化的隐藏状态应用 dropout
    if self.dropout is not None:
        pooled_hidden_states = self.dropout(pooled_hidden_states)
    # 通过分类头计算 logits
    logits = self.classifier_head(pooled_hidden_states)

    # 初始化损失为 None
    loss = None
    # 如果标签不为空，则计算损失
    if labels is not None:
        # 如果问题类型尚未确定，则根据标签数量和类型确定
        if self.config.problem_type is None:
            if self.num_labels == 1:
                self.config.problem_type = "regression"  # 回归问题
            elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                self.config.problem_type = "single_label_classification"  # 单标签分类
            else:
                self.config.problem_type = "multi_label_classification"  # 多标签分类

        # 根据问题类型计算损失
        if self.config.problem_type == "regression":
            loss_fct = MSELoss()  # 均方误差损失
            if self.num_labels == 1:
                # 对于单标签回归，计算损失
                loss = loss_fct(logits.squeeze().float(), labels.squeeze())
            else:
                # 对于多标签回归，计算损失
                loss = loss_fct(logits.float(), labels)
        elif self.config.problem_type == "single_label_classification":
            loss_fct = CrossEntropyLoss()  # 交叉熵损失
            # 计算损失
            loss = loss_fct(logits.view(-1, self.num_labels).float(), labels.view(-1))
        elif self.config.problem_type == "multi_label_classification":
            loss_fct = BCEWithLogitsLoss()  # 二进制交叉熵损失
            # 计算损失
            loss = loss_fct(logits.float(), labels.view(-1, self.num_labels))

    # 如果不返回字典格式，则返回 logits 和其他输出
    if not return_dict:
        output = (logits,) + transformer_outputs[1:]  # 包含 logits 的输出
        # 如果有损失，则返回损失和其他输出；否则只返回输出
        return ((loss,) + output) if loss is not None else output

    # 返回带过去状态的序列分类输出，包含损失、logits 和其他信息
    return SequenceClassifierOutputWithPast(
        loss=loss,  # 损失
        logits=logits,  # 预测的 logits
        past_key_values=transformer_outputs.past_key_values,  # 过去的键值对
        hidden_states=transformer_outputs.hidden_states,  # 隐藏状态
        attentions=transformer_outputs.attentions,  # 注意力信息
    )
```