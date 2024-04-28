# `.\transformers\models\switch_transformers\modeling_switch_transformers.py`

```py
# 设置字符编码格式为 UTF-8
# 声明版权信息，包括作者和版权方
# 根据 Apache 许可证 Version 2.0 授权，对该文件的使用有一定规定
# 在遵守许可证的前提下，可以复制、使用本文件
# 可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 根据适用法律或书面同意，以"现状"基础提供软件
# 没有任何明示或暗示的担保或条件
# 详细了解许可情况，请查看相应的许可证内容
# 本类实现了 PyTorch 下的 SwitchTransformers 模型
import copy
import math
import warnings
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
# 导入激活函数映射表
from ...activations import ACT2FN
# 导入各种输出类型
from ...modeling_outputs import (
    MoEModelOutput,
    MoEModelOutputWithPastAndCrossAttentions,
    Seq2SeqMoEModelOutput,
    Seq2SeqMoEOutput,
)
# 导入模型工具
from ...modeling_utils import PreTrainedModel
# 导入 PyTorch 工具函数
from ...pytorch_utils import ALL_LAYERNORM_LAYERS, find_pruneable_heads_and_indices, prune_linear_layer
# 导入通用工具
from ...utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_fx_proxy,
    logging,
    replace_return_docstrings,
)
# 导入 SwitchTransformers 配置
from .configuration_switch_transformers import SwitchTransformersConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 以下是文档字符串中额外涉及到的配置和检查点信息
_CONFIG_FOR_DOC = "SwitchTransformersConfig"
_CHECKPOINT_FOR_DOC = "google/switch-base-8"

####################################################
# This dict contains ids and associated url
# for the pretrained weights provided with the models
####################################################
# 预训练权重对应的字典，包括 ID 和 URL
SWITCH_TRANSFORMERS_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/switch-base-8",
    "google/switch-base-16",
    "google/switch-base-32",
    "google/switch-base-64",
    "google/switch-base-128",
    "google/switch-base-256",
    "google/switch-large-128",
    "google/switch-xxl-128",
    "google/switch-c-2048",
    # 通过链接查看所有 SwitchTransformers 模型 https://huggingface.co/models?filter=switch_transformers
]

# 定义路由器 z 损失函数
def router_z_loss_func(router_logits: torch.Tensor) -> float:
    r"""
    Compute the router z-loss implemented in PyTorch.

    The router z-loss was introduced in [Designing Effective Sparse Expert Models](https://arxiv.org/abs/2202.08906).
    It encourages router logits to remain small in an effort to improve stability.

    Args:
        router_logits (`float`):
            Input logits of shape [batch_size, sequence_length, num_experts]

    Returns:
        Scalar router z-loss.
    """
    # 获取维度信息
    num_groups, tokens_per_group, _ = router_logits.shape
    # 对 `router_logits` 沿指定维度求 logsumexp
    log_z = torch.logsumexp(router_logits, dim=-1)
    # 计算 z 损失
    z_loss = log_z**2
    return torch.sum(z_loss) / (num_groups * tokens_per_group)

# 定义负载均衡损失函数
def load_balancing_loss_func(router_probs: torch.Tensor, expert_indices: torch.Tensor) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        router_probs (`torch.Tensor`):
            Probability assigned to each expert per token. Shape: [batch_size, seqeunce_length, num_experts].
        expert_indices (`torch.Tensor`):
            Indices tensor of shape [batch_size, seqeunce_length] identifying the selected expert for a given token.

    Returns:
        The auxiliary loss.
    """
    # 获取专家数量
    num_experts = router_probs.shape[-1]

    # 将专家索引转换为int64类型，否则one-hot编码将失败
    if expert_indices.dtype != torch.int64:
        expert_indices = expert_indices.to(torch.int64)

    # 如果专家索引的维度为2，则添加一个维度
    if len(expert_indices.shape) == 2:
        expert_indices = expert_indices.unsqueeze(2)

    # 使用one-hot编码生成专家掩码
    expert_mask = torch.nn.functional.one_hot(expert_indices, num_experts)

    # 对于给定的令牌，确定是否将其路由到特定的专家
    expert_mask = torch.max(expert_mask, axis=-2).values

    # 将掩码转换为float32类型，否则计算平均值将失败
    expert_mask = expert_mask.to(torch.float32)

    # 计算每组和专家的令牌数的平均值
    tokens_per_group_and_expert = torch.mean(expert_mask, axis=-2)

    # 计算每组和专家的路由概率的平均值
    router_prob_per_group_and_expert = torch.mean(router_probs, axis=-2)

    # 计算辅助损失
    return torch.mean(tokens_per_group_and_expert * router_prob_per_group_and_expert) * (num_experts**2)
class SwitchTransformersTop1Router(nn.Module):
    """
    Router using tokens choose top-1 experts assignment.

    This router uses the same mechanism as in Switch Transformer (https://arxiv.org/abs/2101.03961) and V-MoE
    (https://arxiv.org/abs/2106.05974): tokens choose their top experts. Items are sorted by router_probs and then
    routed to their choice of expert until the expert's expert_capacity is reached. **There is no guarantee that each
    token is processed by an expert**, or that each expert receives at least one token.

    """

    # 初始化函数，设定路由器的参数
    def __init__(self, config: SwitchTransformersConfig):
        super().__init__()
        # 设定专家数量和专家容量
        self.num_experts = config.num_experts
        self.expert_capacity = config.expert_capacity
        # 创建线性分类器，输出层为num_experts个神经元
        self.classifier = nn.Linear(config.hidden_size, self.num_experts, bias=config.router_bias)
        # 设定路由器的抖动噪声以及是否忽略填充标记
        self.jitter_noise = config.router_jitter_noise
        self.ignore_padding_tokens = config.router_ignore_padding_tokens
        self.dtype = getattr(torch, config.router_dtype)

    # 计算路由概率
    def _compute_router_probabilities(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Computes router probabilities from input hidden states.

        Args:
            hidden_states (`torch.Tensor`):
                (batch_size, sequence_length, hidden_dim) from which router probabilities are computed.
        Returns:
            router_probabilities (`torch.Tensor`):
                Tensor of shape (batch_size, sequence_length, num_experts) corresponding to the probabilities for each
                token and expert. Used for routing tokens to experts.
            router_logits (`torch.Tensor`):
                Logits tensor of shape (batch_size, sequence_length, num_experts) corresponding to raw router logits.
                This is used later for computing router z-loss.
        """
        # 为了确保稳定性，使用float32
        # 保存原始数据类型，以便计算完之后将数据类型转回原始类型
        self.input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.dtype)

        # 如果处于训练阶段并且有抖动噪声，则给输入的隐藏状态添加一些噪声
        if self.training and self.jitter_noise > 0:
            # 通过均匀分布乘以标记输入，添加一些噪声
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)

        # 计算分类器的输出
        # Shape: [num_groups, tokens_per_group, num_experts]
        self._cast_classifier()
        router_logits = self.classifier(hidden_states)

        # 对输出应用Softmax，并转回原始数据类型
        router_probabilities = nn.functional.softmax(router_logits, dim=-1, dtype=self.dtype).to(self.input_dtype)
        return router_probabilities, router_logits
    # 该函数检查分类器模型是否为 Linear8bitLt 类型，如果不是则将其转换为与 self.dtype 相同的数据类型
    def _cast_classifier(self):
        # 检查分类器模型是否具有 SCB 或 CB 属性，如果没有则进行数据类型转换
        if not (hasattr(self.classifier, "SCB") or hasattr(self.classifier, "CB")):
            self.classifier = self.classifier.to(self.dtype)
    
    # 通用前向传播函数，每个路由器类都需要实现此函数
    def forward(self, hidden_states: torch.Tensor) -> Tuple:
        # 计算路由器概率和逻辑值
        router_probs, router_logits = self._compute_router_probabilities(hidden_states)
    
        # 获取每个token路由到的专家索引，并使用 one-hot 编码表示
        expert_index = torch.argmax(router_probs, dim=-1)
        expert_index = torch.nn.functional.one_hot(expert_index, num_classes=self.num_experts)
    
        # 计算每个序列中已被路由的token数量
        token_priority = torch.cumsum(expert_index, dim=-2)
        # 如果当前token的路由会导致专家容量溢出，则将其 mask 掉
        expert_capacity_mask = token_priority <= self.expert_capacity
        expert_index = expert_index * expert_capacity_mask
    
        # 获取每个token被路由的最大概率
        router_probs = torch.max(router_probs, dim=-1).values.unsqueeze(-1)
    
        # 返回专家索引、路由概率和逻辑值
        return expert_index, router_probs, router_logits
# 从transformers.models.t5.modeling_t5.T5LayerNorm复制的SwitchTransformersLayerNorm类
class SwitchTransformersLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        在SwitchTransformers风格下构建layernorm模块。无偏置和均值减法。
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # SwitchTransformers使用一个只进行缩放而不进行移位的layer_norm，也被称为均方根层归一化，
        # 因此方差是在没有均值和无偏差的情况下计算的。此外，我们要确保对于半精度输入的累积是以fp32进行的
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # 如果必要，转换为半精度
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


ALL_LAYERNORM_LAYERS.append(SwitchTransformersLayerNorm)


# 从transformers.models.t5.modeling_t5.T5DenseActDense复制的SwitchTransformersDenseActDense类
class SwitchTransformersDenseActDense(nn.Module):
    def __init__(self, config: SwitchTransformersConfig):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)
        return hidden_states


# SwitchTransformersSparseMLP类
class SwitchTransformersSparseMLP(nn.Module):
    r"""
    实现Switch Transformers Sparse MLP模块。
    """

    def __init__(self, config: SwitchTransformersConfig, expert_class: nn.Module = SwitchTransformersDenseActDense):
        super().__init__()
        # 步骤1：根据其类获取正确的路由器
        self.router = SwitchTransformersTop1Router(config)

        # 步骤2：获取专家
        self.experts = nn.ModuleDict()
        for idx in range(config.num_experts):
            self.experts[f"expert_{idx}"] = expert_class(config)
    # 前向传播方法
    def forward(self, hidden_states):
        # 从路由器获取路由掩码、路由概率和路由对数概率
        router_mask, router_probs, router_logits = self.router(hidden_states)
        # 获取每个token分配给的专家索引
        expert_index = torch.argmax(router_mask, dim=-1)
    
        # 由于路由器可能不会将所有token映射到专家,一些hidden states可能保持不变,
        # 因此需要克隆hidden_states,只更新选中的部分
        next_states = hidden_states.clone()
        for idx, expert in enumerate(self.experts.values()):
            # 找到分配给当前专家的token索引
            token_indices = router_mask[:, :, idx].bool()
            # 将token传入对应专家,并更新next_states
            next_states[token_indices] = expert(hidden_states[token_indices]).to(next_states.dtype)
    
        # 将更新后的next_states与路由概率相乘,得到最终的hidden_states
        hidden_states = router_probs * next_states
        return hidden_states, (router_logits, expert_index)
class SwitchTransformersLayerFF(nn.Module):
    r"""
    Switch Transformers Feed Forward layer module. This is a wrapper around the Mixture of Experts module.

    Parameters:
        config : ([`SwitchTransformersConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
        is_sparse (`bool`):
            Whether the MLP layer is a `Sparse` layer (contains a Mixture of Experts) or not
    """

    def __init__(self, config: SwitchTransformersConfig, is_sparse=False):
        super().__init__()
        self.is_sparse = is_sparse

        # Check if it is a sparse layer, if not then it is a dense layer
        if not self.is_sparse:
            # If the layer is not sparse, initialize the MLP as dense
            self.mlp = SwitchTransformersDenseActDense(config)
        else:
            # If the layer is sparse, initialize the MLP as sparse
            self.mlp = SwitchTransformersSparseMLP(config)

        # Initialize layer normalization with the specified configuration
        self.layer_norm = SwitchTransformersLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # Initialize dropout with the specified dropout rate
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, output_router_logits):
        # Apply layer normalization to the input hidden states
        forwarded_states = self.layer_norm(hidden_states)
        # Pass the normalized hidden states through the MLP
        forwarded_states = self.mlp(forwarded_states)

        # If the forwarded states return a tuple (used in MoE), unpack it
        if isinstance(forwarded_states, tuple):
            forwarded_states, router_tuple = forwarded_states
        else:
            router_tuple = None

        # Apply dropout to the forwarded states and add them to the original hidden states
        output = hidden_states + self.dropout(forwarded_states)

        # If output_router_logits is True and router_tuple is not None, return both output and router_tuple
        if output_router_logits and router_tuple is not None:
            output = (output, router_tuple)

        return output


# Copied from transformers.models.t5.modeling_t5.T5Attention with T5->SwitchTransformers
class SwitchTransformersAttention(nn.Module):
    # 初始化函数，接受配置参数和是否有相对注意力偏差作为输入
    def __init__(self, config: SwitchTransformersConfig, has_relative_attention_bias=False):
        # 调用父类的初始化函数
        super().__init__()
        # 设置是否为解码器
        self.is_decoder = config.is_decoder
        # 设置是否有相对注意力偏差
        self.has_relative_attention_bias = has_relative_attention_bias
        # 设置相对注意力的桶数量
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        # 设置相对注意力的最大距离
        self.relative_attention_max_distance = config.relative_attention_max_distance
        # 设置模型维度
        self.d_model = config.d_model
        # 设置键和值的投影维度
        self.key_value_proj_dim = config.d_kv
        # 设置注意力头的数量
        self.n_heads = config.num_heads
        # 设置 dropout 率
        self.dropout = config.dropout_rate
        # 计算内部维度
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # 进行 Mesh TensorFlow 初始化以避免在 softmax 前进行缩放
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)  # 创建查询线性层
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)  # 创建键线性层
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)  # 创建值线性层
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)  # 创建输出线性层

        # 如果存在相对注意力偏差，创建相对注意力偏差的嵌入层
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        # 初始化已剪枝的注意力头集合
        self.pruned_heads = set()
        # 是否使用梯度检查点
        self.gradient_checkpointing = False

    # 剪枝注意力头
    def prune_heads(self, heads):
        # 如果剪枝头的数量为0，直接返回
        if len(heads) == 0:
            return
        # 查找可剪枝的注意力头及其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
        )
        # 剪枝线性层
        self.q = prune_linear_layer(self.q, index)  # 剪枝查询线性层
        self.k = prune_linear_layer(self.k, index)  # 剪枝键线性层
        self.v = prune_linear_layer(self.v, index)  # 剪枝值线性层
        self.o = prune_linear_layer(self.o, index, dim=1)  # 剪枝输出线性层
        # 更新超参数
        self.n_heads = self.n_heads - len(heads)  # 更新注意力头的数量
        self.inner_dim = self.key_value_proj_dim * self.n_heads  # 更新内部维度
        self.pruned_heads = self.pruned_heads.union(heads)  # 合并剪枝头集合
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        将相对位置转换为用于相对注意力的桶编号。相对位置定义为memory_position - query_position，即从参与位置到被参与位置的距离（以标记为单位）。如果bidirectional=False，则正相对位置无效。
        我们对小的绝对relative_position使用较小的桶，对大的绝对relative_position使用较大的桶。所有>=max_distance的相对位置映射到同一个桶。所有<=-max_distance的相对位置映射到同一个桶。
        这应该能更优雅地推广到比模型训练时更长的序列

        Args:
            relative_position: 一个 int32 张量
            bidirectional: 一个布尔值 - 注意力是否双向
            num_buckets: 一个整数
            max_distance: 一个整数

        Returns:
            一个具有与relative_position相同形状的张量，包含范围为[0, num_buckets)中的int32值
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # 现在relative_position在范围[0, 无穷)

        # 一半的桶用于位置的准确递增
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # 另一半的桶用于位置上到max_distance的对数增长的更大箱中
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets
    # 计算分箱的相对位置偏差
    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        # 如果设备为空，则使用self.relative_attention_bias.weight的设备
        if device is None:
            device = self.relative_attention_bias.weight.device
        # 生成长度为query_length的张量，用于表示上下文位置
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        # 生成长度为key_length的张量，用于表示记忆位置
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        # 计算相对位置，形状为(query_length, key_length)
        relative_position = memory_position - context_position
        # 根据相对位置和其他参数，将相对位置分配到不同的桶中
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        # 使用相对位置桶计算相对注意力偏置值
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        # 调整形状，得到(1, num_heads, query_length, key_length)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        # 返回计算好的值
        return values

    # 前向传播函数
    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
# 该类实现 Self-Attention 机制，继承自 nn.Module
class SwitchTransformersLayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        # 创建 SwitchTransformersAttention 对象，用于计算 Self-Attention
        self.SelfAttention = SwitchTransformersAttention(
            config, has_relative_attention_bias=has_relative_attention_bias
        )
        # 创建 SwitchTransformersLayerNorm 对象，用于进行层归一化
        self.layer_norm = SwitchTransformersLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # 创建 Dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(config.dropout_rate)

    # 前向传播函数
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        # 执行层归一化
        normed_hidden_states = self.layer_norm(hidden_states)
        # 计算 Self-Attention 输出
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # 将 Self-Attention 输出与原始输入相加，并应用 Dropout
        hidden_states = hidden_states + self.dropout(attention_output[0])
        # 返回输出结果，包括隐藏状态以及可选的注意力权重
        outputs = (hidden_states,) + attention_output[1:]
        return outputs

# 该类实现 Cross-Attention 机制，继承自 nn.Module 
class SwitchTransformersLayerCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建 SwitchTransformersAttention 对象，用于计算 Cross-Attention
        self.EncDecAttention = SwitchTransformersAttention(config, has_relative_attention_bias=False)
        # 创建 SwitchTransformersLayerNorm 对象，用于进行层归一化
        self.layer_norm = SwitchTransformersLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # 创建 Dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(config.dropout_rate)

    # 前向传播函数
    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        # 执行层归一化
        normed_hidden_states = self.layer_norm(hidden_states)
        # 计算 Cross-Attention 输出
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        # 将 Cross-Attention 输出与原始输入相加，并应用 Dropout
        layer_output = hidden_states + self.dropout(attention_output[0])
        # 返回输出结果，包括隐藏状态以及可选的注意力权重
        outputs = (layer_output,) + attention_output[1:]
        return outputs

# SwitchTransformersBlock 类定义
class SwitchTransformersBlock(nn.Module):
    # 初始化对象，设置是否为解码器，是否稀疏
    def __init__(self, config, has_relative_attention_bias=False, is_sparse=False):
        # 调用父类初始化函数
        super().__init__()
        # 设置是否为解码器
        self.is_decoder = config.is_decoder
        # 设置是否稀疏
        self.is_sparse = is_sparse
        # 初始化层列表
        self.layer = nn.ModuleList()
        # 添加自注意力层
        self.layer.append(
            SwitchTransformersLayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias)
        )
        # 如果是解码器，添加交叉注意力层
        if self.is_decoder:
            self.layer.append(SwitchTransformersLayerCrossAttention(config))

        # 添加前馈神经网络层
        self.layer.append(SwitchTransformersLayerFF(config, is_sparse=self.is_sparse))

    # 前向传播函数
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        output_router_logits=True,
        return_dict=True,
# 定义一个 SwitchTransformersPreTrainedModel 类，用于处理权重初始化以及预训练模型的下载和加载
class SwitchTransformersPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置 config_class 属性为 SwitchTransformersConfig
    config_class = SwitchTransformersConfig
    # 设置 base_model_prefix 属性为 "switch_transformers"
    base_model_prefix = "switch_transformers"
    # 设置 supports_gradient_checkpointing 属性为 True
    supports_gradient_checkpointing = True
    # 定义 _no_split_modules 属性为 ["SwitchTransformersBlock"]
    _no_split_modules = ["SwitchTransformersBlock"]

    # 定义 dummy_inputs 属性，返回包含 DUMMY_INPUTS 和 DUMMY_MASK 的字典
    @property
    def dummy_inputs(self):
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs

    # 定义 _shift_right 方法，用于将输入向右移动一个位置
    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        # 检查 decoder_start_token_id 是否已定义
        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In SwitchTransformers it is usually set"
                " to the pad_token_id. See SwitchTransformers docs for more information"
            )

        # 将输入向右移动一个位置
        if is_torch_fx_proxy(input_ids):
            # 对代理对象不支持原生项赋值
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        # 检查 pad_token_id 是否已定义
        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # 将 labels 中可能存在的 -100 值替换为 pad_token_id
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

# 创建 SwitchTransformersStack 类，继承自 SwitchTransformersPreTrainedModel 类
class SwitchTransformersStack(SwitchTransformersPreTrainedModel):
    # 初始化函数，接受配置和嵌入标记作为参数
    def __init__(self, config, embed_tokens=None):
        # 调用父类的初始化函数
        super().__init__(config)

        # 创建一个嵌入层，使用config中的词汇表大小和d_model
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        # 如果给定了embed_tokens，则将其权重赋给embed_tokens层
        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        # 设置解码器和编码器的标志
        self.is_decoder = config.is_decoder

        # 根据config中的解码器/编码器稀疏步数，设置num_layers参数
        sparse_step = config.decoder_sparse_step if self.is_decoder else config.encoder_sparse_step
        config.num_layers = config.num_decoder_layers if self.is_decoder else config.num_layers

        # 初始化块列表
        self.block = nn.ModuleList()
        # 遍历块的数量
        for i in range(config.num_layers):
            # 判断当前层是否稀疏
            is_sparse = (i % sparse_step == 1 or sparse_step == 1) if sparse_step > 0 else False

            # 在块列表中添加SwitchTransformersBlock
            self.block.append(
                SwitchTransformersBlock(config, has_relative_attention_bias=bool(i == 0), is_sparse=is_sparse)
            )

        # 初始化最终层标准化
        self.final_layer_norm = SwitchTransformersLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # 初始化dropout
        self.dropout = nn.Dropout(config.dropout_rate)

        # 初始化权重并应用最终处理
        self.post_init()

        # 初始化设备映射和梯度检查点标志
        self.device_map = None
        self.gradient_checkpointing = False

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置输入嵌入
    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    # 前向传播函数，接受多个参数
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        output_router_logits=True,
        return_dict=None,
# 定义SWITCH_TRANSFORMERS_START_DOCSTRING常量，存储关于Switch Transformers模型的说明文档字符串
SWITCH_TRANSFORMERS_START_DOCSTRING = r"""
    The SWITCH_TRANSFORMERS model was proposed in [Switch Transformers: Scaling to Trillion Parameter Models with
    Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) by [William
    Fedus](https://arxiv.org/search/cs?searchtype=author&query=Fedus%2C+W), [Barret
    Zoph](https://arxiv.org/search/cs?searchtype=author&query=Zoph%2C+B), and [Noam
    Shazeer](https://arxiv.org/search/cs?searchtype=author&query=Shazeer%2C+N). It's an encoder-decoder T5-like model
    with sparse Feed Forward that stands for Mixture of Experts (MoE) architecture.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`SwitchTransformersConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义SWITCH_TRANSFORMERS_INPUTS_DOCSTRING常量，存储关于Switch Transformers模型输入的说明文档字符串
SWITCH_TRANSFORMERS_INPUTS_DOCSTRING = r"""
"""

# 定义SWITCH_TRANSFORMERS_ENCODER_INPUTS_DOCSTRING常量，存储关于Switch Transformers模型编码器输入的说明文档字符串
SWITCH_TRANSFORMERS_ENCODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            # 输入序列标记在词汇表中的索引。SWITCH_TRANSFORMERS 是一个具有相对位置嵌入的模型，因此应该能够在右侧和左侧都可以对输入进行填充。

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for detail.

            To know more on how to prepare `input_ids` for pretraining take a look a [SWITCH_TRANSFORMERS
            Training](./switch_transformers#training).
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
             # 用于避免在填充的标记索引上执行注意力的掩码。掩码的值选择在 `[0, 1]` 之间:

            - 1 表示 **不被掩盖** 的标记,
            - 0 表示 **被掩盖** 的标记。

            [What are attention masks?](../glossary#attention-mask)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于使自注意力模块的某些头部失效的掩码。掩码的值选择在 `[0, 1]` 之间:

            - 1 表示头部 **不被掩盖**，
            - 0 表示头部 **被掩盖**。

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            # 选择直接传入一个嵌入表示，而不是传入 `input_ids`。如果要更多控制权，可以选择这种方式将 `input_ids` 索引转换为关联向量，而不是使用模型内部的嵌入查找矩阵。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多细节，请参见返回张量下的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多细节，请参见返回张量下的 `hidden_states`。
        output_router_logits (`bool`, *optional*):
            # 是否返回所有路由器的逻辑。它们对于计算路由器损失很有用，在推理过程中不应该返回。
        return_dict (`bool`, *optional*):
            # 是否返回一个 [`~utils.ModelOutput`] 而不是一个普通的元组。
# 引入 FutureWarning 警告消息
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""

# 定义一个基本的 SWITCH_TRANSFORMERS Model 类，输出原始隐藏状态，没有特定的头部处理
# 继承自 SwitchTransformersPreTrainedModel
class SwitchTransformersModel(SwitchTransformersPreTrainedModel):
    # 被绑定权重的键
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    # 初始化方法
    def __init__(self, config: SwitchTransformersConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 共享的嵌入层
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # 复制编码器配置
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # 创建编码器对象
        self.encoder = SwitchTransformersStack(encoder_config, self.shared)

        # 复制解码器配置
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        # 创建解码器对象
        self.decoder = SwitchTransformersStack(decoder_config, self.shared)

        # 初始化权重并应用最终处理
        self.post_init()

        # 模型并行
        self.device_map = None

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.shared

    # 设置输入嵌入
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    # 绑定权重
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    # 获取编码器对象
    def get_encoder(self):
        return self.encoder

    # 获取解码器对象
    def get_decoder(self):
        return self.decoder

    # 删减模型的头部
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 向前传播方法的装饰器
    @add_start_docstrings_to_model_forward(SWITCH_TRANSFORMERS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqMoEModelOutput, config_class=_CONFIG_FOR_DOC)
    # Transformer 模型的前向传播函数，用于进行模型推断或者训练
    def forward(
        # 输入序列的 token IDs，可选参数，默认为 None
        input_ids: Optional[torch.LongTensor] = None,
        # 输入序列的 attention mask，可选参数，默认为 None
        attention_mask: Optional[torch.FloatTensor] = None,
        # 解码器输入序列的 token IDs，可选参数，默认为 None
        decoder_input_ids: Optional[torch.LongTensor] = None,
        # 解码器输入序列的 attention mask，可选参数，默认为 None
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        # 自注意力机制的头部 mask，可选参数，默认为 None
        head_mask: Optional[torch.FloatTensor] = None,
        # 解码器自注意力机制的头部 mask，可选参数，默认为 None
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        # 编码器-解码器注意力机制的头部 mask，可选参数，默认为 None
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        # 编码器输出的元组，包含编码器各层的隐藏状态，可选参数，默认为 None
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        # 缓存的键值对，用于注意力机制，可选参数，默认为 None
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        # 输入嵌入向量，可选参数，默认为 None
        inputs_embeds: Optional[torch.Tensor] = None,
        # 解码器输入嵌入向量，可选参数，默认为 None
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        # 是否使用缓存，可选参数，默认为 None
        use_cache: Optional[bool] = None,
        # 是否输出注意力权重，可选参数，默认为 None
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，可选参数，默认为 None
        output_hidden_states: Optional[bool] = None,
        # 是否输出路由器逻辑，可选参数，默认为 None
        output_router_logits: Optional[bool] = None,
        # 是否以字典形式返回结果，可选参数，默认为 None
        return_dict: Optional[bool] = None,
# 使用自定义的文档字符串和SWITCH_TRANSFORMERS_START_DOCSTRING创建SWITCH_TRANSFORMERS Model，并添加一个语言建模头部
@add_start_docstrings(
    """SWITCH_TRANSFORMERS Model with a `language modeling` head on top.""", SWITCH_TRANSFORMERS_START_DOCSTRING
)
# 定义SwitchTransformersForConditionalGeneration类，继承自SwitchTransformersPreTrainedModel类
class SwitchTransformersForConditionalGeneration(SwitchTransformersPreTrainedModel):
    # 要共享权重的关键字列表
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    # 初始化函数，接收SwitchTransformersConfig类型的config对象
    def __init__(self, config: SwitchTransformersConfig):
        super().__init__(config)
        # 模型维度
        self.model_dim = config.d_model

        # 共享的嵌入层
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # 复制并修改编码器配置
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # 创建编码器
        self.encoder = SwitchTransformersStack(encoder_config, self.shared)

        # 复制并修改解码器配置
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        # 创建解码器
        self.decoder = SwitchTransformersStack(decoder_config, self.shared)

        # 语言建模头层
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 路由器损失系数和路由器辅助损失系数
        self.router_z_loss_coef = config.router_z_loss_coef
        self.router_aux_loss_coef = config.router_aux_loss_coef

        # 初始化权重并应用最终处理
        self.post_init()

        # 模型并行
        self.device_map = None

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.shared

    # 设置输入嵌入
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    # 绑定权重
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.lm_head

    # 获取编码器
    def get_encoder(self):
        return self.encoder

    # 获取解码器
    def get_decoder(self):
        return self.decoder

    # 添加模型前向传播的文档字符串和SWITCH_TRANSFORMERS_INPUTS_DOCSTRING
    # 替换返回值文档字符串为Seq2SeqMoEOutput，配置类为_CONFIG_FOR_DOC
    @add_start_docstrings_to_model_forward(SWITCH_TRANSFORMERS_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqMoEOutput, config_class=_CONFIG_FOR_DOC)
    # 定义 forward 方法，用于模型的前向传播
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的 token ID
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力掩码
        decoder_input_ids: Optional[torch.LongTensor] = None,  # 解码器的 token ID
        decoder_attention_mask: Optional[torch.BoolTensor] = None,  # 解码器的注意力掩码
        head_mask: Optional[torch.FloatTensor] = None,  # 多头注意力的掩码
        decoder_head_mask: Optional[torch.FloatTensor] = None,  # 解码器多头注意力的掩码
        cross_attn_head_mask: Optional[torch.Tensor] = None,  # 跨模块的多头注意力的掩码
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,  # 编码器的输出
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,  # 过去的键值
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 嵌入输入
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,  # 解码器的嵌入输入
        labels: Optional[torch.LongTensor] = None,  # 标签
        use_cache: Optional[bool] = None,  # 是否使用缓存
        output_attentions: Optional[bool] = None,  # 是否输出注意力
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        output_router_logits: Optional[bool] = True,  # 默认输出路由器 logit
        return_dict: Optional[bool] = None,  # 是否返回字典形式的结果
    )

    # 解析路由器 logit 的辅助方法
    def _unpack_router_logits(self, router_outputs):
        total_router_logits = []  # 存储总的路由器 logit
        total_expert_indexes = []  # 存储总的专家索引
        for router_output in router_outputs:
            if len(router_output[0].shape) > 1:  # 判断路由器输出的形状是否大于1维
                router_logits, expert_indexes = router_output  # 拆分路由器输出
                total_router_logits.append(router_logits)  # 添加路由器 logit
                total_expert_indexes.append(expert_indexes)  # 添加专家索引
        return torch.cat(total_router_logits, dim=1), torch.cat(total_expert_indexes, dim=1)  # 拼接并返回路由器 logit 和专家索引

    # 准备用于生成的输入
    def prepare_inputs_for_generation(
        self,
        input_ids,  # 输入的 token ID
        past_key_values=None,  # 过去的键值
        attention_mask=None,  # 注意力掩码
        head_mask=None,  # 多头注意力的掩码
        decoder_head_mask=None,  # 解码器多头注意力的掩码
        cross_attn_head_mask=None,  # 跨模块的多头注意力的掩码
        use_cache=None,  # 是否使用缓存
        encoder_outputs=None,  # 编码器的输出
        **kwargs,  # 其他参数
    ):
        # 如果使用了过去的键值
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]  # 获取过去的长度

            # 一些生成方法已经传递了最后一个输入 ID
            if input_ids.shape[1] > past_length:  # 判断输入的形状是否大于过去的长度
                remove_prefix_length = past_length  # 设置移除前缀的长度为过去的长度
            else:
                # 默认保留只有最终 ID
                remove_prefix_length = input_ids.shape[1] - 1  # 设置移除前缀的长度为输入形状的长度减1

            input_ids = input_ids[:, remove_prefix_length:]  # 截取输入

        return {
            "decoder_input_ids": input_ids,  # 返回解码器的输入
            "past_key_values": past_key_values,  # 返回过去的键值
            "encoder_outputs": encoder_outputs,  # 返回编码器的输出
            "attention_mask": attention_mask,  # 返回注意力掩码
            "head_mask": head_mask,  # 返回多头注意力的掩码
            "decoder_head_mask": decoder_head_mask,  # 返回解码器多头注意力的掩码
            "cross_attn_head_mask": cross_attn_head_mask,  # 返回跨模块的多头注意力的掩码
            "use_cache": use_cache,  # 返回是否使用缓存
        }

    # 从标签准备解码器输入的方法
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)  # 返回右移的标签
    # 重新排序解码器的缓存数据，并返回重新排序后的键/值数据
    def _reorder_cache(self, past_key_values, beam_idx):
        # 如果解码器的过去数据没有包含在输出中
        # 快速解码被禁用，无需重新排序
        if past_key_values is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values

        # 重新排序后的解码器过去数据
        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # 从层过去数据的批次维度中获取正确的批次索引
            # `past`的批次维度在第二个位置
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # 需要为四个键/值状态设置正确的`past`
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            if reordered_layer_past_states[0].shape != layer_past_states[0].shape:
                raise ValueError(
                    "expected reordered_layer_past_states to have the same shape than layer_past_states, "
                    f"but got {reordered_layer_past_states[0].shape} and {layer_past_states[0].shape}"
                )
            if len(reordered_layer_past_states) != len(layer_past_states):
                raise ValueError(
                    "expected layer_past_states to have the same length as reordered_layer_past_states, "
                    f"but got {len(layer_past_states)} and {len(reordered_layer_past_states)}"
                )

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past
# 导入必要的模块
@add_start_docstrings(
    # 添加模型文档的起始部分
    "The bare SWITCH_TRANSFORMERS Model transformer outputting encoder's raw hidden-states without any specific head"
    " on top.",
    # 添加与SWITCH_TRANSFORMERS相关的文档
    SWITCH_TRANSFORMERS_START_DOCSTRING,
)
class SwitchTransformersEncoderModel(SwitchTransformersPreTrainedModel):
    # 初始化类变量 _tied_weights_keys，用于指定哪些权重应该被绑定
    _tied_weights_keys = ["encoder.embed_tokens.weight"]

    # 初始化方法，接受一个 SwitchTransformersConfig 类的参数
    def __init__(self, config: SwitchTransformersConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建共享的词嵌入层
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # 复制配置以便修改而不影响原始配置
        encoder_config = copy.deepcopy(config)
        # 禁用缓存以节省内存
        encoder_config.use_cache = False
        # 设置模型为编码器模型
        encoder_config.is_encoder_decoder = False
        # 创建编码器部分
        self.encoder = SwitchTransformersStack(encoder_config, self.shared)

        # 初始化权重并应用最终处理
        self.post_init()

        # 模型并行计算
        self.device_map = None

    # 获取输入词嵌入
    def get_input_embeddings(self):
        return self.shared

    # 设置输入词嵌入
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    # 绑定权重
    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)

    # 获取编码器
    def get_encoder(self):
        return self.encoder

    # 剪枝模型中的注意力头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.block[layer].layer[0].SelfAttention.prune_heads(heads)

    # 模型前向传播方法
    @add_start_docstrings_to_model_forward(SWITCH_TRANSFORMERS_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MoEModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = True,
        return_dict: Optional[bool] = None,
):
    # 定义函数，接收输入的参数 input_ids（输入的文本序列）、attention_mask（注意力掩码）、inputs_embeds（输入的嵌入向量）、head_mask（头部掩码）、output_attentions（是否输出注意力分数）、output_hidden_states（是否输出隐藏状态）、output_router_logits（是否输出路由器 logits）、return_dict（是否返回字典格式的输出）
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        output_router_logits=None,
        return_dict=None,
    ) -> Union[Tuple[torch.FloatTensor], MoEModelOutput]:
        # 设置 return_dict 的默认值为配置文件中的 use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # 将输入的参数传入编码器模型进行处理，并输出编码器的输出
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
        )
    
        # 返回编码器的输出
        return encoder_outputs
```