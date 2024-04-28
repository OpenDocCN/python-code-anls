# `.\models\imagegpt\modeling_imagegpt.py`

```
# 设置文件编码为 UTF-8
# 版权声明，声明代码作者和许可证信息
# 根据 Apache 许可证 2.0 版本，使用此文件需要遵守许可证规定
# 可以在 http://www.apache.org/licenses/LICENSE-2.0 获取许可证副本
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”基础分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关特定语言的权限和限制
"""PyTorch OpenAI ImageGPT model."""

# 导入所需的库和模块
import math
import os
import warnings
from typing import Any, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入自定义的激活函数映射表和模型输出类
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
)
# 导入模型工具类和配置类
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_imagegpt import ImageGPTConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置
_CHECKPOINT_FOR_DOC = "openai/imagegpt-small"
_CONFIG_FOR_DOC = "ImageGPTConfig"

# 预训练模型存档列表
IMAGEGPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "openai/imagegpt-small",
    "openai/imagegpt-medium",
    "openai/imagegpt-large",
    # 查看所有 Image GPT 模型 https://huggingface.co/models?filter=imagegpt
]

# 加载 TensorFlow 模型权重到 PyTorch 模型
def load_tf_weights_in_imagegpt(model, config, imagegpt_checkpoint_path):
    """
    Load tf checkpoints in a pytorch model
    """
    try:
        import re
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(imagegpt_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # 从 TF 模型加载权重
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []

    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array.squeeze())

    return model

# 自定义的 ImageGPT 层归一化类
class ImageGPTLayerNorm(nn.Module):
    def __init__(self, hidden_size: Tuple[int], eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.Tensor(hidden_size))
    # 定义一个前向传播函数，接受一个张量作为输入，并返回一个元组
    def forward(self, tensor: torch.Tensor) -> tuple:
        # 输入张量未经过均值中心化处理
        # 对输入张量进行操作，除以标准差，乘以权重数据
        return (
            tensor
            / torch.sqrt(torch.mean(torch.square(tensor), axis=-1, keepdim=True) + self.eps)
            * self.weight.data[..., :]
        )
class ImageGPTAttention(nn.Module):
    # 定义一个名为ImageGPTAttention的类，继承自nn.Module
    def __init__(self, config, is_cross_attention: Optional[bool] = False, layer_idx: Optional[int] = None):
        # 初始化函数，接受config、is_cross_attention和layer_idx等参数
        super().__init__()
        # 调用父类的初始化函数

        max_positions = config.max_position_embeddings
        # 从config中获取最大位置嵌入的值
        self.register_buffer(
            "bias",
            # 注册一个缓冲区，用于存储下三角矩阵的布尔值
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        # 初始化bias缓冲区，存储下三角矩阵的布尔值
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)
        # 注册一个缓冲区，存储-1e4的张量值

        self.embed_dim = config.hidden_size
        # 从config中获取隐藏大小
        self.num_heads = config.num_attention_heads
        # 从config中获取注意力头的数量
        self.head_dim = self.embed_dim // self.num_heads
        # 计算每个注意力头的维度
        self.split_size = self.embed_dim
        # 初始化分割大小为隐藏大小
        if self.head_dim * self.num_heads != self.embed_dim:
            # 如果隐藏大小不能被注意力头数量整除，则抛出错误
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        # 从config中获取是否缩放注意力权重的标志
        self.is_cross_attention = is_cross_attention
        # 是否为跨注意力的标志

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        # 从config中获取是否按层缩放注意力的标志
        self.layer_idx = layer_idx
        # 层索引
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn
        # 重新排序和上升注意力

        if self.is_cross_attention:
            # 如果是跨注意力
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            # 初始化跨注意力的卷积层
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
            # 初始化查询注意力的卷积层
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
            # 初始化普通注意力的卷积层
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)
        # 初始化卷积层

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        # 初始化注意力的dropout层
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # 初始化残差的dropout层

        self.pruned_heads = set()
        # 初始化一个空集合用于存储被剪枝的注意力头

    def prune_heads(self, heads):
        # 定义一个方法用于剪枝注意力头
        if len(heads) == 0:
            return
        # 如果没有需要剪枝的头，则直接返回
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, self.head_dim, self.pruned_heads)
        # 找到可剪枝的头和索引
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])
        # 拼接注意力索引

        # Prune conv1d layers
        # 剪枝卷积层
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        # 剪枝注意力卷积层
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)
        # 剪枝投影卷积层

        # Update hyper params
        # 更新超参数
        self.split_size = (self.split_size // self.num_heads) * (self.num_heads - len(heads))
        # 更新分割大小
        self.num_heads = self.num_heads - len(heads)
        # 更新注意力头数量
        self.pruned_heads = self.pruned_heads.union(heads)
        # 更新被剪枝的注意力头集合
    # 计算注意力权重，通过将查询和键相乘得到
    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    # 如果需要对注意力权重进行缩放
    if self.scale_attn_weights:
        attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

    # 如果需要对注意力权重进行按层缩放
    if self.scale_attn_by_inverse_layer_idx:
        attn_weights = attn_weights / float(self.layer_idx + 1)

    # 如果不是跨注意力，实现因果掩码
    if not self.is_cross_attention:
        # 获取查询和键的长度
        query_length, key_length = query.size(-2), key.size(-2)
        # 创建因果掩码
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
        # 应用因果掩码
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)

    # 如果存在注意力掩码，则应用
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # 对注意力权重进行 softmax 操作
    attn_weights = nn.Softmax(dim=-1)(attn_weights)

    # 将注意力权重的数据类型转换为 value 的数据类型
    attn_weights = attn_weights.type(value.dtype)
    # 对注意力权重进行 dropout 操作
    attn_weights = self.attn_dropout(attn_weights)

    # 如果需要对头部进行掩码
    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    # 计算注意力输出
    attn_output = torch.matmul(attn_weights, value)

    return attn_output, attn_weights
    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None):
        # 使用 `torch.baddbmm`（与 Megatron-LM 中的 alpha 参数一起更有效率）
        bsz, num_heads, q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = key.size()

        # 为 `baddbmm` 预先分配 attn_weights
        attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)

        # 计算缩放因子
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        # 上转型（关闭自动转型）和重新排序（通过 1 / root(dk) 缩放 K）
        with autocast(enabled=False):
            q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
            attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

        if not self.is_cross_attention:
            # 如果只有“正常”注意力层实现因果掩码
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # 需要是一个张量，否则会出现错误：`RuntimeError: expected scalar type float but found double`。
            # 需要在相同设备上，否则会出现 `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # 应用注意力掩码
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.Softmax(dim=-1)(attn_weights)

        # 如果需要，将上转型（如果是混合精度）回到 V 的数据类型（如果不是则不执行任何操作）
        if attn_weights.dtype != torch.float32:
            raise RuntimeError("Error with upcasting, attn_weights does not have dtype torch.float32")
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # 如果需要，对头部进行掩码
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        将 hidden_size 维度拆分为 attn_head_size 和 num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        # 将张量的维度重新排列，将 num_heads 和 attn_head_size 维度合并到 hidden_size 中
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> tuple:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `ImageGPTAttention(..., is_cross_attention=True)`."
                )

            # 如果存在 encoder_hidden_states，则执行以下操作
            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            # 如果不存在 encoder_hidden_states，则执行以下操作
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # 返回输出元组 (attn_output, present, (attentions))
class ImageGPTMLP(nn.Module):
    # 定义一个名为ImageGPTMLP的类，继承自nn.Module
    def __init__(self, intermediate_size, config):
        # 初始化函数，接受intermediate_size和config两个参数
        super().__init__()
        # 调用父类的初始化函数
        embed_dim = config.hidden_size
        # 从config中获取hidden_size作为embed_dim
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        # 创建一个Conv1D对象，输入参数为intermediate_size和embed_dim，赋值给self.c_fc
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        # 创建一个Conv1D对象，输入参数为embed_dim和intermediate_size，赋值给self.c_proj
        self.act = ACT2FN[config.activation_function]
        # 从ACT2FN字典中获取config.activation_function对应的激活函数，赋值给self.act
        self.dropout = nn.Dropout(config.resid_pdrop)
        # 创建一个Dropout对象，输入参数为config.resid_pdrop，赋值给self.dropout

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 前向传播函数，接受一个torch.Tensor类型的hidden_states参数，返回一个torch.Tensor类型的结果
        hidden_states = self.c_fc(hidden_states)
        # 将hidden_states输入self.c_fc中进行计算
        hidden_states = self.act(hidden_states)
        # 将激活函数act应用到hidden_states上
        hidden_states = self.c_proj(hidden_states)
        # 将hidden_states输入self.c_proj中进行计算
        hidden_states = self.dropout(hidden_states)
        # 对hidden_states应用dropout
        return hidden_states
        # 返回计算结果


class ImageGPTBlock(nn.Module):
    # 定义一个名为ImageGPTBlock的类，继承自nn.Module
    def __init__(self, config, layer_idx=None):
        # 初始化函数，接受config和layer_idx两个参数
        super().__init__()
        # 调用父类的初始化函数
        hidden_size = config.hidden_size
        # 从config中获取hidden_size，赋值给hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size
        # 根据config中的n_inner是否为None来确定inner_dim的值

        self.ln_1 = ImageGPTLayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        # 创建一个ImageGPTLayerNorm对象，输入参数为hidden_size和config.layer_norm_epsilon，赋值给self.ln_1
        self.attn = ImageGPTAttention(config, layer_idx=layer_idx)
        # 创建一个ImageGPTAttention对象，输入参数为config和layer_idx，赋值给self.attn
        self.ln_2 = ImageGPTLayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        # 创建一个ImageGPTLayerNorm对象，输入参数为hidden_size和config.layer_norm_epsilon，赋值给self.ln_2

        if config.add_cross_attention:
            # 如果config中有add_cross_attention属性
            self.crossattention = ImageGPTAttention(config, is_cross_attention=True, layer_idx=layer_idx)
            # 创建一个ImageGPTAttention对象，输入参数为config、is_cross_attention=True和layer_idx，赋值给self.crossattention
            self.ln_cross_attn = ImageGPTLayerNorm(hidden_size, eps=config.layer_norm_epsilon)
            # 创建一个ImageGPTLayerNorm对象，输入参数为hidden_size和config.layer_norm_epsilon，赋值给self.ln_cross_attn

        self.mlp = ImageGPTMLP(inner_dim, config)
        # 创建一个ImageGPTMLP对象，输入参数为inner_dim和config，赋值给self.mlp

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        # 前向传播函数，接受多个参数
    # 定义函数的返回类型为元组
    ) -> tuple:
        # 保存隐藏状态的副本
        residual = hidden_states
        # 对隐藏状态进行 LayerNormalization
        hidden_states = self.ln_1(hidden_states)
        # 进行自注意力机制计算
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # 获取自注意力机制的输出
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        # 保存除了自注意力机制输出之外的其他输出
        outputs = attn_outputs[1:]
        # 使用残差连接更新隐藏状态
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # 添加一个用于交叉注意力的自注意力块
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            # 保存隐藏状态的副本
            residual = hidden_states
            # 对隐藏状态进行 LayerNormalization
            hidden_states = self.ln_cross_attn(hidden_states)
            # 进行交叉注意力机制计算
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            # 获取交叉注意力机制的输出
            attn_output = cross_attn_outputs[0]
            # 使用残差连接更新隐藏状态
            hidden_states = residual + attn_output
            # 如果需要输出注意力权重，则添加交叉注意力的输出
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        # 保存隐藏状态的副本
        residual = hidden_states
        # 对隐藏状态进行 LayerNormalization
        hidden_states = self.ln_2(hidden_states)
        # 使用多层感知机进行前馈计算
        feed_forward_hidden_states = self.mlp(hidden_states)
        # 使用残差连接更新隐藏状态
        hidden_states = residual + feed_forward_hidden_states

        # 根据是否使用缓存，更新输出
        outputs = (hidden_states,) + (outputs if use_cache else outputs[1:])

        # 返回输出结果，包括隐藏状态、present、(attentions, cross_attentions)
        return outputs  # hidden_states, present, (attentions, cross_attentions)
class ImageGPTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置配置类为ImageGPTConfig
    config_class = ImageGPTConfig
    # 加载TF权重的函数为load_tf_weights_in_imagegpt
    load_tf_weights = load_tf_weights_in_imagegpt
    # 基础模型前缀为"transformer"
    base_model_prefix = "transformer"
    # 主输入名称为"input_ids"
    main_input_name = "input_ids"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def __init__(self, *inputs, **kwargs):
        # 调用父类的初始化函数
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        # 如果模块是线性层或Conv1D层
        if isinstance(module, (nn.Linear, Conv1D)):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置，则初始化为0
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是嵌入层
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在填充索引，则将对应权重初始化为0
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果模块是ImageGPTLayerNorm
        elif isinstance(module, ImageGPTLayerNorm):
            # 将权重初始化为1
            module.weight.data.fill_(1.0)

        # 重新初始化选定的权重，遵循OpenAI GPT-2 Paper Scheme
        for name, p in module.named_parameters():
            if "c_proj" in name and "weight" in name:
                # 特殊的缩放初始化，每个Transformer块有2个Layer Norms
                p.data.normal_(mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.n_layer)))


IMAGEGPT_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`ImageGPTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
# 图像GPT模型的输入文档字符串
IMAGEGPT_INPUTS_DOCSTRING = r"""
"""

# 定义图像GPT模型，输出原始隐藏状态，没有特定的头部
@add_start_docstrings(
    "The bare ImageGPT Model transformer outputting raw hidden-states without any specific head on top.",
    IMAGEGPT_START_DOCSTRING,
)
class ImageGPTModel(ImageGPTPreTrainedModel):
    def __init__(self, config: ImageGPTConfig):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        # 词嵌入层
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        # 多层ImageGPTBlock
        self.h = nn.ModuleList([ImageGPTBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = ImageGPTLayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # 模型并行
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    # 前向传播函数
    @add_start_docstrings_to_model_forward(IMAGEGPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPastAndCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any,
@add_start_docstrings(
    """
    The ImageGPT Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    IMAGEGPT_START_DOCSTRING,
)
class ImageGPTForCausalImageModeling(ImageGPTPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    # 初始化函数，接受一个 ImageGPTConfig 类型的参数
    def __init__(self, config: ImageGPTConfig):
        # 调用父类的初始化函数
        super().__init__(config)
        # 创建一个 ImageGPTModel 对象
        self.transformer = ImageGPTModel(config)
        # 创建一个线性层，用于输出
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size - 1, bias=False)

        # Model parallel
        # 是否使用模型并行
        self.model_parallel = False
        # 设备映射
        self.device_map = None
        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 为生成准备输入
    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, past_key_values: Optional[bool] = None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # 忽略过去键值覆盖的标记
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法已经只传递最后一个输入 ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认为旧行为：仅保留最后一个 ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # 为批量生成动态创建 position_ids
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    # 添加模型前向传播的文档字符串
    @add_start_docstrings_to_model_forward(IMAGEGPT_INPUTS_DOCSTRING)
    # 替换返回文档字符串
    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    # 定义一个前向传播函数，接受多个输入参数，包括输入的张量、过去的键值对、注意力掩码、token类型ID、位置ID、头部掩码、输入嵌入、编码器隐藏状态、编码器注意力掩码、标签、是否使用缓存、是否输出注意力、是否输出隐藏状态、是否返回字典等
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any,
    # 静态方法，用于重新排序缓存中的过去键值对，以匹配每一代步骤的正确beam_idx
    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        # 返回重新排序后的过去键值对缓存
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )
# 定义一个带有图像分类头部的 ImageGPT 模型转换器，使用线性层进行分类
# ImageGPTForImageClassification 对隐藏状态进行平均池化以进行分类
class ImageGPTForImageClassification(ImageGPTPreTrainedModel):
    def __init__(self, config: ImageGPTConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置标签数量
        self.num_labels = config.num_labels
        # 初始化 ImageGPTModel
        self.transformer = ImageGPTModel(config)
        # 创建一个线性层用于分类，输入维度为 config.n_embd，输出维度为标签数量，无偏置
        self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any,
```