# `.\models\decision_transformer\modeling_decision_transformer.py`

```py
# 设置文件编码格式为utf-8
# 版权声明
# 根据Apache许可证2.0版本，除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则软件将按"原样"方式分发
# 没有任何明示或暗示的担保或条件，查看特定语言管理权限和约束
"""
PyTorch决策Transformer模型。
"""

# 导入所需的模块和库
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.cuda.amp import autocast

# 从transformers模块中导入其他模块或函数
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

# 获取日志记录器
logger = logging.get_logger(__name__)

# 为文档准备的检查点和配置信息
_CHECKPOINT_FOR_DOC = "edbeeching/decision-transformer-gym-hopper-medium"
_CONFIG_FOR_DOC = "DecisionTransformerConfig"

# 预训练模型存档列表
DECISION_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "edbeeching/decision-transformer-gym-hopper-medium",
    # 查看所有的DecisionTransformer模型
    # https://huggingface.co/models?filter=decision_transformer
]

# 从transformers.models.gpt2.modeling_gpt2.load_tf_weights_in_gpt2复制过来的函数
def load_tf_weights_in_gpt2(model, config, gpt2_checkpoint_path):
    """在pytorch模型中加载tf检查点"""
    try:
        import re
        import tensorflow as tf
    except ImportError:
        # 加载PyTorch中的TensorFlow模型需要安装TensorFlow，请参考https://www.tensorflow.org/install/进行安装
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    # 将tf路径转换为绝对路径
    tf_path = os.path.abspath(gpt2_checkpoint_path)
    # 输出转换过的TensorFlow检查点信息
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # 从TF模型加载权重
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        # 输出加载的TF权重信息
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array.squeeze())
    # 遍历输入的名称和数组
    for name, array in zip(names, arrays):
        # 跳过"model/"前缀
        name = name[6:]
        # 按"/"分割名称
        name = name.split("/")
        # 初始化指针指向模型
        pointer = model
        # 遍历文件名
        for m_name in name:
            # 检查文件名是否符合格式，将非数字的部分分离出来
            if re.fullmatch(r"[A-Za-z]+\d+", m_name):
                scope_names = re.split(r"(\d+)", m_name)
            else:
                scope_names = [m_name]
            # 根据文件名前缀设置指针位置
            if scope_names[0] == "w" or scope_names[0] == "g":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "b":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "wpe" or scope_names[0] == "wte":
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, "weight")
            else:
                pointer = getattr(pointer, scope_names[0])
            # 如果文件名含有数字索引，移动指针到对应位置
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        # 检查当前指针指向的形状是否与数组形状相匹配
        try:
            if pointer.shape != array.shape:
                raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
        except ValueError as e:
            # 更新异常信息
            e.args += (pointer.shape, array.shape)
            # 重新抛出异常
            raise
        # 输出初始化 PyTorch 权重的日志
        logger.info(f"Initialize PyTorch weight {name}")
        # 将数组转换为 PyTorch 张量，赋值给指针
        pointer.data = torch.from_numpy(array)
    # 返回模型
    return model
# 在 transformers.models.gpt2.modeling_gpt2.GPT2Attention 的基础上复制，将 GPT2 改为 DecisionTransformerGPT2
class DecisionTransformerGPT2Attention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()

        max_positions = config.max_position_embeddings
        # 注册一个缓冲区变量 "bias"，包含一个下三角矩阵的上三角部分，数据类型为 bool
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        # 注册一个持久化缓冲区变量 "masked_bias"，包含值为 -1e4 的张量
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

        self.embed_dim = config.hidden_size  # 获取 config 中的 hidden_size 属性值作为 embed_dim
        self.num_heads = config.num_attention_heads  # 获取 config 中的 num_attention_heads 属性值作为 num_heads
        self.head_dim = self.embed_dim // self.num_heads  # 计算 head_dim
        self.split_size = self.embed_dim  # 将 embed_dim 赋值给 split_size
        # 如果 embed_dim 不能被 num_heads 整除，抛出异常
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        # 从 config 中获取 scale_attn_weights 和其他属性值
        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        # 从 config 中获取 scale_attn_by_inverse_layer_idx、layer_idx 和 reorder_and_upcast_attn 作为属性值
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        # 如果是跨注意力的，创建 c_attn 和 q_attn
        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)  # 创建 c_proj

        self.attn_dropout = nn.Dropout(config.attn_pdrop)  # 创建 attn_dropout 层
        self.resid_dropout = nn.Dropout(config.resid_pdrop)  # 创建 resid_dropout 层

        self.pruned_heads = set()  # 创建一个空集合用于存储被修剪的头部信息

    # 修剪头部
    def prune_heads(self, heads):
        if len(heads) == 0:  # 如果头部数为0，则直接返回
            return
        # 找到可修剪的头部并返回头部索引
        heads, index = find_pruneable_heads_and_indices(heads, self.num_heads, self.head_dim, self.pruned_heads)
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])

        # 修剪 conv1d 层
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)

        # 更新超参数
        self.split_size = (self.split_size // self.num_heads) * (self.num_heads - len(heads))
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)
    # 实现注意力机制，计算查询，键，值之间的注意力权重
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # 使用矩阵乘法计算查询和键的转置之间的注意力权重
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        # 如果需要，对注意力权重进行缩放
        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        # 如果需要，对注意力权重按照层索引进行缩放
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        # 如果不是跨层注意力，实现因果掩码
        if not self.is_cross_attention:
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        # 如果有注意力掩码，应用该掩码
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # 使用 softmax 函数计算最终的注意力权重
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # 将注意力权重类型转换为值的类型，如果是混合精度则下降
        attn_weights = attn_weights.type(value.dtype)
        # 对注意力权重进行丢弃操作
        attn_weights = self.attn_dropout(attn_weights)

        # 如果需要，对注意力权重进行头遮罩
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        # 计算最终的注意力输出
        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights
    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None):
        # 使用 torch.baddbmm 进行矩阵相加操作，更高效，可使用 alpha 参数进行缩放（来自 Megatron-LM）
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

        # 升级（关闭自动混合精度）和重新排序（通过 1 / 根号dk 缩放 K）
        with autocast(enabled=False):
            q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
            attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

        if not self.is_cross_attention:
            # 如果只有“正常”注意力层实现因果掩模
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # 需要是张量，否则会出错：“RuntimeError: 期望标量类型为float但发现double”。
            # 需要在相同设备上，否则会出错：“RuntimeError: ...，x 和 y 需要在同一设备上”
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # 应用注意力掩模
            attn_weights = attn_weights + attention_mask

        # 对 attn_weights 执行 softmax 操作
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # 将必要时下放（如混合精度）回到 V 的数据类型（如果不是则不操作）
        if attn_weights.dtype != torch.float32:
            raise RuntimeError("升级出错，attn_weights 的数据类型不是 torch.float32")
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # 如果需要，对头进行掩蔽
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        将 hidden_size 维度分割为 attn_head_size 和 num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        # 将张量维度重新排列，将 attn_head_size 和 num_attn_heads 维度合并到 hidden_size
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        # 计算新的形状
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        # 重新调整张量形状
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        # 如果存在编码器隐藏状态
        if encoder_hidden_states is not None:
            # 如果类被用作跨注意力，确保 q_attn 权重已定义
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `DecisionTransformerGPT2Attention(..., is_cross_attention=True)`."
                )
            # 使用 self.q_attn 对隐藏状态进行查询操作
            query = self.q_attn(hidden_states)
            # 使用 self.c_attn 对编码器隐藏状态进行键值分离操作
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            # 注意力掩码使用编码器注意力掩码
            attention_mask = encoder_attention_mask
        else:
            # 使用 self.c_attn 对隐藏状态进行键值分离操作
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        # 将查询、键、值按照头数和头维度分割
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # 如果存在过去的层，将过去的键和值与当前的键和值拼接
        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        # 如果使用缓存，则保存当前的键和值
        if use_cache is True:
            present = (key, value)
        else:
            present = None

        # 如果重新排列和升级注意力机制开启，则调用_upcast_and_reordered_attn方法
        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            # 否则调用标准的注意力机制
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # 合并多头注意力输出的头
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        # 对合并后的注意力输出进行投影
        attn_output = self.c_proj(attn_output)
        # 对注意力输出进行残差连接和 dropout
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        # 如果输出注意力权重，则将其加入输出
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)
# 从transformers.models.gpt2.modeling_gpt2.GPT2MLP复制并修改为DecisionTransformerGPT2MLP类
class DecisionTransformerGPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)  # 基于输入维度和输出维度创建卷积层对象
        self.c_proj = Conv1D(embed_dim, intermediate_size)  # 基于输入维度和输出维度创建卷积层对象
        self.act = ACT2FN[config.activation_function]  # 根据配置激活函数名称选择相应的激活函数
        self.dropout = nn.Dropout(config.resid_pdrop)  # 使用配置中的dropout概率创建一个dropout层对象

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)  # 将hidden_states输入给卷积层
        hidden_states = self.act(hidden_states)  # 使用激活函数处理hidden_states
        hidden_states = self.c_proj(hidden_states)  # 将处理后的hidden_states再输入给另一个卷积层
        hidden_states = self.dropout(hidden_states)  # 使用dropout层处理hidden_states
        return hidden_states


# 从transformers.models.gpt2.modeling_gpt2.GPT2Block复制并修改为DecisionTransformerGPT2Block类
class DecisionTransformerGPT2Block(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)  # 使用配置创建LayerNorm对象
        self.attn = DecisionTransformerGPT2Attention(config, layer_idx=layer_idx)  # 添加DecisionTransformerGPT2Attention层
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)  # 使用配置创建LayerNorm对象

        if config.add_cross_attention:
            self.crossattention = DecisionTransformerGPT2Attention(
                config, is_cross_attention=True, layer_idx=layer_idx
            )
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)  # 使用配置创建LayerNorm对象

        self.mlp = DecisionTransformerGPT2MLP(inner_dim, config)  # 创建DecisionTransformerGPT2MLP对象

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    # 定义函数，接受隐藏状态、先前的层的输出（如果有）、注意力遮罩、注意力掩码头模、是否使用缓存、是否输出注意力矩阵参数以及交叉注意力时的编码器隐藏状态作为输入，
    # 返回一个元组，其中包含隐藏状态和可能的注意力矩阵参数。
    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = True,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        # 保存残差连接的隐藏状态
        residual = hidden_states
        # 应用层归一化
        hidden_states = self.ln_1(hidden_states)
        # 使用self-attention层计算注意力输出
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # 获得注意力输出（attn_output），剩余输出(outputs)
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        # 添加残差连接
        hidden_states = attn_output + residual
    
        if encoder_hidden_states is not None:
            # 对于交叉注意力，增加一个self-attention块
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            # 保存残差连接的隐藏状态
            residual = hidden_states
            # 应用层归一化
            hidden_states = self.ln_cross_attn(hidden_states)
            # 使用cross-attention层计算交叉注意力输出
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            # 获得交叉注意力输出
            attn_output = cross_attn_outputs[0]
            # 添加残差连接
            hidden_states = residual + attn_output
            # 添加交叉注意力输出
            outputs = outputs + cross_attn_outputs[2:]
    
        # 保存残差连接的隐藏状态
        residual = hidden_states
        # 应用层归一化
        hidden_states = self.ln_2(hidden_states)
        # 使用前馈神经网络计算前馈隐藏层的输出
        feed_forward_hidden_states = self.mlp(hidden_states)
        # 添加残差连接
        hidden_states = residual + feed_forward_hidden_states
    
        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]
    
        # 返回隐藏状态和可能的注意力矩阵参数
        return outputs
class DecisionTransformerGPT2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DecisionTransformerConfig  # 设置属性 config_class 为 DecisionTransformerConfig
    load_tf_weights = load_tf_weights_in_gpt2   # 设置属性 load_tf_weights 为 load_tf_weights_in_gpt2
    base_model_prefix = "transformer"   # 设置属性 base_model_prefix 为 "transformer"
    is_parallelizable = True   # 设置属性 is_parallelizable 为 True
    supports_gradient_checkpointing = True   # 设置属性 supports_gradient_checkpointing 为 True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)   # 调用父类的构造函数

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, Conv1D)):   # 如果 module 的类型是 nn.Linear 或者 Conv1D
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)  # 对 weight 进行正态分布初始化
            if module.bias is not None:   # 如果 module 存在 bias
                module.bias.data.zero_()   # 将 bias 初始化为 0
        elif isinstance(module, nn.Embedding):   # 如果 module 的类型是 nn.Embedding
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)  # 对 weight 进行正态分布初始化
            if module.padding_idx is not None:   # 如果 module 存在 padding_idx
                module.weight.data[module.padding_idx].zero_()   # 将 padding_idx 对应的 weight 初始化为 0
        elif isinstance(module, nn.LayerNorm):   # 如果 module 的类型是 nn.LayerNorm
            module.bias.data.zero_()   # 将 bias 初始化为 0
            module.weight.data.fill_(1.0)   # 将 weight 初始化为 1.0

        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():   # 对 module 中的参数进行遍历
            if "c_proj" in name and "weight" in name:   # 如果参数名包含 "c_proj" 和 "weight"
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.n_layer)))   # 对参数进行特殊的正态分布初始化

class DecisionTransformerGPT2Model(DecisionTransformerGPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)   # 调用父类的构造函数

        self.embed_dim = config.hidden_size   # 设置属性 embed_dim 为 config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)   # 初始化 wte 为 nn.Embedding 对象
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)   # 初始化 wpe 为 nn.Embedding 对象

        self.drop = nn.Dropout(config.embd_pdrop)   # 初始化 drop 为 nn.Dropout 对象
        self.h = nn.ModuleList(   # 初始化 h 为 nn.ModuleList 对象，包含多个 DecisionTransformerGPT2Block 对象
            [DecisionTransformerGPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)   # 初始化 ln_f 为 nn.LayerNorm 对象

        # Model parallel
        self.model_parallel = False   # 设置属性 model_parallel 为 False
        self.device_map = None   # 设置属性 device_map 为 None
        self.gradient_checkpointing = False   # 设置属性 gradient_checkpointing 为 False

        # Initialize weights and apply final processing
        self.post_init()
    # 获取输入词嵌入矩阵
    def get_input_embeddings(self):
        return self.wte

    # 设置输入词嵌入矩阵
    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    # 从transformers.models.gpt2.modeling_gpt2.GPT2Model.forward中复制的前向传播函数
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的标识符序列
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,  # 过去的键值对
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力掩码
        token_type_ids: Optional[torch.LongTensor] = None,  # 标识符类型
        position_ids: Optional[torch.LongTensor] = None,  # 位置标识符
        head_mask: Optional[torch.FloatTensor] = None,  # 头部掩码
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入嵌入
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 编码器隐藏状态
        encoder_attention_mask: Optional[torch.FloatTensor] = None,  # 编码器注意力掩码
        use_cache: Optional[bool] = None,  # 使用缓存
        output_attentions: Optional[bool] = None,  # 输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 输出隐藏状态
        return_dict: Optional[bool] = None,  # 返回字典
# 使用 dataclass 装饰器，声明 DecisionTransformerOutput 类，它是 ModelOutput 的子类
@dataclass
class DecisionTransformerOutput(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.
    Args: 
        # last_hidden_state: 模型最后一层的隐藏状态，形状为(batch_size, sequence_length, hidden_size)
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        # state_preds: 环境状态预测，形状为(batch_size, sequence_length, state_dim)
        state_preds (`torch.FloatTensor` of shape `(batch_size, sequence_length, state_dim)`):
            Environment state predictions
        # action_preds: 模型动作预测, 形状为(batch_size, sequence_length, action_dim)
        action_preds (`torch.FloatTensor` of shape `(batch_size, sequence_length, action_dim)`):
            Model action predictions
        # return_preds: 每个状态的预测收益，形状为(batch_size, sequence_length, 1)
        return_preds (`torch.FloatTensor` of shape `(batch_size, sequence_length, 1)`):
            Predicted returns for each state
        # hidden_states: 每个层的隐藏状态，形状为(batch_size, sequence_length, hidden_size)
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)'.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        # attentions: 注意力权重，形状为(batch_size, num_heads, sequence_length, sequence_length)
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,sequence_length)'.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    """

    # 状态预测
    state_preds: torch.FloatTensor = None
    # 动作预测
    action_preds: torch.FloatTensor = None
    # 收益预测
    return_preds: torch.FloatTensor = None
    # 隐藏状态
    hidden_states: torch.FloatTensor = None
    # 注意力权重
    attentions: torch.FloatTensor = None
    # 最后的隐藏状态
    last_hidden_state: torch.FloatTensor = None

# 决策变换预训练模型基类
class DecisionTransformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 决策变换配置类
    config_class = DecisionTransformerConfig  
    # 基础模型前缀
    base_model_prefix = "decision_transformer"  
    # 主输入名称
    main_input_name = "states"  
    # 是否支持梯度检查点
    supports_gradient_checkpointing = False  
    # 对模型的权重进行初始化
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果是线性层
        if isinstance(module, nn.Linear):
            # 与 TF 版本略有不同，使用正态分布进行初始化
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置项，则初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是嵌入层
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有填充索引，则初始化对应位置为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果是 LayerNorm 层
        elif isinstance(module, nn.LayerNorm):
            # 初始化偏置项为零
            module.bias.data.zero_()
            # 初始化权重项为 1.0
            module.weight.data.fill_(1.0)
# 决策 Transformer 模型的文档字符串，描述模型的基本信息和使用方法
DECISION_TRANSFORMER_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~DecisionTransformerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 决策 Transformer 模型的输入文档字符串，描述模型输入的参数和形状
DECISION_TRANSFORMER_INPUTS_DOCSTRING = r"""
    Args:
        states (`torch.FloatTensor` of shape `(batch_size, episode_length, state_dim)`):
            The states for each step in the trajectory
        actions (`torch.FloatTensor` of shape `(batch_size, episode_length, act_dim)`):
            The actions taken by the "expert" policy for the current state, these are masked for auto regressive
            prediction
        rewards (`torch.FloatTensor` of shape `(batch_size, episode_length, 1)`):
            The rewards for each state, action
        returns_to_go (`torch.FloatTensor` of shape `(batch_size, episode_length, 1)`):
            The returns for each state in the trajectory
        timesteps (`torch.LongTensor` of shape `(batch_size, episode_length)`):
            The timestep for each step in the trajectory
        attention_mask (`torch.FloatTensor` of shape `(batch_size, episode_length)`):
            Masking, used to mask the actions when performing autoregressive prediction
"""

# 使用装饰器为 DecisionTransformerModel 类添加文档字符串，描述模型的基本信息和用途
@add_start_docstrings("The Decision Transformer Model", DECISION_TRANSFORMER_START_DOCSTRING)
class DecisionTransformerModel(DecisionTransformerPreTrainedModel):
    """

    The model builds upon the GPT2 architecture to perform autoregressive prediction of actions in an offline RL
    setting. Refer to the paper for more details: https://arxiv.org/abs/2106.01345

    """
    # 初始化函数，接收配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)
        # 保存配置参数
        self.config = config
        # 保存隐藏层的大小
        self.hidden_size = config.hidden_size
        # 创建DecisionTransformerGPT2Model对象作为编码器
        # 注意：这个GPT2Model和Huggingface默认版本唯一的区别是移除了位置嵌入（因为我们将自己添加位置嵌入）
        self.encoder = DecisionTransformerGPT2Model(config)

        # 创建序列步长嵌入
        self.embed_timestep = nn.Embedding(config.max_ep_len, config.hidden_size)
        # 创建返回值嵌入
        self.embed_return = torch.nn.Linear(1, config.hidden_size)
        # 创建状态嵌入
        self.embed_state = torch.nn.Linear(config.state_dim, config.hidden_size)
        # 创建动作嵌入
        self.embed_action = torch.nn.Linear(config.act_dim, config.hidden_size)

        # 创建LayerNorm层
        self.embed_ln = nn.LayerNorm(config.hidden_size)

        # 注意：在论文中我们不预测状态或返回值
        # 创建预测状态的线性层
        self.predict_state = torch.nn.Linear(config.hidden_size, config.state_dim)
        # 创建预测动作的线性层
        self.predict_action = nn.Sequential(
            *([nn.Linear(config.hidden_size, config.act_dim)] + ([nn.Tanh()] if config.action_tanh else []))
        )
        # 创建预测返回值的线性层
        self.predict_return = torch.nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    # 将文档字符串添加到模型的forward函数
    @add_start_docstrings_to_model_forward(DECISION_TRANSFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=DecisionTransformerOutput, config_class=_CONFIG_FOR_DOC)
    # 模型的前向传播函数
    def forward(
        self,
        states: Optional[torch.FloatTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        rewards: Optional[torch.FloatTensor] = None,
        returns_to_go: Optional[torch.FloatTensor] = None,
        timesteps: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
```