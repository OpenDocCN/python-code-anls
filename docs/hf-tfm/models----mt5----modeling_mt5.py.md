# `.\transformers\models\mt5\modeling_mt5.py`

```py
# 指定字符编码格式为utf-8
# 版权信息
""" PyTorch mT5 model."""
# 导入模块
import copy
import math
import os
import warnings
from typing import List, Optional, Tuple, Union
# 导入 torch 库
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
# 导入相关模块
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_fx_proxy,
    logging,
    replace_return_docstrings,
)
from ...utils.model_parallel_utils import assert_device_map, get_device_map
# 导入 MT5Config 配置
from .configuration_mt5 import MT5Config
# 获取日志记录器
logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MT5Config"
_CHECKPOINT_FOR_DOC = "mt5-small"
# 并行说明文档
PARALLELIZE_DOCSTRING = r"""
    This is an experimental feature and is a subject to change at a moment's notice.
    ...
    ```
    model.parallelize(device_map)
    # 使用给定的 device_map 对模型进行并行化处理
"""
DEPARALLELIZE_DOCSTRING = r"""
    Moves the model to cpu from a model parallel state.

    Example:

    ```py
    # On a 4 GPU machine with mt5-xl:
    model = MT5ForConditionalGeneration.from_pretrained("Mt5-xl")
    device_map = {
        0: [0, 1, 2],
        1: [3, 4, 5, 6, 7, 8, 9],
        2: [10, 11, 12, 13, 14, 15, 16],
        3: [17, 18, 19, 20, 21, 22, 23],
    }
    model.parallelize(device_map)  # Splits the model across several devices
    model.deparallelize()  # Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()
    ```
"""


# Copied from transformers.models.t5.modeling_t5.T5LayerNorm with T5->MT5
# MT5 版本的 LayerNorm 模块，不包含偏置和均值的减法
class MT5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        构建 MT5 风格的 LayerNorm 模块。没有偏置和均值减法。
        """
        super().__init__()
        # 初始化权重参数
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # MT5 使用的 LayerNorm 只进行缩放，不进行平移，这也被称作 Root Mean Square Layer Normalization
        # 因此方差是不带均值计算的，也没有偏置。此外，我们要确保对于半精度输入，累积计算是在 fp32 中进行的

        # 计算方差
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        # 进行缩放操作
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # 如果需要，转换为半精度
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


# Copied from transformers.models.t5.modeling_t5.T5DenseActDense with T5->MT5
# MT5 版本的 DenseActDense 模块
class MT5DenseActDense(nn.Module):
    def __init__(self, config: MT5Config):
        super().__init__()
        # 初始化线性层
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


# Copied from transformers.models.t5.modeling_t5.T5DenseGatedActDense with T5->MT5
# MT5 版本的 DenseGatedActDense 模块
class MT5DenseGatedActDense(nn.Module):
    # 初始化函数，接受一个MT5Config类型的配置参数
    def __init__(self, config: MT5Config):
        # 调用父类的初始化函数
        super().__init__()
        # 初始化第一个线性层，输入维度为config.d_model，输出维度为config.d_ff，无偏置
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        # 初始化第二个线性层，输入维度为config.d_model，输出维度为config.d_ff，无偏置
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        # 初始化输出线性层，输入维度为config.d_ff，输出维度为config.d_model，无偏置
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        # 初始化Dropout层，根据配置的dropout_rate进行dropout
        self.dropout = nn.Dropout(config.dropout_rate)
        # 根据配置的激活函数名称从ACT2FN字典中获取对应的激活函数
        self.act = ACT2FN[config.dense_act_fn]

    # 前向传播函数，接受隐藏状态作为输入，返回处理后的隐藏状态
    def forward(self, hidden_states):
        # 使用第一个线性层对隐藏状态进行线性变换，并经过激活函数处理
        hidden_gelu = self.act(self.wi_0(hidden_states))
        # 使用第二个线性层对隐藏状态进行线性变换
        hidden_linear = self.wi_1(hidden_states)
        # 将经过激活函数处理后的隐藏状态与线性变换后的隐藏状态相乘
        hidden_states = hidden_gelu * hidden_linear
        # 对隐藏状态进行dropout处理
        hidden_states = self.dropout(hidden_states)

        # 为了使8位量化适用于google/flan-t5-xxl模型，self.wo保持为float32类型
        # 参考 https://github.com/huggingface/transformers/issues/20287
        # 同时确保权重不是`int8`类型，以防止用户强制将`_keep_in_fp32_modules`设置为`None`
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            # 将隐藏状态转换为与self.wo权重相同的数据类型
            hidden_states = hidden_states.to(self.wo.weight.dtype)

        # 使用输出线性层对隐藏状态进行线性变换
        hidden_states = self.wo(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states
# 从transformers.models.t5.modeling_t5.T5LayerFF中复制代码，并将T5->MT5
class MT5LayerFF(nn.Module):
    def __init__(self, config: MT5Config):
        # 初始化MT5LayerFF类
        super().__init__()
        # 如果配置中有激活门控制，则使用MT5DenseGatedActDense，否则使用MT5DenseActDense
        if config.is_gated_act:
            self.DenseReluDense = MT5DenseGatedActDense(config)
        else:
            self.DenseReluDense = MT5DenseActDense(config)

        # 初始化层归一化和dropout
        self.layer_norm = MT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        # 对输入进行层归一化
        forwarded_states = self.layer_norm(hidden_states)
        # 通过前向传播网络层
        forwarded_states = self.DenseReluDense(forwarded_states)
        # 添加dropout并与原始隐藏状态相加
        hidden_states = hidden_states + self.dropout(forwarded_states)
        # 返回更新后的隐藏状态
        return hidden_states


# 从transformers.models.t5.modeling_t5.T5Attention中复制代码，并将T5->MT5
class MT5Attention(nn.Module):
    def __init__(self, config: MT5Config, has_relative_attention_bias=False):
        # 初始化MT5Attention类
        super().__init__()
        # 设置是否为解码器的标志
        self.is_decoder = config.is_decoder
        # 设置是否具有相对注意力偏置的标志
        self.has_relative_attention_bias = has_relative_attention_bias
        # 设置相对注意力的桶数和最大距离
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # 使用Mesh TensorFlow初始化以避免softmax之前的缩放
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        # 如果具有相对注意力偏置，则初始化嵌入层
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()  # 用于存储被剪枝的注意力头的集合
        self.gradient_checkpointing = False  # 梯度检查点标志，默认为False

    def prune_heads(self, heads):
        # 如果没有头被剪枝，则直接返回
        if len(heads) == 0:
            return
        # 查找可剪枝头和索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
        )
        # 剪枝线性层
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # 更新超参数
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.key_value_proj_dim * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        # 如果是双向的注意力机制，则缩减一半的桶数
        if bidirectional:
            num_buckets //= 2
            # 根据相对位置是正还是负，决定在哪个桶中
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            # 取相对应的位置的绝对值
            relative_position = torch.abs(relative_position)
        else:
            # 如果不是双向，则将相对位置取负值，确保是负数
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # 现在相对位置在[0, inf)的范围内

        # 将桶的一半用于精确增量的位置
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # 另一半桶用于相对于max_distance的对数增量位置
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        # 确保不超过桶数-1
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        # 根据是否是小数，选择使用哪部分桶
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets
    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        # 如果设备参数为 None，则使用相对注意力偏置权重的设备
        if device is None:
            device = self.relative_attention_bias.weight.device
        # 创建表示查询位置的张量，形状为 (query_length, 1)
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        # 创建表示记忆位置的张量，形状为 (1, key_length)
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        # 计算相对位置，形状为 (query_length, key_length)
        relative_position = memory_position - context_position  
        # 将相对位置映射到桶中，形状为 (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # 形状为 (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        # 获取相对注意力偏置值，形状为 (query_length, key_length, num_heads)
        values = self.relative_attention_bias(relative_position_bucket)  
        # 重新排列张量维度，形状为 (1, num_heads, query_length, key_length)
        values = values.permute([2, 0, 1]).unsqueeze(0)  
        # 返回相对位置偏置值
        return values

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
# MT5LayerSelfAttention 是 T5 模型的自注意力层的实现，负责对输入序列进行自注意力计算
class MT5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        # 创建 MT5Attention 模块用于计算自注意力
        self.SelfAttention = MT5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        # 创建 MT5LayerNorm 模块用于层归一化
        self.layer_norm = MT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # 创建 Dropout 模块用于防止过拟合
        self.dropout = nn.Dropout(config.dropout_rate)

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
        # 对输入序列进行层归一化
        normed_hidden_states = self.layer_norm(hidden_states)
        # 计算自注意力输出
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # 将自注意力输出与输入序列相加，并应用 Dropout
        hidden_states = hidden_states + self.dropout(attention_output[0])
        # 返回隐藏状态和可选的注意力权重
        outputs = (hidden_states,) + attention_output[1:]
        return outputs

# MT5LayerCrossAttention 是 T5 模型的交叉注意力层的实现，负责对输入序列和编码器输出进行交叉注意力计算
class MT5LayerCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建 MT5Attention 模块用于计算交叉注意力
        self.EncDecAttention = MT5Attention(config, has_relative_attention_bias=False)
        # 创建 MT5LayerNorm 模块用于层归一化
        self.layer_norm = MT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # 创建 Dropout 模块用于防止过拟合
        self.dropout = nn.Dropout(config.dropout_rate)

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
        # 对输入序列进行层归一化
        normed_hidden_states = self.layer_norm(hidden_states)
        # 计算交叉注意力输出
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
        # 将交叉注意力输出与输入序列相加，并应用 Dropout
        layer_output = hidden_states + self.dropout(attention_output[0])
        # 返回隐藏状态和可选的注意力权重
        outputs = (layer_output,) + attention_output[1:]
        return outputs

# MT5Block 是 T5 模型的一个块，包含了自注意力层和交叉注意力层
class MT5Block(nn.Module):
        # 初始化方法，接受配置和是否存在相对注意力偏置作为参数
    def __init__(self, config, has_relative_attention_bias=False):
        # 调用父类的初始化方法
        super().__init__()
        # 判断是否为解码器
        self.is_decoder = config.is_decoder
        # 初始化一个层的列表
        self.layer = nn.ModuleList()
        # 添加自注意力层
        self.layer.append(MT5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        # 如果是解码器，添加跨层注意力层
        if self.is_decoder:
            self.layer.append(MT5LayerCrossAttention(config))
        # 添加前馈神经网络层
        self.layer.append(MT5LayerFF(config))

    # 前向传播方法
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
        return_dict=True,
# 将 TensorFlow 权重加载到 PyTorch 模型中
def load_tf_weights_in_mt5(model, config, tf_checkpoint_path):
    # 尝试导入 re、numpy 和 TensorFlow 模块
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        # 如果未安装 TensorFlow，输出错误信息并引发异常
        logger.error("Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
                     "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    # 获取 TensorFlow 检查点文件的绝对路径
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # 从 TensorFlow 模型中加载权重
    init_vars = tf.train.list_variables(tf_path)
    names = []
    tf_weights = {}
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        # 加载每个变量的权重数据
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        tf_weights[name] = array
    logger.info(f"Weights not copied to PyTorch model: {', '.join(tf_weights.keys())}.")
    return model

# 多任务 T5 分类头部
class MT5ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config: MT5Config):
        super().__init__()
        # 将输入特征映射到配置的隐藏层大小
        self.dense = nn.Linear(config.d_model, config.d_model)
        # 添加dropout层
        self.dropout = nn.Dropout(p=config.classifier_dropout)
        # 将隐藏特征映射到分类标签数
        self.out_proj = nn.Linear(config.d_model, config.num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 应用dropout
        hidden_states = self.dropout(hidden_states)
        # 映射到隐藏层大小
        hidden_states = self.dense(hidden_states)
        # 应用 tanh 激活函数
        hidden_states = torch.tanh(hidden_states)
        # 再次应用dropout
        hidden_states = self.dropout(hidden_states)
        # 映射到分类标签数
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

# 多任务 T5 预训练模型
class MT5PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MT5Config
    # 加载 TensorFlow 权重的函数
    load_tf_weights = load_tf_weights_in_mt5
    base_model_prefix = "transformer"
    # 支持梯度检查点
    is_parallelizable = True
    supports_gradient_checkpointing = True
    # 不拆分的模块
    _no_split_modules = ["MT5Block"]
    # 保持 float32 的模块
    _keep_in_fp32_modules = ["wo"]

    @property
    def dummy_inputs(self):
        # 定义用于测试的输入数据
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs
    # 定义了一个私有方法 _shift_right，用于将输入向右移动一位
    def _shift_right(self, input_ids):
        # 获取配置项中的decoder_start_token_id和pad_token_id
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id
        
        # 若未定义decoder_start_token_id，则抛出错误
        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In MT5 it is usually set to the pad_token_id. "
                "See MT5 docs for more information."
            )
        
        # 如果输入是torch_fx_proxy类型，则不能原地修改，需要先创建一个新的tensor对其进行赋值操作
        if is_torch_fx_proxy(input_ids):
            # 创建一个形状为(input_ids的shape - 1) + (1,)的全0 tensor，并用decoder_start_token_id填充
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            # 拼接shifted_input_ids和input_ids去掉最后一位得到新的tensor
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            # 创建一个与input_ids相同形状的全0 tensor
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            # 将input_ids的每一维度按索引进行复制，并将复制得到的tensor赋值给shifted_input_ids的对应位置
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            # 将shifted_input_ids的第一个元素设为decoder_start_token_id
            shifted_input_ids[..., 0] = decoder_start_token_id
    
        # 若未定义pad_token_id，则抛出错误
        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        
        # 将shifted_input_ids中可能的-100值替换为pad_token_id
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    
        # 返回移位后的输入tensor
        return shifted_input_ids
# 从transformers.models.t5.modeling_t5.T5Stack中复制代码，将T5->MT5
class MT5Stack(MT5PreTrainedModel):
    # 初始化函数，接受config和embed_tokens参数
    def __init__(self, config, embed_tokens=None):
        # 调用父类的初始化函数
        super().__init__(config)

        # 初始化embed_tokens属性
        self.embed_tokens = embed_tokens
        # 初始化is_decoder属性
        self.is_decoder = config.is_decoder

        # 创建一个包含多个MT5Block的ModuleList，数量为config.num_layers
        self.block = nn.ModuleList(
            [MT5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        # 初始化final_layer_norm属性，使用MT5LayerNorm，设置eps为config.layer_norm_epsilon
        self.final_layer_norm = MT5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        # 初始化dropout属性，使用nn.Dropout，设置dropout率为config.dropout_rate
        self.dropout = nn.Dropout(config.dropout_rate)

        # 初始化权重并应用最终处理
        self.post_init()
        # 模型并行设置为False
        self.model_parallel = False
        # 设备映射设置为None
        self.device_map = None
        # 梯度检查点设置为False
        self.gradient_checkpointing = False

    # 并行化方法，接受device_map参数，默认为None
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        # 发出警告，提示该方法即将被移除
        warnings.warn(
            "`MT5Stack.parallelize` is deprecated and will be removed in v5 of Transformers, you should load your model"
            " with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'block.0': 0,"
            " 'block.1': 1, ...}",
            FutureWarning,
        )
        # 检查device_map的有效性
        self.device_map = (
            get_device_map(len(self.block), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        # 断言设备映射的正确性
        assert_device_map(self.device_map, len(self.block))
        # 将模型并行设置为True
        self.model_parallel = True
        # 设置第一个设备
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        # 设置最后一个设备
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # 加载到设备
        for k, v in self.device_map.items():
            for layer in v:
                cuda_device = "cuda:" + str(k)
                self.block[layer] = self.block[layer].to(cuda_device)

        # 将embed_tokens设置为第一层
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        # 将final_layer_norm设置为最后一个设备
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)

    # 反并行化方法
    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        # 发出警告，提示该方法即将被移除
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        # 将模型并行设置为False
        self.model_parallel = False
        # 设备映射设置为None
        self.device_map = None
        # 第一个设备设置为cpu
        self.first_device = "cpu"
        # 最后一个设备设置为cpu
        self.last_device = "cpu"
        # 将所有block移到cpu
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")
        # 将embed_tokens移到cpu
        self.embed_tokens = self.embed_tokens.to("cpu")
        # 将final_layer_norm移到cpu
        self.final_layer_norm = self.final_layer_norm.to("cpu")
        # 清空cuda缓存
        torch.cuda.empty_cache()

    # 获取输入嵌入层方法
    def get_input_embeddings(self):
        return self.embed_tokens
    # 设置输入的嵌入向量
    def set_input_embeddings(self, new_embeddings):
        # 将输入的新嵌入向量赋值给模型的embed_tokens属性
        self.embed_tokens = new_embeddings
    
    # 前向传播方法
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
        return_dict=None,
# MT5 模型的文档字符串，提供了有关该模型的说明和参考文献链接
MT5_START_DOCSTRING = r"""

    The MT5 model was proposed in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text
    Transformer](https://arxiv.org/abs/1910.10683) by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan
    Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. It's an encoder decoder transformer pre-trained in a
    text-to-text denoising generative setting.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MT5Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# MT5 模型输入的文档字符串，此处留空，可能是未完成的部分
MT5_INPUTS_DOCSTRING = r"""
"""

# MT5 编码器输入的文档字符串，此处留空，可能是未完成的部分
MT5_ENCODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            # 输入序列令牌在词汇表中的索引。MT5 是一个具有相对位置嵌入的模型，因此您应该能够在左侧和右侧都可以填充输入。

            # 可以使用 `AutoTokenizer` 获得索引。参见 `PreTrainedTokenizer.encode` 和 `PreTrainedTokenizer.__call__` 获取详细信息。

            # 要了解有关如何为预训练准备 `input_ids` 的更多信息，请查看 [MT5 Training](./mt5#training)。
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 避免在填充令牌索引上执行注意力的掩码。选择在 `[0, 1]` 中的掩码值：

            # - 1 表示**未被掩码的**令牌，
            # - 0 表示**已被掩码的**令牌。

            # [什么是注意力掩码？](../glossary#attention-mask)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于使自注意力模块的部分头部失效的掩码。选择的掩码值在 `[0, 1]` 中：

            # - 1 表示头部**未被掩码**，
            # - 0 表示头部**已被掩码**。

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            # 可选地，您可以直接传递嵌入表示，而不是传递 `input_ids`。如果您想更精准地控制如何将 `input_ids` 索引转换为相关的向量，那么这很有用。

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多细节，请查看返回的张量下的 `attentions`。
            
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多细节，请查看返回的张量下的 `hidden_states`。

        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通元组。
# 警告消息，警告head_mask已经分为两个参数head_mask和decoder_head_mask
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""

# MT5模型，输出原始隐藏状态而不是任何特定头部信息
@add_start_docstrings(
    "The bare MT5 Model transformer outputting raw hidden-states without any specific head on top.",
    MT5_START_DOCSTRING,
)
class MT5Model(MT5PreTrainedModel):
    r"""
    Examples:

    ```py
    >>> from transformers import MT5Model, AutoTokenizer

    >>> model = MT5Model.from_pretrained("google/mt5-small")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    >>> summary = "Weiter Verhandlung in Syrien."
    >>> inputs = tokenizer(article, return_tensors="pt")
    >>> labels = tokenizer(text_target=summary, return_tensors="pt")

    >>> outputs = model(input_ids=inputs["input_ids"], decoder_input_ids=labels["input_ids"])
    >>> hidden_states = outputs.last_hidden_state
    ```"""

    # 模型类型为"mt5"
    model_type = "mt5"
    # 配置类为MT5Config
    config_class = MT5Config
    _keys_to_ignore_on_load_missing = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]
    _keys_to_ignore_on_load_unexpected = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    # 初始化函数，构建MT5模型
    def __init__(self, config: MT5Config):
        super().__init__(config)
        # 共享embedding层
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # 复制encoder配置并设置为非解码器的配置
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = MT5Stack(encoder_config, self.shared)

        # 复制decoder配置并设置为解码器的配置
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = MT5Stack(decoder_config, self.shared)

        # 初始化权重和应用最终处理
        self.post_init()

        # 模型并行
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    # 从transformers.models.t5.modeling_t5.T5Model.parallelize中复制
    # 这个函数实现了将 T5 模型并行化的功能
    def parallelize(self, device_map=None):
        # 打印一个警告,提示这个方法已经被废弃,建议使用其他方式进行并行化
        warnings.warn(
            "`T5Model.parallelize` is deprecated and will be removed in v5 of Transformers, you should load your model"
            " with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'encoder.block.0':"
            " 0, 'encoder.block.1': 1, ...}",
            FutureWarning,
        )
        # 如果没有传入 device_map,则根据encoder模块的block数量和GPU设备数量自动生成一个 device_map
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        # 检查 device_map 是否有效
        assert_device_map(self.device_map, len(self.encoder.block))
        # 将 encoder 和 decoder 模块并行化
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        # 设置模型为并行模式
        self.model_parallel = True
    
    # 这个函数实现了将并行化的 T5 模型取消并行化的功能
    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        # 打印一个警告,提示这个方法已经被废弃
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        # 将 encoder 和 decoder 模块取消并行化
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        # 将模型移回 CPU 上
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        # 设置模型为非并行模式
        self.model_parallel = False
        self.device_map = None
        # 清空 GPU 缓存
        torch.cuda.empty_cache()
    
    # 这个函数返回模型的输入嵌入层
    def get_input_embeddings(self):
        return self.shared
    
    # 这个函数设置模型的输入嵌入层
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)
    
    # 这个函数返回模型的编码器模块
    def get_encoder(self):
        return self.encoder
    
    # 这个函数返回模型的解码器模块
    def get_decoder(self):
        return self.decoder
    
    # 这个函数根据给定的 heads_to_prune 参数,修剪模型中对应层的注意力头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)
    
    # 这个函数实现了 T5 模型的前向传播
    @add_start_docstrings_to_model_forward(MT5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, *args, **kwargs):
        # 省略代码...
    # 定义模型的前向传播方法，接受多个输入参数，返回模型的输出结果
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入序列的 token IDs，类型为可选的长整型张量，默认为 None
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力遮罩，类型为可选的浮点数张量，默认为 None
        decoder_input_ids: Optional[torch.LongTensor] = None,  # 解码器输入序列的 token IDs，类型为可选的长整型张量，默认为 None
        decoder_attention_mask: Optional[torch.BoolTensor] = None,  # 解码器的注意力遮罩，类型为可选的布尔张量，默认为 None
        head_mask: Optional[torch.FloatTensor] = None,  # 多头注意力机制的头部遮罩，类型为可选的浮点数张量，默认为 None
        decoder_head_mask: Optional[torch.FloatTensor] = None,  # 解码器多头注意力机制的头部遮罩，类型为可选的浮点数张量，默认为 None
        cross_attn_head_mask: Optional[torch.Tensor] = None,  # 交叉注意力机制的头部遮罩，类型为可选的张量，默认为 None
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 编码器输出结果的元组，默认为 None
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 存储过去的键值对的元组，默认为 None
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入嵌入向量，类型为可选的张量，默认为 None
        decoder_inputs_embeds: Optional[torch.Tensor] = None,  # 解码器输入嵌入向量，类型为可选的张量，默认为 None
        use_cache: Optional[bool] = None,  # 是否使用缓存，默认为 None
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，默认为 None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，默认为 None
        return_dict: Optional[bool] = None,  # 是否返回字典格式的输出结果，默认为 None
# 定义了一个基于 MT5 的条件生成模型，用于语言建模任务
@add_start_docstrings("""MT5 Model with a `language modeling` head on top.""", MT5_START_DOCSTRING)
class MT5ForConditionalGeneration(MT5PreTrainedModel):
    r"""
    Examples:

    ```py
    >>> from transformers import MT5ForConditionalGeneration, AutoTokenizer

    >>> model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
    >>> summary = "Weiter Verhandlung in Syrien."
    >>> inputs = tokenizer(article, text_target=summary, return_tensors="pt")

    >>> outputs = model(**inputs)
    >>> loss = outputs.loss
    ```"""

    # 模型类型
    model_type = "mt5"
    # 配置类
    config_class = MT5Config
    # 加载时要忽略的键
    _keys_to_ignore_on_load_unexpected = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]
    # 要绑定权重的键
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    # 从 MT5 配置初始化模型
    # 从 transformers.models.t5.modeling_t5.T5ForConditionalGeneration.__init__ 复制而来，只是将 T5 替换为 MT5
    def __init__(self, config: MT5Config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 模型维度
        self.model_dim = config.d_model

        # 共享层，用于嵌入
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        # 编码器配置
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # 编码器
        self.encoder = MT5Stack(encoder_config, self.shared)

        # 解码器配置
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        # 解码器
        self.decoder = MT5Stack(decoder_config, self.shared)

        # 语言建模头部
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 初始化权重并进行最终处理
        self.post_init()

        # 模型并行
        self.model_parallel = False
        self.device_map = None

    # 并行化方法的文档注释
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    # 从 transformers.models.t5.modeling_t5.T5ForConditionalGeneration.parallelize 复制而来
```py  
    # 并行化操作，根据给定的device_map在不同设备上并行化模型的encoder和decoder
    def parallelize(self, device_map=None):
        # 发出警告，提示方法将在 Transformers 的 v5 版本中移除
        warnings.warn(
            "`T5ForConditionalGeneration.parallelize` is deprecated and will be removed in v5 of Transformers, you"
            " should load your model with `device_map='balanced'` in the call to `from_pretrained`. You can also"
            " provide your own `device_map` but it needs to be a dictionary module_name to device, so for instance"
            " {'encoder.block.0': 0, 'encoder.block.1': 1, ...}",
            FutureWarning,
        )
        # 获取设备映射或者创建平衡设备映射
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        # 检查设备映射的有效性
        assert_device_map(self.device_map, len(self.encoder.block))
        # 在设备上并行化encoder
        self.encoder.parallelize(self.device_map)
        # 在设备上并行化decoder
        self.decoder.parallelize(self.device_map)
        # 将lm_head移动到第一个设备上
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        # 设置为模型并行化
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    # 反并行化操作，将模型还原为单设备运行
    # 从 transformers.models.t5.modeling_t5.T5ForConditionalGeneration.deparallelize 复制而来
    def deparallelize(self):
        # 发出警告，提示方法将在 Transformers 的 v5 版本中移除
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        # 反并行化encoder
        self.encoder.deparallelize()
        # 反并行化decoder
        self.decoder.deparallelize()
        # 将encoder移回CPU
        self.encoder = self.encoder.to("cpu")
        # 将decoder移回CPU
        self.decoder = self.decoder.to("cpu")
        # 将lm_head移回CPU
        self.lm_head = self.lm_head.to("cpu")
        # 设置为非模型并行化
        self.model_parallel = False
        # 清空CUDA缓存
        torch.cuda.empty_cache()

    # 获取输入的词嵌入层
    # 从 transformers.models.t5.modeling_t5.T5ForConditionalGeneration.get_input_embeddings 复制而来
    def get_input_embeddings(self):
        return self.shared

    # 设置输入的词嵌入层
    # 从 transformers.models.t5.modeling_t5.T5ForConditionalGeneration.set_input_embeddings 复制而来
    def set_input_embeddings(self, new_embeddings):
        # 设置���享的词嵌入层
        self.shared = new_embeddings
        # 为encoder设置新的输入词嵌入层
        self.encoder.set_input_embeddings(new_embeddings)
        # 为decoder设置新的输入词嵌入层
        self.decoder.set_input_embeddings(new_embeddings)

    # 设置输出的词嵌入层
    # 从 transformers.models.t5.modeling_t5.T5ForConditionalGeneration.set_output_embeddings 复制而来
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 获取输出的词嵌入层
    # 从 transformers.models.t5.modeling_t5.T5ForConditionalGeneration.get_output_embeddings 复制而来
    def get_output_embeddings(self):
        return self.lm_head

    # 获取encoder
    # 从 transformers.models.t5.modeling_t5.T5ForConditionalGeneration.get_encoder 复制而来
    def get_encoder(self):
        return self.encoder

    # 获取decoder
    # 从 transformers.models.t5.modeling_t5.T5ForConditionalGeneration.get_decoder 复制而来
    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(MT5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # 从transformers.models.t5.modeling_t5.T5ForConditionalGeneration.forward中复制的代码，并将T5->MT5，t5->mt5改为新的Transformer模型和缩写
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的token ID序列，默认为空
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力掩码，默认为空
        decoder_input_ids: Optional[torch.LongTensor] = None,  # 解码器的token ID序列，默认为空
        decoder_attention_mask: Optional[torch.BoolTensor] = None,  # 解码器的注意力掩码，默认为空
        head_mask: Optional[torch.FloatTensor] = None,  # 头部掩码，默认为空
        decoder_head_mask: Optional[torch.FloatTensor] = None,  # 解码器的头部掩码，默认为空
        cross_attn_head_mask: Optional[torch.Tensor] = None,  # 交叉注意力头部掩码，默认为空
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,  # 编码器输出，默认为空
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,  # 过去的键值对，默认为空
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入嵌入，默认为空
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,  # 解码器的输入嵌入，默认为空
        labels: Optional[torch.LongTensor] = None,  # 标签，默认为空
        use_cache: Optional[bool] = None,  # 是否使用缓存，默认为空
        output_attentions: Optional[bool] = None,  # 是否输出注意力，默认为空
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，默认为空
        return_dict: Optional[bool] = None,  # 是否返回字典，默认为空
    # 从transformers.models.t5.modeling_t5.T5ForConditionalGeneration.prepare_inputs_for_generation中复制的代码
    def prepare_inputs_for_generation(
        self,
        input_ids,  # 输入的token ID序列
        past_key_values=None,  # 过去的键值对，默认为空
        attention_mask=None,  # 注意力掩码，默认为空
        head_mask=None,  # 头部掩码，默认为空
        decoder_head_mask=None,  # 解码器的头部掩码，默认为空
        decoder_attention_mask=None,  # 解码器的注意力掩码，默认为空
        cross_attn_head_mask=None,  # 交叉注意力头部掩码，默认为空
        use_cache=None,  # 是否使用缓存，默认为空
        encoder_outputs=None,  # 编码器输出，默认为空
        **kwargs,  # 其他关键字参数，存储在kwargs字典中
    ):
        # 如果使用了过去的键值对，截取解码器输入序列
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法已经只传递了最后一个输入ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认旧行为：仅保留最终ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        return {
            "decoder_input_ids": input_ids,  # 返回解码器的输入ID
            "past_key_values": past_key_values,  # 返回过去的键值对
            "encoder_outputs": encoder_outputs,  # 返回编码器输出
            "attention_mask": attention_mask,  # 返回注意力掩码
            "head_mask": head_mask,  # 返回头部掩码
            "decoder_head_mask": decoder_head_mask,  # 返回解码器的头部掩码
            "decoder_attention_mask": decoder_attention_mask,  # 返回解码器的注意力掩码
            "cross_attn_head_mask": cross_attn_head_mask,  # 返回交叉注意力头部掩码
            "use_cache": use_cache,  # 返回是否使用缓存
        }

    # 从transformers.models.t5.modeling_t5.T5ForConditionalGeneration.prepare_decoder_input_ids_from_labels中复制的代码
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)  # 返回右移标签后的结果

    # 从transformers.models.t5.modeling_t5.T5ForConditionalGeneration._reorder_cache中
    def _reorder_cache(self):  # 重新排序缓存的函数
    # 重新排列缓存信息，根据给定的 beam 索引（用于 beam search）
    def _reorder_cache(self, past_key_values, beam_idx):
        # 如果过去的解码器状态不包含在输出中，则不启用速度解码，无需重新排列
        if past_key_values is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values

        # 存储重新排列后的解码器过去状态
        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # 根据层过去状态的 batch 维度获取正确的 batch 索引
            # `past` 的 batch 维度是在第二个位置
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # 需要为四个键/值状态中的每一个设置正确的 `past`
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            # 检查重新排列后的第一个层状态形状与原始状态形状是否匹配
            if reordered_layer_past_states[0].shape != layer_past_states[0].shape:
                raise ValueError(
                    f"reordered_layer_past_states[0] shape {reordered_layer_past_states[0].shape} and layer_past_states[0] shape {layer_past_states[0].shape} mismatched"
                )
            
            # 检查重新排列后的状态数量是否与原始状态数量匹配
            if len(reordered_layer_past_states) != len(layer_past_states):
                raise ValueError(
                    f"length of reordered_layer_past_states {len(reordered_layer_past_states)} and length of layer_past_states {len(layer_past_states)} mismatched"
                )

            # 将重新排列后的层状态添加到重新排列后的解码器过去状态中
            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        
        return reordered_decoder_past
# 导入函数装饰器，添加文档字符串，指定模型类型和配置类
@add_start_docstrings(
    "The bare MT5 Model transformer outputting encoder's raw hidden-states without any specific head on top.",
    MT5_START_DOCSTRING,
)
# 定义 MT5EncoderModel 类，继承自 MT5PreTrainedModel
class MT5EncoderModel(MT5PreTrainedModel):
    r"""
    Examples:

    ```python
    >>> from transformers import MT5EncoderModel, AutoTokenizer

    >>> model = MT5EncoderModel.from_pretrained("google/mt5-small")  # 从预训练模型创建实例
    >>> tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")  # 从预训练模型创建分词器
    >>> article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."  # 输入文章
    >>> input_ids = tokenizer(article, return_tensors="pt").input_ids  # 获取输入文章的编码
    >>> outputs = model(input_ids)  # 将输入文章编码输入模型
    >>> hidden_state = outputs.last_hidden_state  # 获取模型输出的最后一个隐藏状态
    ```py"""

    model_type = "mt5"  # 模型类型
    config_class = MT5Config  # 配置类
    _tied_weights_keys = ["encoder.embed_tokens.weight"]  # 将权重绑定的键列表

    # 从 transformers.models.t5.modeling_t5.T5EncoderModel.__init__ 复制过来，将 T5->MT5
    def __init__(self, config: MT5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)  # 使用配置类中的词汇量大小和模型尺寸创建共享嵌入层

        encoder_config = copy.deepcopy(config)  # 复制配置
        encoder_config.use_cache = False  # 设置缓存标志为 False
        encoder_config.is_encoder_decoder = False  # 设置是否是编码器解码器标志为 False
        self.encoder = MT5Stack(encoder_config, self.shared)  # 创建 MT5 堆栈对象，传入配置和共享嵌入层

        # 初始化权重并应用最终处理
        self.post_init()

        # 模型并行化
        self.model_parallel = False
        self.device_map = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)  # 添加并行化文档字符串
    # 从 transformers.models.t5.modeling_t5.T5EncoderModel.parallelize 复制过来
    def parallelize(self, device_map=None):
        warnings.warn(
            "`T5EncoderModel.parallelize` is deprecated and will be removed in v5 of Transformers, you should load"
            " your model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'block.0': 0,"
            " 'block.1': 1, ...}",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))  # 获取设备映射
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))  # 确保设备映射正确
        self.encoder.parallelize(self.device_map)  # 将编码器对象并行化
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)  # 添加去并行化文档字符串
    # 从 transformers.models.t5.modeling_t5.T5EncoderModel.deparallelize 复制过来
    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.encoder.deparallelize()  # 将编码器对象去并行化
        self.encoder = self.encoder.to("cpu")  # 将编码器对象移动到 CPU
        self.model_parallel = False  # 模型并行标志设为 False
        self.device_map = None  # 设备映射置为 None
        torch.cuda.empty_cache()  # 清空 CUDA 缓存

    # 从 transformers.models.t5.modeling_t5.T5EncoderModel.get_input_embeddings 复制过来
    # 获取输入嵌入层的引用
    def get_input_embeddings(self):
        return self.shared

    # 从transformers.models.t5.modeling_t5.T5EncoderModel.set_input_embeddings中复制而来，用于设置输入嵌入层
    def set_input_embeddings(self, new_embeddings):
        # 更新共享的嵌入层
        self.shared = new_embeddings
        # 更新编码器的输入嵌入层
        self.encoder.set_input_embeddings(new_embeddings)

    # 从transformers.models.t5.modeling_t5.T5EncoderModel.get_encoder中复制而来，用于获取编码器
    def get_encoder(self):
        return self.encoder

    # 从transformers.models.t5.modeling_t5.T5EncoderModel._prune_heads中复制而来，用于剪枝模型的注意力头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            # 在指定的层中剪枝注意力头
            self.encoder.block[layer].layer[0].SelfAttention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(MT5_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    # 从transformers.models.t5.modeling_t5.T5EncoderModel.forward复制而来，将T5替换为MT5，t5替换为mt5
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        r"""
        返回：

        示例：

        ```python
        >>> from transformers import AutoTokenizer, MT5EncoderModel

        >>> tokenizer = AutoTokenizer.from_pretrained("mt5-small")
        >>> model = MT5EncoderModel.from_pretrained("mt5-small")
        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model(input_ids=input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```py"""
        # 如果未指定return_dict，则使用配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用编码器进行前向传播
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs
# 使用 add_start_docstrings 装饰器为类添加文档字符串，描述 MT5 模型及其顶部的序列分类/头部的结构和用途，例如 GLUE 任务
@add_start_docstrings(
    """
    MT5 model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE
    tasks.
    """,
    MT5_START_DOCSTRING,
)
# 定义 MT5 序列分类模型类，继承自 MT5PreTrainedModel 类
class MT5ForSequenceClassification(MT5PreTrainedModel):
    # 在加载时忽略的键列表
    _keys_to_ignore_on_load_unexpected = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]
    # 被绑定权重的键列表
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    # 初始化方法，接受一个 MT5Config 对象作为参数
    # Copied from transformers.models.t5.modeling_t5.T5ForSequenceClassification.__init__ with T5->MT5
    def __init__(self, config: MT5Config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 初始化 transformer 层，使用给定的配置
        self.transformer = MT5Model(config)
        # 初始化分类头部，使用给定的配置
        self.classification_head = MT5ClassificationHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

        # 设置模型并行性为 False
        self.model_parallel = False

    # 使用 add_start_docstrings_to_model_forward 装饰器添加模型前向传播方法的文档字符串
    # 并替换返回值的文档字符串类型为 Seq2SeqSequenceClassifierOutput，配置类为 _CONFIG_FOR_DOC
    # Copied from transformers.models.t5.modeling_t5.T5ForSequenceClassification.forward
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 初始化模型
    def __init__(self, config: MT5Config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 保存模型维度的大小
        self.model_dim = config.d_model
    
        # 创建共享的输入嵌入层
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
    
        # 创建编码器配置
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        # 创建编码器
        self.encoder = MT5Stack(encoder_config, self.shared)
    
        # 创建解码器配置
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        # 创建解码器
        self.decoder = MT5Stack(decoder_config, self.shared)
    
        # 设置标签数量
        self.num_labels = config.num_labels
        # 创建输出层
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
    
        # 初始化权重并应用最后的处理
        self.post_init()
    
        # 是否设置模型并行
        self.model_parallel = False
    
    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.shared
    
    # 设置输入嵌入层
    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)
    
    # 获取编码器
    def get_encoder(self):
        return self.encoder
    
    # 获取解码器
    def get_decoder(self):
        return self.decoder
    
    # 模型前向传播
    @add_start_docstrings_to_model_forward(MT5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        pass
```