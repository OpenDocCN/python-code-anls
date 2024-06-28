# `.\models\deprecated\trajectory_transformer\modeling_trajectory_transformer.py`

```
# coding=utf-8
# Copyright 2022 The Trajectory Transformers paper authors and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch TrajectoryTransformer model."""

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F

from ....modeling_utils import PreTrainedModel
from ....utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_trajectory_transformer import TrajectoryTransformerConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置
_CHECKPOINT_FOR_DOC = "CarlCochet/trajectory-transformer-halfcheetah-medium-v2"
_CONFIG_FOR_DOC = "TrajectoryTransformerConfig"

# 预训练模型存档列表
TRAJECTORY_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "CarlCochet/trajectory-transformer-halfcheetah-medium-v2",
    # See all TrajectoryTransformer models at https://huggingface.co/models?filter=trajectory_transformer
]

# 从 TensorFlow 模型加载权重到 PyTorch 模型
def load_tf_weights_in_trajectory_transformer(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    # 获取 TensorFlow 检查点文件的绝对路径
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # 从 TF 模型加载权重
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)
    # 遍历 names 和 arrays 中的每个元素，其中 names 是名称列表，arrays 是数组列表
    for name, array in zip(names, arrays):
        # 将 name 按 "/" 分割成列表
        name = name.split("/")
        
        # 检查是否包含特定的名称，如果是，则跳过当前循环
        # 这些名称通常是不需要用于预训练模型的变量
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        
        # 初始化指针为模型本身
        pointer = model
        
        # 遍历 name 中的每个名称部分
        for m_name in name:
            # 如果名称符合形如 "xxx_0" 的模式，则按下划线拆分
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            
            # 根据第一个名称部分确定要设置的指针位置
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    # 如果找不到相应的属性，记录日志并跳过当前循环
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            
            # 如果名称部分超过一个，则按索引号更新指针位置
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        
        # 如果最后一个名称部分是 "_embeddings"，则将指针定位到权重属性
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            # 如果名称是 "kernel"，则对数组进行转置操作
            array = np.transpose(array)
        
        # 检查指针和数组的形状是否匹配，如果不匹配，则引发 ValueError 异常
        try:
            if pointer.shape != array.shape:
                raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
        except AssertionError as e:
            # 如果发生断言错误，则将错误信息更新后重新引发
            e.args += (pointer.shape, array.shape)
            raise
        
        # 记录日志，指示正在初始化 PyTorch 权重
        logger.info(f"Initialize PyTorch weight {name}")
        
        # 将 numpy 数组转换为 PyTorch 张量，并赋值给指针的数据属性
        pointer.data = torch.from_numpy(array)
    
    # 返回更新后的模型
    return model
# 使用 @dataclass 装饰器定义一个数据类，表示轨迹转换器的模型输出。
@dataclass
class TrajectoryTransformerOutput(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss. 语言建模损失（可选），在提供标签时返回。
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax). 语言建模头的预测分数（SoftMax 之前每个词汇标记的分数）。
        past_key_values (`Tuple[Tuple[torch.Tensor]]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of length `config.n_layers`, containing tuples of tensors of shape `(batch_size, num_heads,
            sequence_length, embed_size_per_head)`). Contains pre-computed hidden-states (key and values in the
            attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
            长度为 `config.n_layers` 的元组，包含形状为 `(batch_size, num_heads, sequence_length, embed_size_per_head)` 的张量元组。
            包含预先计算的隐藏状态（注意力块中的键和值），可用于加速顺序解码。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
            包含模型每一层输出的隐藏状态的元组（每层一个），以及初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. GPT2Attentions weights after the attention softmax, used to compute the weighted average
            in the self-attention heads.
            包含注意力权重的元组（每层一个），用于计算自注意力头中的加权平均值。
    """
    
    # 定义可选的损失值，类型为 torch.FloatTensor，形状为 `(1,)`
    loss: Optional[torch.FloatTensor] = None
    # 定义预测的 logits，类型为 torch.FloatTensor，形状为 `(batch_size, sequence_length, config.vocab_size)`
    logits: torch.FloatTensor = None
    # 定义过去键值对的元组，类型为 `Tuple[Tuple[torch.FloatTensor]]`
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    # 定义隐藏状态的元组，类型为 `tuple(torch.FloatTensor)`
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义注意力权重的元组，类型为 `tuple(torch.FloatTensor)`
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# 定义 TrajectoryTransformerPreTrainedModel 类，继承自 PreTrainedModel，用于处理权重初始化、预训练模型的下载和加载的抽象类。
class TrajectoryTransformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置配置类为 TrajectoryTransformerConfig
    config_class = TrajectoryTransformerConfig
    # 设置加载 TensorFlow 权重的函数为 load_tf_weights_in_trajectory_transformer
    load_tf_weights = load_tf_weights_in_trajectory_transformer
    # 设置基础模型前缀为 "trajectory_transformer"
    base_model_prefix = "trajectory_transformer"
    # 设置主输入名称为 "trajectories"
    main_input_name = "trajectories"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 定义一个初始化权重的方法，用于初始化模块的参数
    def _init_weights(self, module):
        # 如果模块是线性层或嵌入层
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # 使用正态分布初始化权重，均值为0，标准差为配置文件中指定的范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果是线性层，并且存在偏置，则将偏置初始化为零
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是 LayerNorm 层
        elif isinstance(module, nn.LayerNorm):
            # 将偏置初始化为零
            module.bias.data.zero_()
            # 将权重初始化为全1
            module.weight.data.fill_(1.0)
        # 如果模块是 EinLinear 类型
        elif isinstance(module, EinLinear):
            # 针对每个模型，使用 Kaiming 均匀初始化权重
            for i in range(module.n_models):
                nn.init.kaiming_uniform_(module.weight[i], a=math.sqrt(5) / self.config.kaiming_initializer_range)
                # 如果存在偏置，则根据 fan-in 计算初始化范围，并使用均匀分布初始化偏置
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight[i])
                    bound = (1 / math.sqrt(fan_in)) * self.config.initializer_range
                    nn.init.uniform_(module.bias[i], -bound, bound)
# 定义模型文档字符串，描述了这个 PyTorch 模型类的基本信息和参数使用说明
TRAJECTORY_TRANSFORMER_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`TrajectoryTransformerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义输入文档字符串，详细描述了模型类的输入参数及其含义
TRAJECTORY_TRANSFORMER_INPUTS_DOCSTRING = r"""
    Args:
        trajectories (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Batch of trajectories, where a trajectory is a sequence of states, actions and rewards.
        past_key_values (`Tuple[Tuple[torch.Tensor]]` of length `config.n_layers`, *optional*):
            Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see
            `past_key_values` output below). Can be used to speed up sequential decoding. The `input_ids` which have
            their past given to this model should not be passed as `input_ids` as they have already been computed.
        targets (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Desired targets used to compute the loss.
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""
    def __init__(self, n_models, in_features, out_features, bias):
        super().__init__()
        self.n_models = n_models
        self.out_features = out_features
        self.in_features = in_features
        self.weight = nn.Parameter(torch.Tensor(n_models, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_models, out_features))
        else:
            self.register_parameter("bias", None)


        # 初始化函数，设置模型的参数和权重
        super().__init__()  # 调用父类的初始化函数
        self.n_models = n_models  # 设置模型中的子模型数量
        self.out_features = out_features  # 设置输出特征的数量
        self.in_features = in_features  # 设置输入特征的数量
        self.weight = nn.Parameter(torch.Tensor(n_models, out_features, in_features))  # 初始化权重参数
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_models, out_features))  # 如果有偏置，则初始化偏置参数
        else:
            self.register_parameter("bias", None)  # 如果没有偏置，则注册一个空的偏置参数



    def reset_parameters(self):
        for i in range(self.n_models):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[i])
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias[i], -bound, bound)


        # 重置模型参数的函数
        for i in range(self.n_models):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))  # 使用 Kaiming 均匀初始化权重
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[i])
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias[i], -bound, bound)  # 使用均匀分布初始化偏置



    def forward(self, input):
        """
        Args:
            input (`torch.FloatTensor` of shape `(B, n_models, input_dim)`):
                The input to the layer.
        """
        # [ batch_size x n_models x output_dim ]
        output = torch.einsum("eoi,bei->beo", self.weight, input)  # 执行张量乘法运算
        if self.bias is not None:
            raise RuntimeError()  # 如果偏置不为空，则引发运行时错误
        return output


        # 前向传播函数，计算输入的张量乘法结果
        output = torch.einsum("eoi,bei->beo", self.weight, input)  # 使用 Einstein Summation 进行张量乘法运算
        if self.bias is not None:
            raise RuntimeError()  # 如果偏置不为空，则引发运行时错误
        return output  # 返回计算结果的输出张量
        class CausalSelfAttention(nn.Module):
            def __init__(self, config):
                super().__init__()

                if config.n_embd % config.n_head != 0:
                    raise ValueError(f"n_head ({config.n_head}) should be a divisor of n_embd ({config.n_embd})")

                # key, query, value projections for all heads
                # 为所有注意力头创建 key、query、value 的投影层
                self.key = nn.Linear(config.n_embd, config.n_embd)
                self.query = nn.Linear(config.n_embd, config.n_embd)
                self.value = nn.Linear(config.n_embd, config.n_embd)

                # regularization
                # 注意力机制的正则化
                self.attn_drop = nn.Dropout(config.attn_pdrop)
                self.resid_drop = nn.Dropout(config.resid_pdrop)

                # output projection
                # 输出层的投影
                self.proj = nn.Linear(config.n_embd, config.n_embd)

                # causal mask to ensure that attention is only applied to the left in the input sequence
                # 因果掩码确保注意力仅应用于输入序列的左侧
                self.register_buffer(
                    "mask",
                    torch.tril(torch.ones(config.block_size, config.block_size)).view(
                        1, 1, config.block_size, config.block_size
                    ),
                    persistent=False,
                )

                # mask previous value estimates
                # 屏蔽先前的值估计
                joined_dim = config.observation_dim + config.action_dim + 2
                self.mask.squeeze()[:, joined_dim - 1 :: joined_dim] = 0

                self.n_head = config.n_head

            def forward(
                self,
                hidden_states: Optional[Tuple[torch.FloatTensor]],
                layer_past: Optional[Tuple[torch.Tensor]] = None,
                use_cache: Optional[bool] = False,
                output_attentions: Optional[bool] = False,
        ):
            # 获取隐藏状态张量的尺寸信息：批量大小、序列长度、嵌入维度
            batch_size, sequence_length, embedding_dim = hidden_states.size()

            # 计算每个头部的查询、键、值，并将头部维度移动到批量维度之前
            # [ batch_size x n_heads x sequence_length x head_dim ]
            key = (
                self.key(hidden_states)
                .view(batch_size, sequence_length, self.n_head, embedding_dim // self.n_head)
                .transpose(1, 2)
            )
            query = (
                self.query(hidden_states)
                .view(batch_size, sequence_length, self.n_head, embedding_dim // self.n_head)
                .transpose(1, 2)
            )
            value = (
                self.value(hidden_states)
                .view(batch_size, sequence_length, self.n_head, embedding_dim // self.n_head)
                .transpose(1, 2)
            )

            if layer_past is not None:
                past_key, past_value = layer_past
                # 将过去的键和值与当前计算得到的键和值拼接起来
                key = torch.cat((past_key, key), dim=-2)
                value = torch.cat((past_value, value), dim=-2)

            if use_cache is True:
                # 如果需要使用缓存，将当前的键和值存储为当前状态
                present = (key, value)
            else:
                present = None

            # 自回归自注意力机制
            # [ batch_size x n_heads x sequence_length x sequence_length ]
            attn_weights = (torch.matmul(query, key.transpose(-2, -1))) * (1.0 / math.sqrt(key.size(-1)))
            # 掩盖填充部分，防止注意力机制关注填充位置
            attn_weights = attn_weights.masked_fill(
                self.mask[:, :, :sequence_length, :sequence_length] == 0, torch.finfo(attn_weights.dtype).min
            )
            # 对注意力权重进行归一化
            attn_weights = F.softmax(attn_weights, dim=-1)
            # 保存注意力权重映射，便于后续分析或可视化
            self._attn_map = attn_weights.clone()
            # 对注意力权重应用 dropout
            attn_weights = self.attn_drop(attn_weights)

            output = torch.matmul(attn_weights, value)
            # [ batch_size x sequence_length x embedding_dim ]
            # 将所有头部的输出重新组合在一起
            output = output.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_dim)

            # 输出投影层处理
            output = self.resid_drop(self.proj(output))

            outputs = (output, present)
            if output_attentions:
                outputs += (attn_weights,)

            return outputs
# 定义一个名为 Block 的神经网络模块，继承自 nn.Module 类
class Block(nn.Module):
    # 初始化函数，接受一个 config 参数
    def __init__(self, config):
        super().__init__()
        # 第一个 Layer Normalization 层，标准化输入的 embedding 维度
        self.ln1 = nn.LayerNorm(config.n_embd)
        # 第二个 Layer Normalization 层，标准化注意力输出的 embedding 维度
        self.ln2 = nn.LayerNorm(config.n_embd)
        # 自注意力机制，使用 CausalSelfAttention 类进行定义
        self.attn = CausalSelfAttention(config)

        # MLP 部分
        # 第一个线性层，将 embedding 维度转换为 4 倍的 embedding 维度
        self.l1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        # GELU 激活函数
        self.act = nn.GELU()
        # 第二个线性层，将 4 倍的 embedding 维度转换回原始 embedding 维度
        self.l2 = nn.Linear(4 * config.n_embd, config.n_embd)
        # Dropout 层，以指定的概率进行神经元丢弃，防止过拟合
        self.drop = nn.Dropout(config.resid_pdrop)

    # 前向传播函数
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        # 保存输入的 residual 连接
        residual = hidden_states
        # 第一个 Layer Normalization 层
        hidden_states = self.ln1(hidden_states)

        # 自注意力机制的前向传播
        attn_outputs = self.attn(
            hidden_states, layer_past=layer_past, use_cache=use_cache, output_attentions=output_attentions
        )
        attn_output = attn_outputs[0]  # 注意力输出
        outputs = attn_outputs[1:]  # 其他输出（如果有）

        # 残差连接
        hidden_states = attn_output + residual

        # 保存当前的 residual 连接
        residual = hidden_states
        # 第二个 Layer Normalization 层
        hidden_states = self.ln2(hidden_states)
        # MLP 的线性层1
        hidden_states = self.l1(hidden_states)
        # 使用 GELU 激活函数
        hidden_states = self.act(hidden_states)
        # MLP 的线性层2
        hidden_states = self.l2(hidden_states)
        # 残差连接和 Dropout
        hidden_states = residual + self.drop(hidden_states)

        # 如果使用缓存，将当前隐藏状态输出保存到 outputs 中
        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        # 返回最终的输出结果
        return outputs


# 使用装饰器添加文档字符串，描述了 TrajectoryTransformerModel 类的作用
@add_start_docstrings(
    "The bare TrajectoryTransformer Model transformer outputting raw hidden-states without any specific head on top.",
    TRAJECTORY_TRANSFORMER_START_DOCSTRING,
)
# 定义 TrajectoryTransformerModel 类，继承自 TrajectoryTransformerPreTrainedModel 类
class TrajectoryTransformerModel(TrajectoryTransformerPreTrainedModel):
    """the full GPT language model, with a context size of block_size"""
    def __init__(self, config):
        # 调用父类构造函数，初始化模型配置
        super().__init__(config)

        # 输入嵌入层，将输入映射到指定维度的向量空间，考虑停止标记
        self.tok_emb = nn.Embedding(config.vocab_size * config.transition_dim + 1, config.n_embd)

        # 位置嵌入层，用于表示序列中每个位置的信息
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))

        # Dropout层，用于随机失活以防止过拟合
        self.drop = nn.Dropout(config.embd_pdrop)

        # Transformer块列表，用于处理序列信息
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # Decoder头部的LayerNorm，用于标准化解码器输出
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Decoder头部的线性层，用于生成最终的预测结果
        self.head = EinLinear(config.transition_dim, config.n_embd, config.vocab_size + 1, bias=False)

        # 词汇表大小
        self.vocab_size = config.vocab_size

        # 停止标记的值
        self.stop_token = config.vocab_size * config.transition_dim

        # 块大小
        self.block_size = config.block_size

        # 观测维度
        self.observation_dim = config.observation_dim

        # 动作维度
        self.action_dim = config.action_dim

        # 转移维度
        self.transition_dim = config.transition_dim

        # 嵌入维度
        self.embedding_dim = config.n_embd

        # 动作权重
        self.action_weight = config.action_weight

        # 奖励权重
        self.reward_weight = config.reward_weight

        # 值权重
        self.value_weight = config.value_weight

        # 是否使用梯度检查点
        self.gradient_checkpointing = False

        # 执行初始化后处理
        self.post_init()

    def get_block_size(self):
        # 返回块大小
        return self.block_size

    def offset_tokens(self, trajectories):
        # 计算序列长度
        _, sequence_length = trajectories.shape

        # 计算状态数量
        n_states = int(np.ceil(sequence_length / self.transition_dim))

        # 计算偏移量，使每个状态的起始标记不同
        offsets = torch.arange(self.transition_dim) * self.vocab_size
        offsets = offsets.repeat(n_states).to(trajectories.device)

        # 应用偏移量到轨迹数据，将停止标记替换为预定义的停止标记值
        offset_trajectories = trajectories + offsets[:sequence_length]
        offset_trajectories[trajectories == self.vocab_size] = self.stop_token

        return offset_trajectories

    def pad_to_full_observation(self, hidden_states):
        # 获取批处理大小、序列长度、嵌入维度
        batch_size, sequence_length, _ = hidden_states.shape

        # 计算需要填充的数量，使序列长度能够被转移维度整除
        n_pad = (self.transition_dim - sequence_length % self.transition_dim) % self.transition_dim

        # 创建填充的张量，维度为 [batch_size x n_pad x embedding_dim]
        padding = torch.zeros(batch_size, n_pad, self.embedding_dim, device=hidden_states.device)

        # 将填充后的序列连接到隐藏状态中，形成 [batch_size x padded_sequence_length' x embedding_dim]
        hidden_states_pad = torch.cat([hidden_states, padding], dim=1)

        # 将填充后的序列重新组织为 [batch_size*sequence_length'/transition_dim x transition_dim x embedding_dim]
        hidden_states_pad = hidden_states_pad.view(-1, self.transition_dim, self.embedding_dim)

        return hidden_states_pad, n_pad
    # 定义一个方法 `forward`，用于模型的前向传播
    def forward(
        # 输入参数 trajectories：轨迹数据，可选的长整型张量，默认为 None
        trajectories: Optional[torch.LongTensor] = None,
        # 输入参数 past_key_values：过去的键-值对元组，可选的张量元组，默认为 None
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        # 输入参数 targets：目标数据，可选的浮点张量，默认为 None
        targets: Optional[torch.FloatTensor] = None,
        # 输入参数 attention_mask：注意力掩码，可选的浮点张量，默认为 None
        attention_mask: Optional[torch.FloatTensor] = None,
        # 输入参数 use_cache：是否使用缓存，可选的布尔值，默认为 None
        use_cache: Optional[bool] = None,
        # 输入参数 output_attentions：是否输出注意力权重，可选的布尔值，默认为 None
        output_attentions: Optional[bool] = None,
        # 输入参数 output_hidden_states：是否输出隐藏状态，可选的布尔值，默认为 None
        output_hidden_states: Optional[bool] = None,
        # 输入参数 return_dict：是否返回字典格式的输出，可选的布尔值，默认为 None
        return_dict: Optional[bool] = None,
```