# `.\models\deprecated\trajectory_transformer\modeling_trajectory_transformer.py`

```
# 设定编码格式为UTF-8

# 版权声明及许可证信息，该代码版权归 Trajectory Transformers 论文作者和 HuggingFace Inc. 团队所有，
# 使用 Apache License, Version 2.0 进行许可
# 可以在符合许可证的情况下使用此文件
# 获取许可证的副本，请访问 http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则本软件按“原样”提供，不提供任何担保或条件，
# 无论是明示的还是默示的，包括但不限于对适销性或特定用途适用性的默示担保或条件。
# 有关详细信息，请参阅许可证

# 导入必要的库
import math  # 数学运算库
import os  # 操作系统相关功能的库
from dataclasses import dataclass  # 用于创建数据类的装饰器
from typing import Optional, Tuple, Union  # 类型提示相关库

import numpy as np  # 用于处理数组和矩阵的库
import torch  # PyTorch 深度学习库
import torch.utils.checkpoint  # PyTorch 提供的用于内存优化的检查点模块
from torch import nn  # 神经网络模块
from torch.nn import functional as F  # 神经网络函数模块

# 导入模型相关的工具函数和类
from ....modeling_utils import PreTrainedModel  # 预训练模型基类
from ....utils import (  # 其他工具函数
    ModelOutput,  # 模型输出
    add_start_docstrings,  # 添加文档字符串的装饰器
    add_start_docstrings_to_model_forward,  # 添加文档字符串到模型前向传播的装饰器
    logging,  # 日志记录模块
    replace_return_docstrings,  # 替换返回文档字符串的装饰器
)
from .configuration_trajectory_transformer import TrajectoryTransformerConfig  # 导入轨迹变换器的配置类

logger = logging.get_logger(__name__)  # 获取日志记录器

_CHECKPOINT_FOR_DOC = "CarlCochet/trajectory-transformer-halfcheetah-medium-v2"  # 文档用的检查点
_CONFIG_FOR_DOC = "TrajectoryTransformerConfig"  # 文档用的配置类

# 轨迹变换器预训练模型的存档列表
TRAJECTORY_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "CarlCochet/trajectory-transformer-halfcheetah-medium-v2",
    # 可以在 https://huggingface.co/models?filter=trajectory_transformer 查看所有轨迹变换器模型
]


# 加载 TensorFlow 检查点到 PyTorch 模型的函数
def load_tf_weights_in_trajectory_transformer(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re  # 正则表达式库

        import numpy as np  # 处理数组和矩阵的库
        import tensorflow as tf  # TensorFlow 深度学习库
    except ImportError:
        # 如果导入失败，输出错误信息并提醒用户需要安装 TensorFlow
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise  # 抛出 ImportError 异常
    tf_path = os.path.abspath(tf_checkpoint_path)  # 获取 TensorFlow 检查点文件的绝对路径
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")  # 输出日志，提示正在从 TensorFlow 检查点转换
    # 从 TF 模型加载权重
    init_vars = tf.train.list_variables(tf_path)  # 获取 TF 模型中的变量列表
    names = []  # 存储变量名
    arrays = []  # 存储变量值
    for name, shape in init_vars:  # 遍历 TF 模型中的每个变量名和形状
        logger.info(f"Loading TF weight {name} with shape {shape}")  # 输出日志，提示正在加载 TF 权重
        array = tf.train.load_variable(tf_path, name)  # 加载 TF 模型中的权重值
        names.append(name)  # 将变量名添加到列表中
        arrays.append(array)  # 将权重值添加到列表中
    for name, array in zip(names, arrays):
        # 将文件名按 "/" 分割，用于后续处理
        name = name.split("/")
        # 如果文件名中包含指定的变量名，则跳过当前循环，不处理该变量
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            # 记录日志，跳过当前处理的变量
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        pointer = model
        # 遍历文件名各部分，根据文件名的不同部分做不同处理
        for m_name in name:
            # 如果 m_name 符合指定格式，则按照 "_数字" 进行分割
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            # 根据文件名的不同部分对应不同的处理方法
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
                    # 记录日志，跳过当前处理的变量
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            # 如果文件名有多个部分，则根据数字部分更新指针位置
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        # 如果文件名以 "_embeddings" 结尾，则将指针指向权重
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        # 如果文件名为 "kernel"，则对 array 进行转置处理
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            # 检查指针和 array 的形状是否匹配，不匹配则抛出异常
            if pointer.shape != array.shape:
                raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        # 记录日志，初始化 PyTorch 权重
        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)
    # 返回处理后的模型
    return model
@dataclass
class TrajectoryTransformerOutput(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`Tuple[Tuple[torch.Tensor]]`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of length `config.n_layers`, containing tuples of tensors of shape `(batch_size, num_heads,
            sequence_length, embed_size_per_head)`). Contains pre-computed hidden-states (key and values in the
            attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. GPT2Attentions weights after the attention softmax, used to compute the weighted average
            in the self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None  # 语言建模损失，如果提供了标签，则返回损失值
    logits: torch.FloatTensor = None  # 语言建模头部的预测分数（SoftMax之前的每个词汇标记的分数）
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None  # 预先计算的隐藏状态，用于加速顺序解码
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 模型每一层输出的隐藏状态，包括初始嵌入输出
    attentions: Optional[Tuple[torch.FloatTensor]] = None  # 自注意力头部的GPT2注意力权重，在注意力softmax之后使用，用于计算加权平均值
                                                             
class TrajectoryTransformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = TrajectoryTransformerConfig  # 配置类，用于初始化模型
    load_tf_weights = load_tf_weights_in_trajectory_transformer  # 加载 TensorFlow 权重的方法
    base_model_prefix = "trajectory_transformer"  # 模型的基础名称前缀
    main_input_name = "trajectories"  # 主要输入的名称
    supports_gradient_checkpointing = True  # 是否支持梯度检查点
    # 初始化模型的权重
    def _init_weights(self, module):
        # 如果是线性层或者嵌入层，使用正态分布初始化权重，均值为0，标准差为配置中的initializer_range
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果是线性层且存在偏置，初始化偏置为0
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        # 如果是LayerNorm层，初始化偏置为0，权重为1
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        # 如果是自定义的EinLinear层
        elif isinstance(module, EinLinear):
            # 遍历每个模型，使用kaiming_uniform_初始化权重
            for i in range(module.n_models):
                nn.init.kaiming_uniform_(module.weight[i], a=math.sqrt(5) / self.config.kaiming_initializer_range)
                # 如果存在偏置，根据fan_in计算边界，使用uniform_初始化偏置
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight[i])
                    bound = (1 / math.sqrt(fan_in)) * self.config.initializer_range
                    nn.init.uniform_(module.bias[i], -bound, bound)
# 定义了 TrajectoryTransformer 类的模型文档字符串，说明这个模型是一个 PyTorch 的子类，可以像普通 PyTorch 模块一样使用，需要参考 PyTorch 文档了解通用用法和行为
TRAJECTORY_TRANSFORMER_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`TrajectoryTransformerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义了 TrajectoryTransformer 类的输入文档字符串，说明了模型的输入参数和选项
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

# 定义了 EinLinear 类，作为 nn.Module 的子类
class EinLinear(nn.Module):
    def __init__(self, n_models, in_features, out_features, bias):
        # 初始化函数，初始化神经网络层
        super().__init__()
        # 设置参数 n_models、in_features 和 out_features
        self.n_models = n_models
        self.out_features = out_features
        self.in_features = in_features
        # 创建权重参数张量
        self.weight = nn.Parameter(torch.Tensor(n_models, out_features, in_features))
        # 如果需要偏置，创建偏置参数张量
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_models, out_features))
        else:
            # 否则将偏置参数注册为 None
            self.register_parameter("bias", None)

    def reset_parameters(self):
        # 重置权重和偏置参数的值
        for i in range(self.n_models):
            # 使用 kaiming 均匀分布初始化权重参数
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
            # 如果偏置参数不为 None
            if self.bias is not None:
                # 计算输入和输出的扇入和扇出，计算初始化边界
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[i])
                bound = 1 / math.sqrt(fan_in)
                # 使用均匀分布初始化偏置参数
                nn.init.uniform_(self.bias[i], -bound, bound)

    def forward(self, input):
        """
        Args:
            input (`torch.FloatTensor` of shape `(B, n_models, input_dim)`):
                The input to the layer.
        """
        # 使用 einsum 函数进行张量乘法，计算输出
        # [ batch_size x n_models x output_dim ]
        output = torch.einsum("eoi,bei->beo", self.weight, input)
        # 如果偏置参数不为 None，则触发 RuntimeError
        if self.bias is not None:
            raise RuntimeError()
        # 返回输出结果
        return output
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        # 检查 n_embd 是否能被 n_head 整除，如果不能则抛出数值错误
        if config.n_embd % config.n_head != 0:
            raise ValueError(f"n_head ({config.n_head}) should be a divisor of n_embd ({config.n_embd})")

        # 为所有的注意力头创建 key, query, value 的投影
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        # 正则化处理
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)

        # 输出投影
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        # 创建因果遮罩，确保注意力只应用于输入序列的左侧
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
            persistent=False,
        )

        # 遮罩之前的值估计
        joined_dim = config.observation_dim + config.action_dim + 2
        self.mask.squeeze()[:, joined_dim - 1 :: joined_dim] = 0

        self.n_head = config.n_head

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    def forward(self, hidden_states, layer_past=None, use_cache=False, output_attentions=False):
        # 解压隐藏状态的维度
        batch_size, sequence_length, embedding_dim = hidden_states.size()
    
        # 为所有头部的批次计算查询、键、值，并将头向前移为批次维度
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
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
    
        if use_cache is True:
            present = (key, value)
        else:
            present = None
    
        # 因果自注意力
        # [ batch_size x n_heads x sequence_length x sequence_length ]
        attn_weights = (torch.matmul(query, key.transpose(-2, -1))) * (1.0 / math.sqrt(key.size(-1)))
        attn_weights = attn_weights.masked_fill(
            self.mask[:, :, :sequence_length, :sequence_length] == 0, torch.finfo(attn_weights.dtype).min
        )
        attn_weights = F.softmax(attn_weights, dim=-1)
        self._attn_map = attn_weights.clone()
        attn_weights = self.attn_drop(attn_weights)
    
        output = torch.matmul(attn_weights, value)
        # [ batch_size x sequence_length x embedding_dim ]
        # 重新组装所有头部输出
        output = output.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_dim)
    
        # 输出投影
        output = self.resid_drop(self.proj(output))
    
        outputs = (output, present)
        if output_attentions:
            outputs += (attn_weights,)
    
        return outputs
class Block(nn.Module):
    # 定义一个 Block 类，继承自 nn.Module
    def __init__(self, config):
        # 初始化方法
        super().__init__()
        # 调用父类的初始化方法

        # LayerNorm 层1
        self.ln1 = nn.LayerNorm(config.n_embd)
        # LayerNorm 层2
        self.ln2 = nn.LayerNorm(config.n_embd)
        # 自注意力机制
        self.attn = CausalSelfAttention(config)

        # MLP
        self.l1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        # GELU 激活函数
        self.act = nn.GELU()
        self.l2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.drop = nn.Dropout(config.resid_pdrop)
        # 初始化 MLP

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        # 定义前向传播方法
        residual = hidden_states
        # 保存 hidden states 到 residual
        hidden_states = self.ln1(hidden_states)
        # LayerNorm 1

        # 调用自注意力机制
        attn_outputs = self.attn(
            hidden_states, layer_past=layer_past, use_cache=use_cache, output_attentions=output_attentions
        )
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        hidden_states = attn_output + residual
        # 更新 hidden states

        residual = hidden_states
        # 保存当前 hidden states 到 residual
        hidden_states = self.ln2(hidden_states)
        # LayerNorm 2
        hidden_states = self.l1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.l2(hidden_states)
        hidden_states = residual + self.drop(hidden_states)
        # 更新 hidden states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]
        # 生成输出结果

        return outputs


@add_start_docstrings(
    "The bare TrajectoryTransformer Model transformer outputting raw hidden-states without any specific head on top.",
    TRAJECTORY_TRANSFORMER_START_DOCSTRING,
)
# 添加文档注释
class TrajectoryTransformerModel(TrajectoryTransformerPreTrainedModel):
    """the full GPT language model, with a context size of block_size"""
    # 定义 TrajectoryTransformerModel 类，扩展自 TrajectoryTransformerPreTrainedModel
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 输入嵌入的词干（+1 用于停止令牌）
        self.tok_emb = nn.Embedding(config.vocab_size * config.transition_dim + 1, config.n_embd)

        # 位置嵌入
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        # 丢弃层
        self.drop = nn.Dropout(config.embd_pdrop)
        # Transformer 块
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        # 解码器头部
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = EinLinear(config.transition_dim, config.n_embd, config.vocab_size + 1, bias=False)

        # 词汇表大小
        self.vocab_size = config.vocab_size
        # 停止令牌
        self.stop_token = config.vocab_size * config.transition_dim
        # 块大小
        self.block_size = config.block_size

        # 观察维度
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

        # 梯度检查点
        self.gradient_checkpointing = False

        # 执行后初始化
        self.post_init()

    def get_block_size(self):
        # 返回块大小
        return self.block_size

    def offset_tokens(self, trajectories):
        _, sequence_length = trajectories.shape

        # 计算状态数量
        n_states = int(np.ceil(sequence_length / self.transition_dim))

        # 计算偏移量
        offsets = torch.arange(self.transition_dim) * self.vocab_size
        offsets = offsets.repeat(n_states).to(trajectories.device)

        # 偏移轨迹
        offset_trajectories = trajectories + offsets[:sequence_length]
        offset_trajectories[trajectories == self.vocab_size] = self.stop_token
        return offset_trajectories

    def pad_to_full_observation(self, hidden_states):
        batch_size, sequence_length, _ = hidden_states.shape

        # 计算需要填充的数量
        n_pad = (self.transition_dim - sequence_length % self.transition_dim) % self.transition_dim
        # 创建填充张量
        padding = torch.zeros(batch_size, n_pad, self.embedding_dim, device=hidden_states.device)

        # [ batch_size x padded_sequence_length' x embedding_dim ]
        # 将填充的张量与隐藏状态拼接
        hidden_states_pad = torch.cat([hidden_states, padding], dim=1)
        hidden_states_pad = hidden_states_pad.view(-1, self.transition_dim, self.embedding_dim)

        return hidden_states_pad, n_pad

    @add_start_docstrings_to_model_forward(
        TRAJECTORY_TRANSFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(output_type=TrajectoryTransformerOutput, config_class=_CONFIG_FOR_DOC)
    # 使用参数传递神经网络的前向计算
    def forward(
        # 输入的轨迹数据，类型为可选的长整型张量
        trajectories: Optional[torch.LongTensor] = None,
        # 用于之前计算的键-值对的元组
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        # 目标数据
        targets: Optional[torch.FloatTensor] = None,
        # 注意力遮罩
        attention_mask: Optional[torch.FloatTensor] = None,
        # 是否使用缓存
        use_cache: Optional[bool] = None,
        # 输出注意力分布
        output_attentions: Optional[bool] = None,
        # 输出隐藏状态
        output_hidden_states: Optional[bool] = None,
        # 返回字典
        return_dict: Optional[bool] = None,
```