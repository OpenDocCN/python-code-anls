# `.\transformers\models\megatron_bert\modeling_megatron_bert.py`

```
# 设置文件编码为 utf-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，对代码进行许可
# 你可以在遵守许可证的情况下使用此文件
# 可以在 http://www.apache.org/licenses/LICENSE-2.0 获取许可证的副本
# 除非适用法律要求或书面同意，否则按“原样”分发软件
# 没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制
""" PyTorch MegatronBERT model."""

# 导入所需的库
import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入相关模块和类
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_megatron_bert import MegatronBertConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的配置和检查点
_CONFIG_FOR_DOC = "MegatronBertConfig"
_CHECKPOINT_FOR_DOC = "nvidia/megatron-bert-cased-345m"

# MegatronBERT 预训练模型存档列表
MEGATRON_BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "nvidia/megatron-bert-cased-345m",
    # 查看所有 MegatronBERT 模型 https://huggingface.co/models?filter=megatron_bert
]

# 加载 TensorFlow 模型权重到 PyTorch 模型
def load_tf_weights_in_megatron_bert(model, config, tf_checkpoint_path):
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
    # 获取 TensorFlow 检查点的绝对路径
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # 从 TF 模型加载权重
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    # 遍历初始化变量的名称和形状
    for name, shape in init_vars:
        # 记录日志，显示正在加载的 TF 权重的名称和形状
        logger.info(f"Loading TF weight {name} with shape {shape}")
        # 使用 TensorFlow 加载变量数据
        array = tf.train.load_variable(tf_path, name)
        # 将名称添加到列表中
        names.append(name)
        # 将数据添加到列表中
        arrays.append(array)

    # 遍历名称和数据的列表
    for name, array in zip(names, arrays):
        # 将名称拆分为子名称
        name = name.split("/")
        # 检查是否为不需要的变量，如果是则跳过
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        # 初始化指针为模型
        pointer = model
        # 遍历子名称
        for m_name in name:
            # 如果子名称匹配特定模式，则拆分为作用域名称和数字
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            # 根据作用域名称更新指针位置
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
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            # 如果作用域名称包含数字，则更新指针位置
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        # 如果名称以 "_embeddings" 结尾，则更新指针位置
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        # 如果名称为 "kernel"，则转置数组
        elif m_name == "kernel":
            array = np.transpose(array)
        # 检查指针和数组的形状是否匹配，如果不匹配则引发错误
        if pointer.shape != array.shape:
            raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
        # 记录日志，显示正在初始化的 PyTorch 权重的名称
        logger.info("Initialize PyTorch weight {}".format(name))
        # 将数组转换为 PyTorch 张量，并赋值给指针
        pointer.data = torch.from_numpy(array)
    # 返回模型
    return model
class MegatronBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        # 初始化函数，构建 MegatronBertEmbeddings 类
        super().__init__()
        # 创建词嵌入层，词嵌入大小为 config.hidden_size，词汇表大小为 config.vocab_size
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 创建位置嵌入层，位置嵌入大小为 config.hidden_size，最大位置嵌入为 config.max_position_embeddings
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 创建 token 类型嵌入层，token 类型嵌入大小为 config.hidden_size，token 类型数为 config.type_vocab_size

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file

        # 在 Megatron 中，layer-norm 在第一个 dropout 之后应用。
        # self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建 dropout 层，概率为 config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        # 注册 position_ids 缓冲区，用于存储位置 id，长度为 config.max_position_embeddings
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 设置位置嵌入类型，默认为 "absolute"
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        # 前向传播函数
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            # 如果位置 id 为空，则从 position_ids 中获取位置 id
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if token_type_ids is None:
            # 如果 token 类型 id 为空���则创建全零的 token 类型 id
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            # 如果输入的嵌入为空，则使用词嵌入层获取嵌入
            inputs_embeds = self.word_embeddings(input_ids)
        # 获取 token 类型嵌入
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将词嵌入和 token 类型嵌入相加
        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            # 如果位置嵌入类型为 "absolute"，则获取位置嵌入并加到 embeddings 中
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        # Megatron BERT 将 layer norm 放在 drop-out 之后（每一层都有）
        # embeddings = self.LayerNorm(embeddings)
        # 对 embeddings 进行 drop-out 处理
        embeddings = self.dropout(embeddings)
        return embeddings


# Copied from transformers.models.bert.modeling_bert.BertSelfAttention with Bert->MegatronBert
class MegatronBertSelfAttention(nn.Module):
    # 初始化函数，接受配置和位置嵌入类型作为参数
    def __init__(self, config, position_embedding_type=None):
        # 调用父类的初始化函数
        super().__init__()
        # 检查隐藏层大小是否是注意力头数的倍数，如果不是则抛出数值错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 初始化注意力头数和每个注意力头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化查询、键、值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 初始化丢弃层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # 设置位置嵌入类型
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果位置嵌入类型是相对键或相对键查询，则初始化距离嵌入
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        # 设置是否为解码器
        self.is_decoder = config.is_decoder

    # 将输入张量转置以便进行注意力计算
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播函数，接受隐藏状态、注意力掩码、头掩码、编码器隐藏状态、编码器注意力掩码、过去的键值对、输出注意力等参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
# 基于 transformers.models.bert.modeling_bert.BertSelfOutput。将 LayerNorm 移动到 MegatronBertAttention 下面。
class MegatronBertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # 定义一个全连接层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # 定义一个丢弃层

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        # 将输入的隐藏状态通过全连接层得到输出
        hidden_states = self.dense(hidden_states)
        # 对输出进行丢弃处理
        hidden_states = self.dropout(hidden_states)
        # 返回残差加上处理后的隐藏状态
        return residual + hidden_states


# 基于 transformers.models.bert.modeling_bert.BertAttention。添加了 LayerNorm。
class MegatronBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # 初始化 LayerNorm 层
        self.self = MegatronBertSelfAttention(config)  # 创建 MegatronBertSelfAttention 实例
        self.output = MegatronBertSelfOutput(config)  # 创建 MegatronBertSelfOutput 实例
        self.pruned_heads = set()  # 创建一个空的 set 集合用于存储被修剪的头部

    def prune_heads(self, heads):
        if len(heads) == 0:  # 如果输入的头部数为0，则直接返回
            return
        # 调用 find_pruneable_heads_and_indices 函数找到可修剪的头部和其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 修剪线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储被修剪的头部
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 对隐藏状态进行 LayerNorm 处理
        ln_outputs = self.ln(hidden_states)
        # 调用 self 函数进行自注意力操作
        self_outputs = self.self(
            ln_outputs,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 对注意力输出进行处理
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出注意力，则将注意力也加入输出中
        outputs = (attention_output,) + self_outputs[1:]
        # 返回处理后的结果
        return outputs


# 从 transformers.models.bert.modeling_bert.BertIntermediate 复制过来，将Bert->MegatronBert
class MegatronBertIntermediate(nn.Module):
    def __init__(self, config):
            # 初始化函数，继承父类的初始化函数
            super().__init__()
            # 创建一个全连接层，输入维度为config.hidden_size，输出维度为config.intermediate_size
            self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
            # 判断config.hidden_act是否为字符串，如果是则使用对应的激活函数，否则直接使用config.hidden_act传入的函数
            if isinstance(config.hidden_act, str):
                self.intermediate_act_fn = ACT2FN[config.hidden_act]
            else:
                self.intermediate_act_fn = config.hidden_act
    
        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            # 将输入hidden_states经过全连接层dense处理
            hidden_states = self.dense(hidden_states)
            # 将处理后的hidden_states经过激活函数处理
            hidden_states = self.intermediate_act_fn(hidden_states)
            # 返回处理后的hidden_states
            return hidden_states
# 基于transformers.models.bert.modeling_bert.BertOutput。将LayerNorm移到下面的MegatronBertLayer中
class MegatronBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，将输入大小为config.intermediate_size，输出大小为config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个丢弃层，采用config.hidden_dropout_prob的丢弃概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态进行全连接操作
        hidden_states = self.dense(hidden_states)
        # 对全连接结果进行丢弃操作
        hidden_states = self.dropout(hidden_states)
        # 返回输入张量与处理后的隐藏状态的加和结果
        return input_tensor + hidden_states


# 基于transformers.models.bert.modeling_bert.BertLayer。添加LayerNorm。
class MegatronBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 设置feed forward的chunk大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 设置序列长度维度
        self.seq_len_dim = 1
        # 创建BertAttention层
        self.attention = MegatronBertAttention(config)
        # 判断是否为decoder模型
        self.is_decoder = config.is_decoder
        # 判断是否添加cross attention
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                # 如果不是decoder模型，则抛出异常
                raise TypeError(f"{self} should be used as a decoder model if cross attention is added")
            # 创建cross attention层
            self.crossattention = MegatronBertAttention(config)
        # 创建LayerNorm层，设置隐藏大小为config.hidden_size，epsilon值为config.layer_norm_eps
        self.ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建MegatronBertIntermediate层
        self.intermediate = MegatronBertIntermediate(config)
        # 创建MegatronBertOutput层
        self.output = MegatronBertOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 当前层 uni-directional self-attention 的缓存的 key/values 元组在位置 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # 如果是解码器，最后一个输出是 self-attn 缓存的元组
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # 如果需要输出注意力权重，则添加 self attentions

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise AttributeError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross-attn 缓存的 key/values 元组在 past_key_value 元组的位置 3,4
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # 如果需要输出注意力权重，则添加 cross attentions

            # 将 cross-attn 缓存添加到 present_key_value 元组的位置 3,4
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # 如果是解码器，将 attn key/values 作为最后一个输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    # 将前向传播操作分块执行
    def feed_forward_chunk(self, attention_output):
        ln_output = self.ln(attention_output)
        intermediate_output = self.intermediate(ln_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
# 定义 MegatronBertEncoder 类，继承自 nn.Module
class MegatronBertEncoder(nn.Module):
    # 初始化方法
    def __init__(self, config):
        # 调用父类初始化方法
        super().__init__()
        # 存储配置参数
        self.config = config
        # 创建 nn.ModuleList，包含多个 MegatronBertLayer 对象
        self.layer = nn.ModuleList([MegatronBertLayer(config) for _ in range(config.num_hidden_layers)])

        # 创建最终的 LayerNorm 层，Transformer 的 BERT 将每层的 LayerNorm 与隐藏层连接，这个是最终的 LayerNorm
        self.ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 是否梯度检查
        self.gradient_checkpointing = False

    # 前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        
# 定义 MegatronBertPooler 类，继承自 nn.Module
class MegatronBertPooler(nn.Module):
    # 初始化方法
    def __init__(self, config):
        # 调用父类初始化方法
        super().__init__()
        # 全连接层，输入输出维度都是 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 激活函数 Tanh
        self.activation = nn.Tanh()

    # 前向传播方法
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过取第一个 token 的隐藏状态 "pool" 模型
        first_token_tensor = hidden_states[:, 0]
        # 通过全连接层得到池化输出
        pooled_output = self.dense(first_token_tensor)
        # 经过激活函数
        pooled_output = self.activation(pooled_output)
        return pooled_output

# 定义 MegatronBertPredictionHeadTransform 类，继承自 nn.Module
class MegatronBertPredictionHeadTransform(nn.Module):
    # 初始化方法
    def __init__(self, config):
        # 调用父类初始化方法
        super().__init__()
        # 全连接层，输入输出维度都是 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 隐藏层激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # LayerNorm 层
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # 前向传播方法
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 经过全连接层
        hidden_states = self.dense(hidden_states)
        # 经过隐藏层激活函数
        hidden_states = self.transform_act_fn(hidden_states)
        # LayerNorm 处理
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

# 定义 MegatronBertLMPredictionHead 类，继承自 nn.Module
class MegatronBertLMPredictionHead(nn.Module):
    # 这个类继承自基类，初始化相关配置
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个 MegatronBertPredictionHeadTransform 对象，用于对隐藏状态进行变换
        self.transform = MegatronBertPredictionHeadTransform(config)
    
        # 创建一个线性层用于解码，将隐藏状态映射到词汇表大小的输出
        # 这个线性层没有偏置项
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
        # 创建一个可学习的偏置项
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
    
        # 将偏置项与解码器的偏置项绑定在一起，确保它们大小一致
        self.decoder.bias = self.bias
    
    # 前向传播方法
    def forward(self, hidden_states):
        # 使用 transform 对象对隐藏状态进行变换
        hidden_states = self.transform(hidden_states)
        # 使用解码器将变换后的隐藏状态映射到词汇表大小的输出
        hidden_states = self.decoder(hidden_states)
        # 返回输出结果
        return hidden_states
# 从transformers.models.bert.modeling_bert.BertOnlyMLMHead复制代码，将Bert->MegatronBert
class MegatronBertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建 MegatronBertLMPredictionHead 对象
        self.predictions = MegatronBertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        # 通过 MegatronBertLMPredictionHead 模块预测序列的下一步
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


# 从transformers.models.bert.modeling_bert.BertOnlyNSPHead复制代码，将Bert->MegatronBert
class MegatronBertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建 nn.Linear 对象，用于序列关系评分
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        # 通过 nn.Linear 模块计算池化后的输出的序列关系评分
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


# 从transformers.models.bert.modeling_bert.BertPreTrainingHeads复制代码，将Bert->MegatronBert
class MegatronBertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建 MegatronBertLMPredictionHead 和 nn.Linear 对象
        self.predictions = MegatronBertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        # 分别通过 MegatronBertLMPredictionHead 和 nn.Linear 模块计算预测分数和序列关系分数
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class MegatronBertPreTrainedModel(PreTrainedModel):
    """
    处理权重初始化和预训练模型下载/加载的抽象类。
    """

    config_class = MegatronBertConfig
    load_tf_weights = load_tf_weights_in_megatron_bert
    base_model_prefix = "bert"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # 使用标准正态分布初始化权重，与TF版本有细微差别，TF版本使用截断正态分布
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            # 将 LayerNorm 的偏置初始化为0，权重初始化为1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            # 如果是 Linear 模块，并且有偏置，则初始化偏置为0
            module.bias.data.zero_()


@dataclass
# 从transformers.models.bert.modeling_bert.BertForPreTrainingOutput复制代码，将Bert->MegatronBert
class MegatronBertForPreTrainingOutput(ModelOutput):
    """
    [`MegatronBertForPreTraining`] 的输出类型。
    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (`torch.FloatTensor` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    # 定义可选的损失值，如果`labels`被提供，则返回，是形状为`(1,)`的`torch.FloatTensor`
    loss: Optional[torch.FloatTensor] = None
    # 定义预测logits，形状为`(batch_size, sequence_length, config.vocab_size)`的`torch.FloatTensor`
    prediction_logits: torch.FloatTensor = None
    # 定义序列关系logits，形状为`(batch_size, 2)`的`torch.FloatTensor`
    seq_relationship_logits: torch.FloatTensor = None
    # 定义隐藏状态，可选的元组，当`output_hidden_states=True`被传递或`config.output_hidden_states=True`时返回，
    # 元组中包含`torch.FloatTensor`，形状为`(batch_size, sequence_length, hidden_size)`
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义注意力权重，可选的元组，当`output_attentions=True`被传递或`config.output_attentions=True`时返回，
    # 元组中包含`torch.FloatTensor`，形状为`(batch_size, num_heads, sequence_length, sequence_length)`
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# 定义 MEGATRON_BERT_START_DOCSTRING，这个字符串用于文档注释，解释了该模型继承自 PreTrainedModel，以及它是一个 PyTorch 的 nn.Module 子类，因此可以像普通的 PyTorch 模块一样使用。
# 这个字符串还解释了模型参数的含义和配置，以及如何使用 from_pretrained 方法加载模型权重。
# MEGATRON_BERT_INPUTS_DOCSTRING 目前为空，可能是为后续输入参数的文档预留的空间。
    # 定义函数的输入参数和其对应的类型和形状
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列标记在词汇表中的索引
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 用于避免在填充标记索引上执行注意力的掩码。掩码值选取范围在`[0, 1]`之间，1表示不被掩盖，0表示被掩盖
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 指示输入的第一部分和第二部分的段标记索引
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 每个输入序列标记在位置嵌入中的位置索引
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于使自注意力模块中的特定头失效的掩码
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选择直接传递嵌入表示，而不是传递`input_ids`，这对于控制如何将`input_ids`索引转换为相关向量非常有用
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态
        return_dict (`bool`, *optional*):
            # 是否返回`~utils.ModelOutput`而不是普通元组
"""

# 导入必要的库
@add_start_docstrings(
    "The bare MegatronBert Model transformer outputting raw hidden-states without any specific head on top.",
    MEGATRON_BERT_START_DOCSTRING,
)
class MegatronBertModel(MegatronBertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        # 调用父类的构造函数
        super().__init__(config)
        # 保存配置信息
        self.config = config

        # 初始化 MegatronBertEmbeddings 层
        self.embeddings = MegatronBertEmbeddings(config)
        # 初始化 MegatronBertEncoder 层
        self.encoder = MegatronBertEncoder(config)

        # 如果需要添加池化层，则初始化 MegatronBertPooler 层
        self.pooler = MegatronBertPooler(config) if add_pooling_layer else None

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # 裁剪模型的注意力头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # MegatronBertModel 的前向传播方法
    @add_start_docstrings_to_model_forward(MEGATRON_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 定义 MegatronBertForPreTraining 类，包含两个预训练任务头部：`masked language modeling` 头部和 `next sentence prediction` 头部
@add_start_docstrings(
    """
    MegatronBert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a
    `next sentence prediction (classification)` head.
    """,
    MEGATRON_BERT_START_DOCSTRING,
)
class MegatronBertForPreTraining(MegatronBertPreTrainedModel):

    # 预训练模型中需要共享权重的部分
    _tied_weights_keys = ["cls.predictions.decoder"]

    # 初始化方法
    def __init__(self, config, add_binary_head=True):
        # 调用父类的初始化方法
        super().__init__(config)

        # 初始化 MegatronBertModel 模型
        self.bert = MegatronBertModel(config)
        # 初始化 MegatronBertPreTrainingHeads 模型
        self.cls = MegatronBertPreTrainingHeads(config)

        # 初始化权重和应用最终处理
        self.post_init()

    # 获取输出的嵌入
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出的嵌入
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 前向传播方法
    @add_start_docstrings_to_model_forward(MEGATRON_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=MegatronBertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        # 输入参数
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        next_sentence_label: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,



# 定义 MegatronBertForCausalLM 类，包含一个 `language modeling` 头部，用于 CausalLM 微调
@add_start_docstrings(
    """MegatronBert Model with a `language modeling` head on top for CLM fine-tuning.""",
    MEGATRON_BERT_START_DOCSTRING,
)
class MegatronBertForCausalLM(MegatronBertPreTrainedModel):

    # 需要共享权重的部分
    _tied_weights_keys = ["cls.predictions.decoder"]

    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 如果不是解码器，发出警告
        if not config.is_decoder:
            logger.warning("If you want to use `MegatronBertForCausalLM` as a standalone, add `is_decoder=True.`")

        # 初始化 MegatronBertModel 模型
        self.bert = MegatronBertModel(config, add_pooling_layer=False)
        # 初始化 MegatronBertOnlyMLMHead 模型
        self.cls = MegatronBertOnlyMLMHead(config)

        # 初始化权重和应用最终处理
        self.post_init()

    # 获取输出的嵌入
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出的嵌入
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 前向传播方法
    @add_start_docstrings_to_model_forward(MEGATRON_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    # 定义模型的前向传播函数
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 准备模型的输入
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        # 如果没有提供注意力掩码，则创建一个全1的矩阵作为注意力掩码
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)
    
        # 如果使用了过去的关键值，需要根据过去关键值的长度裁剪输入ID
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            # 如果输入ID的长度大于过去关键值的长度，则只保留后面的部分
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            # 否则，只保留最后一个ID
            else:
                remove_prefix_length = input_ids.shape[1] - 1
    
            input_ids = input_ids[:, remove_prefix_length:]
    
        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}
    
    # 重新排序缓存的关键值
    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
# 增加开始注释为“MegatronBert Model with a `language modeling` head on top.”，继承了 MegatronBertPreTrainedModel 类
class MegatronBertForMaskedLM(MegatronBertPreTrainedModel):
    # 定义共享权重的键值列表
    _tied_weights_keys = ["cls.predictions.decoder"]

    # 初始化方法，接受配置参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 如果是解码器，则警告
        if config.is_decoder:
            logger.warning(
                "If you want to use `MegatronBertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 创建 MegatronBertModel 实例
        self.bert = MegatronBertModel(config, add_pooling_layer=False)
        # 创建 MegatronBertOnlyMLMHead 实例
        self.cls = MegatronBertOnlyMLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出embedding
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出embedding
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 定义前向传播方法，接受多个输入参数
    @add_start_docstrings_to_model_forward(MEGATRON_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 添加代码示例文档字符串
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 定义一个 Masked Language Model 输出类型，包含损失、预测得分、隐藏状态和注意力权重
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple, MaskedLMOutput]:
        # 检查是否设置了 return_dict，如果没有则使用配置文件中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # 通过 BERT 模型获取序列输出和其他输出
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
        # 获取序列输出
        sequence_output = outputs[0]
        # 使用分类头进行预测
        prediction_scores = self.cls(sequence_output)
    
        masked_lm_loss = None
        # 如果提供了标签，则计算 Masked Language Model 损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    
        # 如果不使用 return_dict，则返回预测得分和其他输出
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
    
        # 否则返回 MaskedLMOutput 对象
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    # 准备输入用于生成
    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]
    
        # 如果没有设置 PAD 令牌，就抛出异常
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")
    
        # 在注意力掩码后面添加一个虚拟令牌
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)
    
        return {"input_ids": input_ids, "attention_mask": attention_mask}
# 添加对话框开头的文档字符串，描述了这个 Megatron Bert 模型带有一个“下一个句子预测（分类）”头部的结构
@add_start_docstrings(
    """MegatronBert Model with a `next sentence prediction (classification)` head on top.""",
    MEGATRON_BERT_START_DOCSTRING,
)
# 定义 MegatronBert 模型的下一个句子预测类
class MegatronBertForNextSentencePrediction(MegatronBertPreTrainedModel):
    # 初始化方法，接受配置参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建 MegatronBert 模型
        self.bert = MegatronBertModel(config)
        # 创建 MegatronBert 的下一个句子预测头
        self.cls = MegatronBertOnlyNSPHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法，接受多个输入参数，包括输入 ID、注意力掩码、标记类型 ID 等等
    @add_start_docstrings_to_model_forward(MEGATRON_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=NextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, NextSentencePredictorOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see `input_ids` docstring). Indices should be in `[0, 1]`:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MegatronBertForNextSentencePrediction
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("nvidia/megatron-bert-cased-345m")
        >>> model = MegatronBertForNextSentencePrediction.from_pretrained("nvidia/megatron-bert-cased-345m")

        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
        >>> encoding = tokenizer(prompt, next_sentence, return_tensors="pt")

        >>> outputs = model(**encoding, labels=torch.LongTensor([1]))
        >>> logits = outputs.logits
        >>> assert logits[0, 0] < logits[0, 1]  # next sentence was random
        ```"""

        # 如果 kwargs 中包含 'next_sentence_label'，则发出警告，并将其作为 'labels' 参数
        if "next_sentence_label" in kwargs:
            warnings.warn(
                "The `next_sentence_label` argument is deprecated and will be removed in a future version, use"
                " `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("next_sentence_label")

        # 确定是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 BERT 模型进行前向传播
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从 BERT 输出中提取汇总的输出
        pooled_output = outputs[1]

        # 使用分类层预测下一个句子的关系得分
        seq_relationship_scores = self.cls(pooled_output)

        # 初始化下一个句子损失
        next_sentence_loss = None
        # 如果提供了标签，则计算下一个句子的损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))

        # 如果不需要返回字典，则返回包含相关输出的元组
        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

        # 返回下一个句子预测器的输出对象
        return NextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用装饰器为 MegatronBertForSequenceClassification 类添加文档字符串，描述其作用为在 Megatron BERT 模型的基础上增加了一个用于序列分类/回归的头部
# 在头部上添加了一个线性层（线性变换）来处理池化后的输出，例如用于 GLUE 任务
@add_start_docstrings(
    """
    MegatronBert Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    MEGATRON_BERT_START_DOCSTRING,
)
class MegatronBertForSequenceClassification(MegatronBertPreTrainedModel):
    # 初始化函数，接受一个配置参数 config
    def __init__(self, config):
        # 调用父类 MegatronBertPreTrainedModel 的初始化函数
        super().__init__(config)
        # 设置类别数为配置参数中的 num_labels
        self.num_labels = config.num_labels

        # 初始化 Megatron BERT 模型
        self.bert = MegatronBertModel(config)
        # 初始化一个 dropout 层，根据配置参数中的 hidden_dropout_prob 来进行随机失活
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 初始化一个线性层，输入维度为配置参数中的 hidden_size，输出维度为配置参数中的 num_labels
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 调用后处理函数来初始化权重并进行最终处理
        self.post_init()

    # 前向传播函数，接受一系列输入张量，并返回输出结果
    @add_start_docstrings_to_model_forward(MEGATRON_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 确定是否使用默认的返回字典，若不是则使用模型配置中的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 BERT 模型处理输入，并获取输出
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取池化后的输出
        pooled_output = outputs[1]

        # 对池化后的输出进行 dropout 处理
        pooled_output = self.dropout(pooled_output)
        # 将处理后的输出传入分类器，得到 logits
        logits = self.classifier(pooled_output)

        loss = None
        # 如果存在标签，则计算损失
        if labels is not None:
            # 检查问题类型是否已经指定，若未指定，则根据标签值和数量确定
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型计算相应的损失
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
        # 若不需要返回字典，则返回输出值和可能的损失
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回值为 SequenceClassifierOutput 对象，包含损失、logits、隐藏状态和注意力值
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用装饰器添加模型文档字符串，描述了该模型是一个 MegatronBert 模型，用于多选分类任务
class MegatronBertForMultipleChoice(MegatronBertPreTrainedModel):
    # 初始化函数，接受配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)
        
        # 创建 MegatronBert 模型
        self.bert = MegatronBertModel(config)
        # 创建一个丢弃层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 创建一个线性层，用于多选分类
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    # 使用装饰器添加模型前向传播的文档字符串
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # ...
    ) -> Union[Tuple, MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 设置返回字典的标志位，如果未提供，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 获取输入张量的第二维大小，即选择数量
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 对输入张量进行形状变换，将其视为二维张量，其中第一维度为 batch_size * num_choices，第二维度保持不变
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 将输入传递给 BERT 模型，获取输出
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从 BERT 输出中提取池化后的输出
        pooled_output = outputs[1]

        # 对池化后的输出应用 dropout
        pooled_output = self.dropout(pooled_output)
        # 使用分类器对池化后的输出进行分类
        logits = self.classifier(pooled_output)
        # 重新调整 logits 的形状，使其变为二维张量
        reshaped_logits = logits.view(-1, num_choices)

        # 如果提供了标签，则计算损失
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 如果不要求返回字典，则按顺序返回相应的输出
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果要求返回字典，则构建 MultipleChoiceModelOutput 对象并返回
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用 MegatronBert 模型进行标记分类的模型，例如用于命名实体识别（NER）任务
@add_start_docstrings(
    """
    MegatronBert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    MEGATRON_BERT_START_DOCSTRING,
)
class MegatronBertForTokenClassification(MegatronBertPreTrainedModel):
    # 初始化函数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)
        # 获取标签数量
        self.num_labels = config.num_labels

        # 初始化 MegatronBert 模型，不添加池化层
        self.bert = MegatronBertModel(config, add_pooling_layer=False)
        # 添加 dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 添加分类器线性层
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并进行最终处理
        self.post_init()

    # 前向传播函数
    @add_start_docstrings_to_model_forward(MEGATRON_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 如果 return_dict 为 None，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 MegatronBert 模型的前向传播函数
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取序列输出
        sequence_output = outputs[0]

        # 对序列输出进行 dropout 处理
        sequence_output = self.dropout(sequence_output)
        # 经过分类器线性层，得到 logits
        logits = self.classifier(sequence_output)

        # 初始化损失为 None
        loss = None
        # 如果存在标签，则计算损失
        if labels is not None:
            # 使用交叉熵损失函数
            loss_fct = CrossEntropyLoss()
            # 计算损失
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不返回字典，则输出格式为元组
        if not return_dict:
            # 输出包括 logits 和可能的隐藏状态
            output = (logits,) + outputs[2:]
            # 如果损失不为 None，则添加损失到输出中
            return ((loss,) + output) if loss is not None else output

        # 如果返回字典，则输出格式为 TokenClassifierOutput
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 添加文档字符串描述 MegatronBert 模型和其用于抽取式问答任务（如 SQuAD）的分类头部
# 该模型在隐藏状态输出之上有线性层，用于计算 `span start logits` 和 `span end logits`
@add_start_docstrings( 
    """
    MegatronBert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a
    linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    MEGATRON_BERT_START_DOCSTRING,
)

# MegatronBert 问题回答模型类定义
class MegatronBertForQuestionAnswering(MegatronBertPreTrainedModel):
    # 初始化函数
    def __init__(self, config):
        # 调用父类初始化函数
        super().__init__(config)
        # 获取标签数量
        self.num_labels = config.num_labels

        # 创建 MegatronBert 模型
        self.bert = MegatronBertModel(config, add_pooling_layer=False)
        # 创建输出线性层
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 正向传播函数
    @add_start_docstrings_to_model_forward(MEGATRON_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 定义了一个 QuestionAnsweringModel 类的前向传播方法
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        # 如果 return_dict 参数为 None，则使用配置文件中的 use_return_dict 值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # 通过 BERT 模型获取序列输出
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
        # 获取序列输出
        sequence_output = outputs[0]
    
        # 计算问题起始和结束位置的 logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
    
        # 计算损失函数
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 如果 start_positions 和 end_positions 的大小大于 1，则移除最后一个维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 有时候 start/end positions 超出了模型输入的长度，我们忽略这些值
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
    
            # 使用交叉熵损失计算起始和结束位置的损失
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
    
        # 根据 return_dict 参数返回结果
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output
    
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```