# `.\transformers\models\rembert\modeling_rembert.py`

```
# 设置文件编码格式为 UTF-8
# 版权声明
#
# 根据 Apache License, Version 2.0（“许可证”）许可，本文件由2021年HuggingFace团队拥有版权。保留所有权利。
# 仅在遵守许可证的情况下才可以使用此文件。
# 您可以获取许可证的副本，并在以下地址查看许可内容：http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或经书面同意，否则根据该许可协议分发的软件基于“按原样”提供，无论是明示的还是暗示的，均不提供任何形式的保证或条件。
# 请查看许可协议以获取有关特定语言、响应权限和许可下限的规定
# 限制许可协议
# 下面是 PyTorch RemBERT 模型的实现

import math
import os
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
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
    logging,
    replace_return_docstrings,
)

# 导入 RemBert 的配置文件
from .configuration_rembert import RemBertConfig

# 获取 logger 实例
logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "RemBertConfig"
_CHECKPOINT_FOR_DOC = "google/rembert"

# RemBERT 预训练模型列表
REMBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/rembert",
    # 查看所有 RemBERT 模型：https://huggingface.co/models?filter=rembert
]


def load_tf_weights_in_rembert(model, config, tf_checkpoint_path):
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
    # 获取 TF 检查点路径的绝对路径
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # 从 TF 模型中加载权重
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    # 遍历初始变量的名称和形状
    for name, shape in init_vars:
        # 检查点大小为12GB，通过不加载无用的变量来节省内存
        # 输出嵌入和分类器在分类时会被重置
        if any(deny in name for deny in ("adam_v", "adam_m", "output_embedding", "cls")):
            # 如果名称中包含指定关键字，则跳过加载
            # logger.info("Skipping loading of %s", name)
            continue
        logger.info(f"Loading TF weight {name} with shape {shape}")
        # 加载 TensorFlow 变量
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        # 使用正确的前缀替换名称
        name = name.replace("bert/", "rembert/")
        # 对池化层进行线性变换
        # name = name.replace("pooler/dense", "pooler")

        name = name.split("/")
        # adam_v 和 adam_m 是 AdamWeightDecayOptimizer 中用于计算 m 和 v 的变量，对预训练模型不需要
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            # 如果名称中包含指定关键字，则跳过
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        pointer = model
        for m_name in name:
            # 将名称按下划线分割成列表
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
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
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            if pointer.shape != array.shape:
                raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)
    return model
class RemBertEmbeddings(nn.Module):
    """构建来自单词、位置和令牌类型嵌入的嵌入。"""

    def __init__(self, config):
        super().__init__()
        # 单词嵌入层，vocab_size 表示词汇表大小，input_embedding_size 表示嵌入维度，padding_idx 表示填充标记的索引
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.input_embedding_size, padding_idx=config.pad_token_id
        )
        # 位置嵌入层，max_position_embeddings 表示最大位置编码长度
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.input_embedding_size)
        # 令牌类型嵌入层，type_vocab_size 表示令牌类型数目
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.input_embedding_size)

        # self.LayerNorm 的命名不采用蛇形命名法以保持与 TensorFlow 模型变量名一致，并能够加载任何 TensorFlow 检查点文件
        # LayerNorm 层，input_embedding_size 表示输入嵌入维度，eps 表示批归一化的 epsilon 值
        self.LayerNorm = nn.LayerNorm(config.input_embedding_size, eps=config.layer_norm_eps)
        # Dropout 层，hidden_dropout_prob 表示隐藏层的 dropout 概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) 在序列化时是内存中连续的，并在被导出时保持不变
        # 注册位置编码张量，max_position_embeddings 表示最大位置编码长度
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # 如果未提供位置编码，则使用默认的位置编码
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # 如果未提供令牌类型编码，则使用全零的编码
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果未提供嵌入张量，则使用单词嵌入层生成嵌入
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # 获取令牌类型嵌入
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将单词嵌入和令牌类型嵌入相加得到初始嵌入
        embeddings = inputs_embeds + token_type_embeddings
        # 获取位置编码嵌入
        position_embeddings = self.position_embeddings(position_ids)
        # 将位置编码嵌入加到初始嵌入上
        embeddings += position_embeddings
        # LayerNorm 归一化
        embeddings = self.LayerNorm(embeddings)
        # Dropout
        embeddings = self.dropout(embeddings)
        return embeddings


# 从 transformers.models.bert.modeling_bert.BertPooler 复制并将 Bert->RemBert
class RemBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 全连接层，hidden_size 表示隐藏层大小
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 激活函数
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过取第一个令牌的隐藏状态来 "池化" 模型
        first_token_tensor = hidden_states[:, 0]
        # 使用全连接层处理第一个令牌的隐藏状态
        pooled_output = self.dense(first_token_tensor)
        # 使用激活函数
        pooled_output = self.activation(pooled_output)
        return pooled_output
# 定义了一个名为RemBertSelfAttention的类，继承自nn.Module类
class RemBertSelfAttention(nn.Module):
    # 初始化方法，接受一个config参数
    def __init__(self, config):
        super().__init__()
        # 如果config.hidden_size不能被config.num_attention_heads整除，并且config中没有embedding_size属性，则抛出ValueError异常
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        
        # 设置类的属性
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 创建三个线性层，用于生成查询、键和值
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 创建一个dropout层，用于dropout操作
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # 根据config.is_decoder属性，设置是否为解码器结构
        self.is_decoder = config.is_decoder
    
    # 定义一个方法，用于将输入张量转置为相应形状
    def transpose_for_scores(self, x):
        # 生成新的形状
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # 将输入张量视图重塑为新形状
        x = x.view(*new_x_shape)
        # 对新形状的维度进行置换
        return x.permute(0, 2, 1, 3)

    # 定义前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Tuple[Tuple[torch.FloatTensor]] = None,
        output_attentions: bool = False,
    ):
        # ...

# 定义了一个名为RemBertSelfOutput的类，继承自nn.Module类
class RemBertSelfOutput(nn.Module):
    # 初始化方法，接受一个config参数
    def __init__(self, config):
        super().__init__()
        # 创建一个线性层
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个LayerNorm层，用于层标准化操作
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个dropout层，用于dropout操作
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    # 定义前向传播方法
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用线性层对隐藏状态进行变换
        hidden_states = self.dense(hidden_states)
        # 对变换后的张量进行dropout操作
        hidden_states = self.dropout(hidden_states)
        # 对输入张量和变换后的张量进行残差连接，并进行层标准化操作
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回结果张量
        return hidden_states


# 定义了一个名为RemBertAttention的类，继承自nn.Module类
class RemBertAttention(nn.Module):
    # 初始化方法，接受一个config参数
    def __init__(self, config):
        super().__init__()
        # 创建一个RemBertSelfAttention对象
        self.self = RemBertSelfAttention(config)
        # 创建一个RemBertSelfOutput对象
        self.output = RemBertSelfOutput(config)
        # 创建一个记录被修剪头部的集合
        self.pruned_heads = set()

    # ...
    # 剪枝注意力头
    def prune_heads(self, heads):
        # 如果没有需要剪枝的注意力头，则直接返回
        if len(heads) == 0:
            return
        # 找到需要剪枝的注意力头及其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储已剪枝的注意力头
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 从 transformers.models.bert.modeling_bert.BertAttention.forward 复制而来
    # 前向传播函数
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
        # 调用 self 的前向传播函数
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 使用输出层处理自注意力层的输出
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出注意力权重，则将注意力权重加入输出
        outputs = (attention_output,) + self_outputs[1:]  # 如果输出注意力权重，则将其加入输出
        return outputs
# 从transformers.models.bert.modeling_bert.BertIntermediate复制并更改为RemBert
class RemBertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用线性层将输入特征映射到中间状态的尺寸
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果config.hidden_act是字符串，则选择对应的激活函数，否则直接使用config.hidden_act指定的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播过程，将输入的隐藏状态通过线性层和激活函数映射到中间状态
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

# 从transformers.models.bert.modeling_bert.BertOutput复制并更改为RemBert
class RemBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用线性层将中间状态映射回隐藏状态的尺寸
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 使用LayerNorm对隐藏状态进行归一化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 使用dropout对隐藏状态进行随机失活
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播过程，将中间状态映射回隐藏状态并进行LayerNorm和dropout处理
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class RemBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 设置chunk_size_feed_forward和seq_len_dim
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        # 创建RemBertAttention对象
        self.attention = RemBertAttention(config)
        # 设置is_decoder和add_cross_attention
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        # 如果add_cross_attention为True，则创建另一个RemBertAttention对象
        if self.add_cross_attention:
            if not self.is_decoder:
                # 如果不是解码器模型，则抛出错误
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = RemBertAttention(config)
        # 创建RemBertIntermediate对象
        self.intermediate = RemBertIntermediate(config)
        # 创建RemBertOutput对象
        self.output = RemBertOutput(config)

    # 从transformers.models.bert.modeling_bert.BertLayer.forward复制过来
    # 定义前向传播过程，包括attention_mask, head_mask, encoder_hidden_states等可选参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:  # 声明函数的返回类型为包含 torch.Tensor 的元组
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None  # 如果存在过去的键/值对，则使用前两个位置的值，否则为 None
        self_attention_outputs = self.attention(  # 使用 self.attention 方法计算自注意力
            hidden_states,  # 隐藏状态
            attention_mask,  # 注意力掩码
            head_mask,  # 头部掩码
            output_attentions=output_attentions,  # 是否输出注意力权重
            past_key_value=self_attn_past_key_value,  # 过去的键/值对
        )
        attention_output = self_attention_outputs[0]  # 获取自注意力输出中的第一个元素

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:  # 如果是解码器
            outputs = self_attention_outputs[1:-1]  # 输出是自注意力输出中除最后一个元素外的所有元素
            present_key_value = self_attention_outputs[-1]  # 当前的键/值对是自注意力输出中的最后一个元素
        else:
            outputs = self_attention_outputs[1:]  # 输出是自注意力输出中除第一个元素外的所有元素，如果需要输出注意力权重
                                              
        cross_attn_present_key_value = None  # 交叉注意力的当前键/值对初始化为 None
        if self.is_decoder and encoder_hidden_states is not None:  # 如果是解码器并且存在编码器的隐藏状态
            if not hasattr(self, "crossattention"):  # 如果不存在 self.crossattention 属性，则抛出值错误
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None  # 交叉注意力的过去的键/值对在过去键/值对元组的第三、第四位置
            cross_attention_outputs = self.crossattention(  # 使用 self.crossattention 方法计算交叉注意力
                attention_output,  # 注意输出
                attention_mask,  # 注意力掩码
                head_mask,  # 头部掩码
                encoder_hidden_states,  # 编码器隐藏状态
                encoder_attention_mask,  # 编码器注意力权重
                cross_attn_past_key_value,  # 交叉注意力的过去的键/值对
                output_attentions,  # 是否输出注���力权重
            )
            attention_output = cross_attention_outputs[0]  # 获取交叉注意力输出中的第一个元素
            outputs = outputs + cross_attention_outputs[1:-1]  # 输出是自注意力输出和交叉注意力输出中除最后一个元素外的所有元素，如果需要输出注意力权重

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]  # 将交叉注意力的当前的键/值对添加到当前键/值对元组的第三、第四位置
            present_key_value = present_key_value + cross_attn_present_key_value  # 当前的键/值对添加交叉注意力的当前的键/值对

        layer_output = apply_chunking_to_forward(  # 将 attention_output 应用分块处理并向前传播
            self.feed_forward_chunk,  # 使用 feed_forward_chunk 函数
            self.chunk_size_feed_forward,  # 分块大小
            self.seq_len_dim,  # 序列长度维度
            attention_output  # 注意输出
        )
        outputs = (layer_output,) + outputs  # 输出为应用分块处理后的结果加上之前的输出

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:  # 如果是解码器
            outputs = outputs + (present_key_value,)  # 输出为之前的输出加上当前的键/值对

        return outputs  # 返回最终的输出

    # Copied from transformers.models.bert.modeling_bert.BertLayer.feed_forward_chunk
    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)  # 中间输出为插值函数应用于注意输出
        layer_output = self.output(intermediate_output, attention_output)  # 层输出为输出函数应用于中间输出和注意输出
        return layer_output  # 返回层输出
# 定义一个自定义的 RemBert 编码器类，继承自 nn.Module
class RemBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 将输入嵌入大小映射到隐藏大小的线性层
        self.embedding_hidden_mapping_in = nn.Linear(config.input_embedding_size, config.hidden_size)
        # 创建一个由多个 RemBertLayer 层组成的列表，数量等于配置中指定的隐藏层数量
        self.layer = nn.ModuleList([RemBertLayer(config) for _ in range(config.num_hidden_layers)])
        # 是否启用梯度检查点
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
# 从 transformers.models.bert.modeling_bert.BertPredictionHeadTransform 复制并替换为 RemBert
class RemBertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 将隐藏大小映射到隐藏大小的线性层
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 如果配置中隐藏激活函数是字符串，选择相应的函数；否则直接使用配置中的函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # 使用 LayerNorm 对隐藏状态进行归一化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class RemBertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 将隐藏大小映射到输出嵌入大小的线性层
        self.dense = nn.Linear(config.hidden_size, config.output_embedding_size)
        # 将输出嵌入大小映射到词汇量大小的线性层
        self.decoder = nn.Linear(config.output_embedding_size, config.vocab_size)
        # 选择激活函数
        self.activation = ACT2FN[config.hidden_act]
        # 使用 LayerNorm 对输出嵌入进行归一化
        self.LayerNorm = nn.LayerNorm(config.output_embedding_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# 从 transformers.models.bert.modeling_bert.BertOnlyMLMHead 复制并替换为 RemBert
class RemBertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建预测层
        self.predictions = RemBertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        # 预测序列输出
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class RemBertPreTrainedModel(PreTrainedModel):
    """
    # 一个处理权重初始化和简单接口用于下载和加载预训练模型的抽象类。

    # 配置类为 RemBertConfig
    config_class = RemBertConfig
    # 加载 TF 权重的函数为 load_tf_weights_in_rembert
    load_tf_weights = load_tf_weights_in_rembert
    # 基础模型前缀为 "rembert"
    base_model_prefix = "rembert"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    # 初始化权重的函数
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果是线性层
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重，均值为0，标准差为模型配置中的initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置项，将其初始化为0
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是嵌入层
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重，均值为0，标准差为模型配置中的initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有填充索引，将填充索引处的权重初始化为0
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果是 LayerNorm 层
        elif isinstance(module, nn.LayerNorm):
            # 将偏置项初始化为0
            module.bias.data.zero_()
            # 将权重初始化为1
            module.weight.data.fill_(1.0)
# REMBERT_START_DOCSTRING 是一个字符串字面量，包含了 Rembert 模型的文档字符串。
# 这个字符串描述了 Rembert 模型是一个 PyTorch 的 nn.Module 子类，可以像普通的 PyTorch 模块一样使用，
# 并且参考 PyTorch 文档来了解相关的用法和行为。
# 该字符串还描述了初始化 Rembert 模型时需要传入的配置类 RemBertConfig，其中包含了模型的所有参数。
# 需要注意的是，初始化只加载了配置信息，并不会加载模型的预训练权重。
# 需要使用 PreTrainedModel.from_pretrained 方法来加载预训练的模型权重。
REMBERT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`RemBertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# REMBERT_INPUTS_DOCSTRING 是另一个字符串字面量，包含了 Rembert 模型输入的文档字符串。
# 这个字符串可能包含了模型输入的相关信息，但在给定的代码中并没有显示出来。
REMBERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列标记在词汇表中的索引。
            # 可以使用 [`AutoTokenizer`] 获取这些索引。详见 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`]。
            # [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 避免在填充令牌索引上执行注意力操作的掩码。
            # 掩码值选择在 `[0, 1]`：
            # - 1 表示**未掩码**的标记，
            # - 0 表示**已掩码**的标记。
            # [什么是注意力掩码？](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 段标记索引，指示输入的第一部分和第二部分。索引选择在 `[0, 1]`：
            # - 0 对应于*句子 A*的标记，
            # - 1 对应于*句子 B*的标记。
            # [什么是标记类型 ID？](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 每个输入序列标记在位置嵌入中的位置索引。选择范围在 `[0, config.max_position_embeddings - 1]`。
            # [什么是位置 ID？](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于将自注意力模块的特定头部置零的掩码。掩码值选择在 `[0, 1]`：
            # - 1 表示头部**未掩码**，
            # - 0 表示头部**已掩码**。
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选地，您可以选择直接传递嵌入表示，而不是传递 `input_ids`。如果您想要更多控制，将 *input_ids* 索引转换为相关向量，这很有用。
            # 这对于比模型的内部嵌入查找矩阵更好地控制如何转换 *input_ids* 索引为相关向量时很有用。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回的张量下的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多详细信息，请参见返回的张量下的 `hidden_states`。
        return_dict (`bool`, *optional*):
            # 是否返回一个 [`~utils.ModelOutput`] 而不是一个普通元组。
```  
"""
# 导入所需库
import torch
from transformers.modeling_utils import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.rembert.configuration_rembert import RemBertConfig
from transformers.models.rembert.modeling_rembert_utils import (
    RemBertPreTrainedModel,
    RemBertEmbeddings,
    RemBertEncoder,
    RemBertPooler,
    REMBERT_START_DOCSTRING,
    REMBERT_INPUTS_DOCSTRING,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    add_code_sample_docstrings,
)

@add_start_docstrings(
    "The bare RemBERT Model transformer outputting raw hidden-states without any specific head on top.",
    REMBERT_START_DOCSTRING,
)
# 定义 RemBERT 模型类
class RemBertModel(RemBertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """
    
    # 初始化函数
    def __init__(self, config, add_pooling_layer=True):
        # 调用父类初始化函数
        super().__init__(config)
        # 保存配置信息
        self.config = config
        # 初始化词嵌入层
        self.embeddings = RemBertEmbeddings(config)
        # 初始化编码器
        self.encoder = RemBertEncoder(config)
        # 如果需要添加池化层，则初始化池化层
        self.pooler = RemBertPooler(config) if add_pooling_layer else None
        # 初始化模型参数权重
        self.post_init()

    # 获取输入词嵌入层
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入词嵌入层
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # 剪枝模型的注意力头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 前向传播函数
    @add_start_docstrings_to_model_forward(REMBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="google/rembert",
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
"""
# 使用自定义的文档字符串修饰器为模型添加描述
@add_start_docstrings("""RemBERT Model with a `language modeling` head on top.""", REMBERT_START_DOCSTRING)
# 声明一个继承自 RemBertPreTrainedModel 的类 RemBertForMaskedLM
class RemBertForMaskedLM(RemBertPreTrainedModel):
    # 定义一个类变量，指定权重共享的键名
    _tied_weights_keys = ["cls.predictions.decoder.weight"]

    # 构造函数，初始化模型
    def __init__(self, config):
        # 调用父类的构造函数
        super().__init__(config)

        # 如果配置要求是解码器，则发出警告
        if config.is_decoder:
            logger.warning(
                "If you want to use `RemBertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 初始化 RemBERT 模型，不包含池化层
        self.rembert = RemBertModel(config, add_pooling_layer=False)
        # 初始化仅包含 MLM 头部的模型
        self.cls = RemBertOnlyMLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出的嵌入层
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出的嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 重写 forward 方法，实现前向传播
    @add_start_docstrings_to_model_forward(REMBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="google/rembert",
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
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
        # 输入参数说明文档字符串
    ):
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional`):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        # 确保返回值为字典类型
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用rembert模型，传入输入参数，获取输出
        outputs = self.rembert(
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

        # 从输出中获取序列输出和预测分数
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        # 如果有labels，则计算masked lm loss
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果return_dict=False，则返回输出
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 返回MaskedLMOutput对象，包括loss、logits、hidden_states和attentions
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  添加一个虚拟标记
        assert self.config.pad_token_id is not None, "The PAD token should be defined for generation"
        # 调整注意力掩码的形状，增加一列全零数据
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        # 创建一个全为pad token id的张量，并将其与input_ids进行拼接
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        # 返回包含input_ids和attention_mask的字典
        return {"input_ids": input_ids, "attention_mask": attention_mask}
# 添加文档注释到类，用于描述该类的作用和继承自的文档字符串
@add_start_docstrings(
    """RemBERT Model with a `language modeling` head on top for CLM fine-tuning.""", REMBERT_START_DOCSTRING
)

# 定义 RemBertForCausalLM 类，继承自 RemBertPreTrainedModel 类
class RemBertForCausalLM(RemBertPreTrainedModel):
    # 指定共享权重的键值对列表
    _tied_weights_keys = ["cls.predictions.decoder.weight"]

    # 初始化函数，接受一个配置参数对象
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)

        # 如果配置对象不是解码器，发出警告
        if not config.is_decoder:
            logger.warning("If you want to use `RemBertForCausalLM` as a standalone, add `is_decoder=True.`")

        # 创建 RemBertModel 对象，并设置不添加池化层
        self.rembert = RemBertModel(config, add_pooling_layer=False)
        # 创建 RemBertOnlyMLMHead 对象
        self.cls = RemBertOnlyMLMHead(config)

        # 初始化权重并执行最终处理
        self.post_init()

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 前向传播函数，接受多个输入参数
    @add_start_docstrings_to_model_forward(REMBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        # 获取输入 IDs 的形状
        input_shape = input_ids.shape

        # 如果注意力遮罩为空，创建一个全为1的遮罩
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # 如果存在���去的键值对，根据情况截取输入 IDs
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法可能只传入最后一个输入 ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认行为：保留最后一个 ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        # 返回包含输入 IDs、注意力遮罩、过去的键值对的字典
        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}
    # 重新排列缓存，以适应beam搜索
    def _reorder_cache(self, past_key_values, beam_idx):
        # 初始化重新排列后的缓存
        reordered_past = ()
        # 遍历每一层的过去键值
        for layer_past in past_key_values:
            # 重新排列过去的状态，并添加到重新排列后的缓存中
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        # 返回重新排列后的缓存
        return reordered_past
# 创建一个新的类 RemBertForSequenceClassification，继承自 RemBertPreTrainedModel
@add_start_docstrings("""
    RemBERT Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """, 
    REMBERT_START_DOCSTRING,
)
class RemBertForSequenceClassification(RemBertPreTrainedModel):
    # 初始化方法，接受一个配置文件参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置类别数量
        self.num_labels = config.num_labels
        # 创建 RemBertModel 实例
        self.rembert = RemBertModel(config)
        # 创建一个丢弃层实例
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        # 创建一个线性层实例
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法
    @add_start_docstrings_to_model_forward(REMBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="google/rembert",
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        定义模型前向传播函数，接受输入参数设置并返回计算得到的输出结果

        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            用于计算序列分类/回归损失的标签。索引应该在 `[0, ..., config.num_labels - 1]` 范围内。
            如果 `config.num_labels == 1`，则计算回归损失（均方误差损失）；如果 `config.num_labels > 1`，则计算分类损失（交叉熵损失）。
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 利用rembert模型进行前向传播
        outputs = self.rembert(
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

        # 对池化后的输出进行dropout操作
        pooled_output = self.dropout(pooled_output)
        # 将结果输入分类器获取logits
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                # 定义均方误差损失函数，并计算损失
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # 定义交叉熵损失函数，并计算损失
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # 定义带logits的二元交叉熵损失函数，并计算损失
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        
        # 如果不需要返回字典，则返回模型输出信息
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回分类器输出结果对象包括损失、logits、隐藏状态和注意力权重
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用 RemBERT 模型进行多项选择分类任务，顶部加了一个分类头（线性层和 softmax），例如 RocStories/SWAG 任务。
@add_start_docstrings(
    """
    RemBERT Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    REMBERT_START_DOCSTRING,
)
class RemBertForMultipleChoice(RemBertPreTrainedModel):
    # 初始化函数，接受一个配置参数对象
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)

        # 创建一个 RemBERT 模型实例
        self.rembert = RemBertModel(config)
        # 创建一个 Dropout 层实例
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        # 创建一个线性层实例
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数，接受多个输入参数
    @add_start_docstrings_to_model_forward(REMBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="google/rembert",
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 设置是否返回字典形式的输出结果，默认为使用配置文件中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 获取输入张量的第二维大小，即选项数量
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 重塑输入张量，将其转换为二维张量，第一维度为所有样本，第二维度为每个样本的所有输入
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 调用RemBERT模型，传递输入的各种参数，获取输出结果
        outputs = self.rembert(
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

        # 从输出结果中提取池化后的输出
        pooled_output = outputs[1]

        # 对池化输出进行dropout处理
        pooled_output = self.dropout(pooled_output)
        # 使用分类器对dropout后的输出进行分类，得到logits
        logits = self.classifier(pooled_output)
        # 重塑logits张量，将其转换为二维张量，第一维度为所有样本，第二维度为每个样本的所有选项的logits
        reshaped
# 在 RemBERT 模型基础上添加一个标记分类头部的模型，例如用于命名实体识别（NER）任务
class RemBertForTokenClassification(RemBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # 使用 RemBERT 模型作为基础模型，不添加池化层
        self.rembert = RemBertModel(config, add_pooling_layer=False)
        # 添加丢弃层
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        # 添加线性分类器
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 定义前向传播方法
    def forward(
        self,
        input_ids: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.FloatTensor] = None,
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

        # 设置返回字典的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 经过 RemBERT 模型并获取输出
        outputs = self.rembert(
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

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# 添加文档字符串
@add_start_docstrings(
    """
    # 定义一个RemBERT模型，其顶部有一个用于提取问题回答任务（如SQuAD）的跨度分类头（在隐藏状态输出之上有线性层来计算“跨度开始对数”和“跨度结束对数”）。
    # """
    # 定义了一个常量REMENBERT_START_DOCSTRING，可能是用于表示文档开头的字符串
# 定义一个新的类 RemBertForQuestionAnswering，它继承自 RemBertPreTrainedModel 类
class RemBertForQuestionAnswering(RemBertPreTrainedModel):
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)

        # 在配置中获取标签数量
        self.num_labels = config.num_labels

        # 创建一个 RemBertModel 对象，关闭添加池化层的选项
        self.rembert = RemBertModel(config, add_pooling_layer=False)
        # 创建一个线性层，将隐藏层输出的特征映射成标签数量的输出
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并执行最终处理
        self.post_init()

    # 前向传播函数，处理模型输入并返回输出
    @add_start_docstrings_to_model_forward(REMBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="google/rembert",
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 该函数是一个 PyTorch 模型的前向传播过程
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
        # 该段定义了输入和输出参数的类型和含义
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        # 如果 return_dict 为 None，则使用模型的默认配置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # 调用 self.rembert 函数，并将输入参数传递进去
        outputs = self.rembert(
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
    
        # 获取模型输出的序列输出
        sequence_output = outputs[0]
    
        # 将序列输出传递给 self.qa_outputs 层，获得 start_logits 和 end_logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
    
        total_loss = None
        # 如果提供了 start_positions 和 end_positions，则计算损失函数
        if start_positions is not None and end_positions is not None:
            # 如果 start_positions 和 end_positions 的维度大于 1，则去掉最后一个维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 将 start_positions 和 end_positions 限制在模型输入长度内
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
    
            # 计算 start_loss 和 end_loss，并取平均作为总损失
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
    
        # 如果 return_dict 为 False，则返回一个元组
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output
    
        # 否则返回一个 QuestionAnsweringModelOutput 对象
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```