# `.\models\electra\modeling_electra.py`

```py
# 指定编码为 UTF-8
# 版权声明及许可证信息
# 版权所有 2019 年谷歌 AI 语言团队作者和 HuggingFace 公司团队
# 根据 Apache 许可证 2.0 版本（“许可证”）许可；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按“原样”基础分发软件，
# 没有任何明示或暗示的保证或条件。
# 请参阅许可证了解特定语言的权限。
"""PyTorch ELECTRA 模型。"""

# 导入必要的库和模块
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入激活函数相关的模块
from ...activations import ACT2FN, get_activation
# 导入模型输出相关的模块
from ...modeling_outputs import (
    BaseModelOutputWithCrossAttentions,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
# 导入预训练模型相关的工具函数
from ...modeling_utils import PreTrainedModel, SequenceSummary
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
# 导入工具函数和类
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的模型检查点
_CHECKPOINT_FOR_DOC = "google/electra-small-discriminator"
# 用于文档的配置文件
_CONFIG_FOR_DOC = "ElectraConfig"

# ELECTRA 的预训练模型档案列表
ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/electra-small-generator",
    "google/electra-base-generator",
    "google/electra-large-generator",
    "google/electra-small-discriminator",
    "google/electra-base-discriminator",
    "google/electra-large-discriminator",
    # 查看所有 ELECTRA 模型 https://huggingface.co/models?filter=electra
]

# 加载 ELECTRA 模型中的 TensorFlow 权重
def load_tf_weights_in_electra(model, config, tf_checkpoint_path, discriminator_or_generator="discriminator"):
    """Load tf checkpoints in a pytorch model."""
    try:
        # 导入必要的库和模块
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        # 如果导入失败，打印错误信息并抛出 ImportError
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    # 获取 TensorFlow 权重路径的绝对路径
    tf_path = os.path.abspath(tf_checkpoint_path)
    # 打印转换 TensorFlow 检查点的消息
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # 从 TF 模型加载权重
    init_vars = tf.train.list_variables(tf_path)
    # 初始化变量名列表和权重数组列表
    names = []
    arrays = []
    # 遍历初始化变量的元组，其中包含变量名和形状信息
    for name, shape in init_vars:
        # 记录日志，显示正在加载的 TensorFlow 权重的名称和形状
        logger.info(f"Loading TF weight {name} with shape {shape}")
        # 从 TensorFlow 路径加载变量的数组
        array = tf.train.load_variable(tf_path, name)
        # 将变量名添加到列表中
        names.append(name)
        # 将数组添加到列表中
        arrays.append(array)
    # 遍历变量名列表和数组列表
    for name, array in zip(names, arrays):
        # 将原始名称保存在变量中
        original_name: str = name

        try:
            # 如果模型是 ElectraForMaskedLM 类型，则替换变量名中的一部分
            if isinstance(model, ElectraForMaskedLM):
                name = name.replace("electra/embeddings/", "generator/embeddings/")

            # 如果是生成器，则替换变量名中的部分字符串
            if discriminator_or_generator == "generator":
                name = name.replace("electra/", "discriminator/")
                name = name.replace("generator/", "electra/")

            # 替换变量名中的特定字符串
            name = name.replace("dense_1", "dense_prediction")
            name = name.replace("generator_predictions/output_bias", "generator_lm_head/bias")

            # 将变量名按"/"分割成列表
            name = name.split("/")
            # 打印原始名称和变量名列表（用于调试，实际运行时注释掉）
            # print(original_name, name)
            # 如果变量名中包含特定字符串，则跳过该变量的加载
            if any(n in ["global_step", "temperature"] for n in name):
                # 记录日志，显示跳过加载的变量名
                logger.info(f"Skipping {original_name}")
                # 跳过此次循环，继续下一个变量的加载
                continue
            # 初始化指针为模型
            pointer = model
            # 遍历变量名列表
            for m_name in name:
                # 如果变量名匹配特定模式，则分割变量名
                if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                    scope_names = re.split(r"_(\d+)", m_name)
                else:
                    scope_names = [m_name]
                # 根据变量名的首部选择操作
                if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                    pointer = getattr(pointer, "weight")
                elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                    pointer = getattr(pointer, "bias")
                elif scope_names[0] == "output_weights":
                    pointer = getattr(pointer, "weight")
                elif scope_names[0] == "squad":
                    pointer = getattr(pointer, "classifier")
                else:
                    pointer = getattr(pointer, scope_names[0])
                # 如果变量名含有数字，则根据数字选择操作
                if len(scope_names) >= 2:
                    num = int(scope_names[1])
                    pointer = pointer[num]
            # 如果变量名以"_embeddings"结尾，则选择权重操作
            if m_name.endswith("_embeddings"):
                pointer = getattr(pointer, "weight")
            # 如果变量名为"kernel"，则转置数组
            elif m_name == "kernel":
                array = np.transpose(array)
            try:
                # 检查指针和数组的形状是否匹配
                if pointer.shape != array.shape:
                    raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
            except ValueError as e:
                # 如果形状不匹配，则抛出异常
                e.args += (pointer.shape, array.shape)
                raise
            # 打印 PyTorch 权重初始化信息
            print(f"Initialize PyTorch weight {name}", original_name)
            # 将数组转换为 PyTorch 张量，并赋值给指针
            pointer.data = torch.from_numpy(array)
        except AttributeError as e:
            # 如果出现属性错误，则记录日志，并跳过此次循环
            print(f"Skipping {original_name}", name, e)
            continue
    # 返回加载完成的模型
    return model
class ElectraEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    # 定义一个类用于构建词、位置和令牌类型嵌入层

    def __init__(self, config):
        super().__init__()
        # 初始化函数，设置各个嵌入层和其他属性
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        # 创建词嵌入层
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        # 创建位置嵌入层
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)
        # 创建令牌类型嵌入层

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        # 创建 LayerNorm 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 创建 Dropout 层

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 创建 position_ids 张量，表示位置索引
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 设置位置嵌入类型，默认为绝对位置编码
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )
        # 创建 token_type_ids 张量，表示令牌类型索引

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.forward
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
        # 前向传播函数，接收输入等参数进行计算
    # 根据输入的参数，返回经过嵌入层的张量
    
        ) -> torch.Tensor:
        # 如果输入的input_ids不为空，则获取其形状赋值给input_shape
            if input_ids is not None:
                input_shape = input_ids.size()
            # 否则，获取inputs_embeds的形状
            else:
                input_shape = inputs_embeds.size()[:-1]
    
            # 获取序列的长度
            seq_length = input_shape[1]
    
            # 若未提供position_ids，则从构造函数中获取，默认返回seq_length相应的位置的元素
            if position_ids is None:
                position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
    
            # 若未提供token_type_ids，则根据对象的token_type_ids属性获取，若属性不存在，创建一个与input_shape相同的零张量
            if token_type_ids is None:
                if hasattr(self, "token_type_ids"):
                    buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                    buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                    token_type_ids = buffered_token_type_ids_expanded
                else:
                    token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
    
            # 若未提供inputs_embeds，则根据input_ids通过word_embeddings获取
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            # 通过token_type_ids获取token_type_embeddings
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
    
            # 将输入的嵌入和token type编码相加
            embeddings = inputs_embeds + token_type_embeddings
            # 若使用绝对位置编码，则通过position_ids获取position_embeddings并添加到embeddings中
            if self.position_embedding_type == "absolute":
                position_embeddings = self.position_embeddings(position_ids)
                embeddings += position_embeddings
            # 经过LayerNorm层
            embeddings = self.LayerNorm(embeddings)
            # 经过dropout层
            embeddings = self.dropout(embeddings)
            # 返回embeddings张量作为输出结果
            return embeddings
# 自注意力机制模块，从transformers.models.bert.modeling_bert.BertSelfAttention复制过来，并将Bert改为Electra
class ElectraSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 确保hidden_size是attention_heads的倍数，否则抛出异常
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果position_embedding_type是'relative_key'或'relative_key_query'，需要额外的位置嵌入
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    # 调整张量维度，变为[batch_size, num_attention_heads, seq_length, attention_head_size]的形状
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        # 从transformers.models.bert.modeling_bert.BertSelfOutput复制过来
class ElectraSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# 从transformers.models.bert.modeling_bert.BertAttention复制过来，并将Bert改为Electra
# ElectraAttention 类，用于 Electra 模型的自注意力机制
class ElectraAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 初始化 ElectraSelfAttention 层
        self.self = ElectraSelfAttention(config, position_embedding_type=position_embedding_type)
        # 初始化 ElectraSelfOutput 层
        self.output = ElectraSelfOutput(config)
        # 初始化头部修剪集合
        self.pruned_heads = set()

    # 头部修剪方法
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 查找可修剪头部及其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 修剪线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储修剪头部
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 前向传播方法
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
        # 调用自注意力层的前向传播方法
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 使用自注意力输出和原始输入计算注意力输出
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出注意力，则将其添加到输出中
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# ElectraIntermediate 类，用于 Electra 模型的中间层
# 从 transformers.models.bert.modeling_bert.BertIntermediate 复制而来
class ElectraIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 线性层初始化
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 激活函数初始化
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播方法
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 线性变换
        hidden_states = self.dense(hidden_states)
        # 激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# ElectraOutput 类，用于 Electra 模型的输出层
# 从 transformers.models.bert.modeling_bert.BertOutput 复制而来
class ElectraOutput(nn.Module):
```  
    # 初始化函数，用于初始化网络模型的参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个全连接层，将输入特征的维度调整为隐藏层的维度
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个 LayerNorm 层，用于对隐藏状态进行归一化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个 Dropout 层，用于在训练过程中随机丢弃部分神经元，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    # 前向传播函数，用于定义网络模型的前向计算过程
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 输入隐藏状态经过全连接层，得到新的隐藏状态
        hidden_states = self.dense(hidden_states)
        # 对新的隐藏状态进行随机丢弃部分神经元的操作
        hidden_states = self.dropout(hidden_states)
        # 将丢弃部分神经元后的隐藏状态与输入张量相加，并进行 LayerNorm 归一化处理
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的隐藏状态
        return hidden_states
# 从 transformers.models.bert.modeling_bert.BertLayer 复制而来，将其中的 Bert 替换为 Electra
class ElectraLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 设置前向传播中的分块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度的维度索引
        self.seq_len_dim = 1
        # Electra 自注意力机制
        self.attention = ElectraAttention(config)
        # 是否为解码器
        self.is_decoder = config.is_decoder
        # 是否添加交叉注意力
        self.add_cross_attention = config.add_cross_attention
        # 如果添加了交叉注意力
        if self.add_cross_attention:
            # 如果不是解码器则引发值错误
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 使用绝对位置嵌入创建交叉注意力
            self.crossattention = ElectraAttention(config, position_embedding_type="absolute")
        # Electra 中间层
        self.intermediate = ElectraIntermediate(config)
        # Electra 输出层
        self.output = ElectraOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    # 定义函数，接收一个self参数和past_key_value参数，返回一个包含torch.Tensor的元组
    ) -> Tuple[torch.Tensor]:
        # 如果存在past_key_value，则取其中前两个元素作为decoder uni-directional self-attention的缓存key/values
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 使用self.attention函数进行self-attention操作，并获取输出
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # 获取self-attention的输出
        attention_output = self_attention_outputs[0]

        # 如果是decoder，则最后一个输出是self-attention的缓存key/values的元组
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # 如果输出注意力权重，则添加self-attentions

        cross_attn_present_key_value = None
        # 如果是decoder且存在encoder_hidden_states
        if self.is_decoder and encoder_hidden_states is not None:
            # 如果没有crossattention层，则引发异常
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 如果存在past_key_value，则取其中后两个元素作为cross-attention的缓存key/values
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 使用self.crossattention函数进行cross-attention操作，并获取输出
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            # 获取cross-attention的输出和缓存key/values
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # 如果输出注意力权重，则添加cross-attentions

            # 将cross-attn缓存添加到现有的缓存中
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 将attention_output传入apply_chunking_to_forward函数，按照设定的chunk_size和seq_len_dim进行前向传播
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # 如果是decoder，则将attn key/values作为最后一个输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    # 定义feed_forward_chunk函数，接收attention_output作为参数，返回层输出
    def feed_forward_chunk(self, attention_output):
        # 通过intermediate层和output层得到最终的层输出
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
# 将Bert模型的Encoder类复制并修改为Electra模型的Encoder类
class ElectraEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 创建包含ElectraLayer对象的列表，数量为配置中指定的隐藏层数量
        self.layer = nn.ModuleList([ElectraLayer(config) for _ in range(config.num_hidden_layers)])
        # 是否使用梯度检查点
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
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        # 如果设置了输出隐藏状态则初始化空元组，否则设为空
        all_hidden_states = () if output_hidden_states else None
        # 如果设置了输出注意力则初始化空元组，否则设为空
        all_self_attentions = () if output_attentions else None
        # 如果同时设置了输出注意力和配置了交叉注意力，则初始化为空元组，否则设为空
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 如果开启了梯度检查点并且处于训练状态
        if self.gradient_checkpointing and self.training:
            # 如果在使用缓存，则发出警告并将use_cache设置为False
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # 如果use_cache为True则初始化空元组，否则设置为None
        next_decoder_cache = () if use_cache else None
        # 遍历编码器
        for i, layer_module in enumerate(self.layer):
            # 如果设置了输出隐藏状态，则将当前隐藏状态添加到all_hidden_states中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果head_mask不为空，则取出对应位置的head_mask
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 如果past_key_values不为空，则取出对应位置的past_key_value
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 如果开启了梯度检查点并且处于训练状态
            if self.gradient_checkpointing and self.training:
                # 使用梯度检查点函数对当前层进行计算
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                # 否则直接对当前层进行计算
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            # 更新隐藏状态值为当前层输出的第一个值
            hidden_states = layer_outputs[0]
            # 如果使用缓存，则将当前层的结果加入到下一个解码器缓存中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 如果设置了输出注意力，则将当前层输出的注意力加入到all_self_attentions中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果配置了交叉注意力，则将当前层输出的交叉注意力加入到all_cross_attentions中
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果设置了输出隐藏状态，则将当前隐藏状态添加到all_hidden_states中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典格式的结果，则将当前隐藏状态、下一个解码器缓存、全部隐藏状态、全部自注意力、全部交叉注意力进行返回
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        # 返回BaseModelOutputWithPastAndCrossAttentions类型的结果
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
# 定义一个名为 ElectraDiscriminatorPredictions 的类，用于判别器的预测，包含两个密集层
class ElectraDiscriminatorPredictions(nn.Module):
    """Prediction module for the discriminator, made up of two dense layers."""

    def __init__(self, config):
        super().__init__()
        # 定义一个线性层，输入维度为 config.hidden_size，输出维度为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 获取激活函数，激活函数类型由 config.hidden_act 指定
        self.activation = get_activation(config.hidden_act)
        # 定义一个线性层，输入维度为 config.hidden_size，输出维度为 1
        self.dense_prediction = nn.Linear(config.hidden_size, 1)
        # 保存配置信息
        self.config = config

    def forward(self, discriminator_hidden_states):
        # 输入判别器隐藏状态到密集层
        hidden_states = self.dense(discriminator_hidden_states)
        # 对密集层输出进行激活
        hidden_states = self.activation(hidden_states)
        # 经过最后一层密集层输出一个值，使用squeeze(-1)降维
        logits = self.dense_prediction(hidden_states).squeeze(-1)

        return logits


# 定义一个名为 ElectraGeneratorPredictions 的类，用于生成器的预测，包含两个密集层
class ElectraGeneratorPredictions(nn.Module):
    """Prediction module for the generator, made up of two dense layers."""

    def __init__(self, config):
        super().__init__()
        # 获取激活函数，这里指定为 GELU
        self.activation = get_activation("gelu")
        # LayerNorm 层，输入维度为 config.embedding_size，eps 参数指定 layer norm 的 epsilon
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        # 定义一个线性层，输入维度为 config.hidden_size，输出维度为 config.embedding_size
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)

    def forward(self, generator_hidden_states):
        # 输入生成器隐藏状态到密集层
        hidden_states = self.dense(generator_hidden_states)
        # 对密集层输出进行激活
        hidden_states = self.activation(hidden_states)
        # 对激活后的输出进行 LayerNorm
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states


# 定义一个名为 ElectraPreTrainedModel 的类，用于处理权重初始化和预训练模型的下载与加载
class ElectraPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 使用 ElectraConfig 作为配置类
    config_class = ElectraConfig
    # 使用 load_tf_weights_in_electra 函数加载 TensorFlow 权重
    load_tf_weights = load_tf_weights_in_electra
    # 指定模型的 base_model_prefix 为 "electra"
    base_model_prefix = "electra"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    # 来自 transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights 的代码
    # 初始化权重的函数
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # 对线性层的权重进行正态分布初始化
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                # 如果有偏置项，则将其初始化为零
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 对嵌入层的权重进行正态分布初始化
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                # 如果有 padding 索引，则将对应的权重初始化为零
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 对 LayerNorm 层的偏置项初始化为零，权重初始化为1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


@dataclass
# 定义一个名为 ElectraForPreTrainingOutput 的类，用于保存预训练输出结果
class ElectraForPreTrainingOutput(ModelOutput):
    """
    Output type of [`ElectraForPreTraining`].
    """
    # 参数说明：
    # 当提供 `labels` 参数时，返回该参数，类型为 `torch.FloatTensor`，形状为 `(1,)`
    loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
        Total loss of the ELECTRA objective.

    # 预测头部的预测分数（SoftMax 之前的每个标记的分数），类型为 `torch.FloatTensor`，形状为 `(batch_size, sequence_length)`
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
        Prediction scores of the head (scores for each token before SoftMax).

    # 当传递 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时，返回该参数，类型为 `tuple(torch.FloatTensor)`
    # 其中包含 `torch.FloatTensor`（一个用于嵌入的输出 + 一个用于每个层的输出）的元组，形状为 `(batch_size, sequence_length, hidden_size)`
    hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
        Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
        shape `(batch_size, sequence_length, hidden_size)`.
        Hidden-states of the model at the output of each layer plus the initial embedding outputs.

    # 当传递 `output_attentions=True` 或 `config.output_attentions=True` 时，返回该参数，类型为 `tuple(torch.FloatTensor)`
    # 包含每个层的 `torch.FloatTensor`，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`
    attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
        Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.
        Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
        heads.

    # 可选的损失参数，类型为 `Optional[torch.FloatTensor]`
    loss: Optional[torch.FloatTensor] = None
    # 预测分数参数，类型为 `torch.FloatTensor`
    logits: torch.FloatTensor = None
    # 可选的隐藏状态参数，类型为 `Optional[Tuple[torch.FloatTensor]]`
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 可选的注意力参数，类型为 `Optional[Tuple[torch.FloatTensor]]`
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# 这是 ELECTRA 模型的初始文档字符串，用于描述模型的功能和用法
ELECTRA_START_DOCSTRING = r"""
    # 继承自 PreTrainedModel，并提供通用方法，参阅库的超类文档
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    # 也是 PyTorch 的 torch.nn.Module 子类，适用于一般的 PyTorch 模块使用
    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    # 解释模型参数的构造器，强调配置文件不包括模型权重
    Parameters:
        config ([`ElectraConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 这是 ELECTRA 模型的输入文档字符串，通常用来描述输入的格式和要求
ELECTRA_INPUTS_DOCSTRING = r"""
"""

# 使用 `add_start_docstrings` 装饰器为类添加文档
@add_start_docstrings(
    # 类的描述，指出 ElectraModel 输出原始的隐藏状态，没有额外的输出头
    "The bare Electra Model transformer outputting raw hidden-states without any specific head on top. Identical to "
    "the BERT model except that it uses an additional linear layer between the embedding layer and the encoder if the "
    "hidden size and embedding size are different. "
    # 说明可加载生成器和判别器的检查点
    "Both the generator and discriminator checkpoints may be loaded into this model.",
    # 添加 ElectraModel 的通用描述
    ELECTRA_START_DOCSTRING,
)
# ElectraModel 类定义，继承自 ElectraPreTrainedModel
class ElectraModel(ElectraPreTrainedModel):
    # 构造器方法，用于初始化类的属性
    def __init__(self, config):
        # 调用超类的构造器
        super().__init__(config)
        # 创建 Electra 模型的嵌入层
        self.embeddings = ElectraEmbeddings(config)

        # 如果嵌入尺寸和隐藏层尺寸不同，则使用线性层将嵌入层和编码器连接起来
        if config.embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(config.embedding_size, config.hidden_size)

        # 创建 Electra 编码器
        self.encoder = ElectraEncoder(config)
        # 存储配置
        self.config = config
        # 初始化权重，并应用最终的处理
        self.post_init()

    # 返回输入嵌入层
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入嵌入层
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # 对模型的注意力头进行剪枝
    def _prune_heads(self, heads_to_prune):
        # 头部剪枝，接收一个字典，其中键是层的编号，值是该层需要剪枝的头部列表
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 遍历每一层和对应的头部列表，调用编码器的注意力剪枝方法
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 为模型的 forward 方法添加文档和代码示例的装饰器
    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        # 为代码示例提供检查点
        checkpoint=_CHECKPOINT_FOR_DOC,
        # 输出类型为 BaseModelOutputWithCrossAttentions
        output_type=BaseModelOutputWithCrossAttentions,
        # 配置类为 _CONFIG_FOR_DOC
        config_class=_CONFIG_FOR_DOC,
    )
    # 正向传播函数，用于模型的前向推断
    def forward(
        self,
        # 可选参数，输入的 token ID tensor
        input_ids: Optional[torch.Tensor] = None,
        # 可选参数，注意力 mask tensor
        attention_mask: Optional[torch.Tensor] = None,
        # 可选参数，token 类型 ID tensor
        token_type_ids: Optional[torch.Tensor] = None,
        # 可选参数，位置 ID tensor
        position_ids: Optional[torch.Tensor] = None,
        # 可选参数，头部 mask tensor
        head_mask: Optional[torch.Tensor] = None,
        # 可选参数，输入嵌入 tensor
        inputs_embeds: Optional[torch.Tensor] = None,
        # 可选参数，编码器隐藏状态 tensor
        encoder_hidden_states: Optional[torch.Tensor] = None,
        # 可选参数，编码器注意力 mask tensor
        encoder_attention_mask: Optional[torch.Tensor] = None,
        # 可选参数，过去的键值对列表
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        # 可选参数，是否使用缓存
        use_cache: Optional[bool] = None,
        # 可选参数，是否输出注意力权重
        output_attentions: Optional[bool] = None,
        # 可选参数，是否输出隐藏状态
        output_hidden_states: Optional[bool] = None,
        # 可选参数，是否返回字典形式的结果
        return_dict: Optional[bool] = None,
class ElectraClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        # 定义一个全连接层，用于处理 ELECTRA 模型的隐藏状态
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 获取分类器的 dropout 率，如果未指定则使用隐藏状态的 dropout 率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 获取激活函数，这里使用 GELU
        self.activation = get_activation("gelu")
        # 定义一个 dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(classifier_dropout)
        # 定义一个全连接层，用于将隐藏状态映射到标签空间
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        # 提取序列的第一个 token（即 [CLS] 标记）的隐藏状态
        x = features[:, 0, :]
        # 对提取的隐藏状态进行 dropout
        x = self.dropout(x)
        # 将隐藏状态输入全连接层
        x = self.dense(x)
        # 使用激活函数激活全连接层的输出
        x = self.activation(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
        # 再次对输出进行 dropout
        x = self.dropout(x)
        # 将输出输入到最终的全连接层，得到分类结果
        x = self.out_proj(x)
        return x


@add_start_docstrings(
    """
    ELECTRA Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    ELECTRA_START_DOCSTRING,
)
class ElectraForSequenceClassification(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 存储标签数量
        self.num_labels = config.num_labels
        # 存储配置
        self.config = config
        # 实例化 ELECTRA 模型
        self.electra = ElectraModel(config)
        # 实例化分类器头部
        self.classifier = ElectraClassificationHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="bhadresh-savani/electra-base-emotion",
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'joy'",
        expected_loss=0.06,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 确定是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取判别器的隐藏状态
        discriminator_hidden_states = self.electra(
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
        sequence_output = discriminator_hidden_states[0]
        # 通过分类器获取logits
        logits = self.classifier(sequence_output)

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

            # 计算损失
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

        # 如果不使用返回字典，则返回结果
        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        # 使用返回字典的情况下返回结果
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )
# 使用add_start_docstrings函数添加模型描述的开头文档字符串
# 这个Electra模型带有一个用于在预训练期间识别生成的标记的二元分类头
# 建议加载判别器检查点到该模型中
@add_start_docstrings(
    """
    Electra model with a binary classification head on top as used during pretraining for identifying generated tokens.
    
    It is recommended to load the discriminator checkpoint into that model.
    """,
    ELECTRA_START_DOCSTRING,
)
class ElectraForPreTraining(ElectraPreTrainedModel):
    # 初始化方法，传入配置参数
    def __init__(self, config):
        super().__init__(config)
        
        # 实例化Electra模型
        self.electra = ElectraModel(config)
        # 实例化ElectraDiscriminatorPredictions
        self.discriminator_predictions = ElectraDiscriminatorPredictions(config)
        # 初始化权重并应用最终处理
        self.post_init()

    # forward方法用于模型前向传播
    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=ElectraForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,

# 使用add_start_docstrings函数添加模型描述的开头文档字符串
# 这个Electra模型带有一个在顶部的语言建模头
# 即使判别器和生成器都可能加载到该模型中，
# 生成器是两者中唯一经过训练用于遮蔽语言建模任务的模型
@add_start_docstrings(
    """
    Electra model with a language modeling head on top.
    
    Even though both the discriminator and generator may be loaded into this model, the generator is the only model of
    the two to have been trained for the masked language modeling task.
    """,
    ELECTRA_START_DOCSTRING,
)
class ElectraForMaskedLM(ElectraPreTrainedModel):
    # _tied_weights_keys是用于绑定权重的键列表
    _tied_weights_keys = ["generator_lm_head.weight"]

    # 初始化方法，传入配置参数
    def __init__(self, config):
        super().__init__(config)
        
        # 实例化Electra模型
        self.electra = ElectraModel(config)
        # 实例化ElectraGeneratorPredictions
        self.generator_predictions = ElectraGeneratorPredictions(config)

        # generator_lm_head是用于生成器语言建模的线性层
        self.generator_lm_head = nn.Linear(config.embedding_size, config.vocab_size)
        # 初始化权重并应用最终处理
        self.post_init()

    # 返回生成器语言建模的输出嵌入
    def get_output_embeddings(self):
        return self.generator_lm_head

    # 设置生成器语言建模的输出嵌入
    def set_output_embeddings(self, word_embeddings):
        self.generator_lm_head = word_embeddings

    # forward方法用于模型前向传播
    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="google/electra-small-generator",
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        mask="[MASK]",
        expected_output="'paris'",
        expected_loss=1.22,
    )
    # 定义了一个名为 forward 的方法，用于模型的前向传播
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入的 token IDs，可选参数，默认为 None
        attention_mask: Optional[torch.Tensor] = None,  # 注意力遮罩，可选参数，默认为 None
        token_type_ids: Optional[torch.Tensor] = None,  # token 类型 IDs，可选参数，默认为 None
        position_ids: Optional[torch.Tensor] = None,  # 位置 IDs，可选参数，默认为 None
        head_mask: Optional[torch.Tensor] = None,  # 注意力头遮罩，可选参数，默认为 None
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入的嵌入表示，可选参数，默认为 None
        labels: Optional[torch.Tensor] = None,  # 用于计算 MLM 损失的标签，可选参数，默认为 None
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选参数，默认为 None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选参数，默认为 None
        return_dict: Optional[bool] = None,  # 是否返回字典格式的输出，可选参数，默认为 None
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        # 如果 return_dict 为 None，则根据配置确定是否使用字典格式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 通过 Electra 模型进行前向传播
        generator_hidden_states = self.electra(
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
        # 获取生成器的序列输出
        generator_sequence_output = generator_hidden_states[0]

        # 通过生成器预测层进行预测得分的计算
        prediction_scores = self.generator_predictions(generator_sequence_output)
        # 通过生成器语言模型头层计算生成的预测分数
        prediction_scores = self.generator_lm_head(prediction_scores)

        loss = None
        # 如果 labels 不为 None，则计算 MLM 损失
        if labels is not None:
            # 定义交叉熵损失函数，-100 索引代表填充标记
            loss_fct = nn.CrossEntropyLoss()
            # 计算预测分数和标签之间的交叉熵损失
            loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果不返回字典格式的输出，则将结果组合成元组返回
        if not return_dict:
            output = (prediction_scores,) + generator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回 MaskedLMOutput 类型的输出
        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=generator_hidden_states.hidden_states,
            attentions=generator_hidden_states.attentions,
        )
# 为 Electra 模型添加一个标记分类的头
@add_start_docstrings(
    """
    Electra model with a token classification head on top.

    Both the discriminator and generator may be loaded into this model.
    """,
    ELECTRA_START_DOCSTRING,
)
class ElectraForTokenClassification(ElectraPreTrainedModel):
    # 初始化模型
    def __init__(self, config):
        # 调用父类的构造函数
        super().__init__(config)
        # 从配置中获取标签数量
        self.num_labels = config.num_labels

        # 初始化 Electra 基础模型
        self.electra = ElectraModel(config)
        # 设置分类器的 dropout 率，如果配置中指定了 classifier_dropout 则使用它，否则使用 hidden_dropout_prob
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 初始化 dropout 层
        self.dropout = nn.Dropout(classifier_dropout)
        # 初始化线性分类器，用于将 Electra 输出的隐藏状态转为类别
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # 初始化权重并应用最终处理
        self.post_init()

    # 为模型的前向传递函数添加文档字符串
    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 为代码样例添加文档字符串，包括检查点、输出类型、配置类和预期输出
    @add_code_sample_docstrings(
        checkpoint="bhadresh-savani/electra-base-discriminator-finetuned-conll03-english",
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="['B-LOC', 'B-ORG', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'O', 'B-LOC', 'I-LOC']",
        expected_loss=0.11,
    )
    # 模型的前向传递函数，处理输入数据并返回输出
    def forward(
        # 输入 ID 张量，包含输入序列的标记 ID
        self,
        input_ids: Optional[torch.Tensor] = None,
        # 注意力掩码，用于指定哪些标记需要被注意
        attention_mask: Optional[torch.Tensor] = None,
        # 标记类型 ID，通常用于区分段落或标记的类型
        token_type_ids: Optional[torch.Tensor] = None,
        # 位置 ID，用于标记在序列中的位置
        position_ids: Optional[torch.Tensor] = None,
        # 头掩码，用于掩盖特定的注意力头
        head_mask: Optional[torch.Tensor] = None,
        # 输入嵌入，允许直接传入嵌入层的输出
        inputs_embeds: Optional[torch.Tensor] = None,
        # 标签，用于在训练时计算损失
        labels: Optional[torch.Tensor] = None,
        # 是否输出注意力权重
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典形式的输出
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 如果 return_dict 参数未指定，则使用模型配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 Electra 模型处理输入序列，返回包含各种输出的命名元组或字典
        discriminator_hidden_states = self.electra(
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
        # 从 Electra 模型的输出中获取鉴别器的序列输出
        discriminator_sequence_output = discriminator_hidden_states[0]

        # 对鉴别器序列输出进行 dropout 操作
        discriminator_sequence_output = self.dropout(discriminator_sequence_output)
        # 将 dropout 后的输出传递给分类器，得到预测的 logits
        logits = self.classifier(discriminator_sequence_output)

        # 计算损失，如果提供了标签
        loss = None
        if labels is not None:
            # 定义交叉熵损失函数
            loss_fct = CrossEntropyLoss()
            # 将 logits 和标签转换为二维张量并计算损失
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不要求返回字典，则按需返回结果元组
        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        # 否则，返回 TokenClassifierOutput 命名元组，其中包含损失、logits、隐藏状态和注意力
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )
# 导入必要的库
@add_start_docstrings(
    """
    ELECTRA Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    ELECTRA_START_DOCSTRING,
)
# 定义一个 ElectraForQuestionAnswering 类，继承自 ElectraPreTrainedModel 类
class ElectraForQuestionAnswering(ElectraPreTrainedModel):
    # 指定配置类
    config_class = ElectraConfig
    # 指定基础模型前缀
    base_model_prefix = "electra"

    # 初始化函数
    def __init__(self, config):
        # 调用父类初始化方法
        super().__init__(config)
        # 设置标签数量
        self.num_labels = config.num_labels

        # 创建 ElectraModel 实例
        self.electra = ElectraModel(config)
        # 创建用于 QA 的输出线性层
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数
    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="bhadresh-savani/electra-base-squad2",
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        qa_target_start_index=11,
        qa_target_end_index=12,
        expected_output="'a nice puppet'",
        expected_loss=2.64,
    )
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
# 添加模型的文档字符串和示例
@add_start_docstrings(
    """
    ELECTRA Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    ELECTRA_START_DOCSTRING,
)
# 定义一个 ElectraForMultipleChoice 类，继承自 ElectraPreTrainedModel 类
class ElectraForMultipleChoice(ElectraPreTrainedModel):
    # 初始化函数
    def __init__(self, config):
        # 调用父类初始化方法
        super().__init__(config)

        # 创建 ElectraModel 实例
        self.electra = ElectraModel(config)
        # 创建序列摘要
        self.sequence_summary = SequenceSummary(config)
        # 创建分类器
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数
    @add_start_docstrings_to_model_forward(ELECTRA_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 接收输入的Token ID张量，默认为None
        attention_mask: Optional[torch.Tensor] = None,  # 接收注意力遮罩张量，默认为None
        token_type_ids: Optional[torch.Tensor] = None,  # 接收标记类型ID张量，默认为None
        position_ids: Optional[torch.Tensor] = None,  # 接收位置ID张量，默认为None
        head_mask: Optional[torch.Tensor] = None,  # 接收头部遮罩张量，默认为None
        inputs_embeds: Optional[torch.Tensor] = None,  # 接收嵌入输入张量，默认为None
        labels: Optional[torch.Tensor] = None,  # 接收标签张量，默认为None
        output_attentions: Optional[bool] = None,  # 是否输出注意力，默认为None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，默认为None
        return_dict: Optional[bool] = None,  # 是否返回字典，默认为None
    ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:  # 返回的值的类型为元组或MultipleChoiceModelOutput类型
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # 如果return_dict不为None，则使用return_dict；否则使用self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]  # 如果input_ids不为None，则num_choices为input_ids的第二维大小；否则为inputs_embeds的第二维大小

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None  # 如果input_ids不为None，则将其视图重塑为(-1, input_ids.size(-1))；否则为None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None  # 如果attention_mask不为None，则将其视图重塑为(-1, attention_mask.size(-1))；否则为None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None  # 如果token_type_ids不为None，则将其视图重塑为(-1, token_type_ids.size(-1))；否则为None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None  # 如果position_ids不为None，则将其视图重塑为(-1, position_ids.size(-1))；否则为None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))  # 如果inputs_embeds不为None，则将其��图重塑为(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        discriminator_hidden_states = self.electra(  # 使用electra模型
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

        sequence_output = discriminator_hidden_states[0]  # 序列输出为discriminator_hidden_states的第一个值

        pooled_output = self.sequence_summary(sequence_output)  # 汇总输出为sequence_output的汇总
        logits = self.classifier(pooled_output)  # logits为classifier模型的输出
        reshaped_logits = logits.view(-1, num_choices)  # 重塑后的logits

        loss = None  # 初始化损失为None
        if labels is not None:  # 如果标签不为None
            loss_fct = CrossEntropyLoss()  # 交叉熵损失函数
            loss = loss_fct(reshaped_logits, labels)  # 计算损失

        if not return_dict:  # 如果不返回字典
            output = (reshaped_logits,) + discriminator_hidden_states[1:]  # 输出为reshaped_logits和除了第一个值外的所有discriminator_hidden_states
            return ((loss,) + output) if loss is not None else output  # 如果损失不为None，则返回损失和output；否则返回output

        return MultipleChoiceModelOutput(  # 返回MultipleChoiceModelOutput类型
            loss=loss,  # 损失
            logits=reshaped_logits,  # 重塑后的logits
            hidden_states=discriminator_hidden_states.hidden_states,  # 隐藏状态
            attentions=discriminator_hidden_states.attentions,  # 注意力
        )
# 为 ELECTRA CLM 模型添加文档说明，包括其为 `language modeling` 模型的头部 fine-tuning
class ElectraForCausalLM(ElectraPreTrainedModel):
    # 设置与 generator_lm_head 权重相绑定的键
    _tied_weights_keys = ["generator_lm_head.weight"]

    def __init__(self, config):
        # 初始化 ElectraForCausalLM 类
        super().__init__(config)

        # 如果配置不是解码器，发出警告提示
        if not config.is_decoder:
            logger.warning("If you want to use `ElectraForCausalLM` as a standalone, add `is_decoder=True.`")

        # 创建 ElectraModel 实例和 ElectraGeneratorPredictions 实例
        self.electra = ElectraModel(config)
        self.generator_predictions = ElectraGeneratorPredictions(config)
        # 创建线性层，将 embedding_size 映射到 vocab_size
        self.generator_lm_head = nn.Linear(config.embedding_size, config.vocab_size)

        # 初始化权重
        self.init_weights()

    # 获取输出 embeddings
    def get_output_embeddings(self):
        return self.generator_lm_head

    # 设置新的 embeddings
    def set_output_embeddings(self, new_embeddings):
        self.generator_lm_head = new_embeddings

    # 前向传播函数
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 生成序列输入前的准备函数，包括处理 attention_mask 和 past_key_values
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **model_kwargs):
        # 获取输入张量形状
        input_shape = input_ids.shape
        # 如果没有给定注意力遮罩，则创建全为1的注意力遮罩
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # 当使用 past_key_values 时，切割 decoder_input_ids
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法已经传递了只有最后一个输入 ID 的情况
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认旧的行为：保留最后一个 ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        # 返回包含信息的字典
        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past_key_values}
```py  
    # 从 transformers.models.roberta.modeling_roberta.RobertaForCausalLM._reorder_cache 复制的方法，重新排序缓存数据
    def _reorder_cache(self, past_key_values, beam_idx):
        # 重新排序后的缓存
        reordered_past = ()
        # 遍历过去的键值对
        for layer_past in past_key_values:
            # 对每一层的过去状态进行重新排序，并添加到重新排序后的缓存中
            reordered_past += (
                # 使用给定的索引重新排序每个过去状态，然后添加到重新排序后的缓存中
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重新排序后的缓存
        return reordered_past
```