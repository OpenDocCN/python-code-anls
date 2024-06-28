# `.\models\roc_bert\modeling_roc_bert.py`

```
# 设置文件编码为 UTF-8
# 版权声明，版权归 WeChatAI The HuggingFace Inc. team 所有
# 根据 Apache License, Version 2.0 许可协议，除非符合许可协议的要求，否则不得使用此文件
# 可以在以下网址获取许可协议的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按“原样”分发本软件
""" PyTorch RoCBert 模型."""

import math  # 导入 math 库
import os  # 导入 os 库
from typing import List, Optional, Tuple, Union  # 导入类型提示相关的类和函数

import torch  # 导入 PyTorch 库
import torch.utils.checkpoint  # 导入 PyTorch 的 checkpoint 模块
from torch import nn  # 从 PyTorch 中导入 nn 模块
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss  # 从 PyTorch 的 nn 模块导入三种损失函数

from ...activations import ACT2FN  # 导入激活函数相关
from ...modeling_outputs import (  # 导入模型输出相关类
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel  # 导入预训练模型基类
from ...pytorch_utils import (  # 导入 PyTorch 工具函数
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from ...utils import (  # 导入通用工具函数
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_roc_bert import RoCBertConfig  # 导入 RoCBert 的配置文件

logger = logging.get_logger(__name__)  # 获取日志记录器

_CHECKPOINT_FOR_DOC = "weiweishi/roc-bert-base-zh"  # 预期的文档检查点
_CONFIG_FOR_DOC = "RoCBertConfig"  # 预期的配置文件

# Base model docstring
_EXPECTED_OUTPUT_SHAPE = [1, 8, 768]  # 预期的输出形状

# Token Classification output
_CHECKPOINT_FOR_TOKEN_CLASSIFICATION = "ArthurZ/dummy-rocbert-ner"  # 用于令牌分类的检查点
_TOKEN_CLASS_EXPECTED_OUTPUT = [
    "S-EVENT", "S-FAC", "I-ORDINAL", "I-ORDINAL", "E-ORG", "E-LANGUAGE", "E-ORG", "E-ORG", "E-ORG", "E-ORG",
    "I-EVENT", "S-TIME", "S-TIME", "E-LANGUAGE", "S-TIME", "E-DATE", "I-ORDINAL", "E-QUANTITY", "E-LANGUAGE",
    "S-TIME", "B-ORDINAL", "S-PRODUCT", "E-LANGUAGE", "E-LANGUAGE", "E-ORG", "E-LOC", "S-TIME", "I-ORDINAL",
    "S-FAC", "O", "S-GPE", "I-EVENT", "S-GPE", "E-LANGUAGE", "E-ORG", "S-EVENT", "S-FAC", "S-FAC", "S-FAC",
    "E-ORG", "S-FAC", "E-ORG", "S-GPE"
]  # 令牌分类任务的预期输出
_TOKEN_CLASS_EXPECTED_LOSS = 3.62  # 令牌分类任务的预期损失值

# SequenceClassification docstring
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "ArthurZ/dummy-rocbert-seq"  # 用于序列分类的检查点
_SEQ_CLASS_EXPECTED_OUTPUT = "'financial news'"  # 序列分类任务的预期输出
_SEQ_CLASS_EXPECTED_LOSS = 2.31  # 序列分类任务的预期损失值

# QuestionAsnwering docstring
_CHECKPOINT_FOR_QA = "ArthurZ/dummy-rocbert-qa"  # 用于问答任务的检查点
_QA_EXPECTED_OUTPUT = "''"  # 问答任务的预期输出
_QA_EXPECTED_LOSS = 3.75  # 问答任务的预期损失值
_QA_TARGET_START_INDEX = 14  # 问答任务目标答案的起始索引
_QA_TARGET_END_INDEX = 15  # 问答任务目标答案的结束索引

# Maske language modeling
ROC_BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "weiweishi/roc-bert-base-zh",  # RoCBert 预训练模型的存档列表
    # 查看所有 RoCBert 模型，请访问 https://huggingface.co/models?filter=roc_bert
]
# 从 transformers.models.bert.modeling_bert.load_tf_weights_in_bert 复制并修改为 roc_bert
def load_tf_weights_in_roc_bert(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re  # 导入正则表达式模块，用于处理字符串匹配
        import numpy as np  # 导入 NumPy 库，用于处理数组和数值运算
        import tensorflow as tf  # 导入 TensorFlow 库，用于加载 TensorFlow 模型
    except ImportError:
        # 如果导入失败，打印错误信息并抛出异常
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise

    tf_path = os.path.abspath(tf_checkpoint_path)  # 获取 TensorFlow 检查点文件的绝对路径
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")  # 记录日志，指示正在转换的 TensorFlow 检查点路径

    # 从 TensorFlow 模型加载权重
    init_vars = tf.train.list_variables(tf_path)  # 列出 TensorFlow 模型中的所有变量及其形状
    names = []
    arrays = []

    # 遍历 TensorFlow 模型的每个变量
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")  # 记录日志，指示正在加载的 TensorFlow 权重名称和形状
        array = tf.train.load_variable(tf_path, name)  # 加载指定名称的 TensorFlow 变量数据
        names.append(name)  # 将变量名称添加到列表中
        arrays.append(array)  # 将加载的变量数据添加到列表中
    # 遍历输入的 names 和 arrays 列表，每次迭代将 name 和 array 组合在一起
    for name, array in zip(names, arrays):
        # 将 name 按 "/" 分割成列表
        name = name.split("/")
        
        # 检查 name 中是否包含特定的字符串，如果包含则跳过当前迭代
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            # 记录日志，跳过当前迭代
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        
        # 初始化指针为模型本身
        pointer = model
        
        # 遍历 name 中的每个元素
        for m_name in name:
            # 如果 m_name 符合形如 "xxx_0" 的格式，则按下划线分割为列表
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            
            # 根据 scope_names[0] 的不同值，调整指针的位置
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                # 尝试从 pointer 中获取 scope_names[0] 对应的属性，若失败则记录日志并跳过当前迭代
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            
            # 如果 scope_names 的长度大于等于2，则将第二部分作为数字索引，更新 pointer
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        
        # 如果 m_name 的后11个字符是 "_embeddings"，则将 pointer 设置为 "weight" 属性
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            # 如果 m_name 是 "kernel"，则将 array 转置为 numpy 数组
            array = np.transpose(array)
        
        # 检查 pointer 和 array 的形状是否匹配，若不匹配则抛出 ValueError
        try:
            if pointer.shape != array.shape:
                raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
        except ValueError as e:
            e.args += (pointer.shape, array.shape)
            raise
        
        # 记录日志，初始化 PyTorch 权重
        logger.info(f"Initialize PyTorch weight {name}")
        
        # 将 numpy 数组 array 转换为 torch 张量，并赋值给 pointer.data
        pointer.data = torch.from_numpy(array)
    
    # 返回更新后的模型
    return model
class RoCBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position, shape, pronunciation and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        # Word embeddings with vocab_size and hidden_size dimensions, supporting padding
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # Embeddings for pronunciation with pronunciation_vocab_size and pronunciation_embed_dim dimensions, supporting padding
        self.pronunciation_embed = nn.Embedding(
            config.pronunciation_vocab_size, config.pronunciation_embed_dim, padding_idx=config.pad_token_id
        )
        # Embeddings for shape with shape_vocab_size and shape_embed_dim dimensions, supporting padding
        self.shape_embed = nn.Embedding(
            config.shape_vocab_size, config.shape_embed_dim, padding_idx=config.pad_token_id
        )
        # Position embeddings with max_position_embeddings and hidden_size dimensions
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # Token type embeddings with type_vocab_size and hidden_size dimensions
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # Flags to enable/disable pronunciation and shape embeddings
        self.enable_pronunciation = config.enable_pronunciation
        self.enable_shape = config.enable_shape

        # Linear layer to map concatenated input dimensions to hidden_size if concat_input is enabled
        if config.concat_input:
            input_dim = config.hidden_size
            if self.enable_pronunciation:
                pronunciation_dim = config.pronunciation_embed_dim
                input_dim += pronunciation_dim
            if self.enable_shape:
                shape_dim = config.shape_embed_dim
                input_dim += shape_dim
            self.map_inputs_layer = torch.nn.Linear(input_dim, config.hidden_size)
        else:
            self.map_inputs_layer = None

        # Layer normalization over hidden_size dimension with specified epsilon
        # Note: 'self.LayerNorm' maintains naming consistency with TensorFlow for compatibility
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout layer with specified probability
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Register position_ids buffer as a 1D tensor of length max_position_embeddings
        # This is used for position embeddings and exported during serialization
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # Type of position embedding ('absolute' by default)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # Register token_type_ids buffer initialized with zeros, same size as position_ids
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
            persistent=False,
        )

    def forward(
        self,
        input_ids=None,
        input_shape_ids=None,
        input_pronunciation_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
    ):
        # Forward method implementation will be specific to the usage of these embeddings
        pass

# Copied from transformers.models.bert.modeling_bert.BertSelfAttention with Bert->RoCBert
class RoCBertSelfAttention(nn.Module):
    # This class definition will follow here but is not included in the requested annotation
    # 初始化方法，接收配置和位置嵌入类型作为参数
    def __init__(self, config, position_embedding_type=None):
        # 调用父类初始化方法
        super().__init__()
        # 检查隐藏层大小是否是注意力头数的整数倍，如果不是且配置中没有嵌入大小属性，则抛出数值错误
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

        # 初始化dropout层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # 设置位置嵌入类型，默认为绝对位置编码
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果位置嵌入类型是相对键或相对键查询，则初始化距离嵌入
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        # 设置是否为解码器
        self.is_decoder = config.is_decoder
# 从 transformers.models.bert.modeling_bert.BertSelfOutput 复制代码并修改为 RoCBertSelfOutput
class RoCBertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义一个全连接层，输入和输出维度都是 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # LayerNorm 层，输入维度 config.hidden_size，使用给定的 epsilon 值初始化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout 层，根据给定的概率 config.hidden_dropout_prob 随机丢弃输入张量中的部分元素
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用全连接层 dense 对 hidden_states 进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对变换后的 hidden_states 进行 dropout 处理
        hidden_states = self.dropout(hidden_states)
        # 将 dropout 处理后的 hidden_states 与 input_tensor 相加，然后通过 LayerNorm 进行归一化处理
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# 从 transformers.models.bert.modeling_bert.BertAttention 复制代码并修改为 RoCBertAttention
class RoCBertAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # RoCBertSelfAttention 类的实例化对象
        self.self = RoCBertSelfAttention(config, position_embedding_type=position_embedding_type)
        # RoCBertSelfOutput 类的实例化对象
        self.output = RoCBertSelfOutput(config)
        # 用于存储被裁剪掉的注意力头的集合
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 查找可裁剪头部的索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 裁剪线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储裁剪的头部
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
        # 调用 self 层的 forward 方法
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 调用 output 层的 forward 方法，将 self 输出和 hidden_states 作为输入
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果输出注意力信息，则将其添加到输出中
        outputs = (attention_output,) + self_outputs[1:]  # 如果需要输出注意力信息，则添加
        return outputs


# 从 transformers.models.bert.modeling_bert.BertIntermediate 复制代码并修改为 RoCBertIntermediate
class RoCBertIntermediate(nn.Module):
    # 定义类的初始化方法，接收一个配置对象 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个全连接层，输入大小为 config.hidden_size，输出大小为 config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        
        # 判断 config.hidden_act 是否为字符串类型
        if isinstance(config.hidden_act, str):
            # 如果是字符串类型，则从预定义的字典 ACT2FN 中获取对应的激活函数
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            # 如果不是字符串类型，则直接使用配置中的激活函数对象
            self.intermediate_act_fn = config.hidden_act

    # 定义前向传播方法，接收隐藏状态的张量 hidden_states，返回处理后的张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态张量传入全连接层，得到输出的隐藏状态张量
        hidden_states = self.dense(hidden_states)
        # 将全连接层的输出张量传入中间激活函数，得到最终的隐藏状态张量
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回最终的隐藏状态张量作为输出
        return hidden_states
# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->RoCBert
class RoCBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化线性层，将隐藏层大小转换为配置中指定的隐藏大小
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 初始化层归一化，使用配置中指定的隐藏层大小和层归一化参数
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化丢弃层，使用配置中指定的隐藏丢弃概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用线性层对隐藏状态进行转换
        hidden_states = self.dense(hidden_states)
        # 对转换后的隐藏状态进行丢弃处理
        hidden_states = self.dropout(hidden_states)
        # 对丢弃处理后的隐藏状态进行层归一化，并加上输入张量
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的隐藏状态
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLayer with Bert->RoCBert
class RoCBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 设置前向传播中的分块大小，使用配置中指定的前馈传播分块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度维度设定为1
        self.seq_len_dim = 1
        # 初始化注意力层，使用RoCBertAttention类和给定的配置
        self.attention = RoCBertAttention(config)
        # 是否作为解码器使用
        self.is_decoder = config.is_decoder
        # 是否添加交叉注意力
        self.add_cross_attention = config.add_cross_attention
        # 如果添加交叉注意力，且非解码器模型，则引发错误
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 初始化交叉注意力层，使用RoCBertAttention类、绝对位置嵌入类型和给定配置
            self.crossattention = RoCBertAttention(config, position_embedding_type="absolute")
        # 初始化中间层，使用给定配置
        self.intermediate = RoCBertIntermediate(config)
        # 初始化输出层，使用给定配置
        self.output = RoCBertOutput(config)

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
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # Perform self-attention operation using the provided inputs
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            # Extract all outputs except the last two, which are self-attention caches
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            # For encoder or other cases, include all self-attention outputs
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # Perform cross-attention using the provided inputs
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
            # Append cross-attention outputs to existing outputs list
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # Apply chunking mechanism to feed forward layer
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        # Pass attention_output through intermediate and output layers
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
# 从transformers.models.bert.modeling_bert.BertEncoder复制代码并将Bert->RoCBert
class RoCBertEncoder(nn.Module):
    # RoCBertEncoder类的初始化函数，接受一个config对象作为参数
    def __init__(self, config):
        super().__init__()
        # 将传入的config对象保存到实例变量中
        self.config = config
        # 创建一个包含多个RoCBertLayer对象的ModuleList，层数由config.num_hidden_layers指定
        self.layer = nn.ModuleList([RoCBertLayer(config) for _ in range(config.num_hidden_layers)])
        # 设置梯度检查点标志为False
        self.gradient_checkpointing = False

    # RoCBertEncoder类的前向传播函数，接受多个参数
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
        # 如果输出隐藏状态，则初始化一个空元组；否则设为 None
        all_hidden_states = () if output_hidden_states else None
        # 如果输出注意力权重，则初始化一个空元组；否则设为 None
        all_self_attentions = () if output_attentions else None
        # 如果输出交叉注意力权重且配置允许，则初始化一个空元组；否则设为 None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 如果启用了梯度检查点且处于训练模式下
        if self.gradient_checkpointing and self.training:
            # 如果 use_cache 为 True，则发出警告并强制设为 False
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # 如果 use_cache 为 True，则初始化一个空元组；否则设为 None
        next_decoder_cache = () if use_cache else None
        # 遍历每个解码层
        for i, layer_module in enumerate(self.layer):
            # 如果输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码，如果没有则设为 None
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 获取当前层的过去键值对，如果没有则设为 None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 如果启用了梯度检查点且处于训练模式下
            if self.gradient_checkpointing and self.training:
                # 使用梯度检查点功能来计算当前层的输出
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
                # 否则直接调用当前层的模块计算输出
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            # 更新隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]
            # 如果 use_cache 为 True，则将当前层的输出的最后一个元素添加到 next_decoder_cache 中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 如果输出注意力权重，则将当前层的输出的第二个元素添加到 all_self_attentions 中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果配置允许输出交叉注意力权重，则将当前层的输出的第三个元素添加到 all_cross_attentions 中
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果输出隐藏状态，则将最终隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典形式的结果，则以元组形式返回特定的值，排除空值
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
        # 否则返回包含特定字段的 BaseModelOutputWithPastAndCrossAttentions 对象
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
# Copied from transformers.models.bert.modeling_bert.BertPooler with Bert->RoCBert
class RoCBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义一个全连接层，输入和输出大小都为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义激活函数为双曲正切函数
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 从 hidden_states 中取出第一个 token 对应的隐藏状态
        first_token_tensor = hidden_states[:, 0]
        # 将该隐藏状态作为输入，经过全连接层得到池化输出
        pooled_output = self.dense(first_token_tensor)
        # 对池化输出应用激活函数
        pooled_output = self.activation(pooled_output)
        return pooled_output


# Copied from transformers.models.bert.modeling_bert.BertPredictionHeadTransform with Bert->RoCBert
class RoCBertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义一个全连接层，输入和输出大小都为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 根据 config 中的隐藏层激活函数类型，选择相应的激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # 使用 LayerNorm 对隐藏状态进行归一化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 经过全连接层变换隐藏状态
        hidden_states = self.dense(hidden_states)
        # 应用激活函数变换隐藏状态
        hidden_states = self.transform_act_fn(hidden_states)
        # 对变换后的隐藏状态进行 LayerNorm 处理
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLMPredictionHead with Bert->RoCBert
class RoCBertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用 RoCBertPredictionHeadTransform 对隐藏状态进行变换
        self.transform = RoCBertPredictionHeadTransform(config)

        # 输出权重与输入嵌入相同，但每个 token 有一个仅输出的偏置
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # 需要一个链接以确保偏置能够与 `resize_token_embeddings` 正确调整大小
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        # 经过变换层处理隐藏状态
        hidden_states = self.transform(hidden_states)
        # 经过线性层得到预测分数
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOnlyMLMHead with Bert->RoCBert
class RoCBertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用 RoCBertLMPredictionHead 进行 MLM 预测
        self.predictions = RoCBertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        # 调用 RoCBertLMPredictionHead 进行序列输出的预测
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


# Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel with Bert->RoCBert,bert->roc_bert
class RoCBertPreTrainedModel(PreTrainedModel):
    """
    RoCBert 模型的基类，继承自 PreTrainedModel
    """
    # RoCBertConfig 类用于配置 RoCBert 模型的参数和配置
    config_class = RoCBertConfig
    # load_tf_weights 指定了在 RoCBert 中加载 TensorFlow 权重的函数
    load_tf_weights = load_tf_weights_in_roc_bert
    # base_model_prefix 指定了 RoCBert 模型的基础模型前缀
    base_model_prefix = "roc_bert"
    # supports_gradient_checkpointing 表示 RoCBert 模型支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果 module 是 nn.Linear 类型，初始化其权重为正态分布
        if isinstance(module, nn.Linear):
            # PyTorch 版本使用正态分布初始化，与 TensorFlow 的截断正态分布稍有不同
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果 module 有偏置项，将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果 module 是 nn.Embedding 类型，初始化其权重为正态分布
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果定义了 padding_idx，则将对应位置的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果 module 是 nn.LayerNorm 类型，初始化其偏置为零，权重为 1.0
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
# RoCBert 模型的文档字符串，描述该模型是一个 PyTorch 的子类，可以作为普通的 PyTorch Module 使用。
# 参考 PyTorch 文档以了解一般用法和行为。
ROC_BERT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`RoCBertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# RoCBert 模型的输入文档字符串，目前为空，需要进一步填写
ROC_BERT_INPUTS_DOCSTRING = r"""
"""


@add_start_docstrings(
    "The bare RoCBert Model transformer outputting raw hidden-states without any specific head on top.",
    ROC_BERT_START_DOCSTRING,
)
# 定义 RoCBertModel 类，继承自 RoCBertPreTrainedModel
class RoCBertModel(RoCBertPreTrainedModel):
    """
    RoCBertModel 可以作为一个编码器（仅自注意力）或解码器使用。在解码器模式下，将在自注意力层之间添加交叉注意力层，
    遵循 [Attention is all you need](https://arxiv.org/abs/1706.03762) 中描述的架构。
    """

    # 从 transformers.models.bert.modeling_bert.BertModel.__init__ 复制过来的初始化方法，将 Bert 替换为 RoCBert
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        # 初始化 RoCBertEmbeddings、RoCBertEncoder
        self.embeddings = RoCBertEmbeddings(config)
        self.encoder = RoCBertEncoder(config)

        # 如果指定了 add_pooling_layer，则初始化 RoCBertPooler；否则设为 None
        self.pooler = RoCBertPooler(config) if add_pooling_layer else None

        # 初始化权重并应用最终处理
        self.post_init()

    # 从 transformers.models.bert.modeling_bert.BertModel.get_input_embeddings 复制过来的方法
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 从 transformers.models.bert.modeling_bert.BertModel.set_input_embeddings 复制过来的方法
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # 获取发音嵌入的方法
    def get_pronunciation_embeddings(self):
        return self.embeddings.pronunciation_embed

    # 设置发音嵌入的方法
    def set_pronunciation_embeddings(self, value):
        self.embeddings.pronunciation_embed = value

    # 获取形状嵌入的方法
    def get_shape_embeddings(self):
        return self.embeddings.shape_embed

    # 设置形状嵌入的方法
    def set_shape_embeddings(self, value):
        self.embeddings.shape_embed = value

    # 从 transformers.models.bert.modeling_bert.BertModel._prune_heads 复制过来的方法
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 遍历 heads_to_prune 字典，其中每个键是层号，对应的值是要在该层中剪枝的注意力头列表
        for layer, heads in heads_to_prune.items():
            # 在模型的编码器中找到指定层，然后调用其注意力机制的 prune_heads 方法进行剪枝操作
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(ROC_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    # 重写 forward 方法，增加了详细的文档字符串说明和代码示例说明
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_shape_ids: Optional[torch.Tensor] = None,
        input_pronunciation_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
@add_start_docstrings(
    """
    RoCBert Model with contrastive loss and masked_lm_loss during the pretraining.
    """,
    ROC_BERT_START_DOCSTRING,
)
class RoCBertForPreTraining(RoCBertPreTrainedModel):
    _tied_weights_keys = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        # Initialize RoCBert model with the provided configuration
        self.roc_bert = RoCBertModel(config)
        # Initialize RoCBertOnlyMLMHead for masked language modeling
        self.cls = RoCBertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.bert.modeling_bert.BertForPreTraining.get_output_embeddings
    def get_output_embeddings(self):
        # Return the decoder layer of the MLM head
        return self.cls.predictions.decoder

    # Copied from transformers.models.bert.modeling_bert.BertForPreTraining.set_output_embeddings
    def set_output_embeddings(self, new_embeddings):
        # Set new embeddings for the decoder layer of the MLM head
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(ROC_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_shape_ids: Optional[torch.Tensor] = None,
        input_pronunciation_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        attack_input_ids: Optional[torch.Tensor] = None,
        attack_input_shape_ids: Optional[torch.Tensor] = None,
        attack_input_pronunciation_ids: Optional[torch.Tensor] = None,
        attack_attention_mask: Optional[torch.Tensor] = None,
        attack_token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels_input_ids: Optional[torch.Tensor] = None,
        labels_input_shape_ids: Optional[torch.Tensor] = None,
        labels_input_pronunciation_ids: Optional[torch.Tensor] = None,
        labels_attention_mask: Optional[torch.Tensor] = None,
        labels_token_type_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
):
    """
    RoCBert Model with a `language modeling` head on top.
    """
    # Implementation details for the forward method are omitted as per the task instructions.
    def __init__(self, config):
        # 调用父类的构造函数初始化对象
        super().__init__(config)

        # 如果配置中设置为解码器，则发出警告，提醒用户配置应为双向自注意力模型
        if config.is_decoder:
            logger.warning(
                "If you want to use `RoCBertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 初始化 RoCBertModel，禁用添加池化层的选项
        self.roc_bert = RoCBertModel(config, add_pooling_layer=False)

        # 初始化 RoCBertOnlyMLMHead
        self.cls = RoCBertOnlyMLMHead(config)

        # 调用后处理函数，初始化权重并应用最终处理
        self.post_init()

    # 从 transformers.models.bert.modeling_bert.BertForMaskedLM.get_output_embeddings 复制而来
    # 返回 MLM 头部的输出嵌入
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 从 transformers.models.bert.modeling_bert.BertForMaskedLM.set_output_embeddings 复制而来
    # 设置 MLM 头部的输出嵌入为新的嵌入
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(ROC_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 添加了注释字符串，描述了 forward 方法的输入参数
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_shape_ids: Optional[torch.Tensor] = None,
        input_pronunciation_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:
        ```python
        >>> from transformers import AutoTokenizer, RoCBertForMaskedLM
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("weiweishi/roc-bert-base-zh")
        >>> model = RoCBertForMaskedLM.from_pretrained("weiweishi/roc-bert-base-zh")

        >>> inputs = tokenizer("法国是首都[MASK].", return_tensors="pt")

        >>> with torch.no_grad():
        ...     logits = model(**inputs).logits

        >>> # retrieve index of {mask}
        >>> mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

        >>> predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        >>> tokenizer.decode(predicted_token_id)
        '.'
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 根据传入的参数决定是否使用返回字典模式

        outputs = self.roc_bert(
            input_ids,
            input_shape_ids=input_shape_ids,
            input_pronunciation_ids=input_pronunciation_ids,
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
        # 将输入传递给 RoCBert 模型进行前向传播，获取输出结果

        sequence_output = outputs[0]
        # 从模型的输出中获取序列输出

        prediction_scores = self.cls(sequence_output)
        # 将序列输出传递给分类层，生成预测分数

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            # 定义交叉熵损失函数，用于计算掩码语言建模的损失
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            # 计算掩码语言建模的损失，将预测分数和标签视图展平后输入损失函数中

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            # 如果不使用返回字典模式，则组装输出元组，包含预测分数和可能的额外输出
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            # 如果存在掩码语言建模损失，则将其加入输出元组中；否则，只返回预测分数和可能的额外输出

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # 使用 MaskedLMOutput 对象返回结果，包含损失、预测分数、隐藏状态和注意力权重
        ):
            # 获取输入张量的形状
            input_shape = input_ids.shape
            # 获取有效的批处理大小
            effective_batch_size = input_shape[0]

            # 添加一个虚拟标记
            # 如果配置中未定义PAD标记，则抛出数值错误
            if self.config.pad_token_id is None:
                raise ValueError("The PAD token should be defined for generation")

            # 将注意力遮罩张量与一个新的全零张量连接起来，扩展其长度
            attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
            # 创建一个填充了PAD标记的虚拟标记张量
            dummy_token = torch.full(
                (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
            )
            # 将虚拟标记张量连接到输入张量的末尾
            input_ids = torch.cat([input_ids, dummy_token], dim=1)
            # 如果存在输入形状ID，则将虚拟标记张量连接到其末尾
            if input_shape_ids is not None:
                input_shape_ids = torch.cat([input_shape_ids, dummy_token], dim=1)
            # 如果存在输入发音ID，则将虚拟标记张量连接到其末尾
            if input_pronunciation_ids is not None:
                input_pronunciation_ids = torch.cat([input_pronunciation_ids, dummy_token], dim=1)

            # 返回包含更新后张量的字典
            return {
                "input_ids": input_ids,
                "input_shape_ids": input_shape_ids,
                "input_pronunciation_ids": input_pronunciation_ids,
                "attention_mask": attention_mask,
            }
# 定义一个RoCBertForCausalLM类，用于在CLM微调时具有语言建模头部。
# 这里使用了一个装饰器@add_start_docstrings，用来添加类的文档字符串，来自于ROC_BERT_START_DOCSTRING。
class RoCBertForCausalLM(RoCBertPreTrainedModel):
    # 定义了_tied_weights_keys列表，包含与权重绑定相关的键名。
    _tied_weights_keys = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]

    # 从transformers.models.bert.modeling_bert.BertLMHeadModel.__init__方法复制而来，作为RoCBertForCausalLM类的初始化函数。
    def __init__(self, config):
        # 调用父类RoCBertPreTrainedModel的初始化函数。
        super().__init__(config)

        # 如果配置中不是decoder模式，则发出警告信息。
        if not config.is_decoder:
            logger.warning("If you want to use `RoCRoCBertForCausalLM` as a standalone, add `is_decoder=True.`")

        # 创建RoCBertModel对象，用于RoCBertForCausalLM的基础模型，不包含池化层。
        self.roc_bert = RoCBertModel(config, add_pooling_layer=False)
        # 创建RoCBertOnlyMLMHead对象，用于RoCBertForCausalLM的MLM头部。
        self.cls = RoCBertOnlyMLMHead(config)

        # 初始化权重并应用最终处理。
        self.post_init()

    # 从transformers.models.bert.modeling_bert.BertLMHeadModel.get_output_embeddings方法复制而来，返回MLM头部的输出嵌入。
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 从transformers.models.bert.modeling_bert.BertLMHeadModel.set_output_embeddings方法复制而来，设置MLM头部的输出嵌入。
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 从add_start_docstrings_to_model_forward装饰器添加模型前向传播的文档字符串，使用ROC_BERT_INPUTS_DOCSTRING格式化。
    # 还使用replace_return_docstrings装饰器，设置输出类型为CausalLMOutputWithCrossAttentions，配置类为_CONFIG_FOR_DOC。
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_shape_ids: Optional[torch.Tensor] = None,
        input_pronunciation_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        pass  # 这里只是函数定义，实际功能由后续代码实现。

    # 定义了prepare_inputs_for_generation方法，准备生成过程中的输入。
    def prepare_inputs_for_generation(
        self,
        input_ids,
        input_shape_ids=None,
        input_pronunciation_ids=None,
        past_key_values=None,
        attention_mask=None,
        **model_kwargs,
    ):
        pass  # 这里只是函数定义，实际功能由后续代码实现。
    ):
        # 获取输入张量的形状
        input_shape = input_ids.shape

        # 如果注意力遮罩为空，则创建全为1的遮罩张量，保证所有位置都被关注
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # 如果存在过去的键值对，则裁剪输入的decoder_input_ids
        if past_key_values is not None:
            # 获取过去键值对中第一个元素的长度
            past_length = past_key_values[0][0].shape[2]

            # 如果输入的decoder_input_ids长度大于过去长度，则移除前缀部分
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 否则保留最后一个输入ID，即旧的行为
                remove_prefix_length = input_ids.shape[1] - 1

            # 对输入的decoder_input_ids进行裁剪
            input_ids = input_ids[:, remove_prefix_length:]
            # 如果存在input_shape_ids，则也进行相应裁剪
            if input_shape_ids is not None:
                input_shape_ids = input_shape_ids[:, -1:]
            # 如果存在input_pronunciation_ids，则也进行相应裁剪
            if input_pronunciation_ids is not None:
                input_pronunciation_ids = input_pronunciation_ids[:, -1:]

        # 返回重排后的张量和相关信息的字典
        return {
            "input_ids": input_ids,
            "input_shape_ids": input_shape_ids,
            "input_pronunciation_ids": input_pronunciation_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
        }

    # 从transformers.models.bert.modeling_bert.BertLMHeadModel._reorder_cache复制而来
    def _reorder_cache(self, past_key_values, beam_idx):
        # 重新排列过去的键值对，以匹配beam搜索的索引顺序
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                # 将每个层次的过去状态按照beam_idx重新排序，并保持在原设备上
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
# 使用自定义的文档字符串添加类的描述信息，指出这是一个基于 RoCBert 的序列分类/回归模型
@add_start_docstrings(
    """RoCBert Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks.""",
    ROC_BERT_START_DOCSTRING,
)
# 定义 RoCBertForSequenceClassification 类，继承自 RoCBertPreTrainedModel 类
class RoCBertForSequenceClassification(RoCBertPreTrainedModel):
    # 从 transformers.models.bert.modeling_bert.BertForSequenceClassification.__init__ 方法复制而来，将 Bert 替换为 RoCBert，bert 替换为 roc_bert
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置类别数量
        self.num_labels = config.num_labels
        # 保存配置信息
        self.config = config

        # 创建 RoCBertModel 对象
        self.roc_bert = RoCBertModel(config)
        # 获取分类器的 dropout 概率，若未设置则使用隐藏层的 dropout 概率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 定义 dropout 层
        self.dropout = nn.Dropout(classifier_dropout)
        # 定义线性分类器
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并进行最终处理
        self.post_init()

    # 使用自定义的文档字符串添加到模型前向传播方法的描述信息
    @add_start_docstrings_to_model_forward(ROC_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 添加示例代码的文档字符串，指定检查点、输出类型、配置类、预期输出和损失
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    )
    # 定义前向传播方法
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_shape_ids: Optional[torch.Tensor] = None,
        input_pronunciation_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 参数说明: 输入的张量，包括输入 ID、形状 ID、发音 ID、注意力掩码、标记类型 ID、位置 ID、头部掩码、嵌入输入、标签、输出注意力、输出隐藏状态和是否返回字典

        # 参数说明: 输入的张量，包括输入 ID、形状 ID、发音 ID、注意力掩码、标记类型 ID、位置 ID、头部掩码、嵌入输入、标签、输出注意力、输出隐藏状态和是否返回字典
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 如果 return_dict 不为 None，则使用指定的 return_dict；否则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 RoC-BERT 模型进行前向传播
        outputs = self.roc_bert(
            input_ids,
            input_shape_ids=input_shape_ids,
            input_pronunciation_ids=input_pronunciation_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 提取池化后的输出
        pooled_output = outputs[1]

        # 对池化后的输出进行 dropout 处理
        pooled_output = self.dropout(pooled_output)
        # 将 dropout 后的输出传入分类器
        logits = self.classifier(pooled_output)

        # 初始化损失值为 None
        loss = None
        # 如果 labels 不为 None，则计算损失
        if labels is not None:
            # 确定问题类型（如果未指定）
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型选择相应的损失函数
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    # 如果只有一个标签（回归问题），计算均方误差损失
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    # 否则计算均方误差损失
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # 单标签分类问题，使用交叉熵损失
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # 多标签分类问题，使用带 logits 的二元交叉熵损失
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果 return_dict 为 False，则返回损失和输出
        if not return_dict:
            output = (logits,) + outputs[2:]  # 包含输出 logits 和可能的额外隐藏状态
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则以 SequenceClassifierOutput 对象形式返回结果
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用装饰器为该类添加文档字符串，描述其作为RoCBert模型的多选分类头的功能
@add_start_docstrings(
    """RoCBert Model with a multiple choice classification head on top (a linear layer on top of
    the pooled output and a softmax) e.g. for RocStories/SWAG tasks.""",
    ROC_BERT_START_DOCSTRING,
)
# 定义RoCBert模型的多选分类器，继承自RoCBertPreTrainedModel类
class RoCBertForMultipleChoice(RoCBertPreTrainedModel):
    
    # 从transformers库中的BertForMultipleChoice.__init__方法复制而来，修改为RoCBert相关命名
    def __init__(self, config):
        super().__init__(config)
        
        # 初始化RoCBert模型，根据给定配置参数
        self.roc_bert = RoCBertModel(config)
        
        # 设置分类器的dropout率为配置中的classifier_dropout，如果未指定则使用hidden_dropout_prob
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 定义一个dropout层，用于随机失活
        self.dropout = nn.Dropout(classifier_dropout)
        
        # 定义一个线性层，将RoCBert模型输出的隐藏状态转换为一个输出值，用于多选分类任务
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终的处理
        self.post_init()

    # 使用装饰器为forward方法添加文档字符串，描述其输入参数和输出结果
    @add_start_docstrings_to_model_forward(
        ROC_BERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    # 添加示例代码的文档字符串，显示checkpoint、输出类型和配置类的信息
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义模型的前向传播方法
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_shape_ids: Optional[torch.Tensor] = None,
        input_pronunciation_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 结束函数签名
    ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 根据 `return_dict` 参数确定是否返回字典形式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 获取选择题个数，即第二维度的大小
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 重新调整输入数据的形状，将其视作二维的，保留最后一维的大小
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        input_shape_ids = input_shape_ids.view(-1, input_shape_ids.size(-1)) if input_shape_ids is not None else None
        input_pronunciation_ids = (
            input_pronunciation_ids.view(-1, input_pronunciation_ids.size(-1))
            if input_pronunciation_ids is not None
            else None
        )
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 调用 RoCBERT 模型进行推理
        outputs = self.roc_bert(
            input_ids,
            input_shape_ids=input_shape_ids,
            input_pronunciation_ids=input_pronunciation_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取汇聚后的输出
        pooled_output = outputs[1]

        # 应用 dropout 操作
        pooled_output = self.dropout(pooled_output)
        # 通过分类器获取 logits
        logits = self.classifier(pooled_output)
        # 重新调整 logits 的形状，以适应多选题的需求
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            # 计算交叉熵损失
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            # 如果不使用字典形式输出，则返回元组
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回多选题模型的输出对象
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用自定义的文档字符串初始化 RoCBertForTokenClassification 类，用于在隐藏状态输出之上添加一个令牌分类头部，
# 例如用于命名实体识别（NER）任务的线性层。
@add_start_docstrings(
    """RoCBert Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.""",
    ROC_BERT_START_DOCSTRING,
)
# 继承 RoCBertPreTrainedModel 类并重写 __init__ 方法
class RoCBertForTokenClassification(RoCBertPreTrainedModel):
    
    # 从 transformers.models.bert.modeling_bert.BertForTokenClassification.__init__ 复制过来，将 Bert 替换为 RoCBert，bert 替换为 roc_bert
    def __init__(self, config):
        # 调用 RoCBertPreTrainedModel 类的初始化方法
        super().__init__(config)
        # 设置标签数量
        self.num_labels = config.num_labels
        
        # 使用 RoCBertModel 类初始化 roc_bert 属性，禁用添加池化层
        self.roc_bert = RoCBertModel(config, add_pooling_layer=False)
        
        # 根据配置中的 dropout 设置分类器的 dropout 层
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        
        # 使用线性层初始化分类器，输入尺寸为隐藏状态的尺寸，输出尺寸为标签数量
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 将 @add_start_docstrings_to_model_forward 和 @add_code_sample_docstrings 应用于 forward 方法
    @add_start_docstrings_to_model_forward(ROC_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_TOKEN_CLASSIFICATION,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_TOKEN_CLASS_EXPECTED_OUTPUT,
        expected_loss=_TOKEN_CLASS_EXPECTED_LOSS,
    )
    # 定义 forward 方法，处理输入和返回预测结果
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_shape_ids: Optional[torch.Tensor] = None,
        input_pronunciation_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 函数参数说明包含输入张量、掩码、ID 和标签，以及其他相关设置

        # 函数参数说明包含输入张量、掩码、ID 和标签，以及其他相关设置
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_shape_ids: Optional[torch.Tensor] = None,
        input_pronunciation_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    `
    # 定义函数签名和返回类型注解，声明函数返回的是一个元组或者 TokenClassifierOutput 类型的对象
    def forward(
        input_ids: torch.LongTensor,
        input_shape_ids: Optional[torch.LongTensor] = None,
        input_pronunciation_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 如果 return_dict 为 None，则使用 self.config.use_return_dict 的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # 调用底层的 RoC-BERT 模型进行前向传播，获取输出
        outputs = self.roc_bert(
            input_ids,
            input_shape_ids=input_shape_ids,
            input_pronunciation_ids=input_pronunciation_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
        # 获取模型的序列输出
        sequence_output = outputs[0]
    
        # 对序列输出进行 dropout 处理
        sequence_output = self.dropout(sequence_output)
        # 将 dropout 后的结果输入分类器，得到分类 logits
        logits = self.classifier(sequence_output)
    
        # 初始化损失为 None
        loss = None
        # 如果传入了标签 labels，则计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # 计算 logits 和 labels 的交叉熵损失
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    
        # 如果 return_dict 为 False，则返回包含 logits 和其他输出的元组
        if not return_dict:
# 使用装饰器为类添加文档字符串，描述了 RoCBert 模型及其在抽取式问答任务（如 SQuAD）中的作用
@add_start_docstrings(
    """RoCBert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).""",
    ROC_BERT_START_DOCSTRING,
)
# 定义 RoCBertForQuestionAnswering 类，继承自 RoCBertPreTrainedModel 类
class RoCBertForQuestionAnswering(RoCBertPreTrainedModel):
    
    # 从 transformers.models.bert.modeling_bert.BertForQuestionAnswering.__init__ 复制并修改而来，将 Bert 替换为 RoCBert
    def __init__(self, config):
        super().__init__(config)
        # 设置类别数目
        self.num_labels = config.num_labels

        # 初始化 RoCBertModel 实例，关闭 pooling 层
        self.roc_bert = RoCBertModel(config, add_pooling_layer=False)
        # 线性层，用于输出 span 起始位置和结束位置的 logits
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并进行最终处理
        self.post_init()

    # 使用装饰器为 forward 方法添加文档字符串，描述 RoCBertForQuestionAnswering 模型的输入和输出
    @add_start_docstrings_to_model_forward(ROC_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 使用装饰器为 forward 方法添加代码示例的文档字符串，展示其用法和预期输出
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_QA,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        qa_target_start_index=_QA_TARGET_START_INDEX,
        qa_target_end_index=_QA_TARGET_END_INDEX,
        expected_output=_QA_EXPECTED_OUTPUT,
        expected_loss=_QA_EXPECTED_LOSS,
    )
    # 前向传播函数定义，接收多个输入参数，包括输入的 token IDs、注意力掩码等
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_shape_ids: Optional[torch.Tensor] = None,
        input_pronunciation_ids: Optional[torch.Tensor] = None,
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
        # 输入参数说明结束
```