# `.\models\bert\modeling_bert.py`

```py
# coding=utf-8
# 版权声明，包括Google AI Language Team和HuggingFace Inc.的版权声明
# 版权声明，包括NVIDIA CORPORATION的版权声明
#
# 根据Apache许可证2.0版（“许可证”）许可使用本文件；
# 除非符合许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件根据“原样”分发，
# 没有任何形式的担保或条件，包括但不限于明示或暗示的任何担保或条件。
# 有关详细信息，请参阅许可证。
"""PyTorch BERT模型。"""


import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 从本地库中导入一些函数和类
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
# 从configuration_bert模块导入BertConfig类
from .configuration_bert import BertConfig

# 获取logger对象
logger = logging.get_logger(__name__)

# 以下是用于文档的检查点和配置信息
_CHECKPOINT_FOR_DOC = "google-bert/bert-base-uncased"
_CONFIG_FOR_DOC = "BertConfig"

# Token分类任务的文档字符串信息
_CHECKPOINT_FOR_TOKEN_CLASSIFICATION = "dbmdz/bert-large-cased-finetuned-conll03-english"
_TOKEN_CLASS_EXPECTED_OUTPUT = (
    "['O', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'I-LOC', 'O', 'I-LOC', 'I-LOC'] "
)
_TOKEN_CLASS_EXPECTED_LOSS = 0.01

# 问答任务的文档字符串信息
_CHECKPOINT_FOR_QA = "deepset/bert-base-cased-squad2"
_QA_EXPECTED_OUTPUT = "'a nice puppet'"
_QA_EXPECTED_LOSS = 7.41
_QA_TARGET_START_INDEX = 14
_QA_TARGET_END_INDEX = 15

# 序列分类任务的文档字符串信息
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "textattack/bert-base-uncased-yelp-polarity"
_SEQ_CLASS_EXPECTED_OUTPUT = "'LABEL_1'"
_SEQ_CLASS_EXPECTED_LOSS = 0.01

# BERT预训练模型存档列表
BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google-bert/bert-base-uncased",
    "google-bert/bert-large-uncased",
    "google-bert/bert-base-cased",
    "google-bert/bert-large-cased",
    "google-bert/bert-base-multilingual-uncased",
    "google-bert/bert-base-multilingual-cased",
    "google-bert/bert-base-chinese",
    "google-bert/bert-base-german-cased",
    "google-bert/bert-large-uncased-whole-word-masking",
]
    # 定义一个包含多个字符串的列表，每个字符串表示一个预训练的BERT模型的名称
    [
        "google-bert/bert-large-cased-whole-word-masking",
        "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad",
        "google-bert/bert-large-cased-whole-word-masking-finetuned-squad",
        "google-bert/bert-base-cased-finetuned-mrpc",
        "google-bert/bert-base-german-dbmdz-cased",
        "google-bert/bert-base-german-dbmdz-uncased",
        "cl-tohoku/bert-base-japanese",
        "cl-tohoku/bert-base-japanese-whole-word-masking",
        "cl-tohoku/bert-base-japanese-char",
        "cl-tohoku/bert-base-japanese-char-whole-word-masking",
        "TurkuNLP/bert-base-finnish-cased-v1",
        "TurkuNLP/bert-base-finnish-uncased-v1",
        "wietsedv/bert-base-dutch-cased",
        # 查看所有BERT模型，请访问 https://huggingface.co/models?filter=bert
    ]
# 加载 TensorFlow 权重到 PyTorch 模型中
def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re  # 导入正则表达式模块
        import numpy as np  # 导入 NumPy 模块
        import tensorflow as tf  # 导入 TensorFlow 模块
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    
    tf_path = os.path.abspath(tf_checkpoint_path)  # 获取 TensorFlow checkpoint 文件的绝对路径
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")  # 记录日志，显示转换的 TensorFlow checkpoint 路径
    
    # 从 TF 模型中加载权重
    init_vars = tf.train.list_variables(tf_path)  # 获取 TensorFlow checkpoint 中的所有变量名和形状
    names = []
    arrays = []
    
    # 遍历初始化变量，加载变量的值
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")  # 记录日志，显示加载的 TF 权重名和形状
        array = tf.train.load_variable(tf_path, name)  # 加载 TensorFlow checkpoint 中指定变量的值
        names.append(name)
        arrays.append(array)

    # 将加载的 TensorFlow 权重映射到 PyTorch 模型中的相应位置
    for name, array in zip(names, arrays):
        name = name.split("/")
        
        # 跳过不需要加载的变量，如优化器中的动量信息和全局步数等
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")  # 记录日志，显示跳过的变量名
            continue
        
        pointer = model
        
        # 根据变量名将 TensorFlow 权重映射到 PyTorch 模型的对应位置
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
                
            # 根据不同的变量名前缀，映射到 PyTorch 模型的不同部分
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
                    logger.info(f"Skipping {'/'.join(name)}")  # 记录日志，显示跳过的变量名
                    continue
                    
            # 如果变量名包含下划线加数字的格式，则按数字索引进一步定位
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        
        # 如果变量名以 "_embeddings" 结尾，则映射到 PyTorch 模型的权重部分
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)  # 转置数组，用于卷积核权重
        
        # 检查加载的权重形状与 PyTorch 模型对应部分的形状是否匹配
        try:
            if pointer.shape != array.shape:
                raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
        except ValueError as e:
            e.args += (pointer.shape, array.shape)
            raise
        
        logger.info(f"Initialize PyTorch weight {name}")  # 记录日志，显示初始化的 PyTorch 权重名
        pointer.data = torch.from_numpy(array)  # 将 NumPy 数组转换为 PyTorch 张量赋给指针
        
    return model  # 返回加载了 TensorFlow 权重的 PyTorch 模型
    """Construct the embeddings from word, position and token_type embeddings."""

    # 初始化函数，用于构建包含单词、位置和token类型嵌入的模型
    def __init__(self, config):
        super().__init__()
        
        # 创建一个单词嵌入层，用于将单词索引映射到隐藏状态空间
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        
        # 创建一个位置嵌入层，用于将位置索引映射到隐藏状态空间
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # 创建一个token类型嵌入层，用于将token类型索引映射到隐藏状态空间
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 使用LayerNorm来标准化隐藏状态空间中的每个特征向量，以增强模型的训练效果
        # 注意，这里LayerNorm没有采用蛇形命名，是为了兼容TensorFlow模型变量名，以便加载任何TensorFlow检查点文件
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Dropout层，用于在训练过程中随机丢弃部分神经元，以防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # 确定位置嵌入的类型，默认为"absolute"，但可以在配置中指定
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        
        # 注册缓冲区，用于存储位置索引，在序列化时可以导出
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        
        # 注册缓冲区，用于存储token类型索引，在序列化时可以导出
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    # 前向传播函数，定义了模型的正向运算过程
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
        ...
    # 定义函数forward，接受input_ids、inputs_embeds、token_type_ids、position_ids和past_key_values_length作为输入，返回torch.Tensor类型的输出
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        # 如果传入了input_ids，则获取其尺寸作为input_shape
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            # 否则获取inputs_embeds的所有维度尺寸除了最后一个维度
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列的长度seq_length，即input_shape的第二个维度
        seq_length = input_shape[1]

        # 如果未提供position_ids，则从self.position_ids中选择对应序列长度的部分作为position_ids
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # 设置token_type_ids为构造函数中注册的缓冲区，该缓冲区通常为全零，用于在不传递token_type_ids的情况下追踪模型，解决问题＃5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                # 从self.token_type_ids中获取缓冲的token_type_ids，并扩展到与输入形状相匹配
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                # 如果self中没有token_type_ids属性，则创建全零的token_type_ids张量，dtype为torch.long，设备为self.position_ids的设备
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果inputs_embeds为None，则使用self.word_embeddings将input_ids转换为嵌入向量
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        
        # 根据token_type_ids获取对应的token_type_embeddings
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 计算嵌入向量embeddings，将inputs_embeds和token_type_embeddings相加
        embeddings = inputs_embeds + token_type_embeddings
        
        # 如果位置嵌入类型为"absolute"，则从self.position_embeddings获取对应的位置嵌入并加到embeddings中
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        
        # 对embeddings进行LayerNorm处理
        embeddings = self.LayerNorm(embeddings)
        
        # 对LayerNorm后的embeddings进行dropout处理
        embeddings = self.dropout(embeddings)
        
        # 返回处理后的embeddings作为输出
        return embeddings
class BertSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 检查隐藏层大小是否能够被注意力头数整除，或者是否具有嵌入大小属性
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 初始化注意力头数和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化查询、键、值的线性变换层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 初始化 dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        # 设置位置嵌入类型，默认为绝对位置编码
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果位置嵌入类型是相对位置编码之一，则初始化距离嵌入层
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        # 是否为解码器
        self.is_decoder = config.is_decoder

    # 将输入张量重塑为注意力分数的形状
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播函数定义
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,


```    
class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化全连接层、LayerNorm 和 dropout
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数定义
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 全连接层
        hidden_states = self.dense(hidden_states)
        # dropout
        hidden_states = self.dropout(hidden_states)
        # LayerNorm
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    # 初始化方法，接受配置参数和位置嵌入类型，调用父类的初始化方法
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 创建 BertSelfAttention 对象，并保存在 self.self 属性中
        self.self = BertSelfAttention(config, position_embedding_type=position_embedding_type)
        # 创建 BertSelfOutput 对象，并保存在 self.output 属性中
        self.output = BertSelfOutput(config)
        # 初始化一个空集合，用于存储被剪枝的注意力头索引
        self.pruned_heads = set()

    # 剪枝注意力头的方法
    def prune_heads(self, heads):
        # 如果传入的头索引集合为空，直接返回
        if len(heads) == 0:
            return
        # 调用函数找到可剪枝的头部索引和对应的位置索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 对 BertSelfAttention 中的 query、key、value 线性层进行剪枝操作
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        # 对 BertSelfOutput 中的 dense 线性层进行剪枝操作
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新剩余的注意力头数目和总头部大小
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        # 将剪枝的头部索引添加到 pruned_heads 集合中
        self.pruned_heads = self.pruned_heads.union(heads)

    # 前向传播方法，接收多个输入张量和可选参数，返回一个张量元组
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
        # 调用 BertSelfAttention 的 forward 方法进行自注意力计算
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 将自注意力输出结果和原始隐藏状态传入 BertSelfOutput 对象进行输出计算
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出注意力权重，则将注意力权重添加到输出中
        outputs = (attention_output,) + self_outputs[1:]  # 如果需要，添加注意力权重
        return outputs
# BertLayer 类的定义，继承自 nn.Module，表示这是一个神经网络模块
class BertLayer(nn.Module):
    # 初始化方法，接受一个 config 参数
    def __init__(self, config):
        super().__init__()  # 调用父类 nn.Module 的初始化方法
        # 设置用于前向传播中的块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度的维度，通常为 1
        self.seq_len_dim = 1
        # BertAttention 类的实例化，使用 config 参数进行初始化
        self.attention = BertAttention(config)
        # 是否作为解码器使用的标志
        self.is_decoder = config.is_decoder
        # 是否添加跨注意力机制的标志
        self.add_cross_attention = config.add_cross_attention
        # 如果添加了跨注意力机制
        if self.add_cross_attention:
            # 如果不是解码器，则抛出异常
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 使用绝对位置编码类型，实例化 BertAttention 类
            self.crossattention = BertAttention(config, position_embedding_type="absolute")
        # BertIntermediate 类的实例化，使用 config 参数进行初始化
        self.intermediate = BertIntermediate(config)
        # BertOutput 类的实例化，使用 config 参数进行初始化
        self.output = BertOutput(config)

    # 前向传播方法，接收多个输入参数并返回一个 torch.Tensor 对象
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        ):
        # 使用 BertAttention 类处理隐藏状态，根据需要传入不同的参数
        hidden_states = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 如果添加了跨注意力机制，使用 crossattention 处理隐藏状态
        if self.add_cross_attention:
            hidden_states = self.crossattention(
                hidden_states,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )
        # 使用 BertIntermediate 类处理中间状态，传入隐藏状态
        hidden_states = self.intermediate(hidden_states)
        # 使用 BertOutput 类处理输出，传入中间状态和输入张量
        hidden_states = self.output(hidden_states, input_tensor=hidden_states)
        # 返回处理后的隐藏状态张量
        return hidden_states
    ) -> Tuple[torch.Tensor]:  
        # 函数签名说明该函数返回一个包含torch.Tensor的元组
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        # 如果存在过去的key/value缓存，只选择其前两个元素，否则为None
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 调用self.attention方法进行自注意力计算
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # 获取自注意力计算的输出
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        # 如果是解码器，最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            # 去除第一个和最后一个元素，因为它们是self_attention_outputs中的self-attention结果和cross-attention结果
            outputs = self_attention_outputs[1:-1]
            # 最后一个元素是当前时刻的key/value缓存
            present_key_value = self_attention_outputs[-1]
        else:
            # 如果不是解码器，则输出包括self-attention结果（如果输出注意力权重的话）
            outputs = self_attention_outputs[1:]  # 如果输出注意力权重，添加self-attention
        

        cross_attn_present_key_value = None
        # 如果是解码器并且有编码器的隐藏状态
        if self.is_decoder and encoder_hidden_states is not None:
            # 如果没有crossattention属性，抛出值错误
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            # 如果存在过去的key/value缓存，选择其倒数第二个和最后一个元素，否则为None
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 调用self.crossattention方法进行跨注意力计算
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            # 获取跨注意力计算的输出
            attention_output = cross_attention_outputs[0]
            # 添加cross-attention结果到outputs中，去除第一个和最后一个元素
            outputs = outputs + cross_attention_outputs[1:-1]

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            # 将cross-attn缓存添加到present_key_value元组的倒数第二个和最后一个位置
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 应用分块技术对attention_output应用self.feed_forward_chunk方法
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        # 将layer_output作为第一个元素添加到outputs元组中
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        # 如果是解码器，将attn key/values作为最后一个输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        # 返回outputs作为函数的输出结果
        return outputs

    def feed_forward_chunk(self, attention_output):
        # 将attention_output作为输入，首先应用self.intermediate方法
        intermediate_output = self.intermediate(attention_output)
        # 然后将中间输出作为输入，应用self.output方法
        layer_output = self.output(intermediate_output, attention_output)
        # 返回最终的层输出结果
        return layer_output
# 定义一个名为 BertEncoder 的类，继承自 nn.Module 类
class BertEncoder(nn.Module):
    # 初始化方法，接收一个 config 参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 将传入的 config 参数保存在实例变量 self.config 中
        self.config = config
        # 创建一个 nn.ModuleList 对象 self.layer，其中包含 config.num_hidden_layers 个 BertLayer 对象
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        # 初始化一个标志变量 gradient_checkpointing，并设置为 False
        self.gradient_checkpointing = False

    # 前向传播方法定义
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量
        attention_mask: Optional[torch.FloatTensor] = None,  # 可选的注意力掩码张量
        head_mask: Optional[torch.FloatTensor] = None,  # 可选的头部掩码张量
        encoder_hidden_states: Optional[torch.FloatTensor] = None,  # 可选的编码器隐藏状态张量
        encoder_attention_mask: Optional[torch.FloatTensor] = None,  # 可选的编码器注意力掩码张量
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 可选的过去的键-值对
        use_cache: Optional[bool] = None,  # 可选的缓存标志
        output_attentions: Optional[bool] = False,  # 可选的输出注意力张量标志，默认为 False
        output_hidden_states: Optional[bool] = False,  # 可选的输出隐藏状态标志，默认为 False
        return_dict: Optional[bool] = True,  # 可选的返回字典标志，默认为 True
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        # 初始化空的所有隐藏状态列表，如果不需要输出隐藏状态则为 None
        all_hidden_states = () if output_hidden_states else None
        # 初始化空的所有自注意力权重列表，如果不需要输出注意力权重则为 None
        all_self_attentions = () if output_attentions else None
        # 初始化空的所有交叉注意力权重列表，如果不需要输出交叉注意力权重或模型不支持则为 None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 如果启用了梯度检查点且处于训练模式下
        if self.gradient_checkpointing and self.training:
            # 如果设置了 use_cache=True，发出警告并设置 use_cache=False
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # 如果需要使用缓存，初始化下一个解码器缓存为一个空元组
        next_decoder_cache = () if use_cache else None
        # 遍历每一层解码器层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，将当前隐藏状态加入到所有隐藏状态列表中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码，如果未提供则为 None
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 获取先前的键值对，如果未提供则为 None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 如果启用了梯度检查点且处于训练模式下
            if self.gradient_checkpointing and self.training:
                # 使用梯度检查点函数计算当前层的输出
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
                # 否则直接调用当前层模块计算输出
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
            # 如果使用缓存，将当前层的缓存状态添加到下一个解码器缓存中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 如果需要输出注意力权重，将当前层的自注意力权重加入到所有自注意力列表中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果模型支持并需要输出交叉注意力，将当前层的交叉注意力加入到所有交叉注意力列表中
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果需要输出隐藏状态，将最终隐藏状态加入到所有隐藏状态列表中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要返回字典形式的输出，返回一个元组
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
        # 否则返回 BaseModelOutputWithPastAndCrossAttentions 对象
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # 创建一个全连接层，输入输出大小相同
        self.activation = nn.Tanh()  # 创建一个tanh激活函数实例

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 我们通过简单地取对应于第一个标记的隐藏状态来“汇聚”模型。
        first_token_tensor = hidden_states[:, 0]  # 选择隐藏状态张量的第一个标记
        pooled_output = self.dense(first_token_tensor)  # 将第一个标记的隐藏状态传入全连接层
        pooled_output = self.activation(pooled_output)  # 使用tanh激活函数进行激活
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # 创建一个全连接层，输入输出大小相同
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]  # 如果配置中的隐藏激活函数是字符串，则使用预定义的激活函数映射
        else:
            self.transform_act_fn = config.hidden_act  # 否则直接使用配置中的激活函数
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # 创建一个LayerNorm层

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)  # 将隐藏状态传入全连接层
        hidden_states = self.transform_act_fn(hidden_states)  # 应用预定义或配置中的激活函数
        hidden_states = self.LayerNorm(hidden_states)  # 应用LayerNorm
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)  # 创建一个预测头变换器

        # 输出权重与输入嵌入相同，但每个标记都有一个仅输出的偏置项。
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)  # 创建一个线性层，无偏置

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))  # 创建一个偏置参数

        # 需要一个链接来确保偏置随 `resize_token_embeddings` 正确调整大小
        self.decoder.bias = self.bias  # 将偏置参数链接到解码器的偏置

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)  # 应用变换器到隐藏状态
        hidden_states = self.decoder(hidden_states)  # 应用解码器到变换后的隐藏状态
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)  # 创建一个MLM头部预测器

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)  # 使用预测器进行序列输出的预测
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)  # 创建一个线性层，用于NSP头部

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)  # 计算汇聚输出的关系分数
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)  # 创建一个预测头部
        self.seq_relationship = nn.Linear(config.hidden_size, 2)  # 创建一个线性层，用于NSP头部
    # 定义一个方法 `forward`，接收 `sequence_output` 和 `pooled_output` 作为参数
    def forward(self, sequence_output, pooled_output):
        # 调用 `self.predictions` 方法，传入 `sequence_output` 参数，返回预测分数
        prediction_scores = self.predictions(sequence_output)
        # 调用 `self.seq_relationship` 方法，传入 `pooled_output` 参数，返回序列关系分数
        seq_relationship_score = self.seq_relationship(pooled_output)
        # 返回预测分数和序列关系分数作为结果
        return prediction_scores, seq_relationship_score
# 定义一个名为 BertPreTrainedModel 的类，继承自 PreTrainedModel 类
class BertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定配置类为 BertConfig
    config_class = BertConfig
    # 指定加载 TensorFlow 权重的函数为 load_tf_weights_in_bert
    load_tf_weights = load_tf_weights_in_bert
    # 指定基础模型的前缀为 "bert"
    base_model_prefix = "bert"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    # 初始化权重的函数，根据模块类型不同进行初始化
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # 对线性层的权重使用正态分布初始化，标准差为 self.config.initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置项，则将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 对嵌入层的权重使用正态分布初始化，标准差为 self.config.initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果指定了填充索引，则将对应的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 对 LayerNorm 层的偏置项初始化为零
            module.bias.data.zero_()
            # 对 LayerNorm 层的权重初始化为 1.0
            module.weight.data.fill_(1.0)


# 使用 dataclass 装饰器定义 BertForPreTrainingOutput 类，继承自 ModelOutput 类
@dataclass
class BertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`BertForPreTraining`].

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

    # 可选项：损失，当提供 labels 时返回，torch.FloatTensor，形状为 (1,)
    loss: Optional[torch.FloatTensor] = None
    # 预测 logits：语言建模头部的预测分数，形状为 (batch_size, sequence_length, config.vocab_size)
    prediction_logits: torch.FloatTensor = None
    # 序列关系 logits：下一个序列预测头部的预测分数，形状为 (batch_size, 2)
    seq_relationship_logits: torch.FloatTensor = None
    # 定义一个可选的变量 `hidden_states`，类型为包含单个 `torch.FloatTensor` 的元组或空值
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义一个可选的变量 `attentions`，类型为包含单个 `torch.FloatTensor` 的元组或空值
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# BERT_START_DOCSTRING 是一个原始的文档字符串，用于描述 BERT 模型的基本信息和用法。
# 这个模型继承自 `PreTrainedModel`，可以查看其父类文档了解通用方法，如下载、保存、调整输入嵌入大小、剪枝等。
# 同时，这个模型也是 PyTorch 的 `torch.nn.Module` 子类，可以像普通的 PyTorch 模块一样使用，相关用法和行为请参考 PyTorch 文档。

BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列的标记索引，对应词汇表中的位置。

            # 可以使用 [`AutoTokenizer`] 获取这些索引。参见 [`PreTrainedTokenizer.encode`] 和
            # [`PreTrainedTokenizer.__call__`] 获取详细信息。

            # [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 遮盖掩码，用于在填充的标记索引上避免执行注意力操作。遮盖值在 `[0, 1]` 之间：

            # - 1 表示对应的标记是 **未遮盖的**，
            # - 0 表示对应的标记是 **遮盖的**。

            # [什么是注意力遮盖？](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 分段标记索引，指示输入的第一部分和第二部分。索引选择在 `[0, 1]`：

            # - 0 对应 *句子 A* 的标记，
            # - 1 对应 *句子 B* 的标记。

            # [什么是标记类型 ID？](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 输入序列中每个标记的位置索引，在位置嵌入中使用。索引选择范围为 `[0, config.max_position_embeddings - 1]`。

            # [什么是位置 ID？](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于屏蔽自注意力模块中选定头部的掩码。掩码值在 `[0, 1]` 之间：

            # - 1 表示头部 **未被屏蔽**，
            # - 0 表示头部 **被屏蔽**。
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选参数，可以直接传递嵌入表示，而不是传递 `input_ids`。如果希望更好地控制如何将 `input_ids` 索引转换为关联向量，
            # 这种方法非常有用，而不使用模型的内部嵌入查找矩阵。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。返回的张量中有关于 `attentions` 的更多详细信息。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。返回的张量中有关于 `hidden_states` 的更多详细信息。
        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是简单的元组。
"""
@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
class BertModel(BertPreTrainedModel):
    """
    BertModel类，继承自BertPreTrainedModel，表示一个Bert模型，可以输出原始的隐藏状态，没有特定的输出头部。

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    
    该模型可以作为编码器（仅具有自注意力）或解码器，如果是解码器，会在自注意力层之间添加交叉注意力层，遵循[Attention is
    all you need](https://arxiv.org/abs/1706.03762) 中描述的架构，作者为Ashish Vaswani, Noam Shazeer, Niki Parmar,
    Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser 和 Illia Polosukhin。

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    
    要作为解码器使用，模型需要用配置中的`is_decoder`参数初始化为`True`。在Seq2Seq模型中使用时，模型需要用`is_decoder`和
    `add_cross_attention`两个参数初始化为`True`；然后预期在前向传递中作为输入的`encoder_hidden_states`。
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        # 初始化嵌入层
        self.embeddings = BertEmbeddings(config)
        # 初始化编码器层
        self.encoder = BertEncoder(config)

        # 如果需要添加汇聚层，则初始化汇聚层；否则为None
        self.pooler = BertPooler(config) if add_pooling_layer else None

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回嵌入层中的词嵌入
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        # 设置嵌入层的词嵌入为指定的值
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 对模型的注意力头进行修剪
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
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
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        前向传递函数，接受多个参数用于构建Bert模型的输入和控制输出格式。

        Args:
            input_ids (Optional[torch.Tensor], optional): 输入的token ID张量，默认为None。
            attention_mask (Optional[torch.Tensor], optional): 注意力掩码张量，默认为None。
            token_type_ids (Optional[torch.Tensor], optional): 分段类型ID张量，默认为None。
            position_ids (Optional[torch.Tensor], optional): 位置ID张量，默认为None。
            head_mask (Optional[torch.Tensor], optional): 头部掩码张量，默认为None。
            inputs_embeds (Optional[torch.Tensor], optional): 嵌入输入张量，默认为None。
            encoder_hidden_states (Optional[torch.Tensor], optional): 编码器隐藏状态张量，默认为None。
            encoder_attention_mask (Optional[torch.Tensor], optional): 编码器的注意力掩码张量，默认为None。
            past_key_values (Optional[List[torch.FloatTensor]], optional): 过去的键值对列表，默认为None。
            use_cache (Optional[bool], optional): 是否使用缓存，默认为None。
            output_attentions (Optional[bool], optional): 是否输出注意力，默认为None。
            output_hidden_states (Optional[bool], optional): 是否输出隐藏状态，默认为None。
            return_dict (Optional[bool], optional): 是否返回字典格式的输出，默认为None。

        Returns:
            根据参数不同可能返回不同形式的输出，详见具体参数说明。
        """
        # 实际的前向传递逻辑由具体的Bert模型实现，这里是接口定义和文档说明
        pass
    Bert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a `next
    sentence prediction (classification)` head.

# 描述BERT模型的结构，包括两个预训练阶段添加的顶部头部：一个用于掩码语言建模，另一个用于下一句预测分类。


    """,
    BERT_START_DOCSTRING,

# 添加额外的文档字符串注释，并引用了 `BERT_START_DOCSTRING` 变量。
@add_start_docstrings(
    """Bert Model with a `masked language modeling` head on top for MLM fine-tuning.""", BERT_START_DOCSTRING
)
class BertForMaskedLM(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = ["position_ids"]

    def __init__(self, config):
        super().__init__(config)

        if not config.is_decoder:
            logger.warning("If you want to use `BertForMaskedLM` as a standalone, add `is_decoder=True.`")

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        """
        Retrieve the output embedding layer for predictions.
        """
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        """
        Set new output embeddings for the prediction layer.
        """
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
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
        next_sentence_label: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Forward pass for BertForMaskedLM model.
        """
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入的token IDs序列，可选的Tensor类型
        attention_mask: Optional[torch.Tensor] = None,  # 注意力遮罩，可选的Tensor类型
        token_type_ids: Optional[torch.Tensor] = None,  # token类型IDs，可选的Tensor类型
        position_ids: Optional[torch.Tensor] = None,  # 位置IDs，可选的Tensor类型
        head_mask: Optional[torch.Tensor] = None,  # 头部遮罩，可选的Tensor类型
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入的嵌入向量，可选的Tensor类型
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 编码器隐藏状态，可选的Tensor类型
        encoder_attention_mask: Optional[torch.Tensor] = None,  # 编码器注意力遮罩，可选的Tensor类型
        labels: Optional[torch.Tensor] = None,  # 标签，可选的Tensor类型
        past_key_values: Optional[List[torch.Tensor]] = None,  # 过去的键值，可选的Tensor列表类型
        use_cache: Optional[bool] = None,  # 是否使用缓存，可选的布尔类型
        output_attentions: Optional[bool] = None,  # 是否输出注意力，可选的布尔类型
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选的布尔类型
        return_dict: Optional[bool] = None,  # 是否返回字典，可选的布尔类型
    ):
        # 此方法定义了模型的前向传播逻辑，接收多种输入参数，并根据需要返回不同的输出

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=True, **model_kwargs
    ):
        input_shape = input_ids.shape  # 获取输入IDs的形状

        # 如果未提供注意力遮罩，则创建全为1的遮罩，保证所有token被处理
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # 如果传入了过去的键值（用于生成），则截断输入的token IDs
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法可能只传递最后一个输入ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认行为：只保留最后一个ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        # 返回准备好的输入字典，用于生成（或解码）阶段
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # 重排过去的键值，以适应beam搜索的索引顺序
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
# 为带有顶部 `语言建模` 头部的 Bert 模型添加文档字符串
@add_start_docstrings("""Bert Model with a `language modeling` head on top.""", BERT_START_DOCSTRING)
class BertForMaskedLM(BertPreTrainedModel):
    # 定义绑定权重的键名列表
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)

        # 如果配置标记为解码器，则发出警告
        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 创建 BertModel 实例，不包含池化层
        self.bert = BertModel(config, add_pooling_layer=False)
        # 创建 BertOnlyMLMHead 实例
        self.cls = BertOnlyMLMHead(config)

        # 执行额外的初始化操作，如权重初始化和最终处理
        self.post_init()

    # 返回输出嵌入的函数
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出嵌入的函数，接受新的嵌入作为参数
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 前向传播函数，接受多个输入参数，根据文档字符串描述了各参数的含义
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'paris'",
        expected_loss=0.88,
    )
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
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        # 确定是否使用返回字典形式
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给BERT模型，获取输出
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

        # 从BERT模型的输出中获取序列输出
        sequence_output = outputs[0]
        # 通过分类层获取预测得分
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        # 如果提供了标签，则计算masked language modeling loss
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # 交叉熵损失函数，用于计算损失
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果不要求返回字典形式的输出
        if not return_dict:
            # 构造输出元组
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 返回MaskedLMOutput对象，封装了loss、logits、hidden_states和attentions
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        # 添加一个虚拟token，用于生成
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        # 修改attention_mask，在末尾添加一个全零列
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        # 创建全为pad_token_id的虚拟token
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        # 将虚拟token拼接到input_ids末尾
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        # 返回包含输入信息的字典
        return {"input_ids": input_ids, "attention_mask": attention_mask}
# 定义一个带有“下一个句子预测（分类）”头部的 Bert 模型。
# 使用 BERT_START_DOCSTRING 和 BERT_START_DOCSTRING 描述模型的基本信息。
@add_start_docstrings(
    """Bert Model with a `next sentence prediction (classification)` head on top.""",
    BERT_START_DOCSTRING,
)
class BertForNextSentencePrediction(BertPreTrainedModel):
    
    # 初始化方法，接受一个配置对象 config 作为参数。
    def __init__(self, config):
        super().__init__(config)
        
        # 初始化 BertModel，加载预训练的 BERT 模型。
        self.bert = BertModel(config)
        
        # 初始化 BertOnlyNSPHead，用于执行仅包含 NSP（Next Sentence Prediction）的任务。
        self.cls = BertOnlyNSPHead(config)

        # 执行额外的初始化步骤和最终处理。
        self.post_init()

    # 前向传播函数，接受多个输入参数，包括 input_ids、attention_mask 等。
    # 使用 BERT_INPUTS_DOCSTRING 描述输入的详细信息，格式为 batch_size, sequence_length。
    # 使用 NextSentencePredictorOutput 类型描述输出，配置类为 _CONFIG_FOR_DOC。
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=NextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
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
        **kwargs,
        ) -> Union[Tuple[torch.Tensor], NextSentencePredictorOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see `input_ids` docstring). Indices should be in `[0, 1]`:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.

        Returns:
            Depending on `return_dict`:
            - If `return_dict` is `False`, returns a tuple with `next_sentence_loss` (if computed) and `seq_relationship_scores` and possibly additional hidden states.
            - If `return_dict` is `True`, returns a `NextSentencePredictorOutput` object containing `loss`, `logits`, `hidden_states`, and `attentions`.

        Example:

        ```
        >>> from transformers import AutoTokenizer, BertForNextSentencePrediction
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        >>> model = BertForNextSentencePrediction.from_pretrained("google-bert/bert-base-uncased")

        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
        >>> encoding = tokenizer(prompt, next_sentence, return_tensors="pt")

        >>> outputs = model(**encoding, labels=torch.LongTensor([1]))
        >>> logits = outputs.logits
        >>> assert logits[0, 0] < logits[0, 1]  # next sentence was random
        ```
        """

        if "next_sentence_label" in kwargs:
            # 发出警告，`next_sentence_label` 参数已废弃，建议使用 `labels` 参数代替
            warnings.warn(
                "The `next_sentence_label` argument is deprecated and will be removed in a future version, use"
                " `labels` instead.",
                FutureWarning,
            )
            # 将 `next_sentence_label` 的值赋给 `labels` 变量，并从 `kwargs` 中移除该参数
            labels = kwargs.pop("next_sentence_label")

        # 根据 `return_dict` 是否为 `None` 确定是否使用配置中的返回字典设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 BERT 模型处理输入数据
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

        # 从 BERT 输出中获取池化后的输出
        pooled_output = outputs[1]

        # 使用分类层处理池化输出，得到序列关系的分数
        seq_relationship_scores = self.cls(pooled_output)

        # 初始化下一个句子预测的损失为 None
        next_sentence_loss = None
        # 如果提供了 `labels` 参数，则计算下一个句子预测的交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))

        # 如果 `return_dict` 为 False，则返回一个包含 `next_sentence_loss` 和可能的其他隐藏状态的元组
        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

        # 如果 `return_dict` 为 True，则返回一个 `NextSentencePredictorOutput` 对象
        return NextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用装饰器添加开始文档字符串，描述了这是一个在Bert模型基础上增加了顶部序列分类/回归头的转换器类，例如用于GLUE任务。
@add_start_docstrings(
    """
    Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    BERT_START_DOCSTRING,  # 引用了BERT的起始文档字符串
)
class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels  # 从配置中获取标签数量
        self.config = config

        self.bert = BertModel(config)  # 初始化BERT模型
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)  # 初始化dropout层
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)  # 初始化线性分类器层

        # 初始化权重并进行最终处理
        self.post_init()

    # 使用装饰器添加模型前向传播的开始文档字符串，描述了输入参数的预期形状
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 使用装饰器添加代码示例的文档字符串，展示了模型的检查点、输出类型、配置类、预期输出和损失
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
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
        # 如果 return_dict 不为 None，则使用它；否则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给 BERT 模型进行前向传播
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

        # 获取 BERT 输出中的池化输出（通常是 CLS token 的输出）
        pooled_output = outputs[1]

        # 对池化输出应用 dropout
        pooled_output = self.dropout(pooled_output)
        
        # 将 dropout 后的输出传递给分类器，得到 logits
        logits = self.classifier(pooled_output)

        # 初始化损失值
        loss = None
        
        # 如果 labels 不为 None，则计算损失
        if labels is not None:
            # 根据配置决定问题类型，如果未指定，则根据 num_labels 判断是回归还是分类问题
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
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        
        # 如果 return_dict 为 False，则返回 logits 和 BERT 模型的隐藏状态
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则构造 SequenceClassifierOutput 对象返回
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
"""
Bert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
softmax) e.g. for RocStories/SWAG tasks.
"""
# 导入必要的库和模块
@add_start_docstrings(
    """
    Bert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    BERT_START_DOCSTRING,
)
# 定义 BertForMultipleChoice 类，继承自 BertPreTrainedModel
class BertForMultipleChoice(BertPreTrainedModel):
    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 加载预训练的 BERT 模型
        self.bert = BertModel(config)
        # 设置分类器的 dropout 概率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 定义 dropout 层
        self.dropout = nn.Dropout(classifier_dropout)
        # 定义分类器线性层
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
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
        # 执行前向传播，处理输入参数，返回模型输出

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



        # 执行前向传播，处理输入参数，返回模型输出
        ...
    ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 如果 return_dict 为 None，则使用模型配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 获取输入张量 input_ids 的第二维度大小作为选择数
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 将 input_ids 重新视图为二维张量，第一维为 -1，第二维与原始最后一维相同
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        # 将 attention_mask 重新视图为二维张量，第一维为 -1，第二维与原始最后一维相同
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # 将 token_type_ids 重新视图为二维张量，第一维为 -1，第二维与原始最后一维相同
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        # 将 position_ids 重新视图为二维张量，第一维为 -1，第二维与原始最后一维相同
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        # 将 inputs_embeds 重新视图为三维张量，第一维为 -1，第二维和第三维与原始相同
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 将输入张量传递给 BERT 模型进行处理，并获取模型的输出
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

        # 获取汇总输出（pooled_output），这通常是 BERT 模型的第二个输出
        pooled_output = outputs[1]

        # 对汇总输出应用 dropout
        pooled_output = self.dropout(pooled_output)
        # 使用分类器（通常是一个线性层）对汇总输出进行分类预测
        logits = self.classifier(pooled_output)
        # 重新调整 logits 的形状，使其匹配 num_choices 的维度
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        # 如果提供了标签，则计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 如果不使用 return_dict，按照非字典格式返回输出
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果使用 return_dict，按照字典格式返回 MultipleChoiceModelOutput
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 给 BertForTokenClassification 类添加文档字符串，描述其作用和用途，特别是用于命名实体识别 (NER) 等任务的 token 分类模型
@add_start_docstrings(
    """
    Bert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    BERT_START_DOCSTRING,
)
class BertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # 初始化 Bert 模型，不添加池化层
        self.bert = BertModel(config, add_pooling_layer=False)
        # 根据配置设置分类器的 dropout
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        # 线性层，将隐藏状态输出映射到标签数量的空间
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 给 forward 方法添加文档字符串，描述其输入和输出，使用了 BERT_INPUTS_DOCSTRING 中的说明
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 添加代码示例文档字符串，显示了如何从检查点加载模型并进行 token 分类
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_TOKEN_CLASSIFICATION,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_TOKEN_CLASS_EXPECTED_OUTPUT,
        expected_loss=_TOKEN_CLASS_EXPECTED_LOSS,
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
        ):
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 如果 return_dict 不为 None，则使用其值；否则使用 self.config.use_return_dict 的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 BERT 模型处理输入数据，并返回其输出结果
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

        # 获取 BERT 模型的输出序列表示
        sequence_output = outputs[0]

        # 对输出序列进行 dropout 处理
        sequence_output = self.dropout(sequence_output)
        # 将 dropout 后的序列输出结果输入分类器，得到分类器的 logits
        logits = self.classifier(sequence_output)

        # 初始化损失为 None
        loss = None
        # 如果提供了标签（labels），则计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果 return_dict 为 False，则返回额外的输出信息
        if not return_dict:
            output = (logits,) + outputs[2:]  # 保留分类器 logits 和额外的输出信息
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则使用 TokenClassifierOutput 类封装输出结果
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,  # 返回所有隐藏状态
            attentions=outputs.attentions,        # 返回所有注意力权重
        )
# 定义一个 Bert 模型，用于提取式问答任务（如 SQuAD），在隐藏状态的输出上方添加一个线性层，用于计算“起始位置对数”和“结束位置对数”。
@add_start_docstrings(
    """
    Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    BERT_START_DOCSTRING,
)
class BertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 设置模型的标签数目
        self.num_labels = config.num_labels

        # 初始化 Bert 模型，不包含池化层
        self.bert = BertModel(config, add_pooling_layer=False)
        # 线性层，用于输出问题答案的起始位置和结束位置的对数概率
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_QA,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        qa_target_start_index=_QA_TARGET_START_INDEX,
        qa_target_end_index=_QA_TARGET_END_INDEX,
        expected_output=_QA_EXPECTED_OUTPUT,
        expected_loss=_QA_EXPECTED_LOSS,
    )
    # 前向传播方法，接受多种输入参数，计算模型的输出
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
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
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
        # 初始化是否返回字典形式的输出，默认为模型配置中的设定
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 BERT 模型处理输入数据，获取模型的输出
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

        # 从 BERT 输出中获取序列输出
        sequence_output = outputs[0]

        # 将序列输出传入问答模型的输出层，得到起始位置和结束位置的 logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 如果在多 GPU 上运行，需要添加维度以适应多 GPU 并行计算
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 将超出模型输入长度的位置设置为模型最大输入长度，防止超出范围
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 定义交叉熵损失函数，忽略指定的索引
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            # 如果不返回字典形式的输出，返回起始位置 logits 和结束位置 logits
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 返回字典形式的输出，包括损失值、起始位置 logits、结束位置 logits、隐藏状态和注意力权重
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```