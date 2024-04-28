# `.\transformers\models\bert\modeling_bert.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 导入必要的库和模块
import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 从 Hugging Face 库中导入相关模块和类
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

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的预训练模型名称和配置
_CHECKPOINT_FOR_DOC = "bert-base-uncased"
_CONFIG_FOR_DOC = "BertConfig"

# 用于标记分类任务的预训练模型和相关输出
_CHECKPOINT_FOR_TOKEN_CLASSIFICATION = "dbmdz/bert-large-cased-finetuned-conll03-english"
_TOKEN_CLASS_EXPECTED_OUTPUT = (
    "['O', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'I-LOC', 'O', 'I-LOC', 'I-LOC'] "
)
_TOKEN_CLASS_EXPECTED_LOSS = 0.01

# 用于问答任务的预训练模型和相关输出
_CHECKPOINT_FOR_QA = "deepset/bert-base-cased-squad2"
_QA_EXPECTED_OUTPUT = "'a nice puppet'"
_QA_EXPECTED_LOSS = 7.41
_QA_TARGET_START_INDEX = 14
_QA_TARGET_END_INDEX = 15

# 用于序列分类任务的预训练模型和相关输出
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "textattack/bert-base-uncased-yelp-polarity"
_SEQ_CLASS_EXPECTED_OUTPUT = "'LABEL_1'"
_SEQ_CLASS_EXPECTED_LOSS = 0.01

# 列出可用的预训练模型存档列表
BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bert-base-uncased",
    "bert-large-uncased",
    "bert-base-cased",
    "bert-large-cased",
    "bert-base-multilingual-uncased",
    "bert-base-multilingual-cased",
    "bert-base-chinese",
    "bert-base-german-cased",
    "bert-large-uncased-whole-word-masking",
    "bert-large-cased-whole-word-masking",
    "bert-large-uncased-whole-word-masking-finetuned-squad",
]
    # BERT 模型名称列表，包括各种不同语种和预训练方式的模型
        "bert-large-cased-whole-word-masking-finetuned-squad",  # 英文大写，采用了整词遮蔽技术，在 SQuAD 数据集上微调过的 BERT 模型
        "bert-base-cased-finetuned-mrpc",  # 英文大写，采用了整词遮蔽技术，在 MRPC 数据集上微调过的 BERT 模型
        "bert-base-german-dbmdz-cased",  # 德文，来自 DBMDZ 数据集，采用了大小写，cased 表示
        "bert-base-german-dbmdz-uncased",  # 德文，来自 DBMDZ 数据集，不区分大小写，uncased 表示
        "cl-tohoku/bert-base-japanese",  # 日文，基于 cl-tohoku 数据集的 BERT 模型
        "cl-tohoku/bert-base-japanese-whole-word-masking",  # 日文，基于 cl-tohoku 数据集的 BERT 模型，采用了整词遮蔽技术
        "cl-tohoku/bert-base-japanese-char",  # 日文，基于 cl-tohoku 数据集的 BERT 模型，按字符级别
        "cl-tohoku/bert-base-japanese-char-whole-word-masking",  # 日文，基于 cl-tohoku 数据集的 BERT 模型，按字符级别，采用了整词遮蔽技术
        "TurkuNLP/bert-base-finnish-cased-v1",  # 芬兰语，来自 TurkuNLP 数据集，cased 表示
        "TurkuNLP/bert-base-finnish-uncased-v1",  # 芬兰语，来自 TurkuNLP 数据集，uncased 表示
        "wietsedv/bert-base-dutch-cased",  # 荷兰语，来自 wietsedv 数据集，cased 表示
        # 查看所有 BERT 模型请访问 https://huggingface.co/models?filter=bert
# 导入所需模块和库
try:
    import re  # 导入正则表达式模块，用于字符串匹配
    import numpy as np  # 导入 NumPy 库，用于科学计算
    import tensorflow as tf  # 导入 TensorFlow 库，用于加载 TensorFlow 模型
except ImportError:
    logger.error(
        "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
        "https://www.tensorflow.org/install/ for installation instructions."
    )
    raise

# 获取 TensorFlow 模型的绝对路径
tf_path = os.path.abspath(tf_checkpoint_path)
logger.info(f"Converting TensorFlow checkpoint from {tf_path}")

# 从 TensorFlow 模型中加载权重
init_vars = tf.train.list_variables(tf_path)  # 获取 TensorFlow 模型中的变量列表
names = []  # 存储变量名称的列表
arrays = []  # 存储变量值的列表
for name, shape in init_vars:
    logger.info(f"Loading TF weight {name} with shape {shape}")
    array = tf.train.load_variable(tf_path, name)  # 加载 TensorFlow 模型中的变量值
    names.append(name)  # 将变量名称添加到列表
    arrays.append(array)  # 将变量值添加到列表

# 将 TensorFlow 模型的权重转换为 PyTorch 模型的权重
for name, array in zip(names, arrays):
    name = name.split("/")  # 根据斜杠分割变量名称
    # 跳过不需要加载的变量，如优化器中的变量
    if any(
        n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
        for n in name
    ):
        logger.info(f"Skipping {'/'.join(name)}")
        continue
    pointer = model  # 初始化指针为 PyTorch 模型
    for m_name in name:
        # 如果变量名称是以下划线和数字结尾的，如 embedding_1，则根据数字进行切分
        if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
            scope_names = re.split(r"_(\d+)", m_name)
        else:
            scope_names = [m_name]
        # 根据变量名称的不同部分加载对应的 PyTorch 模型权重
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
        # 如果变量名称包含多个部分，则根据索引加载对应的 PyTorch 模型权重
        if len(scope_names) >= 2:
            num = int(scope_names[1])
            pointer = pointer[num]
    if m_name[-11:] == "_embeddings":
        pointer = getattr(pointer, "weight")
    elif m_name == "kernel":
        array = np.transpose(array)  # 转置卷积核权重的数组
    try:
        if pointer.shape != array.shape:
            raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
    except ValueError as e:
        e.args += (pointer.shape, array.shape)
        raise
    logger.info(f"Initialize PyTorch weight {name}")  # 初始化 PyTorch 模型权重
    pointer.data = torch.from_numpy(array)  # 将 NumPy 数组转换为 PyTorch 张量赋值给模型权重
return model  # 返回加载了 TensorFlow 权重的 PyTorch 模型
    """Construct the embeddings from word, position and token_type embeddings."""

    # 定义构造函数，初始化模型参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建词嵌入层，vocab_size 表示词汇表大小，hidden_size 表示隐藏单元大小，padding_idx 表示填充索引
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 创建位置嵌入层，max_position_embeddings 表示最大位置嵌入大小
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 创建标记类型嵌入层，type_vocab_size 表示标记类型的数量
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # LayerNorm 名称未使用蛇形命名法，以与 TensorFlow 模型变量名保持一致，并能够加载任何 TensorFlow 检查点文件
        # 初始化 LayerNorm 层，hidden_size 表示隐藏单元大小，eps 表示 LayerNorm 层的 epsilon 值
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化 Dropout 层，hidden_dropout_prob 表示隐藏单元的 dropout 概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_embedding_type 表示位置嵌入类型，默认为绝对位置编码
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 在序列化时导出 position_ids (1, len position emb) 为连续内存，并保持持久性
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 初始化 token_type_ids，用零填充，持久性为 False
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    # 前向传播函数
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    # 定义函数的输入参数和返回类型
    ) -> torch.Tensor:
        # 如果输入的 input_ids 不为空，则获取其形状
        if input_ids is not None:
            input_shape = input_ids.size()
        # 如果 input_ids 为空，则获取 inputs_embeds 的形状，去掉最后一个维度
        else:
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度
        seq_length = input_shape[1]

        # 如果 position_ids 为空，则从 self.position_ids 中获取对应位置的位置编码
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # 如果 token_type_ids 为空，则根据模型是否有 token_type_ids 属性来处理
        if token_type_ids is None:
            # 如果模型有 token_type_ids 属性，则使用其中的值，扩展到与输入形状相同
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            # 如果模型没有 token_type_ids 属性，则创建全零的 token_type_ids
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果 inputs_embeds 为空，则根据 input_ids 获取对应的词嵌入
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # 获取 token_type_ids 对应的 token 类型嵌入
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将词嵌入和 token 类型嵌入相加得到 embeddings
        embeddings = inputs_embeds + token_type_embeddings
        # 如果位置嵌入类型为 "absolute"，则获取对应的位置嵌入并加到 embeddings 中
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        # 对 embeddings 进行 LayerNorm 处理
        embeddings = self.LayerNorm(embeddings)
        # 对 embeddings 进行 dropout 处理
        embeddings = self.dropout(embeddings)
        # 返回处理后的 embeddings
        return embeddings
# 定义 BertSelfAttention 类，用于 Bert 模型中的自注意力机制
class BertSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 如果隐藏层大小不能被注意力头数整除，并且 config 没有嵌入大小属性，则引发 ValueError
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 定义注意力头数和每个注意力头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 定义查询、键、值的线性映射层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 定义 dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # 设置位置嵌入类型，默认为绝对位置嵌入
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果位置嵌入类型是相对位置嵌入，则创建距离嵌入层
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        # 设置是否为解码器
        self.is_decoder = config.is_decoder

    # 将输入张量变形以便进行注意力计算
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

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
    # 初始化函数，接受配置和位置嵌入类型作为参数
    def __init__(self, config, position_embedding_type=None):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化 self 属性，使用 BertSelfAttention 类创建自注意力层对象
        self.self = BertSelfAttention(config, position_embedding_type=position_embedding_type)
        # 初始化 output 属性，使用 BertSelfOutput 类创建自输出层对象
        self.output = BertSelfOutput(config)
        # 初始化 pruned_heads 属性为一个空的集合，用于存储已剪枝的注意力头索引
        self.pruned_heads = set()

    # 剪枝方法，用于剪枝注意力头
    def prune_heads(self, heads):
        # 如果传入的注意力头集合为空，则直接返回
        if len(heads) == 0:
            return
        # 调用 find_pruneable_heads_and_indices 方法，找到可剪枝的注意力头及其索引
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

    # 前向传播方法，用于模型的前向计算
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
        # 调用自输出层的前向传播方法，得到注意力输出
        attention_output = self.output(self_outputs[0], hidden_states)
        # 构建模型输出元组
        outputs = (attention_output,) + self_outputs[1:]  # 如果有需要，添加注意力权重信息
        return outputs
# 定义 BertIntermediate 类，继承自 nn.Module
class BertIntermediate(nn.Module):
    # 初始化方法，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个全连接层，输入维度为隐藏层大小，输出维度为中间层大小
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果配置中隐藏层激活函数为字符串类型
        if isinstance(config.hidden_act, str):
            # 使用预定义的激活函数字典中对应的函数作为中间层的激活函数
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            # 否则直接使用配置中指定的激活函数
            self.intermediate_act_fn = config.hidden_act

    # 前向传播方法，接受隐藏状态张量作为输入，返回中间层张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态张量输入全连接层，得到中间层张量
        hidden_states = self.dense(hidden_states)
        # 对中间层张量应用激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回中间层张量
        return hidden_states


# 定义 BertOutput 类，继承自 nn.Module
class BertOutput(nn.Module):
    # 初始化方法，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个全连接层，输入维度为中间层大小，输出维度为隐藏层大小
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个 LayerNorm 层，输入维度为隐藏层大小
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个 dropout 层，用于隐藏层输出
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播方法，接受隐藏状态张量和输入张量作为输入，返回隐藏状态张量
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态张量输入全连接层，得到新的隐藏状态张量
        hidden_states = self.dense(hidden_states)
        # 对新的隐藏状态张量应用 dropout
        hidden_states = self.dropout(hidden_states)
        # 将输入张量和新的隐藏状态张量相加，然后输入 LayerNorm 层
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的隐藏状态张量
        return hidden_states


# 定义 BertLayer 类，继承自 nn.Module
class BertLayer(nn.Module):
    # 初始化方法，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 设置用于分块前馈传播的参数
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 设置序列长度维度
        self.seq_len_dim = 1
        # 创建 BertAttention 层，接受一个配置对象作为参数
        self.attention = BertAttention(config)
        # 设置是否为解码器层
        self.is_decoder = config.is_decoder
        # 设置是否添加交叉注意力
        self.add_cross_attention = config.add_cross_attention
        # 如果添加交叉注意力
        if self.add_cross_attention:
            # 如果不是解码器模型，抛出错误
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 创建另一个 BertAttention 层，用于交叉注意力，使用绝对位置编码
            self.crossattention = BertAttention(config, position_embedding_type="absolute")
        # 创建 BertIntermediate 层，接受一个配置对象作为参数
        self.intermediate = BertIntermediate(config)
        # 创建 BertOutput 层，接受一个配置对象作为参数
        self.output = BertOutput(config)

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
        # 定义函数，接收隐藏状态、注意力掩码、头掩码、输出注意力权重的开关、过去的键/值对作为参数，返回元组类型的 torch.Tensor
        ) -> Tuple[torch.Tensor]:
        # 如果过去的键/值对不为空，则从中提取解码器自注意力的缓存键/值对，位置在索引 1、2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 使用解码器自注意力模块处理隐藏状态
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # 提取解码器自注意力模块的输出
        attention_output = self_attention_outputs[0]

        # 如果是解码器，则最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            # 解码器的输出不包含最后一个元组，而是自注意力之前的其他输出
            outputs = self_attention_outputs[1:-1]
            # 提取当前注意力的键/值对
            present_key_value = self_attention_outputs[-1]
        else:
            # 如果不是解码器，输出中包含自注意力权重
            outputs = self_attention_outputs[1:]  # 如果输出注意力权重，添加自注意力权重
          
        # 初始化交叉注意力的键/值对为 None
        cross_attn_present_key_value = None
        # 如果是解码器并且有编码器隐藏状态
        if self.is_decoder and encoder_hidden_states is not None:
            # 如果没有交叉注意力模块，则抛出错误
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )
                
            # 从过去的键/值对中提取交叉注意力的缓存键/值对，位置在索引 -2、-1
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 使用交叉注意力模块处理注意力输出、注意力掩码、头掩码、编码器隐藏状态、编码器注意力掩码、过去的交叉注意力缓存键/值对
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            # 提取交叉注意力模块的输出
            attention_output = cross_attention_outputs[0]
            # 将交叉注意力输出添加到解码器输出中
            outputs = outputs + cross_attention_outputs[1:-1]  # 如果输出注意力权重，添加交叉注意力权重

            # 将交叉注意力的当前键/值对添加到当前键/值对中
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 将注意力输出传递给前馈神经网络块
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        # 将前馈神经网络块的输出添加到输出中
        outputs = (layer_output,) + outputs

        # 如果是解码器，则返回注意力键/值对作为最后一个输出
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        # 返回所有输出
        return outputs

    # 定义前馈神经网络块，接收注意力输出作为输入，返回前馈神经网络的输出
    def feed_forward_chunk(self, attention_output):
        # 使用中间层处理注意力输出
        intermediate_output = self.intermediate(attention_output)
        # 使用输出层处理中间层输出和注意力输出，得到最终层输出
        layer_output = self.output(intermediate_output, attention_output)
        # 返回最终层输出
        return layer_output
# 定义一个 BERT 编码器类，继承自 nn.Module
class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()  # 调用父类的构造函数
        self.config = config  # 保存配置信息
        # 创建一个由多个 BertLayer 组成的层列表，层数由配置中的 num_hidden_layers 决定
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False  # 设置梯度检查点标志为 False

    # 定义前向传播函数
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
        # 初始化变量，根据是否需要输出隐藏状态、注意力权重等信息来决定是否创建空元组
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 梯度检查点和缓存的兼容性检查
        if self.gradient_checkpointing and self.training:
            if use_cache:
                # 如果使用缓存，则警告梯度检查点和缓存不兼容，并将use_cache设置为False
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # 初始化下一个解码器缓存
        next_decoder_cache = () if use_cache else None
        # 遍历每个解码器层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到all_hidden_states中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码和过去的键值对
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 根据是否启用梯度检查点和训练状态来选择调用方式
            if self.gradient_checkpointing and self.training:
                # 使用梯度检查点函数进行前向传播
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
                # 正常调用解码器层
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            # 更新隐藏状态
            hidden_states = layer_outputs[0]
            # 如果使用缓存，则将当前层的输出添加到下一个解码器缓存中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 如果需要输出注意力权重，则将当前层的注意力权重添加到all_self_attentions中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果模型配置中包含交叉注意力，则将当前层的交叉注意力添加到all_cross_attentions中
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果需要输出隐藏状态，则将最终隐藏状态添加到all_hidden_states中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要返回字典形式的结果，则返回元组形式的结果
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
        # 返回包含过去键值对和交叉注意力的模型输出
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
class BertPooler(nn.Module):
    def __init__(self, config):
        # 初始化 BertPooler 类
        super().__init__()
        # 创建线性变换层，输入维度为隐藏层大小，输出维度为隐藏层大小
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建激活函数层，使用双曲正切函数
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # "汇聚"模型，简单地取对应于第一个标记的隐藏状态
        first_token_tensor = hidden_states[:, 0]
        # 将第一个标记的隐藏状态输入线性变换层
        pooled_output = self.dense(first_token_tensor)
        # 将线性变换后的输出应用激活函数
        pooled_output = self.activation(pooled_output)
        # 返回汇聚后的输出
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        # 初始化 BertPredictionHeadTransform 类
        super().__init__()
        # 创建线性变换层，输入维度为隐藏层大小，输出维度为隐藏层大小
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 如果隐藏激活函数为字符串类型，则根据映射字典获取对应的激活函数，否则使用配置中指定的激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # 创建 LayerNorm 层，输入维度为隐藏层大小
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态输入线性变换层
        hidden_states = self.dense(hidden_states)
        # 应用激活函数
        hidden_states = self.transform_act_fn(hidden_states)
        # 将处理后的隐藏状态输入 LayerNorm 层
        hidden_states = self.LayerNorm(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        # 初始化 BertLMPredictionHead 类
        super().__init__()
        # 创建 BertPredictionHeadTransform 实例
        self.transform = BertPredictionHeadTransform(config)

        # 输出权重与输入嵌入相同，但每个标记有一个输出偏置
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 为了能够正确调整偏置大小，需要将输出偏置与 decoder 变量链接起来
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        # 将隐藏状态输入到 BertPredictionHeadTransform 中
        hidden_states = self.transform(hidden_states)
        # 将处理后的隐藏状态输入到线性层中
        hidden_states = self.decoder(hidden_states)
        # 返回线性层的输出
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        # 初始化 BertOnlyMLMHead 类
        super().__init__()
        # 创建 BertLMPredictionHead 实例
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        # 将序列输出传递给 BertLMPredictionHead 实例
        prediction_scores = self.predictions(sequence_output)
        # 返回预测分数
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        # 初始化 BertOnlyNSPHead 类
        super().__init__()
        # 创建线性变换层，输入维度为隐藏层大小，输出维度为2
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        # 将汇聚输出传递给线性变换层
        seq_relationship_score = self.seq_relationship(pooled_output)
        # 返回序列关系分数
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        # 初始化 BertPreTrainingHeads 类
        super().__init__()
        # 创建 BertLMPredictionHead 实例
        self.predictions = BertLMPredictionHead(config)
        # 创建线性变换层，输入维度为隐藏层大小，输出维度为2
        self.seq_relationship = nn.Linear(config.hidden_size, 2)
    # 前向传播函数，接受序列输出和池化输出作为输入
    def forward(self, sequence_output, pooled_output):
        # 使用序列输出作为输入，生成预测分数
        prediction_scores = self.predictions(sequence_output)
        # 使用池化输出作为输入，生成序列关系分数
        seq_relationship_score = self.seq_relationship(pooled_output)
        # 返回预测分数和序列关系分数
        return prediction_scores, seq_relationship_score
class BertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 使用 BertConfig 类作为配置类
    config_class = BertConfig
    # 使用 load_tf_weights_in_bert 函数加载 TensorFlow 权重
    load_tf_weights = load_tf_weights_in_bert
    # 设置基础模型的前缀为 "bert"
    base_model_prefix = "bert"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # 初始化线性层的权重，使用正态分布，均值为 0，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                # 如果有偏置项，则将其初始化为零
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 初始化嵌入层的权重，使用正态分布，均值为 0，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                # 如果存在填充索引，则将填充索引位置的权重初始化为零
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 初始化 LayerNorm 层的偏置为零
            module.bias.data.zero_()
            # 初始化 LayerNorm 层的权重为 1
            module.weight.data.fill_(1.0)


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

    # 可选，当提供了标签时返回，总损失，为掩码语言建模损失和下一个序列预测（分类）损失的总和
    loss: Optional[torch.FloatTensor] = None
    # 语言建模头的预测分数（SoftMax 之前的每个词汇令牌的分数）
    prediction_logits: torch.FloatTensor = None
    # 下一个序列预测（分类）头的预测分数（SoftMax 之前的 True/False 继续分数）
    seq_relationship_logits: torch.FloatTensor = None
    # 定义一个可选的元组类型变量hidden_states，用于存储torch.FloatTensor类型的数据，默认为None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义一个可选的元组类型变量attentions，用于存储torch.FloatTensor类型的数据，默认为None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# BERT_START_DOCSTRING 是用于生成模型文档字符串的模板，包含了模型的继承关系、通用方法的介绍以及 PyTorch 的相关说明
BERT_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`BertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# BERT_INPUTS_DOCSTRING 是用于生成模型输入文档字符串的模板，在这里未给出具体内容，需要根据具体情况填写
BERT_INPUTS_DOCSTRING = r"""

"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列标记在词汇表中的索引。
            # 可以使用 [`AutoTokenizer`] 获得这些索引。参见 [`PreTrainedTokenizer.encode`] 和
            # [`PreTrainedTokenizer.__call__`] 了解详情。
            # [什么是输入 ID?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 避免在填充标记索引上执行注意力操作的掩码。掩码值选择在 `[0, 1]` 范围内：

            # - 对于**未被掩盖**的标记，值为 1，
            # - 对于**被掩盖**的标记，值为 0。

            # [什么是注意力掩码?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 指示输入的第一部分和第二部分的段标记索引。索引选择在 `[0, 1]` 范围内：

            # - 0 对应于*句子 A* 标记，
            # - 1 对应于*句子 B* 标记。

            # [什么是标记类型 ID?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 每个输入序列标记在位置嵌入中的位置索引。选择范围为 `[0, config.max_position_embeddings - 1]`。

            # [什么是位置 ID?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于使自注意力模块中选择的头部失效的掩码。掩码值选择在 `[0, 1]` 范围内：

            # - 1 表示该头部**未被掩盖**，
            # - 0 表示该头部**被掩盖**。

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选地，您可以选择直接传递嵌入表示而不是传递 `input_ids`。如果您想要更多地控制如何将 `input_ids` 索引转换为关联向量，
            # 而不是使用模型的内部嵌入查找矩阵，这将非常有用。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关详细信息，请参见返回张量中的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关详细信息，请参见返回张量中的 `hidden_states`。
        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通元组。
```  
# 导入必要的库
from transformers.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder, BertPooler
from transformers.modeling_utils import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.configuration_bert import BertConfig
from transformers.file_utils import add_start_docstrings, add_code_sample_docstrings, add_start_docstrings_to_model_forward

# 定义 BertModel 类，继承自 BertPreTrainedModel
@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
class BertModel(BertPreTrainedModel):
    """
    BertModel 类可以作为编码器（只有自注意力）或解码器，解码器时在自注意力层之间添加了交叉注意力层，遵循 [Attention is all you need](https://arxiv.org/abs/1706.03762) 中描述的架构。
    
    要作为解码器使用，需要使用 `is_decoder` 参数初始化配置为 `True`。要在 Seq2Seq 模型中使用，需要同时使用 `is_decoder` 参数和 `add_cross_attention` 参数初始化为 `True`；然后预期在前向传递中输入 `encoder_hidden_states`。
    """

    # 初始化方法
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        # 初始化 BertEmbeddings 和 BertEncoder
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        # 如果需要添加池化层，则初始化 BertPooler
        self.pooler = BertPooler(config) if add_pooling_layer else None

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # 剪枝模型的头部
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 前向传递方法
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
    # Bert 模型，顶部有两个头部，分别用于预训练中的 `masked language modeling` 和 `next sentence prediction (classification)`
    # 这段代码似乎是对 BERT 模型的简要描述，但实际上并没有提供具体的代码逻辑，只是一段注释文本
    Bert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a `next
    sentence prediction (classification)` head.
    """,
    BERT_START_DOCSTRING,
# 定义了一个类 BertForPreTraining，用于BERT的预训练任务
class BertForPreTraining(BertPreTrainedModel):
    # 定义了在预训练中共享权重的键列表
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建一个BERT模型实例
        self.bert = BertModel(config)
        # 创建一个预测头部实例
        self.cls = BertPreTrainingHeads(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出嵌入层
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 定义前向传播方法
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
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
# 添加模型前向传播的文档字符串
@add_start_docstrings(
    """Bert Model with a `language modeling` head on top for CLM fine-tuning.""", BERT_START_DOCSTRING
)
# 定义了一个类 BertLMHeadModel，用于BERT的语言建模任务
class BertLMHeadModel(BertPreTrainedModel):
    # 定义了在预训练中共享权重的键列表
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 如果不是解码器模式，则发出警告
        if not config.is_decoder:
            logger.warning("If you want to use `BertLMHeadModel` as a standalone, add `is_decoder=True.`")

        # 创建一个BERT模型实例，不包含池化层
        self.bert = BertModel(config, add_pooling_layer=False)
        # 创建一个仅MLM头部的实例
        self.cls = BertOnlyMLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出嵌入层
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 定义前向传播方法
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    # 此方法用于前向传播，接受一系列输入参数，并返回模型的输出
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
    ):
    
    # 为生成准备输入。对输入进行预处理，返回生成器所需的输入参数。
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=True, **model_kwargs
    ):
        input_shape = input_ids.shape
        # 如果注意力掩码为 None，则创建一个全为 1 的注意力掩码，形状与输入相同
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # 如果使用了过去的键值（past_key_values），则切片输入的 input_ids
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法已经只传递了最后一个输入 ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认行为：只保留最后一个 ID
                remove_prefix_length = input_ids.shape[1] - 1

            # 切片掉前缀长度为 remove_prefix_length 的部分
            input_ids = input_ids[:, remove_prefix_length:]

        # 返回准备好的输入参数字典
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    # 重排 past_key_values，用于束搜索时重排序过去的键值
    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        # 遍历过去的键值
        for layer_past in past_key_values:
            # 将 past_state 按照 beam_idx 重新排列，并添加到 reordered_past 中
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重新排列后的 past_key_values
        return reordered_past
# 使用装饰器添加文档字符串，说明这是一个在 BERT 模型基础上添加了语言建模头部的类
@add_start_docstrings("""Bert Model with a `language modeling` head on top.""", BERT_START_DOCSTRING)
class BertForMaskedLM(BertPreTrainedModel):
    # 定义需要共享权重的键值对
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 如果配置指定为解码器，发出警告
        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 创建 BERT 模型实例，不添加池化层
        self.bert = BertModel(config, add_pooling_layer=False)
        # 创建仅包含 MLM 头部的实例
        self.cls = BertOnlyMLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 前向传播方法，接受多个输入参数
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

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传入 BERT 模型，返回各种输出
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

        sequence_output = outputs[0]  # 获取 BERT 模型的序列输出
        prediction_scores = self.cls(sequence_output)  # 使用序列输出计算预测得分

        masked_lm_loss = None  # 初始化 masked language modeling 损失为 None
        if labels is not None:  # 如果提供了标签
            loss_fct = CrossEntropyLoss()  # 使用交叉熵损失函数
            # 计算 masked language modeling 损失
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:  # 如果不返回字典形式的输出
            output = (prediction_scores,) + outputs[2:]  # 构建输出元组
            # 返回损失和其他输出，如果 masked_lm_loss 不为 None
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 返回 MaskedLMOutput 类型的对象，包括损失、预测得分、隐藏状态和注意力权重
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape  # 获取输入的形状
        effective_batch_size = input_shape[0]  # 获取有效批次大小

        # 添加一个虚拟标记
        if self.config.pad_token_id is None:  # 如果 PAD token 未定义
            raise ValueError("The PAD token should be defined for generation")

        # 在注意力掩码的最后添加一个零向量，代表虚拟标记的注意力
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        # 创建一个全为 PAD token 的虚拟标记
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        # 在输入的最后添加虚拟标记
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}
# 使用BERT模型，顶部有一个用于下一句预测（分类）的头部
class BertForNextSentencePrediction(BertPreTrainedModel):
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 初始化BERT模型
        self.bert = BertModel(config)
        # 初始化用于NSP（Next Sentence Prediction）的头部
        self.cls = BertOnlyNSPHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法，包含输入参数的文档字符串
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
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see `input_ids` docstring). Indices should be in `[0, 1]`:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.

        Returns:

        Example:

        ```py
        >>> from transformers import AutoTokenizer, BertForNextSentencePrediction
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")

        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
        >>> encoding = tokenizer(prompt, next_sentence, return_tensors="pt")

        >>> outputs = model(**encoding, labels=torch.LongTensor([1]))
        >>> logits = outputs.logits
        >>> assert logits[0, 0] < logits[0, 1]  # next sentence was random
        ```
        """

        # 检查是否有 "next_sentence_label" 参数，如果有则发出警告并使用 labels 参数
        if "next_sentence_label" in kwargs:
            warnings.warn(
                "The `next_sentence_label` argument is deprecated and will be removed in a future version, use"
                " `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("next_sentence_label")

        # 确定是否返回字典形式的输出
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

        # 获取池化后的输出
        pooled_output = outputs[1]

        # 使用分类层对池化输出进行分类
        seq_relationship_scores = self.cls(pooled_output)

        next_sentence_loss = None
        # 如果存在 labels，则计算下一个句子预测的损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))

        # 如果不返回字典形式的输出，则返回相应的结果
        if not return_dict:
            output = (seq_relationship_scores,) + outputs[2:]
            return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

        # 返回下一个句子预测的输出
        return NextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 引入必要的库
@add_start_docstrings(
    """
    Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    BERT_START_DOCSTRING,  # 添加起始注释
)
# 定义一个 BertForSequenceClassification 类，继承自 BertPreTrainedModel
class BertForSequenceClassification(BertPreTrainedModel):
    # 初始化函数，接受一个配置对象 config
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)
        # 获取标签数量
        self.num_labels = config.num_labels
        # 保存配置对象
        self.config = config

        # 加载 BERT 模型
        self.bert = BertModel(config)
        # 设置分类器的 dropout，如果未指定，则使用默认的 hidden_dropout_prob
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 使用 dropout
        self.dropout = nn.Dropout(classifier_dropout)
        # 添加一个线性层，用于分类
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数，接受多个输入参数
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 设置返回字典，如果未指定则使用配置中的返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用BERT模型进行前向传播
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

        # 对池化后的输出进行dropout
        pooled_output = self.dropout(pooled_output)
        # 使用分类器对池化后的输出进行分类
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            # 如果问题类型未指定，则根据标签类型和数量设置问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                # 如果问题类型为回归，则使用均方误差损失函数
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # 如果问题类型为单标签分类，则使用交叉熵损失函数
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # 如果问题类型为多标签分类，则使用带Logits的二元交叉熵损失函数
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            # 如果不返回字典，则返回输出和损失
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回序列分类器输出对象
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用 BERT 模型进行多项选择分类任务，顶部有一个分类头部（线性层和 softmax），例如 RocStories/SWAG 任务
class BertForMultipleChoice(BertPreTrainedModel):
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 初始化 BERT 模型
        self.bert = BertModel(config)
        # 设置分类器的 dropout，如果未指定则使用隐藏层的 dropout
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        # 线性层，将隐藏层的输出映射到 1 维
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法
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
    ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 确定是否返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 获取选择数量
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 重塑输入数据
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 使用BERT模型进行前向传播
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

        # 获取池化输出
        pooled_output = outputs[1]

        # 对池化输出进行dropout
        pooled_output = self.dropout(pooled_output)
        # 使用分类器进行分类
        logits = self.classifier(pooled_output)
        # 重塑logits
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        # 计算损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 如果不返回字典，则返回输出
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回多选模型输出
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 引入依赖
@add_start_docstrings(
    """
    Bert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    BERT_START_DOCSTRING,
)
# 定义 BertForTokenClassification 类，继承自 BertPreTrainedModel
class BertForTokenClassification(BertPreTrainedModel):
    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 获取标签的数量
        self.num_labels = config.num_labels

        # 使用 BertModel 创建 bert 层，不添加池化层
        self.bert = BertModel(config, add_pooling_layer=False)
        # 设置分类器的 dropout，如果配置中未指定，则使用隐藏层的 dropout
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 定义 dropout 层
        self.dropout = nn.Dropout(classifier_dropout)
        # 定义分类器线性层，输入维度为隐藏层大小，输出维度为标签数量
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并进行最终处理
        self.post_init()

    # 前向传播方法
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 确保返回字典是不是空的，如果是，使用模型配置中的返回字典设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 BERT 模型，传入输入的 token 序列等参数
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

        # 获取 BERT 模型输出的序列张量
        sequence_output = outputs[0]

        # 对序列输出进行 dropout 处理
        sequence_output = self.dropout(sequence_output)
        # 通过分类器得到标签预测的 logits
        logits = self.classifier(sequence_output)

        # 初始化损失值为 None
        loss = None
        # 如果提供了标签，计算损失值
        if labels is not None:
            # 定义交叉熵损失函数
            loss_fct = CrossEntropyLoss()
            # 计算交叉熵损失
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不要求返回字典形式的结果
        if not return_dict:
            # 组装输出结果，包括 logits 和可能的额外隐藏状态
            output = (logits,) + outputs[2:]
            # 如果存在损失，将损失加入输出结果
            return ((loss,) + output) if loss is not None else output

        # 返回 TokenClassifierOutput 对象，包含损失、logits、隐藏状态和注意力权重
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用 BERT 模型并在其顶部添加一个用于抽取式问答任务的跨度分类头，例如 SQuAD（在隐藏状态输出的基础上添加线性层来计算“跨度起始对数”和“跨度结束对数”）。
@add_start_docstrings(
    """
    Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    BERT_START_DOCSTRING,
)
# 定义 BertForQuestionAnswering 类，继承自 BertPreTrainedModel 类
class BertForQuestionAnswering(BertPreTrainedModel):
    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置标签数量为配置中的标签数量
        self.num_labels = config.num_labels

        # 使用 BertModel 类构建 Bert 模型，不添加池化层
        self.bert = BertModel(config, add_pooling_layer=False)
        # 使用线性层构建 QA 输出层，输入大小为配置中的隐藏大小，输出大小为标签数量
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法
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
        # 确定是否返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用BERT模型进行前向传播
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

        # 获取BERT模型的输出序列
        sequence_output = outputs[0]

        # 使用QA输出层预测起始和结束位置的logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 如果处于多GPU环境，添加一个维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 有时起始/结束位置超出了模型输入的范围，忽略这些位置
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 使用交叉熵损失函数计算起始和结束位置的损失
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            # 如果不返回字典，则返回起始和结束位置的logits以及其他输出
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 返回QA模型输出，包括损失、起始和结束位置的logits、隐藏状态和注意力权重
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```