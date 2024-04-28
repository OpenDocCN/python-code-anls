# `.\transformers\models\roc_bert\modeling_roc_bert.py`

```py
# 设定文件编码为 UTF-8
# 版权声明，包括 Apache License 2.0 的许可条款
# 请查看 http://www.apache.org/licenses/LICENSE-2.0 获得许可条款的副本
""" PyTorch RoCBert 模型。"""

# 导入模块
import math  # 导入数学模块
import os  # 导入操作系统模块
from typing import List, Optional, Tuple, Union  # 导入类型提示相关的模块

import torch  # 导入 PyTorch 库
import torch.utils.checkpoint  # 导入 PyTorch 中用于检查点的工具
from torch import nn  # 导入 PyTorch 中的神经网络模块
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss  # 导入损失函数

# 导入 HuggingFace 库中的一些相关模块和类
from ...activations import ACT2FN  # 导入激活函数
from ...modeling_outputs import (  # 导入模型输出相关的类
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
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer  # 导入 PyTorch 工具函数
from ...utils import (  # 导入工具函数
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_roc_bert import RoCBertConfig  # 导入 RoCBert 的配置类

# 获取日志记录器
logger = logging.get_logger(__name__)

# 以下是一些文档中使用到的一些常量和字符串

# 文档中所预期的输出形状
_EXPECTED_OUTPUT_SHAPE = [1, 8, 768]

# 用于标记 Token 分类的检查点
_CHECKPOINT_FOR_TOKEN_CLASSIFICATION = "ArthurZ/dummy-rocbert-ner"
# Token 分类任务的预期输出
_TOKEN_CLASS_EXPECTED_OUTPUT = ["S-EVENT", "S-FAC", "I-ORDINAL", "I-ORDINAL", "E-ORG", "E-LANGUAGE", "E-ORG", "E-ORG", "E-ORG", "E-ORG", "I-EVENT", "S-TIME", "S-TIME", "E-LANGUAGE", "S-TIME", "E-DATE", "I-ORDINAL", "E-QUANTITY", "E-LANGUAGE", "S-TIME", "B-ORDINAL", "S-PRODUCT", "E-LANGUAGE", "E-LANGUAGE", "E-ORG", "E-LOC", "S-TIME", "I-ORDINAL", "S-FAC", "O", "S-GPE", "I-EVENT", "S-GPE", "E-LANGUAGE", "E-ORG", "S-EVENT", "S-FAC", "S-FAC", "S-FAC", "E-ORG", "S-FAC", "E-ORG", "S-GPE"]
# Token 分类任务的预期损失
_TOKEN_CLASS_EXPECTED_LOSS = 3.62

# 序列分类任务的检查点
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "ArthurZ/dummy-rocbert-seq"
# 序列分类任务的预期输出
_SEQ_CLASS_EXPECTED_OUTPUT = "'financial news'"
# 序列分类任务的预期损失
_SEQ_CLASS_EXPECTED_LOSS = 2.31

# 问答任务的检查点
_CHECKPOINT_FOR_QA = "ArthurZ/dummy-rocbert-qa"
# 问答任务的预期输出
_QA_EXPECTED_OUTPUT = "''"
# 问答任务的预期损失
_QA_EXPECTED_LOSS = 3.75
# 问答任务答案的起始索引
_QA_TARGET_START_INDEX = 14
# 问答任务答案的结束索引
_QA_TARGET_END_INDEX = 15

# RoCBert 预训练模型的存档列表
ROC_BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "weiweishi/roc-bert-base-zh",
    # 在 https://huggingface.co/models?filter=roc_bert 查看所有 RoCBert 模型
]
# 从transformers.models.bert.modeling_bert.load_tf_weights_in_bert复制代码，并将bert->roc_bert
def load_tf_weights_in_roc_bert(model, config, tf_checkpoint_path):
    """加载PyTorch模型中的tf检查点。"""
    # 尝试导入必要的库
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "在PyTorch中加载TensorFlow模型，需要安装TensorFlow。请查看https://www.tensorflow.org/install/以获取安装说明。"
        )
        raise
    # 获取tf检查点的绝对路径
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"从{tf_path}转换TensorFlow检查点")
    # 从TF模型中加载权重
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info(f"加载具有形状{shape}的TF权重{name}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)
    # 遍历给定的名称和数组，名称表示模型的层次结构，数组表示权重或偏差
    for name, array in zip(names, arrays):
        # 将名称按"/"分割
        name = name.split("/")
        # 检查名称中是否包含不需要的变量或优化器相关的内容，如果是，则跳过当前迭代
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            # 记录日志，跳过当前迭代
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        # 初始化指针为模型对象
        pointer = model
        # 遍历名称中的每一部分
        for m_name in name:
            # 如果名称匹配形如"A-Za-z_\d+"的模式，将其拆分为作用域和编号
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            # 根据名称的第一部分确定指针的位置
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                # 尝试获取指针的下一级属性，如果属性不存在则跳过当前迭代
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            # 如果名称包含编号，则根据编号进一步定位指针
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        # 如果名称以"_embeddings"结尾，则将指针设置为权重属性
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        # 如果名称为"kernel"，则转置数组
        elif m_name == "kernel":
            array = np.transpose(array)
        # 检查指针和数组的形状是否匹配，如果不匹配则引发异常
        try:
            if pointer.shape != array.shape:
                raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
        except ValueError as e:
            e.args += (pointer.shape, array.shape)
            raise
        # 记录日志，初始化PyTorch权重
        logger.info(f"Initialize PyTorch weight {name}")
        # 将数组转换为PyTorch张量，并赋值给指针
        pointer.data = torch.from_numpy(array)
    # 返回模型对象
    return model
# 构建 RoCBertEmbeddings 类，用于构建词嵌入、位置嵌入、形态嵌入和发音嵌入
class RoCBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position, shape, pronunciation and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        # 构建词嵌入层，大小为 config.vocab_size 和 config.hidden_size，填充索引为 config.pad_token_id
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 构建发音嵌入层，大小为 config.pronunciation_vocab_size 和 config.pronunciation_embed_dim，填充索引为 config.pad_token_id
        self.pronunciation_embed = nn.Embedding(
            config.pronunciation_vocab_size, config.pronunciation_embed_dim, padding_idx=config.pad_token_id
        )
        # 构建形态嵌入层，大小为 config.shape_vocab_size 和 config.shape_embed_dim，填充索引为 config.pad_token_id
        self.shape_embed = nn.Embedding(
            config.shape_vocab_size, config.shape_embed_dim, padding_idx=config.pad_token_id
        )
        # 构建位置嵌入层，大小为 config.max_position_embeddings 和 config.hidden_size
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 构建token类型嵌入层，大小为 config.type_vocab_size 和 config.hidden_size
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 标记是否启用发音和形态嵌入
        self.enable_pronunciation = config.enable_pronunciation
        self.enable_shape = config.enable_shape

        # 如果需要连接输入，构建一个线性层将输入合并到 config.hidden_size 维度
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

        # 构建层归一化和dropout层
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 构建位置id缓冲区
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
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
# 构建 RoCBertSelfAttention 类，用于执行自注意力机制
class RoCBertSelfAttention(nn.Module):
    # 定义一个 MultiHeadAttention 类，继承 nn.Module
    def __init__(self, config, position_embedding_type=None):
        # 调用父类的初始化方法
        super().__init__()
        # 检查隐藏层大小是否能被注意力头数整除
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            # 如果不能被整除，抛出异常
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        # 设置注意力头数
        self.num_attention_heads = config.num_attention_heads
        # 计算每个注意力头的大小
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        # 计算所有注意力头的总大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # 定义三个线性层，分别用于查询、键和值的变换
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        # 定义一个dropout层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        # 设置位置编码类型
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果使用相对位置编码
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            # 设置最大位置编码长度
            self.max_position_embeddings = config.max_position_embeddings
            # 定义一个用于存储相对位置编码的embedding层
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
            
        # 设置是否为解码器
        self.is_decoder = config.is_decoder
    
    # 定义一个用于将输入变换为多头注意力形式的函数
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # 计算新的张量形状
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # 将输入张量变换为新形状
        x = x.view(new_x_shape)
        # 将张量的维度重新排列
        return x.permute(0, 2, 1, 3)
    
    # 定义前向传播函数
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
        # 函数体省略
# 创建 RoCBertSelfOutput 类，继承 nn.Module
该类定义了 RoCBert 模型中的自注意力输出层



    def __init__(self, config):
        super().__init__()
        # 创建一个线性层，用于将隐藏状态映射为相同维度的特征
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个 LayerNorm 层，用于归一化输出
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个随机失活层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)



    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态通过线性层进行映射
        hidden_states = self.dense(hidden_states)
        # 对映射后的向量进行随机失活
        hidden_states = self.dropout(hidden_states)
        # 将输入向量和映射后的向量相加，并进行 LayerNorm 归一化
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回归一化后的结果
        return hidden_states



# 创建 RoCBertAttention 类，继承 nn.Module
该类定义了 RoCBert 模型中的注意力层



    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 创建一个 RoCBertSelfAttention 类的实例
        self.self = RoCBertSelfAttention(config, position_embedding_type=position_embedding_type)
        # 创建一个 RoCBertSelfOutput 类的实例
        self.output = RoCBertSelfOutput(config)
        # 创建一个用于存储被删除的注意力头的集合
        self.pruned_heads = set()



    def prune_heads(self, heads):
        # 如果被删除的注意力头为空，则直接返回
        if len(heads) == 0:
            return
        # 使用 find_pruneable_heads_and_indices 函数找到要删除的注意力头的索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 删除线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储被删除的注意力头
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
        # 调用 RoCBertSelfAttention 的 forward 方法
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 调用 RoCBertSelfOutput 的 forward 方法，得到注意力输出
        attention_output = self.output(self_outputs[0], hidden_states)
        # 输出结果包括注意力输出和 RoCBertSelfAttention 的输出
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs



# 创建 RoCBertIntermediate 类，继承 nn.Module
该类定义了 RoCBert 模型中的中间层
    # 定义一个继承自 nn.Module 的中间层组件
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个线性全连接层，输入大小为 config.hidden_size，输出大小为 config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果 config.hidden_act 是字符串，则从 ACT2FN 字典中获取对应的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        # 否则直接使用 config.hidden_act 作为激活函数
        else:
            self.intermediate_act_fn = config.hidden_act
    
    # 定义前向传播方法
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过线性全连接层
        hidden_states = self.dense(hidden_states)
        # 应用激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回结果
        return hidden_states
# 从transformers.models.bert.modeling_bert.BertOutput复制代码，并将Bert->RoCBert
class RoCBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，将输入特征维度转换为配置中的hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个LayerNorm层，用于归一化隐藏状态
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个dropout层，用于防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 对输入经过全连接层进行转换
        hidden_states = self.dense(hidden_states)
        # 对转换后的状态进行dropout
        hidden_states = self.dropout(hidden_states)
        # 对dropout后的状态进行LayerNorm
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

# 从transformers.models.bert.modeling_bert.BertLayer复制代码，并将Bert->RoCBert
class RoCBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 设定前馈网络的chunk大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 设定序列长度的维度
        self.seq_len_dim = 1
        # 创建RoCBertAttention层
        self.attention = RoCBertAttention(config)
        # 是否是解码器
        self.is_decoder = config.is_decoder
        # 是否添加交叉注意力
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 如果添加了交叉注意力，创建RoCBertAttention层
            self.crossattention = RoCBertAttention(config, position_embedding_type="absolute")
        # 创建RoCBertIntermediate层
        self.intermediate = RoCBertIntermediate(config)
        # 创建RoCBertOutput层
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
        # 如果有过去的键/值缓存，则将其切片为前两个元素；否则为None
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 对自注意力层进行计算
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # 获取自注意力层的输出
        attention_output = self_attention_outputs[0]

        # 如果是解码器，最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            # 从自注意力输出中提取除了最后一个元素之外的所有元素
            outputs = self_attention_outputs[1:-1]
            # 提取自注意力的现在的键/值缓存
            present_key_value = self_attention_outputs[-1]
        else:
            # 如果不是解码器，从自注意力输出中提取除第一个元素外的所有元素（添加自注意力权重）
            outputs = self_attention_outputs[1:]
        
        cross_attn_present_key_value = None
        # 如果是解码器且存在编码器的隐藏状态
        if self.is_decoder and encoder_hidden_states is not None:
            # 如果未定义交叉注意力层，则引发错误
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 交叉注意力的键/值缓存元组在过去键/值元组的倒数第二、第三个位置
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 对交叉注意力层进行计算
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            # 获取交叉注意力层的输出
            attention_output = cross_attention_outputs[0]
            # 将交叉注意力输出添加到输出列表中（添加交叉注意力权重）
            outputs = outputs + cross_attention_outputs[1:-1]

            # 将交叉注意力的现在的键/值缓存添加到现在的键/值缓存中
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 对输出进行分块处理并应用前向传播
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        # 将层输出添加到输出元组中
        outputs = (layer_output,) + outputs

        # 如果是解码器，将注意力键/值作为最后一个输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        # 返回所有输出
        return outputs

    # 前馈网络的分块处理
    def feed_forward_chunk(self, attention_output):
        # 通过中间层
        intermediate_output = self.intermediate(attention_output)
        # 通过输出层
        layer_output = self.output(intermediate_output, attention_output)
        # 返回层输出
        return layer_output
# 从transformers.models.bert.modeling_bert.BertEncoder中拷贝代码，将Bert->RoCBert
class RoCBertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # 初始化RoCBertEncoder的配置参数
        self.layer = nn.ModuleList([RoCBertLayer(config) for _ in range(config.num_hidden_layers)])  # 创建RoCBertLayer的列表并初始化
        self.gradient_checkpointing = False  # 设置梯度检查点为False

    # 前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量
        attention_mask: Optional[torch.FloatTensor] = None,  # 可选的注意力掩码张量，默认为None
        head_mask: Optional[torch.FloatTensor] = None,  # 可选的头部掩码张量，默认为None
        encoder_hidden_states: Optional[torch.FloatTensor] = None,  # 可选的编码器隐藏状态张量，默认为None
        encoder_attention_mask: Optional[torch.FloatTensor] = None,  # 可选的编码器注意力掩码张量，默认为None
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 可选的过去的键-值对元组，默认为None
        use_cache: Optional[bool] = None,  # 可选的使用缓存标志，默认为None
        output_attentions: Optional[bool] = False,  # 是否输出注意力，默认为False
        output_hidden_states: Optional[bool] = False,  # 是否输出隐藏状态，默认为False
        return_dict: Optional[bool] = True,  # 返回字典类型，默认为True
    # 该函数是一个深度学习模型的前向传播过程
    # 如果 output_hidden_states 为 True，则会返回所有隐藏状态
    # 如果 output_attentions 为 True，则会返回所有的self-attention和cross-attention结果
    # 该函数支持梯度检查点技术来减少显存占用
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        # 初始化输出变量
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
    
        # 如果使用梯度检查点，且模型处于训练状态，且use_cache为True，则禁用use_cache
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
    
        # 如果使用cache，则初始化next_decoder_cache，否则设为空
        next_decoder_cache = () if use_cache else None
    
        # 对每个transformer层进行前向传播
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到all_hidden_states中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
    
            # 获取当前层的head_mask和past_key_value
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
    
            # 如果使用梯度检查点，则使用_gradient_checkpointing_func进行前向传播
            # 否则直接使用layer_module进行前向传播
            if self.gradient_checkpointing and self.training:
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
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
    
            # 更新hidden_states和next_decoder_cache
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
    
            # 如果需要输出attention，则将当前layer的attention添加到all_self_attentions和all_cross_attentions中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
    
        # 如果需要输出隐藏状态，则将最后一个隐藏状态添加到all_hidden_states中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
    
        # 根据返回类型返回相应的输出
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
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
# 根据 BERT 模型中的池化层定义，创建 RoCBert 模型的池化层
class RoCBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义一个全连接层，将隐藏状态的大小转换为相同大小的输出
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 激活函数为双曲正切函数
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过简单地选择对应于第一个标记的隐藏状态来进行“池化”
        first_token_tensor = hidden_states[:, 0]
        # 对第一个标记的隐藏状态进行全连接和激活操作
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# 根据 BERT 模型中的预测头转换定义，创建 RoCBert 模型的预测头转换层
class RoCBertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义一个全连接层，将隐藏状态的大小转换为相同大小的输出
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 如果隐藏激活函数是字符串，则使用对应的激活函数，否则直接使用给定的激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # 使用 LayerNorm 进行归一化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 对隐藏状态应用全连接层、激活函数和 LayerNorm 归一化
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# 根据 BERT 模型中的语言模型预测头定义，创建 RoCBert 模型的语言模型预测头
class RoCBertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建 RoCBert 模型的预测头转换层
        self.transform = RoCBertPredictionHeadTransform(config)

        # 输出权重与输入嵌入相同，但每个标记有一个仅输出的偏置项
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 预测头的偏置项，对每个标记有一个
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # 需要一个链接来确保偏置项随着 `resize_token_embeddings` 的调整正确调整大小
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        # 经过预测头转换和线性层得到最终的语言模型预测结果
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# 根据 BERT 模型中的只有语言模型预测头定义，创建 RoCBert 模型的只有语言模型预测头
class RoCBertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建 RoCBert 模型的语言模型预测头
        self.predictions = RoCBertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        # 前向传播：生成预测结果
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


# 根据 BERT 模型的预训练模型定义，创建 RoCBert 预训练模型
class RoCBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 配置类，用于处理 RoCBert 的配置信息
    config_class = RoCBertConfig
    # 加载 TensorFlow 权重的函数
    load_tf_weights = load_tf_weights_in_roc_bert
    # 基础模型的前缀
    base_model_prefix = "roc_bert"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    # 初始化权重的函数
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果模块是线性层
        if isinstance(module, nn.Linear):
            # 权重初始化为正态分布，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置，则将偏置初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是嵌入层
        elif isinstance(module, nn.Embedding):
            # 权重初始化为正态分布，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在填充索引，则将填充索引对应的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果模块是归一化层
        elif isinstance(module, nn.LayerNorm):
            # 将偏置初始化为零
            module.bias.data.zero_()
            # 将权重初始化为1
            module.weight.data.fill_(1.0)
# RoCBertModel 类的文档字符串，提供了模型的基本介绍、参数说明和使用方法
ROC_BERT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`RoCBertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# RoCBertModel 类的输入文档字符串
ROC_BERT_INPUTS_DOCSTRING = r"""
"""


# 添加了文档字符串的 RoCBertModel 类，继承自 RoCBertPreTrainedModel 类
@add_start_docstrings(
    "The bare RoCBert Model transformer outputting raw hidden-states without any specific head on top.",
    ROC_BERT_START_DOCSTRING,
)
class RoCBertModel(RoCBertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to be initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    # 重写 __init__ 方法，初始化 RoCBertModel 类
    # 参数：config - RoCBert 配置类实例，add_pooling_layer - 是否添加池化层的布尔值，默认为 True
    def __init__(self, config, add_pooling_layer=True):
        # 调用父类的 __init__ 方法
        super().__init__(config)
        # 保存配置对象
        self.config = config

        # 初始化 RoCBertEmbeddings、RoCBertEncoder 和 RoCBertPooler 对象
        self.embeddings = RoCBertEmbeddings(config)
        self.encoder = RoCBertEncoder(config)

        # 如果 add_pooling_layer 为 True，则初始化 RoCBertPooler 对象，否则为 None
        self.pooler = RoCBertPooler(config) if add_pooling_layer else None

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入层的方法
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入嵌入层的方法
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # 获取发音嵌入层的方法
    def get_pronunciation_embeddings(self):
        return self.embeddings.pronunciation_embed

    # 设置发音嵌入层的方法
    def set_pronunciation_embeddings(self, value):
        self.embeddings.pronunciation_embed = value

    # 获取形状嵌入层的方法
    def get_shape_embeddings(self):
        return self.embeddings.shape_embed

    # 设置形状嵌入层的方法
    def set_shape_embeddings(self, value):
        self.embeddings.shape_embed = value

    # 剪枝头部的方法，待实现
    # 私有方法，用于修剪模型中的注意力头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 遍历每个需要修剪的层和对应的需要修剪的头
        for layer, heads in heads_to_prune.items():
            # 调用编码器（encoder）中的层（layer）的注意力（attention）对象的修剪头方法
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 为模型的前向传播添加文档字符串
    @add_start_docstrings_to_model_forward(ROC_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 为代码示例添加文档字符串
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_EXPECTED_OUTPUT_SHAPE,
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
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 定义带有对比损失和掩码语言建模损失的RoCBert预训练模型
@add_start_docstrings(
    """
    RoCBert Model with contrastive loss and masked_lm_loss during the pretraining.
    """,
    ROC_BERT_START_DOCSTRING,
)
class RoCBertForPreTraining(RoCBertPreTrainedModel):
    # 需要特殊绑定的权重矩阵
    _tied_weights_keys = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]

    def __init__(self, config):
        # 调用父类初始化方法
        super().__init__(config)

        # 初始化 RoCBertModel
        self.roc_bert = RoCBertModel(config)
        # 初始化 RoCBertOnlyMLMHead 用于掩码语言建模任务
        self.cls = RoCBertOnlyMLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 复制自 BertForPreTraining 的方法，获取输出embedding层
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 复制自 BertForPreTraining 的方法，设置输出embedding层
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 添加输入说明文档, 返回掩码语言建模输出
    @add_start_docstrings_to_model_forward(ROC_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        # 输入tensor
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
        ...

# 定义带有"语言建模"头的RoCBert模型
@add_start_docstrings("""RoCBert Model with a `language modeling` head on top.""", ROC_BERT_START_DOCSTRING)
class RoCBertForMaskedLM(RoCBertPreTrainedModel):
    # 需要特殊绑定的权重矩阵
    _tied_weights_keys = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]

    # 复制自 BertForMaskedLM 的初始化方法
    # Copied from transformers.models.bert.modeling_bert.BertForMaskedLM.__init__ with Bert->RoCBert,bert->roc_bert
    ...
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 如果配置中is_decoder为True，则输出警告信息
        if config.is_decoder:
            logger.warning(
                "If you want to use `RoCBertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        # 创建RoCBertModel对象，并禁用汇聚层
        self.roc_bert = RoCBertModel(config, add_pooling_layer=False)
        # 创建RoCBertOnlyMLMHead对象
        self.cls = RoCBertOnlyMLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 从transformers.models.bert.modeling_bert.BertForMaskedLM.get_output_embeddings方法复制而来
    def get_output_embeddings(self):
        # 返回预测层的解码器
        return self.cls.predictions.decoder

    # 从transformers.models.bert.modeling_bert.BertForMaskedLM.set_output_embeddings方法复制而来
    def set_output_embeddings(self, new_embeddings):
        # 设置预测层的解码器
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(ROC_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 前向传播方法
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
    # 定义了一个方法，用于蒙面语言建模的预测或训练
    def forward(
            self, input_ids, input_shape_ids=None, input_pronunciation_ids=None, attention_mask=None, token_type_ids=None,
            position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None,
            encoder_attention_mask=None, output_attentions=None, output_hidden_states=None, return_dict=None, 
            labels=None
        ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        
        """
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                用于计算蒙面语言建模损失的标签。索引应在`[-100, 0, ..., config.vocab_size]`之间（请参见`input_ids`的文档）
                索引设置为 `-100` 的token会被忽略（蒙面），仅对标签为`[0, ..., config.vocab_size]`的token计算损失。
    
            示例:
            ```python
            >>> from transformers import AutoTokenizer, RoCBertForMaskedLM
            >>> import torch
    
            >>> tokenizer = AutoTokenizer.from_pretrained("weiweishi/roc-bert-base-zh")
            >>> model = RoCBertForMaskedLM.from_pretrained("weiweishi/roc-bert-base-zh")
    
            >>> inputs = tokenizer("法国是首都[MASK].", return_tensors="pt")
    
            >>> with torch.no_grad():
            ...     logits = model(**inputs).logits
    
            >>> # 获取 {mask} 的索引
            >>> mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    
            >>> predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
            >>> tokenizer.decode(predicted_token_id)
            '.'
            ```py
        """
    
        # 判断是否要返回字典结果
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # 调用 RoCBertModel 进行前向传播
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
    
        # 获取输出的序列张量
        sequence_output = outputs[0]
    
        # 使用分类器对序列张量进行预测
        prediction_scores = self.cls(sequence_output)
    
        masked_lm_loss = None
    
        # 如果有标签数据，则计算蒙面语言建模的损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            # 将预测和标签数据进行损失计算
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    
        # 根据返回类型判断是否需要返回字典结果
        if not return_dict:
            # 如果不返回字典，则返回包含预测结果和其他输出的元组
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
    
        # 如果返回字典，则创建 MaskedLMOutput 对象返回
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    
    # 定义了一个方法，用于为生成准备输入
    def prepare_inputs_for_generation(
            self, input_ids, input_shape_ids=None, input_pronunciation_ids=None, attention_mask=None, **model_kwargs
        ):
        # 为生成准备输入的方法，接收一系列参数
        # 返回输入相关参数的字典
        ):
        # 获取输入数据的形状
        input_shape = input_ids.shape
        # 获取有效的批量大小
        effective_batch_size = input_shape[0]

        #  添加一个虚拟令牌
        # 如果未定义填充令牌，则引发数值错误
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        # 添加注意力遮罩的虚拟令牌
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        # 创建填充令牌
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        # 将填充令牌连接到输入数据中
        input_ids = torch.cat([input_ids, dummy_token], dim=1)
        # 如果存在输入形状ID，则将填充令牌连接到输入形状ID中
        if input_shape_ids is not None:
            input_shape_ids = torch.cat([input_shape_ids, dummy_token], dim=1)
        # 如果存在输入发音ID，则将填充令牌连接到输入发音ID中
        if input_pronunciation_ids is not None:
            input_pronunciation_ids = torch.cat([input_pronunciation_ids, dummy_token], dim=1)

        # 返回结果字典
        return {
            "input_ids": input_ids,
            "input_shape_ids": input_shape_ids,
            "input_pronunciation_ids": input_pronunciation_ids,
            "attention_mask": attention_mask,
        }
# 根据 ROC_BERT 预训练模型添加语言模型(head)用于 CLM fine-tuning 的 RoCBertModel
class RoCBertForCausalLM(RoCBertPreTrainedModel):
    # _tied_weights_keys 列表中的键表示绑定权重的密钥列表
    _tied_weights_keys = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]

    # __init__ 方法从 RoCBertPreTrainedModel 类中复制，BertLMHeadModel->RoCBertForCausalLM,Bert->RoCBert,bert->roc_bert
    def __init__(self, config):
        # 调用 RoCBertPreTrainedModel 类的初始化方法
        super().__init__(config)

        # 如果config.is_decoder不为真，则发出警告
        if not config.is_decoder:
            logger.warning("If you want to use `RoCRoCBertForCausalLM` as a standalone, add `is_decoder=True.`")

        # 使用RoCBertModel类创建 roc_bert 对象，并将 add_pooling_layer 设置为 False
        self.roc_bert = RoCBertModel(config, add_pooling_layer=False)
        # 使用 RoCBertOnlyMLMHead 类创建 cls 对象
        self.cls = RoCBertOnlyMLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 从 transformers.models.bert.modeling_bert.BertLMHeadModel.get_output_embeddings 复制
    def get_output_embeddings(self):
        # 返回 cls.predictions.decoder 属性
        return self.cls.predictions.decoder

    # 从 transformers.models.bert.modeling_bert.BertLMHeadModel.set_output_embeddings 复制
    def set_output_embeddings(self, new_embeddings):
        # 更新 cls.predictions.decoder 属性
        self.cls.predictions.decoder = new_embeddings

    # 将 add_start_docstrings_to_model_forward 和 replace_return_docstrings 注释添加到 forward 函数中
    @add_start_docstrings_to_model_forward(ROC_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
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
    def prepare_inputs_for_generation(
        self,
        input_ids,
        input_shape_ids=None,
        input_pronunciation_ids=None,
        past_key_values=None,
        attention_mask=None,
        **model_kwargs,
    ):
        # 获取输入 input_ids 的shape
        input_shape = input_ids.shape

        # 如果模型被用作编码器-解码器模型中的解码器，则动态创建解码器注意力掩码
        if attention_mask is None:
            # 创建全1的attention_mask
            attention_mask = input_ids.new_ones(input_shape)

        # 如果使用了 past_key_values，则裁剪 decoder_input_ids
        if past_key_values is not None:
            # 获取历史 past_key_values 的长度
            past_length = past_key_values[0][0].shape[2]

            # 某些生成方法已经只传递最后一个输入ID
            if input_ids.shape[1] > past_length:
                # 从输入中去掉前 past_length 个token
                remove_prefix_length = past_length
            else:
                # 默认行为：保留最后一个ID
                remove_prefix_length = input_ids.shape[1] - 1

            # 裁剪 input_ids
            input_ids = input_ids[:, remove_prefix_length:]
            # 裁剪 input_shape_ids
            if input_shape_ids is not None:
                input_shape_ids = input_shape_ids[:, -1:]
            # 裁剪 input_pronunciation_ids
            if input_pronunciation_ids is not None:
                input_pronunciation_ids = input_pronunciation_ids[:, -1:]

        # 返回裁剪后的输入张量和注意力掩码
        return {
            "input_ids": input_ids,
            "input_shape_ids": input_shape_ids,
            "input_pronunciation_ids": input_pronunciation_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
        }

    # 复制自 transformers.models.bert.modeling_bert.BertLMHeadModel._reorder_cache
    def _reorder_cache(self, past_key_values, beam_idx):
        # 重新排序 past_key_values
        reordered_past = ()
        for layer_past in past_key_values:
            # 对每一层的过去状态进行重排序
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
# 使用 add_start_docstrings 装饰器添加 RoCBertForSequenceClassification 类的文档字符串，
# 用于说明该类是一个 RoCBert 模型变换器，顶部带有用于序列分类/回归的头部（在汇总输出之上的线性层），例如用于 GLUE 任务。
@add_start_docstrings(
    """RoCBert Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks.""",
    ROC_BERT_START_DOCSTRING,
)
# 定义 RoCBertForSequenceClassification 类，继承自 RoCBertPreTrainedModel 类
class RoCBertForSequenceClassification(RoCBertPreTrainedModel):
    # 重写 __init__ 方法，初始化 RoCBertForSequenceClassification 类
    # 该方法源自 transformers.models.bert.modeling_bert.BertForSequenceClassification.__init__，
    # 将 Bert 替换为 RoCBert，bert 替换为 roc_bert
    def __init__(self, config):
        # 调用父类 RoCBertPreTrainedModel 的 __init__ 方法进行初始化
        super().__init__(config)
        # 设置 RoCBertForSequenceClassification 类的 num_labels 属性为 config.num_labels
        self.num_labels = config.num_labels
        # 设置 RoCBertForSequenceClassification 类的 config 属性为 config
        self.config = config

        # 创建一个 RoCBertModel 类的实例 roc_bert，传入 config 对象作为参数
        self.roc_bert = RoCBertModel(config)
        
        # 计算分类器的丢弃率，若配置中没有提供，则使用隐藏层丢弃率作为分类器丢弃率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 创建一个丢弃层，丢弃率为 classifier_dropout
        self.dropout = nn.Dropout(classifier_dropout)
        # 创建一个线性层，将隐藏层的输出大小映射到 config.num_labels
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        # 调用 self.post_init() 方法，初始化模型权重和其他参数
        self.post_init()

    # 使用 add_start_docstrings_to_model_forward 装饰器添加 RoCBertForSequenceClassification 类的前向方法的文档字符串，
    # 用于说明前向传播的输入和输出，具体细节可以参考传入的参数
    @add_start_docstrings_to_model_forward(ROC_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 使用 add_code_sample_docstrings 装饰器添加 RoCBertForSequenceClassification 类的前向方法的示例代码文档字符串，
    # 包括模型加载检查点、输出类型、配置类、预期输出和预期损失等信息
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    )
    # 定义 RoCBertForSequenceClassification 类的前向方法 forward
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
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        设置函数的返回类型注解，可以是包含 torch.Tensor 的元组，或者 SequenceClassifierOutput 的类型
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        调用 roc_bert 模型进行序列分类任务的预测
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

        从输出中获取池化特征向量
        pooled_output = outputs[1]

        对池化特征向量进行 dropout 处理
        pooled_output = self.dropout(pooled_output)
        使用分类器获取 logits
        logits = self.classifier(pooled_output)

        初始化 loss 为 None
        if labels is not None:
            检查问题类型是否为 None，根据情况设置为回归或分类
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            根据问题类型计算 loss
            if self.config.problem_type == "regression":
                使用均方误差损失函数
                loss_fct = MSELoss()
                根据标签和 logits 计算损失
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                使用交叉熵损失函数
                loss_fct = CrossEntropyLoss()
                将 logits 和标签视为一维，计算损失
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                使用带 logits 的二值交叉熵损失函数
                loss_fct = BCEWithLogitsLoss()
                计算多标签损失
                loss = loss_fct(logits, labels)
        如果不需要返回字典
        if not return_dict:
            将 logits 与其他输出信息整合为元组输出
            output = (logits,) + outputs[2:]
            如果有损失，返回损失与 output；否则只返回 output
            return ((loss,) + output) if loss is not None else output

        返回 SequenceClassifierOutput 对象，包括 loss、logits、hidden_states 和 attentions
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用给定的多选题分类头部在 RoCBert 模型基础上构建 RoCBert 多选题模型
# 包含一个线性层（在池化输出之上）和一个 softmax 层，用于 RocStories/SWAG 任务
@add_start_docstrings(
    """RoCBert Model with a multiple choice classification head on top (a linear layer on top of
    the pooled output and a softmax) e.g. for RocStories/SWAG tasks.""",
    ROC_BERT_START_DOCSTRING,
)
# 定义 RoCBert 多选题模型类，继承 RoCBertPreTrainedModel 类
class RoCBertForMultipleChoice(RoCBertPreTrainedModel):
    # 从 transformers.models.bert.modeling_bert.BertForMultipleChoice.__init__ 复制而来，将 Bert->RoCBert,bert->roc_bert
    # 初始化函数，接受一个配置参数 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 使用 RoCBertModel 创建 RoCBert 对象
        self.roc_bert = RoCBertModel(config)
        # 初始化分类器的 dropout，如果未指定则使用配置中的 hidden_dropout_prob
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 使用 dropout 层来应用分类器 dropout
        self.dropout = nn.Dropout(classifier_dropout)
        # 定义线性层，将隐藏状态映射到 1 维
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    # 根据输入文档字符串，变量输入和示例代码添加注释
    @add_start_docstrings_to_model_forward(
        ROC_BERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 前向传播函数，接受多种输入参数和标签，返回模型输出结果
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
    ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 确定是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 获取选择个数
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 重塑输入张量
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

        # 调用 RoC-BERT 模型
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

        # 获取汇总输出
        pooled_output = outputs[1]

        # 对汇总输出进行 dropout 处理
        pooled_output = self.dropout(pooled_output)
        # 使用分类器获取 logits
        logits = self.classifier(pooled_output)
        # 重塑 logits
        reshaped_logits = logits.view(-1, num_choices)

        # 初始化损失值为 None
        loss = None
        # 如果存在标签，则计算损失值
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 如果不需要返回字典形式的输出
        if not return_dict:
            # 重塑输出
            output = (reshaped_logits,) + outputs[2:]
            # 返回损失值和重塑输出
            return ((loss,) + output) if loss is not None else output

        # 返回多重选择模型输出
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 基于 RoCBert 的模型，添加了用于标记分类任务的标记分类头部（在隐藏状态输出的顶部添加了一个线性层），例如命名实体识别（NER）任务
class RoCBertForTokenClassification(RoCBertPreTrainedModel):
    # 从 transformers.models.bert.modeling_bert.BertForTokenClassification.__init__ 复制而来，将 Bert 改为 RoCBert，bert 改为 roc_bert
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 获取标签数量
        self.num_labels = config.num_labels

        # 创建 RoCBert 模型，不添加池化层
        self.roc_bert = RoCBertModel(config, add_pooling_layer=False)
        # 分类器的丢弃率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 添加丢弃层
        self.dropout = nn.Dropout(classifier_dropout)
        # 分类器线性层
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 为模型的前向传播添加文档字符串
    @add_start_docstrings_to_model_forward(ROC_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 添加代码示例的文档字符串
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_TOKEN_CLASSIFICATION,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_TOKEN_CLASS_EXPECTED_OUTPUT,
        expected_loss=_TOKEN_CLASS_EXPECTED_LOSS,
    )
    # 模型的前向传播方法
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
    # 该函数用于计算token分类损失，并返回相关的输出结果
    def forward(
        self,
        input_ids=None,
        input_shape_ids=None,
        input_pronunciation_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 如果return_dict为None, 则使用配置的default值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # 通过roc_bert获取输出结果
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
    
        # 获取序列输出
        sequence_output = outputs[0]
    
        # 对序列输出进行dropout
        sequence_output = self.dropout(sequence_output)
        # 通过分类器获取logits
        logits = self.classifier(sequence_output)
    
        # 如果有labels, 计算交叉熵损失
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    
        # 如果不使用返回字典, 返回logits和其他输出
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
    
        # 返回包含loss, logits, hidden_states和attentions的TokenClassifierOutput
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 为提取式问答任务设计的 RoCBert 模型，带有一个用于分类抽取的 span classification head（在隐藏状态输出的线性层上计算 `span start logits` 和 `span end logits`）。
@add_start_docstrings(
    """RoCBert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).""",
    ROC_BERT_START_DOCSTRING,
)
class RoCBertForQuestionAnswering(RoCBertPreTrainedModel):
    # 从 transformers.models.bert.modeling_bert.BertForQuestionAnswering.__init__ 复制而来，将 Bert->RoCBert，bert->roc_bert
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # 初始化 RoCBertModel，不添加池化层
        self.roc_bert = RoCBertModel(config, add_pooling_layer=False)
        # 初始化线性层，输入大小为 config.hidden_size，输出大小为 config.num_labels
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 添加到 model forward 的起始文档字符串，并添加代码示例文档字符串
    @add_start_docstrings_to_model_forward(ROC_BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
```