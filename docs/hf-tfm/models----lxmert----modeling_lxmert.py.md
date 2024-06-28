# `.\models\lxmert\modeling_lxmert.py`

```
# 导入所需的库和模块
import math  # 导入数学函数库
import os  # 导入操作系统功能库
import warnings  # 导入警告处理库
from dataclasses import dataclass  # 导入数据类装饰器
from typing import Dict, Optional, Tuple, Union  # 导入类型提示相关库

import torch  # 导入PyTorch库
from torch import nn  # 导入神经网络模块
from torch.nn import CrossEntropyLoss, SmoothL1Loss  # 导入交叉熵损失和平滑L1损失

from ...activations import ACT2FN, gelu  # 导入激活函数和GELU激活函数
from ...modeling_utils import PreTrainedModel  # 导入预训练模型基类
from ...utils import (  # 导入工具函数和类
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_lxmert import LxmertConfig  # 导入LXMERT配置类

# 获取全局日志记录器
logger = logging.get_logger(__name__)

# 文档化相关常量
_CHECKPOINT_FOR_DOC = "unc-nlp/lxmert-base-uncased"  # 预训练模型的检查点
_CONFIG_FOR_DOC = "LxmertConfig"  # LXMERT模型配置信息

# 预训练模型存档列表
LXMERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "unc-nlp/lxmert-base-uncased",  # LXMERT基础模型存档
]


class GeLU(nn.Module):
    """
    实现Gaussian Error Linear Unit (GELU)激活函数的PyTorch模块。
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        对输入张量应用GELU激活函数。

        Args:
            x (torch.Tensor): 输入张量

        Returns:
            torch.Tensor: 经过GELU激活函数后的张量
        """
        return gelu(x)


@dataclass
class LxmertModelOutput(ModelOutput):
    """
    LXMERT模型的输出，包含语言编码器、视觉编码器和跨模态编码器的最后隐藏状态、汇总输出和注意力概率。
    （注意：在LXMERT中，视觉编码器称为“关系-语义”编码器）
    """

    # 继承自ModelOutput，不需要额外的字段
    pass  # 无需额外的字段声明，直接继承父类的字段和方法
    # 定义函数的参数和它们的类型注释，指定了每个参数的数据类型和形状
    
    Args:
        language_output (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            最后一层语言编码器的隐藏状态序列。
        vision_output (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            最后一层视觉编码器的隐藏状态序列。
        pooled_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            序列中第一个令牌（分类、CLS令牌）的最后一层隐藏状态，通过一个线性层和Tanh激活函数进一步处理。
        language_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            语言编码器的隐藏状态元组，形状为 `(batch_size, sequence_length, hidden_size)`。
        vision_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            视觉编码器的隐藏状态元组，形状为 `(batch_size, sequence_length, hidden_size)`。
        language_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            注意力权重元组，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`，经过注意力softmax后得到，用于计算自注意力头中的加权平均值。
        vision_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            注意力权重元组，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`，经过注意力softmax后得到，用于计算自注意力头中的加权平均值。
        cross_encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            注意力权重元组，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`，经过注意力softmax后得到，用于计算交叉编码器注意力头中的加权平均值。
    """
    
    language_output: Optional[torch.FloatTensor] = None
    vision_output: Optional[torch.FloatTensor] = None
    pooled_output: Optional[torch.FloatTensor] = None
    language_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    vision_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 声明一个可选的类型为 Tuple[torch.FloatTensor] 的变量 language_attentions，并初始化为 None
    language_attentions: Optional[Tuple[torch.FloatTensor]] = None
    
    # 声明一个可选的类型为 Tuple[torch.FloatTensor] 的变量 vision_attentions，并初始化为 None
    vision_attentions: Optional[Tuple[torch.FloatTensor]] = None
    
    # 声明一个可选的类型为 Tuple[torch.FloatTensor] 的变量 cross_encoder_attentions，并初始化为 None
    cross_encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
# 定义 LxmertForQuestionAnsweringOutput 类，用于存储 LXMERT 模型问题回答的输出结果
@dataclass
class LxmertForQuestionAnsweringOutput(ModelOutput):
    """
    LxmertForQuestionAnswering 的输出类型。

    Args:
        loss (*optional*, 当提供 `labels` 时返回，`torch.FloatTensor`，形状为 `(1,)`):
            总损失，包括掩码语言建模损失和下一个序列预测（分类）损失的和。
        question_answering_score (`torch.FloatTensor`，形状为 `(batch_size, n_qa_answers)`，*optional*):
            问题回答目标的预测分数（分类）。
        language_hidden_states (`tuple(torch.FloatTensor)`，*optional*，当传递 `output_hidden_states=True` 或者 `config.output_hidden_states=True` 时返回):
            元组，包含 `torch.FloatTensor`（一个用于输入特征 + 一个用于每个交叉模态层的输出），
            形状为 `(batch_size, sequence_length, hidden_size)`。
        vision_hidden_states (`tuple(torch.FloatTensor)`，*optional*，当传递 `output_hidden_states=True` 或者 `config.output_hidden_states=True` 时返回):
            元组，包含 `torch.FloatTensor`（一个用于输入特征 + 一个用于每个交叉模态层的输出），
            形状为 `(batch_size, sequence_length, hidden_size)`。
        language_attentions (`tuple(torch.FloatTensor)`，*optional*，当传递 `output_attentions=True` 或者 `config.output_attentions=True` 时返回):
            元组，包含 `torch.FloatTensor`（每个层一个），
            形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
            经过注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
        vision_attentions (`tuple(torch.FloatTensor)`，*optional*，当传递 `output_attentions=True` 或者 `config.output_attentions=True` 时返回):
            元组，包含 `torch.FloatTensor`（每个层一个），
            形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
            经过注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
        cross_encoder_attentions (`tuple(torch.FloatTensor)`，*optional*，当传递 `output_attentions=True` 或者 `config.output_attentions=True` 时返回):
            元组，包含 `torch.FloatTensor`（每个层一个），
            形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
            经过注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    # 损失值，类型为可选的浮点张量
    loss: Optional[torch.FloatTensor] = None
    # 问题回答分数，类型为可选的浮点张量
    question_answering_score: Optional[torch.FloatTensor] = None
    # 语言隐藏状态，类型为可选的张量元组
    language_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 视觉隐藏状态，类型为可选的张量元组
    vision_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 语言注意力权重，类型为可选的张量元组
    language_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 视觉注意力权重，类型为可选的张量元组
    vision_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 定义一个可选的类型注解，表示 cross_encoder_attentions 变量可以是一个包含一个 torch.FloatTensor 的元组，或者是 None
    cross_encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
@dataclass
class LxmertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`LxmertForPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        cross_relationship_score (`torch.FloatTensor` of shape `(batch_size, 2)`):
            Prediction scores of the textual matching objective (classification) head (scores of True/False
            continuation before SoftMax).
        question_answering_score (`torch.FloatTensor` of shape `(batch_size, n_qa_answers)`):
            Prediction scores of question answering objective (classification).
        language_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
        vision_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for input features + one for the output of each cross-modality layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
        language_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        vision_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        cross_encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.

    """
    # 定义损失变量，类型为可选的浮点张量
    loss: Optional[torch.FloatTensor] = None
    
    # 定义预测 logits 变量，类型为可选的浮点张量
    prediction_logits: Optional[torch.FloatTensor] = None
    
    # 定义跨关系分数变量，类型为可选的浮点张量
    cross_relationship_score: Optional[torch.FloatTensor] = None
    
    # 定义问答分数变量，类型为可选的浮点张量
    question_answering_score: Optional[torch.FloatTensor] = None
    
    # 定义语言隐藏状态变量，类型为可选的浮点张量元组
    language_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    
    # 定义视觉隐藏状态变量，类型为可选的浮点张量元组
    vision_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    
    # 定义语言注意力变量，类型为可选的浮点张量元组
    language_attentions: Optional[Tuple[torch.FloatTensor]] = None
    
    # 定义视觉注意力变量，类型为可选的浮点张量元组
    vision_attentions: Optional[Tuple[torch.FloatTensor]] = None
    
    # 定义跨编码器注意力变量，类型为可选的浮点张量元组
    cross_encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
def load_tf_weights_in_lxmert(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re  # 导入正则表达式模块
        import numpy as np  # 导入NumPy库
        import tensorflow as tf  # 导入TensorFlow库
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise

    tf_path = os.path.abspath(tf_checkpoint_path)  # 获取TensorFlow checkpoint文件的绝对路径
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")  # 记录日志：转换TensorFlow checkpoint的路径

    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)  # 获取TensorFlow模型中的所有变量及其形状
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")  # 记录日志：加载TensorFlow权重的名称和形状
        array = tf.train.load_variable(tf_path, name)  # 加载TensorFlow模型中的变量值
        names.append(name)  # 将变量名添加到列表中
        arrays.append(array)  # 将变量值添加到列表中

    for name, array in zip(names, arrays):
        name = name.split("/")
        
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n
            in [
                "adam_v",
                "adam_m",
                "AdamWeightDecayOptimizer",
                "AdamWeightDecayOptimizer_1",
                "global_step",
            ]
            for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")  # 记录日志：跳过特定的TensorFlow变量
            continue
        
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            
            # 根据变量名的前缀，设置对应的PyTorch模型指针
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
                    logger.info(f"Skipping {'/'.join(name)}")  # 记录日志：跳过特定的PyTorch变量
                    continue
            
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]  # 根据索引获取嵌套的指针
        
        # 处理特殊情况下的变量名，设置对应的PyTorch模型指针
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)  # 转置权重数组

        try:
            assert pointer.shape == array.shape  # 断言PyTorch模型指针和权重数组的形状相同
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        
        logger.info(f"Initialize PyTorch weight {name}")  # 记录日志：初始化PyTorch权重的名称
        pointer.data = torch.from_numpy(array)  # 使用NumPy数组初始化PyTorch模型的权重

    return model  # 返回加载了TensorFlow权重的PyTorch模型
    """Construct the embeddings from word, position and token_type embeddings."""

    # 初始化函数，接受一个配置对象config作为参数
    def __init__(self, config):
        super().__init__()
        # 创建一个词嵌入层，用于将输入的词汇索引映射为隐藏大小的词嵌入向量，padding_idx=0表示用0进行填充
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        # 创建一个位置嵌入层，用于将位置索引映射为隐藏大小的位置嵌入向量，padding_idx=0表示用0进行填充
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=0)
        # 创建一个标记类型嵌入层，用于将标记类型索引映射为隐藏大小的嵌入向量，padding_idx=0表示用0进行填充
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size, padding_idx=0)

        # LayerNorm不使用蛇形命名以保持与TensorFlow模型变量名的一致性，使得能够加载任何TensorFlow检查点文件
        # 创建LayerNorm层，用于归一化隐藏状态向量，eps=1e-12是一个非常小的数，用于数值稳定性
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        # 创建Dropout层，用于在训练过程中随机失活部分神经元，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，接受输入的词汇ID（input_ids）、标记类型ID（token_type_ids）和预先计算的嵌入（inputs_embeds）
    def forward(self, input_ids, token_type_ids=None, inputs_embeds=None):
        # 如果input_ids不为None，则获取其形状和设备信息；否则获取inputs_embeds的形状和设备信息
        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        else:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        seq_length = input_shape[1]

        # 根据序列长度创建位置ID张量，dtype=torch.long表示数据类型为长整型，device=device表示放置在指定设备上
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)

        # 如果token_type_ids为None，则创建全零张量作为标记类型ID，数据类型为长整型，设备使用self.position_ids的设备
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果inputs_embeds为None，则通过word_embeddings将input_ids转换为嵌入向量
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # 根据位置ID获取位置嵌入向量
        position_embeddings = self.position_embeddings(position_ids)
        # 根据标记类型ID获取标记类型嵌入向量
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将词嵌入向量、位置嵌入向量和标记类型嵌入向量相加得到最终的嵌入向量
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        # 对最终的嵌入向量进行LayerNorm归一化
        embeddings = self.LayerNorm(embeddings)
        # 对归一化后的向量进行Dropout处理
        embeddings = self.dropout(embeddings)
        return embeddings
# 定义 LxmertAttention 类，继承自 nn.Module，用于执行 LXMERT 模型中的自注意力机制
class LxmertAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        # 检查隐藏层大小是否能被注意力头数整除
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.head_size = self.num_attention_heads * self.attention_head_size

        # 如果未指定上下文维度，则使用配置中的隐藏层大小
        if ctx_dim is None:
            ctx_dim = config.hidden_size
        # 创建查询、键、值的线性映射层
        self.query = nn.Linear(config.hidden_size, self.head_size)
        self.key = nn.Linear(ctx_dim, self.head_size)
        self.value = nn.Linear(ctx_dim, self.head_size)

        # 定义 dropout 层，用于注意力概率的 dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    # 对输入张量 x 进行形状转换，以适应多头注意力机制
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播函数，执行自注意力计算
    def forward(self, hidden_states, context, attention_mask=None, output_attentions=False):
        # 通过查询、键、值映射层计算混合的查询、键、值张量
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        # 对混合的查询、键、值张量进行形状转换，以进行多头注意力计算
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # 计算原始注意力分数，通过查询与键的点积得到
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 如果存在注意力掩码，则将其应用于注意力分数
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # 对注意力分数进行 softmax 归一化，得到注意力概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 对注意力概率进行 dropout 操作
        attention_probs = self.dropout(attention_probs)

        # 计算上下文张量，通过注意力概率与值层的乘积得到
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # 重新调整上下文张量的形状，以匹配预期的输出形状
        new_context_layer_shape = context_layer.size()[:-2] + (self.head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # 如果需要输出注意力权重，则将其包含在输出中
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class LxmertAttentionOutput(nn.Module):
    # 初始化函数，用于初始化神经网络模型的参数和层
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个线性层，输入和输出的大小都为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个 LayerNorm 层，对隐藏状态进行归一化处理
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        # 创建一个 Dropout 层，用于在训练过程中随机置零输入张量的部分元素
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，定义了模型的计算过程
    def forward(self, hidden_states, input_tensor):
        # 使用线性层对隐藏状态进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对变换后的隐藏状态进行随机置零处理，以减少过拟合
        hidden_states = self.dropout(hidden_states)
        # 对处理后的隐藏状态进行 LayerNorm 归一化，并与输入张量相加
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的隐藏状态作为输出
        return hidden_states
class LxmertCrossAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化交叉注意力层，包括注意力和输出
        self.att = LxmertAttention(config)
        self.output = LxmertAttentionOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None, output_attentions=False):
        # 执行前向传播，调用注意力层，并返回注意力输出
        output = self.att(input_tensor, ctx_tensor, ctx_att_mask, output_attentions=output_attentions)
        if output_attentions:
            # 如果需要输出注意力权重，则获取注意力概率
            attention_probs = output[1]
        # 使用输出层处理注意力输出和输入张量，得到最终输出
        attention_output = self.output(output[0], input_tensor)
        # 根据需要是否输出注意力权重，构建最终输出结果
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)
        return outputs


class LxmertSelfAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化自注意力层，包括注意力和输出
        self.self = LxmertAttention(config)
        self.output = LxmertAttentionOutput(config)

    def forward(self, input_tensor, attention_mask, output_attentions=False):
        # 自注意力层的前向传播，处理输入张量、注意力掩码，并返回注意力输出
        # 注意：自注意力的键和查询是相同的（即输入张量）
        output = self.self(
            input_tensor,
            input_tensor,
            attention_mask,
            output_attentions=output_attentions,
        )
        if output_attentions:
            # 如果需要输出注意力权重，则获取注意力概率
            attention_probs = output[1]
        # 使用输出层处理注意力输出和输入张量，得到最终输出
        attention_output = self.output(output[0], input_tensor)
        # 根据需要是否输出注意力权重，构建最终输出结果
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)
        return outputs


class LxmertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化中间层，包括线性变换和激活函数
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        # 中间层的前向传播，先进行线性变换，再应用激活函数
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class LxmertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化输出层，包括线性变换、LayerNorm和Dropout
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # 输出层的前向传播，先进行线性变换，再应用Dropout和LayerNorm，最后与输入张量相加
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LxmertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化整个 LXMERT 层，包括自注意力层、中间层和输出层
        self.attention = LxmertSelfAttentionLayer(config)
        self.intermediate = LxmertIntermediate(config)
        self.output = LxmertOutput(config)
    # 定义一个前向传播方法，接受隐藏状态作为输入，并可选地接受注意力掩码和是否输出注意力信息的标志
    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        # 调用注意力层的前向传播方法，得到输出
        outputs = self.attention(hidden_states, attention_mask, output_attentions=output_attentions)
        # 从注意力层的输出中获取注意力输出
        attention_output = outputs[0]
        # 将注意力输出送入中间层处理
        intermediate_output = self.intermediate(attention_output)
        # 将中间层的输出送入输出层处理，得到最终层输出
        layer_output = self.output(intermediate_output, attention_output)
        # 如果输出注意力信息被激活，将注意力信息加入输出元组中
        outputs = (layer_output,) + outputs[1:]  # add attentions if we output them
        # 返回所有输出（包括最终层输出和可能的注意力信息）
        return outputs
class LxmertXLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # The cross-attention Layer
        self.visual_attention = LxmertCrossAttentionLayer(config)

        # Self-attention Layers
        self.lang_self_att = LxmertSelfAttentionLayer(config)
        self.visn_self_att = LxmertSelfAttentionLayer(config)

        # Intermediate and Output Layers (FFNs)
        self.lang_inter = LxmertIntermediate(config)
        self.lang_output = LxmertOutput(config)
        self.visn_inter = LxmertIntermediate(config)
        self.visn_output = LxmertOutput(config)

    def cross_att(
        self,
        lang_input,
        lang_attention_mask,
        visual_input,
        visual_attention_mask,
        output_x_attentions=False,
    ):
        # Cross Attention between language and visual inputs
        lang_att_output = self.visual_attention(
            lang_input,
            visual_input,
            ctx_att_mask=visual_attention_mask,
            output_attentions=output_x_attentions,
        )
        # Cross Attention between visual and language inputs
        visual_att_output = self.visual_attention(
            visual_input,
            lang_input,
            ctx_att_mask=lang_attention_mask,
            output_attentions=False,
        )
        return lang_att_output, visual_att_output

    def self_att(self, lang_input, lang_attention_mask, visual_input, visual_attention_mask):
        # Self Attention for language input
        lang_att_output = self.lang_self_att(lang_input, lang_attention_mask, output_attentions=False)
        # Self Attention for visual input
        visual_att_output = self.visn_self_att(visual_input, visual_attention_mask, output_attentions=False)
        return lang_att_output[0], visual_att_output[0]

    def output_fc(self, lang_input, visual_input):
        # Feed-forward layers for language input
        lang_inter_output = self.lang_inter(lang_input)
        # Feed-forward layers for visual input
        visual_inter_output = self.visn_inter(visual_input)

        # Output layers for language input
        lang_output = self.lang_output(lang_inter_output, lang_input)
        # Output layers for visual input
        visual_output = self.visn_output(visual_inter_output, visual_input)

        return lang_output, visual_output

    def forward(
        self,
        lang_feats,
        lang_attention_mask,
        visual_feats,
        visual_attention_mask,
        output_attentions=False,
    ):
        # Perform cross-attention
        lang_att_output, visual_att_output = self.cross_att(
            lang_feats,
            lang_attention_mask,
            visual_feats,
            visual_attention_mask,
            output_x_attentions=output_attentions,
        )

        # Perform self-attention
        lang_self_output, visual_self_output = self.self_att(
            lang_feats,
            lang_attention_mask,
            visual_feats,
            visual_attention_mask,
        )

        # Perform output FC layers
        lang_output, visual_output = self.output_fc(lang_self_output, visual_self_output)

        return lang_output, visual_output
    # 定义一个方法，执行交叉注意力操作，将语言和视觉特征进行注意力计算
    def forward(
        self,
        lang_feats,              # 输入的语言特征
        lang_attention_mask,     # 语言注意力掩码
        visual_feats,            # 输入的视觉特征
        visual_attention_mask,   # 视觉注意力掩码
        output_attentions=False  # 是否输出注意力矩阵，默认为 False
    ):
        # 执行交叉注意力计算，得到语言和视觉的注意力输出
        lang_att_output, visual_att_output = self.cross_att(
            lang_input=lang_feats,
            lang_attention_mask=lang_attention_mask,
            visual_input=visual_feats,
            visual_attention_mask=visual_attention_mask,
            output_x_attentions=output_attentions,
        )
        # 获取语言注意力输出中除第一个之外的所有部分
        attention_probs = lang_att_output[1:]
        
        # 执行自注意力计算，传入语言和视觉的注意力输出以及对应的注意力掩码
        lang_att_output, visual_att_output = self.self_att(
            lang_att_output[0],
            lang_attention_mask,
            visual_att_output[0],
            visual_attention_mask,
        )
        
        # 将经过注意力计算后的语言和视觉输出，输入到输出全连接层进行最终的输出
        lang_output, visual_output = self.output_fc(lang_att_output, visual_att_output)
        
        # 根据是否需要输出注意力矩阵，决定返回值的格式
        return (
            (
                lang_output,          # 语言输出
                visual_output,        # 视觉输出
                attention_probs[0],   # 第一个注意力矩阵（如果有输出注意力）
            )
            if output_attentions        # 如果需要输出注意力矩阵
            else (lang_output, visual_output)  # 否则只返回语言和视觉输出
        )
        # LXMERT 编码器模型，用于处理多模态输入数据
        super().__init__()

        # 对象级别视觉特征编码层
        self.visn_fc = LxmertVisualFeatureEncoder(config)
        self.config = config

        # 层的数量
        self.num_l_layers = config.l_layers
        self.num_x_layers = config.x_layers
        self.num_r_layers = config.r_layers

        # 层的初始化
        # 使用 self.layer 而不是 self.l_layer 来支持加载 BERT 权重
        self.layer = nn.ModuleList([LxmertLayer(config) for _ in range(self.num_l_layers)])
        self.x_layers = nn.ModuleList([LxmertXLayer(config) for _ in range(self.num_x_layers)])
        self.r_layers = nn.ModuleList([LxmertLayer(config) for _ in range(self.num_r_layers)])

    def forward(
        self,
        lang_feats,
        lang_attention_mask,
        visual_feats,
        visual_pos,
        visual_attention_mask=None,
        output_attentions=None,
        ):
            vision_hidden_states = ()
            language_hidden_states = ()
            # 如果需要输出注意力权重或者配置要求输出注意力权重，则初始化视觉和语言注意力为空元组，否则设为None
            vision_attentions = () if output_attentions or self.config.output_attentions else None
            language_attentions = () if output_attentions or self.config.output_attentions else None
            cross_encoder_attentions = () if output_attentions or self.config.output_attentions else None

            visual_feats = self.visn_fc(visual_feats, visual_pos)

            # 运行语言层
            for layer_module in self.layer:
                # 调用每个语言层模块进行前向传播
                l_outputs = layer_module(lang_feats, lang_attention_mask, output_attentions=output_attentions)
                lang_feats = l_outputs[0]
                # 将每一层的隐藏状态添加到语言隐藏状态元组中
                language_hidden_states = language_hidden_states + (lang_feats,)
                # 如果需要记录注意力权重，将每一层的注意力权重添加到语言注意力元组中
                if language_attentions is not None:
                    language_attentions = language_attentions + (l_outputs[1],)

            # 运行关系层
            for layer_module in self.r_layers:
                # 调用每个关系层模块进行前向传播
                v_outputs = layer_module(visual_feats, visual_attention_mask, output_attentions=output_attentions)
                visual_feats = v_outputs[0]
                # 将每一层的隐藏状态添加到视觉隐藏状态元组中
                vision_hidden_states = vision_hidden_states + (visual_feats,)
                # 如果需要记录注意力权重，将每一层的注意力权重添加到视觉注意力元组中
                if vision_attentions is not None:
                    vision_attentions = vision_attentions + (v_outputs[1],)

            # 运行跨模态层
            for layer_module in self.x_layers:
                # 调用每个跨模态层模块进行前向传播
                x_outputs = layer_module(
                    lang_feats,
                    lang_attention_mask,
                    visual_feats,
                    visual_attention_mask,
                    output_attentions=output_attentions,
                )
                lang_feats, visual_feats = x_outputs[:2]
                # 将每一层的隐藏状态添加到视觉和语言隐藏状态元组中
                vision_hidden_states = vision_hidden_states + (visual_feats,)
                language_hidden_states = language_hidden_states + (lang_feats,)
                # 如果需要记录注意力权重，将每一层的注意力权重添加到跨模态注意力元组中
                if cross_encoder_attentions is not None:
                    cross_encoder_attentions = cross_encoder_attentions + (x_outputs[2],)
            visual_encoder_outputs = (
                vision_hidden_states,
                vision_attentions if output_attentions else None,
            )
            lang_encoder_outputs = (
                language_hidden_states,
                language_attentions if output_attentions else None,
            )
            # 返回最终的视觉编码器输出、语言编码器输出以及跨编码器注意力权重（如果需要的话）
            return (
                visual_encoder_outputs,
                lang_encoder_outputs,
                cross_encoder_attentions if output_attentions else None,
            )
class LxmertVisualObjHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        hid_dim = config.hidden_size
        self.vis_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            nn.LayerNorm(hid_dim * 2, eps=1e-12),
        )

    def forward(self, hidden_states):
        # 进行视觉特征的预测，通过全连接层实现特征转换和归一化
        visual_feats = self.vis_fc(hidden_states)
        return visual_feats
    def __init__(self, config):
        super().__init__()
        self.transform = LxmertPredictionHeadTransform(config)
        # Decide the use of visual losses
        visual_losses = {}
        if config.visual_obj_loss:
            visual_losses["obj"] = {"shape": (-1,), "num": config.num_object_labels}
        if config.visual_attr_loss:
            visual_losses["attr"] = {"shape": (-1,), "num": config.num_attr_labels}
        if config.visual_feat_loss:
            visual_losses["feat"] = {
                "shape": (-1, config.visual_feat_dim),
                "num": config.visual_feat_dim,
            }
        self.visual_losses = visual_losses
        # 定义一个字典，用于存储不同类型的视觉损失

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder_dict = nn.ModuleDict(
            {key: nn.Linear(config.hidden_size, self.visual_losses[key]["num"]) for key in self.visual_losses}
        )
        # 使用 nn.ModuleDict 创建一个 Module 字典，每个 key 对应不同的视觉损失类型，
        # 值为一个 Linear 层，用于处理隐藏状态到对应损失类型的输出映射

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        # 对输入的隐藏状态进行转换
        output = {}
        for key in self.visual_losses:
            output[key] = self.decoder_dict[key](hidden_states)
        # 使用每个视觉损失对应的 Linear 层计算输出
        return output
class LxmertPreTrainingHeads(nn.Module):
    def __init__(self, config, lxmert_model_embedding_weights):
        super(LxmertPreTrainingHeads, self).__init__()
        # 初始化预测头部：语言模型预测头部
        self.predictions = LxmertLMPredictionHead(config, lxmert_model_embedding_weights)
        # 初始化预测头部：序列关系预测头部
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        # 预测语言模型的分数
        prediction_scores = self.predictions(sequence_output)
        # 预测序列关系的分数
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class LxmertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LxmertConfig
    load_tf_weights = load_tf_weights_in_lxmert
    base_model_prefix = "lxmert"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化线性层的权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化嵌入层的权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 将 LayerNorm 层的偏置项初始化为零，权重初始化为 1.0
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


LXMERT_START_DOCSTRING = r"""

    The LXMERT model was proposed in [LXMERT: Learning Cross-Modality Encoder Representations from
    Transformers](https://arxiv.org/abs/1908.07490) by Hao Tan and Mohit Bansal. It's a vision and language transformer
    model, pretrained on a variety of multi-modal datasets comprising of GQA, VQAv2.0, MSCOCO captions, and Visual
    genome, using a combination of masked language modeling, region of interest feature regression, cross entropy loss
    for question answering attribute prediction, and object tag prediction.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
    Parameters:
        config ([`LxmertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

LXMERT_INPUTS_DOCSTRING = r"""
Args:
    batch_size (int): The batch size of the input data.
    sequence_length (int): The length of the input sequences.
"""


@add_start_docstrings(
    "The bare Lxmert Model transformer outputting raw hidden-states without any specific head on top.",
    LXMERT_START_DOCSTRING,
)
class LxmertModel(LxmertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # Initialize the embeddings module with the provided configuration
        self.embeddings = LxmertEmbeddings(config)
        # Initialize the encoder module with the provided configuration
        self.encoder = LxmertEncoder(config)
        # Initialize the pooler module with the provided configuration
        self.pooler = LxmertPooler(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        # Return the word embeddings from the embeddings module
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        # Update the word embeddings in the embeddings module with new_embeddings
        self.embeddings.word_embeddings = new_embeddings

    @add_start_docstrings_to_model_forward(LXMERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=LxmertModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        visual_feats: Optional[torch.FloatTensor] = None,
        visual_pos: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        visual_attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,

        # Perform forward pass through the LxmertModel
        ...
    def __init__(self, config):
        super().__init__(config)
        # Configuration
        self.config = config
        self.num_qa_labels = config.num_qa_labels  # 从配置中获取问答标签数量
        self.visual_loss_normalizer = config.visual_loss_normalizer  # 从配置中获取视觉损失的归一化器

        # Use of pretraining tasks
        self.task_mask_lm = config.task_mask_lm  # 是否执行掩码语言建模任务
        self.task_obj_predict = config.task_obj_predict  # 是否执行对象预测任务
        self.task_matched = config.task_matched  # 是否执行匹配任务
        self.task_qa = config.task_qa  # 是否执行问答任务

        # Lxmert backbone
        self.lxmert = LxmertModel(config)  # 初始化Lxmert模型

        # Pre-training heads
        self.cls = LxmertPreTrainingHeads(config, self.lxmert.embeddings.word_embeddings.weight)  # 初始化预训练头部
        if self.task_obj_predict:
            self.obj_predict_head = LxmertVisualObjHead(config)  # 如果执行对象预测任务，则初始化对象预测头部
        if self.task_qa:
            self.answer_head = LxmertVisualAnswerHead(config, self.num_qa_labels)  # 如果执行问答任务，则初始化问答头部

        # Weight initialization
        # Initialize weights and apply final processing
        self.post_init()  # 执行后初始化操作，包括权重初始化和最终处理

        # Loss functions
        self.loss_fcts = {
            "l2": SmoothL1Loss(reduction="none"),  # 平滑的L1损失函数，不进行降维
            "visual_ce": CrossEntropyLoss(reduction="none"),  # 视觉交叉熵损失函数，不进行降维
            "ce": CrossEntropyLoss(),  # 交叉熵损失函数，进行降维
        }

        visual_losses = {}
        if config.visual_obj_loss:
            visual_losses["obj"] = {
                "shape": (-1,),  # 形状为一维向量
                "num": config.num_object_labels,  # 目标标签数量
                "loss": "visual_ce",  # 使用视觉交叉熵损失
            }
        if config.visual_attr_loss:
            visual_losses["attr"] = {
                "shape": (-1,),  # 形状为一维向量
                "num": config.num_attr_labels,  # 属性标签数量
                "loss": "visual_ce",  # 使用视觉交叉熵损失
            }
        if config.visual_feat_loss:
            visual_losses["feat"] = {
                "shape": (-1, config.visual_feat_dim),  # 形状为二维张量，其中维度为视觉特征维度
                "num": config.visual_feat_dim,  # 视觉特征的维度
                "loss": "l2",  # 使用平滑的L1损失
            }
        self.visual_losses = visual_losses  # 存储视觉损失的配置信息
    def resize_num_qa_labels(self, num_labels):
        """
        从提供的新线性层构建调整大小的问答线性层模块。增加大小会添加新初始化的权重，减小大小会从末尾移除权重。

        Args:
            num_labels (`int`, *optional*):
                线性层权重矩阵中的新标签数量。增加大小会在末尾添加新初始化的权重，减小大小会从末尾移除权重。如果未提供或为 `None`，则仅返回模型的问答标签 `torch.nn.Linear` 模块的指针，而不执行任何操作。

        Returns:
            `torch.nn.Linear`: 调整大小后的线性层指针或旧线性层
        """

        cur_qa_logit_layer = self.get_qa_logit_layer()
        if num_labels is None or cur_qa_logit_layer is None:
            return
        new_qa_logit_layer = self._resize_qa_labels(num_labels)
        self.config.num_qa_labels = num_labels
        self.num_qa_labels = num_labels

        return new_qa_logit_layer

    def _resize_qa_labels(self, num_labels):
        """
        根据指定的标签数量调整当前问答预测线性层。

        Args:
            num_labels (`int`): 线性层权重矩阵中的新标签数量

        Returns:
            `nn.Module`: 调整大小后的问答预测线性层
        """

        cur_qa_logit_layer = self.get_qa_logit_layer()
        new_qa_logit_layer = self._get_resized_qa_labels(cur_qa_logit_layer, num_labels)
        self._set_qa_logit_layer(new_qa_logit_layer)
        return self.get_qa_logit_layer()

    def get_qa_logit_layer(self) -> nn.Module:
        """
        返回生成问答 logits 的线性层模块。

        Returns:
            `nn.Module`: 一个 torch 模块，映射问答预测隐藏状态的线性层，如果 LXMERT 没有视觉回答头部则返回 `None`。
        """
        if hasattr(self, "answer_head"):
            return self.answer_head.logit_fc[-1]

    def _set_qa_logit_layer(self, qa_logit_layer):
        """
        设置问答预测线性层。

        Args:
            qa_logit_layer (`nn.Module`): 新的问答预测线性层
        """
        self.answer_head.logit_fc[-1] = qa_logit_layer
    # 如果 num_labels 为 None，则直接返回当前的 cur_qa_logit_layer
    if num_labels is None:
        return cur_qa_logit_layer

    # 获取当前 cur_qa_logit_layer 的标签数和隐藏维度
    cur_qa_labels, hidden_dim = cur_qa_logit_layer.weight.size()

    # 如果当前标签数等于 num_labels，则直接返回当前的 cur_qa_logit_layer
    if cur_qa_labels == num_labels:
        return cur_qa_logit_layer

    # 如果 cur_qa_logit_layer 存在偏置项，则创建新的线性输出层，否则不创建偏置项的新线性层
    if getattr(cur_qa_logit_layer, "bias", None) is not None:
        new_qa_logit_layer = nn.Linear(hidden_dim, num_labels)
    else:
        new_qa_logit_layer = nn.Linear(hidden_dim, num_labels, bias=False)

    # 将新的线性层放置在与 cur_qa_logit_layer 相同的设备上
    new_qa_logit_layer.to(cur_qa_logit_layer.weight.device)

    # 初始化新标签的权重
    self._init_weights(new_qa_logit_layer)

    # 复制之前权重中的标签
    num_labels_to_copy = min(cur_qa_labels, num_labels)
    new_qa_logit_layer.weight.data[:num_labels_to_copy, :] = cur_qa_logit_layer.weight.data[:num_labels_to_copy, :]
    if getattr(cur_qa_logit_layer, "bias", None) is not None:
        new_qa_logit_layer.bias.data[:num_labels_to_copy] = cur_qa_logit_layer.bias.data[:num_labels_to_copy]

    # 返回新的线性层 new_qa_logit_layer
    return new_qa_logit_layer


@add_start_docstrings_to_model_forward(LXMERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
@replace_return_docstrings(output_type=LxmertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
def forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    visual_feats: Optional[torch.FloatTensor] = None,
    visual_pos: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    visual_attention_mask: Optional[torch.FloatTensor] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    obj_labels: Optional[Dict[str, Tuple[torch.FloatTensor, torch.FloatTensor]]] = None,
    matched_label: Optional[torch.LongTensor] = None,
    ans: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    **kwargs,
):
    # 此处函数定义用于模型的前向传播，接收多个输入参数和可选的返回类型标志
@add_start_docstrings(
    """Lxmert Model with a visual-answering head on top for downstream QA tasks""",
    LXMERT_START_DOCSTRING,
)
class LxmertForQuestionAnswering(LxmertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # Configuration
        self.config = config  # 存储模型配置信息
        self.num_qa_labels = config.num_qa_labels  # 获取问题回答标签的数量
        self.visual_loss_normalizer = config.visual_loss_normalizer  # 获取视觉损失归一化参数

        # Lxmert backbone
        self.lxmert = LxmertModel(config)  # 初始化LXMERT模型作为主干网络

        self.answer_head = LxmertVisualAnswerHead(config, self.num_qa_labels)  # 初始化视觉回答头部

        # Weight initialization
        # Initialize weights and apply final processing
        self.post_init()  # 执行权重初始化和最终处理步骤

        # Loss function
        self.loss = CrossEntropyLoss()  # 定义交叉熵损失函数

    def resize_num_qa_labels(self, num_labels):
        """
        Build a resized question answering linear layer Module from a provided new linear layer. Increasing the size
        will add newly initialized weights. Reducing the size will remove weights from the end

        Args:
            num_labels (`int`, *optional*):
                New number of labels in the linear layer weight matrix. Increasing the size will add newly initialized
                weights at the end. Reducing the size will remove weights from the end. If not provided or `None`, just
                returns a pointer to the qa labels ``torch.nn.Linear``` module of the model without doing anything.

        Return:
            `torch.nn.Linear`: Pointer to the resized Linear layer or the old Linear layer
        """
        cur_qa_logit_layer = self.get_qa_logit_layer()  # 获取当前问题回答对数层

        if num_labels is None or cur_qa_logit_layer is None:
            return  # 如果没有提供num_labels或当前qa_logit_layer为None，则直接返回

        new_qa_logit_layer = self._resize_qa_labels(num_labels)  # 调整问题回答对数层的大小
        self.config.num_qa_labels = num_labels  # 更新模型配置中的问题回答标签数量
        self.num_qa_labels = num_labels  # 更新当前实例的问题回答标签数量

        return new_qa_logit_layer  # 返回调整后的问题回答对数层

    def _resize_qa_labels(self, num_labels):
        cur_qa_logit_layer = self.get_qa_logit_layer()  # 获取当前问题回答对数层
        new_qa_logit_layer = self._get_resized_qa_labels(cur_qa_logit_layer, num_labels)  # 调整问题回答对数层的大小
        self._set_qa_logit_layer(new_qa_logit_layer)  # 设置新的问题回答对数层
        return self.get_qa_logit_layer()  # 返回调整后的问题回答对数层

    def get_qa_logit_layer(self) -> nn.Module:
        """
        Returns the linear layer that produces question answering logits

        Returns:
            `nn.Module`: A torch module mapping the question answering prediction hidden states. `None`: A NoneType
            object if Lxmert does not have the visual answering head.
        """
        if hasattr(self, "answer_head"):
            return self.answer_head.logit_fc[-1]  # 返回最后一个问题回答对数层

    def _set_qa_logit_layer(self, qa_logit_layer):
        self.answer_head.logit_fc[-1] = qa_logit_layer  # 设置最后一个问题回答对数层
    # 如果 num_labels 为 None，则直接返回当前的 cur_qa_logit_layer
    if num_labels is None:
        return cur_qa_logit_layer

    # 获取当前 cur_qa_logit_layer 的标签数量和隐藏层维度
    cur_qa_labels, hidden_dim = cur_qa_logit_layer.weight.size()

    # 如果当前 cur_qa_logit_layer 的标签数量与 num_labels 相同，则直接返回 cur_qa_logit_layer
    if cur_qa_labels == num_labels:
        return cur_qa_logit_layer

    # 如果 cur_qa_logit_layer 具有偏置项，则构建一个新的线性输出层
    if getattr(cur_qa_logit_layer, "bias", None) is not None:
        new_qa_logit_layer = nn.Linear(hidden_dim, num_labels)
    else:
        # 如果 cur_qa_logit_layer 没有偏置项，则构建一个无偏置的新线性输出层
        new_qa_logit_layer = nn.Linear(hidden_dim, num_labels, bias=False)

    # 将新构建的线性输出层放置在与 cur_qa_logit_layer 相同的设备上
    new_qa_logit_layer.to(cur_qa_logit_layer.weight.device)

    # 初始化新线性输出层的权重
    self._init_weights(new_qa_logit_layer)

    # 复制标签从先前权重中的标签
    num_labels_to_copy = min(cur_qa_labels, num_labels)
    new_qa_logit_layer.weight.data[:num_labels_to_copy, :] = cur_qa_logit_layer.weight.data[:num_labels_to_copy, :]

    # 如果 cur_qa_logit_layer 具有偏置项，则同时复制偏置项
    if getattr(cur_qa_logit_layer, "bias", None) is not None:
        new_qa_logit_layer.bias.data[:num_labels_to_copy] = cur_qa_logit_layer.bias.data[:num_labels_to_copy]

    # 返回新构建的线性输出层 new_qa_logit_layer
    return new_qa_logit_layer
        ) -> Union[LxmertForQuestionAnsweringOutput, Tuple[torch.FloatTensor]]:
        r"""
        labels (`Torch.Tensor` of shape `(batch_size)`, *optional*):
            A one-hot representation of the correct answer
        """
        # 根据需要确定是否返回字典格式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 LXMERT 模型进行前向传播
        lxmert_output = self.lxmert(
            input_ids=input_ids,                      # 输入的token IDs
            visual_feats=visual_feats,                # 视觉特征
            visual_pos=visual_pos,                    # 视觉位置编码
            token_type_ids=token_type_ids,            # token类型IDs
            attention_mask=attention_mask,            # 注意力掩码
            visual_attention_mask=visual_attention_mask,  # 视觉注意力掩码
            inputs_embeds=inputs_embeds,              # 输入的嵌入表示
            output_hidden_states=output_hidden_states,  # 输出隐藏状态
            output_attentions=output_attentions,      # 输出注意力
            return_dict=return_dict,                  # 是否返回字典格式的输出
        )

        # 获取经过 LXMERT 模型后的汇总输出
        pooled_output = lxmert_output[2]

        # 使用答案头部对汇总输出进行评分
        answer_score = self.answer_head(pooled_output)

        # 初始化损失值
        loss = None
        # 如果提供了标签，则计算损失值
        if labels is not None:
            loss = self.loss(answer_score.view(-1, self.num_qa_labels), labels.view(-1))

        # 如果不需要返回字典格式的输出，则按元组方式构建输出
        if not return_dict:
            output = (answer_score,) + lxmert_output[3:]
            return (loss,) + output if loss is not None else output

        # 如果需要返回字典格式的输出，则创建相应的输出对象
        return LxmertForQuestionAnsweringOutput(
            loss=loss,  # 损失值
            question_answering_score=answer_score,  # 问题回答分数
            language_hidden_states=lxmert_output.language_hidden_states,  # 语言模型的隐藏状态
            vision_hidden_states=lxmert_output.vision_hidden_states,      # 视觉模型的隐藏状态
            language_attentions=lxmert_output.language_attentions,        # 语言注意力
            vision_attentions=lxmert_output.vision_attentions,            # 视觉注意力
            cross_encoder_attentions=lxmert_output.cross_encoder_attentions,  # 跨编码器注意力
        )
```