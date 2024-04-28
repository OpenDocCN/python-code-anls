# `.\transformers\models\mobilebert\modeling_mobilebert.py`

```py
# 导入所需的库
import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

# 导入 torch 库
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入自定义的库和模块
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_mobilebert import MobileBertConfig

# 获取 logger
logger = logging.get_logger(__name__)

# 设置文档使用的变量和常量
_CHECKPOINT_FOR_DOC = "google/mobilebert-uncased"
_CONFIG_FOR_DOC = "MobileBertConfig"
_CHECKPOINT_FOR_TOKEN_CLASSIFICATION = "mrm8488/mobilebert-finetuned-ner"
_TOKEN_CLASS_EXPECTED_OUTPUT = "['I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'I-LOC', 'O', 'I-LOC', 'I-LOC']"
_TOKEN_CLASS_EXPECTED_LOSS = 0.03
_CHECKPOINT_FOR_QA = "csarron/mobilebert-uncased-squad-v2"
_QA_EXPECTED_OUTPUT = "'a nice puppet'"
_QA_EXPECTED_LOSS = 3.98
_QA_TARGET_START_INDEX = 12
_QA_TARGET_END_INDEX = 13
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "lordtt13/emo-mobilebert"
_SEQ_CLASS_EXPECTED_OUTPUT = "'others'"
_SEQ_CLASS_EXPECTED_LOSS = "4.72"
MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST = ["google/mobilebert-uncased"]

# 加载 TF 模型权重到 MobileBERT 模型中
def load_tf_weights_in_mobilebert(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    # 导入所需的包
    try:
        import re  # 导入正则表达式模块
        import numpy as np  # 导入数值计算模块
        import tensorflow as tf  # 导入 TensorFlow 框架
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise  # 抛出 ImportError 异常
    # 获取 TensorFlow 检查点文件的绝对路径
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")  # 记录日志，显示正在转换的 TensorFlow 检查点文件路径
    # 从 TF 模型加载权重
    init_vars = tf.train.list_variables(tf_path)  # 获取 TensorFlow 模型中的所有变量
    names = []  # 初始化变量名列表
    arrays = []  # 初始化变量值列表
    for name, shape in init_vars:  # 遍历所有变量
        logger.info(f"Loading TF weight {name} with shape {shape}")  # 记录日志，显示正在加载的 TensorFlow 权重的名称和形状
        array = tf.train.load_variable(tf_path, name)  # 加载变量值
        names.append(name)  # 将变量名添加到列表中
        arrays.append(array)  # 将变量值添加到列表中
    # 遍历每个层名称和对应的权重数组
    for name, array in zip(names, arrays):
        # 替换层名称中的一些关键词
        name = name.replace("ffn_layer", "ffn")
        name = name.replace("FakeLayerNorm", "LayerNorm")
        name = name.replace("extra_output_weights", "dense/kernel")
        name = name.replace("bert", "mobilebert")
        # 将层名称分割为多个部分
        name = name.split("/")
        # 跳过一些不需要的变量，如 adam_v、adam_m 等
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        # 获取模型的指针，并根据层名称逐步访问
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            # 根据层名称的不同部分，设置指针指向对应的权重或偏置
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
            # 如果层名称有索引部分，则根据索引访问对应的权重
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        # 如果层名称以 "_embeddings" 结尾，则访问权重
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        # 如果层名称为 "kernel"，则需要转置权重
        elif m_name == "kernel":
            array = np.transpose(array)
        # 检查权重形状是否匹配
        try:
            assert (
                pointer.shape == array.shape
            ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        # 将权重赋值给模型
        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)
    # 返回更新后的模型
    return model
class NoNorm(nn.Module):
    # 定义一个不使用正则化的自定义模块
    def __init__(self, feat_size, eps=None):
        # 初始化函数
        super().__init__()
        # 定义模块参数bias和weight
        self.bias = nn.Parameter(torch.zeros(feat_size))
        self.weight = nn.Parameter(torch.ones(feat_size)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # 前向传播函数，根据输入的张量应用weight和bias的计算
        return input_tensor * self.weight + self.bias


NORM2FN = {"layer_norm": nn.LayerNorm, "no_norm": NoNorm}


class MobileBertEmbeddings(nn.Module):
    # 构建来自单词、位置和令牌类型嵌入的嵌入
    def __init__(self, config):
        # 初始化函数
        super().__init__()
        # 定义类的属性
        self.trigram_input = config.trigram_input
        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size

        # 定义词嵌入、位置嵌入和令牌类型嵌入
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 根据是否使用三元输入确定嵌入输入大小
        embed_dim_multiplier = 3 if self.trigram_input else 1
        embedded_input_size = self.embedding_size * embed_dim_multiplier
        # 定义线性变换层
        self.embedding_transformation = nn.Linear(embedded_input_size, config.hidden_size)

        # 根据配置中的规范化类型选择不同的标准化层
        self.LayerNorm = NORM2FN[config.normalization_type](config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 寄存器缓冲区中的position_ids (1, len position emb)在序列化时是连续的并被导出
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    # 该函数用于生成 BERT 模型的输入嵌入
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 如果传入了 input_ids，则根据它的大小确定输入的形状
        if input_ids is not None:
            input_shape = input_ids.size()
        # 如果没有传入 input_ids，则根据 inputs_embeds 的大小确定输入的形状
        else:
            input_shape = inputs_embeds.size()[:-1]
    
        # 计算输入序列的长度
        seq_length = input_shape[1]
    
        # 如果没有传入 position_ids，则使用预定义的 position_ids
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
    
        # 如果没有传入 token_type_ids，则全部设置为 0
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
    
        # 如果没有传入 inputs_embeds，则使用 word_embeddings 层计算输入的嵌入
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
    
        # 如果使用 trigram 输入，则对输入的嵌入进行特殊处理
        if self.trigram_input:
            inputs_embeds = torch.cat(
                [
                    nn.functional.pad(inputs_embeds[:, 1:], [0, 0, 0, 1, 0, 0], value=0.0),
                    inputs_embeds,
                    nn.functional.pad(inputs_embeds[:, :-1], [0, 0, 1, 0, 0, 0], value=0.0),
                ],
                dim=2,
            )
    
        # 如果使用 trigram 输入或者嵌入大小与隐藏层大小不一致，则使用 embedding_transformation 层对嵌入进行转换
        if self.trigram_input or self.embedding_size != self.hidden_size:
            inputs_embeds = self.embedding_transformation(inputs_embeds)
    
        # 将输入嵌入、位置嵌入和 token 类型嵌入相加，然后进行层归一化和 dropout
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
    
        # 返回最终的输入嵌入
        return embeddings
# 定义 MobileBertSelfAttention 类，继承自 nn.Module
class MobileBertSelfAttention(nn.Module):
    # 初始化函数，接受 config 参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 设置注意力头的数量为 config 中的 num_attention_heads
        self.num_attention_heads = config.num_attention_heads
        # 计算每个注意力头的大小
        self.attention_head_size = int(config.true_hidden_size / config.num_attention_heads)
        # 计算所有注意力头的总大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 定义 query 线性层，输入维度为 config 中的 true_hidden_size，输出维度为 all_head_size
        self.query = nn.Linear(config.true_hidden_size, self.all_head_size)
        # 定义 key 线性层，输入维度为 config 中的 true_hidden_size，输出维度为 all_head_size
        self.key = nn.Linear(config.true_hidden_size, self.all_head_size)
        # 定义 value 线性层，输入维度为 config 中的 true_hidden_size 或 hidden_size，输出维度为 all_head_size
        self.value = nn.Linear(
            config.true_hidden_size if config.use_bottleneck_attention else config.hidden_size, self.all_head_size
        )
        # 定义 dropout 层，丢弃概率为 config 中的 attention_probs_dropout_prob
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    # 定义 transpose_for_scores 方法，用于将输入的张量进行维度转换
    def transpose_for_scores(self, x):
        # 计算新的张量形状
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # 改变张量的形状
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 定义 forward 方法，接受输入张量和一些可选参数，返回一个元组
    def forward(
        self,
        query_tensor: torch.Tensor,
        key_tensor: torch.Tensor,
        value_tensor: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
    ) -> Tuple[torch.Tensor]:
        # 对 query_tensor 进行线性变换
        mixed_query_layer = self.query(query_tensor)
        # 对 key_tensor 进行线性变换
        mixed_key_layer = self.key(key_tensor)
        # 对 value_tensor 进行线性变换
        mixed_value_layer = self.value(value_tensor)

        # 将变换后的张量转换为注意力分数形状
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # 计算 query_layer 和 key_layer 的点积，得到原始注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # 将注意力分数除以注意力头的大小的平方根得到最终的注意力分数
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # 如果存在注意力遮罩，则将其应用到注意力分数上
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        # 对注意力分数进行 softmax 得到注意力概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        # 对注意力概率进行 dropout
        attention_probs = self.dropout(attention_probs)
        # 如果存在头遮罩，则将其应用到注意力概率上
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        # 计算上下文张量，用注意力概率和 value_layer 的点积
        context_layer = torch.matmul(attention_probs, value_layer)
        # 调整上下文张量的形状
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        # 输出结果，如果需要输出注意力分数则包括在返回值中
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs

# 定义 MobileBertSelfOutput 类
class MobileBertSelfOutput(nn.Module):
    # 初始化函数，接受配置参数并初始化网络层
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 根据配置参数设置是否使用 bottleneck（瓶颈）模块
        self.use_bottleneck = config.use_bottleneck
        # 使用线性层将输入的隐藏状态转换为相同维度的输出
        self.dense = nn.Linear(config.true_hidden_size, config.true_hidden_size)
        # 根据配置参数选择规范化方法，并初始化 LayerNorm 层
        self.LayerNorm = NORM2FN[config.normalization_type](config.true_hidden_size, eps=config.layer_norm_eps)
        # 如果不使用 bottleneck，则初始化 dropout 层
        if not self.use_bottleneck:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    # 前向传播函数，接受隐藏状态和残差张量作为输入，返回处理后的张量
    def forward(self, hidden_states: torch.Tensor, residual_tensor: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态经过线性层处理生成输出
        layer_outputs = self.dense(hidden_states)
        # 如果不使用 bottleneck，则对输出进行 dropout 处理
        if not self.use_bottleneck:
            layer_outputs = self.dropout(layer_outputs)
        # 将 output 与 residual_tensor 相加后经过 LayerNorm 处理，并返回处理后的结果
        layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        return layer_outputs
    ``` 
class MobileBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建 MobileBertSelfAttention 实例
        self.self = MobileBertSelfAttention(config)
        # 创建 MobileBertSelfOutput 实例
        self.output = MobileBertSelfOutput(config)
        # 初始化一个空集合用于存储剪枝后的注意力头
        self.pruned_heads = set()

    def prune_heads(self, heads):
        # 如果传入的要剪枝的头数为0，则直接返回
        if len(heads) == 0:
            return
        # 查找可剪枝的头和对应的索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储剪枝后的头
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        query_tensor: torch.Tensor,
        key_tensor: torch.Tensor,
        value_tensor: torch.Tensor,
        layer_input: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
    ) -> Tuple[torch.Tensor]:
        # 运行 MobileBertSelfAttention 的前向传播
        self_outputs = self.self(
            query_tensor,
            key_tensor,
            value_tensor,
            attention_mask,
            head_mask,
            output_attentions,
        )
        # 运行 MobileBertSelfOutput 的前向传播，将注意力输出与层输入相加
        attention_output = self.output(self_outputs[0], layer_input)
        # 如果需要输出注意力权重，则将其添加到输出元组中
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class MobileBertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建 Linear 层
        self.dense = nn.Linear(config.true_hidden_size, config.intermediate_size)
        # 根据隐藏层激活函数配置，选择相应的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过 Linear 层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 使用配置的激活函数处理得到的结果
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class OutputBottleneck(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建 Linear 层
        self.dense = nn.Linear(config.true_hidden_size, config.hidden_size)
        # 根据配置选择归一化方法，并创建相应的归一化层和参数
        self.LayerNorm = NORM2FN[config.normalization_type](config.hidden_size, eps=config.layer_norm_eps)
        # 创建 Dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
```py  
    # 前向传播方法，接收隐藏状态和残差张量，返回处理后的张量
    def forward(self, hidden_states: torch.Tensor, residual_tensor: torch.Tensor) -> torch.Tensor:
        # 使用全连接层处理隐藏状态
        layer_outputs = self.dense(hidden_states)
        # 对全连接层输出进行dropout处理
        layer_outputs = self.dropout(layer_outputs)
        # 将dropout后的输出与残差张量相加，然后经过 LayerNorm 处理
        layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        # 返回处理后的张量
        return layer_outputs
# 定义 MobileBertOutput 类，用于 MobileBERT 的输出
class MobileBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_bottleneck = config.use_bottleneck
        self.dense = nn.Linear(config.intermediate_size, config.true_hidden_size)
        self.LayerNorm = NORM2FN[config.normalization_type](config.true_hidden_size)
        # 如果不使用瓶颈层，则定义 Dropout 层
        if not self.use_bottleneck:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 如果使用瓶颈层，则定义 OutputBottleneck 类
        else:
            self.bottleneck = OutputBottleneck(config)

    def forward(
        self, intermediate_states: torch.Tensor, residual_tensor_1: torch.Tensor, residual_tensor_2: torch.Tensor
    ) -> torch.Tensor:
        # 计算中间状态的线性变换
        layer_output = self.dense(intermediate_states)
        if not self.use_bottleneck:
            # 如果不使用瓶颈层，则进行 Dropout 操作
            layer_output = self.dropout(layer_output)
            # 对结果进行 LayerNorm 和残差连接
            layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        else:
            # 如果使用瓶颈层，则先进行 LayerNorm 和残差连接
            layer_output = self.LayerNorm(layer_output + residual_tensor_1)
            # 然后通过瓶颈层处理
            layer_output = self.bottleneck(layer_output, residual_tensor_2)
        # 返回输出结果
        return layer_output

# 定义 BottleneckLayer 类，用于瓶颈层的操作
class BottleneckLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intra_bottleneck_size)
        self.LayerNorm = NORM2FN[config.normalization_type](config.intra_bottleneck_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 对输入进行线性变换
        layer_input = self.dense(hidden_states)
        # 对结果进行 LayerNorm
        layer_input = self.LayerNorm(layer_input)
        # 返回处理后的结果
        return layer_input

# 定义 Bottleneck 类，用于处理瓶颈层和注意力机制
class Bottleneck(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.key_query_shared_bottleneck = config.key_query_shared_bottleneck
        self.use_bottleneck_attention = config.use_bottleneck_attention
        # 定义输入层为 BottleneckLayer 类的实例
        self.input = BottleneckLayer(config)
        # 如果共享 key query，则注意力机制也使用 BottleneckLayer 类
        if self.key_query_shared_bottleneck:
            self.attention = BottleneckLayer(config)
    # 此方法可以返回三种不同的值元组。这些不同的值利用了瓶颈，这些线性层用于将隐藏状态投影到低维向量，以减少内存使用。
    # 这些线性层具有在训练期间学习的权重。

    # 如果 `config.use_bottleneck_attention`，则会四次返回瓶颈层的结果，用于键、查询、值和"层输入"，供注意力层使用。
    # 这个瓶颈用于投影隐藏状态。最后一层输入将在注意力自输出中用作残差张量，在计算了注意力分数后使用。

    # 如果不是 `config.use_bottleneck_attention` 且 `config.key_query_shared_bottleneck`，则会返回四个值，其中三个通过瓶颈传递：
    # 查询和键通过相同的瓶颈传递，并且在注意力自输出中应用的残差层通过另一个瓶颈传递。

    # 最后，在最后一种情况下，查询、键和值的值是没有瓶颈的隐藏状态，残差层将是这个值通过一个瓶颈传递的结果。

    bottlenecked_hidden_states = self.input(hidden_states)
    # 如果使用瓶颈注意力，则返回瓶颈隐藏状态的四个副本
    if self.use_bottleneck_attention:
        return (bottlenecked_hidden_states,) * 4
    # 如果不使用瓶颈注意力但使用键查询共享瓶颈，则返回四个值，其中三个通过瓶颈传递：共享注意力输入是通过注意力层处理后的结果，
    # 其中查询和键通过相同的瓶颈传递，并且残余层通过另一个瓶颈传递。
    elif self.key_query_shared_bottleneck:
        shared_attention_input = self.attention(hidden_states)
        return (shared_attention_input, shared_attention_input, hidden_states, bottlenecked_hidden_states)
    # 否则，返回隐藏状态、隐藏状态、隐藏状态和瓶颈隐藏状态的四个副本
    else:
        return (hidden_states, hidden_states, hidden_states, bottlenecked_hidden_states)
```  
class FFNOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化全连接层，输入维度为中间层大小，输出维度为真实隐藏层大小
        self.dense = nn.Linear(config.intermediate_size, config.true_hidden_size)
        # 初始化归一化层
        self.LayerNorm = NORM2FN[config.normalization_type](config.true_hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, residual_tensor: torch.Tensor) -> torch.Tensor:
        # 经过全连接层
        layer_outputs = self.dense(hidden_states)
        # 经过归一化层并加上残差连接
        layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        return layer_outputs


class FFNLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化 FFNLayer 中的 MobileBertIntermediate 和 FFNOutput
        self.intermediate = MobileBertIntermediate(config)
        self.output = FFNOutput(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 经过 MobileBertIntermediate 层
        intermediate_output = self.intermediate(hidden_states)
        # 经过 FFNOutput 层
        layer_outputs = self.output(intermediate_output, hidden_states)
        return layer_outputs


class MobileBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 是否使用瓶颈结构
        self.use_bottleneck = config.use_bottleneck
        # FFN 的数量
        self.num_feedforward_networks = config.num_feedforward_networks

        # 初始化 MobileBertLayer 中的注意力、中间层、输出层
        self.attention = MobileBertAttention(config)
        self.intermediate = MobileBertIntermediate(config)
        self.output = MobileBertOutput(config)

        # 如果使用瓶颈结构，初始化瓶颈层
        if self.use_bottleneck:
            self.bottleneck = Bottleneck(config)
        # 如果 FFN 的数量大于 1，初始化 FFNLayer 的列表
        if config.num_feedforward_networks > 1:
            self.ffn = nn.ModuleList([FFNLayer(config) for _ in range(config.num_feedforward_networks - 1)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        ) -> Tuple[torch.Tensor]:  
        # 定义函数的输入和输出类型为 torch.Tensor 的元组
        
        if self.use_bottleneck:
            # 如果使用 bottleneck，从隐藏状态获取查询、键、值张量和层输入
            query_tensor, key_tensor, value_tensor, layer_input = self.bottleneck(hidden_states)
        else:
            # 否则将隐藏状态复制4份作为查询、键、值张量和层输入
            query_tensor, key_tensor, value_tensor, layer_input = [hidden_states] * 4

        self_attention_outputs = self.attention(
            query_tensor,
            key_tensor,
            value_tensor,
            layer_input,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        # 进行自注意力计算
        attention_output = self_attention_outputs[0]
        s = (attention_output,)
        # 初始化一个元组，包含注意力输出

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        # 如果输出注意权重，则添加自注意力输出到输出元组中

        if self.num_feedforward_networks != 1:
            # 如果前馈网络数量不为1
            for i, ffn_module in enumerate(self.ffn):
                # 遍历前馈网络模块
                attention_output = ffn_module(attention_output)
                s += (attention_output,)
                # 更新注意力输出，并添加到元组中

        intermediate_output = self.intermediate(attention_output)
        # 计算中间输出
        layer_output = self.output(intermediate_output, attention_output, hidden_states)
        # 计算最终层输出
        outputs = (
            (layer_output,)
            + outputs
            + (
                torch.tensor(1000),
                query_tensor,
                key_tensor,
                value_tensor,
                layer_input,
                attention_output,
                intermediate_output,
            )
            + s
        )
        # 构建输出元组，包含最终层输出、注意力输出等
        return outputs
        # 返回最终输出
# MobileBertEncoder 类定义
class MobileBertEncoder(nn.Module):
    # 初始化函数
    def __init__(self, config):
        super().__init__()
        # 创建多个 MobileBertLayer 层
        self.layer = nn.ModuleList([MobileBertLayer(config) for _ in range(config.num_hidden_layers)])

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutput]:
        
        # 初始化 all_hidden_states 和 all_attentions
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        # 遍历每个 MobileBertLayer 层
        for i, layer_module in enumerate(self.layer):
            # 当要输出隐藏状态时，更新 all_hidden_states
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 调用 MobileBertLayer 的前向传播函数
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                head_mask[i],
                output_attentions,
            )
            hidden_states = layer_outputs[0]

            # 当要输出注意力时，更新 all_attentions
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        
        # 如果要输出隐藏状态，则将最终隐藏状态加入 all_hidden_states
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # 如果不以字典形式返回，则返回需要的结果
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        # 以 BaseModelOutput 形式返回结果
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


# MobileBertPooler 类定义
class MobileBertPooler(nn.Module):
    # 初始化函数
    def __init__(self, config):
        super().__init__()
        self.do_activate = config.classifier_activation
        # 如果需要激活，则添加线性层
        if self.do_activate:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    # 前向传播函数
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 获取首个 token 对应的隐藏状态
        first_token_tensor = hidden_states[:, 0]
        # 如果不需要激活���则直接返回首个 token 对应的隐藏状态
        if not self.do_activate:
            return first_token_tensor
        else:
            # 添加线性层，并对结果进行 tanh 激活
            pooled_output = self.dense(first_token_tensor)
            pooled_output = torch.tanh(pooled_output)
            return pooled_output


# MobileBertPredictionHeadTransform 类定义
class MobileBertPredictionHeadTransform(nn.Module):
    # 初始化函数
    def __init__(self, config):
        super().__init__()
        # 添加线性层
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 设置激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # 添加 LayerNorm 层
        self.LayerNorm = NORM2FN["layer_norm"](config.hidden_size, eps=config.layer_norm_eps)
    # 前向传播函数
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用一个全连接层对输入的 hidden_states 进行变换
        hidden_states = self.dense(hidden_states)
        # 对变换后的结果应用激活函数
        hidden_states = self.transform_act_fn(hidden_states)
        # 对结果进行层归一化
        hidden_states = self.LayerNorm(hidden_states)
        # 返回最终的输出
        return hidden_states
# MobileBertLMPredictionHead 类负责 BERT 模型的语言模型预测头
class MobileBertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义一个预测头变换层
        self.transform = MobileBertPredictionHeadTransform(config)
        # 输出权重与输入嵌入相同，但每个 token 有一个独立的偏置
        self.dense = nn.Linear(config.vocab_size, config.hidden_size - config.embedding_size, bias=False)
        self.decoder = nn.Linear(config.embedding_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        # 将偏置与解码器偏置链接起来，以便于在调整 token 嵌入大小时调整偏置大小
        self.decoder.bias = self.bias

    # 前向传播方法
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 经过预测头变换层
        hidden_states = self.transform(hidden_states)
        # 计算预测分数
        hidden_states = hidden_states.matmul(torch.cat([self.decoder.weight.t(), self.dense.weight], dim=0))
        hidden_states += self.decoder.bias
        return hidden_states


# MobileBertOnlyMLMHead 类负责 BERT 模型的 Masked Language Model 预测头
class MobileBertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用 MobileBertLMPredictionHead 作为预测头
        self.predictions = MobileBertLMPredictionHead(config)

    # 前向传播方法
    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        # 计算预测分数
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


# MobileBertPreTrainingHeads 类负责 BERT 模型的预训练任务预测头
class MobileBertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用 MobileBertLMPredictionHead 作为语言模型预测头
        self.predictions = MobileBertLMPredictionHead(config)
        # 添加一个序列关系预测层
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    # 前向传播方法
    def forward(self, sequence_output: torch.Tensor, pooled_output: torch.Tensor) -> Tuple[torch.Tensor]:
        # 计算语言模型预测分数
        prediction_scores = self.predictions(sequence_output)
        # 计算序列关系预测分数
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


# MobileBertPreTrainedModel 是一个抽象类，用于处理权重初始化和加载预训练模型
class MobileBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定配置类和预训练模型存档
    config_class = MobileBertConfig
    pretrained_model_archive_map = MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST
    # 载入 TensorFlow 权重的方法
    load_tf_weights = load_tf_weights_in_mobilebert
    # 基础模型前缀
    base_model_prefix = "mobilebert"
    # 初始化神经网络模块的权重
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果模块是全连接层
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重，均值为0，标准差为模型配置的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置项，将其初始化为0
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是嵌入层
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重，均值为0，标准差为模型配置的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在填充索引，将填充索引对应的权重初始化为0
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果模块是 LayerNorm 或者 NoNorm 类型
        elif isinstance(module, (nn.LayerNorm, NoNorm)):
            # 将偏置项初始化为0
            module.bias.data.zero_()
            # 将权重初始化为1
            module.weight.data.fill_(1.0)
# 定义了一个基于`ModelOutput`的数据类`MobileBertForPreTrainingOutput`，用于存储`MobileBertForPreTraining`的输出
class MobileBertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`MobileBertForPreTraining`].

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

    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

# 定义了一个常量`MOBILEBERT_START_DOCSTRING`，存储了一段文档字符串，说明了`MobileBertForPreTraining`的使用方法
MOBILEBERT_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MobileBertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义了一个常量`MOBILEBERT_INPUTS_DOCSTRING`，存储了一段文档字符串
MOBILEBERT_INPUTS_DOCSTRING = r"""
    # 输入序列标记索引
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列标记在词汇表中的索引
            Indices of input sequence tokens in the vocabulary.
    
            # 可以使用 AutoTokenizer 获取索引
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
    
            # 什么是输入 ID？
            [What are input IDs?](../glossary#input-ids)
    # 注意力掩码
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 避免在填充标记索引上执行注意力的掩码
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
    
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
    
            # 什么是注意力掩码？
            [What are attention masks?](../glossary#attention-mask)
    # 标记类型 ID
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 标记类型 ID 用于指示输入的第一部分和第二部分
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:
    
            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.
    
            # 什么是标记类型 ID？
            [What are token type IDs?](../glossary#token-type-ids)
    # 位置 ID
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 输入序列标记在位置嵌入中的索引
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.
    
            # 什么是位置 ID？
            [What are position IDs?](../glossary#position-ids)
    # 头部掩码
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于屏蔽自注意力模块中的选定头部
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
    
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
    
    # 输入嵌入
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选地, 直接传递嵌入表示, 而不是 input_ids
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
    # 输出注意力
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
    # 输出隐藏状态
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
    # 返回字典
        return_dict (`bool`, *optional*):
            # 是否返回 ModelOutput 而不是元组
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""
MobileBert 模型，输出原始隐藏状态，不加任何指定的头部
"""
@add_start_docstrings(
    "The bare MobileBert Model transformer outputting raw hidden-states without any specific head on top.",
    MOBILEBERT_START_DOCSTRING,
)
class MobileBertModel(MobileBertPreTrainedModel):
    """
    MobileBert 模型类
    https://arxiv.org/pdf/2004.02984.pdf
    """

    def __init__(self, config, add_pooling_layer=True):
        # 调用父类初始化函数
        super().__init__(config)
        self.config = config
        # MobileBertEmbeddings 对象
        self.embeddings = MobileBertEmbeddings(config)
        # MobileBertEncoder 对象
        self.encoder = MobileBertEncoder(config)

        # 如果需要添加池化层，则初始化 MobileBertPooler 对象
        self.pooler = MobileBertPooler(config) if add_pooling_layer else None

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 获取输入嵌入层
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        # 设置输入嵌入层权重
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            # 裁剪模型的头部
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPooling,
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
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 定义一个方法，输入参数为input_ids, inputs_embeds, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, output_attentions, output_hidden_states和return_dict，返回值类型为Union[Tuple, BaseModelOutputWithPooling]
        ) -> Union[Tuple, BaseModelOutputWithPooling]:
            # 如果output_attentions参数为None，则取self.config.output_attentions的值
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            # 如果output_hidden_states参数为None，则取self.config.output_hidden_states的值
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            # 如果return_dict参数为None，则取self.config.use_return_dict的值
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
            # 如果input_ids不为None并且inputs_embeds不为None，则抛出数值错误异常
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
            # 如果input_ids不为None，则判断padding和attention_mask
            elif input_ids is not None:
                self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
                input_shape = input_ids.size()
            # 如果inputs_embeds不为None，则获取input_shape
            elif inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
            # 如果input_ids和inputs_embeds都为None，则抛出数值错误异常
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")
    
            # 如果input_ids不为None，则将device设为input_ids的设备，否则设为inputs_embeds的设备
            device = input_ids.device if input_ids is not None else inputs_embeds.device
    
            # 如果attention_mask为None，则创建与input_shape相同维度的全1张量，设备为device
            if attention_mask is None:
                attention_mask = torch.ones(input_shape, device=device)
            # 如果token_type_ids为None，则创建与input_shape相同维度的全0整型张量，设备为device
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
    
            # 获取扩展的注意力掩码
            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)
    
            # 准备头部掩码（如果需要）
            # head_mask中的1.0表示保留此头部
            # attention_probs的形状为 bsz x n_heads x N x N
            # input head_mask的形状为 [num_heads] 或 [num_hidden_layers x num_heads]
            # head_mask转换为形状为 [num_hidden_layers x batch x num_heads x seq_length x seq_length]
            head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
    
            # 将输入传入嵌入层
            embedding_output = self.embeddings(
                input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
            )
            # 将嵌入输出传入编码器
            encoder_outputs = self.encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            # 获取序列输出
            sequence_output = encoder_outputs[0]
            # 如果有池化层，则获取池化输出，否则为None
            pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
    
            # 如果不返回字典，则返回元组
            if not return_dict:
                return (sequence_output, pooled_output) + encoder_outputs[1:]
    
            # 返回带池化的基础模型输出
            return BaseModelOutputWithPooling(
                last_hidden_state=sequence_output,
                pooler_output=pooled_output,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )
# 在MobileBertForPreTraining类上添加文档字符串，并继承自MobileBertPreTrainedModel
@add_start_docstrings(
    """
    MobileBert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a
    `next sentence prediction (classification)` head.
    """,
    MOBILEBERT_START_DOCSTRING,
)
class MobileBertForPreTraining(MobileBertPreTrainedModel):
    _tied_weights_keys = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]

    # 初始化方法，接受config参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建MobileBertModel对象
        self.mobilebert = MobileBertModel(config)
        # 创建MobileBertPreTrainingHeads对象
        self.cls = MobileBertPreTrainingHeads(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddigs):
        self.cls.predictions.decoder = new_embeddigs

    # 调整token嵌入大小
    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> nn.Embedding:
        # 首先调整密集输出嵌入
        self.cls.predictions.dense = self._get_resized_lm_head(
            self.cls.predictions.dense, new_num_tokens=new_num_tokens, transposed=True
        )

        return super().resize_token_embeddings(new_num_tokens=new_num_tokens)

    # 前向传播方法，接受多个输入参数
    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=MobileBertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        next_sentence_label: Optional[torch.LongTensor] = None,
        output_attentions: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[torch.FloatTensor] = None,
        return_dict: Optional[torch.FloatTensor] = None,

# 在MobileBertForMaskedLM类上添加文档字符串，并继承自MobileBertPreTrainedModel
@add_start_docstrings("""MobileBert Model with a `language modeling` head on top.""", MOBILEBERT_START_DOCSTRING)
class MobileBertForMaskedLM(MobileBertPreTrainedModel):
    _tied_weights_keys = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]

    # 初始化方法，接受config参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建MobileBertModel对象，不添加池层
        self.mobilebert = MobileBertModel(config, add_pooling_layer=False)
        # 创建MobileBertOnlyMLMHead对象
        self.cls = MobileBertOnlyMLMHead(config)
        self.config = config

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddigs):
        self.cls.predictions.decoder = new_embeddigs
    # 调整 token embeddings 大小的方法，可选参数为新的 token 数量
    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> nn.Embedding:
        # 首先调整密集输出 embeddings 的大小
        self.cls.predictions.dense = self._get_resized_lm_head(
            self.cls.predictions.dense, new_num_tokens=new_num_tokens, transposed=True
        )
        # 调用父类函数来实际调整 embeddings 的大小，并返回结果
        return super().resize_token_embeddings(new_num_tokens=new_num_tokens)

    # 实现模型的前向传播函数
    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'paris'",
        expected_loss=0.57,
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
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        # 如果 return_dict 不是 None，则使用配置中的 return_dict，否则默认使用配置中的 use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入的参数传递给 MobileBERT 模型进行前向传播
        outputs = self.mobilebert(
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

        # 获取模型输出的序列输出，用于预测
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        # 如果有 labels，则计算 masked language modeling 损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 索引代表填充 token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果不要求返回字典格式的输出
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 返回经过 MaskedLMOutput 封装后的输出
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 定义一个 MobileBertOnlyNSPHead 类，继承自 nn.Module 类
class MobileBertOnlyNSPHead(nn.Module):
    # 初始化方法，接受一个 config 参数
    def __init__(self, config):
        # 调用父类 nn.Module 的初始化方法
        super().__init__()
        # 使用线性层将隐藏层的输出转换为两个类别的得分
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    # 前向传播方法，接受一个 torch.Tensor 类型的参数 pooled_output，返回一个 torch.Tensor 类型的结果
    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        # 使用线性层计算序列关系得分
        seq_relationship_score = self.seq_relationship(pooled_output)
        # 返回序列关系得分
        return seq_relationship_score


# 使用装饰器添加模型的文档说明
@add_start_docstrings(
    """MobileBert Model with a `next sentence prediction (classification)` head on top.""",
    MOBILEBERT_START_DOCSTRING,
)
# 定义 MobileBertForNextSentencePrediction 类，继承自 MobileBertPreTrainedModel 类
class MobileBertForNextSentencePrediction(MobileBertPreTrainedModel):
    # 初始化方法，接受一个 config 参数
    def __init__(self, config):
        # 调用父类 MobileBertPreTrainedModel 的初始化方法
        super().__init__(config)

        # 创建一个 MobileBertModel 对象
        self.mobilebert = MobileBertModel(config)
        # 创建一个 MobileBertOnlyNSPHead 对象
        self.cls = MobileBertOnlyNSPHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 使用装饰器添加模型前向传播方法的文档说明
    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 使用装饰器替换返回结果的文档说明
    @replace_return_docstrings(output_type=NextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
    # 定义模型前向传播方法，接受多个可能为 None 的参数，并返回一个结果
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
    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple, NextSentencePredictorOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see `input_ids` docstring) Indices should be in `[0, 1]`.

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.

        Returns:

        Examples:

        ```py
        >>> from transformers import AutoTokenizer, MobileBertForNextSentencePrediction
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
        >>> model = MobileBertForNextSentencePrediction.from_pretrained("google/mobilebert-uncased")

        >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
        >>> encoding = tokenizer(prompt, next_sentence, return_tensors="pt")

        >>> outputs = model(**encoding, labels=torch.LongTensor([1]))
        >>> loss = outputs.loss
        >>> logits = outputs.logits
        ```"""

        if "next_sentence_label" in kwargs:
            warnings.warn(
                "The `next_sentence_label` argument is deprecated and will be removed in a future version, use"
                " `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("next_sentence_label")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.mobilebert(
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

        pooled_output = outputs[1]  # 获取模型输出的池化表示
        seq_relationship_score = self.cls(pooled_output)  # 通过池化表示计算下一个句子的相关性分数

        next_sentence_loss = None  # 初始化下一个句子的损失值为None
        if labels is not None:  # 如果存在标签
            loss_fct = CrossEntropyLoss()  # 定义交叉熵损失函数
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), labels.view(-1))  # 计算下一个句子的损失值

        if not return_dict:  # 如果不使用返回字典
            output = (seq_relationship_score,) + outputs[2:]  # 构造输出结果
            return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output  # 返回输出结果和损失值

        return NextSentencePredictorOutput(  # 返回下一个句子预测的输出
            loss=next_sentence_loss,  # 输出损失值
            logits=seq_relationship_score,  # 输出相关性分数
            hidden_states=outputs.hidden_states,  # 输出隐藏状态
            attentions=outputs.attentions,  # 输出注意力权重
        )
# 添加起始文档字符串，描述 MobileBert 模型变体，带有顶部的序列分类/回归头（在池化输出的顶部有一个线性层），例如用于 GLUE 任务
@add_start_docstrings(
    """
    MobileBert Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    MOBILEBERT_START_DOCSTRING,
)
# 从 transformers.models.bert.modeling_bert.BertForSequenceClassification 复制而来，将 Bert 改为 MobileBert
class MobileBertForSequenceClassification(MobileBertPreTrainedModel):
    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 设置标签数量
        self.num_labels = config.num_labels
        # 保存配置
        self.config = config

        # 创建 MobileBert 模型
        self.mobilebert = MobileBertModel(config)
        # 设置分类器的丢弃率为配置中的分类器丢弃率，如果没有则使用隐藏层丢弃率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 创建丢弃层
        self.dropout = nn.Dropout(classifier_dropout)
        # 创建线性层，输入大小为隐藏层大小，输出大小为标签数量
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 往前传播方法，接受一些输入
    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        # 确定是否返回字典格式的结果，若未指定则使用模型配置中的设定
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 MobileBERT 模型进行前向传播
        outputs = self.mobilebert(
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

        # 提取池化后的输出
        pooled_output = outputs[1]

        # 对池化后的输出进行 dropout 处理
        pooled_output = self.dropout(pooled_output)
        # 将处理后的输出输入到分类器中得到 logits
        logits = self.classifier(pooled_output)

        loss = None
        # 如果存在标签，则计算损失
        if labels is not None:
            # 确定问题类型，若未指定则根据标签类型和标签数量自动推断
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
        # 如果不返回字典格式的结果，则将输出整合为元组返回
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回结果的字典格式
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用 MobileBert 模型并在其顶部添加一个用于提取式问答任务（如 SQuAD）的 span 分类头部（在隐藏状态输出之上运行线性层，以计算 'span start logits' 和 'span end logits'）。
@add_start_docstrings(
    """
    MobileBert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a
    linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    MOBILEBERT_START_DOCSTRING,
)
# 从 transformers.models.bert.modeling_bert.BertForQuestionAnswering 复制并将 Bert->MobileBert 全部变成小写
class MobileBertForQuestionAnswering(MobileBertPreTrainedModel):
    # 初始化 MobileBertForQuestionAnswering 类
    def __init__(self, config):
        super().__init__(config)
        # 设置类属性 num_labels 为配置文件中的 num_labels
        self.num_labels = config.num_labels

        # 创建 MobileBert 模型（不添加池化层）
        self.mobilebert = MobileBertModel(config, add_pooling_layer=False)
        # 创建 QA 输出层，使用线性变换将隐藏层输出转换为输出标签数
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 定义前向传播方法
    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
    # 定义一个函数，用于执行问答模型的前向推断
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.LongTensor] = None,  # 标签，用于计算答案起始位置的损失值
        end_positions: Optional[torch.LongTensor] = None,  # 标签，用于计算答案结束位置的损失值
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
    
        # 如果返回字典参数为None，则使用默认配置中的返回字典参数
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用MobileBERT模型进行前向传播
        outputs = self.mobilebert(
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

        # 将序列输出传入问答输出层，得到起始位置和结束位置的logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        
        # 计算总的损失值
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 使用交叉熵损失函数计算起始位置和结束位置的损失值
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        # 如果不需要返回字典，则返回起始位置的logits、结束位置的logits和模型的中间输出
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output
        
        # 如果需要返回字典，则返回总的损失值、起始位置的logits、结束位置的logits以及模型的中间隐藏状态和注意力权重
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 添加模型开始注释
@add_start_docstrings(
    """
    MobileBert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and
    a softmax) e.g. for RocStories/SWAG tasks.
    """,
    MOBILEBERT_START_DOCSTRING,
)
# 从 transformers.models.bert.modeling_bert.BertForMultipleChoice 复制，并将所有 Bert 替换为 MobileBert
class MobileBertForMultipleChoice(MobileBertPreTrainedModel):
    def __init__(self, config):
        # 调用父类的 __init__ 方法初始化模型
        super().__init__(config)

        # 初始化 MobileBertModel
        self.mobilebert = MobileBertModel(config)
        
        # 根据配置确定分类器的 dropout 概率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 创建分类器的 dropout 层
        self.dropout = nn.Dropout(classifier_dropout)
        # 创建分类器的全连接层
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    # 添加模型前向传播的开始注释
    @add_start_docstrings_to_model_forward(
        MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    # 添加模型输出示例的注释
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
    ):
        # 在此处添加代码
        pass
    ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 确定是否返回字典格式的输出，如果未提供，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 获取输入数据的选择数量
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 将输入的 input_ids 重新调整形状以适应模型的输入要求
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        # 将输入的 attention_mask 重新调整形状以适应模型的输入要求
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # 将输入的 token_type_ids 重新调整形状以适应模型的输入要求
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        # 将输入的 position_ids 重新调整形状以适应模型的输入要求
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        # 将输入的 inputs_embeds 重新调整形状以适应模型的输入要求
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 将输入传递给 MobileBERT 模型以获取输出
        outputs = self.mobilebert(
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

        # 从模型输出中提取汇总的输出
        pooled_output = outputs[1]

        # 对汇总输出进行 dropout
        pooled_output = self.dropout(pooled_output)
        # 将汇总输出传递给分类器以获取 logits
        logits = self.classifier(pooled_output)
        # 重新调整 logits 的形状以适应损失函数的要求
        reshaped_logits = logits.view(-1, num_choices)

        # 如果提供了标签，则计算损失
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 如果不返回字典格式的输出，则重新构建输出
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回字典格式的输出
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 在顶部增加文档字符串，描述了 MobileBert 模型及其在标记分类任务上的应用
@add_start_docstrings(
    """
    MobileBert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    MOBILEBERT_START_DOCSTRING,
)
# 从 transformers.models.bert.modeling_bert.BertForTokenClassification 复制代码，将 Bert 替换为 MobileBert，并进行全大写
class MobileBertForTokenClassification(MobileBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # 创建 MobileBertModel 对象，指定不添加池化层
        self.mobilebert = MobileBertModel(config, add_pooling_layer=False)
        # 获取分类器的丢弃率，如果未指定则使用隐藏层丢弃率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并进行最终处理
        self.post_init()

    # 添加文档字符串，描述模型的前向传播输入
    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 添加代码示例文档字符串，指定检查点、输出类型、配置类、期望输出和期望损失
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
    # 定义返回类型，可能是一个元组包含 torch.Tensor 的元组，或者是 TokenClassifierOutput 类型
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        # 形状为 (batch_size, sequence_length) 的标签张量，用于计算标记分类损失
        # 标签值应该在 `[0, ..., config.num_labels - 1]` 范围内
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 如果 return_dict 参数未被提供，则使用 self.config.use_return_dict 值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 mobilebert 模型前向传播方法，返回包含输出结果的元组
        outputs = self.mobilebert(
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

        # 提取输出结果中的序列输出张量
        sequence_output = outputs[0]

        # 对序列输出张量进行 Dropout 处理以减少过拟合
        sequence_output = self.dropout(sequence_output)
        # 使用分类器对序列输出张量进行分类，得到 logits 张量
        logits = self.classifier(sequence_output)

        # 初始化损失为 None
        loss = None
        # 如果提供了标签，则计算损失
        if labels is not None:
            # 使用交叉熵损失函数计算损失
            loss_fct = CrossEntropyLoss()
            # 计算损失值
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不需要返回字典，则返回 logits 和其他输出
        if not return_dict:
            output = (logits,) + outputs[2:]
            # 返回损失和输出，或仅输出
            return ((loss,) + output) if loss is not None else output

        # 返回 TokenClassifierOutput 类型的结果，其中包含损失、logits、隐藏状态和注意力
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```