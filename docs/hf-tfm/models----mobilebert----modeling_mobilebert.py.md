# `.\models\mobilebert\modeling_mobilebert.py`

```
# 导入必要的库和模块
import math  # 导入数学库，用于数学运算
import os  # 导入操作系统库，用于操作系统相关功能
import warnings  # 导入警告模块，用于处理警告信息
from dataclasses import dataclass  # 导入 dataclass 模块，用于创建数据类
from typing import Optional, Tuple, Union  # 导入类型提示相关模块

import torch  # 导入 PyTorch 模块
from torch import nn  # 导入神经网络模块
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss  # 导入损失函数

# 导入相关的模型输出类和工具函数
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
from .configuration_mobilebert import MobileBertConfig  # 导入 MobileBert 配置类

logger = logging.get_logger(__name__)  # 获取日志记录器

_CHECKPOINT_FOR_DOC = "google/mobilebert-uncased"  # 预训练模型的文档说明
_CONFIG_FOR_DOC = "MobileBertConfig"  # MobileBert 配置文档说明

# TokenClassification 文档字符串和期望输出
_CHECKPOINT_FOR_TOKEN_CLASSIFICATION = "mrm8488/mobilebert-finetuned-ner"
_TOKEN_CLASS_EXPECTED_OUTPUT = "['I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'I-LOC', 'O', 'I-LOC', 'I-LOC']"
_TOKEN_CLASS_EXPECTED_LOSS = 0.03

# QuestionAnswering 文档字符串和期望输出
_CHECKPOINT_FOR_QA = "csarron/mobilebert-uncased-squad-v2"
_QA_EXPECTED_OUTPUT = "'a nice puppet'"
_QA_EXPECTED_LOSS = 3.98
_QA_TARGET_START_INDEX = 12
_QA_TARGET_END_INDEX = 13

# SequenceClassification 文档字符串和期望输出
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "lordtt13/emo-mobilebert"
_SEQ_CLASS_EXPECTED_OUTPUT = "'others'"
_SEQ_CLASS_EXPECTED_LOSS = "4.72"

MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST = ["google/mobilebert-uncased"]  # 预训练模型的存档列表


def load_tf_weights_in_mobilebert(model, config, tf_checkpoint_path):
    ```
    加载 MobileBERT 模型的 TensorFlow 权重，并将它们转换为 PyTorch 模型权重。

    Args:
        model (PreTrainedModel): 要加载权重的 MobileBERT 模型实例。
        config (MobileBertConfig): MobileBERT 模型的配置对象。
        tf_checkpoint_path (str): TensorFlow 权重的路径。

    Returns:
        None

    Raises:
        ImportError: 如果导入 TensorFlow 失败。
        RuntimeError: 如果无法从 tf_checkpoint_path 加载权重。

    Example usage:
        ```python
        model = MobileBertModel.from_pretrained('google/mobilebert-uncased')
        config = MobileBertConfig.from_pretrained('google/mobilebert-uncased')
        load_tf_weights_in_mobilebert(model, config, 'path/to/tf_checkpoint')
        ```
    ```
    ```
    """Load tf checkpoints in a pytorch model."""
    # 加载 TensorFlow 检查点到 PyTorch 模型中

    try:
        import re  # 导入正则表达式模块
        import numpy as np  # 导入 NumPy 库
        import tensorflow as tf  # 导入 TensorFlow 库
    except ImportError:
        # 如果导入失败，记录错误信息并抛出异常
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise

    tf_path = os.path.abspath(tf_checkpoint_path)
    # 获取 TensorFlow 检查点文件的绝对路径

    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # 记录日志信息，指示正在从 TensorFlow 检查点文件 {tf_path} 转换

    # Load weights from TF model
    # 从 TensorFlow 模型中加载权重
    init_vars = tf.train.list_variables(tf_path)
    # 获取 TensorFlow 模型中所有变量列表

    names = []
    arrays = []
    for name, shape in init_vars:
        # 遍历每个变量名和其形状
        logger.info(f"Loading TF weight {name} with shape {shape}")
        # 记录日志信息，指示正在加载 TensorFlow 权重 {name}，形状为 {shape}
        array = tf.train.load_variable(tf_path, name)
        # 加载 TensorFlow 模型中的变量数据
        names.append(name)
        arrays.append(array)
    # 遍历names和arrays，每次迭代处理一个名字和对应的数组
    for name, array in zip(names, arrays):
        # 替换name中的特定字符串以简化模型参数名
        name = name.replace("ffn_layer", "ffn")
        name = name.replace("FakeLayerNorm", "LayerNorm")
        name = name.replace("extra_output_weights", "dense/kernel")
        name = name.replace("bert", "mobilebert")
        # 将name按"/"分割成列表
        name = name.split("/")
        
        # 检查name中是否包含不需要的变量名，若包含则跳过此次迭代
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        
        pointer = model
        
        # 遍历name中的每个部分，逐级访问model的属性
        for m_name in name:
            # 如果m_name匹配形如"A-Za-z+_\d+"的字符串，则按下划线分割为多个部分
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            
            # 根据scope_names的第一个部分选择指针位置
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
                    # 如果属性不存在，则记录日志并跳过当前迭代
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            
            # 如果scope_names有多个部分，则进一步访问指定位置的属性
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        
        # 如果m_name的结尾是"_embeddings"，则访问pointer的"weight"属性
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            # 如果m_name为"kernel"，则对array进行转置操作
            array = np.transpose(array)
        
        # 检查pointer的形状和array的形状是否相匹配，若不匹配则抛出异常
        try:
            assert (
                pointer.shape == array.shape
            ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        
        # 记录初始化操作的日志信息
        logger.info(f"Initialize PyTorch weight {name}")
        
        # 将array转换为torch.Tensor，并赋值给pointer的data属性
        pointer.data = torch.from_numpy(array)
    
    # 返回处理后的模型
    return model
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,


        # 初始化 MobileBertEmbeddings 类的实例
        super().__init__()
        # 设置是否使用三元输入（trigram_input）和嵌入大小（embedding_size）
        self.trigram_input = config.trigram_input
        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size

        # 初始化词嵌入（word_embeddings），位置嵌入（position_embeddings），和类型嵌入（token_type_embeddings）
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 如果使用三元输入，嵌入维度乘以3，否则为1
        embed_dim_multiplier = 3 if self.trigram_input else 1
        embedded_input_size = self.embedding_size * embed_dim_multiplier

        # 定义嵌入转换层，将输入嵌入映射到隐藏大小
        self.embedding_transformation = nn.Linear(embedded_input_size, config.hidden_size)

        # 初始化归一化层（LayerNorm）和 dropout 层
        self.LayerNorm = NORM2FN[config.normalization_type](config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 注册位置 ID 张量，用于序列化时持久化存储
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
    ) -> torch.Tensor:
        # 如果输入的 input_ids 不为 None，则获取其形状作为 input_shape
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            # 否则，获取 inputs_embeds 的形状去掉最后一个维度作为 input_shape
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度，即 input_shape 的第二个维度
        seq_length = input_shape[1]

        # 如果未提供 position_ids，则使用预设的 position_ids 切片，长度为 seq_length
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # 如果未提供 token_type_ids，则创建一个与 input_shape 相同的零张量作为 token_type_ids
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果未提供 inputs_embeds，则使用 input_ids 通过 word_embeddings 获取其嵌入表示
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # 如果启用了 trigram_input
        if self.trigram_input:
            # 根据 MobileBERT 论文中的描述，对输入的嵌入进行扩展处理
            inputs_embeds = torch.cat(
                [
                    nn.functional.pad(inputs_embeds[:, 1:], [0, 0, 0, 1, 0, 0], value=0.0),
                    inputs_embeds,
                    nn.functional.pad(inputs_embeds[:, :-1], [0, 0, 1, 0, 0, 0], value=0.0),
                ],
                dim=2,
            )

        # 如果启用了 trigram_input 或者嵌入维度不等于隐藏层维度
        if self.trigram_input or self.embedding_size != self.hidden_size:
            # 对输入的嵌入进行额外的变换处理
            inputs_embeds = self.embedding_transformation(inputs_embeds)

        # 添加位置嵌入和 token 类型嵌入到输入嵌入中
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings

        # 对嵌入结果进行层归一化
        embeddings = self.LayerNorm(embeddings)

        # 对归一化后的嵌入结果进行 Dropout 处理
        embeddings = self.dropout(embeddings)

        # 返回处理后的嵌入结果
        return embeddings
class MobileBertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads  # 设置注意力头的数量
        self.attention_head_size = int(config.true_hidden_size / config.num_attention_heads)  # 计算每个注意力头的大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 计算所有注意力头的总大小

        self.query = nn.Linear(config.true_hidden_size, self.all_head_size)  # 创建查询线性层
        self.key = nn.Linear(config.true_hidden_size, self.all_head_size)  # 创建键线性层
        self.value = nn.Linear(
            config.true_hidden_size if config.use_bottleneck_attention else config.hidden_size, self.all_head_size
        )  # 创建值线性层，根据是否使用瓶颈注意力选择隐藏大小

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)  # 创建Dropout层用于注意力概率

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)  # 调整张量形状以适应多头注意力计算
        return x.permute(0, 2, 1, 3)  # 转置张量以便进行注意力得分计算

    def forward(
        self,
        query_tensor: torch.Tensor,
        key_tensor: torch.Tensor,
        value_tensor: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(query_tensor)  # 计算混合查询层
        mixed_key_layer = self.key(key_tensor)  # 计算混合键层
        mixed_value_layer = self.value(value_tensor)  # 计算混合值层

        query_layer = self.transpose_for_scores(mixed_query_layer)  # 转置并准备查询张量
        key_layer = self.transpose_for_scores(mixed_key_layer)  # 转置并准备键张量
        value_layer = self.transpose_for_scores(mixed_value_layer)  # 转置并准备值张量

        # 计算原始的注意力分数，即查询与键的点积
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # 缩放注意力分数的平方根

        if attention_mask is not None:
            # 应用预计算的注意力掩码（适用于BertModel的所有层）
            attention_scores = attention_scores + attention_mask

        # 将注意力分数规范化为概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 通过Dropout层来实现随机遮盖整个待注意的标记，这在原始Transformer论文中也有提到
        attention_probs = self.dropout(attention_probs)

        # 如果需要，掩盖特定的注意力头
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)  # 计算上下文张量
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # 调整上下文张量的维度顺序

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # 调整上下文张量的形状
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)  # 准备输出结果

        return outputs
    # 初始化方法，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 根据配置决定是否使用瓶颈层
        self.use_bottleneck = config.use_bottleneck
        # 创建一个线性层，输入和输出大小都是 true_hidden_size
        self.dense = nn.Linear(config.true_hidden_size, config.true_hidden_size)
        # 根据配置选择合适的归一化方法，并初始化
        self.LayerNorm = NORM2FN[config.normalization_type](config.true_hidden_size, eps=config.layer_norm_eps)
        # 如果不使用瓶颈层，则初始化一个丢弃层
        if not self.use_bottleneck:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播方法，接收隐藏状态和残差张量作为输入，返回一个张量
    def forward(self, hidden_states: torch.Tensor, residual_tensor: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态输入到线性层中进行计算
        layer_outputs = self.dense(hidden_states)
        # 如果没有使用瓶颈层，则对输出进行丢弃操作
        if not self.use_bottleneck:
            layer_outputs = self.dropout(layer_outputs)
        # 将丢弃后的输出与残差张量相加，并通过 LayerNorm 进行归一化处理
        layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        # 返回处理后的输出张量
        return layer_outputs
# MobileBertAttention 类定义
class MobileBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化 self 层，即 MobileBertSelfAttention 类的实例化
        self.self = MobileBertSelfAttention(config)
        # 初始化 output 层，即 MobileBertSelfOutput 类的实例化
        self.output = MobileBertSelfOutput(config)
        # 初始化一个空集合，用于存储已剪枝的注意力头的索引
        self.pruned_heads = set()

    # 剪枝注意力头的方法
    def prune_heads(self, heads):
        # 如果 heads 列表为空，则直接返回
        if len(heads) == 0:
            return
        # 调用 find_pruneable_heads_and_indices 函数找到可剪枝的注意力头及其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储剪枝后的头部索引
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 前向传播方法
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
        # 使用 self 层进行自注意力计算
        self_outputs = self.self(
            query_tensor,
            key_tensor,
            value_tensor,
            attention_mask,
            head_mask,
            output_attentions,
        )
        # 将 self 层的输出经过 output 层的线性投影并添加残差连接
        attention_output = self.output(self_outputs[0], layer_input)
        # 如果需要输出注意力权重，将它们添加到输出中
        outputs = (attention_output,) + self_outputs[1:]  # 如果需要输出注意力权重，则将它们添加到输出中
        return outputs


# MobileBertIntermediate 类定义
class MobileBertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 线性层，用于转换隐藏状态的维度
        self.dense = nn.Linear(config.true_hidden_size, config.intermediate_size)
        # 判断 config.hidden_act 是否是字符串，选择对应的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播方法
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 经过线性层进行维度转换
        hidden_states = self.dense(hidden_states)
        # 经过激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# OutputBottleneck 类定义
class OutputBottleneck(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 线性层，用于输出的维度转换
        self.dense = nn.Linear(config.true_hidden_size, config.hidden_size)
        # 归一化层，根据 config.normalization_type 选择对应的归一化函数
        self.LayerNorm = NORM2FN[config.normalization_type](config.hidden_size, eps=config.layer_norm_eps)
        # Dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    # 定义前向传播方法，接受隐藏状态和残差张量作为输入，返回处理后的张量
    def forward(self, hidden_states: torch.Tensor, residual_tensor: torch.Tensor) -> torch.Tensor:
        # 使用全连接层对隐藏状态进行线性变换
        layer_outputs = self.dense(hidden_states)
        # 对线性变换后的结果应用丢弃部分神经元的dropout操作
        layer_outputs = self.dropout(layer_outputs)
        # 将dropout后的结果与残差张量相加，并通过层归一化处理
        layer_outputs = self.LayerNorm(layer_outputs + residual_tensor)
        # 返回处理后的层输出张量
        return layer_outputs
# 定义 MobileBertOutput 类，继承自 nn.Module，用于处理 MobileBERT 模型的输出
class MobileBertOutput(nn.Module):
    # 初始化方法，接收一个配置对象 config
    def __init__(self, config):
        super().__init__()
        # 根据配置确定是否使用瓶颈层
        self.use_bottleneck = config.use_bottleneck
        # 创建一个线性层，将 intermediate_size 映射到 true_hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.true_hidden_size)
        # 根据配置选择规范化层类型并初始化
        self.LayerNorm = NORM2FN[config.normalization_type](config.true_hidden_size)
        # 如果不使用瓶颈层，则创建一个丢弃层，用于随机丢弃节点以防止过拟合
        if not self.use_bottleneck:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 如果使用瓶颈层，则创建一个输出瓶颈对象
        else:
            self.bottleneck = OutputBottleneck(config)

    # 前向传播方法，接收三个张量作为输入，返回一个张量
    def forward(
        self, intermediate_states: torch.Tensor, residual_tensor_1: torch.Tensor, residual_tensor_2: torch.Tensor
    ) -> torch.Tensor:
        # 将 intermediate_states 输入到线性层中得到 layer_output
        layer_output = self.dense(intermediate_states)
        # 如果不使用瓶颈层，则对 layer_output 进行丢弃操作
        if not self.use_bottleneck:
            layer_output = self.dropout(layer_output)
            # 将丢弃后的输出与 residual_tensor_1 相加，并通过规范化层 LayerNorm 处理
            layer_output = self.LayerNorm(layer_output + residual_tensor_1)
        # 如果使用瓶颈层，则直接使用瓶颈层处理 layer_output 和 residual_tensor_2
        else:
            layer_output = self.LayerNorm(layer_output + residual_tensor_1)
            layer_output = self.bottleneck(layer_output, residual_tensor_2)
        # 返回处理后的输出张量
        return layer_output


# 定义 BottleneckLayer 类，继承自 nn.Module，用于 MobileBERT 中的瓶颈层处理
class BottleneckLayer(nn.Module):
    # 初始化方法，接收一个配置对象 config
    def __init__(self, config):
        super().__init__()
        # 创建一个线性层，将 hidden_size 映射到 intra_bottleneck_size
        self.dense = nn.Linear(config.hidden_size, config.intra_bottleneck_size)
        # 根据配置选择规范化层类型并初始化
        self.LayerNorm = NORM2FN[config.normalization_type](config.intra_bottleneck_size, eps=config.layer_norm_eps)

    # 前向传播方法，接收一个张量作为输入，返回一个张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将 hidden_states 输入到线性层中得到 layer_input
        layer_input = self.dense(hidden_states)
        # 通过规范化层 LayerNorm 处理 layer_input
        layer_input = self.LayerNorm(layer_input)
        # 返回处理后的输出张量
        return layer_input


# 定义 Bottleneck 类，继承自 nn.Module，用于 MobileBERT 中的瓶颈处理
class Bottleneck(nn.Module):
    # 初始化方法，接收一个配置对象 config
    def __init__(self, config):
        super().__init__()
        # 根据配置确定是否共享键值查询瓶颈
        self.key_query_shared_bottleneck = config.key_query_shared_bottleneck
        # 根据配置确定是否使用瓶颈注意力
        self.use_bottleneck_attention = config.use_bottleneck_attention
        # 创建一个输入瓶颈层对象
        self.input = BottleneckLayer(config)
        # 如果共享键值查询瓶颈，则创建一个瓶颈注意力层对象
        if self.key_query_shared_bottleneck:
            self.attention = BottleneckLayer(config)
    # 定义一个方法 `forward`，接收一个名为 `hidden_states` 的张量参数，并返回一个元组
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor]:
        # 该方法可以返回三种不同的元组值。这些不同的值利用了瓶颈层，
        # 这些线性层用于将隐藏状态投影到低维向量，从而降低内存使用。
        # 这些线性层在训练过程中学习其权重。

        # 如果 `config.use_bottleneck_attention` 为真，则返回经过瓶颈层处理的四个值，
        # 分别用于注意力层中的键、查询、值以及“层输入”。
        # 这个瓶颈层用于投影隐藏状态。层输入将作为注意力自我输出的残差张量使用，
        # 在计算注意力分数后添加到输出中。

        # 如果不使用 `config.use_bottleneck_attention`，但使用了 `config.key_query_shared_bottleneck`，
        # 则返回四个值，其中三个值经过了瓶颈层处理：查询和键通过相同的瓶颈层处理，
        # 而残差层则通过另一个瓶颈层处理，将应用于注意力自我输出。

        # 最后，如果都不满足，则查询、键和值的值为未经过瓶颈处理的隐藏状态，
        # 而残差层为经过瓶颈处理的隐藏状态。

        # 使用 `self.input` 方法对隐藏状态进行瓶颈处理
        bottlenecked_hidden_states = self.input(hidden_states)
        # 根据条件返回相应的元组值
        if self.use_bottleneck_attention:
            return (bottlenecked_hidden_states,) * 4
        elif self.key_query_shared_bottleneck:
            shared_attention_input = self.attention(hidden_states)
            return (shared_attention_input, shared_attention_input, hidden_states, bottlenecked_hidden_states)
        else:
            return (hidden_states, hidden_states, hidden_states, bottlenecked_hidden_states)
# MobileBertLayer 类定义，继承自 nn.Module
class MobileBertLayer(nn.Module):
    def __init__(self, config):
        # 调用父类构造函数进行初始化
        super().__init__()
        
        # 根据配置文件初始化各种属性
        self.use_bottleneck = config.use_bottleneck
        self.num_feedforward_networks = config.num_feedforward_networks
        
        # 创建 MobileBertAttention 对象，用于处理注意力机制
        self.attention = MobileBertAttention(config)
        
        # 创建 MobileBertIntermediate 对象，用于处理中间层输出
        self.intermediate = MobileBertIntermediate(config)
        
        # 创建 MobileBertOutput 对象，用于处理最终输出
        self.output = MobileBertOutput(config)
        
        # 如果配置中设置了使用瓶颈层，则创建 Bottleneck 对象
        if self.use_bottleneck:
            self.bottleneck = Bottleneck(config)
        
        # 如果配置中指定了多个前馈网络，则创建对应数量的 FFNLayer 对象组成列表
        if config.num_feedforward_networks > 1:
            self.ffn = nn.ModuleList([FFNLayer(config) for _ in range(config.num_feedforward_networks - 1)])

    # 前向传播函数定义，接收输入 hidden_states 和可选的各种掩码、标志
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        ```
    # 定义一个方法，接收隐藏状态、注意力掩码、头部掩码、是否输出注意力权重等参数，返回一个元组类型的 torch.Tensor
        ) -> Tuple[torch.Tensor]:
            # 如果使用瓶颈模块
            if self.use_bottleneck:
                # 使用瓶颈模块处理隐藏状态，返回查询张量、键张量、值张量和层输入
                query_tensor, key_tensor, value_tensor, layer_input = self.bottleneck(hidden_states)
            else:
                # 否则直接复制隐藏状态到查询张量、键张量、值张量、层输入
                query_tensor, key_tensor, value_tensor, layer_input = [hidden_states] * 4
    
            # 调用 self.attention 方法，处理查询张量、键张量、值张量、层输入、注意力掩码、头部掩码等参数，获取自注意力输出
            self_attention_outputs = self.attention(
                query_tensor,
                key_tensor,
                value_tensor,
                layer_input,
                attention_mask,
                head_mask,
                output_attentions=output_attentions,
            )
            # 获取自注意力输出的第一个元素作为 attention_output
            attention_output = self_attention_outputs[0]
            # 创建一个元组 s，包含 attention_output
            s = (attention_output,)
            # 如果输出注意力权重，则将其添加到 outputs 中
            outputs = self_attention_outputs[1:]  # 如果我们输出注意力权重，添加自注意力权重
    
            # 如果存在多个前馈网络
            if self.num_feedforward_networks != 1:
                # 对于每个前馈网络模块 ffn_module 在 self.ffn 中的枚举 i
                for i, ffn_module in enumerate(self.ffn):
                    # 使用前馈网络模块处理 attention_output
                    attention_output = ffn_module(attention_output)
                    # 将处理后的 attention_output 添加到元组 s 中
                    s += (attention_output,)
    
            # 使用 intermediate 方法处理 attention_output，得到 intermediate_output
            intermediate_output = self.intermediate(attention_output)
            # 使用 output 方法处理 intermediate_output、attention_output 和 hidden_states，得到 layer_output
            layer_output = self.output(intermediate_output, attention_output, hidden_states)
            # 构建 outputs 元组，包含 layer_output 和之前的 outputs、固定的一些张量数据和 s 元组
            outputs = (
                (layer_output,)
                + outputs
                + (
                    torch.tensor(1000),  # 固定值 1000
                    query_tensor,  # 查询张量
                    key_tensor,  # 键张量
                    value_tensor,  # 值张量
                    layer_input,  # 层输入
                    attention_output,  # 自注意力输出
                    intermediate_output,  # intermediate 输出
                )
                + s
            )
            # 返回 outputs 结果
            return outputs
class MobileBertEncoder(nn.Module):
    # MobileBERT 编码器模型，继承自 nn.Module 类
    def __init__(self, config):
        super().__init__()
        # 初始化 MobileBERT 编码器的层列表，每层由 MobileBertLayer 组成
        self.layer = nn.ModuleList([MobileBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutput]:
        # 如果需要输出隐藏状态，则初始化空元组 all_hidden_states
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力权重，则初始化空元组 all_attentions
        all_attentions = () if output_attentions else None
        # 遍历每一层 MobileBERT 编码器
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 调用当前层的 forward 方法，计算层的输出
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                head_mask[i],
                output_attentions,
            )
            # 更新隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力权重，将当前层的注意力权重添加到 all_attentions 中
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # 添加最后一层的隐藏状态到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典形式的输出，则根据需要返回不同的元组或 BaseModelOutput 对象
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


class MobileBertPooler(nn.Module):
    # MobileBERT 池化层模型，继承自 nn.Module 类
    def __init__(self, config):
        super().__init__()
        # 根据配置文件选择是否激活分类器激活函数
        self.do_activate = config.classifier_activation
        if self.do_activate:
            # 如果需要激活，初始化一个线性层 dense
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 池化模型通过简单地选择第一个 token 对应的隐藏状态来实现
        first_token_tensor = hidden_states[:, 0]
        # 如果不需要激活，则直接返回第一个 token 的隐藏状态
        if not self.do_activate:
            return first_token_tensor
        else:
            # 否则，通过线性层和 tanh 激活函数计算池化输出
            pooled_output = self.dense(first_token_tensor)
            pooled_output = torch.tanh(pooled_output)
            return pooled_output


class MobileBertPredictionHeadTransform(nn.Module):
    # MobileBERT 预测头转换层模型，继承自 nn.Module 类
    def __init__(self, config):
        super().__init__()
        # 初始化线性层 dense，用于特征变换
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 根据配置选择激活函数，支持字符串形式和函数形式
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # 初始化 LayerNorm 层，用于归一化
        self.LayerNorm = NORM2FN["layer_norm"](config.hidden_size, eps=config.layer_norm_eps)
    # 定义一个方法 `forward`，用于前向传播计算
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用全连接层 `self.dense` 对输入的隐藏状态进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对变换后的隐藏状态应用激活函数 `self.transform_act_fn`
        hidden_states = self.transform_act_fn(hidden_states)
        # 对激活后的隐藏状态进行层归一化 `self.LayerNorm`
        hidden_states = self.LayerNorm(hidden_states)
        # 返回处理后的隐藏状态作为输出
        return hidden_states
# MobileBertLMPredictionHead 类定义，继承自 nn.Module
class MobileBertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用 MobileBertPredictionHeadTransform 类对隐藏状态进行转换
        self.transform = MobileBertPredictionHeadTransform(config)
        # 创建一个线性层，用于预测输出权重，输入维度为词汇表大小，输出维度为隐藏大小减去嵌入大小，无偏置
        self.dense = nn.Linear(config.vocab_size, config.hidden_size - config.embedding_size, bias=False)
        # 创建一个线性层，用于预测输出偏置，输入维度为嵌入大小，输出维度为词汇表大小，无偏置
        self.decoder = nn.Linear(config.embedding_size, config.vocab_size, bias=False)
        # 创建一个参数化的偏置向量，维度为词汇表大小
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        # 确保输出偏置与预测层的偏置相连接，以便与 `resize_token_embeddings` 方法正确调整大小
        self.decoder.bias = self.bias

    # 前向传播函数，接受隐藏状态输入，返回预测分数张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 对隐藏状态进行转换
        hidden_states = self.transform(hidden_states)
        # 计算预测分数
        hidden_states = hidden_states.matmul(torch.cat([self.decoder.weight.t(), self.dense.weight], dim=0))
        # 加上预测偏置
        hidden_states += self.decoder.bias
        return hidden_states


# MobileBertOnlyMLMHead 类定义，继承自 nn.Module
class MobileBertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建 MobileBertLMPredictionHead 实例作为预测
        self.predictions = MobileBertLMPredictionHead(config)

    # 前向传播函数，接受序列输出输入，返回预测分数
    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        # 调用预测头进行预测
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


# MobileBertPreTrainingHeads 类定义，继承自 nn.Module
class MobileBertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建 MobileBertLMPredictionHead 实例作为预测
        self.predictions = MobileBertLMPredictionHead(config)
        # 创建线性层，用于序列关系分类，输入维度为隐藏大小，输出维度为2（二元分类）
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    # 前向传播函数，接受序列输出和池化输出输入，返回预测分数和序列关系分数的元组
    def forward(self, sequence_output: torch.Tensor, pooled_output: torch.Tensor) -> Tuple[torch.Tensor]:
        # 调用预测头进行预测
        prediction_scores = self.predictions(sequence_output)
        # 计算序列关系分数
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


# MobileBertPreTrainedModel 类定义，继承自 PreTrainedModel
class MobileBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定配置类为 MobileBertConfig
    config_class = MobileBertConfig
    # 预训练模型归档映射
    pretrained_model_archive_map = MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST
    # 加载 TensorFlow 权重的方法
    load_tf_weights = load_tf_weights_in_mobilebert
    # 基础模型前缀名
    base_model_prefix = "mobilebert"
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果 module 是 nn.Linear 类型
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重，均值为 0.0，标准差为模型配置中的 initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置项，将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果 module 是 nn.Embedding 类型
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重，均值为 0.0，标准差为模型配置中的 initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果指定了 padding_idx，则将对应位置的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果 module 是 nn.LayerNorm 或者 NoNorm 类型
        elif isinstance(module, (nn.LayerNorm, NoNorm)):
            # 将偏置项初始化为零
            module.bias.data.zero_()
            # 将权重初始化为全 1.0
            module.weight.data.fill_(1.0)
@dataclass
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


MOBILEBERT_START_DOCSTRING = r"""
    Docstring for `MobileBertForPreTrainingOutput`.

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

MOBILEBERT_INPUTS_DOCSTRING = r"""
    Docstring for `MOBILEBERT_INPUTS_DOCSTRING`.

    """
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列的标记索引，在词汇表中找到对应的标记
            Indices of input sequence tokens in the vocabulary.

            # 可以使用 `AutoTokenizer` 获取这些索引。详见 `PreTrainedTokenizer.encode` 和 `PreTrainedTokenizer.__call__`。
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 用来避免在填充的标记索引上执行注意力机制，值为 `[0, 1]`：

            - 1 表示 **未被屏蔽** 的标记，
            - 0 表示 **被屏蔽** 的标记。
            
            # 注意屏蔽令牌的作用是什么？
            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 指示输入的第一部分和第二部分的分段标记索引。索引选在 `[0, 1]`：

            - 0 对应 *句子 A* 的标记，
            - 1 对应 *句子 B* 的标记。
            
            # 什么是分段标记 ID？
            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 每个输入序列标记在位置嵌入中的位置索引。选在范围 `[0, config.max_position_embeddings - 1]`。

            # 什么是位置 ID？
            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于屏蔽自注意力模块中的特定头部的掩码。掩码值选在 `[0, 1]`：

            - 1 表示头部 **未被屏蔽**，
            - 0 表示头部 **被屏蔽**。
            
            # 控制自注意力头部屏蔽的作用是什么？
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选，可以直接传递嵌入表示，而不是传递 `input_ids`。如果需要更多对转换 `input_ids` 到相关向量的控制权，则很有用。
            # 这对于控制模型内部嵌入查找矩阵的转换方式很有用。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。详见返回张量中的 `attentions` 以获取更多细节。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。详见返回张量中的 `hidden_states` 以获取更多细节。
        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""
The bare MobileBert Model transformer outputting raw hidden-states without any specific head on top.
"""
@add_start_docstrings(
    "The bare MobileBert Model transformer outputting raw hidden-states without any specific head on top.",
    MOBILEBERT_START_DOCSTRING,
)
class MobileBertModel(MobileBertPreTrainedModel):
    """
    MobileBertModel class implementing the MobileBERT architecture.

    https://arxiv.org/pdf/2004.02984.pdf
    """

    def __init__(self, config, add_pooling_layer=True):
        """
        Initializes a MobileBertModel instance.

        Args:
            config (MobileBertConfig): Configuration class for MobileBERT.
            add_pooling_layer (bool): Whether to add a pooling layer. Defaults to True.
        """
        super().__init__(config)
        self.config = config
        self.embeddings = MobileBertEmbeddings(config)  # Initialize MobileBertEmbeddings layer
        self.encoder = MobileBertEncoder(config)        # Initialize MobileBertEncoder layer

        self.pooler = MobileBertPooler(config) if add_pooling_layer else None  # Initialize MobileBertPooler if add_pooling_layer is True

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        """
        Retrieves the input word embeddings from MobileBertEmbeddings.

        Returns:
            torch.nn.Embedding: The word embedding layer.
        """
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        """
        Sets the input word embeddings in MobileBertEmbeddings.

        Args:
            value (torch.Tensor): New tensor for word embeddings.
        """
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model.

        Args:
            heads_to_prune (dict): Dictionary of {layer_num: list of heads to prune in this layer}.

        See base class PreTrainedModel.
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(
        MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
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
        ):
        """
        Forward pass for the MobileBertModel.

        Args:
            input_ids (Optional[torch.LongTensor]): Input ids of shape (batch_size, sequence_length).
            attention_mask (Optional[torch.FloatTensor]): Attention mask of shape (batch_size, sequence_length).
            token_type_ids (Optional[torch.LongTensor]): Token type ids of shape (batch_size, sequence_length).
            position_ids (Optional[torch.LongTensor]): Position ids of shape (batch_size, sequence_length).
            head_mask (Optional[torch.FloatTensor]): Mask to nullify selected heads of shape (num_heads,).
            inputs_embeds (Optional[torch.FloatTensor]): Embedded inputs of shape (batch_size, sequence_length, embedding_size).
            output_hidden_states (Optional[bool]): Whether to return hidden states.
            output_attentions (Optional[bool]): Whether to return attentions.
            return_dict (Optional[bool]): Whether to return a dictionary.

        Returns:
            BaseModelOutputWithPooling or tuple:
                BaseModelOutputWithPooling if output_hidden_states=False and output_attentions=False
                tuple (torch.FloatTensor, ...) otherwise

        """
        # Forward pass logic goes here
        pass
        # 如果未指定输出注意力，使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定输出隐藏状态，使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定返回字典，使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果同时指定了 input_ids 和 inputs_embeds，则抛出 ValueError
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            # 如果指定了 input_ids，则检查是否存在填充并没有注意力掩码
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            # 获取 input_ids 的形状
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            # 如果指定了 inputs_embeds，则获取其形状的前几维
            input_shape = inputs_embeds.size()[:-1]
        else:
            # 如果既未指定 input_ids 也未指定 inputs_embeds，则抛出 ValueError
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 确定设备是 input_ids 的设备还是 inputs_embeds 的设备
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # 如果 attention_mask 未指定，则创建全为 1 的注意力掩码张量
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        # 如果 token_type_ids 未指定，则创建全为 0 的张量作为 token 类型 IDs
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # 生成扩展的注意力掩码张量，以匹配多头注意力的维度需求
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # 准备头部掩码（如果需要）
        # 在头部掩码中，1.0 表示保留该头部
        # attention_probs 的形状为 bsz x n_heads x N x N
        # input head_mask 的形状为 [num_heads] 或 [num_hidden_layers x num_heads]
        # head_mask 被转换为形状 [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # 生成嵌入输出
        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        # 编码器的输出
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
        # 如果定义了池化器，生成池化输出；否则为 None
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        # 如果不要求返回字典，则返回一个元组
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        # 如果要求返回字典，则返回一个 BaseModelOutputWithPooling 对象
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
@add_start_docstrings(
    """
    MobileBert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a
    `next sentence prediction (classification)` head.
    """,
    MOBILEBERT_START_DOCSTRING,
)
class MobileBertForPreTraining(MobileBertPreTrainedModel):
    _tied_weights_keys = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)
        # 初始化 MobileBert 模型
        self.mobilebert = MobileBertModel(config)
        # 初始化 MobileBert 的预训练头部
        self.cls = MobileBertPreTrainingHeads(config)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_output_embeddings(self):
        # 返回预测头部的解码器，用于输出嵌入
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        # 设置新的输出嵌入到预测头部的解码器中
        self.cls.predictions.decoder = new_embeddings

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> nn.Embedding:
        # 调整标记嵌入的大小，首先调整密集输出嵌入
        self.cls.predictions.dense = self._get_resized_lm_head(
            self.cls.predictions.dense, new_num_tokens=new_num_tokens, transposed=True
        )

        # 调用父类的方法来调整标记嵌入的大小
        return super().resize_token_embeddings(new_num_tokens=new_num_tokens)

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
):
    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> nn.Embedding:
        # 调整模型中的token嵌入大小，首先调整密集输出层的嵌入
        self.cls.predictions.dense = self._get_resized_lm_head(
            self.cls.predictions.dense, new_num_tokens=new_num_tokens, transposed=True
        )
        # 调用父类方法以完成token嵌入的调整，并返回调整后的嵌入层
        return super().resize_token_embeddings(new_num_tokens=new_num_tokens)

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
        # 如果return_dict为None，则使用配置文件中的默认设置来确定是否返回字典形式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用MobileBERT模型，传入各种输入参数，并获取输出结果
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

        # 从模型的输出中提取序列输出
        sequence_output = outputs[0]
        # 使用预测头部对序列输出进行预测得分计算
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        # 如果提供了标签，则计算掩码语言建模损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # 使用交叉熵损失函数，-100索引表示填充标记
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果不返回字典形式的输出，则组装输出结果并返回
        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 返回MaskedLMOutput对象，包括损失、预测logits、隐藏状态和注意力分布
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 创建一个新的类 MobileBertOnlyNSPHead，继承自 nn.Module
class MobileBertOnlyNSPHead(nn.Module):
    # 初始化方法，接受一个配置参数 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 定义一个线性层，用于预测下一个句子的关系，输入大小为 config.hidden_size，输出大小为 2
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    # 前向传播方法，接受一个 Tensor 参数 pooled_output，返回预测的下一个句子关系的分数 Tensor
    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        # 计算下一个句子关系的分数，使用 seq_relationship 线性层
        seq_relationship_score = self.seq_relationship(pooled_output)
        # 返回计算得到的分数 Tensor
        return seq_relationship_score


# 添加文档字符串和注解到 MobileBertForNextSentencePrediction 类
@add_start_docstrings(
    """MobileBert Model with a `next sentence prediction (classification)` head on top.""",
    MOBILEBERT_START_DOCSTRING,
)
# 定义 MobileBertForNextSentencePrediction 类，继承自 MobileBertPreTrainedModel
class MobileBertForNextSentencePrediction(MobileBertPreTrainedModel):
    # 初始化方法，接受一个配置参数 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建一个 MobileBertModel 对象，传入配置参数 config
        self.mobilebert = MobileBertModel(config)
        # 创建一个 MobileBertOnlyNSPHead 对象，传入配置参数 config
        self.cls = MobileBertOnlyNSPHead(config)

        # 调用额外的初始化方法来初始化权重并应用最终处理
        self.post_init()

    # 添加文档字符串和注解到 forward 方法，描述输入参数和返回输出
    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 替换返回文档字符串中的输出类型为 NextSentencePredictorOutput，使用指定的配置类 _CONFIG_FOR_DOC
    @replace_return_docstrings(output_type=NextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
    # 前向传播方法，接受多个可选的 Tensor 输入参数和其他关键字参数
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
            (see `input_ids` docstring) Indices should be in `[0, 1]`.

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.

        Returns:
            Depending on `return_dict`:
            - If `return_dict` is `False`, returns a tuple with `seq_relationship_score` and additional outputs.
            - If `return_dict` is `True`, returns a `NextSentencePredictorOutput` object.

        Examples:
        Example usage of the `MobileBertForNextSentencePrediction` model.

        ```python
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
            # Issue a warning that the argument `next_sentence_label` is deprecated and suggest using `labels` instead
            warnings.warn(
                "The `next_sentence_label` argument is deprecated and will be removed in a future version, use"
                " `labels` instead.",
                FutureWarning,
            )
            # Replace `next_sentence_label` with `labels` if found in kwargs
            labels = kwargs.pop("next_sentence_label")

        # Determine whether to return a dictionary based on the provided argument or the default setting
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass inputs through the MobileBERT model for next sentence prediction
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

        # Extract the pooled output from MobileBERT's outputs
        pooled_output = outputs[1]
        # Compute scores for next sentence prediction using a classification layer
        seq_relationship_score = self.cls(pooled_output)

        next_sentence_loss = None
        # Compute loss if labels are provided
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), labels.view(-1))

        # Decide the output format based on `return_dict`
        if not return_dict:
            # Return a tuple with `seq_relationship_score` and optionally other outputs
            output = (seq_relationship_score,) + outputs[2:]
            return ((next_sentence_loss,) + output) if next_sentence_loss is not None else output

        # Return a `NextSentencePredictorOutput` object containing loss, logits, hidden states, and attentions
        return NextSentencePredictorOutput(
            loss=next_sentence_loss,
            logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# MobileBert 模型变换器，顶部带有序列分类/回归头部（线性层在池化输出之上），例如用于 GLUE 任务。
# 此类从 transformers.models.bert.modeling_bert.BertForSequenceClassification 复制，将 Bert 替换为 MobileBert 并全小写处理。
class MobileBertForSequenceClassification(MobileBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels  # 设置标签数量
        self.config = config

        self.mobilebert = MobileBertModel(config)  # 初始化 MobileBert 模型
        # 根据配置初始化分类器的丢弃率，若未指定，则使用隐藏层丢弃率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)  # 定义丢弃层
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)  # 定义分类器线性层

        # 初始化权重并应用最终处理
        self.post_init()

    # 添加输入文档字符串，描述模型前向传播的输入参数
    # 从 MOBILEBERT_INPUTS_DOCSTRING 格式化得到输入参数的说明
    # 添加代码示例文档字符串，包含序列分类检查点、输出类型、配置类和预期输出/损失
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
        # 设置返回字典，如果未指定则使用模型配置中的默认值
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

        # 从模型输出中获取汇聚输出（pooled output）
        pooled_output = outputs[1]

        # 对汇聚输出进行dropout处理
        pooled_output = self.dropout(pooled_output)
        # 使用分类器进行分类任务的预测
        logits = self.classifier(pooled_output)

        # 初始化损失为None
        loss = None
        # 如果提供了标签，计算损失
        if labels is not None:
            # 如果问题类型未指定，则根据标签类型和类别数确定问题类型
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

        # 如果不要求返回字典，则组织输出结果为元组
        if not return_dict:
            output = (logits,) + outputs[2:]  # 包括额外的hidden_states
            return ((loss,) + output) if loss is not None else output

        # 返回包含损失和其他输出的SequenceClassifierOutput对象
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    MobileBert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a
    linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    MOBILEBERT_START_DOCSTRING,
)
# 从transformers.models.bert.modeling_bert.BertForQuestionAnswering复制过来，将Bert改为MobileBert全大写
class MobileBertForQuestionAnswering(MobileBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 设置分类任务的标签数目
        self.num_labels = config.num_labels

        # 创建MobileBert模型，不包含池化层
        self.mobilebert = MobileBertModel(config, add_pooling_layer=False)
        # 创建线性层，用于输出分类标签
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

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
    # 定义前向传播函数，处理输入并返回模型输出
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
        # 如果 return_dict 参数为 None，则使用 self.config.use_return_dict 决定是否返回字典形式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 MobileBERT 模型进行前向传播
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

        # 将序列输出传入 QA 输出层，得到起始和结束位置的 logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()  # 去除多余的维度并保持连续性
        end_logits = end_logits.squeeze(-1).contiguous()  # 去除多余的维度并保持连续性

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 如果在多 GPU 环境下，对 start_positions 和 end_positions 添加一维
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 将超出模型输入的起始/结束位置限制在有效范围内
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 定义交叉熵损失函数，忽略 ignored_index 处的预测
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        # 如果不需要返回字典形式的输出，按元组形式返回结果
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]  # 包括额外的 hidden_states 和 attentions
            return ((total_loss,) + output) if total_loss is not None else output

        # 返回 QuestionAnsweringModelOutput 类型的对象，包含损失、起始和结束 logits、隐藏状态和注意力权重
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 定义一个带有多选分类头的 MobileBert 模型，用于例如 RocStories/SWAG 任务。
# 这个类继承自 MobileBertPreTrainedModel。
class MobileBertForMultipleChoice(MobileBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化 MobileBert 模型
        self.mobilebert = MobileBertModel(config)
        
        # 确定分类器的 dropout 比率，如果未指定则使用隐藏层 dropout 比率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 定义一个 dropout 层，用于分类器
        self.dropout = nn.Dropout(classifier_dropout)
        
        # 定义一个线性层作为分类器，输入大小为隐藏层大小，输出大小为1
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并进行最终处理
        self.post_init()

    # 添加文档字符串到 forward 方法，描述输入参数
    @add_start_docstrings_to_model_forward(
        MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    # 添加代码示例文档字符串到 forward 方法，描述模型的输出类型和相关配置
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
    ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 初始化返回字典，如果未提供则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 计算选择项的数量，根据 input_ids 的第二维度确定
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 重新塑形输入张量，以便适应模型要求
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 将输入传递给 MobileBERT 模型进行处理
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

        # 应用 dropout 正则化
        pooled_output = self.dropout(pooled_output)
        # 将池化后的输出传递给分类器得到 logits
        logits = self.classifier(pooled_output)
        # 重新塑形 logits，以匹配选择项的形状
        reshaped_logits = logits.view(-1, num_choices)

        # 计算损失
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 根据 return_dict 决定输出结果的格式
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 构造 MultipleChoiceModelOutput 对象作为返回结果
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 添加文档字符串描述 MobileBert 模型与顶部的标记分类头部（线性层），例如用于命名实体识别（NER）任务
@add_start_docstrings(
    """
    MobileBert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    MOBILEBERT_START_DOCSTRING,
)
# 从 transformers.models.bert.modeling_bert.BertForTokenClassification 复制并修改为 MobileBert，保持所有大小写一致
class MobileBertForTokenClassification(MobileBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # 使用 MobileBertModel 初始化
        self.mobilebert = MobileBertModel(config, add_pooling_layer=False)
        # 如果配置中指定了 classifier_dropout，则使用该值；否则使用 hidden_dropout_prob
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 定义一个 Dropout 层，用于分类器
        self.dropout = nn.Dropout(classifier_dropout)
        # 定义一个线性层作为分类器，输入大小为 hidden_size，输出大小为 num_labels
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并进行最终处理
        self.post_init()

    # 添加文档字符串描述模型的 forward 方法
    @add_start_docstrings_to_model_forward(MOBILEBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 添加示例代码文档字符串，展示输入输出的示例
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
        # 输入参数说明：input_ids：输入的 token IDs；attention_mask：注意力掩码；token_type_ids：token 类型 IDs；...
    ):
        ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 根据输入的 return_dict 参数确定是否返回字典形式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 MobileBERT 模型进行前向传播
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

        # 获取模型输出的序列特征向量
        sequence_output = outputs[0]

        # 对序列特征向量应用 dropout 操作
        sequence_output = self.dropout(sequence_output)
        
        # 对 dropout 后的特征向量进行分类预测
        logits = self.classifier(sequence_output)

        # 初始化损失为 None
        loss = None
        # 如果提供了标签，则计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果 return_dict 为 False，则返回 logits 和额外的 hidden_states
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则返回 TokenClassifierOutput 对象
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```