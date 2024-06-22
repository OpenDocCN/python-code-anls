# `.\models\convbert\modeling_convbert.py`

```py
# 设定文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，授权信息详见链接：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，软件根据“适用情况”分发，没有明示或暗示的任何保证或条件
# 有关特定语言的更多信息，请查看许可证，以获取详细信息和限制
""" PyTorch ConvBERT model."""

# 导入需要的库
import math
import os
from operator import attrgetter
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN, get_activation
from ...modeling_outputs import (
    BaseModelOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel, SequenceSummary
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_convbert import ConvBertConfig

# 获取日志对象
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置
_CHECKPOINT_FOR_DOC = "YituTech/conv-bert-base"
_CONFIG_FOR_DOC = "ConvBertConfig"

# ConvBERT 预训练模型存档列表
CONVBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "YituTech/conv-bert-base",
    "YituTech/conv-bert-medium-small",
    "YituTech/conv-bert-small",
    # 更多 ConvBERT 模型请参考 https://huggingface.co/models?filter=convbert
]

# 从 TensorFlow 检查点加载权重到 PyTorch 模型
def load_tf_weights_in_convbert(model, config, tf_checkpoint_path):
    try:
        import tensorflow as tf
    except ImportError:
        # 如果导入 TensorFlow 出错，提示用户需安装 TensorFlow
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # 加载 TF 模型的权重
    init_vars = tf.train.list_variables(tf_path)
    tf_data = {}
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        tf_data[name] = array
    # 创建参数映射字典，将PyTorch模型参数名称映射到TensorFlow模型参数名称
    param_mapping = {
        "embeddings.word_embeddings.weight": "electra/embeddings/word_embeddings",
        "embeddings.position_embeddings.weight": "electra/embeddings/position_embeddings",
        "embeddings.token_type_embeddings.weight": "electra/embeddings/token_type_embeddings",
        "embeddings.LayerNorm.weight": "electra/embeddings/LayerNorm/gamma",
        "embeddings.LayerNorm.bias": "electra/embeddings/LayerNorm/beta",
        "embeddings_project.weight": "electra/embeddings_project/kernel",
        "embeddings_project.bias": "electra/embeddings_project/bias",
    }
    
    # 根据配置文件确定组数，选择不同的组名
    if config.num_groups > 1:
        group_dense_name = "g_dense"
    else:
        group_dense_name = "dense"
    
    # 遍历模型的所有命名参数
    for param in model.named_parameters():
        # 获取参数名称
        param_name = param[0]
        # 使用参数名称获取属性
        retriever = attrgetter(param_name)
        # 使用属性获取结果
        result = retriever(model)
        # 使用PyTorch的张量从NumPy数组创建参数值
        value = torch.from_numpy(tf_data[tf_name])
        # 打印日志信息，显示TensorFlow参数名称和PyTorch参数名称
        logger.info(f"TF: {tf_name}, PT: {param_name} ")
        # 如果TensorFlow参数名称以/kernel结尾
        if tf_name.endswith("/kernel"):
            # 如果TensorFlow参数名称不以/intermediate/g_dense/kernel结尾
            if not tf_name.endswith("/intermediate/g_dense/kernel"):
                # 如果TensorFlow参数名称不以/output/g_dense/kernel结尾
                # 则将参数值转置
                value = value.T
        # 如果TensorFlow参数名称以/depthwise_kernel结尾
        # 则对参数值进行维度变换
        if tf_name.endswith("/depthwise_kernel"):
            value = value.permute(1, 2, 0)  # 2, 0, 1
        # 如果TensorFlow参数名称以/pointwise_kernel结尾
        # 则对参数值进行维度变换
        if tf_name.endswith("/pointwise_kernel"):
            value = value.permute(2, 1, 0)  # 2, 1, 0
        # 如果TensorFlow参数名称以/conv_attn_key/bias结尾
        # 则在最后一个维度上添加一个维度
        if tf_name.endswith("/conv_attn_key/bias"):
            value = value.unsqueeze(-1)
        # 将模型参数值赋值给结果
        result.data = value
    
    # 返回更新后的模型
    return model
class ConvBertEmbeddings(nn.Module):
    """构造词、位置和标记类型嵌入的嵌入层。"""

    def __init__(self, config):
        super().__init__()
        # 词嵌入层，将词索引映射到嵌入向量
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        # 位置嵌入层，将位置索引映射到嵌入向量
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        # 标记类型嵌入层，将标记类型索引映射到嵌入向量
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)

        # self.LayerNorm的命名风格与TensorFlow模型变量名称相同，并且可以加载任何TensorFlow检查点文件
        # 归一化层，对嵌入向量进行归一化
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        # Dropout层，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 注册缓冲区，position_ids是固定的，用于序列化和反序列化
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 注册缓冲区，token_type_ids是固定的，用于序列化和反序列化
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.LongTensor:
        # 如果给定input_ids，则获取其维度
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            # 否则获取inputs_embeds的维度
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度
        seq_length = input_shape[1]

        # 如果未给定position_ids，则使用注册的缓冲区的position_ids，并截取到seq_length长度
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # 如果未给定token_type_ids
        if token_type_ids is None:
            # 如果token_type_ids被注册为缓冲区，则获取缓冲区中给定长度的token_type_ids，并在输入形状扩展为(input_shape[0], seq_length)
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            # 否则创建具有与输入形状相同的全0张量
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果未给定inputs_embeds，则使用word_embeddings根据input_ids获取嵌入向量
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # 获取位置嵌入向量
        position_embeddings = self.position_embeddings(position_ids)
        # 获取标记类型嵌入向量
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 计算总的嵌入向量
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        # 对嵌入向量进行归一化
        embeddings = self.LayerNorm(embeddings)
        # Dropout层，防止过拟合
        embeddings = self.dropout(embeddings)
        return embeddings


class ConvBertPreTrainedModel(PreTrainedModel):
    """ConvBert预训练模型基类。"""
    # 一个抽象类，用于处理权重初始化和一个简单的接口用于下载和加载预训练模型
    """

    # 配置类为ConvBertConfig
    config_class = ConvBertConfig
    # 加载TF权重
    load_tf_weights = load_tf_weights_in_convbert
    # 基础模型前缀为"convbert"
    base_model_prefix = "convbert"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    # 初始化权重的函数
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果模块是线性层
        if isinstance(module, nn.Linear):
            # 与TF版本稍有不同，使用正态分布对权重进行初始化
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置项，则将其初始化为0
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是嵌入层
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在填充索引，则将其对应的权重初始化为0
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果模块是LayerNorm层
        elif isinstance(module, nn.LayerNorm):
            # 将偏置项初始化为0
            module.bias.data.zero_()
            # 权重初始化为1
            module.weight.data.fill_(1.0)
class SeparableConv1D(nn.Module):
    """This class implements separable convolution, i.e. a depthwise and a pointwise layer"""

    def __init__(self, config, input_filters, output_filters, kernel_size, **kwargs):
        super().__init__()
        # Depthwise convolution layer
        self.depthwise = nn.Conv1d(
            input_filters,
            input_filters,
            kernel_size=kernel_size,
            groups=input_filters,
            padding=kernel_size // 2,
            bias=False,
        )
        # Pointwise convolution layer
        self.pointwise = nn.Conv1d(input_filters, output_filters, kernel_size=1, bias=False)
        # Bias parameter for the output
        self.bias = nn.Parameter(torch.zeros(output_filters, 1))

        # Initialize weights with normal distribution
        self.depthwise.weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.pointwise.weight.data.normal_(mean=0.0, std=config.initializer_range)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Depthwise convolution
        x = self.depthwise(hidden_states)
        # Pointwise convolution
        x = self.pointwise(x)
        # Add bias
        x += self.bias
        return x


class ConvBertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Check if hidden_size is divisible by num_attention_heads
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # Adjust number of attention heads if necessary
        new_num_attention_heads = config.num_attention_heads // config.head_ratio
        if new_num_attention_heads < 1:
            self.head_ratio = config.num_attention_heads
            self.num_attention_heads = 1
        else:
            self.num_attention_heads = new_num_attention_heads
            self.head_ratio = config.head_ratio

        # Set convolution kernel size
        self.conv_kernel_size = config.conv_kernel_size
        if config.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size should be divisible by num_attention_heads")

        # Calculate attention head size
        self.attention_head_size = (config.hidden_size // self.num_attention_heads) // 2
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Linear layers for query, key, and value
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # Separable convolution layer for key
        self.key_conv_attn_layer = SeparableConv1D(
            config, config.hidden_size, self.all_head_size, self.conv_kernel_size
        )
        # Linear layer for convolution kernel
        self.conv_kernel_layer = nn.Linear(self.all_head_size, self.num_attention_heads * self.conv_kernel_size)
        # Linear layer for convolution output
        self.conv_out_layer = nn.Linear(config.hidden_size, self.all_head_size)

        # Unfold operation for convolution
        self.unfold = nn.Unfold(
            kernel_size=[self.conv_kernel_size, 1], padding=[int((self.conv_kernel_size - 1) / 2), 0]
        )

        # Dropout layer
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    # 将输入张量进行形状转换，以便计算注意力分数
    def transpose_for_scores(self, x):
        # 计算新的张量形状，保持除最后一维外的所有维度不变，最后一维拆分为头数和头大小
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # 调整张量形状为新形状
        x = x.view(*new_x_shape)
        # 对张量进行维度置换，将头维度置换到第二维，以便进行注意力计算
        return x.permute(0, 2, 1, 3)

    # 前向传播函数，实现自注意力机制
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入张量，表示待处理的序列
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力遮罩，用于屏蔽无效位置
        head_mask: Optional[torch.FloatTensor] = None,  # 头部遮罩，用于屏蔽特定注意力头的计算
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 编码器隐藏状态，用于实现编码-解码注意力
        output_attentions: Optional[bool] = False,  # 是否输出注意力权重
# 定义一个处理 BERT 自注意力部分输出的模块
class ConvBertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义一个全连接层，输入和输出大小为配置中隐藏层的大小
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义 LayerNorm 层，输入大小为配置中隐藏层的大小，epsilon 为配置中的层归一化 epsilon
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 定义一个 dropout 层，概率为配置中的隐藏层 dropout 概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播，接受两个张量作为输入，返回一个张量作为输出
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 经过全连接层
        hidden_states = self.dense(hidden_states)
        # 经过 dropout 层
        hidden_states = self.dropout(hidden_states)
        # 经过 LayerNorm 层，并与输入张量相加
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# 定义一个处理 BERT 注意力部分的模块
class ConvBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义一个用于自注意力的模块
        self.self = ConvBertSelfAttention(config)
        # 定义一个用于输出的模块
        self.output = ConvBertSelfOutput(config)
        # 初始化一个空集合，用于存储被剪枝的头
        self.pruned_heads = set()

    # 对头进行剪枝操作
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 找到可以被剪枝的头和索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储被剪枝的头
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 前向传播，接受多个张量作为输入，返回一个元组，包含一个张量和一个可选的张量
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.FloatTensor]]:
        # 进行自注意力操作
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            output_attentions,
        )
        # 将注意力输出通过输出模块处理
        attention_output = self.output(self_outputs[0], hidden_states)
        # 汇总输出，如果需要输出注意力则添加进来
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


# 定义一个分组线性层
class GroupedLinearLayer(nn.Module):
    def __init__(self, input_size, output_size, num_groups):
        super().__init__()
        # 初始化分组线性层的参数
        self.input_size = input_size
        self.output_size = output_size
        self.num_groups = num_groups
        self.group_in_dim = self.input_size // self.num_groups
        self.group_out_dim = self.output_size // self.num_groups
        self.weight = nn.Parameter(torch.empty(self.num_groups, self.group_in_dim, self.group_out_dim))
        self.bias = nn.Parameter(torch.empty(output_size
    # 定义前向传播函数，接受隐藏状态张量作为输入，返回处理后的张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 获取隐藏状态张量的批量大小
        batch_size = list(hidden_states.size())[0]
        # 重塑隐藏状态张量的形状，以便后续处理
        x = torch.reshape(hidden_states, [-1, self.num_groups, self.group_in_dim])
        # 调换张量的维度顺序，便于矩阵相乘
        x = x.permute(1, 0, 2)
        # 执行矩阵乘法，将输入张量与权重矩阵相乘
        x = torch.matmul(x, self.weight)
        # 再次调换张量的维度顺序，恢复到原始形状
        x = x.permute(1, 0, 2)
        # 重塑张量的形状，以便加上偏置项
        x = torch.reshape(x, [batch_size, -1, self.output_size])
        # 加上偏置项
        x = x + self.bias
        # 返回处理后的张量
        return x
# ConvBertIntermediate类，继承自nn.Module，用于实现ConvBERT中间层
class ConvBertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 根据config的num_groups参数判断是否使用GroupedLinearLayer模块
        if config.num_groups == 1:
            self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        else:
            self.dense = GroupedLinearLayer(
                input_size=config.hidden_size, output_size=config.intermediate_size, num_groups=config.num_groups
            )
        # 判断config的hidden_act参数类型，根据对应字典ACT2FN选择激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播函数，输入hidden_states张量，返回处理后的张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 经过线性层变换
        hidden_states = self.dense(hidden_states)
        # 经过激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回处理后的张量
        return hidden_states

# ConvBertOutput类，继承自nn.Module，用于实现ConvBERT输出层
class ConvBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 根据config的num_groups参数判断是否使用GroupedLinearLayer模块
        if config.num_groups == 1:
            self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        else:
            self.dense = GroupedLinearLayer(
                input_size=config.intermediate_size, output_size=config.hidden_size, num_groups=config.num_groups
            )
        # LayerNorm层
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，输入hidden_states和input_tensor张量，返回处理后的张量
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 经过线性层变换
        hidden_states = self.dense(hidden_states)
        # 经过Dropout层
        hidden_states = self.dropout(hidden_states)
        # 加上输入张量input_tensor并经过LayerNorm层
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的张量
        return hidden_states

# ConvBertLayer类，继承自nn.Module，用于实现ConvBERT的一个层
class ConvBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 分块feed forward大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        # ConvBertAttention层
        self.attention = ConvBertAttention(config)
        # 是否是解码器
        self.is_decoder = config.is_decoder
        # 是否添加交叉注意力
        self.add_cross_attention = config.add_cross_attention
        # 如果添加交叉注意力，但不是解码器，抛出TypeError
        if self.add_cross_attention:
            if not self.is_decoder:
                raise TypeError(f"{self} should be used as a decoder model if cross attention is added")
            # ConvBertAttention层
            self.crossattention = ConvBertAttention(config)
        # ConvBertIntermediate层
        self.intermediate = ConvBertIntermediate(config)
        # ConvBertOutput层
        self.output = ConvBertOutput(config)

    # 前向传播函数，输入hidden_states和attention_mask等张量，返回处理后的张量
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    # 定义一个方法，接收隐藏状态、注意力掩码、头部掩码、输出注意力权重
    # 返回自注意力的输出结果和可能的注意力权重
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
    ):
        # 进行自注意力计算
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        # 获取自注意力计算的输出结果
        attention_output = self_attention_outputs[0]
        # 获取额外输出（注意力权重） 
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # 如果是解码器同时传入了编码器隐藏状态
        if self.is_decoder and encoder_hidden_states is not None:
            # 如果没有交叉注意力层，抛出错误
            if not hasattr(self, "crossattention"):
                raise AttributeError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )
            # 进行交叉注意力计算
            cross_attention_outputs = self.crossattention(
                attention_output,
                encoder_attention_mask,
                head_mask,
                encoder_hidden_states,
                output_attentions,
            )
            # 获取交叉注意力计算的输出结果
            attention_output = cross_attention_outputs[0]
            # 将交叉注意力的输出结果添加到额外输出中
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        # 对注意力输出进行分块处理
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        # 将分块处理后的输出结果添加到额外输出中
        outputs = (layer_output,) + outputs
        # 返回所有输出
        return outputs

    # 定义一个方法，接收注意力输出，执行前馈神经网络操作并返回结果
    def feed_forward_chunk(self, attention_output):
        # 中间层的输出结果
        intermediate_output = self.intermediate(attention_output)
        # 最终层的输出结果
        layer_output = self.output(intermediate_output, attention_output)
        # 返回最终输出结果
        return layer_output
# 定义一个名为ConvBertEncoder的类，继承自nn.Module
class ConvBertEncoder(nn.Module):
    # 初始化函数，接受config参数
    def __init__(self, config):
        super().__init__()
        # 将config赋值给self.config
        self.config = config
        # 创建一个nn.ModuleList，包含config.num_hidden_layers个ConvBertLayer对象
        self.layer = nn.ModuleList([ConvBertLayer(config) for _ in range(config.num_hidden_layers)])
        # 初始化梯度检查点为False
        self.gradient_checkpointing = False

    # 前向传播函数定义
    def forward(
        # 输入参数：hidden_states为输入的隐藏状态张量
        self,
        hidden_states: torch.Tensor,
        # attention_mask: 用于遮蔽无关信息的张量，可选
        attention_mask: Optional[torch.FloatTensor] = None,
        # head_mask: 头部遮罩，可选
        head_mask: Optional[torch.FloatTensor] = None,
        # encoder_hidden_states: 编码器的隐藏状态张量，可选
        encoder_hidden_states: Optional[torch.Tensor] = None,
        # encoder_attention_mask: 编码器的attention_mask，可选
        encoder_attention_mask: Optional[torch.Tensor] = None,
        # output_attentions: 是否输出注意力权重，可选，默认为False
        output_attentions: Optional[bool] = False,
        # output_hidden_states: 是否输出隐藏状态，可选，默认为False
        output_hidden_states: Optional[bool] = False,
        # return_dict: 是否返回字典结果，可选，默认为True
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutputWithCrossAttentions]:
        # 初始化空的隐藏状态元组，如果output_hidden_states为True
        all_hidden_states = () if output_hidden_states else None
        # 初始化空的自注意力元组，如果output_attentions为True
        all_self_attentions = () if output_attentions else None
        # 初始化空的交叉注意力元组，如果output_attentions为True且self.config.add_cross_attention为True
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        # 遍历self.layer中的每一层
        for i, layer_module in enumerate(self.layer):
            # 如果output_hidden_states为True，将当前的hidden_states添加到all_hidden_states中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 从head_mask中获取当前层的头部遮罩
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 如果使用梯度检查点并且处于训练状态
            if self.gradient_checkpointing and self.training:
                # 使用_gradient_checkpointing_func函数进行梯度检查点
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )
            else:
                # 否则，直接调用layer_module进行前向传播
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )
            # 更新hidden_states为layer_outputs的第一个元素
            hidden_states = layer_outputs[0]
            # 如果output_attentions为True，将当前层的自注意力加入all_self_attentions
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果self.config.add_cross_attention为True，则将当前层的交叉注意力加入all_cross_attentions
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果output_hidden_states为True，将当前的hidden_states添加到all_hidden_states中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果return_dict为False，则返回tuple
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )
        #
    # 初始化函数，接受配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个全连接层，输入和输出维度都是config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 如果config.hidden_act是字符串类型，则使用对应的激活函数，否则直接使用config.hidden_act指定的激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # 创建LayerNorm层，输入维度为config.hidden_size，eps为config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    # 前向传播函数，接受输入hidden_states，返回经过处理后的hidden_states
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 输入hidden_states经过全连接层处理
        hidden_states = self.dense(hidden_states)
        # 经过激活函数处理
        hidden_states = self.transform_act_fn(hidden_states)
        # 经过LayerNorm处理
        hidden_states = self.LayerNorm(hidden_states)
        # 返回处理后的hidden_states
        return hidden_states
# 定义CONVBERT_START_DOCSTRING字符串常量，用于描述ConvBert模型的参数和用法
CONVBERT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ConvBertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义CONVBERT_INPUTS_DOCSTRING字符串常量，用于描述ConvBert模型的输入参数
CONVBERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列标记在词汇表中的索引。
            # 可以使用 [`AutoTokenizer`] 获取索引。有关详情，请参阅 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`]。
            # [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 避免对填充标记索引执行注意力的掩码。
            # 掩码值在 `[0, 1]` 之间:
            # - 1 表示**未被掩盖**的标记，
            # - 0 表示**被掩盖**的标记。
            # [什么是注意力掩码？](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 指示输入的第一部分和第二部分的段标记索引。
            # 索引选择在 `[0, 1]` 之间:
            # - 0 对应于*句子 A* 标记，
            # - 1 对应于*句子 B* 标记。
            # [什么是标记类型 ID？](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 每个输入序列标记在位置嵌入中的位置索引。
            # 选择范围为 `[0, config.max_position_embeddings - 1]`。
            # [什么是位置 ID？](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于将自注意力模块的部分头部置零的掩码。
            # 掩码值在 `[0, 1]` 之间:
            # - 1 表示头部**未被掩盖**，
            # - 0 表示头部**被掩盖**。
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选地，可以直接传递嵌入表示，而不是传递 `input_ids`。
            # 如果希望对如何将 *input_ids* 索引转换为关联向量具有更多控制权，则这很有用，而不是使用模型的内部嵌入查找矩阵。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。
            # 有关详细信息，请参阅返回的张量下的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。
            # 有关详细信息，请参阅返回的张量下的 `hidden_states`。
        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通元组。
# 定义 ConvBERT 模型，输出原始隐藏状态而不带任何特定的顶部头
# 导入基类的文档字符串和特定配置的文档字符串
class ConvBertModel(ConvBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 定义模型的嵌入层
        self.embeddings = ConvBertEmbeddings(config)
        
        # 如果嵌入维度与隐藏状态维度不相等，则进行线性投影
        if config.embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(config.embedding_size, config.hidden_size)
            
        # 定义编码器层
        self.encoder = ConvBertEncoder(config)
        self.config = config
        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入嵌入
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

    # 在模型前向传播中添加文档字符串
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ):
        # 设置输出注意力权重，如果未指定，则使用配置中的默认设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置是否输出隐藏状态，如果未指定，则使用配置中的默认设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置返回值类型，如果未指定，则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果既指定了 input_ids 又指定了 inputs_embeds，则抛出数值错误
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        # 如果指定了 input_ids，发出警告并检查注意力掩码
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        # 如果指定了 inputs_embeds，获取其形状
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        # 如果两者都未指定，则抛出数值错误
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 获取批量大小和序列长度
        batch_size, seq_length = input_shape
        # 获取设备信息
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # 如果注意力掩码未指定，则使用完整的注意力掩码
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        # 如果令牌类型 ID 未指定
        if token_type_ids is None:
            # 如果嵌入层具有 token_type_ids，则获取并扩展其形状
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            # 否则，创建全零的令牌类型 ID
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # 获取扩展后的注意力掩码和头部遮罩
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # 获取嵌入层输出
        hidden_states = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        # 如果具有 "embeddings_project" 属性，则应用嵌入层的投影
        if hasattr(self, "embeddings_project"):
            hidden_states = self.embeddings_project(hidden_states)

        # 将隐状态输入到编码器中，获取结果隐状态
        hidden_states = self.encoder(
            hidden_states,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 返回编码器的输出
        return hidden_states
class ConvBertGeneratorPredictions(nn.Module):
    """Prediction module for the generator, made up of two dense layers."""
    # 初始化函数
    def __init__(self, config):
        super().__init__()
        # 获取激活函数 "gelu"
        self.activation = get_activation("gelu")
        # 使用配置中的参数初始化 LayerNorm
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        # 使用配置中的参数初始化全连接层
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)

    # 前向传播函数
    def forward(self, generator_hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        # 全连接层
        hidden_states = self.dense(generator_hidden_states)
        # 激活函数
        hidden_states = self.activation(hidden_states)
        # LayerNorm
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states


@add_start_docstrings("""ConvBERT Model with a `language modeling` head on top.""", CONVBERT_START_DOCSTRING)
class ConvBertForMaskedLM(ConvBertPreTrainedModel):
    _tied_weights_keys = ["generator.lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        # 创建 ConvBertModel 对象
        self.convbert = ConvBertModel(config)
        # 创建 ConvBertGeneratorPredictions 对象
        self.generator_predictions = ConvBertGeneratorPredictions(config)
        # 创建线性层对象，用于生成词汇表中的词语
        self.generator_lm_head = nn.Linear(config.embedding_size, config.vocab_size)
        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出词嵌入
    def get_output_embeddings(self):
        return self.generator_lm_head

    # 设置输出词嵌入
    def set_output_embeddings(self, word_embeddings):
        self.generator_lm_head = word_embeddings

    # 前向传播函数
    @add_start_docstrings_to_model_forward(CONVBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        # 设置返回字典的选项，如果未指定则使用模型配置中的设定
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 ConvBERT 模型进行前向传播，获取生成器的隐藏状态
        generator_hidden_states = self.convbert(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        # 获取生成器的序列输出
        generator_sequence_output = generator_hidden_states[0]

        # 生成器的预测分数，包括经过 LM 头后的输出
        prediction_scores = self.generator_predictions(generator_sequence_output)
        prediction_scores = self.generator_lm_head(prediction_scores)

        loss = None
        # 如果存在标签，计算 MLM（Masked Language Modeling）的损失
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果不要求返回字典，返回生成器的预测分数和隐藏状态
        if not return_dict:
            output = (prediction_scores,) + generator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果要求返回字典，返回 MaskedLMOutput 对象，包括损失、预测分数、隐藏状态和注意力权重
        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=generator_hidden_states.hidden_states,
            attentions=generator_hidden_states.attentions,
        )
```  
class ConvBertClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        # 创建一个线性层，输入维度为config.hidden_size，输出维度为config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 根据config中的配置选择分类器的dropout，如果没有配置，则使用隐藏层的dropout
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 对输入进行dropout
        self.dropout = nn.Dropout(classifier_dropout)
        # 创建一个线性层，输入维度为config.hidden_size，输出维度为config.num_labels
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        self.config = config

    # 定义模型的前向传播
    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        # 取出序列中第一个 token 的向量表示，相当于取出[CLS]的向量
        x = hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS])
        # 对向量进行dropout
        x = self.dropout(x)
        # 将向量传入线性层进行变换
        x = self.dense(x)
        # 将变换后的向量输入激活函数中
        x = ACT2FN[self.config.hidden_act](x)
        # 再次进行dropout
        x = self.dropout(x)
        # 将向量传入分类器线性层中
        x = self.out_proj(x)
        return x


@add_start_docstrings(
    """
    ConvBERT Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    CONVBERT_START_DOCSTRING,
)
class ConvBertForSequenceClassification(ConvBertPreTrainedModel):
    # 初始化函数
    def __init__(self, config):
        super().__init__(config)
        # 定义模型可分类的标签数量
        self.num_labels = config.num_labels
        # 将config保存在模型
        self.config = config
        # 创建ConvBERT模型
        self.convbert = ConvBertModel(config)
        # 创建分类器头部
        self.classifier = ConvBertClassificationHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 定义模型的前向传播
    @add_start_docstrings_to_model_forward(CONVBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 确定是否返回字典，默认为配置文件中的返回字典设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用ConvBert模型进行前向传播
        outputs = self.convbert(
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
        # 通过分类器获取logits
        logits = self.classifier(sequence_output)

        # 初始化损失为None
        loss = None
        # 如果有标签，则计算损失
        if labels is not None:
            # 判断问题类型，根据不同类型选取不同损失函数
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

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

        # 如果不返回字典，只返回logits和其他输出
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回SequenceClassifierOutput类实例
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 基于 ConvBERT 模型的多选题分类模型，包含一个线性层用于分类，例如 RocStories/SWAG 任务
@add_start_docstrings(
    """
    ConvBERT Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    CONVBERT_START_DOCSTRING,
)
# 定义 ConvBertForMultipleChoice 类，继承自 ConvBertPreTrainedModel
class ConvBertForMultipleChoice(ConvBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化 ConvBERT 模型
        self.convbert = ConvBertModel(config)
        # 初始化序列摘要
        self.sequence_summary = SequenceSummary(config)
        # 初始化分类器
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(
        CONVBERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义模型的前向传播方法
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
        ) -> Union[Tuple, MultipleChoiceModelOutput]:
        r"""
        labels ('torch.LongTensor' of shape '(batch_size,)', *optional*):
            用于计算多项选择分类损失的标签。索引应在 `[0, ..., num_choices-1]` 范围内，其中 `num_choices` 是输入张量的第二个维度的大小。 (参见上面的 `input_ids`)
        """
        # 确保返回字典不为空
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果输入张量不为空，则确定选择数量
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 如果输入张量不为空，则调整输入张量的形状
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 使用ConvBERT模型对输入进行处理
        outputs = self.convbert(
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

        # 获取序列输出和池化输出
        sequence_output = outputs[0]
        pooled_output = self.sequence_summary(sequence_output)

        # 经过分类器得到logits
        logits = self.classifier(pooled_output)
        # 重新调整logits的形状
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        # 如果存在标签，则计算损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 如果return_dict为False，则返回输出元组
        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果return_dict为True，则返回MultipleChoiceModelOutput对象
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用 ConvBERT 模型进行标记分类的模型，例如用于实体命名识别（NER）任务
@add_start_docstrings(
    """
    ConvBERT Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    CONVBERT_START_DOCSTRING,
)
class ConvBertForTokenClassification(ConvBertPreTrainedModel):
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 获取标签类别数量
        self.num_labels = config.num_labels

        # 创建 ConvBERT 模型
        self.convbert = ConvBertModel(config)
        # 设置分类器的 dropout 概率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        # 创建线性分类器
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(CONVBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
    def forward(
        self, input_ids: torch.LongTensor, 
        attention_mask: Optional[torch.Tensor] = None, 
        token_type_ids: Optional[torch.LongTensor] = None, 
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None, 
        labels: Optional[torch.LongTensor] = None, 
        output_attentions: Optional[bool] = None, 
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.convbert(
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
    
        sequence_output = self.dropout(sequence_output)  # 应用 dropout 对输出进行正则化
        logits = self.classifier(sequence_output)  # 使用分类器对序列输出进行分类
    
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # 交叉熵损失函数，用于计算分类损失
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))  # 计算损失
    
        if not return_dict:
            output = (logits,) + outputs[1:]  # 如果不需要返回字典，则返回 logits 和其他输出
            return ((loss,) + output) if loss is not None else output  # 如果有损失则返回损失和其他输出，否则只返回其他输出
    
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,  # 返回隐藏状态信息
            attentions=outputs.attentions,  # 返回注意力信息
        )
# 使用 add_start_docstrings 装饰器为模型添加文档字符串，描述了在 SQuAD 等问题-回答任务中使用的 ConvBERT 模型
class ConvBertForQuestionAnswering(ConvBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 保存标签数量和 ConvBERT 模型
        self.num_labels = config.num_labels
        self.convbert = ConvBertModel(config)
        # 创建用于答案开始和结束的线性层
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 使用 add_start_docstrings_to_model_forward 装饰器添加模型前向传播的文档字符串，描述输入参数和输出类型
    # 使用 add_code_sample_docstrings 装饰器添加模型前向传播的代码示例和检查点等相关信息
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的token ID
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力遮罩
        token_type_ids: Optional[torch.LongTensor] = None,  # token类型的ID
        position_ids: Optional[torch.LongTensor] = None,  # 位置ID
        head_mask: Optional[torch.FloatTensor] = None,  # 头部遮罩
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 嵌入的输入
        start_positions: Optional[torch.LongTensor] = None,  # 答案开始位置
        end_positions: Optional[torch.LongTensor] = None,  # 答案结束位置
        output_attentions: Optional[bool] = None,  # 是否输出注意力
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.convbert(
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

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 如果GPU数量>1，需要添加一个维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 有时起始/结束位置在模型输入之外，我们忽略这些项
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            如果total_loss不为空，则返回(total_loss, output)，否则返回output
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```