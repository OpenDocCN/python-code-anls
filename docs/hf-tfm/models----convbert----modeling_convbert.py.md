# `.\models\convbert\modeling_convbert.py`

```py
# 设定文件编码为UTF-8
# 版权声明和许可信息
# 版权所有 2021 年 HuggingFace Inc. 团队保留所有权利。
#
# 根据 Apache 许可 2.0 版本（"许可证"）授权；
# 除非符合许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件
# 按"原样"提供，不提供任何明示或暗示的担保
# 或条件。详见许可证。
""" PyTorch ConvBERT 模型。"""


import math
import os
from operator import attrgetter
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入激活函数相关内容
from ...activations import ACT2FN, get_activation
# 导入模型输出相关内容
from ...modeling_outputs import (
    BaseModelOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
# 导入模型工具函数
from ...modeling_utils import PreTrainedModel, SequenceSummary
# 导入PyTorch相关工具函数和模型方法
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
# 导入工具函数中的日志记录功能
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
# 导入ConvBERT配置
from .configuration_convbert import ConvBertConfig

# 获取logger对象
logger = logging.get_logger(__name__)

# 文档中使用的检查点和配置
_CHECKPOINT_FOR_DOC = "YituTech/conv-bert-base"
_CONFIG_FOR_DOC = "ConvBertConfig"

# ConvBERT预训练模型的存档列表
CONVBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "YituTech/conv-bert-base",
    "YituTech/conv-bert-medium-small",
    "YituTech/conv-bert-small",
    # 查看所有ConvBERT模型，请访问 https://huggingface.co/models?filter=convbert
]


def load_tf_weights_in_convbert(model, config, tf_checkpoint_path):
    """从TensorFlow检查点加载权重到PyTorch模型中。"""
    try:
        import tensorflow as tf
    except ImportError:
        # 如果导入失败，记录错误信息并抛出异常
        logger.error(
            "在PyTorch中加载TensorFlow模型需要安装TensorFlow。请参阅 "
            "https://www.tensorflow.org/install/ 获取安装说明。"
        )
        raise
    # 获取TensorFlow检查点文件的绝对路径
    tf_path = os.path.abspath(tf_checkpoint_path)
    # 记录日志：正在从TensorFlow模型转换检查点
    logger.info(f"从 {tf_path} 转换TensorFlow检查点")
    # 从TF模型加载权重
    init_vars = tf.train.list_variables(tf_path)
    tf_data = {}
    # 遍历初始化变量列表，加载每个变量的权重数据到字典中
    for name, shape in init_vars:
        logger.info(f"加载TF权重 {name}，形状为 {shape}")
        array = tf.train.load_variable(tf_path, name)
        tf_data[name] = array
    # 定义参数映射字典，将模型参数名映射到TensorFlow模型中对应的权重名
    param_mapping = {
        "embeddings.word_embeddings.weight": "electra/embeddings/word_embeddings",
        "embeddings.position_embeddings.weight": "electra/embeddings/position_embeddings",
        "embeddings.token_type_embeddings.weight": "electra/embeddings/token_type_embeddings",
        "embeddings.LayerNorm.weight": "electra/embeddings/LayerNorm/gamma",
        "embeddings.LayerNorm.bias": "electra/embeddings/LayerNorm/beta",
        "embeddings_project.weight": "electra/embeddings_project/kernel",
        "embeddings_project.bias": "electra/embeddings_project/bias",
    }
    
    # 根据配置条件设置group_dense_name变量的值
    if config.num_groups > 1:
        group_dense_name = "g_dense"
    else:
        group_dense_name = "dense"

    # 遍历模型的所有命名参数
    for param in model.named_parameters():
        # 获取参数名
        param_name = param[0]
        # 使用attrgetter获取模型中参数名对应的属性
        retriever = attrgetter(param_name)
        result = retriever(model)
        # 根据param_mapping将参数名映射到对应的TensorFlow权重名
        tf_name = param_mapping[param_name]
        # 从tf_data中读取TensorFlow权重值，并转换为PyTorch Tensor
        value = torch.from_numpy(tf_data[tf_name])
        
        # 打印日志，显示转换信息
        logger.info(f"TF: {tf_name}, PT: {param_name} ")
        
        # 根据TensorFlow权重名后缀进行不同的处理
        if tf_name.endswith("/kernel"):
            # 如果不是特定的g_dense相关的kernel，需要对value进行转置操作
            if not tf_name.endswith("/intermediate/g_dense/kernel"):
                if not tf_name.endswith("/output/g_dense/kernel"):
                    value = value.T
        elif tf_name.endswith("/depthwise_kernel"):
            # 如果是深度可分离卷积的kernel，需要对value进行维度置换操作
            value = value.permute(1, 2, 0)  # 2, 0, 1
        elif tf_name.endswith("/pointwise_kernel"):
            # 如果是点卷积的kernel，同样需要对value进行维度置换操作
            value = value.permute(2, 1, 0)  # 2, 1, 0
        elif tf_name.endswith("/conv_attn_key/bias"):
            # 如果是注意力机制中的bias，需要在最后一维添加一个维度
            value = value.unsqueeze(-1)
        
        # 将处理后的value赋值给模型参数的data属性
        result.data = value
    
    # 返回处理后的模型
    return model
class ConvBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        # 创建词嵌入层，根据词汇大小、嵌入大小和填充标识初始化
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        # 创建位置嵌入层，根据最大位置编码和嵌入大小初始化
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        # 创建类型嵌入层，根据类型词汇大小和嵌入大小初始化
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)

        # self.LayerNorm 未采用蛇形命名法，以保持与 TensorFlow 模型变量名一致，以便能够加载任何 TensorFlow 检查点文件
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        # 创建丢弃层，根据隐藏层丢弃概率初始化
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 创建位置编码张量，保持内存连续性并在序列化时导出
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 创建类型编码张量，使用全零初始化，与位置编码张量大小相同
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
        # 如果提供了 input_ids，则获取其形状；否则，根据 inputs_embeds 推断形状
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # 如果未提供位置编码，则使用预先注册的 position_ids，截取适当长度
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # 如果未提供类型编码，检查是否已有注册的 token_type_ids，并扩展以匹配 input_shape；否则初始化全零的类型编码张量
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果未提供输入的嵌入向量，根据 input_ids 获取词嵌入
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        
        # 根据位置编码获取位置嵌入
        position_embeddings = self.position_embeddings(position_ids)
        # 根据类型编码获取类型嵌入
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 合并词嵌入、位置嵌入和类型嵌入
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        # 应用 LayerNorm
        embeddings = self.LayerNorm(embeddings)
        # 应用 Dropout
        embeddings = self.dropout(embeddings)
        return embeddings


class ConvBertPreTrainedModel(PreTrainedModel):
    """
    # 用于处理权重初始化和简单接口以下载和加载预训练模型的抽象类。

    # 指定配置类为ConvBertConfig
    config_class = ConvBertConfig
    # 加载 TensorFlow 权重的函数为load_tf_weights_in_convbert
    load_tf_weights = load_tf_weights_in_convbert
    # 基础模型前缀为"convbert"
    base_model_prefix = "convbert"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """初始化模型的权重"""
        # 如果是线性层(nn.Linear)
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重，均值为0，标准差为配置文件中的initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置，则将偏置初始化为0
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是嵌入层(nn.Embedding)
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重，均值为0，标准差为配置文件中的initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在填充索引(padding_idx)，则将对应位置的权重初始化为0
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果是层归一化层(nn.LayerNorm)
        elif isinstance(module, nn.LayerNorm):
            # 将偏置初始化为0
            module.bias.data.zero_()
            # 将权重初始化为1
            module.weight.data.fill_(1.0)
class SeparableConv1D(nn.Module):
    """This class implements separable convolution, i.e. a depthwise and a pointwise layer"""

    def __init__(self, config, input_filters, output_filters, kernel_size, **kwargs):
        super().__init__()
        # 定义深度卷积层，使用深度卷积（depthwise convolution）方式，groups=input_filters 表示每个输入通道单独卷积
        self.depthwise = nn.Conv1d(
            input_filters,
            input_filters,
            kernel_size=kernel_size,
            groups=input_filters,
            padding=kernel_size // 2,
            bias=False,
        )
        # 定义点卷积层，用于将深度卷积的结果进行升维到输出通道数
        self.pointwise = nn.Conv1d(input_filters, output_filters, kernel_size=1, bias=False)
        # 定义偏置项参数
        self.bias = nn.Parameter(torch.zeros(output_filters, 1))

        # 初始化深度卷积层和点卷积层的权重
        self.depthwise.weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.pointwise.weight.data.normal_(mean=0.0, std=config.initializer_range)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 执行深度卷积操作
        x = self.depthwise(hidden_states)
        # 执行点卷积操作
        x = self.pointwise(x)
        # 添加偏置项
        x += self.bias
        return x


class ConvBertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 检查隐藏层大小是否能被注意力头数整除
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 计算新的注意力头数
        new_num_attention_heads = config.num_attention_heads // config.head_ratio
        if new_num_attention_heads < 1:
            self.head_ratio = config.num_attention_heads
            self.num_attention_heads = 1
        else:
            self.num_attention_heads = new_num_attention_heads
            self.head_ratio = config.head_ratio

        # 设置卷积核大小
        self.conv_kernel_size = config.conv_kernel_size
        # 检查隐藏层大小是否能被注意力头数整除
        if config.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size should be divisible by num_attention_heads")

        # 计算每个注意力头的大小
        self.attention_head_size = (config.hidden_size // self.num_attention_heads) // 2
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 定义查询、键、值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 设置键卷积注意力层
        self.key_conv_attn_layer = SeparableConv1D(
            config, config.hidden_size, self.all_head_size, self.conv_kernel_size
        )
        # 设置卷积核层
        self.conv_kernel_layer = nn.Linear(self.all_head_size, self.num_attention_heads * self.conv_kernel_size)
        # 设置卷积输出层
        self.conv_out_layer = nn.Linear(config.hidden_size, self.all_head_size)

        # 定义卷积展开层
        self.unfold = nn.Unfold(
            kernel_size=[self.conv_kernel_size, 1], padding=[int((self.conv_kernel_size - 1) / 2), 0]
        )

        # 定义 dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    # 将输入张量 x 进行形状变换，用于注意力分数计算
    def transpose_for_scores(self, x):
        # 计算新的形状，保留除了最后一维外的所有维度，增加注意力头数和每个注意力头的大小
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # 对输入张量 x 进行形状重塑，使其符合注意力计算的需要
        x = x.view(*new_x_shape)
        # 对张量进行维度置换，以便进行注意力计算，顺序为 batch, head, seq_length, head_size
        return x.permute(0, 2, 1, 3)

    # 模型的前向传播函数，接收隐藏状态、注意力掩码、头掩码、编码器隐藏状态等作为输入
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
# 定义一个名为 ConvBertSelfOutput 的神经网络模块类
class ConvBertSelfOutput(nn.Module):
    # 初始化函数，接收一个配置参数对象 config
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个线性层，输入和输出大小均为配置参数中的隐藏大小
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个 LayerNorm 层，输入大小为隐藏大小，epsilon 为配置参数中的层归一化 epsilon
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个 Dropout 层，使用配置参数中的隐藏 dropout 概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，接收两个张量参数并返回一个张量
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用线性层处理隐藏状态张量
        hidden_states = self.dense(hidden_states)
        # 使用 Dropout 处理后的隐藏状态张量
        hidden_states = self.dropout(hidden_states)
        # 对加和后的归一化处理隐藏状态张量
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的隐藏状态张量
        return hidden_states


# 定义一个名为 ConvBertAttention 的神经网络模块类
class ConvBertAttention(nn.Module):
    # 初始化函数，接收一个配置参数对象 config
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个 ConvBertSelfAttention 对象，使用给定的配置参数
        self.self = ConvBertSelfAttention(config)
        # 创建一个 ConvBertSelfOutput 对象，使用给定的配置参数
        self.output = ConvBertSelfOutput(config)
        # 初始化一个空集合，用于存储被修剪的注意力头
        self.pruned_heads = set()

    # 修剪注意力头的方法，接收一个头的列表
    def prune_heads(self, heads):
        # 如果头列表为空，直接返回
        if len(heads) == 0:
            return
        # 调用辅助函数查找可修剪的头并返回索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 修剪自注意力层的线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储修剪过的头
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 前向传播函数，接收多个参数并返回一个元组
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.FloatTensor]]:
        # 调用自注意力层的前向传播方法，处理隐藏状态等输入参数
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            output_attentions,
        )
        # 使用输出层处理自注意力层的输出和输入的隐藏状态张量
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出注意力权重，则添加到输出元组中
        outputs = (attention_output,) + self_outputs[1:]  # 如果需要输出注意力权重，则添加到输出元组中
        # 返回处理后的输出元组
        return outputs


# 定义一个名为 GroupedLinearLayer 的神经网络模块类
class GroupedLinearLayer(nn.Module):
    # 初始化函数，接收输入大小、输出大小和分组数量作为参数
    def __init__(self, input_size, output_size, num_groups):
        # 调用父类的初始化函数
        super().__init__()
        # 设置输入大小、输出大小和分组数量的属性
        self.input_size = input_size
        self.output_size = output_size
        self.num_groups = num_groups
        # 计算每个分组的输入维度和输出维度
        self.group_in_dim = self.input_size // self.num_groups
        self.group_out_dim = self.output_size // self.num_groups
        # 创建权重参数张量，形状为 (num_groups, group_in_dim, group_out_dim)
        self.weight = nn.Parameter(torch.empty(self.num_groups, self.group_in_dim, self.group_out_dim))
        # 创建偏置参数张量，形状为 (output_size,)
        self.bias = nn.Parameter(torch.empty(output_size))
    # 定义一个前向传播方法，接收隐藏状态作为输入张量，返回处理后的张量作为输出
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 获取隐藏状态张量的批量大小
        batch_size = list(hidden_states.size())[0]
        # 将隐藏状态张量重塑为 [batch_size, self.num_groups, self.group_in_dim] 的形状
        x = torch.reshape(hidden_states, [-1, self.num_groups, self.group_in_dim])
        # 将张量 x 的维度重新排列为 [self.num_groups, batch_size, self.group_in_dim]
        x = x.permute(1, 0, 2)
        # 使用 self.weight 对 x 进行矩阵乘法运算
        x = torch.matmul(x, self.weight)
        # 再次将张量 x 的维度重新排列为 [batch_size, self.num_groups, self.output_size]
        x = x.permute(1, 0, 2)
        # 将张量 x 重塑为 [batch_size, -1, self.output_size] 的形状
        x = torch.reshape(x, [batch_size, -1, self.output_size])
        # 将张量 x 加上偏置 self.bias
        x = x + self.bias
        # 返回处理后的张量 x 作为输出
        return x
class ConvBertIntermediate(nn.Module):
    # ConvBertIntermediate 类定义
    def __init__(self, config):
        super().__init__()
        # 如果分组数为1，使用普通的线性层
        if config.num_groups == 1:
            self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        else:
            # 否则使用分组线性层
            self.dense = GroupedLinearLayer(
                input_size=config.hidden_size, output_size=config.intermediate_size, num_groups=config.num_groups
            )
        # 根据配置选择激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播方法
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用线性层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 使用中间激活函数处理线性层输出
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class ConvBertOutput(nn.Module):
    # ConvBertOutput 类定义
    def __init__(self, config):
        super().__init__()
        # 如果分组数为1，使用普通的线性层
        if config.num_groups == 1:
            self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        else:
            # 否则使用分组线性层
            self.dense = GroupedLinearLayer(
                input_size=config.intermediate_size, output_size=config.hidden_size, num_groups=config.num_groups
            )
        # LayerNorm 层，用于归一化隐藏状态
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout 层，用于随机丢弃部分隐藏状态
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播方法
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用线性层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 使用 Dropout 随机丢弃部分隐藏状态
        hidden_states = self.dropout(hidden_states)
        # 使用 LayerNorm 层对加和后的隐藏状态进行归一化处理
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class ConvBertLayer(nn.Module):
    # ConvBertLayer 类定义
    def __init__(self, config):
        super().__init__()
        # 用于分块的前馈传播的大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度维度
        self.seq_len_dim = 1
        # 自注意力层
        self.attention = ConvBertAttention(config)
        # 是否为解码器
        self.is_decoder = config.is_decoder
        # 是否添加交叉注意力
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            # 如果添加交叉注意力且不是解码器模型，则抛出类型错误
            if not self.is_decoder:
                raise TypeError(f"{self} should be used as a decoder model if cross attention is added")
            # 否则添加交叉注意力自注意力层
            self.crossattention = ConvBertAttention(config)
        # ConvBertIntermediate 中间层
        self.intermediate = ConvBertIntermediate(config)
        # ConvBertOutput 输出层
        self.output = ConvBertOutput(config)

    # 前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        # 省略号表示可能的其他参数
    # 定义方法，接受隐藏状态，注意力掩码，头部掩码，是否输出注意力权重，返回自注意力模型输出和可能的注意力权重
    ) -> Tuple[torch.Tensor, Optional[torch.FloatTensor]]:
        # 使用自注意力模型计算注意力输出
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        # 取出自注意力模型的输出
        attention_output = self_attention_outputs[0]
        # 如果输出注意力权重，将它们添加到输出中
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # 如果当前模型是解码器且有编码器隐藏状态输入
        if self.is_decoder and encoder_hidden_states is not None:
            # 如果当前模型没有跨注意力层，抛出属性错误
            if not hasattr(self, "crossattention"):
                raise AttributeError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )
            # 使用跨注意力模型计算跨注意力输出
            cross_attention_outputs = self.crossattention(
                attention_output,
                encoder_attention_mask,
                head_mask,
                encoder_hidden_states,
                output_attentions,
            )
            # 取出跨注意力模型的输出
            attention_output = cross_attention_outputs[0]
            # 将跨注意力权重添加到输出中
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        # 对注意力输出应用分块处理
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        # 将处理后的层输出添加到总体输出中
        outputs = (layer_output,) + outputs
        # 返回所有输出
        return outputs

    # 定义方法，接受注意力输出并进行前馈处理
    def feed_forward_chunk(self, attention_output):
        # 将注意力输出送入中间层
        intermediate_output = self.intermediate(attention_output)
        # 将中间层输出和注意力输出送入输出层
        layer_output = self.output(intermediate_output, attention_output)
        # 返回层输出
        return layer_output
# 定义一个名为 ConvBertEncoder 的神经网络模型类，继承自 nn.Module
class ConvBertEncoder(nn.Module):
    # 初始化函数，接受一个配置参数 config
    def __init__(self, config):
        super().__init__()
        # 将传入的配置参数保存到当前对象的 config 属性中
        self.config = config
        # 使用列表推导式创建一个包含多个 ConvBertLayer 实例的 ModuleList，数量为 config.num_hidden_layers
        self.layer = nn.ModuleList([ConvBertLayer(config) for _ in range(config.num_hidden_layers)])
        # 初始化梯度检查点标志为 False
        self.gradient_checkpointing = False

    # 前向传播函数，接受多个输入参数，并返回一个包含多个输出的对象
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量
        attention_mask: Optional[torch.FloatTensor] = None,  # 可选的注意力掩码张量
        head_mask: Optional[torch.FloatTensor] = None,  # 可选的头部掩码张量
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 可选的编码器隐藏状态张量
        encoder_attention_mask: Optional[torch.Tensor] = None,  # 可选的编码器注意力掩码张量
        output_attentions: Optional[bool] = False,  # 是否输出注意力权重，默认为 False
        output_hidden_states: Optional[bool] = False,  # 是否输出所有隐藏状态，默认为 False
        return_dict: Optional[bool] = True,  # 是否返回字典格式的输出，默认为 True
    ) -> Union[Tuple, BaseModelOutputWithCrossAttentions]:  # 返回值可以是元组或 BaseModelOutputWithCrossAttentions 类型
        # 如果需要输出隐藏状态，则初始化空的元组 all_hidden_states
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力权重，则初始化空的元组 all_self_attentions
        all_self_attentions = () if output_attentions else None
        # 如果需要输出交叉注意力权重且配置允许，则初始化空的元组 all_cross_attentions
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        # 遍历每个 ConvBertLayer 实例
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            # 获取当前层的头部掩码，如果头部掩码不为 None，则使用对应的头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None
            
            # 如果开启了梯度检查点且当前处于训练状态
            if self.gradient_checkpointing and self.training:
                # 使用梯度检查点函数来调用当前层的 __call__ 方法
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
                # 否则直接调用当前层的 __call__ 方法
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )
            # 更新隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]
            # 如果需要输出注意力权重
            if output_attentions:
                # 将当前层的注意力权重添加到 all_self_attentions 中
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果配置允许，将当前层的交叉注意力权重添加到 all_cross_attentions 中
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)
        
        # 如果需要输出隐藏状态，则将最终的隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # 如果不需要返回字典格式的输出，则返回包含多个非 None 元素的元组
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )
        # 否则返回一个 BaseModelOutputWithCrossAttentions 类型的对象，包含指定的输出
        return BaseModelOutputWithCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
    # 初始化函数，用于创建一个新的神经网络层对象
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个线性层，输入和输出维度都是config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 根据config中的配置选择激活函数，存储在self.transform_act_fn中
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # 创建一个LayerNorm层，对输入进行归一化处理
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # 前向传播函数，接受一个张量hidden_states作为输入，返回一个张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 输入张量经过线性层dense，输出经过变换后的hidden_states
        hidden_states = self.dense(hidden_states)
        # 经过激活函数变换
        hidden_states = self.transform_act_fn(hidden_states)
        # 经过LayerNorm层进行归一化处理
        hidden_states = self.LayerNorm(hidden_states)
        # 返回处理后的张量作为输出
        return hidden_states
# CONVBERT_INPUTS_DOCSTRING 用于定义 ConvBERT 模型的输入文档字符串，通常用于解释模型的输入参数和格式。
CONVBERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列标记在词汇表中的索引。

            # 可以使用 `AutoTokenizer` 获得这些索引。详见 `PreTrainedTokenizer.encode` 和 `PreTrainedTokenizer.__call__`。

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 遮盖机制，用于避免在填充的标记索引上进行注意力计算。遮盖值在 `[0, 1]` 范围内：

            # - 1 表示 **不被遮盖** 的标记，
            # - 0 表示 **被遮盖** 的标记。

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 段落标记索引，指示输入的第一和第二部分。索引选择在 `[0, 1]` 范围内：

            # - 0 对应于 *句子 A* 的标记，
            # - 1 对应于 *句子 B* 的标记。

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 每个输入序列标记在位置嵌入中的位置索引。选取范围为 `[0, config.max_position_embeddings - 1]`。

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 自注意力模块中选择性屏蔽的头部。遮盖值在 `[0, 1]` 范围内：

            # - 1 表示 **未被遮盖** 的头部，
            # - 0 表示 **被遮盖** 的头部。

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选项，可以直接传递嵌入表示，而不是传递 `input_ids`。这对于想要更好地控制如何将 *input_ids* 索引转换为相关向量比模型内部的嵌入查找矩阵更有用。

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。详见返回张量中的 `attentions` 以获取更多细节。

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。详见返回张量中的 `hidden_states` 以获取更多细节。

        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""
定义一个 ConvBERT 模型类，继承自 ConvBertPreTrainedModel，用于生成原始隐藏状态而不添加特定的输出头部。

@add_start_docstrings(
    "The bare ConvBERT Model transformer outputting raw hidden-states without any specific head on top.",
    CONVBERT_START_DOCSTRING,
)
class ConvBertModel(ConvBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 初始化嵌入层
        self.embeddings = ConvBertEmbeddings(config)

        # 如果嵌入大小不等于隐藏层大小，则添加线性映射层
        if config.embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(config.embedding_size, config.hidden_size)

        # 初始化编码器层
        self.encoder = ConvBertEncoder(config)
        self.config = config
        # 初始化权重并进行最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 获取输入的词嵌入
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        # 设置输入的词嵌入
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        对模型的注意力头进行修剪。heads_to_prune: dict，格式为 {层号: 要修剪的头列表} 参见基类 PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            # 对每个层的指定头进行修剪
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(CONVBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithCrossAttentions,
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
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithCrossAttentions]:
        # 如果未指定 output_attentions 参数，则使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定 output_hidden_states 参数，则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定 return_dict 参数，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果同时指定了 input_ids 和 inputs_embeds，抛出 ValueError 异常
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            # 如果指定了 input_ids，则检查 padding 的情况并提醒
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            # 获取 input_ids 的形状
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            # 如果指定了 inputs_embeds，则获取其形状，去掉最后一维
            input_shape = inputs_embeds.size()[:-1]
        else:
            # 如果既未指定 input_ids 也未指定 inputs_embeds，则抛出 ValueError 异常
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # 获取 batch_size 和 seq_length
        batch_size, seq_length = input_shape
        # 获取 input_ids 或 inputs_embeds 的设备信息
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # 如果未提供 attention_mask，则创建一个全为1的 mask 张量，形状与 input_shape 一致
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        # 如果未提供 token_type_ids
        if token_type_ids is None:
            # 如果 embeddings 拥有 token_type_ids 属性，则使用其提供的 token_type_ids
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                # 否则创建一个全为0的 token_type_ids 张量，dtype 为 long
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # 获取扩展后的 attention_mask
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)
        # 获取 head_mask
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # 使用 embeddings 函数获取 hidden_states
        hidden_states = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        # 如果模型具有 embeddings_project 属性，则对 hidden_states 进行投影处理
        if hasattr(self, "embeddings_project"):
            hidden_states = self.embeddings_project(hidden_states)

        # 使用 encoder 处理 hidden_states
        hidden_states = self.encoder(
            hidden_states,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 返回处理后的 hidden_states
        return hidden_states
# 定义 ConvBERT 模型的预测模块，由两个全连接层组成
class ConvBertGeneratorPredictions(nn.Module):
    """Prediction module for the generator, made up of two dense layers."""

    def __init__(self, config):
        super().__init__()

        # 使用 GELU 激活函数作为激活函数
        self.activation = get_activation("gelu")
        # LayerNorm 层，对隐藏状态进行标准化处理
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        # 全连接层，将隐藏状态映射到指定维度的特征空间
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)

    def forward(self, generator_hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        # 将生成器的隐藏状态传入全连接层
        hidden_states = self.dense(generator_hidden_states)
        # 使用 GELU 激活函数处理全连接层的输出
        hidden_states = self.activation(hidden_states)
        # 对处理后的隐藏状态进行 LayerNorm
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states


# ConvBERT 模型，具有在顶部进行语言建模的头部
@add_start_docstrings("""ConvBERT Model with a `language modeling` head on top.""", CONVBERT_START_DOCSTRING)
class ConvBertForMaskedLM(ConvBertPreTrainedModel):
    _tied_weights_keys = ["generator.lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)

        # 初始化 ConvBERT 模型
        self.convbert = ConvBertModel(config)
        # 初始化生成器预测模块
        self.generator_predictions = ConvBertGeneratorPredictions(config)

        # 生成器的语言建模头部，将隐藏状态映射到词汇表大小的空间
        self.generator_lm_head = nn.Linear(config.embedding_size, config.vocab_size)
        # 初始化权重并应用最终处理
        self.post_init()

    def get_output_embeddings(self):
        return self.generator_lm_head

    def set_output_embeddings(self, word_embeddings):
        self.generator_lm_head = word_embeddings

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
        # 根据 return_dict 是否为 None，确定是否使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 通过 ConvBERT 模型进行前向传播，生成隐状态
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
        # 从生成的隐藏状态中获取序列输出
        generator_sequence_output = generator_hidden_states[0]

        # 使用生成的序列输出进行预测得分计算
        prediction_scores = self.generator_predictions(generator_sequence_output)
        # 将预测得分再经过语言模型头部计算，得到最终预测结果
        prediction_scores = self.generator_lm_head(prediction_scores)

        loss = None
        # 如果提供了 labels，则计算损失
        # 遮罩语言建模的 softmax 层
        if labels is not None:
            # 使用交叉熵损失函数，忽略 -100 索引（填充标记）
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            # 计算损失值
            loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果 return_dict 为 False，则返回元组形式的输出
        if not return_dict:
            output = (prediction_scores,) + generator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则返回 MaskedLMOutput 对象
        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=generator_hidden_states.hidden_states,
            attentions=generator_hidden_states.attentions,
        )
class ConvBertClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        # 初始化一个全连接层，输入和输出维度都是 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 根据配置选择分类器的 dropout 比例，如果未指定则使用隐藏层 dropout 比例
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 定义一个 dropout 层，用于在训练过程中随机失活输入张量
        self.dropout = nn.Dropout(classifier_dropout)
        # 输出层全连接层，输入维度为 config.hidden_size，输出维度为 config.num_labels
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        self.config = config

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        # 取隐藏状态张量的第一个位置的特征向量作为输出
        x = hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        # 根据配置中指定的激活函数对全连接层的输出进行非线性变换
        x = ACT2FN[self.config.hidden_act](x)
        x = self.dropout(x)
        # 将处理后的特征向量传入输出层全连接层，得到最终的分类预测结果
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
    def __init__(self, config):
        super().__init__(config)
        # 标签数量
        self.num_labels = config.num_labels
        self.config = config
        # ConvBERT 模型主体
        self.convbert = ConvBertModel(config)
        # 分类任务的头部
        self.classifier = ConvBertClassificationHead(config)

        # 初始化模型权重并进行后处理
        self.post_init()

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
        # 如果 return_dict 参数为 None，则使用 self.config.use_return_dict 的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 ConvBert 模型进行前向传播，获取输出结果
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

        # 从 ConvBert 的输出中获取序列输出
        sequence_output = outputs[0]
        # 将序列输出 logits 输入分类器，得到分类结果
        logits = self.classifier(sequence_output)

        # 初始化损失值为 None
        loss = None
        # 如果 labels 参数不为 None，则计算损失
        if labels is not None:
            # 如果问题类型未设置，则根据 num_labels 设置问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型选择损失函数和计算损失
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    # 对于单个标签的回归任务，计算均方误差损失
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    # 对于多标签的回归任务，计算均方误差损失
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # 对于单标签分类任务，使用交叉熵损失函数
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # 对于多标签分类任务，使用带Logits的二元交叉熵损失函数
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果 return_dict 为 False，则返回输出元组
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则返回 SequenceClassifierOutput 对象
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    ConvBERT Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    CONVBERT_START_DOCSTRING,
)
class ConvBertForMultipleChoice(ConvBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化 ConvBERT 模型
        self.convbert = ConvBertModel(config)
        # 初始化用于序列摘要的对象
        self.sequence_summary = SequenceSummary(config)
        # 初始化多选题分类的线性层
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
    ):
        """
        前向传播函数，接受多个输入参数并返回模型输出。

        Args:
            input_ids (Optional[torch.LongTensor], optional): 输入 token 的 IDs. Defaults to None.
            attention_mask (Optional[torch.FloatTensor], optional): 表示每个 token 的 attention mask. Defaults to None.
            token_type_ids (Optional[torch.LongTensor], optional): 区分不同句子的 token type IDs. Defaults to None.
            position_ids (Optional[torch.LongTensor], optional): 句子中每个 token 的位置 IDs. Defaults to None.
            head_mask (Optional[torch.FloatTensor], optional): 用于屏蔽不同 attention heads 的掩码. Defaults to None.
            inputs_embeds (Optional[torch.FloatTensor], optional): 直接提供的嵌入输入. Defaults to None.
            labels (Optional[torch.LongTensor], optional): 多选题的标签. Defaults to None.
            output_attentions (Optional[bool], optional): 是否输出 attention. Defaults to None.
            output_hidden_states (Optional[bool], optional): 是否输出隐藏状态. Defaults to None.
            return_dict (Optional[bool], optional): 是否返回字典格式的输出. Defaults to None.

        Returns:
            输出结果，通常为多选题模型的分类输出.
        """
        # 在 ConvBERT 模型上进行前向传播
        return self.convbert(
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
    ) -> Union[Tuple, MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 确定是否返回字典格式的输出结果，如果未指定则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 计算选择题数量，根据输入的 input_ids 或 inputs_embeds 的第二个维度确定
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 将输入的 input_ids 调整为二维张量，以便于后续处理，如果为 None 则置为 None
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        # 将输入的 attention_mask 调整为二维张量，以便于后续处理，如果为 None 则置为 None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # 将输入的 token_type_ids 调整为二维张量，以便于后续处理，如果为 None 则置为 None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        # 将输入的 position_ids 调整为二维张量，以便于后续处理，如果为 None 则置为 None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        # 将输入的 inputs_embeds 调整为三维张量，以便于后续处理，如果为 None 则置为 None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 调用 ConvBERT 模型进行前向传播计算
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

        # 对序列输出进行汇总
        pooled_output = self.sequence_summary(sequence_output)
        # 对汇总后的输出进行分类预测
        logits = self.classifier(pooled_output)
        # 调整 logits 的形状以便与标签进行比较
        reshaped_logits = logits.view(-1, num_choices)

        # 初始化损失值
        loss = None
        # 如果存在标签，则计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 如果不需要返回字典格式的输出，则按原始格式返回结果
        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典格式的输出，则构建 MultipleChoiceModelOutput 对象并返回
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    ConvBERT Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    CONVBERT_START_DOCSTRING,
)
class ConvBertForTokenClassification(ConvBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels  # 从配置中获取标签数量

        self.convbert = ConvBertModel(config)  # 初始化 ConvBERT 模型
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)  # 初始化 dropout 层
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)  # 初始化分类器线性层

        # 初始化权重并应用最终处理
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
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 如果 return_dict 不为 None，则使用传入的 return_dict；否则使用配置中的 use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给 ConvBert 模型进行处理，并获得输出
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

        # 从 ConvBert 模型的输出中获取序列输出（即隐藏状态的最后一层）
        sequence_output = outputs[0]

        # 对序列输出进行 dropout 处理
        sequence_output = self.dropout(sequence_output)
        # 使用分类器对 dropout 后的序列输出进行分类得到 logits
        logits = self.classifier(sequence_output)

        # 初始化损失为 None
        loss = None
        # 如果提供了 labels，则计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # 将 logits 和 labels 展平为二维张量进行损失计算
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果 return_dict 为 False，则按非字典格式输出结果
        if not return_dict:
            # 将 logits 和 ConvBert 模型的其他输出组合成一个元组输出
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则构建 TokenClassifierOutput 对象进行输出
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    ConvBERT Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    CONVBERT_START_DOCSTRING,
)
class ConvBertForQuestionAnswering(ConvBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.convbert = ConvBertModel(config)  # 初始化 ConvBERT 模型
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)  # 线性层用于输出 span 的起始和结束 logits

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(CONVBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
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
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        前向传播函数，接收输入参数，并返回模型的输出结果。

        Args:
            input_ids (Optional[torch.LongTensor], optional): 输入序列的 token IDs. Defaults to None.
            attention_mask (Optional[torch.FloatTensor], optional): 注意力遮罩，掩盖要忽略的位置. Defaults to None.
            token_type_ids (Optional[torch.LongTensor], optional): 区分不同序列的 token 类型 IDs. Defaults to None.
            position_ids (Optional[torch.LongTensor], optional): 指定输入 token 的位置 IDs. Defaults to None.
            head_mask (Optional[torch.FloatTensor], optional): 多头注意力机制中指定屏蔽的头. Defaults to None.
            inputs_embeds (Optional[torch.FloatTensor], optional): 直接输入的嵌入表示. Defaults to None.
            start_positions (Optional[torch.LongTensor], optional): 答案 span 的起始位置. Defaults to None.
            end_positions (Optional[torch.LongTensor], optional): 答案 span 的结束位置. Defaults to None.
            output_attentions (Optional[bool], optional): 是否返回注意力权重. Defaults to None.
            output_hidden_states (Optional[bool], optional): 是否返回隐藏状态. Defaults to None.
            return_dict (Optional[bool], optional): 是否以字典形式返回输出. Defaults to None.

        Returns:
            模型的输出，通常是一个 QuestionAnsweringModelOutput 对象.
        """
        # 略过对输入参数的处理和组合

        # 调用 ConvBERT 模型的 forward 方法，生成隐藏状态
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

        # 使用线性层计算答案 span 的起始和结束 logits
        logits = self.qa_outputs(outputs[0])

        # 构建模型输出结果
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # 返回模型输出
        return QuestionAnsweringModelOutput(
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
        )
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
        # 如果 return_dict 不为空，则使用 return_dict；否则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 ConvBERT 模型，传入参数进行前向传播
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

        # 从模型输出中获取序列输出
        sequence_output = outputs[0]

        # 将序列输出传入 QA 输出层获取 logits
        logits = self.qa_outputs(sequence_output)

        # 将 logits 按最后一个维度分割为 start_logits 和 end_logits
        start_logits, end_logits = logits.split(1, dim=-1)

        # 去除多余的维度，并确保连续的内存布局
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 如果 start_positions 或 end_positions 是多维的，则压缩成一维
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            
            # 忽略超出模型输入范围的 start/end positions
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 定义交叉熵损失函数，忽略 ignore_index
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            # 如果 return_dict 为 False，则返回元组格式的输出
            output = (start_logits, end_logits) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 如果 return_dict 为 True，则返回 QuestionAnsweringModelOutput 类型的对象
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```