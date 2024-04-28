# `.\transformers\models\nystromformer\modeling_nystromformer.py`

```py
# 设定编码格式为 UTF-8
# 版权声明，版权归 UW-Madison The HuggingFace Inc. 团队所有
#
# 根据 Apache 许可证 2.0 版本使用本文件
# 除非符合许可证规定，否则不得使用本文件
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按"原样"分发
# 无论是明示还是暗示的，都没有对本软件的任何保证或条件
# 请参阅许可证以获取具体的语言授权和限制
""" PyTorch Nystromformer model."""

# 导入所需库
import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入自定义的激活函数映射表
from ...activations import ACT2FN
# 导入模型输出类型
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
# 导入模型工具函数
from ...modeling_utils import PreTrainedModel
# 导入 PyTorch 工具函数
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
# 导入日志记录工具
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
# 导入 Nystromformer 配置类
from .configuration_nystromformer import NystromformerConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置
_CHECKPOINT_FOR_DOC = "uw-madison/nystromformer-512"
_CONFIG_FOR_DOC = "NystromformerConfig"

# 预训练模型的存档列表
NYSTROMFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "uw-madison/nystromformer-512",
    # 查看所有 Nyströmformer 模型 https://huggingface.co/models?filter=nystromformer
]

# NystromformerEmbeddings 类定义
class NystromformerEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    # 这是 __init__ 方法，用于初始化模型的各种层和参数
    def __init__(self, config):
        # 调用父类的 __init__ 方法
        super().__init__()
        # 创建单词嵌入层，输入为词汇表大小，输出为隐藏层大小，支持填充
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 创建位置嵌入层，输入为最大位置编码长度加2（起始和结束标记），输出为隐藏层大小
        self.position_embeddings = nn.Embedding(config.max_position_embeddings + 2, config.hidden_size)
        # 创建token类型嵌入层，输入为token类型数量，输出为隐藏层大小
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
    
        # 创建层归一化层，输入为隐藏层大小，eps为层归一化的微小数
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建dropout层，概率为隐藏层dropout概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
        # 创建位置ID缓冲区，大小为(1, max_position_embeddings)，值为[2, 3, ..., max_position_embeddings+1]
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)) + 2, persistent=False
        )
        # 获取位置嵌入类型，默认为"absolute"
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 创建token类型ID缓冲区，大小为(1, max_position_embeddings)，全为0
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
            persistent=False,
        )
    
    # 这是 forward 方法，用于计算嵌入向量
    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        # 如果输入ID存在，获取输入形状
        if input_ids is not None:
            input_shape = input_ids.size()
        # 如果输入嵌入存在，获取输入形状
        else:
            input_shape = inputs_embeds.size()[:-1]
    
        # 获取序列长度
        seq_length = input_shape[1]
    
        # 如果位置ID不存在，使用预定义的位置ID缓冲区
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
    
        # 如果token类型ID不存在，使用预定义的token类型ID缓冲区
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
    
        # 如果输入嵌入不存在，使用单词嵌入层计算输入嵌入
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # 计算token类型嵌入
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
    
        # 将输入嵌入和token类型嵌入相加
        embeddings = inputs_embeds + token_type_embeddings
        # 如果位置嵌入类型为"absolute"，则加上位置嵌入
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        # 进行层归一化和dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        # 返回最终的嵌入向量
        return embeddings
class NystromformerSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        # 调用父类构造函数
        super().__init__()
        # 检查隐藏大小是否能被注意力头数整除，若不能则抛出数值错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 设置注意力头数和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 设置Nystromformer的超参数
        self.num_landmarks = config.num_landmarks
        self.seq_len = config.segment_means_seq_len
        self.conv_kernel_size = config.conv_kernel_size

        # 初始化选项，用于计算初始化系数
        if config.inv_coeff_init_option:
            self.init_option = config["inv_init_coeff_option"]
        else:
            self.init_option = "original"

        # 初始化查询、键和值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 设置dropout层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )

        # 如果卷积核大小不为None，则初始化卷积层
        if self.conv_kernel_size is not None:
            self.conv = nn.Conv2d(
                in_channels=self.num_attention_heads,
                out_channels=self.num_attention_heads,
                kernel_size=(self.conv_kernel_size, 1),
                padding=(self.conv_kernel_size // 2, 0),
                bias=False,
                groups=self.num_attention_heads,
            )

    # 通过迭代方法近似Moore-Penrose逆的函数
    def iterative_inv(self, mat, n_iter=6):
        # 创建单位矩阵
        identity = torch.eye(mat.size(-1), device=mat.device)
        key = mat

        # 由于softmax，key的条目为正且||key||_{\infty} = 1
        if self.init_option == "original":
            # 这个原始实现更保守，用于计算Z_0的系数
            value = 1 / torch.max(torch.sum(key, dim=-2)) * key.transpose(-1, -2)
        else:
            # 这是精确系数计算，初始化Z_0的1 / ||key||_1，导致更快的收敛
            value = 1 / torch.max(torch.sum(key, dim=-2), dim=-1).values[:, :, None, None] * key.transpose(-1, -2)

        # 迭代n_iter次
        for _ in range(n_iter):
            key_value = torch.matmul(key, value)
            value = torch.matmul(
                0.25 * value,
                13 * identity
                - torch.matmul(key_value, 15 * identity - torch.matmul(key_value, 7 * identity - key_value)),
            )
        # 返回近似的逆矩阵
        return value
    # 对输入的 layer 进行转置操作，用于计算注意力分数
    def transpose_for_scores(self, layer):
        # 计算新的 layer 形状，最后两个维度被分成 num_attention_heads 和 attention_head_size
        new_layer_shape = layer.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # 根据新的形状调整 layer 的维度
        layer = layer.view(*new_layer_shape)
        # 对 layer 进行转置，交换第二维和第三维的位置
        return layer.permute(0, 2, 1, 3)
    # 定义前向传播函数，接受隐藏状态、注意力掩码和是否输出注意力作为输入参数
    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        # 通过查询权重计算混合查询层
        mixed_query_layer = self.query(hidden_states)

        # 通过键权重计算键层，并为乘法准备转置
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        # 通过值权重计算值层，并为乘法准备转置
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        # 对混合查询层进行转置以匹配矩阵相乘
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 对查询层除以 sqrt(sqrt(头大小))贡献的缩放
        query_layer = query_layer / math.sqrt(math.sqrt(self.attention_head_size))
        # 对键层除以 sqrt(sqrt(头大小))贡献的缩放
        key_layer = key_layer / math.sqrt(math.sqrt(self.attention_head_size))

        # 如果 num_landmarks 等于序列长度
        if self.num_landmarks == self.seq_len:
            # 计算注意力分数
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

            # 如果存在注意力掩码
            if attention_mask is not None:
                # 应用事先计算好的注意力掩码（在 NystromformerModel forward() 函数中预计算）
                attention_scores = attention_scores + attention_mask

            # 使用 softmax 函数计算注意力概率
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)
            # 计算上下文层
            context_layer = torch.matmul(attention_probs, value_layer)

        # 如果 num_landmarks 不等于序列长度
        else:
            # 重塑查询层和键层以计算注意力分数
            q_landmarks = query_layer.reshape(
                -1,
                self.num_attention_heads,
                self.num_landmarks,
                self.seq_len // self.num_landmarks,
                self.attention_head_size,
            ).mean(dim=-2)
            k_landmarks = key_layer.reshape(
                -1,
                self.num_attention_heads,
                self.num_landmarks,
                self.seq_len // self.num_landmarks,
                self.attention_head_size,
            ).mean(dim=-2)

            # 计算三个不同的核函数
            kernel_1 = torch.nn.functional.softmax(torch.matmul(query_layer, k_landmarks.transpose(-1, -2)), dim=-1)
            kernel_2 = torch.nn.functional.softmax(torch.matmul(q_landmarks, k_landmarks.transpose(-1, -2)), dim=-1)

            # 计算注意力分数
            attention_scores = torch.matmul(q_landmarks, key_layer.transpose(-1, -2))

            # 如果存在注意力掩码
            if attention_mask is not None:
                # 应用事先计算好的注意力掩码（在 NystromformerModel forward() 函数中预计算）
                attention_scores = attention_scores + attention_mask

            # 使用 softmax 函数计算注意力概率
            kernel_3 = nn.functional.softmax(attention_scores, dim=-1)
            # 计算并修正注意力分数
            attention_probs = torch.matmul(kernel_1, self.iterative_inv(kernel_2))
            new_value_layer = torch.matmul(kernel_3, value_layer)
            context_layer = torch.matmul(attention_probs, new_value_layer)

        # 如果存在卷积核大小
        if self.conv_kernel_size is not None:
            # 将上下文层与卷积计算的结果相加
            context_layer += self.conv(value_layer)

        # 对上下文层重新排列维度
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # 重新调整上下文层的形状
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # 如果需要输出注意力，将注意力概率添加到输出中
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        # 返回输出结果
        return outputs
# 从transformers.models.bert.modeling_bert.BertSelfOutput中拷贝代码，创建NystromformerSelfOutput类
class NystromformerSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，输入和输出维度均为config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建LayerNorm层，输入维度为config.hidden_size，eps值为config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个Dropout层，丢弃概率为config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，接受hidden_states和input_tensor两个torch.Tensor类型的输入参数，返回torch.Tensor类型的输出
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将hidden_states输入通过全连接层得到输出
        hidden_states = self.dense(hidden_states)
        # 对输出进行dropout操作
        hidden_states = self.dropout(hidden_states)
        # 将dropout后的输出和input_tensor相加，然后经过LayerNorm层
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回LayerNorm处理后的输出
        return hidden_states


# 创建NystromformerAttention类，继承自nn.Module
class NystromformerAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 创建NystromformerSelfAttention实例
        self.self = NystromformerSelfAttention(config, position_embedding_type=position_embedding_type)
        # 创建NystromformerSelfOutput实例
        self.output = NystromformerSelfOutput(config)
        # 创建一个空的集合，用于存储需要裁剪的头信息
        self.pruned_heads = set()

    # 裁剪头信息的方法
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 根据传入的头信息，计算需要裁剪的头信息和索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 裁剪线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储被裁剪的头信息
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 前向传播函数
    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        # 调用self.self的前向传播方法，得到self_outputs
        self_outputs = self.self(hidden_states, attention_mask, output_attentions)
        # 将self_outputs[0]输入到self.output中，得到attention_output
        attention_output = self.output(self_outputs[0], hidden_states)
        # 将attention_output存入outputs中，如果需要输出注意力权重，则也加入到outputs中
        outputs = (attention_output,) + self_outputs[1:]
        # 返回outputs
        return outputs


# 从transformers.models.bert.modeling_bert.BertIntermediate中拷贝代码，创建NystromformerIntermediate类
class NystromformerIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，输入维度为config.hidden_size，输出维度为config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 判断config.hidden_act是否为字符串，如果是则根据映射关系找到对应的激活函数，如果不是则直接使用config.hidden_act
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播函数，接受hidden_states作为输入参数，返回处理后的hidden_states
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将hidden_states输入通过全连接层得到输出
        hidden_states = self.dense(hidden_states)
        # 将输出输入激活函数进行处理
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回处理后的hidden_states
        return hidden_states
# 定义一个自定义的 NystromformerOutput 类，继承自 nn.Module
class NystromformerOutput(nn.Module):
    # 初始化函数，接受 config 参数
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，将隐藏尺寸从 config.intermediate_size 转换为 config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个 LayerNorm 层，对隐藏状态进行归一化处理
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个 Dropout 层，进行随机失活操作
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，接受隐藏状态 hidden_states 和输入张量 input_tensor，返回处理后的隐藏状态张量
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态通过全连接层 dense 进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的隐藏状态进行随机失活操作
        hidden_states = self.dropout(hidden_states)
        # 将经过处理的隐藏状态与输入张量 input_tensor 相加后通过 LayerNorm 进行归一化处理
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的隐藏状态张量
        return hidden_states


# 定义一个自定义的 NystromformerLayer 类，继承自 nn.Module
class NystromformerLayer(nn.Module):
    # 初始化函数，接受 config 参数
    def __init__(self, config):
        super().__init__()
        # 设定 feed forward 的 chunk_size，定义 seq_len_dim 为 1
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        # 创建一个 NystromformerAttention 实例，用于处理注意力机制
        self.attention = NystromformerAttention(config)
        # 根据配置参数判断是否添加跨层注意力机制
        self.add_cross_attention = config.add_cross_attention
        # 创建一个 NystromformerIntermediate 实例，用于处理中间层操作
        self.intermediate = NystromformerIntermediate(config)
        # 创建一个 NystromformerOutput 实例，用于输出层的操作
        self.output = NystromformerOutput(config)

    # 前向传播函数，接受隐藏状态 hidden_states、注意力遮罩 attention_mask 和是否输出注意力权重 output_attentions
    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        # 通过 self.attention 处理注意力机制，获取 self_attention_outputs
        self_attention_outputs = self.attention(hidden_states, attention_mask, output_attentions=output_attentions)
        attention_output = self_attention_outputs[0]  # 获取注意力输���

        # 如果需要输出注意力权重，则将 self_attention_outputs 中的除了注意力输出之外的内容加入 outputs 中
        outputs = self_attention_outputs[1:]

        # 将注意力输出通过 apply_chunking_to_forward 分块处理，获取层输出 layer_output
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    # feed forward_chunk 函数，用于处理注意力输出并返回层输出
    def feed_forward_chunk(self, attention_output):
        # 使用 intermediate 处理注意力输出得到中间输出
        intermediate_output = self.intermediate(attention_output)
        # 使用 output 处理中间输出和注意力输出得到层输出
        layer_output = self.output(intermediate_output, attention_output)
        # 返回层输出
        return layer_output


# 定义一个自定义的 NystromformerEncoder 类，继承自 nn.Module
class NystromformerEncoder(nn.Module):
    # 初始化函数，接受 config 参数
    def __init__(self, config):
        super().__init__()
        # 将配置参数存储在 self.config 中
        self.config = config
        # 创建一个 nn.ModuleList，包含 config.num_hidden_layers 个 NystromformerLayer 实例
        self.layer = nn.ModuleList([NystromformerLayer(config) for _ in range(config.num_hidden_layers)])
        # 初始化 gradient_checkpointing 为 False
        self.gradient_checkpointing = False

    # 前向传播函数，接受 hidden_states、attention_mask、head_mask 等参数，返回输出结果
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        # 如果需要输出隐藏状态，则初始化一个空的元组，否则置为None
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力权重，则初始化一个空的元组，否则置为None
        all_self_attentions = () if output_attentions else None

        # 遍历每个层次的 Transformer 层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前层的隐藏状态添加到所有隐藏状态的元组中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果启用渐变检查点并且处于训练模式，则使用渐变检查点函数执行当前层的前向传播
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            # 否则，直接调用当前层的前向传播
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, output_attentions)

            # 更新隐藏状态为当前层的输出隐藏状态
            hidden_states = layer_outputs[0]
            # 如果需要输出注意力权重，则将当前层的注意力权重添加到所有注意力权重的元组中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果需要输出隐藏状态，则将最终隐藏状态添加到所有隐藏状态的元组中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要返回字典形式的输出，则以元组形式返回隐藏状态、所有隐藏状态和所有注意力权重
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 如果需要返回字典形式的输出，则以字典形式返回最终隐藏状态、所有隐藏状态和所有注意力权重
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
# 从transformers.models.bert.modeling_bert.BertPredictionHeadTransform复制代码到NystromformerPredictionHeadTransform，替换Bert为Nystromformer
class NystromformerPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # 创建线性变换层
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]  # 若隐藏激活函数是字符串，则使用对应的激活函数
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # Layer Normalization层

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)  # 输入经过线性变换
        hidden_states = self.transform_act_fn(hidden_states)  # 输入经过激活函数
        hidden_states = self.LayerNorm(hidden_states)  # 输入经过Layer Normalization
        return hidden_states


# 从transformers.models.bert.modeling_bert.BertLMPredictionHead复制代码到NystromformerLMPredictionHead，替换Bert为Nystromformer
class NystromformerLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = NystromformerPredictionHeadTransform(config)  # 创建NystromformerPredictionHeadTransform对象

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)  # 创建线性层，输出与输入嵌入向量维度相同，无偏置项
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))  # 初始化偏置项为全零

        self.decoder.bias = self.bias  # 修正偏置项大小以与resize_token_embeddings正确调整大小之间建立连接

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)  # 输入经过NystromformerPredictionHeadTransform
        hidden_states = self.decoder(hidden_states)  # 输入经过线性层
        return hidden_states


# 从transformers.models.bert.modeling_bert.BertOnlyMLMHead复制代码到NystromformerOnlyMLMHead，替换Bert为Nystromformer
class NystromformerOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = NystromformerLMPredictionHead(config)  # 创建NystromformerLMPredictionHead对象

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)  # 输入经过NystromformerLMPredictionHead
        return prediction_scores


class NystromformerPreTrainedModel(PreTrainedModel):
    """
    处理权重初始化和简单接口以下载和加载预训练模型的抽象类。
    """

    config_class = NystromformerConfig  # 设置配置类为NystromformerConfig
    base_model_prefix = "nystromformer"  # 模型的前缀名为nystromformer
    supports_gradient_checkpointing = True  # 支持梯度检查点
    def _init_weights(self, module):
        """Initialize the weights"""  # 初始化权重的函数
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 如果模块是线性层或卷积层
            # 与 TF 版本稍有不同，TF 版本使用截断正态分布进行初始化
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 初始化权重为正态分布
            if module.bias is not None:
                # 如果存在偏置项，初始化为零
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 如果模块是嵌入层
            # 初始化权重为正态分布
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                # 如果存在填充索引，将填充索引处的权重初始化为零
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 如果模块是 LayerNorm 层
            # 初始化偏置项为零
            module.bias.data.zero_()
            # 初始化权重为1
            module.weight.data.fill_(1.0)
# 定义Nystromformer模型的文档字符串，提供模型的用法说明和参数信息
NYSTROMFORMER_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`NystromformerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义Nystromformer模型的输入参数文档字符串
NYSTROMFORMER_INPUTS_DOCSTRING = r"""
    # 定义函数的输入参数
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            输入序列标记在词汇表中的索引。
    
            可以使用 [`AutoTokenizer`] 获取。详见 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`]。
    
            [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            避免对填充的标记索引执行注意力操作的掩码。选取的掩码值范围是`[0, 1]`：
    
            - 对于**未被掩码**的标记，取 1，
            - 对于**被掩码**的标记，取 0。
    
            [什么是注意力掩码？](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            指明输入序列的第一部分和第二部分的分段标记索引。选取的索引范围是`[0, 1]`：
    
            - 0 对应*句子 A* 的标记，
            - 1 对应*句子 B* 的标记。
    
            [什么是分段标记 ID？](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            指明每个输入序列标记在位置嵌入中的位置索引。选取的索引范围是`[0, config.max_position_embeddings - 1]`。
    
            [什么是位置 ID？](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            空值化自注意力模块中所选头部的掩码。选取的掩码值范围是`[0, 1]`：
    
            - 1 表示该头部**未被掩码**，
            - 0 表示该头部**被掩码**。
    
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            可选，而非传递 `input_ids`，可以选择直接传递嵌入表示法。如果你希望对如何将 *input_ids* 索引转换为相关向量有更多的控制权，那么这很有用。模型的内部嵌入查找矩阵。
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回的张量下的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。有关更多详细信息，请参见返回的张量下的 `hidden_states`。
        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而不是普通的元组。
"""
定义一个Nyströmformer模型的Transformer，输出原始的隐藏状态，没有特定的头部
""""
class NystromformerModel(NystromformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = NystromformerEmbeddings(config)
        self.encoder = NystromformerEncoder(config)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        剪除模型中的头部. heads_to_prune: {层号: 需要在此层剪除的头部列表} 参见基类 PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

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
"""

"""
定义一个带有`语言建模`头部的Nyströmformer模型
"""
class NystromformerForMaskedLM(NystromformerPreTrainedModel):
    _tied_weights_keys = ["cls.predictions.decoder"]

    def __init__(self, config):
        super().__init__(config)

        self.nystromformer = NystromformerModel(config)
        self.cls = NystromformerOnlyMLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的token序列的ID，可选
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力遮罩，可选
        token_type_ids: Optional[torch.LongTensor] = None,  # token类型的ID，可选
        position_ids: Optional[torch.LongTensor] = None,  # 位置ID，可选
        head_mask: Optional[torch.FloatTensor] = None,  # 注意力头部的遮罩，可选
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入表示，可选
        labels: Optional[torch.LongTensor] = None,  # 用于计算MLM损失的标签，可选
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选
        return_dict: Optional[bool] = None,  # 是否返回字典形式的输出，可选
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # 如果return_dict为None，则使用配置中的设置

        outputs = self.nystromformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )  # 通过Nystromformer进行前向传播

        sequence_output = outputs[0]  # 获取序列输出
        prediction_scores = self.cls(sequence_output)  # 通过分类器获取预测分数

        masked_lm_loss = None  # 初始化MLM损失为None
        if labels is not None:  # 如果存在标签
            loss_fct = CrossEntropyLoss()  # 创建交叉熵损失函数
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))  # 计算MLM损失

        if not return_dict:  # 如果不返回字典形式的输出
            output = (prediction_scores,) + outputs[1:]  # 组装输出
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output  # 返回输出和MLM损失（如果存在）

        return MaskedLMOutput(
            loss=masked_lm_loss,  # MLM损失
            logits=prediction_scores,  # 预测分数
            hidden_states=outputs.hidden_states,  # 隐藏状态
            attentions=outputs.attentions,  # 注意力权重
        )  # 返回MLM模型输出
class NystromformerClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        # 初始化方法，接收config参数
        super().__init__()
        # 创建线性层，将输入维度为config.hidden_size的数据映射到config.hidden_size的维度上
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个dropout层，以概率config.hidden_dropout_prob进行随机丢弃一部分神经元
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 创建线性层，将输入维度为config.hidden_size的数据映射到config.num_labels的维度上
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        # 存储配置信息
        self.config = config

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # 取首个特征token的输出作为分类任务的输入（对应[CLS]）
        x = self.dropout(x)  # 对输入进行dropout
        x = self.dense(x)  # 使用dense层映射特征
        x = ACT2FN[self.config.hidden_act](x)  # 使用激活函数激活映射后的特征
        x = self.dropout(x)  # 再次进行dropout
        x = self.out_proj(x)  # 最终将特征映射到类别数量上
        return x  # 返回分类结果


@add_start_docstrings(
    """
    Nyströmformer Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    NYSTROMFORMER_START_DOCSTRING,
)
class NystromformerForSequenceClassification(NystromformerPreTrainedModel):
    def __init__(self, config):
        # 初始化方法，接收config参数
        super().__init__(config)
        # 获取config中的类别数量信息
        self.num_labels = config.num_labels
        # 创建Nyströmformer模型实例
        self.nystromformer = NystromformerModel(config)
        # 创建分类器实例
        self.classifier = NystromformerClassificationHead(config)

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(NYSTROMFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 如果 return_dict 不为 None，则使用该值；否则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 NystromFormer 模型进行前向传播
        outputs = self.nystromformer(
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
        # 将序列输出传递给分类器以获取 logits
        logits = self.classifier(sequence_output)

        # 初始化损失为 None
        loss = None
        # 如果 labels 不为 None，则计算损失
        if labels is not None:
            # 如果配置中的问题类型为 None，则根据情况自动设置问题类型
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
        # 如果 return_dict 为 False，则返回 logits 和额外的输出；否则返回完整的 SequenceClassifierOutput
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 这是一个 NystromformerForMultipleChoice 类，它继承自 NystromformerPreTrainedModel
# 该类用于多项选择分类任务，如 RocStories/SWAG 任务
@add_start_docstrings(
    """
    Nyströmformer Model with a multiple choice classification head on top (a linear layer on top of the pooled output
    and a softmax) e.g. for RocStories/SWAG tasks.
    """,
    NYSTROMFORMER_START_DOCSTRING,
)
class NystromformerForMultipleChoice(NystromformerPreTrainedModel):
    def __init__(self, config):
        # 调用父类的构造方法
        super().__init__(config)

        # 创建 NystromformerModel 对象
        self.nystromformer = NystromformerModel(config)
        # 创建一个线性层，将输入的隐藏状态映射到相同大小的隐藏状态
        self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个线性层，将隐藏状态映射到一个标量输出，用于多项选择分类
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    # 为模型的前向传播方法添加文档字符串
    @add_start_docstrings_to_model_forward(
        NYSTROMFORMER_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
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
        ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 确定是否返回字典类型的结果，如果未指定，则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 获取输入的选择数量
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 重新调整输入的形状以便适应模型输入
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 调用 NystromFormer 模型进行前向传播
        outputs = self.nystromformer(
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

        # 获取模型输出中的隐藏状态
        hidden_state = outputs[0]  # (bs * num_choices, seq_len, dim)
        # 获取池化后的输出
        pooled_output = hidden_state[:, 0]  # (bs * num_choices, dim)
        # 将池化输出传递给预分类器
        pooled_output = self.pre_classifier(pooled_output)  # (bs * num_choices, dim)
        # 使用 ReLU 激活函数激活池化输出
        pooled_output = nn.ReLU()(pooled_output)  # (bs * num_choices, dim)
        # 将激活后的输出传递给分类器得到最终预测结果
        logits = self.classifier(pooled_output)

        # 重新调整输出的形状以便计算损失
        reshaped_logits = logits.view(-1, num_choices)

        # 如果提供了标签，则计算交叉熵损失
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 如果不要求返回字典类型的结果，则返回结果的元组形式
        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回字典类型的结果
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 为 Nyströmformer 模型添加一个用于token分类的头部(一个线性层在隐状态输出的顶端)
# 例如用于命名实体识别(NER)任务
@add_start_docstrings(
    """
    Nyströmformer Model with a token classification head on top (a linear layer on top of the hidden-states output)
    e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    NYSTROMFORMER_START_DOCSTRING,
)
class NystromformerForTokenClassification(NystromformerPreTrainedModel):
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)
        # 设置标签数量
        self.num_labels = config.num_labels

        # 实例化 Nyströmformer 模型
        self.nystromformer = NystromformerModel(config)
        # 添加dropout层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 添加线性层进行分类
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并进行最终处理
        self.post_init()

    # 添加对模型输入的描述
    @add_start_docstrings_to_model_forward(NYSTROMFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 添加输出类型和配置类的文档
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        # 接受输入参数
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
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional`):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 确定是否返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用Nystromformer模型处理输入数据
        outputs = self.nystromformer(
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

        # 对序列输出进行Dropout操作
        sequence_output = self.dropout(sequence_output)
        # 使用分类器处理序列输出，得到预测的logits
        logits = self.classifier(sequence_output)

        # 初始化损失
        loss = None
        # 如果存在标签
        if labels is not None:
            # 定义交叉熵损失函数
            loss_fct = CrossEntropyLoss()
            # 计算损失
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不返回字典
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回分类输出对象
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 定义一个带有用于提取问题-回答任务的跨特征图形式和输出分类头部的 Nyströmformer 模型（在隐藏状态输出之上的线性层用于计算 `跨起始内容概率` 和 `跨结束内容概率`）
@add_start_docstrings(
    """
    Nyströmformer Model with a span classification head on top for extractive question-answering tasks like SQuAD (a
    linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    NYSTROMFORMER_START_DOCSTRING,
)
class NystromformerForQuestionAnswering(NystromformerPreTrainedModel):
    # 初始化函数，接收一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)

        # 设置配置对象中的标签数量为2
        config.num_labels = 2
        self.num_labels = config.num_labels

        # 实例化 Nyströmformer 模型
        self.nystromformer = NystromformerModel(config)
        # 创建一个线性层，用于输出分类结果
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 定义前向传播方法
    @add_start_docstrings_to_model_forward(NYSTROMFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        # 输入序列的 token ID
        input_ids: Optional[torch.LongTensor] = None,
        # 注意力遮罩
        attention_mask: Optional[torch.FloatTensor] = None,
        # 分段 ID
        token_type_ids: Optional[torch.LongTensor] = None,
        # 位置 ID
        position_ids: Optional[torch.LongTensor] = None,
        # 头部遮罩
        head_mask: Optional[torch.FloatTensor] = None,
        # 输入嵌入
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # 起始位置
        start_positions: Optional[torch.LongTensor] = None,
        # 结束位置
        end_positions: Optional[torch.LongTensor] = None,
        # 输出注意力
        output_attentions: Optional[bool] = None,
        # 输出隐藏状态
        output_hidden_states: Optional[bool] = None,
        # 返回字典结果
        return_dict: Optional[bool] = None,
    # 返回类型提示函数签名，接受参数并返回一个元组或QuestionAnsweringModelOutput对象
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
        # 确定是否要返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用nystromformer，获取输出
        outputs = self.nystromformer(
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

        # 提取序列输出
        sequence_output = outputs[0]

        # 获取logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 多GPU情况下，展开维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 处理起始/结束位置超出模型输入的情况
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 计算损失
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            # 如果有损失，则返回损失和输出，否则只返回输出
            return ((total_loss,) + output) if total_loss is not None else output

        # 返回QuestionAnsweringModelOutput对象
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```