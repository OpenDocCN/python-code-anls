# `.\transformers\models\canine\modeling_canine.py`

```py
# coding=utf-8
# 版权 2021 Google AI The HuggingFace Inc. 团队保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可；
# 除非符合许可证的要求，否则不得使用此文件。
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按“原样”分发，
# 没有任何形式的担保或条件，包括但不限于
# 适销性、特定用途的适用性和非侵权性担保。
# 有关详细信息，请参见许可证。
""" PyTorch CANINE 模型。"""


import copy
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    ModelOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_canine import CanineConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "google/canine-s"
_CONFIG_FOR_DOC = "CanineConfig"

CANINE_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/canine-s",
    "google/canine-r",
    # 查看所有 CANINE 模型 https://huggingface.co/models?filter=canine
]

# 支持最多 16 个哈希函数。
_PRIMES = [31, 43, 59, 61, 73, 97, 103, 113, 137, 149, 157, 173, 181, 193, 211, 223]


@dataclass
class CanineModelOutputWithPooling(ModelOutput):
    """
    [`CanineModel`] 的输出类型。基于 [`~modeling_outputs.BaseModelOutputWithPooling`]，
    但 `hidden_states` 和 `attentions` 稍有不同，因为这些还包括浅 Transformer 编码器的隐藏状态和注意力。
    """
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            最后一层模型的隐藏状态序列（即最终浅层Transformer编码器的输出）。每个元素是一个3D张量，形状为`(batch_size, sequence_length, hidden_size)`。
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            深层Transformer编码器最后一层中第一个令牌（分类令牌）的隐藏状态，经过线性层和Tanh激活函数进一步处理。在线性层训练的权重来自于预训练期间的下一句预测（分类）目标。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            一个元组，包含每个编码器的输入和每个编码器每一层的输出的`torch.FloatTensor`（每个编码器一个）。形状为`(batch_size, sequence_length, hidden_size)`和`(batch_size, sequence_length // config.downsampling_rate, hidden_size)`。
            模型每一层的隐藏状态以及每个Transformer编码器的初始输入。浅层编码器的隐藏状态长度为`sequence_length`，但深层编码器的隐藏状态长度为`sequence_length` // `config.downsampling_rate`。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            一个元组，包含三个Transformer编码器的注意力权重的`torch.FloatTensor`（每个编码器一个）。形状为`(batch_size, num_heads, sequence_length, sequence_length)`和`(batch_size, num_heads, sequence_length // config.downsampling_rate, sequence_length // config.downsampling_rate)`。
            注意力softmax后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
def load_tf_weights_in_canine(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    # 导入必要的库
    try:
        import re  # 导入正则表达式库
        import numpy as np  # 导入 NumPy 库
        import tensorflow as tf  # 导入 TensorFlow 库
    except ImportError:  # 捕获导入错误
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise  # 抛出 ImportError
    # 获取 TensorFlow 检查点文件的绝对路径
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")  # 记录日志：转换 TensorFlow 检查点的路径
    # 从 TF 模型加载权重
    init_vars = tf.train.list_variables(tf_path)  # 列出 TensorFlow 检查点中的变量
    names = []  # 存储变量名的列表
    arrays = []  # 存储权重数组的列表
    for name, shape in init_vars:  # 遍历 TF 检查点中的变量和形状
        logger.info(f"Loading TF weight {name} with shape {shape}")  # 记录日志：加载 TF 权重的名称和形状
        array = tf.train.load_variable(tf_path, name)  # 加载 TF 检查点中的变量
        names.append(name)  # 将变量名添加到列表中
        arrays.append(array)  # 将权重数组添加到列表中

    return model  # 返回模型对象


class CanineEmbeddings(nn.Module):
    """Construct the character, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()

        self.config = config  # 存储配置信息

        # character embeddings
        shard_embedding_size = config.hidden_size // config.num_hash_functions  # 计算分片嵌入的大小
        for i in range(config.num_hash_functions):  # 遍历哈希函数数量
            name = f"HashBucketCodepointEmbedder_{i}"  # 创建哈希桶编码器的名称
            setattr(self, name, nn.Embedding(config.num_hash_buckets, shard_embedding_size))  # 动态创建哈希桶编码器
        self.char_position_embeddings = nn.Embedding(config.num_hash_buckets, config.hidden_size)  # 字符位置嵌入
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)  # 标记类型嵌入

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # 归一化层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # 随机失活层

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )  # 注册位置标识张量
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")  # 位置嵌入类型
    # 将输入的 ids 通过多重哈希转换为哈希桶 ids
    def _hash_bucket_tensors(self, input_ids, num_hashes: int, num_buckets: int):
        # 如果 num_hashes 大于 _PRIMES 列表的长度，引发 ValueError 异常
        if num_hashes > len(_PRIMES):
            raise ValueError(f"`num_hashes` must be <= {len(_PRIMES)}")

        # 选择 num_hashes 个素数作为哈希函数
        primes = _PRIMES[:num_hashes]

        # 存储每个哈希函数的结果张量的列表
        result_tensors = []
        # 遍历每个素数
        for prime in primes:
            # 计算哈希值并将其添加到结果张量列表中
            hashed = ((input_ids + 1) * prime) % num_buckets
            result_tensors.append(hashed)
        # 返回结果张量列表
        return result_tensors

    # 将 IDs（例如码点）通过多重哈希转换为嵌入
    def _embed_hash_buckets(self, input_ids, embedding_size: int, num_hashes: int, num_buckets: int):
        # 如果 embedding_size 不能被 num_hashes 整除，引发 ValueError 异常
        if embedding_size % num_hashes != 0:
            raise ValueError(f"Expected `embedding_size` ({embedding_size}) % `num_hashes` ({num_hashes}) == 0")

        # 计算哈希桶张量
        hash_bucket_tensors = self._hash_bucket_tensors(input_ids, num_hashes=num_hashes, num_buckets=num_buckets)
        # 存储嵌入分片的列表
        embedding_shards = []
        # 遍历每个哈希桶张量
        for i, hash_bucket_ids in enumerate(hash_bucket_tensors):
            # 构造分片名
            name = f"HashBucketCodepointEmbedder_{i}"
            # 获取对应的分片嵌入
            shard_embeddings = getattr(self, name)(hash_bucket_ids)
            # 将分片嵌入添加到嵌入分片列表中
            embedding_shards.append(shard_embeddings)

        # 在最后一个维度上连接嵌入分片
        return torch.cat(embedding_shards, dim=-1)

    # 前向传播函数
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        # 如果 input_ids 不为空，则获取其形状
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            # 否则获取 inputs_embeds 的形状
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度
        seq_length = input_shape[1]

        # 如果 position_ids 为空，则使用预定义的位置 ids
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # 如果 token_type_ids 为空，则创建全零张量
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果 inputs_embeds 为空，则通过 _embed_hash_buckets 函数将 input_ids 转换为嵌入
        if inputs_embeds is None:
            inputs_embeds = self._embed_hash_buckets(
                input_ids, self.config.hidden_size, self.config.num_hash_functions, self.config.num_hash_buckets
            )

        # 获取 token_type_ids 对应的嵌入
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将输入嵌入与 token 类型嵌入相加
        embeddings = inputs_embeds + token_type_embeddings

        # 如果位置嵌入类型是“absolute”，则获取字符位置嵌入并添加到总嵌入中
        if self.position_embedding_type == "absolute":
            position_embeddings = self.char_position_embeddings(position_ids)
            embeddings += position_embeddings

        # 对总嵌入进行 LayerNorm
        embeddings = self.LayerNorm(embeddings)
        # 对嵌入进行 dropout
        embeddings = self.dropout(embeddings)
        # 返回嵌入
        return embeddings
class CharactersToMolecules(nn.Module):
    """Convert character sequence to initial molecule sequence (i.e. downsample) using strided convolutions."""

    def __init__(self, config):
        super().__init__()

        # 定义一维卷积层，用于将输入的字符序列进行下采样
        self.conv = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=config.downsampling_rate,
            stride=config.downsampling_rate,
        )
        # 激活函数
        self.activation = ACT2FN[config.hidden_act]

        # 层归一化，保持 TensorFlow 模型变量名称以便能够加载任何 TensorFlow 检查点文件
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, char_encoding: torch.Tensor) -> torch.Tensor:
        # 提取字符编码中的 [CLS] 标记
        cls_encoding = char_encoding[:, 0:1, :]

        # 将输入的字符编码进行维度转换，形状变为 [batch, hidden_size, char_seq]
        char_encoding = torch.transpose(char_encoding, 1, 2)
        # 经过卷积层进行下采样
        downsampled = self.conv(char_encoding)
        # 激活函数
        downsampled = self.activation(downsampled)
        # 再次转换维度，还原为 [batch, char_seq, hidden_size]
        downsampled = torch.transpose(downsampled, 1, 2)

        # 截断最后一个分子，为 [CLS] 保留一个位置
        downsampled_truncated = downsampled[:, 0:-1, :]

        # 将 [CLS] 与下采样后的字符序列连接起来
        result = torch.cat([cls_encoding, downsampled_truncated], dim=1)

        # 对结果进行层归一化
        result = self.LayerNorm(result)

        return result


class ConvProjection(nn.Module):
    """
    Project representations from hidden_size*2 back to hidden_size across a window of w = config.upsampling_kernel_size
    characters.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        # 定义一维卷积层，用于将维度为 hidden_size*2 的表示投影回 hidden_size
        self.conv = nn.Conv1d(
            in_channels=config.hidden_size * 2,
            out_channels=config.hidden_size,
            kernel_size=config.upsampling_kernel_size,
            stride=1,
        )
        # 激活函数
        self.activation = ACT2FN[config.hidden_act]
        # 层归一化，保持 TensorFlow 模型变量名称以便能够加载任何 TensorFlow 检查点文件
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        inputs: torch.Tensor,
        final_seq_char_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 将输入张量的形状从 [batch, mol_seq, molecule_hidden_size+char_hidden_final] 转置为 [batch, molecule_hidden_size+char_hidden_final, mol_seq]
        inputs = torch.transpose(inputs, 1, 2)

        # PyTorch < 1.9 不支持 padding="same"（原始实现中使用），因此在传递给卷积层之前手动对张量进行填充
        # 参考 https://github.com/google-research/big_transfer/blob/49afe42338b62af9fbe18f0258197a33ee578a6b/bit_tf2/models.py#L36-L38
        pad_total = self.config.upsampling_kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        pad = nn.ConstantPad1d((pad_beg, pad_end), 0)
        # `result`: 形状为 (batch_size, char_seq_len, hidden_size)
        result = self.conv(pad(inputs))
        result = torch.transpose(result, 1, 2)
        result = self.activation(result)
        result = self.LayerNorm(result)
        result = self.dropout(result)
        final_char_seq = result

        if final_seq_char_positions is not None:
            # 限制变换器查询序列和注意力掩码到这些字符位置，以大大减少计算成本。通常，这仅用于 MLM 训练任务。
            # TODO 添加对 MLM 的支持
            raise NotImplementedError("CanineForMaskedLM is currently not supported")
        else:
            query_seq = final_char_seq

        return query_seq
class CanineSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 检查隐藏大小是否能被注意力头的数量整除，同时也检查是否有嵌入大小属性
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化查询、键和值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 如果位置嵌入类型为相对键或相对键查询，则创建距离嵌入层
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

    # 为得分转置
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播函数
    def forward(
        self,
        from_tensor: torch.Tensor,
        to_tensor: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    Additional arguments related to local attention:

        - **local** (`bool`, *optional*, defaults to `False`) -- Whether to apply local attention.
        - **always_attend_to_first_position** (`bool`, *optional*, defaults to `False`) -- Should all blocks be able to
          attend
        to the `to_tensor`'s first position (e.g. a [CLS] position)? - **first_position_attends_to_all** (`bool`,
        *optional*, defaults to `False`) -- Should the *from_tensor*'s first position be able to attend to all
        positions within the *from_tensor*? - **attend_from_chunk_width** (`int`, *optional*, defaults to 128) -- The
        width of each block-wise chunk in `from_tensor`. - **attend_from_chunk_stride** (`int`, *optional*, defaults to
        128) -- The number of elements to skip when moving to the next block in `from_tensor`. -
        **attend_to_chunk_width** (`int`, *optional*, defaults to 128) -- The width of each block-wise chunk in
        *to_tensor*. - **attend_to_chunk_stride** (`int`, *optional*, defaults to 128) -- The number of elements to
        skip when moving to the next block in `to_tensor`.
    """

    # 初始化方法，初始化自注意力层和输出层，并定义了一些与局部注意力相关的额外参数
    def __init__(
        self,
        config,
        local=False,  # 是否应用局部注意力，默认为False
        always_attend_to_first_position: bool = False,  # 是否所有块都能关注到to_tensor的第一个位置，默认为False
        first_position_attends_to_all: bool = False,  # 是否from_tensor的第一个位置能关注到from_tensor的所有位置，默认为False
        attend_from_chunk_width: int = 128,  # from_tensor中每个块的宽度，默认为128
        attend_from_chunk_stride: int = 128,  # 在移动到from_tensor中的下一个块时跳过的元素数，默认为128
        attend_to_chunk_width: int = 128,  # to_tensor中每个块的宽度，默认为128
        attend_to_chunk_stride: int = 128,  # 在移动到to_tensor中的下一个块时跳过的元素数，默认为128
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化自注意力层和输出层
        self.self = CanineSelfAttention(config)
        self.output = CanineSelfOutput(config)
        # 初始化一个空集合，用于记录被修剪的头部
        self.pruned_heads = set()

        # 针对局部注意力的额外参数
        # 设置是否应用局部注意力
        self.local = local
        # 如果attend_from_chunk_width小于attend_from_chunk_stride，会导致跳过序列位置
        if attend_from_chunk_width < attend_from_chunk_stride:
            # 抛出数值错误
            raise ValueError(
                "`attend_from_chunk_width` < `attend_from_chunk_stride` would cause sequence positions to get skipped."
            )
        # 如果attend_to_chunk_width小于attend_to_chunk_stride，会导致跳过序列位置
        if attend_to_chunk_width < attend_to_chunk_stride:
            # 抛出数值错误
            raise ValueError(
                "`attend_to_chunk_width` < `attend_to_chunk_stride`would cause sequence positions to get skipped."
            )
        # 设置是否始终将注意力放在to_tensor的第一个位置
        self.always_attend_to_first_position = always_attend_to_first_position
        # 设置是否from_tensor的第一个位置能够关注到from_tensor的所有位置
        self.first_position_attends_to_all = first_position_attends_to_all
        # 设置from_tensor中每个块的宽度
        self.attend_from_chunk_width = attend_from_chunk_width
        # 设置在移动到from_tensor中的下一个块时跳过的元素数
        self.attend_from_chunk_stride = attend_from_chunk_stride
        # 设置to_tensor中每个块的宽度
        self.attend_to_chunk_width = attend_to_chunk_width
        # 设置在移动到to_tensor中的下一个块时跳过的元素数
        self.attend_to_chunk_stride = attend_to_chunk_stride
    # 剪枝自注意力机制的头部
    def prune_heads(self, heads):
        # 如果传入的头部列表为空，直接返回
        if len(heads) == 0:
            return
        # 调用 find_pruneable_heads_and_indices 函数，找到可剪枝的头部及其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储已剪枝的头部
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 正向传播函数
    def forward(
        self,
        hidden_states: Tuple[torch.FloatTensor],
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
class CanineIntermediate(nn.Module):
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个全连接层，将输入大小调整为中间大小
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果隐藏激活函数是字符串类型，则选择对应的函数，否则直接使用配置中的函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播函数，接受隐藏状态张量作为输入，并返回处理后的张量
    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        # 先通过全连接层处理隐藏状态张量
        hidden_states = self.dense(hidden_states)
        # 然后使用中间激活函数处理结果
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回处理后的张量
        return hidden_states


class CanineOutput(nn.Module):
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个全连接层，将中间大小调整为隐藏大小
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个 LayerNorm 层，用于规范化隐藏状态
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个 Dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，接受隐藏状态张量和输入张量作为输入，并返回处理后的张量
    def forward(self, hidden_states: Tuple[torch.FloatTensor], input_tensor: torch.FloatTensor) -> torch.FloatTensor:
        # 先通过全连接层处理隐藏状态张量
        hidden_states = self.dense(hidden_states)
        # 再对结果进行 Dropout 处理
        hidden_states = self.dropout(hidden_states)
        # 将处理后的结果与输入张量相加，然后进行 LayerNorm 处理
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的张量
        return hidden_states


class CanineLayer(nn.Module):
    # 初始化函数，接受一系列配置参数
    def __init__(
        self,
        config,
        local,
        always_attend_to_first_position,
        first_position_attends_to_all,
        attend_from_chunk_width,
        attend_from_chunk_stride,
        attend_to_chunk_width,
        attend_to_chunk_stride,
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 设置前馈传播的块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 设置序列长度维度
        self.seq_len_dim = 1
        # 创建注意力机制对象
        self.attention = CanineAttention(
            config,
            local,
            always_attend_to_first_position,
            first_position_attends_to_all,
            attend_from_chunk_width,
            attend_from_chunk_stride,
            attend_to_chunk_width,
            attend_to_chunk_stride,
        )
        # 创建中间层对象
        self.intermediate = CanineIntermediate(config)
        # 创建输出层对象
        self.output = CanineOutput(config)

    # 前向传播函数，接受隐藏状态张量和一些可选参数，并返回处理后的张量及其他信息
    def forward(
        self,
        hidden_states: Tuple[torch.FloatTensor],
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        # 使用注意力机制处理隐藏状态张量，并获取注意力输出
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        # 如果输出注意力权重，则将它们添加到输出中
        outputs = self_attention_outputs[1:]

        # 将前馈传播函数应用于注意力输出
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        # 添加前馈传播函数的结果到输出中
        outputs = (layer_output,) + outputs

        # 返回处理后的输出
        return outputs
    # 实现神经网络的前向传播，处理一个注意力输出块
    def feed_forward_chunk(self, attention_output):
        # 将注意力输出块输入到中间层，进行神经网络的中间处理
        intermediate_output = self.intermediate(attention_output)
        # 将中间层的输出输入到输出层，得到最终的层输出
        layer_output = self.output(intermediate_output, attention_output)
        # 返回最终的层输出
        return layer_output
# 定义一个 CanineEncoder 类，继承自 nn.Module
class CanineEncoder(nn.Module):
    # 初始化方法，接收多个参数
    def __init__(
        self,
        config,
        local=False,
        always_attend_to_first_position=False,
        first_position_attends_to_all=False,
        attend_from_chunk_width=128,
        attend_from_chunk_stride=128,
        attend_to_chunk_width=128,
        attend_to_chunk_stride=128,
    ):
        super().__init__()
        # 存储传入的配置参数
        self.config = config
        # 使用 ModuleList 存储 CanineLayer 实例的列表
        self.layer = nn.ModuleList(
            [
                # 创建 CanineLayer 实例并存储到列表中
                CanineLayer(
                    config,
                    local,
                    always_attend_to_first_position,
                    first_position_attends_to_all,
                    attend_from_chunk_width,
                    attend_from_chunk_stride,
                    attend_to_chunk_width,
                    attend_to_chunk_stride,
                )
                # 根据配置参数的隐藏层数量重复创建 CanineLayer 实例
                for _ in range(config.num_hidden_layers)
            ]
        )
        # 梯度检查点标志，默认为 False
        self.gradient_checkpointing = False

    # 前向传播方法，接收多个参数并返回结果
    def forward(
        self,
        hidden_states: Tuple[torch.FloatTensor],
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutput]:
        # 如果需要输出隐藏状态，则初始化空的隐藏状态列表
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力权重，则初始化空的自注意力权重列表
        all_self_attentions = () if output_attentions else None

        # 遍历 CanineLayer 实例列表
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到列表中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 如果梯度检查点开启并且处于训练模式
            if self.gradient_checkpointing and self.training:
                # 使用梯度检查点函数对当前层进行前向传播
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                # 调用当前层的前向传播方法
                layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, output_attentions)

            # 更新隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]
            # 如果需要输出注意力权重，则将当前层的自注意力权重添加到列表中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果需要输出隐藏状态，则将最后一个隐藏状态添加到列表中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要返回字典形式的结果，则返回各项不为空的元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 返回字典形式的结果
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


# 定义一个 CaninePooler 类，继承自 nn.Module
class CaninePooler(nn.Module):
    # 初始化方法，接收一个配置参数
    def __init__(self, config):
        super().__init__()
        # 全连接层，输入维度为配置参数中的隐藏大小，输出维度为同样的隐藏大小
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # Tanh 激活函数
        self.activation = nn.Tanh()
```  
    # 前向传播函数，接受隐藏状态作为输入，并返回张量作为输出
    def forward(self, hidden_states: Tuple[torch.FloatTensor]) -> torch.FloatTensor:
        # 通过简单地选择与第一个标记对应的隐藏状态来“池化”模型。
        # 选择隐藏状态的第一个标记所对应的张量
        first_token_tensor = hidden_states[:, 0]
        # 使用全连接层处理选定的张量
        pooled_output = self.dense(first_token_tensor)
        # 应用激活函数
        pooled_output = self.activation(pooled_output)
        # 返回经过处理的张量作为输出
        return pooled_output
class CaninePredictionHeadTransform(nn.Module):
    # CaninePredictionHeadTransform 类的构造函数，接受一个配置参数 config
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，输入维度为配置参数中的隐藏层大小，输出维度为隐藏层大小
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 如果配置参数中的隐藏激活函数是字符串类型，则从预定义的激活函数字典中获取对应的激活函数；否则直接使用给定的激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # 创建一个 LayerNorm 层，输入维度为配置参数中的隐藏层大小
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # 前向传播函数，接受一个包含隐藏状态的元组，返回经过预测头变换后的隐藏状态
    def forward(self, hidden_states: Tuple[torch.FloatTensor]) -> torch.FloatTensor:
        # 将隐藏状态通过全连接层
        hidden_states = self.dense(hidden_states)
        # 应用激活函数
        hidden_states = self.transform_act_fn(hidden_states)
        # 应用 LayerNorm
        hidden_states = self.LayerNorm(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states


class CanineLMPredictionHead(nn.Module):
    # CanineLMPredictionHead 类的构造函数，接受一个配置参数 config
    def __init__(self, config):
        super().__init__()
        # 创建一个 CaninePredictionHeadTransform 实例，用于对隐藏状态进行变换
        self.transform = CaninePredictionHeadTransform(config)

        # 输出权重与输入嵌入相同，但每个标记都有一个仅针对输出的偏置
        # 创建一个线性层，输入维度为配置参数中的隐藏层大小，输出维度为词汇表大小
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 创建一个参数，用于存储输出偏置，维度为词汇表大小
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # 需要一个链接来确保偏置能够与 `resize_token_embeddings` 正确调整大小
        self.decoder.bias = self.bias

    # 前向传播函数，接受一个包含隐藏状态的元组，返回预测的分数
    def forward(self, hidden_states: Tuple[torch.FloatTensor]) -> torch.FloatTensor:
        # 首先对隐藏状态进行变换
        hidden_states = self.transform(hidden_states)
        # 将变换后的隐藏状态通过线性层，得到预测分数
        hidden_states = self.decoder(hidden_states)
        # 返回预测分数
        return hidden_states


class CanineOnlyMLMHead(nn.Module):
    # CanineOnlyMLMHead 类的构造函数，接受一个配置参数 config
    def __init__(self, config):
        super().__init__()
        # 创建一个 CanineLMPredictionHead 实例，用于预测下一个词
        self.predictions = CanineLMPredictionHead(config)

    # 前向传播函数，接受一个包含序列输出的元组，返回预测分数
    def forward(
        self,
        sequence_output: Tuple[torch.Tensor],
    ) -> Tuple[torch.Tensor]:
        # 调用 CanineLMPredictionHead 实例进行预测
        prediction_scores = self.predictions(sequence_output)
        # 返回预测分数
        return prediction_scores


class CaninePreTrainedModel(PreTrainedModel):
    """
    一个处理权重初始化和简单接口的抽象类，用于下载和加载预训练模型。
    """

    # 配置类为 CanineConfig
    config_class = CanineConfig
    # 加载 TensorFlow 权重的方法为 load_tf_weights_in_canine
    load_tf_weights = load_tf_weights_in_canine
    # 基础模型前缀为 "canine"
    base_model_prefix = "canine"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    def _init_weights(self, module):
        """初始化权重"""
        # 如果 module 是线性层或者一维卷积层
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # 使用正态分布随机初始化权重，均值为 0，标准差为模型配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置项，将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果 module 是嵌入层
        elif isinstance(module, nn.Embedding):
            # 使用正态分布随机初始化权重，均值为 0，标准差为模型配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在填充索引，将填充索引位置的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果 module 是 LayerNorm 层
        elif isinstance(module, nn.LayerNorm):
            # 将偏置项初始化为零
            module.bias.data.zero_()
            # 将权重初始化为全 1
            module.weight.data.fill_(1.0)
# 定义一个字符串常量，包含有关模型的文档字符串
CANINE_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`CanineConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义一个字符串常量，包含有关模型输入的文档字符串
CANINE_INPUTS_DOCSTRING = r"""
# 空字符串，用于后续添加有关模型输入的文档说明
"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列标记在词汇表中的索引。
            # 可以使用 [`AutoTokenizer`] 获得索引。参见 [`PreTrainedTokenizer.encode`] 和
            # [`PreTrainedTokenizer.__call__`] 获取详细信息。
            # [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 避免在填充标记索引上执行注意力的掩码。
            # 掩码值选在 `[0, 1]` 范围内：
            # - 1 表示**未被掩码**的标记，
            # - 0 表示**被掩码**的标记。
            # [什么是注意力掩码？](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 分段标记索引，用于指示输入的第一部分和第二部分。索引选在 `[0, 1]` 范围内：
            # - 0 对应于 *句子 A* 的标记，
            # - 1 对应于 *句子 B* 的标记。
            # [什么是标记类型 ID？](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 输入序列标记在位置嵌入中的位置索引。在范围 `[0, config.max_position_embeddings - 1]` 中选择。
            # [什么是位置 ID？](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于将自注意模块中选定的头部置零的掩码。掩码值选在 `[0, 1]` 范围内：
            # - 1 表示**未被掩码**的头部，
            # - 0 表示**被掩码**的头部。
        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选，你可以选择直接传递一个嵌入表示而不是传递 `input_ids`。如果你想对如何将 *input_ids* 索引转换为关联向量
            # 比模型内部嵌入查找矩阵更具控制权，则这很有用。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回的张量下的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多详细信息，请参见返回的张量下的 `hidden_states`。
        return_dict (`bool`, *optional*):
            # 是否返回一个 [`~utils.ModelOutput`] 而不是一个普通元组。
```py  
"""
@add_start_docstrings(
    "The bare CANINE Model transformer outputting raw hidden-states without any specific head on top.",
    CANINE_START_DOCSTRING,
)
# 定义 CANINE 模型类，继承自 CaninePreTrainedModel
class CanineModel(CaninePreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        shallow_config = copy.deepcopy(config)
        shallow_config.num_hidden_layers = 1

        # 初始化字符嵌入层
        self.char_embeddings = CanineEmbeddings(config)
        # 使用浅层/低维度的 transformer 编码器来获取初始字符编码
        self.initial_char_encoder = CanineEncoder(
            shallow_config,
            local=True,
            always_attend_to_first_position=False,
            first_position_attends_to_all=False,
            attend_from_chunk_width=config.local_transformer_stride,
            attend_from_chunk_stride=config.local_transformer_stride,
            attend_to_chunk_width=config.local_transformer_stride,
            attend_to_chunk_stride=config.local_transformer_stride,
        )
        self.chars_to_molecules = CharactersToMolecules(config)
        # 深层 transformer 编码器
        self.encoder = CanineEncoder(config)
        self.projection = ConvProjection(config)
        # 使用浅层/低维度的 transformer 编码器来获取最终字符编码
        self.final_char_encoder = CanineEncoder(shallow_config)

        # 如果需要添加池化层，则初始化 CaninePooler
        self.pooler = CaninePooler(config) if add_pooling_layer else None

        # 初始化权重并应用最终处理
        self.post_init()

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def _create_3d_attention_mask_from_input_mask(self, from_tensor, to_mask):
        """
        Create 3D attention mask from a 2D tensor mask.

        Args:
            from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
            to_mask: int32 Tensor of shape [batch_size, to_seq_length].

        Returns:
            float Tensor of shape [batch_size, from_seq_length, to_seq_length].
        """
        batch_size, from_seq_length = from_tensor.shape[0], from_tensor.shape[1]

        to_seq_length = to_mask.shape[1]

        to_mask = torch.reshape(to_mask, (batch_size, 1, to_seq_length)).float()

        # 我们不假设 `from_tensor` 是一个掩码（尽管它可能是）。我们实际上不关心我们是否关注 *from* 填充标记（只关注 *to* 填充标记），
        # 所以我们创建一个全为1的张量。
        broadcast_ones = torch.ones(size=(batch_size, from_seq_length, 1), dtype=torch.float32, device=to_mask.device)

        # 在两个维度上广播以创建掩码。
        mask = broadcast_ones * to_mask

        return mask
    def _downsample_attention_mask(self, char_attention_mask: torch.Tensor, downsampling_rate: int):
        """Downsample 2D character attention mask to 2D molecule attention mask using MaxPool1d layer."""

        # 首先，通过添加一个通道维度使 char_attention_mask 变为 3D
        batch_size, char_seq_len = char_attention_mask.shape
        poolable_char_mask = torch.reshape(char_attention_mask, (batch_size, 1, char_seq_len))

        # 接下来，应用 MaxPool1d 来获取形状为 (batch_size, 1, mol_seq_len) 的 pooled_molecule_mask
        pooled_molecule_mask = torch.nn.MaxPool1d(kernel_size=downsampling_rate, stride=downsampling_rate)(
            poolable_char_mask.float()
        )

        # 最后，压缩以获得形状为 (batch_size, mol_seq_len) 的张量
        molecule_attention_mask = torch.squeeze(pooled_molecule_mask, dim=-1)

        return molecule_attention_mask

    def _repeat_molecules(self, molecules: torch.Tensor, char_seq_length: torch.Tensor) -> torch.Tensor:
        """Repeats molecules to make them the same length as the char sequence."""

        rate = self.config.downsampling_rate

        molecules_without_extra_cls = molecules[:, 1:, :]
        # `repeated`: [batch_size, almost_char_seq_len, molecule_hidden_size]
        repeated = torch.repeat_interleave(molecules_without_extra_cls, repeats=rate, dim=-2)

        # 到目前为止，我们已经重复了足够的元素，以适应任何 `char_seq_length`
        # 是 `downsampling_rate` 的倍数。现在我们考虑最后的 n 个元素（n < `downsampling_rate`），
        # 即 floor division 的余数。我们通过多次重复最后一个分子来实现这一点。
        last_molecule = molecules[:, -1:, :]
        remainder_length = torch.fmod(torch.tensor(char_seq_length), torch.tensor(rate)).item()
        remainder_repeated = torch.repeat_interleave(
            last_molecule,
            # +1 个分子来补偿截断。
            repeats=remainder_length + rate,
            dim=-2,
        )

        # `repeated`: [batch_size, char_seq_len, molecule_hidden_size]
        return torch.cat([repeated, remainder_repeated], dim=-2)

    @add_start_docstrings_to_model_forward(CANINE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CanineModelOutputWithPooling,
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
# 使用 CANINE 模型进行序列分类/回归任务，顶部有一个线性层用于汇总输出，例如用于 GLUE 任务
@add_start_docstrings(
    """
    CANINE Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    CANINE_START_DOCSTRING,
)
class CanineForSequenceClassification(CaninePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # 初始化 CANINE 模型
        self.canine = CanineModel(config)
        # 添加 dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 添加分类器线性层
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数，接受多个输入参数
    @add_start_docstrings_to_model_forward(CANINE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        # 确保返回字典不为空，若为空则使用模型配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给Canine模型进行处理
        outputs = self.canine(
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

        # 获取汇总输出（pooled_output）
        pooled_output = outputs[1]

        # 对汇总输出进行dropout处理
        pooled_output = self.dropout(pooled_output)
        # 将dropout后的输出传递给分类器，得到logits
        logits = self.classifier(pooled_output)

        loss = None
        # 如果提供了标签，则计算损失
        if labels is not None:
            # 如果问题类型未指定，则根据标签和模型配置确定问题类型
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
                    # 若只有一个标签，则计算回归损失
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    # 否则，计算回归损失
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # 若是单标签分类问题，则使用交叉熵损失函数
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # 若是多标签分类问题，则使用带Logits的二元交叉熵损失函数
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        
        # 如果不需要返回字典，则返回元组形式的输出
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回包含损失、logits、隐藏状态和注意力权重的SequenceClassifierOutput对象
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 为多选任务定制的 CANINE 模型，模型顶部有一个多选分类头（在池化输出的顶部有一个线性层和一个 softmax），例如用于 RocStories/SWAG 任务
@add_start_docstrings(
    """
    CANINE 模型，顶部有一个多选分类头（在池化输出的顶部有一个线性层和一个 softmax），例如用于 RocStories/SWAG 任务。
    """,
    CANINE_START_DOCSTRING,
)
class CanineForMultipleChoice(CaninePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化 CANINE 模型
        self.canine = CanineModel(config)
        # 定义一个 dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 定义一个线性层，用于多选分类
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(CANINE_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
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
    ) -> Union[Tuple, MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 确定是否返回字典格式的结果
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 获取选择数量
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 重塑输入数据以便与模型输入匹配
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 将输入传递给 CANINE 模型以获取输出
        outputs = self.canine(
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

        # 从模型输出中提取汇总输出
        pooled_output = outputs[1]

        # 应用 dropout
        pooled_output = self.dropout(pooled_output)
        # 使用分类器获取 logits
        logits = self.classifier(pooled_output)
        # 重塑 logits 以匹配预期形状
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        # 如果提供了标签，则计算损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 如果不返回字典格式的结果，则返回不同的输出元组
        if not return_dict:
            output = (reshaped_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 返回多项选择模型输出
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用一个 token 分类头（在隐藏状态输出的顶部有一个线性层），例如用于命名实体识别（NER）任务的 CANINE 模型
@add_start_docstrings(
    """
    CANINE 模型，在顶部有一个令牌分类头（隐藏状态输出的顶部有一个线性层），例如用于命名实体识别（NER）任务。
    """,
    CANINE_START_DOCSTRING,
)
class CanineForTokenClassification(CaninePreTrainedModel):
    def __init__(self, config):
        # 调用父类构造函数初始化模型配置
        super().__init__(config)
        # 设置标签数量
        self.num_labels = config.num_labels

        # 实例化 CANINE 模型
        self.canine = CanineModel(config)
        # 添加一个丢弃层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 添加一个线性分类器
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数
    @add_start_docstrings_to_model_forward(CANINE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
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

        Returns:
            返回值说明：
            - 若 return_dict 不为 None，则根据 return_dict 决定是否返回字典格式的输出
            - 若 return_dict 为 None，则根据模型配置中的 use_return_dict 参数决定是否返回字典格式的输出

        Example:
            示例：
            ```python
            >>> from transformers import AutoTokenizer, CanineForTokenClassification
            >>> import torch

            >>> tokenizer = AutoTokenizer.from_pretrained("google/canine-s")
            >>> model = CanineForTokenClassification.from_pretrained("google/canine-s")

            >>> inputs = tokenizer(
            ...     "HuggingFace is a company based in Paris and New York", add_special_tokens=False, return_tensors="pt"
            ... )

            >>> with torch.no_grad():
            ...     logits = model(**inputs).logits

            >>> predicted_token_class_ids = logits.argmax(-1)

            >>> # Note that tokens are classified rather then input words which means that
            >>> # there might be more predicted token classes than words.
            >>> # Multiple token classes might account for the same word
            >>> predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]
            >>> predicted_tokens_classes  # doctest: +SKIP
            ```py

            ```python
            >>> labels = predicted_token_class_ids
            >>> loss = model(**inputs, labels=labels).loss
            >>> round(loss.item(), 2)  # doctest: +SKIP
            ```py"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.canine(
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

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 为提取式问答任务设计的 CANINE 模型，包含一个用于分类的 span classification 头部（在隐藏状态输出之上的线性层，用于计算 `span start logits` 和 `span end logits`）。
@add_start_docstrings(
    """
    CANINE Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    CANINE_START_DOCSTRING,
)
class CanineForQuestionAnswering(CaninePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # 初始化 CANINE 模型和 QA 输出层
        self.canine = CanineModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数
    @add_start_docstrings_to_model_forward(CANINE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint="Splend1dchan/canine-c-squad",
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output="'nice puppet'",
        expected_loss=8.81,
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
        # 确定是否返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 Canine 模型进行推理
        outputs = self.canine(
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

        # 获取模型输出的序列表示
        sequence_output = outputs[0]

        # 对序列表示进行问答任务的分类
        logits = self.qa_outputs(sequence_output)
        # 拆分开始位置和结束位置的 logits
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        # 如果有提供起始位置和结束位置
        if start_positions is not None and end_positions is not None:
            # 如果处于多 GPU 模式，添加一个维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 忽略超出模型输入的起始/结束位置
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            # 定义交叉熵损失函数
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            # 计算起始位置和结束位置的损失
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        # 如果不返回字典，则返回模型输出
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 返回 QuestionAnsweringModelOutput 对象
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```