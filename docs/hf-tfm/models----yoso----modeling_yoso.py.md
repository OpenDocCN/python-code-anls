# `.\models\yoso\modeling_yoso.py`

```py
# coding=utf-8
# Copyright 2022 University of Wisconsin-Madison and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch YOSO model."""

import math
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithCrossAttentions,
    MaskedLMOutput,
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
    is_ninja_available,
    is_torch_cuda_available,
    logging,
)
from .configuration_yoso import YosoConfig

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 用于文档的检查点和配置信息
_CHECKPOINT_FOR_DOC = "uw-madison/yoso-4096"
_CONFIG_FOR_DOC = "YosoConfig"

# 预训练模型的存档列表
YOSO_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "uw-madison/yoso-4096",
    # 查看所有 YOSO 模型：https://huggingface.co/models?filter=yoso
]

# 全局变量，用于保存加载的 CUDA 内核
lsh_cumulation = None


def load_cuda_kernels():
    """
    加载 CUDA 内核函数。
    """
    global lsh_cumulation
    from torch.utils.cpp_extension import load

    def append_root(files):
        # 获取内核文件夹的路径
        src_folder = Path(__file__).resolve().parent.parent.parent / "kernels" / "yoso"
        return [src_folder / file for file in files]

    # 待加载的 CUDA 内核文件列表
    src_files = append_root(["fast_lsh_cumulation_torch.cpp", "fast_lsh_cumulation.cu", "fast_lsh_cumulation_cuda.cu"])

    # 使用 Torch 提供的 cpp_extension 加载内核文件并命名为 "fast_lsh_cumulation"
    load("fast_lsh_cumulation", src_files, verbose=True)

    # 导入加载后的 CUDA 内核函数并赋值给全局变量 lsh_cumulation
    import fast_lsh_cumulation as lsh_cumulation


def to_contiguous(input_tensors):
    """
    确保输入张量是连续的（contiguous）。
    """
    if isinstance(input_tensors, list):
        # 如果是列表，对每个张量进行连续性检查和处理
        out = []
        for tensor in input_tensors:
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            out.append(tensor)
        return out
    else:
        # 如果是单个张量，确保其连续性
        if not input_tensors.is_contiguous():
            input_tensors = input_tensors.contiguous()
        return input_tensors


def normalize(input_tensors):
    """
    对输入张量进行 L2 归一化。
    """
    if isinstance(input_tensors, list):
        # 如果是列表，对每个张量进行 L2 归一化处理
        out = []
        for tensor in input_tensors:
            out.append(nn.functional.normalize(tensor, p=2, dim=-1))
        return out
    else:
        # 如果条件不满足，则执行以下操作：
        # 对输入张量进行 L2 范数归一化，沿着最后一个维度（dim=-1）进行操作
        return nn.functional.normalize(input_tensors, p=2, dim=-1)
    # 定义一个哈希函数，用于生成查询和键的哈希码
    def hashing(query, key, num_hash, hash_len):
        # 检查查询张量的维度是否为3
        if len(query.size()) != 3:
            raise ValueError("Query has incorrect size.")
        # 检查键张量的维度是否为3
        if len(key.size()) != 3:
            raise ValueError("Key has incorrect size.")

        # 生成随机投影矩阵，形状为 (查询张量的第一维度大小, 查询张量的第三维度大小, num_hash * hash_len)
        rmat = torch.randn(query.size(0), query.size(2), num_hash * hash_len, device=query.device)
        # 创建用于计算哈希码的指数表达式
        raise_pow = 2 ** torch.arange(hash_len, device=query.device)

        # 计算查询的投影，并重新形状为 (查询张量的第一维度大小, 查询张量的第二维度大小, num_hash, hash_len)
        query_projection = torch.matmul(query, rmat).reshape(query.size(0), query.size(1), num_hash, hash_len)
        # 计算键的投影，并重新形状为 (键张量的第一维度大小, 键张量的第二维度大小, num_hash, hash_len)
        key_projection = torch.matmul(key, rmat).reshape(key.size(0), key.size(1), num_hash, hash_len)
        # 将查询投影结果转换为二进制表示
        query_binary = (query_projection > 0).int()
        # 将键投影结果转换为二进制表示
        key_binary = (key_projection > 0).int()
        # 计算查询的哈希码
        query_hash = torch.sum(query_binary * raise_pow, dim=-1)
        # 计算键的哈希码
        key_hash = torch.sum(key_binary * raise_pow, dim=-1)

        return query_hash.int(), key_hash.int()


    class YosoCumulation(torch.autograd.Function):
        @staticmethod
        # 前向传播函数，计算期望值并进行累积计算
        def forward(ctx, query_mask, key_mask, query, key, value, config):
            # 从配置中获取哈希码长度
            hash_code_len = config["hash_code_len"]

            # 计算期望值，其值取决于查询与键的余弦相似度
            expectation = (1 - torch.acos(torch.matmul(query, key.transpose(-1, -2))) / math.pi) ** hash_code_len
            # 将期望值乘以查询和键的掩码
            expectation = expectation * query_mask[:, :, None] * key_mask[:, None, :]
            # 计算累积值，通过期望值与值张量的乘积
            cumulation_value = torch.matmul(expectation, value)

            # 保存计算所需的张量，以便反向传播使用
            ctx.save_for_backward(query_mask, key_mask, expectation, query, key, value)
            ctx.config = config

            return cumulation_value

        @staticmethod
        # 反向传播函数，计算梯度
        def backward(ctx, grad):
            grad = to_contiguous(grad)

            # 恢复保存的张量
            query_mask, key_mask, expectation, query, key, value = ctx.saved_tensors
            config = ctx.config

            # 从配置中获取哈希码长度
            hash_code_len = config["hash_code_len"]

            # 计算加权期望值
            weighted_exp = torch.matmul(grad, value.transpose(-1, -2)) * expectation
            # 计算查询的梯度
            grad_query = torch.matmul(weighted_exp, (hash_code_len / 2) * key)
            # 计算键的梯度
            grad_key = torch.matmul(weighted_exp.transpose(-1, -2), (hash_code_len / 2) * query)
            # 计算值的梯度
            grad_value = torch.matmul(expectation.transpose(-1, -2), grad)

            # 返回梯度，前两个为None，对应于前向传播的query_mask和key_mask
            return None, None, grad_query, grad_key, grad_value, None


    class YosoLSHCumulation(torch.autograd.Function):
        @staticmethod
    # 前向传播函数，用于计算查询（query）与键（key）之间的关联值（cumulation_value）
    def forward(ctx, query_mask, key_mask, query, key, value, config):
        # 检查输入张量的维度是否匹配
        if query_mask.size(0) != key_mask.size(0):
            raise ValueError("Query mask and Key mask differ in sizes in dimension 0")
        if query_mask.size(0) != query.size(0):
            raise ValueError("Query mask and Query differ in sizes in dimension 0")
        if query_mask.size(0) != key.size(0):
            raise ValueError("Query mask and Key differ in sizes in dimension 0")
        if query_mask.size(0) != value.size(0):
            raise ValueError("Query mask and Value mask differ in sizes in dimension 0")
        if key.size(1) != value.size(1):
            raise ValueError("Key and Value differ in sizes in dimension 1")
        if query.size(2) != key.size(2):
            raise ValueError("Query and Key differ in sizes in dimension 2")

        # 将输入张量转换为连续内存的形式
        query_mask, key_mask, query, key, value = to_contiguous([query_mask, key_mask, query, key, value])

        # 检测是否使用 CUDA 加速
        use_cuda = query_mask.is_cuda
        # 从配置中获取哈希函数的数量和哈希码长度
        num_hash = config["num_hash"]
        hash_code_len = config["hash_code_len"]
        # 计算哈希表的容量
        hashtable_capacity = int(2**hash_code_len)

        # 根据配置决定使用快速哈希还是普通哈希
        if config["use_fast_hash"]:
            # 如果使用快速哈希，调用快速哈希函数计算哈希码
            query_hash_code, key_hash_code = lsh_cumulation.fast_hash(
                query_mask, query, key_mask, key, num_hash, hash_code_len, use_cuda, 1
            )
        else:
            # 否则，调用普通哈希函数计算哈希码
            query_hash_code, key_hash_code = hashing(query, key, num_hash, hash_code_len)

        # 调用累积哈希表的函数，计算最终的累积值
        cumulation_value = lsh_cumulation.lsh_cumulation(
            query_mask, query_hash_code, key_mask, key_hash_code, value, hashtable_capacity, use_cuda, 1
        )

        # 将所有必要的张量和配置保存在上下文对象中，以便反向传播使用
        ctx.save_for_backward(query_mask, key_mask, query_hash_code, key_hash_code, query, key, value)
        ctx.config = config

        # 返回前向传播计算得到的累积值
        return cumulation_value

    @staticmethod
    # 定义一个反向传播函数，计算输入梯度关于查询、键和值的梯度
    def backward(ctx, grad):
        # 将输入梯度转换为连续内存存储
        grad = to_contiguous(grad)

        # 从上下文中获取保存的张量和配置信息
        query_mask, key_mask, query_hash_code, key_hash_code, query, key, value = ctx.saved_tensors
        config = ctx.config

        # 检查是否在CUDA上计算
        use_cuda = grad.is_cuda
        # 从配置中获取哈希码长度
        hash_code_len = config["hash_code_len"]
        # 计算哈希表容量
        hashtable_capacity = int(2**hash_code_len)

        # 如果配置中指定使用LSH反向传播
        if config["lsh_backward"]:
            # 计算键-值映射的累积梯度
            grad_value = lsh_cumulation.lsh_cumulation(
                key_mask, key_hash_code, query_mask, query_hash_code, grad, hashtable_capacity, use_cuda, 1
            )
            # 计算查询的加权累积梯度
            grad_query = lsh_cumulation.lsh_weighted_cumulation(
                query_mask,
                query_hash_code,
                grad,
                key_mask,
                key_hash_code,
                value,
                (hash_code_len / 2) * key,
                hashtable_capacity,
                use_cuda,
                4,
            )
            # 计算键的加权累积梯度
            grad_key = lsh_cumulation.lsh_weighted_cumulation(
                key_mask,
                key_hash_code,
                value,
                query_mask,
                query_hash_code,
                grad,
                (hash_code_len / 2) * query,
                hashtable_capacity,
                use_cuda,
                4,
            )
        else:
            # 计算期望值，用于非LSH方式的加权梯度计算
            expectation = (1 - torch.acos(torch.matmul(query, key.transpose(-1, -2))) / math.pi) ** hash_code_len
            expectation = expectation * query_mask[:, :, None] * key_mask[:, None, :]
            weighted_exp = torch.matmul(grad, value.transpose(-1, -2)) * expectation
            grad_query = torch.matmul(weighted_exp, (hash_code_len / 2) * key)
            grad_key = torch.matmul(weighted_exp.transpose(-1, -2), (hash_code_len / 2) * query)
            grad_value = torch.matmul(expectation.transpose(-1, -2), grad)

        # 返回梯度，此处返回None表示没有对部分变量进行梯度计算
        return None, None, grad_query, grad_key, grad_value, None
# Copied from transformers.models.nystromformer.modeling_nystromformer.NystromformerEmbeddings
# YosoEmbeddings 类用于构建从单词、位置和令牌类型嵌入得到的总体嵌入。

class YosoEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        
        # 创建单词嵌入层，使用 nn.Embedding 类，配置为词汇表大小、隐藏大小，并指定填充标记ID
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        
        # 创建位置嵌入层，使用 nn.Embedding 类，配置为最大位置嵌入数加2和隐藏大小
        self.position_embeddings = nn.Embedding(config.max_position_embeddings + 2, config.hidden_size)
        
        # 创建令牌类型嵌入层，使用 nn.Embedding 类，配置为类型词汇表大小和隐藏大小
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 用于保持 TensorFlow 模型变量名一致，便于加载 TensorFlow 检查点文件，因此未使用 snake_case 命名
        # 创建 LayerNorm 层，使用 nn.LayerNorm 类，配置为隐藏大小和层标准化的 epsilon 值
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # 创建 Dropout 层，使用 nn.Dropout 类，配置为隐藏单元的丢弃概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 创建 position_ids 缓冲区张量，包含从2开始的连续最大位置嵌入数，不进行持久化
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)) + 2, persistent=False
        )
        
        # 设置位置嵌入类型，默认为 "absolute"，通过 config.position_embedding_type 进行配置
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        
        # 创建 token_type_ids 缓冲区张量，与 position_ids 大小相同，值为0，数据类型为长整型，设备与 position_ids 相同
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
            persistent=False,
        )
    # 定义模型的前向传播函数，接受输入的参数：input_ids、token_type_ids、position_ids、inputs_embeds
    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        # 如果输入中包含 input_ids，则获取其形状信息
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            # 否则获取 inputs_embeds 的形状信息，去除最后一维
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列的长度
        seq_length = input_shape[1]

        # 如果 position_ids 为 None，则使用模型构造函数中已注册的 position_ids 的部分切片
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # 如果 token_type_ids 为 None，则根据模型是否有 token_type_ids 属性进行处理
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                # 获取已注册的 token_type_ids 的部分切片，并扩展到与输入形状相匹配
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                # 否则创建一个全零的 token_type_ids 张量，与输入形状相同，位于模型的设备上
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果 inputs_embeds 为 None，则通过 word_embeddings 层获取输入的词嵌入
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # 根据 token_type_ids 获取 token_type_embeddings
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将输入嵌入和 token_type_embeddings 相加作为最终的嵌入表示
        embeddings = inputs_embeds + token_type_embeddings

        # 如果位置嵌入的类型是 "absolute"，则添加位置嵌入到最终的嵌入表示中
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        # 对嵌入表示进行 LayerNorm 处理
        embeddings = self.LayerNorm(embeddings)

        # 对嵌入表示进行 dropout 处理
        embeddings = self.dropout(embeddings)

        # 返回最终的嵌入表示作为模型的输出
        return embeddings
# 定义一个名为 YosoSelfAttention 的自定义神经网络模块，继承自 nn.Module
class YosoSelfAttention(nn.Module):
    # 初始化函数，接收一个 config 对象和一个可选的 position_embedding_type 参数
    def __init__(self, config, position_embedding_type=None):
        super().__init__()  # 调用父类 nn.Module 的初始化函数
        # 检查 hidden_size 是否能被 num_attention_heads 整除，若不能且 config 没有 embedding_size 属性则引发 ValueError
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        # 检查是否可用 CUDA 和 Ninja，并且没有加载 lsh_cumulation 内核，则尝试加载 CUDA 内核
        kernel_loaded = lsh_cumulation is not None
        if is_torch_cuda_available() and is_ninja_available() and not kernel_loaded:
            try:
                load_cuda_kernels()
            except Exception as e:
                # 若加载失败则记录警告信息
                logger.warning(f"Could not load the custom kernel for multi-scale deformable attention: {e}")

        # 设置模块的一些属性值，从 config 中获取
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 分别定义 query、key、value 的线性变换层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 定义 dropout 层，用于注意力概率的 dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # 设置位置编码的类型，默认为 config 中的值，若提供了 position_embedding_type 则使用该值
        self.position_embedding_type = (
            position_embedding_type if position_embedding_type is not None else config.position_embedding_type
        )

        # 设置是否使用期望值
        self.use_expectation = config.use_expectation
        # 设置哈希码的长度
        self.hash_code_len = config.hash_code_len
        # 是否使用卷积
        self.use_conv = config.conv_window is not None
        # 是否使用快速哈希
        self.use_fast_hash = config.use_fast_hash
        # 哈希函数的数量
        self.num_hash = config.num_hash
        # LSH 是否反向
        self.lsh_backward = config.lsh_backward

        # 构建 LSH 配置字典
        self.lsh_config = {
            "hash_code_len": self.hash_code_len,
            "use_fast_hash": self.use_fast_hash,
            "num_hash": self.num_hash,
            "lsh_backward": self.lsh_backward,
        }

        # 如果配置中定义了卷积窗口大小，则定义一个卷积层
        if config.conv_window is not None:
            self.conv = nn.Conv2d(
                in_channels=config.num_attention_heads,
                out_channels=config.num_attention_heads,
                kernel_size=(config.conv_window, 1),
                padding=(config.conv_window // 2, 0),
                bias=False,
                groups=config.num_attention_heads,
            )

    # 定义一个辅助方法，用于将输入层变换为分数矩阵
    def transpose_for_scores(self, layer):
        # 计算新的矩阵形状
        new_layer_shape = layer.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # 改变层的形状
        layer = layer.view(*new_layer_shape)
        # 对维度进行置换，以便正确计算注意力分数
        return layer.permute(0, 2, 1, 3)


# 从 transformers.models.bert.modeling_bert.BertSelfOutput 复制的自定义神经网络模块 YosoSelfOutput
class YosoSelfOutput(nn.Module):
    # 初始化函数，接收一个 config 对象
    def __init__(self, config):
        super().__init__()  # 调用父类 nn.Module 的初始化函数
        # 定义一个线性变换层 dense
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义一个 LayerNorm 层，用于归一化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 定义一个 dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    # 定义一个前向传播函数，接收隐藏状态（torch.Tensor 类型）和输入张量（torch.Tensor 类型），返回一个 torch.Tensor 类型的张量
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态通过全连接层 dense 进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的隐藏状态进行 dropout 操作，以防止过拟合
        hidden_states = self.dropout(hidden_states)
        # 将 dropout 后的隐藏状态与输入张量相加，并通过 LayerNorm 进行归一化处理
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的隐藏状态
        return hidden_states
# 定义 YosoAttention 类，继承自 nn.Module
class YosoAttention(nn.Module):
    # 初始化方法，接受 config 和 position_embedding_type 两个参数
    def __init__(self, config, position_embedding_type=None):
        # 调用父类的初始化方法
        super().__init__()
        # 创建 YosoSelfAttention 实例，并存储在 self.self 中
        self.self = YosoSelfAttention(config, position_embedding_type=position_embedding_type)
        # 创建 YosoSelfOutput 实例，并存储在 self.output 中
        self.output = YosoSelfOutput(config)
        # 创建一个空集合，用于存储被剪枝的注意力头的索引
        self.pruned_heads = set()

    # 剪枝方法，接受 heads 参数作为要剪枝的头部列表
    def prune_heads(self, heads):
        # 如果 heads 列表为空，则直接返回
        if len(heads) == 0:
            return
        # 调用 find_pruneable_heads_and_indices 函数，获取可剪枝头部及其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层：query、key、value 和 output.dense
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储被剪枝的头部
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 前向传播方法，接受 hidden_states、attention_mask 和 output_attentions 三个参数
    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        # 调用 self.self 的前向传播方法，获取 self_outputs
        self_outputs = self.self(hidden_states, attention_mask, output_attentions)
        # 调用 self.output 的前向传播方法，计算 attention_output
        attention_output = self.output(self_outputs[0], hidden_states)
        # 将 attention_output 与 self_outputs 中的其它元素组成 outputs
        outputs = (attention_output,) + self_outputs[1:]  # 如果需要输出 attentions，则添加它们
        return outputs


# 从 transformers.models.bert.modeling_bert.BertIntermediate 复制而来
# 定义 YosoIntermediate 类，继承自 nn.Module
class YosoIntermediate(nn.Module):
    # 初始化方法，接受 config 参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个线性层 dense
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 根据 config 中的 hidden_act 创建激活函数 intermediate_act_fn
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播方法，接受 hidden_states 参数
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 输入 hidden_states 到线性层 dense
        hidden_states = self.dense(hidden_states)
        # 经过激活函数 intermediate_act_fn
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# 从 transformers.models.bert.modeling_bert.BertOutput 复制而来
# 定义 YosoOutput 类，继承自 nn.Module
class YosoOutput(nn.Module):
    # 初始化方法，接受 config 参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个线性层 dense
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建 LayerNorm 层
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建 Dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播方法，接受 hidden_states 和 input_tensor 两个参数
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 输入 hidden_states 到线性层 dense
        hidden_states = self.dense(hidden_states)
        # 经过 Dropout 层
        hidden_states = self.dropout(hidden_states)
        # 输入 hidden_states 和 input_tensor 到 LayerNorm 层
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# 定义 YosoLayer 类，继承自 nn.Module
class YosoLayer(nn.Module):
    pass  # 此处 pass 表示 YosoLayer 类暂无额外代码，仅用作占位符
    # 初始化函数，用于初始化一个 YosoLayer 实例
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 设置前向传播中的块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度的维度，通常为1
        self.seq_len_dim = 1
        # 创建自注意力层 YosoAttention 的实例
        self.attention = YosoAttention(config)
        # 是否添加跨注意力的标志
        self.add_cross_attention = config.add_cross_attention
        # 创建中间层 YosoIntermediate 的实例
        self.intermediate = YosoIntermediate(config)
        # 创建输出层 YosoOutput 的实例
        self.output = YosoOutput(config)

    # 前向传播函数，处理隐藏状态、注意力掩码并返回输出
    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        # 使用自注意力层处理隐藏状态
        self_attention_outputs = self.attention(hidden_states, attention_mask, output_attentions=output_attentions)
        # 获取自注意力层的输出
        attention_output = self_attention_outputs[0]

        # 如果需要输出注意力权重，将自注意力层的输出添加到输出中
        outputs = self_attention_outputs[1:]  # 如果要输出注意力权重，则添加自注意力

        # 将注意力输出分块处理并进行前向传播
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        # 将块处理后的输出添加到总体输出中
        outputs = (layer_output,) + outputs

        # 返回最终的输出
        return outputs

    # 前向传播中的块处理函数，处理注意力输出并返回层输出
    def feed_forward_chunk(self, attention_output):
        # 使用中间层处理注意力输出
        intermediate_output = self.intermediate(attention_output)
        # 使用输出层处理中间层输出和注意力输出，得到层输出
        layer_output = self.output(intermediate_output, attention_output)
        # 返回处理后的层输出
        return layer_output
# YosoEncoder 类，用于实现自定义的编码器模型
class YosoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 创建包含多个 YosoLayer 层的列表，层数由配置文件指定
        self.layer = nn.ModuleList([YosoLayer(config) for _ in range(config.num_hidden_layers)])
        # 是否启用梯度检查点，默认为 False
        self.gradient_checkpointing = False

    # 前向传播方法，接受多个参数并返回多个输出
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # 如果需要输出隐藏状态，则初始化一个空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力矩阵，则初始化一个空元组
        all_self_attentions = () if output_attentions else None

        # 遍历每个 YosoLayer 层进行前向传播
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果启用梯度检查点且处于训练模式，则使用梯度检查点函数进行前向传播
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    output_attentions,
                )
            else:
                # 否则直接调用 YosoLayer 层的前向传播方法
                layer_outputs = layer_module(hidden_states, attention_mask, output_attentions)

            # 更新隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]
            # 如果需要输出注意力矩阵，则将当前层的注意力矩阵添加到 all_self_attentions
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果需要输出隐藏状态，则将最终的隐藏状态添加到 all_hidden_states
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要返回字典形式的输出，则返回包含非空值的元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 否则返回一个包含各输出部分的 BaseModelOutputWithCrossAttentions 对象
        return BaseModelOutputWithCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


# YosoPredictionHeadTransform 类，用于实现预测头部的转换
# 从 transformers.models.bert.modeling_bert.BertPredictionHeadTransform 复制而来
class YosoPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用线性层进行维度转换，输入和输出维度为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 根据配置中的激活函数字符串或函数对象选择激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # 使用 LayerNorm 进行归一化，输入维度为 config.hidden_size
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # 前向传播方法，接受隐藏状态作为输入并返回转换后的隐藏状态
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 线性变换
        hidden_states = self.dense(hidden_states)
        # 应用激活函数
        hidden_states = self.transform_act_fn(hidden_states)
        # LayerNorm 归一化
        hidden_states = self.LayerNorm(hidden_states)
        # 返回转换后的隐藏状态
        return hidden_states


# YosoLMPredictionHead 类，用于实现语言模型预测头部
# 从 transformers.models.bert.modeling_bert.BertLMPredictionHead 复制而来，将 Bert 替换为 Yoso
class YosoLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用给定的配置初始化 YosoPredictionHeadTransform 实例，并将其赋值给 transform 属性
        self.transform = YosoPredictionHeadTransform(config)

        # 创建一个线性层，将隐藏状态的大小映射到词汇表大小，没有偏置项
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 创建一个可学习的偏置项，大小为词汇表大小，并作为 nn.Parameter 对象存储在 bias 属性中
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # 将 decoder 的偏置项设置为当前定义的偏置项，以便在调整 token embeddings 大小时能正确调整其大小
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        # 对输入的隐藏状态进行变换，使用 transform 属性的方法
        hidden_states = self.transform(hidden_states)
        
        # 使用 decoder 属性将变换后的隐藏状态映射到词汇表大小的输出空间
        hidden_states = self.decoder(hidden_states)
        
        # 返回映射后的输出隐藏状态
        return hidden_states
# 从transformers.models.bert.modeling_bert.BertOnlyMLMHead复制而来，将Bert更改为Yoso
class YosoOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化预测头部
        self.predictions = YosoLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        # 前向传播，生成预测分数
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class YosoPreTrainedModel(PreTrainedModel):
    """
    一个抽象类，用于处理权重初始化、下载和加载预训练模型的简单接口。
    """

    config_class = YosoConfig  # 配置类为YosoConfig
    base_model_prefix = "yoso"  # 模型前缀为'yoso'
    supports_gradient_checkpointing = True  # 支持梯度检查点

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            # 与TF版本稍有不同，TF版本使用截断正态分布进行初始化
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


YOSO_START_DOCSTRING = r"""
    这个模型是一个PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)的子类。
    可以像普通的PyTorch模块一样使用，并且可以参考PyTorch文档了解一切与一般使用和行为相关的问题。

    参数:
        config ([`YosoConfig`]): 包含模型所有参数的模型配置类。
            使用配置文件初始化不会加载模型关联的权重，只会加载配置。
            可以查看[`~PreTrainedModel.from_pretrained`]方法来加载模型权重。
"""

YOSO_INPUTS_DOCSTRING = r"""
    输入:
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列中每个token的索引，用于在词汇表中查找对应的token表示
            Indices of input sequence tokens in the vocabulary.

            # 可以使用AutoTokenizer获取这些索引。参见PreTrainedTokenizer.encode和PreTrainedTokenizer.__call__获取更多细节。
            [What are input IDs?](../glossary#input-ids)

        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 注意力掩码，用于避免对填充的token进行注意力计算。掩码值选择在[0, 1]之间：
            # - 1表示该token是**未被掩码**的，
            # - 0表示该token是**被掩码**的。
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            [What are attention masks?](../glossary#attention-mask)

        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 段落token索引，用于指示输入的第一部分和第二部分。索引选择在[0, 1]之间：
            # - 0对应*句子A*的token，
            # - 1对应*句子B*的token。
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:

            [What are token type IDs?](../glossary#token-type-ids)

        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 每个输入序列token在位置嵌入中的位置索引。选取范围为[0, config.max_position_embeddings - 1]。
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于将自注意力模块的特定头部置为零的掩码。掩码值选择在[0, 1]之间：
            # - 1表示该头部**未被掩码**，
            # - 0表示该头部**被掩码**。
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            # 可选项，可以直接传递嵌入表示而不是传递input_ids。如果需要更多控制如何将input_ids索引转换为关联向量，则这是有用的，比模型内部的嵌入查找矩阵更灵活。
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert *input_ids* indices into associated vectors than the model's internal embedding lookup matrix.

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。详细信息请参见返回的张量中的`attentions`。
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。详细信息请参见返回的张量中的`hidden_states`。
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.

        return_dict (`bool`, *optional*):
            # 是否返回`~utils.ModelOutput`而不是普通的元组。
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""
YOSO Model transformer outputting raw hidden-states without any specific head on top.

YOSO_START_DOCSTRING: 表示模型开始文档字符串的示例。

"""

class YosoModel(YosoPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = YosoEmbeddings(config)  # 初始化 YosoEmbeddings 对象，用于处理输入的嵌入层
        self.encoder = YosoEncoder(config)  # 初始化 YosoEncoder 对象，用于编码输入数据

        # Initialize weights and apply final processing
        self.post_init()  # 调用模型初始化后处理函数

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings  # 返回输入嵌入层的单词嵌入

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value  # 设置输入嵌入层的单词嵌入为给定的 value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)  # 剪枝模型中的注意力头部

@add_start_docstrings("""YOSO Model with a `language modeling` head on top.""", YOSO_START_DOCSTRING)
class YosoForMaskedLM(YosoPreTrainedModel):
    _tied_weights_keys = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        self.yoso = YosoModel(config)  # 初始化 YosoModel 作为 YosoForMaskedLM 的基础模型
        self.cls = YosoOnlyMLMHead(config)  # 初始化 YosoOnlyMLMHead 作为 YosoForMaskedLM 的 MLM 头部

        # Initialize weights and apply final processing
        self.post_init()  # 调用模型初始化后处理函数

    def get_output_embeddings(self):
        return self.cls.predictions.decoder  # 返回输出嵌入层的解码器

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings  # 设置输出嵌入层的解码器为给定的新嵌入

    @add_start_docstrings_to_model_forward(YOSO_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入的token IDs，可以为空
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，用于指示哪些token需要注意，可以为空
        token_type_ids: Optional[torch.Tensor] = None,  # token类型IDs，如用于区分segment A和segment B，可以为空
        position_ids: Optional[torch.Tensor] = None,  # 位置IDs，用于指示token的位置信息，可以为空
        head_mask: Optional[torch.Tensor] = None,  # 头部掩码，用于指定哪些注意力头部被屏蔽，可以为空
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入的嵌入表示，可以为空
        labels: Optional[torch.Tensor] = None,  # 用于计算MLM损失的标签，形状为(batch_size, sequence_length)，可选
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选
        return_dict: Optional[bool] = None,  # 是否返回字典形式的输出，若为None，则使用self.config.use_return_dict的设置
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # 确定是否返回字典形式的输出，根据传入的return_dict或self.config.use_return_dict决定

        outputs = self.yoso(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )  # 调用yoso模型，传入各种参数，并根据return_dict确定返回形式

        sequence_output = outputs[0]  # 获取模型输出的序列输出
        prediction_scores = self.cls(sequence_output)  # 使用线性层对序列输出进行预测得分计算

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # 定义交叉熵损失函数，用于计算MLM损失
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))  # 计算MLM损失

        if not return_dict:
            output = (prediction_scores,) + outputs[1:]  # 如果不返回字典形式的输出，则构造输出元组
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output  # 返回带有损失的输出元组或普通的输出元组

        return MaskedLMOutput(
            loss=masked_lm_loss,  # 返回带有损失信息的MaskedLMOutput对象
            logits=prediction_scores,  # 返回预测得分
            hidden_states=outputs.hidden_states,  # 返回隐藏状态
            attentions=outputs.attentions,  # 返回注意力权重
        )
class YosoClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        # 线性层，输入和输出维度均为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # Dropout 层，用于随机屏蔽输入单元，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 输出层线性变换，将隐藏状态映射到标签数量维度
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        self.config = config

    def forward(self, features, **kwargs):
        # 选择 features 中的第一个位置处的特征向量，通常代表 [CLS] 标记的特征
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)  # 应用 Dropout 层
        x = self.dense(x)  # 线性变换
        # 使用配置中指定的激活函数处理 x
        x = ACT2FN[self.config.hidden_act](x)
        x = self.dropout(x)  # 再次应用 Dropout 层
        x = self.out_proj(x)  # 输出层线性变换
        return x


@add_start_docstrings(
    """YOSO Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks.""",
    YOSO_START_DOCSTRING,
)
class YosoForSequenceClassification(YosoPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        # YOSO 模型的初始化
        self.yoso = YosoModel(config)
        # 分类器头部初始化
        self.classifier = YosoClassificationHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(YOSO_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
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
        ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 如果 return_dict 不为 None，则使用该值；否则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用模型的前向传播方法 yoso，传入各种输入参数
        outputs = self.yoso(
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
        # 将序列输出传入分类器，得到 logits
        logits = self.classifier(sequence_output)

        # 初始化损失值
        loss = None
        # 如果给定了标签 labels，则计算相应的损失
        if labels is not None:
            # 如果问题类型未定义，则根据情况进行定义
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型选择对应的损失函数
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    # 如果只有一个标签，使用损失函数计算损失
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                # 使用交叉熵损失函数计算损失
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                # 使用带 logits 的二元交叉熵损失函数计算损失
                loss = loss_fct(logits, labels)

        # 如果 return_dict 为 False，则返回 logits 和额外的输出项
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回带有损失、logits、隐藏状态和注意力权重的 SequenceClassifierOutput 对象
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用装饰器为类添加文档字符串，描述其作为多选分类模型的YOSO模型
@add_start_docstrings(
    """YOSO Model with a multiple choice classification head on top (a linear layer on top of
    the pooled output and a softmax) e.g. for RocStories/SWAG tasks.""",
    YOSO_START_DOCSTRING,
)
# 定义YosoForMultipleChoice类，继承自YosoPreTrainedModel
class YosoForMultipleChoice(YosoPreTrainedModel):
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 初始化Yoso模型
        self.yoso = YosoModel(config)
        # 初始化预分类器，使用线性层将隐藏状态大小映射到相同的隐藏状态大小
        self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)
        # 初始化分类器，使用线性层将隐藏状态大小映射到1，用于多选分类任务
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 调用后处理初始化方法
        self.post_init()

    # 使用装饰器添加文档字符串描述forward方法的输入参数
    @add_start_docstrings_to_model_forward(YOSO_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    # 使用装饰器添加代码示例和检查点等文档字符串，指定输出类型为MultipleChoiceModelOutput
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
        # forward方法的参数说明完毕，没有实现具体的功能逻辑
        ) -> Union[Tuple, MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 确保返回的字典对象不为空，根据配置决定是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 获取输入的选项数量，如果是通过 input_ids 计算得到的话
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 重新组织输入，将各种输入类型展平为二维张量，便于模型处理
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 将处理后的输入传递给模型，获取模型的输出
        outputs = self.yoso(
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

        # 从模型输出中获取隐藏状态
        hidden_state = outputs[0]  # (bs * num_choices, seq_len, dim)
        # 从隐藏状态中提取池化输出，通常是第一个位置的隐藏状态
        pooled_output = hidden_state[:, 0]  # (bs * num_choices, dim)
        # 将池化输出传递给预分类器，进行进一步处理
        pooled_output = self.pre_classifier(pooled_output)  # (bs * num_choices, dim)
        # 使用 ReLU 激活函数处理池化输出
        pooled_output = nn.ReLU()(pooled_output)  # (bs * num_choices, dim)
        # 将处理后的池化输出传递给分类器，得到最终的 logits
        logits = self.classifier(pooled_output)

        # 重新调整 logits 的形状，使其与 labels 的形状匹配
        reshaped_logits = logits.view(-1, num_choices)

        # 计算损失，如果提供了 labels 的话
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 根据 return_dict 决定返回的格式
        if not return_dict:
            # 如果不要求返回字典，则返回一个元组
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果要求返回字典，则返回一个 MultipleChoiceModelOutput 对象
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# YOSO Model with a token classification head on top (a linear layer on top of
# the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.
# 继承自 YosoPreTrainedModel 类的 YosoForTokenClassification 类，用于在 YOSO 模型基础上添加标记分类头部，
# 例如用于命名实体识别（NER）任务。

class YosoForTokenClassification(YosoPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # 初始化 YOSO 模型
        self.yoso = YosoModel(config)
        # Dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 线性分类器，将隐藏状态映射到标签数量的输出
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(YOSO_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 前向传播函数，接受多种输入参数并返回输出，用于模型推理和训练
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
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 如果 return_dict 为 None，则根据配置决定是否使用 return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用模型的 forward 方法进行预测
        outputs = self.yoso(
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

        # 从模型输出中取出序列输出
        sequence_output = outputs[0]

        # 对序列输出应用 dropout
        sequence_output = self.dropout(sequence_output)
        
        # 使用分类器进行分类得到 logits
        logits = self.classifier(sequence_output)

        # 初始化损失为 None
        loss = None
        if labels is not None:
            # 使用交叉熵损失函数
            loss_fct = CrossEntropyLoss()
            
            # 只保留激活部分的损失
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不使用 return_dict，则返回 logits 和额外的输出
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 使用 TokenClassifierOutput 封装并返回结果
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# YOSO模型，用于支持像SQuAD这样的抽取式问答任务，具有一个用于计算“起始位置logits”和“结束位置logits”的线性分类头部。
# 该模型继承自YosoPreTrainedModel。
@add_start_docstrings(
    """YOSO Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).""",
    YOSO_START_DOCSTRING,
)
class YosoForQuestionAnswering(YosoPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 设置模型的标签数目为2
        config.num_labels = 2
        self.num_labels = config.num_labels

        # 初始化YOSO模型和用于问答输出的线性层
        self.yoso = YosoModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(YOSO_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
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
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 下面是模型前向传播所需的输入参数说明
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
        # 如果 return_dict 为 None，则使用配置中的 use_return_dict 值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用模型 yoso，传入各种输入参数，并返回输出
        outputs = self.yoso(
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

        # 从模型输出中取出序列输出（通常是 BERT 输出的第一个元素）
        sequence_output = outputs[0]

        # 将序列输出传入问答模型的输出层，得到起始和结束 logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 如果 start_positions 和 end_positions 不为空，则计算损失
            # 如果在多 GPU 下训练，可能需要增加维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 忽略超出模型输入的 start/end positions
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 使用交叉熵损失函数，忽略索引为 ignored_index 的位置
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            # 如果 return_dict 为 False，则返回元组形式的输出
            output = (start_logits, end_logits) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 如果 return_dict 为 True，则返回 QuestionAnsweringModelOutput 类的实例
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```