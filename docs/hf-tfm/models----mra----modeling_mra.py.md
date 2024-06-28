# `.\models\mra\modeling_mra.py`

```py
# coding=utf-8
# Copyright 2023 University of Wisconsin-Madison and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch MRA model."""


import math  # 导入数学模块
from pathlib import Path  # 导入路径操作模块
from typing import Optional, Tuple, Union  # 导入类型提示模块

import torch  # 导入PyTorch库
import torch.utils.checkpoint  # 导入PyTorch的checkpoint工具
from torch import nn  # 导入神经网络模块
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss  # 导入损失函数
from torch.utils.cpp_extension import load  # 导入C++扩展加载模块

from ...activations import ACT2FN  # 导入激活函数映射
from ...modeling_outputs import (  # 导入模型输出
    BaseModelOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel  # 导入预训练模型工具类
from ...pytorch_utils import (  # 导入PyTorch工具类
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from ...utils import (  # 导入通用工具函数
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_ninja_available,
    is_torch_cuda_available,
    logging,
)
from .configuration_mra import MraConfig  # 导入MRA模型配置


logger = logging.get_logger(__name__)  # 获取日志记录器

_CHECKPOINT_FOR_DOC = "uw-madison/mra-base-512-4"  # 文档中使用的检查点
_CONFIG_FOR_DOC = "MraConfig"  # 文档中使用的配置
_TOKENIZER_FOR_DOC = "AutoTokenizer"  # 文档中使用的分词器

MRA_PRETRAINED_MODEL_ARCHIVE_LIST = [  # MRA预训练模型存档列表
    "uw-madison/mra-base-512-4",
    # 查看所有MRA模型：https://huggingface.co/models?filter=mra
]

mra_cuda_kernel = None  # 初始化MRA CUDA内核为None


def load_cuda_kernels():
    global mra_cuda_kernel  # 使用全局变量
    src_folder = Path(__file__).resolve().parent.parent.parent / "kernels" / "mra"  # 设置CUDA内核源文件夹路径

    def append_root(files):  # 定义一个函数，在文件列表中加入根路径
        return [src_folder / file for file in files]

    src_files = append_root(["cuda_kernel.cu", "cuda_launch.cu", "torch_extension.cpp"])  # CUDA内核源文件列表

    mra_cuda_kernel = load("cuda_kernel", src_files, verbose=True)  # 加载CUDA内核


def sparse_max(sparse_qk_prod, indices, query_num_block, key_num_block):
    """
    Computes maximum values for softmax stability.
    计算softmax稳定性的最大值。
    """
    if len(sparse_qk_prod.size()) != 4:  # 检查输入张量维度是否为4
        raise ValueError("sparse_qk_prod must be a 4-dimensional tensor.")

    if len(indices.size()) != 2:  # 检查索引张量维度是否为2
        raise ValueError("indices must be a 2-dimensional tensor.")

    if sparse_qk_prod.size(2) != 32:  # 检查sparse_qk_prod的第二个维度大小是否为32
        raise ValueError("The size of the second dimension of sparse_qk_prod must be 32.")

    if sparse_qk_prod.size(3) != 32:  # 检查sparse_qk_prod的第三个维度大小是否为32
        raise ValueError("The size of the third dimension of sparse_qk_prod must be 32.")

    index_vals = sparse_qk_prod.max(dim=-2).values.transpose(-1, -2)  # 计算最大值并转置
    # 调用PyTorch的contiguous()方法，确保index_vals是一个连续的Tensor
    index_vals = index_vals.contiguous()
    
    # 将indices转换为整型Tensor，并确保它是连续的
    indices = indices.int()
    indices = indices.contiguous()
    
    # 调用mra_cuda_kernel的index_max方法，使用index_vals和indices进行计算，
    # query_num_block和key_num_block是用于计算的块数量参数
    max_vals, max_vals_scatter = mra_cuda_kernel.index_max(index_vals, indices, query_num_block, key_num_block)
    
    # 将max_vals_scatter的最后两个维度进行转置操作，并在倒数第二个位置插入一个新的维度
    max_vals_scatter = max_vals_scatter.transpose(-1, -2)[:, :, None, :]
    
    # 返回计算结果max_vals和max_vals_scatter
    return max_vals, max_vals_scatter
# 稀疏掩码转换函数，用于生成高分辨率逻辑的稀疏掩码
def sparse_mask(mask, indices, block_size=32):
    # 检查掩码是否为二维张量
    if len(mask.size()) != 2:
        raise ValueError("mask must be a 2-dimensional tensor.")
    
    # 检查索引是否为二维张量
    if len(indices.size()) != 2:
        raise ValueError("indices must be a 2-dimensional tensor.")
    
    # 检查掩码和索引的第一维度是否相等
    if mask.shape[0] != indices.shape[0]:
        raise ValueError("mask and indices must have the same size in the zero-th dimension.")
    
    batch_size, seq_len = mask.shape
    num_block = seq_len // block_size
    
    # 创建一个批次索引张量
    batch_idx = torch.arange(indices.size(0), dtype=torch.long, device=indices.device)
    # 将掩码重塑为批次大小、块数量和块大小的张量
    mask = mask.reshape(batch_size, num_block, block_size)
    # 根据索引和块数量选择相应的掩码块
    mask = mask[batch_idx[:, None], (indices % num_block).long(), :]
    
    return mask


# 执行稀疏密集矩阵乘法的函数
def mm_to_sparse(dense_query, dense_key, indices, block_size=32):
    batch_size, query_size, dim = dense_query.size()
    _, key_size, dim = dense_key.size()
    
    # 检查查询大小是否可以被块大小整除
    if query_size % block_size != 0:
        raise ValueError("query_size (size of first dimension of dense_query) must be divisible by block_size.")
    
    # 检查键大小是否可以被块大小整除
    if key_size % block_size != 0:
        raise ValueError("key_size (size of first dimension of dense_key) must be divisible by block_size.")
    
    # 将密集查询和密集键重塑为批次大小、块数量、块大小和维度的张量，并交换最后两个维度
    dense_query = dense_query.reshape(batch_size, query_size // block_size, block_size, dim).transpose(-1, -2)
    dense_key = dense_key.reshape(batch_size, key_size // block_size, block_size, dim).transpose(-1, -2)
    
    # 检查密集查询和密集键是否为四维张量
    if len(dense_query.size()) != 4:
        raise ValueError("dense_query must be a 4-dimensional tensor.")
    
    if len(dense_key.size()) != 4:
        raise ValueError("dense_key must be a 4-dimensional tensor.")
    
    # 检查索引是否为二维张量，并确保密集查询和密集键的第三维度为32
    if len(indices.size()) != 2:
        raise ValueError("indices must be a 2-dimensional tensor.")
    
    if dense_query.size(3) != 32:
        raise ValueError("The third dimension of dense_query must be 32.")
    
    if dense_key.size(3) != 32:
        raise ValueError("The third dimension of dense_key must be 32.")
    
    # 使得密集查询、密集键和索引的内存连续性
    dense_query = dense_query.contiguous()
    dense_key = dense_key.contiguous()
    indices = indices.int().contiguous()
    
    # 调用底层 CUDA 内核函数执行稀疏化密集矩阵乘法
    return mra_cuda_kernel.mm_to_sparse(dense_query, dense_key, indices.int())


# 执行稀疏密集矩阵乘法的逆操作函数
def sparse_dense_mm(sparse_query, indices, dense_key, query_num_block, block_size=32):
    batch_size, key_size, dim = dense_key.size()
    
    # 检查密集键的大小是否可以被块大小整除
    if key_size % block_size != 0:
        raise ValueError("key_size (size of first dimension of dense_key) must be divisible by block_size.")
    
    # 检查稀疏查询的第二维和第三维大小是否等于块大小
    if sparse_query.size(2) != block_size:
        raise ValueError("The size of the second dimension of sparse_query must be equal to the block_size.")
    
    if sparse_query.size(3) != block_size:
        raise ValueError("The size of the third dimension of sparse_query must be equal to the block_size.")
    # 将密集键 reshape 成指定形状，以便进行后续操作
    dense_key = dense_key.reshape(batch_size, key_size // block_size, block_size, dim).transpose(-1, -2)
    
    # 检查稀疏查询的维度是否为四维，否则引发数值错误异常
    if len(sparse_query.size()) != 4:
        raise ValueError("sparse_query must be a 4-dimensional tensor.")
    
    # 检查密集键的维度是否为四维，否则引发数值错误异常
    if len(dense_key.size()) != 4:
        raise ValueError("dense_key must be a 4-dimensional tensor.")
    
    # 检查索引的维度是否为二维，否则引发数值错误异常
    if len(indices.size()) != 2:
        raise ValueError("indices must be a 2-dimensional tensor.")
    
    # 检查密集键的第三维是否为32，否则引发数值错误异常
    if dense_key.size(3) != 32:
        raise ValueError("The size of the third dimension of dense_key must be 32.")
    
    # 确保稀疏查询在内存中是连续的
    sparse_query = sparse_query.contiguous()
    
    # 将索引转换为整型类型，并确保在内存中是连续的
    indices = indices.int()
    indices = indices.contiguous()
    
    # 确保密集键在内存中是连续的
    dense_key = dense_key.contiguous()
    
    # 使用自定义 CUDA 核函数进行稀疏-密集矩阵乘法，生成密集查询-键乘积
    dense_qk_prod = mra_cuda_kernel.sparse_dense_mm(sparse_query, indices, dense_key, query_num_block)
    
    # 转置乘积张量的后两个维度，并将其 reshape 成指定形状
    dense_qk_prod = dense_qk_prod.transpose(-1, -2).reshape(batch_size, query_num_block * block_size, dim)
    
    # 返回最终的密集查询-键乘积张量
    return dense_qk_prod
def transpose_indices(indices, dim_1_block, dim_2_block):
    # 计算索引的转置，将二维块索引转换为一维块索引
    return ((indices % dim_2_block) * dim_1_block + torch.div(indices, dim_2_block, rounding_mode="floor")).long()


class MraSampledDenseMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dense_query, dense_key, indices, block_size):
        # 计算稠密查询和键的乘积，并将结果转换为稀疏格式
        sparse_qk_prod = mm_to_sparse(dense_query, dense_key, indices, block_size)
        ctx.save_for_backward(dense_query, dense_key, indices)
        ctx.block_size = block_size
        return sparse_qk_prod

    @staticmethod
    def backward(ctx, grad):
        dense_query, dense_key, indices = ctx.saved_tensors
        block_size = ctx.block_size
        query_num_block = dense_query.size(1) // block_size
        key_num_block = dense_key.size(1) // block_size
        # 计算转置后的索引，用于反向传播梯度
        indices_T = transpose_indices(indices, query_num_block, key_num_block)
        grad_key = sparse_dense_mm(grad.transpose(-1, -2), indices_T, dense_query, key_num_block)
        grad_query = sparse_dense_mm(grad, indices, dense_key, query_num_block)
        return grad_query, grad_key, None, None

    @staticmethod
    def operator_call(dense_query, dense_key, indices, block_size=32):
        # 调用前向传播函数
        return MraSampledDenseMatMul.apply(dense_query, dense_key, indices, block_size)


class MraSparseDenseMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sparse_query, indices, dense_key, query_num_block):
        # 计算稀疏查询和键的乘积，并将结果返回
        sparse_qk_prod = sparse_dense_mm(sparse_query, indices, dense_key, query_num_block)
        ctx.save_for_backward(sparse_query, indices, dense_key)
        ctx.query_num_block = query_num_block
        return sparse_qk_prod

    @staticmethod
    def backward(ctx, grad):
        sparse_query, indices, dense_key = ctx.saved_tensors
        query_num_block = ctx.query_num_block
        key_num_block = dense_key.size(1) // sparse_query.size(-1)
        # 计算转置后的索引，用于反向传播梯度
        indices_T = transpose_indices(indices, query_num_block, key_num_block)
        grad_key = sparse_dense_mm(sparse_query.transpose(-1, -2), indices_T, grad, key_num_block)
        grad_query = mm_to_sparse(grad, dense_key, indices)
        return grad_query, None, grad_key, None

    @staticmethod
    def operator_call(sparse_query, indices, dense_key, query_num_block):
        # 调用前向传播函数
        return MraSparseDenseMatMul.apply(sparse_query, indices, dense_key, query_num_block)


class MraReduceSum:
    @staticmethod
    # 定义一个函数operator_call，接受稀疏查询sparse_query、索引indices、查询块数query_num_block和键块数key_num_block作为参数
    def operator_call(sparse_query, indices, query_num_block, key_num_block):
        # 获取稀疏查询sparse_query的尺寸信息，包括批次大小batch_size、块数num_block、块大小block_size和未使用的维度（_）
        batch_size, num_block, block_size, _ = sparse_query.size()

        # 检查稀疏查询sparse_query是否为4维张量，如果不是则抛出ValueError异常
        if len(sparse_query.size()) != 4:
            raise ValueError("sparse_query must be a 4-dimensional tensor.")

        # 检查索引indices是否为2维张量，如果不是则抛出ValueError异常
        if len(indices.size()) != 2:
            raise ValueError("indices must be a 2-dimensional tensor.")

        # 重新获取稀疏查询sparse_query的尺寸信息，只关注批次大小batch_size和块数num_block，以及块大小block_size
        _, _, block_size, _ = sparse_query.size()
        
        # 获取索引indices的尺寸信息，包括批次大小batch_size和块数num_block
        batch_size, num_block = indices.size()

        # 对稀疏查询sparse_query按第2维求和，然后重新形状为(batch_size * num_block, block_size)
        sparse_query = sparse_query.sum(dim=2).reshape(batch_size * num_block, block_size)

        # 创建一个长为indices.size(0)的长整型张量batch_idx，设备与indices.device相同
        batch_idx = torch.arange(indices.size(0), dtype=torch.long, device=indices.device)
        
        # 计算全局索引global_idxes，通过除以key_num_block向下取整并转换为长整型，加上batch_idx的扩展乘以query_num_block
        global_idxes = (
            torch.div(indices, key_num_block, rounding_mode="floor").long() + batch_idx[:, None] * query_num_block
        ).reshape(batch_size * num_block)
        
        # 创建一个全零张量temp，形状为(batch_size * query_num_block, block_size)，数据类型与sparse_query一致，设备与sparse_query.device相同
        temp = torch.zeros(
            (batch_size * query_num_block, block_size), dtype=sparse_query.dtype, device=sparse_query.device
        )
        
        # 在temp的0维度上按照global_idxes进行索引添加sparse_query的数据，然后重新形状为(batch_size, query_num_block, block_size)
        output = temp.index_add(0, global_idxes, sparse_query).reshape(batch_size, query_num_block, block_size)

        # 将output重新形状为(batch_size, query_num_block * block_size)，并返回结果
        output = output.reshape(batch_size, query_num_block * block_size)
        return output
# 计算查询张量的批次大小、序列长度和注意力头维度
batch_size, seq_len, head_dim = query.size()

# 计算每行中的块数
num_block_per_row = seq_len // block_size

# 如果存在掩码，则计算每个块的令牌数量，并分别计算查询、键和可选值的低分辨率估计
value_hat = None
if mask is not None:
    token_count = mask.reshape(batch_size, num_block_per_row, block_size).sum(dim=-1)
    query_hat = query.reshape(batch_size, num_block_per_row, block_size, head_dim).sum(dim=-2) / (
        token_count[:, :, None] + 1e-6
    )
    key_hat = key.reshape(batch_size, num_block_per_row, block_size, head_dim).sum(dim=-2) / (
        token_count[:, :, None] + 1e-6
    )
    if value is not None:
        value_hat = value.reshape(batch_size, num_block_per_row, block_size, head_dim).sum(dim=-2) / (
            token_count[:, :, None] + 1e-6
        )
# 如果没有掩码，则假设所有块具有相同数量的令牌，并计算查询、键和可选值的均值
else:
    token_count = block_size * torch.ones(batch_size, num_block_per_row, dtype=torch.float, device=query.device)
    query_hat = query.reshape(batch_size, num_block_per_row, block_size, head_dim).mean(dim=-2)
    key_hat = key.reshape(batch_size, num_block_per_row, block_size, head_dim).mean(dim=-2)
    if value is not None:
        value_hat = value.reshape(batch_size, num_block_per_row, block_size, head_dim).mean(dim=-2)

# 计算低分辨率对数线性模型，使用查询的估计和键的估计之间的乘积，并除以头维度的平方根
low_resolution_logit = torch.matmul(query_hat, key_hat.transpose(-1, -2)) / math.sqrt(head_dim)

# 计算低分辨率对数线性模型每行的最大值
low_resolution_logit_row_max = low_resolution_logit.max(dim=-1, keepdims=True).values

# 如果存在掩码，则将低分辨率对数线性模型中小于阈值的元素置为负无穷大
if mask is not None:
    low_resolution_logit = (
        low_resolution_logit - 1e4 * ((token_count[:, None, :] * token_count[:, :, None]) < 0.5).float()
    )

# 返回低分辨率对数线性模型、令牌计数、每行最大值和可选值的估计
return low_resolution_logit, token_count, low_resolution_logit_row_max, value_hat
    # 获取 top_k_vals 中的索引信息
    indices = top_k_vals.indices

    # 根据 approx_mode 的取值进行不同的处理
    if approx_mode == "full":
        # 计算 top_k_vals 中每行最小值作为阈值
        threshold = top_k_vals.values.min(dim=-1).values
        # 生成一个高分辨率掩码，使得低分辨率的 logits 大于等于对应的阈值
        high_resolution_mask = (low_resolution_logit >= threshold[:, None, None]).float()
    elif approx_mode == "sparse":
        # 如果是稀疏模式，则高分辨率掩码设为 None
        high_resolution_mask = None
    else:
        # 抛出异常，提示 approx_mode 不是有效的值
        raise ValueError(f"{approx_mode} is not a valid approx_model value.")

    # 返回计算得到的索引和高分辨率掩码
    return indices, high_resolution_mask
    """
    使用 Mra 来近似自注意力机制。
    """
    # 如果未加载 CUDA 核心，返回一个和 query 形状相同的全零张量并标记需要梯度
    if mra_cuda_kernel is None:
        return torch.zeros_like(query).requires_grad_()

    # 获取 query 的形状信息
    batch_size, num_head, seq_len, head_dim = query.size()
    # 计算元批次大小
    meta_batch = batch_size * num_head

    # 检查序列长度是否能整除 block_size
    if seq_len % block_size != 0:
        raise ValueError("sequence length must be divisible by the block_size.")

    # 计算每行的块数
    num_block_per_row = seq_len // block_size

    # 重塑 query, key, value 张量的形状
    query = query.reshape(meta_batch, seq_len, head_dim)
    key = key.reshape(meta_batch, seq_len, head_dim)
    value = value.reshape(meta_batch, seq_len, head_dim)

    # 如果存在掩码，将其应用到 query, key, value 上
    if mask is not None:
        query = query * mask[:, :, None]
        key = key * mask[:, :, None]
        value = value * mask[:, :, None]

    # 根据 approx_mode 调用不同的低分辨率逻辑
    if approx_mode == "full":
        # 获取低分辨率逻辑相关的值
        low_resolution_logit, token_count, low_resolution_logit_row_max, value_hat = get_low_resolution_logit(
            query, key, block_size, mask, value
        )
    elif approx_mode == "sparse":
        # 在无梯度计算环境下获取低分辨率逻辑相关的值
        with torch.no_grad():
            low_resolution_logit, token_count, low_resolution_logit_row_max, _ = get_low_resolution_logit(
                query, key, block_size, mask
            )
    else:
        # 如果 approx_mode 不是 "full" 或 "sparse"，抛出异常
        raise Exception('approx_mode must be "full" or "sparse"')

    # 计算低分辨率逻辑的归一化值
    with torch.no_grad():
        low_resolution_logit_normalized = low_resolution_logit - low_resolution_logit_row_max
        # 获取块索引和高分辨率掩码
        indices, high_resolution_mask = get_block_idxes(
            low_resolution_logit_normalized,
            num_blocks,
            approx_mode,
            initial_prior_first_n_blocks,
            initial_prior_diagonal_n_blocks,
        )

    # 计算高分辨率逻辑
    high_resolution_logit = MraSampledDenseMatMul.operator_call(
        query, key, indices, block_size=block_size
    ) / math.sqrt(head_dim)
    # 计算最大值及其散列版本
    max_vals, max_vals_scatter = sparse_max(high_resolution_logit, indices, num_block_per_row, num_block_per_row)
    # 对高分辨率逻辑进行归一化处理
    high_resolution_logit = high_resolution_logit - max_vals_scatter
    # 如果存在掩码，对高分辨率逻辑进行额外处理
    if mask is not None:
        high_resolution_logit = high_resolution_logit - 1e4 * (1 - sparse_mask(mask, indices)[:, :, :, None])
    # 计算高分辨率注意力分布
    high_resolution_attn = torch.exp(high_resolution_logit)
    # 计算高分辨率注意力输出
    high_resolution_attn_out = MraSparseDenseMatMul.operator_call(
        high_resolution_attn, indices, value, num_block_per_row
    )
    # 计算高分辨率正则化因子
    high_resolution_normalizer = MraReduceSum.operator_call(
        high_resolution_attn, indices, num_block_per_row, num_block_per_row
    )
    # 如果近似模式为 "full"，则进行全模式的注意力计算
    if approx_mode == "full":
        # 计算低分辨率注意力权重
        low_resolution_attn = (
            torch.exp(low_resolution_logit - low_resolution_logit_row_max - 1e4 * high_resolution_mask)
            * token_count[:, None, :]
        )

        # 计算低分辨率注意力输出
        low_resolution_attn_out = (
            torch.matmul(low_resolution_attn, value_hat)[:, :, None, :]
            .repeat(1, 1, block_size, 1)
            .reshape(meta_batch, seq_len, head_dim)
        )

        # 计算低分辨率注意力的归一化因子
        low_resolution_normalizer = (
            low_resolution_attn.sum(dim=-1)[:, :, None].repeat(1, 1, block_size).reshape(meta_batch, seq_len)
        )

        # 计算对数修正项，用于调整低分辨率注意力输出
        log_correction = low_resolution_logit_row_max.repeat(1, 1, block_size).reshape(meta_batch, seq_len) - max_vals
        if mask is not None:
            log_correction = log_correction * mask

        # 计算低分辨率注意力的修正系数
        low_resolution_corr = torch.exp(log_correction * (log_correction <= 0).float())
        low_resolution_attn_out = low_resolution_attn_out * low_resolution_corr[:, :, None]
        low_resolution_normalizer = low_resolution_normalizer * low_resolution_corr

        # 计算高分辨率注意力的修正系数
        high_resolution_corr = torch.exp(-log_correction * (log_correction > 0).float())
        high_resolution_attn_out = high_resolution_attn_out * high_resolution_corr[:, :, None]
        high_resolution_normalizer = high_resolution_normalizer * high_resolution_corr

        # 计算最终的上下文层，结合了高低分辨率的注意力
        context_layer = (high_resolution_attn_out + low_resolution_attn_out) / (
            high_resolution_normalizer[:, :, None] + low_resolution_normalizer[:, :, None] + 1e-6
        )

    # 如果近似模式为 "sparse"，则进行稀疏模式的注意力计算
    elif approx_mode == "sparse":
        # 计算高分辨率注意力输出
        context_layer = high_resolution_attn_out / (high_resolution_normalizer[:, :, None] + 1e-6)
    else:
        # 如果近似模式既不是 "full" 也不是 "sparse"，则抛出异常
        raise Exception('config.approx_mode must be "full" or "sparse"')

    # 如果存在掩码，则应用掩码到上下文层
    if mask is not None:
        context_layer = context_layer * mask[:, :, None]

    # 将上下文层重塑成(batch_size, num_head, seq_len, head_dim)的形状
    context_layer = context_layer.reshape(batch_size, num_head, seq_len, head_dim)

    # 返回最终的上下文层
    return context_layer
class MraEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        # 初始化词嵌入层，根据词汇大小、隐藏大小和填充标识符创建嵌入层
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 初始化位置嵌入层，根据最大位置嵌入大小创建嵌入层
        self.position_embeddings = nn.Embedding(config.max_position_embeddings + 2, config.hidden_size)
        # 初始化标记类型嵌入层，根据类型词汇大小创建嵌入层
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 使用 TensorFlow 模型变量名，并且能够加载任何 TensorFlow 检查点文件，因此未改为蛇形命名的 self.LayerNorm
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 注册缓冲区 "position_ids"，用于存储序列位置 ID
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)) + 2)
        # 位置嵌入类型，默认为绝对位置编码
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 注册缓冲区 "token_type_ids"，存储标记类型 ID，默认为全零
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
            persistent=False,
        )

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            # 如果未提供位置 IDs，则使用注册的 position_ids
            position_ids = self.position_ids[:, :seq_length]

        # 如果未提供 token_type_ids，则使用注册的缓冲区中的全零，扩展以匹配输入形状
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            # 如果未提供 inputs_embeds，则使用 word_embeddings 层对 input_ids 进行嵌入
            inputs_embeds = self.word_embeddings(input_ids)
        # 获取 token_type_embeddings
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将词嵌入和标记类型嵌入相加
        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            # 如果使用绝对位置编码，获取位置嵌入并加到 embeddings 中
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        # LayerNorm 标准化 embeddings
        embeddings = self.LayerNorm(embeddings)
        # 使用 dropout 进行 embeddings 的随机失活
        embeddings = self.dropout(embeddings)
        return embeddings
    # 初始化函数，接受配置参数和位置嵌入类型作为可选参数
    def __init__(self, config, position_embedding_type=None):
        # 调用父类的初始化方法
        super().__init__()
        
        # 检查隐藏层大小是否是注意力头数目的整数倍，如果不是且配置中没有嵌入大小属性，则引发值错误异常
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 检查是否支持 Torch CUDA，Ninja 构建系统可用，并且自定义 CUDA 内核未加载，则尝试加载 CUDA 内核
        kernel_loaded = mra_cuda_kernel is not None
        if is_torch_cuda_available() and is_ninja_available() and not kernel_loaded:
            try:
                load_cuda_kernels()
            except Exception as e:
                # 如果加载失败，记录警告信息
                logger.warning(f"Could not load the custom kernel for multi-scale deformable attention: {e}")

        # 初始化注意力头数和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 分别初始化查询、键、值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 初始化注意力概率的丢弃层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        # 设置位置嵌入类型，如果未提供则使用配置中的位置嵌入类型
        self.position_embedding_type = (
            position_embedding_type if position_embedding_type is not None else config.position_embedding_type
        )

        # 计算块的数量，最多不超过配置中允许的位置嵌入最大数量的平方
        self.num_block = (config.max_position_embeddings // 32) * config.block_per_row
        self.num_block = min(self.num_block, int((config.max_position_embeddings // 32) ** 2))

        # 设置近似模式和初始优先级的前几个块数以及对角线上的块数
        self.approx_mode = config.approx_mode
        self.initial_prior_first_n_blocks = config.initial_prior_first_n_blocks
        self.initial_prior_diagonal_n_blocks = config.initial_prior_diagonal_n_blocks

    # 重塑张量形状以便进行注意力计算，将最后一维分割为注意力头和每个头的大小
    def transpose_for_scores(self, layer):
        new_layer_shape = layer.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        layer = layer.view(*new_layer_shape)
        return layer.permute(0, 2, 1, 3)
    # 定义前向传播函数，用于处理输入的隐藏状态和注意力掩码
    def forward(self, hidden_states, attention_mask=None):
        # 生成查询向量，通过 self.query 函数对隐藏状态进行处理
        mixed_query_layer = self.query(hidden_states)

        # 生成键向量，通过 self.key 函数对隐藏状态进行处理，并转置以便进行注意力计算
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        
        # 生成值向量，通过 self.value 函数对隐藏状态进行处理，并转置以便进行注意力计算
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # 转换查询向量的维度，以便进行注意力计算
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 获取 batch_size, num_heads, seq_len, head_dim 四个维度的大小
        batch_size, num_heads, seq_len, head_dim = query_layer.size()

        # 根据注意力掩码进行调整，将其归一化处理并乘以一个比例因子
        attention_mask = 1.0 + attention_mask / 10000.0
        attention_mask = (
            attention_mask.squeeze().repeat(1, num_heads, 1).reshape(batch_size * num_heads, seq_len).int()
        )

        # 对于 head_dim 小于 GPU 的 warp 大小（32）的情况，进行维度调整和填充操作
        gpu_warp_size = 32
        if head_dim < gpu_warp_size:
            pad_size = batch_size, num_heads, seq_len, gpu_warp_size - head_dim

            # 在查询、键和值向量的最后一个维度上拼接零张量，以满足 GPU warp 大小的要求
            query_layer = torch.cat([query_layer, torch.zeros(pad_size, device=query_layer.device)], dim=-1)
            key_layer = torch.cat([key_layer, torch.zeros(pad_size, device=key_layer.device)], dim=-1)
            value_layer = torch.cat([value_layer, torch.zeros(pad_size, device=value_layer.device)], dim=-1)

        # 调用自定义的多头相对注意力函数 mra2_attention 进行注意力计算
        context_layer = mra2_attention(
            query_layer.float(),
            key_layer.float(),
            value_layer.float(),
            attention_mask.float(),
            self.num_block,
            approx_mode=self.approx_mode,
            initial_prior_first_n_blocks=self.initial_prior_first_n_blocks,
            initial_prior_diagonal_n_blocks=self.initial_prior_diagonal_n_blocks,
        )

        # 如果 head_dim 小于 GPU warp 大小，截取计算后的 context_layer 的最后一个维度
        if head_dim < gpu_warp_size:
            context_layer = context_layer[:, :, :, :head_dim]

        # 调整 context_layer 的维度顺序，以便与输出维度一致
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # 重新整形 context_layer，以适应输出的全部头部大小
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # 输出结果，包含调整后的 context_layer
        outputs = (context_layer,)

        # 返回最终的输出结果
        return outputs
# Copied from transformers.models.bert.modeling_bert.BertSelfOutput

class MraSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义一个线性层，用于变换隐藏状态的维度
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # LayerNorm 层，用于对输入进行归一化处理
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout 层，用于随机失活一部分神经元，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 线性变换
        hidden_states = self.dense(hidden_states)
        # Dropout 处理
        hidden_states = self.dropout(hidden_states)
        # LayerNorm 处理
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class MraAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # MraSelfAttention 类的实例化，用于自注意力计算
        self.self = MraSelfAttention(config, position_embedding_type=position_embedding_type)
        # MraSelfOutput 类的实例化，用于处理自注意力输出
        self.output = MraSelfOutput(config)
        # 用于存储被修剪的注意力头信息的集合
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 根据头信息进行注意力头的修剪
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 修剪线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储被修剪的头信息
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, attention_mask=None):
        # 调用 self 层的 forward 方法，计算自注意力
        self_outputs = self.self(hidden_states, attention_mask)
        # 调用 output 层的 forward 方法，处理自注意力的输出
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # 如果输出了注意力信息，将其加入输出元组
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate

class MraIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 线性层，用于将隐藏状态映射到中间状态
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 中间激活函数，根据配置选择对应的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 线性变换
        hidden_states = self.dense(hidden_states)
        # 中间激活函数处理
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput

class MraOutput(nn.Module):
    # 这部分截断了，无法为其添加注释
    # 初始化函数，用于初始化对象
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个全连接层，输入维度是config.intermediate_size，输出维度是config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个 Layer Normalization 层，输入维度是config.hidden_size，设置epsilon为config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个 Dropout 层，概率为config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，接收两个张量作为输入并返回一个张量
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用全连接层处理hidden_states张量
        hidden_states = self.dense(hidden_states)
        # 对处理后的张量进行 Dropout 操作
        hidden_states = self.dropout(hidden_states)
        # 对结果张量进行 Layer Normalization，并与输入张量相加
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理后的张量作为输出
        return hidden_states
# 定义一个名为 MraLayer 的类，继承自 nn.Module
class MraLayer(nn.Module):
    # 初始化方法，接收一个 config 对象作为参数
    def __init__(self, config):
        super().__init__()
        # 设置自身属性 chunk_size_feed_forward 为 config 中的 chunk_size_feed_forward 属性
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 设置自身属性 seq_len_dim 为 1
        self.seq_len_dim = 1
        # 初始化 self.attention 为 MraAttention 类的一个实例，使用传入的 config 参数
        self.attention = MraAttention(config)
        # 设置 self.add_cross_attention 属性为 config 中的 add_cross_attention 属性
        self.add_cross_attention = config.add_cross_attention
        # 初始化 self.intermediate 为 MraIntermediate 类的一个实例，使用传入的 config 参数
        self.intermediate = MraIntermediate(config)
        # 初始化 self.output 为 MraOutput 类的一个实例，使用传入的 config 参数
        self.output = MraOutput(config)

    # 前向传播方法，接收 hidden_states 和 attention_mask 两个参数
    def forward(self, hidden_states, attention_mask=None):
        # 使用 self.attention 对象处理 hidden_states 和 attention_mask，得到 self_attention_outputs
        self_attention_outputs = self.attention(hidden_states, attention_mask)
        # 从 self_attention_outputs 中获取注意力输出 attention_output
        attention_output = self_attention_outputs[0]

        # 如果需要输出注意力权重，则将额外的自注意力输出添加到 outputs 中
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # 将 attention_output 应用分块处理函数，得到 layer_output
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        # 将 layer_output 添加到 outputs 中
        outputs = (layer_output,) + outputs

        # 返回 outputs
        return outputs

    # 分块处理函数，接收 attention_output 作为参数
    def feed_forward_chunk(self, attention_output):
        # 使用 self.intermediate 处理 attention_output，得到 intermediate_output
        intermediate_output = self.intermediate(attention_output)
        # 使用 self.output 处理 intermediate_output 和 attention_output，得到 layer_output
        layer_output = self.output(intermediate_output, attention_output)
        # 返回 layer_output
        return layer_output


# 定义一个名为 MraEncoder 的类，继承自 nn.Module
class MraEncoder(nn.Module):
    # 初始化方法，接收一个 config 对象作为参数
    def __init__(self, config):
        super().__init__()
        # 将 config 属性设置为 self 的属性
        self.config = config
        # 创建一个包含多个 MraLayer 层的 ModuleList，层数由 config.num_hidden_layers 决定
        self.layer = nn.ModuleList([MraLayer(config) for _ in range(config.num_hidden_layers)])
        # 设置 gradient_checkpointing 属性为 False
        self.gradient_checkpointing = False

    # 前向传播方法，接收多个参数，其中 hidden_states 是必须的
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_hidden_states=False,
        return_dict=True,
    ):
        # 如果需要输出所有隐藏状态，则初始化 all_hidden_states 为一个空元组，否则为 None
        all_hidden_states = () if output_hidden_states else None

        # 遍历每个 MraLayer 层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出所有隐藏状态，并且当前处于训练阶段，则将 hidden_states 加入 all_hidden_states
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果启用了梯度检查点且当前处于训练阶段，则调用 _gradient_checkpointing_func
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                )
            else:
                # 否则，调用 layer_module 的前向传播方法，处理 hidden_states 和 attention_mask
                layer_outputs = layer_module(hidden_states, attention_mask)

            # 更新 hidden_states 为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]

        # 如果需要输出所有隐藏状态，则将最终的 hidden_states 加入 all_hidden_states
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果 return_dict 为 False，则返回隐藏状态和所有隐藏状态中的非空元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)
        
        # 否则，返回一个 BaseModelOutputWithCrossAttentions 对象，包含最终的隐藏状态和所有隐藏状态
        return BaseModelOutputWithCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )
    # 初始化方法，用于创建一个新的对象实例
    def __init__(self, config):
        # 调用父类（nn.Module）的初始化方法
        super().__init__()
        # 创建一个全连接层，输入和输出大小均为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 如果 config.hidden_act 是字符串，则从预定义的映射 ACT2FN 中获取对应的激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            # 否则，使用配置中的激活函数
            self.transform_act_fn = config.hidden_act
        # 创建一个 LayerNorm 层，用于规范化隐藏状态向量
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # 前向传播方法，定义了模型的数据流向
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 全连接层计算，将输入 hidden_states 映射到同一维度空间
        hidden_states = self.dense(hidden_states)
        # 应用激活函数，根据 self.transform_act_fn 定义的方式
        hidden_states = self.transform_act_fn(hidden_states)
        # LayerNorm 层对隐藏状态进行规范化处理
        hidden_states = self.LayerNorm(hidden_states)
        # 返回处理后的隐藏状态作为输出
        return hidden_states
# Copied from transformers.models.bert.modeling_bert.BertLMPredictionHead with Bert->Mra
# 定义了一个用于Masked Language Model预测头部的PyTorch模块。
class MraLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用MraPredictionHeadTransform对隐藏状态进行转换
        self.transform = MraPredictionHeadTransform(config)

        # 输出权重与输入的嵌入向量相同，但每个token有一个仅输出的偏置项。
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化一个全零的偏置项参数
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # 需要在两个变量之间建立链接，以便在调用resize_token_embeddings时正确调整偏置大小
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        # 对输入的隐藏状态进行转换
        hidden_states = self.transform(hidden_states)
        # 使用线性层进行预测
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOnlyMLMHead with Bert->Mra
# 定义了一个仅包含MLM预测头部的PyTorch模块。
class MraOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用MraLMPredictionHead来生成预测分数
        self.predictions = MraLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        # 对序列输出进行预测
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


# Copied from transformers.models.yoso.modeling_yoso.YosoPreTrainedModel with Yoso->Mra,yoso->mra
# MraPreTrainedModel是一个抽象类，用于处理权重初始化以及下载和加载预训练模型的简单接口。
class MraPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    
    # 指定配置类为MraConfig
    config_class = MraConfig
    # 指定基础模型前缀为"mra"
    base_model_prefix = "mra"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化线性层的权重，均值为0，标准差为self.config.initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                # 如果有偏置项，则将其初始化为零
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化嵌入层的权重，均值为0，标准差为self.config.initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                # 如果有padding_idx，则将其对应的权重初始化为零
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 将LayerNorm层的偏置项初始化为零，权重初始化为1.0
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


# MRA_START_DOCSTRING
# MRA_START_DOCSTRING用于定义一个多行字符串，介绍了MraPreTrainedModel的基本信息和用法。
"""
This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
behavior.
"""
    Parameters:
        config ([`MraConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
定义一个多层次的文档字符串，描述了模型输入的各种参数和返回的内容。

"""

@add_start_docstrings(
    "The bare MRA Model transformer outputting raw hidden-states without any specific head on top.",
    MRA_START_DOCSTRING,
)
class MraModel(MraPreTrainedModel):
    """
    MRA模型类，继承自MraPreTrainedModel，用于输出未经任何特定头部处理的原始隐藏状态。

    Args:
        config (MraConfig): 包含模型配置信息的配置对象。

    Attributes:
        config (MraConfig): 模型的配置信息对象。
        embeddings (MraEmbeddings): MRA模型的嵌入层。
        encoder (MraEncoder): MRA模型的编码器层。
    """

    def __init__(self, config):
        """
        初始化方法，设置模型的各个组件。

        Args:
            config (MraConfig): 包含模型配置信息的配置对象。
        """
        super().__init__(config)
        self.config = config

        # 初始化嵌入层和编码器层
        self.embeddings = MraEmbeddings(config)
        self.encoder = MraEncoder(config)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        """
        返回模型的嵌入层的词嵌入。

        Returns:
            torch.nn.Embedding: 返回模型的嵌入层的词嵌入。
        """
        return self.embeddings.word_embeddings
    # 设置模型输入的嵌入向量，用给定的值替换当前的词嵌入
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # 剪枝模型中的注意力头部
    # heads_to_prune: 要剪枝的注意力头部的字典 {层号: 要在此层剪枝的头部列表}，参见基类 PreTrainedModel
    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            # 获取指定层的注意力头部并执行剪枝操作
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 将添加的文档字符串（start_docstrings_to_model_forward 中的格式字符串）应用于模型前向方法
    # MRA_INPUTS_DOCSTRING.format("batch_size, sequence_length") 中的格式字符串是输入说明
    # 同时添加代码示例的文档字符串，包括 checkpoint、输出类型、配置类
    @add_start_docstrings_to_model_forward(MRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithCrossAttentions,
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
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 使用装饰器为模型类添加文档字符串，描述其作为具有“语言建模”头部的MRA模型
@add_start_docstrings("""MRA Model with a `language modeling` head on top.""", MRA_START_DOCSTRING)
# 定义一个继承自MraPreTrainedModel的MraForMaskedLM类
class MraForMaskedLM(MraPreTrainedModel):
    # 定义一个类变量，包含绑定权重的键名列表
    _tied_weights_keys = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]

    # 初始化方法，接受一个配置参数config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建一个MRA模型实例，根据给定的配置参数
        self.mra = MraModel(config)
        # 创建一个仅包含MLM头部的实例，根据给定的配置参数
        self.cls = MraOnlyMLMHead(config)

        # 执行后续初始化权重和应用最终处理
        self.post_init()

    # 返回MLM头部的预测解码器的权重
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置MLM头部的预测解码器的新嵌入
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 使用装饰器为forward方法添加文档字符串，描述其输入格式
    # 并添加代码示例的文档字符串，指定了用于文档的检查点、输出类型和配置类
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 输入参数说明结束
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        # 设置返回字典的选择，如果未提供则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用模型的 masked language modeling heads 处理输入
        outputs = self.mra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从模型输出中提取序列输出
        sequence_output = outputs[0]
        # 通过分类层获取预测分数
        prediction_scores = self.cls(sequence_output)

        # 初始化 masked language modeling 的损失为 None
        masked_lm_loss = None
        # 如果提供了 labels，则计算 masked language modeling 的损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # 交叉熵损失函数，用于计算损失
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # 如果 return_dict 为 False，则返回一个元组
        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 如果 return_dict 为 True，则返回一个 MaskedLMOutput 对象
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 从 transformers.models.yoso.modeling_yoso.YosoClassificationHead 复制并改名为 MraClassificationHead
class MraClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        # 定义一个全连接层，输入维度为 config.hidden_size，输出维度为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义一个 dropout 层，根据 config.hidden_dropout_prob 进行随机置零
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 定义一个全连接层，输入维度为 config.hidden_size，输出维度为 config.num_labels
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        self.config = config

    def forward(self, features, **kwargs):
        # 从 features 中取出第一个 token 的输出作为 x，相当于取出 [CLS] 的输出
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        # 对 x 应用 dropout
        x = self.dropout(x)
        # 将 x 输入到全连接层 dense 中
        x = self.dense(x)
        # 根据 config 中指定的激活函数 ACT2FN 进行激活
        x = ACT2FN[self.config.hidden_act](x)
        # 再次对 x 应用 dropout
        x = self.dropout(x)
        # 将 x 输入到全连接层 out_proj 中，得到最终的分类输出
        x = self.out_proj(x)
        return x


@add_start_docstrings(
    """MRA Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks.""",
    MRA_START_DOCSTRING,
)
# 定义 MraForSequenceClassification 类，继承自 MraPreTrainedModel
class MraForSequenceClassification(MraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 初始化 num_labels 属性
        self.num_labels = config.num_labels
        # 创建 MraModel 实例，并赋值给 self.mra
        self.mra = MraModel(config)
        # 创建 MraClassificationHead 实例，并赋值给 self.classifier
        self.classifier = MraClassificationHead(config)

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(MRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义 forward 方法，接受一系列输入参数并返回分类模型的输出
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 输入参数的注释说明
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 根据需要确定是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用模型的前向传播方法进行计算
        outputs = self.mra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从模型输出中获取序列输出
        sequence_output = outputs[0]
        # 将序列输出传递给分类器获取 logits
        logits = self.classifier(sequence_output)

        # 初始化损失为 None
        loss = None
        # 如果给定了标签
        if labels is not None:
            # 如果问题类型尚未确定，则根据情况自动设定
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型计算相应的损失
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    # 对于单一标签的回归问题，计算均方误差损失
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    # 对于多标签的回归问题，计算均方误差损失
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # 对于单标签分类问题，使用交叉熵损失函数
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # 对于多标签分类问题，使用带 logits 的二元交叉熵损失函数
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        
        # 如果不需要返回字典，则组织输出格式
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典，则构建 SequenceClassifierOutput 对象并返回
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用装饰器添加多项选择分类模型的文档字符串，描述其用途和结构
@add_start_docstrings(
    """MRA Model with a multiple choice classification head on top (a linear layer on top of
    the pooled output and a softmax) e.g. for RocStories/SWAG tasks.""",
    MRA_START_DOCSTRING,
)
# 定义多项选择分类的 MRA 模型类，继承自 MraPreTrainedModel
class MraForMultipleChoice(MraPreTrainedModel):
    
    # 初始化方法，接收一个配置对象 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        
        # 创建 MRA 模型对象
        self.mra = MraModel(config)
        # 创建预分类器，将隐藏状态映射到同样大小的隐藏状态空间
        self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建分类器，将隐藏状态映射到一个标量值（用于多项选择任务）
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    # 使用装饰器为 forward 方法添加输入文档字符串，描述输入参数的作用和形状
    @add_start_docstrings_to_model_forward(MRA_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    # 使用装饰器为 forward 方法添加代码示例的文档字符串，展示其用法和输出类型
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 前向传播方法定义，接收多个输入参数和可选的标签，返回模型输出或损失
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 函数参数列表未完，继续下一行
        ) -> Union[Tuple, MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 根据 return_dict 参数确定是否返回字典类型的结果
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 计算 num_choices，即选择题选项的数量
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 将 input_ids 重新视图化为二维张量，方便后续处理
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        # 将 attention_mask 重新视图化为二维张量，方便后续处理
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # 将 token_type_ids 重新视图化为二维张量，方便后续处理
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        # 将 position_ids 重新视图化为二维张量，方便后续处理
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        # 将 inputs_embeds 重新视图化为三维张量，方便后续处理
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 调用模型的前向传播函数 mra，获取输出结果
        outputs = self.mra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取隐藏状态输出
        hidden_state = outputs[0]  # (bs * num_choices, seq_len, dim)
        # 取每个样本序列的第一个位置的隐藏状态作为池化输出
        pooled_output = hidden_state[:, 0]  # (bs * num_choices, dim)
        # 将池化输出传递给预分类器进行处理
        pooled_output = self.pre_classifier(pooled_output)  # (bs * num_choices, dim)
        # 使用 ReLU 激活函数处理池化输出
        pooled_output = nn.ReLU()(pooled_output)  # (bs * num_choices, dim)
        # 使用分类器获取最终的分类 logits
        logits = self.classifier(pooled_output)

        # 将 logits 重塑为二维张量，以便计算损失
        reshaped_logits = logits.view(-1, num_choices)

        # 初始化损失为 None
        loss = None
        # 如果提供了 labels，则计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 如果 return_dict 参数为 False，则按元组形式返回结果
        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 参数为 True，则按 MultipleChoiceModelOutput 类返回结果
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用装饰器为模型类添加文档字符串，描述该模型在标记分类任务（如命名实体识别）上的作用
@add_start_docstrings(
    """MRA Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.""",
    MRA_START_DOCSTRING,
)
# 定义 MraForTokenClassification 类，继承自 MraPreTrainedModel
class MraForTokenClassification(MraPreTrainedModel):
    def __init__(self, config):
        # 调用父类的构造函数初始化模型
        super().__init__(config)
        # 设置类别数目
        self.num_labels = config.num_labels

        # 初始化 MRA 模型
        self.mra = MraModel(config)
        # 初始化 Dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 初始化分类器线性层
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并进行最终处理
        self.post_init()

    # 使用装饰器为 forward 方法添加文档字符串，描述其输入与输出
    @add_start_docstrings_to_model_forward(MRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 使用装饰器为 forward 方法添加代码示例文档字符串
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
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
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 根据 `return_dict` 是否为 None，确定是否使用配置中的返回字典设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用模型的前向传播方法，获取输出
        outputs = self.mra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取模型输出中的序列输出（通常是隐藏状态的最后一层）
        sequence_output = outputs[0]

        # 对序列输出应用 dropout
        sequence_output = self.dropout(sequence_output)
        
        # 将dropout后的序列输出传入分类器，得到分类器的 logits
        logits = self.classifier(sequence_output)

        # 初始化损失为 None
        loss = None
        # 如果提供了标签，则计算损失
        if labels is not None:
            # 使用交叉熵损失函数
            loss_fct = CrossEntropyLoss()
            
            # 只保留损失的活跃部分（根据 attention_mask）
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不要求返回字典，则返回元组形式的输出
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果要求返回字典，则返回 TokenClassifierOutput 对象
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 为 MRA 问题回答模型添加文档字符串，描述其包含在顶部的跨度分类头部，用于像 SQuAD 这样的抽取式问答任务
# (在隐藏状态输出之上的线性层，用于计算 `span start logits` 和 `span end logits`)。
@add_start_docstrings(
    """MRA Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).""",
    MRA_START_DOCSTRING,
)
class MraForQuestionAnswering(MraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 设定模型的标签数量为2（即开始和结束）
        config.num_labels = 2
        self.num_labels = config.num_labels

        # 初始化 MRA 模型和线性输出层
        self.mra = MraModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并进行最终处理
        self.post_init()

    # 添加模型前向传播函数的文档字符串，描述输入参数和返回类型
    @add_start_docstrings_to_model_forward(MRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 声明模型前向传播函数可能接受的参数及其类型
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
        # 初始化返回字典，如果未提供则使用模型配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用模型的前向传播方法，获取模型输出
        outputs = self.mra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从模型输出中获取序列输出
        sequence_output = outputs[0]

        # 将序列输出传入答案提取的输出层，获取起始位置和结束位置的预测概率
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # 如果提供了起始和结束位置标签，则计算损失函数
            # 处理多GPU情况，添加维度以匹配模型输出
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 忽略超出模型输入范围的起始和结束位置标签
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 使用交叉熵损失函数，计算起始和结束位置的损失
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        # 如果不需要返回字典，则返回包含损失和输出的元组
        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 返回带有损失、起始和结束位置预测、隐藏状态和注意力权重的输出对象
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```