# `.\transformers\models\mra\modeling_mra.py`

```
# 设置文件编码为 utf-8
# 版权声明
# 按照 Apache 许可证 2.0 版在合规的情况下使用该文件
# 可以获取许可证的副本，网址为 http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按“原样”基础分发软件，没有任何担保或条件，无论是明示的还是默示的
# 请查看许可证以了解具体语言的限制和特定语言的权限
""" PyTorch MRA model."""

# 导入所需的模块
import math
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.cpp_extension import load

# 导入自定义的模块
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
from .configuration_mra import MraConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 以下是为了文档生成的变量
_CHECKPOINT_FOR_DOC = "uw-madison/mra-base-512-4"
_CONFIG_FOR_DOC = "MraConfig"
_TOKENIZER_FOR_DOC = "AutoTokenizer"

MRA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "uw-madison/mra-base-512-4",
    # 查看 https://huggingface.co/models?filter=mra 获取所有 Mra 模型
]


def load_cuda_kernels():
    global cuda_kernel
    src_folder = Path(__file__).resolve().parent.parent.parent / "kernels" / "mra"

    def append_root(files):
        return [src_folder / file for file in files]

    src_files = append_root(["cuda_kernel.cu", "cuda_launch.cu", "torch_extension.cpp"])

    # 加载 CUDA 内核
    cuda_kernel = load("cuda_kernel", src_files, verbose=True)

    import cuda_kernel


cuda_kernel = None

# 检查是否安装了所需的 CUDA 和 ninja（用于编译复杂的模块）
if is_torch_cuda_available() and is_ninja_available():
    logger.info("Loading custom CUDA kernels...")

    try:
        load_cuda_kernels()
    except Exception as e:
        logger.warning(
            "Failed to load CUDA kernels. Mra requires custom CUDA kernels. Please verify that compatible versions of"
            f" PyTorch and CUDA Toolkit are installed: {e}"
        )
else:
    pass


def sparse_max(sparse_qk_prod, indices, query_num_block, key_num_block):
    """
    Computes maximum values for softmax stability.
    """
    # 计算用于 softmax 稳定性的最大值
    if len(sparse_qk_prod.size()) != 4:
        raise ValueError("sparse_qk_prod must be a 4-dimensional tensor.")
    # 检查输入张量indices是否为二维张量
    if len(indices.size()) != 2:
        raise ValueError("indices must be a 2-dimensional tensor.")

    # 检查稀疏矩阵sparse_qk_prod的第二维大小是否为32
    if sparse_qk_prod.size(2) != 32:
        raise ValueError("The size of the second dimension of sparse_qk_prod must be 32.")

    # 检查稀疏矩阵sparse_qk_prod的第三维大小是否为32
    if sparse_qk_prod.size(3) != 32:
        raise ValueError("The size of the third dimension of sparse_qk_prod must be 32.")

    # 获取index_vals张量沿倒数第二维的最大值，并将维度进行转置
    index_vals = sparse_qk_prod.max(dim=-2).values.transpose(-1, -2)
    index_vals = index_vals.contiguous()

    # 将indices张量转换为整型张量，并保持连续内存布局
    indices = indices.int()
    indices = indices.contiguous()

    # 调用cuda_kernel中的index_max函数计算最大值和最大值的索引
    max_vals, max_vals_scatter = cuda_kernel.index_max(index_vals, indices, query_num_block, key_num_block)
    # 将max_vals_scatter张量的维度进行转置，然后增加一个维度
    max_vals_scatter = max_vals_scatter.transpose(-1, -2)[:, :, None, :]

    # 返回计算得到的最大值和最大值的索引
    return max_vals, max_vals_scatter
```  
# 将关注度掩码转换为用于高分辨率logits的稀疏掩码
def sparse_mask(mask, indices, block_size=32):
    # 检查掩码是否为2维张量
    if len(mask.size()) != 2:
        raise ValueError("mask must be a 2-dimensional tensor.")
    
    # 检查索引是否为2维张量
    if len(indices.size()) != 2:
        raise ValueError("indices must be a 2-dimensional tensor.")
    
    # 检查掩码和索引在零维上的大小是否一致
    if mask.shape[0] != indices.shape[0]:
        raise ValueError("mask and indices must have the same size in the zero-th dimension.")
    
    # 获取掩码的batch大小和序列长度
    batch_size, seq_len = mask.shape
    # 计算块的数量
    num_block = seq_len // block_size
    
    # 生成batch索引张量
    batch_idx = torch.arange(indices.size(0), dtype=torch.long, device=indices.device)
    # 重塑掩码为3维张量，按块大小拆分
    mask = mask.reshape(batch_size, num_block, block_size)
    # 使用索引和块索引，获取相应块的稀疏掩码
    mask = mask[batch_idx[:, None], (indices % num_block).long(), :]
    
    return mask


# 执行采样稠密矩阵乘法
def mm_to_sparse(dense_query, dense_key, indices, block_size=32):
    # 获取dense_query的batch大小，查询维度和视图维度
    batch_size, query_size, dim = dense_query.size()
    # 获取dense_key的batch大小，关键字维度和视图维度
    _, key_size, dim = dense_key.size()
    
    # 检查query_size是否能被块大小整除
    if query_size % block_size != 0:
        raise ValueError("query_size (size of first dimension of dense_query) must be divisible by block_size.")
    
    # 检查key_size是否能被块大小整除
    if key_size % block_size != 0:
        raise ValueError("key_size (size of first dimension of dense_key) must be divisible by block_size.")
    
    # 重塑dense_query和dense_key为4维张量，并交换后两个维度
    dense_query = dense_query.reshape(batch_size, query_size // block_size, block_size, dim).transpose(-1, -2)
    dense_key = dense_key.reshape(batch_size, key_size // block_size, block_size, dim).transpose(-1, -2)
    
    # 检查dense_query是否是4维张量
    if len(dense_query.size()) != 4:
        raise ValueError("dense_query must be a 4-dimensional tensor.")
    
    # 检查dense_key是否是4维张量
    if len(dense_key.size()) != 4:
        raise ValueError("dense_key must be a 4-dimensional tensor.")
    
    # 检查索引是否是2维张量
    if len(indices.size()) != 2:
        raise ValueError("indices must be a 2-dimensional tensor.")
    
    # 检查dense_query的第三维是否为32
    if dense_query.size(3) != 32:
        raise ValueError("The third dimension of dense_query must be 32.")
    
    # 检查dense_key的第三维是否为32
    if dense_key.size(3) != 32:
        raise ValueError("The third dimension of dense_key must be 32.")
    
    # 确保dense_query和dense_key的存储连续
    dense_query = dense_query.contiguous()
    dense_key = dense_key.contiguous()
    
    # 将索引转换为整数类型，并保证连续存储
    indices = indices.int()
    indices = indices.contiguous()
    
    # 返回计算后的稀疏矩阵结果
    return cuda_kernel.mm_to_sparse(dense_query, dense_key, indices.int())


# 执行稀疏矩阵与稠密矩阵的乘法
def sparse_dense_mm(sparse_query, indices, dense_key, query_num_block, block_size=32):
    # 获取dense_key的batch大小，关键字维度和视图维度
    batch_size, key_size, dim = dense_key.size()
    
    # 检查key_size是否能被块大小整除
    if key_size % block_size != 0:
        raise ValueError("key_size (size of first dimension of dense_key) must be divisible by block_size.")
    
    # 检查sparse_query的第二维是否等于块大小
    if sparse_query.size(2) != block_size:
        raise ValueError("The size of the second dimension of sparse_query must be equal to the block_size.")
    
    # 检查sparse_query的第三维是否等于块大小
    if sparse_query.size(3) != block_size:
        raise ValueError("The size of the third dimension of sparse_query must be equal to the block_size.")
    # 将密集键值重塑为指定形状，以便进行矩阵乘法操作
    dense_key = dense_key.reshape(batch_size, key_size // block_size, block_size, dim).transpose(-1, -2)
    
    # 检查稀疏查询的维度是否为4，如果不是则引发数值错误
    if len(sparse_query.size()) != 4:
        raise ValueError("sparse_query must be a 4-dimensional tensor.")
    
    # 检查密集键的维度是否为4，如果不是则引发数值错误
    if len(dense_key.size()) != 4:
        raise ValueError("dense_key must be a 4-dimensional tensor.")
    
    # 检查索引的维度是否为2，如果不是则引发数值错误
    if len(indices.size()) != 2:
        raise ValueError("indices must be a 2-dimensional tensor.")
    
    # 检查密集键的第三维度的大小是否为32，如果不是则引发数值错误
    if dense_key.size(3) != 32:
        raise ValueError("The size of the third dimension of dense_key must be 32.")
    
    # 使稀疏查询连续，以便于后续操作
    sparse_query = sparse_query.contiguous()
    
    # 将索引转换为整数类型，并使其连续
    indices = indices.int()
    indices = indices.contiguous()
    
    # 使密集键值连续，以便于后续操作
    dense_key = dense_key.contiguous()
    
    # 使用 CUDA 内核进行稀疏-密集矩阵乘法操作，得到密集的查询-键乘积
    dense_qk_prod = cuda_kernel.sparse_dense_mm(sparse_query, indices, dense_key, query_num_block)
    
    # 转置密集的查询-键乘积，并重塑形状
    dense_qk_prod = dense_qk_prod.transpose(-1, -2).reshape(batch_size, query_num_block * block_size, dim)
    
    # 返回得到的密集查询-键乘积
    return dense_qk_prod
# 将 1D 索引转换为 2D 索引的函数
def transpose_indices(indices, dim_1_block, dim_2_block):
    # 使用模运算获取行索引
    # 将行索引乘以列块大小得到新的行索引
    # 使用整数除法获取列索引
    # 将新的行索引和列索引组合成新的 2D 索引并转换为长整型
    return ((indices % dim_2_block) * dim_1_block + torch.div(indices, dim_2_block, rounding_mode="floor")).long()


# 定义一个自定义的 PyTorch 自动微分函数
class MraSampledDenseMatMul(torch.autograd.Function):
    # 定义正向传播操作
    @staticmethod
    def forward(ctx, dense_query, dense_key, indices, block_size):
        # 计算稀疏 query-key 乘积
        sparse_qk_prod = mm_to_sparse(dense_query, dense_key, indices, block_size)
        # 保存中间结果以用于反向传播
        ctx.save_for_backward(dense_query, dense_key, indices)
        ctx.block_size = block_size
        return sparse_qk_prod

    # 定义反向传播操作
    @staticmethod
    def backward(ctx, grad):
        # 从保存的中间结果中恢复变量
        dense_query, dense_key, indices = ctx.saved_tensors
        block_size = ctx.block_size
        # 计算 query 和 key 的块数
        query_num_block = dense_query.size(1) // block_size
        key_num_block = dense_key.size(1) // block_size
        # 转置索引
        indices_T = transpose_indices(indices, query_num_block, key_num_block)
        # 计算 key 的梯度
        grad_key = sparse_dense_mm(grad.transpose(-1, -2), indices_T, dense_query, key_num_block)
        # 计算 query 的梯度
        grad_query = sparse_dense_mm(grad, indices, dense_key, query_num_block)
        return grad_query, grad_key, None, None

    # 定义一个可调用的静态方法
    @staticmethod
    def operator_call(dense_query, dense_key, indices, block_size=32):
        return MraSampledDenseMatMul.apply(dense_query, dense_key, indices, block_size)


# 定义另一个自定义的 PyTorch 自动微分函数
class MraSparseDenseMatMul(torch.autograd.Function):
    # 定义正向传播操作
    @staticmethod
    def forward(ctx, sparse_query, indices, dense_key, query_num_block):
        # 计算稀疏 query-key 乘积
        sparse_qk_prod = sparse_dense_mm(sparse_query, indices, dense_key, query_num_block)
        # 保存中间结果以用于反向传播
        ctx.save_for_backward(sparse_query, indices, dense_key)
        ctx.query_num_block = query_num_block
        return sparse_qk_prod

    # 定义反向传播操作
    @staticmethod
    def backward(ctx, grad):
        # 从保存的中间结果中恢复变量
        sparse_query, indices, dense_key = ctx.saved_tensors
        query_num_block = ctx.query_num_block
        # 计算 key 的块数
        key_num_block = dense_key.size(1) // sparse_query.size(-1)
        # 转置索引
        indices_T = transpose_indices(indices, query_num_block, key_num_block)
        # 计算 key 的梯度
        grad_key = sparse_dense_mm(sparse_query.transpose(-1, -2), indices_T, grad, key_num_block)
        # 计算 query 的梯度
        grad_query = mm_to_sparse(grad, dense_key, indices)
        return grad_query, None, grad_key, None

    # 定义一个可调用的静态方法
    @staticmethod
    def operator_call(sparse_query, indices, dense_key, query_num_block):
        return MraSparseDenseMatMul.apply(sparse_query, indices, dense_key, query_num_block)


# 定义一个用于归约求和的类
class MraReduceSum:
    @staticmethod
    # 其他方法和实现省略
    # 定义一个函数用于操作稀疏查询和索引，生成输出结果
    def operator_call(sparse_query, indices, query_num_block, key_num_block):
        # 获取稀疏查询的尺寸信息
        batch_size, num_block, block_size, _ = sparse_query.size()

        # 检查稀疏查询张量的维度是否为4
        if len(sparse_query.size()) != 4:
            raise ValueError("sparse_query must be a 4-dimensional tensor.")

        # 检查索引张量的维度是否为2
        if len(indices.size()) != 2:
            raise ValueError("indices must be a 2-dimensional tensor.")

        # 重新获取稀疏查询的尺寸信息
        _, _, block_size, _ = sparse_query.size()
        batch_size, num_block = indices.size()

        # 对稀疏查询进行求和，并将其形状重新调整为(batch_size * num_block, block_size)
        sparse_query = sparse_query.sum(dim=2).reshape(batch_size * num_block, block_size)

        # 生成全局索引
        batch_idx = torch.arange(indices.size(0), dtype=torch.long, device=indices.device)
        global_idxes = (
            torch.div(indices, key_num_block, rounding_mode="floor").long() + batch_idx[:, None] * query_num_block
        ).reshape(batch_size * num_block)

        # 初始化一个与输出相关的临时张量
        temp = torch.zeros(
            (batch_size * query_num_block, block_size), dtype=sparse_query.dtype, device=sparse_query.device
        )

        # 使用index_add函数将稀疏查询的结果根据全局索引进行叠加
        output = temp.index_add(0, global_idxes, sparse_query).reshape(batch_size, query_num_block, block_size)

        # 重新调整输出的形状
        output = output.reshape(batch_size, query_num_block * block_size)
        # 返回操作后的输出
        return output
def get_low_resolution_logit(query, key, block_size, mask=None, value=None):
    """
    Compute low resolution approximation.
    """
    # 获取查询的批量大小、序列长度和注意力头维度
    batch_size, seq_len, head_dim = query.size()

    # 计算每行的块数量
    num_block_per_row = seq_len // block_size

    # 如果存在掩码
    if mask is not None:
        # 重新调整掩码的形状以便计算块内令牌数量
        token_count = mask.reshape(batch_size, num_block_per_row, block_size).sum(dim=-1)
        # 计算查询的块内均值
        query_hat = query.reshape(batch_size, num_block_per_row, block_size, head_dim).sum(dim=-2) / (
            token_count[:, :, None] + 1e-6
        )
        # 计算键的块内均值
        key_hat = key.reshape(batch_size, num_block_per_row, block_size, head_dim).sum(dim=-2) / (
            token_count[:, :, None] + 1e-6
        )
        # 如果存在值，则计算值的块内均值
        if value is not None:
            value_hat = value.reshape(batch_size, num_block_per_row, block_size, head_dim).sum(dim=-2) / (
                token_count[:, :, None] + 1e-6
            )
    else:
        # 没有掩码时，假设块内令牌数量为块大小
        token_count = block_size * torch.ones(batch_size, num_block_per_row, dtype=torch.float, device=query.device)
        # 计算查询的块内平均值
        query_hat = query.reshape(batch_size, num_block_per_row, block_size, head_dim).mean(dim=-2)
        # 计算键的块内平均值
        key_hat = key.reshape(batch_size, num_block_per_row, block_size, head_dim).mean(dim=-2)
        # 如果存在值，则计算值的块内平均值
        if value is not None:
            value_hat = value.reshape(batch_size, num_block_per_row, block_size, head_dim).mean(dim=-2)

    # 计算低分辨率对数
    low_resolution_logit = torch.matmul(query_hat, key_hat.transpose(-1, -2)) / math.sqrt(head_dim)

    # 计算每行最大值
    low_resolution_logit_row_max = low_resolution_logit.max(dim=-1, keepdims=True).values

    # 如果存在掩码，调整低分辨率对数以排除无效令牌
    if mask is not None:
        low_resolution_logit = (
            low_resolution_logit - 1e4 * ((token_count[:, None, :] * token_count[:, :, None]) < 0.5).float()
        )

    return low_resolution_logit, token_count, low_resolution_logit_row_max, value_hat


def get_block_idxes(
    low_resolution_logit, num_blocks, approx_mode, initial_prior_first_n_blocks, initial_prior_diagonal_n_blocks
):
    """
    Compute the indices of the subset of components to be used in the approximation.
    """
    batch_size, total_blocks_per_row, _ = low_resolution_logit.shape

    # 如果设置了初始对角线块数
    if initial_prior_diagonal_n_blocks > 0:
        # 计算对角线块的偏移量
        offset = initial_prior_diagonal_n_blocks // 2
        # 创建对角线掩码
        temp_mask = torch.ones(total_blocks_per_row, total_blocks_per_row, device=low_resolution_logit.device)
        diagonal_mask = torch.tril(torch.triu(temp_mask, diagonal=-offset), diagonal=offset)
        # 调整低分辨率对数以增加对角线块的值
        low_resolution_logit = low_resolution_logit + diagonal_mask[None, :, :] * 5e3

    # 如果设置了初始前n个块
    if initial_prior_first_n_blocks > 0:
        # 增加前n个块的值
        low_resolution_logit[:, :initial_prior_first_n_blocks, :] = (
            low_resolution_logit[:, :initial_prior_first_n_blocks, :] + 5e3
        )
        low_resolution_logit[:, :, :initial_prior_first_n_blocks] = (
            low_resolution_logit[:, :, :initial_prior_first_n_blocks] + 5e3
        )

    # 选择低分辨率对数中的前k个最大值
    top_k_vals = torch.topk(
        low_resolution_logit.reshape(batch_size, -1), num_blocks, dim=-1, largest=True, sorted=False
    )
    # 获取排名靠前的元素的索引
    indices = top_k_vals.indices

    # 如果近似模式是 "full"
    if approx_mode == "full":
        # 计算阈值，即排名最低的元素的值
        threshold = top_k_vals.values.min(dim=-1).values
        # 生成高分辨率掩码，对应低分辨率的逻辑值高于阈值的部分为1，否则为0
        high_resolution_mask = (low_resolution_logit >= threshold[:, None, None]).float()
    # 如果近似模式是 "sparse"
    elif approx_mode == "sparse":
        # 不生成高分辨率掩码，设为 None
        high_resolution_mask = None
    # 如果近似模式既不是 "full" 也不是 "sparse"
    else:
        # 抛出异常，提示近似模式值无效
        raise ValueError(f"{approx_mode} is not a valid approx_model value.")

    # 返回排名索引和高分辨率掩码
    return indices, high_resolution_mask
# 使用 Mra 近似自注意力机制
def mra2_attention(
    query, 
    key, 
    value, 
    mask, 
    num_blocks, 
    approx_mode, 
    block_size=32, 
    initial_prior_first_n_blocks=0, 
    initial_prior_diagonal_n_blocks=0,
):
    """
    Use Mra to approximate self-attention.
    """
    如果缺少 CUDA 内核，则返回与查询相同形状的零张量，并要求梯度
    if cuda_kernel is None:
        return torch.zeros_like(query).requires_grad_()

    获取查询张量的批次大小，头部数，序列长度和头部维度
    batch_size, num_head, seq_len, head_dim = query.size()
    计算元信息的批量大小
    meta_batch = batch_size * num_head

    检查序列长度是否可以被块大小整除
    if seq_len % block_size != 0:
        raise ValueError("sequence length must be divisible by the block_size.")

    计算每行的块数
    num_block_per_row = seq_len // block_size

    重塑查询、键和值张量的形状
    query = query.reshape(meta_batch, seq_len, head_dim)
    key = key.reshape(meta_batch, seq_len, head_dim)
    value = value.reshape(meta_batch, seq_len, head_dim)

    如果存在掩码，将查询、键和值张量乘以掩码
    if mask is not None:
        query = query * mask[:, :, None]
        key = key * mask[:, :, None]
        value = value * mask[:, :, None]

    如果近似模式为 "full"
    if approx_mode == "full":
        获取低分辨率对数、令牌计数、低分辨率最大行对数和估计值
        low_resolution_logit, token_count, low_resolution_logit_row_max, value_hat = get_low_resolution_logit(
            query, key, block_size, mask, value
        )
    如果近似模式为 "sparse"
    elif approx_mode == "sparse":
        使用无梯度环境下获取低分辨率对数、令牌计数、低分辨率最大行对数和估计值
        with torch.no_grad():
            low_resolution_logit, token_count, low_resolution_logit_row_max, _ = get_low_resolution_logit(
                query, key, block_size, mask
            )
    否则，抛出异常
    else:
        raise Exception('approx_mode must be "full" or "sparse"')

    使用无梯度环境下获取低分辨率对数的归一化值、块索引和高分辨率掩码
    with torch.no_grad():
        low_resolution_logit_normalized = low_resolution_logit - low_resolution_logit_row_max
        indices, high_resolution_mask = get_block_idxes(
            low_resolution_logit_normalized,
            num_blocks,
            approx_mode,
            initial_prior_first_n_blocks,
            initial_prior_diagonal_n_blocks,
        )

    通过 MraSampledDenseMatMul 运算符调用高分辨率对数乘法
    high_resolution_logit = MraSampledDenseMatMul.operator_call(
        query, key, indices, block_size=block_size
    ) / math.sqrt(head_dim)
    获取矩阵的最大值和最大值的散列
    max_vals, max_vals_scatter = sparse_max(high_resolution_logit, indices, num_block_per_row, num_block_per_row)
    减去最大值的散列，得到高分辨率对数
    high_resolution_logit = high_resolution_logit - max_vals_scatter
    如果存在掩码，将高分辨率对数减去 1e4 乘以掩码的补集
    if mask is not None:
        high_resolution_logit = high_resolution_logit - 1e4 * (1 - sparse_mask(mask, indices)[:, :, :, None])
    计算高分辨率注意力
    high_resolution_attn = torch.exp(high_resolution_logit)
    通过 MraSparseDenseMatMul 运算符调用高分辨率稀疏密集矩阵乘法
    high_resolution_attn_out = MraSparseDenseMatMul.operator_call(
        high_resolution_attn, indices, value, num_block_per_row
    )
    通过 MraReduceSum 运算符调用高分辨率归约和
    high_resolution_normalizer = MraReduceSum.operator_call(
        high_resolution_attn, indices, num_block_per_row, num_block_per_row
    )
    # 如果近似模式为“full”，则执行以下操作
    if approx_mode == "full":
        # 计算低分辨率的注意力分布
        low_resolution_attn = (
            torch.exp(low_resolution_logit - low_resolution_logit_row_max - 1e4 * high_resolution_mask)
            * token_count[:, None, :]
        )

        # 计算低分辨率的注意力输出
        low_resolution_attn_out = (
            torch.matmul(low_resolution_attn, value_hat)[:, :, None, :]
            .repeat(1, 1, block_size, 1)
            .reshape(meta_batch, seq_len, head_dim)
        )
        
        # 计算低分辨率的归一化因子
        low_resolution_normalizer = (
            low_resolution_attn.sum(dim=-1)[:, :, None].repeat(1, 1, block_size).reshape(meta_batch, seq_len)
        )

        # 计算对数校正
        log_correction = low_resolution_logit_row_max.repeat(1, 1, block_size).reshape(meta_batch, seq_len) - max_vals
        if mask is not None:
            log_correction = log_correction * mask

        # 计算低分辨率校正项
        low_resolution_corr = torch.exp(log_correction * (log_correction <= 0).float())
        low_resolution_attn_out = low_resolution_attn_out * low_resolution_corr[:, :, None]
        low_resolution_normalizer = low_resolution_normalizer * low_resolution_corr

        # 计算高分辨率校正项
        high_resolution_corr = torch.exp(-log_correction * (log_correction > 0).float())
        high_resolution_attn_out = high_resolution_attn_out * high_resolution_corr[:, :, None]
        high_resolution_normalizer = high_resolution_normalizer * high_resolution_corr

        # 计算上下文层
        context_layer = (high_resolution_attn_out + low_resolution_attn_out) / (
            high_resolution_normalizer[:, :, None] + low_resolution_normalizer[:, :, None] + 1e-6
        )

    # 如果近似模式为“sparse”，则执行以下操作
    elif approx_mode == "sparse":
        # 计算稀疏近似模式下的上下文层
        context_layer = high_resolution_attn_out / (high_resolution_normalizer[:, :, None] + 1e-6)
    # 如果近似模式不是“full”或“sparse”，则引发异常
    else:
        raise Exception('config.approx_mode must be "full" or "sparse"')

    # 如果存在掩码，则将掩码应用到上下文层
    if mask is not None:
        context_layer = context_layer * mask[:, :, None]

    # 重塑上下文层的形状
    context_layer = context_layer.reshape(batch_size, num_head, seq_len, head_dim)

    # 返回上下文层
    return context_layer
class MraEmbeddings(nn.Module):
    """为单词、位置和令牌类型嵌入构建嵌入层。"""

    def __init__(self, config):
        super().__init__()
        # 创建单词嵌入层，vocab_size 表示词汇量大小，hidden_size 表示隐藏状态大小，padding_idx 表示填充标记的索引
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 创建位置嵌入层，max_position_embeddings 表示最大位置嵌入大小
        self.position_embeddings = nn.Embedding(config.max_position_embeddings + 2, config.hidden_size)
        # 创建令牌类型嵌入层，type_vocab_size 表示令牌类型的数量
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # LayerNorm 命名不使用蛇形命名法，以保持与 TensorFlow 模型变量名称一致，并能够加载任何 TensorFlow 检查点文件
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) 在内存中是连续的，当序列化时会导出
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)) + 2)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
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
            position_ids = self.position_ids[:, :seq_length]

        # 将 token_type_ids 设置为构造函数中注册的缓冲区，其中所有元素都为零，通常在自动生成时出现，
        # 注册的缓冲区在跟踪模型时帮助用户不传递 token_type_ids，解决问题 #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MraSelfAttention(nn.Module):
    # 初始化函数，设置注意力头的数量和大小
    def __init__(self, config, position_embedding_type=None):
        # 继承父类初始化方法
        super().__init__()
        # 如果隐藏层大小不能被注意力头数量整除，并且配置没有嵌入大小属性，则抛出数值错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 设置注意力头数量和大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化查询、键和值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 初始化丢弃层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # 设置位置嵌入类型
        self.position_embedding_type = (
            position_embedding_type if position_embedding_type is not None else config.position_embedding_type
        )

        # 计算块数量
        self.num_block = (config.max_position_embeddings // 32) * config.block_per_row
        self.num_block = min(self.num_block, int((config.max_position_embeddings // 32) ** 2))

        # 设置近似模式、初始优先级第n块和初始优先级对角线n块
        self.approx_mode = config.approx_mode
        self.initial_prior_first_n_blocks = config.initial_prior_first_n_blocks
        self.initial_prior_diagonal_n_blocks = config.initial_prior_diagonal_n_blocks

    # 转置函数，为得分矩阵转置维度
    def transpose_for_scores(self, layer):
        new_layer_shape = layer.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        layer = layer.view(*new_layer_shape)
        return layer.permute(0, 2, 1, 3)
    # 定义前向传播方法，用于执行自注意力机制
    def forward(self, hidden_states, attention_mask=None):
        # 生成混合的查询向量
        mixed_query_layer = self.query(hidden_states)

        # 生成键向量并转置，以便进行注意力计算
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        # 生成值向量并转置，以便进行注意力计算
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        # 对混合的查询向量进行转置，以便进行注意力计算
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 获取批量大小、头数、序列长度和头维度
        batch_size, num_heads, seq_len, head_dim = query_layer.size()

        # 将注意力掩码还原为原始形状
        attention_mask = 1.0 + attention_mask / 10000.0
        attention_mask = (
            attention_mask.squeeze().repeat(1, num_heads, 1).reshape(batch_size * num_heads, seq_len).int()
        )

        # 对于头维度小于GPU线程束大小的情况，进行零填充
        gpu_warp_size = 32

        if head_dim < gpu_warp_size:
            # 计算零填充的尺寸
            pad_size = batch_size, num_heads, seq_len, gpu_warp_size - head_dim

            # 在查询向量、键向量和值向量上进行零填充
            query_layer = torch.cat([query_layer, torch.zeros(pad_size, device=query_layer.device)], dim=-1)
            key_layer = torch.cat([key_layer, torch.zeros(pad_size, device=key_layer.device)], dim=-1)
            value_layer = torch.cat([value_layer, torch.zeros(pad_size, device=value_layer.device)], dim=-1)

        # 执行多头自注意力计算
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

        # 如果头维度小于GPU线程束大小，则修剪上下文层以匹配头维度
        if head_dim < gpu_warp_size:
            context_layer = context_layer[:, :, :, :head_dim]

        # 重塑上下文层的形状以匹配原始形状
        context_layer = context_layer.reshape(batch_size, num_heads, seq_len, head_dim)

        # 对上下文层进行维度置换以满足输出格式要求
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # 返回上下文层作为输出
        outputs = (context_layer,)

        return outputs
# 从transformers.models.bert.modeling_bert.BertSelfOutput复制而来的类MraSelfOutput
class MraSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # 创建一个全连接层
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # 创建一个LayerNorm层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # 创建一个Dropout层

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)  # 全连接层的前向传播
        hidden_states = self.dropout(hidden_states)  # Dropout操作
        hidden_states = self.LayerNorm(hidden_states + input_tensor)  # LayerNorm和残差连接
        return hidden_states


# 创建一个MraAttention类
class MraAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = MraSelfAttention(config, position_embedding_type=position_embedding_type)  # 初始化self属性
        self.output = MraSelfOutput(config)  # 初始化output属性
        self.pruned_heads = set()  # 创建一个空集合用于存储被修剪的注意力头

    # 对注意力头进行修剪
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 对线性层进行修剪
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储被修剪的注意力头
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, attention_mask=None):
        self_outputs = self.self(hidden_states, attention_mask)  # self层的前向传播
        attention_output = self.output(self_outputs[0], hidden_states)  # 输出层的前向传播
        outputs = (attention_output,) + self_outputs[1:]  # 输出结果
        return outputs


# 从transformers.models.bert.modeling_bert.BertIntermediate复制而来的类MraIntermediate
class MraIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)  # 创建一个全连接层
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]  # 获取激活函数
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)  # 全连接层的前向传播
        hidden_states = self.intermediate_act_fn(hidden_states)  # 激活函数的处理
        return hidden_states
    

# 从transformers.models.bert.modeling_bert.BertOutput复制而来的类MraOutput
class MraOutput(nn.Module):
    pass  # 仅展示一个占位符，未定义具体内容
    # 初始化函数，接受配置参数并调用父类的初始化函数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建全连接层，输入特征数为config.intermediate_size，输出特征数为config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建LayerNorm层，对隐藏状态进行归一化处理，设置eps参数为config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建Dropout层，随机丢弃一部分神经元以防止过拟合，丢弃概率为config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    # 前向传播函数，接受隐藏状态和输入张量，返回更新后的隐藏状态
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用全连接层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 使用Dropout层对隐藏状态进行随机丢弃
        hidden_states = self.dropout(hidden_states)
        # 使用LayerNorm层对隐藏状态进行归一化处理，并加上输入张量后返回
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回更新后的隐藏状态
        return hidden_states
# 定义一个名为MraLayer的类，继承自nn.Module
class MraLayer(nn.Module):
    # 初始化方法，接受config作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 设置chunk_size_feed_forward属性为config中的chunk_size_feed_forward
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 设置seq_len_dim属性为1
        self.seq_len_dim = 1
        # 创建MraAttention对象并赋值给attention属性
        self.attention = MraAttention(config)
        # 设置add_cross_attention属性为config中的add_cross_attention
        self.add_cross_attention = config.add_cross_attention
        # 创建MraIntermediate对象并赋值给intermediate属性
        self.intermediate = MraIntermediate(config)
        # 创建MraOutput对象并赋值给output属性
        self.output = MraOutput(config)

    # 前向传播方法，接受hidden_states和attention_mask作为参数
    def forward(self, hidden_states, attention_mask=None):
        # 使用self.attention处理hidden_states和attention_mask，返回self_attention_outputs
        self_attention_outputs = self.attention(hidden_states, attention_mask)
        # 获取self_attention_outputs中的第一个元素，并赋值给attention_output
        attention_output = self_attention_outputs[0]

        # 如果需要输出注意力权重，则将self_attention_outputs中的除第一个元素外的所有元素赋值给outputs
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # 调用apply_chunking_to_forward方法对attention_output进行分块处理，并赋值给layer_output
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        # 将layer_output添加到outputs中
        outputs = (layer_output,) + outputs

        # 返回outputs
        return outputs

    # 定义feed_forward_chunk方法，接受attention_output作为参数
    def feed_forward_chunk(self, attention_output):
        # 使用self.intermediate处理attention_output，并赋值给intermediate_output
        intermediate_output = self.intermediate(attention_output)
        # 使用self.output处理intermediate_output和attention_output，并赋值给layer_output
        layer_output = self.output(intermediate_output, attention_output)
        # 返回layer_output
        return layer_output


# 定义一个名为MraEncoder的类，继承自nn.Module
class MraEncoder(nn.Module):
    # 初始化方法，接受config作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 设置config属性为传入的config
        self.config = config
        # 使用列表推导式创建包含config.num_hidden_layers个MraLayer对象的ModuleList，并赋值给layer属性
        self.layer = nn.ModuleList([MraLayer(config) for _ in range(config.num_hidden_layers)])
        # 设置gradient_checkpointing属性为False
        self.gradient_checkpointing = False

    # 前向传播方法，接受多个参数
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_hidden_states=False,
        return_dict=True,
    ):
        # 如果output_hidden_states为True，则初始化一个空元组赋值给all_hidden_states；否则赋值为None
        all_hidden_states = () if output_hidden_states else None

        # 遍历self.layer中的MraLayer对象
        for i, layer_module in enumerate(self.layer):
            # 如果output_hidden_states为True，则将hidden_states添加到all_hidden_states中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果gradient_checkpointing为True且处于训练状态，则调用_gradient_checkpointing_func方法
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                )
            else:
                # 否则调用layer_module的前向传播方法，并将结果赋值给layer_outputs
                layer_outputs = layer_module(hidden_states, attention_mask)

            # 将layer_outputs的第一个元素赋值给hidden_states
            hidden_states = layer_outputs[0]

        # 如果output_hidden_states为True，则将hidden_states添加到all_hidden_states中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果return_dict为False，则返回包含hidden_states和all_hidden_states中非None元素的元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)
        # 否则返回一个BaseModelOutputWithCrossAttentions对象
        return BaseModelOutputWithCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


# 从transformers.models.bert.modeling_bert.BertPredictionHeadTransform中复制MraPredictionHeadTransform类
class MraPredictionHeadTransform(nn.Module):
    # 初始化函数，用于创建新的对象实例，在此处用于初始化一个新的神经网络层对象
    def __init__(self, config):
        # 调用父类的初始化方法，确保继承自父类的属性也被正确初始化
        super().__init__()
        # 创建线性变换层，将输入特征的维度从 config.hidden_size 转换为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 检查 config.hidden_act 是否为字符串类型，如果是，则将其转换为对应的激活函数，否则直接使用给定的激活函数
        if isinstance(config.hidden_act, str):
            # 使用全局字典 ACT2FN 将激活函数名映射为对应的激活函数
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            # 如果 config.hidden_act 不是字符串类型，则直接使用给定的激活函数
            self.transform_act_fn = config.hidden_act
        # 创建 LayerNorm 层，对输入数据进行归一化处理
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # 前向传播函数，用于定义网络层的前向计算过程
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用线性变换层对输入数据进行线性变换
        hidden_states = self.dense(hidden_states)
        # 使用激活函数对线性变换后的结果进行非线性变换
        hidden_states = self.transform_act_fn(hidden_states)
        # 对非线性变换后的结果进行归一化处理
        hidden_states = self.LayerNorm(hidden_states)
        # 返回处理后的结果
        return hidden_states
# 从transformers.models.bert.modeling_bert.BertLMPredictionHead复制过来，并将Bert->Mra
class MraLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用MraPredictionHeadTransform对隐藏状态进行转换
        self.transform = MraPredictionHeadTransform(config)

        # 输出权重与输入嵌入相同，但每个标记都有一个仅输出的偏置
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 为每个标记添加一个偏置参数
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # 需要在这两个变量之间建立链接，以便在调用`resize_token_embeddings`时正确调整偏置
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        # 将隐藏状态进行转换
        hidden_states = self.transform(hidden_states)
        # 使用解码器对转换后的隐藏状态进行解码
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# 从transformers.models.bert.modeling_bert.BertOnlyMLMHead复制过来，并将Bert->Mra
class MraOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用MraLMPredictionHead作为预测器
        self.predictions = MraLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        # 预测分数
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


# 从transformers.models.yoso.modeling_yoso.YosoPreTrainedModel复制过来，并将Yoso->Mra, yoso->mra
class MraPreTrainedModel(PreTrainedModel):
    """
    一个抽象类，用于处理权重初始化和下载预训练模型的简单接口。
    """

    # 配置类为MraConfig
    config_class = MraConfig
    # 基本模型前缀为“mra”
    base_model_prefix = "mra"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            # 与TF版本略有不同，TF版本使用截断的正态分布进行初始化
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 归一化层的偏置初始化为零
            module.bias.data.zero_()
            # 归一化层的权重初始化为1
            module.weight.data.fill_(1.0)


# MRA_START_DOCSTRING字符串，用于说明模型的基本使用方法
MRA_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.
    Parameters:
        config ([`MraConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

MRA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare MRA Model transformer outputting raw hidden-states without any specific head on top.",
    MRA_START_DOCSTRING,
)
# 定义 MraModel 类，继承自 MraPreTrainedModel 类
class MraModel(MraPreTrainedModel):
    # 初始化方法，接受配置参数 config
    def __init__(self, config):
        # 调用父类构造方法
        super().__init__(config)
        # 将配置参数保存在实例属性中
        self.config = config

        # 初始化嵌入层和编码器
        self.embeddings = MraEmbeddings(config)
        self.encoder = MraEncoder(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入的方法
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings
    # 设置输入的嵌入向量
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # 剪枝模型的注意力头。heads_to_prune: {层编号: 需要剪枝的头列表} 参见基类 PreTrainedModel
    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 在模型前向传播中添加文档字符串和代码示例的装饰器
    @add_start_docstrings_to_model_forward(MRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,  # 参考文档的检查点
        output_type=BaseModelOutputWithCrossAttentions,  # 输出类型
        config_class=_CONFIG_FOR_DOC,  # 参考文档的配置类
    )
    # 模型的前向传播函数，包括多个输入参数和返回值
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
# 在 MRA 模型上添加一个基于语言建模的头
@add_start_docstrings("""MRA Model with a `language modeling` head on top.""", MRA_START_DOCSTRING)
class MraForMaskedLM(MraPreTrainedModel):
    # 定义共享的权重
    _tied_weights_keys = ["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"]

    # 初始化模型
    def __init__(self, config):
        super().__init__(config)

        # 初始化 MRA 模型和 MLM 头
        self.mra = MraModel(config)
        self.cls = MraOnlyMLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    # 正向传播函数
    @add_start_docstrings_to_model_forward(MRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MaskedLMOutput,
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
    # 该函数接收以下参数:
    # input_ids: 输入token序列的ID
    # attention_mask: 指示哪些位置需要attention的掩码
    # token_type_ids: 区分不同类型token的ID
    # position_ids: 位置ID
    # head_mask: 对Attention头的掩码
    # inputs_embeds: 输入的token嵌入
    # output_hidden_states: 是否输出所有隐藏状态
    # return_dict: 是否以字典形式返回输出
    # labels: 用于计算masked language modeling loss的标签, 值应在[-100, 0, ..., config.vocab_size]之间,
    #         -100表示忽略, 其他表示真实标签
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple, MaskedLMOutput]:
        # 如果 return_dict 参数为 None, 则使用配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # 通过多层注意力机制 (mra) 计算输出
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
    
        # 获取序列输出
        sequence_output = outputs[0]
    
        # 通过分类层 (cls) 计算预测分数
        prediction_scores = self.cls(sequence_output)
    
        # 如果提供了标签, 计算 masked language modeling loss
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    
        # 如果不使用返回字典, 返回预测分数和其他输出
        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
    
        # 如果使用返回字典, 返回 MaskedLMOutput 对象
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 从 transformers.models.yoso.modeling_yoso.YosoClassificationHead 复制并修改为 MRAClassificationHead
class MraClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        # 初始化一个全连接层，输入和输出维度都是隐藏大小
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 初始化一个 Dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 初始化一个全连接层，输入为隐藏大小，输出为标签数量
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        # 保存配置
        self.config = config

    def forward(self, features, **kwargs):
        # 取出特征中的第一个 token （等同于 [CLS]）
        x = features[:, 0, :]
        # 对输入进行 dropout 处理
        x = self.dropout(x)
        # 通过全连接层进行线性变换
        x = self.dense(x)
        # 使用激活函数对输出进行非线性变换
        x = ACT2FN[self.config.hidden_act](x)
        # 再次进行 dropout 处理
        x = self.dropout(x)
        # 通过输出层进行线性变换
        x = self.out_proj(x)
        # 返回结果
        return x


# 添加文档字符串，描述了 MRA 模型的序列分类/回归头部结构
@add_start_docstrings(
    """MRA Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks.""",
    MRA_START_DOCSTRING,
)
class MraForSequenceClassification(MraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 标签数量
        self.num_labels = config.num_labels
        # 初始化 MRA 模型
        self.mra = MraModel(config)
        # 初始化分类头部
        self.classifier = MraClassificationHead(config)

        # 初始化权重并进行最终处理
        self.post_init()

    # 添加模型前向传播方法的文档字符串
    @add_start_docstrings_to_model_forward(MRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 确保返回字典不为空，如果为空，则根据配置决定是否使用返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用多任务学习头部模型进行推理
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

        # 获取序列输出
        sequence_output = outputs[0]
        # 序列输出经过分类器得到预测结果
        logits = self.classifier(sequence_output)

        # 初始化损失
        loss = None
        # 如果有标签
        if labels is not None:
            # 如果问题类型未定义，则根据标签类型和类别数量进行自动判断问题类型
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
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        
        # 如果不需要返回字典，则返回元组形式的输出
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典，则构造 SequenceClassifierOutput 对象返回
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 添加多项选择分类头部的 MRA 模型（在池化输出的顶部添加一个线性层和 softmax），例如用于 RocStories/SWAG 任务
class MraForMultipleChoice(MraPreTrainedModel):
    def __init__(self, config):
        # 初始化 MRA 模型
        super().__init__(config)

        # 初始化 MRA 模型
        self.mra = MraModel(config)
        # 在隐藏状态上添加一个线性层作为预分类器
        self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)
        # 添加一个线性层作为分类器
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    # 重写父类的 forward 方法，接受多项选择任务的输入
    @add_start_docstrings_to_model_forward(MRA_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
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
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
```  
        ) -> Union[Tuple, MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 确保 return_dict 不为 None，如果为 None 则使用配置中的 use_return_dict 值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果 input_ids 不为 None，则返回第二个维度的大小作为 num_choices
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 如果 input_ids 不为 None，则将其形状调整为 (-1, input_ids.size(-1))
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        # 如果 attention_mask 不为 None，则将其形状调整为 (-1, attention_mask.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # 如果 token_type_ids 不为 None，则将其形状调整为 (-1, token_type_ids.size(-1))
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        # 如果 position_ids 不为 None，则将其形状调整为 (-1, position_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        # 如果 inputs_embeds 不为 None，则将其形状调整为 (-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 调用 mra 模型进行前向传播
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

        # 获取输出的隐藏状态
        hidden_state = outputs[0]  # (bs * num_choices, seq_len, dim)
        # 获取 pooled_output，将第一个 token 的隐藏状态作为 pooled_output
        pooled_output = hidden_state[:, 0]  # (bs * num_choices, dim)
        # 通过 pre_classifier 处理 pooled_output
        pooled_output = self.pre_classifier(pooled_output)  # (bs * num_choices, dim)
        # 使用 ReLU 激活函数
        pooled_output = nn.ReLU()(pooled_output)  # (bs * num_choices, dim)
        # 通过分类器获取 logits
        logits = self.classifier(pooled_output)

        # 将 logits 调整为 (-1, num_choices)
        reshaped_logits = logits.view(-1, num_choices)

        loss = None
        if labels is not None:
            # 如果有 labels，则计算交叉熵损失
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            # 如果 return_dict 为 False，则返回重新组织后的输出
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则返回 MultipleChoiceModelOutput 对象
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 添加文档字符串作为模型的起始注释，说明这个模型是基于 MRA 模型的一个用于 token classification 的模型，例如用于 NER 任务
# 导入 MRA_START_DOCSTRING，添加到起始文档字符串中
@add_start_docstrings(
    """MRA Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks.""",
    MRA_START_DOCSTRING,
)
# 定义 MraForTokenClassification 类，继承自 MraPreTrainedModel
class MraForTokenClassification(MraPreTrainedModel):
    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 获取标签数量
        self.num_labels = config.num_labels

        # 初始化 MRA 模型
        self.mra = MraModel(config)
        # 添加 dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 添加一个线性分类器
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 定义前向传播方法
    @add_start_docstrings_to_model_forward(MRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
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
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 确定是否返回字典形式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 MRA 模型进行前向传播
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

        # 获取模型输出的序列输出
        sequence_output = outputs[0]

        # 对序列输出进行 dropout 操作
        sequence_output = self.dropout(sequence_output)
        # 将 dropout 后的序列输出传入分类器中得到 logits
        logits = self.classifier(sequence_output)

        # 初始化损失值为 None
        loss = None
        # 如果有提供标签
        if labels is not None:
            # 定义交叉熵损失函数
            loss_fct = CrossEntropyLoss()
            # 只保留激活部分的损失
            if attention_mask is not None:
                # 获取激活部分的掩码
                active_loss = attention_mask.view(-1) == 1
                # 获取激活部分的 logits
                active_logits = logits.view(-1, self.num_labels)
                # 获取激活部分的标签
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                # 计算损失
                loss = loss_fct(active_logits, active_labels)
            else:
                # 计算全部数据的损失
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不要求返回字典形式的输出
        if not return_dict:
            # 按照返回格式返回结果
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回 TokenClassifierOutput 类型的对象，包含损失、logits、隐藏状态和注意力权重
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 基于 MRA 模型，添加一个用于抽取问答任务（如 SQuAD）的跨度分类头部（通过线性层在隐藏状态输出上计算“跨度起始对数”和“跨度结束对数”）进行初始化
@add_start_docstrings(
    """MRA Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).""",
    MRA_START_DOCSTRING,
)
class MraForQuestionAnswering(MraPreTrainedModel):
    def __init__(self, config):
        # 调用父类的构造函数
        super().__init__(config)

        # 将标签数目设置为2
        config.num_labels = 2
        self.num_labels = config.num_labels

        # 创建 MRA 模型对象
        self.mra = MraModel(config)
        # 创建线性层，其输入大小为隐藏层大小，输出大小为标签数目
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 将 MRA_MODEL_FORWARD_DOCSTRING 的注释添加到 model_forward 函数中
    @add_start_docstrings_to_model_forward(MRA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # 将代码示例的注释添加到模型前向传播的函数中
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 模型的前向传播函数，输入参数包括了 MRA_INPUTS_DOCSTRING 中指定的各种参数
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
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            用于标记开始标签的位置（索引）的标签，用于计算词汇分类损失。
            位置被夹紧到序列的长度(`sequence_length`)。序列外的位置不会用于计算损失。
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            用于标记结束标签的位置（索引）的标签，用于计算词汇分类损失。
            位置被夹紧到序列的长度(`sequence_length`)。序列外的位置不会用于计算损失。
            
        """
        根据需要是否返回字典，如果没有指定则使用配置中的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        调用MRA（注意力机制）模型，传入参数input_ids和其他可选参数，并返回结果outputs
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

        从输出中获取序列输出
        sequence_output = outputs[0]

        通过self.qa_outputs对序列输出进行分类，得到logits
        logits = self.qa_outputs(sequence_output)
        将logits按维度dim=-1拆分为start_logits和end_logits
        start_logits, end_logits = logits.split(1, dim=-1)
        去除多余维度，得到start_logits和end_logits
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        初始化总损失为None
        if start_positions is not None and end_positions is not None:
            如果需要计算损失
            如果在多GPU上，则添加维度
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            忽略超出模型输入范围的开始/结束位置
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            定义CrossEntropyLoss，忽略指定索引
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            计算开始位置和结束位置的损失
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            计算总损失
            total_loss = (start_loss + end_loss) / 2

        如果不返回字典
        if not return_dict:
            将输出构建为元组，包含总损失（如果有）和输出
            output = (start_logits, end_logits) + outputs[1:]
            如果总损失不为None，则返回总损失和输出，否则只返回输出
            return ((total_loss,) + output) if total_loss is not None else output

        如果指定返回字典，则构建QuestionAnsweringModelOutput对象返回
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```