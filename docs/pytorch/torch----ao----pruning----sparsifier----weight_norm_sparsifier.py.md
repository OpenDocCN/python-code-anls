# `.\pytorch\torch\ao\pruning\sparsifier\weight_norm_sparsifier.py`

```
# mypy: allow-untyped-defs
# 导入reduce函数和Callable、Optional、Tuple、Union类型提示
from functools import reduce
from typing import Callable, Optional, Tuple, Union

# 导入torch和torch.nn.functional中的F模块
import torch
import torch.nn.functional as F

# 导入BaseSparsifier类
from .base_sparsifier import BaseSparsifier
# 导入operator模块
import operator

# 声明WeightNormSparsifier类是公开的，可以被外部访问
__all__ = ["WeightNormSparsifier"]

# 定义函数_flat_idx_to_2d，将一维索引转换为二维索引
def _flat_idx_to_2d(idx, shape):
    # 计算行号
    rows = idx // shape[1]
    # 计算列号
    cols = idx % shape[1]
    return rows, cols

# 定义WeightNormSparsifier类，继承自BaseSparsifier类
class WeightNormSparsifier(BaseSparsifier):
    r"""Weight-Norm Sparsifier

    This sparsifier computes the norm of every sparse block and "zeroes-out" the
    ones with the lowest norm. The level of sparsity defines how many of the
    blocks is removed.

    This sparsifier is controlled by three variables:
    1. `sparsity_level` defines the number of *sparse blocks* that are zeroed-out
    2. `sparse_block_shape` defines the shape of the sparse blocks. Note that
        the sparse blocks originate at the zero-index of the tensor.
    3. `zeros_per_block` is the number of zeros that we are expecting in each
        sparse block. By default we assume that all elements within a block are
        zeroed-out. However, setting this variable sets the target number of
        zeros per block. The zeros within each block are chosen as the *smallest
        absolute values*.

    Args:

        sparsity_level: The target level of sparsity
        sparse_block_shape: The shape of a sparse block (see note below)
        zeros_per_block: Number of zeros in a sparse block
        norm: Norm to use. Could be either `int` or a callable.
            If `int`, only L1 and L2 are implemented.

    Note::
        The `sparse_block_shape` is tuple representing (block_ROWS, block_COLS),
        irrespective of what the rows / cols mean in the data tensor. That means,
        if you were to sparsify a weight tensor in the nn.Linear, which has a
        weight shape `(Cout, Cin)`, the `block_ROWS` would refer to the output
        channels, while the `block_COLS` would refer to the input channels.

    Note::
        All arguments to the WeightNormSparsifier constructor are "default"
        arguments and could be overriden by the configuration provided in the
        `prepare` step.
    """
    def __init__(self,
                 sparsity_level: float = 0.5,
                 sparse_block_shape: Tuple[int, int] = (1, 4),
                 zeros_per_block: Optional[int] = None,
                 norm: Optional[Union[Callable, int]] = None):
        # 初始化函数，设置稀疏性水平、稀疏块形状、每块零元素数和规范化函数
        if zeros_per_block is None:
            # 如果未指定每块零元素数，则根据稀疏块形状计算
            zeros_per_block = reduce(operator.mul, sparse_block_shape)
        # 构建默认参数字典
        defaults = {
            "sparsity_level": sparsity_level,
            "sparse_block_shape": sparse_block_shape,
            "zeros_per_block": zeros_per_block,
        }
        # 如果未指定规范化函数，则默认为 L2 范数
        if norm is None:
            norm = 2
        # 根据不同的规范化选项，设置规范化函数
        if callable(norm):
            self.norm_fn = norm
        elif norm == 1:
            self.norm_fn = lambda T: T.abs()
        elif norm == 2:
            self.norm_fn = lambda T: T * T
        else:
            # 抛出未实现的规范化选项异常
            raise NotImplementedError(f"L-{norm} is not yet implemented.")
        # 调用父类的初始化方法，传递默认参数字典
        super().__init__(defaults=defaults)

    def _scatter_fold_block_mask(self, output_shape, dim, indices, block_shape,
                                 mask=None, input_shape=None, device=None):
        r"""Creates patches of size `block_shape` after scattering the indices."""
        # 创建块状掩码，在散布索引后生成大小为 `block_shape` 的补丁
        if mask is None:
            # 如果未提供掩码，则要求输入形状不为空
            assert input_shape is not None
            mask = torch.ones(input_shape, device=device)
        # 在指定维度上，使用给定的索引，将值为 0 的点散布到掩码中
        mask.scatter_(dim=dim, index=indices, value=0)
        # 将掩码折叠成指定输出大小的数据块
        mask.data = F.fold(mask, output_size=output_shape, kernel_size=block_shape, stride=block_shape)
        # 返回生成的掩码
        return mask
    def _make_tensor_mask(self, data, input_shape, sparsity_level, sparse_block_shape, mask=None):
        r"""Creates a tensor-level mask.

        Tensor-level mask is described as a mask, where the granularity of sparsification of the
        smallest patch is the sparse_block_shape. That means, that for a given mask and a
        sparse_block_shape, the smallest "patch" of zeros/ones could be the sparse_block_shape.

        In this context, `sparsity_level` describes the fraction of sparse patches.
        """
        # 获取数据的高度和宽度
        h, w = data.shape[-2:]
        # 获取稀疏块的高度和宽度
        block_h, block_w = sparse_block_shape
        # 计算需要增加的行和列数，以确保能够被稀疏块整除
        dh = (block_h - h % block_h) % block_h
        dw = (block_w - w % block_w) % block_w

        # 如果未提供mask，则创建一个全为1的张量作为mask
        if mask is None:
            mask = torch.ones(h + dh, w + dw, device=data.device)

        # 处理极端情况：完全稀疏或者完全非稀疏
        if sparsity_level >= 1.0:
            # 将mask置为全0
            mask.data = torch.zeros_like(mask)
            return mask
        elif sparsity_level <= 0.0:
            # 将mask置为全1
            mask.data = torch.ones_like(mask)
            return mask

        # 计算每个稀疏块中的值的数量
        values_per_block = reduce(operator.mul, sparse_block_shape)
        if values_per_block > 1:
            # 对数据进行降采样
            data = F.avg_pool2d(
                data[None, None, :], kernel_size=sparse_block_shape, stride=sparse_block_shape, ceil_mode=True
            )
        # 将数据展平
        data = data.flatten()
        # 数据的长度即为块的数量
        num_blocks = len(data)

        # 将数据复制以扩展每个块内的值
        data = data.repeat(1, values_per_block, 1)

        # 计算稀疏程度所对应的阈值索引
        threshold_idx = int(round(sparsity_level * num_blocks))
        # 确保阈值索引在合理范围内
        threshold_idx = max(0, min(num_blocks - 1, threshold_idx))  # Sanity check
        # 获取排序后的索引
        _, sorted_idx = torch.topk(data, k=threshold_idx, dim=2, largest=False)

        # 临时重塑mask
        mask_reshape = mask.reshape(data.shape)  # data might be reshaped
        # 使用散射方法将排序后的块索引应用到mask上
        self._scatter_fold_block_mask(
            dim=2, output_shape=(h + dh, w + dw),
            indices=sorted_idx, block_shape=sparse_block_shape, mask=mask_reshape
        )
        # 将重塑后的mask还原成原始形状并保证其连续性
        mask.data = mask_reshape.squeeze().reshape(mask.shape)[:h, :w].contiguous()
        return mask
    def _make_block_mask(self, data, sparse_block_shape, zeros_per_block, mask=None):
        r"""Creates a block-level mask.

        Block-level mask is described as a mask, where the granularity of sparsification of the
        largest patch is the sparse_block_shape. That means that for a given mask and a
        sparse_block_shape, the sparsity is computed only within a patch of a size sparse_block_shape.

        In this context the `zeros_per_block` describes the number of zeroed-out elements within a patch.
        """
        # 获取数据的高度和宽度
        h, w = data.shape[-2:]
        # 获取稀疏块的高度和宽度
        block_h, block_w = sparse_block_shape
        # 计算需要填充的行数和列数，确保数据能被稀疏块的大小整除
        dh = (block_h - h % block_h) % block_h
        dw = (block_w - w % block_w) % block_w
        # 计算每个稀疏块中的元素数量
        values_per_block = reduce(operator.mul, sparse_block_shape)

        # 如果未提供 mask，则创建一个全为 1 的 mask
        if mask is None:
            mask = torch.ones((h + dh, w + dw), device=data.device)

        # 如果每个稀疏块的元素数等于要置零的元素数
        if values_per_block == zeros_per_block:
            # 所有元素应置零
            mask.data = torch.zeros_like(mask)
            return mask

        # 创建一个新的填充张量，与数据相似（以匹配块形状）
        padded_data = torch.ones(h + dh, w + dw, dtype=data.dtype, device=data.device)
        padded_data.fill_(torch.nan)
        padded_data[:h, :w] = data
        # 使用 F.unfold 函数展开数据，以匹配稀疏块的大小和步长
        unfolded_data = F.unfold(padded_data[None, None, :], kernel_size=sparse_block_shape, stride=sparse_block_shape)

        # 临时重塑 mask
        mask_reshape = mask.reshape(unfolded_data.shape)
        # 在展开的数据中找出每个稀疏块中要置零的元素的索引
        _, sorted_idx = torch.topk(unfolded_data, k=zeros_per_block, dim=1, largest=False)

        # 调用 _scatter_fold_block_mask 方法，将找到的置零索引应用到 mask 中
        self._scatter_fold_block_mask(
            dim=1, indices=sorted_idx, output_shape=padded_data.shape, block_shape=sparse_block_shape, mask=mask_reshape
        )

        # 将 mask 数据更新为重塑后的 mask，并确保其连续性
        mask.data = mask_reshape.squeeze().reshape(mask.shape).contiguous()
        return mask
    # 更新模块的稀疏掩码，用于对指定的张量名进行稀疏化处理
    def update_mask(self, module, tensor_name, sparsity_level, sparse_block_shape,
                    zeros_per_block, **kwargs):
        # 计算每个稀疏块中的数值数量
        values_per_block = reduce(operator.mul, sparse_block_shape)
        # 如果每个块中的零的数量大于块中的总元素数，则抛出数值错误异常
        if zeros_per_block > values_per_block:
            raise ValueError(
                "Number of zeros per block cannot be more than the total number of elements in that block."
            )
        # 如果每个块中的零的数量小于0，则抛出数值错误异常
        if zeros_per_block < 0:
            raise ValueError("Number of zeros per block should be positive.")

        # 获取模块参数化对象中指定张量名的掩码
        mask = getattr(module.parametrizations, tensor_name)[0].mask
        # 如果稀疏水平小于等于0或每个块中的零的数量等于0，则将掩码设为全1
        if sparsity_level <= 0 or zeros_per_block == 0:
            mask.data = torch.ones_like(mask)
        # 如果稀疏水平大于等于1.0并且每个块中的零的数量等于块中的总元素数，则将掩码设为全0
        elif sparsity_level >= 1.0 and (zeros_per_block == values_per_block):
            mask.data = torch.zeros_like(mask)
        else:
            # 计算权重张量的范数
            ww = self.norm_fn(getattr(module, tensor_name))
            # 创建张量级掩码，考虑稀疏水平和稀疏块形状
            tensor_mask = self._make_tensor_mask(
                data=ww, input_shape=ww.shape, sparsity_level=sparsity_level, sparse_block_shape=sparse_block_shape
            )
            # 如果每个块中的值数量不等于零的数量，则创建块级掩码，并与张量级掩码逻辑或
            if values_per_block != zeros_per_block:
                block_mask = self._make_block_mask(data=ww, sparse_block_shape=sparse_block_shape,
                                                   zeros_per_block=zeros_per_block)
                tensor_mask = torch.logical_or(tensor_mask, block_mask)
            # 将掩码数据赋给掩码张量
            mask.data = tensor_mask
```