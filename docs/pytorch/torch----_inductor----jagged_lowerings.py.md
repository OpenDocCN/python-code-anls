# `.\pytorch\torch\_inductor\jagged_lowerings.py`

```
# mypy: allow-untyped-defs
# 导入所需的类型
from typing import List, Optional, Tuple, Union

# 导入 sympy 符号计算库
import sympy

# 导入 PyTorch 库
import torch

# 从当前目录下的 ir 模块中导入 Pointwise 和 TensorBox 类
from .ir import Pointwise, TensorBox

# 从 lowering 模块中导入 fallback_handler, is_integer_type, register_lowering 函数
from .lowering import fallback_handler, is_integer_type, register_lowering

# 从 virtualized 模块中导入 ops 模块
from .virtualized import ops


# pyre-ignore[2,3]
# 定义函数 dense_idx_to_jagged_idx，将稠密索引转换为不规则索引
def dense_idx_to_jagged_idx(batch_idx, seq_idx, offsets_loader, jagged_len):
    # 使用 ops.indirect_indexing 函数计算起始索引
    begin_idx = ops.indirect_indexing(
        offsets_loader([batch_idx]),  # 加载偏移量并应用于 batch_idx
        jagged_len + 1,  # 加一是因为最后一个序列长度可能为零
    )
    # 加载偏移量并应用于 batch_idx + 1，获取结束索引
    end_idx = offsets_loader([batch_idx + 1])
    # 计算不规则索引
    jagged_idx = begin_idx + seq_idx
    return jagged_idx, end_idx


def get_inverse_offsets(
    offsets: TensorBox,
    jagged_len: Union[int, sympy.Expr],
    realize: bool = True,
) -> TensorBox:
    """
    Returns "inverse_offsets" - the inverse of the offsets array.
    offsets maps batch index (dense) to jagged index (i.e. offset into jagged tensor).
    inverse_offsets maps jagged index to batch index.

    e.g. for offsets [0, 3, 4, 9, 10] this will return
    inverse_offsets = [0, 0, 0, 1, 2, 2, 2, 2, 2, 3]

    For the given offsets, the computed inverse_offsets are cached
    on the first call and reused in the further calls.
    """

    if hasattr(offsets, "inverse_offsets"):
        # 如果已经计算过 inverse_offsets，则直接返回缓存的结果
        return offsets.inverse_offsets

    # 如果未计算过 inverse_offsets，则需要进行计算
    offsets.realize()  # 将 offsets 实例化以确保在全局内存中可用
    device: torch.device = offsets.get_device()  # 获取 offsets 的设备信息
    dtype: torch.dtype = offsets.get_dtype()  # 获取 offsets 的数据类型信息

    # pyre-ignore[2,3]
    def inner_fn(index):
        idx = index[0]
        # 使用 ops.bucketize 函数进行桶化操作
        bucket = ops.bucketize(
            values=ops.index_expr(idx, dtype),  # 使用 ops.index_expr 函数创建索引表达式
            offsets_name=offsets.get_name(),  # 获取 offsets 的名称
            offsets_size=offsets.get_size()[0],  # 获取 offsets 的尺寸信息
            indexing_dtype=dtype,  # 指定索引数据类型
            right=True,  # 指定桶化时是否右对齐
        )
        # ops.bucketize 返回的是基于 1 的桶索引，需要转换为基于 0 的索引
        return bucket - 1

    # 创建 Pointwise 实例，用于处理内部函数
    inverse_offsets = Pointwise.create(
        device=device,
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=[jagged_len],  # 指定处理范围为 jagged_len
    )

    if realize:
        # 如果 realize 为 True，则冻结节点以防止在下游内联
        inverse_offsets.realize()

    # 将计算得到的 inverse_offsets 缓存起来以便后续复用
    offsets.inverse_offsets = inverse_offsets  # type: ignore[attr-defined]

    return inverse_offsets


def jagged_idx_to_dense_idx(
    jagged_idx,  # pyre-ignore[2]
    inverse_offsets_loader,  # pyre-ignore[2]
    offsets_loader,  # pyre-ignore[2]
    batch_size: Union[int, sympy.Expr],
    max_seq_len: Union[int, sympy.Expr],
    offsets_dtype: torch.dtype,
) -> Tuple[sympy.Expr, sympy.Expr]:
    # 使用 ops 模块的 indirect_indexing 函数进行间接索引操作，将 jagged_idx 转换成 batch_idx
    batch_idx = ops.indirect_indexing(
        # 调用 inverse_offsets_loader 函数，处理 jagged_idx 并返回转换后的值
        inverse_offsets_loader([jagged_idx]),
        # 设置 batch_idx 的大小为 batch_size + 1
        batch_size + 1,
    )
    
    # 使用 offsets_loader 函数获取 batch_idx 对应的起始偏移量
    batch_start = offsets_loader([batch_idx])
    
    # 使用 ops 模块的 index_expr 函数，通过 offsets_dtype 从 jagged_idx 中减去 batch_start，得到 seq 序列
    seq = ops.index_expr(jagged_idx, offsets_dtype) - batch_start
    
    # 使用 ops 模块的 indirect_indexing 函数对 seq 序列进行间接索引操作，限制最大长度为 max_seq_len
    # check=False 是因为可能存在超过 max_seq_len 长度的序列
    seq_idx = ops.indirect_indexing(seq, max_seq_len, check=False)
    
    # 返回计算得到的 batch_idx 和 seq_idx
    return batch_idx, seq_idx
# 注册自定义操作函数 `_jagged_to_padded_dense_forward` 到 Torch 的运算函数注册表中
def register_jagged_ops():
    # 忽略 pyre 检查错误代码 56
    @register_lowering(torch.ops.aten._jagged_to_padded_dense_forward.default)
    # 定义 `_jagged_to_padded_dense_forward` 函数，将不规则张量转换为填充后的密集张量
    def _jagged_to_padded_dense_forward(
        jagged_values: TensorBox,
        jagged_offsets: List[TensorBox],
        max_lengths: List[int],  # 包含整数或符号整数的列表
        padding_value: float = 0.0,
    ) -> TensorBox:
        # 获取不规则值张量的设备和数据类型
        device = jagged_values.get_device()
        dtype = jagged_values.get_dtype()

        # 获取不规则值张量的大小
        jagged_values_size = jagged_values.get_size()

        # 仅处理单个不规则维度的常见情况
        if (
            len(jagged_offsets) != 1
            or device.type != "cuda"
            or device != jagged_offsets[0].get_device()
            or len(jagged_values_size) != 2
            or len(jagged_offsets[0].get_size()) != 1
            or len(max_lengths) != len(jagged_offsets)
            or not is_integer_type(jagged_offsets[0])
        ):
            # 如果不符合处理条件，则返回到回退处理函数
            return fallback_handler(
                torch.ops.aten._jagged_to_padded_dense_forward.default,
                add_to_fallback_set=False,
            )(
                jagged_values,
                jagged_offsets,
                max_lengths,
                padding_value,
            )

        # 获取偏移量张量
        offsets: TensorBox = jagged_offsets[0]
        offsets_len = offsets.get_size()[0]
        offsets_dtype = offsets.get_dtype()
        batch_size = offsets_len - 1
        max_seq_len = max_lengths[0]
        embedding_len = jagged_values_size[1]
        jagged_len = jagged_values_size[0]

        # 设置输出大小为 [batch_size, max_seq_len, embedding_len]
        output_size = [batch_size, max_seq_len, embedding_len]

        # 创建不规则值和偏移量的加载器
        values_loader = jagged_values.make_loader()
        offsets_loader = offsets.make_loader()

        # 忽略 pyre 检查错误代码 2, 3, 53
        def inner_fn(index):
            # 密集张量的索引: [B, N, D]
            batch_idx, seq_idx, emb_idx = index
            # 将密集索引转换为不规则索引
            jagged_idx, end_idx = dense_idx_to_jagged_idx(
                batch_idx=batch_idx,
                seq_idx=seq_idx,
                offsets_loader=offsets_loader,
                jagged_len=jagged_len,
            )
            # 返回经过掩码和填充处理的结果
            return ops.masked(
                ops.lt(
                    ops.index_expr(jagged_idx, offsets_dtype),
                    end_idx,
                ),
                lambda: values_loader([jagged_idx, emb_idx]),
                padding_value,
            )

        # 创建 Pointwise 对象来处理内部函数 `inner_fn`，返回结果张量
        return Pointwise.create(
            device=device,
            dtype=dtype,
            inner_fn=inner_fn,
            ranges=output_size,
        )

    # 定义 `_dense_to_jagged_forward_impl` 函数，用于密集到不规则张量的转换
    def _dense_to_jagged_forward_impl(
        fallback_op,  # 忽略 pyre 检查错误代码 2
        dense: TensorBox,
        jagged_offsets: List[TensorBox],
        jagged_len: Optional[int] = None,
    ) -> TensorBox:
        # 获取 dense 张量的设备信息
        device = dense.get_device()
        # 获取 dense 张量的数据类型信息
        dtype = dense.get_dtype()

        # 获取 dense 张量的大小信息
        dense_size = dense.get_size()

        # 只处理单个不规则维度的常见情况
        if (
            len(jagged_offsets) != 1
            or device.type != "cuda"
            or device != jagged_offsets[0].get_device()
            or len(jagged_offsets[0].get_size()) != 1
            or len(dense_size) != 3
            or jagged_len is None
            or not is_integer_type(jagged_offsets[0])
        ):
            # 返回使用 fallback_handler 处理后的结果
            return fallback_handler(fallback_op, add_to_fallback_set=False)(
                dense,
                jagged_offsets,
                jagged_len,
            )

        # 将 jagged_offsets 的第一个元素作为 offsets 张量
        offsets: TensorBox = jagged_offsets[0]
        # 获取 offsets 张量的数据类型信息
        offsets_dtype = offsets.get_dtype()
        # 获取 dense 张量的批量大小
        batch_size = dense_size[0]
        # 获取 dense 张量的最大序列长度
        max_seq_len = dense_size[1]
        # 获取 dense 张量的嵌入长度
        embedding_len = dense_size[-1]

        # 设置输出的大小为 [jagged_len, embedding_len]
        output_size = [jagged_len, embedding_len]

        # 创建 dense 张量的加载器
        dense_loader = dense.make_loader()
        # 创建 offsets 张量的加载器
        offsets_loader = offsets.make_loader()

        # 计算 offsets 的逆偏移量
        inverse_offsets = get_inverse_offsets(
            offsets=offsets,
            jagged_len=jagged_len,
        )
        # 创建 inverse_offsets 的加载器
        inverse_offsets_loader = inverse_offsets.make_loader()

        # 忽略 pyre 检查的错误类型[2,3,53]
        def inner_fn(index):
            # 计算稀疏张量的大小：[sum_B(N_B), D]
            jagged_idx, emb_idx = index
            # 将稀疏索引转换为密集索引
            batch_idx, seq_idx = jagged_idx_to_dense_idx(
                jagged_idx=jagged_idx,
                offsets_loader=offsets_loader,
                inverse_offsets_loader=inverse_offsets_loader,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                offsets_dtype=offsets_dtype,
            )
            # 返回经过掩码和截断后的结果
            return ops.masked(
                ops.lt(
                    ops.index_expr(seq_idx, offsets_dtype),
                    ops.index_expr(max_seq_len, offsets_dtype),
                ),
                lambda: dense_loader([batch_idx, seq_idx, emb_idx]),
                0.0,  # 对于超过 max_seq_len 的稀疏序列
            )

        # 创建 Pointwise 对象并返回
        return Pointwise.create(
            device=device,
            dtype=dtype,
            inner_fn=inner_fn,
            ranges=output_size,
        )

    # 忽略 pyre 检查的错误类型[56]
    @register_lowering(torch.ops.aten._padded_dense_to_jagged_forward)
    def _dense_to_jagged_forward(
        dense: TensorBox,
        jagged_offsets: List[TensorBox],
        jagged_len: Optional[int] = None,
    ) -> TensorBox:
        # 调用实现函数 _dense_to_jagged_forward_impl 处理 dense 到 jagged 的转换
        return _dense_to_jagged_forward_impl(
            fallback_op=torch.ops.aten._padded_dense_to_jagged_forward.default,
            dense=dense,
            jagged_offsets=jagged_offsets,
            jagged_len=jagged_len,
        )
```