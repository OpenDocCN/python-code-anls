# `.\pytorch\torch\sparse\_semi_structured_ops.py`

```
# 引入上下文管理器contextlib，用于简化管理上下文中的资源
import contextlib

# 引入torch库
import torch

# 定义公开的导出变量列表
__all__ = [
    "fallback_dispatcher",
    "semi_sparse_values",
    "semi_sparse_indices",
    "semi_sparse_t",
    "semi_sparse_view",
    "semi_sparse_detach",
    "semi_sparse_mm",
    "semi_sparse_addmm",
    "semi_sparse_linear",
]

# 定义上下文管理器，禁用Torch分发功能
@contextlib.contextmanager
def no_dispatch():
    guard = torch._C._DisableTorchDispatch()
    try:
        yield
    finally:
        del guard

# 提供一个函数，用于在不分发的情况下调用函数
def fallback_dispatcher(func, types, args, kwargs):
    with no_dispatch():
        return func(*args)

# 返回稀疏半结构化张量的值
def semi_sparse_values(func, types, args=(), kwargs=None) -> torch.Tensor:
    assert len(args) == 1
    A = args[0]
    assert isinstance(A, torch.sparse.SparseSemiStructuredTensor)
    assert A.packed is not None
    if A.meta is None:
        m, k = A.shape
        num_kept_elements = m * k // 2
        return A.packed[:num_kept_elements:].view(m, -1)
    else:
        return A.packed.detach()

# 返回稀疏半结构化张量的索引
def semi_sparse_indices(func, types, args=(), kwargs=None) -> torch.Tensor:
    assert len(args) == 1
    A = args[0]
    assert isinstance(A, torch.sparse.SparseSemiStructuredTensor)
    assert A.packed is not None
    if A.meta is None:
        m, k = A.shape
        num_kept_elements = m * k // 2
        metadata = A.packed[num_kept_elements:].view(m, -1)
        return metadata.view(torch.int32 if A.dtype == torch.int32 else torch.int16)
    else:
        return A.meta

# 返回稀疏半结构化张量的转置
def semi_sparse_t(func, types, args=(), kwargs=None) -> torch.Tensor:
    assert len(args) == 1
    self = args[0]
    assert isinstance(self, torch.sparse.SparseSemiStructuredTensor)
    assert len(self.shape) == 2
    # 由于当前不能从压缩表示转换回密集表示，因此仅跟踪转置的次数
    return self.__class__(
        torch.Size([self.shape[-1], self.shape[0]]),
        packed=self.packed_t,
        meta=self.meta_t,
        packed_t=self.packed,
        meta_t=self.meta,
        compressed_swizzled_bitmask=self.compressed_swizzled_bitmask.transpose(0, 1)
        if self.compressed_swizzled_bitmask is not None
        else None,
        fuse_transpose_cusparselt=args[0].fuse_transpose_cusparselt,
        alg_id_cusparselt=args[0].alg_id_cusparselt,
    )

# 返回稀疏半结构化张量的视图
def semi_sparse_view(func, types, args=(), kwargs=None) -> torch.Tensor:
    assert len(args) == 2
    self, shape = args
    if tuple(shape) != self.shape:
        raise NotImplementedError(
            f"`view` is not implemented for SparseSemiStructuredTensor, except for the dummy case (shape={shape})"
        )
    return self

# 返回稀疏半结构化张量的分离版本
def semi_sparse_detach(func, types, args, kwargs) -> torch.Tensor:
    assert len(args) == 1
    self = args[0]
    # 返回一个新对象，其类与当前对象相同，具有以下属性：
    # - shape: 保持与当前对象相同的形状
    # - packed: 保持与当前对象相同的打包状态
    # - meta: 保持与当前对象相同的元数据
    # - packed_t: 保持与当前对象相同的打包后的转置状态
    # - meta_t: 保持与当前对象相同的元数据的转置状态
    # - compressed_swizzled_bitmask: 保持与当前对象相同的压缩交错掩码
    # - requires_grad: 设置为False，表示新对象不需要梯度计算支持
    return self.__class__(
        shape=self.shape,
        packed=self.packed,
        meta=self.meta,
        packed_t=self.packed_t,
        meta_t=self.meta_t,
        compressed_swizzled_bitmask=self.compressed_swizzled_bitmask,
        requires_grad=False,
    )
# 定义一个函数 `semi_sparse_mm`，执行半稀疏矩阵乘法
def semi_sparse_mm(func, types, args=(), kwargs=None) -> torch.Tensor:
    # 断言参数 args 的长度为 2
    assert len(args) == 2
    # 分别取出 args 中的 A 和 B
    A, B = args
    # 如果 A 或 B 的维度不为 2，则抛出未实现错误
    if A.ndim != 2 or B.ndim != 2:
        raise NotImplementedError(
            "`SparseSemiStructuredTensor` matmul: Broadcasting is not implemented"
        )
    # 如果 A 是 torch.sparse.SparseSemiStructuredTensor 类型
    if isinstance(A, torch.sparse.SparseSemiStructuredTensor):
        # 获取 B 的行列数
        row, col = B.shape
        # 对 B 进行填充，使其与 A 的维度匹配
        B_padded = A._pad_dense_input(B)
        # 执行 A 的矩阵乘法操作
        res = A._mm(B_padded)
        # 返回结果的部分列
        return res[:, :col]
    else:
        # 若 A 不是稀疏半结构化张量，则对 B 进行转置操作
        B_t = B.t()
        # 断言 B_t 是 torch.sparse.SparseSemiStructuredTensor 类型
        assert isinstance(B_t, torch.sparse.SparseSemiStructuredTensor)
        # 获取 A 的行列数
        row, col = A.shape
        # 对 A 进行填充，使其与 B_t 的维度匹配
        A_padded = B._pad_dense_input(A)
        # 执行 B_t 的矩阵乘法操作，并转置结果
        res = B_t._mm(A_padded.t()).t()
        # 返回结果的部分行
        return res[:row, :]


# 定义一个函数 `semi_sparse_addmm`，执行半稀疏张量加权乘法
def semi_sparse_addmm(func, types, args=(), kwargs=None) -> torch.Tensor:
    # 断言参数 args 的长度为 3
    assert len(args) == 3
    # 分别取出 args 中的 bias, A 和 B
    bias, A, B = args
    # 如果 A 或 B 的维度不为 2，则抛出未实现错误
    if A.ndim != 2 or B.ndim != 2:
        raise NotImplementedError(
            "`SparseSemiStructuredTensor` matmul: Broadcasting is not implemented"
        )
    # 如果 bias 的维度不为 1，则抛出未实现错误，显示其形状
    if bias.ndim != 1:
        raise NotImplementedError(
            f"`SparseSemiStructuredTensor` matmul: only bias dim=1 supported. Shape={bias.shape}"
        )
    # 如果 A 是 torch.sparse.SparseSemiStructuredTensor 类型，则抛出未实现错误
    if isinstance(A, torch.sparse.SparseSemiStructuredTensor):
        raise NotImplementedError(
            "`SparseSemiStructuredTensor` matmul: only operand B of `addmm` can be sparse"
        )
    # 对 B 进行转置操作
    B_t = B.t()
    # 断言 B_t 是 torch.sparse.SparseSemiStructuredTensor 类型
    assert isinstance(B_t, torch.sparse.SparseSemiStructuredTensor)
    # 获取 A 的行列数
    row, col = A.shape
    # 对 A 进行填充，使其与 B_t 的维度匹配
    A_padded = B_t._pad_dense_input(A)
    # 执行 B_t 的加权乘法操作，并转置结果
    result = B_t._mm(A_padded.t(), bias=bias).t()
    # 返回结果的部分行
    return result[:row, :]


# 定义一个函数 `semi_sparse_linear`，执行半稀疏线性变换
def semi_sparse_linear(func, types, args=(), kwargs=None) -> torch.Tensor:
    # 断言参数 args 的长度为 2 或 3
    assert len(args) in [2, 3]
    # 取出 args 中的前两个参数 A 和 B，以及可能的第三个参数 bias
    A, B = args[:2]
    bias = args[2] if len(args) == 3 else None

    # 获取 A 的形状
    shape = A.shape
    # 将 A 展开成二维张量 A_2d
    A_2d = A.view(-1, shape[-1])

    # 如果没有提供 bias，则执行简单的矩阵乘法操作
    if bias is None:
        res = A_2d @ B.t()
    else:
        # 如果提供了 bias，则调用 semi_sparse_addmm 函数执行加权乘法
        res = semi_sparse_addmm(
            func=None,
            types=None,
            args=[bias, A_2d, B.t()],
        )

    # 将结果重新变形成原始形状并返回
    return res.view(*shape[:-1], -1)
```