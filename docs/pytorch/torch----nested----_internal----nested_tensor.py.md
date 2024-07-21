# `.\pytorch\torch\nested\_internal\nested_tensor.py`

```py
# mypy: allow-untyped-defs
from typing import Tuple  # 导入必要的类型提示

import torch  # 导入PyTorch库
from torch._C import DispatchKey, DispatchKeySet  # 从torch._C模块导入DispatchKey和DispatchKeySet
from torch._prims_common import is_expandable_to  # 从torch._prims_common模块导入is_expandable_to函数
from torch.fx.experimental.symbolic_shapes import has_free_symbols  # 从torch.fx.experimental.symbolic_shapes导入has_free_symbols函数
from torch.utils.weak import WeakTensorKeyDictionary  # 从torch.utils.weak导入WeakTensorKeyDictionary类
from typing import *  # noqa: F403 导入所有类型提示，忽略F403类型错误

_tensor_id_counter = 0  # 初始化张量计数器
_tensor_symint_registry = WeakTensorKeyDictionary()  # 创建一个弱引用字典用于存储张量符号整数的注册表


def get_tensor_symint(tensor, *, coeff=1):
    global _tensor_id_counter  # 使用全局变量_tensor_id_counter
    tensor_symint = _tensor_symint_registry.get(tensor)  # 获取注册表中张量对应的符号整数
    if tensor_symint is None:  # 如果未找到符号整数
        tensor_symint = torch._C._get_nested_int(_tensor_id_counter, coeff)  # 调用底层函数创建新的嵌套整数
        _tensor_id_counter += 1  # 计数器自增
        _tensor_symint_registry[tensor] = tensor_symint  # 将新创建的符号整数注册到字典中
    return tensor_symint  # 返回张量的符号整数


# SDPA metadata; max / min seqlens are needed for e.g. flash
def _get_sdpa_extreme_seqlen(func, tensor):
    return int(func(tensor).item())  # 返回通过函数func计算得到的张量的极值序列长度


def _store_val_in_tensor(val) -> torch.Tensor:
    # hack to get dynamic shapes support: store in a (val, 0) shaped tensor
    return torch.zeros(val, 0)  # 返回一个形状为(val, 0)的零张量，用于支持动态形状


def _load_val_from_tensor(t: torch.Tensor):
    return t.shape[0]  # 返回张量t的第一个维度的大小


class NestedTensor(torch.Tensor):
    _values: torch.Tensor  # type: ignore[assignment]  # 内部值张量，忽略类型检查
    _offsets: torch.Tensor  # 偏移张量
    _lengths: Optional[torch.Tensor]  # 可选长度张量
    # NOTE [ Nested ints for ragged sizes and strides ]
    #
    # Jagged layout tensors are tensors that represent a n-dim tensor with a
    # ragged dimension, but are backed by an (n-1)-dim tensor underneath, e.g.,
    # a jagged tensor with outer shape [B, x, D] is represented internally by a
    # tensor with shape [sum(x), D] where we introduce what we call a nested int
    # denoted as "x" here (but sometimes denoted with "*" to
    # represent the ragged dimension, and sum(x) represents the dim of the inner
    # tensor or equivalently the sum of all the sizes of the constituent
    # tensors' varying lengths.
    #
    # We also use nested ints to represent the strides of this tensor.
    # For example, a jagged tensor with shape [B, x, D] can be strided in two
    # ways: [xD, D, 1] and [x, 1, sum(x)], where xD represents x multiplied by D
    _size: Tuple[int, ...]  # 大小元组，描述张量的形状
    _strides: Tuple[int, ...]  # 步幅元组，描述张量的步幅
    # Indicates that the nth dimension is ragged
    _ragged_idx: int  # 表示不规则维度的索引
    _metadata_cache: Dict[str, Any]  # 元数据缓存字典

    @staticmethod
    def __new__(
        cls,
        values,
        offsets,
        *,
        lengths=None,
        **kwargs,
    ):
        # 创建一个 DispatchKeySet 对象，用于指定 NestedTensor 的分发键集合
        ks = DispatchKeySet(DispatchKey.NestedTensor)
        # 向 DispatchKeySet 对象添加 AutogradNestedTensor 分发键
        ks = ks.add(DispatchKey.AutogradNestedTensor)

        # 只支持不规则张量（jagged tensor）的情况
        # 断言确保 offsets 参数不为 None
        assert offsets is not None
        # 断言确保 offsets 的维度为 1
        assert offsets.ndim == 1
        # 断言确保 values 不是 NestedTensor 类型
        assert not isinstance(values, NestedTensor)
        # 断言确保 values 张量位于与 offsets 相同的设备上
        assert values.device == offsets.device

        # 查询缓存，获取与 offsets 或 lengths 相关联的符号整数（symint）
        # 如果不存在则创建一个新的符号整数
        ragged_source = offsets if lengths is None else lengths
        ragged_size = get_tensor_symint(ragged_source, coeff=1)
        # 获取 kwargs 中的 _ragged_idx 参数，如果不存在则默认为 1
        _ragged_idx = kwargs.get("_ragged_idx", 1)
        # 计算 B 的值，即 offsets 张量的长度减 1
        B = offsets.shape[0] - 1
        # 如果 lengths 不为 None，则确保 B 与 lengths 张量的长度相同
        if lengths is not None:
            assert B == lengths.shape[0]

        # 减去 1，将 _ragged_idx 转换到 values 张量的维度空间
        r = _ragged_idx - 1
        # 计算 _size，表示创建的张量的形状
        _size = (B, *values.shape[:r], ragged_size, *values.shape[r + 1 :])
        # 获取 values 张量的步幅信息
        stride = values.stride()
        # 计算 _strides，表示创建的张量的步幅信息
        _strides = (ragged_size * stride[r], *stride)

        # 使用 torch.Tensor._make_wrapper_subclass 方法创建一个子类封装器
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            _size,
            _strides,
            0,
            torch.contiguous_format,
            values.dtype,
            torch.jagged,
            values.device,
            False,
            kwargs.get("requires_grad", False),
            "sizes",
            False,
            True,  # dispatch_layout
            ks,
            # 不尝试基于非零大小计算存储空间
            storage_size=values.untyped_storage().size(),
        )
        # 设置 _ragged_idx 到新创建的张量对象中
        r._ragged_idx = _ragged_idx
        r._size = _size
        r._strides = _strides

        # 返回创建的张量对象
        return r

    def __init__(self, values, offsets, *, lengths=None, **kwargs):
        # 调用父类的初始化方法
        super().__init__()

        # 初始化对象的属性
        self._values = values
        self._offsets = offsets
        self._lengths = lengths

        # 初始化元数据缓存，用于存储惰性计算的属性
        self._metadata_cache = kwargs.get("_metadata_cache") or {}

        # 对 collapsed ragged dim（折叠的不规则维度）始终标记为动态
        torch._dynamo.maybe_mark_dynamic(self, self._ragged_idx)
        torch._dynamo.maybe_mark_dynamic(self._values, self._ragged_idx - 1)

        # 如果存在 max_seqlen 属性，则标记为动态
        max_seqlen_tensor = self._metadata_cache.get("max_seqlen", None)
        if max_seqlen_tensor is not None:
            torch._dynamo.mark_dynamic(max_seqlen_tensor, 0)
        # 如果存在 min_seqlen 属性，则标记为动态
        min_seqlen_tensor = self._metadata_cache.get("min_seqlen", None)
        if min_seqlen_tensor is not None:
            torch._dynamo.mark_dynamic(min_seqlen_tensor, 0)

    def values(self):
        # 调用 torch._nested_get_values 方法获取嵌套张量的值视图
        return torch._nested_get_values(self)  # type: ignore[attr-defined]

    def offsets(self):
        # 返回对象的 _offsets 属性
        return self._offsets

    def lengths(self):
        # 返回对象的 _lengths 属性
        return self._lengths

    # 私有访问函数，用于获取 min / max sequence length 属性
    # 故意不使用 @property 装饰器，因为它们在 PT2 中尚不兼容
    # 这些函数用于计算并缓存最大和最小序列长度。
    # TODO: 当 @properties 在 PT2 更好支持时重新审视这部分代码。我认为理想状态是为最小/最大序列长度
    # 提供公共 @properties，以便编译（包括设置器）。
    def _get_max_seqlen(self):
        max_seqlen_tensor = self._max_seqlen_tensor
        if max_seqlen_tensor is None:
            # 计算并缓存最大值
            max_val = _get_sdpa_extreme_seqlen(
                torch.max,
                self._offsets.diff() if self._lengths is None else self._lengths,
            )
            max_seqlen_tensor = _store_val_in_tensor(max_val)
            # 将计算结果存入元数据缓存
            self._metadata_cache["max_seqlen"] = max_seqlen_tensor
        return _load_val_from_tensor(max_seqlen_tensor)

    def _get_min_seqlen(self):
        min_seqlen_tensor = self._min_seqlen_tensor
        if min_seqlen_tensor is None:
            # 计算并缓存最小值
            min_val = _get_sdpa_extreme_seqlen(
                torch.min,
                self._offsets.diff() if self._lengths is None else self._lengths,
            )
            min_seqlen_tensor = _store_val_in_tensor(min_val)
            # 将计算结果存入元数据缓存
            self._metadata_cache["min_seqlen"] = min_seqlen_tensor
        return _load_val_from_tensor(min_seqlen_tensor)

    # 作为内部张量处理 flatten / unflatten 的私有访问器。这些必须是属性，以便与可追踪包装子类逻辑一起使用。
    # 如果不存在，则不计算/缓存。
    @property
    def _max_seqlen_tensor(self) -> Optional[torch.Tensor]:
        return self._metadata_cache.get("max_seqlen", None)

    @property
    def _min_seqlen_tensor(self) -> Optional[torch.Tensor]:
        return self._metadata_cache.get("min_seqlen", None)

    # 这些是旧的私有 @property 访问器，由于内部 BC 原因而保留。TODO: 删除这些！
    @property
    def _max_seqlen(self):
        return self._get_max_seqlen()

    @property
    def _min_seqlen(self):
        return self._get_min_seqlen()

    def __repr__(self):
        # 我们应该在 torch/_tensor_str.py 中实现此方法
        grad_fn_str = (
            f", requires_grad={self.requires_grad}" if self.requires_grad else ""
        )
        if self.grad_fn:
            grad_fn_str = f", grad_fn={self.grad_fn}"
        return f"NestedTensor(size={self._size}, offsets={self._offsets}{grad_fn_str}, contiguous={self._lengths is None})"

    def __reduce_ex__(self, proto):
        state = torch._utils._get_obj_state(self)

        # SymNodes are not serializable
        assert "_size" in state and "_strides" in state
        state = dict(state)
        del state["_size"]
        del state["_strides"]

        # TODO: Update this to handle the other inner tensors
        # 函数用于序列化对象。当前版本仅处理 _values 和 _offsets
        func = NestedTensor
        args = (self._values, self._offsets)
        return (torch._tensor._rebuild_from_type_v2, (func, type(self), args, state))
    # 定义一个静态方法，用于展开张量的结构
    def __tensor_flatten__(self):
        # 构建上下文信息，包括是否需要梯度和不规则索引
        ctx = {
            "requires_grad": self.requires_grad,
            "ragged_idx": self._ragged_idx,
        }
        # 内部张量的名称列表
        inner_tensors = ["_values", "_offsets"]
        # 如果存在长度信息，则加入内部张量列表
        if self._lengths is not None:
            inner_tensors.append("_lengths")
        # 如果存在最小序列长度张量，则加入内部张量列表
        if self._min_seqlen_tensor is not None:
            inner_tensors.append("_min_seqlen_tensor")
        # 如果存在最大序列长度张量，则加入内部张量列表
        if self._max_seqlen_tensor is not None:
            inner_tensors.append("_max_seqlen_tensor")
        # 返回内部张量列表和上下文信息
        return inner_tensors, ctx

    @staticmethod
    # 定义一个静态方法，用于根据给定的内部张量字典和元数据进行张量的重新组织
    def __tensor_unflatten__(inner_tensors: Dict, meta, outer_size, outer_stride):
        # 断言内部张量字典的长度在2到5之间
        assert len(inner_tensors) >= 2 and len(inner_tensors) <= 5
        # 获取内部张量中的值、偏移量、长度、最小序列长度张量和最大序列长度张量
        values = inner_tensors["_values"]
        offsets = inner_tensors["_offsets"]
        lengths = inner_tensors.get("_lengths", None)
        min_seqlen_tensor = inner_tensors.get("_min_seqlen_tensor", None)
        max_seqlen_tensor = inner_tensors.get("_max_seqlen_tensor", None)

        # 初始化元数据缓存字典
        metadata_cache = {}
        # 如果存在最小序列长度张量，则将其存入元数据缓存
        if min_seqlen_tensor is not None:
            metadata_cache["min_seqlen"] = min_seqlen_tensor
        # 如果存在最大序列长度张量，则将其存入元数据缓存
        if max_seqlen_tensor is not None:
            metadata_cache["max_seqlen"] = max_seqlen_tensor

        # 获取不规则索引
        ragged_idx = meta["ragged_idx"]

        # 检查偏移量或长度是否包含自由符号，或者值是否包含自由符号
        if has_free_symbols(offsets) or (lengths is not None and has_free_symbols(lengths)) or has_free_symbols(values):
            # 如果是这种情况，则关联偏移量或长度（可能是虚假的，可能被功能化了）与不规则大小
            ragged_source = offsets if lengths is None else lengths
            ragged_size = outer_size[ragged_idx]
            _tensor_symint_registry[ragged_source] = ragged_size

        # 返回重新组织后的NestedTensor对象
        return NestedTensor(
            values,
            offsets=offsets,
            lengths=lengths,
            requires_grad=meta["requires_grad"],
            _ragged_idx=ragged_idx,
            _metadata_cache=metadata_cache,
        )

    @classmethod
    # 定义一个类方法，用于处理Torch函数的分发
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # 如果kwargs为None，则初始化为空字典
        kwargs = {} if kwargs is None else kwargs

        # 懒加载以避免循环依赖
        from .ops import lookup_jagged

        # 查找具有不规则张量的函数并执行
        fn = lookup_jagged(func, *args, **kwargs)
        if fn is not None:
            return fn(*args, **kwargs)

        # 如果找不到对应的函数，则抛出NotImplementedError
        raise NotImplementedError(func)

    @classmethod
    # 定义一个类方法，用于处理Torch函数的功能化
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # 如果kwargs为None，则初始化为空字典
        if kwargs is None:
            kwargs = {}

        # 导入用于处理不规则张量的Torch函数
        from .ops import jagged_torch_function

        try:
            # 尝试执行不规则张量的Torch函数
            return jagged_torch_function(func, *args, **kwargs)
        except NotImplementedError:
            pass
        # 如果未实现对应的功能，则禁用Torch函数子类化并执行原始函数
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)
# NB: 这些虚构的视图 autograd.Function 已被真实的视图操作取代。不要使用它们！
# TODO: 一旦内部向后兼容期结束，移除 ViewBufferFromNested、ViewNestedFromBuffer 和 buffer_from_jagged。

# 实际上并不是视图！定义了一个不实际作为视图的类
class ViewBufferFromNested(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: NestedTensor):  # type: ignore[override]
        # 保存上下文，记录偏移量
        ctx.save_for_backward(x.offsets())
        ctx.metadata_cache = x._metadata_cache
        ctx.ragged_idx = x._ragged_idx
        # 返回 NestedTensor 中的值 _values
        return x._values

    @staticmethod
    def backward(ctx, gO: torch.Tensor):  # type: ignore[override]
        (offsets,) = ctx.saved_tensors
        # 返回一个新的 NestedTensor，传递梯度 gO 和其他保存的上下文信息
        return NestedTensor(
            gO,
            offsets=offsets,
            _metadata_cache=ctx.metadata_cache,
            _ragged_idx=ctx.ragged_idx,
        )


# 实际上并不是视图！定义了另一个不实际作为视图的类
class ViewNestedFromBuffer(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        values: torch.Tensor,
        offsets: torch.Tensor,
        metadata_cache: Optional[Dict[str, Any]] = None,
    ):  # type: ignore[override]
        # 与旧用法保持向后兼容，其中 seqlens 直接作为非张量/整数存储在元数据缓存中
        if metadata_cache is not None:
            min_seqlen = metadata_cache.get("min_seqlen", None)
            max_seqlen = metadata_cache.get("max_seqlen", None)
            if min_seqlen is not None and not isinstance(min_seqlen, torch.Tensor):
                # 如果 min_seqlen 不是张量，则将其存储为张量
                metadata_cache["min_seqlen"] = _store_val_in_tensor(min_seqlen)
            if max_seqlen is not None and not isinstance(max_seqlen, torch.Tensor):
                # 如果 max_seqlen 不是张量，则将其存储为张量
                metadata_cache["max_seqlen"] = _store_val_in_tensor(max_seqlen)
        # 返回一个新的 NestedTensor，传递 values（已分离的），offsets 和元数据缓存
        return NestedTensor(
            values.detach(),
            offsets=offsets,
            _metadata_cache=metadata_cache,
        )

    @staticmethod
    def backward(ctx, gO: NestedTensor):  # type: ignore[override]
        # 返回梯度 gO 中的 _values，不需要传递梯度到 offsets 和 metadata_cache
        return gO._values, None, None


# 从 jagged 结构创建 ViewBufferFromNested 的实例
def buffer_from_jagged(jagged):
    return ViewBufferFromNested.apply(jagged)


# 提示用户应当传入偏移量
def jagged_from_list(
    tensors: List[torch.Tensor],
    offsets: Optional[torch.Tensor],
    dtype=None,
    device=None,
) -> Tuple[NestedTensor, torch.Tensor]:
    """Constructs a NestedTensor backed by jagged layout from a list of tensors"""

    if not len(set(t.dtype for t in tensors)) == 1:  # noqa: C401
        # 如果列表中的张量不具有相同的数据类型，则引发错误
        raise RuntimeError(
            "When constructing a nested tensor, all tensors in list must have the same dtype"
        )
    if not len(set(t.device for t in tensors)) == 1:  # noqa: C401
        # 如果列表中的张量不在相同的设备上，则引发错误
        raise RuntimeError(
            "When constructing a nested tensor, all tensors in list must be on the same device"
        )

    # 检查是否可以用 jagged 布局表示 NestedTensor
    # Jagged 布局表示 (B, *, D_0, D_1, ..., D_N)，其中唯一的
    # 允许的不规则性仅限于紧邻批处理维度的单个维度。
    sizes = [t.shape for t in tensors]  # 获取每个张量的形状信息
    non_first_sizes = [s[1:] for s in sizes]  # 获取除了第一个维度外的其余维度信息
    at_most_first_ragged = all(s == non_first_sizes[0] for s in non_first_sizes)  # 检查除了第一个维度外的其余维度是否一致
    if not at_most_first_ragged:
        raise RuntimeError(
            "Cannot represent given tensor list as a nested tensor with the jagged layout. "
            "Note that the jagged layout only represents shapes of the form "
            "(B, *, D_0, D_1, ..., D_N), with only * allowed to be ragged."
        )

    # 适当设置属性。
    values = torch.cat(tensors, dim=0)  # 将张量列表连接成一个张量
    to_kwargs = {}
    if device is not None:
        to_kwargs["device"] = device  # 如果提供了设备参数，则设置到 to_kwargs 中
    if dtype is not None:
        to_kwargs["dtype"] = dtype  # 如果提供了数据类型参数，则设置到 to_kwargs 中
    values = values.to(**to_kwargs)  # 将张量 values 转换到指定的设备和数据类型

    # 如果未提供偏移量，则计算不规则布局的偏移量。
    if offsets is None:
        # 不规则布局要求将偏移量存储为 int64 类型，并且与 values 张量在相同设备上。
        # TODO: 另一种构建偏移量的方法是使用 F.pad。这样可以避免在前向过程中创建额外的叶子张量，可能解决兼容性问题。
        offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.int64, device=values.device),  # 创建一个设备为 values 设备的 int64 类型的零张量
                torch.tensor([s[0] for s in sizes], device=values.device).cumsum(dim=0),  # 计算累积和作为偏移量
            ]
        )

    # 现在计算这个，因为这很容易
    min_seqlen = min([t.shape[0] for t in tensors])  # 计算张量列表中最小的序列长度
    max_seqlen = max([t.shape[0] for t in tensors])  # 计算张量列表中最大的序列长度
    ret_nt = nested_view_from_values_offsets(
        values, offsets, min_seqlen=min_seqlen, max_seqlen=max_seqlen
    )
    return (ret_nt, offsets)  # type: ignore[return-value]
# 定义函数，接受三个参数：一个 PyTorch 张量 tensor，一个开始序列的张量 starts，一个序列长度的张量 lengths
# 返回一个 NestedTensor，以及一个张量 offsets 和一个可选的张量长度
def jagged_from_tensor_and_lengths(
    tensor: torch.Tensor, starts: torch.Tensor, lengths: torch.Tensor
) -> Tuple[NestedTensor, torch.Tensor, Optional[torch.Tensor]]:
    """Constructs a NestedTensor backed by jagged layout from a tensor, starts of sequences, and sequence lengths"""
    # 获取批次大小
    batch_size = tensor.shape[0]
    # 检查 starts 和 lengths 是否可以扩展到与输入 tensor 相同的批次大小
    if is_expandable_to(starts.shape, (batch_size,)) and is_expandable_to(
        lengths.shape, (batch_size,)
    ):
        # 将 starts 和 lengths 扩展到与输入 tensor 相同的批次大小
        start_list = starts.expand(batch_size)
        length_list = lengths.expand(batch_size)
    else:
        # 若不能扩展，则抛出运行时错误
        raise RuntimeError(
            "When constructing a jagged nested tensor using narrow(), "
            "your start and length must be Tensors that broadcast to input.shape[0]"
        )

    # 计算 jagged 偏移量
    assert (
        len(tensor.shape) >= 2
    ), "tensor must at least be 2D for the nested narrow op to work"
    # 获取序列的最大长度
    max_seq_len = tensor.shape[1]
    # 计算偏移长度，每个序列的偏移长度为序列最大长度的倍数
    offset_lengths = max_seq_len * torch.arange(
        0, batch_size, dtype=torch.int64, device=tensor.device
    )
    # 根据 jagged 布局要求，偏移量存储为与值张量相同设备上的 int64 类型
    # 拼接偏移量数组，包括起始偏移和最后一个偏移加上最后一个
    # 检查 `_dummy_instance` 是否为 None，如果是，则执行以下操作
    if _dummy_instance is None:
        # 创建一个 NestedTensor 实例 `_dummy_instance`
        _dummy_instance = NestedTensor(
            # 设置 values 属性为 3x3 的零张量，使用 "meta" 设备
            values=torch.zeros(3, 3, device="meta"),
            # 设置 offsets 属性为长度为 3 的零张量，使用 "meta" 设备和 torch.int64 类型
            offsets=torch.zeros(3, device="meta", dtype=torch.int64),
        ).detach()
    # 返回 `_dummy_instance` 实例
    return _dummy_instance
# 从给定的数据和偏移量创建嵌套视图，用于处理不规则的序列数据
def nested_view_from_values_offsets(
    values, offsets, ragged_idx=1, min_seqlen=None, max_seqlen=None
):
    # 如果指定了最小序列长度，则将其存储在张量中
    min_seqlen_tensor = None
    if min_seqlen is not None:
        min_seqlen_tensor = _store_val_in_tensor(min_seqlen)

    # 如果指定了最大序列长度，则将其存储在张量中
    max_seqlen_tensor = None
    if max_seqlen is not None:
        max_seqlen_tensor = _store_val_in_tensor(max_seqlen)

    # 调用 Torch 库中的函数创建嵌套视图，处理不规则的序列数据
    return torch._nested_view_from_jagged(  # type: ignore[attr-defined]
        values,
        offsets,
        _nt_view_dummy(),  # 提供一个占位符参数
        None,  # 长度参数为空，表示不考虑长度信息
        ragged_idx,  # 不规则序列的索引方式
        min_seqlen_tensor,  # 可能包含的最小序列长度
        max_seqlen_tensor,  # 可能包含的最大序列长度
    )  # type: ignore[return-value]


# 从给定的数据、偏移量和长度创建嵌套视图，用于处理不规则的序列数据
def nested_view_from_values_offsets_lengths(
    values, offsets, lengths, ragged_idx=1, min_seqlen=None, max_seqlen=None
):
    # 如果指定了最小序列长度，则将其存储在张量中
    min_seqlen_tensor = None
    if min_seqlen is not None:
        min_seqlen_tensor = _store_val_in_tensor(min_seqlen)

    # 如果指定了最大序列长度，则将其存储在张量中
    max_seqlen_tensor = None
    if max_seqlen is not None:
        max_seqlen_tensor = _store_val_in_tensor(max_seqlen)

    # 调用 Torch 库中的函数创建嵌套视图，处理不规则的序列数据
    return torch._nested_view_from_jagged(  # type: ignore[attr-defined]
        values,
        offsets,
        _nt_view_dummy(),  # 提供一个占位符参数
        lengths,  # 提供序列长度信息
        ragged_idx,  # 不规则序列的索引方式
        min_seqlen_tensor,  # 可能包含的最小序列长度
        max_seqlen_tensor,  # 可能包含的最大序列长度
    )  # type: ignore[return-value]
```