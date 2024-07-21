# `.\pytorch\torch\masked\maskedtensor\core.py`

```py
# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates

# 导入警告模块
import warnings
# 导入类型相关的模块
from typing import Any
from typing_extensions import TypeGuard

# 导入PyTorch库
import torch
# 导入PyTorch的override函数
from torch.overrides import get_default_nowrap_functions

# 声明公开的模块变量
__all__ = [
    "MaskedTensor",
    "is_masked_tensor",
]

# 检查对象是否为MaskedTensor的函数
def is_masked_tensor(obj: Any, /) -> TypeGuard["MaskedTensor"]:
    r"""Returns True if the input is a MaskedTensor, else False

    Args:
        a: any input

    Examples:

        >>> # xdoctest: +SKIP
        >>> from torch.masked import MaskedTensor
        >>> data = torch.arange(6).reshape(2,3)
        >>> mask = torch.tensor([[True, False, False], [True, True, False]])
        >>> mt = MaskedTensor(data, mask)
        >>> is_masked_tensor(mt)
        True
    """
    return isinstance(obj, MaskedTensor)


# 比较两个张量是否匹配的函数
def _tensors_match(a, b, exact=True, rtol=1e-05, atol=1e-08):
    # 如果a或b是MaskedTensor，则引发值错误
    if is_masked_tensor(a) or is_masked_tensor(b):
        raise ValueError("Neither `a` nor `b` can be a MaskedTensor.")
    # 检查布局是否相同
    if a.layout != b.layout:
        raise ValueError(
            f"`a` and `b` must have the same layout. Got {a.layout} and {b.layout}"
        )

    # 如果数据类型不同，将b转换为a的数据类型
    if a.dtype != b.dtype:
        b = b.type(a.dtype)
    
    # 如果布局为稀疏COO格式
    if a.layout == b.layout == torch.sparse_coo:
        # 递归比较值张量和索引张量
        return _tensors_match(a.values(), b.values(), exact) and _tensors_match(
            a.indices(), b.indices(), exact
        )
    # 如果布局为稀疏CSR格式
    elif a.layout == b.layout == torch.sparse_csr:
        # 递归比较行索引、列索引和值张量
        return (
            _tensors_match(a.crow_indices(), b.crow_indices(), exact)
            and _tensors_match(a.col_indices(), b.col_indices(), exact)
            and _tensors_match(a.values(), b.values(), exact)
        )
    
    # 如果要求精确匹配
    if exact:
        # 检查维度是否相同，并且元素值是否完全相等
        return (a.dim() == b.dim()) and torch.eq(a, b).all().item()
    # 否则，使用allclose函数比较是否接近
    return (a.dim() == b.dim()) and torch.allclose(a, b, rtol=rtol, atol=atol)


# 比较两个掩码是否匹配的函数
def _masks_match(a, b):
    # 如果a和b都是MaskedTensor，则比较它们的掩码
    if is_masked_tensor(a) and is_masked_tensor(b):
        mask_a = a.get_mask()
        mask_b = b.get_mask()
        return _tensors_match(mask_a, mask_b, exact=True)
    # 否则返回True
    return True


# 将参数和关键字参数映射为实现函数接受的形式
def _map_mt_args_kwargs(args, kwargs, map_fn):
    # 内部辅助函数，将输入映射为实现函数接受的形式
    def _helper(a, map_fn):
        # 如果a是MaskedTensor，则应用映射函数
        if is_masked_tensor(a):
            return map_fn(a)
        # 如果a是张量，则直接返回
        elif torch.is_tensor(a):
            return a
        # 如果a是列表，则递归映射列表中的每个元素
        elif isinstance(a, list):
            a_impl, _ = _map_mt_args_kwargs(a, {}, map_fn)
            return a_impl
        # 如果a是元组，则递归映射元组中的每个元素
        elif isinstance(a, tuple):
            a_impl, _ = _map_mt_args_kwargs(a, {}, map_fn)
            return tuple(a_impl)
        # 否则直接返回a
        else:
            return a

    # 如果kwargs为None，则初始化为空字典
    if kwargs is None:
        kwargs = {}
    # 初始化实现参数列表
    impl_args = []
    # 对于args中的每个元素，应用_helper函数进行映射
    for a in args:
        impl_args.append(_helper(a, map_fn))
    # 初始化实现关键字参数字典
    impl_kwargs = {}
    # 对于kwargs的每个键值对，应用_helper函数进行映射
    for k in kwargs.keys():
        impl_kwargs[k] = _helper(a, map_fn)
    # 返回映射后的实现参数列表和实现关键字参数字典
    return impl_args, impl_kwargs


# 包装结果数据和掩码数据的函数
def _wrap_result(result_data, result_mask):
    # 如果结果数据是列表，则对每一对结果数据和掩码数据进行递归包装
    if isinstance(result_data, list):
        return [_wrap_result(r, m) for (r, m) in zip(result_data, result_mask)]
    # 如果 result_data 是元组类型，则对每个元组中的数据和对应的掩码执行 _wrap_result 函数，并返回新的元组
    if isinstance(result_data, tuple):
        return tuple(_wrap_result(r, m) for (r, m) in zip(result_data, result_mask))
    # 如果 result_data 是 PyTorch 的张量类型，则创建一个 MaskedTensor 对象，并传入 result_data 和 result_mask
    if torch.is_tensor(result_data):
        return MaskedTensor(result_data, result_mask)
    # 如果 result_data 和 result_mask 不是预期的张量类型，则返回 NotImplemented，表示该操作未实现
    # 期望 result_data 和 result_mask 只能是张量类型
    return NotImplemented
def _masked_tensor_str(data, mask, formatter):
    # 检查数据布局是否为稀疏的 COO 或 CSR，如果是则转换为稠密格式
    if data.layout in {torch.sparse_coo, torch.sparse_csr}:
        data = data.to_dense()
        mask = mask.to_dense()
    # 如果数据是一维的，格式化每个元素并计算最大长度以对齐显示
    if data.dim() == 1:
        formatted_elements = [
            formatter.format(d.item()) if isinstance(d.item(), float) else str(d.item())
            for d in data
        ]
        # 计算格式化后的元素的最大长度，考虑遮罩（mask）的影响
        max_len = max(8 if x[1] else len(x[0]) for x in zip(formatted_elements, ~mask))
        # 构建格式化后的字符串表示，考虑遮罩的显示
        return (
            "["
            + ", ".join(
                [
                    "--".rjust(max_len) if m else e
                    for (e, m) in zip(formatted_elements, ~mask)
                ]
            )
            + "]"
        )
    # 对多维数据递归调用本身，并进行缩进处理
    sub_strings = [_masked_tensor_str(d, m, formatter) for (d, m) in zip(data, mask)]
    sub_strings = ["\n".join(["  " + si for si in s.split("\n")]) for s in sub_strings]
    # 返回格式化后的多维数据字符串表示
    return "[\n" + ",\n".join(sub_strings) + "\n]"


def _get_data(a):
    # 如果输入是 MaskedTensor，则返回其数据部分
    if is_masked_tensor(a):
        return a._masked_data
    # 否则直接返回输入
    return a


def _maybe_get_mask(a):
    # 如果输入是 MaskedTensor，则返回其遮罩（mask）
    if is_masked_tensor(a):
        return a.get_mask()
    # 否则返回 None
    return None


class MaskedTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data, mask, requires_grad=False):
        # 确保数据和遮罩（mask）都是 Tensor 类型
        if is_masked_tensor(data) or not torch.is_tensor(data):
            raise TypeError("data must be a Tensor")
        if is_masked_tensor(mask) or not torch.is_tensor(mask):
            raise TypeError("mask must be a Tensor")
        # 使用给定参数创建 Tensor 的包装器子类
        kwargs = {
            "device": data.device,
            "dtype": data.dtype,
            "layout": data.layout,
            "requires_grad": requires_grad,
            "dispatch_sizes_strides_policy": "strides",
            "dispatch_layout": True,
        }
        # 发出警告，指出 MaskedTensors 的 PyTorch API 处于原型阶段并可能会变化
        warnings.warn(
            (
                "The PyTorch API of MaskedTensors is in prototype stage "
                "and will change in the near future. Please open a Github issue "
                "for features requests and see our documentation on the torch.masked "
                "module for further information about the project."
            ),
            UserWarning,
            stacklevel=2,
        )
        # 如果数据需要梯度计算，则发出警告建议使用非梯度计算的克隆版本
        if data.requires_grad:
            warnings.warn(
                "It is not recommended to create a MaskedTensor with a tensor that requires_grad. "
                "To avoid this, you can use data.clone().detach()",
                UserWarning,
                stacklevel=2,
            )
        # 创建并返回 Tensor 的包装器子类
        return torch.Tensor._make_wrapper_subclass(cls, data.size(), **kwargs)  # type: ignore[attr-defined]
    def _preprocess_data(self, data, mask):
        # 导入稀疏 COO 和 CSR 操作函数
        from .._ops import _sparse_coo_where, _sparse_csr_where

        # 检查数据和掩码的布局是否一致
        if data.layout != mask.layout:
            raise TypeError("data and mask must have the same layout.")
        
        # 处理稀疏 COO 布局的情况
        if data.layout == torch.sparse_coo:
            # 稀疏 COO 格式的数据需要合并
            data = data.coalesce()
            mask = mask.coalesce()
            # 如果非零元素数目不一致，则调用稀疏 COO 操作函数
            if data._nnz() != mask._nnz():
                data = _sparse_coo_where(mask, data, torch.tensor(0))
        
        # 处理稀疏 CSR 布局的情况
        elif data.layout == torch.sparse_csr:
            # 如果非零元素数目不一致，则调用稀疏 CSR 操作函数
            if data._nnz() != mask._nnz():
                data = _sparse_csr_where(mask, data, torch.tensor(0))

        # 使用深拷贝保存预处理后的数据和掩码
        self._masked_data = data.clone()
        self._masked_mask = mask.clone()

    def _validate_members(self):
        # 获取已处理后的数据和掩码
        data = self._masked_data
        mask = self.get_mask()
        
        # 检查数据和掩码是否具有相同的类型
        if type(data) != type(mask):
            raise TypeError(
                f"data and mask must have the same type. Got {type(data)} and {type(mask)}"
            )
        
        # 检查数据布局是否受支持
        if data.layout not in {torch.strided, torch.sparse_coo, torch.sparse_csr}:
            raise TypeError(f"data layout of {data.layout} is not supported.")
        
        # 如果数据布局为稀疏 COO
        if data.layout == torch.sparse_coo:
            # 检查稀疏 COO 数据和掩码是否具有相同的索引
            if not _tensors_match(data.indices(), mask.indices(), exact=True):
                raise ValueError(
                    "data and mask are both sparse COO tensors but do not have the same indices."
                )
        
        # 如果数据布局为稀疏 CSR
        elif data.layout == torch.sparse_csr:
            # 检查稀疏 CSR 数据和掩码是否具有相同的行和列索引
            if not _tensors_match(
                data.crow_indices(), mask.crow_indices(), exact=True
            ) or not _tensors_match(data.col_indices(), mask.col_indices(), exact=True):
                raise ValueError(
                    "data and mask are both sparse CSR tensors but do not share either crow or col indices."
                )
        
        # 检查掩码的数据类型是否为布尔型
        if mask.dtype != torch.bool:
            raise TypeError("mask must have dtype bool.")
        
        # 检查数据的数据类型是否为支持的类型之一
        if not (
            data.dtype == torch.float16
            or data.dtype == torch.float32
            or data.dtype == torch.float64
            or data.dtype == torch.bool
            or data.dtype == torch.int8
            or data.dtype == torch.int16
            or data.dtype == torch.int32
            or data.dtype == torch.int64
        ):
            raise TypeError(f"{data.dtype} is not supported in MaskedTensor.")
        
        # 检查数据和掩码的维度是否相同
        if data.dim() != mask.dim():
            raise ValueError("data.dim() must equal mask.dim()")
        
        # 检查数据和掩码的大小是否相同
        if data.size() != mask.size():
            raise ValueError("data.size() must equal mask.size()")

    def __init__(self, data, mask, requires_grad=False):
        # 预处理数据和掩码
        self._preprocess_data(data, mask)
        # 验证数据成员的一致性
        self._validate_members()

    @staticmethod
    # 定义一个静态方法，用于创建可微分的 MaskedTensor 对象
    def _from_values(data, mask):
        """Differentiable constructor for MaskedTensor"""

        class Constructor(torch.autograd.Function):
            @staticmethod
            def forward(ctx, data, mask):
                # 前向传播：直接返回 MaskedTensor 对象
                return MaskedTensor(data, mask)

            @staticmethod
            def backward(ctx, grad_output):
                # 反向传播：返回梯度和 None，因为这里不需要计算梯度
                return grad_output, None

        # 调用 Constructor 的 apply 方法来创建 MaskedTensor 对象并返回
        result = Constructor.apply(data, mask)
        return result

    # 设置 MaskedTensor 的数据和掩码
    def _set_data_mask(self, data, mask):
        self._masked_data = data  # 设置 MaskedTensor 的数据
        self._masked_mask = mask  # 设置 MaskedTensor 的掩码
        self._validate_members()  # 调用验证方法确保成员有效性

    # 返回 MaskedTensor 对象的字符串表示形式
    def __repr__(self):
        formatter = "{0:8.4f}"  # 格式化字符串的格式
        if self.dim() == 0:
            scalar_data = self.get_data().item()  # 获取标量数据
            # 根据数据类型进行格式化或直接转换成字符串
            data_formatted = (
                formatter.format(scalar_data)
                if isinstance(scalar_data, float)
                else str(scalar_data)
            )
            # 如果掩码为 False，则显示为 "--"
            if not self.get_mask().item():
                data_formatted = "--"
            # 返回带有数据和掩码的字符串表示形式
            return (
                "MaskedTensor("
                + data_formatted
                + ", "
                + str(self.get_mask().item())
                + ")"
            )
        # 对于非标量数据，调用 _masked_tensor_str 方法获取字符串表示形式，并适当缩进
        s = _masked_tensor_str(self.get_data(), self.get_mask(), formatter)
        s = "\n".join("  " + si for si in s.split("\n"))
        return "MaskedTensor(\n" + s + "\n)"

    # 用于处理 Torch 函数调用的特殊方法，实现 MaskedTensor 对象的功能
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        # 导入功能表，以便查找要调用的函数
        from ._ops_refs import _MASKEDTENSOR_FUNCTION_TABLE

        # 如果 func 在功能表中，调用对应的函数并返回结果
        if func in _MASKEDTENSOR_FUNCTION_TABLE:
            return _MASKEDTENSOR_FUNCTION_TABLE[func](*args, **kwargs)

        # 如果不是所有的类型都是 MaskedTensor 类的子类，则返回 NotImplemented
        if not all(issubclass(cls, t) for t in types):
            return NotImplemented

        # 禁用 Torch 函数的子类功能，调用原始的 func 函数
        with torch._C.DisableTorchFunctionSubclass():
            ret = func(*args, **kwargs)
            # 如果 func 在不需要包装的默认函数列表中，则直接返回结果
            if func in get_default_nowrap_functions():
                return ret
            else:
                # 将返回结果转换成 MaskedTensor 类型并返回
                return torch._tensor._convert(ret, cls)

    # 类方法：对给定的函数 fn 和数据掩码进行一元操作，返回新的 MaskedTensor 对象
    @classmethod
    def unary(cls, fn, data, mask):
        return MaskedTensor(fn(data), mask)

    # 类方法的开始，后续可能还有其他类方法定义
    # 当前类的特殊方法，用于分发函数调用到不同的具体实现
    def __torch_dispatch__(cls, func, types, args, kwargs):
        # 获取函数的重载包装器
        func = func.overloadpacket

        # 导入专用操作引用表
        from ._ops_refs import _MASKEDTENSOR_DISPATCH_TABLE

        # 如果函数在分发表中，则调用相应的函数处理参数和关键字参数
        if func in _MASKEDTENSOR_DISPATCH_TABLE:
            return _MASKEDTENSOR_DISPATCH_TABLE[func](*args, **kwargs)

        # 构造未实现警告消息，返回未实现
        msg = (
            f"{func.__name__} is not implemented in __torch_dispatch__ for MaskedTensor.\n"
            "If you would like this operator to be supported, please file an issue for a feature request at "
            "https://github.com/pytorch/maskedtensor/issues with a minimal reproducible code snippet.\n"
            "In the case that the semantics for the operator are not trivial, it would be appreciated "
            "to also include a proposal for the semantics."
        )
        # 发出警告并返回未实现
        warnings.warn(msg)
        return NotImplemented

    # 定义小于运算符的特殊方法，用于处理 MaskedTensor 与其他对象的小于比较
    def __lt__(self, other):
        # 如果 other 是 MaskedTensor，返回数据与其数据的小于比较结果
        if is_masked_tensor(other):
            return MaskedTensor(self.get_data() < _get_data(other), self.get_mask())
        # 否则，返回数据与给定值的小于比较结果
        return MaskedTensor(self.get_data() < other, self.get_mask())

    # 将数据中非掩码部分用指定值填充的方法
    def to_tensor(self, value):
        return self.get_data().masked_fill(~self.get_mask(), value)

    # 获取当前对象的数据部分的方法
    def get_data(self):
        # 内部类，继承自 torch.autograd.Function，用于处理前向传播和反向传播
        class GetData(torch.autograd.Function):
            @staticmethod
            # 前向传播方法，直接返回当前对象的掩码数据
            def forward(ctx, self):
                return self._masked_data

            @staticmethod
            # 反向传播方法，根据梯度输出进行处理，保持梯度传递结构
            def backward(ctx, grad_output):
                # 如果梯度输出是 MaskedTensor，则直接返回，否则构造新的 MaskedTensor 返回
                if is_masked_tensor(grad_output):
                    return grad_output
                return MaskedTensor(grad_output, self.get_mask())

        # 应用内部类的方法并返回结果
        return GetData.apply(self)

    # 获取当前对象的掩码部分的方法
    def get_mask(self):
        return self._masked_mask

    # 检查当前对象是否为稀疏 COO 格式的方法
    def is_sparse_coo(self):
        return self.layout == torch.sparse_coo

    # 检查当前对象是否为稀疏 CSR 格式的方法
    def is_sparse_csr(self):
        return self.layout == torch.sparse_csr

    # 属性方法，用于检查当前对象是否为稀疏格式（COO 或 CSR）
    # 后续更新以支持更多的稀疏布局
    @property
    def is_sparse(self):
        return self.is_sparse_coo() or self.is_sparse_csr()
```