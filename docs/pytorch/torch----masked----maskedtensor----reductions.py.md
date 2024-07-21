# `.\pytorch\torch\masked\maskedtensor\reductions.py`

```
# 设置 mypy 选项，允许未标记类型的定义
# 版权声明
import warnings  # 导入警告模块

import torch  # 导入 PyTorch 库

from .core import is_masked_tensor  # 从当前目录下的 core 模块导入 is_masked_tensor 函数
from .creation import as_masked_tensor, masked_tensor  # 从当前目录下的 creation 模块导入 as_masked_tensor 和 masked_tensor 函数

__all__ = []  # 定义一个空列表，用于存放导出的变量名，忽略类型注解

# 定义一个函数，用于在所有元素上应用 all 操作
def _masked_all_all(data, mask=None):
    if mask is None:
        return data.all()
    return data.masked_fill(~mask, True).all()

# 定义一个函数，用于在指定维度上应用 all 操作
def _masked_all_dim(data, dim, keepdim=False, mask=None):
    if mask is None:
        return torch.all(data, dim=dim, keepdim=keepdim)
    return torch.all(data.masked_fill(~mask, True), dim=dim, keepdim=keepdim)

# 定义一个函数，根据参数个数调用不同的函数
def _masked_all(*args, **kwargs):
    if len(args) == 1 and len(kwargs) == 1:
        return _masked_all_all(args[0], mask=kwargs["mask"])
    return _masked_all_dim(*args, **kwargs)

# 定义一个函数，用于在多维度上应用 any 操作
def _multidim_any(mask, dim, keepdim):
    if isinstance(dim, int):
        return _multidim_any(mask, [dim], keepdim)
    for d in sorted(dim, reverse=True):
        mask = torch.any(mask, dim=d, keepdim=keepdim)
    return mask

# 根据函数名获取对应的函数
def _get_masked_fn(fn):
    if fn == "all":
        return _masked_all
    return getattr(torch.masked, fn)

# 定义一个函数，用于在所有元素上应用指定的函数
def _torch_reduce_all(fn):
    def reduce_all(self):
        masked_fn = _get_masked_fn(fn)
        data = self.get_data()
        mask = self.get_mask().values() if self.is_sparse else self.get_mask()
        # 当减少操作是 "all" 时，torch.argmin/torch.argmax 需要返回对应最小/最大元素的索引，但对于稀疏布局，此操作不正确支持
        # 因此，此实现使用步长来计算
        if fn == "all":
            result_data = masked_fn(data, mask=mask)

        elif fn in {"argmin", "argmax"} and self.is_sparse_coo():
            sparse_idx = masked_fn(data.values(), mask=mask).to(dtype=torch.int)
            indices = (
                data.to_sparse_coo().indices()
                if not self.is_sparse_coo()
                else data.indices()
            )
            idx = indices.unbind(1)[sparse_idx]
            stride = data.size().numel() / torch.tensor(
                data.size(), device=data.device
            ).cumprod(0)
            result_data = torch.sum(idx * stride)

        # 对于稀疏 COO/CSR 张量，我们简单地传入值
        elif self.is_sparse:
            result_data = masked_fn(masked_tensor(data.values(), mask))

        else:
            result_data = masked_fn(self, mask=mask)

        return as_masked_tensor(result_data, torch.any(mask))

    return reduce_all

# 定义一个函数，用于在指定维度上应用指定的函数
def _torch_reduce_dim(fn):
    # 定义一个方法用于降低张量的维度
    def reduce_dim(self, dim, keepdim=False, dtype=None):
        # 如果当前张量是稀疏的，则给出警告信息并返回NotImplemented
        if self.is_sparse:
            msg = (
                f"The sparse version of {fn} is not implemented in reductions.\n"
                "If you would like this operator to be supported, please file an issue for a feature request at "
                "https://github.com/pytorch/maskedtensor/issues with a minimal reproducible code snippet.\n"
                "In the case that the semantics for the operator are not trivial, it would be appreciated "
                "to also include a proposal for the semantics."
            )
            warnings.warn(msg)
            return NotImplemented
        
        # 如果输入的张量不是 MaskedTensor 类型，则抛出 TypeError
        if not is_masked_tensor(self):
            raise TypeError("Input to reduce_dim must be a MaskedTensor")
        
        # 根据当前函数获取相应的 masked 函数
        masked_fn = _get_masked_fn(fn)
        
        # 获取当前张量的数据部分
        data = self.get_data()
        
        # 获取当前张量的掩码部分
        mask = self.get_mask()
        
        # 根据不同的函数名进行相应的操作
        if fn == "all":
            # 对所有元素进行操作
            result_data = masked_fn(data, dim=dim, keepdim=keepdim, mask=mask)
        else:
            # 对当前张量进行操作
            result_data = masked_fn(
                self, dim=dim, keepdim=keepdim, dtype=dtype, mask=self.get_mask()
            )
        
        # 将操作后的结果转换为 MaskedTensor 类型并返回
        return as_masked_tensor(result_data, _multidim_any(mask, dim, keepdim))
    
    # 返回 reduce_dim 方法
    return reduce_dim
# 定义一个装饰器函数，用于包装给定的函数 `fn`
def _torch_reduce(fn):
    # 定义一个新的函数 `reduce_fn`，用于处理被装饰函数 `fn` 的调用
    def reduce_fn(*args, **kwargs):
        # 如果参数只有一个且没有关键字参数
        if len(args) == 1 and len(kwargs) == 0:
            # 调用 `_torch_reduce_all` 函数来处理单个参数的情况
            return _torch_reduce_all(fn)(args[0])
        # 否则，调用 `_torch_reduce_dim` 函数处理所有参数
        return _torch_reduce_dim(fn)(*args, **kwargs)

    return reduce_fn


# 返回输入的数据、维度、是否保持维度不变以及数据类型的元组
def _reduce_dim_args(input, dim, keepdim=False, dtype=None):
    return input, dim, keepdim, dtype


# 定义一个装饰器函数，用于处理梯度相关的降维操作
def _torch_grad_reduce(fn):
    # 定义一个新的函数 `grad_reduce`，用于处理被装饰函数 `fn` 的调用
    def grad_reduce(*args, **kwargs):
        # 如果参数只有一个且没有关键字参数
        if len(args) == 1 and len(kwargs) == 0:
            # 调用 `_torch_reduce_all` 函数来处理单个参数的情况
            return _torch_reduce_all(fn)(args[0])
        # 否则，调用 `_reduce_dim_args` 函数来解包参数并调用 `_torch_reduce_dim` 函数
        input, dim, keepdim, dtype = _reduce_dim_args(*args, **kwargs)
        return _torch_reduce_dim(fn)(input, dim, keepdim, dtype)

    return grad_reduce


# 定义一个包含各种降维操作名称的列表
REDUCE_NAMES = [
    "sum",
    "mean",
    "amin",
    "amax",
    "argmin",
    "argmax",
    "prod",
    "all",
    "norm",
    "var",
    "std",
]

# 创建一个字典，将每个降维操作名映射到对应的 `_torch_reduce` 函数
NATIVE_REDUCE_MAP = {
    getattr(torch.ops.aten, name): _torch_reduce(name) for name in REDUCE_NAMES
}

# 创建一个字典，将每个降维操作名映射到对应的 `_torch_grad_reduce` 函数
TORCH_REDUCE_MAP = {
    getattr(torch, name): _torch_grad_reduce(name) for name in REDUCE_NAMES
}

# 创建一个字典，将每个降维操作名映射到对应的 `_torch_grad_reduce` 函数（针对 `torch.Tensor` 类）
TENSOR_REDUCE_MAP = {
    getattr(torch.Tensor, name): _torch_grad_reduce(name) for name in REDUCE_NAMES
}

# 分别获取三个字典的键列表，即所有支持的降维操作函数
NATIVE_REDUCE_FNS = list(NATIVE_REDUCE_MAP.keys())
TORCH_REDUCE_FNS = list(TORCH_REDUCE_MAP.keys())
TENSOR_REDUCE_FNS = list(TENSOR_REDUCE_MAP.keys())


# 判断给定的函数 `fn` 是否是支持的降维操作函数
def _is_reduction(fn):
    return fn in NATIVE_REDUCE_MAP or fn in TORCH_REDUCE_MAP or fn in TENSOR_REDUCE_MAP


# 应用给定的降维操作函数 `fn` 到指定的参数和关键字参数
def _apply_reduction(fn, *args, **kwargs):
    if fn in NATIVE_REDUCE_MAP:
        return NATIVE_REDUCE_MAP[fn](*args, **kwargs)
    if fn in TORCH_REDUCE_MAP:
        return TORCH_REDUCE_MAP[fn](*args, **kwargs)
    if fn in TENSOR_REDUCE_MAP:
        return TENSOR_REDUCE_MAP[fn](*args, **kwargs)
    # 如果 `fn` 不是支持的降维操作函数，则返回 `NotImplemented`
    return NotImplemented
```