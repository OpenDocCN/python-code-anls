# `.\pytorch\torch\_inductor\inductor_prims.py`

```py
# mypy: allow-untyped-defs
# 引入未类型化的函数定义的兼容性声明

from __future__ import annotations
# 导入未来版本中的类型注解支持

import logging
# 导入日志记录模块
from typing import Optional, Sequence
# 导入类型提示相关模块

import torch
# 导入 PyTorch 库
from torch import _prims, Tensor
# 从 PyTorch 中导入 _prims 模块和 Tensor 类

log = logging.getLogger(__name__)
# 获取当前模块的日志记录器

def make_prim(
    schema: str,
    impl_aten,
    return_type=_prims.RETURN_TYPE.NEW,
    doc: str = "",
    tags: Optional[Sequence[torch.Tag]] = None,
):
    # 创建自定义的原语操作函数
    if isinstance(return_type, tuple):
        # 如果返回类型是元组，则定义元组类型的元信息函数
        def meta(*args, **kwargs):
            return tuple(_prims.TensorMeta(o) for o in impl_aten(*args, **kwargs))
    else:
        # 否则定义单个类型的元信息函数
        def meta(*args, **kwargs):
            return _prims.TensorMeta(impl_aten(*args, **kwargs))

    # 返回通过 _prims._make_prim 创建的原语对象
    return _prims._make_prim(
        schema=schema,
        return_type=return_type,
        meta=meta,
        impl_aten=impl_aten,
        doc=doc,
        tags=tags,
    )

def eager_force_stride(input_tensor: Tensor, stride) -> Tensor:
    # 强制修改输入张量的步幅
    if input_tensor.stride() == stride:
        # 如果输入张量的步幅已经是指定的步幅，则直接返回输入张量
        return input_tensor
    # 否则克隆输入张量，并按照指定的步幅重新构造张量
    new_tensor = input_tensor.clone().as_strided(
        input_tensor.shape,
        stride,
    )
    # 将原始张量的数据拷贝到新的张量中
    new_tensor.copy_(input_tensor)
    # 返回新的张量
    return new_tensor

# 自定义的用于处理随机性的原语操作
seed = make_prim(
    "inductor_seed(Device device) -> Tensor",
    lambda device: torch.randint(2**63 - 1, [], device=device),
    doc="create a fresh seed (one per call) for use with inductor_rand",
    tags=(torch.Tag.nondeterministic_seeded,),
)
seeds = make_prim(
    "inductor_seeds(int count, Device device) -> Tensor",
    lambda count, device: torch.randint(2**63 - 1, [count], device=device),
    doc="Horizontal fusion of many inductor_seed() calls",
    tags=(torch.Tag.nondeterministic_seeded,),
)
lookup_seed = make_prim(
    # 如果 inductor_lookup_seed 发生变化，请更新 partitioners.py
    "inductor_lookup_seed(Tensor seeds, int index) -> Tensor",
    lambda seeds, index: seeds[index],
    doc="Extract a single seed from the result of inductor_seeds()",
)
random = make_prim(
    "inductor_random(SymInt[] size, Tensor seed, str mode) -> Tensor",
    lambda size, seed, mode: getattr(torch, mode)(size, device=seed.device),
    doc="torch.rand()/torch.randn() using backend-specific RNG that can be fused",
)
randint = make_prim(
    "inductor_randint(SymInt low, SymInt high, SymInt[] size, Tensor seed) -> Tensor",
    lambda low, high, size, seed: torch.randint(low, high, size, device=seed.device),
    doc="torch.randint() using backend-specific RNG that can be fused",
)
force_stride_order = make_prim(
    "inductor_force_stride_order(Tensor input, SymInt[] stride) -> Tensor",
    eager_force_stride,
    doc="Force the stride order for input tensor. No-op if the input tensor already has the stride. Do a copy otherwise",
)
_unsafe_index_put_ = make_prim(
    "_unsafe_index_put_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor(a!)",
    lambda self, indices, values, accumulate=False: torch.ops.aten.index_put_(
        self, indices, values, accumulate
    ),
    # 使用底层操作 aten.index_put_ 对张量进行不安全的索引赋值操作
)
    doc="Unsafe index_put_ (doesn't issue device asserts)",


    # 设置变量 doc 为字符串 "Unsafe index_put_ (doesn't issue device asserts)"
# 使用 make_prim 函数创建一个名为 fma 的新函数
fma = make_prim(
    "fma(Tensor a, Tensor b, Tensor c) -> Tensor",  # 函数签名和文档字符串
    lambda a, b, c: (a * b) + c,  # 使用 lambda 表达式定义函数实现：计算 (a * b) + c
    doc="Fused multiply add: fma(a, b, c) -> (a * b) + c without rounding after the multiplication",  # 函数的文档说明
)

# 定义一个名为 _low_memory_max_pool2d_with_offsets_aten 的函数
def _low_memory_max_pool2d_with_offsets_aten(
    self,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode,
):
    # 调用 Torch 的原生操作 max_pool2d_with_indices，返回两个张量：vals 和 indices
    vals, indices = torch.ops.aten.max_pool2d_with_indices(
        self, kernel_size, stride, padding, dilation, ceil_mode
    )
    # 将 indices 转换为 torch.int8 类型，并返回 vals 和 indices
    return vals, indices.to(torch.int8)


# 使用 make_prim 函数创建一个名为 _low_memory_max_pool2d_with_offsets 的新函数
_low_memory_max_pool2d_with_offsets = make_prim(
    "_low_memory_max_pool2d_with_offsets(Tensor self, SymInt[2] kernel_size, SymInt[2] stride,  SymInt[2] padding, SymInt[2] dilation, bool ceil_mode) -> (Tensor, Tensor)",  # 函数签名和文档字符串
    _low_memory_max_pool2d_with_offsets_aten,  # 使用 _low_memory_max_pool2d_with_offsets_aten 函数作为实现
    return_type=(_prims.RETURN_TYPE.NEW, _prims.RETURN_TYPE.NEW),  # 指定返回类型
    doc="Instead of returning indices, returns indices offsets.",  # 函数的文档说明
)

# 使用 make_prim 函数创建一个名为 _low_memory_max_pool2d_offsets_to_indices 的新函数
_low_memory_max_pool2d_offsets_to_indices = make_prim(
    "_low_memory_max_pool2d_offsets_to_indices(Tensor self, SymInt kernel_w, SymInt input_w, SymInt[2] stride, SymInt[2] padding) -> Tensor",  # 函数签名和文档字符串
    lambda self, *args: self.to(torch.int64),  # 使用 lambda 表达式定义函数实现：将 self 转换为 torch.int64 类型
    doc="Convert small int offsets to regular indices.",  # 函数的文档说明
)
```