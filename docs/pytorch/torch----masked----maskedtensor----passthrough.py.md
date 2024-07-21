# `.\pytorch\torch\masked\maskedtensor\passthrough.py`

```
# mypy: allow-untyped-defs
# mypy选项：允许未标记类型的函数定义
# Copyright (c) Meta Platforms, Inc. and affiliates
# 版权声明：Meta Platforms, Inc.及其关联公司
"""
These are functions that should simply be applied to both mask and data.
Take select or stack as an example. This operation can be applied to
both the mask and data of a MaskedTensor and the result wrapped into
a new MaskedTensor as a result.
"""
"""
这些函数应该简单地应用于mask和data。
以select或stack为例。该操作可以应用于MaskedTensor的mask和data，并将结果包装成新的MaskedTensor。
"""

import torch

from .core import _map_mt_args_kwargs, _wrap_result
# 导入torch模块及其他自定义模块

__all__ = []  # type: ignore[var-annotated]
# 将__all__设置为空列表，这意味着没有公开的API

PASSTHROUGH_FNS = [
    torch.ops.aten.select,
    torch.ops.aten.transpose,
    torch.ops.aten.split,
    torch.ops.aten.t,
    torch.ops.aten.slice,
    torch.ops.aten.slice_backward,
    torch.ops.aten.select_backward,
    torch.ops.aten.index,
    torch.ops.aten.expand,
    torch.ops.aten.view,
    torch.ops.aten._unsafe_view,
    torch.ops.aten._reshape_alias,
    torch.ops.aten.cat,
    torch.ops.aten.unsqueeze,
]
# PASSTHROUGH_FNS定义了一系列torch操作函数，这些函数可以直接应用于数据和掩码

def _is_pass_through_fn(fn):
    # 判断函数fn是否在PASSTHROUGH_FNS列表中
    return fn in PASSTHROUGH_FNS


def _apply_pass_through_fn(fn, *args, **kwargs):
    # 将数据和掩码分开处理并应用传递函数fn，然后将结果包装成新的MaskedTensor返回
    data_args, data_kwargs = _map_mt_args_kwargs(args, kwargs, lambda x: x.get_data())
    # 获取数据参数和关键字参数
    result_data = fn(*data_args, **data_kwargs)
    # 应用fn到数据上，得到结果数据

    mask_args, mask_kwargs = _map_mt_args_kwargs(args, kwargs, lambda x: x.get_mask())
    # 获取掩码参数和关键字参数
    result_mask = fn(*mask_args, **mask_kwargs)
    # 应用fn到掩码上，得到结果掩码

    return _wrap_result(result_data, result_mask)
    # 将结果数据和结果掩码包装成新的MaskedTensor并返回
```