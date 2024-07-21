# `.\pytorch\torch\_subclasses\fake_utils.py`

```py
# 忽略类型检查错误的标志
# functools 模块提供的装饰器和其他功能
import functools
# 引入警告模块
import warnings
# 引入类型提示相关内容
from typing import Callable, Union

# 引入 PyTorch 库
import torch
# 引入 PyTorch 的 _pytree 模块
import torch.utils._pytree as pytree
# 引入 PyTorch 的操作重载机制
from torch._ops import OpOverload
# 引入 PyTorch 的假张量模块
from torch._subclasses.fake_tensor import (
    FakeTensorMode,
    tree_flatten_only,
    UnsupportedFakeTensorException,
)
# 引入 PyTorch 的 Python 分发模块
from torch.utils._python_dispatch import TorchDispatchMode

# 从 torch._ops.ops 中导入 aten 符号
aten = torch._ops.ops.aten

# 检查输出是否别名输入的函数
def outputs_alias_inputs(outputs, inputs):
    # 获取输入张量的存储 ID
    input_storages = {
        inp._typed_storage()._cdata
        for inp in tree_flatten_only(torch.Tensor, inputs)
        if torch._C._has_storage(inp)
    }
    # 检查输出张量是否使用了相同的存储 ID
    return any(
        torch._C._has_storage(out) and out._typed_storage()._cdata in input_storages
        for out in tree_flatten_only(torch.Tensor, outputs)
    )

# 检查输出是否是输入的函数
def outputs_are_inputs(outputs, inputs):
    # 获取输入张量的 ID
    input_ids = {id(inp) for inp in tree_flatten_only(torch.Tensor, inputs)}
    # 检查输出张量是否具有相同的 ID
    return any(id(out) in input_ids for out in tree_flatten_only(torch.Tensor, outputs))

# 检查输出是否彼此别名的函数
def output_alias_each_other(outputs):
    storages = set()
    # 遍历输出张量，检查是否有相同的存储 ID
    for out in tree_flatten_only(torch.Tensor, outputs):
        if not torch._C._has_storage(out):
            continue
        stor = out._typed_storage()._cdata
        if stor in storages:
            return True
        storages.add(stor)
    return False

# 判断是否为 SDPA 错误的函数
def is_sdpa_error(func, idx, e):
    if (
        (
            func is aten._scaled_dot_product_flash_attention.default
            or func is aten._flash_attention_forward.default
        )
        and idx in (6, 7)
        and "Devices" in repr(e)
    ):
        return True
    if (
        (
            func is aten._scaled_dot_product_efficient_attention.default
            or func is aten._efficient_attention_forward.default
        )
        and idx in (2, 3)
        and "Devices" in repr(e)
    ):
        return True
    return False

# 交叉引用假张量模式的类
class CrossRefFakeMode(TorchDispatchMode):
    def __init__(
        self,
        ignore_op_fn: Union[Callable[[OpOverload], bool], None] = None,
        *,
        check_strides=True,
        check_aliasing=True,
    ):
        # 初始化函数，设置忽略操作函数和检查步长、别名的标志
        self.ignore_op_fn = (
            ignore_op_fn if ignore_op_fn is not None else lambda fn: False
        )
        self.check_strides = check_strides
        self.check_aliasing = check_aliasing
```