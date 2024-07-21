# `.\pytorch\torchgen\local.py`

```py
# 从未来导入注释，使得类型注释在旧版本Python中也能正常工作
from __future__ import annotations

# 导入线程相关模块
import threading
# 导入上下文管理器相关模块
from contextlib import contextmanager
# 导入类型提示相关模块
from typing import Iterator

# 定义一个特定于线程的类，用于动态作用域的实现
class Locals(threading.local):
    # 控制是否使用常量引用来处理可变张量的标志，初始为None
    use_const_ref_for_mutable_tensors: bool | None = None
    # 控制是否使用ilistref来处理张量列表的标志，初始为None
    use_ilistref_for_tensor_lists: bool | None = None

# 创建Locals类的一个实例
_locals = Locals()

# 返回当前线程是否使用常量引用处理可变张量的标志
def use_const_ref_for_mutable_tensors() -> bool:
    assert _locals.use_const_ref_for_mutable_tensors is not None, (
        "need to initialize local.use_const_ref_for_mutable_tensors with "
        "local.parametrize"
    )
    return _locals.use_const_ref_for_mutable_tensors

# 返回当前线程是否使用ilistref处理张量列表的标志
def use_ilistref_for_tensor_lists() -> bool:
    assert _locals.use_ilistref_for_tensor_lists is not None, (
        "need to initialize local.use_ilistref_for_tensor_lists with "
        "local.parametrize"
    )
    return _locals.use_ilistref_for_tensor_lists

# 定义一个上下文管理器，用于临时修改动态作用域中的参数
@contextmanager
def parametrize(
    *, use_const_ref_for_mutable_tensors: bool, use_ilistref_for_tensor_lists: bool
) -> Iterator[None]:
    # 保存旧的标志值以便后续恢复
    old_use_const_ref_for_mutable_tensors = _locals.use_const_ref_for_mutable_tensors
    old_use_ilistref_for_tensor_lists = _locals.use_ilistref_for_tensor_lists
    try:
        # 设置新的标志值
        _locals.use_const_ref_for_mutable_tensors = use_const_ref_for_mutable_tensors
        _locals.use_ilistref_for_tensor_lists = use_ilistref_for_tensor_lists
        # 执行yield之前的代码，允许用户在此期间使用修改后的标志
        yield
    finally:
        # 恢复旧的标志值
        _locals.use_const_ref_for_mutable_tensors = (
            old_use_const_ref_for_mutable_tensors
        )
        _locals.use_ilistref_for_tensor_lists = old_use_ilistref_for_tensor_lists
```