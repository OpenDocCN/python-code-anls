# `.\pytorch\torch\distributed\_spmd\comm_tensor.py`

```py
# mypy: allow-untyped-defs
# 引入必要的模块和库
from dataclasses import dataclass
from functools import partial
from typing import Any, List, Optional, Tuple

import torch
from torch._C import _disabled_torch_function_impl
from torch.fx.experimental.proxy_tensor import (
    _ProxyTensor,
    fetch_object_proxy,
    get_innermost_proxy_mode,
    get_proxy_slot,
    set_proxy_slot,
    track_tensor_tree,
)
from torch.utils import _pytree as pytree
from torch.utils._mode_utils import no_dispatch
from torch.utils._pytree import tree_flatten, tree_map, tree_map_only


@dataclass
# 定义一个数据类，包含两个字段，一个是torch.Tensor类型，一个是torch.distributed._Work类型
class _CommResult:
    _tensor: torch.Tensor  # 封装了原地输出张量
    _work: torch.distributed._Work  # 封装了工作句柄


# 等待通信结果的函数，参数为_CommResult类型，仅在追踪模式下作为call_function节点使用
def _wait_comm(comm_result: _CommResult):
    comm_result._work.wait()  # 等待工作句柄的完成
    return comm_result._tensor  # 返回封装的张量


# 将通信结果包装成_CommResult类型的函数
def _wrap_comm_result(result: Tuple[Any, Any]) -> Tuple[Any, Any]:
    # 内部函数，接收工作句柄和张量，并返回_CommResult类型
    def wrap(work, e):
        assert isinstance(e, torch.Tensor), (
            "Excepting collection of tensors as the first element in the "
            "return value of communication operations."
        )
        return _CommResult(e, work)

    work = result[1]  # 获取工作句柄
    # 对result[0]中的每个张量应用wrap函数，返回处理后的结果和工作句柄
    return (tree_map(partial(wrap, work), result[0]), work)


# 获取当前最内层的代理模式追踪器
def _get_tracer() -> Optional[torch.fx.Tracer]:
    mode = get_innermost_proxy_mode()
    if mode is None:
        return None
    return mode.tracer


# CommTensor类，继承自torch.Tensor，用于包装用于集体通信的输入张量
class CommTensor(torch.Tensor):
    r"""
    一个用于包装输入张量以进行集体通信的张量子类。

    该张量子类适用于即时模式和追踪模式。
    在即时模式下，它将记录是否使用该张量启动了原地集体通信，并记住相应的工作句柄。
    如果是，则在“__torch_dispatch__”函数中的后续操作中显式调用wait()。
    
    在追踪模式下，“CommTensor”使用“__torch_dispatch__”函数插入两个节点到图中。
    1. 第一个节点是在通信之后插入的，将原地输出张量和返回的工作句柄封装到自定义的“_CommResult”类型中。
    我们必须这样做，因为“ProxyTorchDispatchMode”仅处理“torch.Tensor”、“_ProxyTensor”和“torch.nn.Parameter”对象，
    并且会将工作句柄视为常量并嵌入到图中。因此，在执行期间，它将使用追踪期间创建的工作句柄，并导致错误结果。
    该测试中的解决方案是手动在“allreduce_”的返回值上创建一个代理，即“([tensor], work)”，并将其包装为“[(_CommResult(tensor, work)), work]”。
    这样，后续节点可以
    # 定义一个静态列表，包含支持的通信操作名称
    _supported_comms: List[str] = [
        "_allgather_base_",
        "_reduce_scatter_base_",
        "allreduce_",
        "allgather_",
        "alltoall_",
        "broadcast_",
        "reduce_scatter_",
        "scatter_",
    ]

    # 定义一个 Torch 张量对象
    _tensor: torch.Tensor
    # 可选的 Torch 分布式工作对象
    _work: Optional[torch.distributed._Work]

    @staticmethod
    def __new__(cls, tensor: torch.Tensor):
        # 如果传入的 tensor 是 CommTensor 类型的对象，则使用其内部的 _tensor
        t = tensor._tensor if isinstance(tensor, CommTensor) else tensor
        # 如果当前的代理模式为 None，直接返回传入的 tensor
        if get_innermost_proxy_mode() is None:
            # 在急切模式下不执行任何操作
            return tensor

        # 在非 CommTensor 的情况下使用原始的 torch.Tensor 对象来创建子类
        r = torch.Tensor._make_subclass(cls, t, require_grad=t.requires_grad)
        # 将传入的 tensor 对象保存在创建的 r 对象中
        # 注意：此时 r 可能是 CommTensor 类型；参见 test_nested_comm_tensor_wrapping
        r._tensor = tensor  # type: ignore[attr-defined]
        # 记录由集体通信操作返回的最后一个 'work' 对象
        # 如果为 None，表示自上次 tensor 被 CommTensor 包装以来没有调用集体操作
        r._work = None  # type: ignore[attr-defined]
        return r

    def __repr__(self):
        # 返回 CommTensor 对象的字符串表示形式，包括 _tensor 和 _work 的信息
        return f"CommTensor({self._tensor}, work={self._work})"

    # 禁用 __torch_function__ 方法，以便 CommTensor 可以递归地使用 ProxyTorchDispatchMode 在 make_fx 中调度
    __torch_function__ = _disabled_torch_function_impl

    @classmethod
    def _is_supported(cls, op_name):
        # 检查给定的操作名称是否在支持的通信操作列表中的任何一个中
        return any(comm in op_name for comm in cls._supported_comms)

    @classmethod
```