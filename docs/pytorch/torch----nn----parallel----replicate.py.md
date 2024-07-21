# `.\pytorch\torch\nn\parallel\replicate.py`

```
# 从 collections 模块导入 OrderedDict 类
# 从 typing 模块导入多个类型相关的定义
# 引入 torch 库
# 从 torch._utils 模块导入 _get_device_index 函数
from collections import OrderedDict
from typing import (
    cast,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

import torch  # 导入 torch 库
from torch._utils import _get_device_index  # 从 torch._utils 模块导入 _get_device_index 函数

# 从 ..modules 中导入 Module 类
from ..modules import Module
# 从当前目录下的 comm 模块中导入所有内容
from . import comm

# 如果 TYPE_CHECKING 为 True，则执行以下导入
if TYPE_CHECKING:
    # 从 torch.jit 模块导入 ScriptModule 类型
    from torch.jit import ScriptModule
    # 从 torch.jit._state 模块导入 EnabledProxy 类型
    from torch.jit._state import EnabledProxy

# 定义 __all__ 列表，列出当前模块中公开的符号
__all__ = ["replicate"]

# 判断给定的 module 是否为 ScriptModule 类型
def _is_script_module(module: Module) -> bool:
    import torch.jit  # 导入 torch.jit 模块
    return isinstance(module, torch.jit.ScriptModule)

# 判断给定的 module 是否为 ScriptMethod 类型
def _is_script_method(module: Module) -> bool:
    import torch.jit  # 导入 torch.jit 模块
    return isinstance(module, torch._C.ScriptMethod)

# 初始化一个 ScriptModule 对象并返回
def _init_script_module() -> "ScriptModule":
    import torch.jit  # 导入 torch.jit 模块
    return torch.jit.ScriptModule()

# 检查是否启用了 JIT 编译，并返回 EnabledProxy 对象
def _is_jit_enabled() -> "EnabledProxy":
    import torch.jit._state  # 导入 torch.jit._state 模块
    return torch.jit._state._enabled

# 检查模块是否可以安全地进行复制
# 分两种模块：1. Python 模块；2. ScriptModule
#
# 如果某个 ScriptModule 的子模块包含 Python 模块，则不能正确地复制该模块
def _replicatable_module(module: Module, memo: Optional[Set[Module]] = None) -> bool:
    # 内部函数：生成 module 的后代模块迭代器
    def descendant_modules(module: Module) -> Iterator[Module]:
        gen = module.modules()  # 调用 module 的 modules 方法
        next(gen)  # 跳过第一个元素，即 module 自身
        return gen

    # 如果 JIT 编译未启用，直接返回 True
    if not _is_jit_enabled():
        return True
    if memo is None:
        memo = set()

    memo.add(module)  # 将当前 module 添加到 memo 中
    if _is_script_module(module):  # 如果当前 module 是 ScriptModule
        memo.update(descendant_modules(module))  # 更新 memo 到所有后代模块
        # 检查所有后代模块是否都是 ScriptModule 类型
        return all(
            _is_script_module(descendant) for descendant in descendant_modules(module)
        )

    # 遍历当前 module 的子模块
    for child in module.children():
        if child in memo:  # 如果子模块已经在 memo 中，跳过
            continue
        if not _replicatable_module(child, memo):  # 递归检查子模块是否可复制
            return False

    return True

# 对一组张量进行广播、合并和重塑操作
# tensors: 待广播的张量列表
# devices: 目标设备列表，可以是整数索引或 torch.device 对象
# detach: 是否在广播前分离张量的梯度信息，默认为 False
def _broadcast_coalesced_reshape(
    tensors: Sequence[torch.Tensor],
    devices: Sequence[Union[int, torch.device]],
    detach: bool = False,
) -> List[List[torch.Tensor]]:
    from ._functions import Broadcast  # 从当前目录下的 _functions 模块导入 Broadcast 类

    if detach:  # 如果 detach 为 True
        return comm.broadcast_coalesced(tensors, devices)  # 调用 comm 模块的 broadcast_coalesced 函数
    else:  # 如果 detach 为 False
        if len(tensors) > 0:  # 如果 tensors 列表非空
            tensor_copies = Broadcast.apply(devices, *tensors)  # 使用 Broadcast.apply 进行广播
            # 将 tensor_copies 按照原始 tensors 的分块重新组织，并返回结果
            return [
                tensor_copies[i : i + len(tensors)]
                for i in range(0, len(tensor_copies), len(tensors))
            ]
        else:  # 如果 tensors 列表为空
            return []


T = TypeVar("T", bound=Module)  # 定义一个类型变量 T，它是 Module 类型或其子类的子类型

# 复制一个网络模型到多个设备上
# network: 待复制的网络模型
# devices: 目标设备列表，可以是整数索引或 torch.device 对象
# detach: 是否在复制前分离张量的梯度信息，默认为 False
def replicate(
    network: T,
    devices: Sequence[Union[int, torch.device]],
    detach: bool = False,
) -> List[T]:
    # 检查网络是否可复制，如果不可复制则引发运行时错误
    if not _replicatable_module(network):
        raise RuntimeError(
            "Cannot replicate network where python modules are "
            "childrens of ScriptModule"
        )

    # 如果设备列表为空，则返回空列表
    if not devices:
        return []

    # 将设备列表中的每个设备转换为索引形式，确保返回索引值
    devices = [_get_device_index(x, True) for x in devices]
    # 计算设备的数量，即副本的数量
    num_replicas = len(devices)

    # 提取网络中所有参数，并生成参数到索引的字典
    params = list(network.parameters())
    param_indices = {param: idx for idx, param in enumerate(params)}
    # 对参数进行广播、合并和重塑，以便在不同设备间共享
    param_copies = _broadcast_coalesced_reshape(params, devices, detach)

    # 提取网络中所有缓冲区，并分为需要梯度和不需要梯度的两类
    buffers = list(network.buffers())
    buffers_rg: List[torch.Tensor] = []
    buffers_not_rg: List[torch.Tensor] = []
    for buf in buffers:
        if buf.requires_grad and not detach:
            buffers_rg.append(buf)
        else:
            buffers_not_rg.append(buf)

    # 生成需要梯度缓冲区的索引字典和不需要梯度缓冲区的索引字典
    buffer_indices_rg = {buf: idx for idx, buf in enumerate(buffers_rg)}
    buffer_indices_not_rg = {buf: idx for idx, buf in enumerate(buffers_not_rg)}

    # 对需要梯度的缓冲区进行广播、合并和重塑
    buffer_copies_rg = _broadcast_coalesced_reshape(buffers_rg, devices, detach=detach)
    # 对不需要梯度的缓冲区进行广播、合并和重塑，强制进行分离
    buffer_copies_not_rg = _broadcast_coalesced_reshape(
        buffers_not_rg, devices, detach=True
    )

    # 提取网络中所有模块，并为每个设备创建模块副本的列表
    modules = list(network.modules())
    module_copies: List[List[Module]] = [[] for _ in devices]
    module_indices: Dict[Module, int] = {}

    # 遍历网络中的每个模块，为每个设备创建相应的模块副本
    for i, module in enumerate(modules):
        module_indices[module] = i
        # 对每个设备创建模块的副本，并添加到对应设备的副本列表中
        for j in range(num_replicas):
            replica = module._replicate_for_data_parallel()
            # 添加一个临时修复，用于支持分布式数据并行 (DDP)
            # DDP 需要访问复制模型的参数，之前使用 `module.parameters()`，
            # 但是在 DP 中为了安全性修改了 `parameters()` 方法，不再暴露复制的参数。
            # 因此，在这里添加 `_former_parameters` 字典来支持 DDP。
            replica._former_parameters = OrderedDict()

            # 将模块的副本添加到对应设备的模块副本列表中
            module_copies[j].append(replica)
    # 遍历 modules 列表中的模块，同时获取索引 i 和每个模块 module
    for i, module in enumerate(modules):
        # 遍历当前模块 module 的所有子模块及其键值对
        for key, child in module._modules.items():
            # 如果子模块 child 为空
            if child is None:
                # 对于每个副本（replica），将当前模块的对应子模块置为空
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._modules[key] = None
            else:
                # 获取子模块在 module_copies 中的索引
                module_idx = module_indices[child]
                # 对于每个副本（replica），将当前模块的对应子模块设置为副本中的对应子模块
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    setattr(replica, key, module_copies[j][module_idx])
        
        # 遍历当前模块 module 的所有参数及其键值对
        for key, param in module._parameters.items():
            # 如果参数 param 为空
            if param is None:
                # 对于每个副本（replica），将当前模块的对应参数置为空
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._parameters[key] = None
            else:
                # 获取参数在 param_copies 中的索引
                param_idx = param_indices[param]
                # 对于每个副本（replica），将当前模块的对应参数设置为副本中的对应参数副本
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    param_copy = param_copies[j][param_idx]
                    # 设置副本的非参数属性为参数
                    setattr(replica, key, param_copy)
                    # 将参数副本暴露给分布式数据并行（DDP）
                    replica._former_parameters[key] = param_copy
        
        # 遍历当前模块 module 的所有缓冲区及其键值对
        for key, buf in module._buffers.items():  # type: ignore[assignment]
            # 如果缓冲区 buf 为空
            if buf is None:
                # 对于每个副本（replica），将当前模块的对应缓冲区置为空
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    replica._buffers[key] = None
            else:
                # 根据条件选择要使用的缓冲区副本列表和索引
                if buf.requires_grad and not detach:
                    buffer_copies = buffer_copies_rg
                    buffer_idx = buffer_indices_rg[buf]
                else:
                    buffer_copies = buffer_copies_not_rg
                    buffer_idx = buffer_indices_not_rg[buf]
                # 对于每个副本（replica），将当前模块的对应缓冲区设置为副本中的对应缓冲区副本
                for j in range(num_replicas):
                    replica = module_copies[j][i]
                    setattr(replica, key, buffer_copies[j][buffer_idx])

    # 返回每个副本的第一个模块的类型强制转换列表
    return [cast(T, module_copies[j][0]) for j in range(num_replicas)]
```