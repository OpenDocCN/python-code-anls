# `.\pytorch\torch\distributed\nn\jit\templates\remote_module_template.py`

```py
#!/usr/bin/python3
# mypy: allow-untyped-defs

# 定义一个函数，根据是否允许将 CPU 张量移到 CUDA 设备，返回远程模块的模板
def get_remote_module_template(enable_moving_cpu_tensors_to_cuda: bool):
    return _TEMPLATE_PREFIX + (
        _REMOTE_FORWARD_TEMPLATE_ENABLE_MOVING_CPU_TENSORS_TO_CUDA
        if enable_moving_cpu_tensors_to_cuda
        else _REMOTE_FORWARD_TEMPLATE
    )

# 定义模板的前缀部分，包含所需的导入和模块接口类的赋值
_TEMPLATE_PREFIX = """from typing import *

import torch
import torch.distributed.rpc as rpc
from torch import Tensor
from torch._jit_internal import Future
from torch.distributed.rpc import RRef
from typing import Tuple  # pyre-ignore: unused import

{assign_module_interface_cls}

# 定义异步远程调用函数
def forward_async(self, {arg_types}){arrow_and_future_return_type}:
    args = (self.module_rref, self.device, self.is_device_map_set, {args})
    kwargs = {{{kwargs}}}
    return rpc.rpc_async(
        self.module_rref.owner(),
        _remote_forward,
        args,
        kwargs,
    )

# 定义同步远程调用函数
def forward(self, {arg_types}){arrow_and_return_type}:
    args = (self.module_rref, self.device, self.is_device_map_set, {args})
    kwargs = {{{kwargs}}}
    ret_fut = rpc.rpc_async(
        self.module_rref.owner(),
        _remote_forward,
        args,
        kwargs,
    )
    return ret_fut.wait()

# 生成的方法列表
_generated_methods = [
    forward_async,
    forward,
]

{jit_script_decorator}
"""

# 当允许将 CPU 张量移动到 CUDA 设备时使用的远程前向传播模板
_REMOTE_FORWARD_TEMPLATE_ENABLE_MOVING_CPU_TENSORS_TO_CUDA = """
def _remote_forward(
    module_rref: RRef[module_interface_cls], device: str, is_device_map_set: bool, {arg_types}){arrow_and_return_type}:
    module = module_rref.local_value()
    device = torch.device(device)

    if device.type != "cuda":
        return module.forward({args}, {kwargs})

    # 如果模块在 CUDA 设备上，
    # 将 args 或 kwargs 中的任何 CPU 张量移动到相同的 CUDA 设备上。
    args = ({args},)
    out_args: Tuple[()] = ()
    for arg in args:
        arg = (arg.to(device),) if isinstance(arg, Tensor) else (arg,)
        out_args = out_args + arg

    kwargs = {{{kwargs}}}
    for k, v in kwargs.items():
        if isinstance(v, Tensor):
            kwargs[k] = kwargs[k].to(device)

    if is_device_map_set:
        return module.forward(*out_args, {kwargs})

    # 如果设备映射为空，则只允许发送 CPU 张量，
    # 因此必须将输出中的任何 GPU 张量移动到 CPU 上。
"""

# 当不允许将 CPU 张量移动到 CUDA 设备时使用的远程前向传播模板
_REMOTE_FORWARD_TEMPLATE = """
def _remote_forward(
    module_rref: RRef[module_interface_cls], device: str, is_device_map_set: bool, {arg_types}){arrow_and_return_type}:
    module = module_rref.local_value()
    return module.forward({args}, {kwargs})
"""

# 注释：
# 此模板可能会导致类型错误（``Tuple[()]`` 和 ``Tuple[Any]`` 之间的不匹配），
# 即使代码仅用于实例化而不是执行。
# 因此，仅在必要时包含处理将 CPU 张量移动到 CUDA 设备的代码。
# TODO: 将来一旦 TorchScript 语法改进，可以将这两个模板合并在一起。
    # 初始化一个空的元组，用于存储模型前向传播的输出结果
    ret: Tuple[()] = ()
    # 对于模型前向传播的每个输出值，如果是 Tensor 类型，则将其移到 CPU 上；否则保持原样
    for i in module.forward(*out_args, {kwargs}):
        i = (i.cpu(),) if isinstance(i, Tensor) else (i,)
        # 将处理过的输出值添加到 ret 元组中
        ret = ret + i
    # 返回所有处理过的模型输出结果组成的元组
    return ret
"""
定义一个模板字符串，用于生成远程调用函数 _remote_forward 的代码。

_REMOTE_FORWARD_TEMPLATE = """
def _remote_forward(
    module_rref: RRef[module_interface_cls], device: str, is_device_map_set: bool, {arg_types}){arrow_and_return_type}:
    module = module_rref.local_value()

    return module.forward({args}, {kwargs})
"""
"""

_REMOTE_FORWARD_TEMPLATE 是一个模板字符串，用于生成一个名为 _remote_forward 的函数定义。

def _remote_forward(
    module_rref: RRef[module_interface_cls], device: str, is_device_map_set: bool, {arg_types}){arrow_and_return_type}:

这里定义了一个函数 _remote_forward，它接受以下参数：
- module_rref: 类型为 RRef[module_interface_cls]，是一个远程引用，表示要调用的模块。
- device: 字符串类型，表示目标设备。
- is_device_map_set: 布尔类型，指示设备映射是否已设置。
- {arg_types}: 这是一个占位符，表示函数可能包含不定数量和类型的参数，具体参数类型在实例化模板时定义。

{arrow_and_return_type}:

这一行用于定义函数的返回类型或者箭头 "->" 和返回类型，具体的返回类型在实例化模板时定义。

    module = module_rref.local_value()

获取 module_rref 的本地值，即从远程引用中获取本地模块对象。

    return module.forward({args}, {kwargs})

调用 module 对象的 forward 方法，并传递参数 args 和 kwargs，这些参数具体值在实例化模板时定义。
"""
```