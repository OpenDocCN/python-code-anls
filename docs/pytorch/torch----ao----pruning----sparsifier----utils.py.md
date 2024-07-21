# `.\pytorch\torch\ao\pruning\sparsifier\utils.py`

```
# mypy: allow-untyped-defs
# 允许未类型化的函数定义

from typing import Any, Dict, Optional, Type
# 引入类型提示相关的模块和类

from torch.nn.utils.parametrize import type_before_parametrizations, is_parametrized
# 从 torch.nn.utils.parametrize 模块中导入 type_before_parametrizations 和 is_parametrized 函数

from itertools import chain
# 导入 itertools 模块中的 chain 函数

from torch import nn
# 从 torch 模块中导入 nn 模块

__all__ = [
    "module_contains_param",
    "swap_module",
    "module_to_fqn",
    "fqn_to_module",
    "get_arg_info_from_tensor_fqn",
    "FakeSparsity",
]
# 模块公开的所有接口，包括 module_contains_param, swap_module, module_to_fqn, fqn_to_module, get_arg_info_from_tensor_fqn, FakeSparsity

def module_contains_param(module: nn.Module, parametrization: Type[nn.Module]) -> bool:
    """
    Check if the module contains parameters of a specific parametrization type.
    """
    if is_parametrized(module):
        # 如果模块被参数化了
        return any(
            any(isinstance(param, parametrization) for param in param_list)
            for key, param_list in module.parametrizations.items()  # type: ignore[union-attr,operator]
        )
    return False
    # 如果模块未被参数化，则返回 False

def swap_module(
    mod: nn.Module, mapping: Dict[Type[nn.Module], Type[nn.Module]]
) -> nn.Module:
    """
    Swap a module with its sparse equivalent based on the mapping.
    """
    if type_before_parametrizations(mod) in mapping:
        # 如果模块在映射中有对应的稀疏模块
        sparse_mod = mapping[type_before_parametrizations(mod)]

        # TODO Fix this typing, as Type[Module] has no attribute "from_dense"
        new_mod = sparse_mod.from_dense(mod)  # type: ignore[attr-defined]

        # Preserve module's pre forward hooks. They'll be called on quantized input
        for pre_hook_fn in mod._forward_pre_hooks.values():
            new_mod.register_forward_pre_hook(pre_hook_fn)
        # Preserve module's post forward hooks except _observer_forward_hook
        # After convert they'll work with quantized output
        for hook_fn in mod._forward_hooks.values():
            new_mod.register_forward_hook(hook_fn)

        # respect device affinity when swapping modules
        devices = {p.device for p in chain(mod.parameters(), mod.buffers())}
        assert len(devices) <= 1, (
            f"swap_module only works with cpu or single-device CUDA modules, but got devices {devices}"
        )
        device = next(iter(devices)) if len(devices) > 0 else None
        if device:
            new_mod.to(device)

        return new_mod
        # 返回新的稀疏模块

    else:
        return mod
        # 如果模块不在映射中，则返回原始模块

def module_to_fqn(
    model: nn.Module, module: nn.Module, prefix: str = ""
) -> Optional[str]:
    """
    Convert a module to its fully qualified name (fqn) within a model.
    Returns None if the module is not a descendant of the model.
    """
    if module is model:
        return ""
        # 如果模块就是给定的模型本身，则返回空字符串
    for name, child in model.named_children():
        fqn = module_to_fqn(child, module, ".")
        if isinstance(fqn, str):
            return prefix + name + fqn
            # 如果找到模块，则返回模块的全名（带前缀）
    return None
    # 如果模块不是模型的后代，则返回 None

def fqn_to_module(model: Optional[nn.Module], path: str) -> Optional[nn.Module]:
    """
    Retrieve a module from a model using its fully qualified name (fqn).
    Returns None if the module does not exist in the model.
    """
    # 根据给定的完全限定名（fqn），返回对应的模块或张量，如果路径（path）对应的内容不存在则返回 None。
    # 类似于 model.get_submodule(path)，但适用于张量。
    
    if path != "":
        # 如果路径不为空字符串，则按点分割路径，并逐级获取模块或张量
        for name in path.split("."):
            # 使用 getattr 获取 model 对象中名为 name 的属性（模块或张量），若不存在则返回 None
            model = getattr(model, name, None)
    
    # 返回最终获取到的模块或张量，或者 None（如果路径为空或未找到对应的模块或张量）
    return model
# 使用 tensor_fqn 参数来获取一个包含 module_fqn、module 和 tensor_name 的字典
def get_arg_info_from_tensor_fqn(model: nn.Module, tensor_fqn: str) -> Dict[str, Any]:
    """
    使用 tensor_fqn 获取包含 module_fqn、module 和 tensor_name 的字典
    """
    # 将 tensor_fqn 按最后一个 '.' 分割，得到 tensor_name
    # 如果 tensor_fqn 是 'weight'，则 module_fqn 为空字符串，tensor_name 是 'weight'
    # 如果 tensor_fqn 是 'linear.weight'，则 module_fqn 是 'linear'，tensor_name 是 'weight'
    tensor_name = tensor_fqn.split(".")[-1]
    # 根据 tensor_name 计算 module_fqn，即去掉 tensor_name 和可能存在的最后一个 '.'
    module_fqn = tensor_fqn[: -len(tensor_name) - ("." in tensor_fqn)]

    # 调用 fqn_to_module 函数获取 module 对象
    module = fqn_to_module(model, module_fqn)

    # 返回包含 module_fqn、module、tensor_name 和 tensor_fqn 的字典
    return {
        "module_fqn": module_fqn,
        "module": module,
        "tensor_name": tensor_name,
        "tensor_fqn": tensor_fqn,
    }


# 参数化类 FakeSparsity
class FakeSparsity(nn.Module):
    r"""
    用于权重的参数化。应该附加到 'weight' 或其它需要应用掩码的参数上。

    注意::

        一旦传递了掩码，变量的 id 不应更改。掩码的内容可以更改，但掩码引用本身不应更改。
    """

    def __init__(self, mask):
        # 调用父类的构造方法
        super().__init__()
        # 注册缓冲区 'mask'，将传入的 mask 参数作为 buffer 注册
        self.register_buffer("mask", mask)

    def forward(self, x):
        # 断言掩码的形状与输入 x 的形状相同
        assert self.mask.shape == x.shape
        # 返回掩码乘以输入 x 的结果
        return self.mask * x

    def state_dict(self, *args, **kwargs):
        # 我们不希望参数化保存掩码。
        # 这样可以确保线性模块在其参数化的同时不会保存掩码。
        return {}  # 返回空字典，确保不保存任何状态信息
```