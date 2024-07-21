# `.\pytorch\torch\nn\utils\_named_member_accessor.py`

```
# 导入必要的模块和类型
from typing import Dict, Iterable, List, Tuple

import torch  # 导入 PyTorch 库

# 定义一个特殊的对象作为缺失值的标志
_MISSING: torch.Tensor = object()  # type: ignore[assignment]

# 设置指定模块中的张量
def set_tensor(module: "torch.nn.Module", name: str, tensor: torch.Tensor) -> None:
    # 检查 module 是否为 torch.nn.Module 的实例
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"{module} is not an instance of torch.nn.Module")
    # 检查 tensor 是否为 torch.Tensor 的实例，如果 tensor 不为 None 的话
    if not isinstance(tensor, torch.Tensor) and tensor is not None:
        raise TypeError(f"{tensor} is not an instance of torch.Tensor")
    # 检查名字中是否包含点号
    if "." in name:
        raise KeyError('tensor name can\'t contain "."')
    # 检查名字是否为空字符串
    if name == "":
        raise KeyError('tensor name can\'t be empty string ""')
    
    # 如果名字在模块的参数中，则设置模块参数为给定的张量
    if name in module._parameters:
        module._parameters[name] = tensor  # type: ignore[assignment]
    # 如果名字在模块的缓冲区中，则设置模块缓冲区为给定的张量
    elif name in module._buffers:
        module._buffers[name] = tensor
    # 否则将名字作为模块的属性，并设置为给定的张量
    else:
        setattr(module, name, tensor)

# 交换指定模块中的张量，并返回原始的张量
def swap_tensor(
    module: "torch.nn.Module",
    name: str,
    tensor: torch.Tensor,
    allow_missing: bool = False,
) -> torch.Tensor:
    # 检查 module 是否为 torch.nn.Module 的实例
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"{module} is not an instance of torch.nn.Module")
    # 检查 tensor 是否为 torch.Tensor 的实例，或者是否为缺失值标志，或者是否为 None
    if (
        tensor is not _MISSING
        and not isinstance(tensor, torch.Tensor)
        and tensor is not None
    ):
        raise TypeError(f"{tensor} is not an instance of torch.Tensor")
    # 检查名字中是否包含点号
    if "." in name:
        raise KeyError('tensor name can\'t contain "."')
    # 检查名字是否为空字符串
    if name == "":
        raise KeyError('tensor name can\'t be empty string ""')

    orig_tensor: torch.Tensor  # 声明原始张量变量

    # 如果名字在模块的参数中，则获取原始参数并设置新的参数
    if name in module._parameters:
        orig_tensor = module._parameters[name]  # type: ignore[assignment]
        if tensor is not _MISSING:
            module._parameters[name] = tensor  # type: ignore[assignment]
        else:
            del module._parameters[name]  # 如果 tensor 是缺失值，则删除该参数
    # 如果名字在模块的缓冲区中，则获取原始缓冲区并设置新的缓冲区
    elif name in module._buffers:
        orig_tensor = module._buffers[name]  # type: ignore[assignment]
        if tensor is not _MISSING:
            module._buffers[name] = tensor
        else:
            del module._buffers[name]  # 如果 tensor 是缺失值，则删除该缓冲区
    else:
        try:
            orig_tensor = getattr(module, name)  # 尝试获取模块中的属性
        except AttributeError as ex:
            if not allow_missing:
                raise AttributeError(
                    f"{module._get_name()} has no attribute `{name}`"
                ) from ex
            orig_tensor = _MISSING
        # 如果原始张量不是缺失值，并且不是 torch.Tensor 的实例，则抛出类型错误
        if (
            orig_tensor is not _MISSING
            and not isinstance(orig_tensor, torch.Tensor)
            and orig_tensor is not None
        ):
            raise TypeError(
                f"attribute `{name}`: {orig_tensor} is not an instance of torch.Tensor"
            )
        # 如果 tensor 不是缺失值，则设置模块中的属性为给定的张量；如果 tensor 是缺失值，并且模块中有该属性，则删除该属性
        if tensor is not _MISSING:
            setattr(module, name, tensor)
        elif hasattr(module, name):
            delattr(module, name)

    return orig_tensor  # 返回原始的张量
    submodule: "torch.nn.Module",


    # 定义一个变量 submodule，其值是字符串 "torch.nn.Module"
    submodule: "torch.nn.Module",
def swap_submodule(module: "torch.nn.Module", submodule: "torch.nn.Module", name: str) -> "torch.nn.Module":
    # 确保 module 是 torch.nn.Module 的实例，否则抛出类型错误异常
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"{module} is not an instance of torch.nn.Module")
    # 确保 submodule 是 torch.nn.Module 的实例，否则抛出类型错误异常
    if not isinstance(submodule, torch.nn.Module):
        raise TypeError(f"{submodule} is not an instance of torch.nn.Module")
    # 如果 name 中包含 "."，则抛出键错误异常，因为子模块名称不能包含 "."
    if "." in name:
        raise KeyError('submodule name can\'t contain "."')
    # 如果 name 是空字符串，则抛出键错误异常，因为子模块名称不能为空
    if name == "":
        raise KeyError('submodule name can\'t be empty string ""')
    # 如果 name 不在 module._modules 中，则抛出键错误异常，因为子模块不存在
    if name not in module._modules:
        raise KeyError(f"submodule {name} does not exist")

    # 获取原始的子模块
    orig_submodule = module._modules[name]
    # 确保原始子模块是 torch.nn.Module 的实例，否则抛出类型错误异常
    if not isinstance(orig_submodule, torch.nn.Module):
        raise TypeError(f"{name} attribute is not an instance of torch.nn.Module")
    
    # 将 module._modules[name] 替换为新的 submodule
    module._modules[name] = submodule
    # 返回原始的子模块
    return orig_submodule


class NamedMemberAccessor:
    """
    A class that provides a way to access the submodules and parameters/buffers of a module.

    It provides caching mechanism to speed up submodule lookups.
    This is useful for functional programming to manipulate the module state.
    """

    def __init__(self, module: "torch.nn.Module") -> None:
        # 初始化 NamedMemberAccessor 实例，接受一个 torch.nn.Module 作为参数
        self.module = module
        # 创建一个空的字典 memo，用于缓存子模块
        self.memo: Dict[str, torch.nn.Module] = {}

    # Nested attribute access

    def get_submodule(self, name: str) -> "torch.nn.Module":
        """
        Return the submodule specified by the given path.

        For example, to get the submodule mod.layer1.conv1,
        use accessor.get_submodule("layer1.conv1")

        Compare to mod.get_submodule("layer1.conv1"), this method will cache the
        intermediate submodule access to speed up future lookups.
        """
        # 如果 name 是空字符串，则直接返回 module
        if not name:
            return self.module

        try:
            # 尝试从缓存中获取子模块
            return self.memo[name]
        except KeyError:
            # 如果缓存中不存在，则根据路径获取子模块
            prefix, dot, attr = name.rpartition(".")
            if dot:
                module = self.get_submodule(prefix)
            else:
                module = self.module
            try:
                submodule = getattr(module, attr)
            except AttributeError as ex:
                # 如果属性不存在，则抛出属性错误异常
                raise AttributeError(
                    f"{module._get_name()} has no attribute `{attr}`"
                ) from ex
            # 确保获取到的 submodule 是 torch.nn.Module 的实例，否则抛出类型错误异常
            if not isinstance(submodule, torch.nn.Module):
                raise TypeError(
                    f"submodule `{name}`: {submodule} is not an instance of torch.nn.Module"
                )
            # 将获取到的 submodule 放入缓存 memo 中
            self.memo[name] = submodule
            return submodule

    def swap_submodule(self, path: str, value: "torch.nn.Module") -> "torch.nn.Module":
        """
        Swap the submodule specified by the given ``path`` to ``value``.

        For example, to swap the attribute mod.layer1.conv1 use
        ``accessor.swap_submodule("layer1.conv1", conv2)``.
        """
        # 从路径中获取前缀和属性名
        prefix, _, attr = path.rpartition(".")
        # 调用 get_submodule 方法获取子模块，并调用 swap_submodule 函数进行替换
        return swap_submodule(self.get_submodule(prefix), attr, value)
    # 获取指定路径处的张量对象

    """
    获取指定路径处的张量对象。

    例如，要获取属性 mod.layer1.conv1.weight，
    使用 accessor.get_tensor('layer1.conv1.weight')。

    与 mod.get_parameter("layer1.conv1.weight") 相比，该方法会缓存中间子模块访问，以加速未来的查找。
    """
    prefix, _, attr = name.rpartition(".")
    # 获取指定路径的前缀，并分离出最后的属性名称
    submodule = self.get_submodule(prefix)
    # 获取子模块对象
    try:
        tensor = getattr(submodule, attr)
        # 尝试获取指定属性的张量对象
    except AttributeError as ex:
        raise AttributeError(
            f"{submodule._get_name()} has no attribute `{name}`"
        ) from ex
        # 如果属性不存在，则抛出属性错误异常
    if not isinstance(tensor, torch.Tensor) and tensor is not None:
        raise TypeError(f"{tensor} is not an instance of torch.Tensor")
        # 如果获取的不是张量对象且不为 None，则抛出类型错误异常
    return tensor  # type: ignore[return-value]
    # 返回获取到的张量对象

    # 设置指定路径处的张量属性为给定值

    """
    设置指定路径处的张量属性为给定值。

    例如，要设置属性 mod.layer1.conv1.weight，
    使用 accessor.set_tensor("layer1.conv1.weight", value)。
    """
    prefix, _, attr = name.rpartition(".")
    # 获取指定路径的前缀，并分离出最后的属性名称
    set_tensor(self.get_submodule(prefix), attr, value)
    # 调用外部函数设置子模块中指定属性的张量值为给定值

    # 删除指定路径处的张量属性

    """
    删除指定路径处的张量属性。

    例如，要删除属性 mod.layer1.conv1.weight，
    使用 accessor.del_tensor("layer1.conv1.weight")。
    """
    prefix, _, attr = name.rpartition(".")
    # 获取指定路径的前缀，并分离出最后的属性名称
    submodule = self.get_submodule(prefix)
    # 获取子模块对象
    try:
        delattr(submodule, attr)
        # 尝试删除子模块中指定的属性
    except AttributeError as ex:
        raise AttributeError(
            f"{submodule._get_name()} has no attribute `{name}`"
        ) from ex
        # 如果属性不存在，则抛出属性错误异常

    # 批量操作

    # 获取指定路径列表中的所有张量对象

    """
    获取指定路径列表中的所有张量对象。

    例如，要获取属性 mod.layer1.conv1.weight 和 mod.layer1.conv1.bias，
    使用 accessor.get_tensors(["layer1.conv1.weight", "layer1.conv1.bias"])。
    """
    return [self.get_tensor(name) for name in names]
    # 返回所有指定路径处的张量对象列表
    def set_tensors(self, names: Iterable[str], values: Iterable[torch.Tensor]) -> None:
        """
        Set the attributes specified by the given paths to values.

        For example, to set the attributes mod.layer1.conv1.weight and
        mod.layer1.conv1.bias, use accessor.set_tensors(["layer1.conv1.weight",
        "layer1.conv1.bias"], [weight, bias])
        """
        # 如果names不是list或tuple类型，则转换为list
        if not isinstance(names, (list, tuple)):
            names = list(names)
        # 如果values不是list或tuple类型，则转换为list
        if not isinstance(values, (list, tuple)):
            values = list(values)
        # 断言names和values长度相同
        assert len(names) == len(values), "names and values must have the same length"

        # 遍历names和values，调用self.set_tensor方法逐一设置属性值
        for name, value in zip(names, values):
            self.set_tensor(name, value)

    def set_tensors_dict(self, named_tensors: Dict[str, torch.Tensor]) -> None:
        """
        Set the attributes specified by the given paths to values.

        For example, to set the attributes mod.layer1.conv1.weight and
        mod.layer1.conv1.bias, use accessor.set_tensors_dict({
            "layer1.conv1.weight": weight,
            "layer1.conv1.bias": bias,
        })
        """
        # 遍历named_tensors字典，调用self.set_tensor方法逐一设置属性值
        for name, value in named_tensors.items():
            self.set_tensor(name, value)

    def del_tensors(self, names: Iterable[str]) -> None:
        """
        Delete the attributes specified by the given paths.

        For example, to delete the attributes mod.layer1.conv1.weight and
        mod.layer1.conv1.bias, use accessor.del_tensors(["layer1.conv1.weight",
        "layer1.conv1.bias"])
        """
        # 遍历names列表，调用self.del_tensor方法逐一删除属性
        for name in names:
            self.del_tensor(name)

    def swap_tensors(
        self,
        names: Iterable[str],
        values: Iterable[torch.Tensor],
        allow_missing: bool = False,
    ) -> List[torch.Tensor]:
        """
        Swap the attributes specified by the given paths to values.

        For example, to swap the attributes mod.layer1.conv1.weight and
        mod.layer1.conv1.bias, use accessor.swap_tensors(["layer1.conv1.weight",
        "layer1.conv1.bias"], [weight, bias])
        """
        # 如果names不是list或tuple类型，则转换为list
        if not isinstance(names, (list, tuple)):
            names = list(names)
        # 如果values不是list或tuple类型，则转换为list
        if not isinstance(values, (list, tuple)):
            values = list(values)
        # 断言names和values长度相同
        assert len(names) == len(values), "names and values must have the same length"

        # 调用self.swap_tensor方法逐一交换属性值，并返回结果列表
        return [
            self.swap_tensor(name, value, allow_missing=allow_missing)
            for name, value in zip(names, values)
        ]

    def swap_tensors_dict(
        self, named_tensors: Dict[str, torch.Tensor], allow_missing: bool = False
    ) -> List[torch.Tensor]:
        """
        Swap the attributes specified by the given paths to values.

        For example, to swap the attributes mod.layer1.conv1.weight and
        mod.layer1.conv1.bias, use accessor.swap_tensors_dict({
            "layer1.conv1.weight": weight,
            "layer1.conv1.bias": bias,
        })
        """
        # 调用self.swap_tensor方法逐一交换属性值，并返回结果列表
        return [
            self.swap_tensor(name, value, allow_missing=allow_missing)
            for name, value in named_tensors.items()
        ]
    ) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        """
        Swap the attributes specified by the given paths to values.

        For example, to swap the attributes mod.layer1.conv1.weight and
        mod.layer1.conv1.bias, use accessor.swap_tensors_dict({
            "layer1.conv1.weight": weight,
            "layer1.conv1.bias": bias,
        })
        """
        # 创建一个空字典，用于存储原始张量的名称和值
        orig_named_tensors = {}
        # 创建一个空列表，用于存储找不到的键
        missing_keys = []
        try:
            # 遍历命名张量字典中的每个项
            for name, tensor in named_tensors.items():
                # 调用 swap_tensor 方法来交换指定路径处的属性值
                orig_tensor = self.swap_tensor(name, tensor, allow_missing=True)
                # 如果原始张量是 _MISSING，将该名称添加到 missing_keys 列表中
                if orig_tensor is _MISSING:
                    missing_keys.append(name)
                # 将交换后的原始张量添加到 orig_named_tensors 字典中
                orig_named_tensors[name] = orig_tensor
        except Exception:
            # 如果发生任何异常，回滚所有已交换的张量
            for name, orig_tensor in orig_named_tensors.items():
                self.swap_tensor(name, orig_tensor, allow_missing=True)
            # 抛出异常继续传播
            raise
        if missing_keys and not allow_missing:
            # 如果 allow_missing 为 False 且有缺失的键，回滚所有已交换的张量
            for name, orig_tensor in orig_named_tensors.items():
                self.swap_tensor(name, orig_tensor, allow_missing=True)
            # 抛出 RuntimeError 异常，指示缺失的键
            raise RuntimeError(f"Missing key(s): {', '.join(map(repr, missing_keys))}.")
        # 返回交换后的原始张量字典和找不到的键列表
        return orig_named_tensors, missing_keys

    def check_keys(self, keys: Iterable[str]) -> Tuple[List[str], List[str]]:
        """Check that the given keys are valid."""
        # 将 keys 转换为集合
        keys = set(keys)
        # 获取当前模块中所有命名张量的名称集合
        valid_keys = {name for name, _ in self.named_tensors(remove_duplicate=False)}
        # 计算缺失的键（在模块中存在但未在给定的键集合中）
        missing_keys = valid_keys - keys
        # 计算意外的键（在给定的键集合中但模块中不存在）
        unexpected_keys = keys - valid_keys
        # 返回缺失的键列表和意外的键列表（均已排序）
        return sorted(missing_keys), sorted(unexpected_keys)

    # Shortcut methods

    def named_parameters(
        self,
        remove_duplicate: bool = True,
    ) -> Iterable[Tuple[str, torch.Tensor]]:
        """Iterate over all the parameters in the module."""
        # 委托给模块中的 named_parameters 方法，并返回迭代器
        yield from self.module.named_parameters(remove_duplicate=remove_duplicate)

    def named_buffers(
        self,
        remove_duplicate: bool = True,
    ) -> Iterable[Tuple[str, torch.Tensor]]:
        """Iterate over all the buffers in the module."""
        # 委托给模块中的 named_buffers 方法，并返回迭代器
        yield from self.module.named_buffers(remove_duplicate=remove_duplicate)

    def named_tensors(
        self,
        remove_duplicate: bool = True,
    ) -> Iterable[Tuple[str, torch.Tensor]]:
        """Iterate over all the tensors in the module."""
        # 委托给模块中的 named_parameters 和 named_buffers 方法，并返回迭代器
        yield from self.module.named_parameters(remove_duplicate=remove_duplicate)
        yield from self.module.named_buffers(remove_duplicate=remove_duplicate)

    def named_modules(
        self,
        remove_duplicate: bool = True,
    ) -> Iterable[Tuple[str, "torch.nn.Module"]]:
        """Iterate over all the modules in the module."""
        # 委托给模块中的 named_modules 方法，并返回迭代器
        yield from self.module.named_modules(remove_duplicate=remove_duplicate)
```