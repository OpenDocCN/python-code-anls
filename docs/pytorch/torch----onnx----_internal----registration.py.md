# `.\pytorch\torch\onnx\_internal\registration.py`

```
# mypy: allow-untyped-defs
"""Module for handling symbolic function registration."""

import warnings
from typing import (
    Callable,
    Collection,
    Dict,
    Generic,
    Optional,
    Sequence,
    Set,
    TypeVar,
    Union,
)

from torch.onnx import _constants, errors
from torch.onnx._internal import _beartype

OpsetVersion = int


def _dispatch_opset_version(
    target: OpsetVersion, registered_opsets: Collection[OpsetVersion]
) -> Optional[OpsetVersion]:
    """Finds the registered opset given a target opset version and the available opsets.

    Args:
        target: The target opset version.
        registered_opsets: The available opsets.

    Returns:
        The registered opset version.
    """
    # 如果没有注册的 opset，则返回 None
    if not registered_opsets:
        return None

    # 按照降序对已注册的 opset 进行排序
    descending_registered_versions = sorted(registered_opsets, reverse=True)
    # 线性搜索 opset 版本，因为 opset 的数量很少，所以这种方法效率可以接受。

    if target >= _constants.ONNX_BASE_OPSET:
        # 当目标 opset 大于等于 ONNX_BASE_OPSET (opset 9) 时，始终向下查找到 opset 1。
        # 当自定义 op 在 opset 1 注册时，我们希望能够在所有大于等于 ONNX_BASE_OPSET 的 opset 中作为后备发现它们。
        for version in descending_registered_versions:
            if version <= target:
                return version
        return None

    # target < opset 9。这是支持 opset 7 和 opset 8 的旧行为，用于支持 caffe2。
    # 我们向上搜索，直到 opset 9。
    for version in reversed(descending_registered_versions):
        # 逆向计数，直到 _constants.ONNX_BASE_OPSET
        if target <= version <= _constants.ONNX_BASE_OPSET:
            return version

    return None


_K = TypeVar("_K")
_V = TypeVar("_V")


class OverrideDict(Collection[_K], Generic[_K, _V]):
    """A dictionary that merges built-in and custom symbolic functions.

    It supports overriding and un-overriding built-in symbolic functions with custom
    ones.
    """

    def __init__(self):
        # 初始化基础字典、覆盖字典和合并字典
        self._base: Dict[_K, _V] = {}
        self._overrides: Dict[_K, _V] = {}
        self._merged: Dict[_K, _V] = {}

    def set_base(self, key: _K, value: _V) -> None:
        # 设置基础字典中的键值对，并在未覆盖的情况下更新合并字典
        self._base[key] = value
        if key not in self._overrides:
            self._merged[key] = value

    def in_base(self, key: _K) -> bool:
        """Checks if a key is in the base dictionary."""
        # 检查键是否在基础字典中
        return key in self._base

    def override(self, key: _K, value: _V) -> None:
        """Overrides a base key-value with a new pair."""
        # 覆盖基础字典中的键值对，并更新合并字典
        self._overrides[key] = value
        self._merged[key] = value

    def remove_override(self, key: _K) -> None:
        """Un-overrides a key-value pair."""
        # 取消覆盖指定键的键值对，并更新合并字典；如果键存在于基础字典中，则将其添加回合并字典
        self._overrides.pop(key, None)  # type: ignore[arg-type]
        self._merged.pop(key, None)  # type: ignore[arg-type]
        if key in self._base:
            self._merged[key] = self._base[key]
    # 检查是否有给定键的值被覆盖
    def overridden(self, key: _K) -> bool:
        return key in self._overrides

    # 获取指定键的值
    def __getitem__(self, key: _K) -> _V:
        return self._merged[key]

    # 获取指定键的值，若不存在返回默认值
    def get(self, key: _K, default: Optional[_V] = None):
        return self._merged.get(key, default)

    # 检查是否包含指定的键
    def __contains__(self, key: object) -> bool:
        return key in self._merged

    # 返回一个迭代器，用于迭代所有键
    def __iter__(self):
        return iter(self._merged)

    # 返回合并字典的长度
    def __len__(self) -> int:
        return len(self._merged)

    # 返回表示对象的字符串，用于调试和输出
    def __repr__(self) -> str:
        return f"OverrideDict(base={self._base}, overrides={self._overrides})"

    # 检查合并字典是否为真（非空）
    def __bool__(self) -> bool:
        return bool(self._merged)
class _SymbolicFunctionGroup:
    """Different versions of symbolic functions registered to the same name.

    O(number of registered versions of an op) search is performed to find the most
    recent version of the op.

    The registration is delayed until op is used to improve startup time.

    Function overloads with different arguments are not allowed.
    Custom op overrides are supported.
    """

    def __init__(self, name: str) -> None:
        # Initialize the _SymbolicFunctionGroup with a given name.
        self._name = name
        # A dictionary of functions, keyed by the opset version.
        self._functions: OverrideDict[OpsetVersion, Callable] = OverrideDict()

    def __repr__(self) -> str:
        # Return a string representation of _SymbolicFunctionGroup.
        return f"_SymbolicFunctionGroup({self._name}, registered={self._functions})"

    def __getitem__(self, key: OpsetVersion) -> Callable:
        # Retrieve the function corresponding to the given opset version.
        result = self.get(key)
        if result is None:
            raise KeyError(key)
        return result

    # TODO(justinchuby): Add @functools.lru_cache(maxsize=None) if lookup time becomes
    # a problem.
    def get(self, opset: OpsetVersion) -> Optional[Callable]:
        """Find the most recent version of the function."""
        # Determine the most recent version of the function for a given opset.
        version = _dispatch_opset_version(opset, self._functions)
        if version is None:
            return None

        return self._functions[version]

    def add(self, func: Callable, opset: OpsetVersion) -> None:
        """Adds a symbolic function.

        Args:
            func: The function to add.
            opset: The opset version of the function to add.
        """
        # Add a symbolic function to the registry.
        if self._functions.in_base(opset):
            warnings.warn(
                f"Symbolic function '{self._name}' already registered for opset {opset}. "
                f"Replacing the existing function with new function. This is unexpected. "
                f"Please report it on {_constants.PYTORCH_GITHUB_ISSUES_URL}.",
                errors.OnnxExporterWarning,
            )
        self._functions.set_base(opset, func)

    def add_custom(self, func: Callable, opset: OpsetVersion) -> None:
        """Adds a custom symbolic function.

        Args:
            func: The symbolic function to register.
            opset: The corresponding opset version.
        """
        # Add a custom symbolic function to the registry.
        self._functions.override(opset, func)

    def remove_custom(self, opset: OpsetVersion) -> None:
        """Removes a custom symbolic function.

        Args:
            opset: The opset version of the custom function to remove.
        """
        # Remove a custom symbolic function from the registry.
        if not self._functions.overridden(opset):
            warnings.warn(
                f"No custom function registered for '{self._name}' opset {opset}"
            )
            return
        self._functions.remove_override(opset)

    def get_min_supported(self) -> OpsetVersion:
        """Returns the lowest built-in opset version supported by the function."""
        # Retrieve the minimum built-in opset version supported by the function.
        return min(self._functions)


class SymbolicRegistry:
    """Registry for symbolic functions.
    The registry maintains a mapping from qualified names to symbolic functions.
    It is used to register new symbolic functions and to dispatch calls to
    the appropriate function.
    """

    # 初始化函数，创建一个空的注册表字典，用于存储符号函数组对象
    def __init__(self) -> None:
        self._registry: Dict[str, _SymbolicFunctionGroup] = {}

    # 注册符号函数的方法
    def register(
        self, name: str, opset: OpsetVersion, func: Callable, custom: bool = False
    ) -> None:
        """Registers a symbolic function.

        Args:
            name: The qualified name of the function to register. In the form of 'domain::op'.
                E.g. 'aten::add'.
            opset: The opset version of the function to register.
            func: The symbolic function to register.
            custom: Whether the function is a custom function that overrides existing ones.

        Raises:
            ValueError: If the separator '::' is not in the name.
        """
        # 检查名称是否符合 'domain::op' 的格式，如果不符则抛出异常
        if "::" not in name:
            raise ValueError(
                f"The name must be in the form of 'domain::op', not '{name}'"
            )
        # 获取名称对应的符号函数组对象，如果不存在则创建新的符号函数组对象
        symbolic_functions = self._registry.setdefault(
            name, _SymbolicFunctionGroup(name)
        )
        # 根据 custom 参数决定是添加普通符号函数还是自定义符号函数到符号函数组对象中
        if custom:
            symbolic_functions.add_custom(func, opset)
        else:
            symbolic_functions.add(func, opset)

    # 取消注册符号函数的方法
    def unregister(self, name: str, opset: OpsetVersion) -> None:
        """Unregisters a symbolic function.

        Args:
            name: The qualified name of the function to unregister.
            opset: The opset version of the function to unregister.
        """
        # 如果名称不存在于注册表中，则直接返回
        if name not in self._registry:
            return
        # 从符号函数组对象中移除指定 opset 版本的自定义函数
        self._registry[name].remove_custom(opset)

    # 获取给定名称对应的符号函数组对象的方法
    def get_function_group(self, name: str) -> Optional[_SymbolicFunctionGroup]:
        """Returns the function group for the given name."""
        return self._registry.get(name)

    # 判断指定名称和版本的 op 是否已注册的方法
    def is_registered_op(self, name: str, version: int) -> bool:
        """Returns whether the given op is registered for the given opset version."""
        # 获取名称对应的符号函数组对象
        functions = self.get_function_group(name)
        # 如果符号函数组对象不存在，则返回 False；否则判断给定版本的函数是否存在
        if functions is None:
            return False
        return functions.get(version) is not None

    # 返回所有已注册函数名称的集合的方法
    def all_functions(self) -> Set[str]:
        """Returns the set of all registered function names."""
        return set(self._registry)
# 使用 @_beartype 装饰器，确保函数参数的类型正确性
@_beartype.beartype
# 定义一个装饰器函数 onnx_symbolic，用于注册符号函数
def onnx_symbolic(
    # 函数名称，形如 'domain::op' 的限定名称
    name: str,
    # 操作集版本号或版本号序列，可以是 OpsetVersion 或 Sequence[OpsetVersion]
    opset: Union[OpsetVersion, Sequence[OpsetVersion]],
    # 可选的装饰器序列，应用于函数上的修饰器
    decorate: Optional[Sequence[Callable]] = None,
    # 是否是自定义符号函数，默认为 False
    custom: bool = False,
) -> Callable:
    """Registers a symbolic function.

    Usage::

    ```
    @onnx_symbolic("aten::symbolic_b", opset=10, decorate=[quantized_aten_handler(scale=1/128, zero_point=0)])
    @symbolic_helper.parse_args("v", "v", "b")
    def symbolic_b(g: _C.Graph, x: _C.Value, y: _C.Value, arg1: bool) -> _C.Value:
        ...
    ```

    Args:
        name: The qualified name of the function in the form of 'domain::op'.
            E.g. 'aten::add'.
        opset: The opset versions of the function to register at.
        decorate: A sequence of decorators to apply to the function.
        custom: Whether the function is a custom symbolic function.

    Raises:
        ValueError: If the separator '::' is not in the name.
    """

    # 定义包装器函数，接受一个函数作为参数，并返回一个装饰后的函数
    def wrapper(func: Callable) -> Callable:
        # 初始装饰后的函数为传入的 func
        decorated = func
        # 如果存在 decorate 参数，遍历 apply 每一个装饰器函数到 decorated 上
        if decorate is not None:
            for decorate_func in decorate:
                decorated = decorate_func(decorated)

        # 引用全局变量 registry
        global registry
        # 使用 nonlocal 关键字引用 opset 变量
        nonlocal opset
        # 如果 opset 是单个 OpsetVersion，则转换为元组形式
        if isinstance(opset, OpsetVersion):
            opset = (opset,)
        # 遍历 opset 中的每个 opset_version，注册函数到 registry 中
        for opset_version in opset:
            registry.register(name, opset_version, decorated, custom=custom)

        # 返回原始函数 func，因为 decorate 中的装饰器仅适用于当前注册的实例
        return func

    # 返回包装器函数 wrapper
    return wrapper


# 使用 @_beartype 装饰器，确保函数参数的类型正确性
@_beartype.beartype
# 定义一个自定义符号函数的装饰器 custom_onnx_symbolic
def custom_onnx_symbolic(
    # 函数名称，形如 'domain::op' 的限定名称
    name: str,
    # 操作集版本号或版本号序列，可以是 OpsetVersion 或 Sequence[OpsetVersion]
    opset: Union[OpsetVersion, Sequence[OpsetVersion]],
    # 可选的装饰器序列，应用于函数上的修饰器
    decorate: Optional[Sequence[Callable]] = None,
) -> Callable:
    """Registers a custom symbolic function.

    Args:
        name: the qualified name of the function.
        opset: the opset version of the function.
        decorate: a sequence of decorators to apply to the function.

    Returns:
        The decorator.

    Raises:
        ValueError: If the separator '::' is not in the name.
    """
    # 调用 onnx_symbolic 函数，传入 custom=True，注册自定义符号函数
    return onnx_symbolic(name, opset, decorate, custom=True)


# 用于存储所有符号函数的注册表
registry = SymbolicRegistry()
```