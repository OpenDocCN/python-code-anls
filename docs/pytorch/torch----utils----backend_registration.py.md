# `.\pytorch\torch\utils\backend_registration.py`

```py
# mypy: allow-untyped-defs
# 导入 PyTorch 库
import torch
# 从 torch.overrides 模块导入需要的函数
from torch.overrides import (
    handle_torch_function,
    has_torch_function_unary,
)
# 从 torch._C 模块导入私有函数
from torch._C import _rename_privateuse1_backend, _get_privateuse1_backend_name
# 导入类型提示
from typing import List, Optional, Union

# 定义模块公开的函数列表
__all__ = ["rename_privateuse1_backend", "generate_methods_for_privateuse1_backend"]

# 全局变量，存储私有后端名称的字符串
# 注意：使用全局变量 `_privateuse1_backend_name` 而非 `torch._C._get_privateuse1_backend_name()`，
# 因为后者会导致 torch.jit.script 报错。
_privateuse1_backend_name = "privateuseone"

def rename_privateuse1_backend(backend_name: str) -> None:
    r"""
    重命名 privateuse1 后端设备，使其在 PyTorch API 中更方便使用作为设备名称。

    步骤如下：

    (1)（在 C++ 中）实现各种 torch 操作的内核，并将其注册到 PrivateUse1 调度键上。
    (2)（在 Python 中）调用 torch.utils.rename_privateuse1_backend("foo")

    现在可以在 Python 中将 "foo" 作为普通设备字符串使用。

    注意：此 API 每个进程只能调用一次。尝试在设置后再次更改外部后端将导致错误。

    注意（AMP）：如果要在您的设备上支持 AMP，请注册自定义后端模块。
    后端必须使用 ``torch._register_device_module("foo", BackendModule)`` 注册一个自定义后端模块。
    BackendModule 需要具有以下 API：

    (1) ``get_amp_supported_dtype() -> List[torch.dtype]``
        获取 AMP 模式下在 "foo" 设备上支持的数据类型列表。

    注意（random）：如果要支持在设备上设置种子，BackendModule 需要具有以下 API：

    (1) ``_is_in_bad_fork() -> bool``
        如果当前处于 bad_fork 状态，则返回 True，否则返回 False。

    (2) ``manual_seed_all(seed int) -> None``
        设置生成随机数所用的种子。

    (3) ``device_count() -> int``
        返回 "foo" 设备的数量。

    (4) ``get_rng_state(device: Union[int, str, torch.device] = 'foo') -> Tensor``
        返回表示所有设备的随机数状态的 ByteTensor 列表。

    (5) ``set_rng_state(new_state: Tensor, device: Union[int, str, torch.device] = 'foo') -> None``
        设置指定 "foo" 设备的随机数生成器状态。

    以及一些通用函数：

    (1) ``is_available() -> bool``
        返回一个布尔值，指示当前 "foo" 是否可用。

    (2) ``current_device() -> int``
        返回当前选择设备的索引。

    详细信息请参阅 https://pytorch.org/tutorials/advanced/extend_dispatcher.html#get-a-dispatch-key-for-your-backend
    可查看现有示例 https://github.com/bdhirsh/pytorch_open_registration_example
    """
    Example::

        >>> # xdoctest: +SKIP("failing")
        >>> torch.utils.rename_privateuse1_backend("foo")
        # 调用私有函数 `_rename_privateuse1_backend`，用于更改私有后端名称
        # 假设正确实现了对应的 C++ 内核来实现 torch.ones 函数。
        >>> a = torch.ones(2, device="foo")
# 检查给定模块是否已经注册了指定属性，如果是则抛出运行时错误
def _check_register_once(module, attr):
    if hasattr(module, attr):
        raise RuntimeError(f"The custom device module of {module} has already been registered with {attr}")


# 根据指定的自定义后端名称和设备，返回设备的标准化索引
def _normalization_device(custom_backend_name: str, device: Optional[Union[int, str, torch.device]] = None) -> int:
    # 内部函数：获取当前设备索引
    def _get_current_device_index():
        _get_device_index = "current_device"
        # 如果 torch 中存在指定的自定义后端名称，并且具有 _get_device_index 方法
        if hasattr(torch, custom_backend_name) and \
                hasattr(getattr(torch, custom_backend_name), _get_device_index):
            return getattr(getattr(torch, custom_backend_name), _get_device_index)()
        else:
            # 默认设备索引为 0
            return 0

    # 如果未指定设备，则返回当前设备索引
    if device is None:
        return _get_current_device_index()
    # 如果设备参数是字符串类型，将其转换为 torch.device 对象再处理
    elif isinstance(device, str):
        device = torch.device(device)

    # 确保设备参数是 torch.device 类型或整数类型
    if isinstance(device, torch.device):
        # 如果设备类型与自定义后端名称不匹配，则抛出运行时错误
        if device.type != custom_backend_name:
            raise RuntimeError(f"Invalid device, must be {custom_backend_name} device")
        elif device.index is None:
            device_idx = _get_current_device_index()
        else:
            device_idx = device.index
    # 如果设备参数是整数类型，直接使用作为设备索引
    else:
        device_idx = device
    return device_idx


# 为指定的私有后端生成张量方法
def _generate_tensor_methods_for_privateuse1_backend(custom_backend_name: str) -> None:
    # 定义属性装饰器，用于检查张量是否属于指定的后端
    @property  # type: ignore[misc]
    def wrap_tensor_backend(self: torch.Tensor) -> bool:
        if has_torch_function_unary(self):
            # 如果张量有 Torch 函数支持，则调用处理 Torch 函数的方法
            return handle_torch_function(wrap_tensor_backend.__get__, (self,), self)  # type: ignore[attr-defined]
        # 否则直接判断张量的设备类型是否与自定义后端名称匹配
        return self.device.type == custom_backend_name

    # 检查注册指定后端方法是否仅注册一次
    _check_register_once(torch.Tensor, f'is_{custom_backend_name}')
    # 设置属性装饰器的名称
    wrap_tensor_backend.fget.__name__ = f'is_{custom_backend_name}'  # type: ignore[attr-defined]
    # 将属性装饰器设置到 torch.Tensor 类中
    setattr(torch.Tensor, f'is_{custom_backend_name}', wrap_tensor_backend)
    def wrap_tensor_to(self: torch.Tensor, device: Optional[Union[int, torch.device]] = None, non_blocking=False,
                       **kwargs) -> torch.Tensor:
        r"""Perform Tensor device conversion. Call the to operator implementation.

        .. note::
            If the ``self`` Tensor already
            has the correct :class:`torch.device`, then ``self`` is returned.
            Otherwise, the returned tensor is a copy of ``self`` with the desired :class:`torch.device`.

        Args:
            device (int, optional): if specified, all parameters will be copied to that device
            non_blocking (bool): If ``True`` and the source is in pinned memory,
                the copy will be asynchronous with respect to the host. Otherwise,
                the argument has no effect.
            **kwargs (dict): For compatibility, may contain the key ``memory_format`` argument.
        """
        # 如果 self Tensor 已经具有正确的 torch.device，直接返回 self
        if has_torch_function_unary(self):
            # 如果 self 有 Torch 函数重载，调用 Torch 函数处理
            return handle_torch_function(wrap_tensor_to, (self,), self, device=device, non_blocking=False, **kwargs)
        
        # 根据自定义后端名称和设备，规范化设备索引
        device_idx = _normalization_device(custom_backend_name, device)
        # 将 self Tensor 转换到指定的 torch.device
        return self.to(device=torch.device(f'{custom_backend_name}:{device_idx}'), non_blocking=non_blocking, **kwargs)

    # 检查并注册一次自定义后端名称到 torch.Tensor 的映射关系
    _check_register_once(torch.Tensor, custom_backend_name)
    # 设置 wrap_tensor_to 函数的名称为自定义后端名称
    wrap_tensor_to.__name__ = custom_backend_name
    # 将 wrap_tensor_to 函数作为自定义后端名称的属性绑定到 torch.Tensor 上
    setattr(torch.Tensor, custom_backend_name, wrap_tensor_to)
def _generate_module_methods_for_privateuse1_backend(custom_backend_name: str) -> None:
    # 根据自定义后端名称生成模块属性和方法，依赖于Tensor方法，
    # 因此需要检查Tensor方法是否已经注册。
    if not hasattr(torch.Tensor, custom_backend_name):
        # 如果torch.Tensor没有定义custom_backend_name方法，则抛出运行时错误。
        raise RuntimeError(
            f"Can not automatically generate {custom_backend_name}() method for torch.nn.Module."
            f"Because torch.Tensor doesn't has the method {custom_backend_name}()."
            f"For this error, you can try setting for_tensor=True.")

    def wrap_module_to(self: torch.nn.modules.module.T,
                       device: Optional[Union[int, torch.device]] = None) -> torch.nn.modules.module.T:
        r"""Move all model parameters and buffers to the custom device.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on device while being optimized.

        .. note::
            This method modifies the module in-place.

        Args:
            device (int, optional): if specified, all parameters will be copied to that device
        """
        # 应用自定义后端的方法到Tensor上，返回修改后的模块
        return self._apply(lambda t: getattr(t, custom_backend_name)(device))

    _check_register_once(torch.nn.Module, custom_backend_name)
    # 将wrap_module_to方法注册为torch.nn.Module的custom_backend_name方法
    setattr(torch.nn.Module, custom_backend_name, wrap_module_to)

def _generate_packed_sequence_methods_for_privateuse1_backend(custom_backend_name: str) -> None:
    # 根据自定义后端名称生成PackedSequence模块属性和方法，依赖于Tensor方法，
    # 因此需要检查Tensor方法是否已经注册。
    if not hasattr(torch.Tensor, f'is_{custom_backend_name}') or \
       not hasattr(torch.Tensor, custom_backend_name):
        # 如果torch.Tensor没有定义is_custom_backend_name或custom_backend_name方法，则抛出运行时错误。
        raise RuntimeError(
            f"Can not automatically generate is_{custom_backend_name}() or "
            f"{custom_backend_name}() method for torch.nn.utils.rnn.PackedSequence."
            f"Because torch.Tensor doesn't has the method is_{custom_backend_name}()"
            f"or {custom_backend_name}()."
            f"For this error, you can try setting for_tensor=True.")

    @property  # type: ignore[misc]
    def wrap_tensor_backend(self: torch.nn.utils.rnn.PackedSequence) -> bool:
        # 返回PackedSequence数据是否存储在自定义后端设备上的布尔值
        return self.data.device.type == custom_backend_name

    _check_register_once(torch.nn.utils.rnn.PackedSequence, f'is_{custom_backend_name}')
    # 将wrap_tensor_backend方法注册为torch.nn.utils.rnn.PackedSequence的is_custom_backend_name方法
    setattr(torch.nn.utils.rnn.PackedSequence, f'is_{custom_backend_name}', wrap_tensor_backend)
    # 定义一个方法，将模型参数和缓冲区移动到自定义设备上的包装方法
    def wrap_module_to(self: torch.nn.utils.rnn.PackedSequence,
                       *args, **kwargs) -> torch.nn.utils.rnn.PackedSequence:
        r"""Move all model parameters and buffers to the custom device.

        This also makes associated parameters and buffers different objects. So
        it should be called before constructing optimizer if the module will
        live on device while being optimized.

        .. note::
            This method modifies the module in-place.

        Args:
            device (int, optional): if specified, all parameters will be copied to that device
        """
        # 创建一个空的张量 ex，使用与 self.data 相同的数据类型和设备类型
        ex = torch.tensor((), dtype=self.data.dtype, device=self.data.device).to(*args, **kwargs)
        # 如果 ex 的设备类型与 custom_backend_name 相同，则将 self 模块移动到指定设备
        if ex.device.type == custom_backend_name:
            return self.to(*args, **kwargs)
        # 更新 kwargs，添加一个键为 'device'，值为 custom_backend_name 的项
        kwargs.update({'device': custom_backend_name})
        # 将 self 模块移动到指定设备
        return self.to(*args, **kwargs)

    # 调用 _check_register_once 函数，确保只注册一次 custom_backend_name
    _check_register_once(torch.nn.utils.rnn.PackedSequence, custom_backend_name)
    # 将 wrap_module_to 方法作为 torch.nn.utils.rnn.PackedSequence 类的一个属性，属性名为 custom_backend_name
    setattr(torch.nn.utils.rnn.PackedSequence, custom_backend_name, wrap_module_to)
def _generate_storage_methods_for_privateuse1_backend(custom_backend_name: str,
                                                      unsupported_dtype: Optional[List[torch.dtype]] = None) -> None:
    # 在 _StorageBase 类中注册属性，并由 UntypedStorage 通过继承获得。
    @property  # type: ignore[misc]
    def wrap_storage_backend(self: torch.storage._StorageBase) -> bool:
        r"""Return the internal :class:`torch.UntypedStorage`."""
        return self.device.type == custom_backend_name

    # 在 _StorageBase 类中注册属性，名称为 is_custom_backend_name
    _check_register_once(torch.storage._StorageBase, f'is_{custom_backend_name}')
    setattr(torch.storage._StorageBase, f'is_{custom_backend_name}', wrap_storage_backend)

    def wrap_storage_to(self, device=None, non_blocking=False):
        r"""Return a copy of this object in custom device memory.

        If this object is already in device memory and on the correct device, then
        no copy is performed and the original object is returned.

        Args:
            device (int): The destination device id. Defaults to the current device.
            non_blocking (bool): If ``True`` and the source is in pinned memory,
            the copy will be asynchronous with respect to the host. Otherwise,
            the argument has no effect.
        """
        # 应该有与存储设备相关的判断，以及与存储类型相关的判断，
        # 但这取决于扩展功能，因此在自动生成中暂时省略此部分。

        # 标准化设备索引
        device_idx = _normalization_device(custom_backend_name, device)

        if getattr(self, f'is_{custom_backend_name}'):
            # 存储已经在期望的设备上。
            if self.get_device() == device_idx:
                return self

        # 对于稀疏存储，自定义需要自行扩展实现。
        if self.is_sparse:
            raise RuntimeError(f"Can not support a sparse storage move to {custom_backend_name} backend")

        # 创建 UntypedStorage 并复制数据
        untyped_storage = torch.UntypedStorage(
            self.size(), device=torch.device(f'{custom_backend_name}:{device_idx}')
        )
        untyped_storage.copy_(self, non_blocking)
        return untyped_storage

    # 在 _StorageBase 类中注册属性，名称为 custom_backend_name
    _check_register_once(torch.storage._StorageBase, custom_backend_name)
    setattr(torch.storage._StorageBase, custom_backend_name, wrap_storage_to)

    # 注册 TypedStorage 类的对应属性。
    # 当 TypedStorage 类被移除时，此注册也会被移除。

    @property  # type: ignore[misc]
    def wrap_typed_storage_backend(self: torch.storage.TypedStorage) -> bool:
        torch.storage._warn_typed_storage_removal()
        return self._untyped_storage.device.type == custom_backend_name

    # 在 TypedStorage 类中注册属性，名称为 is_custom_backend_name
    _check_register_once(torch.TypedStorage, f'is_{custom_backend_name}')
    setattr(torch.storage.TypedStorage, f'is_{custom_backend_name}', wrap_typed_storage_backend)
    # 定义一个方法，将当前对象（torch.storage.TypedStorage）包装为指定类型的存储对象
    def wrap_typed_storage_to(self: torch.storage.TypedStorage,
                              device=None, non_blocking=False, **kwargs) -> torch.storage.TypedStorage:
        # 警告：已弃用的类型存储功能
        torch.storage._warn_typed_storage_removal()
        # 如果不支持的数据类型列表存在，并且当前存储对象的数据类型在其中，则抛出运行时错误
        if unsupported_dtype and self.dtype in unsupported_dtype:
            raise RuntimeError(f"Cannot create {custom_backend_name} storage "
                               f"as {self.dtype} dtype is not supported by this backend")
        # 使用getattr方法获取未类型化存储对象的特定自定义后端存储对象
        custom_backend_storage: torch.UntypedStorage = getattr(
            self._untyped_storage, custom_backend_name)(device, non_blocking, **kwargs)
        # 调用_new_wrapped_storage方法，创建并返回一个新的包装后的存储对象
        return self._new_wrapped_storage(custom_backend_storage)

    # 检查注册一次函数，确保torch.TypedStorage和custom_backend_name的关联只注册一次
    _check_register_once(torch.TypedStorage, custom_backend_name)
    # 将custom_backend_name作为属性设置到torch.TypedStorage类中，属性值为wrap_typed_storage_to方法
    setattr(torch.TypedStorage, custom_backend_name, wrap_typed_storage_to)
def generate_methods_for_privateuse1_backend(for_tensor: bool = True, for_module: bool = True,
                                             for_packed_sequence: bool = True,
                                             for_storage: bool = False,
                                             unsupported_dtype: Optional[List[torch.dtype]] = None) -> None:
    r"""
    Automatically generate attributes and methods for the custom backend after rename privateuse1 backend.

    In the default scenario, storage-related methods will not be generated automatically.

    When you implement kernels for various torch operations, and register them to the PrivateUse1 dispatch key.
    And call the function torch.rename_privateuse1_backend("foo") to rename your backend name.
    At this point, you can easily register specific methods and attributes by calling this function.
    Just like torch.Tensor.foo(), torch.Tensor.is_foo, torch.Storage.foo(), torch.Storage.is_foo.

    Note: We recommend you use generic functions (check devices are equal or to(device=)).
    We provide these methods for convenience only and they will be "monkey patched" onto the objects
    and so will not be properly typed. For Storage methods generate, if you need to support sparse data storage,
    you need to extend the implementation yourself.

    Args:
        for_tensor (bool): whether register related methods for torch.Tensor class.
        for_module (bool): whether register related methods for torch.nn.Module class.
        for_storage (bool): whether register related methods for torch.Storage class.
        unsupported_dtype (List[torch.dtype]): takes effect only when the storage method needs to be generated,
            indicating that the storage does not support the torch.dtype type.

    Example::

        >>> # xdoctest: +SKIP("failing")
        >>> torch.utils.rename_privateuse1_backend("foo")
        >>> torch.utils.generate_methods_for_privateuse1_backend()
        # Then automatically generate backend-related attributes and methods.
        >>> a = torch.tensor(2).foo()
        >>> a.is_foo
        >>> hasattr(torch.nn.Module, 'foo')

    """
    # 获取自定义私有后端的名称
    custom_backend_name = _get_privateuse1_backend_name()

    # 如果需要为 torch.Tensor 类注册相关方法，则调用对应函数生成方法
    if for_tensor:
        _generate_tensor_methods_for_privateuse1_backend(custom_backend_name)

    # 如果需要为 torch.nn.Module 类注册相关方法，则调用对应函数生成方法
    if for_module:
        _generate_module_methods_for_privateuse1_backend(custom_backend_name)

    # 如果需要为 torch.Storage 类注册相关方法，则调用对应函数生成方法
    if for_storage:
        _generate_storage_methods_for_privateuse1_backend(custom_backend_name, unsupported_dtype)

    # 如果需要为 packed sequence 注册相关方法，则调用对应函数生成方法
    if for_packed_sequence:
        _generate_packed_sequence_methods_for_privateuse1_backend(custom_backend_name)


def _get_custom_mod_func(func_name: str):
    r"""
    Return the func named `func_name` defined in custom device module. If not defined,
    return `None`. And the func is registered with `torch.utils.rename_privateuse1_backend('foo')`
    and `torch._register_device_module('foo', BackendModule)`.

    """
    # 如果自定义设备模块或函数未定义，将会给出警告或错误消息。
    # Args:
    #     func_name (str): 要返回的自定义设备模块中名为 func_name 的可调用函数名称。
    # Example::
    #     class DummyfooModule:
    #         @staticmethod
    #         def is_available():
    #             return True
    #         @staticmethod
    #         def func_name(*args, **kwargs):
    #             ....
    #     torch.utils.rename_privateuse1_backend("foo")
    #     torch._register_device_module("foo", DummyfooModule)
    #     foo_is_available_func = torch.utils.backend_registration._get_custom_mod_func("is_available")
    #     if foo_is_available_func:
    #         foo_is_available = foo_is_available_func()
    #     func_ = torch.utils.backend_registration._get_custom_mod_func("func_name")
    #     if func_:
    #         result = func_(*args, **kwargs)
    # Attention: 此函数不应直接由用户使用，因此标记为私有。这是一个方便的函数，供后端实现者更轻松地调用其后端扩展的钩子。
    """
    确保 func_name 是字符串类型，否则会抛出类型错误。
    获取私有使用1后端的名称。
    获取 torch 中的自定义设备模块，如果不存在则为 None。
    尝试获取 custom_device_mod 中名为 func_name 的函数对象，如果不存在则为 None。
    如果 custom_device_mod 或 function 为空，则抛出运行时错误，提示用户注册自定义后端模块并确保其具有相应的 API。
    返回找到的 function 函数对象。
    """
    assert isinstance(func_name, str), f"func_name must be `str`, but got `{type(func_name)}`."
    backend_name = _get_privateuse1_backend_name()
    custom_device_mod = getattr(torch, backend_name, None)  # type: ignore[arg-type]
    function = getattr(custom_device_mod, func_name, None)  # type: ignore[arg-type]
    if custom_device_mod is None or function is None:
        message = f'Try to call torch.{backend_name}.{func_name}. The backend must register a custom backend '
        message += f"module with `torch._register_device_module('{backend_name}', BackendModule)`. And "
        message += f"BackendModule needs to have the following API's:\n `{func_name}(*args, **kwargs)`. \n"
        raise RuntimeError(message)
    return function
```