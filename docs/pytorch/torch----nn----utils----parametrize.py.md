# `.\pytorch\torch\nn\utils\parametrize.py`

```py
# 配置类型系统以允许未类型化的函数定义
# 导入必要的模块
import collections
import copyreg
from contextlib import contextmanager
from copy import deepcopy
from typing import Dict, Optional, Sequence, Tuple, Union

# 导入 PyTorch 相关模块和类
import torch
from torch import Tensor
from torch.nn.modules.container import Module, ModuleDict, ModuleList
from torch.nn.parameter import Parameter
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

# 定义 __all__ 列表，指定可以从该模块导入的公共名称
__all__ = [
    "cached",
    "ParametrizationList",
    "register_parametrization",
    "is_parametrized",
    "remove_parametrizations",
    "type_before_parametrizations",
    "transfer_parametrizations_and_params",
]

# 全局变量，用于启用或禁用缓存系统
_cache_enabled = 0
# 缓存字典，键为元组 (int, str)，值为可选的 Tensor 对象
_cache: Dict[Tuple[int, str], Optional[Tensor]] = {}


@contextmanager
def cached():
    r"""上下文管理器，启用与 :func:`register_parametrization` 注册的参数化对象相关的缓存系统。

    当此上下文管理器激活时，首次需要参数化对象时，其值将被计算并缓存。
    离开上下文管理器时，缓存的值将被丢弃。

    这在前向传播过程中多次使用参数化参数时非常有用。
    例如，当参数化 RNN 的循环核或共享权重时。

    通过以下简单方式可以激活缓存：
    
    .. code-block:: python

        import torch.nn.utils.parametrize as P
        ...
        with P.cached():
            output = model(inputs)

    在训练和评估中可以使用此方式。还可以包装多次使用参数化张量的模块部分。
    例如，具有参数化循环核的 RNN 的循环：

    .. code-block:: python

        with P.cached():
            for x in xs:
                out_rnn = self.rnn_cell(x, out_rnn)
    """
    global _cache
    global _cache_enabled
    _cache_enabled += 1
    try:
        yield
    finally:
        _cache_enabled -= 1
        if not _cache_enabled:
            _cache = {}


def _register_parameter_or_buffer(module, name, X):
    # 根据 X 的类型（Parameter 或 Tensor），向 module 注册参数或缓冲区
    if isinstance(X, Parameter):
        module.register_parameter(name, X)
    else:
        module.register_buffer(name, X)


def _maybe_set(dest: Tensor, src: Tensor) -> None:
    # 根据条件确定是否需要交换 Tensor 对象
    should_swap = (
        get_swap_module_params_on_conversion() or is_traceable_wrapper_subclass(dest)
    )
    if should_swap:
        if isinstance(dest, Parameter) and not isinstance(src, Parameter):
            src = Parameter(src, requires_grad=dest.requires_grad)
        torch.utils.swap_tensors(dest, src)
    else:
        dest.set_(src)  # type: ignore[call-overload]


class ParametrizationList(ModuleList):
    r"""一个顺序容器，用于保存和管理参数化 :class:`torch.nn.Module` 的原始参数或缓冲区。

    当 ``module[tensor_name]`` 是一个参数化对象时，它是 ``module.parametrizations[tensor_name]`` 的类型。

    这是一个自定义的 ModuleList 子类，专门用于管理参数化模块的参数和缓冲区。
    ```
    has been parametrized with :func:`register_parametrization`.

    If the first registered parametrization has a ``right_inverse`` that returns one tensor or
    does not have a ``right_inverse`` (in which case we assume that ``right_inverse`` is the identity),
    it will hold the tensor under the name ``original``.
    If it has a ``right_inverse`` that returns more than one tensor, these will be registered as
    ``original0``, ``original1``, ...

    .. warning::
        This class is used internally by :func:`register_parametrization`. It is documented
        here for completeness. It shall not be instantiated by the user.

    Args:
        modules (sequence): sequence of modules representing the parametrizations
        original (Parameter or Tensor): parameter or buffer that is parametrized
        unsafe (bool): a boolean flag that denotes whether the parametrization
            may change the dtype and shape of the tensor. Default: `False`
            Warning: the parametrization is not checked for consistency upon registration.
            Enable this flag at your own risk.
    ```py
    
    original: Tensor
    unsafe: bool
    
    ```
    Define class attributes `original` of type `Tensor` and `unsafe` of type `bool`.
    ```py
    
    def __init__(
        self,
        modules: Sequence[Module],
        original: Union[Tensor, Parameter],
        unsafe: bool = False,
    ):
        ```
        Constructor method for the class.
        
        Args:
            modules (Sequence[Module]): A sequence of modules representing the parametrizations.
            original (Union[Tensor, Parameter]): The tensor or parameter to be parametrized.
            unsafe (bool, optional): Flag indicating if the parametrization may alter dtype and shape. Defaults to False.
        ```py
        
        self.original = original
        self.unsafe = unsafe
        ```
        Initialize instance attributes `self.original` and `self.unsafe` with provided arguments.
        ```py
    
    def forward(self) -> Tensor:
        ```
        Defines the forward method of the class, which computes the output tensor.
        
        Returns:
            Tensor: The output tensor after applying parametrizations.
        
        Raises:
            RuntimeError: If parametrization is attempted during script execution mode.
        ```py
        
        if torch.jit.is_scripting():
            raise RuntimeError("Parametrization is not working with scripting.")
        ```
        Raise a runtime error if the script is in execution mode, as parametrization is not supported.
        ```py
        
        # Unpack the originals for the first parametrization
        if self.is_tensor:
            x = self[0](self.original)
        else:
            originals = (getattr(self, f"original{i}") for i in range(self.ntensors))
            x = self[0](*originals)
        ```
        Depending on whether `self` is a tensor or not, unpack the original tensor or tensors (if more than one).
        ```py
        
        # It's not possible to call self[1:] here, so we have to be a bit more cryptic
        # Also we want to skip all non-integer keys
        curr_idx = 1
        while hasattr(self, str(curr_idx)):
            x = self[curr_idx](x)
            curr_idx += 1
        ```
        Iterate over attributes indexed from 1 onwards, applying parametrizations sequentially to `x`.
        ```py
        
        return x
        ```
        Return the final computed tensor `x`.
    ```py
def _inject_new_class(module: Module) -> None:
    r"""Set up a module to be parametrized.

    This works by substituting the class of the module by a class
    that extends it to be able to inject a property

    Args:
        module (nn.Module): module into which to inject the property
    """
    # 获取模块的类对象
    cls = module.__class__

    def default_deepcopy(self, memo):
        # 当当前类没有定义 __deepcopy__ 方法时，模拟标准的深拷贝过程
        obj = memo.get(id(self), None)
        if obj is not None:
            return obj
        replica = self.__new__(self.__class__)
        memo[id(self)] = replica
        # 深拷贝当前对象的 __dict__ 属性
        replica.__dict__ = deepcopy(self.__dict__, memo)
        # 如果存在 slots，也保存它们
        slots_to_save = copyreg._slotnames(self.__class__)  # type: ignore[attr-defined]
        for slot in slots_to_save:
            if hasattr(self, slot):
                setattr(replica, slot, deepcopy(getattr(self, slot), memo))
        return replica

    def getstate(self):
        # 抛出运行时异常，指示不支持序列化参数化模块
        raise RuntimeError(
            "Serialization of parametrized modules is only "
            "supported through state_dict(). See:\n"
            "https://pytorch.org/tutorials/beginner/saving_loading_models.html"
            "#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training"
        )

    # 构造包含 getstate 方法的字典 dct
    dct = {"__getstate__": getstate}
    # 如果当前类没有定义 __deepcopy__ 方法，则添加 default_deepcopy 方法
    if not hasattr(cls, "__deepcopy__"):
        dct["__deepcopy__"] = default_deepcopy  # type: ignore[assignment]

    # 创建新的类 param_cls，继承自原始类 cls，并添加 dct 中的方法
    param_cls = type(
        f"Parametrized{cls.__name__}",
        (cls,),
        dct,
    )

    # 将 module 的类设置为 param_cls
    module.__class__ = param_cls


def _inject_property(module: Module, tensor_name: str) -> None:
    r"""Injects a property into module[tensor_name].

    It assumes that the class in the module has already been modified from its
    original one using _inject_new_class and that the tensor under :attr:`tensor_name`
    has already been moved out

    Args:
        module (nn.Module): module into which to inject the property
        tensor_name (str): name of the name of the property to create
    """
    # 检查前提条件，确保 module 没有 tensor_name 属性
    assert not hasattr(module, tensor_name)

    @torch.jit.unused
    def get_cached_parametrization(parametrization) -> Tensor:
        # 获取全局变量 _cache
        global _cache
        # 构造缓存键值对的键
        key = (id(module), tensor_name)
        # 从缓存 _cache 中获取 tensor
        tensor = _cache.get(key)
        if tensor is None:
            # 如果缓存中没有，则调用 parametrization() 方法生成 tensor，并缓存起来
            tensor = parametrization()
            _cache[key] = tensor
        return tensor
    # 定义一个方法，返回一个张量（Tensor），用于动态参数化
    def get_parametrized(self) -> Tensor:
        # 如果当前处于脚本化状态，则抛出运行时错误，因为参数化与脚本化不兼容
        if torch.jit.is_scripting():
            raise RuntimeError("Parametrization is not working with scripting.")
        
        # 获取当前张量名称对应的参数化对象
        parametrization = self.parametrizations[tensor_name]
        
        # 如果启用了缓存
        if _cache_enabled:
            # 如果当前处于脚本化状态，则抛出运行时错误，因为脚本化不支持缓存
            if torch.jit.is_scripting():
                # 脚本化状态下抛出异常
                raise RuntimeError(
                    "Caching is not implemented for scripting. "
                    "Either disable caching or avoid scripting."
                )
            # 如果当前处于追踪（tracing）状态，则抛出运行时错误，因为追踪模型时不能缓存参数化
            elif torch._C._get_tracing_state() is not None:
                # 追踪状态下抛出异常
                raise RuntimeError(
                    "Cannot trace a model while caching parametrizations."
                )
            else:
                # 返回缓存中的参数化结果
                return get_cached_parametrization(parametrization)
        else:
            # 如果未启用缓存，则直接计算并返回参数化结果
            return parametrization()

    # 定义一个方法，设置张量的原始值
    def set_original(self, value: Tensor) -> None:
        # 如果当前处于脚本化状态，则抛出运行时错误，因为参数化与脚本化不兼容
        if torch.jit.is_scripting():
            raise RuntimeError("Parametrization is not working with scripting.")
        
        # 使用右逆函数设置张量的原始值
        self.parametrizations[tensor_name].right_inverse(value)

    # 将动态参数化的方法和设置原始值的方法绑定到指定的模块类和张量名称上
    setattr(module.__class__, tensor_name, property(get_parametrized, set_original))
# 定义注册参数化的函数，将一个参数化方法注册到模块的张量中
def register_parametrization(
    module: Module,
    tensor_name: str,
    parametrization: Module,
    *,
    unsafe: bool = False,
) -> Module:
    r"""Register a parametrization to a tensor in a module.

    # 假设 tensor_name="weight" 用于简化。访问 module.weight 时，模块将返回参数化版本 parametrization(module.weight)。
    # 如果原始张量需要梯度，则反向传播将通过 parametrization 进行微分，优化器会相应更新张量。
    # 第一次模块注册参数化时，此函数将向模块添加一个类型为 ParametrizationList 的属性 parametrizations。

    # 在 module.parametrizations.weight 下可访问张量 "weight" 的参数化列表。
    # 原始张量可以通过 module.parametrizations.weight.original 访问。

    # 可以通过在同一属性上注册多个参数化来连接参数化。

    # 注册参数化时，已注册的参数化的训练模式将更新以匹配宿主模块的训练模式。

    # 参数化的参数和缓冲区具有内置的缓存系统，可以通过上下文管理器 cached 激活。

    # 参数化 parametrization 可以选择实现具有如下签名的方法
    # def right_inverse(self, X: Tensor) -> Union[Tensor, Sequence[Tensor]]
    # 当注册第一个参数化时，会调用未参数化张量的 right_inverse 方法来计算原始张量的初始值。
    # 如果未实现此方法，则原始张量将仅为未参数化张量。

    # 如果所有在张量上注册的参数化都实现了 right_inverse，则可以通过分配来初始化参数化张量，如下例所示。

    # 第一个参数化可以依赖多个输入。这可以通过从 right_inverse 返回张量元组来实现（参见下面的 RankOne 参数化示例实现）。

    # 在这种情况下，非约束张量也位于 module.parametrizations.weight 下，名称为 "original0"、"original1" 等。

    # 注意：
    # 如果 unsafe=False（默认值），将调用前向和 right_inverse 方法一次，以执行一些一致性检查。
    # 如果 unsafe=True，则如果张量未被参数化，则将调用 right_inverse，否则什么也不会被调用。
    
    # 返回更新后的模块对象
    return module
    .. note::

        In most situations, ``right_inverse`` will be a function such that
        ``forward(right_inverse(X)) == X`` (see
        `right inverse <https://en.wikipedia.org/wiki/Inverse_function#Right_inverses>`_).
        Sometimes, when the parametrization is not surjective, it may be reasonable
        to relax this.

    .. warning::

        If a parametrization depends on several inputs, :func:`~register_parametrization`
        will register a number of new parameters. If such parametrization is registered
        after the optimizer is created, these new parameters will need to be added manually
        to the optimizer. See :meth:`torch.Optimizer.add_param_group`.

    Args:
        module (nn.Module): module on which to register the parametrization
        tensor_name (str): name of the parameter or buffer on which to register
            the parametrization
        parametrization (nn.Module): the parametrization to register
    Keyword args:
        unsafe (bool): a boolean flag that denotes whether the parametrization
            may change the dtype and shape of the tensor. Default: `False`
            Warning: the parametrization is not checked for consistency upon registration.
            Enable this flag at your own risk.

    Raises:
        ValueError: if the module does not have a parameter or a buffer named :attr:`tensor_name`
    parametrization.train(module.training)
    # 根据当前模块的训练状态，调整参数化对象的训练状态

    elif tensor_name in module._buffers or tensor_name in module._parameters:
        # 如果指定的张量名存在于模块的缓冲区或参数中：
        
        # 获取模块中指定张量名对应的原始缓冲区或参数
        original = getattr(module, tensor_name)
        
        # 创建参数化列表对象，用于存储参数化方法及其相关信息
        parametrizations = ParametrizationList(
            [parametrization], original, unsafe=unsafe
        )
        
        # 删除模块中原有的参数或缓冲区
        delattr(module, tensor_name)
        
        # 如果模块尚未被参数化，进行准备工作以注入属性
        if not is_parametrized(module):
            # 修改模块的类属性
            _inject_new_class(module)
            
            # 在模块实例中注入一个 ModuleDict，用于存储参数化方法
            module.parametrizations = ModuleDict()
        
        # 向模块的类中注入属性
        _inject_property(module, tensor_name)
        
        # 将参数化列表对象添加到模块的 parametrizations 中
        assert isinstance(module.parametrizations, ModuleDict)  # 确保类型与预期一致
        module.parametrizations[tensor_name] = parametrizations
    
    else:
        # 如果模块中不存在指定的参数、缓冲区或参数化元素，则抛出异常
        raise ValueError(
            f"Module '{module}' does not have a parameter, a buffer, or a "
            f"parametrized element with name '{tensor_name}'"
        )
    # 返回当前模块对象，使其可以在其他代码中被导入和使用
    return module
def is_parametrized(module: Module, tensor_name: Optional[str] = None) -> bool:
    r"""Determine if a module has a parametrization.

    Args:
        module (nn.Module): module to query
        tensor_name (str, optional): name of the parameter in the module
            Default: ``None``
    Returns:
        ``True`` if :attr:`module` has a parametrization for the parameter named :attr:`tensor_name`,
        or if it has any parametrization when :attr:`tensor_name` is ``None``;
        otherwise ``False``
    """
    # 获取模块的 parametrizations 属性，如果不存在或不是 ModuleDict 类型，则返回 False
    parametrizations = getattr(module, "parametrizations", None)
    if parametrizations is None or not isinstance(parametrizations, ModuleDict):
        return False
    if tensor_name is None:
        # 如果 tensor_name 为 None，则检查是否至少有一个参数被 parametrize
        return len(parametrizations) > 0
    else:
        # 否则，检查指定的 tensor_name 是否在 parametrizations 中
        return tensor_name in parametrizations


def remove_parametrizations(
    module: Module,
    tensor_name: str,
    leave_parametrized: bool = True,
) -> Module:
    r"""Remove the parametrizations on a tensor in a module.

    - If ``leave_parametrized=True``, ``module[tensor_name]`` will be set to
      its current output. In this case, the parametrization shall not change the ``dtype``
      of the tensor.
    - If ``leave_parametrized=False``, ``module[tensor_name]`` will be set to
      the unparametrised tensor in ``module.parametrizations[tensor_name].original``.
      This is only possible when the parametrization depends on just one tensor.

    Args:
        module (nn.Module): module from which remove the parametrization
        tensor_name (str): name of the parametrization to be removed
        leave_parametrized (bool, optional): leave the attribute :attr:`tensor_name` parametrized.
            Default: ``True``

    Returns:
        Module: module

    Raises:
        ValueError: if ``module[tensor_name]`` is not parametrized
        ValueError: if ``leave_parametrized=False`` and the parametrization depends on several tensors
    """
    if not is_parametrized(module, tensor_name):
        raise ValueError(
            f"Module {module} does not have a parametrization on {tensor_name}"
        )

    # Assert that parametrizations is indeed a ModuleDict to satisfy type checking
    assert isinstance(module.parametrizations, ModuleDict)

    # 获取 tensor_name 对应的 parametrization 对象
    parametrizations = module.parametrizations[tensor_name]
    # 如果参数化对象是一个张量
    if parametrizations.is_tensor:
        # 获取原始张量
        original = parametrizations.original
        # 如果保留参数化状态
        if leave_parametrized:
            # 使用 torch.no_grad() 来避免梯度计算
            with torch.no_grad():
                # 获取模块中的张量
                t = getattr(module, tensor_name)
            # 因为已经确认它们具有相同的数据类型，可以使用 set_() 方法来设置参数值
            # 这样可以保证参数的 id() 不变，用户不需要更新优化器
            with torch.no_grad():
                if type(original) is torch.Tensor:
                    _maybe_set(original, t)
                else:
                    try:
                        _maybe_set(original, t)
                    except RuntimeError as e:
                        # 如果遇到 RuntimeError，说明对于张量子类参数，需要正确实现 set_()
                        raise RuntimeError(
                            "Calling remove_parametrizations() with leave_parametrized=True "
                            "for a parameter that is an instance of a tensor subclass requires "
                            "set_() to be implemented correctly for the tensor subclass."
                            "Alternatively, one can opt into the swap_tensors path"
                            "Either set leave_parametrized=False or provide a working implementation"
                            "for set_() in the tensor subclass or set "
                            "torch.__future__.set_swap_module_params_on_conversion(True)."
                        ) from e
        else:
            # 如果不保留参数化状态，抛出异常
            raise ValueError(
                "Cannot leave unparametrized (`leave_parametrized=False`) a tensor "
                "that is parametrized in terms of a sequence of tensors."
            )
    else:
        # 如果参数化对象不是张量
        if leave_parametrized:
            # 不能使用 no_grad，因为需要知道原始张量是否需要梯度
            t = getattr(module, tensor_name)
            # 直接信任用户会把它加入到优化器中
            original = Parameter(t) if t.requires_grad else t
        else:
            # 如果不保留参数化状态，抛出异常
            raise ValueError(
                "Cannot leave unparametrized (`leave_parametrized=False`) a tensor "
                "that is parametrized in terms of a sequence of tensors."
            )

    # 删除管理参数化的属性
    delattr(module.__class__, tensor_name)
    # 删除 ParametrizationList
    del module.parametrizations[tensor_name]

    # 将参数或缓冲器重新注册到主类中
    _register_parameter_or_buffer(module, tensor_name, original)

    # 如果该类没有任何其他参数或缓冲器被参数化，则回滚参数化类
    if not is_parametrized(module):
        # 删除 parametrizations 属性
        delattr(module, "parametrizations")
        # 恢复原始类
        orig_cls = module.__class__.__bases__[0]
        module.__class__ = orig_cls
    
    # 返回模块对象
    return module
def type_before_parametrizations(module: Module) -> type:
    r"""Return the module type before parametrizations were applied and if not, then it returns the module type.

    Args:
        module (nn.Module): module to get type of
    """
    # 检查给定模块是否已经被参数化
    if is_parametrized(module):
        # 如果已经被参数化，则返回其参数化前的类型
        return module.__class__.__bases__[0]
    else:
        # 如果未被参数化，则直接返回当前的类型
        return type(module)


def transfer_parametrizations_and_params(
    from_module: Module,
    to_module: Module,
    tensor_name: Optional[str] = None,
) -> Module:
    r"""Transfer parametrizations and the parameters they parametrize from :attr:`from_module` to :attr:`to_module`.

    If :attr:`tensor_name` is specified, only transfers the specified parameter, otherwise
    transfers all parametrized parameters. If those parameters do not exist in to_module, it will create them.
    Does nothing if from_module is not parametrized.

    Args:
        from_module (nn.Module): module to transfer from
        to_module (nn.Module): module to transfer to
        tensor_name (str, optional): parameter to transfer

    Returns:
        Module: to_module
    """
    # 检查源模块是否已经被参数化
    if is_parametrized(from_module):
        # 如果指定了要转移的具体参数名，则只转移该参数
        if tensor_name:
            # 检查该参数是否存在于目标模块中，如果不存在则创建
            if hasattr(to_module, tensor_name):
                setattr(to_module, tensor_name, getattr(from_module, tensor_name))
            else:
                setattr(to_module, tensor_name, getattr(from_module, tensor_name).clone())
        else:
            # 否则，转移所有被参数化的参数
            for name, param in from_module.named_parameters(recurse=False):
                if hasattr(to_module, name):
                    setattr(to_module, name, param)
                else:
                    setattr(to_module, name, param.clone())
    # 返回目标模块
    return to_module
    # 检查是否需要对 from_module 进行参数化处理
    if is_parametrized(from_module):
        # 断言 from_module.parametrizations 是 ModuleDict 类型，用于类型检查（对于类型检查工具如 mypy）
        assert isinstance(from_module.parametrizations, ModuleDict)

        # 根据 tensor_name 是否为 None，确定要传输的参数列表或单个参数
        parameters_to_transfer: Union[list, ModuleDict] = (
            from_module.parametrizations if tensor_name is None else [tensor_name]
        )

        # 断言 parameters_to_transfer 是可迭代的对象，用于类型检查（对于类型检查工具如 mypy）
        assert hasattr(parameters_to_transfer, "__iter__")

        # 遍历要传输的每个参数名称
        for parameter_name in parameters_to_transfer:
            # 如果 to_module 中不存在该参数，就在 to_module 中初始化该参数为 from_module 中对应参数的副本
            if not hasattr(to_module, parameter_name):
                setattr(
                    to_module,
                    parameter_name,
                    Parameter(getattr(from_module, parameter_name)),
                )

            # 将 from_module 中参数的 parametrizations 应用到 to_module 中的参数上
            for param_func in from_module.parametrizations[parameter_name]:
                register_parametrization(to_module, parameter_name, param_func)

            # 断言 to_module.parametrizations 是 ModuleDict 类型，用于类型检查（对于类型检查工具如 mypy）
            assert isinstance(to_module.parametrizations, ModuleDict)

            # 将参数值匹配到 to_module 中，原始值可以存储在 original 或 original0, original1... 中，需要检查所有情况
            if hasattr(from_module.parametrizations[parameter_name], "original"):
                to_module.parametrizations[
                    parameter_name
                ].original = from_module.parametrizations[parameter_name].original
            else:
                num = 0
                orig_num = "original" + str(num)
                # 循环直到找到所有的 original# 并将其设置到 to_module 中
                while hasattr(from_module.parametrizations[parameter_name], orig_num):
                    setattr(
                        to_module.parametrizations[parameter_name],
                        orig_num,
                        getattr(from_module.parametrizations[parameter_name], orig_num),
                    )
                    num = num + 1
                    orig_num = "original" + str(num)

    # 返回经过参数传递处理后的 to_module
    return to_module
```