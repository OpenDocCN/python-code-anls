# `.\pytorch\torch\distributed\nn\api\remote_module.py`

```
# 指定 Python 解释器路径和 mypy 选项
#!/usr/bin/python3
# 允许未类型化的函数定义
# mypy: allow-untyped-defs

# 导入必要的模块
import collections  # 导入 collections 模块
import io  # 导入 io 模块
import sys  # 导入 sys 模块
import types  # 导入 types 模块

# 导入类型提示相关的模块
from typing import (
    Any,  # 任意类型
    Callable,  # 可调用对象
    Dict,  # 字典类型
    Iterator,  # 迭代器
    List,  # 列表类型
    Mapping,  # 映射类型
    Optional,  # 可选类型
    Set,  # 集合类型
    Tuple,  # 元组类型
    Type,  # 类型对象
    TypeVar,  # 类型变量
    Union,  # 联合类型
)

# 导入 PyTorch 相关模块
import torch  # 导入 PyTorch 模块
import torch.distributed.rpc as rpc  # 导入 PyTorch 分布式 RPC 模块
from torch import device, dtype, nn, Tensor  # 导入设备、数据类型、神经网络模块、张量类型
from torch.distributed import _remote_device  # 导入远程设备相关模块
from torch.distributed.nn.jit import instantiator  # 导入分布式神经网络即时编译模块
from torch.distributed.rpc.internal import _internal_rpc_pickler  # 导入内部 RPC Pickler 模块
from torch.nn import Module  # 导入神经网络模块
from torch.nn.parameter import Parameter  # 导入神经网络参数模块
from torch.utils.hooks import RemovableHandle  # 导入可移除的处理句柄模块


__all__ = ["RemoteModule"]  # 模块中公开的符号列表

_grad_t = Union[Tuple[Tensor, ...], Tensor]  # 梯度类型，可以是元组或张量的联合类型

# 使用 `T` 泛型类型变量来注解 `self`，以便将返回类型限制为其子类的类型
# 参考 https://mypy.readthedocs.io/en/latest/generics.html#generic-methods-and-generic-self
T = TypeVar("T", bound="Module")

# 创建一个非可脚本化远程模块的模板实例
_NON_SCRIPTABLE_REMOTE_MODULE_MODULE = (
    instantiator.instantiate_non_scriptable_remote_module_template()
)

# 定义要序列化为远程模块的属性列表
_REMOTE_MODULE_PICKLED_ATTRIBUTES = (
    "on",
    "device",
    "is_device_map_set",
    "is_scriptable",
    "generated_methods",
    "module_rref",
)

# 使用具名元组 `_SerializedRemoteModule` 来表示序列化后的远程模块
_SerializedRemoteModule = collections.namedtuple("_SerializedRemoteModule", _REMOTE_MODULE_PICKLED_ATTRIBUTES)  # type: ignore[misc]

# `_REMOTE_MODULE_ATTRIBUTES_IGNORE_FOR_PICKLING` 中的属性大多来自 RemoteModule 的父类，有意地不进行 Pickling
# 如果 RemoteModule 的新属性不在 `_REMOTE_MODULE_PICKLED_ATTRIBUTES` 或 `_REMOTE_MODULE_ATTRIBUTES_IGNORE_FOR_PICKLING` 中，
# 则不会被 Pickling
_REMOTE_MODULE_ATTRIBUTES_IGNORE_FOR_PICKLING = (
    "training",  # 训练状态
    "_parameters",  # 参数
    "_buffers",  # 缓冲区
    "_non_persistent_buffers_set",  # 非持久性缓冲区集合
    "_backward_hooks",  # 反向钩子
    "_backward_pre_hooks",  # 反向预钩子
    "_is_full_backward_hook",  # 是否完整反向钩子
    "_forward_hooks",  # 前向钩子
    "_forward_hooks_with_kwargs",  # 带关键字参数的前向钩子
    "_forward_hooks_always_called",  # 总是调用的前向钩子
    "_forward_pre_hooks",  # 前向预钩子
    "_forward_pre_hooks_with_kwargs",  # 带关键字参数的前向预钩子
    "_state_dict_hooks",  # 状态字典钩子
    "_state_dict_pre_hooks",  # 状态字典预钩子
    "_load_state_dict_pre_hooks",  # 加载状态字典前钩子
    "_load_state_dict_post_hooks",  # 加载状态字典后钩子
    "_state_dict_pre_hooks",  # 状态字典预钩子
    "_modules",  # 模块
    # 以下两个属性是生成的方法，在 Pickling 时不可用
    "forward_async",  # 异步前向方法
    "forward",  # 前向方法
)


# RPC 处理器
def _instantiate_template(module_interface_cls, enable_moving_cpu_tensors_to_cuda):
    # 实例化可脚本化远程模块模板
    instantiator.instantiate_scriptable_remote_module_template(
        module_interface_cls, enable_moving_cpu_tensors_to_cuda
    )


def _create_module(module_cls, args, kwargs, device):
    # 创建模块实例
    module = module_cls(*args, **kwargs)
    # 检查模块是否是 nn.Module 的实例
    if not isinstance(module, nn.Module):
        raise ValueError(
            "Expect `module_cls(*args, **kwargs)` returns an instance of <class nn.Module>, "
            f"but it returns an instance of {type(module)}."
        )
    # 将模块移到指定设备
    module.to(device)
    return module


def _create_module_with_interface(
    # 定义变量 module_cls, args, kwargs, device, module_interface_cls
    module_cls, args, kwargs, device, module_interface_cls
    module = _create_module(module_cls, args, kwargs, device)
    # 调用 _create_module 函数创建一个模块实例
    if module_interface_cls is not None:
        # 如果指定了模块接口类，则对模块进行脚本化处理
        module = torch.jit.script(module)
    # 使用 rpc.RRef 包装模块实例，返回一个远程引用对象
    return rpc.RRef(module, module_interface_cls)


def _param_rrefs(module_rref, recurse) -> List[rpc.RRef[Parameter]]:
    ret: List[rpc.RRef[Parameter]] = []
    # 遍历远程模块的本地值的参数，如果递归标志为真，则包括子模块的参数
    for param in module_rref.local_value().parameters(recurse):
        ret.append(rpc.RRef(param))
    return ret


def _raise_not_supported(name: str) -> None:
    # 抛出 ValueError 异常，表明该方法不支持远程模块
    raise ValueError(f"Method ``{name}`` not supported for RemoteModule")


class _RemoteModule(nn.Module):
    def __new__(cls, *args, **kwargs):
        # 使用 __new__ 方法记录 API 使用情况
        torch._C._log_api_usage_once("torch.distributed.nn.api.remote_module")
        return super().__new__(cls)

    def __init__(
        self,
        remote_device: str,
        module_cls: Type[nn.Module],
        args: Optional[Tuple] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        _module_interface_cls: Any = None,
    ):
        # 初始化远程模块对象
        module = _create_module(module_cls, args, kwargs, remote_device)
        # 创建远程模块的远程引用对象
        self.module_rref = rpc.remote(remote_device, module_cls, args=args, kwargs=kwargs)
        self.module_rref = rpc.remote(remote_device, module_cls, args=args, kwargs=kwargs)
        

 __getstate__MethodManager
    def apply(self: T, fn: Callable[[Module], None]) -> T:  # type: ignore[return]
        _raise_not_supported(self.apply.__name__)
        # 抛出不支持的操作异常，显示函数名

    def cuda(self: T, device: Optional[Union[int, device]] = None) -> T:  # type: ignore[return]
        _raise_not_supported(self.cuda.__name__)
        # 抛出不支持的操作异常，显示函数名

    def ipu(self: T, device: Optional[Union[int, device]] = None) -> T:  # type: ignore[return]
        _raise_not_supported(self.ipu.__name__)
        # 抛出不支持的操作异常，显示函数名

    def xpu(self: T, device: Optional[Union[int, device]] = None) -> T:  # type: ignore[return]
        _raise_not_supported(self.xpu.__name__)
        # 抛出不支持的操作异常，显示函数名

    def cpu(self: T) -> T:  # type: ignore[return]
        _raise_not_supported(self.cpu.__name__)
        # 抛出不支持的操作异常，显示函数名

    def type(self: T, dst_type: Union[dtype, str]) -> T:  # type: ignore[return]
        _raise_not_supported(self.type.__name__)
        # 抛出不支持的操作异常，显示函数名

    def float(self: T) -> T:  # type: ignore[return]
        _raise_not_supported(self.float.__name__)
        # 抛出不支持的操作异常，显示函数名

    def double(self: T) -> T:  # type: ignore[return]
        _raise_not_supported(self.double.__name__)
        # 抛出不支持的操作异常，显示函数名

    def half(self: T) -> T:  # type: ignore[return]
        _raise_not_supported(self.half.__name__)
        # 抛出不支持的操作异常，显示函数名

    def bfloat16(self: T) -> T:  # type: ignore[return]
        _raise_not_supported(self.bfloat16.__name__)
        # 抛出不支持的操作异常，显示函数名

    def to(self, *args, **kwargs) -> T:  # type: ignore[misc, return, type-var]
        _raise_not_supported(self.to.__name__)
        # 抛出不支持的操作异常，显示函数名

    def register_backward_hook(  # type: ignore[return]
        self, hook: Callable[[Module, _grad_t, _grad_t], Union[None, _grad_t]]
    ) -> RemovableHandle:
        _raise_not_supported(self.register_backward_hook.__name__)
        # 抛出不支持的操作异常，显示函数名

    def register_forward_pre_hook(  # type: ignore[return]
        self,
        hook: Union[
            Callable[[T, Tuple[Any, ...]], Optional[Any]],
            Callable[
                [T, Tuple[Any, ...], Dict[str, Any]],
                Optional[Tuple[Any, Dict[str, Any]]],
            ],
        ],
        prepend: bool = False,
        with_kwargs: bool = False,
    ) -> RemovableHandle:
        _raise_not_supported(self.register_forward_pre_hook.__name__)
        # 抛出不支持的操作异常，显示函数名

    def register_forward_hook(  # type: ignore[return, override]
        self,
        hook: Union[
            Callable[[T, Tuple[Any, ...], Any], Optional[Any]],
            Callable[[T, Tuple[Any, ...], Dict[str, Any], Any], Optional[Any]],
        ],
        prepend: bool = False,
        with_kwargs: bool = False,
    ) -> RemovableHandle:
        _raise_not_supported(self.register_forward_hook.__name__)
        # 抛出不支持的操作异常，显示函数名

    def state_dict(self, *args, **kwargs):
        _raise_not_supported(self.state_dict.__name__)
        # 抛出不支持的操作异常，显示函数名

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ):
        _raise_not_supported(self.load_state_dict.__name__)
        # 抛出不支持的操作异常，显示函数名

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        raise ValueError(
            "Method ``parameters`` not supported for RemoteModule. Please use ``remote_parameters`` instead."
        )
        # 抛出值错误，指出不支持的操作消息
    # 声明一个方法用于返回模块的参数的迭代器，参数可以指定前缀、是否递归以及是否移除重复的参数
    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Parameter]]:
        # 调用一个私有方法，抛出不支持的异常，传递当前方法名作为参数
        _raise_not_supported(self.named_parameters.__name__)
    
    # 声明一个方法用于返回模块的缓冲区的迭代器，可以指定是否递归
    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        # 调用一个私有方法，抛出不支持的异常，传递当前方法名作为参数
        _raise_not_supported(self.buffers.__name__)
    
    # 声明一个方法用于返回命名的缓冲区的迭代器，可以指定前缀、是否递归以及是否移除重复的缓冲区
    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Tensor]]:
        # 调用一个私有方法，抛出不支持的异常，传递当前方法名作为参数
        _raise_not_supported(self.named_buffers.__name__)
    
    # 声明一个方法用于返回模块的子模块的迭代器
    def children(self) -> Iterator[Module]:
        # 调用一个私有方法，抛出不支持的异常，传递当前方法名作为参数
        _raise_not_supported(self.children.__name__)
    
    # 声明一个方法用于返回命名的子模块的迭代器
    def named_children(self) -> Iterator[Tuple[str, Module]]:
        # 调用一个私有方法，抛出不支持的异常，传递当前方法名作为参数
        _raise_not_supported(self.named_children.__name__)
    
    # 声明一个方法用于返回模块的所有子模块及其自身的迭代器
    def modules(self) -> Iterator[Module]:
        # 调用一个私有方法，抛出不支持的异常，传递当前方法名作为参数
        _raise_not_supported(self.modules.__name__)
    
    # 声明一个方法用于返回命名的模块及其自身的迭代器，可以指定前缀、是否移除重复的模块
    def named_modules(
        self,
        memo: Optional[Set[Module]] = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ):
        # 调用一个私有方法，抛出不支持的异常，传递当前方法名作为参数
        _raise_not_supported(self.named_modules.__name__)
    
    # 声明一个方法用于设置模块处于训练模式
    def train(self: T, mode: bool = True) -> T:
        # 调用模块的远程过程调用（RPC）方法，设置模块为训练模式，并返回结果
        return self.module_rref.rpc_sync().train()  # type: ignore[operator, union-attr]
    
    # 声明一个方法用于设置模块处于评估模式
    def eval(self: T) -> T:
        # 调用模块的远程过程调用（RPC）方法，设置模块为评估模式，并返回结果
        return self.module_rref.rpc_sync().eval()  # type: ignore[operator, union-attr]
    
    # 声明一个方法用于设置模块参数是否需要梯度
    def requires_grad_(self: T, requires_grad: bool = True) -> T:
        # 调用一个私有方法，抛出不支持的异常，传递当前方法名作为参数
        _raise_not_supported(self.requires_grad_.__name__)
    
    # 声明一个方法用于将模块的梯度清零
    def zero_grad(self, set_to_none: bool = True) -> None:
        # 调用一个私有方法，抛出不支持的异常，传递当前方法名作为参数
        _raise_not_supported(self.zero_grad.__name__)
    
    # 声明一个方法用于共享模块的内存
    def share_memory(self: T) -> T:
        # 调用一个私有方法，抛出不支持的异常，传递当前方法名作为参数
        _raise_not_supported(self.share_memory.__name__)
    
    # 声明一个方法用于返回模块的额外表示字符串
    def extra_repr(self) -> str:
        # 调用一个私有方法，抛出不支持的异常，传递当前方法名作为参数
        _raise_not_supported(self.extra_repr.__name__)
    def _prepare_init(self, remote_device_str: str) -> bool:
        """Prepare the initialization and returns whether to enable automatically moving CPU tensors to CUDA devices."""
        # Sanity check.
        assert rpc._is_current_rpc_agent_set(), "RemoteModule only works in RPC."

        # 将远程设备字符串转换为远程设备对象
        remote_device = _remote_device(remote_device_str)
        # 设置当前远程设备的名称或者排名作为 `on` 属性
        self.on = (
            remote_device.worker_name()
            if remote_device.worker_name() is not None
            else remote_device.rank()
        )
        # 将远程设备的设备名称转换为字符串并存储在 `device` 属性中
        self.device = str(remote_device.device())
        # 获取当前 RPC 代理对象
        agent = rpc._get_current_rpc_agent()
        # 检查远程工作器的设备映射是否已设置
        self.is_device_map_set = bool(
            agent._get_device_map(agent.get_worker_info(self.on))  # type: ignore[arg-type]
        )
        # 检查当前设备是否为 CUDA 设备，决定是否启用将 CPU 张量自动移动到 CUDA 设备
        enable_moving_cpu_tensors_to_cuda = torch.device(self.device).type == "cuda"
        return enable_moving_cpu_tensors_to_cuda

    def _init_template(self, module_interface_cls, enable_moving_cpu_tensors_to_cuda):
        """Instantiate template on local side."""
        # 在本地端实例化模板
        generated_module = instantiator.instantiate_scriptable_remote_module_template(
            module_interface_cls, enable_moving_cpu_tensors_to_cuda
        )
        # 存储生成的方法列表
        self.generated_methods = generated_module._generated_methods

    def _check_attribute_picklability(self):
        """Check if all the attribute has explicitly defined whether to be pickled (i.e., picklability)."""
        # 检查所有属性是否已经明确指定是否可以被序列化
        for k in self.__dict__.keys():
            if (
                k not in _REMOTE_MODULE_PICKLED_ATTRIBUTES
                and k not in _REMOTE_MODULE_ATTRIBUTES_IGNORE_FOR_PICKLING
            ):
                # 如果属性未在允许序列化的列表中，则抛出错误
                raise AttributeError(
                    f"Attribute {k} must be either in ``_REMOTE_MODULE_PICKLED_ATTRIBUTES`` or "
                    "``_REMOTE_MODULE_ATTRIBUTES_IGNORE_FOR_PICKLING``."
                )

    def _install_generated_methods(self):
        # 安装生成的方法到当前对象中
        for method in self.generated_methods:
            method_name = method.__name__
            method = torch.jit.export(method)
            setattr(self, method_name, types.MethodType(method, self))

    @staticmethod
    def init_from_module_rref(
        remote_device: str,
        module_rref: rpc.RRef[nn.Module],
        _module_interface_cls: Any = None,
# 定义一个 RemoteModule 类，继承自 _RemoteModule 类
class RemoteModule(_RemoteModule):
    """
    A RemoteModule instance can only be created after RPC initialization.

    It creates a user-specified module on a specified remote node.
    It behaves like a regular ``nn.Module`` except that the ``forward`` method is
    executed on the remote node.
    It takes care of autograd recording to ensure the backward pass propagates
    gradients back to the corresponding remote module.

    It generates two methods ``forward_async`` and ``forward`` based on the
    signature of the ``forward`` method of ``module_cls``. ``forward_async``
    runs asynchronously and returns a Future. The arguments of ``forward_async``
    and ``forward`` are the same as the ``forward`` method of the module
    returned by the ``module_cls``.

    For example, if ``module_cls`` returns an instance of ``nn.Linear``,
    that has ``forward`` method signature: ``def forward(input: Tensor) -> Tensor:``,
    the generated ``RemoteModule`` will have 2 methods with the signatures:

    | ``def forward(input: Tensor) -> Tensor:``
    | ``def forward_async(input: Tensor) -> Future[Tensor]:``

    Args:
        remote_device (str): Device on the destination worker where we'd like to place this module.
            The format should be "<workername>/<device>", where the device field can be parsed as torch.device type.
            E.g., "trainer0/cpu", "trainer0", "ps0/cuda:0".
            In addition, the device field can be optional and the default value is "cpu".
        module_cls (nn.Module): Class for the module to be created remotely. For example,

            >>> class MyModule(nn.Module):
            >>>     def forward(input):
            >>>         return input + 1
            >>>
            >>> module_cls = MyModule

        args (Sequence, optional): args to be passed to ``module_cls``.
        kwargs (Dict, optional): kwargs to be passed to ``module_cls``.

    Returns:
        A remote module instance which wraps the :class:`~nn.Module` created by the
        user-provided ``module_cls``, it has a blocking ``forward`` method and an
        asynchronous ``forward_async`` method that returns a future of the ``forward`` call
        on the user-provided module on the remote side.
    """
    # 定义 RemoteModule 类的初始化方法
    def __init__(
        self,
        remote_device: str,              # 远程设备的名称或标识符，指定模块运行的位置
        module_cls: Type[nn.Module],     # 要远程实例化的 nn.Module 类型
        args: Optional[Tuple] = None,    # 可选参数，用于传递给模块构造函数的位置参数元组
        kwargs: Optional[Dict[str, Any]] = None,  # 可选参数，用于传递给模块构造函数的关键字参数字典
    ):
        # 调用父类（RemoteModule）的构造函数进行初始化
        super().__init__(remote_device, module_cls, args, kwargs)
# 定义函数 _remote_module_receiver，用于反序列化 RemoteModule 对象
def _remote_module_receiver(
    *remote_module_pickled_attrs,
):
    """Deserializes a RemoteModule."""
    # 将传入的序列化属性解包成 _SerializedRemoteModule 对象
    serialized_remote_module = _SerializedRemoteModule._make(
        remote_module_pickled_attrs
    )
    # 使用 object 类的 __new__ 方法创建 RemoteModule 实例
    m = object.__new__(RemoteModule)
    # 将 _SerializedRemoteModule 对象的属性更新到 RemoteModule 实例的 __dict__ 中
    m.__dict__.update(serialized_remote_module._asdict())

    # 对于属性 `module_rref`，进行反序列化，调用 RRef 的 _deserialize() 方法
    m.module_rref = rpc.PyRRef._deserialize(m.module_rref)

    # 在反序列化时安装生成的方法
    for method in m.generated_methods:
        method_name = method.__name__
        # 使用 torch.jit.export() 方法导出方法
        method = torch.jit.export(method)
        # 将方法设置为 RemoteModule 实例的属性
        setattr(m, method_name, types.MethodType(method, m))

    # 返回反序列化后的 RemoteModule 实例
    return m


# 定义函数 _remote_module_reducer，用于序列化 RemoteModule 对象
def _remote_module_reducer(remote_module):
    """Serialize a RemoteModule."""
    # 初始化一个空字典，用于存储序列化后的属性
    pickled_attrs = {}
    # 遍历 RemoteModule 实例的 __dict__ 属性
    for k, v in remote_module.__dict__.items():
        # 对于属性 `module_rref`，进行序列化，调用 RRef 的 _serialize() 方法
        if k == "module_rref":
            pickled_attrs[k] = v._serialize()
        # 如果属性在 _REMOTE_MODULE_PICKLED_ATTRIBUTES 中，则直接存储
        elif k in _REMOTE_MODULE_PICKLED_ATTRIBUTES:
            pickled_attrs[k] = v
        # 检查未序列化的属性是否在 _REMOTE_MODULE_ATTRIBUTES_IGNORE_FOR_PICKLING 中
        elif k not in _REMOTE_MODULE_ATTRIBUTES_IGNORE_FOR_PICKLING:
            # 输出警告信息，提示该属性在 RPC 序列化时被忽略
            print(
                f"The new attribute ``{k}`` of RemoteModule is ignored during RPC pickling. "
                "To pickle this attribute, please add it to ``_REMOTE_MODULE_PICKLED_ATTRIBUTES``. "
                "Otherwise, please explicitly add it to ``_REMOTE_MODULE_ATTRIBUTES_IGNORE_FOR_PICKLING``.",
                file=sys.stderr,
            )

    # 返回一个元组，包含反序列化函数 _remote_module_receiver 和序列化后的属性值
    return (
        _remote_module_receiver,
        tuple(pickled_attrs.values()),
    )


# 定义函数 _recursive_script_module_receiver，用于反序列化不含脚本 RemoteModule 的 RecursiveScriptModule
def _recursive_script_module_receiver(
    recursive_script_module_serialized,
):
    """Deserializes a RecursiveScriptModule that does not contain a script RemoteModule."""
    # 将传入的序列化数据转换成 BytesIO 对象
    f = io.BytesIO(recursive_script_module_serialized)
    # 使用 torch.jit.load() 方法加载模型
    m = torch.jit.load(f)
    # 返回反序列化后的模型对象
    return m


# 定义函数 _recursive_script_module_reducer，用于序列化 RecursiveScriptModule
def _recursive_script_module_reducer(recursive_script_module):
    """Serialize a RecursiveScriptModule that does not contain a script RemoteModule, and raises an error otherwise."""
    # 检查是否有属性 `_c` 中包含 `module_rref`，若有则抛出异常
    if hasattr(recursive_script_module._c, "module_rref"):
        raise RuntimeError(
            "Passing a script RemoteModule over RPC is not supported. Please create a RemoteModule in the sender, "
            "send the `module_rref` to the receiver, and create a new instance on the receiver end by passing this `module_rref`."
        )

    # 创建一个 BytesIO 对象
    f = io.BytesIO()
    # 使用 torch.jit.save() 方法将模型保存到 BytesIO 对象中
    torch.jit.save(recursive_script_module, f)
    # 返回一个元组，包含反序列化函数 _recursive_script_module_receiver 和序列化后的数据
    return (_recursive_script_module_receiver, (f.getvalue(),))


# 注册 RemoteModule 和 RecursiveScriptModule 的序列化和反序列化函数到 _internal_rpc_pickler
_internal_rpc_pickler._register_reducer(RemoteModule, _remote_module_reducer)
_internal_rpc_pickler._register_reducer(
    torch.jit.RecursiveScriptModule, _recursive_script_module_reducer
)
```