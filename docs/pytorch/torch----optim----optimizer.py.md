# `.\pytorch\torch\optim\optimizer.py`

```
# 设置类型别名，用于更清晰地表示不同类型的参数和数据结构
Args: TypeAlias = Tuple[Any, ...]
Kwargs: TypeAlias = Dict[str, Any]
StateDict: TypeAlias = Dict[str, Any]
TensorListList: TypeAlias = List[List[torch.Tensor]]
DeviceDict = Dict[Optional[torch.device], torch.Tensor]

# 全局优化器预处理钩子的类型别名，接受优化器、位置参数和关键字参数，返回一个可选的元组
# 元组中包含处理后的位置参数和关键字参数
GlobalOptimizerPreHook: TypeAlias = Callable[
    ["Optimizer", Args, Kwargs], Optional[Tuple[Args, Kwargs]]
]
# 全局优化器后处理钩子的类型别名，接受优化器、位置参数和关键字参数，无返回值
GlobalOptimizerPostHook: TypeAlias = Callable[["Optimizer", Args, Kwargs], None]

# 导出的模块成员列表，包括"Optimizer"类和两个钩子函数注册方法
__all__ = [
    "Optimizer",
    "register_optimizer_step_pre_hook",
    "register_optimizer_step_post_hook",
]

# 全局变量：存储全局优化器预处理钩子的有序字典，键为整数，值为预处理钩子函数
_global_optimizer_pre_hooks: Dict[int, GlobalOptimizerPreHook] = OrderedDict()
# 全局变量：存储全局优化器后处理钩子的有序字典，键为整数，值为后处理钩子函数
_global_optimizer_post_hooks: Dict[int, GlobalOptimizerPostHook] = OrderedDict()

# 支持foreach操作的数据类型列表，包括torch.Tensor和torch.nn.parameter.Parameter
_foreach_supported_types = [torch.Tensor, torch.nn.parameter.Parameter]

# 内部类：表示优化器中一个必需的参数的单例类
class _RequiredParameter:
    """Singleton class representing a required parameter for an Optimizer."""

    # 返回类的字符串表示形式
    def __repr__(self) -> str:
        return "<required parameter>"

# 单例对象：表示优化器中一个必需的参数
required = _RequiredParameter()

# 函数装饰器：允许使用梯度进行不同iable的功能
def _use_grad_for_differentiable(func):
    # 定义一个名为 _use_grad 的方法，接受任意参数和关键字参数
    def _use_grad(self, *args, **kwargs):
        # 导入 torch._dynamo 模块
        import torch._dynamo

        # 保存当前梯度是否开启的状态
        prev_grad = torch.is_grad_enabled()
        try:
            # 设置梯度是否开启为指定的默认状态
            torch.set_grad_enabled(self.defaults["differentiable"])
            # 执行图中断，确保 AOT（Ahead of Time）编译尊重 no_grad 注释
            # 这对性能很重要，因为如果没有这个步骤，功能化会生成一个 epilogue
            # 该 epilogue 会更新优化器中变异的参数，但这对 inductor 不可见
            # 因此，inductor 会为模型中的每个参数分配内存，这是不好的
            # 通过这个步骤，AOT 正确地将其识别为推断图，并生成附加到图中的 epilogue
            # 这样 inductor 就能看到 step 是就地操作，并能避免额外的内存分配
            # 在将来，我们要么继续在反向传播时进行图中断，这样图中断就不重要了
            # 要么有一个完全融合的前向和后向图，该图将默认为 no_grad，我们可以移除这个图中断
            # 以允许编译完全融合的前向-后向-优化器图
            # 参见 https://github.com/pytorch/pytorch/issues/104053
            torch._dynamo.graph_break()
            # 调用 func 函数，并返回其结果
            ret = func(self, *args, **kwargs)
        finally:
            # 再次执行图中断
            torch._dynamo.graph_break()
            # 恢复之前保存的梯度开启状态
            torch.set_grad_enabled(prev_grad)
        # 返回 func 函数的结果
        return ret

    # 将 _use_grad 方法的属性和文档字符串更新为 func 函数的属性和文档字符串
    functools.update_wrapper(_use_grad, func)
    # 返回 _use_grad 方法
    return _use_grad
# item is significantly faster than a cpu tensor in eager mode
def _get_value(x):
    # 如果不是TorchScript模式且正在编译，则直接返回输入x
    if not torch.jit.is_scripting() and is_compiling():
        return x
    else:
        # 如果x是torch.Tensor类型，则返回其标量值；否则直接返回x
        return x.item() if isinstance(x, torch.Tensor) else x


def _stack_if_compiling(x):
    # 如果不是TorchScript模式且正在编译，则将输入x堆叠成张量
    if not torch.jit.is_scripting() and is_compiling():
        return torch.stack(x)
    else:
        # 否则直接返回输入x
        return x


def _dispatch_sqrt(
    x: float,
):  # 由于torchscript类型推断的需要，这里需要float注释
    # 如果不是TorchScript模式且x是torch.Tensor类型，则返回x的平方根
    if not torch.jit.is_scripting() and isinstance(x, torch.Tensor):
        return x.sqrt()
    else:
        # 否则返回x的数学平方根
        return math.sqrt(x)


def _disable_dynamo_if_unsupported(single_tensor_fn=None):
    # 为了兼容TorchScript，需要将所有调用的函数置于创建maybe_fallback闭包的位置的全局环境中
    if single_tensor_fn:
        # 将单张量函数注册到全局环境中
        globals()[single_tensor_fn.__name__] = single_tensor_fn

    def wrapper(func):
        import inspect

        # 禁用Dynamo系统对于给定函数
        disabled_func = torch._disable_dynamo(func)
        ps = inspect.signature(func).parameters
        has_state_steps = True
        try:
            # 检查函数签名中是否有"state_steps"参数
            state_steps_ind = list(ps.keys()).index("state_steps")
        except ValueError:
            has_state_steps = False

        # 今天，存在一些情况，我们堆叠状态步骤并将它们作为foreach操作的值参数传递。
        # 在eager模式下，将状态步骤作为值参数传递到cuda上不受支持，但这仅在用户显式删除capturable标志的罕见情况下发生。
        # 如果capturable=True，则不会出现问题。
        @functools.wraps(func)
        def maybe_fallback(*args, **kwargs):
            if is_compiling() and (
                not kwargs.get("capturable", False)
                and has_state_steps
                and (args[state_steps_ind] and args[state_steps_ind][0].is_cuda)
                or (
                    "state_steps" in kwargs
                    and kwargs["state_steps"]
                    and kwargs["state_steps"][0].is_cuda
                )
            ):
                # 如果正在编译且满足特定条件，则调用禁用的函数
                return disabled_func(*args, **kwargs)
            else:
                # 否则正常调用原函数
                return func(*args, **kwargs)

        return maybe_fallback

    return wrapper


# 对于任何带有更快实现的优化器，我们尽可能地默认选择最快且最稳定的实现方式。
# 对于foreach，要求所有本地参数都在CUDA上；对于fused，还要求张量的数据类型必须是浮点数。
# 两者都不支持torch.jit.script或可微分，因此在这些情况下，我们退回到单张量实现。
def _default_to_fused_or_foreach(
    params: List[torch.Tensor], differentiable: bool, use_fused: bool = False
) -> Tuple[bool, bool]:
    # 如果是TorchScript模式或者要求可微分，则直接返回False, False
    if torch.jit.is_scripting() or differentiable:
        return False, False

    # 获取支持融合内核的设备列表
    fused_supported_devices = _get_fused_kernels_supported_devices()
    # 获取支持的设备列表，用于后续判断
    foreach_supported_devices = _get_foreach_kernels_supported_devices()
    
    # 判断是否使用融合操作，并且所有参数都满足以下条件：
    # 1. 参数为 None，或者
    # 2. 参数的类型在支持的类型列表中，并且参数的设备类型在融合操作支持的设备列表中，并且参数是浮点数类型
    fused = use_fused and all(
        p is None
        or (
            type(p) in _foreach_supported_types
            and p.device.type in fused_supported_devices
            and torch.is_floating_point(p)
        )
        for p in params
    )
    
    # 如果没有使用融合操作，且所有参数都满足以下条件：
    # 1. 参数为 None，或者
    # 2. 参数的类型在支持的类型列表中，并且参数的设备类型在不同操作支持的设备列表中
    foreach = not fused and all(
        p is None
        or (
            type(p) in _foreach_supported_types
            and p.device.type in foreach_supported_devices
        )
        for p in params
    )
    
    # 返回是否使用融合操作和是否使用 foreach 操作的布尔值
    return fused, foreach
# 将复杂张量列表中的每个复杂张量视为实部张量
def _view_as_real(params, *state_and_grads):
    # 遍历参数列表中的每个参数及其索引
    for i, p in enumerate(params):
        # 检查当前参数是否为复数类型
        if torch.is_complex(p):
            # 如果是复数，则将其视为实部张量
            params[i] = torch.view_as_real(params[i])
            # 遍历状态和梯度列表中的每个元素
            for s in state_and_grads:
                # 将对应的状态或梯度张量视为实部张量
                s[i] = torch.view_as_real(s[i])


# 根据是否融合返回标量数据类型
def _get_scalar_dtype(is_fused=None):
    # 如果 is_fused 参数为真，则返回 float32 类型
    if is_fused:
        return torch.float32
    # 否则根据默认数据类型返回 float64 或 float32 类型
    return (
        torch.float64 if torch.get_default_dtype() == torch.float64 else torch.float32
    )


# 返回支持 capturable 优化器的设备类型列表
def _get_capturable_supported_devices(supports_xla: bool = True) -> List[str]:
    # 初始化支持 capturable 优化器的设备类型列表，至少包含 "cuda"
    capturable_supported_devices = ["cuda"]
    # 如果当前未处于脚本模式，则添加私有使用后端名称
    if not torch.jit.is_scripting():
        capturable_supported_devices.append(torch._C._get_privateuse1_backend_name())
    # 如果支持 XLA，则添加 "xla"
    if supports_xla:
        capturable_supported_devices.append("xla")
    return capturable_supported_devices


# 优化器共用文档字符串模板
_foreach_doc = r"""foreach (bool, optional): 是否使用 foreach 实现优化器
            如果用户未指定（foreach 为 None），我们将尝试在 CUDA 上使用 foreach 而不是 for 循环实现优化器，
            因为通常更高效。注意，由于中间结果是 tensorlist 而不是单个张量，foreach 实现使用的峰值内存大约
            大小为 sizeof(params)。如果内存有限，可以一次批处理更少的参数，或将此标志切换为 False（默认值：None）"""

# 融合优化器共用文档字符串模板
_fused_doc = r"""fused (bool, optional): 是否使用融合实现优化器
            当前支持 `torch.float64`, `torch.float32`, `torch.float16`, 和 `torch.bfloat16`。
            （默认值：None）

    .. 注意:: foreach 和融合实现通常比 for 循环单张量实现更快。因此，如果用户未同时指定两个标志
              （例如，当 foreach = fused = None 时），我们将尝试在所有张量都在 CUDA 上时默认使用 foreach
              实现。例如，如果用户为 fused 指定了 True，但没有指定 foreach，则我们将运行融合实现。
              如果用户为 foreach 指定了 False，但没有为 fused 指定任何内容（或者为 fused 指定了 False，
              但没有为 foreach 指定任何内容），我们将运行 for 循环实现。如果用户同时为 foreach 和
              fused 指定了 True，则我们将优先选择融合实现，因为它通常更快。我们尽力使用最快的实现，
              因此优先级为融合 -> foreach -> for-loop。但是，由于融合实现相对较新，我们希望给它足够的
              时间进行验证，因此在用户未指定任何标志时，默认为 foreach 而不是融合。"""
_capturable_doc = r"""capturable (bool, optional): whether this instance is safe to
            capture in a CUDA graph. Passing True can impair ungraphed performance,
            so if you don't intend to graph capture this instance, leave it False
            (default: False)"""
# 用于描述是否可以在 CUDA 图中捕获该实例的文档字符串
_differentiable_doc = r"""differentiable (bool, optional): whether autograd should
            occur through the optimizer step in training. Otherwise, the step()
            function runs in a torch.no_grad() context. Setting to True can impair
            performance, so leave it False if you don't intend to run autograd
            through this instance (default: False)"""
# 用于描述是否应该通过优化器步骤进行自动求导的文档字符串
_maximize_doc = r"""maximize (bool, optional): maximize the objective with respect to the
            params, instead of minimizing (default: False)"""
# 用于描述是否应最大化目标相对于参数的文档字符串


def register_optimizer_step_pre_hook(hook: GlobalOptimizerPreHook) -> RemovableHandle:
    r"""Register a pre hook common to all optimizers.

    The hook should have the following signature::

        hook(optimizer, args, kwargs) -> None or modified args and kwargs

    Args:
        hook (Callable): A user defined hook which is registered on all optimizers.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
    # 创建一个前置钩子，用于所有优化器的注册
    handle = hooks.RemovableHandle(_global_optimizer_pre_hooks)
    # 将钩子添加到全局优化器前置钩子字典中
    _global_optimizer_pre_hooks[handle.id] = hook
    return handle
    # 返回一个可用于移除已添加钩子的句柄


def register_optimizer_step_post_hook(hook: GlobalOptimizerPostHook) -> RemovableHandle:
    r"""Register a post hook common to all optimizers.

    The hook should have the following signature::

        hook(optimizer, args, kwargs) -> None

    Args:
        hook (Callable): A user defined hook which is registered on all optimizers.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
    # 创建一个后置钩子，用于所有优化器的注册
    handle = hooks.RemovableHandle(_global_optimizer_post_hooks)
    # 将钩子添加到全局优化器后置钩子字典中
    _global_optimizer_post_hooks[handle.id] = hook
    return handle
    # 返回一个可用于移除已添加钩子的句柄


ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]

_P = ParamSpec("_P")
R = TypeVar("R")
T = TypeVar("T")


class Optimizer:
    r"""Base class for all optimizers.

    .. warning::
        Parameters need to be specified as collections that have a deterministic
        ordering that is consistent between runs. Examples of objects that don't
        satisfy those properties are sets and iterators over values of dictionaries.

    Args:
        params (iterable): an iterable of :class:`torch.Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).
    """
    # 优化器的基类，用于所有优化器的基础实现
    # 定义类型别名，表示预优化器钩子函数类型，用于修改优化步骤前的参数和关键字参数，忽略类型检查警告
    OptimizerPreHook: TypeAlias = Callable[[Self, Args, Kwargs], Optional[Tuple[Args, Kwargs]]]  # type: ignore[misc]
    # 定义类型别名，表示后优化器钩子函数类型，用于优化步骤后的操作，忽略类型检查警告
    OptimizerPostHook: TypeAlias = Callable[[Self, Args, Kwargs], None]  # type: ignore[misc]
    
    # 用于存储优化器步骤前钩子函数的字典，键为整数，值为 OptimizerPreHook 函数
    _optimizer_step_pre_hooks: Dict[int, OptimizerPreHook]
    # 用于存储优化器步骤后钩子函数的字典，键为整数，值为 OptimizerPostHook 函数
    _optimizer_step_post_hooks: Dict[int, OptimizerPostHook]
    # 用于存储优化器状态字典序列化前钩子函数的有序字典，键为整数，值为接收 Optimizer 对象并返回 None 的函数
    _optimizer_state_dict_pre_hooks: 'OrderedDict[int, Callable[["Optimizer"], None]]'
    # 用于存储优化器状态字典序列化后钩子函数的有序字典，键为整数，值为接收 Optimizer 和 StateDict 返回可选 StateDict 的函数
    _optimizer_state_dict_post_hooks: 'OrderedDict[int, Callable[["Optimizer", StateDict], Optional[StateDict]]]'
    # 用于存储加载优化器状态字典前钩子函数的有序字典，键为整数，值为接收 Optimizer 和 StateDict 返回可选 StateDict 的函数
    _optimizer_load_state_dict_pre_hooks: 'OrderedDict[int, Callable[["Optimizer", StateDict], Optional[StateDict]]]'
    # 用于存储加载优化器状态字典后钩子函数的有序字典，键为整数，值为接收 Optimizer 的函数
    _optimizer_load_state_dict_post_hooks: 'OrderedDict[int, Callable[["Optimizer"], None]]'
    
    # 初始化函数，接收参数 params 和 defaults
    def __init__(self, params: ParamsT, defaults: Dict[str, Any]) -> None:  # noqa: D107
        # 记录使用了 torch.optimizer 模块的 API
        torch._C._log_api_usage_once("python.optimizer")
        self.defaults = defaults
        # 初始化优化器步骤前钩子函数字典
        self._optimizer_step_pre_hooks = OrderedDict()
        # 初始化优化器步骤后钩子函数字典
        self._optimizer_step_post_hooks = OrderedDict()
        # 初始化优化器状态字典序列化前钩子函数的有序字典
        self._optimizer_state_dict_pre_hooks = OrderedDict()
        # 初始化优化器状态字典序列化后钩子函数的有序字典
        self._optimizer_state_dict_post_hooks = OrderedDict()
        # 初始化加载优化器状态字典前钩子函数的有序字典
        self._optimizer_load_state_dict_pre_hooks = OrderedDict()
        # 初始化加载优化器状态字典后钩子函数的有序字典
        self._optimizer_load_state_dict_post_hooks = OrderedDict()
    
        # 调用 _patch_step_function 方法来进行步骤函数的补丁操作
        self._patch_step_function()
    
        # 检查 params 是否为 torch.Tensor 类型，如果是则抛出类型错误异常
        if isinstance(params, torch.Tensor):
            raise TypeError(
                "params argument given to the optimizer should be "
                "an iterable of Tensors or dicts, but got " + torch.typename(params)
            )
    
        # 使用 defaultdict 创建 self.state，用于存储张量到任意类型值的映射
        self.state: DefaultDict[torch.Tensor, Any] = defaultdict(dict)
        # 初始化 param_groups 为空列表
        self.param_groups: List[Dict[str, Any]] = []
    
        # 将 params 转换为列表形式，如果长度为 0，则抛出数值错误异常
        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        # 如果 param_groups 的第一个元素不是字典，则将其包装成字典形式
        if not isinstance(param_groups[0], dict):
            param_groups = [{"params": param_groups}]
    
        # 遍历 param_groups 列表，对每个 param_group 调用 add_param_group 方法
        for param_group in param_groups:
            self.add_param_group(cast(dict, param_group))
    
        # 允许 _cuda_graph_capture_health_check 来设置一个类似 TORCH_WARN_ONCE 的警告机制
        # 在 Python 中通过这种方式实现，详细信息参见链接
        self._warned_capturable_if_run_uncaptured = True
    
    # __getstate__ 方法用于返回对象的序列化状态
    def __getstate__(self) -> Dict[str, Any]:  # noqa: D105
        return {
            "defaults": self.defaults,
            "state": self.state,
            "param_groups": self.param_groups,
        }
    # 定义 __setstate__ 方法，用于反序列化对象状态
    def __setstate__(self, state: Dict[str, Any]) -> None:  # noqa: D105
        # 更新对象的 __dict__ 属性，用传入的状态更新对象的属性
        self.__dict__.update(state)
        
        # 如果对象的 __dict__ 中没有 "_optimizer_step_pre_hooks" 键，则创建一个有序字典
        if "_optimizer_step_pre_hooks" not in self.__dict__:
            self._optimizer_step_pre_hooks = OrderedDict()
        
        # 如果对象的 __dict__ 中没有 "_optimizer_step_post_hooks" 键，则创建一个有序字典
        if "_optimizer_step_post_hooks" not in self.__dict__:
            self._optimizer_step_post_hooks = OrderedDict()
        
        # 如果对象的 __dict__ 中没有 "_optimizer_state_dict_pre_hooks" 键，则创建一个有序字典
        if "_optimizer_state_dict_pre_hooks" not in self.__dict__:
            self._optimizer_state_dict_pre_hooks = OrderedDict()
        
        # 如果对象的 __dict__ 中没有 "_optimizer_state_dict_post_hooks" 键，则创建一个有序字典
        if "_optimizer_state_dict_post_hooks" not in self.__dict__:
            self._optimizer_state_dict_post_hooks = OrderedDict()
        
        # 如果对象的 __dict__ 中没有 "_optimizer_load_state_dict_pre_hooks" 键，则创建一个有序字典
        if "_optimizer_load_state_dict_pre_hooks" not in self.__dict__:
            self._optimizer_load_state_dict_pre_hooks = OrderedDict()
        
        # 如果对象的 __dict__ 中没有 "_optimizer_load_state_dict_post_hooks" 键，则创建一个有序字典
        if "_optimizer_load_state_dict_post_hooks" not in self.__dict__:
            self._optimizer_load_state_dict_post_hooks = OrderedDict()
        
        # 调用对象的 _patch_step_function 方法，支持多进程的 pickle/unpickle 操作
        self._patch_step_function()  # To support multiprocessing pickle/unpickle
        
        # 如果默认设置中没有 "differentiable" 键，则将其设置为 False
        self.defaults.setdefault("differentiable", False)

    # 定义 __repr__ 方法，返回对象的字符串表示形式
    def __repr__(self) -> str:  # noqa: D105
        # 初始化格式字符串为对象类名 + "("
        format_string = self.__class__.__name__ + " ("
        
        # 遍历参数组列表 self.param_groups
        for i, group in enumerate(self.param_groups):
            format_string += "\n"  # 添加换行符
            format_string += f"Parameter Group {i}\n"  # 添加参数组编号
            
            # 对参数组字典中的键进行排序，并遍历
            for key in sorted(group.keys()):
                if key != "params":
                    # 将参数组的键值对添加到格式字符串中
                    format_string += f"    {key}: {group[key]}\n"
        
        format_string += ")"  # 添加 ")" 结束符
        
        # 返回格式化后的字符串表示形式
        return format_string

    # 当前仅被 Adam 和 AdamW 使用
    def _cuda_graph_capture_health_check(self) -> None:
        # Note [torch.compile x capturable]
        # 如果正在进行编译，我们尝试在追踪期间自动采用可捕获路径，通过设置标志为True。
        # 因此，我们跳过通常用于确定是否可以使用CUDA图形的所有检查，并将责任转移到torch.inductor。
        # 这在追踪期间节省时间，因为这些检查通常很慢，但不会牺牲用户体验，因为inductor稍后会发出警告，
        # 如果无法启用CUDA图形，例如，
        # https://github.com/pytorch/pytorch/blob/d3ba8901d8640eb16f88b2bfef9df7fa383d4b47/torch/_inductor/compile_fx.py#L390。
        # 因此，在编译时，inductor将根据是否存在输入变化或CPU张量来确定是否可以启用CUDA图形。
        if (
            not is_compiling()
            and torch.backends.cuda.is_built()
            and torch.cuda.is_available()
        ):
            # 检查当前流是否正在捕获CUDA图形
            capturing = torch.cuda.is_current_stream_capturing()

            # 如果正在捕获CUDA图形，但其中某些param_group的"capturable"为False，则引发异常
            if capturing and not all(
                group["capturable"] for group in self.param_groups
            ):
                raise RuntimeError(
                    "Attempting CUDA graph capture of step() for an instance of "
                    + self.__class__.__name__
                    + " but param_groups' capturable is False."
                )

            # 如果之前未发出过警告，并且所有param_group的"capturable"均为True，但未进行CUDA图形捕获，则发出警告
            if (
                (not getattr(self, "_warned_capturable_if_run_uncaptured", False))
                and all(group["capturable"] for group in self.param_groups)
                and (not capturing)
            ):
                warnings.warn(
                    "This instance was constructed with capturable=True or some of all the param_groups came with capturable=True, "
                    "but step() is running without CUDA graph capture. If you never intend to graph-capture this "
                    "instance, capturable=True can impair performance, and you should set capturable=False."
                )
                self._warned_capturable_if_run_uncaptured = True

    def _optimizer_step_code(self) -> None:
        """Entry point for `torch.profile.profiler`.

        When python tracing is enabled the profiler will hook into this
        function at the CPython level to inspect the optimizer's parameters and
        param groups. It is called it after `step()` since many optimizers
        lazily initialize state.

        This is a workaround due to lack of a proper step hook on the optimizer,
        and will be removed if it exists.
        """
        pass

    @staticmethod
    def profile_hook_step(func: Callable[_P, R]) -> Callable[_P, R]:  # noqa: D102
        """Decorator function to profile the execution of a step function.

        Wraps the input function to add profiling capabilities using torch.autograd.profiler.record_function.
        """

        @functools.wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> R:
            """Wrapper function that profiles the execution of the decorated step function.

            Profiles the execution by calling pre and post hooks and records the function's execution.
            """
            self, *_ = args
            self = cast(Optimizer, self)
            profile_name = f"Optimizer.step#{self.__class__.__name__}.step"
            with torch.autograd.profiler.record_function(profile_name):
                # call optimizer step pre hooks
                for pre_hook in chain(
                    _global_optimizer_pre_hooks.values(),
                    self._optimizer_step_pre_hooks.values(),
                ):
                    result = pre_hook(self, args, kwargs)
                    if result is not None:
                        if isinstance(result, tuple) and len(result) == 2:
                            args, kwargs = result  # type: ignore[assignment]
                        else:
                            raise RuntimeError(
                                f"{func} must return None or a tuple of (new_args, new_kwargs), but got {result}."
                            )

                out = func(*args, **kwargs)
                self._optimizer_step_code()

                # call optimizer step post hooks
                for post_hook in chain(
                    self._optimizer_step_post_hooks.values(),
                    _global_optimizer_post_hooks.values(),
                ):
                    post_hook(self, args, kwargs)

                return out

        return wrapper

    @staticmethod
    def _group_tensors_by_device_and_dtype(
        tensorlistlist: TensorListList,
        with_indices: bool = False,
    ) -> Union[
        Dict[Tuple[None, None], Tuple[TensorListList, Indices]],
        Dict[Tuple[torch.device, torch.dtype], Tuple[TensorListList, Indices]],
    ]:
        """Group a list of lists of tensors by device and dtype.

        Skips this step if we are compiling since this will occur during inductor lowering.
        """
        if is_compiling():
            return {(None, None): (tensorlistlist, list(range(len(tensorlistlist[0]))))}
        else:
            return _group_tensors_by_device_and_dtype(tensorlistlist, with_indices)  # type: ignore[return-value, arg-type]

    def _patch_step_function(self) -> None:
        """Patch the step function of the optimizer class for profiling.

        Assigns a profiled version of the step function to the optimizer class if not already patched.
        """
        self._zero_grad_profile_name = (
            f"Optimizer.zero_grad#{self.__class__.__name__}.zero_grad"
        )
        hooked = getattr(self.__class__.step, "hooked", None)
        if not hooked:
            self.__class__.step = self.profile_hook_step(self.__class__.step)  # type: ignore[assignment]
            self.__class__.step.hooked = True  # type: ignore[attr-defined]
    def register_step_pre_hook(self, hook: OptimizerPreHook) -> RemovableHandle:
        r"""Register an optimizer step pre hook which will be called before optimizer step.

        It should have the following signature::

            hook(optimizer, args, kwargs) -> None or modified args and kwargs

        The ``optimizer`` argument is the optimizer instance being used. If
        args and kwargs are modified by the pre-hook, then the transformed
        values are returned as a tuple containing the new_args and new_kwargs.

        Args:
            hook (Callable): The user defined hook to be registered.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        # 创建一个可移除的句柄，用于存储预处理步骤钩子的字典
        handle = hooks.RemovableHandle(self._optimizer_step_pre_hooks)
        # 将传入的 hook 添加到 _optimizer_step_pre_hooks 字典中，使用 handle.id 作为键
        self._optimizer_step_pre_hooks[handle.id] = hook
        # 返回句柄，以便用户可以使用 handle.remove() 来移除添加的钩子
        return handle

    def register_step_post_hook(self, hook: OptimizerPostHook) -> RemovableHandle:
        r"""Register an optimizer step post hook which will be called after optimizer step.

        It should have the following signature::

            hook(optimizer, args, kwargs) -> None

        The ``optimizer`` argument is the optimizer instance being used.

        Args:
            hook (Callable): The user defined hook to be registered.

        Returns:
            :class:`torch.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        # 创建一个可移除的句柄，用于存储后处理步骤钩子的字典
        handle = hooks.RemovableHandle(self._optimizer_step_post_hooks)
        # 将传入的 hook 添加到 _optimizer_step_post_hooks 字典中，使用 handle.id 作为键
        self._optimizer_step_post_hooks[handle.id] = hook
        # 返回句柄，以便用户可以使用 handle.remove() 来移除添加的钩子
        return handle

    def register_state_dict_pre_hook(
        self, hook: Callable[["Optimizer"], None], prepend: bool = False
    ):
        r"""Register a hook to modify the state_dict before it is serialized.

        This hook will be called when :meth:`state_dict` is called. It allows
        modifying the state_dict before it is serialized and saved.

        Args:
            hook (Callable): The hook function to be registered.
                It should have the following signature: ``hook(optimizer) -> None``.
            prepend (bool, optional): If True, the hook will be prepended (inserted at the beginning).
                Defaults to False.
        """
        # 注册一个用于修改状态字典的钩子函数，用于在序列化之前调用
        # 这个钩子将在调用 state_dict 方法时被调用，允许在序列化和保存之前修改状态字典
        if prepend:
            # 如果 prepend 参数为 True，则将钩子添加到列表的开头
            self._state_dict_pre_hooks.insert(0, hook)
        else:
            # 否则，将钩子添加到列表的末尾
            self._state_dict_pre_hooks.append(hook)
    ) -> RemovableHandle:  # noqa: D101
    r"""Register a state dict pre-hook which will be called before :meth:`~torch.optim.Optimizer.state_dict` is called.

    It should have the following signature::

        hook(optimizer) -> None

    The ``optimizer`` argument is the optimizer instance being used.
    The hook will be called with argument ``self`` before calling ``state_dict`` on ``self``.
    The registered hook can be used to perform pre-processing before the ``state_dict``
    call is made.

    Args:
        hook (Callable): The user defined hook to be registered.
        prepend (bool): If True, the provided pre ``hook`` will be fired before
            all the already registered pre-hooks on ``state_dict``. Otherwise,
            the provided ``hook`` will be fired after all the already registered
            pre-hooks. (default: False)

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
    handle = hooks.RemovableHandle(self._optimizer_state_dict_pre_hooks)
    self._optimizer_state_dict_pre_hooks[handle.id] = hook
    if prepend:
        self._optimizer_state_dict_pre_hooks.move_to_end(handle.id, last=False)
    return handle

def register_state_dict_post_hook(
    self,
    hook: Callable[["Optimizer", StateDict], Optional[StateDict]],
    prepend: bool = False,
) -> RemovableHandle:
    r"""Register a state dict post-hook which will be called after :meth:`~torch.optim.Optimizer.state_dict` is called.

    It should have the following signature::

        hook(optimizer, state_dict) -> state_dict or None

    The hook will be called with arguments ``self`` and ``state_dict`` after generating
    a ``state_dict`` on ``self``. The hook may modify the state_dict inplace or optionally
    return a new one. The registered hook can be used to perform post-processing
    on the ``state_dict`` before it is returned.

    Args:
        hook (Callable): The user defined hook to be registered.
        prepend (bool): If True, the provided post ``hook`` will be fired before
            all the already registered post-hooks on ``state_dict``. Otherwise,
            the provided ``hook`` will be fired after all the already registered
            post-hooks. (default: False)

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
    handle = hooks.RemovableHandle(self._optimizer_state_dict_post_hooks)
    self._optimizer_state_dict_post_hooks[handle.id] = hook
    if prepend:
        self._optimizer_state_dict_post_hooks.move_to_end(handle.id, last=False)
    return handle
    # 禁用 torch._disable_dynamo 装饰器，确保该方法不会被动态改变
    @torch._disable_dynamo
    # 将方法标记为静态方法，即不需要实例化对象即可调用
    @staticmethod
    # 根据参数策略处理给定的参数值
    def _process_value_according_to_param_policy(
        # param: torch.Tensor，表示待处理的参数张量
        param: torch.Tensor,
        # value: torch.Tensor，表示待处理的值张量
        value: torch.Tensor,
        # param_id: int，参数在 param_groups 中的索引
        param_id: int,
        # param_groups: List[Dict[Any, Any]]，包含参数组相关信息的列表
        param_groups: List[Dict[Any, Any]],
        # key: Hashable = None，可选参数，用于特定键的处理
        key: Hashable = None,
    ) -> torch.Tensor:
        # 浮点类型参数特殊处理，假设其类型总是与 params 一致
        # 确保 state['step'] 没有被强制转换 https://github.com/pytorch/pytorch/issues/74424
        # 除非是融合的或可捕获的情况，参见 [special device hosting for step] 注释
        fused = False
        capturable = False
        # 断言 param_groups 不为 None
        assert param_groups is not None
        # 遍历 param_groups 中的每个参数组 pg
        for pg in param_groups:
            # 如果 param_id 在 pg["params"] 中
            if param_id in pg["params"]:
                # 获取融合标志，若不存在则为 False
                fused = pg["fused"] if "fused" in pg else False
                # 获取可捕获标志，若不存在则为 False
                capturable = pg["capturable"] if "capturable" in pg else False
                break
        # 如果 key 为 "step"
        if key == "step":
            # 若可捕获或融合，则将 value 转换为 torch.float32 类型，并发送到 param 所在的设备
            if capturable or fused:
                return value.to(dtype=torch.float32, device=param.device)
            else:
                # 否则直接返回 value
                return value
        else:
            # 如果 param 是浮点类型
            if param.is_floating_point():
                # 将 value 转换为与 param 相同的数据类型，并发送到 param 所在的设备
                return value.to(dtype=param.dtype, device=param.device)
            else:
                # 否则将 value 发送到 param 所在的设备
                return value.to(device=param.device)

    # 注册 load_state_dict 前置钩子函数
    def register_load_state_dict_pre_hook(
        # hook: Callable[["Optimizer", StateDict], Optional[StateDict]]，表示加载状态字典前的钩子函数
        hook: Callable[["Optimizer", StateDict], Optional[StateDict]],
        # prepend: bool = False，可选参数，指示是否将钩子函数添加到前面
        prepend: bool = False,
    ) -> RemovableHandle:  # noqa: D205 D400
        r"""Register a load_state_dict pre-hook which will be called before
        :meth:`~torch.optim.Optimizer.load_state_dict` is called. It should have the
        following signature::

            hook(optimizer, state_dict) -> state_dict or None

        The ``optimizer`` argument is the optimizer instance being used and the
        ``state_dict`` argument is a shallow copy of the ``state_dict`` the user
        passed in to ``load_state_dict``. The hook may modify the state_dict inplace
        or optionally return a new one. If a state_dict is returned, it will be used
        to be loaded into the optimizer.

        The hook will be called with argument ``self`` and ``state_dict`` before
        calling ``load_state_dict`` on ``self``. The registered hook can be used to
        perform pre-processing before the ``load_state_dict`` call is made.

        Args:
            hook (Callable): The user defined hook to be registered.
                用户定义的钩子函数，用于注册
            prepend (bool): If True, the provided pre ``hook`` will be fired before
                all the already registered pre-hooks on ``load_state_dict``. Otherwise,
                the provided ``hook`` will be fired after all the already registered
                pre-hooks. (default: False)
                如果为True，则提供的预先定义的“hook”将在所有已注册的预钩子函数之前被调用。
                否则，提供的“hook”将在所有已注册的预钩子函数之后被调用。

        Returns:
            :class:`torch.utils.hooks.RemoveableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
                可以通过调用 ``handle.remove()`` 来移除添加的钩子函数的句柄
        """
        handle = hooks.RemovableHandle(self._optimizer_load_state_dict_pre_hooks)
        self._optimizer_load_state_dict_pre_hooks[handle.id] = hook
        if prepend:
            self._optimizer_load_state_dict_pre_hooks.move_to_end(handle.id, last=False)
        return handle
    ) -> RemovableHandle:  # noqa: D205 D400
        r"""Register a load_state_dict post-hook which will be called after
        :meth:`~torch.optim.Optimizer.load_state_dict` is called. It should have the
        following signature::

            hook(optimizer) -> None

        The ``optimizer`` argument is the optimizer instance being used.

        The hook will be called with argument ``self`` after calling
        ``load_state_dict`` on ``self``. The registered hook can be used to
        perform post-processing after ``load_state_dict`` has loaded the
        ``state_dict``.

        Args:
            hook (Callable): The user defined hook to be registered.
            prepend (bool): If True, the provided post ``hook`` will be fired before
                all the already registered post-hooks on ``load_state_dict``. Otherwise,
                the provided ``hook`` will be fired after all the already registered
                post-hooks. (default: False)

        Returns:
            :class:`torch.utils.hooks.RemoveableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        # 创建一个可移除的处理器实例，用于管理加载状态字典后钩子函数的注册
        handle = hooks.RemovableHandle(self._optimizer_load_state_dict_post_hooks)
        # 将用户定义的钩子函数 hook 添加到 _optimizer_load_state_dict_post_hooks 字典中
        self._optimizer_load_state_dict_post_hooks[handle.id] = hook
        # 如果 prepend 为 True，则将新添加的钩子函数 handle 移动到已注册钩子函数之前
        if prepend:
            self._optimizer_load_state_dict_post_hooks.move_to_end(handle.id, last=False)  # type: ignore[attr-defined]
        # 返回创建的 handle，用于后续移除该钩子函数
        return handle

    @torch._disable_dynamo
    @torch._disable_dynamo
    def zero_grad(self, set_to_none: bool = True) -> None:
        r"""Reset the gradients of all optimized :class:`torch.Tensor` s.

        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                This will in general have lower memory footprint, and can modestly improve performance.
                However, it changes certain behaviors. For example:
                1. When the user tries to access a gradient and perform manual ops on it,
                a None attribute or a Tensor full of 0s will behave differently.
                2. If the user requests ``zero_grad(set_to_none=True)`` followed by a backward pass, ``.grad``\ s
                are guaranteed to be None for params that did not receive a gradient.
                3. ``torch.optim`` optimizers have a different behavior if the gradient is 0 or None
                (in one case it does the step with a gradient of 0 and in the other it skips
                the step altogether).
        """
        # Check if 'foreach' or 'fused' is in self.defaults and set foreach flag accordingly
        foreach = self.defaults.get("foreach", False) or self.defaults.get(
            "fused", False
        )

        # Patch step function if it's not already patched
        if not hasattr(self, "_zero_grad_profile_name"):
            self._patch_step_function()

        # Define a nested defaultdict structure to store gradients per device and dtype
        per_device_and_dtype_grads: Optional[
            DefaultDict[torch.device, DefaultDict[torch.dtype, List[torch.Tensor]]]
        ]
        if foreach:
            per_device_and_dtype_grads = defaultdict(lambda: defaultdict(list))
        else:
            per_device_and_dtype_grads = None

        # Use torch.autograd.profiler.record_function to profile the zero_grad operation
        with torch.autograd.profiler.record_function(self._zero_grad_profile_name):
            # Iterate over each parameter group in self.param_groups
            for group in self.param_groups:
                # Iterate over each parameter 'p' in the current group
                for p in group["params"]:
                    # Check if 'p' has a valid gradient
                    if p.grad is not None:
                        # If set_to_none is True, set gradient to None
                        if set_to_none:
                            p.grad = None
                        else:
                            # Detach gradient if it has a gradient function attached
                            if p.grad.grad_fn is not None:
                                p.grad.detach_()
                            else:
                                # Mark gradient as not requiring gradient computation
                                p.grad.requires_grad_(False)
                            
                            # Zero out the gradient tensor if not in foreach mode or if it's sparse
                            if not foreach or p.grad.is_sparse:
                                p.grad.zero_()
                            else:
                                # Append gradient to per_device_and_dtype_grads structure
                                assert per_device_and_dtype_grads is not None
                                per_device_and_dtype_grads[p.grad.device][p.grad.dtype].append(p.grad)
            
            # If foreach mode is enabled, zero out gradients using torch._foreach_zero_
            if foreach:
                assert per_device_and_dtype_grads is not None
                for per_dtype_grads in per_device_and_dtype_grads.values():
                    for grads in per_dtype_grads.values():
                        torch._foreach_zero_(grads)
    # 定义一个方法 `step`，用于执行一次优化步骤以更新参数
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        # 执行单个优化步骤以更新参数
        
        # Args:
        #     closure (Callable): 一个闭包，重新评估模型并返回损失。大多数优化器可选。
        
        # .. note::
        #     除非另有说明，否则此函数不应修改参数的 `.grad` 字段。
        
        # 抛出 `NotImplementedError` 异常，提示子类实现具体优化步骤
        raise NotImplementedError

    # 在 Torch 中禁用 Dynamo 编译优化，使用该装饰器 `_disable_dynamo`
    @torch._disable_dynamo
    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        r"""Add a param group to the :class:`Optimizer`'s `param_groups`.

        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.

        Args:
            param_group (dict): Specifies what Tensors should be optimized along with group
                specific optimization options.
        """
        # 检查 param_group 是否为字典类型，如果不是则抛出类型错误异常
        if not isinstance(param_group, dict):
            raise TypeError(f"param_group must be a dict, but got {type(param_group)}")

        # 从 param_group 中获取参数列表 params
        params = param_group["params"]
        # 如果 params 是 torch.Tensor 类型，则将其转换为列表形式
        if isinstance(params, torch.Tensor):
            param_group["params"] = [params]
        # 如果 params 是集合类型 (set)，则抛出类型错误异常，建议使用列表而不是集合
        elif isinstance(params, set):
            raise TypeError(
                "optimizer parameters need to be organized in ordered collections, but "
                "the ordering of tensors in sets will change between runs. Please use a list instead."
            )
        else:
            # 否则，将 params 转换为列表形式
            param_group["params"] = list(params)

        # 遍历 param_group["params"] 中的每个参数，确保都是 torch.Tensor 类型
        for param in param_group["params"]:
            if not isinstance(param, torch.Tensor):
                raise TypeError(
                    "optimizer can only optimize Tensors, "
                    "but one of the params is " + torch.typename(param)
                )
            # 如果优化器默认配置中不允许非叶子节点或不保留梯度的张量参与优化，则抛出值错误异常
            if not self.defaults.get("differentiable", None) and not (
                param.is_leaf or param.retains_grad
            ):
                raise ValueError("can't optimize a non-leaf Tensor")

        # 确保 param_group 中所有默认参数都有值
        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError(
                    f"parameter group didn't specify a value of required optimization parameter {name}"
                )
            else:
                param_group.setdefault(name, default)

        # 检查是否有重复的参数组
        params = param_group["params"]
        if len(params) != len(set(params)):
            warnings.warn(
                "optimizer contains a parameter group with duplicate parameters; "
                "in future, this will cause an error; "
                "see github.com/pytorch/pytorch/issues/40967 for more information",
                stacklevel=3,
            )

        # 检查新添加的参数组是否与已有的参数组有重叠
        param_set: Set[torch.Tensor] = set()
        for group in self.param_groups:
            param_set.update(set(group["params"]))

        if not param_set.isdisjoint(set(param_group["params"])):
            raise ValueError("some parameters appear in more than one parameter group")

        # 将新的参数组添加到 param_groups 中
        self.param_groups.append(param_group)
```