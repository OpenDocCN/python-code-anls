# `.\pytorch\torch\amp\autocast_mode.py`

```py
# mypy: allow-untyped-defs
# 导入必要的模块和库
import collections
import functools
import warnings
from typing import Any, Optional

# 导入 PyTorch 库
import torch
from torch.types import _dtype  # 导入 _dtype 类型

# 尝试导入 NumPy 库，标记是否成功导入
try:
    import numpy as np
    HAS_NUMPY = True
except ModuleNotFoundError:
    HAS_NUMPY = False
    np = None  # type: ignore[assignment]

# 定义 __all__ 列表，用于模块导出
__all__ = [
    "autocast_decorator",
    "autocast",
    "is_autocast_available",
    "custom_fwd",
    "custom_bwd",
]

# 检查特定设备类型上是否支持自动混合精度
def is_autocast_available(device_type: str) -> bool:
    r"""
    Return a bool indicating if autocast is available on :attr:`device_type`.

    Args:
        device_type(str):  Device type to use. Possible values are: 'cuda', 'cpu', 'xpu' and so on.
            The type is the same as the `type` attribute of a :class:`torch.device`.
            Thus, you may obtain the device type of a tensor using `Tensor.device.type`.
    """
    return torch._C._is_autocast_available(device_type)

# 自动混合精度装饰器函数
def autocast_decorator(autocast_instance, func):
    @functools.wraps(func)
    def decorate_autocast(*args, **kwargs):
        # 在装饰的函数执行期间使用 autocast 上下文管理器
        with autocast_instance:
            return func(*args, **kwargs)

    # 指定装饰器不支持脚本模式
    decorate_autocast.__script_unsupported = "@autocast() decorator is not supported in script mode"  # type: ignore[attr-defined]
    return decorate_autocast

# 自动混合精度上下文管理器类
class autocast:
    r"""
    Instances of :class:`autocast` serve as context managers or decorators that
    allow regions of your script to run in mixed precision.

    In these regions, ops run in an op-specific dtype chosen by autocast
    to improve performance while maintaining accuracy.
    See the :ref:`Autocast Op Reference<autocast-op-reference>` for details.

    When entering an autocast-enabled region, Tensors may be any type.
    You should not call ``half()`` or ``bfloat16()`` on your model(s) or inputs when using autocasting.

    :class:`autocast` should wrap only the forward pass(es) of your network, including the loss
    computation(s).  Backward passes under autocast are not recommended.
    Backward ops run in the same type that autocast used for corresponding forward ops.

    Example for CUDA Devices::

        # Creates model and optimizer in default precision
        model = Net().cuda()
        optimizer = optim.SGD(model.parameters(), ...)

        for input, target in data:
            optimizer.zero_grad()

            # Enables autocasting for the forward pass (model + loss)
            with torch.autocast(device_type="cuda"):
                output = model(input)
                loss = loss_fn(output, target)

            # Exits the context manager before backward()
            loss.backward()
            optimizer.step()

    See the :ref:`Automatic Mixed Precision examples<amp-examples>` for usage (along with gradient scaling)
    in more complex scenarios (e.g., gradient penalty, multiple models/losses, custom autograd functions).
    """
    # autocast 类也可以作为装饰器使用，例如，可以用在模型的 forward 方法上
    class AutocastModel(nn.Module):
        ...
        @torch.autocast(device_type="cuda")
        def forward(self, input):
            ...
    
    # 在启用 autocast 的区域产生的浮点数张量可能是 float16 类型。
    # 在返回到禁用 autocast 的区域后，将其与不同 dtype 的浮点数张量一起使用可能导致类型不匹配错误。
    # 如果出现这种情况，请将 autocast 区域产生的张量转回 float32（或者其他所需的 dtype）。
    # 如果 autocast 区域的张量已经是 float32，则转换不会产生额外开销。
    CUDA Example::
    
        # 在默认 dtype（假设为 float32）下创建一些张量
        a_float32 = torch.rand((8, 8), device="cuda")
        b_float32 = torch.rand((8, 8), device="cuda")
        c_float32 = torch.rand((8, 8), device="cuda")
        d_float32 = torch.rand((8, 8), device="cuda")
    
        with torch.autocast(device_type="cuda"):
            # torch.mm 在 autocast 的操作列表中，应在 float16 下运行。
            # 输入是 float32，但操作在 float16 下运行并产生 float16 输出。
            # 不需要手动进行类型转换。
            e_float16 = torch.mm(a_float32, b_float32)
            # 也可以处理混合输入类型
            f_float16 = torch.mm(d_float32, e_float16)
    
        # 退出 autocast 后，调用 f_float16.float() 将其与 d_float32 使用
        g_float32 = torch.mm(d_float32, f_float16.float())
    
    CPU Training Example::
    
        # 在默认精度下创建模型和优化器
        model = Net()
        optimizer = optim.SGD(model.parameters(), ...)
    
        for epoch in epochs:
            for input, target in data:
                optimizer.zero_grad()
    
                # 使用 autocasting 进行前向传播
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    output = model(input)
                    loss = loss_fn(output, target)
    
                loss.backward()
                optimizer.step()
    
    
    CPU Inference Example::
    
        # 在默认精度下创建模型
        model = Net().eval()
    
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            for input in data:
                # 使用 autocasting 进行前向传播
                output = model(input)
    CPU Inference Example with Jit Trace::

        # 定义一个简单的神经网络模型，包含一个线性层
        class TestModel(nn.Module):
            def __init__(self, input_size, num_classes):
                super().__init__()
                self.fc1 = nn.Linear(input_size, num_classes)
            # 定义模型的前向传播方法
            def forward(self, x):
                return self.fc1(x)

        # 设置输入特征维度和类别数目
        input_size = 2
        num_classes = 2
        # 创建模型实例并设置为评估模式
        model = TestModel(input_size, num_classes).eval()

        # 由于存在此问题: https://github.com/pytorch/pytorch/issues/75956，
        # 目前建议禁用 JIT 自动类型转换 pass
        torch._C._jit_set_autocast_mode(False)

        # 使用 Torch JIT 对模型进行追踪，生成优化后的图形式模型
        with torch.cpu.amp.autocast(cache_enabled=False):
            model = torch.jit.trace(model, torch.randn(1, input_size))
        # 冻结 JIT 追踪后的模型，以便优化和部署
        model = torch.jit.freeze(model)

        # 模型运行示例
        for _ in range(3):
            model(torch.randn(1, input_size))

    Type mismatch errors *in* an autocast-enabled region are a bug; if this is what you observe,
    please file an issue.

    ``autocast(enabled=False)`` subregions can be nested in autocast-enabled regions.
    Locally disabling autocast can be useful, for example, if you want to force a subregion
    to run in a particular ``dtype``.  Disabling autocast gives you explicit control over
    the execution type.  In the subregion, inputs from the surrounding region
    should be cast to ``dtype`` before use::

        # Creates some tensors in default dtype (here assumed to be float32)
        a_float32 = torch.rand((8, 8), device="cuda")
        b_float32 = torch.rand((8, 8), device="cuda")
        c_float32 = torch.rand((8, 8), device="cuda")
        d_float32 = torch.rand((8, 8), device="cuda")

        # 进入自动混合精度计算环境，针对 CUDA 设备
        with torch.autocast(device_type="cuda"):
            # 使用自动混合精度进行矩阵乘法计算，得到 float16 类型的结果
            e_float16 = torch.mm(a_float32, b_float32)
            # 在此子区域禁用自动混合精度，强制使用 float32 类型执行计算
            with torch.autocast(device_type="cuda", enabled=False):
                # 需要将 e_float16 转换为 float32 类型，确保计算精度
                f_float32 = torch.mm(c_float32, e_float16.float())

            # 重新进入自动混合精度环境，torch.mm 操作再次以 float16 类型执行，无需手动类型转换
            g_float16 = torch.mm(d_float32, f_float32)

    The autocast state is thread-local.  If you want it enabled in a new thread, the context manager or decorator
    must be invoked in that thread.  This affects :class:`torch.nn.DataParallel` and
    :class:`torch.nn.parallel.DistributedDataParallel` when used with more than one GPU per process
    (see :ref:`Working with Multiple GPUs<amp-multigpu>`).
    # 初始化方法，接受设备类型和其他可选参数
    def __init__(
        self,
        device_type: str,
        dtype: Optional[_dtype] = None,
        enabled: bool = True,
        cache_enabled: Optional[bool] = None,
    ):
        # 如果当前运行在脚本模式下，则跳过自动混合精度设置
        if torch._jit_internal.is_scripting():
            assert self.fast_dtype is not None  # 断言快速数据类型不为空
            return self
        
        # 记录当前的自动混合精度缓存和状态
        self.prev_cache_enabled = torch.is_autocast_cache_enabled()
        self.prev = torch.is_autocast_enabled(self.device)
        self.prev_fastdtype = torch.get_autocast_dtype(self.device)
        
        # 设置当前设备的自动混合精度状态
        torch.set_autocast_enabled(self.device, self._enabled)
        torch.set_autocast_dtype(self.device, self.fast_dtype)  # 设置自动混合精度的数据类型
        torch.autocast_increment_nesting()  # 自动混合精度嵌套层级加一
        torch.set_autocast_cache_enabled(self._cache_enabled)  # 设置自动混合精度缓存是否启用

    # 退出方法，在退出上下文管理器时调用
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):  # type: ignore[override]
        # 如果当前运行在脚本模式下，则直接返回
        if torch._jit_internal.is_scripting():
            return
        
        # 当退出到没有任何自动混合精度实例的嵌套级别时，清除自动混合精度缓存
        if torch.autocast_decrement_nesting() == 0:
            torch.clear_autocast_cache()
        
        # 恢复之前的自动混合精度状态
        torch.set_autocast_enabled(self.device, self.prev)
        torch.set_autocast_dtype(self.device, self.prev_fastdtype)
        torch.set_autocast_cache_enabled(self.prev_cache_enabled)
        
        return False  # 返回 False 表示不压制异常的传播

    # 调用方法，用于装饰函数以实现自动混合精度
    def __call__(self, func):
        # 如果当前运行在脚本模式下，则直接返回函数本身
        if torch._jit_internal.is_scripting():
            return func
        
        # 返回一个使用自动混合精度装饰器修饰的函数
        return autocast_decorator(self, func)
# 这些函数不适用于公共使用。
# 它们用于在 pre_dispatch 跟踪期间将自动类型转换上下文管理器跟踪到图形中，
# 当遇到自动类型转换上下文管理器时。

# 进入自动混合精度上下文管理器。
def _enter_autocast(*vals):
    # 如果 TorchFunction 模式处于活动状态，我们将希望将其跟踪到图形中。
    if torch._C._is_torch_function_mode_enabled():
        # 处理 TorchFunction，将其转发到 torch.amp._enter_autocast 函数
        return torch.overrides.handle_torch_function(
            torch.amp._enter_autocast, [], *vals
        )
    # 否则，开启自动混合精度模式并进入该模式
    mode = torch.amp.autocast(*vals)
    mode.__enter__()
    return mode


# 退出自动混合精度上下文管理器。
def _exit_autocast(mode):
    # 如果 TorchFunction 模式处于活动状态，将其转发到 torch.amp._exit_autocast 函数
    if torch._C._is_torch_function_mode_enabled():
        return torch.overrides.handle_torch_function(torch.amp._exit_autocast, [], mode)
    # 否则，正常退出自动混合精度模式
    mode.__exit__(None, None, None)


# 将张量和张量容器进行类型转换。对于字符串和 np.ndarray，进行特殊处理以避免错误地被检测为“可迭代对象”。
def _cast(value, device_type: str, dtype: _dtype):
    if isinstance(value, torch.Tensor):
        # 判断是否符合条件进行类型转换
        is_eligible = (
            value.is_floating_point()
            and value.device.type == device_type
            and (value.dtype is not torch.float64)
        )
        return value.to(dtype) if is_eligible else value
    elif isinstance(value, (str, bytes)):
        # 字符串和字节串直接返回
        return value
    elif HAS_NUMPY and isinstance(value, np.ndarray):
        # NumPy 数组直接返回
        return value
    elif isinstance(value, collections.abc.Mapping):
        # 对字典类型进行递归类型转换
        return {
            _cast(k, device_type, dtype): _cast(v, device_type, dtype)
            for k, v in value.items()
        }
    elif isinstance(value, collections.abc.Iterable):
        # 对可迭代对象进行递归类型转换
        iterable = (_cast(v, device_type, dtype) for v in value)
        if isinstance(value, (list, tuple)):
            return type(value)(iterable)
        else:
            return iterable
    else:
        # 其他类型直接返回
        return value


# 定义一个用于定制自动微分函数的辅助装饰器，用于 forward 方法。
def custom_fwd(
    fwd=None,
    *,
    device_type: str,
    cast_inputs: Optional[_dtype] = None,
):
    """
    创建一个用于自定义自动微分函数 forward 方法的辅助装饰器。

    自动微分函数是 torch.autograd.Function 的子类。
    更多详细信息请参见示例页面 <amp-custom-examples>。

    Args:
        device_type(str): 要使用的设备类型。'cuda'、'cpu'、'xpu' 等。
            类型与 torch.device 的 `type` 属性相同。
            因此，可以使用 `Tensor.device.type` 获取张量的设备类型。
        cast_inputs (:class:`torch.dtype` or None, optional, default=None): 如果不是 ``None``,
            当 ``forward`` 在启用自动混合精度区域时，将传入的浮点张量转换为目标 dtype
            （非浮点张量不受影响），然后以禁用自动混合精度的状态执行 ``forward``。
            如果为 ``None``, ``forward`` 的内部操作将以当前自动混合精度状态执行。

    """
    pass
    """
    如果 `device_type` 不是字符串类型，抛出数值错误异常
    """
    if not isinstance(device_type, str):
        raise ValueError(
            f"Expected `device_type` of type `str`, got: `{type(device_type)}`"
        )
    """
    如果 `fwd` 为 None，返回一个部分应用了 `custom_fwd` 函数的函数对象
    """
    if fwd is None:
        return functools.partial(
            custom_fwd, device_type=device_type, cast_inputs=cast_inputs
        )

    """
    装饰 `fwd` 函数的函数定义，保留 `fwd` 函数的元数据信息
    """
    @functools.wraps(fwd)
    def decorate_fwd(*args, **kwargs):
        """
        设置第一个参数 `_dtype` 属性为根据 `device_type` 获取的自动类型转换的数据类型
        """
        args[0]._dtype = torch.get_autocast_dtype(device_type)
        """
        如果 `cast_inputs` 为 None，则检查自动类型转换是否启用，并根据情况调用 `fwd`
        """
        if cast_inputs is None:
            args[0]._fwd_used_autocast = torch.is_autocast_enabled(device_type)
            return fwd(*args, **kwargs)
        else:
            """
            检查当前是否处于自动类型转换上下文中，设置 `_fwd_used_autocast` 为 False，
            然后在禁用自动类型转换的上下文中调用 `fwd` 函数
            """
            autocast_context = torch.is_autocast_enabled(device_type)
            args[0]._fwd_used_autocast = False
            if autocast_context:
                with autocast(device_type=device_type, enabled=False):
                    """
                    在禁用自动类型转换的上下文中，使用 `_cast` 函数对参数和关键字参数进行转换，
                    然后调用 `fwd` 函数
                    """
                    return fwd(
                        *_cast(args, device_type, cast_inputs),
                        **_cast(kwargs, device_type, cast_inputs),
                    )
            else:
                """
                如果当前未启用自动类型转换，则直接调用 `fwd` 函数
                """
                return fwd(*args, **kwargs)

    """
    返回装饰后的 `fwd` 函数定义
    """
    return decorate_fwd
# Autograd ensures incoming gradients are the same type as forward outputs. Allowing a separate
# cast_inputs argument on custom_bwd is unnecessary and could cause errors if it doesn't match
# cast_inputs supplied to custom_fwd.
def custom_bwd(bwd=None, *, device_type: str):
    """Create a helper decorator for backward methods of custom autograd functions.

    Autograd functions are subclasses of :class:`torch.autograd.Function`.
    Ensures that ``backward`` executes with the same autocast state as ``forward``.
    See the :ref:`example page<amp-custom-examples>` for more detail.

    Args:
        device_type(str): Device type to use. 'cuda', 'cpu', 'xpu' and so on.
            The type is the same as the `type` attribute of a :class:`torch.device`.
            Thus, you may obtain the device type of a tensor using `Tensor.device.type`.
    """

    # Check if device_type argument is a string, otherwise raise an error
    if not isinstance(device_type, str):
        raise ValueError(
            f"Expected `device_type` of type `str`, got: `{type(device_type)}`"
        )

    # If bwd is None, return a partial function with device_type fixed
    if bwd is None:
        return functools.partial(custom_bwd, device_type=device_type)

    # Define a decorated backward function that ensures autocasting matches the forward pass
    @functools.wraps(bwd)
    def decorate_bwd(*args, **kwargs):
        # Use autocast context manager to set the correct device and dtype for the backward pass
        with autocast(
            device_type=device_type,
            enabled=args[0]._fwd_used_autocast,  # Use the autocast state from the forward pass
            dtype=args[0]._dtype,  # Use the dtype from the forward pass
        ):
            return bwd(*args, **kwargs)  # Call the original backward function with autocasting

    return decorate_bwd  # Return the decorated backward function
```