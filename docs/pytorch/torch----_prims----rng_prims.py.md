# `.\pytorch\torch\_prims\rng_prims.py`

```
# 添加类型检查注释声明，允许未经类型化的定义
# mypy: allow-untyped-defs
# 导入所需的类型
from typing import Optional, Tuple

# 导入 torch 库
import torch
# 导入 torch.utils._pytree 模块
import torch.utils._pytree as pytree
# 导入 _prims 模块
from torch import _prims
# 导入 DispatchKey 类型
from torch._C import DispatchKey
# 导入 autograd_not_implemented 函数
from torch._higher_order_ops.utils import autograd_not_implemented
# 导入 HigherOrderOperator 类
from torch._ops import HigherOrderOperator

# 导入 CUDARngStateHelper 和 make_contiguous_strides_for 函数
from torch._prims_common import CUDARngStateHelper, make_contiguous_strides_for
# 导入 FakeTensorMode 枚举
from torch._subclasses.fake_tensor import FakeTensorMode
# 导入 proxy_tensor 模块中的函数和枚举
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
# 导入 _device 和 _dtype 类型
from torch.types import _device, _dtype

# 定义 throw_on_non_cuda 函数，当设备不是 CUDA 时抛出 RuntimeError 异常
def throw_on_non_cuda(device):
    raise RuntimeError(
        f"You are trying to functionalize a {device.type} RNG operator but {device.type} does not "
        f"use Philox/counter-based RNG. Therefore, functionalizing a {device.type} RNG operator is "
        "not supported. We are discussing the possibility of a Philox-based RNG implementation for CPU."
    )

# 定义 register_rng_prim 函数，注册 RNG 运算的原语
def register_rng_prim(name, schema, impl_aten, impl_meta, doc, tags=None):
    # 创建 RNG 原语定义对象
    rngprim_def = torch.library.custom_op(
        "rngprims::" + name, impl_aten, mutates_args=(), schema=schema
    )
    # 注册实现元数据
    rngprim_def.register_fake(impl_meta)

    # 获取 RNG 原语包和默认原语对象
    prim_packet = getattr(torch._ops.ops.rngprims, name)
    prim = prim_packet.default
    # 如果存在标签，则将其设置为原语对象的标签
    if tags:
        prim._tags = tags

    # 为每个原语对象设置文档和返回类型
    for p in (prim_packet, prim):
        p.__doc__ = doc
        p.return_type = torch._prims_common.RETURN_TYPE.NEW  # type: ignore[attr-defined]
        p.schema = name + schema
        p.impl_aten = impl_aten
        p.prim_meta_impl = impl_meta

# 定义 philox_rand_offset_meta 函数，返回 Philox 随机数偏移的元数据
def philox_rand_offset_meta(
    shape: torch.Size,
):
    return _prims.TensorLike(torch.tensor(0, dtype=torch.int64))

# 定义 philox_rand_offset 函数，计算 Philox 随机数的偏移量
def philox_rand_offset(
    shape: torch.Size,
):
    # 计算张量的元素数量
    numel_scalar = 1
    for dim_size in shape:
        numel_scalar *= dim_size
    numel = torch.scalar_tensor(numel_scalar, dtype=torch.int64)

    # 定义并计算块大小、展开次数和 curand4 引擎调用次数
    block_size = 256
    unroll = 4
    curand4_engine_calls = 4
    # 获取当前设备的属性
    device_property = torch.cuda.get_device_properties(torch.cuda.current_device())
    blocks_per_sm = device_property.max_threads_per_multi_processor // block_size
    grid_size = (numel + block_size - 1) // block_size
    grid_size = min(grid_size, device_property.multi_processor_count * blocks_per_sm)
    # 计算偏移量
    offset = (
        (numel - 1) // (block_size * grid_size * unroll) + 1
    ) * curand4_engine_calls
    return offset

# 定义 register_philox_rand 函数，注册 Philox 随机数生成器
def register_philox_rand():
    # 设置名称和模式的架构
    name = "philox_rand"
    schema = "(SymInt[] size, Tensor seed, Tensor offset, int[]? stride, Device? device=None, ScalarType? dtype=None) -> (Tensor, Tensor)"  # noqa: B950
    def _philox_rand_meta(
        shape: torch.Size,
        seed: torch.Tensor,
        offset: torch.Tensor,
        stride: Optional[Tuple[int, ...]],
        device: _device,
        dtype: _dtype,
    ):
        # stride参数将在分布式场景中有用，目前未使用。
        assert stride is None  # 断言确保stride为None
        stride = make_contiguous_strides_for(shape)  # 根据shape创建连续的步长
        random_values = _prims.TensorMeta(
            shape=shape, strides=stride, dtype=dtype, device=device
        )  # 创建TensorMeta对象，描述了张量的形状、步长、数据类型和设备信息
        offset = philox_rand_offset_meta(shape)  # 计算随机偏移量的元信息
        return (random_values, offset)  # 返回元信息的元组

    def _philox_rand(
        shape: torch.Size,
        seed: torch.Tensor,
        offset: torch.Tensor,
        stride: Optional[Tuple[int, ...]],
        device: _device,
        dtype: _dtype,
    ):
        # stride参数将在分布式场景中有用，目前未使用。
        assert stride is None  # 断言确保stride为None
        if device.type == "cpu":
            devices = []  # 如果设备类型是CPU，设备列表为空
        else:
            devices = [device]  # 否则设备列表包含给定的设备

        if device.type != "cuda":
            raise throw_on_non_cuda(device)  # 如果设备不是CUDA，则抛出异常

        with torch.random.fork_rng(devices):
            CUDARngStateHelper.set_torch_state_tensor(seed, offset)
            random_values = torch.rand(shape, device=device, dtype=dtype)  # 在指定设备上生成指定形状的随机张量

        return random_values, philox_rand_offset(shape)  # 返回随机张量和对应的偏移量

    register_rng_prim(
        name=name,
        schema=schema,
        impl_aten=_philox_rand,
        impl_meta=_philox_rand_meta,
        doc="Philox based stateless rand operator",
        tags=(torch.Tag.nondeterministic_seeded,),
    )
# 创建一个函数，用于根据参数和关键字参数获取设备类型
def get_device(args, kwargs):
    # 如果关键字参数中包含 "device"
    if kwargs.get("device"):
        # 获取设备值
        device = kwargs.get("device")
        # 如果设备是字符串类型，则转换成 torch 设备对象
        if isinstance(device, str):
            device = torch.device(device)
        # 返回设备类型
        return device.type

    # 收集所有输入参数中是 torch.Tensor 类型的设备类型
    devices = {arg.device.type for arg in args if isinstance(arg, torch.Tensor)}
    # 如果存在 CUDA 设备，则返回 "cuda"
    if any(dev == "cuda" for dev in devices):
        return "cuda"
    # 如果存在 CPU 设备，则返回 "cpu"
    elif any(dev == "cpu" for dev in devices):
        return "cpu"
    # 如果都不存在，则返回 None
    return None


# 创建一个 HigherOrderOperator 对象并注册 "run_and_save_rng_state" 操作
def register_run_and_save_rng_state_op():
    run_and_save_rng_state = HigherOrderOperator("run_and_save_rng_state")

    # 为 Autograd 分发键注册未实现的操作
    run_and_save_rng_state.py_impl(DispatchKey.Autograd)(
        autograd_not_implemented(run_and_save_rng_state, deferred_error=True)
    )

    # 定义 CUDA 分发键的实现函数
    @run_and_save_rng_state.py_impl(DispatchKey.CUDA)
    def impl_cuda(op, *args, **kwargs):
        # 获取当前 CUDA 的随机数生成器状态并执行操作
        return torch.cuda.get_rng_state(), op(*args, **kwargs)

    # 定义 CPU 分发键的实现函数
    @run_and_save_rng_state.py_impl(DispatchKey.CPU)
    def impl_cpu(op, *args, **kwargs):
        # 获取当前 CPU 的随机数生成器状态并执行操作
        return torch.get_rng_state(), op(*args, **kwargs)

    # 定义 BackendSelect 分发键的实现函数
    @run_and_save_rng_state.py_impl(DispatchKey.BackendSelect)
    def impl_backend_select(op, *args, **kwargs):
        # 设备映射表，根据设备类型选择相应的实现函数
        impl_map = {"cuda": impl_cuda, "cpu": impl_cpu}
        # 获取当前设备类型
        device = get_device(args, kwargs)
        # 断言当前设备类型在映射表中，否则抛出异常
        assert device in impl_map, f"Backend not supported for {device}"
        # 根据设备类型选择实现函数并执行操作
        impl = impl_map[device]
        return impl(op, *args, **kwargs)

    # 定义 FakeTensorMode 分发键的实现函数
    @run_and_save_rng_state.py_impl(FakeTensorMode)
    def impl_fake_tensor_mode(mode, op, *args, **kwargs):
        # 检查设备类型以调用正确的实现函数
        with mode:
            return impl_backend_select(op, *args, **kwargs)

    # 定义 ProxyTorchDispatchMode 分发键的实现函数
    @run_and_save_rng_state.py_impl(ProxyTorchDispatchMode)
    def impl_proxy_dispatch_mode(mode, op, *args, **kwargs):
        # 如果启用追踪，则执行后续操作
        if mode.enable_tracing:
            # 调用 BackendSelect 实现函数获取结果
            out = impl_backend_select(op, *args, **kwargs)
            # 对参数和关键字参数进行代理解包
            proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, (op, *args))
            proxy_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, kwargs)
            # 使用追踪器创建代理对象并返回结果
            out_proxy = mode.tracer.create_proxy(
                "call_function", run_and_save_rng_state, proxy_args, proxy_kwargs
            )
            return track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)
        else:
            # 否则直接执行 run_and_save_rng_state 操作并返回结果
            return run_and_save_rng_state(op, *args, **kwargs)

    # 返回注册好的 run_and_save_rng_state 操作对象
    return run_and_save_rng_state


# 创建一个 HigherOrderOperator 对象并注册 "run_with_rng_state" 操作
def register_run_with_rng_state_op():
    run_with_rng_state = HigherOrderOperator("run_with_rng_state")

    # 为 Autograd 分发键注册未实现的操作
    run_with_rng_state.py_impl(DispatchKey.Autograd)(
        autograd_not_implemented(run_with_rng_state, deferred_error=True)
    )

    # 定义 CUDA 分发键的实现函数
    @run_with_rng_state.py_impl(DispatchKey.CUDA)
    def impl_cuda(rng_state, op, *args, **kwargs):
        # 获取当前 CUDA 的随机数生成器状态并设置传入的状态
        current_state = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(rng_state.cpu())
        # 执行操作并恢复之前的随机数生成器状态
        out = op(*args, **kwargs)
        torch.cuda.set_rng_state(current_state)
        return out

    # 定义 CPU 分发键的实现函数，略
    # 定义一个函数impl_cpu，用于在指定的随机数生成器状态下执行操作
    def impl_cpu(rng_state, op, *args, **kwargs):
        # 获取当前的随机数生成器状态并保存
        current_state = torch.get_rng_state()
        # 设置随机数生成器状态为函数参数中传入的状态
        torch.set_rng_state(rng_state)
        # 执行传入的操作op，并接收返回值
        out = op(*args, **kwargs)
        # 恢复之前保存的随机数生成器状态
        torch.set_rng_state(current_state)
        # 返回操作的输出结果
        return out
    
    # 使用特定的调度模式运行函数，这里使用的是ProxyTorchDispatchMode
    @run_with_rng_state.py_impl(ProxyTorchDispatchMode)
    def impl_proxy_dispatch_mode(mode, rng_state, op, *args, **kwargs):
        # 如果启用跟踪模式
        if mode.enable_tracing:
            # 禁用代理模式的跟踪
            with disable_proxy_modes_tracing():
                # 在指定的随机数生成器状态下运行操作op，并接收返回值
                out = run_with_rng_state(rng_state, op, *args, **kwargs)
            # 对参数和关键字参数进行代理解包
            proxy_args = pytree.tree_map(
                mode.tracer.unwrap_proxy, (rng_state, op, *args)
            )
            proxy_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, kwargs)
            # 创建代理对象，用于跟踪函数调用
            out_proxy = mode.tracer.create_proxy(
                "call_function", run_with_rng_state, proxy_args, proxy_kwargs
            )
            # 跟踪张量树的输出，返回跟踪结果
            return track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)
        else:
            # 如果未启用跟踪模式，直接运行函数并返回结果
            return run_with_rng_state(rng_state, op, *args, **kwargs)
    
    # 使用特定的调度模式运行函数，这里使用的是DispatchKey.BackendSelect
    @run_with_rng_state.py_impl(DispatchKey.BackendSelect)
    def impl_backend_select(rng_state, op, *args, **kwargs):
        # 定义实现映射表，根据设备类型选择相应的实现函数
        impl_map = {"cuda": impl_cuda, "cpu": impl_cpu}
        # 获取操作的设备类型
        device = get_device(args, kwargs)
        # 断言设备类型在映射表中，否则抛出异常
        assert device in impl_map, f"Backend not supported for {device}"
        # 根据设备类型选择对应的实现函数
        impl = impl_map[device]
        # 使用选定的实现函数执行操作并返回结果
        return impl(rng_state, op, *args, **kwargs)
    
    # 使用特定的调度模式运行函数，这里使用的是FakeTensorMode
    @run_with_rng_state.py_impl(FakeTensorMode)
    def impl_fake_tensor_mode(mode, rng_state, op, *args, **kwargs):
        # 跳过设置随机数生成器状态，因为它对假张量不起作用
        # 在假张量模式下这并不重要
        with mode:
            # 直接执行操作op，并返回结果
            return op(*args, **kwargs)
    
    # 函数功能化实现，用于在上下文中处理张量
    @run_with_rng_state.py_functionalize_impl
    def impl_functional(ctx, rng_state, op, *args, **kwargs):
        # 解包传入的张量状态、参数和关键字参数
        unwrapped_rng_state = ctx.unwrap_tensors(rng_state)
        unwrapped_args = ctx.unwrap_tensors(args)
        unwrapped_kwargs = ctx.unwrap_tensors(kwargs)
    
        # 将处理转发到下一个上下文
        with ctx.redispatch_to_next():
            # 在解包后的张量状态下运行操作op，并接收返回值
            out = run_with_rng_state(
                unwrapped_rng_state, op, *unwrapped_args, **unwrapped_kwargs
            )
            # 包装处理后的张量输出，并返回结果
            return ctx.wrap_tensors(out)
    
    # 返回run_with_rng_state函数对象，供外部调用
    return run_with_rng_state
# 注册并返回一个操作，用于运行并保存随机数生成器的状态
run_and_save_rng_state = register_run_and_save_rng_state_op()

# 注册并返回一个操作，用于运行时使用保存的随机数生成器状态
run_with_rng_state = register_run_with_rng_state_op()

# 注册 Philox 随机数生成器的相关操作
def register_rng_prims():
    register_philox_rand()
```