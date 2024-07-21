# `.\pytorch\torch\jit\_trace.py`

```py
"""
Tracing.

This module contains functionality to support the JIT's tracing frontend, notably:
    * torch.jit.trace
    * torch.jit.trace_module

This is not intended to be imported directly; please use the exposed
functionalities in `torch.jit`.
"""

# 导入上下文管理器模块
import contextlib

# 导入复制功能模块
import copy

# 导入函数装饰器模块
import functools

# 导入检查模块
import inspect

# 导入操作系统功能模块
import os

# 导入正则表达式模块
import re

# 导入警告模块
import warnings

# 导入枚举模块
from enum import Enum

# 导入类型注解模块
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar

# 导入参数规范模块
from typing_extensions import ParamSpec

# 导入PyTorch核心模块
import torch

# 导入PyTorch内部JIT相关模块
from torch._jit_internal import (
    _qualified_name,
    get_callable_argument_names,
    is_scripting,
)

# 导入PyTorch自动求导函数模块
from torch.autograd import function

# 导入PyTorch脚本模块和脚本模块类型
from torch.jit._script import _CachedForward, script, ScriptModule

# 导入PyTorch JIT状态模块
from torch.jit._state import _enabled, _python_cu

# 导入PyTorch神经网络模块
from torch.nn import Module

# 导入PyTorch测试比较模块中的默认容差值
from torch.testing._comparison import default_tolerances

# 获取PyTorch C++扩展模块中的函数：_jit_flatten 和 _jit_unflatten
_flatten = torch._C._jit_flatten
_unflatten = torch._C._jit_unflatten

# 定义类型变量 R，表示返回类型（总是协变）
R = TypeVar("R", covariant=True)

# 定义参数规范变量 P
P = ParamSpec("P")


def _create_interpreter_name_lookup_fn(frames_up=1):
    # 定义获取解释器变量名的函数
    def _get_interpreter_name_for_var(var):
        # 获取当前调用帧
        frame = inspect.currentframe()
        if not frame:
            raise RuntimeError("failed to inspect frame")

        i = 0
        while i < frames_up + 1:
            # 向上遍历帧
            frame = frame.f_back
            if not frame:
                raise RuntimeError("failed to get frame")
            i += 1

        # 获取本地和全局变量
        f_locals = frame.f_locals
        f_globals = frame.f_globals

        # 遍历本地变量，找到与给定变量相等的 torch.Tensor 类型的变量
        for k, v in f_locals.items():
            if isinstance(v, torch.Tensor) and var is v:
                return k if k != "self" else ""
        return ""

    return _get_interpreter_name_for_var


def _unique_state_dict(module, keep_vars=False):
    # 由于 Parameter.detach() 总是创建一个新的 torch.Tensor 实例，
    # 因此 id(v) 不能用于它。因此我们始终使用参数和缓冲的值，并使用 Parameters 和 Buffers 进行去重
    state_dict = module.state_dict(keep_vars=True)
    filtered_dict = type(state_dict)()
    seen_ids: Set[int] = set()
    for k, v in state_dict.items():
        if id(v) in seen_ids:
            continue
        seen_ids.add(id(v))
        if keep_vars:
            filtered_dict[k] = v
        else:
            filtered_dict[k] = v.detach()
    return filtered_dict


class ONNXTracedModule(torch.nn.Module):
    def __init__(
        self,
        inner,
        strict=True,
        force_outplace=False,
        return_inputs=False,
        return_inputs_states=False,
        ):
            super().__init__()
            # inner may be a Module, or it may be an arbitrary callable
            # If it's a Module, we get its parameters automatically, which lets
            # us avoid a special casing functions versus modules.
            self.inner = inner
            self.strict = strict
            self._force_outplace = force_outplace
            self._return_inputs = return_inputs
            self._return_inputs_states = return_inputs_states

        def forward(self, *args: torch.Tensor):
            # Flatten input arguments and their descriptions
            in_vars, in_desc = _flatten(args)
            # NOTE: use full state, because we need it for BatchNorm export
            # This differs from the compiler path, which doesn't support it at the moment.
            # Retrieve the unique state dictionary including tensors with their gradients
            module_state = list(_unique_state_dict(self, keep_vars=True).values())

            ret_inputs = []
            inputs_states = []
            outs = []

            def wrapper(*args):
                # Initialize a list for input arguments
                in_args: List[torch.Tensor] = []
                # Iterate through flattened input variables
                for i in range(len(in_vars)):
                    # Ensure each argument is a Tensor; raise error otherwise
                    if not isinstance(args[i], torch.Tensor):
                        raise RuntimeError("Expected Tensor argument")
                    # Append validated argument to in_args list
                    in_args.append(args[i])

                # Reconstruct the trace inputs from the flattened arguments
                trace_inputs = _unflatten(in_args, in_desc)

                # Conditionally store original inputs if specified
                if self._return_inputs:
                    ret_inputs.append(
                        tuple(x.clone(memory_format=torch.preserve_format) for x in args)
                    )
                # Conditionally store inputs states if specified
                if self._return_inputs_states:
                    inputs_states.append(_unflatten(in_args, in_desc))
                # Apply the inner module or callable to trace inputs
                outs.append(self.inner(*trace_inputs))
                # Update inputs states if specified
                if self._return_inputs_states:
                    inputs_states[0] = (inputs_states[0], trace_inputs)
                # Flatten output variables
                out_vars, _ = _flatten(outs)
                # Return single output variable or tuple of output variables
                if len(out_vars) == 1:
                    return out_vars[0]
                else:
                    return tuple(out_vars)

            # Create a computation graph by tracing the wrapper function
            graph, out = torch._C._create_graph_by_tracing(
                wrapper,
                in_vars + module_state,
                _create_interpreter_name_lookup_fn(),
                self.strict,
                self._force_outplace,
            )

            # Return appropriate outputs based on conditions
            if self._return_inputs:
                return graph, outs[0], ret_inputs[0]
            if self._return_inputs_states:
                return graph, outs[0], inputs_states[0]
            else:
                return graph, outs[0]
# 复制输入参数列表中的所有对象，对于 torch.Tensor 对象进行特殊处理
def _clone_inputs(args):
    # 复制单个输入对象
    def clone_input(a):
        # 如果输入为 None，则直接返回 None
        if a is None:
            return None
        # 如果输入为 torch.Tensor 对象
        elif isinstance(a, torch.Tensor):
            # TODO: 找出一行代码实现 .clone() 和设置 requires_grad 的功能
            # 分离张量 a 的计算图，并克隆其数据，保留存储格式
            v = (
                a.detach()
                .clone(memory_format=None if a.is_mkldnn else torch.preserve_format)
                .requires_grad_(a.requires_grad)
            )
            # 如果张量 a 有梯度，则递归克隆其梯度
            if a.grad is not None:
                v.grad = clone_input(v.grad)
            return v
        else:
            # 对于非 torch.Tensor 对象，按照保留存储格式克隆
            return a.clone(memory_format=torch.preserve_format)

    # 使用 function._nested_map 函数将 clone_input 应用于所有 torch.Tensor 对象，并返回结果
    return function._nested_map(
        lambda x: isinstance(x, torch.Tensor), clone_input, condition_msg="tensors"
    )(args)


# 仅用于开发者调试，不会进行广告宣传
_JIT_TIME = os.environ.get("PYTORCH_JIT_TIME", False)  # 仅在 CUDA 下计时
_JIT_DISABLE = os.environ.get("PYTORCH_JIT_DISABLE", False)
_JIT_STATS = os.environ.get("PYTORCH_JIT_STATS", False)


# 创建一个计时器上下文管理器，用于 CUDA 时间测量
@contextlib.contextmanager
def _time(trace_name, name, time=True):
    # 如果未启用 JIT 时间测量，或者 CUDA 不可用，则直接返回
    if (not _JIT_TIME and not time) or not torch.cuda.is_available():
        yield
        return
    # 获取当前 CUDA 流
    stream = torch.cuda.current_stream()
    # 创建开始和结束事件，用于记录时间
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # 记录开始时间
    stream.record_event(start)
    try:
        yield
    finally:
        # 记录结束时间，并同步 CUDA 流
        stream.record_event(end)
        end.synchronize()
        # 打印跟踪名称、名称和经过的时间
        print(f"{trace_name} {name} time: {start.elapsed_time(end)} ms")


# 验证 JIT 编译模型与其未编译版本的行为是否一致，包括反向传播
def verify(model, args, loss_fn=torch.sum, devices=None):
    """
    验证 JIT 编译模型与其未编译版本的行为是否一致，包括反向传播。

    如果您的模型返回多个输出，则必须指定一个 `loss_fn` 函数来计算损失，以便进行反向传播。

    该函数具有副作用（例如，执行模型、保存和加载参数），因此不要期望输出与输入完全相同。
    """
    """
    Args:
        model (compiled torch.nn.Module or function): 要验证的模块或函数。模块/函数定义必须已经使用`@torch.jit.compile`进行装饰。
        args (tuple or Tensor): 传递给编译函数/模块进行验证的位置参数。如果不是元组，则假定为要传递给模型的单个位置参数。
        loss_fn (function, optional): 应用于模型输出的损失函数，在调用反向传播之前。默认情况下，我们假设模型返回单个结果，在调用反向传播之前使用`torch.sum`；如果不适用，请传递您自己的损失函数。
            注意，如果模型返回结果的元组，则这些结果将作为单独的位置参数传递给`loss_fn`。
        devices (iterable of device IDs, optional): 编译模块将在其上运行的GPU设备。这决定了我们在运行编译和未编译版本的模型时必须保存的随机数生成器状态。
    """
    # TODO: 原则上，我们在跟踪中跟踪设备信息，因此应该可以检查执行是否实际遵守了用户提供的 'devices'。

    # TODO: 考虑添加一个工具函数到torch.jit，用于测试这种情况

    # 检查模型是否已编译为torch._C.CompiledFunction类型（注意：type: ignore[attr-defined]用于类型检查时忽略属性定义）
    if not isinstance(model, torch._C.CompiledFunction):  # type: ignore[attr-defined]
        raise TypeError(
            "Cannot verify an uncompiled module.  Add @torch.jit.compile to compile it"
        )
    
    # 判断model是否为Module类型
    is_module = isinstance(model, Module)

    # 如果args不是元组，则将其转换为元组形式
    if not isinstance(args, tuple):
        args = (args,)

    # 复制输入参数args的值
    saved_args = _clone_inputs(args)

    # 如果model是Module类型，则深度复制其状态字典，并保存在saved_state中
    if is_module:
        saved_state = copy.deepcopy(model.state_dict())
    # 定义函数 `run_fwd_bwd`，接受参数 `args`、`force_trace`（是否强制重置追踪）、`assert_compiled`（是否断言已编译）
    def run_fwd_bwd(args, force_trace=False, assert_compiled=False):
        # 如果 `is_module` 为真，则将模型的参数列表赋给 `params`
        params = list(model.parameters()) if is_module else []
        # 将输入参数 `args` 和 `params` 展平并保存到 `in_vars`
        in_vars, _ = _flatten((args, params))
        
        # 如果 `force_trace` 为真，则清除模型的缓存
        compiled_fn = model
        if force_trace:
            compiled_fn.clear_cache()
        
        # 如果 `assert_compiled` 为真，则获取当前已编译函数的命中数 `hits`
        if assert_compiled:
            hits = compiled_fn.hits
        
        # 调用模型 `model`，并将结果保存到 `out`
        out = model(*args)
        
        # 如果 `assert_compiled` 为真且 `compiled_fn.hits` 与之前保存的 `hits` 相等，则抛出运行时错误
        if assert_compiled and compiled_fn.hits == hits:  # type: ignore[possibly-undefined]
            raise RuntimeError("failed to use the compiled function")
        
        # 如果 `out` 不是元组，则转换为单元素元组
        if not isinstance(out, tuple):
            out = (out,)
        
        # 如果损失函数 `loss_fn` 是 `torch.sum`，且 `out` 的长度不为1，则抛出值错误
        if loss_fn == torch.sum and len(out) != 1:
            raise ValueError(
                f"Model returns {len(out)} outputs, but default loss function "
                "(torch.sum) can only handle a single output"
            )
        
        # 将 `out` 展平并保存到 `out_vars`
        out_vars, _ = _flatten(out)
        
        # 对 `out_vars` 中的每个张量进行深拷贝并保存到 `saved_outs`
        saved_outs = [
            v.detach().clone(memory_format=torch.preserve_format) for v in out_vars
        ]
        
        # 计算损失函数 `loss` 对输入参数 `in_vars` 的梯度，并保存到 `grads`
        loss = loss_fn(*out)
        grads = torch.autograd.grad([loss], in_vars)
        
        # 对 `grads` 中的每个张量进行深拷贝并保存到 `saved_grads`
        saved_grads = [
            v.detach().clone(memory_format=torch.preserve_format) for v in grads
        ]
        
        # 返回 `saved_outs` 和 `saved_grads` 组成的元组
        return (saved_outs, saved_grads)

    # 使用 `torch.random.fork_rng` 函数分叉随机数生成器，并设置调用者为 "torch.jit.verify"
    with torch.random.fork_rng(devices, _caller="torch.jit.verify"):
        # 运行 `run_fwd_bwd` 函数，强制重置追踪，并将结果保存到 `uncompiled_outs` 和 `uncompiled_grads`
        uncompiled_outs, uncompiled_grads = run_fwd_bwd(args, force_trace=True)
        
        # 断言模型 `model` 对 `args` 已经有追踪记录
        assert model.has_trace_for(*args)

    # 如果 `is_module` 为真，则加载已保存的模型状态 `saved_state`
    if is_module:
        model.load_state_dict(saved_state)  # type: ignore[possibly-undefined]
    
    # 运行 `run_fwd_bwd` 函数，断言已编译，并将结果保存到 `compiled_outs` 和 `compiled_grads`
    compiled_outs, compiled_grads = run_fwd_bwd(args, assert_compiled=True)

    # 使用 `_verify_equal` 函数检查 `uncompiled_outs` 和 `compiled_outs` 是否相等
    _verify_equal(uncompiled_outs, compiled_outs)
    
    # 使用 `_verify_equal` 函数检查 `uncompiled_grads` 和 `compiled_grads` 是否相等
    _verify_equal(uncompiled_grads, compiled_grads)
def _verify_equal(xs, ys):
    # 对两个可迭代对象 xs 和 ys 进行逐元素比较
    for x, y in zip(xs, ys):
        # 检查 x 和 y 的差值的绝对值的最大值是否大于 1e-6
        if x.sub(y).abs().max() > 1e-6:
            # 如果超出允许的误差范围，抛出 RuntimeError 异常
            raise RuntimeError("JIT and real computation mismatch")


def indent(s):
    # 将字符串 s 中的每一行开头添加制表符，返回修改后的字符串
    return "\n".join(["\t" + line for line in s.splitlines()])


class TracingCheckError(Exception):
    def __init__(self, graph_diff_error, tensor_compare_error, extra_msg=None):
        # 初始化异常消息
        self.message = "Tracing failed sanity checks!\n"
        # 如果有额外的消息，将其添加到异常消息中
        if extra_msg is not None:
            self.message += extra_msg + "\n"
        # 如果存在图形差异错误，将其添加到异常消息中
        if graph_diff_error is not None:
            self.message += "ERROR: Graphs differed across invocations!\n"
            self.message += indent(graph_diff_error) + "\n"
        # 如果存在张量比较错误，将其添加到异常消息中
        if tensor_compare_error is not None:
            self.message += (
                "ERROR: Tensor-valued Constant nodes differed in value "
                "across invocations. This often indicates that the tracer has"
                " encountered untraceable code.\n"
            )
            self.message += indent(tensor_compare_error) + "\n"
        # 调用父类 Exception 的初始化方法，传入完整的异常消息
        super().__init__(self.message)


# Check the traced module against a set of user-provided validation inputs
@torch.no_grad()
def _check_trace(
    check_inputs,
    func,
    traced_func,
    check_tolerance,
    strict,
    force_outplace,
    is_trace_module,
    _module_class,
    example_inputs_is_kwarg=False,
):
    # 注意：追踪操作独立于优化过程，优化过程会消耗追踪信息
    pass  # 这里只有注释，函数体暂时为空


class TracerWarning(Warning):
    @staticmethod
    def ignore_lib_warnings():
        # 忽略来自所有子模块（JIT 除外）的 TracerWarning 警告
        warnings.filterwarnings(
            "ignore", category=TracerWarning, module="torch.(?!jit)"
        )
        # 忽略 "torch::jit::fuser::cuda" 的警告信息
        warnings.filterwarnings("ignore", "torch::jit::fuser::cuda")


# We ignore the tracer warnings coming form inside the library, because all our shape
# checks in nn will trigger them.
# 忽略库内部来自追踪器的警告，因为 nn 中的所有形状检查都会触发它们
TracerWarning.ignore_lib_warnings()
# 调用 torch._C._tracer_warn_use_python() 函数
torch._C._tracer_warn_use_python()


def make_tuple(example_inputs):
    # 如果 example_inputs 是 torch.Tensor 或 dict 类型，则返回包含它的元组
    if isinstance(example_inputs, (torch.Tensor, dict)):
        return (example_inputs,)
    # 如果 example_inputs 不是元组类型，则将其转换为元组并返回
    # 主要是为了处理奇怪的可迭代对象错误，而不是在 pybind11 代码中发生错误
    if not isinstance(example_inputs, tuple):
        return tuple(example_inputs)
    # 如果 example_inputs 已经是元组类型，则直接返回
    return example_inputs


def make_module(mod, _module_class, _compilation_unit):
    # 如果 mod 是 ScriptModule 类型，则直接返回该对象
    if isinstance(mod, ScriptModule):
        return mod
    # 如果 mod 具有导出方法，则使用导出方法创建 ScriptModule 对象
    elif torch._jit_internal.module_has_exports(mod):
        infer_methods_stubs_fn = torch.jit._recursive.make_stubs_from_exported_methods
        return torch.jit._recursive.create_script_module(
            mod, infer_methods_stubs_fn, share_types=False, is_tracing=True
        )
    # 如果 mod 既非 ScriptModule 也没有导出方法，则根据情况创建 _module_class 对象
    else:
        if _module_class is None:
            _module_class = TopLevelTracedModule
        return _module_class(mod, _compilation_unit=_compilation_unit)


def wrap_check_inputs(check_inputs):
    # 如果 check_inputs 为 None，则直接返回 None
    if check_inputs is None:
        return None
    # 创建一个列表，其中每个元素是一个字典，字典中只有一个键值对，键是"forward"，值是check_inputs中的每个元素c
    return [{"forward": c} for c in check_inputs]
# 对给定导出和追踪结果进行分析，检查它们是否一致
def analyze_ts_result_with_export_result(export, trace):
    # 导入 torch.utils._pytree 模块中的 pytree
    import torch.utils._pytree as pytree

    # 将导出结果展平为列表
    flat_export = pytree.tree_leaves(export)
    # 将追踪结果展平为列表
    flat_trace = pytree.tree_leaves(trace)

    # 遍历展平后的导出和追踪结果
    for orig, loaded in zip(flat_export, flat_trace):
        # 检查布局是否相同
        if orig.layout != loaded.layout:
            return False
        # 对于 torch._mkldnn 布局，不支持 torch.allclose
        if orig.layout == torch._mkldnn:  # type: ignore[attr-defined]
            return True
        # 检查类型是否相同
        if type(orig) != type(loaded):
            return False

        # 对于 FakeTensor 类型的数据，跳过检查
        if isinstance(orig, torch._subclasses.FakeTensor):
            return True
        elif isinstance(orig, torch.Tensor):
            # 对于 Torch Tensor，检查数据类型和数值是否相近
            if orig.dtype != loaded.dtype:
                return False
            if not torch.allclose(orig, loaded):
                return False
        else:
            # 对于其他类型的数据，检查其值是否相同
            if orig != loaded:
                return False
    # 如果所有检查通过，则返回 True
    return True


# 执行函数追踪的实现
def _trace_impl(
    func,
    example_inputs=None,
    optimize=None,
    check_trace=True,
    check_inputs=None,
    check_tolerance=1e-5,
    strict=True,
    _force_outplace=False,
    _module_class=None,
    _compilation_unit=_python_cu,
    example_kwarg_inputs=None,
    _store_inputs=True,
):
    # 如果 func 是 torch.jit.ScriptModule，则直接返回，不进行追踪
    if isinstance(func, torch.jit.ScriptModule):
        warnings.warn(
            "The input to trace is already a ScriptModule, tracing it is a no-op. Returning the object as is."
        )
        return func

    # 如果 func 是 torch.nn.Module 类型
    if isinstance(func, torch.nn.Module):
        # 如果 example_inputs 为 None，则要求 example_kwarg_inputs 必须是字典类型
        if example_inputs is None:
            if isinstance(example_kwarg_inputs, dict):
                example_inputs = example_kwarg_inputs
            else:
                raise RuntimeError("example_kwarg_inputs should be a dict")
        # 对模块进行追踪
        return trace_module(
            func,
            {"forward": example_inputs},
            None,
            check_trace,
            wrap_check_inputs(check_inputs),
            check_tolerance,
            strict,
            _force_outplace,
            _module_class,
            example_inputs_is_kwarg=isinstance(example_kwarg_inputs, dict),
            _store_inputs=_store_inputs,
        )

    # 如果 func 是 torch.nn.Module 的实例，并且其 __name__ 为 "forward"
    if (
        hasattr(func, "__self__")
        and isinstance(func.__self__, torch.nn.Module)
        and func.__name__ == "forward"
    ):
        if example_inputs is None:
            if isinstance(example_kwarg_inputs, dict):
                example_inputs = example_kwarg_inputs
            else:
                raise RuntimeError("example_kwarg_inputs should be a dict")
        return trace_module(
            func.__self__,
            {"forward": example_inputs},
            None,
            check_trace,
            wrap_check_inputs(check_inputs),
            check_tolerance,
            strict,
            _force_outplace,
            _module_class,
            example_inputs_is_kwarg=isinstance(example_kwarg_inputs, dict),
            _store_inputs=_store_inputs,
        )



    # Special case for common case of passing a single Tensor
    if (
        isinstance(example_inputs, (torch.Tensor, dict))
        and example_kwarg_inputs is None
    ):
        example_inputs = (example_inputs,)
    # done primarily so that weird iterables fail here and not pybind11 code
    elif example_kwarg_inputs is None and not isinstance(example_inputs, tuple):
        example_inputs = tuple(example_inputs)



    var_lookup_fn = _create_interpreter_name_lookup_fn(0)



    if hasattr(func, "__self__") and isinstance(func.__self__, torch.nn.Module):
        raise AttributeError(
            "trace doesn't support compiling individual module's functions.\n"
            "Please use trace_module"
        )



    name = _qualified_name(func)
    if isinstance(example_kwarg_inputs, dict):
        example_inputs = example_kwarg_inputs
        traced = torch._C._create_function_from_trace_with_dict(
            name,
            func,
            example_kwarg_inputs,
            var_lookup_fn,
            strict,
            _force_outplace,
            get_callable_argument_names(func),
        )
    else:
        traced = torch._C._create_function_from_trace(
            name,
            func,
            example_inputs,
            var_lookup_fn,
            strict,
            _force_outplace,
            get_callable_argument_names(func),
        )



    # Check the trace against new traces created from user-specified inputs
    if check_trace:
        if check_inputs is not None:
            _check_trace(
                check_inputs,
                func,
                traced,
                check_tolerance,
                strict,
                _force_outplace,
                False,
                _module_class,
                example_inputs_is_kwarg=isinstance(example_kwarg_inputs, dict),
            )
        else:
            _check_trace(
                [example_inputs],
                func,
                traced,
                check_tolerance,
                strict,
                _force_outplace,
                False,
                _module_class,
                example_inputs_is_kwarg=isinstance(example_kwarg_inputs, dict),
            )



    # Allow torch.compile() to inline
    traced._torchdynamo_inline = func  # type: ignore[attr-defined]
    return traced


    # 返回变量 traced 的值作为函数的返回值
class _ExportType(str, Enum):
    # 定义导出类型枚举，包括直接导出、追踪和导出、源到源导出
    DIRECT_EXPORT = "DIRECT_EXPORT"
    TRACE_AND_EXPORT = "TRACE_AND_EXPORT"
    SOURCE_TO_SOURCE = "SOURCE_TO_SOURCE"

    def __str__(self) -> str:
        return self.value


class _ExportOutcome(str, Enum):
    # 定义导出结果枚举，包括成功、导出失败、运行失败、精度错误
    SUCCESS = "SUCCESS"
    FAILED_TO_EXPORT = "FAILED_TO_EXPORT"
    FAILED_TO_RUN = "FAILED_TO_RUN"
    ACCURACY_ERROR = "ACCURACY_ERROR"

    def __str__(self) -> str:
        return self.value


def trace(
    func,
    example_inputs=None,
    optimize=None,
    check_trace=True,
    check_inputs=None,
    check_tolerance=1e-5,
    strict=True,
    _force_outplace=False,
    _module_class=None,
    _compilation_unit=_python_cu,
    example_kwarg_inputs=None,
    _store_inputs=True,
):
    r"""
    Trace a function and return an executable  or :class:`ScriptFunction` that will be optimized using just-in-time compilation.

    Tracing is ideal for code that operates only on
    ``Tensor``\\s and lists, dictionaries, and
    tuples of ``Tensor``\\s.

    Using `torch.jit.trace` and `torch.jit.trace_module`, you can turn an
    existing module or Python function into a TorchScript
    :class:`ScriptFunction` or :class:`ScriptModule`. You must provide example
    inputs, and we run the function, recording the operations performed on all
    the tensors.

    * The resulting recording of a standalone function produces `ScriptFunction`.
    * The resulting recording of `nn.Module.forward` or `nn.Module` produces
      `ScriptModule`.

    This module also contains any parameters that the original
    module had as well.
    # 警告：
    # 跟踪仅正确记录那些不依赖数据（例如没有张量数据上的条件语句）且没有未跟踪的外部依赖（例如执行输入/输出或访问全局变量）的函数和模块。跟踪仅记录在给定张量上运行给定函数时执行的操作。因此，返回的 `ScriptModule` 在任何输入上始终运行相同的跟踪图。这在模块预期根据输入和/或模块状态运行不同操作集合时具有重要意义。例如，
    #
    # * 跟踪不会记录任何类似 if-语句或循环的控制流。当此控制流在整个模块中保持不变时，这通常是可接受的，并且通常会将控制流决策内联化。但有时控制流实际上是模型本身的一部分。例如，递归网络是对输入序列的（可能是动态的）长度进行循环。
    # * 在返回的 :class:`ScriptModule` 中，那些在 ``training`` 和 ``eval`` 模式下行为不同的操作将始终表现为在跟踪期间所处模式的行为，无论 `ScriptModule` 当前处于哪种模式。
    #
    # 在这些情况下，跟踪是不合适的，应该选择 :func:`scripting <torch.jit.script>`。如果对这些模型进行跟踪，则可能会在模型的后续调用中悄悄地得到不正确的结果。当执行可能导致生成不正确跟踪的操作时，跟踪器将尝试发出警告。
    #
    # 参数：
    # func (callable 或 torch.nn.Module)：一个 Python 函数或 `torch.nn.Module`，将使用 `example_inputs` 运行。`func` 的参数和返回值必须是张量或（可能是嵌套的）包含张量的元组。当传递模块给 `torch.jit.trace` 时，只运行和跟踪 ``forward`` 方法（详情参见 :func:`torch.jit.trace <torch.jit.trace_module>`）。
    # 定义一个函数，用于将函数进行跟踪（tracing）并生成一个跟踪结果（trace）。
    # 被跟踪的函数可以接受不同类型和形状的输入（如果跟踪操作支持这些类型和形状）。
    # 可以通过指定 example_inputs 或者 example_kwarg_inputs 中的一个来传递示例输入。
    # 如果 example_inputs 是一个单一的 Tensor，则会自动将其封装成一个元组。
    # 当 example_inputs 为 None 时，应指定 example_kwarg_inputs。
    def trace(
        example_inputs=None,
        check_trace=True,
        check_inputs=None,
        check_tolerance=None,
        strict=True,
        example_kwarg_inputs=None
    ):
        # 检查跟踪的代码是否在相同的输入下产生相同的输出。
        # 默认开启检查（True）。如果网络包含非确定性操作，或者你确信网络是正确的但检查失败了，可以禁用此选项。
        # check_inputs 是一个包含输入参数元组的列表，用于检查跟踪结果是否符合预期。
        # 最好传入一组代表输入空间形状和类型的检查输入。如果未指定，则使用原始的 example_inputs 进行检查。
        # check_tolerance 是浮点数比较的容差，用于在检查程序中使用。可以用于在已知数值分歧的情况下放宽检查的严格性，例如操作融合。
        # strict 是一个布尔值，控制跟踪器是否运行在严格模式下（默认 True）。
        # 当你希望跟踪器记录你的可变容器类型（目前支持 list/dict），并且你确信在你的问题中使用的容器是一个常量结构且不会被作为控制流（如 if、for）的条件时，可以将此选项设置为 False。
        # example_kwarg_inputs 是一个包含示例输入关键字参数的字典。在跟踪时会作为关键字参数传递给函数。
        # 默认为 None。必须指定 example_inputs 或 example_kwarg_inputs 中的一个。字典中的键必须与跟踪函数的参数名匹配，否则会引发运行时异常。
    """
    如果 `_enabled` 不为真，则直接返回 `func`。
    如果 `optimize` 参数不为 None，则发出警告，提示 `optimize` 已废弃且不再起作用，建议使用 `with torch.jit.optimized_execution()` 替代。

    从 torch._utils_internal 模块导入必要的函数：
        - check_if_torch_exportable: 检查是否可以导出到 TorchScript
        - log_torch_jit_trace_exportability: 记录 TorchScript 导出性能日志
        - log_torchscript_usage: 记录 TorchScript 使用情况日志

    记录使用 TorchScript 进行跟踪操作的日志
    使用 `_trace_impl` 函数进行实际的跟踪操作：
        - func: 待跟踪的函数或模块
        - example_inputs: 示例输入
        - optimize: 优化参数（已废弃）
        - check_trace: 是否检查跟踪
        - check_inputs: 是否检查输入
        - check_tolerance: 检查容差
        - strict: 是否严格模式
        - _force_outplace: 是否强制使用 outplace 操作
        - _module_class: 模块类
        - _compilation_unit: 编译单元
        - example_kwarg_inputs: 示例关键字参数输入
        - _store_inputs: 是否存储输入

    返回跟踪后的函数或模块对象 `traced_func`。
    """
    if not _enabled:
        return func
    if optimize is not None:
        warnings.warn(
            "`optimize` is deprecated and has no effect. "
            "Use `with torch.jit.optimized_execution()` instead",
            FutureWarning,
            stacklevel=2,
        )

    from torch._utils_internal import (
        check_if_torch_exportable,
        log_torch_jit_trace_exportability,
        log_torchscript_usage,
    )

    log_torchscript_usage("trace")
    traced_func = _trace_impl(
        func,
        example_inputs,
        optimize,
        check_trace,
        check_inputs,
        check_tolerance,
        strict,
        _force_outplace,
        _module_class,
        _compilation_unit,
        example_kwarg_inputs,
        _store_inputs,
    )

    return traced_func
_trace_module_map: Optional[Dict[Any, Any]] = None
# 定义一个全局变量 _trace_module_map，类型为可选的字典，初始值为 None


def trace_module(
    mod,
    inputs,
    optimize=None,
    check_trace=True,
    check_inputs=None,
    check_tolerance=1e-5,
    strict=True,
    _force_outplace=False,
    _module_class=None,
    _compilation_unit=_python_cu,
    example_inputs_is_kwarg=False,
    _store_inputs=True,
):
    """
    Trace a module and return an executable :class:`ScriptModule` that will be optimized using just-in-time compilation.

    When a module is passed to :func:`torch.jit.trace <torch.jit.trace>`, only
    the ``forward`` method is run and traced. With ``trace_module``, you can specify a dictionary of
    method names to example inputs to trace (see the ``inputs``) argument below.

    See :func:`torch.jit.trace <torch.jit.trace>` for more information on tracing.

    Args:
        mod (torch.nn.Module):  A ``torch.nn.Module`` containing methods whose names are
                                specified in ``inputs``. The given methods will be compiled
                                as a part of a single `ScriptModule`.
        inputs (dict):  A dict containing sample inputs indexed by method names in ``mod``.
                                The inputs will be passed to methods whose names correspond to inputs'
                                keys while tracing.
                                ``{ 'forward' : example_forward_input, 'method2': example_method2_input}``
    """
    # 追踪一个模块并返回一个可执行的 ScriptModule，该模块将使用即时编译进行优化。

    # 当一个模块传递给 torch.jit.trace 时，只有 'forward' 方法会被执行和追踪。
    # 使用 trace_module，可以指定一个方法名到示例输入的字典以进行追踪（见下面的 inputs 参数）。

    # 有关追踪的更多信息，请参见 torch.jit.trace。

    # 参数:
    # mod (torch.nn.Module): 包含在 inputs 中指定方法的 torch.nn.Module。
    #                        给定的方法将作为单个 ScriptModule 的一部分进行编译。
    # inputs (dict): 一个字典，包含按 mod 中方法名索引的示例输入。
    #                输入将传递给方法，其名称与输入的键对应而追踪。
    #                {'forward' : example_forward_input, 'method2': example_method2_input}
    # 定义函数 torch.jit.trace()，用于将 torch.nn.Module 中的 forward 方法转换为 TorchScript
    # 以便进行优化和部署
    Keyword arguments:
        # check_trace 是一个布尔值，指示是否检查通过追踪代码运行的相同输入是否产生相同输出，默认为 True。
        # 如果你的网络包含非确定性操作，或者你确信网络正确但检查失败，可以禁用此选项。
        check_trace (bool, optional): Check if the same inputs run through traced code produce the same outputs. Default: True.

        # check_inputs 是一个字典列表，用于检查追踪的代码是否符合预期的输入参数。
        # 每个字典对应一组输入参数，应与 inputs 中指定的输入对应。为了获得最佳结果，
        # 传入一个代表网络可能接收的输入形状和类型空间的检查输入集合。
        # 如果未指定，将使用原始的 inputs 进行检查。
        check_inputs (list of dicts, optional): A list of dicts of input arguments that should be used
                                                 to check the trace against what is expected.

        # check_tolerance 是浮点数比较容差，用于检查过程中的浮点数比较容差。
        # 可以在结果数值上发生分歧的已知原因下，如运算符融合，可用于放宽检查的严格性。
        check_tolerance (float, optional): Floating-point comparison tolerance to use in the checker procedure.

        # example_inputs_is_kwarg 是一个布尔值，指示示例输入是否是关键字参数包。
        # 默认为 False。
        example_inputs_is_kwarg (bool, optional): This parameter indicate whether the example inputs is a pack of keyword arguments. Default: False.

    Returns:
        # 返回一个 ScriptModule 对象，其中包含一个单一的 forward 方法，包含追踪的代码。
        # 当 func 是一个 torch.nn.Module 时，返回的 ScriptModule 将具有与 func 相同的子模块和参数。
        A :class:`ScriptModule` object with a single ``forward`` method containing the traced code.
        When ``func`` is a ``torch.nn.Module``, the returned :class:`ScriptModule` will have the same set of
        sub-modules and parameters as ``func``.
    if not _enabled:
        return mod


    # 如果 _enabled 不为真，则直接返回 mod
    if not _enabled:
        return mod



    if optimize is not None:
        warnings.warn(
            "`optimize` is deprecated and has no effect. "
            "Use `with torch.jit.optimized_execution()` instead",
            FutureWarning,
            stacklevel=2,
        )


    # 如果 optimize 参数不为 None，则发出警告，提醒用户该参数已被废弃并且不起作用
    if optimize is not None:
        warnings.warn(
            "`optimize` is deprecated and has no effect. "
            "Use `with torch.jit.optimized_execution()` instead",
            FutureWarning,
            stacklevel=2,
        )



    var_lookup_fn = _create_interpreter_name_lookup_fn(0)


    # 使用 _create_interpreter_name_lookup_fn(0) 创建一个变量查找函数 var_lookup_fn
    var_lookup_fn = _create_interpreter_name_lookup_fn(0)



    if not isinstance(mod, torch.nn.Module):
        raise AttributeError("expected torch.nn.Module as the first argument")


    # 如果 mod 不是 torch.nn.Module 类型，则抛出 AttributeError 异常
    if not isinstance(mod, torch.nn.Module):
        raise AttributeError("expected torch.nn.Module as the first argument")



    if not isinstance(inputs, dict):
        raise AttributeError("expected a dictionary of (method_name, input) pairs")


    # 如果 inputs 不是字典类型，则抛出 AttributeError 异常，要求其为 (method_name, input) 对的字典
    if not isinstance(inputs, dict):
        raise AttributeError("expected a dictionary of (method_name, input) pairs")



    old_module_map = torch.jit._trace._trace_module_map


    # 将当前的 _trace_module_map 赋值给 old_module_map
    old_module_map = torch.jit._trace._trace_module_map



    finally:
        torch.jit._trace._trace_module_map = old_module_map


    # 在结束时，将 _trace_module_map 恢复为之前保存的 old_module_map
    finally:
        torch.jit._trace._trace_module_map = old_module_map



    return module


    # 返回 trace 后的 module 对象
    return module
# 定义函数 is_tracing，用于判断当前是否处于追踪状态
def is_tracing():
    """Return a boolean value.

    Returns ``True`` in tracing (if a function is called during the
    tracing of code with ``torch.jit.trace``) and ``False`` otherwise.
    """
    # 调用 is_scripting 函数判断是否处于脚本化状态，如果是，则返回 False
    if is_scripting():
        return False
    # 否则，调用 torch._C._is_tracing() 函数判断是否正在追踪代码，返回其结果
    return torch._C._is_tracing()


# 定义类 TracedModule，继承自 ScriptModule 类
class TracedModule(ScriptModule):
    # 禁用脚本化元数据，设置类属性 _disable_script_meta 为 True
    _disable_script_meta = True
    def __init__(self, orig, id_set=None, _compilation_unit=None):
        # XXX: orig can be a nn.Module or a function!
        # 初始化函数，用于创建一个新的 TracedModule 对象
        super().__init__()
        # 断言 orig 是 torch.nn.Module 类型的对象
        assert isinstance(orig, torch.nn.Module)

        # Copy a subset of `orig` to a temporary nn.Module.
        # This is a way to customize what will actually get compiled by create_script_module
        # 复制 orig 的部分内容到临时的 nn.Module 对象中，用于自定义编译过程中的内容
        id_set = set()  # 创建一个空集合 id_set

        # This allows us to preserve the original module's qualified name by defining a new
        # type with the attribute _jit_override_qualname. In torch._jit_internal._qualified_name
        # we have a special case that will look up this attribute to override whatever qualname
        # we would get from the python type system
        # 定义一个新的类型 QualnameWrapper，用于保留原始模块的限定名称
        class QualnameWrapper(torch.nn.Module):
            pass

        # 设置 QualnameWrapper 类型的 _jit_override_qualname 属性，用于保存原始模块的限定名称
        QualnameWrapper._jit_override_qualname = torch._jit_internal._qualified_name(  # type: ignore[attr-defined]
            type(orig)
        )

        # 创建一个 QualnameWrapper 类的实例 tmp_module
        tmp_module = QualnameWrapper()

        # 定义一个函数 check_unique，用于检查参数的唯一性
        def check_unique(param):
            if param in id_set:
                raise ValueError(
                    "TracedModules don't support parameter sharing between modules"
                )
            id_set.add(param)

        # 将原始模块的 training 状态赋给 tmp_module
        tmp_module.training = orig.training

        # 将原始模块的参数复制到 tmp_module 中，并检查参数的唯一性
        for name, param in orig._parameters.items():
            if param is not None:
                tmp_module._parameters[name] = param
                check_unique(param)
        # 将原始模块的缓冲区复制到 tmp_module 中，并检查缓冲区的唯一性
        for name, buf in orig._buffers.items():
            if buf is not None:
                tmp_module._buffers[name] = buf
                check_unique(buf)
        # 将原始模块的其他属性复制到 tmp_module 中，如果属性是脚本对象则赋值给 tmp_module
        for name, val in orig.__dict__.items():
            if (
                torch._C._jit_is_script_object(val)
                and name not in orig._parameters
                and name not in orig._buffers
            ):
                setattr(tmp_module, name, val)

        # 如果原始模块有 backward hooks，则抛出 ValueError
        if orig._backward_hooks:
            raise ValueError(
                "Modules that have backward hooks assigned can't be compiled: "
                + str(orig)
            )

        # 递归地创建 tmp_module 的脚本模块 script_module
        for name, submodule in orig._modules.items():
            if submodule is None:
                continue
            tmp_module._modules[name] = make_module(
                submodule, TracedModule, _compilation_unit=None
            )

        # 使用 create_script_module 创建脚本模块 script_module，并保存到 self.__dict__
        script_module = torch.jit._recursive.create_script_module(
            tmp_module, lambda module: (), share_types=False, is_tracing=True
        )

        # 设置对象的 _name 属性为 orig 的类型名称
        self.__dict__["_name"] = type(orig).__name__
        # 设置对象的 _actual_script_module 属性为 script_module
        self.__dict__["_actual_script_module"] = script_module

        # 删除对象的 _parameters、_buffers、_modules 和 training 属性
        for name in ("_parameters", "_buffers", "_modules", "training"):
            delattr(self, name)

    def forward(self, *args, **kwargs):
        # 抛出运行时异常，不允许调用 forward 方法
        raise RuntimeError("Trace submodules cannot be called.")

    def __getattr__(self, attr):
        if "_actual_script_module" not in self.__dict__:
            return super().__getattr__(attr)
        # 返回 _actual_script_module 的属性
        return getattr(self._actual_script_module, attr)
    # 覆盖默认的属性设置方法，检查是否存在实际的脚本模块，若不存在则调用父类方法设置属性值
    def __setattr__(self, attr, value):
        # 检查是否存在实际的脚本模块
        if "_actual_script_module" not in self.__dict__:
            # 如果不存在实际的脚本模块，则调用父类的属性设置方法
            return super().__setattr__(attr, value)
        # 如果存在实际的脚本模块，则通过实际脚本模块设置属性值
        setattr(self._actual_script_module, attr, value)
    
    # 返回对象的名称属性 _name
    def _get_name(self):
        return self._name
    
    # 返回一个额外的字符串表示，展示对象的原始名称 _name
    def extra_repr(self):
        return f"original_name={self._name}"
class TopLevelTracedModule(TracedModule):
    # 类型标注：forward 是一个接受任意参数并返回任意类型的 Callable 对象，初始为 _CachedForward() 的实例
    forward: Callable[..., Any] = _CachedForward()  # type: ignore[assignment]

    def _reconstruct(self, cpp_module):
        """
        用给定的 C++ 模块实例重构一个 TopLevelTracedModule 实例。

        Args:
            cpp_module: 将用于重建此 TopLevelTracedModule 的 C++ 模块实例。
        """
        # 调用 _actual_script_module 的 _reconstruct 方法，使用 cpp_module 重构对象
        self.__dict__["_actual_script_module"]._reconstruct(cpp_module)


def _script_if_tracing(fn: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if not is_tracing():
            # 如果不处于追踪状态，直接返回原始函数 fn 的执行结果
            return fn(*args, **kwargs)

        # 将 wrapper 的原始函数 fn 脚本化并编译为 compiled_fn
        compiled_fn: Callable[P, R] = script(wrapper.__original_fn)  # type: ignore[attr-defined]
        return compiled_fn(*args, **kwargs)

    # 记录 wrapper 的原始函数 fn
    wrapper.__original_fn = fn  # type: ignore[attr-defined]
    # 设置标记，表示此函数是在追踪时使用的包装器
    wrapper.__script_if_tracing_wrapper = True  # type: ignore[attr-defined]

    return wrapper


def _get_trace_graph(
    f,
    args=(),
    kwargs=None,
    strict=True,
    _force_outplace=False,
    return_inputs=False,
    _return_inputs_states=False,
):
    """返回一个函数或模型的追踪图的元组。

    .. warning::
        此函数仅限内部使用，应由 ONNX 导出器使用。如果要通过追踪获取图，请使用公共 API::

            trace = torch.jit.trace(nn.LSTMCell(), (input, hidden))
            trace_graph = trace.graph

    追踪函数或模型，返回一个元组，包含执行的追踪和原始返回值。如果 return_inputs 为真，则还返回追踪的输入作为元组的一部分。

    追踪保证不会改变被追踪函数/模块的语义。

    Args:
        f (torch.nn.Module or function): 要追踪的函数或模块。
        args (tuple or Tensor): 传递给要追踪的函数/模块的位置参数。非元组被假定为要传递给模型的单个位置参数。
        kwargs (dict): 传递给要追踪的函数/模块的关键字参数。

    示例（追踪一个单元）：

    .. testcode::

        trace = torch.jit.trace(nn.LSTMCell(), (input, hidden))
    """
    if kwargs is None:
        kwargs = {}
    if not isinstance(args, tuple):
        args = (args,)
    
    # 使用 ONNXTracedModule 追踪 f，并返回其输出
    outs = ONNXTracedModule(
        f, strict, _force_outplace, return_inputs, _return_inputs_states
    )(*args, **kwargs)
    return outs
```