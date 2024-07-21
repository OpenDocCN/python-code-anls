# `.\pytorch\torch\onnx\verification.py`

```py
# mypy: allow-untyped-defs
"""Functions to verify exported ONNX model is functionally equivalent to original PyTorch model.

ONNX Runtime is required, and is used as the ONNX backend for export verification.
"""

# 引入必要的模块和库
import contextlib               # 上下文管理模块，用于支持上下文管理器和相关功能
import copy                     # 复制对象的模块
import dataclasses              # 支持数据类的模块，用于定义数据结构
import datetime                 # 处理日期和时间的模块
import difflib                  # 生成差异，比较文本的模块
import enum                     # 枚举类型的支持模块
import functools                # 函数工具模块，提供用于函数操作的工具
import io                       # 输入输出流的核心工具
import itertools                # 生成迭代器的工具模块
import os                       # 操作系统相关的功能模块
import tempfile                 # 创建临时文件和目录的模块
import warnings                 # 警告处理的模块
from typing import (            # 类型提示模块，定义函数和变量的类型
    Any,                       # 任意类型
    Callable,                  # 可调用对象类型
    Collection,                # 集合类型
    Dict,                      # 字典类型
    FrozenSet,                 # 不可变集合类型
    List,                      # 列表类型
    Mapping,                   # 映射类型
    Optional,                  # 可选类型
    Sequence,                  # 序列类型
    Set,                       # 集合类型
    Tuple,                     # 元组类型
    Union,                     # 联合类型
)

import numpy as np              # 数值计算库，支持多维数组与矩阵运算

import torch                    # PyTorch深度学习框架
import torch._C._onnx as _C_onnx # PyTorch的ONNX C++接口
from torch import _C            # PyTorch C++扩展模块
from torch.onnx import (        # PyTorch的ONNX导出相关模块
    _constants,                # 常量定义模块
    _experimental,             # 实验性功能模块
    _exporter_states,          # 导出状态模块
    utils                      # 实用工具模块
)
from torch.onnx._globals import GLOBALS  # 全局变量模块
from torch.onnx._internal import _beartype, onnx_proto_utils  # ONNX内部工具模块
from torch.types import Number  # 类型定义模块，包括数字类型

_ORT_PROVIDERS = ("CPUExecutionProvider",)  # ONNX Runtime提供的执行提供者，仅包括CPUExecutionProvider

_NumericType = Union[Number, torch.Tensor, np.ndarray]  # 数值类型，可以是数字、PyTorch张量或NumPy数组
_ModelType = Union[torch.nn.Module, torch.jit.ScriptModule]  # 模型类型，可以是PyTorch模块或脚本模块
_InputArgsType = Union[torch.Tensor, Tuple[Any, ...]]  # 输入参数类型，可以是张量或任意元组
_InputKwargsType = Mapping[str, Any]  # 输入关键字参数类型，键为字符串，值为任意类型
_OutputsType = Union[Sequence[_NumericType], Sequence]  # 输出类型，可以是数值类型序列或一般序列


class OnnxBackend(enum.Enum):
    """Enum class for ONNX backend used for export verification."""
    
    REFERENCE = "ONNXReferenceEvaluator"       # 参考实现的ONNX后端
    ONNX_RUNTIME_CPU = "CPUExecutionProvider"  # ONNX Runtime的CPU后端
    ONNX_RUNTIME_CUDA = "CUDAExecutionProvider" # ONNX Runtime的CUDA后端


@dataclasses.dataclass
class VerificationOptions:
    """Options for ONNX export verification."""
    # ONNX导出验证的选项类
    # 定义类的属性和默认取值
    
    Attributes:
        # 如果为 True，将嵌套的列表/元组/字典输入展开为 ONNX 所需的平铺张量。当导出 ScriptModule 时，通常需要保留嵌套结构，此时应将其设置为 False。默认为 True。
        flatten: bool = True
    
        # 是否在 torch 输出中忽略 None 类型，这通常在跟踪时使用。如果 torch 输出应保留 None 类型（通常在导出 ScriptModule 时），则将此选项设置为 False。默认为 True。
        ignore_none: bool = True
    
        # 是否检查 PyTorch 和 ONNX Runtime 输出的形状是否完全相同。将此设置为 False 允许输出形状广播。默认为 True。
        check_shape: bool = True
    
        # 是否检查 PyTorch 和 ONNX Runtime 输出的数据类型是否一致。默认为 True。
        check_dtype: bool = True
    
        # 用于验证的 ONNX 后端。默认为 OnnxBackend.ONNX_RUNTIME_CPU。
        backend: OnnxBackend = OnnxBackend.ONNX_RUNTIME_CPU
    
        # 在 ONNX 和 PyTorch 输出之间的相对容差。默认为 1e-3。
        rtol: float = 1e-3
    
        # 在 ONNX 和 PyTorch 输出之间的绝对容差。默认为 1e-7。
        atol: float = 1e-7
    
        # 如果提供了，则仅传递指定的输入到 ONNX 模型中。当模型存在未使用的输入时，应提供一个输入列表。由于在导出的 ONNX 模型中会移除未使用的输入，提供所有输入会导致意外输入的错误。此参数告诉验证器应传递哪些输入到 ONNX 模型中。
        remained_onnx_input_idx: Optional[Sequence[int]] = None
    
        # 允许的元素不匹配百分比。应为一个介于 0.0 和 1.0 之间的浮点数。
        acceptable_error_percentage: Optional[float] = None
# 使用 @_beartype 装饰器对函数进行类型检查和验证
@_beartype.beartype
# 递归地将元组展平为列表的函数
def _flatten_tuples(elem):
    flattened = []
    for t in elem:
        if isinstance(t, tuple):  # 如果当前元素是元组
            flattened.extend(_flatten_tuples(t))  # 递归展平元组
        else:
            flattened.append(t)  # 如果不是元组，直接添加到结果列表中
    return flattened  # 返回展平后的列表


# TODO(justinchuby): 在输入为 None 时，通过缩小返回类型来添加类型检查
# 将输入元素转换为 numpy 数组或列表的函数，返回类型可能是 list 或 np.ndarray
def _to_numpy(elem) -> Union[list, np.ndarray]:
    if isinstance(elem, torch.Tensor):  # 如果输入是 torch.Tensor
        if elem.requires_grad:  # 如果 Tensor 需要梯度追踪
            return elem.detach().cpu().numpy()  # 返回去除梯度并转为 numpy 数组的值
        else:
            return elem.cpu().numpy()  # 返回转为 numpy 数组的值
    elif isinstance(elem, (list, tuple)):  # 如果输入是列表或元组
        return [_to_numpy(inp) for inp in elem]  # 递归地将列表或元组中的每个元素转为 numpy 类型
    elif isinstance(elem, (bool, int, float)):  # 如果输入是布尔值、整数或浮点数
        return np.array(elem)  # 返回转为 numpy 数组的值
    elif isinstance(elem, dict):  # 如果输入是字典
        flattened = []
        for k in elem:
            flattened.extend([_to_numpy(k), _to_numpy(elem[k])])  # 递归地将字典的键和值转为 numpy 类型并展平
        return flattened  # 返回展平后的列表
    return elem  # 对于其他类型的输入，直接返回原始值


# 使用 @_beartype 装饰器对函数进行类型检查和验证
@_beartype.beartype
# 将嵌套的列表展开为单层列表的函数
def _inline_flatten_list(inputs, res_list) -> list:
    for i in inputs:
        # 如果当前元素不是列表或元组，则直接添加到结果列表中；否则递归展开列表或元组
        res_list.append(i) if not isinstance(i, (list, tuple)) else _inline_flatten_list(i, res_list)
    return res_list  # 返回展平后的列表


# 使用 @_beartype 装饰器对函数进行类型检查和验证
@_beartype.beartype
# 将输入的值解包并转换为 numpy 数组列表的函数
def _unpack_to_numpy(values, cast_onnx_accepted=True) -> list:
    value_unpacked = []
    for value in values:
        # 使用 utils.unpack_quantized_tensor 函数解包量化的张量值
        value_unpacked.extend(
            utils.unpack_quantized_tensor(value, cast_onnx_accepted=cast_onnx_accepted)
        )
    return [_to_numpy(v) for v in value_unpacked]  # 返回解包后并转为 numpy 数组的值的列表


# 使用 @_beartype 装饰器对函数进行类型检查和验证
@_beartype.beartype
# 运行 ONNX 会话的函数，返回类型为 _OutputsType
def _run_onnx(onnx_session, inputs) -> _OutputsType:
    kw_inputs = {}
    if inputs and isinstance(inputs[-1], dict):  # 如果输入非空且最后一个元素是字典
        kw_inputs = inputs[-1]  # 将最后一个元素作为关键字参数输入
        inputs = inputs[:-1]  # 去掉最后一个元素，更新输入列表
    inputs = _unpack_to_numpy(_flatten_tuples(inputs))  # 解包并展平输入，并转为 numpy 数组
    ort_inputs = {}
    for input_name, input in kw_inputs.items():
        ort_inputs[input_name] = _to_numpy(input)  # 将关键字参数转为 numpy 数组并添加到 ort_inputs 中
    inputs = _to_numpy(inputs)  # 将输入列表转为 numpy 数组
    if hasattr(onnx_session, "get_inputs"):  # 如果 onnx_session 具有 get_inputs 方法
        # 获取 ONNX 模型输入的名称列表
        input_names = [i.name for i in onnx_session.get_inputs()]
    elif hasattr(onnx_session, "input_names"):  # 如果 onnx_session 具有 input_names 属性
        # 获取 ONNX 模型输入的名称列表
        input_names = onnx_session.input_names
    else:
        # 抛出异常，表示未知的 ONNX 后端类型
        raise ValueError(f"Unknown ONNX backend type: {type(onnx_session)}.")
    
    for i, input in enumerate(inputs):
        if i == len(input_names) or input_names[i] in ort_inputs:
            # 如果位置输入过多，或者 input_names 中的名称在 ort_inputs 中
            raise ValueError(
                f"got too many positional inputs. inputs: {inputs}. kw_inputs: {kw_inputs}. "
                f"input names: {input_names}."
            )
        ort_inputs[input_names[i]] = input  # 将输入添加到 ort_inputs 中
    
    onnx_outs = onnx_session.run(None, ort_inputs)  # 运行 ONNX 会话并获取输出
    return onnx_outs  # 返回 ONNX 的输出结果


# 使用 @_beartype 装饰器对函数进行类型检查和验证
@_beartype.beartype
# 使用 ONNX 运行会话的函数，接受模型名称或字节流作为输入
def _ort_session(
    model: Union[str, io.BytesIO], ort_providers: Sequence[str] = _ORT_PROVIDERS
):
    try:
        import onnxruntime  # 导入 onnxruntime 库
    except ImportError as e:
        raise ImportError("onnxruntime is required for export verification.") from e  # 抛出导入异常
    # 如果 ort_providers 参数为 None，则使用默认的 _ORT_PROVIDERS
    if ort_providers is None:
        ort_providers = _ORT_PROVIDERS

    # 创建一个 SessionOptions 对象，用于配置 ONNX Runtime 的会话选项
    session_options = onnxruntime.SessionOptions()

    # 设置日志的严重程度级别：
    # 0: 详细信息 (Verbose)
    # 1: 信息 (Info)
    # 2: 警告 (Warning)
    # 3: 错误 (Error)
    # 4: 致命错误 (Fatal)
    # 在这里将日志级别设置为 3，即只显示错误信息
    session_options.log_severity_level = 3

    # 创建 ONNX Runtime 推理会话对象 InferenceSession
    # model 参数可以是字符串（文件路径）或者是字节流对象，这里根据类型来选择
    ort_session = onnxruntime.InferenceSession(
        model if isinstance(model, str) else model.getvalue(),  # 如果 model 是字符串，则直接使用；如果是字节流，则获取其内容
        session_options,  # 传入前面创建的会话选项对象
        providers=ort_providers,  # 设置推理会话的提供者，可以是 CPU、GPU 等
    )

    # 返回创建好的 ONNX Runtime 推理会话对象
    return ort_session
# 使用装饰器 beartype 对 _onnx_reference_evaluator_session 函数进行类型检查和验证
@_beartype.beartype
def _onnx_reference_evaluator_session(model: Union[str, io.BytesIO]):
    try:
        # 尝试导入 onnx 模块和 onnx 的 reference 子模块，如果导入失败则抛出 ImportError
        import onnx
        from onnx import reference as onnx_reference  # type: ignore[attr-defined]
    except ImportError as exc:
        # 如果导入失败，则抛出 ImportError，提示需要安装 onnx >= 1.13 版本
        raise ImportError("onnx >= 1.13 is required for reference evaluator.") from exc

    # 根据输入的 model 参数类型，加载对应的 ONNX 模型数据
    proto = (
        onnx.load(model)  # type: ignore[attr-defined]
        if isinstance(model, str)
        else onnx.load_model_from_string(model.getvalue())  # type: ignore[attr-defined]
    )
    # 使用加载的 ONNX 模型数据创建 ReferenceEvaluator 对象
    onnx_session = onnx_reference.ReferenceEvaluator(proto)
    return onnx_session


# 使用装饰器 beartype 对 _onnx_backend_session 函数进行类型检查和验证
@_beartype.beartype
def _onnx_backend_session(model: Union[str, io.BytesIO], backend: OnnxBackend):
    # 根据指定的 backend 参数选择不同的 ONNX 后端执行会话
    if backend == OnnxBackend.REFERENCE:
        # 如果选择的是 REFERENCE 后端，则调用 _onnx_reference_evaluator_session 函数
        onnx_session = _onnx_reference_evaluator_session(model)
    elif backend in {OnnxBackend.ONNX_RUNTIME_CPU, OnnxBackend.ONNX_RUNTIME_CUDA}:
        # 如果选择的是 ONNX_RUNTIME_CPU 或 ONNX_RUNTIME_CUDA 后端，则调用 _ort_session 函数
        onnx_session = _ort_session(model, (backend.value,))
    else:
        # 如果 backend 参数不在支持的列表中，则抛出 ValueError 异常
        raise ValueError(f"Unsupported backend: {backend}")
    return onnx_session


# 使用装饰器 beartype 对 _compare_onnx_pytorch_outputs_in_np 函数进行类型检查和验证
@_beartype.beartype
def _compare_onnx_pytorch_outputs_in_np(
    onnx_outs: _OutputsType,
    pt_outs: _OutputsType,
    options: VerificationOptions,
):
    # 断言 ONNX 输出和 PyTorch 输出的数量相同
    assert len(onnx_outs) == len(
        pt_outs
    ), f"Number of outputs differ ONNX runtime: ({len(onnx_outs)}) PyTorch: ({len(pt_outs)})"

    # 获取可接受的误差百分比阈值
    acceptable_error_percentage = options.acceptable_error_percentage
    # 如果设置了 acceptable_error_percentage 并且其值不在 0.0 到 1.0 之间，则抛出 ValueError 异常
    if acceptable_error_percentage and (
        acceptable_error_percentage > 1.0 or acceptable_error_percentage < 0.0
    ):
        raise ValueError(
            "If set, acceptable_error_percentage should be between 0.0 and 1.0"
        )
    for ort_out, pt_out in zip(onnx_outs, pt_outs):
        try:
            # TODO: Remove `check_shape` option once every shape inconsistent issue is addressed.
            # 如果没有启用检查形状选项，则允许不同但是可广播的输出形状。
            if not options.check_shape:
                ort_out, pt_out = np.broadcast_arrays(ort_out, pt_out)
            # 使用 Torch 的测试工具进行近似相等性检查
            torch.testing.assert_close(
                ort_out,
                pt_out,
                rtol=options.rtol,
                atol=options.atol,
                check_dtype=options.check_dtype,
                equal_nan=True,
            )
        except AssertionError as e:
            # 如果设置了可接受的误差百分比，则计算误差百分比并发出警告
            if acceptable_error_percentage:
                error_percentage = 1 - np.sum(
                    np.isclose(ort_out, pt_out, rtol=options.rtol, atol=options.atol)
                ) / np.prod(ort_out.shape)
                if error_percentage <= acceptable_error_percentage:
                    warnings.warn(
                        f"Suppressed AssertionError:\n{e}.\n"
                        f"Error percentage {error_percentage} "
                        f"within acceptable range {acceptable_error_percentage}."
                    )
                    continue
            # 如果 ONNX 输出或 PyTorch 输出的数据类型是 uint8 或 int8，则发出量化警告
            if ort_out.dtype == np.uint8 or ort_out.dtype == np.int8:
                warnings.warn("ONNX output is quantized")
            if pt_out.dtype == np.uint8 or pt_out.dtype == np.int8:
                warnings.warn("PyTorch output is quantized")
            # 抛出原始的 AssertionError
            raise
# 使用装饰器确保函数输入类型符合预期
@_beartype.beartype
# 函数用于比较 ONNX 和 PyTorch 模型的输出结果
def _compare_onnx_pytorch_outputs(
    onnx_outs: _OutputsType,
    pt_outs: Any,
    options: VerificationOptions,
):
    """
    Compare ONNX and PyTorch outputs.

    Args:
        onnx_outs: outputs from ONNX backend.
        pt_outs: outputs from PyTorch.
        options: options for verification.

    Raises:
        AssertionError: if outputs from ONNX model and PyTorch model are not
            equal up to specified precision.
        ValueError: if arguments provided are invalid.
    """
    # 如果设置了忽略 None 类型的选项
    if options.ignore_none:
        # 使用 torch.jit._flatten 过滤掉 None 类型的输出
        pt_outs, _ = torch.jit._flatten(pt_outs)
    else:
        # 否则对输出进行扁平化处理
        pt_outs = _inline_flatten_list([pt_outs], [])
    
    # 将 PyTorch 输出转换为 NumPy 数组
    pt_outs_np = _unpack_to_numpy(pt_outs, cast_onnx_accepted=False)
    
    # 对 ONNX 输出进行扁平化处理
    onnx_outs = _inline_flatten_list(onnx_outs, [])
    
    # 调用函数，比较两个输出的 NumPy 数组
    _compare_onnx_pytorch_outputs_in_np(onnx_outs, pt_outs_np, options)


# 使用装饰器确保函数输入类型符合预期
@_beartype.beartype
# 函数用于为 PyTorch 模型的执行准备输入
def _prepare_input_for_pytorch(args, kwargs):
    """Prepare input for PyTorch model execution.

    Any future changes/formatting to the input before dispatching to the PyTorch
    model should be made in this function.

    Args:
        args: positional arguments for PyTorch model forward method.
        kwargs: keyword arguments for PyTorch model forward method.

    Returns:
        args: positional arguments for PyTorch model forward method.
        kwargs: keyword arguments for PyTorch model forward method.
    """
    # 如果参数是 torch.Tensor 或 dict 类型
    if isinstance(args, (torch.Tensor, dict)):
        args = (args,)
    
    # 深度复制输入参数，以防止原始数据被修改
    args = copy.deepcopy(args)
    
    # 如果存在关键字参数 kwargs，同样进行深度复制
    if kwargs:
        kwargs = copy.deepcopy(kwargs)
    else:
        kwargs = {}
    
    return args, kwargs


# 使用装饰器确保函数输入类型符合预期
@_beartype.beartype
# 函数用于为导出 ONNX 模型的执行准备输入
def _prepare_input_for_export(args, kwargs):
    """Prepare input for ONNX model export.

    Any future changes/formatting to the input before dispatching to the
    :func:`torch.onnx.export` api should be made in this function.

    Args:
        args: positional arguments for PyTorch model forward method.
        kwargs: keyword arguments for PyTorch model forward method.

    Returns:
        onnx_inputs: positional arguments for ONNX model export, as `args` in
            :func:`torch.onnx.export`.
    """
    # 调用函数，为 PyTorch 模型的执行准备输入
    args, kwargs = _prepare_input_for_pytorch(args, kwargs)
    
    # 如果没有关键字参数 kwargs，并且 args 的最后一个元素是 dict 类型
    if not kwargs and len(args) > 0 and isinstance(args[-1], dict):
        # 将一个空的 dict 加入 args 中
        onnx_inputs = args + ({},)
    elif kwargs:
        # 将 kwargs 加入 args 中
        onnx_inputs = args + (kwargs,)
    else:
        # 否则直接使用 args
        onnx_inputs = args
    
    return onnx_inputs


# 使用装饰器确保函数输入类型符合预期
@_beartype.beartype
# 函数用于为 ONNX 模型的执行准备输入
def _prepare_input_for_onnx(
    args, kwargs, remained_onnx_input_idx: Optional[Sequence[int]], flatten: bool
):
    """Prepare input for ONNX model execution in ONNX backend.

    Any future changes/formatting to the input before dispatching to the ONNX backend
    run should be made in this function.
    """
    # 在这里实现将输入参数格式化为 ONNX 后端可以接受的格式
    # 准备输入以导出为 ONNX 模型执行所需的位置参数
    onnx_inputs = _prepare_input_for_export(args, kwargs)
    
    # 如果 flatten 标志为 True，则将输入展平以便传递给 ONNX 模型执行
    if flatten:
        onnx_inputs, _ = torch.jit._flatten(onnx_inputs)
    
    # 如果输入非空且最后一个元素是空字典，处理空的关键字参数（通常由 flatten 移除）
    elif onnx_inputs and onnx_inputs[-1] == {}:
        onnx_inputs = onnx_inputs[:-1]
    
    # 如果指定了 remained_onnx_input_idx，返回指定位置上的输入列表
    if remained_onnx_input_idx is not None:
        return [onnx_inputs[i] for i in remained_onnx_input_idx]
    else:
        # 否则返回所有的 onnx_inputs 作为 ONNX 模型执行的位置参数
        return onnx_inputs
@_beartype.beartype
def _try_clone_model(model):
    """尝试克隆模型，以防前向传播改变模型状态。

    Args:
        model: 要克隆的模型对象。

    Returns:
        copy.deepcopy(model): 深度克隆后的模型对象。

    Raises:
        如果克隆失败，发出警告，并返回原始模型对象。
    """
    try:
        return copy.deepcopy(model)
    except Exception:
        warnings.warn(
            "Failed to clone model. Model state might be mutated during verification."
        )
        return model


@_beartype.beartype
def _compare_onnx_pytorch_model(
    pt_model: _ModelType,
    onnx_model_f: Union[str, io.BytesIO],
    input_args: _InputArgsType,
    input_kwargs: Optional[_InputKwargsType],
    additional_test_inputs: Optional[Sequence[_InputArgsType]],
    options: VerificationOptions,
):
    """比较 ONNX 模型运行和 PyTorch 模型运行的输出结果。

    Args:
        pt_model: PyTorch 模型对象。
        onnx_model_f: ONNX 模型文件路径或文件对象。
        input_args: PyTorch 模型前向方法的位置参数。
        input_kwargs: PyTorch 模型前向方法的关键字参数。
        additional_test_inputs: PyTorch 模型前向方法的额外位置参数。
        options: 验证选项。

    Raises:
        AssertionError: 如果 ONNX 模型和 PyTorch 模型的输出结果不符合指定精度要求。
    """
    onnx_session = _onnx_backend_session(onnx_model_f, options.backend)

    @_beartype.beartype
    def compare_onnx_pytorch_model_with_input(input_args, input_kwargs):
        """使用给定的输入参数比较 ONNX 模型和 PyTorch 模型的输出结果。

        Args:
            input_args: PyTorch 模型前向方法的位置参数。
            input_kwargs: PyTorch 模型前向方法的关键字参数。
        """
        pt_args, pt_kwargs = _prepare_input_for_pytorch(input_args, input_kwargs)
        # TODO: remove this and treat mutating model separately. See #77679
        pt_model_copy = _try_clone_model(pt_model)
        pt_outs = pt_model_copy(*pt_args, **pt_kwargs)

        onnx_inputs = _prepare_input_for_onnx(
            input_args, input_kwargs, options.remained_onnx_input_idx, options.flatten
        )

        onnx_outs = _run_onnx(onnx_session, onnx_inputs)

        _compare_onnx_pytorch_outputs(
            onnx_outs=onnx_outs,
            pt_outs=pt_outs,
            options=options,
        )

    compare_onnx_pytorch_model_with_input(input_args, input_kwargs)

    if additional_test_inputs:
        for test_input_args in additional_test_inputs:
            compare_onnx_pytorch_model_with_input(test_input_args, {})


class _GraphDiff:
    """表示两个图之间差异的类。"""

    @_beartype.beartype
    def __init__(self, graph_a: _C.Graph, graph_b: _C.Graph):
        """构造 _GraphDiff 对象。

        Args:
            graph_a (_C.Graph): 第一个要比较的图。
            graph_b (_C.Graph): 第二个要比较的图。
        """
        self.graph_a = graph_a
        self.graph_b = graph_b

    @_beartype.beartype
    def __str__(self):
        """参见函数 :func:`diff_report`。"""
        return self.diff_report()

    @_beartype.beartype
    def _indent(self, lines: str) -> str:
        """将每行文本缩进一个制表符。

        Args:
            lines: 要缩进的文本字符串。

        Returns:
            str: 缩进后的文本字符串。
        """
        return "\n".join(["\t" + line for line in lines.splitlines()])
    def diff_report(self) -> str:
        """Return a string representation of the graph difference.

        The report shows the first pair of nodes that diverges. It also shows the source
        location of the pair of nodes.

        Returns:
            graph_diff_report (str): A string representation of the graph difference.
        """
        # 获取图形比较的两个图形对象
        graph_a = self.graph_a
        graph_b = self.graph_b

        # 将图形对象转换为字符串表示形式
        graph_a_str = str(graph_a)
        graph_b_str = str(graph_b)

        # 如果两个图形对象的字符串表示相同，则返回空字符串
        if graph_a_str == graph_b_str:
            return ""

        # 使用 difflib 库对两个图形对象的字符串表示进行行级别的差异比较
        graph_diff = difflib.ndiff(
            graph_a_str.splitlines(True), graph_b_str.splitlines(True)
        )
        # 初始化报告列表，包含基本的图形差异标识
        graph_diff_report = ["Graph diff:", self._indent("".join(graph_diff))]

        # 遍历两个图形的节点，寻找第一对不同的节点
        for node_a, node_b in itertools.zip_longest(graph_a.nodes(), graph_b.nodes()):
            if str(node_a) != str(node_b):
                graph_diff_report.append("First diverging operator:")
                # 对第一对不同节点进行行级别的差异比较
                node_diff = difflib.ndiff(
                    str(node_a).splitlines(True), str(node_b).splitlines(True)
                )
                # 初始化节点差异报告列表
                source_printout = ["node diff:", self._indent("".join(node_diff))]

                # 获取第一个节点的源代码位置范围（如果存在）
                stack_a = node_a.sourceRange() if node_a else None
                if stack_a:
                    source_printout.extend(
                        ["Former source location:", self._indent(str(stack_a))]
                    )
                # 获取第二个节点的源代码位置范围（如果存在）
                stack_b = node_b.sourceRange() if node_b else None
                if stack_b:
                    source_printout.extend(
                        ["Latter source location:", self._indent(str(stack_b))]
                    )

                # 将节点差异报告添加到图形差异报告中
                graph_diff_report.extend(source_printout)

                break

        # 将所有差异报告组合成一个字符串，并返回
        return "\n".join(graph_diff_report)
# 使用 @_beartype 装饰器，确保函数输入参数类型检查
@_beartype.beartype
# 检查通过 model_to_graph_func 转换的图在 test_input_groups 中是否相同
def _check_graph_diff(
    model: Union[torch.nn.Module, torch.jit.ScriptModule],
    test_input_groups: Sequence[Tuple[Tuple[Any, ...], Mapping[str, Any]]],
    export_options: _experimental.ExportOptions,
    model_to_graph_func: Callable[
        [
            torch.nn.Module,
            Tuple[Any, ...],
            Mapping[str, Any],
            _experimental.ExportOptions,
        ],
        _C.Graph,
    ],
) -> str:
    """Check if graph produced by `model_to_graph_func` is the same across `test_input_groups`.

    Args:
        model: See :func:`check_export_model_diff`.
        test_input_groups: See :func:`check_export_model_diff`.
        export_options: See :func:`check_export_model_diff`.
        model_to_graph_func: A function to convert a PyTorch model to a JIT IR graph.

    Returns:
        graph_diff_report (str): A string representation of the graph difference.
    """
    # 至少需要两组测试输入来进行比较
    if len(test_input_groups) < 2:
        raise ValueError("Need at least two groups of test inputs to compare.")

    ref_jit_graph = None
    # 遍历 test_input_groups 中的每组输入
    for args, kwargs in test_input_groups:
        # 使用 model_to_graph_func 将模型转换为 JIT IR 图
        jit_graph = model_to_graph_func(model, args, kwargs, export_options)
        # 如果是第一次循环，设置 ref_jit_graph 为当前 jit_graph
        if ref_jit_graph is None:
            ref_jit_graph = jit_graph
            continue

        # 比较当前 jit_graph 和参考的 ref_jit_graph 的差异报告
        graph_diff_report = _GraphDiff(ref_jit_graph, jit_graph).diff_report()
        # 如果存在差异报告，则返回报告字符串
        if graph_diff_report:
            return graph_diff_report
    # 如果没有差异报告，则返回空字符串
    return ""


# 使用 @_beartype 装饰器，确保函数输入参数类型检查
@_beartype.beartype
# 作为 ONNX 导出步骤的一部分，从 PyTorch 模型创建一个追踪的 JIT 图
def _traced_graph_from_model(
    model: Union[torch.nn.Module, torch.jit.ScriptModule],
    args: Tuple[Any, ...],
    kwargs: Mapping[str, Any],
    export_options: _experimental.ExportOptions,
) -> _C.Graph:
    """As part of the ONNX export steps, create a traced JIT graph from a PyTorch model.

    Args:
        model: See :func:`check_export_model_diff`.
        args: See :func:`check_export_model_diff`.
        kwargs: See :func:`check_export_model_diff`.
        export_options: See :func:`check_export_model_diff`.

    Returns:
        jit_graph (_C.Graph): A traced JIT graph.
    """
    # 获取导出选项中的训练和详细信息设置
    training = export_options.training
    verbose = export_options.verbose

    # 使用 utils.exporter_context 上下文，导出模型
    with utils.exporter_context(model, training, verbose):
        # 为导出准备输入数据
        export_inputs = _prepare_input_for_export(args, kwargs)
        # 对导出输入进行预追踪量化模型处理
        model = utils._pre_trace_quant_model(model, export_inputs)
        # 创建 JIT 图，获取结果 jit_graph 和其他信息
        jit_graph, _, _, _ = utils._create_jit_graph(model, export_inputs)
        # 返回创建的追踪 JIT 图
        return jit_graph


# 使用 @_beartype 装饰器，确保函数输入参数类型检查
@_beartype.beartype
# 作为 ONNX 导出步骤的一部分，从 PyTorch 模型导出一个 ONNX JIT 图
def _onnx_graph_from_model(
    model: Union[torch.nn.Module, torch.jit.ScriptModule],
    args: Tuple[Any, ...],
    kwargs: Mapping[str, Any],
    export_options: _experimental.ExportOptions,
) -> _C.Graph:
    """As part of the ONNX export steps, export an ONNX JIT graph from a PyTorch model.

    Args:
        model: See :func:`check_export_model_diff`.
        args: See :func:`check_export_model_diff`.
        kwargs: See :func:`check_export_model_diff`.
        export_options: See :func:`check_export_model_diff`.
    """
    # 返回 ONNX JIT 图的创建过程

        返回:
            jit_graph (_C.Graph): 导出的 ONNX JIT 图。
    # 获取导出选项中的操作集版本号
    opset_version = export_options.opset_version
    # 获取导出选项中的操作符导出类型
    operator_export_type = export_options.operator_export_type
    # 获取导出选项中的将模块导出为函数的设置
    export_modules_as_functions = export_options.export_modules_as_functions
    # 获取导出选项中的训练标志
    training = export_options.training
    # 获取导出选项中的详细输出标志
    verbose = export_options.verbose
    # 获取导出选项中的动态轴设置
    dynamic_axes = export_options.dynamic_axes
    # 获取导出选项中的输入名称列表
    input_names = export_options.input_names
    # 获取导出选项中的输出名称列表
    output_names = export_options.output_names

    # 如果操作集版本号未指定，则使用默认的 ONNX 操作集版本号
    if opset_version is None:
        opset_version = _constants.ONNX_DEFAULT_OPSET

    # 设置跟踪模块映射，用于导出过程中的使用
    utils._setup_trace_module_map(model, export_modules_as_functions)

    # 如果未指定操作符导出类型，则默认为 ONNX
    if not operator_export_type:
        operator_export_type = _C_onnx.OperatorExportTypes.ONNX

    # 设置全局变量中的导出选项
    GLOBALS.export_onnx_opset_version = opset_version
    GLOBALS.operator_export_type = operator_export_type

    # 使用上下文管理器导出模块，在导出过程中进行环境设置
    with utils.exporter_context(model, training, verbose):
        # 决定是否进行常量折叠
        do_constant_folding = utils._decide_constant_folding(
            export_options.do_constant_folding, operator_export_type, training
        )

        # 如果动态轴未指定，则设为空字典
        if dynamic_axes is None:
            dynamic_axes = {}
        # 验证动态轴设置的有效性
        utils._validate_dynamic_axes(dynamic_axes, model, input_names, output_names)

        # 准备导出的输入数据格式
        export_inputs = _prepare_input_for_export(args, kwargs)
        export_inputs = utils._decide_input_format(model, export_inputs)
        # 将模型转换为 ONNX 图
        onnx_graph, _, _ = utils._model_to_graph(
            model,
            export_inputs,
            verbose,
            input_names,
            output_names,
            operator_export_type,
            do_constant_folding,
            training=training,
            dynamic_axes=dynamic_axes,
        )

        # 返回生成的 ONNX 图
        return onnx_graph
# 使用装饰器 beartype 对函数进行类型检查和注解
@_beartype.beartype
# 从 torch.Graph 对象和导出选项中构建 ONNX 图和参数字典
def _onnx_graph_from_aten_graph(
    graph: torch.Graph,
    export_options: _experimental.ExportOptions,
    params_dict: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Graph, Dict[str, Any]]:
    # 如果未提供参数字典，则初始化为空字典
    if params_dict is None:
        params_dict = {}
    
    # 获取导出选项中的操作符导出类型、动态轴、输入名称和训练标志
    operator_export_type = export_options.operator_export_type
    dynamic_axes = export_options.dynamic_axes or {}
    input_names = export_options.input_names
    training = export_options.training
    do_constant_folding = export_options.do_constant_folding
    opset_version = export_options.opset_version or _constants.ONNX_DEFAULT_OPSET

    # 设置全局变量中的 ONNX 导出选项
    GLOBALS.export_onnx_opset_version = opset_version
    GLOBALS.operator_export_type = operator_export_type

    # 决定是否进行常数折叠
    do_constant_folding = utils._decide_constant_folding(
        do_constant_folding, operator_export_type, training
    )

    # 复制输入的图对象，以免修改原始图
    graph = graph.copy()

    # 优化图结构，可能包括参数字典、动态轴和输入名称
    graph = utils._optimize_graph(
        graph,
        operator_export_type,
        params_dict=params_dict,
        dynamic_axes=dynamic_axes,
        input_names=input_names,
    )

    # 如果训练标志为空或者为 EVAL 模式，则应用 ONNX EVAL 的微观视孔优化
    if training is None or training == _C_onnx.TrainingMode.EVAL:
        params_dict = torch._C._jit_pass_onnx_eval_peephole(graph, params_dict)

    # 如果开启了常数折叠且 ONNX 版本大于等于指定最小版本，则进行常数折叠操作
    if (
        do_constant_folding
        and opset_version >= _constants.ONNX_CONSTANT_FOLDING_MIN_OPSET
    ):
        params_dict = _C._jit_pass_onnx_constant_fold(graph, params_dict, opset_version)
        # 删除具有副作用节点的优化
        _C._jit_pass_dce_allow_deleting_nodes_with_side_effects(graph)

    # 如果开启了 ONNX 形状推断，则进行形状和类型推断
    if GLOBALS.onnx_shape_inference:
        _C._jit_pass_onnx_graph_shape_type_inference(graph, params_dict, opset_version)

    # 删除图中未使用的项目
    params_dict = _C._jit_pass_onnx_eliminate_unused_items(graph, params_dict)

    # 对于较低版本的 ONNX，将所有常数转换为浮点数类型
    if opset_version < 9:
        _C._jit_pass_onnx_cast_all_constant_to_floating(graph)

    # 过滤掉非张量参数
    params_dict = _C._jit_pass_filter_non_tensor_arguments(params_dict)

    # 对图中的打包参数输入类型进行衰减
    _C._jit_decay_packed_param_input_types(graph)

    # 删除具有副作用节点的优化
    _C._jit_pass_dce_allow_deleting_nodes_with_side_effects(graph)

    # 如果导出选项中设置了 verbose 标志，则打印 ONNX 图信息
    if export_options.verbose:
        print("ONNX graph: ", graph)

    # 返回最终的 ONNX 图和参数字典
    return graph, params_dict


# 使用装饰器 beartype 对函数进行类型检查和注解
@_beartype.beartype
# 从 ONNX 图和导出选项中构建 ONNX 协议和参数字典
def _onnx_proto_from_onnx_graph(
    onnx_graph: torch.Graph,
    export_options: _experimental.ExportOptions,
    params_dict: Dict[str, Any],
) -> Tuple[bytes, Mapping[str, bytes]]:
    # 获取导出选项中的 ONNX 版本、动态轴和操作符导出类型
    opset_version = export_options.opset_version or _constants.ONNX_DEFAULT_OPSET
    dynamic_axes = export_options.dynamic_axes or {}
    operator_export_type = export_options.operator_export_type

    # 决定是否保持初始化器作为输入
    val_keep_init_as_ip = utils._decide_keep_init_as_input(
        export_options.keep_initializers_as_inputs,
        operator_export_type,
        opset_version,
    )
    # 调用 utils 模块中的 _decide_add_node_names 函数，确定是否添加节点名称，并将结果赋给 val_add_node_names
    val_add_node_names = utils._decide_add_node_names(True, operator_export_type)
    # 获取导出选项中的自定义操作集，如果不存在则设为空字典
    custom_opsets = export_options.custom_opsets or {}

    # 调用 onnx_graph 对象的 _export_onnx 方法导出 ONNX 模型
    # 参数分别为:
    # - params_dict: 参数字典
    # - opset_version: ONNX 操作集的版本号
    # - dynamic_axes: 动态轴
    # - False: 是否包含初始化为输入的保持节点
    # - operator_export_type: 操作导出类型
    # - not export_options.verbose: 是否非详细输出
    # - val_keep_init_as_ip: 是否保持初始化为输入
    # - custom_opsets: 自定义操作集
    # - val_add_node_names: 是否添加节点名称
    # - "": 导出路径（空字符串表示内存中导出）
    # - {}: 扩展名映射
    proto, export_map, _, _ = onnx_graph._export_onnx(  # type: ignore[attr-defined]
        params_dict,
        opset_version,
        dynamic_axes,
        False,
        operator_export_type,
        not export_options.verbose,
        val_keep_init_as_ip,
        custom_opsets,
        val_add_node_names,
        "",
        {},
    )

    # 返回导出的 ONNX 模型 proto 和导出映射 export_map
    return proto, export_map
# 使用装饰器 @_beartype.beartype 对函数进行类型检查和装饰
@_beartype.beartype
# 函数定义，验证导出模型在不同输入组之间的差异
def check_export_model_diff(
    model: Union[torch.nn.Module, torch.jit.ScriptModule],  # 参数：PyTorch 模型，可以是普通模型或脚本模型
    test_input_groups: Sequence[Tuple[Tuple[Any, ...], Mapping[str, Any]]],  # 参数：用于导出模型的输入组序列，每组是(args, kwargs)的元组
    export_options: Optional[_experimental.ExportOptions] = None,  # 参数：导出选项，控制导出行为的可选对象
) -> str:  # 返回类型：返回包含导出模型差异的字符串

    """Verify exported model discrepancy between different groups of inputs.

    A graph is exported for each group of inputs. The exported graphs are then compared
    to each other, and discrepancies of first pair of nodes are reported. This function
    first checks the jit graph. If no discrepancies were found, it then checks the onnx
    graph.

    Unless otherwise specified, the jit/ONNX graph is expected to be the same, regardless
    of the inputs used for exporting. A discrepancy implies the graph exported is
    not accurate when run on other groups of inputs, which will typically results in
    runtime errors or mismatching output.
    """

    # 如果未提供导出选项，则创建一个默认的 _experimental.ExportOptions 对象
    export_options = (
        _experimental.ExportOptions() if export_options is None else export_options
    )

    # 检查 jit 图的差异并生成报告
    jit_diff_report = _check_graph_diff(
        model, test_input_groups, export_options, _traced_graph_from_model
    )
    # 如果发现了差异，则返回 JIT 图的差异报告
    if jit_diff_report:
        return jit_diff_report

    # 否则检查 ONNX 图的差异并返回报告
    return _check_graph_diff(
        model, test_input_groups, export_options, _onnx_graph_from_model
    )


# 使用装饰器 @_beartype.beartype 对函数进行类型检查和装饰
@_beartype.beartype
# 函数定义，验证模型导出到 ONNX 格式是否正确
def verify(
    model: _ModelType,  # 参数：原始 PyTorch 模型
    input_args: _InputArgsType,  # 参数：输入参数
    input_kwargs: Optional[_InputKwargsType] = None,  # 参数：输入关键字参数的可选项
    do_constant_folding: bool = True,  # 参数：是否进行常量折叠，默认为 True
    dynamic_axes: Optional[
        Mapping[str, Union[Mapping[int, str], Mapping[str, Sequence[int]]]]
    ] = None,  # 参数：动态轴的映射，指定 ONNX 导出中的动态维度
    input_names: Optional[Sequence[str]] = None,  # 参数：输入的名称
    output_names: Optional[Sequence[str]] = None,  # 参数：输出的名称
    training: _C_onnx.TrainingMode = _C_onnx.TrainingMode.EVAL,  # 参数：ONNX 导出的训练模式
    opset_version: Optional[int] = None,  # 参数：ONNX 操作集的版本号
    keep_initializers_as_inputs: bool = True,  # 参数：是否将初始化器保留为输入
    verbose: bool = False,  # 参数：是否输出详细信息
    fixed_batch_size: bool = False,  # 参数：是否使用固定的批次大小
    use_external_data: bool = False,  # 参数：是否使用外部数据
    additional_test_inputs: Optional[Sequence[_InputArgsType]] = None,  # 参数：额外的测试输入
    options: Optional[VerificationOptions] = None,  # 参数：验证选项
):
    """Verify model export to ONNX against original PyTorch model.
    """
    Args:
        model (torch.nn.Module or torch.jit.ScriptModule): 用于导出为ONNX格式的PyTorch模型。
            参见 :func:`torch.onnx.export`。
        input_args (tuple): 传递给模型的输入参数。
            参见 :func:`torch.onnx.export`。
        input_kwargs (dict): 传递给模型的输入关键字参数。
            参见 :func:`torch.onnx.export`。
        do_constant_folding (bool, optional): 是否进行常量折叠。
            参见 :func:`torch.onnx.export`。
        dynamic_axes (dict, optional): 动态轴的映射关系。
            参见 :func:`torch.onnx.export`。
        input_names (list, optional): 输入名称列表。
            参见 :func:`torch.onnx.export`。
        output_names (list, optional): 输出名称列表。
            参见 :func:`torch.onnx.export`。
        training (torch.onnx.TrainingMode): 模型的训练模式。
            参见 :func:`torch.onnx.export`。
        opset_version (int, optional): 使用的ONNX运算集的版本。
            参见 :func:`torch.onnx.export`。
        keep_initializers_as_inputs (bool, optional): 是否将初始化参数保留为输入。
            参见 :func:`torch.onnx.export`。
        verbose (bool, optional): 是否输出详细信息。
            参见 :func:`torch.onnx.export`。
        fixed_batch_size (bool, optional): 是否固定批处理大小，仅用于RNN测试用例。
        use_external_data (bool, optional): 明确指定是否使用外部数据导出模型。
        additional_test_inputs (list, optional): 测试输入参数的列表。
            每个元组都是一组测试输入参数。
        options (_VerificationOptions, optional): 控制验证行为的选项对象。

    Raises:
        AssertionError: 如果ONNX模型的输出与PyTorch模型的输出在指定的精度上不相等。
        ValueError: 如果提供的参数无效。
    """
    if options is None:
        options = VerificationOptions()

    if training == torch.onnx.TrainingMode.TRAINING:
        # 如果处于训练模式，设置模型为训练状态
        model.train()
    elif training == torch.onnx.TrainingMode.EVAL:
        # 如果处于评估模式，设置模型为评估状态
        model.eval()
    
    # 使用torch.no_grad()上下文管理器，确保不计算梯度，并使用contextlib.ExitStack()管理资源
    with torch.no_grad(), contextlib.ExitStack() as stack:
        model_f: Union[str, io.BytesIO] = io.BytesIO()
        
        # 如果需要使用外部数据，则创建临时目录并将ONNX模型写入其中
        if use_external_data:
            tmpdir_path = stack.enter_context(tempfile.TemporaryDirectory())
            model_f = os.path.join(tmpdir_path, "model.onnx")

        # 准备输入以进行模型导出
        inputs_for_export = _prepare_input_for_export(input_args, input_kwargs)

        # 克隆模型以避免改变原始模型
        model_copy = _try_clone_model(model)

        # 调用utils._export函数将模型导出为ONNX格式
        utils._export(
            model,
            inputs_for_export,
            model_f,
            opset_version=opset_version,
            do_constant_folding=do_constant_folding,
            keep_initializers_as_inputs=keep_initializers_as_inputs,
            dynamic_axes=dynamic_axes,
            input_names=input_names,
            output_names=output_names,
            fixed_batch_size=fixed_batch_size,
            training=training,
            verbose=verbose,
        )

        # 比较ONNX模型和PyTorch模型的输出
        _compare_onnx_pytorch_model(
            pt_model=model_copy,
            onnx_model_f=model_f,
            input_args=input_args,
            input_kwargs=input_kwargs,
            additional_test_inputs=additional_test_inputs,
            options=options,
        )
# 使用 @_beartype.beartype 装饰器来验证输入参数的类型和合法性
@_beartype.beartype
# 验证 ATen 图的正确性，并返回包含断言错误对象、torch 图、两种输出类型的元组
def verify_aten_graph(
    graph: torch.Graph,  # 输入参数：torch 图对象
    input_args: Tuple[Any, ...],  # 输入参数：任意类型的元组，用于输入图的参数
    export_options: _experimental.ExportOptions,  # 输入参数：导出选项对象
    params_dict: Optional[Dict[str, Any]] = None,  # 可选输入参数：字符串到任意类型值的字典
    verification_options: Optional[VerificationOptions] = None,  # 可选输入参数：验证选项对象
) -> Tuple[Optional[AssertionError], torch.Graph, _OutputsType, _OutputsType]:  # 返回类型：包含断言错误对象、torch 图、两种输出类型的元组
    # 如果验证选项未提供，使用默认的验证选项对象
    if verification_options is None:
        verification_options = VerificationOptions()
    # 如果参数字典未提供，使用空字典
    if params_dict is None:
        params_dict = {}

    # 保存原始的 JIT 图对象，并复制输入的图对象以防止修改
    original_jit_graph = graph
    graph = graph.copy()

    # 执行 ATen 图并获取参考的 torch JIT 输出
    graph_inputs = list(graph.inputs())  # 获取图的输入节点列表
    jit_inputs = tuple([arg for arg in input_args if arg is not None])  # 过滤掉输入参数中为 None 的部分并转为元组
    # 获取图中权重节点的值，并确保所有权重都已定义
    weights = [params_dict[v.debugName()] for v in graph_inputs[len(jit_inputs):]]
    assert all(w is not None for w in weights)  # 断言所有权重值均不为 None
    # TODO: 只有在检测到图中有变异时才复制参数
    jit_inputs = copy.deepcopy(jit_inputs)  # 深度复制 JIT 输入参数
    jit_input_and_parameters = jit_inputs + tuple(weights)  # 将 JIT 输入参数和权重合并为一个元组
    # 使用 torch._C._jit_interpret_graph 执行 JIT 图的解释
    jit_outs = torch._C._jit_interpret_graph(graph, jit_input_and_parameters)  # type: ignore[attr-defined]
    if not isinstance(jit_outs, (list, tuple)):
        jit_outs = [jit_outs]

    # 将 ATen 图转换为 ONNX 图
    graph, onnx_params_dict = _onnx_graph_from_aten_graph(
        graph, export_options, params_dict
    )

    # 从 ONNX 图中生成 ONNX 协议和导出映射
    proto, export_map = _onnx_proto_from_onnx_graph(
        graph, export_options, onnx_params_dict
    )
    # 创建一个字节流对象以保存 ONNX 模型
    model_f: Union[str, io.BytesIO] = io.BytesIO()
    export_type = _exporter_states.ExportTypes.PROTOBUF_FILE
    # 使用 onnx_proto_utils._export_file 导出 ONNX 协议到字节流中，并指定导出类型和映射
    onnx_proto_utils._export_file(proto, model_f, export_type, export_map)

    # NOTE: 验证不稳定，使用 try-catch 块来捕获调试信息
    try:
        # NOTE: 输入可能被 DCE（Dead Code Elimination）删除，因此需要从输入参数中删除这些部分
        new_input_names = {v.debugName() for v in graph.inputs()}  # 获取图的新输入节点名称集合
        new_input_args = []
        # 遍历原始 JIT 图的输入节点和输入参数，仅保留存在于新输入名称集合中的参数
        for v, arg in zip(original_jit_graph.inputs(), input_args):
            if v.debugName() in new_input_names:
                new_input_args.append(arg)
        input_args = tuple(new_input_args)  # 更新输入参数为新的有效输入参数集合

        # 为 ONNX 准备输入数据
        onnx_inputs = _prepare_input_for_onnx(
            input_args,
            {},
            verification_options.remained_onnx_input_idx,
            verification_options.flatten,
        )

        # 创建 ONNX 后端会话对象，并使用指定的后端运行 ONNX 模型
        onnx_session = _onnx_backend_session(model_f, verification_options.backend)
        onnx_outs = _run_onnx(onnx_session, onnx_inputs)
        del onnx_session  # 释放设备内存

        # 比较 ONNX 和 PyTorch 的输出结果
        try:
            _compare_onnx_pytorch_outputs(
                onnx_outs=onnx_outs,
                pt_outs=jit_outs,
                options=verification_options,
            )
        except AssertionError as e:
            return e, graph, jit_outs, onnx_outs  # 如果断言失败，则返回 AssertionError 和相关对象

        return None, graph, jit_outs, onnx_outs  # 返回 None 表示验证通过，同时返回图和输出结果

    except Exception as ex:
        # 捕获所有异常并返回，以便进行调试
        raise ex
    # 捕获所有异常并处理
    except Exception as e:
        # 打印出错信息
        print("Unexpected error during verification.")
        # 打印原始的 JIT 图形信息
        print("jit graph: ", original_jit_graph)
        # 打印转换后的 ONNX 图形信息
        print("onnx graph: ", graph)
        # 抛出捕获到的异常对象 e，以便于调试和进一步处理
        raise e
class GraphInfoPrettyPrinter:
    # 图信息对象，可选
    graph_info: Optional[GraphInfo]
    # 上层打印器，可选
    upper_printer: Optional[GraphInfoPrettyPrinter]
    # 下层打印器，可选
    lower_printer: Optional[GraphInfoPrettyPrinter]

    # 图字符串的 Lambda 映射，整数到字符串
    graph_str_lambdas: Mapping[int, str]
    # 连接器字符串的 Lambda 映射，整数到字符串
    connector_str_lambdas: Mapping[int, str]
    # 子图字符串的 Lambda 映射，整数到字符串
    children_str_lambdas: Mapping[int, str]

    # 初始化方法，接受图信息对象作为参数
    def __init__(self, graph_info: Optional[GraphInfo]):
        self.graph_info = graph_info
        # 如果图信息不为 None，且上层和下层图信息均存在
        if (
            graph_info is not None
            and graph_info.upper_graph_info is not None
            and graph_info.lower_graph_info is not None
        ):
            # 创建上层和下层的打印器对象
            self.upper_printer = GraphInfoPrettyPrinter(graph_info.upper_graph_info)
            self.lower_printer = GraphInfoPrettyPrinter(graph_info.lower_graph_info)
        else:
            self.upper_printer = None
            self.lower_printer = None

    # 计算总行数的方法
    @_beartype.beartype
    def _total_rows(self) -> int:
        # 如果图信息为 None，返回 1
        if self.graph_info is None:
            return 1
        # 如果存在上层和下层打印器，返回其行数总和加 1
        if self.upper_printer and self.lower_printer:
            return (
                self.upper_printer._total_rows() + self.lower_printer._total_rows() + 1
            )
        # 否则返回 2，表示两行：节点数 + ID
        return 2  # Two lines: node count + id.

    # 返回节点计数段的字符串表示方法
    @_beartype.beartype
    def _node_count_segment_str(self) -> str:
        # 如果图信息为 None，返回省略符号
        if self.graph_info is None:
            return "..."
        # 获取必要节点数和是否存在不匹配的标志
        node_count = self.graph_info.essential_node_count()
        has_mismatch = self.graph_info.has_mismatch()
        # 如果节点数为 1 且存在不匹配，则加上错误节点种类的括号表示
        error_node_kind = (
            f"({self.graph_info.essential_node_kinds().pop()})"
            if node_count == 1 and has_mismatch
            else ""
        )
        # 返回节点计数、是否匹配和错误节点种类的字符串表示
        return f"{node_count} {'X' if has_mismatch else chr(0x2713)} {error_node_kind}"

    # 返回图ID段的字符串表示方法
    @_beartype.beartype
    def _graph_id_segment_str(self) -> str:
        # 如果图信息为 None，返回空字符串
        if self.graph_info is None:
            return ""
        # 返回包含ID的字符串表示
        return f"id: {self.graph_info.id}"

    # 返回最大段落列数的方法
    @_beartype.beartype
    def _max_segment_columns(self) -> int:
        # 返回节点计数段和图ID段中最长的字符串长度
        return max(
            map(len, (self._node_count_segment_str(), self._graph_id_segment_str()))
        )

    # 返回指定行数的图段字符串表示方法
    @_beartype.beartype
    def _graph_segment_str_at_line(self, line: int) -> str:
        """Get the string representation of the graph segment at the given line."""
        # 如果行数为 0，返回节点计数段的字符串表示
        if line == 0:
            result_str = self._node_count_segment_str()
            result_str += " " * (self._max_segment_columns() - len(result_str))
            return result_str
        # 如果行数为 1，返回图ID段的字符串表示
        if line == 1:
            result_str = self._graph_id_segment_str()
            result_str += " " * (self._max_segment_columns() - len(result_str))
            return result_str
        # 如果行数在 0 到总行数之间，返回空白字符串，长度为最大段落列数
        if 0 <= line < self._total_rows():
            return " " * self._max_segment_columns()
        # 否则返回空字符串
        return ""
    # 获取给定行数的连接器段字符串
    def _connector_segment_str_at_line(self, line: int) -> str:
        """Get the connector segment string at the given line."""
        # 如果没有上方打印器和下方打印器，则返回空字符串
        if self.upper_printer is None and self.lower_printer is None:
            return ""
        # 获取上方打印器总行数，如果不存在则默认为1行
        upper_total_rows = self.upper_printer._total_rows() if self.upper_printer else 1
        # 获取下方打印器总行数，如果不存在则默认为1行
        lower_total_rows = self.lower_printer._total_rows() if self.lower_printer else 1
        # 根据行数返回对应的连接器段字符串
        if line == 0:
            return "  __"
        elif line < upper_total_rows + 1:
            return " |  "
        elif line == upper_total_rows + 1:
            return " |__"
        elif line < upper_total_rows + lower_total_rows + 1:
            return "    "
        # 如果行数超出已知范围，则返回空字符串
        return ""

    # 使用装饰器进行类型检查，获取给定行数的子节点字符串表示
    @_beartype.beartype
    def _children_str_at_line(self, line: int) -> str:
        """Get the string representation of the children at the given line.

        Recursively calls `_str_at_line` on children nodes.
        """
        # 如果没有上方打印器和下方打印器，则返回空字符串
        if self.upper_printer is None and self.lower_printer is None:
            return ""
        # 获取上方打印器总行数，如果不存在则默认为1行
        upper_total_rows = self.upper_printer._total_rows() if self.upper_printer else 1
        # 获取下方打印器总行数，如果不存在则默认为1行
        lower_total_rows = self.lower_printer._total_rows() if self.lower_printer else 1
        # 根据行数返回对应的子节点字符串表示
        if 0 <= line < upper_total_rows:
            return (
                self.upper_printer._str_at_line(line) if self.upper_printer else "..."
            )
        elif upper_total_rows < line < upper_total_rows + lower_total_rows + 1:
            return (
                self.lower_printer._str_at_line(line - upper_total_rows - 1)
                if self.lower_printer
                else "..."
            )
        # 如果行数超出已知范围，则返回空字符串
        return ""

    # 获取给定行数的图形表示字符串
    @_beartype.beartype
    def _str_at_line(self, line: int) -> str:
        """Get the string representation of the graph at the given line."""
        # 返回连接器段、图形段和子节点段组合而成的字符串
        return (
            self._graph_segment_str_at_line(line)
            + self._connector_segment_str_at_line(line)
            + self._children_str_at_line(line)
        )
    # 定义一个方法用于美化打印图形信息
    def pretty_print(self):
        # 如果图形信息为空，打印 None 并返回
        if self.graph_info is None:
            print(None)
            return
        
        # 打印树形结构
        print(" Tree: ".center(80, "="))
        # 计算总行数
        total_rows = self._total_rows()
        # 遍历每一行
        for line in range(total_rows):
            # 打印指定行的字符串表示，并去除右侧的空白字符
            print(self._str_at_line(line).rstrip())
        
        # 如果图形信息中存在不匹配的情况
        if self.graph_info.has_mismatch():
            # 打印不匹配的叶子子图的摘要信息
            print(" Mismatch leaf subgraphs: ".center(80, "="))
            print(
                [
                    graph_info.id
                    for graph_info in self.graph_info.all_mismatch_leaf_graph_info()
                ]
            )
            
            # 统计具有不匹配的节点类型
            mismatch_node_kinds: Dict[str, int] = {}
            for graph_info in self.graph_info.all_mismatch_leaf_graph_info():
                node_kinds = graph_info.essential_node_kinds()
                # 如果节点类型只有一个，将其统计到不匹配节点类型的字典中
                if len(node_kinds) == 1:
                    node_kind = node_kinds.pop()
                    mismatch_node_kinds[node_kind] = (
                        mismatch_node_kinds.get(node_kind, 0) + 1
                    )
            
            # 打印不匹配节点类型的摘要信息
            print(" Mismatch node kinds: ".center(80, "="))
            print(mismatch_node_kinds)
        else:
            # 如果没有找到不匹配的情况，打印相应信息
            print(" No mismatch found. ".center(80, "="))
class OnnxTestCaseRepro:
    def __init__(self, repro_dir):
        # 初始化方法，保存重现目录路径
        self.repro_dir = repro_dir
        # 调用 onnx_proto_utils 模块的 load_test_case 函数加载测试用例的 proto、inputs 和 outputs
        self.proto, self.inputs, self.outputs = onnx_proto_utils.load_test_case(
            repro_dir
        )

    @classmethod
    @_beartype.beartype
    def create_test_case_repro(
        cls, proto: bytes, inputs, outputs, dir: str, name: Optional[str] = None
    ):
        """Create a repro under "{dir}/test_{name}" for an ONNX test case.

        The test case contains the model and the inputs/outputs data. The directory
        structure is as follows:

        dir
        ├── test_<name>
        │   ├── model.onnx
        │   └── test_data_set_0
        │       ├── input_0.pb
        │       ├── input_1.pb
        │       ├── output_0.pb
        │       └── output_1.pb

        Args:
            proto: ONNX model proto.
            inputs: Inputs to the model.
            outputs: Outputs of the model.
            dir: Directory to save the repro.
            name: Name of the test case. If not specified, a name based on current time
                will be generated.
        Returns:
            Path to the repro.
        """
        # 如果未指定名称，使用当前时间生成名称
        if name is None:
            name = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        # 调用 onnx_proto_utils 模块的 export_as_test_case 函数导出测试用例
        return onnx_proto_utils.export_as_test_case(
            proto,
            _to_numpy(inputs),
            _to_numpy(outputs),
            name,
            dir,
        )

    @_beartype.beartype
    def validate(self, options: VerificationOptions):
        """Run the ONNX test case with options.backend, and compare with the expected outputs.

        Args:
            options: Options for validation.

        Raise:
            AssertionError: if outputs from options.backend and expected outputs are not
                equal up to specified precision.
        """
        # 将 self.proto 转换为字节流，创建 ONNX 后端会话
        onnx_session = _onnx_backend_session(io.BytesIO(self.proto), options.backend)
        # 运行 ONNX 会话，获取运行时输出
        run_outputs = onnx_session.run(None, self.inputs)
        # 获取输出的名称列表
        if hasattr(onnx_session, "get_outputs"):
            output_names = [o.name for o in onnx_session.get_outputs()]
        elif hasattr(onnx_session, "output_names"):
            output_names = onnx_session.output_names
        else:
            raise ValueError(f"Unknown onnx session type: {type(onnx_session)}")
        # 根据输出名称获取期望的输出数据
        expected_outs = [self.outputs[name] for name in output_names]
        # 使用 _compare_onnx_pytorch_outputs_in_np 函数比较运行时输出和期望输出
        _compare_onnx_pytorch_outputs_in_np(run_outputs, expected_outs, options)


@dataclasses.dataclass
class GraphInfo:
    """GraphInfo contains validation information of a TorchScript graph and its converted ONNX graph."""

    # TorchScript 图的信息：图本身、输入参数、参数字典及导出选项
    graph: torch.Graph
    input_args: Tuple[Any, ...]
    params_dict: Dict[str, Any]
    export_options: _experimental.ExportOptions = dataclasses.field(
        default_factory=_experimental.ExportOptions
    )
    # 定义一个可选的断言错误类型，用于表示不匹配的错误，初始为None
    mismatch_error: Optional[AssertionError] = dataclasses.field(
        default=None, init=False
    )
    # 定义一个可选的数值类型序列，用于存储输出结果，初始为None
    pt_outs: Optional[Sequence[_NumericType]] = dataclasses.field(
        default=None, init=False
    )
    # 定义一个可选的图信息对象，用于存储上层图信息，初始为None
    upper_graph_info: Optional[GraphInfo] = dataclasses.field(default=None, init=False)
    # 定义一个可选的图信息对象，用于存储下层图信息，初始为None
    lower_graph_info: Optional[GraphInfo] = dataclasses.field(default=None, init=False)
    # 定义一个字符串类型的属性，用于存储标识符，初始为空字符串
    id: str = dataclasses.field(default="")
    # 定义一个可选的torch.Graph对象，用于存储ONNX图的表示，初始为None
    _onnx_graph: Optional[torch.Graph] = dataclasses.field(init=False, default=None)

    # 定义一个不可变的集合，用于存储要排除的节点类型
    _EXCLUDED_NODE_KINDS: FrozenSet[str] = frozenset(
        {"prim::Constant", "prim::ListConstruct", "aten::ScalarImplicit"}
    )

    def clear(self):
        """清除之前验证的状态和结果。"""
        # 将不匹配错误重置为None
        self.mismatch_error = None
        # 将输出结果序列重置为None
        self.pt_outs = None
        # 将ONNX图对象重置为None
        self._onnx_graph = None
        # 将上层图信息对象重置为None
        self.upper_graph_info = None
        # 将下层图信息对象重置为None
        self.lower_graph_info = None

    def pretty_print_tree(self):
        """美观地打印`GraphInfo`树形结构。

        每个节点表示一个子图，显示子图中节点的数量，并在torch和ONNX之间存在输出不匹配时显示一个检查标记。

        子图的标识符显示在节点下方。可以通过调用`graph_info.find_partition(id)`来获取任何子图的`GraphInfo`对象。

        示例::

            ==================================== Tree: =====================================
            5 X   __2 X    __1 ✓
            id:  |  id: 0 |  id: 00
                 |        |
                 |        |__1 X (aten::relu)
                 |           id: 01
                 |
                 |__3 X    __1 ✓
                    id: 1 |  id: 10
                          |
                          |__2 X     __1 X (aten::relu)
                             id: 11 |  id: 110
                                    |
                                    |__1 ✓
                                       id: 111
            =========================== Mismatch leaf subgraphs: ===========================
            ['01', '110']
            ============================= Mismatch node kinds: =============================
            {'aten::relu': 2}

        """
        # 使用GraphInfoPrettyPrinter对象打印美观的树形结构
        GraphInfoPrettyPrinter(self).pretty_print()
    def pretty_print_mismatch(self, graph: bool = False):
        """Pretty print details of the mismatch between torch and ONNX.

        Args:
            graph: If True, print the ATen JIT graph and ONNX graph.
        """
        # 打印美化的标题，显示图分区的不匹配信息
        print(f" Mismatch info for graph partition {self.id}: ".center(80, "="))
        if graph:
            # 如果指定了 graph=True，打印 ATen JIT 图和 ONNX 图的信息
            print(" ATen JIT graph ".center(80, "="))
            # TODO: 更紧凑的图形打印器。
            #   * 去掉步长、梯度、设备信息。
            #   * 在单独的行上显示源位置。
            print(self.graph)
            if self._onnx_graph is not None:
                print(" ONNX graph ".center(80, "="))
                print(self._onnx_graph)
        if self.has_mismatch():
            # 如果存在不匹配，打印不匹配错误信息
            print(" Mismatch error ".center(80, "="))
            print(self.mismatch_error)
        else:
            # 如果没有不匹配，打印无不匹配信息
            print(" No mismatch ".center(80, "="))

    @_beartype.beartype
    def has_mismatch(self) -> bool:
        """Return True if the subgraph has output mismatch between torch and ONNX."""
        # 返回是否存在 torch 和 ONNX 之间输出不匹配的错误
        return self.mismatch_error is not None

    @_beartype.beartype
    def essential_node_count(self) -> int:
        """Return the number of nodes in the subgraph excluding those in `_EXCLUDED_NODE_KINDS`."""
        # 返回子图中除了 `_EXCLUDED_NODE_KINDS` 中的节点之外的节点数
        return sum(
            1 for n in self.graph.nodes() if n.kind() not in self._EXCLUDED_NODE_KINDS
        )

    @_beartype.beartype
    def essential_node_kinds(self) -> Set[str]:
        """Return the set of node kinds in the subgraph excluding those in `_EXCLUDED_NODE_KINDS`."""
        # 返回子图中除了 `_EXCLUDED_NODE_KINDS` 中的节点之外的节点种类集合
        return {
            n.kind()
            for n in self.graph.nodes()
            if n.kind() not in self._EXCLUDED_NODE_KINDS
        }

    @_beartype.beartype
    def all_mismatch_leaf_graph_info(self) -> List["GraphInfo"]:
        """Return a list of all leaf `GraphInfo` objects that have mismatch."""
        # 如果没有不匹配，返回空列表
        if not self.has_mismatch():
            return []

        # 检查是否所有的子图信息都没有不匹配
        no_mismatch_children = (
            self.upper_graph_info is None or not self.upper_graph_info.has_mismatch()
        ) and (
            self.lower_graph_info is None or not self.lower_graph_info.has_mismatch()
        )

        # 如果所有子图信息都没有不匹配，返回当前对象组成的列表
        if no_mismatch_children:
            return [self]

        # 否则，递归地获取所有有不匹配的叶子图信息对象
        results = []
        if self.upper_graph_info is not None:
            results += self.upper_graph_info.all_mismatch_leaf_graph_info()
        if self.lower_graph_info is not None:
            results += self.lower_graph_info.all_mismatch_leaf_graph_info()

        return results

    @_beartype.beartype
    # 使用给定的 id 查找对应的 GraphInfo 对象，如果当前对象的 id 符合则返回自身
    def find_partition(self, id: str) -> Optional["GraphInfo"]:
        """Find the `GraphInfo` object with the given id."""
        if id == self.id:
            return self
        
        # 计算当前对象 id 的长度
        current_length = len(self.id)
        
        # 如果给定 id 比当前 id 长度更长
        if len(id) > current_length:
            # 如果 id 在当前 id 后面的第一个字符是 "0"，且存在上层 GraphInfo 对象，则递归查找
            if id[current_length] == "0" and self.upper_graph_info is not None:
                return self.upper_graph_info.find_partition(id)
            # 如果 id 在当前 id 后面的第一个字符是 "1"，且存在下层 GraphInfo 对象，则递归查找
            elif id[current_length] == "1" and self.lower_graph_info is not None:
                return self.lower_graph_info.find_partition(id)
        
        # 如果以上条件都不满足，则返回 None
        return None

    @_beartype.beartype
    # 将子图导出到 ONNX 格式，并生成用于重现的输入输出数据
    def export_repro(
        self, repro_dir: Optional[str] = None, name: Optional[str] = None
    ) -> str:
        """Export the subgraph to ONNX along with the input/output data for repro.

        The repro directory will contain the following files::

            dir
            ├─ test_<name>
            │   ├─ model.onnx
            │   └─ test_data_set_0
            │       ├─ input_0.pb
            │       ├─ input_1.pb
            │       ├─ output_0.pb
            │       └─ output_1.pb

        Args:
            repro_dir: The directory to export the repro files to. Defaults to current
                working directory if None.
            name: An optional name for the test case folder: "test_{name}".

        Returns:
            The path to the exported repro directory.
        """
        
        # 如果 repro_dir 为 None，则使用当前工作目录
        if repro_dir is None:
            repro_dir = os.getcwd()
        
        # 拼接 repro_dir 和子目录名称 "onnx_debug"，作为最终导出的目录
        repro_dir = os.path.join(repro_dir, "onnx_debug")

        # 从 Aten 图生成 ONNX 图和参数字典
        onnx_graph, onnx_params_dict = _onnx_graph_from_aten_graph(
            self.graph, self.export_options, self.params_dict
        )

        # 从 ONNX 图生成 ONNX protobuf 格式
        proto, _ = _onnx_proto_from_onnx_graph(
            onnx_graph, self.export_options, onnx_params_dict
        )
        
        # 创建一个 ONNX 测试用例的重现，并返回导出目录的路径
        return OnnxTestCaseRepro.create_test_case_repro(
            proto, self.input_args, self.pt_outs, repro_dir, name
        )

    @_beartype.beartype
    def _graph_partition_pivot(self) -> int:
        """Find the pivot index to partition the graph.

        The pivot is the node that splits the graph into two parts. Each part should
        have the similar amount of nodes, excluding non essential ops, defined in
        `_EXCLUDED_NODE_KINDS`, such as `prim::Constant`.
        If the graph has an odd number of nodes, the upper part will have one more node.
        If the graph does not have any node that can be partitioned, return -1.

        Returns:
            The index of the pivot node.
        """
        # Collect indices of nodes that are not in `_EXCLUDED_NODE_KINDS`
        included_node_indices = [
            i
            for i, n in enumerate(self.graph.nodes())
            if n.kind() not in self._EXCLUDED_NODE_KINDS
        ]
        # Calculate the index of the pivot node (middle of the included nodes)
        half_idx = len(included_node_indices) // 2 - 1
        if half_idx >= 0 and len(included_node_indices) > half_idx:
            return included_node_indices[half_idx] + 1
        # Return -1 if no valid pivot node found
        return -1

    @_beartype.beartype
    def _partition_upper_graph(self) -> torch.Graph:
        # Determine the pivot index for partitioning the graph
        pivot = self._graph_partition_pivot()
        if pivot == -1:
            # Return an empty torch.Graph if no valid pivot found
            return torch.Graph()
        # Make a copy of the current graph to avoid mutation
        graph = self.graph.copy()

        # Capture original outputs of the graph
        original_outputs = list(graph.outputs())

        def _process_bridge_value_for_upper(
            new_outputs: List[torch.Value], bridge_value: torch.Value
        ) -> torch.Value:
            # Add bridge values as outputs to the new upper graph
            new_outputs.append(bridge_value)
            return bridge_value

        # Initialize a list to collect new outputs for the upper graph
        new_outputs: List[torch.Value] = []
        # Prepare a function to process bridge values for the upper graph
        process_bridge_value_for_upper = functools.partial(
            _process_bridge_value_for_upper, new_outputs
        )

        # Partition nodes of the graph based on the pivot index
        _, dropped_nodes, complete_upper_nodes_set, _ = self._partition_nodes(
            graph, pivot, process_bridge_value_for_upper
        )

        # Erase original outputs from the graph
        for _ in enumerate(original_outputs):
            graph.eraseOutput(0)
        
        # Register new outputs to the graph for the upper part
        for output in new_outputs:
            graph.registerOutput(output)

        # Destroy dropped nodes from the graph
        for node in reversed(dropped_nodes):
            node.destroy()

        # Erase unused inputs from the graph that do not contribute to the upper part
        for i, input in reversed(list(enumerate(list(graph.inputs())))):
            if (
                not _has_uses_by_nodes(input, complete_upper_nodes_set)
                and input not in new_outputs
            ):
                try:
                    graph.eraseInput(i)
                except RuntimeError as e:
                    print(input, graph)
                    raise e

        # Return the modified graph representing the upper partition
        return graph

    @_beartype.beartype
    # 定义一个方法用于将图分割成较低部分
    def _partition_lower_graph(self) -> torch.Graph:
        # 获取分割的枢轴节点
        pivot = self._graph_partition_pivot()
        # 如果枢轴为 -1，返回一个空的 torch.Graph 对象
        if pivot == -1:
            return torch.Graph()
        # 复制图以防止修改原始图
        graph = self.graph.copy()

        # 获取原始图的输出和输入
        original_outputs = list(graph.outputs())
        original_inputs = list(graph.inputs())

        # 用于存储新的输出节点
        new_outputs = []

        # 内部函数，用于处理较低部分的桥接值
        def _process_bridge_value_for_lower(
            graph: torch.Graph, bridge_value: torch.Value
        ) -> torch.Value:
            # 将桥接值作为较低图的输入添加
            new_input = graph.addInput()
            bridge_value.replaceAllUsesWith(new_input)
            new_input.copyMetadata(bridge_value)
            return new_input

        # 部分应用内部函数，固定图参数
        process_bridge_value_for_lower = functools.partial(
            _process_bridge_value_for_lower, graph
        )

        # 获取分割后的上部节点、下部节点、较低部分完整节点集合
        upper_nodes, lower_nodes, _, complete_lower_nodes_set = self._partition_nodes(
            graph, pivot, process_bridge_value_for_lower
        )

        # 更新图的输出节点
        for output in original_outputs:
            if _produced_by(output, lower_nodes):
                new_outputs.append(output)
        for _ in enumerate(original_outputs):
            graph.eraseOutput(0)
        for output in new_outputs:
            graph.registerOutput(output)

        # 更新图的输入节点
        for input in original_inputs:
            if _has_uses_by_nodes(input, complete_lower_nodes_set):
                new_input = graph.addInput()
                input.replaceAllUsesWith(new_input)
                new_input.copyMetadata(input)

        # 销毁不在完整下部节点集合中的上部节点
        for node in reversed(upper_nodes):
            if node not in complete_lower_nodes_set:
                try:
                    node.destroy()
                except RuntimeError as e:
                    print(node, graph)
                    raise e

        # 清除图的原始输入节点
        for _ in original_inputs:
            graph.eraseInput(0)

        # 返回分割后的图
        return graph

    # 使用 beartype 装饰器定义一个方法，用于分割节点
    @_beartype.beartype
    def _partition_node(
        self,
        node: torch.Node,
        complete_upper_nodes_set: Set[torch.Node],
        complete_lower_nodes_set: Set[torch.Node],
        original_graph_outputs: Set[torch.Value],
        covered_bridge_values: Set[torch.Value],
        process_bridge_value: Callable[[torch.Value], torch.Value],
    ):
        # 如果当前节点已经在完整的下层节点集合中，则直接返回，无需处理
        if node in complete_lower_nodes_set:
            return

        # 如果当前节点具有被下层节点使用的情况，并且节点类型在排除列表中，则将当前节点及其子节点添加到完整下层节点集合中
        if (
            _node_has_uses_by(node, complete_lower_nodes_set)
            and node.kind() in self._EXCLUDED_NODE_KINDS
        ):
            complete_lower_nodes_set.update(_all_nodes([node]))
            # 遍历当前节点的输入，处理未覆盖的桥接值
            for input in node.inputs():
                if input in covered_bridge_values:
                    continue
                # 递归分割节点，处理输入节点的完整上层和下层节点集合，原始图输出，覆盖的桥接值，桥接值处理函数
                self._partition_node(
                    input.node(),
                    complete_upper_nodes_set,
                    complete_lower_nodes_set,
                    original_graph_outputs,
                    covered_bridge_values,
                    process_bridge_value,
                )
        else:
            # 如果当前节点没有被下层节点使用或者节点类型不在排除列表中，则处理当前节点的输出
            for output in node.outputs():
                if output in covered_bridge_values:
                    continue
                # 如果输出节点被下层节点使用或者是原始图的输出之一，则将其标记为覆盖的桥接值
                if (
                    _has_uses_by_nodes(output, complete_lower_nodes_set)
                    or output in original_graph_outputs
                ):
                    covered_bridge_values.add(process_bridge_value(output))

    @_beartype.beartype
    # 分割图中节点的方法，接受图对象、分割点索引、桥接值处理函数，返回上层节点列表、下层节点列表、完整上层节点集合、完整下层节点集合
    def _partition_nodes(
        self,
        graph: torch.Graph,
        pivot: int,
        process_bridge_value: Callable[[torch.Value], torch.Value],
    ) -> Tuple[List[torch.Node], List[torch.Node], Set[torch.Node], Set[torch.Node]]:
        # 获取图中所有节点列表
        nodes = list(graph.nodes())
        # 根据分割点将节点分为上层节点和下层节点
        upper_nodes = nodes[:pivot]
        lower_nodes = nodes[pivot:]
        
        # `upper_nodes` 和 `complete_upper_nodes_set` 的区别在于后者递归包含了 `upper_nodes` 子块中的节点
        # `lower_nodes` 和 `complete_lower_nodes_set` 也是类似的关系
        # 此外，`complete_lower_nodes_set` 还包括从 `upper_nodes` 复制到 `lower_nodes` 的节点
        complete_upper_nodes_set = _all_nodes(upper_nodes)
        complete_lower_nodes_set = _all_nodes(lower_nodes)
        
        # 获取原始图的输出节点集合
        original_graph_outputs = set(graph.outputs())
        
        # 桥接值是由上层图产生并被下层图消耗的值，需要成为上层图的输出和下层图的输入，以桥接两者之间的交互
        # 从所有图输入开始，标记为已覆盖的。如果任何图输入被下层图需要，稍后保留在下层图输入中。
        covered_bridge_values = set(graph.inputs())
        
        # 对上层节点列表中的每个节点调用 _partition_node 方法进行分割处理
        for node in upper_nodes:
            self._partition_node(
                node,
                complete_upper_nodes_set,
                complete_lower_nodes_set,
                original_graph_outputs,
                covered_bridge_values,
                process_bridge_value,
            )
        
        # 返回分割结果，包括上层节点列表、下层节点列表、完整上层节点集合和完整下层节点集合
        return (
            upper_nodes,
            lower_nodes,
            complete_upper_nodes_set,
            complete_lower_nodes_set,
        )

    @_beartype.beartype
    # 获取当前对象的 pt_outs 属性，即 TorchScript IR 图的输出
    pt_outs = self.pt_outs
    # 获取当前对象关联图的输出节点列表
    graph_outputs = list(self.graph.outputs())
    # 断言 pt_outs 不为空
    assert pt_outs is not None
    # 断言图的输出节点数量与 pt_outs 数量相等，否则抛出异常并显示详细信息
    assert len(graph_outputs) == len(
        pt_outs
    ), f"{len(graph_outputs)} vs {len(pt_outs)}\nGraph: {self.graph}"
    # 构建并返回一个字典，将输出节点的调试名称映射到其对应的输出对象
    return {v.debugName(): o for v, o in zip(graph_outputs, pt_outs)}

@_beartype.beartype
def _args_and_params_for_partition_graph(
    self,
    graph: torch.Graph,
    bridge_kwargs: Mapping[str, Union[_NumericType, Sequence[_NumericType]]],
    full_kwargs: Mapping[str, torch.Tensor],
    full_params: Mapping[str, torch.Tensor],
):
    # 获取图的输入节点名称列表
    input_names = [input.debugName() for input in graph.inputs()]
    # 构建参数元组，包括所有在 bridge_kwargs 中存在的输入名称对应的值
    args = tuple(bridge_kwargs[k] for k in input_names if k in bridge_kwargs)
    # 将 full_kwargs 中存在的输入名称对应的值添加到参数元组中
    args += tuple(full_kwargs[k] for k in input_names if k in full_kwargs)
    # 构建参数字典，包括所有在 full_params 中存在的输入名称对应的值
    params = {k: full_params[k] for k in input_names if k in full_params}
    # 断言参数元组和参数字典的长度之和等于输入节点名称列表的长度，否则抛出异常并显示详细信息
    assert len(args) + len(params) == len(
        input_names
    ), f"{len(args)} + {len(params)} vs {len(input_names)}: {input_names}"
    # 返回参数元组和参数字典
    return args, params

@_beartype.beartype
def verify_export(
    self, options: VerificationOptions
) -> Tuple[Optional[AssertionError], torch.Graph, _OutputsType, _OutputsType]:
    """
    Verify the export from TorchScript IR graph to ONNX.

    Export the TorchScript IR graph to ONNX, with the inputs, parameters and export
    options recorded in this object. Then verify the exported ONNX graph against
    the original TorchScript IR graph under the provided verification options.

    Args:
        options: The verification options.

    Returns:
        error: The AssertionError raised during the verification. Returns None if no
        error is raised.
        onnx_graph: The exported ONNX graph in TorchScript IR format.
        onnx_outs: The outputs from running exported ONNX model under the onnx
        backend in `options`.
        pt_outs: The outputs from running the TorchScript IR graph.
    """
    # 调用 verify_aten_graph 函数，验证 TorchScript IR 图导出到 ONNX 的过程
    return verify_aten_graph(
        self.graph,
        input_args=self.input_args,
        params_dict=self.params_dict,
        export_options=self.export_options,
        verification_options=options,
    )

@_beartype.beartype
def find_mismatch(
    self,
    options: Optional[VerificationOptions] = None,
        """
        Find all mismatches between the TorchScript IR graph and the exported onnx model.

        Binary searches the model graph to find the minimal subgraph that exhibits the
        mismatch. A `GraphInfo` object is created for each subgraph, recording the test
        inputs and export options, as well as the validation results.

        Args:
            options: The verification options.
        """
        # 清空当前对象的状态
        self.clear()

        # 如果未提供选项，则使用默认的验证选项
        if options is None:
            options = VerificationOptions()

        # 如果导出选项中设置了详细输出，打印当前图形的信息
        if self.export_options.verbose:
            print(self.graph)

        # 如果模型输出的图形中没有任何输出节点，直接返回
        if len(list(self.graph.outputs())) == 0:
            return

        # 确保图形输入节点的数量与提供的张量参数数量相匹配
        assert len(self.input_args) + len(self.params_dict) == len(
            list(self.graph.inputs())
        ), (
            f"Number of graph inputs({len(list(self.graph.inputs()))}) does not match "
            f"the provided tensor arguments({len(self.input_args)} + {len(self.params_dict)})."
        )

        # 执行模型导出验证，并获取验证过程中的错误信息、导出的ONNX图形、PyTorch输出和分析结果
        self.mismatch_error, self._onnx_graph, self.pt_outs, _ = self.verify_export(
            options
        )

        # 如果在图形中未发现任何不匹配，直接返回
        if self.mismatch_error is None:
            # No mismatch found in graph.
            return

        # 如果基础节点数量小于等于1，已经到达叶子节点，停止进一步分割
        if self.essential_node_count() <= 1:
            # Reached leaf node, no more partitioning.
            return

        # 创建包含所有输入节点及其对应参数的字典
        full_kwargs = {
            k.debugName(): v for k, v in zip(self.graph.inputs(), self.input_args)
        }
        full_params = self.params_dict

        # 对上部分图形进行分割，并获取其输入参数
        upper_graph = self._partition_upper_graph()
        upper_args, upper_params = self._args_and_params_for_partition_graph(
            upper_graph, {}, full_kwargs, full_params
        )
        # 创建上部分图形的信息对象，并进行进一步的不匹配查找
        self.upper_graph_info = GraphInfo(
            upper_graph,
            upper_args,
            upper_params,
            self.export_options,
            id=self.id + "0",
        )
        self.upper_graph_info.find_mismatch(options)

        # 获取上部分图形分割的桥接参数
        bridge_kwargs = self.upper_graph_info._bridge_kwargs()
        # 对下部分图形进行分割，并获取其输入参数
        lower_graph = self._partition_lower_graph()
        lower_args, lower_params = self._args_and_params_for_partition_graph(
            lower_graph, bridge_kwargs, full_kwargs, full_params
        )
        # 创建下部分图形的信息对象，并进行进一步的不匹配查找
        self.lower_graph_info = GraphInfo(
            lower_graph,
            lower_args,
            lower_params,
            self.export_options,
            id=self.id + "1",
        )
        self.lower_graph_info.find_mismatch(options)
# 使用 Beartype 进行类型注解，确保 nodes 是 torch.Node 的集合，返回一个包含所有节点的集合
@_beartype.beartype
def _all_nodes(nodes: Collection[torch.Node]) -> Set[torch.Node]:
    # 初始化一个包含所有节点的集合
    all_nodes = set(nodes)
    # 遍历传入的节点集合
    for n in nodes:
        # 遍历当前节点 n 的每个块 b
        for b in n.blocks():
            # 递归更新 all_nodes，将块 b 中的所有节点添加到集合中
            all_nodes.update(_all_nodes(list(b.nodes())))
    # 返回包含所有节点的集合
    return all_nodes


# 使用 Beartype 进行类型注解，确保 value 是 torch.Value，nodes 是 torch.Node 的集合，返回布尔值
@_beartype.beartype
def _has_uses_by_nodes(value: torch.Value, nodes: Collection[torch.Node]) -> bool:
    # 检查 value 的每个使用是否在 nodes 集合中的节点中
    if any(use.user in nodes for use in value.uses()):
        return True
    return False


# 使用 Beartype 进行类型注解，确保 node 是 torch.Node，nodes 是 torch.Node 的集合，返回布尔值
@_beartype.beartype
def _node_has_uses_by(node: torch.Node, nodes: Collection[torch.Node]) -> bool:
    # 遍历节点 node 的每个输出
    for output in node.outputs():
        # 检查 output 是否被 nodes 集合中的节点使用
        if _has_uses_by_nodes(output, nodes):
            return True
    return False


# 使用 Beartype 进行类型注解，确保 value 是 torch.Value，nodes 是 torch.Node 的集合，返回布尔值
@_beartype.beartype
def _produced_by(value: torch.Value, nodes: Collection[torch.Node]) -> bool:
    # 检查 value 所在的节点是否在 nodes 集合中
    return value.node() in nodes


# 使用 Beartype 进行类型注解，定义 find_mismatch 函数，返回 GraphInfo 对象
@_beartype.beartype
def find_mismatch(
    model: Union[torch.nn.Module, torch.jit.ScriptModule],
    input_args: Tuple[Any, ...],
    do_constant_folding: bool = True,
    training: _C_onnx.TrainingMode = _C_onnx.TrainingMode.EVAL,
    opset_version: Optional[int] = None,
    keep_initializers_as_inputs: bool = True,
    verbose: bool = False,
    options: Optional[VerificationOptions] = None,
) -> GraphInfo:
    r"""Find all mismatches between the original model and the exported model.

    Experimental. The API is subject to change.

    This tool helps debug the mismatch between the original PyTorch model and exported
    ONNX model. It binary searches the model graph to find the minimal subgraph that
    exhibits the mismatch.

    Args:
        model: The model to be exported.
        input_args: The input arguments to the model.
        do_constant_folding: Same as `do_constant_folding` in :func:`torch.onnx.export`.
        training: Same as `training` in :func:`torch.onnx.export`.
        opset_version: Same as `opset_version` in :func:`torch.onnx.export`.
        keep_initializers_as_inputs: Same as `keep_initializers_as_inputs` in :func:`torch.onnx.export`.
        verbose: Same as `verbose` in :func:`torch.onnx.export`.
        options: The options for the mismatch verification.

    Returns:
        A GraphInfo object that contains the mismatch information.
    """
    """
    Example::

        >>> import torch
        >>> import torch.onnx.verification
        >>> torch.manual_seed(0)
        >>> opset_version = 15
        >>> # Define a custom symbolic function for aten::relu.
        >>> # The custom symbolic function is incorrect, which will result in mismatches.
        >>> def incorrect_relu_symbolic_function(g, self):
        ...     return self
        >>> # Register the custom symbolic function for the 'aten::relu' operator with specified opset_version.
        >>> torch.onnx.register_custom_op_symbolic(
        ...     "aten::relu",
        ...     incorrect_relu_symbolic_function,
        ...     opset_version=opset_version,
        ... )
        >>> class Model(torch.nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         # Define layers for the model with sequential composition.
        ...         self.layers = torch.nn.Sequential(
        ...             torch.nn.Linear(3, 4),
        ...             torch.nn.ReLU(),
        ...             torch.nn.Linear(4, 5),
        ...             torch.nn.ReLU(),
        ...             torch.nn.Linear(5, 6),
        ...         )
        ...     def forward(self, x):
        ...         # Forward pass through the defined layers.
        ...         return self.layers(x)
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX)
        >>> # Find mismatches between the PyTorch model and ONNX export using specified opset_version.
        >>> graph_info = torch.onnx.verification.find_mismatch(
        ...     Model(),
        ...     (torch.randn(2, 3),),
        ...     opset_version=opset_version,
        ... )
        ===================== Mismatch info for graph partition : ======================
        ================================ Mismatch error ================================
        Tensor-likes are not close!
        Mismatched elements: 12 / 12 (100.0%)
        Greatest absolute difference: 0.2328854203224182 at index (1, 2) (up to 1e-07 allowed)
        Greatest relative difference: 0.699536174352349 at index (1, 3) (up to 0.001 allowed)
        ==================================== Tree: =====================================
        5 X   __2 X    __1 ✓
        id:  |  id: 0 |  id: 00
             |        |
             |        |__1 X (aten::relu)
             |           id: 01
             |
             |__3 X    __1 ✓
                id: 1 |  id: 10
                      |
                      |__2 X     __1 X (aten::relu)
                         id: 11 |  id: 110
                                |
                                |__1 ✓
                                   id: 111
        =========================== Mismatch leaf subgraphs: ===========================
        ['01', '110']
        ============================= Mismatch node kinds: =============================
        {'aten::relu': 2}

    """
    if options is None:
        # Set default options for verification if not provided.
        options = VerificationOptions()
    if opset_version is None:
        # Set default ONNX opset version if not specified.
        opset_version = _constants.ONNX_DEFAULT_OPSET
    """From aten graph, do binary search on graph partition to find operator export discrepancy."""
    # TODO: Copied from utils.py `export` until `_optimize_graph`.
    if training == torch.onnx.TrainingMode.TRAINING:
        # Set the model in training mode if the training flag is set.
        model.train()
    # 如果训练模式是 EVAL，则设置模型为评估模式
    elif training == torch.onnx.TrainingMode.EVAL:
        model.eval()
    
    # 在无需计算梯度的上下文中执行以下操作
    with torch.no_grad():
        # 准备输入以进行导出，并确定输入格式
        inputs_for_export = _prepare_input_for_export(input_args, {})
        args = utils._decide_input_format(model, inputs_for_export)

        # 对模型进行预跟踪量化
        model = utils._pre_trace_quant_model(model, args)

        # 创建 JIT 图形、参数、Torch 输出和模块
        graph, params, torch_out, module = utils._create_jit_graph(model, args)

        # 获取图形中的命名参数字典
        params_dict = utils._get_named_param_dict(graph, params)

        # 将友好的调试名称应用到图形中的节点和参数
        utils._apply_friendly_debug_names(graph, params_dict)

        # 创建 GraphInfo 对象来包含导出所需的所有信息
        graph_info = GraphInfo(
            graph,
            input_args,
            params_dict,
            _experimental.ExportOptions(
                do_constant_folding=do_constant_folding,
                training=training,
                opset_version=opset_version,
                keep_initializers_as_inputs=keep_initializers_as_inputs,
                verbose=verbose,
            ),
        )

        # 在导出过程中查找不匹配，如果有则打印详细信息
        graph_info.find_mismatch(options)
        graph_info.pretty_print_mismatch()
        graph_info.pretty_print_tree()

        # 返回包含导出图形信息的 GraphInfo 对象
        return graph_info
```