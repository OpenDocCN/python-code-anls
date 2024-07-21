# `.\pytorch\torch\onnx\_internal\fx\op_validation.py`

```py
# mypy: allow-untyped-defs
"""Module for handling op-level validation during exporting."""

# 引入必要的模块和库
import logging
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

# 引入onnxscript模块，忽略类型检查
import onnxscript  # type: ignore[import]
from onnxscript import evaluator  # type: ignore[import]

# 引入PyTorch相关模块
import torch
import torch.fx
from torch.fx.experimental import symbolic_shapes
from torch.onnx import _constants, _type_utils as jit_type_utils
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import (
    diagnostics,
    fx_onnx_interpreter,
    type_utils as fx_type_utils,
)
from torch.utils import _pytree

# 定义一个装饰器函数，用于格式化操作级别调试信息
@_beartype.beartype
def _op_level_debug_message_formatter(
    fn: Callable,
    self,
    node: torch.fx.Node,
    symbolic_fn: Union[onnxscript.OnnxFunction, onnxscript.TracedOnnxFunction],
    *args,
    **kwargs,
) -> str:
    return (
        f"FX Node: {node.op}::{node.target}[name={node.name}]. \n"
        f"ONNX Node: {symbolic_fn.name}[opset={symbolic_fn.opset}]."
    )

# 装饰器函数，用于诊断调用，应用于验证操作之间的函数
@_beartype.beartype
@diagnostics.diagnose_call(
    diagnostics.rules.op_level_debugging,
    diagnostic_message_formatter=_op_level_debug_message_formatter,
)
def validate_op_between_ort_torch(
    diagnostic_context: diagnostics.DiagnosticContext,
    node: torch.fx.Node,
    symbolic_fn: Union[onnxscript.OnnxFunction, onnxscript.TracedOnnxFunction],
    fx_args: List[fx_type_utils.Argument],
    fx_kwargs: Dict[str, fx_type_utils.Argument],
    fx_graph_module: torch.fx.GraphModule,
):
    """Validate the op between ONNX Runtime and PyTorch.

    The function will run the op in ONNX Runtime and PyTorch and compare the
    results. It doesn't break the exporting process, but saves each op validated
    result into SARIF, under the section of `fx_onnx_interpreter`.

    There are three signs can be found:
    1. Blue: Pass
    2. Yellow: Bypass

    Args:
        node (torch.fx.Node): The validated fx.node
        symbolic_fn (Union[onnxscript.OnnxFunction, onnxscript.TracedOnnxFunction]): The corresponded ONNX node
        fx_args (List[fx_type_utils.Argument]): List of arguments for the function
        fx_kwargs (Dict[str, fx_type_utils.Argument]): Dictionary of keyword arguments for the function
        fx_graph_module (torch.fx.GraphModule): The fx.GraphModule that contains the nodes
    """
    # op-level validation
    # Symbolic_fn should have the same output as node.target (torch ops)

    try:
        # 将fx_args和fx_kwargs包装为torch的参数形式
        torch_args, torch_kwargs = _wrap_fx_args_as_torch_args(
            fx_args, fx_kwargs, fx_graph_module
        )
    except ValueError as value_error:
        # 如果出现值错误，记录警告日志
        diagnostic = diagnostic_context.inflight_diagnostic()
        with diagnostic.log_section(
            logging.WARNING, "Op level debug fails due to unsupported input types"
        ):
            diagnostic.log_source_exception(logging.WARNING, value_error)
        diagnostic.level = diagnostics.levels.WARNING
        return

@_beartype.beartype
def _convert_symint_to_int_in_shape(shape: torch.Size) -> torch.Size:
    # 函数用于将torch.Size中的symbolic integers转换为integers
    """Convert SymInt to int in shape

    Args:
        shape (torch.Size): The shape of a tensor
    Raises:
        ValueError: When SymInt is found in shape
    Returns:
        torch.Size: The shape of a tensor with SymInt converted to int

    """
    # 创建一个空列表，用于存储转换后的维度信息
    list_int_shape = []
    
    # 遍历输入的 shape 中的每一个维度
    for dim in shape:
        # 检查当前维度是否为 torch.SymInt 类型
        if isinstance(dim, torch.SymInt):
            # 如果是 SymInt 类型，检查是否存在该 SymInt 的提示信息
            if symbolic_shapes.has_hint(dim):
                # 如果存在提示信息，则使用提示信息中的整数值替换 SymInt
                list_int_shape.append(symbolic_shapes.hint_int(dim))
            else:
                # 如果不存在提示信息，则抛出 ValueError 异常
                raise ValueError(
                    f"An unbacked SymInt found in shape. SymInt: {dim}; "
                    f"torch.Size: {shape}. There is no hint for SymInt."
                )
        else:
            # 如果当前维度不是 SymInt 类型，则直接添加到列表中
            list_int_shape.append(dim)
    
    # 使用转换后的列表创建一个新的 torch.Size 对象，并返回
    return torch.Size(list_int_shape)
# 使用 @_beartype 装饰器对函数进行类型检查和验证
@_beartype.beartype
# 生成指定形状和数据类型的随机张量
def generate_random_tensors(shape: torch.Size, dtype: torch.dtype):
    # 将 shape 中的符号整数转换为整数
    shape = _convert_symint_to_int_in_shape(shape)

    # 根据数据类型不同生成不同范围的随机整数张量
    if dtype == torch.uint8:
        return torch.randint(
            low=_constants.UINT8_MIN, high=_constants.UINT8_MAX, size=shape, dtype=dtype
        )
    if dtype == torch.int8:
        return torch.randint(
            low=_constants.INT8_MIN, high=_constants.INT8_MAX, size=shape, dtype=dtype
        )
    if dtype == torch.int16:
        return torch.randint(
            low=_constants.INT16_MIN, high=_constants.INT16_MAX, size=shape, dtype=dtype
        )
    if dtype == torch.int32:
        return torch.randint(
            low=_constants.INT32_MIN, high=_constants.INT32_MAX, size=shape, dtype=dtype
        )
    if dtype == torch.int64:
        return torch.randint(
            low=_constants.INT64_MIN, high=_constants.INT64_MAX, size=shape, dtype=dtype
        )
    # 若数据类型为 bool，则生成随机数并根据条件返回 True 或 False 的张量
    if dtype == torch.bool:
        random_numbers = torch.rand(shape)
        return torch.where(
            random_numbers > 0.5, torch.tensor(True), torch.tensor(False)
        )
    # 若数据类型为复数类型，则生成对应的随机复数张量，但返回其实部的表示
    if fx_type_utils.is_torch_complex_dtype(dtype):
        # ONNX 不支持复数值，但支持其实部的表示
        return torch.view_as_complex(
            torch.randn((*shape, 2), dtype=fx_type_utils.from_complex_to_float(dtype))
        )
    # 默认情况下，生成指定形状和数据类型的随机张量
    return torch.randn(shape, dtype=dtype)


# 使用 @_beartype 装饰器对函数进行类型检查和验证
@_beartype.beartype
# 将 FX 图参数转换为 Torch 参数的递归函数
def _fx_args_to_torch_args(
    fx_args: List[fx_type_utils.Argument], fx_graph_module: torch.fx.GraphModule
) -> List[fx_type_utils.Argument]:
    """Recursively convert fx args to torch args"""
    # 包装后的参数列表
    wrapped_args: List[fx_type_utils.Argument] = []
    for arg in fx_args:
        # 检查当前参数是否为 torch.fx.Node 类型
        if isinstance(arg, torch.fx.Node):
            # 从 meta 属性中获取 "val"，这可能是一个 FakeTensor
            fake_tensor = arg.meta.get("val")
            # 如果 "val" 为空且操作是 "get_attr"，尝试从 fx_graph_module 中获取对应的属性
            if fake_tensor is None and arg.op == "get_attr":
                fake_tensor = getattr(fx_graph_module, arg.target)  # type: ignore[operator]
            
            # NOTE: 目前已知可能在 arg.meta["val"]/get_attr 中包含 FakeTensor/Tensor/SymInt/SymFloat/Symbool/int/float/bool 类型
            # 根据 fake_tensor 的类型进行不同的处理
            if isinstance(fake_tensor, torch.Tensor):
                # 生成一个与 fake_tensor 相同形状和数据类型的随机 Tensor
                real_tensor = generate_random_tensors(
                    fake_tensor.shape, fake_tensor.dtype
                )
                wrapped_args.append(real_tensor)
            elif isinstance(fake_tensor, (int, float, bool)):
                # 直接将 fake_tensor 加入 wrapped_args
                wrapped_args.append(fake_tensor)
            elif symbolic_shapes.has_hint(fake_tensor):
                # 如果 fake_tensor 是符号形状提示，则将其转换为整数并加入 wrapped_args
                wrapped_args.append(symbolic_shapes.hint_int(fake_tensor))
            else:
                # 抛出异常，因为发现了不预期的输入参数类型
                raise ValueError(
                    f"Unexpected input argument type found inside fx.Node. arg: {arg}; "
                    f"arg.meta['val']/get_attr: {fake_tensor}; type(arg.meta['val']/get_attr): "
                    f"{type(fake_tensor)}."
                )
        
        # 如果当前参数是 Sequence 类型，则递归地转换成对应的 torch 参数并加入 wrapped_args
        elif isinstance(arg, Sequence):
            wrapped_args.append(_fx_args_to_torch_args(arg, fx_graph_module))
        
        # 如果当前参数是 int、float、torch.dtype 或者为 None，则直接加入 wrapped_args
        elif isinstance(arg, (int, float, torch.dtype)) or arg is None:
            wrapped_args.append(arg)
        
        # 如果当前参数是 torch.device 类型，则将其转换为字符串并加入 wrapped_args
        elif isinstance(arg, torch.device):
            wrapped_args.append(str(arg))
        
        # 如果以上条件都不符合，则抛出异常，因为发现了不预期的输入参数类型
        else:
            raise ValueError(
                f"Unexpected input argument type is found in node arguments. arg: {arg}; "
            )

    # 返回最终处理后的 wrapped_args
    return wrapped_args
# 使用 @_beartype.beartype 装饰器对函数进行类型检查和参数验证
@_beartype.beartype
# 准备将函数的输入参数转换为适用于 Torch 操作级验证的格式
def _wrap_fx_args_as_torch_args(
    fx_args: List[fx_type_utils.Argument],  # 输入参数列表，每个参数是 fx_type_utils.Argument 类型
    fx_kwargs: Dict[str, fx_type_utils.Argument],  # 输入关键字参数字典，键为参数名，值为 fx_type_utils.Argument 类型
    fx_graph_module: torch.fx.GraphModule,  # Torch 的图模块，用于处理图相关的操作
) -> Tuple[List[fx_type_utils.Argument], Dict[str, fx_type_utils.Argument]]:
    """Prepare torch format args and kwargs for op-level validation by using fake tensor to create real tensor to feed in ops"""

    # NOTE: This function only supports FakeTensor with concrete shapes
    # 将 fx_args 转换为 Torch 格式的参数列表
    torch_args: List[fx_type_utils.Argument] = _fx_args_to_torch_args(
        fx_args, fx_graph_module
    )
    # 返回转换后的 Torch 参数列表和未变动的关键字参数字典
    return torch_args, fx_kwargs


# NOTE: Referenced from onnxscript internal function: _tag_arguments_with_param_schemas.
# 使用 @_beartype.beartype 装饰器对函数进行类型检查和参数验证
@_beartype.beartype
# 将 Torch 格式的参数转换为 OnnxFunction 可接受的参数格式
def _convert_torch_args_to_onnxfunction_args(
    param_schemas: Sequence[onnxscript.values.ParamSchema],  # ONNX 参数模式序列
    args: List[fx_type_utils.Argument],  # 调用者提供的 Python 位置参数列表
    kwargs: Dict[str, fx_type_utils.Argument],  # 调用者提供的 Python 关键字参数字典
    allow_extra_kwargs: bool = False,  # 是否允许额外的关键字参数
) -> Tuple[List[Any], Dict[str, Any],]:
    """Convert Python args and kwargs to OnnxFunction acceptable with matching ONNX ParamSchema.

    NOTE: This is different from the param_schema separating in dispatcher, since at this point
    we are already sure that the args and kwargs are in order and matched.

    Args:
        param_schemas: The parameter schemas of an Op or a OnnxFunction.
        args: The Python positional arguments supplied by the caller.
        kwargs: The Python keyword arguments supplied by the caller.
        allow_extra_kwargs: Whether to allow extra keyword arguments.
            When set to True, extra/unknown arguments will be ignored.

    Returns:
        A tuple of two elements:
        - A list of Python positional argument.
        - An ordered dictionary of Python keyword argument names and its values.

    Raises:
        TypeError: When allow_extra_kwargs is False and there are unknown kwargs.
        TypeError: When a required input is not provided.
    """
    # args, kwargs and param_schemas should be all in order
    # user may not specify all inputs or attributes

    # 获取所有参数模式的名称集合
    all_param_names = {param.name for param in param_schemas}
    # 查找额外的关键字参数
    extra_kwargs = set(kwargs).difference(all_param_names)
    # 如果存在额外的关键字参数且不允许额外参数，则抛出 TypeError 异常
    if extra_kwargs and not allow_extra_kwargs:
        raise TypeError(f"Unexpected keyword arguments '{extra_kwargs}'")

    # 初始化标记后的位置参数列表和关键字参数字典
    tagged_args: list[Any] = []
    tagged_kwargs: dict[str, Any] = {}
    # 遍历参数模式列表，并使用索引 i 和对应的参数 param 进行迭代
    for i, param in enumerate(param_schemas):
        # 如果当前参数允许接收可变数量的输入
        if param.is_variadic_input:
            # 将剩余的所有参数都添加到 tagged_args 中
            tagged_args.extend(arg for arg in args[i:])
            # 将 args 清空，因为所有参数都已经处理完毕
            args = []
            # 继续处理下一个参数模式
            continue
        
        # 如果当前索引 i 小于 args 的长度
        if i < len(args):
            # 如果参数是输入或者是 torch 数据类型
            if param.is_input or isinstance(args[i], torch.dtype):
                # 将参数转换成 numpy 格式并添加到 tagged_args 中
                tagged_args.append(_convert_tensor_to_numpy(args[i]))
            else:
                # 直接将参数添加到 tagged_args 中
                tagged_args.append(args[i])
        # 如果参数的名称在 kwargs 中
        elif param.name in kwargs:
            # 如果参数是输入类型
            if param.is_input:
                # 将 kwargs 中的参数值转换成 numpy 格式并添加到 tagged_kwargs 中
                tagged_kwargs[param.name] = _convert_tensor_to_numpy(kwargs[param.name])
            else:
                # 将 kwargs 中的参数直接添加到 tagged_kwargs 中
                tagged_kwargs[param.name] = kwargs[param.name]
        # 如果参数是必需的但没有被提供
        elif param.required:
            # 抛出类型错误异常，指示缺少必需的输入或属性
            raise TypeError(f"Required input/attribute '{param}' was not provided")

    # 返回处理后的 tagged_args 和 tagged_kwargs
    return tagged_args, tagged_kwargs
# 使用装饰器进行参数类型检查
@_beartype.beartype
# 将输入的 fx_type_utils.Argument 转换为任意类型的返回值
def _convert_tensor_to_numpy(input: fx_type_utils.Argument) -> Any:
    # 尝试导入 numpy 库，如果失败则抛出 ModuleNotFoundError 异常
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"{__name__} needs numpy, but it's not installed."
        ) from exc

    # 如果输入是 torch.Tensor 类型
    if isinstance(input, torch.Tensor):
        # 如果是复数类型的 Tensor，转换为实数表示
        if torch.is_complex(input):
            input = torch.view_as_real(input.resolve_conj())
        # 将 Tensor 转换为 numpy 数组，先分离计算图，再移动到 CPU，最后转换为 numpy 数组
        return input.detach().cpu().numpy()
    
    # 如果输入是 torch.dtype 类型
    if isinstance(input, torch.dtype):
        # 将 dtype 转换为 JitScalarType，然后获取其对应的 ONNX 类型
        return int(jit_type_utils.JitScalarType.from_dtype(input).onnx_type())  # type: ignore[union-attr]
    
    # 如果输入是 tuple 或者 list 类型
    if isinstance(input, (tuple, list)):
        # 如果为空序列，返回一个空的 numpy 数组，数据类型为 np.int64
        if len(input) == 0:
            return np.array((), dtype=np.int64)
        # 如果序列的第一个元素是 torch.Tensor 类型，递归地将每个元素转换为 numpy 数组
        if isinstance(input[0], torch.Tensor):
            return [_convert_tensor_to_numpy(x) for x in input]
        # 如果序列的第一个元素是 bool 类型，转换为 numpy 数组，数据类型为 np.bool_
        if isinstance(input[0], bool):
            return np.array(input, dtype=np.bool_)

        # 如果序列是一系列数字
        # 如果第一个元素是 int 类型，转换为 numpy 数组，数据类型为 np.int64
        if isinstance(input[0], int):
            return np.array(input, dtype=np.int64)
        # 如果第一个元素是 float 类型，转换为 numpy 数组，自动推断数据类型
        if isinstance(input[0], float):
            return np.array(input)
    
    # 对于其他类型的输入，直接返回该输入
    return input
```