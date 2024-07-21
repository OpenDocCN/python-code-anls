# `.\pytorch\torch\onnx\_internal\onnxruntime.py`

```py
# mypy: allow-untyped-defs
import dataclasses  # 导入用于数据类的模块
import importlib  # 导入用于动态导入模块的模块
import logging  # 导入用于日志记录的模块
import os  # 导入操作系统相关功能的模块

from typing import (  # 导入类型提示相关模块
    Any,
    Callable,
    Dict,
    Final,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)
from typing_extensions import TypeAlias  # 导入类型别名相关模块

import torch  # 导入PyTorch主模块
import torch._C  # 导入PyTorch的C++前端接口模块
import torch._ops  # 导入PyTorch运算符操作模块
import torch._prims.executor  # 导入PyTorch的执行器模块
import torch.fx  # 导入PyTorch的特效图模块
from torch._subclasses.fake_tensor import FakeTensor  # 从PyTorch的子类模块导入FakeTensor
from torch.fx._compatibility import compatibility  # 导入PyTorch特效图兼容性模块
from torch.fx.passes.fake_tensor_prop import FakeTensorProp  # 导入PyTorch特效图的FakeTensorProp模块
from torch.fx.passes.operator_support import OperatorSupport  # 导入PyTorch特效图操作支持模块
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS  # 导入PyTorch特效图工具通用模块
from torch.utils import _pytree  # 从PyTorch的工具模块导入_pytree

try:
    # 尝试导入依赖的ONNX和ONNX Runtime相关模块
    import onnx
    import onnxruntime  # type: ignore[import]
    from onnxruntime.capi import _pybind_state as ORTC  # type: ignore[import]

    # 虽然不直接在DORT中使用，但由底层导出器需要，因此仍需检查其是否存在。
    importlib.import_module("onnxscript")

    import torch.onnx
    import torch.onnx._internal
    import torch.onnx._internal.diagnostics
    import torch.onnx._internal.exporter
    import torch.onnx._internal.fx.decomposition_table
    import torch.onnx._internal.fx.passes
    from torch.onnx._internal.fx import fx_onnx_interpreter
    from torch.onnx._internal.fx.type_utils import (
        _TORCH_DTYPE_TO_NUMPY_DTYPE,
        _TORCH_DTYPE_TO_ONNX_TENSOR_ELEMENT_TYPE,
        from_python_type_to_onnx_tensor_element_type,
    )

    _SUPPORT_ONNXRT = True  # 设置ONNX Runtime支持标志为True
except ImportError:
    _SUPPORT_ONNXRT = False  # 如果导入失败，设置ONNX Runtime支持标志为False

__all__ = [  # 将公开的API名称列入__all__列表
    "is_onnxrt_backend_supported",
    "torch_compile_backend",
    "OrtExecutionProvider",
    "OrtBackendOptions",
    "OrtBackend",
]


def is_onnxrt_backend_supported() -> bool:
    """Returns ``True`` if ONNX Runtime dependencies are installed and usable
    to support TorchDynamo backend integration; ``False`` otherwise.

    Example::

        # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX)
        >>> import torch
        >>> if torch.onnx.is_onnxrt_backend_supported():
        ...     @torch.compile(backend="onnxrt")
        ...     def f(x):
        ...             return x * x
        ...     print(f(torch.randn(10)))
        ... else:
        ...     print("pip install onnx onnxscript onnxruntime")
        ...
    """
    return _SUPPORT_ONNXRT  # 返回ONNX Runtime支持标志的值


_dumped_onnx_model: Dict[str, int] = {}  # 初始化用于存储ONNX模型的字典


def _dump_onnx_model(
    model_string: bytes, graph_module: Optional[torch.fx.GraphModule] = None
) -> str:
    """Stores the onnx model into a file.
    The name is "{ONNXRT_DUMP_PATH}{N}.onnx"
    where *N* is the number of files already stored with
    this prefix.
    If graph_module is not None, the graph is stored as a string with
    the same filename except the extension (.txt).
    """
    prefix = os.environ.get("ONNXRT_DUMP_PATH", None)  # 获取环境变量ONNXRT_DUMP_PATH的值作为前缀
    if not prefix:
        return ""  # 如果前缀为空，则返回空字符串
    n = _dumped_onnx_model.get(prefix, -1) + 1  # 获取已存储的以该前缀为键的文件数，并加1作为新文件编号
    # 拼接文件名，格式为 "{prefix}{n}.onnx"
    filename = f"{prefix}{n}.onnx"
    # 打开文件并以二进制写入模型字符串
    with open(filename, "wb") as f:
        f.write(model_string)
    # 将文件名对应的模型索引保存到 _dumped_onnx_model 字典中
    _dumped_onnx_model[prefix] = n
    # 如果有图模块，创建对应的文本文件名 "{prefix}{n}.txt"
    if graph_module is not None:
        # 打开文本文件并以 UTF-8 编码写入图模块的图结构字符串表示
        filename_txt = f"{prefix}{n}.txt"
        with open(filename_txt, "w", encoding="utf-8") as f:
            f.write(str(graph_module.graph))
    # 返回保存模型的文件名
    return filename
def _infer_default_eps() -> Sequence[str]:
    # TODO: 根据主机的能力选择一个良好的默认值
    # 例如，在 Windows 上可能选择 DML 等。
    return ["CPUExecutionProvider"]


def _nvtx_range_push(name: str):
    """如果 PyTorch 安装了 CUDA 支持，则启动 NVTX 范围。

    详细信息请参阅 torch.cuda.nvtx.range_push 的文档。
    """
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)


def _nvtx_range_pop():
    """如果 PyTorch 安装了 CUDA 支持，则终止 NVTX 范围。

    详细信息请参阅 torch.cuda.nvtx.range_pop 的文档。
    """
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_pop()


def _get_ort_device_type(device_type: str):
    if device_type == "cuda":
        return ORTC.OrtDevice.cuda()
    if device_type == "cpu":
        return ORTC.OrtDevice.cpu()
    # ort pytorch device is mapped to NPU OrtDevice type
    if device_type == "maia":
        return ORTC.OrtDevice.npu()
    raise ValueError("Unsupported device type: " + device_type)


logger = logging.getLogger(__name__)
# Uncomment the following lines to print out development info.
# logging.basicConfig(level=logging.WARNING)
# logger.setLevel(logging.WARNING)


class OrtOperatorSupport(OperatorSupport):
    """ONNXRuntime 后端的操作符支持。

    它具有两级支持决策。一级是通过 support_dict，另一级
    是通过 extra_support_dict。在 OrtOperatorSupport 中实现了使用
    support_dict 的逻辑，extra_support_dict 则由 OperatorSupport.is_node_supported 使用。
    """

    def __init__(self, support_dict: Set[Any], extra_support_dict: Dict[str, Any]):
        # Use extra_support_dict[op_name] = None to indicate
        # we support op_name with all input types. Otherwise,
        # see support_dict (type: SupportDict) in operator_support.py
        # for specifying supported types.
        super().__init__(extra_support_dict)
        self._onnx_support_dict = support_dict

    def is_node_supported(
        self, submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node
    ):
        # 确定给定节点是否由 ONNX 支持
    ) -> bool:
        # OperatorSupport.is_node_supported 返回 True 表示不是可调用节点。
        # 由于ORT不能执行这些节点，我们在这里返回 False 以覆盖基本行为。
        if node.op not in CALLABLE_NODE_OPS:
            # 如果节点操作不在可调用节点操作列表中，则返回 False
            return False
        # 这是唯一的地方决定 aten 操作是否被支持。
        if node.op == "call_function" and node.target in self._onnx_support_dict:
            # 如果节点操作是 "call_function" 并且目标在支持字典中，则记录支持信息
            logger.info(
                "support_dict supports node.target: %s (type: %s)",
                node.target,
                type(node.target),
            )
            return True
        # 如果 node.target 不在支持字典中，我们仍然希望检查 torch.jit.script 是否能将其转换为 ONNX 等效形式。
        # 让我们使用基本机制来完成这个操作。参见 extra_support_dict 中支持的操作。
        if super().is_node_supported(submodules, node):
            # 如果基类的 is_node_supported 方法返回 True，则记录支持信息
            logger.info(
                "extra_support_dict supports node.target: %s (type: %s)",
                node.target,
                type(node.target),
            )
            return True
        # 如果以上条件都不满足，则记录警告信息，表明支持字典和额外支持字典均不支持该节点目标。
        logger.warning(
            "support_dict and extra_support_dict don't support node.target: %s (type: %s)",
            node.target,
            type(node.target),
        )
        # 返回 False 表示节点不受支持
        return False
# 将 torch.fx.GraphModule 中的占位符节点移动到节点列表的前面，确保其在上游节点计算值之前执行。
def _move_placeholder_to_front(graph_module: torch.fx.GraphModule) -> None:
    graph = graph_module.graph  # 获取图模块的计算图对象
    placeholders = []  # 存储占位符节点的列表
    first_not_placeholder = None  # 第一个非占位符节点
    for node in graph.nodes:
        if node.op == "placeholder":
            placeholders.append(node)  # 将占位符节点添加到列表中
        if first_not_placeholder is None and node.op != "placeholder":
            first_not_placeholder = node  # 找到第一个非占位符节点
    if first_not_placeholder is None:
        return  # 如果没有非占位符节点，则直接返回
    for placeholder in placeholders:
        first_not_placeholder.prepend(placeholder)  # 将占位符节点移动到第一个非占位符节点之前


# 从参数列表中返回第一个有效的设备（GPU 或 CPU）
def _infer_ep_from_device(*args) -> Tuple[str, ...]:
    eps = []  # 存储有效设备的列表
    for arg in args:
        if hasattr(arg, "device"):
            device = arg.device  # 获取参数的设备信息
            if device.type == "cuda":
                eps.append("CUDAExecutionProvider")  # 如果是 GPU 设备，添加 CUDAExecutionProvider
            elif device.type == "cpu":
                eps.append("CPUExecutionProvider")  # 如果是 CPU 设备，添加 CPUExecutionProvider
    return tuple(eps)  # 返回元组形式的有效设备列表


# 从 torch.fx.GraphModule 中提取所有占位符节点作为元组输出
def _extract_graph_module_inputs(graph_module: torch.fx.GraphModule) -> Tuple[Any, ...]:
    placeholders = []  # 存储所有占位符节点的列表
    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            if hasattr(node, "meta") and "val" in node.meta:
                assert isinstance(node.meta["val"], torch.Tensor)  # 断言确认 meta 字段中的 "val" 是 torch.Tensor 类型
            placeholders.append(node)  # 将符合条件的占位符节点添加到列表中
    return tuple(placeholders)  # 返回元组形式的占位符节点列表


# 从 torch.fx.GraphModule 中收集输出节点的 "val" 字段作为输出
def _extract_graph_module_outputs(graph_module: torch.fx.GraphModule) -> Any:
    """Collect "val" fields from outputs metadata in this torch.fx.GraphModule."""
    for node in graph_module.graph.nodes:
        if node.op == "output":
            # Output node is unique. Let's retrieve output values from
            # this node's input list. And then just return.
            return node.args[0]  # 返回输出节点的第一个参数作为输出值
    raise ValueError("No output node found in this torch.fx.GraphModule.")  # 如果没有找到输出节点，则抛出异常


# 从 torch.fx.GraphModule 中推断所有有效设备（GPU 或 CPU）
def _infer_ep_from_graph_module(graph_module: torch.fx.GraphModule) -> Tuple[str, ...]:
    """Return the all valid devices (i.e., GPU or CPU) among outputs of this torch.fx.GraphModule."""
    flattened_output_args, _ = _pytree.tree_flatten(
        _extract_graph_module_outputs(graph_module)
    )
    # Output arguments with example value (type: torch.Tensor) in the `graph_module`.
    selected_output_args = [
        output_arg.meta["val"]
        for output_arg in flattened_output_args
        # output_arg must have tensor for its device information.
        # Otherwise, skip it.
        if (hasattr(output_arg, "meta") and "val" in output_arg.meta)
    ]
    return _infer_ep_from_device(*selected_output_args)  # 返回输出中有效设备的元组


# 根据预设优先级对 eps 中的执行提供程序进行排序
def _sort_eps(eps: Tuple[str, ...]) -> Tuple[str, ...]:
    """Sort execution providers in eps based on pre-set priority."""
    # 定义函数，根据执行提供程序的名称返回其优先级
    def get_execution_provider_priority(ep: str) -> int:
        # 如果执行提供程序是 "CPUExecutionProvider"
        if ep == "CPUExecutionProvider":
            # 返回最低优先级
            return 2
        # 如果执行提供程序是 "CUDAExecutionProvider"
        if ep == "CUDAExecutionProvider":
            # 返回比 CPUExecutionProvider 高但比其他专用执行提供程序低的优先级
            return 1
        # 其他情况下返回最高优先级
        return 0

    # 使用集合去除重复的执行提供程序名称
    unique_eps = set(eps)
    # 返回按照执行提供程序优先级排序的元组，降序排列
    return tuple(sorted(unique_eps, key=get_execution_provider_priority, reverse=True))
# 定义一个函数 _get_onnx_devices，接收一个包含多种数据类型的元组作为参数，返回一个包含 ORT 设备的元组
def _get_onnx_devices(
    values: Tuple[
        Union[
            torch.Tensor, torch.SymInt, int, torch.SymFloat, float, torch.SymBool, bool
        ],
        ...,
    ]
) -> Tuple["ORTC.OrtDevice", ...]:
    # 定义一个内部函数 _device_id_or_zero，将设备 ID 转换为整数，如果为 None 则返回 0
    def _device_id_or_zero(device_id: int) -> int:
        return device_id or 0

    # 定义一个内部函数 _map_tensor_or_sym_to_device，将输入值映射为 ORT 设备对象
    def _map_tensor_or_sym_to_device(
        value: Union[
            torch.Tensor, torch.SymInt, int, torch.SymFloat, float, torch.SymBool, bool
        ],
    ) -> int:
        # 如果值是 torch.Tensor 类型，则创建对应的 ORT 设备对象
        if isinstance(value, torch.Tensor):
            return ORTC.OrtDevice(
                _get_ort_device_type(value.device.type),  # 获取 PyTorch 设备类型并转换为 ORT 设备类型
                ORTC.OrtDevice.default_memory(),  # 使用默认内存选项
                _device_id_or_zero(value.device.index),  # 获取设备索引，如果为 None 则使用 0
            )
        # 如果值是 torch.SymInt、int、torch.SymFloat、float、torch.SymBool、bool 类型之一，则创建默认的 CPU ORT 设备对象
        elif isinstance(
            value, (torch.SymInt, int, torch.SymFloat, float, torch.SymBool, bool)
        ):
            return ORTC.OrtDevice(
                _get_ort_device_type("cpu"),  # 使用 CPU 设备类型
                ORTC.OrtDevice.default_memory(),  # 使用默认内存选项
                0,  # 设备索引为 0
            )
        else:
            raise ValueError("Unsupported value type: " + str(type(value)))  # 抛出值类型不支持的错误

    # 如果输入的值的长度大于 0，则对每个值调用 _map_tensor_or_sym_to_device 函数，并返回结果元组
    if len(values) > 0:
        ort_devices = tuple(_map_tensor_or_sym_to_device(value) for value in values)
        return ort_devices
    else:
        return (_map_tensor_or_sym_to_device(1),)  # 如果没有输入值，则返回一个包含默认设备对象的元组


# 定义函数 _get_ortvalues_from_torch_tensors，接收 torch.Tensor 元组和 ORT 设备元组作为参数，返回 ORT 值向量
def _get_ortvalues_from_torch_tensors(
    tensors: Tuple[torch.Tensor, ...], devices: Tuple["ORTC.OrtDevice", ...]
) -> Tuple[torch.Tensor, ...]:
    # 创建一个 ORT 值向量对象
    ortvalues = ORTC.OrtValueVector()
    # 预留足够的空间来存储张量
    ortvalues.reserve(len(tensors))
    dtypes = []  # 用于存储数据类型列表
    shapes = []  # 用于存储形状列表
    data_ptrs = []  # 用于存储数据指针列表

    # 遍历输入的张量列表
    for tensor in tensors:
        dtypes.append(_TORCH_DTYPE_TO_NUMPY_DTYPE[tensor.dtype])  # 将 PyTorch 数据类型转换为 NumPy 数据类型
        shapes.append(tensor.size())  # 获取张量的形状并添加到 shapes 列表中
        data_ptrs.append(tensor.data_ptr())  # 获取张量的数据指针并添加到 data_ptrs 列表中
    # 将张量、数据指针、数据类型、形状和设备传递给 ORT 值向量对象
    ortvalues.push_back_batch(tensors, data_ptrs, dtypes, shapes, devices)
    return ortvalues  # 返回填充好数据的 ORT 值向量对象


# 定义函数 _to_real_tensor，将 FakeTensor 转换为真实的 torch.Tensor
def _to_real_tensor(tensor: FakeTensor) -> torch.Tensor:
    if tensor.is_sparse:  # 如果输入的 FakeTensor 是稀疏张量，则抛出错误，暂时不支持稀疏张量
        raise ValueError("sparse tensor is not yet supported.")
    # 创建一个与输入 FakeTensor 相同形状和数据类型的空张量
    out = torch.empty(tensor.size(), dtype=tensor.dtype, device=tensor.device)
    return out  # 返回创建的空张量


# 定义函数 _adjust_scalar_from_fx_to_onnx，将动态值转换为 ONNX 格式的 torch.Tensor
def _adjust_scalar_from_fx_to_onnx(
    dynamo_value: Union[
        torch.Tensor,
        int,
        float,
        bool,
    ],
    value_info: "onnx.ValueInfoProto",  # 声明 value_info 参数类型为 onnx.ValueInfoProto，忽略类型检查
) -> torch.Tensor:
    """Helper function to wrap PyTorch variables as torch.Tensor"""
    # 如果 dynamo_value 是 torch.Tensor 类型，且 value_info 的类型是标量且形状为空，且 dynamo_value 的形状为 (1,)
    if (
        isinstance(dynamo_value, torch.Tensor)
        and len(value_info.type.tensor_type.shape.dim) == 0
        and dynamo_value.shape == (1,)
    ):
        # ONNX 期望一个形状为空的标量
        # 而 PyTorch 通常允许形状为 () 和 (1,) 之间的隐式转换
        #
        # 下面的代码将 PyTorch 的形状 (1,) 重塑为 ()
        return torch.squeeze(dynamo_value)  # 去除维度为 1 的维度，返回标量张量
    elif isinstance(dynamo_value, int):
        return torch.tensor(dynamo_value, dtype=torch.int64)  # 将整数转换为 torch.Tensor，数据类型为 torch.int64
    elif isinstance(dynamo_value, float):
        return torch.tensor(dynamo_value, dtype=torch.float32)  # 将浮点数转换为 torch.Tensor，数据类型为 torch.float32
    # 如果 dynamo_value 是布尔类型，则将其转换为 PyTorch 的布尔张量
    elif isinstance(dynamo_value, bool):
        return torch.tensor(dynamo_value, dtype=torch.bool)
    # 如果 dynamo_value 不是布尔类型，则断言其为 PyTorch 张量
    else:
        assert isinstance(dynamo_value, torch.Tensor)
        # 返回 dynamo_value 的连续存储版本（contiguous 表示连续存储）
        return dynamo_value.contiguous()
# 定义一个函数，将从 ONNX 到 TorchFX 的标量值进行调整
def _adjust_scalar_from_onnx_to_fx(
    tensor: torch.Tensor,
    prim_value: Union[
        torch.Tensor,
        torch.SymInt,
        int,
        torch.SymFloat,
        float,
        torch.SymBool,
        bool,
    ],
) -> Union[torch.Tensor, int, float, bool,]:
    """Helper function to wrap ORT-produced torch.Tensor as PyTorch variables"""
    # 确保输入的 tensor 是 torch.Tensor 类型，因为 ORT 的输出必须是 tensor
    assert isinstance(tensor, torch.Tensor), "ORT's output must be tensor."
    
    # 如果 prim_value 是 torch.SymInt, int, torch.SymFloat, float, torch.SymBool 或 bool 类型之一
    if isinstance(
        prim_value,
        (torch.SymInt, int, torch.SymFloat, float, torch.SymBool, bool),
    ):
        # 将 tensor 转换为标量以匹配 Dynamo 的预期输出
        return tensor.item()
    
    # 如果 prim_value 不是上述类型之一，则直接返回 tensor
    return tensor


# 定义一个函数，使用 ORTValueVector 运行 ONNX 会话
def _run_onnx_session_with_ortvaluevector(
    sess: "onnxruntime.InferenceSession",
    input_names: Tuple[str, ...],
    inputs: Tuple[torch.Tensor, ...],
    input_devices: Tuple["ORTC.OrtDevice", ...],
    output_names: Tuple[str, ...],
    outputs: Tuple[torch.Tensor, ...],
    output_devices: Tuple["ORTC.OrtDevice", ...],
    preallocate_output: bool,
    input_value_infos: Tuple["onnx.ValueInfoProto", ...],  # type: ignore[name-defined]
    normalized_prim_outputs: Tuple[
        Union[
            torch.Tensor, torch.SymInt, int, torch.SymFloat, float, torch.SymBool, bool
        ],
        ...,
    ],
) -> Tuple[Union[torch.Tensor, int, float, bool], ...]:
    # 记录 NVidia Tools Extension (NVTX) 范围，标记为 "contiguous"
    _nvtx_range_push("contiguous")
    
    # 调整输入以匹配 ONNX 的期望格式
    inputs = tuple(
        _adjust_scalar_from_fx_to_onnx(arg, value_info)
        for arg, value_info in zip(inputs, input_value_infos)
    )
    
    # 结束 "contiguous" 范围标记
    _nvtx_range_pop()

    # 记录 NVTX 范围，标记为 "push_back_batch"
    _nvtx_range_push("push_back_batch")
    
    # 如果需要预先分配输出
    if preallocate_output:
        # 将 FakeTensor 转换为真实的 PyTorch Tensor，以便使用 torch 设备上的缓冲区
        pth_outputs = tuple(
            _to_real_tensor(t) if isinstance(t, FakeTensor) else t for t in outputs
        )
        # 将 PyTorch Tensor 转换为 ORTValues
        ort_outputs = _get_ortvalues_from_torch_tensors(pth_outputs, output_devices)
    else:
        # 如果不需要预先分配输出，则创建一个空的 ORTValueVector
        ort_outputs = ORTC.OrtValueVector()
    
    # 结束 "push_back_batch" 范围标记
    _nvtx_range_pop()

    # 记录 NVTX 范围，标记为 "run_with_ortvaluevector"
    _nvtx_range_push("run_with_ortvaluevector")
    
    # 创建运行选项
    run_options = onnxruntime.RunOptions()
    # 添加运行配置项，禁用执行提供程序的同步
    run_options.add_run_config_entry("disable_synchronize_execution_providers", "1")
    
    # 使用 ORTValueVector 运行 ONNX 会话
    sess.run_with_ortvaluevector(
        run_options, input_names, ort_inputs, output_names, ort_outputs, output_devices
    )
    
    # 结束 "run_with_ortvaluevector" 范围标记
    _nvtx_range_pop()

    # 后处理步骤：
    # 将 ORT 的输出包装成由 prim_output 表示的模式（通过运行原始的 torch.fx.GraphModule 获得）。
    # 如果设置了预分配输出内存的选项，则执行以下操作
    if preallocate_output:
        # 在运行 ortvaluevector 后，开始性能分析
        _nvtx_range_push("after run_with_ortvaluevector")
        # 输出结果存储在预分配的 torch.Tensor 内存中，
        # 因此此处不需要将 ORTValue 转换为 torch.Tensor。
        pth_outputs = tuple(
            _adjust_scalar_from_onnx_to_fx(onnx_output, prim_output)  # type: ignore[misc]
            for onnx_output, prim_output in zip(pth_outputs, normalized_prim_outputs)
        )
        # 结束性能分析
        _nvtx_range_pop()
        # 返回处理后的输出
        return pth_outputs
    else:
        # 在运行 ortvaluevector 后，开始性能分析
        _nvtx_range_push("after run_with_ortvaluevector")
        # 将 ORTValue 映射到 torch.Tensor。
        pth_outputs = onnxruntime.training.ortmodule._utils._ortvalues_to_torch_tensor(
            ort_outputs
        )
        # 将一些 torch.Tensor 转换为整数、浮点数或布尔值。
        pth_outputs = tuple(
            _adjust_scalar_from_onnx_to_fx(onnx_output, prim_output)  # type: ignore[misc]
            for onnx_output, prim_output in zip(pth_outputs, normalized_prim_outputs)
        )
        # 结束性能分析
        _nvtx_range_pop()
        # 返回处理后的输出
        return pth_outputs
# 定义一个函数，用于在 ONNX 会话中运行并获取输出
def _run_onnx_session_with_fetch(
    sess: "onnxruntime.InferenceSession",  # ONNX 推理会话对象
    input_names: Tuple[str, ...],  # 输入张量的名称元组
    inputs: Tuple[torch.Tensor, ...],  # 输入张量的数据元组
    input_devices: Tuple["ORTC.OrtDevice", ...],  # 输入张量的设备类型元组
    output_names: Tuple[str, ...],  # 输出张量的名称元组
    outputs: Tuple[torch.Tensor, ...],  # 输出张量的数据元组
    output_devices: Tuple["ORTC.OrtDevice", ...],  # 输出张量的设备类型元组
    preallocate_output: bool,  # 是否预分配输出
    input_value_infos: Tuple["onnx.ValueInfoProto", ...],  # 输入值信息元组
    normalized_prim_outputs: Tuple[  # 规范化后的主要输出元组，包括多种类型的张量
        Union[
            torch.Tensor, torch.SymInt, int, torch.SymFloat, float, torch.SymBool, bool
        ],
        ...,
    ],
) -> Tuple[Union[torch.Tensor, int, float, bool], ...]:  # 返回值为多种类型的张量元组
    # 将输入张量从 TorchScript 调整到 ONNX 格式
    inputs = tuple(
        _adjust_scalar_from_fx_to_onnx(arg, value_info)
        for arg, value_info in zip(inputs, input_value_infos)
    )
    # 创建用于 ONNX 推理会话的输入字典
    feed = {
        name: onnxruntime.OrtValue.ortvalue_from_numpy(tensor.cpu().numpy())
        for name, tensor in zip(input_names, inputs)
    }
    # 在 ONNX 会话中运行并获取输出
    ort_outputs = sess.run(output_names, feed)
    # 将 ONNX 输出从 ONNX 格式调整到 TorchScript 格式
    pth_outputs = tuple(
        _adjust_scalar_from_onnx_to_fx(
            torch.from_numpy(value),
            prim_output,
        )
        for value, prim_output in zip(ort_outputs, normalized_prim_outputs)
    )
    # 返回 TorchScript 格式的输出张量元组
    return pth_outputs


class OrtExecutionInfoPerSession:
    """用于使用 onnxruntime.InferenceSession 执行 torch.fx.GraphModule 所需的信息"""

    def __init__(
        self,
        session: "onnxruntime.InferenceSession",  # ONNX 推理会话对象
        input_names: Tuple[str, ...],  # 输入张量的名称元组
        input_value_infos: Tuple["onnx.ValueInfoProto", ...],  # 输入值信息元组
        output_names: Tuple[str, ...],  # 输出张量的名称元组
        output_value_infos: Tuple["onnx.ValueInfoProto", ...],  # 输出值信息元组
        input_devices: Tuple["ORTC.OrtDevice", ...],  # 输入张量的设备类型元组
        output_devices: Tuple["ORTC.OrtDevice", ...],  # 输出张量的设备类型元组
        example_outputs: Union[Tuple[torch.Tensor, ...], torch.Tensor],  # 示例输出张量或张量元组
    ):
        # 定义了一个类，用于管理 ONNX 模型及其执行器的相关信息。
        self.session: onnxruntime.InferenceSession = session
        # 对于存储在 self.session 中的 ONNX 模型，self.input_names[i] 是第 i 个位置输入的名称。
        self.input_names: Tuple[str, ...] = input_names
        # 存储了 self.input_names[i] 的类型信息在 self.input_value_infos[i] 中。
        self.input_value_infos: Tuple[onnx.ValueInfoProto, ...] = input_value_infos  # type: ignore[name-defined]
        # 类似于 self.input_names，但用于输出。
        self.output_names: Tuple[str, ...] = output_names
        # 类似于 self.input_value_infos，但用于输出。
        self.output_value_infos: Tuple[onnx.ValueInfoProto, ...] = output_value_infos  # type: ignore[name-defined]
        # 对于存储在 self.session 中的 ONNX 模型，self.input_devices[i] 是第 i 个位置输入的设备。
        self.input_devices: Tuple["ORTC.OrtDevice", ...] = input_devices
        # 类似于 self.input_devices，但用于输出。
        self.output_devices: Tuple["ORTC.OrtDevice", ...] = output_devices
        # 这是使用示例输入（即传递给 OrtBackend._ort_acclerated_call 的参数）执行原始 torch.fx.GraphModule 的输出。
        self.example_outputs: Union[
            Tuple[torch.Tensor, ...], torch.Tensor
        ] = example_outputs

    def is_supported(self, *args):
        # 比较 args 和 ONNX 模型中的输入架构，并返回第一个匹配项。
        if len(args) != len(self.input_value_infos):
            return False
        for arg, value_info in zip(args, self.input_value_infos):
            if not isinstance(arg, (torch.Tensor, float, int)):
                return False

            # 检查 Python 标量，如 int、float 和 bool。
            if isinstance(arg, (int, float, bool)):
                # 将 float 等映射到 onnx.TensorProto.FLOAT 等。
                onnx_dtype = from_python_type_to_onnx_tensor_element_type(type(arg))
                if onnx_dtype != value_info.type.tensor_type.elem_type:
                    return False
                if len(value_info.type.tensor_type.shape.dim) != 0:
                    return False
                continue

            # 检查张量。
            onnx_dtype = _TORCH_DTYPE_TO_ONNX_TENSOR_ELEMENT_TYPE[arg.dtype]
            if onnx_dtype != value_info.type.tensor_type.elem_type:
                return False
            for dim, onnx_dim in zip(arg.shape, value_info.type.tensor_type.shape.dim):
                if isinstance(dim, int) and (
                    onnx_dim.dim_value == dim or onnx_dim.dim_param
                ):
                    continue
                elif isinstance(dim, torch.SymInt) and onnx_dim.dim_param:
                    continue
                else:
                    return False
        return True
@dataclasses.dataclass
class OrtExecutionInfoForAllGraphModules:
    def __init__(self):
        # 存储每个 GraphModule 导出的模型的所有会话及其相关信息，这些模型使用不同的输入。
        self.execution_info_per_graph_module: Dict[
            torch.fx.GraphModule, List[OrtExecutionInfoPerSession]
        ] = {}

    def search_reusable_session_execution_info(
        self, graph_module: torch.fx.GraphModule, *args
    ):
        # 如果给定的 graph_module 不在 execution_info_per_graph_module 中，返回 None。
        if graph_module not in self.execution_info_per_graph_module:
            return None
        # 获取与给定 graph_module 相关的所有执行信息对象的列表。
        candidates = self.execution_info_per_graph_module[graph_module]

        # 遍历 candidates 列表中的每个 OrtExecutionInfoPerSession 对象。
        for candidate in candidates:
            # 如果 candidate 支持给定的参数 *args，则返回该 candidate。
            if candidate.is_supported(*args):
                # 返回第一个接受此输入模式的会话对象。
                return candidate
        # 如果没有找到可重用的会话对象，返回 None。
        # 没有找到可重用的会话对象。
        return None

    def cache_session_execution_info(
        self, graph_module: torch.fx.GraphModule, info: OrtExecutionInfoPerSession
    ):
        # 如果给定的 graph_module 不在 execution_info_per_graph_module 中，创建一个新的列表来存储 info。
        if graph_module not in self.execution_info_per_graph_module:
            self.execution_info_per_graph_module[graph_module] = [info]
        else:
            # 否则，将 info 添加到对应 graph_module 的执行信息列表中。
            self.execution_info_per_graph_module[graph_module].append(info)


OrtExecutionProvider: TypeAlias = Union[str, Tuple[str, Mapping[str, Any]]]
"""表示 ONNX Runtime 执行提供者的类型别名，可以是一个字符串，也可以是一个包含名称和选项字典的二元组。

Examples::

    >>> "CPUExecutionProvider"

    >>> ("CUDAExecutionProvider", {"device_id": 3})

"""


@dataclasses.dataclass(frozen=True)
@compatibility(is_backward_compatible=False)
class OrtBackendOptions:
    """构造 ``OrtBackend``（即 ONNX Runtime 后端 ``"onnxrt"``）的选项。

    Example::

        >>> @torch.compile(
        ...     backend="onnxrt",
        ...     options=torch.onnx._OrtBackendOptions(...),
        ... )
        ... def ort_function(x):
        ...     return x ** x
    """

    preferred_execution_providers: Optional[Sequence[OrtExecutionProvider]] = None
    """可选的执行提供者序列，优先级高于推断的执行提供者（见 ``infer_execution_providers``）."""

    infer_execution_providers: bool = True
    """是否从输入绑定的 torch.device 或图中找到执行提供者进行推断。"""

    default_execution_providers: Optional[Sequence[OrtExecutionProvider]] = None
    """默认的回退执行提供者列表。如果未指定，默认为根据主机环境选择（可能是 ``"CPUExecutionProvider"``）。"""

    # preallocate_output 允许预分配输出的 torch Tensor 缓冲区，并将其提供给 InferenceSession，
    # 以避免在 InferenceSession 内部分配输出缓冲区。
    # 是否预先为 ONNX Runtime 的输出在 PyTorch 端分配内存。
    preallocate_output: bool = False
    """如果为 ``True``，则在 PyTorch 端为 ONNX Runtime 的输出分配内存。"""

    # 是否使用 AOT 自动求导，将 ``OrtBackend`` 包装为 TorchDynamo 的 aot_autograd 后端，
    # 以支持训练（即将反向图也发送到 ``OrtBackend``）。
    use_aot_autograd: bool = True
    """是否将 ``OrtBackend`` 包装为 TorchDynamo 的 aot_autograd 后端，以支持训练。 
    
    使用符号执行捕捉前向传播和反向传播为单一图。然后，选择的图分割算法（``min_cut_rematerialization_partition``）
    将整个图分成前向子图和后向子图。最后，两个子图都由 ``OrtBackend`` 编译。"""

    # TorchDynamo-based ONNX 导出器的选项，由 ``OrtBackend`` 使用。
    export_options: Optional["torch.onnx.ExportOptions"] = None
    """TorchDynamo 基于的 ONNX 导出器的选项，由 ``OrtBackend`` 使用。"""

    # ``onnxruntime.InferenceSession`` 使用的选项，由 ``OrtBackend`` 使用。
    ort_session_options: Optional["onnxruntime.SessionOptions"] = None
    """``onnxruntime.InferenceSession`` 使用的选项，由 ``OrtBackend`` 使用。"""

    # 应用于输入到 ONNXRuntime 的 ONNX 模型之前的图转换列表。
    pre_ort_model_transforms: Optional[  # type: ignore[name-defined]
        Sequence[Callable[["onnx.ModelProto"], None]]
    ] = None
    """应用于输入到 ONNXRuntime 的 ONNX 模型之前的图转换列表。"""
# 定义 OrtBackend 类，用于将 torch.fx.GraphModule 中的子图编译为 onnxruntime.InferenceSession 调用的后端。

@compatibility(is_backward_compatible=False)
class OrtBackend:
    """A backend compiles (sub-)graphs in torch.fx.GraphModule to onnxruntime.InferenceSession calls.
    
    The compiler entry point is OrtBackend.compile, which
        1. partitions the original graph into supported sub-graphs (type: torch.fx.GraphModule) and unsupported
           sub-graphs.
        2. For each supported sub-graph, it replaces its _wrapped_call function with _ort_accelerated_call.
        3. Inside _ort_accelerated_call, it creates onnxruntime.InferenceSession and calls it to execute the sub-graph.
    """

    def _select_eps(
        self, graph_module: torch.fx.GraphModule, *args
    ) -> Sequence[Tuple[str, Mapping[str, Any]]]:
        # 选择执行提供者的函数，返回一个由 (名称, 参数映射) 组成的序列

        # 推断出的执行提供者，默认为空元组
        inferred_eps: Tuple[str, ...] = tuple()
        
        # 如果设置了推断执行提供者选项
        if self._options.infer_execution_providers:
            # 如果从参数中推断出执行提供者
            if eps_from_args := _infer_ep_from_device(*args):
                # 如果用户将 CUDA 张量作为输入参数，优先使用 CUDA 执行提供者
                inferred_eps = eps_from_args
            # 否则，从图模块中推断出执行提供者
            elif eps_from_graph_module := _infer_ep_from_graph_module(graph_module):
                # 如果没有在输入参数中指定执行提供者，从图模块的输出中推断
                # 这些输出可能来自 FakeTensorProp 或 Dynamo 的内置符号形状推断
                inferred_eps = eps_from_graph_module

        # 选定的执行提供者列表
        selected_eps = []

        # 遍历执行提供者的排序后的顺序：
        #   - 用户首选的执行提供者列表
        #   - 从参数推断的执行提供者
        #   - 默认的执行提供者列表或推断的默认执行提供者
        for ep in (
            *(self._options.preferred_execution_providers or []),
            *_sort_eps(inferred_eps),
            *(self._options.default_execution_providers or _infer_default_eps()),
        ):
            # 如果执行提供者是字符串，则转换为 (名称, {}) 格式
            if isinstance(ep, str):
                ep = (ep, {})
            # 如果执行提供者是元组且第二个元素为 None，则转换为 (名称, {}) 格式
            elif isinstance(ep, tuple) and ep[1] is None:
                ep = (ep[0], {})
            # 如果执行提供者不为空且未在选定的执行提供者列表中，则添加到列表中
            if ep is not None and ep not in selected_eps:
                selected_eps.append(ep)

        # 返回选定的执行提供者列表
        return selected_eps
    def compile(self, graph_module: torch.fx.GraphModule, args) -> torch.fx.GraphModule:
        # 延迟导入，因为CapabilityBasedPartitioner没有被@compatibility装饰；
        # 在模块级别导入会导致测试失败：
        # pytest test/test_fx.py -k test_public_api_surface
        # 因为这个模块被导入到torch.onnx中。
        from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner

        # 基于ONNX支持的操作对FX图进行分区。
        # 给定一个图模块graph_module
        #  GraphModule0
        #   node_0
        #   node_1
        #   node_2
        #   node_3
        #   node_4
        # 如果只有node_2不被ONNX支持，这个图模块将被分区成
        #  GraphModule0
        #   GraphModule1
        #    node_0
        #    node_1
        #   node_2
        #   GraphModule2
        #    node_3
        #    node_4
        # 通过调用CapabilityBasedPartitioner.partition_and_fuse。
        # 然后，GraphModule1和GraphModule2的forward方法(GraphModule._wrapped_call)
        # 将被替换为OrtBackend._ort_accelerated_call来委托计算给ORT。
        if graph_module in self._partitioner_cache:
            # 如果图模块已经在缓存中，直接从缓存中获取分区后的主要图模块
            partitioned_prim_graph_module = self._partitioner_cache[graph_module]
        else:
            # 否则，创建一个主要图模块
            prim_graph_module = graph_module
            # 使用CapabilityBasedPartitioner对主要图模块进行分区
            partitioner = CapabilityBasedPartitioner(
                prim_graph_module,
                self._supported_ops,
                allows_single_node_partition=True,
            )
            # 进行分区和融合，并得到分区后的主要图模块
            partitioned_prim_graph_module = partitioner.partition_and_fuse()
            # 将分区后的主要图模块存入缓存
            self._partitioner_cache[graph_module] = partitioned_prim_graph_module

            # 替换融合模块(fused_module)的__call__()函数为ort_acclerated_call()
            # 此循环遍历所有图分区（每个分区都是一个可以被ONNX表示的图），
            # 并将它们的_wrapped_call函数重写为_ort_accelerated_call。
            # 在_ort_accelerated_call内部，分区的图被导出为ONNX并由ORT执行。
            for node in partitioned_prim_graph_module.graph.nodes:
                # TODO(wschin): 使用更好的方式识别融合子模块
                # 参见 https://github.com/pytorch/pytorch/issues/106872。
                if node.op == "call_module" and "fused_" in node.name:
                    fused_module = getattr(partitioned_prim_graph_module, node.name)
                    # self.ort_acclerated_call负责将图导出到ONNX，
                    # 创建ORT会话，并运行ORT会话。
                    fused_module._wrapped_call = self._ort_accelerated_call

        # 返回分区后的主要图模块
        return partitioned_prim_graph_module
    ) -> torch.fx.GraphModule:
        """If ``OrtBackendOptions.use_aot_autograd`` is ``True``, the `auto_autograd` compiler
        will be invoked, wrapping this ``OrtBackend`` instance's ``compile`` method. Otherwise,
        the ``compile`` method is invoked directly."""
        # 如果设置了 OrtBackendOptions.use_aot_autograd 为 True，则调用 auto_autograd 编译器，
        # 包装当前 OrtBackend 实例的 compile 方法；否则直接调用 compile 方法。
        if self._options.use_aot_autograd:
            # 导入必要的函数和模块
            from functorch.compile import min_cut_rematerialization_partition
            from torch._dynamo.backends.common import aot_autograd

            # 使用 auto_autograd 函数，传入相关参数，返回编译后的图模块
            return aot_autograd(
                fw_compiler=self.compile,
                partition_fn=min_cut_rematerialization_partition,
                decompositions=self._resolved_onnx_exporter_options.decomposition_table,
            )(graph_module, args)

        # 如果未启用 AOT 自动微分，直接调用 compile 方法并返回结果
        return self.compile(graph_module, args)

    __instance_cache_max_count: Final = 8
    __instance_cache: Final[List["OrtBackend"]] = []

    @staticmethod
    def get_cached_instance_for_options(
        options: Optional[Union[OrtBackendOptions, Mapping[str, Any]]] = None,
    @staticmethod
    def clear_cached_instances():
        # 清空 OrtBackend.__instance_cache 中的所有实例
        OrtBackend.__instance_cache.clear()

    @staticmethod
    def get_cached_instances():
        # 返回当前 OrtBackend.__instance_cache 中的所有实例
        return tuple(OrtBackend.__instance_cache)
# 使用装饰器指定函数的兼容性信息，声明该函数不向后兼容
@compatibility(is_backward_compatible=False)
# 定义函数，将 Torch 的图模块和参数传入，以及可选的后端选项
def torch_compile_backend(
    graph_module: torch.fx.GraphModule,
    args,
    *,
    options: Optional[Union[OrtBackendOptions, Mapping[str, Any]]] = None,
):
    # 调用 OrtBackend 类的方法，根据给定的选项获取已缓存的实例，并将图模块和参数传递给该实例
    return OrtBackend.get_cached_instance_for_options(options)(graph_module, args)
```