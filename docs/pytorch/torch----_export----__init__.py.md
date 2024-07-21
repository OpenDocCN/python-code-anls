# `.\pytorch\torch\_export\__init__.py`

```py
# mypy: allow-untyped-defs
import copy  # 导入copy模块，用于复制对象
import dataclasses  # 导入dataclasses模块，用于数据类的装饰器
import functools  # 导入functools模块，用于高阶函数的工具
import io  # 导入io模块，用于核心Python I/O功能的工具
import json  # 导入json模块，用于处理JSON数据
import logging  # 导入logging模块，用于Python的日志记录工具
import os  # 导入os模块，用于与操作系统交互
import re  # 导入re模块，用于正则表达式操作
import sys  # 导入sys模块，提供了与Python解释器相关的变量和函数
import types  # 导入types模块，用于操作Python类型和运行时对象
import warnings  # 导入warnings模块，用于警告控制
import weakref  # 导入weakref模块，用于对象的弱引用
import zipfile  # 导入zipfile模块，用于ZIP文件的读写

from collections import OrderedDict  # 导入OrderedDict类，实现了有序字典
from contextlib import contextmanager  # 导入contextmanager装饰器，用于创建上下文管理器
from functools import lru_cache  # 导入lru_cache装饰器，用于缓存函数的调用结果

from typing import Any, Callable, Dict, List, Optional, Tuple, Union  # 导入类型提示相关的声明

from unittest.mock import patch  # 从unittest.mock模块导入patch，用于模拟对象

import sympy  # 导入sympy库，用于符号计算

import torch  # 导入PyTorch库
import torch._dynamo  # 导入torch._dynamo模块
import torch.fx  # 导入torch.fx模块
import torch.utils._pytree as pytree  # 导入torch.utils._pytree模块，作为pytree的别名

from torch._decomp import core_aten_decompositions, get_decompositions  # 从torch._decomp模块导入函数
from torch._dispatch.python import enable_python_dispatcher  # 从torch._dispatch.python模块导入函数
from torch._dynamo.exc import UserError, UserErrorType  # 从torch._dynamo.exc模块导入异常类和类型
from torch._dynamo.source import ConstantSource  # 从torch._dynamo.source模块导入ConstantSource类
from torch._export.non_strict_utils import make_constraints  # 从torch._export.non_strict_utils模块导入函数
from torch._export.passes.collect_tracepoints_pass import CollectTracepointsPass  # 从torch._export.passes.collect_tracepoints_pass模块导入类
from torch._functorch.aot_autograd import aot_export_module, GraphSignature  # 从torch._functorch.aot_autograd模块导入函数和类
from torch._functorch.eager_transforms import functionalize  # 从torch._functorch.eager_transforms模块导入函数
from torch._guards import detect_fake_mode  # 从torch._guards模块导入函数
from torch._inductor import config  # 从torch._inductor模块导入config对象
from torch._ops import OpOverload  # 从torch._ops模块导入OpOverload类
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode  # 从torch._subclasses.fake_tensor模块导入类和枚举
from torch._subclasses.functional_tensor import FunctionalTensor  # 从torch._subclasses.functional_tensor模块导入FunctionalTensor类
from torch._utils_internal import log_export_usage  # 从torch._utils_internal模块导入log_export_usage函数
from torch.export._tree_utils import reorder_kwargs  # 从torch.export._tree_utils模块导入reorder_kwargs函数
from torch.export._unlift import _create_stateful_graph_module  # 从torch.export._unlift模块导入_create_stateful_graph_module函数
from torch.export.dynamic_shapes import _combine_args, Constraint, dims, dynamic_dim  # 从torch.export.dynamic_shapes模块导入函数和类
from torch.export.exported_program import (  # 从torch.export.exported_program模块导入多个类
    _disable_prexisiting_fake_mode,
    ExportedProgram,
    ModuleCallEntry,
    ModuleCallSignature,
)
from torch.export.graph_signature import (  # 从torch.export.graph_signature模块导入多个类和函数
    _sig_to_specs,
    ArgumentSpec,
    ConstantArgument,
    ExportGraphSignature,
    InputKind,
    InputSpec,
    OutputKind,
    OutputSpec,
    SymIntArgument,
    TensorArgument,
)
from torch.fx import traceback as fx_traceback  # 从torch.fx模块导入traceback模块，并将其重命名为fx_traceback
from torch.fx._compatibility import compatibility  # 从torch.fx._compatibility模块导入compatibility函数
from torch.fx.experimental.proxy_tensor import make_fx, maybe_disable_fake_tensor_mode  # 从torch.fx.experimental.proxy_tensor模块导入函数
from torch.fx.experimental.symbolic_shapes import (  # 从torch.fx.experimental.symbolic_shapes模块导入多个类
    ConstraintViolationError,
    GuardOnDataDependentSymNode,
    ShapeEnv,
    StrictMinMaxConstraint,
)
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo  # 从torch.fx.graph模块导入类
from torch.utils._sympy.value_ranges import ValueRangeError, ValueRanges  # 从torch.utils._sympy.value_ranges模块导入类和异常

from .wrappers import _wrap_submodules  # 从当前包的wrappers模块导入_wrap_submodules函数

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


@dataclasses.dataclass
class ExportDynamoConfig:
    """
    Manage Export-specific configurations of Dynamo.
    """
    allow_rnn: bool = True  # 数据类，用于管理Dynamo的导出特定配置，包括是否允许RNN，默认为True


# We only want to print this once to avoid flooding logs in workflows where capture_pre_autograd_graph
# is called multiple times.
@lru_cache
def capture_pre_autograd_graph_warning():
    log.warning("+============================+")
    log.warning("|     !!!   WARNING   !!!    |")
    log.warning("+============================+")
    # 输出警告日志，指出 capture_pre_autograd_graph() 方法已被弃用且未来不再提供功能保证。
    log.warning("capture_pre_autograd_graph() is deprecated and doesn't provide any function guarantee moving forward.")
    # 输出警告日志，建议使用 torch.export 替代原来的方法。
    log.warning("Please switch to use torch.export instead.")
    # 检查当前配置是否为 Facebook 的 fbcode 环境。
    if config.is_fbcode():
        # 如果单元测试不在阻止列表中，输出警告日志说明 capture_pre_autograd_graph() 方法将回退到使用 torch.export。
        log.warning("Unless the unittest is in the blocklist, capture_pre_autograd_graph() will fallback to torch.export.")
# 定义一个装饰器函数，指定函数的兼容性为不向后兼容
@compatibility(is_backward_compatible=False)
# 定义一个函数，用于捕获自动求导图之前的模块追踪
def capture_pre_autograd_graph(
    # 参数 f：需要被追踪的 nn.Module 对象
    f: torch.nn.Module,
    # 参数 args：示例的位置输入参数
    args: Tuple[Any],
    # 参数 kwargs：可选的示例关键字输入参数，默认为 None
    kwargs: Optional[Dict[str, Any]] = None,
    # 参数 dynamic_shapes：动态形状的规格说明，可以是字典或元组
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
) -> torch.nn.Module:
    """
    A helper function that is intended to trace a module before any pre-autograd
    decomposition is run. The produced module will be "non-functional" and
    composed of aten operators. Later this API will be deleted in favor of more general
    torch.export API.

    Args:
      f: nn.Module to be traced

      args: example positional inputs.

      kwargs: optional example keyword inputs.

      dynamic_shapes: Should either be:
         1) a dict from argument names of ``f`` to their dynamic shape specifications,
         2) a tuple that specifies dynamic shape specifications for each input in original order.
         If you are specifying dynamism on keyword args, you will need to pass them in the order that
         is defined in the original function signature.

         The dynamic shape of a tensor argument can be specified as either
         (1) a dict from dynamic dimension indices to :func:`Dim` types, where it is
         not required to include static dimension indices in this dict, but when they are,
         they should be mapped to None; or (2) a tuple / list of :func:`Dim` types or None,
         where the :func:`Dim` types correspond to dynamic dimensions, and static dimensions
         are denoted by None. Arguments that are dicts or tuples / lists of tensors are
         recursively specified by using mappings or sequences of contained specifications.

    Returns:
        An nn.Module containing the traced method.

    """
    # 导入必要的模块和函数
    from torch.export._trace import _convert_input_to_fake, DEFAULT_EXPORT_DYNAMO_CONFIG, _ignore_backend_decomps
    from torch._utils_internal import export_api_rollout_check

    # 调用捕获自动求导图警告函数
    capture_pre_autograd_graph_warning()

    # 断言参数 f 必须是 torch.nn.Module 的实例
    assert isinstance(f, torch.nn.Module), "Expected an nn.Module instance."

    # 如果 kwargs 为 None，则将其设为空字典
    if kwargs is None:
        kwargs = {}

    # 如果满足导出 API 检查条件
    if export_api_rollout_check():
        # 定义一个带有 lru_cache 装饰器的内部函数 print_export_warning
        @lru_cache
        def print_export_warning():
            # 输出警告信息，指示使用 torch.export._trace._export
            log.warning("Using torch.export._trace._export")
        # 调用 print_export_warning 函数
        print_export_warning()
        # 调用 torch.export._trace._export 函数，生成被追踪方法的模块
        module = torch.export._trace._export(f, args, kwargs, dynamic_shapes=dynamic_shapes, pre_dispatch=True).module()
    else:
        log_export_usage(event="export.private_api", flags={"capture_pre_autograd_graph"})

        # 导出模型时不要分解 dropout 操作，因为在 eval 模式下，dropout 操作从图中消失，
        # 这使得难以切换到 train 模式。参考：https://github.com/pytorch/pytorch/pull/115258#issuecomment-1900755832.
        # 创建分解表，排除默认的 dropout 操作
        decomp_table = {
            op: op.decompose
            for op in FunctionalTensor.maybe_aliasing_or_mutating_ops
            if op != torch.ops.aten.dropout.default
        }

        # 使用指定的导出配置和忽略后端分解操作上下文管理器
        with torch._dynamo.config.patch(dataclasses.asdict(DEFAULT_EXPORT_DYNAMO_CONFIG)), _ignore_backend_decomps():
            # 导出模型
            m = torch._dynamo.export(
                f,
                dynamic_shapes=dynamic_shapes,
                assume_static_by_default=True,
                tracing_mode="symbolic",
                decomposition_table=decomp_table,
                pre_dispatch=True,
                aten_graph=True,
                _log_export_usage=False,
            )(
                *args,
                **kwargs,
            )[0]

            # 将模型和输入转换为虚拟模式
            _, _, _, fake_mode = _convert_input_to_fake(m, args, kwargs)

            # 在模型元数据中添加内联约束
            m.meta["inline_constraints"] = {
                k: v
                for k, v in fake_mode.shape_env.var_to_range.items()
                if re.match(r"^[if]\d+$", str(k))
            }

            # 如果导出的是 torch.nn.Module 类型的模型，则恢复其状态字典
            if isinstance(f, torch.nn.Module):
                from torch.export._trace import _restore_state_dict
                _restore_state_dict(f, m)

            # 扁平化参数列表并合并参数
            flat_args, _ = pytree.tree_flatten((args, kwargs or {}))
            combined_args = _combine_args(f, args, kwargs)

            # 生成约束条件
            range_constraints = make_constraints(
                fake_mode,
                m,
                combined_args,
                dynamic_shapes,
                0,
            )

            # 创建具有状态的图形模块
            module = _create_stateful_graph_module(
                m,
                range_constraints=range_constraints,
            )

        # 设置错误消息，指出不支持对导出模型调用 train() 或 eval()
        error_message = \
            """
            Calling train() or eval() is not supported for exported models.
            Alternatively, you may override these methods to do custom user behavior as follows:

                def _my_train(self, mode: bool = True):
                    ...

                def _my_eval(self):
                    ...

                model.train = types.MethodType(_my_train, model)
                model.eval = types.MethodType(_my_eval, model)
            """

    # 定义 _train 方法，当调用时抛出 NotImplementedError
    def _train(self, mode: bool = True):
        raise NotImplementedError(error_message)

    # 定义 _eval 方法，当调用时抛出 NotImplementedError
    def _eval(self, mode: bool = True):
        raise NotImplementedError(error_message)

    # 将 _train 和 _eval 方法绑定到模块的 train 和 eval 方法上，并返回模块
    module.train = types.MethodType(_train, module)  # type: ignore[method-assign]
    module.eval = types.MethodType(_eval, module)  # type: ignore[method-assign]
    return module
def save(
    ep: ExportedProgram,
    f: Union[str, os.PathLike, io.BytesIO],
    *,
    extra_files: Optional[Dict[str, Any]] = None,
    opset_version: Optional[Dict[str, int]] = None,
) -> None:
    # 确保 ep 是 ExportedProgram 类型，否则抛出类型错误异常
    if not isinstance(ep, ExportedProgram):
        raise TypeError(f"save() expects an ExportedProgram but got {type(ep)}")

    # 导入序列化函数和序列化后的工件类型
    from .serde.serialize import serialize, SerializedArtifact
    # 导入架构版本号
    from .serde.schema import SCHEMA_VERSION
    # 序列化 ExportedProgram 对象，并获取序列化后的工件
    artifact: SerializedArtifact = serialize(ep, opset_version)

    # 如果 f 是字符串或者路径对象，则将其转换为文件路径字符串
    if isinstance(f, (str, os.PathLike)):
        f = os.fspath(f)

    # 使用 zipfile.ZipFile 打开文件 f，模式为写入 ('w')
    with zipfile.ZipFile(f, 'w') as zipf:
        # 将 artifact 的各个字段写入 ZIP 文件中对应的文件名
        assert isinstance(artifact.exported_program, bytes)
        zipf.writestr("serialized_exported_program.json", artifact.exported_program)
        zipf.writestr("serialized_state_dict.pt", artifact.state_dict)
        zipf.writestr("serialized_constants.pt", artifact.constants)
        zipf.writestr("serialized_example_inputs.pt", artifact.example_inputs)

        # 将当前使用的架构版本写入 'version' 文件
        zipf.writestr('version', ".".join(map(str, SCHEMA_VERSION)))

        # 如果提供了额外文件 (extra_files)，将其写入 'extra_files' 文件夹中
        if extra_files:
            for extra_file_name, content in extra_files.items():
                encoded_content = content.encode('utf-8')
                zipf.writestr(f"extra_files/{extra_file_name}", encoded_content)


def load(
    f: Union[str, os.PathLike, io.BytesIO],
    *,
    extra_files: Optional[Dict[str, Any]] = None,
    expected_opset_version: Optional[Dict[str, int]] = None,
) -> ExportedProgram:
    # 如果 f 是字符串或者路径对象，则将其转换为文件路径字符串
    if isinstance(f, (str, os.PathLike)):
        f = os.fspath(f)

    # 初始化 extra_files 为一个空字典，如果没有提供额外文件
    extra_files = extra_files or {}
    # 使用 zipfile 库打开 ZIP 文件 f，并将其作为 zipf 对象进行处理
    with zipfile.ZipFile(f, 'r') as zipf:
        # 读取 ZIP 文件中的 'version' 文件内容，解码成字符串后分割为版本号列表
        version = zipf.read('version').decode().split('.')
        
        # 导入 SCHEMA_VERSION 变量，确保版本号长度匹配
        from .serde.schema import SCHEMA_VERSION
        assert len(version) == len(SCHEMA_VERSION)
        
        # 检查主版本号是否与当前 schema 版本相符
        if version[0] != str(SCHEMA_VERSION[0]):
            raise RuntimeError(
                f"Serialized version {version} does not match our current "
                f"schema version {SCHEMA_VERSION}."
            )
        
        # 导入 deserialize 和 SerializedArtifact 类
        from .serde.serialize import deserialize, SerializedArtifact
        
        # 初始化用于存储序列化数据的变量
        serialized_exported_program: Optional[bytes] = None
        serialized_state_dict: Optional[bytes] = None
        serialized_constants: Optional[bytes] = None
        serialized_example_inputs: Optional[bytes] = None
        
        # 遍历 ZIP 文件中的所有文件信息
        for file_info in zipf.infolist():
            # 读取当前文件的内容
            file_content = zipf.read(file_info.filename)
            
            # 根据文件名将内容分配到相应的变量中
            if file_info.filename == "serialized_exported_program.json":
                serialized_exported_program = file_content
            elif file_info.filename == "serialized_state_dict.json":
                warnings.warn("This version of file is deprecated")
                serialized_state_dict = file_content
            elif file_info.filename == "serialized_constants.json":
                warnings.warn("This version of file is deprecated")
                serialized_constants = file_content
            elif file_info.filename == "serialized_state_dict.pt":
                serialized_state_dict = file_content
            elif file_info.filename == "serialized_constants.pt":
                serialized_constants = file_content
            elif file_info.filename == "serialized_example_inputs.pt":
                serialized_example_inputs = file_content
            elif file_info.filename.startswith("extra_files"):
                # 处理额外的文件，提取文件名并将内容解码为 UTF-8 存入 extra_files 字典
                filename = file_info.filename.split("/", 1)[1]
                extra_files[filename] = file_content.decode('utf-8')
        
        # 确保所有必要的序列化数据都已经被读取
        assert serialized_exported_program is not None
        assert serialized_state_dict is not None
        assert serialized_constants is not None
        assert serialized_example_inputs is not None
        
        # 使用 SerializedArtifact 类创建 artifact 对象
        artifact: SerializedArtifact = SerializedArtifact(
            serialized_exported_program,
            serialized_state_dict,
            serialized_constants,
            serialized_example_inputs,
        )
        
        # 反序列化 ExportedProgram，得到 ep 对象
        ep = deserialize(artifact, expected_opset_version)
        
        # 返回反序列化后的 ExportedProgram 对象
        return ep
# 编译函数，将 nn.Module 的 forward 函数或包含 PyTorch 操作的可调用对象进行跟踪，生成可执行的 cpp 代码，并返回生成的共享库的路径
def aot_compile(
    f: Callable,  # 要跟踪的 nn.Module 或可调用对象
    args: Tuple[Any],  # 示例的位置输入
    kwargs: Optional[Dict[str, Any]] = None,  # 可选的关键字输入
    *,
    dynamic_shapes: Optional[Dict[str, Any]] = None,  # 动态形状的规范
    options: Optional[Dict[str, Any]] = None,  # 控制编译器的选项
    remove_runtime_assertions: bool = False,  # 是否移除运行时断言
    disable_constraint_solver: bool = False,  # 是否禁用约束求解器
    same_signature: bool = True,  # 是否保持相同的签名
) -> str:  # 返回生成的共享库的路径
    """
    Note: this function is not stable yet

    Traces either an nn.Module's forward function or just a callable with PyTorch
    operations inside, generates executable cpp code from the program, and returns
    the path to the generated shared library

    Args:
        f: the `nn.Module` or callable to trace.

        args: example positional inputs.

        kwargs: optional example keyword inputs.

        dynamic_shapes: Should either be:
            1) a dict from argument names of ``f`` to their dynamic shape specifications,
            2) a tuple that specifies dynamic shape specifications for each input in original order.
            If you are specifying dynamism on keyword args, you will need to pass them in the order that
            is defined in the original function signature.

            The dynamic shape of a tensor argument can be specified as either
            (1) a dict from dynamic dimension indices to :func:`Dim` types, where it is
            not required to include static dimension indices in this dict, but when they are,
            they should be mapped to None; or (2) a tuple / list of :func:`Dim` types or None,
            where the :func:`Dim` types correspond to dynamic dimensions, and static dimensions
            are denoted by None. Arguments that are dicts or tuples / lists of tensors are
            recursively specified by using mappings or sequences of contained specifications.

        options: A dictionary of options to control inductor

        disable_constraint_solver: Whether the dim constraint solver must be disabled.

    Returns:
        Path to the generated shared library
    """
    # 导入所需的模块
    from torch.export._trace import _export_to_torch_ir
    from torch._inductor.decomposition import select_decomp_table

    # 如果是预分发状态，则导出模型
    if config.is_predispatch:
        gm = torch.export._trace._export(f, args, kwargs, dynamic_shapes, pre_dispatch=True).module()
    else:
        # 在这里导出到 Torch IR，以利用 inductor 中运行在 Torch IR 上的预梯度传递
        gm = _export_to_torch_ir(
            f,
            args,
            kwargs,
            dynamic_shapes,
            disable_constraint_solver=disable_constraint_solver,
            same_signature=same_signature,
            # 禁用此标志，因为我们可以依赖来自 Dynamo 的 dynamo_flat_name_to_original_fqn 映射
            restore_fqn=False,
        )

    # 禁用梯度计算，编译生成共享库
    with torch.no_grad():
        so_path = torch._inductor.aot_compile(gm, args, kwargs, options=options)  # type: ignore[arg-type]

    # 返回生成的共享库路径
    return so_path
def aot_load(so_path: str, device: str) -> Callable:
    """
    Loads a shared library generated by aot_compile and returns a callable

    Args:
        so_path: Path to the shared library
        device: Specifies the device ("cpu" or "cuda") to run the model

    Returns:
        A callable function that runs the optimized model
    """
    # 根据设备类型选择不同的 AOTIModelContainerRunner 对象
    if device == "cpu":
        runner = torch._C._aoti.AOTIModelContainerRunnerCpu(so_path, 1)  # type: ignore[call-arg]
    elif device == "cuda" or device.startswith("cuda:"):
        runner = torch._C._aoti.AOTIModelContainerRunnerCuda(so_path, 1, device)  # type: ignore[assignment, call-arg]
    else:
        # 抛出运行时异常，提示设备类型不支持
        raise RuntimeError("Unsupported device " + device)

    # 定义并返回一个优化后的可调用函数
    def optimized(*args, **kwargs):
        # 获取模型调用的规范信息
        call_spec = runner.get_call_spec()  # type: ignore[attr-defined]
        # 加载输入数据规范
        in_spec = pytree.treespec_loads(call_spec[0])
        # 加载输出数据规范
        out_spec = pytree.treespec_loads(call_spec[1])
        # 将输入参数展平，并按照输入规范重新排序关键字参数
        flat_inputs = pytree.tree_flatten((args, reorder_kwargs(kwargs, in_spec)))[0]
        # 使用 runner 对象运行模型，并获取展平后的输出
        flat_outputs = runner.run(flat_inputs)  # type: ignore[attr-defined]
        # 根据输出规范，重新构建输出数据结构，并返回
        return pytree.tree_unflatten(flat_outputs, out_spec)

    return optimized
```