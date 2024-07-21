# `.\pytorch\torch\onnx\_internal\exporter.py`

```py
# mypy: allow-untyped-defs
from __future__ import (  # 允许未标注的函数定义（用于 onnx.ModelProto（ONNX 程序）和 onnxruntime（ONNXRuntimeOptions））
    annotations,
)

import abc  # 引入抽象基类模块

import contextlib  # 引入上下文管理模块
import dataclasses  # 引入数据类模块
import io  # 引入 I/O 模块
import logging  # 引入日志模块
import os  # 引入操作系统功能模块

import tempfile  # 引入临时文件模块
import warnings  # 引入警告模块
from collections import defaultdict  # 从 collections 模块中引入 defaultdict
from typing import (  # 引入类型提示模块，包括多种类型
    Any,
    Callable,
    Dict,
    Final,
    List,
    Mapping,
    Optional,
    Protocol,
    runtime_checkable,
    Sequence,
    Set,
    Tuple,
    TYPE_CHECKING,
    TypeVar,
    Union,
)
from typing_extensions import Self  # 从 typing_extensions 模块中引入 Self 类型

import torch  # 引入 PyTorch 模块

import torch._ops  # 引入 PyTorch 的操作模块
import torch.export as torch_export  # 引入 PyTorch 导出模块
import torch.utils._pytree as pytree  # 引入 PyTorch 工具中的 _pytree 模块
from torch._subclasses import fake_tensor  # 从 torch._subclasses 模块中引入 fake_tensor

from torch.onnx._internal import _beartype, io_adapter  # 从 torch.onnx._internal 模块中引入 _beartype 和 io_adapter
from torch.onnx._internal.diagnostics import infra  # 从 torch.onnx._internal.diagnostics 模块中引入 infra
from torch.onnx._internal.fx import (  # 从 torch.onnx._internal.fx 模块中引入多个子模块
    decomposition_table,
    patcher as patcher,
    registration,
    serialization as fx_serialization,
)

# 我们只能在类型检查的情况下从这个模块导入 onnx，以确保在没有安装 'onnx' 的情况下继续工作。我们完全在 dynamo_export 中导入 'onnx'（通过 _assert_dependencies）。
if TYPE_CHECKING:
    import onnx  # 引入 onnx 模块
    import onnxruntime  # 引入 onnxruntime 模块（忽略导入错误）
    import onnxscript  # 引入 onnxscript 模块（忽略导入错误）
    from onnxscript.function_libs.torch_lib import (  # 从 onnxscript.function_libs.torch_lib 模块中引入 registration
        registration as torchlib_registry,
    )

    from torch.onnx._internal.fx import diagnostics  # 从 torch.onnx._internal.fx 模块中引入 diagnostics
else:
    try:
        # 由于运行时类型检查，beartype 需要此导入。由于 https://github.com/pytorch/pytorch/issues/103764，这不能正常地在顶层导入。
        from torch.onnx._internal.fx import diagnostics
    except ImportError:
        # 如果导入失败，错误将在导出器使用时处理。
        pass

_DEFAULT_OPSET_VERSION: Final[int] = 18
"""导出器将使用的默认 ONNX opset 版本，如果未通过 :class:`ExportOptions` 显式指定，则使用此版本。在此模块外部绝对不应访问此变量！"""

_PYTORCH_GITHUB_ISSUES_URL = "https://github.com/pytorch/pytorch/issues"
"""指向 PyTorch GitHub 问题页面的 URL。"""

_DEFAULT_FAILED_EXPORT_SARIF_LOG_PATH = "report_dynamo_export.sarif"
"""如果导出失败，写入 SARIF 日志的默认路径。"""

_PROTOBUF_SIZE_MAX_LIMIT = 2 * 1024 * 1024 * 1024
"""Protobuf 文件的最大大小（以字节为单位）。这用于确定是否使用外部数据序列化模型。"""

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


DiagnosticOptions = infra.DiagnosticOptions  # 将 infra 模块中的 DiagnosticOptions 类赋值给 DiagnosticOptions


@dataclasses.dataclass
class ONNXFakeContext:
    """用于使用 FakeTensor 导出模型的上下文信息的数据类。

    此数据类存储用于将真实张量和模型参数转换为伪张量的 FakeTensorMode 实例。这个 :attr:`ONNXFakeContext.fake_mode` 是
    reused internally during tracing of a :class:`torch.nn.Module` into a FX :class:`GraphModule`.
    """
    # 在将 :class:`torch.nn.Module` 跟踪为 FX :class:`GraphModule` 过程中内部重用的状态

    fake_mode: fake_tensor.FakeTensorMode
    """The fake tensor mode used for tracing model using fake tensors and parameters."""
    # 用于使用虚拟张量和参数跟踪模型的虚拟张量模式

    state_dict_paths: Optional[Tuple[Union[str, io.BytesIO, Dict[str, Any]]]] = None
    """List of paths of files that contain the model :meth:`state_dict`"""
    # 包含模型 :meth:`state_dict` 的文件路径列表
    def _initiate_registry_from_torchlib(
        self, torchlib_registry: torchlib_registry.Registry
    ) -> None:
        """
        Initializes the registry by populating it with functions from the provided torchlib_registry.

        Args:
        - torchlib_registry: Registry containing ONNX functions from torchlib.

        This method iterates over the entries in torchlib_registry and registers each ONNXFunction
        under its corresponding OpName in the _registry dictionary.
        """
    ):
        """
        从 torchlib 中的 ATen 函数填充注册表。

        Args:
            torchlib_registry: 用于填充注册表的 torchlib 注册表。
        """
        for aten_name, aten_overloads_func in torchlib_registry.items():
            # 根据 ATen 函数名创建内部名称实例
            internal_name_instance = registration.OpName.from_qualified_name(aten_name)
            # 遍历每个 ATen 函数的重载
            for overload_func in aten_overloads_func.overloads:
                # 创建符号函数实例
                symbolic_function = registration.ONNXFunction(
                    onnx_function=overload_func,
                    op_full_name=internal_name_instance.qualified_name(),
                    is_custom=False,
                    is_complex=False,
                )
                # 注册符号函数到内部名称实例
                self._register(internal_name_instance, symbolic_function)

            # 遍历每个 ATen 函数的复杂函数
            for complex_func in aten_overloads_func.complex:
                # 创建复杂符号函数实例
                symbolic_function = registration.ONNXFunction(
                    onnx_function=complex_func,
                    op_full_name=internal_name_instance.qualified_name(),
                    is_custom=False,
                    is_complex=True,
                )
                # 注册复杂符号函数到内部名称实例
                self._register(internal_name_instance, symbolic_function)

    @_beartype.beartype
    def _register(
        self,
        internal_qualified_name: registration.OpName,
        symbolic_function: registration.ONNXFunction,
    ) -> None:
        """
        将 ONNXFunction 注册到操作符。

        Args:
            internal_qualified_name: 要注册的操作符的限定名称: OpName。
            symbolic_function: 要注册的 ONNXFunction。
        """
        self._registry[internal_qualified_name].append(symbolic_function)

    @_beartype.beartype
    def register_op(
        self,
        function: Union["onnxscript.OnnxFunction", "onnxscript.TracedOnnxFunction"],
        namespace: str,
        op_name: str,
        overload: Optional[str] = None,
        is_complex: bool = False,
    ) -> None:
        """Registers a custom operator: torch.ops.<namespace>.<op_name>.<overload>.

        Args:
            function: The onnx-sctip function to register.
                要注册的 onnx-sctip 函数。
            namespace: The namespace of the operator to register.
                要注册的操作符的命名空间。
            op_name: The name of the operator to register.
                要注册的操作符的名称。
            overload: The overload of the operator to register. If it's default overload,
                leave it to None.
                要注册的操作符的重载版本。如果是默认重载版本，则为 None。
            is_complex: Whether the function is a function that handles complex valued inputs.
                函数是否处理复数输入的函数。

        Raises:
            ValueError: If the name is not in the form of 'namespace::op'.
                如果名称不符合 'namespace::op' 的形式，则引发 ValueError。
        """
        internal_name_instance = registration.OpName.from_name_parts(
            namespace=namespace, op_name=op_name, overload=overload
        )
        symbolic_function = registration.ONNXFunction(
            onnx_function=function,
            op_full_name=internal_name_instance.qualified_name(),
            is_custom=True,
            is_complex=is_complex,
        )
        self._register(internal_name_instance, symbolic_function)

    @_beartype.beartype
    def get_op_functions(
        self, namespace: str, op_name: str, overload: Optional[str] = None
    ) -> Optional[List[registration.ONNXFunction]]:
        """Returns a list of ONNXFunctions for the given op: torch.ops.<namespace>.<op_name>.<overload>.

        The list is ordered by the time of registration. The custom operators should be
        in the second half of the list.

        Args:
            namespace: The namespace of the operator to get.
                要获取的操作符的命名空间。
            op_name: The name of the operator to get.
                要获取的操作符的名称。
            overload: The overload of the operator to get. If it's default overload,
                leave it to None.
                要获取的操作符的重载版本。如果是默认重载版本，则为 None。

        Returns:
            A list of ONNXFunctions corresponding to the given name, or None if
            the name is not in the registry.
            给定名称对应的 ONNXFunctions 列表，如果名称不在注册表中则返回 None。
        """
        internal_name_instance = registration.OpName.from_name_parts(
            namespace=namespace, op_name=op_name, overload=overload
        )
        return self._registry.get(internal_name_instance)

    @_beartype.beartype
    def is_registered_op(
        self, namespace: str, op_name: str, overload: Optional[str] = None
    ) -> bool:
        """Returns whether the given op is registered: torch.ops.<namespace>.<op_name>.<overload>.

        Args:
            namespace: The namespace of the operator to check.
                要检查的操作符的命名空间。
            op_name: The name of the operator to check.
                要检查的操作符的名称。
            overload: The overload of the operator to check. If it's default overload,
                leave it to None.
                要检查的操作符的重载版本。如果是默认重载版本，则为 None。

        Returns:
            True if the given op is registered, otherwise False.
            如果给定的操作符已注册，则返回 True；否则返回 False。
        """
        functions = self.get_op_functions(
            namespace=namespace, op_name=op_name, overload=overload
        )
        return functions is not None
    # 返回所有已注册操作的名称集合
    def _all_registered_ops(self) -> Set[str]:
        """Returns the set of all registered function names."""
        # 使用集合推导式，遍历 self._registry 字典中的所有键值，生成操作名的限定名称集合
        return {
            op_name_class.qualified_name() for op_name_class in self._registry.keys()
        }
class ExportOptions:
    """Options to influence the TorchDynamo ONNX exporter.

    Attributes:
        dynamic_shapes: Shape information hint for input/output tensors.
            When ``None``, the exporter determines the most compatible setting.
            When ``True``, all input shapes are considered dynamic.
            When ``False``, all input shapes are considered static.
        op_level_debug: Whether to export the model with op-level debug information
        diagnostic_options: The diagnostic options for the exporter.
        fake_context: The fake context used for symbolic tracing.
        onnx_registry: The ONNX registry used to register ATen operators to ONNX functions.
    """

    dynamic_shapes: Optional[bool] = None
    """Shape information hint for input/output tensors.

    - ``None``: the exporter determines the most compatible setting.
    - ``True``: all input shapes are considered dynamic.
    - ``False``: all input shapes are considered static.
    """

    op_level_debug: Optional[bool] = None
    """When True export the model with op-level debug running ops through ONNX Runtime."""

    diagnostic_options: DiagnosticOptions
    """The diagnostic options for the exporter."""

    fake_context: Optional[ONNXFakeContext] = None
    """The fake context used for symbolic tracing."""

    onnx_registry: Optional[OnnxRegistry] = None
    """The ONNX registry used to register ATen operators to ONNX functions."""

    @_beartype.beartype
    def __init__(
        self,
        *,
        dynamic_shapes: Optional[bool] = None,
        op_level_debug: Optional[bool] = None,
        fake_context: Optional[ONNXFakeContext] = None,
        onnx_registry: Optional[OnnxRegistry] = None,
        diagnostic_options: Optional[DiagnosticOptions] = None,
    ):
        # Initialize ExportOptions object with specified or default values
        self.dynamic_shapes = dynamic_shapes
        self.op_level_debug = op_level_debug
        self.fake_context = fake_context
        self.onnx_registry = onnx_registry
        # Use specified DiagnosticOptions or default to a new instance
        self.diagnostic_options = diagnostic_options or DiagnosticOptions()


class ResolvedExportOptions(ExportOptions):
    """Consolidates :class:`ExportOptions` with default values.
    All unspecified options from :class:`ExportOptions` are assigned a default value.
    This is an internal class and its API may be changed at any time without notice.
    """

    # Public attributes MUST be redefined below without ``Optional[]`` from ``ExportOptions``
    dynamic_shapes: bool
    op_level_debug: bool
    diagnostic_options: DiagnosticOptions
    fake_context: ONNXFakeContext
    onnx_registry: OnnxRegistry

    decomposition_table: Dict[torch._ops.OpOverload, Callable]
    """A dictionary that maps operators to their decomposition functions."""

    onnxfunction_dispatcher: (
        torch.onnx._internal.fx.onnxfunction_dispatcher.OnnxFunctionDispatcher
    )
    """The ONNX dispatcher used to dispatch ATen operators to ONNX functions."""

    fx_tracer: FXGraphExtractor
    """FXGraphExtractor instance used for tracing FX graphs."""
    """The FXGraphExtractor instance used to extract the FX graph from the model."""
    
    diagnostic_context: diagnostics.DiagnosticContext
    """The diagnostics context for the export. Responsible for recording diagnostics,
    logging diagnostics, and generating the SARIF log."""
    
    @_beartype.beartype
    def __init__(
        self,
        options: Union[ExportOptions, "ResolvedExportOptions"],
        model: Optional[Union[torch.nn.Module, Callable, torch_export.ExportedProgram]] = None,  # type: ignore[name-defined]
@contextlib.contextmanager
def enable_fake_mode():
    """
    Enable fake mode for the duration of the context.

    Internally it instantiates a :class:`torch._subclasses.fake_tensor.FakeTensorMode` context manager
    that converts user input and model parameters into :class:`torch._subclasses.fake_tensor.FakeTensor`.

    A :class:`torch._subclasses.fake_tensor.FakeTensor`
    is a :class:`torch.Tensor` with the ability to run PyTorch code without having to
    actually do computation through tensors allocated on a ``meta`` device. Because
    there is no actual data being allocated on the device, this API allows for
    exporting large models without the actual memory footprint needed for executing it.

    It is highly recommended to enable fake mode when exporting models that
    are too large to fit into memory.

    Returns:
        A :class:`ONNXFakeContext` object that must be passed to :func:`dynamo_export`
        through the :attr:`ExportOptions.fake_context` argument.

    Example::

        # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX)
        >>> import torch
        >>> import torch.onnx
        >>> class MyModel(torch.nn.Module):  # Dummy model
        ...     def __init__(self) -> None:
        ...         super().__init__()
        ...         self.linear = torch.nn.Linear(2, 2)
        ...     def forward(self, x):
        ...         out = self.linear(x)
        ...         return out
        >>> with torch.onnx.enable_fake_mode() as fake_context:
        ...     my_nn_module = MyModel()
        ...     arg1 = torch.randn(2, 2, 2)  # positional input 1
        >>> export_options = torch.onnx.ExportOptions(fake_context=fake_context)
        >>> onnx_program = torch.onnx.dynamo_export(
        ...     my_nn_module,
        ...     arg1,
        ...     export_options=export_options
        ... )
        >>> # Saving model WITHOUT initializers
        >>> onnx_program.save("my_model_without_initializers.onnx")
        >>> # Saving model WITH initializers
        >>> onnx_program.save("my_model_with_initializers.onnx", model_state=MyModel().state_dict())

    .. warning::
        This API is experimental and is *NOT* backward-compatible.

    """
    from torch._subclasses import fake_tensor  # 导入 torch._subclasses 模块中的 fake_tensor
    from torch.fx.experimental.symbolic_shapes import ShapeEnv  # 导入 torch.fx.experimental.symbolic_shapes 模块中的 ShapeEnv

    # This overrides the internal `FakeTensorMode` instance created by `torch._dynamo.export`[1].
    # It is a good idea to keep them in sync (constructor args) to maintain the same default behavior
    # [1] `torch/_dynamo/output_graph.py::InstructionTranslator::OutputGraph.__init__`
    # Mixed fake/real tensors are only allowed when `torch.onnx.dynamo_export` is not called within `FakeTensorMode`
    # This is needed because models can create new parameters during `forward(self, *args, **kwargs)` run
    ```
    这里的注释解释了为什么需要确保 `torch.onnx.dynamo_export` 不在 `FakeTensorMode` 中调用，
    以及如何维护 `FakeTensorMode` 的实例和行为的同步。
    ```py
    # 创建 FakeTensorMode 对象，用于控制虚拟张量的行为
    fake_mode = fake_tensor.FakeTensorMode(
        allow_non_fake_inputs=not torch._guards.detect_fake_mode(),
        shape_env=ShapeEnv(
            allow_scalar_outputs=False, allow_dynamic_output_shape_ops=False
        ),
    )
    # 创建 ONNXTorchPatcher 对象，用于在虚拟模式下加载模型状态字典时进行补丁操作
    patcher_context = patcher.ONNXTorchPatcher()
    # 创建 ONNXFakeContext 对象，用于管理 ONNX 模型在虚拟模式下的上下文
    fake_context = ONNXFakeContext(fake_mode=fake_mode)
    # 进入虚拟模式和补丁器上下文，确保加载模型状态时能够正常运行
    with fake_mode, patcher_context:
        # 返回 ONNXFakeContext 对象，作为上下文管理器的结果
        yield fake_context
    # 将补丁器上下文中的路径元组赋给 fake_context 的 state_dict_paths 属性
    fake_context.state_dict_paths = tuple(
        patcher_context.paths,
    )  # type: ignore[assignment]
@runtime_checkable
class ONNXProgramSerializer(Protocol):
    """Protocol for serializing an ONNX graph into a specific format (e.g. Protobuf).
    Note that this is an advanced usage scenario."""

    def serialize(
        self, onnx_program: ONNXProgram, destination: io.BufferedIOBase
    ) -> None:
        """Protocol method that must be implemented for serialization.

        Args:
            onnx_program: Represents the in-memory exported ONNX model
            destination: A binary IO stream or pre-allocated buffer into which
                the serialized model should be written.

        Example:

            A simple serializer that writes the exported :py:obj:`onnx.ModelProto` in Protobuf
            format to ``destination``:

            ::

                # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX)
                >>> import io
                >>> import torch
                >>> import torch.onnx
                >>> class MyModel(torch.nn.Module):  # Dummy model
                ...     def __init__(self) -> None:
                ...         super().__init__()
                ...         self.linear = torch.nn.Linear(2, 2)
                ...     def forward(self, x):
                ...         out = self.linear(x)
                ...         return out
                >>> class ProtobufONNXProgramSerializer:
                ...     def serialize(
                ...         self, onnx_program: torch.onnx.ONNXProgram, destination: io.BufferedIOBase
                ...     ) -> None:
                ...         destination.write(onnx_program.model_proto.SerializeToString())
                >>> model = MyModel()
                >>> arg1 = torch.randn(2, 2, 2)  # positional input 1
                >>> torch.onnx.dynamo_export(model, arg1).save(
                ...     destination="exported_model.onnx",
                ...     serializer=ProtobufONNXProgramSerializer(),
                ... )
        """
        # Placeholder for implementation; this method must be implemented by classes conforming to this protocol.
        ...


class ProtobufONNXProgramSerializer:
    """Serializes ONNX graph as Protobuf."""

    @_beartype.beartype
    def serialize(
        self, onnx_program: ONNXProgram, destination: io.BufferedIOBase
    ) -> None:
        """Serialize method for ProtobufONNXProgramSerializer.

        Args:
            onnx_program: Represents the in-memory exported ONNX model
            destination: A binary IO stream or pre-allocated buffer into which
                the serialized model should be written.
        """
        import onnx

        # Check if onnx_program.model_proto is an instance of onnx.ModelProto
        if not isinstance(onnx_program.model_proto, onnx.ModelProto):  # type: ignore[attr-defined]
            raise ValueError("onnx_program.ModelProto is not an onnx.ModelProto")
        
        # Serialize the model_proto to destination as a string
        destination.write(onnx_program.model_proto.SerializeToString())


class LargeProtobufONNXProgramSerializer:
    """Serializes ONNX graph as Protobuf.

    Fallback to serializing as Protobuf with external data for models larger than 2GB.
    """

    _destination_path: Final[str]  # type: ignore[misc]

    def __init__(self, destination_path: str):
        self._destination_path = destination_path

    @_beartype.beartype
    def serialize(
        self, onnx_program: ONNXProgram, destination: io.BufferedIOBase
    ) -> None:
        """Serialize method for LargeProtobufONNXProgramSerializer.

        Args:
            onnx_program: Represents the in-memory exported ONNX model
            destination: A binary IO stream or pre-allocated buffer into which
                the serialized model should be written.
        """
        # Implementation would handle large model serialization, not fully provided in the snippet
    ) -> None:
        """`destination` is ignored. The model is saved to `self._destination_path` instead."""
        # 导入onnx模块，用于操作ONNX模型
        import onnx

        # 检查ONNX模型的Proto对象大小是否小于最大限制
        if onnx_program.model_proto.ByteSize() < _PROTOBUF_SIZE_MAX_LIMIT:
            # 如果小于限制，将ONNX模型保存到指定路径self._destination_path
            onnx.save_model(onnx_program.model_proto, self._destination_path)  # type: ignore[attr-defined]
        else:
            # 如果超过限制，抛出异常并使用外部数据序列化模型
            # ValueError: Message onnx.ModelProto exceeds maximum protobuf size of 2GB
            # 以外部数据的形式保存模型
            onnx.save_model(  # type: ignore[attr-defined]
                onnx_program.model_proto,
                self._destination_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
            )
class ONNXRuntimeOptions:
    """Options to influence the execution of the ONNX model through ONNX Runtime.

    Attributes:
        session_options: ONNX Runtime session options.
        execution_providers: ONNX Runtime execution providers to use during model execution.
        execution_provider_options: ONNX Runtime execution provider options.
    """

    session_options: Optional[Sequence["onnxruntime.SessionOptions"]] = None
    """ONNX Runtime session options."""

    execution_providers: Optional[
        Sequence[Union[str, Tuple[str, Dict[Any, Any]]]]
    ] = None
    """ONNX Runtime execution providers to use during model execution."""

    execution_provider_options: Optional[Sequence[Dict[Any, Any]]] = None
    """ONNX Runtime execution provider options."""

    @_beartype.beartype
    def __init__(
        self,
        *,
        session_options: Optional[Sequence["onnxruntime.SessionOptions"]] = None,
        execution_providers: Optional[
            Sequence[Union[str, Tuple[str, Dict[Any, Any]]]]
        ] = None,
        execution_provider_options: Optional[Sequence[Dict[Any, Any]]] = None,
    ):
        self.session_options = session_options  # 设置 ONNX 运行时的会话选项
        self.execution_providers = execution_providers  # 设置 ONNX 运行时的执行提供者
        self.execution_provider_options = execution_provider_options  # 设置 ONNX 运行时的执行提供者选项
    # 初始化方法，用于创建一个新的实例
    def __init__(
        self,
        model_proto: onnx.ModelProto,  # 模型的 ONNX 表示
        input_adapter: io_adapter.InputAdapter,  # 输入适配器对象
        output_adapter: io_adapter.OutputAdapter,  # 输出适配器对象
        diagnostic_context: diagnostics.DiagnosticContext,  # 诊断上下文对象
        *,
        fake_context: Optional[ONNXFakeContext] = None,  # 可选的虚拟上下文对象
        export_exception: Optional[Exception] = None,  # 可选的导出异常对象
        model_signature: Optional[torch.export.ExportGraphSignature] = None,  # 可选的模型签名对象
        model_torch: Optional[
            Union[torch.nn.Module, Callable, torch_export.ExportedProgram]
        ] = None,  # 可选的 Torch 模型对象
    ):
        self._model_proto = model_proto  # 设置模型的 ONNX 表示
        self._model_signature = model_signature  # 设置模型签名对象
        self._model_torch = model_torch  # 设置 Torch 模型对象
        self._input_adapter = input_adapter  # 设置输入适配器对象
        self._output_adapter = output_adapter  # 设置输出适配器对象
        self._diagnostic_context = diagnostic_context  # 设置诊断上下文对象
        self._fake_context = fake_context  # 设置虚拟上下文对象
        self._export_exception = export_exception  # 设置导出异常对象

    # 调用方法，允许将实例当作函数调用
    def __call__(
        self,
        *args: Any,
        model_with_state_dict: Optional[
            Union[torch.nn.Module, Callable, torch_export.ExportedProgram]
        ] = None,  # 可选的带状态字典的模型对象
        options: Optional[ONNXRuntimeOptions] = None,  # 可选的 ONNX 运行时选项
        **kwargs: Any,
    ) -> Any:
        """Runs the ONNX model using ONNX Runtime

        Args:
            args: The positional inputs to the model.
            kwargs: The keyword inputs to the model.
            model_with_state_dict: The PyTorch model to fetch state from.
                Required when :func:`enable_fake_mode` is used to extract real initializers as needed by the ONNX graph.
            options: The options to use for running the model with ONNX Runtime.

        Returns:
            The model output as computed by ONNX Runtime
        """

        # TODO: If ONNX used absolute paths on the initializers external data files,
        # users could call ONNXProgram.save and use ONNXProgram.__call__ without the internal save below

        with contextlib.ExitStack() as stack:
            # model specified by the user has precedence, when specified
            model_with_state_dict = model_with_state_dict or self._model_torch

            if self.fake_context:
                # If fake_context is enabled, prepare a temporary directory to save the model
                tmpdir_path = stack.enter_context(tempfile.TemporaryDirectory())
                warnings.warn(
                    "Cannot run model directly from `ONNXProgram` because"
                    " the model was exported using `enable_fake_mode`."
                    " The model will be serialized to disk using a temporary folder ({tmpdir_path})"
                    " to populate the model with initializers before being execution."
                )
                # Determine the path where the ONNX model will be saved
                onnx_model = os.path.join(tmpdir_path, "model.onnx")
                # Determine the state dictionary of the model based on its type
                if isinstance(model_with_state_dict, torch.nn.Module):
                    model_state = model_with_state_dict.state_dict()
                elif isinstance(model_with_state_dict, torch_export.ExportedProgram):
                    model_state = model_with_state_dict.state_dict
                else:
                    model_state = None
                # Save the ONNX model to disk
                self.save(
                    onnx_model,
                    model_state=model_state,
                )
            else:
                # If fake_context is not enabled, use the serialized model from model_proto
                onnx_model = self.model_proto.SerializeToString()  # type: ignore[assignment]

            import onnxruntime  # type: ignore[import]

            # Adapt Torch inputs to ONNX format
            onnx_input = self.adapt_torch_inputs_to_onnx(
                *args, model_with_state_dict=model_with_state_dict, **kwargs
            )
            # Configure ONNX Runtime options
            options = options or ONNXRuntimeOptions()
            providers = (
                options.execution_providers or onnxruntime.get_available_providers()
            )
            # Create an ONNX Runtime inference session
            ort_session = onnxruntime.InferenceSession(onnx_model, providers=providers)

            # Prepare input data for ONNX Runtime session
            onnxruntime_input = {
                k.name: v.numpy(force=True)
                for k, v in zip(ort_session.get_inputs(), onnx_input)
            }

            # Execute the ONNX model and return the output
            return ort_session.run(None, onnxruntime_input)

    @property


这段代码主要是一个方法，用于使用ONNX Runtime运行ONNX模型。代码中涉及了模型的加载、输入数据的适配以及模型执行过程中的一些准备工作。
    # 返回属性值，表示导出的 ONNX 模型作为 onnx.ModelProto 类型
    def model_proto(self) -> onnx.ModelProto:  # type: ignore[name-defined]
        """The exported ONNX model as an :py:obj:`onnx.ModelProto`."""

        # 如果存在导出异常，则抛出异常
        if self._export_exception is not None:
            raise self._export_exception
        # 返回私有属性 _model_proto，即导出的 ONNX 模型
        return self._model_proto

    # 返回属性值，表示与导出相关联的诊断上下文
    @property
    def diagnostic_context(self) -> diagnostics.DiagnosticContext:
        """The diagnostic context associated with the export."""

        # 返回私有属性 _diagnostic_context，即导出相关的诊断上下文
        return self._diagnostic_context

    # 返回属性值，表示与导出相关联的虚拟上下文
    @property
    def fake_context(self) -> Optional[ONNXFakeContext]:
        """The fake context associated with the export."""

        # 返回私有属性 _fake_context，即导出相关的虚拟上下文
        return self._fake_context

    # 装饰器函数，用于适配将 Torch 模型输入转换为 ONNX 格式
    @_beartype.beartype
    def adapt_torch_inputs_to_onnx(
        self,
        *model_args,
        model_with_state_dict: Optional[
            Union[torch.nn.Module, Callable, torch_export.ExportedProgram]
        ] = None,
        **model_kwargs,
    ) -> Sequence[Union[torch.Tensor, int, float, bool, torch.dtype]]:
        """Converts the PyTorch model inputs to exported ONNX model inputs format.

        Due to design differences, input/output format between PyTorch model and exported
        ONNX model are often not the same. E.g., None is allowed for PyTorch model, but are
        not supported by ONNX. Nested constructs of tensors are allowed for PyTorch model,
        but only flattened tensors are supported by ONNX, etc.

        The actual adapting steps are associated with each individual export. It
        depends on the PyTorch model, the particular set of model_args and model_kwargs
        used for the export, and export options.

        This method replays the adapting steps recorded during export.

        Args:
            model_args: The PyTorch model inputs.
            model_with_state_dict: The PyTorch model to get extra state from.
                If not specified, the model used during export is used.
                Required when :func:`enable_fake_mode` is used to extract real initializers as needed by the ONNX graph.
            model_kwargs: The PyTorch model keyword inputs.

        Returns:
            A sequence of tensors converted from PyTorch model inputs.

        Example::

            # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX)
            >>> import torch
            >>> import torch.onnx
            >>> from typing import Dict, Tuple
            >>> def func_nested_input(
            ...     x_dict: Dict[str, torch.Tensor],
            ...     y_tuple: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            ... ):
            ...     if "a" in x_dict:
            ...         x = x_dict["a"]
            ...     elif "b" in x_dict:
            ...         x = x_dict["b"]
            ...     else:
            ...         x = torch.randn(3)
            ...
            ...     y1, (y2, y3) = y_tuple
            ...
            ...     return x + y1 + y2 + y3
            >>> x_dict = {"a": torch.tensor(1.)}
            >>> y_tuple = (torch.tensor(2.), (torch.tensor(3.), torch.tensor(4.)))
            >>> onnx_program = torch.onnx.dynamo_export(func_nested_input, x_dict, y_tuple)
            >>> print(x_dict, y_tuple)
            {'a': tensor(1.)} (tensor(2.), (tensor(3.), tensor(4.)))
            >>> print(onnx_program.adapt_torch_inputs_to_onnx(x_dict, y_tuple, model_with_state_dict=func_nested_input))
            (tensor(1.), tensor(2.), tensor(3.), tensor(4.))

        .. warning::
            This API is experimental and is *NOT* backward-compatible.

        """
        # model specified by the user has precedence, when specified
        model_with_state_dict = model_with_state_dict or self._model_torch
        assert (
            model_with_state_dict is not None
        ), "model_with_state_dict must be specified."
        # 使用输入适配器对模型输入进行转换，返回转换后的结果
        return self._input_adapter.apply(
            *model_args, model=model_with_state_dict, **model_kwargs
        )
    # 使用 @_beartype 装饰器对 adapt_torch_outputs_to_onnx 方法进行类型检查和验证
    @_beartype.beartype
    # 将 PyTorch 模型输出适配为导出的 ONNX 模型输出格式
    def adapt_torch_outputs_to_onnx(
        self,
        model_outputs: Any,
        model_with_state_dict: Optional[
            Union[torch.nn.Module, Callable, torch_export.ExportedProgram]
        ] = None,
    ) -> Sequence[Union[torch.Tensor, int, float, bool]]:
        """Converts the PyTorch model outputs to exported ONNX model outputs format.
    
        Due to design differences, input/output format between PyTorch model and exported
        ONNX model are often not the same. E.g., None is allowed for PyTorch model, but are
        not supported by ONNX. Nested constructs of tensors are allowed for PyTorch model,
        but only flattened tensors are supported by ONNX, etc.
    
        The actual adapting steps are associated with each individual export. It
        depends on the PyTorch model, the particular set of model_args and model_kwargs
        used for the export, and export options.
    
        This method replays the adapting steps recorded during export.
    
        Args:
            model_outputs: The PyTorch model outputs.
            model_with_state_dict: The PyTorch model to get extra state from.
                If not specified, the model used during export is used.
                Required when :func:`enable_fake_mode` is used to extract real initializers as needed by the ONNX graph.
    
        Returns:
            PyTorch model outputs in exported ONNX model outputs format.
    
        Example::
    
            # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX)
            >>> import torch
            >>> import torch.onnx
            >>> def func_returning_tuples(x, y, z):
            ...     x = x + y
            ...     y = y + z
            ...     z = x + y
            ...     return (x, (y, z))
            >>> x = torch.tensor(1.)
            >>> y = torch.tensor(2.)
            >>> z = torch.tensor(3.)
            >>> onnx_program = torch.onnx.dynamo_export(func_returning_tuples, x, y, z)
            >>> pt_output = func_returning_tuples(x, y, z)
            >>> print(pt_output)
            (tensor(3.), (tensor(5.), tensor(8.)))
            >>> print(onnx_program.adapt_torch_outputs_to_onnx(pt_output, model_with_state_dict=func_returning_tuples))
            [tensor(3.), tensor(5.), tensor(8.)]
    
        .. warning::
            This API is experimental and is *NOT* backward-compatible.
    
        """
        # 当用户指定了 model_with_state_dict 参数时，优先使用用户指定的模型
        model_with_state_dict = model_with_state_dict or self._model_torch
        # 确保 model_with_state_dict 参数不为 None，必须指定模型
        assert (
            model_with_state_dict is not None
        ), "model_with_state_dict must be specified."
        # 调用 _output_adapter 对象的 apply 方法，将模型输出适配为 ONNX 模型输出格式
        return self._output_adapter.apply(model_outputs, model=model_with_state_dict)
    @_beartype.beartype
    def save_diagnostics(self, destination: str) -> None:
        """保存导出诊断信息为 SARIF 日志到指定的目标路径。

        Args:
            destination: 要保存诊断 SARIF 日志的目标路径。
                必须以 `.sarif` 扩展名结尾。

        Raises:
            ValueError: 如果目标路径不以 `.sarif` 扩展名结尾。
        """
        if not destination.endswith(".sarif"):
            message = f"'destination' must have a .sarif extension, got {destination}"
            log.fatal(message)
            raise ValueError(message)

        self.diagnostic_context.dump(destination)

    @classmethod
    def _from_failure(
        cls,
        export_exception: Exception,
        diagnostic_context: diagnostics.DiagnosticContext,
    ) -> Self:
        """
        当导出过程遇到失败时，创建一个 :class:`ONNXProgram` 实例。

        在导出失败时，使用此方法将异常和相关的诊断上下文封装在一个 :class:`ONNXProgram` 实例中，
        以便更轻松地处理和调试。

        Args:
            export_exception: 导出过程中引发的异常。
            diagnostic_context: 导出过程中的诊断上下文。

        Returns:
            表示失败的 ONNX 程序的 :class:`ONNXProgram` 实例。
        """
        # 将 `import onnx` 推迟到 `import torch` 路径之外
        # https://github.com/pytorch/pytorch/issues/103764
        import onnx

        # TODO: 是否应该使用更多信息填充 ONNXProgram，如 _model_torch，以便更容易调试？
        return ONNXProgram(
            onnx.ModelProto(),  # type: ignore[attr-defined]
            io_adapter.InputAdapter(),
            io_adapter.OutputAdapter(),
            diagnostic_context,
            export_exception=export_exception,
        )
class FXGraphExtractor(abc.ABC):
    """Abstract interface for FX graph extractor engines.
    This class isolates FX extraction logic from the rest of the export logic.
    That allows a single ONNX exporter that can leverage different FX graphs."""

    def __init__(self) -> None:
        super().__init__()
        # 初始化输入适配器和输出适配器
        self.input_adapter: io_adapter.InputAdapter = io_adapter.InputAdapter()
        self.output_adapter: io_adapter.OutputAdapter = io_adapter.OutputAdapter()

    @abc.abstractmethod
    def generate_fx(
        self,
        options: ResolvedExportOptions,
        model: Union[torch.nn.Module, Callable],
        model_args: Sequence[Any],
        model_kwargs: Mapping[str, Any],
    ) -> torch.fx.GraphModule:
        """Analyzes user ``model`` and generates a FX graph.
        Args:
            options: The export options.
            model: The user model.
            model_args: The model's positional input arguments.
            model_kwargs: The model's keyword input arguments.
        Returns:
            The generated FX Graph.
        """
        ...

    # TODO: Design the passes API
    @abc.abstractmethod
    def pre_export_passes(
        self,
        options: ResolvedExportOptions,
        original_model: Union[torch.nn.Module, Callable],
        fx_module: torch.fx.GraphModule,
        fx_module_args: Sequence[Any],
    ):
        """Applies pre-export passes to the FX graph.

        Pre-export passes are FX-to-FX graph transformations that make the graph
        more palatable for the FX-to-ONNX conversion.
        For example, it can be used to flatten model input/output, add explicit
        casts to the graph, replace/decompose operators, functionalize the graph, etc.
        """
        ...


class Exporter:
    @_beartype.beartype
    def __init__(
        self,
        options: ResolvedExportOptions,
        model: Union[torch.nn.Module, Callable, torch_export.ExportedProgram],
        model_args: Sequence[Any],
        model_kwargs: Mapping[str, Any],
    ):
        # 存储传入的导出选项
        self.options = options
        # 断言选项对象不为空
        assert self.options is not None

        # 存储传入的模型、模型参数和模型关键字参数
        self.model = model
        self.model_args = model_args
        self.model_kwargs = model_kwargs

        # TODO: https://github.com/pytorch/pytorch/issues/107714
        # NOTE: FXSymbolicTracer would fail in this assert, as it does not use `enable_fake_mode`
        from torch.onnx._internal.fx import fx_symbolic_graph_extractor

        # 检查选项中的 FX tracer 是否为 FXSymbolicTracer 类型
        if not isinstance(
            self.options.fx_tracer, fx_symbolic_graph_extractor.FXSymbolicTracer
        ):
            # 如果不是，则调用断言假张量模式的方法
            self._assert_fake_tensor_mode()
    # 断言模型和其输入不包含虚假张量的情况

    # Case 1: 模型有虚假输入/权重但未启用虚假模式的情况
    # 检查模型参数和关键字参数中是否存在任何虚假张量
    has_any_fake_tensor = pytree.tree_any(
        lambda x: isinstance(x, torch._subclasses.FakeTensor),
        (self.model_args, self.model_kwargs),
    )
    has_any_fake_param_or_buffer = False
    if isinstance(self.model, torch.nn.Module):
        # 检查模型的参数和缓冲区中是否存在任何虚假张量
        has_any_fake_param_or_buffer = pytree.tree_any(
            lambda x: isinstance(x, torch._subclasses.FakeTensor),
            (self.model.parameters(), self.model.buffers()),
        )
    # 如果存在任何虚假张量，并且未启用虚假上下文，则引发运行时错误
    if (
        has_any_fake_tensor or has_any_fake_param_or_buffer
    ) and not self.options.fake_context:
        raise RuntimeError(
            "Cannot export a model with fake inputs/weights without enabling fake mode.",
        )

    # Case 2: 模型有非虚假输入/权重并启用虚假模式的情况
    # 检查模型参数和关键字参数中是否存在任何非虚假张量
    has_any_non_fake_tensors = pytree.tree_any(
        lambda x: isinstance(x, torch.Tensor)
        and not isinstance(x, torch._subclasses.FakeTensor),
        (self.model_args, self.model_kwargs),
    )
    has_any_non_fake_param_or_buffer = False
    if isinstance(self.model, torch.nn.Module):
        # 检查模型的参数和缓冲区中是否存在任何非虚假张量
        has_any_non_fake_param_or_buffer = pytree.tree_any(
            lambda x: isinstance(x, torch.Tensor)
            and not isinstance(x, torch._subclasses.FakeTensor),
            (self.model.parameters(), self.model.buffers()),
        )
    # 如果存在任何非虚假张量，并且已启用虚假上下文，则引发运行时错误
    if (
        has_any_non_fake_tensors or has_any_non_fake_param_or_buffer
    ) and self.options.fake_context:
        raise RuntimeError(
            "Cannot export a model with non fake inputs/weights and enabled fake mode.",
        )
class UnsatisfiedDependencyError(RuntimeError):
    """Raised when an ONNX exporter dependency cannot be satisfied."""

    def __init__(self, package_name: str, message: str):
        super().__init__(message)
        self.package_name = package_name


class OnnxExporterError(RuntimeError):
    """Raised when an ONNX exporter error occurs.

    This exception is thrown when there's an error during the ONNX export process.
    It encapsulates the :class:`ONNXProgram` object generated until the failure, allowing
    access to the partial export results and associated metadata.
    """

    onnx_program: Final[ONNXProgram]  # type: ignore[misc]

    def __init__(self, onnx_program: ONNXProgram, message: str):
        """
        Initializes the OnnxExporterError with the given ONNX program and message.

        Args:
            onnx_program (ONNXProgram): The partial results of the ONNX export.
            message (str): The error message to be displayed.
        """
        super().__init__(message)
        self.onnx_program = onnx_program


class InvalidExportOptionsError(RuntimeError):
    """Raised when user specified an invalid value for the :class:`ExportOptions`."""

    pass


@_beartype.beartype
def _assert_dependencies(export_options: ResolvedExportOptions):
    opset_version = export_options.onnx_registry.opset_version

    def missing_package(package_name: str, exc_info: logging._ExcInfoType):
        message = (
            f"Please install the `{package_name}` package "
            f"(e.g. `python -m pip install {package_name}`)."
        )
        log.fatal(message, exc_info=exc_info)
        return UnsatisfiedDependencyError(package_name, message)

    def missing_opset(package_name: str):
        message = (
            f"The installed `{package_name}` does not support the specified ONNX opset "
            f"version {opset_version}. Install a newer `{package_name}` package or "
            f"specify an older opset version."
        )
        log.fatal(message)
        return UnsatisfiedDependencyError(package_name, message)

    try:
        import onnx
    except ImportError as e:
        raise missing_package("onnx", e) from e

    if onnx.defs.onnx_opset_version() < opset_version:
        raise missing_opset("onnx")

    try:
        # PyTorch runs lintrunner in CI without onnxscript installed
        import onnxscript  # type: ignore[import]
    except ImportError as e:
        raise missing_package("onnxscript", e) from e

    if not isinstance(
        onnxscript.onnx_opset.all_opsets[("", opset_version)],
        onnxscript.values.Opset,
    ):
        raise missing_opset("onnxscript")


@_beartype.beartype
def dynamo_export(
    model: Union[torch.nn.Module, Callable, torch_export.ExportedProgram],  # type: ignore[name-defined]
    /,
    *model_args,
    export_options: Optional[ExportOptions] = None,
    **model_kwargs,
) -> ONNXProgram:
    """Export a torch.nn.Module to an ONNX graph.
    
    This function exports a PyTorch neural network module to an ONNX graph representation,
    facilitating interoperability with other frameworks that support ONNX.

    Args:
        model (Union[torch.nn.Module, Callable, torch_export.ExportedProgram]): The model
            to be exported. It can be a PyTorch Module, a callable object, or an exported
            program.
        export_options (Optional[ExportOptions], optional): Options for the export process.
            Defaults to None.
        *model_args: Positional arguments to be passed to the model during export.
        **model_kwargs: Keyword arguments to be passed to the model during export.

    Returns:
        ONNXProgram: The ONNXProgram object representing the exported model in ONNX format.

    """
    # 如果提供了导出选项，则使用提供的选项；如果未提供，则创建默认选项并使用
    if export_options is not None:
        resolved_export_options = (
            export_options
            if isinstance(export_options, ResolvedExportOptions)  # 如果已经是ResolvedExportOptions类型，则直接使用
            else ResolvedExportOptions(export_options, model=model)  # 否则创建一个ResolvedExportOptions对象
        )
    else:
        resolved_export_options = ResolvedExportOptions(ExportOptions(), model=model)  # 使用默认导出选项和给定的模型创建ResolvedExportOptions对象

    # 确保所需的依赖项已经安装和配置
    _assert_dependencies(resolved_export_options)

    try:
        # 创建Exporter对象，并调用其export方法执行模型导出
        return Exporter(
            options=resolved_export_options,
            model=model,
            model_args=model_args,
            model_kwargs=model_kwargs,
        ).export()
    # 如果发生任何异常，捕获异常并执行以下代码块
    except Exception as e:
        # 将 SARIF 报告路径设置为默认的失败导出 SARIF 日志路径
        sarif_report_path = _DEFAULT_FAILED_EXPORT_SARIF_LOG_PATH
        # 将解析后的导出选项的诊断上下文转储到 SARIF 报告路径
        resolved_export_options.diagnostic_context.dump(sarif_report_path)
        # 构造失败导出到 ONNX 模型的详细错误信息
        message = (
            f"Failed to export the model to ONNX. Generating SARIF report at '{sarif_report_path}'. "
            "SARIF is a standard format for the output of static analysis tools. "
            "SARIF logs can be loaded in VS Code SARIF viewer extension, "
            "or SARIF web viewer (https://microsoft.github.io/sarif-web-component/). "
            f"Please report a bug on PyTorch Github: {_PYTORCH_GITHUB_ISSUES_URL}"
        )
        # 抛出自定义的 OnnxExporterError 异常，包含导出失败的详细信息和相关消息
        raise OnnxExporterError(
            ONNXProgram._from_failure(e, resolved_export_options.diagnostic_context),
            message,
        ) from e
# 定义函数，执行常见的导出前处理步骤
def common_pre_export_passes(
    options: ResolvedExportOptions,  # 导出选项，包含解析后的导出选项
    original_model: Union[torch.nn.Module, Callable],  # 原始模型，可以是torch.nn.Module或可调用对象
    fx_module: torch.fx.GraphModule,  # FX图模块，表示转换后的模块
    fx_module_args: Sequence[Any],  # FX模块的参数序列
):
    # TODO: 为了防止循环依赖，在此处导入必要的模块
    from torch.onnx._internal.fx import analysis, passes

    diagnostic_context = options.diagnostic_context  # 诊断上下文

    # 将分解表应用于输入图形。
    module = passes.Decompose(
        diagnostic_context,
        fx_module,
        options.decomposition_table,
        enable_dynamic_axes=options.dynamic_shapes,
        allow_fake_constant=options.fake_context is not None,
    ).run(*fx_module_args)

    # ONNX不支持视图和变异。
    # 通过功能化获得一个在语义上等效的图形，而不进行变异。
    module = passes.Functionalize(
        diagnostic_context,
        module,
        enable_dynamic_axes=options.dynamic_shapes,
        allow_fake_constant=options.fake_context is not None,
    ).run(*fx_module_args)

    # 检测并精炼输入变异，以后进行'Functionalize'处理。
    # 由于ONNX推理不需要它们，因此删除它们。
    module = passes.RemoveInputMutation(diagnostic_context, module).run(*fx_module_args)

    # ONNX不支持（隐式）类型提升的概念。
    # 在需要时显式插入类型转换。
    module = passes.InsertTypePromotion(diagnostic_context, module).run()

    # 分析不支持的FX节点，报告错误级别。
    analysis.UnsupportedFxNodesAnalysis(
        diagnostic_context, module, options.onnxfunction_dispatcher
    ).analyze(infra.levels.ERROR)

    if isinstance(original_model, torch.nn.Module):
        # 恢复参数和缓冲区名称，如果原始模型是torch.nn.Module。
        module = passes.RestoreParameterAndBufferNames(
            diagnostic_context, module, original_model
        ).run()

    # 此操作应作为最后一个导出前处理步骤调用。
    # 参见[NOTE: Modularize pass ordering]
    module = passes.Modularize(diagnostic_context, module).run()

    # ONNX不支持None输入。在图构建期间，所有None输入都被移除。
    # 在此处向输入适配器注册此步骤。
    options.fx_tracer.input_adapter.append_step(io_adapter.RemoveNoneInputStep())

    # 注意：临时解决方案，用于https://github.com/pytorch/pytorch/issues/99534
    # Dynamo不支持非张量输入。
    options.fx_tracer.input_adapter.append_step(io_adapter.RemoveNonTensorInputStep())

    # ONNX不支持复杂输入。在图构建期间，所有复杂输入都被转换为实际表示输入。
    # 在此处向输入/输出适配器注册此步骤。
    options.fx_tracer.input_adapter.append_step(
        io_adapter.ConvertComplexToRealRepresentationInputStep()
    )

    # ONNX无法表示集合类型（例如，字典，张量的元组等），我们将集合展平，并注册每个元素作为输出。
    options.fx_tracer.output_adapter.append_step(io_adapter.FlattenOutputStep())
    # 在 `FlattenOutputStep` 之后执行输出后处理步骤。
    options.fx_tracer.output_adapter.append_step(
        io_adapter.ConvertComplexToRealRepresentationOutputStep()
    )
    
    # 返回修改后的模块对象
    return module
# 定义一个列表 __all__，包含了模块中公开的所有符号名称
__all__ = [
    "DiagnosticOptions",              # 引入 DiagnosticOptions 类
    "ExportOptions",                  # 引入 ExportOptions 类
    "ONNXProgram",                    # 引入 ONNXProgram 类
    "ONNXProgramSerializer",          # 引入 ONNXProgramSerializer 类
    "ONNXRuntimeOptions",             # 引入 ONNXRuntimeOptions 类
    "InvalidExportOptionsError",      # 引入 InvalidExportOptionsError 异常类
    "OnnxExporterError",              # 引入 OnnxExporterError 异常类
    "OnnxRegistry",                   # 引入 OnnxRegistry 类
    "UnsatisfiedDependencyError",     # 引入 UnsatisfiedDependencyError 异常类
    "dynamo_export",                  # 引入 dynamo_export 函数
    "enable_fake_mode",               # 引入 enable_fake_mode 函数
]
```