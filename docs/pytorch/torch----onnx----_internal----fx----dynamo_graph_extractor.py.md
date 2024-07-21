# `.\pytorch\torch\onnx\_internal\fx\dynamo_graph_extractor.py`

```
# mypy: allow-untyped-defs
# NOTE: This file is referenced by name at
#       /opt/pytorch/torch/_dynamo/eval_frame.py::DONT_WRAP_FILES.
#       introduced by https://github.com/pytorch/pytorch/pull/98894.
#       If this file is renamed, moved, etc please update the reference there!

from __future__ import annotations

import contextlib
import functools
import inspect
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import torch._dynamo  # 导入torch._dynamo模块
import torch.export as torch_export  # 导入torch.export模块
import torch.fx  # 导入torch.fx模块
import torch.onnx  # 导入torch.onnx模块
from torch.onnx._internal import _beartype, exporter, io_adapter  # 从torch.onnx._internal模块导入指定对象
from torch.utils import _pytree as pytree  # 从torch.utils模块导入_pytree模块，重命名为pytree


class _PyTreeExtensionContext:
    """Context manager to register PyTree extension."""

    _extensions: Dict[Type, Tuple[pytree.FlattenFunc, pytree.UnflattenFunc]]  # 类属性_extensions是一个字典，键为Type类型，值为(pytree.FlattenFunc, pytree.UnflattenFunc)元组类型

    def __init__(self):
        self._extensions = {}  # 初始化空字典_extensions
        # Register PyTree extension for HuggingFace model output.
        self._register_huggingface_model_output_extension()  # 调用注册HuggingFace模型输出扩展的方法_register_huggingface_model_output_extension()

    def __enter__(self):
        for class_type, (flatten_func, unflatten_func) in self._extensions.items():
            pytree._private_register_pytree_node(
                class_type,
                flatten_func,
                unflatten_func,
            )  # 遍历_extensions字典，调用pytree._private_register_pytree_node注册PyTree节点
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for class_type in self._extensions:
            pytree.SUPPORTED_NODES.pop(class_type)  # 在退出上下文时，从pytree.SUPPORTED_NODES中移除注册的节点类型

    @_beartype.beartype
    def register_pytree_node(
        self,
        class_type: Type,  # 参数class_type的类型为Type
        flatten_func: pytree.FlattenFunc,  # 参数flatten_func的类型为pytree.FlattenFunc
        unflatten_func: pytree.UnflattenFunc,  # 参数unflatten_func的类型为pytree.UnflattenFunc
    ):
        """Register PyTree extension for a custom python type.

        Args:
            class_type: The custom python type.
            flatten_func: The flatten function.
            unflatten_func: The unflatten function.

        Raises:
            AssertionError: If the custom python type is already registered.
        """
        if class_type in pytree.SUPPORTED_NODES or class_type in self._extensions:
            # PyTree node already registered.
            # E.g., `huggingface/transformer` registers `ModelOutput` as PyTree node after
            # https://github.com/huggingface/transformers/pull/25358.
            return  # 如果class_type已经在SUPPORTED_NODES或_extensions中注册，则直接返回
        self._extensions[class_type] = (flatten_func, unflatten_func)  # 否则将class_type及其对应的(flatten_func, unflatten_func)元组添加到_extensions字典中
        # 尝试导入 'transformers' 库中的 'modeling_outputs' 模块，如果导入失败则退出函数
        try:
            from transformers import modeling_outputs  # type: ignore[import]
        except ImportError as e:
            return

        # 定义一个函数 'model_output_flatten'，接受 'ModelOutput' 类型的输出对象，并返回其值列表和上下文元组
        @_beartype.beartype
        def model_output_flatten(
            output: modeling_outputs.ModelOutput,
        ) -> Tuple[List[Any], pytree.Context]:
            return list(output.values()), (type(output), list(output.keys()))

        # 定义一个函数 'model_output_unflatten'，接受值列表和上下文元组，返回重新构造的 'ModelOutput' 对象
        @_beartype.beartype
        def model_output_unflatten(
            values: List[Any], context: pytree.Context
        ) -> modeling_outputs.ModelOutput:
            output_type, keys = context
            return output_type(**dict(zip(keys, values)))

        # 使用 'inspect.getmembers' 函数获取 'modeling_outputs' 模块中所有符合条件的类
        # 条件是：是 'ModelOutput' 的子类但不是 'ModelOutput' 本身
        named_model_output_classes = inspect.getmembers(
            modeling_outputs,
            lambda x: (
                inspect.isclass(x)
                and issubclass(x, modeling_outputs.ModelOutput)
                and x is not modeling_outputs.ModelOutput
            ),
        )

        # 遍历获取到的 'ModelOutput' 子类，并注册它们到 PyTree 节点
        for _, class_type in named_model_output_classes:
            self.register_pytree_node(
                class_type, model_output_flatten, model_output_unflatten
            )
class DynamoFlattenOutputStep(io_adapter.FlattenOutputStep):
    """Flatten nested collection and custom python types and return a flat list of elements.

    Extended from :class:`io_adapter.FlattenOutputStep` to support flattening arbitrary
    types via pytree extension. By default this supports many common user defined python
    types such as :class:`ModelOutput` from HuggingFace transformers.

    The pytree extension can be customized by passing in a ``_PyTreeExtensionContext``
    object. See :meth:`_PyTreeExtensionContext.register_pytree_node`.
    """

    def __init__(
        self, pytree_extension_context: Optional[_PyTreeExtensionContext] = None
    ):
        super().__init__()
        self._pytree_extension_context = (
            pytree_extension_context or _PyTreeExtensionContext()
        )
        # 初始化方法，接收一个可选的 _PyTreeExtensionContext 对象用于扩展 pytree
        # 如果未提供，则使用默认的 _PyTreeExtensionContext 实例

    def apply(
        self,
        model_outputs: Any,
        model: Optional[
            Union[torch.nn.Module, Callable, torch_export.ExportedProgram]
        ] = None,
    ) -> Sequence[Any]:
        """Flatten the model outputs, under the context of pytree extension."""
        with self._pytree_extension_context:
            return super().apply(model_outputs, model=model)
        # 应用方法，使用 pytree 扩展上下文来扁平化模型输出，并返回扁平化后的序列


def _wrap_model_with_output_adapter(
    model: Union[torch.nn.Module, Callable],
    output_adapter: DynamoFlattenOutputStep,
) -> Callable:
    """Wrap model with output adapter.

    This is a helper function to enable :func:`dynamo.export` on models that produce
    custom user defined types outputs. It wraps the model with an output adapter to
    convert the outputs to :func:`dynamo.export` compatible types, i.e. :class:`torch.Tensor`.

    The adapting logic is controlled by ``output_adapter``.

    Args:
        model: PyTorch model or function.
        output_adapter: Output adapter to apply to model output.
    Returns:
        Wrapped model.
    """
    model_func = model.forward if isinstance(model, torch.nn.Module) else model
    # 根据 model 的类型选择对应的 model_func

    # 保留原始函数签名。
    @functools.wraps(model_func)
    def wrapped(*args, **kwargs):
        return output_adapter.apply(model_func(*args, **kwargs), model=model)
    # 包装函数，将 model 的输出应用 output_adapter，以兼容 dynamo.export 的类型要求

    return wrapped


class DynamoExport(exporter.FXGraphExtractor):
    """Generates a FX GraphModule using torch.dynamo.export API
    Args:
        aten_graph: If True, exports a graph with ATen operators.
                    If False, exports a graph with Python operators.
    """

    def __init__(
        self,
        aten_graph: Optional[bool] = None,
    ):
        super().__init__()
        self.aten_graph = aten_graph or True
        # 初始化方法，生成一个 FX GraphModule，使用 torch.dynamo.export API
        # 可选参数 aten_graph 控制是否导出带有 ATen 操作符的图，True 表示是，False 表示否

    def generate_fx(
        self,
        options: exporter.ResolvedExportOptions,
        model: Union[torch.nn.Module, Callable],
        model_args: Sequence[Any],
        model_kwargs: Mapping[str, Any],
    ):
        """Generate a FX GraphModule."""
        ...
        # 生成 FX GraphModule 的方法，具体实现需要查看代码的其余部分
    ) -> torch.fx.GraphModule:
        # `dynamo.export` does not recognize custom user defined classes as output type.
        # Apply wrapper to adapt the outputs back to `dynamo.export` compatible types,
        # i.e. :class:`torch.Tensor`.
        # 创建一个用于扁平化输出的步骤对象
        dynamo_flatten_output_step = DynamoFlattenOutputStep()
        # 使用输出适配器包装模型，以便将输出适配到 `dynamo.export` 兼容的类型，例如 `torch.Tensor`
        wrapped_model = _wrap_model_with_output_adapter(
            model, dynamo_flatten_output_step
        )
        # 记录输出适配器步骤
        self.output_adapter.append_step(dynamo_flatten_output_step)

        # 将可调用对象转换为 FX 图形表示。
        # 根据选项选择虚拟执行模式，若动态形状选项开启则为符号模式，否则为伪模式
        fake_mode = (
            options.fake_context.fake_mode
            if options.fake_context
            else contextlib.nullcontext()
        )
        fx_mode = "symbolic" if options.dynamic_shapes else "fake"
        with fake_mode:  # type: ignore[attr-defined]
            # 使用 torch._dynamo.export 将包装后的模型导出为 FX 图形表示
            graph_module, graph_guard = torch._dynamo.export(
                wrapped_model,
                tracing_mode=fx_mode,
            )(
                *model_args,
                **model_kwargs,
            )
        del graph_guard  # 未使用的变量，删除之
        torch._dynamo.reset()

        # 将 FX 图形表示导出为 ONNX ModelProto
        self.input_adapter.append_step(
            io_adapter.FlattenInputWithTreeSpecValidationInputStep()
        )

        # 应用输入适配器，更新模型参数
        updated_model_args = self.input_adapter.apply(
            *model_args, model=model, **model_kwargs
        )

        # 执行导出前通用处理步骤，并返回结果
        return self.pre_export_passes(options, model, graph_module, updated_model_args)  # type: ignore[return-value]

    @_beartype.beartype
    def pre_export_passes(
        self,
        options: exporter.ResolvedExportOptions,
        original_model: Union[torch.nn.Module, Callable],
        fx_module: torch.fx.GraphModule,
        fx_module_args: Sequence[Any],
    ):
        # 调用 exporter.common_pre_export_passes 执行导出前的通用处理步骤，并返回结果
        return exporter.common_pre_export_passes(
            options, original_model, fx_module, fx_module_args
        )
```