# `.\pytorch\torch\onnx\_internal\io_adapter.py`

```py
# mypy: allow-untyped-defs
# 导入未定义类型检查的函数和类

from __future__ import annotations
# 使用未来版本的语法特性，以支持注解中的自引用类型

import inspect
# 导入用于检查对象的属性和方法的工具模块

from typing import (
    Any,
    Callable,
    List,
    Mapping,
    Optional,
    Protocol,
    runtime_checkable,
    Sequence,
    Tuple,
    Union,
)
# 导入用于类型注解的模块，包括多种数据类型和运行时检查支持

import torch
# 导入PyTorch库

import torch.export as torch_export
# 导入PyTorch的导出模块

from torch.onnx._internal import _beartype
# 从PyTorch的ONNX内部模块中导入_beartype

from torch.utils import _pytree as pytree
# 从PyTorch的utils模块中导入_pytree作为pytree

# TODO(bowbao): Add diagnostics for IO adapters.
# 添加对IO适配器的诊断功能的TODO注释


@runtime_checkable
# 声明一个运行时可检查的装饰器，用于声明InputAdaptStep协议的子类

class InputAdaptStep(Protocol):
    """A protocol that defines a step in the input adapting process.

    The input adapting process is a sequence of steps that are applied to the
    PyTorch model inputs to transform them into the inputs format expected by the
    exported ONNX model. Each step takes the PyTorch model inputs as arguments and
    returns the transformed inputs.

    This serves as a base formalized construct for the transformation done to model
    input signature by any individual component in the exporter.
    """
    # 输入适配过程的步骤协议，用于定义输入适配过程中的步骤

    def apply(
        self,
        model_args: Sequence[Any],
        model_kwargs: Mapping[str, Any],
        model: Optional[
            Union[torch.nn.Module, Callable, torch_export.ExportedProgram]
        ] = None,
    ) -> Tuple[Sequence[Any], Mapping[str, Any]]:
        # 定义一个抽象方法apply，接受PyTorch模型输入和关键字参数，返回转换后的输入
        ...


class InputAdapter:
    """A class that adapts the PyTorch model inputs to exported ONNX model inputs format."""

    def __init__(self, steps: Optional[List[InputAdaptStep]] = None):
        # 初始化方法，接受一个可选的InputAdaptStep类型的列表steps，默认为None
        self._steps = steps or []

    @_beartype.beartype
    # 使用_beartype模块的装饰器对下面的方法进行类型检查
    def append_step(self, step: InputAdaptStep) -> None:
        """Appends a step to the input adapt steps.

        Args:
            step: The step to append.
        """
        # 向步骤列表_steps中添加一个步骤
        self._steps.append(step)

    @_beartype.beartype
    # 使用_beartype模块的装饰器对下面的方法进行类型检查
    def apply(
        self,
        *model_args,
        model: Optional[
            Union[torch.nn.Module, Callable, torch_export.ExportedProgram]
        ] = None,
        **model_kwargs,
    ) -> Sequence[Union[int, float, bool, str, "torch.Tensor", torch.dtype, None]]:
        """Converts the PyTorch model inputs to exported ONNX model inputs format.

        Args:
            model_args: The PyTorch model inputs.
            model: The PyTorch model.
            model_kwargs: The PyTorch model keyword inputs.
        Returns:
            A sequence of tensors converted from PyTorch model inputs.
        """
        # 将PyTorch模型的输入转换为导出到ONNX模型的输入格式
        args: Sequence[Any] = model_args
        kwargs: Mapping[str, Any] = model_kwargs
        for step in self._steps:
            # 遍历_steps中的每个步骤，依次应用它们的apply方法
            args, kwargs = step.apply(args, kwargs, model=model)
        assert not kwargs
        # 确保kwargs为空
        return args
        # 返回转换后的模型输入


@runtime_checkable
# 声明一个运行时可检查的装饰器，用于声明OutputAdaptStep协议的子类

class OutputAdaptStep(Protocol):
    """A protocol that defines a step in the output adapting process.

    The output adapting process is a sequence of steps that are applied to the
    PyTorch model outputs to transform them into the outputs format produced by the
    exported ONNX model. Each step takes the PyTorch model outputs as arguments and
    """
    # 输出适配过程的步骤协议，用于定义输出适配过程中的步骤
    # 该方法实现了将模型输出进行转换的功能。
    # 它作为一个基本的、正式的结构，用于描述导出器中任何单个组件对模型输出签名的转换过程。
    
    class SignatureTransformer:
        """
        SignatureTransformer 类用于定义和处理模型输出的转换。
        """
    
        def apply(
            self,
            model_outputs: Any,
            model: Optional[
                Union[torch.nn.Module, Callable, torch_export.ExportedProgram]
            ] = None,
        ) -> Any:
            """
            apply 方法接收模型输出和模型对象作为参数，对模型输出进行转换，并返回转换后的结果。
    
            Args:
                model_outputs (Any): 待转换的模型输出。
                model (Optional[Union[torch.nn.Module, Callable, torch_export.ExportedProgram]], optional):
                    模型对象，可以是 torch 模块、可调用对象或导出的程序。默认为 None。
    
            Returns:
                Any: 转换后的输出结果。
            """
            ...
class OutputAdapter:
    """A class that adapts the PyTorch model outputs to exported ONNX model outputs format."""

    def __init__(self, steps: Optional[List[OutputAdaptStep]] = None):
        # 初始化 OutputAdapter 类，设置步骤列表，默认为空列表
        self._steps = steps or []

    @_beartype.beartype
    def append_step(self, step: OutputAdaptStep) -> None:
        """Appends a step to the output format steps.

        Args:
            step: The step to append.
        """
        # 将给定的步骤对象添加到步骤列表中
        self._steps.append(step)

    @_beartype.beartype
    def apply(
        self,
        model_outputs: Any,
        model: Optional[
            Union[torch.nn.Module, Callable, torch_export.ExportedProgram]
        ] = None,
    ) -> Sequence[Union["torch.Tensor", int, float, bool, str]]:
        """Converts the PyTorch model outputs to exported ONNX model outputs format.

        Args:
            model_outputs: The PyTorch model outputs.
            model: The PyTorch model.

        Returns:
            PyTorch model outputs in exported ONNX model outputs format.
        """
        # 对模型输出应用所有的步骤，将其转换为导出的 ONNX 模型输出格式
        for step in self._steps:
            model_outputs = step.apply(model_outputs, model=model)
        return model_outputs


# TODO: make_fx lose stack info https://github.com/pytorch/pytorch/issues/90276


def _replace_tuple_with_list(spec: pytree.TreeSpec) -> pytree.TreeSpec:
    # 如果规范类型为元组，则替换为列表类型
    _type = list if spec.type == tuple else spec.type
    return pytree.TreeSpec(
        _type, spec.context, list(map(_replace_tuple_with_list, spec.children_specs))
    )


def _open_top_level_list_if_single_element(spec: pytree.TreeSpec) -> pytree.TreeSpec:
    # 如果规范类型为列表且只有一个子元素，则打开顶层列表
    if spec.type == list and spec.num_children == 1:
        return spec.children_specs[0]
    return spec


def _assert_identical_pytree_spec(
    spec1: pytree.TreeSpec, spec2: pytree.TreeSpec, error_message: str
) -> None:
    """Assert the two `TreeSpec` objects are identical.

    Args:
        spec1: The first `TreeSpec` object.
        spec2: The second `TreeSpec` object.
        error_message: The error message to raise if the two `TreeSpec` objects are not
            identical.

    Raises:
        ValueError: If the two `TreeSpec` objects are not identical.
    """
    # TODO(bowbao): Turn this check into diagnostic. Consider warning instead of error.
    pass_if_any_checks: Sequence[Callable[[], bool]] = [
        lambda: spec1 == spec2,
        # FIXME: Bug in `dynamo.export`. Sometimes outputs returned in 'list' instead of 'tuple'.
        lambda: _replace_tuple_with_list(spec1) == _replace_tuple_with_list(spec2),
        # FIXME: Bug in `dynamo.export`. Sometimes single function return is wrapped in list.
        lambda: _open_top_level_list_if_single_element(spec1) == spec2,
        lambda: spec1 == _open_top_level_list_if_single_element(spec2),
    ]

    # 如果没有任何检查通过，则引发 ValueError 异常，显示错误消息
    if not any(check() for check in pass_if_any_checks):
        raise ValueError(f"{error_message}\nExpect {spec1}.\nActual {spec2}.")


class BindInputStep(InputAdaptStep):
    """Bind the input arguments to the model signature."""
    def __init__(self, model_signature: inspect.Signature):
        # 初始化方法，接收一个模型签名对象作为参数，并将其保存在实例变量中
        self._model_signature = model_signature

    def apply(
        self,
        model_args: Sequence[Any],
        model_kwargs: Mapping[str, Any],
        model: Optional[
            Union[torch.nn.Module, Callable, torch_export.ExportedProgram]
        ] = None,
    ) -> Tuple[Sequence[Any], Mapping[str, Any]]:
        """Bind the input arguments to the model signature.

        We hope the input kwargs will be mapped to bound.args after binding.
        If not, we will raise an error.

        Args:
            model_args: The model args.
            model_kwargs: The model kwargs.
            model: The PyTorch model.

        Returns:
            A tuple of the model args and kwargs. args is always empty.

        Raises:
            ValueError: If there are keyword-only arguments left after binding args and
                kwargs to model signature.
        """
        # 使用模型签名对象绑定输入的模型参数和关键字参数
        bound = self._model_signature.bind(*model_args, **model_kwargs)
        # 应用默认值到绑定对象中
        bound.apply_defaults()

        # 检查是否有仅限关键字参数（keyword-only arguments），若有则抛出错误
        # 绑定对象的 kwargs 属性只包含在调用 bind 和 apply_defaults 后剩余的仅限关键字参数
        if bound.kwargs:
            raise ValueError("Keyword-only arguments are not supported.")
        # 返回一个空的元组和绑定对象的参数字典
        return (), bound.arguments
class MergeKwargsIntoArgsInputStep(InputAdaptStep):
    """Merge the input kwargs into the input args."""

    def apply(
        self,
        model_args: Sequence[Any],
        model_kwargs: Mapping[str, Any],
        model: Optional[
            Union[torch.nn.Module, Callable, torch_export.ExportedProgram]
        ] = None,
    ) -> Tuple[Sequence[Any], Mapping[str, Any]]:
        """Merge the input kwargs into the input args.

        Args:
            model_args: The model args.
            model_kwargs: The model kwargs.
            model: The PyTorch model.

        Returns:
            A tuple of the model args and kwargs. kwargs is always empty.
        """
        # Combine model_args (positional arguments) and model_kwargs (keyword arguments)
        # into a single tuple for positional arguments, and an empty dictionary for kwargs.
        return tuple(model_args) + tuple(model_kwargs.values()), {}


class LiftParametersAndBuffersIntoArgsInputStep(InputAdaptStep):
    """Append parameters and buffers to model's positional argument list."""

    def __init__(self, inputs: Tuple["torch.Tensor", ...]) -> None:
        self.inputs = inputs

    def apply(
        self,
        model_args: Sequence[Any],
        model_kwargs: Mapping[str, Any],
        model: Optional[
            Union[torch.nn.Module, Callable, torch_export.ExportedProgram]
        ] = None,
    ) -> Tuple[Sequence[Any], Mapping[str, Any]]:
        """Append model's parameters and buffers into its input.

        Args:
            model_args: The model args.
            model_kwargs: The model kwargs.
            model: The PyTorch model.

        Returns:
            A tuple of the model args + appended inputs and kwargs.
        """
        # Append self.inputs (model parameters and buffers) to model_args,
        # and keep model_kwargs unchanged.
        return (*model_args, *self.inputs), model_kwargs


class ConvertComplexToRealRepresentationInputStep(InputAdaptStep):
    """Convert complex dtype tensors to real representation tensors.

    ONNX does not support complex dtype tensors. Thus, we convert complex dtype tensors
    to real representation tensors (i.e., float dtype tensors with an extra dimension
    representing the real and imaginary parts of the complex number).

    """

    def apply(
        self,
        model_args: Sequence[Any],
        model_kwargs: Mapping[str, Any],
        model: Optional[
            Union[torch.nn.Module, Callable, torch_export.ExportedProgram]
        ] = None,
    ) -> Tuple[Sequence[Any], Mapping[str, Any]]:
        """Convert complex tensors to float tensors.

        Args:
            model_args: The model args.
            model_kwargs: The model kwargs.
            model: The PyTorch model.

        Returns:
            A tuple of the model args and kwargs.
        """
        # Convert any complex dtype tensors in model_args to their real representation
        # using torch.view_as_real(), and keep model_kwargs unchanged.
        return (
            tuple(
                torch.view_as_real(arg.resolve_conj())
                if isinstance(arg, torch.Tensor) and arg.is_complex()
                else arg
                for arg in model_args
            ),
            model_kwargs,
        )


class RemoveNoneInputStep(InputAdaptStep):
    """Remove `None` from arguments.

    This adapt step assumes ``model_kwargs`` is empty. It also assumes ``model_args``
    are already properly structured for input.

    """
    """
    Remove `None` values from model arguments.

    This class provides a method `apply` to filter out `None` values from
    the provided model arguments (`model_args`). It ensures that the argument list
    is flattened, i.e. it does not check `None` inside nested collections.
    """

    def apply(
        self,
        model_args: Sequence[Any],
        model_kwargs: Mapping[str, Any],
        model: Optional[
            Union[torch.nn.Module, Callable, torch_export.ExportedProgram]
        ] = None,
    ) -> Tuple[Sequence[Any], Mapping[str, Any]]:
        """Remove `None` from arguments.

        Args:
            model_args: The model args.
            model_kwargs: The model kwargs.
            model: The PyTorch model.

        Returns:
            A tuple of the model args and an empty dictionary.

        Raises:
            ValueError: If `model_kwargs` is not empty.
        """
        # Assert that model_kwargs is empty
        assert not model_kwargs
        # Filter out `None` values from model_args using a generator expression
        return tuple(arg for arg in model_args if arg is not None), {}
# 继承自InputAdaptStep类，用于移除非张量类型的输入参数
class RemoveNonTensorInputStep(InputAdaptStep):
    """Remove the non-tensor input arguments.

    Dynamo does not support non-tensor input arguments (https://github.com/pytorch/pytorch/issues/99534).

    Specifically, it does put the input into graph with an empty node, but consumed by no ones.
    The concrete value is embedded into the graph as a constant arg of a target node. Meta
    suggests in this case that one should rewrite the model code to make it tensor if the
    input value is supposed to change at runtime. We might need to further investigate
    the feasibility of that suggestion.

    For example,

        def func(x, b=1.0):
            y = x + b
            z = y.relu()
            return (y, z)

        x = torch.randn(1, 1, 2, dtype=torch.float32)
        gm_fun, _ = dynamo.export(func, x, b=8.0, aten_graph=True, tracing_mode="real")

        # class GraphModule(torch.nn.Module):
        #     def forward(self, x, b):
        #         arg0: f32[1, 1, 2], arg1, = fx_pytree.tree_flatten_spec(([x, b], {}), self._in_spec)
        #         # File: path/to/pytorch/test_constant_input.py:5, code: y = x + b
        #         add_tensor: f32[1, 1, 2] = torch.ops.aten.add.Tensor(arg0, 8.0);  arg0 = None

        #         # File: path/to/pytorch/test_constant_input.py:6, code: z = y.relu()
        #         relu_default: f32[1, 1, 2] = torch.ops.aten.relu.default(add_tensor)
        #         return pytree.tree_unflatten([add_tensor, relu_default], self._out_spec)

    Empty torch.fx.Node input leading to a mismatched number of input with PyTorch, as
    it's ignored in ONNX graph. Thus, we delete the useless input here.
    """

    def apply(
        self,
        model_args: Sequence[Any],
        model_kwargs: Mapping[str, Any],
        model: Optional[
            Union[torch.nn.Module, Callable, torch_export.ExportedProgram]
        ] = None,
    ) -> Tuple[Sequence[Any], Mapping[str, Any]]:
        """Remove Constant from arguments.

        Args:
            model_args: The model args.
            model_kwargs: The model kwargs.
            model: The PyTorch model.

        Returns:
            A tuple of the model args and kwargs.

        Raises:
            ValueError: If `model_kwargs` is not empty.
        """
        assert not model_kwargs
        # 过滤掉 model_args 中的整数、浮点数、布尔值和字符串类型的参数，保留其它类型参数
        return (
            tuple(
                arg
                for arg in model_args
                if not isinstance(arg, (int, float, bool, str))
            ),
            {},  # 返回空的 model_kwargs
        )


# 继承自InputAdaptStep类，用于展开嵌套的集合类型并返回扁平化的元素列表
class FlattenInputWithTreeSpecValidationInputStep(InputAdaptStep):
    """Flatten nested collection types and return a flat list of elements.

    ONNX can't represent collection types (e.g., dictionary, tuple of tuple of tensor,
    etc).

    This class stores the `SpecTree` output produced when `adapt` was called the first
    time. It then validates the `SpecTree` output produced from later `adapt` calls.
    """

    _spec: Optional[pytree.TreeSpec] = None
    def apply(
        self,
        model_args: Sequence[Any],
        model_kwargs: Mapping[str, Any],
        model: Optional[
            Union[torch.nn.Module, Callable, torch_export.ExportedProgram]
        ] = None,
    ) -> Tuple[Sequence[Any], Mapping[str, Any]]:
        """Flatten the model args and kwargs and validate the `SpecTree` output.

        Args:
            model_args: The model args.
            model_kwargs: The model kwargs.
            model: The PyTorch model.

        Returns:
            A tuple of the flattened model args and kwargs. The kwargs is empty, because
            they are flattened and merged into the args.

        Raises:
            ValueError: If the `SpecTree` output produced from the current `model_outputs`
                is not identical to the `SpecTree` output produced from the first
                `model_outputs` that was passed to this method.
        """
        # 将模型的参数和关键字参数展平，并验证 `SpecTree` 的输出
        flattened_args, spec = pytree.tree_flatten((model_args, model_kwargs))
        
        # 如果对象的 `_spec` 属性为空，则将其设为当前的 `spec`
        if self._spec is None:
            self._spec = spec
        else:
            # 否则，确保当前的 `spec` 与存储的 `_spec` 相同，否则抛出错误
            _assert_identical_pytree_spec(
                self._spec,
                spec,
                error_message="Model inputs incompatible with the format that was exported. ",
            )
        
        # 返回展平后的参数和一个空的字典，因为关键字参数已经被展平并合并到参数中了
        return flattened_args, {}
class FlattenOutputStep(OutputAdaptStep):
    """Flatten nested collection types and return a flat list of elements.

    ONNX can't represent collection types (e.g., dictionary, tuple of tuple of tensor,
    etc).

    NOTE: Ideally we would want to use ``FlattenOutputWithTreeSpecValidationOutputStep``, such
    that `SpecTree` can be validate for new model outputs. However, this is not possible
    currently because we never have access to real PyTorch model outputs during export.
    Only traced outputs may be available, but they are not an accurate reflection of the
    original PyTorch model outputs format as they are typically in their own unique format,
    depending on the tracing strategy.
    """

    def apply(
        self,
        model_outputs: Any,
        model: Optional[
            Union[torch.nn.Module, Callable, torch_export.ExportedProgram]
        ] = None,
    ) -> Sequence[Any]:
        """Flatten the model outputs.

        Args:
            model_outputs: The model outputs to flatten.
            model: The PyTorch model.

        Returns:
            A tuple of the flattened model outputs.
        """
        # 使用 pytree 库将模型输出展平为列表
        return pytree.tree_leaves(model_outputs)


class ConvertComplexToRealRepresentationOutputStep(OutputAdaptStep):
    """Convert complex dtype tensors to real representation tensors.

    ONNX does not support complex dtype tensors. Thus, we convert complex dtype tensors
    to real representation tensors (i.e., float dtype tensors with an extra dimension
    representing the real and imaginary parts of the complex number).

    """

    def apply(
        self,
        model_outputs: Any,
        model: Optional[
            Union[torch.nn.Module, Callable, torch_export.ExportedProgram]
        ] = None,
    ) -> Any:
        """Convert float tensors to complex tensors.

        Args:
            model_output: The model output.
            model: The PyTorch model.

        Returns:
            A tuple of the model output.
        """
        # 将复杂数据类型张量转换为实数表示的张量
        return [
            torch.view_as_real(output.resolve_conj())
            if isinstance(output, torch.Tensor) and torch.is_complex(output)
            else output
            for output in model_outputs
        ]


class FlattenOutputWithTreeSpecValidationOutputStep(OutputAdaptStep):
    """Same as ``FlattenOutputStep``, with additional `TreeSpec` validation.

    This class stores the `SpecTree` output produced when `adapt` was called the first
    time. It then validates the `SpecTree` output produced from later `adapt` calls.
    """

    _spec: Optional[pytree.TreeSpec] = None

    def apply(
        self,
        model_outputs: Any,
        model: Optional[
            Union[torch.nn.Module, Callable, torch_export.ExportedProgram]
        ] = None,
    ) -> Sequence[Any]:
        """Flatten the model outputs and validate using `TreeSpec`.

        Args:
            model_outputs: The model outputs to flatten.
            model: The PyTorch model.

        Returns:
            A tuple of the flattened model outputs.
        """
        # 使用 pytree 库将模型输出展平为列表，并根据存储的 SpecTree 对输出进行验证
        return pytree.tree_leaves(model_outputs)
    ) -> Sequence[Any]:
        """
        Flatten the model outputs and validate the `SpecTree` output.

        Args:
            model_outputs: The model outputs to flatten.
            model: The PyTorch model.

        Returns:
            flattened_outputs: The flattened model outputs.

        Raises:
            ValueError: If the `SpecTree` output produced from the current `model_outputs`
                is not identical to the `SpecTree` output produced from the first
                `model_outputs` that was passed to this method.
        """
        # 使用 PyTree 库中的 tree_flatten 函数，将模型输出展平，并返回展平后的输出和规范 spec
        flattened_outputs, spec = pytree.tree_flatten(model_outputs)
        
        # 如果 self._spec 还未初始化，则将当前的 spec 赋值给 self._spec
        if self._spec is None:
            self._spec = spec
        else:
            # 否则，调用 _assert_identical_pytree_spec 函数，确保当前的 spec 与已保存的 self._spec 相同
            _assert_identical_pytree_spec(
                self._spec,
                spec,
                error_message="Model outputs incompatible with the format that was exported. ",
            )
        
        # 返回展平后的模型输出
        return flattened_outputs
class PrependParamsBuffersConstantAotAutogradInputStep(InputAdaptStep):
    """Prepend model parameters, buffers and constants to the user input.

    :func:`torch.export.export` lifts model parameters, buffers and constants as model input, thus, they
    must be added to the user input before the model is executed.

    Args:
        model: The PyTorch model with embedded parameters and buffers.
    """

    def apply(
        self,
        model_args: Sequence[Any],
        model_kwargs: Mapping[str, Any],
        model: Optional[
            Union[torch.nn.Module, Callable, torch_export.ExportedProgram]
        ] = None,
    ) -> Tuple[Sequence[Any], Mapping[str, Any]]:
        """Convert complex tensors to float tensors.

        Args:
            model_args: The model args.
            model_kwargs: The model kwargs.
            model: The PyTorch model.

        Returns:
            A tuple of the model args and kwargs.
        """
        # Collect model parameters in the order specified by graph signature
        ordered_params = tuple(
            model.state_dict[name] for name in model.graph_signature.parameters  # type: ignore[union-attr,index]
        )
        # Identify non-persistent buffers to handle separately
        non_persistent_buffers = set(model.graph_signature.non_persistent_buffers)  # type: ignore[union-attr]
        ordered_buffers = []
        # Collect model buffers in the order specified by graph signature
        for name in model.graph_signature.buffers:  # type: ignore[union-attr]
            if name in non_persistent_buffers:
                ordered_buffers.append(model.constants[name])  # type: ignore[union-attr]
            else:
                ordered_buffers.append(model.state_dict[name])  # type: ignore[union-attr,index]
        # Collect lifted constant tensors in the order specified by graph signature
        ordered_constant_tensors = tuple(
            model.constants[fqn] for fqn in model.graph_signature.lifted_tensor_constants  # type: ignore[union-attr,index]
        )

        # NOTE: calling convention is first params, then buffers, then args as user supplied them.
        # See: torch/_functorch/aot_autograd.py#L1034
        # Prepare updated_args by concatenating ordered parameters, buffers, constant tensors, and original args
        updated_args = (
            *ordered_params,
            *ordered_buffers,
            *ordered_constant_tensors,
            *model_args,
        )
        # If model_kwargs are provided, merge them into updated_args using MergeKwargsIntoArgsInputStep
        if model_kwargs:
            return MergeKwargsIntoArgsInputStep().apply(
                updated_args, model_kwargs, model=model
            )
        # Return updated_args and an empty dictionary for kwargs
        return updated_args, {}


class PrependParamsAndBuffersAotAutogradOutputStep(OutputAdaptStep):
    """Prepend model's mutated buffers to the user output.

    :func:`torch.export.export` lifts model's mutated buffers as outputs, thus, they
    must be added to the user output after the model is executed.

    Args:
        model: The PyTorch model with mutated buffers.
    """

    def apply(
        self,
        model_outputs: Any,
        model: Optional[
            Union[torch.nn.Module, Callable, torch_export.ExportedProgram]
        ] = None,
    ) -> Any:
        """Convert model outputs to final format.

        Args:
            model_outputs: The model outputs.
            model: The PyTorch model.

        Returns:
            The final formatted model outputs.
        """
    ) -> Sequence[Any]:
        """
        Flatten the model outputs and validate the `SpecTree` output.

        Args:
            model_outputs: The model outputs to flatten.
            model: The PyTorch model.

        Returns:
            flattened_outputs: The flattened model outputs.
        """

        assert isinstance(
            model, torch_export.ExportedProgram
        ), "'model' must be torch_export.ExportedProgram"

        # Create a tuple of ordered buffers based on the model's state_dict or constants
        ordered_buffers = tuple(
            model.state_dict[name] if name in model.state_dict else model.constants[name]
            for name in model.graph_signature.buffers_to_mutate.values()
        )

        # Define updated_outputs by concatenating ordered buffers and model_outputs
        updated_outputs = (*ordered_buffers, *model_outputs)

        # Return the concatenated outputs
        return updated_outputs
```