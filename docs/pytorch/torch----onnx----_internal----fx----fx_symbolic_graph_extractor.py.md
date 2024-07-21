# `.\pytorch\torch\onnx\_internal\fx\fx_symbolic_graph_extractor.py`

```
# mypy: allow-untyped-defs
# 从未来导入注释，允许未类型化的定义

import functools  # 导入 functools 模块

from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union  # 导入类型提示相关内容

import torch  # 导入 PyTorch 库
import torch.fx  # 导入 PyTorch FX 模块
import torch.onnx  # 导入 PyTorch ONNX 模块

import torch.onnx._internal.fx.passes as passes  # 导入 ONNX 内部的 FX passes 模块
from torch.onnx._internal import _beartype, exporter, io_adapter  # 导入内部的 _beartype, exporter, io_adapter 模块

# 要直接包装的函数，以产生 torch.fx.Proxy，以便符号数据可以流经这些函数。
# 未在 C++ 的 pybind11 中定义的 Python 函数（例如 `torch.arange`）不经过 Python 调度器，
# 因此 FX 的 Python 调度器不会自动修补它们。
# 下面的列表意味着 `torch.arange`、`torch.tensor` 等将被修补。
_TORCH_METHODS_TO_PATCH: Tuple[str, ...] = (
    "arange",
    "tensor",
    "finfo",
    "full",
    "empty",
)


class ModuleExpansionTracer(torch.fx._symbolic_trace.Tracer):
    """用于创建友好的 ONNX 导出 FX 图的追踪器。

    此追踪器将模型跟踪为运算符。即，跟踪的图大多包含 call_function 节点，
    而不包含 call_module 节点。call_module 节点对于 ONNX 导出器使用 make_fx(...) 是有问题的。
    """

    @_beartype.beartype
    def is_leaf_module(
        self, module: torch.nn.Module, module_qualified_name: str
    ) -> bool:
        # 返回 False，以便所有子模块都被视为非叶子节点，因此在 torch.fx._symbolic_trace.Tracer.call_module 中扩展为运算符。
        return False

    @_beartype.beartype
    def to_bool(self, obj: torch.fx.Proxy) -> bool:
        # FIXME: 这是一种通过 if-else Python 块进行追踪的 hack 方法。
        # 如果 if-else 块，可能生成不正确的 ONNX 图。
        return False


def _wrap_for_symbolic_trace(target: Callable) -> Tuple[Callable, Callable]:
    """将目标函数包装为符号化追踪的函数。

    这个函数包装目标函数，使得其包装器在符号计算中产生 torch.fx.Proxy。
    返回的值是包装器和原始函数。根据 `_TORCH_METHODS_TO_PATCH`，
    这个函数将接收 `torch.arange`、`torch.tensor` 等作为输入。
    """

    @functools.wraps(target)
    def wrapper(*args, **kwargs):
        proxy = None

        def check_has_proxy(v):
            if isinstance(v, torch.fx.Proxy):
                nonlocal proxy
                proxy = v

        torch.fx.node.map_aggregate(args, check_has_proxy)
        torch.fx.node.map_aggregate(kwargs, check_has_proxy)

        if proxy is not None:
            return proxy.tracer.create_proxy("call_function", target, args, kwargs)
        else:
            return target(*args, **kwargs)

    return wrapper, target


@_beartype.beartype
def _module_expansion_symbolic_trace(
    root: Union[torch.nn.Module, Callable[..., Any]],
    concrete_args: Optional[Dict[str, Any]] = None,
) -> torch.fx.GraphModule:
    """将可调用对象追踪为 FX 图。

    这个函数追踪一个可调用对象，转换为 FX 图。
    """
    """
    When "root" is torch.nn.Module, calls to its submodule (type: torch.nn.Module) will be
    expanded into operators (e.g., torch.matmul, torch.add, +, and -) to simplify graph
    structure.
    """
    # 对于不支持符号跟踪的函数，创建包装器，在跟踪期间生成符号结果。
    patched_torch_methods = {
        target_name: _wrap_for_symbolic_trace(getattr(torch, target_name))
        for target_name in _TORCH_METHODS_TO_PATCH
    }

    # 设置符号跟踪友好的函数，以便下面的 `tracer.trace` 可以工作。
    for name, (wrapper, _) in patched_torch_methods.items():
        setattr(torch, name, wrapper)

    try:
        # 设置一个跟踪器。
        tracer = ModuleExpansionTracer()
        # 跟踪模型。
        graph = tracer.trace(root, concrete_args)
        # 获取模型类名作为名称，如果是 torch.nn.Module 类型的实例。
        name = (
            root.__class__.__name__
            if isinstance(root, torch.nn.Module)
            else root.__name__
        )
        # 返回一个包含跟踪结果的 torch.fx.GraphModule 对象。
        return torch.fx.GraphModule(tracer.root, graph, name)
    finally:
        # 恢复符号跟踪的补丁。
        for name, (_, wrapped) in patched_torch_methods.items():
            # wrapped 是 `torch.name` 的原始版本。
            setattr(torch, name, wrapped)
# TODO: Migrate to `DynamoExporter` after fake model tracing is supported.
# Proposal at https://github.com/pytorch/pytorch/issues/95900.

# FXSymbolicTracer 类，继承自 exporter.FXGraphExtractor，用于生成 FX GraphModule，使用 torch.fx.symbolic_trace API
class FXSymbolicTracer(exporter.FXGraphExtractor):

    """Generates a FX GraphModule using torch.fx.symbolic_trace API
    Args:
        concrete_args: Inputs to be partially specialized
            It can be used to remove control flow or data structures.
            For example::
                def f(a, b):
                    if b == True:
                        return a
                    else:
                        return a*2
            FX can typically not trace through this due to the presence of control
            flow. However, we can use `concrete_args` to specialize on the value of
            `b` to trace through this::
                f = fx.symbolic_trace(f, concrete_args={'b': False})
                assert f(3, False)  == 6
            Note that although you can still pass in different values of `b`, they will be ignored.
            It can also be used to eliminate data-structure handling from
            our function. This will use pytrees to flatten your input. To avoid
            overspecializing, pass in `fx.PH` for values that shouldn't be
            specialized. For example::
                def f(x):
                    out = 0
                    for v in x.values():
                        out += v
                    return out
                f = fx.symbolic_trace(f, concrete_args={'x': {'a': fx.PH, 'b': fx.PH, 'c': fx.PH}})
                assert f({'a': 1, 'b': 2, 'c': 4}) == 7
    """

    # 初始化方法，接受 concrete_args 参数，用于部分特化
    def __init__(self, concrete_args: Optional[Dict[str, Any]] = None):
        # 调用父类的初始化方法
        super().__init__()
        # TODO: plumb ``concrete_args`` to symbolic_trace call at ``generate_fx``
        # 设置实例变量 concrete_args，用于后续的 symbolic_trace 调用
        self.concrete_args = concrete_args

    # 使用装饰器进行类型检查，将 model、model_args 和 model_kwargs 传递给 symbolic_trace，以生成 FX 图形的方法
    @_beartype.beartype
    def _trace_into_fx_graph_via_fx_symbolic_trace(
        self, model, model_args, model_kwargs
    ) -> torch.fx.GraphModule:
        # 定义方法签名，返回一个torch.fx.GraphModule对象
        # 绑定模型参数和关键字参数，使用模型签名检索未提供参数的默认值，用于构建“concrete_args”。
        bind_input_step = io_adapter.BindInputStep(
            torch.onnx.utils.model_signature(model)
        )
        # 将绑定输入步骤添加到输入适配器中
        self.input_adapter.append_step(bind_input_step)
        # 应用绑定步骤，获取模型参数和关键字参数的命名参数
        _, named_args = bind_input_step.apply(model_args, model_kwargs, model=model)

        # 创建调用符号化跟踪（torch.fx.symbolic_trace）的输入
        # “concrete_args”示例内容：
        # concrete_args["x"] = torch.fx._symbolic_trace.PH
        # concrete_args["b"] = 1
        # 其中“x”和“b”是“signature”中的参数名称。
        concrete_args = {}
        for param_name, param_value in named_args.items():
            if isinstance(param_value, torch.Tensor):
                # 如果参数值是torch.Tensor类型，则将其视为可替换的张量符号（占位符）。
                concrete_args[param_name] = torch.fx._symbolic_trace.PH
            else:
                concrete_args[param_name] = param_value

        # 将关键字参数合并回参数中，因为FX图期望的格式是这样的。
        merge_kwargs_step = io_adapter.MergeKwargsIntoArgsInputStep()
        # 将合并关键字参数步骤添加到输入适配器中
        self.input_adapter.append_step(merge_kwargs_step)
        # 返回模块扩展的符号化跟踪结果
        return _module_expansion_symbolic_trace(model, concrete_args=concrete_args)
    ) -> torch.fx.GraphModule:
        diagnostic_context = options.diagnostic_context
        # 使用选项中的诊断上下文
        graph_module = self._trace_into_fx_graph_via_fx_symbolic_trace(
            model, model_args, model_kwargs
        )
        # 通过 FX 符号化跟踪将模型转换为 FX 图模块

        # 确保所有占位节点在 get_attr 节点之前执行。
        # 否则，输入可能与初始化器在最终 ModeoProto.graph.input 中交错。
        # 我们期望 ModeoProto.graph.input = [input_0, input_1, ..., input_n, weight_0, weight_1, ..., weight_m]
        # 而不希望 ModeoProto.graph.input = [input_0, weight_0, input_1, weight_1, ..., input_n, weight_0, weight_1, ..., weight_m]
        graph_module = passes.MovePlaceholderToFront(
            diagnostic_context, graph_module
        ).run()
        # 将 get_attr 替换为占位符以节省内存，使生成的模型不包含权重张量。
        # "replaced_attrs" 是被替换的权重张量的元组。
        replace_get_attr_with_placeholder_pass = passes.ReplaceGetAttrWithPlaceholder(
            diagnostic_context, graph_module
        )
        graph_module = replace_get_attr_with_placeholder_pass.run()
        replaced_attrs = replace_get_attr_with_placeholder_pass.replaced_attrs
        # 将参数和缓冲区提升为输入参数的步骤，并添加到输入适配器中。
        append_extra_input_step = io_adapter.LiftParametersAndBuffersIntoArgsInputStep(
            replaced_attrs
        )
        self.input_adapter.append_step(append_extra_input_step)
        # 将所有新创建的占位节点移动到图的前面。
        graph_module = passes.MovePlaceholderToFront(
            diagnostic_context, graph_module
        ).run()
        # 完成图的编辑。
        graph_module.recompile()

        updated_model_args = self.input_adapter.apply(
            *model_args, model=model, **model_kwargs
        )

        # 执行导出前通用的处理步骤，并返回结果。
        return self.pre_export_passes(options, model, graph_module, updated_model_args)  # type: ignore[return-value]

    @_beartype.beartype
    def pre_export_passes(
        self,
        options: exporter.ResolvedExportOptions,
        original_model: Union[torch.nn.Module, Callable],
        fx_module: torch.fx.GraphModule,
        fx_module_args: Sequence[Any],
    ):
        # 调用通用的导出前处理步骤函数，并返回其结果。
        return exporter.common_pre_export_passes(
            options, original_model, fx_module, fx_module_args
        )
```