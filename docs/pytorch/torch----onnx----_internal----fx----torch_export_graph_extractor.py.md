# `.\pytorch\torch\onnx\_internal\fx\torch_export_graph_extractor.py`

```
# 用于声明未来的类型注解可以使用未定义的函数或类
# mypy: allow-untyped-defs

# 此文件被引用，位置在 /opt/pytorch/torch/_dynamo/eval_frame.py::DONT_WRAP_FILES 处。
# 引入自 https://github.com/pytorch/pytorch/pull/98894。
# 如果此文件被重命名、移动等，请更新该处的引用！

# 导入必要的模块和库
from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence, TYPE_CHECKING, Union

# 导入 Torch 相关模块和子模块
import torch._dynamo
import torch.fx
import torch.onnx
from torch.onnx._internal import _beartype, exporter, io_adapter
from torch.onnx._internal.diagnostics import infra

# 如果是类型检查阶段，导入 ExportedProgram 类
if TYPE_CHECKING:
    from torch.export.exported_program import ExportedProgram


class TorchExport(exporter.FXGraphExtractor):
    """使用 torch.export API 生成一个 FX GraphModule 的类

    Args:
        aten_graph: 如果为 True，导出包含 ATen 操作符的图。
                    如果为 False，导出包含 Python 操作符的图。
    """

    def __init__(
        self,
        aten_graph: Optional[bool] = None,
    ):
        super().__init__()
        # 初始化时设定是否使用 ATen 图，默认为 True
        self.aten_graph = aten_graph or True

    def generate_fx(
        self,
        options: exporter.ResolvedExportOptions,
        model: "ExportedProgram",  # type: ignore[override]
        model_args: Sequence[Any],
        model_kwargs: Mapping[str, Any],
    ) -> torch.fx.GraphModule:
        # 不需要将可调用对象转换为 FX 图。
        # 此 FX 图提取器假设 `model` 是通过以下方式获取的：
        #     exported_program = torch.export.export(
        #         model,
        #         args=model_args,  # type: ignore[arg-type]
        #         kwargs=model_kwargs,  # type: ignore[arg-type]
        #     )

        # 将 FX 图导出为 ONNX ModelProto。
        self.input_adapter.append_step(
            io_adapter.FlattenInputWithTreeSpecValidationInputStep()
        )
        self.input_adapter.append_step(
            io_adapter.PrependParamsBuffersConstantAotAutogradInputStep()
        )

        # ONNX 不支持 None 类型的输入。在构建图时，所有 None 类型的输入都将被移除。
        # 这里注册此步骤到输入适配器。
        options.fx_tracer.input_adapter.append_step(io_adapter.RemoveNoneInputStep())

        # 注意：针对 https://github.com/pytorch/pytorch/issues/99534 的临时解决方法。
        # Dynamo 不支持非张量类型的输入。
        options.fx_tracer.input_adapter.append_step(
            io_adapter.RemoveNonTensorInputStep()
        )

        # ONNX 不支持复杂类型的输入。在构建图时，所有复杂类型的输入都将被转换为实数表示。
        # 这里注册此步骤到输入/输出适配器。
        options.fx_tracer.input_adapter.append_step(
            io_adapter.ConvertComplexToRealRepresentationInputStep()
        )

        updated_model_args = self.input_adapter.apply(
            *model_args, model=model, **model_kwargs
        )

        # ONNX 无法表示集合类型（如字典、张量的元组等），我们将集合展平并注册每个元素作为输出。
        options.fx_tracer.output_adapter.append_step(io_adapter.FlattenOutputStep())

        # 输出后处理步骤应在 `FlattenOutputStep` 之后进行。
        options.fx_tracer.output_adapter.append_step(
            io_adapter.ConvertComplexToRealRepresentationOutputStep()
        )

        options.fx_tracer.output_adapter.append_step(
            io_adapter.PrependParamsAndBuffersAotAutogradOutputStep()
        )

        # run_decomposition 生成一个具有分解操作的新图模块。
        # 因此，我们需要在 io 适配器之后运行此步骤。
        model = model.run_decompositions(options.decomposition_table)

        # 将 FX 图导出为 ONNX ModelProto。
        return self.pre_export_passes(options, model, model.graph_module, updated_model_args)  # type: ignore[return-value]
        # TODO: Import here to prevent circular dependency
        # 导入这里以避免循环依赖

        from torch.onnx._internal.fx import analysis, passes
        # 从torch.onnx._internal.fx模块中导入analysis和passes

        diagnostic_context = options.diagnostic_context
        # 从参数options中获取诊断上下文对象

        # ONNX does not support concept of (implicit) type promotion.
        # Insert type casts explicitly where needed.
        # ONNX不支持隐式类型提升的概念。在需要时明确插入类型转换。
        fx_module = passes.InsertTypePromotion(diagnostic_context, fx_module).run()
        # 使用InsertTypePromotion pass在fx_module中运行，将需要的地方显式插入类型转换。

        analysis.UnsupportedFxNodesAnalysis(
            diagnostic_context, fx_module, options.onnxfunction_dispatcher
        ).analyze(infra.levels.ERROR)
        # 分析不支持的FX节点，在诊断上下文、fx_module和onnxfunction_dispatcher选项中分析，错误级别为infra.levels.ERROR。

        # This operation should be invoked as the last pre export pass.
        # See [NOTE: Modularize pass ordering]
        # 这个操作应该作为最后一个导出前的pass被调用。参见[NOTE: Modularize pass ordering]
        fx_module = passes.Modularize(
            diagnostic_context, fx_module, is_exported_program=True
        ).run()
        # 使用Modularize pass在fx_module中运行，设置is_exported_program为True。

        return fx_module
        # 返回处理后的fx_module
```