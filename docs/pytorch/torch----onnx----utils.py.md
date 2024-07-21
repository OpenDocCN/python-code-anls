# `.\pytorch\torch\onnx\utils.py`

```py
# mypy: allow-untyped-defs
"""Functions to export models into the ONNX IR format.

These models can be loaded with the ONNX library and then
converted to models which run on other deep learning frameworks.
"""
from __future__ import annotations

import contextlib  # 上下文管理器相关模块
import copy  # 复制对象相关模块
import inspect  # 获取对象信息的模块
import io  # 输入输出流操作的模块
import re  # 正则表达式模块
import typing  # 类型提示相关模块
import warnings  # 警告相关模块
from typing import (
    Any,
    Callable,
    cast,
    Collection,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

import torch  # PyTorch深度学习库
import torch._C._onnx as _C_onnx  # PyTorch的ONNX相关C扩展
import torch.jit._trace  # PyTorch的jit跟踪模块
import torch.serialization  # PyTorch的序列化模块
from torch import _C  # PyTorch的C扩展
from torch.onnx import (  # noqa: F401
    _constants,
    _exporter_states,
    errors,
    symbolic_caffe2,
    symbolic_helper,
)
from torch.onnx._globals import GLOBALS  # 全局变量
from torch.onnx._internal import (
    _beartype,
    diagnostics,
    jit_utils,
    onnx_proto_utils,
    registration,
)

__all__ = [  # 公开的模块成员列表
    "is_in_onnx_export",
    "select_model_mode_for_export",
    "disable_apex_o2_state_dict_hook",
    "setup_onnx_logging",
    "exporter_context",
    "export",
    "model_signature",
    "warn_on_static_input_change",
    "unpack_quantized_tensor",
    "export_to_pretty_string",
    "unconvertible_ops",
    "register_custom_op_symbolic",
    "unregister_custom_op_symbolic",
]


def is_in_onnx_export() -> bool:
    """Returns whether it is in the middle of ONNX export."""
    return GLOBALS.in_onnx_export


# TODO(justinchuby): Remove dependency to this global variable from constant_fold.cpp
# Skip check due to cannot import IValue from torch._C
_params_dict = {}  # type: ignore[var-annotated]


@contextlib.contextmanager
@_beartype.beartype
def select_model_mode_for_export(model, mode: _C_onnx.TrainingMode):
    r"""A context manager to temporarily set the training mode of ``model``
    to ``mode``, resetting it when we exit the with-block.

    Args:
        model: Same type and meaning as ``model`` arg to :func:`export`.
        mode: Same type and meaning as ``training`` arg to :func:`export`.
    """
    if not isinstance(mode, _C_onnx.TrainingMode):
        raise TypeError(
            f"'mode' should be a torch.onnx.TrainingMode enum, but got '{type(mode)}'."
        )
    originally_training: bool = False
    # 检查模型是否具有训练属性
    if hasattr(model, "training"):
        # 保存原始的训练状态
        originally_training = model.training

        # 如果是 ONNX opset 12，对于可训练的模型有更好的支持，更新了 dropout 和 batch_norm 操作符的版本
        if mode == _C_onnx.TrainingMode.TRAINING or (
            mode == _C_onnx.TrainingMode.PRESERVE and originally_training
        ):
            # 设置全局变量，指示导出模式为训练模式
            GLOBALS.export_training = True
            
            # 如果导出的 ONNX opset 版本低于 12，发出警告，因为低于 opset 12 的版本无法正确导出 Dropout 和 BatchNorm 等节点
            if GLOBALS.export_onnx_opset_version < 12:
                warnings.warn(
                    "You are exporting the model in training mode with onnx opset "
                    f"version {GLOBALS.export_onnx_opset_version}. "
                    "Opset versions lower than opset 12 will not be able to export "
                    "nodes such as Dropout and BatchNorm correctly."
                )
        else:
            # 设置全局变量，指示导出模式为非训练模式
            GLOBALS.export_training = False

        # 设置全局变量，记录当前的训练模式
        GLOBALS.training_mode = mode
        
        # 根据不同的模式设置模型的训练状态
        if mode == _C_onnx.TrainingMode.TRAINING:
            model.train(True)
        elif mode == _C_onnx.TrainingMode.EVAL:
            model.train(False)
        # 如果 mode == _C_onnx.TrainingMode.PRESERVE，则不做任何操作

    try:
        # 执行 yield 表达式
        yield
    finally:
        # 如果模型具有训练属性，并且模式不是 _C_onnx.TrainingMode.PRESERVE
        if hasattr(model, "training") and not mode == _C_onnx.TrainingMode.PRESERVE:
            # 恢复模型原始的训练状态
            model.train(originally_training)
@contextlib.contextmanager
@_beartype.beartype
def disable_apex_o2_state_dict_hook(
    model: Union[torch.nn.Module, torch.jit.ScriptFunction]
):
    # 禁用 Apex O2 的 state_dict 钩子，将 fp16 权重返回为 fp32
    # 导出器无法识别它们为相同的张量
    # 由于此钩子仅被优化器使用，在导出时移除它是安全的
    if not isinstance(model, torch.jit.ScriptFunction):
        model_hooks = {}  # 定义模型钩子的字典
        # 遍历模型中的每个模块
        for module in model.modules():
            # 检查模块中的 state_dict 钩子
            for key, hook in module._state_dict_hooks.items():
                # 如果钩子的类型是 "O2StateDictHook"
                if type(hook).__name__ == "O2StateDictHook":
                    if module not in model_hooks:
                        model_hooks[module] = {}
                    model_hooks[module][key] = hook
            # 如果模块有保存的钩子，则从 _state_dict_hooks 中移除
            if module in model_hooks:
                for key in model_hooks[module]:
                    module._state_dict_hooks.pop(key)
        try:
            yield
        finally:
            # 最终将移除的钩子重新添加回去
            for module, m_map in model_hooks.items():
                for key, hook in m_map.items():
                    module._state_dict_hooks[key] = hook
    else:
        try:
            yield
        finally:
            pass


@contextlib.contextmanager
@_beartype.beartype
def setup_onnx_logging(verbose: bool):
    # 设置 ONNX 的日志记录
    is_originally_enabled = torch.onnx.is_onnx_log_enabled()
    # 如果原本已启用日志记录或者设置为详细模式，则启用 ONNX 日志
    if is_originally_enabled or verbose:
        torch.onnx.enable_log()
    try:
        yield
    finally:
        # 如果原本未启用日志记录，则在结束时禁用 ONNX 日志
        if not is_originally_enabled:
            torch.onnx.disable_log()


@contextlib.contextmanager
@_beartype.beartype
def exporter_context(model, mode: _C_onnx.TrainingMode, verbose: bool):
    # 导出器的上下文管理器，集成了模型模式选择、Apex O2 钩子禁用、ONNX 日志设置、导出诊断
    with select_model_mode_for_export(
        model, mode
    ) as mode_ctx, disable_apex_o2_state_dict_hook(
        model
    ) as apex_ctx, setup_onnx_logging(
        verbose
    ) as log_ctx, diagnostics.create_export_diagnostic_context() as diagnostic_ctx:
        yield (mode_ctx, apex_ctx, log_ctx, diagnostic_ctx)


def export(
    model: Union[torch.nn.Module, torch.jit.ScriptModule, torch.jit.ScriptFunction],
    args: Union[Tuple[Any, ...], torch.Tensor],
    f: Optional[Union[str, io.BytesIO]] = None,
    export_params: bool = True,
    verbose: bool = False,
    training: _C_onnx.TrainingMode = _C_onnx.TrainingMode.EVAL,
    input_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    operator_export_type: _C_onnx.OperatorExportTypes = _C_onnx.OperatorExportTypes.ONNX,
    opset_version: Optional[int] = None,
    do_constant_folding: bool = True,
    dynamic_axes: Optional[
        Union[Mapping[str, Mapping[int, str]], Mapping[str, Sequence[int]]]
    ] = None,
    keep_initializers_as_inputs: Optional[bool] = None,
    custom_opsets: Optional[Mapping[str, int]] = None,
    export_modules_as_functions: Union[bool, Collection[Type[torch.nn.Module]]] = False,
    autograd_inlining: Optional[bool] = True,
    dynamo: bool = False,
def export_onnx(
    model: Union[torch.nn.Module, torch.jit.ScriptModule, torch.jit.ScriptFunction],
    f: Optional[Union[IO[str], IO[bytes]]] = None,
    export_params: bool = True,
    verbose: bool = False,
    training: bool = False,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    operator_export_type: Optional[torch.onnx.OperatorExportTypes] = None,
    opset_version: Optional[int] = None,
    do_constant_folding: bool = True,
    keep_initializers_as_inputs: bool = False,
    custom_opsets: Optional[Dict[str, int]] = None,
    export_modules_as_functions: bool = False,
    autograd_inlining: bool = False,
    dynamo: bool = False,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    input_names: Optional[List[str]] = None
) -> Optional[torch.onnx.ONNXProgram]:
    r"""Exports a model into ONNX format.

    If ``model`` is not a :class:`torch.jit.ScriptModule` nor a
    :class:`torch.jit.ScriptFunction`, this runs
    ``model`` once in order to convert it to a TorchScript graph to be exported
    (the equivalent of :func:`torch.jit.trace`). Thus this has the same limited support
    for dynamic control flow as :func:`torch.jit.trace`.

    Raises:
        :class:`torch.onnx.errors.CheckerError`: If the ONNX checker detects an invalid ONNX graph.
        :class:`torch.onnx.errors.UnsupportedOperatorError`: If the ONNX graph cannot be exported because it
            uses an operator that is not supported by the exporter.
        :class:`torch.onnx.errors.OnnxExporterError`: Other errors that can occur during export.
            All errors are subclasses of :class:`errors.OnnxExporterError`.
    """

    # If dynamo export is requested
    if dynamo:
        # Check if the model is not a ScriptModule or ScriptFunction
        if isinstance(model, (torch.jit.ScriptModule, torch.jit.ScriptFunction)):
            # Raise TypeError for unsupported model types
            raise TypeError(
                "Dynamo export does not support ScriptModule or ScriptFunction."
            )

        # Warn about unsupported parameters for dynamo export
        # TODO: These are not supported AT THE TIME
        warnings.warn(
            "f, export_params, verbose, training, input_names, output_names, operator_export_type, opset_version, "
            "do_constant_folding, keep_initializers_as_inputs, custom_opsets, export_modules_as_functions, and "
            "autograd_inlining are not supported for dynamo export at the moment."
        )

        # Decide input format based on the model and its arguments
        args = _decide_input_format(model, args)
        kwargs = {}

        # Extract kwargs if present as the last argument
        if args is not None and isinstance(args[-1], dict):
            kwargs = args[-1]
            args = args[:-1]

        # Convert dynamic axes to dynamic shapes
        dynamic_shapes = _from_dynamic_axes_to_dynamic_shapes(
            model, dynamic_axes, input_names
        )

        # Export the model using TorchScript and obtain exported_program
        exported_program = torch.export.export(
            model, args=args, kwargs=kwargs, dynamic_shapes=dynamic_shapes  # type: ignore[arg-type]
        )

        # Perform dynamo export and obtain onnx_program
        onnx_program = torch.onnx.dynamo_export(exported_program, *args, **kwargs)

        # Save the ONNX program to the specified file if f is provided
        if f is not None:
            onnx_program.save(f)

        # Return the exported ONNX program
        return onnx_program

    # If f is None and dynamo export is not requested, raise ValueError
    if f is None:
        raise ValueError(
            "Export destination must be specified for torchscript-onnx export."
        )
    # 调用 _export 函数，导出模型和相关参数
    _export(
        model,  # 模型对象
        args,  # 导出参数
        f,  # 文件对象，用于导出
        export_params,  # 是否导出模型参数的标志
        verbose,  # 是否输出详细信息的标志
        training,  # 是否在训练模式下导出的标志
        input_names,  # 模型输入名称列表
        output_names,  # 模型输出名称列表
        operator_export_type=operator_export_type,  # 操作符导出类型
        opset_version=opset_version,  # 导出的ONNX opset版本
        do_constant_folding=do_constant_folding,  # 是否进行常量折叠的标志
        dynamic_axes=dynamic_axes,  # 是否保留动态轴信息的标志
        keep_initializers_as_inputs=keep_initializers_as_inputs,  # 是否将初始化器保留为输入的标志
        custom_opsets=custom_opsets,  # 自定义的opset字典
        export_modules_as_functions=export_modules_as_functions,  # 是否将模块导出为函数的标志
        autograd_inlining=autograd_inlining,  # 是否进行自动求导内联的标志
    )
    
    # 函数返回 None
    return None
# 使用装饰器_beartype.beartype确保函数输入的参数类型正确
@_beartype.beartype
# 判断给定的节点是否为常量张量列表
def _is_constant_tensor_list(node):
    # 如果节点类型不是"prim::Constant"，则返回False
    if node.kind() != "prim::Constant":
        return False
    # 获取节点输出的类型
    output_type = node.output().type()
    # 如果输出类型是张量列表类型，则返回True
    if output_type.isSubtypeOf(_C.ListType.ofTensors()):
        return True
    # 如果输出类型是张量的可选列表类型，则返回True
    if output_type.isSubtypeOf(_C.ListType(_C.OptionalType.ofTensor())):
        return True


# ONNX 无法处理常量为张量列表的情况，这种情况可能由常量传播生成。
# 因此我们需要将这些节点拆分回 prim::ListConstructs

# 使用装饰器_beartype.beartype确保函数输入的参数类型正确
@_beartype.beartype
# 在图中拆分张量列表常量
def _split_tensor_list_constants(g, block):
    # 遍历每个节点
    for node in block.nodes():
        # 遍历节点的子块
        for subblock in node.blocks():
            _split_tensor_list_constants(g, subblock)
        # 如果节点是常量张量列表，则执行以下操作
        if _is_constant_tensor_list(node):
            inputs = []
            # 对节点输出的每个值执行操作
            for val in node.output().toIValue():
                # 在图中插入常量值，并将其移动到当前节点之前
                input = g.insertConstant(val)
                input.node().moveBefore(node)
                # 复制节点的元数据到插入的常量节点
                input.node().copyMetadata(node)
                inputs.append(input)

            # 创建 prim::ListConstruct 节点，用插入的常量节点作为输入
            lc = (
                g.create("prim::ListConstruct", inputs)
                .insertBefore(node)
                .output()
                .setType(_C.ListType.ofTensors())
            )
            # 将节点的元数据复制到 prim::ListConstruct 节点
            lc.node().copyMetadata(node)
            # 替换当前节点的输出使用为 prim::ListConstruct 节点的输出
            node.output().replaceAllUsesWith(lc)


# 使用装饰器_beartype.beartype确保函数输入的参数类型正确
@_beartype.beartype
# 优化图的操作
def _optimize_graph(
    graph: _C.Graph,
    operator_export_type: _C_onnx.OperatorExportTypes,
    _disable_torch_constant_prop: bool = False,
    fixed_batch_size: bool = False,
    params_dict=None,
    dynamic_axes=None,
    input_names=None,
    module=None,
):
    # 如果参数字典为空，则初始化为空字典
    if params_dict is None:
        params_dict = {}

    # 内联所有函数调用
    _C._jit_pass_inline(graph)

    # 移除 fork/wait 节点
    _C._jit_pass_inline_fork_wait(graph)
    # 对图进行静态分析
    _C._jit_pass_lint(graph)
    # 如果全局变量 autograd_inlining 为真，则处理 Autograd 函数
    if GLOBALS.autograd_inlining:
        _C._jit_pass_onnx_autograd_function_process(graph)
    # 降低所有元组到单个值
    _C._jit_pass_lower_all_tuples(graph)

    # 将一些操作（如 ones/zeros）记录到跟踪中，以前这些操作会被记录为常量。
    # 使用常量传播来维持当前的 ONNX 支持水平，而不需要为所有操作实现符号计算
    if _disable_torch_constant_prop is False:
        _C._jit_pass_constant_propagation(graph)

    # 在图中执行拆分张量列表常量的操作
    _split_tensor_list_constants(graph, graph)
    # 运行 DCE（死代码消除）来清除图中可能由符号覆盖留下的未使用部分
    _C._jit_pass_dce(graph)
    # 对图进行静态分析
    _C._jit_pass_lint(graph)

    # CSE 应该在禁用缓存的情况下使用 Autocast 来提高性能
    # 由于追踪器的限制，禁用了 Autocast，详见 https://github.com/pytorch/pytorch/issues/84092
    # 必须在 _C._jit_pass_erase_number_types 之前运行，以防止类型替换
    if _C._jit_pass_cse(graph):
        _C._jit_pass_onnx_lint(graph)

    # 规范化图中融合操作的节点
    _C._jit_pass_canonicalize_graph_fuser_ops(graph)
    # 对图进行静态分析
    _C._jit_pass_lint(graph)
    # 进行 peephole 优化
    _C._jit_pass_peephole(graph, True)
    # 将 addmm 操作融合到一起
    _C._jit_pass_fuse_addmm(graph)
    # 对图进行静态分析
    _C._jit_pass_lint(graph)
    _C._jit_pass_peephole(graph, True)
    # 运行 JIT 编译器的 peephole pass，进行优化措施，处理图中的简单模式
    
    _C._jit_pass_lower_all_tuples(graph)
    # 降低图中所有元组操作的优先级，将它们转换为更基本的操作
    
    # 在 _jit_pass_onnx 中，为每个节点调用符号函数进行转换。
    # 然而，有些节点如果没有额外的上下文信息是无法转换的。
    # 例如，split 操作的输出数量（以及它是静态还是动态的）在被 listUnpack 节点解包前是未知的。
    # 此 pass 进行预处理，准备节点以便符号函数能够获得足够的上下文信息。
    _C._jit_pass_onnx_remove_inplace_ops_for_onnx(graph, module)
    _C._jit_pass_onnx_preprocess(graph)
    
    # onnx 不支持元组，因此尝试删除它们
    _C._jit_pass_lint(graph)
    
    # onnx 只支持张量，但 1 / 2 = 0.5 而 tensor(1) / tensor(2) = 0
    # 准备 division 操作以便在 onnx 中能正确运行
    _C._jit_pass_prepare_division_for_onnx(graph)
    
    _C._jit_pass_onnx_remove_print(graph)
    _C._jit_pass_onnx_preprocess_caffe2(graph)
    
    symbolic_helper._quantized_ops.clear()
    # 为 conv 和 linear 操作解包量化权重并插入到图中
    _C._jit_pass_onnx_unpack_quantized_weights(graph, params_dict)
    # onnx 只支持张量，因此将所有的数值类型转换为张量
    _C._jit_pass_erase_number_types(graph)
    
    if GLOBALS.onnx_shape_inference:
        input_names = [] if input_names is None else input_names
        dynamic_axes = {} if dynamic_axes is None else dynamic_axes
        # 设置动态输入形状以进行 onnx 导出
        _C._jit_pass_onnx_set_dynamic_input_shape(graph, dynamic_axes, input_names)
    
    # 对 onnx 图进行 lint
    _C._jit_pass_onnx_lint(graph)
    
    # 对图进行 onnx 导出，指定操作符导出类型
    graph = _C._jit_pass_onnx(graph, operator_export_type)
    # 再次对 onnx 图进行 lint
    _C._jit_pass_onnx_lint(graph)
    # 对图进行 lint
    _C._jit_pass_lint(graph)
    
    # 分析标量类型在 onnx 中的使用情况
    _C._jit_pass_onnx_scalar_type_analysis(graph, True, GLOBALS.export_onnx_opset_version)
    # 对图进行 lint
    _C._jit_pass_lint(graph)
    
    # 进行 onnx 图的 peephole pass，根据指定的 opset 版本和固定的批次大小
    _C._jit_pass_onnx_peephole(graph, GLOBALS.export_onnx_opset_version, fixed_batch_size)
    # 对图进行 lint
    _C._jit_pass_lint(graph)
    
    # 由于类型已被替换（例如 int 被替换为 Tensor），因此图不再是一个有效的 jit 图。
    # 它现在包含实际上不存在的操作符。我们不能运行普通的死代码消除，因为它会尝试查找操作符是否有副作用，
    # 但我们可以运行一个不需要查找操作符是否有副作用的死代码消除变体。
    _C._jit_pass_dce_allow_deleting_nodes_with_side_effects(graph)
    # 对图进行 lint
    _C._jit_pass_lint(graph)
    
    # 对图进行规范化，使其更符合标准形式
    graph = _C._jit_pass_canonicalize(graph)
    # 对图进行 lint
    _C._jit_pass_lint(graph)
    
    if GLOBALS.onnx_shape_inference:
        # 在 onnx 中进行图形形状类型推断
        _C._jit_pass_onnx_graph_shape_type_inference(graph, params_dict, GLOBALS.export_onnx_opset_version)
    
    # 返回处理后的图
    return graph
# 使用装饰器_beartype.beartype对函数进行类型检查
@_beartype.beartype
# 提醒用户对输入字典和字符串进行更改不会在跟踪的ONNX图中生效
def warn_on_static_input_change(input_states):
    """Warns that changes to input dictionaries and strings won't take effect in the traced ONNX graph.

    We accept dictionaries and strings as ONNX inputs, but they should be only for
    configuration use. we detect here if these inputs are modified, and if so we warn
    the user that the changes won't take effect in the traced ONNX graph.
    """
    # 遍历输入状态的两个列表，一个是原始输入，一个是跟踪的输入
    for input, traced_input in zip(input_states[0], input_states[1]):
        # 如果输入是字典
        if isinstance(input, dict):
            # 如果输入字典的键列表与跟踪输入字典的键列表不相等
            if list(input.keys()) != list(traced_input.keys()):
                # 提示用户不建议修改字典输入
                warning = (
                    "We detected that you are modifying a dictionary that is an input to your "
                    "model. "
                    "Note that dictionaries are allowed as inputs in ONNX but they should be "
                    "handled with care. "
                    "Usages of dictionaries is not recommended, and should not be used except "
                    "for configuration use. "
                    "Also note that the order and values of the keys must remain the same. "
                )
                # 发出警告
                warnings.warn(warning)
        # 如果输入是字符串
        elif isinstance(input, str):
            # 如果输入字符串与跟踪输入字符串不相等
            if input != traced_input:
                # 提示用户字符串输入/输出不会出现在ONNX图中
                warning = (
                    "The model seems to have string inputs/outputs. "
                    "Note that strings will not appear as inputs/outputs of the ONNX graph. "
                )
                # 发出警告
                warnings.warn(warning)


@_beartype.beartype
# 解决在export_type != operator_export_type.ONNX时被忽略的参数
def _resolve_args_by_export_type(arg_name, arg_value, operator_export_type):
    """Resolves the arguments that are ignored when export_type != operator_export_type.ONNX."""
    # 返回参数值
    return arg_value


@_beartype.beartype
# 决定是否将图中的初始值设为ONNX图的输入
def _decide_keep_init_as_input(
    keep_initializers_as_inputs: Optional[bool],
    operator_export_type: _C_onnx.OperatorExportTypes,
    opset_version: int,
):
    """Decides whether the initializers in the graph should be listed as ONNX graph inputs.

    This method encapsulates the logic to decide whether the initializers in the graph
    should be listed as ONNX graph inputs (i.e., whether to choose ONNX IR v3 or v4).
    If keep_initializers_as_inputs is not specified (None), then we decide whether to keep
    initializers as graph inputs (val_keep_init_as_ip) based on export type. If export type
    is ONNX, then do not keep initializers as input (val_keep_init_as_ip=False). For all other
    export types keep initializers as input (val_keep_init_as_ip=True).
    If keep_initializers_as_inputs is specified, then respect it. Unless opset version <= 8,
    in which case it must be ignored because for opset version <= 8, all initializers MUST be
    part of graph input (only ONNX IR v3 is allowed), i.e. val_keep_init_as_ip=True.

    Special handling is needed for opset version 8 or lower, because irrespective
    of user input for keep_initializers_as_inputs, the graph must follow ONNX IR v3
    """
    semantics, i.e. all initializers must be listed as ONNX graph input.
    """

    # 如果 opset_version 小于 9
    if opset_version < 9:
        # 如果 keep_initializers_as_inputs 为 False
        if keep_initializers_as_inputs is False:
            # 发出警告，因为在 opset 版本为 8 或更低时设置 keep_initializers_as_inputs=False 会导致 ONNX 图无效
            warnings.warn(
                "Setting 'keep_initializers_as_inputs=False' for opset version"
                "8 or lower would lead to an invalid ONNX graph. Therefore, "
                "'keep_initializers_as_inputs=False' is ignored during export."
                "Exported model will have initializers as graph inputs (compliant "
                " to ONNX IR v3)."
            )
        return True  # 表示初始值将作为图输入的一部分（符合 ONNX IR v3）

    # 如果 keep_initializers_as_inputs 为 None，则使用默认值 True
    val_keep_init_as_ip = (
        True if keep_initializers_as_inputs is None else keep_initializers_as_inputs
    )

    # 如果 keep_initializers_as_inputs 为 None，并且 operator_export_type 是 ONNX 类型
    if (
        keep_initializers_as_inputs is None
        and operator_export_type is _C_onnx.OperatorExportTypes.ONNX
    ):
        val_keep_init_as_ip = False

    # 返回最终的 keep_initializers_as_inputs 值
    return val_keep_init_as_ip
# 使用装饰器进行类型检查和验证
@_beartype.beartype
# 根据操作符导出类型确定是否添加节点名称
def _decide_add_node_names(add_node_names, operator_export_type):
    return _resolve_args_by_export_type(
        "add_node_names", add_node_names, operator_export_type
    )


# 使用装饰器进行类型检查和验证
@_beartype.beartype
# 根据操作符导出类型确定是否进行常量折叠优化
def _decide_constant_folding(do_constant_folding, operator_export_type, training):
    # 根据导出类型解析参数
    do_constant_folding = _resolve_args_by_export_type(
        "do_constant_folding", do_constant_folding, operator_export_type
    )
    # 如果开启了常量折叠优化且不处于评估模式下的训练中，发出警告
    if do_constant_folding and (
        training is not None and training is not _C_onnx.TrainingMode.EVAL
    ):
        warnings.warn(
            "It is recommended that constant folding be turned off ('do_constant_folding=False') "
            "when exporting the model in training-amenable mode, i.e. with 'training=TrainingMode.TRAIN' "
            "or 'training=TrainingMode.PRESERVE' (when model is in training mode). Otherwise, some "
            "learnable model parameters may not translate correctly in the exported ONNX model "
            "because constant folding mutates model parameters. Please consider "
            "turning off constant folding or setting the training=TrainingMode.EVAL."
        )
    return do_constant_folding


# 使用装饰器进行类型检查和验证
@_beartype.beartype
# 获取模型的签名信息
def _signature(model) -> inspect.Signature:
    # 检查模型是否可调用
    should_be_callable = getattr(model, "forward", model)
    if callable(should_be_callable):
        return inspect.signature(should_be_callable)
    # 抛出数值错误异常，指出模型没有forward方法且不可调用
    raise ValueError("model has no forward method and is not callable")


# 使用装饰器进行类型检查和验证
@_beartype.beartype
# 根据模型和参数解析输入格式
def _decide_input_format(model, args):
    try:
        # 获取模型的签名信息
        sig = _signature(model)
    except ValueError as e:
        # 发出警告并跳过解析输入格式的步骤
        warnings.warn(f"{e}, skipping _decide_input_format")
        return args
    try:
        # 获取参数的有序键列表
        ordered_list_keys = list(sig.parameters.keys())
        if ordered_list_keys[0] == "self":
            ordered_list_keys = ordered_list_keys[1:]
        args_dict: Dict = {}
        # 如果参数是列表或元组，则转换成列表
        if isinstance(args, list):
            args_list = args
        elif isinstance(args, tuple):
            args_list = list(args)
        else:
            args_list = [args]
        # 如果最后一个参数是字典，则将其解析为字典
        if isinstance(args_list[-1], dict):
            args_dict = args_list[-1]
            args_list = args_list[:-1]
        n_nonkeyword = len(args_list)
        # 处理非关键字参数
        for optional_arg in ordered_list_keys[n_nonkeyword:]:
            if optional_arg in args_dict:
                args_list.append(args_dict[optional_arg])
            # 检查该参数是否有默认值
            else:
                param = sig.parameters[optional_arg]
                if param.default != param.empty:
                    args_list.append(param.default)
        args = args_list if isinstance(args, list) else tuple(args_list)
    # 处理没有输入参数的模型情况
    except IndexError:
        warnings.warn("No input args, skipping _decide_input_format")
    except Exception as e:
        warnings.warn(f"Skipping _decide_input_format\n {e.args[0]}")
    return args


# 使用装饰器进行类型检查和验证
@_beartype.beartype
# 将动态轴转换为动态形状
def _from_dynamic_axes_to_dynamic_shapes(
    model,
    dynamic_axes: Optional[
        Union[Mapping[str, Mapping[int, str]], Mapping[str, Sequence[int]]]
    ] = None,
    input_names: Optional[Sequence[str]] = None,


# 定义 dynamic_axes 变量，它是一个可选的类型注解，可以是两种形式之一：
# 1. 字符串到字典的映射，字典的键是整数，值是字符串
# 2. 字符串到整数序列的映射
# 如果未提供值，则默认为 None
dynamic_axes: Optional[
    Union[Mapping[str, Mapping[int, str]], Mapping[str, Sequence[int]]]
] = None,

# 定义 input_names 变量，它是一个可选的类型注解，类型为字符串序列
# 如果未提供值，则默认为 None
input_names: Optional[Sequence[str]] = None,
def _trace_and_get_graph_from_model(model: Any, args: Tuple[Any, ...]) -> Optional[Dict[str, Optional[Any]]]:
    """
    从模型中跟踪并获取计算图。

    Args:
        model: 要跟踪的模型。
        args: 传递给模型的参数元组。

    Returns:
        Optional[Dict[str, Optional[Any]]]: 动态维度信息字典，用于描述模型输入的形状。
            如果获取过程中出现问题，则返回 None。

    Notes:
        此函数的核心是跟踪模型并获取其计算图，以用于后续的导出或优化操作。
    """

    # 进行模型状态字典的基本完整性检查
    orig_state_dict_keys = torch.jit._unique_state_dict(model).keys()

    # 禁用 Autocast 缓存，因为它会替换核心的权重和偏置为（不必要的）常量。
    # 这一步是为了确保模型运行前后状态字典的键保持一致，以防出现潜在的错误。
    # 缓存自动混合精度状态，以便稍后恢复
    prev_autocast_cache_enabled = torch.is_autocast_cache_enabled()
    
    # 禁用自动混合精度缓存，以确保跟踪图的一致性
    torch.set_autocast_cache_enabled(False)
    
    # 获取模型的跟踪图、Torch 输出以及输入状态
    trace_graph, torch_out, inputs_states = torch.jit._get_trace_graph(
        model,
        args,
        strict=False,
        _force_outplace=False,
        _return_inputs_states=True,
    )
    
    # 恢复之前的自动混合精度缓存状态
    torch.set_autocast_cache_enabled(prev_autocast_cache_enabled)
    
    # 检查静态输入是否发生变化，给出警告
    warn_on_static_input_change(inputs_states)
    
    # 如果模型的状态字典键发生变化，则引发运行时错误
    if orig_state_dict_keys != torch.jit._unique_state_dict(model).keys():
        raise RuntimeError(
            "state_dict changed after running the tracer; "
            "something weird is happening in your model!"
        )
    
    # 返回模型的跟踪图、Torch 输出
    return trace_graph, torch_out
# 使用装饰器进行参数类型检查和类型注解
@_beartype.beartype
# 定义函数，获取方法图和参数列表，返回参数计数列表
def _get_param_count_list(method_graph, args_params):
    # 初始化空的参数计数列表
    param_count_list = []
    # 遍历方法图的输入和参数列表的对应关系
    for input_, arg_params_ in zip(method_graph.inputs(), args_params):
        # 检查输入的类型名中是否包含"PackedParams"
        if "PackedParams" in str(input_.type()):
            # 如果包含，使用 torch.jit._flatten 将参数扁平化，并获取其中变量的数量
            in_vars, _ = torch.jit._flatten(arg_params_)
            param_count_list.append(len(in_vars))
        else:
            # 如果不包含"PackedParams"，直接判断 arg_params 是否为 None，并将结果添加到列表中
            param_count_list.append(arg_params_ is not None)

    # 返回参数计数列表
    return param_count_list


# 使用装饰器进行参数类型检查和类型注解
@_beartype.beartype
# 定义函数，检查 torch.jit._flatten 是否移除了 None
def _check_flatten_did_not_remove(original, jit_flattened):
    """torch.jit._flatten removes None. Check if it did so in this case."""

    # 定义嵌套函数 flatten，用于递归地展开列表、元组和字典，生成扁平化后的元素列表
    @_beartype.beartype
    def flatten(x):
        if isinstance(x, (list, tuple)):
            for inner in x:
                yield from flatten(inner)
        elif isinstance(x, dict):
            for inner in x.values():
                yield from flatten(inner)
        else:
            yield x

    # 将原始参数扁平化，包括 None
    flattened_with_none = list(flatten(original))
    # 计算扁平化后包含的 None 的数量
    num_none = len(flattened_with_none) - len(jit_flattened)
    # 断言 None 的数量不少于 0
    assert num_none >= 0
    # 如果存在 None，则抛出 ValueError 异常
    if num_none:
        raise ValueError(
            f"args contained {num_none} None's after flattening. "
            "When exporting a ScriptModule or ScriptFunction, no args may "
            "be None because that breaks type propagation."
        )


# 定义函数，创建 JIT 图
def _create_jit_graph(
    model: Union[torch.nn.Module, torch.jit.ScriptFunction], args: Sequence[Any]
) -> Tuple[_C.Graph, List[_C.IValue], Optional[Any], Optional[_C.ScriptModule]]:
    # 检查 model 是否为 torch.jit.ScriptFunction 或 torch.jit.ScriptModule 的实例
    if isinstance(model, (torch.jit.ScriptFunction, torch.jit.ScriptModule)):
        # 将输入参数 args 进行扁平化处理并返回扁平化后的元组
        flattened_args = tuple(torch.jit._flatten(tuple(args))[0])
        # 检查扁平化处理是否正确，确保没有移除任何参数
        _check_flatten_did_not_remove(args, flattened_args)
        # 初始化 torch_out 为 None
        torch_out = None

        # 如果 model 是 torch.jit.ScriptModule 的实例
        if isinstance(model, torch.jit.ScriptModule):
            try:
                # 获取模型的 forward 方法的图形表示
                graph = model.forward.graph  # type: ignore[attr-defined]
            except AttributeError as e:
                # 如果没有找到 forward 方法，抛出运行时错误
                raise RuntimeError("'forward' method must be a script method") from e
            # 替换模型中的 ONNX 函数
            _C._jit_pass_onnx_function_substitution(graph)
            # 冻结模型中的参数并返回冻结后的模型及其参数
            freezed_module = _C._freeze_module(
                cast(_C.ScriptModule, model._c), preserveParameters=True
            )
            module, params = _C._jit_onnx_list_model_parameters(freezed_module)
            # 获取 forward 方法的图形表示
            method_graph = module._get_method("forward").graph
            # 将输入参数 args 与模型参数 params 合并
            args_params = tuple(args) + tuple(params)
            # 获取参数数量列表
            param_count_list = _get_param_count_list(method_graph, args_params)
            # 扁平化输入参数与模型参数
            in_vars, _ = torch.jit._flatten(args_params)
            # 传播和分配输入形状到方法图形表示中
            graph = _C._propagate_and_assign_input_shapes(
                method_graph, tuple(in_vars), param_count_list, False, False
            )
            # 返回处理后的图形表示、模型参数、torch_out 以及模块
            return graph, params, torch_out, module

        # 如果 model 是 torch.jit.ScriptFunction 的实例
        # 处理 torch.jit.ScriptFunction 的情况
        params = []
        # 获取模型的图形表示
        graph = model.graph
        # 替换模型中的 ONNX 函数
        _C._jit_pass_onnx_function_substitution(graph)
        # 获取参数数量列表
        param_count_list = _get_param_count_list(graph, args)
        # 传播和分配输入形状到图形表示中
        graph = _C._propagate_and_assign_input_shapes(
            graph, flattened_args, param_count_list, False, False
        )
        # 返回处理后的图形表示、参数、torch_out 以及空模块
        return graph, params, torch_out, None

    # 如果 model 不是 torch.jit.ScriptFunction 或 torch.jit.ScriptModule 的实例
    # 执行模型跟踪并从模型获取图形表示
    graph, torch_out = _trace_and_get_graph_from_model(model, args)
    # 对图形表示进行 ONNX 的 lint 检查
    _C._jit_pass_onnx_lint(graph)
    # 获取模型的唯一状态字典
    state_dict = torch.jit._unique_state_dict(model)
    # 将状态字典的值转换为列表，作为参数
    params = list(state_dict.values())
    # 获取图形表示的输入
    graph_inputs = list(graph.inputs())
    # 计算用户输入数量
    user_input_num = len(graph_inputs) - len(state_dict)
    # 获取状态字典的键作为参数名称
    param_names = list(state_dict.keys())
    # 为图形表示中的每个输入设置调试名称
    for i, inp in enumerate(graph_inputs):
        if i >= user_input_num:
            inp.setDebugName(param_names[i - user_input_num])
    # 替换模型中的 ONNX 函数
    _C._jit_pass_onnx_function_substitution(graph)
    # 返回处理后的图形表示、参数、torch_out 以及空模块
    return graph, params, torch_out, None
# 使用装饰器进行参数检查和类型注解
@_beartype.beartype
# 根据图形和参数生成命名参数字典
def _get_named_param_dict(graph, params):
    # 获取图形中输入节点的调试名称列表
    input_and_param_names = [val.debugName() for val in graph.inputs()]
    # 根据参数列表确定参数名称列表
    param_names = input_and_param_names[len(input_and_param_names) - len(params) :]
    # 将参数名称与参数值对应起来，生成参数字典
    _params_dict = dict(zip(param_names, params))
    return _params_dict


# 使用装饰器进行参数检查和类型注解
@_beartype.beartype
# 获取模型的示例输出
def _get_example_outputs(model, args):
    # 深度复制参数列表
    input_args = copy.deepcopy(args)
    input_kwargs = {}
    # 如果最后一个参数是字典，则作为关键字参数处理
    if input_args and isinstance(input_args[-1], dict):
        input_kwargs = input_args[-1]
        input_args = input_args[:-1]

    # 调用模型并获取示例输出
    example_outputs = model(*input_args, **input_kwargs)
    # 如果示例输出是列表，则转换为列表形式
    if isinstance(example_outputs, list):
        example_outputs = [example_outputs]
    # 如果示例输出不是元组，则转换为单元素元组
    elif not isinstance(example_outputs, tuple):
        example_outputs = (example_outputs,)

    return example_outputs


# 定义量化类型到对应数据类型的映射
_qtype_vtype_map = {
    torch.quint8: torch.uint8,
    torch.qint8: torch.int8,
    torch.qint32: torch.int32,
    torch.quint4x2: torch.int8,
}


# 使用装饰器进行参数检查和类型注解
@_beartype.beartype
# 解包量化的张量
def unpack_quantized_tensor(value, cast_onnx_accepted=True):
    # 如果值是张量且数据类型在映射表中
    if isinstance(value, torch.Tensor) and value.dtype in _qtype_vtype_map:
        # 对量化的值进行反量化
        q_value_dequantize = value.dequantize()
        # 获取量化的尺度
        q_scale = (
            torch.tensor(value.q_scale(), dtype=torch.double)
            if cast_onnx_accepted
            else torch.tensor(value.q_scale(), dtype=torch.float32)
        )
        # 获取量化的零点
        q_zero_point = (
            torch.tensor(value.q_zero_point(), dtype=torch.int64)
            if cast_onnx_accepted
            else torch.tensor(value.q_zero_point(), dtype=_qtype_vtype_map[value.dtype])
        )
        # 计算反量化后的值
        q_value = q_value_dequantize / q_scale + q_zero_point
        # 将结果转换为目标数据类型
        q_value = q_value.to(dtype=_qtype_vtype_map[value.dtype])
        return q_value, q_scale, q_zero_point
    else:
        return (value,)


# 使用装饰器进行参数检查和类型注解
@_beartype.beartype
# 预处理量化模型进行跟踪
def _pre_trace_quant_model(model, args):
    r"""如果模型是量化的，则返回 `torch.jit.trace(model, args)`。否则不做处理并返回原始模型。

    这是由于 https://github.com/pytorch/pytorch/issues/75761 的问题。
    """
    # 如果模型中任何模块具有 "_packed_params" 属性，或者任何参数是量化的，则进行跟踪
    if any(
        hasattr(m, "_packed_params") for m in getattr(model, "modules", list)()
    ) or any(getattr(arg, "is_quantized", False) for arg in args):
        return torch.jit.trace(model, args)
    # 否则返回原始模型
    return model


# 使用装饰器进行参数检查和类型注解
@_beartype.beartype
# 将模型转换为图形表示
def _model_to_graph(
    model,
    args,
    verbose=False,
    input_names=None,
    output_names=None,
    operator_export_type=_C_onnx.OperatorExportTypes.ONNX,
    do_constant_folding=True,
    _disable_torch_constant_prop=False,
    fixed_batch_size=False,
    training=_C_onnx.TrainingMode.EVAL,
    dynamic_axes=None,
) -> Tuple[
    _C.Graph,
    Dict[str, torch.Tensor],
    Optional[
        Union[
            torch.Tensor,
            Tuple[torch.Tensor, ...],
            List[torch.Tensor],
            Dict[str, torch.Tensor],
            Any,  # 可以是嵌套的元组等
        ]
    ],
]:
    """将模型转换为 ONNX 图形表示。
    
    Args:
        model: 要转换的模型
        args: 用于调用模型的参数
        verbose: 是否显示详细信息
        input_names: 输入名称列表
        output_names: 输出名称列表
        operator_export_type: 操作符导出类型
        do_constant_folding: 是否进行常量折叠
        _disable_torch_constant_prop: 是否禁用 Torch 常量优化
        fixed_batch_size: 是否使用固定的批处理大小
        training: ONNX 模型的训练模式
        dynamic_axes: 动态轴配置

    Returns:
        Tuple[_C.Graph, Dict[str, torch.Tensor], Optional[Union[...]]]: ONNX 图形表示、输入映射字典、可选的输出

    """
    # 将模型转换为 ONNX 图形表示
    return torch.onnx._model_to_graph(
        model, args, verbose, input_names, output_names, operator_export_type,
        do_constant_folding, _disable_torch_constant_prop, fixed_batch_size, training, dynamic_axes
    )
    # TODO: can we simplify this to always return a tuple of Tensor or None?

    # 对于常见情况下传递单个 Tensor 的特殊处理
    if isinstance(args, (torch.Tensor, int, float, bool)):
        args = (args,)

    # 预处理量化模型，返回处理后的模型对象
    model = _pre_trace_quant_model(model, args)
    # 创建 JIT 图并返回图、参数、Torch 输出和模块
    graph, params, torch_out, module = _create_jit_graph(model, args)
    # 获取图中的命名参数字典
    params_dict = _get_named_param_dict(graph, params)

    try:
        # 优化图的结构
        graph = _optimize_graph(
            graph,
            operator_export_type,
            _disable_torch_constant_prop=_disable_torch_constant_prop,
            fixed_batch_size=fixed_batch_size,
            params_dict=params_dict,
            dynamic_axes=dynamic_axes,
            input_names=input_names,
            module=module,
        )
    except Exception as e:
        # 在异常情况下记录 Torch IR 图的状态
        torch.onnx.log("Torch IR graph at exception: ", graph)
        raise

    # 检查模型是否为脚本模块或脚本函数
    is_script = isinstance(model, (torch.jit.ScriptFunction, torch.jit.ScriptModule))
    if is_script:
        # 获取模型的示例输出并展开量化张量
        example_outputs = _get_example_outputs(model, args)
        example_outputs_final = ()
        for example_output in example_outputs:
            example_outputs_final += unpack_quantized_tensor(example_output)
        # 展平示例输出并获取输出变量和描述
        out_vars, desc = torch.jit._flatten(example_outputs_final)
        # 在 ONNX 中为输出变量分配输出形状
        _C._jit_pass_onnx_assign_output_shape(
            graph,
            out_vars,
            desc,
            GLOBALS.onnx_shape_inference,
            is_script,
            GLOBALS.export_onnx_opset_version,
        )
    else:
        # 如果不是脚本模块或脚本函数，则处理 Torch 输出
        if not isinstance(torch_out, (list, tuple)):
            output_wrapped = [torch_out]
        else:
            output_wrapped = torch_out  # type: ignore[assignment]

        # 展平 Torch 输出并获取输出张量和描述
        output_tensors, out_desc = torch.jit._flatten(tuple(output_wrapped))
        # 如果没有量化输出，则在 ONNX 中为输出张量分配输出形状
        if not any(getattr(out, "is_quantized", False) for out in output_tensors):
            _C._jit_pass_onnx_assign_output_shape(
                graph,
                output_tensors,
                out_desc,
                GLOBALS.onnx_shape_inference,
                is_script,
                GLOBALS.export_onnx_opset_version,
            )

    # 设置输入和输出的名称到图中
    _set_input_and_output_names(graph, input_names, output_names)
    # 获取更新后的命名参数字典
    params_dict = _get_named_param_dict(graph, params)
    # 如果启用常量折叠，并且导出的 ONNX opset 版本大于等于 ONNX_CONSTANT_FOLDING_MIN_OPSET
    if (
        do_constant_folding
        and GLOBALS.export_onnx_opset_version
        >= _constants.ONNX_CONSTANT_FOLDING_MIN_OPSET
    ):
        # 如果处于训练模式为 None 或者 EVAL 模式
        if training is None or training == _C_onnx.TrainingMode.EVAL:
            # 对 ONNX 图进行 peephole 优化
            params_dict = _C._jit_pass_onnx_eval_peephole(graph, params_dict)

        # 对 ONNX 图中的常量进行折叠
        params_dict = _C._jit_pass_onnx_constant_fold(
            graph, params_dict, GLOBALS.export_onnx_opset_version
        )
        # 删除具有副作用节点的死代码
        _C._jit_pass_dce_allow_deleting_nodes_with_side_effects(graph)

    # 如果启用 ONNX 图形形状推断
    if GLOBALS.onnx_shape_inference:
        # 执行 ONNX 图形形状和类型推断
        _C._jit_pass_onnx_graph_shape_type_inference(
            graph, params_dict, GLOBALS.export_onnx_opset_version
        )

    # 消除 ONNX 图中未使用的项目
    params_dict = _C._jit_pass_onnx_eliminate_unused_items(graph, params_dict)

    # 对于 ONNX opset 版本小于 9，将常量转换为浮点数或双精度数值类型
    if GLOBALS.export_onnx_opset_version < 9:
        _C._jit_pass_onnx_cast_all_constant_to_floating(graph)

    # 过滤非张量参数
    params_dict = _C._jit_pass_filter_non_tensor_arguments(params_dict)
    # 降低打包参数输入类型的优先级
    _C._jit_decay_packed_param_input_types(graph)

    # 为了调试方便，如果输出名称不合适且仅由其唯一标识识别，则给它们一个易读的名称
    _apply_friendly_debug_names(graph, params_dict)

    # 返回处理后的图、参数字典和 torch 输出
    return graph, params_dict, torch_out
# 使用 @_beartype 装饰器对函数进行类型检查和类型提示
# 使用 torch._disable_dynamo 函数装饰器来禁用 Torch 的动态图模式
def export_to_pretty_string(
    model,
    args,
    export_params=True,
    verbose=False,
    training=_C_onnx.TrainingMode.EVAL,  # 指定 ONNX 导出时的训练模式，默认为 EVAL
    input_names=None,  # 模型输入的名称列表
    output_names=None,  # 模型输出的名称列表
    operator_export_type=_C_onnx.OperatorExportTypes.ONNX,  # 指定操作符导出类型，默认为 ONNX
    export_type=None,  # 导出类型，与 export 函数保持一致
    google_printer=False,  # 是否使用 Google 打印器打印更详细的模型表示
    opset_version=None,  # 指定 ONNX 操作集的版本，默认使用常量 ONNX_DEFAULT_OPSET
    keep_initializers_as_inputs=None,  # 是否将初始化器保持为输入
    custom_opsets=None,  # 自定义的操作集合
    add_node_names=True,  # 是否为 NodeProto 设置名称，仅在 google_printer=True 时有影响
    do_constant_folding=True,  # 是否进行常量折叠
    dynamic_axes=None,  # 动态轴列表
):
    r"""
    类似于 :func:`export`，但返回 ONNX 模型的文本表示形式。以下是与原函数不同的参数说明，其余参数与 :func:`export` 相同。

    Args:
        add_node_names (bool, default True): 是否设置 NodeProto.name。仅在 ``google_printer=True`` 时生效。
        google_printer (bool, default False): 如果为 False，则返回自定义、紧凑的模型表示；如果为 True，则返回更详细的 protobuf `Message::DebugString()`。

    Returns:
        返回一个 UTF-8 字符串，包含 ONNX 模型的人类可读表示形式。
    """
    # 如果 opset_version 未指定，则使用默认的 ONNX 操作集版本
    if opset_version is None:
        opset_version = _constants.ONNX_DEFAULT_OPSET
    # 如果 custom_opsets 未指定，则初始化为空字典
    if custom_opsets is None:
        custom_opsets = {}
    
    # 设置全局变量，指定导出的 ONNX 操作集版本和操作符导出类型
    GLOBALS.export_onnx_opset_version = opset_version
    GLOBALS.operator_export_type = operator_export_type

    # 使用 exporter_context 上下文管理器，设定模型导出的相关参数
    with exporter_context(model, training, verbose):
        # 确定是否将初始化器保持为输入
        val_keep_init_as_ip = _decide_keep_init_as_input(
            keep_initializers_as_inputs, operator_export_type, opset_version
        )
        # 确定是否添加节点名称
        val_add_node_names = _decide_add_node_names(
            add_node_names, operator_export_type
        )
        # 确定是否进行常量折叠
        val_do_constant_folding = _decide_constant_folding(
            do_constant_folding, operator_export_type, training
        )
        # 确定输入参数的格式
        args = _decide_input_format(model, args)
        # 将模型转换为图形表示，并获取图、参数字典和 Torch 输出
        graph, params_dict, torch_out = _model_to_graph(
            model,
            args,
            verbose,
            input_names,
            output_names,
            operator_export_type,
            val_do_constant_folding,
            training=training,
            dynamic_axes=dynamic_axes,
        )

        # 调用 graph 对象的 _pretty_print_onnx 方法，生成 ONNX 模型的人类可读表示
        return graph._pretty_print_onnx(  # type: ignore[attr-defined]
            params_dict,
            opset_version,
            False,
            operator_export_type,
            google_printer,
            val_keep_init_as_ip,
            custom_opsets,
            val_add_node_names,
        )


# 使用 @_beartype 装饰器对函数进行类型检查和类型提示
def unconvertible_ops(
    model,
    args,
    training: _C_onnx.TrainingMode = _C_onnx.TrainingMode.EVAL,  # 指定 ONNX 导出时的训练模式，默认为 EVAL
    opset_version: Optional[int] = None,  # 指定 ONNX 操作集的版本号，可选
) -> Tuple[_C.Graph, List[str]]:
    """返回一个近似列表，其中包含所有尚未由 :mod:`torch.onnx` 支持的操作。

    此列表是近似的，因为某些操作在转换过程中可能会被移除。

    返回一个元组，包含一个 _C.Graph 对象和一个字符串列表。
    """
    def discover_unconvertible_ops(model, args, training=False, opset_version=None):
        """
        Discover unconvertible operations in a PyTorch model when exporting to ONNX.
    
        Args:
            model: Same as the `model` parameter in :func:`torch.onnx.export`.
            args: Same as the `args` parameter in :func:`torch.onnx.export`.
            training: Same as the `training` parameter in :func:`torch.onnx.export`.
            opset_version: Same as the `opset_version` parameter in :func:`torch.onnx.export`.
    
        Returns:
            The JIT graph and a list of unconvertible ops in the format of "domain::op".
        """
    
        # Set the ONNX opset version to use
        opset_version = opset_version or _constants.ONNX_DEFAULT_OPSET
        GLOBALS.export_onnx_opset_version = opset_version
    
        try:
            with exporter_context(model, training, verbose=False):
                # Create a mostly clean JIT graph that contains the plain aten and
                # other ops we can check with the symbolic registry.
                # NOTE: We don't want to actually convert any ops to ONNX or run any
                # symbolic functions because there is a higher chance that a pass
                # fails or an unconvertible op messes up the graph during ONNX conversion.
                # This way we can always generate a list just by looking at the names
                # of the ops in the graph.
    
                # Decide the input format for the model
                args = _decide_input_format(model, args)
                # Prepare the model for tracing and quantization (if needed)
                model = _pre_trace_quant_model(model, args)
                # Create a JIT graph for the model
                graph, _, _, module = _create_jit_graph(model, args)
                # Inline the functions in the JIT graph
                _C._jit_pass_inline(graph)
                # Remove inplace operations for ONNX compatibility
                _C._jit_pass_onnx_remove_inplace_ops_for_onnx(graph, module)
                # Erase number types from the graph
                _C._jit_pass_erase_number_types(graph)
                # Allow deleting nodes with side effects from the graph
                _C._jit_pass_dce_allow_deleting_nodes_with_side_effects(graph)
        except Exception as e:
            # Handle errors during JIT graph generation
            raise errors.OnnxExporterError(
                "Failed to discover unconvertible ops because of errors during the JIT graph "
                "generation process."
            ) from e
    
        # Initialize list to store unsupported ops
        unsupported_ops = []
        # Iterate through nodes in the JIT graph
        for node in graph.nodes():
            domain_op = node.kind()
            # Check if the operation belongs to 'onnx::' or 'prim::' domain
            if domain_op.startswith(("onnx::", "prim::")):
                # Supported operations; continue to next node
                continue
            # Check if the operation is registered for the given opset version
            if not registration.registry.is_registered_op(
                domain_op.rstrip("_"), opset_version
            ):
                # Operation is not fully supported; add to unsupported_ops list
                # TODO(justinchuby): Create a way to check if an op is fully supported.
                unsupported_ops.append(domain_op)
    
        # Return the JIT graph and the list of unsupported operations
        return graph, unsupported_ops
@_beartype.beartype
def _setup_trace_module_map(
    model: Union[torch.nn.Module, torch.jit.ScriptModule],
    export_modules_as_functions: Union[bool, Collection[Type[torch.nn.Module]]],
) -> Set[str]:
    """
    设置跟踪模块映射表，并返回导出模块的类型名称集合。

    Args:
        model: 要设置跟踪的模型，可以是 `torch.nn.Module` 或 `torch.jit.ScriptModule` 类型。
        export_modules_as_functions: 是否导出模块作为函数，可以是布尔值或 `torch.nn.Module` 类型集合。

    Returns:
        module_typenames: 导出模块的类型名称集合。

    """
    def __register_attribute_hook():
        """
        注册属性钩子函数 `_track_module_attributes_forward_pre_hook` 和 `_track_module_attributes_forward_hook`。
        """
        attr_name = "_onnx_attrs"

        def _track_module_attributes_forward_pre_hook(module, input):
            """
            模块前向预处理钩子函数，用于跟踪模块属性。
            """
            setattr(module, attr_name, _get_module_attributes(module))

        def _track_module_attributes_forward_hook(module, input, output):
            """
            模块前向钩子函数，用于跟踪模块属性，并将属性传递给 ONNX 图。
            """
            tracing_state = _C._get_tracing_state()
            if not tracing_state:
                return

            graph = tracing_state.graph()
            onnx_attrs = {}
            if hasattr(module, attr_name):
                onnx_attrs = getattr(module, attr_name)
                delattr(module, attr_name)

            _C._jit_pass_onnx_track_scope_attributes(graph, onnx_attrs)

        for m in model.modules():
            m.register_forward_hook(_track_module_attributes_forward_hook)
            m.register_forward_pre_hook(_track_module_attributes_forward_pre_hook)

    def _unqualified_variable_name(qualified_name: str) -> str:
        """
        解析限定变量名，并返回未限定版本。

        纯数字原子被视为不足够，因此此函数将跳过它们，并从第一个非数字原子开始。

        Example:
            >>> _unqualified_variable_name('__main__.Foo.bar')
            'bar'
            >>> _unqualified_variable_name('__main__.Foo.bar.0')
            'bar.0'
        """
        name_atoms = qualified_name.split(".")
        for i, atom in reversed(list(enumerate(name_atoms))):
            if not atom.isnumeric():
                return ".".join(name_atoms[i:])
        return qualified_name

    trace_module_map = {
        _m: torch._C._jit_onnx_create_full_scope_name(
            torch.typename(type(_m)), _unqualified_variable_name(_n)
        )
        for _n, _m in model.named_modules()
    }
    torch.jit._trace._trace_module_map = trace_module_map
    if isinstance(export_modules_as_functions, bool) and export_modules_as_functions:
        module_typenames = {torch.typename(type(module)) for module in trace_module_map}
    elif isinstance(export_modules_as_functions, set) and export_modules_as_functions:

        def _find_typename(v):
            if isinstance(v, type):
                return torch.typename(v)
            else:
                raise RuntimeError(
                    "Only type of the `nn.Module` should be "
                    "passed in the set for argument `export_modules_as_functions`. "
                    f"Got `{type(v).__name__}`."
                )

        module_typenames = {_find_typename(v) for v in export_modules_as_functions}
    else:
        module_typenames = set()

    if module_typenames:
        __register_attribute_hook()

    return module_typenames


@_beartype.beartype
def _reset_trace_module_map():
    """
    重置跟踪模块映射表。
    """
    # 设置 torch.jit 模块下的 _trace_module_map 为 None，清空跟踪模块映射
    torch.jit._trace._trace_module_map = None
    # 调用 C++ 扩展模块 _C 中的函数 _jit_pass_onnx_clear_scope_records，清除 ONNX 作用域记录
    _C._jit_pass_onnx_clear_scope_records()
# 使用装饰器进行类型检查，确保输入的模块(module)具有正确的类型注解
@_beartype.beartype
def _get_module_attributes(module):
    # 获取模块类型的类型提示信息
    annotations = typing.get_type_hints(type(module))
    # 获取torch.nn.Module的类型提示信息
    base_m_annotations = typing.get_type_hints(torch.nn.Module)
    # 移除base_m_annotations中的键，这些键对应的属性是在annotations中定义的但没有在构造函数中创建的
    [annotations.pop(k, None) for k in base_m_annotations]

    # 检查模块属性是否可访问。有些类定义了属性但在构造函数中没有提供对它们的访问。
    #
    # 例如，torch.nn.Embedding 类有 `freeze` 变量及其类型在类中指定，
    # 但该属性并未在构造函数中创建。换句话说，构造函数中没有 `self.freeze = <True | False>` 这样的语句。
    #
    # 参考链接: https://github.com/pytorch/pytorch/blob/92de1d322223fb5584e384971b32c46b93bc2f4b/torch/nn/modules/sparse.py#L120
    attrs = {}
    for k in annotations:
        try:
            # 获取模块(module)的属性值，并添加到attrs字典中
            attrs[k] = getattr(module, k)
        except AttributeError:
            # 如果属性不存在，则记录日志并继续下一个属性的处理
            torch.onnx.log(f"Skipping module attribute '{k}'")
            continue
    return attrs


# 使用装饰器进行类型检查，确保导出函数的参数具有正确的类型注解
@_beartype.beartype
def _export(
    model,
    args,
    f,
    export_params=True,
    verbose=False,
    training=_C_onnx.TrainingMode.EVAL,
    input_names=None,
    output_names=None,
    operator_export_type=_C_onnx.OperatorExportTypes.ONNX,
    export_type=None,
    opset_version=None,
    do_constant_folding=True,
    dynamic_axes=None,
    keep_initializers_as_inputs=None,
    fixed_batch_size=False,
    custom_opsets=None,
    add_node_names=True,
    onnx_shape_inference=True,
    export_modules_as_functions=False,
    autograd_inlining=True,
):
    # 确保不在ONNX导出模式下
    assert GLOBALS.in_onnx_export is False

    # 如果未指定导出类型，默认使用PROTOBUF_FILE格式
    if export_type is None:
        export_type = _exporter_states.ExportTypes.PROTOBUF_FILE

    # 如果模型是torch.nn.DataParallel类型，则抛出异常，因为ONNX导出器不支持DataParallel模型
    if isinstance(model, torch.nn.DataParallel):
        raise ValueError(
            "torch.nn.DataParallel is not supported by ONNX "
            "exporter, please use 'attribute' module to "
            "unwrap model from torch.nn.DataParallel. Try "
            "torch.onnx.export(model.module, ...)"
        )

    # 设置全局变量，指示是否进行ONNX形状推断
    GLOBALS.onnx_shape_inference = onnx_shape_inference

    # 如果未指定opset版本，默认使用ONNX_DEFAULT_OPSET
    if opset_version is None:
        opset_version = _constants.ONNX_DEFAULT_OPSET

    # torch.onnx.export不支持opset版本>=18
    # 因此这里确保opset_version不超过18
    # 检查所选的 ONNX 操作集版本是否超出支持范围
    if opset_version > _constants.ONNX_TORCHSCRIPT_EXPORTER_MAX_OPSET:
        # 如果超出支持范围，发出警告
        warnings.warn(
            f"Exporting to ONNX opset version {opset_version} is not supported. "
            f"by 'torch.onnx.export()'. "
            f"The highest opset version supported is {_constants.ONNX_TORCHSCRIPT_EXPORTER_MAX_OPSET}. "
            f"To use a newer opset version, consider 'torch.onnx.dynamo_export()'. "
            f"Note that dynamo_export() is in preview. Please report errors with "
            f"dynamo_export() as Github issues to https://github.com/pytorch/pytorch/issues.",
            category=errors.OnnxExporterWarning,
        )

    # 如果选择将模块导出为函数，并且操作集版本小于 15，则抛出 ValueError
    if export_modules_as_functions and opset_version < 15:
        raise ValueError(
            "`export_modules_as_functions` is not supported for `opset_version` < 15."
            "This is because `opset_version` < 15 implies IR version < 8, which means "
            "no local function support. "
        )

    # 如果未指定 operator_export_type，则将其设为默认值 _C_onnx.OperatorExportTypes.ONNX
    if not operator_export_type:
        operator_export_type = _C_onnx.OperatorExportTypes.ONNX

    # 将全局变量中的 ONNX 导出操作集版本设置为指定的 opset_version
    GLOBALS.export_onnx_opset_version = opset_version
    # 将全局变量中的 operator_export_type 设置为指定的操作类型
    GLOBALS.operator_export_type = operator_export_type

    # 在结束导出过程时执行以下操作
    finally:
        # 断言当前处于 ONNX 导出状态
        assert GLOBALS.in_onnx_export
        # 将全局变量中的 in_onnx_export 标志位设为 False
        GLOBALS.in_onnx_export = False
        # 恢复自动求导内联设置为之前的状态
        GLOBALS.autograd_inlining = _autograd_inlining_previous
        # 重置跟踪模块映射表
        _reset_trace_module_map()

    # 返回导出的 torch_out 结果
    return torch_out
# 应用 Beartype 装饰器来检查函数参数类型
@_beartype.beartype
# 为图中每个节点的输入设置友好的调试名称
def _apply_friendly_debug_names(graph, params):
    # 遍历图中的每个节点
    for n in graph.nodes():
        # 遍历节点的每个输入变量
        for v in n.inputs():
            # 获取当前输入变量的调试名称
            old_name = v.debugName()
            # 如果调试名称不等于其唯一标识符的字符串形式，则跳过
            if old_name != str(v.unique()):
                continue
            # 构造新的调试名称，格式为 "<节点类型>_<唯一标识符>"
            new_name = f"{n.kind()}_{v.unique()}"
            # 设置新的调试名称
            v.setDebugName(new_name)
            # 如果旧名称存在于参数字典中，则更新为新的调试名称
            if old_name in params:
                params[new_name] = params.pop(old_name)


# 应用 Beartype 装饰器来检查函数参数类型
@_beartype.beartype
# 设置图的输入和输出节点名称
def _set_input_and_output_names(graph, input_names, output_names):
    # 内部函数定义，用于设置节点的名称
    @_beartype.beartype
    def set_names(node_list, name_list, descriptor):
        # 如果名称列表为 None，则直接返回
        if name_list is None:
            return
        # 如果名称列表长度大于节点列表长度，则抛出运行时异常
        if len(name_list) > len(node_list):
            raise RuntimeError(
                "number of %s names provided (%d) exceeded number of %ss (%d)"
                % (descriptor, len(name_list), descriptor, len(node_list))
            )

        # 如果是输出节点，用于记录已设置调试名称的节点
        output_node_set = set()
        # 遍历名称列表和节点列表，设置节点的调试名称
        for i, (name, node) in enumerate(zip(name_list, node_list)):
            # 如果是输出节点且节点已经在集合中，则插入 "onnx::Identity" 节点以避免重复设置调试名称
            if descriptor == "output":
                if node in output_node_set:
                    identity_node = graph.create("onnx::Identity")
                    identity_node.insertAfter(node.node())
                    identity_node.addInput(node)
                    identity_node.output().setType(node.type())
                    graph.return_node().replaceInput(i, identity_node.output())
                    node = identity_node.output()
                output_node_set.add(node)

            # 如果节点的调试名称不等于目标名称，则设置为目标名称
            if node.debugName() != name:
                node.setDebugName(name)

    # 设置输入节点的名称
    set_names(list(graph.inputs()), input_names, "input")
    # 设置输出节点的名称
    set_names(list(graph.outputs()), output_names, "output")


# 应用 Beartype 装饰器来检查函数参数类型
@_beartype.beartype
# 运行符号方法的跳板函数，每次从 C++ 调用符号方法时会被调用
def _run_symbolic_method(g, op_name, symbolic_fn, args):
    # 尝试创建 GraphContext 对象，用于运行符号方法
    try:
        graph_context = jit_utils.GraphContext(
            graph=g,
            block=g.block(),
            opset=GLOBALS.export_onnx_opset_version,
            original_node=None,  # 原始节点为 None
            params_dict=_params_dict,
            env={},
            values_in_env=set(),
            new_nodes=[],
        )
        # 调用符号方法并返回结果
        return symbolic_fn(graph_context, *args)
    # 捕获 TypeError 异常，处理无法调度到符号方法的情况
    except TypeError as e:
        # 修改异常信息，指明具体的操作名称
        e.args = (f"{e.args[0]} (occurred when translating {op_name})",)
        raise


# 应用 Beartype 装饰器来检查函数参数类型
@_beartype.beartype
# 向节点添加一个新的块
def _add_block(node: _C.Node) -> _C.Block:
    # 返回节点添加的新块对象
    return node.addBlock()


# 应用 Beartype 装饰器来检查函数参数类型
@_beartype.beartype
# 向块添加输入
def _add_input_to_block(block: _C.Block):
    # 返回向块添加输入的操作结果（类型说明忽略）
    return block.addInputToBlock()
# 注册输出到代码块中，使用给定值
def _add_output_to_block(block: _C.Block, value: _C.Value) -> int:
    return block.registerOutput(value)


@_beartype.beartype
# 检查是否应该回退到 ATen 操作
def _should_aten_fallback(
    name: str, opset_version: int, operator_export_type: _C_onnx.OperatorExportTypes
):
    # 对所有构建类型而言，如果 domain=="aten" 并且 operator_export_type==ONNX_ATEN，
    # 将创建一个 aten::ATen 运算符，而不考虑符号存在与否

    # 检查是否注册了可导出的 ATen 操作
    is_exportable_aten_op = registration.registry.is_registered_op(name, opset_version)
    # 检查是否是 ONNX_ATEN 导出类型
    is_onnx_aten_export = operator_export_type == _C_onnx.OperatorExportTypes.ONNX_ATEN
    # 检查是否是 ONNX_ATEN_FALLBACK 导出类型
    is_aten_fallback_export = (
        operator_export_type == _C_onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
    )

    if not name.startswith("aten::"):
        return False

    # 如果是 ONNX_ATEN 导出或者是 ATen 回退导出且不是可导出的 ATen 操作，则返回 True
    if is_onnx_aten_export or (is_aten_fallback_export and not is_exportable_aten_op):
        return True

    return False


@_beartype.beartype
# 检查是否需要符号化上下文
def _need_symbolic_context(symbolic_fn: Callable) -> bool:
    """检查 symbolic_fn 的第一个参数是否标注为 `torch.onnx.SymbolicContext` 类型。"""
    # 获取 symbolic_fn 的所有参数
    params = tuple(inspect.signature(symbolic_fn).parameters.values())
    # 当注释是延迟评估时，注释是字符串而不是类型。我们需要使用 get_type_hints 来获取真实的类型。
    if not params:
        return False
    first_param_name = params[0].name
    # 获取 symbolic_fn 的类型提示
    type_hints = typing.get_type_hints(symbolic_fn)
    # 如果第一个参数名不在类型提示中，则返回 False
    if first_param_name not in type_hints:
        return False
    # 获取第一个参数的类型
    param_type = type_hints[first_param_name]
    # 判断 param_type 是否是 _exporter_states.SymbolicContext 的子类
    return issubclass(param_type, _exporter_states.SymbolicContext)


@_beartype.beartype
# 符号化上下文处理器装饰器
def _symbolic_context_handler(symbolic_fn: Callable) -> Callable:
    """为符号化函数提供符号化上下文的装饰器。"""
    if _need_symbolic_context(symbolic_fn):
        # TODO(justinchuby): 当 GraphContext 成为公共对象时，更新 GraphContext 的模块名称
        warnings.warn(
            "The first argument to symbolic functions is deprecated in 1.13 and will be "
            "removed in the future. Please annotate treat the first argument (g) as GraphContext "
            "and use context information from the object instead.",
            category=FutureWarning,
        )

        def wrapper(graph_context: jit_utils.GraphContext, *args, **kwargs):
            # 创建符号化上下文对象
            symbolic_context = _exporter_states.SymbolicContext(
                params_dict=graph_context.params_dict,
                env=graph_context.env,
                cur_node=graph_context.original_node,
                onnx_block=graph_context.block,
            )
            # 调用符号化函数，传入符号化上下文和图上下文参数
            return symbolic_fn(symbolic_context, graph_context, *args, **kwargs)

        return wrapper
    return symbolic_fn


@_beartype.beartype
# 获取 ATen 操作重载名称
def _get_aten_op_overload_name(n: _C.Node) -> str:
    # 返回非 Caffe2 构建上的 ATen 操作的 `overload_name` 属性
    schema = n.schema()
    if not schema.startswith("aten::"):
        return ""
    return _C.parse_schema(schema).overload_name
# 使用装饰器进行类型检查和验证
@_beartype.beartype
# 定义运行符号函数的方法，接受以下参数：
# - graph: 图对象，表示计算图
# - block: 块对象，表示计算块
# - node: 节点对象，表示计算节点
# - inputs: 任意类型，表示节点的输入
# - env: 字典，将值映射到值的环境
# - values_in_env: 集合，表示环境中值的集合
# - new_nodes: 列表，表示新节点的列表
# - operator_export_type: _C_onnx.OperatorExportTypes.ONNX，表示运算符导出类型，默认为ONNX
# 返回值为可选类型，可以是单个值或值的序列
def _run_symbolic_function(
    graph: _C.Graph,
    block: _C.Block,
    node: _C.Node,
    inputs: Any,
    env: Dict[_C.Value, _C.Value],
    values_in_env: Set[_C.Value],
    new_nodes: List[_C.Node],
    operator_export_type=_C_onnx.OperatorExportTypes.ONNX,
) -> Optional[Union[_C.Value, Sequence[Optional[_C.Value]]]]:
    """Runs a symbolic function.

    The function is used in C++ to export the node to ONNX.

    Returns:
        A single or a tuple of Values.
        None when the node gets cloned as is into the new graph.
    """

    # 获取全局变量中的ONNX操作集版本号
    opset_version = GLOBALS.export_onnx_opset_version

    # 查看是否有“Export inplace”注释
    node_kind = node.kind()
    if node_kind.endswith("_"):
        # 如果节点操作名以“_”结尾，则移除最后一个字符
        ns_op_name = node_kind[:-1]
    else:
        ns_op_name = node_kind

    # 解析节点操作名，获取命名空间和操作名
    namespace, op_name = jit_utils.parse_node_kind(ns_op_name)

    # 创建图的上下文，包括图、块、操作集版本、原始节点等信息
    graph_context = jit_utils.GraphContext(
        graph=graph,
        block=block,
        opset=opset_version,
        original_node=node,
        params_dict=_params_dict,
        env=env,
        values_in_env=values_in_env,
        new_nodes=new_nodes,
    )

    # 如果请求直接进行ATen导出
    if _should_aten_fallback(ns_op_name, opset_version, operator_export_type):
        # 收集节点的属性到attrs字典中，包括属性名和属性值
        attrs = {
            k + "_" + node.kindOf(k)[0]: symbolic_helper._node_get(node, k)
            for k in node.attributeNames()
        }
        # 获取节点的输出大小
        outputs = node.outputsSize()
        attrs["outputs"] = outputs
        # 调用ATen操作函数，并传递节点的输入和收集的属性
        return graph_context.aten_op(
            op_name,
            *inputs,
            overload_name=_get_aten_op_overload_name(node),
            **attrs,
        )

    try:
        # 设置域名为命名空间
        domain = namespace
        # 构建符号函数名称，包括域名和操作名
        symbolic_function_name = f"{domain}::{op_name}"

        # 获取注册表中的符号函数组
        symbolic_function_group = registration.registry.get_function_group(
            symbolic_function_name
        )
        if symbolic_function_group is not None:
            # 获取符号函数组中指定操作集版本的符号函数
            symbolic_fn = symbolic_function_group.get(opset_version)
            if symbolic_fn is not None:
                # 收集节点的属性到attrs字典中
                attrs = {
                    k: symbolic_helper._node_get(node, k) for k in node.attributeNames()
                }
                # 调用符号函数，并传递图的上下文、节点输入和收集的属性
                return symbolic_fn(graph_context, *inputs, **attrs)

        # 如果未找到符合条件的符号函数，继续收集节点的属性到attrs字典中
        attrs = {
            k + "_" + node.kindOf(k)[0]: symbolic_helper._node_get(node, k)
            for k in node.attributeNames()
        }
        if namespace == "onnx":
            # 克隆节点以触发ONNX形状推断，并传递节点的输入和收集的属性
            return graph_context.op(op_name, *inputs, **attrs, outputs=node.outputsSize())  # type: ignore[attr-defined]

        # 如果未能匹配到符合条件的操作，抛出不支持的操作符异常
        raise errors.UnsupportedOperatorError(
            symbolic_function_name,
            opset_version,
            symbolic_function_group.get_min_supported()
            if symbolic_function_group
            else None,
        )
    # 捕获 RuntimeError 异常
    except RuntimeError:
        # 如果 operator_export_type 等于 ONNX_FALLTHROUGH，则返回 None
        if operator_export_type == _C_onnx.OperatorExportTypes.ONNX_FALLTHROUGH:
            return None
        # 如果 operator_export_type 等于 ONNX_ATEN_FALLBACK
        elif operator_export_type == _C_onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK:
            # 创建包含属性的字典，属性名称为 k + "_" + node.kindOf(k)[0]，属性值为 symbolic_helper._node_get(node, k)
            attrs = {
                k + "_" + node.kindOf(k)[0]: symbolic_helper._node_get(node, k)
                for k in node.attributeNames()
            }
            # 返回一个 ATen 操作，使用 graph_context.aten_op 方法调用
            return graph_context.aten_op(
                op_name,
                *inputs,
                overload_name=_get_aten_op_overload_name(node),
                **attrs,
            )
        # 抛出其它异常
        raise
    # 捕获 TypeError 异常
    except TypeError as e:
        # 处理特定情况，当无法成功分发时
        e.args = (f"{e.args[0]} \n(Occurred when translating {op_name}).",)
        # 抛出异常
        raise
# 使用 @_beartype 装饰器对函数进行类型检查
@_beartype.beartype
# 验证自定义操作符名称的格式是否正确
def _verify_custom_op_name(symbolic_name: str):
    # 如果符号名称不匹配指定的正则表达式，则抛出错误
    if not re.match(r"^[a-zA-Z0-9-_]+::[a-zA-Z-_]+[a-zA-Z0-9-_]*$", symbolic_name):
        raise errors.OnnxExporterError(
            f"Failed to register operator {symbolic_name}. "
            "The symbolic name must match the format domain::name, "
            "and should start with a letter and contain only "
            "alphanumerical characters"
        )

    # 解析符号名称的命名空间
    ns, _ = jit_utils.parse_node_kind(symbolic_name)
    # 如果命名空间为 "onnx"，则抛出值错误
    if ns == "onnx":
        raise ValueError(
            f"Failed to register operator {symbolic_name}. {ns} domain cannot be modified."
        )


# 使用 @_beartype 装饰器对函数进行类型检查
@_beartype.beartype
# 注册自定义操作符的符号函数
def register_custom_op_symbolic(
    symbolic_name: str,
    symbolic_fn: Callable,
    opset_version: int,
):
    """Registers a symbolic function for a custom operator.

    When the user registers symbolic for custom/contrib ops,
    it is highly recommended to add shape inference for that operator via setType API,
    otherwise the exported graph may have incorrect shape inference in some extreme cases.
    An example of setType is `test_aten_embedding_2` in `test_operators.py`.

    See "Custom Operators" in the module documentation for an example usage.

    Args:
        symbolic_name (str): The name of the custom operator in "<domain>::<op>"
            format.
        symbolic_fn (Callable): A function that takes in the ONNX graph and
            the input arguments to the current operator, and returns new
            operator nodes to add to the graph.
        opset_version (int): The ONNX opset version in which to register.
    """
    # 如果符号名称以 "::" 开头，则添加 "aten" 前缀
    if symbolic_name.startswith("::"):
        symbolic_name = f"aten{symbolic_name}"

    # 验证符号名称的格式是否正确
    _verify_custom_op_name(symbolic_name)

    # 调用 registration.custom_onnx_symbolic 注册符号名称、操作集版本及装饰器列表
    registration.custom_onnx_symbolic(
        symbolic_name,
        opset_version,
        decorate=[
            _symbolic_context_handler,
        ],
    )(symbolic_fn)


# 使用 @_beartype 装饰器对函数进行类型检查
@_beartype.beartype
# 取消注册自定义操作符的符号函数
def unregister_custom_op_symbolic(symbolic_name: str, opset_version: int):
    """Unregisters ``symbolic_name``.

    See "Custom Operators" in the module documentation for an example usage.

    Args:
        symbolic_name (str): The name of the custom operator in "<domain>::<op>"
            format.
        opset_version (int): The ONNX opset version in which to unregister.
    """
    # 如果符号名称以 "::" 开头，则添加 "aten" 前缀
    if symbolic_name.startswith("::"):
        symbolic_name = f"aten{symbolic_name}"

    # 验证符号名称的格式是否正确
    _verify_custom_op_name(symbolic_name)

    # 调用 registration.registry.unregister 取消注册符号名称及操作集版本
    registration.registry.unregister(symbolic_name, opset_version)


# 使用 @_beartype 装饰器对函数进行类型检查
@_beartype.beartype
# 验证动态轴参数是否符合预期格式
def _validate_dynamic_axes(dynamic_axes, model, input_names, output_names):
    """Ensures dynamic axes argument is follows the expected format."""
    # 如果动态轴的长度为零，则直接返回
    if len(dynamic_axes) == 0:
        return
    # 检查模型是否具有属性 "graph"
    if hasattr(model, "graph"):
        # 如果未提供输入名称或列表为空，则使用模型图中输入节点的调试名称作为输入名称列表
        if (input_names is None) or len(input_names) == 0:
            input_names = [x.debugName() for x in model.graph.inputs()]
        # 如果未提供输出名称或列表为空，则使用模型图中输出节点的调试名称作为输出名称列表
        if (output_names is None) or len(output_names) == 0:
            output_names = [y.debugName() for y in model.graph.outputs()]

    # 将输入名称和输出名称的集合合并为有效名称的集合
    valid_names = set((input_names or []) + (output_names or []))

    # 如果动态轴以列表形式提供，应将其转换为期望的字典格式。
    # 如果未为动态轴提供所需的轴名称，则将为每个指定的输入/输出的动态轴自动生成名称。
    for key, value in dynamic_axes.items():
        # 如果键不在有效名称集合中，则发出警告
        if key not in valid_names:
            warnings.warn(
                f"Provided key {key} for dynamic axes is not a valid input/output name"
            )
        # 如果值是列表，则发出警告并自动生成轴名称
        if isinstance(value, list):
            warnings.warn(
                "No names were found for specified dynamic axes of provided input. "
                f"Automatically generated names will be applied to each dynamic axis of input {key}"
            )

            # 自动生成轴名称的字典
            value_dict = {}
            for i, x in enumerate(value):
                # 检查轴索引类型是否为整数
                if not isinstance(x, int):
                    raise ValueError(
                        "The type of axis index is expected to be an integer"
                    )
                # 如果轴索引已经存在于字典中，则发出警告
                if x in value_dict:
                    warnings.warn(
                        f"Duplicate dynamic axis index {x} was provided for input {key}."
                    )
                else:
                    # 为动态轴生成名称并存储在字典中
                    value_dict[x] = str(key) + "_dynamic_axes_" + str(i + 1)
            # 更新动态轴字典为自动生成的名称
            dynamic_axes[key] = value_dict
# 定义函数 model_signature，用于获取给定模型的签名信息
def model_signature(model: Union[torch.nn.Module, Callable]) -> inspect.Signature:
    # 使用 inspect.signature 函数获取模型的签名信息
    return inspect.signature(
        model.forward if isinstance(model, torch.nn.Module) else model
    )
```