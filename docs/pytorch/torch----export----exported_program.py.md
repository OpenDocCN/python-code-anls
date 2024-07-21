# `.\pytorch\torch\export\exported_program.py`

```py
# mypy: allow-untyped-defs
# 导入上下文管理模块，用于处理上下文相关的操作
import contextlib
# 复制对象模块，用于创建对象的深拷贝
import copy
# 数据类模块，用于声明数据类
import dataclasses
# 函数工具模块，提供了用于函数操作的装饰器等工具
import functools
# 正则表达式模块，用于处理正则表达式操作
import re
# 类型模块，用于处理类型相关的操作
import types
# 警告模块，用于发出警告信息
import warnings
# 命名元组模块，用于创建命名元组
from collections import namedtuple
# 上下文管理装饰器，用于创建上下文管理器
from contextlib import contextmanager
# 类型提示模块，用于提供类型提示信息
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TYPE_CHECKING,
    Union,
)

# 导入 autograd_not_implemented 函数
from torch._higher_order_ops.utils import autograd_not_implemented

# 导入 FakeScriptObject 类
from torch._library.fake_class_registry import FakeScriptObject
# 导入 _PyTreeCodeGen 和 _PyTreeInfo 类
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo

# 导入不可变字典和列表函数
from torch.fx.immutable_collections import immutable_dict, immutable_list

# 如果在类型检查时，导入 sympy 模块和 ValueRanges 类
if TYPE_CHECKING:
    import sympy
    from torch.utils._sympy.value_ranges import ValueRanges

# 导入 torch 库
import torch
# 导入 pytree 模块
import torch.utils._pytree as pytree
# 导入 FunctionalTensor 类
from torch._subclasses.functional_tensor import FunctionalTensor

# 导入用于导出的工具函数和类
from torch.export._tree_utils import is_equivalent, reorder_kwargs
# 导入兼容性函数
from torch.fx._compatibility import compatibility

# 导入用于 fx 的实用函数和模块
from torch.fx._utils import first_call_function_nn_module_stack
# 导入用于实验性代理张量的模块
from torch.fx.experimental.proxy_tensor import maybe_disable_fake_tensor_mode

# 导入用于 fx 传递的基础模块和 PassResult 类
from torch.fx.passes.infra.pass_base import PassResult
# 导入用于 fx 传递管理器的 PassManager 类
from torch.fx.passes.infra.pass_manager import PassManager
# 导入插入延迟运行断言的函数
from torch.fx.passes.runtime_assert import insert_deferred_runtime_asserts

# 从 graph_signature 模块导入以下类和函数，忽略 F401 错误
from .graph_signature import (
    _sig_to_specs,
    ArgumentSpec,
    ConstantArgument,
    CustomObjArgument,
    ExportGraphSignature,
    InputKind,
    InputSpec,
    OutputKind,
    OutputSpec,
    SymIntArgument,
    TensorArgument,
    TokenArgument,
)

# 导出的模块列表
__all__ = [
    "ExportedProgram",
    "ModuleCallEntry",
    "ModuleCallSignature",
]

# 定义 PassType 类型别名，用于表示 fx 传递函数
PassType = Callable[[torch.fx.GraphModule], Optional[PassResult]]


@dataclasses.dataclass
# 定义 ModuleCallSignature 数据类，包含输入输出参数和树形规范
class ModuleCallSignature:
    inputs: List[ArgumentSpec]
    outputs: List[ArgumentSpec]
    in_spec: pytree.TreeSpec
    out_spec: pytree.TreeSpec


@dataclasses.dataclass
# 定义 ModuleCallEntry 数据类，包含完全限定名和调用签名
class ModuleCallEntry:
    fqn: str
    signature: Optional[ModuleCallSignature] = None


# 装饰函数，用于禁用现有的伪张量模式
def _disable_prexisiting_fake_mode(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # 使用上下文管理器禁用伪张量模式
        with maybe_disable_fake_tensor_mode():
            return fn(*args, **kwargs)

    return wrapper


# 函数用于比较 fx 集合的等价性
def _fx_collection_equivalence_fn(
    spec1_type: Optional[type],
    spec1_context: pytree.Context,
    spec2_type: Optional[type],
    spec2_context: pytree.Context,
) -> bool:
    """Treat containers and their immutable variants as the same type. Otherwise
    compare as normal.
    """
    # 如果 spec1_type 或 spec2_type 为 None，则比较它们是否相等，并比较上下文是否相等
    if spec1_type is None or spec2_type is None:
        return spec1_type is spec2_type and spec1_context == spec2_context

    # 如果 spec1_type 是 dict 或 immutable_dict，并且 spec2_type 也是，则视为相同类型
    if issubclass(spec1_type, (dict, immutable_dict)) and issubclass(
        spec2_type, (dict, immutable_dict)
    ):
        return True
    # 检查特定类型为列表或不可变列表，并比较它们的内容是否相等
    ):
        # 如果两个对象的内容相等，则返回 True
        return spec1_context == spec2_context

    # 如果 spec1_type 和 spec2_type 都是 list 或 immutable_list 的子类
    if issubclass(spec1_type, (list, immutable_list)) and issubclass(
        spec2_type, (list, immutable_list)
    ):
        # 比较 spec1_context 和 spec2_context 的内容是否相等
        return spec1_context == spec2_context

    # 如果 spec1_type 和 spec2_type 类型相同，并且它们的内容相等
    return spec1_type is spec2_type and spec1_context == spec2_context
# 用于注册 CIA 到 meta 的函数，接受任意参数和关键字参数
def _register_cia_to_meta(*args, **kwargs):
    # 从关键字参数中获取 kernel
    kernel = kwargs["kernel"]
    # 删除 kwargs 中的 kernel 键
    del kwargs["kernel"]

    # 断言确保 torch._C._dispatch_has_kernel_for_dispatch_key 函数可以使用 CompositeImplicitAutograd 调度键找到 kernel
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        kernel.name(), torch._C.DispatchKey.CompositeImplicitAutograd
    )

    # 调用 kernel 的 _op_dk 方法，使用 CompositeImplicitAutograd 调度键，并传入剩余的参数和关键字参数
    return kernel._op_dk(
        torch._C.DispatchKey.CompositeImplicitAutograd, *args, **kwargs
    )


# 编译自 DispatchKey.cpp 的列表，用于在导出时覆盖 CIA decomp
# 思路是使用这些键来覆盖 CIA decomp
_AUTOGRAD_ALIAS_BACKEND_KEYS_TO_OVERRIDE = [
    torch._C.DispatchKey.AutogradCPU,
    torch._C.DispatchKey.AutogradCUDA,
    torch._C.DispatchKey.AutogradMeta,
    torch._C.DispatchKey.AutogradXLA,
    torch._C.DispatchKey.AutogradLazy,
    torch._C.DispatchKey.AutogradIPU,
    torch._C.DispatchKey.AutogradXPU,
    torch._C.DispatchKey.AutogradMPS,
    torch._C.DispatchKey.AutogradHPU,
    torch._C.DispatchKey.AutogradPrivateUse1,
    torch._C.DispatchKey.AutogradPrivateUse2,
    torch._C.DispatchKey.AutogradPrivateUse3,
]


# 上下文管理器，用于覆盖 CompositeImplicitAutograd 的解析
@contextmanager
def _override_composite_implicit_decomp(ops_to_preserve):
    # 此函数用于为用户指定的功能组合操作覆盖 CompositeImplicitAutograd 的解析
    # 我们希望不对所有组合操作进行解析，但目前的 C++ 功能化依赖于解析后的操作集
    # 所以我们只能对功能操作进行此操作。但有一个注意点是，有些组合操作会伪造其模式（声称是功能性的但实际不是，比如 dropout），对于这些情况，我们只能解析。
    saved_tables = {}  # 保存原始表的字典
    patched_ops = set()  # 已修改的操作集合
    try:
        yield
    finally:
        # 还原已修改操作的状态
        for op in patched_ops:
            op.py_kernels.clear()
            op.py_kernels.update(saved_tables[op])
            op._dispatch_cache.clear()


# 避免名称冲突而重命名的函数
def _rename_without_collisions(
    name_map: Dict[str, str],
    orig_name: str,
    name: str,
    is_placeholder: bool = False,
):
    """
    根据需要重命名节点以避免名称冲突，可以添加后缀。
    name_map: 原始名称到新名称的映射
    orig_name: 要映射的原始名称
    name: 候选名称（可能带后缀，例如 mul_2）
    is_placeholder: 如果节点是占位符，则避免检测后缀
    """
    if name in name_map.values():
        # 如果名称已经在映射中，尝试增加序数后缀而不是添加新后缀
        match = re.match(r"(.*)_(\d+)", name)
        if match and not is_placeholder:
            name, n = match.group(1), int(match.group(2))
        else:
            n = 0
        while (dup_name := f"{name}_{n + 1}") in name_map.values():
            n += 1
        name_map[orig_name] = dup_name
    else:
        name_map[orig_name] = name
    return name_map[orig_name]


# 将顶级图中的占位符名称传播到 HigherOrderOp 子图中的函数
def _name_hoo_subgraph_placeholders(gm: torch.fx.GraphModule) -> None:
    """
    将顶级图中的占位符名称传播到 HigherOrderOp 子图中。
    """
    # 函数的具体实现在这里省略，因为文档字符串中并未提供更多的详细信息。
    """
    # 收集所有 HOO 子图及其顶层命名占位节点
    subgraph_ph_tuples: List[Tuple[torch.fx.GraphModule, List[torch.fx.Node]]] = []
    
    # 遍历主模块图中的每个节点
    for node in gm.graph.nodes:
        # 检查是否为调用函数操作且目标是 HigherOrderOperator
        if node.op == "call_function" and isinstance(
            node.target, torch._ops.HigherOrderOperator
        ):
            # 根据不同的 HigherOrderOperator 类型，处理不同的输入模式
            if node.target._name == "cond":
                _, true_graph, false_graph, cond_args = node._args
                # 将真假分支的子图及其参数添加到元组列表中
                subgraph_ph_tuples.append((getattr(gm, true_graph.target), cond_args))
                subgraph_ph_tuples.append((getattr(gm, false_graph.target), cond_args))
            elif node.target._name == "wrap_with_set_grad_enabled":
                subgraph, phs = node._args[1], node._args[2:]
                # 将包装子图及其参数添加到元组列表中
                subgraph_ph_tuples.append((getattr(gm, subgraph.target), phs))
            elif node.target._name == "map_impl":
                body_graph, array, args = node._args
                # 将映射实现子图及其参数添加到元组列表中
                subgraph_ph_tuples.append(
                    (getattr(gm, body_graph.target), array + args)
                )
    
    # 传播名称
    for subgraph, hoo_phs in subgraph_ph_tuples:
        # 创建名称映射字典
        name_map: Dict[str, str] = {}
        # 遍历子图中的每个节点
        for i, node in enumerate(subgraph.graph.nodes):
            if i < len(hoo_phs):  # 如果是占位符节点，保留名称
                name_map[node.name] = hoo_phs[i].name
                node.name = node.target = hoo_phs[i].name
            else:  # 如果不是占位符节点，检查是否存在名称冲突
                node.name = _rename_without_collisions(name_map, node.name, node.name)
    
        # 递归处理并重新编译子图
        _name_hoo_subgraph_placeholders(subgraph)
        subgraph.recompile()
    """
def _decompose_and_get_gm_with_new_signature_constants(
    ep,
    *,
    decomp_table: Dict[torch._ops.OperatorBase, Callable],
    _preserve_ops: Tuple[torch._ops.OpOverload],
    joint_loss_index: Optional[int],
):
    # 引入必要的模块和函数
    from torch._export.non_strict_utils import (
        _gather_constant_attrs,
        make_fake_params_buffers,
    )
    from torch._functorch.aot_autograd import aot_export_module
    from torch._guards import detect_fake_mode

    from torch.export._trace import (
        _export_to_aten_ir,
        _get_params_buffers,
        _ignore_backend_decomps,
        _verify_nn_module_stack,
        _verify_placeholder_names,
        _verify_stack_trace,
    )

    # 如果验证器的方言是 "TRAINING"
    if ep.verifier.dialect == "TRAINING":
        # 获取模型对象
        mod = ep.module()
        fake_args = []
        # 遍历模型图中的节点，获取所有占位符节点的值
        for node in mod.graph.nodes:
            if node.op == "placeholder":
                fake_args.append(node.meta["val"])

        # 将获取的占位符值进行解包成对应的数据结构
        fake_args_unwrapped = pytree.tree_unflatten(fake_args, mod._in_spec)
        # 检测模型中的伪模式
        fake_mode = detect_fake_mode(fake_args)

        # 如果输出规范不是列表或元组，则转换为元组
        out_spec = mod._out_spec
        if out_spec.type not in (list, tuple):
            out_spec = pytree.TreeSpec(tuple, None, [out_spec])

        # 更新代码生成器的信息
        mod.graph._codegen = _PyTreeCodeGen(
            _PyTreeInfo(
                mod.graph._codegen.pytree_info.orig_args,  # type: ignore[attr-defined]
                mod._in_spec,
                out_spec,
            )
        )

        # 重新编译模型
        mod.recompile()

        # 生成伪参数和缓冲区，并收集常量属性
        fake_params_buffers = make_fake_params_buffers(
            fake_mode, _get_params_buffers(mod)
        )
        constant_attrs = _gather_constant_attrs(mod)

        # 导出至 ATen IR
        aten_export_artifact = _export_to_aten_ir(
            mod,
            fake_args_unwrapped[0],
            fake_args_unwrapped[1],
            fake_params_buffers,
            constant_attrs,
        )

        # 获取导出的图模型和新的图签名
        gm = aten_export_artifact.gm
        new_graph_signature = aten_export_artifact.sig

        # 更新节点的 nn_module_stack 属性
        for node in gm.graph.nodes:
            if node.op not in ["placeholder", "output"]:
                for key, (fqn, mod_cls) in node.meta["nn_module_stack"].items():
                    if isinstance(mod_cls, type):
                        node.meta["nn_module_stack"][key] = (
                            fqn,
                            mod_cls.__module__ + "." + mod_cls.__qualname__,
                        )

        # 验证 nn_module_stack、堆栈跟踪和占位符名称
        _verify_nn_module_stack(gm)
        _verify_stack_trace(gm)
        _verify_placeholder_names(gm, new_graph_signature)

        # 返回图模型和新的图签名
        return gm, new_graph_signature

    # 获取旧占位符节点
    old_placeholders = [
        node for node in ep.graph_module.graph.nodes if node.op == "placeholder"
    ]
    # 获取旧占位符节点的值
    fake_args = [node.meta["val"] for node in old_placeholders]

    # 获取要移除的缓冲区名称列表
    buffers_to_remove = [name for name, _ in ep.graph_module.named_buffers()]
    # 遍历要删除的缓冲区名称列表，从图模块中删除对应属性
    for name in buffers_to_remove:
        delattr(ep.graph_module, name)

    # 从 torch._guards 模块中导入 detect_fake_mode 函数
    from torch._guards import detect_fake_mode

    # TODO(zhxhchen17) 直接返回新的 graph_signature

    # 检测是否为假模式
    fake_mode = detect_fake_mode(fake_args)
    # 如果 fake_mode 为 None，则创建一个空的上下文管理器
    fake_mode = contextlib.nullcontext() if fake_mode is None else fake_mode
    # 忽略后端分解，使用 fake_mode 上下文管理器，覆盖 _preserve_ops，调用 aot_export_module 导出模块
    with _ignore_backend_decomps(), fake_mode, _override_composite_implicit_decomp(
        _preserve_ops
    ):
        gm, graph_signature = aot_export_module(
            ep.graph_module,
            fake_args,
            decompositions=decomp_table,
            trace_joint=True if joint_loss_index is not None else False,
            output_loss_index=joint_loss_index
            if joint_loss_index is not None
            else None,
        )

    # 更新签名，以便在调用 aot_export 时更改占位符名称
    def update_arg(old_arg, new_ph):
        if isinstance(old_arg, ConstantArgument):
            return old_arg
        elif isinstance(old_arg, TensorArgument):
            return TensorArgument(name=new_ph.name)
        elif isinstance(old_arg, SymIntArgument):
            return SymIntArgument(name=new_ph.name)
        raise RuntimeError(f"Type of old_arg not supported: {type(old_arg)}")

    # 获取新的占位符列表和新的输出
    new_placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    new_outputs = list(gm.graph.nodes)[-1].args[0]

    # 重命名占位符
    assert len(new_placeholders) == len(old_placeholders)
    for old_ph, new_ph in zip(old_placeholders, new_placeholders):
        new_ph.name = new_ph.target = old_ph.name

    # 处理新分解图节点的名称冲突
    name_map = {ph.name: ph.name for ph in new_placeholders}
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            continue
        node.name = _rename_without_collisions(name_map, node.name, node.name)

    # 传播名称到高阶操作子图的占位符
    _name_hoo_subgraph_placeholders(gm)

    # 为了匹配输入突变的正确输入输出目标，需要找到旧到新占位符的映射
    old_new_placeholder_map = {
        spec.arg.name: new_placeholders[i].name
        for i, spec in enumerate(ep.graph_signature.input_specs)
        if not isinstance(spec.arg, ConstantArgument)
    }

    # 更新输入规范
    input_specs = [
        InputSpec(
            spec.kind,
            update_arg(spec.arg, new_placeholders[i]),
            spec.target,
            spec.persistent,
        )
        for i, spec in enumerate(ep.graph_signature.input_specs)
    ]
    # 更新输出规范
    output_specs = [
        OutputSpec(
            spec.kind,
            update_arg(spec.arg, new_outputs[i]),
            old_new_placeholder_map.get(spec.target, spec.target),
        )
        for i, spec in enumerate(ep.graph_signature.output_specs)
    ]
    # 如果存在联合损失索引，则进行以下操作
    if joint_loss_index is not None:
        # 断言图签名的反向签名不为空
        assert graph_signature.backward_signature is not None
        # 获取梯度映射到用户输入的字典
        gradients = graph_signature.backward_signature.gradients_to_user_inputs
        # 断言用户输入的数量与图签名的输入规范数量相等
        assert len(graph_signature.user_inputs) == len(ep.graph_signature.input_specs)
        
        # 创建输入规范的字典，键为用户输入，值为规范，仅包含张量参数的规范
        specs = {
            graph_signature.user_inputs[i]: spec
            for i, spec in enumerate(ep.graph_signature.input_specs)
            if isinstance(spec.arg, TensorArgument)
        }
        
        # 遍历从新输出中得到的节点，并处理它们的输出规范
        for i, node in enumerate(new_outputs[len(output_specs):]):
            # 获取节点对应的梯度源
            source = gradients[node.name]
            # 根据源获取规范
            spec = specs[source]  # type: ignore[index]
            
            # 根据规范的类型确定输出的种类和目标
            if spec.kind == InputKind.PARAMETER:
                kind = OutputKind.GRADIENT_TO_PARAMETER
                target = spec.target
            elif spec.kind == InputKind.USER_INPUT:
                kind = OutputKind.GRADIENT_TO_USER_INPUT
                target = source
            else:
                raise AssertionError(f"Unknown input kind: {spec.kind}")
            
            # 将新的输出规范添加到输出规范列表中
            output_specs.append(
                OutputSpec(
                    kind,
                    TensorArgument(name=node.name),
                    target,
                )
            )

    # 断言新旧占位符列表的长度相同
    assert len(new_placeholders) == len(old_placeholders)

    # 创建新的图签名对象，包含输入规范和输出规范
    new_graph_signature = ExportGraphSignature(
        input_specs=input_specs, output_specs=output_specs
    )
    
    # 注意：aot_export为具有整数值的占位符添加符号整数元数据；由于这些被专门化了，我们将这些元数据替换为原始值。
    # 同时，将参数/缓冲区元数据设置回占位符。
    # 遍历新旧占位符列表，将未包含torch.Tensor的元数据值复制回新占位符的元数据
    for old_node, new_node in zip(old_placeholders, new_placeholders):
        if not isinstance(old_node.meta["val"], torch.Tensor):
            new_node.meta["val"] = old_node.meta["val"]

        # 如果新占位符的目标是新图签名的输入参数或缓冲区之一，则复制旧占位符的所有元数据到新占位符
        if (
            new_node.target in new_graph_signature.inputs_to_parameters
            or new_node.target in new_graph_signature.inputs_to_buffers
        ):
            for k, v in old_node.meta.items():
                new_node.meta[k] = v
    
    # 返回图管理器和新的图签名对象
    return gm, new_graph_signature
# 将输入的导出程序（ep）分解，并获取包含常量签名的新图（gm）及其新图签名
def _decompose_exported_program(
    ep,
    *,
    decomp_table: Dict[torch._ops.OperatorBase, Callable],
    _preserve_ops: Tuple[torch._ops.OpOverload],
    joint_loss_index: Optional[int],
):
    # 导入 Dynamo 配置模块
    from torch._dynamo import config as _dynamo_config
    # 导入节点元数据钩子相关模块
    from torch._export.passes._node_metadata_hook import (
        _node_metadata_hook,
        _set_node_metadata_hook,
    )
    # 导入常量提升模块相关内容
    from torch._export.passes.lift_constants_pass import (
        ConstantAttrMap,
        lift_constants_pass,
    )

    # 调用 _decompose_and_get_gm_with_new_signature_constants 函数分解并获取带有新签名常量的图（gm）及其新图签名
    gm, new_graph_signature = _decompose_and_get_gm_with_new_signature_constants(
        ep,
        decomp_table=decomp_table,
        _preserve_ops=_preserve_ops,
        joint_loss_index=joint_loss_index,
    )

    # TODO: 不幸的是，与 aot_export 一起保留图级元数据效果不佳。因此我们手动复制它。
    # （节点级元数据在上面已处理。）
    # 更新 gm 的元数据以包含 ep.graph_module 的元数据
    gm.meta.update(ep.graph_module.meta)

    # 获取更新后的范围约束
    new_range_constraints = _get_updated_range_constraints(
        gm,
        ep.range_constraints,
        _is_executorch=False,
    )

    # 将常量提升到 gm 中，并将结果添加到 ep.constants 中
    constants = lift_constants_pass(gm, new_graph_signature, ConstantAttrMap())
    for k, v in constants.items():
        assert k not in ep.constants
        ep.constants[k] = v

    # 如果未禁用运行时断言，设置堆栈跟踪和形状环境，并在导出程序中插入延迟运行时断言
    if not _dynamo_config.do_not_emit_runtime_asserts:
        stack_trace = (
            'File "torch/fx/passes/runtime_assert.py", line 24, '
            "in insert_deferred_runtime_asserts"
        )
        shape_env = _get_shape_env(gm)
        if shape_env is not None:
            with _set_node_metadata_hook(
                gm, functools.partial(_node_metadata_hook, stack_trace=stack_trace)
            ):
                insert_deferred_runtime_asserts(
                    gm,
                    shape_env,
                    f"exported program: {first_call_function_nn_module_stack(gm.graph)}",
                    export=True,
                )

    # 创建 ExportedProgram 对象，包含根图（gm）、图对象（gm.graph）、图签名（new_graph_signature）、状态字典（ep.state_dict）、范围约束（new_range_constraints）、模块调用图（ep.module_call_graph）、示例输入（ep.example_inputs）和常量字典（ep.constants）
    exported_program = ExportedProgram(
        root=gm,
        graph=gm.graph,
        graph_signature=new_graph_signature,
        state_dict=ep.state_dict,
        range_constraints=new_range_constraints,
        module_call_graph=copy.deepcopy(ep.module_call_graph),
        example_inputs=ep.example_inputs,
        constants=ep.constants,
    )
    # 返回导出程序对象
    return exported_program


class ExportedProgram:
    """
    包含通过 :func:`export` 导出的程序的封装。它包含了一个表示张量计算的 :class:`torch.fx.Graph`，
    一个包含所有提升参数和缓冲区张量值的 state_dict，以及各种元数据。

    您可以像调用由 :func:`export` 追踪的原始可调用对象一样调用 ExportedProgram，具有相同的调用约定。

    要在图上执行转换，请使用 ``.module`` 属性访问 :class:`torch.fx.GraphModule`。
    您可以使用 `FX 转换 <https://pytorch.org/docs/stable/fx.html#writing-transformations>`_
    来重写图。之后，您可以简单地使用 :func:`export`
    """
    again to construct a correct ExportedProgram.
    """

    # 定义构造函数，初始化导出程序的各项属性
    def __init__(
        self,
        root: Union[torch.nn.Module, Dict[str, Any]],
        graph: torch.fx.Graph,
        graph_signature: ExportGraphSignature,
        state_dict: Dict[str, Union[torch.Tensor, torch.nn.Parameter]],
        range_constraints: "Dict[sympy.Symbol, Any]",
        module_call_graph: List[ModuleCallEntry],
        example_inputs: Optional[Tuple[Tuple[Any, ...], Dict[str, Any]]] = None,
        verifier: Optional[Type[Any]] = None,  # TODO Change typing hint to Verifier.
        tensor_constants: Optional[
            Dict[str, torch.Tensor]
        ] = None,  # TODO: deprecate this
        constants: Optional[
            Dict[str, Union[torch.Tensor, FakeScriptObject, torch._C.ScriptObject]]
        ] = None,
    ):
        # 从图中移除与代码生成相关的部分，确保图是扁平化的
        graph._codegen = torch.fx.graph.CodeGen()
        # 根据根模块和图创建图模块用于导出
        self._graph_module = _create_graph_module_for_export(root, graph)
        # 如果根是 torch.fx.GraphModule 类型，更新图模块的 meta 信息
        if isinstance(root, torch.fx.GraphModule):
            self._graph_module.meta.update(root.meta)

        # 设置导出图的签名
        self._graph_signature: ExportGraphSignature = graph_signature
        # 设置导出程序的状态字典
        self._state_dict: Dict[str, Any] = state_dict
        # 设置符号和取值范围的约束
        self._range_constraints: Dict[sympy.Symbol, ValueRanges] = range_constraints
        # 断言模块调用图不为空
        assert module_call_graph is not None
        # 设置模块调用图
        self._module_call_graph: List[ModuleCallEntry] = module_call_graph
        # 设置示例输入（可选）
        self._example_inputs = example_inputs

        # 设置常量字典，优先选择 tensor_constants，然后 constants，都为空则默认为空字典
        self._constants = tensor_constants or constants or {}
        assert self._constants is not None

        # 导入 Verifier 类并设置验证器，若 verifier 为空则使用默认 Verifier 类
        from torch._export.verifier import Verifier

        if verifier is None:
            verifier = Verifier
        # 断言 verifier 是 Verifier 类的子类
        assert issubclass(verifier, Verifier)
        # 设置验证器
        self._verifier = verifier
        # 调用验证器的检查方法，确保导出程序的正确性
        self.verifier().check(self)

    @property
    @compatibility(is_backward_compatible=False)
    def graph_module(self):
        # 返回导出程序的图模块
        return self._graph_module

    @property
    @compatibility(is_backward_compatible=False)
    def graph(self):
        # 返回导出程序的图
        return self.graph_module.graph

    @property
    @compatibility(is_backward_compatible=False)
    def graph_signature(self):
        # 返回导出程序的图签名
        return self._graph_signature

    @property
    @compatibility(is_backward_compatible=False)
    def state_dict(self):
        # 返回导出程序的状态字典
        return self._state_dict

    @compatibility(is_backward_compatible=False)
    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """
        Returns an iterator over original module's parameters.
        """
        # 返回原始模块参数的迭代器
        for _, param in self.named_parameters():
            yield param

    @compatibility(is_backward_compatible=False)
    # 返回一个迭代器，遍历模块的命名参数，每个元素是参数名和对应的 torch.nn.Parameter 对象
    def named_parameters(self) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        for param_name in self.graph_signature.parameters:
            # 返回参数名及其对应的状态字典中的值（通常是 torch.nn.Parameter 对象）
            yield param_name, self.state_dict[param_name]

    # 返回一个迭代器，遍历模块的缓冲区
    @compatibility(is_backward_compatible=False)
    def buffers(self) -> Iterator[torch.Tensor]:
        for _, buf in self.named_buffers():
            # 返回缓冲区的值，通常是 torch.Tensor 对象
            yield buf

    # 返回一个迭代器，遍历模块的命名缓冲区，每个元素是缓冲区名和对应的 torch.Tensor 对象
    @compatibility(is_backward_compatible=False)
    def named_buffers(self) -> Iterator[Tuple[str, torch.Tensor]]:
        non_persistent_buffers = set(self.graph_signature.non_persistent_buffers)
        for buffer_name in self.graph_signature.buffers:
            if buffer_name in non_persistent_buffers:
                # 如果是非持久缓冲区，返回缓冲区名及其对应的常量
                yield buffer_name, self.constants[buffer_name]
            else:
                # 否则返回缓冲区名及其对应的状态字典中的值（通常是 torch.Tensor 对象）
                yield buffer_name, self.state_dict[buffer_name]

    # 返回 _range_constraints 属性，通常是模块的范围约束
    @property
    @compatibility(is_backward_compatible=False)
    def range_constraints(self):
        return self._range_constraints

    # 返回 _module_call_graph 属性，通常是模块的调用图
    @property
    @compatibility(is_backward_compatible=False)
    def module_call_graph(self):
        return self._module_call_graph

    # 返回 _example_inputs 属性，通常是模块的示例输入
    @property
    @compatibility(is_backward_compatible=False)
    def example_inputs(self):
        return self._example_inputs

    # 返回 _call_spec 属性，通常是模块的调用规范
    @property
    @compatibility(is_backward_compatible=False)
    def call_spec(self):
        CallSpec = namedtuple("CallSpec", ["in_spec", "out_spec"])

        # 如果模块调用图为空，返回空的调用规范
        if len(self.module_call_graph) == 0:
            return CallSpec(in_spec=None, out_spec=None)
        # 断言第一个调用的全限定名称为空字符串
        assert self.module_call_graph[0].fqn == ""
        # 返回第一个调用的输入规范和输出规范作为调用规范
        return CallSpec(
            in_spec=self.module_call_graph[0].signature.in_spec,
            out_spec=self.module_call_graph[0].signature.out_spec,
        )

    # 返回 _verifier 属性，通常是模块的验证器
    @property
    @compatibility(is_backward_compatible=False)
    def verifier(self) -> Any:
        return self._verifier

    # 返回 _verifier.dialect 属性，通常是验证器的方言
    @property
    @compatibility(is_backward_compatible=False)
    def dialect(self) -> str:
        assert self._verifier is not None
        return self._verifier.dialect

    # 返回 _constants 属性，通常是模块的张量常量
    @property
    @compatibility(is_backward_compatible=False)
    def tensor_constants(self):
        return self._constants

    # 返回 _constants 属性，通常是模块的常量
    @property
    @compatibility(is_backward_compatible=False)
    def constants(self):
        return self._constants
    def _get_flat_args_with_check(self, args, kwargs):
        """Flatten args, kwargs using pytree, then, check specs.

        Args:
            args: List[Any] original args passed to __call__
            kwargs: Dict[str, Any] original kwargs passed to __call

        Returns:
            A tuple of (flat_args, received_spec)
            flat_args is flattend args / kwargs
            received_spec is the pytree spec produced while flattening the
            tuple (args, kwargs)
        """
        # 获取调用规范中的输入规范
        in_spec = self.call_spec.in_spec
        # 如果存在输入规范，则重新排序kwargs
        if in_spec is not None:
            kwargs = reorder_kwargs(kwargs, in_spec)
        # 使用pytree库将args和kwargs展平，同时返回展平后的路径和接收到的规范
        flat_args_with_path, received_spec = pytree.tree_flatten_with_path(
            (args, kwargs)
        )  # type: ignore[possibly-undefined]
        # 检查输入约束条件
        self._check_input_constraints(flat_args_with_path)
        # 从展平后的路径中提取出flat_args
        flat_args = tuple(x[1] for x in flat_args_with_path)
        # 返回展平后的args和接收到的规范
        return flat_args, received_spec

    def _graph_module_flat_inputs(self, args: Any, kwargs: Any) -> Any:
        """Transform args, kwargs of __call__ to args for graph_module.

        self.graph_module takes stuff from state dict as inputs.
        The invariant is for ep: ExportedProgram is
        ep(args, kwargs) ==
          ep.postprocess(ep.graph_module(ep.graph_module_flat_inputs(args, kwargs)))
        """

        # 获取调用规范中的输入规范
        in_spec = self.call_spec.in_spec
        # 使用_get_flat_args_with_check方法展平args和kwargs
        flat_args, received_spec = self._get_flat_args_with_check(args, kwargs)
        # 如果存在输入规范且接收到的规范与之不等效，则抛出值错误
        if in_spec is not None and not is_equivalent(
            received_spec, in_spec, _fx_collection_equivalence_fn
        ):
            raise ValueError(
                "Trying to flatten user inputs with exported input tree spec: \n"
                f"{in_spec}\n"
                "but actually got inputs with tree spec of: \n"
                f"{received_spec}"
            )

        additional_inputs = []
        # 遍历图模块的输入规范
        for input_ in self.graph_signature.input_specs:
            # 如果输入类型是用户输入，则跳过
            if input_.kind == InputKind.USER_INPUT:
                continue
            # 如果输入类型是参数或者缓冲区
            elif input_.kind in (
                InputKind.PARAMETER,
                InputKind.BUFFER,
            ):
                # 如果缓冲区不是持久的，则从常量中获取；否则从状态字典中获取
                if input_.persistent is False:
                    additional_inputs.append(self.constants[input_.target])
                else:
                    additional_inputs.append(self.state_dict[input_.target])
            # 如果输入类型是常量张量或者自定义对象
            elif input_.kind in (
                InputKind.CONSTANT_TENSOR,
                InputKind.CUSTOM_OBJ,
            ):
                additional_inputs.append(self.constants[input_.target])
        # 将additional_inputs转换为元组
        additional_inputs = tuple(additional_inputs)

        # 注意：调用约定是首先参数，然后是缓冲区，最后是用户提供的args
        # 参考：torch/_functorch/aot_autograd.py#L1034
        return additional_inputs + flat_args
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # 当尝试直接调用 ExportedProgram 实例时，引发 RuntimeError
        raise RuntimeError(
            "Unable to call ExportedProgram directly. "
            "You should use `exported_program.module()` instead."
        )

    def _postprocess_graph_module_outputs(self, res, orig_args, orig_kwargs):
        """Process potential mutations to the input.

        Because self.graph_module is functional, so mutations has to be written
        back after execution of graph_module.
        """
        # 导入 torch._export.error 模块
        import torch._export.error as error

        # 获取扁平化的参数列表及其检查
        flat_args, _ = self._get_flat_args_with_check(orig_args, orig_kwargs)

        # 如果存在输出规范，处理输出中的可能变异
        if self.call_spec.out_spec is not None:
            # 获取需要变异的缓冲区和用户输入
            buffer_mutation = self.graph_signature.buffers_to_mutate
            user_input_mutation = self.graph_signature.user_inputs_to_mutate
            num_mutated = len(buffer_mutation) + len(user_input_mutation)
            mutated_values = res[:num_mutated]

            # 从最终结果中排除依赖令牌
            assertion_dep_token = self.graph_signature.assertion_dep_token
            if assertion_dep_token is not None:
                assertion_dep_token_index = next(iter(assertion_dep_token.keys()))
                res = res[:assertion_dep_token_index]

            # 剩余部分是实际的输出结果
            res = res[num_mutated:]

            try:
                # 尝试使用输出规范来展开输出结果
                res = pytree.tree_unflatten(res, self.call_spec.out_spec)
            except Exception:
                # 如果展开失败，抛出内部错误
                _, received_spec = pytree.tree_flatten(res)
                raise error.InternalError(  # noqa: B904
                    "Trying to flatten user outputs with exported output tree spec: \n"
                    f"{self.call_spec.out_spec}\n"
                    "but actually got outputs with tree spec of: \n"
                    f"{received_spec}"
                )
            finally:
                # 获取所有用户输入的输入规范
                user_inputs = [
                    spec
                    for spec in self.graph_signature.input_specs
                    if spec.kind == InputKind.USER_INPUT
                ]
                # 根据输出规范处理变异的值
                for i, value in enumerate(mutated_values):
                    output_spec = self.graph_signature.output_specs[i]
                    if output_spec.kind == OutputKind.BUFFER_MUTATION:
                        # 如果是缓冲区变异，确保目标不为空，更新状态字典中的目标
                        assert output_spec.target is not None
                        self.state_dict[output_spec.target] = value
                    elif output_spec.kind == OutputKind.USER_INPUT_MUTATION:
                        # 如果是用户输入变异，确保目标不为空，找到目标索引并复制值
                        assert output_spec.target is not None
                        index = next(
                            i
                            for i, spec in enumerate(user_inputs)
                            if spec.arg.name == output_spec.target
                        )
                        flat_args[index].copy_(value)
                    else:
                        # 否则，抛出断言错误，表明出现了意外的输出类型
                        raise AssertionError(f"Unexpected kind: {output_spec.kind}")

        # 返回处理后的结果
        return res
    def __str__(self) -> str:
        # 获取可读的图形模块表示，以字符串形式返回，每行缩进四个空格
        graph_module = self.graph_module.print_readable(
            print_output=False, colored=True
        ).replace("\n", "\n    ")
        # 构建对象的字符串表示，包括图形模块、图形签名和范围约束信息
        string = (
            "ExportedProgram:\n"
            f"    {graph_module}\n"
            f"Graph signature: {self.graph_signature}\n"
            f"Range constraints: {self.range_constraints}\n"
        )
        # 返回对象的字符串表示
        return string

    def module(self) -> torch.nn.Module:
        """
        Returns a self contained GraphModule with all the parameters/buffers inlined.
        """
        from ._unlift import _unlift_exported_program_lifted_states
        
        # 使用 _unlift_exported_program_lifted_states 函数获取包含所有参数/缓冲区的自包含 GraphModule
        module = _unlift_exported_program_lifted_states(self)
        
        # 定义 _train 和 _eval 方法用于抛出未实现错误
        def _train(self, mode: bool = True):
            raise NotImplementedError("Calling train() is not supported yet.")
        
        def _eval(self, mode: bool = True):
            raise NotImplementedError("Calling eval() is not supported yet.")
        
        # 将 _train 和 _eval 方法绑定到 module 实例上
        module.train = types.MethodType(_train, module)  # type: ignore[method-assign]
        module.eval = types.MethodType(_eval, module)  # type: ignore[method-assign]
        
        # 返回处理后的 module 实例
        return module

    def _num_lifted_params_buffers(self):
        # 返回输入规范中用户输入的数量或默认的输入规范长度
        return next(
            (
                i
                for i, s in enumerate(self._graph_signature.input_specs)
                if s.kind == InputKind.USER_INPUT
            ),
            len(self._graph_signature.input_specs),
        )

    @_disable_prexisiting_fake_mode
    def run_decompositions(
        self,
        decomp_table: Optional[Dict[torch._ops.OperatorBase, Callable]] = None,
        _preserve_ops: Tuple[torch._ops.OpOverload] = (),  # type: ignore[assignment]
    ) -> "ExportedProgram":
        """
        Run a set of decompositions on the exported program and returns a new
        exported program. By default we will run the Core ATen decompositions to
        get operators in the
        `Core ATen Operator Set <https://pytorch.org/docs/stable/torch.compiler_ir.html>`_.

        For now, we do not decompose joint graphs.
        """
        from torch._decomp import core_aten_decompositions
        
        # 如果未提供 decomp_table，则使用默认的 Core ATen decompositions
        if decomp_table is None:
            decomp_table = core_aten_decompositions()
        
        # 调用 _decompose_exported_program 函数执行分解操作，并返回新的 ExportedProgram 对象
        return _decompose_exported_program(
            self,
            decomp_table=decomp_table,
            _preserve_ops=_preserve_ops,  # type: ignore[arg-type]
            joint_loss_index=None,
        )

    def _check_input_constraints(self, flat_args_with_path):
        from torch._export.utils import _check_input_constraints_for_graph
        
        # 获取所有节点中操作为 "placeholder" 的占位符节点
        placeholders = [p for p in self.graph.nodes if p.op == "placeholder"]
        
        # 获取所有用户输入的占位符节点
        input_placeholders = [
            p
            for p, s in zip(placeholders, self.graph_signature.input_specs)
            if s.kind == InputKind.USER_INPUT
        ]
        
        # 调用 _check_input_constraints_for_graph 函数检查图形的输入约束
        _check_input_constraints_for_graph(
            input_placeholders, flat_args_with_path, self.range_constraints
        )
    # 调用对象的验证方法，对当前对象进行验证
    def _validate(self):
        self.verifier().check(self)

    # TODO(zhxchen17) Formalize this.
    # 更新对象的方法，返回一个新的 ExportedProgram 实例
    def _update(
        self, graph_module, graph_signature, state_dict=None
    ) -> "ExportedProgram":
        return ExportedProgram(
            root=graph_module,  # 设置根模块
            graph=graph_module.graph,  # 设置图结构
            graph_signature=graph_signature,  # 设置图签名
            state_dict=state_dict or self.state_dict,  # 设置状态字典，如果未提供则使用当前对象的状态字典
            range_constraints=copy.deepcopy(self.range_constraints),  # 深拷贝范围约束
            module_call_graph=copy.deepcopy(self._module_call_graph),  # 深拷贝模块调用图
            example_inputs=self.example_inputs,  # 设置示例输入
            verifier=self.verifier,  # 设置验证器
            tensor_constants=self.tensor_constants,  # 设置张量常量
        )
# 创建一个函数 _get_shape_env，接收一个 GraphModule 对象 gm 作为参数
def _get_shape_env(gm):
    # 从 gm 的图中收集所有节点的 meta 属性中的 "val" 值，存入列表 vals
    vals = [
        node.meta["val"]
        for node in gm.graph.nodes
        if node.meta.get("val", None) is not None
    ]
    # 导入 detect_fake_mode 函数用于检测虚假模式（fake mode）
    from torch._guards import detect_fake_mode

    # 调用 detect_fake_mode 函数，传入 vals 列表，返回 fake_mode
    fake_mode = detect_fake_mode(vals)
    # 如果 fake_mode 不为 None，则返回其 shape_env 属性
    if fake_mode is not None:
        return fake_mode.shape_env
    # 如果没有找到 fake_mode，则遍历 vals 列表
    for v in vals:
        # 如果 v 是 torch.SymInt 类型的对象，则返回其 node 的 shape_env 属性
        if isinstance(v, torch.SymInt):
            return v.node.shape_env


# 创建一个函数 _get_updated_range_constraints，接收三个参数：
# gm：torch.fx.GraphModule 对象，
# old_range_constraints：可选的字典类型参数，默认为 None，
# _is_executorch：布尔类型参数，默认为 True
# 返回类型为 Dict[sympy.Symbol, Any]
def _get_updated_range_constraints(
    gm: torch.fx.GraphModule,
    old_range_constraints: "Optional[Dict[sympy.Symbol, Any]]" = None,
    _is_executorch: bool = True,
) -> "Dict[sympy.Symbol, Any]":
    # 对于 _is_executorch 为 True 的情况，应该执行以下分支
    # FIXME(tmanlaibaatar) Remove this whole branch once https://github.com/pytorch/pytorch/pull/123764
    if _is_executorch:
        # 断言 old_range_constraints 必须为 None
        assert old_range_constraints is None
        # 调用 _get_shape_env 函数获取 shape_env
        shape_env = _get_shape_env(gm)
        # 如果 shape_env 为 None，则返回空字典
        if shape_env is None:
            return {}
        # 从 shape_env 中筛选出 var_to_range 字典的一部分作为 range_constraints
        range_constraints = {
            k: v
            for k, v in shape_env.var_to_range.items()
            if k not in shape_env.replacements
        }
        # 再次遍历 shape_env 的 var_to_range，更新 range_constraints
        # 只有在 unbacked symint 存在且用作构造函数输入时，runtime_var_to_range 与 var_to_range 才有差异
        # 例如 [2, oo) -> [0, oo)
        for k, v in shape_env.var_to_range.items():
            if k not in shape_env.replacements:
                range_constraints[k] = v
        return range_constraints

    # 如果 _is_executorch 不为 True，则执行以下分支
    # 断言 old_range_constraints 不为 None
    assert old_range_constraints is not None
    # 调用 _get_shape_env 函数获取 shape_env
    shape_env = _get_shape_env(gm)
    # 如果 shape_env 为 None，则返回空字典
    if shape_env is None:
        return {}

    # 复制 old_range_constraints 到 range_constraints
    range_constraints = copy.copy(old_range_constraints)
    # 从 range_constraints 中筛选出不在 shape_env.replacements 中的键值对，更新 range_constraints
    range_constraints = {
        k: v for k, v in range_constraints.items() if k not in shape_env.replacements
    }
    # 再次遍历 shape_env 的 var_to_range，更新 range_constraints
    for k, v in shape_env.var_to_range.items():
        if k not in shape_env.replacements and k not in range_constraints:
            range_constraints[k] = v
    return range_constraints


# 创建一个函数 _create_graph_module_for_export，接收两个参数：
# root：根节点，
# graph：torch.fx.Graph 对象
def _create_graph_module_for_export(root, graph):
    try:
        # 尝试创建一个 torch.fx.GraphModule 对象 gm，使用 root 和 graph 作为参数
        gm = torch.fx.GraphModule(root, graph)
    except SyntaxError:
        # 如果抛出 SyntaxError 异常，则警告用户
        warnings.warn(
            "Unable to execute the generated python source code from "
            "the graph. The graph module will no longer be directly callable, "
            "but you can still run the ExportedProgram, and if needed, you can "
            "run the graph module eagerly using torch.fx.Interpreter."
        )
        # 创建一个带有警告的空的 torch.fx.GraphModule 对象 gm
        gm = torch.fx.GraphModule(root, torch.fx.Graph())
        # 将输入的 graph 赋值给 gm 的 _graph 属性
        gm._graph = graph

    return gm
```