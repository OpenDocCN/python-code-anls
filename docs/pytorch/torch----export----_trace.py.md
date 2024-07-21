# `.\pytorch\torch\export\_trace.py`

```py
# mypy: allow-untyped-defs
# 导入必要的模块和库
import dataclasses  # 用于数据类的装饰器
import functools  # 用于高阶函数的工具函数
import inspect  # 用于检查对象的内省工具
import logging  # 日志记录模块
import re  # 正则表达式操作
import time  # 时间操作
import warnings  # 警告控制
from contextlib import contextmanager, nullcontext  # 上下文管理器
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union  # 类型提示

import torch  # PyTorch 深度学习库
import torch._dynamo  # Torch 内部动态图模块
import torch.fx  # Torch FX 图操作库

import torch.utils._pytree as pytree  # Torch 工具函数模块，支持 Python 树结构操作
from torch._dispatch.python import enable_python_dispatcher  # Torch 分发器的 Python 接口
from torch._dynamo.exc import UserError, UserErrorType  # Torch 动态图异常类
from torch._export.non_strict_utils import (  # Torch 导出相关的非严格工具函数
    _fakify_script_objects,
    _gather_constant_attrs,
    make_constraints,
    make_fake_inputs,
    make_fake_params_buffers,
    produce_guards_and_solve_constraints,
)
from torch._export.passes._node_metadata_hook import (  # Torch 导出相关的节点元数据钩子
    _node_metadata_hook,
    _set_node_metadata_hook,
)
from torch._export.passes.add_runtime_assertions_for_constraints_pass import (  # Torch 导出相关的运行时断言插入
    _AddRuntimeAssertionsForInlineConstraintsPass,
)
from torch._export.passes.collect_tracepoints_pass import CollectTracepointsPass  # Torch 导出相关的收集追踪点的通行证
from torch._export.passes.lift_constants_pass import (  # Torch 导出相关的常量提升通行证
    ConstantAttrMap,
    lift_constants_pass,
    rewrite_script_object_meta,
)
from torch._export.utils import (  # Torch 导出相关的实用工具函数
    placeholder_naming_pass,
    placeholder_prefixes,
)
from torch._export.verifier import SpecViolationError  # Torch 导出相关的规范违规错误
from torch._export.wrappers import _wrap_submodules  # Torch 模块子模块包装器
from torch._functorch._aot_autograd.traced_function_transforms import (  # Torch AOT 自动求导追踪函数转换工具
    create_functional_call,
)

from torch._functorch._aot_autograd.utils import create_tree_flattened_fn  # Torch AOT 自动求导工具函数
from torch._functorch.aot_autograd import aot_export_module  # Torch AOT 自动求导导出模块
from torch._guards import detect_fake_mode  # Torch 伪模式检测器

from torch._library.fake_class_registry import FakeScriptObject  # Torch 假脚本对象注册器
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode  # Torch 假张量类
from torch._utils_internal import log_export_usage  # Torch 内部工具日志导出使用情况
from torch.export.dynamic_shapes import _combine_args  # Torch 导出动态形状组合参数
from torch.export.exported_program import OutputKind  # Torch 导出程序输出类型
from torch.fx._utils import first_call_function_nn_module_stack  # Torch FX 工具函数
from torch.fx.experimental.proxy_tensor import make_fx  # Torch FX 实验性代理张量
from torch.fx.experimental.symbolic_shapes import (  # Torch FX 实验性符号形状
    ConstraintViolationError,
    free_unbacked_symbols,
    GuardOnDataDependentSymNode,
    ShapeEnv,
)
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo  # Torch FX 图相关操作
from torch.fx.passes.runtime_assert import insert_deferred_runtime_asserts  # Torch FX 运行时断言插入
from torch.utils._pytree import TreeSpec  # Torch 树结构规范
from torch.utils._sympy.value_ranges import ValueRangeError  # Torch SymPy 值范围错误

from ._safeguard import AutogradStateOpsFailSafeguard  # 导出保护状态操作失败的安全措施

from .exported_program import (  # 导出程序相关
    _disable_prexisiting_fake_mode,
    ExportedProgram,
    InputKind,
    ModuleCallEntry,
    ModuleCallSignature,
)
from .graph_signature import (  # 图签名相关
    _sig_to_specs,
    ArgumentSpec,
    ConstantArgument,
    CustomObjArgument,
    ExportGraphSignature,
    SymIntArgument,
    TensorArgument,
    TokenArgument,
)

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器

@dataclasses.dataclass
class ExportDynamoConfig:
    """
    Manage Export-specific configurations of Dynamo.
    """
    allow_rnn: bool = True  # 是否允许导出 RNN，默认为 True
    # 定义一个类字段 reorderable_logging_functions，其类型为集合(Set)，集合元素为可调用对象（Callable）
    reorderable_logging_functions: Set[Callable] = dataclasses.field(
        # 使用 dataclasses.field 定义字段，默认工厂为 set，用于存储可重排的日志函数
        default_factory=set
    )
@dataclasses.dataclass
class ATenExportArtifact:
    gm: torch.fx.GraphModule  # 用于存储经过 TorchFX 转换后的图模块对象
    sig: ExportGraphSignature  # 表示导出图签名的对象
    constants: Dict[  # 用于存储常量的字典，键为常量名称，值为常量的不同类型之一
        str,
        Union[
            torch.Tensor,
            FakeScriptObject,
            torch.ScriptObject,
        ],
    ]


@dataclasses.dataclass(frozen=True)
class ExportArtifact:
    aten: ATenExportArtifact  # ATen 模块导出的对象
    out_spec: TreeSpec  # 导出模块的输出规范
    fake_mode: FakeTensorMode  # 表示是否为虚拟张量模式的枚举
    module_call_specs: Dict[str, Dict[str, pytree.TreeSpec]]  # 模块调用规范的字典


DEFAULT_EXPORT_DYNAMO_CONFIG = ExportDynamoConfig()  # 创建默认的导出配置对象

# 设置可重新排序的日志记录函数集合，包括标准的 logging 模块函数和 print、warnings 中的函数
DEFAULT_EXPORT_DYNAMO_CONFIG.reorderable_logging_functions = {
    logging.critical,
    logging.debug,
    logging.error,
    logging.exception,
    logging.info,
    logging.log,
    logging.warning,
    print,
    warnings.warn,
}


@contextmanager
def _ignore_backend_decomps():
    # 临时禁用 MKLDNN 和 NNPACK 的后端标志，并保存原始状态
    orig_mkldnn_flag = torch.backends.mkldnn.set_flags(False)
    orig_nnpack_flag = torch.backends.nnpack.set_flags(False)
    try:
        yield
    finally:
        # 恢复 MKLDNN 和 NNPACK 的后端标志到原始状态
        torch.backends.mkldnn.set_flags(*orig_mkldnn_flag)
        torch.backends.nnpack.set_flags(*orig_nnpack_flag)


def _fixup_key(x):
    # 修正键值，将输入的 x 处理为 "L__self__" 开头并去掉可能的 "_export_root" 前缀
    return "L__self__" + _strip_root(x)


def _strip_root(x):
    # 如果 x 是字符串并且以 "_export_root" 开头，则去掉该前缀并返回
    if isinstance(x, str) and x.startswith("_export_root"):
        stripped = x[len("_export_root") :]
        return stripped[1:] if stripped.startswith(".") else stripped
    return x  # 否则直接返回 x


def _add_runtime_assertions_to_cond_in_subgraph(range_constraints, gm, fake_mode):
    # 添加运行时断言到子图中的条件节点
    # 目前仍然需要此步骤，因为某些情况下 insert_deferred_runtime_assertions 未能添加断言到条件子图中
    if len(range_constraints) > 0:
        # 定义堆栈跟踪信息
        stack_trace = (
            'File "torch/_export/passes/add_runtime_assertions_for_constraints_pass.py", line 46, '
            "in _AddRuntimeAssertionsForInlineConstraintsPass"
        )
        # 使用 fake_mode 上下文和 _set_node_metadata_hook 来设置节点元数据的钩子
        with fake_mode, _set_node_metadata_hook(
            gm, functools.partial(_node_metadata_hook, stack_trace=stack_trace)
        ):
            # 应用运行时断言到内联约束
            res = _AddRuntimeAssertionsForInlineConstraintsPass(range_constraints)(gm)
        assert res is not None  # 确保返回结果不为空
        gm = res.graph_module  # 更新图模块对象


def _rewrite_node(gm):
    # 重写图中的节点
    for node in gm.graph.nodes:
        if node.target == torch.ops.higher_order._export_tracepoint:
            if "path" in node.kwargs:
                path = _strip_root(node.kwargs["path"])  # 处理节点参数中的路径
                with gm.graph.inserting_before(node):
                    # 创建新的调用节点来替换原始节点
                    new_node = gm.graph.create_node(
                        "call_function",
                        torch.ops.higher_order._export_tracepoint,
                        args=node.args,
                        kwargs={
                            "path": path,
                            "kind": node.kwargs["kind"],
                        },
                    )
                    new_node.meta = node.meta  # 复制原始节点的元数据
                    node.replace_all_uses_with(new_node)  # 替换所有使用该节点的地方
                    gm.graph.erase_node(node)  # 删除原始节点


def _convert_input_to_fake(gm, args, kwargs):
    params_buffers = _get_params_buffers(gm)  # 获取图模块的参数和缓冲区信息
    # 创建一个空列表用于存储假输入张量
    fake_inps: List[torch.Tensor] = []
    # 遍历计算图中的每个节点
    for node in gm.graph.nodes:
        # 检查节点是否为占位符，并且具有元数据中的 'val' 属性
        if node.op == "placeholder" and "val" in node.meta:
            # 获取节点的假值
            fake_val = node.meta["val"]
            # 如果假值不为 None，并且是 torch.Tensor 类型，则将其添加到 fake_inps 中
            if fake_val is not None and isinstance(fake_val, torch.Tensor):
                fake_inps.append(fake_val)

    # 检测是否存在假模式，如果有则使用检测到的假模式，否则创建一个新的假张量模式
    if detected_fake_mode := detect_fake_mode(fake_inps):
        fake_mode = detected_fake_mode
    else:
        fake_mode = FakeTensorMode(shape_env=ShapeEnv(), export=True)

    # 如果没有输入参数和关键字参数，则返回空元组、空字典、params_buffers 和 fake_mode
    if len(args) == 0 and len(kwargs) == 0:
        return (), {}, params_buffers, fake_mode

    # 初始化一个计数器 count 为 0
    count = 0

    # 定义一个函数 convert_to_fake，用于将输入转换为对应的假张量
    def convert_to_fake(x):
        nonlocal count
        # 获取 fake_inps 中的第 count 个假输入张量
        val = fake_inps[count]
        # 更新 count，指向下一个假输入张量
        count += 1
        return val

    # 使用 tree_map_only 函数将 args 中的每个 torch.Tensor 转换为对应的假张量
    fake_args = pytree.tree_map_only(torch.Tensor, convert_to_fake, args)

    # 使用 tree_map_only 函数将 kwargs 中的每个 torch.Tensor 转换为假张量，使用 fake_mode.from_tensor 方法
    fake_kwargs = pytree.tree_map_only(torch.Tensor, fake_mode.from_tensor, kwargs)

    # 使用 tree_map_only 函数将 params_buffers 中的每个 torch.Tensor 转换为假张量，
    # 使用 functools.partial 为 fake_mode.from_tensor 方法设置 static_shapes=True 参数
    fake_params_buffers = pytree.tree_map_only(
        torch.Tensor,
        functools.partial(fake_mode.from_tensor, static_shapes=True),
        params_buffers,
    )

    # 返回转换后的假输入 args、假关键字参数 kwargs、假参数缓冲区 params_buffers 和假模式 fake_mode
    return fake_args, fake_kwargs, fake_params_buffers, fake_mode
# 替换参数和缓冲区名字的映射关系到输入和输出规格中
def _replace_param_buffer_names(param_buffer_table, sig):
    # 遍历输入规格中的每一个规格对象
    for spec in sig.input_specs:
        # 如果规格的类型是参数或者缓冲区
        if spec.kind in (
            InputKind.PARAMETER,
            InputKind.BUFFER,
        ):
            # 使用 param_buffer_table 中对应的映射替换 spec.target
            spec.target = param_buffer_table[spec.target]
    
    # 遍历输出规格中的每一个规格对象
    for spec in sig.output_specs:
        # 如果规格的类型是缓冲区变化或者梯度到参数的映射
        if spec.kind in (
            OutputKind.BUFFER_MUTATION,
            OutputKind.GRADIENT_TO_PARAMETER,
        ):
            # 使用 param_buffer_table 中对应的映射替换 spec.target
            spec.target = param_buffer_table[spec.target]


# 将原始参数名、位置参数和关键字参数转换为位置参数
def _convert_to_positional_args(orig_arg_names, args, kwargs):
    # 断言参数名的总数应该等于位置参数数和关键字参数数的和
    assert len(orig_arg_names) == len(args) + len(kwargs), (
        f"Total number of arg names is expected to be {len(orig_arg_names)} "
        f"but got {len(args)} positional args, {len(kwargs)} kwargs."
    )
    # 重新排序关键字参数，使其按照原始参数名的顺序排列
    reordered_kwargs = [kwargs[kw_name] for kw_name in orig_arg_names[len(args):]]
    # 返回所有位置参数和重新排序的关键字参数
    return (
        *args,
        *reordered_kwargs,
    )


# 标准化 nn_module_stack 中的路径，添加根模块到每一个 nn_module_stack
def _normalize_nn_module_stack(gm_torch_level, root_cls):
    # 根模块的路径字符串表示
    root = "L['self']"
    # 根据路径字符串生成一个合法的键名
    root_key = re.sub(r"[^a-zA-Z0-9]", "_", root)
    
    # 遍历每一个 gm_torch_level 中的 GraphModule
    for gm in gm_torch_level.modules():
        # 如果不是 torch.fx.GraphModule 类型，则跳过
        if not isinstance(gm, torch.fx.GraphModule):
            continue
        
        # 遍历每一个节点
        for node in gm.graph.nodes:
            # 如果节点的操作是占位符或输出，跳过
            if node.op in ["placeholder", "output"]:
                continue
            
            # 默认添加根模块
            add_root = True
            
            # 获取节点的 meta 中的 nn_module_stack
            if nn_module_stack := node.meta.get("nn_module_stack", {}):
                # 获取第一个 nn_module_stack 的路径和类型
                path, ty = next(iter(nn_module_stack.values()))
                # 如果类 ty 是 torch.nn.Module 的子类，则可能会存在 root 的情况
                if inspect.isclass(ty) and issubclass(ty, torch.nn.Module):
                    # TODO 理解为什么有时候有根模块有时候没有
                    if path == root and ty is root_cls:
                        add_root = False
                else:
                    assert isinstance(ty, str)
            
            # 如果需要添加根模块
            if add_root:
                # 定义一个函数用于标准化路径
                def normalize_path(path):
                    try:
                        parts = []
                        
                        # 定义一个 Path 类，用于处理路径字符串
                        class Path:
                            def __getattr__(self, name):
                                parts.append(name)
                                return self
                            
                            def __getitem__(self, idx):
                                parts.append(str(idx))
                                return self
                        
                        # 通过 eval 函数将路径字符串转换为路径对象
                        eval(path, {"L": {"self": Path()}})
                        # 返回路径对象的字符串表示形式
                        return ".".join(parts)
                    except Exception:  # TODO(zhxchen17) Remove this.
                        return path
                
                # 更新 nn_module_stack，添加根模块
                nn_module_stack = {
                    root_key: (root, root_cls.__module__ + "." + root_cls.__qualname__),
                    **nn_module_stack,
                }
                # 更新节点的 nn_module_stack，标准化路径
                node.meta["nn_module_stack"] = {
                    key: (normalize_path(path), ty)
                    for key, (path, ty) in nn_module_stack.items()
                }
# 定义函数，获取从原始模型到跟踪模型参数/缓冲的名称映射字典
def _get_param_buffer_mapping(
    original_module: torch.nn.Module,
    traced_module: torch.nn.Module,
) -> Dict[str, str]:
    """
    Returns a mapping of parameter/buffer names from the new module to the
    original model. This is to help with restoring the FQN for parameter/buffers
    of a traced module to what the original module contains.
    """

    # 创建参数查找字典，键为参数 ID，值为参数名列表
    param_lookup: Dict[int, List[str]] = {}
    for name, param in original_module.named_parameters(remove_duplicate=False):
        param_lookup.setdefault(id(param), []).append(name)

    # 创建缓冲区查找字典，键为缓冲区 ID，值为缓冲区名列表
    buffer_lookup: Dict[int, List[str]] = {}
    for name, buffer in original_module.named_buffers(remove_duplicate=False):
        buffer_lookup.setdefault(id(buffer), []).append(name)

    # 将参数名列表反转，以便按照模型结构 FIFO 方式分配 FQN
    for name, fqns in param_lookup.items():
        param_lookup[name] = fqns[::-1]

    # 将缓冲区名列表反转，以便按照模型结构 FIFO 方式分配 FQN
    for name, fqns in buffer_lookup.items():
        buffer_lookup[name] = fqns[::-1]

    # 创建参数-缓冲区映射表，从跟踪模型到原始模型的参数/缓冲区名称
    param_buffer_table: Dict[str, str] = {}
    for dynamo_name, dynamo_param in traced_module.named_parameters(
        remove_duplicate=False
    ):
        # 断言确保动态参数名称不在映射表中
        assert dynamo_name not in param_buffer_table
        # 如果跟踪模型参数的 ID 存在于原始模型参数查找字典中
        if id(dynamo_param) in param_lookup:
            # 将动态参数名称映射到原始模型参数名称列表的末尾，并弹出该名称
            param_buffer_table[dynamo_name] = param_lookup[id(dynamo_param)].pop()

    for dynamo_name, dynamo_buffer in traced_module.named_buffers(
        remove_duplicate=False
    ):
        # 断言确保动态缓冲区名称不在映射表中
        assert dynamo_name not in param_buffer_table
        # 如果跟踪模型缓冲区的 ID 存在于原始模型缓冲区查找字典中
        if id(dynamo_buffer) in buffer_lookup:
            # 将动态缓冲区名称映射到原始模型缓冲区名称列表的末尾，并弹出该名称
            param_buffer_table[dynamo_name] = buffer_lookup[id(dynamo_buffer)].pop()

    # 返回参数-缓冲区映射表
    return param_buffer_table


# 定义函数，保留要求梯度传递的过程
def _preserve_requires_grad_pass(
    gm: torch.fx.GraphModule,
    sig: ExportGraphSignature,
    fake_params_buffers: Dict[str, torch.Tensor],
    constants: Dict[str, Union[torch.Tensor, FakeScriptObject, torch.ScriptObject]],
    flat_fake_args: List[Any],
):
    # 获取图中所有占位符节点，这些节点的操作类型为“placeholder”
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    # 断言确保签名的输入规范数量与占位符节点数量相等
    assert len(sig.input_specs) == len(placeholders)
    i = 0
    # 遍历占位符节点及其对应的输入规范
    for node, spec in zip(placeholders, sig.input_specs):
        # 如果规范类型为参数或者缓冲区
        if spec.kind in (
            InputKind.PARAMETER,
            InputKind.BUFFER,
        ):
            # 断言确保规范的目标不为空
            assert spec.target is not None
            # 设置占位符节点值的 requires_grad 属性为对应 fake_params_buffers 中目标张量的 requires_grad 属性
            node.meta["val"].requires_grad = fake_params_buffers[
                spec.target
            ].requires_grad
        elif spec.kind == InputKind.USER_INPUT:
            # 获取平坦化后的虚假参数/缓冲区列表中的当前元素
            fake_arg = flat_fake_args[i]
            if isinstance(fake_arg, torch.Tensor):
                # 设置占位符节点值的 requires_grad 属性为当前 fake_arg 张量的 requires_grad 属性
                node.meta["val"].requires_grad = fake_arg.requires_grad
            i += 1
        elif spec.kind == InputKind.CONSTANT_TENSOR:
            # 断言确保规范的目标不为空
            assert spec.target is not None
            # 获取常量字典中目标张量
            constant = constants[spec.target]
            if isinstance(constant, torch.Tensor):
                # 设置占位符节点值的 requires_grad 属性为 constant 张量的 requires_grad 属性
                node.meta["val"].requires_grad = constant.requires_grad
        elif spec.kind in (InputKind.CUSTOM_OBJ, InputKind.TOKEN):
            # 对于自定义对象或者令牌，跳过处理
            continue
        else:
            # 对于未知的规范类型，引发断言错误
            raise AssertionError(spec.kind)
# 重新映射常量
def _remap_constants(
    orig_constant_attrs: ConstantAttrMap,
    graph_signature: ExportGraphSignature,
    constants: Dict[str, Union[torch.Tensor, FakeScriptObject, torch.ScriptObject]],
) -> None:
    """Rewrite the graph signature and constants table to use the FQN from the original module."""
    # 创建一个空的重映射表，用于存储原始模块中常量属性的完全限定名称(FQN)
    remap_table: Dict[str, List[str]] = {}

    # 遍历常量字典，检查其值是否在原始常量属性中，若在则将其映射至重映射表
    for name, value in constants.items():
        if value in orig_constant_attrs:
            remap_table[name] = orig_constant_attrs[value]

    # 遍历输入规范中的规范对象，处理常量张量和自定义对象类型的输入规范
    for spec in graph_signature.input_specs:
        if spec.kind in (
            InputKind.CONSTANT_TENSOR,
            InputKind.CUSTOM_OBJ,
        ):
            # 获取原始目标并确保其不为空，将其映射至重映射表中的目标列表的第一个
            orig_target = spec.target
            assert orig_target is not None
            targets = remap_table.get(orig_target, [orig_target])
            spec.target = targets[0]

            # 删除原始目标常量，并将映射表中的每个目标与其对应的常量进行重新设置
            constant = constants[orig_target]
            del constants[orig_target]
            for target in targets:
                constants[target] = constant


# 重命名常量节点
def _rename_constants_nodes(
    gm: torch.fx.GraphModule,
    graph_signature: ExportGraphSignature,
) -> None:
    """
    For strict mode, rename constants nodes that were previously annotated as buffers.
    """
    # 处理与现有常量名称冲突的情况
    node_names = {node.name for node in gm.graph.nodes}

    # 定义重命名常量的函数
    def rename_constant(name):
        if name in node_names:
            n = 1
            # 处理重名情况，通过在名称后添加序号来解决
            while (dup_name := f"{name}_{n}") in node_names:
                n += 1
            name = dup_name
        node_names.add(name)
        return name

    # 使用输入规范将缓冲区名称映射到常量名称
    buffer_prefix = placeholder_prefixes[InputKind.BUFFER]
    const_prefix = placeholder_prefixes[InputKind.CONSTANT_TENSOR]
    buffer_to_constant = {}
    for spec in graph_signature.input_specs:
        if spec.kind == InputKind.CONSTANT_TENSOR and not spec.arg.name.startswith(
            const_prefix
        ):
            if spec.arg.name.startswith(buffer_prefix):  # 从缓冲区映射到常量
                c_name = rename_constant(
                    const_prefix + spec.arg.name[len(buffer_prefix) :]
                )
            else:  # 提升的常量
                c_name = rename_constant(const_prefix + spec.arg.name)
            buffer_to_constant[spec.arg.name] = c_name
            spec.arg.name = c_name
    
    # 根据输出规范，检查常量名称是否在缓冲区到常量映射表中，若在则进行重命名
    for spec in graph_signature.output_specs:
        if spec.arg.name in buffer_to_constant:
            spec.arg.name = buffer_to_constant[spec.arg.name]

    # 为所有模块重命名常量节点
    for mod in gm.modules():
        if not isinstance(mod, torch.fx.GraphModule):
            continue
        for node in mod.graph.nodes:
            if node.name in buffer_to_constant:
                node.name = node.target = buffer_to_constant[node.name]
        mod.recompile()


# 恢复状态字典
def _restore_state_dict(
    original_module: torch.nn.Module, traced_module: torch.fx.GraphModule
) -> None:
    """
    Restore the state dictionary from the original module to the traced module.
    This function is typically used during model export or saving.
    """
    # 将跟踪模块的状态字典恢复为原始模块的状态字典
    """
    param_buffer_table = _get_param_buffer_mapping(original_module, traced_module)
    # 由于图模块是扁平化的（没有模块层次结构），我们需要通过将“.”替换为“_”来规范化模块。
    # 如果不这样做，它将尝试将权重保存到一个不再存在的子模块中。
    for name, fqn in param_buffer_table.items():
        param_buffer_table[name] = fqn.replace(".", "_")

    # 用 fqn 替换状态字典属性名称
    for name, fqn in param_buffer_table.items():
        if not hasattr(traced_module, name):
            continue

        attr = getattr(traced_module, name)
        if isinstance(attr, torch.Tensor) and not isinstance(attr, torch.nn.Parameter):
            traced_module.register_buffer(fqn, attr)
        else:
            setattr(traced_module, fqn, attr)
        delattr(traced_module, name)

    # 用正确的名称替换图中的 getattr 节点
    for node in traced_module.graph.nodes:
        if node.op == "get_attr":
            attr_name = node.target
            if attr_name in param_buffer_table:
                node.target = param_buffer_table[attr_name]

    # 重新编译跟踪模块
    traced_module.recompile()
# 定义一个函数，用于获取给定神经网络模块的层次结构字典，将每个模块的名称映射到其类型的名称
def _get_module_hierarchy(mod: torch.nn.Module) -> Dict[str, str]:
    return {
        name: type(m).__name__ for name, m in mod.named_modules(remove_duplicate=False)
    }

# 定义一个函数，创建模块调用图，返回一个包含模块调用信息的列表
def _make_module_call_graph(
    module_hierarchy: Dict[str, str],  # 模块的层次结构字典，映射模块名称到类型名称
    in_spec: TreeSpec,  # 输入的树形规范
    out_spec: TreeSpec,  # 输出的树形规范
    module_call_signatures: Dict[str, ModuleCallSignature],  # 模块调用签名字典，映射模块全限定名到调用签名
) -> List[ModuleCallEntry]:  # 返回一个模块调用条目的列表
    # 创建模块调用条目列表，每个条目包含全限定名和对应的模块调用签名
    ret = [
        ModuleCallEntry(fqn=fqn, signature=module_call_signatures.get(fqn))
        for fqn in module_hierarchy
    ]
    # 确保第一个模块调用条目的全限定名为空字符串
    assert ret[0].fqn == ""
    # 设置第一个模块调用条目的调用签名，指定输入、输出和输入规范、输出规范
    ret[0].signature = ModuleCallSignature(
        inputs=[], outputs=[], in_spec=in_spec, out_spec=out_spec
    )
    return ret  # 返回模块调用条目列表

# 定义一个函数，将函数或nn.Module的前向函数追踪为torch IR表示的图模块
def _export_to_torch_ir(
    f: Callable,  # 要追踪的函数或nn.Module的前向函数
    args: Tuple[Any, ...],  # 示例的位置输入参数元组
    kwargs: Optional[Dict[str, Any]] = None,  # 可选的关键字参数字典
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any], List[Any]]] = None,  # 可选的动态形状信息
    *,
    preserve_module_call_signature: Tuple[str, ...] = (),  # 保留模块调用签名的模块名称元组
    disable_constraint_solver: bool = False,  # 是否禁用约束求解器的标志
    _allow_complex_guards_as_runtime_asserts: bool = False,  # 是否允许复杂的守卫作为运行时断言的标志
    restore_fqn: bool = True,  # 是否恢复全限定名的标志
    _log_export_usage: bool = True,  # 是否记录导出使用情况的标志
    same_signature: bool = True,  # 是否要求相同的签名的标志
) -> torch.fx.GraphModule:  # 返回torch IR表示的图模块
    """
    追踪nn.Module的前向函数或包含PyTorch操作的可调用对象，并生成torch IR表示的图模块。
    """

    if _log_export_usage:
        log_export_usage(event="export.private_api", flags={"_export_to_torch_ir"})

    if not isinstance(args, tuple):
        raise UserError(
            UserErrorType.INVALID_INPUT,
            f"Expecting `args` to be a tuple of example positional inputs, got {type(args)}",
        )

    kwargs = kwargs or {}  # 如果kwargs为None，则设置为空字典
    # 使用指定的配置参数对 torch._dynamo 进行配置
    with torch._dynamo.config.patch(dataclasses.asdict(DEFAULT_EXPORT_DYNAMO_CONFIG)):
        # 初始化一个空字典，用于存储模块调用规范
        module_call_specs: Dict[str, Dict[str, pytree.TreeSpec]] = {}
        # 包装子模块，保留模块调用签名，并忽略后端分解
        with _wrap_submodules(
            f, preserve_module_call_signature, module_call_specs
        ), _ignore_backend_decomps():
            # 导出模型到 torch 级别，获取导出的模型和元数据
            gm_torch_level, _ = torch._dynamo.export(
                f,
                dynamic_shapes=dynamic_shapes,  # type: ignore[arg-type]
                assume_static_by_default=True,
                tracing_mode="symbolic",
                disable_constraint_solver=disable_constraint_solver,
                # 当前这两个标志在导出目的上是绑定在一起的，
                # 但为了 dynamo 导出 API 的清晰性，将其解开
                prefer_deferred_runtime_asserts_over_guards=_allow_complex_guards_as_runtime_asserts,
                _allow_complex_guards_as_runtime_asserts=_allow_complex_guards_as_runtime_asserts,
                _log_export_usage=_log_export_usage,
                same_signature=same_signature,
            )(
                *args,
                **kwargs,
            )
    # 捕获约束违规错误和值范围错误，抛出用户错误
    except (ConstraintViolationError, ValueRangeError) as e:
        raise UserError(UserErrorType.CONSTRAINT_VIOLATION, str(e))  # noqa: B904
    # 捕获数据相关符号节点上的守卫错误，抛出用户错误
    except GuardOnDataDependentSymNode as e:
        raise UserError(  # noqa: B904
            UserErrorType.ANTI_PATTERN,
            f"Consider annotating your code using torch._check*(). {str(e)}",
            case_name="constrain_as_size_example",
        )

    # 将模块调用规范存储在 gm_torch_level 的元数据中
    gm_torch_level.meta["module_call_specs"] = module_call_specs

    # 如果 f 是 torch.nn.Module 类型且 restore_fqn 为真，则恢复状态字典
    if isinstance(f, torch.nn.Module) and restore_fqn:
        _restore_state_dict(f, gm_torch_level)

    # 返回导出的 torch 级别模型
    return gm_torch_level
# 定义一个函数用于将模型导出为 ATen IR 格式
def _export_to_aten_ir(
    mod: torch.nn.Module,
    fake_args,
    fake_kwargs,
    fake_params_buffers,
    constant_attrs: ConstantAttrMap,
    *,
    transform=lambda x: x,  # 定义一个可选的转换函数，默认为返回输入对象本身
    pre_dispatch=False,  # 控制是否进行预分派的布尔值，默认为 False
    _is_torch_jit_trace=False,  # 控制是否为 Torch JIT 的跟踪状态的布尔值，默认为 False
) -> ATenExportArtifact:
    # [NOTE] 如果用户在训练模式下导出，我们希望检测自动求导全局状态是否有状态更改，并报错。
    # 如果用户在推断模式下导出，我们不关心状态。
    # 在预分派级别，我们也不关心状态更改。
    is_grad_enabled = torch._C.is_grad_enabled()
    grad_safe_guard = nullcontext()
    if not pre_dispatch and is_grad_enabled:
        # 如果不是预分派且梯度启用，则创建一个自动求导状态操作的容错保护器
        grad_safe_guard = AutogradStateOpsFailSafeguard()  # 类型标注忽略赋值的警告

    @contextmanager
    def _compiling_state_context():
        # 定义一个上下文管理器函数，用于设置和恢复编译状态的标志位
        old_value = torch.compiler._is_compiling_flag
        try:
            torch.compiler._is_compiling_flag = True
            yield
        finally:
            torch.compiler._is_compiling_flag = old_value

    # 这个 _reparametrize_module 函数确保输入和 module.params/buffers 具有相同的 fake_mode，
    # 否则 aot_export_module 将会因为看到混合的 fake_mode 而报错。
    # 我们希望 aot_export_module 使用 dynamo 中的 fake_tensor 模式，以便保持流水线的易于理解性。
    with torch.nn.utils.stateless._reparametrize_module(
        mod,
        fake_params_buffers,
        tie_weights=True,
        strict=True,
        stack_weights=True,
    ), grad_safe_guard, _ignore_backend_decomps(), _compiling_state_context():  # 类型标注忽略属性定义的警告
        # 调用 transform(aot_export_module) 函数，传入模型和伪参数，获取导出的图模块和图签名
        gm, graph_signature = transform(aot_export_module)(
            mod,
            fake_args,
            trace_joint=False,
            pre_dispatch=pre_dispatch,
            kwargs=fake_kwargs,
        )
    # TODO 不幸的是，保留图级元数据与 aot_export 不兼容。因此我们手动复制它。
    # （节点级别的元数据在上面已经处理过。）
    if isinstance(mod, torch.fx.GraphModule) and hasattr(mod, "meta"):
        # 如果模型是 torch.fx.GraphModule 类型且具有 "meta" 属性，则更新 gm 的元数据
        gm.meta.update(mod.meta)
    # 定义一个函数，用于生成参数规范，接受两个参数：索引 i 和节点 node，返回 ArgumentSpec 类型的对象
    def make_argument_spec(i, node) -> ArgumentSpec:
        # 如果节点是 int、bool、float 或者 NoneType 中的一种，返回一个 ConstantArgument 对象
        if isinstance(node, (int, bool, float, type(None))):
            # 对于常量输出，直接返回 ConstantArgument 对象
            return ConstantArgument(name="", value=node)

        # 确保节点的 meta 字典中包含 'val' 键
        assert (
            "val" in node.meta
        ), f"{node} is not a constant or a node with a 'val' metadata field"
        # 获取节点的 'val' 值
        val = node.meta["val"]
        
        # 根据索引 i 判断节点类型，并返回相应的 ArgumentSpec 对象
        if i < len(graph_signature.input_tokens):
            # TODO: 一旦添加了新的类型，我们应该检查不同的类型
            return TokenArgument(name=node.name)
        elif isinstance(val, FakeTensor):
            return TensorArgument(name=node.name)
        elif isinstance(val, torch.SymInt):
            return SymIntArgument(name=node.name)
        elif isinstance(val, torch.ScriptObject):
            # 如果 val 是 torch.ScriptObject 类型，则返回 CustomObjArgument 对象
            return CustomObjArgument(name=node.name, class_fqn=val._type().qualified_name())  # type: ignore[attr-defined]
        elif isinstance(val, FakeScriptObject):
            # 如果 val 是 FakeScriptObject 类型，则返回 CustomObjArgument 对象
            return CustomObjArgument(
                name=node.name, class_fqn=val.script_class_name, fake_val=val
            )
        elif isinstance(val, (int, bool, str, float, type(None))):
            # 如果 val 是 int、bool、str、float 或 NoneType 中的一种，则返回 ConstantArgument 对象
            return ConstantArgument(name=node.name, value=val)
        else:
            # 如果 val 是不支持的类型，则抛出 AssertionError
            raise AssertionError(
                f"Encountered an unsupported object of type {type(val)} "
                f"while writing the metadata for exported program"
            )

    # 检查是否存在反向签名，用于确定是否是联合训练
    is_joint = graph_signature.backward_signature is not None

    # 注意：aot_export 为具有 int 值的占位符添加 symint 元数据；
    # 由于这些占位符变得专门化，我们用原始值替换这样的元数据
    flat_fake_args = pytree.tree_leaves((fake_args, fake_kwargs))
    index = 0
    # 计算总的非用户输入参数数量，包括 parameters、buffers 和 input_tokens
    total_non_user_inputs = (
        len(graph_signature.parameters)
        + len(graph_signature.buffers)
        + len(graph_signature.input_tokens)
    )
    # 遍历计算图中的每个节点
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            # 如果节点的操作类型为 "placeholder"
            if index >= total_non_user_inputs:
                # 如果索引超过了总的非用户输入参数数量
                user_arg = flat_fake_args[index - total_non_user_inputs]
                if not isinstance(user_arg, torch.Tensor):
                    # 如果用户参数不是 torch.Tensor 类型，则将其赋值给节点的 'val' 元数据
                    node.meta["val"] = user_arg
            index += 1
    # 调用 _sig_to_specs 函数，将图形签名转换为输入和输出规范
    input_specs, output_specs = _sig_to_specs(
        user_inputs=set(graph_signature.user_inputs),
        # 将图形签名中的用户输入转换为集合
        inputs_to_parameters=graph_signature.inputs_to_parameters,  # type: ignore[arg-type]
        # 将图形签名中的输入到参数映射传递给函数，忽略类型检查
        inputs_to_buffers=graph_signature.inputs_to_buffers,  # type: ignore[arg-type]
        # 将图形签名中的输入到缓冲区映射传递给函数，忽略类型检查
        user_outputs=set(graph_signature.user_outputs),  # type: ignore[arg-type]
        # 将图形签名中的用户输出转换为集合，忽略类型检查
        buffer_mutations=graph_signature.buffers_to_mutate,  # type: ignore[arg-type]
        # 将图形签名中的要变异的缓冲区映射传递给函数，忽略类型检查
        user_input_mutations=graph_signature.user_inputs_to_mutate,  # type: ignore[arg-type]
        # 将图形签名中的要变异的用户输入映射传递给函数，忽略类型检查
        grad_params=graph_signature.backward_signature.gradients_to_parameters if is_joint else {},  # type: ignore[arg-type, union-attr]
        # 如果是联合图，将反向传播中的梯度到参数映射传递给函数，忽略类型检查
        grad_user_inputs=graph_signature.backward_signature.gradients_to_user_inputs if is_joint else {},  # type: ignore[arg-type, union-attr]
        # 如果是联合图，将反向传播中的梯度到用户输入映射传递给函数，忽略类型检查
        loss_output=graph_signature.backward_signature.loss_output if is_joint else None,  # type: ignore[arg-type, union-attr]
        # 如果是联合图，将反向传播中的损失输出传递给函数，忽略类型检查
        inputs=[
            make_argument_spec(i, node)
            for i, node in enumerate(gm.graph.nodes)
            if node.op == "placeholder"
        ],
        # 生成图形中所有占位符节点的参数规范列表，如果节点操作为"placeholder"
        outputs=[
            make_argument_spec(i, node)
            for i, node in enumerate(
                pytree.tree_leaves(next(iter(reversed(gm.graph.nodes))).args)
            )
        ],
        # 生成图形中所有输出节点的参数规范列表，使用 PyTree 库处理图形节点
        input_tokens=graph_signature.input_tokens,
        # 传递图形签名中的输入令牌给函数
        output_tokens=graph_signature.output_tokens,
        # 传递图形签名中的输出令牌给函数
    )

    # 创建 ExportGraphSignature 对象，包含输入和输出的规范
    export_graph_signature = ExportGraphSignature(
        input_specs=input_specs, output_specs=output_specs
    )

    # 从 torch._guards 模块导入 detect_fake_mode 函数
    fake_mode = detect_fake_mode(flat_fake_args)

    # 从 torch._dynamo 模块导入 config as _dynamo_config
    if not _dynamo_config.do_not_emit_runtime_asserts:
        # 定义堆栈跟踪信息
        stack_trace = (
            'File "torch/fx/passes/runtime_assert.py", line 24, '
            "in insert_deferred_runtime_asserts"
        )
        # 设置节点元数据挂钩，调用 _node_metadata_hook 函数，并传递堆栈跟踪信息
        with _set_node_metadata_hook(
            gm, functools.partial(_node_metadata_hook, stack_trace=stack_trace)
        ):
            # 插入延迟运行时断言到图中
            insert_deferred_runtime_asserts(
                gm,
                fake_mode.shape_env,
                f"exported program: {first_call_function_nn_module_stack(gm.graph)}",
                export=True,
            )

    if pre_dispatch:
        # 从 torch._export.passes.replace_set_grad_with_hop_pass 模块导入 replace_set_grad_with_hop_pass 函数
        gm, export_graph_signature = replace_set_grad_with_hop_pass(
            gm, export_graph_signature
        )

    # 遍历所有模块，并处理占位符和输出节点的元数据
    for _mod in gm.modules():
        if not isinstance(_mod, torch.fx.GraphModule):
            continue
        for node in _mod.graph.nodes:
            if node.op in ["placeholder", "output"]:
                # 删除节点的 nn_module_stack 和 stack_trace 元数据
                node.meta.pop("nn_module_stack", None)
                node.meta.pop("stack_trace", None)

    # 重写脚本对象的元数据
    constants = rewrite_script_object_meta(gm)
    # 更新常量表，应用提取常量的优化函数
    constants.update(lift_constants_pass(gm, export_graph_signature, constant_attrs))

    # 为占位节点改善命名
    placeholder_naming_pass(
        gm,  # 图形模型对象
        export_graph_signature,  # 导出图形的签名
        mod,  # 模块
        fake_args,  # 虚拟参数
        fake_kwargs,  # 虚拟关键字参数
        fake_params_buffers,  # 虚拟参数缓冲区
        constants,  # 常量表
    )

    # 保留梯度要求传递的优化函数
    _preserve_requires_grad_pass(
        gm,  # 图形模型对象
        export_graph_signature,  # 导出图形的签名
        fake_params_buffers,  # 虚拟参数缓冲区
        constants,  # 常量表
        flat_fake_args  # 扁平化的虚拟参数
    )

    # 返回 ATenExportArtifact 对象，其中包含图形模型、导出图形的签名和常量表
    return ATenExportArtifact(
        gm,  # 图形模型对象
        export_graph_signature,  # 导出图形的签名
        constants,  # 常量表
    )
# 获取模块中所有参数和缓冲区的字典，包括重复的名称
def _get_params_buffers(mod: torch.nn.Module) -> Dict[str, torch.Tensor]:
    # 初始化一个空字典，用于存储参数和缓冲区
    params_buffers: Dict[str, torch.Tensor] = {}
    # 遍历模块中的所有参数，将其添加到字典中
    for name, param in mod.named_parameters(remove_duplicate=False):
        params_buffers[name] = param

    # 遍历模块中的所有缓冲区，将其添加到字典中
    for name, buffer in mod.named_buffers(remove_duplicate=False):
        params_buffers[name] = buffer
    # 返回参数和缓冲区的字典
    return params_buffers


def _get_forward_arg_names(
    mod: torch.nn.Module,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    获取模块前向传播函数的参数名列表，用于恢复导出模块的原始签名。
    - 位置参数：保留原始参数名，并枚举*args为args_0, args_1, ...
    - 关键字参数：保留用户指定顺序的原始关键字参数名，当前导出的模块对参数顺序敏感。
    """
    # 获取前向传播函数的签名
    sig = inspect.signature(mod.forward)
    # 绑定部分参数以获取实际使用的参数名和值
    _args = sig.bind_partial(*args).arguments

    # 初始化一个空列表，用于存储参数名
    names: List[str] = []
    # 遍历绑定的参数名和值
    for name, value in _args.items():
        # 处理可变数量的位置参数
        if sig.parameters[name].kind == inspect._ParameterKind.VAR_POSITIONAL:
            # 枚举位置参数并添加到列表中，例如args_0, args_1, ...
            names.extend([f"{name}_{i}" for i, _ in enumerate(value)])
        else:
            # 添加非可变参数名到列表中
            names.append(name)
    # 如果有关键字参数，按照指定顺序将其添加到列表中
    if kwargs:
        names.extend([kwarg for kwarg, _ in kwargs.items()])

    # 返回参数名列表
    return names


def _rewrite_dynamo_tensor_constants(
    orig_mod_buffers: Set[torch.Tensor],
    traced_mod_buffers: Dict[str, torch.Tensor],
    graph_signature: ExportGraphSignature,
    constants: Dict[str, Union[torch.Tensor, FakeScriptObject, torch.ScriptObject]],
):
    """Dynamo错误地将模块上的张量属性标记为缓冲区。

    将它们重写为张量常量。
    """
    # 遍历图签名中的输入规格
    for spec in graph_signature.input_specs:
        # 如果规格类型为缓冲区
        if spec.kind == InputKind.BUFFER:
            assert spec.target is not None
            # 获取跟踪到的模块缓冲区的值
            value = traced_mod_buffers[spec.target]
            # 如果该值不在原始模块缓冲区中
            if value not in orig_mod_buffers:
                # 将其标记为图签名中的张量常量，并将其值添加到常量表中
                spec.kind = InputKind.CONSTANT_TENSOR
                constants[spec.target] = value  # type: ignore[arg-type]


def _rewrite_non_persistent_buffers(
    orig_mod: torch.nn.Module,
    graph_signature: ExportGraphSignature,
    constants: Dict[str, Union[torch.Tensor, FakeScriptObject, torch.ScriptObject]],
):
    """Dynamo错误地删除了缓冲区的持久标志。

    重写非持久缓冲区以反映原始模块。
    """
    # 获取原始模块的状态字典
    state_dict = orig_mod.state_dict()
    # 遍历图签名中的输入规范列表
    for spec in graph_signature.input_specs:
        # 检查输入规范的类型是否为 BUFFER
        if spec.kind == InputKind.BUFFER:
            # 确保目标不为空
            assert spec.target is not None
            # 如果目标不在状态字典中
            if spec.target not in state_dict:
                # 确保目标不在常量列表中
                assert spec.target not in constants
                # 将规范的持久性设置为 False
                spec.persistent = False
                # 从原始模型中获取指定目标的缓冲区并加入常量字典中
                constants[spec.target] = orig_mod.get_buffer(spec.target)  # type: ignore[arg-type]
def _verify_nn_module_stack(graph_module: torch.fx.GraphModule) -> None:
    """
    Perform nn_module_stack checks on the graph.
    Current constraints:
        For the top level graph:
        - populated for 'call_function', 'get_attr'
        - None for 'placeholder', 'output'
        For submodule graphs:
        - None for 'placeholder', output'

    TODO(pianpwk): make this a consistent node-level check once nn_module_stack is populated for cond submodules.
    """
    # Check top-level graph for all nodes, all graphs for placeholder & output nodes
    # 检查顶层图的所有节点，所有子模块图的占位符和输出节点
    for i, mod in enumerate([graph_module] + list(graph_module.modules())):
        # 遍历顶层图和所有子模块图
        if not isinstance(mod, torch.fx.GraphModule):
            continue
        # 如果不是GraphModule类型，跳过
        for node in mod.graph.nodes:
            # 遍历每个节点
            if node.op in ["call_function", "get_attr"]:
                # 如果节点操作是'call_function'或'get_attr'
                if i == 0:
                    # 对于顶层图
                    if (
                        nn_module_stack := node.meta.get("nn_module_stack", None)
                    ) is None:
                        # 获取节点的nn_module_stack元数据，若为None则抛出异常
                        raise SpecViolationError(
                            f"Node {node} of type {node.op} is missing nn_module_stack metadata"
                        )
                    if not all(
                        isinstance(k, str)
                        and isinstance(v, tuple)
                        and len(v) == 2
                        and all(isinstance(x, str) for x in v)
                        for k, v in nn_module_stack.items()
                    ):
                        # 检查nn_module_stack的格式是否正确，若不正确则抛出异常
                        raise SpecViolationError(
                            f"Node {node} of type {node.op} has incorrect nn_module_stack metadata format"
                            f"expected Dict[str, Tuple[str, str]], but got {nn_module_stack}"
                        )
            elif node.op in ["placeholder", "output"]:
                # 如果节点操作是'placeholder'或'output'
                if node.meta.get("nn_module_stack", None):
                    # 如果节点的nn_module_stack不为None，则抛出异常
                    raise SpecViolationError(
                        f"Node {node} of type {node.op} contains nn_module_stack metadata, this should be None"
                    )


def _verify_stack_trace(graph_module: torch.fx.GraphModule) -> None:
    """
    Perform stack trace checks on the graph.
    Constraints:
        - None or non-empty str for 'call_function', 'get_attr'
        - None for 'placeholder', 'output'
    """
    # Perform stack trace checks on the graph
    # 在图上执行堆栈跟踪检查
    constraints_description = (
        "- None or non-empty str for 'call_function', 'get_attr'\n"
        "- None for 'placeholder', 'output'"
    )
    # 堆栈跟踪检查的约束描述
    # 遍历 graph_module 及其所有子模块的列表，同时保留索引信息
    for i, mod in enumerate([graph_module] + list(graph_module.modules())):
        # 如果当前模块不是 torch.fx.GraphModule 类型，则跳过本次循环
        if not isinstance(mod, torch.fx.GraphModule):
            continue
        # 遍历当前模块的计算图中的所有节点
        for node in graph_module.graph.nodes:
            # 获取节点的元数据中的 stack_trace，如果不存在则为 None
            stack_trace = node.meta.get("stack_trace", None)
            # 如果节点操作类型是 "call_function" 或者 "get_attr"
            if node.op in ["call_function", "get_attr"]:
                # 如果 stack_trace 不为 None 且不是字符串类型，则抛出异常
                if not (stack_trace is None or isinstance(stack_trace, str)):
                    raise SpecViolationError(
                        f"Node {node} of type {node.op} has invalid stack_trace metadata, "
                        f"expected a string or None but instead found: {stack_trace}"
                    )
            # 如果节点操作类型是 "placeholder" 或者 "output"
            elif node.op in ["placeholder", "output"]:
                # 如果 stack_trace 存在（不为 None），则抛出异常
                if stack_trace:
                    raise SpecViolationError(
                        f"Node {node} of type {node.op} contains stack_trace metadata, "
                        f"expected None but instead found: {stack_trace}"
                    )
def _verify_placeholder_names(gm: torch.fx.GraphModule, sig: ExportGraphSignature):
    """
    Performs a sanity check on the placeholder node names.
    - User input nodes: no restrictions, should match the original forward() signature
    - Params/buffers/constants/custom_obj/token nodes: should start with prefixes defined in <placeholder_prefixes>
    """
    # 创建一个字典，将输入规范中的名称映射到它们的类型
    name_to_kind = {spec.arg.name: spec.kind for spec in sig.input_specs}
    
    # 遍历所有模块
    for mod in gm.modules():
        # 如果模块不是 torch.fx.GraphModule 类型，则继续下一个模块
        if not isinstance(mod, torch.fx.GraphModule):
            continue
        
        # 遍历当前模块的所有节点
        for node in mod.graph.nodes:
            # 检查节点操作是否为 "placeholder"
            if node.op == "placeholder":
                # 如果节点名称不在输入规范字典中，则继续下一个节点
                if node.name not in name_to_kind:
                    continue
                
                # 获取节点的类型（从输入规范中获取）
                node_kind = name_to_kind[node.name]
                # 根据节点类型获取对应的前缀
                prefix = placeholder_prefixes[node_kind]
                
                # 检查节点名称是否以正确的前缀开头，否则抛出异常
                if not node.name.startswith(prefix):
                    raise SpecViolationError(
                        f"Placeholder node name {node.name} does not follow spec for {node_kind}, name should have prefix: {prefix}"
                    )


def get_ep_stats(ep: ExportedProgram) -> Dict[str, Any]:
    op_count = 0
    op_set = set()
    
    # 遍历 ExportedProgram 中的所有模块
    for m in ep.graph_module.modules():
        # 如果模块不是 torch.fx.GraphModule 类型，则继续下一个模块
        if not isinstance(m, torch.fx.GraphModule):
            continue
        
        # 遍历当前模块的所有节点
        for node in m.graph.nodes:
            # 检查节点操作是否为 "call_function"
            if node.op != "call_function":
                continue
            
            # 统计操作数量
            op_count += 1
            # 确保节点目标具有 "__module__" 和 "__name__" 属性
            assert hasattr(node.target, "__module__")
            assert hasattr(node.target, "__name__")
            # 将调用函数的模块和名称添加到集合中
            op_set.add(f"{node.target.__module__}.{node.target.__name__}")
    
    # 返回包含操作数量和操作集合的字典
    return {"op_count": op_count, "op_set": op_set}


_EXPORT_FLAGS: Optional[Set[str]] = None
_EXPORT_MODULE_HIERARCHY: Optional[Dict[str, str]] = None


def _log_export_wrapper(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # 声明全局变量
        global _EXPORT_FLAGS, _EXPORT_MODULE_HIERARCHY
        try:
            # 记录函数执行开始时间
            start = time.time()
            # 执行函数并获取结果
            ep = fn(*args, **kwargs)
            # 记录函数执行结束时间
            end = time.time()
            # 记录导出使用情况，包括时间、标志和统计信息
            log_export_usage(
                event="export.time",
                metrics=end - start,
                flags=_EXPORT_FLAGS,
                **get_ep_stats(ep),
            )
        except Exception as e:
            # 获取异常类型
            t = type(e)
            error_type = t.__module__ + "." + t.__qualname__
            # 记录导出使用情况中的错误信息
            log_export_usage(
                event="export.error",
                type=error_type,
                message=str(e),
                flags=_EXPORT_FLAGS,
            )
            # 重新抛出异常
            raise e
        finally:
            # 重置全局标志和模块层次结构
            _EXPORT_FLAGS = None
            _EXPORT_MODULE_HIERARCHY = None
        
        # 返回函数执行的结果
        return ep


def _process_jit_trace_inputs_for_export(example_inputs, example_kwarg_inputs):
    # 如果 example_inputs 不是元组、列表或字典，则转换为元组
    if not isinstance(example_inputs, (tuple, list, dict)):
        example_inputs = (example_inputs,)
    
    # 如果 example_inputs 是列表，则转换为元组
    elif isinstance(example_inputs, list):
        example_inputs = tuple(example_inputs)
    # 如果 example_kwarg_inputs 为 None，并且 example_inputs 是 torch.Tensor 或 dict 类型之一
    # 将 example_inputs 转换为包含它自身的元组
    elif (
        isinstance(example_inputs, (torch.Tensor, dict))
        and example_kwarg_inputs is None
    ):
        example_inputs = (example_inputs,)
    
    # 如果 example_kwarg_inputs 仍然为 None，将其设为一个空字典
    if example_kwarg_inputs is None:
        example_kwarg_inputs = {}
    
    # 返回处理后的 example_inputs 和 example_kwarg_inputs
    return example_inputs, example_kwarg_inputs
@contextmanager
def patch_forward(obj: torch.nn.Module, new_method):
    """Helper method to make it easier to cleanly torch.export() a method on a
    module that is not `forward`.
    """
    # 保存原始方法
    original_method = obj.forward

    # 替换方法为新方法
    obj.forward = new_method.__get__(obj, obj.__class__)

    try:
        yield
    finally:
        # 恢复原始方法
        obj.forward = original_method


@contextmanager
def _temp_disable_texpr_fuser():
    """Context manager to temporarily disable the Torch tensor expression fuser."""
    original_state = torch._C._jit_texpr_fuser_enabled()
    # 禁用 Torch 张量表达式融合器
    torch._C._jit_set_texpr_fuser_enabled(False)
    try:
        yield
    finally:
        # 恢复原始状态
        torch._C._jit_set_texpr_fuser_enabled(original_state)


class _WrapperModule(torch.nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, *args, **kwargs):
        return self.f(*args, **kwargs)


def _convert_ts_to_export_experimental(traced_callable, args, kwargs=None):
    """Convert a traced callable to an exportable Torch module."""
    with _temp_disable_texpr_fuser():
        from torch.jit._trace import TopLevelTracedModule

        export_args, export_kwargs = _process_jit_trace_inputs_for_export(args, kwargs)

        if isinstance(traced_callable, (TopLevelTracedModule, torch._C.ScriptModule)):  # type: ignore[operator]
            # Export Torch module if it's already traced or scripted
            return _export(
                traced_callable,
                export_args,
                export_kwargs,
                strict=False,
                _is_torch_jit_trace=True,
            ).module()

        elif isinstance(traced_callable, torch.ScriptMethod) and isinstance(
            traced_callable.owner(), (torch._C.ScriptModule, torch.nn.Module)  # type: ignore[operator]
        ):
            with patch_forward(traced_callable.owner(), traced_callable):  # type: ignore[operator]
                # Export Torch module with patched forward method
                return _export(
                    traced_callable.owner(),  # type: ignore[operator]
                    export_args,
                    export_kwargs,
                    strict=False,
                    _is_torch_jit_trace=True,
                ).module()

        else:
            # Export Torch module using a wrapper module
            return _export(
                _WrapperModule(traced_callable),
                export_args,
                export_kwargs,
                strict=False,
                _is_torch_jit_trace=True,
            ).module()


def _strict_export(
    mod: torch.nn.Module,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any], List[Any]]],
    preserve_module_call_signature: Tuple[str, ...],
    pre_dispatch: bool,
    original_state_dict: Dict[str, Any],
    orig_in_spec: TreeSpec,
    _allow_complex_guards_as_runtime_asserts: bool,
    _disable_forced_specializations: Optional[bool],
    _is_torch_jit_trace: bool,
) -> ExportArtifact:
    """Strict export function for Torch modules."""
    lower_to_aten = functools.partial(_export_to_aten_ir, pre_dispatch=pre_dispatch)
    # 调用函数 `_strict_export_lower_to_aten_ir`，并传入以下参数：
    # - `mod`: 模型对象
    # - `args`: 位置参数
    # - `kwargs`: 关键字参数
    # - `dynamic_shapes`: 是否使用动态形状
    # - `preserve_module_call_signature`: 是否保留模块调用签名
    # - `pre_dispatch`: 预调度选项
    # - `original_state_dict`: 原始状态字典
    # - `orig_in_spec`: 是否在规范中使用原始对象
    # - `_allow_complex_guards_as_runtime_asserts`: 是否允许复杂守卫作为运行时断言
    # - `_disable_forced_specializations`: 是否禁用强制特化
    # - `_is_torch_jit_trace`: 是否为 Torch JIT 追踪
    # - `lower_to_aten_callback`: 用于降低到 ATen 操作的回调函数
    return _strict_export_lower_to_aten_ir(
        mod=mod,
        args=args,
        kwargs=kwargs,
        dynamic_shapes=dynamic_shapes,
        preserve_module_call_signature=preserve_module_call_signature,
        pre_dispatch=pre_dispatch,
        original_state_dict=original_state_dict,
        orig_in_spec=orig_in_spec,
        _allow_complex_guards_as_runtime_asserts=_allow_complex_guards_as_runtime_asserts,
        _disable_forced_specializations=_disable_forced_specializations,
        _is_torch_jit_trace=_is_torch_jit_trace,
        lower_to_aten_callback=lower_to_aten,
    )
# 定义一个函数，用于将模型转换为 ATen IR 表示的导出工具
def _strict_export_lower_to_aten_ir(
    mod: torch.nn.Module,  # 输入的 PyTorch 模型
    args: Tuple[Any, ...],  # 位置参数的元组
    kwargs: Dict[str, Any],  # 关键字参数的字典
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any], List[Any]]],  # 动态形状的可选参数
    preserve_module_call_signature: Tuple[str, ...],  # 保留模块调用签名的元组
    pre_dispatch: bool,  # 是否预调度的布尔值
    original_state_dict: Dict[str, Any],  # 原始状态字典的字典
    orig_in_spec: TreeSpec,  # 原始输入规范的 TreeSpec 对象
    _allow_complex_guards_as_runtime_asserts: bool,  # 允许复杂保护作为运行时断言的布尔值
    _disable_forced_specializations: Optional[bool],  # 禁用强制特化的可选布尔值
    _is_torch_jit_trace: bool,  # 是否为 Torch JIT 跟踪的布尔值
    lower_to_aten_callback: Callable,  # 用于降低至 ATen 的回调函数
) -> ExportArtifact:  # 返回导出工件的类型注解

    # 将模型导出为 Torch IR 表示
    gm_torch_level = _export_to_torch_ir(
        mod,
        args,
        kwargs,
        dynamic_shapes,
        preserve_module_call_signature=preserve_module_call_signature,
        restore_fqn=False,  # 不需要恢复全限定名，因为稍后会执行
        _allow_complex_guards_as_runtime_asserts=_allow_complex_guards_as_runtime_asserts,
        _log_export_usage=False,  # 不记录导出使用情况
    )

    # 通过转换输入来获取 fake_mode，这是 dynamo 中创建的 fake_mode
    (
        fake_args,
        fake_kwargs,
        fake_params_buffers,
        dynamo_fake_mode,
    ) = _convert_input_to_fake(gm_torch_level, args, kwargs)

    # 遍历图中的节点，尝试为未定义 "val" 的 get_attr 节点填充 "val" 字段
    # 这可能发生在量化过程中添加额外参数但忘记更新 "val" 的情况
    for node in gm_torch_level.graph.nodes:
        if node.op == "get_attr" and "val" not in node.meta:
            attr = getattr(gm_torch_level, node.target)
            # 检查是否不是 HigherOrderOp 分支或模块
            if not isinstance(attr, torch.nn.Module):
                assert (
                    dynamo_fake_mode is not None
                ), "Cannot find dynamo_fake_mode. This could be due to the exported graph module have no placeholders."
                node.meta["val"] = dynamo_fake_mode.from_tensor(
                    attr, static_shapes=True
                )

    # 当 aot_export 提升参数时，我们会丢失参数节点的元数据（例如 source_fn_stack, stack_trace）
    # 因为它们被视为全新的输入，所以在调用 aot_export 之前我们手动提取它们
    params_buffers_to_node_meta = {}
    # 遍历 TorchScript 图中的每个节点
    for node in gm_torch_level.graph.nodes:
        # 获取节点的目标
        target = node.target
        # 获取节点的元数据
        meta = node.meta
        
        # 如果节点操作是调用模块
        if node.op == "call_module":
            # 获取模块对象
            submodule = getattr(gm_torch_level, target)
            # 如果是 torch.nn.Module 实例
            if isinstance(submodule, torch.nn.Module):
                # 遍历模块的命名参数，将参数名与节点的元数据关联起来
                for name, _ in submodule.named_parameters(
                    recurse=True, remove_duplicate=False
                ):
                    params_buffers_to_node_meta[target + "." + name] = meta

                # 遍历模块的命名缓冲区，将缓冲区名与节点的元数据关联起来
                for name, _ in submodule.named_buffers(
                    recurse=True, remove_duplicate=False
                ):
                    params_buffers_to_node_meta[target + "." + name] = meta
        
        # 如果节点操作是获取属性
        if node.op == "get_attr":
            # 获取属性值
            submodule = getattr(gm_torch_level, target)
            # 如果不是 torch.fx.GraphModule 实例
            if not isinstance(submodule, torch.fx.GraphModule):
                # 将属性名与节点的元数据关联起来
                params_buffers_to_node_meta[target] = meta

        # 如果节点操作是调用函数，并且目标不是高阶运算符
        if node.op == "call_function" and not isinstance(
            node.target, torch._ops.HigherOrderOperator
        ):
            # 遍历节点的输入节点
            for arg in node._input_nodes:
                # 如果输入节点的操作是获取属性
                if arg.op == "get_attr":
                    # 将节点的元数据字段复制到参数的元数据中
                    for entry in torch.fx.proxy._COPY_META_FIELDS:
                        if entry in meta:
                            params_buffers_to_node_meta[arg.target][entry] = meta[entry]

    # 修正图输出签名为元组，以适应标量情况
    out_spec = orig_out_spec = gm_torch_level._out_spec

    # 断言确保输出签名和原始输出签名都不为空
    assert out_spec is not None
    assert orig_out_spec is not None

    # 如果输出签名的类型不是列表或元组，则将其修正为单元素元组
    if out_spec.type not in (list, tuple):
        out_spec = pytree.TreeSpec(tuple, None, [out_spec])

    # 获取原始参数名
    orig_arg_names = gm_torch_level.graph._codegen.pytree_info.orig_args  # type: ignore[attr-defined]

    # 重新设置代码生成器和代码树信息
    gm_torch_level.graph._codegen = _PyTreeCodeGen(
        _PyTreeInfo(
            orig_arg_names,
            gm_torch_level._in_spec,
            out_spec,
        )
    )
    # 重新编译 TorchScript 图
    gm_torch_level.recompile()

    # 标准化 nn.Module 的堆栈信息
    _normalize_nn_module_stack(gm_torch_level, type(mod))

    # 注意：图模块期望只有位置参数
    # 收集模块中的常量属性
    constant_attrs = _gather_constant_attrs(mod)
    # 在虚拟 Dynamo 模式下，将 TorchScript 图转换为 ATen 格式的回调函数
    with dynamo_fake_mode:
        aten_export_artifact = lower_to_aten_callback(
            gm_torch_level,
            _convert_to_positional_args(orig_arg_names, fake_args, fake_kwargs),
            {},
            fake_params_buffers,
            constant_attrs,
        )

    # 解构以提升可读性
    gm = aten_export_artifact.gm
    export_graph_signature = aten_export_artifact.sig
    constants = aten_export_artifact.constants

    # 不复制 nn_module_stack 和 params/buffers 节点的 stack_trace 元数据
    # 遍历 params_buffers_to_node_meta 字典中的每个元数据对象
    for metadata in params_buffers_to_node_meta.values():
        # 移除元数据中的 "nn_module_stack" 和 "stack_trace" 键
        metadata.pop("nn_module_stack", None)
        metadata.pop("stack_trace", None)

    # 在执行 aot_export 后，将参数/缓冲区的元数据重新设置到占位符中
    # 技术上，用户仍然可以根据参数名称构建这些数据，而不依赖于这些元数据
    for node in gm.graph.nodes:
        # 如果节点是占位符
        if node.op == "placeholder":
            # 如果节点的目标在 export_graph_signature.inputs_to_parameters 中
            if node.target in export_graph_signature.inputs_to_parameters:
                # 获取参数名称
                param_name = export_graph_signature.inputs_to_parameters[node.target]
                # 如果参数名称存在于 params_buffers_to_node_meta 中
                if param_name in params_buffers_to_node_meta:
                    # 将 params_buffers_to_node_meta 中的元数据复制到节点的 meta 属性中
                    for k, v in params_buffers_to_node_meta[param_name].items():
                        node.meta[k] = v
            # 如果节点的目标在 export_graph_signature.inputs_to_buffers 中
            if node.target in export_graph_signature.inputs_to_buffers:
                # 获取缓冲区名称
                buffer_name = export_graph_signature.inputs_to_buffers[node.target]
                # 如果缓冲区名称存在于 params_buffers_to_node_meta 中
                if buffer_name in params_buffers_to_node_meta:
                    # 将 params_buffers_to_node_meta 中的元数据复制到节点的 meta 属性中
                    for k, v in params_buffers_to_node_meta[buffer_name].items():
                        node.meta[k] = v

    # 对图模块进行一些清理操作，将状态字典恢复到预期的形式
    # 这些步骤每一个都可能需要在上游进行修复

    # 1. 移除作为缓冲区添加的张量常量
    _rewrite_dynamo_tensor_constants(
        orig_mod_buffers=set(mod.buffers()),
        traced_mod_buffers=dict(gm_torch_level.named_buffers()),
        graph_signature=export_graph_signature,
        constants=constants,
    )

    # 2. 恢复参数/缓冲区的完全限定名称（FQN）
    param_buffer_table: Dict[str, str] = _get_param_buffer_mapping(mod, gm_torch_level)
    _replace_param_buffer_names(param_buffer_table, export_graph_signature)

    # 3. 从图签名中移除非持久性缓冲区
    _rewrite_non_persistent_buffers(mod, export_graph_signature, constants)

    # 4. 重写常量，使其具有与原始模块相同的完全限定名称（FQN）
    _remap_constants(constant_attrs, export_graph_signature, constants)

    # 5. 将图模块中的常量节点从缓冲区重命名为常量
    _rename_constants_nodes(gm, export_graph_signature)

    # 返回导出的结果对象 ExportArtifact
    return ExportArtifact(
        aten=aten_export_artifact,
        out_spec=orig_out_spec,
        fake_mode=dynamo_fake_mode,
        module_call_specs=gm_torch_level.meta["module_call_specs"],
    )
# 定义一个函数 `_export_to_aten_ir_make_fx`，接受多个参数和返回一个 `ATenExportArtifact` 对象
def _export_to_aten_ir_make_fx(
    mod: torch.nn.Module,
    fake_args,  # 用于模拟参数的占位符
    fake_kwargs,  # 用于模拟关键字参数的占位符
    fake_params_buffers,  # 用于模拟模块参数和缓冲区的占位符
    constant_attrs: ConstantAttrMap,  # 常量属性映射，用于映射常量属性
) -> ATenExportArtifact:
    
    # 定义一个上下文管理器 `_compiling_state_context`，用于管理编译状态
    @contextmanager
    def _compiling_state_context():
        old_value = torch.compiler._is_compiling_flag  # 保存当前编译标志的旧值
        try:
            torch.compiler._is_compiling_flag = True  # 设置编译标志为 True
            yield  # 执行操作
        finally:
            torch.compiler._is_compiling_flag = old_value  # 恢复编译标志的旧值

    # 这个 `_reparametrize_module` 函数确保输入和模块参数/缓冲区具有相同的 fake_mode，
    # 否则 `aot_export_module` 将因为看到混合的 fake_mode 而报错。
    # 我们希望 `aot_export_module` 在 dynamo 中使用 fake_tensor 模式，以保持管道的易于理解性。
    with torch.nn.utils.stateless._reparametrize_module(
        mod,
        fake_params_buffers,
        tie_weights=True,
        strict=True,
        stack_weights=True,
    ), _ignore_backend_decomps(), _compiling_state_context():  # type: ignore[attr-defined]
        # 提取模块中的命名参数，不移除重复项
        named_parameters = dict(mod.named_parameters(remove_duplicate=False))
        # 计算命名参数的数量
        param_len = len(named_parameters)
        # 提取模块中的命名缓冲区，不移除重复项
        named_buffers = dict(mod.named_buffers(remove_duplicate=False))
        # 计算命名缓冲区的数量
        buffer_len = len(named_buffers)

        # 将命名参数和命名缓冲区合并成一个字典
        params_and_buffers = {
            **dict(named_parameters),
            **dict(named_buffers),
        }
        # 将合并后的参数和缓冲区扁平化为列表
        params_and_buffers_flat, params_spec = pytree.tree_flatten(params_and_buffers)
        params_and_buffers_flat = tuple(params_and_buffers_flat)
        # 计算合并后参数和缓冲区的总数
        params_len = len(params_and_buffers)

        # 如果 fake_kwargs 为 None，则初始化为空字典
        fake_kwargs = fake_kwargs or {}

        # 创建功能调用对象
        functional_call = create_functional_call(
            mod, params_spec, params_len, store_orig_mod=True
        )

        # 初始化完整参数列表
        full_args: List[Any] = []
        full_args.extend(params_and_buffers_flat)

        # 创建扁平化的函数和输出规范
        flat_fn, out_spec = create_tree_flattened_fn(
            functional_call, fake_args, fake_kwargs
        )
        # 将 fake_args 和 fake_kwargs 扁平化为列表并添加到完整参数列表中
        flat_args, in_spec = pytree.tree_flatten((fake_args, fake_kwargs))
        full_args.extend(flat_args)

        # 启用 Python 调度器上下文
        with enable_python_dispatcher():
            # 使用 make_fx 创建功能调用对象的图模块
            gm = make_fx(functional_call, pre_dispatch=True)(*full_args)

        # 如果 mod 是 torch.fx.GraphModule 类型并且具有 "meta" 属性，则更新 gm.meta
        if isinstance(mod, torch.fx.GraphModule) and hasattr(mod, "meta"):
            gm.meta.update(mod.meta)

        # 定义函数，用于创建参数规范
        def make_argument_spec(i, node) -> ArgumentSpec:
            # 如果节点是常量类型，则直接返回常量参数对象
            if isinstance(node, (int, bool, float, type(None))):
                return ConstantArgument(name="", value=node)

            # 确保节点具有 'val' 元数据字段
            assert (
                "val" in node.meta
            ), f"{node} is not a constant or a node with a 'val' metadata field"
            val = node.meta["val"]
            # 根据不同的值类型返回相应的参数对象
            if isinstance(val, FakeTensor):
                return TensorArgument(name=node.name)
            elif isinstance(val, torch.SymInt):
                return SymIntArgument(name=node.name)
            elif isinstance(val, torch.ScriptObject):
                return CustomObjArgument(name=node.name, class_fqn=val._type().qualified_name())  # type: ignore[attr-defined]
            elif isinstance(val, FakeScriptObject):
                return CustomObjArgument(
                    name=node.name, class_fqn=val.script_class_name, fake_val=val
                )
            elif isinstance(val, (int, bool, str, float, type(None))):
                return ConstantArgument(name=node.name, value=val)
            else:
                # 抛出异常，表示遇到不支持的对象类型
                raise AssertionError(
                    f"Encountered an unsupported object of type {type(val)} "
                    f"while writing the metadata for exported program"
                )

    # 扁平化 fake_args 和 fake_kwargs 并获取叶子节点列表
    flat_args = pytree.tree_leaves((fake_args, fake_kwargs))
    # 初始化索引值为 0
    index = 0
    # 遍历计算图中的所有节点
    for node in gm.graph.nodes:
        # 检查节点是否为占位符
        if node.op == "placeholder":
            # 如果索引超过参数长度，则使用平坦化参数列表中的参数
            if index >= params_len:
                user_arg = flat_args[index - params_len]
                # 如果用户参数不是 torch.Tensor 类型，则将其存储在节点的元数据中
                if not isinstance(user_arg, torch.Tensor):
                    node.meta["val"] = user_arg
            # 增加索引计数
            index += 1

    # 定义内部函数 _graph_input_names，用于返回计算图中所有占位符节点的名称列表
    def _graph_input_names(gm):
        return [node.name for node in gm.graph.find_nodes(op="placeholder")]

    # 定义内部函数 _graph_output_names，用于返回计算图中输出节点的名称列表
    def _graph_output_names(gm):
        # 获取计算图中的输出节点
        output_node = next(iter(reversed(gm.graph.nodes)))
        # 断言输出节点为 "output"，并且其参数长度为 1
        assert output_node.op == "output" and len(output_node.args) == 1
        # 获取返回值节点的参数，并返回其名称列表
        return_args = output_node.args[0]
        return [getattr(return_arg, "name", None) for return_arg in return_args]

    # 获取计算图中的输入和输出节点的名称列表
    input_names = _graph_input_names(gm)
    output_names = _graph_output_names(gm)

    # 将输入和输出的规范转换为具体的规范
    input_specs, output_specs = _sig_to_specs(
        user_inputs=set(input_names[params_len:]),  # 用户输入参数的名称集合
        inputs_to_parameters=dict(zip(input_names[0:param_len], named_parameters)),  # 输入到参数的映射字典
        inputs_to_buffers=dict(zip(input_names[param_len : param_len + buffer_len], named_buffers)),  # 输入到缓冲区的映射字典
        user_outputs=set(output_names),  # 用户输出参数的名称集合
        buffer_mutations={},  # 缓冲区变异的空字典
        user_input_mutations={},  # 用户输入变异的空字典
        grad_params={},  # 梯度参数的空字典
        grad_user_inputs={},  # 梯度用户输入的空字典
        loss_output=None,  # 损失输出为空
        inputs=[
            make_argument_spec(i, node)  # 生成输入参数规范的列表
            for i, node in enumerate(gm.graph.nodes)  # 遍历计算图中的所有节点
            if node.op == "placeholder"  # 仅选择占位符节点
        ],
        outputs=[
            make_argument_spec(i, node)  # 生成输出参数规范的列表
            for i, node in enumerate(
                pytree.tree_leaves(next(iter(reversed(gm.graph.nodes))).args)  # 计算输出节点的树叶
            )
        ],
        input_tokens=[],  # 输入标记为空列表
        output_tokens=[],  # 输出标记为空列表
    )

    # 创建 ExportGraphSignature 对象，包含输入和输出的规范
    export_graph_signature = ExportGraphSignature(
        input_specs=input_specs, output_specs=output_specs
    )

    # 导入 detect_fake_mode 函数，检测是否处于虚假模式
    from torch._guards import detect_fake_mode

    # 检测是否处于虚假模式，并返回结果
    fake_mode = detect_fake_mode(flat_args)

    # 导入 _dynamo_config 模块的 config 对象
    from torch._dynamo import config as _dynamo_config

    # 如果不禁用运行时断言，则执行以下代码块
    if not _dynamo_config.do_not_emit_runtime_asserts:
        # 定义 stack_trace 字符串，用于调试信息
        stack_trace = (
            'File "torch/fx/passes/runtime_assert.py", line 24, '
            "in insert_deferred_runtime_asserts"
        )
        # 设置节点元数据钩子，为节点添加元数据信息
        with _set_node_metadata_hook(
            gm, functools.partial(_node_metadata_hook, stack_trace=stack_trace)
        ):
            # 插入延迟运行时断言到计算图中
            insert_deferred_runtime_asserts(
                gm,
                fake_mode.shape_env,  # 虚假模式的形状环境
                f"exported program: {first_call_function_nn_module_stack(gm.graph)}",  # 导出的程序信息
                export=True,  # 导出标志位
            )

    # 移除所有占位符/输入节点的 nn_module_stack 和 stack_trace 元数据信息
    # 遍历所有模块 `_mod` 在 `gm` 中
    for _mod in gm.modules():
        # 如果 `_mod` 不是 `torch.fx.GraphModule` 类型，则跳过当前循环
        if not isinstance(_mod, torch.fx.GraphModule):
            continue
        # 遍历当前 `_mod` 的图中的每个节点 `node`
        for node in _mod.graph.nodes:
            # 如果节点 `node` 的操作类型是 "placeholder" 或 "output"
            if node.op in ["placeholder", "output"]:
                # 移除节点 `node` 的元数据中的 "nn_module_stack" 和 "stack_trace" 键
                node.meta.pop("nn_module_stack", None)
                node.meta.pop("stack_trace", None)

    # 重写脚本对象的元数据，并更新到 `constants`
    constants = rewrite_script_object_meta(gm)
    # 将常量提升为模块级别，并更新到 `constants`
    constants.update(lift_constants_pass(gm, export_graph_signature, constant_attrs))

    # 保留需要梯度的参数的传递过程
    _preserve_requires_grad_pass(
        gm, export_graph_signature, fake_params_buffers, constants, flat_args
    )

    # 为占位符节点进行命名美化的传递过程
    placeholder_naming_pass(
        gm,
        export_graph_signature,
        mod,
        fake_args,
        fake_kwargs,
        fake_params_buffers,
        constants,
    )

    # 返回 ATenExportArtifact 对象，其中包含重写后的图 `gm`、导出的图签名 `export_graph_signature` 和常量 `constants`
    return ATenExportArtifact(
        gm,
        export_graph_signature,
        constants,
    )
# 定义一个函数 _non_strict_export，用于非严格模式导出模型
def _non_strict_export(
    # 参数 mod: 导出的 PyTorch 模型
    mod: torch.nn.Module,
    # 参数 args: 位置参数的元组
    args: Tuple[Any, ...],
    # 参数 kwargs: 关键字参数的字典
    kwargs: Dict[str, Any],
    # 参数 dynamic_shapes: 动态形状的字典、元组或列表，可选
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any], List[Any]]],
    # 参数 preserve_module_call_signature: 保留模块调用签名的元组
    preserve_module_call_signature: Tuple[str, ...],
    # 参数 pre_dispatch: 是否预分发
    pre_dispatch: bool,
    # 参数 original_state_dict: 原始状态字典的字典
    original_state_dict: Dict[str, Any],
    # 参数 orig_in_spec: 原始输入规范的 TreeSpec 对象
    orig_in_spec: TreeSpec,
    # 参数 _allow_complex_guards_as_runtime_asserts: 允许复杂保护作为运行时断言
    _allow_complex_guards_as_runtime_asserts: bool,
    # 参数 _disable_forced_specializations: 禁用强制特化的可选布尔值
    _disable_forced_specializations: Optional[bool],
    # 参数 _is_torch_jit_trace: 是否是 PyTorch JIT 跟踪
    _is_torch_jit_trace: bool,
) -> ExportArtifact:
    # 初始化输出规范为 None
    out_spec: Optional[TreeSpec] = None

    # 初始化模块调用规范的字典
    module_call_specs: Dict[str, Dict[str, pytree.TreeSpec]] = {}
    def _tuplify_outputs(aot_export):
        # 定义内部函数 _aot_export_non_strict，用于对模型进行导出
        def _aot_export_non_strict(mod, args, kwargs=None, **flags):
            kwargs = kwargs or {}

            # 定义包装器类 Wrapper，继承自 torch.nn.Module
            class Wrapper(torch.nn.Module):
                def __init__(self, mod):
                    super().__init__()
                    self._export_root = mod

                # 重写 forward 方法，处理模型推理过程
                def forward(self, *args, **kwargs):
                    nonlocal out_spec
                    # 根据模型类型选择合适的解释器进行推理
                    if isinstance(self._export_root, torch.fx.GraphModule):
                        with torch.fx.traceback.preserve_node_meta():
                            tree_out = torch.fx.Interpreter(self._export_root).run(
                                *args, **kwargs
                            )
                    else:
                        tree_out = self._export_root(*args, **kwargs)
                    # 对推理结果进行扁平化处理，并保存输出规范
                    flat_outs, out_spec = pytree.tree_flatten(tree_out)
                    return tuple(flat_outs)

            # 使用模型创建 Wrapper 实例
            wrapped_mod = Wrapper(mod)
            # 将 _export_root 对象路径添加到保留的调用签名中，以便包装器模块正确填充输入/输出规范
            new_preserved_call_signatures = [
                "_export_root." + i for i in preserve_module_call_signature
            ]
            # 使用 _wrap_submodules 函数对包装器模块进行处理
            with _wrap_submodules(
                wrapped_mod, new_preserved_call_signatures, module_call_specs
            ):
                # 调用 aot_export 函数导出模型，并获取导出的图和签名
                gm, sig = aot_export(wrapped_mod, args, kwargs=kwargs, **flags)
                log.debug("Exported program from AOTAutograd:\n%s", gm)

            # 将签名中的参数、缓冲区等对象的根路径去除
            sig.parameters = pytree.tree_map(_strip_root, sig.parameters)
            sig.buffers = pytree.tree_map(_strip_root, sig.buffers)
            sig.inputs_to_buffers = pytree.tree_map(_strip_root, sig.inputs_to_buffers)
            sig.inputs_to_parameters = pytree.tree_map(
                _strip_root, sig.inputs_to_parameters
            )
            sig.buffers_to_mutate = pytree.tree_map(_strip_root, sig.buffers_to_mutate)

            # 遍历图中的节点，修正 nn_module_stack 中的键名，以便正确映射模型组件
            for node in gm.graph.nodes:
                if "nn_module_stack" in node.meta:
                    nn_module_stack = node.meta["nn_module_stack"]
                    node.meta["nn_module_stack"] = {
                        _fixup_key(key): val
                        for key, val in pytree.tree_map(
                            _strip_root, nn_module_stack
                        ).items()
                    }

            # 返回导出的图和签名
            return gm, sig

        # 返回 _aot_export_non_strict 函数对象
        return _aot_export_non_strict

    # 使用 make_fake_inputs 函数创建虚假模式、参数和关键字参数，以及其他输入数据
    (
        fake_mode,
        fake_args,
        fake_kwargs,
        equalities_inputs,
        original_signature,
    ) = make_fake_inputs(
        mod,
        args,
        kwargs,
        dynamic_shapes,
        _is_torch_jit_trace=_is_torch_jit_trace,
        _allow_complex_guards_as_runtime_asserts=_allow_complex_guards_as_runtime_asserts,  # 用于形状环境初始化
    )

    # 使用 make_fake_params_buffers 函数创建虚假参数和缓冲区
    fake_params_buffers = make_fake_params_buffers(fake_mode, _get_params_buffers(mod))
    # 进入假模式上下文
    with fake_mode:
        # 在假模式下，使用给定参数和对象修改脚本对象
        with _fakify_script_objects(mod, fake_args, fake_kwargs, fake_mode) as (
            patched_mod,
            new_fake_args,
            new_fake_kwargs,
            new_fake_constant_attrs,
            map_fake_to_real,
        ):
            # 导出到 ATen IR
            aten_export_artifact = _export_to_aten_ir(
                patched_mod,
                new_fake_args,
                new_fake_kwargs,
                fake_params_buffers,
                new_fake_constant_attrs,
                pre_dispatch=pre_dispatch,
                transform=_tuplify_outputs,
                _is_torch_jit_trace=_is_torch_jit_trace,
            )
            # 将常量中的假脚本对象映射回真实对象
            aten_export_artifact.constants = {
                fqn: map_fake_to_real[obj] if isinstance(obj, FakeScriptObject) else obj
                for fqn, obj in aten_export_artifact.constants.items()
            }

    try:
        # 生成保护条件并解决约束
        produce_guards_and_solve_constraints(
            fake_mode,
            aten_export_artifact.gm,
            dynamic_shapes,
            equalities_inputs,
            original_signature,
            _disable_forced_specializations=_disable_forced_specializations,
            _is_torch_jit_trace=_is_torch_jit_trace,
        )
    except (ConstraintViolationError, ValueRangeError) as e:
        # 如果出现约束违反或值范围错误，抛出用户错误
        raise UserError(UserErrorType.CONSTRAINT_VIOLATION, str(e))  # noqa: B904

    # 重写非持久性缓冲区
    _rewrite_non_persistent_buffers(
        mod, aten_export_artifact.sig, aten_export_artifact.constants
    )

    # 确保输出规范不为空
    assert out_spec is not None
    # 返回导出的艺术品对象
    return ExportArtifact(
        aten=aten_export_artifact,
        out_spec=out_spec,
        fake_mode=fake_mode,
        module_call_specs=module_call_specs,
    )
# TODO (tmanlaibaatar) We need to preserve aten.to here somehow
@_log_export_wrapper
@_disable_prexisiting_fake_mode
def _export_for_training(
    mod: torch.nn.Module,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any], List[Any]]] = None,
    *,
    strict: bool = True,
    preserve_module_call_signature: Tuple[str, ...] = (),
) -> ExportedProgram:
    # 如果 strict 不为 True，则抛出 NotImplementedError
    if not strict:
        raise NotImplementedError("Non-strict export for training is not supported yet")

    # 如果 args 不是 tuple 类型，抛出 UserError 异常
    if not isinstance(args, tuple):
        raise UserError(
            UserErrorType.INVALID_INPUT,
            f"Expecting `args` to be a tuple of example positional inputs, got {type(args)}",
        )

    # 设置全局变量 _EXPORT_MODULE_HIERARCHY 为模块层次结构
    global _EXPORT_MODULE_HIERARCHY
    _EXPORT_MODULE_HIERARCHY = _get_module_hierarchy(mod)

    # TODO (tmanlaibaatar) setup logging here
    # 如果 kwargs 为 None，则设为空字典
    kwargs = kwargs or {}

    # 如果 dynamic_shapes 是 torch.export.ShapesCollection 类型，调用其 dynamic_shapes 方法
    if isinstance(dynamic_shapes, torch.export.ShapesCollection):
        dynamic_shapes = dynamic_shapes.dynamic_shapes(mod, args, kwargs)

    # 将 args 和 kwargs 扁平化处理，获取原始输入规范
    flat_args, orig_in_spec = pytree.tree_flatten((args, kwargs))

    # 获取模型当前状态的字典表示，保留变量信息
    original_state_dict = mod.state_dict(keep_vars=True)

    # 获取模型前向传播方法的参数名列表
    forward_arg_names = _get_forward_arg_names(mod, args, kwargs)

    # 进行严格导出至 ATen IR
    export_artifact = _strict_export_lower_to_aten_ir(
        mod=mod,
        args=args,
        kwargs=kwargs,
        dynamic_shapes=dynamic_shapes,
        preserve_module_call_signature=preserve_module_call_signature,
        pre_dispatch=False,
        original_state_dict=original_state_dict,
        orig_in_spec=orig_in_spec,
        _allow_complex_guards_as_runtime_asserts=False,
        _disable_forced_specializations=False,
        _is_torch_jit_trace=False,
        lower_to_aten_callback=_export_to_aten_ir_make_fx,
    )

    # 为了提升可读性，将 export_artifact 中的几个属性解构出来
    gm = export_artifact.aten.gm  # 获取导出后的图形对象
    export_graph_signature = export_artifact.aten.sig  # 获取导出图形的签名
    out_spec = export_artifact.out_spec  # 获取输出规范
    fake_mode = export_artifact.fake_mode  # 获取虚拟模式信息
    module_call_specs = export_artifact.module_call_specs  # 获取模块调用规范

    # TODO(tmanlaibaatar) Not sure why i need this, but need to re-normalize it.
    # 遍历导出后的图形节点，调整 nn_module_stack 的元数据
    for node in gm.graph.nodes:
        # 对于非占位符和输出节点，遍历其 nn_module_stack
        if node.op not in ["placeholder", "output"]:
            for key, (fqn, mod_cls) in node.meta["nn_module_stack"].items():
                # 如果 mod_cls 是类的类型，则更新其模块类的完整限定名
                if isinstance(mod_cls, type):
                    node.meta["nn_module_stack"][key] = (
                        fqn,
                        mod_cls.__module__ + "." + mod_cls.__qualname__,
                    )

    # 添加前向传播参数的元数据至导出图形元数据
    gm.meta["forward_arg_names"] = forward_arg_names

    # 在 AOT 导出中更新未支持的符号，因此这里序列化它们而不是在 dynamo 内部进行
    gm.meta["inline_constraints"] = {
        k: v
        for k, v in fake_mode.shape_env.var_to_range.items()
        if free_unbacked_symbols(k)
    }
    # 找到第一个用户输入的输入规格在 export_graph_signature.input_specs 中的索引
    num_lifted = next(
        (
            i
            for i, s in enumerate(export_graph_signature.input_specs)
            if s.kind == InputKind.USER_INPUT
        ),
        len(export_graph_signature.input_specs),  # 如果没有找到用户输入规格，返回 input_specs 的长度
    )
    # 合并参数 args 和 kwargs
    combined_args = _combine_args(mod, args, kwargs)
    # 创建约束条件
    range_constraints = make_constraints(
        fake_mode,
        gm,
        combined_args,
        dynamic_shapes,
        num_lifted,
    )

    # 创建模块调用签名字典
    module_call_signatures = {}
    for fqn, specs in module_call_specs.items():
        mod_fqn = _strip_root(fqn) if not strict else fqn
        module_call_signatures[mod_fqn] = ModuleCallSignature(
            inputs=[], outputs=[], **specs
        )

    # 确保 out_spec 不为 None
    assert out_spec is not None

    # 验证神经网络模块的堆栈
    _verify_nn_module_stack(gm)
    # 验证堆栈跟踪
    _verify_stack_trace(gm)
    # 验证占位符名称
    _verify_placeholder_names(gm, export_graph_signature)

    # 确保 _EXPORT_MODULE_HIERARCHY 不为 None
    assert _EXPORT_MODULE_HIERARCHY is not None
    # 导入训练 IR 校验器
    from torch._export.verifier import TrainingIRVerifier

    # 创建 ExportedProgram 对象
    exported_program = ExportedProgram(
        root=gm,  # 根模块
        graph=gm.graph,  # 图结构
        graph_signature=export_graph_signature,  # 导出图的签名
        state_dict=original_state_dict,  # 原始状态字典
        range_constraints=range_constraints,  # 范围约束
        module_call_graph=_make_module_call_graph(
            _EXPORT_MODULE_HIERARCHY,  # 模块调用层级
            orig_in_spec,
            out_spec,
            module_call_signatures,
        ),  # 创建模块调用图
        example_inputs=(args, kwargs),  # 示例输入
        constants=export_artifact.aten.constants,  # 导出的常量
        verifier=TrainingIRVerifier,  # 训练 IR 校验器
    )

    # 返回 ExportedProgram 对象作为输出
    return exported_program
# 使用装饰器 @_log_export_wrapper 和 @_disable_prexisiting_fake_mode 对 _export 函数进行修饰
@_log_export_wrapper
@_disable_prexisiting_fake_mode
# 定义一个函数 _export，接受以下参数：
# - mod: torch.nn.Module 类型，表示一个 PyTorch 模型
# - args: Tuple[Any, ...]，表示位置参数的元组
# - kwargs: Optional[Dict[str, Any]]，表示关键字参数的可选字典，默认为 None
# - dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any], List[Any]]]，表示动态形状的可选参数，可以是字典、元组或列表
# - strict: bool 类型的关键字参数，默认为 True，表示是否启用严格模式
# - preserve_module_call_signature: Tuple[str, ...]，表示要保留的模块调用签名的元组，默认为空元组
# - pre_dispatch: bool 类型的关键字参数，默认为 False，表示是否预先调度
# - _allow_complex_guards_as_runtime_asserts: bool 类型的关键字参数，默认为 False，表示是否允许复杂的守卫作为运行时断言
# - _disable_forced_specializations: Optional[bool]，表示是否禁用强制特化的可选参数，默认为 False
# - _is_torch_jit_trace: bool 类型的关键字参数，默认为 False，表示是否为 Torch JIT 跟踪
# 返回值为 ExportedProgram 类型的对象，表示一个导出的程序
def _export(
    mod: torch.nn.Module,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    dynamic_shapes: Optional[Union[Dict[str, Any], Tuple[Any], List[Any]]] = None,
    *,
    strict: bool = True,
    preserve_module_call_signature: Tuple[str, ...] = (),
    pre_dispatch: bool = False,
    _allow_complex_guards_as_runtime_asserts: bool = False,
    _disable_forced_specializations: Optional[bool] = False,
    _is_torch_jit_trace: bool = False,
) -> ExportedProgram:
    """
    Traces either an nn.Module's forward function or just a callable with PyTorch
    operations inside and produce a ExportedProgram.
    """
    # 定义函数签名，用于对 nn.Module 进行跟踪
    Args:
        f: the `nn.Module` to trace.  # 要跟踪的 nn.Module 对象

        args: example positional inputs.  # 示例的位置输入参数

        kwargs: optional example keyword inputs.  # 可选的示例关键字输入参数

        dynamic_shapes:
         An optional argument where the type should either be:
         1) a dict from argument names of ``f`` to their dynamic shape specifications,
         2) a tuple that specifies dynamic shape specifications for each input in original order.
         If you are specifying dynamism on keyword args, you will need to pass them in the order that
         is defined in the original function signature.
         # 动态形状的可选参数，其类型应为：
         # 1) 从 ``f`` 的参数名到其动态形状规范的字典，
         # 2) 指定原始顺序中每个输入的动态形状规范的元组。
         # 如果在关键字参数上指定动态性，则需要按照原始函数签名中定义的顺序传递它们。

        preserve_module_call_signature: A list of submodule paths for which the original
            calling conventions are preserved as metadata.
        # 保留子模块调用签名的列表，以元数据的形式保留原始调用约定。

        _allow_complex_guards_as_runtime_asserts:
         With the current dynamic shapes language for dims and derived dims, we can run into constraints
         that are not expressible with the language. For example, flattening a matrix and adding to a vector,
         both fully dynamic (i.e. x.reshape([-1]) + y) emits a guard s0 * s1 = s2, which is not expressible.
         By default, we either raise a constraint violation error or specialize to static values.
         If this flag is set to True, we avoid erroring out and instead allow complex constraints to exist as runtime
         assertions in the graph. The sympy interpreter (torch/utils/_sympy/interp.py) will produce the math ops
         required to compute and assert the value of the guard (e.g. sym_size_int, eq, _assert_scalar).
         Additionally, if TORCH_DYNAMO_DO_NOT_EMIT_RUNTIME_ASSERTS=1 is specified, we will allow complex constraints
         while not emitting runtime asserts, returning a cleaner graph with lesser guarantees around dynamic shapes.
         # 对于当前的动态形状语言和衍生维度，我们可能会遇到无法用该语言表达的约束条件。
         # 例如，将矩阵展平并加到向量上，这两者都是完全动态的（即 x.reshape([-1]) + y）会产生一个约束条件 s0 * s1 = s2，这是无法表达的。
         # 默认情况下，我们要么引发约束违规错误，要么专门化为静态值。
         # 如果设置了此标志为 True，则避免错误并允许复杂约束作为图中的运行时断言存在。
         # sympy 解释器（torch/utils/_sympy/interp.py）将生成所需的数学运算，以计算和断言约束条件的值（例如 sym_size_int，eq，_assert_scalar）。
         # 此外，如果指定了 TORCH_DYNAMO_DO_NOT_EMIT_RUNTIME_ASSERTS=1，则允许复杂约束，而不会发出运行时断言，
         # 返回一个更干净的图，对动态形状的保证较少。

        _disable_forced_specializations:
         Similar to _allow_complex_guards_as_runtime_asserts, but only avoids specializing to static values if set to True.
         For complex guards that don't specialize, this flag doesn't have any effect. Ideally this would be subsumed by
         _allow_complex_guards_as_runtime_asserts, but this handles one additional case: single-variable equalities where
         the symbol is solvable for a concrete value (e.g. Eq(s0 // 4, 400) -> s0 = 1600). If set to True, this flag will
         avoid specializations. Direct equalities (e.g. s0 = 4), will still specialize.
         # 类似于 _allow_complex_guards_as_runtime_asserts，但只有在设置为 True 时避免专门化为静态值。
         # 对于不专门化的复杂约束，此标志无效。理想情况下，这应该由 _allow_complex_guards_as_runtime_asserts 覆盖，
         # 但这处理了一个额外的情况：单变量等式，其中符号可解为具体值（例如 Eq(s0 // 4, 400) -> s0 = 1600）。
         # 如果设置为 True，则此标志将避免专门化。直接等式（例如 s0 = 4）仍将专门化。
    # 检查输入参数是否为元组，如果不是则抛出用户错误异常
    if not isinstance(args, tuple):
        raise UserError(
            UserErrorType.INVALID_INPUT,
            f"Expecting `args` to be a tuple of example positional inputs, got {type(args)}",
        )

    # 如果启用了强制特化禁用并且设置了严格模式，则抛出用户错误异常
    if _disable_forced_specializations and strict:
        raise UserError(
            UserErrorType.INVALID_INPUT,
            "_disable_forced_specializations can be only be specified in non-strict mode.",
        )

    # 获取模块的层次结构
    global _EXPORT_FLAGS, _EXPORT_MODULE_HIERARCHY
    _EXPORT_MODULE_HIERARCHY = _get_module_hierarchy(mod)

    # 设置导出标志
    flags = set()
    flags.add("strict" if strict else "non_strict")
    flags.add("pre_dispatch" if pre_dispatch else "aot_dispatch")
    # 记录导出使用情况的日志事件
    log_export_usage(event="export.enter", flags=flags)
    _EXPORT_FLAGS = flags

    # 初始化关键字参数
    kwargs = kwargs or {}

    # 如果动态形状是 torch.export.ShapesCollection 类型，则根据模块、参数和关键字参数获取动态形状
    if isinstance(dynamic_shapes, torch.export.ShapesCollection):
        dynamic_shapes = dynamic_shapes.dynamic_shapes(mod, args, kwargs)

    # 将参数 args 和 kwargs 展平成一维数组，并保留原始输入规范
    flat_args, orig_in_spec = pytree.tree_flatten((args, kwargs))

    # 获取模块的原始状态字典（包含变量）
    original_state_dict = mod.state_dict(keep_vars=True)

    # 如果不是 Torch JIT 追踪，则获取前向传递参数的名称
    if not _is_torch_jit_trace:
        forward_arg_names = _get_forward_arg_names(mod, args, kwargs)
    else:
        forward_arg_names = None

    # 根据追踪严格性调用相应的导出函数
    export_func = _strict_export if strict else _non_strict_export

    # 执行导出函数并获取导出的结果
    export_artifact = export_func(
        mod,
        args,
        kwargs,
        dynamic_shapes,
        preserve_module_call_signature,
        pre_dispatch,
        original_state_dict,
        orig_in_spec,
        _allow_complex_guards_as_runtime_asserts,
        _disable_forced_specializations,
        _is_torch_jit_trace,
    )

    # 解析导出结果以便获取所需的内容
    gm = export_artifact.aten.gm
    export_graph_signature = export_artifact.aten.sig
    out_spec = export_artifact.out_spec
    fake_mode = export_artifact.fake_mode
    module_call_specs = export_artifact.module_call_specs

    # 将前向传递参数的名称添加到 gm 的元数据中
    gm.meta["forward_arg_names"] = forward_arg_names

    # 在 aot_export 中更新非支持的 symint 符号，因此在此处而不是 dynamo 中序列化它们
    gm.meta["inline_constraints"] = {
        k: v
        for k, v in fake_mode.shape_env.var_to_range.items()
        if free_unbacked_symbols(k)
    }

    # 获取导出图的输入规范中第一个用户输入参数之前的索引，用于限制解析变量
    num_lifted = next(
        (
            i
            for i, s in enumerate(export_graph_signature.input_specs)
            if s.kind == InputKind.USER_INPUT
        ),
        len(export_graph_signature.input_specs),
    )

    # 组合参数以创建约束条件
    combined_args = _combine_args(
        mod, args, kwargs, _is_torch_jit_trace=_is_torch_jit_trace
    )

    # 生成约束条件
    range_constraints = make_constraints(
        fake_mode,
        gm,
        combined_args,
        dynamic_shapes,
        num_lifted,
    )
    # 如果 strict 参数为真，则在子图中添加运行时断言条件
    if strict:
        _add_runtime_assertions_to_cond_in_subgraph(
            range_constraints,
            gm,
            fake_mode,
        )

    # 创建模块调用签名的字典
    module_call_signatures = {}
    # 遍历模块调用规格字典中的每一项
    for fqn, specs in module_call_specs.items():
        # 如果 strict 参数为假，则去除根目录后的全限定名
        mod_fqn = _strip_root(fqn) if not strict else fqn
        # 将模块调用签名加入字典中
        module_call_signatures[mod_fqn] = ModuleCallSignature(
            inputs=[], outputs=[], **specs
        )

    # 如果保留模块调用签名的长度大于 0
    if len(preserve_module_call_signature) > 0:
        # 如果 strict 参数为假，则重写图中的节点
        if not strict:
            _rewrite_node(gm)
        # 运行 CollectTracepointsPass 收集迹点，导出图的签名
        res = CollectTracepointsPass(module_call_signatures, export_graph_signature)(gm)
        # 确保结果不为空
        assert res is not None
        # 更新图模块
        gm = res.graph_module

    # 确保输出规格不为空
    assert out_spec is not None

    # 验证神经网络模块的堆栈
    _verify_nn_module_stack(gm)
    # 验证堆栈跟踪
    _verify_stack_trace(gm)
    # 如果不是 Torch JIT 跟踪，则验证占位符名称
    if not _is_torch_jit_trace:
        _verify_placeholder_names(gm, export_graph_signature)

    # 确保导出模块层次结构不为空
    assert _EXPORT_MODULE_HIERARCHY is not None
    # 创建导出程序对象
    exported_program = ExportedProgram(
        root=gm,
        graph=gm.graph,
        graph_signature=export_graph_signature,
        state_dict=original_state_dict,
        range_constraints=range_constraints,
        module_call_graph=_make_module_call_graph(
            _EXPORT_MODULE_HIERARCHY,
            orig_in_spec,
            out_spec,
            module_call_signatures,
        ),
        example_inputs=(args, kwargs),
        constants=export_artifact.aten.constants,
    )

    # 返回导出的程序对象
    return exported_program
```