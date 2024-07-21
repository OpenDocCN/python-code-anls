# `.\pytorch\torch\_dynamo\variables\higher_order_ops.py`

```py
# 忽略类型检查错误，针对 mypy 工具的声明
# Import necessary modules and symbols
import contextlib                  # 提供支持上下文管理的工具
import functools                  # 提供创建装饰器的工具
import itertools                  # 提供用于迭代操作的工具
import logging                    # 提供日志记录功能
import types                      # 提供操作 Python 类型和对象的工具

from typing import Dict, List, Optional, TYPE_CHECKING   # 引入类型提示

import torch._C                   # 导入 PyTorch 的 C++ 扩展模块
import torch.fx                   # 导入 PyTorch FX 模块
import torch.nn                   # 导入 PyTorch 神经网络模块
import torch.onnx.operators       # 导入 PyTorch ONNX 运算符模块
from torch._dynamo.utils import get_fake_value   # 从内部模块导入函数
from torch._dynamo.variables import ConstantVariable   # 从内部模块导入类
from torch._dynamo.variables.base import VariableTracker   # 从内部模块导入类
from torch._dynamo.variables.builtin import BuiltinVariable   # 从内部模块导入类
from torch._dynamo.variables.functions import UserFunctionVariable   # 从内部模块导入类
from torch._dynamo.variables.tensor import SymNodeVariable   # 从内部模块导入类
from torch._guards import Source   # 从内部模块导入类
from torch._ops import HigherOrderOperator   # 从内部模块导入类
from torch.fx.passes.shape_prop import _extract_tensor_metadata   # 从内部模块导入函数
from torch.utils import _pytree as pytree   # 导入 PyTorch 内部的 pytree 模块
from .. import variables   # 导入当前包中的 variables 模块

from ..exc import UncapturedHigherOrderOpError, unimplemented, Unsupported   # 导入自定义异常类和函数
from ..source import AttrSource   # 从源码模块导入类
from ..utils import proxy_args_kwargs   # 从工具模块导入函数
from .dicts import ConstDictVariable   # 从当前包的 dicts 模块导入类
from .lazy import LazyVariableTracker   # 从当前包的 lazy 模块导入类
from .lists import ListVariable, TupleVariable   # 从当前包的 lists 模块导入类

if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator   # 类型检查下引入符号转换模块的类

log = logging.getLogger(__name__)   # 获取当前模块的日志记录器对象


def raise_hard_error_if_graph_break(reason):
    # 装饰器函数，用于捕获异常并抛出特定错误信息
    def deco(fn):
        @functools.wraps(fn)
        def graph_break_as_hard_error(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Unsupported as e:
                msg = " Scroll up to find out what causes the graph break."
                raise UncapturedHigherOrderOpError(reason + msg) from e

        return graph_break_as_hard_error

    return deco


@contextlib.contextmanager
def dynamo_enable_grad(tx, enable=True):
    # 上下文管理器函数，用于临时开启或关闭梯度计算
    from . import GradModeVariable   # 导入当前模块的 GradModeVariable 类

    org_value = torch.is_grad_enabled()   # 获取当前梯度计算是否启用的状态
    try:
        GradModeVariable.create(tx, enable, initialized=True)   # 创建梯度模式变量
        yield   # 执行上下文中的代码块
    finally:
        GradModeVariable.create(tx, org_value, initialized=True)   # 恢复原始梯度计算状态


def only_consist_of(var, types, allow_none=False):
    # 检查变量是否完全由指定类型组成的函数
    if isinstance(var, types):
        return True
    if allow_none and var.is_python_constant() and var.as_python_constant() is None:
        return True
    if isinstance(var, (TupleVariable, ListVariable)):
        return all(only_consist_of(item, types, allow_none) for item in var.items)
    if isinstance(var, ConstDictVariable):
        return all(
            only_consist_of(item, types, allow_none) for item in var.items.values()
        )
    return False


# A more read-able syntax sugar for creating a UserFunctionVariable for f
# and run call_function on it. Make it return a function to preserve the calling
# convention of the original f.
def _make_inlined(tx, f):
    # 创建一个 UserFunctionVariable 对象来代表 f，并调用其 call_function 方法
    assert callable(f), "Expect f to be a python callable."

    def inline_call(*args, **kwargs):
        return UserFunctionVariable(f).call_function(tx, args, kwargs)

    return inline_call


def _call_function_and_unflatten_output(
    tx, fn, args, kwargs, flat_example_value, ret_treespec
):
    # 从.builder模块中导入wrap_fx_proxy函数
    from .builder import wrap_fx_proxy

    # 将调用存储为一个变量
    flat_variable = wrap_fx_proxy(
        tx=tx,  # 传递tx参数给wrap_fx_proxy函数
        proxy=tx.output.create_proxy(  # 调用tx.output.create_proxy方法创建代理对象
            "call_function",  # 代理类型为"call_function"
            fn,  # 传递fn参数给create_proxy，表示函数名
            args=args,  # 传递args参数给create_proxy，表示位置参数
            kwargs=kwargs,  # 传递kwargs参数给create_proxy，表示关键字参数
        ),
        example_value=flat_example_value,  # 传递flat_example_value参数给wrap_fx_proxy函数
    )

    # 将变量转换回列表形式（之前由speculate_subgraph函数转换为元组），以符合pytree API的类型要求
    flat_list_variable = BuiltinVariable(list).call_function(tx, [flat_variable], {})
    
    # 如果ret_treespec为真，则使用_make_inlined函数和pytree.tree_unflatten方法来内联处理变量
    # 否则直接返回flat_variable
    return (
        _make_inlined(tx, pytree.tree_unflatten)(flat_list_variable, ret_treespec)
        if ret_treespec
        else flat_variable
    )
# 确保输入张量与输出张量不重复使用同一内存地址的函数断言
def _assert_tensors_nonaliasing(inputs, outputs):
    # 获取输入中所有张量对象的内存地址集合
    input_tensor_ids = {
        id(t) for t in pytree.tree_leaves(inputs) if isinstance(t, torch.Tensor)
    }
    # 获取输出中所有张量对象的内存地址集合
    output_tensor_ids = {
        id(t) for t in pytree.tree_leaves(outputs) if isinstance(t, torch.Tensor)
    }
    # 断言输入张量的内存地址与输出张量的内存地址没有重叠
    assert input_tensor_ids.isdisjoint(
        output_tensor_ids
    ), "inputs to function body cannot alias outputs"


# 检查可支持的可调用参数是否为支持的类型
def _check_supported_callable_arg(tx, func_var: VariableTracker, arg_name):
    # 判断 func_var 是否为可调用对象
    is_callable = (
        BuiltinVariable(callable).call_function(tx, [func_var], {}).as_python_constant()
    )
    if not is_callable:
        # 如果 func_var 不是可调用对象，抛出未实现异常
        unimplemented(f"{arg_name} is of unsupported callable type {str(func_var)}.")


# 验证参数并可能创建图输入
def validate_args_and_maybe_create_graph_inputs(
    sub_args,
    tracer,
    tx,
    set_subgraph_inputs,
    description,
):
    # 导入需要的模块
    from . import AutogradFunctionContextVariable
    from .builder import wrap_fx_proxy_cls

    # 断言 tracer 的父对象不为空
    assert tracer.parent is not None

    # 如果 set_subgraph_inputs 为 "flatten_manual"
    if set_subgraph_inputs == "flatten_manual":
        # 将 sub_args 展平化，并获取展平化后的参数列表和树形规范
        flat_args, tree_spec = _make_inlined(tx, pytree.tree_flatten)(
            ListVariable(sub_args)
        ).unpack_var_sequence(tx)

        # 递归验证展平化后的参数，并可能创建图输入
        flat_inputs = validate_args_and_maybe_create_graph_inputs(
            flat_args.unpack_var_sequence(tx),
            tracer,
            tx,
            set_subgraph_inputs="manual",
            description=description,
        )

        # 将展平化后的参数重新组合成原始树形结构
        return _make_inlined(tx, pytree.tree_unflatten)(
            ListVariable(flat_inputs), tree_spec
        ).unpack_var_sequence(tx)
    else:
        args = []
        # 遍历子参数列表中的每个变量追踪器对象
        for a in sub_args:
            assert isinstance(a, VariableTracker)
            # 如果设置子图输入为"automatic"
            if set_subgraph_inputs == "automatic":
                # 直接将变量追踪器对象添加到参数列表中
                args.append(a)
                continue
            # 如果设置子图输入为"semi_automatic"
            elif set_subgraph_inputs == "semi_automatic":
                # 如果变量追踪器对象是AutogradFunctionContextVariable类型
                if isinstance(a, AutogradFunctionContextVariable):
                    # 创建图输入，并使用其代理节点的名称
                    tracer.create_graph_input(a.as_proxy().node.name)
                # 如果变量追踪器对象可能是fx节点
                elif a.maybe_fx_node() is not None:
                    # 获取fx节点
                    node = a.maybe_fx_node()
                    # 创建图输入，并使用节点的名称
                    new_proxy = tracer.create_graph_input(node.name)
                    # 获取示例值（如果存在）
                    example_value = (
                        node.meta["example_value"]
                        if "example_value" in node.meta
                        else None
                    )
                    # 使用包装的fx代理类创建新的参数对象
                    a = wrap_fx_proxy_cls(
                        target_cls=type(a),
                        tx=tx,
                        proxy=new_proxy,
                        example_value=example_value,
                    )
                # 将处理过的参数对象添加到参数列表中
                args.append(a)
                continue

            # 如果参数对象是Python常量
            if a.is_python_constant():
                # 创建一个名为"const"的图输入
                tracer.create_graph_input("const")
                # 新的参数对象保持不变
                new_arg = a
            # 特殊情况，可能需要删除或合并到下一个情况
            elif isinstance(a, AutogradFunctionContextVariable):
                # 创建图输入，并使用其代理节点的名称
                tracer.create_graph_input(a.as_proxy().node.name)
                # 新的参数对象保持不变
                new_arg = a
            # 如果参数对象可以放入图中
            elif a.maybe_fx_node() is not None:
                # 获取fx节点
                node = a.maybe_fx_node()
                # 创建图输入，并使用节点的名称
                new_proxy = tracer.create_graph_input(node.name)
                # 获取示例值（如果存在）
                example_value = (
                    node.meta["example_value"] if "example_value" in node.meta else None
                )
                # 使用包装的fx代理类创建新的参数对象
                new_arg = wrap_fx_proxy_cls(
                    target_cls=type(a),
                    tx=tx,
                    proxy=new_proxy,
                    example_value=example_value,
                )
            # 如果参数对象无法放入图中
            else:
                # 报告未实现的情况，说明不能处理非张量类型的输入
                unimplemented(
                    f"{description} with body that accepts non-Tensors as input. "
                    f"Got: {a.python_type()}"
                )
            # 将处理过的参数对象添加到参数列表中
            args.append(new_arg)
        # 返回最终的参数列表
        return args
# 定义一个辅助函数，用于确保两个图共享相同的输入签名。例如，在 torch.cond 中，
# 两个分支可能会将不同的张量集合作为输入提升。该函数帮助消除重复的输入，并修改图以接受相同的输入集合。
def _merge_graph_inputs(
    l_graph, l_lifted_freevars, l_name,  # 左图的图形、提升的自由变量集合、名称
    r_graph, r_lifted_freevars, r_name  # 右图的图形、提升的自由变量集合、名称
):
    # 定义一个函数 dedup_and_sort_lifted_freevars，用于处理两个 lifted freevars 集合的重复和排序
    def dedup_and_sort_lifted_freevars(l_lifted_freevars, r_lifted_freevars):
        # shared_getattrs 函数用于找出两个 lifted proxies 集合中共享的 get_attr 节点
        def shared_getattrs(l_lifted_proxies, r_lifted_proxies):
            # 从左侧 lifted proxies 中提取所有 get_attr 节点的目标和对应的 proxy
            true_targets = {
                proxy.node.target: proxy
                for proxy in l_lifted_proxies
                if proxy.node.op == "get_attr"
            }
            l_shared_getattrs = {}
            r_shared_getattrs = {}

            # 遍历右侧 lifted proxies，如果发现同样的 get_attr 目标在左侧中也存在，则将它们作为共享的节点存储
            for false_proxy in r_lifted_proxies:
                if (
                    false_proxy.node.op == "get_attr"
                    and false_proxy.node.target in true_targets
                ):
                    true_proxy = true_targets[false_proxy.node.target]
                    l_shared_getattrs[true_proxy] = true_proxy
                    r_shared_getattrs[false_proxy] = true_proxy
            return l_shared_getattrs, r_shared_getattrs

        # 调用 shared_getattrs 函数，获取左右 lifted freevars 集合中共享的 get_attr 节点
        l_shared_getattrs, r_shared_getattrs = shared_getattrs(
            l_lifted_freevars.keys(), r_lifted_freevars.keys()
        )

        # 计算共享的 freevars 和独特的 freevars
        l_shared_freevars = (l_lifted_freevars.keys() & r_lifted_freevars.keys()).union(
            l_shared_getattrs.keys()
        )
        r_shared_freevars = (l_lifted_freevars.keys() & r_lifted_freevars.keys()).union(
            r_shared_getattrs.keys()
        )
        unique_l_freevars = l_lifted_freevars.keys() - l_shared_freevars
        unique_r_freevars = r_lifted_freevars.keys() - r_shared_freevars

        # 定义一个内部函数 _sort_by_name，用于按节点名称对变量进行排序
        def _sort_by_name(vars):
            return sorted(vars, key=lambda var: var.node.name)

        # 返回四个列表，分别为排序后的共享左侧变量、共享右侧变量、独特左侧变量和独特右侧变量
        return (
            list(_sort_by_name(list(l_shared_freevars))),
            list(_sort_by_name(list(r_shared_freevars))),
            list(_sort_by_name(list(unique_l_freevars))),
            list(_sort_by_name(list(unique_r_freevars))),
        )

    # 调用 dedup_and_sort_lifted_freevars 函数，获取处理后的四个结果列表
    (l_shared, r_shared, unique_l, unique_r) = dedup_and_sort_lifted_freevars(
        l_lifted_freevars, r_lifted_freevars
    )

    # 以下是对 cond(pred, true_fn, false_fn, (x,)) 的描述，假设 set_graph_input 自动设置
    # 在这种情况下，true_fn 包含了被提升的变量 x, a, b, c
    # false_fn has lifted variables x, a, b, d
    # Then fixup_branch_inps make sure both branches have the same signature, i.e.:
    # - true_fn(x, a, b, c_true_branch, d_false_branch)
    # - false_fn(x, a, b, c_true_branch, d_false_branch)
    #
    # More formally, the signature has three parts in the following order:
    # 1. used in both branches: x, a, b
    # 2. only used in true branches: c, suffixed with _true_branch
    # 3. only used in false branches: d, suffixed with _false_branch
    # Within each part, we re-order the nodes by name to have a deterministic ordering for testing.
    def fixup_branch_inps(graph, lifted_freevars, shared, unique_l, unique_r):
        def _insert_or_replace_phs(new_args, name_suffix):
            # Iterate over each argument in new_args list.
            for arg in new_args:
                # Create a new placeholder node with a name suffixed by name_suffix.
                new_ph = graph.placeholder(arg.node.name + name_suffix)
                # Check if the argument exists in lifted_freevars.
                if arg in lifted_freevars:
                    # Replace all uses of the old placeholder with the new placeholder.
                    old_ph = lifted_freevars[arg].node
                    old_ph.replace_all_uses_with(new_ph)
                    # Manually clean users of the old placeholder to erase it completely.
                    old_ph.users = {}
                    graph.erase_node(old_ph)
    
        # Find the first non-placeholder node in the graph.
        first_not_ph_node = next(node for node in graph.nodes if node.op != "placeholder")
        # Insert or replace placeholders in the graph before the first non-placeholder node.
        with graph.inserting_before(first_not_ph_node):
            _insert_or_replace_phs(shared, "")
            _insert_or_replace_phs(unique_l, "_" + l_name)  # Assuming l_name is defined elsewhere
            _insert_or_replace_phs(unique_r, "_" + r_name)  # Assuming r_name is defined elsewhere
    
    # Apply fixup_branch_inps function to both left and right graphs, returning modified objects.
    fixup_branch_inps(l_graph, l_lifted_freevars, l_shared, unique_l, unique_r)
    fixup_branch_inps(r_graph, r_lifted_freevars, r_shared, unique_l, unique_r)
    # Return the modified left and right graphs along with shared variables and unique variables for both branches.
    return l_graph, r_graph, l_shared, r_shared, unique_l, unique_r
# See NOTE [HigherOrderOperator tracing design] for details of the design
def speculate_subgraph(
    tx,
    f,
    sub_args,
    sub_kwargs,
    description,
    *,
    # source_target is the .value of HigherOrderOpVariable and is the
    # target of the proxy that we created for the higherOrderOperator.
    source_target=None,
    always_restore=False,
    enable_grad=None,
    # NOTE [argument `set_subgraph_inputs`]
    # set_subgraph_inputs controls how to construct subgraphs' placeholders from sub_args.
    # 1. if your HOP supports arbitrary inputs, use set_subgraph_inputs="automatic" (most recommended).
    # 2. if your HOP supports only Tensor and symnode inputs, use set_subgraph_inputs="flatten_manual" (recommended).
    # If sub_args contain Pytree structure (e.g. dict/list/tuple/set), the sub_args will be flattened first.
    # Then the flattened args are manually set as subgraph's placeholders.
    # 3. if your HOP must preserve inputs that are not tensor or symnode as placeholders e.g. AutogradFunctionContextVariable
    # use set_subgraph_inputs="manual" (not recommended). We do not recommend it in general because it has the
    # restriction that user need to manually control how to create placeholders and VariableTrackers for the args.
    set_subgraph_inputs="automatic",
    restore_side_effects=True,
    should_flatten_outputs=False,
    # Pass in an originating tracer - this is needed for preserving context
    # across fwd-bwd for autograd.Function
    tracer=None,
):
    if sub_kwargs is None:
        sub_kwargs = {}

    assert set_subgraph_inputs in {
        "automatic",
        "semi_automatic",
        "flatten_manual",
        "manual",
    }, "Please use one of the supported set_subgraph_inputs options."

    # See NOTE [Temporary argument `set_subgraph_inputs`]
    if sub_kwargs and set_subgraph_inputs != "automatic":
        # Raise an unimplemented exception if sub_kwargs are passed and set_subgraph_inputs is not "automatic"
        unimplemented("Use `set_subgraph_inputs=automatic` when passing `sub_kwargs`.")

    # Handle Unsupported exception
    except Unsupported as ex:
        f_name = f"{type(f).__name__}"
        if isinstance(f, UserFunctionVariable):
            f_name = f.get_name()
        # Construct an error message for logging
        msg = (
            f"speculate_subgraph: while introspecting {description}, we were unable "
            f"to trace function `{f_name}` into a single graph. This means "
            f"that Dynamo was unable to prove safety for this API and will "
            f"fall back to eager-mode PyTorch, which could lead to a slowdown."
        )
        # Log the error message
        log.info(msg)
        # Log the exception
        log.info(ex)
        # Raise the exception
        raise ex


# Create a proxy node for accessing an attribute in the transaction output
def make_attr(tx, name):
    node = tx.output.create_proxy(
        "get_attr",
        name,
        (),  # No positional arguments for the attribute
        {},  # No keyword arguments for the attribute
    )
    return node


# Add a subgraph to the transaction output with a unique name based on `name`
def add_subgraph(tx, name, gm):
    next_name = None
    i = 0
    while not next_name:
        candidate = f"{name}_{i}"
        if candidate in tx.output.nn_modules:
            i += 1
        else:
            next_name = candidate

    # Assign the unique name to the graph module
    gm.__name__ = next_name
    # Disable dynamic behavior for TorchDynamo for this subgraph
    gm.torchdynamo_force_dynamic = False
    # 将图形模块注册为属性或模块。由于图形模块不在用户空间中，因此无法从源代码中访问它。
    # 因此，将参数 source 设为 None，表示没有特定的来源。
    tx.output.register_attr_or_module(gm, next_name, source=None)
    # 返回变量 next_name 的值作为函数的结果
    return next_name
# 继承自变量追踪器的 TorchHigherOrderOperatorVariable 类，用于跟踪高阶操作符变量
class TorchHigherOrderOperatorVariable(VariableTracker):
    def __init__(
        self, value: HigherOrderOperator, source: Optional[Source] = None, **kwargs
    ):
        super().__init__(**kwargs)
        # 初始化高阶操作符的值
        self.value = value
        # 初始化源对象
        self.source = source

    @staticmethod
    def make(value, source=None, **kwargs):
        # 根据 value 的不同值，创建不同类型的高阶操作变量
        if value.__name__ == "cond":
            return CondHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ == "while_loop":
            return WhileLoopHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ in ("map", "map_impl"):
            return MapHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ == "executorch_call_delegate":
            return ExecutorchCallDelegateHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ == "out_dtype":
            return OutDtypeHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ == "wrap":
            return WrapHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ == "flex_attention":
            return TemplatedAttentionHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ in (
            "wrap_activation_checkpoint",
            "tag_activation_checkpoint",
        ):
            return CheckpointHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ == "_export_tracepoint":
            return ExportTracepointHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ == "trace_wrapped":
            return TraceWrappedHigherOrderOperatorVariable(value, source, **kwargs)
        elif value.__name__ == "strict_mode":
            return StrictModeHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ == "run_with_rng_state":
            return RunWithRNGStateHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ == "associative_scan":
            return AssociativeScanHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ == "call_torchbind":
            return CallTorchbindHigherOrderVariable(value, source, **kwargs)
        else:
            # 抛出未实现的异常，如果 value 的类型未被处理
            unimplemented(f"HigherOrderOperator {value.__name__}")

    def call_function(
        self, tx, args: List[VariableTracker], kwargs: Dict[str, VariableTracker]
    ) -> VariableTracker:
        # 抛出未实现的异常，表示该高阶操作符的函数调用尚未实现
        unimplemented(f"HigherOrderOperator {self.value.__name__}")


# 继承自 TorchHigherOrderOperatorVariable 的 CondHigherOrderVariable 类
class CondHigherOrderVariable(TorchHigherOrderOperatorVariable):
    # 装饰器，如果图形中断，则抛出硬错误，说明 Cond 必须完全捕获才能正常工作，通过 torch.compile
    @raise_hard_error_if_graph_break(
        reason="Cond doesn't work unless it is captured completely with torch.compile."
    )
    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ):
        # 方法定义未完整，缺少右括号和方法主体
        ...
        

# 继承自 TorchHigherOrderOperatorVariable 的 CallTorchbindHigherOrderVariable 类
class CallTorchbindHigherOrderVariable(TorchHigherOrderOperatorVariable):
    def __init__(self, hop, source, script_obj_var, method_name):
        super().__init__(hop, source)
        # 初始化脚本对象变量
        self.script_obj_var = script_obj_var
        # 初始化方法名称
        self.method_name = method_name
    # 定义一个方法，用于调用函数，并返回函数调用的结果
    def call_function(
        self, tx, args: List[VariableTracker], kwargs: Dict[str, VariableTracker]
    ) -> VariableTracker:
        # 从本地模块导入 wrap_fx_proxy 函数
        from .builder import wrap_fx_proxy

        # 对传入的参数和关键字参数进行延迟实现，确保所有参数都已被实例化
        args, kwargs = LazyVariableTracker.realize_all((args, kwargs))

        # 将参数列表中的每个参数转换为代理对象
        args_proxy = [arg.as_proxy() for arg in args]
        # 将关键字参数中的每个值转换为代理对象
        kwargs_proxy = {k: v.as_proxy() for k, v in kwargs.items()}
        
        # 调用 wrap_fx_proxy 函数，将函数调用的结果作为返回值
        return wrap_fx_proxy(
            tx=tx,
            # 创建一个代理对象，表示函数调用
            proxy=tx.output.create_proxy(
                "call_function",
                self.value,
                args=tuple(
                    [self.script_obj_var.as_proxy(), self.method_name] + args_proxy
                ),
                kwargs=kwargs_proxy,
            ),
        )
class WhileLoopHigherOrderVariable(TorchHigherOrderOperatorVariable):
    # 使用装饰器确保在图形中完整捕获 while_loop，否则抛出硬错误
    @raise_hard_error_if_graph_break(
        reason="while_loop doesn't work unless it is captured completely with torch.compile."
    )
    # 重写父类方法，用于调用函数
    def call_function(
        self, tx, args: List[VariableTracker], kwargs: Dict[str, VariableTracker]



class AssociativeScanHigherOrderVariable(TorchHigherOrderOperatorVariable):
    # 使用装饰器确保在图形中完整捕获 associative_scan，否则抛出硬错误
    @raise_hard_error_if_graph_break(
        reason="associative_scan must be captured completely with torch.compile."
    )
    # 重写父类方法，用于调用函数
    def call_function(
        self, tx, args: List[VariableTracker], kwargs: Dict[str, VariableTracker]



def non_single_tensor_return_unsupported(api, ret):
    from . import TensorVariable

    # 检查返回值是否为 TensorVariable 类型，否则抛出不支持的异常
    if not isinstance(ret, TensorVariable):
        raise Unsupported(
            f"{api} over function that returns something " f"other than one Tensor"
        )



class MapHigherOrderVariable(TorchHigherOrderOperatorVariable):
    # 重写父类方法，用于调用函数
    def call_function(
        self, tx, args: List[VariableTracker], kwargs: Dict[str, VariableTracker]
    # 返回类型声明为 VariableTracker
    ) -> VariableTracker:
        # 从当前目录中导入 TensorVariable 类
        from . import TensorVariable
        # 从 builder 模块导入 wrap_fx_proxy_cls 函数

        # 如果 kwargs 不为空，则抛出未实现的异常
        if len(kwargs) > 0:
            unimplemented(
                "torch.ops.higher_order.map: kwargs are not supported in the map operator."
            )

        # 检查 args[0] 是否可调用，并支持的参数类型
        _check_supported_callable_arg(tx, args[0].realize(), "map_fn")

        # 确保 args[1] 是 TensorVariable 类型
        assert type(args[1].realize()) is TensorVariable

        # 获取 args[1] 的虚拟代理对象，并通过其节点获取示例形状
        sample_shape = get_fake_value(args[1].as_proxy().node, tx).size()

        # 如果示例形状维度小于1或第一个维度为0，则抛出未实现的异常
        if len(sample_shape) < 1 or sample_shape[0] == 0:
            unimplemented(
                "map() operator doesn't support scalar or zero-sized tensors during tracing."
            )

        # 用于展示 map() 输出示例的第一个维度的包装代理类
        first_dim = wrap_fx_proxy_cls(
            target_cls=TensorVariable, tx=tx, proxy=args[1].as_proxy()[0]
        )

        # 推测子图的输出及其特性
        (
            (body_r, body_spec),
            body_graph,
            body_lifted_freevars,
        ) = speculate_subgraph(
            tx,
            args[0],
            [
                first_dim,
                *args[2:],
            ],
            {},
            "torch.ops.higher_order.map",
            source_target=self.value,
            set_subgraph_inputs="flatten_manual",
            should_flatten_outputs=True,
        )

        # 获取子图示例值的代理节点的 meta 属性
        subgraph_example_value = [
            proxy.node.meta["example_value"] for proxy in body_r.as_proxy()
        ]

        # 在输出模式下扩展 map() 的示例输出，使其具有与映射输入相同的第一个维度
        map_example_out = [
            t.expand(sample_shape[0], *t.size()).clone(
                memory_format=torch.contiguous_format
            )
            for t in subgraph_example_value
        ]

        # 复制 tx.output.nn_modules 并存储为字典
        body_nn_modules = dict(tx.output.nn_modules)

        # 添加子图到 Torch FX 中，并获取子图节点名称
        body_name = add_subgraph(
            tx,
            "map_body",
            torch.fx.GraphModule(body_nn_modules, body_graph),
        )

        # 创建代表子图节点的属性节点
        body_node = make_attr(tx, body_name)

        # 准备传递给 _call_function_and_unflatten_output 函数的参数
        p_args = (
            body_node,
            [args[1].as_proxy()],
            [arg.as_proxy() for arg in args[2:]] + list(body_lifted_freevars.keys()),
        )

        # 调用函数并展开输出，返回结果
        return _call_function_and_unflatten_output(
            tx, torch.ops.higher_order.map_impl, p_args, {}, map_example_out, body_spec
        )
class ExecutorchCallDelegateHigherOrderVariable(TorchHigherOrderOperatorVariable):
    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        from .builder import wrap_fx_proxy

        # 这是Executorch内部的委托操作符，用于调用给定降阶模块中的特定函数，并传递给定的操作符。
        # 实际的操作符定义在Executorch代码库中。
        # 这里存在层次结构上的问题，因为executorch_call_delegate位于比dynamo更高的层次，
        # 但目前还没有真正解决这个问题的方案。

        # 如果传入的kwargs不为空，抛出未实现的异常，因为暂时不支持使用kwargs参数
        if len(kwargs) > 0:
            unimplemented(
                "executorch_call_delegate: kwargs arguments were not enabled."
            )

        # 从tx输出中获取子模块对应的降阶模块
        lowered_module = tx.output.get_submodule(args[0].module_key)

        # 创建降阶节点
        lowered_node = make_attr(tx, args[0].module_key)

        # 将参数转换为代理对象，用于处理FakeTensor
        p_args = tuple(arg.as_proxy() for arg in args[1:])
        real_sub_args = pytree.tree_map_only(
            torch.fx.Proxy, lambda a: get_fake_value(a.node, tx), p_args
        )

        # 在虚拟模式下执行降阶模块的函数，获取示例值
        with tx.fake_mode:
            example_value = lowered_module.original_module.module()(*real_sub_args)

        # 注释：
        # [确保FakeTensor和实际张量的一对一对应关系]:
        # executorch模块承诺不会给输入和输出起别名。
        # 因此，输出的FakeTensor不会错误地引用输入的FakeTensor。
        _assert_tensors_nonaliasing(real_sub_args, example_value)

        # 更新参数列表，加入降阶节点
        p_args = (lowered_node,) + p_args

        # 封装成FX代理对象，用于输出调用函数的结果
        return wrap_fx_proxy(
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_function",
                self.value,
                args=tuple(p_args),
                kwargs={},
            ),
            example_value=example_value,
        )
    def create_wrapped_node(self, tx, args, kwargs, description):
        # See NOTE [HigherOrderOperator tracing design] for more details

        (
            (body_r, treespec),  # 解构赋值：从 speculate_subgraph 的返回值中获取 body_r 和 treespec
            body_graph,  # 调用 speculate_subgraph 返回的图形对象
            body_lifted_freevars,  # 调用 speculate_subgraph 返回的 lifted free variables
        ) = speculate_subgraph(
            tx,
            args[0],  # 第一个参数是函数
            [*args[1:]],  # 剩余的位置参数
            kwargs,  # 关键字参数
            description,  # 描述字符串
            source_target=self.value,  # 源目标对象
            should_flatten_outputs=True,  # 是否展平输出
        )

        body_gmod = torch.fx.GraphModule(tx.output.nn_modules, body_graph)  # 创建一个 GraphModule 对象
        body_name = add_subgraph(
            tx,
            "wrap_body",  # 子图的名称
            body_gmod,  # 包含的 GraphModule 对象
        )

        body_node = make_attr(tx, body_name)  # 创建一个属性节点对象

        # Since, we call `speculate_subgraph` with `set_subgraph_inputs="automatic`,
        # all the arguments are lifted.
        lifted_args = tuple(arg for arg in body_lifted_freevars.keys())  # 获取所有 lifted 的参数名组成的元组

        proxy_args = (body_node,) + lifted_args  # 组合成代理参数元组
        example_value = pytree.tree_map_only(
            torch.fx.Proxy,
            lambda a: a.node.meta["example_value"],  # 使用 lambda 函数从 body_r 的代理中获取示例值
            body_r.as_proxy(),  # 将 body_r 转换为代理对象
        )

        return proxy_args, {}, example_value, body_r, treespec, body_gmod  # 返回所有计算得到的值和对象

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        # This flattens the kwargs into lifted args
        p_args, p_kwargs, example_value, body_r, treespec, _ = self.create_wrapped_node(
            tx, args, kwargs, "wrap"
        )  # 调用 create_wrapped_node 方法获得相关值

        if len(p_kwargs) > 0:
            unimplemented("kwargs should have been flattened into lifted args")  # 如果 p_kwargs 不为空则报未实现异常

        flat_example_value = pytree.tree_map_only(
            torch.fx.Proxy,
            lambda a: a.node.meta["example_value"],  # 使用 lambda 函数从 body_r 的代理中获取示例值
            body_r.as_proxy(),  # 将 body_r 转换为代理对象
        )

        return _call_function_and_unflatten_output(
            tx, self.value, tuple(p_args), p_kwargs, flat_example_value, treespec  # 调用另一个函数来执行函数调用和展开输出
        )
# OutDtypeHigherOrderVariable 类继承自 TorchHigherOrderOperatorVariable 类，表示一种特定的高阶变量类型

    # 重写 call_function 方法，处理函数调用操作
    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        # 导入 wrap_fx_proxy 函数，用于包装 fx 代理对象
        from .builder import wrap_fx_proxy

        # 如果关键字参数 kwargs 的长度大于 0，则抛出未实现的异常
        if len(kwargs) > 0:
            unimplemented("out_dtype does not handle kwargs")

        # 将 args 中的每个元素转换为其代理对象的元组 p_args
        p_args = tuple(arg.as_proxy() for arg in args)

        # 取出操作符 op 和输出数据类型 output_dtype
        op = p_args[0]
        output_dtype = p_args[1]

        # 使用 pytree.tree_map_only 函数，将 p_args[2:] 中的每个元素转换为 torch.fx.Proxy 对象，构成 fake_sub_args
        fake_sub_args = pytree.tree_map_only(
            torch.fx.Proxy, lambda a: a.node.meta["example_value"], p_args[2:]
        )

        # 这是用于追踪的简化操作符实现，实际实现可能会先提升参数
        # 调用 op 函数，传入 fake_sub_args，并指定输出数据类型为 output_dtype，得到示例值 example_value
        example_value = op(*fake_sub_args).to(dtype=output_dtype)

        # 将该调用保存为一个 call 函数
        return wrap_fx_proxy(
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_function",
                self.value,
                args=tuple(p_args),
                kwargs={},
            ),
            example_value=example_value,
        )


# StrictModeHigherOrderVariable 类继承自 TorchHigherOrderOperatorVariable 类，表示另一种特定的高阶变量类型

    # 装饰器 raise_hard_error_if_graph_break，用于捕获图中断错误并抛出硬错误
    @raise_hard_error_if_graph_break(
        reason="strict_mode HOO doesn't work unless it is captured completely with torch.compile."
    )
    # 重写 call_function 方法，处理函数调用操作
    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
        ) -> "VariableTracker":
            # 定义函数签名，返回类型为 "VariableTracker"
            callable = args[0]

            # 解包第二个参数中的变量序列
            unpacked_sequence = args[1].unpack_var_sequence(tx)
            # TODO (tmanlaibaatar) support pytree here
            # TODO: 在这里支持 pytree，暂未实现

            # 检查是否有复杂类型的参数，目前只支持扁平化输入
            for arg in unpacked_sequence:
                if isinstance(arg, (ListVariable, TupleVariable, ConstDictVariable)):
                    unimplemented("strict_mode HOO only works for flat inputs for now")
                    # 未实现的功能：目前 strict_mode HOO 只能处理扁平化的输入

            if kwargs:
                unimplemented(
                    f"strict_mode HOO received unexpected kwargs: {list(kwargs.keys())}"
                )
                # 未实现的功能：strict_mode HOO 收到了意外的关键字参数列表

            # 推测子图
            (
                (ret_val, ret_treespec),
                ret_graph,
                ret_lifted_freevars,
            ) = speculate_subgraph(
                tx,
                args[0],
                unpacked_sequence,
                {},
                "strict_mode",
                source_target=self.value,
                should_flatten_outputs=True,
            )
            # 调用 speculate_subgraph 函数进行子图推测，获取返回值、树形规范、提升的自由变量

            # 复制输出的神经网络模块
            strict_mode_nn_modules = dict(tx.output.nn_modules)

            # 添加子图
            strict_mode_name = add_subgraph(
                tx,
                "strict_mode_body",
                torch.fx.GraphModule(strict_mode_nn_modules, ret_graph),
            )
            # 在事务中添加名为 "strict_mode_body" 的子图

            # 创建属性节点
            strict_mode_node = make_attr(tx, strict_mode_name)
            p_args = (
                strict_mode_node,
                tuple(arg for arg in ret_lifted_freevars.keys()),
            )
            # 准备参数元组 p_args

            # 扁平化示例值
            flat_example_value = pytree.tree_map_only(
                torch.fx.Proxy,
                lambda a: a.node.meta["example_value"],
                ret_val.as_proxy(),
            )
            # 使用 pytree 将示例值扁平化

            # 调用函数并展开输出
            return _call_function_and_unflatten_output(
                tx,
                torch.ops.higher_order.strict_mode,
                p_args,
                {},
                flat_example_value,
                ret_treespec,
            )
            # 调用 _call_function_and_unflatten_output 函数，执行严格模式函数，返回结果
# 定义一个名为 CheckpointHigherOrderVariable 的类，继承自 WrapHigherOrderVariable 类
class CheckpointHigherOrderVariable(WrapHigherOrderVariable):
    
    # 定义 call_function 方法，用于执行函数调用操作
    def call_function(
        self, tx, args: List[VariableTracker], kwargs: Dict[str, VariableTracker]
    ) -> VariableTracker:
        # 导入需要使用的模块和函数
        from torch._higher_order_ops.wrap import TagActivationCheckpoint
        from torch.utils.checkpoint import noop_context_fn
        from .builder import wrap_fx_proxy
        
        # 初始化 context_fn 变量
        context_fn = None
        
        # 检查 kwargs 中是否包含 "context_fn" 键，并且其值不是 noop_context_fn
        if "context_fn" in kwargs and kwargs["context_fn"] != noop_context_fn:
            # 弹出并获取 "context_fn" 对应的值
            ctx = kwargs.pop("context_fn")
            # 根据 ctx 的类型不同进行处理
            if isinstance(ctx, torch._dynamo.variables.UserFunctionVariable):
                context_fn = ctx.fn  # 获取 UserFunctionVariable 的函数
            elif isinstance(
                ctx, torch._dynamo.variables.functions.FunctoolsPartialVariable
            ):
                context_fn = ctx.as_python_constant()  # 获取 FunctoolsPartialVariable 的常量值
            else:
                # 抛出未实现错误，显示不支持的 context_fn 类型
                raise NotImplementedError(
                    f"checkpoint not implemented for {type(ctx)} context_fn"
                )
        
        # 调用 TagActivationCheckpoint.divide_kwargs 函数分离 checkpoint_kwargs 和 gmod_kwargs
        checkpoint_kwargs, gmod_kwargs = TagActivationCheckpoint.divide_kwargs(kwargs)
        
        # 调用父类方法 create_wrapped_node 创建包装后的节点
        (
            p_args,
            _,
            example_value,
            body_r,
            treespec,
            checkpointed_gmod,
        ) = self.create_wrapped_node(
            tx, args, gmod_kwargs, "torch.utils.checkpoint.checkpoint"
        )
        
        # 如果 context_fn 不为 None，则将其存储在 checkpointed_gmod 的 meta 字典中
        if context_fn is not None:
            checkpointed_gmod.meta["_checkpoint_context_fn"] = context_fn
        
        # 调用 proxy_args_kwargs 函数代理空列表和 checkpoint_kwargs
        _, checkpoint_kwargs = proxy_args_kwargs([], checkpoint_kwargs)
        
        # 使用 wrap_fx_proxy 函数封装函数调用的代理
        variable = wrap_fx_proxy(
            tx=tx,
            proxy=tx.output.create_proxy(
                "call_function",
                self.value,
                args=tuple(p_args),
                kwargs=checkpoint_kwargs,
            ),
            example_value=example_value,
        )
        
        # 如果 treespec 为 None，则直接返回 variable
        if treespec is None:
            return variable
        
        # 如果 treespec 不为 None，则将 variable 转换为列表，以符合 pytree API 的类型要求
        variable = BuiltinVariable(list).call_function(tx, [variable], {})
        
        # 调用 _make_inlined 函数，将变量重新转换为树形结构
        return _make_inlined(tx, pytree.tree_unflatten)(variable, treespec)


# 定义一个名为 ExportTracepointHigherOrderVariable 的类，继承自 TorchHigherOrderOperatorVariable 类
class ExportTracepointHigherOrderVariable(TorchHigherOrderOperatorVariable):
    
    # 定义 call_function 方法，用于执行函数调用操作
    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ):
    ) -> "VariableTracker":
        # 导入 wrap_fx_proxy 函数
        from .builder import wrap_fx_proxy
        
        # 创建一个包含参数代理的元组 p_args
        p_args = tuple(arg.as_proxy() for arg in args)
        
        # 创建一个包含关键字参数代理的字典 p_kwargs
        p_kwargs = {key: arg.as_proxy() for key, arg in kwargs.items()}
        
        # 使用 wrap_fx_proxy 函数封装一个函数调用的代理对象
        return wrap_fx_proxy(
            tx=tx,  # 传递事务对象 tx
            proxy=tx.output.create_proxy(
                "call_function",  # 创建一个函数调用类型的代理
                self.value,        # 使用 self 的值作为函数调用的对象
                args=p_args,       # 传递位置参数的代理列表
                kwargs=p_kwargs,   # 传递关键字参数的代理字典
            ),
            example_value=None,    # 示例值设为 None
        )
class RunWithRNGStateHigherOrderVariable(TorchHigherOrderOperatorVariable):
    # 定义一个继承自TorchHigherOrderOperatorVariable的类，用于运行带有随机数生成状态的高阶变量操作

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        # 定义一个方法call_function，接受事务tx、参数列表args和关键字参数字典kwargs，并返回VariableTracker对象
        from .builder import wrap_fx_proxy
        # 导入wrap_fx_proxy函数，用于包装fx代理对象

        p_args = tuple(arg.as_proxy() for arg in args)
        # 将args中每个元素转换为其代理形式，并组成元组p_args

        p_kwargs = {key: arg.as_proxy() for key, arg in kwargs.items()}
        # 将kwargs中每个值转换为其代理形式，并构建成字典p_kwargs

        return wrap_fx_proxy(
            tx=tx,
            # 调用wrap_fx_proxy函数，传入tx事务对象
            proxy=tx.output.create_proxy(
                "call_function",
                self.value,
                args=p_args,
                kwargs=p_kwargs,
            ),
            # 创建一个fx代理对象，用于tx的输出，在此处命名为"call_function"
            example_value=None,
            # 例子值设为None
        )


class TraceWrappedHigherOrderOperatorVariable(TorchHigherOrderOperatorVariable):
    """
    Handles torch._dynamo._trace_wrapped_higher_order_op.inner_trace
    by unwrapping the higher order op and inlining through it.  This op
    is created by dynamo to survive through AotAutograd, then unwrapped
    here in the call to dynamo from compiled autograd.
    """
    # 定义一个继承自TorchHigherOrderOperatorVariable的类，处理被包装的高阶操作符变量

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        # 定义一个方法call_function，接受事务tx、参数列表args和关键字参数字典kwargs，并返回VariableTracker对象

        kwargs = dict(kwargs)
        # 将kwargs转换为标准字典形式

        fn = kwargs.pop("fn")
        # 弹出关键字参数字典kwargs中的键为"fn"的值，并赋给变量fn

        return fn.call_function(tx, args, kwargs)
        # 调用fn对象的call_function方法，传入tx、args和kwargs，并返回其结果


class TemplatedAttentionHigherOrderVariable(TorchHigherOrderOperatorVariable):
    # 定义一个继承自TorchHigherOrderOperatorVariable的类，用于处理模板化的注意力高阶变量操作

    @staticmethod
    def normalize_to_args(args, kwargs):
        # 定义一个静态方法normalize_to_args，用于将输入的args和kwargs标准化为参数列表

        # input signature is (query, key, value, score_mod, *other_buffers)
        # Flatten args and kwargs into lists
        flat_args = pytree.tree_flatten(args)[0]
        # 将args展开为列表flat_args

        flat_kwargs = pytree.tree_flatten(kwargs)[0]
        # 将kwargs展开为列表flat_kwargs

        # Combine the flattened lists
        all_args = flat_args + flat_kwargs
        # 将flat_args和flat_kwargs合并成一个完整的参数列表all_args

        return all_args
        # 返回标准化后的参数列表all_args

    def create_wrapped_node(
        self, tx, query: "VariableTracker", score_function: "VariableTracker"
        # 定义一个方法create_wrapped_node，接受事务tx、查询query和得分函数score_function作为参数
    ):
        # Import necessary modules for specific functionalities
        from torch._higher_order_ops.flex_attention import TransformGetItemToIndex
        from .builder import SourcelessBuilder

        # Assign the given InstructionTranslator instance to tx
        tx: InstructionTranslator = tx

        # Determine if scores should require gradients based on query's requires_grad attribute
        scores_require_grad: bool = query.requires_grad
        
        # Create a score tensor using query.call_method with specific arguments and settings
        score = query.call_method(
            tx,
            "new_empty",
            (SourcelessBuilder.create(tx, []),),
            {"requires_grad": SourcelessBuilder.create(tx, scores_require_grad)},
        )

        # Define a function create_scalar() that returns a scalar tensor using query.call_method
        def create_scalar():
            return query.call_method(
                tx,
                "new_empty",
                (SourcelessBuilder.create(tx, []),),
                {
                    "dtype": SourcelessBuilder.create(tx, torch.int32),
                },
            )

        # Generate a list bhmn containing four scalar tensors using create_scalar()
        bhmn = [create_scalar() for _ in range(4)]
        
        # Prepare new_args list containing score and bhmn tensors
        new_args = [score, *bhmn]

        # Use TransformGetItemToIndex context to perform the following operations in a modified environment
        with TransformGetItemToIndex():
            (
                (body_output, body_treespec),
                body_graph,
                body_lifted_freevars,
            ) = speculate_subgraph(
                tx,
                score_function,
                new_args,
                {},  # expect only args no kwargs for now
                description="flex_attention",
                source_target=self.value,
                set_subgraph_inputs="flatten_manual",
            )

        # Add a subgraph to the tx with "flex_attention" description and body_graph
        body_name = add_subgraph(
            tx,
            "flex_attention",
            torch.fx.GraphModule(tx.output.nn_modules, body_graph),
        )

        # Create a node for the body_name using make_attr
        body_node = make_attr(tx, body_name)

        # Explain the necessity of speculating subgraph for capturing free variables
        # and creating proxies for inputs not passed as arguments
        # lifted_args contains keys from body_lifted_freevars, representing additional arguments
        lifted_args = tuple(arg for arg in body_lifted_freevars.keys())
        
        # proxy_args combines body_node with lifted_args to form the final return value
        proxy_args = (body_node,) + lifted_args

        # Return proxy_args containing the body_node and additional lifted arguments
        return proxy_args

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        # 导入 wrap_fx_proxy 函数，用于创建函数代理
        from .builder import wrap_fx_proxy

        (
            # 使用 normalize_to_args 方法将参数规范化为标准形式
            query,
            key,
            value,
            score_mod,
            sparse_kv_num_blocks,
            sparse_kv_indices,
            sparse_q_num_blocks,
            sparse_q_indices,
            SPARSE_KV_BLOCK_SIZE,
            SPARSE_Q_BLOCK_SIZE,
        ) = self.normalize_to_args(args, kwargs)

        # 创建包装节点，并传入 tx, query, score_mod 参数
        p_args = self.create_wrapped_node(tx, query, score_mod)
        # 构建代理参数列表，除了 score_mod 以外都包含在 proxied_args 中
        proxied_args = [
            query,
            key,
            value,
            sparse_kv_num_blocks,
            sparse_kv_indices,
            sparse_q_num_blocks,
            sparse_q_indices,
            SPARSE_KV_BLOCK_SIZE,
            SPARSE_Q_BLOCK_SIZE,
        ]

        # 将 proxied_args 代理化，排除掉 score_function 参数，因为不支持代理用户定义函数
        inp_args, _ = proxy_args_kwargs(proxied_args, {})

        # 从 query 的代理中获取元数据中的 example_value
        query_meta = query.as_proxy().node.meta["example_value"]
        # 计算 logsumexp_shape，获取其形状，例如 [B, H, M]
        logsumexp_shape = query_meta.size()[:-1]  # [B, H, M]
        
        # 使用 fake_mode 创建 torch 上下文，生成 out_meta 和 lse_meta
        with torch._guards.TracingContext.try_get().fake_mode:
            out_meta = torch.empty_like(
                query_meta, memory_format=torch.contiguous_format
            )
            lse_meta = query_meta.new_empty(logsumexp_shape, dtype=torch.float32)
        example_value = (out_meta, lse_meta)

        # 组合有序的 HOO 参数，包括 inp_args 和 p_args 的一部分
        # - inp_args: [query, key, value, sparse_kv_num_blocks, sparse_kv_indices,
        #   sparse_q_num_blocks, sparse_q_indices, SPARSE_KV_BLOCK_SIZE, SPARSE_Q_BLOCK_SIZE]
        # - p_args: [score_mod, *other_buffers]
        return wrap_fx_proxy(
            tx=tx,
            # 创建函数调用代理，并传入 self.value 和组合后的参数
            proxy=tx.output.create_proxy(
                "call_function",
                self.value,
                args=inp_args[:3] + p_args[:1] + inp_args[3:] + p_args[1:],
                kwargs={},
            ),
            example_value=example_value,
        )
# 定义名为 AutogradFunctionApplyVariable 的类，继承自 VariableTracker 类
class AutogradFunctionApplyVariable(VariableTracker):
    # 初始化方法，接受 fwd_graph、bwd_graph、parent_source 和其他关键字参数
    def __init__(self, fwd_graph, bwd_graph, parent_source, **kwargs):
        # 调用父类 VariableTracker 的初始化方法
        super().__init__(**kwargs)
        # 将参数 fwd_graph 存储在实例变量 self.fwd_graph 中
        self.fwd_graph = fwd_graph
        # 将参数 bwd_graph 存储在实例变量 self.bwd_graph 中
        self.bwd_graph = bwd_graph
        # 将参数 parent_source 存储在实例变量 self.parent_source 中
        self.parent_source = parent_source

    # 定义名为 call_function 的方法，接受 tx、args 和 kwargs 作为参数
    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"):
```