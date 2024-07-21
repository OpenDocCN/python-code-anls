# `.\pytorch\torch\distributed\_spmd\distribute.py`

```py
# 引入日志模块
import logging
# 引入操作符模块
import operator
# 引入数据类模块
from dataclasses import dataclass
# 引入自动枚举模块
from enum import auto, Enum
# 引入偏函数模块
from functools import partial
# 引入类型提示模块
from typing import Any, Callable, cast, Dict, List, Optional, Sequence, Tuple, Union

# 引入PyTorch主模块
import torch
# 引入分布式操作模块
import torch.distributed._spmd.experimental_ops
# 引入FX模块
import torch.fx as fx
# 引入通信张量模块
from torch.distributed._spmd.comm_tensor import _get_tracer
# 引入图工具模块
from torch.distributed._spmd.graph_utils import OP
# 引入日志工具模块
from torch.distributed._spmd.log_utils import get_logger
# 引入张量模块
from torch.distributed._tensor import DeviceMesh, DTensor
# 引入操作架构模块
from torch.distributed._tensor._op_schema import OpSchema
# 引入本地张量分布模块
from torch.distributed._tensor._redistribute import redistribute_local_tensor
# 引入张量位置类型模块
from torch.distributed._tensor.placement_types import (
    _Partial,
    DTensorSpec,
    Placement,
    Replicate,
    Shard,
    TensorMeta,
)
# 引入FX实验代理张量模块
from torch.fx.experimental.proxy_tensor import make_fx, proxy_slot
# 引入PyTree模块
from torch.utils import _pytree as pytree
# 引入PyTree扁平化模块
from torch.utils._pytree import tree_flatten, tree_map, tree_map_only, tree_unflatten

# 定义日志记录器变量，可选类型为logging.Logger或None
logger: Optional[logging.Logger] = None

# 使用torch.ops.aten别名aten来引用PyTorch中的aten操作

class TrainingPhase(Enum):
    FORWARD = auto()
    BACKWARD = auto()

# 数据类Schema，用于表示张量的设备网格和位置列表
@dataclass
class Schema:
    mesh: DeviceMesh  # 张量的设备网格
    placements: List[Placement]  # 张量的位置列表

# DSymInt类表示从DTensor中由SymInt操作检索到的值
@dataclass
class DSymInt:
    """DSymInt represents a value retrieved by a SymInt op from a DTensor.

    DSymInt helps View and Factory ops to determine the placement and shape of the
    output tensor, as those operators either do not have an input DTensor or
    the input DTensor is insufficient to determine the output tensor's placement.
    """

    global_value: int  # SymInt评估的全局值
    local_value: int  # SymInt在本地片上评估的值
    mesh: DeviceMesh  # 包含此SymInt的DTensor的设备网格

    # 判断是否为片上操作
    def is_shard(self) -> bool:
        return self.local_value != self.global_value

    # 从FX节点和DTensor创建DSymInt对象的类方法
    @classmethod
    def from_node(cls, node: fx.Node, dtensor: DTensor) -> "DSymInt":
        dim: int = 0
        # 如果目标为aten.sym_size，则创建并返回DSymInt对象
        if node.target == aten.sym_size:
            dim = cast(int, node.args[1])
            return cls(
                global_value=dtensor.size(dim),
                local_value=dtensor.to_local().size(dim),
                mesh=dtensor.device_mesh,
            )
        # 如果目标为aten.sym_numel，则创建并返回DSymInt对象
        elif node.target == aten.sym_numel:
            return cls(
                global_value=dtensor.numel(),
                local_value=dtensor.to_local().numel(),
                mesh=dtensor.device_mesh,
            )
        # 如果目标为aten.sym_stride，则创建并返回DSymInt对象
        elif node.target == aten.sym_stride:
            dim = cast(int, node.args[1])
            return cls(
                global_value=dtensor.stride(dim),
                local_value=dtensor.to_local().stride(dim),
                mesh=dtensor.device_mesh,
            )
        else:
            # 抛出未实现的异常，说明不支持该类型的DSymInt
            raise NotImplementedError(f"DSymInt does not support {node.target}")

# 检查对象是否为部分DTensor的函数
def _is_partial_dtensor(obj: Any) -> bool:
    """检查对象是否是 DTensor 类型，并且是否包含任何 _Partial 实例。"""
    # 如果对象不是 DTensor 类型，则返回 False
    if not isinstance(obj, DTensor):
        return False

    # 初始化一个标志位，用于指示是否存在 _Partial 实例
    is_partial = False

    # 遍历对象中的 placements 属性
    for placement in obj.placements:
        # 如果当前遍历到的 placement 是 _Partial 的实例
        if isinstance(placement, _Partial):
            # 将标志位设置为 True
            is_partial = True
            # 停止遍历，因为已经找到了 _Partial 实例
            break

    # 返回标志位，指示是否存在 _Partial 实例
    return is_partial
# 定义一个函数，用于处理带有本地张量的操作
def _dispatch_with_local_tensors(
    op: torch._ops.OpOverload,  # op 参数指定了操作的类型
    local_args: Tuple[Any, ...],  # local_args 是一个元组，包含所有本地参数
    kwargs: Optional[Dict[str, Any]] = None,  # kwargs 是一个可选的字典，包含关键字参数
    specs: Optional[  # specs 是一个可选的字典，指定了张量的详细规格
        Dict[
            torch.Tensor,  # 键是 torch.Tensor 对象
            Tuple[torch.Size, DeviceMesh, Sequence[Placement], Sequence[Placement]],  # 值是包含张量规格信息的元组
        ]
    ] = None,
) -> Any:  # 函数返回一个任意类型的值

    if kwargs is None:
        kwargs = {}  # 如果 kwargs 为 None，则将其初始化为空字典
    if specs is None:
        specs = {}  # 如果 specs 为 None，则将其初始化为空字典

    # 定义内部函数 redistribute，用于处理参数重新分配的逻辑
    def redistribute(arg: Any) -> Any:
        tensor_shape, mesh, current_placement, target_placement = specs[arg]
        # 创建 TensorMeta 对象，描述张量的元数据
        tensor_meta = TensorMeta(
            tensor_shape,
            stride=arg.stride(),
            dtype=arg.dtype,
        )
        # 创建当前和目标规格的 DTensorSpec 对象
        current_spec = DTensorSpec(
            mesh, tuple(current_placement), tensor_meta=tensor_meta
        )
        target_spec = DTensorSpec(
            mesh, tuple(target_placement), tensor_meta=tensor_meta
        )

        # 如果 arg 是 torch.Tensor 类型且在 specs 字典中存在，则进行本地张量的重新分配
        return (
            redistribute_local_tensor(arg, current_spec, target_spec)  # type: ignore[index]
            if isinstance(arg, torch.Tensor) and arg in specs  # type: ignore[operator]
            else arg  # 否则直接返回 arg
        )

    # TODO: this is broken because it won't redistributed potential tensors on the kwargs
    # 调用 op 函数，并对 local_args 中的每个参数应用 redistribute 函数，同时传递 kwargs
    return op(*tree_map(redistribute, local_args), **kwargs)


# Figure out how to specify a type spec for the return specs value
# without the entire structure.
# pyre-fixme
# 定义一个函数，用于更新为重新分配准备的规格（specs），同时调整参数以适应本地张量的重新分配
def _update_specs_for_redistribute(args, target_schema, redistribute):
    # 从 pack_args_kwargs_with_local_tensor 中适配的代码
    flatten_args, args_tree_spec = tree_flatten(args)
    flatten_args_schema = pytree.tree_leaves(target_schema.args_schema)

    # 初始化一个空的 specs 字典，用于存储张量的详细规格
    specs: Dict[
        torch.Tensor,
        Tuple[
            torch.Size,
            DeviceMesh,
            Sequence[Placement],
            Sequence[Placement],
        ],
    ] = {}

    # 遍历 flatten_args 中的每个元素
    for i, arg in enumerate(flatten_args):
        if isinstance(arg, DTensor):
            if redistribute:
                # 如果需要进行重新分配，则将张量及其相关规格添加到 specs 字典中
                specs[arg._local_tensor] = (
                    arg.size(),
                    flatten_args_schema[i].mesh,
                    arg.placements,
                    flatten_args_schema[i].placements,
                )
            flatten_args_schema[i] = arg._local_tensor  # 更新 flatten_args_schema 中的张量标记

    # 将 flatten_args_schema 还原成原始结构的参数列表
    unflattened_args = tree_unflatten(flatten_args_schema, args_tree_spec)
    return specs, unflattened_args  # 返回更新后的 specs 和参数列表


# When no tensor redistribution is required, we only need to update non-tensor args
# of the node according to op_schema and avoid building a GraphModule just for the
# node.
# 定义一个函数，根据操作模式 op_schema 更新节点中不需要重新分配的非张量参数
def _update_node_from_op_schema(node: torch.fx.Node, op_schema: OpSchema) -> None:
    # 对节点参数进行扁平化，并获取参数树结构
    flat_args, args_tree_spec = tree_flatten(node.args)
    flat_args_schema = pytree.tree_leaves(op_schema.args_schema)
    # ...
    # 定义一个函数，用于判断参数是否为 torch.fx.Node 类型或整数类型
    def is_sym_int_or_int(arg: Union[int, torch.fx.Node]) -> bool:
        # 如果参数是 torch.fx.Node 类型，则检查其目标是否在指定的符号集合中
        if isinstance(arg, torch.fx.Node):
            return arg.target in [
                aten.sym_size,
                aten.sym_numel,
                aten.sym_stride,
            ]
        # 如果参数是整数类型，则返回 True
        return isinstance(arg, int)

    # 断言：验证 flat_args 和 flat_args_schema 的长度是否相等
    assert len(flat_args) == len(flat_args_schema)
    
    # 遍历 flat_args 和 flat_args_schema 中的每个元素，进行处理
    for i, (arg, arg_schema) in enumerate(zip(flat_args, flat_args_schema)):
        # 如果 arg 是 torch.fx.Node 类型且 arg_schema 是整数类型，则将 flat_args 中的元素替换为 arg_schema
        if is_sym_int_or_int(arg) and isinstance(arg_schema, int):
            flat_args[i] = arg_schema

    # 将 flat_args 根据 args_tree_spec 规范重新构建成 args
    args = tree_unflatten(flat_args, args_tree_spec)
    
    # 遍历 args 中的每个元素，通过 node.update_arg 方法更新 node 的参数
    for idx, arg in enumerate(args):
        node.update_arg(idx, arg)
    
    # 函数返回 None
    return None
# 重新映射参数函数，将节点到对象的映射应用到参数中
def _remap_arg(node_to_obj: Dict[fx.Node, Any], arg: Any) -> Any:
    if isinstance(arg, torch.fx.Node):
        obj = node_to_obj[arg]
        if _get_tracer():
            # 如果参数是共享的，已经在之前的跟踪中有一个跟踪器。删除跟踪器。
            del cast(Dict[Any, Any], obj.__dict__)[proxy_slot]
        return obj
    else:
        return arg


# 解包尺寸和维度信息，返回本地尺寸列表和放置信息列表的元组
def unpack_sizes_and_dims(
    sizes: List[Union[DSymInt, int]], mesh: DeviceMesh
) -> Tuple[List[int], List[Placement]]:
    # 将包含DSymInt类型的对象转换为其本地值，保留整数不变
    local_sizes: List[int] = [
        s.local_value if isinstance(s, DSymInt) else s for s in sizes
    ]
    # 为每个尺寸创建一个放置对象列表，如果尺寸是DSymInt类型且是分片类型则创建Shard对象，否则创建Replicate对象
    placements: List[Placement] = [
        Shard(i)
        for i, a in enumerate(sizes)
        if (isinstance(a, DSymInt) and a.is_shard())
    ] or [Replicate()]

    # 断言放置对象的数量与设备网格的维度数量相匹配
    assert len(placements) == mesh.ndim, (
        f"The number of sharded dimensions ({len(placements)}) must "
        f"match number of dimensions in device mesh ({mesh.ndim})."
    )

    return local_sizes, placements


# 二元操作符规则，处理符号整数消费者
def binop_sym_int_consumer_rule(node: fx.Node, args: Tuple[Any, ...]) -> DTensor:
    # 断言参数数量为2，以及第一个参数是DTensor类型
    assert len(args) == 2, f"Expect two args but got op {node.target} with args {args}"
    assert isinstance(
        args[0], DTensor
    ), f"Expect 1st argument to be DTensor but got {args[0]}"
    assert isinstance(args[1], list), f"Expect 2nd argument as list but got {args[1]}"

    # 从尺寸列表中提取分片维度，用于输出的DTensor应该遵循这些放置要求
    local_sizes, placements = unpack_sizes_and_dims(args[1], args[0].device_mesh)

    # 将节点参数设置为真实整数尺寸
    node.args = (node.args[0], local_sizes)
    op = cast(torch._ops.OpOverload, node.target)
    # 返回基于本地数据的DTensor对象
    return DTensor.from_local(
        local_tensor=op(args[0]._local_tensor, local_sizes),
        device_mesh=args[0].device_mesh,
        placements=placements,
        run_check=False,
    )


# 切片反向符号整数消费者规则
def slice_backwad_sym_int_consumer_rule(
    node: fx.Node, args: Tuple[Any, ...]
) -> DTensor:
    # 解包参数：梯度输出，输入尺寸，维度，起始索引，结束索引，步长
    grad_output, input_sizes, dim, start, end, step = args

    # 将包含DSymInt类型的对象转换为其本地值，保留整数不变
    local_sizes: List[int] = [
        s.local_value if isinstance(s, DSymInt) else s for s in input_sizes
    ]

    # 使用零张量创建输入张量，设备为grad_output的设备，数据类型为grad_output的数据类型
    input_tensor = torch.zeros(
        local_sizes, device=grad_output.device, dtype=grad_output.dtype
    )

    # 返回基于本地数据的DTensor对象
    return DTensor.from_local(
        local_tensor=torch.slice_scatter(
            input_tensor, grad_output.to_local(), dim, start, end, step
        ),
        device_mesh=grad_output.device_mesh,
        placements=grad_output.placements,
        run_check=False,
    )


# 具有尺寸信息的工厂规则
def factory_with_sizes_rule(
    node: fx.Node,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    default_mesh: DeviceMesh,
) -> DTensor:
    # 展平参数列表
    flat_args = pytree.arg_tree_leaves(*args)
    # 断言不希望DTensor作为工厂操作的参数，但是获得了这些参数
    assert not any(isinstance(a, DTensor) for a in flat_args), (
        f"Not expect DTensor argument for factory op, but got {node.target} "
        f"with arguments {args}."
    )
    # 确保第一个参数是列表类型，否则触发断言错误
    assert isinstance(args[0], list), f"Expect 2nd argument as list but got {args[1]}"

    # 解包第一个参数，获取本地尺寸和放置信息
    local_sizes, placements = unpack_sizes_and_dims(args[0], default_mesh)
    
    # 更新节点的参数，将本地尺寸和剩余参数作为新参数列表
    node.args = (local_sizes, *args[1:])
    
    # 将节点的目标转换为 OpOverload 类型
    op = cast(torch._ops.OpOverload, node.target)
    
    # 使用 op 对象执行节点的参数操作，生成本地张量
    return DTensor.from_local(
        local_tensor=op(*node.args, **kwargs),
        device_mesh=default_mesh,
        placements=placements,
        run_check=False,
    )
def factory_arange_rule(
    node: fx.Node,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    default_mesh: DeviceMesh,
) -> DTensor:
    # 将节点的参数映射为本地值（如果是 DSymInt 类型的参数则转换为其本地值）
    node.args = tree_map(lambda a: a.local_value if isinstance(a, DSymInt) else a, args)
    # 获取节点的目标操作符，并进行类型转换为 OpOverload
    op = cast(torch._ops.OpOverload, node.target)
    # 使用操作符执行节点的参数和关键字参数，返回本地 DTensor
    return DTensor.from_local(
        local_tensor=op(*node.args, **kwargs),
        device_mesh=default_mesh,
        placements=[Replicate()],
        run_check=False,
    )


def default_factory_op_rule(
    node: fx.Node,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    default_mesh: DeviceMesh,
) -> DTensor:
    # 将节点的参数和关键字参数直接赋值给节点
    node.args, node.kwargs = args, kwargs
    # 获取节点的目标操作符，并进行类型转换为 OpOverload
    op = cast(torch._ops.OpOverload, node.target)
    # 使用操作符执行节点的参数和关键字参数，返回本地 DTensor
    return DTensor.from_local(
        local_tensor=op(*node.args, **node.kwargs),
        device_mesh=default_mesh,
        placements=[Replicate()],
        run_check=False,
    )


# Dispatch override for view and factory ops that consume SymInt arguments,
# where the output spec should follow dimension placement where the SymInt comes
# from.
VIEW_SYM_INT_CONSUMERS: Dict[torch._ops.OpOverload, Callable] = {
    # 为处理使用 SymInt 参数的视图和工厂操作的调度重载
    # 这些操作的输出规范应该根据 SymInt 所在的维度布置来确定
    aten._unsafe_view.default: binop_sym_int_consumer_rule,
    aten.expand.default: binop_sym_int_consumer_rule,
    aten.slice_backward.default: slice_backwad_sym_int_consumer_rule,
    aten.view.default: binop_sym_int_consumer_rule,
}

FACTORY_SYM_INT_CONSUMERS: Dict[torch._ops.OpOverload, Callable] = {
    # 为处理使用 SymInt 参数的工厂操作的调度重载
    aten.full.default: factory_with_sizes_rule,
    aten.arange.default: factory_arange_rule,
    aten.arange.start: factory_arange_rule,
}


# Dispatch override for factory ops, as DTensor cannot propagate sharding spec
# without DTensor inputs.
FACTORY_OPS: Dict[torch._ops.OpOverload, Callable] = {
    # 为处理工厂操作的调度重载，因为 DTensor 无法在没有 DTensor 输入的情况下传播分片规范
    aten.scalar_tensor.default: default_factory_op_rule,
    aten.arange.start: default_factory_op_rule,
    aten.zeros.default: default_factory_op_rule,
}


def _get_dtensor_dispatch_graph(
    node: fx.Node,
    node_to_obj: Dict[fx.Node, Any],
    *,
    force_make_fx: bool = False,
    default_mesh: Optional[DeviceMesh] = None,
) -> Optional[fx.GraphModule]:
    # 略
    pass


def _build_dummy_add_graph(
    dt: DTensor, node_to_obj: Dict[fx.Node, Any]
) -> Tuple[fx.GraphModule, Any]:
    """Create a graph for a dummy add function from a partial DTensor.

    This dummy add is used for triggering all_reduce on a Partial DTensor
    during the DTensor expansion of the traced graph.
    Also returns the actual DTensor after resharding.
    """

    def dummy_add(grad: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
        # 简单的加法函数，用于在部分 DTensor 的 all_reduce 中触发
        return grad + zero

    grad: torch.Tensor = dt._local_tensor
    zero: torch.Tensor = torch.zeros_like(dt._local_tensor)

    # 制作一个函数调用图，用于 dummy_add 函数
    traced_add = make_fx(dummy_add)(grad, zero)

    # 确保图中有正确数量的占位符和函数调用
    placeholders = [n for n in traced_add.graph.nodes if n.op == OP.PLACEHOLDER]
    call_functions = [n for n in traced_add.graph.nodes if n.op == OP.CALL_FUNCTION]
    assert len(placeholders) == 2
    assert len(call_functions) == 1
    # 将第一个占位符关联到实际的 DTensor 对象
    node_to_obj[placeholders[0]] = dt
    # 将 placeholders[1] 对应的节点映射到一个新创建的 DTensor 对象
    node_to_obj[placeholders[1]] = DTensor.from_local(
        zero, dt.device_mesh, [Replicate()], run_check=False
    )
    
    # 使用 _get_dtensor_dispatch_graph 函数获取调度图，这个调度图是基于 call_functions[0] 的执行情况
    # node_to_obj 是一个节点到对象的映射，force_make_fx=True 表示强制生成效果函数图
    traced_dispatch = _get_dtensor_dispatch_graph(
        call_functions[0], node_to_obj, force_make_fx=True
    )
    # 确保 traced_dispatch 不为 None
    assert traced_dispatch is not None
    
    # TODO(anj): 这依赖于调用函数节点到实际 DTensor 输出的映射，我们希望在 SPMD 扩展中避免这种映射
    # 返回 traced_dispatch 和与 call_functions[0] 相关联的节点对象 node_to_obj[call_functions[0]]
    return traced_dispatch, node_to_obj[call_functions[0]]
# 重新构建输出节点，并在有需要时修改其参数
def _convert_output(
    gm: fx.GraphModule,
    node: fx.Node,
    node_to_obj: Dict[fx.Node, Any],
) -> fx.Node:
    # 存储替换后的参数列表
    new_args = []
    # 标志是否存在部分替换
    has_partial = False
    
    # 遍历节点的第一个参数（假设是一个列表），处理其中的每个元素
    for argument in node.args[0]:  # type: ignore[union-attr]
        # 如果参数不是 fx.Node 对象，则直接添加到新参数列表中并继续下一个循环
        if not isinstance(argument, fx.Node):
            new_args.append(argument)
            continue
        
        # 获取参数对应的对象
        obj = node_to_obj[argument]
        
        # 如果对象不是部分 DTensor，则直接添加到新参数列表中并继续下一个循环
        if not _is_partial_dtensor(obj):
            new_args.append(argument)
            continue
        
        # 标记存在部分替换
        has_partial = True
        
        # 强制转换为 DTensor 类型
        dt = cast(DTensor, obj)
        
        # 构建虚拟添加图，并获取结果对象
        traced_dispatch, result_obj = _build_dummy_add_graph(dt, node_to_obj)
        
        # 获取待处理的节点列表：等待通信或等待张量
        wait = [
            n
            for n in traced_dispatch.graph.nodes
            if n.name == "wait_comm" or n.name == "wait_tensor"
        ]
        add = [n for n in traced_dispatch.graph.nodes if n.name == "add"]
        assert len(wait) == 1 and len(add) == 1
        
        # 替换 add 节点并使用 wait 节点
        add[0].replace_all_uses_with(wait[0])
        traced_dispatch.graph.eliminate_dead_code()
        
        # 更新节点对应的最终 DTensor 对象
        node_to_obj[wait[0]] = result_obj
        
        # 值重映射字典，用于映射值节点到新参数
        value_remap: Dict[fx.Node, fx.Node] = {}
        
        # 遍历虚拟添加图的节点
        for dtn in traced_dispatch.graph.nodes:
            # 如果节点是占位符操作，则无需处理，已经在值重映射中准备好
            if dtn.op == OP.PLACEHOLDER:
                value_remap[dtn] = argument
            # 如果节点是输出操作
            elif dtn.op == OP.OUTPUT:
                assert (
                    len(dtn.args) == 1 and len(dtn.args[0]) == 1
                ), f"Expecting single output, but got {dtn.args} {len(dtn.args)}"
                # 将单个输出添加到新参数列表中
                new_args.append(value_remap[dtn.args[0][0]])
                # 将输出节点的具体 DTensor 值添加到节点对象映射中
                node_to_obj[value_remap[dtn.args[0][0]]] = node_to_obj[dtn.args[0][0]]
            else:
                # 如果节点是获取属性操作，则更新 gm 对象的属性
                if dtn.op == OP.GET_ATTR:
                    setattr(
                        gm,
                        dtn.target,
                        getattr(traced_dispatch, dtn.target),
                    )
                # 在当前节点之前插入节点，并更新值重映射
                with gm.graph.inserting_before(node):
                    value_remap[dtn] = gm.graph.node_copy(dtn, lambda n: value_remap[n])
    
    # 如果存在部分替换，则擦除当前节点并返回新参数作为输出节点
    if has_partial:
        gm.graph.erase_node(node)
        return gm.graph.output(new_args)
    else:
        # 否则直接返回当前节点
        return node
    # 在本地跟踪的图中用 DTensor 的调度图替换节点
    gm.graph.eliminate_dead_code()
    # 重新编译图模型
    gm.recompile()
# 定义函数 _get_last_consumer_to_nodes，接收一个 fx.Graph 类型的参数 graph，返回一个字典，
# 其中键为 fx.Node，值为列表类型的 fx.Node。该函数用于获取每个节点的最后一个使用者节点。
def _get_last_consumer_to_nodes(
    graph: fx.Graph,
) -> Dict[fx.Node, List[fx.Node]]:
    # 创建一个空字典，用于记录每个节点的最后一个使用者节点
    node_to_last_consumer: Dict[fx.Node, fx.Node] = {}
    # 创建一个空字典，用于记录每个最后使用者节点所对应的节点列表
    last_consumer_to_nodes: Dict[fx.Node, List[fx.Node]] = {}

    # 定义内部函数 _register_final_consumer，用于注册节点的最后使用者
    def _register_final_consumer(arg_node: fx.Node, consumer: fx.Node) -> None:
        # 如果节点 arg_node 不在 node_to_last_consumer 中，则将其添加进去
        if arg_node not in node_to_last_consumer:
            node_to_last_consumer[arg_node] = consumer
            # 将节点 arg_node 添加到其最后使用者节点 consumer 对应的列表中
            last_consumer_to_nodes.setdefault(consumer, []).append(arg_node)

    # 遍历图中的节点，从后向前遍历
    for node in reversed(graph.nodes):
        # 对节点的位置参数进行映射，调用 _register_final_consumer 函数
        fx.node.map_arg(
            node.args, lambda arg_node: _register_final_consumer(arg_node, node)
        )
        # 对节点的关键字参数进行映射，调用 _register_final_consumer 函数
        fx.node.map_arg(
            node.kwargs,
            lambda kwarg_node: _register_final_consumer(kwarg_node, node),
        )

    # 返回记录每个最后使用者节点的字典 last_consumer_to_nodes
    return last_consumer_to_nodes


# 定义函数 _convert_to_distributed，将给定的图模块 gm 转换为分布式图模块，并返回转换后的结果
# 以及输出名称到 Schema 映射的字典
def _convert_to_distributed(
    gm: fx.GraphModule,
    inps: List[torch.Tensor],
    schemas: List[Schema],
    default_mesh: Optional[DeviceMesh] = None,
    _allow_partial: bool = False,
) -> Tuple[fx.GraphModule, Dict[str, Schema]]:
    """Transform a graph module to a distributed graph module.

    Returns:
        - transformed graph module
        - map from output name to DTensorSpec

    """
    # 设置全局日志记录器为 spmd_exp 的记录器
    global logger
    logger = get_logger("spmd_exp")
    # 获取所有运算符的集合
    operators = {getattr(operator, name) for name in operator.__all__}
    # 创建一个空字典，用于存储节点到对象的映射
    node_to_obj: Dict[fx.Node, Any] = {}
    # 创建一个空字典，用于存储节点替换关系的映射
    node_replacements: Dict[torch.fx.Node, torch.fx.GraphModule] = {}

    # 调用函数 _get_last_consumer_to_nodes 获取最后使用者节点的映射关系
    last_consumer_to_nodes = _get_last_consumer_to_nodes(gm.graph)

    # 创建一个空字典，用于存储输出名称到 Schema 的映射
    output_schemas: Dict[str, Schema] = {}
    # 调用函数 _rebuild_graph 重建图结构，更新节点替换关系
    _rebuild_graph(gm, node_replacements)

    # 返回转换后的图模块 gm 和输出名称到 Schema 的映射字典 output_schemas
    return gm, output_schemas
```