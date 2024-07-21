# `.\pytorch\torch\distributed\pipelining\_IR.py`

```
# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
# 导入必要的库和模块
import copy  # 导入copy模块，用于复制对象
import logging  # 导入logging模块，用于日志记录
import operator  # 导入operator模块，用于操作符函数的集合
from collections import defaultdict  # 导入defaultdict类，用于创建默认值为列表的字典
from enum import Enum  # 导入Enum类，用于创建枚举类型
from inspect import Parameter, Signature, signature  # 导入inspect模块中的Parameter、Signature、signature类和函数
from types import MethodType  # 导入MethodType类，用于在类的实例中绑定方法
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union  # 导入多个类型提示类和函数

import torch  # 导入PyTorch深度学习库
import torch.fx as fx  # 导入PyTorch的FX模块
from torch.distributed import ProcessGroup  # 导入ProcessGroup类，用于分布式训练中进程组的管理
from torch.export import ExportedProgram  # 导入ExportedProgram类，用于导出的程序
from torch.export.unflatten import (  # 导入unflatten模块中的多个函数和类
    _assign_attr,
    _AttrKind,
    _sink_params,
    InterpreterModule,
)
from torch.fx.node import map_aggregate  # 导入map_aggregate函数，用于FX图节点的聚合处理
from torch.fx.passes.split_module import split_module  # 导入split_module函数，用于模块的拆分处理

from ._backward import _null_coalesce_accumulate, stage_backward  # 导入自定义模块中的函数和类
from ._unflatten import _outline_submodules  # 导入自定义模块中的函数
from ._utils import PipeInfo  # 导入自定义模块中的PipeInfo类
from .stage import _PipelineStage  # 导入自定义模块中的_PipelineStage类


logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象

# TODO:
# 1. investigate gradient sync for shared parameters. how does DDP do it?
#    研究共享参数的梯度同步问题，了解分布式数据并行（DDP）如何实现
# 2. Add parameter movement to split_module
#    将参数移动到split_module中


def _find_loss_from_output_and_spec(output_val, spec_val):
    # 根据输出值和规范值查找损失值

    # 如果规范值为False，则返回None
    if spec_val is False:
        return None
    # 如果规范值为True
    if spec_val is True:
        # 如果输出值不是fx.Node类型，则抛出运行时错误
        if not isinstance(output_val, fx.Node):
            raise RuntimeError(
                f"Loss spec must specify a dynamic value but got {output_val}"
            )
        return output_val  # 返回输出值作为损失值

    # 如果规范值为tuple或list类型
    if isinstance(spec_val, (tuple, list)):
        # 如果输出值不是tuple或list类型，则抛出运行时错误
        if not isinstance(output_val, (tuple, list)):
            raise RuntimeError(
                f"Output value {output_val} must match type of loss specification "
                f"{spec_val}"
            )
        # 如果输出值和规范值长度不匹配，则抛出运行时错误
        if len(output_val) != len(spec_val):
            raise RuntimeError(
                f"Output value {output_val} must match length of loss specification "
                f"{spec_val}"
            )
        # 遍历输出值和规范值，递归查找损失值
        for out, spec in zip(output_val, spec_val):
            loss_val = _find_loss_from_output_and_spec(out, spec)
            if loss_val is not None:
                return loss_val
        # 如果未找到损失值，则抛出运行时错误
        raise RuntimeError(f"Did not find loss value in specification {spec_val}")

    # 如果规范值为dict类型
    if isinstance(spec_val, dict):
        # 如果输出值不是dict类型，则抛出运行时错误
        if not isinstance(output_val, dict):
            raise RuntimeError(
                f"Output value {output_val} must match type of loss specification "
                f"{spec_val}"
            )
        # 如果输出值的键集合与规范值的键集合不匹配，则抛出运行时错误
        if set(output_val.keys()) != set(spec_val.keys()):
            raise RuntimeError(
                f"Output value {output_val} must match keys of loss specification "
                f"{spec_val}"
            )
        # 遍历规范值的键，递归查找损失值
        for k in spec_val:
            loss_val = _find_loss_from_output_and_spec(output_val[k], spec_val[k])
            if loss_val is not None:
                return loss_val
        # 如果未找到损失值，则抛出运行时错误
        raise RuntimeError(f"Did not find loss value in specification {spec_val}")

    # 如果规范值类型不受支持，则抛出运行时错误
    raise RuntimeError(f"Unsupported type {type(spec_val)} in loss specification")


def _find_loss_output(mod: torch.nn.Module, g: fx.Graph, output_loss_value_spec):
    # 查找损失输出
    # 从图形 g 中找到所有 op 属性为 "output" 的节点，存入列表 output_nodes
    output_nodes = [n for n in g.nodes if n.op == "output"]
    # 确保 output_nodes 列表中只有一个节点
    assert len(output_nodes) == 1
    # 获取 output_nodes 中的唯一节点，存入 output_node
    output_node = output_nodes[0]
    # 获取 output_node 的第一个参数，存入 output_val
    output_val = output_node.args[0]
    # 初始化 generated_spec 变量为 None
    generated_spec: Any = None

    # 如果 mod 是 TrivialLossWrapper 类型
    if isinstance(mod, TrivialLossWrapper):
        # TrivialLossWrapper 是 PiPPy 预定义的类
        # 该类的唯一输出是损失值，因此可以安全地假设第一个输出参数是损失值
        assert len(output_node.args) == 1
        # 将 output_val 视为损失节点
        loss_node = output_val
        # 设置 generated_spec 为 TrivialLossWrapper 类的损失规范
        generated_spec = TrivialLossWrapper.loss_spec
    # 如果 output_loss_value_spec 为 None
    elif output_loss_value_spec is None:
        # 使用默认规范，在输出值中搜索 "loss"
        if isinstance(output_val, dict) and "loss" in output_val.keys():
            # 如果 output_val 是字典且包含 "loss" 键，则将其视为损失节点
            loss_node = output_val["loss"]
            # 生成 generated_spec，标记包含 "loss" 键的所有输出值
            generated_spec = {k: k == "loss" for k in output_val}
        else:
            # 否则，损失节点和 generated_spec 都设置为 None
            loss_node = None
            generated_spec = None
    else:
        # 从 output_val 和 output_loss_value_spec 中查找损失节点
        loss_node = _find_loss_from_output_and_spec(output_val, output_loss_value_spec)
        # 设置 generated_spec 为 output_loss_value_spec
        generated_spec = output_loss_value_spec

    # 返回损失节点、输出节点和生成的规范 generated_spec
    return loss_node, output_node, generated_spec
# 在图 g 中插入符号反向传播阶段
def _insert_stage_symbolic_backward(
    g: fx.Graph,
    loss_node: fx.Node,
    output_node: fx.Node,
):
    # 收集关于元组输出值的元数据。TODO: 将此移到 split_module 或 FX IR
    tuples: Dict[fx.Node, Tuple] = {}
    
    # 反向遍历图中的节点
    for node in reversed(g.nodes):
        if node.op == "call_function":
            # 在前向传播中，仅生成占位符、模块调用和getitem调用。如果在这段（仅限前向的）代码中有除getitem之外的目标，则存在错误。
            assert node.target == operator.getitem, (
                "Found non-getitem call in forward pass. "
                "Please report a bug to PiPPy"
            )
            assert (
                len(node.args) == 2
            ), "Found malformed getitem call. Please report a bug to PiPPy"
            indexed_value, node_idx = tuple(node.args)

            # indexed_value 是我们正在索引的集合。如果我们已经处理了另一个 `getitem`，则它可能已经存在于 tuples 映射中。
            existing_list_size = (
                len(tuples[indexed_value]) if indexed_value in tuples else -1
            )
            new_list_size = max(node_idx + 1, existing_list_size)

            # 构建重建后的列表，用于存储节点
            reconstructed_list = [None for _ in range(new_list_size)]

            # 如果存在，则复制现有元素
            if indexed_value in tuples:
                for i, val in enumerate(tuples[indexed_value]):
                    reconstructed_list[i] = val

            # 将该节点表示的值填入重建后的列表中
            reconstructed_list[node_idx] = node

            tuples[indexed_value] = tuple(reconstructed_list)

    # 跟踪能够支配损失节点的节点。
    # 我们仅对能够对指定损失值产生影响的节点生成反向操作。
    live_nodes = {loss_node: None}
    val_to_grad: Dict[fx.Node, Optional[fx.Node]] = {loss_node: None}

    def assign_or_accumulate_grad(forward_node, grad_value):
        # 如果 forward_node 在 val_to_grad 中，并且不是占位符，则使用 _null_coalesce_accumulate 函数累积梯度值。
        if forward_node in val_to_grad and forward_node.op != "placeholder":
            grad_value = g.call_function(
                _null_coalesce_accumulate,
                (val_to_grad[forward_node], grad_value),
            )
        val_to_grad[forward_node] = grad_value
    # 在输出节点之前插入操作节点
    with g.inserting_before(output_node):
        # 遍历反转后的图中的每个节点
        for node in reversed(g.nodes):
            # 如果节点不在活跃节点列表中，则跳过
            if node not in live_nodes:
                continue
            
            # 定义一个函数，将节点添加到活跃节点字典中
            def add_to_live_nodes(n):
                live_nodes.setdefault(n, None)
            
            # 将节点的位置参数映射到活跃节点字典
            fx.node.map_arg(node.args, add_to_live_nodes)
            # 将节点的关键字参数映射到活跃节点字典
            fx.node.map_arg(node.kwargs, add_to_live_nodes)
            
            # 如果节点的操作是调用模块
            if node.op == "call_module":
                # 定义一个变量来存储输出梯度
                output_grads: Union[Tuple[Optional[fx.Node], ...], Optional[fx.Node]]
                
                # 如果节点在元组中
                if node in tuples:
                    # 取出与节点相关的元组
                    stage_output = tuples[node]
                    # 根据值到梯度映射获取输出梯度
                    output_grads = tuple(val_to_grad.get(n, None) for n in tuples[node])
                    # 获取具有梯度的输出索引列表
                    outputs_with_grads_idxs = [
                        i for i, n in enumerate(tuples[node]) if n in live_nodes
                    ]
                else:
                    # 如果节点不在元组中，则以节点本身为元组
                    stage_output = (node,)
                    # 获取节点的值到梯度映射
                    output_grads = val_to_grad[node]
                    # 指定只有一个输出具有梯度
                    outputs_with_grads_idxs = [0]
                
                # 将输出梯度转换为元组形式（如果尚未是元组）
                output_grads = (
                    (output_grads,)
                    if not isinstance(output_grads, tuple)
                    else output_grads
                )
                
                # 在图中调用函数，执行反向传播阶段
                grad_call = g.call_function(
                    stage_backward,
                    kwargs={
                        "stage_output": stage_output,
                        "output_grads": output_grads,
                        "input_values": list(node.all_input_nodes),
                        "outputs_with_grads_idxs": outputs_with_grads_idxs,
                    },
                )
                
                # 复制关键字参数以插入反向传播调试信息
                kwargs_copy = dict(grad_call.kwargs)
                grad_call.kwargs = kwargs_copy
                
                # 使用 Proxy 对象获取反向传播阶段的梯度节点
                grad_call_proxy = fx.Proxy(grad_call)
                grads = grad_call_proxy.node
                
                # 获取节点的所有输入节点
                input_nodes = list(node.all_input_nodes)
                grads_proxy = fx.Proxy(grads)
                # 遍历输入节点，并将梯度分配或累积到对应的输入节点上
                for i, input_node in enumerate(input_nodes):
                    assign_or_accumulate_grad(input_node, grads_proxy[i].node)
    
    # 返回修改后的图
    return g
class PipeSequential(torch.nn.Sequential):
    @staticmethod
    def from_sequential(sequential_instance: torch.nn.Sequential):
        # 从给定的 torch.nn.Sequential 实例创建一个 PipeSequential 实例
        return PipeSequential(*[copy.copy(m) for m in sequential_instance])

    def forward(self, input):
        # 对序列中的每个模块进行前向传播
        for i, module in enumerate(self):
            input = module(input)
            # 如果不是最后一个模块，则调用 pipe_split()
            if i != len(self) - 1:
                pipe_split()
        return input


class LossWrapper(torch.nn.Module):
    """
    LossWrapper 是一个方便的抽象类，允许您包装模型及其损失函数，并指定输入、模型、损失函数和输出值之间的连接方式。示例::

        class MyModelWrapper(LossWrapper):
            def forward(self, x, targets):
                model_out = self.module(x)
                loss_value = self.loss_fn(model_out, targets)
                return loss_value

    上述示例定义了一个连接方式，我们期望前向传播/损失计算/反向传播训练过程接受两个参数（x 和 targets），将 x 输入模型得到前馈计算输出，
    将模型输出和目标值传递给损失函数，得到损失值，并返回该损失值，PiPPy 将对其进行反向传播。上述类可以像这样实例化::

        model = ...  # 实例化模型
        loss_fn = torch.nn.MSELoss()  # 仅作为示例

        wrapper = MyModelWrapper(model, loss_fn)
        pipe = Pipe.from_tracing(wrapper, ...)

    """

    def __init__(self, module, loss_fn):
        super().__init__()
        self.module = module
        self.loss_fn = loss_fn

    def forward(self, *args, **kwargs):
        # 抽象方法，需要子类实现具体的前向传播逻辑
        raise NotImplementedError(
            "This instance of LossWrapper does not have an overridden"
            "forward(). Please implement forward() to specify the arguments, "
            "connection between the module and loss, and loss output "
            "value."
        )


class TrivialLossWrapper(LossWrapper):
    def forward(self, x, targets):
        # 调用模型计算输出，并使用损失函数计算损失值
        model_out = self.module(x)
        return self.loss_fn(model_out, targets)

    loss_spec = True


# 管道模型表示
#
# 管道模型可以被视为 `nn.Sequential++`。也就是说：它指定了管道“阶段”的单一拓扑排序，按顺序运行这些阶段将构成程序的所有操作。
# 然而，与 `nn.Sequential` 不同，管道允许非局部值的使用，只要这些使用仍然遵循拓扑排序。特别是：
#
# 1. 非局部激活。这种类型的使用可以出现在例如跳跃连接中。这些值将从“def”阶段直接传输到所有使用它们的阶段，跳过中间阶段。
#    在自动求导期间，梯度将沿着这种跳跃连接的反向传播，与激活在前向传播中传播的方式相反。
# Register `_pipe_split()` as an ATen operator. This is necessary for Export to
# preserve this marker in the graph.
torch.library.define("pippy::_pipe_split", "() -> ()")

# Implementation of `_pipe_split()` as a no-operation function.
@torch.library.impl("pippy::_pipe_split", "BackendSelect")
def _pipe_split():
    return None

# Register `_pipe_split()` as a fake function to ensure its existence.
@torch.library.register_fake("pippy::_pipe_split")  # type: ignore[no-redef]
def _pipe_split():  # noqa: F811
    return None

# Add an alias `aten_pipe_split_alias` for `_pipe_split()` for convenience.
aten_pipe_split_alias = torch.ops.pippy._pipe_split.default

# Instruct Export to preserve the `_pipe_split` op during optimization.
# See examples in pytorch/torch/fx/node.py
fx.node._side_effectful_functions.add(aten_pipe_split_alias)

# Definition of `pipe_split()`, a user-facing API to mark module stages.
def pipe_split():
    """
    pipe_split is a special operator that is used to mark the boundary between
    stages in a module. It is used to split the module into stages. It is a
    no-op if your annotated module is run eagerly.

    Example:
        >>> # xdoctest: +SKIP
        >>> def forward(self, x):
        >>>     x = torch.mm(x, self.mm_param)
        >>>     x = torch.relu(x)
        >>>     pipe_split()
        >>>     x = self.lin(x)
        >>>     return x

    The above example will be split into two stages.
    """
    return torch.ops.pippy._pipe_split()

# Definition of `MultiUseParameterConfig`, an enumeration for parameter handling.
class MultiUseParameterConfig(Enum):
    TRANSMIT = 1
    REPLICATE = 2

# Type specification for `MultiUseParamSpec`, allowing both enum and dictionary input.
MultiUseParamSpec = Union[MultiUseParameterConfig, Dict[str, MultiUseParameterConfig]]

# Definition of `DetachExecutor`, a specialized interpreter for detached execution.
class DetachExecutor(fx.Interpreter):
    """
    Special interpreter to run the split_gm in testing that detaches all inputs to
    a module invocation. This is needed so that the values at the boundary are
    leaf modules in autograd execution.
    """

    def __init__(self, module, garbage_collect_values=True):
        garbage_collect_values = False  # Disable garbage collection of values.
        super().__init__(module, garbage_collect_values)
        self.value_remap = {}  # Initialize a dictionary for value remapping.

    def run(self, *args, initial_env=None):
        self.value_remap = {}  # Reset the value remapping dictionary.
        return super().run(*args, initial_env=initial_env)
    # 调用模块的方法，传入目标函数、参数和关键字参数
    def call_module(self, target, args, kwargs):
        # 定义内部函数，用于递归地处理需要分离梯度的张量
        def detach_tensors(a):
            # 如果是需要梯度的张量且尚未处理过，则进行分离并存储映射关系
            if isinstance(a, torch.Tensor) and a.requires_grad:
                if a not in self.value_remap:
                    new_val = a.detach().requires_grad_(True)
                    self.value_remap[a] = new_val
                return self.value_remap[a]
            else:
                return a

        """
        def dont_traverse_size(a):
            return type(a) != torch.Size
        """

        # 使用 `detach_tensors` 函数处理 `args` 中的每个元素
        args = map_aggregate(
            args,
            detach_tensors,  # dont_traverse_size
        )
        # 使用 `detach_tensors` 函数处理 `kwargs` 中的每个元素
        kwargs = map_aggregate(
            kwargs,
            detach_tensors,  # dont_traverse_size
        )

        # 调用父类方法 `call_module`，传入处理后的参数和关键字参数
        return super().call_module(target, args, kwargs)

    # 调用函数的方法，传入目标函数、参数和关键字参数
    def call_function(self, target, args, kwargs):
        # 用于将保存的输入张量重定向到已分离的版本的 HACK 方法
        if target == stage_backward:
            # 复制关键字参数 `kwargs`，以防止修改原始参数
            kwargs = dict(kwargs)
            # 将 `kwargs` 中的 `input_values` 替换为分离版本或保留映射的张量
            kwargs["input_values"] = [
                self.value_remap.get(v, v) for v in kwargs["input_values"]
            ]
        # 调用父类方法 `call_function`，传入处理后的目标函数、参数和关键字参数
        return super().call_function(target, args, kwargs)
class _NodeReference:
    # _NodeReference 类，表示一个节点的引用，具有名称属性
    def __init__(self, name):
        self.name = name

    # 节点名称，字符串类型
    name: str


class _LinearNodeList:
    # _LinearNodeList 类，线性节点列表
    def __init__(self, node_list):
        self.serialize_node_list = []
        # 遍历给定的节点列表，将每个节点转换为 _NodeReference 实例，并序列化为新的节点列表
        for node in node_list:
            node_args = fx.node.map_arg(node.args, lambda n: _NodeReference(n.name))
            node_kwargs = fx.node.map_arg(node.kwargs, lambda n: _NodeReference(n.name))
            serialize_node = fx.Node(
                graph=None,
                name=node.name,
                op=node.op,
                target=node.target,
                args=node_args,
                kwargs=node_kwargs,
                return_type=node.type,
            )
            serialize_node.meta = copy.copy(node.meta)
            self.serialize_node_list.append(serialize_node)

    # 将序列化的节点列表转换为图对象
    def to_graph(self):
        graph = fx.Graph()

        ref_str_to_node: Dict[str, fx.Node] = {}

        # 将 _NodeReference 转换为实际的节点对象
        def ref_to_node(arg):
            if isinstance(arg, _NodeReference):
                return ref_str_to_node[arg.name]
            else:
                return arg

        # 在图中创建每个节点，并建立引用映射
        for node in self.serialize_node_list:
            node_args = map_aggregate(node.args, ref_to_node)
            node_kwargs = map_aggregate(node.kwargs, ref_to_node)
            deser_node = graph.create_node(
                op=node.op,
                target=node.target,
                args=node_args,
                kwargs=node_kwargs,
                name=node.name,
                type_expr=node.type,
            )
            ref_str_to_node[node.name] = deser_node

        return graph


def _direct_serialization_deserialize(body, nodes):
    """
    Custom `__reduce__` method for serialization.
    DO AS I SAY -- NOT AS I DO. This violates the principle that
    GraphModules serialize via code export & re-tracing. We allow
    for this here because **PIPE STAGES SHOULD NOT BE PERSISTED
    TO DISK -- THIS IS ONLY FOR TRANSMISSION VIA RPC**. Persisting
    these instances to disk will expose internal implementation
    details of `fx.Graph` and related data structures and is
    NOT advised.
    """
    # 创建一个虚拟模块，用于反序列化
    class DummyModule(torch.nn.Module):
        def __init__(self, body):
            super().__init__()
            self.__dict__.update(body)

    # 使用给定的主体数据创建虚拟模块
    dummy = DummyModule(body)

    # 使用反序列化函数和节点图返回一个 GraphModule 对象
    return fx.GraphModule(dummy, nodes.to_graph())


def _direct_serialization_reduce(self):
    # 创建一个包含对象所有属性的字典副本
    serialization_dict = dict(self.__dict__)
    # 删除字典中的 _graph 键
    serialization_dict.pop("_graph")
    # 返回一个元组，包含反序列化函数和序列化字典以及节点列表的线性化表示
    return (
        _direct_serialization_deserialize,
        (serialization_dict, _LinearNodeList(self.graph.nodes)),
    )


def _modify_graph_op_device(
    gm: torch.fx.GraphModule,
    new_device: torch.device,
):
    """
    Modify the device argument of all "call_function" nodes in the graph.  This
    is useful for moving the graph to a different device. In particular for
    generator ops, like torch.ones.
    """
    # 修改图中所有“call_function”节点的设备参数，将图移动到不同的设备上
    modified = False
    # 遍历计算图中的每个节点
    for node in gm.graph.nodes:
        # 如果节点是通过函数调用产生的
        if node.op == "call_function":
            # 检查节点的关键字参数中是否有 "device"，并且其值不等于新的设备
            if "device" in node.kwargs and node.kwargs["device"] != new_device:
                # 记录调试信息，说明正在将节点的设备从旧设备更改为新设备
                logger.debug(
                    f"Changing device of Node {node.name} from {node.kwargs['device']} to {new_device}"  # noqa: G004
                )
                # 更新节点的 "device" 关键字参数为新设备
                node.update_kwarg("device", new_device)
                # 设置修改标志为 True，表示有节点被修改过
                modified = True
        # 如果节点是通过模块调用产生的
        elif node.op == "call_module":
            # 递归地修改子模块中的 "device"
            submod = gm.get_submodule(node.target)
            # 如果子模块是 torch.fx.GraphModule 类型
            if isinstance(submod, torch.fx.GraphModule):
                # 递归调用 _modify_graph_op_device 函数来修改子模块中的设备
                _modify_graph_op_device(submod, new_device)
            # 如果子模块是 InterpreterModule 类型
            elif isinstance(submod, InterpreterModule):
                # 如果已经执行了解扁平化操作，需要通过 `.graph_module` 访问其图模块
                _modify_graph_op_device(submod.graph_module, new_device)
            else:
                # 记录警告信息，说明跳过修改子模块设备，因为其类型未知
                logger.warning(
                    f"Skipping device modification for submodule {node.target} because it is a {type(submod)}"  # noqa: G004
                )

    # 如果有节点被修改过
    if modified:
        # 重新编译计算图模块 gm
        gm.recompile()
    # 定义一个名为 Pipe 的类，继承自 torch.nn.Module
    class Pipe(torch.nn.Module):
        # 初始化方法，接受以下参数：
        # - split_gm: fx.GraphModule 类型，表示分割的图模块
        # - num_stages: int 类型，表示阶段数目
        # - has_loss_and_backward: bool 类型，表示是否有损失和反向传播
        # - loss_spec: 未指定类型的损失规范
        def __init__(
            self,
            split_gm: fx.GraphModule,
            num_stages: int,
            has_loss_and_backward: bool,
            loss_spec,
        ):
            # 调用父类的初始化方法
            super().__init__()

        # 前向传播方法，接受任意数量的位置参数和关键字参数
        def forward(self, *args, **kwargs):
            # 默认使用位置参数作为执行器的参数
            executor_args = args
            # 如果关键字参数不为空，则根据节点信息生成参数列表
            if len(kwargs) > 0:
                parameters = []
                # 遍历分割图模块中的节点
                for node in self.split_gm.graph.nodes:
                    # 如果节点操作为 "placeholder"
                    if node.op == "placeholder":
                        # 如果节点有参数并且参数列表不为空，则将参数添加到列表中
                        if node.args and len(node.args) > 0:
                            parameters.append(
                                Parameter(
                                    node.target,
                                    Parameter.POSITIONAL_OR_KEYWORD,
                                    default=node.args[0],
                                )
                            )
                        # 否则根据节点类型决定参数种类，并将其添加到列表中
                        else:
                            parameter_kind = Parameter.POSITIONAL_OR_KEYWORD
                            param_name = node.target
                            if node.target.startswith("**"):
                                parameter_kind = Parameter.VAR_KEYWORD  # type: ignore[assignment]
                                param_name = param_name[2:]
                            elif node.target.startswith("*"):
                                parameter_kind = Parameter.VAR_POSITIONAL  # type: ignore[assignment]
                                param_name = param_name[1:]
                            parameters.append(Parameter(param_name, parameter_kind))
                # 根据参数列表生成签名对象
                signature = Signature(parameters)
                # 绑定实际传入的参数到签名对象
                ba = signature.bind(*args, **kwargs)
                # 应用默认值
                ba.apply_defaults()
                # 将绑定后的参数作为执行器的参数
                executor_args = ba.arguments.values()  # type: ignore[assignment]

            # 调用执行器的 run 方法并传入参数，获取结果
            res = self.executor.run(*executor_args)

            # 返回执行结果
            return res

        # 返回指定阶段索引处的子模块
        def get_stage_module(self, stage_idx: int) -> torch.nn.Module:
            """
            Return a stage module corresponding to `stage_idx` of the `pipe`.
            """
            # 如果阶段索引小于 0 或大于等于阶段数目，则抛出 ValueError 异常
            if stage_idx < 0 or stage_idx >= self.num_stages:
                raise ValueError(f"Invalid stage index {stage_idx}!")
            # 返回 split_gm 对象中对应的子模块
            return getattr(self.split_gm, f"submod_{stage_idx}")

        # 静态方法，用于计算给定图模块中前向阶段的数量
        @staticmethod
        def _number_and_count_forward_stages(gm: fx.GraphModule):
            num_stages = 0  # 初始化阶段数量为 0
            found_idxs: Dict[int, None] = {}  # 使用字典记录找到的阶段索引
            # 遍历图模块中的节点
            for node in gm.graph.nodes:
                # 如果节点操作为 "call_module" 并且目标以 "submod_" 开头
                if node.op == "call_module" and node.target.startswith("submod_"):
                    # 提取阶段索引并存储在节点的元数据中
                    node.meta["stage_idx"] = int(node.target[len("submod_"):])
                    found_idxs.setdefault(node.meta["stage_idx"])  # 记录找到的阶段索引
                    num_stages += 1  # 增加阶段数量计数

            # 返回计算得到的阶段数量
            return num_stages
    # 从跟踪过的模型生成管道
    def _from_traced(
        mod: torch.nn.Module,
        exported_program: ExportedProgram,
        multi_use_param_spec: Optional[MultiUseParamSpec] = None,
        output_loss_value_spec=None,
        split_policy: Optional[
            Callable[[torch.fx.GraphModule], torch.fx.GraphModule]
        ] = None,
    ):
        # 打印可读的格式化管道
        def print_readable(self):
            """
            Print the pipe in a human-readable format.
            This will print both the root pipe and each stage module.
            """
            self.split_gm.print_readable()

        @staticmethod
        # 使用导出功能对模型进行跟踪
        def _trace_with_export(
            mod: torch.nn.Module,
            example_args: Tuple[Any, ...],
            example_kwargs: Optional[Dict[str, Any]] = None,
        ) -> ExportedProgram:
            logger.info("Tracing model ...")
            try:
                # 尝试导出模型作为一个完整的图形
                ep = torch.export.export(
                    mod,
                    example_args,
                    example_kwargs,
                )
            except Exception as e:
                # 如果无法捕获模型作为完整图形的情况，抛出运行时错误
                raise RuntimeError(
                    "It seems that we cannot capture your model as a full graph. "
                    "Typical reasons include graph breaks, data/shape-dependent "
                    "control flow, or missing meta kernels for custom operators. "
                    "You can use our manual pipeline interfaces, or try to fix the "
                    "graph breaks, see https://pytorch.org/docs/stable/export.html"
                ) from e

            return ep

        @staticmethod
        # 从跟踪中生成管道
        def from_tracing(
            mod: torch.nn.Module,
            example_args: Tuple[Any, ...],
            example_kwargs: Optional[Dict[str, Any]] = None,
            split_policy: Optional[Callable[[fx.GraphModule], fx.GraphModule]] = None,
        ):
        # If a param will be used in multiple pipeline stages, we default the strategy to REPLICATE'ing the param across
        # stages instead of TRANSMIT'ting it
        multi_use_param_spec = MultiUseParameterConfig.REPLICATE

        # Figure out which output is loss from output_chunk_spec
        output_loss_value_spec: Any = None
        # Deprecated
        """
        if output_chunk_spec is not None:
            output_loss_value_spec = map_aggregate(
                output_chunk_spec, lambda v: isinstance(v, _LossReducer)
            )
        """

        # Trace with export
        exported_program = Pipe._trace_with_export(
            mod,
            example_args,
            example_kwargs,
        )

        # Create a pipeline from the traced model
        pipe = Pipe._from_traced(
            mod,
            exported_program,
            multi_use_param_spec,
            output_loss_value_spec=output_loss_value_spec,
            split_policy=split_policy,
        )

        # Users want the first pipeline stage to accept kwargs if the original
        # program does. This is controlled by the `_codegen` field of the graph,
        # so we make a copy here. Note: we only want the input spec and not the
        # output spec, because the output spec is for the last stage. Maybe a
        # TODO? Not sure yet.
        split = pipe.split_gm
        traced = exported_program.module()
        submod0 = next(iter(split.children()))
        submod0_sign = signature(submod0.forward)
        model_sign = signature(traced.forward)
        if len(model_sign.parameters) != len(submod0_sign.parameters):
            # We don't change the signature of the first stage if it takes
            # different number of args than original model
            logger.info(
                f"Original model takes {len(model_sign.parameters)} args but the "  # noqa: G004
                f"first pipeline stage takes {len(submod0_sign.parameters)}. "
                "Please provide args to respective pipeline stages."
            )
        else:
            # Support kwargs for the first stage
            submod0.graph._codegen = copy.deepcopy(traced.graph._codegen)
            # `_replace` is actually not "private" or internal. based on this doc:
            # To prevent conflicts with field names, the method and attribute names
            # start with an underscore
            submod0.graph._codegen.pytree_info = (
                submod0.graph._codegen.pytree_info._replace(out_spec=None)
            )
            submod0.recompile()

        # Return the constructed pipeline
        return pipe

    # Define the string representation of the class instance
    def __str__(self):
        return self.split_gm.__str__()

    # Define the representation of the class instance
    def __repr__(self):
        return self.split_gm.__repr__()
    # 获取管道的信息。
    def info(self) -> PipeInfo:
        """
        Get information about the pipe.

        Returns
        -------
        PipeInfo
            A dataclass containing information about the pipe.
        """
        # 返回一个包含管道信息的数据类实例
        return PipeInfo(
            graph=self.split_gm.graph,
            num_stages=self.num_stages,
            has_loss_and_backward=self.has_loss_and_backward,
        )

    # 构建管道阶段。
    def build_stage(
        self,
        stage_index: int,
        device: torch.device,
        group: Optional[ProcessGroup] = None,
    ) -> _PipelineStage:
        """
        Create a `PipelineStage` given a stage index and distributed group.
        The `PipelineStage` can run with `PipelineSchedule`s.
        """
        # 获取阶段模块
        stage_module = self.get_stage_module(stage_index)

        # 将操作参数移动到指定设备上
        # 当前的 PT2 追踪器不会将 `x.device` 视为符号设备；
        # 而是在生成的代码中烧录了追踪时的设备。这里提供一个解决方案，允许用户手动修改操作的 "device" 关键字参数。
        # 此类操作可能包括：`torch.ones`, `torch.zeros`, `torch.rand` 等。
        if isinstance(stage_module, torch.fx.GraphModule):
            _modify_graph_op_device(stage_module, device)
        else:
            logger.warning(
                f"Expected a `torch.fx.GraphModule` but got {type(stage_module)}"  # noqa: G004
            )

        # 分离管道信息
        # 注意：要小心包含在 `pipe_info` 中的内容。我们不希望保留对 `Pipe` 或 `Pipe.split_gm` 的引用，
        # 因为这会阻止 Python 回收它们。当 Python 回收它们时，其他阶段模块（对当前排名无关的模块）可以自动释放。
        pipe_info = self.info()
        # 返回一个 `_PipelineStage` 实例，表示构建完成的管道阶段
        return _PipelineStage(stage_module, stage_index, pipe_info, device, group)
class SplitPoint(Enum):
    BEGINNING = 1  # 定义枚举类型，表示分割点在函数开始
    END = 2  # 定义枚举类型，表示分割点在函数结束


# 为了向后兼容，保留了 PipeSplitWrapper 类，因为之前在这个类中定义了 `class SplitPoint`
class PipeSplitWrapper:
    # 创建一个类别名以供向后兼容
    SplitPoint = SplitPoint  # 引用上面定义的枚举类型 SplitPoint


def _split_before_forward(self, *args, **kwargs):
    pipe_split()  # 调用 pipe_split 函数
    return self._orig_forward(*args, **kwargs)


def _split_after_forward(self, *args, **kwargs):
    try:
        return self._orig_forward(*args, **kwargs)
    finally:
        pipe_split()  # 在 finally 语句中调用 pipe_split 函数


def annotate_split_points(mod: torch.nn.Module, spec: Dict[str, SplitPoint]):
    # TODO: make this implementation out-of-place?  # 标记：是否将此实现改为非原地操作？
    for qualname, split_type in spec.items():  # 遍历给定的 spec 字典中的每一个条目
        atoms = qualname.split(".")  # 使用点号分割限定名称字符串
        predecessor_module = mod  # 将当前模块设为前驱模块
        for i, atom in enumerate(atoms[:-1]):  # 遍历限定名称的所有部分（除了最后一个）
            try:
                predecessor_module = getattr(predecessor_module, atom)  # 获取属性或子模块
            except AttributeError as e:
                raise AttributeError(
                    f"Specified target {qualname} referenced "
                    f'nonexistent module {".".join(atoms[: i + 1])}'
                ) from e  # 抛出异常，指出引用了不存在的模块

        mod_to_wrap = getattr(predecessor_module, atoms[-1])  # 获取要包装的模块对象
        mod_to_wrap._orig_forward = mod_to_wrap.forward  # 将原始的 forward 方法保存在 _orig_forward 属性中
        if split_type == SplitPoint.BEGINNING:
            mod_to_wrap.forward = MethodType(_split_before_forward, mod_to_wrap)  # 设置模块的 forward 方法为 _split_before_forward
        elif split_type == SplitPoint.END:
            mod_to_wrap.forward = MethodType(_split_after_forward, mod_to_wrap)  # 设置模块的 forward 方法为 _split_after_forward
        else:
            raise ValueError("Unknown split point type.")  # 如果分割点类型未知，则抛出值错误异常


def pipeline(
    module: torch.nn.Module,
    mb_args: Tuple[Any, ...],
    mb_kwargs: Optional[Dict[str, Any]] = None,
    split_spec: Optional[Dict[str, SplitPoint]] = None,
    split_policy: Optional[Callable[[fx.GraphModule], fx.GraphModule]] = None,
) -> Pipe:
    """
    Split a module based on a specification.

    See `Pipe` for more details.

    Arguments
    ---------
    module:
        The module to be splitted.
    mb_args:
        Example positional inputs, in micro-batch form.
    mb_kwargs:
        Example keyword inputs, in micro-batch form. (default: `None`)
    split_spec:
        A dictionary using submodule names as split marker. (default: `None`)
    split_policy:
        The policy to use for splitting the module. (default: `None`)

    Returns
    -------
    A pipeline representation of class `Pipe`.
    """
    if split_spec is not None and split_policy is not None:
        raise ValueError(
            "Cannot specify both `split_spec` and `split_policy`. Please use only one of them."
        )

    if split_spec is not None:
        # Annotate split points in the module based on user spec
        annotate_split_points(module, split_spec)  # 根据用户提供的 spec 注释模块中的分割点
        return Pipe.from_tracing(
            mod=module,
            example_args=mb_args,
            example_kwargs=mb_kwargs,
        )  # 返回使用跟踪功能创建的管道对象
    else:
        # 如果不满足上述条件，则使用分割策略来创建 Pipe 对象
        return Pipe.from_tracing(
            # 指定模块参数
            mod=module,
            # 传递示例参数列表
            example_args=mb_args,
            # 传递示例关键字参数
            example_kwargs=mb_kwargs,
            # 指定使用的分割策略
            split_policy=split_policy,
        )
```