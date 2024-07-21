# `.\pytorch\torch\_inductor\subgraph_lowering.py`

```py
# mypy: allow-untyped-defs
"""
用于高阶操作符使用的降低子图的实用程序

"""

import functools  # 导入 functools 模块，用于函数操作的工具函数
import operator  # 导入 operator 模块，用于操作符的函数实现
from dataclasses import dataclass  # 导入 dataclass 类型，用于定义数据类
from typing import List, Optional, TypeVar  # 导入类型提示相关的模块

import torch  # 导入 PyTorch 深度学习框架

from . import ir  # 导入当前包内的 ir 模块，用于中间表示
from .exc import SubgraphLoweringException  # 导入子图降低异常类
from .ops_handler import SimpleCSEHandler  # 导入简单公共子表达式处理器
from .virtualized import ops, V, WrapperHandler  # 导入虚拟化相关的模块和类

T = TypeVar("T")  # 定义一个类型变量 T


class PointwiseSubgraphLowering(torch.fx.Interpreter):
    graph_outputs: Optional[List[ir.IRNode]]

    def __init__(
        self,
        gm: torch.fx.GraphModule,
        root_graph_lowering: "torch._inductor.graph.GraphLowering",
    ):
        super().__init__(gm)
        self.graph_outputs = None  # 初始化图输出为 None
        self.root_graph = root_graph_lowering  # 设置根图降低对象

    @property
    def sizevars(self):
        return self.root_graph.sizevars  # 返回根图的尺寸变量

    def mark_buffer_mutated(self, name):
        raise SubgraphLoweringException("Mutations are not supported in this context")
        # 抛出异常，不支持在此上下文中进行变异操作

    def register_buffer(self, data):
        raise SubgraphLoweringException(
            "Buffer creation is not supported in this context"
        )
        # 抛出异常，不支持在此上下文中创建缓冲区

    def call_function(self, target, args, kwargs):
        from .lowering import lowerings  # 从降低模块中导入 lowerings

        if target is operator.getitem and isinstance(args[0], (list, tuple, dict)):
            return super().call_function(target, args, kwargs)
            # 如果目标是 getitem，并且参数的第一个元素是列表、元组或字典，则调用父类的函数

        assert isinstance(target, torch._ops.OpOverload)

        if target not in lowerings:
            raise SubgraphLoweringException(
                f"{target} not supported in subgraph, (missing lowering)"
            )
            # 如果目标不在 lowerings 中，抛出子图降低异常

        if torch.Tag.pointwise not in target.tags:
            raise SubgraphLoweringException(
                f"Only pointwise operators are supported in this context, but got {target}"
            )
            # 如果目标操作不是 pointwise 类型，则抛出子图降低异常

        return lowerings[target](*args, **kwargs)
        # 返回 lowerings 中目标对应的操作函数的调用结果

    def output(self, target, args, kwargs):
        assert len(args) == 1
        self.graph_outputs = args[0]
        # 将输出设置为传入的参数列表中的第一个参数


@dataclass
class InputDescriptor:
    dtype: torch.dtype  # 输入数据类型
    device: torch.device  # 输入设备类型


class TracingOpsHandler(WrapperHandler[T]):
    def __init__(self, tracer, num_inputs):
        parent = tracer.create_proxy("placeholder", "ops", (), {})
        super().__init__(parent)
        self.tracer = tracer

        self.placeholders = [
            self.tracer.create_proxy("placeholder", f"input{i}", (), {})
            for i in range(num_inputs)
        ]
        # 创建输入的占位符列表，使用追踪器创建

    def placeholder(self, idx):
        return self.placeholders[idx]
        # 返回指定索引的占位符

    def output(self, *args):
        return self.tracer.create_node(
            "output", "output", (tuple(self.tracer.create_arg(a) for a in args),), {}
        )
        # 创建输出节点


def lower_pointwise_subgraph(subgraph: ir.Subgraph, inputs: List[InputDescriptor]):
    # 降低子图到 ir.Pointwise 节点
    def fake_inner_fn(loop_idx, input_idx):
        return ops.placeholder(input_idx)
        # 返回输入索引处的占位符操作
    # 创建一个列表，包含多个 Pointwise 对象，每个对象使用指定的设备和数据类型，
    # 并指定一个偏函数作为内部函数，传递当前输入索引作为参数，同时设置空的范围列表。
    graph_inputs = [
        ir.Pointwise.create(
            device=desc.device,
            dtype=desc.dtype,
            inner_fn=functools.partial(fake_inner_fn, input_idx=i),
            ranges=[],
        )
        for i, desc in enumerate(inputs)
    ]
    
    # 获取子图的图模块
    gm = subgraph.graph_module
    
    # 使用 PointwiseSubgraphLowering 类对 gm 进行降低，根图降低为 V.graph
    pw_subgraph = PointwiseSubgraphLowering(gm, root_graph_lowering=V.graph)
    
    # 设置当前作用域的图处理程序为 pw_subgraph
    with V.set_graph_handler(pw_subgraph):  # type: ignore[arg-type]
        # 运行 pw_subgraph，传入 graph_inputs 作为参数
        pw_subgraph.run(*graph_inputs)

    # 将多个点积计算组合成单个图模块
    # 通过逐个追踪每个计算并进行公共子表达式消除（CSE）
    tracer = torch.fx.Tracer()
    
    # 创建一个新的图对象，使用 tracer 作为追踪器类
    tracer.graph = torch.fx.Graph(tracer_cls=tracer.__class__)
    
    # 创建 SimpleCSEHandler 对象，将 TracingOpsHandler 和输入数量作为参数
    trace_ops = SimpleCSEHandler(TracingOpsHandler(tracer, len(inputs)))
    
    # 断言 pw_subgraph 的图输出不为空
    assert pw_subgraph.graph_outputs is not None

    # 设置当前作用域的操作处理程序为 trace_ops
    with V.set_ops_handler(trace_ops):
        # 初始化一个空列表，用于存储输出的中间表示
        output_irs = []

        # 遍历 pw_subgraph 的图输出
        for out_var in pw_subgraph.graph_outputs:
            # 断言 out_var 是 ir.TensorBox 类型
            assert isinstance(out_var, ir.TensorBox), type(out_var)
            # 断言 out_var 的大小为空列表
            assert out_var.get_size() == []
            # 断言 out_var 的数据是 ir.StorageBox 类型
            assert isinstance(out_var.data, ir.StorageBox)
            # 断言 out_var 的数据的数据成员是 ir.Pointwise 类型
            assert isinstance(out_var.data.data, ir.Pointwise)

            # 初始化一个空元组作为索引
            idx = ()
            # 调用 inner_fn 方法，传入 idx 参数，获取中间表示结果
            ir_out = out_var.data.data.inner_fn(idx)

            # 将中间表示结果添加到 output_irs 列表中
            output_irs.append(ir_out)

        # 调用 ops.output 方法，传入 output_irs 中间表示作为参数
        ops.output(*output_irs)

    # 使用 tracer.graph 创建一个新的图模块对象，并存储为 lowered_gm
    lowered_gm = torch.fx.GraphModule({}, tracer.graph)

    # 定义一个新的内部函数 inner_fn，调用 lowered_gm 方法，传入 V.get_ops_handler() 和参数
    def inner_fn(*args, **kwargs):
        return lowered_gm(V.get_ops_handler(), *args, **kwargs)

    # 返回 inner_fn 函数作为最终结果
    return inner_fn
```