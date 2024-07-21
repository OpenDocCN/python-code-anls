# `.\pytorch\torch\_dynamo\backends\debugging.py`

```py
# 忽略类型检查错误，适用于mypy类型检查工具
# 导入必要的模块和函数
import dataclasses  # 用于数据类的装饰器
import functools  # 用于函数装饰器
from importlib import import_module  # 用于动态导入模块
from typing import Any, List, Optional  # 引入类型提示

import torch  # 导入PyTorch库

# 从functorch库中导入编译相关的函数和配置
from functorch.compile import min_cut_rematerialization_partition
from torch import _guards  # 导入私有模块
from torch._functorch import config as functorch_config  # 引入functorch配置
from torch._functorch.compilers import ts_compile  # 导入编译器函数

# 从当前包中导入自定义模块和函数
from .common import aot_autograd  # 导入AOT自动求导相关功能
from .registry import register_debug_backend as register_backend  # 导入注册调试后端函数

"""
This file contains TorchDynamo backends intended for debugging uses.
"""

# 注册一个名为'eager'的调试后端函数
@register_backend
def eager(gm, fake_tensor_inputs):
    return gm.forward

# 注册一个名为'eager_noexcept'的调试后端函数
@register_backend
def eager_noexcept(gm, fake_tensor_inputs):
    # 该后端用于检查生成的GraphModule是否会引发错误
    def inner(*args):
        try:
            return gm(*args)
        except Exception as e:
            raise torch._dynamo.exc.TorchDynamoException(
                "Unexpected exception when running generated GraphModule"
            ) from e

    return inner

# 注册一个名为'pre_dispatch_eager'的调试后端函数
@register_backend
def pre_dispatch_eager(gm, fake_tensor_inputs):
    from torch.fx.experimental.proxy_tensor import make_fx

    def runnable_gm(*args):
        return torch.fx.Interpreter(gm).run(*args)

    # 创建一个可运行的预分发版本的GraphModule对象，并打印可读输出
    pre_dispatch_gm = make_fx(runnable_gm, pre_dispatch=True)(*fake_tensor_inputs)
    pre_dispatch_gm.print_readable()

    return pre_dispatch_gm

# 注册一个名为'eager_debug'的调试后端函数
@register_backend
def eager_debug(gm, fake_tensor_inputs):
    from torch._subclasses.schema_check_mode import SchemaCheckMode

    # 用于添加更多调试信息
    # 目前，该后端用于检查具有不正确模式的自定义调度操作是否会出错
    def inner(*args):
        with SchemaCheckMode():
            return torch.fx.Interpreter(gm).run(*args)

    return inner

# 注册一个名为'torchscript'的调试后端函数
@register_backend(name="ts")
def torchscript(gm, fake_tensor_inputs):
    # 对GraphModule对象执行Torch脚本编译
    return torch.jit.script(gm)

# 定义一个函数'boxed_nop'，用于忽略输入，当输入不再需要时
def boxed_nop(fx_g, example_inputs):
    def run(args):
        return torch.fx.Interpreter(fx_g).boxed_run(args)

    run._boxed_call = True  # 标记为使用boxed调用
    return run

# aot_eager用于调试目的
# aot_eager使用AOT Autograd后端，配合nop编译器，有助于调试
aot_eager = aot_autograd(
    fw_compiler=boxed_nop,
    partition_fn=min_cut_rematerialization_partition,
    keep_inference_input_mutations=True,
)
register_backend(name="aot_eager", compiler_fn=aot_eager)

# 定义'aot_eager_default_partitioner'，使用默认分区方式的aot_eager调试后端
aot_eager_default_partitioner = aot_autograd(
    fw_compiler=boxed_nop, keep_inference_input_mutations=True
)
register_backend(
    name="aot_eager_default_partitioner", compiler_fn=aot_eager_default_partitioner
)

# 使用TorchInductor AOT Autograd的解析器和分区器来隔离aot与inductor问题
# aot_eager_decomp_partition用nop替换inductor编译器，以帮助隔离inductor与aot_eager错误
def aot_eager_decomp_partition(gm, fake_tensor_inputs):
    # 省略具体实现，需要根据上下文和项目需求进行实现
    pass
    # 使用functorch_config.patch上下文管理器来配置函数式Torch，设置unlift_effect_tokens为True
    with functorch_config.patch(unlift_effect_tokens=True):
        # 调用aot_autograd函数，并传入以下参数
        return aot_autograd(
            # 在memory_efficient_fusion()中提取的参数设置
            fw_compiler=boxed_nop,  # 正向编译器设置为boxed_nop
            bw_compiler=boxed_nop,  # 反向编译器设置为boxed_nop
            # 注意：此处使用lambda延迟导入inductor模块
            decompositions=lambda: import_module(
                "torch._inductor.compile_fx"
            ).select_decomp_table(),  # 调用torch._inductor.compile_fx模块的select_decomp_table方法
            # 使用min_cut_rematerialization_partition函数的部分应用，设置compiler为"inductor"
            partition_fn=functools.partial(
                min_cut_rematerialization_partition, compiler="inductor"
            ),
        )(gm, fake_tensor_inputs)  # 调用aot_autograd返回的函数，并传入gm和fake_tensor_inputs作为参数
# 注册自定义后端，将名称 "aot_eager_decomp_partition" 和编译器函数 aot_eager_decomp_partition 关联起来
register_backend(
    name="aot_eager_decomp_partition", compiler_fn=aot_eager_decomp_partition
)

# 使用 torchscript 后端的 AOT 自动求导。默认使用分区器。
# aot_ts 使用 torchscript 后端。可以通过 torch.jit.fuser(...) 使用相关的 fuser（nnc 或 nvfuser）
aot_ts = aot_autograd(fw_compiler=ts_compile)
register_backend(name="aot_ts", compiler_fn=aot_ts)

# 这些有错误的后端用于引入 bug，以便我们可以测试我们的重现提取 / 最小化脚本

# 定义一个自定义异常类，用于表示 Relu 编译错误
class ReluCompileError(Exception):
    pass

# 定义一个自定义异常类，用于表示仅用于测试的编译错误
class TestingOnlyCompileError(Exception):
    pass

# 注册一个名为 relu_compile_error_TESTING_ONLY 的后端函数
@register_backend
def relu_compile_error_TESTING_ONLY(gm: torch.fx.GraphModule, example_inputs):
    # 遍历图中的节点，如果节点目标是 torch.relu，则抛出 ReluCompileError 异常
    for node in gm.graph.nodes:
        if node.target == torch.relu:
            raise ReluCompileError
    return gm

# 注册一个名为 relu_runtime_error_TESTING_ONLY 的后端函数
@register_backend
def relu_runtime_error_TESTING_ONLY(gm: torch.fx.GraphModule, example_inputs):
    # 遍历图中的节点，如果节点目标是 torch.relu，则修改节点目标和参数，然后重新编译图
    for node in gm.graph.nodes:
        if node.target == torch.relu:
            node.target = torch._assert
            node.args = (False, "ReluRuntimeError")
    gm.recompile()
    return gm

# 注册一个名为 relu_accuracy_error_TESTING_ONLY 的后端函数
@register_backend
def relu_accuracy_error_TESTING_ONLY(gm: torch.fx.GraphModule, example_inputs):
    # 遍历图中的节点，如果节点目标是 torch.relu，则修改节点目标和参数，然后重新编译图
    for node in gm.graph.nodes:
        if node.target == torch.relu:
            node.target = torch.add
            node.args = (node.args[0], 1)
    gm.recompile()
    return gm

# 注册一个名为 non_leaf_compile_error_TESTING_ONLY 的后端函数
@register_backend
def non_leaf_compile_error_TESTING_ONLY(gm: torch.fx.GraphModule, example_inputs):
    # 需要在图中至少有一个非平凡元素，详见 https://github.com/pytorch/pytorch/issues/102898
    for node in gm.graph.nodes:
        if node.op == "call_function":
            break
    else:
        return gm
    # 检查示例输入是否包含非叶子张量，如果没有，则抛出 TestingOnlyCompileError 异常
    for t in example_inputs:
        if not t.is_leaf:
            raise TestingOnlyCompileError
    return gm

# 定义一个数据类 ExplainOutput，用于描述 torch._dynamo.explain() 的输出
@dataclasses.dataclass
class ExplainOutput:
    """
    This is the output of :func:`torch._dynamo.explain()`
    There is no reason to create this class directly.
    """

    graphs: List[torch.fx.GraphModule]  # 包含的图模块列表
    graph_count: int  # 图模块数量
    graph_break_count: int  # 中断的图模块数量
    break_reasons: List[Any]  # 中断原因列表（类型为 GraphCompileReason，这里并不关心具体类型）
    op_count: int  # 操作数量
    ops_per_graph: Optional[List[torch.fx.Node]] = None  # 每个图模块的操作节点列表（可选）
    out_guards: Optional[List[_guards.Guard]] = None  # 输出保护列表（可选）
    compile_times: Optional[str] = None  # 编译时间（可选）
    # 定义对象的字符串表示方法，用于返回对象的详细信息字符串
    def __str__(self):
        # 初始化输出字符串，包含图数量和图中断点数量的信息
        output = f"Graph Count: {self.graph_count}\n"
        output += f"Graph Break Count: {self.graph_break_count}\n"
        output += f"Op Count: {self.op_count}\n"

        # 添加中断原因的部分
        output += "Break Reasons:\n"
        # 遍历中断原因列表，列出每个中断原因及其对应的用户栈信息
        for idx, break_reason in enumerate(self.break_reasons):
            output += f"  Break Reason {idx+1}:\n"
            output += f"    Reason: {break_reason.reason}\n"
            output += "    User Stack:\n"
            # 遍历用户栈信息，列出每一帧的摘要
            for frame_summary in break_reason.user_stack:
                output += f"      {frame_summary}\n"

        # 如果存在每个图的操作数信息，添加该部分
        if self.ops_per_graph is not None:
            output += "Ops per Graph:\n"
            # 遍历每个图的操作数列表，列出每个图中的操作信息
            for idx, ops in enumerate(self.ops_per_graph):
                output += f"  Ops {idx+1}:\n"
                # 遍历每个操作，列出操作的详细信息
                for op in ops:
                    output += f"    {op}\n"

        # 如果存在外部保护信息列表，添加该部分
        if self.out_guards is not None:
            output += "Out Guards:\n"
            # 遍历外部保护信息列表，列出每个保护信息的详细内容
            for i, guard in enumerate(self.out_guards):
                output += f"  Guard {i+1}:\n"
                output += f"    {str(guard)}"

        # 如果存在编译时间信息，添加该部分
        if self.compile_times is not None:
            output += f"Compile Times: {self.compile_times}\n"

        # 返回完整的对象信息字符串
        return output
# 定义一个函数用于详细解析 torch.fx.GraphModule 的信息，包括操作数量、图中断点原因等
def _explain_graph_detail(
    gm: torch.fx.GraphModule, graphs, op_count, ops_per_graph, break_reasons
):
    """
    This function is a utility which processes a torch.fx.GraphModule and
    accumulates information about its ops, graph breaks, and other details. It
    is intended to be used by the ExplainWithBackend class and
    `torch._dynamo.explain()` to provide details from Dynamo's graph capture.

    Parameters:
        gm (torch.fx.GraphModule): The GraphModule to be processed.
        graphs (list): A list that accumulates all the GraphModules processed.
        op_count (int): The total count of operations in all GraphModules processed so far.
        ops_per_graph (list): A list that accumulates the operations of each GraphModule.
        break_reasons (list): A list that accumulates the reasons for breaks in each GraphModule.

    Returns:
        tuple: A tuple containing the processed GraphModule, the updated lists of graphs,
               operations per graph, and break reasons, and the updated operation count.
    """
    # 将当前的 GraphModule 添加到 graphs 列表中
    graphs.append(gm)
    # 获取当前 GraphModule 中所有 call_function 操作的目标，并统计数量加到 op_count 中
    ops = [node.target for node in gm.graph.nodes if node.op == "call_function"]
    op_count += len(ops)
    # 将当前 GraphModule 的操作列表 ops 添加到 ops_per_graph 列表中
    ops_per_graph.append(ops)
    # 如果当前 GraphModule 存在子图断点原因，则将其添加到 break_reasons 列表中
    if gm.compile_subgraph_reason.graph_break:
        break_reasons.append(gm.compile_subgraph_reason)

    # 返回更新后的 gm、graphs、op_count、ops_per_graph 和 break_reasons
    return gm, graphs, op_count, ops_per_graph, break_reasons


class ExplainWithBackend:
    """
    This class is intended to be used as a backend for `torch.compile`. It is
    composable with other backends. When used in this way, it accumulates
    information about graph breaks, ops, and other info and provides a string
    representation summarizing this information.

    Attributes:
        backend (str): The name of the backend to use for optimization.
        graphs (list): A list of the graphs captured by TorchDynamo.
        op_count (int): The total number of operations in all optimized graphs.
        break_reasons (list): A list of graph break reasons with stack traces.

    Example Usage:
        def fn(x):
            x = torch.sigmoid(x)
            return x

        torch._dynamo.reset()
        eb = ExplainWithBackend("inductor")
        optimized_fn = torch.compile(fn, backend=eb)
        result = optimized_fn(torch.randn(5))
        print(eb.output())
    """

    def __init__(self, backend):
        from .registry import lookup_backend

        # 初始化 ExplainWithBackend 实例，设定优化后端的名称
        self.backend = lookup_backend(backend)
        # 初始化一个空列表来存储捕获的图形模块
        self.graphs = []
        # 初始化操作总数为 0
        self.op_count = 0
        # 初始化一个空列表来存储图断点原因及其堆栈跟踪
        self.break_reasons = []

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):
        # 调用 _explain_graph_detail 函数来处理传入的 GraphModule，更新相关信息
        gm, self.graphs, self.op_count, _, self.break_reasons = _explain_graph_detail(
            gm, self.graphs, self.op_count, [], self.break_reasons
        )
        # 调用后端对象来处理更新后的 GraphModule，并返回处理结果
        return self.backend(gm, example_inputs)
    # 定义一个方法 output，返回类型为 ExplainOutput
    def output(self) -> ExplainOutput:
        # 计算 self.graphs 列表中的图形数量
        graph_count = len(self.graphs)
        # 创建 ExplainOutput 对象并赋值给 output 变量，传入参数为：
        # self.graphs: 图形列表
        # graph_count: 图形数量
        # graph_count - 1: 图形数量减一
        # self.break_reasons: 中断原因列表
        # self.op_count: 操作计数
        output = ExplainOutput(
            self.graphs,
            graph_count,
            graph_count - 1,
            self.break_reasons,
            self.op_count,
        )

        # 返回创建的 ExplainOutput 对象
        return output
```