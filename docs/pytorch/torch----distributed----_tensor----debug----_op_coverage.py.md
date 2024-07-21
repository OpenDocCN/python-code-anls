# `.\pytorch\torch\distributed\_tensor\debug\_op_coverage.py`

```
# mypy: allow-untyped-defs
# 导入 itemgetter 函数和 List 类型
from operator import itemgetter
from typing import List

# 导入 PyTorch 相关模块
import torch
import torch.fx
import torch.nn as nn

# 导入编译相关的函数和模块
from functorch.compile import make_boxed_func
from torch._functorch.compilers import aot_module

# 从 torch._inductor.decomposition 模块导入 select_decomp_table 函数
from torch._inductor.decomposition import select_decomp_table

# 从 torch.distributed._tensor 模块导入 DTensor 类
from torch.distributed._tensor import DTensor

# 使用 select_decomp_table 函数获取感应器分解表
inductor_decomps = select_decomp_table()

# 定义一个空的列表，用于存储 torch.fx.GraphModule 对象
graphs: List[torch.fx.GraphModule] = []


# 定义一个编译器函数 fwd_bwd_compiler，用于将 forward 和 backward 图形装箱
def fwd_bwd_compiler(fx_g, _):
    # 将输入的 fx_g 添加到 graphs 列表中
    graphs.append(fx_g)
    # 调用 make_boxed_func 函数，返回装箱后的函数
    return make_boxed_func(fx_g)


# 定义函数 get_inductor_decomp_graphs，获取感应器分解图形
def get_inductor_decomp_graphs(model: nn.Module, args, kwargs):
    """
    Obtain forward and backward graphs of a model with inductor decompositions using tracing and aot_module.

    Convenient util to get the fwd and bwd graphs of an arbitrary model
    with inductor decompositions. Note that this would simply do tracing
    with aot_module and don't ensure correctness. This is useful to track
    the ops needed in DTensor.
    """
    # 调用 aot_module 函数，生成编译后的模块
    compiled_mod = aot_module(
        model, fw_compiler=fwd_bwd_compiler, decompositions=inductor_decomps
    )
    # 对编译后的模块执行，传入 args 和 kwargs
    output = compiled_mod(*args, **kwargs)

    # 如果输出张量的维度不为零
    if output.ndim != 0:
        # 默认对非标量张量进行求和，以便进行反向传播
        output = output.sum()

    # 执行反向传播
    output.backward()

    # 断言 graphs 列表长度为 2，确保包含一个前向和一个后向图形
    assert len(graphs) == 2
    return graphs


# 定义函数 print_op_coverage_summary，打印模型的运算符覆盖摘要
def print_op_coverage_summary(model: nn.Module, args, kwargs, *, output_csv=False):
    """
    Util to print the operator coverage summary of a certain model with tabulute.

    Must have tabulate module installed.
    """
    # 导入必要的 Python 模块
    import csv
    from tabulate import tabulate

    # 获取前向和后向图形
    fwd_graph, bwd_graph = get_inductor_decomp_graphs(model, args, kwargs)

    # 初始化一个空字典，用于统计操作的调用次数
    op_counts = {}

    # 遍历前向图形的所有节点
    for node in fwd_graph.graph.nodes:
        # 如果节点操作为 "call_function"，并且目标是 torch._ops.OpOverload 的实例
        if node.op == "call_function" and isinstance(
            node.target, torch._ops.OpOverload
        ):
            # 如果操作目标不在 op_counts 中，则初始化为 0
            if node.target not in op_counts:
                op_counts[node.target] = 0
            # 增加操作目标的调用次数
            op_counts[node.target] += 1

    # 遍历后向图形的所有节点，与前向图形类似的统计操作调用次数
    for node in bwd_graph.graph.nodes:
        if node.op == "call_function" and isinstance(
            node.target, torch._ops.OpOverload
        ):
            if node.target not in op_counts:
                op_counts[node.target] = 0
            op_counts[node.target] += 1

    # 初始化一个空列表，用于存储操作信息
    op_infos = []

    # 遍历 op_counts 字典的每个项，获取操作和其调用次数
    for op, count in op_counts.items():
        # 判断操作是否在 DTensor._op_dispatcher.sharding_propagator.op_to_rules 中支持
        supported = op in DTensor._op_dispatcher.sharding_propagator.op_to_rules
        # 将操作信息添加到 op_infos 列表中
        op_infos.append([op, str(op._schema), count, supported])

    # 根据调用次数降序排序操作信息列表
    count_idx = 2
    op_infos.sort(key=itemgetter(count_idx), reverse=True)

    # 定义表头
    headers = ["Operator", "Schema", "Total Count", "Supported"]

    # 使用 tabulate 函数打印操作信息表格
    print(tabulate(op_infos, headers=headers))
    if output_csv:
        # 如果需要生成 CSV 文件

        # 打开一个 CSV 文件进行写操作，指定文件名为 "op_summary.csv"，使用 newline="" 来确保换行符处理正确
        with open("op_summary.csv", "w", newline="") as csv_file:
            # 创建一个 CSV writer 对象
            csv_writer = csv.writer(csv_file)

            # 写入 CSV 文件的表头行
            csv_writer.writerow(headers)

            # 逐行将操作信息写入 CSV 文件
            for row in op_infos:
                csv_writer.writerow(row)
```