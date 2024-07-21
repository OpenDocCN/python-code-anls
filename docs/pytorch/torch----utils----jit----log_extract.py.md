# `.\pytorch\torch\utils\jit\log_extract.py`

```
# mypy: allow-untyped-defs
# 导入所需模块和类型声明
from contextlib import contextmanager
from typing import Any, List, Tuple, cast
import random
import torch
import time
from torch.utils.benchmark import Timer

# 从文件中提取包含在特定标记之间的图形信息，并返回列表
def extract_ir(filename: str) -> List[str]:
    BEGIN = "<GRAPH_EXPORT>"
    END = "</GRAPH_EXPORT>"
    pfx = None
    current = ""
    graphs = []
    # 打开文件并按照标记分割内容
    with open(filename) as f:
        split_strs = f.read().split(BEGIN)
        for i, split_str in enumerate(split_strs):
            if i == 0:
                continue
            end_loc = split_str.find(END)
            if end_loc == -1:
                continue
            s = split_str[:end_loc]
            pfx = split_strs[i - 1].splitlines()[-1]
            # 提取每个图形的行，去除前缀并连接成一个字符串
            lines = [x[len(pfx):] for x in s.splitlines(keepends=True)]
            graphs.append(''.join(lines))

    return graphs


# 根据输入的类型描述创建一个张量
def make_tensor_from_type(inp_type: torch._C.TensorType):
    size = inp_type.sizes()
    stride = inp_type.strides()
    device = inp_type.device()
    dtype = inp_type.dtype()
    assert size is not None
    assert stride is not None
    assert device is not None
    assert dtype is not None
    # 使用给定的大小、步幅、设备和数据类型创建一个空的张量
    return torch.empty_strided(size=size, stride=stride, device=device, dtype=dtype)

# 加载图形和输入数据
def load_graph_and_inputs(ir: str) -> Tuple[Any, List[Any]]:
    # 解析输入的图形描述字符串，将其中的张量常量解析为张量
    graph = torch._C.parse_ir(ir, parse_tensor_constants=True)
    # 将图形中多个输出合并为一个元组
    graph.makeMultiOutputIntoTuple()
    inputs = []
    # 遍历图形的输入，根据类型生成随机或者默认值
    for inp in graph.inputs():
        if isinstance(inp.type(), torch._C.FloatType):
            inputs.append(random.uniform(.1, 100))
        elif isinstance(inp.type(), torch._C.IntType):
            inputs.append(random.randint(1, 100))
        elif isinstance(inp.type(), torch._C.TensorType):
            tensorType = cast(torch._C.TensorType, inp.type())
            inputs.append(make_tensor_from_type(tensorType))
        elif isinstance(inp.type(), torch._C.BoolType):
            inputs.append(random.randint(0, 1) == 1)
        else:
            raise NotImplementedError(f"A default value is not implemented for type {inp.type()}")

    # 根据解析后的图形创建一个函数对象
    func = torch._C._create_function_from_graph("forward", graph)
    # 移除函数图形中的形状信息
    torch._C._jit_pass_erase_shape_information(func.graph)
    return (func, inputs)

# 在 CUDA 上测量函数执行时间
def time_cuda(fn, inputs, test_runs):
    # 创建一个计时器对象，测量函数执行时间
    t = Timer(stmt="fn(*inputs)", globals={"fn": fn, "inputs" : inputs})
    times = t.blocked_autorange()
    return times.median * 1000  # 返回中位数的执行时间（毫秒）

# 在 CPU 上测量函数执行时间
def time_cpu(fn, inputs, test_runs):
    s = time.perf_counter()
    for _ in range(test_runs):
        fn(*inputs)
    e = time.perf_counter()
    return (e - s) / test_runs * 1000  # 返回平均每次执行的时间（毫秒）

# 运行测试，包括预热和测试运行
def run_test(ir, inputs, *, warmup_runs=10, test_runs=20) -> float:
    # 加载图形和输入
    graph, _ = load_graph_and_inputs(ir)
    # 预热运行图形函数
    for _ in range(warmup_runs):
        graph(*inputs)

    is_cpu = None
    # 确定输入数据是否在 CPU 上运行
    for input in inputs:
        if isinstance(input, torch.Tensor):
            is_cpu = input.device.type == "cpu"
            break
    assert is_cpu is not None  # 断言确认 is_cpu 不为 None
    # 根据条件选择性调用 CPU 或 CUDA 版本的时间测试函数，并返回其结果
    out = time_cpu(graph, inputs, test_runs) if is_cpu else time_cuda(graph, inputs, test_runs)
    # 返回时间测试函数的输出作为函数的返回值
    return out
@contextmanager
def no_fuser(*args, **kwargs):
    # 保存当前的图执行器优化设置，并设置为不优化
    old_optimize = torch._C._get_graph_executor_optimize(False)
    try:
        # 在此上下文中什么都不做，直接yield过去
        yield
    finally:
        # 恢复之前保存的图执行器优化设置
        torch._C._get_graph_executor_optimize(old_optimize)

def run_baseline_no_fusion(ir, inputs) -> float:
    # 使用no_fuser上下文管理器，运行测试并返回结果
    with no_fuser():
        return run_test(ir, inputs)

def run_nnc(ir, inputs, dynamic) -> float:
    try:
        # 根据dynamic参数选择不同的策略
        strat = [("DYNAMIC", 10)] if dynamic else [("STATIC", 10)]
        # 设置新的融合策略，并保存旧的策略设置
        old_strat = torch.jit.set_fusion_strategy(strat)
        # 在"fuser1"融合环境中运行测试并返回结果
        with torch.jit.fuser("fuser1"):
            return run_test(ir, inputs)
    finally:
        # 在finally块中恢复旧的融合策略设置
        torch.jit.set_fusion_strategy(old_strat)

def run_nvfuser(ir, inputs) -> float:
    # 在"fuser2"融合环境中运行测试并返回结果
    with torch.jit.fuser("fuser2"):
        return run_test(ir, inputs)
```