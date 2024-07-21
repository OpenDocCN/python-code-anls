# `.\pytorch\benchmarks\fastrnns\bench.py`

```
# 导入必要的库和模块
import argparse  # 用于解析命令行参数的库
import copy  # 用于复制对象的库
import gc  # Python的垃圾回收模块
import json  # 用于处理 JSON 数据的库
import sys  # 系统相关的功能模块
import time  # 时间相关的功能模块
from collections import namedtuple  # 命名元组，用于创建命名的数据结构

import torch  # PyTorch 深度学习框架
from torch.autograd.profiler import record_function  # 用于记录 PyTorch 自动求导函数的性能分析器

from .fuser import set_fuser  # 导入本地的 fuser 模块中的 set_fuser 函数
from .runner import get_nn_runners  # 导入本地的 runner 模块中的 get_nn_runners 函数

# 定义命名元组 BenchResult，包含多个性能指标的名称
BenchResult = namedtuple(
    "BenchResult",
    [
        "name",
        "avg_fwd",
        "std_fwd",
        "info_fwd",
        "avg_bwd",
        "std_bwd",
        "info_bwd",
    ],
)


# 辅助函数：将字符串格式化成指定宽度，不足部分用空格填充
def fit_str(string, colwidth=16):
    if len(string) < colwidth:
        return (colwidth - len(string)) * " " + string
    else:
        return string[:colwidth]


# 辅助函数：将对象转换为字符串表示
def to_str(item):
    if isinstance(item, float):
        return f"{item:.4g}"  # 对于浮点数，保留四位有效数字的格式化字符串
    return str(item)


# 打印表头的函数，用于性能测试结果的输出
def print_header(colwidth=16, sep=" "):
    items = []
    for item in BenchResult._fields:
        items.append(fit_str(item))  # 格式化字段名到指定宽度
    return sep.join(items)  # 返回用指定分隔符连接的字符串


# 打印性能测试结果的函数，将 BenchResult 元组中的内容格式化输出
def pretty_print(benchresult, colwidth=16, sep=" "):
    items = []
    for thing in benchresult:
        items.append(fit_str(to_str(thing)))  # 格式化每个项目的字符串表示到指定宽度
    return sep.join(items)  # 返回用指定分隔符连接的字符串


# 用于在 CPU 上运行时替代 torch.cuda.Event 的类定义
class Event:
    def __init__(self, enable_timing):
        pass  # 初始化函数，仅占位，无具体实现

    def record(self):
        self.time = time.perf_counter()  # 记录当前时间戳

    def elapsed_time(self, end_event):
        assert isinstance(end_event, Event)  # 断言检查 end_event 是 Event 类的实例
        return end_event.time - self.time  # 返回两个事件记录的时间差


# 性能测试函数的定义，用于评估模型训练过程的性能指标
def trainbench(
    name,
    rnn_creator,
    nloops=100,
    warmup=10,
    seqLength=100,
    numLayers=1,
    inputSize=512,
    hiddenSize=512,
    miniBatch=64,
    device="cuda",
    seed=None,
):
    # 定义一个函数，用于批量训练神经网络模型
    def train_batch(modeldef):
        # 根据设备类型选择合适的计时器类
        if device == "cuda":
            timer_class = torch.cuda.Event
        else:
            timer_class = Event
        
        # 创建前向传播计时事件对象
        fwd_start_event = timer_class(enable_timing=True)
        fwd_end_event = timer_class(enable_timing=True)
        # 创建反向传播计时事件对象
        bwd_start_event = timer_class(enable_timing=True)
        bwd_end_event = timer_class(enable_timing=True)

        # 垃圾回收，清理不再使用的内存
        gc.collect()

        # 记录前向传播开始时间
        fwd_start_event.record()
        # 使用记录函数记录前向传播过程
        with record_function("## forward ##"):
            forward_output = modeldef.forward(*modeldef.inputs)
        # 记录前向传播结束时间
        fwd_end_event.record()

        # XXX: 如果需要打印一些信息可以使用
        # print(modeldef.forward.graph_for(*modeldef.inputs))

        # 如果定义了反向传播设置函数，则根据前向输出设置反向传播输入
        if modeldef.backward_setup is not None:
            backward_input = modeldef.backward_setup(forward_output)
        else:
            backward_input = forward_output

        # 垃圾回收，清理不再使用的内存
        gc.collect()

        # 记录反向传播开始时间
        bwd_start_event.record()
        # 如果定义了反向传播函数，则执行反向传播
        if modeldef.backward is not None:
            modeldef.backward(*backward_input)
        # 记录反向传播结束时间
        bwd_end_event.record()

        # 如果定义了反向传播函数，则执行参数梯度清零操作
        if modeldef.backward is not None:
            with torch.no_grad():
                for param in modeldef.params:
                    assert param.grad is not None
                    param.grad.zero_()

        # 如果运行在CUDA设备上，则等待计算完成
        if device == "cuda":
            torch.cuda.synchronize()

        # 计算前向传播和反向传播的时间间隔
        fwd_time = fwd_start_event.elapsed_time(fwd_end_event)
        bwd_time = bwd_start_event.elapsed_time(bwd_end_event)
        
        # 返回前向传播和反向传播的时间
        return fwd_time, bwd_time

    # 创建神经网络模型的参数字典
    creator_args = {
        "seqLength": seqLength,
        "numLayers": numLayers,
        "inputSize": inputSize,
        "hiddenSize": hiddenSize,
        "miniBatch": miniBatch,
        "device": device,
        "seed": seed,
    }

    # 使用指定参数创建神经网络模型定义
    modeldef = rnn_creator(**creator_args)

    # 进行预热阶段的多次模型训练批处理调用
    [train_batch(modeldef) for _ in range(warmup)]

    # 执行多次循环，进行模型训练批处理调用，并记录每次前向和反向传播时间
    results = [train_batch(modeldef) for _ in range(nloops)]
    fwd_times, bwd_times = zip(*results)

    # 将前向传播和反向传播的时间转换为PyTorch张量
    fwd_times = torch.tensor(fwd_times)
    bwd_times = torch.tensor(bwd_times)
    
    # 返回基准测试结果对象，包括模型名称、前向传播平均时间、前向传播标准差、前向传播时间信息、
    # 反向传播平均时间、反向传播标准差、反向传播时间信息
    return BenchResult(
        name=name,
        avg_fwd=fwd_times.mean().item(),
        std_fwd=fwd_times.std().item(),
        info_fwd=fwd_times,
        avg_bwd=bwd_times.mean().item(),
        std_bwd=bwd_times.std().item(),
        info_bwd=bwd_times,
    )
# 将输出打印到标准错误流中
def print_stderr(*args, **kwargs):
    # 设置打印到文件参数为标准错误流
    kwargs["file"] = sys.stderr
    # 调用内置的 print 函数，将参数输出到标准错误流
    return print(*args, **kwargs)


# 输出适用于 OSS 格式的 JSON 字符串
def print_json_oss_format(results):
    # 初始化一个空字典来存储转换后的结果
    oss_results = {}
    # 遍历结果字典的每个组名和对应的值
    for group_name, group_val in results.items():
        # 初始化 OSS 结果字典的组名条目
        oss_results[group_name] = {}
        # 遍历每个组的模型名和运行时间数据
        for model_name, run_time in group_val.items():
            # 将每个模型的平均运行时间添加到 OSS 结果中
            oss_results[group_name][model_name] = run_time["avg"]

    # 打印转换后的结果为 JSON 格式
    print(json.dumps(oss_results))


# 输出适用于 AI-PEP 格式的 JSON 字符串
def print_json_pep_format(results):
    # 遍历结果字典的每个组名和对应的值
    for group_name, group_val in results.items():
        # 遍历每个组的模型名和运行时间数据
        for model_name, run_time in group_val.items():
            # 计算运行时间信息的长度
            num_iters = len(run_time["info"])
            # 将运行时间信息转换为列表形式
            info = run_time["info"].tolist()
            # 遍历每个迭代的信息并输出为 AI-PEP 格式的 JSON 字符串
            for i in range(num_iters):
                print(
                    "Caffe2Observer "
                    + json.dumps(
                        {
                            "type": "NET",
                            "metric": group_name + "-" + model_name,
                            "unit": "ms",
                            "value": str(info[i]),
                        }
                    )
                )


# 对一组 RNN 运行器执行基准测试，并返回结果字典
def bench(rnn_runners, group_name, print_json=False, sep=" ", **params):
    # 打印基准测试的标题行到标准错误流中
    print_stderr(print_header(sep=sep))
    # 初始化一个空字典来存储测试结果
    results = {}
    # 遍历每个 RNN 运行器的名称、创建函数和上下文管理器
    for name, creator, context in rnn_runners:
        # 使用上下文管理器执行 RNN 训练基准测试
        with context():
            try:
                # 调用训练基准测试函数，获取测试结果
                result = trainbench(name, creator, **params)
                # 将结果中的 info_fwd 和 info_bwd 替换为 "None"
                result_with_no_info = result._replace(info_fwd="None", info_bwd="None")
                # 将处理后的结果打印到标准错误流中
                print_stderr(pretty_print(result_with_no_info, sep=sep))
                # 将完整结果存储到结果字典中
                results[name] = result
            except Exception as e:
                # 如果打印 JSON 格式未设置，则抛出异常
                if not print_json:
                    raise

    # 返回整理后的结果字典，包含平均值、标准差和信息列表
    return {
        group_name: {
            k: {"avg": v.avg_fwd, "std": v.std_fwd, "info": v.info_fwd}
            for k, v in results.items()
        },
        group_name
        + "-backward": {
            k: {"avg": v.avg_bwd, "std": v.std_bwd, "info": v.info_bwd}
            for k, v in results.items()
        },
    }


# 对一组模型执行基准测试，并返回结果字典
def bench_group(model_list, bench_name, bench_group, bench_args):
    # 打印基准测试组的标题行到标准错误流中
    print_stderr(f"Benchmarking {bench_name}s...")
    # 获取神经网络运行结果
    nn_results = bench(get_nn_runners(*model_list), bench_group, **bench_args)
    # 打印空行到标准错误流中
    print_stderr("")
    # 返回神经网络运行结果字典
    return nn_results


# 主程序入口点，用于分析 RNN
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile RNNs")

    # 定义命令行参数解析器，描述 RNN 分析功能
    parser.add_argument("--seqLength", default="100", type=int)
    parser.add_argument("--numLayers", default="1", type=int)
    parser.add_argument("--inputSize", default="512", type=int)
    parser.add_argument("--hiddenSize", default="512", type=int)
    # 添加一个整数类型的命令行参数 `--miniBatch`，默认为 64
    parser.add_argument("--miniBatch", default="64", type=int)
    # 添加一个整数类型的命令行参数 `--warmup`，默认为 10
    parser.add_argument("--warmup", default="10", type=int)
    # 添加一个整数类型的命令行参数 `--nloops`，默认为 100
    parser.add_argument("--nloops", default="100", type=int)
    # 添加一个字符串类型的命令行参数 `--device`，默认为 "cuda"
    parser.add_argument("--device", default="cuda", type=str)
    # 添加一个布尔类型的命令行参数 `--variable-lstms`，并提供帮助信息
    parser.add_argument(
        "--variable-lstms",
        "--variable_lstms",
        action="store_true",
        help="Also benchmark variable sequence length lstms "
        "Note that some of these run really slowly "
        "and that the `seqLength` flag will be ignored.",
    )
    # 添加一个字符串类型的命令行参数 `--sep`，默认为 " "
    parser.add_argument("--sep", default=" ", type=str)
    # 添加一个可选参数 `--print-json`，可以带一个参数，默认为 None，常量为 "oss"
    parser.add_argument("--print-json", nargs="?", default=None, const="oss")
    # 添加一个列表类型的命令行参数 `--rnns`，提供帮助信息指明可以运行的选项
    parser.add_argument("--rnns", nargs="*", help="What to run. cudnn, aten, jit, etc")
    # 添加一个列表类型的命令行参数 `--cnns`，提供帮助信息指明可以运行的选项
    parser.add_argument(
        "--cnns", nargs="*", help="What to run. resnet18, resnet18_jit, resnet50, etc"
    )
    # 添加一个列表类型的命令行参数 `--group`，默认值为预设的 `default_groups`，提供帮助信息
    parser.add_argument(
        "--group",
        nargs="*",
        default=default_groups,
        help="Which group to run. cnns, rnns, etc.",
    )
    # 添加一个字符串类型的命令行参数 `--fuser`，默认为 "te"，提供帮助信息
    parser.add_argument(
        "--fuser",
        default="te",
        type=str,
        help="The fuser backend to use. One of: te, old, or none",
    )
    # 添加一个字符串类型的命令行参数 `--executor`，默认为 None，提供帮助信息
    parser.add_argument(
        "--executor",
        default=None,
        type=str,
        help="The executor to use. One of: legacy, simple, profiling",
    )
    # 添加一个整数类型的命令行参数 `--cuda-pointwise-loop-level`，默认为 None
    parser.add_argument(
        "--cuda-pointwise-loop-level",
        "--cuda_pointwise_loop_level",
        default=None,
        type=int,
    )
    # 添加一个整数类型的命令行参数 `--cuda-pointwise-block-count`，默认为 None
    parser.add_argument(
        "--cuda-pointwise-block-count",
        "--cuda_pointwise_block_count",
        default=None,
        type=int,
    )
    # 添加一个整数类型的命令行参数 `--cuda-pointwise-block-size`，默认为 None
    parser.add_argument(
        "--cuda-pointwise-block-size",
        "--cuda_pointwise_block_size",
        default=None,
        type=int,
    )

    # 解析命令行参数并将结果保存到 args 对象中
    args = parser.parse_args()
    # 设置 fuser 和 executor 的值
    set_fuser(args.fuser, args.executor)

    # 如果设置了 `--cuda-pointwise-loop-level` 参数，则设置对应的 Torch 选项
    if args.cuda_pointwise_loop_level:
        torch._C._jit_set_te_cuda_pointwise_loop_levels(args.cuda_pointwise_loop_level)
    # 如果设置了 `--cuda-pointwise-block-count` 参数，则设置对应的 Torch 选项
    if args.cuda_pointwise_block_count:
        torch._C._jit_set_te_cuda_pointwise_block_count(args.cuda_pointwise_block_count)
    # 如果设置了 `--cuda-pointwise-block-size` 参数，则设置对应的 Torch 选项
    if args.cuda_pointwise_block_size:
        torch._C._jit_set_te_cuda_pointwise_block_size(args.cuda_pointwise_block_size)

    # 如果没有指定 `--rnns` 参数，则使用预设的 rnns 列表
    rnns = args.rnns or [
        "cudnn",
        "aten",
        "jit",
        "jit_premul",
        "jit_premul_bias",
        "jit_simple",
        "jit_multilayer",
        "py",
    ]
    # 如果没有指定 `--cnns` 参数，则使用预设的 cnns 列表
    cnns = args.cnns or ["resnet18", "resnet18_jit", "resnet50", "resnet50_jit"]
    # 定义一个变量 vlrnns，包含特定的变量长度的 LSTM 列表
    vlrnns = ["vl_cudnn", "vl_jit", "vl_py"]

    # 如果设置了 `--print-json` 参数，则定义一个函数 print_stderr 用于打印信息
    if args.print_json:
        print_stderr = lambda *args, **kwargs: None  # noqa: E731,F811
    # 打印命令行参数信息
    print_stderr(args)

    # 复制 args 变量的所有值到 bench_args 中，并删除其中的 "group" 键
    bench_args = copy.deepcopy(vars(args))
    del bench_args["group"]

    # 判断是否需要对变长 LSTM 进行基准测试
    should_bench_varlen_lstms = args.variable_lstms
    # 删除 bench_args 字典中的 "rnns" 键及其对应的值
    del bench_args["rnns"]
    # 删除 bench_args 字典中的 "cnns" 键及其对应的值
    del bench_args["cnns"]
    # 删除 bench_args 字典中的 "variable_lstms" 键及其对应的值
    del bench_args["variable_lstms"]
    # 删除 bench_args 字典中的 "fuser" 键及其对应的值
    del bench_args["fuser"]
    # 删除 bench_args 字典中的 "executor" 键及其对应的值
    del bench_args["executor"]
    # 删除 bench_args 字典中的 "cuda_pointwise_loop_level" 键及其对应的值
    del bench_args["cuda_pointwise_loop_level"]
    # 删除 bench_args 字典中的 "cuda_pointwise_block_count" 键及其对应的值
    del bench_args["cuda_pointwise_block_count"]
    # 删除 bench_args 字典中的 "cuda_pointwise_block_size" 键及其对应的值
    del bench_args["cuda_pointwise_block_size"]

    # 初始化空字典 results 用于存储性能测试的结果
    results = {}

    # 如果应该对变长 LSTM 进行性能测试
    if should_bench_varlen_lstms:
        # 如果总循环次数和预热次数大于30，打印警告信息到标准错误输出
        if args.nloops + args.warmup > 30:
            print_stderr(
                "WARNING: some of the variable sequence length lstms are "
                "very unoptimized and therefore take forever to run."
            )
        # 将变长 LSTM 的性能测试结果更新到 results 字典中
        results.update(
            bench_group(vlrnns, "variable-length sequence LSTM", "vl_lstm", bench_args)
        )

    # 如果 args.group 中包含 "rnns"，将 LSTM 的性能测试结果更新到 results 字典中
    if "rnns" in args.group:
        results.update(bench_group(rnns, "LSTM", "lstm", bench_args))
    # 如果 args.group 中包含 "cnns"，将 ResNet 的性能测试结果更新到 results 字典中
    if "cnns" in args.group:
        results.update(bench_group(cnns, "ResNet", "resnet", bench_args))

    # 根据 args.print_json 的值选择性地以 OSS 或 PEP 格式打印 results 字典中的结果
    if args.print_json == "oss":
        print_json_oss_format(results)
    elif args.print_json == "pep":
        print_json_pep_format(results)
```