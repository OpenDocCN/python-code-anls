# `.\pytorch\torch\_functorch\benchmark_utils.py`

```py
# 忽略类型检查错误，针对特定工具的声明
# 导入上下文管理模块
import contextlib
# 导入处理 JSON 数据的模块
import json
# 导入操作符模块
import operator
# 导入操作系统相关功能的模块
import os
# 导入时间相关功能的模块
import time

# 导入 PyTorch 深度学习框架
import torch
# 从 PyTorch 中导入性能分析器相关模块
from torch.profiler import profile, ProfilerActivity


# 空函数，用于同步操作
def synchronize():
    pass


def dump_chrome_trace(
    f,
    input,
    trace_filename,
    optimize_ctx,
    activities,
    num_runs=1,
    devices=None,
    kwargs_for_f=None,
    kwargs_for_profiler=None,
):
    """
    输出函数 f(input, **kwargs_for_f) 在 [optimize_ctx] 环境中，运行 [num_runs] 次的 Chrome Trace 到 [trace_filename]。

    [activities] 是性能分析器记录的活动类型，例如 ProfilerActivity.CUDA。
    返回不使用性能分析器时的总运行时间。

    输出结果到 trace_filename
    """

    if devices is None:
        devices = ["cuda"]

    # 如果未指定设备或设备列表不为 CPU 且 CUDA 可用，则同步函数使用 CUDA 的同步方法
    global synchronize
    if devices != ["cpu"] and torch.cuda.is_available():
        synchronize = torch.cuda.synchronize

    # 如果未指定函数的关键字参数，则初始化为空字典
    if kwargs_for_f is None:
        kwargs_for_f = {}
    # 如果未指定性能分析器的关键字参数，则初始化为空字典
    if kwargs_for_profiler is None:
        kwargs_for_profiler = {}

    # 在优化上下文中执行以下操作
    with optimize_ctx:
        # 设置随机数种子
        torch.manual_seed(1337)
        # 预热运行5次
        for _ in range(5):  # warmup runs
            f(input, **kwargs_for_f)
            synchronize()
        
        torch.manual_seed(1337)
        # 记录主要运行时间
        t0 = time.perf_counter()
        for _ in range(num_runs):
            f(input, **kwargs_for_f)
            synchronize()
        t1 = time.perf_counter()
    timing = t1 - t0

    # 使用性能分析器进行记录
    with profile(activities=activities, **kwargs_for_profiler) as prof:
        with optimize_ctx:
            synchronize()
            torch.manual_seed(1337)
            for _ in range(num_runs):
                f(input, **kwargs_for_f)
                synchronize()
    # 将性能分析结果导出为 Chrome Trace 格式
    prof.export_chrome_trace(trace_filename)

    # 返回总运行时间
    return timing


def get_chrome_trace_events(filename):
    # 打开指定的文件
    f = open(filename)
    # 加载 JSON 数据
    data = json.load(f)
    # 获取跟踪事件列表
    events = data["traceEvents"]
    return events


def is_gpu_compute_event(event):
    # 判断事件是否为 GPU 计算事件的函数
    global gpu_pids
    return (
        "pid" in event
        and event["pid"] in gpu_pids
        and "ph" in event
        and event["ph"] == "X"
    )


def get_sorted_gpu_events(events):
    # 获取排序后的 GPU 事件列表
    sorted_gpu_events = []
    for event in events:
        if not is_gpu_compute_event(event):
            continue
        sorted_gpu_events.append(event)
    # 按时间戳排序事件列表
    return sorted(sorted_gpu_events, key=operator.itemgetter("ts"))


def get_duration(sorted_gpu_events):
    # 计算 GPU 事件列表的总持续时间
    if len(sorted_gpu_events) == 0:
        return 0
    event = sorted_gpu_events[0]
    current_end_time = event["ts"] + event["dur"]
    total_duration = event["dur"]
    for event in sorted_gpu_events[1:]:
        start_time = max(event["ts"], current_end_time)
        end_time = event["ts"] + event["dur"]
        total_duration = total_duration + max(end_time - start_time, 0)
        current_end_time = max(current_end_time, end_time)
    return total_duration


def get_sorted_gpu_mm_conv_events(events):
    # 获取排序后的 GPU 内存管理和卷积事件列表
    # 定义一个函数，用于判断给定的事件是否为矩阵乘法或卷积相关事件
    def is_mm_conv_event(event):
        return "name" in event and (
            "gemm" in event["name"]
            or "conv" in event["name"]
            or "cutlass" in event["name"]
            or "wgrad" in event["name"]
        )
    
    # 获取按照某种顺序排列的 GPU 事件列表
    gpu_events = get_sorted_gpu_events(events)
    
    # 初始化一个空列表，用于存储排序后的事件
    sorted_events = []
    
    # 遍历 GPU 事件列表
    for event in gpu_events:
        # 如果事件不是矩阵乘法或卷积相关事件，则跳过当前事件
        if not is_mm_conv_event(event):
            continue
        # 将满足条件的事件添加到排序后的事件列表中
        sorted_events.append(event)
    
    # 返回排序后的事件列表
    return sorted_events
# 初始化一个空列表，用于存储 GPU 进程的 PID
gpu_pids = []


def compute_utilization(filename: str, total_length: float):
    """
    处理由 pytorch 分析器生成的 Chrome 追踪文件，计算 GPU 利用率以及在矩阵乘法和卷积上花费的时间百分比

    Args:
        filename(str): pytorch 分析器生成的 Chrome 追踪文件名

        total_length(float): 进程总长度（不包括分析器）的秒数

    Return:
        tuple: (GPU 利用率, 矩阵乘法和卷积操作所占时间百分比)
    """
    # 获取 Chrome 追踪文件中的事件
    events = get_chrome_trace_events(filename)

    # 获取 GPU 事件的进程 ID
    global gpu_pids
    gpu_pids = []
    for event in events:
        if "name" not in event:
            continue
        if event["name"] == "process_labels" and "GPU" in event["args"]["labels"]:
            gpu_pids.append(event["pid"])

    # 将总长度转换为微秒
    total_length = total_length * 1e6

    # 获取排序后的 GPU 事件
    sorted_gpu_events = get_sorted_gpu_events(events)

    # 计算 GPU 利用率
    utilization = get_duration(sorted_gpu_events) / total_length

    # 获取排序后的矩阵乘法和卷积事件
    sorted_gpu_mm_conv_events = get_sorted_gpu_mm_conv_events(events)

    # 计算矩阵乘法和卷积操作所占时间百分比
    mm_conv_utilization = get_duration(sorted_gpu_mm_conv_events) / total_length

    return utilization, mm_conv_utilization


def benchmark_utilization(
    f,
    input,
    trace_folder,
    optimize_ctx=None,
    trace_file_name="tmp_chrome_trace",
    num_runs=1,
):
    """
    对运行 f(input, **kwargs_for_f) [num_runs] 次（不包括热身运行）进行 GPU 利用率和矩阵乘法和卷积操作所占时间百分比的基准测试。
    它将在 trace_folder/trace_file_name.json 中生成一个 Chrome 追踪文件。

    Args:
        f: 要进行基准测试的函数

        input: :attr:`f` 的输入

        trace_folder: 存储 Chrome 追踪文件的文件夹名

        optimize_ctx: 运行 f 的上下文

        trace_file_name: 导出的 Chrome 追踪文件名，默认为 "tmp_chrome_trace"

        num_runs: 运行 f 的次数，默认为 1

    Return:
        tuple: (GPU 利用率, 矩阵乘法和卷积操作所占时间百分比)
    """
    # 检查目标文件夹是否存在，不存在则创建
    isExist = os.path.exists(trace_folder)
    if not isExist:
        os.makedirs(trace_folder)
        print("create folder " + trace_folder)

    # 如果 optimize_ctx 为 None，则使用空上下文
    if optimize_ctx is None:
        optimize_ctx = contextlib.nullcontext()

    # 设置导出的 Chrome 追踪文件名
    chrome_trace_file_name = os.path.join(trace_folder, trace_file_name + ".json")

    # 导出 Chrome 追踪文件并获取总长度
    total_length = dump_chrome_trace(
        f,
        input,
        chrome_trace_file_name,
        optimize_ctx,
        [ProfilerActivity.CUDA],
        num_runs=num_runs,
        devices="cuda",
    )

    # 计算 GPU 利用率和矩阵乘法和卷积操作所占时间百分比
    utilization, mm_conv_utilization = compute_utilization(
        chrome_trace_file_name, total_length
    )

    return utilization, mm_conv_utilization
```