# `.\pytorch\torch\_inductor\wrapper_benchmark.py`

```py
#`
# 允许未指定类型的函数定义
# mypy: allow-untyped-defs
import dataclasses  # 导入 dataclasses 模块，提供数据类功能
import tempfile  # 导入 tempfile 模块，提供创建临时文件的功能
from collections import defaultdict  # 导入 defaultdict，提供默认字典类型

import torch  # 导入 torch 库，用于张量计算和深度学习功能
from torch.autograd import DeviceType  # 从 torch.autograd 导入 DeviceType，用于设备类型定义
from .runtime.runtime_utils import (  # 从当前包的 runtime.runtime_utils 模块导入函数
    create_bandwidth_info_str,  # 导入 create_bandwidth_info_str 函数，创建带宽信息字符串
    do_bench_gpu,  # 导入 do_bench_gpu 函数，执行 GPU 基准测试
    get_num_bytes,  # 导入 get_num_bytes 函数，获取数据字节数
)

# 定义内核类别的选择列表
_kernel_category_choices = [
    "foreach",  # 遍历
    "persistent_reduction",  # 持久化归约
    "pointwise",  # 点运算
    "reduction",  # 归约
    "split_scan",  # 分裂扫描
    "template",  # 模板
]


def get_kernel_category_by_source_code(src_code):
    """
    根据源代码获取内核类别。调用此 API 之前需要先将源代码编译为模块。
    """
    # 从选择列表中筛选出包含特定装饰器的内核类别
    choices = [
        ch for ch in _kernel_category_choices if f"@triton_heuristics.{ch}" in src_code
    ]
    # 如果找到一个类别，返回该类别；否则，返回 "unknown"
    if len(choices) == 1:
        return choices[0]
    else:
        return "unknown"


def get_kernel_category(kernel_mod):
    """
    给定定义 Triton 内核的模块，返回内核类别。
    类别可以是：pointwise, reduction, persistent_reduction。
    根据内核导入的装饰器来简单判断类别。
    """
    # 从选择列表中筛选出在模块字典中存在的类别
    choices = [ch for ch in _kernel_category_choices if ch in kernel_mod.__dict__]
    # 如果找到一个类别，返回该类别；否则，返回 "unknown"
    if len(choices) == 1:
        return choices[0]
    else:
        return "unknown"


def get_triton_kernel(mod):
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner  # 从特定路径导入 CachingAutotuner 类

    # 从模块字典中筛选出所有以 "triton_" 开头且为 CachingAutotuner 实例的项目
    cand_list = [
        v
        for k, v in mod.__dict__.items()
        if k.startswith("triton_") and isinstance(v, CachingAutotuner)
    ]
    # 确保只找到一个候选内核，否则抛出断言错误
    assert len(cand_list) == 1
    return cand_list[0]


def benchmark_all_kernels(benchmark_name, benchmark_all_configs):
    """
    一个实验性 API，仅在 config.benchmark_kernel 为真时使用。
    为 PyCodeCache 中缓存的所有内核运行基准测试。
    用于编译后的模块。
    将此方法放在这里而不是代码生成中是为了方便，因为其实现不会因不同图模块的编译而变化。
    """
    from torch._inductor.codecache import PyCodeCache  # 从特定路径导入 PyCodeCache 类

    nfound = 0  # 初始化找到的内核数量为 0
    # 遍历 PyCodeCache 缓存中的每个键值对，其中键是 kernel_key，值是 kernel_mod
    for kernel_key, kernel_mod in PyCodeCache.cache.items():
        # 如果 kernel_mod 没有 get_args 方法或者没有 call 方法，则跳过本次循环
        if not hasattr(kernel_mod, "get_args") or not hasattr(kernel_mod, "call"):
            continue

        # 获取 Triton 内核对象
        triton_kernel = get_triton_kernel(kernel_mod)
        # 获取内核的类别信息
        kernel_category = get_kernel_category(kernel_mod)
        # 调用 kernel_mod 的 get_args 方法获取参数
        args = kernel_mod.get_args()
        # 计算 triton_kernel 中以 "in_out_ptr" 开头的参数个数，作为输入输出指针的数量
        num_in_out_ptrs = len(
            [
                arg_name
                for arg_name in triton_kernel.fn.arg_names
                if arg_name.startswith("in_out_ptr")
            ]
        )
        # 获取 Triton 内核的全局字节数，如果没有则计算基于参数的字节数除以十亿的结果
        num_gb = triton_kernel.inductor_meta.get("kernel_num_gb", None)
        if num_gb is None:
            num_gb = get_num_bytes(*args, num_in_out_args=num_in_out_ptrs) / 1e9

        # 定义一个函数，用于生成内核信息字符串
        def get_info_str(ms, n_regs, n_spills, shared, prefix=""):
            # 如果 n_regs, n_spills, shared 都不为 None，则生成详细的内核信息字符串
            if not any(x is None for x in [n_regs, n_spills, shared]):
                kernel_detail_str = (
                    f"  {n_regs:3} regs  {n_spills:3} spills  {shared:8} shared mem"
                )
            else:
                kernel_detail_str = ""

            # 计算每秒的传输带宽并生成带宽信息字符串
            gb_per_s = num_gb / (ms / 1e3)
            return create_bandwidth_info_str(
                ms, num_gb, gb_per_s, prefix=prefix, suffix=kernel_detail_str
            )

        # 生成内核描述字符串，包括基准名称、内核类别的前三个字母大写、内核键的前十个字符
        kernel_desc = (
            f"{benchmark_name:20} {kernel_category[:3].upper()} {kernel_key[:10]}"
        )
        # 如果需要测试所有配置，则执行以下代码块
        if benchmark_all_configs:
            # 确保 kernel_mod 具有 benchmark_all_configs 方法
            assert hasattr(kernel_mod, "benchmark_all_configs")
            # 对 kernel_mod 使用所有配置进行基准测试，并获取结果
            bench_result = kernel_mod.benchmark_all_configs(args)
            # 打印内核描述信息
            print(kernel_desc)
            # 遍历每个配置及其对应的执行时间并打印
            for launcher, ms in bench_result.items():
                print(
                    f"  {get_info_str(ms, launcher.n_regs, launcher.n_spills, launcher.shared)} @ {launcher.config}"
                )
        else:
            # 否则，使用 do_bench_gpu 函数对 kernel_mod 进行 GPU 基准测试，获取执行时间
            ms = do_bench_gpu(lambda: kernel_mod.call(args), rep=40, fast_flush=True)
            # 确保 Triton 内核只选择了一个最佳配置
            assert (
                len(triton_kernel.launchers) == 1
            ), "Autotuner should have selected the best config"
            # 获取唯一的 launcher
            launcher = triton_kernel.launchers[0]
            # 打印带有内核描述信息的带宽信息字符串
            print(
                get_info_str(
                    ms,
                    launcher.n_regs,
                    launcher.n_spills,
                    launcher.shared,
                    prefix=f"{kernel_desc} ",
                )
            )

        # 增加找到的内核计数
        nfound += 1
    # 如果找到的内核计数为零，则打印未找到基准功能内核的消息
    if nfound == 0:
        print(
            "No kernel with benchmark functionality found. Make sure you run inductor with config.benchmark_kernel being True"
        )
@dataclasses.dataclass
class ProfileEvent:
    category: str  # 事件类别，存储事件的分类信息，例如 "triton_pointwise"
    key: str  # 事件的关键字，用于标识特定的事件
    self_cuda_time_ms: float  # 事件的自身 CUDA 执行时间（毫秒）
    # benchmark 被运行多次，我们对所有运行的计数进行平均。应该是整数，但定义为 float 以防万一。
    count: float  # 事件发生的次数，平均数，用于统计计算次数的平均值


def parse_profile_event_list(benchmark_name, event_list, wall_time_ms, nruns):
    def get_self_cuda_time(ev):
        """
        ev.self_cuda_time_total is in microsecond. Convert to millisecond.
        """
        # 计算事件的自身 CUDA 执行时间，并将微秒转换为毫秒
        return ev.self_cuda_time_total / 1000 / nruns

    all_events = defaultdict(list)  # 创建一个默认值为列表的字典，用于存储所有事件的列表

    def add_event(ev, category):
        # 创建 ProfileEvent 对象，并添加到相应类别的事件列表中
        profile_ev = ProfileEvent(
            category=category,
            key=ev.key,
            self_cuda_time_ms=get_self_cuda_time(ev),
            count=ev.count / nruns,  # 平均值，所有运行的计数
        )
        all_events[category].append(profile_ev)

    for ev in event_list:
        assert not ev.is_legacy, "Don't support the legacy profiler"  # 断言：不支持旧的性能分析器
        if ev.device_type == DeviceType.CPU:
            # 如果事件在 CPU 上，则忽略该事件
            continue

        category = "unknown"  # 默认类别为 "unknown"
        if ev.key.startswith("triton_"):
            if ev.key.startswith("triton_poi"):
                category = "triton_pointwise"  # 如果以 "triton_poi" 开头，则类别为 "triton_pointwise"
            elif ev.key.startswith("triton_red"):
                category = "triton_reduction"  # 如果以 "triton_red" 开头，则类别为 "triton_reduction"
            elif ev.key.startswith("triton_per"):
                category = "triton_persistent_reduction"  # 如果以 "triton_per" 开头，则类别为 "triton_persistent_reduction"
            else:
                category = "triton_unknown"  # 否则，类别为 "triton_unknown"

        add_event(ev, category)  # 将事件添加到相应的类别中

    def report_category(category, profile_events):
        from tabulate import tabulate

        profile_events.sort(key=lambda ev: ev.self_cuda_time_ms, reverse=True)  # 根据 CUDA 执行时间对事件列表进行排序

        rows = []
        total_time = 0.0
        print(f"\n  == {category} category kernels == ")
        for ev in profile_events:
            total_time += ev.self_cuda_time_ms
            percent = f"{ev.self_cuda_time_ms / wall_time_ms * 100:.2f}%"  # 计算事件所占总时间的百分比
            rows.append([ev.key[:120], ev.self_cuda_time_ms, ev.count, percent])  # 添加每个事件的详细信息到行中
        rows.append(
            ["Total", total_time, "", f"{total_time / wall_time_ms * 100:.2f}%"]
        )  # 添加总时间的行信息
        print(
            tabulate(
                rows, headers=["Kernel", "Self CUDA TIME (ms)", "Count", "Percent"]
            )  # 使用 tabulate 打印表格化的事件信息
        )
        return total_time  # 返回总时间
    # 定义一个名为 report 的函数，用于生成性能报告
    def report():
        # 定义一个包含不同类别名称的列表
        category_list = [
            "triton_pointwise",
            "triton_reduction",
            "triton_persistent_reduction",
            "triton_unknown",
            "unknown",
        ]
        
        # 使用断言检查所有事件的键是否是 category_list 的子集，如果不是则抛出异常并显示当前所有事件的键
        assert set(all_events.keys()).issubset(set(category_list)), f"{list(all_events.keys())}"
    
        # 初始化一个空字典，用于存储每个类别的墙上时间
        per_category_wall_time = {}
        # 初始化总的 CUDA 时间为 0.0
        total_cuda_ms = 0.0
        
        # 遍历每个类别
        for category in category_list:
            # 如果当前类别存在于 all_events 中
            if category in all_events:
                # 调用 report_category 函数，获取该类别的时间并存储到 per_category_wall_time 中
                _time = report_category(category, all_events[category])
                per_category_wall_time[category] = _time
                # 累加总的 CUDA 时间
                total_cuda_ms += _time
    
        # 计算 GPU 繁忙的百分比
        gpu_busy_percent = f"{total_cuda_ms / wall_time_ms * 100:.2f}%"
        # 打印 GPU 繁忙的百分比和总的墙上时间
        print(f"\nPercent of time when GPU is busy: {gpu_busy_percent}")
        print(f"Total wall time {wall_time_ms:.3f} ms")
    
        # 输出一个用于汇总的行，以便从所有编译模块和基准测试中收集此类行，并制表
        # 列：benchmark_name, pointwise_percent, reduction_percent, persistent_reduction_percent,
        # unknown_category_percent, GPU_busy_percent, wall_time_ms
        tabulate_line = f"Output for tabulate: {benchmark_name}"
        # 遍历每个类别，计算其占总墙上时间的百分比并添加到汇总行中
        for category in category_list:
            percent = f"{per_category_wall_time.get(category, 0.0) / wall_time_ms * 100:.2f}%"
            tabulate_line += f", {percent}"
        # 添加 GPU 繁忙百分比和总的墙上时间到汇总行中
        tabulate_line += f", {gpu_busy_percent}, {wall_time_ms:.3f}ms"
    
        # 打印汇总行
        print(tabulate_line)
    
    # 调用 report 函数，开始生成报告
    report()
def compiled_module_main(benchmark_name, benchmark_compiled_module_fn):
    """
    This is the function called in __main__ block of a compiled module.
    """
    # 导入 argparse 模块，用于解析命令行参数
    import argparse

    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser()
    
    # 添加命令行参数 --benchmark-kernels 或 -k，用于标志是否对每个单独的内核进行基准测试
    parser.add_argument(
        "--benchmark-kernels",
        "-k",
        action="store_true",
        help="Whether to benchmark each individual kernels",
    )
    
    # 添加命令行参数 --benchmark-all-configs 或 -c，用于标志是否对每个内核的每个配置进行基准测试
    parser.add_argument(
        "--benchmark-all-configs",
        "-c",
        action="store_true",
        help="Whether to benchmark each individual config for a kernel",
    )
    
    # 添加命令行参数 --profile 或 -p，用于标志是否对编译模块进行性能分析
    parser.add_argument(
        "--profile",
        "-p",
        action="store_true",
        help="Whether to profile the compiled module",
    )
    
    # 解析命令行参数
    args = parser.parse_args()

    # 如果命令行参数中包含 --benchmark-kernels 标志
    if args.benchmark_kernels:
        # 调用 benchmark_all_kernels 函数，传入基准测试名称和是否基准测试所有配置的参数
        benchmark_all_kernels(benchmark_name, args.benchmark_all_configs)
    else:
        # 定义基准测试运行次数和重复次数
        times = 10
        repeat = 10
        
        # 调用 benchmark_compiled_module_fn 函数进行基准测试，并计算其运行时间
        wall_time_ms = benchmark_compiled_module_fn(times=times, repeat=repeat) * 1000

        # 如果没有指定 --profile 标志，则直接返回
        if not args.profile:
            return

        # 使用 torch.profiler 进行性能分析
        with torch.profiler.profile(record_shapes=True) as p:
            benchmark_compiled_module_fn(times=times, repeat=repeat)

        # 指定性能分析结果输出路径
        path = f"{tempfile.gettempdir()}/compiled_module_profile.json"
        
        # 将性能分析结果导出为 Chrome trace 格式
        p.export_chrome_trace(path)
        
        # 输出编译模块基准测试的性能分析结果信息
        print(f"Profiling result for a compiled module of benchmark {benchmark_name}:")
        print(f"Chrome trace for the profile is written to {path}")
        
        # 获取并打印性能分析事件列表，按 CUDA 时间总和进行排序，并设置行数限制为 10
        event_list = p.key_averages(group_by_input_shape=True)
        print(event_list.table(sort_by="self_cuda_time_total", row_limit=10))
        
        # 调用 parse_profile_event_list 函数，解析性能分析事件列表
        parse_profile_event_list(
            benchmark_name, event_list, wall_time_ms, times * repeat
        )
```