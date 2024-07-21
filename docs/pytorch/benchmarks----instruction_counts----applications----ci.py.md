# `.\pytorch\benchmarks\instruction_counts\applications\ci.py`

```
"""Collect instruction counts for continuous integration."""
# 导入必要的模块和库
import argparse  # 导入解析命令行参数的模块
import hashlib  # 导入用于生成哈希值的模块
import json  # 导入处理 JSON 数据的模块
import time  # 导入处理时间的模块
from typing import Dict, List, Union  # 导入类型提示相关的模块

from core.expand import materialize  # 从核心模块中导入 materialize 函数
from definitions.standard import BENCHMARKS  # 从标准定义模块中导入基准测试数据
from execution.runner import Runner  # 从执行模块中导入 Runner 类
from execution.work import WorkOrder  # 从工作模块中导入 WorkOrder 类


REPEATS = 5  # 定义重复执行次数为 5
TIMEOUT = 600  # Seconds  # 设置超时时间为 600 秒
RETRIES = 2  # 设置重试次数为 2

VERSION = 0  # 设置版本号为 0
MD5 = "4d55e8abf881ad38bb617a96714c1296"  # 预定义 MD5 值用于校验


def main(argv: List[str]) -> None:
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=str, default=None)  # 解析目标文件路径参数
    parser.add_argument("--subset", action="store_true")  # 解析子集模式开关参数
    args = parser.parse_args(argv)  # 解析传入的命令行参数列表

    t0 = int(time.time())  # 获取当前时间的时间戳，并转为整数
    version = VERSION  # 初始化版本号为预设的版本号
    benchmarks = materialize(BENCHMARKS)  # 根据定义的基准测试数据，生成具体的基准测试列表

    # 如果处于调试模式或者未指定目标文件路径，则使用子集模式
    in_debug_mode = args.subset or args.destination is None
    if args.subset:
        version = -1  # 如果是子集模式，则将版本号设为 -1
        benchmarks = benchmarks[:10]  # 如果是子集模式，则只取前 10 个基准测试数据

    # 根据基准测试数据和重复次数生成工作任务列表
    work_orders = tuple(
        WorkOrder(label, autolabels, timer_args, timeout=TIMEOUT, retries=RETRIES)
        for label, autolabels, timer_args in benchmarks * REPEATS
    )

    # 生成工作任务列表的键集合，并计算其 MD5 值
    keys = tuple({str(work_order): None for work_order in work_orders}.keys())
    md5 = hashlib.md5()
    for key in keys:
        md5.update(key.encode("utf-8"))

    # 如果计算得到的 MD5 值与预设值不符且不是子集模式，则修改版本号并发出警告
    if md5.hexdigest() != MD5 and not args.subset:
        version = -1
        print(f"WARNING: Expected {MD5}, got {md5.hexdigest()} instead")

    # 运行工作任务列表，并获取结果
    results = Runner(work_orders, cadence=30.0).run()

    # 初始化用于存储分组结果的字典，结构为键为任务字符串，值为包含时间和计数列表的字典
    grouped_results: Dict[str, Dict[str, List[Union[float, int]]]] = {
        key: {"times": [], "counts": []} for key in keys
    }

    # 将运行结果按任务字符串分组存储到 grouped_results 中
    for work_order, r in results.items():
        key = str(work_order)
        grouped_results[key]["times"].extend(r.wall_times)
        grouped_results[key]["counts"].extend(r.instructions)

    # 构建最终的结果字典，包含版本号、MD5 值、开始和结束时间以及分组结果
    final_results = {
        "version": version,
        "md5": md5.hexdigest(),
        "start_time": t0,
        "end_time": int(time.time()),
        "values": grouped_results,
    }

    # 如果指定了目标文件路径，则将最终结果以 JSON 格式写入该文件
    if args.destination:
        with open(args.destination, "w") as f:
            json.dump(final_results, f)

    # 如果处于调试模式，则输出部分结果并进入调试模式
    if in_debug_mode:
        result_str = json.dumps(final_results)
        print(f"{result_str[:30]} ... {result_str[-30:]}\n")
        import pdb
        pdb.set_trace()  # 进入调试模式
```