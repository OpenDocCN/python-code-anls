# `.\pytorch\benchmarks\instruction_counts\main.py`

```py
"""Basic runner for the instruction count microbenchmarks.

The contents of this file are placeholders, and will be replaced by more
expressive and robust components (e.g. better runner and result display
components) in future iterations. However this allows us to excercise the
underlying benchmark generation infrastructure in the mean time.
"""

# 引入命令行参数解析模块
import argparse
# 引入系统模块
import sys
# 引入列表类型提示
from typing import List

# 引入自定义应用程序模块 ci
from applications import ci
# 引入核心扩展模块 materialize
from core.expand import materialize
# 引入标准定义模块 BENCHMARKS
from definitions.standard import BENCHMARKS
# 引入执行模块中的 Runner 类
from execution.runner import Runner
# 引入工作模块中的 WorkOrder 类
from execution.work import WorkOrder

# 定义主函数，接收命令行参数列表，并无返回值
def main(argv: List[str]) -> None:
    # 使用 materialize 函数生成 WorkOrder 对象的元组
    work_orders = tuple(
        WorkOrder(label, autolabels, timer_args, timeout=600, retries=2)
        for label, autolabels, timer_args in materialize(BENCHMARKS)
    )

    # 创建 Runner 对象，运行工作订单并获取结果
    results = Runner(work_orders).run()
    # 遍历工作订单并逐个打印标签、自动标签、定时器线程数和指令数
    for work_order in work_orders:
        print(
            work_order.label,
            work_order.autolabels,
            work_order.timer_args.num_threads,
            results[work_order].instructions,
        )

# 如果作为脚本直接运行
if __name__ == "__main__":
    # 定义不同模式下的处理函数映射
    modes = {
        "debug": main,
        "ci": ci.main,
    }

    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数 --mode，类型为字符串，选择范围为 modes 字典的键列表，默认为 "debug"
    parser.add_argument("--mode", type=str, choices=list(modes.keys()), default="debug")

    # 解析命令行参数，args 包含 mode 参数，remaining_args 包含其它参数
    args, remaining_args = parser.parse_known_args(sys.argv)
    # 根据 mode 参数选择相应的处理函数，并传递剩余参数给它
    modes[args.mode](remaining_args[1:])
```