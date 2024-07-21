# `.\pytorch\torch\_dynamo\profiler.py`

```py
# mypy: allow-untyped-defs
# 引入必要的模块和库
import dataclasses  # 导入 dataclasses 模块，用于创建数据类
import os  # 导入 os 模块，用于与操作系统交互
from typing import Any, List  # 导入类型提示相关内容

import torch  # 导入 PyTorch 库

from .utils import print_once  # 从当前包的 utils 模块中导入 print_once 函数


@dataclasses.dataclass
class ProfileMetrics:
    # 定义性能指标数据类 ProfileMetrics，用于存储微秒、操作数、融合次数和图的数量
    microseconds: float = 0.0  # 微秒数，默认为 0.0
    operators: int = 0  # 操作数，默认为 0
    fusions: int = 0  # 融合次数，默认为 0
    graphs: int = 0  # 图的数量，默认为 0

    def __iadd__(self, other: "ProfileMetrics"):
        # 实现 += 运算符重载，用于逐项累加另一个 ProfileMetrics 对象的值到当前对象
        self.microseconds += other.microseconds
        self.operators += other.operators
        self.fusions += other.fusions
        return self

    def __add__(self, other: "ProfileMetrics"):
        # 实现 + 运算符重载，用于创建一个新的 ProfileMetrics 对象，包含当前对象和另一个对象的值的总和
        assert isinstance(other, ProfileMetrics)
        return ProfileMetrics(
            self.microseconds + other.microseconds,
            self.operators + other.operators,
            self.fusions + other.fusions,
        )

    def __truediv__(self, other):
        # 实现 / 运算符重载，用于计算当前对象各项值与另一个 ProfileMetrics 对象或整数的比值
        if isinstance(other, int):
            other = ProfileMetrics(other, other, other)
        return ProfileMetrics(
            self.microseconds / max(1, other.microseconds),
            self.operators / max(1, other.operators),
            self.fusions / max(1, other.fusions),
        )

    def __str__(self):
        # 返回描述性能指标对象的字符串，显示操作数和时间占比
        return f"{self.operators:4.0%} ops {self.microseconds:4.0%} time"

    def tocsv(self):
        # 将性能指标对象转换为 CSV 格式，返回包含操作数和微秒数的列表
        return [self.operators, self.microseconds]


class ProfileResult:
    def __init__(self, captured, total, unique_graphs):
        # 定义性能分析结果类 ProfileResult，初始化捕获的性能指标、总体性能指标和唯一图的数量
        self.captured: ProfileMetrics = captured or ProfileMetrics()  # 捕获的性能指标，默认为空对象
        self.total: ProfileMetrics = total or ProfileMetrics()  # 总体性能指标，默认为空对象
        self.unique_graphs: int = unique_graphs  # 唯一图的数量

    def __iadd__(self, other: "ProfileResult"):
        # 实现 += 运算符重载，用于逐项累加另一个 ProfileResult 对象的值到当前对象
        self.captured += other.captured
        self.total += other.total
        self.unique_graphs += other.unique_graphs
        return self

    def percent(self):
        # 计算捕获性能指标与总体性能指标的百分比，并返回结果
        return self.captured / self.total

    def __str__(self):
        # 返回描述性能分析结果对象的字符串，显示唯一图数量、捕获的图调用次数和操作数的比率
        return (
            f"{self.unique_graphs:2} graphs {self.captured.graphs:2} graph calls "
            f"{self.captured.operators:4}/{self.total.operators:4} = "
            + str(self.percent())
        )

    def tocsv(self):
        # 将性能分析结果对象转换为 CSV 格式，返回包含唯一图数量、捕获的图调用次数、操作数和总体操作数的列表
        return [
            self.unique_graphs,
            self.captured.graphs,
            self.captured.operators,
            self.total.operators,
        ] + self.percent().tocsv()


def should_print_missing():
    # 判断是否应打印缺失的信息，根据环境变量 TORCHDYNAMO_PRINT_MISSING 的值
    return os.environ.get("TORCHDYNAMO_PRINT_MISSING") == "1"


def print_missing(stack):
    # 打印缺失的信息，但不包括特定的栈帧路径
    if any("/torch/autograd/profiler.py" in x for x in stack):
        return
    stack = [
        x for x in stack if ("<built-in" not in x and "site-packages/torch/" not in x)
    ]
    print_once("MISSING", " >> ".join(stack[-3:]))


class Profiler:
    # 定义性能分析器类 Profiler
    unique_graphs = 0  # 类级别的唯一图数量初始化为 0

    def __init__(self):
        # 初始化方法，创建一个 Torch 分析器对象 prof
        self.prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],  # 指定分析的活动类型为 CPU
            with_stack=should_print_missing(),  # 根据环境变量判断是否应打印缺失信息的堆栈
        )
    def results(self):
        # 初始化捕获的区域数、操作数、捕获的微秒数、总操作数、总微秒数
        captured_regions = 0
        captured_ops = 0
        captured_microseconds = 0
        total_ops = 0
        total_microseconds = 0

        # 初始化上一个操作结束时间和捕获区域结束时间
        last_op_end_time = -1
        captured_region_end_time = -1
        
        # 获取按照事件起始时间排序的事件列表
        events = sorted(self.prof.events(), key=lambda x: x.time_range.start)
        
        # 遍历事件列表
        for e in events:
            if e.name == "TORCHDYNAMO":
                # 如果事件名为 TORCHDYNAMO，则更新捕获区域结束时间并增加捕获的区域数
                captured_region_end_time = e.time_range.end
                captured_regions += 1
                # 忽略记录函数初始化中的 torch.zeros(1)
                total_ops -= 1
            elif e.time_range.start >= last_op_end_time:
                # 如果事件起始时间大于等于上一个操作结束时间，则更新上一个操作结束时间
                last_op_end_time = e.time_range.end
                if e.time_range.end <= captured_region_end_time:
                    # 如果事件结束时间在捕获区域结束时间之前，则增加捕获的操作数和捕获的微秒数
                    captured_ops += 1
                    captured_microseconds += e.time_range.elapsed_us()
                elif should_print_missing():
                    # 否则，如果应该打印缺失的信息，则打印事件堆栈信息
                    print_missing(e.stack)
                # 增加总操作数和总微秒数
                total_ops += 1
                total_microseconds += e.time_range.elapsed_us()
            else:
                # 其他情况下，忽略递归调用的操作
                pass

        # 获取唯一的图形数量，并重置静态变量 Profiler.unique_graphs
        unique_graphs = Profiler.unique_graphs
        Profiler.unique_graphs = 0
        # 由于分析器设置代码中计算了一个额外的操作，因此减去一个总操作数
        total_ops -= 1

        # 返回分析结果对象
        return ProfileResult(
            captured=ProfileMetrics(
                microseconds=captured_microseconds,
                operators=captured_ops,
                fusions=captured_ops - captured_regions,
                graphs=captured_regions,
            ),
            total=ProfileMetrics(
                microseconds=total_microseconds,
                operators=total_ops,
                fusions=total_ops - 1,
            ),
            unique_graphs=unique_graphs,
        )
# 定义一个函数，用于在 Torch FX 图模块上插入性能分析的包装
def fx_insert_profiling(gm: torch.fx.GraphModule, example_inputs: List[Any]):
    # 定义一个内部函数，用于包装原始的 forward 方法，并添加性能分析记录
    def _wrapped(*args):
        # 使用 Torch Profiler 记录函数执行时间，标签为 "TORCHDYNAMO"
        with torch.profiler.record_function("TORCHDYNAMO"):
            # 调用原始图模块的 forward 方法
            return gm.forward(*args)

    # 增加 Profiler 类的静态变量，用于跟踪已处理的图数量
    Profiler.unique_graphs += 1
    # 返回包装后的函数
    return _wrapped
```