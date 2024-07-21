# `.\pytorch\torch\profiler\_utils.py`

```py
# mypy: allow-untyped-defs
# 导入 functools 模块，用于创建偏函数
import functools
# 导入 operator 模块，用于操作符的函数
import operator
# 导入 re 模块，用于正则表达式操作
import re
# 从 collections 模块导入 deque，用于高效实现双向队列
from collections import deque
# 从 dataclasses 模块导入 dataclass 装饰器，用于轻松创建数据类
from dataclasses import dataclass
# 从 typing 模块导入 Dict 和 List 类型提示
from typing import Dict, List, TYPE_CHECKING

# 导入 Torch 的性能分析相关模块
from torch.autograd.profiler import profile
from torch.profiler import DeviceType

# 如果处于类型检查模式，导入 _KinetoEvent 类
if TYPE_CHECKING:
    from torch.autograd import _KinetoEvent


# 定义一个通用的树遍历函数 _traverse
def _traverse(tree, next_fn, children_fn=lambda x: x.children, reverse: bool = False):
    # 定义顺序函数，根据 reverse 参数决定正向还是反向遍历
    order = reversed if reverse else lambda x: x
    # 使用双向队列初始化剩余待处理节点
    remaining = deque(order(tree))
    while remaining:
        # 获取下一个节点
        curr_event = next_fn(remaining)
        yield curr_event  # 返回当前节点
        # 遍历当前节点的子节点
        for child_event in order(children_fn(curr_event)):
            remaining.append(child_event)


# 定义深度优先搜索的树遍历函数
traverse_dfs = functools.partial(_traverse, next_fn=lambda x: x.pop(), reverse=True)
# 定义广度优先搜索的树遍历函数
traverse_bfs = functools.partial(_traverse, next_fn=lambda x: x.popleft(), reverse=False)


# 定义事件度量数据类 EventMetrics
@dataclass
class EventMetrics:
    duration_time_ns: int = 0  # 事件持续时间（纳秒）
    self_time_ns: int = 0  # 事件自身消耗时间（纳秒）
    idle_time_ns: int = 0  # 事件空闲时间（纳秒）
    queue_depth: int = 0  # 事件队列深度

    @property
    def fraction_idle_time(self):
        # 计算空闲时间占总时间的比例
        if self.duration_time_ns == 0:
            return 0.0
        return self.idle_time_ns / self.duration_time_ns


# 定义时间间隔类 Interval
@dataclass
class Interval:
    start: int  # 时间间隔开始时间
    end: int  # 时间间隔结束时间
    queue_depth: int = 0  # 时间间隔队列深度


# 定义事件键类 EventKey
class EventKey:
    def __init__(self, event):
        self.event = event  # 初始化事件

    def __hash__(self):
        return hash(self.event.id)  # 计算事件 ID 的哈希值

    def __eq__(self, other):
        return self.event.id == other.event.id  # 判断两个事件键是否相等

    def __repr__(self):
        return f"{self.event.name}"  # 返回事件名称的字符串表示

    def intervals_overlap(self, intervals: List[Interval]):
        overlap_time = 0  # 初始化重叠时间为 0
        intervals = sorted(intervals, key=lambda x: x.start)  # 按开始时间排序间隔列表

        if intervals:
            # 计算与第一个间隔的重叠时间
            overlap_start = max(self.event.start_time_ns, intervals[0].start)
            overlap_end = min(self.event.end_time_ns, intervals[0].end)

            if overlap_start < overlap_end:
                overlap_time += overlap_end - overlap_start

        i, j = 0, 1
        while j < len(intervals):
            prev_interval = intervals[i]
            curr_interval = intervals[j]
            j += 1
            if prev_interval.end > curr_interval.start:
                # 如果完全被前一个间隔包含，则跳过
                if prev_interval.end > curr_interval.end:
                    j += 1
                    continue
                else:
                    curr_interval.start = prev_interval.end
                    i = j

            overlap_start = max(self.event.start_time_ns, curr_interval.start)
            overlap_end = min(self.event.end_time_ns, curr_interval.end)
            if overlap_start < overlap_end:
                overlap_time += overlap_end - overlap_start

        return overlap_time  # 返回总重叠时间


# 定义基本评估类 BasicEvaluation
class BasicEvaluation:
    pass  # 占位符，表示该类当前没有额外的实现
    # 初始化方法，接受一个 profile 对象作为参数
    def __init__(self, prof: profile):
        # 将传入的 profile 对象赋值给实例变量 self.profile
        self.profile = prof
        # 创建一个空的字典，用于存储事件关键字到事件指标的映射关系
        self.metrics: Dict[EventKey, EventMetrics] = {}
        # 计算事件自身时间
        self.compute_self_time()
        # 对事件关键字进行排序，按照事件的开始时间从早到晚排序
        self.event_keys = sorted(
            (e for e in self.metrics.keys()), key=lambda x: x.event.start_time_ns
        )
        # 获取所有事件对象并存储在 self.events 中
        self.events = [e.event for e in self.event_keys]
        # 初始化一个空的 CUDA 事件列表
        self.cuda_events: List[_KinetoEvent] = []
        # 计算队列深度列表
        self.queue_depth_list = self.compute_queue_depth()
        # 计算空闲时间
        self.compute_idle_time()

    # 计算事件自身时间的方法
    def compute_self_time(self):
        """
        Computes event's self time(total time - time in child ops).
        """
        # 断言确保 kineto_results 不为空
        assert self.profile.kineto_results is not None
        # 使用 deque 数据结构创建一个事件栈，用于深度优先搜索
        stack = deque(self.profile.kineto_results.experimental_event_tree())

        # 标准的深度优先搜索迭代过程
        while stack:
            # 弹出当前事件
            curr_event = stack.pop()
            # 计算当前事件的自身时间
            self_time = curr_event.duration_time_ns
            # 遍历当前事件的子事件，减去子事件的持续时间来计算自身时间
            for child_event in curr_event.children:
                self_time -= child_event.duration_time_ns
                stack.append(child_event)
            # 断言确保同一个事件关键字不会被重复添加到 metrics 中
            assert (
                EventKey(curr_event) not in self.metrics
            ), f"Duplicate id: {curr_event.id}, {curr_event.name}"
            # 将事件关键字及其对应的事件指标存储到 metrics 中
            self.metrics[EventKey(curr_event)] = EventMetrics(self_time_ns=self_time)
            self.metrics[
                EventKey(curr_event)
            ].duration_time_ns = curr_event.duration_time_ns

    # 计算空闲时间的方法
    def compute_idle_time(self):
        """
        Computes idle time of the profile.
        """
        # 根据队列深度列表计算整个事件序列的空闲时间
        idle = False  # 空闲状态标志
        idle_start = 0  # 空闲开始时间初始化为 0
        idle_intervals: List[Interval] = []  # 空闲时间间隔列表初始化为空

        # 如果队列深度列表和事件列表都不为空，则计算空闲时间间隔
        if self.queue_depth_list and self.events:
            idle_intervals += [
                # 添加开始时间到第一个事件的开始时间的间隔
                Interval(self.events[0].start_time_ns, self.queue_depth_list[0].start),
                # 添加最后一个队列深度结束时间到最后一个事件结束时间的间隔
                Interval(self.queue_depth_list[-1].end, self.events[-1].end_time_ns),
            ]

        # 遍历队列深度列表，根据队列深度变化计算空闲时间间隔
        for data_point in self.queue_depth_list:
            if data_point.queue_depth == 0 and not idle:
                idle_start = data_point.end
                idle = True
            if data_point.queue_depth > 0 and idle:
                idle_intervals.append(Interval(idle_start, data_point.start))
                idle = False

        # 获取所有事件关键字，并计算它们的空闲时间
        event_list = [e.event for e in self.metrics.keys()]
        for event in event_list:
            self.metrics[EventKey(event)].idle_time_ns = EventKey(
                event
            ).intervals_overlap(idle_intervals)
    def rank_events(self, length):
        """
        Filter and Rank the events based on some heuristics:
        1) Events that are in the falling phase of the queue depth.
        2) Events that have a high idle_time, self_time difference.

        Parameters:
            length: The number of events to return.
        """

        # 导入 torch 库，用于科学计算
        import torch

        # 将队列深度列表反转
        queue_depth_list = list(reversed(self.queue_depth_list))
        # 提取队列深度值
        qd_values = [e.queue_depth for e in queue_depth_list]

        # 设置队列深度的下限和上限阈值
        bottom_threashold = 0
        top_threashold = 4
        decrease_interval = []
        i = 0
        while i < len(qd_values):
            if qd_values[i] > bottom_threashold:
                i += 1
                continue
            for j in range(i + 1, len(qd_values)):
                # 寻找下一个小于等于 bottom_threashold 的值的索引
                next_minimum_idx = index_of_first_match(
                    qd_values, lambda x: x <= bottom_threashold, start=j
                )
                # 在当前位置 j 和下一个最小值之间找到最大值的索引
                peak_idx = argmax(qd_values, start=j, end=next_minimum_idx)

                # 如果找到有效的峰值，则将其加入减少区间列表并继续搜索
                if peak_idx is not None and qd_values[peak_idx] >= top_threashold:
                    decrease_interval.append(
                        Interval(
                            queue_depth_list[peak_idx].start, queue_depth_list[i].start
                        )
                    )
                    i = next_minimum_idx if next_minimum_idx is not None else i
                    break
            i += 1

        # 根据减少区间过滤掉不在其中的事件
        event_list = [
            event
            for event in self.metrics.keys()
            if event.intervals_overlap(decrease_interval)
        ]

        if event_list:
            # 计算事件的自身时间 self_time_ns 的张量
            self_time = torch.tensor(
                [self.metrics[event].self_time_ns for event in event_list],
                dtype=torch.float32,
            )
            # 计算事件的空闲时间 fraction_idle_time 的张量
            idle_time = torch.tensor(
                [self.metrics[event].fraction_idle_time for event in event_list],
                dtype=torch.float32,
            )
            # 计算空闲时间的标准化增益
            normalized_gain = (idle_time - torch.mean(idle_time)) / torch.std(idle_time)
            # 计算自身时间的标准化值
            normalized_self = (self_time - torch.mean(self_time)) / torch.std(self_time)
            # 计算启发式评分列表
            heuristic_score_list = normalized_gain + 0.6 * normalized_self

            # 根据启发式评分对事件列表进行排序
            event_list = [
                event
                for _, event in sorted(
                    zip(heuristic_score_list, event_list),
                    key=operator.itemgetter(0),
                    reverse=True,
                )
            ]
            # 截取前 length 个事件作为结果
            event_list = event_list[:length]

        # 返回排名靠前的事件列表
        return event_list
    def get_optimizable_events(self, length: int = 1, print_enable: bool = True):
        # 调用对象方法 rank_events，获取排名事件列表
        event_list = self.rank_events(length)
        # 如果 print_enable 为 False，则直接返回事件列表
        if not print_enable:
            return event_list
        # 如果 print_enable 为 True，则构建输出字符串
        # 如果 event_list 不为空，则输出 "Optimizable events:\n"，否则输出 "No events to optimize\n"
        output = "Optimizable events:\n" if event_list else "No events to optimize\n"

        # 将一行分隔线添加到输出字符串
        output += "\n".join(
            [
                f"""{'-'*80}
                # 构造包含 80 个 '-' 的分隔线字符串
def emit_events_summary(event_list, print_enable=False):
    """
    Generate a summary of events.

    Parameters:
    - event_list: List of events to summarize.
    - print_enable: Boolean flag indicating whether to print the summary.

    Returns:
    - event_list: The original list of events.

    This function generates a formatted summary of events. It constructs a detailed output
    string for each event in the event_list, including the event name, source code location
    (if available), and percentage of idle time. If print_enable is True, it prints the 
    generated summary to the console. Finally, it returns the event_list unchanged.
    """
    output = "\n".join(
        [
            f"""
Event:                {event}
Source code location: {source_code_location(event.event)}
Percentage idle time: {self.metrics[event].fraction_idle_time * 100:.2f}%
{'-'*80}"""
            for event in event_list
        ]
    )
    if print_enable:
        print(output)
    return event_list


def index_of_first_match(seq, predicate, start=0, end=None):
    """
    Find the index of the first element in seq that satisfies the predicate.

    Parameters:
    - seq: Sequence to search within.
    - predicate: Function that returns True for the desired element.
    - start: Optional start index for the search (default is 0).
    - end: Optional end index for the search (default is None, meaning end of sequence).

    Returns:
    - Index of the first matching element, or None if no match is found within the specified range.

    This function iterates over the elements of seq from start to end (exclusive), applying 
    the predicate function to each element. It returns the index of the first element that 
    satisfies the predicate, or None if no such element is found.
    """
    if end is None or end >= len(seq):
        end = len(seq)
    for i in range(start, end):
        if predicate(seq[i]):
            return i
    return None


def argmax(seq, key=lambda x: x, start=0, end=None):
    """
    Find the index of the maximum element in seq based on a specified key function.

    Parameters:
    - seq: Sequence in which to find the maximum element.
    - key: Function to extract a comparison key from each element (default is identity function).
    - start: Optional start index for the search (default is 0).
    - end: Optional end index for the search (default is None, meaning end of sequence).

    Returns:
    - Index of the maximum element within the specified range.

    This function computes the index of the maximum element in seq[start:end] based on the 
    key function. If multiple elements are tied for the maximum value, it returns the index 
    of the first such element encountered.
    """
    seq = seq[start:end]
    if len(seq) == 0:
        return None
    return seq.index(max(seq, key=key)) + start


def source_code_location(event):
    """
    Determine the source code location associated with an event.

    Parameters:
    - event: Event object containing information about the event.

    Returns:
    - String representing the source code location, or "No source code location found" if not found.

    This function attempts to find a source code location associated with the given event. It 
    searches through the event's parent chain until it finds a match for a pattern indicating 
    a Python file location (.py(...) format). If found, it returns the corresponding event name 
    as the source code location. If no suitable location is found in the event's ancestry, it 
    returns a default message indicating no source code location was found.
    """
    while event is not None:
        match = re.search(r"\.py\(.*\)", event.name)
        if match is None:
            event = event.parent
            continue
        return event.name
    return "No source code location found"


# Provide an OSS workaround for cudagraphs + CUPTI issue
# https://github.com/pytorch/pytorch/issues/75504
# TODO(dberard) - deprecate / remove workaround for CUDA >= 12, when
# we stop supporting older CUDA versions.
def _init_for_cuda_graphs():
    """
    Initialize for CUDA graphs profiling workaround.

    This function imports profile from torch.autograd.profiler and uses it within a profiling 
    context to initialize for CUDA graphs. This is a workaround for issues related to cudagraphs 
    and CUPTI in certain versions of CUDA. It is intended as a temporary solution until support 
    for older CUDA versions is deprecated or removed.
    """
    from torch.autograd.profiler import profile

    with profile():
        pass
```