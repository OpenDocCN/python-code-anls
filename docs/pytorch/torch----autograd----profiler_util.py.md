# `.\pytorch\torch\autograd\profiler_util.py`

```py
    # 启用 mypy 的未类型定义检查
    # 导入标准库模块
    import bisect
    import itertools
    import math

    # 导入 collections 中的 defaultdict 和 namedtuple
    from collections import defaultdict, namedtuple
    # 导入 operator 模块中的 attrgetter 函数
    from operator import attrgetter

    # 导入 typing 模块中的类型提示
    from typing import Any, Dict, List, Optional, Tuple
    # 导入 typing_extensions 中的 deprecated 类型
    from typing_extensions import deprecated

    # 导入 torch 库
    import torch
    # 从 torch.autograd 中导入 DeviceType 类
    from torch.autograd import DeviceType

    # 定义 __all__ 列表，指定公开的模块成员
    __all__ = [
        "EventList",
        "FormattedTimesMixin",
        "Interval",
        "Kernel",
        "FunctionEvent",
        "FunctionEventAvg",
        "StringTable",
        "MemRecordsAcc",
    ]


    class EventList(list):
        """A list of Events (for pretty printing)."""

        def __init__(self, *args, **kwargs):
            # 初始化 EventList 对象
            # 设置是否使用设备
            use_device = kwargs.pop("use_device", None)
            # 是否记录内存使用情况
            profile_memory = kwargs.pop("profile_memory", False)
            # 是否包含 FLOPs (Floating Point Operations) 统计信息
            with_flops = kwargs.pop("with_flops", False)
            super().__init__(*args, **kwargs)
            # 存储属性值
            self._use_device = use_device
            self._profile_memory = profile_memory
            self._tree_built = False
            self._with_flops = with_flops

        def _build_tree(self):
            # 构建事件列表的树形结构
            self._populate_cpu_children()
            self._remove_dup_nodes()
            self._set_backward_stacktraces()
            self._tree_built = True

        def __str__(self):
            # 返回事件列表的字符串表示形式
            return self.table()

        def _remove_dup_nodes(self):
            # 移除重复节点
            while True:
                to_delete = set()
                for idx in range(len(self)):
                    if (
                        self[idx].cpu_parent is not None
                        and self[idx].cpu_parent.name == self[idx].name
                        and len(self[idx].cpu_parent.cpu_children) == 1
                    ):
                        # 将子节点移到父节点上，并提升内核信息
                        self[idx].cpu_parent.cpu_children = self[idx].cpu_children
                        self[idx].cpu_parent.kernels = self[idx].kernels  # lift kernels up
                        for ch in self[idx].cpu_children:
                            ch.cpu_parent = self[idx].cpu_parent
                        to_delete.add(idx)
                if len(to_delete) == 0:
                    break
                # 创建新的事件列表，不包括要删除的事件
                new_evts = [ev for ind, ev in enumerate(self) if ind not in to_delete]
                self.clear()
                self.extend(new_evts)

        def _set_backward_stacktraces(self):
            # 设置反向传播的堆栈跟踪信息
            def bw_parent(evt):
                if evt is None:
                    return None
                elif evt.scope == 1:  # BACKWARD_FUNCTION
                    return evt
                else:
                    return bw_parent(evt.cpu_parent)

            fwd_stacks = {}
            for evt in self:
                if bw_parent(evt) is None and evt.stack is not None:
                    t = (evt.sequence_nr, evt.thread)
                    if t not in fwd_stacks:
                        fwd_stacks[t] = evt.stack

            for evt in self:
                p = bw_parent(evt)
                if p is not None:
                    assert p.fwd_thread is not None
                    t = (p.sequence_nr, p.fwd_thread)
                    if t in fwd_stacks:
                        evt.stack = fwd_stacks[t]
                    else:
                        evt.stack = []

        @property
    # 计算所有事件中self_cpu_time_total属性的总和，并返回结果
    def self_cpu_time_total(self):
        return sum(event.self_cpu_time_total for event in self)

    # 将EventList格式化为一个漂亮的表格并打印出来
    def table(
        self,
        sort_by=None,  # 指定用于排序条目的属性。默认按照注册顺序打印。
        row_limit=100,  # 表格行数限制，默认为100行。
        max_src_column_width=75,  # 源代码列的最大宽度限制，默认为75个字符。
        max_name_column_width=55,  # 名称列的最大宽度限制，默认为55个字符。
        max_shapes_column_width=80,  # 形状列的最大宽度限制，默认为80个字符。
        header=None,  # 表格的标题，默认为None。
        top_level_events_only=False,  # 是否仅显示顶层事件的布尔标志。默认为False。
    ):
        """Print an EventList as a nicely formatted table.

        Args:
            sort_by (str, optional): Attribute used to sort entries. By default
                they are printed in the same order as they were registered.
                Valid keys include: ``cpu_time``, ``cuda_time``, ``xpu_time``,
                ``cpu_time_total``, ``cuda_time_total``, ``xpu_time_total``,
                ``cpu_memory_usage``, ``cuda_memory_usage``, ``xpu_memory_usage``,
                ``self_cpu_memory_usage``, ``self_cuda_memory_usage``,
                ``self_xpu_memory_usage``, ``count``.
            top_level_events_only(bool, optional): Boolean flag to determine the
                selection of events to display. If true, the profiler will only
                display events at top level like top-level invocation of python
                `lstm`, python `add` or other functions, nested events like low-level
                cpu/cuda/xpu ops events are omitted for profiler result readability.

        Returns:
            A string containing the table.
        """
        # 调用内部函数_build_table来构建并返回格式化后的表格字符串
        return _build_table(
            self,
            sort_by=sort_by,
            row_limit=row_limit,
            max_src_column_width=max_src_column_width,
            max_name_column_width=max_name_column_width,
            max_shapes_column_width=max_shapes_column_width,
            header=header,
            profile_memory=self._profile_memory,
            with_flops=self._with_flops,
            top_level_events_only=top_level_events_only,
        )
    def export_chrome_trace(self, path):
        """将 EventList 导出为 Chrome 跟踪工具文件。

        可以在后续通过 ``chrome://tracing`` URL 加载和检查生成的跟踪文件。

        Args:
            path (str): 跟踪文件将被写入的路径。
        """
        import os  # 导入操作系统相关的模块

        # 根据是否使用设备来确定设备名称，如果未指定则使用默认的 "cuda"
        device_name = "cuda" if not self._use_device else self._use_device

        with open(path, "w") as f:  # 打开文件以写入模式
            chrome_events = []  # 初始化 Chrome 跟踪事件列表
            next_id = 0  # 初始化事件 ID 计数器

            # 使用文件 IO 而不是 json.dump，因为后者速度较慢，这种技术已被证明可以提升 4 倍速度
            f.write("[")  # 写入 JSON 数组的起始标志

            for evt in self:  # 遍历事件列表中的每个事件
                if evt.trace_name is None:  # 如果事件没有跟踪名称，则跳过
                    continue

                # 写入 Chrome 跟踪事件的开始部分
                f.write(
                    '{{"name": "{}", '  # 事件名称
                    '"ph": "X", '  # 事件类型标志为 "X"
                    '"ts": {}, '  # 事件开始时间戳
                    '"dur": {}, '  # 事件持续时间
                    '"tid": {}, '  # 线程 ID
                    '"pid": "CPU functions", '  # 进程 ID
                    '"args": {{}}}}, '.format(  # 事件的附加参数为空字典
                        evt.trace_name,  # 使用事件的跟踪名称
                        evt.time_range.start,  # 获取事件的起始时间
                        evt.time_range.elapsed_us(),  # 获取事件持续的微秒数
                        evt.thread  # 获取事件所在的线程 ID
                        if not evt.is_remote  # 如果事件不是远程事件
                        else f'" node_id:{evt.node_id}, thread_id:{evt.thread} "',  # 使用节点 ID 和线程 ID
                    )
                )

                for k in evt.kernels:  # 遍历事件的内核操作
                    # 写入 CPU 到 GPU 内核操作的起始部分
                    f.write(
                        f'{{"name": "{evt.trace_name}", '  # 事件名称
                        '"ph": "s", '  # 事件类型标志为 "s"
                        f'"ts": {evt.time_range.start}, '  # 内核操作开始时间戳
                        f'"tid": {evt.thread}, '  # 线程 ID
                        '"pid": "CPU functions", '  # 进程 ID
                        f'"id": {next_id}, '  # 内核操作 ID
                        f'"cat": "cpu_to_{device_name}", '  # 操作类别
                        '"args": {}}, '  # 附加参数为空字典
                    )
                    # 注意：使用 torch.profiler 获取设备内核跟踪信息
                    next_id += 1  # 增加下一个内核操作的 ID

            if len(self) > 0:
                # 去除末尾的空格和逗号
                f.seek(f.tell() - 2, os.SEEK_SET)
                f.truncate()

            f.write("]")  # 写入 JSON 数组的结束标志

    def supported_export_stacks_metrics(self):
        return [
            "self_cpu_time_total",
            "self_cuda_time_total",
            "self_xpu_time_total",
            "self_privateuse1_time_total",
        ]
    def export_stacks(self, path: str, metric: str):
        # 检查指定的 metric 是否在支持的导出堆栈的指标列表中，如果不在则抛出 ValueError 异常
        if metric not in self.supported_export_stacks_metrics():
            raise ValueError(
                "metric should be one of: "
                + str(self.supported_export_stacks_metrics())
            )
        # 准备用于字符转换的转换表，将空格、制表符和换行符替换为下划线
        translate_table = str.maketrans(" ;\t\n", "____")
        # 打开指定路径的文件，以写入模式
        with open(path, "w") as f:
            # 遍历当前对象的事件列表
            for evt in self:
                # 如果事件具有堆栈信息且堆栈长度大于 0
                if evt.stack and len(evt.stack) > 0:
                    # 获取指定 metric 对应的属性值，将特定字符串替换为统一的设备字符串
                    metric_value = getattr(
                        evt,
                        metric.replace("cuda", "device")
                        .replace("xpu", "device")
                        .replace("privateuse1", "device"),
                    )
                    # 如果 metric_value 转换为整数大于 0
                    if int(metric_value) > 0:
                        # 初始化堆栈字符串
                        stack_str = ""
                        # 反向遍历事件的堆栈信息
                        for entry in reversed(evt.stack):
                            # 将堆栈条目按照 translate_table 进行字符转换并拼接
                            stack_str += entry.translate(translate_table)
                            stack_str += ";"
                        # 去除最后一个多余的分号并添加 metric_value 后写入文件
                        stack_str = stack_str[:-1] + " " + str(int(metric_value))
                        f.write(stack_str + "\n")

    def key_averages(self, group_by_input_shapes=False, group_by_stack_n=0):
        """Averages all function events over their keys.

        Args:
            group_by_input_shapes: 是否按输入形状分组，而不仅仅是事件名称。
                这对于查看哪些输入形状对运行时贡献最大很有用，并可能有助于特定大小的优化或选择最佳的量化候选项（即适应性拟合顶线）

            group_by_stack_n: 按前 n 个堆栈跟踪条目分组

        Returns:
            包含 FunctionEventAvg 对象的 EventList。
        """
        # 确保树结构已构建
        assert self._tree_built
        # 使用 defaultdict 创建统计字典，键为元组，值为 FunctionEventAvg 对象
        stats: Dict[Tuple[str, ...], FunctionEventAvg] = defaultdict(FunctionEventAvg)

        def get_key(event, group_by_input_shapes, group_by_stack_n) -> Tuple[str, ...]:
            # 构建事件的关键字列表
            key = [
                str(event.key),
                str(event.node_id),
                str(event.device_type),
                str(event.is_legacy),
            ]
            # 如果按输入形状分组，则添加输入形状到关键字中
            if group_by_input_shapes:
                key.append(str(event.input_shapes))
            # 如果按堆栈条目数分组，则将事件的堆栈前 group_by_stack_n 个条目添加到关键字中
            if group_by_stack_n > 0:
                key += event.stack[:group_by_stack_n]
            return tuple(key)

        # 遍历当前对象的事件列表，并将每个事件添加到统计字典中对应的 FunctionEventAvg 对象中
        for evt in self:
            stats[get_key(evt, group_by_input_shapes, group_by_stack_n)].add(evt)

        # 创建一个 EventList 对象，其中包含 stats 字典的值作为初始事件列表
        avg_list = EventList(
            stats.values(),
            use_device=self._use_device,
            profile_memory=self._profile_memory,
            with_flops=self._with_flops,
        )
        # 对于 avg_list 中的每个事件，截取堆栈信息至 group_by_stack_n 条目
        for evt in avg_list:
            evt.stack = evt.stack[:group_by_stack_n]
            # 如果不按输入形状分组，则将输入形状置为空字符串
            if not group_by_input_shapes:
                evt.input_shapes = ""
        # 返回包含平均值的事件列表
        return avg_list
    def total_average(self):
        """Averages all events.

        Returns:
            A FunctionEventAvg object.
        """
        # 创建一个空的 FunctionEventAvg 对象，用于存储总和和平均值
        total_stat = FunctionEventAvg()
        
        # 遍历 self 中的每一个事件（假设 self 是一个包含事件的容器）
        for evt in self:
            # 将当前事件 evt 添加到 total_stat 中，这里假设 FunctionEventAvg 类支持重载操作符 +
            total_stat += evt
            # 将 key 属性设置为 None，可能用于标记不同的状态
            total_stat.key = None
        
        # 将 key 属性设置为 "Total"，表明这是总和的标识
        total_stat.key = "Total"
        
        # 返回存储总和和平均值的 FunctionEventAvg 对象
        return total_stat
# 定义了一个函数，用于格式化时间，以便在 FunctionEvent 中使用
def _format_time(time_us):
    """Define how to format time in FunctionEvent."""
    US_IN_SECOND = 1000.0 * 1000.0  # 定义常量，微秒到秒的转换因子
    US_IN_MS = 1000.0  # 定义常量，微秒到毫秒的转换因子
    if time_us >= US_IN_SECOND:  # 如果时间大于等于一秒的微秒数
        return f"{time_us / US_IN_SECOND:.3f}s"  # 将时间转换为秒并格式化输出
    if time_us >= US_IN_MS:  # 如果时间大于等于一毫秒的微秒数
        return f"{time_us / US_IN_MS:.3f}ms"  # 将时间转换为毫秒并格式化输出
    return f"{time_us:.3f}us"  # 否则直接输出微秒

# 定义了一个函数，用于格式化时间占比，以便在 FunctionEvent 中使用
def _format_time_share(time_us, total_time_us):
    """Define how to format time in FunctionEvent."""
    if total_time_us == 0:  # 如果总时间为零
        assert time_us == 0, f"Expected time_us == 0 but got {time_us}"  # 验证时间为零
        return "NaN"  # 返回非数值标识
    return f"{time_us * 100.0 / total_time_us:.2f}%"  # 计算时间占比并格式化输出

# 定义了一个函数，用于格式化内存大小，以便在 FunctionEvent 中使用
def _format_memory(nbytes):
    """Return a formatted memory size string."""
    KB = 1024  # 定义常量，字节到千字节的转换因子
    MB = 1024 * KB  # 定义常量，字节到兆字节的转换因子
    GB = 1024 * MB  # 定义常量，字节到吉字节的转换因子
    if abs(nbytes) >= GB:  # 如果字节数的绝对值大于或等于一吉字节
        return f"{nbytes * 1.0 / GB:.2f} Gb"  # 将字节数转换为吉字节并格式化输出
    elif abs(nbytes) >= MB:  # 如果字节数的绝对值大于或等于一兆字节
        return f"{nbytes * 1.0 / MB:.2f} Mb"  # 将字节数转换为兆字节并格式化输出
    elif abs(nbytes) >= KB:  # 如果字节数的绝对值大于或等于一千字节
        return f"{nbytes * 1.0 / KB:.2f} Kb"  # 将字节数转换为千字节并格式化输出
    else:
        return str(nbytes) + " b"  # 否则直接输出字节数

# 定义了一个属性格式化器的函数，用于在 FunctionEvent 中处理时间属性的格式化
def _attr_formatter(name):
    return property(lambda self: _format_time(getattr(self, name)))

# 定义了一个帮助类，为 FunctionEvent 和 FunctionEventAvg 提供辅助方法
class FormattedTimesMixin:
    """Helpers for FunctionEvent and FunctionEventAvg.

    The subclass should define `*_time_total` and `count` attributes.
    """

    cpu_time_str = _attr_formatter("cpu_time")  # 格式化 CPU 时间属性
    device_time_str = _attr_formatter("device_time")  # 格式化设备时间属性
    cpu_time_total_str = _attr_formatter("cpu_time_total")  # 格式化总 CPU 时间属性
    device_time_total_str = _attr_formatter("device_time_total")  # 格式化总设备时间属性
    self_cpu_time_total_str = _attr_formatter("self_cpu_time_total")  # 格式化自身总 CPU 时间属性
    self_device_time_total_str = _attr_formatter("self_device_time_total")  # 格式化自身总设备时间属性

    @property
    def cpu_time(self):
        return 0.0 if self.count == 0 else 1.0 * self.cpu_time_total / self.count  # 计算并返回 CPU 时间，如果 count 为零则返回零

    @property
    def device_time(self):
        return 0.0 if self.count == 0 else 1.0 * self.device_time_total / self.count  # 计算并返回设备时间，如果 count 为零则返回零

    @property
    @deprecated(
        "`cuda_time` is deprecated, please use `device_time` instead.",
        category=FutureWarning,
    )
    def cuda_time(self):  # 标记为即将弃用的 CUDA 时间属性
        return self.device_time  # 返回设备时间作为 CUDA 时间

# 定义了一个时间间隔类，用于计算时间间隔的微秒数
class Interval:
    def __init__(self, start, end):
        self.start = start  # 起始时间
        self.end = end  # 结束时间

    def elapsed_us(self):
        r"""
        Returns the length of the interval
        """
        return self.end - self.start  # 返回时间间隔的微秒数

# 定义了一个命名元组 Kernel，用于表示内核的名称、设备和持续时间
Kernel = namedtuple("Kernel", ["name", "device", "duration"])

# 定义了一个类 FunctionEvent，用于记录单个函数的性能分析信息
class FunctionEvent(FormattedTimesMixin):
    """Profiling information about a single function."""
    # 初始化函数，用于创建一个 FunctionEvent 对象
    def __init__(
        self,
        id,
        name,
        thread,
        start_us,
        end_us,
        fwd_thread=None,
        input_shapes=None,
        stack=None,
        scope=0,
        use_device=None,
        cpu_memory_usage=0,
        device_memory_usage=0,
        is_async=False,
        is_remote=False,
        sequence_nr=-1,
        node_id=-1,
        device_type=DeviceType.CPU,
        device_index=0,
        device_resource_id=None,
        is_legacy=False,
        flops=None,
        trace_name=None,
        concrete_inputs=None,
    ):
        # 设置 FunctionEvent 对象的属性
        self.id: int = id
        self.node_id: int = node_id
        self.name: str = name
        self.trace_name: str = trace_name
        self.time_range: Interval = Interval(start_us, end_us)
        self.thread: int = thread
        self.fwd_thread: Optional[int] = fwd_thread
        self.kernels: List[Kernel] = []  # 初始化空的 Kernel 列表
        self.count: int = 1  # 初始化计数器为 1
        self.cpu_children: List[FunctionEvent] = []  # 初始化空的 CPU 子事件列表
        self.cpu_parent: Optional[FunctionEvent] = None  # 初始化 CPU 父事件为 None
        self.input_shapes: Tuple[int, ...] = input_shapes  # 设置输入形状元组
        self.concrete_inputs: List[Any] = concrete_inputs  # 设置具体输入列表
        self.stack: List = stack  # 设置堆栈列表
        self.scope: int = scope  # 设置作用域
        self.use_device: Optional[str] = use_device  # 设置使用的设备
        self.cpu_memory_usage: int = cpu_memory_usage  # 设置 CPU 内存使用量
        self.device_memory_usage: int = device_memory_usage  # 设置设备内存使用量
        self.is_async: bool = is_async  # 设置是否异步标志
        self.is_remote: bool = is_remote  # 设置是否远程标志
        self.sequence_nr: int = sequence_nr  # 设置序列号
        self.device_type: DeviceType = device_type  # 设置设备类型
        self.device_index: int = device_index  # 设置设备索引
        self.device_resource_id: int = (
            thread if device_resource_id is None else device_resource_id
        )  # 设置设备资源 ID
        self.is_legacy: bool = is_legacy  # 设置是否遗留代码标志
        self.flops: Optional[int] = flops  # 设置 FLOPS

    # 向 kernels 列表中添加一个新的 Kernel 对象
    def append_kernel(self, name, device, duration):
        assert self.device_type == DeviceType.CPU  # 断言当前设备类型为 CPU
        self.kernels.append(Kernel(name, device, duration))  # 添加新的 Kernel 对象到 kernels 列表

    # 向 cpu_children 列表中添加一个新的 CPU 子事件对象
    def append_cpu_child(self, child):
        """Append a CPU child of type FunctionEvent.

        One is supposed to append only direct children to the event to have
        correct self cpu time being reported.
        """
        assert self.device_type == DeviceType.CPU  # 断言当前设备类型为 CPU
        assert isinstance(child, FunctionEvent)  # 断言 child 是 FunctionEvent 类型的对象
        assert child.device_type == DeviceType.CPU  # 断言 child 的设备类型为 CPU
        self.cpu_children.append(child)  # 添加 child 到 cpu_children 列表

    # 设置当前 CPU 事件对象的 CPU 父事件
    def set_cpu_parent(self, parent):
        """Set the immediate CPU parent of type FunctionEvent.

        One profiling FunctionEvent should have only one CPU parent such that
        the child's range interval is completely inside the parent's. We use
        this connection to determine the event is from top-level op or not.
        """
        assert self.device_type == DeviceType.CPU  # 断言当前设备类型为 CPU
        assert isinstance(parent, FunctionEvent)  # 断言 parent 是 FunctionEvent 类型的对象
        assert parent.device_type == DeviceType.CPU  # 断言 parent 的设备类型为 CPU
        self.cpu_parent = parent  # 设置当前 CPU 事件对象的 CPU 父事件为 parent
    # Note: async events don't have children, are not used when computing 'self'
    # metrics of other events, have only total cpu time

    @property
    def self_cpu_memory_usage(self):
        # 如果事件是异步的或者设备类型不是CPU，返回0
        if self.is_async or self.device_type != DeviceType.CPU:
            return 0
        # 返回自身的CPU内存使用量减去所有子事件的CPU内存使用量之和
        return self.cpu_memory_usage - sum(
            child.cpu_memory_usage for child in self.cpu_children
        )

    @property
    def self_device_memory_usage(self):
        # 如果事件是异步的或者设备类型不是CPU，返回0
        if self.is_async or self.device_type != DeviceType.CPU:
            return 0
        # 返回自身的设备内存使用量减去所有子事件的设备内存使用量之和
        return self.device_memory_usage - sum(
            child.device_memory_usage for child in self.cpu_children
        )

    @property
    @deprecated(
        "`self_cuda_memory_usage` is deprecated. Use `self_device_memory_usage` instead.",
        category=FutureWarning,
    )
    def self_cuda_memory_usage(self):  # To be deprecated
        # 返回自身的CUDA内存使用量（已废弃，建议使用self_device_memory_usage）
        return self.self_device_memory_usage

    @property
    def cpu_time_total(self):
        # 如果设备类型是CPU，返回时间范围内的总CPU时间
        if self.device_type == DeviceType.CPU:
            return self.time_range.elapsed_us()
        else:
            return 0

    @property
    def self_cpu_time_total(self):
        # 如果事件是异步的或者设备类型不是CPU，返回0
        if self.is_async or self.device_type != DeviceType.CPU:
            return 0
        # 返回自身的CPU时间减去所有子事件的CPU时间之和
        return self.cpu_time_total - sum(
            child.cpu_time_total for child in self.cpu_children
        )

    @property
    def device_time_total(self):
        # 如果事件是异步的或者不使用设备，返回0
        if self.is_async or not self.use_device:
            return 0
        # 如果设备类型是CPU
        if self.device_type == DeviceType.CPU:
            # 如果不是旧版本事件，计算子操作中的内核持续时间总和和所有子事件的设备时间总和
            if not self.is_legacy:
                return sum(kinfo.duration for kinfo in self.kernels) + sum(
                    ch.device_time_total for ch in self.cpu_children
                )
            else:
                # 对于旧版本CPU事件，每个事件有一个虚拟内核的持续时间总和
                return sum(kinfo.duration for kinfo in self.kernels)
        else:
            # 对于CUDA和私有设备，返回时间范围内的总设备时间
            assert self.device_type in [DeviceType.CUDA, DeviceType.PrivateUse1]
            return self.time_range.elapsed_us()

    @property
    @deprecated(
        "`cuda_time_total` is deprecated. Use `device_time_total` instead.",
        category=FutureWarning,
    )
    def cuda_time_total(self):  # To be deprecated
        # 返回自身的CUDA时间总量（已废弃，建议使用device_time_total）
        return self.device_time_total

    @property
    def self_device_time_total(self):
        # 如果事件是异步的或者不使用设备，返回0
        if self.is_async or not self.use_device:
            return 0
        # 如果设备类型是CPU，返回自身的设备时间减去所有子事件的设备时间之和
        if self.device_type == DeviceType.CPU:
            return self.device_time_total - sum(
                [child.device_time_total for child in self.cpu_children]
            )
        else:
            # 对于CUDA和私有设备，返回自身的设备时间总量
            assert self.device_type in [DeviceType.CUDA, DeviceType.PrivateUse1]
            return self.device_time_total

    @property
    @deprecated(
        "`self_cuda_time_total` is deprecated. Use `self_device_time_total` instead.",
        category=FutureWarning,
    )
    # 返回对象的 self_device_time_total 属性，该方法将被弃用
    def self_cuda_time_total(self):  # To be deprecated
        return self.self_device_time_total

    # 返回对象的 name 属性作为 key
    @property
    def key(self):
        return self.name

    # 返回对象的详细字符串表示，包括各种属性的值
    def __repr__(self):
        # 获取使用的设备名称
        device_name = self.use_device
        # 获取设备时间字符串
        device_time = self.device_time_str
        # 获取设备内存使用情况
        device_memory_usage = self.device_memory_usage
        # 返回对象的详细字符串表示，包括各种属性的值
        return (
            f"<FunctionEvent id={self.id} name={self.name} device_type={self.device_type} node_id={self.node_id} "
            f"cpu_time={self.cpu_time_str} start_us={self.time_range.start} end_us={self.time_range.end} "
            f"cpu_children={str([child.id for child in self.cpu_children])} {device_name}_time={device_time} "
            f"name={self.name} thread={self.thread} input_shapes={str(self.input_shapes)} "
            f"cpu_memory_usage={self.cpu_memory_usage} {device_name}_memory_usage={device_memory_usage} "
            f"is_async={self.is_async} is_remote={self.is_remote} seq_nr={self.sequence_nr} is_legacy={self.is_legacy}>"
        )
    # FunctionEventAvg 类，用于计算多个 FunctionEvent 对象的平均统计信息，并继承自 FormattedTimesMixin
    class FunctionEventAvg(FormattedTimesMixin):
    
        """Used to average stats over multiple FunctionEvent objects."""
        # 初始化方法，设置各种实例变量的默认值
        def __init__(self):
            self.key: Optional[str] = None  # 标识符字符串，可能为 None
            self.count: int = 0  # 计数器，记录添加的 FunctionEvent 或 FunctionEventAvg 对象数量
            self.node_id: int = 0  # 节点 ID，初始化为 0
            self.is_async: bool = False  # 是否异步执行的标志，默认为 False
            self.is_remote: bool = False  # 是否远程执行的标志，默认为 False
            self.use_device: Optional[str] = None  # 使用的设备名称，可能为 None
            self.cpu_time_total: int = 0  # CPU 总时间，初始化为 0
            self.device_time_total: int = 0  # 设备总时间，初始化为 0
            self.self_cpu_time_total: int = 0  # 自身 CPU 时间总和，初始化为 0
            self.self_device_time_total: int = 0  # 自身设备时间总和，初始化为 0
            self.input_shapes: Optional[List[List[int]]] = None  # 输入形状的列表，可能为 None
            self.stack: Optional[List] = None  # 栈信息的列表，可能为 None
            self.scope: Optional[int] = None  # 作用域的整数值，可能为 None
            self.cpu_memory_usage: int = 0  # CPU 内存使用量，初始化为 0
            self.device_memory_usage: int = 0  # 设备内存使用量，初始化为 0
            self.self_cpu_memory_usage: int = 0  # 自身 CPU 内存使用量总和，初始化为 0
            self.self_device_memory_usage: int = 0  # 自身设备内存使用量总和，初始化为 0
            self.cpu_children: Optional[List[FunctionEvent]] = None  # CPU 子事件列表，可能为 None
            self.cpu_parent: Optional[FunctionEvent] = None  # CPU 父事件，可能为 None
            self.device_type: DeviceType = DeviceType.CPU  # 设备类型，默认为 CPU
            self.is_legacy: bool = False  # 是否是遗留代码的标志，默认为 False
            self.flops: int = 0  # FLOPS 数量，初始化为 0
        
        # 添加方法，将另一个 FunctionEvent 或 FunctionEventAvg 对象的统计信息合并到当前对象
        def add(self, other):
            if self.key is None:
                # 如果当前对象的 key 属性为 None，则将其初始化为 other 对象的 key 属性
                # 作为第一个被记录的 FunctionEventAvg 对象，需要传播其他字段的值
                self.key = other.key
                self.node_id = other.node_id
                self.is_async = other.is_async
                self.is_remote = other.is_remote
                self.cpu_parent = other.cpu_parent
                self.cpu_children = other.cpu_children
        
                self.input_shapes = other.input_shapes
                self.stack = other.stack
                self.scope = other.scope
                self.device_type = other.device_type
                self.is_legacy = other.is_legacy
                self.use_device = other.use_device
        
            assert isinstance(other, (FunctionEvent, FunctionEventAvg))  # 断言 other 是 FunctionEvent 或 FunctionEventAvg 类型的对象
            assert other.key == self.key  # 断言 other 对象的 key 属性与当前对象的 key 属性相等
            
            # 将 other 对象的各项统计信息累加到当前对象的对应属性上
            self.cpu_time_total += other.cpu_time_total
            self.device_time_total += other.device_time_total
            self.self_cpu_time_total += other.self_cpu_time_total
            self.self_device_time_total += other.self_device_time_total
            self.cpu_memory_usage += other.cpu_memory_usage
            self.device_memory_usage += other.device_memory_usage
            self.self_cpu_memory_usage += other.self_cpu_memory_usage
            self.self_device_memory_usage += other.self_device_memory_usage
            self.count += other.count
            
            # 如果当前对象的 flops 属性为 None，则将其初始化为 other 对象的 flops 属性
            # 否则将 other 对象的 flops 属性加到当前对象的 flops 属性上
            if self.flops is None:
                self.flops = other.flops
            elif other.flops is not None:
                self.flops += other.flops
            
            return self  # 返回当前对象
        
        # 增量赋值方法，将另一个 FunctionEvent 或 FunctionEventAvg 对象的统计信息合并到当前对象（重载 +=
        def __iadd__(self, other):
            return self.add(other)  # 调用 add 方法实现 +=
    # 定义对象的字符串表示形式，用于返回对象的详细信息
    def __repr__(self):
        # 根据使用的设备情况确定设备名称，默认为 "cuda"，除非指定了特定设备
        device_name = "cuda" if not self.use_device else self.use_device
        # 获取对象的总自身CPU时间的字符串表示
        self_device_time = self.self_device_time_total_str
        # 获取对象的CPU时间的字符串表示
        device_time = self.device_time_str
        # 获取对象的设备内存使用情况
        device_memory = self.device_memory_usage
        # 返回对象的字符串表示，包括对象的关键键值、总自身CPU时间、CPU时间、
        # 自身设备时间、设备时间、输入形状、CPU内存使用情况和设备内存使用情况等信息
        return (
            f"<FunctionEventAvg key={self.key} self_cpu_time={self.self_cpu_time_total_str} cpu_time={self.cpu_time_str} "
            f"self_{device_name}_time={self_device_time} {device_name}_time={device_time} input_shapes={str(self.input_shapes)} "
            f"cpu_memory_usage={self.cpu_memory_usage} {device_name}_memory_usage={device_memory}>"
        )
class StringTable(defaultdict):
    # 自定义的字典类，继承自 defaultdict
    def __missing__(self, key):
        # 在键缺失时执行的方法，处理类似 't'（解码为 'unsigned short'）的情况，
        # 目前仅通过检查长度来避免对短序列产生意外结果
        self[key] = torch._C._demangle(key) if len(key) > 1 else key
        return self[key]


class MemRecordsAcc:
    """Acceleration structure for accessing mem_records in interval."""

    def __init__(self, mem_records):
        # 初始化方法，接受 mem_records 参数作为构造器
        self._mem_records = mem_records
        # 用于存储记录起始时间戳的列表
        self._start_nses: List[int] = []
        # 记录 mem_records 索引的列表
        self._indices: List[int] = []
        if len(mem_records) > 0:
            # 对 mem_records 中的元素按起始时间戳进行排序，并保存起始时间戳和索引
            tmp = sorted([(r[0].start_ns(), i) for i, r in enumerate(mem_records)])
            self._start_nses, self._indices = zip(*tmp)  # type: ignore[assignment]

    def in_interval(self, start_us, end_us):
        r"""
        Return all records in the given interval
        To maintain backward compatibility, convert us to ns in function
        """
        # 返回给定时间区间内的所有记录
        start_idx = bisect.bisect_left(self._start_nses, start_us * 1000)
        end_idx = bisect.bisect_right(self._start_nses, end_us * 1000)
        for i in range(start_idx, end_idx):
            yield self._mem_records[self._indices[i]]


def _filter_stack_entry(entry):
    # 过滤器函数，用于排除特定的栈条目
    filtered_entries = [
        ("autograd/__init__", "_make_grads"),
        ("autograd/__init__", "backward"),
        ("torch/tensor", "backward"),
        ("_internal/common_utils", "prof_callable"),
        ("_internal/common_utils", "prof_func_call"),
        ("_internal/common_utils", "prof_meth_call"),
    ]
    return all(not (f[0] in entry and f[1] in entry) for f in filtered_entries)


MEMORY_EVENT_NAME = "[memory]"
OUT_OF_MEMORY_EVENT_NAME = "[OutOfMemory]"


def _filter_name(name):
    # 名称过滤器函数，用于忽略以下实用操作的名称
    filtered_out_names = [
        MEMORY_EVENT_NAME,  # 仅用于顶层内存事件
        OUT_OF_MEMORY_EVENT_NAME,
        "profiler::_record_function_enter",
        "profiler::_record_function_enter_new",
        "profiler::_record_function_exit",
        "aten::is_leaf",
        "aten::output_nr",
        "aten::_version",
    ]
    return name in filtered_out_names


# Demangles and optionally rewrites the provided event name,
# with_wildcard - whether to replace certain numbered event names
# with a wildcard name to aggregate them together in the profiler table
# output
def _rewrite_name(name, with_wildcard=False):
    # 解码并可选地重写提供的事件名称，
    # with_wildcard - 是否将某些编号的事件名称替换为通配符名称以便在分析器表中聚合
    string_table = StringTable()
    name = string_table[name]
    if with_wildcard:
        if name.startswith("ProfilerStep#"):
            name = "ProfilerStep*"
    return name


def _build_table(
    events,
    sort_by=None,
    header=None,
    row_limit=100,
    max_src_column_width=75,
    max_name_column_width=55,
    max_shapes_column_width=80,
    with_flops=False,
    profile_memory=False,
    top_level_events_only=False,
):
    """Print a summary of events (which can be a list of FunctionEvent or FunctionEventAvg)."""
    # 构建表格的函数，打印事件的摘要（可以是 FunctionEvent 或 FunctionEventAvg 的列表）
    # 如果事件列表为空，返回空字符串
    if len(events) == 0:
        return ""

    # 检查事件列表中是否存在使用设备时间的事件
    has_device_time = any(event.self_device_time_total > 0 for event in events)
    
    # 检查事件列表中是否存在设备内存使用的事件
    has_device_mem = any(event.self_device_memory_usage > 0 for event in events)
    
    # 获取第一个事件的 use_device 属性
    use_device = events[0].use_device
    
    # 如果 use_device 为 None 并且存在使用设备时间的事件，则抛出运行时错误
    # 该错误表明虽然 use_device 为 None，但却有设备性能数据
    if not use_device and has_device_time:
        raise RuntimeError("use_device is None, but there is device performance data.")

    # 检查事件列表中是否存在输入形状信息
    has_input_shapes = any(
        (event.input_shapes is not None and len(event.input_shapes) > 0)
        for event in events
    )

    # 如果指定了排序方式，则按照指定的属性对事件列表进行排序
    if sort_by is not None:
        events = EventList(
            sorted(
                events,
                key=lambda evt: getattr(
                    evt,
                    sort_by.replace("cuda", "device")
                    .replace("xpu", "device")
                    .replace("privateuse1", "device"),
                ),
                reverse=True,
            ),
            use_device=use_device,
            profile_memory=profile_memory,
            with_flops=with_flops,
        )

    # 计算名称列的宽度，确保至少能容纳最长名称加上一些额外空间
    name_column_width = max(len(evt.key) for evt in events) + 4
    if max_name_column_width is not None:
        name_column_width = min(name_column_width, max_name_column_width)

    # 计算形状列的宽度，确保至少能容纳最长形状描述字符串加上一些额外空间
    shapes_column_width = max(len(str(evt.input_shapes)) for evt in events) + 4
    if max_shapes_column_width is not None:
        shapes_column_width = min(shapes_column_width, max_shapes_column_width)

    # 默认的 FLOPS 列宽度
    DEFAULT_COLUMN_WIDTH = 12
    flops_column_width = DEFAULT_COLUMN_WIDTH

    # 收集事件中的调用栈信息
    src_column_width = None
    stacks = []
    for evt in events:
        if evt.stack is not None and len(evt.stack) > 0:
            stacks.append(evt.stack)
    has_stack = len(stacks) > 0
    
    # 如果存在调用栈信息，则计算调用栈列的宽度
    if has_stack:
        src_column_width = (
            max(max(len(entry) for entry in stack) for stack in stacks) + 4
        )
        if max_src_column_width is not None:
            src_column_width = min(src_column_width, max_src_column_width)

    # 定义表格的列标题
    headers = [
        "Name",
        "Self CPU %",
        "Self CPU",
        "CPU total %",
        "CPU total",
        "CPU time avg",
    ]
    
    # 获取设备名称（大写形式），如果 use_device 为 None 则设为 "None"
    device_name = use_device.upper() if use_device is not None else "None"
    
    # 如果存在使用设备时间的事件，则添加设备性能相关的列标题
    if has_device_time:
        headers.extend(
            [
                f"Self {device_name}",
                f"Self {device_name} %",
                f"{device_name} total",
                f"{device_name} time avg",
            ]
        )
    
    # 如果开启了内存分析，则添加内存相关的列标题
    if profile_memory:
        headers.extend(
            [
                "CPU Mem",
                "Self CPU Mem",
            ]
        )
        
        # 如果使用了设备且存在设备内存使用的事件，则添加设备内存相关的列标题
        if use_device and has_device_mem:
            headers.extend(
                [
                    f"{device_name} Mem",
                    f"Self {device_name} Mem",
                ]
            )
    
    # 添加最后的调用次数列标题
    headers.append("# of Calls")
    # 只有在事件中存在有效的（>= 0）Node ID 时才添加 Node ID 到 headers 中
    append_node_id = any(evt.node_id != -1 for evt in events)
    if append_node_id:
        headers.append("Node ID")

    # 使用列表存储行格式、标题分隔线和行长度，因为 nonlocal 只在 Python 3 中支持
    SPACING_SIZE = 2
    row_format_lst = [""]
    header_sep_lst = [""]
    line_length_lst = [-SPACING_SIZE]

    def add_column(padding, text_dir=">"):
        # 添加列到行格式和标题分隔线列表中
        row_format_lst[0] += (
            "{: " + text_dir + str(padding) + "}" + (" " * SPACING_SIZE)
        )
        header_sep_lst[0] += "-" * padding + (" " * SPACING_SIZE)
        line_length_lst[0] += padding + SPACING_SIZE

    def auto_scale_flops(flops):
        # 根据浮点操作数（FLOPs）自动调整 FLOPs 的标题
        flop_headers = [
            "FLOPs",
            "KFLOPs",
            "MFLOPs",
            "GFLOPs",
            "TFLOPs",
            "PFLOPs",
        ]
        assert flops > 0
        log_flops = max(0, min(math.log10(flops) / 3, float(len(flop_headers) - 1)))
        assert log_flops >= 0 and log_flops < len(flop_headers)
        return (pow(10, (math.floor(log_flops) * -3.0)), flop_headers[int(log_flops)])

    # 添加名称列的宽度到列格式中
    add_column(name_column_width)
    # 为 headers 中的每个元素添加默认列宽
    for _ in headers[1:]:
        add_column(DEFAULT_COLUMN_WIDTH)

    if has_input_shapes:
        # 如果存在输入形状，则添加 "Input Shapes" 到 headers 中，并设置对应的列宽
        headers.append("Input Shapes")
        add_column(shapes_column_width)

    if has_stack:
        # 如果存在堆栈信息，则添加 "Source Location" 到 headers 中，并设置对应的列宽和对齐方式
        headers.append("Source Location")
        add_column(src_column_width, text_dir="<")

    if with_flops:
        # 自动缩放 FLOPs 的标题
        raw_flops = []
        for evt in events:
            if evt.flops > 0:
                raw_flops.append(evt.flops)
        if len(raw_flops) != 0:
            (flops_scale, flops_header) = auto_scale_flops(min(raw_flops))
            # 添加总体 FLOPs 的标题到 headers 中，并设置对应的列宽
            headers.append(f"Total {flops_header}")
            add_column(flops_column_width)
        else:
            with_flops = False  # 无法找到任何有效的 FLOPs 数据

    # 提取并累加每个事件的 self_cpu_time_total 和 self_device_time_total
    sum_self_cpu_time_total = 0
    sum_self_device_time_total = 0
    for evt in events:
        sum_self_cpu_time_total += evt.self_cpu_time_total
        if evt.device_type == DeviceType.CPU and evt.is_legacy:
            # 在旧版分析器中，核心信息存储在 CPU 事件中
            sum_self_device_time_total += evt.self_device_time_total
        elif evt.device_type in [DeviceType.CUDA, DeviceType.PrivateUse1]:
            # 在 kineto 分析器中，使用正确的设备类型存储设备时间
            sum_self_device_time_total += evt.self_device_time_total

    # 实际打印输出
    if header is not None:
        # 添加等号作为分隔线，并添加标题
        append("=" * line_length)
        append(header)
    # 如果只显示顶层事件统计信息
    if top_level_events_only:
        # 添加一行等号，用于分隔不同部分
        append("=" * line_length)
        # 添加提示信息，表明只显示顶层操作的统计数据
        append("This report only display top-level ops statistics")
    
    # 添加表头分隔行
    append(header_sep)
    # 添加格式化后的表头行到结果列表中
    append(row_format.format(*headers))

    # 添加表头分隔行
    append(header_sep)

    # 定义一个函数，用于截断路径字符串以适应给定的列宽
    def trim_path(path, src_column_width):
        if len(path) > src_column_width:
            # 计算需要截断的长度
            offset = len(path) - src_column_width
            # 对路径进行截断
            path = path[offset:]
            # 如果路径长度仍大于3，添加省略号
            if len(path) > 3:
                path = "..." + path[3:]
        return path

    # 初始化事件限制数为0
    event_limit = 0

    # 添加表头分隔行
    append(header_sep)
    # 添加自身 CPU 时间总和的格式化输出到结果列表中
    append(f"Self CPU time total: {_format_time(sum_self_cpu_time_total)}")
    
    # 如果存在设备时间信息
    if has_device_time:
        # 添加自身设备时间总和的格式化输出到结果列表中，根据是否指定了设备类型决定显示的设备类型
        append(
            f"Self {use_device.upper() if use_device is not None else 'None'} "
            f"time total: {_format_time(sum_self_device_time_total)}"
        )
    
    # 将结果列表中的所有字符串连接成一个完整的报告并返回
    return "".join(result)
```