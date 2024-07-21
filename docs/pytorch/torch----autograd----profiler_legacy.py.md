# `.\pytorch\torch\autograd\profiler_legacy.py`

```py
# 设置 mypy 让未定义的函数可以被使用
mypy: allow-untyped-defs
# 导入 itertools 模块，用于生成迭代器
import itertools
# 导入 warnings 模块，用于发出警告信息
import warnings
# 从 typing_extensions 中导入 deprecated 装饰器，用于标记过时的函数或类
from typing_extensions import deprecated

# 导入 torch 库
import torch
# 导入 torch.cuda 模块，用于 CUDA 相关操作

# 从 torch.autograd 模块导入多个符号
from torch.autograd import (
    _disable_profiler_legacy,     # 导入用于禁用旧有分析器的函数
    _enable_profiler_legacy,      # 导入用于启用旧有分析器的函数
    DeviceType,                   # 设备类型
    ProfilerConfig,               # 分析器配置
    ProfilerState,                # 分析器状态
)

# 从 torch.autograd.profiler_util 模块导入多个符号
from torch.autograd.profiler_util import (
    _filter_name,                  # 函数名过滤器
    _filter_stack_entry,           # 栈条目过滤器
    _rewrite_name,                 # 重写函数名
    EventList,                     # 事件列表
    FunctionEvent,                 # 函数事件
    MEMORY_EVENT_NAME,             # 内存事件名
)

# 声明 __all__ 变量，指示可以导出的符号
__all__ = ["profile"]

# 使用 deprecated 装饰器标记 profile 类，提醒用户该类已被废弃
@deprecated(
    "`torch.autograd.profiler_legacy.profile` is deprecated and will be removed in a future release. "
    "Please use `torch.profiler` instead.",
    category=None,  # TODO: change to `FutureWarning`
)
# 定义 profile 类，用于性能分析
class profile:
    """DEPRECATED: use torch.profiler instead."""

    # 初始化方法，设置性能分析器的各种参数
    def __init__(
        self,
        enabled=True,               # 是否启用性能分析
        *,
        use_cuda=False,             # 是否使用 CUDA
        record_shapes=False,        # 是否记录张量形状
        with_flops=False,           # 是否记录 FLOPs（浮点操作数）
        profile_memory=False,       # 是否进行内存分析
        with_stack=False,           # 是否记录调用栈
        with_modules=False,         # 是否记录模块信息
    ):
        self.enabled: bool = enabled   # 记录是否启用性能分析
        if not self.enabled:
            return

        self.use_cuda = use_cuda   # 记录是否使用 CUDA
        self.function_events = None   # 函数事件列表初始化为空
        self.entered = False       # 记录上下文管理器是否已进入
        self.record_shapes = record_shapes   # 记录是否记录张量形状
        self.with_flops = with_flops   # 记录是否记录 FLOPs
        self.record_shapes |= self.with_flops   # 若记录 FLOPs，则同时记录张量形状
        self.profile_memory = profile_memory   # 记录是否进行内存分析
        self.with_stack = with_stack   # 记录是否记录调用栈
        self.with_modules = with_modules   # 记录是否记录模块信息

        # 如果使用 CUDA 但 CUDA 不可用，则发出警告并禁用 CUDA 分析
        if self.use_cuda and not torch.cuda.is_available():
            warnings.warn(
                "CUDA is not available, disabling CUDA profiling",
                stacklevel=2,
            )
            self.use_cuda = False

        # 根据是否使用 CUDA 设置性能分析器类型
        if self.use_cuda:
            self.profiler_kind = ProfilerState.CUDA
        else:
            self.profiler_kind = ProfilerState.CPU

    # 返回性能分析器的配置信息
    def config(self):
        return ProfilerConfig(
            self.profiler_kind,         # 分析器类型（CUDA 或 CPU）
            self.record_shapes,         # 是否记录张量形状
            self.profile_memory,        # 是否进行内存分析
            self.with_stack,            # 是否记录调用栈
            self.with_flops,            # 是否记录 FLOPs
            self.with_modules,          # 是否记录模块信息
            # 避免在旧有公共 API 中暴露 _ExperimentalConfig
            torch._C._profiler._ExperimentalConfig(),
        )

    # 进入上下文管理器时调用的方法，开始性能分析
    def __enter__(self):
        if not self.enabled:
            return
        # 若上下文管理器已进入，则抛出运行时错误
        if self.entered:
            raise RuntimeError("Profiler context manager is not reentrant")
        self.entered = True   # 标记上下文管理器已进入
        self._start_trace()   # 开始跟踪性能分析
        return self

    # 开始跟踪性能分析的私有方法
    def _start_trace(self):
        _enable_profiler_legacy(self.config())   # 调用旧有分析器的启用方法
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 如果未启用性能分析，直接返回
        if not self.enabled:
            return
        # 如果使用 CUDA，同步 CUDA 运行时
        if self.use_cuda:
            torch.cuda.synchronize()

        # 禁用旧版性能分析器，并解析记录
        records = _disable_profiler_legacy()
        parsed_results = _parse_legacy_records(records)
        
        # 创建事件列表对象，基于解析的结果
        self.function_events = EventList(
            parsed_results,
            use_device="cuda" if self.use_cuda else None,
            profile_memory=self.profile_memory,
            with_flops=self.with_flops,
        )
        
        # 构建事件树结构
        self.function_events._build_tree()
        
        # 返回 False，表示退出方法成功
        return False

    def __repr__(self):
        # 如果未完成性能分析，返回未完成的提示
        if self.function_events is None:
            return "<unfinished profiler_legacy.profile>"
        # 否则返回事件列表对象的字符串表示形式
        return repr(self.function_events)

    def __str__(self):
        # 如果未完成性能分析，返回未完成的提示
        if self.function_events is None:
            return "<unfinished profile.profiler_legacy.profile>"
        # 否则返回事件列表对象的字符串表示形式
        return str(self.function_events)

    def _check_finish(self):
        # 检查是否完成了性能分析，若未完成则抛出异常
        if self.function_events is None:
            raise RuntimeError("Profiler didn't finish running")

    def table(
        self,
        sort_by=None,
        row_limit=100,
        max_src_column_width=75,
        max_name_column_width=55,
        max_shapes_column_width=80,
        header=None,
        top_level_events_only=False,
    ):
        # 检查是否完成了性能分析
        self._check_finish()
        assert self.function_events is not None
        
        # 调用事件列表对象的 table 方法，返回表格结果
        return self.function_events.table(
            sort_by=sort_by,
            row_limit=row_limit,
            max_src_column_width=max_src_column_width,
            max_name_column_width=max_name_column_width,
            max_shapes_column_width=max_shapes_column_width,
            header=header,
            top_level_events_only=top_level_events_only,
        )

    table.__doc__ = EventList.table.__doc__

    def export_chrome_trace(self, path):
        # 检查是否完成了性能分析
        self._check_finish()
        assert self.function_events is not None
        
        # 调用事件列表对象的 export_chrome_trace 方法，导出 Chrome Trace 文件
        return self.function_events.export_chrome_trace(path)

    export_chrome_trace.__doc__ = EventList.export_chrome_trace.__doc__

    def export_stacks(self, path: str, metric: str = "self_cpu_time_total"):
        # 检查是否完成了性能分析
        self._check_finish()
        assert self.function_events is not None, "Expected profiling results"
        assert self.with_stack, "export_stacks() requires with_stack=True"
        
        # 调用事件列表对象的 export_stacks 方法，导出堆栈信息
        return self.function_events.export_stacks(path, metric)

    def key_averages(self, group_by_input_shape=False, group_by_stack_n=0):
        # 检查是否完成了性能分析
        self._check_finish()
        assert self.function_events is not None, "Expected profiling results"
        
        # 调用事件列表对象的 key_averages 方法，返回关键平均值
        return self.function_events.key_averages(group_by_input_shape, group_by_stack_n)

    key_averages.__doc__ = EventList.key_averages.__doc__

    def total_average(self):
        # 检查是否完成了性能分析
        self._check_finish()
        assert self.function_events is not None, "Expected profiling results"
        
        # 调用事件列表对象的 total_average 方法，返回总平均值
        return self.function_events.total_average()

    total_average.__doc__ = EventList.total_average.__doc__

    @property
    # 返回所有事件中自身时间的总和作为 CPU 时间
    def self_cpu_time_total(self):
        """Return CPU time as the sum of self times across all events."""
        # 确保分析已完成
        self._check_finish()
        # 断言函数事件不为 None，即确保存在函数事件数据
        assert self.function_events is not None
        # 返回所有事件中自身时间的总和
        return self.function_events.self_cpu_time_total
def _parse_legacy_records(thread_records):
    def _get_record_key(record):
        """Return a tuple for correlating start and end records in `_parse_legacy_records`."""
        return (record.handle(), record.node_id())

    next_id = 0  # 初始化一个变量，用于记录下一个 ID
    start_record = None  # 初始化一个变量，用于存储起始记录
    functions = []  # 初始化一个空列表，用于存储函数记录
    record_stack = []  # 初始化一个空列表，用于记录堆栈信息

    # '__start_profile' 不一定是第一个记录，因此在这里查找它
    for record in itertools.chain.from_iterable(thread_records):
        name = record.name()
        if start_record is None and name == "__start_profile":
            start_record = record  # 找到 '__start_profile' 的记录并存储

    assert start_record is not None and not start_record.is_remote()

    # 按照开始时间和结束时间对函数记录进行排序，确保在有相同开始时间的嵌套事件中，
    # 外层嵌套调用优先显示，这增加了函数事件显示的稳定性
    functions.sort(key=lambda evt: [evt.time_range.start, -evt.time_range.end])

    return functions  # 返回排序后的函数记录列表
```