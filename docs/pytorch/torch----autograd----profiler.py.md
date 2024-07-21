# `.\pytorch\torch\autograd\profiler.py`

```
# mypy: allow-untyped-defs
# 引入所需模块和类
from collections import defaultdict  # 导入 defaultdict 类
from dataclasses import dataclass  # 导入 dataclass 装饰器
from time import perf_counter_ns  # 导入 perf_counter_ns 函数
from typing import Any, Dict, List, Optional  # 导入类型注解所需的类和函数
from warnings import warn  # 导入 warn 函数

import torch  # 导入 PyTorch 模块

import torch.cuda  # 导入 torch.cuda 模块
from torch._C import _get_privateuse1_backend_name  # 导入 _get_privateuse1_backend_name 函数
from torch._C._profiler import _ExperimentalConfig  # 导入 _ExperimentalConfig 类

from torch.autograd import (  # 导入 autograd 模块中的多个函数和类
    _disable_profiler,
    _enable_profiler,
    _kineto_step,
    _prepare_profiler,
    _ProfilerResult,
    _supported_activities,
    DeviceType,
    kineto_available,
    ProfilerActivity,
    ProfilerConfig,
    ProfilerState,
)
from torch.autograd.profiler_util import (  # 导入 profiler_util 模块中的多个函数和类
    _filter_name,
    _filter_stack_entry,
    _rewrite_name,
    EventList,
    FunctionEvent,
    MEMORY_EVENT_NAME,
    MemRecordsAcc,
    OUT_OF_MEMORY_EVENT_NAME,
)
from torch.futures import Future  # 导入 Future 类

__all__ = [  # 设置模块中可以导出的公共接口列表
    "profile",
    "record_function",
    "emit_itt",
    "emit_nvtx",
    "load_nvprof",
    "EnforceUnique",
    "parse_nvprof_trace",
    "KinetoStepTracker",
    "EventList",
    "FunctionEvent",
    "MemRecordsAcc",
]

try:
    # 尝试导入 Python >= 3.2 中的 contextlib 模块中的 ContextDecorator 类作为 _ContextDecorator
    from contextlib import ContextDecorator as _ContextDecorator
except ImportError:
    import functools

    # 如果导入失败，则定义 _ContextDecorator 类以兼容旧版本 Python
    class _ContextDecorator:
        # 实现 __enter__ 方法，用于进入上下文
        def __enter__(self):
            raise NotImplementedError

        # 实现 __exit__ 方法，用于退出上下文
        def __exit__(self, exc_type, exc_val, exc_tb):
            raise NotImplementedError

        # 实现 __call__ 方法，将函数装饰为上下文管理器
        def __call__(self, func):
            @functools.wraps(func)
            def wrapped(*args, **kwargs):
                with self:
                    return func(*args, **kwargs)

            return wrapped


# 全局变量，表示当前是否启用了分析器
# 在快速 Python 检查中很有用，以减少延迟
_is_profiler_enabled: bool = False


# 函数：设置全局变量 _is_profiler_enabled 的值
def _set_is_profiler_enabled(enable: bool):
    global _is_profiler_enabled
    _is_profiler_enabled = enable


# 函数：在启动分析器时运行的回调函数，设置 _is_profiler_enabled 为 True
def _run_on_profiler_start():
    _set_is_profiler_enabled(True)


# 函数：在停止分析器时运行的回调函数，设置 _is_profiler_enabled 为 False
def _run_on_profiler_stop():
    _set_is_profiler_enabled(False)


@dataclass
class _ProfilerStats:
    "用于开发者捕捉问题/回归的分析器时间和统计信息"
    profiling_window_duration_sec: float = 0  # 分析窗口的持续时间（秒）
    number_of_events: int = 0  # 事件数量
    profiler_prepare_call_duration_us: int = 0  # 分析器准备调用的持续时间（微秒）
    profiler_enable_call_duration_us: int = 0  # 分析器启用调用的持续时间（微秒）
    profiler_disable_call_duration_us: int = 0  # 分析器禁用调用的持续时间（微秒）
    parse_kineto_call_duration_us: int = 0  # 解析 kineto 调用的持续时间（微秒）
    function_events_build_tree_call_duration_us: int = 0  # 构建函数事件树的持续时间（微秒）


# 类：profile，上下文管理器，管理自动求导分析器状态并保存结果摘要
# 在底层，它记录在 C++ 中执行的函数事件，并将这些事件暴露给 Python
# 可以将任何代码包装在其中，并且只报告 PyTorch 函数的运行时
# 注意：分析器是线程局部的，并且会自动传播到异步任务中
class profile:
    """Context manager that manages autograd profiler state and holds a summary of results.

    Under the hood it just records events of functions being executed in C++ and
    exposes those events to Python. You can wrap any code into it and it will
    only report runtime of PyTorch functions.
    Note: profiler is thread local and is automatically propagated into the async tasks
    """
    Args:
        enabled (bool, optional): 如果设置为 False，则该上下文管理器不执行任何操作。

        use_cuda (bool, optional): 启用使用 cudaEvent API 记录 CUDA 事件的时间。

        use_device (str, optional): 启用记录设备事件的时间。在使用 CUDA 时，每个张量操作大约增加 4 微秒的开销。有效的设备选项包括 'cuda'、'xpu'、'mtia' 和 'privateuseone'。

        record_shapes (bool, optional): 如果设置为 True，则收集关于输入维度的信息。这允许查看在底层使用了哪些维度，并可以通过 prof.key_averages(group_by_input_shape=True) 进一步按维度分组。注意，形状记录可能会扭曲性能分析数据。建议使用分别带有和不带有形状记录的运行来验证计时。对于嵌套函数调用的底层事件，扭曲可能会微乎其微。但对于高级函数，由于形状收集，总的自身 CPU 时间可能会人为增加。

        with_flops (bool, optional): 如果设置为 True，则分析器将使用操作符的输入形状估算 FLOPs（浮点操作）值。这允许估算硬件性能。目前，此选项仅适用于矩阵乘法和2D卷积操作符。

        profile_memory (bool, optional): 跟踪张量的内存分配和释放。

        with_stack (bool, optional): 记录操作的源信息（文件和行号）。

        with_modules (bool): 记录调用栈中的模块层次结构（包括函数名）。例如，如果模块 A 的 forward 调用了模块 B 的 forward，并包含一个 aten::add 操作，那么 aten::add 的模块层次结构为 A.B。请注意，此支持目前仅适用于 TorchScript 模型，不适用于急切模式模型。

        use_kineto (bool, optional): 实验性功能，启用使用 Kineto 分析器进行分析。

        use_cpu (bool, optional): 记录 CPU 事件；设置为 ``False`` 需要 ``use_kineto=True``，可用于降低仅 GPU 分析的开销。

        experimental_config (_ExperimentalConfig): 一组实验选项，由像 Kineto 这样的分析器库使用。请注意，不保证向后兼容性。

    .. warning:
        启用内存分析或源属性会增加额外的分析器开销。

    .. warning:
        不应递归调用此上下文管理器，即不允许嵌套实例。
    """
    定义了一个名为 Profiler 的类，用于性能分析和跟踪代码执行情况的上下文管理器。

    """
    
    def __init__(
        self,
        enabled=True,
        *,
        use_cuda=False,  # Deprecated，使用 CUDA 的标志，已弃用
        use_device=None,  # 指定使用的设备，如 CUDA 或 CPU
        record_shapes=False,  # 是否记录张量形状
        with_flops=False,  # 是否计算 FLOPs（浮点操作数）
        profile_memory=False,  # 是否分析内存使用情况
        with_stack=False,  # 是否包含调用栈信息
        with_modules=False,  # 是否包含模块信息
        use_kineto=False,  # 是否使用 Kineto 进行跟踪
        use_cpu=True,  # 是否使用 CPU 进行跟踪
        experimental_config=None,  # 实验性配置
    ):
        """
        初始化 Profiler 类的实例。
        """

    def config(self):
        """
        返回一个 ProfilerConfig 对象，用于描述当前的性能分析配置。
        """
        return ProfilerConfig(
            self.profiler_kind,
            self.record_shapes,
            self.profile_memory,
            self.with_stack,
            self.with_flops,
            self.with_modules,
            self.experimental_config,
        )

    def __enter__(self):
        """
        进入上下文管理器，准备开始性能跟踪。
        """
        if not self.enabled:
            return
        if self.entered:
            raise RuntimeError("Profiler context manager is not reentrant")
        self._prepare_trace()  # 准备跟踪器
        self._start_trace()  # 开始跟踪
        return self

    def _prepare_trace(self):
        """
        准备性能跟踪的过程。
        """
        self.entered = True
        t0 = perf_counter_ns()
        _prepare_profiler(self.config(), self.kineto_activities)  # 调用内部函数准备性能分析器
        t1 = perf_counter_ns()
        self._stats.profiler_prepare_call_duration_us = int((t1 - t0) / 1000)
    # 标记进入函数调用链追踪
    def _start_trace(self):
        # 设置标志位表示进入了函数调用链追踪
        self.entered = True
        # 执行启动性能分析器的回调函数
        _run_on_profiler_start()
        # 记录启用性能分析器的时间起点
        t0 = perf_counter_ns()
        # 启用性能分析器并记录启用耗时
        _enable_profiler(self.config(), self.kineto_activities)
        # 记录启用性能分析器的时间终点
        t1 = perf_counter_ns()
        # 计算性能分析器启用耗时并保存到统计信息中（单位为微秒）
        self._stats.profiler_enable_call_duration_us = int((t1 - t0) / 1000)
        # 记录性能分析器启动的时间戳（纳秒）
        self.profiling_start_time_ns = t1

    # 离开函数调用链追踪的上下文管理器方法
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 如果未启用性能分析器，则直接返回
        if not self.enabled:
            return
        # 如果使用设备并且该设备在torch模块中存在
        if self.use_device and hasattr(torch, self.use_device):
            # 获取设备模块对象
            device_module = getattr(torch, self.use_device)
            # 如果设备模块中存在同步方法，则执行同步操作
            if hasattr(device_module, "synchronize"):
                device_module.synchronize()

        # 如果之前有记录函数事件，则保存至old_function_events中
        old_function_events: Optional[EventList] = None
        if self.function_events:
            old_function_events = self.function_events

        # 记录禁用性能分析器的时间起点
        t0 = perf_counter_ns()

        # TODO 在此处覆盖之前的kineto结果
        # 应将之前的结果与新结果合并，否则只记录最后一次“repeat”的结果
        self.kineto_results = _disable_profiler()
        # 记录禁用性能分析器的时间终点
        t1 = perf_counter_ns()
        # 计算禁用性能分析器的耗时并保存到统计信息中（单位为微秒）
        self._stats.profiler_disable_call_duration_us = int((t1 - t0) / 1000)
        # 记录性能分析器禁用的时间戳（纳秒）
        self.profiling_end_time_ns = t0

        # 执行停止性能分析器的回调函数
        _run_on_profiler_stop()
        # 记录解析kineto结果的时间起点
        t0 = perf_counter_ns()
        # 解析kineto结果并保存解析耗时到统计信息中（单位为微秒）
        parsed_results = self._parse_kineto_results(self.kineto_results)
        # 记录解析kineto结果的时间终点
        t1 = perf_counter_ns()
        # 计算解析kineto结果的耗时并保存到统计信息中（单位为微秒）
        self._stats.parse_kineto_call_duration_us = int((t1 - t0) / 1000)

        # 根据解析后的结果创建事件列表对象function_events
        self.function_events = EventList(
            parsed_results,
            use_device=self.use_device,
            profile_memory=self.profile_memory,
            with_flops=self.with_flops,
        )
        # 记录事件列表对象构建树结构的时间起点
        t0 = perf_counter_ns()
        # 构建事件列表对象的树结构
        self.function_events._build_tree()
        # 记录事件列表对象构建树结构的时间终点
        t1 = perf_counter_ns()
        # 计算事件列表对象构建树结构的耗时并保存到统计信息中（单位为微秒）
        self._stats.function_events_build_tree_call_duration_us = int((t1 - t0) / 1000)

        # 记录事件列表对象中事件数量到统计信息中
        self._stats.number_of_events = len(self.function_events)
        # 计算性能分析窗口持续时间并保存到统计信息中（单位为秒）
        self._stats.profiling_window_duration_sec = (
            (self.profiling_end_time_ns - self.profiling_start_time_ns) * 1.0 / 1e9
        )
        # 如果存在旧的函数事件记录，则将其添加到当前的function_events中
        if old_function_events:
            for evt in old_function_events:
                self.function_events.append(evt)
        # 返回False，表示退出上下文管理器
        return False

    # 返回对象的字符串表示形式，用于显示对象信息
    def __repr__(self):
        # 如果函数事件为空，则返回未完成的信息字符串
        if self.function_events is None:
            return "<unfinished torch.autograd.profile>"
        # 否则返回函数事件的字符串表示形式
        return repr(self.function_events)

    # 返回对象的字符串形式，用于显示对象信息
    def __str__(self):
        # 如果函数事件为空，则返回未完成的信息字符串
        if self.function_events is None:
            return "<unfinished torch.autograd.profile>"
        # 否则返回函数事件的字符串形式
        return str(self.function_events)

    # 检查函数调用链追踪是否已完成
    def _check_finish(self):
        # 如果函数事件为空，则抛出运行时错误
        if self.function_events is None:
            raise RuntimeError("Profiler didn't finish running")

    # 显示事件列表的表格形式
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
        # 调用私有方法 _check_finish() 来确保收集操作完成
        self._check_finish()
        # 断言 function_events 不为 None
        assert self.function_events is not None
        # 返回 function_events 中的 table 方法结果，可选参数包括排序、行限制和列宽限制
        return self.function_events.table(
            sort_by=sort_by,
            row_limit=row_limit,
            max_src_column_width=max_src_column_width,
            max_name_column_width=max_name_column_width,
            max_shapes_column_width=max_shapes_column_width,
            header=header,
            top_level_events_only=top_level_events_only,
        )

    # 将 table 方法的文档字符串设置为 export_chrome_trace 方法的文档字符串
    table.__doc__ = EventList.table.__doc__

    def export_chrome_trace(self, path):
        """
        Exports the collected trace in Chrome JSON format. If kineto is enabled, only
        last cycle in schedule is exported.
        """
        self._check_finish()
        # 如果 kineto 可用，则保存 kineto_results 到指定路径
        if kineto_available():
            self.kineto_results.save(path)  # type: ignore[union-attr]
        else:
            # 否则调用 function_events 的 export_chrome_trace 方法保存数据到路径
            return self.function_events.export_chrome_trace(path)  # type: ignore[union-attr]

    # 将 export_chrome_trace 方法的文档字符串设置为 EventList.export_chrome_trace 方法的文档字符串
    export_chrome_trace.__doc__ = EventList.export_chrome_trace.__doc__

    def export_stacks(self, path: str, metric: str = "self_cpu_time_total"):
        self._check_finish()
        # 断言 function_events 不为 None，确保有性能分析结果
        assert self.function_events is not None, "Expected profiling results"
        # 断言 with_stack 为 True，因为 export_stacks 方法需要启用堆栈信息
        assert self.with_stack, "export_stacks() requires with_stack=True"
        # 调用 function_events 的 export_stacks 方法导出堆栈信息到指定路径，使用指定的度量标准
        return self.function_events.export_stacks(path, metric)

    def key_averages(self, group_by_input_shape=False, group_by_stack_n=0):
        self._check_finish()
        # 断言 function_events 不为 None，确保有性能分析结果
        assert self.function_events is not None, "Expected profiling results"
        # 返回 function_events 的 key_averages 方法结果，可选参数为是否按输入形状分组和堆栈深度
        return self.function_events.key_averages(group_by_input_shape, group_by_stack_n)

    # 将 key_averages 方法的文档字符串设置为 EventList.key_averages 方法的文档字符串
    key_averages.__doc__ = EventList.key_averages.__doc__

    def total_average(self):
        self._check_finish()
        # 断言 function_events 不为 None，确保有性能分析结果
        assert self.function_events is not None, "Expected profiling results"
        # 返回 function_events 的 total_average 方法结果，获取所有事件的总平均值
        return self.function_events.total_average()

    # 将 total_average 方法的文档字符串设置为 EventList.total_average 方法的文档字符串
    total_average.__doc__ = EventList.total_average.__doc__

    @property
    def self_cpu_time_total(self):
        """Returns total time spent on CPU.

        The total time is a sum of all self times across all the events.
        """
        self._check_finish()
        # 断言 function_events 不为 None，确保有性能分析结果
        assert self.function_events is not None
        # 返回 function_events 的 self_cpu_time_total 属性，表示在 CPU 上花费的总时间
        return self.function_events.self_cpu_time_total
# 定义一个继承自 _ContextDecorator 的上下文管理器/函数装饰器，在运行自动梯度分析器时给代码块或函数添加标签。
class record_function(_ContextDecorator):
    """Context manager/function decorator that adds a label to a code block/function when running autograd profiler.

    It is useful when tracing the code profile.

    Args:
        name (str): Label assigned to the block of code.
        node_id (int): ID of node, for distributed profiling. Unset in
        non-distributed cases.

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD_PROFILER)
        >>> x = torch.randn((1, 1), requires_grad=True)
        >>> with torch.autograd.profiler.profile() as prof:
        ...     y = x ** 2
        ...     with torch.autograd.profiler.record_function("label-z"): # label the block
        ...         z = y ** 3
        ...     y.backward()
        ...
        >>> # xdoctest: +IGNORE_WANT
        >>> # NOTE: some columns were removed for brevity
        >>> print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        -----------------------------------  ---------------  ---------------  ---------------
        Name                                 Self CPU total %  CPU time avg     Number of Calls
        -----------------------------------  ---------------  ---------------  ---------------
        pow                                  60.77%           47.470us         3
        mul                                  21.73%           25.465us         2
        PowBackward0                         12.03%           121.891us        1
        torch::autograd::AccumulateGrad      2.70%            6.324us          1
        label-z                              2.13%            12.421us         1
        torch::autograd::GraphRoot           0.64%            1.503us          1
        -----------------------------------  ---------------  ---------------  ---------------
        Self CPU time total: 234.344us
        CUDA time total: 0.000us

    """

    # 初始化函数，设置标签名和可选参数
    def __init__(self, name: str, args: Optional[str] = None):
        self.name: str = name
        self.args: Optional[str] = args
        # 是否在退出时运行记录函数的结束回调
        self.run_callbacks_on_exit: bool = True
        # TODO: TorchScript ignores standard type annotation here
        # 记录函数对象，初始时为 None
        self.record = torch.jit.annotate(
            Optional["torch.classes.profiler._RecordFunction"], None
        )

    # 进入上下文管理器时调用的方法
    def __enter__(self):
        # 调用 Torch 的底层操作接口，创建一个新的记录函数
        self.record = torch.ops.profiler._record_function_enter_new(
            self.name, self.args
        )
        return self
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any):
        # 如果不需要在退出时运行回调函数，则直接返回
        if not self.run_callbacks_on_exit:
            return

        # 将 record 变量设为本地变量，以便 TorchScript 将 Optional[T] 精化为 T
        record = self.record
        assert record is not None

        # TODO: 启用 __torch_function__ 处理后速度太慢
        # 参见 https://github.com/pytorch/pytorch/issues/76410
        # 如果不是在 TorchScript 模式下，则禁用 Torch 函数子类化
        if not torch.jit.is_scripting():
            with torch._C.DisableTorchFunctionSubclass():
                torch.ops.profiler._record_function_exit._RecordFunction(record)
        else:
            # 在 TorchScript 模式下调用记录函数退出
            torch.ops.profiler._record_function_exit(record)

    def _call_end_callbacks_on_future(self, fut: Future[Any]) -> Future[Any]:
        """用于对返回 Future 的异步调用进行性能分析。

        调用此函数将延伸记录范围至 future 完成。适用于分析异步调用的端到端时间。只能调用一次，否则会抛出异常。

        Args:
            fut: (torch._C.Future): 要安排回调的 future。

        Returns:
            当分析回调运行完毕时，完成的 future 对象。

        """
        # 如果已经在退出时附加了回调，则抛出运行时错误
        if not self.run_callbacks_on_exit:
            raise RuntimeError("_call_end_callbacks_on_future 只能调用一次。")

        # 禁用在退出时运行回调
        self.run_callbacks_on_exit = False

        # 将 record 变量设为本地变量，以便 TorchScript 将 Optional[T] 精化为 T
        record = self.record
        assert record is not None

        # TODO: 启用 __torch_function__ 处理后速度太慢
        # 参见 https://github.com/pytorch/pytorch/issues/76410
        if not torch.jit.is_scripting():
            with torch._C.DisableTorchFunctionSubclass():
                # 在 TorchScript 模式下调用记录函数退出的 JIT future
                profiled_future = (
                    torch.ops.profiler._call_end_callbacks_on_jit_fut._RecordFunction(
                        record, fut
                    )
                )
        else:
            # 在 TorchScript 模式下调用 JIT future 上的记录函数退出
            profiled_future = torch.ops.profiler._call_end_callbacks_on_jit_fut(
                record, fut
            )
        return profiled_future
class emit_nvtx:
    """Context manager that makes every autograd operation emit an NVTX range.

    It is useful when running the program under nvprof::

        nvprof --profile-from-start off -o trace_name.prof -- <regular command here>

    This context manager allows you to annotate CUDA kernels with NVTX ranges for profiling
    purposes. NVTX (NVIDIA Tools Extension) is a C-based API for annotating events and ranges
    within CUDA applications.

    Args:
        None

    Example:
        >>> # xdoctest: +SKIP("Undefined variables")
        >>> with torch.autograd.profiler.emit_nvtx():
        ...     model(x)

    """

    def __init__(self):
        pass

    def __enter__(self):
        # Start profiling with NVTX ranges for autograd operations
        nvtx_range_push("autograd_profiling")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Stop profiling with NVTX ranges
        nvtx_range_pop()
        return False
    Unfortunately, there's no way to force nvprof to flush the data it collected
    to disk, so for CUDA profiling one has to use this context manager to annotate
    nvprof traces and wait for the process to exit before inspecting them.
    Then, either NVIDIA Visual Profiler (nvvp) can be used to visualize the timeline, or
    :func:`torch.autograd.profiler.load_nvprof` can load the results for inspection
    e.g. in Python REPL.

    .. warning:
        This context manager should not be called recursively, i.e. at most one
        instance should be enabled at any given time.

    Args:
        enabled (bool, optional): Setting ``enabled=False`` makes this context manager a no-op.
            Default: ``True``.
        record_shapes (bool, optional): If ``record_shapes=True``, the nvtx range wrapping
            each autograd op will append information about the sizes of Tensor arguments received
            by that op, in the following format:
            ``[[arg0.size(0), arg0.size(1), ...], [arg1.size(0), arg1.size(1), ...], ...]``
            Non-tensor arguments will be represented by ``[]``.
            Arguments will be listed in the order they are received by the backend op.
            Please note that this order may not match the order in which those arguments were passed
            on the Python side.  Also note that shape recording may increase the overhead of nvtx range creation.
            Default: ``False``

    Example:
        >>> # xdoctest: +SKIP("undefined variables")
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD_PROFILER)
        >>> with torch.cuda.profiler.profile():
        ...     model(x)  # Warmup CUDA memory allocator and profiler
        ...     with torch.autograd.profiler.emit_nvtx():
        ...         model(x)

    **Forward-backward correlation**

    When viewing a profile created using :class:`emit_nvtx` in the Nvidia Visual Profiler,
    correlating each backward-pass op with the corresponding forward-pass op can be difficult.
    To ease this task, :class:`emit_nvtx` appends sequence number information to the ranges it
    generates.

    During the forward pass, each function range is decorated with ``seq=<N>``.  ``seq`` is a running
    counter, incremented each time a new backward Function object is created and stashed for backward.
    Thus, the ``seq=<N>`` annotation associated with each forward function range tells you that
    if a backward Function object is created by this forward function,
    the backward object will receive sequence number N.
    During the backward pass, the top-level range wrapping each C++ backward Function's
    ``apply()`` call is decorated with ``stashed seq=<M>``.  ``M`` is the sequence number that
    the backward object was created with.  By comparing ``stashed seq`` numbers in backward with ``seq``
    numbers in forward, you can track down which forward op created each backward Function.
    """
    The following code defines a context manager for NVTX (NVIDIA Tools Extension) annotations,
    used for profiling CUDA code. It enables and disables the profiler with specific configurations
    and ensures that the context is entered and exited properly.

    """

    # 定义一个 NVTX 注解的上下文管理器
    def __init__(self, enabled=True, record_shapes=False):
        # 初始化上下文管理器，设置是否启用，默认为启用
        self.enabled = enabled
        # 记录上下文管理器是否已进入的状态，默认为未进入
        self.entered = False
        # 是否记录张量形状的标志，默认为不记录
        self.record_shapes = record_shapes

    # 进入上下文管理器时执行的操作
    def __enter__(self):
        # 如果未启用上下文管理器，则直接返回
        if not self.enabled:
            return
        # 如果上下文已经进入过，则抛出运行时错误，因为不支持重新进入
        if self.entered:
            raise RuntimeError("NVTX annotation context manager is not reentrant")
        # 将上下文标记为已进入
        self.entered = True
        # 同步 CUDA 设备，确保操作的同步性
        torch.cuda.synchronize()
        # 运行启用性能分析器的函数
        _run_on_profiler_start()
        # 启用性能分析器，使用指定的配置参数
        _enable_profiler(
            ProfilerConfig(
                ProfilerState.NVTX,
                self.record_shapes,
                False,
                False,
                False,
                False,
                _ExperimentalConfig(),
            ),
            set(),
        )
        # 返回当前上下文管理器实例
        return self

    # 退出上下文管理器时执行的操作
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 如果未启用上下文管理器，则直接返回
        if not self.enabled:
            return
        # 同步 CUDA 设备，确保操作的同步性
        torch.cuda.synchronize()
        # 禁用性能分析器
        _disable_profiler()
        # 运行停止性能分析器的函数
        _run_on_profiler_stop()
        # 返回 False，以允许任何异常继续传播
        return False
def load_nvprof(path):
    """Open an nvprof trace file and parses autograd annotations.

    Args:
        path (str): path to nvprof trace
    """
    # 调用parse_nvprof_trace函数解析nvprof跟踪文件，并返回处理后的事件列表
    return EventList(parse_nvprof_trace(path))


class EnforceUnique:
    """Raises an error if a key is seen more than once."""

    def __init__(self):
        # 初始化一个空的集合，用于存储已经观察到的键
        self.seen = set()

    def see(self, *key):
        r"""
        Observe a key and raise an error if it is seen multiple times.
        """
        # 如果键已经在集合中存在，则抛出运行时错误
        if key in self.seen:
            raise RuntimeError("duplicate key: " + str(key))
        # 否则，将键加入到集合中
        self.seen.add(key)


def parse_nvprof_trace(path):
    import sqlite3

    # 连接到指定路径的SQLite数据库文件
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row

    # 解析字符串表
    strings = {}
    for r in conn.execute("SELECT _id_ as id, value FROM StringTable"):
        # 将字符串表中的每一项解码，并存储到strings字典中
        strings[r["id"]] = torch._C._demangle(r["value"])

    # 查询所有函数并创建对应的FunctionEvent对象
    marker_query = """
    SELECT
        start.id AS marker_id, start.name, start.timestamp AS start_time, end.timestamp AS end_time
    FROM
        CUPTI_ACTIVITY_KIND_MARKER AS start INNER JOIN CUPTI_ACTIVITY_KIND_MARKER AS end
        ON start.id = end.id
    WHERE
        start.name != 0 AND end.name = 0
    """
    functions = []
    functions_map = {}
    unique = EnforceUnique()
    for row in conn.execute(marker_query):
        # 确保每个marker_id只出现一次，否则抛出运行时错误
        unique.see(row["marker_id"])
        # 创建一个FunctionEvent对象，并将其添加到functions列表和functions_map字典中
        evt = FunctionEvent(
            id=row["marker_id"],
            node_id=0,  # 调用FunctionEvent时缺少node_id，这里使用0作为占位符
            name=strings[row["name"]],
            start_us=row["start_time"],
            end_us=row["end_time"],
            thread=0,
        )  # TODO: 在SQLite数据库中查找
        functions.append(evt)
        functions_map[evt.id] = evt

    # 查询所有内核函数并与FunctionEvents进行关联
    kernel_query = """
    SELECT
        start.id AS marker_id, start.name, start.timestamp, end.timestamp,
        runtime._id_ AS runtime_id, runtime.cbid, runtime.start AS runtime_start, runtime.end AS runtime_end,
        kernel.start AS kernel_start, kernel.end AS kernel_end, kernel.name AS kernel_name
    FROM
        CUPTI_ACTIVITY_KIND_MARKER AS start
        INNER JOIN CUPTI_ACTIVITY_KIND_MARKER AS end
            ON start.id = end.id
        INNER JOIN CUPTI_ACTIVITY_KIND_RUNTIME as runtime
            ON (start.timestamp < runtime.start AND runtime.end < end.timestamp)
        INNER JOIN CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL AS kernel
            ON kernel.correlationId = runtime.correlationId
    """
    unique = EnforceUnique()
    # 使用数据库连接执行给定的查询，遍历结果集中的每一行
    for row in conn.execute(kernel_query):
        # 将行中的 "marker_id" 和 "runtime_id" 添加到 unique 对象中
        unique.see(row["marker_id"], row["runtime_id"])
        # 断言 "cbid" 字段的值为 211，用于检查 cudaKernelLaunch 的条件（适用于 cuda >= 9.2）
        assert row["cbid"] == 211
        # 根据 "marker_id" 查找对应的事件，并将当前行的核函数信息附加到事件中
        evt = functions_map[row["marker_id"]]
        evt.append_kernel(
            row["kernel_name"], 0, row["kernel_end"] - row["kernel_start"]
        )

    # 对 functions 列表进行排序，排序依据是事件的时间范围的起始时间
    functions.sort(key=lambda evt: evt.time_range.start)
    # 返回排序后的 functions 列表作为函数的结果
    return functions
class KinetoStepTracker:
    """Provides an abstraction for incrementing the step count globally.

    Previously, we only had one place to mark that a step() has occurred
    in the program via pytorch profiler step(). We will now add step hooks
    in the Optimizer class https://github.com/pytorch/pytorch/issues/88446

    - This could mean programs that already call profiler.step() every
      iteration can end up double incrementing step count.
    - If a model uses multiple optimizers we can also have double or more
      counting of the step.

    We fix this by adding a layer of abstraction before calling step()
    to the kineto library. The idea is to maintain steps per requester in a dict:

    .. code-block::

        {
           "ProfilerStep": 100,  # triggered by profiler step() call
           "Optimizer1Step": 100,   # Optimizer 1 or 2 are just examples, could be SGD, Adam etc
           "Optimizer2Step": 100,
        }

    To figure out the global step count just take the max of dict values (100).

    If one of the count increments the max will go up.

    .. code-block::

        {
           "ProfilerStep": 100,
           "Optimizer1Step": 101,   # Optimizer1 got incremented first say
           "Optimizer2Step": 100,
        }

    Then global step count is 101
    We only call the kineto step() function when global count increments.

    NOTE: Please do not use the KinetoStepTracker in modules beside the Optimizer
    for now. The result could be incorrect increments of the step count.
    """

    _current_step = 0  # 当前的全局步数计数器
    _step_dict: Dict[str, int] = defaultdict(int)  # 存储每个请求者的步数计数，默认为0

    @classmethod
    def init_step_count(cls, requester: str):
        r"""
        Initialize for a given requester.
        """
        cls._step_dict[requester] = cls._current_step  # 初始化给定请求者的步数计数为当前全局步数

    @classmethod
    def erase_step_count(cls, requester: str) -> bool:
        r"""
        Remove a given requester.
        """
        return cls._step_dict.pop(requester, None) is not None  # 移除给定请求者的步数计数

    @classmethod
    def increment_step(cls, requester: str) -> int:
        """Increments the step count for the requester.

        Additionally if the max over all step counts has incremented then
        trigger the _kineto_step() returns global step count
        """
        if requester not in cls._step_dict:
            cls.init_step_count(requester)  # 如果请求者不在步数字典中，则初始化步数计数

        cls._step_dict[requester] += 1  # 增加给定请求者的步数计数

        new_step = max(cls._step_dict.values())  # 获取所有请求者中的最大步数计数
        if new_step > cls._current_step:
            delta = new_step - cls._current_step
            if delta > 1:
                warn(
                    "Profiler step count has increased more than 1 - "
                    f"current_step = {cls._current_step} step dict =  {cls._step_dict}"
                )
            for _ in range(0, delta):
                _kineto_step()  # 当全局步数计数增加时触发 kineto step() 函数
            cls._current_step = new_step  # 更新当前全局步数计数

        return cls._current_step  # 返回当前全局步数计数
    # 定义一个类方法 current_step，用于获取任何请求者的最新步骤
    def current_step(cls) -> int:
        """
        Get the latest step for any requester
        """
        # 返回类变量 _current_step 的当前值
        return cls._current_step
```