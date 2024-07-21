# `.\pytorch\torch\distributed\rpc\server_process_global_profiler.py`

```py
#!/usr/bin/python3
# mypy: allow-untyped-defs

# 导入 itertools 模块
import itertools
# 导入 List 类型
from typing import List

# 导入 torch 库
import torch
# 导入 torch.autograd.profiler_legacy 模块中的 profile 类
from torch.autograd.profiler_legacy import profile

# 导入当前目录下的 _disable_server_process_global_profiler 和 _enable_server_process_global_profiler
from . import (
    _disable_server_process_global_profiler,
    _enable_server_process_global_profiler,
)

# 初始化 __all__ 列表为空列表
__all__: List[str] = []

# 定义 _server_process_global_profile 类，继承自 profile 类
class _server_process_global_profile(profile):
    """
    It has the same API as ``torch.autograd.profiler.profile`` class,
    except that it enables profiling on all threads running RPC server request callbacks.

    Context manager that manages autograd profiler state and holds a summary of results.
    Under the hood it just records events of functions being executed in C++ and
    exposes those events to Python. You can wrap any code into it and it will
    only report runtime of PyTorch functions.
    Note: profiler is thread local and is automatically propagated into the async tasks

    Args:
        enabled (bool, optional): Setting this to False makes this context manager a no-op.
            Default: ``True``.

        use_cuda (bool, optional): Enables timing of CUDA events as well using the cudaEvent API.
            Adds approximately 4us of overhead to each tensor operation.
            Default: ``False``

        record_shapes (bool, optional): If shapes recording is set, information
            about input dimensions will be collected. This allows one to see which
            dimensions have been used under the hood and further group by them
            using prof.key_averages(group_by_input_shape=True). Please note that
            shape recording might skew your profiling data. It is recommended to
            use separate runs with and without shape recording to validate the timing.
            Most likely the skew will be negligible for bottom most events (in a case
            of nested function calls). But for higher level functions the total
            self cpu time might be artificially increased because of the shape
            collection.

        profile_memory (bool, optional): Whether to report memory usage, default: ``False``

    .. warning:
        Enabling memory profiling incurs additional profiler overhead

    .. warning:
        Due to some CUDA multiprocessing limitations (multiprocessing-cuda-note_),
        one cannot use the profiler with ``use_cuda = True`` to benchmark
        DataLoaders with ``num_workers > 0``. If you wish to benchmark data loading,
        please use ``use_cuda = False`` or ``num_workers = 0``.
    """
    pass  # 类定义结束，暂时没有额外的方法或属性定义
    # 定义一个新的类，继承自默认父类
    def __init__(self, *args, **kwargs):
        # 调用父类的构造函数，传入所有位置参数和关键字参数
        super().__init__(*args, **kwargs)
    def __enter__(self):
        """
        Turn on server-side process-global profiling.
        This enables thread-local profiler on all RPC threads running server-side request callbacks.
        """
        # 检查是否禁用了自动梯度分析，如果禁用则直接返回，不执行后续操作
        if not self.enabled:
            return
        
        # 检查是否已经进入过自动梯度分析，如果是，则抛出运行时错误，不允许重入
        if self.entered:  # type: ignore[has-type]
            raise RuntimeError("autograd profiler traces are not reentrant")
        
        # 将进入标志设置为 True，表示当前已经进入自动梯度分析
        self.entered = True

        # 根据 self.use_cuda 的设置确定使用 CUDA 还是 CPU 进行分析
        profiler_kind = (
            torch.autograd.ProfilerState.CUDA
            if self.use_cuda
            else torch.autograd.ProfilerState.CPU
        )

        # 创建梯度分析器的配置对象，包括分析器的种类、记录形状、内存分析等设置
        profiler_config = torch.autograd.ProfilerConfig(
            profiler_kind,
            self.record_shapes,
            self.profile_memory,
            False,
            False,
            False,
            torch.profiler._ExperimentalConfig(),
        )

        # 调用内部函数启用服务器进程全局分析器，并传入配置对象
        _enable_server_process_global_profiler(profiler_config)
        
        # 返回当前对象自身，用于支持上下文管理器的 with 语句
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        关闭服务器端的全局进程级分析。
        聚合所有由RPC线程记录的分析事件。

        这些属性在退出上下文时被赋值。

        Attributes:
            function_events (torch.autograd.profiler.EventList). 它是一个列表，具有辅助方法，
            如1）以漂亮的打印表格显示记录项。
            2）通过键分组进行平均值计算。3）等等。

            process_global_function_events (List[torch.autograd.profiler.FunctionEvent]).
            它是一个列表，包含``FunctionEvent``元素。每个元素是在分析范围内处理RPC请求的分析结果。
        """
        if not self.enabled:
            return

        # 关闭服务器端的全局进程级分析，并获取分析事件
        process_global_events = _disable_server_process_global_profiler()

        # 每个元素在此列表中是一个线程分析结果，来自处理RPC请求的线程
        process_global_function_events = []
        for thread_local_events in process_global_events:
            # 将``Event``解析为``FunctionEvent``
            thread_local_function_events = (
                torch.autograd.profiler_legacy._parse_legacy_records(
                    thread_local_events
                )
            )
            # 按开始时间和负结束时间排序
            thread_local_function_events.sort(
                key=lambda function_event: [
                    function_event.time_range.start,
                    -(function_event.time_range.end),
                ]
            )
            process_global_function_events.append(thread_local_function_events)

        # 展开的函数事件列表，用于构建事件树
        flattened_function_events = list(
            itertools.chain.from_iterable(process_global_function_events)
        )
        self.function_events = torch.autograd.profiler_util.EventList(
            flattened_function_events,
            use_cuda=self.use_cuda,
            profile_memory=self.profile_memory,
        )
        self.function_events._build_tree()

        # 设置处理全局函数事件列表
        self.process_global_function_events = process_global_function_events

        return False
```