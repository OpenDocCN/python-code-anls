# `.\pytorch\torch\profiler\profiler.py`

```py
# mypy: allow-untyped-defs
# 导入gzip、json、os、shutil、tempfile等标准库
import gzip
import json
import os
import shutil
import tempfile
# 导入ABC和abstractmethod用于定义抽象基类
from abc import ABC, abstractmethod
# 导入Enum用于定义枚举类型
from enum import Enum
# 导入partial用于创建偏函数
from functools import partial
# 导入类型提示相关模块
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
# 导入Self来自typing_extensions，用于类型提示中指代当前类
from typing_extensions import Self
# 导入warn函数用于发出警告
from warnings import warn

# 导入torch和相关模块
import torch
# 导入torch.autograd.profiler中的prof模块
import torch.autograd.profiler as prof
# 从torch._C中导入私有use1_backend_name函数
from torch._C import _get_privateuse1_backend_name
# 从torch._C._profiler中导入执行追踪相关的函数和类
from torch._C._profiler import (
    _add_execution_trace_observer,
    _disable_execution_trace_observer,
    _enable_execution_trace_observer,
    _ExperimentalConfig,
    _remove_execution_trace_observer,
)
# 导入autograd模块中的kineto_available和ProfilerActivity
from torch.autograd import kineto_available, ProfilerActivity
# 导入torch.profiler._memory_profiler中的MemoryProfile和MemoryProfileTimeline类
from torch.profiler._memory_profiler import MemoryProfile, MemoryProfileTimeline

# 定义__all__列表，指定模块导入时的公开接口
__all__ = [
    "supported_activities",
    "ProfilerAction",
    "schedule",
    "tensorboard_trace_handler",
    "profile",
    "ExecutionTraceObserver",
]
# 定义常量PROFILER_STEP_NAME，值为字符串"ProfilerStep"
PROFILER_STEP_NAME = "ProfilerStep"


def supported_activities():
    """
    Returns a set of supported profiler tracing activities.

    Note: profiler uses CUPTI library to trace on-device CUDA kernels.
    In case when CUDA is enabled but CUPTI is not available, passing
    ``ProfilerActivity.CUDA`` to profiler results in using the legacy CUDA
    profiling code (same as in the legacy ``torch.autograd.profiler``).
    This, in turn, results in including CUDA time in the profiler table output,
    but not in the JSON trace.
    """
    # 调用torch.autograd._supported_activities()函数获取支持的追踪活动集合
    return torch.autograd._supported_activities()


class _ITraceObserver(ABC):
    """Abstract interface for a Trace observer.
    This satisfies 3 methods: start, stop and cleanup"""

    @abstractmethod
    def start(self):
        # 抽象方法，用于开始追踪
        pass

    @abstractmethod
    def stop(self):
        # 抽象方法，用于停止追踪
        pass

    @abstractmethod
    def cleanup(self):
        # 抽象方法，用于清理追踪资源
        pass


class _KinetoProfile:
    """Low-level profiler wrap the autograd profile
    """
    # 定义一个初始化方法，用于配置和启动分析器
    def __init__(
        self,
        *,
        activities: Optional[Iterable[ProfilerActivity]] = None,  # 指定要分析的活动类型，如CPU、CUDA等
        record_shapes: bool = False,  # 是否记录操作的输入形状信息
        profile_memory: bool = False,  # 是否跟踪张量的内存分配/释放情况
        with_stack: bool = False,  # 是否记录操作的源信息（文件和行号）
        with_flops: bool = False,  # 是否使用公式估算特定操作的浮点运算数（如矩阵乘法和2D卷积）
        with_modules: bool = False,  # 是否记录操作的模块层次结构（包括函数名），对于TorchScript模型有效
        experimental_config: Optional[_ExperimentalConfig] = None,  # 用于Kineto等分析库的实验性配置选项
        execution_trace_observer: Optional[_ITraceObserver] = None,  # PyTorch执行跟踪观察器对象
    ):
        self.activities = set(activities) if activities else supported_activities()
        # 初始化记录形状、浮点运算、内存分析、调用栈、模块信息等属性
        self.record_shapes = record_shapes
        self.with_flops = with_flops
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_modules = with_modules
        self.experimental_config = experimental_config
        self.execution_trace_observer = execution_trace_observer
        self.profiler: Optional[prof.profile] = None
        self.mem_tl: Optional[MemoryProfileTimeline] = None
        self.use_device = None
        # 根据活动类型设置使用的设备类型
        if ProfilerActivity.CUDA in self.activities:
            self.use_device = "cuda"
        elif ProfilerActivity.XPU in self.activities:
            self.use_device = "xpu"
        elif ProfilerActivity.MTIA in self.activities:
            self.use_device = "mtia"
        elif ProfilerActivity.PrivateUse1 in self.activities:
            self.use_device = _get_privateuse1_backend_name()

        # 用户定义的元数据，将附加到追踪中
        self.preset_metadata: Dict[str, str] = dict()

    def start(self):
        # 准备追踪工作
        self.prepare_trace()
        # 开始追踪

    def stop(self):
        # 停止追踪
        self.stop_trace()

    def prepare_trace(self):
        # 如果尚未创建分析器对象，则根据配置初始化
        if self.profiler is None:
            self.profiler = prof.profile(
                use_cpu=(ProfilerActivity.CPU in self.activities),
                use_device=self.use_device,
                record_shapes=self.record_shapes,
                with_flops=self.with_flops,
                profile_memory=self.profile_memory,
                with_stack=self.with_stack,
                with_modules=self.with_modules,
                use_kineto=True,
                experimental_config=self.experimental_config,
            )
        # 准备追踪数据
        self.profiler._prepare_trace()
    # 启动执行跟踪，如果已设置执行跟踪观察器，则启动它
    def start_trace(self):
        if self.execution_trace_observer:
            self.execution_trace_observer.start()
        # 断言：确保分析器对象已经初始化
        assert self.profiler is not None
        # 调用分析器对象的开始跟踪方法
        self.profiler._start_trace()

        # 如果需要记录内存使用情况，则添加相应的元数据标记
        if self.profile_memory:
            self.add_metadata_json("profile_memory", "1")
        # 如果需要记录调用堆栈信息，则添加相应的元数据标记
        if self.with_stack:
            self.add_metadata_json("with_stack", "1")
        # 如果需要记录张量形状信息，则添加相应的元数据标记
        if self.record_shapes:
            self.add_metadata_json("record_shapes", "1")
        # 如果需要记录模块信息，则添加相应的元数据标记
        if self.with_modules:
            self.add_metadata_json("with_modules", "1")
        # 如果需要记录浮点操作信息，则添加相应的元数据标记
        if self.with_flops:
            self.add_metadata_json("with_flops", "1")

        # 如果支持 Kineto，则获取分布式信息并添加到元数据中
        if kineto_available():
            dist_info = self._get_distributed_info()
            if dist_info:
                self.add_metadata_json("distributedInfo", json.dumps(dist_info))

            # 如果存在 torch._inductor 模块，则进一步检查设置
            if hasattr(torch, "_inductor"):
                import torch._inductor.config as inductor_config

                # 如果 Triton 使用了 CUDA 图形，则设置环境变量以避免 CUPTI 惰性重新初始化
                if inductor_config.triton.cudagraphs:
                    os.environ["DISABLE_CUPTI_LAZY_REINIT"] = "1"
                    self.add_metadata_json("DISABLE_CUPTI_LAZY_REINIT", "1")
                    # FIXME: CUDA 图形与 CUPTI 拆解不兼容的问题
                    #   1) 在第一次 CUPTI 惰性重新初始化后拆解时崩溃（CUDA 11）
                    #   2) 在第二次非惰性 CUPTI 重新初始化后拆解时崩溃（CUDA 12）
                    # 解决方法：使用 CUDA 图形时禁用 CUPTI 拆解
                    os.environ["TEARDOWN_CUPTI"] = "0"

            # 将预设的用户元数据插入到跟踪中
            for k, v in self.preset_metadata.items():
                self.add_metadata_json(k, v)

    # 停止执行跟踪，如果已设置执行跟踪观察器，则停止它
    def stop_trace(self):
        if self.execution_trace_observer:
            self.execution_trace_observer.stop()
        # 断言：确保分析器对象已经初始化
        assert self.profiler is not None
        # 调用分析器对象的退出方法，停止跟踪
        self.profiler.__exit__(None, None, None)

    # 导出采集的跟踪数据为 Chrome JSON 格式
    def export_chrome_trace(self, path: str):
        """
        Exports the collected trace in Chrome JSON format. If kineto is enabled, only
        last cycle in schedule is exported.
        """
        # 断言：确保分析器对象已经初始化
        assert self.profiler
        # 如果路径以 .gz 结尾，则使用临时文件转换为 gzip 压缩的 JSON 文件
        if path.endswith(".gz"):
            fp = tempfile.NamedTemporaryFile("w+t", suffix=".json", delete=False)
            fp.close()
            retvalue = self.profiler.export_chrome_trace(fp.name)
            with open(fp.name) as fin:
                with gzip.open(path, "wt") as fout:
                    fout.writelines(fin)
            os.remove(fp.name)
            return retvalue
        else:
            # 否则直接导出到指定路径
            return self.profiler.export_chrome_trace(path)

    # 导出堆栈跟踪信息到文件
    def export_stacks(self, path: str, metric: str = "self_cpu_time_total"):
        """Save stack traces to a file

        Args:
            path (str): save stacks file to this location;
            metric (str): metric to use: "self_cpu_time_total" or "self_cuda_time_total"
        """
        # 断言：确保分析器对象已经初始化
        assert self.profiler
        # 调用分析器对象的导出堆栈方法，指定导出路径和使用的度量标准
        return self.profiler.export_stacks(path, metric)
    def key_averages(
        self, group_by_input_shape: bool = False, group_by_stack_n: int = 0
    ):
        """
        Calculates average profiler events, optionally grouping by input shape and stack.

        .. note::
            Ensure `record_shapes` and `with_stack` are set when creating the profiler context manager.
        """
        assert self.profiler
        # 调用 profiler 对象的 key_averages 方法，传入参数进行事件平均计算
        return self.profiler.key_averages(group_by_input_shape, group_by_stack_n)

    def events(self):
        """
        Retrieves unaggregated profiler events.

        Returns:
            list: List of unaggregated profiler events.
        """
        assert self.profiler
        # 返回 profiler 对象的 function_events 属性，即未聚合的事件列表
        return self.profiler.function_events

    def add_metadata(self, key: str, value: str):
        """
        Adds user-defined metadata with a string key and value to the trace file.

        Args:
            key (str): Metadata key.
            value (str): Metadata value.
        """
        wrapped_value = '"' + value.replace('"', '\\"') + '"'
        # 调用 torch.autograd._add_metadata_json 方法，将键值对添加到跟踪文件中
        torch.autograd._add_metadata_json(key, wrapped_value)

    def add_metadata_json(self, key: str, value: str):
        """
        Adds user-defined metadata with a string key and valid JSON value to the trace file.

        Args:
            key (str): Metadata key.
            value (str): JSON-encoded metadata value.
        """
        # 调用 torch.autograd._add_metadata_json 方法，将键值对添加到跟踪文件中
        torch.autograd._add_metadata_json(key, value)

    def preset_metadata_json(self, key: str, value: str):
        """
        Presets user-defined metadata when the profiler is not yet started,
        to be added into the trace file later.

        Args:
            key (str): Metadata key.
            value (str): JSON-encoded metadata value.
        """
        # 将键值对添加到预设的 metadata 字典中，稍后将会被添加到跟踪文件中
        self.preset_metadata[key] = value

    def _get_distributed_info(self):
        """
        Retrieves information about the distributed environment if available.

        Returns:
            dict or None: Dictionary with distributed information or None if not initialized.
        """
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return None

        backend = dist.get_backend()
        # 构建包含分布式信息的字典
        dist_info = {
            "backend": backend,
            "rank": dist.get_rank(),
            "world_size": dist.get_world_size(),
            "pg_count": dist.get_pg_count(),
            "pg_config": dist.distributed_c10d._get_all_pg_configs(),
        }
        if backend == "nccl":
            nccl_version = torch.cuda.nccl.version()
            dist_info["nccl_version"] = ".".join(str(v) for v in nccl_version)
        return dist_info

    def _memory_profile(self) -> MemoryProfile:
        """
        Performs memory profiling and returns MemoryProfile object.

        Returns:
            MemoryProfile: Object containing memory profiling results.
        
        Raises:
            ValueError: If required profiling options are not set.
        """
        required = ("record_shapes", "profile_memory", "with_stack")
        missing = [f"{i}=True" for i in required if not getattr(self, i)]
        if missing:
            raise ValueError(f"{', '.join(missing)} required for memory profiling.")

        assert self.profiler is not None and self.profiler.kineto_results is not None
        # 返回内存分析的结果，使用 profiler 对象的 kineto_results 属性
        return MemoryProfile(self.profiler.kineto_results)
    def export_memory_timeline(self, path: str, device: Optional[str] = None) -> None:
        """
        Export memory event information from the profiler collected
        tree for a given device, and export a timeline plot. There are 3
        exportable files using ``export_memory_timeline``, each controlled by the
        ``path``'s suffix.

        - For an HTML compatible plot, use the suffix ``.html``, and a memory timeline
          plot will be embedded as a PNG file in the HTML file.

        - For plot points consisting of ``[times, [sizes by category]]``, where
          ``times`` are timestamps and ``sizes`` are memory usage for each category.
          The memory timeline plot will be saved a JSON (``.json``) or gzipped JSON
          (``.json.gz``) depending on the suffix.

        - For raw memory points, use the suffix ``.raw.json.gz``. Each raw memory
          event will consist of ``(timestamp, action, numbytes, category)``, where
          ``action`` is one of ``[PREEXISTING, CREATE, INCREMENT_VERSION, DESTROY]``,
          and ``category`` is one of the enums from
          ``torch.profiler._memory_profiler.Category``.

        Output: Memory timeline written as gzipped JSON, JSON, or HTML.
        """
        # Default to device 0, if unset. Fallback on cpu.
        if device is None and self.use_device and self.use_device != "cuda":
            device = self.use_device + ":0"

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Construct the memory timeline plot data
        self.mem_tl = MemoryProfileTimeline(self._memory_profile())

        # Depending on the file suffix, save the data as json.gz or json.
        # For html, we can embed the image into an HTML file.
        if path.endswith(".html"):
            # Export memory timeline as HTML with embedded PNG plot
            self.mem_tl.export_memory_timeline_html(path, device)
        elif path.endswith(".gz"):
            # Create a temporary file to store JSON data
            fp = tempfile.NamedTemporaryFile("w+t", suffix=".json", delete=False)
            fp.close()
            if path.endswith("raw.json.gz"):
                # Export raw memory timeline to a gzipped JSON file
                self.mem_tl.export_memory_timeline_raw(fp.name, device)
            else:
                # Export memory timeline to a gzipped JSON file
                self.mem_tl.export_memory_timeline(fp.name, device)
            # Compress the temporary JSON file into the specified path
            with open(fp.name) as fin:
                with gzip.open(path, "wt") as fout:
                    fout.writelines(fin)
            # Remove the temporary JSON file
            os.remove(fp.name)
        else:
            # Export memory timeline to a JSON file
            self.mem_tl.export_memory_timeline(path, device)
# 定义一个枚举，表示分析器可以在指定时间间隔内执行的操作
class ProfilerAction(Enum):
    """
    Profiler actions that can be taken at the specified intervals
    """

    NONE = 0  # 无操作
    WARMUP = 1  # 预热阶段
    RECORD = 2  # 记录数据
    RECORD_AND_SAVE = 3  # 记录并保存数据


def schedule(
    *, wait: int, warmup: int, active: int, repeat: int = 0, skip_first: int = 0
) -> Callable:
    """
    Returns a callable that can be used as profiler ``schedule`` argument. The profiler will skip
    the first ``skip_first`` steps, then wait for ``wait`` steps, then do the warmup for the next ``warmup`` steps,
    then do the active recording for the next ``active`` steps and then repeat the cycle starting with ``wait`` steps.
    The optional number of cycles is specified with the ``repeat`` parameter, the zero value means that
    the cycles will continue until the profiling is finished.
    """

    def schedule_fn(step: int) -> ProfilerAction:
        assert step >= 0  # 确保步数不小于0
        if step < skip_first:
            return ProfilerAction.NONE  # 如果步数小于跳过的步数，返回无操作
        else:
            step -= skip_first  # 跳过指定步数后重新计算当前步数
        num_steps = wait + warmup + active  # 总步数为等待 + 预热 + 活动步数
        if repeat > 0 and step / num_steps >= repeat:
            return ProfilerAction.NONE  # 如果指定了重复次数且超过，则返回无操作
        mod_step = step % num_steps  # 计算当前步数模总步数后的余数
        if mod_step < wait:
            return ProfilerAction.NONE  # 在等待阶段内，返回无操作
        elif mod_step < wait + warmup:
            return ProfilerAction.WARMUP  # 在预热阶段内，返回预热操作
        else:
            return (
                ProfilerAction.RECORD  # 在活动记录阶段内，返回记录操作
                if mod_step < num_steps - 1
                else ProfilerAction.RECORD_AND_SAVE  # 在最后一步时，返回记录并保存操作
            )

    assert (
        wait >= 0 and warmup >= 0 and active > 0 and repeat >= 0 and skip_first >= 0
    ), "Invalid profiler schedule arguments"  # 确保调度参数的有效性
    if warmup == 0:
        warn("Profiler won't be using warmup, this can skew profiler results")  # 如果预热时间为0，发出警告
    return schedule_fn  # 返回调度函数


def _default_schedule_fn(_: int) -> ProfilerAction:
    """
    Default profiler behavior - immediately starts recording the events,
    keeps doing it on every profiler step.
    """
    return ProfilerAction.RECORD  # 默认的分析器行为是立即记录事件


def tensorboard_trace_handler(
    dir_name: str, worker_name: Optional[str] = None, use_gzip: bool = False
):
    """
    Outputs tracing files to directory of ``dir_name``, then that directory can be
    directly delivered to tensorboard as logdir.
    ``worker_name`` should be unique for each worker in distributed scenario,
    it will be set to '[hostname]_[pid]' by default.
    """
    import os
    import socket
    import time

    # 输出追踪文件到指定目录，用于传递给tensorboard作为日志目录
    # 如果在分布式场景中，每个worker的worker_name应该是唯一的，默认设置为'[hostname]_[pid]'
    # 定义一个名为 handler_fn 的函数，参数为 prof，返回 None
    def handler_fn(prof) -> None:
        # 声明 worker_name 为非局部变量
        nonlocal worker_name
        # 如果 dir_name 不是一个目录
        if not os.path.isdir(dir_name):
            try:
                # 尝试创建目录 dir_name，如果已存在则忽略
                os.makedirs(dir_name, exist_ok=True)
            except Exception as e:
                # 如果创建目录失败，抛出运行时异常，带有目录名信息
                raise RuntimeError("Can't create directory: " + dir_name) from e
        # 如果 worker_name 为空
        if not worker_name:
            # 使用当前主机名和进程 ID 组成 worker_name
            worker_name = f"{socket.gethostname()}_{os.getpid()}"
        # 使用纳秒级时间戳避免导出追踪时的命名冲突
        file_name = f"{worker_name}.{time.time_ns()}.pt.trace.json"
        # 如果 use_gzip 为真，则在文件名末尾添加 '.gz' 后缀
        if use_gzip:
            file_name = file_name + ".gz"
        # 调用 prof 对象的 export_chrome_trace 方法，将追踪数据导出到指定路径
        prof.export_chrome_trace(os.path.join(dir_name, file_name))

    # 返回定义好的 handler_fn 函数作为结果
    return handler_fn
# 定义一个名为profile的类，继承自_KinetoProfile
class profile(_KinetoProfile):
    """Profiler context manager.

    Args:
        activities (iterable): list of activity groups (CPU, CUDA) to use in profiling, supported values:
            ``torch.profiler.ProfilerActivity.CPU``, ``torch.profiler.ProfilerActivity.CUDA``,
            ``torch.profiler.ProfilerActivity.XPU``.
            Default value: ProfilerActivity.CPU and (when available) ProfilerActivity.CUDA
            or (when available) ProfilerActivity.XPU.
        schedule (Callable): callable that takes step (int) as a single parameter and returns
            ``ProfilerAction`` value that specifies the profiler action to perform at each step.
        on_trace_ready (Callable): callable that is called at each step when ``schedule``
            returns ``ProfilerAction.RECORD_AND_SAVE`` during the profiling.
        record_shapes (bool): save information about operator's input shapes.
        profile_memory (bool): track tensor memory allocation/deallocation.
        with_stack (bool): record source information (file and line number) for the ops.
        with_flops (bool): use formula to estimate the FLOPs (floating point operations) of specific operators
            (matrix multiplication and 2D convolution).
        with_modules (bool): record module hierarchy (including function names)
            corresponding to the callstack of the op. e.g. If module A's forward call's
            module B's forward which contains an aten::add op,
            then aten::add's module hierarchy is A.B
            Note that this support exist, at the moment, only for TorchScript models
            and not eager mode models.
        experimental_config (_ExperimentalConfig) : A set of experimental options
            used for Kineto library features. Note, backward compatibility is not guaranteed.
        execution_trace_observer (ExecutionTraceObserver) : A PyTorch Execution Trace Observer object.
            `PyTorch Execution Traces <https://arxiv.org/pdf/2305.14516.pdf>`__ offer a graph based
            representation of AI/ML workloads and enable replay benchmarks, simulators, and emulators.
            When this argument is included the observer start() and stop() will be called for the
            same time window as PyTorch profiler. See the examples section below for a code sample.
        use_cuda (bool):
            .. deprecated:: 1.8.1
                use ``activities`` instead.

    .. note::
        Use :func:`~torch.profiler.schedule` to generate the callable schedule.
        Non-default schedules are useful when profiling long training jobs
        and allow the user to obtain multiple traces at the different iterations
        of the training process.
        The default schedule simply records all the events continuously for the
        duration of the context manager.
    """
    .. note::
        使用 :func:`~torch.profiler.tensorboard_trace_handler` 函数生成 TensorBoard 的结果文件：

        ``on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name)``
        当分析完成后，结果文件将会保存在指定的目录中。使用以下命令来查看 TensorBoard 中的结果：

        ``tensorboard --logdir dir_name``
        更多信息请参考 `PyTorch Profiler TensorBoard 插件 <https://github.com/pytorch/kineto/tree/master/tb_plugin>`__

    .. note::
        开启形状和堆栈跟踪会增加额外的开销。
        当设置 record_shapes=True 时，分析器将暂时持有张量的引用；
        这可能会阻止依赖于引用计数的某些优化，并引入额外的张量拷贝。

    Examples:

    .. code-block:: python

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as p:
            code_to_profile()
        print(p.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=-1))

    使用分析器的 ``schedule``, ``on_trace_ready`` 和 ``step`` 函数：

    .. code-block:: python

        # 非默认的分析器调度允许用户在训练循环的不同迭代中启用和禁用分析器；
        # 每当新的跟踪可用时都会调用 trace_handler
        def trace_handler(prof):
            print(prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=-1))
            # prof.export_chrome_trace("/tmp/test_trace_" + str(prof.step_num) + ".json")

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],

            # 在这个例子中，wait=1, warmup=1, active=2, repeat=1，
            # 分析器将跳过第一个步骤/迭代，
            # 在第二个开始预热，在第三和第四次迭代中记录，
            # 跟踪可用后调用 on_trace_ready；
            # 下一个步骤循环开始

            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=2,
                repeat=1),
            on_trace_ready=trace_handler
            # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
            # 在输出到 TensorBoard 时使用
            ) as p:
                for iter in range(N):
                    code_iteration_to_profile(iter)
                    # 发送信号给分析器，表明下一个迭代已开始
                    p.step()
    """
    This class defines a profiler for TorchScript programs, supporting various profiling features.

    Initialize the profiler with optional parameters and an execution trace observer.

    Parameters:
    - activities: Optional iterable of ProfilerActivity enums.
    - schedule: Optional callable to determine ProfilerAction based on step number.
    - on_trace_ready: Optional callback function invoked when trace data is ready.
    - record_shapes: Flag indicating whether to record tensor shapes.
    - profile_memory: Flag indicating whether to profile memory usage.
    - with_stack: Flag indicating whether to include stack information in profiling.
    - with_flops: Flag indicating whether to compute FLOPs during profiling.
    - with_modules: Flag indicating whether to profile individual modules.
    - experimental_config: Optional experimental configuration object.
    - execution_trace_observer: Optional observer implementing _ITraceObserver interface.
    - use_cuda: Deprecated flag for CUDA usage.

    Methods:
    - __enter__: Starts the profiler when used in a context manager.
    - __exit__: Stops the profiler and cleans up resources.
    - start: Initiates profiling actions including step recording.
    - stop: Ends current profiling actions.
    - step: Signals the start of the next profiling step.
    - _trace_ready: Invoked internally when the profiling trace data is ready.
    """
    def __init__(
        self,
        *,
        activities: Optional[Iterable[ProfilerActivity]] = None,
        schedule: Optional[Callable[[int], ProfilerAction]] = None,
        on_trace_ready: Optional[Callable[..., Any]] = None,
        record_shapes: bool = False,
        profile_memory: bool = False,
        with_stack: bool = False,
        with_flops: bool = False,
        with_modules: bool = False,
        experimental_config: Optional[_ExperimentalConfig] = None,
        execution_trace_observer: Optional[_ITraceObserver] = None,
        use_cuda: Optional[bool] = None,
    ):
        """
        Initialize the profiler with specified parameters.

        Note:
        - `activities`: Specifies which profiling activities to record.
        - `schedule`: Determines the action to perform at each profiling step.
        - `on_trace_ready`: Callback function to handle trace data when ready.
        - Flags (`record_shapes`, `profile_memory`, etc.) control additional profiling details.
        - `execution_trace_observer`: Observer object to capture execution traces.
        - `use_cuda`: Deprecated CUDA flag.
        """

    def __enter__(self):
        """
        Context manager entry point to start the profiler.
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit point to stop the profiler and clean up resources.

        Args:
        - exc_type: Exception type raised, if any.
        - exc_val: Exception value.
        - exc_tb: Exception traceback.
        """
        self.stop()
        prof.KinetoStepTracker.erase_step_count(PROFILER_STEP_NAME)
        if self.execution_trace_observer:
            self.execution_trace_observer.cleanup()

    def start(self):
        """
        Starts profiling activities including step recording if enabled.
        """
        self._transit_action(ProfilerAction.NONE, self.current_action)
        if self.record_steps:
            self.step_rec_fn = prof.record_function(
                "ProfilerStep#" + str(self.step_num)
            )
            self.step_rec_fn.__enter__()

    def stop(self):
        """
        Stops the current profiling activities.
        """
        if self.record_steps and self.step_rec_fn:
            self.step_rec_fn.__exit__(None, None, None)
        self._transit_action(self.current_action, None)

    def step(self):
        """
        Signals the start of the next profiling step.
        """
        if self.record_steps and self.step_rec_fn:
            self.step_rec_fn.__exit__(None, None, None)
        prev_action = self.current_action
        self.step_num += 1
        self.current_action = self.schedule(self.step_num)

        self._transit_action(prev_action, self.current_action)
        prof.KinetoStepTracker.increment_step(PROFILER_STEP_NAME)

        if self.record_steps:
            self.step_rec_fn = prof.record_function(
                "ProfilerStep#" + str(self.step_num)
            )
            self.step_rec_fn.__enter__()

    def _trace_ready(self):
        """
        Invoked internally when profiling trace data is ready.
        """
        if self.on_trace_ready:
            self.on_trace_ready(self)
    # 定义一个私有方法 `_transit_action`，用于执行状态转换时的动作
    def _transit_action(self, prev_action, current_action):
        # 从 `action_map` 字典中获取与给定前后动作对应的动作列表
        action_list = self.action_map.get((prev_action, current_action))
        # 如果动作列表存在
        if action_list:
            # 遍历动作列表，逐个执行动作函数
            for action in action_list:
                action()

    # 定义一个私有方法 `_stats`，返回一个可选的 `_ProfilerStats` 对象
    def _stats(self) -> Optional[prof._ProfilerStats]:
        # 如果 `profiler` 属性为 `None`，返回 `None`
        if self.profiler is None:
            return None
        # 否则，返回 `profiler` 对象的 `_stats` 属性
        return self.profiler._stats
# ExecutionTraceObserver 类，用于跟踪执行过程

class ExecutionTraceObserver(_ITraceObserver):
    """
    Execution Trace Observer

    每个进程可以拥有一个 ExecutionTraceObserver 实例。该观察器可以通过显式调用
    register_callback() 来记录函数回调。如果不调用 unregister_callback()，
    反复调用 register_callback() 不会添加额外的观察器来记录函数回调。一旦创建了
    ExecutionTraceObserver，start() 和 stop() 方法控制事件数据的记录。

    删除或调用 unregister_callback() 将从记录函数回调中移除观察器，完成输出文件，
    并停止产生任何开销。
    """

    def __init__(self):
        """
        初始化默认状态。
        """
        self._registered = False  # 是否已注册观察器的标志
        self._execution_trace_running = False  # 执行跟踪是否正在运行的标志

    def __del__(self):
        """
        在对象销毁时调用 unregister_callback() 确保输出的最终化。
        """
        self.unregister_callback()

    def register_callback(self, output_file_path: str) -> Self:
        """
        添加 ET 观察器以记录函数回调。数据将写入 output_file_path。
        """
        if not self._registered:
            self._output_file_path = output_file_path
            self._registered = _add_execution_trace_observer(output_file_path)
        return self

    def unregister_callback(self):
        """
        从记录函数回调中移除 ET 观察器。
        """

        def _save_triton_kernels():
            # 保存生成的内核的路径
            from torch._inductor.codecache import PyCodeCache as PyCodeCache

            kernel_files = [
                v.__file__
                for v in PyCodeCache.cache.values()
                if getattr(v, "__file__", None) is not None
            ]
            work_dir, file_name = os.path.split(self._output_file_path)
            resource_dir = os.path.join(
                work_dir, os.path.splitext(file_name)[0] + "_resources"
            )
            if not os.path.exists(resource_dir):
                os.mkdir(resource_dir)

            for kernel_file in kernel_files:
                if kernel_file is None:
                    continue
                path, name = os.path.split(kernel_file)
                dst = os.path.join(resource_dir, name)
                shutil.copyfile(kernel_file, dst)

        if self._registered:
            self.stop()  # 停止执行跟踪
            try:
                _save_triton_kernels()  # 尝试保存 Triton 内核
            except Exception as e:
                warn(f"Execution trace failed to save kernels: {e}")  # 保存 Triton 内核失败时的警告
            _remove_execution_trace_observer()  # 移除执行跟踪观察器
            self._registered = False  # 更新注册状态为未注册

    @property
    def is_registered(self):
        """
        如果执行跟踪观察器已注册，则返回 True，否则返回 False。
        """
        return self._registered
    # 返回观察器是否正在运行，如果是返回 True，否则返回 False
    def is_running(self):
        return self._execution_trace_running

    # 开始捕获执行跟踪
    def start(self):
        # 如果已注册并且执行跟踪未运行，则启用执行跟踪观察器，并设置执行跟踪状态为 True，记录 PG 配置信息
        if self._registered and not self._execution_trace_running:
            _enable_execution_trace_observer()
            self._execution_trace_running = True
            self._record_pg_config()

    # 停止捕获执行跟踪
    def stop(self):
        # 如果执行跟踪正在运行，则禁用执行跟踪观察器，并设置执行跟踪状态为 False
        if self._execution_trace_running:
            _disable_execution_trace_observer()
            self._execution_trace_running = False

    # 清理方法，调用 unregister_callback() 来确保最终输出的完成
    def cleanup(self):
        self.unregister_callback()

    # 获取输出文件路径的方法
    def get_output_file_path(self) -> str:
        # 如果已注册，则返回输出文件路径，否则引发 RuntimeError
        if self.is_registered:
            return self._output_file_path
        else:
            raise RuntimeError(
                "A callback to the ET profiler needs to be registered "
                "first before getting the output file path"
            )

    # 记录 PG 配置信息到执行跟踪中的私有方法
    def _record_pg_config(self) -> None:
        # 将 PG 配置信息记录到跟踪中作为节点："## process_group:init ##"
        if (
            self.is_registered
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            pg_config_info = torch.distributed.distributed_c10d._world.pg_config_info
            torch.autograd._record_function_with_args_enter(
                "## process_group:init ##", json.dumps(pg_config_info)
            )
```