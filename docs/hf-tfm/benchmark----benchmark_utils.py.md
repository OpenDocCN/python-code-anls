# `.\benchmark\benchmark_utils.py`

```
# 导入所需的模块和库
import copy  # 导入 copy 模块，用于对象的复制操作
import csv  # 导入 csv 模块，用于 CSV 文件的读写操作
import linecache  # 导入 linecache 模块，用于按行缓存操作
import os  # 导入 os 模块，提供了与操作系统交互的功能
import platform  # 导入 platform 模块，用于访问平台相关属性
import sys  # 导入 sys 模块，提供了对 Python 解释器的访问
import warnings  # 导入 warnings 模块，用于警告控制
from abc import ABC, abstractmethod  # 从 abc 模块导入 ABC 和 abstractmethod 用于抽象基类定义
from collections import defaultdict, namedtuple  # 导入 defaultdict 和 namedtuple 类型，用于默认值字典和命名元组
from datetime import datetime  # 导入 datetime 类，用于日期时间操作
from multiprocessing import Pipe, Process, Queue  # 导入多进程相关模块，包括 Pipe、Process 和 Queue
from multiprocessing.connection import Connection  # 导入 Connection 类，用于多进程通信
from typing import Callable, Iterable, List, NamedTuple, Optional, Union  # 导入类型提示相关功能

from .. import AutoConfig, PretrainedConfig  # 导入上层目录的 AutoConfig 和 PretrainedConfig 类
from .. import __version__ as version  # 导入版本号
from ..utils import (
    is_psutil_available, is_py3nvml_available, is_tf_available, is_torch_available, logging
)  # 从上层目录中导入一些工具函数和变量
from .benchmark_args_utils import BenchmarkArguments  # 从当前目录中导入 BenchmarkArguments 类


if is_torch_available():
    from torch.cuda import empty_cache as torch_empty_cache  # 如果 Torch 可用，导入清空 GPU 缓存函数

if is_tf_available():
    from tensorflow.python.eager import context as tf_context  # 如果 TensorFlow 可用，导入 TensorFlow context

if is_psutil_available():
    import psutil  # 如果 psutil 可用，导入 psutil 模块

if is_py3nvml_available():
    import py3nvml.py3nvml as nvml  # 如果 py3nvml 可用，导入 py3nvml 模块

if platform.system() == "Windows":
    from signal import CTRL_C_EVENT as SIGKILL  # 如果是 Windows 系统，导入 CTRL_C_EVENT 作为 SIGKILL
else:
    from signal import SIGKILL  # 如果是其他系统，导入 SIGKILL 信号

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器对象，用于日志输出

_is_memory_tracing_enabled = False  # 内存追踪开关，默认关闭

BenchmarkOutput = namedtuple(
    "BenchmarkOutput",
    [
        "time_inference_result",
        "memory_inference_result",
        "time_train_result",
        "memory_train_result",
        "inference_summary",
        "train_summary",
    ],
)  # 定义命名元组 BenchmarkOutput，用于存储基准测试的输出结果


def separate_process_wrapper_fn(func: Callable[[], None], do_multi_processing: bool) -> Callable[[], None]:
    """
    This function wraps another function into its own separated process. In order to ensure accurate memory
    measurements it is important that the function is executed in a separate process

    Args:
        - `func`: (`callable`): function() -> ... generic function which will be executed in its own separate process
        - `do_multi_processing`: (`bool`) Whether to run function on separate process or not
    """
    # 这个函数将另一个函数包装成自己的独立进程。为了确保精确的内存测量，重要的是函数在独立进程中执行。
    # `func`: 要执行的函数，必须是一个可以在独立进程中运行的通用函数
    # `do_multi_processing`: 是否在独立进程上运行函数的布尔值
    # 定义一个函数，用于在单独的进程中执行给定的函数，并确保内存的正确使用

    def multi_process_func(*args, **kwargs):
        # 定义一个内部函数，用于在单独进程中运行指定的函数，并将结果放入队列
        def wrapper_func(queue: Queue, *args):
            try:
                # 调用传入的函数，获取其结果
                result = func(*args)
            except Exception as e:
                # 捕获异常，记录错误日志并打印异常信息
                logger.error(e)
                print(e)
                # 将结果设置为 "N/A"
                result = "N/A"
            # 将结果放入队列
            queue.put(result)

        # 创建一个队列对象
        queue = Queue()
        # 创建一个进程对象，目标为内部定义的 wrapper_func 函数，传入队列和参数
        p = Process(target=wrapper_func, args=[queue] + list(args))
        # 启动进程
        p.start()
        # 从队列中获取结果
        result = queue.get()
        # 等待进程结束
        p.join()
        # 返回从进程中获取的结果
        return result

    # 如果需要多进程处理
    if do_multi_processing:
        # 记录信息，指示函数将在自己的进程中执行
        logger.info(f"Function {func} is executed in its own process...")
        # 返回多进程处理的函数 multi_process_func
        return multi_process_func
    else:
        # 如果不需要多进程处理，直接返回原始的函数 func
        return func
def is_memory_tracing_enabled():
    # 返回全局变量 `_is_memory_tracing_enabled` 的值，表示内存追踪是否启用
    global _is_memory_tracing_enabled
    return _is_memory_tracing_enabled


class Frame(NamedTuple):
    """
    `Frame` 是一个 NamedTuple，用于收集当前帧的状态。`Frame` 有以下字段:

        - 'filename' (string): 当前执行的文件名
        - 'module' (string): 当前执行的模块名
        - 'line_number' (int): 当前执行的行号
        - 'event' (string): 触发追踪的事件（默认为 "line"）
        - 'line_text' (string): Python 脚本中行的文本内容
    """

    filename: str
    module: str
    line_number: int
    event: str
    line_text: str


class UsedMemoryState(NamedTuple):
    """
    `UsedMemoryState` 是一个命名元组，具有以下字段:

        - 'frame': 一个 `Frame` 命名元组，存储当前追踪帧的信息（当前文件、当前文件中的位置）
        - 'cpu_memory': 执行该行前的 CPU RSS 内存状态
        - 'gpu_memory': 执行该行前的 GPU 使用内存（所有 GPU 的总和，或者仅限于 `gpus_to_trace` 指定的 GPU）
    """

    frame: Frame
    cpu_memory: int
    gpu_memory: int


class Memory(NamedTuple):
    """
    `Memory` 命名元组只有一个字段 `bytes`，可以通过调用 `__repr__` 方法得到以兆字节为单位的人类可读字符串。

        - `bytes` (integer): 字节数
    """

    bytes: int

    def __repr__(self) -> str:
        return str(bytes_to_mega_bytes(self.bytes))


class MemoryState(NamedTuple):
    """
    `MemoryState` 是一个命名元组，列出了带有以下字段的帧 + CPU/GPU 内存:

        - `frame` (`Frame`): 当前帧（参见上面的定义）
        - `cpu`: 当前帧期间消耗的 CPU 内存，作为 `Memory` 命名元组
        - `gpu`: 当前帧期间消耗的 GPU 内存，作为 `Memory` 命名元组
        - `cpu_gpu`: 当前帧期间消耗的 CPU + GPU 内存，作为 `Memory` 命名元组
    """

    frame: Frame
    cpu: Memory
    gpu: Memory
    cpu_gpu: Memory


class MemorySummary(NamedTuple):
    """
    `MemorySummary` 是一个命名元组，还未定义字段，将来可能会添加关于内存概述的信息。
    """
    # 定义一个名为 `MemorySummary` 的命名元组，包含以下字段：

    # - `sequential`: 从 `memory_trace` 计算而来的 `MemoryState` 命名元组列表，表示每行代码执行前后内存的差值。
    # - `cumulative`: 从 `memory_trace` 计算而来的 `MemoryState` 命名元组列表，表示每行代码的累积内存增加量，
    #   如果某行代码被多次执行，其内存增加量会被多次累加。列表按内存消耗从大到小排序（可能为负数，表示释放内存）。
    # - `current`: 当前内存状态的 `MemoryState` 命名元组列表。
    # - `total`: `Memory` 命名元组，表示完整追踪期间的内存总增加量。如果 `ignore_released_memory` 为 `True`
    #   （默认值），则忽略内存释放（消耗为负数）的行。

    sequential: List[MemoryState]
    cumulative: List[MemoryState]
    current: List[MemoryState]
    total: Memory
MemoryTrace = List[UsedMemoryState]
# 定义了一个类型别名 MemoryTrace，表示一个列表，列表元素是 UsedMemoryState 类型的对象

def measure_peak_memory_cpu(function: Callable[[], None], interval=0.5, device_idx=None) -> int:
    """
    测量给定函数 `function` 的 CPU 内存峰值消耗，运行时间至少 interval 秒，最多 20 * interval 秒。
    此函数受 `memory_profiler` 包中 `memory_usage` 函数的启发：
    https://github.com/pythonprofilers/memory_profiler/blob/895c4ac7a08020d66ae001e24067da6dcea42451/memory_profiler.py#L239

    Args:
        - `function`: (`callable`): 无参数函数，用于测量其内存消耗的函数

        - `interval`: (`float`, `optional`, 默认为 `0.5`): 测量内存使用的时间间隔（秒）

        - `device_idx`: (`int`, `optional`, 默认为 `None`): 要测量 GPU 使用情况的设备 ID

    Returns:
        - `max_memory`: (`int`) 字节单位的内存峰值消耗
    """

    def get_cpu_memory(process_id: int) -> int:
        """
        测量给定 `process_id` 的当前 CPU 内存使用量

        Args:
            - `process_id`: (`int`) 要测量内存的进程 ID

        Returns:
            - `memory`: (`int`) 字节单位的内存消耗
        """
        process = psutil.Process(process_id)
        try:
            # 获取进程内存信息，根据 psutil 版本不同选择不同的方法
            meminfo_attr = "memory_info" if hasattr(process, "memory_info") else "get_memory_info"
            memory = getattr(process, meminfo_attr)()[0]
        except psutil.AccessDenied:
            raise ValueError("Psutil 访问错误.")
        return memory

    # 检查是否安装了 psutil 库，如果没有则给出警告
    if not is_psutil_available():
        logger.warning(
            "未安装 Psutil，无法记录 CPU 内存使用情况。"
            "安装 Psutil (pip install psutil) 以使用 CPU 内存跟踪。"
        )
        max_memory = "N/A"
        else:
            # 定义一个继承自 Process 的 MemoryMeasureProcess 类，用于测量进程的内存使用情况
            class MemoryMeasureProcess(Process):

                """
                `MemoryMeasureProcess` inherits from `Process` and overwrites its `run()` method. Used to measure the
                memory usage of a process
                """

                def __init__(self, process_id: int, child_connection: Connection, interval: float):
                    super().__init__()
                    self.process_id = process_id
                    self.interval = interval
                    self.connection = child_connection
                    self.num_measurements = 1
                    self.mem_usage = get_cpu_memory(self.process_id)

                def run(self):
                    # 发送信号给父进程，表示开始测量
                    self.connection.send(0)
                    stop = False
                    while True:
                        # 更新内存使用情况
                        self.mem_usage = max(self.mem_usage, get_cpu_memory(self.process_id))
                        self.num_measurements += 1

                        if stop:
                            break

                        # 检查是否要停止测量
                        stop = self.connection.poll(self.interval)

                    # 将测量结果发送给父进程管道
                    self.connection.send(self.mem_usage)
                    self.connection.send(self.num_measurements)

            while True:
                # 创建子进程与父进程之间的管道
                child_connection, parent_connection = Pipe()

                # 实例化 MemoryMeasureProcess 进程对象
                mem_process = MemoryMeasureProcess(os.getpid(), child_connection, interval)
                mem_process.start()

                # 等待直到收到内存测量的信号
                parent_connection.recv()

                try:
                    # 执行指定的函数
                    function()

                    # 向父进程发送信号，表示执行完毕
                    parent_connection.send(0)

                    # 接收内存使用情况和测量次数
                    max_memory = parent_connection.recv()
                    num_measurements = parent_connection.recv()
                except Exception:
                    # 在一个干净的方式下终止进程
                    parent = psutil.Process(os.getpid())
                    for child in parent.children(recursive=True):
                        os.kill(child.pid, SIGKILL)
                    mem_process.join(0)
                    # 抛出运行时异常，表示进程被终止，有错误发生
                    raise RuntimeError("Process killed. Error in Process")

                # 等待进程至少运行 20 倍的时间间隔或者直到它完成
                mem_process.join(20 * interval)

                # 如果测量次数大于 4 或者间隔小于 1e-6，则跳出循环
                if (num_measurements > 4) or (interval < 1e-6):
                    break

                # 减小时间间隔
                interval /= 10

            # 返回最大内存使用情况
            return max_memory
# 定义一个函数 `start_memory_tracing`，用于设置内存跟踪，记录模块或子模块每行的 RAM 使用情况。
def start_memory_tracing(
    modules_to_trace: Optional[Union[str, Iterable[str]]] = None,
    modules_not_to_trace: Optional[Union[str, Iterable[str]]] = None,
    events_to_trace: str = "line",
    gpus_to_trace: Optional[List[int]] = None,
) -> MemoryTrace:
    """
    设置逐行跟踪，记录模块或子模块每行的 RAM 使用情况。详见 `./benchmark.py` 示例。

    Args:
        - `modules_to_trace`: (None, string, list/tuple of string) 如果为 None，则记录所有事件；如果是字符串或字符串列表：仅记录列出的模块/子模块的事件（例如 'fairseq' 或 'transformers.models.gpt2.modeling_gpt2'）
        - `modules_not_to_trace`: (None, string, list/tuple of string) 如果为 None，则不避免任何模块；如果是字符串或字符串列表：不记录列出的模块/子模块的事件（例如 'torch'）
        - `events_to_trace`: 要记录的事件字符串或事件字符串列表（参见官方 Python 文档的 `sys.settrace` 关于事件的列表），默认为 line
        - `gpus_to_trace`: (可选列表，默认为 None) 要跟踪的 GPU 列表。默认为跟踪所有 GPU

    Return:
        - `memory_trace`: 一个包含每个事件的 `UsedMemoryState` 列表（默认为跟踪脚本的每行）。

            - `UsedMemoryState` 是命名元组，包含以下字段：
                - 'frame': 一个 `Frame` 命名元组（见下文），存储当前追踪帧的信息（当前文件、当前文件中的位置）
                - 'cpu_memory': 执行该行前的 CPU RSS 内存状态
                - 'gpu_memory': 执行该行前的 GPU 使用内存（所有 GPU 的总和或仅对 `gpus_to_trace` 如果提供的 GPU）

    `Frame` 是由 `UsedMemoryState` 使用的命名元组，列出当前帧的状态。`Frame` 具有以下字段：
        - 'filename' (字符串): 当前执行的文件名
        - 'module' (字符串): 当前执行的模块名
        - 'line_number' (整数): 当前执行的行号
        - 'event' (字符串): 触发跟踪的事件（默认为 "line"）
        - 'line_text' (字符串): Python 脚本中该行的文本

    """
    # 检查是否安装了 psutil 库
    if is_psutil_available():
        # 获取当前进程的 psutil.Process 对象
        process = psutil.Process(os.getpid())
    else:
        # 如果未安装 psutil，则记录警告信息，并设置 process 为 None
        logger.warning(
            "Psutil not installed, we won't log CPU memory usage. "
            "Install psutil (pip install psutil) to use CPU memory tracing."
        )
        process = None
    # 检查是否可以使用 py3nvml 模块进行 GPU 监控
    if is_py3nvml_available():
        try:
            # 初始化 nvml 库
            nvml.nvmlInit()
            # 如果没有指定具体要追踪的 GPU 列表，则获取所有 GPU 设备的索引列表
            devices = list(range(nvml.nvmlDeviceGetCount())) if gpus_to_trace is None else gpus_to_trace
            # 关闭 nvml 库
            nvml.nvmlShutdown()
        # 捕获可能出现的 OSError 或 nvml.NVMLError 异常
        except (OSError, nvml.NVMLError):
            # 输出警告信息，指出初始化与 GPU 的通信时出现错误，因此无法进行 GPU 内存追踪
            logger.warning("Error while initializing communication with GPU. We won't perform GPU memory tracing.")
            # 禁用 GPU 内存追踪功能
            log_gpu = False
        else:
            # 如果没有异常，则根据条件决定是否记录 GPU 内存使用情况
            log_gpu = is_torch_available() or is_tf_available()
    else:
        # 如果 py3nvml 模块不可用，则输出警告信息，提示用户安装该模块以启用 GPU 内存追踪功能
        logger.warning(
            "py3nvml not installed, we won't log GPU memory usage. "
            "Install py3nvml (pip install py3nvml) to use GPU memory tracing."
        )
        # 禁用 GPU 内存追踪功能
        log_gpu = False

    # 初始化内存追踪列表
    memory_trace = []
    def traceit(frame, event, args):
        """
        定义一个追踪函数，在模块或子模块的每行执行之前执行，记录分配的内存到带有调试信息的列表中
        """
        global _is_memory_tracing_enabled

        # 如果内存追踪未启用，则直接返回 traceit 函数自身
        if not _is_memory_tracing_enabled:
            return traceit

        # 过滤事件类型
        if events_to_trace is not None:
            if isinstance(events_to_trace, str) and event != events_to_trace:
                return traceit
            elif isinstance(events_to_trace, (list, tuple)) and event not in events_to_trace:
                return traceit

        # 如果当前 frame 的全局变量中不存在 "__name__"，则返回 traceit 函数自身
        if "__name__" not in frame.f_globals:
            return traceit

        # 获取模块名
        name = frame.f_globals["__name__"]
        # 如果模块名不是字符串类型，则返回 traceit 函数自身
        if not isinstance(name, str):
            return traceit
        else:
            # 过滤要追踪的模块白名单
            if modules_to_trace is not None:
                if isinstance(modules_to_trace, str) and modules_to_trace not in name:
                    return traceit
                elif isinstance(modules_to_trace, (list, tuple)) and all(m not in name for m in modules_to_trace):
                    return traceit

            # 过滤不需要追踪的模块黑名单
            if modules_not_to_trace is not None:
                if isinstance(modules_not_to_trace, str) and modules_not_to_trace in name:
                    return traceit
                elif isinstance(modules_not_to_trace, (list, tuple)) and any(m in name for m in modules_not_to_trace):
                    return traceit

        # 记录当前追踪状态（文件名、文件中的行号等）
        lineno = frame.f_lineno
        filename = frame.f_globals["__file__"]
        # 如果文件名以 ".pyc" 或 ".pyo" 结尾，则去除最后一个字符
        if filename.endswith(".pyc") or filename.endswith(".pyo"):
            filename = filename[:-1]
        # 获取当前行的代码内容，并去除末尾的换行符
        line = linecache.getline(filename, lineno).rstrip()
        # 创建一个 Frame 对象来保存追踪状态信息
        traced_state = Frame(filename, name, lineno, event, line)

        # 记录当前内存状态（进程的 RSS 内存），并计算与先前内存状态的差异
        cpu_mem = 0
        if process is not None:
            mem = process.memory_info()
            cpu_mem = mem.rss

        gpu_mem = 0
        if log_gpu:
            # 清除 GPU 缓存
            if is_torch_available():
                torch_empty_cache()
            if is_tf_available():
                tf_context.context()._clear_caches()  # 参见 https://github.com/tensorflow/tensorflow/issues/20218#issuecomment-416771802

            # 统计所有 GPU 的已使用内存
            nvml.nvmlInit()

            for i in devices:
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                meminfo = nvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_mem += meminfo.used

            nvml.nvmlShutdown()

        # 创建一个 UsedMemoryState 对象，记录当前的追踪状态、CPU 内存和 GPU 内存
        mem_state = UsedMemoryState(traced_state, cpu_mem, gpu_mem)
        # 将当前内存状态添加到 memory_trace 列表中
        memory_trace.append(mem_state)

        # 返回 traceit 函数自身，以便在每行执行前再次调用
        return traceit

    # 将 traceit 函数设置为系统的追踪函数
    sys.settrace(traceit)
    # 设置全局变量 _is_memory_tracing_enabled 为 True，表示启用内存追踪功能
    global _is_memory_tracing_enabled
    _is_memory_tracing_enabled = True
    
    # 返回内存追踪对象或值，这可能是一个函数、类或者一个特定的对象
    return memory_trace
# 停止内存追踪并清理相关设置，如果提供了内存追踪，则返回内存追踪的摘要信息。
def stop_memory_tracing(
    memory_trace: Optional[MemoryTrace] = None, ignore_released_memory: bool = True
) -> Optional[MemorySummary]:
    """
    停止内存追踪并返回内存追踪的摘要信息。

    Args:
        `memory_trace` (optional output of start_memory_tracing, default: None):
            要转换为摘要的内存追踪
        `ignore_released_memory` (boolean, default: None):
            如果为 True，则仅计算内存增加量以获取总内存

    Return:
        - 如果 `memory_trace` 为 None，则返回 None
        - 否则返回一个 `MemorySummary` 命名元组，包含以下字段:

            - `sequential`：从提供的 `memory_trace` 计算而来的 `MemoryState` 命名元组列表，通过减去每行执行后的内存从而计算出来。
            - `cumulative`：每行的累积内存增加量的 `MemoryState` 命名元组列表，如果一行被多次执行，则累加其内存增加量。列表按内存消耗最大到最小排序（如果内存释放则可能为负数），如果 `ignore_released_memory` 为 True（默认）则忽略释放内存的行。
            - `total`：完整追踪期间的总内存增加量，作为 `Memory` 命名元组。

    `Memory` 命名元组包含以下字段：

        - `byte` (integer): 字节数
        - `string` (string): 人类可读的字符串表示 (例如："3.5MB")

    `Frame` 是命名元组，用于列出当前帧状态，包含以下字段：

        - 'filename' (string): 当前执行的文件名
        - 'module' (string): 当前执行的模块名
        - 'line_number' (int): 当前执行的行号
        - 'event' (string): 触发追踪的事件（默认为 "line"）
        - 'line_text' (string): Python 脚本中行的文本

    `MemoryState` 是命名元组，列出了帧 + CPU/GPU 内存，包含以下字段：

        - `frame` (`Frame`): 当前帧 (参见上文)
        - `cpu`: 当前帧期间消耗的 CPU 内存，作为 `Memory` 命名元组
        - `gpu`: 当前帧期间消耗的 GPU 内存，作为 `Memory` 命名元组
        - `cpu_gpu`: 当前帧期间消耗的 CPU + GPU 内存，作为 `Memory` 命名元组
    """
    global _is_memory_tracing_enabled
    # 禁用内存追踪标志
    _is_memory_tracing_enabled = False
    # 如果内存跟踪不为None且长度大于1，则执行以下操作
    if memory_trace is not None and len(memory_trace) > 1:
        # 初始化存储内存变化的列表和当前内存状态的列表
        memory_diff_trace = []
        memory_curr_trace = []

        # 使用默认字典创建累积内存字典，每个键值对的默认值为[0, 0, 0]
        cumulative_memory_dict = defaultdict(lambda: [0, 0, 0])

        # 遍历内存跟踪列表中每对相邻的帧及其内存状态
        for (
            (frame, cpu_mem, gpu_mem),
            (next_frame, next_cpu_mem, next_gpu_mem),
        ) in zip(memory_trace[:-1], memory_trace[1:]):
            # 计算 CPU 内存增量和 GPU 内存增量
            cpu_mem_inc = next_cpu_mem - cpu_mem
            gpu_mem_inc = next_gpu_mem - gpu_mem
            cpu_gpu_mem_inc = cpu_mem_inc + gpu_mem_inc
            
            # 将帧及其内存增量封装成 MemoryState 对象，加入内存差异追踪列表
            memory_diff_trace.append(
                MemoryState(
                    frame=frame,
                    cpu=Memory(cpu_mem_inc),
                    gpu=Memory(gpu_mem_inc),
                    cpu_gpu=Memory(cpu_gpu_mem_inc),
                )
            )

            # 将帧及其下一个内存状态封装成 MemoryState 对象，加入当前内存追踪列表
            memory_curr_trace.append(
                MemoryState(
                    frame=frame,
                    cpu=Memory(next_cpu_mem),
                    gpu=Memory(next_gpu_mem),
                    cpu_gpu=Memory(next_cpu_mem + next_gpu_mem),
                )
            )

            # 更新累积内存字典中当前帧的累积内存增量
            cumulative_memory_dict[frame][0] += cpu_mem_inc
            cumulative_memory_dict[frame][1] += gpu_mem_inc
            cumulative_memory_dict[frame][2] += cpu_gpu_mem_inc

        # 按照 CPU + GPU 内存增量的总和降序排序累积内存字典
        cumulative_memory = sorted(
            cumulative_memory_dict.items(), key=lambda x: x[1][2], reverse=True
        )

        # 将排序后的累积内存字典转换为 MemoryState 对象列表
        cumulative_memory = [
            MemoryState(
                frame=frame,
                cpu=Memory(cpu_mem_inc),
                gpu=Memory(gpu_mem_inc),
                cpu_gpu=Memory(cpu_gpu_mem_inc),
            )
            for frame, (cpu_mem_inc, gpu_mem_inc, cpu_gpu_mem_inc) in cumulative_memory
        ]

        # 按照当前内存追踪列表中的 CPU + GPU 内存字节数降序排序
        memory_curr_trace = sorted(memory_curr_trace, key=lambda x: x.cpu_gpu.bytes, reverse=True)

        # 如果忽略已释放的内存，则计算非负数内存的总和；否则计算所有内存的总和
        if ignore_released_memory:
            total_memory = sum(max(0, step_trace.cpu_gpu.bytes) for step_trace in memory_diff_trace)
        else:
            total_memory = sum(step_trace.cpu_gpu.bytes for step_trace in memory_diff_trace)

        # 将总内存字节数转换为 Memory 对象
        total_memory = Memory(total_memory)

        # 返回内存摘要对象，包括顺序内存追踪、累积内存、当前内存追踪和总内存
        return MemorySummary(
            sequential=memory_diff_trace,
            cumulative=cumulative_memory,
            current=memory_curr_trace,
            total=total_memory,
        )

    # 如果内存跟踪为None或长度不大于1，则返回None
    return None
# 定义一个函数，用于将字节数转换为兆字节数
def bytes_to_mega_bytes(memory_amount: int) -> int:
    """Utility to convert a number of bytes (int) into a number of mega bytes (int)"""
    return memory_amount >> 20


# 抽象基类 Benchmark，用于比较模型在 Transformers 中的内存和时间性能
class Benchmark(ABC):
    """
    Benchmarks is a simple but feature-complete benchmarking script to compare memory and time performance of models in
    Transformers.
    """

    # 类属性
    args: BenchmarkArguments
    configs: PretrainedConfig
    framework: str

    def __init__(self, args: BenchmarkArguments = None, configs: PretrainedConfig = None):
        # 初始化方法
        self.args = args
        # 如果未提供配置，则根据 args 中的模型名称动态创建配置字典
        if configs is None:
            self.config_dict = {
                model_name: AutoConfig.from_pretrained(model_name) for model_name in self.args.model_names
            }
        else:
            self.config_dict = dict(zip(self.args.model_names, configs))

        # 发出未来警告，提示该类已过时
        warnings.warn(
            f"The class {self.__class__} is deprecated. Hugging Face Benchmarking utils"
            " are deprecated in general and it is advised to use external Benchmarking libraries "
            " to benchmark Transformer models.",
            FutureWarning,
        )

        # 如果要测量内存，并且未启用多进程，则发出警告
        if self.args.memory and os.getenv("TRANSFORMERS_USE_MULTIPROCESSING") == 0:
            logger.warning(
                "Memory consumption will not be measured accurately if `args.multi_process` is set to `False.` The"
                " flag 'TRANSFORMERS_USE_MULTIPROCESSING' should only be disabled for debugging / testing."
            )

        # 初始化打印函数和其他环境信息
        self._print_fn = None
        self._framework_version = None
        self._environment_info = None

    @property
    def print_fn(self):
        # 打印函数的属性访问器
        if self._print_fn is None:
            if self.args.log_print:

                # 如果需要记录打印信息，则创建一个打印并写入日志的函数
                def print_and_log(*args):
                    with open(self.args.log_filename, "a") as log_file:
                        log_file.write("".join(args) + "\n")
                    print(*args)

                self._print_fn = print_and_log
            else:
                # 否则直接使用 print 函数
                self._print_fn = print
        return self._print_fn

    @property
    @abstractmethod
    def framework_version(self):
        # 框架版本的抽象属性
        pass

    @abstractmethod
    def _inference_speed(self, model_name: str, batch_size: int, sequence_length: int) -> float:
        # 抽象方法，用于计算推理速度
        pass

    @abstractmethod
    def _train_speed(self, model_name: str, batch_size: int, sequence_length: int) -> float:
        # 抽象方法，用于计算训练速度
        pass

    @abstractmethod
    def _inference_memory(
        self, model_name: str, batch_size: int, sequence_length: int
    ) -> [Memory, Optional[MemorySummary]]:
        # 抽象方法，用于计算推理内存消耗
        pass

    @abstractmethod
    def _train_memory(
        self, model_name: str, batch_size: int, sequence_length: int
    ) -> [Memory, Optional[MemorySummary]]:
        # 抽象方法，用于计算训练内存消耗
        pass

    def inference_speed(self, *args, **kwargs) -> float:
        # 推理速度方法，调用分离进程的包装函数执行推理速度计算
        return separate_process_wrapper_fn(self._inference_speed, self.args.do_multi_processing)(*args, **kwargs)
    # 定义一个方法，用于获取训练速度，返回一个浮点数
    def train_speed(self, *args, **kwargs) -> float:
        # 调用 separate_process_wrapper_fn 函数，用于包装 self._train_speed 方法，根据 self.args.do_multi_processing 参数决定是否多进程处理
        return separate_process_wrapper_fn(self._train_speed, self.args.do_multi_processing)(*args, **kwargs)

    # 定义一个方法，用于推断内存占用，返回一个元组，包含 Memory 对象和可选的 MemorySummary 对象
    def inference_memory(self, *args, **kwargs) -> [Memory, Optional[MemorySummary]]:
        # 调用 separate_process_wrapper_fn 函数，用于包装 self._inference_memory 方法，根据 self.args.do_multi_processing 参数决定是否多进程处理
        return separate_process_wrapper_fn(self._inference_memory, self.args.do_multi_processing)(*args, **kwargs)

    # 定义一个方法，用于获取训练内存占用，返回一个元组，包含 Memory 对象和可选的 MemorySummary 对象
    def train_memory(self, *args, **kwargs) -> [Memory, Optional[MemorySummary]]:
        # 调用 separate_process_wrapper_fn 函数，用于包装 self._train_memory 方法，根据 self.args.do_multi_processing 参数决定是否多进程处理
        return separate_process_wrapper_fn(self._train_memory, self.args.do_multi_processing)(*args, **kwargs)

    @property
    # 返回当前环境信息字典，如果还未初始化则进行初始化
    def environment_info(self):
        if self._environment_info is None:
            # 初始化空字典用于存储环境信息
            info = {}
            # 添加Transformers版本信息到环境信息字典中
            info["transformers_version"] = version
            # 添加框架名称到环境信息字典中
            info["framework"] = self.framework
            # 如果框架是PyTorch，添加是否使用TorchScript到环境信息字典中
            if self.framework == "PyTorch":
                info["use_torchscript"] = self.args.torchscript
            # 如果框架是TensorFlow，添加是否使用Eager Mode和是否使用XLA到环境信息字典中
            if self.framework == "TensorFlow":
                info["eager_mode"] = self.args.eager_mode
                info["use_xla"] = self.args.use_xla
            # 添加框架版本信息到环境信息字典中
            info["framework_version"] = self.framework_version
            # 添加Python版本信息到环境信息字典中
            info["python_version"] = platform.python_version()
            # 添加系统平台信息到环境信息字典中
            info["system"] = platform.system()
            # 添加CPU处理器信息到环境信息字典中
            info["cpu"] = platform.processor()
            # 添加系统架构信息到环境信息字典中
            info["architecture"] = platform.architecture()[0]
            # 添加当前日期到环境信息字典中
            info["date"] = datetime.date(datetime.now())
            # 添加当前时间到环境信息字典中
            info["time"] = datetime.time(datetime.now())
            # 添加是否使用FP16到环境信息字典中
            info["fp16"] = self.args.fp16
            # 添加是否使用多进程处理到环境信息字典中
            info["use_multiprocessing"] = self.args.do_multi_processing
            # 添加是否仅预训练模型到环境信息字典中
            info["only_pretrain_model"] = self.args.only_pretrain_model

            # 如果可以使用psutil库，添加CPU内存信息（单位MB）到环境信息字典中
            if is_psutil_available():
                info["cpu_ram_mb"] = bytes_to_mega_bytes(psutil.virtual_memory().total)
            else:
                # 如果psutil库不可用，记录警告信息并将CPU内存信息标记为不可用
                logger.warning(
                    "Psutil not installed, we won't log available CPU memory. "
                    "Install psutil (pip install psutil) to log available CPU memory."
                )
                info["cpu_ram_mb"] = "N/A"

            # 添加是否使用GPU到环境信息字典中
            info["use_gpu"] = self.args.is_gpu
            # 如果使用GPU，添加GPU数量信息到环境信息字典中
            if self.args.is_gpu:
                info["num_gpus"] = 1  # TODO(PVP) Currently only single GPU is supported
                # 如果可以使用py3nvml库，记录GPU相关信息到环境信息字典中
                if is_py3nvml_available():
                    nvml.nvmlInit()
                    handle = nvml.nvmlDeviceGetHandleByIndex(self.args.device_idx)
                    info["gpu"] = nvml.nvmlDeviceGetName(handle)
                    info["gpu_ram_mb"] = bytes_to_mega_bytes(nvml.nvmlDeviceGetMemoryInfo(handle).total)
                    info["gpu_power_watts"] = nvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000
                    info["gpu_performance_state"] = nvml.nvmlDeviceGetPerformanceState(handle)
                    nvml.nvmlShutdown()
                else:
                    # 如果py3nvml库不可用，记录警告信息并将GPU相关信息标记为不可用
                    logger.warning(
                        "py3nvml not installed, we won't log GPU memory usage. "
                        "Install py3nvml (pip install py3nvml) to log information about GPU."
                    )
                    info["gpu"] = "N/A"
                    info["gpu_ram_mb"] = "N/A"
                    info["gpu_power_watts"] = "N/A"
                    info["gpu_performance_state"] = "N/A"

            # 添加是否使用TPU到环境信息字典中
            info["use_tpu"] = self.args.is_tpu
            # TODO(PVP): 查看是否可以添加更多关于TPU的信息
            # 参考: https://github.com/pytorch/xla/issues/2180

            # 将完整的环境信息字典保存到实例变量中
            self._environment_info = info
        
        # 返回存储的环境信息字典
        return self._environment_info
    def print_results(self, result_dict, type_label):
        # 打印结果表头，包括模型名称、批量大小、序列长度和类型标签
        self.print_fn(80 * "-")
        self.print_fn(
            "Model Name".center(30) + "Batch Size".center(15) + "Seq Length".center(15) + type_label.center(15)
        )
        self.print_fn(80 * "-")
        # 遍历每个模型名称
        for model_name in self.args.model_names:
            # 遍历结果字典中模型名称对应的批量大小列表
            for batch_size in result_dict[model_name]["bs"]:
                # 遍历结果字典中模型名称对应的序列长度列表
                for sequence_length in result_dict[model_name]["ss"]:
                    # 获取结果字典中模型名称对应的结果数据
                    result = result_dict[model_name]["result"][batch_size][sequence_length]
                    # 如果结果是浮点数，进行格式化处理，保留三位小数或显示 "< 0.001"
                    if isinstance(result, float):
                        result = round(1000 * result) / 1000
                        result = "< 0.001" if result == 0.0 else str(result)
                    else:
                        result = str(result)
                    # 打印模型名称、批量大小、序列长度和结果数据
                    self.print_fn(
                        model_name[:30].center(30) + str(batch_size).center(15),
                        str(sequence_length).center(15),
                        result.center(15),
                    )
        # 打印结果表尾
        self.print_fn(80 * "-")

    def print_memory_trace_statistics(self, summary: MemorySummary):
        # 打印逐行内存消耗的摘要信息
        self.print_fn(
            "\nLine by line memory consumption:\n"
            + "\n".join(
                f"{state.frame.filename}:{state.frame.line_number}: mem {state.cpu_gpu}: {state.frame.line_text}"
                for state in summary.sequential
            )
        )
        # 打印具有最高内存消耗的行摘要信息
        self.print_fn(
            "\nLines with top memory consumption:\n"
            + "\n".join(
                f"=> {state.frame.filename}:{state.frame.line_number}: mem {state.cpu_gpu}: {state.frame.line_text}"
                for state in summary.cumulative[:6]
            )
        )
        # 打印具有最低内存消耗的行摘要信息
        self.print_fn(
            "\nLines with lowest memory consumption:\n"
            + "\n".join(
                f"=> {state.frame.filename}:{state.frame.line_number}: mem {state.cpu_gpu}: {state.frame.line_text}"
                for state in summary.cumulative[-6:]
            )
        )
        # 打印总内存增加量摘要信息
        self.print_fn(f"\nTotal memory increase: {summary.total}")
    # 如果未设置保存到 CSV 标志，则直接返回，不执行保存操作
    def save_to_csv(self, result_dict, filename):
        if not self.args.save_to_csv:
            return
        # 打印提示信息，表示正在保存结果到 CSV 文件
        self.print_fn("Saving results to csv.")
        # 打开指定文件名的 CSV 文件，以写入模式
        with open(filename, mode="w") as csv_file:
            # 如果模型名称列表为空，抛出数值错误异常，提示至少需要定义一个模型
            if len(self.args.model_names) <= 0:
                raise ValueError(f"At least 1 model should be defined, but got {self.model_names}")

            # 定义 CSV 文件的列名
            fieldnames = ["model", "batch_size", "sequence_length"]
            # 创建 CSV 写入器对象，指定列名和额外的 "result" 列
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames + ["result"])
            # 写入 CSV 文件的表头行
            writer.writeheader()

            # 遍历每个模型名称
            for model_name in self.args.model_names:
                # 获取当前模型在结果字典中的结果
                result_dict_model = result_dict[model_name]["result"]
                # 遍历每个批量大小（batch_size）
                for bs in result_dict_model:
                    # 遍历每个序列长度（sequence_length）
                    for ss in result_dict_model[bs]:
                        # 获取当前模型在给定批量大小和序列长度下的结果
                        result_model = result_dict_model[bs][ss]
                        # 将结果写入 CSV 文件，格式化结果值为字符串，保留小数点后四位（如果是浮点数）
                        writer.writerow(
                            {
                                "model": model_name,
                                "batch_size": bs,
                                "sequence_length": ss,
                                "result": ("{}" if not isinstance(result_model, float) else "{:.4f}").format(
                                    result_model
                                ),
                            }
                        )
```