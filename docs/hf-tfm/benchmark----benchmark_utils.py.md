# `.\transformers\benchmark\benchmark_utils.py`

```
# 导入必要的库和模块
import copy  # 导入copy模块，用于复制对象
import csv  # 导入csv模块，用于CSV文件的读写
import linecache  # 导入linecache模块，用于缓存行
import os  # 导入os模块，用于与操作系统交互
import platform  # 导入platform模块，用于获取平台信息
import sys  # 导入sys模块，用于与Python解释器交互
import warnings  # 导入warnings模块，用于警告处理
from abc import ABC, abstractmethod  # 从abc模块中导入ABC和abstractmethod装饰器
from collections import defaultdict, namedtuple  # 导入collections模块中的defaultdict和namedtuple
from datetime import datetime  # 导入datetime模块，用于处理日期和时间
from multiprocessing import Pipe, Process, Queue  # 导入multiprocessing模块中的Pipe、Process和Queue
from multiprocessing.connection import Connection  # 导入multiprocessing.connection模块中的Connection
from typing import Callable, Iterable, List, NamedTuple, Optional, Union  # 导入typing模块中的类型提示

from .. import AutoConfig, PretrainedConfig  # 导入AutoConfig和PretrainedConfig
from .. import __version__ as version  # 导入__version__并重命名为version
from ..utils import is_psutil_available, is_py3nvml_available, is_tf_available, is_torch_available, logging  # 导入自定义工具函数和模块
from .benchmark_args_utils import BenchmarkArguments  # 从benchmark_args_utils模块中导入BenchmarkArguments类

# 检查是否有torch可用，如果有则导入相应模块
if is_torch_available():
    from torch.cuda import empty_cache as torch_empty_cache

# 检查是否有tensorflow可用，如果有则导入相应模块
if is_tf_available():
    from tensorflow.python.eager import context as tf_context

# 检查是否有psutil可用，如果有则导入相应模块
if is_psutil_available():
    import psutil

# 检查是否有py3nvml可用，如果有则导入相应模块
if is_py3nvml_available():
    import py3nvml.py3nvml as nvml

# 根据操作系统类型导入不同的信号
if platform.system() == "Windows":
    from signal import CTRL_C_EVENT as SIGKILL
else:
    from signal import SIGKILL

# 获取logger对象
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 内存追踪是否启用的标志
_is_memory_tracing_enabled = False

# 定义BenchmarkOutput命名元组，包含多个字段
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
)

# 包装函数，将函数放入独立的进程中执行，以确保准确的内存测量
def separate_process_wrapper_fn(func: Callable[[], None], do_multi_processing: bool) -> Callable[[], None]:
    """
    This function wraps another function into its own separated process. In order to ensure accurate memory
    measurements it is important that the function is executed in a separate process

    Args:
        - `func`: (`callable`): function() -> ... generic function which will be executed in its own separate process
        - `do_multi_processing`: (`bool`) Whether to run function on separate process or not
    """
    # 定义一个多进程函数，用于在单独的进程中运行函数以获取正确的内存
    def multi_process_func(*args, **kwargs):
        # 在一个独立的进程中运行函数以获取正确的内存
        def wrapper_func(queue: Queue, *args):
            try:
                # 调用函数并获取结果
                result = func(*args)
            except Exception as e:
                # 捕获异常并记录错误信息
                logger.error(e)
                print(e)
                # 将结果设置为"N/A"
                result = "N/A"
            # 将结果放入队列
            queue.put(result)

        # 创建一个队列
        queue = Queue()
        # 创建一个进程，目标函数为wrapper_func，参数为队列和args
        p = Process(target=wrapper_func, args=[queue] + list(args))
        # 启动进程
        p.start()
        # 获取队列中的结果
        result = queue.get()
        # 等待进程结束
        p.join()
        # 返回结果
        return result

    # 如果需要进行多进程处理
    if do_multi_processing:
        # 记录信息，说明函数将在自己的进程中执行
        logger.info(f"Function {func} is executed in its own process...")
        # 返回多进程函数
        return multi_process_func
    else:
        # 否则直接返回原函数
        return func
# 判断内存追踪是否启用
def is_memory_tracing_enabled():
    # 全局变量，用于标识内存追踪是否启用
    global _is_memory_tracing_enabled
    # 返回内存追踪是否启用的状态
    return _is_memory_tracing_enabled


# 定义名为 Frame 的命名元组，用于收集当前帧的状态信息
class Frame(NamedTuple):
    """
    `Frame` 是一个命名元组，用于收集当前帧的状态信息。`Frame` 具有以下字段:

        - 'filename' (string): 当前执行的文件名
        - 'module' (string): 当前执行的模块名
        - 'line_number' (int): 当前执行的行号
        - 'event' (string): 触发追踪的事件（默认为 "line"）
        - 'line_text' (string): Python 脚本中行的文本
    """

    filename: str
    module: str
    line_number: int
    event: str
    line_text: str


# 定义名为 UsedMemoryState 的命名元组，具有以下字段:
# - 'frame': 存储有关当前追踪帧（当前文件、当前文件中的位置）的信息的 `Frame` 命名元组
# - 'cpu_memory': 执行该行之前 CPU RSS 内存的状态
# - 'gpu_memory': 执行该行之前 GPU 使用的内存（如果提供了，则为所有 GPU 或仅为 `gpus_to_trace` 提供的 GPU 的总和）
class UsedMemoryState(NamedTuple):
    """
    `UsedMemoryState` 是具有以下字段的命名元组:

        - 'frame': 存储有关当前追踪帧（当前文件、当前文件中的位置）的信息的 `Frame` 命名元组
        - 'cpu_memory': 执行该行之前 CPU RSS 内存的状态
        - 'gpu_memory': 执行该行之前 GPU 使用的内存（如果提供了，则为所有 GPU 或仅为 `gpus_to_trace` 提供的 GPU 的总和）
    """

    frame: Frame
    cpu_memory: int
    gpu_memory: int


# 定义名为 Memory 的命名元组，具有单个字段 `bytes`，通过调用 `__repr__` 方法可以获得以兆字节为单位的人类可读的字符串
# - `byte` (integer): 字节的数量
class Memory(NamedTuple):
    """
    `Memory` 命名元组具有单个字段 `bytes`，通过调用 `__repr__` 方法可以获得以兆字节为单位的人类可读的字符串

        - `byte` (integer): 字节的数量
    """

    bytes: int

    def __repr__(self) -> str:
        return str(bytes_to_mega_bytes(self.bytes))


# 定义名为 MemoryState 的命名元组，列出了以下字段: 
# - `frame` (`Frame`): 当前帧（如上所述）
# - `cpu`: 在当前帧期间消耗的 CPU 内存，作为 `Memory` 命名元组
# - `gpu`: 在当前帧期间消耗的 GPU 内存，作为 `Memory` 命名元组
# - `cpu_gpu`: 在当前帧期间消耗的 CPU + GPU 内存，作为 `Memory` 命名元组
class MemoryState(NamedTuple):
    """
    `MemoryState` 是命名元组，列出了以下字段: 
    - `frame` (`Frame`): 当前帧（如上所述）
    - `cpu`: 在当前帧期间消耗的 CPU 内存，作为 `Memory` 命名元组
    - `gpu`: 在当前帧期间消耗的 GPU 内存，作为 `Memory` 命名元组
    - `cpu_gpu`: 在当前帧期间消耗的 CPU + GPU 内存，作为 `Memory` 命名元组
    """

    frame: Frame
    cpu: Memory
    gpu: Memory
    cpu_gpu: Memory


class MemorySummary(NamedTuple):
    """
    # 定义一个名为 MemorySummary 的命名元组，其包含以下字段：
    #   - sequential：从提供的 memory_trace 中计算出的 MemoryState 命名元组列表，通过减去执行每行代码后的内存与执行之前的内存得出。
    #   - cumulative：每行代码的累积内存增加量的 MemoryState 命名元组列表，通过对一行代码的内存增加量进行累加，如果该行代码被执行多次，则累加多次。列表按内存消耗最大到最小的顺序排序（如果内存被释放，则可能为负数）。
    #   - total：完整跟踪期间的总内存增加量，作为 Memory 命名元组。如果 ignore_released_memory 为 True（默认），则忽略内存释放（负数）的行。
    MemorySummary = namedtuple('MemorySummary', [
        'sequential',  # 从 memory_trace 计算的逐行内存状态列表
        'cumulative',  # 每行代码的累积内存增加量列表
        'current',     # 当前内存状态列表
        'total'        # 总内存增加量
    ])
MemoryTrace = List[UsedMemoryState]

# 定义一个别名，用于表示内存追踪的列表，其中每个元素是 UsedMemoryState 类型的对象


def measure_peak_memory_cpu(function: Callable[[], None], interval=0.5, device_idx=None) -> int:
    """
    measures peak cpu memory consumption of a given `function` running the function for at least interval seconds and
    at most 20 * interval seconds. This function is heavily inspired by: `memory_usage` of the package
    `memory_profiler`:
    https://github.com/pythonprofilers/memory_profiler/blob/895c4ac7a08020d66ae001e24067da6dcea42451/memory_profiler.py#L239

    Args:
        - `function`: (`callable`): function() -> ... function without any arguments to measure for which to measure
          the peak memory

        - `interval`: (`float`, `optional`, defaults to `0.5`) interval in second for which to measure the memory usage

        - `device_idx`: (`int`, `optional`, defaults to `None`) device id for which to measure gpu usage

    Returns:

        - `max_memory`: (`int`) consumed memory peak in Bytes
    """

    def get_cpu_memory(process_id: int) -> int:
        """
        measures current cpu memory usage of a given `process_id`

        Args:
            - `process_id`: (`int`) process_id for which to measure memory

        Returns

            - `memory`: (`int`) consumed memory in Bytes
        """
        # 使用给定的进程 ID 创建一个 Psutil 进程对象
        process = psutil.Process(process_id)
        try:
            # 获取进程的内存信息，若有 memory_info 属性则调用 memory_info()，否则调用 get_memory_info()
            meminfo_attr = "memory_info" if hasattr(process, "memory_info") else "get_memory_info"
            # 调用对应的方法获取内存使用情况，并取得使用的物理内存部分
            memory = getattr(process, meminfo_attr)()[0]
        except psutil.AccessDenied:
            # 如果 Psutil 访问权限受限，则抛出错误
            raise ValueError("Error with Psutil.")
        return memory

    # 如果 Psutil 不可用，则给出警告信息
    if not is_psutil_available():
        logger.warning(
            "Psutil not installed, we won't log CPU memory usage. "
            "Install Psutil (pip install psutil) to use CPU memory tracing."
        )
        # 设置最大内存为字符串 "N/A"
        max_memory = "N/A"

```  
    else:
        # 定义一个继承自Process的MemoryMeasureProcess类，用于测量进程的内存使用情况
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
                # 发送信号给父进程
                self.connection.send(0)
                stop = False
                while True:
                    # 更新内存使用情况
                    self.mem_usage = max(self.mem_usage, get_cpu_memory(self.process_id))
                    self.num_measurements += 1

                    if stop:
                        break

                    # 检查是否有停止信号
                    stop = self.connection.poll(self.interval)

                # 将结果发送给父进程
                self.connection.send(self.mem_usage)
                self.connection.send(self.num_measurements)

        while True:
            # 创建子进程和父进程之间的连接
            child_connection, parent_connection = Pipe()

            # 实例化MemoryMeasureProcess进程
            mem_process = MemoryMeasureProcess(os.getpid(), child_connection, interval)
            mem_process.start()

            # 等待获取内存信息
            parent_connection.recv()

            try:
                # 执行函数
                function()

                # 启动父进程连接
                parent_connection.send(0)

                # 接收内存和测量次数
                max_memory = parent_connection.recv()
                num_measurements = parent_connection.recv()
            except Exception:
                # 清理地终止进程
                parent = psutil.Process(os.getpid())
                for child in parent.children(recursive=True):
                    os.kill(child.pid, SIGKILL)
                mem_process.join(0)
                raise RuntimeError("Process killed. Error in Process")

            # 运行进程至少20 * interval次或直到完成
            mem_process.join(20 * interval)

            if (num_measurements > 4) or (interval < 1e-6):
                break

            # 减小interval
            interval /= 10

        return max_memory
# 设置内存追踪，记录模块或子模块每行的 rss 内存（RAM）。查看 `./benchmark.py` 获取用法示例。当前内存消耗使用 psutil 返回，特别是 RSS 内存 "Resident Set Size”（进程正在使用的非交换物理内存）。参考 https://psutil.readthedocs.io/en/latest/#psutil.Process.memory_info

def start_memory_tracing(
    # 要追踪的模块或子模块的名称，如果为 None，则记录所有事件；如果为字符串或字符串列表：仅记录列出的模块/子模块的事件（例如 'fairseq' 或 'transformers.models.gpt2.modeling_gpt2'）
    modules_to_trace: Optional[Union[str, Iterable[str]] = None,
    # 不要追踪的模块或子模块的名称，如果为 None，则不避免任何模块；如果为字符串或字符串列表：不记录列出的模块/子模块的事件（例如 'torch'）
    modules_not_to_trace: Optional[Union[str, Iterable[str]] = None,
    # 要记录的事件的字符串或字符串列表（查看官方 python 文档的 `sys.settrace` 获取事件列表），默认为 line
    events_to_trace: str = "line",
    # 要追踪的 GPU 列表（可选列表，默认为 None），要追踪所有 GPU 则默认为所有 GPU
    gpus_to_trace: Optional[List[int]] = None,
) -> MemoryTrace:
    """
    Args:
        - `modules_to_trace`: (None, string, list/tuple of string) if None, all events are recorded if string or list
          of strings: only events from the listed module/sub-module will be recorded (e.g. 'fairseq' or
          'transformers.models.gpt2.modeling_gpt2')
        - `modules_not_to_trace`: (None, string, list/tuple of string) if None, no module is avoided if string or list
          of strings: events from the listed module/sub-module will not be recorded (e.g. 'torch')
        - `events_to_trace`: string or list of string of events to be recorded (see official python doc for
          `sys.settrace` for the list of events) default to line
        - `gpus_to_trace`: (optional list, default None) list of GPUs to trace. Default to tracing all GPUs

    Return:

        - `memory_trace` is a list of `UsedMemoryState` for each event (default each line of the traced script).

            - `UsedMemoryState` are named tuples with the following fields:

                - 'frame': a `Frame` namedtuple (see below) storing information on the current tracing frame (current
                  file, location in current file)
                - 'cpu_memory': CPU RSS memory state *before* executing the line
                - 'gpu_memory': GPU used memory *before* executing the line (sum for all GPUs or for only
                  `gpus_to_trace` if provided)

    `Frame` is a namedtuple used by `UsedMemoryState` to list the current frame state. `Frame` has the following
    fields: - 'filename' (string): Name of the file currently executed - 'module' (string): Name of the module
    currently executed - 'line_number' (int): Number of the line currently executed - 'event' (string): Event that
    triggered the tracing (default will be "line") - 'line_text' (string): Text of the line in the python script

    """
    # 检查是否安装了 psutil
    if is_psutil_available():
        # 获取当前进程的 psutil 进程对象
        process = psutil.Process(os.getpid())
    else:
        # 如果未安装 psutil，则记录警告信息
        logger.warning(
            "Psutil not installed, we won't log CPU memory usage. "
            "Install psutil (pip install psutil) to use CPU memory tracing."
        )
        # 将进程对象设为 None
        process = None
    # 检查是否安装了 py3nvml 库
    if is_py3nvml_available():
        try:
            # 初始化 NVML 库
            nvml.nvmlInit()
            # 获取所有 GPU 设备的索引列表，如果未指定要跟踪的 GPU，则默认跟踪所有 GPU
            devices = list(range(nvml.nvmlDeviceGetCount())) if gpus_to_trace is None else gpus_to_trace
            # 关闭 NVML 库
            nvml.nvmlShutdown()
        except (OSError, nvml.NVMLError):
            # 如果初始化与 GPU 的通信时出现错误，则记录警告信息，并禁用 GPU 内存跟踪
            logger.warning("Error while initializing communication with GPU. We won't perform GPU memory tracing.")
            log_gpu = False
        else:
            # 如果成功初始化与 GPU 的通信，并且 Torch 或 TensorFlow 可用，则启用 GPU 内存跟踪
            log_gpu = is_torch_available() or is_tf_available()
    else:
        # 如果未安装 py3nvml 库，则记录警告信息，并禁用 GPU 内存跟踪
        logger.warning(
            "py3nvml not installed, we won't log GPU memory usage. "
            "Install py3nvml (pip install py3nvml) to use GPU memory tracing."
        )
        log_gpu = False

    # 初始化内存跟踪列表
    memory_trace = []
    # 定义追踪函数，该函数在模块或子模块的每行代码执行之前执行，记录内存分配情况到带有调试信息的列表中
    def traceit(frame, event, args):
        """
        Tracing method executed before running each line in a module or sub-module Record memory allocated in a list
        with debugging information
        """
        # 声明全局变量_is_memory_tracing_enabled，表示内存追踪是否启用
        global _is_memory_tracing_enabled

        # 如果内存追踪未启用，则返回追踪函数本身
        if not _is_memory_tracing_enabled:
            return traceit

        # 过滤事件
        if events_to_trace is not None:
            if isinstance(events_to_trace, str) and event != events_to_trace:
                return traceit
            elif isinstance(events_to_trace, (list, tuple)) and event not in events_to_trace:
                return traceit

        # 如果 frame 中没有 __name__，则返回追踪函数本身
        if "__name__" not in frame.f_globals:
            return traceit

        # 过滤模块
        name = frame.f_globals["__name__"]
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

        # 记录当前追踪状态（文件、文件中的位置...）
        lineno = frame.f_lineno
        filename = frame.f_globals["__file__"]
        if filename.endswith(".pyc") or filename.endswith(".pyo"):
            filename = filename[:-1]
        line = linecache.getline(filename, lineno).rstrip()
        # 构造追踪状态对象
        traced_state = Frame(filename, name, lineno, event, line)

        # 记录当前内存状态（rss 内存）并计算与之前内存状态的差异
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
                tf_context.context()._clear_caches()  # See https://github.com/tensorflow/tensorflow/issues/20218#issuecomment-416771802

            # 计算所有 GPU 的已使用内存总和
            nvml.nvmlInit()

            for i in devices:
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                meminfo = nvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_mem += meminfo.used

            nvml.nvmlShutdown()

        # 构造内存状态对象
        mem_state = UsedMemoryState(traced_state, cpu_mem, gpu_mem)
        # 将内存状态对象添加到内存追踪列表中
        memory_trace.append(mem_state)

        return traceit

    # 将追踪函数设置为系统追踪函数
    sys.settrace(traceit)
    # 声明一个全局变量，用于指示内存追踪是否启用
    global _is_memory_tracing_enabled
    # 将内存追踪标志设置为 True，表示内存追踪已启用
    _is_memory_tracing_enabled = True
    
    # 返回内存追踪函数的引用
    return memory_trace
def stop_memory_tracing(
    memory_trace: Optional[MemoryTrace] = None, ignore_released_memory: bool = True
) -> Optional[MemorySummary]:
    """
    Stop memory tracing cleanly and return a summary of the memory trace if a trace is given.

    Args:
        `memory_trace` (optional output of start_memory_tracing, default: None):
            memory trace to convert in summary
        `ignore_released_memory` (boolean, default: None):
            if True we only sum memory increase to compute total memory

    Return:

        - None if `memory_trace` is None
        - `MemorySummary` namedtuple otherwise with the fields:

            - `sequential`: a list of `MemoryState` namedtuple (see below) computed from the provided `memory_trace` by
              subtracting the memory after executing each line from the memory before executing said line.
            - `cumulative`: a list of `MemoryState` namedtuple (see below) with cumulative increase in memory for each
              line obtained by summing repeated memory increase for a line if it's executed several times. The list is
              sorted from the frame with the largest memory consumption to the frame with the smallest (can be negative
              if memory is released)
            - `total`: total memory increase during the full tracing as a `Memory` named tuple (see below). Line with
              memory release (negative consumption) are ignored if `ignore_released_memory` is `True` (default).

    `Memory` named tuple have fields

        - `byte` (integer): number of bytes,
        - `string` (string): same as human readable string (ex: "3.5MB")

    `Frame` are namedtuple used to list the current frame state and have the following fields:

        - 'filename' (string): Name of the file currently executed
        - 'module' (string): Name of the module currently executed
        - 'line_number' (int): Number of the line currently executed
        - 'event' (string): Event that triggered the tracing (default will be "line")
        - 'line_text' (string): Text of the line in the python script

    `MemoryState` are namedtuples listing frame + CPU/GPU memory with the following fields:

        - `frame` (`Frame`): the current frame (see above)
        - `cpu`: CPU memory consumed at during the current frame as a `Memory` named tuple
        - `gpu`: GPU memory consumed at during the current frame as a `Memory` named tuple
        - `cpu_gpu`: CPU + GPU memory consumed at during the current frame as a `Memory` named tuple
    """
    # 设置全局变量 _is_memory_tracing_enabled 为 False，停止内存追踪
    global _is_memory_tracing_enabled
    _is_memory_tracing_enabled = False
    # 检查内存追踪是否存在且长度大于1
    if memory_trace is not None and len(memory_trace) > 1:
        # 初始化存储内存差异追踪和当前内存追踪的列表
        memory_diff_trace = []
        memory_curr_trace = []

        # 创建默认字典，用于累积内存增量
        cumulative_memory_dict = defaultdict(lambda: [0, 0, 0])

        # 遍历内存追踪列表，计算内存增量
        for (
            (frame, cpu_mem, gpu_mem),
            (next_frame, next_cpu_mem, next_gpu_mem),
        ) in zip(memory_trace[:-1], memory_trace[1:]):
            # 计算 CPU 内存增量和 GPU 内存增量
            cpu_mem_inc = next_cpu_mem - cpu_mem
            gpu_mem_inc = next_gpu_mem - gpu_mem
            cpu_gpu_mem_inc = cpu_mem_inc + gpu_mem_inc
            # 将内存增量信息添加到内存差异追踪列表和当前内存追踪列表中
            memory_diff_trace.append(
                MemoryState(
                    frame=frame,
                    cpu=Memory(cpu_mem_inc),
                    gpu=Memory(gpu_mem_inc),
                    cpu_gpu=Memory(cpu_gpu_mem_inc),
                )
            )

            memory_curr_trace.append(
                MemoryState(
                    frame=frame,
                    cpu=Memory(next_cpu_mem),
                    gpu=Memory(next_gpu_mem),
                    cpu_gpu=Memory(next_gpu_mem + next_cpu_mem),
                )
            )

            # 更新累积内存字典中的值
            cumulative_memory_dict[frame][0] += cpu_mem_inc
            cumulative_memory_dict[frame][1] += gpu_mem_inc
            cumulative_memory_dict[frame][2] += cpu_gpu_mem_inc

        # 按照 CPU + GPU 内存增量的总和对累积内存进行排序
        cumulative_memory = sorted(
            cumulative_memory_dict.items(), key=lambda x: x[1][2], reverse=True
        )  # order by the total CPU + GPU memory increase
        # 生成累积内存列表
        cumulative_memory = [
            MemoryState(
                frame=frame,
                cpu=Memory(cpu_mem_inc),
                gpu=Memory(gpu_mem_inc),
                cpu_gpu=Memory(cpu_gpu_mem_inc),
            )
            for frame, (cpu_mem_inc, gpu_mem_inc, cpu_gpu_mem_inc) in cumulative_memory
        ]

        # 按照当前 CPU + GPU 内存大小对当前内存追踪列表进行排序
        memory_curr_trace = sorted(memory_curr_trace, key=lambda x: x.cpu_gpu.bytes, reverse=True)

        # 根据是否忽略已释放的内存计算总内存
        if ignore_released_memory:
            total_memory = sum(max(0, step_trace.cpu_gpu.bytes) for step_trace in memory_diff_trace)
        else:
            total_memory = sum(step_trace.cpu_gpu.bytes for step_trace in memory_diff_trace)

        total_memory = Memory(total_memory)

        # 返回内存摘要信息
        return MemorySummary(
            sequential=memory_diff_trace,
            cumulative=cumulative_memory,
            current=memory_curr_trace,
            total=total_memory,
        )

    # 如果内存追踪不存在或长度不足1，则返回空
    return None
# 定义一个函数，用于将字节数转换为兆字节数
def bytes_to_mega_bytes(memory_amount: int) -> int:
    """Utility to convert a number of bytes (int) into a number of mega bytes (int)"""
    # 将字节数右移 20 位，相当于除以 2^20，得到兆字节数
    return memory_amount >> 20


# 定义一个抽象基类 Benchmark
class Benchmark(ABC):
    """
    Benchmarks is a simple but feature-complete benchmarking script to compare memory and time performance of models in
    Transformers.
    """

    # 类属性：BenchmarkArguments 实例和 PretrainedConfig 实例
    args: BenchmarkArguments
    configs: PretrainedConfig
    framework: str

    # 初始化方法
    def __init__(self, args: BenchmarkArguments = None, configs: PretrainedConfig = None):
        # 初始化 args 属性
        self.args = args
        # 如果 configs 参数为 None，则为每个模型名称创建一个 AutoConfig 实例，存储到 config_dict 中
        if configs is None:
            self.config_dict = {
                model_name: AutoConfig.from_pretrained(model_name) for model_name in self.args.model_names
            }
        else:
            # 否则，将 model_names 和 configs 参数打包成字典，存储到 config_dict 中
            self.config_dict = dict(zip(self.args.model_names, configs))

        # 引发 FutureWarning 警告，说明 Benchmark 类已被弃用，建议使用外部的 Benchmarking 库来对 Transformer 模型进行基准测试
        warnings.warn(
            f"The class {self.__class__} is deprecated. Hugging Face Benchmarking utils"
            " are deprecated in general and it is advised to use external Benchmarking libraries "
            " to benchmark Transformer models.",
            FutureWarning,
        )

        # 如果参数中启用了 memory 选项，并且环境变量 TRANSFORMERS_USE_MULTIPROCESSING 为 0，则发出警告
        if self.args.memory and os.getenv("TRANSFORMERS_USE_MULTIPROCESSING") == 0:
            logger.warning(
                "Memory consumption will not be measured accurately if `args.multi_process` is set to `False.` The"
                " flag 'TRANSFORMERS_USE_MULTIPROCESSING' should only be disabled for debugging / testing."
            )

        # 初始化打印函数、框架版本和环境信息
        self._print_fn = None
        self._framework_version = None
        self._environment_info = None

    # 打印函数属性的 getter 方法
    @property
    def print_fn(self):
        # 如果打印函数属性为 None
        if self._print_fn is None:
            # 如果启用了日志打印选项
            if self.args.log_print:

                # 定义一个打印并记录日志的函数
                def print_and_log(*args):
                    with open(self.args.log_filename, "a") as log_file:
                        log_file.write("".join(args) + "\n")
                    print(*args)

                # 将打印函数设置为定义的打印并记录日志的函数
                self._print_fn = print_and_log
            else:
                # 否则，设置打印函数为默认的 print 函数
                self._print_fn = print
        # 返回打印函数
        return self._print_fn

    # 框架版本属性的抽象 getter 方法
    @property
    @abstractmethod
    def framework_version(self):
        pass

    # 抽象方法：推理速度
    @abstractmethod
    def _inference_speed(self, model_name: str, batch_size: int, sequence_length: int) -> float:
        pass

    # 抽象方法：训练速度
    @abstractmethod
    def _train_speed(self, model_name: str, batch_size: int, sequence_length: int) -> float:
        pass

    # 抽象方法：推理内存占用
    @abstractmethod
    def _inference_memory(
        self, model_name: str, batch_size: int, sequence_length: int
    ) -> [Memory, Optional[MemorySummary]]:
        pass

    # 抽象方法：训练内存占用
    @abstractmethod
    def _train_memory(
        self, model_name: str, batch_size: int, sequence_length: int
    ) -> [Memory, Optional[MemorySummary]]:
        pass

    # 推理速度方法，通过 separate_process_wrapper_fn 进行包装，根据参数决定是否多进程执行
    def inference_speed(self, *args, **kwargs) -> float:
        return separate_process_wrapper_fn(self._inference_speed, self.args.do_multi_processing)(*args, **kwargs)
    # 训练速度方法，使用多进程包装器调用_train_speed方法
    def train_speed(self, *args, **kwargs) -> float:
        return separate_process_wrapper_fn(self._train_speed, self.args.do_multi_processing)(*args, **kwargs)

    # 推理内存方法，使用多进程包装器调用_inference_memory方法
    def inference_memory(self, *args, **kwargs) -> [Memory, Optional[MemorySummary]]:
        return separate_process_wrapper_fn(self._inference_memory, self.args.do_multi_processing)(*args, **kwargs)

    # 训练内存方法，使用多进程包装器调用_train_memory方法
    def train_memory(self, *args, **kwargs) -> [Memory, Optional[MemorySummary]]:
        return separate_process_wrapper_fn(self._train_memory, self.args.do_multi_processing)(*args, **kwargs)

    # 属性
    @property
    # 返回环境信息字典，如果信息尚未初始化，则进行初始化
    def environment_info(self):
        # 如果环境信息为空
        if self._environment_info is None:
            # 创建一个空字典来存储环境信息
            info = {}
            # 添加 transformers 版本信息到环境信息字典
            info["transformers_version"] = version
            # 添加框架信息到环境信息字典
            info["framework"] = self.framework
            # 如果框架为 PyTorch
            if self.framework == "PyTorch":
                # 添加是否使用 TorchScript 到环境信息字典
                info["use_torchscript"] = self.args.torchscript
            # 如果框架为 TensorFlow
            if self.framework == "TensorFlow":
                # 添加是否使用 eager 模式到环境信息字典
                info["eager_mode"] = self.args.eager_mode
                # 添加是否使用 XLA 到环境信息字典
                info["use_xla"] = self.args.use_xla
            # 添加框架版本信息到环境信息字典
            info["framework_version"] = self.framework_version
            # 添加 Python 版本信息到环境信息字典
            info["python_version"] = platform.python_version()
            # 添加系统信息到环境信息字典
            info["system"] = platform.system()
            # 添加 CPU 信息到环境信息字典
            info["cpu"] = platform.processor()
            # 添加架构信息到环境信息字典
            info["architecture"] = platform.architecture()[0]
            # 添加当前日期信息到环境信息字典
            info["date"] = datetime.date(datetime.now())
            # 添加当前时间信息到环境信息字典
            info["time"] = datetime.time(datetime.now())
            # 添加是否使用 FP16 到环境信息字典
            info["fp16"] = self.args.fp16
            # 添加是否使用多进程到环境信息字典
            info["use_multiprocessing"] = self.args.do_multi_processing
            # 添加是否仅使用预训练模型到环境信息字典
            info["only_pretrain_model"] = self.args.only_pretrain_model

            # 如果安装了 psutil
            if is_psutil_available():
                # 添加可用 CPU 内存信息到环境信息字典
                info["cpu_ram_mb"] = bytes_to_mega_bytes(psutil.virtual_memory().total)
            else:
                # 如果未安装 psutil，则发出警告并将 CPU 内存信息设为 N/A
                logger.warning(
                    "Psutil not installed, we won't log available CPU memory. "
                    "Install psutil (pip install psutil) to log available CPU memory."
                )
                info["cpu_ram_mb"] = "N/A"

            # 添加是否使用 GPU 到环境信息字典
            info["use_gpu"] = self.args.is_gpu
            # 如果使用 GPU
            if self.args.is_gpu:
                # 目前仅支持单 GPU，因此将 GPU 数量设为 1
                info["num_gpus"] = 1  # TODO(PVP) Currently only single GPU is supported
                # 如果安装了 py3nvml
                if is_py3nvml_available():
                    # 初始化 NVML
                    nvml.nvmlInit()
                    # 获取 GPU 句柄
                    handle = nvml.nvmlDeviceGetHandleByIndex(self.args.device_idx)
                    # 添加 GPU 名称到环境信息字典
                    info["gpu"] = nvml.nvmlDeviceGetName(handle)
                    # 添加 GPU 内存信息到环境信息字典
                    info["gpu_ram_mb"] = bytes_to_mega_bytes(nvml.nvmlDeviceGetMemoryInfo(handle).total)
                    # 添加 GPU 功耗信息到环境信息字典
                    info["gpu_power_watts"] = nvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000
                    # 添加 GPU 性能状态信息到环境信息字典
                    info["gpu_performance_state"] = nvml.nvmlDeviceGetPerformanceState(handle)
                    # 关闭 NVML
                    nvml.nvmlShutdown()
                else:
                    # 如果未安装 py3nvml，则发出警告并将 GPU 相关信息设为 N/A
                    logger.warning(
                        "py3nvml not installed, we won't log GPU memory usage. "
                        "Install py3nvml (pip install py3nvml) to log information about GPU."
                    )
                    info["gpu"] = "N/A"
                    info["gpu_ram_mb"] = "N/A"
                    info["gpu_power_watts"] = "N/A"
                    info["gpu_performance_state"] = "N/A"

            # 添加是否使用 TPU 到环境信息字典
            info["use_tpu"] = self.args.is_tpu
            # TODO(PVP): 查看是否能够添加更多关于 TPU 的信息
            # 参见：https://github.com/pytorch/xla/issues/2180

            # 将初始化好的环境信息字典存储在 self._environment_info 中
            self._environment_info = info
        # 返回环境信息字典
        return self._environment_info
    # 打印结果字典中的内容，包括模型名称、批处理大小、序列长度和类型标签
    def print_results(self, result_dict, type_label):
        # 打印分隔线
        self.print_fn(80 * "-")
        # 打印表头，包括模型名称、批处理大小、序列长度和类型标签
        self.print_fn(
            "Model Name".center(30) + "Batch Size".center(15) + "Seq Length".center(15) + type_label.center(15)
        )
        # 打印分隔线
        self.print_fn(80 * "-")
        # 遍历模型名称、批处理大小和序列长度，打印结果
        for model_name in self.args.model_names:
            for batch_size in result_dict[model_name]["bs"]:
                for sequence_length in result_dict[model_name]["ss"]:
                    result = result_dict[model_name]["result"][batch_size][sequence_length]
                    # 如果结果是浮点数，保留三位小数并转换为字符串
                    if isinstance(result, float):
                        result = round(1000 * result) / 1000
                        result = "< 0.001" if result == 0.0 else str(result)
                    else:
                        result = str(result)
                    # 打印模型名称、批处理大小、序列长度和结果
                    self.print_fn(
                        model_name[:30].center(30) + str(batch_size).center(15),
                        str(sequence_length).center(15),
                        result.center(15),
                    )
        # 打印分隔线
        self.print_fn(80 * "-")

    # 打印内存跟踪统计信息
    def print_memory_trace_statistics(self, summary: MemorySummary):
        # 打印每行的内存消耗情况
        self.print_fn(
            "\nLine by line memory consumption:\n"
            + "\n".join(
                f"{state.frame.filename}:{state.frame.line_number}: mem {state.cpu_gpu}: {state.frame.line_text}"
                for state in summary.sequential
            )
        )
        # 打印内存消耗最高的行
        self.print_fn(
            "\nLines with top memory consumption:\n"
            + "\n".join(
                f"=> {state.frame.filename}:{state.frame.line_number}: mem {state.cpu_gpu}: {state.frame.line_text}"
                for state in summary.cumulative[:6]
            )
        )
        # 打印内存消耗最低的行
        self.print_fn(
            "\nLines with lowest memory consumption:\n"
            + "\n".join(
                f"=> {state.frame.filename}:{state.frame.line_number}: mem {state.cpu_gpu}: {state.frame.line_text}"
                for state in summary.cumulative[-6:]
            )
        )
        # 打印总内存增加量
        self.print_fn(f"\nTotal memory increase: {summary.total}")
    # 将结果保存到 CSV 文件中
    def save_to_csv(self, result_dict, filename):
        # 如果不需要保存到 CSV 文件，则直接返回
        if not self.args.save_to_csv:
            return
        # 打印提示信息
        self.print_fn("Saving results to csv.")
        # 打开 CSV 文件
        with open(filename, mode="w") as csv_file:
            # 如果模型名称列表为空，则抛出数值错误
            if len(self.args.model_names) <= 0:
                raise ValueError(f"At least 1 model should be defined, but got {self.model_names}")

            # 定义 CSV 文件的字段名
            fieldnames = ["model", "batch_size", "sequence_length"]
            # 创建 CSV 写入对象
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames + ["result"])
            # 写入 CSV 文件头部
            writer.writeheader()

            # 遍历模型名称列表
            for model_name in self.args.model_names:
                # 获取当前模型的结果字典
                result_dict_model = result_dict[model_name]["result"]
                # 遍历不同批次大小
                for bs in result_dict_model:
                    # 遍历不同序列长度
                    for ss in result_dict_model[bs]:
                        # 获取当前模型在当前批次大小和序列长度下的结果
                        result_model = result_dict_model[bs][ss]
                        # 写入 CSV 文件
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