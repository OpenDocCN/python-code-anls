# `.\pytorch\torch\_inductor\autotune_process.py`

```py
# mypy: allow-untyped-defs
from __future__ import annotations

import contextlib
import ctypes
import dataclasses
import functools
import logging
import os
import queue
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from ctypes import byref, c_size_t, c_void_p, CDLL
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    TYPE_CHECKING,
    Union,
)

import torch
import torch._inductor.async_compile  # noqa: F401 required to warm up AsyncCompile pools
from torch import multiprocessing
from torch._dynamo.testing import rand_strided

from torch._inductor import ir
from torch._inductor.codecache import (
    CppCodeCache,
    CUDACodeCache,
    DLLWrapper,
    get_hash,
    PyCodeCache,
)

if TYPE_CHECKING:
    from multiprocessing.process import BaseProcess
    from multiprocessing.queues import Queue
    from types import ModuleType

    from torch._inductor.select_algorithm import TritonTemplateCaller

from . import config
from .runtime.runtime_utils import do_bench_cpu, do_bench_gpu
from .virtualized import V

CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"
EXIT_HANDLER_REGISTERED = False

log = logging.getLogger(__name__)


# Used to synchronize between parent and child processes
class Ping:
    pass


class Pong:
    pass


class NonzeroWorkspaceNotSupportedError(Exception):
    pass


@contextlib.contextmanager
def set_cuda_visible_device(device: Optional[int]):
    """
    Context manager to set the CUDA_VISIBLE_DEVICES environment variable to the
    specified single device. If device is None, don't manipulate the environment.
    """
    if device is None:
        yield
        return

    current = os.environ.get(CUDA_VISIBLE_DEVICES)
    os.environ[CUDA_VISIBLE_DEVICES] = str(device)
    try:
        yield
    finally:
        if current is None:
            del os.environ[CUDA_VISIBLE_DEVICES]
        else:
            os.environ[CUDA_VISIBLE_DEVICES] = current


@dataclasses.dataclass
class TuningProcess:
    """
    Abstraction for launching a helper process to benchmark kernels. Spawns
    the parent process and uses multiprocessing queues to send benchmark
    requests and return results.
    """

    device: Optional[int] = None
    process: Optional[BaseProcess] = None
    request_queue: Optional[Queue[Any]] = None
    response_queue: Optional[Queue[Any]] = None

    @staticmethod
    def process_main(
        request_queue: Queue[Any],
        response_queue: Queue[Any],
    ) -> None:
        """
        Entry point for the child process.
        """
        log.debug(
            "Entering TuningProcess child. Visible devices = %s",
            os.environ.get(CUDA_VISIBLE_DEVICES),
        )
        try:
            TuningProcess.workloop(request_queue, response_queue)
        except Exception as ex:
            log.exception("Exception in TuningProcess")
    def workloop(request_queue: Queue[Any], response_queue: Queue[Any]) -> None:
        """
        Work loop for the benchmarking subprocess.
        """
        while True:
            obj = request_queue.get()  # 从请求队列获取一个对象

            if obj is None:
                break  # 如果对象是 None，则表示子进程应该终止
            elif isinstance(obj, Ping):
                response_queue.put(Pong())  # 如果对象是 Ping 类型，将 Pong 对象放入响应队列
            elif isinstance(obj, BenchmarkRequest):
                response_queue.put(obj.benchmark())  # 如果对象是 BenchmarkRequest 类型，将其执行结果放入响应队列
            else:
                raise RuntimeError(f"Invalid request type {type(obj)}")  # 如果对象类型无效，则引发运行时错误

    def valid(self) -> bool:
        """
        True if the sub-process has been initialized.
        """
        return (
            self.process is not None
            and self.request_queue is not None
            and self.response_queue is not None
        )  # 如果子进程、请求队列和响应队列都不为 None，则返回 True，表示子进程已初始化

    def clear(self) -> None:
        """
        Reset to an uninitialized state.
        """
        self.process = self.request_queue = self.response_queue = None  # 将子进程、请求队列和响应队列都设置为 None，重置为未初始化状态

    def initialize(self) -> None:
        """
        Create child process, request/response queues, and do the warm up.
        Set the environment to make only the provided GPU device visible
        to the process.
        """
        if self.valid():
            return  # 如果已经初始化，则直接返回

        # cuda runtime does not work with "fork", use "spawn" to start processes.
        ctx = multiprocessing.get_context("spawn")  # 获取使用 "spawn" 上下文的 multiprocessing 上下文
        self.request_queue = ctx.Queue()  # 创建请求队列
        self.response_queue = ctx.Queue()  # 创建响应队列

        self.process = ctx.Process(
            target=self.process_main,
            args=(
                self.request_queue,
                self.response_queue,
            ),
        )  # 创建子进程，目标函数为 self.process_main，参数为请求队列和响应队列

        assert self.process is not None  # 断言子进程不为 None
        with set_cuda_visible_device(self.device):
            self.process.start()  # 启动子进程

    def put(self, obj: Any) -> None:
        """
        Push a work item to the child process.
        """
        # In case of a prior crash, ensure the subprocess is running
        self.initialize()  # 确保子进程已经运行，即使之前崩溃过
        assert self.request_queue is not None  # 断言请求队列不为 None
        self.request_queue.put(obj)  # 将对象放入请求队列

    def get(
        self, result_timeout=120.0, graceful_timeout=3.0, terminate_timeout=1.0
    ):
    def get_response(self, result_timeout: Optional[float] = None, graceful_timeout: Optional[float] = None, terminate_timeout: Optional[float] = None) -> Any:
        """
        Get a response from the child process. Raises queue.Empty on timeout
        or if the process dies.

        This method is (so far) only used by TuningProcessPool, where torch._inductor.config entries are being used
        to populate the timeouts:

        Arguments:

            @param result_timeout: Timeout in seconds, defaults to 120.0 or to
                                   config.max_autotune_subproc_result_timeout_seconds when called by TuningProcessPool
            @param graceful_timeout: Timeout in seconds to allow graceful shutdown (SIGTERM is sent after this time).
                                    Defaults to 3.0 or to config.max_autotune_subproc_graceful_timeout_seconds
            @param terminate_timeout: Timeout in seconds after SIGTERM, until we send SIGKILL if the process
                                      remains alive. Defaults to 1.0 or to
                                      config.max_autotune_subproc_terminate_timeout_seconds.
        Returns:
            A response from the child process (Any type)
        """
        assert self.process is not None  # 断言确保子进程存在
        assert self.response_queue is not None  # 断言确保响应队列存在
        while True:
            try:
                remaining_timeout = result_timeout  # 设置剩余超时时间为初始超时时间
                res = None  # 初始化响应变量
                while remaining_timeout is not None and remaining_timeout >= 1.0:
                    remaining_timeout -= 0.5  # 每次循环减少0.5秒超时时间
                    try:
                        res = self.response_queue.get(timeout=0.5)  # 从响应队列中获取响应，超时时间为0.5秒
                        break
                    except queue.Empty:
                        if not self.process.is_alive():
                            raise  # 如果进程不再存活，抛出异常（将在几行下面捕获）
                if res is None:
                    res = self.response_queue.get(timeout=remaining_timeout)  # 如果未获取到响应，继续等待剩余超时时间
                return res  # 返回获取的响应
            except queue.Empty:
                status = self.process.exitcode  # 获取进程的退出码
                if status is None:
                    self.kill(
                        graceful_timeout=graceful_timeout,  # 执行优雅关闭操作
                        terminate_timeout=terminate_timeout,
                    )
                else:
                    # child process crashed
                    self.clear()  # 清理资源
                raise  # 抛出异常

    def terminate(self) -> None:
        """
        Signal the child process to terminate.
        """
        if self.valid():  # 如果对象有效
            assert self.process is not None  # 断言确保进程存在
            assert self.request_queue is not None  # 断言确保请求队列存在
            self.request_queue.put(None)  # 将None放入请求队列，通知子进程终止

    def wait(self) -> None:
        """
        Wait for the child process to exit.
        """
        if self.process is not None:  # 如果进程存在
            self.process.join()  # 等待进程退出
            self.clear()  # 清理资源
    def kill(self, graceful_timeout=5.0, terminate_timeout=1.0) -> None:
        # 尝试终止进程，首先使用 graceful_timeout 允许进程优雅退出。
        # 如果进程仍然存活，在 terminate_timeout 秒内无法结束时，将强制终止它。
        if self.process is not None:
            # 终止进程
            self.terminate()
            # 等待进程结束，超时时间为 graceful_timeout
            self.process.join(timeout=graceful_timeout)
            # 如果进程仍然存活
            if self.process.is_alive():
                # 记录警告日志，发送 SIGTERM 信号给进程的 PID
                log.warning(
                    "Sending SIGTERM to process with PID %d",
                    self.process.pid,
                )
                # 强制终止进程
                self.process.terminate()
                # 再次等待进程结束，超时时间为 terminate_timeout
                self.process.join(timeout=terminate_timeout)
                # 如果进程仍然存活
                if self.process.is_alive():
                    # 记录错误日志，发送 SIGKILL 信号给进程的 PID
                    log.error(
                        "Sending SIGKILL to process with PID %d",
                        self.process.pid,
                    )
                    # 强制杀死进程，确保结束进程
                    self.process.kill()  # This should definitely end the process
            # 清理资源
            self.clear()
@dataclasses.dataclass
class TuningProcessPool:
    """
    Maintains a pool of TuningProcesses to benchmark kernels in parallel
    across devices. By default, we create one TuningProcess per device and
    set the sub-process environment to make only that device visible.
    """

    processes: Optional[queue.Queue[TuningProcess]] = None
    executor: Optional[ThreadPoolExecutor] = None

    def initialize(self) -> None:
        """
        Start the child processes.
        """
        # 断言确保 processes 和 executor 都为 None 或都不为 None
        assert (self.processes is None) == (self.executor is None)
        if self.processes is not None:
            return

        # 获取设备列表
        devices = self.get_device_list()
        # 记录日志，显示子进程自动调谐的设备列表
        log.debug("Sub-process autotune device list: %s", devices)

        # 启动子进程并发送一个 Ping 消息来进行初始化
        self.processes = queue.Queue()
        for device in devices:
            # 创建 TuningProcess 对象
            p = TuningProcess(device=device)
            # 初始化 TuningProcess
            p.initialize()
            # 将 Ping 消息放入队列
            p.put(Ping())
            # 将 TuningProcess 对象放入 processes 队列
            self.processes.put(p)

        # 等待初始化完成
        for p in self.processes.queue:
            # 断言确保每个进程都返回 Pong 消息，即初始化成功
            assert isinstance(p.get(result_timeout=None), Pong)

        # 使用线程池来管理向子进程分发工作
        # 线程会阻塞等待可用进程，因此最大线程数应与设备数匹配
        self.executor = ThreadPoolExecutor(max_workers=len(devices))

        # 注册父进程的退出处理程序，以便终止子进程
        global EXIT_HANDLER_REGISTERED
        if not EXIT_HANDLER_REGISTERED:
            EXIT_HANDLER_REGISTERED = True
            import atexit

            # 注册 terminate 方法作为退出时的回调
            atexit.register(self.terminate)

    def get_device_list(self) -> Sequence[Optional[int]]:
        """
        Gather the list of devices to be used in the pool.
        """
        if not config.autotune_multi_device:
            # 如果未启用多设备自动调谐，则返回只含一个 None 的列表
            return [None]

        # 获取 CUDA 设备数
        count = torch.cuda.device_count()

        # 如果用户在环境变量中指定了可见设备，则使用这些设备
        if CUDA_VISIBLE_DEVICES in os.environ:
            devices = [int(d) for d in os.environ[CUDA_VISIBLE_DEVICES].split(",")]
            # 断言确保设备数不超过 CUDA 设备数
            assert len(devices) <= count
            return devices

        # 否则返回所有 CUDA 设备的列表
        return list(range(count))

    def terminate(self) -> None:
        """
        Signal all child processes to terminate.
        """
        # 如果存在 executor，则关闭线程池
        if self.executor is not None:
            self.executor.shutdown()
            self.executor = None

        # 如果存在 processes，则终止所有子进程
        if self.processes is not None:
            for p in self.processes.queue:
                p.terminate()
            for p in self.processes.queue:
                p.wait()
            self.processes = None
    # 定义一个方法用于处理目标函数调用者的选择，并返回一个浮点数作为结果
    def target(self, choice: TritonTemplateCaller) -> float:
        """
        Entry point for the thread-pool helper threads: Wait for an open TuningProcess,
        remove it from the queue, execute the benchmark in that subprocess, and return
        the TuningProcess to the queue.
        """
        # 断言选择的基准请求不为空
        assert choice.bmreq is not None
        # 断言进程池对象不为空
        assert self.processes is not None

        # 从进程池中获取一个进程
        process = self.processes.get()
        # 将基准请求放入获取的进程中
        process.put(choice.bmreq)
        try:
            # 执行基准测试，并返回结果，同时设置超时时间
            return process.get(
                config.max_autotune_subproc_result_timeout_seconds,
                config.max_autotune_subproc_graceful_timeout_seconds,
                config.max_autotune_subproc_terminate_timeout_seconds,
            )
        except queue.Empty:
            # 如果超时未能获取结果，发出警告并返回一个无穷大的浮点数，表示该选择将被忽略
            warnings.warn(
                f"Failed to benchmark choice '{choice}'. It will be ignored. "
                "Please debug the root cause in case the choice can bring perf gains."
            )
            # 将结果设为无穷大，以便忽略此选择
            return float("inf")
        finally:
            # 将处理过的进程放回进程池中
            self.processes.put(process)

    # 定义一个方法用于对每个选择进行基准测试，并返回结果字典
    def benchmark(
        self,
        choices: List[TritonTemplateCaller],
    ) -> Dict[TritonTemplateCaller, float]:
        """
        Benchmark each choice in a separate process.
        """
        # 断言进程池对象已经初始化
        assert self.processes is not None, "Tuning process pool is not initialized"
        assert self.executor is not None

        # 初始化结果字典
        results = {}

        # 使用线程池执行器来在多个子进程中分发工作并获取空闲的子进程
        for choice, result in zip(choices, self.executor.map(self.target, choices)):
            results[choice] = result

        # 返回基准测试的结果字典
        return results
# 创建一个调优处理池对象
tuning_pool = TuningProcessPool()


# LayoutOrBuffer 是一个类型别名，可以是 ir.Layout 或 ir.Buffer
@dataclasses.dataclass
class TensorMeta:
    # 表示张量的元数据，包括设备、数据类型、尺寸、步长、偏移和可选的名称
    device: torch.device
    dtype: torch.dtype
    sizes: torch._prims_common.ShapeType
    strides: torch._prims_common.StrideType
    offset: int
    name: Optional[str] = None

    # 从 IR 节点创建 TensorMeta 实例或列表
    @classmethod
    def from_irnodes(
        cls, irnodes: Union[LayoutOrBuffer, Sequence[LayoutOrBuffer]]
    ) -> Union[TensorMeta, List[TensorMeta]]:
        # 如果 irnodes 是序列，则递归处理每个元素
        if isinstance(irnodes, Sequence):
            result: List[Any] = [cls.from_irnodes(x) for x in irnodes]
            assert all(isinstance(x, TensorMeta) for x in result)
            return result
        
        # 如果 irnodes 是 Layout 类型，则创建一个虚拟的 Buffer 对象
        node = irnodes
        if isinstance(node, ir.Layout):
            node = ir.Buffer("fake", node)
        
        # 获取节点的数据类型，并确保其不为空
        dtype = node.get_dtype()
        assert dtype is not None

        # 创建并返回 TensorMeta 实例
        return TensorMeta(
            device=node.get_device(),
            dtype=dtype,
            sizes=V.graph.sizevars.size_hints(
                node.get_size(),
                fallback=config.unbacked_symint_fallback,
            ),
            strides=V.graph.sizevars.size_hints(
                node.get_stride(),
                fallback=config.unbacked_symint_fallback,
            ),
            offset=V.graph.sizevars.size_hint(
                node.get_layout().offset,
                fallback=config.unbacked_symint_fallback,
            ),
            name=node.get_name(),
        )

    # 将 TensorMeta 转换为 Torch 张量
    def to_tensor(self) -> torch.Tensor:
        return rand_strided(
            self.sizes,
            self.strides,
            device=self.device,
            dtype=self.dtype,
            extra_size=self.offset,
        )


@dataclasses.dataclass
class BenchmarkRequest:
    """
    BenchmarkRequest 类用于定义基准测试请求对象，目前仅处理 Triton 模板基准测试。
    外部核心基准测试可以在同一进程中完成，因为它们通常不会导致崩溃。

    重要提示：此类及其子类的实例必须能够跨进程边界序列化。不要在这里放置 CUDA 张量！
    """

    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: Union[TensorMeta, List[TensorMeta]],
        output_tensor_meta: Union[TensorMeta, List[TensorMeta]],
        extra_args: Iterable[Any],
    ):
        # 定义内核名称
        self.kernel_name = kernel_name

        # 如果输入张量元数据是单个 TensorMeta 实例，则转换为列表
        if isinstance(input_tensor_meta, TensorMeta):
            input_tensor_meta = [input_tensor_meta]
        self.input_tensor_meta = input_tensor_meta

        # 如果输出张量元数据是元组或列表，则确保长度为1后转换为单个 TensorMeta 实例
        if isinstance(output_tensor_meta, (tuple, list)):
            assert len(output_tensor_meta) == 1
            output_tensor_meta = output_tensor_meta[0]
        self.output_tensor_meta = output_tensor_meta

        # 额外的参数列表
        self.extra_args = extra_args

    # 创建一个运行函数，用于运行基准测试
    def make_run_fn(
        self, *input_tensors: torch.Tensor, output_tensor: torch.Tensor
    ) -> Callable[[], None]:
        raise NotImplementedError

    # 清理运行函数的资源
    def cleanup_run_fn(self) -> None:
        pass
    def do_bench(
        self,
        fn,
        *input_tensors: torch.Tensor,
        output_tensor: Optional[torch.Tensor] = None,
    ) -> float:
        # 此方法用于执行基准测试，计算并返回执行时间
        raise NotImplementedError

    def benchmark(
        self,
        *input_tensors: torch.Tensor,
        output_tensor: Optional[torch.Tensor] = None,
    ) -> float:
        # 检查是否启用调试模式
        debug = log.isEnabledFor(logging.DEBUG)
        if debug:
            # 记录起始时间戳
            start_ts = time.time()

        # 创建输入和输出张量
        if output_tensor is None:
            # 如果输出张量为None，则预期输入张量也应为空，从self.input_tensor_meta中创建输入张量，从self.output_tensor_meta创建输出张量
            assert len(input_tensors) == 0
            input_tensors = tuple(x.to_tensor() for x in self.input_tensor_meta)
            output_tensor = self.output_tensor_meta.to_tensor()

        if debug:
            # 计算创建张量的耗时
            create_tensor_elapse = time.time() - start_ts  # type: ignore[possibly-undefined]
            # 记录当前时间戳
            start_ts = time.time()
        try:
            # 创建运行函数fn，传入输入和输出张量
            fn = self.make_run_fn(*input_tensors, output_tensor=output_tensor)
        except NonzeroWorkspaceNotSupportedError:
            # 如果发生NonzeroWorkspaceNotSupportedError异常，记录日志并返回无穷大
            log.info("Skipping op due to nonzero workspace requirement")
            return float("inf")

        if debug:
            # 计算加载耗时
            load_elapse = time.time() - start_ts  # type: ignore[possibly-undefined]
            # 记录当前时间戳
            start_ts = time.time()

        # 执行基准测试，调用self.do_bench方法
        out = self.do_bench(fn, *input_tensors, output_tensor)

        if debug:
            # 计算基准测试耗时
            bench_elapse = time.time() - start_ts  # type: ignore[possibly-undefined]
            # 记录调试信息到日志
            log.debug(
                "InChildProcess %s: load %f, create tensor %f, bench %f",
                str(self),
                load_elapse,  # type: ignore[possibly-undefined]
                create_tensor_elapse,  # type: ignore[possibly-undefined]
                bench_elapse,
            )
        
        # 清理运行函数相关资源
        self.cleanup_run_fn()
        # 返回基准测试结果
        return out
class TestBenchmarkRequest(BenchmarkRequest):
    """
    Supports unit testing. Defined in this file so that the TuningProcess
    sub-process knows how to unpickle these objects.
    """

    def __init__(self, value: Optional[float] = None) -> None:
        # 初始化方法，设置对象的值属性，可选参数为浮点数
        self.value = value

    def benchmark(
        self, *input_tensors: torch.Tensor, output_tensor: Optional[torch.Tensor] = None
    ) -> float:
        # 基准测试方法，接收任意数量的输入张量和一个可选的输出张量，返回浮点数
        if self.value is None:
            # 如果值属性为 None，则抛出异常
            raise Exception("Failed to run")  # noqa: TRY002
        return self.value


class GPUDeviceBenchmarkRequest(BenchmarkRequest):
    def do_bench(
        self,
        fn,
        *input_tensors: torch.Tensor,
        output_tensor: Optional[torch.Tensor] = None,
    ) -> float:
        # 执行基准测试方法，接收一个函数和任意数量的输入张量以及一个可选的输出张量，返回浮点数
        device_idx_set = {
            tensor.device.index
            for tensor in [*input_tensors, output_tensor]
            if isinstance(tensor, torch.Tensor)
            and tensor.is_cuda
            and tensor.device.index is not None
        }
        # 断言：设备索引集合的长度应小于等于 1，否则抛出异常
        assert len(device_idx_set) <= 1, f"Can not mix devices {device_idx_set}"
        if len(device_idx_set) == 1:
            device_idx = next(iter(device_idx_set))
        else:
            device_idx = torch.cuda.current_device()

        with torch.cuda.device(device_idx):
            out = do_bench_gpu(fn)  # 执行 GPU 上的基准测试函数
            torch.cuda.synchronize()  # 同步 CUDA 操作，检测并清除可能的错误

        return out


class TritonBenchmarkRequest(GPUDeviceBenchmarkRequest):
    # 重要提示：此类的实例必须跨进程边界可序列化。不要放置 CUDA 张量在这里！
    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: Union[TensorMeta, List[TensorMeta]],
        output_tensor_meta: Union[TensorMeta, List[TensorMeta]],
        extra_args: Iterable[Any],
        module_path: str,  # 定义 triton 内核的模块路径
        module_cache_key: str,  # 模块缓存的键
        grid: List[int],  # 网格参数列表
        num_stages: int,  # 阶段数
        num_warps: int,  # warp 数
        matrix_instr_nonkdim: int = 0,  # 仅用于 hip，选择 mfma 指令的形状
    ):
        super().__init__(kernel_name, input_tensor_meta, output_tensor_meta, extra_args)
        # 调用父类的初始化方法，设置属性
        self.module_path = module_path  # 初始化模块路径属性
        self.module_cache_key = module_cache_key  # 初始化模块缓存键属性
        self.grid = grid  # 初始化网格参数属性
        self.num_stages = num_stages  # 初始化阶段数属性
        self.num_warps = num_warps  # 初始化 warp 数属性
        self.matrix_instr_nonkdim = matrix_instr_nonkdim  # 初始化矩阵指令非 k 维度属性

    def make_run_fn(
        self, *input_tensors: torch.Tensor, output_tensor: torch.Tensor
    ) -> Callable:
        # 创建并返回一个运行函数，接收输入张量和输出张量参数，返回一个可调用对象
        pass  # 该方法目前没有实现，保留了一个占位符，需要根据需要进一步实现
    # 定义一个返回空函数的函数签名，没有输入参数
    ) -> Callable[[], None]:
        # 通过模块缓存键路径和模块路径加载模块
        mod = PyCodeCache.load_by_key_path(self.module_cache_key, self.module_path)
        # 记录调试信息，包括模块缓存键和路径
        log.debug(
            "benchmark module key: %s, path: %s",
            self.module_cache_key,
            self.module_path,
        )
    
        # 获取模块中指定内核名称的运行方法
        run_method = getattr(mod, self.kernel_name).run
        # 复制额外参数列表
        extra_args = list(self.extra_args)
    
        # 新版 Triton 将 warmup 参数添加到 JITFunction.run 方法中
        # 此代码用于处理向后兼容性
        warmup_arg = {}
        import inspect
    
        # 检查运行方法的参数签名是否包含 'warmup'
        if "warmup" in inspect.signature(run_method).parameters:
            warmup_arg["warmup"] = False
    
        from torch._C import _cuda_getCurrentRawStream as get_raw_stream
    
        # 如果是 HIP 平台且矩阵指令非 K 维度，则返回部分应用的运行方法
        if torch.version.hip and self.matrix_instr_nonkdim != 0:
            return functools.partial(
                run_method,
                *input_tensors,
                output_tensor,
                *self.extra_args,
                grid=self.grid,
                **warmup_arg,
                stream=get_raw_stream(self.output_tensor_meta.device.index),
            )
        else:
            # 否则返回部分应用的运行方法，传递输入张量、输出张量和额外参数
            return functools.partial(
                run_method,
                *input_tensors,
                output_tensor,
                *self.extra_args,
                grid=self.grid,
                **warmup_arg,
                stream=get_raw_stream(self.output_tensor_meta.device.index),
            )
    
    # 函数用于预编译模块中指定内核的代码
    def precompile(self):
        # 通过模块缓存键路径和模块路径加载模块
        mod = PyCodeCache.load_by_key_path(self.module_cache_key, self.module_path)
        # 调用模块中指定内核的预编译方法
        getattr(mod, self.kernel_name).precompile()
    
    # 返回对象的字符串表示形式，包括内核名称、模块路径和模块缓存键
    def __str__(self) -> str:
        return f"{self.kernel_name=}, {self.module_path=}, {self.module_cache_key=}"
# 定义一个继承自GPUDeviceBenchmarkRequest的类，用于表示CUDA的性能基准请求
class CUDABenchmarkRequest(GPUDeviceBenchmarkRequest):
    # 重要提示：此类的实例必须在进程边界上可序列化。不要在这里放置CUDA张量！

    def __init__(
        self,
        kernel_name: str,  # CUDA内核函数的名称
        input_tensor_meta: Union[TensorMeta, List[TensorMeta]],  # 输入张量的元数据，可以是单个或列表
        output_tensor_meta: Union[TensorMeta, List[TensorMeta]],  # 输出张量的元数据，可以是单个或列表
        extra_args: Iterable[Any],  # 额外的参数，任意可迭代对象
        source_code: str,  # CUDA源代码字符串
    ):
        super().__init__(kernel_name, input_tensor_meta, output_tensor_meta, extra_args)
        self.source_code = source_code  # 存储CUDA源代码
        self.workspace_size: int = 0  # 工作空间大小，默认为0
        self.workspace: Optional[torch.Tensor] = None  # 工作空间张量，默认为空
        self.DLL: Optional[DLLWrapper] = None  # DLL包装器对象，默认为空
        self._workspace_size_updated = False  # 工作空间大小是否已更新标志，默认为False
        self.hash_key: str = ""  # 源代码的哈希键，默认为空字符串
        self.source_file: str = ""  # 源文件路径，默认为空字符串
        # 调用CUDACodeCache的write方法，将CUDA源代码写入缓存并返回哈希键和源文件路径
        self.hash_key, self.source_file = CUDACodeCache.write(self.source_code, "so")

    def precompile(self):
        # 预编译CUDA源代码，可能在单独的线程池中进行
        log.debug("Precompiling %s", self)  # 记录调试信息，显示正在预编译的对象信息
        CUDACodeCache.compile(self.source_code, "so")  # 调用CUDACodeCache的compile方法编译CUDA源代码
        log.debug("Done precompiling %s", self)  # 记录调试信息，显示预编译完成的对象信息

    def make_run_fn(
        self, *input_tensors: torch.Tensor, output_tensor: torch.Tensor
    ) -> Callable[[], None]:
        self.ensure_dll_loaded()  # 确保DLL已加载
        self.update_workspace_size()  # 更新工作空间大小
        args = [
            c_void_p(tensor.data_ptr())
            for tensor in list(input_tensors) + [output_tensor]
        ]  # 生成参数列表，包括输入张量和输出张量的指针

        # 记录详细的调试信息，显示生成运行函数时的关键参数和对象状态
        log.debug(
            "make_run_fn: self.kernel_name=%s, self.source_file=%s, self.hash_key=%s, self.DLL=%s, args=%s, self.extra_args=%s",
            self.kernel_name,
            self.source_file,
            self.hash_key,
            self.DLL,
            args,
            self.extra_args,
        )

        stream_ptr = c_void_p(torch.cuda.current_stream().cuda_stream)  # 获取当前CUDA流的指针
        run_method = getattr(self.DLL, self.kernel_name)  # 获取DLL中对应内核名称的方法

        workspace_ptr = c_void_p(0)  # 默认工作空间指针为0
        if self.workspace_size > 0:
            # 如果工作空间大小大于0，则创建一个与输出张量设备相同的浮点64位零张量作为工作空间
            self.workspace = torch.zeros(
                (self.workspace_size + 7) // 8,
                dtype=torch.float64,
                device=output_tensor.device,
            )
            workspace_ptr = c_void_p(self.workspace.data_ptr())  # 获取工作空间张量的数据指针

        # 生成部分函数，该函数包含了运行CUDA内核所需的所有参数
        return functools.partial(
            run_method,
            *args,  # 输入和输出张量的指针参数
            *self.extra_args,  # 额外的参数
            None,  # 空工作空间大小指针
            workspace_ptr,  # 设置工作空间指针
            stream_ptr,  # CUDA流指针
        )
    def update_workspace_size(self) -> None:
        if self._workspace_size_updated:
            return  # 如果已经更新过工作空间大小，则直接返回
        self.ensure_dll_loaded()  # 确保动态链接库已加载
        unique_input_count = len({meta.name for meta in self.input_tensor_meta})  # 统计输入张量元数据中的唯一名称数量
        args = [c_void_p(None) for _ in range(unique_input_count + 1)]  # 创建输入参数指针列表
        stream_ptr = c_void_p(torch.cuda.current_stream().cuda_stream)  # 获取当前 CUDA 流的指针

        run_method = getattr(self.DLL, self.kernel_name)  # 获取动态链接库中对应的内核函数
        # 获取工作空间大小并初始化工作空间
        c_workspace_size = c_size_t()
        run_method(
            *args,  # 输入指针和输出指针
            *self.extra_args,  # 额外的参数
            byref(
                c_workspace_size
            ),  # 设置工作空间大小指针以获取工作空间大小
            None,  # 空工作空间指针
            stream_ptr,
        )
        torch.cuda.synchronize()  # 同步 CUDA，检查是否有 CUDA 错误
        self.workspace_size = c_workspace_size.value  # 设置对象的工作空间大小属性
        log.debug(
            "update_workspace_size called: new workspace size=%d, self.kernel_name=%s, self.source_file=%s, self.hash_key=%s, self.DLL=%s, args=%s, self.extra_args=%s",  # noqa: B950
            self.workspace_size,
            self.kernel_name,
            self.source_file,
            self.hash_key,
            self.DLL,
            args,
            self.extra_args,
        )
        self._workspace_size_updated = True  # 标记工作空间大小已更新

    def ensure_dll_loaded(self):
        if self.DLL is None:
            # 如果动态链接库未加载，则加载并返回 DLL、哈希键和源文件
            self.DLL, self.hash_key, self.source_file = CUDACodeCache.load(
                self.source_code, "so"
            )

    def cleanup_run_fn(self) -> None:
        if self.DLL is not None:
            self.DLL.close()  # 如果动态链接库存在，则关闭它
        self.workspace = None  # 清空工作空间

    def __str__(self) -> str:
        return f"{self.kernel_name=}, {self.source_file=}, {self.hash_key=}"  # 返回对象的字符串表示形式，包括内核名称、源文件和哈希键
class CPUDeviceBenchmarkRequest(BenchmarkRequest):
    # Inherits from BenchmarkRequest class, specific to CPU device benchmarks

    def do_bench(
        self,
        fn,
        *input_tensors: torch.Tensor,
        output_tensor: Optional[torch.Tensor] = None,
    ) -> float:
        # Delegate benchmarking to CPU-specific function
        return do_bench_cpu(fn)


class CppBenchmarkRequest(CPUDeviceBenchmarkRequest):
    # Inherits from CPUDeviceBenchmarkRequest, meant for C++ kernel benchmarking

    # Important: Instances of this class have to be serializable
    # across process boundaries. Do not put Tensors in here!

    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: Union[TensorMeta, List[TensorMeta]],
        output_tensor_meta: Union[TensorMeta, List[TensorMeta]],
        extra_args: Iterable[Any],
        source_code: str,
    ):
        # Constructor for CppBenchmarkRequest
        super().__init__(kernel_name, input_tensor_meta, output_tensor_meta, extra_args)
        self.source_code = source_code
        self.hash_key = get_hash(source_code)
        self.DLL: Optional[Union[CDLL, ModuleType]] = None

    def precompile(self):
        # Prepopulate CppCodeCache with source code (no CUDA)
        # May occur in a separate ThreadPool
        log.debug("Precompiling %s", self)
        CppCodeCache.load(self.source_code, cuda=False)
        log.debug("Done precompiling %s", self)

    def make_run_fn(
        self, *input_tensors: torch.Tensor, output_tensor: torch.Tensor
    ) -> Callable[[], None]:
        # Generates a function to run the C++ kernel with given input and output tensors
        # Loads DLL from CppCodeCache (no CUDA), prepares arguments and logs details
        self.DLL = CppCodeCache.load(self.source_code, cuda=False)
        args = [tensor.data_ptr() for tensor in list(input_tensors) + [output_tensor]]
        log.debug(
            "make_run_fn: self.kernel_name=%s, self.DLL=%s, args=%s, self.extra_args=%s",
            self.kernel_name,
            self.DLL,
            args,
            self.extra_args,
        )
        run_method = getattr(self.DLL, self.kernel_name)
        # Ensure all extra_args are of type ctypes.c_ulonglong
        assert all(isinstance(arg, ctypes.c_ulonglong) for arg in self.extra_args)
        run_method.argtypes = [ctypes.c_ulonglong] * (
            len(args) + len(list(self.extra_args))
        )

        # Generate partial function.
        return functools.partial(
            run_method,
            *args,
            *self.extra_args,
        )

    def cleanup_run_fn(self) -> None:
        # Closes the DLL handle if it's open
        if self.DLL is not None:
            self.DLL.close()

    def __str__(self) -> str:
        # Returns a string representation of the instance, showing kernel_name
        return f"{self.kernel_name=}"


def benchmark_in_sub_process(
    choices: List[TritonTemplateCaller],
) -> Dict[TritonTemplateCaller, float]:
    """
    Do benchmarking in a subprocess and return the perf number (latency).
    """
    # Delegates benchmarking to tuning_pool for the given choices
    return tuning_pool.benchmark(choices)
```