# `.\pytorch\torch\_inductor\async_compile.py`

```
# 导入必要的模块和函数
# 允许未类型化的函数定义
from __future__ import annotations

# 导入 functools 模块，支持高阶函数操作
import functools

# 导入 logging 模块，用于记录日志信息
import logging

# 导入 multiprocessing 模块，支持多进程处理
import multiprocessing

# 导入 os 模块，提供与操作系统交互的功能
import os

# 导入 sys 模块，提供与解释器交互的功能
import sys

# 导入 concurrent.futures 模块的 Future、ProcessPoolExecutor 和 ThreadPoolExecutor 类
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor

# 导入 functools 模块的 partial 函数
from functools import partial

# 导入 time 模块的 time 函数
from time import time

# 导入 typing 模块的各种类型定义
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

# 导入 torch 库
import torch

# 导入特定模块和类，用于编译和缓存操作
from torch._dynamo.device_interface import get_registered_device_interfaces
from torch._inductor import config
from torch._inductor.codecache import (
    CodeCacheFuture,
    CppCodeCache,
    CppPythonBindingsCodeCache,
    CUDACodeCache,
    HalideCodeCache,
    LambdaFuture,
    ROCmCodeCache,
    TritonCodeCache,
    TritonFuture,
)
# 导入编译工作相关的子进程池和异步编译初始化函数
from torch._inductor.compile_worker.subproc_pool import (
    _warm_process_pool,
    AnyPool,
    SubprocPool,
)
from torch._inductor.compile_worker.watchdog import _async_compile_initializer

# 导入编译任务函数
from torch._inductor.runtime.compile_tasks import (
    _set_triton_ptxas_path,
    _worker_compile_triton,
)

# 导入 torch.hub 模块的 _Faketqdm 和 tqdm 类
from torch.hub import _Faketqdm, tqdm

# 如果在类型检查模式下
if TYPE_CHECKING:
    # 导入 HalideMeta 类型提示
    from torch._inductor.runtime.hints import HalideMeta

# 编译时间统计的累计值
_cumulative_compile_time = 0.0

# 记录开始编译的时间戳
_t0: Optional[float] = None

# 日志记录器，用于记录 kernel_code 相关信息
kernel_code_log = torch._logging.getArtifactLogger(__name__, "kernel_code")


def pre_fork_setup():
    """
    在使用进程池进行分叉之前必须进行的设置。
    """
    # 确保在分叉进程之前计算好属性
    caching_device_properties()

    # 计算 Triton 密钥可能会很慢。如果在分叉之前调用它，
    # 则会缓存到分叉的子进程中。
    try:
        from triton.compiler.compiler import triton_key

        triton_key()
    except ModuleNotFoundError:
        # 可能未安装 Triton 模块
        pass


def caching_device_properties():
    # 遍历所有注册的设备接口，获取设备属性
    for _, device_interface in get_registered_device_interfaces():
        if device_interface.is_available():
            device_interface.Worker.get_device_properties()


def _compile_start() -> None:
    """
    记录编译开始的时间戳。
    """
    global _t0
    if _t0 is None:
        _t0 = time()


def _compile_end() -> None:
    """
    计算编译时间并累加到全局统计中。
    """
    global _cumulative_compile_time, _t0
    if _t0 is not None:
        t1 = time()
        _cumulative_compile_time += t1 - _t0
        _t0 = None
        # print("CUMULATIVE COMPILE TIME", _cumulative_compile_time)


# 判断当前操作系统是否为 Windows
_IS_WINDOWS = sys.platform == "win32"

# 全局日志记录器
log = logging.getLogger(__name__)


# 用于跟踪所有已调用的进程池
_pool_set: Set[AnyPool] = set()


def shutdown_compile_workers() -> None:
    """
    关闭所有未完成的编译工作进程池。
    """
    for pool in _pool_set:
        pool.shutdown()
    # 执行分叉后的重置操作
    after_fork()


def after_fork():
    """
    在不关闭的情况下重置进程池的初始状态。
    """
    _pool_set.clear()
    # 清除异步编译的进程池缓存
    AsyncCompile.process_pool.cache_clear()


# 尝试在分叉时注册重置函数
try:
    os.register_at_fork(after_in_child=after_fork)
except AttributeError:
    pass  # 在 Windows 系统上 register_at_fork 函数不存在
class AsyncCompile:
    # 异步编译类，用于管理并发编译任务
    def __init__(self) -> None:
        # 初始化方法，不做任何操作
        pass

    @staticmethod
    @functools.lru_cache(1)
    def pool() -> ThreadPoolExecutor:
        # 返回线程池对象，确保编译线程数大于1
        assert config.compile_threads > 1
        return ThreadPoolExecutor(config.compile_threads)

    @staticmethod
    @functools.lru_cache(1)
    def process_pool() -> AnyPool:
        # 返回进程池对象，确保编译线程数大于1
        assert config.compile_threads > 1
        pool: AnyPool
        if config.worker_start_method == "subprocess":
            # 使用SubprocPool创建进程池对象，用于子进程控制
            pool = SubprocPool(config.compile_threads)
        else:
            # 执行进程池的预设置和初始化
            pre_fork_setup()
            ctx = multiprocessing.get_context(config.worker_start_method)
            pool = ProcessPoolExecutor(
                config.compile_threads,
                mp_context=ctx,
                initializer=partial(_async_compile_initializer, os.getpid()),
            )
            # 当进程池在子进程对象中创建时，正常的退出处理程序不会运行，
            # 需要注册自定义的退出处理程序
            # 由于另一个终结程序将终止发送关闭消息到工作线程，因此exitpriority必须很高
            multiprocessing.util.Finalize(None, pool.shutdown, exitpriority=sys.maxsize)

        # 将新创建的池对象添加到全局池集合中
        _pool_set.add(pool)
        return pool

    @classmethod
    def warm_pool(cls) -> None:
        # 预热线程池，确保编译线程数大于1
        if config.compile_threads <= 1:
            return
        _compile_start()
        _warm_process_pool(cls.process_pool(), config.compile_threads)
        _compile_end()

    @classmethod
    def submit(cls, task: Callable[..., Any]) -> Any:
        # 提交任务到线程池执行，如果编译线程数小于等于1，则直接执行任务
        if config.compile_threads <= 1:
            return task()
        return cls.pool().submit(task)

    def triton(self, kernel_name: str, source_code: str, device_str: str = "cuda"):
        # Triton编译器方法，编译给定的内核代码
        kernel_code_log.info("Triton Kernel:\n%s", source_code)
        _compile_start()
        _set_triton_ptxas_path()

        # 加载或重新加载指定名称和源代码的内核对象
        kernel = TritonCodeCache.load(kernel_name, source_code)
        if config.compile_threads > 1:
            # 在进程池中异步编译内核代码，支持在运行时修改这些环境变量
            env_vars = ["TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"]
            extra_env = {v: os.environ[v] for v in env_vars if v in os.environ}
            return TritonFuture(
                kernel,
                self.process_pool().submit(
                    _worker_compile_triton,
                    kernel._reload_in_subproc,
                    extra_env,
                ),
            )
        else:
            # 单线程编译内核代码
            kernel.precompile()
            return kernel

    def multi_kernel(self, *args, **kwargs) -> Any:
        # 多内核调用方法，不需要并行调用，因为子内核已经是并行任务
        from torch._inductor.codegen.multi_kernel import MultiKernelCall

        return MultiKernelCall(*args, **kwargs)
    # 定义一个方法用于处理 C++ 源代码，接受一个字符串类型的源代码作为参数
    def cpp(self, source_code: str):
        # 记录日志，记录传入的 C++ 源代码
        kernel_code_log.info("CPP Kernel:\n%s", source_code)
        # 如果编译线程数小于等于1，则直接从缓存中加载代码并返回内核对象
        if config.compile_threads <= 1:
            return CppCodeCache.load(source_code).kernel
        else:
            # 否则，异步加载 C++ 代码，使用 LambdaFuture 包装以支持延迟加载
            get_result = CppCodeCache.load_async(source_code, submit_fn=self.submit)
            return LambdaFuture(lambda: get_result().kernel)

    # 定义一个方法用于处理带 Python 绑定的 C++ 源代码，接受参数类型列表和源代码字符串作为参数
    def cpp_pybinding(self, argtypes: List[str], source_code: str):
        # 记录日志，记录传入的带 Python 绑定的 C++ 源代码
        kernel_code_log.info("CPP+Bindings Kernel:\n%s", source_code)
        # 如果编译线程数小于等于1，则直接从缓存中加载 Python 绑定的 C++ 代码
        if config.compile_threads <= 1:
            return CppPythonBindingsCodeCache.load_pybinding(argtypes, source_code)
        else:
            # 否则，异步加载 Python 绑定的 C++ 代码，使用 LambdaFuture 包装以支持延迟加载
            get_result = CppPythonBindingsCodeCache.load_pybinding_async(
                argtypes, source_code, submit_fn=self.submit
            )
            return LambdaFuture(get_result)

    # 定义一个方法用于处理 CUDA 源代码，接受源代码字符串和目标文件扩展名作为参数
    def cuda(self, source_code, dst_file_ext):
        # 记录日志，记录传入的 CUDA 源代码
        kernel_code_log.info("CUDA Kernel:\n%s", source_code)

        # 定义一个任务函数，用于加载 CUDA 代码并返回结果
        def task():
            return CUDACodeCache.load(source_code, dst_file_ext)[0]

        # 提交任务并返回结果
        return self.submit(task)

    # 定义一个方法用于处理 ROCm 源代码，接受源代码字符串和目标文件扩展名作为参数
    def rocm(self, source_code, dst_file_ext):
        # 记录日志，记录传入的 ROCm 源代码
        kernel_code_log.info("ROCm Kernel:\n%s", source_code)

        # 定义一个任务函数，用于加载 ROCm 代码并返回结果
        def task():
            return ROCmCodeCache.load(source_code, dst_file_ext)[0]

        # 提交任务并返回结果
        return self.submit(task)

    # 定义一个方法用于处理 Halide 元数据和源代码，接受元数据对象和源代码字符串作为参数
    def halide(self, meta: HalideMeta, source_code: str):
        # 记录日志，记录传入的 Halide 源代码和元数据
        kernel_code_log.info("Halide Kernel:\n%r\n%s", meta, source_code)
        # 如果编译线程数小于等于1，则直接生成 Halide 内核并返回
        if config.compile_threads <= 1:
            return HalideCodeCache.generate_halide(meta, source_code)
        else:
            # 否则，异步生成 Halide 内核，使用 LambdaFuture 包装以支持延迟加载
            get_result = HalideCodeCache.generate_halide_async(
                meta, source_code, submit_fn=self.submit
            )
            return LambdaFuture(get_result)

    # 定义一个方法用于等待编译完成，接受一个作用域字典作为参数
    def wait(self, scope: Dict[str, Any]) -> None:
        # 计算作用域中是 Future 或 CodeCacheFuture 类型的对象个数
        num_kernels = len(
            [
                value
                for key, value in scope.items()
                if isinstance(value, (Future, CodeCacheFuture))
            ]
        )
        # 创建一个进度条，显示编译进度
        pbar = tqdm(
            total=num_kernels,
            desc="Inductor Compilation",
            disable=config.disable_progress,
            delay=0,
        )
        # 如果编译线程数大于1，则并行等待所有任务完成
        if config.compile_threads > 1:
            for key, result in scope.items():
                # 如果启用详细进度显示并且不是伪装的 tqdm 进度条，则更新后缀信息
                if config.verbose_progress and not isinstance(pbar, _Faketqdm):
                    pbar.set_postfix_str(key)
                # 如果结果是 Future 或 CodeCacheFuture 对象，则等待其完成并更新作用域中的值
                if isinstance(result, (Future, CodeCacheFuture)):
                    scope[key] = result.result()
                    pbar.update(1)

        # 编译结束的清理工作
        _compile_end()
# 检查环境变量中是否设置了 TORCH_TNT_IN_USE 为 "1" 或者 TORCH_WARM_POOL 不为 "1"
if (
    os.environ.get("TORCH_TNT_IN_USE", "0") == "1"
    or os.environ.get("TORCH_WARM_POOL", "1") != "1"
):
    # 如果条件不满足，即 TORCH_TNT_IN_USE 不为 "1" 且 TORCH_WARM_POOL 为 "1"，则执行以下操作
    pass
else:
    # 调用 AsyncCompile 类的 warm_pool() 方法，用于异步编译的预热池操作
    AsyncCompile.warm_pool()
```