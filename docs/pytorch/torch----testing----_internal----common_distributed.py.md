# `.\pytorch\torch\testing\_internal\common_distributed.py`

```py
# 忽略类型检查错误的标记
# 导入必要的模块
import abc  # 抽象基类模块
import faulthandler  # 故障处理模块
import itertools  # 迭代工具模块
import logging  # 日志记录模块
import multiprocessing  # 多进程模块
import os  # 系统操作模块
import queue  # 队列模块
import subprocess  # 子进程管理模块
import sys  # 系统模块
import tempfile  # 临时文件和目录模块
import threading  # 线程模块
import time  # 时间模块
import traceback  # 跟踪异常模块
import types  # 类型操作模块
import unittest  # 单元测试模块
from contextlib import contextmanager  # 上下文管理模块
from dataclasses import dataclass  # 数据类模块
from datetime import timedelta  # 时间间隔模块
from enum import Enum  # 枚举模块
from functools import partial, reduce, wraps  # 函数工具模块
from io import StringIO  # IO流模块
from typing import Dict, NamedTuple, Optional, Union, List, Any, Callable, Tuple  # 类型提示模块
from unittest.mock import patch  # 单元测试模块的模拟模块

import torch  # PyTorch 深度学习库
import torch._dynamo.test_case  # PyTorch 测试框架的内部工具
import torch.cuda.nccl  # PyTorch CUDA并行计算库
import torch.distributed as c10d  # PyTorch 分布式训练库
import torch.nn as nn  # PyTorch 神经网络模块
from torch.testing._internal.common_utils import (  # PyTorch 测试的内部工具
    FILE_SCHEMA,
    find_free_port,
    IS_SANDCASTLE,
    retry_on_connect_failures,
    skip_but_pass_in_sandcastle,
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_ROCM,
    TEST_WITH_TSAN,
    TestCase,
    run_tests,
)
from torch.testing._internal.distributed.multi_threaded_pg import (  # PyTorch 多线程进程组模块
    _install_threaded_pg,
    _uninstall_threaded_pg,
    ProcessLocalGroup,
)
import operator  # 操作符模块

# 设置日志记录的基本配置
logging.basicConfig(level=logging.INFO)
# 创建一个名为 __name__ 的日志记录器
logger = logging.getLogger(__name__)


# 定义一个命名元组 TestSkip 用于跳过测试的信息和退出代码
class TestSkip(NamedTuple):
    exit_code: int  # 退出代码字段
    message: str  # 消息字段


# 定义测试跳过的字典，包含各种跳过条件和相关信息
TEST_SKIPS = {
    "backend_unavailable": TestSkip(
        72, "Skipped because distributed backend is not available."
    ),
    "small_worldsize": TestSkip(73, "Skipped due to small world size."),
    "odd_worldsize": TestSkip(87, "Skipped due to odd world size."),
    "no_cuda": TestSkip(74, "CUDA is not available."),
    "multi-gpu-1": TestSkip(75, "Need at least 1 CUDA device"),
    "multi-gpu-2": TestSkip(77, "Need at least 2 CUDA devices"),
    "multi-gpu-3": TestSkip(80, "Need at least 3 CUDA devices"),
    "multi-gpu-4": TestSkip(81, "Need at least 4 CUDA devices"),
    "multi-gpu-5": TestSkip(82, "Need at least 5 CUDA devices"),
    "multi-gpu-6": TestSkip(83, "Need at least 6 CUDA devices"),
    "multi-gpu-7": TestSkip(84, "Need at least 7 CUDA devices"),
    "multi-gpu-8": TestSkip(85, "Need at least 8 CUDA devices"),
    "nccl": TestSkip(76, "c10d not compiled with NCCL support"),
    "skipIfRocm": TestSkip(78, "Test skipped for ROCm"),
    "no_peer_access": TestSkip(79, "Test skipped because no GPU peer access"),
    "generic": TestSkip(
        86, "Test skipped at subprocess level, look at subprocess log for skip reason"
    ),
    "importerror": TestSkip(88, "Test skipped due to missing import"),
}


@dataclass
class DistTestCases:
    # 不支持特定集体操作的后端
    skip_collective = {}  # 跳过集体操作的字典
    skip_collective["allgather_coalesced"] = {"nccl", "mpi", "ucc"}  # 不支持的集体操作
    skip_collective["reduce"] = set()  # 空集合，表示无需跳过
    skip_collective["sendrecv anysource"] = {"nccl", "ucc"}  # 不支持的集体操作
    skip_collective["cpu barrier"] = {"nccl", "ucc"}  # 不支持的集体操作

    # 显示某些功能是否已实现的后端特性
    backend_feature = {}  # 后端功能字典
    backend_feature["gpu"] = {"nccl", "gloo", "ucc"}  # 支持 GPU 的后端
    # 在 backend_feature 字典中添加一个名为 "cuda" 的键，并将一个包含 "nccl", "gloo", "ucc" 的集合作为其值
    backend_feature["cuda"] = {"nccl", "gloo", "ucc"}
    # 在 backend_feature 字典中添加一个名为 "ddp" 的键，并将一个包含 "nccl", "gloo", "ucc" 的集合作为其值
    backend_feature["ddp"] = {"nccl", "gloo", "ucc"}
    # 在 backend_feature 字典中添加一个名为 "subgroup" 的键，并将一个包含 "nccl", "gloo", "ucc" 的集合作为其值
    backend_feature["subgroup"] = {"nccl", "gloo", "ucc"}
    # 在 backend_feature 字典中添加一个名为 "plugin" 的键，并将一个空集合作为其值
    backend_feature["plugin"] = set()
# 根据 GPU 可用性和环境变量设置，跳过测试函数，确保每个进程在多 GPU 情况下有自己的 GPU
def skip_if_no_gpu(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 检查是否存在可用的 CUDA 设备，如果没有，则退出并返回相应退出代码
        if not torch.cuda.is_available():
            sys.exit(TEST_SKIPS["no_cuda"].exit_code)
        # 从环境变量中获取世界大小，并检查是否超过 CUDA 设备数量
        world_size = int(os.environ["WORLD_SIZE"])
        if torch.cuda.device_count() < world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{world_size}"].exit_code)

        return func(*args, **kwargs)

    return wrapper


# 根据环境变量设置，跳过测试函数，如果使用 MPI 后端且世界大小小于等于 2，则退出
def skip_if_small_worldsize(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if (os.environ["BACKEND"] != "mpi") and int(os.environ["WORLD_SIZE"]) <= 2:
            sys.exit(TEST_SKIPS["small_worldsize"].exit_code)

        return func(*args, **kwargs)

    return wrapper


# 根据环境变量设置，跳过测试函数，如果使用 MPI 后端且世界大小为奇数，则退出
def skip_if_odd_worldsize(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if (os.environ["BACKEND"] != "mpi") and int(os.environ["WORLD_SIZE"]) % 2 == 1:
            sys.exit(TEST_SKIPS["odd_worldsize"].exit_code)

        return func(*args, **kwargs)

    return wrapper


# 根据所需的 GPU 数量和后端类型设置，确保在 NCCL 后端时有足够的 GPU 设备，否则跳过
def require_n_gpus_for_nccl_backend(n, backend):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if backend == "nccl" and torch.cuda.device_count() < n:
                sys.exit(TEST_SKIPS[f"multi-gpu-{n}"].exit_code)
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


# 尝试导入 transformers 库，如果导入失败则退出测试
def import_transformers_or_skip():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                from transformers import (  # noqa: F401
                    AutoModelForMaskedLM,
                    BertConfig,
                )

                return func(*args, **kwargs)
            except ImportError:
                sys.exit(TEST_SKIPS["importerror"].exit_code)

        return wrapper

    return decorator


# 根据所需的最小 GPU 数量设置，确保至少有 x 个 GPU 设备可用，否则跳过
def skip_if_lt_x_gpu(x):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if torch.cuda.is_available() and torch.cuda.device_count() >= x:
                return func(*args, **kwargs)
            # 如果条件不满足，则退出并返回相应退出代码
            sys.exit(TEST_SKIPS[f"multi-gpu-{x}"].exit_code)

        return wrapper

    return decorator


# 如果不是 NCCL 后端或者 NCCL 后端且有足够的 GPU 设备可用，则执行测试函数；否则跳过
def nccl_skip_if_lt_x_gpu(backend, x):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if backend != "nccl":
                return func(*args, **kwargs)
            if torch.cuda.is_available() and torch.cuda.device_count() >= x:
                return func(*args, **kwargs)
            # 如果条件不满足，则退出并返回相应退出代码
            sys.exit(TEST_SKIPS[f"multi-gpu-{x}"].exit_code)

        return wrapper

    return decorator


# 验证在 ddp_logging_data 中是否记录了指定模型的错误信息
def verify_ddp_error_logged(model_DDP, err_substr):
    # Verify error was logged in ddp_logging_data.
    # 从模型的DDP对象中获取日志数据
    ddp_logging_data = model_DDP._get_ddp_logging_data()
    # 断言确保日志数据中包含"iteration"字段
    assert "iteration" in ddp_logging_data
    # 断言确保日志数据中包含"has_error"字段
    assert "has_error" in ddp_logging_data
    # 断言确保日志数据中包含"error"字段
    assert "error" in ddp_logging_data
    # 从日志数据中获取"error"字段的值
    logging_err = ddp_logging_data["error"]
    # 如果需要，移除C++的堆栈跟踪信息
    actual = (
        err_substr
        if err_substr.find("\nException raised from ") == -1
        else err_substr.split("\nException raised from ")[0]
    )
    # 断言确保实际的错误信息(actual)在日志数据的错误信息(logging_err)中
    assert (
        actual in logging_err
    ), f"Did not find expected {actual} in ddp logging data error: {logging_err}"
def with_nccl_blocking_wait(func):
    """
    Convenience decorator to set/unset TORCH_NCCL_BLOCKING_WAIT flag. Note that use of
    this decorator will override the setting of TORCH_NCCL_ASYNC_ERROR_HANDLING for
    the particular test. After the test, both TORCH_NCCL_BLOCKING_WAIT and
    TORCH_NCCL_ASYNC_ERROR_HANDLING will be restored to their original values.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Save and unset TORCH_NCCL_ASYNC_ERROR_HANDLING
        try:
            # 尝试获取并移除环境变量 TORCH_NCCL_ASYNC_ERROR_HANDLING 的值
            cached_nccl_async_error_handling: Union[str, None] = os.environ[
                "TORCH_NCCL_ASYNC_ERROR_HANDLING"
            ]
            del os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"]
        except KeyError:
            # 如果 TORCH_NCCL_ASYNC_ERROR_HANDLING 未设置
            cached_nccl_async_error_handling = None

        # Save val of TORCH_NCCL_BLOCKING_WAIT and set it.
        try:
            # 尝试获取环境变量 TORCH_NCCL_BLOCKING_WAIT 的值
            cached_nccl_blocking_wait: Union[str, None] = os.environ[
                "TORCH_NCCL_BLOCKING_WAIT"
            ]
        except KeyError:
            # 如果 TORCH_NCCL_BLOCKING_WAIT 未设置
            cached_nccl_blocking_wait = None
        finally:
            # 设置环境变量 TORCH_NCCL_BLOCKING_WAIT 为 "1"
            os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"

        try:
            # 调用被装饰的函数，并返回其结果
            ret = func(*args, **kwargs)
            return ret
        finally:
            # 恢复旧值
            if cached_nccl_async_error_handling is not None:
                os.environ[
                    "TORCH_NCCL_ASYNC_ERROR_HANDLING"
                ] = cached_nccl_async_error_handling

            if cached_nccl_blocking_wait is not None:
                os.environ["TORCH_NCCL_BLOCKING_WAIT"] = cached_nccl_blocking_wait

    return wrapper


def with_dist_debug_levels(levels):
    """
    Runs a test for each distributed debug level specified in levels.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 保存旧的 TORCH_DISTRIBUTED_DEBUG 环境变量值
            old_level = os.environ.get("TORCH_DISTRIBUTED_DEBUG", None)
            # 遍历每个指定的 debug level
            for level in levels:
                # 设置 TORCH_DISTRIBUTED_DEBUG 环境变量为当前 level
                os.environ["TORCH_DISTRIBUTED_DEBUG"] = level
                # 从环境变量设置 debug level
                c10d.set_debug_level_from_env()
                # 调用被装饰的函数，并返回其结果
                ret = func(*args, **kwargs)
                # 在所有进程间创建 barrier
                c10d.barrier()
                # 如果存在旧的 debug level，恢复它
                if old_level is not None:
                    os.environ["TORCH_DISTRIBUTED_DEBUG"] = old_level
            # 只返回最后一个测试的返回值，但由于这些是单元测试，返回值实际上并不重要，
            # 早期的测试如果失败会抛出异常。
            return ret

        return wrapper

    return decorator


def requires_gloo():
    return skip_but_pass_in_sandcastle_if(
        # 如果 c10d 不支持 Gloo 后端，跳过测试但在 Sandcastle 中标记为通过
        not c10d.is_gloo_available(),
        "c10d was not compiled with the Gloo backend",
    )


def requires_nccl_version(version, msg):
    if not c10d.is_nccl_available():
        return skip_but_pass_in_sandcastle(
            # 如果 c10d 不支持指定版本的 NCCL 后端，跳过测试但在 Sandcastle 中标记为通过
            "c10d was not compiled with the NCCL backend",
        )
    else:
        # 如果当前 CUDA NCCL 版本小于所需版本，则跳过但在沙堡中通过
        return skip_but_pass_in_sandcastle_if(
            torch.cuda.nccl.version() < version,
            f"Requires NCCL version greater than or equal to: {version}, found: {torch.cuda.nccl.version()}, reason: {msg}",
        )
def requires_nccl():
    # 如果不可用，跳过但在沙堡中通过
    return skip_but_pass_in_sandcastle_if(
        not c10d.is_nccl_available(),
        "c10d was not compiled with the NCCL backend",
    )

def requires_ucc():
    # 如果不可用，跳过但在沙堡中通过
    return skip_but_pass_in_sandcastle_if(
        not c10d.is_ucc_available(),
        "c10d was not compiled with the UCC backend",
    )

def requires_mpi():
    # 如果不可用，跳过但在沙堡中通过
    return skip_but_pass_in_sandcastle_if(
        not c10d.is_mpi_available(),
        "c10d was not compiled with the MPI backend",
    )


def skip_if_rocm(func):
    """Skips a test for ROCm"""
    # 标记函数以便在 ROCm 下跳过测试
    func.skip_if_rocm = True

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not TEST_WITH_ROCM:
            return func(*args, **kwargs)
        sys.exit(TEST_SKIPS["skipIfRocm"].exit_code)

    return wrapper


def skip_if_win32():
    # 如果在 Windows 平台上，跳过此单元测试用例
    return skip_but_pass_in_sandcastle_if(
        sys.platform == "win32",
        "This unit test case is not supported on Windows platform",
    )


@retry_on_connect_failures
def create_tcp_store(
    addr="localhost",
    world_size=1,
    is_master=True,
    timeout=timedelta(minutes=5),
    wait_for_workers=True,
    jit_class=False,
    use_libuv=True,
):
    """
    Creates a TCP store. Retries if the chosen port is already in use.
    """
    # 查找空闲端口
    port = find_free_port()
    if jit_class:
        # 如果使用 JIT 类，将超时转换为毫秒
        timeout_millisecond = int(timeout / timedelta(milliseconds=1))
        return torch.classes.dist_c10d.TCPStore(
            addr, port, world_size, is_master, timeout_millisecond
        )
    else:
        # 否则创建 c10d.TCPStore 对象
        return c10d.TCPStore(
            addr, port, world_size, is_master, wait_for_workers=wait_for_workers, use_libuv=use_libuv
        )


if TEST_WITH_TSAN:
    # TSAN 运行速度较慢。
    TIMEOUT_DEFAULT = 500
else:
    TIMEOUT_DEFAULT = int(os.getenv('DISTRIBUTED_TESTS_DEFAULT_TIMEOUT', '300'))
TIMEOUT_OVERRIDE = {"test_ddp_uneven_inputs": 400}


# https://github.com/pytorch/pytorch/issues/75665
if TEST_WITH_ROCM:
    # 对于 ROCm，重写超时设置
    TIMEOUT_OVERRIDE["test_join_kwargs"] = 200


def create_device(interface=None):
    if sys.platform == "win32" or interface is None:
        # 如果在 Windows 平台上或接口未指定，使用 Gloo 创建设备
        return c10d.ProcessGroupGloo.create_device(hostname="127.0.0.1")
    else:
        # 否则使用指定接口创建设备
        return c10d.ProcessGroupGloo.create_device(interface=interface)


def get_timeout(test_id) -> int:
    # 获取测试用例超时时间，如果未指定则使用默认超时
    return TIMEOUT_OVERRIDE.get(test_id.split(".")[-1], TIMEOUT_DEFAULT)


@contextmanager
def captured_output():
    # 捕获输出内容的上下文管理器
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def simple_sparse_reduce_tests(rank: int, world_size: int, num_inputs: int = 1):
    """
    Generate a number of basic test cases for sparse reduction.
    These cover tensors with a varying number of sparse dimensions and a varying
    number of dense dimensions. The only reduction operation we support is sum.
    """
    def generate(rank: int, world_size: int, sparse_dims: int = 1, dense_dims: int = 0):
        # 定义生成稀疏张量的函数，接受排名、世界大小、稀疏维度和密集维度作为参数

        # 第一个稀疏维度是 [0..rank]。
        # 后续的维度始终为 0，因此我们知道任意两个稀疏张量之间存在非空交集。
        indices = torch.reshape(torch.arange(rank + 1), (1, rank + 1))
        # 设置稠密张量的形状，第一维为世界大小，其余维度为 2
        shape = [world_size] + [2 for _ in range(dense_dims)]
        for _ in range(sparse_dims - 1):
            # 添加额外的稀疏维度，其指标为全零
            indices = torch.cat((indices, torch.zeros(1, rank + 1)))
            shape.append(world_size)
        # 创建值为 1 的稀疏张量
        values = torch.ones([rank + 1] + [2 for _ in range(dense_dims)])
        return torch.sparse_coo_tensor(indices, values, shape)

    def compute_sum(fn, world_size: int):
        # 计算给定函数 fn 在各个排名下生成的稀疏张量的总和
        return reduce(
            operator.add, [fn(rank, world_size) for rank in range(world_size)]
        )

    # 返回一个列表，其中包含对每个生成函数 fn 的结果
    return [
        (
            # 对于每个输入索引 i，生成 fn(num_inputs * rank + i, num_inputs * world_size) 的列表
            [
                fn(num_inputs * rank + i, num_inputs * world_size)
                for i in range(num_inputs)
            ],
            # 对于每个输入索引 i，生成 compute_sum(fn, num_inputs * world_size) 的列表
            [compute_sum(fn, num_inputs * world_size) for i in range(num_inputs)],
        )
        # 对于每个生成函数 fn，分别生成包含不同稀疏或密集维度设置的部分函数的偏函数
        for fn in [
            partial(generate, sparse_dims=1),
            partial(generate, sparse_dims=2),
            partial(generate, sparse_dims=3),
            partial(generate, dense_dims=1),
            partial(generate, dense_dims=2),
            partial(generate, dense_dims=3),
        ]
    ]
# HELPER FOR MULTIGPU TESTS
# 多GPU测试的辅助函数

def init_multigpu_helper(world_size: int, backend: str):
    """Multigpu tests are designed to simulate the multi nodes with multi
    GPUs on each node. Nccl backend requires equal #GPUs in each process.
    On a single node, all visible GPUs are evenly
    divided to subsets, each process only uses a subset.
    """
    # 获取当前可用的GPU数量
    nGPUs = torch.cuda.device_count()
    visible_devices = range(nGPUs)

    # 如果进程数大于GPU数量，则每个进程只能使用一个GPU
    nGPUs_per_process = 1
    if world_size > nGPUs:
        nGPUs_per_process = nGPUs // world_size

    # 将每个进程映射到对应的GPU列表
    rank_to_GPU = {
        i: list(visible_devices[i * nGPUs_per_process : (i + 1) * nGPUs_per_process])
        for i in range(world_size)
    }
    return rank_to_GPU


tmp_dir: Optional[tempfile.TemporaryDirectory] = None

# 初始化临时目录和环境变量
def initialize_temp_directories(init_method: Optional[str] = None) -> None:
    global tmp_dir
    tmp_dir = tempfile.TemporaryDirectory()
    os.environ["TEMP_DIR"] = tmp_dir.name
    os.mkdir(os.path.join(tmp_dir.name, "barrier"))
    os.mkdir(os.path.join(tmp_dir.name, "test_dir"))
    init_dir_path = os.path.join(tmp_dir.name, "init_dir")
    os.mkdir(init_dir_path)
    
    # 如果指定了初始化方法，则设置环境变量INIT_METHOD
    if init_method is not None:
        os.environ["INIT_METHOD"] = init_method
    else:
        # 否则设置默认的初始化方法为文件路径
        os.environ["INIT_METHOD"] = FILE_SCHEMA + os.path.join(
            init_dir_path, "shared_init_file"
        )


# 清理临时目录
def cleanup_temp_dir() -> None:
    if tmp_dir is not None:
        tmp_dir.cleanup()


# Most tests operate with this worldsize
# 大多数测试使用这个worldsize
DEFAULT_WORLD_SIZE = 4

# [How does MultiProcessTestCase work?]
# MultiProcessTestCase类如何工作的说明：

class MultiProcessTestCase(TestCase):
    # 主进程的排名设为-1
    MAIN_PROCESS_RANK = -1

    # This exit code is used to indicate that the test code had an error and
    # exited abnormally. There are certain tests that might use sys.exit() to
    # simulate failures and in those cases, we can't have an exit code of 0,
    # but we still want to ensure we didn't run into any other errors.
    # 这个退出码用于指示测试代码遇到错误并异常退出。某些测试可能使用sys.exit()来模拟失败，
    # 在这些情况下，我们不能使用退出码0，但我们仍希望确保没有遇到其他错误。
    TEST_ERROR_EXIT_CODE = 10

    # do not early terminate for distributed tests.
    # 分布式测试不要提前终止。
    def _should_stop_test_suite(self) -> bool:
        return False

    @property
    def world_size(self) -> int:
        return DEFAULT_WORLD_SIZE
    # 定义装饰器函数，用于处理测试方法的执行逻辑
    def join_or_run(self, fn):
        @wraps(fn)
        def wrapper(self):
            # 如果当前进程是主进程
            if self.rank == self.MAIN_PROCESS_RANK:
                # 执行方法来加入子进程
                self._join_processes(fn)
            else:
                # 否则直接运行原始测试函数
                fn()

        return types.MethodType(wrapper, self)

    # 初始化方法，用于设置测试实例的初始状态
    # method_name 是 unittest 中的方法名称，methodName 是 testslide 使用的方法名称
    def __init__(self, method_name: str = "runTest", methodName: str = "runTest") -> None:
        # 如果 methodName 不是 "runTest"，则使用 methodName 作为 method_name
        if methodName != "runTest":
            method_name = methodName
        # 调用父类的初始化方法，传入 method_name 作为参数
        super().__init__(method_name)
        # 获取当前实例中的测试方法 fn
        fn = getattr(self, method_name)
        # 将原始测试方法替换为加入或运行子进程的包装方法
        setattr(self, method_name, self.join_or_run(fn))

    # 设置方法，在每个测试方法运行之前调用，用于初始化测试环境
    def setUp(self) -> None:
        # 调用父类的 setUp 方法
        super().setUp()
        # 初始化跳过返回码检查的列表
        self.skip_return_code_checks = []  # type: ignore[var-annotated]
        # 初始化进程列表为空
        self.processes = []  # type: ignore[var-annotated]
        # 将当前进程的排名设为主进程的排名
        self.rank = self.MAIN_PROCESS_RANK
        # 创建一个临时文件并获取其文件名
        self.file_name = tempfile.NamedTemporaryFile(delete=False).name
        # 初始化 pid 到管道的映射，用于存储进程的错误消息管道
        self.pid_to_pipe = {}  # type: ignore[var-annotated]

    # 清理方法，在每个测试方法运行之后调用，用于清理测试环境
    def tearDown(self) -> None:
        # 调用父类的 tearDown 方法
        super().tearDown()
        # 终止所有当前运行的子进程
        for p in self.processes:
            p.terminate()
        # 清空进程列表，以防止文件描述符泄漏
        self.processes = []

    # 返回当前测试方法的名称
    def _current_test_name(self) -> str:
        # self.id() 返回测试方法的完整标识，例如 '__main__.TestDistributed.TestAdditive.test_get_rank'
        return self.id().split(".")[-1]

    # 启动多个子进程来运行测试函数
    def _start_processes(self, proc) -> None:
        # 清空进程列表
        self.processes = []
        # 对于每个进程排名，在指定的世界大小范围内
        for rank in range(int(self.world_size)):
            # 创建父子进程间的管道
            parent_conn, child_conn = torch.multiprocessing.Pipe()
            # 创建一个新的子进程
            process = proc(
                target=self.__class__._run,
                name="process " + str(rank),
                args=(rank, self._current_test_name(), self.file_name, child_conn),
            )
            # 启动子进程
            process.start()
            # 记录日志，标记子进程的启动
            logger.info("Started process %s with pid %s", rank, process.pid)
            # 将子进程的 PID 关联到其父进程的管道
            self.pid_to_pipe[process.pid] = parent_conn
            # 将子进程对象添加到进程列表中
            self.processes.append(process)

    # 使用 'spawn' 方法生成多个子进程
    def _spawn_processes(self) -> None:
        # 获取 'spawn' 上下文的进程创建器
        proc = torch.multiprocessing.get_context("spawn").Process
        # 启动子进程
        self._start_processes(proc)

    # 定义一个枚举类，包含测试事件的类型
    class Event(Enum):
        GET_TRACEBACK = 1

    @staticmethod
    def _event_listener(parent_pipe, signal_pipe, rank: int):
        # 记录信息日志，标明正在为特定排名的进程启动事件监听线程
        logger.info("Starting event listener thread for rank %s", rank)
        # 无限循环，监听来自父进程管道和信号管道的准备就绪事件
        while True:
            # 等待多进程连接上的管道准备就绪
            ready_pipes = multiprocessing.connection.wait([parent_pipe, signal_pipe])

            # 如果父进程管道准备就绪
            if parent_pipe in ready_pipes:
                # 如果父进程管道已关闭
                if parent_pipe.closed:
                    # 记录信息日志，标明进程排名为 rank 的管道已关闭，停止事件监听线程
                    logger.info(
                        "Pipe closed for process %s, stopping event listener thread", rank
                    )
                    return

                # 接收从父进程管道接收到的事件
                event = parent_pipe.recv()
                # 记录信息日志，标明接收到进程排名为 rank 的事件 event
                logger.info("Received event %s on process %s", event, rank)

                # 如果事件是获取 traceback
                if event == MultiProcessTestCase.Event.GET_TRACEBACK:
                    # 创建一个临时命名文件用于写入 traceback
                    with tempfile.NamedTemporaryFile(mode="r+") as tmp_file:
                        # 使用 faulthandler 将 traceback 写入临时文件
                        faulthandler.dump_traceback(tmp_file)
                        # 刷新缓冲区并将文件指针定位到开头以便读取
                        tmp_file.flush()
                        tmp_file.seek(0)
                        # 将临时文件内容发送回父进程管道
                        parent_pipe.send(tmp_file.read())

                        # 记录信息日志，标明进程排名为 rank 已发送 traceback
                        logger.info("Process %s sent traceback", rank)

            # 如果信号管道准备就绪，则直接返回，停止事件监听线程
            if signal_pipe in ready_pipes:
                return

    @classmethod
    def _run(cls, rank: int, test_name: str, file_name: str, parent_pipe) -> None:
        # 创建当前类的实例，用于运行特定测试
        self = cls(test_name)
        # 设置实例的排名和文件名属性
        self.rank = rank
        self.file_name = file_name
        # 运行测试，并将父进程管道作为参数传入
        self.run_test(test_name, parent_pipe)
    def run_test(self, test_name: str, parent_pipe) -> None:
        # Start event listener thread.
        signal_recv_pipe, signal_send_pipe = torch.multiprocessing.Pipe(duplex=False)
        # 创建一个双向管道用于进程间通信，并分别得到接收和发送端的管道对象

        event_listener_thread = threading.Thread(
            target=MultiProcessTestCase._event_listener,
            args=(parent_pipe, signal_recv_pipe, self.rank),
            daemon=True,
        )
        # 创建一个线程，用于监听事件，将事件监听器方法作为目标，传入所需参数，并设置为守护线程

        event_listener_thread.start()
        # 启动事件监听线程

        if sys.platform != "win32" and sys.platform != "darwin":
            # Register signal handler to dump stack traces on FATALs.
            # Windows和MacOS不支持信号处理器
            torch._C._set_print_stack_traces_on_fatal_signal(True)
        # 在发生致命信号时打印完整的堆栈跟踪信息

        # Show full C++ stacktraces when a Python error originating from C++ is raised.
        os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"
        # 设置环境变量以在Python引发的C++错误时显示完整的C++堆栈跟踪信息

        # self.id() == e.g. '__main__.TestDistributed.test_get_rank'
        # We're retrieving a corresponding test and executing it.
        try:
            getattr(self, test_name)()
            # 获取并执行指定名称的测试方法
        except unittest.SkipTest as se:
            logger.info(
                "Process %s skipping test %s for following reason: %s", self.rank, test_name, str(se)
            )
            sys.exit(TEST_SKIPS["generic"].exit_code)
            # 如果测试被跳过，记录日志并退出进程
        except Exception as e:
            logger.error(
                "Caught exception: \n%s exiting "
                "process %s with exit code: %s",
                traceback.format_exc(), self.rank, MultiProcessTestCase.TEST_ERROR_EXIT_CODE
            )
            # 如果发生异常，记录错误日志和堆栈信息，并退出进程
            parent_pipe.send(traceback.format_exc())
            # 将异常信息发送到父进程
            sys.exit(MultiProcessTestCase.TEST_ERROR_EXIT_CODE)
            # 使用指定的错误退出码退出进程
        finally:
            if signal_send_pipe is not None:
                signal_send_pipe.send(None)
                # 向信号发送管道发送空消息，通知监听线程结束

            assert event_listener_thread is not None
            event_listener_thread.join()
            # 等待事件监听线程结束
            parent_pipe.close()
            # 在测试完成后关闭管道
    # 获取已超时进程的回溯信息，不返回任何内容
    def _get_timedout_process_traceback(self) -> None:
        # 初始化一个空列表以存储管道对象
        pipes = []
        # 遍历每个进程及其索引
        for i, process in enumerate(self.processes):
            # 如果进程的退出码为None，表示进程未结束
            if process.exitcode is None:
                # 从PID到管道的映射中获取管道对象
                pipe = self.pid_to_pipe[process.pid]
                try:
                    # 向管道发送获取回溯事件
                    pipe.send(MultiProcessTestCase.Event.GET_TRACEBACK)
                    # 将进程索引和管道添加到列表中
                    pipes.append((i, pipe))
                except ConnectionError as e:
                    # 记录连接错误信息
                    logger.error(
                        "Encountered error while trying to get traceback for process %s: %s", i, e
                    )

        # 等待获取结果
        for rank, pipe in pipes:
            try:
                # 等待获取回溯信息，超时时间为5秒
                if pipe.poll(5):
                    # 如果管道已关闭，则记录无法获取回溯信息的信息
                    if pipe.closed:
                        logger.info(
                            "Pipe closed for process %s, cannot retrieve traceback", rank
                        )
                        continue

                    # 接收并记录进程超时时的回溯信息
                    traceback = pipe.recv()
                    logger.error(
                        "Process %s timed out with traceback: \n\n%s", rank, traceback
                    )
                else:
                    # 记录无法获取超时进程回溯信息的信息
                    logger.error(
                        "Could not retrieve traceback for timed out process: %s", rank
                    )
            except ConnectionError as e:
                # 记录连接错误信息
                logger.error(
                    "Encountered error while trying to get traceback for process %s: %s", rank, e
                )
    # 合并子进程的执行结果，直到所有子进程完成或超时
    def _join_processes(self, fn) -> None:
        # 获取当前测试的超时时间
        timeout = get_timeout(self.id())
        # 记录开始时间
        start_time = time.time()
        # 标记是否有子进程出现错误
        subprocess_error = False
        try:
            while True:
                # 检查是否有子进程因异常退出
                for (i, p) in enumerate(self.processes):
                    # 子进程遇到异常时的退出码
                    if p.exitcode == MultiProcessTestCase.TEST_ERROR_EXIT_CODE:
                        print(
                            f"Process {i} terminated with exit code {p.exitcode}, terminating remaining processes."
                        )
                        # 终止所有活跃的子进程
                        active_children = torch.multiprocessing.active_children()
                        for ac in active_children:
                            ac.terminate()
                        subprocess_error = True
                        break
                if subprocess_error:
                    break
                # 如果所有子进程都有有效的退出码，则退出循环
                if all(p.exitcode is not None for p in self.processes):
                    break
                # 检查是否应该超时测试，如果超时，则终止每个子进程
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    self._get_timedout_process_traceback()
                    print(
                        f"Timing out after {timeout} seconds and killing subprocesses."
                    )
                    for p in self.processes:
                        p.terminate()
                    break
                # 避免过多地忙等待，睡眠一段时间
                time.sleep(0.1)

            # 计算总耗时
            elapsed_time = time.time() - start_time

            # 如果函数在跳过返回码检查列表中，则检查是否有测试错误
            if fn in self.skip_return_code_checks:
                self._check_no_test_errors(elapsed_time)
            else:
                # 否则，检查所有子进程的返回码
                self._check_return_codes(elapsed_time)
        finally:
            # 关闭所有管道
            for pipe in self.pid_to_pipe.values():
                pipe.close()

    def _check_no_test_errors(self, elapsed_time) -> None:
        """
        检查子进程中是否有错误抛出。
        """
        for i, p in enumerate(self.processes):
            if p.exitcode is None:
                raise RuntimeError(
                    f"Process {i} timed out after {elapsed_time} seconds"
                )
            # 断言子进程的退出码不是测试错误退出码
            self.assertNotEqual(self.TEST_ERROR_EXIT_CODE, p.exitcode)

    @property
    def is_master(self) -> bool:
        # 返回当前进程是否为主进程
        return self.rank == 0
# 定义一个函数，用于运行子测试
def run_subtests(
    cls_inst,
    subtest_config: Dict[str, List[Any]],
    test_fn: Callable,
    *test_args,
    **test_kwargs: Any,
):
    """
    Runs a test function given by ``test_fn`` as a subtest according to the
    configurations specified by ``subtest_config``. This amortizes the
    costly setup overhead (including process spawn and initializing the
    process group) over the subtests.

    Args:
        subtest_config (Dict[str, List[Any]]): A mapping from subtest
            keyword argument name to a list of its possible values.
        test_fn (Callable): A callable that runs the actual test.
        test_args: Positional arguments to pass to ``test_fn``.
        test_kwargs: Keyword arguments to pass to ``test_fn``.
    """
    # 将配置项映射转换为列表，以确保固定顺序
    subtest_config_items: List[Tuple[str, List[Any]]] = list(subtest_config.items())
    subtest_config_keys: List[str] = [item[0] for item in subtest_config_items]
    subtest_config_values: List[List[Any]] = [item[1] for item in subtest_config_items]
    # 遍历配置值的笛卡尔积，生成子测试的各种组合
    for values in itertools.product(*subtest_config_values):
        # 将关键字与选择的值映射为字典
        subtest_kwargs = dict(zip(subtest_config_keys, values))
        # 在测试上下文中运行子测试
        with cls_inst.subTest(**subtest_kwargs):
            # 重置 Torch 的 Dynamo 模块状态
            torch._dynamo.reset()
            # 运行测试函数，传递位置参数、关键字参数以及子测试的关键字参数
            test_fn(*test_args, **test_kwargs, **subtest_kwargs)
            # 再次重置 Torch 的 Dynamo 模块状态
            torch._dynamo.reset()
        # 执行 c10d 的屏障操作，同步各个进程的执行
        c10d.barrier()


# 不能使用 functools.cache，因为它需要 Python 3.9
EFA_PROBE_RESULT = None


def has_efa() -> bool:
    """
    If shell command `fi_info -p efa -t FI_EP_RDM` returns exit code 0 then we assume that the machine has
    Libfabric EFA interfaces and EFA software components installed,
    see https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html.
    """
    global EFA_PROBE_RESULT
    # 如果已经有探测结果，则直接返回
    if EFA_PROBE_RESULT is not None:
        return EFA_PROBE_RESULT

    try:
        # 运行 shell 命令 `fi_info -p efa -t FI_EP_RDM`，并检查其返回码是否为 0
        EFA_PROBE_RESULT = (
            subprocess.run(["fi_info", "-p", "efa", "-t", "FI_EP_RDM"], check=False).returncode == 0
        )
    except FileNotFoundError:
        # 如果命令未找到，则设置探测结果为 False
        EFA_PROBE_RESULT = False
    # 返回探测结果
    return EFA_PROBE_RESULT


def tp_transports():
    """
    If the machine has Libfabric EFA interfaces and EFA software components installed it may cause
    'RuntimeError: In operator() at tensorpipe/common/ibv.h:172 "": Operation not supported' if tensorpipe
    uses InfiniBand transport, so we exclude it from tensorpipe transports,
    see https://github.com/pytorch/pytorch/issues/73885 and https://github.com/pytorch/pytorch/issues/65022
    """
    # 如果机器安装了 Libfabric EFA 接口和软件组件，则排除 InfiniBand 传输，否则返回 None
    return ["shm", "uv"] if has_efa() else None


def spawn_threads_and_init_comms(
    func=None, timeout=TIMEOUT_DEFAULT, world_size=DEFAULT_WORLD_SIZE
):
    """
    Wrapper to use with a test method
    """
    # 如果 func 参数为空，则返回一个部分应用了 timeout 和 world_size 的 spawn_threads_and_init_comms 函数
    if func is None:
        return partial(
            spawn_threads_and_init_comms, timeout=timeout, world_size=world_size
        )
    def _run_test_method_with_multi_threads(world_size, callback):
        # 安装多线程 PyTorch 分布式包并返回全局世界对象
        world = _install_threaded_pg()
        # 创建全局哈希存储对象
        global_store = c10d.HashStore()

        def world_is_valid():
            # 检查当前世界对象是否与全局世界对象相等
            return world == c10d.distributed_c10d._world

        def worker(rank, world_pg, store):
            # 初始化进程组，使用线程后端，指定 rank、世界大小和存储对象
            c10d.init_process_group(
                backend="threaded", rank=rank, world_size=world_size, store=store
            )
            try:
                callback()  # 执行回调函数
            except BaseException as ex:
                # 异常由 MultiThreadedTestCase 处理
                MultiThreadedTestCase.exception_queue.put((rank, sys.exc_info()))
                ProcessLocalGroup.exception_handle(ex)  # 触发 _terminate 事件并唤醒工作线程
            finally:
                if world_is_valid():
                    c10d.destroy_process_group()  # 销毁进程组

        threads = []
        for rank in range(world_size):
            t = threading.Thread(target=worker, args=(rank, world, global_store))
            t.start()  # 启动线程
            threads.append(t)

        return threads  # 返回线程列表


    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # TODO: 从 kwargs 中获取测试名称
        torch._C._distributed_c10d._set_thread_isolation_mode(True)
        try:
            threads = _run_test_method_with_multi_threads(world_size, lambda: func(self, *args, **kwargs))
            # 等待所有线程结束并处理可能的错误
            MultiThreadedTestCase._join_threads(threads, func)
        finally:
            torch._C._distributed_c10d._set_thread_isolation_mode(False)

    return wrapper
class MultiThreadedTestCase(TestCase):
    """
    Test runner that runs all tests with the in-proc process group using
    multiple threads with the threaded process group.

    Each test spawns world_size threads and run the test method in each thread.

    Difference from regular MultiProcess test runner:
    Must explicitly defines SetUp and call self._spawn_threads() to run the tests.
    Cannot use setUp / tearDown (must use perThreadSetup / perThreadShutdown)
        to set up / tear down each thread when running each test.
    No global state possible
        How bad of a limitation is this?
    """
    # 定义异常队列，用于在线程间传递异常信息
    exception_queue = queue.Queue()

    # 主线程的排名，设为-1
    MAIN_THREAD_RANK = -1

    # 方法装饰器，根据当前线程的排名选择运行方法或者等待主线程的方法
    def join_or_run(self, fn):
        @wraps(fn)
        def wrapper(self):
            if self.rank == self.MAIN_THREAD_RANK:
                self._join_threads(self.threads, fn)
            else:
                fn()
        return types.MethodType(wrapper, self)

    def __init__(self, method_name: str = "runTest", methodName: str = "runTest") -> None:
        # methodName 是 unittest 中的正确命名方式，testslide 使用关键字参数。
        # 因此我们需要同时支持两者，以兼容旧版本并支持 testslide。
        if methodName != "runTest":
            method_name = methodName
        # 调用父类的初始化方法，传入指定的方法名
        super().__init__(method_name)
        # 获取指定方法名对应的方法对象
        fn = getattr(self, method_name)
        # 用装饰后的方法替换原始方法，根据当前线程的排名选择执行方式
        setattr(self, method_name, self.join_or_run(fn))

    def perThreadSetUp(self):
        # 子类需实现此方法来设置每个线程的环境
        # super().setUp()  # TestCase.setUp() 调用 torch.manual_seed()
        pass

    def perThreadTearDown(self):
        # 子类需实现此方法来清理每个线程的环境
        pass

    def setUp(self) -> None:
        """
        设置主线程的测试环境，如需配置子线程环境，请使用 perThreadSetUp
        """
        # 调用父类的 setUp 方法，设置主线程的测试环境
        super().setUp()
        # 设置当前线程的排名为主线程的排名
        self.rank = self.MAIN_THREAD_RANK
        # 初始化线程列表为空
        self.threads = []
        # 设置环境变量，以显示当从 C++ 引发 Python 错误时的完整 C++ 堆栈跟踪
        os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"

    def tearDown(self):
        """
        清理主线程的测试环境，如需清理子线程环境，请使用 perThreadTearDown
        """
        # 调用父类的 tearDown 方法，清理主线程的测试环境
        super().tearDown()
        # 清空线程列表
        self.threads = []
    def _spawn_threads(self):
        """
        class method to spawn threads and run test, use this method in the SetUp of your TestCase
        """
        # 设置线程隔离模式为True，确保每个线程有独立的状态
        torch._C._distributed_c10d._set_thread_isolation_mode(True)
        # 获取当前测试用例的名称
        test_name = self._current_test_name
        # 为每个测试用例创建线程本地的进程组，并且创建一个全局存储
        world = _install_threaded_pg()
        self.__class__.global_store = c10d.HashStore()

        def world_is_valid():
            return world == c10d.distributed_c10d._world

        # 检查当前的进程组是否有效
        if not world_is_valid():
            raise RuntimeError("Invalid world")

        # 为每个进程（线程）创建并启动线程，每个线程会执行 _run 方法
        for rank in range(self.world_size):
            t = threading.Thread(target=self.__class__._run, args=(test_name, rank, self.world_size))
            t.start()
            self.threads.append(t)

    @classmethod
    def _run(cls, test_name, rank, world_size):
        """
        Initialize a new instance of the test case class, set up thread-local settings,
        and run the test associated with `test_name` using the threaded process group.
        """
        self = cls(test_name)
        self.rank = rank

        # 精度/相对容差是线程本地的设置，因为每个测试可能需要覆盖它们，确保每个线程具有相同的值
        # 这在使用操作数据库测试时特别重要，例如使用 instantiate_device_type_tests() 时
        # TODO: 寻找更好的方法来处理这一问题
        if hasattr(self, "_tls"):
            self._tls = threading.local()
            self._tls.precision = TestCase._precision
            self._tls.rel_tol = TestCase._rel_tol

        # 运行使用线程化进程组的测试
        self.run_test_with_threaded_pg(test_name, rank, world_size)

    def run_test_with_threaded_pg(self, test_name, rank, world_size):
        """
        Run the current test associated with `test_name` using the threaded process group.
        """
        # 初始化使用线程化进程组的过程组
        c10d.init_process_group(
            backend="threaded", rank=rank, world_size=world_size, store=self.__class__.global_store
        )
        # 执行每个线程的测试前设置
        self.perThreadSetUp()

        try:
            # 调用当前测试用例的指定测试方法
            getattr(self, test_name)()
        except BaseException as ex:
            # 将异常信息放入异常队列中，以便后续处理
            self.exception_queue.put((rank, sys.exc_info()))
            ProcessLocalGroup.exception_handle(ex)  # 触发 _terminate 事件并唤醒工作线程
        finally:
            # 销毁进程组
            c10d.destroy_process_group()
            # 执行每个线程的测试后清理
            self.perThreadTearDown()
    # 定义类方法 _join_threads，用于等待多线程任务完成并处理超时异常
    def _join_threads(cls, threads, fn):
        # 设置超时时间，默认为 TIMEOUT_DEFAULT
        timeout = TIMEOUT_DEFAULT
        try:
            # 遍历所有线程，并等待每个线程完成，最长等待时间为 timeout 秒
            for idx, thread in enumerate(threads):
                thread.join(max(0, timeout))
                # 如果线程仍在运行，则将超时异常信息放入异常队列中
                if thread.is_alive():
                    MultiThreadedTestCase.exception_queue.put(
                        (
                            idx,
                            (
                                TimeoutError,
                                TimeoutError(
                                    f"Rank failed to join in under {timeout} seconds"
                                ),
                                None,
                            ),
                        )
                    )
            # 重置进程本地组
            ProcessLocalGroup.reset()
            # 初始化一个列表，用于存储失败的线程信息
            failed_ranks = []
            # 处理异常队列中的所有异常信息，将其添加到失败线程列表中
            while not cls.exception_queue.empty():
                failure = cls.exception_queue.get()
                failed_ranks.append(failure)
        finally:
            # 卸载线程化的进程组
            _uninstall_threaded_pg()
            # 关闭分布式 C10D 的线程隔离模式
            torch._C._distributed_c10d._set_thread_isolation_mode(False)

        # 调用类方法 _check_return_codes，检查返回码并处理失败线程信息
        cls._check_return_codes(failed_ranks, timeout, fn)

    @classmethod
    def _check_return_codes(cls, failed_ranks, timeout, fn):
        # 检查并处理多线程返回的异常情况
        #   SkipTest: 对于每个线程打印信息
        #   TimeoutError: 如果有线程超时，则抛出 RuntimeError
        #   普通 Exception: 对于每个抛出异常的线程打印错误信息，并最终抛出 RuntimeError
        error_msg = ""  # 初始化错误信息字符串
        skip_code = -1  # 初始化跳过测试的退出码

        # 遍历每个失败的线程及其异常信息
        for rank, exc_info in failed_ranks:
            exc = exc_info[1]  # 获取异常对象
            if isinstance(exc, unittest.SkipTest):
                # 如果是 SkipTest 异常，记录日志并设置跳过测试的退出码
                logger.info(
                    "Thread %s skipping test %s for following reason: %s", rank, fn, str(exc)
                )
                if skip_code < 0:
                    skip_code = TEST_SKIPS["generic"].exit_code
            elif isinstance(exc, TimeoutError):
                # 如果是 TimeoutError 异常，记录错误日志并抛出 RuntimeError
                msg = f"Thread {rank} terminated or timed out after {timeout} seconds\n"
                logger.error(msg)
                raise RuntimeError(msg)
            elif isinstance(exc, Exception):
                # 如果是普通异常，记录错误日志并将异常信息添加到错误信息字符串中
                msg = "".join(traceback.format_exception(*exc_info))
                logger.error(
                    "Caught exception: \n%s exiting thread %s", msg, rank
                )
                error_msg += (
                    f"Thread {rank} exited with exception:\n{msg}\n"
                )
            elif isinstance(exc, SystemExit):
                # 如果是 SystemExit 异常，检查退出码是否为整数并更新跳过测试的退出码
                if type(exc.code) == int and skip_code < 0:
                    skip_code = exc.code

        # 检查是否有异常信息，如果有则抛出 RuntimeError
        if len(error_msg) > 0:
            raise RuntimeError(error_msg)
        
        # 检查是否有跳过测试的退出码，如果有则根据退出码处理跳过测试
        if skip_code > 0:
            for skip in TEST_SKIPS.values():
                if skip_code == skip.exit_code:
                    if IS_SANDCASTLE:
                        # 在沙堡环境中跳过测试并记录日志
                        logger.info(
                            "Skipping %s on sandcastle for the following reason: %s", fn, skip.message
                        )
                        return
                    else:
                        # 在非沙堡环境中抛出 unittest.SkipTest
                        raise unittest.SkipTest(skip.message)

    @property
    def world_size(self) -> int:
        # 返回默认的世界大小
        return DEFAULT_WORLD_SIZE

    @property
    def _current_test_name(self) -> str:
        # 返回当前测试的名称，通过分割 id() 字符串得到
        # self.id() == e.g. '__main__.TestDistributed.TestAdditive.test_get_rank'
        return self.id().split(".")[-1]

    def assertEqualOnRank(self, x, y, msg=None, *, rank=0):
        """
        这个工具函数之所以存在，而不是使用 self.assertEqual，
        是因为所有线程共享一个 CPU 随机数生成器，
        因此断言结果仅在 rank 为 0 时可靠。
        """
        if self.rank == rank:
            # 如果当前 rank 等于指定的 rank，调用 self.assertEqual 进行断言
            self.assertEqual(x, y, msg)

    def assertNotEqualOnRank(self, x, y, msg=None, *, rank=0):
        # 如果当前 rank 等于指定的 rank，调用 self.assertNotEqual 进行断言
        if self.rank == rank:
            self.assertNotEqual(x, y, msg)
class SaveForwardInputsModule(nn.Module):
    # 定义保存前向输入的模块
    def __init__(
        self,
        forward_inputs: Dict[nn.Module, torch.Tensor],  # 前向输入的字典，键为 nn.Module，值为 torch.Tensor
        cast_forward_inputs: bool,  # 是否进行前向输入类型转换的标志
    ) -> None:
        super().__init__()  # 调用父类的初始化方法
        self.l = nn.Linear(100, 100)  # 定义一个线性层
        self.forward_inputs = forward_inputs  # 保存传入的前向输入字典
        self.cast_forward_inputs = cast_forward_inputs  # 保存是否需要进行类型转换的标志

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_inputs[self] = x  # 将当前模块自身及其前向输入记录到字典中
        return self.l(x.to(self.l.weight.dtype) if self.cast_forward_inputs else x)
        # 根据 cast_forward_inputs 决定是否将输入 x 转换成与 self.l.weight 类型相同的张量类型，然后经过线性层返回结果


class SaveForwardInputsModel(nn.Module):
    # 定义保存前向输入的模型
    def __init__(
        self,
        forward_inputs: Dict[nn.Module, torch.Tensor],  # 前向输入的字典，键为 nn.Module，值为 torch.Tensor
        cast_forward_inputs: bool,  # 是否进行前向输入类型转换的标志
    ) -> None:
        super().__init__()  # 调用父类的初始化方法
        self.c1 = SaveForwardInputsModule(forward_inputs, cast_forward_inputs)  # 创建 SaveForwardInputsModule 实例 c1
        self.c2 = SaveForwardInputsModule(forward_inputs, cast_forward_inputs)  # 创建 SaveForwardInputsModule 实例 c2
        self.forward_inputs = forward_inputs  # 保存传入的前向输入字典

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_inputs[self] = x  # 将当前模型自身及其前向输入记录到字典中
        return self.c2(self.c1(x))  # 将输入 x 分别经过 c1 和 c2 模块，并返回 c2 的输出结果


@contextmanager
def _dynamo_dist_per_rank_init(rank, world_size, init_pg=True):
    # 为每个 rank 初始化 Dynamo 分布式环境
    # 避免从 _dynamo.test_case.TestCase 和 MultiProcessTestCase 多重继承，手动实现 dynamo 行为的重置和清除
    torch.cuda.set_device(rank)  # 设置当前 CUDA 设备
    os.environ['MASTER_ADDR'] = 'localhost'  # 设置主节点地址
    os.environ['MASTER_PORT'] = '6789'  # 设置主节点端口
    if init_pg:
        c10d.init_process_group("nccl", rank=rank, world_size=world_size)  # 初始化分布式进程组
    torch._dynamo.reset()  # 重置 Dynamo
    torch._dynamo.utils.counters.clear()  # 清除计数器
    try:
        yield
    finally:
        torch._dynamo.reset()  # 重置 Dynamo
        torch._dynamo.utils.counters.clear()  # 清除计数器
        if init_pg:
            c10d.destroy_process_group()  # 销毁分布式进程组


class DynamoDistributedSingleProcTestCase(torch._dynamo.test_case.TestCase):
    """
    单进程 Dynamo 分布式测试的测试框架，初始化分布式进程组。

    适合简单的测试用例，便于调试。
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()  # 调用父类的类初始化方法
        cls._exit_stack.enter_context(
            patch.dict(
                os.environ,
                {
                    "MASTER_ADDR": "localhost",  # 设置主节点地址
                    "MASTER_PORT": "12355",  # 设置主节点端口
                },
            )
        )
        cls.rank = 0  # 当前 rank 设置为 0
        cls.device = f"cuda:{cls.rank}"  # 设置 CUDA 设备
        cls.device_ids = None if "cuda" in cls.device else [cls.rank]  # 如果不是 CUDA 设备，设备列表为 None，否则为 [0]
        c10d.init_process_group("nccl", rank=cls.rank, world_size=1)  # 初始化分布式进程组

    @classmethod
    def tearDownClass(cls):
        c10d.destroy_process_group()  # 销毁分布式进程组
        super().tearDownClass()  # 调用父类的类清理方法


class DynamoDistributedMultiProcTestCase(MultiProcessTestCase):
    """
    用于多 GPU 运行的测试，每个测试用例都会生成新的进程。

    在需要运行多 GPU 测试时使用。

    注意：MultiProcTestCase 每个测试用例都会生成新的进程，因此速度较慢。
    """
    Prefer MultiThreadedTestCase for most tests. Perhaps use this one
    sparingly for integration tests.
    """
    # 设置测试环境，调用父类的setUp方法初始化
    def setUp(self):
        super().setUp()
        # 启动进程或者线程池
        self._spawn_processes()

    # 清理测试环境，调用父类的tearDown方法
    def tearDown(self):
        super().tearDown()
        try:
            # 尝试删除测试中创建的文件
            os.remove(self.file_name)
        except OSError:
            # 如果文件不存在，捕获OSError异常并忽略
            pass

    # 定义一个属性，返回当前CUDA设备的数量
    @property
    def world_size(self) -> int:
        return torch.cuda.device_count()

    # 类方法，用于执行测试
    @classmethod
    def _run(cls, rank: int, test_name: str, file_name: str, parent_pipe) -> None:
        # The rest is copypasta from MultiProcessTestCase._run
        # 创建当前测试类的实例
        self = cls(test_name)
        # 设置当前进程的等级
        self.rank = rank
        # 设置文件名
        self.file_name = file_name
        # 运行测试，传入测试名和父进程管道
        self.run_test(test_name, parent_pipe)
    # 定义一个多进程连续测试的测试用例类
class MultiProcContinousTest(TestCase):
    # Class variables:
    # number of test processes
    world_size: int = 2
    # rank of the current process
    rank: int = -1  # unset state
    # Rendezvous file
    rdvz_file: Optional[str] = None

    @classmethod
    @abc.abstractmethod
    def backend_str(cls) -> str:
        """
        ProcessGroup backend str.
        To be customized by sub test classes, e.g. "nccl".
        Here we raise error.
        """
        # 抽象方法，返回一个字符串表示进程组的后端类型
        raise NotImplementedError("Please implement backend_str in your test class")

    @classmethod
    def opts(cls, high_priority_stream=False):
        """
        ProcessGroup init options.
        To be customized by sub test classes, e.g. ProcessGroupNCCLOpTest
        Here we return None.
        """
        # 返回进程组初始化选项，由子测试类自定义，这里返回None
        return None

    @classmethod
    def setUpClass(cls):
        """
        Class-scope test fixture. Run once for entire test class, before any test starts.
        Set up the process group.
        """
        # 执行整个测试类的设置，确保rank设置正确且在0到world_size范围内
        super().setUpClass()
        if not 0 <= cls.rank < cls.world_size:
            raise RuntimeError(
                "Rank must be set and in the range of 0 to world_size. "
                f"World size: {cls.world_size} Rank: {cls.rank}"
            )
        if cls.rdvz_file:
            # 根据rdvz_file创建c10d.FileStore对象，用于进程组之间的通信
            store = c10d.FileStore(cls.rdvz_file, cls.world_size)
        else:
            # 如果没有指定rdvz_file，则由torchrun处理rendezvous
            store = None
        opts = cls.opts()
        backend = cls.backend_str()
        print(f"Testing {backend=}")
        # 使用指定的后端和选项初始化进程组
        c10d.init_process_group(
            backend=backend,
            world_size=cls.world_size,
            rank=cls.rank,
            store=store,
            pg_options=opts,
        )
        # 获取默认的进程组
        cls.pg = c10d.distributed_c10d._get_default_group()
        print(f"Rank {cls.rank} setup complete")

    @classmethod
    def tearDownClass(cls):
        """
        Class-scope test fixture. Run once for entire test class, after all tests finish.
        Tear down the process group.
        """
        # 结束整个测试类的清理工作，销毁进程组
        c10d.destroy_process_group()
        super().tearDownClass()
        # 清理rendezvous文件
        if cls.rdvz_file:
            try:
                os.remove(cls.rdvz_file)
            except OSError:
                pass
        print(f"Rank {cls.rank} teardown complete")

    @classmethod
    def run_rank(
        cls,
        rank: int,
        world_size: int,
        rdvz_file: Optional[str] = None,
    ):
        """
        这是每个进程用来运行 `MultiProcContinousTest` 测试的入口点。
        在这个入口点中，我们为测试类设置类变量。
        然后运行所有的测试。

        注意:
        - 此辅助函数仅适用于 `MultiProcContinousTest` 的子类。

        示例:
        - 参考 `test_c10d_ops_nccl.py`。
        """
        # 设置测试类的类变量
        cls.rank = rank
        cls.world_size = world_size
        cls.rdvz_file = rdvz_file
        # 通过 `common_utils` 基础设施启动测试
        run_tests()
```