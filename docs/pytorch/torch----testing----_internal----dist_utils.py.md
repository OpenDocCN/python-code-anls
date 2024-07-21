# `.\pytorch\torch\testing\_internal\dist_utils.py`

```
# 忽略类型检查错误，对于类型检查工具 mypy
# 导入标准库模块
import re
import sys
import time
# 导入 functools 模块中的 partial 和 wraps 函数
from functools import partial, wraps
# 导入 typing 模块中的 Tuple 类型
from typing import Tuple
# 导入 torch 分布式相关模块
import torch.distributed as dist
import torch.distributed.rpc as rpc
# 从 torch.distributed.rpc 中导入 _rref_context_get_debug_info 函数
from torch.distributed.rpc import _rref_context_get_debug_info
# 导入 torch.testing._internal.common_utils 模块中的 FILE_SCHEMA 和 TEST_WITH_TSAN 变量
from torch.testing._internal.common_utils import FILE_SCHEMA, TEST_WITH_TSAN

# 如果分布式环境不可用，则在标准错误输出中显示消息并退出程序
if not dist.is_available():
    print("c10d not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 初始化方法模板，使用 FILE_SCHEMA 和 {file_name} 的格式化字符串
INIT_METHOD_TEMPLATE = FILE_SCHEMA + "{file_name}"

# 分布式初始化函数，用于设置和撤销状态
def dist_init(
    old_test_method=None,
    setup_rpc: bool = True,
    clean_shutdown: bool = True,
    faulty_messages=None,
    messages_to_delay=None,
):
    """
    We use this decorator for setting up and tearing down state since
    MultiProcessTestCase runs each `test*` method in a separate process and
    each process just runs the `test*` method without actually calling
    'setUp' and 'tearDown' methods of unittest.

    Note: pass the string representation of MessageTypes that should be used
    with the faulty agent's send function. By default, all retriable messages
    ("RREF_FORK_REQUEST", "RREF_CHILD_ACCEPT", "RREF_USER_DELETE",
    "CLEANUP_AUTOGRAD_CONTEXT_REQ") will use the faulty send (this default is
    set from faulty_rpc_agent_test_fixture.py).
    """
    # 如果 old_test_method 参数为 None，则返回一个 partial 函数，
    # 部分应用 dist_init 函数，并传递其余参数
    if old_test_method is None:
        return partial(
            dist_init,
            setup_rpc=setup_rpc,
            clean_shutdown=clean_shutdown,
            faulty_messages=faulty_messages,
            messages_to_delay=messages_to_delay,
        )

    # 使用 functools.wraps 将原始测试方法的元数据复制到包装后的函数中
    @wraps(old_test_method)
    # 定义一个新的测试方法，接受任意位置参数和关键字参数
    def new_test_method(self, *arg, **kwargs):
        # 设置 _ignore_rref_leak 为 False，确保在测试中 OwnerRRefs 能够正确删除
        # 在测试中导入 torch 分布式 RPC 的 API
        import torch.distributed.rpc.api as api

        # 设置 _ignore_rref_leak 为 False，以确保 OwnerRRefs 能够正确删除
        api._ignore_rref_leak = False
        # 将 worker_id 设置为当前实例的 rank
        self.worker_id = self.rank
        # 调用 setup_fault_injection 方法，设置故障注入相关参数
        self.setup_fault_injection(faulty_messages, messages_to_delay)

        # 获取当前实例的 rpc_backend_options
        rpc_backend_options = self.rpc_backend_options
        # 如果需要设置 RPC
        if setup_rpc:
            if TEST_WITH_TSAN:
                # 如果是在 TSAN 下运行，增加 RPC 超时时间
                rpc_backend_options.rpc_timeout = rpc.constants.DEFAULT_RPC_TIMEOUT_SEC * 5
                # 设置默认关闭超时时间为 60 秒
                rpc.constants.DEFAULT_SHUTDOWN_TIMEOUT = 60

            # 初始化 RPC，传入实例名称、后端类型、rank、world_size 和 rpc_backend_options
            rpc.init_rpc(
                name="worker%d" % self.rank,
                backend=self.rpc_backend,
                rank=self.rank,
                world_size=self.world_size,
                rpc_backend_options=rpc_backend_options,
            )

        # 调用旧的测试方法 old_test_method，并返回其返回值
        return_value = old_test_method(self, *arg, **kwargs)

        # 如果需要设置 RPC
        if setup_rpc:
            # 关闭 RPC，根据需要使用优雅关闭或立即关闭
            rpc.shutdown(graceful=clean_shutdown)

        # 返回旧测试方法的返回值
        return return_value

    # 返回新测试方法的引用
    return new_test_method
# 定义一个空函数 noop，不执行任何操作
def noop() -> None:
    pass


# 循环直到对给定 rank 的 RPC 失败为止。用于在单元测试中指示节点失败。
def wait_until_node_failure(rank: int, expected_error_regex: str = ".*") -> str:
    """
    Loops until an RPC to the given rank fails. This is used to
    indicate that the node has failed in unit tests.
    Args:
    rank (int): Rank of the node expected to fail
    expected_error_regex (optional, str): Regex of exception message expected. Useful to ensure a specific failure
    occurs, not just any.
    """
    while True:
        try:
            # 同步执行名为 worker{rank} 的 RPC，调用 noop 函数，没有参数
            rpc.rpc_sync(f"worker{rank}", noop, args=())
            # 等待 0.1 秒
            time.sleep(0.1)
        except Exception as e:
            # 如果捕获到异常，检查异常消息是否符合预期的正则表达式
            if re.search(pattern=expected_error_regex, string=str(e)):
                return str(e)


# 循环直到等待超时，确保挂起的 futures 和用户已经刷新
def wait_until_pending_futures_and_users_flushed(timeout: int = 20) -> None:
    """
    The RRef protocol holds forkIds of rrefs in a map until those forks are
    confirmed by the owner. The message confirming the fork may arrive after
    our tests check whether this map is empty, which leads to failures and
    flaky tests. to_here also does not guarantee that we have finished
    processind the owner's confirmation message for the RRef. This function
    loops until the map is empty, which means the messages have been received
    as processed. Call this function before asserting the map returned by
    _get_debug_info is empty.
    """
    start = time.time()
    while True:
        # 获取当前 RRef 上下文的调试信息
        debug_info = _rref_context_get_debug_info()
        # 获取挂起 futures 和用户的数量，并转换为整数
        num_pending_futures = int(debug_info["num_pending_futures"])
        num_pending_users = int(debug_info["num_pending_users"])
        # 如果挂起的 futures 和用户数量均为零，退出循环
        if num_pending_futures == 0 and num_pending_users == 0:
            break
        # 等待 0.1 秒
        time.sleep(0.1)
        # 如果等待超时，则抛出 ValueError 异常
        if time.time() - start > timeout:
            raise ValueError(
                f"Timed out waiting to flush pending futures and users, "
                f"had {num_pending_futures} pending futures and {num_pending_users} pending users"
            )


# 获取当前节点上的 OwnerRRefs 和 forks 数量的元组
def get_num_owners_and_forks() -> Tuple[str, str]:
    """
    Retrieves number of OwnerRRefs and forks on this node from
    _rref_context_get_debug_info.
    """
    # 获取当前 RRef 上下文的调试信息
    rref_dbg_info = _rref_context_get_debug_info()
    # 获取 OwnerRRefs 和 forks 的数量
    num_owners = rref_dbg_info["num_owner_rrefs"]
    num_forks = rref_dbg_info["num_forks"]
    return num_owners, num_forks


# 等待直到 rank 节点上的 num_forks 和 num_owners 存在，确保在测试中正确删除 RRefs
def wait_until_owners_and_forks_on_rank(
    num_owners: int, num_forks: int, rank: int, timeout: int = 20
) -> None:
    """
    Waits until timeout for num_forks and num_owners to exist on the rank. Used
    to ensure proper deletion of RRefs in tests.
    """
    start = time.time()
    # 进入无限循环，等待条件满足或超时
    while True:
        # 使用远程过程调用同步获取当前排名的所有者数和分支数
        num_owners_on_rank, num_forks_on_rank = rpc.rpc_sync(
            worker_name(rank), get_num_owners_and_forks, args=(), timeout=5
        )
        # 将获取的所有者数和分支数转换为整数类型
        num_owners_on_rank = int(num_owners_on_rank)
        num_forks_on_rank = int(num_forks_on_rank)
        # 如果当前所有者数和分支数与目标相等，则退出循环
        if num_owners_on_rank == num_owners and num_forks_on_rank == num_forks:
            return
        # 休眠1秒，等待下一次检查
        time.sleep(1)
        # 如果超过设置的超时时间，抛出超时异常，并显示详细信息
        if time.time() - start > timeout:
            raise ValueError(
                f"Timed out waiting {timeout} sec for {num_owners} owners and {num_forks} forks on rank,"
                f" had {num_owners_on_rank} owners and {num_forks_on_rank} forks"
            )
# 初始化 PyTorch 分布式进程组，用于使用 `dist.barrier` 进行测试
def initialize_pg(init_method, rank: int, world_size: int) -> None:
    # 检查当前是否已经初始化了分布式环境
    if not dist.is_initialized():
        # 使用指定的初始化方法和参数初始化分布式进程组
        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )


# 根据给定的进程 rank 返回对应的 worker 名称
def worker_name(rank: int) -> str:
    return f"worker{rank}"


# 从给定的 function_events 中查找并返回首个匹配 partial_event_name 的事件
def get_function_event(function_events, partial_event_name):
    """
    Returns the first event that matches partial_event_name in the provided
    function_events. These function_events should be the output of
    torch.autograd.profiler.function_events().

    Args:
    function_events: function_events returned by the profiler.
    partial_event_name (str): partial key that the event was profiled with.
    """
    # 使用列表推导式从 function_events 中筛选出名称包含 partial_event_name 的第一个事件
    event = [event for event in function_events if partial_event_name in event.name][0]  # noqa: RUF015
    return event
```