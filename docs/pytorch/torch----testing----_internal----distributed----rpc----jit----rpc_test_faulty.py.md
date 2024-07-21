# `.\pytorch\torch\testing\_internal\distributed\rpc\jit\rpc_test_faulty.py`

```py
# 忽略类型检查错误
# 导入必要的类型
from typing import Dict, Tuple

# 导入 PyTorch 库
import torch
import torch.distributed.rpc as rpc
from torch import Tensor
from torch.distributed.rpc import RRef
from torch.testing._internal.dist_utils import (
    dist_init,         # 导入初始化分布式环境的函数
    worker_name,       # 导入获取当前工作节点名称的函数
    wait_until_pending_futures_and_users_flushed  # 导入等待未决期任务和用户刷新的函数
)
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture  # 导入用于测试的 RPC 代理测试固件类
)


@torch.jit.script
# 定义一个接受两个位置参数和两个关键字参数的 Torch 脚本函数
def two_args_two_kwargs(
    first_arg,                                      # 第一个位置参数
    second_arg,                                     # 第二个位置参数
    first_kwarg=torch.tensor([3, 3]),                # 第一个关键字参数，默认为 [3, 3]
    second_kwarg=torch.tensor([4, 4]),               # 第二个关键字参数，默认为 [4, 4]
):
    return first_arg + second_arg + first_kwarg + second_kwarg  # 返回所有参数的求和结果


@torch.jit.script
# 定义一个 Torch 脚本函数，执行异步 RPC 调用
def script_rpc_async_call(
    dst_worker_name: str,                           # 目标工作节点的名称
    args: Tuple[Tensor, Tensor],                    # 位置参数，包含两个 Tensor 对象的元组
    kwargs: Dict[str, Tensor]                       # 关键字参数字典，包含 Tensor 对象
):
    fut = rpc.rpc_async(dst_worker_name, two_args_two_kwargs, args, kwargs)  # 发起异步 RPC 调用
    ret = fut.wait()                                # 等待调用完成并获取结果
    return ret                                       # 返回调用结果


@torch.jit.script
# 定义一个 Torch 脚本函数，执行带超时的异步 RPC 调用
def rpc_async_call_with_timeout(
    dst_worker_name: str,                           # 目标工作节点的名称
    args: Tuple[Tensor, Tensor],                    # 位置参数，包含两个 Tensor 对象的元组
    kwargs: Dict[str, Tensor],                      # 关键字参数字典，包含 Tensor 对象
    timeout: float                                  # 超时时间
):
    fut = rpc.rpc_async(dst_worker_name, two_args_two_kwargs, args, kwargs, timeout)  # 发起带超时的异步 RPC 调用
    ret = fut.wait()                                # 等待调用完成并获取结果
    return ret                                       # 返回调用结果


@torch.jit.script
# 定义一个 Torch 脚本函数，执行带超时并返回 Future 对象的异步 RPC 调用
def rpc_async_call_with_timeout_future_ret(
    dst_worker_name: str,                           # 目标工作节点的名称
    args: Tuple[Tensor, Tensor],                    # 位置参数，包含两个 Tensor 对象的元组
    kwargs: Dict[str, Tensor],                      # 关键字参数字典，包含 Tensor 对象
    timeout: float                                  # 超时时间
):
    fut = rpc.rpc_async(dst_worker_name, two_args_two_kwargs, args, kwargs, timeout)  # 发起带超时的异步 RPC 调用
    return fut                                       # 直接返回 Future 对象


@torch.jit.script
# 定义一个 Torch 脚本函数，执行返回 Future 对象的异步 RPC 调用
def rpc_async_call_future_ret(
    dst_worker_name: str,                           # 目标工作节点的名称
    args: Tuple[Tensor, Tensor],                    # 位置参数，包含两个 Tensor 对象的元组
    kwargs: Dict[str, Tensor]                       # 关键字参数字典，包含 Tensor 对象
):
    fut = rpc.rpc_async(dst_worker_name, two_args_two_kwargs, args, kwargs)  # 发起异步 RPC 调用
    return fut                                       # 直接返回 Future 对象


@torch.jit.script
# 定义一个 Torch 脚本函数，将 RRef 对象同步到本地 Tensor
def rref_to_here(rref_var: RRef[Tensor]) -> Tensor:
    return rref_var.to_here()                       # 将 RRef 对象同步到本地 Tensor 并返回


@torch.jit.script
# 定义一个 Torch 脚本函数，带超时将 RRef 对象同步到本地 Tensor
def rref_to_here_with_timeout(rref_var: RRef[Tensor], timeout: float) -> Tensor:
    return rref_var.to_here(timeout)                # 带超时将 RRef 对象同步到本地 Tensor 并返回


@torch.jit.script
# 定义一个 Torch 脚本函数，执行异步 RPC 调用并传递 RRef 对象作为参数
def rpc_async_with_rref_arg(dst_worker_name: str, args: Tuple[RRef[Tensor]]) -> Tensor:
    fut = rpc.rpc_async(dst_worker_name, rref_to_here, args)  # 发起异步 RPC 调用，传递 RRef 对象
    ret = fut.wait()                                # 等待调用完成并获取结果
    return ret                                       # 返回调用结果


# 定义一个测试类，测试在故障代理测试固件下的 JIT 中的异步 RPC 调用
class JitFaultyAgentRpcTest(RpcAgentTestFixture):
    """
    Run tests for rpc_async in JIT under the faulty agent test fixture to test
    arbitrary timeouts.
    """
    @dist_init(faulty_messages=[], messages_to_delay={"SCRIPT_CALL": 1.5})
    # 初始化分布式环境，设定故障消息为空，SCRIPT_CALL 延迟 1.5 秒
    def test_timeout_in_torchscript_function(self):
        # 在 torchscript 函数中调用 rpc_async + fut.wait()，确保超时会被触发。
        if self.rank != 0:
            return

        dst_worker_name = worker_name((self.rank + 1) % self.world_size)

        args = (torch.tensor([1, 1]), torch.tensor([2, 2]))
        kwargs = {
            "first_kwarg": torch.tensor([2, 2]),
            "second_kwarg": torch.tensor([3, 3]),
        }
        expected_error = self.get_timeout_error_regex()
        # 确保在覆盖默认超时并且 RPC 执行时间较长时会出现超时。
        with self.assertRaisesRegex(RuntimeError, expected_error):
            rpc_async_call_with_timeout(dst_worker_name, args, kwargs, 0.5)

        # 确保如果没有指定超时但默认超时时间小于 RPC 执行时间时会超时。
        rpc._set_rpc_timeout(0.001)
        with self.assertRaisesRegex(RuntimeError, expected_error):
            script_rpc_async_call(
                dst_worker_name, args, kwargs
            )

        # 确保如果指定了零超时时间，则会完整运行。
        ret = rpc_async_call_with_timeout(dst_worker_name, args, kwargs, 0)
        self.assertEqual(ret, torch.tensor([8, 8]))
        # 为了清理关闭，重置默认的 RPC 超时时间。
        rpc._set_rpc_timeout(rpc.constants.DEFAULT_RPC_TIMEOUT_SEC)

    @dist_init(faulty_messages=[], messages_to_delay={"SCRIPT_CALL": 1.5})
    def test_timeout_in_python(self):
        # 确保如果从 torchscript 函数中调用 rpc_async，但在 python 中等待 future，会触发超时。
        if self.rank != 0:
            return

        dst_worker_name = worker_name((self.rank + 1) % self.world_size)
        args = (torch.tensor([1, 1]), torch.tensor([2, 2]))
        kwargs = {
            "first_kwarg": torch.tensor([2, 2]),
            "second_kwarg": torch.tensor([3, 3]),
        }
        expected_error = self.get_timeout_error_regex()

        fut = rpc_async_call_with_timeout_future_ret(dst_worker_name, args, kwargs, 0.5)
        with self.assertRaisesRegex(RuntimeError, expected_error):
            fut.wait()

        # 确保如果没有指定超时时间但默认超时时间小于 RPC 执行时间时会超时。
        rpc._set_rpc_timeout(0.001)
        fut = rpc_async_call_future_ret(dst_worker_name, args, kwargs)
        with self.assertRaisesRegex(RuntimeError, expected_error):
            fut.wait()

        # 确保如果指定了零超时时间，则会完整运行。
        fut = rpc_async_call_with_timeout_future_ret(dst_worker_name, args, kwargs, 0)
        result = fut.wait()
        self.assertEqual(result, torch.tensor([8, 8]))
        # 为了清理关闭，重置默认的 RPC 超时时间。
        rpc._set_rpc_timeout(rpc.constants.DEFAULT_RPC_TIMEOUT_SEC)

    @dist_init(faulty_messages=["SCRIPT_REMOTE_CALL"])
    def test_remote_timeout_to_here_in_jit(self):
        # 测试在 JIT 中调用 to_here() 是否会在 rpc.remote 失败时引发超时错误。
        if self.rank != 0:
            return
        # 确定目标的排名
        dst_rank = (self.rank + 1) % self.world_size
        # 根据目标排名获取工作节点名称
        dst_worker = f"worker{dst_rank}"
        # 远程调用 dst_worker 上的 torch.add 函数，传入参数 torch.tensor(1), torch.tensor(1)
        rref = rpc.remote(
            dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1))
        )
        # 等待直到所有待处理的 futures 和用户被清空
        wait_until_pending_futures_and_users_flushed()
        # 在 ScriptFunction 中调用 to_here() 并确保引发异常
        with self.assertRaisesRegex(RuntimeError, "RRef creation"):
            rref_to_here(rref)

    @dist_init(faulty_messages=[], messages_to_delay={"SCRIPT_RREF_FETCH_CALL": 1})
    def test_rref_to_here_timeout_in_jit(self):
        if self.rank != 0:
            return

        dst_rank = (self.rank + 1) % self.world_size
        dst_worker = f"worker{dst_rank}"
        # 远程调用 dst_worker 上的 torch.add 函数，传入参数 torch.tensor(1), torch.tensor(1)
        rref = rpc.remote(
            dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1))
        )
        # 获取超时错误的正则表达式
        expected_error = self.get_timeout_error_regex()
        # 确保在超时情况下调用 rref_to_here_with_timeout 并引发异常
        with self.assertRaisesRegex(RuntimeError, expected_error):
            rref_to_here_with_timeout(rref, 0.01)

        # 在长超时情况下调用 rref_to_here_with_timeout
        rref_to_here_with_timeout(rref, 100)

    @dist_init(faulty_messages=["SCRIPT_REMOTE_CALL"])
    def test_rref_timeout_pickle_in_jit(self):
        if self.rank != 0:
            return
        dst_rank = (self.rank + 1) % self.world_size
        dst_worker = f"worker{dst_rank}"
        # 远程调用 dst_worker 上的 torch.add 函数，传入参数 torch.tensor(1), torch.tensor(1)
        rref = rpc.remote(
            dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1))
        )
        # 等待直到所有待处理的 futures 和用户被清空
        wait_until_pending_futures_and_users_flushed()
        # 在 JIT 中调用 RPC，使用 RRef 参数，确保在序列化期间引发超时异常
        with self.assertRaisesRegex(RuntimeError, "RRef creation"):
            rpc_async_with_rref_arg(dst_worker, (rref, ))

    @dist_init(faulty_messages=["SCRIPT_REMOTE_CALL"])
    def test_rref_timeout_pickle_script_func(self):
        # 类似于上述测试，但调用具有脚本函数的 Python RPC。
        if self.rank != 0:
            return
        dst_rank = (self.rank + 1) % self.world_size
        dst_worker = f"worker{dst_rank}"
        # 远程调用 dst_worker 上的 torch.add 函数，传入参数 torch.tensor(1), torch.tensor(1)
        rref = rpc.remote(
            dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1))
        )
        # 等待直到所有待处理的 futures 和用户被清空
        wait_until_pending_futures_and_users_flushed()
        # 使用接受 RRef 的脚本函数调用 RPC，确保在序列化期间引发超时异常
        with self.assertRaisesRegex(RuntimeError, "RRef creation"):
            rpc.rpc_sync(dst_worker, rref_to_here, args=(rref, ))
```