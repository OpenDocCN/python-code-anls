# `.\pytorch\torch\testing\_internal\distributed\rpc\faulty_agent_rpc_test.py`

```py
# 忽略类型检查错误
# 导入PyTorch库
import torch
# 导入时间模块
import time
# 导入分布式RPC模块
import torch.distributed.rpc as rpc
# 从torch.distributed.rpc.api中导入_delete_all_user_and_unforked_owner_rrefs函数
from torch.distributed.rpc.api import _delete_all_user_and_unforked_owner_rrefs
# 从torch.testing._internal.dist_utils中导入多个函数和变量
from torch.testing._internal.dist_utils import (
    dist_init,
    wait_until_pending_futures_and_users_flushed,
    wait_until_owners_and_forks_on_rank,
    worker_name,
)
# 从torch.testing._internal.distributed.rpc.rpc_agent_test_fixture中导入RpcAgentTestFixture类
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,
)

# 定义一个自定义的休眠函数，参数默认为1秒
def my_sleep_func(seconds=1):
    # 使用time模块的sleep函数实现休眠
    time.sleep(seconds)
    # 返回两个张量的乘积
    return torch.mul(torch.tensor(1), torch.tensor(1))

# 使用Torch脚本装饰器定义一个脚本函数，用于对输入张量进行加法操作
@torch.jit.script
def my_script_func(tensor):
    # 返回输入张量与自身的加法结果
    return torch.add(tensor, tensor)

# 定义一个函数，将远程引用和值相加并返回结果
def add_rref_to_value(rref, value):
    # 将远程引用传输到本地并与给定值相加，返回相加后的结果
    return rref.to_here() + value

# 定义一个测试类，用于测试故障情况下的RPC功能
class FaultyAgentRpcTest(RpcAgentTestFixture):

    # 初始化分布式环境，并测试检查失败消息的功能
    @dist_init(messages_to_delay={})
    def test_check_failed_messages(self):
        # 如果当前进程的排名为0
        if self.rank == 0:
            # 获取下一个和下下个工作节点的名称
            dst_worker_b = worker_name((self.rank + 1) % self.world_size)
            dst_worker_c = worker_name((self.rank + 2) % self.world_size)

            # Worker0向Worker1发送RPC，并在那里创建一个远程引用
            rref = rpc.remote(dst_worker_b, torch.add, args=(torch.ones(2, 2), torch.ones(2, 2)))
            # Worker0向Worker2发送带有远程引用作为参数的RPC
            rpc.remote(dst_worker_c, add_rref_to_value, args=(rref, torch.ones(2, 2)))
            # 检查输出是否符合预期
            self.assertEqual(rref.to_here(), torch.add(torch.ones(2, 2), torch.ones(2, 2)))
        # 明确删除所有用户远程引用和未fork的所有者远程引用
        _delete_all_user_and_unforked_owner_rrefs()

    # 初始化分布式环境，并验证后端选项设置
    @dist_init
    def test_verify_backend_options(self):
        # 断言当前RPC后端为FAULTY_TENSORPIPE
        self.assertEqual(self.rpc_backend, rpc.backend_registry.BackendType.FAULTY_TENSORPIPE)
        # 断言后端选项中的工作线程数为8
        self.assertEqual(self.rpc_backend_options.num_worker_threads, 8)
        # 断言后端选项中的发送失败数为3
        self.assertEqual(self.rpc_backend_options.num_fail_sends, 3)
        # 断言后端选项中失败消息列表的长度为4
        self.assertEqual(len(self.rpc_backend_options.messages_to_fail), 4)
        # 断言后端选项中延迟消息列表的长度为2
        self.assertEqual(len(self.rpc_backend_options.messages_to_delay), 2)
        # 断言后端选项中的RPC超时时间为默认值
        self.assertEqual(self.rpc_backend_options.rpc_timeout, rpc.constants.DEFAULT_RPC_TIMEOUT_SEC)

    # 初始化分布式环境，并测试自定义故障消息
    @dist_init(faulty_messages=["RREF_FORK_REQUEST", "RREF_CHILD_ACCEPT"])
    def test_custom_faulty_messages(self):
        # 断言设置的故障消息集合与预期集合相匹配
        self.assertEqual(
            {"RREF_FORK_REQUEST", "RREF_CHILD_ACCEPT"},
            set(self.rpc_backend_options.messages_to_fail),
        )

    # 初始化分布式环境，并测试无故障消息情况
    @dist_init(faulty_messages=[])
    def test_no_faulty_messages(self):
        # 断言后端选项中的失败消息列表长度为0
        self.assertEqual(len(self.rpc_backend_options.messages_to_fail), 0)

    # 初始化分布式环境，并测试自定义延迟消息
    @dist_init(messages_to_delay={"SCRIPT_CALL": 1.5})
    def test_custom_messages_to_delay(self):
        # 断言后端选项中的延迟消息字典与预期字典相匹配
        self.assertEqual(self.rpc_backend_options.messages_to_delay, {"SCRIPT_CALL": 1.5})
    # 测试远程消息丢弃的情况，使用 pickle 序列化
    def _test_remote_message_dropped_pickle(self, dst=None):
        # 如果当前进程的排名不是0，则直接返回
        if self.rank != 0:
            return
        # 确定目标进程的排名，默认为当前进程排名加1，取余世界大小
        dst_rank = dst if dst is not None else (self.rank + 1) % self.world_size
        # 构建目标进程的名称
        dst_worker = f"worker{dst_rank}"
        # 由于我们同步失败 python_remote_call 消息，因此与此远程调用对应的 future
        # 在此函数返回时将标记为错误。
        rref = rpc.remote(dst_worker, my_sleep_func, args=(1,))
        # 调用以确保运行挂起的回调
        wait_until_pending_futures_and_users_flushed()
        # 尝试对 RRef 进行序列化应该引发错误，指示 rpc.remote 超时。
        with self.assertRaisesRegex(RuntimeError, "RRef creation"):
            rref._serialize()
        # 测试通过 RPC 使用 RRef 作为参数（这将进行 fork 操作）是否导致相同的错误
        with self.assertRaisesRegex(RuntimeError, "RRef creation"):
            rpc.rpc_async(dst_worker, add_rref_to_value, args=(rref, 1))

    # 使用 dist_init 装饰器初始化测试函数，设置故障消息类型为 PYTHON_REMOTE_CALL
    @dist_init(faulty_messages=["PYTHON_REMOTE_CALL"])
    def test_remote_message_dropped_pickle(self):
        # 调用 _test_remote_message_dropped_pickle 函数进行测试
        self._test_remote_message_dropped_pickle()

    # 使用 dist_init 装饰器初始化测试函数，设置故障消息类型为 PYTHON_REMOTE_CALL
    @dist_init(faulty_messages=["PYTHON_REMOTE_CALL"])
    def test_remote_message_dropped_pickle_to_self(self):
        # 调用 _test_remote_message_dropped_pickle 函数进行测试，并指定目标为当前进程的排名
        self._test_remote_message_dropped_pickle(self.rank)


    # 测试远程消息超时的情况
    def _test_remote_message_dropped_timeout(self, func, args, dst=None):
        # 如果当前进程的排名不是0，则直接返回
        if self.rank != 0:
            return

        # 测试 rpc.remote() 消息完全丢弃的情况
        dst_rank = dst if dst is not None else (self.rank + 1) % self.world_size
        # 构建目标进程的名称
        dst_worker = f"worker{dst_rank}"
        # 由于我们同步失败 python_remote_call 消息，因此与此远程调用对应的 future
        # 在此函数返回时将标记为错误。
        rref = rpc.remote(dst_worker, func, args=args)
        # 调用以确保运行挂起的回调
        wait_until_pending_futures_and_users_flushed()
        # 使用 to_here 方法应该引发错误
        with self.assertRaisesRegex(RuntimeError, "RRef creation"):
            rref.to_here()
        # 注意：在关闭期间，日志将显示 "Could not find OwnerRRef..."，在拥有节点上，这是预期的，
        # 因为 OwnerRRef 从未成功创建。因此，delAllUsers 将按预期工作。

    # 使用 dist_init 装饰器初始化测试函数，设置故障消息类型为 SCRIPT_REMOTE_CALL
    @dist_init(faulty_messages=["SCRIPT_REMOTE_CALL"])
    def test_builtin_remote_message_dropped_timeout(self):
        # 测试内置函数远程消息超时的情况，使用 torch.add 函数
        func = torch.add
        args = (torch.tensor(1), torch.tensor(1))
        self._test_remote_message_dropped_timeout(func, args)

    # 使用 dist_init 装饰器初始化测试函数，设置故障消息类型为 SCRIPT_REMOTE_CALL
    @dist_init(faulty_messages=["SCRIPT_REMOTE_CALL"])
    def test_builtin_remote_message_dropped_timeout_to_self(self):
        # 测试内置函数远程消息超时的情况，使用 torch.add 函数，并指定目标为当前进程的排名
        func = torch.add
        args = (torch.tensor(1), torch.tensor(1))
        self._test_remote_message_dropped_timeout(func, args, dst=0)

    # 使用 dist_init 装饰器初始化测试函数，设置故障消息类型为 PYTHON_REMOTE_CALL
    # 定义测试函数，用于测试用户定义的远程消息在超时后的行为
    def test_udf_remote_message_dropped_timeout(self):
        # 将函数设置为 my_sleep_func
        func = my_sleep_func
        # 设置函数参数为 (2,)
        args = (2,)
        # 调用 _test_remote_message_dropped_timeout 方法进行测试
        self._test_remote_message_dropped_timeout(func, args)

    # 在初始化分布式环境时设置故障消息为 ["PYTHON_REMOTE_CALL"] 的测试函数
    @dist_init(faulty_messages=["PYTHON_REMOTE_CALL"])
    def test_udf_remote_message_dropped_timeout_to_self(self):
        # 将函数设置为 my_sleep_func
        func = my_sleep_func
        # 设置函数参数为 (2,)
        args = (2,)
        # 调用 _test_remote_message_dropped_timeout 方法进行测试，指定目标为 0
        self._test_remote_message_dropped_timeout(func, args, dst=0)

    # 私有方法，用于测试远程消息延迟超时的情况
    def _test_remote_message_delay_timeout(self, func, args, dst=None):
        # 如果当前进程不是 rank 为 0 的进程，则直接返回
        if self.rank != 0:
            return
        # 测试远程消息在消息所有者最终处理之前超时的情况
        dst_rank = dst if dst is not None else (self.rank + 1) % self.world_size
        dst_worker = f"worker{dst_rank}"
        # 设置 10 毫秒超时
        rref = rpc.remote(dst_worker, func, args=args, timeout=0.001)
        # 预期远程创建的 Future 应该超时
        expected_error = self.get_timeout_error_regex()
        with self.assertRaisesRegex(RuntimeError, expected_error):
            rref._get_future().wait()

        # 调用以确保运行挂起的回调
        wait_until_pending_futures_and_users_flushed()
        # 现在 to_here() 应该会捕获到 rpc.remote() 创建失败的情况
        with self.assertRaisesRegex(RuntimeError, "RRef creation"):
            rref.to_here()

        # 测试 rpc.remote() 超时，但 to_here() 在此之前已开始阻塞的情况
        # 注意：仅在非发送给自己的情况下测试此处，因为 to_here() 调用了 localValue()，
        # 这不会发送 RPC，因此没有超时。这可以通过允许 future.wait() 接受可选的超时来支持。
        if dst_rank != self.rank:
            slow_rref = rpc.remote(dst_worker, func, args=args, timeout=2)

            with self.assertRaisesRegex(RuntimeError, expected_error):
                # to_here() 应该会引发超时错误，因为它不知道 rpc.remote() 的状态。
                slow_rref.to_here(0.001)
        # 注意：如果继续关闭，UserRRef 将发送 RRefUserDelete，但这可能是一个 noop，
        # 因为它可能尚不存在于所有者那里。稍后，所有者可以处理 RRef 的创建并等待删除消息，
        # 从而导致超时。因此，在发送 RRefUserDeletes 之前，我们等待所有者和分叉在 rank 上的确认。
        if dst_rank != self.rank:
            wait_until_owners_and_forks_on_rank(2, 2, rank=dst_rank)

    # 在初始化分布式环境时设置故障消息为空列表，设置延迟的消息为 {"PYTHON_REMOTE_CALL": 2} 的测试函数
    @dist_init(faulty_messages=[], messages_to_delay={"PYTHON_REMOTE_CALL": 2})
    def test_udf_remote_message_delay_timeout(self):
        # 将函数设置为 my_sleep_func
        func = my_sleep_func
        # 设置函数参数为 (2,)
        args = (2,)
        # 调用 _test_remote_message_delay_timeout 方法进行测试
        self._test_remote_message_delay_timeout(func, args)
    # 使用 @dist_init 装饰器初始化测试函数，设置故障消息为空列表，并将指定消息延迟次数设定为 {"PYTHON_REMOTE_CALL": 2}
    @dist_init(faulty_messages=[], messages_to_delay={"PYTHON_REMOTE_CALL": 2})
    # 定义测试函数 test_udf_remote_message_delay_timeout_to_self，测试用户定义函数的远程消息延迟超时情况
    def test_udf_remote_message_delay_timeout_to_self(self):
        # 将函数 my_sleep_func 赋给 func 变量
        func = my_sleep_func
        # 设置参数 args 为元组 (1,)
        args = (1,)
        # 调用 _test_remote_message_delay_timeout 方法测试远程消息延迟超时，目标为进程 0
        self._test_remote_message_delay_timeout(func, args, dst=0)

    # 使用 @dist_init 装饰器初始化测试函数，设置故障消息为空列表，设置脚本远程调用消息延迟次数为 {"SCRIPT_REMOTE_CALL": 2, "SCRIPT_RREF_FETCH_CALL": 1}
    @dist_init(
        faulty_messages=[],
        messages_to_delay={"SCRIPT_REMOTE_CALL": 2, "SCRIPT_RREF_FETCH_CALL": 1},
    )
    # 定义测试函数 test_remote_message_builtin_delay_timeout，测试内置函数远程消息的延迟超时情况
    def test_remote_message_builtin_delay_timeout(self):
        # 将 torch.add 函数赋给 func 变量
        func = torch.add
        # 设置参数 args 为 torch.tensor(1), torch.tensor(1) 组成的元组
        args = (torch.tensor(1), torch.tensor(1))
        # 调用 _test_remote_message_delay_timeout 方法测试远程消息延迟超时
        self._test_remote_message_delay_timeout(func, args)

    # 使用 @dist_init 装饰器初始化测试函数，设置故障消息为空列表，设置脚本远程调用消息延迟次数为 {"SCRIPT_REMOTE_CALL": 2, "SCRIPT_RREF_FETCH_CALL": 1}
    @dist_init(
        faulty_messages=[],
        messages_to_delay={"SCRIPT_REMOTE_CALL": 2, "SCRIPT_RREF_FETCH_CALL": 1},
    )
    # 定义测试函数 test_remote_message_builtin_delay_timeout_to_self，测试内置函数远程消息的延迟超时情况，目标为进程 0
    def test_remote_message_builtin_delay_timeout_to_self(self):
        # 将 torch.add 函数赋给 func 变量
        func = torch.add
        # 设置参数 args 为 torch.tensor(1), torch.tensor(1) 组成的元组
        args = (torch.tensor(1), torch.tensor(1))
        # 调用 _test_remote_message_delay_timeout 方法测试远程消息延迟超时，目标为进程 0
        self._test_remote_message_delay_timeout(func, args, dst=0)

    # 使用 @dist_init 装饰器初始化测试函数，设置故障消息为空列表，设置脚本远程调用消息延迟次数为 {"SCRIPT_REMOTE_CALL": 2, "SCRIPT_RREF_FETCH_CALL": 1}
    @dist_init(
        faulty_messages=[],
        messages_to_delay={"SCRIPT_REMOTE_CALL": 2, "SCRIPT_RREF_FETCH_CALL": 1},
    )
    # 定义测试函数 test_remote_message_script_delay_timeout，测试脚本函数远程消息的延迟超时情况
    def test_remote_message_script_delay_timeout(self):
        # 将 my_script_func 函数赋给 func 变量
        func = my_script_func
        # 设置参数 args 为 torch.tensor(1) 组成的元组
        args = (torch.tensor(1),)
        # 调用 _test_remote_message_delay_timeout 方法测试远程消息延迟超时
        self._test_remote_message_delay_timeout(func, args)

    # 使用 @dist_init 装饰器初始化测试函数，设置故障消息为空列表，设置脚本远程调用消息延迟次数为 {"SCRIPT_REMOTE_CALL": 2, "SCRIPT_RREF_FETCH_CALL": 1}
    @dist_init(
        faulty_messages=[],
        messages_to_delay={"SCRIPT_REMOTE_CALL": 2, "SCRIPT_RREF_FETCH_CALL": 1},
    )
    # 定义测试函数 test_remote_message_script_delay_timeout_to_self，测试脚本函数远程消息的延迟超时情况，目标为进程 0
    def test_remote_message_script_delay_timeout_to_self(self):
        # 将 my_script_func 函数赋给 func 变量
        func = my_script_func
        # 设置参数 args 为 torch.tensor(1) 组成的元组
        args = (torch.tensor(1),)
        # 调用 _test_remote_message_delay_timeout 方法测试远程消息延迟超时，目标为进程 0
        self._test_remote_message_delay_timeout(func, args, dst=0)

    # 使用 @dist_init 装饰器初始化测试函数，设置故障消息为空列表，不设置延迟任何消息
    @dist_init(faulty_messages=[])
    # 定义测试函数 test_rref_to_here_timeout，测试远程引用（RRef）到当前的超时情况
    def test_rref_to_here_timeout(self):
        # 如果当前进程不是进程 0，则直接返回
        if self.rank != 0:
            return

        # 计算目标进程的排名号，确保循环使用，避免超过世界大小
        dst_rank = (self.rank + 1) % self.world_size
        # 构建目标工作节点的名称
        dst_worker = f"worker{dst_rank}"
        # 在目标工作节点上执行远程调用，调用 torch.add 函数，参数为 torch.tensor(1), torch.tensor(1)
        rref = rpc.remote(
            dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1))
        )
        # 获取预期的超时错误的正则表达式
        expected_error = self.get_timeout_error_regex()
        # 使用断言确保在指定时间内抛出 RuntimeError 异常，异常信息匹配预期的错误正则表达式
        with self.assertRaisesRegex(RuntimeError, expected_error):
            rref.to_here(0.01)

        # 等待远程引用（RRef）到达当前节点
        rref.to_here()

    # 使用 @dist_init 装饰器初始化测试函数，设置故障消息为空列表
    @dist_init(faulty_messages=[])
    # 定义测试函数，用于测试 RPC 超时机制
    def test_rpc_builtin_timeout(self):
        # 计算下一个工作进程的排名
        next_rank = (self.rank + 1) % self.world_size
        # 获取下一个工作进程的名称
        dst_worker = worker_name(next_rank)
        # 获取预期的超时错误正则表达式
        expected_error = self.get_timeout_error_regex()

        # 测试同步 RPC 调用，期望抛出指定的 RuntimeError 异常
        with self.assertRaisesRegex(RuntimeError, expected_error):
            rpc.rpc_sync(
                dst_worker,
                torch.add,
                args=(torch.tensor(1), torch.tensor(1)),
                timeout=1,
            )

        # 发起异步 RPC 调用并测试，期望抛出指定的 RuntimeError 异常
        fut = rpc.rpc_async(
            dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1)), timeout=1
        )
        with self.assertRaisesRegex(RuntimeError, expected_error):
            fut.wait()

        # 确保当前设置的默认超时时间足够大，使得带有延迟的 RPC 调用仍然能够完成
        fut = rpc.rpc_async(
            dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1))
        )
        fut.wait()

        # 设置一个新的默认超时时间，并确保未指定超时时会发生超时
        rpc._set_rpc_timeout(0.001)
        fut = rpc.rpc_async(
            dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1))
        )
        with self.assertRaisesRegex(RuntimeError, expected_error):
            fut.wait()

        # 设置超时时间为 0，并确保能够正常完成 RPC 调用
        fut = rpc.rpc_async(
            dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1)), timeout=0
        )
        fut.wait()

        # 恢复默认的 RPC 超时时间以便进行清理关闭
        rpc._set_rpc_timeout(rpc.constants.DEFAULT_RPC_TIMEOUT_SEC)

    @dist_init(faulty_messages=[], messages_to_delay={"SCRIPT_CALL": 1.5})


这段代码是一个测试类中的方法，用于测试远程过程调用（RPC）的超时机制。代码中使用了PyTorch的RPC框架进行RPC调用，并验证在不同超时设置下的行为是否符合预期。
    # 测试远程过程调用（RPC）脚本超时情况

    # 计算下一个工作进程的排名
    next_rank = (self.rank + 1) % self.world_size
    # 获取下一个工作进程的名称
    dst_worker = worker_name(next_rank)
    # 获取预期的超时错误的正则表达式
    expected_error = self.get_timeout_error_regex()

    # 测试同步 RPC 调用，确保在超时情况下抛出预期的运行时错误
    with self.assertRaisesRegex(RuntimeError, expected_error):
        rpc.rpc_sync(dst_worker, my_script_func, args=(torch.tensor(1),), timeout=1)

    # 测试异步 RPC 调用，确保在超时情况下抛出预期的运行时错误
    fut = rpc.rpc_async(dst_worker, my_script_func, args=(torch.tensor(1),), timeout=1)
    with self.assertRaisesRegex(RuntimeError, expected_error):
        fut.wait()

    # 确保当前设置的默认超时时间足够长，以便在存在延迟的情况下 RPC 调用仍然完成
    fut = rpc.rpc_async(dst_worker, my_script_func, args=(torch.tensor(1),))
    fut.wait()

    # 确保在设置新的默认超时时间并且不覆盖的情况下，RPC 调用会超时
    rpc._set_rpc_timeout(0.001)
    fut = rpc.rpc_async(dst_worker, my_script_func, args=(torch.tensor(1),))
    with self.assertRaisesRegex(RuntimeError, expected_error):
        fut.wait()

    # 确保在指定超时时间为 0 的情况下，RPC 调用能够正常完成
    rpc._set_rpc_timeout(0.001)
    fut = rpc.rpc_async(dst_worker, my_script_func, args=(torch.tensor(1),), timeout=0)
    fut.wait()

    # 重置默认 RPC 超时时间以便进行清理关闭
    rpc._set_rpc_timeout(rpc.constants.DEFAULT_RPC_TIMEOUT_SEC)
```