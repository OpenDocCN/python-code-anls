# `.\pytorch\torch\testing\_internal\distributed\rpc\faulty_rpc_agent_test_fixture.py`

```py
# 忽略类型检查错误，这通常是由于特定的类型声明方式导致的问题
mypy: ignore-errors

# 导入RPC模块，用于分布式远程过程调用
import torch.distributed.rpc as rpc

# 导入用于测试的内部模块，并通过`noqa: F401`告知 linter 忽略未使用的导入警告
import torch.distributed.rpc._testing  # noqa: F401

# 导入RPC代理测试框架，用于设置和运行测试
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,
)

# 定义可以在RREF协议和分布式自动微分中重试的消息类型列表
retryable_message_types = ["RREF_FORK_REQUEST",
                           "RREF_CHILD_ACCEPT",
                           "RREF_USER_DELETE",
                           "CLEANUP_AUTOGRAD_CONTEXT_REQ"]

# 定义在`FaultyTensorPipeAgent`的`enqueueSend()`函数中处理时的默认消息延迟时间
default_messages_to_delay = {
    "PYTHON_CALL": 1.5,  # Python用户定义函数
    "SCRIPT_CALL": 1.5,  # 脚本/内置函数调用
}

# 定义`FaultyRpcAgentTestFixture`类，继承自`RpcAgentTestFixture`类
class FaultyRpcAgentTestFixture(RpcAgentTestFixture):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 设置失败的消息类型列表
        self.messages_to_fail = retryable_message_types
        # 设置延迟的消息类型和延迟时间
        self.messages_to_delay = default_messages_to_delay

    # 返回RPC后端类型
    @property
    def rpc_backend(self):
        return rpc.backend_registry.BackendType[
            "FAULTY_TENSORPIPE"
        ]

    # 返回RPC后端选项
    @property
    def rpc_backend_options(self):
        return rpc.backend_registry.construct_rpc_backend_options(
            self.rpc_backend,
            init_method=self.init_method,
            num_worker_threads=8,
            num_fail_sends=3,
            messages_to_fail=self.messages_to_fail,
            messages_to_delay=self.messages_to_delay,
        )

    # 设置故障注入的方法，可以更改故障消息列表和延迟消息列表
    def setup_fault_injection(self, faulty_messages, messages_to_delay):
        if faulty_messages is not None:
            self.messages_to_fail = faulty_messages
        if messages_to_delay is not None:
            self.messages_to_delay = messages_to_delay

    # 返回用于关闭错误的正则表达式，用于匹配特定的错误消息
    def get_shutdown_error_regex(self):
        error_regexes = [
            "Exception in thread pool task",
            "Connection reset by peer",
            "Connection closed by peer"
        ]
        return "|".join([f"({error_str})" for error_str in error_regexes])

    # 返回用于超时错误的正则表达式，用于匹配超时的错误消息
    def get_timeout_error_regex(self):
        return "RPC ran for more than"
```