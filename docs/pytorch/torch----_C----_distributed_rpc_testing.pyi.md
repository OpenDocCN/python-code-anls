# `.\pytorch\torch\_C\_distributed_rpc_testing.pyi`

```py
# 导入PyTorch库
import torch
# 导入分布式通信库中的Store类
from torch._C._distributed_c10d import Store
# 导入分布式RPC中的TensorPipe后端选项基类和TensorPipeAgent类
from torch._C._distributed_rpc import _TensorPipeRpcBackendOptionsBase, TensorPipeAgent

# 以下代码为测试目的，定义在torch/csrc/distributed/rpc/testing/init.cpp中

# 定义一个故障模拟的TensorPipe后端选项类，继承自_TensorPipeRpcBackendOptionsBase类
class FaultyTensorPipeRpcBackendOptions(_TensorPipeRpcBackendOptionsBase):
    def __init__(
        self,
        num_worker_threads: int,
        rpc_timeout: float,
        init_method: str,
        messages_to_fail: list[str],
        messages_to_delay: dict[str, float],
        num_fail_sends: int,
    ) -> None: ...
    # 发送接收线程数
    num_send_recv_threads: int
    # 需要模拟失败的消息列表
    messages_to_fail: list[str]
    # 需要延迟的消息及其延迟时间的映射
    messages_to_delay: dict[str, float]
    # 需要模拟失败的发送次数
    num_fail_sends: int

# 定义一个故障模拟的TensorPipe代理类，继承自TensorPipeAgent类
class FaultyTensorPipeAgent(TensorPipeAgent):
    def __init__(
        self,
        store: Store,
        name: str,
        rank: int,
        world_size: int,
        options: FaultyTensorPipeRpcBackendOptions,
        reverse_device_maps: dict[str, dict[torch.device, torch.device]],
        devices: list[torch.device],
    ) -> None: ...
```