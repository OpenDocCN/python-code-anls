# `.\pytorch\test\distributed\rpc\test_share_memory.py`

```py
#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]

# 导入必要的库和模块
import contextlib  # 上下文管理器工具模块
import copyreg  # 复制注册模块，用于自定义对象的序列化和反序列化
import os  # 系统相关操作模块
import sys  # 系统相关操作模块

import torch  # PyTorch 深度学习库
import torch.distributed as dist  # PyTorch 分布式通信模块

# 检查分布式模块是否可用，不可用则输出错误信息并退出
if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 导入 PyTorch 分布式 RPC 相关模块和函数
import torch.distributed.rpc as rpc  # PyTorch 分布式 RPC 模块
import torch.multiprocessing.reductions as TorchMpReductions  # PyTorch 多进程共享内存工具模块
from torch import multiprocessing  # PyTorch 多进程模块
from torch.distributed.rpc.api import _use_rpc_pickler  # RPC Pickler API
from torch.distributed.rpc.internal import _InternalRPCPickler  # 内部 RPC Pickler 实现
from torch.testing._internal.common_utils import run_tests, TestCase  # 测试相关工具

# 定义一个上下文管理器，用于设置多进程共享文件系统策略
@contextlib.contextmanager
def fs_sharing():
    prev_strategy = multiprocessing.get_sharing_strategy()
    multiprocessing.set_sharing_strategy("file_system")
    try:
        yield
    finally:
        multiprocessing.set_sharing_strategy(prev_strategy)

# 自定义的 RPC Pickler 类，继承自 _InternalRPCPickler
class ShareMemoryRPCPickler(_InternalRPCPickler):
    def __init__(self) -> None:
        super().__init__()
        self._dispatch_table  # RPC Pickler 的分发表
        # 复制全局的复制注册分发表到本地分发表
        self._dispatch_table = copyreg.dispatch_table.copy()

        # 为所有注册的 PyTorch 存储类添加到分发表中指定的 reduce 函数
        for t in torch._storage_classes:
            self._dispatch_table[t] = TorchMpReductions.reduce_storage

        # 为所有注册的 PyTorch 张量类添加到分发表中指定的 reduce 函数
        for t in torch._tensor_classes:
            self._dispatch_table[t] = TorchMpReductions.reduce_tensor
        # 将 torch.Tensor 类型映射到指定的 reduce 函数
        self._dispatch_table[torch.Tensor] = TorchMpReductions.reduce_tensor
        # 将 torch.nn.parameter.Parameter 类型映射到指定的 reduce 函数
        self._dispatch_table[
            torch.nn.parameter.Parameter
        ] = TorchMpReductions.reduce_tensor

# 定义一个简单的工作函数，初始化并关闭 RPC
def worker_loop(a):
    rpc.init_rpc("worker1", rank=1, world_size=2)
    rpc.shutdown()

# 空的工作函数
def worker_fn(m):
    pass

# 测试类，继承自 TestCase
class TestRPCPickler(TestCase):
    def test_case(self):
        # 设置主节点地址和端口环境变量
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"

        # 使用文件系统共享策略的上下文
        with fs_sharing():
            # 在一个新进程中运行 worker_loop 函数
            r = multiprocessing.spawn(worker_loop, join=False)

            try:
                # 使用自定义的 RPC Pickler 开始 RPC 初始化
                with _use_rpc_pickler(ShareMemoryRPCPickler()):
                    rpc.init_rpc("worker0", rank=0, world_size=2)
                    m = torch.nn.Linear(1, 2)
                    m.share_memory()
                    # 在远程 worker1 上调用 worker_fn 函数
                    rref = rpc.remote("worker1", worker_fn, args=(m,))

                    # 等待远程调用完成
                    rref.to_here()
            finally:
                # 关闭 RPC
                rpc.shutdown()
                # 等待进程结束
                r.join()

# 如果当前脚本被直接执行，则运行测试
if __name__ == "__main__":
    run_tests()
```