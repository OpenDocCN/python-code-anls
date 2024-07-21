# `.\pytorch\torch\testing\_internal\distributed\_shard\sharded_tensor\__init__.py`

```py
# 忽略 mypy 的错误信息

# 导入系统模块 sys
import sys
# 导入 functools 模块中的 wraps 和 partial 函数
from functools import wraps, partial

# 导入 PyTorch 库
import torch
# 导入 torch 分布式模块
import torch.distributed as dist
# 从 torch.distributed 模块导入 rpc
from torch.distributed import rpc
# 从 torch.testing._internal.common_distributed 导入测试相关的模块和常量
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    TEST_SKIPS,
    tp_transports,
)

# 定义全局测试 GPU 数量
TEST_GPU_NUM = 4

# 定义 ShardedTensorTestBase 类，继承自 MultiProcessTestCase 类
class ShardedTensorTestBase(MultiProcessTestCase):
    # 定义 world_size 属性，返回测试 GPU 数量
    @property
    def world_size(self):
        return TEST_GPU_NUM

    # 初始化进程组方法，指定后端为 nccl/gloo/mpi 中的一个，初始化进程组
    def init_pg(self, backend="nccl"):
        if backend not in ["nccl", "gloo", "mpi"]:
            raise RuntimeError(f"Backend {backend} not supported!")

        dist.init_process_group(
            backend=backend,
            world_size=self.world_size,
            rank=self.rank,
            init_method=f"file://{self.file_name}",
        )

        # 如果使用 nccl 后端，设置当前设备为对应 rank 的 GPU 设备
        if backend == "nccl":
            torch.cuda.set_device(self.rank)

    # 初始化 RPC 方法，设置 TensorPipe 作为 RPC 后端
    def init_rpc(self):
        rpc_backend_options = rpc.TensorPipeRpcBackendOptions(_transports=tp_transports())
        rpc_backend_options.init_method = f"file://{self.file_name}"
        for rank in range(self.world_size):
            # 设置设备映射，使得 worker 之间可以相互访问彼此的设备
            rpc_backend_options.set_device_map(
                f"worker{rank}", {rank: self.rank, self.rank: rank}
            )

        # 初始化 RPC，为当前进程设置名称和排名
        rpc.init_rpc(
            name="worker%d" % self.rank,
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=rpc_backend_options,
        )

    # 初始化通信方法，可以选择是否初始化 RPC 和指定通信后端
    def init_comms(self, init_rpc=True, backend="nccl"):
        if init_rpc:
            self.init_rpc()
        self.init_pg(backend=backend)

    # 销毁通信方法，可以选择是否销毁 RPC
    def destroy_comms(self, destroy_rpc=True):
        # 等待所有进程都到达此处再开始关闭
        dist.barrier()

        if destroy_rpc:
            # 关闭 RPC
            rpc.shutdown()
        # 销毁进程组
        dist.destroy_process_group()

    # 设置测试前的准备工作
    def setUp(self) -> None:
        super().setUp()
        # 生成多进程
        self._spawn_processes()

    # 断言两个 ShardedTensor 对象是否相等的方法
    def assert_sharded_tensor_equal(self, st1, st2):
        st1_local_shards = st1.local_shards()
        st2_local_shards = st2.local_shards()
        self.assertEqual(len(st1_local_shards), len(st2_local_shards))
        for i, st1_local_shard in enumerate(st1_local_shards):
            # 检查本地分片的张量数据和元数据是否相等
            self.assertEqual(st1_local_shard.tensor, st2_local_shards[i].tensor)
            self.assertEqual(st1_local_shard.metadata, st2_local_shards[i].metadata)

        # 检查整体的元数据和分片规格是否相等
        self.assertEqual(st1.metadata(), st2.metadata())
        self.assertEqual(st1.sharding_spec(), st2.sharding_spec())
        # 检查远程分片的数量是否相等
        self.assertEqual(len(st1.remote_shards()), len(st2.remote_shards()))


# 用于初始化通信（进程组 + RPC）的装饰器函数
def with_comms(func=None, init_rpc=True, backend="nccl"):
    if func is None:
        return partial(
            with_comms,
            init_rpc=init_rpc,
            backend=backend,
        )

    @wraps(func)
    # 包装函数，添加了通信初始化的参数
    # 定义装饰器函数，接受 `self`，可变位置参数 `args`，可变关键字参数 `kwargs`
    def wrapper(self, *args, **kwargs):
        # 如果后端是 "nccl" 并且 CUDA 设备数量小于总节点数 `self.world_size`
        if backend == "nccl" and torch.cuda.device_count() < self.world_size:
            # 以特定退出码退出程序，该退出码根据 `self.world_size` 来自测试跳过字典 `TEST_SKIPS` 中获取
            sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)
        # 初始化通信机制，根据参数 `init_rpc` 和 `backend`
        self.init_comms(init_rpc=init_rpc, backend=backend)
        # 调用被装饰的函数 `func`，传入 `self`，`args`，`kwargs`
        func(self, *args, **kwargs)
        # 销毁通信机制，根据参数 `destroy_rpc`
        self.destroy_comms(destroy_rpc=init_rpc)
    # 返回装饰器函数 `wrapper`
    return wrapper
```