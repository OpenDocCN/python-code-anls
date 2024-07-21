# `.\pytorch\test\distributed\algorithms\ddp_comm_hooks\test_ddp_hooks.py`

```
# Owner(s): ["oncall: distributed"]

import os  # 导入操作系统相关功能模块
import sys  # 导入系统相关功能模块

import torch  # 导入PyTorch库
import torch.distributed as dist  # 导入PyTorch分布式模块
from torch import nn  # 导入PyTorch神经网络模块

if not dist.is_available():  # 如果分布式功能不可用
    print("Distributed not available, skipping tests", file=sys.stderr)  # 打印消息，跳过测试
    sys.exit(0)  # 退出程序

from torch.distributed.algorithms.ddp_comm_hooks import (  # 导入分布式数据并行通信钩子相关模块
    DDPCommHookType,
    register_ddp_comm_hook,
)
from torch.nn.parallel import DistributedDataParallel  # 导入分布式数据并行模块
from torch.testing._internal.common_distributed import (  # 导入内部分布式测试相关模块
    MultiProcessTestCase,
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN  # 导入测试相关工具函数及常量

if TEST_WITH_DEV_DBG_ASAN:  # 如果处于开发/调试模式（ASAN），则打印消息并退出
    print("Multiprocessing spawn is not compatible with dev/dbg asan", file=sys.stderr)
    sys.exit(0)


def gpus_for_rank(world_size):
    visible_devices = list(range(torch.cuda.device_count()))  # 获取可见的GPU设备列表
    gpus_per_process = torch.cuda.device_count() // world_size  # 计算每个进程分配的GPU数量
    gpus_for_rank = []  # 初始化存储每个进程分配的GPU列表
    for rank in range(world_size):
        gpus_for_rank.append(  # 将当前进程分配的GPU列表添加到结果中
            visible_devices[rank * gpus_per_process : (rank + 1) * gpus_per_process]
        )
    return gpus_for_rank  # 返回每个进程分配的GPU列表


class Task(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)  # 设置随机种子
        self.p = nn.Parameter(torch.randn(40, 20))  # 创建一个参数p，形状为(40, 20)

    def forward(self, x):
        return self.p * x  # 返回参数p乘以输入x的结果


class TestDdpCommHook(nn.Module):
    def __init__(self):
        super().__init__()
        self.t0 = Task()  # 创建Task对象作为成员变量t0

    def forward(self, x, rank):
        return self.t0(x ** (1 + rank))  # 对成员变量t0执行指数运算，返回结果


class DistributedDataParallelCommHookTest(MultiProcessTestCase):
    def setUp(self):
        super().setUp()  # 调用父类的setUp方法

    def tearDown(self):
        try:
            os.remove(self.file_name)  # 尝试删除文件名对应的文件
        except OSError:
            pass  # 忽略OSError异常

    def _get_process_group_nccl(self):
        store = dist.FileStore(self.file_name, self.world_size)  # 创建FileStore对象
        dist.init_process_group(
            backend="nccl",  # 指定后端为NCCL
            world_size=self.world_size,  # 设置进程组大小
            rank=self.rank,  # 设置当前进程的排名
            store=store,  # 使用之前创建的FileStore对象
        )
        return dist.distributed_c10d._get_default_group()  # 返回默认的进程组

    @property
    def world_size(self):
        return 2  # 返回进程组的大小为2

    def _local_model(self):
        local_model = TestDdpCommHook().cpu()  # 创建本地的TestDdpCommHook模型，并将其放在CPU上运行
        return local_model  # 返回本地模型对象

    def _get_grads(self, process_group, hook_type=None):
        device_id = gpus_for_rank(self.world_size)[self.rank][0]  # 获取当前进程的GPU设备ID
        gpu_model = DistributedDataParallel(
            TestDdpCommHook().to(device_id),  # 创建分布式数据并行模型，并将其放在指定GPU设备上
            device_ids=[device_id],  # 指定设备ID列表
            process_group=process_group,  # 使用给定的进程组
        )

        # Register DDP Communication Hook if defined
        if hook_type is not None:  # 如果定义了DDP通信钩子类型
            register_ddp_comm_hook(  # 注册DDP通信钩子
                comm_hook_type=hook_type,  # 指定通信钩子类型
                model=gpu_model,  # 指定应用通信钩子的模型
                state=process_group,  # 指定通信钩子的状态
            )

        return self._run_and_get_grads(gpu_model)  # 运行模型并获取梯度
    def _run_and_get_grads(self, model):
        # 设置随机种子为2020
        torch.manual_seed(2020)
        # 生成大小为(40, 20)的随机输入张量
        input = torch.randn(40, 20)
        
        # 执行前向传播
        output = model(input, self.rank)

        # 执行反向传播
        output.mean().backward()

        # 获取模型中的第一个参数的梯度
        param = next(model.parameters())
        return param.grad

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_allreduce_hook(self):
        """
        This unit test verifies the ``allreduce`` hook registered case gives same result
        with no hook registered case.
        """
        # 获取NCCL进程组
        process_group = self._get_process_group_nccl()

        # 无hook注册情况，获取参考梯度
        reference_grads = self._get_grads(process_group, None)
        # 注册allreduce hook情况，获取hook梯度
        hook_grads = self._get_grads(process_group, DDPCommHookType.ALLREDUCE)

        # 使用测试工具验证hook梯度与参考梯度之间的接近度
        torch.testing.assert_close(hook_grads, reference_grads, rtol=1e-5, atol=0)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_fp16compress_hook(self):
        """
        This unit test verifies the ``fp16 compress`` hook registered case
        gives close result with no hook registered case.
        """
        # 获取NCCL进程组
        process_group = self._get_process_group_nccl()

        # 无hook注册情况，获取参考梯度
        reference_grads = self._get_grads(process_group, None)
        # 注册fp16 compress hook情况，获取hook梯度
        hook_grads = self._get_grads(process_group, DDPCommHookType.FP16_COMPRESS)

        # 使用测试工具验证hook梯度与参考梯度之间的接近度
        torch.testing.assert_close(hook_grads, reference_grads, rtol=1e-5, atol=1e-4)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_quantize_per_tensor_hook(self):
        """
        This unit test verifies the ``quantize per tensor`` hook registered case
        gives close result with no hook registered case.
        """
        # 获取NCCL进程组
        process_group = self._get_process_group_nccl()

        # 无hook注册情况，获取参考梯度
        reference_grads = self._get_grads(process_group, None)
        # 注册quantize per tensor hook情况，获取hook梯度
        hook_grads = self._get_grads(process_group, DDPCommHookType.QUANTIZE_PER_TENSOR)

        # 使用测试工具验证hook梯度与参考梯度之间的接近度
        torch.testing.assert_close(hook_grads, reference_grads, rtol=1e-5, atol=1e-4)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_quantize_per_channel_hook(self):
        """
        This unit test verifies the ``quantize per channel`` hook registered case
        gives close result with no hook registered case.
        """
        # 获取NCCL进程组
        process_group = self._get_process_group_nccl()

        # 无hook注册情况，获取参考梯度
        reference_grads = self._get_grads(process_group, None)
        # 注册quantize per channel hook情况，获取hook梯度
        hook_grads = self._get_grads(
            process_group, DDPCommHookType.QUANTIZE_PER_CHANNEL
        )

        # 使用测试工具验证hook梯度与参考梯度之间的接近度
        torch.testing.assert_close(hook_grads, reference_grads, rtol=1e-5, atol=1e-4)
    # 使用装饰器确保需要 NCCL 支持
    @requires_nccl()
    # 要求至少有两个 GPU 才能运行此测试，否则跳过
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_noop_hook(self):
        """
        This unit test verifies the ``noop`` hook registered case and a subsequent allreduce
        gives same result with no hook registered case.
        """
        # 获取 NCCL 进程组
        process_group = self._get_process_group_nccl()

        # 在没有注册 hook 的情况下，获取参考梯度
        reference_grads = self._get_grads(process_group, None)
        # 注册 hook 的情况下，获取 hook 梯度
        hook_grads = self._get_grads(process_group, DDPCommHookType.NOOP)
        # 对 hook 梯度进行后续的 allreduce 以平均梯度
        hook_grads.div_(self.world_size)
        dist.all_reduce(hook_grads, group=process_group)

        # 断言 hook 梯度与参考梯度在一定误差范围内相等
        torch.testing.assert_close(hook_grads, reference_grads, rtol=1e-5, atol=0)

    # 使用装饰器确保需要 NCCL 支持
    @requires_nccl()
    # 要求至少有两个 GPU 才能运行此测试，否则跳过
    @skip_if_lt_x_gpu(2)
    def test_is_last_hook(self):
        # 获取 NCCL 进程组
        process_group = self._get_process_group_nccl()

        # 定义通信 hook 函数
        def hook(flags, bucket):
            flags.append(bucket.is_last())
            fut = torch.futures.Future()
            fut.set_result(bucket.buffer())
            return fut

        # 初始化标志列表
        flags = []
        # 获取当前进程的 GPU 设备 ID
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        # 构建模型：包含线性层和多个线性层的序列模型
        model = nn.Sequential(
            nn.Linear(2, 4000, bias=False),
            *[nn.Linear(4000, 4000, bias=False) for _ in range(10)],
        )
        # 将模型放置在指定 GPU 设备上，并使用 DistributedDataParallel 包装
        gpu_model = DistributedDataParallel(
            model.to(device_id),
            device_ids=[device_id],
            process_group=process_group,
        )
        # 注册通信 hook 函数
        gpu_model.register_comm_hook(state=flags, hook=hook)
        # 创建输入数据
        input = torch.randn(10, 2)
        # 前向传播、反向传播
        gpu_model(input).sum().backward()
        # 断言最后一个标志为 True
        self.assertTrue(flags[-1])
        # 断言除了最后一个标志外，其他标志均为 False
        self.assertFalse(any(flags[:-1]))
if __name__ == "__main__":
    # 检查当前模块是否作为主程序运行
    assert (
        not torch.cuda._initialized
    ), "test_distributed must not have initialized CUDA context on main process"
    # 断言确保 CUDA 上下文在主进程中没有被初始化，否则抛出异常信息

    # 运行测试函数
    run_tests()
```