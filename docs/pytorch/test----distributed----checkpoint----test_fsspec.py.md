# `.\pytorch\test\distributed\checkpoint\test_fsspec.py`

```py
# Owner(s): ["oncall: distributed"]

import shutil  # 导入 shutil 模块，用于文件和目录操作
import tempfile  # 导入 tempfile 模块，用于创建临时文件和目录
from functools import wraps  # 导入 wraps 函数，用于包装函数保留原函数的元信息
from typing import Any, Callable, Dict, Optional, Tuple  # 导入类型提示相关模块

import torch  # 导入 PyTorch 深度学习框架
import torch.distributed as dist  # 导入 PyTorch 分布式通信模块
import torch.distributed.checkpoint as dcp  # 导入 PyTorch 分布式检查点模块
import torch.nn as nn  # 导入 PyTorch 神经网络模块
from torch.distributed.checkpoint._fsspec_filesystem import FsspecReader, FsspecWriter  # 导入分布式检查点相关模块
from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict  # 导入加载分片优化器状态的函数
from torch.distributed.checkpoint.utils import CheckpointException  # 导入检查点异常相关模块
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # 导入 FullyShardedDataParallel 类
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType  # 导入状态字典类型
from torch.testing._internal.common_distributed import requires_nccl, skip_if_lt_x_gpu  # 导入测试相关模块
from torch.testing._internal.common_utils import run_tests, TestCase  # 导入测试相关模块和测试用例基类
from torch.testing._internal.distributed._shard.sharded_tensor import (
    ShardedTensorTestBase,
    with_comms,
)  # 导入分片张量相关测试基类和通信装饰器


def with_temp_dir(
    func: Optional[Callable] = None,
) -> Optional[Callable]:
    """
    Wrapper to initialize temp directory for distributed checkpoint.
    """
    assert func is not None

    @wraps(func)
    def wrapper(self, *args: Tuple[object], **kwargs: Dict[str, Any]) -> None:
        # Only create temp_dir when rank is 0
        if dist.get_rank() == 0:
            temp_dir = tempfile.mkdtemp()  # 创建临时目录
            print(f"Using temp directory: {temp_dir}")
        else:
            temp_dir = ""
        object_list = [temp_dir]

        # Broadcast temp_dir to all the other ranks
        dist.broadcast_object_list(object_list)  # 将 temp_dir 广播给所有其他进程
        self.temp_dir = object_list[0]

        try:
            func(self, *args, **kwargs)  # 调用被装饰的函数
        finally:
            if dist.get_rank() == 0:
                shutil.rmtree(self.temp_dir, ignore_errors=True)  # 在进程0上删除临时目录，忽略可能的错误

    return wrapper


class MyTestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = nn.Sequential(nn.Linear(8, 16), nn.ReLU())  # 定义神经网络序列 net1
        self.net2 = nn.Sequential(nn.Linear(16, 32), nn.ReLU())  # 定义神经网络序列 net2
        self.net3 = nn.Linear(32, 64)  # 定义线性层 net3
        self.net4 = nn.Sequential(nn.ReLU(), nn.Linear(64, 8))  # 定义神经网络序列 net4

    def forward(self, x):
        return self.net4(self.net3(self.net2(self.net1(x))))  # 前向传播函数，依次通过网络层 net1, net2, net3, net4 处理输入 x


class TestFSSpecNoDist(TestCase):
    # 测试用例类，继承自 TestCase
    # 定义一个测试方法，用于测试在没有分布式支持的情况下的文件系统规范（Fsspec）操作
    def test_fsspec_no_dist(self) -> None:
        # 使用临时目录作为测试环境
        with tempfile.TemporaryDirectory() as path:
            # 获取要保存的模块状态字典
            state_dict_to_save = MyTestModule().state_dict()

            # 调用保存状态字典的函数，将状态字典保存到指定路径下的FsspecWriter中
            dcp.save_state_dict(
                state_dict=state_dict_to_save,
                storage_writer=FsspecWriter(path),
                no_dist=True,
            )

            # 获取要加载的模块状态字典
            state_dict_to_load_to = MyTestModule().state_dict()

            # 遍历两个状态字典的每一项，确保它们不相等
            for p1, p2 in zip(
                state_dict_to_save.items(),
                state_dict_to_load_to.items(),
            ):
                self.assertNotEqual(p1, p2)

            # 从文件加载状态字典，不进行任何重新分片操作
            dcp.load_state_dict(
                state_dict=state_dict_to_load_to,
                storage_reader=FsspecReader(path),
                no_dist=True,
            )

            # 再次遍历两个状态字典的每一项，确保它们相等
            for p1, p2 in zip(
                state_dict_to_save.items(),
                state_dict_to_load_to.items(),
            ):
                self.assertEqual(p1, p2)
class TestFSSpecWithDist(ShardedTensorTestBase):
    # 定义一个测试类 TestFSSpecWithDist，继承自 ShardedTensorTestBase

    @property
    def world_size(self) -> int:
        # 返回一个整数属性 world_size
        return 2

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(2)
    @requires_nccl()
    @with_temp_dir
    def test_fsspec_with_dist(self):
        # 定义一个测试方法 test_fsspec_with_dist，使用装饰器配置测试环境

        CHECKPOINT_DIR = self.temp_dir
        # 设置检查点目录为临时目录

        model = FSDP(MyTestModule().cuda())
        # 创建一个 FSDP 模型实例，使用 MyTestModule 的 CUDA 版本
        optim = torch.optim.Adam(model.parameters(), lr=0.1)
        # 使用 Adam 优化器优化模型参数
        model(torch.rand(8, 8, device=dist.get_rank())).sum().backward()
        # 对模型进行前向传播、求和、反向传播

        optim.step()
        # 执行优化步骤

        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            # 使用 FSDP 的状态字典类型 SHARDED_STATE_DICT，保存模型状态
            state_dict = {
                "model": model.state_dict(),
                "optim": FSDP.optim_state_dict(model, optim),
            }

            dcp.save_state_dict(
                state_dict=state_dict,
                storage_writer=FsspecWriter(CHECKPOINT_DIR),
                planner=dcp.DefaultSavePlanner(),
            )
            # 使用 dcp 存储状态字典到 FsspecWriter 写入检查点目录

        model_2 = FSDP(MyTestModule().cuda())
        # 创建另一个 FSDP 模型实例，使用 MyTestModule 的 CUDA 版本
        optim_2 = torch.optim.Adam(model_2.parameters(), lr=0.1)
        # 使用 Adam 优化器优化第二个模型实例的参数

        with FSDP.summon_full_params(model):
            # 调用 FSDP 方法获取完整参数的上下文
            with FSDP.summon_full_params(model_2):
                # 调用 FSDP 方法获取第二个模型实例的完整参数的上下文
                for n_p1, n_p2 in zip(
                    model.named_parameters(), model_2.named_parameters()
                ):
                    # 遍历并比较两个模型实例的命名参数
                    self.assertNotEqual(n_p1[1], n_p2[1])

        # 现在加载模型并确保值相同
        with FSDP.state_dict_type(model_2, StateDictType.SHARDED_STATE_DICT):
            # 使用 FSDP 的状态字典类型 SHARDED_STATE_DICT，加载第二个模型的状态
            state_dict = {
                "model": model_2.state_dict(),
            }

            dcp.load_state_dict(
                state_dict=state_dict,
                storage_reader=FsspecReader(CHECKPOINT_DIR),
                planner=dcp.DefaultLoadPlanner(),
            )
            # 使用 dcp 加载状态字典，从 FsspecReader 读取检查点目录

            model_2.load_state_dict(state_dict["model"])
            # 加载第二个模型的状态字典

            optim_state = load_sharded_optimizer_state_dict(
                model_state_dict=state_dict["model"],
                optimizer_key="optim",
                storage_reader=FsspecReader(CHECKPOINT_DIR),
            )
            # 加载分片优化器状态字典

            flattened_osd = FSDP.optim_state_dict_to_load(
                model_2, optim_2, optim_state["optim"]
            )
            # 使用 FSDP 将优化器状态字典展平为加载格式
            optim_2.load_state_dict(flattened_osd)
            # 加载优化器的展平状态字典

        with FSDP.summon_full_params(model):
            # 调用 FSDP 方法获取完整参数的上下文
            with FSDP.summon_full_params(model_2):
                # 调用 FSDP 方法获取第二个模型实例的完整参数的上下文
                for n_p1, n_p2 in zip(
                    model.named_parameters(), model_2.named_parameters()
                ):
                    # 遍历并比较两个模型实例的命名参数
                    self.assertEqual(n_p1[1], n_p2[1])

        def opt_at(opt, idx):
            # 定义一个获取优化器状态的辅助函数
            return list(iter(opt.state.values()))[idx]

        # Adam 惰性创建其状态
        self.assertEqual(opt_at(optim, 0)["exp_avg"], opt_at(optim_2, 0)["exp_avg"])
        self.assertEqual(
            opt_at(optim, 0)["exp_avg_sq"], opt_at(optim_2, 0)["exp_avg_sq"]
        )
        # 断言两个优化器的统计信息是否相等
    # 定义一个测试方法，用于测试覆盖保存功能
    def test_overwrite(self):
        # 生成两个长度为 10 的随机张量 t1 和 t2
        t1, t2 = torch.randn(10), torch.randn(10)

        # 调用 dcp.save 方法，将张量 t1 保存到临时目录，不允许覆盖已存在的文件
        dcp.save(
            {"random": t1}, storage_writer=FsspecWriter(self.temp_dir, overwrite=False)
        )

        # 再次调用 dcp.save 方法，将张量 t2 保存到临时目录，允许覆盖已存在的文件
        dcp.save(
            {"random": t2}, storage_writer=FsspecWriter(self.temp_dir, overwrite=True)
        )

        # 初始化一个包含零张量的字典 sd
        sd = {"random": torch.zeros(10)}
        
        # 调用 dcp.load 方法，加载指定检查点目录中的数据到 sd 字典中
        dcp.load(sd, checkpoint_id=self.temp_dir)
        
        # 断言检查，确保 sd["random"] 中的数据与 t2 相近
        self.assertTrue(torch.allclose(sd["random"], t2))

        # 使用断言检查，预期会抛出 CheckpointException 异常，异常信息包含 "Checkpoint already exists"
        with self.assertRaisesRegex(
            CheckpointException, ".*Checkpoint already exists.*"
        ):
            # 尝试再次保存张量 t2 到临时目录，不允许覆盖已存在的文件
            dcp.save(
                {"random": t2},
                storage_writer=FsspecWriter(self.temp_dir, overwrite=False),
            )
# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 调用函数运行测试
    run_tests()
```