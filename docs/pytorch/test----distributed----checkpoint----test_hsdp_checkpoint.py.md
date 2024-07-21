# `.\pytorch\test\distributed\checkpoint\test_hsdp_checkpoint.py`

```py
# 所有者：["oncall: distributed"]
from copy import deepcopy  # 导入深拷贝函数 deepcopy

import torch  # 导入 PyTorch 库
import torch.distributed.checkpoint as dist_cp  # 导入分布式检查点模块
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入神经网络函数模块
from torch.distributed._tensor import init_device_mesh, Replicate  # 导入分布式相关模块

from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)  # 导入默认的检查点加载和保存计划器

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # 导入完全分片数据并行模块
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    ShardingStrategy,
    StateDictType,
)  # 导入分片策略和状态字典类型

from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 导入用于测试的 GPU 数量检查函数
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)  # 导入测试实例化、参数化和运行测试的工具函数

from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)  # 导入分布式张量测试基类和通信装饰器

from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir  # 导入临时目录装饰器


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = nn.Linear(5, 8)  # 定义输入维度为 5，输出维度为 8 的线性层
        self.relu = nn.ReLU()  # 定义 ReLU 激活函数
        self.net2 = nn.Linear(8, 4)  # 定义输入维度为 8，输出维度为 4 的线性层
        self.net3 = nn.Linear(4, 12)  # 定义输入维度为 4，输出维度为 12 的线性层

    def forward(self, x):
        x = F.relu(self.net1(x))  # 使用 ReLU 激活函数对 net1 的输出进行非线性变换
        x = F.relu(self.net2(x))  # 使用 ReLU 激活函数对 net2 的输出进行非线性变换
        x = F.relu(self.net3(x))  # 使用 ReLU 激活函数对 net3 的输出进行非线性变换
        return x  # 返回处理后的张量

    def get_input(self):
        return torch.rand(4, 5, device="cuda")  # 返回一个形状为 [4, 5] 的随机张量，放置在 CUDA 设备上


class SimpleModelUneven(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = nn.Linear(5, 10)  # 定义输入维度为 5，输出维度为 10 的线性层
        self.relu = nn.ReLU()  # 定义 ReLU 激活函数
        self.net2 = nn.Linear(10, 15)  # 定义输入维度为 10，输出维度为 15 的线性层
        self.net3 = nn.Linear(15, 30)  # 定义输入维度为 15，输出维度为 30 的线性层
        self.net4 = nn.Linear(30, 5)  # 定义输入维度为 30，输出维度为 5 的线性层

    def forward(self, x):
        x = F.relu(self.net1(x))  # 使用 ReLU 激活函数对 net1 的输出进行非线性变换
        x = F.relu(self.net2(x))  # 使用 ReLU 激活函数对 net2 的输出进行非线性变换
        x = F.relu(self.net3(x))  # 使用 ReLU 激活函数对 net3 的输出进行非线性变换
        x = F.relu(self.net4(x))  # 使用 ReLU 激活函数对 net4 的输出进行非线性变换
        return x  # 返回处理后的张量

    def get_input(self):
        return torch.rand(4, 5, device="cuda")  # 返回一个形状为 [4, 5] 的随机张量，放置在 CUDA 设备上


class TestHSDPCheckpoint(DTensorTestBase):
    @property
    def backend(self):
        return "cpu:gloo,cuda:nccl"  # 返回测试使用的后端为 "cpu:gloo,cuda:nccl"

    @with_comms  # 使用通信装饰器
    @skip_if_lt_x_gpu(4)  # 如果 GPU 数量少于 4，跳过测试
    @with_temp_dir  # 使用临时目录装饰器
    @parametrize("is_even_sharded_model", [True, False])  # 参数化测试，包括偶数分片模型和非偶数分片模型
    def test_hsdp_checkpoint(self, is_even_sharded_model) -> None:
        # 设置检查点目录为临时目录
        CHECKPOINT_DIR = self.temp_dir
        # 根据是否是偶数分片模型选择简单模型类
        simple_model = SimpleModel if is_even_sharded_model else SimpleModelUneven

        # 初始化二维设备网格
        mesh_2d = init_device_mesh(self.device_type, (2, self.world_size // 2))
        # 创建分片数据并行模型，使用CUDA加速
        model = FSDP(
            simple_model().cuda(),
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            device_mesh=mesh_2d,
        )
        # 使用Adam优化器
        optim = torch.optim.Adam(model.parameters(), lr=0.1)

        # 设置FSDP模型的状态字典类型为分片状态字典
        FSDP.set_state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
        )
        # 获取模型的当前状态字典
        state_dict = {"model": model.state_dict()}
        # 深拷贝状态字典以备份
        state_dict_to_save = deepcopy(state_dict)

        # 保存模型状态字典到文件系统
        dist_cp.save_state_dict(
            state_dict=state_dict_to_save,
            storage_writer=dist_cp.FileSystemWriter(CHECKPOINT_DIR),
            planner=DefaultSavePlanner(),
        )

        # 更新模型参数，使当前模型状态字典与备份状态字典不同
        model(model.get_input()).sum().backward()
        optim.step()

        # 此时，当前状态字典应与备份状态字典不同
        for (k1, v1), (k2, v2) in zip(
            state_dict_to_save["model"].items(), model.state_dict().items()
        ):
            self.assertEqual(k1, k2)
            self.assertEqual(v1.device_mesh, v2.device_mesh)
            self.assertEqual(v1.placements, v2.placements)
            self.assertNotEqual(v1.to_local(), v2.to_local())

        # 从文件系统加载状态字典
        dist_cp.load_state_dict(
            state_dict=state_dict_to_save,
            storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
            planner=DefaultLoadPlanner(),
        )
        # 加载备份状态字典到模型
        model.load_state_dict(state_dict_to_save["model"])

        # 加载后，当前模型状态字典应与备份状态字典相同
        state_dict_after_load = model.state_dict()
        for (k1, v1), (k2, v2) in zip(
            state_dict_to_save["model"].items(), model.state_dict().items()
        ):
            self.assertEqual(k1, k2)
            self.assertEqual(v1.device_mesh, v2.device_mesh)
            self.assertEqual(v1.placements, v2.placements)
            self.assertEqual(v1.to_local(), v2.to_local())
    def test_hsdp_fsdp_checkpoint_conversion(self, is_even_sharded_model) -> None:
        # 设置检查点保存的目录为临时目录
        CHECKPOINT_DIR = self.temp_dir
        # 根据条件选择简单模型类
        simple_model = SimpleModel if is_even_sharded_model else SimpleModelUneven

        # 保存 hsdp 模型的状态字典
        # 初始化设备网格为二维，并分配到设备上
        mesh_2d = init_device_mesh(self.device_type, (2, self.world_size // 2))
        # 创建并初始化 FSDP 模型，使用 CUDA 加速
        hsdp_model = FSDP(
            simple_model().cuda(),
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            device_mesh=mesh_2d,
        )
        # 设置 hsdp_model 的状态字典类型为分片状态字典
        FSDP.set_state_dict_type(
            hsdp_model,
            StateDictType.SHARDED_STATE_DICT,
        )
        # 获取 hsdp_model 的状态字典
        hsdp_state_dict = {"model": hsdp_model.state_dict()}
        # 保存状态字典到指定目录
        dist_cp.save_state_dict(
            state_dict=hsdp_state_dict,
            storage_writer=dist_cp.FileSystemWriter(CHECKPOINT_DIR),
            planner=DefaultSavePlanner(),
        )

        # 初始化一个 fsdp 模型以加载检查点
        # 初始化设备网格为一维，并分配到设备上
        mesh_1d = init_device_mesh(self.device_type, (self.world_size,))
        # 创建并初始化 FSDP 模型，使用 CUDA 加速
        fsdp_model = FSDP(
            simple_model().cuda(),
            device_mesh=mesh_1d,
        )
        # 设置 fsdp_model 的状态字典类型为分片状态字典
        FSDP.set_state_dict_type(
            fsdp_model,
            StateDictType.SHARDED_STATE_DICT,
        )
        # 获取 fsdp_model 的状态字典
        fsdp_state_dict = {"model": fsdp_model.state_dict()}

        # 在此时，hsdp 模型的参数与 fsdp 模型的参数不同。
        # 比较 hsdp_model 和 fsdp_model 的状态字典中的键和设备分布
        for (k1, v1), (k2, v2) in zip(
            hsdp_state_dict["model"].items(), fsdp_state_dict["model"].items()
        ):
            # 断言键相等
            self.assertEqual(k1, k2)
            # 断言设备网格不相等
            self.assertNotEqual(v1.device_mesh, v2.device_mesh)
            # 断言设备放置不相等
            self.assertNotEqual(v1.placements, v2.placements)
            # 对 v1 和 v2 进行重新分布，使用指定的网格和放置策略
            v1_all_gather = v1.redistribute(
                mesh_2d, placements=(Replicate(), Replicate())
            )
            v2_all_gather = v2.redistribute(mesh_1d, placements=(Replicate(),))
            # 断言重新分布后的本地数据相等
            self.assertNotEqual(v1_all_gather.to_local(), v2_all_gather.to_local())

        # 从存储中加载 fsdp 模型的状态字典
        dist_cp.load_state_dict(
            state_dict=fsdp_state_dict,
            storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
            planner=DefaultLoadPlanner(),
        )
        # 加载 fsdp 模型的状态字典
        fsdp_model.load_state_dict(fsdp_state_dict["model"])

        # 加载后，当前模型状态字典应与 hsdp_state_dict 相同。
        for (k1, v1), (k2, v2) in zip(
            hsdp_state_dict["model"].items(), fsdp_model.state_dict().items()
        ):
            # 断言键相等
            self.assertEqual(k1, k2)
            # 断言设备网格不相等
            self.assertNotEqual(v1.device_mesh, v2.device_mesh)
            # 断言设备放置不相等
            self.assertNotEqual(v1.placements, v2.placements)
            # 对 v1 和 v2 进行重新分布，使用指定的网格和放置策略
            v1_all_gather = v1.redistribute(
                mesh_2d, placements=(Replicate(), Replicate())
            )
            v2_all_gather = v2.redistribute(mesh_1d, placements=(Replicate(),))
            # 断言重新分布后的本地数据相等
            self.assertEqual(v1_all_gather.to_local(), v2_all_gather.to_local())
# 实例化参数化测试用例，传入 TestHSDPCheckpoint 类作为参数
instantiate_parametrized_tests(TestHSDPCheckpoint)
# 如果当前脚本作为主程序执行
if __name__ == "__main__":
    # 运行测试
    run_tests()
```