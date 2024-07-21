# `.\pytorch\test\distributed\checkpoint\test_fsdp_optim_state.py`

```py
# Owner(s): ["oncall: distributed"]

import torch  # 导入 PyTorch 库

import torch.distributed.checkpoint as DCP  # 导入分布式检查点模块
import torch.nn as nn  # 导入神经网络模块
from torch.distributed._shard.sharded_tensor.api import ShardedTensor  # 导入分片张量 API
from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict  # 导入加载分片优化器状态字典的函数

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # 导入全分片数据并行模块
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType  # 导入状态字典类型
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu  # 导入 GPU 数量检查装饰器
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,  # 导入实例化参数化测试函数
    parametrize,  # 导入参数化装饰器
    run_tests,  # 导入运行测试函数
)

from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,  # 导入分布式张量测试基类
    with_comms,  # 导入通信装饰器
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir  # 导入临时目录装饰器


class FsdpOptimStateCheckpoint(DTensorTestBase):
    def _create_model(self):
        # make weight tensor dim_0 as large as the world size for scaling test
        layer1_weight_dim = self.world_size  # 设置第一层权重维度为世界大小
        layer2_weight_dim = self.world_size * 2  # 设置第二层权重维度为世界大小的两倍
        layer3_weight_dim = self.world_size * 3  # 设置第三层权重维度为世界大小的三倍

        class TestDummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net1 = nn.Sequential(nn.Linear(8, layer1_weight_dim), nn.ReLU())  # 设置网络第一层
                self.net2 = nn.Sequential(
                    nn.Linear(layer1_weight_dim, layer2_weight_dim), nn.ReLU()
                )  # 设置网络第二层
                self.net3 = nn.Sequential(
                    nn.Linear(layer2_weight_dim, layer3_weight_dim), nn.ReLU()
                )  # 设置网络第三层

            def forward(self, x):
                return self.net3(self.net2(self.net1(x)))  # 网络前向传播

            def get_input(self):
                return torch.rand(8, 8, device="cuda")  # 返回随机输入数据

        model = TestDummyModel().cuda()  # 创建测试模型并放在 GPU 上
        return model

    @property
    def backend(self):
        return "cpu:gloo,cuda:nccl"  # 返回后端设备配置

    @with_comms  # 使用通信装饰器
    @skip_if_lt_x_gpu(2)  # 如果 GPU 数量小于 2 则跳过测试
    @with_temp_dir  # 使用临时目录装饰器
    @parametrize("pass_planner", [True, False])  # 参数化测试函数，传入 pass_planner 参数
    # 定义测试函数，用于加载分片优化器状态字典
    def test_load_sharded_optimizer_state_dict(self, pass_planner) -> None:
        # 设置检查点目录为临时目录
        CHECKPOINT_DIR = self.temp_dir
        # 根据条件选择是否传递加载计划器
        planner = DCP.DefaultLoadPlanner() if pass_planner else None

        # 创建模型
        model = self._create_model()
        # 对模型应用深度分片数据并行（FSDP）
        model = FSDP(model)
        # 使用Adam优化器初始化模型参数
        optim = torch.optim.Adam(model.parameters(), lr=0.1)

        # 前进一步以初始化优化器
        model(model.get_input()).sum().backward()
        optim.step()

        # 设置模型状态字典类型为分片状态字典
        FSDP.set_state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
        )
        # 获取分片优化器状态字典
        optim_osd = FSDP.optim_state_dict(model, optim)

        # 构建状态字典，包括模型和优化器状态
        state_dict = {
            "model": model.state_dict(),
            "optim": optim_osd,
        }
        # 保存状态字典到文件系统
        DCP.save_state_dict(
            state_dict=state_dict,
            storage_writer=DCP.FileSystemWriter(CHECKPOINT_DIR),
        )

        # 现在加载模型并确保值相同
        model_2 = self._create_model()
        model_2 = FSDP(model_2)
        optim_2 = torch.optim.Adam(model_2.parameters(), lr=0.1)

        # 设置模型状态字典类型为分片状态字典
        FSDP.set_state_dict_type(
            model_2,
            StateDictType.SHARDED_STATE_DICT,
        )
        # 验证Adam优化器延迟创建其状态
        self.assertEqual(0, len(optim_2.state))

        # 构建状态字典，仅包括模型状态（无法同时加载优化器和模型）
        state_dict = {
            "model": model_2.state_dict(),
            # 无法同时加载模型和优化器
        }
        # 从文件系统读取状态字典
        DCP.load_state_dict(
            state_dict=state_dict,
            storage_reader=DCP.FileSystemReader(CHECKPOINT_DIR),
        )
        # 加载模型状态字典
        model_2.load_state_dict(state_dict["model"])

        # 加载分片优化器状态字典
        optim_state = load_sharded_optimizer_state_dict(
            model_state_dict=state_dict["model"],
            optimizer_key="optim",
            storage_reader=DCP.FileSystemReader(CHECKPOINT_DIR),
            planner=planner,
        )
        # 将分片优化器状态字典转换为加载格式
        flattened_osd = FSDP.optim_state_dict_to_load(
            model_2, optim_2, optim_state["optim"]
        )
        # 加载优化器状态字典
        optim_2.load_state_dict(flattened_osd)
        # 获取加载后的优化器状态字典
        osd_after_load = FSDP.optim_state_dict(model_2, optim_2)

        # 比较保存前后的优化器状态字典
        before_optim_state = optim_osd["state"]
        after_optim_state = osd_after_load["state"]
        self.assertEqual(len(before_optim_state), len(after_optim_state))
        # 遍历比较每个完全限定名（FQN）下的状态
        for fqn, states in before_optim_state.items():
            for state_name, state in states.items():
                # 如果状态是分片张量，则进行张量值的比较
                state2 = after_optim_state.get(fqn).get(state_name)
                if isinstance(state, ShardedTensor):
                    self.assertTrue(isinstance(state2, ShardedTensor))
                    self.assertTrue(torch.allclose(state, state2))
                else:
                    self.assertEqual(state, state2)
# 实例化参数化测试，使用 FsdpOptimStateCheckpoint 类
instantiate_parametrized_tests(FsdpOptimStateCheckpoint)
# 如果当前脚本作为主程序执行，则运行测试
if __name__ == "__main__":
    run_tests()
```