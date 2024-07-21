# `.\pytorch\test\distributed\fsdp\test_hsdp_dtensor_state_dict.py`

```
# Owner(s): ["oncall: distributed"]

# 导入必要的库和模块
import io
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._shard.sharded_tensor import ShardedTensor

from torch.distributed._tensor import DTensor, Replicate, Shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import (
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    ShardingStrategy,
    StateDictType,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
)


# Simple and boring model to test interface and some corner cases that do not
# require complicated wrapping strategy.
# 定义一个简单的模型用于测试接口和一些不需要复杂封装策略的边缘情况
class DenseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.net1 = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        self.net2 = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
        self.net3 = nn.Sequential(nn.Linear(32, 64), nn.ReLU())
        self.net4 = nn.Sequential(nn.ReLU(), nn.Linear(64, 8))

    def forward(self, x):
        # 前向传播函数，依次经过四个网络层
        return self.net4(self.net3(self.net2(self.net1(x))))

    def get_input(self):
        # 返回一个在CUDA设备上的随机张量作为输入
        return torch.rand(4, 8, device="cuda")


# TODO: Consolidate DeviceMesh based FSDP and HSDP test cases.
# TODO: 合并基于 DeviceMesh 的 FSDP 和 HSDP 测试用例
class TestHSDPWithDeviceMeshAndDTensor(DTensorTestBase):
    def _create_model(self, device_mesh=None):
        if device_mesh:
            # 如果提供了 device_mesh 参数，则使用设备网格初始化 FSDP
            model = FSDP(
                DenseModel().cuda(),
                device_mesh=device_mesh,
                sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            )
        else:
            # 否则，初始化一个二维设备网格，并创建进程组
            mesh_2d = init_device_mesh(self.device_type, (2, self.world_size // 2))
            intra_node_pg = mesh_2d.get_group(mesh_dim=1)
            inter_node_pg = mesh_2d.get_group(mesh_dim=0)
            model = FSDP(
                DenseModel().cuda(),
                process_group=(intra_node_pg, inter_node_pg),
                sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            )

        # 使用 Adam 优化器优化模型参数
        optim = torch.optim.Adam(model.parameters(), lr=0.1)
        # 对模型进行一次前向传播、求和操作和反向传播
        model(model.get_input()).sum().backward()
        # 执行优化步骤
        optim.step()

        return model, optim

    @with_comms
    @skip_if_lt_x_gpu(4)
    # 测试初始化具有设备网格的情况
    def test_hsdp_init_with_device_mesh(self):
        # 初始化一个二维设备网格
        mesh_2d = init_device_mesh(self.device_type, (2, self.world_size // 2))
        # 创建模型和优化器
        model, optim = self._create_model(mesh_2d)

        # 设置模型的状态字典类型为SHARDED_STATE_DICT
        FSDP.set_state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
        )
        # 获取模型的状态字典
        state_dict = model.state_dict()
        # 获取优化器的状态字典
        optim_state_dict = FSDP.optim_state_dict(model, optim)

        # 遍历模型状态字典中的所有值
        for v in state_dict.values():
            # 断言值的类型是DTensor
            self.assertEqual(type(v), DTensor)
            # 断言值的放置（placements）长度为2
            self.assertEqual(len(v.placements), 2)
            # 断言值的放置是(Replicate(), Shard(0))
            self.assertEqual(v.placements, (Replicate(), Shard(0)))
            # 断言值的设备网格是初始化的mesh_2d
            self.assertEqual(v.device_mesh, mesh_2d)

        # 遍历优化器状态字典中的所有状态
        for state in optim_state_dict["state"].values():
            for k, v in state.items():
                if k != "step":
                    # 断言状态值的类型是DTensor
                    self.assertEqual(type(v), DTensor)
                    # 断言状态值的放置长度为2
                    self.assertEqual(len(v.placements), 2)
                    # 断言状态值的放置是(Replicate(), Shard(0))
                    self.assertEqual(v.placements, (Replicate(), Shard(0)))
                    # 断言状态值的设备网格是初始化的mesh_2d
                    self.assertEqual(v.device_mesh, mesh_2d)

        # 获取模型的状态字典类型
        state_dict_type = model.get_state_dict_type(model)
        # 断言如果在初始化FSDP时使用了device_mesh，则字段_use_dtensor将自动设置为True
        self.assertEqual(state_dict_type.state_dict_config._use_dtensor, True)
        self.assertEqual(state_dict_type.optim_state_dict_config._use_dtensor, True)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @parametrize("offload_to_cpu", [True, False])
    # 定义一个测试方法，用于验证分布式张量和分片张量的状态字典是否相同
    def test_dtensor_sharded_tensor_state_dict_identical(self, offload_to_cpu):
        # 初始化一个二维设备网格，根据当前设备类型和世界大小初始化
        mesh_2d = init_device_mesh(self.device_type, (2, self.world_size // 2))
        # 创建模型和优化器
        model, optim = self._create_model(mesh_2d)

        # 设置模型的状态字典类型为分片状态字典，并配置状态字典参数和优化状态字典参数
        FSDP.set_state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
            state_dict_config=ShardedStateDictConfig(offload_to_cpu=offload_to_cpu),
            optim_state_dict_config=ShardedOptimStateDictConfig(
                offload_to_cpu=offload_to_cpu
            ),
        )
        # 获取模型的状态字典和优化器状态字典
        dtensor_sd = model.state_dict()
        dtensor_osd = FSDP.optim_state_dict(model, optim)

        # 创建参考模型和优化器
        ref_model, ref_optim = self._create_model()
        # 设置参考模型的状态字典类型为分片状态字典，并配置状态字典参数和优化状态字典参数
        FSDP.set_state_dict_type(
            ref_model,
            StateDictType.SHARDED_STATE_DICT,
            state_dict_config=ShardedStateDictConfig(offload_to_cpu=offload_to_cpu),
            optim_state_dict_config=ShardedOptimStateDictConfig(
                offload_to_cpu=offload_to_cpu
            ),
        )
        # 获取参考模型的状态字典和优化器状态字典
        sharded_tensor_sd = ref_model.state_dict()
        sharded_tensor_osd = FSDP.optim_state_dict(ref_model, ref_optim)

        # 检查分布式张量和分片张量模型状态字典的值是否相同
        for dtensor_sd_item, sharded_tensor_sd_item in zip(
            dtensor_sd.items(), sharded_tensor_sd.items()
        ):
            k1, v1 = dtensor_sd_item
            k2, v2 = sharded_tensor_sd_item
            self.assertEqual(k1, k2)  # 检查键是否相同

            self.assertEqual(type(v1), DTensor)  # 检查值的类型是否为DTensor
            self.assertEqual(type(v2), ShardedTensor)  # 检查值的类型是否为ShardedTensor
            # 检查本地张量是否相同
            self.assertEqual(v1.to_local(), v2.local_tensor())
            # 检查设备是否相同
            self.assertEqual(v1.to_local().device, v2.local_tensor().device)

        # 检查分布式张量和分片张量优化状态字典的值是否相同
        for dtensor_osd_state, sharded_tensor_osd_state in zip(
            dtensor_osd["state"].items(), sharded_tensor_osd["state"].items()
        ):
            # 检查全限定名称是否相同
            self.assertEqual(dtensor_osd_state[0], sharded_tensor_osd_state[0])
            # 遍历超参数字典并检查每个超参数的键和值是否相同
            for dtensor_hyper_param, sharded_tensor_hyper_param in zip(
                dtensor_osd_state[1].items(),
                sharded_tensor_osd_state[1].items(),
            ):
                k1, v1 = dtensor_hyper_param
                k2, v2 = sharded_tensor_hyper_param
                self.assertEqual(k1, k2)  # 检查键是否相同

                if k1 != "step":  # 如果键不是"step"
                    self.assertEqual(type(v1), DTensor)  # 检查值的类型是否为DTensor
                    self.assertEqual(type(v2), ShardedTensor)  # 检查值的类型是否为ShardedTensor
                    # 检查本地张量是否相同
                    self.assertEqual(v1.to_local(), v2.local_tensor())
                    # 检查设备是否相同
                    self.assertEqual(v1.to_local().device, v2.local_tensor().device)
                else:
                    self.assertEqual(v1, v2)  # 如果键是"step"，则直接比较值是否相同

    @with_comms
    # 装饰器：如果 GPU 数量小于 4，则跳过测试
    @skip_if_lt_x_gpu(4)
    # 参数化测试，测试 offload_to_cpu 参数为 True 和 False 两种情况
    @parametrize("offload_to_cpu", [True, False])
    def test_dtensor_sharded_optim_load_state_dict(self, offload_to_cpu):
        # 初始化一个二维设备网格
        mesh_2d = init_device_mesh(self.device_type, (2, self.world_size // 2))
        # 创建模型和优化器
        model, optim = self._create_model(mesh_2d)

        # 设置模型的状态字典类型为 SHARDED_STATE_DICT
        FSDP.set_state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
            optim_state_dict_config=ShardedOptimStateDictConfig(
                offload_to_cpu=offload_to_cpu
            ),
        )

        # 创建一个内存中的字节流对象
        checkpoint = io.BytesIO()
        # 将模型和优化器的优化器状态字典保存到字节流对象中
        torch.save(FSDP.optim_state_dict(model, optim), checkpoint)
        # 深度拷贝，保存当前的优化器状态字典以便后续比较
        ref_optim_state_dict = deepcopy(FSDP.optim_state_dict(model, optim))

        # 更新参数，使得 FSDP.optim_state_dict() 和 ref_optim_state_dict 不同
        model(model.get_input()).sum().backward()
        optim.step()

        # 从字节流中加载 ref_optim_state_dict
        checkpoint.seek(0)
        load_ref_optim_state_dict = torch.load(checkpoint)
        # 加载 ref_optim_state_dict 到优化器中
        optim.load_state_dict(
            FSDP.optim_state_dict_to_load(model, optim, load_ref_optim_state_dict)
        )
        # 获取新的优化器状态字典
        new_optim_state_dict = FSDP.optim_state_dict(model, optim)

        # 检查 new_optim_state_dict 是否与 ref_optim_state_dict 相同
        for new_optim_state_dict_item, ref_optim_state_dict_item in zip(
            new_optim_state_dict["state"].items(),
            ref_optim_state_dict["state"].items(),
        ):
            # 检查 FQN 是否相同
            self.assertEqual(new_optim_state_dict_item[0], ref_optim_state_dict_item[0])
            # 检查每个超参数是否相同
            for new_optim_hyper_param, ref_optim_hyper_param in zip(
                new_optim_state_dict_item[1].items(),
                ref_optim_state_dict_item[1].items(),
            ):
                k1, v1 = new_optim_hyper_param
                k2, v2 = ref_optim_hyper_param
                # 检查键是否相同
                self.assertEqual(k1, k2)
                # 检查 DTensor 是否相同
                self.assertEqual(v1, v2)
                # 如果键不是 "step"，则检查值的类型是否为 DTensor
                if k1 != "step":
                    self.assertEqual(type(v1), DTensor)
                    self.assertEqual(type(v2), DTensor)

    # 装饰器：包装测试方法以使用通信
    @with_comms
    # 装饰器：如果 GPU 数量小于 4，则跳过测试
    @skip_if_lt_x_gpu(4)
    # 参数化测试，测试 offload_to_cpu 参数为 True 和 False 两种情况
    @parametrize("offload_to_cpu", [True, False])
    # 定义测试方法，用于加载分布式张量模型的状态字典
    def test_dtensor_sharded_model_load_state_dict(self, offload_to_cpu):
        # 初始化一个二维设备网格，根据设备类型和世界大小分配设备
        mesh_2d = init_device_mesh(self.device_type, (2, self.world_size // 2))
        # 创建模型和优化器
        model, optim = self._create_model(mesh_2d)

        # 设置模型的状态字典类型为分片状态字典
        FSDP.set_state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
            state_dict_config=ShardedStateDictConfig(offload_to_cpu=offload_to_cpu),
        )

        # 创建一个内存中的字节流对象作为检查点
        checkpoint = io.BytesIO()
        # 将模型的状态字典保存到检查点
        torch.save(model.state_dict(), checkpoint)
        # 深拷贝当前状态字典，以便与下面加载的状态字典进行比较
        ref_state_dict = deepcopy(model.state_dict())

        # 更新模型参数，使得模型的状态字典与ref_state_dict不同
        model(model.get_input()).sum().backward()
        optim.step()

        # 将ref_state_dict从检查点中加载回来
        checkpoint.seek(0)
        load_ref_state_dict = torch.load(checkpoint)
        model.load_state_dict(load_ref_state_dict)
        # 获取新加载的状态字典
        new_state_dict = model.state_dict()

        # 检查new_state_dict是否与ref_state_dict相同
        for (k1, v1), (k2, v2) in zip(ref_state_dict.items(), new_state_dict.items()):
            # 检查fqn是否相同
            self.assertEqual(k1, k2)

            # 检查v1和v2的类型是否为DTensor
            self.assertEqual(type(v1), DTensor)
            self.assertEqual(type(v2), DTensor)
            # 检查DTensor是否相同
            self.assertEqual(v1, v2)

    # 使用装饰器设置通信环境，并跳过GPU少于4个的情况
    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_root_module_is_not_FSDP(self):
        # 定义一个名为 FakeMPModel 的内部类，继承自 torch.nn.Module
        class FakeMPModel(torch.nn.Module):
            # 初始化方法，接受设备网格参数 device_mesh
            def __init__(self, device_mesh):
                super().__init__()
                # 设置随机种子为0
                torch.manual_seed(0)
                # 创建一个 FSDP 对象，作为 self.dense 的属性
                self.dense = FSDP(
                    DenseModel().cuda(),
                    use_orig_params=True,
                    sharding_strategy=ShardingStrategy.HYBRID_SHARD,
                    device_mesh=device_mesh,
                )
                # 如果当前进程的分布式进程编号为0
                if dist.get_rank() == 0:
                    # 创建 self.sparse0 属性，包含线性层和 ReLU 激活函数
                    self.sparse0 = nn.Sequential(nn.Linear(8, 8), nn.ReLU())
                else:
                    # 创建 self.sparse1 属性，包含线性层和 ReLU 激活函数
                    self.sparse1 = nn.Sequential(nn.Linear(8, 8), nn.ReLU())

            # 前向传播方法，接受输入 x
            def forward(self, x):
                # 如果当前进程的分布式进程编号为0
                if dist.get_rank() == 0:
                    # 使用 self.sparse0 对 x 进行前向传播计算
                    sparse = self.sparse0(x)
                else:
                    # 使用 self.sparse1 对 x 进行前向传播计算
                    sparse = self.sparse1(x)
                # 对 sparse 进行全局同步操作
                dist.all_reduce(sparse)
                # 返回 self.dense 对 sparse 的计算结果
                return self.dense(sparse)

        # 初始化设备网格为 2D 的网格
        mesh_2d = init_device_mesh(self.device_type, (2, self.world_size // 2))
        # 创建 FakeMPModel 的实例 model，并将其移到 CUDA 设备上
        model = FakeMPModel(device_mesh=mesh_2d).cuda()
        # 使用 Adam 优化器来优化 model 的参数，学习率为 0.01
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)

        # 创建一个大小为 [5, 8] 的随机张量 batch，移动到 CUDA 设备上
        batch = torch.rand(5, 8, device=torch.device("cuda"))
        # 对 model 进行前向传播计算，并对计算结果求和后反向传播
        model(batch).sum().backward()
        # 使用优化器执行一步优化操作
        optim.step()
        # 获取优化器的状态字典
        osd = optim.state_dict()

        # 使用 FSDP 提供的状态字典类型来保存 model 的状态字典，类型为 SHARDED_STATE_DICT
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            osd = FSDP.optim_state_dict(model, optim, osd)

        # 遍历优化器状态字典中的参数和状态
        for param, state in osd["state"].items():
            # 如果参数名中包含 "dense"
            if "dense" in param:
                # 断言 exp_avg 的类型为 DTensor
                self.assertIsInstance(state["exp_avg"], DTensor)
                # 断言 exp_avg_sq 的类型为 DTensor
                self.assertIsInstance(state["exp_avg_sq"], DTensor)
                # 断言 exp_avg 的分布策略为 (Replicate(), Shard(0))
                self.assertEqual(state["exp_avg"].placements, (Replicate(), Shard(0)))
                # 断言 exp_avg_sq 的分布策略为 (Replicate(), Shard(0))
                self.assertEqual(
                    state["exp_avg_sq"].placements, (Replicate(), Shard(0))
                )
            else:
                # 断言 exp_avg 的类型为 torch.Tensor
                self.assertIsInstance(state["exp_avg"], torch.Tensor)
                # 断言 exp_avg_sq 的类型为 torch.Tensor
                self.assertIsInstance(state["exp_avg_sq"], torch.Tensor)
# 实例化带有参数的测试用例 TestHSDPWithDeviceMeshAndDTensor
instantiate_parametrized_tests(TestHSDPWithDeviceMeshAndDTensor)

# 如果当前脚本作为主程序运行，则执行测试
if __name__ == "__main__":
    # 运行测试
    run_tests()
```