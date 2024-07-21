# `.\pytorch\test\distributed\fsdp\test_fsdp_dtensor_state_dict.py`

```
# Owner(s): ["oncall: distributed"]

# 引入所需模块和类
import io
from copy import deepcopy

import torch
import torch.nn as nn
from torch.distributed._shard.sharded_tensor import ShardedTensor

# 引入相关分布式张量和优化相关的模块
from torch.distributed._tensor import DTensor, Shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import (
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    StateDictType,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

# 引入分布式测试相关的模块和函数
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
)

# Simple and boring model to test interface and some corner cases that do not
# require complicated wrapping strategy.
# 定义一个简单而无聊的模型用于测试接口和一些不需要复杂封装策略的边缘情况。
class TestDummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.net1 = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        self.net2 = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
        self.net3 = nn.Sequential(nn.Linear(32, 64), nn.ReLU())
        self.net4 = nn.Sequential(nn.ReLU(), nn.Linear(64, 8))

    def forward(self, x):
        return self.net4(self.net3(self.net2(self.net1(x))))

    def get_input(self):
        return torch.rand(8, 8, device="cuda")


# Another simple model to test interface, but with uneven layers.
# 定义另一个简单的模型用于测试接口，但是包含不均匀的层次。
class TestDummyModelUneven(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.net1 = nn.Sequential(nn.Linear(5, 10), nn.ReLU())
        self.net2 = nn.Sequential(nn.Linear(10, 15), nn.ReLU())
        self.net3 = nn.Linear(15, 30)
        self.net4 = nn.Sequential(nn.ReLU(), nn.Linear(30, 5))

    def forward(self, x):
        return self.net4(self.net3(self.net2(self.net1(x))))

    def get_input(self):
        return torch.rand(5, 5, device="cuda")


# Test case for using FSDP with device mesh and DTensor.
# 测试在设备网格和分布式张量上使用 FSDP。
class TestFSDPWithDeviceMeshAndDTensor(DTensorTestBase):

    # Helper function to create a model and optimizer for testing.
    # 创建用于测试的模型和优化器的辅助函数。
    def _create_model(self, is_even_sharded_model, device_mesh=None):
        # Instantiate either an even or uneven sharded model.
        # 实例化一个均匀或不均匀的分片模型。
        dummy_model = (
            TestDummyModel() if is_even_sharded_model else TestDummyModelUneven()
        )

        # Wrap the model with FSDP and move it to CUDA.
        # 使用 FSDP 封装模型并将其移动到 CUDA。
        model = FSDP(dummy_model.cuda(), device_mesh=device_mesh)

        # Define an Adam optimizer for the model parameters.
        # 为模型参数定义 Adam 优化器。
        optim = torch.optim.Adam(model.parameters(), lr=0.1)

        # Perform a forward pass and backward pass on the model.
        # 在模型上进行前向传播和反向传播。
        model(model.get_input()).sum().backward()

        # Take a step with the optimizer.
        # 使用优化器进行优化步骤。
        optim.step()

        return model, optim

    # Decorated test function that uses communications and runs on at least 2 GPUs.
    # 使用通信功能，并且至少在两个 GPU 上运行的装饰测试函数。
    @with_comms
    @skip_if_lt_x_gpu(2)
    @parametrize("is_even_sharded_model", [True, False])
    # 使用带有设备网格初始化设备网格
    device_mesh = init_device_mesh(self.device_type, (self.world_size,))
    # 创建模型和优化器
    model, optim = self._create_model(is_even_sharded_model, device_mesh)

    # 设置模型的状态字典类型为 SHARDED_STATE_DICT
    FSDP.set_state_dict_type(
        model,
        StateDictType.SHARDED_STATE_DICT,
    )
    # 获取模型的状态字典
    state_dict = model.state_dict()
    # 获取模型和优化器的优化器状态字典
    optim_state_dict = FSDP.optim_state_dict(model, optim)

    # 验证模型状态字典中的每个值的类型和分布情况
    for v in state_dict.values():
        self.assertEqual(type(v), DTensor)
        self.assertEqual(len(v.placements), 1)
        self.assertEqual(v.placements[0], (Shard(dim=0)))
        self.assertEqual(v.device_mesh, device_mesh)

    # 验证优化器状态字典中每个状态项的类型和分布情况
    for state in optim_state_dict["state"].values():
        for k, v in state.items():
            if k != "step":
                self.assertEqual(type(v), DTensor)
                self.assertEqual(len(v.placements), 1)
                self.assertEqual(v.placements[0], (Shard(dim=0)))
                self.assertEqual(v.device_mesh, device_mesh)

    # 获取模型的状态字典类型
    state_dict_type = FSDP.get_state_dict_type(model)
    # 如果初始化 FSDP 时使用了 device_mesh，并且 StateDictType 设置为 SHARDED_STATE_DICT，
    # 那么字段 _use_dtensor 将自动设置为 True。
    self.assertEqual(state_dict_type.state_dict_config._use_dtensor, True)
    self.assertEqual(state_dict_type.optim_state_dict_config._use_dtensor, True)
    ):
        # 初始化设备网格，根据设备类型和世界大小创建
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        # 创建模型和优化器
        model, optim = self._create_model(is_even_sharded_model, device_mesh)

        # 设置模型的状态字典类型为SHARDED_STATE_DICT，并且配置优化器状态字典
        FSDP.set_state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
            optim_state_dict_config=ShardedOptimStateDictConfig(
                offload_to_cpu=offload_to_cpu
            ),
        )

        # 创建一个字节流对象用于保存检查点
        checkpoint = io.BytesIO()
        # 将模型和优化器的优化状态字典保存到checkpoint中
        torch.save(FSDP.optim_state_dict(model, optim), checkpoint)
        # 深拷贝以保存当前优化状态字典，以便与下面加载回来的优化状态字典进行比较
        ref_optim_state_dict = deepcopy(FSDP.optim_state_dict(model, optim))

        # 更新模型参数，使得FSDP.optim_state_dict()与ref_optim_state_dict不同
        model(model.get_input()).sum().backward()
        optim.step()

        # 加载回ref_optim_state_dict
        checkpoint.seek(0)
        load_ref_optim_state_dict = torch.load(checkpoint)
        optim.load_state_dict(
            FSDP.optim_state_dict_to_load(model, optim, load_ref_optim_state_dict)
        )
        # 获取新的优化状态字典
        new_optim_state_dict = FSDP.optim_state_dict(model, optim)

        # 检查new_optim_state_dict是否与ref_optim_state_dict相同
        for new_optim_state_dict_item, ref_optim_state_dict_item in zip(
            new_optim_state_dict["state"].items(),
            ref_optim_state_dict["state"].items(),
        ):
            # 检查FQN是否相同
            self.assertEqual(new_optim_state_dict_item[0], ref_optim_state_dict_item[0])
            for new_optim_hyper_param, ref_optim_hyper_param in zip(
                new_optim_state_dict_item[1].items(),
                ref_optim_state_dict_item[1].items(),
            ):
                k1, v1 = new_optim_hyper_param
                k2, v2 = ref_optim_hyper_param

                # 检查键是否相同
                self.assertEqual(k1, k2)
                # 检查值是否相同
                self.assertEqual(v1, v2)

                # 如果键名不是"step"，则检查值的类型是否为DTensor
                if k1 != "step":
                    self.assertEqual(type(v1), DTensor)
                    self.assertEqual(type(v2), DTensor)
    ):
        # 使用 init_device_mesh 函数初始化设备网格，传入设备类型和世界大小参数
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        # 调用 _create_model 方法创建模型和优化器实例
        model, optim = self._create_model(is_even_sharded_model, device_mesh)

        # 设置模型的状态字典类型为 SHARDED_STATE_DICT
        FSDP.set_state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
            state_dict_config=ShardedStateDictConfig(offload_to_cpu=offload_to_cpu),
        )

        # 创建一个空的字节流对象 checkpoint
        checkpoint = io.BytesIO()
        # 将模型的状态字典保存到 checkpoint 中
        torch.save(model.state_dict(), checkpoint)
        # 深拷贝当前的模型状态字典 ref_state_dict 用于后续比较
        ref_state_dict = deepcopy(model.state_dict())

        # 更新模型的参数，使得模型状态字典与 ref_state_dict 不同
        model(model.get_input()).sum().backward()
        optim.step()

        # 将 ref_state_dict 加载回模型中
        checkpoint.seek(0)
        load_ref_state_dict = torch.load(checkpoint)
        model.load_state_dict(load_ref_state_dict)
        new_state_dict = model.state_dict()

        # 检查 new_state_dict 是否与 ref_state_dict 相同
        for (k1, v1), (k2, v2) in zip(ref_state_dict.items(), new_state_dict.items()):
            # 检查键名 k1 和 k2 是否相同
            self.assertEqual(k1, k2)

            # 检查 v1 和 v2 的类型是否为 DTensor
            self.assertEqual(type(v1), DTensor)
            self.assertEqual(type(v2), DTensor)
            # 检查 v1 和 v2 是否相等
            self.assertEqual(v1, v2)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_raises_warning_or_errors(self):
        # 初始化设备网格
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        # 创建模型和优化器实例
        model, optim = self._create_model(
            is_even_sharded_model=True, device_mesh=device_mesh
        )
        # 对模型的输入进行前向传播和反向传播
        model(model.get_input()).sum().backward()
        optim.step()

        # 检查是否抛出 RuntimeError，并输出相应的错误信息
        with self.assertRaisesRegex(
            RuntimeError, "DeviceMesh is not compatible with LOCAL_STATE_DICT."
        ):
            # 设置模型的状态字典类型为 LOCAL_STATE_DICT，并获取模型的状态字典
            with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
                state_dict = model.state_dict()

        # 检查是否抛出 RuntimeError，并输出相应的错误信息
        with self.assertRaisesRegex(
            RuntimeError, "DeviceMesh is not compatible with LOCAL_STATE_DICT."
        ):
            # 设置模型的状态字典类型为 LOCAL_STATE_DICT，并获取模型和优化器的状态字典
            with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
                optim_state_dict = FSDP.optim_state_dict(model, optim)
# 实例化带有参数化测试的 TestFSDPWithDeviceMeshAndDTensor 类
instantiate_parametrized_tests(TestFSDPWithDeviceMeshAndDTensor)
# 如果当前脚本作为主程序执行，则运行测试函数
if __name__ == "__main__":
    run_tests()
```