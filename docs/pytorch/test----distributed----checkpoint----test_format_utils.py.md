# `.\pytorch\test\distributed\checkpoint\test_format_utils.py`

```py
# Owner(s): ["oncall: distributed"]

# 引入PyTorch库和分布式相关模块
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn

# 引入PyTorch函数库和特定分布式Tensor设备网格初始化
import torch.nn.functional as F
from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.distributed.checkpoint.format_utils import (
    BroadcastingTorchSaveReader,
    dcp_to_torch_save,
    DynamicMetaLoadPlanner,
    torch_save_to_dcp,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir

# 定义一个简单的非均匀模型
class SimpleModelUneven(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # 定义神经网络的层结构
        self.net1 = nn.Linear(5, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 15)
        self.net3 = nn.Linear(15, 30)
        self.net4 = nn.Linear(30, 5)

    # 前向传播函数定义
    def forward(self, x):
        x = F.relu(self.net1(x))
        x = F.relu(self.net2(x))
        x = F.relu(self.net3(x))
        x = self.net4(x)
        return x

    # 获取模型输入数据
    def get_input(self):
        return torch.rand(4, 5, device="cuda")


# 测试格式化工具类
class TestFormatUtils(DTensorTestBase):
    # 测试将dcp格式转换为torch保存格式
    @with_temp_dir
    def test_dcp_to_torch_save(self) -> None:
        model = SimpleModelUneven()
        # 将模型保存为dcp格式
        dcp.save({"model": model}, checkpoint_id=self.temp_dir)

        torch_path = self.temp_dir + "/model.pt"
        # 将dcp格式转换为torch保存格式
        dcp_to_torch_save(self.temp_dir, torch_path)

        # 加载torch保存的模型数据
        loaded_sd = torch.load(torch_path)
        # 断言加载的数据与原始模型状态字典一致
        self.assertEqual(loaded_sd, {"model": model.state_dict()})

    # 测试将torch保存格式转换为dcp格式
    @with_temp_dir
    def test_torch_save_to_dcp(self) -> None:
        model = SimpleModelUneven()
        sd = {"model": model.state_dict()}
        torch_path = self.temp_dir + "/model.pt"
        # 将模型状态字典保存为torch格式
        torch.save(sd, torch_path)

        # 将torch保存格式转换为dcp格式
        torch_save_to_dcp(torch_path, self.temp_dir)

        model = SimpleModelUneven()
        # 加载dcp格式的模型数据
        dcp.load({"model": model}, checkpoint_id=self.temp_dir)

        # 断言加载的模型状态字典与原始保存的一致
        self.assertEqual({"model": model.state_dict()}, sd)

    # 包含通信操作的测试
    @with_comms
    @with_temp_dir
    # 如果GPU数量小于2，则跳过测试
    @skip_if_lt_x_gpu(2)
    def test_online_torch_save_to_dcp(self) -> None:
        """Tests loading a model saved by torch.save directly into a sharded model
        using dcp.load
        """
        # Save a model with torch.save
        # 创建一个简单的模型 SimpleModelUneven 的实例
        model = SimpleModelUneven()
        # 获取模型的状态字典
        sd = {"model": model.state_dict()}

        # 定义保存模型的文件路径
        torch_fn = self.temp_dir + "/model.pt"
        # 如果当前进程的排名为 0，将模型状态字典保存到文件中
        if dist.get_rank() == 0:
            torch.save(sd, torch_fn)
        # 等待所有进程执行到此处再继续
        dist.barrier()

        # Load into a sharded model
        # 初始化设备网格，用于分布式设备配置
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        # 创建一个 SimpleModelUneven 模型的 CUDA 版本，并用 FSDP 进行封装
        model = SimpleModelUneven().cuda()
        model = FSDP(
            model,
            device_mesh=device_mesh,
            use_orig_params=True,
        )
        # 使用 dcp.load 加载模型
        dcp.load(
            {"model": model},
            planner=DynamicMetaLoadPlanner(),
            storage_reader=BroadcastingTorchSaveReader(),
            checkpoint_id=torch_fn,
        )

        # 断言加载后的模型状态字典与保存前的一致
        self.assertEqual(sd["model"], model.state_dict())
# 如果当前脚本被直接执行（而不是被导入为模块），则执行以下代码
if __name__ == "__main__":
    # 调用名为 run_tests 的函数来执行测试
    run_tests()
```