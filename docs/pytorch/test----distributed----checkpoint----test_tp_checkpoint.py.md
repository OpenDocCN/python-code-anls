# `.\pytorch\test\distributed\checkpoint\test_tp_checkpoint.py`

```
# 导入所需模块和类，声明代码的所有者信息
from copy import deepcopy  # 导入深拷贝函数

import torch  # 导入 PyTorch 模块
import torch.distributed.checkpoint as DCP  # 导入分布式检查点函数

from torch.distributed._tensor import init_device_mesh  # 导入设备网格初始化函数

# 导入默认的加载和保存计划者类
from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)

# 导入并行模块，包括列并行和行并行
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)

# 导入运行测试的函数
from torch.testing._internal.common_utils import run_tests

# 导入分布式测试中的公共数据张量测试基类、MLP 模型、GPU 数量检测函数和通信装饰器
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    MLPModule,
    skip_if_lt_x_gpu,
    with_comms,
)

# 导入用于临时目录操作的函数装饰器
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


class UnevenShardedModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        torch.manual_seed(5)
        self.net1 = torch.nn.Linear(5, 10, device=device)  # 创建线性层网络 net1
        self.relu = torch.nn.ReLU()  # 创建 ReLU 激活函数对象
        self.net2 = torch.nn.Linear(10, 15, device=device)  # 创建线性层网络 net2
        self.net3 = torch.nn.Linear(15, 1, device=device)  # 创建线性层网络 net3

    def forward(self, x):
        return self.net3(self.net2(self.relu(self.net1(x))))
        # 执行前向传播：net1 -> ReLU -> net2 -> net3


class TestTpCheckpoint(DTensorTestBase):
    @with_comms  # 使用通信装饰器，确保在测试期间通信已经设置
    @skip_if_lt_x_gpu(2)  # 如果 GPU 数量小于 2，则跳过测试
    @with_temp_dir  # 使用临时目录装饰器，在测试期间创建临时目录
    def test_tp_checkpoint(self):
        CHECKPOINT_DIR = self.temp_dir  # 获取临时目录路径，用于存储检查点文件
        mesh_shpe = (self.world_size,)  # 定义一个元组，表示网格的形状，由世界大小确定
        tp_mesh = init_device_mesh(self.device_type, mesh_shpe)  # 初始化设备网格，根据设备类型和网格形状

        # 创建模型并将其移动到具有特定ID的GPU上
        model = MLPModule(self.device_type).cuda(self.rank)
        # 并行化模块，基于给定的并行计划
        parallelize_plan = {
            "net1": ColwiseParallel(),
            "net2": RowwiseParallel(),
        }
        model = parallelize_module(model, tp_mesh, parallelize_plan)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
        original_state_dict = deepcopy(model.state_dict())  # 复制当前模型的状态字典

        # 使用DCP保存模型的状态字典到文件系统中的指定目录
        DCP.save_state_dict(
            state_dict=original_state_dict,
            storage_writer=DCP.FileSystemWriter(CHECKPOINT_DIR),
            planner=DefaultSavePlanner(),
        )

        # 更新参数，使得模型的状态字典与原始状态字典不同
        torch.manual_seed(0)
        inp = torch.rand(20, 10).cuda(self.rank)
        output = model(inp)
        output.sum().backward()
        optimizer.step()
        state_dict = model.state_dict()

        # 在从检查点加载之前，确保当前模型参数与原始状态字典不同
        for param1, param2 in zip(original_state_dict.values(), state_dict.values()):
            self.assertNotEqual(param1.to_local(), param2.to_local())

        # 使用DCP从文件系统中的指定目录加载模型的状态字典
        DCP.load_state_dict(
            state_dict=state_dict,
            storage_reader=DCP.FileSystemReader(CHECKPOINT_DIR),
            planner=DefaultLoadPlanner(),
        )

        # 确保从检查点加载后，当前模型参数与原始状态字典相同
        for param1, param2 in zip(original_state_dict.values(), state_dict.values()):
            self.assertEqual(param1.to_local(), param2.to_local())

    @with_comms
    @skip_if_lt_x_gpu(2)
    @with_temp_dir
    # 定义测试函数，用于测试在元设备上加载检查点
    def test_tp_checkpoint_load_on_meta_device(self):
        # 设置检查点存储目录为临时目录
        CHECKPOINT_DIR = self.temp_dir
        # 定义网格形状为 (self.world_size,)
        mesh_shpe = (self.world_size,)
        # 初始化设备网格并返回
        tp_mesh = init_device_mesh(self.device_type, mesh_shpe)

        # 创建 UnevenShardedModel 模型实例，并将其移动到 GPU 上，使用 self.rank 作为 GPU 设备 id
        model = UnevenShardedModel(self.device_type).cuda(self.rank)

        # 定义并行化计划，按照给定的并行化风格对模块进行并行化处理
        parallelize_plan = {
            "net1": RowwiseParallel(),
            "net2": ColwiseParallel(),
        }
        model = parallelize_module(model, tp_mesh, parallelize_plan=parallelize_plan)

        # 复制模型的状态字典，以备后续比较使用
        original_state_dict = deepcopy(model.state_dict())

        # 使用 DCP.save_state_dict 将模型的原始状态字典保存到指定的 CHECKPOINT_DIR 中
        DCP.save_state_dict(
            state_dict=original_state_dict,
            storage_writer=DCP.FileSystemWriter(CHECKPOINT_DIR),
        )

        # 创建另一个 UnevenShardedModel 模型实例 model2，并在元设备上进行并行化处理
        model2 = parallelize_module(
            UnevenShardedModel("meta"), tp_mesh, parallelize_plan=parallelize_plan
        )
        
        # 获取模型 model2 的状态字典，准备加载检查点
        state_dict_to_load = model2.state_dict()

        # 使用 DCP.load_state_dict 从 CHECKPOINT_DIR 中读取检查点并加载到 model2 中
        DCP.load_state_dict(
            state_dict=state_dict_to_load,
            storage_reader=DCP.FileSystemReader(CHECKPOINT_DIR),
        )
        
        # 将加载后的状态字典 state_dict_to_load 再次加载到 model2 中
        model2.load_state_dict(state_dict_to_load, assign=True)
        
        # 获取加载后的模型 model2 的最终状态字典
        state_dict_after_load = model2.state_dict()

        # 检查加载后的状态字典 state_dict_after_load 是否与原始状态字典 original_state_dict 相等
        for param1, param2 in zip(
            original_state_dict.values(), state_dict_after_load.values()
        ):
            self.assertEqual(param1, param2)
# 如果当前模块被直接运行而不是被导入到其他模块中，则执行下面的代码块
if __name__ == "__main__":
    # 调用运行测试函数，用于执行模块的测试用例
    run_tests()
```