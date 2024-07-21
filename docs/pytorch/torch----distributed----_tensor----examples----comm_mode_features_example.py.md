# `.\pytorch\torch\distributed\_tensor\examples\comm_mode_features_example.py`

```
# 导入标准库 os
import os

# 导入类型提示相关库
from typing import Callable, Dict

# 导入 PyTorch 库
import torch

# 导入 PyTorch 分布式张量相关模块
from torch.distributed._tensor import DeviceMesh
from torch.distributed._tensor.debug import CommDebugMode
from torch.distributed._tensor.examples.comm_mode_features_example_argparser import args

# 导入 PyTorch 并行化模块
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)

# 导入 PyTorch 分布式张量测试相关库
from torch.testing._internal.distributed._tensor.common_dtensor import (
    MLPModule,
    MLPStacked,
    ModelArgs,
    NUM_DEVICES,
    Transformer,
)

# 定义函数，获取当前可用的设备类型
def get_device_type() -> str:
    return (
        "cuda"
        if torch.cuda.is_available() and torch.cuda.device_count() >= 4
        else "cpu"
    )

# 获取 C10D 函数功能
c10d_functional = torch.ops.c10d_functional

# 获取 aten 操作
aten = torch.ops.aten

# 支持的操作列表
supported_ops = [aten.view.default, aten._to_copy.default]

# 定义 CommDebugModeExample 类
class CommDebugModeExample:
    """
    Checks if the set of keys in ground truth dictionary and the set
    produced in advanced_module_tracker are in the same order
    """

    # 类的初始化方法
    def __init__(self, world_size: int, rank: int) -> None:
        self.world_size = world_size
        self.rank = rank
        self.device_type = get_device_type()  # 获取当前设备类型

    # 测试 MLP 分布式分片显示方法
    def test_MLP_distributed_sharding_display(self) -> None:
        """
        Example of obtaining all module's FQN and parameters for a given distributed model and printing the sharding info

        Expected output:
        MLPModule.net1.weight: (Shard(dim=0),)
        MLPModule.net1.bias: (Shard(dim=0),)
        MLPModule.net2.weight: (Shard(dim=1),)
        MLPModule.net2.bias: (Replicate(),)
        """
        device_mesh = DeviceMesh(
            self.device_type,  # 设备类型
            torch.arange(0, NUM_DEVICES),  # 设备索引范围
        )
        inp_size = [8, 10]
        rng_seed = 0
        torch.manual_seed(rng_seed)
        inp = torch.rand(*inp_size, device=self.device_type)  # 在指定设备上创建随机张量
        model = MLPModule(self.device_type)  # 创建 MLP 模型对象

        LR = 0.25

        parallelize_plan = {
            "net1": ColwiseParallel(),   # 网络 'net1' 使用列并行化
            "net2": RowwiseParallel(),   # 网络 'net2' 使用行并行化
        }

        model = parallelize_module(model, device_mesh, parallelize_plan)  # 应用并行化计划到模型

        comm_mode = CommDebugMode()  # 创建通信调试模式对象

        with comm_mode:  # 进入通信调试模式上下文
            output_tp = model(inp)  # 在模型上执行前向传播
            output_tp.sum().backward()  # 对输出张量求和并执行反向传播

        comm_mode.print_sharding_info()  # 打印分片信息
    def test_MLPStacked_distributed_sharding_display(self) -> None:
        """
        Example of obtaining all module's FQN and parameters for a given
        distributed model with nested modules and printing the sharding info

        Expected output:
        MLPStacked.layers.0.net1.weight: (Shard(dim=0),)
        MLPStacked.layers.0.net1.bias: (Shard(dim=0),)
        MLPStacked.layers.0.net2.weight: (Shard(dim=1),)
        MLPStacked.layers.0.net2.bias: (Replicate(),)
        MLPStacked.layers.1.net1.weight: (Shard(dim=0),)
        MLPStacked.layers.1.net1.bias: (Shard(dim=0),)
        MLPStacked.layers.1.net2.weight: (Shard(dim=1),)
        MLPStacked.layers.1.net2.bias: (Replicate(),)
        """
        # 定义设备的网格
        device_mesh = DeviceMesh(
            self.device_type,  # 使用指定设备类型
            torch.arange(0, NUM_DEVICES),  # 创建设备索引的张量
        )
        # 输入大小
        inp_size = [8, 10]
        # 随机数种子
        rng_seed = 0
        # 设置随机种子
        torch.manual_seed(rng_seed)
        # 创建输入张量
        inp = torch.rand(*inp_size, device=self.device_type)
        # 创建 MLPStacked 模型实例
        model = MLPStacked(self.device_type)

        LR = 0.25  # 学习率

        # 并行化计划，指定每个子模块的并行策略
        parallelize_plan = {
            "MLPStacked.layers.0.net1": ColwiseParallel(),  # 第一层的 net1 列并行
            "MLPStacked.layers.0.net2": RowwiseParallel(),  # 第一层的 net2 行并行
            "MLPStacked.layers.1.net1": ColwiseParallel(),  # 第二层的 net1 列并行
            "MLPStacked.layers.1.net2": RowwiseParallel(),  # 第二层的 net2 行并行
        }

        # 应用并行化计划到模型
        model = parallelize_module(model, device_mesh, parallelize_plan)

        # 通信调试模式
        comm_mode = CommDebugMode()

        # 进入通信调试模式的上下文
        with comm_mode:
            # 使用模型进行前向传播
            output_tp = model(inp)
            # 计算输出的和并反向传播
            output_tp.sum().backward()

        # 打印分片信息
        comm_mode.print_sharding_info()

    def test_MLP_module_tracing(self) -> None:
        """
        Example code to demonstrate CommModeDebug's module level tracing using a MLP model.
        Prints a table of module level collective tracing information and logs table to output.txt

        Expected Output
        Global
        *c10d_functional.all_reduce: 1
        MLPModule
            *c10d_functional.all_reduce: 1
            MLPModule.net1
            MLPModule.relu
            MLPModule.net2
            *c10d_functional.all_reduce: 1
        """

        # 定义设备的网格
        device_mesh = DeviceMesh(
            self.device_type,  # 使用指定设备类型
            torch.arange(0, NUM_DEVICES),  # 创建设备索引的张量
        )
        # 输入大小
        inp_size = [8, 10]
        # 随机数种子
        rng_seed = 0
        # 设置随机种子
        torch.manual_seed(rng_seed)
        # 创建输入张量
        inp = torch.rand(*inp_size, device=self.device_type)
        # 创建 MLPModule 模型实例
        model = MLPModule(self.device_type)

        LR = 0.25  # 学习率

        # 并行化计划，指定每个子模块的并行策略
        parallelize_plan = {
            "net1": ColwiseParallel(),  # net1 列并行
            "net2": RowwiseParallel(),  # net2 行并行
        }

        # 应用并行化计划到模型
        model = parallelize_module(model, device_mesh, parallelize_plan)

        # 通信调试模式
        comm_mode = CommDebugMode()

        # 进入通信调试模式的上下文
        with comm_mode:
            # 使用模型进行前向传播
            output_tp = model(inp)
            # 计算输出的和并反向传播
            output_tp.sum().backward()

        # 生成模块级别追踪信息表并打印
        print(comm_mode.generate_module_tracing_table())
        # 将模块级别追踪信息表记录到文件
        comm_mode.log_module_tracing_table_to_file()
# 设置随机种子为0，确保实验的随机性可重复
torch.manual_seed(0)
# 创建 CommDebugModeExample 类的实例，用于执行通信调试模式示例
instantiated_test = CommDebugModeExample(world_size, rank)
# 字典，将示例函数名称映射到对应的函数对象
name_to_example_code: Dict[str, Callable[[], None]] = {
    "MLP_distributed_sharding_display": instantiated_test.test_MLP_distributed_sharding_display,
    "MLPStacked_distributed_sharding_display": instantiated_test.test_MLPStacked_distributed_sharding_display,
    "MLP_module_tracing": instantiated_test.test_MLP_module_tracing,
    "transformer_module_tracing": instantiated_test.test_transformer_module_tracing,
}

# 根据传入的示例名称，执行对应的示例函数
name_to_example_code[example_name]()
```