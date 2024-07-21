# `.\pytorch\torch\distributed\_tensor\examples\checkpoint_example.py`

```
"""
The following example contains a simple MLP model that uses
different DTensor layouts, and use the checkpointing API to
checkpoint save/load the model.
"""
import os  # 导入操作系统功能模块
from typing import cast, List  # 导入类型提示相关模块

import torch  # 导入PyTorch深度学习框架
import torch.distributed as dist  # 导入PyTorch分布式训练相关模块
import torch.multiprocessing as mp  # 导入PyTorch多进程处理模块
import torch.nn as nn  # 导入PyTorch神经网络模块
import torch.nn.functional as F  # 导入PyTorch的函数式接口
from torch.distributed._tensor import (  # 导入PyTorch分布式张量相关模块
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    DTensor,
    Replicate,
    Shard,
)
from torch.distributed._tensor.placement_types import Placement  # 导入分布式张量的放置类型
from torch.distributed.tensor.parallel import ColwiseParallel, parallelize_module  # 导入张量并行处理相关模块


class SimpleMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = torch.nn.Linear(5, 128)  # 第一个全连接层，输入维度5，输出维度128
        self.relu = torch.nn.ReLU()  # ReLU激活函数
        self.net2 = torch.nn.Linear(128, 12)  # 第二个全连接层，输入维度128，输出维度12

    def forward(self, x):
        return self.net2(F.relu(self.net1(x)))  # 前向传播函数，包括两个全连接层和ReLU激活函数的组合


def gen_tensor_parallel_model(model: nn.Module, mesh: DeviceMesh) -> nn.Module:
    """
    generates a nn.Module where parameters are sharded in the tensor-parallel
    fashion.
    """
    # 将模型参数按照张量并行的方式分片
    return parallelize_module(
        model,
        mesh,
        {"net1": ColwiseParallel()},  # net1 层使用列并行策略
    )


def gen_partial_replicate_2d(model: nn.Module, mesh: DeviceMesh) -> nn.Module:
    """
    generates a nn.Module where parameters are replicated in the first mesh
    dimension, and sharded in the second mesh dimension.
    """

    def parallel_fn(name, module, device_mesh):
        assert device_mesh.ndim == 2  # 断言：设备网格的维度为2
        if isinstance(module, torch.nn.Linear) and name == "net1":
            for name, param in module.named_parameters():
                # 对于 net1 层的参数，按照 [Replicate(), Shard(0)] 的方式复制和分片
                dist_param = torch.nn.Parameter(
                    distribute_tensor(param, device_mesh, [Replicate(), Shard(0)])
                )
                module.register_parameter(name, dist_param)
        elif isinstance(module, torch.nn.Linear) and name == "net2":
            for name, param in module.named_parameters():
                # 对于 net2 层的参数，根据参数名称选择 [Replicate(), Shard(1)] 或者 [Replicate(), Replicate()] 的方式复制和分片
                dist_spec = (
                    [Replicate(), Shard(1)]
                    if name == "weight"
                    else [Replicate(), Replicate()]
                )
                dist_param = torch.nn.Parameter(
                    distribute_tensor(param, device_mesh, dist_spec)
                )
                module.register_parameter(name, dist_param)

    # 标记输入在网格上复制
    def input_fn(mod, inputs, device_mesh):
        return DTensor.from_local(inputs[0], device_mesh, [Replicate(), Replicate()])

    # 标记输出在网格上本地化
    def output_fn(mod, outputs, device_mesh):
        assert isinstance(outputs, DTensor)
        return outputs.to_local()

    # 将模型按照指定的分布式策略进行模块分布
    return distribute_module(
        model,
        mesh,
        partition_fn=parallel_fn,
        input_fn=input_fn,
        output_fn=output_fn,
    )


def gen_model_param_in_submesh(model: nn.Module, sub_mesh: DeviceMesh) -> nn.Module:
    """
    generates a nn.Module where parameters are in the submesh
    """
    """
    generates a nn.Module where parameters are sharded/replicated only on a
    sub-mesh (i.e. mesh(0, 2) in a world size of 4)
    """

    # 定义并行处理函数，根据模块的类型和名称分配参数到特定的设备子网格
    def parallel_fn(name, module, device_mesh):
        assert device_mesh.ndim == 1
        # 如果模块是线性层并且名称是"net1"
        if isinstance(module, torch.nn.Linear) and name == "net1":
            # 遍历模块的参数，将其分布到指定的设备子网格
            for name, param in module.named_parameters():
                dist_param = torch.nn.Parameter(
                    distribute_tensor(param, device_mesh, [Shard(0)])
                )
                module.register_parameter(name, dist_param)
        # 如果模块是线性层并且名称是"net2"
        elif isinstance(module, torch.nn.Linear) and name == "net2":
            # 遍历模块的参数，根据名称将其分布到不同的设备子网格
            for name, param in module.named_parameters():
                dist_spec = cast(
                    List[Placement], [Shard(1)] if name == "weight" else [Replicate()]
                )
                dist_param = torch.nn.Parameter(
                    distribute_tensor(param, device_mesh, dist_spec)
                )
                module.register_parameter(name, dist_param)

    # 将输入数据复制到设备子网格上
    def input_fn(mod, inputs, device_mesh):
        return DTensor.from_local(inputs[0], device_mesh, [Replicate()])

    # 将输出数据从设备子网格恢复到本地
    def output_fn(mod, outputs, device_mesh):
        assert isinstance(outputs, DTensor)
        return outputs.to_local()

    # 返回通过分布模块进行处理后的模型
    return distribute_module(
        model,
        sub_mesh,
        partition_fn=parallel_fn,
        input_fn=input_fn,
        output_fn=output_fn,
    )
# 使用 torch 模块提供的 nn.Module 类定义 checkpoint 函数，它接受一个 nn.Module 类型的模型和一个 DeviceMesh 类型的参数，并返回一个 nn.Module 类型的对象
def checkpoint(model: nn.Module, mesh: DeviceMesh) -> nn.Module:  # type: ignore[empty-body]
    """
    checkpoint save/load models with DTensor parameters
    """
    # TODO: implement this checkpoint save/load example
    # 该函数目前仅有一个 TODO 注释，暂未实现具体功能
    pass


# 定义一个运行 checkpoint 示例的函数，接受两个参数：rank 和 world_size
def run_checkpoint_example(rank, world_size):
    # 设置环境变量，指定主节点地址和端口
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # 初始化进程组，使用 gloo 后端，设置当前进程的 rank 和进程总数
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # 创建一个 DeviceMesh 对象，使用 "cpu" 设备，其范围为 0 到 world_size-1
    mesh = DeviceMesh("cpu", torch.arange(world_size))

    # 使用 gen_tensor_parallel_model 函数创建一个 tensor 并行模型 model_tp，并在 mesh 上进行分片
    model_tp = gen_tensor_parallel_model(SimpleMLP(), mesh)
    model_tp(torch.rand(5, 5))  # 对模型进行前向传播

    # 创建一个二维的 DeviceMesh 对象 mesh_2d，使用 "cpu" 设备，将原始范围转换为 2x2 的形状
    mesh_2d = DeviceMesh("cpu", torch.arange(world_size).reshape(2, 2))

    # 使用 gen_partial_replicate_2d 函数在 mesh_2d 上创建一个部分复制的模型 model_2d
    model_2d = gen_partial_replicate_2d(SimpleMLP(), mesh_2d)
    model_2d(torch.rand(5, 5))  # 对模型进行前向传播

    # 创建一个子网格 submesh，使用 "cpu" 设备，包含索引 0 和 2
    submesh = DeviceMesh("cpu", [0, 2])

    # 使用 gen_model_param_in_submesh 函数在 submesh 上创建一个模型 model_submesh
    model_submesh = gen_model_param_in_submesh(SimpleMLP(), submesh)
    model_submesh(torch.rand(5, 5))  # 对模型进行前向传播
    print(f"partial replicate model state_dict: {model_submesh.state_dict()}")

    # 调用 checkpoint 函数，将 model_2d 和 mesh 作为参数，返回一个被检查点保存的模型对象
    model = checkpoint(model_2d, mesh)

    # 销毁进程组，结束并释放资源
    dist.destroy_process_group()


# 如果当前脚本被作为主程序运行
if __name__ == "__main__":
    world_size = 4
    # 使用 mp.spawn 方法启动多个进程运行 run_checkpoint_example 函数，每个进程的 rank 从 0 到 world_size-1
    mp.spawn(run_checkpoint_example, args=(world_size,), nprocs=world_size, join=True)
```