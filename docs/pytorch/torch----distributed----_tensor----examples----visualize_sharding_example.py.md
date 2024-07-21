# `.\pytorch\torch\distributed\_tensor\examples\visualize_sharding_example.py`

```
# 导入必要的库
import os

# 导入 PyTorch 库
import torch
from torch.distributed._tensor import DeviceMesh, distribute_tensor, Replicate, Shard
from torch.distributed._tensor.debug.visualize_sharding import visualize_sharding

# 从环境变量中获取全局变量 WORLD_SIZE 和 RANK
world_size = int(os.environ["WORLD_SIZE"])  # 获取全局变量 WORLD_SIZE 并转换为整数
rank = int(os.environ["RANK"])  # 获取全局变量 RANK 并转换为整数

# Example 1: 创建一个随机张量，分发到指定的设备网格上，并可视化分片情况
tensor = torch.randn(4, 4)  # 创建一个形状为 (4, 4) 的随机张量
mesh = DeviceMesh("cuda", list(range(world_size)))  # 创建一个 CUDA 设备网格
dtensor = distribute_tensor(tensor, mesh, [Shard(dim=1)])  # 在设备网格上分发张量，并指定沿维度 1 进行分片
visualize_sharding(dtensor)  # 可视化张量分片结果

# Example 2: 创建另一个随机张量，分发到设备网格上不同的维度，并可视化分片情况
tensor = torch.randn(4, 4)  # 创建一个形状为 (4, 4) 的随机张量
mesh = DeviceMesh("cuda", list(range(world_size)))  # 创建一个 CUDA 设备网格
dtensor = distribute_tensor(tensor, mesh, [Shard(dim=0)])  # 在设备网格上分发张量，并指定沿维度 0 进行分片
visualize_sharding(dtensor)  # 可视化张量分片结果

# Example 3: 创建另一个随机张量，分发到自定义的设备网格上，并可视化分片情况
tensor = torch.randn(4, 4)  # 创建一个形状为 (4, 4) 的随机张量
mesh = DeviceMesh("cuda", [[0, 1], [2, 3]])  # 创建一个自定义的 CUDA 设备网格
dtensor = distribute_tensor(tensor, mesh, [Shard(dim=0), Replicate()])  # 在设备网格上分发张量，沿维度 0 进行分片并复制到每个设备
visualize_sharding(dtensor)  # 可视化张量分片结果

# Example 4: 创建另一个随机张量，分发到自定义的设备网格上，并可视化分片情况（不同分片顺序）
tensor = torch.randn(4, 4)  # 创建一个形状为 (4, 4) 的随机张量
mesh = DeviceMesh("cuda", [[0, 1], [2, 3]])  # 创建一个自定义的 CUDA 设备网格
dtensor = distribute_tensor(tensor, mesh, [Replicate(), Shard(dim=0)])  # 在设备网格上分发张量，先复制到每个设备再沿维度 0 进行分片
visualize_sharding(dtensor)  # 可视化张量分片结果

# Example 5: 创建一个随机张量，分发到仅一个设备上，并可视化分片情况（每个 rank 一个子网格）
tensor = torch.randn(4, 4)  # 创建一个形状为 (4, 4) 的随机张量
mesh = DeviceMesh("cuda", [rank])  # 创建一个包含当前 rank 的 CUDA 设备网格
dtensor = distribute_tensor(tensor, mesh, [Replicate()])  # 在设备网格上分发张量，并复制到每个设备
visualize_sharding(dtensor, header=f"Example 5 rank {rank}:")  # 可视化张量分片结果，并指定标题
```