# `.\pytorch\torch\distributed\checkpoint\examples\stateful_example.py`

```py
# 设置 mypy，允许未类型化的定义
# 所有者信息，指定为 ["oncall: distributed"]

# 导入必要的库
import os
import shutil

# 导入 PyTorch 相关模块
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.distributed.checkpoint.state_dict import (
    _patch_model_state_dict,
    _patch_optimizer_state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# 定义检查点保存目录
CHECKPOINT_DIR = f"~/{os.environ['LOGNAME']}/checkpoint"

# 定义神经网络模型类
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.net1 = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        self.net2 = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
        self.net3 = nn.Linear(32, 64)
        self.net4 = nn.Sequential(nn.ReLU(), nn.Linear(64, 8))

    def forward(self, x):
        return self.net4(self.net3(self.net2(self.net1(x))))

    def get_input(self):
        return torch.rand(8, 8, device="cuda")

# 函数：为模型和优化器应用状态修补
def _make_stateful(model, optim):
    _patch_model_state_dict(model)  # 修补模型状态字典
    _patch_optimizer_state_dict(model, optimizers=optim)  # 修补优化器状态字典

# 函数：训练模型
def _train(model, optim, train_steps=1):
    torch.manual_seed(0)
    loss = None
    for _ in range(train_steps):
        loss = model(model.get_input()).sum()  # 计算损失
        loss.backward()  # 反向传播
        optim.step()  # 更新参数
        optim.zero_grad()  # 梯度清零

    return loss  # 返回损失值

# 函数：初始化模型和设备网格
def _init_model(device, world_size):
    device_mesh = init_device_mesh(device, (world_size,))  # 初始化设备网格
    model = Model().cuda()  # 创建模型并移到 GPU
    model = FSDP(
        model,
        device_mesh=device_mesh,
        use_orig_params=True,
    )  # 使用 FullyShardedDataParallel 封装模型
    optim = torch.optim.Adam(model.parameters(), lr=0.1)  # 创建优化器
    _make_stateful(model, optim)  # 为模型和优化器应用状态修补

    return model, optim  # 返回模型和优化器

# 函数：执行每个进程的训练和检查点保存/加载
def run(rank, world_size, device="cuda"):
    # 设置主地址和端口
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # 初始化分布式进程组
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)  # 设置当前设备为指定 GPU

    # 初始化模型和优化器
    model, optim = _init_model(device, world_size)
    _train(model, optim, train_steps=2)  # 进行训练

    # 保存检查点
    dcp.save(
        state_dict={"model": model, "optimizer": optim},
        checkpoint_id=CHECKPOINT_DIR,
    )

    # 可能执行其他操作

    # 初始化新的模型和优化器
    model, optim = _init_model(device, world_size)
    dcp.load(
        state_dict={"model": model, "optimizer": optim},
        checkpoint_id=CHECKPOINT_DIR,
    )  # 加载检查点
    _train(model, optim, train_steps=2)  # 继续训练

# 主程序入口
if __name__ == "__main__":
    world_size = torch.cuda.device_count()  # 获取 GPU 数量
    print(f"Running stateful checkpoint example on {world_size} devices.")
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)  # 清除之前的检查点目录
    mp.spawn(
        run,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )  # 启动多进程运行
```