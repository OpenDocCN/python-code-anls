# `.\pytorch\torch\distributed\checkpoint\examples\async_checkpointing_example.py`

```py
# mypy: allow-untyped-defs
# Owner(s): ["oncall: distributed"]

# 引入标准库和第三方库
import os
import shutil
import traceback

# 引入 PyTorch 库及其分布式模块
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.distributed.checkpoint.state_dict import (
    _patch_model_state_dict,
    _patch_optimizer_state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# 设定设备类型为 CUDA
DEVICE = "cuda"
# 定义训练轮数
NUM_EPOCHS = 1000
# 模型保存周期
SAVE_PERIOD = 10
# 模拟故障发生周期
FAULT_PERIOD = 25
# 检查点保存目录，使用用户的登录名作为一部分路径
CHECKPOINT_DIR = f"~/{os.environ.get('LOGNAME', '')}/checkpoint"

# 自定义异常类
class InjectedException(Exception):
    pass

# 定义神经网络模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 定义神经网络的层
        self.net1 = nn.Linear(8, 32)
        self.net2 = nn.Linear(32, 128)
        self.net3 = nn.Linear(128, 64)
        self.net4 = nn.Linear(64, 8)
        self.net5 = nn.Linear(8, 1)

    def forward(self, x):
        # 定义神经网络的前向传播过程
        x = F.relu(self.net1(x))
        x = F.relu(self.net2(x))
        x = F.relu(self.net3(x))
        x = F.relu(self.net4(x))
        x = torch.sigmoid(self.net5(x))  # 使用 sigmoid 激活函数
        return x

# 初始化模型函数，使用 FSDP 进行模型的分片处理
def _init_model(rank, world_size):
    # 初始化设备网格
    device_mesh = init_device_mesh(DEVICE, (world_size,))
    # 创建一个虚拟的神经网络模型，并应用 FSDP
    model = Model().cuda()
    # 再次初始化设备网格（可能有冗余）
    device_mesh = init_device_mesh(DEVICE, (world_size,))
    model = FSDP(model, device_mesh=device_mesh, use_orig_params=True)

    # 定义优化器为 Adam
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)

    # 对模型进行状态字典的修补
    _patch_model_state_dict(model)
    # 对优化器的状态字典进行修补
    _patch_optimizer_state_dict(model, optimizers=optim)

    return model, optim

# 打印消息函数，仅在主节点（rank 0）上打印
def _print(msg):
    if dist.get_rank() == 0:
        print(msg)

# 输入数据函数，生成随机输入和对应的标签
def _input():
    x = torch.rand(128, 8, device="cuda")
    y = torch.zeros(128, 1, device="cuda")

    y[torch.sum(x, dim=1) >= 4] = 1.0

    return x, y

# 主运行函数，设置分布式训练环境并启动训练过程
def run(rank, world_size):
    # 设置主节点地址和端口
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # 初始化进程组，选择 CPU:gloo 和 CUDA:nccl 作为后端
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world_size)
    # 设置当前 CUDA 设备
    torch.cuda.set_device(rank)

    # 初始化模型和优化器
    model, optim = _init_model(rank, world_size)
    # 定义状态字典，包含模型和优化器
    state_dict = {"model": model, "optim": optim}
    # 定义损失计算方法为二元交叉熵损失
    loss_calc = torch.nn.BCELoss()

    f = None  # 文件句柄初始化为空
    for epoch in range(NUM_EPOCHS):
        try:
            # 使用当前 epoch 数作为随机数种子
            torch.manual_seed(epoch)
            # 调用 _input 函数获取输入数据 x, y
            x, y = _input()

            # 计算模型预测结果并计算损失
            loss = loss_calc(model(x), y)

            # 打印当前 epoch 和损失值
            _print(f"{epoch=} {loss=}")

            # 反向传播，计算梯度
            loss.backward()
            # 根据梯度更新优化器参数
            optim.step()
            # 清空梯度
            optim.zero_grad()

            # 如果当前 epoch 是保存周期的倍数
            if epoch % SAVE_PERIOD == 0:
                # 如果 f 不为空，则等待异步保存完成
                if f is not None:
                    f.result()
                # 异步保存模型参数
                f = dcp.state_dict_saver.async_save(
                    state_dict, checkpoint_id=CHECKPOINT_DIR
                )

            # 如果设置了故障注入周期，并且当前 epoch 是其倍数
            if FAULT_PERIOD > 0 and epoch % FAULT_PERIOD == 0:
                # 抛出故障注入异常
                raise InjectedException("Fault injection!")

        except InjectedException as e:
            # 进行进程间同步
            dist.barrier()

            # 打印异常信息
            _print("Trainer encountered exception:")
            # 打印异常堆栈信息
            traceback.print_tb(e.__traceback__)

            # 打印信息，从最近的检查点重新加载模型
            _print("Reloading model from last checkpoint!")
            # 如果 f 不为空，则等待异步保存完成
            if f is not None:
                f.result()
            # 重新加载模型参数
            dcp.load(state_dict)
# 如果当前脚本作为主程序执行（而非作为模块被导入执行），则执行以下代码块
if __name__ == "__main__":
    # 获取当前系统上可用的 CUDA 设备数量，并赋值给 world_size 变量
    world_size = torch.cuda.device_count()
    # 打印消息，显示 Async Checkpointing 示例在多少个设备上执行
    print(f"Running an example of Async Checkpointing on {world_size} devices.")
    # 如果已存在 CHECKPOINT_DIR 目录，则递归删除该目录及其内容，忽略删除过程中的错误
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)

    # 使用 multiprocessing 进程池的方式，启动多个进程来执行 run 函数
    mp.spawn(
        run,                # 执行的函数名
        args=(world_size,), # 传递给 run 函数的参数，这里是一个元组，包含 world_size 变量
        nprocs=world_size,  # 指定启动的进程数量，与 CUDA 设备数量一致
        join=True,          # 是否等待所有进程执行完毕再继续执行后续代码
    )
```