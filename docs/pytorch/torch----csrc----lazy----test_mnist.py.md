# `.\pytorch\torch\csrc\lazy\test_mnist.py`

```py
# 忽略 mypy 的错误信息

# 导入标准库 os
import os

# 导入 torchvision 库中的 datasets 和 transforms 模块
from torchvision import datasets, transforms

# 导入 PyTorch 核心库
import torch
import torch._lazy
import torch._lazy.metrics
import torch._lazy.ts_backend

# 导入 PyTorch 中的神经网络模块和函数
import torch.nn as nn
import torch.nn.functional as F

# 导入 PyTorch 中的优化器模块
import torch.optim as optim

# 从 PyTorch 中的优化器模块导入学习率调度器 StepLR
from torch.optim.lr_scheduler import StepLR

# 初始化 PyTorch 懒加载模块的时间序列后端
torch._lazy.ts_backend.init()

# 定义神经网络模型类 Net，继承自 nn.Module
class Net(nn.Module):
    # 初始化函数
    def __init__(self):
        super().__init__()
        # 定义第一层卷积层，输入通道 1，输出通道 32，卷积核大小 3x3，步长 1
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # 定义第二层卷积层，输入通道 32，输出通道 64，卷积核大小 3x3，步长 1
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # 定义第一个 Dropout 层，丢弃概率为 0.25
        self.dropout1 = nn.Dropout(0.25)
        # 定义第二个 Dropout 层，丢弃概率为 0.5
        self.dropout2 = nn.Dropout(0.5)
        # 定义第一个全连接层，输入特征数 9216，输出特征数 128
        self.fc1 = nn.Linear(9216, 128)
        # 定义第二个全连接层，输入特征数 128，输出特征数 10
        self.fc2 = nn.Linear(128, 10)

    # 前向传播函数
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# 定义训练函数 train，接受日志间隔 log_interval、模型 model、设备 device、训练数据加载器 train_loader、优化器 optimizer 和当前轮次 epoch
def train(log_interval, model, device, train_loader, optimizer, epoch):
    # 将模型设置为训练模式
    model.train()
    # 遍历训练数据加载器中的每个批次
    for batch_idx, (data, target) in enumerate(train_loader):
        # 将数据和标签移动到指定设备上
        data, target = data.to(device), target.to(device)
        # 清空优化器的梯度，设置为 None
        optimizer.zero_grad(set_to_none=True)
        # 前向传播得到模型输出
        output = model(data)
        # 计算损失函数，使用负对数似然损失函数
        loss = F.nll_loss(output, target)
        # 反向传播计算梯度
        loss.backward()
        # 根据梯度更新优化器的参数
        optimizer.step()
        # 标记当前训练步骤
        torch._lazy.mark_step()

        # 如果当前批次是日志输出间隔的倍数
        if batch_idx % log_interval == 0:
            # 打印训练日志，显示当前轮次、已处理数据量、总数据量的百分比和损失值
            print(
                f"Train Epoch: {epoch} "
                f"[{batch_idx * len(data)}/{len(train_loader.dataset)} ({100.0 * batch_idx / len(train_loader):.0f}%)]"
                f"\tLoss: {loss.item():.6f}"
            )

# 如果当前脚本是主程序
if __name__ == "__main__":
    # 定义批次大小、设备、轮次、日志输出间隔、学习率和学习率衰减系数
    bsz = 64
    device = "lazy"
    epochs = 14
    log_interval = 10
    lr = 1
    gamma = 0.7
    # 定义训练参数字典
    train_kwargs = {"batch_size": bsz}
    # 如果环境变量中存在 CUDA 相关设置
    if "LTC_TS_CUDA" in os.environ:
        # 更新训练参数字典，添加 CUDA 相关设置
        cuda_kwargs = {
            "num_workers": 1,
            "pin_memory": True,
            "shuffle": True,
            "batch_size": bsz,
        }
        train_kwargs.update(cuda_kwargs)

    # 定义数据预处理方法 transform，包括 ToTensor 和 Normalize 操作
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    # 下载并加载 MNIST 数据集
    dataset1 = datasets.MNIST("./data", train=True, download=True, transform=transform)
    # 创建训练数据加载器
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    # 创建神经网络模型实例，并移动到指定设备
    model = Net().to(device)
    # 使用 Adadelta 优化器，并传入模型参数和学习率
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    # 使用 StepLR 学习率调度器，并传入优化器和学习率衰减系数
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    # 遍历每个轮次
    for epoch in range(1, epochs + 1):
        # 调用训练函数进行模型训练
        train(log_interval, model, device, train_loader, optimizer, epoch)
        # 更新学习率调度器
        scheduler.step()
```