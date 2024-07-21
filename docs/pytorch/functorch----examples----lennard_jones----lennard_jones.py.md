# `.\pytorch\functorch\examples\lennard_jones\lennard_jones.py`

```
# 导入 PyTorch 库
import torch
# 导入神经网络模块
from torch import nn
# 导入自动求导相关函数
from torch.func import jacrev, vmap
# 导入均方误差损失函数
from torch.nn.functional import mse_loss

# 定义常数 sigma 和 epsilon
sigma = 0.5
epsilon = 4.0

# 定义 Lennard-Jones 势能函数
def lennard_jones(r):
    return epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)

# 定义 Lennard-Jones 力的大小函数
def lennard_jones_force(r):
    """Get magnitude of LJ force"""
    return -epsilon * ((-12 * sigma**12 / r**13) + (6 * sigma**6 / r**7))

# 设置训练数据大小
training_size = 1000
# 在指定范围内创建张量 r，用于表示距离
r = torch.linspace(0.5, 2 * sigma, steps=training_size, requires_grad=True)

# 创建一组指向正 x 方向的向量
drs = torch.outer(r, torch.tensor([1.0, 0, 0]))
# 计算向量的范数，并重塑维度
norms = torch.norm(drs, dim=1).reshape(-1, 1)

# 创建训练能量值
training_energies = torch.stack(list(map(lennard_jones, norms))).reshape(-1, 1)
# 使用随机方向向量创建训练力值
training_forces = torch.stack(
    [force * dr for force, dr in zip(map(lennard_jones_force, norms), drs)]
)

# 定义神经网络模型结构
model = nn.Sequential(
    nn.Linear(1, 16),
    nn.Tanh(),
    nn.Linear(16, 16),
    nn.Tanh(),
    nn.Linear(16, 16),
    nn.Tanh(),
    nn.Linear(16, 16),
    nn.Tanh(),
    nn.Linear(16, 1),
)

# 定义生成预测的函数
def make_prediction(model, drs):
    norms = torch.norm(drs, dim=1).reshape(-1, 1)
    energies = model(norms)

    # 计算模型的雅可比矩阵
    network_derivs = vmap(jacrev(model))(norms).squeeze(-1)
    # 计算预测力
    forces = -network_derivs * drs / norms
    return energies, forces

# 定义损失函数
def loss_fn(energies, forces, predicted_energies, predicted_forces):
    return (
        mse_loss(energies, predicted_energies)
        + 0.01 * mse_loss(forces, predicted_forces) / 3
    )

# 使用 Adam 优化器来优化模型参数
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

# 开始训练循环
for epoch in range(400):
    optimiser.zero_grad()
    energies, forces = make_prediction(model, drs)
    # 计算损失值
    loss = loss_fn(training_energies, training_forces, energies, forces)
    # 执行反向传播
    loss.backward(retain_graph=True)
    # 更新模型参数
    optimiser.step()

    # 每 20 次迭代输出一次损失值
    if epoch % 20 == 0:
        print(loss.cpu().item())
```