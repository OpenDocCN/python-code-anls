# `.\pytorch\functorch\examples\maml_regression\evjang_transforms_module.py`

```
# Eric Jang originally wrote an implementation of MAML in JAX
# (https://github.com/ericjang/maml-jax).
# We translated his implementation from JAX to PyTorch.
# 导入需要的库和模块
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import torch

# 从functorch库中导入需要的函数和类
from functorch import grad, make_functional, vmap
from torch import nn
from torch.nn import functional as F

# 设置matplotlib的后端为非交互模式
mpl.use("Agg")

# 定义一个三层神经网络模型类
class ThreeLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义网络的层：输入层到第一隐藏层，线性变换
        self.fc1 = nn.Linear(1, 40)
        self.relu1 = nn.ReLU()
        # 第一隐藏层到第二隐藏层，线性变换
        self.fc2 = nn.Linear(40, 40)
        self.relu2 = nn.ReLU()
        # 第二隐藏层到输出层，线性变换
        self.fc3 = nn.Linear(40, 1)

    def forward(self, x):
        # 网络的前向传播过程
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# TODO: Use F.mse_loss

# 定义均方误差损失函数
def mse_loss(x, y):
    return torch.mean((x - y) ** 2)

# 将网络模型转换为函数式模型，并获取其参数和优化器
net, params = make_functional(ThreeLayerNet())
opt = torch.optim.Adam(params, lr=1e-3)
alpha = 0.1

# 定义元学习中的超参数和变量
K = 20
losses = []
num_tasks = 4

# 定义用于生成任务的函数
def sample_tasks(outer_batch_size, inner_batch_size):
    # 为每个任务随机选择振幅和相位
    As = []
    phases = []
    for _ in range(outer_batch_size):
        As.append(np.random.uniform(low=0.1, high=0.5))
        phases.append(np.random.uniform(low=0.0, high=np.pi))

    def get_batch():
        xs, ys = [], []
        for A, phase in zip(As, phases):
            # 在给定范围内随机生成输入数据
            x = np.random.uniform(low=-5.0, high=5.0, size=(inner_batch_size, 1))
            y = A * np.sin(x + phase)
            xs.append(x)
            ys.append(y)
        return torch.tensor(xs, dtype=torch.float), torch.tensor(ys, dtype=torch.float)

    x1, y1 = get_batch()
    x2, y2 = get_batch()
    return x1, y1, x2, y2

# 开始元学习的训练循环
for it in range(20000):
    loss2 = 0.0
    opt.zero_grad()

    # 定义任务内部损失计算函数
    def get_loss_for_task(x1, y1, x2, y2):
        def inner_loss(params, x1, y1):
            # 计算当前参数下的网络输出
            f = net(params, x1)
            # 计算均方误差损失
            loss = mse_loss(f, y1)
            return loss

        # 使用梯度函数获取参数关于内部损失的梯度
        grads = grad(inner_loss)(params, x1, y1)
        # 更新参数
        new_params = [(params[i] - alpha * grads[i]) for i in range(len(params))]

        # 在更新后的参数下计算在第二批数据上的损失
        v_f = net(new_params, x2)
        return mse_loss(v_f, y2)

    # 随机抽取多个任务进行元学习
    task = sample_tasks(num_tasks, K)
    # 使用向量化映射函数计算多个任务的损失
    inner_losses = vmap(get_loss_for_task)(task[0], task[1], task[2], task[3])
    # 计算平均损失
    loss2 = sum(inner_losses) / len(inner_losses)
    # 反向传播更新参数
    loss2.backward()

    opt.step()

    # 每隔100次迭代打印一次外部损失
    if it % 100 == 0:
        print("Iteration %d -- Outer Loss: %.4f" % (it, loss2))
    losses.append(loss2.detach())

# 随机生成测试数据
t_A = torch.tensor(0.0).uniform_(0.1, 0.5)
t_b = torch.tensor(0.0).uniform_(0.0, math.pi)

t_x = torch.empty(4, 1).uniform_(-5, 5)
t_y = t_A * torch.sin(t_x + t_b)

# 清空梯度
opt.zero_grad()

# 元学习中的进一步优化循环
t_params = params
for k in range(5):
    t_f = net(t_params, t_x)
    # 使用平均绝对误差损失
    t_loss = F.l1_loss(t_f, t_y)

    # 计算损失关于参数的梯度
    grads = torch.autograd.grad(t_loss, t_params, create_graph=True)
    # 更新参数
    t_params = [(t_params[i] - alpha * grads[i]) for i in range(len(params))]
# 创建一个张量 `test_x`，包含从 `-2π` 到 `2π` 的数值，步长为 `0.01`，并将其转换为列向量
test_x = torch.arange(-2 * math.pi, 2 * math.pi, step=0.01).unsqueeze(1)

# 使用给定的线性变换参数 `t_A` 和 `t_b`，计算对应于 `test_x` 的正弦函数值，存储在 `test_y` 中
test_y = t_A * torch.sin(test_x + t_b)

# 使用神经网络 `net` 对给定参数 `t_params` 和输入 `test_x` 进行前向传播，得到预测值 `test_f`
test_f = net(t_params, test_x)

# 绘制正弦函数曲线，使用 `test_x` 和 `test_y` 数据，标记为 "sin(x)"
plt.plot(test_x.data.numpy(), test_y.data.numpy(), label="sin(x)")

# 绘制神经网络模型预测值曲线，使用 `test_x` 和 `test_f` 数据，标记为 "net(x)"
plt.plot(test_x.data.numpy(), test_f.data.numpy(), label="net(x)")

# 绘制训练样本点，使用 `t_x` 和 `t_y` 数据，标记为 "Examples"，显示为圆点
plt.plot(t_x.data.numpy(), t_y.data.numpy(), "o", label="Examples")

# 添加图例，展示每条曲线的标签
plt.legend()

# 将当前图保存为 PNG 格式的文件 "maml-sine.png"
plt.savefig("maml-sine.png")

# 创建一个新的图形窗口
plt.figure()

# 计算损失值 `losses` 的移动平均，窗口大小为 20，并绘制其图像
plt.plot(np.convolve(losses, [0.05] * 20))

# 将当前图保存为 PNG 格式的文件 "losses.png"
plt.savefig("losses.png")
```