# `.\pytorch\functorch\examples\maml_regression\evjang_transforms.py`

```
# Eric Jang originally wrote an implementation of MAML in JAX
# (https://github.com/ericjang/maml-jax).
# We translated his implementation from JAX to PyTorch.

import math  # 导入数学库

import matplotlib as mpl  # 导入matplotlib库
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot模块
import numpy as np  # 导入numpy库

import torch  # 导入PyTorch库
from torch.func import grad, vmap  # 从torch.func模块导入grad和vmap函数
from torch.nn import functional as F  # 从torch.nn模块导入functional，并用F作为别名

mpl.use("Agg")  # 设置matplotlib使用"Agg"后端

def net(params, x):
    x = F.linear(x, params[0], params[1])  # 使用线性函数对输入x进行变换
    x = F.relu(x)  # 使用ReLU激活函数

    x = F.linear(x, params[2], params[3])  # 使用线性函数对输入x进行变换
    x = F.relu(x)  # 使用ReLU激活函数

    x = F.linear(x, params[4], params[5])  # 使用线性函数对输入x进行变换
    return x  # 返回变换后的输出

params = [
    torch.Tensor(40, 1).uniform_(-1.0, 1.0).requires_grad_(),  # 创建形状为(40, 1)的张量，并初始化为均匀分布的随机数
    torch.Tensor(40).zero_().requires_grad_(),  # 创建形状为(40,)的张量，并初始化为0
    torch.Tensor(40, 40)
    .uniform_(-1.0 / math.sqrt(40), 1.0 / math.sqrt(40))
    .requires_grad_(),  # 创建形状为(40, 40)的张量，并初始化为特定范围内的均匀分布随机数
    torch.Tensor(40).zero_().requires_grad_(),  # 创建形状为(40,)的张量，并初始化为0
    torch.Tensor(1, 40)
    .uniform_(-1.0 / math.sqrt(40), 1.0 / math.sqrt(40))
    .requires_grad_(),  # 创建形状为(1, 40)的张量，并初始化为特定范围内的均匀分布随机数
    torch.Tensor(1).zero_().requires_grad_(),  # 创建形状为(1,)的张量，并初始化为0
]

# TODO: use F.mse_loss

def mse_loss(x, y):
    return torch.mean((x - y) ** 2)  # 计算均方误差损失函数

opt = torch.optim.Adam(params, lr=1e-3)  # 使用Adam优化器来优化模型参数，学习率为1e-3
alpha = 0.1  # 设置步长参数alpha

K = 20  # 内部任务的批次大小
losses = []  # 用于存储损失值的列表
num_tasks = 4  # 外部任务的数量

def sample_tasks(outer_batch_size, inner_batch_size):
    # Select amplitude and phase for the task
    As = []  # 存储振幅的列表
    phases = []  # 存储相位的列表
    for _ in range(outer_batch_size):
        As.append(np.random.uniform(low=0.1, high=0.5))  # 从均匀分布中随机选择振幅A，并添加到列表中
        phases.append(np.random.uniform(low=0.0, high=np.pi))  # 从均匀分布中随机选择相位phi，并添加到列表中

    def get_batch():
        xs, ys = [], []  # 存储输入x和输出y的列表
        for A, phase in zip(As, phases):
            x = np.random.uniform(low=-5.0, high=5.0, size=(inner_batch_size, 1))  # 从均匀分布中生成输入x，并添加到列表中
            y = A * np.sin(x + phase)  # 计算对应的输出y，并添加到列表中
            xs.append(x)
            ys.append(y)
        return torch.tensor(xs, dtype=torch.float), torch.tensor(ys, dtype=torch.float)  # 返回张量形式的输入x和输出y

    x1, y1 = get_batch()  # 获取第一个批次的输入x1和输出y1
    x2, y2 = get_batch()  # 获取第二个批次的输入x2和输出y2
    return x1, y1, x2, y2  # 返回两个批次的输入和输出

for it in range(20000):
    loss2 = 0.0  # 初始化内部损失值为0
    opt.zero_grad()  # 清零梯度

    def get_loss_for_task(x1, y1, x2, y2):
        def inner_loss(params, x1, y1):
            f = net(params, x1)  # 使用神经网络模型net对x1进行预测
            loss = mse_loss(f, y1)  # 计算预测值f和真实值y1之间的均方误差损失
            return loss  # 返回损失值

        grads = grad(inner_loss)(tuple(params), x1, y1)  # 计算关于内部任务的梯度
        new_params = [(params[i] - alpha * grads[i]) for i in range(len(params))]  # 更新模型参数

        v_f = net(new_params, x2)  # 使用更新后的参数对x2进行预测
        return mse_loss(v_f, y2)  # 返回预测值v_f和真实值y2之间的均方误差损失

    task = sample_tasks(num_tasks, K)  # 生成外部任务的数据集
    inner_losses = vmap(get_loss_for_task)(task[0], task[1], task[2], task[3])  # 对所有任务应用get_loss_for_task函数
    loss2 = sum(inner_losses) / len(inner_losses)  # 计算平均内部损失
    loss2.backward()  # 反向传播，计算梯度

    opt.step()  # 使用优化器更新模型参数

    if it % 100 == 0:
        print("Iteration %d -- Outer Loss: %.4f" % (it, loss2))  # 每100次迭代输出一次外部损失值
    losses.append(loss2.detach())  # 将当前损失值添加到损失列表中

t_A = torch.tensor(0.0).uniform_(0.1, 0.5)  # 创建一个均匀分布的振幅张量t_A
t_b = torch.tensor(0.0).uniform_(0.0, math.pi)  # 创建一个均匀分布的相位张量t_b

t_x = torch.empty(4, 1).uniform_(-5, 5)  # 创建形状为(4, 1)的均匀分布的输入张量t_x
t_y = t_A * torch.sin(t_x + t_b)  # 计算对应的输出张量t_y

opt.zero_grad()  # 清零梯度

t_params = params  # 复制当前模型参数到t_params
for k in range(5):
    t_f = net(t_params, t_x)  # 使用当前模型参数对t_x进行预测
    t_loss = F.l1_loss(t_f, t_y)  # 计算预测值t_f和真实值t_y之间的L1损失

    grads = torch.autograd.grad(t_loss, t_params, create_graph=True)  # 计算关于模型参数的梯度，同时保留计算图
    # 使用列表推导式对参数列表中的每个参数进行更新，更新规则为当前参数减去学习率乘以对应的梯度
    t_params = [(t_params[i] - alpha * grads[i]) for i in range(len(params))]
# 使用 torch 库中的 arange 函数生成从 -2π 到 2π 的数据点，步长为 0.01，并增加一个维度
test_x = torch.arange(-2 * math.pi, 2 * math.pi, step=0.01).unsqueeze(1)

# 根据给定的参数 t_A 和 t_b，计算 sin 函数在 test_x 上的值
test_y = t_A * torch.sin(test_x + t_b)

# 使用神经网络 net 对给定的参数 t_params 计算 test_x 上的函数值
test_f = net(t_params, test_x)

# 绘制 sin(x) 的图像，使用 test_x 和 test_y 的数据
plt.plot(test_x.data.numpy(), test_y.data.numpy(), label="sin(x)")

# 绘制神经网络计算结果的图像，使用 test_x 和 test_f 的数据
plt.plot(test_x.data.numpy(), test_f.data.numpy(), label="net(x)")

# 绘制给定的样本数据点，使用 t_x 和 t_y 的数据，并用圆圈标识
plt.plot(t_x.data.numpy(), t_y.data.numpy(), "o", label="Examples")

# 添加图例到图中
plt.legend()

# 将当前图形保存为名为 "maml-sine.png" 的图像文件
plt.savefig("maml-sine.png")

# 创建一个新的图形
plt.figure()

# 绘制损失值的移动平均，使用 losses 数组，并设置窗口大小为 20
plt.plot(np.convolve(losses, [0.05] * 20))

# 将当前图形保存为名为 "losses.png" 的图像文件
plt.savefig("losses.png")
```