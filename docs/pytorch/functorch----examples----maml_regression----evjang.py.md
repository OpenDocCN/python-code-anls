# `.\pytorch\functorch\examples\maml_regression\evjang.py`

```py
# Eric Jang originally wrote an implementation of MAML in JAX
# (https://github.com/ericjang/maml-jax).
# We translated his implementation from JAX to PyTorch.

import math  # 导入数学库

import matplotlib as mpl  # 导入matplotlib库，用于绘图
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot模块，用于绘图
import numpy as np  # 导入numpy库，用于数值计算

import torch  # 导入PyTorch深度学习库
from torch.nn import functional as F  # 导入PyTorch的函数模块，用于定义神经网络层的操作

mpl.use("Agg")  # 设置matplotlib使用"Agg"后端，用于无需图形界面的图像生成

# 定义神经网络模型
def net(x, params):
    x = F.linear(x, params[0], params[1])  # 线性变换
    x = F.relu(x)  # ReLU激活函数

    x = F.linear(x, params[2], params[3])  # 线性变换
    x = F.relu(x)  # ReLU激活函数

    x = F.linear(x, params[4], params[5])  # 线性变换
    return x

# 初始化模型参数
params = [
    torch.Tensor(40, 1).uniform_(-1.0, 1.0).requires_grad_(),  # 第一层权重参数
    torch.Tensor(40).zero_().requires_grad_(),  # 第一层偏置参数
    torch.Tensor(40, 40)
    .uniform_(-1.0 / math.sqrt(40), 1.0 / math.sqrt(40))
    .requires_grad_(),  # 第二层权重参数
    torch.Tensor(40).zero_().requires_grad_(),  # 第二层偏置参数
    torch.Tensor(1, 40)
    .uniform_(-1.0 / math.sqrt(40), 1.0 / math.sqrt(40))
    .requires_grad_(),  # 第三层权重参数
    torch.Tensor(1).zero_().requires_grad_(),  # 第三层偏置参数
]

opt = torch.optim.Adam(params, lr=1e-3)  # 使用Adam优化器进行参数优化
alpha = 0.1  # 定义学习率

K = 20  # 内部循环迭代次数
losses = []  # 记录损失函数值列表
num_tasks = 4  # 每轮任务数

# 生成任务数据集函数
def sample_tasks(outer_batch_size, inner_batch_size):
    # 随机选择振幅和相位
    As = []
    phases = []
    for _ in range(outer_batch_size):
        As.append(np.random.uniform(low=0.1, high=0.5))  # 随机振幅
        phases.append(np.random.uniform(low=0.0, high=np.pi))  # 随机相位

    # 生成内部数据集
    def get_batch():
        xs, ys = [], []
        for A, phase in zip(As, phases):
            x = np.random.uniform(low=-5.0, high=5.0, size=(inner_batch_size, 1))  # 随机生成输入数据
            y = A * np.sin(x + phase)  # 计算对应的输出数据
            xs.append(x)
            ys.append(y)
        return torch.tensor(xs, dtype=torch.float), torch.tensor(ys, dtype=torch.float)

    x1, y1 = get_batch()  # 第一个任务数据集
    x2, y2 = get_batch()  # 第二个任务数据集
    return x1, y1, x2, y2

# 主循环，训练网络
for it in range(20000):
    loss2 = 0.0  # 内部损失函数值归零
    opt.zero_grad()  # 优化器梯度清零

    # 定义任务损失计算函数
    def get_loss_for_task(x1, y1, x2, y2):
        f = net(x1, params)  # 计算模型在第一组数据上的输出
        loss = F.mse_loss(f, y1)  # 计算均方误差损失

        # 使用create_graph=True，因为这里的梯度计算是前向传播的一部分。
        # 我们希望通过SGD更新步骤进行微分，并在反向传播中获得高阶导数。
        grads = torch.autograd.grad(loss, params, create_graph=True)  # 计算梯度
        new_params = [(params[i] - alpha * grads[i]) for i in range(len(params))]  # 更新参数

        v_f = net(x2, new_params)  # 计算模型在第二组数据上的输出
        return F.mse_loss(v_f, y2)  # 返回均方误差损失

    task = sample_tasks(num_tasks, K)  # 采样任务数据集
    inner_losses = [
        get_loss_for_task(task[0][i], task[1][i], task[2][i], task[3][i])  # 计算每个任务的损失
        for i in range(num_tasks)
    ]
    loss2 = sum(inner_losses) / len(inner_losses)  # 计算平均内部损失
    loss2.backward()  # 反向传播，计算梯度

    opt.step()  # 更新模型参数

    if it % 100 == 0:
        print("Iteration %d -- Outer Loss: %.4f" % (it, loss2))  # 打印外部损失值
    losses.append(loss2.detach())  # 记录损失值

t_A = torch.tensor(0.0).uniform_(0.1, 0.5)  # 随机初始化振幅
t_b = torch.tensor(0.0).uniform_(0.0, math.pi)  # 随机初始化相位

t_x = torch.empty(4, 1).uniform_(-5, 5)  # 随机初始化输入数据
t_y = t_A * torch.sin(t_x + t_b)  # 计算对应的输出数据

opt.zero_grad()  # 优化器梯度清零

t_params = params  # 使用训练后的参数进行测试
for k in range(5):
    t_f = net(t_x, t_params)  # 在测试数据上进行模型预测
    # 计算两个张量 t_f 和 t_y 之间的 L1 损失
    t_loss = F.l1_loss(t_f, t_y)
    
    # 计算关于参数 t_params 的梯度，同时创建一个计算图以便进行高阶梯度计算
    grads = torch.autograd.grad(t_loss, t_params, create_graph=True)
    
    # 使用梯度下降法更新参数 t_params，alpha 是学习率
    t_params = [(t_params[i] - alpha * grads[i]) for i in range(len(params))]
# 生成一个 Torch 张量，表示从 -2π 到 2π 的数值序列，步长为 0.01，并将其转为列向量
test_x = torch.arange(-2 * math.pi, 2 * math.pi, step=0.01).unsqueeze(1)

# 计算使用 t_A 和 t_b 参数对 test_x 的正弦函数进行变换
test_y = t_A * torch.sin(test_x + t_b)

# 使用神经网络 net 对 test_x 应用前向传播，得到预测结果
test_f = net(test_x, t_params)

# 绘制 test_x 与 test_y 的正弦函数曲线
plt.plot(test_x.data.numpy(), test_y.data.numpy(), label="sin(x)")

# 绘制 test_x 与 test_f 的神经网络预测结果曲线
plt.plot(test_x.data.numpy(), test_f.data.numpy(), label="net(x)")

# 绘制训练样本 t_x 与 t_y 的散点图
plt.plot(t_x.data.numpy(), t_y.data.numpy(), "o", label="Examples")

# 添加图例
plt.legend()

# 将当前图形保存为 "maml-sine.png" 文件
plt.savefig("maml-sine.png")

# 创建新的图形窗口
plt.figure()

# 绘制损失值列表 losses 的卷积曲线，平滑窗口大小为 20
plt.plot(np.convolve(losses, [0.05] * 20))

# 将当前图形保存为 "losses.png" 文件
plt.savefig("losses.png")
```