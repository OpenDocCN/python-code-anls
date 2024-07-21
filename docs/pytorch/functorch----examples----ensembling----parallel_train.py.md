# `.\pytorch\functorch\examples\ensembling\parallel_train.py`

```py
# 导入必要的库
import argparse  # 导入命令行参数解析库
import math  # 导入数学库

import torch  # 导入PyTorch深度学习库
import torch.nn as nn  # 导入PyTorch神经网络模块
import torch.nn.functional as F  # 导入PyTorch函数式模块
from torch.func import functional_call, grad_and_value, stack_module_state, vmap  # 导入自定义PyTorch函数

# 代码来源和引用信息
# 该代码段源自Will Whitney的JAX模型集成教程
# 原文引用如下：
# @misc{Whitney2021Parallelizing,
#     author = {William F. Whitney},
#     title = { {Parallelizing neural networks on one GPU with JAX} },
#     year = {2021},
#     url = {http://willwhitney.com/parallel-training-jax.html},
# }

# GOAL: Demonstrate that it is possible to use eager-mode vmap
# to parallelize training over models.

# 解析命令行参数
parser = argparse.ArgumentParser(description="Functorch Ensembled Models")
parser.add_argument(
    "--device",
    type=str,
    default="cpu",
    help="CPU or GPU ID for this process (default: 'cpu')",
)
args = parser.parse_args()

DEVICE = args.device  # 设置设备为命令行参数指定的设备，默认为CPU

# Step 1: Make some spirals

def make_spirals(n_samples, noise_std=0.0, rotations=1.0):
    ts = torch.linspace(0, 1, n_samples, device=DEVICE)  # 在指定设备上生成均匀间隔的张量
    rs = ts**0.5  # 对生成的张量进行操作
    thetas = rs * rotations * 2 * math.pi  # 对生成的张量进行操作
    signs = torch.randint(0, 2, (n_samples,), device=DEVICE) * 2 - 1  # 生成随机整数张量，对生成的张量进行操作
    labels = (signs > 0).to(torch.long).to(DEVICE)  # 将张量转换为长整型，设置设备

    xs = (
        rs * signs * torch.cos(thetas)  # 对生成的张量进行操作
        + torch.randn(n_samples, device=DEVICE) * noise_std  # 生成随机张量，对生成的张量进行操作
    )
    ys = (
        rs * signs * torch.sin(thetas)  # 对生成的张量进行操作
        + torch.randn(n_samples, device=DEVICE) * noise_std  # 生成随机张量，对生成的张量进行操作
    )
    points = torch.stack([xs, ys], dim=1)  # 在指定维度上堆叠张量
    return points, labels  # 返回生成的张量

points, labels = make_spirals(100, noise_std=0.05)  # 调用函数生成螺旋数据点和标签

# Step 2: Define two-layer MLP and loss function
class MLPClassifier(nn.Module):
    def __init__(self, hidden_dim=32, n_classes=2):
        super().__init__()
        self.hidden_dim = hidden_dim  # 隐藏层维度
        self.n_classes = n_classes  # 类别数量

        self.fc1 = nn.Linear(2, self.hidden_dim)  # 第一个全连接层
        self.fc2 = nn.Linear(self.hidden_dim, self.n_classes)  # 第二个全连接层

    def forward(self, x):
        x = self.fc1(x)  # 第一层前向传播
        x = F.relu(x)  # ReLU激活函数
        x = self.fc2(x)  # 第二层前向传播
        x = F.log_softmax(x, -1)  # 对最后一层输出进行log_softmax处理
        return x  # 返回输出结果

loss_fn = nn.NLLLoss()  # 定义损失函数为负对数似然损失
model = MLPClassifier().to(DEVICE)  # 创建MLP分类器模型并将其移动到指定设备

def train_step_fn(weights, batch, targets, lr=0.2):
    def compute_loss(weights, batch, targets):
        output = functional_call(model, weights, batch)  # 使用函数调用模型
        loss = loss_fn(output, targets)  # 计算损失
        return loss  # 返回损失值

    grad_weights, loss = grad_and_value(compute_loss)(weights, batch, targets)  # 计算梯度和损失值

    # NB: PyTorch is missing a "functional optimizer API" (possibly coming soon)
    # so we are going to re-implement SGD here.
    new_weights = {}  # 初始化新的权重字典
    with torch.no_grad():
        for key in grad_weights:
            new_weights[key] = weights[key] - grad_weights[key] * lr  # 使用SGD更新权重

    return loss, new_weights  # 返回损失和更新后的权重

# Step 4: Let's verify this actually trains.
# We should see the loss decrease.
def step4():
    global weights  # 使用全局变量weights
    # 循环执行 2000 次，每次迭代一个训练步骤
    for i in range(2000):
        # 调用 train_step_fn 函数进行训练步骤，返回损失值和更新后的权重
        loss, weights = train_step_fn(dict(model.named_parameters()), points, labels)
        # 如果迭代次数能被 100 整除，则打印当前损失值
        if i % 100 == 0:
            print(loss)
# 执行步骤4函数
step4()

# 步骤5: 准备使用多个模型。定义一个初始化函数 init_fn，
# 给定模型数量，返回所有模型的权重参数。

def init_fn(num_models):
    # 创建 num_models 个 MLPClassifier 模型，并移至指定的 DEVICE
    models = [MLPClassifier().to(DEVICE) for _ in range(num_models)]
    # 获取模型的参数状态，stack_module_state 函数的返回值为参数和无关状态
    params, _ = stack_module_state(models)
    return params

# 步骤6: 现在，我们可以同时尝试多个模型了。
# 答案是：可以！loss 是一个二元组，我们可以看到其值不断减小。

def step6():
    # 使用 vmap 将 train_step_fn 映射到每个模型的训练步骤上，in_dims 指定输入维度
    parallel_train_step_fn = vmap(train_step_fn, in_dims=(0, None, None))
    # 初始化两个模型的权重参数
    batched_weights = init_fn(num_models=2)
    # 进行 2000 次迭代训练
    for i in range(2000):
        # 并行应用训练步骤函数到 batched_weights 上，points 和 labels 是输入数据
        loss, batched_weights = parallel_train_step_fn(batched_weights, points, labels)
        # 每200次迭代打印一次损失值
        if i % 200 == 0:
            print(loss)

# 执行步骤6函数
step6()

# 步骤7: 步骤6的缺陷在于，我们在完全相同的数据上进行训练。
# 这可能导致集成中的所有模型都以相同的方式过拟合。
# http://willwhitney.com/parallel-training-jax.html 提出的解决方案是，
# 以一种方式随机子集化数据，使得模型在每个训练步骤中不完全接收相同的数据！
# 由于本文档的目标是展示我们可以使用即时模式的 vmap 来实现与 JAX 类似的功能，
# 其余内容留给读者作为练习。

# 总结，为了实现 http://willwhitney.com/parallel-training-jax.html 所做的事情，
# 我们使用了以下 PyTorch 不具备的额外项目：
# 1. 将 NN 模块转换为状态和无状态函数的功能 API
# 2. 函数式优化器
# 3. "函数式" grad API（有效地包装 autograd.grad）
# 4. 函数式 grad API 与 torch.vmap 之间的可组合性。
```