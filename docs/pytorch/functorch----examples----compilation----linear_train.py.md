# `.\pytorch\functorch\examples\compilation\linear_train.py`

```py
# 导入时间模块，用于测量代码执行时间
import time

# 导入PyTorch相关模块
import torch
import torch.nn as nn

# 从functorch模块中导入函数和编译器
from functorch import make_functional
from functorch.compile import nnc_jit

# 设置在CPU上启用融合优化
torch._C._jit_override_can_fuse_on_cpu(True)

# 定义基准测试函数，用于评估函数执行时间
def bench(f, iters=100, warmup=10):
    # 预热阶段，执行f()函数warmup次
    for _ in range(warmup):
        f()
    # 开始计时
    begin = time.time()
    # 执行iters次f()函数调用并计时
    for _ in range(iters):
        f()
    # 输出总执行时间
    print(time.time() - begin)

# 定义一个简单的神经网络模块
class Foo(nn.Module):
    def __init__(self, num_layers=3, features=100):
        super().__init__()
        mods = []
        # 构建包含num_layers个线性层的神经网络
        for _ in range(num_layers):
            mods.append(nn.Linear(features, features, bias=False))
        self.mod = nn.Sequential(*mods)

    def forward(self, x):
        # 前向传播：将输入x通过网络self.mod，然后计算平方和
        return (self.mod(x) ** 2).sum()

# 设置训练时的批量大小、特征数和层数
batch_size = 16
features = 64
num_layers = 8
# 生成随机输入数据
inp = torch.randn((batch_size, features))

# 创建Foo类的实例mod
mod = Foo(num_layers, features)

# 使用torch.jit.script对模型进行脚本化
jit_mod = torch.jit.script(mod)

# 将模型转换为函数式表达形式和权重
func_model, weights = make_functional(mod)
lr = 1.0

# 定义函数式表达形式的训练步骤
def functional_step(x, weights):
    # 对权重进行分离并要求梯度
    weights = [weight.detach().requires_grad_() for weight in weights]
    # 使用func_model进行前向传播
    out = func_model(weights, x)
    # 计算损失并进行反向传播
    out.backward()
    # 更新权重
    new_weights = [weight - lr * weight.grad for weight in weights]
    return out, new_weights

# 使用torch.optim定义SGD优化器
optim = torch.optim.SGD(
    jit_mod.parameters(), lr=lr, momentum=0, dampening=0, weight_decay=0
)

# 定义JIT编译后模型的训练步骤
def jit_step(x, weights):
    # 梯度清零
    optim.zero_grad()
    # 前向传播计算损失
    loss = jit_mod(x)
    # 反向传播计算梯度
    loss.backward()
    # 更新模型参数
    optim.step()
    return loss, None

# 定义训练函数，用于执行训练步骤并计算训练时间
def train(train_step, weights):
    # 设定随机种子
    torch.manual_seed(16)
    # 执行一次训练步骤，以初始化权重
    train_step(inp, weights)
    # 开始计时
    begin = time.time()
    # 迭代训练1000次
    for itr in range(1000):
        # 执行训练步骤，并获取损失和更新后的权重
        loss, weights = train_step(torch.randn(batch_size, features), weights)
        # 每200次打印一次损失值
        if itr % 200 == 0:
            print(f"Loss at {itr}: {loss}")
    # 输出训练所用时间
    print("Time taken: ", time.time() - begin)
    print()

# 设置使用functional_step作为训练步骤进行PyTorch训练
grad_pt = functional_step
# 使用nnc_jit将functional_step编译为NNC格式用于训练
grad_nnc = nnc_jit(functional_step)

# 打印提示信息，开始使用PyTorch训练
print("Starting PT training")
# 执行训练过程
train(grad_pt, weights)

# 打印提示信息，开始使用NNC训练
print("Starting NNC training")
# 执行训练过程
train(grad_nnc, weights)

# 打印提示信息，开始使用JIT编译后的模型训练
print("Starting JIT training")
# 执行训练过程
train(jit_step, None)
```