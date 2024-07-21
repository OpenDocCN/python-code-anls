# `.\pytorch\test\bottleneck_test\test_cuda.py`

```
# Owner(s): ["module: unknown"]

# 导入PyTorch库
import torch
# 导入神经网络模块
import torch.nn as nn

# 定义模型类，继承自nn.Module
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义一个线性层，输入维度为20，输出维度为20
        self.linear = nn.Linear(20, 20)

    # 定义前向传播方法
    def forward(self, input):
        # 取输入张量的第二个维度的第10到29列作为输入，经过线性层处理
        out = self.linear(input[:, 10:30])
        # 返回处理后的输出张量的和
        return out.sum()

# 定义主函数
def main():
    # 生成一个大小为10x50的随机张量，并将其放在GPU上
    data = torch.randn(10, 50).cuda()
    # 创建一个Model实例并放在GPU上
    model = Model().cuda()
    # 定义SGD优化器，将模型参数传入，学习率为0.0001
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    # 迭代10次
    for i in range(10):
        # 梯度清零
        optimizer.zero_grad()
        # 计算模型的输出
        loss = model(data)
        # 计算损失的梯度
        loss.backward()
        # 更新参数
        optimizer.step()

# 如果运行的是主程序，则执行main函数
if __name__ == "__main__":
    main()
```