# `.\pytorch\test\create_dummy_torchscript_model.py`

```
# Usage: python create_dummy_model.py <name_of_the_file>
# 导入系统模块
import sys

# 导入PyTorch相关模块
import torch
from torch import nn

# 定义神经网络类，继承自 nn.Module
class NeuralNetwork(nn.Module):
    # 初始化方法
    def __init__(self):
        super().__init__()
        # 将输入展平
        self.flatten = nn.Flatten()
        # 定义包含线性层和ReLU激活函数的顺序容器
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),  # 输入大小为28*28，输出大小为512
            nn.ReLU(),                # ReLU激活函数
            nn.Linear(512, 512),      # 输入大小为512，输出大小为512
            nn.ReLU(),                # ReLU激活函数
            nn.Linear(512, 10),       # 输入大小为512，输出大小为10
        )

    # 前向传播方法
    def forward(self, x):
        # 将输入展平
        x = self.flatten(x)
        # 经过线性层和ReLU激活函数的堆叠
        logits = self.linear_relu_stack(x)
        return logits


# 程序的入口点
if __name__ == "__main__":
    # 使用 TorchScript 将神经网络模型转换为脚本模块
    jit_module = torch.jit.script(NeuralNetwork())
    # 将 TorchScript 模块保存到指定文件中
    torch.jit.save(jit_module, sys.argv[1])
    
    # 创建原始的神经网络模块（非 TorchScript）
    orig_module = nn.Sequential(
        nn.Linear(28 * 28, 512),  # 输入大小为28*28，输出大小为512
        nn.ReLU(),                # ReLU激活函数
        nn.Linear(512, 512),      # 输入大小为512，输出大小为512
        nn.ReLU(),                # ReLU激活函数
        nn.Linear(512, 10),       # 输入大小为512，输出大小为10
    )
    # 将原始神经网络模块保存到指定文件名后加上".orig"的文件中
    torch.save(orig_module, sys.argv[1] + ".orig")
```