# `.\pytorch\test\mobile\nnc\aot_test_model.py`

```
# 导入 PyTorch 库
import torch
# 从 torch 库中导入 nn 模块
from torch import nn

# 定义一个神经网络模型类，继承自 nn.Module
class NeuralNetwork(nn.Module):
    # 定义模型的前向传播方法
    def forward(self, x):
        # 对输入张量 x 中的每个元素加上 10，返回结果
        return torch.add(x, 10)

# 创建神经网络模型的实例
model = NeuralNetwork()
# 对模型进行即时编译成 Torch 脚本
script = torch.jit.script(model)
# 将编译好的 Torch 脚本保存到文件 "aot_test_model.pt"
torch.jit.save(script, "aot_test_model.pt")
```