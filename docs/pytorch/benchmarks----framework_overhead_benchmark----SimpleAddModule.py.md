# `.\pytorch\benchmarks\framework_overhead_benchmark\SimpleAddModule.py`

```py
# 从 utils 模块导入 NUM_LOOP_ITERS 常量
from utils import NUM_LOOP_ITERS

# 导入 PyTorch 库
import torch


# 定义一个函数用于循环添加张量的操作
def add_tensors_loop(x, y):
    # 执行张量的加法操作，得到结果张量 z
    z = torch.add(x, y)
    # 使用 NUM_LOOP_ITERS 次循环，将 x 与 z 相加，更新 z 的值
    for i in range(NUM_LOOP_ITERS):
        z = torch.add(z, x)
    # 返回最终的结果张量 z
    return z


# 定义一个简单的 PyTorch 模块，用于执行添加操作
class SimpleAddModule(torch.nn.Module):
    # 初始化方法，接受一个 add_op 函数作为参数
    def __init__(self, add_op):
        # 调用父类的初始化方法
        super().__init__()
        # 将传入的 add_op 函数赋值给实例变量 self.add_op
        self.add_op = add_op

    # 前向传播方法，接受两个张量 x 和 y 作为输入
    def forward(self, x, y):
        # 调用实例变量 self.add_op 执行添加操作，并返回结果
        return self.add_op(x, y)
```