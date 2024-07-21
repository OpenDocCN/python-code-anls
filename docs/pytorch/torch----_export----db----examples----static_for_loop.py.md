# `.\pytorch\torch\_export\db\examples\static_for_loop.py`

```py
# 导入 torch 模块，用于神经网络和张量操作
import torch

# 定义一个继承自 torch.nn.Module 的类 StaticForLoop，表示一个具有固定迭代次数的 for 循环，其在导出的计算图中应该展开。
class StaticForLoop(torch.nn.Module):
    """
    A for loop with constant number of iterations should be unrolled in the exported graph.
    """
    
    # 定义 forward 方法，用于模型的前向传播
    def forward(self, x):
        # 初始化一个空列表 ret，用于存储计算结果
        ret = []
        # 开始一个 for 循环，循环变量 i 的取值范围是 range(10)，即 [0, 1, 2, ..., 9]
        for i in range(10):  # constant
            # 将 i + x 的结果添加到 ret 列表中
            ret.append(i + x)
        # 返回存储计算结果的列表 ret
        return ret

# 定义一个示例输入 example_inputs，包含一个形状为 (3, 2) 的随机张量
example_inputs = (torch.randn(3, 2),)

# 定义一个标签 tags，用于标记 Python 控制流
tags = {"python.control-flow"}

# 创建一个 StaticForLoop 类的实例 model，用于后续的模型操作
model = StaticForLoop()
```