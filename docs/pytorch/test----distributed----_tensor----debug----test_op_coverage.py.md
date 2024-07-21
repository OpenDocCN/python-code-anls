# `.\pytorch\test\distributed\_tensor\debug\test_op_coverage.py`

```py
# Owner(s): ["oncall: distributed"]

# 导入 PyTorch 库和模块
import torch
import torch.nn as nn

# 导入用于获取张量操作覆盖率图的函数
from torch.distributed._tensor.debug._op_coverage import get_inductor_decomp_graphs

# 导入测试相关的工具函数和类
from torch.testing._internal.common_utils import run_tests, TestCase

# 定义一个简单的多层感知机模型类
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义模型的第一层线性层，输入大小为 50，输出大小为 32
        self.net1 = nn.Linear(50, 32)
        # 定义激活函数 ReLU
        self.relu = nn.ReLU()
        # 定义模型的第二层线性层，输入大小为 32，输出大小为 8
        self.net2 = nn.Linear(32, 8)

    # 定义模型的前向传播函数
    def forward(self, x):
        # 应用 net1、relu 和 net2，最后使用 sigmoid 激活函数输出
        return torch.sigmoid(self.net2(self.relu(self.net1(x))))


# 定义测试类 TestOpCoverage，继承自 TestCase
class TestOpCoverage(TestCase):
    # 定义测试方法，测试带有感应器分解的跟踪
    def test_trace_with_inductor_decomp(self):
        # 创建 SimpleMLP 模型的实例
        model = SimpleMLP()
        # 创建输入参数 args，这里是一个大小为 (8, 50) 的随机张量
        args = (torch.randn(8, 50),)
        # 创建空的关键字参数 kwargs
        kwargs = {}
        # 调用 get_inductor_decomp_graphs 函数获取模型的感应器分解图
        graphs = get_inductor_decomp_graphs(model, args, kwargs)
        # 断言 graphs 的长度为 2，表示期望有前向传播和反向传播两个图
        assert len(graphs) == 2, "Expect fwd + bwd graphs"
        # 断言 graphs[0] 的类型为 torch.fx.GraphModule
        self.assertIsInstance(graphs[0], torch.fx.GraphModule)
        # 断言 graphs[1] 的类型为 torch.fx.GraphModule
        self.assertIsInstance(graphs[1], torch.fx.GraphModule)

# 如果当前脚本是主程序，则运行测试
if __name__ == "__main__":
    run_tests()
```