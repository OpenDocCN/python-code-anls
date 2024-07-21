# `.\pytorch\test\distributed\fsdp\test_fsdp_fx.py`

```
# Owner(s): ["oncall: distributed"]

# 导入PyTorch库
import torch
# 导入分布式FSDP模块的跟踪工具函数
from torch.distributed.fsdp._trace_utils import _ExecOrderTracer
# 导入PyTorch内部测试的常用工具函数和类
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
    TestCase,
)

# 定义一个模型类，继承自torch.nn.Module
class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 定义模型参数
        self.weight1 = torch.nn.Parameter(torch.randn(6, 6))
        self.weight2 = torch.nn.Parameter(torch.randn(6, 6))
        self.weight_unused = torch.nn.Parameter(torch.randn(2, 2))
        # 定义模型层
        self.layer0 = torch.nn.Linear(6, 6)
        self.layer1 = torch.nn.Linear(6, 6, bias=False)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(6, 3, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(3, 6, bias=False),
        )
        self.relu = torch.nn.ReLU()

    # 前向传播函数，接收输入张量x和运行所有层的标志run_all_layers
    def forward(self, x: torch.Tensor, run_all_layers: bool) -> torch.Tensor:
        # 第一层处理
        z = self.relu(self.layer0(x))
        # 第二层处理
        z = self.relu(self.layer2(z))
        # 使用weight1进行线性变换
        z = z @ self.weight1
        # 如果需要运行所有层
        if run_all_layers:
            # 第三层处理（如果运行所有层）
            z = self.relu(self.layer1(z))
            # 使用weight2进行线性变换（如果运行所有层）
            z = z @ self.weight2
            # 再次使用layer0以检查在保存数据结构中对多重性的处理
            z = self.relu(self.layer0(x))
        # 返回处理后的张量z
        return z


# 定义测试类TestSymbolicTracing，继承自TestCase
class TestSymbolicTracing(TestCase):
    # 实例化参数化测试
    instantiate_parametrized_tests(TestSymbolicTracing)

# 如果该脚本作为主程序运行，则执行测试
if __name__ == "__main__":
    run_tests()
```