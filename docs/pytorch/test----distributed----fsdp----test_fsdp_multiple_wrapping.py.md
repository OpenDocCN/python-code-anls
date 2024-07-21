# `.\pytorch\test\distributed\fsdp\test_fsdp_multiple_wrapping.py`

```
# Owner(s): ["oncall: distributed"]

# 导入系统模块
import sys

# 导入 PyTorch 库及分布式相关模块
import torch
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import Linear, Module, Sequential
from torch.optim import SGD

# 导入测试相关的模块和函数
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN

# 如果分布式不可用，则输出错误信息并退出
if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

# 如果开启了开发者调试模式（dev-asan），则输出相应提示信息并退出
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)

# 内部模型类，继承自 Module
class InnerModel(Module):
    def __init__(self):
        super().__init__()
        # 创建包含 FSDP 包装的线性层的顺序模型
        self.layers = Sequential(FSDP(Linear(5, 5)))

    # 前向传播函数
    def forward(self, x):
        return self.layers(x)

# 测试类，继承自 FSDPTest
class TestMultipleWrapping(FSDPTest):
    
    # 修饰器函数，用于在 GPU 数量少于 2 时跳过测试
    @skip_if_lt_x_gpu(2)
    def test_multiple_wrapping(self):
        """
        This test simulates wrapping the module after training to run inference.
        This is required in cases where later in a session, the model is wrapped again in FSDP but
        contains nested FSDP wrappers within the module.
        """
        # 创建内部模型实例
        inner_model = InnerModel()
        # 使用 FSDP 包装内部模型，并移动到 GPU
        model = FSDP(inner_model).cuda()
        # 使用 SGD 优化器
        optim = SGD(model.parameters(), lr=0.1)

        # 训练循环
        for i in range(3):
            input = torch.rand((1, 5), dtype=torch.float).cuda()
            input.requires_grad = True
            output = model(input)
            output.sum().backward()
            optim.step()
            optim.zero_grad()
        
        # 运行推理
        input = torch.rand((1, 5), dtype=torch.float).cuda()
        output = model(input)

        # 第二次使用 FSDP 重新包装内部模型
        rewrapped_model = FSDP(inner_model).cuda()
        rewrapped_output = rewrapped_model(input)

        # 断言输出与重新包装后的输出相等
        self.assertEqual(output, rewrapped_output)

# 如果是主程序入口，则运行测试
if __name__ == "__main__":
    run_tests()
```