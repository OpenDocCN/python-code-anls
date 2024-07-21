# `.\pytorch\test\distributed\pipelining\test_unflatten.py`

```py
# 导入 PyTorch 库
import torch
# 导入分布式训练相关模块
from torch.distributed.pipelining import pipe_split, pipeline
# 导入测试相关工具类和函数
from torch.testing._internal.common_utils import run_tests, TestCase


# 模型的基本构建模块
class Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 定义卷积层，输入通道数为16，输出通道数为16，卷积核大小为3x3，填充为1
        self.conv = torch.nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, padding=1
        )
        # 定义线性层，输入大小为256，输出大小为256
        self.lin0 = torch.nn.Linear(256, 256)
        # 定义 ReLU 激活函数
        self.relu = torch.nn.ReLU()
        # 定义另一个线性层，输入大小为256，输出大小为256
        self.lin1 = torch.nn.Linear(256, 256)

    def forward(self, x: torch.Tensor, constant=None) -> torch.Tensor:
        # 前向传播函数，先卷积，再线性变换，加上常数，再线性变换，最后经过 ReLU 激活函数
        x = self.conv(x)
        x = self.lin0(x)
        # 执行管道拆分操作
        pipe_split()
        x.add_(constant)
        x = self.lin1(x)
        return self.relu(x)


# 完整的模型类
class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 创建两个 Block 类的实例
        self.block0 = Block()
        self.block1 = Block()

    def forward(self, x: torch.Tensor, constant=None) -> torch.Tensor:
        # 模型的前向传播过程，依次通过两个 Block 类实例
        x = self.block0(x, constant=constant)
        # 执行管道拆分操作
        pipe_split()
        x = self.block1(x, constant=constant)
        return x


# 测试类，用于测试 Unflatten 操作
class UnflattenTests(TestCase):
    def test_unflatten(self):
        # 创建输入数据 x 和常数 constant
        x = torch.randn(1, 16, 256, 256)
        constant = torch.ones(1, 16, 256, 256)

        # 创建模型实例
        mod = M()

        # 创建管道，使用 pipeline 函数
        pipe = pipeline(
            mod,
            (x,),
            {"constant": constant},
        )

        # 断言管道的阶段数为4
        assert pipe.num_stages == 4
        # 获取原始模型的状态字典
        orig_state_dict = mod.state_dict()

        # 检查参数的限定名（qualname）
        for stage_idx in range(pipe.num_stages):
            # 获取管道中的每个阶段模型
            stage_mod = pipe.get_stage_module(stage_idx)
            for param_name, param in stage_mod.named_parameters():
                # 断言参数名在原始模型的状态字典中存在
                assert (
                    param_name in orig_state_dict
                ), f"{param_name} not in original state dict"
        print("Param qualname test passed")

        # 检查管道和原始模型输出的等价性
        ref = mod(x, constant)
        out = pipe(x, constant)[0]
        torch.testing.assert_close(out, ref)
        print(f"Equivalence test passed {torch.sum(out)} ref {torch.sum(ref)}")


if __name__ == "__main__":
    # 运行测试
    run_tests()
```