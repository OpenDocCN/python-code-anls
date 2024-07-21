# `.\pytorch\test\dynamo\test_fx_passes_pre_grad.py`

```
# Owner(s): ["module: dynamo"]

# 从 unittest 模块导入 mock 对象
from unittest import mock

# 导入 torch 库
import torch

# 导入 torch._dynamo 模块
import torch._dynamo
# 导入 torch._dynamo.test_case 模块
import torch._dynamo.test_case
# 从 torch._inductor.utils 模块导入 pass_execution_and_save 函数
from torch._inductor.utils import pass_execution_and_save

# 定义一个测试类 FxPassesPreGradTests，继承自 torch._dynamo.test_case.TestCase
class FxPassesPreGradTests(torch._dynamo.test_case.TestCase):

    # 使用 mock.patch 装饰器，模拟 "torch._inductor.utils.ShapeProp.propagate" 方法
    @mock.patch("torch._inductor.utils.ShapeProp.propagate")
    def test_pass_execution_and_save(self, mock_shape_prop):
        # 定义一个内部测试模块类 TestModule，继承自 torch.nn.Module
        class TestModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # 创建一个形状为 (4, 4) 的张量参数 self.param
                self.param = torch.nn.Parameter(torch.ones(4, 4))

            # 实现前向传播方法 forward，接受输入 x，返回 self.param + x
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.param + x

        # 定义一个空的函数 fx_pass，接受 torch.fx.GraphModule 类型的参数 graph，无返回值
        def fx_pass(graph: torch.fx.GraphModule) -> None:
            return

        # 创建一个形状为 (4, 4) 的随机输入样本 sample_input
        sample_input = torch.randn(4, 4)
        # 创建 TestModule 的实例 m
        m = TestModule()
        # 使用实例 m 对样本 sample_input 进行前向传播计算
        m(sample_input)
        # 导出模型 m，传入样本输入 sample_input，获取导出程序 exported_program
        exported_program = torch.export.export(m, (sample_input,))
        # 获取导出程序的图模块 gm
        gm = exported_program.graph_module

        # 调用 pass_execution_and_save 函数，传入 fx_pass 函数、图模块 gm、样本输入 sample_input 和描述字符串
        pass_execution_and_save(fx_pass, gm, sample_input, "Apply testing pass")
        # 断言 mock_shape_prop 方法被调用了一次
        mock_shape_prop.assert_called_once()

# 如果当前脚本被直接执行，则调用 torch._dynamo.test_case 模块的 run_tests 函数执行测试
if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
```