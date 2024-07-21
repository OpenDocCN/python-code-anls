# `.\pytorch\test\inductor\test_torchbind.py`

```
# Owner(s): ["module: functorch"]

# 导入 PyTorch 库
import torch
# 导入 PyTorch 的内部模块
import torch._dynamo
import torch._functorch
import torch._inductor
import torch._inductor.decomposition
# 导入 torchbind 相关函数
from torch._higher_order_ops.torchbind import enable_torchbind_tracing
# 导入测试相关函数和类
from torch._inductor.test_case import run_tests, TestCase
# 导入内部测试实现
from torch.testing._internal.torchbind_impls import init_torchbind_implementations

# 定义测试类 TestTorchbind，继承自 TestCase
class TestTorchbind(TestCase):
    # 设置测试前的准备工作
    def setUp(self):
        super().setUp()
        # 初始化 torchbind 实现
        init_torchbind_implementations()

    # 获取导出模型的方法
    def get_exported_model(self):
        """
        Returns the ExportedProgram, example inputs, and result from calling the
        eager model with those inputs
        """
        # 定义内部类 M，继承自 torch.nn.Module
        class M(torch.nn.Module):
            # 初始化函数
            def __init__(self):
                super().__init__()
                # 创建一个 torch.classes._TorchScriptTesting._Foo 实例
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)
                # 创建一个形状为 (2, 3) 的随机张量
                self.b = torch.randn(2, 3)

            # 前向传播函数
            def forward(self, x):
                # 张量 x 加上 self.b
                x = x + self.b
                # 调用 torch.ops._TorchScriptTesting.takes_foo_tuple_return 方法
                a = torch.ops._TorchScriptTesting.takes_foo_tuple_return(self.attr, x)
                # 计算 a 中两个元素的和
                y = a[0] + a[1]
                # 调用 torch.ops._TorchScriptTesting.takes_foo 方法
                b = torch.ops._TorchScriptTesting.takes_foo(self.attr, y)
                # 返回 x 加上 b 的结果
                return x + b

        # 创建 M 类的实例 m
        m = M()
        # 创建输入数据，这里是形状为 (2, 3) 的全为 1 的张量
        inputs = (torch.ones(2, 3),)
        # 使用输入数据计算原始结果 orig_res
        orig_res = m(*inputs)

        # 通过 enable_torchbind_tracing 开启 torchbind 追踪
        with enable_torchbind_tracing():
            # 导出模型 m，并得到 ExportedProgram 对象 ep
            ep = torch.export.export(m, inputs, strict=False)

        # 返回 ExportedProgram 对象 ep，输入数据 inputs，和原始结果 orig_res
        return ep, inputs, orig_res

    # 测试 torchbind inductor 的方法
    def test_torchbind_inductor(self):
        # 获取导出模型的 ExportedProgram 对象 ep，输入数据 inputs，和原始结果 orig_res
        ep, inputs, orig_res = self.get_exported_model()
        # 编译 ExportedProgram 对象 ep，使用输入数据 inputs
        compiled = torch._inductor.compile(ep.module(), inputs)

        # 使用编译后的模型计算新的结果 new_res
        new_res = compiled(*inputs)
        # 断言原始结果 orig_res 与新的结果 new_res 相近
        self.assertTrue(torch.allclose(orig_res, new_res))

# 如果当前脚本是主程序，运行测试
if __name__ == "__main__":
    run_tests()
```