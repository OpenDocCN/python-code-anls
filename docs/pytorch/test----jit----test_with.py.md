# `.\pytorch\test\jit\test_with.py`

```
# Owner(s): ["oncall: jit"]

import os  # 导入操作系统相关的模块
import sys  # 导入系统相关的模块

from typing import Any, List  # 导入类型提示相关的模块

import torch  # 导入PyTorch深度学习库
from torch.testing._internal.common_utils import skipIfTorchDynamo  # 从测试工具中导入条件跳过装饰器
from torch.testing._internal.jit_utils import JitTestCase, make_global  # 从测试工具中导入JIT测试基类和全局变量创建函数

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # 获取当前文件所在目录的上层目录
sys.path.append(pytorch_test_dir)  # 将该目录添加到系统路径中，使得其中的辅助文件可以被导入使用

if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )  # 如果脚本被直接运行，抛出运行时错误提示用户正确的运行方式

class TestWith(JitTestCase):
    """
    A suite of tests for with statements.
    """

    def test_with_no_grad(self):
        """
        Check that torch.no_grad() works. Most of these are adapted from
        corresponding tests for eager-mode no_grad.
        """

        # Basic no_grad test.
        def test_no_grad(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():  # 使用torch.no_grad()上下文管理器，禁用梯度计算
                w = x + y  # 执行张量操作，不会记录梯度信息

            return w  # 返回结果张量

        s = torch.jit.script(test_no_grad)  # 对test_no_grad函数进行脚本化，转换为Torch脚本
        x = torch.ones(5, 5, requires_grad=True)  # 创建一个需要梯度的张量
        y = torch.ones(5, 5) * 4  # 创建一个张量
        w = s(x, y)  # 执行脚本化的函数

        self.assertFalse(w.requires_grad)  # 断言结果张量不需要梯度
        self.assertRaises(RuntimeError, lambda: w.backward(torch.ones(5, 5)))  # 断言在禁用梯度时调用反向传播会引发运行时错误
        self.assertIsNone(w.grad_fn)  # 断言结果张量没有梯度函数

        # Test assignment of a grad-less Tensor to a Tensor with gradients
        # in a no_grad block.
        def test_no_grad_assignment(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():  # 使用torch.no_grad()上下文管理器，禁用梯度计算
                x[0] = y  # 在禁用梯度的块中将y的值赋给x的第一个元素

            return x  # 返回修改后的张量

        s = torch.jit.script(test_no_grad_assignment)  # 对test_no_grad_assignment函数进行脚本化
        z = torch.randn(5)  # 创建一个张量
        w = s(x, z)  # 执行脚本化的函数
        self.assertTrue(w.requires_grad)  # 断言结果张量仍然需要梯度
        self.assertIsNone(w.grad_fn)  # 断言结果张量没有梯度函数

        # Check that @torch.jit.ignored functions respect no_grad when it is
        # called in JIT mode.
        class NoGradModule(torch.nn.Module):
            @torch.jit.ignore
            def adder(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                w = x + y  # 执行张量操作

                return w  # 返回结果张量

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                with torch.no_grad():  # 使用torch.no_grad()上下文管理器，禁用梯度计算
                    w = self.adder(x, y)  # 调用被@torch.jit.ignore修饰的方法

                return w  # 返回结果张量

        s = torch.jit.script(NoGradModule())  # 对NoGradModule模块进行脚本化
        w = s(x, y)  # 执行脚本化的模块

        self.assertFalse(w.requires_grad)  # 断言结果张量不需要梯度

    @skipIfTorchDynamo("Torchdynamo cannot correctly handle profiler.profile calls")
    # 定义一个测试函数，用于验证 torch.autograd.profiler.record_function 上下文管理器在 torchscript 中的可用性
    def test_with_record_function(self):
        """
        Check that torch.autograd.profiler.record_function context manager is
        torchscriptable.
        """

        # 定义一个函数，使用 record_function 上下文管理器来记录操作
        def with_rf(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # 进入名为 "foo" 的 record_function 上下文
            with torch.autograd.profiler.record_function("foo"):
                # 嵌套的 record_function
                with torch.autograd.profiler.record_function("nested"):
                    # 执行张量的加法操作
                    a = x + y
            return a  # 返回加法结果 a

        # 对 with_rf 函数进行 TorchScript 脚本化
        scripted = torch.jit.script(with_rf)
        
        # 创建两个包含全 1 的张量 x 和 y
        x, y = torch.ones(2), torch.ones(2)
        
        # 使用 profiler 对脚本化的函数进行性能分析
        with torch.autograd.profiler.profile() as p:
            scripted(x, y)

        # 需要调用 key_averages 方法来填充 CPU 子事件
        p.key_averages()
        
        # 获取函数事件列表
        function_events = p.function_events
        
        # 检查是否记录了名为 "foo" 的事件
        rf_events = [evt for evt in function_events if evt.name == "foo"]
        self.assertEqual(len(rf_events), 1)
        rf_event = rf_events[0]
        
        # 获取 "foo" 事件的 CPU 子事件
        child_events = rf_event.cpu_children
        
        # 确保我们找到了嵌套的 record_function 事件 "nested"
        self.assertTrue("nested" in (child.name for child in child_events))
        
        # 获取 "nested" record function 事件
        nested_function_event = [
            evt for evt in function_events if evt.name == "nested"
        ][0]
        
        # 获取嵌套事件 "nested" 的 CPU 子事件
        nested_child_events = nested_function_event.cpu_children
        
        # 确保嵌套的 record function 事件有一个名为 "aten::add" 的子事件
        self.assertTrue("aten::add" in (child.name for child in nested_child_events))
```