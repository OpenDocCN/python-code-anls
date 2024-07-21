# `.\pytorch\test\inductor\test_codegen_triton.py`

```
# Owner(s): ["module: inductor"]
# 导入上下文管理模块
import contextlib

# 导入 sympy 模块
import sympy

# 导入 PyTorch 库
import torch

# 导入 PyTorch 自定义的模块和函数
import torch._inductor.config as inductor_config
from torch._inductor.codegen import triton_utils
from torch._inductor.codegen.common import SizeArg
from torch._inductor.graph import GraphLowering
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.virtualized import V

# 导入 PyTorch 内部测试工具函数
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_GPU


# 定义测试类 TestCodegenTriton，继承自 InductorTestCase
class TestCodegenTriton(InductorTestCase):

    # 设置测试前的准备工作
    def setUp(self):
        super().setUp()

        # 定义一个简单的 torch.nn.Module 类
        class DummyModule(torch.nn.Module):
            # 定义 forward 方法
            def forward(self, x):
                return x * 2

        # 对 DummyModule 进行符号化跟踪
        self._gm = torch.fx.symbolic_trace(DummyModule())
        # 创建一个 GraphLowering 对象
        self._graph = GraphLowering(self._gm)

        # 创建一个上下文管理堆栈
        self._stack = contextlib.ExitStack()
        # 将 GraphLowering 对象作为上下文管理堆栈的上下文
        self._stack.enter_context(V.set_graph_handler(self._graph))

    # 清理测试后的工作
    def tearDown(self):
        # 关闭上下文管理堆栈
        self._stack.close()
        super().tearDown()

    # 测试 triton 的 SizeArg 配置
    @inductor_config.patch("triton.divisible_by_16", True)
    def test_config_of_sizearg(self):
        # 定义几个 sympy 的符号和整数对象
        two = sympy.Integer(2)
        eight = sympy.Integer(8)
        sixteen = sympy.Integer(16)
        s0 = sympy.Symbol("s0", positive=True, integer=True)
        s1 = sympy.Symbol("s1", positive=True, integer=True)

        # 第一个断言：验证 triton_utils.config_of 的返回值的 divisible_by_16 属性
        self.assertEqual(
            (2,),
            triton_utils.config_of(
                [
                    SizeArg("A", two),  # no
                    SizeArg("B", eight),  # no
                    SizeArg("C", sixteen),  # yes
                    SizeArg("D", s0),  # no
                    SizeArg("E", s1),  # no
                ]
            ).divisible_by_16,
        )

        # 第二个断言：再次验证 triton_utils.config_of 的返回值的 divisible_by_16 属性
        self.assertEqual(
            (0, 2, 4, 5, 6),
            triton_utils.config_of(
                [
                    SizeArg("A", two * eight),  # 0: yes
                    SizeArg("B", eight * s0),  # 1: no
                    SizeArg("C", two * eight * s0),  # 2: yes
                    SizeArg("D", s0 * s1),  # 3: no
                    SizeArg("E", sixteen * s0),  # 4: yes
                    SizeArg("F", sixteen * eight * s0 * s1),  # 5: yes
                    SizeArg("G", two * eight * s0 * s1),  # 6: yes
                ]
            ).divisible_by_16,
        )


# 如果当前脚本为主程序，则执行测试
if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    # 如果有 CPU 或 GPU，运行 sympy 的测试
    if HAS_CPU or HAS_GPU:
        run_tests("sympy")
```