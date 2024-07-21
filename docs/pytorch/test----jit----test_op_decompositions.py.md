# `.\pytorch\test\jit\test_op_decompositions.py`

```
# Owner(s): ["oncall: jit"]

# 导入 PyTorch 相关模块
import torch
from torch.testing import FileCheck
from torch.testing._internal.jit_utils import JitTestCase

# 如果该脚本作为主程序执行，抛出运行时错误，建议使用测试框架运行
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# 定义一个测试类，继承自 JitTestCase
class TestOpDecompositions(JitTestCase):
    
    # 测试函数：测试操作分解
    def test_op_decomposition(self):
        
        # 定义一个简单的函数 foo，计算输入张量 x 的方差
        def foo(x):
            return torch.var(x, unbiased=True)

        # 使用 TorchScript 对函数 foo 进行脚本化
        foo_s = torch.jit.script(foo)
        
        # 使用 FileCheck 检查脚本化后的函数图中是否包含 "aten::var" 操作
        FileCheck().check("aten::var").run(foo_s.graph)
        
        # 运行 Torch JIT 的操作分解传递，优化函数图
        torch._C._jit_pass_run_decompositions(foo_s.graph)
        
        # 创建一个随机张量作为输入
        inp = torch.rand([10, 10])
        
        # 断言原始函数和脚本化函数在相同输入上的结果一致
        self.assertEqual(foo(inp), foo_s(inp))
        
        # 使用 FileCheck 检查脚本化后的函数图中是否不再包含 "aten::var" 操作
        FileCheck().check_not("aten::var").run(foo_s.graph)

    # 测试函数：注册操作分解
    def test_registered_decomposition(self):
        
        # 使用 TorchScript 对函数 foo 进行脚本化
        @torch.jit.script
        def foo(x):
            return torch.square(x)

        # 使用 TorchScript 对函数 square_decomp 进行脚本化
        @torch.jit.script
        def square_decomp(x):
            return torch.pow(x, 2)

        # 注册自定义的操作分解，将 torch.ops.aten.square.default 替换为 square_decomp 的图形表示
        torch.jit._register_decomposition(
            torch.ops.aten.square.default, square_decomp.graph
        )
        
        # 运行 Torch JIT 的操作分解传递，优化函数图
        torch._C._jit_pass_run_decompositions(foo.graph)
        
        # 使用 FileCheck 检查函数图中是否不再包含 "aten::square" 操作，并且包含 "aten::pow" 操作
        FileCheck().check_not("aten::square").check("aten::pow").run(foo.graph)
        
        # 创建一个随机张量作为输入
        x = torch.rand([4])
        
        # 断言注册后的函数在相同输入上与 torch.square 函数的结果一致
        self.assertEqual(foo(x), torch.square(x))
```