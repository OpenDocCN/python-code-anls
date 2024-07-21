# `.\pytorch\test\custom_operator\test_custom_ops.py`

```
# Owner(s): ["module: unknown"]

# 导入必要的库和模块
import os.path
import sys
import tempfile
import unittest

# 从自定义模块中导入函数和类
from model import get_custom_op_library_path, Model

# 导入 PyTorch 相关库和模块
import torch
import torch._library.utils as utils
from torch import ops
from torch.testing._internal.common_utils import IS_WINDOWS, run_tests, TestCase

# 导入自定义的运算符模块
torch.ops.import_module("pointwise")


class TestCustomOperators(TestCase):
    def setUp(self):
        # 设置测试环境：获取自定义运算符库的路径并加载
        self.library_path = get_custom_op_library_path()
        ops.load_library(self.library_path)

    def test_custom_library_is_loaded(self):
        # 测试：确保自定义库已加载
        self.assertIn(self.library_path, ops.loaded_libraries)

    def test_op_with_no_abstract_impl_pystub(self):
        # 测试：对没有抽象实现的运算符进行调用（假装有Python存根）
        x = torch.randn(3, device="meta")
        if utils.requires_set_python_module():
            # 在需要设置 Python 模块时，测试是否会抛出指定异常
            with self.assertRaisesRegex(RuntimeError, "pointwise"):
                torch.ops.custom.tan(x)
        else:
            # 烟雾测试：简单调用运算符
            torch.ops.custom.tan(x)

    def test_op_with_incorrect_abstract_impl_pystub(self):
        # 测试：对具有不正确抽象实现的运算符进行调用（假装有Python存根）
        x = torch.randn(3, device="meta")
        with self.assertRaisesRegex(RuntimeError, "pointwise"):
            torch.ops.custom.cos(x)

    @unittest.skipIf(IS_WINDOWS, "torch.compile not supported on windows")
    def test_dynamo_pystub_suggestion(self):
        # 测试：对动态编译运算符（假装有Python存根）的建议
        x = torch.randn(3)

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            return torch.ops.custom.asin(x)

        with self.assertRaisesRegex(
            RuntimeError,
            r"unsupported operator: .* you may need to `import nonexistent`",
        ):
            f(x)

    def test_abstract_impl_pystub_faketensor(self):
        # 测试：对具有假张量的抽象实现运算符进行调用（假装有Python存根）
        from functorch import make_fx

        x = torch.randn(3, device="cpu")
        self.assertNotIn("my_custom_ops", sys.modules.keys())

        with self.assertRaises(
            torch._subclasses.fake_tensor.UnsupportedOperatorException
        ):
            gm = make_fx(torch.ops.custom.nonzero.default, tracing_mode="symbolic")(x)

        torch.ops.import_module("my_custom_ops")
        gm = make_fx(torch.ops.custom.nonzero.default, tracing_mode="symbolic")(x)
        self.assertExpectedInline(
            """\
def forward(self, arg0_1):
    nonzero = torch.ops.custom.nonzero.default(arg0_1);  arg0_1 = None
    return nonzero
""".strip(),
            gm.code.strip(),
        )

    def test_abstract_impl_pystub_meta(self):
        # 测试：对具有元设备的抽象实现运算符进行调用（假装有Python存根）
        x = torch.randn(3, device="meta")
        self.assertNotIn("my_custom_ops2", sys.modules.keys())
        with self.assertRaisesRegex(NotImplementedError, r"'my_custom_ops2'"):
            y = torch.ops.custom.sin.default(x)
        torch.ops.import_module("my_custom_ops2")
        y = torch.ops.custom.sin.default(x)

    def test_calling_custom_op_string(self):
        # 测试：调用自定义运算符处理字符串输入
        output = ops.custom.op2("abc", "def")
        self.assertLess(output, 0)
        output = ops.custom.op2("abc", "abc")
        self.assertEqual(output, 0)
    # 测试调用自定义操作函数 `ops.custom.op`
    def test_calling_custom_op(self):
        # 调用自定义操作函数 `op`，传入参数为一个包含5个元素的张量、浮点数2.0和整数3
        output = ops.custom.op(torch.ones(5), 2.0, 3)
        # 断言输出的类型为列表
        self.assertEqual(type(output), list)
        # 断言输出列表的长度为3
        self.assertEqual(len(output), 3)
        # 对输出的每个张量进行遍历
        for tensor in output:
            # 断言每个张量与全为2的张量在接近范围内相等
            self.assertTrue(tensor.allclose(torch.ones(5) * 2))

        # 调用具有默认参数的自定义操作函数 `op_with_defaults`，传入参数为一个包含5个元素的张量
        output = ops.custom.op_with_defaults(torch.ones(5))
        # 断言输出的类型为列表
        self.assertEqual(type(output), list)
        # 断言输出列表的长度为1
        self.assertEqual(len(output), 1)
        # 断言输出列表中的唯一元素与全为1的张量在接近范围内相等
        self.assertTrue(output[0].allclose(torch.ones(5)))

    # 测试调用带自动求导的自定义操作函数 `ops.custom.op_with_autograd`
    def test_calling_custom_op_with_autograd(self):
        # 创建两个形状为(5, 5)的张量，并设置它们需要进行自动求导
        x = torch.randn((5, 5), requires_grad=True)
        y = torch.randn((5, 5), requires_grad=True)
        # 调用带自动求导的自定义操作函数 `op_with_autograd`，传入参数为张量 x、浮点数2和张量 y
        output = ops.custom.op_with_autograd(x, 2, y)
        # 断言输出张量与表达式 x + 2 * y + x * y 在接近范围内相等
        self.assertTrue(output.allclose(x + 2 * y + x * y))

        # 创建一个形状为空的张量 go，并设置它需要进行自动求导
        go = torch.ones((), requires_grad=True)
        # 对输出张量的和进行反向传播
        output.sum().backward(go, False, True)
        # 创建一个全为1的5x5张量 grad
        grad = torch.ones(5, 5)

        # 断言张量 x 的梯度与 y 加上 grad 在接近范围内相等
        self.assertEqual(x.grad, y + grad)
        # 断言张量 y 的梯度与 x 加上 grad 的两倍在接近范围内相等
        self.assertEqual(y.grad, x + grad * 2)

        # 测试带可选参数 z 的情况
        # 清零张量 x 和 y 的梯度
        x.grad.zero_()
        y.grad.zero_()
        # 创建一个形状为(5, 5)的张量 z，并设置它需要进行自动求导
        z = torch.randn((5, 5), requires_grad=True)
        # 再次调用带自动求导的自定义操作函数 `op_with_autograd`，传入参数为张量 x、浮点数2、张量 y 和张量 z
        output = ops.custom.op_with_autograd(x, 2, y, z)
        # 断言输出张量与表达式 x + 2 * y + x * y + z 在接近范围内相等
        self.assertTrue(output.allclose(x + 2 * y + x * y + z))

        # 对输出张量的和进行反向传播
        output.sum().backward(go, False, True)
        # 断言张量 x 的梯度与 y 加上 grad 在接近范围内相等
        self.assertEqual(x.grad, y + grad)
        # 断言张量 y 的梯度与 x 加上 grad 的两倍在接近范围内相等
        self.assertEqual(y.grad, x + grad * 2)
        # 断言张量 z 的梯度与 grad 在接近范围内相等
        self.assertEqual(z.grad, grad)

    # 测试在无梯度模式下调用带自动求导的自定义操作函数 `ops.custom.op_with_autograd`
    def test_calling_custom_op_with_autograd_in_nograd_mode(self):
        # 进入无梯度上下文
        with torch.no_grad():
            # 创建两个形状为(5, 5)的张量，并设置它们需要进行自动求导
            x = torch.randn((5, 5), requires_grad=True)
            y = torch.randn((5, 5), requires_grad=True)
            # 调用带自动求导的自定义操作函数 `op_with_autograd`，传入参数为张量 x、浮点数2和张量 y
            output = ops.custom.op_with_autograd(x, 2, y)
            # 断言输出张量与表达式 x + 2 * y + x * y 在接近范围内相等
            self.assertTrue(output.allclose(x + 2 * y + x * y))

    # 测试在脚本模块中调用自定义操作函数 `ops.custom.op`
    def test_calling_custom_op_inside_script_module(self):
        # 创建一个模型对象
        model = Model()
        # 调用模型对象的前向传播方法 `forward`，传入参数为一个包含5个元素的张量
        output = model.forward(torch.ones(5))
        # 断言输出张量与全为2的张量在接近范围内相等
        self.assertTrue(output.allclose(torch.ones(5) + 1))

    # 测试保存和加载带自定义操作函数的脚本模块
    def test_saving_and_loading_script_module_with_custom_op(self):
        # 创建一个模型对象
        model = Model()
        # 创建一个临时文件对象
        file = tempfile.NamedTemporaryFile(delete=False)
        try:
            # 关闭临时文件
            file.close()
            # 将模型保存到临时文件中
            model.save(file.name)
            # 加载保存的模型文件
            loaded = torch.jit.load(file.name)
        finally:
            # 尝试删除临时文件
            os.unlink(file.name)

        # 调用加载的模型对象的前向传播方法 `forward`，传入参数为一个包含5个元素的张量
        output = loaded.forward(torch.ones(5))
        # 断言输出张量与全为2的张量在接近范围内相等
        self.assertTrue(output.allclose(torch.ones(5) + 1))
# 如果当前脚本被直接执行（而不是被导入为模块），则执行以下代码
if __name__ == "__main__":
    # 调用名为 run_tests() 的函数，通常用于执行测试
    run_tests()
```