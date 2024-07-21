# `.\pytorch\test\jit\test_custom_operators.py`

```py
# Owner(s): ["oncall: jit"]

# 导入必要的模块
import os
import sys
import unittest

import torch

# 让 test/ 中的辅助文件可以被导入
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

# 导入测试用例基类 JitTestCase
from torch.testing._internal.jit_utils import JitTestCase

# 如果直接运行该文件，则抛出运行时错误，建议使用指定的方式运行测试
if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# 定义一个函数 canonical，用于对图进行规范化处理，并返回字符串表示
def canonical(graph):
    return torch._C._jit_pass_canonicalize(graph).str(False)

# 定义一个测试类 TestCustomOperators，继承自 JitTestCase
class TestCustomOperators(JitTestCase):

    # 测试动态操作注册
    def test_dynamic_op_registry(self):
        from torch._ops import _OpNamespace

        # 断言 torch 模块中有 ops 属性
        self.assertTrue(hasattr(torch, "ops"))

        # 如果 _test 在 torch.ops.__dict__ 中，则将其移除
        if "_test" in torch.ops.__dict__:
            torch.ops.__dict__.pop("_test")

        # 不要使用 hasattr() 检查是否存在 _test，因为会调用 __getattr__
        self.assertNotIn("_test", torch.ops.__dict__)

        # 使用 torch.ops._test 来确保 _test 已经被创建
        torch.ops._test

        # 断言 _test 已经在 torch.ops.__dict__ 中
        self.assertIn("_test", torch.ops.__dict__)

        # 断言 torch.ops._test 的类型是 _OpNamespace
        self.assertEqual(type(torch.ops._test), _OpNamespace)

        # 断言 _test.__dict__ 中不包含 leaky_relu
        self.assertNotIn("leaky_relu", torch.ops._test.__dict__)

        # 获取 torch.ops._test.leaky_relu，并断言其可调用
        op = torch.ops._test.leaky_relu
        self.assertTrue(callable(op))

        # 再次断言 _test.__dict__ 中包含 leaky_relu
        self.assertIn("leaky_relu", torch.ops._test.__dict__)

        # 再次获取 torch.ops._test.leaky_relu，并断言与之前获取的 op 相等
        op2 = torch.ops._test.leaky_relu
        self.assertEqual(op, op2)

    # 测试获取无效属性时的异常情况
    def test_getting_invalid_attr(self):
        for attr in ["__origin__", "__self__"]:
            with self.assertRaisesRegexWithHighlight(
                AttributeError,
                f"Invalid attribute '{attr}' for '_OpNamespace' '_test'",
                "",
            ):
                getattr(torch.ops._test, attr)

    # 测试简单调用一个操作符
    def test_simply_calling_an_operator(self):
        input = torch.randn(100)
        output = torch.ops.aten.relu(input)
        self.assertEqual(output, input.relu())

    # 测试使用默认参数调用操作符
    def test_default_arguments_are_used(self):
        output = torch.ops._test.leaky_relu(torch.tensor([-1.0, 1.0]))
        self.assertEqual(output, torch.tensor([-0.01, 1]))

    # 测试传递过多参数时的异常情况
    def test_passing_too_many_args(self):
        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            r"aten::relu\(\) expected at most 1 argument\(s\) but received 2 argument\(s\)",
            "",
        ):
            torch.ops.aten.relu(1, 2)

    # 测试参数过少时的异常情况
    def test_passing_too_few_args(self):
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, r"aten::relu\(\) is missing value for argument 'self'.", ""
        ):
            torch.ops.aten.relu()

    # 测试传递一个位置参数但没有第二个参数时的异常情况
    def test_passing_one_positional_but_not_the_second(self):
        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            r"aten::type_as\(\) is missing value for argument 'other'.",
            "",
        ):
            torch.ops.aten.type_as(torch.ones(5, 5))
    # 定义测试函数，用于验证当传递未知关键字参数时是否引发异常
    def test_passing_unknown_kwargs(self):
        # 使用断言验证是否会抛出指定异常，并检查异常消息是否包含特定文本
        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            "Unknown keyword argument 'foo' for operator '_test::leaky_relu'",
            "",
        ):
            # 调用自定义的 Torch 操作函数，传递一个未定义的关键字参数 'foo'
            torch.ops._test.leaky_relu(torch.ones(5), foo=torch.ones(5))

    # 定义测试函数，用于验证当传递和返回列表时是否工作正常（这里只是占位，实际测试功能尚不支持）
    def test_passing_and_returning_lists(self):
        # 创建两个随机张量 a 和 b，每个张量包含 5 个元素
        a, b = torch.rand(5), torch.rand(5)
        # 调用自定义 Torch 操作函数 cat，用于在轴上连接张量列表 [a, b]
        output = torch.ops._test.cat([a, b])
        # 使用 Torch 自带的 cat 函数来对比结果
        output_ref = torch.cat([a, b])
        # 断言自定义操作的输出与 Torch 自带操作的输出是否一致
        self.assertEqual(output, output_ref)

    # 定义测试函数，用于验证调用 Torch 脚本中包含自定义操作的情况
    def test_calling_scripted_custom_op(self):
        # 定义一个 Torch 脚本函数 func，该函数调用自定义的 Torch 操作函数 aten.relu
        @torch.jit.script
        def func(x):
            return torch.ops.aten.relu(x)

        # 创建一个输入张量 input，全为 1，大小为 5x5
        input = torch.ones(5, 5)
        # 断言 Torch 脚本函数 func 对输入的操作结果与直接调用 input.relu() 的结果是否一致
        self.assertEqual(func(input), input.relu())

    # 定义测试函数，用于验证调用经过追踪的自定义操作的情况
    def test_calling_traced_custom_op(self):
        # 创建一个输入张量 input，全为 1，大小为 5x5
        input = torch.ones(5, 5)
        # 使用 torch.jit.trace 追踪自定义的 Torch 操作函数 aten.relu，并传入输入 input
        func = torch.jit.trace(torch.ops.aten.relu, [input])
        # 断言追踪得到的函数 func 对输入的操作结果与直接调用 input.relu() 的结果是否一致
        self.assertEqual(func(input), input.relu())

    # 标记为跳过的测试函数，用于验证脚本图与追踪图中自定义操作的默认数据类型差异
    @unittest.skip(
        "Need to figure out default dtype differences between fbcode and oss"
    )
    def test_script_graph_for_custom_ops_matches_traced_graph(self):
        # 创建一个输入张量 input，全为 1，大小为 5x5
        input = torch.ones(5, 5)
        # 使用 torch.jit.trace 追踪自定义的 Torch 操作函数 aten.relu，并传入输入 input
        trace = torch.jit.trace(torch.ops.aten.relu, [input])
        # 断言追踪到的脚本图与追踪图中自定义操作的行为是否一致
        self.assertExpectedInline(
            canonical(trace.graph),
            """\
    def test_script_graph_contains_custom_op(self):
        # 定义一个使用 Torch 脚本装饰器的函数
        @torch.jit.script
        def func(x):
            # 调用 Torch 操作符 relu
            return torch.ops.aten.relu(x)

        # 断言函数的规范化图形与预期输出相符
        self.assertExpectedInline(
            canonical(func.graph),
            """\
graph(%x.1 : Tensor):
  %1 : Tensor = aten::relu(%x.1)
  return (%1)
""",
        )

    def test_generic_list(self):
        # 断言获取列表的第一个元素是否为 "hello"
        self.assertEqual(torch.ops._test.get_first([["hello"]]), "hello")

    # https://github.com/pytorch/pytorch/issues/80508
    def test_where_no_scalar(self):
        # 创建一个随机张量 x
        x = torch.rand(1, 3, 224, 224)
        # 调用 Torch 操作符 where，条件为 x > 0.5，如果条件不满足则不抛出异常
        torch.ops.aten.where(x > 0.5, -1.5, 1.5)  # does not raise
```