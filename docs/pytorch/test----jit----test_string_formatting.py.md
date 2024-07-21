# `.\pytorch\test\jit\test_string_formatting.py`

```py
# Owner(s): ["oncall: jit"]

# 导入必要的模块
import os  # 导入操作系统接口模块
import sys  # 导入系统相关模块
from typing import List  # 导入类型提示中的List

import torch  # 导入PyTorch模块

# 将test/中的辅助文件变为可导入状态
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase  # 导入测试工具类

if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

# 定义测试类TestStringFormatting，继承JitTestCase
class TestStringFormatting(JitTestCase):

    # 测试取模运算符
    def test_modulo_operator(self):
        def fn(dividend: int, divisor: int) -> int:
            return dividend % divisor

        self.checkScript(fn, (5, 2))

    # 测试字符串插值，使用字符串占位符和字符串变量
    def test_string_interpolation_with_string_placeholder_and_string_variable(self):
        def fn(arg1: str):
            return "%s in template" % arg1

        self.checkScript(fn, ("foo",))

    # 测试字符串插值，使用字符串占位符和格式化字符串变量
    def test_string_interpolation_with_string_placeholder_and_format_string_variable(
        self,
    ):
        def fn(arg1: str):
            return arg1 % "foo"

        self.checkScript(fn, ("%s in template",))

    # 测试字符串插值，包含双百分号在字符串中
    def test_string_interpolation_with_double_percent_in_string(self):
        def fn(arg1: str):
            return "%s in template %%" % arg1

        self.checkScript(fn, ("foo",))

    # 测试字符串插值，包含百分号在字符串中，引发运行时错误
    def test_string_interpolation_with_percent_in_string(self):
        @torch.jit.script
        def fn(arg1: str) -> str:
            return "%s in template %" % arg1  # noqa: F501

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Incomplete format specifier", '"%s in template %" % arg1'
        ):
            fn("foo")

    # 测试字符串插值，使用字符串占位符和数字变量
    def test_string_interpolation_with_string_placeholder_and_digit_variable(self):
        def fn(arg1: int) -> str:
            return "%s in template" % arg1

        self.checkScript(fn, (1,))

    # 测试字符串插值，使用数字占位符和数字变量
    def test_string_interpolation_with_digit_placeholder_and_digit_variable(self):
        def fn(arg1: int) -> str:
            return "%d in template" % arg1

        self.checkScript(fn, (1,))

    # 测试字符串插值，使用替代数字占位符
    def test_string_interpolation_with_alternate_digit_placeholder(self):
        def fn(arg1: int) -> str:
            return "%i in template" % arg1

        self.checkScript(fn, (1,))

    # 测试字符串插值，使用数字占位符和字符串变量，引发运行时错误
    def test_string_interpolation_with_digit_placeholder_and_string_variable(self):
        @torch.jit.script
        def fn(arg1: str) -> str:
            return "%d in template" % arg1

        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            "%d requires a number for formatting, but got String",
            '"%d in template" % arg1',
        ):
            fn("1")
    # 定义测试函数，测试带有指数占位符和字符串变量的字符串插值
    def test_string_interpolation_with_exponent_placeholder_and_string_variable(self):
        # 使用 torch.jit.script 装饰器将函数转换为 TorchScript 脚本
        @torch.jit.script
        def fn(arg1: str) -> str:
            # 返回带有指数占位符的模板字符串
            return "%e in template" % arg1

        # 使用断言检查运行时错误，确保正确捕获到异常
        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            "%e requires a number for formatting, but got String",
            '"%e in template" % arg1',
        ):
            # 调用函数并传入字符串参数
            fn("1")

    # 测试带有小写指数占位符和整数变量的字符串插值
    def test_string_interpolation_with_lowercase_exponent_placeholder_and_digit_variable(
        self,
    ):
        # 定义普通 Python 函数
        def fn(arg1: int) -> str:
            # 返回带有小写指数占位符的模板字符串
            return "%e in template" % arg1

        # 调用自定义的测试函数进行脚本检查
        self.checkScript(fn, (1,))

    # 测试带有大写指数占位符和整数变量的字符串插值
    def test_string_interpolation_with_capital_exponent_placeholder_and_digit_variable(
        self,
    ):
        # 定义普通 Python 函数
        def fn(arg1: int) -> str:
            # 返回带有大写指数占位符的模板字符串
            return "%E in template" % arg1

        # 调用自定义的测试函数进行脚本检查
        self.checkScript(fn, (1,))

    # 测试带有浮点数占位符和浮点数变量的字符串插值
    def test_string_interpolation_with_float_placeholder_and_float_variable(self):
        # 定义普通 Python 函数
        def fn(arg1: float) -> str:
            # 返回带有浮点数占位符的模板字符串
            return "%f in template" % arg1

        # 调用自定义的测试函数进行脚本检查，传入浮点数参数
        self.checkScript(fn, (1.0,))

    # 测试带有浮点数占位符和整数变量的字符串插值
    def test_string_interpolation_with_float_placeholder_and_digit_variable(self):
        # 定义普通 Python 函数
        def fn(arg1: int) -> str:
            # 返回带有浮点数占位符的模板字符串
            return "%f in template" % arg1

        # 调用自定义的测试函数进行脚本检查，传入整数参数
        self.checkScript(fn, (1,))

    # 测试带有字符占位符和字符变量的字符串插值
    def test_string_interpolation_with_char_placeholder_and_char_variable(self):
        # 定义普通 Python 函数
        def fn(arg1: str) -> str:
            # 返回带有字符占位符的模板字符串
            return "%c in template" % arg1

        # 调用自定义的测试函数进行脚本检查，传入字符参数
        self.checkScript(fn, ("a",))

    # 测试带有字符占位符和整数变量的字符串插值
    def test_string_interpolation_with_char_placeholder_and_digit_variable(self):
        # 定义普通 Python 函数
        def fn(arg1: int) -> str:
            # 返回带有字符占位符的模板字符串
            return "%c in template" % arg1

        # 调用自定义的测试函数进行脚本检查，传入整数参数
        self.checkScript(fn, (97,))

    # 测试带有字符占位符和真实字符串变量的字符串插值
    def test_string_interpolation_with_char_placeholder_and_true_string_variable(self):
        # 使用 torch.jit.script 装饰器将函数转换为 TorchScript 脚本
        @torch.jit.script
        def fn(arg1: str) -> str:
            # 返回带有字符占位符的模板字符串
            return "%c in template" % arg1

        # 使用断言检查运行时错误，确保正确捕获到异常
        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            "%c requires an int or char for formatting, but got String",
            '"%c in template" % arg1',
        ):
            # 调用函数并传入字符串参数
            fn("foo")

    # 测试带有多个占位符的字符串插值
    def test_string_interpolation_with_multiple_placeholders(self):
        # 定义普通 Python 函数，接受不同类型的参数
        def fn(arg1: str, arg2: int, arg3: float) -> str:
            # 返回带有多个占位符的模板字符串
            return "%s %d %f in template" % (arg1, arg2, arg3)

        # 调用自定义的测试函数进行脚本检查，传入参数组成的元组
        self.checkScript(fn, ("foo", 1, 1))

    # 测试带有下标操作的字符串插值
    def test_string_interpolation_with_subscript(self):
        # 定义普通 Python 函数，接受字符串列表作为参数
        def fn(arg1: List[str]) -> str:
            # 返回带有下标操作的模板字符串
            return "%s in template" % arg1[0]

        # 调用自定义的测试函数进行脚本检查，传入包含字符串的列表作为参数
        self.checkScript(fn, (["foo", "bar"],))

    # 测试带有不足参数的字符串插值
    def test_string_interpolation_with_too_few_arguments(self):
        # 使用 torch.jit.script 装饰器将函数转换为 TorchScript 脚本
        @torch.jit.script
        def fn(arg1: str) -> str:
            # 返回带有多个占位符但参数不足的模板字符串
            return "%s %s in template" % arg1

        # 使用断言检查运行时错误，确保正确捕获到异常
        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            "Too few arguments for format string",
            '"%s %s in template" % arg1',
        ):
            # 调用函数并传入字符串参数
            fn("foo")
    # 定义测试方法，检验字符串插值时使用了过多的参数的情况
    def test_string_interpolation_with_too_many_arguments(self):
        # 使用 TorchScript 注解声明一个函数，接受两个参数并返回字符串
        @torch.jit.script
        def fn(arg1: str, arg2: str) -> str:
            # 使用 %s 占位符进行字符串格式化，引发 F507 错误时不报告
            return "%s in template" % (arg1, arg2)  # noqa: F507
        
        # 断言运行时错误，并在错误消息中突出显示“Too many arguments for format string”
        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            "Too many arguments for format string",
            '"%s in template" % (arg1, arg2',  # 检查异常消息时期望的字符串格式
        ):
            # 调用函数 fn，传入两个参数，预期抛出运行时异常
            fn("foo", "bar")

    # 定义测试方法，检验字符串插值时使用了未知的格式说明符的情况
    def test_string_interpolation_with_unknown_format_specifier(self):
        # 使用 TorchScript 注解声明一个函数，接受一个参数并返回字符串
        @torch.jit.script
        def fn(arg1: str) -> str:
            # 使用 %a 格式说明符进行字符串格式化，引发 F501 错误时不报告
            return "%a in template" % arg1  # noqa: F501
        
        # 断言运行时错误，并在错误消息中突出显示“%a 是不支持的 TorchScript 格式字符串说明符”
        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            "The specifier %a is not supported in TorchScript format strings",
            '"%a in template" % arg1',  # 检查异常消息时期望的字符串格式
        ):
            # 调用函数 fn，传入一个参数，预期抛出运行时异常
            fn("foo")
```