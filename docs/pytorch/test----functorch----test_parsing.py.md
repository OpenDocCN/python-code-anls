# `.\pytorch\test\functorch\test_parsing.py`

```
# 引入必要的模块和函数
from typing import Any, Callable, Dict
from unittest import mock

# 从 functorch.einops._parsing 模块导入所需的函数和类
from functorch.einops._parsing import (
    _ellipsis,
    AnonymousAxis,
    parse_pattern,
    ParsedExpression,
    validate_rearrange_expressions,
)

# 定义用于模拟匿名轴对象相等性的函数
mock_anonymous_axis_eq: Callable[[AnonymousAxis, object], bool] = (
    lambda self, other: isinstance(other, AnonymousAxis) and self.value == other.value
)

# 定义测试类 TestAnonymousAxis，继承自 TestCase
class TestAnonymousAxis(TestCase):
    # 测试匿名轴对象的行为
    def test_anonymous_axes(self) -> None:
        # 创建两个具有相同值的匿名轴对象 a 和 b
        a, b = AnonymousAxis("2"), AnonymousAxis("2")
        # 断言 a 和 b 不相等
        self.assertNotEqual(a, b)

        # 使用 mock.patch.object 临时替换 AnonymousAxis 类中的 __eq__ 方法为 mock_anonymous_axis_eq 函数
        with mock.patch.object(AnonymousAxis, "__eq__", mock_anonymous_axis_eq):
            # 创建两个匿名轴对象 c 和 d，分别具有值 "2" 和 "3"
            c, d = AnonymousAxis("2"), AnonymousAxis("3")
            # 断言 a 与 c 相等，b 与 c 相等
            self.assertEqual(a, c)
            self.assertEqual(b, c)
            # 断言 a 与 d 不相等，b 与 d 不相等
            self.assertNotEqual(a, d)
            self.assertNotEqual(b, d)
            # 断言包含匿名轴对象的列表 [a, 2, b] 等于 [c, 2, c]
            self.assertListEqual([a, 2, b], [c, 2, c])


# 定义测试类 TestParsedExpression，继承自 TestCase
class TestParsedExpression(TestCase):
    # 测试检查元素轴名的函数 check_axis_name
    def test_elementary_axis_name(self) -> None:
        # 遍历不同的轴名进行测试
        for name in [
            "a",
            "b",
            "h",
            "dx",
            "h1",
            "zz",
            "i9123",
            "somelongname",
            "Alex",
            "camelCase",
            "u_n_d_e_r_score",
            "unreasonablyLongAxisName",
        ]:
            # 断言 ParsedExpression.check_axis_name(name) 返回 True
            self.assertTrue(ParsedExpression.check_axis_name(name))

        # 遍历不符合要求的轴名进行测试
        for name in [
            "",
            "2b",
            "12",
            "_startWithUnderscore",
            "endWithUnderscore_",
            "_",
            "...",
            _ellipsis,
        ]:
            # 断言 ParsedExpression.check_axis_name(name) 返回 False
            self.assertFalse(ParsedExpression.check_axis_name(name))
    # 定义一个测试方法，用于测试无效的表达式
    def test_invalid_expressions(self) -> None:
        # 测试句中包含两个省略号应该引发错误
        ParsedExpression("... a b c d")
        # 使用断言检查是否抛出 ValueError 异常
        with self.assertRaises(ValueError):
            ParsedExpression("... a b c d ...")
        with self.assertRaises(ValueError):
            ParsedExpression("... a b c (d ...)")
        with self.assertRaises(ValueError):
            ParsedExpression("(... a) b c (d ...)")

        # 测试双重/缺失/括号错误的情况
        ParsedExpression("(a) b c (d ...)")
        with self.assertRaises(ValueError):
            ParsedExpression("(a)) b c (d ...)")
        with self.assertRaises(ValueError):
            ParsedExpression("(a b c (d ...)")
        with self.assertRaises(ValueError):
            ParsedExpression("(a) (()) b c (d ...)")
        with self.assertRaises(ValueError):
            ParsedExpression("(a) ((b c) (d ...))")

        # 测试无效的标识符情况
        ParsedExpression("camelCase under_scored cApiTaLs \u00DF ...")
        with self.assertRaises(ValueError):
            ParsedExpression("1a")
        with self.assertRaises(ValueError):
            ParsedExpression("_pre")
        with self.assertRaises(ValueError):
            ParsedExpression("...pre")
        with self.assertRaises(ValueError):
            ParsedExpression("pre...")
        
    # 使用 mock.patch.object 方法来模拟 AnonymousAxis 类的 __eq__ 方法，使用 mock_anonymous_axis_eq 作为替代
    @mock.patch.object(AnonymousAxis, "__eq__", mock_anonymous_axis_eq)
class TestParsingUtils(TestCase):
    # 定义测试类 TestParsingUtils，继承自 TestCase 类

    def test_parse_pattern_number_of_arrows(self) -> None:
        # 定义测试方法 test_parse_pattern_number_of_arrows，返回 None
        axes_lengths: Dict[str, int] = {}
        # 创建空字典 axes_lengths，用于存储轴长度信息

        too_many_arrows_pattern = "a -> b -> c -> d"
        # 定义过多箭头的模式字符串
        with self.assertRaises(ValueError):
            # 使用断言验证抛出 ValueError 异常
            parse_pattern(too_many_arrows_pattern, axes_lengths)

        too_few_arrows_pattern = "a"
        # 定义过少箭头的模式字符串
        with self.assertRaises(ValueError):
            # 使用断言验证抛出 ValueError 异常
            parse_pattern(too_few_arrows_pattern, axes_lengths)

        just_right_arrows = "a -> a"
        # 定义正常箭头数量的模式字符串
        parse_pattern(just_right_arrows, axes_lengths)
        # 解析正常箭头数量的模式

    def test_ellipsis_invalid_identifier(self) -> None:
        # 定义测试方法 test_ellipsis_invalid_identifier，返回 None
        axes_lengths: Dict[str, int] = {"a": 1, _ellipsis: 2}
        # 创建包含轴长度信息和特殊标识符 _ellipsis 的字典
        pattern = f"a {_ellipsis} -> {_ellipsis} a"
        # 使用特殊标识符 _ellipsis 构建模式字符串
        with self.assertRaises(ValueError):
            # 使用断言验证抛出 ValueError 异常
            parse_pattern(pattern, axes_lengths)

    def test_ellipsis_matching(self) -> None:
        # 定义测试方法 test_ellipsis_matching，返回 None
        axes_lengths: Dict[str, int] = {}

        pattern = "a -> a ..."
        # 定义带省略符号的模式字符串
        with self.assertRaises(ValueError):
            # 使用断言验证抛出 ValueError 异常
            parse_pattern(pattern, axes_lengths)

        # raising an error on this pattern is handled by the rearrange expression validation
        pattern = "a ... -> a"
        # 定义带省略符号的模式字符串
        parse_pattern(pattern, axes_lengths)

        pattern = "a ... -> ... a"
        # 定义带双向省略符号的模式字符串
        parse_pattern(pattern, axes_lengths)

    def test_left_parenthesized_ellipsis(self) -> None:
        # 定义测试方法 test_left_parenthesized_ellipsis，返回 None
        axes_lengths: Dict[str, int] = {}

        pattern = "(...) -> ..."
        # 定义带括号省略符号的模式字符串
        with self.assertRaises(ValueError):
            # 使用断言验证抛出 ValueError 异常
            parse_pattern(pattern, axes_lengths)


class MaliciousRepr:
    # 定义恶意的 __repr__ 方法，用于测试

    def __repr__(self) -> str:
        # 定义返回字符串 "print('hello world!')" 的方法
        return "print('hello world!')"


class TestValidateRearrangeExpressions(TestCase):
    # 定义测试类 TestValidateRearrangeExpressions，继承自 TestCase 类

    def test_validate_axes_lengths_are_integers(self) -> None:
        # 定义测试方法 test_validate_axes_lengths_are_integers，返回 None
        axes_lengths: Dict[str, Any] = {"a": 1, "b": 2, "c": 3}
        # 创建包含整数类型轴长度信息的字典
        pattern = "a b c -> c b a"
        # 定义模式字符串
        left, right = parse_pattern(pattern, axes_lengths)
        # 解析模式字符串，获取左右表达式

        validate_rearrange_expressions(left, right, axes_lengths)
        # 验证重排表达式的有效性

        axes_lengths = {"a": 1, "b": 2, "c": MaliciousRepr()}
        # 创建包含恶意 __repr__ 方法的轴长度信息的字典
        left, right = parse_pattern(pattern, axes_lengths)
        # 解析模式字符串，获取左右表达式
        with self.assertRaises(TypeError):
            # 使用断言验证抛出 TypeError 异常
            validate_rearrange_expressions(left, right, axes_lengths)

    def test_non_unitary_anonymous_axes_raises_error(self) -> None:
        # 定义测试方法 test_non_unitary_anonymous_axes_raises_error，返回 None
        axes_lengths: Dict[str, int] = {}

        left_non_unitary_axis = "a 2 -> 1 1 a"
        # 定义左侧包含非单位匿名轴的模式字符串
        left, right = parse_pattern(left_non_unitary_axis, axes_lengths)
        # 解析模式字符串，获取左右表达式
        with self.assertRaises(ValueError):
            # 使用断言验证抛出 ValueError 异常
            validate_rearrange_expressions(left, right, axes_lengths)

        right_non_unitary_axis = "1 1 a -> a 2"
        # 定义右侧包含非单位匿名轴的模式字符串
        left, right = parse_pattern(right_non_unitary_axis, axes_lengths)
        # 解析模式字符串，获取左右表达式
        with self.assertRaises(ValueError):
            # 使用断言验证抛出 ValueError 异常
            validate_rearrange_expressions(left, right, axes_lengths)
    # 定义一个测试方法，用于测试标识符不匹配的情况
    def test_identifier_mismatch(self) -> None:
        # 初始化一个空字典 axes_lengths，用于存储轴的长度信息
        axes_lengths: Dict[str, int] = {}
    
        # 定义标识符不匹配的模式字符串
        mismatched_identifiers = "a -> a b"
        # 调用 parse_pattern 函数解析模式字符串，获取左右两边的表达式
        left, right = parse_pattern(mismatched_identifiers, axes_lengths)
        # 断言抛出 ValueError 异常
        with self.assertRaises(ValueError):
            # 调用 validate_rearrange_expressions 函数验证重排表达式的有效性
            validate_rearrange_expressions(left, right, axes_lengths)
    
        # 定义另一个标识符不匹配的模式字符串
        mismatched_identifiers = "a b -> a"
        # 再次调用 parse_pattern 函数解析模式字符串，获取左右两边的表达式
        left, right = parse_pattern(mismatched_identifiers, axes_lengths)
        # 断言抛出 ValueError 异常
        with self.assertRaises(ValueError):
            # 再次调用 validate_rearrange_expressions 函数验证重排表达式的有效性
            validate_rearrange_expressions(left, right, axes_lengths)
    
    # 定义一个测试方法，用于测试意外的轴长度信息
    def test_unexpected_axes_lengths(self) -> None:
        # 初始化一个包含预设轴长度信息的字典 axes_lengths
        axes_lengths: Dict[str, int] = {"c": 2}
    
        # 定义一个包含意外轴长度信息的重排模式字符串
        pattern = "a b -> b a"
        # 调用 parse_pattern 函数解析模式字符串，获取左右两边的表达式
        left, right = parse_pattern(pattern, axes_lengths)
        # 断言抛出 ValueError 异常
        with self.assertRaises(ValueError):
            # 调用 validate_rearrange_expressions 函数验证重排表达式的有效性
            validate_rearrange_expressions(left, right, axes_lengths)
# 如果当前脚本作为主程序运行（而不是被导入为模块），则执行 run_tests() 函数
if __name__ == "__main__":
    run_tests()
```