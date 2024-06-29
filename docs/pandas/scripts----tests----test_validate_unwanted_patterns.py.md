# `D:\src\scipysrc\pandas\scripts\tests\test_validate_unwanted_patterns.py`

```
# 导入所需的模块
import io  # 导入io模块用于处理输入输出操作
import pytest  # 导入pytest模块用于单元测试框架

# 从scripts模块中导入validate_unwanted_patterns函数
from scripts import validate_unwanted_patterns

# 定义一个测试类，用于测试字符串中不正确放置空白字符的情况
class TestStringsWithWrongPlacedWhitespace:

    # 使用pytest的参数化装饰器，对多组数据进行测试
    @pytest.mark.parametrize(
        "data",
        [
            (
                """
    msg = (
        "foo\n"
        " bar"
    )
    """
            ),
            (
                """
    msg = (
        "foo"
        "  bar"
        "baz"
    )
    """
            ),
            (
                """
    msg = (
        f"foo"
        "  bar"
    )
    """
            ),
            (
                """
    msg = (
        "foo"
        f"  bar"
    )
    """
            ),
            (
                """
    msg = (
        "foo"
        rf"  bar"
    )
    """
            ),
        ],
    )
    # 定义测试方法，测试字符串中不正确放置空白字符的情况
    def test_strings_with_wrong_placed_whitespace(self, data) -> None:
        # 创建一个内存中的文本流对象，用于模拟数据输入
        fd = io.StringIO(data.strip())
        # 调用validate_unwanted_patterns模块中的strings_with_wrong_placed_whitespace函数
        result = list(validate_unwanted_patterns.strings_with_wrong_placed_whitespace(fd))
        # 断言结果为空列表，表示未发现不正确放置空白字符的情况
        assert result == []

    # 使用pytest的参数化装饰器，对多组数据进行测试
    @pytest.mark.parametrize(
        "data, expected",
        [
            (
                (
                    """
    msg = (
        "foo"
        " bar"
    )
    """
                ),
                [
                    (
                        3,
                        (
                            "String has a space at the beginning instead "
                            "of the end of the previous string."
                        ),
                    )
                ],
            ),
            (
                (
                    """
    msg = (
        f"foo"
        " bar"
    )
    """
                ),
                [
                    (
                        3,
                        (
                            "String has a space at the beginning instead "
                            "of the end of the previous string."
                        ),
                    )
                ],
            ),
            (
                (
                    """
    msg = (
        "foo"
        f" bar"
    )
    """
                ),
                [
                    (
                        3,
                        (
                            "String has a space at the beginning instead "
                            "of the end of the previous string."
                        ),
                    )
                ],
            ),
            (
                (
                    """
    msg = (
        f"foo"
        f" bar"
    )
    """
                ),
                [
                    (
                        3,
                        (
                            "String has a space at the beginning instead "
                            "of the end of the previous string."
                        ),
                    )
                ],
            ),
            (
                (
                    """
    msg = (
        "foo"
        rf" bar"
        " baz"
    )
    """
                ),
                # 预期结果：包含一个元组，表示第3行字符串前面放置了空格而不是在前一个字符串末尾
                [
                    (
                        3,
                        (
                            "String has a space at the beginning instead "
                            "of the end of the previous string."
                        ),
                    )
                ],
            ),
        ],
    )
    # 定义测试方法，测试字符串中不正确放置空白字符的情况，并验证预期结果
    def test_strings_with_wrong_placed_whitespace_expected(self, data, expected) -> None:
        # 创建一个内存中的文本流对象，用于模拟数据输入
        fd = io.StringIO(data.strip())
        # 调用validate_unwanted_patterns模块中的strings_with_wrong_placed_whitespace函数
        result = list(validate_unwanted_patterns.strings_with_wrong_placed_whitespace(fd))
        # 断言结果与预期结果一致
        assert result == expected
    """
    # 定义测试函数，检查带有错误位置空白的字符串是否引发异常
    def test_strings_with_wrong_placed_whitespace_raises(self, data, expected) -> None:
        # 从给定的数据创建一个字符串IO对象
        fd = io.StringIO(data.strip())
        # 调用被测试函数，验证不希望出现的模式在字符串中的位置
        result = list(
            validate_unwanted_patterns.strings_with_wrong_placed_whitespace(fd)
        )
        # 断言测试结果是否与期望结果一致
        assert result == expected
    ```
class TestNoDefaultUsedNotOnlyForTyping:
    @pytest.mark.parametrize(
        "data",
        [
            (
                """
def f(
    a: int | NoDefault,
    b: float | lib.NoDefault = 0.1,
    c: pandas._libs.lib.NoDefault = lib.no_default,
) -> lib.NoDefault | None:
    pass
"""
            ),
            (
                """
# var = lib.NoDefault
# the above is incorrect
a: NoDefault | int
b: lib.NoDefault = lib.no_default
"""
            ),
        ],
    )
    def test_nodefault_used_not_only_for_typing(self, data) -> None:
        # 创建一个内存中的文本流，用给定的data初始化
        fd = io.StringIO(data.strip())
        # 调用验证函数，检查不良模式，返回结果列表
        result = list(validate_unwanted_patterns.nodefault_used_not_only_for_typing(fd))
        # 断言结果为空列表
        assert result == []

    @pytest.mark.parametrize(
        "data, expected",
        [
            (
                (
                    """
def f(
    a = lib.NoDefault,
    b: Any
        = pandas._libs.lib.NoDefault,
):
    pass
"""
                ),
                [
                    (2, "NoDefault is used not only for typing"),
                    (4, "NoDefault is used not only for typing"),
                ],
            ),
            (
                (
                    """
a: Any = lib.NoDefault
if a is NoDefault:
    pass
"""
                ),
                [
                    (1, "NoDefault is used not only for typing"),
                    (2, "NoDefault is used not only for typing"),
                ],
            ),
        ],
    )
    def test_nodefault_used_not_only_for_typing_raises(self, data, expected) -> None:
        # 创建一个内存中的文本流，用给定的data初始化
        fd = io.StringIO(data.strip())
        # 调用验证函数，检查不良模式，返回结果列表
        result = list(validate_unwanted_patterns.nodefault_used_not_only_for_typing(fd))
        # 断言结果与预期列表相同
        assert result == expected
```