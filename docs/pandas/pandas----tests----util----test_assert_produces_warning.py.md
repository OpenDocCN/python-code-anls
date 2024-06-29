# `D:\src\scipysrc\pandas\pandas\tests\util\test_assert_produces_warning.py`

```
"""
Test module for testing ``pandas._testing.assert_produces_warning``.
"""

# 导入所需的模块和库
import warnings  # 导入警告模块

import pytest  # 导入 pytest 测试框架

from pandas.errors import (  # 导入 pandas 错误模块中的特定异常类
    DtypeWarning,
    PerformanceWarning,
)

import pandas._testing as tm  # 导入 pandas 测试模块


@pytest.fixture(
    params=[  # 参数化装饰器，用于多次测试不同的警告对
        (RuntimeWarning, UserWarning),
        (UserWarning, FutureWarning),
        (FutureWarning, RuntimeWarning),
        (DeprecationWarning, PerformanceWarning),
        (PerformanceWarning, FutureWarning),
        (DtypeWarning, DeprecationWarning),
        (ResourceWarning, DeprecationWarning),
        (FutureWarning, DeprecationWarning),
    ],
    ids=lambda x: type(x).__name__,  # 根据参数类型名设置每个测试用例的 ID
)
def pair_different_warnings(request):
    """
    Return pair or different warnings.

    Useful for testing how several different warnings are handled
    in tm.assert_produces_warning.
    """
    return request.param  # 返回参数化的警告对


def f():
    warnings.warn("f1", FutureWarning)  # 发出 FutureWarning 警告
    warnings.warn("f2", RuntimeWarning)  # 发出 RuntimeWarning 警告


def test_assert_produces_warning_honors_filter():
    # 测试函数：验证 assert_produces_warning 是否正确处理警告过滤器
    # 默认情况下会触发 AssertionError
    msg = r"Caused unexpected warning\(s\)"
    with pytest.raises(AssertionError, match=msg):
        with tm.assert_produces_warning(RuntimeWarning):
            f()

    # 使用 raise_on_extra_warnings=False 可以忽略额外的警告
    with tm.assert_produces_warning(RuntimeWarning, raise_on_extra_warnings=False):
        f()


@pytest.mark.parametrize(
    "category",
    [
        RuntimeWarning,
        ResourceWarning,
        UserWarning,
        FutureWarning,
        DeprecationWarning,
        PerformanceWarning,
        DtypeWarning,
    ],
)
@pytest.mark.parametrize(
    "message, match",
    [
        ("", None),
        ("", ""),
        ("Warning message", r".*"),
        ("Warning message", "War"),
        ("Warning message", r"[Ww]arning"),
        ("Warning message", "age"),
        ("Warning message", r"age$"),
        ("Message 12-234 with numbers", r"\d{2}-\d{3}"),
        ("Message 12-234 with numbers", r"^Mes.*\d{2}-\d{3}"),
        ("Message 12-234 with numbers", r"\d{2}-\d{3}\s\S+"),
        ("Message, which we do not match", None),
    ],
)
def test_catch_warning_category_and_match(category, message, match):
    # 测试函数：验证 assert_produces_warning 能够捕获特定类别和匹配的警告
    with tm.assert_produces_warning(category, match=match):
        warnings.warn(message, category)


def test_fail_to_match_runtime_warning():
    # 测试函数：验证无法匹配 RuntimeWarning 警告
    category = RuntimeWarning
    match = "Did not see this warning"
    unmatched = (
        r"Did not see warning 'RuntimeWarning' matching 'Did not see this warning'. "
        r"The emitted warning messages are "
        r"\[RuntimeWarning\('This is not a match.'\), "
        r"RuntimeWarning\('Another unmatched warning.'\)\]"
    )
    with pytest.raises(AssertionError, match=unmatched):
        with tm.assert_produces_warning(category, match=match):
            warnings.warn("This is not a match.", category)
            warnings.warn("Another unmatched warning.", category)


def test_fail_to_match_future_warning():
    # 测试函数：验证无法匹配 FutureWarning 警告
    category = FutureWarning
    match = "Warning"
    unmatched = (
        r"Did not see warning 'FutureWarning' matching 'Warning'. "
        r"The emitted warning messages are "
        r"\[FutureWarning\('This is not a match.'\), "
        r"FutureWarning\('Another unmatched warning.'\)\]"
    )
    # 定义未匹配到的预期警告消息，使用原始字符串（r 开头），包含了预期的警告信息格式
    with pytest.raises(AssertionError, match=unmatched):
        # 使用 pytest 模块的 raises 函数检查是否抛出特定类型的异常，并匹配异常信息
        with tm.assert_produces_warning(category, match=match):
            # 在测试环境中，使用 assert_produces_warning 函数来确保特定类别和匹配模式的警告被触发
            warnings.warn("This is not a match.", category)
            # 发出一个不匹配的警告消息
            warnings.warn("Another unmatched warning.", category)
            # 再次发出另一个不匹配的警告消息
# 测试函数：验证无法匹配资源警告
def test_fail_to_match_resource_warning():
    # 设定期望的警告类别为 ResourceWarning
    category = ResourceWarning
    # 设定匹配模式为数字序列
    match = r"\d+"
    # 设定未匹配到期望警告时的错误信息
    unmatched = (
        r"Did not see warning 'ResourceWarning' matching '\\d\+'. "
        r"The emitted warning messages are "
        r"\[ResourceWarning\('This is not a match.'\), "
        r"ResourceWarning\('Another unmatched warning.'\)\]"
    )
    # 使用 pytest 检查是否抛出 AssertionError，并匹配未匹配到的警告信息
    with pytest.raises(AssertionError, match=unmatched):
        # 使用 assert_produces_warning 上下文检查是否发出特定类别和匹配模式的警告
        with tm.assert_produces_warning(category, match=match):
            # 发出一个不匹配的 ResourceWarning
            warnings.warn("This is not a match.", category)
            # 再发出一个不匹配的 ResourceWarning
            warnings.warn("Another unmatched warning.", category)


# 测试函数：验证无法捕获实际警告
def test_fail_to_catch_actual_warning(pair_different_warnings):
    # 分别设定期望的和实际的警告类别
    expected_category, actual_category = pair_different_warnings
    # 设定匹配模式为包含特定错误信息
    match = "Did not see expected warning of class"
    # 使用 pytest 检查是否抛出 AssertionError，并匹配未捕获到预期警告的信息
    with pytest.raises(AssertionError, match=match):
        # 使用 assert_produces_warning 上下文检查是否发出特定类别的警告
        with tm.assert_produces_warning(expected_category):
            # 发出一个预期外的警告
            warnings.warn("warning message", actual_category)


# 测试函数：验证忽略额外的警告
def test_ignore_extra_warning(pair_different_warnings):
    # 分别设定期望的和额外的警告类别
    expected_category, extra_category = pair_different_warnings
    # 使用 assert_produces_warning 上下文检查是否发出特定类别的警告，忽略额外的警告
    with tm.assert_produces_warning(expected_category, raise_on_extra_warnings=False):
        # 发出一个预期的警告
        warnings.warn("Expected warning", expected_category)
        # 发出一个未预期的警告
        warnings.warn("Unexpected warning OK", extra_category)


# 测试函数：验证对额外的警告抛出异常
def test_raise_on_extra_warning(pair_different_warnings):
    # 分别设定期望的和额外的警告类别
    expected_category, extra_category = pair_different_warnings
    # 设定匹配模式为包含异常引起的额外警告信息
    match = r"Caused unexpected warning\(s\)"
    # 使用 pytest 检查是否抛出 AssertionError，并匹配引起额外警告的信息
    with pytest.raises(AssertionError, match=match):
        # 使用 assert_produces_warning 上下文检查是否发出特定类别的警告
        with tm.assert_produces_warning(expected_category):
            # 发出一个预期的警告
            warnings.warn("Expected warning", expected_category)
            # 发出一个不预期的警告
            warnings.warn("Unexpected warning NOT OK", extra_category)


# 测试函数：验证相同类别的不同消息中的第一个匹配
def test_same_category_different_messages_first_match():
    # 设定警告类别为 UserWarning
    category = UserWarning
    # 使用 assert_produces_warning 上下文检查是否发出特定类别和匹配模式的警告
    with tm.assert_produces_warning(category, match=r"^Match this"):
        # 发出一个匹配的警告消息
        warnings.warn("Match this", category)
        # 发出两个不匹配的警告消息
        warnings.warn("Do not match that", category)
        warnings.warn("Do not match that either", category)


# 测试函数：验证相同类别的不同消息中的最后一个匹配
def test_same_category_different_messages_last_match():
    # 设定警告类别为 DeprecationWarning
    category = DeprecationWarning
    # 使用 assert_produces_warning 上下文检查是否发出特定类别和匹配模式的警告
    with tm.assert_produces_warning(category, match=r"^Match this"):
        # 发出两个不匹配的警告消息
        warnings.warn("Do not match that", category)
        warnings.warn("Do not match that either", category)
        # 发出一个匹配的警告消息
        warnings.warn("Match this", category)


# 测试函数：验证匹配多个警告
def test_match_multiple_warnings():
    # 设定警告类别为 FutureWarning 和 UserWarning
    category = (FutureWarning, UserWarning)
    # 使用 assert_produces_warning 上下文检查是否发出特定类别和匹配模式的警告
    with tm.assert_produces_warning(category, match=r"^Match this"):
        # 分别发出两个匹配的警告消息
        warnings.warn("Match this", FutureWarning)
        warnings.warn("Match this too", UserWarning)


# 测试函数：验证必须匹配多个警告
def test_must_match_multiple_warnings():
    # 设定警告类别为 FutureWarning 和 UserWarning
    category = (FutureWarning, UserWarning)
    # 设定错误信息为未捕获到预期警告类 'UserWarning'
    msg = "Did not see expected warning of class 'UserWarning'"
    # 使用 pytest 来测试断言错误是否引发，并检查警告信息是否与给定的消息模式匹配
    with pytest.raises(AssertionError, match=msg):
        # 在测试中，确保产生一个特定类别的警告，并且警告消息以指定的正则表达式模式开头
        with tm.assert_produces_warning(category, match=r"^Match this"):
            # 在此处发出一个警告，消息为 "Match this"，类型为 FutureWarning
            warnings.warn("Match this", FutureWarning)
def test_must_match_multiple_warnings_messages():
    # 设置警告的类别为 FutureWarning 和 UserWarning
    category = (FutureWarning, UserWarning)
    # 定义匹配的消息正则表达式
    msg = r"The emitted warning messages are \[UserWarning\('Not this'\)\]"
    # 断言期望捕获到 Assertion 错误，并且匹配特定的消息内容
    with pytest.raises(AssertionError, match=msg):
        # 断言在上下文中会产生警告，同时满足特定的条件
        with tm.assert_produces_warning(category, match=r"^Match this"):
            # 发出一个 FutureWarning 类型的警告
            warnings.warn("Match this", FutureWarning)
            # 发出一个 UserWarning 类型的警告
            warnings.warn("Not this", UserWarning)


def test_allow_partial_match_for_multiple_warnings():
    # 设置警告的类别为 FutureWarning 和 UserWarning
    category = (FutureWarning, UserWarning)
    # 断言在上下文中会产生警告，允许部分匹配，不要求捕获所有的警告
    with tm.assert_produces_warning(
        category, match=r"^Match this", must_find_all_warnings=False
    ):
        # 发出一个 FutureWarning 类型的警告
        warnings.warn("Match this", FutureWarning)


def test_allow_partial_match_for_multiple_warnings_messages():
    # 设置警告的类别为 FutureWarning 和 UserWarning
    category = (FutureWarning, UserWarning)
    # 断言在上下文中会产生警告，允许部分匹配，不要求捕获所有的警告
    with tm.assert_produces_warning(
        category, match=r"^Match this", must_find_all_warnings=False
    ):
        # 发出一个 FutureWarning 类型的警告
        warnings.warn("Match this", FutureWarning)
        # 发出一个 UserWarning 类型的警告
        warnings.warn("Not this", UserWarning)


def test_right_category_wrong_match_raises(pair_different_warnings):
    # 从参数化的测试获取目标和其他不同的警告类别
    target_category, other_category = pair_different_warnings
    # 断言期望捕获到 Assertion 错误，并且匹配特定的消息内容
    with pytest.raises(AssertionError, match="Did not see warning.*matching"):
        # 断言在上下文中会产生特定类别的警告，并且匹配特定的消息内容
        with tm.assert_produces_warning(target_category, match=r"^Match this"):
            # 发出一个目标类别不匹配的警告
            warnings.warn("Do not match it", target_category)
            # 发出一个符合目标类别和消息的警告
            warnings.warn("Match this", other_category)


@pytest.mark.parametrize("false_or_none", [False, None])
class TestFalseOrNoneExpectedWarning:
    def test_raise_on_warning(self, false_or_none):
        # 定义期望捕获到 Assertion 错误的消息内容
        msg = r"Caused unexpected warning\(s\)"
        # 断言期望捕获到 Assertion 错误，并且匹配特定的消息内容
        with pytest.raises(AssertionError, match=msg):
            # 断言在上下文中会产生警告，但是要求与期望的情况不符
            with tm.assert_produces_warning(false_or_none):
                # 调用一个产生警告的函数或代码段
                f()

    def test_no_raise_without_warning(self, false_or_none):
        # 断言在上下文中不会产生任何警告
        with tm.assert_produces_warning(false_or_none):
            # 空操作，不产生警告
            pass

    def test_no_raise_with_false_raise_on_extra(self, false_or_none):
        # 断言在上下文中不会产生任何警告，即使有额外的警告
        with tm.assert_produces_warning(false_or_none, raise_on_extra_warnings=False):
            # 调用一个产生警告的函数或代码段
            f()


def test_raises_during_exception():
    # 定义期望捕获到 Assertion 错误的消息内容
    msg = "Did not see expected warning of class 'UserWarning'"
    # 断言期望捕获到 Assertion 错误，并且匹配特定的消息内容
    with pytest.raises(AssertionError, match=msg):
        # 断言在上下文中会产生特定类别的警告，并且匹配特定的消息内容
        with tm.assert_produces_warning(UserWarning):
            # 触发一个 ValueError 异常
            raise ValueError

    with pytest.raises(AssertionError, match=msg):
        # 断言在上下文中会产生特定类别的警告，并且匹配特定的消息内容
        with tm.assert_produces_warning(UserWarning):
            # 发出一个 FutureWarning 类型的警告
            warnings.warn("FutureWarning", FutureWarning)
            # 触发一个 IndexError 异常
            raise IndexError

    # 定义期望捕获到 Assertion 错误的消息内容
    msg = "Caused unexpected warning"
    with pytest.raises(AssertionError, match=msg):
        # 断言在上下文中不会产生任何警告
        with tm.assert_produces_warning(None):
            # 发出一个 FutureWarning 类型的警告
            warnings.warn("FutureWarning", FutureWarning)
            # 触发一个 SystemError 异常
            raise SystemError
    # 使用 pytest 模块来测试抛出的异常是否符合预期，并匹配指定的错误信息 "Error"
    with pytest.raises(SyntaxError, match="Error"):
        # 在测试语境中，使用 assert_produces_warning 方法确保没有产生警告
        with tm.assert_produces_warning(None):
            # 抛出 SyntaxError 异常，消息为 "Error"
            raise SyntaxError("Error")
    
    # 使用 pytest 模块来测试抛出的异常是否符合预期，并匹配指定的错误信息 "Error"
    with pytest.raises(ValueError, match="Error"):
        # 在测试语境中，使用 assert_produces_warning 方法确保产生 FutureWarning 警告，并匹配 "FutureWarning"
        with tm.assert_produces_warning(FutureWarning, match="FutureWarning"):
            # 发出一个 FutureWarning 类型的警告
            warnings.warn("FutureWarning", FutureWarning)
            # 抛出 ValueError 异常，消息为 "Error"
            raise ValueError("Error")
```