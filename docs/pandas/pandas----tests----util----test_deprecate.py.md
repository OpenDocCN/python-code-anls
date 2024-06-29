# `D:\src\scipysrc\pandas\pandas\tests\util\test_deprecate.py`

```
# 从 textwrap 模块中导入 dedent 函数
from textwrap import dedent

# 导入 pytest 测试框架
import pytest

# 从 pandas.util._decorators 中导入 deprecate 装饰器
from pandas.util._decorators import deprecate

# 导入 pandas 测试工具模块作为 tm
import pandas._testing as tm


# 定义一个新函数 new_func，无参数
def new_func():
    """
    This is the summary. The deprecate directive goes next.

    This is the extended summary. The deprecate directive goes before this.
    """
    return "new_func called"


# 定义一个新函数 new_func_no_docstring，无参数
def new_func_no_docstring():
    return "new_func_no_docstring called"


# 定义一个新函数 new_func_wrong_docstring，无参数，但文档字符串格式有误
def new_func_wrong_docstring():
    """Summary should be in the next line."""
    return "new_func_wrong_docstring called"


# 定义一个新函数 new_func_with_deprecation，无参数
def new_func_with_deprecation():
    """
    This is the summary. The deprecate directive goes next.

    .. deprecated:: 1.0
        Use new_func instead.

    This is the extended summary. The deprecate directive goes before this.
    """


# 定义测试函数 test_deprecate_ok，测试 deprecate 装饰器正常工作
def test_deprecate_ok():
    # 使用 deprecate 装饰器将 new_func 标记为过时，指定消息和版本号
    depr_func = deprecate("depr_func", new_func, "1.0", msg="Use new_func instead.")

    # 使用 pytest 的 assert_produces_warning 上下文来确保将来会产生警告
    with tm.assert_produces_warning(FutureWarning):
        result = depr_func()

    # 断言调用结果与预期结果一致
    assert result == "new_func called"
    # 断言 deprec_func 的文档字符串与 new_func_with_deprecation 的去除缩进后的文档字符串一致
    assert depr_func.__doc__ == dedent(new_func_with_deprecation.__doc__)


# 定义测试函数 test_deprecate_no_docstring，测试没有文档字符串的函数使用 deprecate
def test_deprecate_no_docstring():
    # 使用 deprecate 装饰器将 new_func_no_docstring 标记为过时，指定消息和版本号
    depr_func = deprecate(
        "depr_func", new_func_no_docstring, "1.0", msg="Use new_func instead."
    )
    # 使用 pytest 的 assert_produces_warning 上下文来确保将来会产生警告
    with tm.assert_produces_warning(FutureWarning):
        result = depr_func()
    # 断言调用结果与预期结果一致
    assert result == "new_func_no_docstring called"


# 定义测试函数 test_deprecate_wrong_docstring，测试文档字符串格式错误的函数使用 deprecate
def test_deprecate_wrong_docstring():
    # 准备断言消息
    msg = "deprecate needs a correctly formatted docstring"
    # 使用 pytest 的 raises 断言来捕获预期的 AssertionError 异常，并检查消息匹配
    with pytest.raises(AssertionError, match=msg):
        # 使用 deprecate 装饰器将 new_func_wrong_docstring 标记为过时，指定消息和版本号
        deprecate(
            "depr_func", new_func_wrong_docstring, "1.0", msg="Use new_func instead."
        )
```