# `.\DB-GPT-src\dbgpt\util\time_utils.py`

```py
import contextlib
from datetime import datetime

# 定义上下文管理器，用于在单元测试中模拟 datetime.now() 的返回值
@contextlib.contextmanager
def mock_now(dt_value):  # type: ignore
    """Context manager for mocking out datetime.now() in unit tests.

    Adapted from langchain.utils.mock_now.

    Example:
    with mock_now(datetime.datetime(2011, 2, 3, 10, 11)):
        assert datetime.datetime.now() == datetime.datetime(2011, 2, 3, 10, 11)
    """

    # 定义一个 MockDateTime 类，继承自 datetime.datetime
    class MockDateTime(datetime.datetime):
        @classmethod
        def now(cls):  # type: ignore
            # 创建并返回 dt_value 的副本
            return datetime.datetime(
                dt_value.year,
                dt_value.month,
                dt_value.day,
                dt_value.hour,
                dt_value.minute,
                dt_value.second,
                dt_value.microsecond,
                dt_value.tzinfo,
            )

    # 备份真实的 datetime.datetime 类
    real_datetime = datetime.datetime
    # 替换当前作用域内的 datetime.datetime 为 MockDateTime
    datetime.datetime = MockDateTime
    try:
        # 进入上下文管理器的代码块，执行 yield 语句
        yield datetime.datetime
    finally:
        # 无论如何都要恢复原始的 datetime.datetime 类
        datetime.datetime = real_datetime
```