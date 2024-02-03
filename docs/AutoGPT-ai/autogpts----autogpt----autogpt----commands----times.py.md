# `.\AutoGPT\autogpts\autogpt\autogpt\commands\times.py`

```py
# 导入 datetime 模块中的 datetime 类
from datetime import datetime

# 定义一个函数，返回当前日期和时间的字符串表示
def get_datetime() -> str:
    """Return the current date and time

    Returns:
        str: The current date and time
    """
    # 返回当前日期和时间的字符串表示，格式为年-月-日 时:分:秒
    return "Current date and time: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```