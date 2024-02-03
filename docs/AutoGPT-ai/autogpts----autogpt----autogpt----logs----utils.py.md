# `.\AutoGPT\autogpts\autogpt\autogpt\logs\utils.py`

```py
# 导入 re 模块，用于处理正则表达式
import re

# 定义函数 remove_color_codes，用于移除字符串中的颜色代码
def remove_color_codes(s: str) -> str:
    # 使用正则表达式替换字符串中的颜色代码
    return re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", s)

# 定义函数 fmt_kwargs，用于格式化关键字参数字典为字符串
def fmt_kwargs(kwargs: dict) -> str:
    # 使用列表推导式将关键字参数字典中的键值对格式化为字符串，并用逗号连接
    return ", ".join(f"{n}={repr(v)}" for n, v in kwargs.items())
```