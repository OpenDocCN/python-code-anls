# `.\DB-GPT-src\tests\unit_tests\llms\wenxin.py`

```py
# 导入Python的正则表达式模块
import re

# 定义函数match_pattern，接收一个字符串参数pattern
def match_pattern(pattern):
    # 使用正则表达式模块的compile函数编译传入的模式字符串，生成正则表达式对象
    regex = re.compile(pattern)
    # 返回编译后的正则表达式对象
    return regex

# 调用match_pattern函数，传入模式字符串'\d+'，返回相应的正则表达式对象
pattern_obj = match_pattern(r'\d+')
```