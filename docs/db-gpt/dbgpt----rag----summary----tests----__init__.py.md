# `.\DB-GPT-src\dbgpt\rag\summary\tests\__init__.py`

```py
# 定义一个名为 count_upper_lower 的函数，接受一个字符串参数 s
def count_upper_lower(s):
    # 初始化大写字母计数器为 0
    upper_count = 0
    # 初始化小写字母计数器为 0
    lower_count = 0
    
    # 遍历字符串 s 中的每一个字符
    for char in s:
        # 如果当前字符是大写字母
        if char.isupper():
            # 将大写字母计数器加 1
            upper_count += 1
        # 如果当前字符是小写字母
        elif char.islower():
            # 将小写字母计数器加 1
            lower_count += 1
    
    # 返回大写字母和小写字母的计数结果，以元组形式返回
    return (upper_count, lower_count)
```