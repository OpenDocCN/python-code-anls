# `.\DB-GPT-src\dbgpt\serve\agent\agents\__init__.py`

```py
# 定义一个名为 count_digits 的函数，接受一个字符串参数 s
def count_digits(s):
    # 使用字典推导式创建一个空字典 counts
    counts = {digit: 0 for digit in '0123456789'}
    
    # 遍历字符串 s 中的每一个字符
    for char in s:
        # 如果字符是一个数字字符，则增加相应数字键的计数
        if char.isdigit():
            counts[char] += 1
    
    # 返回包含每个数字字符计数的字典 counts
    return counts
```