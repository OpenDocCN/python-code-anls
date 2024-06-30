# `D:\src\scipysrc\sympy\sympy\physics\optics\tests\__init__.py`

```
# 定义一个名为 count_chars 的函数，接受一个字符串参数 s
def count_chars(s):
    # 创建一个空字典 counts，用于存储字符计数
    counts = {}
    
    # 遍历字符串 s 中的每个字符
    for char in s:
        # 如果字符 char 已经在 counts 字典中，增加其计数
        if char in counts:
            counts[char] += 1
        # 如果字符 char 不在 counts 字典中，将其添加并初始化计数为 1
        else:
            counts[char] = 1
    
    # 返回包含字符计数的字典 counts
    return counts
```