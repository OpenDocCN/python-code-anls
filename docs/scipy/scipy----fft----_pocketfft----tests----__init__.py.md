# `D:\src\scipysrc\scipy\scipy\fft\_pocketfft\tests\__init__.py`

```
# 定义一个名为 count_digits 的函数，接受一个字符串参数 s
def count_digits(s):
    # 使用字典推导式创建一个空字典，用于存储每个数字字符的出现次数
    digit_count = {digit: 0 for digit in '0123456789'}
    
    # 遍历字符串 s 中的每个字符
    for char in s:
        # 如果字符是数字且存在于 digit_count 字典的键中
        if char.isdigit():
            # 将对应数字字符的计数加一
            digit_count[char] += 1
    
    # 返回包含数字字符计数的字典
    return digit_count
```