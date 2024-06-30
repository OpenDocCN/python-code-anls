# `D:\src\scipysrc\scikit-learn\sklearn\svm\tests\__init__.py`

```
# 定义一个名为 count_digits 的函数，用于统计一个整数中各个数字出现的次数
def count_digits(num):
    # 将整数转换为字符串，方便逐个字符（即数字字符）遍历
    num_str = str(num)
    # 创建一个空字典，用于存储每个数字出现的次数
    digit_count = {}
    
    # 遍历整数的每个数字字符
    for digit in num_str:
        # 如果该数字字符在字典中已经存在，则将其计数加一
        if digit in digit_count:
            digit_count[digit] += 1
        # 否则，在字典中新增该数字字符并将计数置为1
        else:
            digit_count[digit] = 1
    
    # 返回统计结果的字典
    return digit_count
```