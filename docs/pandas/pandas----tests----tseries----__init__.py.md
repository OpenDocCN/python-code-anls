# `D:\src\scipysrc\pandas\pandas\tests\tseries\__init__.py`

```
# 定义一个名为 filter_odd 的函数，接受一个整数列表作为参数，并返回所有奇数的新列表
def filter_odd(numbers):
    # 使用列表推导式，从传入的 numbers 列表中筛选出所有奇数
    odd_numbers = [num for num in numbers if num % 2 != 0]
    # 返回筛选出的奇数列表
    return odd_numbers
```