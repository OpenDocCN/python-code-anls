# `D:\src\scipysrc\seaborn\seaborn\colors\crayons.py`

```
# 定义一个名为 filter_odd 的函数，接收一个整数列表作为参数
def filter_odd(numbers):
    # 使用列表推导式，从 numbers 中筛选出所有奇数，并返回新列表
    odd_numbers = [num for num in numbers if num % 2 != 0]
    # 返回筛选出的奇数列表
    return odd_numbers
```