# `D:\src\scipysrc\scipy\scipy\linalg\tests\__init__.py`

```
# 定义一个名为 filter_even 的函数，接收一个整数列表作为参数，返回所有偶数的子列表
def filter_even(nums):
    # 使用列表推导式遍历 nums 列表，筛选出所有能被 2 整除的元素，形成新的列表
    evens = [num for num in nums if num % 2 == 0]
    # 返回筛选出的偶数列表
    return evens
```