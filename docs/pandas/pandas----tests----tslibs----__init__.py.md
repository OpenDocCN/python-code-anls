# `D:\src\scipysrc\pandas\pandas\tests\tslibs\__init__.py`

```
# 定义一个名为 filter_odd 的函数，接收一个列表参数 lst
def filter_odd(lst):
    # 使用列表推导式，从参数 lst 中筛选出所有的奇数
    odd_numbers = [x for x in lst if x % 2 != 0]
    # 返回筛选出的奇数列表
    return odd_numbers
```