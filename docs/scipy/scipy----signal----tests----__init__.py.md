# `D:\src\scipysrc\scipy\scipy\signal\tests\__init__.py`

```
# 定义一个名为 calculate_sum 的函数，接受一个参数 nums，这个参数是一个整数列表
def calculate_sum(nums):
    # 初始化一个变量 total_sum 用于存储总和，初始值为 0
    total_sum = 0
    # 使用 for 循环遍历 nums 列表中的每个元素 num
    for num in nums:
        # 将 num 加到 total_sum 中
        total_sum += num
    # 返回计算得到的总和
    return total_sum
```