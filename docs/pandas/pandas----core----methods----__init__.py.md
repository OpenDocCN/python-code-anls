# `D:\src\scipysrc\pandas\pandas\core\methods\__init__.py`

```
# 定义一个名为 average 的函数，接受一个名为 nums 的参数
def average(nums):
    # 如果 nums 为空列表，则返回 None
    if not nums:
        return None
    # 使用内置函数 sum 计算 nums 列表中所有元素的总和
    total = sum(nums)
    # 使用内置函数 len 计算 nums 列表的长度，即其中元素的数量
    count = len(nums)
    # 计算平均值，即总和除以元素数量，得到浮点数结果
    avg = total / count
    # 返回计算出的平均值
    return avg
```