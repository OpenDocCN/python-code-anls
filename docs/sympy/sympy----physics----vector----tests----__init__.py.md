# `D:\src\scipysrc\sympy\sympy\physics\vector\tests\__init__.py`

```
# 定义一个名为 find_max 的函数，接受一个名为 nums 的参数
def find_max(nums):
    # 如果 nums 为空列表，则返回 None
    if not nums:
        return None
    # 假设最大值为列表中的第一个元素
    max_num = nums[0]
    # 遍历列表中的每个元素
    for num in nums:
        # 如果当前元素比已知的最大值大，则更新最大值
        if num > max_num:
            max_num = num
    # 返回最大值
    return max_num
```