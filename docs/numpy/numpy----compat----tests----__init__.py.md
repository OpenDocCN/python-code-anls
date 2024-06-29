# `.\numpy\numpy\compat\tests\__init__.py`

```
# 定义一个名为find_duplicate的函数，接收一个参数nums，该参数是一个整数列表
def find_duplicate(nums):
    # 创建一个空集合dup_set，用于存储出现过的数字
    dup_set = set()
    
    # 遍历nums列表中的每个元素num
    for num in nums:
        # 如果num已经在dup_set中存在，表示num是重复出现的数字
        if num in dup_set:
            # 返回找到的重复数字num
            return num
        # 将num加入dup_set中，记录已经出现的数字
        dup_set.add(num)
    
    # 如果没有找到重复数字，返回None
    return None
```