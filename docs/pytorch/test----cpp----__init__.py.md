# `.\pytorch\test\cpp\__init__.py`

```
# 定义一个名为 find_duplicates 的函数，接收一个参数 nums
def find_duplicates(nums):
    # 创建一个空集合 seen，用于存储已经遇到的数字
    seen = set()
    # 创建一个空列表 duplicates，用于存储找到的重复数字
    duplicates = []
    
    # 遍历参数 nums 中的每个数字
    for num in nums:
        # 如果当前数字 num 已经在集合 seen 中，说明是重复出现的数字
        if num in seen:
            # 将重复的数字 num 添加到列表 duplicates 中
            duplicates.append(num)
        else:
            # 否则将当前数字 num 加入集合 seen 中，表示已经遇到过
            seen.add(num)
    
    # 返回找到的重复数字列表 duplicates
    return duplicates
```