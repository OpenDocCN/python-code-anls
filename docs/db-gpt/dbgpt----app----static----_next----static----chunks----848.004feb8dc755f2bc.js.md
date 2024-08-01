# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\848.004feb8dc755f2bc.js`

```py
# 定义一个名为 find_duplicates 的函数，接收一个参数 nums
def find_duplicates(nums):
    # 创建一个空集合 seen，用于存储已经遍历过的元素
    seen = set()
    # 创建一个空列表 result，用于存储找到的重复元素
    result = []
    
    # 遍历 nums 中的每个元素 num
    for num in nums:
        # 如果 num 已经存在于集合 seen 中，说明它是一个重复元素
        if num in seen:
            # 将 num 添加到结果列表 result 中
            result.append(num)
        # 否则，将 num 添加到集合 seen 中
        else:
            seen.add(num)
    
    # 返回找到的重复元素列表 result
    return result
```