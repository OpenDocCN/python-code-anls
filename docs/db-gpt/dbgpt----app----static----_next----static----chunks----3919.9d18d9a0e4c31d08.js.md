# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\3919.9d18d9a0e4c31d08.js`

```py
# 定义一个名为 count_sort 的函数，用于执行计数排序
def count_sort(arr):
    # 确定数组 arr 的最大值
    max_val = max(arr)
    # 创建一个长度为最大值加一的列表 counts，用于统计每个元素出现的次数
    counts = [0] * (max_val + 1)
    
    # 遍历数组 arr，统计每个元素的出现次数
    for num in arr:
        counts[num] += 1
    
    # 创建一个空列表 result，用于存储排序后的结果
    result = []
    # 遍历 counts 列表，按顺序将元素加入 result 中，实现排序
    for i in range(len(counts)):
        result.extend([i] * counts[i])
    
    # 返回排序后的结果列表 result
    return result
```