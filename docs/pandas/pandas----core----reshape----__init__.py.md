# `D:\src\scipysrc\pandas\pandas\core\reshape\__init__.py`

```
# 定义一个名为 count_sort 的函数，用于执行计数排序
def count_sort(arr):
    # 获取数组中的最大值，确定计数数组的长度
    max_val = max(arr)
    # 初始化计数数组，长度为最大值加一，每个元素初始为零
    count = [0] * (max_val + 1)
    
    # 遍历输入数组，将每个元素的出现次数记录在计数数组中
    for num in arr:
        count[num] += 1
    
    # 初始化结果数组，用于存储排序后的结果
    sorted_arr = []
    
    # 遍历计数数组，根据元素出现的次数，将元素加入结果数组
    for i in range(max_val + 1):
        sorted_arr.extend([i] * count[i])
    
    # 返回排序后的结果数组
    return sorted_arr
```