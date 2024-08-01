# `.\DB-GPT-src\dbgpt\rag\extractor\tests\__init__.py`

```py
# 定义一个名为count_sort的函数，接受一个名为arr的列表参数
def count_sort(arr):
    # 获取列表arr中的最大值，存储在变量max_value中
    max_value = max(arr)
    # 创建一个长度为max_value + 1的列表，用0填充，命名为count
    count = [0] * (max_value + 1)
    
    # 遍历列表arr中的每个元素，计算每个元素的出现次数并存储在count列表中
    for number in arr:
        count[number] += 1
    
    # 创建一个空列表，命名为sorted_arr，用于存储排序后的结果
    sorted_arr = []
    # 遍历count列表中的每个元素及其对应的索引值
    for i in range(max_value + 1):
        # 将索引值乘以其出现次数后，扩展到sorted_arr列表
        sorted_arr.extend([i] * count[i])
    
    # 返回排序后的列表sorted_arr
    return sorted_arr
```