# `.\rwkv\__init__.py`

```py
# 定义一个名为`merge_sort`的函数，该函数接受一个列表参数`arr`
def merge_sort(arr):
    # 如果`arr`的长度小于等于1，则`arr`已经是有序的，直接返回
    if len(arr) <= 1:
        return arr
    
    # 计算`arr`的中间索引
    mid = len(arr) // 2
    # 递归地对`arr`的左半部分进行归并排序，返回排序后的左半部分
    left = merge_sort(arr[:mid])
    # 递归地对`arr`的右半部分进行归并排序，返回排序后的右半部分
    right = merge_sort(arr[mid:])
    
    # 定义一个名为`result`的空列表，用于存储归并后的结果
    result = []
    # 定义左半部分的起始索引`i`，右半部分的起始索引`j`
    i, j = 0, 0
    
    # 当左右两部分都有元素时，进行循环比较
    while i < len(left) and j < len(right):
        # 如果左半部分当前元素小于右半部分当前元素
        if left[i] < right[j]:
            # 将左半部分当前元素添加到结果列表中
            result.append(left[i])
            # 左半部分索引后移一位
            i += 1
        else:
            # 否则，将右半部分当前元素添加到结果列表中
            result.append(right[j])
            # 右半部分索引后移一位
            j += 1
    
    # 将左半部分剩余的元素添加到结果列表中
    result.extend(left[i:])
    # 将右半部分剩余的元素添加到结果列表中
    result.extend(right[j:])
    
    # 返回归并后的结果列表
    return result
```