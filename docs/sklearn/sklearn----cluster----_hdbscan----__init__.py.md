# `D:\src\scipysrc\scikit-learn\sklearn\cluster\_hdbscan\__init__.py`

```
# 定义一个名为 `merge_sort` 的函数，用于实现归并排序算法
def merge_sort(arr):
    # 如果数组的长度小于等于1，直接返回数组本身，不需要排序
    if len(arr) <= 1:
        return arr
    
    # 计算数组的中间位置
    mid = len(arr) // 2
    
    # 递归调用归并排序，对数组的左半部分进行排序
    left_sorted = merge_sort(arr[:mid])
    
    # 递归调用归并排序，对数组的右半部分进行排序
    right_sorted = merge_sort(arr[mid:])
    
    # 调用 `merge` 函数，将左右两个已排序的子数组进行合并
    return merge(left_sorted, right_sorted)


# 定义一个名为 `merge` 的函数，用于将两个已排序的数组合并成一个排序的数组
def merge(left, right):
    # 创建一个空列表 `result`，用于存储合并后的排序结果
    result = []
    # 创建两个指针 `i` 和 `j`，分别指向左右两个数组的起始位置
    i = j = 0
    
    # 循环比较两个数组中的元素，并将较小的元素添加到 `result` 列表中
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # 将剩余的元素追加到 `result` 列表中（其中一个数组可能有剩余元素）
    result.extend(left[i:])
    result.extend(right[j:])
    
    # 返回合并后的排序数组 `result`
    return result
```