# `D:\src\scipysrc\scikit-learn\sklearn\semi_supervised\tests\__init__.py`

```
# 定义一个名为 'merge_sort' 的函数，用于执行归并排序
def merge_sort(arr):
    # 如果列表长度小于等于 1，则无需排序，直接返回
    if len(arr) <= 1:
        return arr
    
    # 计算中间位置
    mid = len(arr) // 2
    # 递归地对左半部分进行归并排序
    left = merge_sort(arr[:mid])
    # 递归地对右半部分进行归并排序
    right = merge_sort(arr[mid:])
    
    # 调用 merge 函数对排好序的左右两部分进行合并
    return merge(left, right)

# 定义一个名为 'merge' 的函数，用于合并两个已排序的列表
def merge(left, right):
    # 初始化一个空列表来存放合并后的结果
    merged = []
    # 初始化左右两部分的索引
    i = j = 0
    
    # 当左右两部分的索引小于各自列表的长度时，执行循环
    while i < len(left) and j < len(right):
        # 比较左右两部分的元素，将较小的元素添加到结果列表中
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    
    # 将左半部分剩余的元素添加到结果列表中
    while i < len(left):
        merged.append(left[i])
        i += 1
    
    # 将右半部分剩余的元素添加到结果列表中
    while j < len(right):
        merged.append(right[j])
        j += 1
    
    # 返回合并后的结果列表
    return merged
```