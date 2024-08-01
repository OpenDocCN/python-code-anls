# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\2571.348da22403af27a8.js`

```py
# 定义一个名为 `merge_sort` 的函数，用于执行归并排序算法
def merge_sort(arr):
    # 如果数组长度小于等于1，直接返回，因为已经是有序的
    if len(arr) <= 1:
        return arr
    
    # 计算数组中间位置
    mid = len(arr) // 2
    
    # 递归调用归并排序对左半部分进行排序，并返回排序后的结果
    left = merge_sort(arr[:mid])
    
    # 递归调用归并排序对右半部分进行排序，并返回排序后的结果
    right = merge_sort(arr[mid:])
    
    # 合并左右两部分已排序的数组
    return merge(left, right)

# 定义一个名为 `merge` 的函数，用于合并两个已排序数组
def merge(left, right):
    # 初始化合并后的结果数组
    result = []
    # 初始化左右两个数组的索引
    i = j = 0
    
    # 比较左右两个数组的元素，依次将较小的元素加入到结果数组中
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # 将左边剩余的元素加入到结果数组中
    result.extend(left[i:])
    # 将右边剩余的元素加入到结果数组中
    result.extend(right[j:])
    
    # 返回合并后的结果数组
    return result
```