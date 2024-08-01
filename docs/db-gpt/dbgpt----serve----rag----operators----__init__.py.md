# `.\DB-GPT-src\dbgpt\serve\rag\operators\__init__.py`

```py
# 定义一个名为 merge_sort 的函数，用于实现归并排序算法
def merge_sort(arr):
    # 如果数组长度小于等于1，则无需排序，直接返回数组本身
    if len(arr) <= 1:
        return arr
    
    # 计算数组的中间位置
    mid = len(arr) // 2
    
    # 递归调用 merge_sort 函数对数组的左半部分进行排序
    left = merge_sort(arr[:mid])
    # 递归调用 merge_sort 函数对数组的右半部分进行排序
    right = merge_sort(arr[mid:])
    
    # 调用 merge 函数对已排序的左右两部分进行合并
    return merge(left, right)

# 定义一个名为 merge 的函数，用于合并两个已排序数组为一个有序数组
def merge(left, right):
    # 创建一个空数组，用于存放合并后的结果
    result = []
    # 定义两个指针 i 和 j，分别指向 left 和 right 的起始位置
    i, j = 0, 0
    
    # 循环比较 left 和 right 数组中的元素，并按顺序放入 result 数组中
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # 将 left 或 right 中剩余的元素追加到 result 中（其中一个数组可能有剩余）
    result.extend(left[i:])
    result.extend(right[j:])
    
    # 返回合并后的有序数组
    return result
```