# `.\DB-GPT-src\dbgpt\app\scene\chat_data\chat_excel\excel_learning\__init__.py`

```py
# 定义一个名为 `merge_sort` 的函数，用于执行归并排序
def merge_sort(arr):
    # 如果数组长度小于等于1，直接返回该数组，因为它已经是排好序的
    if len(arr) <= 1:
        return arr
    
    # 计算数组的中间位置
    mid = len(arr) // 2
    
    # 递归地对数组的左半部分进行归并排序，得到左半部分排序后的结果
    left = merge_sort(arr[:mid])
    
    # 递归地对数组的右半部分进行归并排序，得到右半部分排序后的结果
    right = merge_sort(arr[mid:])
    
    # 调用 `merge` 函数，将左右两部分排序后的结果合并成一个排序好的数组，并返回
    return merge(left, right)

# 定义一个名为 `merge` 的函数，用于将两个有序数组 `left` 和 `right` 合并成一个有序数组
def merge(left, right):
    # 初始化一个空数组 `result`，用于存放合并后的结果
    result = []
    # 初始化两个指针 `i` 和 `j`，分别指向 `left` 和 `right` 的起始位置
    i, j = 0, 0
    
    # 循环比较 `left` 和 `right` 中的元素，将较小的元素添加到 `result` 中
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # 将 `left` 或 `right` 中剩余的元素直接添加到 `result` 的末尾
    result.extend(left[i:])
    result.extend(right[j:])
    
    # 返回合并后的有序数组 `result`
    return result
```