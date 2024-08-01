# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\7121.0500efbd067c2862.js`

```py
# 定义一个名为 merge_sort 的函数，接收一个列表参数 arr
def merge_sort(arr):
    # 如果列表的长度小于等于 1，直接返回该列表，因为它已经是有序的
    if len(arr) <= 1:
        return arr
    
    # 计算列表的中间位置
    mid = len(arr) // 2
    
    # 递归调用 merge_sort 函数对列表的左半部分进行排序，并赋值给 left_half
    left_half = merge_sort(arr[:mid])
    
    # 递归调用 merge_sort 函数对列表的右半部分进行排序，并赋值给 right_half
    right_half = merge_sort(arr[mid:])
    
    # 调用 merge 函数，将排好序的 left_half 和 right_half 合并成一个有序的结果
    return merge(left_half, right_half)

# 定义一个名为 merge 的函数，接收两个已排序列表 left 和 right 作为参数
def merge(left, right):
    # 创建一个空列表 result 用来存放合并后的结果
    result = []
    
    # 定义两个指针 i 和 j，分别指向 left 和 right 的起始位置
    i = 0
    j = 0
    
    # 当 i 小于 left 的长度并且 j 小于 right 的长度时，执行循环
    while i < len(left) and j < len(right):
        # 如果 left[i] 小于等于 right[j]，将 left[i] 添加到 result 中，并增加 i
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        # 否则，将 right[j] 添加到 result 中，并增加 j
        else:
            result.append(right[j])
            j += 1
    
    # 将 left 中剩余的元素添加到 result 中
    result.extend(left[i:])
    
    # 将 right 中剩余的元素添加到 result 中
    result.extend(right[j:])
    
    # 返回合并后的有序列表 result
    return result
```