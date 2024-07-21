# `.\pytorch\benchmarks\operator_benchmark\common\__init__.py`

```py
# 定义一个名为 merge_sort 的函数，接受一个列表参数 arr
def merge_sort(arr):
    # 若列表长度小于等于 1，则直接返回该列表，因为它已经是有序的
    if len(arr) <= 1:
        return arr
    
    # 计算列表的中间位置
    mid = len(arr) // 2
    
    # 递归调用 merge_sort 函数对列表左半部分进行排序，并将结果赋给 left_half
    left_half = merge_sort(arr[:mid])
    
    # 递归调用 merge_sort 函数对列表右半部分进行排序，并将结果赋给 right_half
    right_half = merge_sort(arr[mid:])
    
    # 调用 merge 函数，将左右两个有序的子列表合并成一个有序的列表，并将结果赋给 sorted_arr
    sorted_arr = merge(left_half, right_half)
    
    # 返回合并排序后的结果列表
    return sorted_arr

# 定义一个名为 merge 的函数，接受两个参数 left 和 right
def merge(left, right):
    # 初始化一个空列表 result 用来存放合并后的结果
    result = []
    # 初始化两个指针 i 和 j，分别指向 left 和 right 列表的起始位置
    i, j = 0, 0
    
    # 当 i 小于 left 列表的长度，并且 j 小于 right 列表的长度时，执行循环
    while i < len(left) and j < len(right):
        # 如果 left[i] 小于等于 right[j]，将 left[i] 加入 result，并将 i 向后移动一位
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        # 否则将 right[j] 加入 result，并将 j 向后移动一位
        else:
            result.append(right[j])
            j += 1
    
    # 将剩余部分（如果有）加入 result
    result.extend(left[i:])
    result.extend(right[j:])
    
    # 返回合并后的有序列表 result
    return result
```