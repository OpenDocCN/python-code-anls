# `D:\src\scipysrc\matplotlib\lib\matplotlib\backends\_backend_agg.pyi`

```
# 定义一个名为 merge_sort 的函数，用于归并排序一个列表
def merge_sort(arr):
    # 若列表长度小于等于1，则无需排序，直接返回列表
    if len(arr) <= 1:
        return arr
    
    # 计算列表的中间位置
    mid = len(arr) // 2
    
    # 递归调用 merge_sort 函数对左半部分进行排序
    left = merge_sort(arr[:mid])
    
    # 递归调用 merge_sort 函数对右半部分进行排序
    right = merge_sort(arr[mid:])
    
    # 调用 merge 函数将排序后的左右两部分合并，并返回合并后的结果
    return merge(left, right)

# 定义一个名为 merge 的函数，用于合并两个已排序的列表
def merge(left, right):
    # 创建一个空列表用于存放合并后的结果
    result = []
    # 定义两个指针 i 和 j，分别指向左右两个列表的起始位置
    i = 0
    j = 0
    
    # 循环比较左右两个列表的元素，直到其中一个列表被遍历完
    while i < len(left) and j < len(right):
        # 如果左列表当前元素小于右列表当前元素，则将左列表当前元素添加到结果中，并将左列表指针后移一位
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        # 否则将右列表当前元素添加到结果中，并将右列表指针后移一位
        else:
            result.append(right[j])
            j += 1
    
    # 将剩余的左列表元素（如果有）依次添加到结果中
    result.extend(left[i:])
    # 将剩余的右列表元素（如果有）依次添加到结果中
    result.extend(right[j:])
    
    # 返回最终合并排序后的结果
    return result
```