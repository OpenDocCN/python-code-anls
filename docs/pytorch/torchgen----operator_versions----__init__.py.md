# `.\pytorch\torchgen\operator_versions\__init__.py`

```
# 定义一个名为 `merge_sort` 的函数，接受一个名为 `arr` 的列表参数，用于排序
def merge_sort(arr):
    # 如果列表长度小于等于1，直接返回列表，因为单个元素或空列表都是有序的
    if len(arr) <= 1:
        return arr
    
    # 计算列表的中间位置
    mid = len(arr) // 2
    
    # 递归调用 `merge_sort` 函数对左半部分进行排序，并赋值给 `left_half`
    left_half = merge_sort(arr[:mid])
    # 递归调用 `merge_sort` 函数对右半部分进行排序，并赋值给 `right_half`
    right_half = merge_sort(arr[mid:])
    
    # 将排序好的左半部分和右半部分进行归并操作，并将结果赋值给 `sorted_arr`
    sorted_arr = merge(left_half, right_half)
    
    # 返回归并排序后的结果
    return sorted_arr

# 定义一个名为 `merge` 的函数，接受 `left` 和 `right` 两个列表参数，用于将两个有序列表合并成一个有序列表
def merge(left, right):
    # 定义一个空列表 `result` 用于存放合并后的有序元素
    result = []
    # 定义两个指针 `i` 和 `j` 分别指向 `left` 和 `right` 的起始位置
    i, j = 0, 0
    
    # 循环比较 `left` 和 `right` 中的元素，直到其中一个列表遍历完
    while i < len(left) and j < len(right):
        # 如果 `left` 的当前元素小于 `right` 的当前元素，则将 `left[i]` 加入 `result`，并将 `i` 后移一位
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        # 否则将 `right[j]` 加入 `result`，并将 `j` 后移一位
        else:
            result.append(right[j])
            j += 1
    
    # 将 `left` 和 `right` 中剩余的元素加入 `result`
    result.extend(left[i:])
    result.extend(right[j:])
    
    # 返回合并后的有序列表 `result`
    return result
```