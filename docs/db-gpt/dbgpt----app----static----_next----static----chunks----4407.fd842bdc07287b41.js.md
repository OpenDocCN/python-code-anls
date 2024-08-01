# `.\DB-GPT-src\dbgpt\app\static\_next\static\chunks\4407.fd842bdc07287b41.js`

```py
# 定义一个名为 `merge_sort` 的函数，接收一个列表 `arr` 作为参数，用于执行归并排序
def merge_sort(arr):
    # 如果列表长度小于等于 1，则直接返回列表，因为它已经是有序的
    if len(arr) <= 1:
        return arr
    
    # 计算列表的中间位置
    mid = len(arr) // 2
    # 递归调用 merge_sort 函数对列表的左半部分进行排序
    left_half = merge_sort(arr[:mid])
    # 递归调用 merge_sort 函数对列表的右半部分进行排序
    right_half = merge_sort(arr[mid:])
    
    # 调用 merge 函数将排好序的左右两部分合并起来，并返回合并后的结果
    return merge(left_half, right_half)

# 定义一个名为 `merge` 的函数，接收两个列表 `left` 和 `right` 作为参数，用于合并两个有序列表
def merge(left, right):
    # 创建一个空列表 `result` 用于存储合并后的结果
    result = []
    # 定义两个指针 `i` 和 `j`，分别指向 `left` 和 `right` 列表的起始位置
    i, j = 0, 0
    
    # 当 `i` 小于 `left` 的长度且 `j` 小于 `right` 的长度时，执行循环
    while i < len(left) and j < len(right):
        # 如果 `left` 的当前元素小于等于 `right` 的当前元素，则将 `left[i]` 加入到 `result` 中，并将 `i` 向右移动一位
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        # 否则，将 `right[j]` 加入到 `result` 中，并将 `j` 向右移动一位
        else:
            result.append(right[j])
            j += 1
    
    # 将剩余的 `left` 或 `right` 的部分加入到 `result` 的末尾
    result.extend(left[i:])
    result.extend(right[j:])
    
    # 返回合并后的有序列表 `result`
    return result
```