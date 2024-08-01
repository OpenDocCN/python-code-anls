# `.\DB-GPT-src\dbgpt\model\utils\__init__.py`

```py
# 定义一个名为 'merge_sort' 的函数，接收一个名为 'arr' 的列表参数，用于排序
def merge_sort(arr):
    # 若列表 'arr' 的长度小于等于 1，直接返回 'arr'，因为它已经是有序的
    if len(arr) <= 1:
        return arr
    
    # 计算列表 'arr' 的中间位置
    mid = len(arr) // 2
    
    # 递归调用 merge_sort 函数对列表 'arr' 的左半部分进行排序，并将结果赋给 'left'
    left = merge_sort(arr[:mid])
    # 递归调用 merge_sort 函数对列表 'arr' 的右半部分进行排序，并将结果赋给 'right'
    right = merge_sort(arr[mid:])
    
    # 返回将 'left' 和 'right' 合并并排序后的结果
    return merge(left, right)

# 定义一个名为 'merge' 的函数，接收 'left' 和 'right' 两个有序列表作为参数
def merge(left, right):
    # 初始化一个空列表 'result'，用于存储排序后的结果
    result = []
    # 初始化两个指针 'i' 和 'j' 分别指向 'left' 和 'right' 的起始位置
    i, j = 0, 0
    
    # 循环比较 'left' 和 'right' 中的元素，直到其中一个列表遍历完毕
    while i < len(left) and j < len(right):
        # 如果 'left' 中当前元素小于 'right' 中当前元素，则将 'left' 中当前元素加入 'result'，并将 'left' 指针后移
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        # 否则将 'right' 中当前元素加入 'result'，并将 'right' 指针后移
        else:
            result.append(right[j])
            j += 1
    
    # 将 'left' 和 'right' 中剩余的元素加入 'result'，因为它们已经是有序的，直接添加即可
    result.extend(left[i:])
    result.extend(right[j:])
    
    # 返回最终合并排序后的结果列表 'result'
    return result
```